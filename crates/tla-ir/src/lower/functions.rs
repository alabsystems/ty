// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Function operation lowering: FuncApply, Domain, FuncExcept, FuncDefBegin,
//! LoopNext (FuncDef).

use crate::TrustIrError;
use tla_jit_abi::{CompoundLayout, JitRuntimeErrorKind, SetBitmaskElement, VarLayout};
use trust_ir::inst::*;
use trust_ir::ty::Ty;
use trust_ir::value::ValueId;
use trust_ir::Constant;
use trust_ir::InstrNode;

use super::{Ctx, LoopNextKind, LoopNextState, QuantifierLoopState};

#[derive(Clone, Copy)]
enum DomainKeyRef {
    Raw(i64),
    Exact(SetBitmaskElement),
}

impl<'cp> Ctx<'cp> {
    // =====================================================================
    // Function operations
    // =====================================================================
    //
    // TLA+ functions are total mappings from domain keys to values.
    // Aggregate layout: slot[0] = pair_count, then interleaved key-value pairs:
    //   slot[1] = key1, slot[2] = val1, slot[3] = key2, slot[4] = val2, ...
    //
    // For a function with N pairs, total slots = 1 + 2*N.
    // Key at index i (0-based): slot[1 + 2*i]
    // Value at index i (0-based): slot[2 + 2*i]

    pub(super) fn copy_compact_slots_from_dynamic_base(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        source_base_slot: ValueId,
        slot_count: u32,
    ) -> ValueId {
        let result_ptr = self.alloc_aggregate(block_idx, slot_count);
        for slot in 0..slot_count {
            let slot_offset = self.emit_i64_const(block_idx, i64::from(slot));
            let source_slot = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: source_base_slot,
                    rhs: slot_offset,
                },
            );
            let value = self.load_at_dynamic_offset(block_idx, source_ptr, source_slot);
            self.store_at_offset(block_idx, result_ptr, slot, value);
        }
        result_ptr
    }

    pub(super) fn store_compact_aggregate_result(
        &mut self,
        block_idx: usize,
        rd: u8,
        source_ptr: ValueId,
        shape: super::AggregateShape,
    ) -> Result<(), TrustIrError> {
        let explicit_domain = shape.function_explicit_domain();
        self.store_reg_ptr(block_idx, rd, source_ptr)?;
        self.aggregate_pointer_regs
            .insert(rd, super::AggregatePointerKind::Compact);
        self.compact_state_slots.insert(
            rd,
            super::CompactStateSlot::pointer_backed_in_block(source_ptr, 0, block_idx),
        );
        if let Some(domain) = explicit_domain {
            self.compact_function_domains.insert(rd, domain);
        } else {
            self.compact_function_domains.remove(&rd);
        }
        if let Some(len) = shape.tracked_len() {
            self.const_set_sizes.insert(rd, len);
        } else {
            self.const_set_sizes.remove(&rd);
        }
        self.aggregate_shapes.insert(rd, shape);
        self.const_scalar_values.remove(&rd);
        Ok(())
    }

    pub(super) fn store_single_slot_compact_result(
        &mut self,
        block_idx: usize,
        rd: u8,
        value: ValueId,
        source_slot: super::CompactStateSlot,
        shape: super::AggregateShape,
    ) -> Result<(), TrustIrError> {
        let explicit_domain = shape.function_explicit_domain();
        self.store_reg_value(block_idx, rd, value)?;
        let reg_slot = self.reg_ptr(rd)?;
        let result_slot = if source_slot.is_raw_compact_slot()
            && (source_slot.source_ptr == self.state_in_ptr
                || self.state_out_ptr == Some(source_slot.source_ptr))
        {
            source_slot
        } else {
            super::CompactStateSlot::raw(reg_slot, 0)
        };
        if let Some(len) = shape.tracked_len() {
            self.const_set_sizes.insert(rd, len);
        } else {
            self.const_set_sizes.remove(&rd);
        }
        self.compact_state_slots.insert(rd, result_slot);
        if let Some(domain) = explicit_domain {
            self.compact_function_domains.insert(rd, domain);
        } else {
            self.compact_function_domains.remove(&rd);
        }
        self.aggregate_shapes.insert(rd, shape);
        self.const_scalar_values.remove(&rd);
        Ok(())
    }

    fn store_compact_function_apply_result_at_index(
        &mut self,
        block_idx: usize,
        rd: u8,
        source_slot: super::CompactStateSlot,
        value_index: u32,
        value_stride: u32,
        value_shape: Option<super::AggregateShape>,
    ) -> Result<(), TrustIrError> {
        let value_offset = value_index
            .checked_mul(value_stride)
            .and_then(|offset| source_slot.offset.checked_add(offset))
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "FuncApply: compact function value slot overflows".to_owned(),
                )
            })?;
        if value_stride == 1 {
            let result_val = self.load_at_offset(block_idx, source_slot.source_ptr, value_offset);
            if let Some(shape) = value_shape {
                self.store_single_slot_compact_result(
                    block_idx,
                    rd,
                    result_val,
                    super::CompactStateSlot::raw(source_slot.source_ptr, value_offset),
                    shape,
                )?;
            } else {
                self.store_reg_value(block_idx, rd, result_val)?;
                self.aggregate_shapes.remove(&rd);
                self.const_set_sizes.remove(&rd);
                self.compact_state_slots.remove(&rd);
                self.compact_function_domains.remove(&rd);
                self.const_scalar_values.remove(&rd);
            }
        } else {
            if let Some(shape) = value_shape {
                let result_ptr = self.emit_state_slot_ptr_at_slot(
                    block_idx,
                    source_slot.source_ptr,
                    value_offset,
                );
                self.store_compact_aggregate_result(block_idx, rd, result_ptr, shape)?;
            } else {
                let source_base = self.emit_i64_const(block_idx, i64::from(value_offset));
                let result_ptr = self.copy_compact_slots_from_dynamic_base(
                    block_idx,
                    source_slot.source_ptr,
                    source_base,
                    value_stride,
                );
                self.store_reg_ptr(block_idx, rd, result_ptr)?;
                self.compact_state_slots.insert(
                    rd,
                    super::CompactStateSlot::pointer_backed_in_block(result_ptr, 0, block_idx),
                );
                self.compact_function_domains.remove(&rd);
                self.aggregate_shapes.remove(&rd);
                self.const_set_sizes.remove(&rd);
                self.const_scalar_values.remove(&rd);
            }
        }
        Ok(())
    }

    // `pub(super)`: WP-27 (item B1) reuses this raw-member-space projection in
    // `set_ops`'s `SetIn` element decode, so the two sides of the union-index
    // contract (decode here, sort proof there) cannot drift apart.
    pub(super) fn domain_key_raw_value(key: &SetBitmaskElement) -> i64 {
        match key {
            SetBitmaskElement::Int(value) => *value,
            SetBitmaskElement::Bool(value) => i64::from(*value),
            SetBitmaskElement::String(name) | SetBitmaskElement::ModelValue(name) => {
                i64::from(name.0)
            }
        }
    }

    /// WP-32: whether a `ScalarIntDomain`'s declared finite INTEGER domain is
    /// entirely contained in the closed interval `[lo, hi]`.
    ///
    /// Both obligations a tagged-scalar-union int-arm encode needs are read off
    /// the universe descriptor: `IntRange` / `ExplicitInt` are integer-sorted by
    /// construction (so the raw i64 lane cannot alias an interned member's
    /// `NameId`), and their member list is exact, so containment is decidable at
    /// compile time. An `Exact` (possibly mixed-sort) or `Unknown` universe
    /// proves neither and returns `false` — the caller then fails closed.
    ///
    /// An EMPTY declared domain returns `false` rather than vacuously `true`: a
    /// zero-member domain describes a register that can hold no value at all, so
    /// admitting it would be claiming a proof about a shape the rest of the
    /// pipeline never produces.
    pub(super) fn int_domain_members_within(
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        lo: i64,
        hi: i64,
    ) -> bool {
        if universe_len == 0 || lo > hi {
            return false;
        }
        match universe {
            super::SetBitmaskUniverse::IntRange { lo: domain_lo } => {
                let Some(span) = i64::from(universe_len).checked_sub(1) else {
                    return false;
                };
                let Some(domain_hi) = domain_lo.checked_add(span) else {
                    return false;
                };
                *domain_lo >= lo && domain_hi <= hi
            }
            super::SetBitmaskUniverse::ExplicitInt(values) => {
                values.len() == usize::try_from(universe_len).unwrap_or(usize::MAX)
                    && values.iter().all(|value| *value >= lo && *value <= hi)
            }
            super::SetBitmaskUniverse::Exact(_) | super::SetBitmaskUniverse::Unknown => false,
        }
    }

    fn domain_key_scalar_shape(key: &SetBitmaskElement) -> super::ScalarShape {
        match key {
            SetBitmaskElement::Int(_) => super::ScalarShape::Int,
            SetBitmaskElement::Bool(_) => super::ScalarShape::Bool,
            SetBitmaskElement::String(_) => super::ScalarShape::String,
            SetBitmaskElement::ModelValue(_) => super::ScalarShape::ModelValue,
        }
    }

    fn exact_domain_all_one_shape(keys: &[SetBitmaskElement]) -> bool {
        let Some(first) = keys.first().map(Self::domain_key_scalar_shape) else {
            return true;
        };
        keys.iter()
            .all(|key| Self::domain_key_scalar_shape(key) == first)
    }

    /// Whether every key in a (possibly mixed-shape) exact domain has a raw
    /// i64 encoding distinct from every other key.
    ///
    /// A `FuncApply` against an explicit-domain compact function lowers each key
    /// to its raw i64 (`domain_key_raw_value`) and compares the argument by raw
    /// equality. When the argument carries a typed scalar shape, the comparison
    /// is gated on a matching shape, so type confusion cannot occur. When it does
    /// not (e.g. the argument is a quantifier binding ranging over a mixed-type
    /// set, where the binding shape collapses to `None`), a raw-equality match is
    /// still *sound* as long as no two keys share a raw value: each raw value then
    /// identifies exactly one key, so an argument can match at most one slot and
    /// any non-key argument falls through to the not-found runtime error. This is
    /// strictly more general than [`exact_domain_all_one_shape`] (homogeneous
    /// domains trivially have distinct raw values when their TLA+ elements are
    /// distinct) and keys off the structural raw-encoding of the domain rather
    /// than its element types.
    fn exact_domain_all_distinct_raw_values(keys: &[SetBitmaskElement]) -> bool {
        let mut seen = std::collections::HashSet::with_capacity(keys.len());
        keys.iter()
            .all(|key| seen.insert(Self::domain_key_raw_value(key)))
    }

    fn complete_compact_shape_from_source_layout(
        inferred: &super::AggregateShape,
        layout_shape: &super::AggregateShape,
    ) -> Option<super::AggregateShape> {
        if inferred.compact_slot_count().is_some() {
            return Some(inferred.clone());
        }
        if Self::is_single_slot_flat_aggregate_value(inferred)
            && Self::is_single_slot_flat_aggregate_value(layout_shape)
            && Self::compatible_flat_aggregate_value(inferred, layout_shape)
        {
            return Some(inferred.clone());
        }

        match (inferred, layout_shape) {
            (
                super::AggregateShape::Record { fields: inferred },
                super::AggregateShape::Record { fields: layout },
            ) => {
                if inferred.len() != layout.len() {
                    return None;
                }
                let mut fields = Vec::with_capacity(inferred.len());
                for (name, inferred_shape) in inferred {
                    let (_, layout_field_shape) =
                        layout.iter().find(|(layout_name, _)| layout_name == name)?;
                    let layout_field_shape = layout_field_shape.as_deref()?;
                    let field_shape = match inferred_shape.as_deref() {
                        Some(inferred_shape) => Self::complete_compact_shape_from_source_layout(
                            inferred_shape,
                            layout_field_shape,
                        )?,
                        None => layout_field_shape.clone(),
                    };
                    fields.push((*name, Some(Box::new(field_shape))));
                }
                Some(super::AggregateShape::Record { fields })
            }
            (
                super::AggregateShape::Sequence {
                    extent: inferred_extent,
                    element: inferred_element,
                },
                super::AggregateShape::Sequence {
                    extent: layout_extent,
                    element: layout_element,
                },
            ) => {
                if inferred_extent.capacity() != layout_extent.capacity() {
                    return None;
                }
                let element = match (inferred_element.as_deref(), layout_element.as_deref()) {
                    (Some(inferred_element), Some(layout_element)) => {
                        Some(Box::new(Self::complete_compact_shape_from_source_layout(
                            inferred_element,
                            layout_element,
                        )?))
                    }
                    (None, Some(layout_element)) if inferred_extent.capacity() > 0 => {
                        Some(Box::new(layout_element.clone()))
                    }
                    (None, _) => None,
                    _ => return None,
                };
                Some(super::AggregateShape::Sequence {
                    extent: *inferred_extent,
                    element,
                })
            }
            (
                super::AggregateShape::Function {
                    len: inferred_len,
                    domain_lo: inferred_domain_lo,
                    domain: inferred_domain,
                    value: inferred_value,
                },
                super::AggregateShape::Function {
                    len: layout_len,
                    domain_lo: layout_domain_lo,
                    domain: layout_domain,
                    value: layout_value,
                },
            ) => {
                if inferred_len != layout_len
                    || inferred_domain_lo != layout_domain_lo
                    || (inferred_domain.is_some()
                        && layout_domain.is_some()
                        && inferred_domain != layout_domain)
                {
                    return None;
                }
                let value = match (inferred_value.as_deref(), layout_value.as_deref()) {
                    (Some(inferred_value), Some(layout_value)) => {
                        Some(Box::new(Self::complete_compact_shape_from_source_layout(
                            inferred_value,
                            layout_value,
                        )?))
                    }
                    (None, Some(layout_value)) if *inferred_len > 0 => {
                        Some(Box::new(layout_value.clone()))
                    }
                    (None, _) => None,
                    _ => return None,
                };
                Some(super::AggregateShape::Function {
                    len: *inferred_len,
                    domain_lo: *inferred_domain_lo,
                    domain: inferred_domain.clone().or_else(|| layout_domain.clone()),
                    value,
                })
            }
            _ => None,
        }
    }

    pub(super) fn compact_function_value_shape_from_source_layout(
        &self,
        source_slot: super::CompactStateSlot,
        func_len: u32,
        func_domain_lo: Option<i64>,
        func_domain: Option<&super::CompactFunctionDomain>,
        value_shape: Option<&super::AggregateShape>,
    ) -> Option<super::AggregateShape> {
        if value_shape.is_some_and(|shape| shape.compact_slot_count().is_some()) {
            return value_shape.cloned();
        }

        let layout =
            self.compound_function_layout_at_raw_state_sub_slot(source_slot, None, true)?;
        let CompoundLayout::Function {
            pair_count: Some(pair_count),
            domain_lo,
            value_layout,
            ..
        } = layout
        else {
            return None;
        };
        if u32::try_from(*pair_count).ok()? != func_len || *domain_lo != func_domain_lo {
            return None;
        }
        let layout_domain = self.compact_function_domain_from_layout(layout);
        if func_domain.is_some()
            && layout_domain.as_ref().is_some()
            && func_domain != layout_domain.as_ref()
        {
            return None;
        }
        let layout_value_shape = Self::tracked_shape_from_compound_layout(value_layout.as_ref())?;
        match value_shape {
            Some(value_shape) => {
                Self::complete_compact_shape_from_source_layout(value_shape, &layout_value_shape)
            }
            None => Some(layout_value_shape),
        }
    }

    fn emit_false_cond(&mut self, block_idx: usize) -> ValueId {
        let zero = self.emit_i64_const(block_idx, 0);
        self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: zero,
                rhs: zero,
            },
        )
    }

    fn emit_exact_domain_key_match(
        &mut self,
        block_idx: usize,
        arg_val: ValueId,
        arg_shape: Option<&super::ScalarShape>,
        allow_untyped_raw_match: bool,
        key: &SetBitmaskElement,
    ) -> Result<ValueId, TrustIrError> {
        if let Some(shape) = arg_shape {
            if *shape != Self::domain_key_scalar_shape(key) {
                return Ok(self.emit_false_cond(block_idx));
            }
        } else if !allow_untyped_raw_match {
            return Err(TrustIrError::UnsupportedOpcode(
                "FuncApply/FuncExcept: mixed compact scalar domain requires typed scalar argument"
                    .to_owned(),
            ));
        }
        let key_val = self.emit_i64_const(block_idx, Self::domain_key_raw_value(key));
        Ok(self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: arg_val,
                rhs: key_val,
            },
        ))
    }

    fn lower_explicit_domain_compact_func_apply(
        &mut self,
        block_idx: usize,
        rd: u8,
        func_reg: u8,
        arg_reg: u8,
        source_slot: super::CompactStateSlot,
        len: u32,
        value: Option<Box<super::AggregateShape>>,
        domain_keys: super::CompactFunctionDomain,
    ) -> Result<Option<usize>, TrustIrError> {
        if domain_keys.len() != usize::try_from(len).expect("u32 length must fit in usize") {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: compact function r{func_reg} domain metadata length {} does not match pair_count {len}",
                domain_keys.len()
            )));
        }
        let value_shape = self
            .compact_function_value_shape_from_source_layout(
                source_slot,
                len,
                None,
                Some(&domain_keys),
                value.as_deref(),
            )
            .or_else(|| value.as_deref().cloned());
        let value_stride = value_shape
            .as_ref()
            .and_then(super::AggregateShape::compact_slot_count)
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncApply: compact function r{func_reg} requires fixed-width value shape, got {value_shape:?}"
                ))
            })?;
        let result_function_domain = match value_shape.as_ref() {
            Some(super::AggregateShape::Function {
                domain_lo: None, ..
            }) => {
                let value_shape = value_shape
                    .as_ref()
                    .expect("matched function value shape should be present");
                // Prefer the value shape's OWN explicit domain metadata: for a
                // nested function layout (`[S -> [T -> v]]`) the tracked value
                // shape carries the exact `T` keys from the state layout
                // (`tracked_shape_from_compound_layout`). The same-sized
                // fallback below reuses the OUTER `S` keys and is only correct
                // when the two domains coincide — with `|S| == |T|` but
                // `S != T` (e.g. PaxosCommit's `aState : [RM -> [Acceptor ->
                // rec]]` under a 2-RM/2-acceptor model) it used to poison the
                // result register with RM keys, making every later apply of an
                // Acceptor key a compile-time domain miss.
                value_shape
                    .function_explicit_domain()
                    .or_else(|| {
                        self.compact_function_value_domain_from_raw_state_sub_slot(
                            source_slot,
                            value_shape,
                        )
                    })
                    .or_else(|| self.compact_function_value_domain_from_raw_state_slot(source_slot))
                    .or_else(|| {
                        Self::same_sized_explicit_function_value_domain(
                            &domain_keys,
                            Some(value_shape),
                        )
                    })
            }
            _ => None,
        };

        if let Some(arg) = self.scalar_of(arg_reg) {
            let index = match &domain_keys {
                super::CompactFunctionDomain::Raw(keys) => keys.iter().position(|key| *key == arg),
                super::CompactFunctionDomain::Exact(keys) => self
                    .const_scalar_domain_key_of(arg_reg)
                    .and_then(|arg_key| keys.iter().position(|key| *key == arg_key)),
            };
            let Some(index) = index else {
                self.emit_runtime_error_and_return(block_idx, JitRuntimeErrorKind::TypeMismatch);
                return Ok(None);
            };
            self.store_compact_function_apply_result_at_index(
                block_idx,
                rd,
                source_slot,
                u32::try_from(index).expect("domain index must fit in u32"),
                value_stride,
                value_shape,
            )?;
            if let Some(domain) = result_function_domain {
                self.compact_function_domains.insert(rd, domain);
            }
            return Ok(Some(block_idx));
        }

        if domain_keys.is_empty() {
            self.emit_runtime_error_and_return(block_idx, JitRuntimeErrorKind::TypeMismatch);
            return Ok(None);
        }

        // A `TaggedScalarUnion` argument holds a universe INDEX, not the raw
        // domain key; decode it back to its member raw value (WP-18's
        // branch-free compare-fold, which routes an out-of-universe index to
        // the not-found guard below) so the linear scan compares raw key
        // against raw domain key. A decoded union arg is reported as
        // `Scalar(Int)` so exact-domain key matching treats it as an int lane.
        // `decode_scalar_key_reg_raw_value` returns the loaded value unchanged
        // for every non-union register, so this is byte-identical there.
        let arg_val = self.load_reg(block_idx, arg_reg)?;
        let (arg_val, arg_shape) = if self.reg_is_tagged_scalar_union(arg_reg) {
            (
                self.decode_scalar_key_reg_raw_value(block_idx, arg_reg, arg_val),
                Some(super::ScalarShape::Int),
            )
        } else {
            (arg_val, self.scalar_shape_of(arg_reg))
        };
        let allow_untyped_exact_raw_match = match &domain_keys {
            super::CompactFunctionDomain::Raw(_) => true,
            // A homogeneous (single-shape) exact domain is the common case; a
            // mixed-shape domain is still safe to match by raw value when every
            // key has a distinct raw encoding, because each raw value then maps
            // to exactly one key (see `exact_domain_all_distinct_raw_values`).
            super::CompactFunctionDomain::Exact(keys) => {
                Self::exact_domain_all_one_shape(keys)
                    || Self::exact_domain_all_distinct_raw_values(keys)
            }
        };
        let not_found_blk = self.new_aux_block("compact_fapply_explicit_not_found");
        let merge_blk = self.new_aux_block("compact_fapply_explicit_merge");
        let merge_id = self.block_id_of(merge_blk);
        let mut check_blk = block_idx;
        let domain_key_refs: Vec<DomainKeyRef> = match &domain_keys {
            super::CompactFunctionDomain::Raw(keys) => keys
                .iter()
                .map(|key| DomainKeyRef::Raw(*key))
                .collect::<Vec<_>>(),
            super::CompactFunctionDomain::Exact(keys) => keys
                .iter()
                .copied()
                .map(DomainKeyRef::Exact)
                .collect::<Vec<_>>(),
        };
        for (index, key) in domain_key_refs.into_iter().enumerate() {
            let found_blk = self.new_aux_block("compact_fapply_explicit_found");
            let next_blk =
                if index + 1 == usize::try_from(len).expect("u32 length must fit in usize") {
                    not_found_blk
                } else {
                    self.new_aux_block("compact_fapply_explicit_check")
                };
            let matches_key = match key {
                DomainKeyRef::Raw(key) => {
                    let key_val = self.emit_i64_const(check_blk, key);
                    self.emit_with_result(
                        check_blk,
                        Inst::ICmp {
                            op: ICmpOp::Eq,
                            ty: Ty::I64,
                            lhs: arg_val,
                            rhs: key_val,
                        },
                    )
                }
                DomainKeyRef::Exact(key) => self.emit_exact_domain_key_match(
                    check_blk,
                    arg_val,
                    arg_shape.as_ref(),
                    allow_untyped_exact_raw_match,
                    &key,
                )?,
            };
            self.emit(
                check_blk,
                InstrNode::new(Inst::CondBr {
                    cond: matches_key,
                    then_target: self.block_id_of(found_blk),
                    then_args: vec![],
                    else_target: self.block_id_of(next_blk),
                    else_args: vec![],
                }),
            );

            self.store_compact_function_apply_result_at_index(
                found_blk,
                rd,
                source_slot,
                u32::try_from(index).expect("domain index must fit in u32"),
                value_stride,
                value_shape.clone(),
            )?;
            self.emit(
                found_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );
            check_blk = next_blk;
        }
        self.emit_runtime_error_and_return(not_found_blk, JitRuntimeErrorKind::TypeMismatch);
        if let Some(domain) = result_function_domain {
            self.compact_function_domains.insert(rd, domain);
        }

        Ok(Some(merge_blk))
    }

    fn same_sized_explicit_function_value_domain(
        source_domain_keys: &super::CompactFunctionDomain,
        value_shape: Option<&super::AggregateShape>,
    ) -> Option<super::CompactFunctionDomain> {
        match value_shape {
            Some(super::AggregateShape::Function {
                len,
                domain_lo: None,
                ..
            }) if usize::try_from(*len).ok()? == source_domain_keys.len() => {
                Some(source_domain_keys.clone())
            }
            _ => None,
        }
    }

    fn compact_function_domain_for_explicit_function(
        &self,
        opcode: &str,
        func_reg: u8,
        source_slot: super::CompactStateSlot,
    ) -> Result<super::CompactFunctionDomain, TrustIrError> {
        let expected_shape = self.aggregate_shapes.get(&func_reg);
        self.compact_function_domain_of(func_reg)
            .cloned()
            .or_else(|| expected_shape.and_then(super::AggregateShape::function_explicit_domain))
            .or_else(|| {
                self.compact_function_domain_from_raw_state_sub_slot(source_slot, expected_shape)
            })
            .or_else(|| self.compact_function_domain_from_raw_state_slot(source_slot))
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "{opcode}: compact function r{func_reg} requires explicit-domain metadata"
                ))
            })
    }

    /// Recover the ordered tuple-key table of a tuple / cross-product-keyed
    /// compact function so a `FuncApply` can be lowered natively.
    ///
    /// Returns `Some(keys)` only when the recovery is **provably** the function's
    /// real domain in the canonical flat-buffer slot order, and `None` (fail
    /// closed) otherwise. The returned `keys[i]` is the per-position typed
    /// element encoding of the `i`-th canonical domain key; slot `i` of the
    /// compact flat buffer holds the range value at `keys[i]` (see
    /// `crates/tla-check/src/state/flat_state.rs::write_tuple_keyed_array_slots`,
    /// which assigns one contiguous range slot per `domain_keys[i]`).
    ///
    /// Two recovery channels, in order:
    ///
    /// 1. **Layout-carried table** (`CompoundLayout::ExplicitTupleDomain`): the
    ///    bridge emits the exact ordered key table from the check-side layout
    ///    metadata, so recovery is deterministic — no pool dependency. The
    ///    carried order IS the compact-slot order (ABI contract; never
    ///    re-sorted here). Model-value/String tuple positions are admitted:
    ///    their raw compare encoding is the interned `NameId`, mirroring how
    ///    scalar `ExplicitScalarDomain` keys encode.
    /// 2. **Const-pool mining** (bare `CompoundLayout::Tuple` key layout, the
    ///    pre-carrier fallback): the candidate must be the *unique*
    ///    constant-pool tuple set of cardinality `len` whose per-position
    ///    scalar shapes match the layout's `Tuple` key shape (mirroring
    ///    `unique_const_pool_scalar_domain`). If zero or multiple distinct
    ///    candidates exist, recovery returns `None`.
    ///
    /// ## Soundness
    /// The check-side flat layout sorts the fully-enumerated static tuple domain
    /// by `Value::cmp` (lexicographic) and maps slot `i` -> `domain_keys[i]`
    /// (`layout_inference.rs`, the `TupleKeyedArray` branch:
    /// `entries.sort_by(|(a, _), (b, _)| a.cmp(b))`). Channel 1 carries that
    /// exact order; channel 2 reproduces it by sorting the pool set with the
    /// identical `Value::cmp`. Both channels additionally require every
    /// position to be sort-homogeneous across all keys and all key rows to be
    /// pairwise distinct in raw encoding (the H5 String/ModelValue NameId
    /// discipline), so a raw-value runtime compare maps each argument to at
    /// most one slot.
    pub(super) fn tuple_function_domain_keys_for_explicit_function(
        &self,
        func_reg: u8,
        source_slot: super::CompactStateSlot,
        len: u32,
    ) -> Result<Option<Vec<Vec<SetBitmaskElement>>>, TrustIrError> {
        let _ = func_reg;
        // The flat-buffer layout must describe a tuple-keyed function with a
        // statically-known pair count and no contiguous-int fast domain.
        let Some(CompoundLayout::Function {
            key_layout,
            pair_count: Some(pair_count),
            domain_lo: None,
            ..
        }) = self.compound_layout_for_raw_state_slot(source_slot)
        else {
            return Ok(None);
        };
        if usize::try_from(len).expect("u32 length must fit in usize") != *pair_count {
            return Ok(None);
        }

        // Channel 1: the ABI-carried explicit tuple-key table (deterministic).
        if let CompoundLayout::ExplicitTupleDomain {
            key_layout: inner_key_layout,
            keys,
        } = key_layout.as_ref()
        {
            let CompoundLayout::Tuple { element_layouts } = inner_key_layout.as_ref() else {
                return Ok(None);
            };
            return Ok(Self::validated_tuple_key_table(
                keys,
                element_layouts,
                *pair_count,
            ));
        }

        let CompoundLayout::Tuple { element_layouts } = key_layout.as_ref() else {
            return Ok(None);
        };
        if element_layouts.is_empty() {
            return Ok(None);
        }

        // Channel 2: search the constant pool for the unique fully-enumerated
        // tuple key set of this cardinality whose per-position shapes are
        // compatible with the layout. Sorting by `Value::cmp` reproduces the
        // canonical flat order.
        let Some(pool) = self.config.const_pool else {
            return Ok(None);
        };
        let mut candidate: Option<Vec<Vec<SetBitmaskElement>>> = None;
        for idx in 0..pool.value_count() {
            let idx = u16::try_from(idx).expect("constant pool index must fit in u16");
            let Some(keys) = Self::tuple_domain_keys_from_value(
                pool.get_value(idx),
                element_layouts,
                *pair_count,
            ) else {
                continue;
            };
            match &candidate {
                None => candidate = Some(keys),
                Some(existing) if *existing == keys => {}
                // A second, distinct candidate of the same cardinality/shape: the
                // domain is ambiguous from the pool alone. Fail closed.
                Some(_) => return Ok(None),
            }
        }
        Ok(candidate
            .and_then(|keys| Self::validated_tuple_key_table(&keys, element_layouts, *pair_count)))
    }

    /// Validate a candidate ordered tuple-key table against the layout's tuple
    /// element shapes and the H5 sort discipline, returning the table only when
    /// every check passes (fail closed otherwise):
    ///
    /// - exactly `pair_count` rows, each of the layout's arity (non-zero);
    /// - each position sort-compatible with its element layout (`Int`↔Int,
    ///   `Bool`↔Bool, `String`↔String/ModelValue) and sort-HOMOGENEOUS across
    ///   all rows (a position never mixes String with ModelValue, so equal raw
    ///   `NameId`s cannot alias across sorts);
    /// - all rows pairwise distinct in raw-i64 encoding, so a runtime raw
    ///   compare selects at most one slot.
    fn validated_tuple_key_table(
        keys: &[Vec<SetBitmaskElement>],
        element_layouts: &[CompoundLayout],
        pair_count: usize,
    ) -> Option<Vec<Vec<SetBitmaskElement>>> {
        let arity = element_layouts.len();
        if arity == 0 || keys.len() != pair_count || pair_count == 0 {
            return None;
        }
        if keys.iter().any(|row| row.len() != arity) {
            return None;
        }
        for (position, layout) in element_layouts.iter().enumerate() {
            let first_sort = Self::tuple_key_element_sort(&keys[0][position]);
            for row in keys {
                let element = &row[position];
                if Self::tuple_key_element_sort(element) != first_sort {
                    return None;
                }
                let compatible = matches!(
                    (layout, element),
                    (CompoundLayout::Int, SetBitmaskElement::Int(_))
                        | (CompoundLayout::Bool, SetBitmaskElement::Bool(_))
                        | (
                            CompoundLayout::String,
                            SetBitmaskElement::String(_) | SetBitmaskElement::ModelValue(_)
                        )
                );
                if !compatible {
                    return None;
                }
            }
        }
        let raw_rows: Vec<Vec<i64>> = keys
            .iter()
            .map(|row| row.iter().map(Self::tuple_key_element_raw_value).collect())
            .collect();
        for (index, row) in raw_rows.iter().enumerate() {
            if raw_rows[index + 1..].iter().any(|other| other == row) {
                return None;
            }
        }
        Some(keys.to_vec())
    }

    /// Sort tag of one tuple-key element (H5: `String` and `ModelValue` intern
    /// to the same `NameId` space, so sorts must be compared before raws).
    fn tuple_key_element_sort(element: &SetBitmaskElement) -> u8 {
        match element {
            SetBitmaskElement::Int(_) => 0,
            SetBitmaskElement::Bool(_) => 1,
            SetBitmaskElement::String(_) => 2,
            SetBitmaskElement::ModelValue(_) => 3,
        }
    }

    /// Raw-i64 compare encoding of one tuple-key element, matching the flat
    /// buffer scalar encoding (`flat_state.rs::value_to_scalar_i64`):
    /// `Int(n) -> n`, `Bool(b) -> 0/1`, `String`/`ModelValue` -> interned
    /// `NameId` (the universe-index raw value scalar `ExplicitScalarDomain`
    /// keys use).
    fn tuple_key_element_raw_value(element: &SetBitmaskElement) -> i64 {
        match element {
            SetBitmaskElement::Int(n) => *n,
            SetBitmaskElement::Bool(b) => i64::from(*b),
            SetBitmaskElement::String(name) | SetBitmaskElement::ModelValue(name) => {
                i64::from(name.0)
            }
        }
    }

    /// Extract the canonical ordered typed tuple keys from a constant-pool
    /// `Value`, or `None` when it is not a matching fully-enumerated tuple set.
    ///
    /// Accepts either a `Value::Set` of tuples (the materialized domain set,
    /// e.g. `Pos`) or a `Value::Func` keyed by tuples. Each key must be a tuple
    /// (`Value::Tuple` or 1-indexed `Value::Seq`) whose arity and per-position
    /// scalar shapes match `element_layouts`. Keys are sorted by `Value::cmp`
    /// to match the flat-buffer canonical order.
    fn tuple_domain_keys_from_value(
        value: &tla_value::Value,
        element_layouts: &[CompoundLayout],
        pair_count: usize,
    ) -> Option<Vec<Vec<SetBitmaskElement>>> {
        use tla_value::Value;
        let key_values: Vec<Value> = match value {
            Value::Set(set) if set.len() == pair_count => set.iter().cloned().collect(),
            Value::Func(func) if func.domain_len() == pair_count => {
                func.domain_iter().cloned().collect()
            }
            _ => return None,
        };
        // Sort the keys by canonical `Value` order so `keys[i]` matches flat
        // slot `i` exactly (the check-side layout sorts identically). A
        // `Value::Set` is already canonically sorted, but a `Value::Func`
        // domain iterator is not guaranteed to be, so sort unconditionally.
        let mut sorted_keys = key_values;
        sorted_keys.sort();
        let mut out = Vec::with_capacity(pair_count);
        for key in &sorted_keys {
            out.push(Self::tuple_key_elements(key, element_layouts)?);
        }
        Some(out)
    }

    /// Convert one tuple key `Value` to its per-position typed element
    /// encoding, validating arity and per-position scalar shape against
    /// `element_layouts`.
    ///
    /// `SmallInt`/`Bool` map to their raw lanes; `String`/`ModelValue`
    /// positions (a `CompoundLayout::String` element layout) intern to their
    /// `NameId` lane, mirroring `scalar_key_from_value_for_layout`. Any
    /// non-matching shape (wrong arity, wrong element type, big integer)
    /// returns `None` so recovery fails closed.
    fn tuple_key_elements(
        key: &tla_value::Value,
        element_layouts: &[CompoundLayout],
    ) -> Option<Vec<SetBitmaskElement>> {
        use tla_value::Value;
        let elems: Vec<&Value> = match key {
            Value::Tuple(elems) => elems.iter().collect(),
            Value::Seq(seq) => seq.iter().collect(),
            _ => return None,
        };
        if elems.len() != element_layouts.len() {
            return None;
        }
        let mut out = Vec::with_capacity(elems.len());
        for (elem, layout) in elems.iter().zip(element_layouts.iter()) {
            let element = match (layout, *elem) {
                (CompoundLayout::Int, Value::SmallInt(n)) => SetBitmaskElement::Int(*n),
                (CompoundLayout::Bool, Value::Bool(b)) => SetBitmaskElement::Bool(*b),
                (CompoundLayout::String, Value::String(name)) => {
                    SetBitmaskElement::String(tla_core::intern_name(name.as_ref()))
                }
                (CompoundLayout::String, Value::ModelValue(name)) => {
                    SetBitmaskElement::ModelValue(tla_core::intern_name(name.as_ref()))
                }
                _ => return None,
            };
            out.push(element);
        }
        Some(out)
    }

    /// Lower `FuncApply` against a tuple / cross-product-keyed compact function
    /// by comparing the tuple argument element-wise against each canonical key.
    ///
    /// `tuple_keys[i]` is the per-position typed element encoding of the `i`-th
    /// canonical domain key; the matching range value lives at compact flat slot
    /// `i` (`store_compact_function_apply_result_at_index` computes
    /// `source_slot.offset + i * value_stride`).
    ///
    /// When the argument tuple's elements are compile-time-known typed scalars
    /// (a `SeqNew`/`TupleNew` of tracked constants, e.g. an exists-expanded
    /// `grid[<<1,2>>]`), the ordinal is resolved at compile time — an O(1)
    /// direct slot load, sort-aware per the H5 discipline (a String key never
    /// matches a ModelValue argument), with a compile-time-known domain miss
    /// lowering to the runtime error path, mirroring the scalar
    /// explicit-domain const path.
    ///
    /// Otherwise: for the argument tuple pointer the element at TLA+ position
    /// `j` (1-indexed) is `arg_ptr[j]` (slot 0 is the length header). A key
    /// matches when **every** element compares equal (AND-folded per-position
    /// `ICmp`), reproducing TLA+ tuple equality exactly; a non-key argument
    /// falls through to the not-found runtime error (fail-closed miss →
    /// interpreter-visible `TypeMismatch`), matching the interpreter's
    /// total-function domain check. Raw-value comparison is sound because the
    /// recovery validated per-position sort homogeneity and whole-row raw
    /// distinctness.
    #[allow(clippy::too_many_arguments)]
    fn lower_tuple_keyed_compact_func_apply(
        &mut self,
        block_idx: usize,
        rd: u8,
        func_reg: u8,
        arg_reg: u8,
        source_slot: super::CompactStateSlot,
        len: u32,
        value: Option<Box<super::AggregateShape>>,
        tuple_keys: Vec<Vec<SetBitmaskElement>>,
    ) -> Result<Option<usize>, TrustIrError> {
        if tuple_keys.len() != usize::try_from(len).expect("u32 length must fit in usize") {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: tuple-keyed compact function r{func_reg} domain length {} does not match pair_count {len}",
                tuple_keys.len()
            )));
        }
        let arity = tuple_keys.first().map(Vec::len).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: tuple-keyed compact function r{func_reg} has empty domain"
            ))
        })?;
        if arity == 0 || tuple_keys.iter().any(|key| key.len() != arity) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: tuple-keyed compact function r{func_reg} has non-uniform key arity"
            )));
        }
        // The argument must be a tuple of the same arity for TLA+ equality to be
        // well-typed. The JIT models a tuple value as a `Sequence` shape with an
        // exact extent; require that extent to equal the key arity so element
        // slots `1..=arity` are guaranteed present. Fail closed otherwise.
        let arg_arity_ok = matches!(
            self.aggregate_shapes.get(&arg_reg),
            Some(super::AggregateShape::Sequence {
                extent: super::SequenceExtent::Exact(extent),
                ..
            }) if usize::try_from(*extent).ok() == Some(arity)
        );
        if !arg_arity_ok {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: tuple-keyed compact function r{func_reg} argument is not a fixed-arity tuple matching the key arity"
            )));
        }

        let value_shape = self
            .compact_function_value_shape_from_source_layout(
                source_slot,
                len,
                None,
                None,
                value.as_deref(),
            )
            .or_else(|| value.as_deref().cloned());
        let value_stride = value_shape
            .as_ref()
            .and_then(super::AggregateShape::compact_slot_count)
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncApply: tuple-keyed compact function r{func_reg} requires fixed-width value shape, got {value_shape:?}"
                ))
            })?;

        // Const-key ordinal fast path: the argument tuple's elements were all
        // compile-time-known typed scalars at construction. Typed equality is
        // exact (sort-aware), so the ordinal — or the domain miss — is decided
        // here, mirroring the scalar explicit-domain const path.
        if let Some(arg_elements) = self.const_tuple_key_elements_of(arg_reg, arity) {
            let Some(index) = tuple_keys.iter().position(|row| *row == arg_elements) else {
                self.emit_runtime_error_and_return(block_idx, JitRuntimeErrorKind::TypeMismatch);
                return Ok(None);
            };
            self.store_compact_function_apply_result_at_index(
                block_idx,
                rd,
                source_slot,
                u32::try_from(index).expect("domain index must fit in u32"),
                value_stride,
                value_shape,
            )?;
            return Ok(Some(block_idx));
        }

        // Load the argument tuple's element slots once (slot 0 is the length
        // header; TLA+ position `j` is stored at slot `j`). A tuple position
        // constructed from a tagged-scalar-union register holds the union
        // INDEX — decode it to the member's raw value so the AND-folded
        // compares below run in the key table's raw-member space (WP-18
        // follow-on).
        let arg_ptr = self.load_reg_as_ptr(block_idx, arg_reg)?;
        let arg_elems: Vec<ValueId> = (1..=arity)
            .map(|j| {
                let offset = u32::try_from(j).expect("tuple arity must fit in u32");
                let raw = self.load_at_offset(block_idx, arg_ptr, offset);
                self.decode_tuple_key_elem_raw_value(block_idx, arg_reg, j - 1, raw)
            })
            .collect();

        let not_found_blk = self.new_aux_block("compact_fapply_tuple_not_found");
        let merge_blk = self.new_aux_block("compact_fapply_tuple_merge");
        let merge_id = self.block_id_of(merge_blk);
        let mut check_blk = block_idx;
        for (index, key) in tuple_keys.iter().enumerate() {
            let found_blk = self.new_aux_block("compact_fapply_tuple_found");
            let next_blk =
                if index + 1 == usize::try_from(len).expect("u32 length must fit in usize") {
                    not_found_blk
                } else {
                    self.new_aux_block("compact_fapply_tuple_check")
                };
            // Element-wise equality, AND-folded across all tuple positions.
            let mut matches_key: Option<ValueId> = None;
            for (elem_val, key_elem) in arg_elems.iter().zip(key.iter()) {
                let key_const =
                    self.emit_i64_const(check_blk, Self::tuple_key_element_raw_value(key_elem));
                let elem_eq = self.emit_with_result(
                    check_blk,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: *elem_val,
                        rhs: key_const,
                    },
                );
                matches_key = Some(match matches_key {
                    None => elem_eq,
                    // `elem_eq` and `prev` are `ICmp` results (Bool); the
                    // AND-fold that combines them must be typed `Bool`, not
                    // `I64` (the codegen adapter rejects a `BinOp::And` whose
                    // declared type disagrees with its Bool operands). Mirrors
                    // the boolean-`And` convention in `lower_boolean_binary`.
                    Some(prev) => self.emit_with_result(
                        check_blk,
                        Inst::BinOp {
                            op: BinOp::And,
                            ty: Ty::Bool,
                            lhs: prev,
                            rhs: elem_eq,
                        },
                    ),
                });
            }
            let matches_key =
                matches_key.expect("non-empty key arity guarantees at least one comparison");
            self.emit(
                check_blk,
                InstrNode::new(Inst::CondBr {
                    cond: matches_key,
                    then_target: self.block_id_of(found_blk),
                    then_args: vec![],
                    else_target: self.block_id_of(next_blk),
                    else_args: vec![],
                }),
            );
            self.store_compact_function_apply_result_at_index(
                found_blk,
                rd,
                source_slot,
                u32::try_from(index).expect("domain index must fit in u32"),
                value_stride,
                value_shape.clone(),
            )?;
            self.emit(
                found_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );
            check_blk = next_blk;
        }
        self.emit_runtime_error_and_return(not_found_blk, JitRuntimeErrorKind::TypeMismatch);
        Ok(Some(merge_blk))
    }

    /// WP-09 (item 7 write half): `[f EXCEPT ![<<k1, .., kn>>] = v]` on a
    /// tuple/cross-product-keyed compact function (btree `childOf`/`valOf`).
    ///
    /// The ordered tuple-key table is recovered by
    /// [`Self::tuple_function_domain_keys_for_explicit_function`] (channel 1 =
    /// the WP-04 ABI-carried table, channel 2 = const-pool fallback), so slot
    /// `i` of the compact buffer provably stores the range value at `keys[i]`.
    ///
    /// * **const tuple key** — the ordinal (or the out-of-domain identity) is
    ///   decided at compile time: an in-domain key copies the source buffer and
    ///   overwrites the `ordinal * value_stride` window; an out-of-domain key
    ///   is an IDENTITY copy (TLA+ EXCEPT semantics: the function is
    ///   unchanged — pinned by test).
    /// * **runtime tuple key** — per-position AND-folded raw compares (the same
    ///   compare scaffold as the tuple-keyed `FuncApply`) produce one
    ///   `key_match` per ordinal; every slot is written through
    ///   `Select(key_match, new, old)` (mirroring the scalar runtime-key path
    ///   of `lower_explicit_domain_compact_func_except`), so a key matching no
    ///   row degrades to the identity copy — the same EXCEPT semantics, with
    ///   no error path.
    /// * **replacement values** reuse the masked replacement-source ladder, so
    ///   a union-range destination (`AggregateShape::TaggedScalarUnion`)
    ///   encodes via WP-05's arm-aware universe-index path
    ///   (`compact_tagged_scalar_union_replacement_source`), and anything
    ///   unencodable fails closed to the interpreter.
    ///
    /// The H5 raw-compare discipline is inherited from the validated table
    /// (`validated_tuple_key_table`): every position is sort-homogeneous and
    /// all rows are pairwise raw-distinct, so a runtime tuple maps to at most
    /// one ordinal — identical to the read half's soundness argument.
    #[allow(clippy::too_many_arguments)]
    fn lower_tuple_keyed_compact_func_except(
        &mut self,
        block_idx: usize,
        rd: u8,
        func_reg: u8,
        path_reg: u8,
        val_reg: u8,
        source_slot: super::CompactStateSlot,
        len: u32,
        value: Option<Box<super::AggregateShape>>,
        tuple_keys: Vec<Vec<SetBitmaskElement>>,
    ) -> Result<Option<usize>, TrustIrError> {
        if tuple_keys.len() != usize::try_from(len).expect("u32 length must fit in usize") {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: tuple-keyed compact function r{func_reg} domain length {} does not match pair_count {len}",
                tuple_keys.len()
            )));
        }
        let arity = tuple_keys.first().map(Vec::len).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: tuple-keyed compact function r{func_reg} has empty domain"
            ))
        })?;
        if arity == 0 || tuple_keys.iter().any(|key| key.len() != arity) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: tuple-keyed compact function r{func_reg} has non-uniform key arity"
            )));
        }
        let value_shape = self
            .compact_function_value_shape_from_source_layout(
                source_slot,
                len,
                None,
                None,
                value.as_deref(),
            )
            .or_else(|| value.as_deref().cloned())
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept: tuple-keyed compact function r{func_reg} requires fixed-width value shape, got {value:?}"
                ))
            })?;
        let value_stride = value_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: tuple-keyed compact function r{func_reg} requires fixed-width value shape, got {value_shape:?}"
            ))
        })?;
        let total_slots = len.checked_mul(value_stride).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: compact function slot count overflows: {len} * {value_stride}"
            ))
        })?;
        let result_shape = super::AggregateShape::Function {
            len,
            domain_lo: None,
            domain: None,
            value: Some(Box::new(value_shape.clone())),
        };

        // Const tuple key: the ordinal (or the out-of-domain identity) is
        // decided at compile time via typed, sort-aware equality.
        if let Some(path_elements) = self.const_tuple_key_elements_of(path_reg, arity) {
            let Some(replace_idx) = tuple_keys.iter().position(|row| *row == path_elements) else {
                // Out-of-domain EXCEPT key: TLA+ semantics leave the function
                // unchanged — identity copy, never an error.
                self.store_compact_identity_result(block_idx, rd, source_slot, result_shape)?;
                return Ok(Some(block_idx));
            };
            let replace_idx = u32::try_from(replace_idx).expect("domain index must fit in u32");
            let replacement = if let Some(replacement) = self
                .compact_tagged_scalar_or_set_replacement_source(block_idx, val_reg, &value_shape)?
            {
                replacement
            } else if let Some(replacement) = self.compact_tagged_scalar_union_replacement_source(
                block_idx,
                val_reg,
                &value_shape,
            )? {
                replacement
            } else if let Some(replacement) = self
                .compact_set_bitmask_dynamic_set_replacement_source(
                    block_idx,
                    val_reg,
                    &value_shape,
                )? {
                replacement
            } else {
                self.compact_value_source_for_reg(block_idx, val_reg, &value_shape)?
            };
            let block_idx = replacement.block_idx;
            let replacement_source = replacement.slot;
            let result_ptr = self.alloc_aggregate(block_idx, total_slots);
            for slot in 0..total_slots {
                let old_val = self.load_at_offset(
                    block_idx,
                    source_slot.source_ptr,
                    source_slot.offset + slot,
                );
                self.store_at_offset(block_idx, result_ptr, slot, old_val);
            }
            let replace_base = replace_idx.checked_mul(value_stride).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept: compact function replacement slot overflows: {replace_idx} * {value_stride}"
                ))
            })?;
            for value_offset in 0..value_stride {
                let new_val = self.load_at_offset(
                    block_idx,
                    replacement_source.source_ptr,
                    replacement_source.offset + value_offset,
                );
                self.store_at_offset(block_idx, result_ptr, replace_base + value_offset, new_val);
            }
            if total_slots == 1 {
                let first = self.load_at_offset(block_idx, result_ptr, 0);
                self.store_single_slot_compact_result(
                    block_idx,
                    rd,
                    first,
                    super::CompactStateSlot::raw(result_ptr, 0),
                    result_shape,
                )?;
            } else {
                self.store_compact_aggregate_result(block_idx, rd, result_ptr, result_shape)?;
            }
            return Ok(Some(block_idx));
        }

        // Runtime tuple key: the key must be a fixed-arity tuple matching the
        // key arity (the same contract the tuple-keyed FuncApply enforces), so
        // element slots `1..=arity` are guaranteed present. Fail closed
        // otherwise.
        let arg_arity_ok = matches!(
            self.aggregate_shapes.get(&path_reg),
            Some(super::AggregateShape::Sequence {
                extent: super::SequenceExtent::Exact(extent),
                ..
            }) if usize::try_from(*extent).ok() == Some(arity)
        );
        if !arg_arity_ok {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: tuple-keyed compact function r{func_reg} EXCEPT key is not a fixed-arity tuple matching the key arity"
            )));
        }

        let masked_replacement = if value_stride == 1 {
            if let Some(replacement) = self.compact_tagged_scalar_or_set_replacement_source(
                block_idx,
                val_reg,
                &value_shape,
            )? {
                Some(replacement)
            } else if let Some(replacement) = self.compact_tagged_scalar_union_replacement_source(
                block_idx,
                val_reg,
                &value_shape,
            )? {
                Some(replacement)
            } else {
                self.compact_set_bitmask_dynamic_set_replacement_source(
                    block_idx,
                    val_reg,
                    &value_shape,
                )?
            }
        } else {
            None
        };
        let (block_idx, new_scalar_val, new_compact_source) = if let Some(materialized) =
            masked_replacement
        {
            (materialized.block_idx, None, Some(materialized.slot))
        } else {
            let new_scalar_val =
                if value_stride == 1 && Self::is_single_slot_flat_aggregate_value(&value_shape) {
                    Some(self.load_reg_as_compatible_single_slot_value(
                        block_idx,
                        val_reg,
                        &value_shape,
                        "FuncExcept tuple-keyed compact function replacement",
                    )?)
                } else {
                    None
                };
            let new_compact_source = if new_scalar_val.is_none() {
                let materialized =
                    self.compact_value_source_for_reg(block_idx, val_reg, &value_shape)?;
                Some((materialized.block_idx, materialized.slot))
            } else {
                None
            };
            if let Some((materialized_block, source_slot)) = new_compact_source {
                (materialized_block, new_scalar_val, Some(source_slot))
            } else {
                (block_idx, new_scalar_val, None)
            }
        };

        // Load the key tuple's element slots once (slot 0 is the length
        // header; TLA+ position `j` is stored at slot `j`). Union-index
        // positions are decoded to raw member space exactly like the
        // tuple-keyed FuncApply chain (WP-18 follow-on).
        let arg_ptr = self.load_reg_as_ptr(block_idx, path_reg)?;
        let arg_elems: Vec<ValueId> = (1..=arity)
            .map(|j| {
                let offset = u32::try_from(j).expect("tuple arity must fit in u32");
                let raw = self.load_at_offset(block_idx, arg_ptr, offset);
                self.decode_tuple_key_elem_raw_value(block_idx, path_reg, j - 1, raw)
            })
            .collect();

        let result_ptr = self.alloc_aggregate(block_idx, total_slots);
        for (idx, key) in tuple_keys.iter().enumerate() {
            let idx = u32::try_from(idx).expect("domain index must fit in u32");
            // Element-wise equality, AND-folded across all tuple positions
            // (Bool-typed fold, mirroring the tuple-keyed FuncApply chain).
            let mut key_match: Option<ValueId> = None;
            for (elem_val, key_elem) in arg_elems.iter().zip(key.iter()) {
                let key_const =
                    self.emit_i64_const(block_idx, Self::tuple_key_element_raw_value(key_elem));
                let elem_eq = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: *elem_val,
                        rhs: key_const,
                    },
                );
                key_match = Some(match key_match {
                    None => elem_eq,
                    Some(prev) => self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::And,
                            ty: Ty::Bool,
                            lhs: prev,
                            rhs: elem_eq,
                        },
                    ),
                });
            }
            let key_match =
                key_match.expect("non-empty key arity guarantees at least one comparison");
            for value_offset in 0..value_stride {
                let source_offset = source_slot.offset + idx * value_stride + value_offset;
                let old_val = self.load_at_offset(block_idx, source_slot.source_ptr, source_offset);
                let new_val = if let Some(new_scalar_val) = new_scalar_val {
                    new_scalar_val
                } else {
                    let new_compact_source =
                        new_compact_source.expect("compact update source was checked above");
                    self.load_at_offset(
                        block_idx,
                        new_compact_source.source_ptr,
                        new_compact_source.offset + value_offset,
                    )
                };
                let selected_val = self.emit_with_result(
                    block_idx,
                    Inst::Select {
                        ty: Ty::I64,
                        cond: key_match,
                        then_val: new_val,
                        else_val: old_val,
                    },
                );
                self.store_at_offset(
                    block_idx,
                    result_ptr,
                    idx * value_stride + value_offset,
                    selected_val,
                );
            }
        }

        if total_slots == 1 {
            let first = self.load_at_offset(block_idx, result_ptr, 0);
            self.store_single_slot_compact_result(
                block_idx,
                rd,
                first,
                super::CompactStateSlot::raw(result_ptr, 0),
                result_shape,
            )?;
        } else {
            self.store_compact_aggregate_result(block_idx, rd, result_ptr, result_shape)?;
        }
        Ok(Some(block_idx))
    }

    fn compact_function_apply_result_domain(
        &self,
        source_slot: super::CompactStateSlot,
        value_shape: Option<&super::AggregateShape>,
    ) -> Option<super::CompactFunctionDomain> {
        match value_shape {
            Some(
                value_shape @ super::AggregateShape::Function {
                    domain_lo: None, ..
                },
            ) => value_shape
                .function_explicit_domain()
                .or_else(|| {
                    self.compact_function_value_domain_from_raw_state_sub_slot(
                        source_slot,
                        value_shape,
                    )
                })
                .or_else(|| self.compact_function_value_domain_from_raw_state_slot(source_slot)),
            _ => None,
        }
    }

    fn compact_nested_function_domain_from_raw_state_value_slot(
        &self,
        source_slot: super::CompactStateSlot,
        value_shape: &super::AggregateShape,
    ) -> Option<super::CompactFunctionDomain> {
        match value_shape {
            super::AggregateShape::Function {
                domain_lo: None, ..
            } => {
                self.compact_function_domain_from_raw_state_sub_slot(source_slot, Some(value_shape))
            }
            _ => None,
        }
    }

    fn compact_function_domain_from_raw_state_sub_slot(
        &self,
        source_slot: super::CompactStateSlot,
        expected_shape: Option<&super::AggregateShape>,
    ) -> Option<super::CompactFunctionDomain> {
        let layout = self.compound_function_layout_at_raw_state_sub_slot(
            source_slot,
            expected_shape,
            false,
        )?;
        self.compact_function_domain_from_layout(layout)
    }

    fn compact_function_value_domain_from_raw_state_sub_slot(
        &self,
        source_slot: super::CompactStateSlot,
        expected_value_shape: &super::AggregateShape,
    ) -> Option<super::CompactFunctionDomain> {
        let layout = self.compound_function_layout_at_raw_state_sub_slot(
            source_slot,
            Some(expected_value_shape),
            false,
        )?;
        self.compact_function_domain_from_layout(layout)
    }

    fn compound_function_layout_at_raw_state_sub_slot(
        &self,
        source_slot: super::CompactStateSlot,
        expected_shape: Option<&super::AggregateShape>,
        allow_current: bool,
    ) -> Option<&CompoundLayout> {
        if !source_slot.is_raw_compact_slot() {
            return None;
        }
        if source_slot.source_ptr != self.state_in_ptr
            && self.state_out_ptr != Some(source_slot.source_ptr)
        {
            return None;
        }

        let state_layout = self.config.state_layout.as_ref()?;
        let offsets = state_layout.compute_compact_var_offsets();
        for (var_idx, offset) in offsets.into_iter().enumerate() {
            let var_base = u32::try_from(offset).ok()?;
            let relative_offset = source_slot.offset.checked_sub(var_base)?;
            let VarLayout::Compound(layout) = state_layout.var_layout(var_idx)? else {
                continue;
            };
            let var_slot_count = u32::try_from(layout.compact_slot_count()).ok()?;
            if relative_offset >= var_slot_count {
                continue;
            }
            if let Some(nested) = Self::compound_function_layout_at_compact_offset(
                layout,
                relative_offset,
                expected_shape,
                allow_current,
            ) {
                return Some(nested);
            }
        }
        None
    }

    fn compound_function_layout_at_compact_offset<'layout>(
        layout: &'layout CompoundLayout,
        offset: u32,
        expected_shape: Option<&super::AggregateShape>,
        allow_current: bool,
    ) -> Option<&'layout CompoundLayout> {
        if allow_current
            && offset == 0
            && Self::function_layout_matches_shape(layout, expected_shape)
        {
            return Some(layout);
        }

        match layout {
            CompoundLayout::Function {
                value_layout,
                pair_count: Some(pair_count),
                ..
            } => {
                let value_stride = u32::try_from(value_layout.compact_slot_count()).ok()?;
                if value_stride == 0 {
                    return None;
                }
                let pair_count = u32::try_from(*pair_count).ok()?;
                let total = pair_count.checked_mul(value_stride)?;
                if offset >= total {
                    return None;
                }
                Self::compound_function_layout_at_compact_offset(
                    value_layout,
                    offset % value_stride,
                    expected_shape,
                    true,
                )
            }
            CompoundLayout::Record { fields } => {
                let mut field_base = 0_u32;
                for (_, field_layout) in fields {
                    let field_slots = u32::try_from(field_layout.compact_slot_count()).ok()?;
                    if offset < field_base.checked_add(field_slots)? {
                        return Self::compound_function_layout_at_compact_offset(
                            field_layout,
                            offset - field_base,
                            expected_shape,
                            true,
                        );
                    }
                    field_base = field_base.checked_add(field_slots)?;
                }
                None
            }
            CompoundLayout::Tuple { element_layouts } => {
                let mut element_base = 0_u32;
                for element_layout in element_layouts {
                    let element_slots = u32::try_from(element_layout.compact_slot_count()).ok()?;
                    if offset < element_base.checked_add(element_slots)? {
                        return Self::compound_function_layout_at_compact_offset(
                            element_layout,
                            offset - element_base,
                            expected_shape,
                            true,
                        );
                    }
                    element_base = element_base.checked_add(element_slots)?;
                }
                None
            }
            CompoundLayout::Sequence {
                element_layout,
                element_count: Some(element_count),
                ..
            } => {
                if offset == 0 {
                    return None;
                }
                let element_stride = u32::try_from(element_layout.compact_slot_count()).ok()?;
                if element_stride == 0 {
                    return None;
                }
                let element_count = u32::try_from(*element_count).ok()?;
                let elements_total = element_count.checked_mul(element_stride)?;
                let element_offset = offset.checked_sub(1)?;
                if element_offset >= elements_total {
                    return None;
                }
                Self::compound_function_layout_at_compact_offset(
                    element_layout,
                    element_offset % element_stride,
                    expected_shape,
                    true,
                )
            }
            _ => None,
        }
    }

    fn function_layout_matches_shape(
        layout: &CompoundLayout,
        expected_shape: Option<&super::AggregateShape>,
    ) -> bool {
        let CompoundLayout::Function {
            pair_count: Some(pair_count),
            domain_lo,
            value_layout,
            ..
        } = layout
        else {
            return false;
        };
        let Some(super::AggregateShape::Function {
            len,
            domain_lo: expected_domain_lo,
            value,
            ..
        }) = expected_shape
        else {
            return true;
        };
        let Ok(pair_count) = u32::try_from(*pair_count) else {
            return false;
        };
        if pair_count != *len || domain_lo != expected_domain_lo {
            return false;
        }
        let Some(value_shape) = value.as_deref() else {
            return true;
        };
        let Some(layout_value_shape) = super::tracked_shape_from_compound_layout(value_layout)
        else {
            return false;
        };
        Self::same_compact_physical_layout(&layout_value_shape, value_shape)
    }

    fn lower_explicit_domain_compact_func_except(
        &mut self,
        block_idx: usize,
        rd: u8,
        func_reg: u8,
        path_reg: u8,
        val_reg: u8,
        source_slot: super::CompactStateSlot,
        len: u32,
        value_shape: super::AggregateShape,
        domain_keys: super::CompactFunctionDomain,
    ) -> Result<Option<usize>, TrustIrError> {
        if domain_keys.len() != usize::try_from(len).expect("u32 length must fit in usize") {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: compact function r{func_reg} domain metadata length {} does not match pair_count {len}",
                domain_keys.len()
            )));
        }
        let value_stride = value_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: compact function r{func_reg} requires fixed-width value shape, got {value_shape:?}"
            ))
        })?;
        let total_slots = len.checked_mul(value_stride).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: compact function slot count overflows: {len} * {value_stride}"
            ))
        })?;
        let result_shape = super::AggregateShape::Function {
            len,
            domain_lo: None,
            domain: Some(domain_keys.clone()),
            value: Some(Box::new(value_shape.clone())),
        };

        if let Some(path_raw) = self.scalar_of(path_reg) {
            let replace_idx = match &domain_keys {
                super::CompactFunctionDomain::Raw(keys) => {
                    keys.iter().position(|key| *key == path_raw)
                }
                super::CompactFunctionDomain::Exact(keys) => self
                    .const_scalar_domain_key_of(path_reg)
                    .and_then(|path_key| keys.iter().position(|key| *key == path_key)),
            };
            let Some(replace_idx) = replace_idx else {
                self.store_compact_identity_result(block_idx, rd, source_slot, result_shape)?;
                self.compact_function_domains.insert(rd, domain_keys);
                return Ok(Some(block_idx));
            };
            let replace_idx = u32::try_from(replace_idx).expect("domain index must fit in u32");
            let replacement = if let Some(replacement) = self
                .compact_tagged_scalar_or_set_replacement_source(block_idx, val_reg, &value_shape)?
            {
                replacement
            } else if let Some(replacement) = self.compact_tagged_scalar_union_replacement_source(
                block_idx,
                val_reg,
                &value_shape,
            )? {
                replacement
            } else if let Some(replacement) = self.compact_scalar_domain_set_replacement_source(
                block_idx,
                val_reg,
                &value_shape,
                &domain_keys,
            )? {
                replacement
            } else if let Some(replacement) = self
                .compact_set_bitmask_dynamic_set_replacement_source(
                    block_idx,
                    val_reg,
                    &value_shape,
                )? {
                replacement
            } else {
                self.compact_value_source_for_reg(block_idx, val_reg, &value_shape)?
            };
            let block_idx = replacement.block_idx;
            let replacement_source = replacement.slot;
            let result_ptr = self.alloc_aggregate(block_idx, total_slots);
            for slot in 0..total_slots {
                let old_val = self.load_at_offset(
                    block_idx,
                    source_slot.source_ptr,
                    source_slot.offset + slot,
                );
                self.store_at_offset(block_idx, result_ptr, slot, old_val);
            }
            let replace_base = replace_idx.checked_mul(value_stride).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept: compact function replacement slot overflows: {replace_idx} * {value_stride}"
                ))
            })?;
            for value_offset in 0..value_stride {
                let new_val = self.load_at_offset(
                    block_idx,
                    replacement_source.source_ptr,
                    replacement_source.offset + value_offset,
                );
                self.store_at_offset(block_idx, result_ptr, replace_base + value_offset, new_val);
            }
            if total_slots == 1 {
                let first = self.load_at_offset(block_idx, result_ptr, 0);
                self.store_single_slot_compact_result(
                    block_idx,
                    rd,
                    first,
                    super::CompactStateSlot::raw(result_ptr, 0),
                    result_shape,
                )?;
            } else {
                self.store_compact_aggregate_result(block_idx, rd, result_ptr, result_shape)?;
            }
            self.compact_function_domains.insert(rd, domain_keys);
            return Ok(Some(block_idx));
        }

        let result_ptr = self.alloc_aggregate(block_idx, total_slots);
        let path_val = self.load_reg(block_idx, path_reg)?;
        let path_shape = self.scalar_shape_of(path_reg);
        let allow_untyped_exact_raw_match = match &domain_keys {
            super::CompactFunctionDomain::Raw(_) => true,
            // A homogeneous (single-shape) exact domain is the common case; a
            // mixed-shape domain is still safe to match by raw value when every
            // key has a distinct raw encoding, because each raw value then maps
            // to exactly one key (see `exact_domain_all_distinct_raw_values`).
            super::CompactFunctionDomain::Exact(keys) => {
                Self::exact_domain_all_one_shape(keys)
                    || Self::exact_domain_all_distinct_raw_values(keys)
            }
        };
        let masked_replacement = if value_stride == 1 {
            if let Some(replacement) = self.compact_tagged_scalar_or_set_replacement_source(
                block_idx,
                val_reg,
                &value_shape,
            )? {
                Some(replacement)
            } else if let Some(replacement) = self.compact_tagged_scalar_union_replacement_source(
                block_idx,
                val_reg,
                &value_shape,
            )? {
                Some(replacement)
            } else if let Some(replacement) = self.compact_scalar_domain_set_replacement_source(
                block_idx,
                val_reg,
                &value_shape,
                &domain_keys,
            )? {
                Some(replacement)
            } else {
                self.compact_set_bitmask_dynamic_set_replacement_source(
                    block_idx,
                    val_reg,
                    &value_shape,
                )?
            }
        } else {
            None
        };
        let (block_idx, new_scalar_val, new_compact_source) = if let Some(materialized) =
            masked_replacement
        {
            (materialized.block_idx, None, Some(materialized.slot))
        } else {
            let new_scalar_val =
                if value_stride == 1 && Self::is_single_slot_flat_aggregate_value(&value_shape) {
                    Some(self.load_reg_as_compatible_single_slot_value(
                        block_idx,
                        val_reg,
                        &value_shape,
                        "FuncExcept compact function replacement",
                    )?)
                } else {
                    None
                };
            let new_compact_source = if new_scalar_val.is_none() {
                let materialized =
                    self.compact_value_source_for_reg(block_idx, val_reg, &value_shape)?;
                Some((materialized.block_idx, materialized.slot))
            } else {
                None
            };
            if let Some((materialized_block, source_slot)) = new_compact_source {
                (materialized_block, new_scalar_val, Some(source_slot))
            } else {
                (block_idx, new_scalar_val, None)
            }
        };

        let domain_key_refs: Vec<DomainKeyRef> = match &domain_keys {
            super::CompactFunctionDomain::Raw(keys) => {
                keys.iter().copied().map(DomainKeyRef::Raw).collect()
            }
            super::CompactFunctionDomain::Exact(keys) => {
                keys.iter().copied().map(DomainKeyRef::Exact).collect()
            }
        };
        for (idx, key) in domain_key_refs.into_iter().enumerate() {
            let idx = u32::try_from(idx).expect("domain index must fit in u32");
            let key_match = match key {
                DomainKeyRef::Raw(key) => {
                    let key_val = self.emit_i64_const(block_idx, key);
                    self.emit_with_result(
                        block_idx,
                        Inst::ICmp {
                            op: ICmpOp::Eq,
                            ty: Ty::I64,
                            lhs: path_val,
                            rhs: key_val,
                        },
                    )
                }
                DomainKeyRef::Exact(key) => self.emit_exact_domain_key_match(
                    block_idx,
                    path_val,
                    path_shape.as_ref(),
                    allow_untyped_exact_raw_match,
                    &key,
                )?,
            };
            for value_offset in 0..value_stride {
                let source_offset = source_slot.offset + idx * value_stride + value_offset;
                let old_val = self.load_at_offset(block_idx, source_slot.source_ptr, source_offset);
                let new_val = if let Some(new_scalar_val) = new_scalar_val {
                    new_scalar_val
                } else {
                    let new_compact_source =
                        new_compact_source.expect("compact update source was checked above");
                    self.load_at_offset(
                        block_idx,
                        new_compact_source.source_ptr,
                        new_compact_source.offset + value_offset,
                    )
                };
                let selected_val = self.emit_with_result(
                    block_idx,
                    Inst::Select {
                        ty: Ty::I64,
                        cond: key_match,
                        then_val: new_val,
                        else_val: old_val,
                    },
                );
                self.store_at_offset(
                    block_idx,
                    result_ptr,
                    idx * value_stride + value_offset,
                    selected_val,
                );
            }
        }

        if total_slots == 1 {
            let first = self.load_at_offset(block_idx, result_ptr, 0);
            self.store_single_slot_compact_result(
                block_idx,
                rd,
                first,
                super::CompactStateSlot::raw(result_ptr, 0),
                result_shape,
            )?;
        } else {
            self.store_compact_aggregate_result(block_idx, rd, result_ptr, result_shape)?;
        }
        self.compact_function_domains.insert(rd, domain_keys);
        Ok(Some(block_idx))
    }

    fn store_compact_identity_result(
        &mut self,
        block_idx: usize,
        rd: u8,
        source_slot: super::CompactStateSlot,
        shape: super::AggregateShape,
    ) -> Result<(), TrustIrError> {
        let slot_count = shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "compact identity result requires fixed-width shape for r{rd}, got {shape:?}"
            ))
        })?;
        if slot_count == 1 {
            let value = self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset);
            self.store_single_slot_compact_result(block_idx, rd, value, source_slot, shape)
        } else {
            let result_ptr = self.alloc_aggregate(block_idx, slot_count);
            for slot in 0..slot_count {
                let value = self.load_at_offset(
                    block_idx,
                    source_slot.source_ptr,
                    source_slot.offset + slot,
                );
                self.store_at_offset(block_idx, result_ptr, slot, value);
            }
            self.store_compact_aggregate_result(block_idx, rd, result_ptr, shape)
        }
    }

    fn compact_record_field_from_scalar_key(
        shape: &super::AggregateShape,
        key: i64,
        mode: super::RecordSelectorMode,
    ) -> Option<(tla_core::NameId, u32, Option<super::AggregateShape>)> {
        let super::AggregateShape::Record { fields } = shape else {
            return None;
        };

        let target_idx = match mode {
            super::RecordSelectorMode::FieldName => {
                let field = tla_core::NameId(u32::try_from(key).ok()?);
                fields.iter().position(|(name, _)| *name == field)
            }
            super::RecordSelectorMode::Positional => {
                let positional = if let Ok(idx) = usize::try_from(key) {
                    (idx < fields.len()).then_some(idx)
                } else {
                    None
                };
                positional.or_else(|| {
                    let field = tla_core::NameId(u32::try_from(key).ok()?);
                    fields.iter().position(|(name, _)| *name == field)
                })
            }
        }?;

        let mut compact_offset = 0_u32;
        for (_, field_shape) in &fields[..target_idx] {
            compact_offset =
                compact_offset.checked_add(field_shape.as_deref()?.compact_slot_count()?)?;
        }
        let (field_name, field_shape) = &fields[target_idx];
        Some((*field_name, compact_offset, field_shape.as_deref().cloned()))
    }

    /// WP-08 (item 6): FuncExcept replacement of a compact `SetBitmask` range
    /// slot from a DYNAMIC materialized small set (elements without static
    /// provenance, e.g. `![newRoot] = {pivot}` with a runtime pivot).
    ///
    /// Delegates to the shared fail-closed runtime loop
    /// (`emit_dynamic_materialized_set_bitmask_mask_i64`): every element must
    /// map to a universe bit; an out-of-universe element raises a typed
    /// runtime error (per-state interpreter fallback) — never the silent
    /// Select-zero drop. Sources the other replacement paths already handle
    /// (SetBitmask identity, exact sets, empty set) return `None` here and
    /// keep their existing lowerings.
    fn compact_set_bitmask_dynamic_set_replacement_source(
        &mut self,
        block_idx: usize,
        reg: u8,
        expected_shape: &super::AggregateShape,
    ) -> Result<Option<super::CompactMaterializationResult>, TrustIrError> {
        let super::AggregateShape::SetBitmask {
            universe_len,
            universe,
        } = expected_shape
        else {
            return Ok(None);
        };
        let Some(capacity) = self.dynamic_set_to_bitmask_source_capacity(reg, universe) else {
            return Ok(None);
        };
        let (block_idx, mask) = self.emit_dynamic_materialized_set_bitmask_mask_i64(
            block_idx,
            reg,
            capacity,
            *universe_len,
            universe,
            "FuncExcept compact SetBitmask dynamic set replacement",
        )?;
        let result_ptr = self.alloc_aggregate(block_idx, 1);
        self.store_at_offset(block_idx, result_ptr, 0, mask);
        Ok(Some(super::CompactMaterializationResult {
            slot: super::CompactStateSlot::raw(result_ptr, 0),
            block_idx,
        }))
    }

    fn compact_scalar_domain_set_replacement_source(
        &mut self,
        block_idx: usize,
        reg: u8,
        expected_shape: &super::AggregateShape,
        domain_keys: &super::CompactFunctionDomain,
    ) -> Result<Option<super::CompactMaterializationResult>, TrustIrError> {
        // This is not general set-to-scalar compatibility. It only handles
        // legacy one-slot string/model-value function ranges whose replacement
        // is a finite set over the same recovered explicit domain; every
        // runtime element is checked against domain_keys before we store a mask.
        if !matches!(
            expected_shape,
            super::AggregateShape::Scalar(
                super::ScalarShape::String | super::ScalarShape::ModelValue
            )
        ) {
            return Ok(None);
        }
        let super::CompactFunctionDomain::Raw(domain_keys) = domain_keys else {
            return Ok(None);
        };

        let universe_len = u32::try_from(domain_keys.len()).map_err(|_| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept compact scalar set replacement: domain length {} does not fit in u32",
                domain_keys.len()
            ))
        })?;
        Self::compact_set_bitmask_valid_mask(
            universe_len,
            "FuncExcept compact scalar set replacement",
        )?;

        let Some(source_shape) = self.aggregate_shapes.get(&reg).cloned() else {
            return Ok(None);
        };
        let (capacity, element_shape) = match source_shape {
            super::AggregateShape::SetBitmask {
                universe_len: source_universe_len,
                universe,
            } if source_universe_len == universe_len
                && universe == super::SetBitmaskUniverse::ExplicitInt(domain_keys.to_vec()) =>
            {
                let result_ptr = self.alloc_aggregate(block_idx, 1);
                let mask = self.load_reg(block_idx, reg)?;
                self.store_at_offset(block_idx, result_ptr, 0, mask);
                return Ok(Some(super::CompactMaterializationResult {
                    slot: super::CompactStateSlot::raw(result_ptr, 0),
                    block_idx,
                }));
            }
            super::AggregateShape::Set { len: 0, .. } => {
                let result_ptr = self.alloc_aggregate(block_idx, 1);
                let zero = self.emit_i64_const(block_idx, 0);
                self.store_at_offset(block_idx, result_ptr, 0, zero);
                return Ok(Some(super::CompactMaterializationResult {
                    slot: super::CompactStateSlot::raw(result_ptr, 0),
                    block_idx,
                }));
            }
            super::AggregateShape::ExactScalarSet {
                scalar: super::ScalarShape::String | super::ScalarShape::ModelValue,
                ref values,
            } => {
                // Values are compile-time known string/model-value NameIds.
                // Domain-membership is checked at compile time (out-of-domain
                // elements fail closed) but the mask is folded via runtime Or
                // instructions to match the recovered-domain encoding shape
                // produced by the dynamic Set/BoundedSet path.
                for value in values {
                    if !domain_keys.iter().any(|key| key == value) {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "FuncExcept compact scalar set replacement exact value {value} not in domain keys {domain_keys:?}"
                        )));
                    }
                }
                let result_ptr = self.alloc_aggregate(block_idx, 1);
                let mut mask = self.emit_i64_const(block_idx, 0);
                for value in values {
                    let idx = domain_keys
                        .iter()
                        .position(|key| key == value)
                        .expect("guard above ensured every value lies in the domain");
                    let bit = self.emit_i64_const(block_idx, 1_i64 << idx);
                    mask = self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::Or,
                            ty: Ty::I64,
                            lhs: mask,
                            rhs: bit,
                        },
                    );
                }
                self.store_at_offset(block_idx, result_ptr, 0, mask);
                return Ok(Some(super::CompactMaterializationResult {
                    slot: super::CompactStateSlot::raw(result_ptr, 0),
                    block_idx,
                }));
            }
            super::AggregateShape::Set {
                len,
                element: Some(element),
            } => (len, *element),
            super::AggregateShape::BoundedSet {
                max_len,
                element: Some(element),
            } => (max_len, *element),
            _ => return Ok(None),
        };
        if !matches!(
            element_shape,
            super::AggregateShape::Scalar(
                super::ScalarShape::String | super::ScalarShape::ModelValue
            )
        ) {
            return Ok(None);
        }

        let source_ptr = self.load_reg_as_ptr(block_idx, reg)?;
        let len_value = self.load_at_offset(block_idx, source_ptr, 0);
        let guard: super::CompactSequenceLenGuardResult = self
            .guard_compact_sequence_len_in_bounds(
                block_idx,
                len_value,
                capacity,
                "compact_scalar_domain_set_replacement",
            );
        let block_idx = guard.block_idx;
        let len_value = guard.len_value;

        let zero = self.emit_i64_const(block_idx, 0);
        let idx_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        let mask_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: mask_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let header_blk = self.new_aux_block("compact_scalar_domain_set_replacement_header");
        let body_blk = self.new_aux_block("compact_scalar_domain_set_replacement_body");
        let accept_blk = self.new_aux_block("compact_scalar_domain_set_replacement_accept");
        let error_blk = self.new_aux_block("compact_scalar_domain_set_replacement_error");
        let done_blk = self.new_aux_block("compact_scalar_domain_set_replacement_done");
        let header_id = self.block_id_of(header_blk);
        let body_id = self.block_id_of(body_blk);
        let accept_id = self.block_id_of(accept_blk);
        let error_id = self.block_id_of(error_blk);
        let done_id = self.block_id_of(done_blk);
        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        let cur_idx = self.emit_with_result(
            header_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let in_bounds = self.emit_with_result(
            header_blk,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: len_value,
            },
        );
        self.emit(
            header_blk,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: body_id,
                then_args: vec![],
                else_target: done_id,
                else_args: vec![],
            }),
        );

        let cur_idx_body = self.emit_with_result(
            body_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(body_blk, 1);
        let source_slot = self.emit_with_result(
            body_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx_body,
                rhs: one,
            },
        );
        let elem = self.load_at_dynamic_offset(body_blk, source_ptr, source_slot);
        let mut elem_bit = self.emit_i64_const(body_blk, 0);
        for (idx, key) in domain_keys.iter().copied().enumerate() {
            let key_value = self.emit_i64_const(body_blk, key);
            let key_match = self.emit_with_result(
                body_blk,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: elem,
                    rhs: key_value,
                },
            );
            let bit_value = self.emit_i64_const(body_blk, 1_i64 << idx);
            let selected_bit = self.emit_with_result(
                body_blk,
                Inst::Select {
                    ty: Ty::I64,
                    cond: key_match,
                    then_val: bit_value,
                    else_val: zero,
                },
            );
            elem_bit = self.emit_with_result(
                body_blk,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: elem_bit,
                    rhs: selected_bit,
                },
            );
        }
        let present = self.emit_with_result(
            body_blk,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: elem_bit,
                rhs: zero,
            },
        );
        self.emit(
            body_blk,
            InstrNode::new(Inst::CondBr {
                cond: present,
                then_target: accept_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );

        let old_mask = self.emit_with_result(
            accept_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: mask_alloca,
                align: None,
                volatile: false,
            },
        );
        let new_mask = self.emit_with_result(
            accept_blk,
            Inst::BinOp {
                op: BinOp::Or,
                ty: Ty::I64,
                lhs: old_mask,
                rhs: elem_bit,
            },
        );
        self.emit(
            accept_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: mask_alloca,
                value: new_mask,
                align: None,
                volatile: false,
            }),
        );
        let next_idx = self.emit_with_result(
            accept_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx_body,
                rhs: one,
            },
        );
        self.emit(
            accept_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            accept_blk,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);

        let final_mask = self.emit_with_result(
            done_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: mask_alloca,
                align: None,
                volatile: false,
            },
        );
        let result_ptr = self.alloc_aggregate(done_blk, 1);
        self.store_at_offset(done_blk, result_ptr, 0, final_mask);
        Ok(Some(super::CompactMaterializationResult {
            slot: super::CompactStateSlot::raw(result_ptr, 0),
            block_idx: done_blk,
        }))
    }

    fn compact_tagged_scalar_or_set_replacement_source(
        &mut self,
        block_idx: usize,
        reg: u8,
        expected_shape: &super::AggregateShape,
    ) -> Result<Option<super::CompactMaterializationResult>, TrustIrError> {
        let super::AggregateShape::TaggedScalarOrSet {
            scalar,
            universe_len,
            universe,
            ..
        } = expected_shape
        else {
            return Ok(None);
        };
        let Some(source_shape) = self.aggregate_shapes.get(&reg).cloned() else {
            return Ok(None);
        };
        let exact_universe_values =
            Self::tagged_scalar_or_set_exact_universe_values(scalar, *universe_len, universe)?;

        let (block_idx, tagged_value) = match &source_shape {
            super::AggregateShape::TaggedScalarOrSet { .. } if source_shape == *expected_shape => {
                (block_idx, self.load_reg(block_idx, reg)?)
            }
            super::AggregateShape::Scalar(source_scalar)
                if Self::tagged_scalar_source_compatible(source_scalar, scalar) =>
            {
                if matches!(source_scalar, super::ScalarShape::Int)
                    || matches!(scalar, super::ScalarShape::Int)
                {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "FuncExcept compact tagged scalar-or-set replacement requires a nonnegative scalar proof for Int values"
                            .to_owned(),
                    ));
                }
                // Scalar lane contract (`TaggedScalarSetRangeProof` /
                // `encode_tagged_scalar_set_scalar`): ANY nonnegative scalar
                // of the slot's scalar sort is a legal scalar-lane value —
                // the SET universe restricts only the negative tagged-mask
                // lane. Do NOT guard the scalar against the set universe
                // (e.g. Dijkstra `temp[p] := defaultInitValue` writes a
                // scalar OUTSIDE `Proc`; TypeOK — the checked invariant —
                // is the semantic wall, not an encode-time trap; a
                // universe-membership guard here spuriously kills the whole
                // native tier at runtime). Keep a fail-closed nonnegativity
                // backstop so a mis-shaped source can never alias the tagged
                // set-mask sign convention. String↔ModelValue cross-sort
                // sources are admitted by `tagged_scalar_source_compatible`
                // because the ABI carrier already erases that distinction
                // for scalar slots (both bridge to `CompoundLayout::String`
                // and share the interned-NameId payload).
                let value = self.load_reg(block_idx, reg)?;
                self.guard_tagged_scalar_replacement_nonnegative(block_idx, value)?
            }
            super::AggregateShape::SetBitmask {
                universe_len: source_universe_len,
                universe: source_universe,
            } => {
                if *source_universe_len != *universe_len || source_universe != universe {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "FuncExcept compact tagged scalar-or-set replacement universe mismatch: source={source_shape:?}, expected={expected_shape:?}"
                    )));
                }
                let mask = self.load_reg(block_idx, reg)?;
                let neg_one = self.emit_i64_const(block_idx, -1);
                (
                    block_idx,
                    self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::I64,
                            lhs: neg_one,
                            rhs: mask,
                        },
                    ),
                )
            }
            super::AggregateShape::ExactScalarSet {
                scalar: source_scalar,
                values,
            } if Self::tagged_scalar_source_compatible(source_scalar, scalar) => {
                let Some(universe_values) = exact_universe_values.as_deref() else {
                    return Ok(None);
                };
                let mut mask = 0_i64;
                for value in values {
                    let Some(idx) = universe_values
                        .iter()
                        .position(|universe_value| universe_value == value)
                    else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "FuncExcept compact tagged scalar-or-set exact-set replacement requires all values inside the destination universe, got source={source_shape:?}, expected={expected_shape:?}"
                        )));
                    };
                    mask |= 1_i64 << idx;
                }
                let neg_one = self.emit_i64_const(block_idx, -1);
                let mask = self.emit_i64_const(block_idx, mask);
                (
                    block_idx,
                    self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::I64,
                            lhs: neg_one,
                            rhs: mask,
                        },
                    ),
                )
            }
            super::AggregateShape::Set { len: 0, .. } => {
                (block_idx, self.emit_i64_const(block_idx, -1))
            }
            super::AggregateShape::Set {
                len,
                element: Some(element),
            }
            | super::AggregateShape::BoundedSet {
                max_len: len,
                element: Some(element),
            } if Self::tagged_set_element_source_compatible(element, scalar) => {
                let Some(universe_values) = exact_universe_values.as_deref() else {
                    return Ok(None);
                };
                self.materialized_set_as_tagged_scalar_or_set_value(
                    block_idx,
                    reg,
                    *len,
                    universe_values,
                )?
            }
            _ => return Ok(None),
        };

        let result_ptr = self.alloc_aggregate(block_idx, 1);
        self.store_at_offset(block_idx, result_ptr, 0, tagged_value);
        Ok(Some(super::CompactMaterializationResult {
            slot: super::CompactStateSlot::raw(result_ptr, 0),
            block_idx,
        }))
    }

    fn tagged_scalar_source_compatible(
        source: &super::ScalarShape,
        expected: &super::ScalarShape,
    ) -> bool {
        source == expected
            || matches!(
                (source, expected),
                (
                    super::ScalarShape::String | super::ScalarShape::ModelValue,
                    super::ScalarShape::String | super::ScalarShape::ModelValue
                )
            )
    }

    fn tagged_set_element_source_compatible(
        source: &super::AggregateShape,
        expected: &super::ScalarShape,
    ) -> bool {
        matches!(
            source,
            super::AggregateShape::Scalar(source)
                if Self::tagged_scalar_source_compatible(source, expected)
        )
    }

    fn tagged_scalar_or_set_exact_universe_values(
        scalar: &super::ScalarShape,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
    ) -> Result<Option<Vec<i64>>, TrustIrError> {
        let super::SetBitmaskUniverse::Exact(elements) = universe else {
            return Ok(None);
        };
        if usize::try_from(universe_len).ok() != Some(elements.len()) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept compact tagged scalar-or-set replacement universe length mismatch: universe_len={universe_len}, elements={}",
                elements.len()
            )));
        }
        Self::compact_set_bitmask_valid_mask(
            universe_len,
            "FuncExcept compact tagged scalar-or-set replacement",
        )?;
        let mut values = Vec::with_capacity(elements.len());
        for element in elements {
            if !Self::tagged_universe_element_matches_scalar(scalar, element) {
                return Ok(None);
            }
            let value = Self::scalar_domain_element_to_compact_value(element).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept compact tagged scalar-or-set replacement has non-scalar universe element {element:?}"
                ))
            })?;
            values.push(value);
        }
        Ok(Some(values))
    }

    fn tagged_universe_element_matches_scalar(
        scalar: &super::ScalarShape,
        element: &SetBitmaskElement,
    ) -> bool {
        matches!(
            (scalar, element),
            (super::ScalarShape::Int, SetBitmaskElement::Int(_))
                | (super::ScalarShape::Bool, SetBitmaskElement::Bool(_))
                | (super::ScalarShape::String, SetBitmaskElement::String(_))
                | (
                    super::ScalarShape::ModelValue,
                    SetBitmaskElement::ModelValue(_)
                )
        )
    }

    /// Fail-closed injectivity backstop for a scalar write into a tagged
    /// scalar-or-set slot: the tagged encoding stores scalars as their raw
    /// nonnegative payload and sets as `-1 - mask`, so a NEGATIVE "scalar"
    /// would silently alias a set mask. Sort/shape analysis already proves
    /// the source is a String/ModelValue/Bool scalar (nonnegative by
    /// construction); this runtime guard turns any upstream shape bug into a
    /// recoverable runtime error instead of a wrong verdict. Intentionally
    /// does NOT restrict the scalar to the slot's SET universe — the scalar
    /// lane admits any scalar of the sort (`encode_tagged_scalar_set_scalar`),
    /// and out-of-`TypeOK` writes must be caught by the CHECKED invariant,
    /// not an encode-time wall (H6).
    fn guard_tagged_scalar_replacement_nonnegative(
        &mut self,
        block_idx: usize,
        value: ValueId,
    ) -> Result<(usize, ValueId), TrustIrError> {
        let zero = self.emit_i64_const(block_idx, 0);
        let nonnegative = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: value,
                rhs: zero,
            },
        );
        let accept_blk = self.new_aux_block("compact_tagged_scalar_replacement_accept");
        let error_blk = self.new_aux_block("compact_tagged_scalar_replacement_error");
        let accept_id = self.block_id_of(accept_blk);
        let error_id = self.block_id_of(error_blk);
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: nonnegative,
                then_target: accept_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );
        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);
        Ok((accept_blk, value))
    }

    /// WP-18 follow-on (union-keyed apply/except soundness): decode a
    /// tagged-scalar-union INDEX into the member's raw comparison value.
    ///
    /// A register whose tracked shape is `TaggedScalarUnion` physically holds
    /// the union-slot INDEX (0..universe_len), not the member value. Every
    /// compact function/sequence key consumer works in RAW MEMBER space (ints
    /// as themselves, String/ModelValue as their interned ids —
    /// `domain_key_raw_value`'s space), so consuming the index directly
    /// aliases into the wrong row: `f[focus]` with `focus = node n` (index
    /// `n-1`) read/wrote node `n-1`'s row — the btree AddToLeaf/UpdateLeaf
    /// divergence the hybrid shadow differential caught (mismatch_fallback,
    /// plus lo-bound `native_errors` for index 0).
    ///
    /// The decode is a branch-free compare-fold over the compile-time-known
    /// universe: `index == i` selects member `i`'s raw value. The fold's
    /// initial accumulator is a sentinel far below any real member raw, so an
    /// impossible out-of-range index routes to the caller's existing
    /// bounds-guard not-found / identity path (fail closed, matching the
    /// interpreter's out-of-domain behavior).
    ///
    /// Returns `raw` unchanged for every non-union-shaped register.
    pub(super) fn decode_scalar_key_reg_raw_value(
        &mut self,
        block_idx: usize,
        reg: u8,
        raw: ValueId,
    ) -> ValueId {
        let Some(super::AggregateShape::TaggedScalarUnion { universe, .. }) =
            self.aggregate_shapes.get(&reg)
        else {
            return raw;
        };
        let members: Vec<i64> = universe.iter().map(Self::domain_key_raw_value).collect();
        self.emit_tagged_scalar_union_index_decode(block_idx, raw, &members)
    }

    /// Emission body of the union-index decode compare-fold; `members` are the
    /// universe's raw comparison values in index order.
    pub(super) fn emit_tagged_scalar_union_index_decode(
        &mut self,
        block_idx: usize,
        index_val: ValueId,
        members: &[i64],
    ) -> ValueId {
        // Far outside any int domain and any interned id; never equal to a
        // domain key, so every consumer's compare/bounds guard misses.
        let mut acc = self.emit_i64_const(block_idx, i64::MIN / 2);
        for (index, member_raw) in members.iter().enumerate() {
            let index_const =
                self.emit_i64_const(block_idx, i64::try_from(index).expect("universe fits i64"));
            let is_index = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: index_val,
                    rhs: index_const,
                },
            );
            let member_val = self.emit_i64_const(block_idx, *member_raw);
            acc = self.emit_with_result(
                block_idx,
                Inst::Select {
                    ty: Ty::I64,
                    cond: is_index,
                    then_val: member_val,
                    else_val: acc,
                },
            );
        }
        acc
    }

    /// Per-position union-index decode for a RUNTIME tuple key: element `j`
    /// (1-based tuple position, `j-1` in `tuple_element_shapes` order) is
    /// decoded IFF its recorded construction shape is a tagged scalar union.
    /// Positions without a recorded shape keep the raw slot value (today's
    /// behavior for shape-less tuples).
    fn decode_tuple_key_elem_raw_value(
        &mut self,
        block_idx: usize,
        tuple_reg: u8,
        position: usize,
        raw: ValueId,
    ) -> ValueId {
        let Some(shapes) = self.tuple_element_shapes.get(&tuple_reg) else {
            return raw;
        };
        let Some(super::AggregateShape::TaggedScalarUnion { universe, .. }) =
            shapes.get(position)
        else {
            return raw;
        };
        let members: Vec<i64> = universe.iter().map(Self::domain_key_raw_value).collect();
        self.emit_tagged_scalar_union_index_decode(block_idx, raw, &members)
    }

    /// FuncExcept replacement source for a [`AggregateShape::TaggedScalarUnion`]
    /// range value (the scalar-union sibling of
    /// [`Self::compact_tagged_scalar_or_set_replacement_source`]). Returns
    /// `Ok(None)` for any non-union destination so it composes with the other
    /// replacement strategies; a union destination whose source cannot be
    /// encoded fails closed with a precise error (there is no other legal
    /// strategy for an index slot).
    pub(super) fn compact_tagged_scalar_union_replacement_source(
        &mut self,
        block_idx: usize,
        reg: u8,
        expected_shape: &super::AggregateShape,
    ) -> Result<Option<super::CompactMaterializationResult>, TrustIrError> {
        let super::AggregateShape::TaggedScalarUnion {
            universe,
            int_arm,
            proof_source,
        } = expected_shape
        else {
            return Ok(None);
        };
        let (block_idx, index_value) = self.encode_tagged_scalar_union_index(
            block_idx,
            reg,
            universe,
            *int_arm,
            *proof_source,
            "FuncExcept compact tagged scalar-union replacement",
        )?;
        let result_ptr = self.alloc_aggregate(block_idx, 1);
        self.store_at_offset(block_idx, result_ptr, 0, index_value);
        Ok(Some(super::CompactMaterializationResult {
            slot: super::CompactStateSlot::raw(result_ptr, 0),
            block_idx,
        }))
    }

    /// Encode the value in `reg` into its universe INDEX for a
    /// [`AggregateShape::TaggedScalarUnion`] destination, arm-aware and always
    /// fail-closed on an unencodable source:
    ///
    /// 1. identical-universe union source -> passthrough (raw index copy);
    /// 2. compile-time const scalar that is a universe member -> const index
    ///    (member outside the universe -> `UnsupportedOpcode`);
    /// 3. runtime `Scalar(Int)` with a contiguous int arm -> a fail-closed
    ///    `lo <= v <= hi` range guard (typed `TypeMismatch` runtime error on a
    ///    miss, never a wrong slot) then `(v - lo) + base`;
    /// 4. anything else (non-const String/ModelValue, mismatched universe, Int
    ///    without an int arm, untracked shape) -> `UnsupportedOpcode`.
    pub(super) fn encode_tagged_scalar_union_index(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe: &[SetBitmaskElement],
        int_arm: Option<super::TaggedUnionIntArm>,
        _proof_source: tla_core::NameId,
        context: &str,
    ) -> Result<(usize, ValueId), TrustIrError> {
        // (1) identical-universe union source: the slot already holds the index.
        if let Some(super::AggregateShape::TaggedScalarUnion {
            universe: source_universe,
            ..
        }) = self.aggregate_shapes.get(&reg)
        {
            if source_universe.as_slice() == universe {
                let value = self.load_reg(block_idx, reg)?;
                return Ok((block_idx, value));
            }
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: source r{reg} tagged-scalar-union universe does not match the destination universe"
            )));
        }

        // (2) compile-time const scalar member -> const universe index.
        if let Some(key) = self.const_scalar_domain_key_of(reg) {
            return match universe.iter().position(|element| *element == key) {
                Some(index) => {
                    let index = i64::try_from(index).map_err(|_| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "{context}: universe index for r{reg} overflows i64"
                        ))
                    })?;
                    Ok((block_idx, self.emit_i64_const(block_idx, index)))
                }
                None => Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: constant source r{reg} value {key:?} is outside the union universe"
                ))),
            };
        }

        // (3) runtime Scalar(Int) — or a `ScalarIntDomain` whose declared finite
        // integer domain is PROVEN inside the universe's int arm (WP-32) — with
        // a contiguous int arm -> guarded (v-lo)+base.
        //
        // WP-32: `ScalarIntDomain` is a raw Int lane (`compatible_flat_aggregate_value`
        // already treats it as bit-identical to `Scalar(Int)`) that additionally
        // carries a finite declared domain. btree's `SplitRootLeaf` /
        // `SplitRootInner` write `childOf' = [childOf EXCEPT ![newRoot, pivot] = n1]`
        // where `n1` came out of `Head(toSplit)` / `ChooseFreeNode` as
        // `ScalarIntDomain { universe_len: 8, universe: IntRange { lo: 1 } }` —
        // literally `Nodes` — into the `Nodes \cup {NIL}` union range. Both
        // proof obligations are discharged statically here: the SORT is Int (so
        // the raw payload cannot alias an interned member's `NameId`, the H5
        // hazard that keeps arm (3b) sort-separated), and MEMBERSHIP is proven
        // by containment of the declared domain in `[arm.lo, arm.hi]`. Only
        // integer-sorted universes (`IntRange` / `ExplicitInt`) are admitted;
        // an `Exact` / `Unknown` universe cannot prove either obligation and
        // keeps failing closed. The runtime range guard below is emitted
        // regardless — statically dead for this arm, but it means an unsound
        // domain claim degrades to a typed `TypeMismatch` interpreter fallback
        // instead of a wrong index.
        let scalar_int_domain_in_arm = match (self.aggregate_shapes.get(&reg), int_arm) {
            (
                Some(super::AggregateShape::ScalarIntDomain {
                    universe_len,
                    universe,
                }),
                Some(arm),
            ) => Self::int_domain_members_within(*universe_len, universe, arm.lo, arm.hi),
            _ => false,
        };
        if matches!(self.scalar_shape_of(reg), Some(super::ScalarShape::Int))
            || scalar_int_domain_in_arm
        {
            let Some(arm) = int_arm else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: runtime integer source r{reg} requires a contiguous ascending Int arm in the union universe"
                )));
            };
            let value = self.load_reg(block_idx, reg)?;
            let block_idx =
                self.guard_tagged_scalar_union_int_in_range(block_idx, value, arm.lo, arm.hi)?;
            let lo_val = self.emit_i64_const(block_idx, arm.lo);
            let shifted = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: value,
                    rhs: lo_val,
                },
            );
            let base_val = self.emit_i64_const(block_idx, i64::from(arm.base));
            let index = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: shifted,
                    rhs: base_val,
                },
            );
            return Ok((block_idx, index));
        }

        // (3b) runtime INTERNED scalar (String/ModelValue sort) — e.g. btree
        // UpdateLeaf's `val` (a model value read out of the `args` tuple)
        // written into valOf's `Vals ∪ {NIL}` union range. The register's raw
        // payload is the interned `NameId`; fold a per-member equality chain
        // over exactly the universe's OWN interned (String/ModelValue)
        // members, Select-accumulating that member's index. `Int`/`Bool`
        // members are excluded from the compare set: the source is
        // interned-sorted, so comparing its `NameId` against an integer raw
        // lane could alias (H5). If the universe's interned members are not
        // pairwise raw-distinct (a pathological String/ModelValue text
        // collision), fail closed — the compare could not distinguish them. A
        // runtime value matching no member takes the typed `TypeMismatch`
        // runtime error (recoverable interpreter fallback) — never a wrong
        // index. This mirrors the read half's raw-compare discipline
        // (`validated_tuple_key_table`), which treats String/ModelValue as one
        // interned sort against `CompoundLayout::String` positions.
        if matches!(
            self.scalar_shape_of(reg),
            Some(super::ScalarShape::String | super::ScalarShape::ModelValue)
        ) {
            let interned_members: Vec<(usize, i64)> = universe
                .iter()
                .enumerate()
                .filter_map(|(index, element)| match element {
                    SetBitmaskElement::String(name) | SetBitmaskElement::ModelValue(name) => {
                        Some((index, i64::from(name.0)))
                    }
                    SetBitmaskElement::Int(_) | SetBitmaskElement::Bool(_) => None,
                })
                .collect();
            if interned_members.is_empty() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: runtime interned source r{reg} has no interned member in the union universe"
                )));
            }
            for (position, (_, raw)) in interned_members.iter().enumerate() {
                if interned_members[position + 1..]
                    .iter()
                    .any(|(_, other)| other == raw)
                {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: union universe interned members are not raw-distinct; a runtime compare cannot select a unique index"
                    )));
                }
            }
            let value = self.load_reg(block_idx, reg)?;
            let mut matched: Option<ValueId> = None;
            let mut index_val: Option<ValueId> = None;
            for (index, raw) in &interned_members {
                let member_const = self.emit_i64_const(block_idx, *raw);
                let member_eq = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: value,
                        rhs: member_const,
                    },
                );
                let index_const = i64::try_from(*index).map_err(|_| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: universe index for r{reg} overflows i64"
                    ))
                })?;
                let index_const = self.emit_i64_const(block_idx, index_const);
                index_val = Some(match index_val {
                    // Seed with the first member's index; the membership guard
                    // below rejects a no-match value before the seed could
                    // ever be stored.
                    None => index_const,
                    Some(prev) => self.emit_with_result(
                        block_idx,
                        Inst::Select {
                            ty: Ty::I64,
                            cond: member_eq,
                            then_val: index_const,
                            else_val: prev,
                        },
                    ),
                });
                matched = Some(match matched {
                    None => member_eq,
                    // `member_eq` operands are ICmp results (Bool); the OR-fold
                    // must be Bool-typed, mirroring the tuple-key AND-fold
                    // convention.
                    Some(prev) => self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::Or,
                            ty: Ty::Bool,
                            lhs: prev,
                            rhs: member_eq,
                        },
                    ),
                });
            }
            let matched = matched.expect("non-empty interned member set");
            let index_val = index_val.expect("non-empty interned member set");
            let error_blk = self.new_aux_block("tagged_scalar_union_member_error");
            let accept_blk = self.new_aux_block("tagged_scalar_union_member_accept");
            let error_id = self.block_id_of(error_blk);
            let accept_id = self.block_id_of(accept_blk);
            self.emit(
                block_idx,
                InstrNode::new(Inst::CondBr {
                    cond: matched,
                    then_target: accept_id,
                    then_args: vec![],
                    else_target: error_id,
                    else_args: vec![],
                }),
            );
            self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);
            return Ok((accept_blk, index_val));
        }

        // (4) ambiguous raw source (untracked shape): the raw payload would
        // alias a foreign universe index. Fail closed.
        Err(TrustIrError::UnsupportedOpcode(format!(
            "{context}: source r{reg} with shape {:?} cannot be encoded into a tagged-scalar-union slot",
            self.aggregate_shapes.get(&reg)
        )))
    }

    /// Fail-closed `lo <= value <= hi` range guard for a runtime integer being
    /// encoded into a tagged-scalar-union int arm. On an out-of-range value the
    /// action returns a typed `TypeMismatch` runtime error (recoverable
    /// interpreter fallback) rather than storing a wrong universe index. Returns
    /// the accept block to continue in.
    pub(super) fn guard_tagged_scalar_union_int_in_range(
        &mut self,
        block_idx: usize,
        value: ValueId,
        lo: i64,
        hi: i64,
    ) -> Result<usize, TrustIrError> {
        let error_blk = self.new_aux_block("tagged_scalar_union_range_error");
        let error_id = self.block_id_of(error_blk);

        let lo_const = self.emit_i64_const(block_idx, lo);
        let ge_lo = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: value,
                rhs: lo_const,
            },
        );
        let check_hi_blk = self.new_aux_block("tagged_scalar_union_range_check_hi");
        let check_hi_id = self.block_id_of(check_hi_blk);
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: ge_lo,
                then_target: check_hi_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );

        let hi_const = self.emit_i64_const(check_hi_blk, hi);
        let le_hi = self.emit_with_result(
            check_hi_blk,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: value,
                rhs: hi_const,
            },
        );
        let accept_blk = self.new_aux_block("tagged_scalar_union_range_accept");
        let accept_id = self.block_id_of(accept_blk);
        self.emit(
            check_hi_blk,
            InstrNode::new(Inst::CondBr {
                cond: le_hi,
                then_target: accept_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );
        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);
        Ok(accept_blk)
    }

    fn materialized_set_as_tagged_scalar_or_set_value(
        &mut self,
        block_idx: usize,
        reg: u8,
        capacity: u32,
        universe_values: &[i64],
    ) -> Result<(usize, ValueId), TrustIrError> {
        if universe_values.is_empty() {
            return Err(TrustIrError::UnsupportedOpcode(
                "FuncExcept compact tagged scalar-or-set set replacement requires a nonempty exact universe"
                    .to_owned(),
            ));
        }
        let source_ptr = self.load_reg_as_ptr(block_idx, reg)?;
        let len_value = self.load_at_offset(block_idx, source_ptr, 0);
        let guard: super::CompactSequenceLenGuardResult = self
            .guard_compact_sequence_len_in_bounds(
                block_idx,
                len_value,
                capacity,
                "compact_tagged_scalar_or_set_replacement",
            );
        let block_idx = guard.block_idx;
        let len_value = guard.len_value;

        let zero = self.emit_i64_const(block_idx, 0);
        let idx_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        let mask_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: mask_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let header_blk = self.new_aux_block("compact_tagged_set_replacement_header");
        let body_blk = self.new_aux_block("compact_tagged_set_replacement_body");
        let accept_blk = self.new_aux_block("compact_tagged_set_replacement_accept");
        let error_blk = self.new_aux_block("compact_tagged_set_replacement_error");
        let done_blk = self.new_aux_block("compact_tagged_set_replacement_done");
        let header_id = self.block_id_of(header_blk);
        let body_id = self.block_id_of(body_blk);
        let accept_id = self.block_id_of(accept_blk);
        let error_id = self.block_id_of(error_blk);
        let done_id = self.block_id_of(done_blk);
        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        let cur_idx = self.emit_with_result(
            header_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let in_bounds = self.emit_with_result(
            header_blk,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: len_value,
            },
        );
        self.emit(
            header_blk,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: body_id,
                then_args: vec![],
                else_target: done_id,
                else_args: vec![],
            }),
        );

        let cur_idx_body = self.emit_with_result(
            body_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(body_blk, 1);
        let source_slot = self.emit_with_result(
            body_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx_body,
                rhs: one,
            },
        );
        let elem = self.load_at_dynamic_offset(body_blk, source_ptr, source_slot);
        let mut elem_bit = self.emit_i64_const(body_blk, 0);
        for (idx, value) in universe_values.iter().copied().enumerate() {
            let expected_value = self.emit_i64_const(body_blk, value);
            let value_match = self.emit_with_result(
                body_blk,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: elem,
                    rhs: expected_value,
                },
            );
            let bit_value = self.emit_i64_const(body_blk, 1_i64 << idx);
            let selected_bit = self.emit_with_result(
                body_blk,
                Inst::Select {
                    ty: Ty::I64,
                    cond: value_match,
                    then_val: bit_value,
                    else_val: zero,
                },
            );
            elem_bit = self.emit_with_result(
                body_blk,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: elem_bit,
                    rhs: selected_bit,
                },
            );
        }
        let present = self.emit_with_result(
            body_blk,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: elem_bit,
                rhs: zero,
            },
        );
        self.emit(
            body_blk,
            InstrNode::new(Inst::CondBr {
                cond: present,
                then_target: accept_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );

        let old_mask = self.emit_with_result(
            accept_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: mask_alloca,
                align: None,
                volatile: false,
            },
        );
        let new_mask = self.emit_with_result(
            accept_blk,
            Inst::BinOp {
                op: BinOp::Or,
                ty: Ty::I64,
                lhs: old_mask,
                rhs: elem_bit,
            },
        );
        self.emit(
            accept_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: mask_alloca,
                value: new_mask,
                align: None,
                volatile: false,
            }),
        );
        let next_idx = self.emit_with_result(
            accept_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx_body,
                rhs: one,
            },
        );
        self.emit(
            accept_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            accept_blk,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);

        let final_mask = self.emit_with_result(
            done_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: mask_alloca,
                align: None,
                volatile: false,
            },
        );
        let neg_one = self.emit_i64_const(done_blk, -1);
        let tagged_value = self.emit_with_result(
            done_blk,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: neg_one,
                rhs: final_mask,
            },
        );
        Ok((done_blk, tagged_value))
    }

    fn compact_value_source_for_reg(
        &mut self,
        block_idx: usize,
        reg: u8,
        expected_shape: &super::AggregateShape,
    ) -> Result<super::CompactMaterializationResult, TrustIrError> {
        self.materialize_reg_as_compact_source(block_idx, reg, expected_shape)
    }

    fn fold_sum_operand_is_function_like(shape: Option<&super::AggregateShape>) -> bool {
        matches!(
            shape,
            Some(super::AggregateShape::Function { .. } | super::AggregateShape::StateValue) | None
        )
    }

    fn fold_sum_should_swap_operands(
        first_shape: Option<&super::AggregateShape>,
        second_shape: Option<&super::AggregateShape>,
    ) -> bool {
        let first_is_set = first_shape.is_some_and(super::AggregateShape::is_finite_set_shape);
        let second_is_set = second_shape.is_some_and(super::AggregateShape::is_finite_set_shape);
        if second_is_set {
            return false;
        }
        if first_is_set && Self::fold_sum_operand_is_function_like(second_shape) {
            return true;
        }
        first_shape.is_none()
            && matches!(second_shape, Some(super::AggregateShape::Function { .. }))
    }

    fn contiguous_int_domain_lo(shape: Option<&super::AggregateShape>, len: u32) -> Option<i64> {
        super::dense_ordered_int_domain_lo(shape?, len)
    }

    fn resolve_fold_sum_set_shape(
        set_reg: u8,
        set_shape: Option<&super::AggregateShape>,
        func_shape: Option<&super::AggregateShape>,
    ) -> Result<Option<super::AggregateShape>, TrustIrError> {
        if let Some(shape) = set_shape {
            if !shape.is_finite_set_shape() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "FoldFunctionOnSetSum: set argument r{set_reg} must be a tracked finite set, got {shape:?}"
                )));
            }
        }

        let function_domain_shape =
            func_shape.and_then(super::AggregateShape::function_domain_shape);
        match (set_shape, function_domain_shape.as_ref()) {
            (Some(set_shape), Some(function_domain_shape)) => {
                if let (
                    super::AggregateShape::Interval {
                        lo: set_lo,
                        hi: set_hi,
                    },
                    super::AggregateShape::Interval {
                        lo: domain_lo,
                        hi: domain_hi,
                    },
                ) = (set_shape, function_domain_shape)
                {
                    let set_is_empty = *set_hi < *set_lo;
                    if !set_is_empty && (*set_lo < *domain_lo || *set_hi > *domain_hi) {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "FoldFunctionOnSetSum: set argument r{set_reg} is incompatible with function domain: set={set_shape:?}, domain={function_domain_shape:?}"
                        )));
                    }
                    return Ok(Some(set_shape.clone()));
                }

                super::merge_compatible_shapes(Some(set_shape), Some(function_domain_shape))
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "FoldFunctionOnSetSum: set argument r{set_reg} is incompatible with function domain: set={set_shape:?}, domain={function_domain_shape:?}"
                        ))
                    })
                    .map(Some)
            }
            (Some(set_shape), None) => Ok(Some(set_shape.clone())),
            (None, Some(function_domain_shape)) => Ok(Some(function_domain_shape.clone())),
            (None, None) => Ok(None),
        }
    }

    /// Lower FoldFunctionOnSet(+, 0, f, S): sum `f[x]` for every `x` in `S`.
    ///
    /// This builtin is deliberately narrow: TIR only emits it for the
    /// recognized `FoldFunctionOnSet(+, 0, f, S)` shape, and this lowering
    /// uses tracked finite shapes when they are available. Generic user
    /// operators such as `Sum(f, S)` may lower this builtin from inside an
    /// arity-positive callee where formal parameter shapes are unknown; those
    /// loops still lower with dynamic lengths and are marked unbounded.
    pub(super) fn lower_fold_function_on_set_sum(
        &mut self,
        block_idx: usize,
        rd: u8,
        mut func_reg: u8,
        mut set_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        let first_shape = self.aggregate_shapes.get(&func_reg).cloned();
        let second_shape = self.aggregate_shapes.get(&set_reg).cloned();
        let swapped_set_first =
            Self::fold_sum_should_swap_operands(first_shape.as_ref(), second_shape.as_ref());
        if swapped_set_first {
            std::mem::swap(&mut func_reg, &mut set_reg);
        }

        let set_shape = self.aggregate_shapes.get(&set_reg).cloned();
        let func_shape = self.aggregate_shapes.get(&func_reg).cloned();
        let resolved_set_shape =
            Self::resolve_fold_sum_set_shape(set_reg, set_shape.as_ref(), func_shape.as_ref())?;
        let set_exact_len = resolved_set_shape
            .as_ref()
            .and_then(super::AggregateShape::tracked_len);
        if let Some(shape) = resolved_set_shape.clone() {
            if let Some(len) = shape.tracked_len().or_else(|| shape.finite_set_len_bound()) {
                self.const_set_sizes.insert(set_reg, len);
            } else {
                self.const_set_sizes.remove(&set_reg);
            }
            self.aggregate_shapes.insert(set_reg, shape);
        } else {
            self.const_set_sizes.remove(&set_reg);
        }

        let value = match &func_shape {
            Some(super::AggregateShape::Function { value, .. }) => value.clone(),
            Some(super::AggregateShape::StateValue) | None => {
                let value = Some(Box::new(super::AggregateShape::Scalar(
                    super::ScalarShape::Int,
                )));
                if let Some(len) = set_exact_len {
                    self.const_set_sizes.insert(func_reg, len);
                    self.aggregate_shapes.insert(
                        func_reg,
                        super::AggregateShape::Function {
                            len,
                            domain_lo: None,
                            domain: None,
                            value: value.clone(),
                        },
                    );
                } else {
                    self.const_set_sizes.remove(&func_reg);
                    self.aggregate_shapes.remove(&func_reg);
                }
                value
            }
            Some(other) => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "FoldFunctionOnSetSum: first argument r{func_reg} must be a tracked finite function, got {other:?}"
                )));
            }
        };
        if let Some(value_shape) = value.as_deref() {
            if !value_shape.is_numeric_scalar_shape() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "FoldFunctionOnSetSum: function values must be Int, got {value_shape:?}"
                )));
            }
        }

        let func_ptr = self.load_reg_as_ptr_or_materialize_raw_compact(
            block_idx,
            func_reg,
            "FoldFunctionOnSetSum function",
        )?;
        let set_ptr = self.load_reg_as_ptr_or_materialize_raw_compact(
            block_idx,
            set_reg,
            "FoldFunctionOnSetSum set",
        )?;
        let set_len = self.load_at_offset(block_idx, set_ptr, 0);
        let pair_count = self.load_at_offset(block_idx, func_ptr, 0);

        let zero = self.emit_i64_const(block_idx, 0);

        let set_idx_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: set_idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let func_idx_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: func_idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let acc_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: acc_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let set_header = self.new_aux_block("fold_sum_set_header");
        let set_body = self.new_aux_block("fold_sum_set_body");
        let func_header = self.new_aux_block("fold_sum_func_header");
        let func_body = self.new_aux_block("fold_sum_func_body");
        let func_inc = self.new_aux_block("fold_sum_func_inc");
        let func_found = self.new_aux_block("fold_sum_func_found");
        let add_overflow = self.new_aux_block("fold_sum_overflow");
        let add_ok = self.new_aux_block("fold_sum_add_ok");
        let func_missing = self.new_aux_block("fold_sum_missing");
        let done = self.new_aux_block("fold_sum_done");

        let set_header_id = self.block_id_of(set_header);
        let set_body_id = self.block_id_of(set_body);
        let func_header_id = self.block_id_of(func_header);
        let func_body_id = self.block_id_of(func_body);
        let func_inc_id = self.block_id_of(func_inc);
        let func_found_id = self.block_id_of(func_found);
        let add_overflow_id = self.block_id_of(add_overflow);
        let add_ok_id = self.block_id_of(add_ok);
        let func_missing_id = self.block_id_of(func_missing);
        let done_id = self.block_id_of(done);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: set_header_id,
                args: vec![],
            }),
        );

        let set_idx = self.emit_with_result(
            set_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: set_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let set_in_bounds = self.emit_with_result(
            set_header,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: set_idx,
                rhs: set_len,
            },
        );
        self.emit(
            set_header,
            InstrNode::new(Inst::CondBr {
                cond: set_in_bounds,
                then_target: set_body_id,
                then_args: vec![],
                else_target: done_id,
                else_args: vec![],
            }),
        );
        if !self.annotate_loop_bound(set_header, set_reg) {
            self.mark_unbounded_loop();
        }

        let one = self.emit_i64_const(set_body, 1);
        let set_idx_for_load = self.emit_with_result(
            set_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: set_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let set_slot = self.emit_with_result(
            set_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: set_idx_for_load,
                rhs: one,
            },
        );
        let set_elem = self.load_at_dynamic_offset(set_body, set_ptr, set_slot);
        let zero_for_func_idx = self.emit_i64_const(set_body, 0);
        self.emit(
            set_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: func_idx_alloca,
                value: zero_for_func_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            set_body,
            InstrNode::new(Inst::Br {
                target: func_header_id,
                args: vec![],
            }),
        );

        let func_idx = self.emit_with_result(
            func_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: func_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let func_in_bounds = self.emit_with_result(
            func_header,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: func_idx,
                rhs: pair_count,
            },
        );
        self.emit(
            func_header,
            InstrNode::new(Inst::CondBr {
                cond: func_in_bounds,
                then_target: func_body_id,
                then_args: vec![],
                else_target: func_missing_id,
                else_args: vec![],
            }),
        );
        if !self.annotate_loop_bound(func_header, func_reg) {
            self.mark_unbounded_loop();
        }

        let two = self.emit_i64_const(func_body, 2);
        let one_for_key = self.emit_i64_const(func_body, 1);
        let func_idx_for_key = self.emit_with_result(
            func_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: func_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let key_offset = self.emit_with_result(
            func_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: func_idx_for_key,
                rhs: two,
            },
        );
        let key_slot = self.emit_with_result(
            func_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: key_offset,
                rhs: one_for_key,
            },
        );
        let key = self.load_at_dynamic_offset(func_body, func_ptr, key_slot);
        let key_matches = self.emit_with_result(
            func_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: key,
                rhs: set_elem,
            },
        );
        self.emit(
            func_body,
            InstrNode::new(Inst::CondBr {
                cond: key_matches,
                then_target: func_found_id,
                then_args: vec![],
                else_target: func_inc_id,
                else_args: vec![],
            }),
        );

        let func_idx_for_inc = self.emit_with_result(
            func_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: func_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one_for_inc = self.emit_i64_const(func_inc, 1);
        let next_func_idx = self.emit_with_result(
            func_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: func_idx_for_inc,
                rhs: one_for_inc,
            },
        );
        self.emit(
            func_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: func_idx_alloca,
                value: next_func_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            func_inc,
            InstrNode::new(Inst::Br {
                target: func_header_id,
                args: vec![],
            }),
        );

        let two_for_value = self.emit_i64_const(func_found, 2);
        let func_idx_for_value = self.emit_with_result(
            func_found,
            Inst::Load {
                ty: Ty::I64,
                ptr: func_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let value_offset = self.emit_with_result(
            func_found,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: func_idx_for_value,
                rhs: two_for_value,
            },
        );
        let value_slot = self.emit_with_result(
            func_found,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: value_offset,
                rhs: two_for_value,
            },
        );
        let func_value = self.load_at_dynamic_offset(func_found, func_ptr, value_slot);
        let acc = self.emit_with_result(
            func_found,
            Inst::Load {
                ty: Ty::I64,
                ptr: acc_alloca,
                align: None,
                volatile: false,
            },
        );
        let add_result = self.alloc_value();
        let overflow_flag = self.alloc_value();
        self.emit(
            func_found,
            InstrNode::new(Inst::Overflow {
                op: OverflowOp::AddOverflow,
                ty: Ty::I64,
                lhs: acc,
                rhs: func_value,
            })
            .with_result(add_result)
            .with_result(overflow_flag),
        );
        self.emit(
            func_found,
            InstrNode::new(Inst::CondBr {
                cond: overflow_flag,
                then_target: add_overflow_id,
                then_args: vec![],
                else_target: add_ok_id,
                else_args: vec![],
            }),
        );

        self.emit_runtime_error_and_return(add_overflow, JitRuntimeErrorKind::ArithmeticOverflow);
        self.emit_runtime_error_and_return(func_missing, JitRuntimeErrorKind::TypeMismatch);

        self.emit(
            add_ok,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: acc_alloca,
                value: add_result,
                align: None,
                volatile: false,
            }),
        );
        let set_idx_for_inc = self.emit_with_result(
            add_ok,
            Inst::Load {
                ty: Ty::I64,
                ptr: set_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one_for_set_inc = self.emit_i64_const(add_ok, 1);
        let next_set_idx = self.emit_with_result(
            add_ok,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: set_idx_for_inc,
                rhs: one_for_set_inc,
            },
        );
        self.emit(
            add_ok,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: set_idx_alloca,
                value: next_set_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            add_ok,
            InstrNode::new(Inst::Br {
                target: set_header_id,
                args: vec![],
            }),
        );

        let result = self.emit_with_result(
            done,
            Inst::Load {
                ty: Ty::I64,
                ptr: acc_alloca,
                align: None,
                volatile: false,
            },
        );
        self.store_reg_value(done, rd, result)?;
        self.compact_state_slots.remove(&rd);
        self.aggregate_shapes
            .insert(rd, super::AggregateShape::Scalar(super::ScalarShape::Int));
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);

        Ok(Some(done))
    }

    /// WP-ARGS read side: lower `u[i]` where `u` is a
    /// [`super::AggregateShape::TaggedUnion`] carrier — btree's `args[1]` /
    /// `args[2]`.
    ///
    /// The payload slots carry no self-describing type, so the live arm must be
    /// PROVEN before any slot is read. This emits a **tag guard**: the loaded
    /// tag is compared against exactly the arms that have position `i`, and any
    /// other tag takes the typed `TypeMismatch` runtime-error branch (per-state
    /// interpreter fallback), never a neighbouring arm's slot. A read of
    /// `args[2]` on a `<<key>>` state therefore falls back — it never returns
    /// `<<key>>`'s only element or a stale zero.
    ///
    /// Everything not statically provable fails closed with `UnsupportedOpcode`
    /// so the whole action routes to the interpreter:
    ///   * a non-constant index (the admissible arm set would be unknown);
    ///   * no arm having that position at all;
    ///   * arms disagreeing on that position's layout (one decode cannot serve
    ///     both);
    ///   * any position wider than one slot (the payload offset of position `i`
    ///     is only `i` when every earlier position is exactly one slot).
    ///
    /// Returns `Ok(None)` when `func_reg` is not a union, so the caller falls
    /// through to the ordinary record/function/sequence paths.
    fn lower_func_apply_tagged_union(
        &mut self,
        block_idx: usize,
        rd: u8,
        func_reg: u8,
        arg_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        let Some(super::AggregateShape::TaggedUnion { variants, .. }) =
            self.aggregate_shapes.get(&func_reg).cloned()
        else {
            return Ok(None);
        };
        let Some(raw_index) = self.scalar_of(arg_reg) else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: tagged-union r{func_reg} requires a compile-time constant position, r{arg_reg} is not constant"
            )));
        };
        // TLA+ positions are 1-indexed.
        let position = usize::try_from(raw_index - 1).map_err(|_| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: tagged-union r{func_reg} position {raw_index} is not a valid 1-indexed tuple position"
            ))
        })?;
        let Some(source_slot) = self.compact_state_slot_for_use(block_idx, func_reg)? else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: tagged-union r{func_reg} requires a compact state slot to read the tag from"
            )));
        };

        let mut admissible_tags: Vec<i64> = Vec::new();
        let mut result_shape: Option<super::AggregateShape> = None;
        for (tag, variant) in variants.iter().enumerate() {
            let super::AggregateShape::Tuple { elements } = variant else {
                // A scalar sentinel arm has no positions at all.
                continue;
            };
            let Some(dest) = elements.get(position) else {
                continue;
            };
            if elements
                .iter()
                .any(|element| element.compact_slot_count() != Some(1))
            {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "FuncApply: tagged-union r{func_reg} arm {tag} has a multi-slot position, so the offset of position {position} is not statically known"
                )));
            }
            match &result_shape {
                None => result_shape = Some(dest.clone()),
                Some(existing) if existing == dest => {}
                Some(existing) => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "FuncApply: tagged-union r{func_reg} arms disagree on position {position}: {existing:?} vs {dest:?}"
                    )));
                }
            }
            admissible_tags.push(i64::try_from(tag).map_err(|_| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncApply: tagged-union r{func_reg} tag {tag} overflows i64"
                ))
            })?);
        }
        let Some(result_shape) = result_shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncApply: tagged-union r{func_reg} has no arm carrying position {position}"
            )));
        };

        let found_blk = self.new_aux_block("tagged_union_apply_found");
        let wrong_arm_blk = self.new_aux_block("tagged_union_apply_wrong_arm");
        let merge_blk = self.new_aux_block("tagged_union_apply_merge");
        let found_id = self.block_id_of(found_blk);
        let merge_id = self.block_id_of(merge_blk);

        // Tag guard chain: `tag == t0 || tag == t1 || ..`, otherwise the
        // runtime-error branch. The tag is re-loaded per block so no SSA value
        // crosses a block boundary.
        let mut current = block_idx;
        for (index, tag) in admissible_tags.iter().enumerate() {
            let tag_value =
                self.load_at_offset(current, source_slot.source_ptr, source_slot.offset);
            let tag_const = self.emit_i64_const(current, *tag);
            let is_arm = self.emit_with_result(
                current,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: tag_value,
                    rhs: tag_const,
                },
            );
            let next_blk = if index + 1 < admissible_tags.len() {
                self.new_aux_block("tagged_union_apply_next_arm")
            } else {
                wrong_arm_blk
            };
            let next_id = self.block_id_of(next_blk);
            self.emit(
                current,
                InstrNode::new(Inst::CondBr {
                    cond: is_arm,
                    then_target: found_id,
                    then_args: vec![],
                    else_target: next_id,
                    else_args: vec![],
                }),
            );
            current = next_blk;
        }

        // Payload slot: tag slot + `position` single-slot predecessors.
        let payload_offset = u32::try_from(position)
            .ok()
            .and_then(|position| position.checked_add(1))
            .and_then(|delta| source_slot.offset.checked_add(delta))
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncApply: tagged-union r{func_reg} payload slot for position {position} overflows"
                ))
            })?;
        let result_val = self.load_at_offset(found_blk, source_slot.source_ptr, payload_offset);
        self.store_single_slot_compact_result(
            found_blk,
            rd,
            result_val,
            super::CompactStateSlot::raw(source_slot.source_ptr, payload_offset),
            result_shape,
        )?;
        self.emit(
            found_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );
        self.emit_runtime_error_and_return(wrong_arm_blk, JitRuntimeErrorKind::TypeMismatch);

        Ok(Some(merge_blk))
    }

    /// Lower FuncApply { rd, func, arg }: function application f[x].
    ///
    /// Linear scan: for each key in the function, compare with arg.
    /// If found, return the corresponding value. If not found, runtime error.
    ///
    /// CFG:
    ///   entry -> header
    ///   header -> body (if i < len) | not_found (if i >= len)
    ///   body -> found (if key == arg) | inc (if key != arg)
    ///   inc -> header
    ///   found -> merge (rd = value)
    ///   not_found -> runtime_error
    pub(super) fn lower_func_apply(
        &mut self,
        block_idx: usize,
        rd: u8,
        func_reg: u8,
        arg_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        if let Some(merge_blk) =
            self.lower_func_apply_tagged_union(block_idx, rd, func_reg, arg_reg)?
        {
            return Ok(Some(merge_blk));
        }
        // A `TaggedScalarUnion` argument is a universe INDEX, not the raw key
        // payload, so using it to index a function (`f[g[k]]` where `g[k]` is a
        // union value) would look up the wrong slot with the raw index. It can
        // be used as a domain key ONLY for a contiguous-int-domain compact
        // function (the `domain_lo: Some` indexed-load path below), where the
        // index is DECODED back to its raw int node value before the domain
        // lookup (`decode_scalar_key_reg_raw_value` — the exact
        // inverse of the union encode). For every other function shape
        // (explicit / tuple-keyed / record / sequence), the union index space
        // does not match the domain key encoding, so we fail closed to the
        // interpreter. The union-range function being READ arrives as
        // `func_reg`; only a union KEY is handled here.
        if self.reg_is_tagged_scalar_union(arg_reg) {
            // The union index can be decoded to a raw int domain key for the
            // compact apply paths that take an int key:
            //   * a compact FUNCTION apply — the contiguous-int-domain indexed
            //     load (`domain_lo: Some`) or the explicit-domain linear scan
            //     (`domain_lo: None`);
            //   * a compact SEQUENCE apply — a `[1..N -> V]` total function is
            //     modeled as a `Sequence` (e.g. btree `keysOf \in [Nodes ->
            //     SUBSET Keys]` with `Nodes = 1..8`), applied by 1-based index.
            // Each path decodes the index in its own `arg_val` computation below
            // (see `decode_scalar_key_reg_raw_value`). Gate on the
            // register SHAPE alone (checking the compact state slot here would
            // emit a spurious pointer-reload); a matching shape with no compact
            // slot simply matches none of the paths below and fails closed
            // generically. The tuple-keyed path requires a fixed-arity
            // `Sequence` ARG and fails closed on a scalar union, so it is never
            // mishandled.
            let func_shape_takes_int_key = matches!(
                self.aggregate_shapes.get(&func_reg),
                Some(
                    super::AggregateShape::Function { .. }
                        | super::AggregateShape::Sequence { .. }
                )
            );
            if !func_shape_takes_int_key {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "FuncApply: argument r{arg_reg} is a TaggedScalarUnion (universe index) and function r{func_reg} is not a compact function or sequence; failing closed to interpreter"
                )));
            }
            // Otherwise fall through: the compact apply paths below decode the
            // union index to its raw int domain key.
        }
        if let Some(path_raw) = self.scalar_of(arg_reg) {
            let selector_mode = self.scalar_record_selector_mode(arg_reg);
            if let (Some(source_slot), Some(record_shape)) = (
                self.compact_state_slot_for_use(block_idx, func_reg)?,
                self.aggregate_shapes.get(&func_reg).cloned(),
            ) {
                if let Some((_field_name, field_offset, field_shape)) =
                    Self::compact_record_field_from_scalar_key(
                        &record_shape,
                        path_raw,
                        selector_mode,
                    )
                {
                    let Some(field_shape) = field_shape else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "FuncApply: compact record field r{func_reg}[{path_raw}] has no tracked shape"
                        )));
                    };
                    let field_stride = field_shape.compact_slot_count().ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "FuncApply: compact record field r{func_reg}[{path_raw}] requires fixed-width shape, got {field_shape:?}"
                        ))
                    })?;
                    let field_source_slot = super::CompactStateSlot::raw(
                        source_slot.source_ptr,
                        source_slot.offset + field_offset,
                    );
                    let result_function_domain = self
                        .compact_nested_function_domain_from_raw_state_value_slot(
                            field_source_slot,
                            &field_shape,
                        );
                    if field_stride == 1 {
                        let result_val = self.load_at_offset(
                            block_idx,
                            source_slot.source_ptr,
                            source_slot.offset + field_offset,
                        );
                        self.store_single_slot_compact_result(
                            block_idx,
                            rd,
                            result_val,
                            super::CompactStateSlot::raw(
                                source_slot.source_ptr,
                                source_slot.offset + field_offset,
                            ),
                            field_shape,
                        )?;
                    } else {
                        let source_base = self.emit_i64_const(
                            block_idx,
                            i64::from(source_slot.offset + field_offset),
                        );
                        let result_ptr = self.copy_compact_slots_from_dynamic_base(
                            block_idx,
                            source_slot.source_ptr,
                            source_base,
                            field_stride,
                        );
                        self.store_compact_aggregate_result(
                            block_idx,
                            rd,
                            result_ptr,
                            field_shape,
                        )?;
                    }
                    if let Some(domain) = result_function_domain {
                        self.compact_function_domains.insert(rd, domain);
                    }
                    return Ok(Some(block_idx));
                }
            }

            if let Some((_field_name, field_idx, field_shape)) = self
                .aggregate_shapes
                .get(&func_reg)
                .and_then(|shape| shape.record_field_from_scalar_key(path_raw, selector_mode))
            {
                self.reject_raw_compact_pointer_fallback(func_reg, "FuncApply")?;
                let rec_ptr = self.load_reg_as_ptr(block_idx, func_reg)?;
                let result_val = self.load_at_offset(block_idx, rec_ptr, field_idx);
                self.store_reg_value(block_idx, rd, result_val)?;
                self.compact_state_slots.remove(&rd);
                if let Some(shape) = field_shape {
                    if let Some(len) = shape.tracked_len() {
                        self.const_set_sizes.insert(rd, len);
                    } else {
                        self.const_set_sizes.remove(&rd);
                    }
                    self.aggregate_shapes.insert(rd, shape);
                } else {
                    self.aggregate_shapes.remove(&rd);
                    self.const_set_sizes.remove(&rd);
                }
                self.const_scalar_values.remove(&rd);
                return Ok(Some(block_idx));
            }
        }

        if let (
            Some(source_slot),
            Some(super::AggregateShape::Function {
                len,
                domain_lo: None,
                value,
                ..
            }),
        ) = (
            self.compact_state_slot_for_use(block_idx, func_reg)?,
            self.aggregate_shapes.get(&func_reg).cloned(),
        ) {
            // Tuple / cross-product-keyed compact function (e.g. GameOfLife's
            // `grid \in [{<<x,y>> : x,y \in 1..N} -> BOOLEAN]`). The flat layout
            // carries a `Tuple` key shape but no scalar `SetBitmaskElement`
            // domain, so the scalar explicit-domain recovery below cannot apply.
            // Recover the ordered tuple key table and compare the tuple argument
            // element-wise. Recovery fails closed when the keys are not provably
            // reconstructible, leaving the interpreter fallback intact.
            if let Some(tuple_keys) =
                self.tuple_function_domain_keys_for_explicit_function(func_reg, source_slot, len)?
            {
                return self.lower_tuple_keyed_compact_func_apply(
                    block_idx,
                    rd,
                    func_reg,
                    arg_reg,
                    source_slot,
                    len,
                    value,
                    tuple_keys,
                );
            }
            let domain_keys = self.compact_function_domain_for_explicit_function(
                "FuncApply",
                func_reg,
                source_slot,
            )?;
            return self.lower_explicit_domain_compact_func_apply(
                block_idx,
                rd,
                func_reg,
                arg_reg,
                source_slot,
                len,
                value,
                domain_keys,
            );
        }

        if let (
            Some(source_slot),
            Some(super::AggregateShape::Function {
                len,
                domain_lo: Some(domain_lo),
                value,
                ..
            }),
        ) = (
            self.compact_state_slot_for_use(block_idx, func_reg)?,
            self.aggregate_shapes.get(&func_reg).cloned(),
        ) {
            let value_shape = value.as_deref().cloned();
            let value_shape = self
                .compact_function_value_shape_from_source_layout(
                    source_slot,
                    len,
                    Some(domain_lo),
                    None,
                    value_shape.as_ref(),
                )
                .or(value_shape);
            let value_stride = value_shape
                .as_ref()
                .and_then(super::AggregateShape::compact_slot_count)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "FuncApply: compact function r{func_reg} requires fixed-width value shape, got {value_shape:?}"
                    ))
                })?;
            let result_function_domain =
                self.compact_function_apply_result_domain(source_slot, value_shape.as_ref());

            let arg_val = self.load_reg(block_idx, arg_reg)?;
            // WP-18 follow-on: a tagged-scalar-union key holds the union
            // INDEX; decode it to the member value (out-of-arm members fall
            // to the not-found guard below).
            let arg_val = self.decode_scalar_key_reg_raw_value(block_idx, arg_reg, arg_val);
            let lo_val = self.emit_i64_const(block_idx, domain_lo);
            let hi_val = self.emit_i64_const(block_idx, domain_lo + i64::from(len) - 1);
            let ge_lo = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Sge,
                    ty: Ty::I64,
                    lhs: arg_val,
                    rhs: lo_val,
                },
            );
            let check_hi_blk = self.new_aux_block("compact_fapply_check_hi");
            let found_blk = self.new_aux_block("compact_fapply_found");
            let not_found_blk = self.new_aux_block("compact_fapply_not_found");
            let merge_blk = self.new_aux_block("compact_fapply_merge");
            let check_hi_id = self.block_id_of(check_hi_blk);
            let found_id = self.block_id_of(found_blk);
            let not_found_id = self.block_id_of(not_found_blk);
            let merge_id = self.block_id_of(merge_blk);
            self.emit(
                block_idx,
                InstrNode::new(Inst::CondBr {
                    cond: ge_lo,
                    then_target: check_hi_id,
                    then_args: vec![],
                    else_target: not_found_id,
                    else_args: vec![],
                }),
            );

            let le_hi = self.emit_with_result(
                check_hi_blk,
                Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: arg_val,
                    rhs: hi_val,
                },
            );
            self.emit(
                check_hi_blk,
                InstrNode::new(Inst::CondBr {
                    cond: le_hi,
                    then_target: found_id,
                    then_args: vec![],
                    else_target: not_found_id,
                    else_args: vec![],
                }),
            );

            let rel_idx = self.emit_with_result(
                found_blk,
                Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: arg_val,
                    rhs: lo_val,
                },
            );
            let base = self.emit_i64_const(found_blk, i64::from(source_slot.offset));
            let value_slot = if value_stride == 1 {
                self.emit_with_result(
                    found_blk,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: base,
                        rhs: rel_idx,
                    },
                )
            } else {
                let stride = self.emit_i64_const(found_blk, i64::from(value_stride));
                let offset = self.emit_with_result(
                    found_blk,
                    Inst::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::I64,
                        lhs: rel_idx,
                        rhs: stride,
                    },
                );
                self.emit_with_result(
                    found_blk,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: base,
                        rhs: offset,
                    },
                )
            };
            if value_stride == 1 {
                let result_val =
                    self.load_at_dynamic_offset(found_blk, source_slot.source_ptr, value_slot);
                if let Some(shape) = value_shape.clone() {
                    // A `TaggedScalarUnion` range slot is a single i64 universe
                    // INDEX — a scalar-like value, NOT a pointer-backed compound.
                    // Route it through the same raw-value store as `Scalar` so
                    // `rd` holds the index directly (a downstream equality reads
                    // it via `load_reg`); the pointer-backed path would mark `rd`
                    // as an aggregate pointer and a consumer could deref the
                    // index as a pointer.
                    if matches!(
                        &shape,
                        super::AggregateShape::Scalar(_)
                            | super::AggregateShape::TaggedScalarUnion { .. }
                    ) {
                        self.store_single_slot_compact_result(
                            found_blk,
                            rd,
                            result_val,
                            super::CompactStateSlot::raw(self.reg_ptr(rd)?, 0),
                            shape,
                        )?;
                    } else {
                        let result_ptr = self.emit_state_slot_ptr_at_dynamic_slot(
                            found_blk,
                            source_slot.source_ptr,
                            value_slot,
                        );
                        self.store_single_slot_compact_result(
                            found_blk,
                            rd,
                            result_val,
                            super::CompactStateSlot::pointer_backed_in_block(
                                result_ptr, 0, found_blk,
                            ),
                            shape,
                        )?;
                    }
                } else {
                    self.store_reg_value(found_blk, rd, result_val)?;
                    self.aggregate_shapes.remove(&rd);
                    self.const_set_sizes.remove(&rd);
                    self.compact_state_slots.remove(&rd);
                    self.const_scalar_values.remove(&rd);
                }
            } else {
                if let Some(shape) = value_shape.clone() {
                    let result_ptr = self.emit_state_slot_ptr_at_dynamic_slot(
                        found_blk,
                        source_slot.source_ptr,
                        value_slot,
                    );
                    self.store_compact_aggregate_result(found_blk, rd, result_ptr, shape)?;
                } else {
                    let result_ptr = self.copy_compact_slots_from_dynamic_base(
                        found_blk,
                        source_slot.source_ptr,
                        value_slot,
                        value_stride,
                    );
                    self.store_reg_ptr(found_blk, rd, result_ptr)?;
                    self.compact_state_slots.insert(
                        rd,
                        super::CompactStateSlot::pointer_backed_in_block(result_ptr, 0, found_blk),
                    );
                    self.aggregate_shapes.remove(&rd);
                    self.const_set_sizes.remove(&rd);
                    self.const_scalar_values.remove(&rd);
                }
            }
            if let Some(domain) = result_function_domain {
                self.compact_function_domains.insert(rd, domain);
            }
            self.emit(
                found_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );
            self.emit_runtime_error_and_return(not_found_blk, JitRuntimeErrorKind::TypeMismatch);

            return Ok(Some(merge_blk));
        }

        if let Some(super::AggregateShape::Sequence { extent, element }) =
            self.aggregate_shapes.get(&func_reg).cloned()
        {
            let seq_shape = super::AggregateShape::Sequence {
                extent,
                element: element.clone(),
            };
            let seq_capacity = extent.capacity();
            let element_shape = element.as_deref().cloned();
            let element_stride = element_shape
                .as_ref()
                .and_then(super::AggregateShape::compact_slot_count)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "FuncApply: compact sequence r{func_reg} requires fixed-width element shape, got {element_shape:?}"
                    ))
                })?;
            let (block_idx, source_slot) =
                if let Some(source_slot) = self.compact_state_slot_for_use(block_idx, func_reg)? {
                    (block_idx, source_slot)
                } else {
                    let source =
                        self.materialize_reg_as_compact_source(block_idx, func_reg, &seq_shape)?;
                    (source.block_idx, source.slot)
                };

            let arg_val = self.load_reg(block_idx, arg_reg)?;
            // WP-18 follow-on: decode a tagged-scalar-union index key to its
            // member value before using it as a 1-based sequence index.
            let arg_val = self.decode_scalar_key_reg_raw_value(block_idx, arg_reg, arg_val);
            let _seq_len =
                self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset);
            let one = self.emit_i64_const(block_idx, 1);
            let ge_one = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Sge,
                    ty: Ty::I64,
                    lhs: arg_val,
                    rhs: one,
                },
            );
            let check_capacity_blk = self.new_aux_block("compact_seq_apply_check_capacity");
            let check_hi_blk = self.new_aux_block("compact_seq_apply_check_hi");
            let found_blk = self.new_aux_block("compact_seq_apply_found");
            let not_found_blk = self.new_aux_block("compact_seq_apply_not_found");
            let merge_blk = self.new_aux_block("compact_seq_apply_merge");
            let check_capacity_id = self.block_id_of(check_capacity_blk);
            let check_hi_id = self.block_id_of(check_hi_blk);
            let found_id = self.block_id_of(found_blk);
            let not_found_id = self.block_id_of(not_found_blk);
            let merge_id = self.block_id_of(merge_blk);

            self.emit(
                block_idx,
                InstrNode::new(Inst::CondBr {
                    cond: ge_one,
                    then_target: check_capacity_id,
                    then_args: vec![],
                    else_target: not_found_id,
                    else_args: vec![],
                }),
            );

            let capacity_seq_len = self.load_at_offset(
                check_capacity_blk,
                source_slot.source_ptr,
                source_slot.offset,
            );
            let capacity_val = self.emit_i64_const(check_capacity_blk, i64::from(seq_capacity));
            let within_capacity = self.emit_with_result(
                check_capacity_blk,
                Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: capacity_seq_len,
                    rhs: capacity_val,
                },
            );
            self.emit(
                check_capacity_blk,
                InstrNode::new(Inst::CondBr {
                    cond: within_capacity,
                    then_target: check_hi_id,
                    then_args: vec![],
                    else_target: not_found_id,
                    else_args: vec![],
                }),
            );

            let check_hi_arg = self.load_reg(check_hi_blk, arg_reg)?;
            let check_hi_arg =
                self.decode_scalar_key_reg_raw_value(check_hi_blk, arg_reg, check_hi_arg);
            let check_hi_seq_len =
                self.load_at_offset(check_hi_blk, source_slot.source_ptr, source_slot.offset);
            let le_len = self.emit_with_result(
                check_hi_blk,
                Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: check_hi_arg,
                    rhs: check_hi_seq_len,
                },
            );
            self.emit(
                check_hi_blk,
                InstrNode::new(Inst::CondBr {
                    cond: le_len,
                    then_target: found_id,
                    then_args: vec![],
                    else_target: not_found_id,
                    else_args: vec![],
                }),
            );

            let found_arg = self.load_reg(found_blk, arg_reg)?;
            let found_arg = self.decode_scalar_key_reg_raw_value(found_blk, arg_reg, found_arg);
            let one_found = self.emit_i64_const(found_blk, 1);
            let rel_idx = self.emit_with_result(
                found_blk,
                Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: found_arg,
                    rhs: one_found,
                },
            );
            let first_elem_slot = source_slot.offset.checked_add(1).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "FuncApply: compact sequence first element slot overflows".to_owned(),
                )
            })?;
            let result_function_domain = element_shape.as_ref().and_then(|shape| {
                self.compact_nested_function_domain_from_raw_state_value_slot(
                    super::CompactStateSlot::raw(source_slot.source_ptr, first_elem_slot),
                    shape,
                )
            });
            let first_elem = self.emit_i64_const(found_blk, i64::from(first_elem_slot));
            let elem_slot = if element_stride == 1 {
                self.emit_with_result(
                    found_blk,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: first_elem,
                        rhs: rel_idx,
                    },
                )
            } else {
                let stride = self.emit_i64_const(found_blk, i64::from(element_stride));
                let offset = self.emit_with_result(
                    found_blk,
                    Inst::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::I64,
                        lhs: rel_idx,
                        rhs: stride,
                    },
                );
                self.emit_with_result(
                    found_blk,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: first_elem,
                        rhs: offset,
                    },
                )
            };
            if element_stride == 1 {
                let result_val =
                    self.load_at_dynamic_offset(found_blk, source_slot.source_ptr, elem_slot);
                if let Some(shape) = element_shape {
                    if matches!(
                        &shape,
                        super::AggregateShape::Scalar(_)
                            | super::AggregateShape::ScalarIntDomain { .. }
                            // A `TaggedScalarUnion` sequence element is a single
                            // i64 universe INDEX — a scalar-like value; load it
                            // directly into `rd` (a downstream equality reads it
                            // via `load_reg`).
                            | super::AggregateShape::TaggedScalarUnion { .. }
                    ) {
                        self.store_single_slot_compact_result(
                            found_blk,
                            rd,
                            result_val,
                            super::CompactStateSlot::raw(self.reg_ptr(rd)?, 0),
                            shape,
                        )?;
                    } else {
                        let result_ptr = self.copy_compact_slots_from_dynamic_base(
                            found_blk,
                            source_slot.source_ptr,
                            elem_slot,
                            element_stride,
                        );
                        let stored_val = self.load_at_offset(found_blk, result_ptr, 0);
                        self.store_single_slot_compact_result(
                            found_blk,
                            rd,
                            stored_val,
                            super::CompactStateSlot::raw(result_ptr, 0),
                            shape,
                        )?;
                    }
                } else {
                    self.store_reg_value(found_blk, rd, result_val)?;
                    self.aggregate_shapes.remove(&rd);
                    self.const_set_sizes.remove(&rd);
                    self.compact_state_slots.remove(&rd);
                    self.const_scalar_values.remove(&rd);
                }
            } else {
                let result_ptr = self.copy_compact_slots_from_dynamic_base(
                    found_blk,
                    source_slot.source_ptr,
                    elem_slot,
                    element_stride,
                );
                if let Some(shape) = element_shape {
                    self.store_compact_aggregate_result(found_blk, rd, result_ptr, shape)?;
                } else {
                    self.store_reg_ptr(found_blk, rd, result_ptr)?;
                    self.compact_state_slots.insert(
                        rd,
                        super::CompactStateSlot::pointer_backed_in_block(result_ptr, 0, found_blk),
                    );
                    self.aggregate_shapes.remove(&rd);
                    self.const_set_sizes.remove(&rd);
                    self.const_scalar_values.remove(&rd);
                }
            }
            if let Some(domain) = result_function_domain {
                self.compact_function_domains.insert(rd, domain);
            }
            self.emit(
                found_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );
            self.emit_runtime_error_and_return(not_found_blk, JitRuntimeErrorKind::TypeMismatch);

            return Ok(Some(merge_blk));
        }

        self.reject_raw_compact_pointer_fallback(func_reg, "FuncApply")?;
        let arg_val = self.load_reg(block_idx, arg_reg)?;
        // WP-18 follow-on: pair-list keys are stored as raw member values;
        // decode a tagged-scalar-union index key before the compare scan.
        let arg_val = self.decode_scalar_key_reg_raw_value(block_idx, arg_reg, arg_val);
        let func_ptr = self.load_reg_as_ptr(block_idx, func_reg)?;
        let pair_count = self.load_at_offset(block_idx, func_ptr, 0);

        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);
        let two = self.emit_i64_const(block_idx, 2);

        let idx_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let loop_header = self.new_aux_block("fapply_header");
        let loop_body = self.new_aux_block("fapply_body");
        let loop_inc = self.new_aux_block("fapply_inc");
        let found_blk = self.new_aux_block("fapply_found");
        let not_found_blk = self.new_aux_block("fapply_not_found");
        let merge_blk = self.new_aux_block("fapply_merge");

        let header_id = self.block_id_of(loop_header);
        let body_id = self.block_id_of(loop_body);
        let inc_id = self.block_id_of(loop_inc);
        let found_id = self.block_id_of(found_blk);
        let not_found_id = self.block_id_of(not_found_blk);
        let merge_id = self.block_id_of(merge_blk);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        // Header: i < pair_count?
        let cur_idx = self.emit_with_result(
            loop_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp = self.emit_with_result(
            loop_header,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: pair_count,
            },
        );
        self.emit(
            loop_header,
            InstrNode::new(Inst::CondBr {
                cond: cmp,
                then_target: body_id,
                then_args: vec![],
                else_target: not_found_id,
                else_args: vec![],
            }),
        );

        // Body: load key at slot[1 + 2*i], compare with arg
        let cur_idx2 = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let key_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: cur_idx2,
                rhs: two,
            },
        );
        let key_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: key_offset,
                rhs: one,
            },
        );
        let key_val = self.load_at_dynamic_offset(loop_body, func_ptr, key_slot);
        let eq = self.emit_with_result(
            loop_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: key_val,
                rhs: arg_val,
            },
        );
        self.emit(
            loop_body,
            InstrNode::new(Inst::CondBr {
                cond: eq,
                then_target: found_id,
                then_args: vec![],
                else_target: inc_id,
                else_args: vec![],
            }),
        );

        // Increment: i++, goto header
        let cur_idx3 = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_idx = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx3,
                rhs: one,
            },
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        // Found: load value at slot[2 + 2*i], store to rd
        let cur_idx4 = self.emit_with_result(
            found_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let val_offset = self.emit_with_result(
            found_blk,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: cur_idx4,
                rhs: two,
            },
        );
        let val_slot = self.emit_with_result(
            found_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: val_offset,
                rhs: two,
            },
        );
        let result_val = self.load_at_dynamic_offset(found_blk, func_ptr, val_slot);
        self.store_reg_value(found_blk, rd, result_val)?;
        self.compact_state_slots.remove(&rd);
        self.emit(
            found_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        // Not found: runtime error (function applied to value not in domain)
        self.emit_runtime_error_and_return(not_found_blk, JitRuntimeErrorKind::TypeMismatch);

        if let Some(shape) = self
            .aggregate_shapes
            .get(&func_reg)
            .and_then(|shape| shape.function_value_shape())
        {
            if let Some(len) = shape.tracked_len() {
                self.const_set_sizes.insert(rd, len);
            } else {
                self.const_set_sizes.remove(&rd);
            }
            self.aggregate_shapes.insert(rd, shape);
        } else {
            self.aggregate_shapes.remove(&rd);
            self.const_set_sizes.remove(&rd);
        }
        self.const_scalar_values.remove(&rd);

        Ok(Some(merge_blk))
    }

    /// Lower Domain { rd, rs }: extract domain set from function.
    ///
    /// Reads the function aggregate at rs and builds a new set containing
    /// all keys. Function layout: [pair_count, k1, v1, k2, v2, ...].
    /// Result set layout: [pair_count, k1, k2, ...].
    pub(super) fn lower_domain(
        &mut self,
        block_idx: usize,
        rd: u8,
        rs: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        let domain_shape = self
            .aggregate_shapes
            .get(&rs)
            .and_then(super::AggregateShape::function_domain_shape);
        if self
            .compact_state_slots
            .get(&rs)
            .copied()
            .is_some_and(super::CompactStateSlot::is_raw_compact_slot)
        {
            if let Some(super::AggregateShape::Function {
                len,
                domain_lo: Some(domain_lo),
                ..
            }) = self.aggregate_shapes.get(&rs).cloned()
            {
                let total_slots = len.checked_add(1).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Domain: raw compact function r{rs} domain slot count overflows"
                    ))
                })?;
                let result_ptr = self.alloc_aggregate(block_idx, total_slots);
                let len_value = self.emit_i64_const(block_idx, i64::from(len));
                self.store_at_offset(block_idx, result_ptr, 0, len_value);

                for idx in 0..len {
                    let key = domain_lo.checked_add(i64::from(idx)).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "Domain: raw compact function r{rs} domain key overflows"
                        ))
                    })?;
                    let key_value = self.emit_i64_const(block_idx, key);
                    self.store_at_offset(block_idx, result_ptr, idx + 1, key_value);
                }

                self.store_reg_ptr(block_idx, rd, result_ptr)?;
                self.compact_state_slots.remove(&rd);
                self.aggregate_shapes.insert(
                    rd,
                    domain_shape
                        .clone()
                        .unwrap_or(super::AggregateShape::Set { len, element: None }),
                );
                self.record_set_size(rd, len);
                self.const_scalar_values.remove(&rd);
                return Ok(Some(block_idx));
            }
        }

        let func_ptr = self.load_reg_as_ptr_or_materialize_raw_compact(block_idx, rs, "Domain")?;
        let pair_count = self.load_at_offset(block_idx, func_ptr, 0);

        let one = self.emit_i64_const(block_idx, 1);
        let two = self.emit_i64_const(block_idx, 2);
        let zero = self.emit_i64_const(block_idx, 0);

        // Allocate result set: pair_count + 1 slots (length header + keys)
        let total = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: pair_count,
                rhs: one,
            },
        );
        let result_ptr = if let Some(len) = self.const_set_sizes.get(&rs).copied() {
            self.alloc_aggregate(block_idx, len + 1)
        } else {
            self.alloc_dynamic_i64_slots(block_idx, total)
        };

        // Store length = pair_count
        self.store_at_offset(block_idx, result_ptr, 0, pair_count);

        // Loop: for i in 0..pair_count, result[i+1] = func[1 + 2*i]
        let i_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let loop_hdr = self.new_aux_block("domain_hdr");
        let loop_body = self.new_aux_block("domain_body");
        let loop_done = self.new_aux_block("domain_done");

        let hdr_id = self.block_id_of(loop_hdr);
        let body_id = self.block_id_of(loop_body);
        let done_id = self.block_id_of(loop_done);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: hdr_id,
                args: vec![],
            }),
        );

        // Header: i < pair_count?
        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: pair_count,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp,
                then_target: body_id,
                then_args: vec![],
                else_target: done_id,
                else_args: vec![],
            }),
        );

        // Body: result[i+1] = func[1 + 2*i]
        let i_val2 = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let key_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: two,
            },
        );
        let key_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: key_offset,
                rhs: one,
            },
        );
        let key_val = self.load_at_dynamic_offset(loop_body, func_ptr, key_slot);

        let dst_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        self.store_at_dynamic_offset(loop_body, result_ptr, dst_slot, key_val);

        let next_i = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        self.emit(
            loop_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_body,
            InstrNode::new(Inst::Br {
                target: hdr_id,
                args: vec![],
            }),
        );

        // Done
        self.store_reg_ptr(loop_done, rd, result_ptr)?;
        self.compact_state_slots.remove(&rd);
        if let Some(n) = self.const_set_sizes.get(&rs).copied() {
            self.aggregate_shapes.insert(
                rd,
                domain_shape.unwrap_or(super::AggregateShape::Set {
                    len: n,
                    element: None,
                }),
            );
            self.record_set_size(rd, n);
        } else {
            self.aggregate_shapes.remove(&rd);
            self.const_set_sizes.remove(&rd);
        }
        self.const_scalar_values.remove(&rd);

        Ok(Some(loop_done))
    }

    /// Lower FuncExcept { rd, func, path, val }: [f EXCEPT ![x] = y].
    ///
    /// Creates a new function identical to func but with the value at key `path`
    /// replaced by `val`. If the key does not exist, the original function is
    /// returned unchanged (TLA+ semantics: EXCEPT on non-existent key is identity).
    ///
    /// Implementation: allocate new function aggregate of same size, copy all pairs,
    /// but when key matches `path`, store `val` instead of the original value.
    pub(super) fn lower_func_except(
        &mut self,
        block_idx: usize,
        rd: u8,
        func_reg: u8,
        path_reg: u8,
        val_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        // SOUNDNESS (compound-state-var merge SIGSEGV): if the aggregate being
        // updated is a compound register that lost its flat-buffer alias at a
        // control-flow merge (so its register word is now just a length / first
        // slot, not a pointer), every compact path below would eventually
        // `IntToPtr` that word and dereference it as a heap aggregate — a wild
        // load. We cannot faithfully rebuild the aggregate from a length, so
        // fail closed to the byte-correct interpreter for this state.
        if self.compound_reg_lacks_pointer_representation(func_reg) {
            self.emit_fallback_needed_and_return(block_idx);
            // Returning `None` makes the main lowering loop skip the remaining
            // (now dead) opcodes of this straight-line region until the next
            // labelled block, so the dependent `StoreVar v1 = r{rd}` is never
            // lowered. At runtime control already returned `FallbackNeeded`
            // above, so nothing downstream executes.
            return Ok(None);
        }
        if let Some(path_raw) = self.scalar_of(path_reg) {
            let selector_mode = self.scalar_record_selector_mode(path_reg);
            if let (Some(source_slot), Some(record_shape)) = (
                self.compact_state_slot_for_use(block_idx, func_reg)?,
                self.aggregate_shapes.get(&func_reg).cloned(),
            ) {
                if let Some((field_name, field_offset, field_shape)) =
                    Self::compact_record_field_from_scalar_key(
                        &record_shape,
                        path_raw,
                        selector_mode,
                    )
                {
                    let Some(field_shape) = field_shape else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "FuncExcept: compact record field r{func_reg}[{path_raw}] has no tracked shape"
                        )));
                    };
                    let field_stride = field_shape.compact_slot_count().ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "FuncExcept: compact record field r{func_reg}[{path_raw}] requires fixed-width shape, got {field_shape:?}"
                        ))
                    })?;
                    let total_slots = record_shape.compact_slot_count().ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "FuncExcept: compact record r{func_reg} requires fixed-width shape, got {record_shape:?}"
                        ))
                    })?;
                    let result_ptr = self.alloc_aggregate(block_idx, total_slots);
                    let new_scalar_val = if field_stride == 1
                        && Self::is_single_slot_flat_aggregate_value(&field_shape)
                    {
                        Some(self.load_reg_as_compatible_single_slot_value(
                            block_idx,
                            val_reg,
                            &field_shape,
                            "FuncExcept compact record replacement",
                        )?)
                    } else {
                        None
                    };
                    let new_compact_source = if new_scalar_val.is_none() {
                        let materialized =
                            self.compact_value_source_for_reg(block_idx, val_reg, &field_shape)?;
                        let block_idx = materialized.block_idx;
                        Some((materialized.slot, block_idx))
                    } else {
                        None
                    };
                    let block_idx = new_compact_source
                        .map_or(block_idx, |(_, materialized_block)| materialized_block);
                    let new_compact_source =
                        new_compact_source.map(|(slot, _materialized_block)| slot);

                    for slot in 0..total_slots {
                        let old_val = self.load_at_offset(
                            block_idx,
                            source_slot.source_ptr,
                            source_slot.offset + slot,
                        );
                        let in_replaced_field =
                            slot >= field_offset && slot < field_offset + field_stride;
                        let value = if in_replaced_field {
                            if let Some(new_scalar_val) = new_scalar_val {
                                new_scalar_val
                            } else {
                                let new_compact_source = new_compact_source
                                    .expect("compact record replacement source was checked above");
                                self.load_at_offset(
                                    block_idx,
                                    new_compact_source.source_ptr,
                                    new_compact_source.offset + slot - field_offset,
                                )
                            }
                        } else {
                            old_val
                        };
                        self.store_at_offset(block_idx, result_ptr, slot, value);
                    }

                    let updated_shape = record_shape.with_record_field_shape(
                        field_name,
                        self.aggregate_shapes.get(&val_reg).cloned(),
                    );
                    if total_slots == 1 {
                        let first = self.load_at_offset(block_idx, result_ptr, 0);
                        self.store_single_slot_compact_result(
                            block_idx,
                            rd,
                            first,
                            super::CompactStateSlot::raw(result_ptr, 0),
                            updated_shape,
                        )?;
                    } else {
                        self.store_compact_aggregate_result(
                            block_idx,
                            rd,
                            result_ptr,
                            updated_shape,
                        )?;
                    }
                    return Ok(Some(block_idx));
                } else if matches!(&record_shape, super::AggregateShape::Record { .. }) {
                    self.store_compact_identity_result(block_idx, rd, source_slot, record_shape)?;
                    return Ok(Some(block_idx));
                }
            }

            if let Some(record_shape) = self.aggregate_shapes.get(&func_reg).cloned() {
                if let Some((field_name, field_idx, _)) =
                    record_shape.record_field_from_scalar_key(path_raw, selector_mode)
                {
                    self.reject_raw_compact_pointer_fallback(func_reg, "FuncExcept")?;
                    let rec_ptr = self.load_reg_as_ptr(block_idx, func_reg)?;
                    let new_val = self.load_reg(block_idx, val_reg)?;
                    let super::AggregateShape::Record { fields } = &record_shape else {
                        unreachable!("record_field returned Some only for record shapes");
                    };
                    let result_ptr = self.alloc_aggregate(
                        block_idx,
                        u32::try_from(fields.len()).expect("record field count must fit in u32"),
                    );
                    for slot in 0..fields.len() {
                        let slot_u32 =
                            u32::try_from(slot).expect("record field index must fit in u32");
                        let src_val = self.load_at_offset(block_idx, rec_ptr, slot_u32);
                        self.store_at_offset(
                            block_idx,
                            result_ptr,
                            slot_u32,
                            if slot_u32 == field_idx {
                                new_val
                            } else {
                                src_val
                            },
                        );
                    }
                    self.store_reg_ptr(block_idx, rd, result_ptr)?;
                    self.compact_state_slots.remove(&rd);
                    self.aggregate_shapes.insert(
                        rd,
                        record_shape.with_record_field_shape(
                            field_name,
                            self.aggregate_shapes.get(&val_reg).cloned(),
                        ),
                    );
                    self.const_set_sizes.remove(&rd);
                    self.const_scalar_values.remove(&rd);
                    return Ok(Some(block_idx));
                }
            }
        } else if let (Some(_source_slot), Some(super::AggregateShape::Record { .. })) = (
            self.compact_state_slots.get(&func_reg).copied(),
            self.aggregate_shapes.get(&func_reg).cloned(),
        ) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "FuncExcept: dynamic compact record path r{path_reg} for r{func_reg} is not supported"
            )));
        }

        if let (
            Some(source_slot),
            Some(super::AggregateShape::Function {
                len,
                domain_lo: None,
                value,
                ..
            }),
        ) = (
            self.compact_state_slot_for_use(block_idx, func_reg)?,
            self.aggregate_shapes.get(&func_reg).cloned(),
        ) {
            // WP-09: tuple/cross-product-keyed compact function EXCEPT (btree
            // `childOf'`/`valOf'`). The flat layout carries a `Tuple` key shape
            // but no scalar `SetBitmaskElement` domain, so the scalar
            // explicit-domain recovery below cannot apply. Recover the ordered
            // tuple-key table (ABI channel first, const-pool fallback) and
            // lower the masked ordinal write. Recovery fails closed when the
            // keys are not provably reconstructible, leaving the interpreter
            // fallback intact via the explicit-domain metadata error below.
            if let Some(tuple_keys) =
                self.tuple_function_domain_keys_for_explicit_function(func_reg, source_slot, len)?
            {
                return self.lower_tuple_keyed_compact_func_except(
                    block_idx, rd, func_reg, path_reg, val_reg, source_slot, len, value,
                    tuple_keys,
                );
            }
            let domain_keys = self.compact_function_domain_for_explicit_function(
                "FuncExcept",
                func_reg,
                source_slot,
            )?;
            let value_shape = self
                .compact_function_value_shape_from_source_layout(
                    source_slot,
                    len,
                    None,
                    Some(&domain_keys),
                    value.as_deref(),
                )
                .or_else(|| value.as_deref().cloned())
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "FuncExcept: compact function r{func_reg} requires fixed-width value shape, got {value:?}"
                    ))
                })?;
            return self.lower_explicit_domain_compact_func_except(
                block_idx,
                rd,
                func_reg,
                path_reg,
                val_reg,
                source_slot,
                len,
                value_shape,
                domain_keys,
            );
        }

        if let (
            Some(source_slot),
            Some(super::AggregateShape::Function {
                len,
                domain_lo: Some(domain_lo),
                domain: None,
                value,
            }),
        ) = (
            self.compact_state_slot_for_use(block_idx, func_reg)?,
            self.aggregate_shapes.get(&func_reg).cloned(),
        ) {
            let value_shape = self
                .compact_function_value_shape_from_source_layout(
                    source_slot,
                    len,
                    Some(domain_lo),
                    None,
                    value.as_deref(),
                )
                .or_else(|| value.as_deref().cloned())
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "FuncExcept: compact function r{func_reg} requires fixed-width value shape, got {value:?}"
                    ))
                })?;
            let result_function_shape = super::AggregateShape::Function {
                len,
                domain_lo: Some(domain_lo),
                domain: None,
                value: Some(Box::new(value_shape.clone())),
            };
            let value_stride = value_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept: compact function r{func_reg} requires fixed-width value shape, got {value_shape:?}"
                ))
            })?;
            let total_slots = len.checked_mul(value_stride).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept: compact function slot count overflows: {len} * {value_stride}"
                ))
            })?;

            if let Some(path_raw) = self.scalar_of(path_reg) {
                let replace_idx = path_raw
                    .checked_sub(domain_lo)
                    .and_then(|relative_key| u32::try_from(relative_key).ok());
                let Some(replace_idx) = replace_idx else {
                    self.store_compact_identity_result(
                        block_idx,
                        rd,
                        source_slot,
                        result_function_shape,
                    )?;
                    return Ok(Some(block_idx));
                };
                if replace_idx >= len {
                    self.store_compact_identity_result(
                        block_idx,
                        rd,
                        source_slot,
                        result_function_shape,
                    )?;
                    return Ok(Some(block_idx));
                }

                let (result_shape, replacement_shape, result_slots) = if len == 1 {
                    let replacement_shape = value_shape.clone();
                    let replacement_slots = value_stride;
                    (
                        super::AggregateShape::Function {
                            len,
                            domain_lo: Some(domain_lo),
                            domain: None,
                            value: Some(Box::new(replacement_shape.clone())),
                        },
                        replacement_shape,
                        replacement_slots,
                    )
                } else {
                    (
                        result_function_shape.clone(),
                        value_shape.clone(),
                        total_slots,
                    )
                };
                let replacement_stride = replacement_shape.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "FuncExcept: compact function replacement requires fixed-width value shape, got {replacement_shape:?}"
                    ))
                })?;
                let replacement =
                    self.compact_value_source_for_reg(block_idx, val_reg, &replacement_shape)?;
                let block_idx = replacement.block_idx;
                let replacement_source = replacement.slot;
                let result_ptr = self.alloc_aggregate(block_idx, result_slots);
                if len > 1 {
                    for slot in 0..total_slots {
                        let old_val = self.load_at_offset(
                            block_idx,
                            source_slot.source_ptr,
                            source_slot.offset + slot,
                        );
                        self.store_at_offset(block_idx, result_ptr, slot, old_val);
                    }
                }
                let replace_base = replace_idx.checked_mul(replacement_stride).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "FuncExcept: compact function replacement slot overflows: {replace_idx} * {replacement_stride}"
                    ))
                })?;
                for value_offset in 0..replacement_stride {
                    let new_val = self.load_at_offset(
                        block_idx,
                        replacement_source.source_ptr,
                        replacement_source.offset + value_offset,
                    );
                    self.store_at_offset(
                        block_idx,
                        result_ptr,
                        replace_base + value_offset,
                        new_val,
                    );
                }
                if result_slots == 1 {
                    let first = self.load_at_offset(block_idx, result_ptr, 0);
                    self.store_single_slot_compact_result(
                        block_idx,
                        rd,
                        first,
                        super::CompactStateSlot::raw(result_ptr, 0),
                        result_shape,
                    )?;
                } else {
                    self.store_compact_aggregate_result(block_idx, rd, result_ptr, result_shape)?;
                }
                return Ok(Some(block_idx));
            }

            let result_ptr = self.alloc_aggregate(block_idx, total_slots);
            let path_val = self.load_reg(block_idx, path_reg)?;
            // WP-18 follow-on: decode a tagged-scalar-union index key to its
            // member value; a non-int member misses every `domain_lo + idx`
            // compare below (out-of-domain EXCEPT = identity).
            let path_val = self.decode_scalar_key_reg_raw_value(block_idx, path_reg, path_val);
            // WP-08 (item 6): a DYNAMIC materialized set source for a
            // SetBitmask range slot cannot be loaded as a single-slot scalar;
            // route it to the compact-materialization path below, which runs
            // the fail-closed dynamic-set-to-mask loop.
            let dynamic_set_bitmask_source = match &value_shape {
                super::AggregateShape::SetBitmask { universe, .. } => self
                    .dynamic_set_to_bitmask_source_capacity(val_reg, universe)
                    .is_some(),
                _ => false,
            };
            let new_scalar_val = if value_stride == 1
                && Self::is_single_slot_flat_aggregate_value(&value_shape)
                && !dynamic_set_bitmask_source
            {
                Some(self.load_reg_as_compatible_single_slot_value(
                    block_idx,
                    val_reg,
                    &value_shape,
                    "FuncExcept compact function replacement",
                )?)
            } else {
                None
            };
            let new_compact_source = if new_scalar_val.is_none() {
                let materialized =
                    self.compact_value_source_for_reg(block_idx, val_reg, &value_shape)?;
                let block_idx = materialized.block_idx;
                Some((materialized.slot, block_idx))
            } else {
                None
            };
            let block_idx =
                new_compact_source.map_or(block_idx, |(_, materialized_block)| materialized_block);
            let new_compact_source = new_compact_source.map(|(slot, _materialized_block)| slot);

            for idx in 0..len {
                let key_val = self.emit_i64_const(block_idx, domain_lo + i64::from(idx));
                let key_match = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: path_val,
                        rhs: key_val,
                    },
                );
                for value_offset in 0..value_stride {
                    let source_offset = source_slot.offset + idx * value_stride + value_offset;
                    let src_ptr = self.emit_state_slot_ptr_at_slot(
                        block_idx,
                        source_slot.source_ptr,
                        source_offset,
                    );
                    let old_val = self.emit_with_result(
                        block_idx,
                        Inst::Load {
                            ty: Ty::I64,
                            ptr: src_ptr,
                            align: None,
                            volatile: false,
                        },
                    );
                    let new_val = if let Some(new_scalar_val) = new_scalar_val {
                        new_scalar_val
                    } else {
                        let new_compact_source = new_compact_source
                            .expect("multi-slot compact update source was checked above");
                        let new_ptr = self.emit_state_slot_ptr_at_slot(
                            block_idx,
                            new_compact_source.source_ptr,
                            new_compact_source.offset + value_offset,
                        );
                        self.emit_with_result(
                            block_idx,
                            Inst::Load {
                                ty: Ty::I64,
                                ptr: new_ptr,
                                align: None,
                                volatile: false,
                            },
                        )
                    };
                    let selected_val = self.emit_with_result(
                        block_idx,
                        Inst::Select {
                            ty: Ty::I64,
                            cond: key_match,
                            then_val: new_val,
                            else_val: old_val,
                        },
                    );
                    self.store_at_offset(
                        block_idx,
                        result_ptr,
                        idx * value_stride + value_offset,
                        selected_val,
                    );
                }
            }

            if total_slots == 1 {
                let first = self.load_at_offset(block_idx, result_ptr, 0);
                self.store_single_slot_compact_result(
                    block_idx,
                    rd,
                    first,
                    super::CompactStateSlot::raw(result_ptr, 0),
                    result_function_shape.clone(),
                )?;
            } else {
                self.store_compact_aggregate_result(
                    block_idx,
                    rd,
                    result_ptr,
                    result_function_shape.clone(),
                )?;
            }
            return Ok(Some(block_idx));
        }

        if let Some(super::AggregateShape::Sequence { extent, element }) =
            self.aggregate_shapes.get(&func_reg).cloned()
        {
            let seq_shape = super::AggregateShape::Sequence {
                extent,
                element: element.clone(),
            };
            let seq_capacity = extent.capacity();
            let Some(element_shape) = element.as_deref() else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept: compact sequence r{func_reg} requires tracked element shape"
                )));
            };
            let element_stride = element_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept: compact sequence r{func_reg} requires fixed-width element shape, got {element_shape:?}"
                ))
            })?;
            let total_slots = seq_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncExcept: compact sequence r{func_reg} requires fixed-width shape, got {seq_shape:?}"
                ))
            })?;
            let source = self.materialize_reg_as_compact_source(block_idx, func_reg, &seq_shape)?;
            let block_idx = source.block_idx;
            let source_slot = source.slot;
            if let Some(path_raw) = self.scalar_of(path_reg) {
                let replace_idx = path_raw
                    .checked_sub(1)
                    .and_then(|relative_key| u32::try_from(relative_key).ok());
                if match replace_idx {
                    Some(idx) => idx >= seq_capacity,
                    None => true,
                } {
                    self.store_compact_identity_result(block_idx, rd, source_slot, seq_shape)?;
                    return Ok(Some(block_idx));
                }
            }
            let replacement =
                self.compact_value_source_for_reg(block_idx, val_reg, element_shape)?;
            let block_idx = replacement.block_idx;
            let replacement_source = replacement.slot;
            let result_ptr = self.alloc_aggregate(block_idx, total_slots);
            for slot in 0..total_slots {
                let old_val = self.load_at_offset(
                    block_idx,
                    source_slot.source_ptr,
                    source_slot.offset + slot,
                );
                self.store_at_offset(block_idx, result_ptr, slot, old_val);
            }

            let runtime_len =
                self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset);
            let path_val = if let Some(path_raw) = self.scalar_of(path_reg) {
                self.emit_i64_const(block_idx, path_raw)
            } else {
                let raw = self.load_reg(block_idx, path_reg)?;
                // WP-18 follow-on: decode a tagged-scalar-union index key
                // before the 1-based sequence bounds guards below.
                self.decode_scalar_key_reg_raw_value(block_idx, path_reg, raw)
            };
            let zero = self.emit_i64_const(block_idx, 0);
            let one = self.emit_i64_const(block_idx, 1);
            let capacity = self.emit_i64_const(block_idx, i64::from(seq_capacity));

            let len_nonnegative = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Sge,
                    ty: Ty::I64,
                    lhs: runtime_len,
                    rhs: zero,
                },
            );

            let check_capacity_blk = self.new_aux_block("compact_seq_except_check_capacity");
            let check_path_lo_blk = self.new_aux_block("compact_seq_except_check_path_lo");
            let check_path_hi_blk = self.new_aux_block("compact_seq_except_check_path_hi");
            let update_blk = self.new_aux_block("compact_seq_except_update");
            let merge_blk = self.new_aux_block("compact_seq_except_merge");
            let check_capacity_id = self.block_id_of(check_capacity_blk);
            let check_path_lo_id = self.block_id_of(check_path_lo_blk);
            let check_path_hi_id = self.block_id_of(check_path_hi_blk);
            let update_id = self.block_id_of(update_blk);
            let merge_id = self.block_id_of(merge_blk);

            self.emit(
                block_idx,
                InstrNode::new(Inst::CondBr {
                    cond: len_nonnegative,
                    then_target: check_capacity_id,
                    then_args: vec![],
                    else_target: merge_id,
                    else_args: vec![],
                }),
            );

            let len_within_capacity = self.emit_with_result(
                check_capacity_blk,
                Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: runtime_len,
                    rhs: capacity,
                },
            );
            self.emit(
                check_capacity_blk,
                InstrNode::new(Inst::CondBr {
                    cond: len_within_capacity,
                    then_target: check_path_lo_id,
                    then_args: vec![],
                    else_target: merge_id,
                    else_args: vec![],
                }),
            );

            let path_ge_one = self.emit_with_result(
                check_path_lo_blk,
                Inst::ICmp {
                    op: ICmpOp::Sge,
                    ty: Ty::I64,
                    lhs: path_val,
                    rhs: one,
                },
            );
            self.emit(
                check_path_lo_blk,
                InstrNode::new(Inst::CondBr {
                    cond: path_ge_one,
                    then_target: check_path_hi_id,
                    then_args: vec![],
                    else_target: merge_id,
                    else_args: vec![],
                }),
            );

            let path_le_len = self.emit_with_result(
                check_path_hi_blk,
                Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: path_val,
                    rhs: runtime_len,
                },
            );
            self.emit(
                check_path_hi_blk,
                InstrNode::new(Inst::CondBr {
                    cond: path_le_len,
                    then_target: update_id,
                    then_args: vec![],
                    else_target: merge_id,
                    else_args: vec![],
                }),
            );

            let rel_idx = self.emit_with_result(
                update_blk,
                Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: path_val,
                    rhs: one,
                },
            );
            let first_elem = self.emit_i64_const(update_blk, 1);
            let elem_offset = if element_stride == 1 {
                rel_idx
            } else {
                let stride = self.emit_i64_const(update_blk, i64::from(element_stride));
                self.emit_with_result(
                    update_blk,
                    Inst::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::I64,
                        lhs: rel_idx,
                        rhs: stride,
                    },
                )
            };
            let elem_base_slot = self.emit_with_result(
                update_blk,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: first_elem,
                    rhs: elem_offset,
                },
            );
            for value_offset in 0..element_stride {
                let new_val = self.load_at_offset(
                    update_blk,
                    replacement_source.source_ptr,
                    replacement_source.offset + value_offset,
                );
                let target_slot = if value_offset == 0 {
                    elem_base_slot
                } else {
                    let offset = self.emit_i64_const(update_blk, i64::from(value_offset));
                    self.emit_with_result(
                        update_blk,
                        Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: elem_base_slot,
                            rhs: offset,
                        },
                    )
                };
                self.store_at_dynamic_offset(update_blk, result_ptr, target_slot, new_val);
            }
            self.emit(
                update_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );

            self.store_compact_aggregate_result(merge_blk, rd, result_ptr, seq_shape)?;
            return Ok(Some(merge_blk));
        }

        self.reject_raw_compact_pointer_fallback(func_reg, "FuncExcept")?;
        let path_val = self.load_reg(block_idx, path_reg)?;
        // WP-18 follow-on: pair-list keys are stored as raw member values;
        // decode a tagged-scalar-union index key before the compare scan.
        let path_val = self.decode_scalar_key_reg_raw_value(block_idx, path_reg, path_val);
        let new_val = self.load_reg(block_idx, val_reg)?;
        let func_ptr = self.load_reg_as_ptr(block_idx, func_reg)?;
        let pair_count = self.load_at_offset(block_idx, func_ptr, 0);

        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);
        let two = self.emit_i64_const(block_idx, 2);

        // Allocate new function: 1 + 2*pair_count slots
        let pairs_x2 = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: pair_count,
                rhs: two,
            },
        );
        let total = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: pairs_x2,
                rhs: one,
            },
        );
        let result_ptr = if let Some(pair_count_u32) = self.const_set_sizes.get(&func_reg).copied()
        {
            self.alloc_aggregate(block_idx, 1 + (2 * pair_count_u32))
        } else {
            self.alloc_dynamic_i64_slots(block_idx, total)
        };

        // Store pair_count in new function
        self.store_at_offset(block_idx, result_ptr, 0, pair_count);

        // Loop: for i in 0..pair_count, copy key, conditionally replace value
        let i_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let loop_hdr = self.new_aux_block("fexcept_hdr");
        let loop_body = self.new_aux_block("fexcept_body");
        let loop_done = self.new_aux_block("fexcept_done");

        let hdr_id = self.block_id_of(loop_hdr);
        let body_id = self.block_id_of(loop_body);
        let done_id = self.block_id_of(loop_done);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: hdr_id,
                args: vec![],
            }),
        );

        // Header: i < pair_count?
        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: pair_count,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp,
                then_target: body_id,
                then_args: vec![],
                else_target: done_id,
                else_args: vec![],
            }),
        );

        // Body: copy key, select value based on key match
        let i_val2 = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );

        // Source key slot: 1 + 2*i
        let src_key_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: two,
            },
        );
        let src_key_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: src_key_offset,
                rhs: one,
            },
        );
        let key = self.load_at_dynamic_offset(loop_body, func_ptr, src_key_slot);

        // Source value slot: 2 + 2*i
        let src_val_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: src_key_offset,
                rhs: two,
            },
        );
        let orig_val = self.load_at_dynamic_offset(loop_body, func_ptr, src_val_slot);

        // Select: if key == path then new_val else orig_val
        let key_match = self.emit_with_result(
            loop_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: key,
                rhs: path_val,
            },
        );
        let selected_val = self.emit_with_result(
            loop_body,
            Inst::Select {
                ty: Ty::I64,
                cond: key_match,
                then_val: new_val,
                else_val: orig_val,
            },
        );

        // Store key and selected value in result
        self.store_at_dynamic_offset(loop_body, result_ptr, src_key_slot, key);
        self.store_at_dynamic_offset(loop_body, result_ptr, src_val_slot, selected_val);

        // Increment
        let next_i = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        self.emit(
            loop_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_body,
            InstrNode::new(Inst::Br {
                target: hdr_id,
                args: vec![],
            }),
        );

        // Done
        self.store_reg_ptr(loop_done, rd, result_ptr)?;
        self.compact_state_slots.remove(&rd);
        self.mark_flat_funcdef_pair_list(rd);
        if let Some(shape) = self.aggregate_shapes.get(&func_reg).cloned() {
            self.aggregate_shapes.insert(rd, shape);
        } else {
            self.aggregate_shapes.remove(&rd);
        }
        if let Some(n) = self.const_set_sizes.get(&func_reg).copied() {
            self.record_set_size(rd, n);
        } else {
            self.const_set_sizes.remove(&rd);
        }
        self.const_scalar_values.remove(&rd);

        Ok(Some(loop_done))
    }

    /// Lower FuncDefBegin: initialize function builder, iterate domain.
    ///
    /// Allocates a function aggregate sized for the domain, then iterates
    /// over domain elements. For each element, the body evaluates the
    /// mapping expression, and LoopNext stores the (key, value) pair.
    ///
    /// Function layout: [pair_count, key1, val1, key2, val2, ...]
    ///
    /// CFG produced:
    ///   current_block -> header
    ///   header -> body_block (if i < len) | exit_block (if i >= len)
    pub(super) fn lower_func_def_begin(
        &mut self,
        pc: usize,
        block: usize,
        rd: u8,
        r_binding: u8,
        r_domain: u8,
        loop_end: i32,
    ) -> Result<Option<usize>, TrustIrError> {
        let exit_pc = self.resolve_forward_target(pc, loop_end, "FuncDefBegin")?;
        let body_pc = pc + 1;
        let exit_block = self.block_index_for_pc(exit_pc)?;
        let body_block = self.block_index_for_pc(body_pc)?;

        if let Some(range) = self.runtime_int_ranges.get(&r_domain).copied() {
            return self.lower_func_def_begin_runtime_int_range(
                block, rd, r_binding, r_domain, range, exit_block, body_block,
            );
        }

        // Load domain pointer and length.
        self.reject_compact_set_bitmask_powerset_iteration(r_domain, "FuncDefBegin")?;
        let binding_shape = self
            .aggregate_shapes
            .get(&r_domain)
            .and_then(super::binding_shape_from_domain);
        let domain_ptr =
            self.load_reg_as_ptr_or_materialize_raw_compact(block, r_domain, "FuncDefBegin")?;
        let domain_len = self.load_at_offset(block, domain_ptr, 0);

        let zero = self.emit_i64_const(block, 0);
        let one = self.emit_i64_const(block, 1);
        let two = self.emit_i64_const(block, 2);

        // Allocate function aggregate: 1 + 2*domain_len slots
        let pairs_x2 = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: domain_len,
                rhs: two,
            },
        );
        let total = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: pairs_x2,
                rhs: one,
            },
        );
        let static_domain_capacity = self.const_set_sizes.get(&r_domain).copied().or_else(|| {
            self.aggregate_shapes
                .get(&r_domain)
                .and_then(super::AggregateShape::finite_set_len_bound)
        });
        let func_ptr = if let Some(domain_len_u32) = static_domain_capacity {
            self.alloc_aggregate(block, 1 + (2 * domain_len_u32))
        } else {
            self.alloc_dynamic_i64_slots(block, total)
        };

        // Store pair_count = domain_len
        self.store_at_offset(block, func_ptr, 0, domain_len);

        // Store function pointer in rd now (so LoopNext can access it).
        self.store_reg_ptr(block, rd, func_ptr)?;
        self.compact_state_slots.remove(&rd);
        self.mark_flat_funcdef_pair_list(rd);

        // Allocate index counter, initialize to 0.
        let idx_alloca = self.emit_with_result(
            block,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        // Create loop header block.
        let header_block = self.new_aux_block("funcdef_header");
        let header_id = self.block_id_of(header_block);
        let body_id = self.block_id_of(body_block);
        let exit_id = self.block_id_of(exit_block);

        // Branch to header.
        self.emit(
            block,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        // Header: check i < domain_len.
        let cur_idx = self.emit_with_result(
            header_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let in_bounds = self.emit_with_result(
            header_block,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: domain_len,
            },
        );

        let load_block = self.new_aux_block("funcdef_load");
        let load_id = self.block_id_of(load_block);

        self.emit(
            header_block,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: load_id,
                then_args: vec![],
                else_target: exit_id,
                else_args: vec![],
            }),
        );

        // Load element: r_binding = domain[i+1] (skip length header).
        let cur_idx2 = self.emit_with_result(
            load_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot_idx = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx2,
                rhs: one,
            },
        );
        let elem = self.load_at_dynamic_offset(load_block, domain_ptr, slot_idx);
        self.store_reg_value(load_block, r_binding, elem)?;
        self.invalidate_reg_tracking(r_binding);
        if let Some(binding_shape) = binding_shape {
            // Flat-aggregate value ABI: a set/sequence slot holding a *boxed
            // compound* element (tuple/record/function — anything that is not a
            // single-slot scalar) stores a POINTER to that element's aggregate,
            // not an inlined value. The element we just loaded into `r_binding`
            // is therefore an aggregate base pointer; record that provenance so
            // a downstream `FuncApply`/`Domain`/etc. that needs to dereference
            // the tuple (e.g. GameOfLife's `grid[p]` with `p` ranging over the
            // 2-tuple set `Pos`) can `load_reg_as_ptr` it. Single-slot scalar
            // bindings keep their inlined value and no pointer provenance, so
            // `load_reg_as_ptr`'s scalar-rejection wall stays intact.
            if Self::binding_element_is_boxed_compound(&binding_shape) {
                self.mark_flat_aggregate_pointer(r_binding);
            }
            self.aggregate_shapes.insert(r_binding, binding_shape);
        }

        // Also store the key into the function aggregate at slot[1 + 2*i].
        let func_ptr_reload = self.load_reg_as_ptr(load_block, rd)?;
        let key_offset = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: cur_idx2,
                rhs: two,
            },
        );
        let key_slot = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: key_offset,
                rhs: one,
            },
        );
        self.store_at_dynamic_offset(load_block, func_ptr_reload, key_slot, elem);

        // Branch to the body block.
        self.emit(
            load_block,
            InstrNode::new(Inst::Br {
                target: body_id,
                args: vec![],
            }),
        );

        // Save loop state for LoopNext (use the shared builder-loop stack,
        // not quantifier_loops, because LoopNext does not carry rd or kind).
        self.loop_next_stack.push(LoopNextState {
            rd,
            kind: LoopNextKind::FuncDef,
            loop_state: QuantifierLoopState {
                idx_alloca,
                header_block,
                exit_block,
            },
            funcdef_capture: Some(super::FuncDefCaptureState {
                preheader_block: block,
                domain_len,
                static_domain_capacity,
            }),
            set_builder_dedup: None,
        });

        // FuncDef's per-key body writes to a distinct slot (slot 2+2*i in
        // the function aggregate) and reads only from its own binding
        // register. Iterations are independent → mark the loop header as
        // a `trust_ir.parallel_map` candidate.
        self.annotate_parallel_map(header_block);

        // If the domain size is compile-time known, also emit
        // `trust_ir.bounded_loop(n)`. If not, mark the function as unbounded
        // so Terminates is suppressed.
        if !self.annotate_loop_bound(header_block, r_domain) {
            self.mark_unbounded_loop();
        }

        // The function-aggregate value lives in rd; its cardinality equals
        // the domain size.
        if let Some(&n) = self.const_set_sizes.get(&r_domain) {
            let domain_lo = Self::contiguous_int_domain_lo(self.aggregate_shapes.get(&r_domain), n);
            let explicit_domain = if domain_lo.is_none() {
                self.aggregate_shapes
                    .get(&r_domain)
                    .and_then(super::explicit_function_domain_from_domain_shape)
            } else {
                None
            };
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::Function {
                    len: n,
                    domain_lo,
                    domain: explicit_domain,
                    value: None,
                },
            );
            self.record_set_size(rd, n);
        } else {
            self.aggregate_shapes.remove(&rd);
            self.const_set_sizes.remove(&rd);
        }
        self.const_scalar_values.remove(&rd);

        // Body block is the next PC's block — return None to let lower_body
        // transition to it naturally.
        Ok(None)
    }

    fn lower_func_def_begin_runtime_int_range(
        &mut self,
        block: usize,
        rd: u8,
        r_binding: u8,
        r_domain: u8,
        range: super::RuntimeIntRange,
        exit_block: usize,
        body_block: usize,
    ) -> Result<Option<usize>, TrustIrError> {
        let range_lo = self.load_reg(block, range.lo_reg)?;
        let range_hi = self.load_reg(block, range.hi_reg)?;
        let zero = self.emit_i64_const(block, 0);
        let one = self.emit_i64_const(block, 1);
        let two = self.emit_i64_const(block, 2);

        let is_empty = self.emit_with_result(
            block,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: range_hi,
                rhs: range_lo,
            },
        );
        let diff = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: range_hi,
                rhs: range_lo,
            },
        );
        let domain_len_nonempty = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: diff,
                rhs: one,
            },
        );
        let domain_len = self.emit_with_result(
            block,
            Inst::Select {
                ty: Ty::I64,
                cond: is_empty,
                then_val: zero,
                else_val: domain_len_nonempty,
            },
        );

        // Allocate function aggregate: 1 + 2*domain_len slots.
        let pairs_x2 = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: domain_len,
                rhs: two,
            },
        );
        let total = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: pairs_x2,
                rhs: one,
            },
        );
        let static_domain_capacity = self.const_set_sizes.get(&r_domain).copied().or_else(|| {
            self.aggregate_shapes
                .get(&r_domain)
                .and_then(super::AggregateShape::finite_set_len_bound)
        });
        let func_ptr = if let Some(domain_len_u32) = static_domain_capacity {
            self.alloc_aggregate(block, 1 + (2 * domain_len_u32))
        } else {
            self.alloc_dynamic_i64_slots(block, total)
        };
        self.store_at_offset(block, func_ptr, 0, domain_len);
        self.store_reg_ptr(block, rd, func_ptr)?;
        self.compact_state_slots.remove(&rd);
        self.mark_flat_funcdef_pair_list_with_info(
            rd,
            Some(super::FlatFuncDefPointerInfo {
                domain_lo: self.scalar_of(range.lo_reg),
                value: None,
                values_are_captured_compact: false,
            }),
        );

        let idx_alloca = self.emit_with_result(
            block,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let header_block = self.new_aux_block("funcdef_range_header");
        let header_id = self.block_id_of(header_block);
        let body_id = self.block_id_of(body_block);
        let exit_id = self.block_id_of(exit_block);
        self.emit(
            block,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        let cur_idx = self.emit_with_result(
            header_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let in_bounds = self.emit_with_result(
            header_block,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: domain_len,
            },
        );
        let load_block = self.new_aux_block("funcdef_range_load");
        let load_id = self.block_id_of(load_block);
        self.emit(
            header_block,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: load_id,
                then_args: vec![],
                else_target: exit_id,
                else_args: vec![],
            }),
        );

        let cur_idx2 = self.emit_with_result(
            load_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let elem = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: range_lo,
                rhs: cur_idx2,
            },
        );
        self.store_reg_value(load_block, r_binding, elem)?;
        self.invalidate_reg_tracking(r_binding);

        let func_ptr_reload = self.load_reg_as_ptr(load_block, rd)?;
        let key_offset = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: cur_idx2,
                rhs: two,
            },
        );
        let key_slot = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: key_offset,
                rhs: one,
            },
        );
        self.store_at_dynamic_offset(load_block, func_ptr_reload, key_slot, elem);
        self.emit(
            load_block,
            InstrNode::new(Inst::Br {
                target: body_id,
                args: vec![],
            }),
        );

        self.loop_next_stack.push(LoopNextState {
            rd,
            kind: LoopNextKind::FuncDef,
            loop_state: QuantifierLoopState {
                idx_alloca,
                header_block,
                exit_block,
            },
            funcdef_capture: Some(super::FuncDefCaptureState {
                preheader_block: block,
                domain_len,
                static_domain_capacity,
            }),
            set_builder_dedup: None,
        });

        self.annotate_parallel_map(header_block);
        if !self.annotate_loop_bound(header_block, r_domain) {
            self.mark_unbounded_loop();
        }

        if let Some(&n) = self.const_set_sizes.get(&r_domain) {
            let domain_lo = Self::contiguous_int_domain_lo(self.aggregate_shapes.get(&r_domain), n);
            let explicit_domain = if domain_lo.is_none() {
                self.aggregate_shapes
                    .get(&r_domain)
                    .and_then(super::explicit_function_domain_from_domain_shape)
            } else {
                None
            };
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::Function {
                    len: n,
                    domain_lo,
                    domain: explicit_domain,
                    value: None,
                },
            );
            self.record_set_size(rd, n);
        } else {
            self.aggregate_shapes.remove(&rd);
            self.const_set_sizes.remove(&rd);
        }
        self.const_scalar_values.remove(&rd);
        self.runtime_int_ranges.remove(&r_domain);

        Ok(None)
    }

    pub(super) fn lower_loop_next(
        &mut self,
        _pc: usize,
        block: usize,
        r_binding: u8,
        r_body: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        let state = self
            .loop_next_stack
            .pop()
            .ok_or_else(|| TrustIrError::Emission("LoopNext without matching Begin".to_owned()))?;

        match state.kind {
            LoopNextKind::FuncDef => {
                let capture = state.funcdef_capture.ok_or_else(|| {
                    TrustIrError::Emission("FuncDef LoopNext missing capture state".to_owned())
                })?;
                self.lower_loop_next_func_def(block, r_body, state.rd, state.loop_state, capture)
            }
            LoopNextKind::SetFilter => self.lower_loop_next_set_filter(
                block,
                r_binding,
                r_body,
                state.rd,
                state.loop_state,
            ),
            LoopNextKind::SetBuilder => self.lower_loop_next_set_builder(
                block,
                r_body,
                state.rd,
                state.loop_state,
                state.set_builder_dedup,
            ),
        }
    }

    fn insert_preheader_with_result(&mut self, block_idx: usize, inst: Inst) -> ValueId {
        let result = self.alloc_value();
        let node = InstrNode::new(inst).with_result(result);
        let func = &mut self.module.functions[self.func_idx];
        let body = &mut func.blocks[block_idx].body;
        let insert_idx = body
            .iter()
            .rposition(|node| node.is_terminator())
            .unwrap_or(body.len());
        body.insert(insert_idx, node);
        result
    }

    fn alloc_funcdef_capture_backing(
        &mut self,
        capture_state: super::FuncDefCaptureState,
        value_slots: u32,
    ) -> Result<ValueId, TrustIrError> {
        let count = if let Some(domain_capacity) = capture_state.static_domain_capacity {
            let total_slots = domain_capacity.checked_mul(value_slots).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncDef compact capture backing overflows: {domain_capacity} * {value_slots}"
                ))
            })?;
            self.insert_preheader_with_result(
                capture_state.preheader_block,
                Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(i128::from(total_slots)),
                },
            )
        } else {
            let value_slots = self.insert_preheader_with_result(
                capture_state.preheader_block,
                Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(i128::from(value_slots)),
                },
            );
            let total_slots = self.insert_preheader_with_result(
                capture_state.preheader_block,
                Inst::BinOp {
                    op: BinOp::Mul,
                    ty: Ty::I64,
                    lhs: capture_state.domain_len,
                    rhs: value_slots,
                },
            );
            self.insert_preheader_with_result(
                capture_state.preheader_block,
                Inst::Cast {
                    op: CastOp::Trunc,
                    src_ty: Ty::I64,
                    dst_ty: Ty::I32,
                    operand: total_slots,
                },
            )
        };

        Ok(self.insert_preheader_with_result(
            capture_state.preheader_block,
            Inst::Alloca {
                ty: Ty::I64,
                count: Some(count),
                align: None,
            },
        ))
    }

    /// Lower LoopNext for FuncDef: store body result as value, advance iterator.
    ///
    /// Stores the body result (r_body) as the value for the current key in the
    /// function aggregate, then increments the index and branches back to the header.
    fn lower_loop_next_func_def(
        &mut self,
        block: usize,
        r_body: u8,
        rd: u8,
        loop_state: QuantifierLoopState,
        capture_state: super::FuncDefCaptureState,
    ) -> Result<Option<usize>, TrustIrError> {
        let mut block = block;
        let body_shape = self.aggregate_shapes.get(&r_body).cloned();
        let body_val = if let Some(shape) = body_shape
            .as_ref()
            .filter(|shape| Self::is_compact_compound_aggregate(shape))
        {
            let materialized = self.materialize_reg_as_compact_source(block, r_body, shape)?;
            block = materialized.block_idx;
            let slot_count = shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "FuncDef LoopNext compact body r{r_body} requires fixed-width shape, got {shape:?}"
                ))
            })?;
            let cur_idx = self.emit_with_result(
                block,
                Inst::Load {
                    ty: Ty::I64,
                    ptr: loop_state.idx_alloca,
                    align: None,
                    volatile: false,
                },
            );
            let backing_ptr = self.alloc_funcdef_capture_backing(capture_state, slot_count)?;
            let slot_count_value = self.emit_i64_const(block, i64::from(slot_count));
            let capture_offset = self.emit_with_result(
                block,
                Inst::BinOp {
                    op: BinOp::Mul,
                    ty: Ty::I64,
                    lhs: cur_idx,
                    rhs: slot_count_value,
                },
            );
            let capture_offset_i32 = self.emit_with_result(
                block,
                Inst::Cast {
                    op: CastOp::Trunc,
                    src_ty: Ty::I64,
                    dst_ty: Ty::I32,
                    operand: capture_offset,
                },
            );
            let capture_ptr = self.emit_with_result(
                block,
                Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: backing_ptr,
                    indices: vec![capture_offset_i32],
                    inbounds: false,
                },
            );
            for offset in 0..slot_count {
                let value = self.load_at_offset(
                    block,
                    materialized.slot.source_ptr,
                    materialized.slot.offset + offset,
                );
                self.store_at_offset(block, capture_ptr, offset, value);
            }
            self.ptr_to_i64(block, capture_ptr)
        } else {
            self.load_reg(block, r_body)?
        };
        let two = self.emit_i64_const(block, 2);
        let one = self.emit_i64_const(block, 1);

        // Store value at slot[2 + 2*i] in the function aggregate.
        let func_ptr = self.load_reg_as_ptr(block, rd)?;
        let cur_idx = self.emit_with_result(
            block,
            Inst::Load {
                ty: Ty::I64,
                ptr: loop_state.idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let val_offset = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: two,
            },
        );
        let val_slot = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: val_offset,
                rhs: two,
            },
        );
        self.store_at_dynamic_offset(block, func_ptr, val_slot, body_val);
        if let Some(shape) = self.aggregate_shapes.get(&r_body).cloned() {
            if let Some(info) = self.flat_funcdef_pointer_infos.get_mut(&rd) {
                info.values_are_captured_compact = Self::is_compact_compound_aggregate(&shape);
                info.value = Some(shape.clone());
            }
            if let Some(super::AggregateShape::Function { len, domain_lo, .. }) =
                self.aggregate_shapes.get(&rd)
            {
                self.aggregate_shapes.insert(
                    rd,
                    super::AggregateShape::Function {
                        len: *len,
                        domain_lo: *domain_lo,
                        domain: None,
                        value: Some(Box::new(shape)),
                    },
                );
            }
        }

        // Advance: increment index, branch to header.
        let next_idx = self.emit_with_result(
            block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: one,
            },
        );
        self.emit(
            block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: loop_state.idx_alloca,
                value: next_idx,
                align: None,
                volatile: false,
            }),
        );

        let header_id = self.block_id_of(loop_state.header_block);
        self.emit(
            block,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        // Return None — the exit block is the next PC's block, lower_body handles it.
        Ok(None)
    }
}
