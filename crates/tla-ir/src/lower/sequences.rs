// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Sequence, tuple, record, and builtin lowering: SeqNew, TupleNew, TupleGet,
//! RecordNew, RecordGet, Cardinality, Len, Head, Tail, Append.

use crate::TrustIrError;
use tla_jit_abi::JitRuntimeErrorKind;
use tla_value::Value;
use trust_ir::inst::*;
use trust_ir::ty::Ty;
use trust_ir::value::ValueId;
use trust_ir::Constant;
use trust_ir::InstrNode;

use super::Ctx;

impl<'cp> Ctx<'cp> {
    /// Lower SeqNew { rd, start, count }: build a sequence from consecutive regs.
    ///
    /// Same layout as sets: slot[0] = length, slot[1..=count] = elements.
    pub(super) fn lower_seq_new(
        &mut self,
        block_idx: usize,
        rd: u8,
        start: u8,
        count: u8,
    ) -> Result<(), TrustIrError> {
        let total_slots = u32::from(count) + 1;
        let agg_ptr = self.alloc_aggregate(block_idx, total_slots);

        let len_val = self.emit_i64_const(block_idx, i64::from(count));
        self.store_at_offset(block_idx, agg_ptr, 0, len_val);

        let mut element_shapes = Vec::with_capacity(usize::from(count));
        let mut tracked_shape = true;
        // Snapshot the elements' compile-time typed constants (if every one is
        // known) for the tuple-keyed compact FuncApply const-key ordinal fast
        // path. The values are captured NOW — they describe the immutable
        // aggregate built here, not the (rebindable) source registers.
        let mut const_elements: Option<Vec<tla_jit_abi::SetBitmaskElement>> =
            Some(Vec::with_capacity(usize::from(count)));
        for i in 0..count {
            let reg = start.checked_add(i).ok_or_else(|| {
                TrustIrError::Emission(format!("SeqNew register overflow: start={start} + i={i}"))
            })?;
            let elem = self.load_reg(block_idx, reg)?;
            self.store_at_offset(block_idx, agg_ptr, u32::from(i) + 1, elem);
            element_shapes.push(self.aggregate_shapes.get(&reg).cloned());
            tracked_shape &= element_shapes.last().is_some_and(Option::is_some);
            if let Some(elements) = const_elements.as_mut() {
                match self.const_scalar_domain_key_of(reg) {
                    Some(element) => elements.push(element),
                    None => const_elements = None,
                }
            }
        }

        self.store_reg_ptr(block_idx, rd, agg_ptr)?;
        self.compact_state_slots.remove(&rd);
        self.clear_flat_funcdef_pair_list(rd);
        self.const_scalar_values.remove(&rd);
        // (`store_reg_ptr` above cleared any previous entry for `rd`.)
        if let Some(elements) = const_elements.filter(|elements| !elements.is_empty()) {
            self.const_tuple_key_elements.insert(rd, elements);
        }
        // WP-ARGS: keep the PER-POSITION shapes, which `AggregateShape::Sequence`
        // below folds away into one uniform element shape (`None` for a
        // mixed-kind tuple). A `FlatValueLayout::Tuple` destination needs them to
        // prove each payload slot's lane. Only recorded when EVERY position is
        // tracked — a partial list would leave later offsets unprovable.
        if tracked_shape && !element_shapes.is_empty() {
            let per_position: Option<Vec<super::AggregateShape>> =
                element_shapes.iter().cloned().collect();
            if let Some(per_position) = per_position {
                self.tuple_element_shapes.insert(rd, per_position);
            }
        }
        if tracked_shape {
            let shape = super::AggregateShape::Sequence {
                extent: super::SequenceExtent::Exact(u32::from(count)),
                element: super::uniform_tuple_element_shape(&element_shapes),
            };
            if shape.compact_slot_count() == Some(total_slots) {
                self.compact_state_slots.insert(
                    rd,
                    super::CompactStateSlot::pointer_backed_in_block(agg_ptr, 0, block_idx),
                );
            }
            if let Some(len) = shape.tracked_len() {
                self.record_set_size(rd, len);
            } else {
                self.const_set_sizes.remove(&rd);
            }
            self.aggregate_shapes.insert(rd, shape);
        } else {
            self.aggregate_shapes.remove(&rd);
            self.const_set_sizes.remove(&rd);
        }
        Ok(())
    }

    /// Lower TupleNew { rd, start, count }: build a tuple from consecutive regs.
    ///
    /// Identical layout to sequences.
    pub(super) fn lower_tuple_new(
        &mut self,
        block_idx: usize,
        rd: u8,
        start: u8,
        count: u8,
    ) -> Result<(), TrustIrError> {
        self.lower_seq_new(block_idx, rd, start, count)
    }

    /// Lower TupleGet { rd, rs, idx }: get tuple element (1-indexed per TLA+).
    ///
    /// slot[idx] is the element (since slot[0] is length, 1-indexed access
    /// naturally maps to the array layout).
    pub(super) fn lower_tuple_get(
        &mut self,
        block_idx: usize,
        rd: u8,
        rs: u8,
        idx: u16,
    ) -> Result<(), TrustIrError> {
        let seq_ptr = self.load_reg_as_ptr(block_idx, rs)?;
        let elem = self.load_at_offset(block_idx, seq_ptr, u32::from(idx));
        self.store_reg_value(block_idx, rd, elem)?;
        self.compact_state_slots.remove(&rd);
        Ok(())
    }

    // =====================================================================
    // Record operations
    // =====================================================================

    /// Lower RecordNew { rd, fields_start, values_start, count }.
    ///
    /// Records are stored as flat arrays: slot[i] = value for field i.
    /// No length header needed since count is static.
    pub(super) fn lower_record_new(
        &mut self,
        block_idx: usize,
        rd: u8,
        fields_start: u16,
        values_start: u8,
        count: u8,
    ) -> Result<(), TrustIrError> {
        let agg_ptr = self.alloc_aggregate(block_idx, u32::from(count));

        for i in 0..count {
            let val = self.load_reg(block_idx, values_start + i)?;
            self.store_at_offset(block_idx, agg_ptr, u32::from(i), val);
        }

        self.store_reg_ptr(block_idx, rd, agg_ptr)?;
        self.compact_state_slots.remove(&rd);
        self.clear_flat_funcdef_pair_list(rd);

        if let Some(pool) = self.config.const_pool {
            let mut fields = Vec::with_capacity(usize::from(count));
            for i in 0..count {
                let field_name = match pool.get_value(fields_start + u16::from(i)) {
                    Value::String(name) => tla_core::intern_name(name),
                    _ => {
                        self.aggregate_shapes.remove(&rd);
                        self.const_set_sizes.remove(&rd);
                        return Ok(());
                    }
                };
                fields.push((
                    field_name,
                    self.aggregate_shapes
                        .get(&(values_start + i))
                        .cloned()
                        .map(Box::new),
                ));
            }
            let shape = super::AggregateShape::Record { fields };
            if shape.compact_slot_count() == Some(u32::from(count)) {
                self.compact_state_slots.insert(
                    rd,
                    super::CompactStateSlot::pointer_backed_in_block(agg_ptr, 0, block_idx),
                );
            } else {
                self.compact_state_slots.remove(&rd);
            }
            self.aggregate_shapes.insert(rd, shape);
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
        }

        Ok(())
    }

    /// Lower RecordSet { rd, fields_start, values_start, count }.
    ///
    /// Record sets are represented lazily as an array of field-domain values.
    /// Field names and domain shapes are tracked separately so SetIn can lower
    /// record membership without enumerating the Cartesian product.
    pub(super) fn lower_record_set(
        &mut self,
        block_idx: usize,
        rd: u8,
        fields_start: u16,
        values_start: u8,
        count: u8,
    ) -> Result<(), TrustIrError> {
        let pool = self.config.const_pool.ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "RecordSet: constant pool is required to resolve field names".to_owned(),
            )
        })?;

        let agg_ptr = self.alloc_aggregate(block_idx, u32::from(count));
        let mut fields = Vec::with_capacity(usize::from(count));

        for i in 0..count {
            let value_reg = values_start + i;
            let field_name = match pool.get_value(fields_start + u16::from(i)) {
                Value::String(name) => tla_core::intern_name(name),
                other => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "RecordSet: field constant at {} is not a string: {other:?}",
                        fields_start + u16::from(i)
                    )));
                }
            };
            let domain_shape = self
                .aggregate_shapes
                .get(&value_reg)
                .cloned()
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "RecordSet: field {field_name:?} domain in r{value_reg} has no tracked shape"
                    ))
                })?;
            let domain_value = self.load_reg(block_idx, value_reg)?;
            self.store_at_offset(block_idx, agg_ptr, u32::from(i), domain_value);
            fields.push((field_name, domain_shape));
        }

        self.store_reg_ptr(block_idx, rd, agg_ptr)?;
        self.compact_state_slots.remove(&rd);
        self.aggregate_shapes
            .insert(rd, super::AggregateShape::RecordSet { fields });
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);
        Ok(())
    }

    /// Lower RecordGet { rd, rs, field_idx }.
    ///
    /// Loads the value at the resolved field offset in the record aggregate.
    pub(super) fn lower_record_get(
        &mut self,
        block_idx: usize,
        rd: u8,
        rs: u8,
        field_idx: u16,
    ) -> Result<(), TrustIrError> {
        let field_name = super::record_get_field_name(self.config.const_pool, field_idx);
        let source_slot = self.compact_state_slot_for_use(block_idx, rs)?;
        let record_shape = match self.aggregate_shapes.get(&rs).cloned() {
            Some(record_shape @ super::AggregateShape::Record { .. }) => Some(record_shape),
            Some(other) => Some(other),
            None => source_slot.and_then(|slot| {
                self.tracked_record_shape_from_raw_state_sub_slot(slot, field_name, field_idx)
            }),
        };
        let result_shape =
            super::record_get_shape(record_shape.as_ref(), self.config.const_pool, field_idx);

        if let (Some(source_slot), Some(record_shape @ super::AggregateShape::Record { .. })) =
            (source_slot, record_shape.clone())
        {
            let (field_offset, field_shape, field_desc) = if let Some(field_name) = field_name {
                let Some((field_offset, field_shape)) =
                    record_shape.compact_record_field(field_name)
                else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "RecordGet: field {:?} has no fixed compact offset in r{rs}",
                        tla_core::resolve_name_id(field_name)
                    )));
                };
                (
                    field_offset,
                    field_shape,
                    format!("{:?}", tla_core::resolve_name_id(field_name)),
                )
            } else {
                let Some((field_offset, field_shape)) =
                    record_shape.compact_record_field_at_index(field_idx)
                else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "RecordGet: positional field {field_idx} has no fixed compact offset in r{rs}"
                    )));
                };
                (
                    field_offset,
                    field_shape,
                    format!("positional field {field_idx}"),
                )
            };
            let Some(field_shape) = field_shape else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "RecordGet: {field_desc} in r{rs} has no tracked shape"
                )));
            };
            let field_stride = field_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "RecordGet: {field_desc} in r{rs} requires fixed-width shape, got {field_shape:?}"
                ))
            })?;
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
                let source_base =
                    self.emit_i64_const(block_idx, i64::from(source_slot.offset + field_offset));
                let result_ptr = self.copy_compact_slots_from_dynamic_base(
                    block_idx,
                    source_slot.source_ptr,
                    source_base,
                    field_stride,
                );
                self.store_compact_aggregate_result(block_idx, rd, result_ptr, field_shape)?;
            }
            return Ok(());
        }

        self.reject_raw_compact_pointer_fallback(rs, "RecordGet")?;
        let rec_ptr = self.load_reg_as_ptr(block_idx, rs)?;
        let offset = if let (Some(field_name), Some(record_shape)) = (field_name, record_shape) {
            record_shape
                .record_field(field_name)
                .map(|(offset, _)| offset)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "RecordGet: field {:?} is not present in r{rs}",
                        tla_core::resolve_name_id(field_name)
                    ))
                })?
        } else {
            u32::from(field_idx)
        };
        let val = self.load_at_offset(block_idx, rec_ptr, offset);
        let field_source_slot =
            super::CompactStateSlot::pointer_backed_in_block(rec_ptr, offset, block_idx);
        if let Some(shape) = result_shape {
            if super::Ctx::is_single_slot_flat_aggregate_value(&shape) {
                self.store_single_slot_compact_result(
                    block_idx,
                    rd,
                    val,
                    field_source_slot,
                    shape,
                )?;
            } else {
                self.store_reg_value(block_idx, rd, val)?;
                self.record_aggregate_shape(rd, Some(shape));
            }
        } else {
            self.store_reg_value(block_idx, rd, val)?;
            self.record_aggregate_shape(rd, None);
            self.compact_state_slots.insert(rd, field_source_slot);
        }
        Ok(())
    }

    // =====================================================================
    // Cardinality (CallBuiltin Cardinality)
    // =====================================================================

    /// Lower Cardinality: returns the length field (slot 0) of a set aggregate.
    pub(super) fn lower_cardinality(
        &mut self,
        block_idx: usize,
        rd: u8,
        set_reg: u8,
    ) -> Result<(), TrustIrError> {
        if let Some(super::AggregateShape::SetBitmask {
            universe_len,
            universe,
        }) = self.aggregate_shapes.get(&set_reg).cloned()
        {
            if matches!(universe, super::SetBitmaskUniverse::Unknown) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Cardinality: compact SetBitmask r{set_reg} requires exact universe metadata"
                )));
            }
            let valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, "Cardinality")?;
            let raw_mask = self.load_reg(block_idx, set_reg)?;
            let valid_mask_value = self.emit_i64_const(block_idx, valid_mask);
            let mask = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: raw_mask,
                    rhs: valid_mask_value,
                },
            );
            let count = self.emit_with_result(
                block_idx,
                Inst::UnOp {
                    op: UnOp::CtPop,
                    ty: Ty::I64,
                    operand: mask,
                },
            );
            self.store_reg_value(block_idx, rd, count)?;
            self.compact_state_slots.remove(&rd);
            return Ok(());
        }

        // RecordSetBitmask cardinality (RecordSetBitmask step 3/5): a record-set
        // is a multi-slot bitmask over its record universe, so |S| is the sum of
        // the per-slot popcounts. This is byte-exact against the interpreter's
        // count of present bits — the same universe indexing the `set_ops`
        // membership / union / diff lowering uses. Each slot is loaded from the
        // pointer-backed compact region (never `IntToPtr`-dereferenced) and
        // AND-ed with its per-slot valid mask (defensive; the region is stored
        // canonical) so out-of-universe bits can never inflate the count.
        if let Some(super::AggregateShape::RecordSetBitmask {
            universe_len,
            slot_count,
            universe,
        }) = self.aggregate_shapes.get(&set_reg).cloned()
        {
            let slots = self.record_set_bitmask_operand_slots(
                block_idx,
                set_reg,
                universe_len,
                slot_count,
                &universe,
                "Cardinality",
            )?;
            let mut total: Option<ValueId> = None;
            for (slot_index, slot_value) in slots.iter().enumerate() {
                let valid_mask =
                    super::record_set_bitmask_slot_valid_mask_ir(universe_len, slot_index)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "Cardinality: RecordSetBitmask slot {slot_index} out of range for \
                                 universe_len {universe_len}"
                            ))
                        })?;
                let valid_mask_val = self.emit_i64_const(block_idx, valid_mask as i64);
                let masked = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: *slot_value,
                        rhs: valid_mask_val,
                    },
                );
                let popcount = self.emit_with_result(
                    block_idx,
                    Inst::UnOp {
                        op: UnOp::CtPop,
                        ty: Ty::I64,
                        operand: masked,
                    },
                );
                total = Some(match total {
                    None => popcount,
                    Some(prev) => self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: prev,
                            rhs: popcount,
                        },
                    ),
                });
            }
            let count = total.unwrap_or_else(|| self.emit_i64_const(block_idx, 0));
            self.store_reg_value(block_idx, rd, count)?;
            self.compact_state_slots.remove(&rd);
            return Ok(());
        }

        let set_ptr = self.load_reg_as_ptr(block_idx, set_reg)?;
        let len = self.load_at_offset(block_idx, set_ptr, 0);
        self.store_reg_value(block_idx, rd, len)?;
        self.compact_state_slots.remove(&rd);
        Ok(())
    }

    // =====================================================================
    // Sequence builtins (Len, Head, Tail, Append)
    // =====================================================================

    /// Lower Len(seq): returns slot[0] of the sequence aggregate.
    pub(super) fn lower_seq_len(
        &mut self,
        block_idx: usize,
        rd: u8,
        seq_reg: u8,
    ) -> Result<(), TrustIrError> {
        if let Some(source_slot) = self.compact_state_slot_for_use(block_idx, seq_reg)? {
            let source_shape = self.aggregate_shapes.get(&seq_reg).cloned();
            if !matches!(source_shape, Some(super::AggregateShape::Sequence { .. })) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Len: compact source r{seq_reg} requires a tracked sequence shape, got {source_shape:?}"
                )));
            }
            let len = self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset);
            self.store_reg_value(block_idx, rd, len)?;
            self.compact_state_slots.remove(&rd);
            return Ok(());
        }

        let seq_ptr = self.load_reg_as_ptr(block_idx, seq_reg)?;
        let len = self.load_at_offset(block_idx, seq_ptr, 0);
        self.store_reg_value(block_idx, rd, len)?;
        self.compact_state_slots.remove(&rd);
        Ok(())
    }

    /// Lower Head(seq): returns slot[1] of the sequence aggregate.
    pub(super) fn lower_seq_head(
        &mut self,
        block_idx: usize,
        rd: u8,
        seq_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        if let Some(source_slot) = self.compact_state_slot_for_use(block_idx, seq_reg)? {
            let source_shape = self.aggregate_shapes.get(&seq_reg).cloned();
            let Some(super::AggregateShape::Sequence {
                extent,
                element: Some(element_shape),
            }) = source_shape
            else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Head: compact source r{seq_reg} requires a tracked sequence element shape, got {source_shape:?}"
                )));
            };
            let element_stride = element_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "Head: compact sequence r{seq_reg} requires fixed-width element shape, got {element_shape:?}"
                ))
            })?;
            let capacity = extent.capacity();
            let head_offset = source_slot.offset.checked_add(1).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "Head: compact source slot overflows u32".to_owned(),
                )
            })?;

            let old_len =
                self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset);
            let zero = self.emit_i64_const(block_idx, 0);
            let non_empty = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: old_len,
                    rhs: zero,
                },
            );

            let check_capacity_blk = self.new_aux_block("compact_head_check_capacity");
            let ok_blk = self.new_aux_block("compact_head_ok");
            let error_blk = self.new_aux_block("compact_head_error");
            let check_capacity_id = self.block_id_of(check_capacity_blk);
            let ok_id = self.block_id_of(ok_blk);
            let error_id = self.block_id_of(error_blk);
            self.emit(
                block_idx,
                InstrNode::new(Inst::CondBr {
                    cond: non_empty,
                    then_target: check_capacity_id,
                    then_args: vec![],
                    else_target: error_id,
                    else_args: vec![],
                }),
            );

            let capacity_val = self.emit_i64_const(check_capacity_blk, i64::from(capacity));
            let within_capacity = self.emit_with_result(
                check_capacity_blk,
                Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: old_len,
                    rhs: capacity_val,
                },
            );
            self.emit(
                check_capacity_blk,
                InstrNode::new(Inst::CondBr {
                    cond: within_capacity,
                    then_target: ok_id,
                    then_args: vec![],
                    else_target: error_id,
                    else_args: vec![],
                }),
            );
            self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);

            if element_stride == 1 {
                let head = self.load_at_offset(ok_blk, source_slot.source_ptr, head_offset);
                self.store_single_slot_compact_result(
                    ok_blk,
                    rd,
                    head,
                    super::CompactStateSlot::raw(source_slot.source_ptr, head_offset),
                    *element_shape,
                )?;
            } else {
                let head_base = self.emit_i64_const(ok_blk, i64::from(head_offset));
                let head_ptr = self.copy_compact_slots_from_dynamic_base(
                    ok_blk,
                    source_slot.source_ptr,
                    head_base,
                    element_stride,
                );
                self.store_compact_aggregate_result(ok_blk, rd, head_ptr, *element_shape)?;
            }
            return Ok(Some(ok_blk));
        }

        let seq_ptr = self.load_reg_as_ptr(block_idx, seq_reg)?;
        let head = self.load_at_offset(block_idx, seq_ptr, 1);
        self.store_reg_value(block_idx, rd, head)?;
        self.compact_state_slots.remove(&rd);
        Ok(Some(block_idx))
    }

    /// Lower Tail(seq): creates a new sequence with all elements except the first.
    ///
    /// Result: slot[0] = len-1, slot[1..] = original slot[2..].
    pub(super) fn lower_seq_tail(
        &mut self,
        block_idx: usize,
        rd: u8,
        seq_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        if let (
            Some(source_slot),
            Some(super::AggregateShape::Sequence {
                extent,
                element: Some(element_shape),
            }),
        ) = (
            self.compact_state_slot_for_use(block_idx, seq_reg)?,
            self.aggregate_shapes.get(&seq_reg).cloned(),
        ) {
            let capacity = extent.capacity();
            let element_stride = element_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "Tail: compact sequence r{seq_reg} requires fixed-width element shape, got {element_shape:?}"
                ))
            })?;
            let result_shape = super::AggregateShape::Sequence {
                extent,
                element: Some(element_shape.clone()),
            };
            let total_slots = result_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "Tail: compact result sequence requires fixed-width shape, got {result_shape:?}"
                ))
            })?;
            let old_len =
                self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset);
            let zero = self.emit_i64_const(block_idx, 0);
            let one = self.emit_i64_const(block_idx, 1);
            let non_empty = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: old_len,
                    rhs: zero,
                },
            );

            let check_capacity_blk = self.new_aux_block("compact_tail_check_capacity");
            let ok_blk = self.new_aux_block("compact_tail_ok");
            let error_blk = self.new_aux_block("compact_tail_error");
            let check_capacity_id = self.block_id_of(check_capacity_blk);
            let ok_id = self.block_id_of(ok_blk);
            let error_id = self.block_id_of(error_blk);
            self.emit(
                block_idx,
                InstrNode::new(Inst::CondBr {
                    cond: non_empty,
                    then_target: check_capacity_id,
                    then_args: vec![],
                    else_target: error_id,
                    else_args: vec![],
                }),
            );

            let capacity_val = self.emit_i64_const(check_capacity_blk, i64::from(capacity));
            let within_capacity = self.emit_with_result(
                check_capacity_blk,
                Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: old_len,
                    rhs: capacity_val,
                },
            );
            self.emit(
                check_capacity_blk,
                InstrNode::new(Inst::CondBr {
                    cond: within_capacity,
                    then_target: ok_id,
                    then_args: vec![],
                    else_target: error_id,
                    else_args: vec![],
                }),
            );
            self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);

            let result_ptr = self.alloc_aggregate(ok_blk, total_slots);
            let new_len = self.emit_with_result(
                ok_blk,
                Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: old_len,
                    rhs: one,
                },
            );
            self.store_at_offset(ok_blk, result_ptr, 0, new_len);

            for elem_idx in 0..capacity {
                let elem_idx_val = self.emit_i64_const(ok_blk, i64::from(elem_idx));
                let is_live = self.emit_with_result(
                    ok_blk,
                    Inst::ICmp {
                        op: ICmpOp::Slt,
                        ty: Ty::I64,
                        lhs: elem_idx_val,
                        rhs: new_len,
                    },
                );
                for value_offset in 0..element_stride {
                    let value = if elem_idx + 1 < capacity {
                        let source_offset =
                            1 + (elem_idx + 1).checked_mul(element_stride).ok_or_else(|| {
                                TrustIrError::UnsupportedOpcode(
                                    "Tail: compact source slot overflows u32".to_owned(),
                                )
                            })? + value_offset;
                        self.load_at_offset(
                            ok_blk,
                            source_slot.source_ptr,
                            source_slot.offset + source_offset,
                        )
                    } else {
                        zero
                    };
                    let value = if elem_idx + 1 < capacity {
                        self.emit_with_result(
                            ok_blk,
                            Inst::Select {
                                ty: Ty::I64,
                                cond: is_live,
                                then_val: value,
                                else_val: zero,
                            },
                        )
                    } else {
                        value
                    };
                    let dest_offset =
                        1 + elem_idx.checked_mul(element_stride).ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "Tail: compact destination slot overflows u32".to_owned(),
                            )
                        })? + value_offset;
                    self.store_at_offset(ok_blk, result_ptr, dest_offset, value);
                }
            }

            self.store_compact_aggregate_result(ok_blk, rd, result_ptr, result_shape)?;
            return Ok(Some(ok_blk));
        }

        if self.compact_state_slots.contains_key(&seq_reg) {
            let source_shape = self.aggregate_shapes.get(&seq_reg);
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Tail: compact source r{seq_reg} requires a tracked fixed-width sequence element shape, got {source_shape:?}"
            )));
        }

        let seq_ptr = self.load_reg_as_ptr(block_idx, seq_reg)?;
        let old_len = self.load_at_offset(block_idx, seq_ptr, 0);
        let one = self.emit_i64_const(block_idx, 1);
        let new_len = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: old_len,
                rhs: one,
            },
        );

        // Allocate new aggregate: new_len + 1 slots
        let total = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: new_len,
                rhs: one,
            },
        );
        let total_i32 = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::Trunc,
                src_ty: Ty::I64,
                dst_ty: Ty::I32,
                operand: total,
            },
        );
        let new_ptr = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: Some(total_i32),
                align: None,
            },
        );

        // Store new length
        self.store_at_offset(block_idx, new_ptr, 0, new_len);

        // Copy loop: for i in 0..new_len, new[i+1] = old[i+2]
        let zero = self.emit_i64_const(block_idx, 0);
        let two = self.emit_i64_const(block_idx, 2);
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

        let loop_hdr = self.new_aux_block("tail_hdr");
        let loop_body = self.new_aux_block("tail_body");
        let loop_done = self.new_aux_block("tail_done");

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

        // Header
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
                rhs: new_len,
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

        // Body
        let i_val2 = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let src_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: two,
            },
        );
        let elem = self.load_at_dynamic_offset(loop_body, seq_ptr, src_slot);
        let dst_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        self.store_at_dynamic_offset(loop_body, new_ptr, dst_slot, elem);
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
        self.store_reg_ptr(loop_done, rd, new_ptr)?;

        Ok(Some(loop_done))
    }

    /// Lower RemoveAt(seq, i): drop the element at 1-indexed position `i`.
    ///
    /// Result has length `Len(seq) - 1`. Output 0-indexed position `q` copies
    /// source position `q` when `q < i-1`, otherwise source position `q+1`
    /// (the element at `i-1` is skipped).
    ///
    /// Only the handle-based (flat `[len, e0, e1, ...]` pointer) layout is
    /// lowered natively; the compact fixed-width state-slot layout bails to the
    /// interpreter via `UnsupportedOpcode` so the dynamic skip index never has
    /// to be reconciled with multi-slot compact element strides. An out-of-range
    /// index emits a runtime error, mirroring `Tail` on the empty sequence, so
    /// the interpreter surfaces the precise `IndexOutOfBounds` diagnostic.
    pub(super) fn lower_seq_remove_at(
        &mut self,
        block_idx: usize,
        rd: u8,
        seq_reg: u8,
        idx_reg: u8,
        _result_shape: Option<super::AggregateShape>,
    ) -> Result<Option<usize>, TrustIrError> {
        // Compact state-slot sources are not supported: a dynamic removal index
        // combined with fixed-width multi-slot elements would require dynamic
        // multi-slot shifts. Fall back to the interpreter for soundness.
        if self.compact_state_slots.contains_key(&seq_reg) {
            let source_shape = self.aggregate_shapes.get(&seq_reg);
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "RemoveAt: compact source r{seq_reg} is not supported (dynamic index over compact slots), got {source_shape:?}"
            )));
        }

        let idx = self.load_reg(block_idx, idx_reg)?;
        let seq_ptr = self.load_reg_as_ptr(block_idx, seq_reg)?;
        let old_len = self.load_at_offset(block_idx, seq_ptr, 0);
        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);
        let two = self.emit_i64_const(block_idx, 2);

        // Bounds check: 1 <= idx <= old_len. Bail to the interpreter otherwise.
        // Use nested conditional branches (mirroring `lower_seq_tail`) rather
        // than a boolean `And`, matching the existing lowering convention.
        let check_upper_blk = self.new_aux_block("remove_at_check_upper");
        let ok_blk = self.new_aux_block("remove_at_ok");
        let error_blk = self.new_aux_block("remove_at_error");
        let check_upper_id = self.block_id_of(check_upper_blk);
        let ok_id = self.block_id_of(ok_blk);
        let error_id = self.block_id_of(error_blk);

        let ge_one = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: idx,
                rhs: one,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: ge_one,
                then_target: check_upper_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );
        let le_len = self.emit_with_result(
            check_upper_blk,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: idx,
                rhs: old_len,
            },
        );
        self.emit(
            check_upper_blk,
            InstrNode::new(Inst::CondBr {
                cond: le_len,
                then_target: ok_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );
        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);

        // new_len = old_len - 1; allocate new_len + 1 slots.
        let new_len = self.emit_with_result(
            ok_blk,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: old_len,
                rhs: one,
            },
        );
        let total = self.emit_with_result(
            ok_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: new_len,
                rhs: one,
            },
        );
        let total_i32 = self.emit_with_result(
            ok_blk,
            Inst::Cast {
                op: CastOp::Trunc,
                src_ty: Ty::I64,
                dst_ty: Ty::I32,
                operand: total,
            },
        );
        let new_ptr = self.emit_with_result(
            ok_blk,
            Inst::Alloca {
                ty: Ty::I64,
                count: Some(total_i32),
                align: None,
            },
        );
        self.store_at_offset(ok_blk, new_ptr, 0, new_len);

        // Copy loop: for q in 0..new_len, choose source slot.
        //   src_slot = (q + 1 < idx) ? (q + 1) : (q + 2)
        //   dst_slot = q + 1
        let q_alloca = self.emit_with_result(
            ok_blk,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            ok_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: q_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let loop_hdr = self.new_aux_block("remove_at_hdr");
        let loop_body = self.new_aux_block("remove_at_body");
        let loop_done = self.new_aux_block("remove_at_done");
        let hdr_id = self.block_id_of(loop_hdr);
        let body_id = self.block_id_of(loop_body);
        let done_id = self.block_id_of(loop_done);

        self.emit(
            ok_blk,
            InstrNode::new(Inst::Br {
                target: hdr_id,
                args: vec![],
            }),
        );

        // Header: q < new_len ?
        let q_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: q_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: q_val,
                rhs: new_len,
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

        // Body
        let q_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: q_alloca,
                align: None,
                volatile: false,
            },
        );
        let dst_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: q_body,
                rhs: one,
            },
        );
        // before_pivot = (q + 1) < idx  <=>  dst_slot < idx
        let before_pivot = self.emit_with_result(
            loop_body,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: dst_slot,
                rhs: idx,
            },
        );
        let src_lo = dst_slot; // q + 1
        let src_hi = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: q_body,
                rhs: two,
            },
        );
        let src_slot = self.emit_with_result(
            loop_body,
            Inst::Select {
                ty: Ty::I64,
                cond: before_pivot,
                then_val: src_lo,
                else_val: src_hi,
            },
        );
        let elem = self.load_at_dynamic_offset(loop_body, seq_ptr, src_slot);
        self.store_at_dynamic_offset(loop_body, new_ptr, dst_slot, elem);
        let next_q = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: q_body,
                rhs: one,
            },
        );
        self.emit(
            loop_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: q_alloca,
                value: next_q,
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
        self.store_reg_ptr(loop_done, rd, new_ptr)?;
        Ok(Some(loop_done))
    }

    /// Lower Append(seq, elem): creates a new sequence with elem appended.
    ///
    /// Result: slot[0] = old_len+1, slot[1..=old_len] = original, slot[old_len+1] = elem.
    pub(super) fn lower_seq_append(
        &mut self,
        block_idx: usize,
        rd: u8,
        seq_reg: u8,
        elem_reg: u8,
        result_shape: Option<super::AggregateShape>,
    ) -> Result<Option<usize>, TrustIrError> {
        let source_shape = self.aggregate_shapes.get(&seq_reg).cloned();
        let elem_shape = self.aggregate_shapes.get(&elem_reg).cloned();

        // Statically-empty compact source: a `Sequence { extent, .. }` whose
        // capacity is 0 is provably always empty (its compact slot is sized for
        // zero elements -- just the length header), so `Append(s, e)` is
        // unconditionally the single-element sequence `<<e>>`. The source's
        // tracked element shape is irrelevant (no source element ever exists),
        // which is exactly why the general compact path below rejects this case
        // when the appended element's shape disagrees with the stale source
        // element shape (e.g. Huang appends a record to a `Capacity(0)`
        // `Sequence<Int>`). The result is fully determined by the appended
        // element, so flat-materialize `[1, <e slots>]` and produce an
        // `Exact(1)` sequence of the appended element's shape.
        if let (
            Some(source_slot),
            Some(super::AggregateShape::Sequence {
                extent: source_extent,
                ..
            }),
            Some(appended_element),
        ) = (
            self.compact_state_slot_for_use(block_idx, seq_reg)?,
            source_shape.clone(),
            elem_shape.clone(),
        ) {
            if source_extent.capacity() == 0 {
                let element_stride =
                    appended_element.compact_slot_count().ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "Append: empty compact source r{seq_reg} requires a fixed-width appended element shape, got {appended_element:?}"
                        ))
                    })?;
                let result_shape = super::AggregateShape::Sequence {
                    extent: super::SequenceExtent::Exact(1),
                    element: Some(Box::new(appended_element.clone())),
                };
                let result_slots = result_shape.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Append: empty compact result sequence requires fixed-width shape, got {result_shape:?}"
                    ))
                })?;

                // Defensive soundness guard: the source length must be 0 (the
                // compact slot can hold no elements). If a future shape
                // inference ever mislabels a non-empty sequence as `Capacity(0)`
                // this routes to the interpreter rather than corrupting state.
                let old_len =
                    self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset);
                let zero = self.emit_i64_const(block_idx, 0);
                let is_empty = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: old_len,
                        rhs: zero,
                    },
                );
                let ok_blk = self.new_aux_block("compact_append_empty_ok");
                let error_blk = self.new_aux_block("compact_append_empty_error");
                let ok_id = self.block_id_of(ok_blk);
                let error_id = self.block_id_of(error_blk);
                self.emit(
                    block_idx,
                    InstrNode::new(Inst::CondBr {
                        cond: is_empty,
                        then_target: ok_id,
                        then_args: vec![],
                        else_target: error_id,
                        else_args: vec![],
                    }),
                );
                self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);

                let result_ptr = self.alloc_aggregate(ok_blk, result_slots);
                let one = self.emit_i64_const(ok_blk, 1);
                self.store_at_offset(ok_blk, result_ptr, 0, one);

                let elem_materialized =
                    self.materialize_reg_as_compact_source(ok_blk, elem_reg, &appended_element)?;
                let ok_blk = elem_materialized.block_idx;
                let elem_source = elem_materialized.slot;
                for value_offset in 0..element_stride {
                    let value = self.load_at_offset(
                        ok_blk,
                        elem_source.source_ptr,
                        elem_source.offset + value_offset,
                    );
                    self.store_at_offset(ok_blk, result_ptr, 1 + value_offset, value);
                }

                self.store_compact_aggregate_result(ok_blk, rd, result_ptr, result_shape)?;
                return Ok(Some(ok_blk));
            }
        }

        let compact_result_shape = match (&source_shape, elem_shape.as_ref(), result_shape.clone())
        {
            (
                Some(super::AggregateShape::Sequence {
                    extent: source_extent,
                    element: Some(source_element),
                }),
                Some(elem_shape),
                Some(super::AggregateShape::Sequence {
                    extent: result_extent,
                    element: None,
                }),
            ) if result_extent == *source_extent
                && Self::compatible_compact_materialization_value(elem_shape, source_element) =>
            {
                Some(super::AggregateShape::Sequence {
                    extent: result_extent,
                    element: Some(source_element.clone()),
                })
            }
            _ => result_shape.clone(),
        };

        if let (
            Some(source_slot),
            Some(super::AggregateShape::Sequence {
                extent: source_extent,
                element: Some(source_element),
            }),
            Some(super::AggregateShape::Sequence {
                extent: result_extent,
                element: Some(result_element),
            }),
        ) = (
            self.compact_state_slot_for_use(block_idx, seq_reg)?,
            source_shape.clone(),
            compact_result_shape.clone(),
        ) {
            let result_capacity = result_extent.capacity();
            let preserves_capacity = result_extent == source_extent;
            let widens_exact_by_one = match (source_extent, result_extent) {
                (
                    super::SequenceExtent::Exact(source_len),
                    super::SequenceExtent::Exact(result_len),
                ) => source_len.checked_add(1) == Some(result_len),
                _ => false,
            };
            if (preserves_capacity || widens_exact_by_one) && *source_element == *result_element {
                let result_shape = super::AggregateShape::Sequence {
                    extent: result_extent,
                    element: Some(source_element.clone()),
                };
                let source_slots = super::AggregateShape::Sequence {
                    extent: source_extent,
                    element: Some(source_element.clone()),
                }
                .compact_slot_count()
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Append: compact source sequence r{seq_reg} requires fixed-width shape"
                    ))
                })?;
                let result_slots = result_shape.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Append: compact result sequence requires fixed-width shape, got {result_shape:?}"
                    ))
                })?;
                let element_stride = source_element.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Append: compact element requires fixed-width shape, got {source_element:?}"
                    ))
                })?;

                let old_len =
                    self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset);
                let zero = self.emit_i64_const(block_idx, 0);
                let one = self.emit_i64_const(block_idx, 1);
                let non_negative = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Sge,
                        ty: Ty::I64,
                        lhs: old_len,
                        rhs: zero,
                    },
                );

                let check_capacity_blk = self.new_aux_block("compact_append_check_capacity");
                let ok_blk = self.new_aux_block("compact_append_ok");
                let error_blk = self.new_aux_block("compact_append_error");
                let check_capacity_id = self.block_id_of(check_capacity_blk);
                let ok_id = self.block_id_of(ok_blk);
                let error_id = self.block_id_of(error_blk);
                self.emit(
                    block_idx,
                    InstrNode::new(Inst::CondBr {
                        cond: non_negative,
                        then_target: check_capacity_id,
                        then_args: vec![],
                        else_target: error_id,
                        else_args: vec![],
                    }),
                );

                let capacity_val =
                    self.emit_i64_const(check_capacity_blk, i64::from(result_capacity));
                let has_capacity = self.emit_with_result(
                    check_capacity_blk,
                    Inst::ICmp {
                        op: ICmpOp::Slt,
                        ty: Ty::I64,
                        lhs: old_len,
                        rhs: capacity_val,
                    },
                );
                self.emit(
                    check_capacity_blk,
                    InstrNode::new(Inst::CondBr {
                        cond: has_capacity,
                        then_target: ok_id,
                        then_args: vec![],
                        else_target: error_id,
                        else_args: vec![],
                    }),
                );
                self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);

                let result_ptr = self.alloc_aggregate(ok_blk, result_slots);
                let new_len = self.emit_with_result(
                    ok_blk,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: old_len,
                        rhs: one,
                    },
                );
                self.store_at_offset(ok_blk, result_ptr, 0, new_len);

                let elem_materialized =
                    self.materialize_reg_as_compact_source(ok_blk, elem_reg, &source_element)?;
                let ok_blk = elem_materialized.block_idx;
                let elem_source = elem_materialized.slot;

                for elem_idx in 0..result_capacity {
                    let elem_idx_val = self.emit_i64_const(ok_blk, i64::from(elem_idx));
                    let before_append = self.emit_with_result(
                        ok_blk,
                        Inst::ICmp {
                            op: ICmpOp::Slt,
                            ty: Ty::I64,
                            lhs: elem_idx_val,
                            rhs: old_len,
                        },
                    );
                    let at_append = self.emit_with_result(
                        ok_blk,
                        Inst::ICmp {
                            op: ICmpOp::Eq,
                            ty: Ty::I64,
                            lhs: elem_idx_val,
                            rhs: old_len,
                        },
                    );
                    for value_offset in 0..element_stride {
                        let old_offset =
                            1 + elem_idx.checked_mul(element_stride).ok_or_else(|| {
                                TrustIrError::UnsupportedOpcode(
                                    "Append: compact source slot overflows u32".to_owned(),
                                )
                            })? + value_offset;
                        let result_offset =
                            1 + elem_idx.checked_mul(element_stride).ok_or_else(|| {
                                TrustIrError::UnsupportedOpcode(
                                    "Append: compact result slot overflows u32".to_owned(),
                                )
                            })? + value_offset;
                        let old_val = if old_offset < source_slots {
                            self.load_at_offset(
                                ok_blk,
                                source_slot.source_ptr,
                                source_slot.offset + old_offset,
                            )
                        } else {
                            zero
                        };
                        let elem_val = self.load_at_offset(
                            ok_blk,
                            elem_source.source_ptr,
                            elem_source.offset + value_offset,
                        );
                        let old_or_zero = self.emit_with_result(
                            ok_blk,
                            Inst::Select {
                                ty: Ty::I64,
                                cond: before_append,
                                then_val: old_val,
                                else_val: zero,
                            },
                        );
                        let value = self.emit_with_result(
                            ok_blk,
                            Inst::Select {
                                ty: Ty::I64,
                                cond: at_append,
                                then_val: elem_val,
                                else_val: old_or_zero,
                            },
                        );
                        self.store_at_offset(ok_blk, result_ptr, result_offset, value);
                    }
                }

                self.store_compact_aggregate_result(ok_blk, rd, result_ptr, result_shape)?;
                return Ok(Some(ok_blk));
            }
        }

        if self.compact_state_slots.contains_key(&seq_reg) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Append: compact source r{seq_reg} is incompatible with append result: source_shape={source_shape:?}, elem_shape={elem_shape:?}, result_shape={compact_result_shape:?}"
            )));
        }

        self.compact_state_slots.remove(&rd);
        let const_total_slots = match self.aggregate_shapes.get(&seq_reg) {
            Some(super::AggregateShape::Sequence { extent, .. }) => extent
                .exact_count()
                .and_then(|len| len.checked_add(2))
                .and_then(|slots| i32::try_from(slots).ok()),
            _ => None,
        };
        let seq_ptr = self.load_reg_as_ptr(block_idx, seq_reg)?;
        let elem_val = self.load_reg(block_idx, elem_reg)?;
        let old_len = self.load_at_offset(block_idx, seq_ptr, 0);
        let one = self.emit_i64_const(block_idx, 1);
        let new_len = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: old_len,
                rhs: one,
            },
        );

        // Allocate: new_len + 1 slots
        let total_i32 = if let Some(total_slots) = const_total_slots {
            self.emit_with_result(
                block_idx,
                Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(i128::from(total_slots)),
                },
            )
        } else {
            let total = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: new_len,
                    rhs: one,
                },
            );
            self.emit_with_result(
                block_idx,
                Inst::Cast {
                    op: CastOp::Trunc,
                    src_ty: Ty::I64,
                    dst_ty: Ty::I32,
                    operand: total,
                },
            )
        };
        let new_ptr = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: Some(total_i32),
                align: None,
            },
        );

        // Store new length
        self.store_at_offset(block_idx, new_ptr, 0, new_len);

        // Copy old elements: for i in 0..old_len, new[i+1] = old[i+1]
        let zero = self.emit_i64_const(block_idx, 0);
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

        let loop_hdr = self.new_aux_block("append_hdr");
        let loop_body = self.new_aux_block("append_body");
        let loop_done = self.new_aux_block("append_done");

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

        // Header
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
                rhs: old_len,
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

        // Body
        let i_val2 = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        let elem_i = self.load_at_dynamic_offset(loop_body, seq_ptr, slot);
        self.store_at_dynamic_offset(loop_body, new_ptr, slot, elem_i);
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

        // Done: store the new element at slot[old_len+1] = slot[new_len]
        let append_slot = self.emit_with_result(
            loop_done,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: old_len,
                rhs: one,
            },
        );
        self.store_at_dynamic_offset(loop_done, new_ptr, append_slot, elem_val);
        self.store_reg_ptr(loop_done, rd, new_ptr)?;

        Ok(Some(loop_done))
    }

    /// Lower `SubSeq(s, lo, hi)`: extract the 1-indexed inclusive sub-range
    /// `s[lo..=hi]` into a brand-new flat aggregate.
    ///
    /// This mirrors [`Self::lower_seq_tail`]'s pure flat manipulation (alloc a
    /// new `[len, e1, ...]` aggregate, store the new length at offset 0, run an
    /// indexed copy loop) but with SubSeq's range arithmetic. It NEVER calls the
    /// handle-based runtime helper `tla_seq_subseq`: trust-ir operates on flat
    /// alloca'd aggregates, not tagged runtime handles.
    ///
    /// Semantics match the bytecode VM (`BuiltinOp::SubSeq`) exactly for the
    /// sequence/tuple operand:
    /// - `lo > hi`           => empty sequence (length 0).
    /// - otherwise, if either bound is outside `1..=Len(s)` => the VM raises
    ///   `IndexOutOfBounds`; we emit a JIT runtime error so the model checker
    ///   falls back to the interpreter for the precise error rather than
    ///   silently fabricating a clamped result.
    /// - otherwise           => length `hi - lo + 1`, with output slot `q`
    ///   (0-indexed) copied from source flat offset `lo + q` (since TLA+
    ///   1-indexed element `p` lives at flat offset `p`).
    pub(super) fn lower_seq_subseq(
        &mut self,
        block_idx: usize,
        rd: u8,
        seq_reg: u8,
        lo_reg: u8,
        hi_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        // Resolve the source sequence to a `(base_ptr, base_off)` flat aggregate
        // `[len, e1, ...]` where the length lives at `base_off + 0` and the
        // 1-indexed element `p` lives at `base_off + p`.
        //
        // `s` frequently lives in a compact state slot (its bytes packed inline
        // in the encoded state rather than behind a standalone flat pointer), so
        // a bare `load_reg_as_ptr` would reinterpret packed scalar bytes as a
        // pointer -- a memory-safety hazard. `materialize_reg_as_compact_source`
        // handles every compact provenance for us: it copies a raw / pointer-
        // backed / register-backed compact sequence into (or addresses it as) a
        // fixed-width aggregate, returning `source_ptr` plus the in-aggregate
        // `offset` base. Genuinely-flat sources (e.g. a `SeqNew` result that was
        // never tracked as a compact slot) stay on the fast `load_reg_as_ptr`
        // path with a zero base. We require fixed-width (scalar, single-slot)
        // elements so the `base_off + p` slot arithmetic stays valid; multi-slot
        // element shapes fall back to the VM.
        let (seq_ptr, base_off, block_idx) = if self.compact_state_slots.contains_key(&seq_reg) {
            let source_shape = self
                .aggregate_shapes
                .get(&seq_reg)
                .cloned()
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "SubSeq: compact source r{seq_reg} has no tracked shape to materialize"
                    ))
                })?;
            if let super::AggregateShape::Sequence {
                element: Some(element),
                ..
            } = &source_shape
            {
                if element.compact_slot_count().is_some_and(|n| n != 1) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "SubSeq: compact source r{seq_reg} has multi-slot elements unsupported by the flat copy loop, got {source_shape:?}"
                    )));
                }
            }
            let materialized =
                self.materialize_reg_as_compact_source(block_idx, seq_reg, &source_shape)?;
            (
                materialized.slot.source_ptr,
                materialized.slot.offset,
                materialized.block_idx,
            )
        } else {
            (self.load_reg_as_ptr(block_idx, seq_reg)?, 0, block_idx)
        };
        let base_off_val = self.emit_i64_const(block_idx, i64::from(base_off));
        let old_len = self.load_at_offset(block_idx, seq_ptr, base_off);
        let lo = self.load_reg(block_idx, lo_reg)?;
        let hi = self.load_reg(block_idx, hi_reg)?;
        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

        // Branch on `lo > hi` (empty range) vs the populated path.
        let lo_gt_hi = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sgt,
                ty: Ty::I64,
                lhs: lo,
                rhs: hi,
            },
        );

        let empty_blk = self.new_aux_block("subseq_empty");
        let bounds_blk = self.new_aux_block("subseq_bounds");
        let error_blk = self.new_aux_block("subseq_error");
        let setup_blk = self.new_aux_block("subseq_setup");
        let loop_hdr = self.new_aux_block("subseq_hdr");
        let loop_body = self.new_aux_block("subseq_body");
        let copy_done = self.new_aux_block("subseq_copy_done");
        let cont_blk = self.new_aux_block("subseq_done");

        let empty_id = self.block_id_of(empty_blk);
        let bounds_id = self.block_id_of(bounds_blk);
        let error_id = self.block_id_of(error_blk);
        let setup_id = self.block_id_of(setup_blk);
        let hdr_id = self.block_id_of(loop_hdr);
        let body_id = self.block_id_of(loop_body);
        let copy_done_id = self.block_id_of(copy_done);
        let cont_id = self.block_id_of(cont_blk);

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: lo_gt_hi,
                then_target: empty_id,
                then_args: vec![],
                else_target: bounds_id,
                else_args: vec![],
            }),
        );

        // Empty range: allocate a single length slot holding 0.
        let empty_ptr = self.alloc_aggregate(empty_blk, 1);
        self.store_at_offset(empty_blk, empty_ptr, 0, zero);
        self.store_reg_ptr(empty_blk, rd, empty_ptr)?;
        self.emit(
            empty_blk,
            InstrNode::new(Inst::Br {
                target: cont_id,
                args: vec![],
            }),
        );

        // Bounds check: require 1 <= lo <= len AND 1 <= hi <= len. The VM raises
        // IndexOutOfBounds otherwise, so route out-of-range to a runtime error.
        let lo_ge_one = self.emit_with_result(
            bounds_blk,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: lo,
                rhs: one,
            },
        );
        let lo_le_len = self.emit_with_result(
            bounds_blk,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: lo,
                rhs: old_len,
            },
        );
        let hi_ge_one = self.emit_with_result(
            bounds_blk,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: hi,
                rhs: one,
            },
        );
        let hi_le_len = self.emit_with_result(
            bounds_blk,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: hi,
                rhs: old_len,
            },
        );
        let lo_ok = self.emit_with_result(
            bounds_blk,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::Bool,
                lhs: lo_ge_one,
                rhs: lo_le_len,
            },
        );
        let hi_ok = self.emit_with_result(
            bounds_blk,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::Bool,
                lhs: hi_ge_one,
                rhs: hi_le_len,
            },
        );
        let in_bounds = self.emit_with_result(
            bounds_blk,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::Bool,
                lhs: lo_ok,
                rhs: hi_ok,
            },
        );
        self.emit(
            bounds_blk,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: setup_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );

        // Out-of-range bound with lo <= hi: behave like the VM's IndexOutOfBounds.
        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);

        // Setup: new_len = hi - lo + 1; allocate new_len + 1 slots; i = 0.
        let span = self.emit_with_result(
            setup_blk,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: hi,
                rhs: lo,
            },
        );
        let new_len = self.emit_with_result(
            setup_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: span,
                rhs: one,
            },
        );
        let total = self.emit_with_result(
            setup_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: new_len,
                rhs: one,
            },
        );
        let new_ptr = self.alloc_dynamic_i64_slots(setup_blk, total);
        self.store_at_offset(setup_blk, new_ptr, 0, new_len);

        let i_alloca = self.emit_with_result(
            setup_blk,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            setup_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            setup_blk,
            InstrNode::new(Inst::Br {
                target: hdr_id,
                args: vec![],
            }),
        );

        // Header: while i < new_len.
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
                rhs: new_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp,
                then_target: body_id,
                then_args: vec![],
                else_target: copy_done_id,
                else_args: vec![],
            }),
        );

        // Body: new[i + 1] = old[base_off + lo + i]; i += 1. The `base_off`
        // term addresses the source sequence's `[len, ...]` within a compact
        // aggregate (0 for genuinely-flat or freshly-materialized base-0
        // sources, so it is a no-op add in the common case).
        let i_val2 = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let lo_plus_i = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: lo,
                rhs: i_val2,
            },
        );
        let src_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: base_off_val,
                rhs: lo_plus_i,
            },
        );
        let elem = self.load_at_dynamic_offset(loop_body, seq_ptr, src_slot);
        let dst_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        self.store_at_dynamic_offset(loop_body, new_ptr, dst_slot, elem);
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

        // Copy done: publish the populated result and fall through to cont.
        self.store_reg_ptr(copy_done, rd, new_ptr)?;
        self.emit(
            copy_done,
            InstrNode::new(Inst::Br {
                target: cont_id,
                args: vec![],
            }),
        );

        Ok(Some(cont_blk))
    }

    /// Lower Concat(left, right): creates a new sequence with all elements from
    /// left followed by all elements from right.
    pub(super) fn lower_seq_concat(
        &mut self,
        block_idx: usize,
        rd: u8,
        left_reg: u8,
        right_reg: u8,
        result_shape: Option<super::AggregateShape>,
    ) -> Result<Option<usize>, TrustIrError> {
        let left_shape = self.aggregate_shapes.get(&left_reg).cloned();
        let right_shape = self.aggregate_shapes.get(&right_reg).cloned();
        let Some(result_shape) = result_shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Concat requires tracked sequence operands, got left={left_shape:?}, right={right_shape:?}"
            )));
        };
        let super::AggregateShape::Sequence {
            extent: result_extent,
            element: result_element,
        } = &result_shape
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Concat requires tracked sequence operands, got left={left_shape:?}, right={right_shape:?}"
            )));
        };
        let result_slots = result_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Concat requires fixed-width result sequence shape, got result_shape={result_shape:?}"
            ))
        })?;
        let result_capacity = result_extent.capacity();
        let result_element = if result_capacity == 0 {
            None
        } else {
            Some(result_element.as_deref().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "Concat requires tracked result element shape, got result_shape={result_shape:?}"
                ))
            })?)
        };
        let element_stride = if let Some(result_element) = result_element {
            result_element.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "Concat requires fixed-width result element shape, got {result_element:?}"
                ))
            })?
        } else {
            0
        };

        let left_shape_for_source = left_shape.as_ref().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Concat requires tracked left sequence shape for r{left_reg}"
            ))
        })?;
        let right_shape_for_source = right_shape.as_ref().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Concat requires tracked right sequence shape for r{right_reg}"
            ))
        })?;
        let left_expected_shape =
            Self::concat_operand_shape(left_shape_for_source, result_element)?;
        let right_expected_shape =
            Self::concat_operand_shape(right_shape_for_source, result_element)?;
        let left_source =
            self.materialize_reg_as_compact_source(block_idx, left_reg, &left_expected_shape)?;
        let right_source = self.materialize_reg_as_compact_source(
            left_source.block_idx,
            right_reg,
            &right_expected_shape,
        )?;
        let block_idx = right_source.block_idx;
        let left_ptr = left_source.slot.source_ptr;
        let right_ptr = right_source.slot.source_ptr;
        let left_base = left_source.slot.offset;
        let right_base = right_source.slot.offset;
        let left_len = self.load_at_offset(block_idx, left_ptr, left_base);
        let right_len = self.load_at_offset(block_idx, right_ptr, right_base);
        let left_capacity = match left_shape.as_ref() {
            Some(super::AggregateShape::Sequence { extent, .. }) => extent.capacity(),
            _ => 0,
        };
        let right_capacity = match right_shape.as_ref() {
            Some(super::AggregateShape::Sequence { extent, .. }) => extent.capacity(),
            _ => 0,
        };
        let guarded_left: super::CompactSequenceLenGuardResult = self
            .guard_compact_sequence_len_in_bounds(
                block_idx,
                left_len,
                left_capacity,
                "Concat_left_len",
            );
        let guarded_right: super::CompactSequenceLenGuardResult = self
            .guard_compact_sequence_len_in_bounds(
                guarded_left.block_idx,
                right_len,
                right_capacity,
                "Concat_right_len",
            );
        let block_idx = guarded_right.block_idx;
        let left_len = guarded_left.len_value;
        let right_len = guarded_right.len_value;
        let total_len = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: left_len,
                rhs: right_len,
            },
        );
        let guarded_total: super::CompactSequenceLenGuardResult = self
            .guard_compact_sequence_len_in_bounds(
                block_idx,
                total_len,
                result_capacity,
                "Concat_total_len",
            );
        let block_idx = guarded_total.block_idx;
        let total_len = guarded_total.len_value;
        let result_ptr = self.alloc_aggregate(block_idx, result_slots);
        self.store_at_offset(block_idx, result_ptr, 0, total_len);
        let i_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        let zero = self.emit_i64_const(block_idx, 0);
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

        let one = self.emit_i64_const(block_idx, 1);
        let left_done = self.copy_concat_sequence_elements(
            block_idx,
            left_ptr,
            left_base,
            left_len,
            result_ptr,
            one,
            element_stride,
            i_alloca,
        )?;
        self.emit(
            left_done,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        let element_stride_value = self.emit_i64_const(left_done, i64::from(element_stride));
        let left_slot_count = self.emit_with_result(
            left_done,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: left_len,
                rhs: element_stride_value,
            },
        );
        let one = self.emit_i64_const(left_done, 1);
        let right_base_slot = self.emit_with_result(
            left_done,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: left_slot_count,
                rhs: one,
            },
        );
        let right_done = self.copy_concat_sequence_elements(
            left_done,
            right_ptr,
            right_base,
            right_len,
            result_ptr,
            right_base_slot,
            element_stride,
            i_alloca,
        )?;
        self.emit(
            right_done,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: total_len,
                align: None,
                volatile: false,
            }),
        );
        let done = self.zero_concat_sequence_tail(
            right_done,
            result_ptr,
            result_capacity,
            element_stride,
            i_alloca,
        )?;
        self.store_compact_aggregate_result(done, rd, result_ptr, result_shape.clone())?;
        Ok(Some(done))
    }

    fn concat_operand_shape(
        source_shape: &super::AggregateShape,
        result_element: Option<&super::AggregateShape>,
    ) -> Result<super::AggregateShape, TrustIrError> {
        let super::AggregateShape::Sequence { extent, .. } = source_shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Concat requires sequence operand shape, got {source_shape:?}"
            )));
        };
        let element = if extent.capacity() == 0 {
            None
        } else {
            Some(Box::new(
                result_element
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                    "Concat requires result element shape for non-empty operand {source_shape:?}"
                ))
                    })?
                    .clone(),
            ))
        };
        Ok(super::AggregateShape::Sequence {
            extent: *extent,
            element,
        })
    }

    fn copy_concat_sequence_elements(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        source_base: u32,
        len: ValueId,
        result_ptr: ValueId,
        result_base_slot: ValueId,
        element_stride: u32,
        i_alloca: ValueId,
    ) -> Result<usize, TrustIrError> {
        let loop_hdr = self.new_aux_block("concat_copy_hdr");
        let loop_body = self.new_aux_block("concat_copy_body");
        let loop_done = self.new_aux_block("concat_copy_done");
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

        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let keep_going = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: keep_going,
                then_target: body_id,
                then_args: vec![],
                else_target: done_id,
                else_args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let stride = self.emit_i64_const(loop_body, i64::from(element_stride));
        let i_slot_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_val,
                rhs: stride,
            },
        );
        let source_start = source_base.checked_add(1).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode("Concat source base slot overflows u32".to_owned())
        })?;
        let source_start = self.emit_i64_const(loop_body, i64::from(source_start));
        let element_source_base = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_slot_offset,
                rhs: source_start,
            },
        );
        let element_dest_base = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: result_base_slot,
                rhs: i_slot_offset,
            },
        );
        for value_offset in 0..element_stride {
            let value_offset = self.emit_i64_const(loop_body, i64::from(value_offset));
            let source_slot = self.emit_with_result(
                loop_body,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: element_source_base,
                    rhs: value_offset,
                },
            );
            let value = self.load_at_dynamic_offset(loop_body, source_ptr, source_slot);
            let dest_slot = self.emit_with_result(
                loop_body,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: element_dest_base,
                    rhs: value_offset,
                },
            );
            self.store_at_dynamic_offset(loop_body, result_ptr, dest_slot, value);
        }
        let step = self.emit_i64_const(loop_body, 1);
        let next_i = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val,
                rhs: step,
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
        Ok(loop_done)
    }

    fn zero_concat_sequence_tail(
        &mut self,
        block_idx: usize,
        result_ptr: ValueId,
        result_capacity: u32,
        element_stride: u32,
        i_alloca: ValueId,
    ) -> Result<usize, TrustIrError> {
        if result_capacity == 0 {
            return Ok(block_idx);
        }

        let loop_hdr = self.new_aux_block("concat_zero_tail_hdr");
        let loop_body = self.new_aux_block("concat_zero_tail_body");
        let loop_done = self.new_aux_block("concat_zero_tail_done");
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

        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let capacity = self.emit_i64_const(loop_hdr, i64::from(result_capacity));
        let keep_going = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: capacity,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: keep_going,
                then_target: body_id,
                then_args: vec![],
                else_target: done_id,
                else_args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let stride = self.emit_i64_const(loop_body, i64::from(element_stride));
        let i_slot_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_val,
                rhs: stride,
            },
        );
        let one = self.emit_i64_const(loop_body, 1);
        let element_dest_base = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_slot_offset,
                rhs: one,
            },
        );
        let zero = self.emit_i64_const(loop_body, 0);
        for value_offset in 0..element_stride {
            let value_offset = self.emit_i64_const(loop_body, i64::from(value_offset));
            let dest_slot = self.emit_with_result(
                loop_body,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: element_dest_base,
                    rhs: value_offset,
                },
            );
            self.store_at_dynamic_offset(loop_body, result_ptr, dest_slot, zero);
        }
        let step = self.emit_i64_const(loop_body, 1);
        let next_i = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val,
                rhs: step,
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
        Ok(loop_done)
    }

    /// Lower Seq(S) as a lazy sequence-set domain.
    ///
    /// Runtime storage mirrors `SUBSET S`: the destination register carries
    /// the base set pointer unchanged, while shape metadata records that
    /// membership must be checked as sequence-element membership in `S`.
    pub(super) fn lower_seq_set(
        &mut self,
        block_idx: usize,
        rd: u8,
        base_reg: u8,
    ) -> Result<(), TrustIrError> {
        let base_shape = self
            .aggregate_shapes
            .get(&base_reg)
            .cloned()
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "Seq: base must have a tracked set/domain shape".to_owned(),
                )
            })?;
        base_shape.validate_seq_base("Seq")?;

        let base_value = self.load_reg(block_idx, base_reg)?;
        self.store_reg_value(block_idx, rd, base_value)?;
        self.compact_state_slots.remove(&rd);
        self.aggregate_shapes.insert(
            rd,
            super::AggregateShape::SeqSet {
                base: Box::new(base_shape),
            },
        );
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);
        Ok(())
    }
}
