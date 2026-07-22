// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Set operation lowering: SetEnum, SetIn, SetUnion, SetIntersect, SetDiff,
//! SubsetEq, Range.

use crate::TrustIrError;
use tla_jit_abi::JitRuntimeErrorKind;
use trust_ir::inst::*;
use trust_ir::ty::Ty;
use trust_ir::value::{BlockId, ValueId};
use trust_ir::InstrNode;

use super::{Ctx, LoopNextKind, LoopNextState, QuantifierLoopState, SetBuilderDedupTable};

/// One SET-sort arm of a lazy union, projected onto a candidate's compact
/// universe at COMPILE time (lever L1, soundness amendment H3: per-arm
/// disjunction — arm masks are never merged).
struct LazyUnionSetArm {
    /// Universe bits allowed by this arm's static powerset base.
    allowed_mask: i64,
    /// `(SUBSET S) \ {{}}` arms additionally require a non-empty candidate.
    require_nonempty: bool,
}

/// WP-27 (item B1): the compile-time RAW-VALUE space a `SetIn` membership test
/// compares its ELEMENT operand against.
///
/// Every membership arm in [`Ctx::lower_set_in`] that does not walk a
/// materialized `Value` works in *raw member space* — ints as themselves,
/// `Bool` as 0/1, `String`/`ModelValue` as their interned `NameId`
/// (`Ctx::domain_key_raw_value`'s space). A register whose tracked shape is
/// [`super::AggregateShape::TaggedScalarUnion`] does NOT hold a raw member: it
/// holds the union-slot INDEX (`0..universe_len`). WP-18 established that
/// contract and decoded the index at its eight compact-function-key sites
/// (`Ctx::decode_scalar_key_reg_raw_value`); the membership sites were left
/// undecoded, which is the residue this carrier closes.
///
/// The decode alone is not enough here, and that is why this type exists. A
/// key table's raw space is the table's own; a membership universe's raw space
/// is the SET's, and the element union may carry members of a *different*
/// scalar sort whose raw `NameId` can numerically collide with an `Int` member
/// of that universe (btree's `focus \in Nodes \union {NIL}` decodes `NIL` to
/// `NameId(103)`, which is a perfectly ordinary key integer). So the decode is
/// admitted only after [`Ctx::check_set_in_union_element_sort`] proves no
/// cross-sort member of the element union lands inside the consumer's space.
enum SetInRawSpace {
    /// The consumer compares for equality against exactly `values`, all of
    /// scalar kind `kind` (`Ctx::exact_universe_element_kind`'s encoding).
    Exact { kind: u8, values: Vec<i64> },
    /// The consumer accepts the inclusive integer interval `[lo, hi]`
    /// (kind `Int`), e.g. an `Interval` set, a `SetBitmaskUniverse::IntRange`
    /// universe, or the `Nat` / `Int` symbolic domains.
    IntRange { lo: i64, hi: i64 },
    /// The consumer has NO compile-time universe — the generic materialized-set
    /// linear scan, whose element slots' raw space is whatever the producer
    /// wrote. Nothing can be proven, so a union-shaped element fails closed.
    Unproven,
}

impl<'cp> Ctx<'cp> {
    /// Lower Powerset { rd, rs }: build a lazy representation of SUBSET(rs).
    ///
    /// The register stores the base set pointer unchanged. Shape tracking marks
    /// the value as a powerset so SetIn can lower `x \in SUBSET S` as
    /// `x \subseteq S` without enumerating subsets.
    pub(super) fn lower_powerset(
        &mut self,
        block_idx: usize,
        rd: u8,
        rs: u8,
    ) -> Result<(), TrustIrError> {
        let base_shape = self.aggregate_shapes.get(&rs).cloned().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "Powerset: base must have a tracked finite shape".to_owned(),
            )
        })?;
        base_shape.validate_powerset_base("Powerset")?;

        let base_value = self.load_reg(block_idx, rs)?;
        self.store_reg_value(block_idx, rd, base_value)?;
        self.compact_state_slots.remove(&rd);
        self.aggregate_shapes.insert(
            rd,
            super::AggregateShape::Powerset {
                base: Box::new(base_shape),
            },
        );
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);
        Ok(())
    }

    /// Lower SetEnum { rd, start, count }: build a set from consecutive registers.
    ///
    /// Layout: alloca[0] = count, alloca[1..=count] = elements from registers.
    pub(super) fn lower_set_enum(
        &mut self,
        block_idx: usize,
        rd: u8,
        start: u8,
        count: u8,
    ) -> Result<(), TrustIrError> {
        // Native-on-general-Value handle path (#4318): in an action that
        // operates on an Unknown-universe compound `Set` state var, a set
        // literal `{e_1, …, e_N}` (N <= 8) over integer elements is built as a
        // `TlaHandle` via `tla_set_enum_N`, so it can be unioned with the
        // compound-set handle by `tla_set_union`. We require every element to
        // be a known int scalar so each can be boxed with `tla_handle_box_int`
        // (interpreter-parity encoding); any non-int element falls through to
        // the existing bitmask / materialized paths (and ultimately fails
        // closed to the interpreter if those cannot service it).
        //
        // WP-10 (item 8) added the third conjunct: the literal must actually be
        // consumed by a `SetUnion` in this body. `tla_set_union` is the only op
        // that can usefully retire a literal handle, so a literal that no
        // `SetUnion` reads can never reach it — boxing it would allocate
        // `count + 1` arena entries and, worse, return early past the Value-free
        // bitmask arm immediately below. Failing this conjunct simply falls
        // through to that arm (or to the materialized path, or fails closed),
        // exactly as in a spec with no compound-set var at all. Combined with
        // the narrowed `action_uses_compound_set_state` (which now also requires
        // the ACTION, not merely the layout, to name the Unknown-universe set
        // var) this retires `tla_handle_box_int`/`tla_set_enum_N` emission from
        // every set literal that provably cannot participate in a handle union.
        if self.action_uses_compound_set_state()
            && self.set_union_operand_regs.contains(&rd)
            && (1..=8).contains(&count)
            && (0..count).all(|i| {
                matches!(
                    self.scalar_shape_of(start + i),
                    Some(super::ScalarShape::Int)
                )
            })
        {
            let mut elem_handles = Vec::with_capacity(usize::from(count));
            for i in 0..count {
                let raw = self.load_reg(block_idx, start + i)?;
                let boxed = self.emit_sanctioned_handle_extern_i64(
                    block_idx,
                    super::SanctionedHandleExternSite::HandleSetEnumBoxInt,
                    "tla_handle_box_int",
                    1,
                    vec![raw],
                )?;
                elem_handles.push(boxed);
            }
            let symbol: &'static str = match count {
                1 => "tla_set_enum_1",
                2 => "tla_set_enum_2",
                3 => "tla_set_enum_3",
                4 => "tla_set_enum_4",
                5 => "tla_set_enum_5",
                6 => "tla_set_enum_6",
                7 => "tla_set_enum_7",
                8 => "tla_set_enum_8",
                _ => unreachable!("count bounded to 1..=8 above"),
            };
            let set_handle = self.emit_sanctioned_handle_extern_i64(
                block_idx,
                super::SanctionedHandleExternSite::HandleSetEnumLiteral,
                symbol,
                usize::from(count),
                elem_handles,
            )?;
            self.store_reg_value(block_idx, rd, set_handle)?;
            self.set_handle_provenance(rd);
            return Ok(());
        }

        if let Some((universe_len, universe)) =
            self.set_enum_scalar_int_domain_universe_from_registers(start, count)
        {
            let mut mask = self.emit_i64_const(block_idx, 0);
            for i in 0..count {
                let elem = self.load_reg(block_idx, start + i)?;
                let bit = self.emit_set_bitmask_universe_bit_i64(
                    block_idx,
                    elem,
                    universe_len,
                    &universe,
                    "SetEnum",
                )?;
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
            self.store_reg_value(block_idx, rd, mask)?;
            self.compact_state_slots.remove(&rd);
            return Ok(());
        }

        let total_slots = u32::from(count) + 1; // length header + elements
        let agg_ptr = self.alloc_aggregate(block_idx, total_slots);

        // Store length
        let len_val = self.emit_i64_const(block_idx, i64::from(count));
        self.store_at_offset(block_idx, agg_ptr, 0, len_val);

        // Store each element
        for i in 0..count {
            let elem = self.load_reg(block_idx, start + i)?;
            self.store_at_offset(block_idx, agg_ptr, u32::from(i) + 1, elem);
        }

        self.store_reg_ptr(block_idx, rd, agg_ptr)?;
        self.compact_state_slots.remove(&rd);

        // Track B increment 1b — expected-shape threading for record-set
        // `{e_1, …, e_N}` literals. If EVERY element register holds a tracked
        // compact `Record`, remember the element source registers keyed by the
        // SetEnum result `rd`. A downstream `v \cup {rec}` / `v \ {rec}` whose
        // other operand is a `RecordSetBitmask` state var then recovers these
        // element records and dispatches the byte-exact
        // `emit_record_set_bitmask_enum_fold` over the var's record universe,
        // instead of fail-closing on the literal's non-pointer-backed mask.
        //
        // The materialized set is still built above (for non-bitmask consumers
        // and the const-set-size tracking the dispatcher attaches to `rd`); this
        // map is a PURE ADDITION that the bitmask binary path consults first and
        // the enum-fold re-validates each element's shape/slot before emitting,
        // so a stale or non-record element fails closed rather than mis-encoding.
        self.record_set_literal_element_regs.remove(&rd);
        if count >= 1
            && (0..count).all(|i| {
                matches!(
                    self.aggregate_shapes.get(&(start + i)),
                    Some(super::AggregateShape::Record { .. })
                )
            })
        {
            let elems: Vec<u8> = (0..count).map(|i| start + i).collect();
            self.record_set_literal_element_regs.insert(rd, elems);
        }
        Ok(())
    }

    /// Lower Times { rd, start, count }: materialize `S1 \X ... \X Sn`.
    ///
    /// The result uses the normal set layout. Each result element is a tuple
    /// aggregate laid out like a sequence: `[arity, elem1, ..., elemN]`.
    pub(super) fn lower_times(
        &mut self,
        block_idx: usize,
        rd: u8,
        start: u8,
        count: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        if count == 0 {
            return Err(TrustIrError::UnsupportedOpcode(
                "Times requires at least one operand".to_owned(),
            ));
        }

        let mut domain_ptrs = Vec::with_capacity(usize::from(count));
        let mut domain_lens = Vec::with_capacity(usize::from(count));
        let mut static_product = Some(1_u32);

        for i in 0..count {
            let reg = start.checked_add(i).ok_or_else(|| {
                TrustIrError::Emission(format!("Times register overflow: start={start} + i={i}"))
            })?;
            self.ensure_materialized_set_reg(reg, "Times")?;
            let domain_ptr =
                self.load_reg_as_ptr_or_materialize_raw_compact(block_idx, reg, "Times")?;
            let domain_len = self.load_at_offset(block_idx, domain_ptr, 0);
            domain_ptrs.push(domain_ptr);
            domain_lens.push(domain_len);
            static_product = static_product.and_then(|product| {
                self.aggregate_shapes
                    .get(&reg)
                    .and_then(super::AggregateShape::tracked_len)
                    .and_then(|len| product.checked_mul(len))
            });
        }

        let one = self.emit_i64_const(block_idx, 1);
        let zero = self.emit_i64_const(block_idx, 0);
        let mut total_len = one;
        for &domain_len in &domain_lens {
            total_len = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Mul,
                    ty: Ty::I64,
                    lhs: total_len,
                    rhs: domain_len,
                },
            );
        }

        let total_slots = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: total_len,
                rhs: one,
            },
        );
        let result_ptr = if let Some(product) = static_product {
            let slots = product.checked_add(1).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "Times static result allocation size overflows u32".to_owned(),
                )
            })?;
            self.alloc_aggregate(block_idx, slots)
        } else {
            self.alloc_dynamic_i64_slots(block_idx, total_slots)
        };
        self.store_at_offset(block_idx, result_ptr, 0, total_len);

        let mut suffix_strides = vec![one; domain_lens.len()];
        let mut stride = one;
        for idx in (0..domain_lens.len()).rev() {
            suffix_strides[idx] = stride;
            stride = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Mul,
                    ty: Ty::I64,
                    lhs: stride,
                    rhs: domain_lens[idx],
                },
            );
        }

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

        let header_block = self.new_aux_block("times_header");
        let body_block = self.new_aux_block("times_body");
        let done_block = self.new_aux_block("times_done");
        let header_id = self.block_id_of(header_block);
        let body_id = self.block_id_of(body_block);
        let done_id = self.block_id_of(done_block);

        self.emit(
            block_idx,
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
                rhs: total_len,
            },
        );
        self.emit(
            header_block,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: body_id,
                then_args: vec![],
                else_target: done_id,
                else_args: vec![],
            }),
        );

        let product_idx = self.emit_with_result(
            body_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let tuple_slots = u32::from(count).checked_add(1).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode("Times tuple slot count overflows u32".to_owned())
        })?;
        let tuple_ptr = self.alloc_aggregate(body_block, tuple_slots);
        let tuple_len = self.emit_i64_const(body_block, i64::from(count));
        self.store_at_offset(body_block, tuple_ptr, 0, tuple_len);

        for component_idx in 0..domain_lens.len() {
            let stride = suffix_strides[component_idx];
            let domain_len = domain_lens[component_idx];
            let quotient = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::SDiv,
                    ty: Ty::I64,
                    lhs: product_idx,
                    rhs: stride,
                },
            );
            let element_idx = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::SRem,
                    ty: Ty::I64,
                    lhs: quotient,
                    rhs: domain_len,
                },
            );
            let domain_slot = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: element_idx,
                    rhs: one,
                },
            );
            let elem =
                self.load_at_dynamic_offset(body_block, domain_ptrs[component_idx], domain_slot);
            let tuple_slot = u32::try_from(component_idx + 1).map_err(|_| {
                TrustIrError::UnsupportedOpcode("Times tuple element slot overflows u32".to_owned())
            })?;
            self.store_at_offset(body_block, tuple_ptr, tuple_slot, elem);
        }

        let tuple_value = self.ptr_to_i64(body_block, tuple_ptr);
        let result_slot = self.emit_with_result(
            body_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: product_idx,
                rhs: one,
            },
        );
        self.store_at_dynamic_offset(body_block, result_ptr, result_slot, tuple_value);

        let next_idx = self.emit_with_result(
            body_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: product_idx,
                rhs: one,
            },
        );
        self.emit(
            body_block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            body_block,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        self.store_reg_ptr(done_block, rd, result_ptr)?;
        self.compact_state_slots.remove(&rd);
        self.compact_function_domains.remove(&rd);
        self.clear_flat_funcdef_pair_list(rd);
        self.const_scalar_values.remove(&rd);
        self.load_imm_scalar_regs.remove(&rd);
        if let Some(shape) = self.times_shape_from_registers(start, count) {
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

        Ok(Some(done_block))
    }

    /// Lower SetFilterBegin: build `{x \in S : P(x)}` as a bounded set.
    ///
    /// Materialized results are allocated up front with capacity `|S| + 1`
    /// slots and length initialized to zero. Compact SetBitmask results stay
    /// mask-native: the result starts at mask zero and each matching LoopNext
    /// sets the current universe bit.
    pub(super) fn lower_set_filter_begin(
        &mut self,
        pc: usize,
        block_idx: usize,
        rd: u8,
        r_binding: u8,
        r_domain: u8,
        loop_end: i32,
    ) -> Result<Option<usize>, TrustIrError> {
        let exit_pc = self.resolve_forward_target(pc, loop_end, "SetFilterBegin")?;
        let body_pc = pc + 1;
        let exit_block = self.block_index_for_pc(exit_pc)?;
        let body_block = self.block_index_for_pc(body_pc)?;

        let domain_max_len = self.finite_set_len_bound_of(r_domain);
        if let Some((universe_len, universe)) = self
            .aggregate_shapes
            .get(&r_domain)
            .and_then(super::exact_scalar_powerset_submask_universe)
        {
            let total_slots =
                super::powerset_submask_result_capacity_u32(universe_len, "SetFilterBegin")?;
            let max_len = total_slots - 1;
            let result_ptr = self.alloc_aggregate(block_idx, total_slots);
            let zero = self.emit_i64_const(block_idx, 0);
            self.store_at_offset(block_idx, result_ptr, 0, zero);
            self.store_reg_ptr(block_idx, rd, result_ptr)?;
            self.compact_state_slots.remove(&rd);

            let frame = self.emit_exact_scalar_powerset_submask_binding_frame_prelude(
                pc,
                block_idx,
                r_binding,
                loop_end,
                "setfilter_subset_submask_header",
                "setfilter_subset_submask_load",
                "SetFilterBegin",
                universe_len,
                &universe,
                None,
            )?;
            self.loop_next_stack.push(LoopNextState {
                rd,
                kind: LoopNextKind::SetFilter,
                loop_state: QuantifierLoopState {
                    idx_alloca: frame.idx_alloca,
                    header_block: frame.header_block,
                    exit_block: frame.exit_block,
                },
                funcdef_capture: None,
                // SetFilter selects a subset of an existing set: no duplicates.
                set_builder_dedup: None,
            });
            self.annotate_loop_bound_n(frame.header_block, max_len);
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::BoundedSet {
                    max_len,
                    element: Some(Box::new(super::AggregateShape::SetBitmask {
                        universe_len,
                        universe: universe.clone(),
                    })),
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(None);
        }
        if let Some(super::AggregateShape::SetBitmask {
            universe_len,
            universe,
        }) = self.aggregate_shapes.get(&r_domain).cloned()
        {
            match &universe {
                super::SetBitmaskUniverse::IntRange { .. }
                | super::SetBitmaskUniverse::ExplicitInt(_) => {}
                super::SetBitmaskUniverse::Exact(elements)
                    if super::homogeneous_exact_universe_scalar_shape(elements).is_some() => {}
                super::SetBitmaskUniverse::Exact(_) => {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "SetFilterBegin: compact SetBitmask iteration requires a homogeneous exact scalar universe"
                            .to_owned(),
                    ));
                }
                super::SetBitmaskUniverse::Unknown => {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "SetFilterBegin: compact SetBitmask domain requires exact universe metadata"
                            .to_owned(),
                    ));
                }
            }
            Self::compact_set_bitmask_valid_mask(universe_len, "SetFilterBegin")?;
            let zero = self.emit_i64_const(block_idx, 0);
            self.store_reg_value(block_idx, rd, zero)?;

            let frame = self.emit_compact_set_bitmask_binding_frame_prelude(
                pc,
                block_idx,
                r_binding,
                r_domain,
                loop_end,
                "setfilter_header",
                "setfilter_load",
                "SetFilterBegin",
                universe_len,
                &universe,
                None,
            )?;
            self.loop_next_stack.push(LoopNextState {
                rd,
                kind: LoopNextKind::SetFilter,
                loop_state: QuantifierLoopState {
                    idx_alloca: frame.idx_alloca,
                    header_block: frame.header_block,
                    exit_block: frame.exit_block,
                },
                funcdef_capture: None,
                // SetFilter selects a subset of an existing set: no duplicates.
                set_builder_dedup: None,
            });
            if !self.annotate_loop_bound(frame.header_block, r_domain) {
                self.mark_unbounded_loop();
            }
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::SetBitmask {
                    universe_len,
                    universe,
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(None);
        }
        self.reject_compact_set_bitmask_powerset_iteration(r_domain, "SetFilterBegin")?;
        // RecordSetBitmask is a multi-slot mask, NOT a materialized
        // `[len, elem...]` set; reading its pointer-backed region's slot 0 as an
        // iteration length would iterate a mask word as a count. Native
        // iteration over a record-set bitmask is not wired (RecordSetBitmask
        // step 3/5) — fail closed (the interpreter is the oracle).
        if matches!(
            self.aggregate_shapes.get(&r_domain),
            Some(super::AggregateShape::RecordSetBitmask { .. })
        ) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetFilterBegin: RecordSetBitmask r{r_domain} is a multi-slot mask, not a \
                 materialized set; native record-set-bitmask iteration is not wired — failing closed"
            )));
        }
        let binding_shape = self
            .aggregate_shapes
            .get(&r_domain)
            .and_then(super::binding_shape_from_domain);
        let domain_ptr = self.load_reg_as_ptr(block_idx, r_domain)?;
        let domain_len = self.load_at_offset(block_idx, domain_ptr, 0);
        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

        let total_slots = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: domain_len,
                rhs: one,
            },
        );
        let result_ptr = match domain_max_len.and_then(|max_len| max_len.checked_add(1)) {
            Some(static_slots) => self.alloc_aggregate(block_idx, static_slots),
            None => self.alloc_dynamic_i64_slots(block_idx, total_slots),
        };

        self.store_at_offset(block_idx, result_ptr, 0, zero);
        self.store_reg_ptr(block_idx, rd, result_ptr)?;
        self.compact_state_slots.remove(&rd);

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

        let header_block = self.new_aux_block("setfilter_header");
        let load_block = self.new_aux_block("setfilter_load");
        let header_id = self.block_id_of(header_block);
        let load_id = self.block_id_of(load_block);
        let body_id = self.block_id_of(body_block);
        let exit_id = self.block_id_of(exit_block);

        self.emit(
            block_idx,
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

        let load_idx = self.emit_with_result(
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
                lhs: load_idx,
                rhs: one,
            },
        );
        let elem = self.load_at_dynamic_offset(load_block, domain_ptr, slot_idx);
        self.store_reg_value(load_block, r_binding, elem)?;
        self.invalidate_reg_tracking(r_binding);
        if let Some(binding_shape) = binding_shape.clone() {
            self.aggregate_shapes.insert(r_binding, binding_shape);
        }
        self.emit(
            load_block,
            InstrNode::new(Inst::Br {
                target: body_id,
                args: vec![],
            }),
        );

        self.loop_next_stack.push(LoopNextState {
            rd,
            kind: LoopNextKind::SetFilter,
            loop_state: QuantifierLoopState {
                idx_alloca,
                header_block,
                exit_block,
            },
            funcdef_capture: None,
            // SetFilter selects a subset of an existing set: no duplicates.
            set_builder_dedup: None,
        });

        if !self.annotate_loop_bound(header_block, r_domain) {
            self.mark_unbounded_loop();
        }

        if let Some(max_len) = domain_max_len {
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::BoundedSet {
                    max_len,
                    element: binding_shape.map(Box::new),
                },
            );
        } else if self
            .aggregate_shapes
            .get(&r_domain)
            .is_some_and(super::AggregateShape::is_finite_set_shape)
        {
            self.aggregate_shapes
                .insert(rd, super::AggregateShape::FiniteSet);
        } else {
            self.aggregate_shapes.remove(&rd);
        }
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);

        Ok(None)
    }

    /// Allocate and zero-initialize the open-addressing hash side index used by
    /// the SetBuilder dedup path, returning its descriptor.
    ///
    /// `capacity` is `2 * result_total_slots`. Because `result_total_slots` is
    /// `|S| + 1`, the table has strictly more slots than the maximum number of
    /// distinct elements (`|S|`), so an empty (`0`) slot always remains and the
    /// linear-probe loop is guaranteed to terminate.
    ///
    /// When `static_total_slots` is `Some`, the result buffer was allocated with
    /// a compile-time-constant capacity (fixed-size domains such as `SetEnum` or
    /// an interval); the table is then sized with a *constant* count so it does
    /// not introduce a dynamic-count `alloca` that those fixed-capacity domains
    /// are required not to emit. Otherwise the table is sized from the runtime
    /// `result_total_slots` value.
    ///
    /// The caller must NOT emit its own terminator into `entry_block`: this
    /// helper makes `entry_block` branch into the zero-fill loop, and the loop
    /// branches to `continue_to` when finished. Allocations are not implicitly
    /// zeroed, so the explicit fill (`0` == empty slot) is required.
    fn emit_set_builder_dedup_table(
        &mut self,
        entry_block: usize,
        continue_to: BlockId,
        result_total_slots: ValueId,
        static_total_slots: Option<u32>,
    ) -> SetBuilderDedupTable {
        let (table_ptr, capacity) = match static_total_slots.and_then(|slots| slots.checked_mul(2))
        {
            Some(static_capacity) => {
                let table_ptr = self.alloc_aggregate(entry_block, static_capacity);
                let capacity = self.emit_i64_const(entry_block, i64::from(static_capacity));
                (table_ptr, capacity)
            }
            None => {
                let two = self.emit_i64_const(entry_block, 2);
                let capacity = self.emit_with_result(
                    entry_block,
                    Inst::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::I64,
                        lhs: result_total_slots,
                        rhs: two,
                    },
                );
                let table_ptr = self.alloc_dynamic_i64_slots(entry_block, capacity);
                (table_ptr, capacity)
            }
        };

        let init_idx_alloca = self.emit_with_result(
            entry_block,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        let zero = self.emit_i64_const(entry_block, 0);
        self.emit(
            entry_block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: init_idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let init_header = self.new_aux_block("setbuilder_dedup_init_header");
        let init_body = self.new_aux_block("setbuilder_dedup_init_body");
        let init_header_id = self.block_id_of(init_header);
        let init_body_id = self.block_id_of(init_body);

        self.emit(
            entry_block,
            InstrNode::new(Inst::Br {
                target: init_header_id,
                args: vec![],
            }),
        );

        let cur = self.emit_with_result(
            init_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: init_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let in_bounds = self.emit_with_result(
            init_header,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur,
                rhs: capacity,
            },
        );
        self.emit(
            init_header,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: init_body_id,
                then_args: vec![],
                else_target: continue_to,
                else_args: vec![],
            }),
        );

        let body_idx = self.emit_with_result(
            init_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: init_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let body_zero = self.emit_i64_const(init_body, 0);
        self.store_at_dynamic_offset(init_body, table_ptr, body_idx, body_zero);
        self.emit_advance_loop(init_body, init_idx_alloca, init_header_id);

        SetBuilderDedupTable {
            table_ptr,
            capacity,
        }
    }

    /// Lower SetBuilderBegin: build `{expr : x \in S}` as a bounded set.
    ///
    /// The result aggregate is allocated with capacity `|S| + 1` slots and
    /// length initialized to zero. Each LoopNext appends the computed body
    /// value when it is not already present, then advances the shared binding
    /// iterator.
    pub(super) fn lower_set_builder_begin(
        &mut self,
        pc: usize,
        block_idx: usize,
        rd: u8,
        r_binding: u8,
        r_domain: u8,
        loop_end: i32,
    ) -> Result<Option<usize>, TrustIrError> {
        let domain_max_len = self.finite_set_len_bound_of(r_domain);
        if let Some((universe_len, universe)) = self
            .aggregate_shapes
            .get(&r_domain)
            .and_then(super::exact_scalar_powerset_submask_universe)
        {
            let total_slots =
                super::powerset_submask_result_capacity_u32(universe_len, "SetBuilderBegin")?;
            let max_len = total_slots - 1;
            let result_ptr = self.alloc_aggregate(block_idx, total_slots);
            let zero = self.emit_i64_const(block_idx, 0);
            self.store_at_offset(block_idx, result_ptr, 0, zero);
            self.store_reg_ptr(block_idx, rd, result_ptr)?;
            self.compact_state_slots.remove(&rd);

            let frame = self.emit_exact_scalar_powerset_submask_binding_frame_prelude(
                pc,
                block_idx,
                r_binding,
                loop_end,
                "setbuilder_subset_submask_header",
                "setbuilder_subset_submask_load",
                "SetBuilderBegin",
                universe_len,
                &universe,
                None,
            )?;
            self.loop_next_stack.push(LoopNextState {
                rd,
                kind: LoopNextKind::SetBuilder,
                loop_state: QuantifierLoopState {
                    idx_alloca: frame.idx_alloca,
                    header_block: frame.header_block,
                    exit_block: frame.exit_block,
                },
                funcdef_capture: None,
                // Bounded by the SUBSET universe size: the linear scan is fine.
                set_builder_dedup: None,
            });
            self.annotate_loop_bound_n(frame.header_block, max_len);
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::BoundedSet {
                    max_len,
                    element: None,
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(None);
        }
        if let Some(super::AggregateShape::SetBitmask {
            universe_len,
            universe,
        }) = self.aggregate_shapes.get(&r_domain).cloned()
        {
            match &universe {
                super::SetBitmaskUniverse::IntRange { .. }
                | super::SetBitmaskUniverse::ExplicitInt(_) => {}
                super::SetBitmaskUniverse::Exact(elements)
                    if super::homogeneous_exact_universe_scalar_shape(elements).is_some() => {}
                super::SetBitmaskUniverse::Exact(_) => {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "SetBuilderBegin: compact SetBitmask iteration requires a homogeneous exact scalar universe"
                            .to_owned(),
                    ));
                }
                super::SetBitmaskUniverse::Unknown => {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "SetBuilderBegin: compact SetBitmask domain requires exact universe metadata"
                            .to_owned(),
                    ));
                }
            }
            Self::compact_set_bitmask_valid_mask(universe_len, "SetBuilderBegin")?;
            let total_slots = universe_len.checked_add(1).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "SetBuilderBegin: compact SetBitmask result capacity overflows u32".to_owned(),
                )
            })?;
            let result_ptr = self.alloc_aggregate(block_idx, total_slots);
            let zero = self.emit_i64_const(block_idx, 0);
            self.store_at_offset(block_idx, result_ptr, 0, zero);
            self.store_reg_ptr(block_idx, rd, result_ptr)?;
            self.compact_state_slots.remove(&rd);

            let frame = self.emit_compact_set_bitmask_binding_frame_prelude(
                pc,
                block_idx,
                r_binding,
                r_domain,
                loop_end,
                "setbuilder_header",
                "setbuilder_load",
                "SetBuilderBegin",
                universe_len,
                &universe,
                None,
            )?;
            self.loop_next_stack.push(LoopNextState {
                rd,
                kind: LoopNextKind::SetBuilder,
                loop_state: QuantifierLoopState {
                    idx_alloca: frame.idx_alloca,
                    header_block: frame.header_block,
                    exit_block: frame.exit_block,
                },
                funcdef_capture: None,
                // Bounded by the bitmask universe size: the linear scan is fine.
                set_builder_dedup: None,
            });
            if !self.annotate_loop_bound(frame.header_block, r_domain) {
                self.mark_unbounded_loop();
            }
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::BoundedSet {
                    max_len: universe_len,
                    element: None,
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(None);
        }

        self.reject_compact_set_bitmask_powerset_iteration(r_domain, "SetBuilderBegin")?;
        let binding_shape = self
            .aggregate_shapes
            .get(&r_domain)
            .and_then(super::binding_shape_from_domain);
        let exit_pc = self.resolve_forward_target(pc, loop_end, "SetBuilderBegin")?;
        let body_pc = pc + 1;
        let exit_block = self.block_index_for_pc(exit_pc)?;
        let body_block = self.block_index_for_pc(body_pc)?;
        let domain_ptr = self.load_reg_as_ptr(block_idx, r_domain)?;
        let domain_len = self.load_at_offset(block_idx, domain_ptr, 0);
        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

        let total_slots = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: domain_len,
                rhs: one,
            },
        );
        let result_ptr = match domain_max_len.and_then(|max_len| max_len.checked_add(1)) {
            Some(static_slots) => self.alloc_aggregate(block_idx, static_slots),
            None => self.alloc_dynamic_i64_slots(block_idx, total_slots),
        };

        self.store_at_offset(block_idx, result_ptr, 0, zero);
        self.store_reg_ptr(block_idx, rd, result_ptr)?;
        self.compact_state_slots.remove(&rd);

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

        let header_block = self.new_aux_block("setbuilder_header");
        let load_block = self.new_aux_block("setbuilder_load");
        let header_id = self.block_id_of(header_block);
        let load_id = self.block_id_of(load_block);
        let body_id = self.block_id_of(body_block);
        let exit_id = self.block_id_of(exit_block);

        // Allocate the hash side index that turns per-element dedup from an O(n)
        // linear scan into an O(1) amortized probe (so whole-set construction is
        // O(n) amortized instead of O(n^2)). This is the general domain path --
        // the bounded SUBSET/bitmask paths above keep the linear scan because
        // their element count is statically small. The table is sized with a
        // constant count exactly when the result buffer was (fixed-size domains),
        // so it never introduces a dynamic-count alloca those domains forbid. The
        // helper emits the table allocation + zero-fill loop into `block_idx` and
        // routes control through it into the loop header, so we must NOT emit our
        // own branch to `header_id` here.
        let static_total_slots = domain_max_len.and_then(|max_len| max_len.checked_add(1));
        let dedup = self.emit_set_builder_dedup_table(
            block_idx,
            header_id,
            total_slots,
            static_total_slots,
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

        let load_idx = self.emit_with_result(
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
                lhs: load_idx,
                rhs: one,
            },
        );
        let elem = self.load_at_dynamic_offset(load_block, domain_ptr, slot_idx);
        self.store_reg_value(load_block, r_binding, elem)?;
        self.invalidate_reg_tracking(r_binding);
        if let Some(binding_shape) = binding_shape {
            // Flat-aggregate value ABI (mirrors the `FuncDef` binding load): a
            // set slot holding a *boxed compound* element (tuple/record/function)
            // stores a POINTER to that element's aggregate, not an inlined value.
            // The element just loaded into `r_binding` is therefore an aggregate
            // base pointer; record that provenance so a downstream tuple-component
            // `FuncApply` (e.g. `<<dx,dy>>[1]` where `<<dx,dy>>` ranges over the
            // constant 2-tuple offset set `nbrs` in GameOfLife's `score`) can
            // `load_reg_as_ptr` it. Single-slot scalar bindings keep their inlined
            // value and no pointer provenance, so the scalar-rejection wall in
            // `load_reg_as_ptr` stays intact.
            if Self::binding_element_is_boxed_compound(&binding_shape) {
                self.mark_flat_aggregate_pointer(r_binding);
            }
            self.aggregate_shapes.insert(r_binding, binding_shape);
        }
        self.emit(
            load_block,
            InstrNode::new(Inst::Br {
                target: body_id,
                args: vec![],
            }),
        );

        self.loop_next_stack.push(LoopNextState {
            rd,
            kind: LoopNextKind::SetBuilder,
            loop_state: QuantifierLoopState {
                idx_alloca,
                header_block,
                exit_block,
            },
            funcdef_capture: None,
            set_builder_dedup: Some(dedup),
        });

        if !self.annotate_loop_bound(header_block, r_domain) {
            self.mark_unbounded_loop();
        }

        if let Some(max_len) = domain_max_len {
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::BoundedSet {
                    max_len,
                    element: None,
                },
            );
        } else if self
            .aggregate_shapes
            .get(&r_domain)
            .is_some_and(super::AggregateShape::is_finite_set_shape)
        {
            self.aggregate_shapes
                .insert(rd, super::AggregateShape::FiniteSet);
        } else {
            self.aggregate_shapes.remove(&rd);
        }
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);

        Ok(None)
    }

    /// Lower LoopNext for SetFilter: append the current binding when the
    /// predicate is true, then advance the iterator.
    pub(super) fn lower_loop_next_set_filter(
        &mut self,
        block_idx: usize,
        r_binding: u8,
        r_body: u8,
        rd: u8,
        loop_state: QuantifierLoopState,
    ) -> Result<Option<usize>, TrustIrError> {
        let body_val = self.load_reg(block_idx, r_body)?;
        let zero = self.emit_i64_const(block_idx, 0);
        let predicate_true = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: body_val,
                rhs: zero,
            },
        );

        let append_block = self.new_aux_block("setfilter_append");
        let advance_block = self.new_aux_block("setfilter_advance");
        let append_id = self.block_id_of(append_block);
        let advance_id = self.block_id_of(advance_block);
        let header_id = self.block_id_of(loop_state.header_block);

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: predicate_true,
                then_target: append_id,
                then_args: vec![],
                else_target: advance_id,
                else_args: vec![],
            }),
        );

        if let Some(super::AggregateShape::SetBitmask { universe_len, .. }) =
            self.aggregate_shapes.get(&rd).cloned()
        {
            Self::compact_set_bitmask_valid_mask(universe_len, "SetFilterBegin")?;
            let current_mask = self.load_reg(append_block, rd)?;
            let cur_idx = self.emit_with_result(
                append_block,
                Inst::Load {
                    ty: Ty::I64,
                    ptr: loop_state.idx_alloca,
                    align: None,
                    volatile: false,
                },
            );
            let one = self.emit_i64_const(append_block, 1);
            let bit = self.emit_with_result(
                append_block,
                Inst::BinOp {
                    op: BinOp::Shl,
                    ty: Ty::I64,
                    lhs: one,
                    rhs: cur_idx,
                },
            );
            let new_mask = self.emit_with_result(
                append_block,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: current_mask,
                    rhs: bit,
                },
            );
            self.store_reg_value(append_block, rd, new_mask)?;
        } else {
            let result_ptr = self.load_reg_as_ptr(append_block, rd)?;
            let current_len = self.load_at_offset(append_block, result_ptr, 0);
            let one_for_slot = self.emit_i64_const(append_block, 1);
            let slot_idx = self.emit_with_result(
                append_block,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: current_len,
                    rhs: one_for_slot,
                },
            );
            let binding_val = self.load_reg(append_block, r_binding)?;
            self.store_at_dynamic_offset(append_block, result_ptr, slot_idx, binding_val);
            let new_len = self.emit_with_result(
                append_block,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: current_len,
                    rhs: one_for_slot,
                },
            );
            self.store_at_offset(append_block, result_ptr, 0, new_len);
        }
        self.emit(
            append_block,
            InstrNode::new(Inst::Br {
                target: advance_id,
                args: vec![],
            }),
        );

        let cur_idx = self.emit_with_result(
            advance_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: loop_state.idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one_for_inc = self.emit_i64_const(advance_block, 1);
        let next_idx = self.emit_with_result(
            advance_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: one_for_inc,
            },
        );
        self.emit(
            advance_block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: loop_state.idx_alloca,
                value: next_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            advance_block,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        Ok(None)
    }

    /// Lower LoopNext for SetBuilder: append the body value if it is not
    /// already present, then advance the iterator.
    pub(super) fn lower_loop_next_set_builder(
        &mut self,
        block_idx: usize,
        r_body: u8,
        rd: u8,
        loop_state: QuantifierLoopState,
        dedup: Option<SetBuilderDedupTable>,
    ) -> Result<Option<usize>, TrustIrError> {
        let result_ptr = self.load_reg_as_ptr(block_idx, rd)?;
        let current_len = self.load_at_offset(block_idx, result_ptr, 0);
        let body_shape = self
            .aggregate_shapes
            .get(&r_body)
            .cloned()
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "SetBuilderBegin: body value shape is unknown; cannot safely deduplicate result set"
                        .to_owned(),
                )
            })?;
        // Only slot-identity element shapes (a register holding a single i64
        // whose bit-pattern *is* the TLA+ value -- scalars and interned
        // String/ModelValue names) can be deduplicated with a raw i64 `Eq`.
        //
        // A fixed-extent tuple-of-scalars body (`<<a, b>>`) is *not* slot
        // identity: its register holds a *pointer* to a `[arity, e1, ..., ek]`
        // buffer that the loop body re-`alloca`s every iteration. The trust-cg
        // backend assigns each such loop-body `alloca` a *single* reused stack
        // slot (verified: disabling post-RA opt does not change it, and a
        // membership probe against `(1..3) \X (1..2)` -- which `lower_times`
        // builds the same way -- finds only the *last* tuple, all earlier
        // elements aliasing the final buffer). Storing those transient pointers
        // therefore yields a set whose elements are all corrupt past the last
        // one; structural dedup/membership over it diverges from the
        // interpreter. Closing this soundly needs a stable per-element backing
        // arena allocated *outside* the loop (a larger change to the
        // SetBuilderBegin/LoopNext contract), so until that exists tuple bodies
        // fail closed and the whole spec falls back to the interpreter.
        if !body_shape.has_slot_identity_equality() {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetBuilderBegin: body value shape {body_shape:?} requires structural set-builder deduplication"
            )));
        }
        let body_val = self.load_reg(block_idx, r_body)?;

        // Propagate the element shape onto the result aggregate regardless of
        // which dedup strategy runs below, so downstream consumers observe the
        // same type information they did before this optimization existed.
        if let Some(super::AggregateShape::BoundedSet { max_len, .. }) =
            self.aggregate_shapes.get(&rd).cloned()
        {
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::BoundedSet {
                    max_len,
                    element: Some(Box::new(body_shape)),
                },
            );
        }

        // When the matching SetBuilderBegin allocated a hash side index, use the
        // O(1) amortized hash-probe dedup. The result buffer itself is written in
        // the exact same insertion order as the linear-scan path, and membership
        // is decided with the identical i64-slot equality (`body_val` is a
        // slot-identity shape, so i64 equality *is* TLA+ value equality), so the
        // produced set is byte-for-byte identical -- only the asymptotics change.
        if let Some(table) = dedup {
            return self.lower_loop_next_set_builder_hashed(
                block_idx,
                result_ptr,
                current_len,
                body_val,
                loop_state,
                table,
            );
        }

        let scan_idx_alloca = self.emit_with_result(
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
                ptr: scan_idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let scan_header_block = self.new_aux_block("setbuilder_dedup_header");
        let scan_body_block = self.new_aux_block("setbuilder_dedup_body");
        let scan_inc_block = self.new_aux_block("setbuilder_dedup_inc");
        let append_block = self.new_aux_block("setbuilder_append");
        let advance_block = self.new_aux_block("setbuilder_advance");
        let scan_header_id = self.block_id_of(scan_header_block);
        let scan_body_id = self.block_id_of(scan_body_block);
        let scan_inc_id = self.block_id_of(scan_inc_block);
        let append_id = self.block_id_of(append_block);
        let advance_id = self.block_id_of(advance_block);
        let header_id = self.block_id_of(loop_state.header_block);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: scan_header_id,
                args: vec![],
            }),
        );

        let scan_idx = self.emit_with_result(
            scan_header_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: scan_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let scan_in_bounds = self.emit_with_result(
            scan_header_block,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: scan_idx,
                rhs: current_len,
            },
        );
        self.emit(
            scan_header_block,
            InstrNode::new(Inst::CondBr {
                cond: scan_in_bounds,
                then_target: scan_body_id,
                then_args: vec![],
                else_target: append_id,
                else_args: vec![],
            }),
        );

        let scan_body_idx = self.emit_with_result(
            scan_body_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: scan_idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let scan_one = self.emit_i64_const(scan_body_block, 1);
        let scan_slot_idx = self.emit_with_result(
            scan_body_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: scan_body_idx,
                rhs: scan_one,
            },
        );
        let existing = self.load_at_dynamic_offset(scan_body_block, result_ptr, scan_slot_idx);
        let already_present = self.emit_with_result(
            scan_body_block,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: existing,
                rhs: body_val,
            },
        );
        self.emit(
            scan_body_block,
            InstrNode::new(Inst::CondBr {
                cond: already_present,
                then_target: advance_id,
                then_args: vec![],
                else_target: scan_inc_id,
                else_args: vec![],
            }),
        );

        self.emit_advance_loop(scan_inc_block, scan_idx_alloca, scan_header_id);

        let one = self.emit_i64_const(append_block, 1);
        let slot_idx = self.emit_with_result(
            append_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: current_len,
                rhs: one,
            },
        );
        self.store_at_dynamic_offset(append_block, result_ptr, slot_idx, body_val);
        let new_len = self.emit_with_result(
            append_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: current_len,
                rhs: one,
            },
        );
        self.store_at_offset(append_block, result_ptr, 0, new_len);
        self.emit(
            append_block,
            InstrNode::new(Inst::Br {
                target: advance_id,
                args: vec![],
            }),
        );

        self.emit_advance_loop(advance_block, loop_state.idx_alloca, header_id);

        Ok(None)
    }

    /// Hash-probe variant of the SetBuilder LoopNext body.
    ///
    /// Replaces the O(n) per-element linear membership scan with an O(1)
    /// amortized open-addressing probe against the side-index hash table the
    /// matching `SetBuilderBegin` allocated, turning whole-set construction from
    /// O(n^2) into O(n) amortized.
    ///
    /// Invariants that make this a pure optimization (identical results):
    /// * The result buffer (`result_ptr`) is appended to in the *same insertion
    ///   order* as the linear-scan path; the hash table is purely a side index.
    /// * Membership is decided with the identical i64 `Eq` the scan used. The
    ///   caller has already proven `body_val` carries slot-identity equality, so
    ///   i64 equality is exactly TLA+ value equality -- collisions are resolved
    ///   by a true value compare (`result[entry] == body_val`), never the hash.
    /// * The table stores *1-based* result indices (`0` == empty slot). Because a
    ///   1-based index `i` is also the slot offset of the i-th element in the
    ///   result buffer (slot 0 holds the length), a stored index doubles as the
    ///   element's slot offset, and the `0` sentinel can never collide with a
    ///   real (>= 1) index.
    /// * `capacity` is strictly greater than the maximum number of distinct
    ///   elements, so an empty slot always exists and the linear probe is
    ///   guaranteed to terminate.
    ///
    /// CFG (all blocks are fresh aux blocks):
    ///   entry          -> probe_header               (after computing h0)
    ///   probe_header   -> append (miss) | probe_hit  (table[h] == 0 ? miss)
    ///   probe_hit      -> advance (skip) | probe_inc  (result[entry] == body ?)
    ///   probe_inc      -> probe_header               (h = (h + 1) % capacity)
    ///   append         -> advance                    (write element + record h)
    fn lower_loop_next_set_builder_hashed(
        &mut self,
        block_idx: usize,
        result_ptr: ValueId,
        current_len: ValueId,
        body_val: ValueId,
        loop_state: QuantifierLoopState,
        table: SetBuilderDedupTable,
    ) -> Result<Option<usize>, TrustIrError> {
        // Mutable probe cursor `h`, initialized to hash(body_val) % capacity.
        let probe_alloca = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        let initial_hash = self.emit_dedup_hash(block_idx, body_val, table.capacity);
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: probe_alloca,
                value: initial_hash,
                align: None,
                volatile: false,
            }),
        );

        let probe_header_block = self.new_aux_block("setbuilder_probe_header");
        let probe_hit_block = self.new_aux_block("setbuilder_probe_hit");
        let probe_inc_block = self.new_aux_block("setbuilder_probe_inc");
        let append_block = self.new_aux_block("setbuilder_append");
        let advance_block = self.new_aux_block("setbuilder_advance");
        let probe_header_id = self.block_id_of(probe_header_block);
        let probe_hit_id = self.block_id_of(probe_hit_block);
        let probe_inc_id = self.block_id_of(probe_inc_block);
        let append_id = self.block_id_of(append_block);
        let advance_id = self.block_id_of(advance_block);
        let header_id = self.block_id_of(loop_state.header_block);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: probe_header_id,
                args: vec![],
            }),
        );

        // probe_header: entry = table[h]; if entry == 0 -> miss (append), else hit.
        let probe_h = self.emit_with_result(
            probe_header_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: probe_alloca,
                align: None,
                volatile: false,
            },
        );
        let entry = self.load_at_dynamic_offset(probe_header_block, table.table_ptr, probe_h);
        let zero = self.emit_i64_const(probe_header_block, 0);
        let is_empty = self.emit_with_result(
            probe_header_block,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: entry,
                rhs: zero,
            },
        );
        self.emit(
            probe_header_block,
            InstrNode::new(Inst::CondBr {
                cond: is_empty,
                then_target: append_id,
                then_args: vec![],
                else_target: probe_hit_id,
                else_args: vec![],
            }),
        );

        // probe_hit: entry is a 1-based result index, which is also the slot
        // offset of that element. existing = result[entry]; if existing ==
        // body_val the element is already present -> advance (skip), else this
        // is a hash collision -> probe the next slot.
        let hit_h = self.emit_with_result(
            probe_hit_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: probe_alloca,
                align: None,
                volatile: false,
            },
        );
        let hit_entry = self.load_at_dynamic_offset(probe_hit_block, table.table_ptr, hit_h);
        let existing = self.load_at_dynamic_offset(probe_hit_block, result_ptr, hit_entry);
        let already_present = self.emit_with_result(
            probe_hit_block,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: existing,
                rhs: body_val,
            },
        );
        self.emit(
            probe_hit_block,
            InstrNode::new(Inst::CondBr {
                cond: already_present,
                then_target: advance_id,
                then_args: vec![],
                else_target: probe_inc_id,
                else_args: vec![],
            }),
        );

        // probe_inc: h = (h + 1) % capacity; branch back to probe_header.
        let inc_h = self.emit_with_result(
            probe_inc_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: probe_alloca,
                align: None,
                volatile: false,
            },
        );
        let inc_one = self.emit_i64_const(probe_inc_block, 1);
        let inc_next = self.emit_with_result(
            probe_inc_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: inc_h,
                rhs: inc_one,
            },
        );
        let inc_wrapped = self.emit_with_result(
            probe_inc_block,
            Inst::BinOp {
                op: BinOp::URem,
                ty: Ty::I64,
                lhs: inc_next,
                rhs: table.capacity,
            },
        );
        self.emit(
            probe_inc_block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: probe_alloca,
                value: inc_wrapped,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            probe_inc_block,
            InstrNode::new(Inst::Br {
                target: probe_header_id,
                args: vec![],
            }),
        );

        // append: result[len + 1] = body_val; len += 1; table[h] = new_len
        // (the 1-based index of the just-appended element); branch to advance.
        let append_one = self.emit_i64_const(append_block, 1);
        let slot_idx = self.emit_with_result(
            append_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: current_len,
                rhs: append_one,
            },
        );
        self.store_at_dynamic_offset(append_block, result_ptr, slot_idx, body_val);
        self.store_at_offset(append_block, result_ptr, 0, slot_idx);
        let append_h = self.emit_with_result(
            append_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: probe_alloca,
                align: None,
                volatile: false,
            },
        );
        self.store_at_dynamic_offset(append_block, table.table_ptr, append_h, slot_idx);
        self.emit(
            append_block,
            InstrNode::new(Inst::Br {
                target: advance_id,
                args: vec![],
            }),
        );

        self.emit_advance_loop(advance_block, loop_state.idx_alloca, header_id);

        Ok(None)
    }

    /// Emit `hash(value) % capacity` for the SetBuilder dedup table.
    ///
    /// Uses a fixed integer finalizer (xor-shift / multiply, à la the
    /// SplitMix64 / MurmurHash3 finalizers) purely to scatter slot-identity i64
    /// values across the table; the value of the hash is irrelevant to
    /// correctness because collisions are always resolved by a true value
    /// compare. `capacity` is always >= 2, so the `URem` is well defined, and
    /// the unsigned remainder is a non-negative slot index in `[0, capacity)`.
    fn emit_dedup_hash(&mut self, block_idx: usize, value: ValueId, capacity: ValueId) -> ValueId {
        // mixed = value ^ (value >> 33)
        let shift_a = self.emit_i64_const(block_idx, 33);
        let xs_a = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::LShr,
                ty: Ty::I64,
                lhs: value,
                rhs: shift_a,
            },
        );
        let mixed_a = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Xor,
                ty: Ty::I64,
                lhs: value,
                rhs: xs_a,
            },
        );
        // mixed = mixed * 0xff51afd7ed558ccd
        let mul_a = self.emit_i64_const(block_idx, 0xff51_afd7_ed55_8ccd_u64 as i64);
        let mixed_b = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: mixed_a,
                rhs: mul_a,
            },
        );
        // mixed = mixed ^ (mixed >> 33)
        let shift_b = self.emit_i64_const(block_idx, 33);
        let xs_b = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::LShr,
                ty: Ty::I64,
                lhs: mixed_b,
                rhs: shift_b,
            },
        );
        let mixed_c = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Xor,
                ty: Ty::I64,
                lhs: mixed_b,
                rhs: xs_b,
            },
        );
        // h = mixed (unsigned) % capacity
        self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::URem,
                ty: Ty::I64,
                lhs: mixed_c,
                rhs: capacity,
            },
        )
    }

    /// Lower SetIn { rd, elem, set }: set membership test.
    ///
    /// Emits a linear scan loop: for each element in the set, compare with elem.
    /// Result is 1 (found) or 0 (not found).
    ///
    /// CFG:
    ///   entry -> header
    ///   header -> body (if i < len) | not_found (if i >= len)
    ///   body -> found (if equal) | inc (if not equal)
    ///   inc -> header
    ///   found -> merge (rd = 1)
    ///   not_found -> merge (rd = 0)
    pub(super) fn lower_set_in(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        set_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        if let Some(shape) = self.aggregate_shapes.get(&set_reg).cloned() {
            match shape {
                super::AggregateShape::Interval { lo, hi } => {
                    let elem_val = self.load_reg(block_idx, elem_reg)?;
                    // WP-27 (item B1): an `Interval` compares the element in
                    // raw int space, so a tagged-scalar-union index must be
                    // decoded first (or fail closed if the decode could alias).
                    let elem_val = self.set_in_element_raw_value_i64(
                        block_idx,
                        elem_reg,
                        elem_val,
                        &SetInRawSpace::IntRange { lo, hi },
                        "SetIn: interval membership",
                    )?;
                    self.lower_interval_membership_value(block_idx, rd, elem_val, lo, hi)?;
                    return Ok(Some(block_idx));
                }
                super::AggregateShape::RecordSet { fields } => {
                    return self
                        .lower_record_set_membership(block_idx, rd, elem_reg, set_reg, fields);
                }
                super::AggregateShape::Powerset { base } => {
                    return self
                        .lower_powerset_membership(block_idx, rd, elem_reg, set_reg, *base, false);
                }
                super::AggregateShape::NonEmptyPowerset { base } => {
                    return self
                        .lower_powerset_membership(block_idx, rd, elem_reg, set_reg, *base, true);
                }
                // Lazy union membership (lever L1): fully static per-arm
                // lowering over the candidate value; the set register's
                // placeholder payload is never loaded (amendment H1).
                super::AggregateShape::LazyUnion { left, right } => {
                    let elem_shape = self.aggregate_shapes.get(&elem_reg).cloned();
                    let elem_val = self.load_reg(block_idx, elem_reg)?;
                    let true_blk = self.new_aux_block("lazy_union_member_true");
                    let false_blk = self.new_aux_block("lazy_union_member_false");
                    let merge_blk = self.new_aux_block("lazy_union_member_merge");
                    let true_id = self.block_id_of(true_blk);
                    let false_id = self.block_id_of(false_blk);
                    let merge_id = self.block_id_of(merge_blk);
                    self.lower_entry_in_lazy_union_range_branch(
                        block_idx,
                        elem_val,
                        elem_shape.as_ref(),
                        &left,
                        &right,
                        true_id,
                        false_id,
                        "SetIn: lazy-union membership",
                    )?;
                    self.store_reg_imm(true_blk, rd, 1)?;
                    self.emit(
                        true_blk,
                        InstrNode::new(Inst::Br {
                            target: merge_id,
                            args: vec![],
                        }),
                    );
                    self.store_reg_imm(false_blk, rd, 0)?;
                    self.emit(
                        false_blk,
                        InstrNode::new(Inst::Br {
                            target: merge_id,
                            args: vec![],
                        }),
                    );
                    return Ok(Some(merge_blk));
                }
                super::AggregateShape::FunctionSet { domain, range } => {
                    return self.lower_function_set_membership(
                        block_idx, rd, elem_reg, set_reg, *domain, *range,
                    );
                }
                super::AggregateShape::SeqSet { base } => {
                    return self.lower_seq_set_membership(block_idx, rd, elem_reg, set_reg, *base);
                }
                super::AggregateShape::Set { len, element } => {
                    if let (
                        Some(source_slot),
                        Some(elem_shape @ super::AggregateShape::Record { .. }),
                    ) = (
                        self.compact_state_slot_for_use(block_idx, elem_reg)?,
                        self.aggregate_shapes.get(&elem_reg).cloned(),
                    ) {
                        return self.lower_compact_record_materialized_set_membership(
                            block_idx,
                            rd,
                            source_slot,
                            &elem_shape,
                            set_reg,
                            len,
                            element.as_deref(),
                        );
                    }
                }
                super::AggregateShape::SymbolicDomain(domain) => {
                    self.lower_symbolic_domain_membership(block_idx, rd, elem_reg, domain)?;
                    return Ok(Some(block_idx));
                }
                super::AggregateShape::SetBitmask { .. } => {
                    let Some((universe_len, universe)) = shape.set_bitmask_universe() else {
                        return Err(TrustIrError::UnsupportedOpcode(
                            "SetIn: compact SetBitmask membership requires exact universe metadata"
                                .to_owned(),
                        ));
                    };
                    // Bool-typed SetBitmask universes do not yet have a typed
                    // scalar encoding distinct from Int — accepting them would
                    // silently treat `LoadImm 1` (Int) as Bool(true). Reject
                    // until a typed Bool scalar encoding exists.
                    if let super::SetBitmaskUniverse::Exact(elements) = &universe {
                        if elements
                            .iter()
                            .any(|element| matches!(element, super::SetBitmaskElement::Bool(_)))
                        {
                            return Err(TrustIrError::UnsupportedOpcode(
                                "SetIn: compact SetBitmask Bool universe requires typed scalar encoding (exact universe metadata for Bool not yet supported)"
                                    .to_owned(),
                            ));
                        }
                    }
                    let elem_val = self.load_reg(block_idx, elem_reg)?;
                    // WP-27 (item B1): the universe bit is selected by
                    // comparing the element in RAW MEMBER space, so a
                    // tagged-scalar-union index must be decoded first.
                    let space = Self::set_in_raw_space_of_universe(
                        &universe,
                        universe_len,
                        "SetIn: compact SetBitmask membership",
                    )?;
                    let elem_val = self.set_in_element_raw_value_i64(
                        block_idx,
                        elem_reg,
                        elem_val,
                        &space,
                        "SetIn: compact SetBitmask membership",
                    )?;
                    let mask = self.load_reg(block_idx, set_reg)?;
                    let member = self.emit_compact_bitmask_set_membership_i64(
                        block_idx,
                        elem_val,
                        mask,
                        universe_len,
                        &universe,
                        "SetIn: compact SetBitmask membership",
                    )?;
                    self.store_reg_value(block_idx, rd, member)?;
                    return Ok(Some(block_idx));
                }
                super::AggregateShape::TaggedScalarOrSet { .. } => {
                    let Some((universe_len, universe)) = shape.tagged_set_branch_universe() else {
                        return Err(TrustIrError::UnsupportedOpcode(
                            "SetIn: tagged scalar-or-set membership requires exact universe metadata"
                                .to_owned(),
                        ));
                    };
                    let elem_val = self.load_reg(block_idx, elem_reg)?;
                    // WP-27 (item B1): same raw-member-space contract as the
                    // `SetBitmask` arm — the set-branch universe bit is chosen
                    // by a raw compare, so decode a union-shaped element index.
                    let space = Self::set_in_raw_space_of_universe(
                        &universe,
                        universe_len,
                        "SetIn: tagged scalar-or-set membership",
                    )?;
                    let elem_val = self.set_in_element_raw_value_i64(
                        block_idx,
                        elem_reg,
                        elem_val,
                        &space,
                        "SetIn: tagged scalar-or-set membership",
                    )?;
                    let (block_idx, mask) = self.emit_tagged_scalar_or_set_mask_i64(
                        block_idx,
                        set_reg,
                        universe_len,
                        "SetIn: tagged scalar-or-set membership",
                    )?;
                    let member = self.emit_compact_bitmask_set_membership_i64(
                        block_idx,
                        elem_val,
                        mask,
                        universe_len,
                        &universe,
                        "SetIn: tagged scalar-or-set membership",
                    )?;
                    self.store_reg_value(block_idx, rd, member)?;
                    return Ok(Some(block_idx));
                }
                // RecordSetBitmask membership (RecordSetBitmask step 2/5). The
                // set operand is a multi-slot record bitmask; the element is a
                // record. We compute, for each universe record, whether the
                // element equals it (AND of per-field equalities) and whether
                // its statically-known bit is set in the mask slots, then OR the
                // matches. This MUST `return` early — it never falls through to
                // the generic pointer scan (which would IntToPtr-dereference the
                // mask slots as a set base pointer: the rc=139 trap).
                super::AggregateShape::RecordSetBitmask {
                    universe_len,
                    slot_count,
                    universe,
                } => {
                    return self.lower_record_set_bitmask_membership(
                        block_idx,
                        rd,
                        elem_reg,
                        set_reg,
                        universe_len,
                        slot_count,
                        &universe,
                    );
                }
                _ => {}
            }
        }

        let elem_val = self.load_reg(block_idx, elem_reg)?;
        // WP-27 (item B1): the generic linear scan compares the element against
        // whatever raw values the producer materialized into the set's element
        // slots. There is no compile-time universe to prove that space against,
        // so a tagged-scalar-union element (which holds an INDEX, not a member)
        // fails closed here rather than comparing an index against members.
        let elem_val = self.set_in_element_raw_value_i64(
            block_idx,
            elem_reg,
            elem_val,
            &SetInRawSpace::Unproven,
            "SetIn: materialized-set membership",
        )?;
        let set_ptr = self.load_reg_as_ptr(block_idx, set_reg)?;
        let set_len = self.load_at_offset(block_idx, set_ptr, 0);

        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

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

        let loop_header = self.new_aux_block("setin_header");
        let loop_body = self.new_aux_block("setin_body");
        let loop_inc = self.new_aux_block("setin_inc");
        let found_blk = self.new_aux_block("setin_found");
        let not_found_blk = self.new_aux_block("setin_not_found");
        let merge_blk = self.new_aux_block("setin_merge");

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

        // Header: i < len?
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
                rhs: set_len,
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

        // Body: load set[i+1], compare with elem
        let cur_idx2 = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot_idx = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx2,
                rhs: one,
            },
        );
        let set_elem = self.load_at_dynamic_offset(loop_body, set_ptr, slot_idx);
        let eq = self.emit_with_result(
            loop_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: set_elem,
                rhs: elem_val,
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

        // Found: rd = 1
        self.store_reg_imm(found_blk, rd, 1)?;
        self.emit(
            found_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        // Not found: rd = 0
        self.store_reg_imm(not_found_blk, rd, 0)?;
        self.emit(
            not_found_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge_blk))
    }

    fn i64_value_as_ptr(&mut self, block_idx: usize, value: ValueId) -> ValueId {
        self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::IntToPtr,
                src_ty: Ty::I64,
                dst_ty: Ty::Ptr,
                operand: value,
            },
        )
    }

    fn emit_bool_to_i64(&mut self, block_idx: usize, value: ValueId) -> ValueId {
        self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: value,
            },
        )
    }

    pub(super) fn compact_set_bitmask_valid_mask(
        universe_len: u32,
        context: &str,
    ) -> Result<i64, TrustIrError> {
        match universe_len {
            0 => Ok(0),
            1..=62 => Ok((1_i64 << universe_len) - 1),
            63 => Ok(i64::MAX),
            _ => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetBitmask universe length {universe_len} exceeds i64 bitmask capacity"
            ))),
        }
    }

    pub(super) fn emit_compact_bitmask_canonical_i64(
        &mut self,
        block_idx: usize,
        mask: ValueId,
        universe_len: u32,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        let valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        let invalid_mask = self.emit_i64_const(block_idx, !valid_mask);
        let zero = self.emit_i64_const(block_idx, 0);
        let non_negative = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: mask,
                rhs: zero,
            },
        );
        let invalid_bits = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: mask,
                rhs: invalid_mask,
            },
        );
        let high_bits_clear = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: invalid_bits,
                rhs: zero,
            },
        );
        let non_negative_i64 = self.emit_bool_to_i64(block_idx, non_negative);
        let high_bits_clear_i64 = self.emit_bool_to_i64(block_idx, high_bits_clear);
        Ok(self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: non_negative_i64,
                rhs: high_bits_clear_i64,
            },
        ))
    }

    fn emit_compact_bitmask_powerset_membership_i64(
        &mut self,
        block_idx: usize,
        mask: ValueId,
        base_shape: &super::AggregateShape,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        base_shape.validate_powerset_base(context)?;
        let valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        let base_mask = if base_shape.compatible_set_bitmask_universe(universe_len, universe) {
            valid_mask
        } else if let Some(base_mask) =
            super::static_int_base_mask_for_set_bitmask_universe(base_shape, universe_len, universe)
        {
            base_mask
        } else if base_shape.matches_set_bitmask_base(universe_len, universe) {
            valid_mask
        } else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetBitmask universe does not match SUBSET base {base_shape:?}"
            )));
        };

        let canonical =
            self.emit_compact_bitmask_canonical_i64(block_idx, mask, universe_len, context)?;
        let invalid_base_mask = self.emit_i64_const(block_idx, !base_mask);
        let zero = self.emit_i64_const(block_idx, 0);
        let outside_base_bits = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: mask,
                rhs: invalid_base_mask,
            },
        );
        let subset = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: outside_base_bits,
                rhs: zero,
            },
        );
        let subset_i64 = self.emit_bool_to_i64(block_idx, subset);
        Ok(self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: canonical,
                rhs: subset_i64,
            },
        ))
    }

    fn lazy_domain_runtime_payload_is_compact_mask(shape: &super::AggregateShape) -> bool {
        match shape {
            super::AggregateShape::SetBitmask { .. } => true,
            super::AggregateShape::Powerset { base } => {
                matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. })
            }
            _ => false,
        }
    }

    fn guard_compact_sequence_dynamic_len_in_bounds(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        len_slot: ValueId,
        capacity: u32,
        context: &str,
    ) -> usize {
        let len_value = self.load_at_dynamic_offset(block_idx, source_ptr, len_slot);
        let zero = self.emit_i64_const(block_idx, 0);
        let non_negative = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: len_value,
                rhs: zero,
            },
        );

        let check_capacity_blk = self.new_aux_block(&format!("{context}_check_capacity"));
        let ok_blk = self.new_aux_block(&format!("{context}_ok"));
        let error_blk = self.new_aux_block(&format!("{context}_error"));
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

        let checked_len = self.load_at_dynamic_offset(check_capacity_blk, source_ptr, len_slot);
        let capacity_val = self.emit_i64_const(check_capacity_blk, i64::from(capacity));
        let within_capacity = self.emit_with_result(
            check_capacity_blk,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: checked_len,
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
        ok_blk
    }

    fn lower_compact_bitmask_runtime_powerset_mask_branch(
        &mut self,
        block_idx: usize,
        mask: ValueId,
        base_mask: ValueId,
        base_shape: &super::AggregateShape,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        if !base_shape.compatible_set_bitmask_universe(universe_len, universe) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetBitmask universe does not match runtime SUBSET base {base_shape:?}"
            )));
        }
        let all_in_universe = self.emit_i64_const(block_idx, 1);
        self.lower_set_bitmask_subseteq_mask_branch(
            block_idx,
            mask,
            all_in_universe,
            base_mask,
            universe_len,
            success_target,
            failure_target,
            context,
        )
    }

    fn lower_compact_bitmask_powerset_branch(
        &mut self,
        block_idx: usize,
        mask: ValueId,
        base_shape: &super::AggregateShape,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        let member = self.emit_compact_bitmask_powerset_membership_i64(
            block_idx,
            mask,
            base_shape,
            universe_len,
            universe,
            context,
        )?;
        self.branch_on_i64_truth(block_idx, member, success_target, failure_target);
        Ok(())
    }

    fn lower_scalar_in_set_bitmask_shape_branch(
        &mut self,
        block_idx: usize,
        value: ValueId,
        value_shape: Option<&super::AggregateShape>,
        mask: ValueId,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        if matches!(value_shape, Some(super::AggregateShape::SetBitmask { .. })) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetBitmask values are set-valued masks, not scalar elements"
            )));
        }
        let member = self.emit_compact_bitmask_set_membership_i64(
            block_idx,
            value,
            mask,
            universe_len,
            universe,
            context,
        )?;
        self.branch_on_i64_truth(block_idx, member, success_target, failure_target);
        Ok(())
    }

    fn lower_set_bitmask_subseteq_mask_branch(
        &mut self,
        block_idx: usize,
        left: ValueId,
        left_in_universe: ValueId,
        right: ValueId,
        universe_len: u32,
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        let valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        let valid_mask_val = self.emit_i64_const(block_idx, valid_mask);
        let right_complement = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Xor,
                ty: Ty::I64,
                lhs: right,
                rhs: valid_mask_val,
            },
        );
        let missing = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: left,
                rhs: right_complement,
            },
        );
        let zero = self.emit_i64_const(block_idx, 0);
        let no_missing = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: missing,
                rhs: zero,
            },
        );
        let no_missing_i64 = self.emit_bool_to_i64(block_idx, no_missing);
        let left_canonical =
            self.emit_compact_bitmask_canonical_i64(block_idx, left, universe_len, context)?;
        let right_canonical =
            self.emit_compact_bitmask_canonical_i64(block_idx, right, universe_len, context)?;
        let canonical = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: left_canonical,
                rhs: right_canonical,
            },
        );
        let subset = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: no_missing_i64,
                rhs: left_in_universe,
            },
        );
        let member = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: subset,
                rhs: canonical,
            },
        );
        self.branch_on_i64_truth(block_idx, member, success_target, failure_target);
        Ok(())
    }

    fn lower_scalar_in_function_set_range_shape_branch(
        &mut self,
        block_idx: usize,
        value: ValueId,
        value_shape: Option<&super::AggregateShape>,
        range_value: ValueId,
        range_shape: super::AggregateShape,
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        match range_shape {
            super::AggregateShape::Powerset { base }
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) =>
            {
                let Some(value_shape @ super::AggregateShape::SetBitmask { .. }) = value_shape
                else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SUBSET range requires SetBitmask entries, got {value_shape:?}"
                    )));
                };
                let Some((universe_len, universe)) = value_shape.set_bitmask_universe() else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SUBSET range requires exact universe metadata"
                    )));
                };
                self.lower_compact_bitmask_runtime_powerset_mask_branch(
                    block_idx,
                    value,
                    range_value,
                    &base,
                    universe_len,
                    &universe,
                    success_target,
                    failure_target,
                    context,
                )
            }
            // `(SUBSET S) \ {{}}` with a compact `SetBitmask` base. Identical to
            // the `Powerset` arm above for the subset half, but the candidate
            // must additionally be non-empty (bitmask != 0). We route the
            // subset-success edge through a guard block that enforces this.
            super::AggregateShape::NonEmptyPowerset { base }
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) =>
            {
                let Some(value_shape @ super::AggregateShape::SetBitmask { .. }) = value_shape
                else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact non-empty SUBSET range requires SetBitmask entries, got {value_shape:?}"
                    )));
                };
                let Some((universe_len, universe)) = value_shape.set_bitmask_universe() else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact non-empty SUBSET range requires exact universe metadata"
                    )));
                };
                let guard_blk = self.new_aux_block("nonempty_powerset_member_guard");
                let guard_id = self.block_id_of(guard_blk);
                self.lower_compact_bitmask_runtime_powerset_mask_branch(
                    block_idx,
                    value,
                    range_value,
                    &base,
                    universe_len,
                    &universe,
                    guard_id,
                    failure_target,
                    context,
                )?;
                self.branch_on_set_nonempty(
                    guard_blk,
                    value,
                    Some(value_shape),
                    success_target,
                    failure_target,
                    context,
                )
            }
            super::AggregateShape::SeqSet { base } => {
                let seq_element_shape = match value_shape {
                    Some(super::AggregateShape::Sequence { element, .. }) => element.as_deref(),
                    _ => None,
                };
                let base_value = if Self::lazy_domain_runtime_payload_is_compact_mask(&base) {
                    range_value
                } else {
                    self.i64_value_as_ptr(block_idx, range_value)
                };
                self.lower_seq_value_in_seq_set_ptr_branch(
                    block_idx,
                    value,
                    seq_element_shape,
                    base_value,
                    *base,
                    success_target,
                    failure_target,
                    context,
                )
            }
            super::AggregateShape::SetBitmask {
                universe_len,
                universe,
            } => self.lower_scalar_in_set_bitmask_shape_branch(
                block_idx,
                value,
                value_shape,
                range_value,
                universe_len,
                &universe,
                success_target,
                failure_target,
                context,
            ),
            // Lazy union range (lever L1): STATIC-ONLY membership over the
            // entry value — `range_value` is deliberately NOT passed through
            // (a lazy-union range slot holds an inert placeholder, and the
            // generic catch-all below would IntToPtr-scan it: amendment H1).
            super::AggregateShape::LazyUnion { left, right } => self
                .lower_entry_in_lazy_union_range_branch(
                    block_idx,
                    value,
                    value_shape,
                    &left,
                    &right,
                    success_target,
                    failure_target,
                    context,
                ),
            range_shape => {
                let range_ptr = self.i64_value_as_ptr(block_idx, range_value);
                self.lower_value_in_domain_ptr_branch(
                    block_idx,
                    value,
                    value_shape,
                    range_ptr,
                    range_shape,
                    success_target,
                    failure_target,
                    context,
                )
            }
        }
    }

    fn compact_powerset_mask_universe_for_value_shape(
        value_shape: Option<&super::AggregateShape>,
        context: &str,
    ) -> Result<(u32, super::SetBitmaskUniverse), TrustIrError> {
        match value_shape {
            Some(shape @ super::AggregateShape::SetBitmask { .. }) => {
                shape.set_bitmask_universe().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SUBSET range requires exact SetBitmask universe metadata"
                    ))
                })
            }
            Some(other) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SUBSET range requires SetBitmask entries, got {other:?}"
            ))),
            None => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SUBSET range requires tracked sequence entry shape"
            ))),
        }
    }

    fn emit_set_bitmask_universe_bit_i64(
        &mut self,
        block_idx: usize,
        elem_val: ValueId,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        if universe_len == 0 {
            return Ok(self.emit_i64_const(block_idx, 0));
        }
        match universe {
            super::SetBitmaskUniverse::IntRange { lo } => {
                self.emit_int_range_universe_bit_i64(block_idx, elem_val, *lo, universe_len)
            }
            super::SetBitmaskUniverse::ExplicitInt(values) => {
                self.emit_explicit_universe_bit_i64(block_idx, elem_val, values)
            }
            super::SetBitmaskUniverse::Exact(elements) => {
                let values = Self::exact_universe_compact_values(elements, context)?;
                self.emit_explicit_universe_bit_i64(block_idx, elem_val, &values)
            }
            super::SetBitmaskUniverse::Unknown => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetBitmask operation requires exact universe metadata"
            ))),
        }
    }

    fn emit_int_range_universe_bit_i64(
        &mut self,
        block_idx: usize,
        elem_val: ValueId,
        lo: i64,
        universe_len: u32,
    ) -> Result<ValueId, TrustIrError> {
        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);
        let lo_val = self.emit_i64_const(block_idx, lo);
        let hi = lo
            .checked_add(i64::from(universe_len).saturating_sub(1))
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "compact SetBitmask integer universe {lo} plus len {universe_len} overflows i64"
                ))
            })?;
        let hi_val = self.emit_i64_const(block_idx, hi);
        let ge_lo = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: elem_val,
                rhs: lo_val,
            },
        );
        let le_hi = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: elem_val,
                rhs: hi_val,
            },
        );
        let ge_lo_i64 = self.emit_bool_to_i64(block_idx, ge_lo);
        let le_hi_i64 = self.emit_bool_to_i64(block_idx, le_hi);
        let in_range_i64 = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: ge_lo_i64,
                rhs: le_hi_i64,
            },
        );
        let in_range = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: in_range_i64,
                rhs: zero,
            },
        );
        let raw_idx = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: elem_val,
                rhs: lo_val,
            },
        );
        let safe_idx = self.emit_with_result(
            block_idx,
            Inst::Select {
                ty: Ty::I64,
                cond: in_range,
                then_val: raw_idx,
                else_val: zero,
            },
        );
        let raw_bit = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Shl,
                ty: Ty::I64,
                lhs: one,
                rhs: safe_idx,
            },
        );
        Ok(self.emit_with_result(
            block_idx,
            Inst::Select {
                ty: Ty::I64,
                cond: in_range,
                then_val: raw_bit,
                else_val: zero,
            },
        ))
    }

    fn emit_explicit_universe_bit_i64(
        &mut self,
        block_idx: usize,
        elem_val: ValueId,
        values: &[i64],
    ) -> Result<ValueId, TrustIrError> {
        let mut mask = self.emit_i64_const(block_idx, 0);
        for (idx, element) in values.iter().copied().enumerate() {
            if idx >= 63 {
                return Err(TrustIrError::UnsupportedOpcode(
                    "compact SetBitmask explicit universe exceeds i64 capacity".to_owned(),
                ));
            }
            let expected = self.emit_i64_const(block_idx, element);
            let is_elem = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: elem_val,
                    rhs: expected,
                },
            );
            let bit = self.emit_i64_const(block_idx, 1_i64 << idx);
            let zero = self.emit_i64_const(block_idx, 0);
            let selected = self.emit_with_result(
                block_idx,
                Inst::Select {
                    ty: Ty::I64,
                    cond: is_elem,
                    then_val: bit,
                    else_val: zero,
                },
            );
            mask = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: mask,
                    rhs: selected,
                },
            );
        }
        Ok(mask)
    }

    fn exact_universe_compact_values(
        elements: &[super::SetBitmaskElement],
        context: &str,
    ) -> Result<Vec<i64>, TrustIrError> {
        let Some(first) = elements.first() else {
            return Ok(Vec::new());
        };
        let kind = Self::exact_universe_element_kind(first);
        let mut values = Vec::with_capacity(elements.len());
        for element in elements {
            if Self::exact_universe_element_kind(element) != kind {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask exact universe requires one scalar element kind"
                )));
            }
            values.push(match element {
                super::SetBitmaskElement::Int(value) => *value,
                super::SetBitmaskElement::Bool(value) => i64::from(*value),
                super::SetBitmaskElement::String(name)
                | super::SetBitmaskElement::ModelValue(name) => i64::from(name.0),
            });
        }
        Ok(values)
    }

    fn exact_universe_element_kind(element: &super::SetBitmaskElement) -> u8 {
        match element {
            super::SetBitmaskElement::Int(_) => 0,
            super::SetBitmaskElement::Bool(_) => 1,
            super::SetBitmaskElement::String(_) => 2,
            super::SetBitmaskElement::ModelValue(_) => 3,
        }
    }

    /// WP-27 (item B1): the [`SetInRawSpace`] a compact-bitmask universe
    /// compares against, or `None` when the universe carries no exact metadata
    /// (the caller has already rejected `Unknown` before reaching here).
    fn set_in_raw_space_of_universe(
        universe: &super::SetBitmaskUniverse,
        universe_len: u32,
        context: &str,
    ) -> Result<SetInRawSpace, TrustIrError> {
        Ok(match universe {
            super::SetBitmaskUniverse::IntRange { lo } => {
                // `emit_int_range_universe_bit_i64` accepts exactly `lo ..=
                // lo + universe_len - 1`; an empty universe accepts nothing.
                if universe_len == 0 {
                    SetInRawSpace::Exact {
                        kind: 0,
                        values: Vec::new(),
                    }
                } else {
                    let hi = lo
                        .checked_add(i64::from(universe_len).saturating_sub(1))
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "{context}: compact SetBitmask integer universe {lo} plus len \
                                 {universe_len} overflows i64"
                            ))
                        })?;
                    SetInRawSpace::IntRange { lo: *lo, hi }
                }
            }
            super::SetBitmaskUniverse::ExplicitInt(values) => SetInRawSpace::Exact {
                kind: 0,
                values: values.clone(),
            },
            super::SetBitmaskUniverse::Exact(elements) => {
                let Some(first) = elements.first() else {
                    return Ok(SetInRawSpace::Exact {
                        kind: 0,
                        values: Vec::new(),
                    });
                };
                SetInRawSpace::Exact {
                    kind: Self::exact_universe_element_kind(first),
                    // `exact_universe_compact_values` additionally enforces the
                    // one-kind rule the `Exact` arm already relies on.
                    values: Self::exact_universe_compact_values(elements, context)?,
                }
            }
            super::SetBitmaskUniverse::Unknown => SetInRawSpace::Unproven,
        })
    }

    /// WP-27 (item B1): prove that decoding a tagged-scalar-union ELEMENT index
    /// into raw member space cannot alias a cross-sort member into `space`.
    ///
    /// After [`Ctx::decode_scalar_key_reg_raw_value`], union slot `i` becomes
    /// `domain_key_raw_value(universe[i])`. For a member whose scalar kind
    /// MATCHES the consumer's, that raw is exactly the value the consumer's own
    /// universe would carry, so equality/range compares are exact. For a member
    /// of a DIFFERENT kind the raw is a bare `NameId` (or a 0/1 `Bool`) that
    /// carries no sort tag, so if it happens to land inside `space` the
    /// membership test would answer `TRUE` for a value the interpreter says is
    /// not a member — a silent divergence, not a fallback.
    ///
    /// So: reject unless every cross-kind member's raw lies OUTSIDE `space`.
    /// Same-kind members need no check. `Unproven` rejects unconditionally.
    fn check_set_in_union_element_sort(
        universe: &[super::SetBitmaskElement],
        space: &SetInRawSpace,
        elem_reg: u8,
        context: &str,
    ) -> Result<(), TrustIrError> {
        let (space_kind, in_space): (u8, &dyn Fn(i64) -> bool) = match space {
            SetInRawSpace::Unproven => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: element r{elem_reg} holds a tagged-scalar-union INDEX but the set \
                     operand has no compile-time universe, so the index cannot be decoded into the \
                     comparison's raw member space; failing closed to the interpreter"
                )));
            }
            SetInRawSpace::Exact { kind, values } => (*kind, &|raw| values.contains(&raw)),
            SetInRawSpace::IntRange { lo, hi } => (0, &|raw| raw >= *lo && raw <= *hi),
        };
        for element in universe {
            let kind = Self::exact_universe_element_kind(element);
            if kind == space_kind {
                continue;
            }
            let raw = Self::domain_key_raw_value(element);
            if in_space(raw) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: element r{elem_reg} is a tagged-scalar-union whose member \
                     {element:?} decodes to raw {raw}, which collides with the set operand's raw \
                     comparison space (kind {space_kind}); decoding the union index would make a \
                     non-member compare EQUAL. Failing closed to the interpreter"
                )));
            }
        }
        Ok(())
    }

    /// WP-27 (item B1): the raw member value a `SetIn` ELEMENT operand
    /// contributes to a raw-member-space membership test.
    ///
    /// Returns `elem_val` untouched for every register that is not
    /// tagged-scalar-union shaped, so this is a no-op on every arm and every
    /// action that was already correct. For a union-shaped element it applies
    /// WP-18's `decode_scalar_key_reg_raw_value` — but only after
    /// [`Ctx::check_set_in_union_element_sort`] proves the decode cannot alias
    /// a cross-sort member into `space`. See [`SetInRawSpace`].
    fn set_in_element_raw_value_i64(
        &mut self,
        block_idx: usize,
        elem_reg: u8,
        elem_val: ValueId,
        space: &SetInRawSpace,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        let Some(super::AggregateShape::TaggedScalarUnion { universe, .. }) =
            self.aggregate_shapes.get(&elem_reg)
        else {
            return Ok(elem_val);
        };
        let universe = universe.clone();
        Self::check_set_in_union_element_sort(&universe, space, elem_reg, context)?;
        Ok(self.decode_scalar_key_reg_raw_value(block_idx, elem_reg, elem_val))
    }

    fn emit_compact_bitmask_set_membership_i64(
        &mut self,
        block_idx: usize,
        elem_val: ValueId,
        mask: ValueId,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        let bit = self.emit_set_bitmask_universe_bit_i64(
            block_idx,
            elem_val,
            universe_len,
            universe,
            context,
        )?;
        let present_bits = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: mask,
                rhs: bit,
            },
        );
        let zero = self.emit_i64_const(block_idx, 0);
        let present = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: present_bits,
                rhs: zero,
            },
        );
        let present_i64 = self.emit_bool_to_i64(block_idx, present);
        let canonical =
            self.emit_compact_bitmask_canonical_i64(block_idx, mask, universe_len, context)?;
        Ok(self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: present_i64,
                rhs: canonical,
            },
        ))
    }

    pub(super) fn emit_tagged_scalar_or_set_mask_i64(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        context: &str,
    ) -> Result<(usize, ValueId), TrustIrError> {
        Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        let raw = self.load_reg(block_idx, reg)?;
        let zero = self.emit_i64_const(block_idx, 0);
        let is_set_branch = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: raw,
                rhs: zero,
            },
        );
        let decode_blk = self.new_aux_block("tagged_scalar_or_set_decode");
        let ok_blk = self.new_aux_block("tagged_scalar_or_set_ok");
        let error_blk = self.new_aux_block("tagged_scalar_or_set_type_error");
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: is_set_branch,
                then_target: self.block_id_of(decode_blk),
                then_args: vec![],
                else_target: self.block_id_of(error_blk),
                else_args: vec![],
            }),
        );

        let neg_one = self.emit_i64_const(decode_blk, -1);
        let mask = self.emit_with_result(
            decode_blk,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: neg_one,
                rhs: raw,
            },
        );
        let canonical =
            self.emit_compact_bitmask_canonical_i64(decode_blk, mask, universe_len, context)?;
        self.branch_on_i64_truth(
            decode_blk,
            canonical,
            self.block_id_of(ok_blk),
            self.block_id_of(error_blk),
        );
        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);
        Ok((ok_blk, mask))
    }

    // =====================================================================
    // LazyUnion membership (lever L1) — static-only lowering helpers.
    //
    // Soundness contract (amendment H1): every helper below consumes ONLY the
    // candidate element's runtime value plus COMPILE-TIME arm metadata from
    // the `AggregateShape::LazyUnion` shape. None of them takes a set
    // register or a funcset slot-1 `range_value` payload — the LazyUnion
    // register (and the funcset range slot mirroring it) holds an inert
    // placeholder `0` that must never be loaded as data.
    // =====================================================================

    /// Flatten a (possibly nested) lazy union into its leaf arms.
    fn flattened_lazy_union_arms(
        left: &super::AggregateShape,
        right: &super::AggregateShape,
    ) -> Vec<super::AggregateShape> {
        fn push(shape: &super::AggregateShape, arms: &mut Vec<super::AggregateShape>) {
            if let super::AggregateShape::LazyUnion { left, right } = shape {
                push(left, arms);
                push(right, arms);
            } else {
                arms.push(shape.clone());
            }
        }
        let mut arms = Vec::new();
        push(left, &mut arms);
        push(right, &mut arms);
        arms
    }

    /// Compile-time element values of the SCALAR-sort arms matching `sort`.
    ///
    /// Strict sort matching (soundness amendment H5): `String` and
    /// `ModelValue` intern to the SAME NameId, so a `String` candidate must
    /// never satisfy a `ModelValue` arm (and vice versa); likewise `Bool`
    /// values 0/1 must never satisfy an `Int` arm. Set-sort arms (powersets)
    /// are skipped — value-sort disjointness makes them unsatisfiable by a
    /// scalar.
    fn lazy_union_scalar_arm_values(
        arms: &[super::AggregateShape],
        sort: &super::ScalarShape,
    ) -> Vec<i64> {
        let mut values = Vec::new();
        for arm in arms {
            match arm {
                super::AggregateShape::ExactScalarSet {
                    scalar,
                    values: arm_values,
                } if scalar == sort => values.extend_from_slice(arm_values),
                super::AggregateShape::ExactIntSet { values: arm_values }
                    if matches!(sort, super::ScalarShape::Int) =>
                {
                    values.extend_from_slice(arm_values);
                }
                super::AggregateShape::Interval { lo, hi }
                    if matches!(sort, super::ScalarShape::Int) =>
                {
                    // Bounded by lazy-union admissibility (arm length is
                    // capped at MAX_LAZY_POWERSET_BASE_LEN at union time).
                    let mut candidate = *lo;
                    while candidate <= *hi {
                        values.push(candidate);
                        let Some(next) = candidate.checked_add(1) else {
                            break;
                        };
                        candidate = next;
                    }
                }
                _ => {}
            }
        }
        values.sort_unstable();
        values.dedup();
        values
    }

    /// WP-08 (item 6): the SymbolicDomain arms of a lazy union.
    ///
    /// Admitted into a lazy union by `lazy_union_arm_admissible` for
    /// `Int`/`Nat` only, and consumed EXCLUSIVELY by the scalar-candidate
    /// membership paths below (membership-only contract): every materialized
    /// consumer of the union still fails closed on the `LazyUnion` shape
    /// itself (`reject_lazy_set_operand` / `load_reg_as_ptr` walls).
    fn lazy_union_symbolic_domain_arms(
        arms: &[super::AggregateShape],
    ) -> Vec<super::SymbolicDomain> {
        arms.iter()
            .filter_map(|arm| match arm {
                super::AggregateShape::SymbolicDomain(domain) => Some(*domain),
                _ => None,
            })
            .collect()
    }

    /// WP-08 (item 6): scalar-candidate membership in the scalar-sort arms of
    /// a lazy union — the static exact-element compare chain OR'd with the
    /// membership of every SymbolicDomain arm (sort-strict: a non-numeric
    /// candidate contributes const 0 for a numeric domain, mirroring
    /// `emit_symbolic_domain_membership_i64`).
    fn emit_scalar_in_lazy_union_scalar_arms_i64(
        &mut self,
        block_idx: usize,
        value: ValueId,
        sort: &super::ScalarShape,
        arms: &[super::AggregateShape],
    ) -> Result<ValueId, TrustIrError> {
        let values = Self::lazy_union_scalar_arm_values(arms, sort);
        let mut member = self.emit_scalar_in_static_values_i64(block_idx, value, &values);
        let candidate_shape = super::AggregateShape::Scalar(sort.clone());
        for domain in Self::lazy_union_symbolic_domain_arms(arms) {
            let domain_member = self.emit_symbolic_domain_membership_i64(
                block_idx,
                value,
                Some(&candidate_shape),
                domain,
            )?;
            member = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: member,
                    rhs: domain_member,
                },
            );
        }
        Ok(member)
    }

    /// Enumerate a compact set universe as TYPED elements in bit-index order.
    fn lazy_union_universe_typed_elements(
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<Vec<super::SetBitmaskElement>, TrustIrError> {
        let len = usize::try_from(universe_len).map_err(|_| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact set universe length {universe_len} overflows usize"
            ))
        })?;
        match universe {
            super::SetBitmaskUniverse::Exact(elements) => {
                if elements.len() != len {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: exact universe element count {} does not match universe length {universe_len}",
                        elements.len()
                    )));
                }
                Ok(elements.clone())
            }
            super::SetBitmaskUniverse::ExplicitInt(values) => {
                if values.len() != len {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: explicit integer universe element count {} does not match universe length {universe_len}",
                        values.len()
                    )));
                }
                Ok(values
                    .iter()
                    .map(|value| super::SetBitmaskElement::Int(*value))
                    .collect())
            }
            super::SetBitmaskUniverse::IntRange { lo } => (0..i64::from(universe_len))
                .map(|offset| {
                    lo.checked_add(offset)
                        .map(super::SetBitmaskElement::Int)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "{context}: integer-range universe {lo}+{offset} overflows i64"
                            ))
                        })
                })
                .collect(),
            super::SetBitmaskUniverse::Unknown => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: lazy-union set membership requires exact universe metadata"
            ))),
        }
    }

    /// Sort-aware compile-time membership of one typed universe element in a
    /// FULLY STATIC powerset base (soundness amendment H5: strict sorts).
    fn lazy_union_element_in_static_base(
        element: &super::SetBitmaskElement,
        base: &super::AggregateShape,
        context: &str,
    ) -> Result<bool, TrustIrError> {
        match base {
            super::AggregateShape::ExactIntSet { values } => Ok(matches!(
                element,
                super::SetBitmaskElement::Int(value) if values.contains(value)
            )),
            super::AggregateShape::Interval { lo, hi } => Ok(matches!(
                element,
                super::SetBitmaskElement::Int(value) if *lo <= *value && *value <= *hi
            )),
            super::AggregateShape::ExactScalarSet { scalar, values } => {
                Ok(match (scalar, element) {
                    (super::ScalarShape::String, super::SetBitmaskElement::String(name))
                    | (
                        super::ScalarShape::ModelValue,
                        super::SetBitmaskElement::ModelValue(name),
                    ) => values.contains(&i64::from(name.0)),
                    (super::ScalarShape::Bool, super::SetBitmaskElement::Bool(value)) => {
                        values.contains(&i64::from(*value))
                    }
                    (super::ScalarShape::Int, super::SetBitmaskElement::Int(value)) => {
                        values.contains(value)
                    }
                    _ => false,
                })
            }
            other => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: lazy-union powerset arm base must be a fully static set, got {other:?}"
            ))),
        }
    }

    /// Compile-time per-arm allowed masks for the SET-sort arms of a lazy
    /// union, projected onto the candidate's compact universe.
    ///
    /// Per-arm disjunction is REQUIRED (soundness amendment H3):
    /// `x \in (SUBSET A) \cup (SUBSET B)` is NOT `x \subseteq A \cup B`, so
    /// the arms' masks must never be merged. Each entry is
    /// `(allowed_mask, require_nonempty)`; membership in arm `i` is
    /// `mask & !allowed_i == 0` (AND `mask != 0` for `NonEmptyPowerset`).
    fn lazy_union_set_arm_masks(
        arms: &[super::AggregateShape],
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<Vec<LazyUnionSetArm>, TrustIrError> {
        let set_arms: Vec<(&super::AggregateShape, bool)> = arms
            .iter()
            .filter_map(|arm| match arm {
                super::AggregateShape::Powerset { base } => Some((base.as_ref(), false)),
                super::AggregateShape::NonEmptyPowerset { base } => Some((base.as_ref(), true)),
                _ => None,
            })
            .collect();
        if set_arms.is_empty() {
            return Ok(Vec::new());
        }
        Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        let elements = Self::lazy_union_universe_typed_elements(universe_len, universe, context)?;
        let mut result = Vec::with_capacity(set_arms.len());
        for (base, require_nonempty) in set_arms {
            let mut allowed_mask = 0_i64;
            for (index, element) in elements.iter().enumerate() {
                if Self::lazy_union_element_in_static_base(element, base, context)? {
                    allowed_mask |= 1_i64 << index;
                }
            }
            result.push(LazyUnionSetArm {
                allowed_mask,
                require_nonempty,
            });
        }
        Ok(result)
    }

    /// Emit `value \in {values...}` as a compile-time cmp/or chain (i64 0/1).
    fn emit_scalar_in_static_values_i64(
        &mut self,
        block_idx: usize,
        value: ValueId,
        values: &[i64],
    ) -> ValueId {
        let mut member = self.emit_i64_const(block_idx, 0);
        for candidate in values {
            let expected = self.emit_i64_const(block_idx, *candidate);
            let eq = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: value,
                    rhs: expected,
                },
            );
            let eq_i64 = self.emit_bool_to_i64(block_idx, eq);
            member = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: member,
                    rhs: eq_i64,
                },
            );
        }
        member
    }

    /// Branch on membership of a compact set-bitmask candidate in the SET-sort
    /// arms of a lazy union: short-circuit per-arm disjunction (amendment H3).
    ///
    /// A candidate whose mask is non-canonical is not a member of any arm
    /// (mirrors `emit_compact_bitmask_set_membership_i64`'s AND-with-canonical
    /// convention) unless `mask_known_canonical` says the caller already
    /// established canonicality.
    #[allow(clippy::too_many_arguments)]
    fn lower_lazy_union_set_mask_membership_branch(
        &mut self,
        block_idx: usize,
        mask: ValueId,
        universe_len: u32,
        arms: &[LazyUnionSetArm],
        mask_known_canonical: bool,
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        if arms.is_empty() {
            // No set-sort arm: a set-valued candidate can never be a member
            // (TLC value-sort disjointness).
            self.emit(
                block_idx,
                InstrNode::new(Inst::Br {
                    target: failure_target,
                    args: vec![],
                }),
            );
            return Ok(());
        }
        let mut cur_block = block_idx;
        if !mask_known_canonical {
            let canonical =
                self.emit_compact_bitmask_canonical_i64(cur_block, mask, universe_len, context)?;
            let arms_blk = self.new_aux_block("lazy_union_set_arms");
            let arms_id = self.block_id_of(arms_blk);
            self.branch_on_i64_truth(cur_block, canonical, arms_id, failure_target);
            cur_block = arms_blk;
        }
        for (index, arm) in arms.iter().enumerate() {
            let is_last = index + 1 == arms.len();
            let next_block = if is_last {
                None
            } else {
                Some(self.new_aux_block("lazy_union_set_arm_next"))
            };
            let next_target =
                next_block.map_or(failure_target, |next_block| self.block_id_of(next_block));
            let not_allowed = self.emit_i64_const(cur_block, !arm.allowed_mask);
            let outside_bits = self.emit_with_result(
                cur_block,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: mask,
                    rhs: not_allowed,
                },
            );
            let zero = self.emit_i64_const(cur_block, 0);
            let subset = self.emit_with_result(
                cur_block,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: outside_bits,
                    rhs: zero,
                },
            );
            let subset_i64 = self.emit_bool_to_i64(cur_block, subset);
            let arm_ok = if arm.require_nonempty {
                // The tagged/bitmask encoding represents {} as mask 0; a
                // `(SUBSET S) \ {{}}` arm additionally requires mask != 0.
                let nonempty = self.emit_with_result(
                    cur_block,
                    Inst::ICmp {
                        op: ICmpOp::Ne,
                        ty: Ty::I64,
                        lhs: mask,
                        rhs: zero,
                    },
                );
                let nonempty_i64 = self.emit_bool_to_i64(cur_block, nonempty);
                self.emit_with_result(
                    cur_block,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: subset_i64,
                        rhs: nonempty_i64,
                    },
                )
            } else {
                subset_i64
            };
            self.branch_on_i64_truth(cur_block, arm_ok, success_target, next_target);
            if let Some(next_block) = next_block {
                cur_block = next_block;
            }
        }
        Ok(())
    }

    /// Branch on membership of a tagged `scalar | set` slot in a lazy union.
    ///
    /// Branching mirror of `emit_tagged_scalar_or_set_mask_i64`'s decode
    /// split: negative payload = set lane (`mask = -1 - raw`), non-negative =
    /// scalar lane. The scalar lane tests the raw slot against the strict
    /// same-sort scalar arm elements; the set lane runs the per-arm mask
    /// disjunction. A lane with no matching-sort arm branches to failure
    /// (sound: TLC value-sort disjointness).
    #[allow(clippy::too_many_arguments)]
    fn lower_tagged_slot_in_lazy_union_branch(
        &mut self,
        block_idx: usize,
        raw: ValueId,
        scalar_sort: &super::ScalarShape,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        arms: &[super::AggregateShape],
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        let has_set_arms = arms.iter().any(|arm| {
            matches!(
                arm,
                super::AggregateShape::Powerset { .. }
                    | super::AggregateShape::NonEmptyPowerset { .. }
            )
        });
        // Only demand exact universe metadata when a set-sort arm exists; a
        // scalar-only union never inspects the set lane's universe.
        let set_arms = if has_set_arms {
            Self::lazy_union_set_arm_masks(arms, universe_len, universe, context)?
        } else {
            Vec::new()
        };

        let scalar_blk = self.new_aux_block("lazy_union_tagged_scalar_lane");
        let set_blk = self.new_aux_block("lazy_union_tagged_set_lane");
        let scalar_id = self.block_id_of(scalar_blk);
        let set_id = self.block_id_of(set_blk);
        let zero = self.emit_i64_const(block_idx, 0);
        let is_set_branch = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: raw,
                rhs: zero,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: is_set_branch,
                then_target: set_id,
                then_args: vec![],
                else_target: scalar_id,
                else_args: vec![],
            }),
        );

        // Scalar lane: the raw slot is the (non-negative) scalar encoding.
        let member =
            self.emit_scalar_in_lazy_union_scalar_arms_i64(scalar_blk, raw, scalar_sort, arms)?;
        self.branch_on_i64_truth(scalar_blk, member, success_target, failure_target);

        // Set lane.
        if set_arms.is_empty() {
            self.emit(
                set_blk,
                InstrNode::new(Inst::Br {
                    target: failure_target,
                    args: vec![],
                }),
            );
            return Ok(());
        }
        let neg_one = self.emit_i64_const(set_blk, -1);
        let mask = self.emit_with_result(
            set_blk,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: neg_one,
                rhs: raw,
            },
        );
        // A non-canonical negative payload is a corrupt slot no writer can
        // produce; mirror `emit_tagged_scalar_or_set_mask_i64` and raise a
        // runtime type error rather than guess a truth value.
        let canonical =
            self.emit_compact_bitmask_canonical_i64(set_blk, mask, universe_len, context)?;
        let arms_blk = self.new_aux_block("lazy_union_tagged_set_arms");
        let error_blk = self.new_aux_block("lazy_union_tagged_set_type_error");
        let arms_id = self.block_id_of(arms_blk);
        let error_id = self.block_id_of(error_blk);
        self.branch_on_i64_truth(set_blk, canonical, arms_id, error_id);
        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);
        self.lower_lazy_union_set_mask_membership_branch(
            arms_blk,
            mask,
            universe_len,
            &set_arms,
            true,
            success_target,
            failure_target,
            context,
        )
    }

    /// Branch on membership of one candidate value in a lazy union, given the
    /// candidate's tracked shape. This is the single entry point for every
    /// LazyUnion consumer (`SetIn`, the compact/non-compact function-set range
    /// checks): it takes the candidate VALUE and the union's STATIC arm
    /// metadata only — never a set register or funcset range payload
    /// (soundness amendment H1). Unclassifiable candidates fail closed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_entry_in_lazy_union_range_branch(
        &mut self,
        block_idx: usize,
        entry_value: ValueId,
        entry_shape: Option<&super::AggregateShape>,
        left: &super::AggregateShape,
        right: &super::AggregateShape,
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        let arms = Self::flattened_lazy_union_arms(left, right);
        match entry_shape {
            Some(super::AggregateShape::TaggedScalarOrSet {
                scalar,
                universe_len,
                universe,
                ..
            }) => self.lower_tagged_slot_in_lazy_union_branch(
                block_idx,
                entry_value,
                scalar,
                *universe_len,
                universe,
                &arms,
                success_target,
                failure_target,
                context,
            ),
            Some(shape @ super::AggregateShape::SetBitmask { .. }) => {
                let Some((universe_len, universe)) = shape.set_bitmask_universe() else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: lazy-union set membership requires exact universe metadata"
                    )));
                };
                let set_arms =
                    Self::lazy_union_set_arm_masks(&arms, universe_len, &universe, context)?;
                self.lower_lazy_union_set_mask_membership_branch(
                    block_idx,
                    entry_value,
                    universe_len,
                    &set_arms,
                    false,
                    success_target,
                    failure_target,
                    context,
                )
            }
            Some(super::AggregateShape::Scalar(sort)) => {
                let sort = sort.clone();
                let member = self.emit_scalar_in_lazy_union_scalar_arms_i64(
                    block_idx,
                    entry_value,
                    &sort,
                    &arms,
                )?;
                self.branch_on_i64_truth(block_idx, member, success_target, failure_target);
                Ok(())
            }
            Some(super::AggregateShape::ScalarIntDomain { .. }) => {
                let member = self.emit_scalar_in_lazy_union_scalar_arms_i64(
                    block_idx,
                    entry_value,
                    &super::ScalarShape::Int,
                    &arms,
                )?;
                self.branch_on_i64_truth(block_idx, member, success_target, failure_target);
                Ok(())
            }
            other => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: lazy-union membership requires a tagged scalar-or-set slot, a \
                 compact SetBitmask mask, or a typed scalar candidate, got {other:?}"
            ))),
        }
    }

    fn emit_setdiff_operand_bitmask_i64_allow_tagged_or_materialized(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<(usize, ValueId), TrustIrError> {
        if let Some(shape @ super::AggregateShape::TaggedScalarOrSet { .. }) =
            self.aggregate_shapes.get(&reg).cloned()
        {
            if shape.tagged_set_branch_universe() != Some((universe_len, universe.clone())) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: tagged scalar-or-set universe mismatch for r{reg}: {shape:?}"
                )));
            }
            return self.emit_tagged_scalar_or_set_mask_i64(block_idx, reg, universe_len, context);
        }
        self.emit_set_operand_bitmask_i64_allow_materialized(
            block_idx,
            reg,
            universe_len,
            universe,
            context,
        )
    }

    /// WP-08 (item 6): capacity of a DYNAMIC materialized small-set source
    /// (elements without static provenance) that the fail-closed runtime
    /// loop [`Self::emit_dynamic_materialized_set_bitmask_mask_i64`] can
    /// convert into a compact SetBitmask mask over `universe`.
    ///
    /// Admission is deliberately strict: the element shape must be a scalar
    /// whose sort EQUALS the universe's scalar sort
    /// (`materialized_set_element_disjoint_from_universe == Some(false)`), so
    /// a raw i64 equality/range compare against the universe encoding can
    /// never cross-sort alias (e.g. a String NameId numerically colliding
    /// with an Int universe value). Everything else returns `None` and the
    /// caller keeps today's fail-closed walls.
    pub(super) fn dynamic_set_to_bitmask_source_capacity(
        &self,
        reg: u8,
        universe: &super::SetBitmaskUniverse,
    ) -> Option<u32> {
        let (capacity, element) = match self.aggregate_shapes.get(&reg)? {
            super::AggregateShape::Set {
                len,
                element: Some(element),
            } if *len > 0 => (*len, element.as_ref()),
            super::AggregateShape::BoundedSet {
                max_len,
                element: Some(element),
            } if *max_len > 0 => (*max_len, element.as_ref()),
            _ => return None,
        };
        if super::materialized_set_element_disjoint_from_universe(element, universe) != Some(false)
        {
            return None;
        }
        Some(capacity)
    }

    /// WP-08 (item 6): runtime FAIL-CLOSED conversion of a dynamic
    /// materialized set register (a `Set`/`BoundedSet` whose element values
    /// are only known at runtime) into a compact SetBitmask mask.
    ///
    /// Generalizes the two shipped templates
    /// (`compact_scalar_domain_set_replacement_source` and
    /// `materialized_set_as_tagged_scalar_or_set_value` in functions.rs) for
    /// an arbitrary `SetBitmaskUniverse` destination: per element, the
    /// universe bit is computed with
    /// [`Self::emit_set_bitmask_universe_bit_i64`]; an element OUTSIDE the
    /// universe (bit == 0) branches to a typed runtime error
    /// (`TypeMismatch`, per-state interpreter fallback) — NEVER the silent
    /// Select-zero drop, which is sound only on the SetDiff-RHS path.
    ///
    /// The caller must have admitted the source via
    /// [`Self::dynamic_set_to_bitmask_source_capacity`] (same-sort scalar
    /// elements), so the raw i64 compare inside the universe-bit helper is
    /// sort-sound.
    pub(super) fn emit_dynamic_materialized_set_bitmask_mask_i64(
        &mut self,
        block_idx: usize,
        reg: u8,
        capacity: u32,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<(usize, ValueId), TrustIrError> {
        Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        let source_ptr = self.load_reg_as_ptr(block_idx, reg)?;
        let len_value = self.load_at_offset(block_idx, source_ptr, 0);
        let guard: super::CompactSequenceLenGuardResult = self
            .guard_compact_sequence_len_in_bounds(
                block_idx,
                len_value,
                capacity,
                "dynamic_set_to_bitmask_mask",
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

        let header_blk = self.new_aux_block("dynamic_set_bitmask_header");
        let body_blk = self.new_aux_block("dynamic_set_bitmask_body");
        let accept_blk = self.new_aux_block("dynamic_set_bitmask_accept");
        let error_blk = self.new_aux_block("dynamic_set_bitmask_error");
        let done_blk = self.new_aux_block("dynamic_set_bitmask_done");
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
        let elem_bit =
            self.emit_set_bitmask_universe_bit_i64(body_blk, elem, universe_len, universe, context)?;
        let zero_body = self.emit_i64_const(body_blk, 0);
        // Every in-universe element yields exactly one nonzero bit, so
        // `elem_bit == 0` iff the element lies outside the universe.
        let present = self.emit_with_result(
            body_blk,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: elem_bit,
                rhs: zero_body,
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
        Ok((done_blk, final_mask))
    }

    fn emit_set_operand_bitmask_i64_allow_materialized(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<(usize, ValueId), TrustIrError> {
        match self.aggregate_shapes.get(&reg).cloned() {
            Some(shape @ super::AggregateShape::SetBitmask { .. }) => {
                if !shape.compatible_set_bitmask_universe(universe_len, universe) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SetBitmask universe mismatch for r{reg}: {shape:?}"
                    )));
                }
                Ok((block_idx, self.load_reg(block_idx, reg)?))
            }
            Some(super::AggregateShape::ExactIntSet { values }) => {
                let mask = self.emit_exact_int_set_operand_bitmask_i64(
                    block_idx,
                    &values,
                    universe_len,
                    universe,
                    context,
                    true,
                )?;
                Ok((block_idx, mask))
            }
            Some(super::AggregateShape::ExactScalarSet { scalar, values }) => {
                let mask = self.emit_exact_scalar_set_operand_bitmask_i64(
                    block_idx,
                    &scalar,
                    &values,
                    universe_len,
                    universe,
                    context,
                    true,
                )?;
                Ok((block_idx, mask))
            }
            Some(super::AggregateShape::Set { len: 0, .. }) => {
                Ok((block_idx, self.emit_i64_const(block_idx, 0)))
            }
            // WP-08 (item 6): a DYNAMIC materialized set operand (elements
            // without static provenance, e.g. `@ \union {key}` with a runtime
            // `key`) converts through the fail-closed runtime loop — every
            // element must hit a universe bit or the state takes a typed
            // runtime error (interpreter fallback), never a silent drop.
            Some(
                super::AggregateShape::Set { .. } | super::AggregateShape::BoundedSet { .. },
            ) if self
                .dynamic_set_to_bitmask_source_capacity(reg, universe)
                .is_some() =>
            {
                let capacity = self
                    .dynamic_set_to_bitmask_source_capacity(reg, universe)
                    .expect("guard above established convertibility");
                self.emit_dynamic_materialized_set_bitmask_mask_i64(
                    block_idx,
                    reg,
                    capacity,
                    universe_len,
                    universe,
                    context,
                )
            }
            Some(super::AggregateShape::Set { .. }) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact set operation cannot infer that materialized Set r{reg} is confined to the compact SetBitmask universe"
            ))),
            Some(super::AggregateShape::Interval { lo, hi }) => {
                let mask = self.emit_interval_bitmask_i64(
                    block_idx,
                    lo,
                    hi,
                    universe_len,
                    universe,
                    context,
                )?;
                Ok((block_idx, mask))
            }
            Some(shape) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact set operation requires SetBitmask or compatible materialized integer Set operand, got r{reg} = {shape:?}"
            ))),
            None => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact set operation requires tracked set operand r{reg}"
            ))),
        }
    }

    fn emit_setdiff_rhs_bitmask_i64_allow_tagged_or_materialized(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<(usize, ValueId), TrustIrError> {
        if let Some(shape @ super::AggregateShape::TaggedScalarOrSet { .. }) =
            self.aggregate_shapes.get(&reg).cloned()
        {
            if shape.tagged_set_branch_universe() != Some((universe_len, universe.clone())) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: tagged scalar-or-set universe mismatch for r{reg}: {shape:?}"
                )));
            }
            return self.emit_tagged_scalar_or_set_mask_i64(block_idx, reg, universe_len, context);
        }

        match self.aggregate_shapes.get(&reg).cloned() {
            Some(shape @ super::AggregateShape::SetBitmask { .. }) => {
                if !shape.compatible_set_bitmask_universe(universe_len, universe) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SetBitmask universe mismatch for r{reg}: {shape:?}"
                    )));
                }
                Ok((block_idx, self.load_reg(block_idx, reg)?))
            }
            Some(super::AggregateShape::ExactIntSet { values }) => {
                if super::set_bitmask_universe_accepts_integer_values(universe) {
                    let mask = self.emit_exact_int_set_operand_bitmask_i64(
                        block_idx,
                        &values,
                        universe_len,
                        universe,
                        context,
                        false,
                    )?;
                    Ok((block_idx, mask))
                } else if let Some(true) =
                    super::integer_values_disjoint_from_set_bitmask_universe(universe)
                {
                    Ok((block_idx, self.emit_i64_const(block_idx, 0)))
                } else {
                    Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SetDiff RHS exact integer set cannot be safely mapped into non-integer compact universe {universe:?}"
                    )))
                }
            }
            Some(super::AggregateShape::ExactScalarSet { scalar, values }) => {
                if let Some(true) =
                    super::scalar_values_disjoint_from_set_bitmask_universe(&scalar, universe)
                {
                    Ok((block_idx, self.emit_i64_const(block_idx, 0)))
                } else {
                    let mask = self.emit_exact_scalar_set_operand_bitmask_i64(
                        block_idx,
                        &scalar,
                        &values,
                        universe_len,
                        universe,
                        context,
                        false,
                    )?;
                    Ok((block_idx, mask))
                }
            }
            Some(super::AggregateShape::Interval { lo, hi }) => {
                if super::set_bitmask_universe_accepts_integer_values(universe) {
                    let mask = self.emit_interval_bitmask_i64_allow_clamped(
                        block_idx,
                        lo,
                        hi,
                        universe_len,
                        universe,
                        context,
                    )?;
                    Ok((block_idx, mask))
                } else if let Some(true) =
                    super::integer_values_disjoint_from_set_bitmask_universe(universe)
                {
                    Ok((block_idx, self.emit_i64_const(block_idx, 0)))
                } else {
                    Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SetDiff RHS interval cannot be safely mapped into non-integer compact universe {universe:?}"
                    )))
                }
            }
            Some(super::AggregateShape::Set { len: 0, .. }) => {
                Ok((block_idx, self.emit_i64_const(block_idx, 0)))
            }
            Some(super::AggregateShape::Set {
                len,
                element: Some(element),
            }) => {
                let Some(disjoint) =
                    super::materialized_set_element_disjoint_from_universe(&element, universe)
                else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SetDiff RHS r{reg} requires a scalar materialized Set element shape, got {element:?}"
                    )));
                };
                if disjoint {
                    return Ok((block_idx, self.emit_i64_const(block_idx, 0)));
                }

                let set_ptr = self.load_reg_as_ptr(block_idx, reg)?;
                let mut mask = self.emit_i64_const(block_idx, 0);
                for slot in 0..len {
                    let elem = self.load_at_offset(block_idx, set_ptr, slot + 1);
                    let bit = self.emit_set_bitmask_universe_bit_i64(
                        block_idx,
                        elem,
                        universe_len,
                        universe,
                        context,
                    )?;
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
                Ok((block_idx, mask))
            }
            Some(super::AggregateShape::Set { .. }) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetDiff RHS r{reg} requires a scalar materialized Set element shape"
            ))),
            Some(shape) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetDiff RHS requires SetBitmask or compatible materialized set operand, got r{reg} = {shape:?}"
            ))),
            None => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetDiff RHS requires tracked set operand r{reg}"
            ))),
        }
    }

    fn emit_exact_int_set_operand_bitmask_i64(
        &mut self,
        block_idx: usize,
        values: &[i64],
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
        require_all_in_universe: bool,
    ) -> Result<ValueId, TrustIrError> {
        Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        let mut mask = 0_i64;
        for value in values {
            let bit_idx = match universe {
                super::SetBitmaskUniverse::IntRange { lo } => value
                    .checked_sub(*lo)
                    .filter(|idx| *idx >= 0 && *idx < i64::from(universe_len))
                    .and_then(|idx| u32::try_from(idx).ok()),
                super::SetBitmaskUniverse::ExplicitInt(universe_values) => universe_values
                    .iter()
                    .position(|elem| elem == value)
                    .filter(|idx| *idx < usize::try_from(universe_len).unwrap_or(usize::MAX))
                    .and_then(|idx| u32::try_from(idx).ok()),
                super::SetBitmaskUniverse::Exact(elements) => {
                    match super::integer_values_disjoint_from_set_bitmask_universe(universe) {
                        Some(true) if require_all_in_universe => {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "{context}: exact integer Set cannot fit non-integer compact SetBitmask universe"
                            )));
                        }
                        Some(true) => None,
                        Some(false) => {
                            let universe_values =
                                Self::exact_universe_compact_values(elements, context)?;
                            universe_values
                                .iter()
                                .position(|elem| elem == value)
                                .filter(|idx| {
                                    *idx < usize::try_from(universe_len).unwrap_or(usize::MAX)
                                })
                                .and_then(|idx| u32::try_from(idx).ok())
                        }
                        None => {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "{context}: exact integer Set cannot be safely mapped into mixed or unknown compact SetBitmask universe"
                            )));
                        }
                    }
                }
                super::SetBitmaskUniverse::Unknown => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SetBitmask operation requires exact universe metadata"
                    )));
                }
            };
            let Some(bit_idx) = bit_idx else {
                if require_all_in_universe {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: exact integer Set contains values outside compact SetBitmask universe"
                    )));
                }
                continue;
            };
            mask |= 1_i64 << bit_idx;
        }
        Ok(self.emit_i64_const(block_idx, mask))
    }

    fn emit_exact_scalar_set_operand_bitmask_i64(
        &mut self,
        block_idx: usize,
        scalar: &super::ScalarShape,
        values: &[i64],
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
        require_all_in_universe: bool,
    ) -> Result<ValueId, TrustIrError> {
        Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        match super::scalar_values_disjoint_from_set_bitmask_universe(scalar, universe) {
            Some(true) if require_all_in_universe => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: exact scalar Set cannot fit disjoint compact SetBitmask universe"
                )));
            }
            Some(true) => return Ok(self.emit_i64_const(block_idx, 0)),
            Some(false) => {}
            None => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: exact scalar Set cannot be safely mapped into mixed or unknown compact SetBitmask universe"
                )));
            }
        }

        let mut mask = 0_i64;
        for value in values {
            let bit_idx =
                super::set_bitmask_scalar_value_index(scalar, *value, universe_len, universe);
            let Some(bit_idx) = bit_idx else {
                if require_all_in_universe {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: exact scalar Set contains values outside compact SetBitmask universe"
                    )));
                }
                continue;
            };
            mask |= 1_i64 << bit_idx;
        }
        Ok(self.emit_i64_const(block_idx, mask))
    }

    fn emit_set_intersect_operand_bitmask_i64(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<(usize, ValueId), TrustIrError> {
        match self.aggregate_shapes.get(&reg).cloned() {
            Some(super::AggregateShape::ExactIntSet { values }) => {
                let mask = self.emit_exact_int_set_operand_bitmask_i64(
                    block_idx,
                    &values,
                    universe_len,
                    universe,
                    context,
                    false,
                )?;
                Ok((block_idx, mask))
            }
            Some(super::AggregateShape::ExactScalarSet { scalar, values }) => {
                let mask = self.emit_exact_scalar_set_operand_bitmask_i64(
                    block_idx,
                    &scalar,
                    &values,
                    universe_len,
                    universe,
                    context,
                    false,
                )?;
                Ok((block_idx, mask))
            }
            Some(super::AggregateShape::Interval { lo, hi }) => {
                let mask = self.emit_interval_bitmask_i64_allow_clamped(
                    block_idx,
                    lo,
                    hi,
                    universe_len,
                    universe,
                    context,
                )?;
                Ok((block_idx, mask))
            }
            _ => self.emit_set_operand_bitmask_i64_allow_materialized(
                block_idx,
                reg,
                universe_len,
                universe,
                context,
            ),
        }
    }

    pub(super) fn emit_set_subseteq_operand_bitmask_i64(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<(usize, ValueId, ValueId), TrustIrError> {
        match self.aggregate_shapes.get(&reg).cloned() {
            Some(shape @ super::AggregateShape::SetBitmask { .. }) => {
                if !shape.compatible_set_bitmask_universe(universe_len, universe) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SetBitmask universe mismatch for r{reg}: {shape:?}"
                    )));
                }
                let all_in_universe = self.emit_i64_const(block_idx, 1);
                Ok((block_idx, self.load_reg(block_idx, reg)?, all_in_universe))
            }
            Some(super::AggregateShape::ExactIntSet { values }) => {
                let mask = self.emit_exact_int_set_operand_bitmask_i64(
                    block_idx,
                    &values,
                    universe_len,
                    universe,
                    context,
                    false,
                )?;
                let all_in_universe = values.iter().all(|value| {
                    super::int_value_in_set_bitmask_universe(*value, universe_len, universe)
                });
                let all_in_universe = self.emit_i64_const(block_idx, i64::from(all_in_universe));
                Ok((block_idx, mask, all_in_universe))
            }
            Some(super::AggregateShape::ExactScalarSet { scalar, values }) => {
                let mask = self.emit_exact_scalar_set_operand_bitmask_i64(
                    block_idx,
                    &scalar,
                    &values,
                    universe_len,
                    universe,
                    context,
                    false,
                )?;
                let all_in_universe = values.iter().all(|value| {
                    super::exact_scalar_value_in_set_bitmask_universe(
                        &scalar,
                        *value,
                        universe_len,
                        universe,
                    )
                });
                let all_in_universe = self.emit_i64_const(block_idx, i64::from(all_in_universe));
                Ok((block_idx, mask, all_in_universe))
            }
            Some(super::AggregateShape::Set { len: 0, .. }) => {
                let zero = self.emit_i64_const(block_idx, 0);
                let one = self.emit_i64_const(block_idx, 1);
                Ok((block_idx, zero, one))
            }
            Some(super::AggregateShape::Interval { lo, hi }) => {
                let mask = self.emit_interval_bitmask_i64_allow_clamped(
                    block_idx,
                    lo,
                    hi,
                    universe_len,
                    universe,
                    context,
                )?;
                let all_in_universe = hi < lo
                    || super::interval_convertible_to_set_bitmask(lo, hi, universe_len, universe);
                let all_in_universe = self.emit_i64_const(block_idx, i64::from(all_in_universe));
                Ok((block_idx, mask, all_in_universe))
            }
            Some(super::AggregateShape::Set { .. }) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact subset operation cannot infer that materialized Set r{reg} is confined to the compact SetBitmask universe"
            ))),
            Some(shape) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact subset operation requires SetBitmask or compatible finite integer Set operand, got r{reg} = {shape:?}"
            ))),
            None => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact subset operation requires tracked set operand r{reg}"
            ))),
        }
    }

    fn materialized_domain_shape_for_pointer(
        shape: super::AggregateShape,
    ) -> super::AggregateShape {
        match shape {
            super::AggregateShape::ExactIntSet { values } => super::AggregateShape::Set {
                len: u32::try_from(values.len()).unwrap_or(u32::MAX),
                element: Some(Box::new(super::AggregateShape::Scalar(
                    super::ScalarShape::Int,
                ))),
            },
            super::AggregateShape::ExactScalarSet { scalar, values } => {
                super::AggregateShape::Set {
                    len: u32::try_from(values.len()).unwrap_or(u32::MAX),
                    element: Some(Box::new(super::AggregateShape::Scalar(scalar))),
                }
            }
            other => other,
        }
    }

    fn emit_interval_bitmask_i64(
        &mut self,
        block_idx: usize,
        lo: i64,
        hi: i64,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        if hi < lo {
            return Ok(self.emit_i64_const(block_idx, 0));
        }
        let count = hi
            .checked_sub(lo)
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "{context}: materialized interval rangelength overflows i64"
                ))
            })?;
        if count > i64::from(universe_len) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: materialized interval {lo}..{hi} is wider than compact SetBitmask universe length {universe_len}"
            )));
        }

        let mut mask = 0_i64;
        match universe {
            super::SetBitmaskUniverse::IntRange { lo: universe_lo } => {
                let universe_hi = universe_lo
                    .checked_add(i64::from(universe_len).saturating_sub(1))
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "{context}: compact SetBitmask integer universe {universe_lo} plus len {universe_len} overflows i64"
                        ))
                    })?;
                if lo < *universe_lo || hi > universe_hi {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: materialized interval {lo}..{hi} is outside compact SetBitmask universe {universe_lo}..{universe_hi}"
                    )));
                }
                for elem in lo..=hi {
                    let bit_idx = elem - *universe_lo;
                    mask |= 1_i64 << bit_idx;
                }
            }
            super::SetBitmaskUniverse::ExplicitInt(values) => {
                for elem in lo..=hi {
                    let Some(bit_idx) = values.iter().position(|value| *value == elem) else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "{context}: materialized interval element {elem} is outside compact SetBitmask explicit universe"
                        )));
                    };
                    mask |= 1_i64 << bit_idx;
                }
            }
            super::SetBitmaskUniverse::Exact(_) => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask operation requires exact universe metadata with integer elements"
                )));
            }
            super::SetBitmaskUniverse::Unknown => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask operation requires exact universe metadata"
                )));
            }
        }
        Ok(self.emit_i64_const(block_idx, mask))
    }

    fn emit_interval_bitmask_i64_allow_clamped(
        &mut self,
        block_idx: usize,
        lo: i64,
        hi: i64,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        Self::compact_set_bitmask_valid_mask(universe_len, context)?;
        if hi < lo {
            return Ok(self.emit_i64_const(block_idx, 0));
        }

        let mut mask = 0_i64;
        match universe {
            super::SetBitmaskUniverse::IntRange { lo: universe_lo } => {
                let universe_hi = universe_lo
                    .checked_add(i64::from(universe_len).saturating_sub(1))
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "{context}: compact SetBitmask integer universe {universe_lo} plus len {universe_len} overflows i64"
                        ))
                    })?;
                let start = lo.max(*universe_lo);
                let end = hi.min(universe_hi);
                if start <= end {
                    for elem in start..=end {
                        let bit_idx = elem - *universe_lo;
                        mask |= 1_i64 << bit_idx;
                    }
                }
            }
            super::SetBitmaskUniverse::ExplicitInt(values) => {
                for (bit_idx, elem) in values.iter().enumerate() {
                    if bit_idx >= usize::try_from(universe_len).unwrap_or(usize::MAX) {
                        break;
                    }
                    if *elem >= lo && *elem <= hi {
                        mask |= 1_i64 << bit_idx;
                    }
                }
            }
            super::SetBitmaskUniverse::Exact(_) => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask operation requires exact universe metadata with integer elements"
                )));
            }
            super::SetBitmaskUniverse::Unknown => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask operation requires exact universe metadata"
                )));
            }
        }
        Ok(self.emit_i64_const(block_idx, mask))
    }

    pub(super) fn compact_binary_set_universe(
        &self,
        op: &str,
        r1: u8,
        r2: u8,
    ) -> Result<Option<(u32, super::SetBitmaskUniverse)>, TrustIrError> {
        let left = match self.aggregate_shapes.get(&r1) {
            Some(shape @ super::AggregateShape::SetBitmask { .. }) => {
                Some(shape.set_bitmask_universe().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{op}: compact SetBitmask operand r{r1} requires exact universe metadata"
                    ))
                })?)
            }
            _ => None,
        };
        let right = match self.aggregate_shapes.get(&r2) {
            Some(shape @ super::AggregateShape::SetBitmask { .. }) => {
                Some(shape.set_bitmask_universe().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{op}: compact SetBitmask operand r{r2} requires exact universe metadata"
                    ))
                })?)
            }
            _ => None,
        };
        match (left, right) {
            (Some(left), Some(right)) if left != right => {
                Err(TrustIrError::UnsupportedOpcode(format!(
                    "{op}: compact SetBitmask universe mismatch: r{r1} has {left:?}, r{r2} has {right:?}"
                )))
            }
            (Some(universe), _) | (_, Some(universe)) => Ok(Some(universe)),
            _ => Ok(None),
        }
    }

    /// Shared record-set-bitmask universe of two operands for a binary set op
    /// (`SetUnion` / `SetDiff`) — the RecordSetBitmask sibling of
    /// [`Self::compact_binary_set_universe`] (RecordSetBitmask step 3/5).
    ///
    /// Returns `Some((universe_len, slot_count, universe))` when AT LEAST one
    /// operand is a `RecordSetBitmask` and (if both are) they carry the
    /// identical universe; `None` when neither operand is a RecordSetBitmask.
    /// Mismatched record universes fail closed: there is no sound bit-aligned
    /// op across distinct universes. The carried `(universe_len, slot_count)`
    /// is validated against `record_set_bitmask_slot_count_ir` so a malformed
    /// shape can never reach the per-slot emitters.
    pub(super) fn record_set_bitmask_binary_universe(
        &self,
        op: &str,
        r1: u8,
        r2: u8,
    ) -> Result<Option<(u32, u32, Vec<super::RecordBitKey>)>, TrustIrError> {
        let extract = |reg: u8| match self.aggregate_shapes.get(&reg) {
            Some(super::AggregateShape::RecordSetBitmask {
                universe_len,
                slot_count,
                universe,
            }) => Some((*universe_len, *slot_count, universe.clone())),
            _ => None,
        };
        let left = extract(r1);
        let right = extract(r2);
        let shape = match (left, right) {
            (Some(left), Some(right)) => {
                if left != right {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{op}: RecordSetBitmask universe mismatch between r{r1} and r{r2}"
                    )));
                }
                left
            }
            (Some(shape), None) | (None, Some(shape)) => shape,
            (None, None) => return Ok(None),
        };
        let (universe_len, slot_count, universe) = shape;
        if super::record_set_bitmask_slot_count_ir(universe_len) != slot_count as usize
            || universe.len() != universe_len as usize
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{op}: RecordSetBitmask shape inconsistent: universe_len={universe_len}, \
                 slot_count={slot_count}, universe.len()={}",
                universe.len()
            )));
        }
        Ok(Some((universe_len, slot_count, universe)))
    }

    /// Load the `slot_count` mask slots of a RecordSetBitmask operand from its
    /// pointer-backed compact region (RecordSetBitmask step 3/5).
    ///
    /// The operand register holds a `PtrToInt` of the multi-slot mask region;
    /// `compact_state_slot_for_use` recovers (and, across blocks, reloads) the
    /// base pointer. Slot `j` is `load_at_offset(base, offset + j)`. There is no
    /// `IntToPtr` of a slot value — the slots are loaded as i64 bit-vectors,
    /// never dereferenced — so the rc=139 trap cannot occur.
    fn load_record_set_bitmask_slots(
        &mut self,
        block_idx: usize,
        reg: u8,
        slot_count: u32,
        op: &str,
    ) -> Result<Vec<ValueId>, TrustIrError> {
        let Some(slot) = self.compact_state_slot_for_use(block_idx, reg)? else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{op}: RecordSetBitmask operand r{reg} requires a compact (pointer-backed) mask"
            )));
        };
        let mut slots = Vec::with_capacity(slot_count as usize);
        for slot_index in 0..slot_count {
            slots.push(self.load_at_offset(block_idx, slot.source_ptr, slot.offset + slot_index));
        }
        Ok(slots)
    }

    /// Resolve the `slot_count` mask slots of one operand of a RecordSetBitmask
    /// binary op (`SetUnion` / `SetDiff`), dispatching on operand kind (Track B
    /// increment 1b — the byte-exact record-set native ACTION compile):
    ///
    ///   * If the operand is a tracked `RecordSetBitmask` (the state var, a
    ///     prior bitmask result), load its `slot_count` slots from the
    ///     pointer-backed compact region via [`Self::load_record_set_bitmask_slots`].
    ///   * If the operand is a record-set `{e_1, …, e_N}` LITERAL whose element
    ///     registers are tracked compact records (recorded in
    ///     `record_set_literal_element_regs` by [`Self::lower_set_enum`]), build
    ///     its mask in-place via [`Self::emit_record_set_bitmask_enum_fold`] over
    ///     the SHARED `universe` — byte-identical to the interpreter's
    ///     `record_set_bitmask_value_to_slots`. NO `IntToPtr` of the mask is
    ///     emitted; element fields are loaded from their pointer-backed compact
    ///     slots, the literal is never read as a materialized set pointer.
    ///
    /// Fails closed (`UnsupportedOpcode`) if neither kind applies — e.g. an
    /// untracked materialized literal, or a literal whose element registers lost
    /// their compact record slot — so the whole action routes to the interpreter
    /// rather than mis-encode the operand.
    pub(super) fn record_set_bitmask_operand_slots(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        slot_count: u32,
        universe: &[super::RecordBitKey],
        op: &str,
    ) -> Result<Vec<ValueId>, TrustIrError> {
        self.record_set_bitmask_operand_slots_with_found(
            block_idx,
            reg,
            universe_len,
            slot_count,
            universe,
            op,
        )
        .map(|(slots, _)| slots)
    }

    /// [`Self::record_set_bitmask_operand_slots`] variant that also surfaces
    /// the enum-fold's strictness flag ("every literal element matched some
    /// universe key"; `None` for real bitmask operands, which are canonical by
    /// construction). Successor-producing consumers (`SetUnion`/`SetDiff`)
    /// must branch to a `FallbackNeeded` return when the flag is 0.
    pub(super) fn record_set_bitmask_operand_slots_with_found(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        slot_count: u32,
        universe: &[super::RecordBitKey],
        op: &str,
    ) -> Result<(Vec<ValueId>, Option<ValueId>), TrustIrError> {
        // A real RecordSetBitmask operand (the state var / a prior bitmask
        // result) loads its slots from the pointer-backed compact region.
        if matches!(
            self.aggregate_shapes.get(&reg),
            Some(super::AggregateShape::RecordSetBitmask { .. })
        ) {
            return self
                .load_record_set_bitmask_slots(block_idx, reg, slot_count, op)
                .map(|slots| (slots, None));
        }

        // A record-set `{e_1, …, e_N}` literal: recover its element registers
        // (each a tracked compact record) and dispatch the enum-fold over the
        // shared universe. The element registers and their shapes are cloned up
        // front so the resolution holds no borrow on `self`.
        let Some(elem_regs) = self.record_set_literal_element_regs.get(&reg).cloned() else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{op}: RecordSetBitmask operand r{reg} requires a compact (pointer-backed) mask \
                 or a tracked record-set literal"
            )));
        };
        let mut elements: Vec<(super::CompactStateSlot, super::AggregateShape)> =
            Vec::with_capacity(elem_regs.len());
        for elem_reg in elem_regs {
            let Some(elem_shape @ super::AggregateShape::Record { .. }) =
                self.aggregate_shapes.get(&elem_reg).cloned()
            else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{op}: RecordSetBitmask literal element r{elem_reg} lost its tracked record \
                     shape; failing closed"
                )));
            };
            let Some(elem_slot) = self.compact_state_slot_for_use(block_idx, elem_reg)? else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{op}: RecordSetBitmask literal element r{elem_reg} requires a compact \
                     (pointer-backed) record; failing closed"
                )));
            };
            elements.push((elem_slot, elem_shape));
        }
        self.emit_record_set_bitmask_enum_fold_with_found(
            block_idx,
            universe_len,
            slot_count,
            universe,
            &elements,
            op,
        )
    }

    /// Emit a RecordSetBitmask `{e_1, …, e_N}` enum-fold over a known record
    /// universe (RecordSetBitmask step 3/5).
    ///
    /// For each universe record `i`, OR `(element matches universe record i)`
    /// into bit `i % 64` of result slot `i / 64`. An element is "matched" iff
    /// the AND of its per-field equalities against universe key `i`'s field
    /// constants holds. The bit index `i` is the EXACT index the interpreter's
    /// `record_set_bitmask_value_to_slots` assigns, so the produced mask is
    /// byte-identical to the interpreter encoding for the same logical set.
    ///
    /// `elements` lists each literal element as `(compact_slot, record_shape)`.
    /// An element that matches no universe record contributes nothing (and the
    /// caller must independently fail closed if an out-of-universe element is
    /// possible — a closed-universe enum-fold proves all elements are inside).
    ///
    /// Returns the `slot_count` result slot values (each already masked to its
    /// per-slot valid bits). NO `IntToPtr` is emitted: element fields are loaded
    /// from their pointer-backed compact slots, never dereferenced as pointers.
    ///
    /// Dispatched (Track B increment 1b) by
    /// [`Self::record_set_bitmask_operand_slots`] for the `{rec}` literal operand
    /// of a `v \cup {rec}` / `v \ {rec}` record-set ACTION, over the record-set
    /// state var's universe. The bit-set logic is byte-identical to the
    /// interpreter's `record_set_bitmask_value_to_slots` for the same logical
    /// set, so the compiled action's successor mask matches the interpreter
    /// exactly. Also exercised directly by the golden tests.
    /// True when `shape` is a FULLY tracked record shape (every field carries
    /// a tracked shape) that provably does NOT contain `field`. Records with
    /// different field sets are never `Value`-equal, so a universe key
    /// requiring such a field can never match this element — a STATIC
    /// no-match, not an error (the heterogeneous-universe case, e.g.
    /// TwoPhase's `Message == [type,rm] \cup [type]`).
    ///
    /// Any partially tracked shape returns `false` so callers stay
    /// fail-closed: `compact_record_field`'s `None` is ambiguous between
    /// "field absent" and "an earlier field has no tracked shape" (it
    /// early-returns on the first untracked field), and only a complete field
    /// list proves absence. Soundness rides on the same shape-completeness
    /// invariant the compact offset computation already relies on.
    fn record_shape_field_certainly_absent(
        shape: &super::AggregateShape,
        field: tla_core::NameId,
    ) -> bool {
        let super::AggregateShape::Record { fields } = shape else {
            return false;
        };
        fields.iter().all(|(_, s)| s.is_some()) && fields.iter().all(|(name, _)| *name != field)
    }

    #[cfg_attr(not(test), allow(dead_code))] // exercised directly by the golden tests
    pub(super) fn emit_record_set_bitmask_enum_fold(
        &mut self,
        block_idx: usize,
        universe_len: u32,
        slot_count: u32,
        universe: &[super::RecordBitKey],
        elements: &[(super::CompactStateSlot, super::AggregateShape)],
        op: &str,
    ) -> Result<Vec<ValueId>, TrustIrError> {
        self.emit_record_set_bitmask_enum_fold_with_found(
            block_idx,
            universe_len,
            slot_count,
            universe,
            elements,
            op,
        )
        .map(|(slots, _)| slots)
    }

    /// [`Self::emit_record_set_bitmask_enum_fold`] variant that ALSO returns a
    /// runtime `all_found` flag (0/1 i64): AND over elements of "this element
    /// matched at least one universe key".
    ///
    /// `None` iff `elements` is empty (nothing to check). Callers producing a
    /// SUCCESSOR mask from elements with runtime scalar fields MUST branch on
    /// the flag and take a `FallbackNeeded` return when it is 0 — an unmatched
    /// element means the true logical set contains a record OUTSIDE the
    /// universe, which the bitmask cannot represent; silently dropping the bit
    /// would fabricate a wrong successor. (Byte-exactness note: for reachable
    /// closed-universe states the flag is constant 1 and the branch is never
    /// taken, so existing behavior is unchanged.)
    pub(super) fn emit_record_set_bitmask_enum_fold_with_found(
        &mut self,
        block_idx: usize,
        universe_len: u32,
        slot_count: u32,
        universe: &[super::RecordBitKey],
        elements: &[(super::CompactStateSlot, super::AggregateShape)],
        op: &str,
    ) -> Result<(Vec<ValueId>, Option<ValueId>), TrustIrError> {
        if super::record_set_bitmask_slot_count_ir(universe_len) != slot_count as usize
            || universe.len() != universe_len as usize
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{op}: RecordSetBitmask shape inconsistent: universe_len={universe_len}, \
                 slot_count={slot_count}, universe.len()={}",
                universe.len()
            )));
        }
        let mut result_slots: Vec<ValueId> = (0..slot_count)
            .map(|_| self.emit_i64_const(block_idx, 0))
            .collect();
        let one = self.emit_i64_const(block_idx, 1);
        // Per-element "matched at least one key" accumulator (strictness flag).
        let mut elem_found: Vec<Option<ValueId>> = vec![None; elements.len()];
        for (index, key) in universe.iter().enumerate() {
            // present_for_key = OR over elements of (element == universe key i).
            let mut present_for_key: Option<ValueId> = None;
            for (elem_index, (elem_slot, elem_shape)) in elements.iter().enumerate() {
                // Records with different field COUNTS are never Value-equal: a
                // fully tracked element whose field count differs from the
                // key's is a STATIC no-match (this also closes the
                // superset-field false-match hazard: an element carrying every
                // key field plus extras must not match the shorter key).
                if let super::AggregateShape::Record { fields } = elem_shape {
                    if fields.iter().all(|(_, s)| s.is_some()) && fields.len() != key.fields.len() {
                        continue;
                    }
                }
                let mut record_matches: Option<ValueId> = None;
                for (field_name, element) in &key.fields {
                    let Some((compact_offset, Some(field_shape))) =
                        elem_shape.compact_record_field(*field_name)
                    else {
                        // Heterogeneous record universe (e.g. TwoPhase's
                        // `[type,rm] ∪ [type]`): when the element's FULLY
                        // tracked record shape provably lacks this key field,
                        // the element can never Value-equal the key — a STATIC
                        // no-match for this (element, key) pair, not an error.
                        // Anything short of proven absence stays fail-closed.
                        if Self::record_shape_field_certainly_absent(elem_shape, *field_name) {
                            record_matches = Some(self.emit_i64_const(block_idx, 0));
                            break;
                        }
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "{op}: RecordSetBitmask enum element is missing universe field \
                             {field_name:?} (or it has no tracked shape)"
                        )));
                    };
                    if !Self::is_single_slot_flat_aggregate_value(&field_shape) {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "{op}: RecordSetBitmask enum requires single-slot scalar fields, field \
                             {field_name:?} has shape {field_shape:?}"
                        )));
                    }
                    let elem_field = self.load_at_offset(
                        block_idx,
                        elem_slot.source_ptr,
                        elem_slot.offset + compact_offset,
                    );
                    let expected = self.emit_i64_const(
                        block_idx,
                        super::set_bitmask_element_compact_value(element),
                    );
                    let eq = self.emit_with_result(
                        block_idx,
                        Inst::ICmp {
                            op: ICmpOp::Eq,
                            ty: Ty::I64,
                            lhs: elem_field,
                            rhs: expected,
                        },
                    );
                    let eq_i64 = self.emit_bool_to_i64(block_idx, eq);
                    record_matches = Some(match record_matches {
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
                let record_matches = record_matches.unwrap_or(one);
                elem_found[elem_index] = Some(match elem_found[elem_index] {
                    None => record_matches,
                    Some(prev) => self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::Or,
                            ty: Ty::I64,
                            lhs: prev,
                            rhs: record_matches,
                        },
                    ),
                });
                present_for_key = Some(match present_for_key {
                    None => record_matches,
                    Some(prev) => self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::Or,
                            ty: Ty::I64,
                            lhs: prev,
                            rhs: record_matches,
                        },
                    ),
                });
            }
            // OR the (0/1) presence bit into bit (index % 64) of slot index/64.
            let present_for_key =
                present_for_key.unwrap_or_else(|| self.emit_i64_const(block_idx, 0));
            let shift = self.emit_i64_const(block_idx, (index % 64) as i64);
            let shifted = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Shl,
                    ty: Ty::I64,
                    lhs: present_for_key,
                    rhs: shift,
                },
            );
            let slot = index / 64;
            result_slots[slot] = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: result_slots[slot],
                    rhs: shifted,
                },
            );
        }
        // AND each slot with its per-slot valid mask (defensive; presence bits
        // are already in-range, but this keeps the encoding canonical).
        for (slot_index, slot_value) in result_slots.iter_mut().enumerate() {
            let valid_mask = super::record_set_bitmask_slot_valid_mask_ir(universe_len, slot_index)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{op}: RecordSetBitmask slot {slot_index} out of range for universe_len \
                         {universe_len}"
                    ))
                })?;
            let valid_mask_val = self.emit_i64_const(block_idx, valid_mask as i64);
            *slot_value = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: *slot_value,
                    rhs: valid_mask_val,
                },
            );
        }
        // all_found = AND over elements of "matched some key". An element that
        // statically matched NO key folds to constant 0 (always fallback).
        let mut all_found: Option<ValueId> = None;
        for found in elem_found {
            let found = match found {
                Some(v) => v,
                None => self.emit_i64_const(block_idx, 0),
            };
            all_found = Some(match all_found {
                None => found,
                Some(prev) => self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: prev,
                        rhs: found,
                    },
                ),
            });
        }
        Ok((result_slots, all_found))
    }

    /// Fast path for `SetIn { elem, set }` where `set` carries a
    /// `RecordSetBitmask` shape and `elem` is a fully COMPILE-TIME-CONSTANT
    /// record (a `RecordNew` whose field values all chase to
    /// `LoadConst`/`LoadImm`/`LoadBool` in a straight-line prefix): resolve the
    /// universe bit index at compile time and emit a single mask bit test
    /// instead of the full per-key universe compare scan.
    ///
    /// SOUNDNESS:
    /// * The bytecode chase only runs when `instructions[..pc]` contains no
    ///   control flow, so the nearest writer IS the dominating definition.
    /// * Field comparison is exact `SetBitmaskElement` equality (variant AND
    ///   payload), plus a field-COUNT check — record `Value` equality demands
    ///   both, so a match is exactly `Value`-equality with the universe key.
    /// * A constant record equal to NO universe key is statically absent from
    ///   every value the mask can encode (masks only encode universe bits):
    ///   the result folds to constant FALSE.
    /// * Returns `Ok(None)` (caller falls through to the general lowering) on
    ///   any shape/chase mismatch.
    pub(super) fn try_lower_const_record_membership_bit(
        &mut self,
        pc: usize,
        instructions: &[tla_tir::bytecode::Opcode],
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        set_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        use tla_tir::bytecode::Opcode as Op;
        use tla_value::Value;
        let Some(super::AggregateShape::RecordSetBitmask {
            universe_len,
            slot_count,
            universe,
        }) = self.aggregate_shapes.get(&set_reg).cloned()
        else {
            return Ok(None);
        };
        if super::record_set_bitmask_slot_count_ir(universe_len) != slot_count as usize
            || universe.len() != universe_len as usize
        {
            return Ok(None);
        }
        // Straight-line prefix: no control flow before this SetIn, so the
        // nearest writer of a register is its dominating definition.
        if pc > instructions.len()
            || instructions[..pc].iter().any(|op| {
                matches!(
                    op,
                    Op::Jump { .. }
                        | Op::JumpTrue { .. }
                        | Op::JumpFalse { .. }
                        | Op::ForallBegin { .. }
                        | Op::ForallNext { .. }
                        | Op::ExistsBegin { .. }
                        | Op::ExistsNext { .. }
                        | Op::ChooseBegin { .. }
                        | Op::ChooseNext { .. }
                        | Op::SetFilterBegin { .. }
                        | Op::SetBuilderBegin { .. }
                        | Op::LoopNext { .. }
                        | Op::Ret { .. }
                )
            })
        {
            return Ok(None);
        }
        let nearest_writer = |reg: u8, before: usize| -> Option<usize> {
            (0..before.min(instructions.len()))
                .rev()
                .find(|p| instructions[*p].dest_register() == Some(reg))
        };
        let Some(rn_pc) = nearest_writer(elem_reg, pc) else {
            return Ok(None);
        };
        let Op::RecordNew {
            fields_start,
            values_start,
            count,
            ..
        } = instructions[rn_pc].clone()
        else {
            return Ok(None);
        };
        if count == 0 {
            return Ok(None);
        }
        // Resolve field names and constant field values.
        let mut fields: Vec<(tla_core::NameId, tla_jit_abi::SetBitmaskElement)> =
            Vec::with_capacity(usize::from(count));
        {
            let Ok(pool) = self.require_const_pool() else {
                return Ok(None);
            };
            for i in 0..count {
                let name_idx = fields_start.checked_add(u16::from(i));
                let Some(name_idx) = name_idx else {
                    return Ok(None);
                };
                if usize::from(name_idx) >= pool.value_count() {
                    return Ok(None);
                }
                let Value::String(name) = pool.get_value(name_idx) else {
                    return Ok(None);
                };
                let name_id = tla_core::intern_name(name.as_ref());
                let Some(value_reg) = values_start.checked_add(i) else {
                    return Ok(None);
                };
                let Some(writer_pc) = nearest_writer(value_reg, rn_pc) else {
                    return Ok(None);
                };
                let value = match instructions[writer_pc].clone() {
                    Op::LoadConst { idx, .. } => {
                        if usize::from(idx) >= pool.value_count() {
                            return Ok(None);
                        }
                        pool.get_value(idx).clone()
                    }
                    Op::LoadImm { value, .. } => Value::SmallInt(value),
                    Op::LoadBool { value, .. } => Value::Bool(value),
                    _ => return Ok(None),
                };
                let Some(element) = Self::scalar_key_from_dynamic_scalar_value(&value) else {
                    return Ok(None);
                };
                if fields.iter().any(|(existing, _)| *existing == name_id) {
                    return Ok(None); // duplicate field name: malformed
                }
                fields.push((name_id, element));
            }
        }
        // Locate the (unique) universe key Value-equal to this record: same
        // field count and exact per-field SetBitmaskElement equality.
        let mut bit: Option<usize> = None;
        for (index, key) in universe.iter().enumerate() {
            if key.fields.len() != fields.len() {
                continue;
            }
            let matches = key.fields.iter().all(|(key_name, key_element)| {
                fields
                    .iter()
                    .any(|(name, element)| name == key_name && element == key_element)
            });
            if matches {
                bit = Some(index);
                break;
            }
        }
        let Some(index) = bit else {
            // Statically outside the universe: constant FALSE (the mask can
            // only ever encode universe records).
            let zero = self.emit_i64_const(block_idx, 0);
            self.store_reg_value(block_idx, rd, zero)?;
            return Ok(Some(block_idx));
        };
        // Single bit test: (slots[index/64] >> (index%64)) & 1.
        let Some(set_slot) = self.compact_state_slot_for_use(block_idx, set_reg)? else {
            return Ok(None);
        };
        let word = self.load_at_offset(
            block_idx,
            set_slot.source_ptr,
            set_slot.offset + (index / 64) as u32,
        );
        let shift = self.emit_i64_const(block_idx, (index % 64) as i64);
        let shifted = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::LShr,
                ty: Ty::I64,
                lhs: word,
                rhs: shift,
            },
        );
        let one = self.emit_i64_const(block_idx, 1);
        let bit_val = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: shifted,
                rhs: one,
            },
        );
        self.store_reg_value(block_idx, rd, bit_val)?;
        Ok(Some(block_idx))
    }

    /// Branch on the enum-fold strictness flags of a record-set binary op's
    /// operands: when any literal element matched NO universe key at runtime,
    /// the true logical result contains an out-of-universe record the bitmask
    /// cannot represent — return `FallbackNeeded` so the interpreter owns this
    /// state (never silently drop the element). Returns the continuation block
    /// for the all-found path (unchanged block when both flags are absent).
    fn emit_record_set_strictness_guard(
        &mut self,
        block_idx: usize,
        left_found: Option<ValueId>,
        right_found: Option<ValueId>,
    ) -> Result<usize, TrustIrError> {
        let combined = match (left_found, right_found) {
            (None, None) => return Ok(block_idx),
            (Some(v), None) | (None, Some(v)) => v,
            (Some(a), Some(b)) => self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: a,
                    rhs: b,
                },
            ),
        };
        let zero = self.emit_i64_const(block_idx, 0);
        let ok = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: combined,
                rhs: zero,
            },
        );
        let cont_block = self.new_aux_block("rsb_strict_ok");
        let fb_block = self.new_aux_block("rsb_strict_fallback");
        self.emit_fallback_needed_and_return(fb_block);
        let cont_id = self.block_id_of(cont_block);
        let fb_id = self.block_id_of(fb_block);
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: ok,
                then_target: cont_id,
                then_args: vec![],
                else_target: fb_id,
                else_args: vec![],
            }),
        );
        Ok(cont_block)
    }

    /// Store a freshly-computed RecordSetBitmask result (the `slot_count`
    /// per-slot mask values) into `rd` as a pointer-backed compact region, and
    /// tag `rd` with the RecordSetBitmask shape (RecordSetBitmask step 3/5).
    fn store_record_set_bitmask_result(
        &mut self,
        block_idx: usize,
        rd: u8,
        result_slots: &[ValueId],
        universe_len: u32,
        slot_count: u32,
        universe: Vec<super::RecordBitKey>,
    ) -> Result<Option<usize>, TrustIrError> {
        let result_ptr = self.alloc_aggregate(block_idx, slot_count);
        for (slot_index, value) in result_slots.iter().enumerate() {
            self.store_at_offset(block_idx, result_ptr, slot_index as u32, *value);
        }
        self.store_compact_aggregate_result(
            block_idx,
            rd,
            result_ptr,
            super::AggregateShape::RecordSetBitmask {
                universe_len,
                slot_count,
                universe,
            },
        )?;
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);
        Ok(Some(block_idx))
    }

    fn compact_binary_setdiff_universe(
        &self,
        r1: u8,
        r2: u8,
    ) -> Result<Option<(u32, super::SetBitmaskUniverse)>, TrustIrError> {
        let operand_universe = |reg: u8| -> Result<
            Option<(u32, super::SetBitmaskUniverse)>,
            TrustIrError,
        > {
            match self.aggregate_shapes.get(&reg) {
                Some(shape @ super::AggregateShape::SetBitmask { .. }) => {
                    Ok(Some(shape.set_bitmask_universe().ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "SetDiff: compact SetBitmask operand r{reg} requires exact universe metadata"
                        ))
                    })?))
                }
                Some(shape @ super::AggregateShape::TaggedScalarOrSet { .. }) => {
                    Ok(Some(shape.tagged_set_branch_universe().ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "SetDiff: tagged scalar-or-set operand r{reg} requires exact universe metadata"
                        ))
                    })?))
                }
                _ => Ok(None),
            }
        };
        let left = operand_universe(r1)?;
        let right = operand_universe(r2)?;
        match (left, right) {
            (Some(left), Some(right)) if left != right => {
                Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetDiff: compact set universe mismatch: r{r1} has {left:?}, r{r2} has {right:?}"
                )))
            }
            (Some(universe), _) | (_, Some(universe)) => Ok(Some(universe)),
            _ => Ok(None),
        }
    }

    fn small_interval_setdiff_universe(
        &self,
        r1: u8,
        r2: u8,
    ) -> Result<Option<(u32, super::SetBitmaskUniverse)>, TrustIrError> {
        let Some(super::AggregateShape::Interval { lo, hi }) = self.aggregate_shapes.get(&r1)
        else {
            return Ok(None);
        };
        let universe_len = if hi < lo {
            0
        } else {
            let len = hi
                .checked_sub(*lo)
                .and_then(|span| span.checked_add(1))
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "SetDiff: interval source {lo}..{hi} length overflows i64"
                    ))
                })?;
            let Ok(len) = u32::try_from(len) else {
                return Ok(None);
            };
            if len > 63 {
                return Ok(None);
            }
            len
        };

        let Some(subtract_shape) = self.aggregate_shapes.get(&r2) else {
            return Ok(None);
        };
        if !Self::can_lower_small_setdiff_rhs_as_int_mask(subtract_shape) {
            return Ok(None);
        }

        Ok(Some((
            universe_len,
            super::SetBitmaskUniverse::IntRange { lo: *lo },
        )))
    }

    fn can_lower_small_setdiff_rhs_as_int_mask(shape: &super::AggregateShape) -> bool {
        match shape {
            super::AggregateShape::ExactIntSet { .. } | super::AggregateShape::Interval { .. } => {
                true
            }
            super::AggregateShape::Set { len: 0, .. } => true,
            super::AggregateShape::Set { .. } => false,
            _ => false,
        }
    }

    fn emit_small_setdiff_rhs_int_mask_i64(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
    ) -> Result<ValueId, TrustIrError> {
        if universe_len == 0 {
            return Ok(self.emit_i64_const(block_idx, 0));
        }

        match self.aggregate_shapes.get(&reg).cloned() {
            Some(super::AggregateShape::ExactIntSet { values }) => self
                .emit_exact_int_set_operand_bitmask_i64(
                    block_idx,
                    &values,
                    universe_len,
                    universe,
                    "SetDiff",
                    false,
                ),
            Some(super::AggregateShape::Interval { lo, hi }) => self
                .emit_interval_bitmask_i64_allow_clamped(
                    block_idx,
                    lo,
                    hi,
                    universe_len,
                    universe,
                    "SetDiff",
                ),
            Some(super::AggregateShape::Set { len: 0, .. }) => {
                Ok(self.emit_i64_const(block_idx, 0))
            }
            Some(shape) => Err(TrustIrError::UnsupportedOpcode(format!(
                "SetDiff: cannot lower r{reg} as a small integer mask: {shape:?}"
            ))),
            None => Err(TrustIrError::UnsupportedOpcode(format!(
                "SetDiff: cannot lower untracked r{reg} as a small integer mask"
            ))),
        }
    }

    fn emit_symbolic_domain_membership_i64(
        &mut self,
        block_idx: usize,
        elem_val: ValueId,
        elem_shape: Option<&super::AggregateShape>,
        domain: super::SymbolicDomain,
    ) -> Result<ValueId, TrustIrError> {
        let Some(elem_shape) = elem_shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: {domain:?} membership requires a known numeric operand shape"
            )));
        };
        if !elem_shape.is_numeric_scalar_shape() {
            return Ok(self.emit_i64_const(block_idx, 0));
        }

        match domain {
            super::SymbolicDomain::Nat => {
                let zero = self.emit_i64_const(block_idx, 0);
                let ge_zero = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Sge,
                        ty: Ty::I64,
                        lhs: elem_val,
                        rhs: zero,
                    },
                );
                Ok(self.emit_bool_to_i64(block_idx, ge_zero))
            }
            super::SymbolicDomain::Int | super::SymbolicDomain::Real => {
                Ok(self.emit_i64_const(block_idx, 1))
            }
        }
    }

    fn emit_finite_set_membership_i64(
        &mut self,
        block_idx: usize,
        elem_val: ValueId,
        set_ptr: ValueId,
        len: u32,
    ) -> ValueId {
        let mut result = self.emit_i64_const(block_idx, 0);
        for slot in 0..len {
            let set_elem = self.load_at_offset(block_idx, set_ptr, slot + 1);
            let eq = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: elem_val,
                    rhs: set_elem,
                },
            );
            let eq_i64 = self.emit_bool_to_i64(block_idx, eq);
            result = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: result,
                    rhs: eq_i64,
                },
            );
        }
        result
    }

    fn compact_record_materialized_set_fields(
        value_shape: &super::AggregateShape,
        set_element_shape: Option<&super::AggregateShape>,
    ) -> Result<Vec<(u32, u32)>, TrustIrError> {
        let Some(set_element_shape) = set_element_shape else {
            return Err(TrustIrError::UnsupportedOpcode(
                "SetIn: compact record finite-set membership requires tracked record element shape"
                    .to_owned(),
            ));
        };
        let super::AggregateShape::Record { fields: set_fields } = set_element_shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: compact record finite-set membership requires record element shape, got {set_element_shape:?}"
            )));
        };

        let super::AggregateShape::Record {
            fields: value_fields,
        } = value_shape
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: compact record finite-set membership requires record value shape, got {value_shape:?}"
            )));
        };
        if value_fields.len() != set_fields.len() {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: compact record finite-set membership requires compatible record shape, got value {value_shape:?} and set element {set_element_shape:?}"
            )));
        }

        let mut fields = Vec::with_capacity(set_fields.len());
        for (set_field_idx, (field_name, set_field_shape)) in set_fields.iter().enumerate() {
            let Some((compact_offset, value_field_shape)) =
                value_shape.compact_record_field(*field_name)
            else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: compact record finite-set membership missing field {field_name:?} in value shape {value_shape:?}"
                )));
            };
            let (Some(value_field_shape), Some(set_field_shape)) =
                (value_field_shape.as_ref(), set_field_shape.as_deref())
            else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: compact record finite-set membership requires tracked field shape for {field_name:?}"
                )));
            };
            if !Self::is_single_slot_flat_aggregate_value(value_field_shape)
                || !Self::is_single_slot_flat_aggregate_value(set_field_shape)
                || !Self::compatible_flat_aggregate_value(value_field_shape, set_field_shape)
            {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: compact record finite-set membership incompatible field {field_name:?}: value {value_field_shape:?}, set element {set_field_shape:?}"
                )));
            }
            fields.push((
                compact_offset,
                u32::try_from(set_field_idx).expect("record field index must fit in u32"),
            ));
        }

        Ok(fields)
    }

    fn lower_compact_record_materialized_set_membership(
        &mut self,
        block_idx: usize,
        rd: u8,
        source_slot: super::CompactStateSlot,
        value_shape: &super::AggregateShape,
        set_reg: u8,
        set_len: u32,
        set_element_shape: Option<&super::AggregateShape>,
    ) -> Result<Option<usize>, TrustIrError> {
        let fields = Self::compact_record_materialized_set_fields(value_shape, set_element_shape)?;
        if fields.is_empty() {
            self.store_reg_imm(block_idx, rd, i64::from(set_len != 0))?;
            return Ok(Some(block_idx));
        }

        let set_ptr = self.load_reg_as_ptr(block_idx, set_reg)?;
        let mut found = self.emit_i64_const(block_idx, 0);
        for elem_idx in 0..set_len {
            let record_value = self.load_at_offset(block_idx, set_ptr, elem_idx + 1);
            let record_ptr = self.i64_value_as_ptr(block_idx, record_value);
            let mut record_matches: Option<ValueId> = None;
            for (compact_offset, set_field_idx) in &fields {
                let value_field = self.load_at_offset(
                    block_idx,
                    source_slot.source_ptr,
                    source_slot.offset + *compact_offset,
                );
                let set_field = self.load_at_offset(block_idx, record_ptr, *set_field_idx);
                let eq = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: value_field,
                        rhs: set_field,
                    },
                );
                let eq_i64 = self.emit_bool_to_i64(block_idx, eq);
                record_matches = Some(match record_matches {
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
            found = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: found,
                    rhs: record_matches.expect("non-empty record must produce equality result"),
                },
            );
        }

        self.store_reg_value(block_idx, rd, found)?;
        Ok(Some(block_idx))
    }

    /// Lower `record \in recordSetBitmask` (RecordSetBitmask step 2/5).
    ///
    /// The set operand is a multi-slot record bitmask (`slot_count` i64 slots,
    /// bit `i` = universe record `i` is present), the element is a record. For
    /// each universe record `i` we emit:
    ///   * `record_matches_i` = AND over the universe key's fields of
    ///     (element-field == key-field-constant); and
    ///   * `bit_present_i` = `(mask_slot[i/64] >> (i % 64)) & 1`, the membership
    ///     bit at the EXACT index `record_bit_key_index` / the interpreter's
    ///     `record_set_bitmask_value_to_slots` assigns universe record `i`.
    /// The result is `OR_i (record_matches_i AND bit_present_i)`.
    ///
    /// Soundness:
    ///   * The element record's fields are loaded from its pointer-backed
    ///     compact slot via `load_at_offset` — there is NO `IntToPtr` of the
    ///     mask slots, so the rc=139 trap (treating a mask slot as a set base
    ///     pointer) cannot occur. The mask slots are loaded directly from the
    ///     set's pointer-backed compact region.
    ///   * A record OUTSIDE the universe matches no `record_matches_i`, so the
    ///     result is `0` (False) — which is the correct verdict for membership
    ///     in a universe-closed set, never a silent wrong bit.
    ///   * This returns early; the operand never reaches the generic pointer
    ///     scan.
    fn lower_record_set_bitmask_membership(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        set_reg: u8,
        universe_len: u32,
        slot_count: u32,
        universe: &[super::RecordBitKey],
    ) -> Result<Option<usize>, TrustIrError> {
        // Validate the carried shape is internally consistent before emitting
        // any IR (fail-closed, never silently mis-encode).
        if super::record_set_bitmask_slot_count_ir(universe_len) != slot_count as usize
            || universe.len() != universe_len as usize
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: RecordSetBitmask shape inconsistent: universe_len={universe_len}, \
                 slot_count={slot_count}, universe.len()={}",
                universe.len()
            )));
        }

        // The element must be a tracked compact record with a pointer-backed
        // slot. Anything else (untracked, scalar, materialized) fails closed —
        // we never IntToPtr a non-record element here.
        let Some(elem_shape @ super::AggregateShape::Record { .. }) =
            self.aggregate_shapes.get(&elem_reg).cloned()
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: RecordSetBitmask membership requires a tracked record element in r{elem_reg}, \
                 got {:?}",
                self.aggregate_shapes.get(&elem_reg)
            )));
        };
        let Some(elem_slot) = self.compact_state_slot_for_use(block_idx, elem_reg)? else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: RecordSetBitmask membership requires a compact (pointer-backed) record \
                 element in r{elem_reg}"
            )));
        };

        // Resolve every universe key's fields to (element compact offset,
        // expected constant) up front. `None` marks a STATICALLY UNMATCHABLE
        // key: in a heterogeneous record universe (e.g. TwoPhase's
        // `[type,rm] ∪ [type]`), a key field provably absent from the
        // element's fully tracked record shape means the element can never
        // Value-equal that key — the key simply contributes no bit test.
        // Anything short of proven absence (untracked shape, non-record,
        // non-single-slot field) fails closed as before.
        let mut resolved: Vec<Option<Vec<(u32, i64)>>> = Vec::with_capacity(universe.len());
        for key in universe {
            let mut key_fields = Vec::with_capacity(key.fields.len());
            let mut unmatchable = false;
            // Records with different field COUNTS are never Value-equal: a
            // fully tracked element with a different field count than the key
            // is statically unmatchable. (This also closes the superset-field
            // false match: an element carrying every key field PLUS extras
            // would otherwise pass the per-field equalities below.)
            if let super::AggregateShape::Record { fields } = &elem_shape {
                if fields.iter().all(|(_, s)| s.is_some()) && fields.len() != key.fields.len() {
                    resolved.push(None);
                    continue;
                }
            }
            for (field_name, element) in &key.fields {
                let Some((compact_offset, Some(field_shape))) =
                    elem_shape.compact_record_field(*field_name)
                else {
                    if Self::record_shape_field_certainly_absent(&elem_shape, *field_name) {
                        unmatchable = true;
                        break;
                    }
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "SetIn: RecordSetBitmask membership element r{elem_reg} is missing \
                         universe field {field_name:?} (or it has no tracked shape)"
                    )));
                };
                if !Self::is_single_slot_flat_aggregate_value(&field_shape) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "SetIn: RecordSetBitmask membership requires single-slot scalar fields, \
                         field {field_name:?} has shape {field_shape:?}"
                    )));
                }
                key_fields.push((
                    compact_offset,
                    super::set_bitmask_element_compact_value(element),
                ));
            }
            resolved.push((!unmatchable).then_some(key_fields));
        }

        // Load the mask slots from the set's pointer-backed compact region.
        let Some(set_slot) = self.compact_state_slot_for_use(block_idx, set_reg)? else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: RecordSetBitmask membership requires a compact (pointer-backed) mask in \
                 r{set_reg}"
            )));
        };
        let mut slots = Vec::with_capacity(slot_count as usize);
        for slot_index in 0..slot_count {
            slots.push(self.load_at_offset(
                block_idx,
                set_slot.source_ptr,
                set_slot.offset + slot_index,
            ));
        }

        let one = self.emit_i64_const(block_idx, 1);
        let mut found = self.emit_i64_const(block_idx, 0);
        for (index, key_fields) in resolved.iter().enumerate() {
            // A statically unmatchable key (heterogeneous universe: the
            // element's record shape provably lacks one of the key's fields)
            // contributes no bit test — its match is constant false.
            let Some(key_fields) = key_fields else {
                continue;
            };
            // record_matches_i = AND of per-field equalities. An empty universe
            // key (a record with no fields) is treated as a vacuous match (1).
            let mut record_matches: Option<ValueId> = None;
            for (compact_offset, expected) in key_fields {
                let elem_field = self.load_at_offset(
                    block_idx,
                    elem_slot.source_ptr,
                    elem_slot.offset + *compact_offset,
                );
                let expected_val = self.emit_i64_const(block_idx, *expected);
                let eq = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: elem_field,
                        rhs: expected_val,
                    },
                );
                let eq_i64 = self.emit_bool_to_i64(block_idx, eq);
                record_matches = Some(match record_matches {
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
            let record_matches = match record_matches {
                Some(rm) => rm,
                None => one,
            };

            // bit_present_i = (mask_slot[i/64] >> (i % 64)) & 1.
            let slot_value = slots[index / 64];
            let shift = self.emit_i64_const(block_idx, (index % 64) as i64);
            let shifted = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::LShr,
                    ty: Ty::I64,
                    lhs: slot_value,
                    rhs: shift,
                },
            );
            let bit_present = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: shifted,
                    rhs: one,
                },
            );
            let matched_and_present = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: record_matches,
                    rhs: bit_present,
                },
            );
            found = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: found,
                    rhs: matched_and_present,
                },
            );
        }

        self.store_reg_value(block_idx, rd, found)?;
        Ok(Some(block_idx))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_compact_record_materialized_set_membership_branch(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        record_base_slot: ValueId,
        value_shape: &super::AggregateShape,
        set_ptr: ValueId,
        set_len: u32,
        set_element_shape: Option<&super::AggregateShape>,
        success_target: BlockId,
        failure_target: BlockId,
    ) -> Result<(), TrustIrError> {
        let fields = Self::compact_record_materialized_set_fields(value_shape, set_element_shape)?;
        if fields.is_empty() {
            self.emit(
                block_idx,
                InstrNode::new(Inst::Br {
                    target: if set_len == 0 {
                        failure_target
                    } else {
                        success_target
                    },
                    args: vec![],
                }),
            );
            return Ok(());
        }

        let mut found = self.emit_i64_const(block_idx, 0);
        for elem_idx in 0..set_len {
            let record_value = self.load_at_offset(block_idx, set_ptr, elem_idx + 1);
            let record_ptr = self.i64_value_as_ptr(block_idx, record_value);
            let mut record_matches: Option<ValueId> = None;
            for (compact_offset, set_field_idx) in &fields {
                let field_offset = self.emit_i64_const(block_idx, i64::from(*compact_offset));
                let field_slot = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: record_base_slot,
                        rhs: field_offset,
                    },
                );
                let value_field = self.load_at_dynamic_offset(block_idx, source_ptr, field_slot);
                let set_field = self.load_at_offset(block_idx, record_ptr, *set_field_idx);
                let eq = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: value_field,
                        rhs: set_field,
                    },
                );
                let eq_i64 = self.emit_bool_to_i64(block_idx, eq);
                record_matches = Some(match record_matches {
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
            found = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: found,
                    rhs: record_matches.expect("non-empty record must produce equality result"),
                },
            );
        }

        self.branch_on_i64_truth(block_idx, found, success_target, failure_target);
        Ok(())
    }

    fn lower_symbolic_domain_membership(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        domain: super::SymbolicDomain,
    ) -> Result<(), TrustIrError> {
        let elem_val = self.load_reg(block_idx, elem_reg)?;
        let elem_shape = self.aggregate_shapes.get(&elem_reg).cloned();

        // WP-27 (item B1): a tagged-scalar-union register holds the union-slot
        // INDEX, which is always a small non-negative integer. Before this,
        // `is_numeric_scalar_shape` was false for the union shape and the
        // emitter answered a CONSTANT 0 — i.e. `focus \in Nat` compiled to
        // FALSE even when `focus` was an ordinary node integer, a silent
        // divergence rather than a fallback. Decode the index into raw member
        // space first (which fails closed whenever a cross-sort member such as
        // `NIL` would land inside the domain), then answer in the Int lane the
        // decoded value now occupies.
        if matches!(
            elem_shape,
            Some(super::AggregateShape::TaggedScalarUnion { .. })
        ) {
            let space = match domain {
                super::SymbolicDomain::Nat => SetInRawSpace::IntRange {
                    lo: 0,
                    hi: i64::MAX,
                },
                super::SymbolicDomain::Int | super::SymbolicDomain::Real => {
                    SetInRawSpace::IntRange {
                        lo: i64::MIN,
                        hi: i64::MAX,
                    }
                }
            };
            let raw = self.set_in_element_raw_value_i64(
                block_idx,
                elem_reg,
                elem_val,
                &space,
                "SetIn: symbolic-domain membership",
            )?;
            let int_shape = super::AggregateShape::Scalar(super::ScalarShape::Int);
            let member = self.emit_symbolic_domain_membership_i64(
                block_idx,
                raw,
                Some(&int_shape),
                domain,
            )?;
            return self.store_reg_value(block_idx, rd, member);
        }

        let member = self.emit_symbolic_domain_membership_i64(
            block_idx,
            elem_val,
            elem_shape.as_ref(),
            domain,
        )?;
        self.store_reg_value(block_idx, rd, member)
    }

    fn ensure_materialized_set_reg(&self, reg: u8, context: &str) -> Result<(), TrustIrError> {
        match self.aggregate_shapes.get(&reg) {
            Some(super::AggregateShape::StateValue) => Ok(()),
            Some(super::AggregateShape::SetBitmask { .. }) => {
                Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask in r{reg} is a raw mask, not a materialized set aggregate"
                )))
            }
            Some(super::AggregateShape::TaggedScalarOrSet { .. }) => {
                Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: tagged scalar-or-set in r{reg} is a raw slot, not a materialized set aggregate"
                )))
            }
            Some(shape) if shape.is_finite_set_shape() => Ok(()),
            Some(shape) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: expected tracked finite set or state value in r{reg}, got {shape:?}"
            ))),
            None => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: expected tracked finite set or state value in r{reg}"
            ))),
        }
    }

    fn reject_lazy_set_operand(&self, op: &str, reg: u8) -> Result<(), TrustIrError> {
        if let Some(shape) = self.aggregate_shapes.get(&reg) {
            if shape.is_lazy_set_shape() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{op}: lazy set shape is only supported by SetIn membership, got r{reg} = {shape:?}"
                )));
            }
            if matches!(shape, super::AggregateShape::SetBitmask { .. }) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{op}: compact SetBitmask operands require mask-native lowering, got r{reg} = {shape:?}"
                )));
            }
            // RecordSetBitmask is inert scaffolding (RecordSetBitmask step
            // 1/5): its multi-slot record bitmask has no materialized pointer
            // representation, so fail closed in every materialized set op
            // (union/diff/...) rather than mis-reading the mask slots as a set
            // pointer. The native mask lowering arrives in a later step.
            if matches!(shape, super::AggregateShape::RecordSetBitmask { .. }) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{op}: RecordSetBitmask operands require native mask lowering (not yet wired), got r{reg} = {shape:?}"
                )));
            }
            if matches!(shape, super::AggregateShape::TaggedScalarOrSet { .. }) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{op}: tagged scalar-or-set operands require proof-guarded set-branch lowering, got r{reg} = {shape:?}"
                )));
            }
        }
        Ok(())
    }

    fn symbolic_domain_union_source_reg(
        &self,
        r1: u8,
        r2: u8,
    ) -> Option<(u8, super::SymbolicDomain)> {
        let left = self.aggregate_shapes.get(&r1)?;
        let right = self.aggregate_shapes.get(&r2)?;
        match (left, right) {
            (super::AggregateShape::SymbolicDomain(domain), other)
                if super::finite_set_shape_subset_of_symbolic_domain(other, *domain) =>
            {
                Some((r1, *domain))
            }
            (other, super::AggregateShape::SymbolicDomain(domain))
                if super::finite_set_shape_subset_of_symbolic_domain(other, *domain) =>
            {
                Some((r2, *domain))
            }
            _ => None,
        }
    }

    fn lower_powerset_membership(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        powerset_reg: u8,
        base_shape: super::AggregateShape,
        require_nonempty: bool,
    ) -> Result<Option<usize>, TrustIrError> {
        base_shape.validate_powerset_base("SetIn: SUBSET membership")?;

        if let Some(elem_shape @ super::AggregateShape::SetBitmask { .. }) =
            self.aggregate_shapes.get(&elem_reg).cloned()
        {
            let Some((universe_len, universe)) = elem_shape.set_bitmask_universe() else {
                return Err(TrustIrError::UnsupportedOpcode(
                    "SetIn: SUBSET compact membership requires exact universe metadata".to_owned(),
                ));
            };
            let elem_mask = self.load_reg(block_idx, elem_reg)?;
            let true_blk = self.new_aux_block("powerset_member_compact_true");
            let false_blk = self.new_aux_block("powerset_member_compact_false");
            let merge_blk = self.new_aux_block("powerset_member_compact_merge");
            let true_id = self.block_id_of(true_blk);
            let false_id = self.block_id_of(false_blk);
            let merge_id = self.block_id_of(merge_blk);

            if matches!(&base_shape, super::AggregateShape::SetBitmask { .. }) {
                let base_mask = self.load_reg(block_idx, powerset_reg)?;
                self.lower_compact_bitmask_runtime_powerset_mask_branch(
                    block_idx,
                    elem_mask,
                    base_mask,
                    &base_shape,
                    universe_len,
                    &universe,
                    true_id,
                    false_id,
                    "SetIn: SUBSET compact membership",
                )?;
            } else {
                self.lower_compact_bitmask_powerset_branch(
                    block_idx,
                    elem_mask,
                    &base_shape,
                    universe_len,
                    &universe,
                    true_id,
                    false_id,
                    "SetIn: SUBSET compact membership",
                )?;
            }

            self.finish_powerset_membership_success(
                true_blk,
                false_blk,
                merge_id,
                rd,
                elem_mask,
                Some(&elem_shape),
                require_nonempty,
                "SetIn: non-empty SUBSET compact membership",
            )?;
            self.store_reg_imm(false_blk, rd, 0)?;
            self.emit(
                false_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );
            return Ok(Some(merge_blk));
        }

        if let super::AggregateShape::SetBitmask {
            universe_len,
            universe,
        } = &base_shape
        {
            let (block_idx, elem_mask, elem_in_universe) = self
                .emit_set_subseteq_operand_bitmask_i64(
                    block_idx,
                    elem_reg,
                    *universe_len,
                    universe,
                    "SetIn: SUBSET compact base membership",
                )?;
            let base_mask = self.load_reg(block_idx, powerset_reg)?;
            let true_blk = self.new_aux_block("powerset_member_compact_base_true");
            let false_blk = self.new_aux_block("powerset_member_compact_base_false");
            let merge_blk = self.new_aux_block("powerset_member_compact_base_merge");
            let true_id = self.block_id_of(true_blk);
            let false_id = self.block_id_of(false_blk);
            let merge_id = self.block_id_of(merge_blk);

            self.lower_set_bitmask_subseteq_mask_branch(
                block_idx,
                elem_mask,
                elem_in_universe,
                base_mask,
                *universe_len,
                true_id,
                false_id,
                "SetIn: SUBSET compact base membership",
            )?;

            // On the subset-success edge every candidate element is in-universe,
            // so the projected `elem_mask` is `0` iff the candidate is empty.
            // `true_blk` means "subset holds"; route it through the non-empty
            // guard (when required) before committing `rd = 1`.
            let member_blk = self.new_aux_block("powerset_member_compact_base_member");
            let member_id = self.block_id_of(member_blk);
            if require_nonempty {
                self.branch_on_i64_truth(true_blk, elem_mask, member_id, false_id);
            } else {
                self.emit(
                    true_blk,
                    InstrNode::new(Inst::Br {
                        target: member_id,
                        args: vec![],
                    }),
                );
            }
            self.store_reg_imm(member_blk, rd, 1)?;
            self.emit(
                member_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );
            self.store_reg_imm(false_blk, rd, 0)?;
            self.emit(
                false_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );
            return Ok(Some(merge_blk));
        }

        self.ensure_materialized_set_reg(elem_reg, "SetIn: SUBSET membership element")?;

        let elem_ptr = self.load_reg_as_ptr(block_idx, elem_reg)?;
        // Powerset registers carry the base set pointer as their runtime value.
        let base_ptr = self.load_reg_as_ptr(block_idx, powerset_reg)?;

        let true_blk = self.new_aux_block("powerset_member_true");
        let false_blk = self.new_aux_block("powerset_member_false");
        let merge_blk = self.new_aux_block("powerset_member_merge");
        let true_id = self.block_id_of(true_blk);
        let false_id = self.block_id_of(false_blk);
        let merge_id = self.block_id_of(merge_blk);

        self.lower_subseteq_ptr_branch(
            block_idx,
            elem_ptr,
            base_ptr,
            true_id,
            false_id,
            "powerset_member",
        );

        // `true_blk` means "candidate is a subset of the base". For the
        // non-empty powerset we additionally require a non-empty candidate; the
        // pointer-backed cardinality lives in slot 0.
        let member_blk = self.new_aux_block("powerset_member_member");
        let member_id = self.block_id_of(member_blk);
        if require_nonempty {
            let len = self.load_at_offset(true_blk, elem_ptr, 0);
            self.branch_on_i64_truth(true_blk, len, member_id, false_id);
        } else {
            self.emit(
                true_blk,
                InstrNode::new(Inst::Br {
                    target: member_id,
                    args: vec![],
                }),
            );
        }
        self.store_reg_imm(member_blk, rd, 1)?;
        self.emit(
            member_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        self.store_reg_imm(false_blk, rd, 0)?;
        self.emit(
            false_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge_blk))
    }

    /// Commits the membership result on the subset-success edge (`true_blk`).
    ///
    /// For an ordinary powerset, subset-success directly means membership
    /// (`rd = 1`). For a non-empty powerset (`require_nonempty`), the candidate
    /// must also be non-empty, so `true_blk` is routed through the non-empty
    /// guard, which sends empty candidates to `false_blk` (`rd = 0`) instead.
    #[allow(clippy::too_many_arguments)]
    fn finish_powerset_membership_success(
        &mut self,
        true_blk: usize,
        false_blk: usize,
        merge_id: BlockId,
        rd: u8,
        candidate_value: ValueId,
        candidate_shape: Option<&super::AggregateShape>,
        require_nonempty: bool,
        context: &str,
    ) -> Result<(), TrustIrError> {
        let member_blk = self.new_aux_block("powerset_member_committed");
        let member_id = self.block_id_of(member_blk);
        let false_id = self.block_id_of(false_blk);
        if require_nonempty {
            self.branch_on_set_nonempty(
                true_blk,
                candidate_value,
                candidate_shape,
                member_id,
                false_id,
                context,
            )?;
        } else {
            self.emit(
                true_blk,
                InstrNode::new(Inst::Br {
                    target: member_id,
                    args: vec![],
                }),
            );
        }
        self.store_reg_imm(member_blk, rd, 1)?;
        self.emit(
            member_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );
        Ok(())
    }

    fn lower_seq_set_membership(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        seq_set_reg: u8,
        base_shape: super::AggregateShape,
    ) -> Result<Option<usize>, TrustIrError> {
        base_shape.validate_seq_base("SetIn: Seq membership")?;
        let elem_shape = self.aggregate_shapes.get(&elem_reg).cloned();
        if let (Some(source_slot), Some(seq_shape @ super::AggregateShape::Sequence { .. })) = (
            self.compact_state_slot_for_use(block_idx, elem_reg)?,
            elem_shape.as_ref(),
        ) {
            return self.lower_compact_sequence_seq_set_membership_to_reg(
                block_idx,
                rd,
                source_slot,
                seq_shape,
                seq_set_reg,
                base_shape,
            );
        }

        let seq_element_shape = match elem_shape {
            Some(super::AggregateShape::Sequence { element, .. }) => element.as_deref().cloned(),
            Some(super::AggregateShape::StateValue) | None => None,
            Some(other) => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: Seq membership requires sequence element shape, got {other:?}"
                )));
            }
        };

        let seq_ptr = self.load_reg_as_ptr(block_idx, elem_reg)?;
        // Seq(S) carries S's runtime payload unchanged; compact bases are masks.
        let base_value = if Self::lazy_domain_runtime_payload_is_compact_mask(&base_shape) {
            self.load_reg(block_idx, seq_set_reg)?
        } else {
            self.load_reg_as_ptr(block_idx, seq_set_reg)?
        };

        let true_blk = self.new_aux_block("seqset_member_true");
        let false_blk = self.new_aux_block("seqset_member_false");
        let merge_blk = self.new_aux_block("seqset_member_merge");
        let true_id = self.block_id_of(true_blk);
        let false_id = self.block_id_of(false_blk);
        let merge_id = self.block_id_of(merge_blk);

        self.lower_seq_ptr_in_seq_set_ptr_branch(
            block_idx,
            seq_ptr,
            seq_element_shape.as_ref(),
            base_value,
            base_shape,
            true_id,
            false_id,
            "seqset_member",
        )?;

        self.store_reg_imm(true_blk, rd, 1)?;
        self.emit(
            true_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        self.store_reg_imm(false_blk, rd, 0)?;
        self.emit(
            false_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge_blk))
    }

    fn lower_compact_sequence_seq_set_membership_to_reg(
        &mut self,
        block_idx: usize,
        rd: u8,
        source_slot: super::CompactStateSlot,
        seq_shape: &super::AggregateShape,
        seq_set_reg: u8,
        base_shape: super::AggregateShape,
    ) -> Result<Option<usize>, TrustIrError> {
        let true_blk = self.new_aux_block("compact_seqset_member_true");
        let false_blk = self.new_aux_block("compact_seqset_member_false");
        let merge_blk = self.new_aux_block("compact_seqset_member_merge");
        let true_id = self.block_id_of(true_blk);
        let false_id = self.block_id_of(false_blk);
        let merge_id = self.block_id_of(merge_blk);

        let seq_base_slot = self.emit_i64_const(block_idx, i64::from(source_slot.offset));
        let base_value = if Self::lazy_domain_runtime_payload_is_compact_mask(&base_shape) {
            self.load_reg(block_idx, seq_set_reg)?
        } else {
            self.load_reg_as_ptr(block_idx, seq_set_reg)?
        };
        self.lower_compact_sequence_value_in_seq_set_branch(
            block_idx,
            source_slot.source_ptr,
            seq_base_slot,
            Some(seq_shape),
            base_value,
            base_shape,
            true_id,
            false_id,
            "compact_seqset_member",
        )?;

        self.store_reg_imm(true_blk, rd, 1)?;
        self.emit(
            true_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );
        self.store_reg_imm(false_blk, rd, 0)?;
        self.emit(
            false_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge_blk))
    }

    fn lower_seq_value_in_seq_set_ptr_branch(
        &mut self,
        block_idx: usize,
        seq_value: ValueId,
        seq_element_shape: Option<&super::AggregateShape>,
        base_value: ValueId,
        base_shape: super::AggregateShape,
        success_target: BlockId,
        failure_target: BlockId,
        prefix: &str,
    ) -> Result<(), TrustIrError> {
        let seq_ptr = self.i64_value_as_ptr(block_idx, seq_value);
        self.lower_seq_ptr_in_seq_set_ptr_branch(
            block_idx,
            seq_ptr,
            seq_element_shape,
            base_value,
            base_shape,
            success_target,
            failure_target,
            prefix,
        )
    }

    fn lower_seq_ptr_in_seq_set_ptr_branch(
        &mut self,
        block_idx: usize,
        seq_ptr: ValueId,
        seq_element_shape: Option<&super::AggregateShape>,
        base_value: ValueId,
        base_shape: super::AggregateShape,
        success_target: BlockId,
        failure_target: BlockId,
        prefix: &str,
    ) -> Result<(), TrustIrError> {
        base_shape.validate_seq_base(prefix)?;

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

        let loop_hdr = self.new_aux_block(&format!("{prefix}_seq_hdr"));
        let loop_body = self.new_aux_block(&format!("{prefix}_seq_body"));
        let loop_inc = self.new_aux_block(&format!("{prefix}_seq_inc"));
        let loop_hdr_id = self.block_id_of(loop_hdr);
        let loop_body_id = self.block_id_of(loop_body);
        let loop_inc_id = self.block_id_of(loop_inc);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        let idx = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let seq_len = self.load_at_offset(loop_hdr, seq_ptr, 0);
        let in_bounds = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: idx,
                rhs: seq_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: loop_body_id,
                then_args: vec![],
                else_target: success_target,
                else_args: vec![],
            }),
        );

        let idx_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_body, 1);
        let slot_idx = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: idx_body,
                rhs: one,
            },
        );
        let elem = self.load_at_dynamic_offset(loop_body, seq_ptr, slot_idx);
        match base_shape {
            super::AggregateShape::SetBitmask {
                universe_len,
                universe,
            } => self.lower_scalar_in_set_bitmask_shape_branch(
                loop_body,
                elem,
                seq_element_shape,
                base_value,
                universe_len,
                &universe,
                loop_inc_id,
                failure_target,
                &format!("{prefix}_elem"),
            )?,
            base_shape => self.lower_value_in_domain_ptr_branch(
                loop_body,
                elem,
                seq_element_shape,
                base_value,
                base_shape,
                loop_inc_id,
                failure_target,
                &format!("{prefix}_elem"),
            )?,
        }

        let idx_inc = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_inc, 1);
        let next_idx = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: idx_inc,
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
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        Ok(())
    }

    fn lower_value_in_set_ptr_branch(
        &mut self,
        block_idx: usize,
        elem_val: ValueId,
        set_ptr: ValueId,
        success_target: BlockId,
        failure_target: BlockId,
        prefix: &str,
    ) {
        let set_len = self.load_at_offset(block_idx, set_ptr, 0);
        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

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

        let loop_header = self.new_aux_block(&format!("{prefix}_setin_header"));
        let loop_body = self.new_aux_block(&format!("{prefix}_setin_body"));
        let loop_inc = self.new_aux_block(&format!("{prefix}_setin_inc"));
        let header_id = self.block_id_of(loop_header);
        let body_id = self.block_id_of(loop_body);
        let inc_id = self.block_id_of(loop_inc);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

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
                rhs: set_len,
            },
        );
        self.emit(
            loop_header,
            InstrNode::new(Inst::CondBr {
                cond: cmp,
                then_target: body_id,
                then_args: vec![],
                else_target: failure_target,
                else_args: vec![],
            }),
        );

        let cur_idx2 = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot_idx = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx2,
                rhs: one,
            },
        );
        let set_elem = self.load_at_dynamic_offset(loop_body, set_ptr, slot_idx);
        let eq = self.emit_with_result(
            loop_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: set_elem,
                rhs: elem_val,
            },
        );
        self.emit(
            loop_body,
            InstrNode::new(Inst::CondBr {
                cond: eq,
                then_target: success_target,
                then_args: vec![],
                else_target: inc_id,
                else_args: vec![],
            }),
        );

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
    }

    fn lower_value_in_domain_ptr_branch(
        &mut self,
        block_idx: usize,
        elem_val: ValueId,
        elem_shape: Option<&super::AggregateShape>,
        domain_ptr: ValueId,
        domain_shape: super::AggregateShape,
        success_target: BlockId,
        failure_target: BlockId,
        prefix: &str,
    ) -> Result<(), TrustIrError> {
        let domain_shape = Self::materialized_domain_shape_for_pointer(domain_shape);
        match domain_shape {
            super::AggregateShape::Interval { lo, hi } => {
                let member = self.emit_interval_membership_i64(block_idx, elem_val, lo, hi);
                self.branch_on_i64_truth(block_idx, member, success_target, failure_target);
            }
            super::AggregateShape::Set { .. }
            | super::AggregateShape::ExactIntSet { .. }
            | super::AggregateShape::ExactScalarSet { .. }
            | super::AggregateShape::FiniteSet
            | super::AggregateShape::BoundedSet { .. } => {
                // Soundness wall (lever L1 fallout): the raw materialized scan
                // below compares the candidate's i64 REPRESENTATION against
                // stored element slots. A tagged scalar-or-set slot or a
                // compact set mask is an ENCODING, not a value: its set lane
                // can never equal a stored aggregate POINTER, so the scan
                // would return false for genuinely-member set values (a
                // spurious invariant violation on e.g. a const-folded
                // `(SUBSET Proc) \cup ...` TypeOK range). Fail closed instead
                // of scanning; the const-set lazy-union reconstruction
                // (`lazy_union_shape_from_const_set_elements`) covers the
                // sound cases natively.
                if matches!(
                    elem_shape,
                    Some(
                        super::AggregateShape::TaggedScalarOrSet { .. }
                            | super::AggregateShape::SetBitmask { .. }
                            | super::AggregateShape::RecordSetBitmask { .. }
                    )
                ) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{prefix}: raw-slot-encoded candidate (tagged scalar-or-set / compact \
                         set mask) cannot be compared against materialized set elements by raw \
                         scan — failing closed, got {elem_shape:?}"
                    )));
                }
                self.lower_value_in_set_ptr_branch(
                    block_idx,
                    elem_val,
                    domain_ptr,
                    success_target,
                    failure_target,
                    prefix,
                );
            }
            super::AggregateShape::SymbolicDomain(domain) => {
                let member = self
                    .emit_symbolic_domain_membership_i64(block_idx, elem_val, elem_shape, domain)?;
                self.branch_on_i64_truth(block_idx, member, success_target, failure_target);
            }
            super::AggregateShape::RecordSet { fields } => {
                self.lower_record_value_in_record_set_ptr_branch(
                    block_idx,
                    elem_val,
                    domain_ptr,
                    fields,
                    success_target,
                    failure_target,
                )?;
            }
            super::AggregateShape::Powerset { base } => {
                base.validate_powerset_base(&format!("{prefix}: SUBSET base"))?;
                if let Some(elem_shape @ super::AggregateShape::SetBitmask { .. }) = elem_shape {
                    let Some((universe_len, universe)) = elem_shape.set_bitmask_universe() else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "{prefix}: SUBSET compact membership requires exact universe metadata"
                        )));
                    };
                    if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                        self.lower_compact_bitmask_runtime_powerset_mask_branch(
                            block_idx,
                            elem_val,
                            domain_ptr,
                            &base,
                            universe_len,
                            &universe,
                            success_target,
                            failure_target,
                            prefix,
                        )?;
                    } else {
                        self.lower_compact_bitmask_powerset_branch(
                            block_idx,
                            elem_val,
                            &base,
                            universe_len,
                            &universe,
                            success_target,
                            failure_target,
                            prefix,
                        )?;
                    }
                } else {
                    if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "{prefix}: compact SUBSET membership requires SetBitmask element shape, got {elem_shape:?}"
                        )));
                    }
                    let elem_ptr = self.i64_value_as_ptr(block_idx, elem_val);
                    self.lower_subseteq_ptr_branch(
                        block_idx,
                        elem_ptr,
                        domain_ptr,
                        success_target,
                        failure_target,
                        prefix,
                    );
                }
            }
            // `(SUBSET S) \ {{}}`: same subset machinery as `Powerset`, but the
            // candidate must additionally be non-empty. The subset-success edge
            // is routed through a guard block enforcing non-emptiness; the
            // subset-failure edge goes straight to `failure_target`.
            super::AggregateShape::NonEmptyPowerset { base } => {
                base.validate_powerset_base(&format!("{prefix}: non-empty SUBSET base"))?;
                let guard_blk = self.new_aux_block("nonempty_powerset_ptr_guard");
                let guard_id = self.block_id_of(guard_blk);
                if let Some(elem_shape @ super::AggregateShape::SetBitmask { .. }) = elem_shape {
                    let Some((universe_len, universe)) = elem_shape.set_bitmask_universe() else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "{prefix}: non-empty SUBSET compact membership requires exact universe metadata"
                        )));
                    };
                    if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                        self.lower_compact_bitmask_runtime_powerset_mask_branch(
                            block_idx,
                            elem_val,
                            domain_ptr,
                            &base,
                            universe_len,
                            &universe,
                            guard_id,
                            failure_target,
                            prefix,
                        )?;
                    } else {
                        self.lower_compact_bitmask_powerset_branch(
                            block_idx,
                            elem_val,
                            &base,
                            universe_len,
                            &universe,
                            guard_id,
                            failure_target,
                            prefix,
                        )?;
                    }
                    self.branch_on_set_nonempty(
                        guard_blk,
                        elem_val,
                        Some(elem_shape),
                        success_target,
                        failure_target,
                        prefix,
                    )?;
                } else {
                    if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "{prefix}: compact non-empty SUBSET membership requires SetBitmask element shape, got {elem_shape:?}"
                        )));
                    }
                    let elem_ptr = self.i64_value_as_ptr(block_idx, elem_val);
                    self.lower_subseteq_ptr_branch(
                        block_idx,
                        elem_ptr,
                        domain_ptr,
                        guard_id,
                        failure_target,
                        prefix,
                    );
                    self.branch_on_set_nonempty(
                        guard_blk,
                        elem_val,
                        elem_shape,
                        success_target,
                        failure_target,
                        prefix,
                    )?;
                }
            }
            super::AggregateShape::FunctionSet { domain, range } => {
                self.lower_function_like_value_in_function_set_ptr_branch(
                    block_idx,
                    elem_val,
                    elem_shape,
                    domain_ptr,
                    *domain,
                    *range,
                    success_target,
                    failure_target,
                    prefix,
                )?;
            }
            super::AggregateShape::SeqSet { base } => {
                let seq_element_shape = match elem_shape {
                    Some(super::AggregateShape::Sequence { element, .. }) => element.as_deref(),
                    _ => None,
                };
                self.lower_seq_value_in_seq_set_ptr_branch(
                    block_idx,
                    elem_val,
                    seq_element_shape,
                    domain_ptr,
                    *base,
                    success_target,
                    failure_target,
                    prefix,
                )?;
            }
            other => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{prefix}: unsupported membership domain shape: {other:?}"
                )));
            }
        }
        Ok(())
    }

    fn branch_on_i64_truth(
        &mut self,
        block_idx: usize,
        value: ValueId,
        success_target: BlockId,
        failure_target: BlockId,
    ) {
        let zero = self.emit_i64_const(block_idx, 0);
        let is_true = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: value,
                rhs: zero,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: is_true,
                then_target: success_target,
                then_args: vec![],
                else_target: failure_target,
                else_args: vec![],
            }),
        );
    }

    /// Branches to `success_target` iff the candidate set `value` (with the
    /// given `value_shape`) is non-empty, otherwise to `failure_target`.
    ///
    /// This is the membership guard for `NonEmptyPowerset`: a candidate is a
    /// member of `(SUBSET S) \ {{}}` only if it is a subset of `S` *and* it is
    /// not the empty set. The subset half is emitted by the existing powerset
    /// branch helpers; this guard enforces the non-empty half.
    ///
    /// Emptiness is decided from the candidate's runtime representation:
    /// * `SetBitmask` (and tagged/scalar-or-set) values are bitmasks, so the
    ///   empty set is mask `0`; non-empty is `value != 0`.
    /// * pointer-backed sets store their cardinality in slot 0, so non-empty is
    ///   `len != 0`.
    ///
    /// Any other candidate shape is rejected (returns an error) rather than
    /// guessed, so an unrecognised representation can never silently admit the
    /// empty set.
    fn branch_on_set_nonempty(
        &mut self,
        block_idx: usize,
        value: ValueId,
        value_shape: Option<&super::AggregateShape>,
        success_target: BlockId,
        failure_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        match value_shape {
            Some(
                super::AggregateShape::SetBitmask { .. }
                | super::AggregateShape::TaggedScalarOrSet { .. },
            ) => {
                // Bitmask representation: empty set == mask 0.
                self.branch_on_i64_truth(block_idx, value, success_target, failure_target);
                Ok(())
            }
            Some(
                super::AggregateShape::Set { .. }
                | super::AggregateShape::ExactIntSet { .. }
                | super::AggregateShape::ExactScalarSet { .. }
                | super::AggregateShape::FiniteSet
                | super::AggregateShape::BoundedSet { .. },
            ) => {
                // Pointer-backed set: cardinality lives in slot 0.
                let set_ptr = self.i64_value_as_ptr(block_idx, value);
                let len = self.load_at_offset(block_idx, set_ptr, 0);
                self.branch_on_i64_truth(block_idx, len, success_target, failure_target);
                Ok(())
            }
            other => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: non-empty powerset membership requires a recognised set candidate shape, got {other:?}"
            ))),
        }
    }

    fn lower_subseteq_ptr_branch(
        &mut self,
        block_idx: usize,
        set1_ptr: ValueId,
        set2_ptr: ValueId,
        success_target: BlockId,
        failure_target: BlockId,
        prefix: &str,
    ) {
        let len1 = self.load_at_offset(block_idx, set1_ptr, 0);
        let len2 = self.load_at_offset(block_idx, set2_ptr, 0);

        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

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

        let outer_hdr = self.new_aux_block(&format!("{prefix}_subseteq_outer_hdr"));
        let outer_body = self.new_aux_block(&format!("{prefix}_subseteq_outer_body"));
        let inner_hdr = self.new_aux_block(&format!("{prefix}_subseteq_inner_hdr"));
        let inner_body = self.new_aux_block(&format!("{prefix}_subseteq_inner_body"));
        let inner_inc = self.new_aux_block(&format!("{prefix}_subseteq_inner_inc"));
        let outer_inc = self.new_aux_block(&format!("{prefix}_subseteq_outer_inc"));

        let outer_hdr_id = self.block_id_of(outer_hdr);
        let outer_body_id = self.block_id_of(outer_body);
        let inner_hdr_id = self.block_id_of(inner_hdr);
        let inner_body_id = self.block_id_of(inner_body);
        let inner_inc_id = self.block_id_of(inner_inc);
        let outer_inc_id = self.block_id_of(outer_inc);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: outer_hdr_id,
                args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            outer_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp = self.emit_with_result(
            outer_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: len1,
            },
        );
        self.emit(
            outer_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp,
                then_target: outer_body_id,
                then_args: vec![],
                else_target: success_target,
                else_args: vec![],
            }),
        );

        let i_val2 = self.emit_with_result(
            outer_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot = self.emit_with_result(
            outer_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        let elem1 = self.load_at_dynamic_offset(outer_body, set1_ptr, slot);

        let j_alloca = self.emit_with_result(
            outer_body,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            outer_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: j_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            outer_body,
            InstrNode::new(Inst::Br {
                target: inner_hdr_id,
                args: vec![],
            }),
        );

        let j_val = self.emit_with_result(
            inner_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp2 = self.emit_with_result(
            inner_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: j_val,
                rhs: len2,
            },
        );
        self.emit(
            inner_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp2,
                then_target: inner_body_id,
                then_args: vec![],
                else_target: failure_target,
                else_args: vec![],
            }),
        );

        let j_val2 = self.emit_with_result(
            inner_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot2 = self.emit_with_result(
            inner_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: j_val2,
                rhs: one,
            },
        );
        let elem2 = self.load_at_dynamic_offset(inner_body, set2_ptr, slot2);
        let eq = self.emit_with_result(
            inner_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: elem1,
                rhs: elem2,
            },
        );
        self.emit(
            inner_body,
            InstrNode::new(Inst::CondBr {
                cond: eq,
                then_target: outer_inc_id,
                then_args: vec![],
                else_target: inner_inc_id,
                else_args: vec![],
            }),
        );

        let j_val3 = self.emit_with_result(
            inner_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_j = self.emit_with_result(
            inner_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: j_val3,
                rhs: one,
            },
        );
        self.emit(
            inner_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: j_alloca,
                value: next_j,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            inner_inc,
            InstrNode::new(Inst::Br {
                target: inner_hdr_id,
                args: vec![],
            }),
        );

        let i_val3 = self.emit_with_result(
            outer_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_i = self.emit_with_result(
            outer_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val3,
                rhs: one,
            },
        );
        self.emit(
            outer_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            outer_inc,
            InstrNode::new(Inst::Br {
                target: outer_hdr_id,
                args: vec![],
            }),
        );
    }

    fn lower_function_like_set_membership_to_reg(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        funcset_reg: u8,
        elem_shape: Option<&super::AggregateShape>,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
    ) -> Result<Option<usize>, TrustIrError> {
        let true_blk = self.new_aux_block("funcset_member_true");
        let false_blk = self.new_aux_block("funcset_member_false");
        let merge_blk = self.new_aux_block("funcset_member_merge");
        let true_id = self.block_id_of(true_blk);
        let false_id = self.block_id_of(false_blk);
        let merge_id = self.block_id_of(merge_blk);

        let elem_value = self.load_reg(block_idx, elem_reg)?;
        let funcset_ptr = self.load_reg_as_ptr(block_idx, funcset_reg)?;
        self.lower_function_like_value_in_function_set_ptr_branch(
            block_idx,
            elem_value,
            elem_shape,
            funcset_ptr,
            domain_shape,
            range_shape,
            true_id,
            false_id,
            "funcset_member",
        )?;

        self.store_reg_imm(true_blk, rd, 1)?;
        self.emit(
            true_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        self.store_reg_imm(false_blk, rd, 0)?;
        self.emit(
            false_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge_blk))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_compact_sequence_function_set_membership_to_reg(
        &mut self,
        block_idx: usize,
        rd: u8,
        source_slot: super::CompactStateSlot,
        seq_shape: &super::AggregateShape,
        funcset_reg: u8,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
    ) -> Result<Option<usize>, TrustIrError> {
        let true_blk = self.new_aux_block("compact_seq_funcset_member_true");
        let false_blk = self.new_aux_block("compact_seq_funcset_member_false");
        let merge_blk = self.new_aux_block("compact_seq_funcset_member_merge");
        let true_id = self.block_id_of(true_blk);
        let false_id = self.block_id_of(false_blk);
        let merge_id = self.block_id_of(merge_blk);

        let seq_base_slot = self.emit_i64_const(block_idx, i64::from(source_slot.offset));
        let funcset_ptr = self.load_reg_as_ptr(block_idx, funcset_reg)?;
        self.lower_compact_sequence_value_in_function_set_ptr_branch(
            block_idx,
            source_slot.source_ptr,
            seq_base_slot,
            seq_shape,
            funcset_ptr,
            domain_shape,
            range_shape,
            true_id,
            false_id,
            "compact_seq_funcset_member",
        )?;

        self.store_reg_imm(true_blk, rd, 1)?;
        self.emit(
            true_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );
        self.store_reg_imm(false_blk, rd, 0)?;
        self.emit(
            false_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge_blk))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_compact_sequence_value_in_function_set_ptr_branch(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        seq_base_slot: ValueId,
        seq_shape: &super::AggregateShape,
        funcset_ptr: ValueId,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
        then_target: BlockId,
        else_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        domain_shape.validate_powerset_base(&format!("{context}: function-set domain"))?;
        range_shape.validate_function_set_range(&format!("{context}: function-set range"))?;

        let domain_len = domain_shape.tracked_len().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: function-set domain cardinality is not statically known: {domain_shape:?}"
            ))
        })?;
        let super::AggregateShape::Sequence {
            extent: seq_extent,
            element,
        } = seq_shape
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact sequence membership requires tracked sequence shape, got {seq_shape:?}"
            )));
        };
        let domain_is_sequence_domain =
            matches!(domain_shape, super::AggregateShape::Interval { lo: 1, .. });
        let seq_capacity = seq_extent.capacity();
        let compatible_domain = seq_extent
            .exact_count()
            .map_or(domain_len <= seq_capacity, |seq_len| seq_len == domain_len);
        if !domain_is_sequence_domain || !compatible_domain {
            self.emit(
                block_idx,
                InstrNode::new(Inst::Br {
                    target: else_target,
                    args: vec![],
                }),
            );
            return Ok(());
        }

        let value_shape = element.as_deref();
        let Some(value_stride) = value_shape.and_then(super::AggregateShape::compact_slot_count)
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact sequence membership requires fixed-width element layout, got {value_shape:?}"
            )));
        };

        let range_value = self.load_at_offset(block_idx, funcset_ptr, 1);

        let len_ok_blk = self.new_aux_block("compact_seq_funcset_member_len_ok");
        let loop_hdr = self.new_aux_block("compact_seq_funcset_member_hdr");
        let loop_body = self.new_aux_block("compact_seq_funcset_member_body");
        let range_ok_blk = self.new_aux_block("compact_seq_funcset_member_range_ok");
        let loop_inc = self.new_aux_block("compact_seq_funcset_member_inc");
        let len_ok_id = self.block_id_of(len_ok_blk);
        let loop_hdr_id = self.block_id_of(loop_hdr);
        let loop_body_id = self.block_id_of(loop_body);
        let range_ok_id = self.block_id_of(range_ok_blk);
        let loop_inc_id = self.block_id_of(loop_inc);

        let len_check_blk = self.guard_compact_sequence_dynamic_len_in_bounds(
            block_idx,
            source_ptr,
            seq_base_slot,
            seq_capacity,
            &format!("{context}_sequence_len"),
        );
        let actual_len = self.load_at_dynamic_offset(len_check_blk, source_ptr, seq_base_slot);
        let expected_len_for_cmp = self.emit_i64_const(len_check_blk, i64::from(domain_len));
        let len_matches = self.emit_with_result(
            len_check_blk,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: actual_len,
                rhs: expected_len_for_cmp,
            },
        );
        self.emit(
            len_check_blk,
            InstrNode::new(Inst::CondBr {
                cond: len_matches,
                then_target: len_ok_id,
                then_args: vec![],
                else_target,
                else_args: vec![],
            }),
        );

        let zero = self.emit_i64_const(len_ok_blk, 0);
        let idx_alloca = self.emit_with_result(
            len_ok_blk,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let expected_len = self.emit_i64_const(loop_hdr, i64::from(domain_len));
        let in_bounds = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: expected_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: loop_body_id,
                then_args: vec![],
                else_target: then_target,
                else_args: vec![],
            }),
        );

        let i_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let stride = self.emit_i64_const(loop_body, i64::from(value_stride));
        let one = self.emit_i64_const(loop_body, 1);
        let elem_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_body,
                rhs: stride,
            },
        );
        let first_elem_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: seq_base_slot,
                rhs: one,
            },
        );
        let value_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: first_elem_slot,
                rhs: elem_offset,
            },
        );
        let value_val = self.load_at_dynamic_offset(loop_body, source_ptr, value_slot);

        match range_shape {
            super::AggregateShape::Powerset { base } => {
                let (universe_len, universe) =
                    Self::compact_powerset_mask_universe_for_value_shape(value_shape, context)?;
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_compact_bitmask_runtime_powerset_mask_branch(
                        loop_body,
                        value_val,
                        range_value,
                        &base,
                        universe_len,
                        &universe,
                        range_ok_id,
                        else_target,
                        context,
                    )?;
                } else {
                    self.lower_compact_bitmask_powerset_branch(
                        loop_body,
                        value_val,
                        &base,
                        universe_len,
                        &universe,
                        range_ok_id,
                        else_target,
                        context,
                    )?;
                }
            }
            // `(SUBSET S) \ {{}}` per-element range: same compact subset check
            // as `Powerset`, but each entry must additionally be non-empty. The
            // subset-success edge is routed through a guard block; for compact
            // `SetBitmask` entries the empty set is mask `0`.
            super::AggregateShape::NonEmptyPowerset { base } => {
                let (universe_len, universe) =
                    Self::compact_powerset_mask_universe_for_value_shape(value_shape, context)?;
                let guard_blk = self.new_aux_block("compact_seq_nonempty_powerset_guard");
                let guard_id = self.block_id_of(guard_blk);
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_compact_bitmask_runtime_powerset_mask_branch(
                        loop_body,
                        value_val,
                        range_value,
                        &base,
                        universe_len,
                        &universe,
                        guard_id,
                        else_target,
                        context,
                    )?;
                } else {
                    self.lower_compact_bitmask_powerset_branch(
                        loop_body,
                        value_val,
                        &base,
                        universe_len,
                        &universe,
                        guard_id,
                        else_target,
                        context,
                    )?;
                }
                self.branch_on_set_nonempty(
                    guard_blk,
                    value_val,
                    value_shape,
                    range_ok_id,
                    else_target,
                    context,
                )?;
            }
            super::AggregateShape::Set { .. }
            | super::AggregateShape::ExactIntSet { .. }
            | super::AggregateShape::ExactScalarSet { .. }
            | super::AggregateShape::SetBitmask { .. }
            | super::AggregateShape::FiniteSet
            | super::AggregateShape::BoundedSet { .. }
            | super::AggregateShape::Interval { .. }
            | super::AggregateShape::SymbolicDomain(_) => {
                self.lower_scalar_in_function_set_range_shape_branch(
                    loop_body,
                    value_val,
                    value_shape,
                    range_value,
                    range_shape,
                    range_ok_id,
                    else_target,
                    context,
                )?;
            }
            super::AggregateShape::RecordSet { fields } => {
                let range_ptr = self.i64_value_as_ptr(loop_body, range_value);
                self.lower_compact_record_value_in_record_set_branch(
                    loop_body,
                    source_ptr,
                    value_slot,
                    value_shape,
                    range_ptr,
                    fields,
                    range_ok_id,
                    else_target,
                )?;
            }
            super::AggregateShape::FunctionSet { domain, range } => match value_shape {
                Some(seq_shape @ super::AggregateShape::Sequence { .. }) => {
                    let range_ptr = self.i64_value_as_ptr(loop_body, range_value);
                    self.lower_compact_sequence_value_in_function_set_ptr_branch(
                        loop_body,
                        source_ptr,
                        value_slot,
                        seq_shape,
                        range_ptr,
                        *domain,
                        *range,
                        range_ok_id,
                        else_target,
                        context,
                    )?;
                }
                _ => {
                    let range_ptr = self.i64_value_as_ptr(loop_body, range_value);
                    self.lower_compact_function_value_in_function_set_ptr_branch(
                        loop_body,
                        source_ptr,
                        value_slot,
                        value_shape,
                        range_ptr,
                        *domain,
                        *range,
                        range_ok_id,
                        else_target,
                        context,
                    )?;
                }
            },
            super::AggregateShape::SeqSet { base } => {
                let range_value = if Self::lazy_domain_runtime_payload_is_compact_mask(&base) {
                    range_value
                } else {
                    self.i64_value_as_ptr(loop_body, range_value)
                };
                self.lower_compact_sequence_value_in_seq_set_branch(
                    loop_body,
                    source_ptr,
                    value_slot,
                    value_shape,
                    range_value,
                    *base,
                    range_ok_id,
                    else_target,
                    context,
                )?;
            }
            other => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: unsupported compact sequence function-set range shape: {other:?}"
                )));
            }
        }

        self.emit(
            range_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_inc_id,
                args: vec![],
            }),
        );

        let i_inc = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_inc, 1);
        let next_i = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_inc,
                rhs: one,
            },
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        Ok(())
    }

    fn lower_function_like_value_in_function_set_ptr_branch(
        &mut self,
        block_idx: usize,
        func_value: ValueId,
        elem_shape: Option<&super::AggregateShape>,
        funcset_ptr: ValueId,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
        then_target: BlockId,
        else_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        match elem_shape {
            Some(super::AggregateShape::Sequence { extent, element }) => self
                .lower_sequence_value_in_function_set_ptr_branch(
                    block_idx,
                    func_value,
                    *extent,
                    element.as_deref(),
                    funcset_ptr,
                    domain_shape,
                    range_shape,
                    then_target,
                    else_target,
                    context,
                ),
            Some(super::AggregateShape::Function { value, .. }) => self
                .lower_function_value_in_function_set_ptr_branch(
                    block_idx,
                    func_value,
                    value.as_deref(),
                    funcset_ptr,
                    domain_shape,
                    range_shape,
                    then_target,
                    else_target,
                    context,
                ),
            Some(super::AggregateShape::StateValue) | None => self
                .lower_function_value_in_function_set_ptr_branch(
                    block_idx,
                    func_value,
                    None,
                    funcset_ptr,
                    domain_shape,
                    range_shape,
                    then_target,
                    else_target,
                    context,
                ),
            Some(other) => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: function-set membership requires function or sequence element shape, got {other:?}"
            ))),
        }
    }

    fn lower_sequence_value_in_function_set_ptr_branch(
        &mut self,
        block_idx: usize,
        seq_value: ValueId,
        seq_extent: super::SequenceExtent,
        seq_element_shape: Option<&super::AggregateShape>,
        funcset_ptr: ValueId,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
        then_target: BlockId,
        else_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        domain_shape.validate_powerset_base(&format!("{context}: function-set domain"))?;
        range_shape.validate_function_set_range(&format!("{context}: function-set range"))?;

        let domain_len = domain_shape.tracked_len().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: function-set domain cardinality is not statically known: {domain_shape:?}"
            ))
        })?;
        let domain_is_sequence_domain =
            matches!(domain_shape, super::AggregateShape::Interval { lo: 1, .. });
        let seq_capacity = seq_extent.capacity();
        let compatible_domain = seq_extent
            .exact_count()
            .map_or(domain_len <= seq_capacity, |seq_len| seq_len == domain_len);
        if !domain_is_sequence_domain || !compatible_domain {
            self.emit(
                block_idx,
                InstrNode::new(Inst::Br {
                    target: else_target,
                    args: vec![],
                }),
            );
            return Ok(());
        }

        if matches!(
            range_shape,
            super::AggregateShape::Powerset { .. } | super::AggregateShape::NonEmptyPowerset { .. }
        ) {
            let Some(value_shape) = seq_element_shape else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: SUBSET range membership requires tracked set-valued sequence entries"
                )));
            };
            if !value_shape.is_finite_set_shape() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: SUBSET range membership requires set-valued sequence entries, got {value_shape:?}"
                )));
            }
        }

        let seq_ptr = self.i64_value_as_ptr(block_idx, seq_value);
        let range_value = self.load_at_offset(block_idx, funcset_ptr, 1);

        let len_ok_blk = self.new_aux_block("seq_funcset_member_len_ok");
        let loop_hdr = self.new_aux_block("seq_funcset_member_hdr");
        let loop_body = self.new_aux_block("seq_funcset_member_body");
        let loop_inc = self.new_aux_block("seq_funcset_member_inc");
        let len_ok_id = self.block_id_of(len_ok_blk);
        let loop_hdr_id = self.block_id_of(loop_hdr);
        let loop_body_id = self.block_id_of(loop_body);
        let loop_inc_id = self.block_id_of(loop_inc);

        let actual_len = self.load_at_offset(block_idx, seq_ptr, 0);
        let expected_len = self.emit_i64_const(block_idx, i64::from(domain_len));
        let len_matches = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: actual_len,
                rhs: expected_len,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: len_matches,
                then_target: len_ok_id,
                then_args: vec![],
                else_target,
                else_args: vec![],
            }),
        );

        let zero = self.emit_i64_const(len_ok_blk, 0);
        let idx_alloca = self.emit_with_result(
            len_ok_blk,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let expected_len = self.emit_i64_const(loop_hdr, i64::from(domain_len));
        let in_bounds = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: expected_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: loop_body_id,
                then_args: vec![],
                else_target: then_target,
                else_args: vec![],
            }),
        );

        let i_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_body, 1);
        let value_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_body,
                rhs: one,
            },
        );
        let value_val = self.load_at_dynamic_offset(loop_body, seq_ptr, value_slot);
        self.lower_scalar_in_function_set_range_shape_branch(
            loop_body,
            value_val,
            seq_element_shape,
            range_value,
            range_shape,
            loop_inc_id,
            else_target,
            "seq_funcset_member_range",
        )?;

        let i_inc = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_inc, 1);
        let next_i = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_inc,
                rhs: one,
            },
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        Ok(())
    }

    fn lower_function_value_in_function_set_ptr_branch(
        &mut self,
        block_idx: usize,
        func_value: ValueId,
        function_value_shape: Option<&super::AggregateShape>,
        funcset_ptr: ValueId,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
        then_target: BlockId,
        else_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        domain_shape.validate_powerset_base(&format!("{context}: function-set domain"))?;
        range_shape.validate_function_set_range(&format!("{context}: function-set range"))?;

        let value_shape = function_value_shape.cloned().map(Box::new).or_else(|| {
            super::AggregateShape::function_value_shape_from_range(&range_shape).map(Box::new)
        });
        let domain_len = domain_shape.tracked_len().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: function-set domain cardinality is not statically known: {domain_shape:?}"
            ))
        })?;
        if matches!(
            range_shape,
            super::AggregateShape::Powerset { .. } | super::AggregateShape::NonEmptyPowerset { .. }
        ) {
            let Some(value_shape) = value_shape.as_deref() else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: SUBSET range membership requires tracked set-valued function shape"
                )));
            };
            if !value_shape.is_finite_set_shape() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: SUBSET range membership requires set-valued function entries, got {value_shape:?}"
                )));
            }
        }

        let func_ptr = self.i64_value_as_ptr(block_idx, func_value);
        let domain_value = self.load_at_offset(block_idx, funcset_ptr, 0);
        let range_value = self.load_at_offset(block_idx, funcset_ptr, 1);
        let domain_ptr = self.i64_value_as_ptr(block_idx, domain_value);

        let len_ok_blk = self.new_aux_block("nested_funcset_member_len_ok");
        let loop_hdr = self.new_aux_block("nested_funcset_member_hdr");
        let loop_body = self.new_aux_block("nested_funcset_member_body");
        let domain_ok_blk = self.new_aux_block("nested_funcset_member_domain_ok");
        let range_ok_blk = self.new_aux_block("nested_funcset_member_range_ok");
        let loop_inc = self.new_aux_block("nested_funcset_member_inc");

        let len_ok_id = self.block_id_of(len_ok_blk);
        let loop_hdr_id = self.block_id_of(loop_hdr);
        let loop_body_id = self.block_id_of(loop_body);
        let domain_ok_id = self.block_id_of(domain_ok_blk);
        let range_ok_id = self.block_id_of(range_ok_blk);
        let loop_inc_id = self.block_id_of(loop_inc);

        let actual_len = self.load_at_offset(block_idx, func_ptr, 0);
        let expected_len = self.emit_i64_const(block_idx, i64::from(domain_len));
        let len_matches = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: actual_len,
                rhs: expected_len,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: len_matches,
                then_target: len_ok_id,
                then_args: vec![],
                else_target,
                else_args: vec![],
            }),
        );

        let zero = self.emit_i64_const(len_ok_blk, 0);
        let idx_alloca = self.emit_with_result(
            len_ok_blk,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let expected_len = self.emit_i64_const(loop_hdr, i64::from(domain_len));
        let in_bounds = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: expected_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: loop_body_id,
                then_args: vec![],
                else_target: then_target,
                else_args: vec![],
            }),
        );

        let i_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_body, 1);
        let two = self.emit_i64_const(loop_body, 2);
        let pair_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_body,
                rhs: two,
            },
        );
        let key_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: pair_offset,
                rhs: one,
            },
        );
        let value_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: pair_offset,
                rhs: two,
            },
        );
        let key_val = self.load_at_dynamic_offset(loop_body, func_ptr, key_slot);
        let value_val = self.load_at_dynamic_offset(loop_body, func_ptr, value_slot);
        self.lower_value_in_set_ptr_branch(
            loop_body,
            key_val,
            domain_ptr,
            domain_ok_id,
            else_target,
            "nested_funcset_member_domain",
        );

        match range_shape {
            super::AggregateShape::Powerset { base } => {
                base.validate_powerset_base("nested function-set SUBSET range")?;
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_scalar_in_function_set_range_shape_branch(
                        domain_ok_blk,
                        value_val,
                        value_shape.as_deref(),
                        range_value,
                        super::AggregateShape::Powerset { base },
                        range_ok_id,
                        else_target,
                        "nested_funcset_member_range_subset",
                    )?;
                } else {
                    let value_ptr = self.i64_value_as_ptr(domain_ok_blk, value_val);
                    let range_ptr = self.i64_value_as_ptr(domain_ok_blk, range_value);
                    self.lower_subseteq_ptr_branch(
                        domain_ok_blk,
                        value_ptr,
                        range_ptr,
                        range_ok_id,
                        else_target,
                        "nested_funcset_member_range_subset",
                    );
                }
            }
            // `(SUBSET S) \ {{}}` nested range: subset half as `Powerset`, plus
            // a per-entry non-empty guard.
            super::AggregateShape::NonEmptyPowerset { base } => {
                base.validate_powerset_base("nested function-set non-empty SUBSET range")?;
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_scalar_in_function_set_range_shape_branch(
                        domain_ok_blk,
                        value_val,
                        value_shape.as_deref(),
                        range_value,
                        super::AggregateShape::NonEmptyPowerset { base },
                        range_ok_id,
                        else_target,
                        "nested_funcset_member_range_nonempty_subset",
                    )?;
                } else {
                    let guard_blk = self.new_aux_block("nested_funcset_nonempty_powerset_guard");
                    let guard_id = self.block_id_of(guard_blk);
                    let value_ptr = self.i64_value_as_ptr(domain_ok_blk, value_val);
                    let range_ptr = self.i64_value_as_ptr(domain_ok_blk, range_value);
                    self.lower_subseteq_ptr_branch(
                        domain_ok_blk,
                        value_ptr,
                        range_ptr,
                        guard_id,
                        else_target,
                        "nested_funcset_member_range_nonempty_subset",
                    );
                    self.branch_on_set_nonempty(
                        guard_blk,
                        value_val,
                        value_shape.as_deref(),
                        range_ok_id,
                        else_target,
                        "nested_funcset_member_range_nonempty_subset",
                    )?;
                }
            }
            super::AggregateShape::Set { .. }
            | super::AggregateShape::ExactIntSet { .. }
            | super::AggregateShape::ExactScalarSet { .. }
            | super::AggregateShape::SetBitmask { .. }
            | super::AggregateShape::FiniteSet
            | super::AggregateShape::BoundedSet { .. }
            | super::AggregateShape::Interval { .. } => {
                self.lower_scalar_in_function_set_range_shape_branch(
                    domain_ok_blk,
                    value_val,
                    value_shape.as_deref(),
                    range_value,
                    range_shape,
                    range_ok_id,
                    else_target,
                    "nested_funcset_member_range",
                )?;
            }
            super::AggregateShape::RecordSet { fields } => {
                let range_ptr = self.i64_value_as_ptr(domain_ok_blk, range_value);
                self.lower_record_value_in_record_set_ptr_branch(
                    domain_ok_blk,
                    value_val,
                    range_ptr,
                    fields,
                    range_ok_id,
                    else_target,
                )?;
            }
            super::AggregateShape::SymbolicDomain(domain) => {
                let member = self.emit_symbolic_domain_membership_i64(
                    domain_ok_blk,
                    value_val,
                    value_shape.as_deref(),
                    domain,
                )?;
                let zero = self.emit_i64_const(domain_ok_blk, 0);
                let is_member = self.emit_with_result(
                    domain_ok_blk,
                    Inst::ICmp {
                        op: ICmpOp::Ne,
                        ty: Ty::I64,
                        lhs: member,
                        rhs: zero,
                    },
                );
                self.emit(
                    domain_ok_blk,
                    InstrNode::new(Inst::CondBr {
                        cond: is_member,
                        then_target: range_ok_id,
                        then_args: vec![],
                        else_target,
                        else_args: vec![],
                    }),
                );
            }
            super::AggregateShape::FunctionSet { domain, range } => {
                let range_ptr = self.i64_value_as_ptr(domain_ok_blk, range_value);
                self.lower_function_like_value_in_function_set_ptr_branch(
                    domain_ok_blk,
                    value_val,
                    value_shape.as_deref(),
                    range_ptr,
                    *domain,
                    *range,
                    range_ok_id,
                    else_target,
                    "nested_funcset_member_range_function",
                )?;
            }
            super::AggregateShape::SeqSet { base } => {
                let seq_element_shape = match value_shape.as_deref() {
                    Some(super::AggregateShape::Sequence { element, .. }) => element.as_deref(),
                    _ => None,
                };
                let range_value = if Self::lazy_domain_runtime_payload_is_compact_mask(&base) {
                    range_value
                } else {
                    self.i64_value_as_ptr(domain_ok_blk, range_value)
                };
                self.lower_seq_value_in_seq_set_ptr_branch(
                    domain_ok_blk,
                    value_val,
                    seq_element_shape,
                    range_value,
                    *base,
                    range_ok_id,
                    else_target,
                    "nested_funcset_member_range_seq",
                )?;
            }
            other => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: unsupported function-set range shape: {other:?}"
                )));
            }
        }

        self.emit(
            range_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_inc_id,
                args: vec![],
            }),
        );

        let i_inc = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_inc, 1);
        let next_i = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_inc,
                rhs: one,
            },
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn lower_compact_state_function_set_membership(
        &mut self,
        block_idx: usize,
        rd: u8,
        _elem_reg: u8,
        funcset_reg: u8,
        source_slot: super::CompactStateSlot,
        len: u32,
        function_domain_lo: Option<i64>,
        function_domain: Option<&super::CompactFunctionDomain>,
        value_shape: Option<&super::AggregateShape>,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
    ) -> Result<Option<usize>, TrustIrError> {
        let domain_len = domain_shape.tracked_len().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "SetIn: compact function-set domain cardinality is not statically known: {domain_shape:?}"
            ))
        })?;
        if len != domain_len {
            self.store_reg_imm(block_idx, rd, 0)?;
            return Ok(Some(block_idx));
        }
        // Key-set equality gate (soundness amendment H2): this path iterates
        // only the function's RANGE slots, so cardinality alone does not
        // prove `DOMAIN f = D`. Prove the actual keys equal the funcset
        // domain's element set at compile time, or fail closed.
        super::function_keys_match_funcset_domain(
            len,
            function_domain_lo,
            function_domain,
            &domain_shape,
            "SetIn: compact function-set domain",
        )?;
        range_shape.validate_function_set_range("SetIn: compact function-set range")?;
        if matches!(
            range_shape,
            super::AggregateShape::Powerset { .. } | super::AggregateShape::NonEmptyPowerset { .. }
        ) {
            let Some(value_shape) = value_shape else {
                return Err(TrustIrError::UnsupportedOpcode(
                    "SetIn: compact function-set SUBSET range requires tracked set-valued function entries"
                        .to_owned(),
                ));
            };
            if !matches!(value_shape, super::AggregateShape::SetBitmask { .. }) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: compact function-set SUBSET range requires SetBitmask entries, got {value_shape:?}"
                )));
            }
            if value_shape.set_bitmask_universe().is_none() {
                return Err(TrustIrError::UnsupportedOpcode(
                    "SetIn: compact function-set SUBSET range requires exact universe metadata"
                        .to_owned(),
                ));
            }
        }
        let Some(value_stride) = value_shape.and_then(super::AggregateShape::compact_slot_count)
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "SetIn: compact function-set range requires fixed-width value layout, got {value_shape:?}"
            )));
        };

        let funcset_ptr = self.load_reg_as_ptr(block_idx, funcset_reg)?;
        let range_value = self.load_at_offset(block_idx, funcset_ptr, 1);

        let false_blk = self.new_aux_block("compact_funcset_member_false");
        let true_blk = self.new_aux_block("compact_funcset_member_true");
        let merge_blk = self.new_aux_block("compact_funcset_member_merge");
        let loop_hdr = self.new_aux_block("compact_funcset_member_hdr");
        let loop_body = self.new_aux_block("compact_funcset_member_body");
        let range_ok_blk = self.new_aux_block("compact_funcset_member_range_ok");
        let loop_inc = self.new_aux_block("compact_funcset_member_inc");

        let false_id = self.block_id_of(false_blk);
        let true_id = self.block_id_of(true_blk);
        let merge_id = self.block_id_of(merge_blk);
        let loop_hdr_id = self.block_id_of(loop_hdr);
        let loop_body_id = self.block_id_of(loop_body);
        let range_ok_id = self.block_id_of(range_ok_blk);
        let loop_inc_id = self.block_id_of(loop_inc);

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
        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let expected_len = self.emit_i64_const(loop_hdr, i64::from(domain_len));
        let in_bounds = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: expected_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: loop_body_id,
                then_args: vec![],
                else_target: true_id,
                else_args: vec![],
            }),
        );

        let i_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let stride = self.emit_i64_const(loop_body, i64::from(value_stride));
        let base = self.emit_i64_const(loop_body, i64::from(source_slot.offset));
        let offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_body,
                rhs: stride,
            },
        );
        let value_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: base,
                rhs: offset,
            },
        );
        let value_val = self.load_at_dynamic_offset(loop_body, source_slot.source_ptr, value_slot);

        match range_shape {
            super::AggregateShape::Powerset { base } => {
                let Some(value_shape @ super::AggregateShape::SetBitmask { .. }) = value_shape
                else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "SetIn: compact function-set SUBSET range requires SetBitmask entries"
                            .to_owned(),
                    ));
                };
                let Some((universe_len, universe)) = value_shape.set_bitmask_universe() else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "SetIn: compact function-set SUBSET range requires exact universe metadata"
                            .to_owned(),
                    ));
                };
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_compact_bitmask_runtime_powerset_mask_branch(
                        loop_body,
                        value_val,
                        range_value,
                        &base,
                        universe_len,
                        &universe,
                        range_ok_id,
                        false_id,
                        "compact_funcset_member_range_subset",
                    )?;
                } else {
                    self.lower_compact_bitmask_powerset_branch(
                        loop_body,
                        value_val,
                        &base,
                        universe_len,
                        &universe,
                        range_ok_id,
                        false_id,
                        "compact_funcset_member_range_subset",
                    )?;
                }
            }
            // `(SUBSET S) \ {{}}` compact function-set range: subset half as
            // `Powerset`, plus a per-entry non-empty guard (bitmask != 0).
            super::AggregateShape::NonEmptyPowerset { base } => {
                let Some(value_shape @ super::AggregateShape::SetBitmask { .. }) = value_shape
                else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "SetIn: compact function-set non-empty SUBSET range requires SetBitmask entries"
                            .to_owned(),
                    ));
                };
                let Some((universe_len, universe)) = value_shape.set_bitmask_universe() else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "SetIn: compact function-set non-empty SUBSET range requires exact universe metadata"
                            .to_owned(),
                    ));
                };
                let guard_blk = self.new_aux_block("compact_funcset_nonempty_powerset_guard");
                let guard_id = self.block_id_of(guard_blk);
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_compact_bitmask_runtime_powerset_mask_branch(
                        loop_body,
                        value_val,
                        range_value,
                        &base,
                        universe_len,
                        &universe,
                        guard_id,
                        false_id,
                        "compact_funcset_member_range_nonempty_subset",
                    )?;
                } else {
                    self.lower_compact_bitmask_powerset_branch(
                        loop_body,
                        value_val,
                        &base,
                        universe_len,
                        &universe,
                        guard_id,
                        false_id,
                        "compact_funcset_member_range_nonempty_subset",
                    )?;
                }
                self.branch_on_set_nonempty(
                    guard_blk,
                    value_val,
                    Some(value_shape),
                    range_ok_id,
                    false_id,
                    "compact_funcset_member_range_nonempty_subset",
                )?;
            }
            // Lazy union range (lever L1): static-only per-entry membership.
            // `range_value` (the funcset slot-1 payload) is dead in this arm —
            // it mirrors the LazyUnion register's inert placeholder and must
            // never be consumed (soundness amendment H1).
            super::AggregateShape::LazyUnion { left, right } => {
                self.lower_entry_in_lazy_union_range_branch(
                    loop_body,
                    value_val,
                    value_shape,
                    &left,
                    &right,
                    range_ok_id,
                    false_id,
                    "compact_funcset_member_range_lazy_union",
                )?;
            }
            super::AggregateShape::Set { .. }
            | super::AggregateShape::ExactIntSet { .. }
            | super::AggregateShape::ExactScalarSet { .. }
            | super::AggregateShape::SetBitmask { .. }
            | super::AggregateShape::FiniteSet
            | super::AggregateShape::BoundedSet { .. }
            | super::AggregateShape::Interval { .. }
            | super::AggregateShape::SymbolicDomain(_) => {
                self.lower_scalar_in_function_set_range_shape_branch(
                    loop_body,
                    value_val,
                    value_shape,
                    range_value,
                    range_shape,
                    range_ok_id,
                    false_id,
                    "compact_funcset_member_range",
                )?;
            }
            super::AggregateShape::RecordSet { fields } => {
                let range_ptr = self.i64_value_as_ptr(loop_body, range_value);
                self.lower_compact_record_value_in_record_set_branch(
                    loop_body,
                    source_slot.source_ptr,
                    value_slot,
                    value_shape,
                    range_ptr,
                    fields,
                    range_ok_id,
                    false_id,
                )?;
            }
            super::AggregateShape::FunctionSet { domain, range } => {
                let range_ptr = self.i64_value_as_ptr(loop_body, range_value);
                self.lower_compact_function_value_in_function_set_ptr_branch(
                    loop_body,
                    source_slot.source_ptr,
                    value_slot,
                    value_shape,
                    range_ptr,
                    *domain,
                    *range,
                    range_ok_id,
                    false_id,
                    "compact_funcset_member_range_function",
                )?;
            }
            super::AggregateShape::SeqSet { base } => {
                let range_value = if Self::lazy_domain_runtime_payload_is_compact_mask(&base) {
                    range_value
                } else {
                    self.i64_value_as_ptr(loop_body, range_value)
                };
                self.lower_compact_sequence_value_in_seq_set_branch(
                    loop_body,
                    source_slot.source_ptr,
                    value_slot,
                    value_shape,
                    range_value,
                    *base,
                    range_ok_id,
                    false_id,
                    "compact_funcset_member_range_seq",
                )?;
            }
            other => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: unsupported compact function-set range shape: {other:?}"
                )));
            }
        }

        self.emit(
            range_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_inc_id,
                args: vec![],
            }),
        );

        let i_inc = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_inc, 1);
        let next_i = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_inc,
                rhs: one,
            },
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        self.store_reg_imm(true_blk, rd, 1)?;
        self.emit(
            true_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );
        self.store_reg_imm(false_blk, rd, 0)?;
        self.emit(
            false_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge_blk))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_compact_function_value_in_function_set_ptr_branch(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        func_base_slot: ValueId,
        function_shape: Option<&super::AggregateShape>,
        funcset_ptr: ValueId,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
        then_target: BlockId,
        else_target: BlockId,
        context: &str,
    ) -> Result<(), TrustIrError> {
        domain_shape.validate_powerset_base(&format!("{context}: function-set domain"))?;
        range_shape.validate_function_set_range(&format!("{context}: function-set range"))?;

        let domain_len = domain_shape.tracked_len().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: function-set domain cardinality is not statically known: {domain_shape:?}"
            ))
        })?;
        let Some(super::AggregateShape::Function { len, value, .. }) = function_shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact nested function membership requires tracked function shape, got {function_shape:?}"
            )));
        };
        if *len != domain_len {
            self.emit(
                block_idx,
                InstrNode::new(Inst::Br {
                    target: else_target,
                    args: vec![],
                }),
            );
            return Ok(());
        }
        let value_shape = value.as_deref();
        let Some(value_stride) = value_shape.and_then(super::AggregateShape::compact_slot_count)
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact nested function requires fixed-width value layout, got {value_shape:?}"
            )));
        };

        let range_value = self.load_at_offset(block_idx, funcset_ptr, 1);

        let loop_hdr = self.new_aux_block(&format!("{context}_hdr"));
        let loop_body = self.new_aux_block(&format!("{context}_body"));
        let range_ok_blk = self.new_aux_block(&format!("{context}_range_ok"));
        let loop_inc = self.new_aux_block(&format!("{context}_inc"));

        let loop_hdr_id = self.block_id_of(loop_hdr);
        let loop_body_id = self.block_id_of(loop_body);
        let range_ok_id = self.block_id_of(range_ok_blk);
        let loop_inc_id = self.block_id_of(loop_inc);

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
        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let expected_len = self.emit_i64_const(loop_hdr, i64::from(domain_len));
        let in_bounds = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: expected_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: loop_body_id,
                then_args: vec![],
                else_target: then_target,
                else_args: vec![],
            }),
        );

        let i_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let stride = self.emit_i64_const(loop_body, i64::from(value_stride));
        let offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_body,
                rhs: stride,
            },
        );
        let value_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: func_base_slot,
                rhs: offset,
            },
        );
        let value_val = self.load_at_dynamic_offset(loop_body, source_ptr, value_slot);

        match range_shape {
            super::AggregateShape::Powerset { base } => {
                let Some(value_shape @ super::AggregateShape::SetBitmask { .. }) = value_shape
                else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SUBSET range requires SetBitmask entries, got {value_shape:?}"
                    )));
                };
                let Some((universe_len, universe)) = value_shape.set_bitmask_universe() else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SUBSET range requires exact universe metadata"
                    )));
                };
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_compact_bitmask_runtime_powerset_mask_branch(
                        loop_body,
                        value_val,
                        range_value,
                        &base,
                        universe_len,
                        &universe,
                        range_ok_id,
                        else_target,
                        context,
                    )?;
                } else {
                    self.lower_compact_bitmask_powerset_branch(
                        loop_body,
                        value_val,
                        &base,
                        universe_len,
                        &universe,
                        range_ok_id,
                        else_target,
                        context,
                    )?;
                }
            }
            // `(SUBSET S) \ {{}}` compact range: subset half as `Powerset`,
            // plus a per-entry non-empty guard (bitmask != 0).
            super::AggregateShape::NonEmptyPowerset { base } => {
                let Some(value_shape @ super::AggregateShape::SetBitmask { .. }) = value_shape
                else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact non-empty SUBSET range requires SetBitmask entries, got {value_shape:?}"
                    )));
                };
                let Some((universe_len, universe)) = value_shape.set_bitmask_universe() else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact non-empty SUBSET range requires exact universe metadata"
                    )));
                };
                let guard_blk = self.new_aux_block("compact_nonempty_powerset_range_guard");
                let guard_id = self.block_id_of(guard_blk);
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_compact_bitmask_runtime_powerset_mask_branch(
                        loop_body,
                        value_val,
                        range_value,
                        &base,
                        universe_len,
                        &universe,
                        guard_id,
                        else_target,
                        context,
                    )?;
                } else {
                    self.lower_compact_bitmask_powerset_branch(
                        loop_body,
                        value_val,
                        &base,
                        universe_len,
                        &universe,
                        guard_id,
                        else_target,
                        context,
                    )?;
                }
                self.branch_on_set_nonempty(
                    guard_blk,
                    value_val,
                    Some(value_shape),
                    range_ok_id,
                    else_target,
                    context,
                )?;
            }
            super::AggregateShape::Set { .. }
            | super::AggregateShape::ExactIntSet { .. }
            | super::AggregateShape::ExactScalarSet { .. }
            | super::AggregateShape::SetBitmask { .. }
            | super::AggregateShape::FiniteSet
            | super::AggregateShape::BoundedSet { .. }
            | super::AggregateShape::Interval { .. }
            | super::AggregateShape::SymbolicDomain(_) => {
                self.lower_scalar_in_function_set_range_shape_branch(
                    loop_body,
                    value_val,
                    value_shape,
                    range_value,
                    range_shape,
                    range_ok_id,
                    else_target,
                    context,
                )?;
            }
            super::AggregateShape::RecordSet { fields } => {
                let range_ptr = self.i64_value_as_ptr(loop_body, range_value);
                self.lower_compact_record_value_in_record_set_branch(
                    loop_body,
                    source_ptr,
                    value_slot,
                    value_shape,
                    range_ptr,
                    fields,
                    range_ok_id,
                    else_target,
                )?;
            }
            super::AggregateShape::FunctionSet { domain, range } => {
                let range_ptr = self.i64_value_as_ptr(loop_body, range_value);
                self.lower_compact_function_value_in_function_set_ptr_branch(
                    loop_body,
                    source_ptr,
                    value_slot,
                    value_shape,
                    range_ptr,
                    *domain,
                    *range,
                    range_ok_id,
                    else_target,
                    context,
                )?;
            }
            super::AggregateShape::SeqSet { base } => {
                let range_value = if Self::lazy_domain_runtime_payload_is_compact_mask(&base) {
                    range_value
                } else {
                    self.i64_value_as_ptr(loop_body, range_value)
                };
                self.lower_compact_sequence_value_in_seq_set_branch(
                    loop_body,
                    source_ptr,
                    value_slot,
                    value_shape,
                    range_value,
                    *base,
                    range_ok_id,
                    else_target,
                    context,
                )?;
            }
            other => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: unsupported compact nested function-set range shape: {other:?}"
                )));
            }
        }

        self.emit(
            range_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_inc_id,
                args: vec![],
            }),
        );
        let i_inc = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_inc, 1);
        let next_i = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_inc,
                rhs: one,
            },
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_compact_sequence_value_in_seq_set_branch(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        seq_base_slot: ValueId,
        seq_shape: Option<&super::AggregateShape>,
        base_value: ValueId,
        base_shape: super::AggregateShape,
        success_target: BlockId,
        failure_target: BlockId,
        prefix: &str,
    ) -> Result<(), TrustIrError> {
        base_shape.validate_seq_base(prefix)?;
        let Some(super::AggregateShape::Sequence { extent, element }) = seq_shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{prefix}: compact Seq membership requires tracked sequence shape, got {seq_shape:?}"
            )));
        };
        let element_shape = element.as_deref();
        let Some(element_stride) =
            element_shape.and_then(super::AggregateShape::compact_slot_count)
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{prefix}: compact Seq membership requires fixed-width element layout, got {element_shape:?}"
            )));
        };

        let len_ok_blk = self.guard_compact_sequence_dynamic_len_in_bounds(
            block_idx,
            source_ptr,
            seq_base_slot,
            extent.capacity(),
            &format!("{prefix}_seq_len"),
        );
        let loop_hdr = self.new_aux_block(&format!("{prefix}_seq_hdr"));
        let loop_body = self.new_aux_block(&format!("{prefix}_seq_body"));
        let elem_ok_blk = self.new_aux_block(&format!("{prefix}_seq_elem_ok"));
        let loop_inc = self.new_aux_block(&format!("{prefix}_seq_inc"));
        let loop_hdr_id = self.block_id_of(loop_hdr);
        let loop_body_id = self.block_id_of(loop_body);
        let elem_ok_id = self.block_id_of(elem_ok_blk);
        let loop_inc_id = self.block_id_of(loop_inc);

        let idx_alloca = self.emit_with_result(
            len_ok_blk,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        let zero = self.emit_i64_const(len_ok_blk, 0);
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        let idx = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let seq_len = self.load_at_dynamic_offset(loop_hdr, source_ptr, seq_base_slot);
        let in_bounds = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: idx,
                rhs: seq_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: loop_body_id,
                then_args: vec![],
                else_target: success_target,
                else_args: vec![],
            }),
        );

        let idx_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_body, 1);
        let stride = self.emit_i64_const(loop_body, i64::from(element_stride));
        let elem_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: idx_body,
                rhs: stride,
            },
        );
        let first_elem_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: seq_base_slot,
                rhs: one,
            },
        );
        let elem_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: first_elem_slot,
                rhs: elem_offset,
            },
        );
        let elem_val = self.load_at_dynamic_offset(loop_body, source_ptr, elem_slot);

        match base_shape {
            super::AggregateShape::SetBitmask {
                universe_len,
                universe,
            } => {
                self.lower_scalar_in_set_bitmask_shape_branch(
                    loop_body,
                    elem_val,
                    element_shape,
                    base_value,
                    universe_len,
                    &universe,
                    elem_ok_id,
                    failure_target,
                    prefix,
                )?;
            }
            super::AggregateShape::RecordSet { fields } => {
                self.lower_compact_record_value_in_record_set_branch(
                    loop_body,
                    source_ptr,
                    elem_slot,
                    element_shape,
                    base_value,
                    fields,
                    elem_ok_id,
                    failure_target,
                )?;
            }
            super::AggregateShape::Set { len, element } => {
                let base_element_is_record = matches!(
                    element.as_deref(),
                    Some(super::AggregateShape::Record { .. })
                );
                let seq_element_is_record =
                    matches!(element_shape, Some(super::AggregateShape::Record { .. }));
                if base_element_is_record && seq_element_is_record {
                    self.lower_compact_record_materialized_set_membership_branch(
                        loop_body,
                        source_ptr,
                        elem_slot,
                        element_shape.ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "{prefix}: compact Seq finite-record-set membership requires tracked record element shape"
                            ))
                        })?,
                        base_value,
                        len,
                        element.as_deref(),
                        elem_ok_id,
                        failure_target,
                    )?;
                } else if base_element_is_record || seq_element_is_record {
                    self.emit(
                        loop_body,
                        InstrNode::new(Inst::Br {
                            target: failure_target,
                            args: vec![],
                        }),
                    );
                } else {
                    self.lower_value_in_domain_ptr_branch(
                        loop_body,
                        elem_val,
                        element_shape,
                        base_value,
                        super::AggregateShape::Set { len, element },
                        elem_ok_id,
                        failure_target,
                        prefix,
                    )?;
                }
            }
            super::AggregateShape::ExactIntSet { .. }
            | super::AggregateShape::ExactScalarSet { .. }
            | super::AggregateShape::FiniteSet
            | super::AggregateShape::BoundedSet { .. }
            | super::AggregateShape::Interval { .. }
            | super::AggregateShape::SymbolicDomain(_) => {
                self.lower_value_in_domain_ptr_branch(
                    loop_body,
                    elem_val,
                    element_shape,
                    base_value,
                    base_shape,
                    elem_ok_id,
                    failure_target,
                    prefix,
                )?;
            }
            super::AggregateShape::Powerset { base } => {
                let (universe_len, universe) =
                    Self::compact_powerset_mask_universe_for_value_shape(element_shape, prefix)?;
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_compact_bitmask_runtime_powerset_mask_branch(
                        loop_body,
                        elem_val,
                        base_value,
                        &base,
                        universe_len,
                        &universe,
                        elem_ok_id,
                        failure_target,
                        prefix,
                    )?;
                } else {
                    self.lower_compact_bitmask_powerset_branch(
                        loop_body,
                        elem_val,
                        &base,
                        universe_len,
                        &universe,
                        elem_ok_id,
                        failure_target,
                        prefix,
                    )?;
                }
            }
            other => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{prefix}: unsupported compact sequence base shape: {other:?}"
                )));
            }
        }

        self.emit(
            elem_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_inc_id,
                args: vec![],
            }),
        );
        let idx_inc = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_inc, 1);
        let next_idx = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: idx_inc,
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
                target: loop_hdr_id,
                args: vec![],
            }),
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_compact_record_value_in_record_set_branch(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        record_base_slot: ValueId,
        record_shape: Option<&super::AggregateShape>,
        record_set_ptr: ValueId,
        fields: Vec<(tla_core::NameId, super::AggregateShape)>,
        success_target: BlockId,
        failure_target: BlockId,
    ) -> Result<(), TrustIrError> {
        let Some(record_shape @ super::AggregateShape::Record { .. }) = record_shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "compact record-set membership requires tracked record shape, got {record_shape:?}"
            )));
        };
        if fields.is_empty() {
            self.emit(
                block_idx,
                InstrNode::new(Inst::Br {
                    target: success_target,
                    args: vec![],
                }),
            );
            return Ok(());
        }

        let mut result: Option<ValueId> = None;
        for (domain_slot, (field_name, field_set_shape)) in fields.iter().enumerate() {
            let Some((field_idx, field_shape)) = record_shape.compact_record_field(*field_name)
            else {
                self.emit(
                    block_idx,
                    InstrNode::new(Inst::Br {
                        target: failure_target,
                        args: vec![],
                    }),
                );
                return Ok(());
            };
            let field_offset = self.emit_i64_const(block_idx, i64::from(field_idx));
            let field_slot = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: record_base_slot,
                    rhs: field_offset,
                },
            );
            let field_val = self.load_at_dynamic_offset(block_idx, source_ptr, field_slot);
            let field_set_shape =
                Self::materialized_domain_shape_for_pointer(field_set_shape.clone());
            let field_ok = match &field_set_shape {
                super::AggregateShape::Interval { lo, hi } => {
                    self.emit_interval_membership_i64(block_idx, field_val, *lo, *hi)
                }
                super::AggregateShape::Set { len, .. } => {
                    let domain_slot =
                        u32::try_from(domain_slot).expect("record set slot index must fit in u32");
                    let domain_value = self.load_at_offset(block_idx, record_set_ptr, domain_slot);
                    let domain_ptr = self.i64_value_as_ptr(block_idx, domain_value);
                    self.emit_finite_set_membership_i64(block_idx, field_val, domain_ptr, *len)
                }
                super::AggregateShape::SymbolicDomain(domain) => self
                    .emit_symbolic_domain_membership_i64(
                        block_idx,
                        field_val,
                        field_shape.as_ref(),
                        *domain,
                    )?,
                other => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compact record-set field domain is not directly lowerable: {other:?}"
                    )));
                }
            };
            result = Some(match result {
                None => field_ok,
                Some(prev) => self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: prev,
                        rhs: field_ok,
                    },
                ),
            });
        }

        self.branch_on_i64_truth(
            block_idx,
            result.expect("non-empty record-set fields must produce a result"),
            success_target,
            failure_target,
        );
        Ok(())
    }

    fn lower_function_set_membership(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        funcset_reg: u8,
        domain_shape: super::AggregateShape,
        range_shape: super::AggregateShape,
    ) -> Result<Option<usize>, TrustIrError> {
        domain_shape.validate_powerset_base("SetIn: function-set domain")?;
        range_shape.validate_function_set_range("SetIn: function-set range")?;

        let elem_shape = self.aggregate_shapes.get(&elem_reg).cloned();
        if let (
            Some(base_slot),
            Some(super::AggregateShape::Function {
                len,
                value,
                domain_lo,
                domain,
            }),
        ) = (
            self.compact_state_slot_for_use(block_idx, elem_reg)?,
            elem_shape.clone(),
        ) {
            return self.lower_compact_state_function_set_membership(
                block_idx,
                rd,
                elem_reg,
                funcset_reg,
                base_slot,
                len,
                domain_lo,
                domain.as_ref(),
                value.as_deref(),
                domain_shape,
                range_shape,
            );
        }
        if let (Some(source_slot), Some(seq_shape @ super::AggregateShape::Sequence { .. })) = (
            self.compact_state_slot_for_use(block_idx, elem_reg)?,
            elem_shape.as_ref(),
        ) {
            return self.lower_compact_sequence_function_set_membership_to_reg(
                block_idx,
                rd,
                source_slot,
                seq_shape,
                funcset_reg,
                domain_shape,
                range_shape,
            );
        }
        if matches!(elem_shape, Some(super::AggregateShape::Sequence { .. })) {
            return self.lower_function_like_set_membership_to_reg(
                block_idx,
                rd,
                elem_reg,
                funcset_reg,
                elem_shape.as_ref(),
                domain_shape,
                range_shape,
            );
        }
        let value_shape = match elem_shape {
            Some(super::AggregateShape::Function {
                len,
                value,
                domain_lo,
                domain,
            }) => {
                let domain_len = domain_shape.tracked_len().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "SetIn: function-set domain cardinality is not statically known: {domain_shape:?}"
                    ))
                })?;
                if len != domain_len {
                    self.store_reg_imm(block_idx, rd, 0)?;
                    return Ok(Some(block_idx));
                }
                // Key-set equality gate (soundness amendment H2). This path
                // DOES verify each key against the domain at runtime (the
                // `funcset_member_domain` branch below), so statically-unknown
                // keys remain sound here — but when the keys ARE statically
                // known, prove the equality at compile time and fail closed on
                // mismatch rather than rely on runtime reachability.
                if domain_lo.is_some() || domain.is_some() {
                    super::function_keys_match_funcset_domain(
                        len,
                        domain_lo,
                        domain.as_ref(),
                        &domain_shape,
                        "SetIn: function-set domain",
                    )?;
                }
                value
            }
            Some(super::AggregateShape::StateValue) => {
                let inferred = super::AggregateShape::function_from_function_set_domains(
                    &domain_shape,
                    &range_shape,
                )?;
                let super::AggregateShape::Function { value, .. } = inferred else {
                    unreachable!("function_from_function_set_domains must return Function");
                };
                value
            }
            Some(shape) => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: function-set membership requires tracked function element shape, got {shape:?}"
                )));
            }
            None => {
                return Err(TrustIrError::UnsupportedOpcode(
                    "SetIn: function-set membership requires tracked function element shape"
                        .to_owned(),
                ));
            }
        };
        let domain_len = domain_shape.tracked_len().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "SetIn: function-set domain cardinality is not statically known: {domain_shape:?}"
            ))
        })?;
        if matches!(
            range_shape,
            super::AggregateShape::Powerset { .. } | super::AggregateShape::NonEmptyPowerset { .. }
        ) {
            let Some(value_shape) = value_shape.as_deref() else {
                return Err(TrustIrError::UnsupportedOpcode(
                    "SetIn: function-set range SUBSET membership requires tracked set-valued function shape"
                        .to_owned(),
                ));
            };
            if !value_shape.is_finite_set_shape() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: function-set range SUBSET membership requires set-valued function entries, got {value_shape:?}"
                )));
            }
        }

        let func_ptr = self.load_reg_as_ptr(block_idx, elem_reg)?;
        let funcset_ptr = self.load_reg_as_ptr(block_idx, funcset_reg)?;
        let domain_value = self.load_at_offset(block_idx, funcset_ptr, 0);
        let range_value = self.load_at_offset(block_idx, funcset_ptr, 1);
        let domain_ptr = self.i64_value_as_ptr(block_idx, domain_value);

        let false_blk = self.new_aux_block("funcset_member_false");
        let true_blk = self.new_aux_block("funcset_member_true");
        let merge_blk = self.new_aux_block("funcset_member_merge");
        let len_ok_blk = self.new_aux_block("funcset_member_len_ok");
        let loop_hdr = self.new_aux_block("funcset_member_hdr");
        let loop_body = self.new_aux_block("funcset_member_body");
        let domain_ok_blk = self.new_aux_block("funcset_member_domain_ok");
        let range_ok_blk = self.new_aux_block("funcset_member_range_ok");
        let loop_inc = self.new_aux_block("funcset_member_inc");

        let false_id = self.block_id_of(false_blk);
        let true_id = self.block_id_of(true_blk);
        let merge_id = self.block_id_of(merge_blk);
        let len_ok_id = self.block_id_of(len_ok_blk);
        let loop_hdr_id = self.block_id_of(loop_hdr);
        let loop_body_id = self.block_id_of(loop_body);
        let domain_ok_id = self.block_id_of(domain_ok_blk);
        let range_ok_id = self.block_id_of(range_ok_blk);
        let loop_inc_id = self.block_id_of(loop_inc);

        let actual_len = self.load_at_offset(block_idx, func_ptr, 0);
        let expected_len = self.emit_i64_const(block_idx, i64::from(domain_len));
        let len_matches = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: actual_len,
                rhs: expected_len,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: len_matches,
                then_target: len_ok_id,
                then_args: vec![],
                else_target: false_id,
                else_args: vec![],
            }),
        );

        let zero = self.emit_i64_const(len_ok_blk, 0);
        let idx_alloca = self.emit_with_result(
            len_ok_blk,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            len_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        let i_val = self.emit_with_result(
            loop_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let expected_len = self.emit_i64_const(loop_hdr, i64::from(domain_len));
        let in_bounds = self.emit_with_result(
            loop_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: expected_len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: loop_body_id,
                then_args: vec![],
                else_target: true_id,
                else_args: vec![],
            }),
        );

        let i_body = self.emit_with_result(
            loop_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_body, 1);
        let two = self.emit_i64_const(loop_body, 2);
        let pair_offset = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: i_body,
                rhs: two,
            },
        );
        let key_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: pair_offset,
                rhs: one,
            },
        );
        let value_slot = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: pair_offset,
                rhs: two,
            },
        );
        let key_val = self.load_at_dynamic_offset(loop_body, func_ptr, key_slot);
        let value_val = self.load_at_dynamic_offset(loop_body, func_ptr, value_slot);
        self.lower_value_in_set_ptr_branch(
            loop_body,
            key_val,
            domain_ptr,
            domain_ok_id,
            false_id,
            "funcset_member_domain",
        );

        match range_shape {
            super::AggregateShape::Powerset { base } => {
                base.validate_powerset_base("SetIn: function-set SUBSET range")?;
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_scalar_in_function_set_range_shape_branch(
                        domain_ok_blk,
                        value_val,
                        value_shape.as_deref(),
                        range_value,
                        super::AggregateShape::Powerset { base },
                        range_ok_id,
                        false_id,
                        "funcset_member_range_subset",
                    )?;
                } else {
                    let value_ptr = self.i64_value_as_ptr(domain_ok_blk, value_val);
                    let range_ptr = self.i64_value_as_ptr(domain_ok_blk, range_value);
                    self.lower_subseteq_ptr_branch(
                        domain_ok_blk,
                        value_ptr,
                        range_ptr,
                        range_ok_id,
                        false_id,
                        "funcset_member_range_subset",
                    );
                }
            }
            // `(SUBSET S) \ {{}}` function-set range: subset half as `Powerset`,
            // plus a per-entry non-empty guard.
            super::AggregateShape::NonEmptyPowerset { base } => {
                base.validate_powerset_base("SetIn: function-set non-empty SUBSET range")?;
                if matches!(base.as_ref(), super::AggregateShape::SetBitmask { .. }) {
                    self.lower_scalar_in_function_set_range_shape_branch(
                        domain_ok_blk,
                        value_val,
                        value_shape.as_deref(),
                        range_value,
                        super::AggregateShape::NonEmptyPowerset { base },
                        range_ok_id,
                        false_id,
                        "funcset_member_range_nonempty_subset",
                    )?;
                } else {
                    let guard_blk = self.new_aux_block("funcset_nonempty_powerset_guard");
                    let guard_id = self.block_id_of(guard_blk);
                    let value_ptr = self.i64_value_as_ptr(domain_ok_blk, value_val);
                    let range_ptr = self.i64_value_as_ptr(domain_ok_blk, range_value);
                    self.lower_subseteq_ptr_branch(
                        domain_ok_blk,
                        value_ptr,
                        range_ptr,
                        guard_id,
                        false_id,
                        "funcset_member_range_nonempty_subset",
                    );
                    self.branch_on_set_nonempty(
                        guard_blk,
                        value_val,
                        value_shape.as_deref(),
                        range_ok_id,
                        false_id,
                        "funcset_member_range_nonempty_subset",
                    )?;
                }
            }
            // Lazy union range (lever L1): static-only per-entry membership;
            // `range_value` (funcset slot 1) mirrors the LazyUnion register's
            // inert placeholder and is dead in this arm (amendment H1).
            super::AggregateShape::LazyUnion { left, right } => {
                self.lower_entry_in_lazy_union_range_branch(
                    domain_ok_blk,
                    value_val,
                    value_shape.as_deref(),
                    &left,
                    &right,
                    range_ok_id,
                    false_id,
                    "funcset_member_range_lazy_union",
                )?;
            }
            super::AggregateShape::Set { .. }
            | super::AggregateShape::ExactIntSet { .. }
            | super::AggregateShape::ExactScalarSet { .. }
            | super::AggregateShape::SetBitmask { .. }
            | super::AggregateShape::FiniteSet
            | super::AggregateShape::BoundedSet { .. }
            | super::AggregateShape::Interval { .. } => {
                self.lower_scalar_in_function_set_range_shape_branch(
                    domain_ok_blk,
                    value_val,
                    value_shape.as_deref(),
                    range_value,
                    range_shape,
                    range_ok_id,
                    false_id,
                    "funcset_member_range",
                )?;
            }
            super::AggregateShape::RecordSet { fields } => {
                let range_ptr = self.i64_value_as_ptr(domain_ok_blk, range_value);
                self.lower_record_value_in_record_set_ptr_branch(
                    domain_ok_blk,
                    value_val,
                    range_ptr,
                    fields,
                    range_ok_id,
                    false_id,
                )?;
            }
            super::AggregateShape::SymbolicDomain(domain) => {
                let member = self.emit_symbolic_domain_membership_i64(
                    domain_ok_blk,
                    value_val,
                    value_shape.as_deref(),
                    domain,
                )?;
                let zero = self.emit_i64_const(domain_ok_blk, 0);
                let is_member = self.emit_with_result(
                    domain_ok_blk,
                    Inst::ICmp {
                        op: ICmpOp::Ne,
                        ty: Ty::I64,
                        lhs: member,
                        rhs: zero,
                    },
                );
                self.emit(
                    domain_ok_blk,
                    InstrNode::new(Inst::CondBr {
                        cond: is_member,
                        then_target: range_ok_id,
                        then_args: vec![],
                        else_target: false_id,
                        else_args: vec![],
                    }),
                );
            }
            super::AggregateShape::FunctionSet { domain, range } => {
                let range_ptr = self.i64_value_as_ptr(domain_ok_blk, range_value);
                self.lower_function_like_value_in_function_set_ptr_branch(
                    domain_ok_blk,
                    value_val,
                    value_shape.as_deref(),
                    range_ptr,
                    *domain,
                    *range,
                    range_ok_id,
                    false_id,
                    "funcset_member_range_function",
                )?;
            }
            super::AggregateShape::SeqSet { base } => {
                let seq_element_shape = match value_shape.as_deref() {
                    Some(super::AggregateShape::Sequence { element, .. }) => element.as_deref(),
                    _ => None,
                };
                let range_value = if Self::lazy_domain_runtime_payload_is_compact_mask(&base) {
                    range_value
                } else {
                    self.i64_value_as_ptr(domain_ok_blk, range_value)
                };
                self.lower_seq_value_in_seq_set_ptr_branch(
                    domain_ok_blk,
                    value_val,
                    seq_element_shape,
                    range_value,
                    *base,
                    range_ok_id,
                    false_id,
                    "funcset_member_range_seq",
                )?;
            }
            other => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: unsupported function-set range shape: {other:?}"
                )));
            }
        }

        self.emit(
            range_ok_blk,
            InstrNode::new(Inst::Br {
                target: loop_inc_id,
                args: vec![],
            }),
        );

        let i_inc = self.emit_with_result(
            loop_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(loop_inc, 1);
        let next_i = self.emit_with_result(
            loop_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_inc,
                rhs: one,
            },
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            loop_inc,
            InstrNode::new(Inst::Br {
                target: loop_hdr_id,
                args: vec![],
            }),
        );

        self.store_reg_imm(true_blk, rd, 1)?;
        self.emit(
            true_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        self.store_reg_imm(false_blk, rd, 0)?;
        self.emit(
            false_blk,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge_blk))
    }

    fn lower_interval_membership_value(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_val: trust_ir::value::ValueId,
        lo: i64,
        hi: i64,
    ) -> Result<(), TrustIrError> {
        let in_range = self.emit_interval_membership_i64(block_idx, elem_val, lo, hi);
        self.store_reg_value(block_idx, rd, in_range)
    }

    fn lower_record_set_membership(
        &mut self,
        block_idx: usize,
        rd: u8,
        elem_reg: u8,
        record_set_reg: u8,
        fields: Vec<(tla_core::NameId, super::AggregateShape)>,
    ) -> Result<Option<usize>, TrustIrError> {
        let record_shape = match self.aggregate_shapes.get(&elem_reg).cloned() {
            Some(record_shape @ super::AggregateShape::Record { .. }) => record_shape,
            Some(super::AggregateShape::StateValue) => {
                super::AggregateShape::record_from_record_set_domains(&fields)
            }
            Some(other) => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "SetIn: record-set membership requires tracked record element shape, got {other:?}"
                )));
            }
            None => {
                return Err(TrustIrError::UnsupportedOpcode(
                    "SetIn: record-set membership requires tracked record element shape".to_owned(),
                ));
            }
        };

        let super::AggregateShape::Record {
            fields: record_fields,
        } = &record_shape
        else {
            unreachable!("record_shape was matched as a record");
        };
        if record_fields.len() != fields.len() {
            self.store_reg_imm(block_idx, rd, 0)?;
            return Ok(Some(block_idx));
        }

        if fields.is_empty() {
            self.store_reg_imm(block_idx, rd, 1)?;
            return Ok(Some(block_idx));
        }

        if let Some(source_slot) = self.compact_state_slot_for_use(block_idx, elem_reg)? {
            let true_blk = self.new_aux_block("record_set_compact_true");
            let false_blk = self.new_aux_block("record_set_compact_false");
            let merge_blk = self.new_aux_block("record_set_compact_merge");
            let true_id = self.block_id_of(true_blk);
            let false_id = self.block_id_of(false_blk);
            let merge_id = self.block_id_of(merge_blk);

            let record_base_slot = self.emit_i64_const(block_idx, i64::from(source_slot.offset));
            let record_set_ptr = self.load_reg_as_ptr(block_idx, record_set_reg)?;
            self.lower_compact_record_value_in_record_set_branch(
                block_idx,
                source_slot.source_ptr,
                record_base_slot,
                Some(&record_shape),
                record_set_ptr,
                fields,
                true_id,
                false_id,
            )?;

            self.store_reg_imm(true_blk, rd, 1)?;
            self.emit(
                true_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );
            self.store_reg_imm(false_blk, rd, 0)?;
            self.emit(
                false_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );

            return Ok(Some(merge_blk));
        }

        let rec_ptr = self.load_reg_as_ptr(block_idx, elem_reg)?;
        let record_set_ptr = self.load_reg_as_ptr(block_idx, record_set_reg)?;
        let mut result: Option<trust_ir::value::ValueId> = None;

        for (domain_slot, (field_name, field_set_shape)) in fields.iter().enumerate() {
            let Some((field_idx, _)) = record_shape.record_field(*field_name) else {
                self.store_reg_imm(block_idx, rd, 0)?;
                return Ok(Some(block_idx));
            };
            let field_shape = record_shape
                .record_field(*field_name)
                .and_then(|(_, shape)| shape);
            let field_val = self.load_at_offset(block_idx, rec_ptr, field_idx);
            let field_set_shape =
                Self::materialized_domain_shape_for_pointer(field_set_shape.clone());
            let field_ok = match &field_set_shape {
                super::AggregateShape::Interval { lo, hi } => {
                    self.emit_interval_membership_i64(block_idx, field_val, *lo, *hi)
                }
                super::AggregateShape::Set { len, .. } => {
                    let domain_slot =
                        u32::try_from(domain_slot).expect("record set slot index must fit in u32");
                    let domain_value = self.load_at_offset(block_idx, record_set_ptr, domain_slot);
                    let domain_ptr = self.i64_value_as_ptr(block_idx, domain_value);
                    self.emit_finite_set_membership_i64(block_idx, field_val, domain_ptr, *len)
                }
                super::AggregateShape::SymbolicDomain(domain) => self
                    .emit_symbolic_domain_membership_i64(
                        block_idx,
                        field_val,
                        field_shape.as_ref(),
                        *domain,
                    )?,
                other => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "SetIn: record-set field domain is not directly lowerable: {other:?}"
                    )));
                }
            };

            result = Some(match result {
                None => field_ok,
                Some(prev) => self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: prev,
                        rhs: field_ok,
                    },
                ),
            });
        }

        self.store_reg_value(
            block_idx,
            rd,
            result.expect("non-empty record-set fields must produce a result"),
        )?;
        Ok(Some(block_idx))
    }

    fn lower_record_value_in_record_set_ptr_branch(
        &mut self,
        block_idx: usize,
        record_value: ValueId,
        record_set_ptr: ValueId,
        fields: Vec<(tla_core::NameId, super::AggregateShape)>,
        success_target: BlockId,
        failure_target: BlockId,
    ) -> Result<(), TrustIrError> {
        let record_shape = super::AggregateShape::record_from_record_set_domains(&fields);
        let super::AggregateShape::Record {
            fields: record_fields,
        } = &record_shape
        else {
            unreachable!("record_from_record_set_domains must return Record");
        };
        if record_fields.len() != fields.len() {
            self.emit(
                block_idx,
                InstrNode::new(Inst::Br {
                    target: failure_target,
                    args: vec![],
                }),
            );
            return Ok(());
        }
        if fields.is_empty() {
            self.emit(
                block_idx,
                InstrNode::new(Inst::Br {
                    target: success_target,
                    args: vec![],
                }),
            );
            return Ok(());
        }

        let rec_ptr = self.i64_value_as_ptr(block_idx, record_value);
        let mut result: Option<ValueId> = None;
        for (domain_slot, (field_name, field_set_shape)) in fields.iter().enumerate() {
            let Some((field_idx, _)) = record_shape.record_field(*field_name) else {
                self.emit(
                    block_idx,
                    InstrNode::new(Inst::Br {
                        target: failure_target,
                        args: vec![],
                    }),
                );
                return Ok(());
            };
            let field_shape = record_shape
                .record_field(*field_name)
                .and_then(|(_, shape)| shape);
            let field_val = self.load_at_offset(block_idx, rec_ptr, field_idx);
            let field_set_shape =
                Self::materialized_domain_shape_for_pointer(field_set_shape.clone());
            let field_ok = match &field_set_shape {
                super::AggregateShape::Interval { lo, hi } => {
                    self.emit_interval_membership_i64(block_idx, field_val, *lo, *hi)
                }
                super::AggregateShape::Set { len, .. } => {
                    let domain_slot =
                        u32::try_from(domain_slot).expect("record set slot index must fit in u32");
                    let domain_value = self.load_at_offset(block_idx, record_set_ptr, domain_slot);
                    let domain_ptr = self.i64_value_as_ptr(block_idx, domain_value);
                    self.emit_finite_set_membership_i64(block_idx, field_val, domain_ptr, *len)
                }
                super::AggregateShape::SymbolicDomain(domain) => self
                    .emit_symbolic_domain_membership_i64(
                        block_idx,
                        field_val,
                        field_shape.as_ref(),
                        *domain,
                    )?,
                other => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "SetIn: record-set field domain is not directly lowerable: {other:?}"
                    )));
                }
            };

            result = Some(match result {
                None => field_ok,
                Some(prev) => self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: prev,
                        rhs: field_ok,
                    },
                ),
            });
        }

        self.branch_on_i64_truth(
            block_idx,
            result.expect("non-empty record-set fields must produce a result"),
            success_target,
            failure_target,
        );
        Ok(())
    }

    fn emit_interval_membership_i64(
        &mut self,
        block_idx: usize,
        elem_val: trust_ir::value::ValueId,
        lo: i64,
        hi: i64,
    ) -> trust_ir::value::ValueId {
        if hi < lo {
            return self.emit_i64_const(block_idx, 0);
        }

        let lo_val = self.emit_i64_const(block_idx, lo);
        let hi_val = self.emit_i64_const(block_idx, hi);
        let ge_lo = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: elem_val,
                rhs: lo_val,
            },
        );
        let le_hi = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: elem_val,
                rhs: hi_val,
            },
        );
        let ge_lo_i64 = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: ge_lo,
            },
        );
        let le_hi_i64 = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: le_hi,
            },
        );
        self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: ge_lo_i64,
                rhs: le_hi_i64,
            },
        )
    }

    /// Lower SetUnion { rd, r1, r2 }: union of two sets.
    ///
    /// Creates a new set containing all elements from both sets.
    pub(super) fn lower_set_union(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        // Native-on-general-Value handle path (#4318): `s \cup {n}` where `s` is
        // a compound-set handle (from a handle-mode LoadVar) and `{n}` is a
        // handle-mode set literal. Both operands must be handles; we call the
        // `tla_set_union` host op (which unboxes, computes `SortedSet::union`
        // via tla_value, and reboxes — interpreter parity) and mark the result
        // a handle. If only one operand is a handle we fail closed: a mixed
        // handle/non-handle union has no sound encoding here, so the whole
        // action routes to the interpreter rather than silently mis-unioning.
        if self.has_handle_provenance(r1) || self.has_handle_provenance(r2) {
            if !(self.has_handle_provenance(r1) && self.has_handle_provenance(r2)) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "handle-mode SetUnion: exactly one operand (r{r1}/r{r2}) holds a TlaHandle; \
                     a mixed handle/non-handle union is not soundly lowerable — failing closed"
                )));
            }
            let h1 = self.load_reg(block_idx, r1)?;
            let h2 = self.load_reg(block_idx, r2)?;
            let result = self.emit_sanctioned_handle_extern_i64(
                block_idx,
                super::SanctionedHandleExternSite::HandleSetUnion,
                "tla_set_union",
                2,
                vec![h1, h2],
            )?;
            self.store_reg_value(block_idx, rd, result)?;
            self.set_handle_provenance(rd);
            return Ok(Some(block_idx));
        }

        if let Some((source_reg, domain)) = self.symbolic_domain_union_source_reg(r1, r2) {
            let value = self.load_reg(block_idx, source_reg)?;
            self.store_reg_value(block_idx, rd, value)?;
            self.aggregate_shapes
                .insert(rd, super::AggregateShape::SymbolicDomain(domain));
            if let Some(value) = self.const_scalar_values.get(&source_reg).copied() {
                self.const_scalar_values.insert(rd, value);
            } else {
                self.const_scalar_values.remove(&rd);
            }
            self.const_set_sizes.remove(&rd);
            self.load_imm_scalar_regs.remove(&rd);
            self.compact_function_domains.remove(&rd);
            self.clear_flat_funcdef_pair_list(rd);
            return Ok(Some(block_idx));
        }

        // RecordSetBitmask union (RecordSetBitmask step 3/5): per-slot OR, then
        // AND each slot with its per-slot valid mask. Runs BEFORE the scalar
        // `compact_binary_set_universe` (which returns `None` for the multi-slot
        // record bitmask) and BEFORE `reject_lazy_set_operand` / the materialized
        // copy-loop fall-through, so a RecordSetBitmask operand is never read as
        // a materialized set pointer.
        if let Some((universe_len, slot_count, universe)) =
            self.record_set_bitmask_binary_universe("SetUnion", r1, r2)?
        {
            let (left, left_found) = self.record_set_bitmask_operand_slots_with_found(
                block_idx,
                r1,
                universe_len,
                slot_count,
                &universe,
                "SetUnion",
            )?;
            let (right, right_found) = self.record_set_bitmask_operand_slots_with_found(
                block_idx,
                r2,
                universe_len,
                slot_count,
                &universe,
                "SetUnion",
            )?;
            // Strictness: a literal element that matches no universe key means
            // the true union contains an out-of-universe record; fall back to
            // the interpreter for this state instead of dropping the element.
            let block_idx =
                self.emit_record_set_strictness_guard(block_idx, left_found, right_found)?;
            let mut result_slots = Vec::with_capacity(slot_count as usize);
            for slot_index in 0..slot_count as usize {
                let or = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Or,
                        ty: Ty::I64,
                        lhs: left[slot_index],
                        rhs: right[slot_index],
                    },
                );
                let valid_mask =
                    super::record_set_bitmask_slot_valid_mask_ir(universe_len, slot_index)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "SetUnion: RecordSetBitmask slot {slot_index} out of range for \
                                 universe_len {universe_len}"
                            ))
                        })?;
                let valid_mask_val = self.emit_i64_const(block_idx, valid_mask as i64);
                let masked = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: or,
                        rhs: valid_mask_val,
                    },
                );
                result_slots.push(masked);
            }
            return self.store_record_set_bitmask_result(
                block_idx,
                rd,
                &result_slots,
                universe_len,
                slot_count,
                universe,
            );
        }

        if let Some((universe_len, universe)) =
            self.compact_binary_set_universe("SetUnion", r1, r2)?
        {
            let (block_idx, left) = self.emit_set_operand_bitmask_i64_allow_materialized(
                block_idx,
                r1,
                universe_len,
                &universe,
                "SetUnion",
            )?;
            let (block_idx, right) = self.emit_set_operand_bitmask_i64_allow_materialized(
                block_idx,
                r2,
                universe_len,
                &universe,
                "SetUnion",
            )?;
            let raw = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: left,
                    rhs: right,
                },
            );
            let valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, "SetUnion")?;
            let valid_mask_val = self.emit_i64_const(block_idx, valid_mask);
            let result = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: raw,
                    rhs: valid_mask_val,
                },
            );
            self.store_reg_value(block_idx, rd, result)?;
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::SetBitmask {
                    universe_len,
                    universe,
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(Some(block_idx));
        }

        // Lazy-union tracking (lever L1): `A \cup B` where at least one
        // operand is a lazy SUBSET-style shape and every flattened arm has
        // exact compile-time metadata. The result register stores an inert
        // placeholder (0) — soundness amendment H1: no admitted consumer ever
        // loads a LazyUnion payload (membership lowering is fully static and
        // `load_reg_as_ptr` fails closed on the shape), so the placeholder
        // can never be dereferenced. Anything not admissible falls through to
        // today's `reject_lazy_set_operand` rejection below.
        let lazy_union_shape = match (
            self.aggregate_shapes.get(&r1),
            self.aggregate_shapes.get(&r2),
        ) {
            (Some(left), Some(right)) => super::lazy_union_shape_from_operands(left, right),
            _ => None,
        };
        if let Some(shape) = lazy_union_shape {
            self.invalidate_reg_tracking(rd);
            self.clear_handle_provenance(rd);
            self.store_reg_imm(block_idx, rd, 0)?;
            self.aggregate_shapes.insert(rd, shape);
            return Ok(Some(block_idx));
        }

        self.reject_lazy_set_operand("SetUnion", r1)?;
        self.reject_lazy_set_operand("SetUnion", r2)?;

        // Load both set pointers and lengths
        let set1_ptr = self.load_reg_as_ptr(block_idx, r1)?;
        let set2_ptr = self.load_reg_as_ptr(block_idx, r2)?;
        let len1 = self.load_at_offset(block_idx, set1_ptr, 0);
        let len2 = self.load_at_offset(block_idx, set2_ptr, 0);

        let result_ptr = match (
            self.const_set_sizes.get(&r1).copied(),
            self.const_set_sizes.get(&r2).copied(),
        ) {
            (Some(len1), Some(len2)) => {
                let total_slots = len1
                    .checked_add(len2)
                    .and_then(|slots| slots.checked_add(1))
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "SetUnion: static result allocation size overflows u32: {len1} + {len2} + 1"
                        ))
                    })?;
                self.alloc_aggregate(block_idx, total_slots)
            }
            _ => {
                // Max result size = len1 + len2 + 1 (header)
                let total_elem = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: len1,
                        rhs: len2,
                    },
                );
                let one_64 = self.emit_i64_const(block_idx, 1);
                let total_slots = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: total_elem,
                        rhs: one_64,
                    },
                );
                let total_i32 = self.emit_with_result(
                    block_idx,
                    Inst::Cast {
                        op: CastOp::Trunc,
                        src_ty: Ty::I64,
                        dst_ty: Ty::I32,
                        operand: total_slots,
                    },
                );
                self.emit_with_result(
                    block_idx,
                    Inst::Alloca {
                        ty: Ty::I64,
                        count: Some(total_i32),
                        align: None,
                    },
                )
            }
        };

        // Copy all elements from set1 (slots 1..=len1)
        // For the trust-ir-level representation, we use a simple loop to copy elements.
        // Store initial result length as len1 (we'll copy set1 completely first).
        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

        // Alloca for write cursor (starts at 1)
        let cursor_alloca = self.emit_with_result(
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
                ptr: cursor_alloca,
                value: one,
                align: None,
                volatile: false,
            }),
        );

        // Copy loop for set1
        let copy1_header = self.new_aux_block("union_copy1_hdr");
        let copy1_body = self.new_aux_block("union_copy1_body");
        let copy1_done = self.new_aux_block("union_copy1_done");

        let hdr1_id = self.block_id_of(copy1_header);
        let body1_id = self.block_id_of(copy1_body);
        let done1_id = self.block_id_of(copy1_done);

        // Alloca for loop index
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
        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: hdr1_id,
                args: vec![],
            }),
        );

        // Header: i < len1?
        let i_val = self.emit_with_result(
            copy1_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp1 = self.emit_with_result(
            copy1_header,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: len1,
            },
        );
        self.emit(
            copy1_header,
            InstrNode::new(Inst::CondBr {
                cond: cmp1,
                then_target: body1_id,
                then_args: vec![],
                else_target: done1_id,
                else_args: vec![],
            }),
        );

        // Body: copy element
        let i_val2 = self.emit_with_result(
            copy1_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let src_slot = self.emit_with_result(
            copy1_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        let elem = self.load_at_dynamic_offset(copy1_body, set1_ptr, src_slot);
        let cursor = self.emit_with_result(
            copy1_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: cursor_alloca,
                align: None,
                volatile: false,
            },
        );
        self.store_at_dynamic_offset(copy1_body, result_ptr, cursor, elem);
        // Increment
        let next_i = self.emit_with_result(
            copy1_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        self.emit(
            copy1_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        let next_cursor = self.emit_with_result(
            copy1_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cursor,
                rhs: one,
            },
        );
        self.emit(
            copy1_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: cursor_alloca,
                value: next_cursor,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            copy1_body,
            InstrNode::new(Inst::Br {
                target: hdr1_id,
                args: vec![],
            }),
        );

        // After copying set1, copy set2 elements
        let copy2_header = self.new_aux_block("union_copy2_hdr");
        let copy2_body = self.new_aux_block("union_copy2_body");
        let finalize = self.new_aux_block("union_finalize");

        let hdr2_id = self.block_id_of(copy2_header);
        let body2_id = self.block_id_of(copy2_body);
        let finalize_id = self.block_id_of(finalize);

        // Reset loop index
        self.emit(
            copy1_done,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            copy1_done,
            InstrNode::new(Inst::Br {
                target: hdr2_id,
                args: vec![],
            }),
        );

        // Header: i < len2?
        let i_val3 = self.emit_with_result(
            copy2_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp2 = self.emit_with_result(
            copy2_header,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val3,
                rhs: len2,
            },
        );
        self.emit(
            copy2_header,
            InstrNode::new(Inst::CondBr {
                cond: cmp2,
                then_target: body2_id,
                then_args: vec![],
                else_target: finalize_id,
                else_args: vec![],
            }),
        );

        // Body: copy element
        let i_val4 = self.emit_with_result(
            copy2_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let src_slot2 = self.emit_with_result(
            copy2_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val4,
                rhs: one,
            },
        );
        let elem2 = self.load_at_dynamic_offset(copy2_body, set2_ptr, src_slot2);
        let cursor2 = self.emit_with_result(
            copy2_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: cursor_alloca,
                align: None,
                volatile: false,
            },
        );
        self.store_at_dynamic_offset(copy2_body, result_ptr, cursor2, elem2);
        let next_i2 = self.emit_with_result(
            copy2_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val4,
                rhs: one,
            },
        );
        self.emit(
            copy2_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: next_i2,
                align: None,
                volatile: false,
            }),
        );
        let next_cursor2 = self.emit_with_result(
            copy2_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cursor2,
                rhs: one,
            },
        );
        self.emit(
            copy2_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: cursor_alloca,
                value: next_cursor2,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            copy2_body,
            InstrNode::new(Inst::Br {
                target: hdr2_id,
                args: vec![],
            }),
        );

        // Finalize: store final length = cursor - 1
        let final_cursor = self.emit_with_result(
            finalize,
            Inst::Load {
                ty: Ty::I64,
                ptr: cursor_alloca,
                align: None,
                volatile: false,
            },
        );
        let final_len = self.emit_with_result(
            finalize,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: final_cursor,
                rhs: one,
            },
        );
        self.store_at_offset(finalize, result_ptr, 0, final_len);
        self.store_reg_ptr(finalize, rd, result_ptr)?;

        Ok(Some(finalize))
    }

    /// Lower SetIntersect { rd, r1, r2 }: intersection of two sets.
    ///
    /// Creates a new set containing elements present in both sets.
    /// Uses nested loops: for each element in set1, scan set2 for a match.
    pub(super) fn lower_set_intersect(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        if let Some((universe_len, universe)) =
            self.compact_binary_set_universe("SetIntersect", r1, r2)?
        {
            let (block_idx, left) = self.emit_set_intersect_operand_bitmask_i64(
                block_idx,
                r1,
                universe_len,
                &universe,
                "SetIntersect",
            )?;
            let (block_idx, right) = self.emit_set_intersect_operand_bitmask_i64(
                block_idx,
                r2,
                universe_len,
                &universe,
                "SetIntersect",
            )?;
            let result = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: left,
                    rhs: right,
                },
            );
            self.store_reg_value(block_idx, rd, result)?;
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::SetBitmask {
                    universe_len,
                    universe,
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(Some(block_idx));
        }

        self.reject_lazy_set_operand("SetIntersect", r1)?;
        self.reject_lazy_set_operand("SetIntersect", r2)?;

        let set1_ptr = self.load_reg_as_ptr(block_idx, r1)?;
        let set2_ptr = self.load_reg_as_ptr(block_idx, r2)?;
        let len1 = self.load_at_offset(block_idx, set1_ptr, 0);
        let _len2 = self.load_at_offset(block_idx, set2_ptr, 0);

        // Allocate result set with max size = len1 + 1
        let one_64 = self.emit_i64_const(block_idx, 1);
        let max_slots = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: len1,
                rhs: one_64,
            },
        );
        let result_ptr = if let Some(max_len) = self.const_set_sizes.get(&r1).copied() {
            self.alloc_aggregate(block_idx, max_len + 1)
        } else {
            let max_i32 = self.emit_with_result(
                block_idx,
                Inst::Cast {
                    op: CastOp::Trunc,
                    src_ty: Ty::I64,
                    dst_ty: Ty::I32,
                    operand: max_slots,
                },
            );
            self.emit_with_result(
                block_idx,
                Inst::Alloca {
                    ty: Ty::I64,
                    count: Some(max_i32),
                    align: None,
                },
            )
        };

        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

        // Write cursor for result
        let cursor_alloca = self.emit_with_result(
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
                ptr: cursor_alloca,
                value: one,
                align: None,
                volatile: false,
            }),
        );

        // Outer loop index
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

        let outer_hdr = self.new_aux_block("isect_outer_hdr");
        let outer_body = self.new_aux_block("isect_outer_body");
        let inner_hdr = self.new_aux_block("isect_inner_hdr");
        let inner_body = self.new_aux_block("isect_inner_body");
        let inner_inc = self.new_aux_block("isect_inner_inc");
        let found_blk = self.new_aux_block("isect_found");
        let outer_inc = self.new_aux_block("isect_outer_inc");
        let finalize = self.new_aux_block("isect_finalize");

        let outer_hdr_id = self.block_id_of(outer_hdr);
        let outer_body_id = self.block_id_of(outer_body);
        let inner_hdr_id = self.block_id_of(inner_hdr);
        let inner_body_id = self.block_id_of(inner_body);
        let inner_inc_id = self.block_id_of(inner_inc);
        let found_blk_id = self.block_id_of(found_blk);
        let outer_inc_id = self.block_id_of(outer_inc);
        let finalize_id = self.block_id_of(finalize);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: outer_hdr_id,
                args: vec![],
            }),
        );

        // Outer header: i < len1?
        let i_val = self.emit_with_result(
            outer_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp_outer = self.emit_with_result(
            outer_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: len1,
            },
        );
        self.emit(
            outer_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp_outer,
                then_target: outer_body_id,
                then_args: vec![],
                else_target: finalize_id,
                else_args: vec![],
            }),
        );

        // Outer body: load element from set1
        let i_val2 = self.emit_with_result(
            outer_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot = self.emit_with_result(
            outer_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        let elem1 = self.load_at_dynamic_offset(outer_body, set1_ptr, slot);

        // Inner loop: search set2 for elem1
        let j_alloca = self.emit_with_result(
            outer_body,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            outer_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: j_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            outer_body,
            InstrNode::new(Inst::Br {
                target: inner_hdr_id,
                args: vec![],
            }),
        );

        // Inner header: j < len2?
        let j_val = self.emit_with_result(
            inner_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp_inner = self.emit_with_result(
            inner_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: j_val,
                rhs: _len2,
            },
        );
        self.emit(
            inner_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp_inner,
                then_target: inner_body_id,
                then_args: vec![],
                else_target: outer_inc_id,
                else_args: vec![], // not found
            }),
        );

        // Inner body: compare
        let j_val2 = self.emit_with_result(
            inner_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot2 = self.emit_with_result(
            inner_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: j_val2,
                rhs: one,
            },
        );
        let elem2 = self.load_at_dynamic_offset(inner_body, set2_ptr, slot2);
        let eq = self.emit_with_result(
            inner_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: elem1,
                rhs: elem2,
            },
        );
        self.emit(
            inner_body,
            InstrNode::new(Inst::CondBr {
                cond: eq,
                then_target: found_blk_id,
                then_args: vec![],
                else_target: inner_inc_id,
                else_args: vec![],
            }),
        );

        // Inner increment
        let j_val3 = self.emit_with_result(
            inner_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_j = self.emit_with_result(
            inner_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: j_val3,
                rhs: one,
            },
        );
        self.emit(
            inner_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: j_alloca,
                value: next_j,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            inner_inc,
            InstrNode::new(Inst::Br {
                target: inner_hdr_id,
                args: vec![],
            }),
        );

        // Found: add elem1 to result
        let cursor = self.emit_with_result(
            found_blk,
            Inst::Load {
                ty: Ty::I64,
                ptr: cursor_alloca,
                align: None,
                volatile: false,
            },
        );
        self.store_at_dynamic_offset(found_blk, result_ptr, cursor, elem1);
        let next_cursor = self.emit_with_result(
            found_blk,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cursor,
                rhs: one,
            },
        );
        self.emit(
            found_blk,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: cursor_alloca,
                value: next_cursor,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            found_blk,
            InstrNode::new(Inst::Br {
                target: outer_inc_id,
                args: vec![],
            }),
        );

        // Outer increment
        let i_val3 = self.emit_with_result(
            outer_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_i = self.emit_with_result(
            outer_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val3,
                rhs: one,
            },
        );
        self.emit(
            outer_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            outer_inc,
            InstrNode::new(Inst::Br {
                target: outer_hdr_id,
                args: vec![],
            }),
        );

        // Finalize
        let final_cursor = self.emit_with_result(
            finalize,
            Inst::Load {
                ty: Ty::I64,
                ptr: cursor_alloca,
                align: None,
                volatile: false,
            },
        );
        let final_len = self.emit_with_result(
            finalize,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: final_cursor,
                rhs: one,
            },
        );
        self.store_at_offset(finalize, result_ptr, 0, final_len);
        self.store_reg_ptr(finalize, rd, result_ptr)?;

        Ok(Some(finalize))
    }

    /// Lower SetDiff { rd, r1, r2 }: set difference (r1 \ r2).
    ///
    /// Creates a new set containing elements in r1 that are NOT in r2.
    /// Returns the base shape of `reg` when it tracks a (possibly already
    /// non-empty) lazy powerset, i.e. `SUBSET S` or `(SUBSET S) \ {{}}`.
    fn powerset_like_base_shape(&self, reg: u8) -> Option<super::AggregateShape> {
        match self.aggregate_shapes.get(&reg) {
            Some(
                super::AggregateShape::Powerset { base }
                | super::AggregateShape::NonEmptyPowerset { base },
            ) => Some((**base).clone()),
            _ => None,
        }
    }

    /// Detects the `(SUBSET S) \ {{}}` idiom: a lazy powerset lhs minus the
    /// statically-known singleton `{{}}`. Both operands must carry the exact
    /// tracked shapes; anything else is handled by the generic SetDiff paths.
    fn is_nonempty_powerset_setdiff(&self, r1: u8, r2: u8) -> bool {
        if self.powerset_like_base_shape(r1).is_none() {
            return false;
        }
        self.aggregate_shapes
            .get(&r2)
            .is_some_and(super::is_singleton_empty_set_shape)
    }

    pub(super) fn lower_set_diff(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        // `(SUBSET S) \ {{}}` (the non-empty powerset of `S`). The lazy
        // `Powerset`/`NonEmptyPowerset` lhs is represented at runtime solely by
        // the base value of `S` (see `lower_powerset`); subtracting the
        // statically-known singleton `{{}}` does not change that runtime
        // representation — it only tightens membership to reject the empty set,
        // which the `NonEmptyPowerset` shape records. So we mirror
        // `lower_powerset` here: copy the base value into `rd` and tag it as a
        // non-empty powerset. The shape is also recomputed by the dispatcher
        // via `finite_set_diff_shape`; tagging it here keeps the two paths in
        // agreement and is required because the generic lazy-operand rejection
        // below would otherwise abort lowering.
        if self.is_nonempty_powerset_setdiff(r1, r2) {
            let base_shape = self.powerset_like_base_shape(r1).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "SetDiff: non-empty powerset lhs lost its tracked base shape".to_owned(),
                )
            })?;
            let base_value = self.load_reg(block_idx, r1)?;
            self.store_reg_value(block_idx, rd, base_value)?;
            self.compact_state_slots.remove(&rd);
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::NonEmptyPowerset {
                    base: Box::new(base_shape),
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(Some(block_idx));
        }

        // RecordSetBitmask difference (RecordSetBitmask step 3/5): per-slot
        // AND-NOT, then AND each slot with its per-slot valid mask. Mirrors the
        // scalar diff (`left & (right XOR valid_mask) & valid_mask`) per slot.
        // Runs BEFORE the scalar `compact_binary_setdiff_universe` and the
        // materialized fall-through so the multi-slot operand is never read as a
        // materialized set pointer.
        if let Some((universe_len, slot_count, universe)) =
            self.record_set_bitmask_binary_universe("SetDiff", r1, r2)?
        {
            let (left, left_found) = self.record_set_bitmask_operand_slots_with_found(
                block_idx,
                r1,
                universe_len,
                slot_count,
                &universe,
                "SetDiff",
            )?;
            let (right, right_found) = self.record_set_bitmask_operand_slots_with_found(
                block_idx,
                r2,
                universe_len,
                slot_count,
                &universe,
                "SetDiff",
            )?;
            // Strictness: an unmatched LEFT element means the true difference
            // contains an out-of-universe record (unrepresentable). An
            // unmatched RIGHT element cannot remove anything from the left
            // mask, but its very occurrence means the surrounding action built
            // a record outside the proven universe — fall back for byte-exact
            // semantics rather than reason about it here.
            let block_idx =
                self.emit_record_set_strictness_guard(block_idx, left_found, right_found)?;
            let mut result_slots = Vec::with_capacity(slot_count as usize);
            for slot_index in 0..slot_count as usize {
                let valid_mask =
                    super::record_set_bitmask_slot_valid_mask_ir(universe_len, slot_index)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "SetDiff: RecordSetBitmask slot {slot_index} out of range for \
                                 universe_len {universe_len}"
                            ))
                        })?;
                let valid_mask_val = self.emit_i64_const(block_idx, valid_mask as i64);
                let right_complement = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Xor,
                        ty: Ty::I64,
                        lhs: right[slot_index],
                        rhs: valid_mask_val,
                    },
                );
                let and_not = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: left[slot_index],
                        rhs: right_complement,
                    },
                );
                let masked = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: and_not,
                        rhs: valid_mask_val,
                    },
                );
                result_slots.push(masked);
            }
            return self.store_record_set_bitmask_result(
                block_idx,
                rd,
                &result_slots,
                universe_len,
                slot_count,
                universe,
            );
        }

        if let Some((universe_len, universe)) = self.compact_binary_setdiff_universe(r1, r2)? {
            let (block_idx, left) = self
                .emit_setdiff_operand_bitmask_i64_allow_tagged_or_materialized(
                    block_idx,
                    r1,
                    universe_len,
                    &universe,
                    "SetDiff",
                )?;
            let (block_idx, right) = self
                .emit_setdiff_rhs_bitmask_i64_allow_tagged_or_materialized(
                    block_idx,
                    r2,
                    universe_len,
                    &universe,
                    "SetDiff",
                )?;
            let valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, "SetDiff")?;
            let valid_mask_val = self.emit_i64_const(block_idx, valid_mask);
            let right_complement = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Xor,
                    ty: Ty::I64,
                    lhs: right,
                    rhs: valid_mask_val,
                },
            );
            let result = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: left,
                    rhs: right_complement,
                },
            );
            let result = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: result,
                    rhs: valid_mask_val,
                },
            );
            self.store_reg_value(block_idx, rd, result)?;
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::SetBitmask {
                    universe_len,
                    universe,
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(Some(block_idx));
        }

        if let Some((universe_len, universe)) = self.small_interval_setdiff_universe(r1, r2)? {
            let valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, "SetDiff")?;
            let left = self.emit_i64_const(block_idx, valid_mask);
            let right =
                self.emit_small_setdiff_rhs_int_mask_i64(block_idx, r2, universe_len, &universe)?;
            let valid_mask_val = self.emit_i64_const(block_idx, valid_mask);
            let right_complement = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Xor,
                    ty: Ty::I64,
                    lhs: right,
                    rhs: valid_mask_val,
                },
            );
            let result = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: left,
                    rhs: right_complement,
                },
            );
            self.store_reg_value(block_idx, rd, result)?;
            self.aggregate_shapes.insert(
                rd,
                super::AggregateShape::SetBitmask {
                    universe_len,
                    universe,
                },
            );
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(Some(block_idx));
        }

        self.reject_lazy_set_operand("SetDiff", r1)?;
        self.reject_lazy_set_operand("SetDiff", r2)?;

        let set1_ptr = self.load_reg_as_ptr(block_idx, r1)?;
        let set2_ptr = self.load_reg_as_ptr(block_idx, r2)?;
        let len1 = self.load_at_offset(block_idx, set1_ptr, 0);
        let len2 = self.load_at_offset(block_idx, set2_ptr, 0);

        // Allocate result set with max size = len1 + 1
        let one_64 = self.emit_i64_const(block_idx, 1);
        let max_slots = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: len1,
                rhs: one_64,
            },
        );
        let result_ptr = if let Some(max_len) = self.const_set_sizes.get(&r1).copied() {
            self.alloc_aggregate(block_idx, max_len + 1)
        } else {
            let max_i32 = self.emit_with_result(
                block_idx,
                Inst::Cast {
                    op: CastOp::Trunc,
                    src_ty: Ty::I64,
                    dst_ty: Ty::I32,
                    operand: max_slots,
                },
            );
            self.emit_with_result(
                block_idx,
                Inst::Alloca {
                    ty: Ty::I64,
                    count: Some(max_i32),
                    align: None,
                },
            )
        };

        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

        let cursor_alloca = self.emit_with_result(
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
                ptr: cursor_alloca,
                value: one,
                align: None,
                volatile: false,
            }),
        );

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

        let outer_hdr = self.new_aux_block("sdiff_outer_hdr");
        let outer_body = self.new_aux_block("sdiff_outer_body");
        let inner_hdr = self.new_aux_block("sdiff_inner_hdr");
        let inner_body = self.new_aux_block("sdiff_inner_body");
        let inner_inc = self.new_aux_block("sdiff_inner_inc");
        let not_found = self.new_aux_block("sdiff_not_found");
        let outer_inc = self.new_aux_block("sdiff_outer_inc");
        let finalize = self.new_aux_block("sdiff_finalize");

        let outer_hdr_id = self.block_id_of(outer_hdr);
        let outer_body_id = self.block_id_of(outer_body);
        let inner_hdr_id = self.block_id_of(inner_hdr);
        let inner_body_id = self.block_id_of(inner_body);
        let inner_inc_id = self.block_id_of(inner_inc);
        let not_found_id = self.block_id_of(not_found);
        let outer_inc_id = self.block_id_of(outer_inc);
        let finalize_id = self.block_id_of(finalize);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: outer_hdr_id,
                args: vec![],
            }),
        );

        // Outer header
        let i_val = self.emit_with_result(
            outer_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp_outer = self.emit_with_result(
            outer_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: len1,
            },
        );
        self.emit(
            outer_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp_outer,
                then_target: outer_body_id,
                then_args: vec![],
                else_target: finalize_id,
                else_args: vec![],
            }),
        );

        // Outer body: load elem from set1
        let i_val2 = self.emit_with_result(
            outer_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot = self.emit_with_result(
            outer_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        let elem1 = self.load_at_dynamic_offset(outer_body, set1_ptr, slot);

        // Inner loop: search set2 for elem1
        let j_alloca = self.emit_with_result(
            outer_body,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            outer_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: j_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            outer_body,
            InstrNode::new(Inst::Br {
                target: inner_hdr_id,
                args: vec![],
            }),
        );

        // Inner header
        let j_val = self.emit_with_result(
            inner_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp_inner = self.emit_with_result(
            inner_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: j_val,
                rhs: len2,
            },
        );
        self.emit(
            inner_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp_inner,
                then_target: inner_body_id,
                then_args: vec![],
                else_target: not_found_id,
                else_args: vec![], // not found = include in result
            }),
        );

        // Inner body
        let j_val2 = self.emit_with_result(
            inner_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot2 = self.emit_with_result(
            inner_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: j_val2,
                rhs: one,
            },
        );
        let elem2 = self.load_at_dynamic_offset(inner_body, set2_ptr, slot2);
        let eq = self.emit_with_result(
            inner_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: elem1,
                rhs: elem2,
            },
        );
        self.emit(
            inner_body,
            InstrNode::new(Inst::CondBr {
                cond: eq,
                then_target: outer_inc_id,
                then_args: vec![], // found in set2 => skip
                else_target: inner_inc_id,
                else_args: vec![],
            }),
        );

        // Inner increment
        let j_val3 = self.emit_with_result(
            inner_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_j = self.emit_with_result(
            inner_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: j_val3,
                rhs: one,
            },
        );
        self.emit(
            inner_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: j_alloca,
                value: next_j,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            inner_inc,
            InstrNode::new(Inst::Br {
                target: inner_hdr_id,
                args: vec![],
            }),
        );

        // Not found in set2: add to result
        let cursor = self.emit_with_result(
            not_found,
            Inst::Load {
                ty: Ty::I64,
                ptr: cursor_alloca,
                align: None,
                volatile: false,
            },
        );
        self.store_at_dynamic_offset(not_found, result_ptr, cursor, elem1);
        let next_cursor = self.emit_with_result(
            not_found,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cursor,
                rhs: one,
            },
        );
        self.emit(
            not_found,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: cursor_alloca,
                value: next_cursor,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            not_found,
            InstrNode::new(Inst::Br {
                target: outer_inc_id,
                args: vec![],
            }),
        );

        // Outer increment
        let i_val3 = self.emit_with_result(
            outer_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_i = self.emit_with_result(
            outer_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val3,
                rhs: one,
            },
        );
        self.emit(
            outer_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            outer_inc,
            InstrNode::new(Inst::Br {
                target: outer_hdr_id,
                args: vec![],
            }),
        );

        // Finalize
        let final_cursor = self.emit_with_result(
            finalize,
            Inst::Load {
                ty: Ty::I64,
                ptr: cursor_alloca,
                align: None,
                volatile: false,
            },
        );
        let final_len = self.emit_with_result(
            finalize,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: final_cursor,
                rhs: one,
            },
        );
        self.store_at_offset(finalize, result_ptr, 0, final_len);
        self.store_reg_ptr(finalize, rd, result_ptr)?;

        Ok(Some(finalize))
    }

    /// Lower `r1 \subseteq r2` for RecordSetBitmask operands over a shared
    /// record universe. Returns `Ok(true)` when it handled the op (both
    /// operands are compatible record-set bitmasks), `Ok(false)` to fall
    /// through to the scalar/pointer paths.
    ///
    /// `L ⊆ R` iff every valid bit present in L is also present in R, i.e.
    /// there is no slot with a bit in L and not in R:
    ///   `OR_i ( L[i] & (R[i] XOR valid_mask[i]) ) == 0`
    /// where `R[i] XOR valid_mask[i]` is the complement of R within the valid
    /// universe. Byte-exact vs the interpreter: masks are canonicalized with
    /// the per-slot valid mask before the test, so out-of-universe bits (which
    /// never occur) cannot affect the result.
    fn lower_record_set_bitmask_subseteq(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<bool, TrustIrError> {
        let Some((universe_len, slot_count, universe)) =
            self.record_set_bitmask_binary_universe("Subseteq", r1, r2)?
        else {
            return Ok(false);
        };
        let left = self.record_set_bitmask_operand_slots(
            block_idx,
            r1,
            universe_len,
            slot_count,
            &universe,
            "Subseteq",
        )?;
        let right = self.record_set_bitmask_operand_slots(
            block_idx,
            r2,
            universe_len,
            slot_count,
            &universe,
            "Subseteq",
        )?;
        let mut missing: Option<trust_ir::value::ValueId> = None;
        for slot_index in 0..slot_count as usize {
            let valid_mask = super::record_set_bitmask_slot_valid_mask_ir(universe_len, slot_index)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Subseteq: RecordSetBitmask slot {slot_index} out of range for \
                         universe_len {universe_len}"
                    ))
                })?;
            let valid_mask_val = self.emit_i64_const(block_idx, valid_mask as i64);
            let l = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: left[slot_index],
                    rhs: valid_mask_val,
                },
            );
            let r = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: right[slot_index],
                    rhs: valid_mask_val,
                },
            );
            // complement of R within the valid universe = R XOR valid_mask
            let r_complement = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Xor,
                    ty: Ty::I64,
                    lhs: r,
                    rhs: valid_mask_val,
                },
            );
            // bits present in L but absent from R
            let slot_missing = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: l,
                    rhs: r_complement,
                },
            );
            missing = Some(match missing {
                None => slot_missing,
                Some(prev) => self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Or,
                        ty: Ty::I64,
                        lhs: prev,
                        rhs: slot_missing,
                    },
                ),
            });
        }
        let missing = missing.unwrap_or_else(|| self.emit_i64_const(block_idx, 0));
        let zero = self.emit_i64_const(block_idx, 0);
        let no_missing = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: missing,
                rhs: zero,
            },
        );
        let result = self.emit_bool_to_i64(block_idx, no_missing);
        self.store_reg_value(block_idx, rd, result)?;
        Ok(true)
    }

    /// Lower Subseteq { rd, r1, r2 }: test r1 \subseteq r2.
    ///
    /// For each element in r1, check if it exists in r2. If any element
    /// is not found, result is 0 (false). Otherwise 1 (true).
    pub(super) fn lower_subseteq(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        // RecordSetBitmask subseteq (RecordSetBitmask step 3/5): per-slot
        // L ⊆ R iff no valid bit is present in L and absent from R, i.e.
        // OR_i (L[i] & (R[i] XOR valid_mask[i])) == 0. Runs BEFORE the scalar
        // `compact_binary_set_universe` (which returns `None` for the multi-slot
        // record bitmask) and the pointer fall-through, so a RecordSetBitmask
        // operand is never read as a materialized set pointer.
        if self.lower_record_set_bitmask_subseteq(block_idx, rd, r1, r2)? {
            return Ok(Some(block_idx));
        }

        if let Some((universe_len, universe)) =
            self.compact_binary_set_universe("Subseteq", r1, r2)?
        {
            let (block_idx, left, left_in_universe) = self.emit_set_subseteq_operand_bitmask_i64(
                block_idx,
                r1,
                universe_len,
                &universe,
                "Subseteq",
            )?;
            let (block_idx, right, _right_in_universe) = self
                .emit_set_subseteq_operand_bitmask_i64(
                    block_idx,
                    r2,
                    universe_len,
                    &universe,
                    "Subseteq",
                )?;
            let valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, "Subseteq")?;
            let valid_mask_val = self.emit_i64_const(block_idx, valid_mask);
            let right_complement = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Xor,
                    ty: Ty::I64,
                    lhs: right,
                    rhs: valid_mask_val,
                },
            );
            let missing = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: left,
                    rhs: right_complement,
                },
            );
            let zero = self.emit_i64_const(block_idx, 0);
            let no_missing = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: missing,
                    rhs: zero,
                },
            );
            let no_missing_i64 = self.emit_bool_to_i64(block_idx, no_missing);
            let left_canonical =
                self.emit_compact_bitmask_canonical_i64(block_idx, left, universe_len, "Subseteq")?;
            let right_canonical = self.emit_compact_bitmask_canonical_i64(
                block_idx,
                right,
                universe_len,
                "Subseteq",
            )?;
            let canonical = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: left_canonical,
                    rhs: right_canonical,
                },
            );
            let present_subset = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: no_missing_i64,
                    rhs: left_in_universe,
                },
            );
            let result = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: present_subset,
                    rhs: canonical,
                },
            );
            self.store_reg_value(block_idx, rd, result)?;
            return Ok(Some(block_idx));
        }

        self.reject_lazy_set_operand("Subseteq", r1)?;
        self.reject_lazy_set_operand("Subseteq", r2)?;

        let set1_ptr = self.load_reg_as_ptr(block_idx, r1)?;
        let set2_ptr = self.load_reg_as_ptr(block_idx, r2)?;
        let len1 = self.load_at_offset(block_idx, set1_ptr, 0);
        let len2 = self.load_at_offset(block_idx, set2_ptr, 0);

        let zero = self.emit_i64_const(block_idx, 0);
        let one = self.emit_i64_const(block_idx, 1);

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

        let outer_hdr = self.new_aux_block("subseteq_outer_hdr");
        let outer_body = self.new_aux_block("subseteq_outer_body");
        let inner_hdr = self.new_aux_block("subseteq_inner_hdr");
        let inner_body = self.new_aux_block("subseteq_inner_body");
        let inner_inc = self.new_aux_block("subseteq_inner_inc");
        let not_found = self.new_aux_block("subseteq_not_found");
        let outer_inc = self.new_aux_block("subseteq_outer_inc");
        let result_true = self.new_aux_block("subseteq_true");

        let outer_hdr_id = self.block_id_of(outer_hdr);
        let outer_body_id = self.block_id_of(outer_body);
        let inner_hdr_id = self.block_id_of(inner_hdr);
        let inner_body_id = self.block_id_of(inner_body);
        let inner_inc_id = self.block_id_of(inner_inc);
        let not_found_id = self.block_id_of(not_found);
        let outer_inc_id = self.block_id_of(outer_inc);
        let result_true_id = self.block_id_of(result_true);

        self.emit(
            block_idx,
            InstrNode::new(Inst::Br {
                target: outer_hdr_id,
                args: vec![],
            }),
        );

        // Outer header: i < len1?
        let i_val = self.emit_with_result(
            outer_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp = self.emit_with_result(
            outer_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: i_val,
                rhs: len1,
            },
        );
        self.emit(
            outer_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp,
                then_target: outer_body_id,
                then_args: vec![],
                else_target: result_true_id,
                else_args: vec![],
            }),
        );

        // Outer body
        let i_val2 = self.emit_with_result(
            outer_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot = self.emit_with_result(
            outer_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val2,
                rhs: one,
            },
        );
        let elem1 = self.load_at_dynamic_offset(outer_body, set1_ptr, slot);

        let j_alloca = self.emit_with_result(
            outer_body,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            outer_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: j_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            outer_body,
            InstrNode::new(Inst::Br {
                target: inner_hdr_id,
                args: vec![],
            }),
        );

        // Inner header
        let j_val = self.emit_with_result(
            inner_hdr,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let cmp2 = self.emit_with_result(
            inner_hdr,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: j_val,
                rhs: len2,
            },
        );
        self.emit(
            inner_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp2,
                then_target: inner_body_id,
                then_args: vec![],
                else_target: not_found_id,
                else_args: vec![],
            }),
        );

        // Inner body
        let j_val2 = self.emit_with_result(
            inner_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let slot2 = self.emit_with_result(
            inner_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: j_val2,
                rhs: one,
            },
        );
        let elem2 = self.load_at_dynamic_offset(inner_body, set2_ptr, slot2);
        let eq = self.emit_with_result(
            inner_body,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: elem1,
                rhs: elem2,
            },
        );
        self.emit(
            inner_body,
            InstrNode::new(Inst::CondBr {
                cond: eq,
                then_target: outer_inc_id,
                then_args: vec![], // found
                else_target: inner_inc_id,
                else_args: vec![],
            }),
        );

        // Inner increment
        let j_val3 = self.emit_with_result(
            inner_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: j_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_j = self.emit_with_result(
            inner_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: j_val3,
                rhs: one,
            },
        );
        self.emit(
            inner_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: j_alloca,
                value: next_j,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            inner_inc,
            InstrNode::new(Inst::Br {
                target: inner_hdr_id,
                args: vec![],
            }),
        );

        // Not found: result is false
        self.store_reg_imm(not_found, rd, 0)?;
        // We need a merge block for the final result
        let merge = self.new_aux_block("subseteq_merge");
        let merge_id = self.block_id_of(merge);
        self.emit(
            not_found,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        // Outer increment
        let i_val3 = self.emit_with_result(
            outer_inc,
            Inst::Load {
                ty: Ty::I64,
                ptr: i_alloca,
                align: None,
                volatile: false,
            },
        );
        let next_i = self.emit_with_result(
            outer_inc,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: i_val3,
                rhs: one,
            },
        );
        self.emit(
            outer_inc,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            outer_inc,
            InstrNode::new(Inst::Br {
                target: outer_hdr_id,
                args: vec![],
            }),
        );

        // Result true
        self.store_reg_imm(result_true, rd, 1)?;
        self.emit(
            result_true,
            InstrNode::new(Inst::Br {
                target: merge_id,
                args: vec![],
            }),
        );

        Ok(Some(merge))
    }

    /// Lower Range { rd, lo, hi }: build the integer interval set lo..hi.
    ///
    /// Layout: slot[0] = max(hi - lo + 1, 0) (length), slot[1..=len] = lo, lo+1, ..., hi.
    pub(super) fn lower_range(
        &mut self,
        block_idx: usize,
        rd: u8,
        lo_reg: u8,
        hi_reg: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        let lo = self.load_reg(block_idx, lo_reg)?;
        let hi = self.load_reg(block_idx, hi_reg)?;

        let is_empty = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: hi,
                rhs: lo,
            },
        );
        let empty_block = self.new_aux_block("range_empty");
        let nonempty_block = self.new_aux_block("range_nonempty");
        let done_block = self.new_aux_block("range_done");

        let empty_id = self.block_id_of(empty_block);
        let nonempty_id = self.block_id_of(nonempty_block);
        let done_id = self.block_id_of(done_block);

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: is_empty,
                then_target: empty_id,
                then_args: vec![],
                else_target: nonempty_id,
                else_args: vec![],
            }),
        );

        let zero = self.emit_i64_const(empty_block, 0);
        let empty_ptr = self.alloc_aggregate(empty_block, 1);
        self.store_at_offset(empty_block, empty_ptr, 0, zero);
        self.store_reg_ptr(empty_block, rd, empty_ptr)?;
        self.emit(
            empty_block,
            InstrNode::new(Inst::Br {
                target: done_id,
                args: vec![],
            }),
        );

        let one = self.emit_i64_const(nonempty_block, 1);

        // len = hi - lo + 1
        let diff = self.emit_with_result(
            nonempty_block,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: hi,
                rhs: lo,
            },
        );
        let len = self.emit_with_result(
            nonempty_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: diff,
                rhs: one,
            },
        );

        // total slots = len + 1
        let total = self.emit_with_result(
            nonempty_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: len,
                rhs: one,
            },
        );
        let agg_ptr = if let (Some(lo_imm), Some(hi_imm)) =
            (self.scalar_of(lo_reg), self.scalar_of(hi_reg))
        {
            if hi_imm >= lo_imm {
                let total_slots = hi_imm
                    .checked_sub(lo_imm)
                    .and_then(|diff| diff.checked_add(2))
                    .and_then(|slots| u32::try_from(slots).ok());
                if let Some(total_slots) = total_slots {
                    self.alloc_aggregate(nonempty_block, total_slots)
                } else {
                    self.alloc_dynamic_i64_slots(nonempty_block, total)
                }
            } else {
                self.alloc_aggregate(nonempty_block, 1)
            }
        } else {
            self.alloc_dynamic_i64_slots(nonempty_block, total)
        };

        // Store length at slot 0
        self.store_at_offset(nonempty_block, agg_ptr, 0, len);

        // Fill loop: for i in 0..len, store lo+i at slot i+1
        let zero = self.emit_i64_const(nonempty_block, 0);
        let i_alloca = self.emit_with_result(
            nonempty_block,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            nonempty_block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: i_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let loop_hdr = self.new_aux_block("range_hdr");
        let loop_body = self.new_aux_block("range_body");
        let loop_done = self.new_aux_block("range_store");

        let hdr_id = self.block_id_of(loop_hdr);
        let body_id = self.block_id_of(loop_body);
        let loop_done_id = self.block_id_of(loop_done);

        self.emit(
            nonempty_block,
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
                rhs: len,
            },
        );
        self.emit(
            loop_hdr,
            InstrNode::new(Inst::CondBr {
                cond: cmp,
                then_target: body_id,
                then_args: vec![],
                else_target: loop_done_id,
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
        let elem = self.emit_with_result(
            loop_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: lo,
                rhs: i_val2,
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
        self.store_at_dynamic_offset(loop_body, agg_ptr, slot, elem);
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
        self.store_reg_ptr(loop_done, rd, agg_ptr)?;
        self.emit(
            loop_done,
            InstrNode::new(Inst::Br {
                target: done_id,
                args: vec![],
            }),
        );

        Ok(Some(done_block))
    }
}
