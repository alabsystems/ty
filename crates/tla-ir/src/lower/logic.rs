// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Comparison and boolean operation lowering: Eq, Neq, Lt, Le, Gt, Ge,
//! And, Or, Not, Implies, Equiv, CondMove.

use std::collections::BTreeSet;

use crate::TrustIrError;
use tla_jit_abi::JitRuntimeErrorKind;
use trust_ir::inst::*;
use trust_ir::ty::Ty;
use trust_ir::Constant;
use trust_ir::InstrNode;

use super::{AggregateShape, Ctx, ScalarShape, SequenceExtent, SetBitmaskUniverse};

impl<'cp> Ctx<'cp> {
    fn store_bool_reg(
        &mut self,
        block_idx: usize,
        rd: u8,
        value: trust_ir::value::ValueId,
    ) -> Result<(), TrustIrError> {
        self.store_reg_value(block_idx, rd, value)?;
        self.aggregate_shapes
            .insert(rd, AggregateShape::Scalar(ScalarShape::Bool));
        self.const_scalar_values.remove(&rd);
        self.const_set_sizes.remove(&rd);
        self.compact_state_slots.remove(&rd);
        Ok(())
    }

    pub(super) fn lower_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<usize, TrustIrError> {
        // A `TaggedScalarUnion` operand stores a universe INDEX, not a raw
        // scalar payload. It MUST be intercepted before the generic i64 fallback
        // (which would compare an index against a raw NameId/int and take the
        // WRONG branch → wrong successor). Handle it or fail closed.
        if let Some(done_blk) =
            self.lower_tagged_scalar_union_comparison(block_idx, rd, r1, r2, predicate)?
        {
            return Ok(done_blk);
        }
        if let Some(value) = self.scalar_equality_static_result(r1, r2, predicate) {
            let result = self.emit_i64_const(block_idx, i64::from(value));
            self.store_bool_reg(block_idx, rd, result)?;
            return Ok(block_idx);
        }
        if self.lower_set_bitmask_empty_comparison(block_idx, rd, r1, r2, predicate)? {
            return Ok(block_idx);
        }
        if self.lower_compact_set_bitmask_comparison(block_idx, rd, r1, r2, predicate)? {
            return Ok(block_idx);
        }
        if let Some(done_blk) =
            self.lower_tagged_scalar_or_set_finite_set_comparison(block_idx, rd, r1, r2, predicate)?
        {
            return Ok(done_blk);
        }
        if self.lower_exact_finite_int_set_comparison(block_idx, rd, r1, r2, predicate)? {
            return Ok(block_idx);
        }
        if self.lower_exact_finite_scalar_set_comparison(block_idx, rd, r1, r2, predicate)? {
            return Ok(block_idx);
        }
        if self.lower_sequence_comparison(block_idx, rd, r1, r2, predicate)? {
            return Ok(block_idx);
        }
        if self
            .lower_materialized_finite_set_emptiness_comparison(block_idx, rd, r1, r2, predicate)?
        {
            return Ok(block_idx);
        }
        if self.lower_record_set_bitmask_comparison(block_idx, rd, r1, r2, predicate)? {
            return Ok(block_idx);
        }
        self.reject_unhandled_finite_set_comparison(r1, r2, predicate)?;
        let lhs = self.load_reg(block_idx, r1)?;
        let rhs = self.load_reg(block_idx, r2)?;
        let cmp = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: predicate,
                ty: Ty::I64,
                lhs,
                rhs,
            },
        );
        // Zero-extend bool to i64 (0 or 1).
        let result = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: cmp,
            },
        );
        self.store_bool_reg(block_idx, rd, result)?;
        Ok(block_idx)
    }

    /// WP-05 item 2 read side: compare a tagged scalar-union slot (which stores a
    /// universe INDEX, not the raw scalar payload) for equality/inequality.
    ///
    /// * union `=`/`/=` a compile-time universe member -> integer INDEX compare
    ///   (the const is translated to its universe index); a const OUTSIDE the
    ///   universe folds to a constant verdict, because a well-typed union value
    ///   is always a universe member so it can never equal a non-member.
    /// * two identical-universe unions -> raw index compare (the index is
    ///   bijective over the shared universe, so index equality is value
    ///   equality).
    ///
    /// Returns `Ok(None)` only when NEITHER operand is a union (other passes
    /// handle it). Any other pairing against a union operand — an ordering
    /// predicate, a non-const scalar, a divergent universe — FAILS CLOSED so the
    /// raw index never reaches the generic `load_reg` + `ICmp` fall-through
    /// (which would compare an index against a differently-encoded value).
    fn lower_tagged_scalar_union_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<Option<usize>, TrustIrError> {
        let lhs = self.aggregate_shapes.get(&r1).cloned();
        let rhs = self.aggregate_shapes.get(&r2).cloned();
        let lhs_union = matches!(lhs, Some(AggregateShape::TaggedScalarUnion { .. }));
        let rhs_union = matches!(rhs, Some(AggregateShape::TaggedScalarUnion { .. }));
        if !lhs_union && !rhs_union {
            return Ok(None);
        }
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "ordering comparison over a tagged scalar-union operand is unsupported \
                 (its index space is unordered), got r{r1}={lhs:?}, r{r2}={rhs:?}"
            )));
        }

        // Both operands are same-universe unions: their indices are directly
        // comparable (index equality == value equality over a deduped universe).
        if let (
            Some(AggregateShape::TaggedScalarUnion { universe: u1, .. }),
            Some(AggregateShape::TaggedScalarUnion { universe: u2, .. }),
        ) = (&lhs, &rhs)
        {
            if u1 != u2 {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "tagged scalar-union equality requires identical universes, got r{r1} and r{r2} over different universes"
                )));
            }
            let a = self.load_reg(block_idx, r1)?;
            let b = self.load_reg(block_idx, r2)?;
            self.emit_index_equality(block_idx, rd, a, b, predicate)?;
            return Ok(Some(block_idx));
        }

        // Exactly one operand is a union; the other must be a compile-time member.
        let (union_reg, universe, other_reg) = if lhs_union {
            let Some(AggregateShape::TaggedScalarUnion { universe, .. }) = lhs else {
                unreachable!("lhs_union checked above")
            };
            (r1, universe, r2)
        } else {
            let Some(AggregateShape::TaggedScalarUnion { universe, .. }) = rhs else {
                unreachable!("rhs_union checked above")
            };
            (r2, universe, r1)
        };
        let Some(key) = self.const_scalar_domain_key_of(other_reg) else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "tagged scalar-union `=`/`/=` requires a compile-time universe member on the other side, got r{other_reg}"
            )));
        };
        match universe.iter().position(|element| *element == key) {
            Some(index) => {
                let index = i64::try_from(index).map_err(|_| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "tagged scalar-union index for r{union_reg} overflows i64"
                    ))
                })?;
                let union_index = self.load_reg(block_idx, union_reg)?;
                let const_index = self.emit_i64_const(block_idx, index);
                self.emit_index_equality(block_idx, rd, union_index, const_index, predicate)?;
                Ok(Some(block_idx))
            }
            None => {
                // A well-typed union value is always a universe member, so it can
                // never equal a non-member const: fold to a constant verdict.
                let equal = false;
                let value = match predicate {
                    ICmpOp::Eq => equal,
                    ICmpOp::Ne => !equal,
                    _ => unreachable!("filtered to Eq/Ne above"),
                };
                let result = self.emit_i64_const(block_idx, i64::from(value));
                self.store_bool_reg(block_idx, rd, result)?;
                Ok(Some(block_idx))
            }
        }
    }

    /// Emit `zext((lhs <pred> rhs) : bool -> i64)` into `rd` for two i64
    /// universe indices.
    fn emit_index_equality(
        &mut self,
        block_idx: usize,
        rd: u8,
        lhs: trust_ir::value::ValueId,
        rhs: trust_ir::value::ValueId,
        predicate: ICmpOp,
    ) -> Result<(), TrustIrError> {
        let cmp = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: predicate,
                ty: Ty::I64,
                lhs,
                rhs,
            },
        );
        let result = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: cmp,
            },
        );
        self.store_bool_reg(block_idx, rd, result)
    }

    /// Lower `X = {}` / `X /= {}` (and the symmetric forms) when one operand is a
    /// statically-empty set literal and the other operand's *emptiness* is
    /// statically or cheaply decidable — without any element-universe metadata.
    ///
    /// Emptiness is a property of a set's cardinality alone, so it needs neither
    /// the element universe nor lane (int/scalar) agreement between the two
    /// operands. Two cases are handled here:
    ///
    /// * **Compact `SetBitmask` vs empty:** the value is `{}` iff every bit of
    ///   its raw i64 mask slot is clear, i.e. `mask == 0`. The empty set encodes
    ///   canonically as mask `0` regardless of `universe_len`, so this is sound
    ///   even when the operand's universe is [`SetBitmaskUniverse::Unknown`]
    ///   (e.g. a function-range `SUBSET` slot whose element universe was dropped
    ///   by the materialized-set layout). This must run *before*
    ///   [`Self::lower_compact_set_bitmask_comparison`], which requires exact
    ///   universe metadata and would otherwise reject an `Unknown`-universe
    ///   operand via [`Self::compact_binary_set_universe`].
    /// * **Statically-sized exact set vs empty:** when the other operand has an
    ///   exactly-known element count (`ExactIntSet` / `ExactScalarSet` /
    ///   `Interval` / materialized `Set`), the comparison folds to a compile-time
    ///   constant. This decides cross-lane literals such as
    ///   `{m1} /= {}` (an `ExactScalarSet{ModelValue}` vs the empty-int `{}`
    ///   literal `ExactIntSet{[]}`) that the lane-specific exact-set passes below
    ///   reject for mismatched element lanes.
    fn lower_set_bitmask_empty_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<bool, TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(false);
        }
        let lhs_shape = self.aggregate_shapes.get(&r1).cloned();
        let rhs_shape = self.aggregate_shapes.get(&r2).cloned();
        let (Some(lhs_shape), Some(rhs_shape)) = (lhs_shape, rhs_shape) else {
            return Ok(false);
        };

        // Identify the statically-empty operand and bind the other ("subject").
        let (subject_reg, subject_shape) = if super::is_exact_empty_set_shape(&rhs_shape) {
            (r1, &lhs_shape)
        } else if super::is_exact_empty_set_shape(&lhs_shape) {
            (r2, &rhs_shape)
        } else {
            return Ok(false);
        };

        // Case 1: the subject is a statically-sized exact set whose emptiness is
        // known at compile time. Fold to a constant `(empty == empty)` verdict.
        if let Some(subject_nonempty) = Self::static_exact_set_is_nonempty(subject_shape) {
            // `subject = {}` is true iff the subject is also empty.
            let equal = !subject_nonempty;
            let value = match predicate {
                ICmpOp::Eq => equal,
                ICmpOp::Ne => !equal,
                _ => unreachable!("caller filters to Eq/Ne"),
            };
            let result = self.emit_i64_const(block_idx, i64::from(value));
            self.store_bool_reg(block_idx, rd, result)?;
            return Ok(true);
        }

        // Case 2: the subject is a compact SetBitmask. Its raw i64 mask slot is
        // loaded directly by `load_reg`; `{}` iff mask == 0.
        if matches!(subject_shape, AggregateShape::SetBitmask { .. }) {
            let mask = self.load_reg(block_idx, subject_reg)?;
            let zero = self.emit_i64_const(block_idx, 0);
            let cmp = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: predicate,
                    ty: Ty::I64,
                    lhs: mask,
                    rhs: zero,
                },
            );
            let result = self.emit_logic_bool_to_i64(block_idx, cmp);
            self.store_bool_reg(block_idx, rd, result)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Returns `Some(true)` / `Some(false)` when a set shape has an exactly-known
    /// element count (non-empty / empty respectively), or `None` when the count
    /// is not statically exact (e.g. a `BoundedSet`/`SetBitmask` whose tracked
    /// length is only a universe *bound*, not the actual cardinality).
    fn static_exact_set_is_nonempty(shape: &AggregateShape) -> Option<bool> {
        match shape {
            AggregateShape::ExactIntSet { values } => Some(!values.is_empty()),
            AggregateShape::ExactScalarSet { values, .. } => Some(!values.is_empty()),
            AggregateShape::Interval { lo, hi } => Some(hi >= lo),
            AggregateShape::Set { len, .. } => Some(*len > 0),
            _ => None,
        }
    }

    fn lower_compact_set_bitmask_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<bool, TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(false);
        }

        let Some((universe_len, universe)) =
            self.compact_binary_set_universe("Set equality", r1, r2)?
        else {
            return Ok(false);
        };

        let (block_idx, left, left_in_universe) = self.emit_set_subseteq_operand_bitmask_i64(
            block_idx,
            r1,
            universe_len,
            &universe,
            "Set equality",
        )?;
        let (block_idx, right, right_in_universe) = self.emit_set_subseteq_operand_bitmask_i64(
            block_idx,
            r2,
            universe_len,
            &universe,
            "Set equality",
        )?;

        self.lower_set_bitmask_mask_comparison(
            block_idx,
            rd,
            predicate,
            universe_len,
            left,
            left_in_universe,
            right,
            right_in_universe,
        )?;
        Ok(true)
    }

    fn lower_tagged_scalar_or_set_finite_set_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<Option<usize>, TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(None);
        }

        let lhs_shape = self.aggregate_shapes.get(&r1).cloned();
        let rhs_shape = self.aggregate_shapes.get(&r2).cloned();
        let (tagged_reg, finite_reg, tagged_shape) = match (&lhs_shape, &rhs_shape) {
            (Some(shape @ AggregateShape::TaggedScalarOrSet { .. }), Some(finite))
                if Self::is_exact_finite_set_shape_for_tagged_comparison(finite) =>
            {
                (r1, r2, shape.clone())
            }
            (Some(finite), Some(shape @ AggregateShape::TaggedScalarOrSet { .. }))
                if Self::is_exact_finite_set_shape_for_tagged_comparison(finite) =>
            {
                (r2, r1, shape.clone())
            }
            _ => return Ok(None),
        };

        let Some((universe_len, universe)) = tagged_shape.tagged_set_branch_universe() else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Set equality: tagged scalar-or-set r{tagged_reg} requires exact universe metadata"
            )));
        };

        let (block_idx, finite_mask, finite_in_universe) = self
            .emit_set_subseteq_operand_bitmask_i64(
                block_idx,
                finite_reg,
                universe_len,
                &universe,
                "Set equality",
            )?;
        let tagged_raw = self.load_reg(block_idx, tagged_reg)?;
        let zero = self.emit_i64_const(block_idx, 0);
        let is_set_branch = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: tagged_raw,
                rhs: zero,
            },
        );

        let set_blk = self.new_aux_block("tagged_scalar_or_set_eq_set");
        let scalar_blk = self.new_aux_block("tagged_scalar_or_set_eq_scalar");
        let error_blk = self.new_aux_block("tagged_scalar_or_set_eq_type_error");
        let done_blk = self.new_aux_block("tagged_scalar_or_set_eq_done");
        let done_value = self.add_block_param(done_blk, Ty::I64);

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: is_set_branch,
                then_target: self.block_id_of(set_blk),
                then_args: vec![],
                else_target: self.block_id_of(scalar_blk),
                else_args: vec![],
            }),
        );

        let scalar_result = self.emit_i64_const(
            scalar_blk,
            match predicate {
                ICmpOp::Eq => 0,
                ICmpOp::Ne => 1,
                _ => unreachable!("caller filters to Eq/Ne"),
            },
        );
        self.emit(
            scalar_blk,
            InstrNode::new(Inst::Br {
                target: self.block_id_of(done_blk),
                args: vec![scalar_result],
            }),
        );

        let neg_one = self.emit_i64_const(set_blk, -1);
        let tagged_mask = self.emit_with_result(
            set_blk,
            Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: neg_one,
                rhs: tagged_raw,
            },
        );
        let canonical = self.emit_compact_bitmask_canonical_i64(
            set_blk,
            tagged_mask,
            universe_len,
            "Set equality",
        )?;
        let canonical_bool = self.emit_with_result(
            set_blk,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: canonical,
                rhs: zero,
            },
        );
        let compare_blk = self.new_aux_block("tagged_scalar_or_set_eq_compare");
        self.emit(
            set_blk,
            InstrNode::new(Inst::CondBr {
                cond: canonical_bool,
                then_target: self.block_id_of(compare_blk),
                then_args: vec![],
                else_target: self.block_id_of(error_blk),
                else_args: vec![],
            }),
        );

        let same_mask = self.emit_with_result(
            compare_blk,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: tagged_mask,
                rhs: finite_mask,
            },
        );
        let same_mask = self.emit_logic_bool_to_i64(compare_blk, same_mask);
        let set_result = match predicate {
            ICmpOp::Eq => self.emit_with_result(
                compare_blk,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: same_mask,
                    rhs: finite_in_universe,
                },
            ),
            ICmpOp::Ne => {
                let masks_differ = self.emit_with_result(
                    compare_blk,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: same_mask,
                        rhs: zero,
                    },
                );
                let masks_differ = self.emit_logic_bool_to_i64(compare_blk, masks_differ);
                let outside_universe = self.emit_with_result(
                    compare_blk,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: finite_in_universe,
                        rhs: zero,
                    },
                );
                let outside_universe = self.emit_logic_bool_to_i64(compare_blk, outside_universe);
                self.emit_with_result(
                    compare_blk,
                    Inst::BinOp {
                        op: BinOp::Or,
                        ty: Ty::I64,
                        lhs: masks_differ,
                        rhs: outside_universe,
                    },
                )
            }
            _ => unreachable!("caller filters to Eq/Ne"),
        };
        self.emit(
            compare_blk,
            InstrNode::new(Inst::Br {
                target: self.block_id_of(done_blk),
                args: vec![set_result],
            }),
        );

        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);
        self.store_bool_reg(done_blk, rd, done_value)?;
        // Control flow resumes in `done_blk`; the caller MUST continue
        // emitting there (returning only "handled" left `done_blk`
        // unterminated and the continuation dead — an emitted infinite loop).
        Ok(Some(done_blk))
    }

    fn is_exact_finite_set_shape_for_tagged_comparison(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Set { len: 0, .. }
                | AggregateShape::ExactIntSet { .. }
                | AggregateShape::ExactScalarSet { .. }
                | AggregateShape::Interval { .. }
        )
    }

    fn lower_exact_finite_int_set_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<bool, TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(false);
        }

        let lhs_shape = self.aggregate_shapes.get(&r1).cloned();
        let rhs_shape = self.aggregate_shapes.get(&r2).cloned();
        let (Some(lhs_shape), Some(rhs_shape)) = (lhs_shape, rhs_shape) else {
            return Ok(false);
        };
        if matches!(
            (&lhs_shape, &rhs_shape),
            (AggregateShape::SetBitmask { .. }, _) | (_, AggregateShape::SetBitmask { .. })
        ) {
            return Ok(false);
        }
        if !Self::is_exact_finite_int_set_shape(&lhs_shape)
            || !Self::is_exact_finite_int_set_shape(&rhs_shape)
        {
            return Ok(false);
        }

        let (universe_len, universe) =
            Self::synthesized_exact_int_set_equality_universe(&lhs_shape, &rhs_shape)?;
        let (block_idx, left, left_in_universe) = self.emit_set_subseteq_operand_bitmask_i64(
            block_idx,
            r1,
            universe_len,
            &universe,
            "Set equality",
        )?;
        let (block_idx, right, right_in_universe) = self.emit_set_subseteq_operand_bitmask_i64(
            block_idx,
            r2,
            universe_len,
            &universe,
            "Set equality",
        )?;

        self.lower_set_bitmask_mask_comparison(
            block_idx,
            rd,
            predicate,
            universe_len,
            left,
            left_in_universe,
            right,
            right_in_universe,
        )?;
        Ok(true)
    }

    fn lower_exact_finite_scalar_set_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<bool, TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(false);
        }

        let lhs_shape = self.aggregate_shapes.get(&r1).cloned();
        let rhs_shape = self.aggregate_shapes.get(&r2).cloned();
        let (Some(lhs_shape), Some(rhs_shape)) = (lhs_shape, rhs_shape) else {
            return Ok(false);
        };
        if matches!(
            (&lhs_shape, &rhs_shape),
            (AggregateShape::SetBitmask { .. }, _) | (_, AggregateShape::SetBitmask { .. })
        ) {
            return Ok(false);
        }
        if !Self::is_exact_finite_scalar_set_shape(&lhs_shape)
            || !Self::is_exact_finite_scalar_set_shape(&rhs_shape)
        {
            return Ok(false);
        }

        if let Some(value) =
            Self::exact_scalar_set_lane_mismatch_static_result(&lhs_shape, &rhs_shape, predicate)
        {
            let result = self.emit_i64_const(block_idx, i64::from(value));
            self.store_bool_reg(block_idx, rd, result)?;
            return Ok(true);
        }

        let (universe_len, universe) =
            Self::synthesized_exact_scalar_set_equality_universe(&lhs_shape, &rhs_shape)?;
        let (block_idx, left, left_in_universe) = self.emit_set_subseteq_operand_bitmask_i64(
            block_idx,
            r1,
            universe_len,
            &universe,
            "Set equality",
        )?;
        let (block_idx, right, right_in_universe) = self.emit_set_subseteq_operand_bitmask_i64(
            block_idx,
            r2,
            universe_len,
            &universe,
            "Set equality",
        )?;

        self.lower_set_bitmask_mask_comparison(
            block_idx,
            rd,
            predicate,
            universe_len,
            left,
            left_in_universe,
            right,
            right_in_universe,
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_set_bitmask_mask_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        predicate: ICmpOp,
        universe_len: u32,
        left: trust_ir::value::ValueId,
        left_in_universe: trust_ir::value::ValueId,
        right: trust_ir::value::ValueId,
        right_in_universe: trust_ir::value::ValueId,
    ) -> Result<(), TrustIrError> {
        let same_mask = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: left,
                rhs: right,
            },
        );
        let same_mask = self.emit_logic_bool_to_i64(block_idx, same_mask);
        let both_in_universe = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: left_in_universe,
                rhs: right_in_universe,
            },
        );
        let left_canonical =
            self.emit_compact_bitmask_canonical_i64(block_idx, left, universe_len, "Set equality")?;
        let right_canonical = self.emit_compact_bitmask_canonical_i64(
            block_idx,
            right,
            universe_len,
            "Set equality",
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

        let result = match predicate {
            ICmpOp::Eq => {
                let equal_core = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: same_mask,
                        rhs: both_in_universe,
                    },
                );
                self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: equal_core,
                        rhs: canonical,
                    },
                )
            }
            ICmpOp::Ne => {
                let zero = self.emit_i64_const(block_idx, 0);
                let masks_differ = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: same_mask,
                        rhs: zero,
                    },
                );
                let masks_differ = self.emit_logic_bool_to_i64(block_idx, masks_differ);
                let outside_universe = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: both_in_universe,
                        rhs: zero,
                    },
                );
                let outside_universe = self.emit_logic_bool_to_i64(block_idx, outside_universe);
                let not_equal_core = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Or,
                        ty: Ty::I64,
                        lhs: masks_differ,
                        rhs: outside_universe,
                    },
                );
                self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: not_equal_core,
                        rhs: canonical,
                    },
                )
            }
            _ => unreachable!("caller filters to Eq/Ne"),
        };

        self.store_bool_reg(block_idx, rd, result)?;
        Ok(())
    }

    fn is_exact_finite_int_set_shape(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::ExactIntSet { .. }
                | AggregateShape::Interval { .. }
                | AggregateShape::Set { len: 0, .. }
        )
    }

    fn is_exact_finite_scalar_set_shape(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::ExactScalarSet { .. } | AggregateShape::Set { len: 0, .. }
        )
    }

    fn exact_scalar_set_lane_mismatch_static_result(
        lhs_shape: &AggregateShape,
        rhs_shape: &AggregateShape,
        predicate: ICmpOp,
    ) -> Option<bool> {
        let (
            AggregateShape::ExactScalarSet {
                scalar: lhs_scalar,
                values: lhs_values,
            },
            AggregateShape::ExactScalarSet {
                scalar: rhs_scalar,
                values: rhs_values,
            },
        ) = (lhs_shape, rhs_shape)
        else {
            return None;
        };
        if lhs_scalar == rhs_scalar {
            return None;
        }

        let equal = lhs_values.is_empty() && rhs_values.is_empty();
        match predicate {
            ICmpOp::Eq => Some(equal),
            ICmpOp::Ne => Some(!equal),
            _ => None,
        }
    }

    fn synthesized_exact_scalar_set_equality_universe(
        lhs_shape: &AggregateShape,
        rhs_shape: &AggregateShape,
    ) -> Result<(u32, SetBitmaskUniverse), TrustIrError> {
        let mut scalar = None;
        let mut values = BTreeSet::new();
        Self::collect_exact_scalar_set_values_for_equality(lhs_shape, &mut scalar, &mut values)?;
        Self::collect_exact_scalar_set_values_for_equality(rhs_shape, &mut scalar, &mut values)?;

        let scalar = scalar.ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "Set equality: exact scalar set universe requires a scalar lane".to_owned(),
            )
        })?;
        let values = values.into_iter().collect::<Vec<_>>();
        super::exact_scalar_set_bitmask_universe(&scalar, &values).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "Set equality: exact scalar set values cannot be represented as a compact bitmask universe"
                    .to_owned(),
            )
        })
    }

    fn collect_exact_scalar_set_values_for_equality(
        shape: &AggregateShape,
        scalar: &mut Option<ScalarShape>,
        values: &mut BTreeSet<i64>,
    ) -> Result<(), TrustIrError> {
        match shape {
            AggregateShape::ExactScalarSet {
                scalar: exact_scalar,
                values: exact_values,
            } => {
                if scalar
                    .as_ref()
                    .is_some_and(|existing| existing != exact_scalar)
                {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Set equality: exact scalar set operands use incompatible scalar lanes {scalar:?} and {exact_scalar:?}"
                    )));
                }
                *scalar = Some(exact_scalar.clone());
                for value in exact_values {
                    values.insert(*value);
                    if values.len() > 63 {
                        return Err(TrustIrError::UnsupportedOpcode(
                            "Set equality: exact scalar set universe exceeds i64 bitmask capacity"
                                .to_owned(),
                        ));
                    }
                }
                Ok(())
            }
            AggregateShape::Set { len: 0, .. } => Ok(()),
            _ => Err(TrustIrError::UnsupportedOpcode(format!(
                "Set equality: unsupported exact scalar set shape {shape:?}"
            ))),
        }
    }

    fn synthesized_exact_int_set_equality_universe(
        lhs_shape: &AggregateShape,
        rhs_shape: &AggregateShape,
    ) -> Result<(u32, SetBitmaskUniverse), TrustIrError> {
        let mut values = BTreeSet::new();
        Self::collect_exact_int_set_values_for_equality(lhs_shape, &mut values)?;
        Self::collect_exact_int_set_values_for_equality(rhs_shape, &mut values)?;
        let universe_len = u32::try_from(values.len()).map_err(|_| {
            TrustIrError::UnsupportedOpcode(
                "Set equality: exact integer set universe is too large".to_owned(),
            )
        })?;
        Self::compact_set_bitmask_valid_mask(universe_len, "Set equality")?;

        let values = values.into_iter().collect::<Vec<_>>();
        let universe = match super::dense_ordered_i64_values_lo(&values) {
            Some((lo, dense_len)) if dense_len == universe_len => {
                SetBitmaskUniverse::IntRange { lo }
            }
            _ => SetBitmaskUniverse::ExplicitInt(values),
        };
        Ok((universe_len, universe))
    }

    fn collect_exact_int_set_values_for_equality(
        shape: &AggregateShape,
        values: &mut BTreeSet<i64>,
    ) -> Result<(), TrustIrError> {
        match shape {
            AggregateShape::ExactIntSet {
                values: exact_values,
            } => {
                for value in exact_values {
                    values.insert(*value);
                    if values.len() > 63 {
                        return Err(TrustIrError::UnsupportedOpcode(
                            "Set equality: exact integer set universe exceeds i64 bitmask capacity"
                                .to_owned(),
                        ));
                    }
                }
                Ok(())
            }
            AggregateShape::Interval { lo, hi } => {
                let len = super::interval_len_u32(*lo, *hi).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Set equality: exact integer interval {lo}..{hi} length overflows"
                    ))
                })?;
                if len > 63 {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Set equality: exact integer interval {lo}..{hi} exceeds i64 bitmask capacity"
                    )));
                }
                for offset in 0..len {
                    let value = (*lo).checked_add(i64::from(offset)).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "Set equality: exact integer interval {lo}..{hi} element overflows"
                        ))
                    })?;
                    values.insert(value);
                }
                Ok(())
            }
            AggregateShape::Set { len: 0, .. } => Ok(()),
            _ => Err(TrustIrError::UnsupportedOpcode(format!(
                "Set equality: unsupported exact integer set shape {shape:?}"
            ))),
        }
    }

    /// Resolve one operand of a RecordSetBitmask comparison to its `slot_count`
    /// mask slots over the shared record `universe`. A statically-empty operand
    /// (`{}`) has no present records, so it resolves to all-zero slots; every
    /// other operand routes through
    /// [`Self::record_set_bitmask_operand_slots`] (the RecordSetBitmask state
    /// var / prior bitmask result / tracked record-set literal), which fails
    /// closed for anything it cannot byte-exactly reconstruct.
    fn record_set_bitmask_comparison_operand_slots(
        &mut self,
        block_idx: usize,
        reg: u8,
        universe_len: u32,
        slot_count: u32,
        universe: &[super::RecordBitKey],
    ) -> Result<Vec<trust_ir::value::ValueId>, TrustIrError> {
        if self
            .aggregate_shapes
            .get(&reg)
            .is_some_and(super::is_exact_empty_set_shape)
        {
            let zeros = (0..slot_count)
                .map(|_| self.emit_i64_const(block_idx, 0))
                .collect();
            return Ok(zeros);
        }
        self.record_set_bitmask_operand_slots(
            block_idx,
            reg,
            universe_len,
            slot_count,
            universe,
            "Set equality",
        )
    }

    /// Lower `S = T` / `S # T` when at least one operand is a native
    /// RecordSetBitmask (RecordSetBitmask step 3/5). Two record-sets over the
    /// SAME record universe are equal iff their multi-slot masks are bit-equal,
    /// so the verdict is `(OR_i (left[i] XOR right[i])) == 0`. Each operand's
    /// slots are resolved over the shared universe and AND-ed with the per-slot
    /// valid mask (defensive; the region is stored canonical) so out-of-universe
    /// bits can never make two logically-equal sets compare unequal. Returns
    /// `Ok(false)` when neither operand is a RecordSetBitmask (the caller's
    /// reject path then runs); fails closed (`Err`) when a RecordSetBitmask
    /// operand cannot be resolved, routing the whole action to the interpreter.
    fn lower_record_set_bitmask_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<bool, TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(false);
        }
        let Some((universe_len, slot_count, universe)) =
            self.record_set_bitmask_binary_universe("Set equality", r1, r2)?
        else {
            return Ok(false);
        };
        let left = self.record_set_bitmask_comparison_operand_slots(
            block_idx,
            r1,
            universe_len,
            slot_count,
            &universe,
        )?;
        let right = self.record_set_bitmask_comparison_operand_slots(
            block_idx,
            r2,
            universe_len,
            slot_count,
            &universe,
        )?;
        let mut diff: Option<trust_ir::value::ValueId> = None;
        for slot_index in 0..slot_count as usize {
            let valid_mask = super::record_set_bitmask_slot_valid_mask_ir(universe_len, slot_index)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Set equality: RecordSetBitmask slot {slot_index} out of range for \
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
            let xor = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Xor,
                    ty: Ty::I64,
                    lhs: l,
                    rhs: r,
                },
            );
            diff = Some(match diff {
                None => xor,
                Some(prev) => self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Or,
                        ty: Ty::I64,
                        lhs: prev,
                        rhs: xor,
                    },
                ),
            });
        }
        let diff = diff.unwrap_or_else(|| self.emit_i64_const(block_idx, 0));
        let zero = self.emit_i64_const(block_idx, 0);
        let cmp = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: predicate,
                ty: Ty::I64,
                lhs: diff,
                rhs: zero,
            },
        );
        let result = self.emit_logic_bool_to_i64(block_idx, cmp);
        self.store_bool_reg(block_idx, rd, result)?;
        Ok(true)
    }

    fn reject_unhandled_finite_set_comparison(
        &self,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<(), TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(());
        }
        let lhs_shape = self.aggregate_shapes.get(&r1);
        let rhs_shape = self.aggregate_shapes.get(&r2);
        // RecordSetBitmask is its own category (neither `is_finite_set_shape`
        // nor `is_lazy_set_shape`), so it would otherwise slip past the
        // finite-set check below and reach the raw `load_reg` + `ICmp`
        // fall-through — comparing the two operands' POINTER-ints, not their
        // mask contents (two logically-equal record-sets at different
        // allocations would compare unequal). Native multi-slot record-set-
        // bitmask equality is not wired (RecordSetBitmask step 3/5); fail closed
        // so set equality routes to the interpreter oracle.
        if matches!(lhs_shape, Some(AggregateShape::RecordSetBitmask { .. }))
            || matches!(rhs_shape, Some(AggregateShape::RecordSetBitmask { .. }))
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Set equality over a RecordSetBitmask operand requires native multi-slot mask \
                 comparison (not yet wired), got r{r1}={lhs_shape:?}, r{r2}={rhs_shape:?}"
            )));
        }
        // A LazyUnion register holds an inert placeholder (0), never a value
        // encoding (soundness amendment H1). The raw `load_reg` + `ICmp`
        // fall-through below would compare placeholders (always equal) —
        // unsound. LazyUnion is not a finite-set shape, so it would slip past
        // the finite-set check; fail closed explicitly.
        if matches!(lhs_shape, Some(AggregateShape::LazyUnion { .. }))
            || matches!(rhs_shape, Some(AggregateShape::LazyUnion { .. }))
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Set equality over a lazy-union operand has no runtime encoding to compare \
                 (LazyUnion is membership-only), got r{r1}={lhs_shape:?}, r{r2}={rhs_shape:?}"
            )));
        }
        if lhs_shape.is_some_and(AggregateShape::is_finite_set_shape)
            || rhs_shape.is_some_and(AggregateShape::is_finite_set_shape)
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Set equality requires SetBitmask operands or compatible exact finite integer/scalar set shapes, got r{r1}={lhs_shape:?}, r{r2}={rhs_shape:?}"
            )));
        }
        Ok(())
    }

    /// Lower `S = {}` / `S /= {}` (and the symmetric forms) when one operand is
    /// a statically-empty set and the other is a *materialized*, pointer-backed
    /// finite set (`Set` / `BoundedSet`).
    ///
    /// Emptiness is a property of the set's cardinality alone — it needs no
    /// element universe — so this is sound for any materialized finite set
    /// regardless of element type. The materialized set ABI stores the element
    /// count in slot 0, so the comparison reduces to `len == 0` (`=`) or
    /// `len != 0` (`/=`). Without this, subset-builder results such as
    /// `{p \in Person : ...} /= {}` fall through to
    /// `reject_unhandled_finite_set_comparison` and force the whole action to
    /// the interpreter.
    ///
    /// Compact `SetBitmask` / `TaggedScalarOrSet` operands are deliberately
    /// excluded: they are raw one-slot values (not aggregate pointers) and are
    /// already handled by the mask-native comparison passes above — including
    /// the universe-free empty case in
    /// [`Self::lower_set_bitmask_empty_comparison`], which decides a compact
    /// `SetBitmask` `={}`/`#{}` directly as `mask == 0` even when the operand's
    /// universe is `Unknown`. Exact finite shapes (`ExactIntSet` /
    /// `ExactScalarSet` / `Interval`) are likewise handled earlier; here they
    /// only appear as the statically-empty operand.
    fn lower_materialized_finite_set_emptiness_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<bool, TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(false);
        }
        let lhs_shape = self.aggregate_shapes.get(&r1).cloned();
        let rhs_shape = self.aggregate_shapes.get(&r2).cloned();
        let (Some(lhs_shape), Some(rhs_shape)) = (lhs_shape, rhs_shape) else {
            return Ok(false);
        };

        // Identify the materialized (pointer-backed) finite-set operand and
        // require the other operand to be a statically-empty set literal.
        let materialized_reg = if Self::is_materialized_pointer_finite_set_shape(&lhs_shape)
            && super::is_exact_empty_set_shape(&rhs_shape)
        {
            r1
        } else if Self::is_materialized_pointer_finite_set_shape(&rhs_shape)
            && super::is_exact_empty_set_shape(&lhs_shape)
        {
            r2
        } else {
            return Ok(false);
        };

        let materialized_shape = self
            .aggregate_shapes
            .get(&materialized_reg)
            .cloned()
            .expect("materialized operand shape was just matched");

        // A statically-empty materialized set (`Set { len: 0, .. }`) collapses
        // to a constant: empty == empty.
        if super::is_exact_empty_set_shape(&materialized_shape) {
            let value = i64::from(matches!(predicate, ICmpOp::Eq));
            let result = self.emit_i64_const(block_idx, value);
            self.store_bool_reg(block_idx, rd, result)?;
            return Ok(true);
        }

        // General case: compare the element-count header (slot 0) against zero.
        let set_ptr = self.load_reg_as_ptr(block_idx, materialized_reg)?;
        let len = self.load_at_offset(block_idx, set_ptr, 0);
        let zero = self.emit_i64_const(block_idx, 0);
        let cmp = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: predicate,
                ty: Ty::I64,
                lhs: len,
                rhs: zero,
            },
        );
        let result = self.emit_logic_bool_to_i64(block_idx, cmp);
        self.store_bool_reg(block_idx, rd, result)?;
        Ok(true)
    }

    /// True for finite-set shapes whose runtime value is a pointer to a
    /// materialized self-describing set buffer (element count in slot 0). These
    /// are the shapes `load_reg_as_ptr` accepts for set operands.
    fn is_materialized_pointer_finite_set_shape(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Set { .. } | AggregateShape::BoundedSet { .. }
        )
    }

    fn lower_sequence_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        predicate: ICmpOp,
    ) -> Result<bool, TrustIrError> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return Ok(false);
        }
        let lhs_shape = self.aggregate_shapes.get(&r1).cloned();
        let rhs_shape = self.aggregate_shapes.get(&r2).cloned();
        let lhs_is_sequence = matches!(lhs_shape, Some(AggregateShape::Sequence { .. }));
        let rhs_is_sequence = matches!(rhs_shape, Some(AggregateShape::Sequence { .. }));
        if !lhs_is_sequence && !rhs_is_sequence {
            return Ok(false);
        }
        if lhs_is_sequence != rhs_is_sequence {
            if lhs_shape
                .as_ref()
                .zip(rhs_shape.as_ref())
                .is_some_and(|(lhs, rhs)| {
                    !matches!(lhs, AggregateShape::StateValue)
                        && !matches!(rhs, AggregateShape::StateValue)
                })
            {
                let value = match predicate {
                    ICmpOp::Eq => 0,
                    ICmpOp::Ne => 1,
                    _ => unreachable!("caller filters to Eq/Ne"),
                };
                let result = self.emit_i64_const(block_idx, value);
                self.store_bool_reg(block_idx, rd, result)?;
                return Ok(true);
            }
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "sequence equality requires both operands to have tracked sequence shapes, got r{r1}={lhs_shape:?}, r{r2}={rhs_shape:?}"
            )));
        }

        let (
            Some(
                lhs_shape @ AggregateShape::Sequence {
                    extent: lhs_extent, ..
                },
            ),
            Some(
                rhs_shape @ AggregateShape::Sequence {
                    extent: rhs_extent, ..
                },
            ),
        ) = (lhs_shape.clone(), rhs_shape.clone())
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "sequence equality requires tracked sequence shapes, got r{r1}={lhs_shape:?}, r{r2}={rhs_shape:?}"
            )));
        };

        let lhs_empty = lhs_extent.exact_count() == Some(0);
        let rhs_empty = rhs_extent.exact_count() == Some(0);
        let result = match (lhs_empty, rhs_empty) {
            (true, true) => {
                let value = match predicate {
                    ICmpOp::Eq => 1,
                    ICmpOp::Ne => 0,
                    _ => unreachable!("caller filters to Eq/Ne"),
                };
                self.emit_i64_const(block_idx, value)
            }
            (false, true) => {
                self.emit_sequence_len_comparison(block_idx, r1, &lhs_shape, predicate)?
            }
            (true, false) => {
                self.emit_sequence_len_comparison(block_idx, r2, &rhs_shape, predicate)?
            }
            (false, false) => {
                return self.lower_structural_sequence_comparison(
                    block_idx, rd, r1, &lhs_shape, r2, &rhs_shape, predicate,
                );
            }
        };
        self.store_bool_reg(block_idx, rd, result)?;
        Ok(true)
    }

    fn lower_structural_sequence_comparison(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        lhs_shape: &AggregateShape,
        r2: u8,
        rhs_shape: &AggregateShape,
        predicate: ICmpOp,
    ) -> Result<bool, TrustIrError> {
        let (
            AggregateShape::Sequence {
                extent: lhs_extent, ..
            },
            AggregateShape::Sequence {
                extent: rhs_extent, ..
            },
        ) = (lhs_shape, rhs_shape)
        else {
            return Ok(false);
        };

        if let (Some(lhs_len), Some(rhs_len)) = (lhs_extent.exact_count(), rhs_extent.exact_count())
        {
            if lhs_len != rhs_len {
                let value = match predicate {
                    ICmpOp::Eq => 0,
                    ICmpOp::Ne => 1,
                    _ => unreachable!("caller filters to Eq/Ne"),
                };
                let result = self.emit_i64_const(block_idx, value);
                self.store_bool_reg(block_idx, rd, result)?;
                return Ok(true);
            }
        }

        let comparison_shape = Self::sequence_comparison_shape(lhs_shape, rhs_shape)?;
        let lhs_materialized =
            self.materialize_reg_as_compact_source(block_idx, r1, &comparison_shape)?;
        let rhs_materialized = self.materialize_reg_as_compact_source(
            lhs_materialized.block_idx,
            r2,
            &comparison_shape,
        )?;
        let block_idx = rhs_materialized.block_idx;
        let slot_count = comparison_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "sequence equality requires fixed-width compact shape, got {comparison_shape:?}"
            ))
        })?;

        let mut equal = self.emit_i64_const(block_idx, 1);
        for slot in 0..slot_count {
            let lhs_offset = lhs_materialized
                .slot
                .offset
                .checked_add(slot)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "sequence equality left compact slot offset overflows u32".to_owned(),
                    )
                })?;
            let rhs_offset = rhs_materialized
                .slot
                .offset
                .checked_add(slot)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "sequence equality right compact slot offset overflows u32".to_owned(),
                    )
                })?;
            let lhs = self.load_at_offset(block_idx, lhs_materialized.slot.source_ptr, lhs_offset);
            let rhs = self.load_at_offset(block_idx, rhs_materialized.slot.source_ptr, rhs_offset);
            let slot_eq = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs,
                    rhs,
                },
            );
            let slot_eq = self.emit_logic_bool_to_i64(block_idx, slot_eq);
            equal = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: equal,
                    rhs: slot_eq,
                },
            );
        }

        let result = match predicate {
            ICmpOp::Eq => equal,
            ICmpOp::Ne => {
                let zero = self.emit_i64_const(block_idx, 0);
                let neq = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: equal,
                        rhs: zero,
                    },
                );
                self.emit_logic_bool_to_i64(block_idx, neq)
            }
            _ => unreachable!("caller filters to Eq/Ne"),
        };
        self.store_bool_reg(block_idx, rd, result)?;
        Ok(true)
    }

    fn sequence_comparison_shape(
        lhs_shape: &AggregateShape,
        rhs_shape: &AggregateShape,
    ) -> Result<AggregateShape, TrustIrError> {
        let (
            AggregateShape::Sequence {
                extent: lhs_extent,
                element: lhs_element,
            },
            AggregateShape::Sequence {
                extent: rhs_extent,
                element: rhs_element,
            },
        ) = (lhs_shape, rhs_shape)
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "sequence equality requires sequence shapes, got {lhs_shape:?} and {rhs_shape:?}"
            )));
        };

        let capacity = lhs_extent.capacity().max(rhs_extent.capacity());
        let extent = match (lhs_extent.exact_count(), rhs_extent.exact_count()) {
            (Some(lhs_len), Some(rhs_len)) if lhs_len == rhs_len => SequenceExtent::Exact(lhs_len),
            _ => SequenceExtent::Capacity(capacity),
        };
        let element = if capacity == 0 {
            None
        } else {
            let lhs_element = lhs_element.as_deref().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "structural sequence equality requires tracked left element shape, got {lhs_shape:?}"
                ))
            })?;
            let rhs_element = rhs_element.as_deref().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "structural sequence equality requires tracked right element shape, got {rhs_shape:?}"
                ))
            })?;
            let element = super::merge_compatible_shapes(Some(lhs_element), Some(rhs_element))
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "structural sequence equality requires compatible element shapes, got {lhs_element:?} and {rhs_element:?}"
                    ))
                })?;
            if !Self::compatible_compact_materialization_value(lhs_element, &element)
                || !Self::compatible_compact_materialization_value(rhs_element, &element)
                || element.compact_slot_count().is_none()
            {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "structural sequence equality requires fixed-width compatible element materialization, got {lhs_element:?}, {rhs_element:?}, merged={element:?}"
                )));
            }
            Some(Box::new(element))
        };
        Ok(AggregateShape::Sequence { extent, element })
    }

    fn emit_logic_bool_to_i64(
        &mut self,
        block_idx: usize,
        value: trust_ir::value::ValueId,
    ) -> trust_ir::value::ValueId {
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

    fn emit_sequence_len_comparison(
        &mut self,
        block_idx: usize,
        reg: u8,
        shape: &AggregateShape,
        predicate: ICmpOp,
    ) -> Result<trust_ir::value::ValueId, TrustIrError> {
        let len = self.sequence_len_value(block_idx, reg, shape)?;
        let zero = self.emit_i64_const(block_idx, 0);
        let cmp = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: predicate,
                ty: Ty::I64,
                lhs: len,
                rhs: zero,
            },
        );
        Ok(self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: cmp,
            },
        ))
    }

    fn sequence_len_value(
        &mut self,
        block_idx: usize,
        reg: u8,
        shape: &AggregateShape,
    ) -> Result<trust_ir::value::ValueId, TrustIrError> {
        let AggregateShape::Sequence { extent, .. } = shape else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "sequence equality expected sequence shape for r{reg}, got {shape:?}"
            )));
        };
        if let Some(exact) = extent.exact_count() {
            return Ok(self.emit_i64_const(block_idx, i64::from(exact)));
        }
        if let Some(source_slot) = self.compact_state_slot_for_use(block_idx, reg)? {
            return Ok(self.load_at_offset(block_idx, source_slot.source_ptr, source_slot.offset));
        }
        let ptr = self.load_reg_as_ptr(block_idx, reg)?;
        Ok(self.load_at_offset(block_idx, ptr, 0))
    }

    fn scalar_equality_static_result(&self, r1: u8, r2: u8, predicate: ICmpOp) -> Option<bool> {
        if !matches!(predicate, ICmpOp::Eq | ICmpOp::Ne) {
            return None;
        }

        let lhs = self.aggregate_shapes.get(&r1);
        let rhs = self.aggregate_shapes.get(&r2);
        match (lhs, rhs) {
            (Some(AggregateShape::Scalar(left)), Some(AggregateShape::Scalar(right)))
                if left != right =>
            {
                if self.scalar_int_string_atom_bridge(r1, left, r2, right)
                    || self.scalar_int_string_atom_bridge(r2, right, r1, left)
                {
                    return None;
                }
                // The String lane is the shared interned-NameId encoding for
                // both TLA strings AND model values after flat-primary
                // promotion (`layout_bridge` maps `FixedScalar { base:
                // ModelValue }` to `CompoundLayout::String`). A String-shaped
                // register that is NOT a known constant may therefore have
                // been loaded from a model-value state slot, and its
                // comparison against a ModelValue operand is a legitimate
                // runtime NameId compare. Folding it as a static type
                // mismatch is unsound: dijkstra's `k # self` guard folded to
                // TRUE, silently truncating the reachable state space. Fold
                // this pair only when BOTH operands are known constants.
                let string_model_value_pair = matches!(
                    (left, right),
                    (ScalarShape::String, ScalarShape::ModelValue)
                        | (ScalarShape::ModelValue, ScalarShape::String)
                );
                if string_model_value_pair
                    && (self.scalar_of(r1).is_none() || self.scalar_of(r2).is_none())
                {
                    return None;
                }
                Some(matches!(predicate, ICmpOp::Ne))
            }
            _ => None,
        }
    }

    fn scalar_int_string_atom_bridge(
        &self,
        int_reg: u8,
        int_shape: &ScalarShape,
        string_reg: u8,
        string_shape: &ScalarShape,
    ) -> bool {
        if !matches!(int_shape, ScalarShape::Int)
            || !matches!(string_shape, ScalarShape::String | ScalarShape::ModelValue)
        {
            return false;
        }
        if !self.is_load_imm_scalar(int_reg) || !self.aggregate_shapes.contains_key(&string_reg) {
            return false;
        }
        // The bridge is meant to cover specialized NameId LoadImm vs *dynamic*
        // String/ModelValue compact slots (where the runtime needs an ICmp).
        // A bare LoadImm compared against a *known constant* String/ModelValue
        // (e.g., LoadConst) has no NameId provenance — both sides are
        // statically known, so the comparison must fold as a TLA+ type
        // mismatch instead of being papered over as a runtime equality.
        if self.scalar_of(string_reg).is_some() {
            return false;
        }
        self.scalar_of(int_reg).is_some_and(|value| {
            u32::try_from(value).is_ok_and(|id| {
                usize::try_from(id).is_ok_and(|idx| idx < tla_core::interned_name_count())
            })
        })
    }

    pub(super) fn lower_boolean_binary(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        op: BinOp,
    ) -> Result<(), TrustIrError> {
        let lhs = self.load_reg(block_idx, r1)?;
        let rhs = self.load_reg(block_idx, r2)?;
        let zero = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            },
        );
        let lhs_bool = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs,
                rhs: zero,
            },
        );
        let rhs_bool = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: rhs,
                rhs: zero,
            },
        );
        let result = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op,
                ty: Ty::Bool,
                lhs: lhs_bool,
                rhs: rhs_bool,
            },
        );
        let result_i64 = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: result,
            },
        );
        self.store_bool_reg(block_idx, rd, result_i64)
    }

    pub(super) fn lower_not(
        &mut self,
        block_idx: usize,
        rd: u8,
        rs: u8,
    ) -> Result<(), TrustIrError> {
        let value = self.load_reg(block_idx, rs)?;
        let zero = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            },
        );
        // NOT: value == 0
        let cmp = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: value,
                rhs: zero,
            },
        );
        let result = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: cmp,
            },
        );
        self.store_bool_reg(block_idx, rd, result)
    }

    pub(super) fn lower_implies(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<(), TrustIrError> {
        let lhs = self.load_reg(block_idx, r1)?;
        let rhs = self.load_reg(block_idx, r2)?;
        let zero = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            },
        );
        // !lhs: lhs == 0
        let not_lhs = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs,
                rhs: zero,
            },
        );
        // rhs != 0
        let rhs_bool = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: rhs,
                rhs: zero,
            },
        );
        // implies = !lhs || rhs
        let or_result = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::Or,
                ty: Ty::Bool,
                lhs: not_lhs,
                rhs: rhs_bool,
            },
        );
        let result = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: or_result,
            },
        );
        self.store_bool_reg(block_idx, rd, result)
    }

    pub(super) fn lower_equiv(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<(), TrustIrError> {
        let lhs = self.load_reg(block_idx, r1)?;
        let rhs = self.load_reg(block_idx, r2)?;
        let zero = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            },
        );
        let lhs_bool = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs,
                rhs: zero,
            },
        );
        let rhs_bool = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: rhs,
                rhs: zero,
            },
        );
        // Equivalence on boolean truth values, then canonicalize to i64.
        let cmp = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::Bool,
                lhs: lhs_bool,
                rhs: rhs_bool,
            },
        );
        let result = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: cmp,
            },
        );
        self.store_bool_reg(block_idx, rd, result)
    }

    pub(super) fn lower_cond_move(
        &mut self,
        block_idx: usize,
        rd: u8,
        cond: u8,
        rs: u8,
    ) -> Result<usize, TrustIrError> {
        let merged_shape = super::merge_compatible_shapes(
            self.aggregate_shapes.get(&rd),
            self.aggregate_shapes.get(&rs),
        );
        if merged_shape.is_none() {
            if let (Some(AggregateShape::Scalar(left)), Some(AggregateShape::Scalar(right))) = (
                self.aggregate_shapes.get(&rd),
                self.aggregate_shapes.get(&rs),
            ) {
                if left != right {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "CondMove over incompatible scalar lanes requires tagged selection: {left:?} vs {right:?}"
                    )));
                }
            }
        }
        if merged_shape
            .as_ref()
            .is_some_and(Self::is_compact_compound_aggregate)
        {
            let shape = merged_shape.as_ref().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "CondMove over compact aggregate values requires a merged shape".to_owned(),
                )
            })?;
            let current_shape = self.aggregate_shapes.get(&rd).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "CondMove over compact aggregate r{rd} requires tracked current shape"
                ))
            })?;
            let source_shape = self.aggregate_shapes.get(&rs).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "CondMove over compact aggregate r{rs} requires tracked source shape"
                ))
            })?;
            if !Self::compatible_compact_materialization_value(current_shape, shape)
                || !Self::compatible_compact_materialization_value(source_shape, shape)
            {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "CondMove over compact aggregate values requires compatible slot materialization, got current={current_shape:?}, source={source_shape:?}, merged={shape:?}"
                )));
            }
            let slot_count = shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "CondMove over compact aggregate requires fixed-width shape, got {shape:?}"
                ))
            })?;
            let current_materialized =
                self.materialize_reg_as_compact_source(block_idx, rd, shape)?;
            let source_materialized =
                self.materialize_reg_as_compact_source(current_materialized.block_idx, rs, shape)?;
            let block_idx = source_materialized.block_idx;
            let current_slot = current_materialized.slot;
            let source_slot = source_materialized.slot;
            let cond_value = self.load_reg(block_idx, cond)?;
            let zero = self.emit_with_result(
                block_idx,
                Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                },
            );
            let cond_bool = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Ne,
                    ty: Ty::I64,
                    lhs: cond_value,
                    rhs: zero,
                },
            );
            let result_ptr = self.alloc_aggregate(block_idx, slot_count);
            for offset in 0..slot_count {
                let current_offset = current_slot.offset.checked_add(offset).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "CondMove compact aggregate current slot overflows u32".to_owned(),
                    )
                })?;
                let source_offset = source_slot.offset.checked_add(offset).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "CondMove compact aggregate source slot overflows u32".to_owned(),
                    )
                })?;
                let current_value =
                    self.load_at_offset(block_idx, current_slot.source_ptr, current_offset);
                let source_value =
                    self.load_at_offset(block_idx, source_slot.source_ptr, source_offset);
                let result = self.emit_with_result(
                    block_idx,
                    Inst::Select {
                        ty: Ty::I64,
                        cond: cond_bool,
                        then_val: source_value,
                        else_val: current_value,
                    },
                );
                self.store_at_offset(block_idx, result_ptr, offset, result);
            }
            self.store_reg_ptr(block_idx, rd, result_ptr)?;
            self.compact_state_slots
                .insert(rd, super::CompactStateSlot::pointer_backed(result_ptr, 0));
            self.const_scalar_values.remove(&rd);
            self.const_set_sizes.remove(&rd);
            if let Some(len) = shape.tracked_len() {
                self.const_set_sizes.insert(rd, len);
            }
            self.aggregate_shapes.insert(rd, shape.clone());
            return Ok(block_idx);
        }
        let cond_value = self.load_reg(block_idx, cond)?;
        let zero = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            },
        );
        let cond_bool = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: cond_value,
                rhs: zero,
            },
        );

        // Scalar conditional move lowers to a single LLVM `select` (cond ? rs : rd)
        // rather than a CFG diamond: both lanes are side-effect-free i64 register
        // reads, so eager evaluation is free and avoids the extra blocks/branches
        // (a per-op deopt in IF-heavy actions). Operand order matches the prior
        // diamond exactly: then = source `rs`, else = current `rd`.
        let source = self.load_reg(block_idx, rs)?;
        let current = self.load_reg(block_idx, rd)?;
        let result = self.emit_with_result(
            block_idx,
            Inst::Select {
                ty: Ty::I64,
                cond: cond_bool,
                then_val: source,
                else_val: current,
            },
        );
        self.store_reg_value(block_idx, rd, result)?;
        self.compact_state_slots.remove(&rd);
        self.const_scalar_values.remove(&rd);
        self.const_set_sizes.remove(&rd);
        if let Some(shape) = merged_shape {
            if let Some(len) = shape.tracked_len() {
                self.const_set_sizes.insert(rd, len);
            }
            self.aggregate_shapes.insert(rd, shape);
        } else {
            self.aggregate_shapes.remove(&rd);
        }
        Ok(block_idx)
    }
}
