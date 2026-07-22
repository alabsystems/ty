// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Arithmetic lowering: overflow-checked Add, Sub, Mul, Neg, Div, Mod, real division.

use crate::TrustIrError;
use tla_jit_abi::JitRuntimeErrorKind;
use trust_ir::inst::*;
use trust_ir::ty::Ty;
use trust_ir::value::ValueId;
use trust_ir::{Constant, InstrNode};

use super::Ctx;

impl<'cp> Ctx<'cp> {
    fn record_int_result(&mut self, rd: u8) {
        self.compact_state_slots.remove(&rd);
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);
        self.aggregate_shapes
            .insert(rd, super::AggregateShape::Scalar(super::ScalarShape::Int));
    }

    /// Like `record_int_result`, but also records a statically known scalar
    /// value for `rd` when one was computed by constant-folding the operands.
    ///
    /// Soundness: the folded value is recorded only when both operands are
    /// compile-time constants on this path and the operation does not overflow
    /// (`checked_*` returned `Some`). In that case the emitted overflow check
    /// provably does not trap, so the runtime result is deterministically the
    /// folded value — recording it cannot diverge from interpreter semantics.
    /// The emitted instructions are unchanged; only compile-time tracking
    /// metadata is enriched so downstream shape inference (e.g. `Range`
    /// recovering an `Interval` for `0..(N-1)`) sees a known endpoint.
    fn record_int_result_with_fold(&mut self, rd: u8, folded: Option<i64>) {
        self.record_int_result(rd);
        if let Some(value) = folded {
            self.const_scalar_values.insert(rd, value);
        }
    }

    /// Fail closed when an integer op would consume a `TaggedScalarUnion`
    /// operand: the slot stores a universe INDEX, not the scalar payload, so raw
    /// i64 arithmetic on it computes a wrong value (e.g. for `Nodes = 1..8`, node
    /// `v` is stored at index `v - 1`, so a native `f[k] + 1` would yield `v`
    /// instead of `v + 1`). The union is only soundly consumed by the union-aware
    /// equality lowering and the write converter; every other op routes here or
    /// through an equivalent guard and hands the action to the interpreter.
    fn reject_tagged_scalar_union_arith_operand(
        &self,
        r1: u8,
        r2: u8,
        context: &str,
    ) -> Result<(), TrustIrError> {
        if self.reg_is_tagged_scalar_union(r1) || self.reg_is_tagged_scalar_union(r2) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context} on a TaggedScalarUnion operand (a universe index, not a scalar value); failing closed to interpreter"
            )));
        }
        Ok(())
    }

    pub(super) fn lower_checked_binary_overflow(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        overflow_op: OverflowOp,
    ) -> Result<Option<usize>, TrustIrError> {
        self.reject_tagged_scalar_union_arith_operand(r1, r2, "integer arithmetic")?;
        // Constant-fold for compile-time shape tracking only (see
        // `record_int_result_with_fold`). Does not alter emitted code.
        let folded = match (self.scalar_of(r1), self.scalar_of(r2)) {
            (Some(a), Some(b)) => match overflow_op {
                OverflowOp::AddOverflow => a.checked_add(b),
                OverflowOp::SubOverflow => a.checked_sub(b),
                OverflowOp::MulOverflow => a.checked_mul(b),
            },
            _ => None,
        };
        let lhs = self.load_reg(block_idx, r1)?;
        let rhs = self.load_reg(block_idx, r2)?;

        // Emit overflow-checked operation: returns (result, overflow_flag).
        let result = self.alloc_value();
        let overflow_flag = self.alloc_value();
        self.emit(
            block_idx,
            InstrNode::new(Inst::Overflow {
                op: overflow_op,
                ty: Ty::I64,
                lhs,
                rhs,
            })
            .with_result(result)
            .with_result(overflow_flag),
        );

        let overflow_block = self.new_aux_block("overflow");
        let continue_block = self.new_aux_block("continue");

        let overflow_id = self.block_id_of(overflow_block);
        let continue_id = self.block_id_of(continue_block);

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: overflow_flag,
                then_target: overflow_id,
                then_args: vec![],
                else_target: continue_id,
                else_args: vec![],
            }),
        );

        self.emit_runtime_error_and_return(overflow_block, JitRuntimeErrorKind::ArithmeticOverflow);
        self.store_reg_value(continue_block, rd, result)?;
        self.record_int_result_with_fold(rd, folded);

        Ok(Some(continue_block))
    }

    pub(super) fn lower_checked_negation(
        &mut self,
        block_idx: usize,
        rd: u8,
        rs: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        self.reject_tagged_scalar_union_arith_operand(rs, rs, "integer negation")?;
        let value = self.load_reg(block_idx, rs)?;

        // Negate via 0 - value with overflow check.
        let zero = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            },
        );

        let result = self.alloc_value();
        let overflow_flag = self.alloc_value();
        self.emit(
            block_idx,
            InstrNode::new(Inst::Overflow {
                op: OverflowOp::SubOverflow,
                ty: Ty::I64,
                lhs: zero,
                rhs: value,
            })
            .with_result(result)
            .with_result(overflow_flag),
        );

        let overflow_block = self.new_aux_block("overflow");
        let continue_block = self.new_aux_block("continue");

        let overflow_id = self.block_id_of(overflow_block);
        let continue_id = self.block_id_of(continue_block);

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: overflow_flag,
                then_target: overflow_id,
                then_args: vec![],
                else_target: continue_id,
                else_args: vec![],
            }),
        );

        self.emit_runtime_error_and_return(overflow_block, JitRuntimeErrorKind::ArithmeticOverflow);
        self.store_reg_value(continue_block, rd, result)?;
        self.record_int_result(rd);

        Ok(Some(continue_block))
    }

    /// Emit the `i64::MIN / -1` overflow guard shared by `\div` and `/`.
    ///
    /// `sdiv`/`srem` of `i64::MIN` by `-1` is undefined behaviour (SIGFPE on
    /// x86, silent wrap on AArch64) and the true quotient `2^63` does not fit
    /// in `i64` (the interpreter widens to `BigInt` there). Emitting the
    /// `ArithmeticOverflow` runtime error routes exactly that state to the
    /// interpreter, which produces the correct wide result.
    ///
    /// Control flow (chained `CondBr`s, no i1 `And` needed):
    ///
    /// ```text
    /// block_idx:      lhs == i64::MIN ? -> neg1_check : continue
    /// neg1_check:     rhs == -1       ? -> overflow   : continue
    /// overflow:       runtime error ArithmeticOverflow, return
    /// continue:       (returned; caller emits the division here)
    /// ```
    fn emit_div_min_overflow_guard(
        &mut self,
        block_idx: usize,
        lhs: ValueId,
        rhs: ValueId,
    ) -> usize {
        let min_const = self.emit_i64_const(block_idx, i64::MIN);
        let is_min = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs,
                rhs: min_const,
            },
        );

        let neg1_check_block = self.new_aux_block("div_min_neg1_check");
        let overflow_block = self.new_aux_block("div_min_overflow");
        let continue_block = self.new_aux_block("div_min_continue");

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: is_min,
                then_target: self.block_id_of(neg1_check_block),
                then_args: vec![],
                else_target: self.block_id_of(continue_block),
                else_args: vec![],
            }),
        );

        let neg1_const = self.emit_i64_const(neg1_check_block, -1);
        let is_neg1 = self.emit_with_result(
            neg1_check_block,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: rhs,
                rhs: neg1_const,
            },
        );
        self.emit(
            neg1_check_block,
            InstrNode::new(Inst::CondBr {
                cond: is_neg1,
                then_target: self.block_id_of(overflow_block),
                then_args: vec![],
                else_target: self.block_id_of(continue_block),
                else_args: vec![],
            }),
        );

        self.emit_runtime_error_and_return(overflow_block, JitRuntimeErrorKind::ArithmeticOverflow);

        continue_block
    }

    /// Lower TLA+ `\div` (floored division) and `%` (Euclidean modulo) with
    /// the exact interpreter semantics ([`tla-eval` `eval_arith`/`int_arith`]).
    ///
    /// `\div` (`use_sdiv == true`):
    ///
    /// * `b == 0` → `DivisionByZero` runtime error;
    /// * `a == i64::MIN && b == -1` → `ArithmeticOverflow` runtime error (the
    ///   interpreter widens to `BigInt` and yields `2^63`, which `i64` cannot
    ///   represent — the runtime error routes that state to the interpreter);
    /// * otherwise `q = sdiv(a, b)`, `r = srem(a, b)`,
    ///   `result = ((a ^ b) < 0 && r != 0) ? q - 1 : q` — the floor adjustment
    ///   for opposite-sign operands (e.g. `-7 \div 2 = -4`, not `-3`).
    ///
    /// `%` (`use_sdiv == false`):
    ///
    /// * `b <= 0` → `ModulusNotPositive` runtime error (TLC requires a
    ///   strictly-positive divisor; this also covers division by zero and
    ///   makes `i64::MIN % -1` unreachable);
    /// * otherwise `r = srem(a, b)`, `result = r < 0 ? r + b : r` — the
    ///   Euclidean correction (`b > 0` so `|b| = b`; e.g. `-7 % 3 = 2`).
    ///
    /// The floor/Euclidean corrections use `Select` on `ICmp` results (the
    /// established `CondMove` idiom). The eagerly computed `q - 1` / `r + b`
    /// lanes may wrap on their never-selected extremes (`q == i64::MIN` only
    /// when `b == 1`, where no adjustment happens; `r + b > i64::MAX` only
    /// when `r >= 0`, where no correction happens); plain `BinOp` `Sub`/`Add`
    /// wrap without UB and the `Select` discards the unused lane.
    pub(super) fn lower_checked_division(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
        use_sdiv: bool, // true = `\div` (floored), false = `%` (Euclidean)
    ) -> Result<Option<usize>, TrustIrError> {
        self.reject_tagged_scalar_union_arith_operand(r1, r2, "integer division")?;
        let lhs = self.load_reg(block_idx, r1)?;
        let rhs = self.load_reg(block_idx, r2)?;
        let zero = self.emit_i64_const(block_idx, 0);

        if use_sdiv {
            // ---- `\div`: guard b == 0 -----------------------------------
            let is_zero = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: rhs,
                    rhs: zero,
                },
            );
            let div_zero_block = self.new_aux_block("div_zero");
            let min_check_block = self.new_aux_block("div_min_check");
            self.emit(
                block_idx,
                InstrNode::new(Inst::CondBr {
                    cond: is_zero,
                    then_target: self.block_id_of(div_zero_block),
                    then_args: vec![],
                    else_target: self.block_id_of(min_check_block),
                    else_args: vec![],
                }),
            );
            self.emit_runtime_error_and_return(div_zero_block, JitRuntimeErrorKind::DivisionByZero);

            // ---- guard a == i64::MIN && b == -1 -------------------------
            let body_block = self.emit_div_min_overflow_guard(min_check_block, lhs, rhs);

            // ---- q = sdiv(a, b); r = srem(a, b) -------------------------
            let q = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::SDiv,
                    ty: Ty::I64,
                    lhs,
                    rhs,
                },
            );
            let r = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::SRem,
                    ty: Ty::I64,
                    lhs,
                    rhs,
                },
            );

            // ---- floor adjust: ((a ^ b) < 0 && r != 0) ? q - 1 : q ------
            let a_xor_b = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::Xor,
                    ty: Ty::I64,
                    lhs,
                    rhs,
                },
            );
            let signs_differ = self.emit_with_result(
                body_block,
                Inst::ICmp {
                    op: ICmpOp::Slt,
                    ty: Ty::I64,
                    lhs: a_xor_b,
                    rhs: zero,
                },
            );
            let r_nonzero = self.emit_with_result(
                body_block,
                Inst::ICmp {
                    op: ICmpOp::Ne,
                    ty: Ty::I64,
                    lhs: r,
                    rhs: zero,
                },
            );
            let one = self.emit_i64_const(body_block, 1);
            let q_minus_1 = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: q,
                    rhs: one,
                },
            );
            // The `&&` is realized as nested selects on the two ICmp bools
            // (no i1 `And` needed): outer picks the adjusted lane only when
            // the signs differ, inner only when the remainder is nonzero.
            let adjusted = self.emit_with_result(
                body_block,
                Inst::Select {
                    ty: Ty::I64,
                    cond: r_nonzero,
                    then_val: q_minus_1,
                    else_val: q,
                },
            );
            let result = self.emit_with_result(
                body_block,
                Inst::Select {
                    ty: Ty::I64,
                    cond: signs_differ,
                    then_val: adjusted,
                    else_val: q,
                },
            );
            self.store_reg_value(body_block, rd, result)?;
            self.record_int_result(rd);
            Ok(Some(body_block))
        } else {
            // ---- `%`: guard b <= 0 (ModulusNotPositive) -----------------
            let is_nonpositive = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: rhs,
                    rhs: zero,
                },
            );
            let nonpositive_block = self.new_aux_block("mod_nonpositive");
            let body_block = self.new_aux_block("mod_body");
            self.emit(
                block_idx,
                InstrNode::new(Inst::CondBr {
                    cond: is_nonpositive,
                    then_target: self.block_id_of(nonpositive_block),
                    then_args: vec![],
                    else_target: self.block_id_of(body_block),
                    else_args: vec![],
                }),
            );
            self.emit_runtime_error_and_return(
                nonpositive_block,
                JitRuntimeErrorKind::ModulusNotPositive,
            );

            // ---- r = srem(a, b); result = r < 0 ? r + b : r -------------
            // b > 0 here, so srem is safe (b != 0, and i64::MIN % -1 is
            // unreachable) and |b| = b for the Euclidean correction.
            let r = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::SRem,
                    ty: Ty::I64,
                    lhs,
                    rhs,
                },
            );
            let r_negative = self.emit_with_result(
                body_block,
                Inst::ICmp {
                    op: ICmpOp::Slt,
                    ty: Ty::I64,
                    lhs: r,
                    rhs: zero,
                },
            );
            let r_plus_b = self.emit_with_result(
                body_block,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: r,
                    rhs,
                },
            );
            let result = self.emit_with_result(
                body_block,
                Inst::Select {
                    ty: Ty::I64,
                    cond: r_negative,
                    then_val: r_plus_b,
                    else_val: r,
                },
            );
            self.store_reg_value(body_block, rd, result)?;
            self.record_int_result(rd);
            Ok(Some(body_block))
        }
    }

    /// Lower TLA+ `/` (real division on integers): exact-or-error.
    ///
    /// * `b == 0` → `DivisionByZero` runtime error;
    /// * `a == i64::MIN && b == -1` → `ArithmeticOverflow` runtime error (the
    ///   quotient `2^63` does not fit in `i64`; the guard also protects the
    ///   exactness `srem`, which is equally UB on that operand pair);
    /// * `a % b != 0` → `TypeMismatch` runtime error (the true quotient is a
    ///   non-integer real; the runtime error routes the state to the
    ///   interpreter rather than flowing a truncated value);
    /// * otherwise `result = sdiv(a, b)` (exact, so truncated == floored).
    pub(super) fn lower_real_division(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        self.reject_tagged_scalar_union_arith_operand(r1, r2, "real division")?;
        let lhs = self.load_reg(block_idx, r1)?;
        let rhs = self.load_reg(block_idx, r2)?;

        // ---- guard b == 0 -----------------------------------------------
        let zero = self.emit_i64_const(block_idx, 0);
        let is_zero = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: rhs,
                rhs: zero,
            },
        );
        let div_zero_block = self.new_aux_block("realdiv_zero");
        let min_check_block = self.new_aux_block("realdiv_min_check");
        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: is_zero,
                then_target: self.block_id_of(div_zero_block),
                then_args: vec![],
                else_target: self.block_id_of(min_check_block),
                else_args: vec![],
            }),
        );
        self.emit_runtime_error_and_return(div_zero_block, JitRuntimeErrorKind::DivisionByZero);

        // ---- guard a == i64::MIN && b == -1 (before the srem/sdiv) -------
        let check_exact_block = self.emit_div_min_overflow_guard(min_check_block, lhs, rhs);

        // ---- exactness: a % b must be 0 -----------------------------------
        let remainder = self.emit_with_result(
            check_exact_block,
            Inst::BinOp {
                op: BinOp::SRem,
                ty: Ty::I64,
                lhs,
                rhs,
            },
        );
        let is_inexact = self.emit_with_result(
            check_exact_block,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: remainder,
                rhs: zero,
            },
        );
        let inexact_block = self.new_aux_block("realdiv_inexact");
        let continue_block = self.new_aux_block("realdiv_continue");
        self.emit(
            check_exact_block,
            InstrNode::new(Inst::CondBr {
                cond: is_inexact,
                then_target: self.block_id_of(inexact_block),
                then_args: vec![],
                else_target: self.block_id_of(continue_block),
                else_args: vec![],
            }),
        );
        self.emit_runtime_error_and_return(inexact_block, JitRuntimeErrorKind::TypeMismatch);

        let result = self.emit_with_result(
            continue_block,
            Inst::BinOp {
                op: BinOp::SDiv,
                ty: Ty::I64,
                lhs,
                rhs,
            },
        );
        self.store_reg_value(continue_block, rd, result)?;
        self.record_int_result(rd);

        Ok(Some(continue_block))
    }
}
