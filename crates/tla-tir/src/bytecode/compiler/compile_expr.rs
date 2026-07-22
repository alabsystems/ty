// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `compile_expr` dispatch table — the main TIR expression→opcode translation.
//!
//! Support-heavy lowering (name resolution, EXCEPT, records, Prime, UNCHANGED)
//! lives in `compile_expr_support.rs`.

use super::super::opcode::{Opcode, Register};
use super::compile_const_fold::is_const_fold_candidate;
use super::{CompileError, FnCompileState};
use crate::nodes::{TirArithOp, TirCmpOp, TirExpr, TirNameKind, TirNameRef, TirSetOp};
use tla_core::Spanned;

const SET_ENUM_SUBSETEQ_FUSION_MAX_ARITY: usize = 2;

impl<'a> FnCompileState<'a> {
    /// Compile a TIR expression, returning the register holding the result.
    pub(super) fn compile_expr(
        &mut self,
        expr: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        // F1 (lever L2): constant EAGER set-constructor subtrees (the
        // SetEnum / SetBinOp / BigUnion arms below; lazy constructors fold
        // only as inner nodes — see is_const_fold_candidate) fold to a
        // single LoadConst at compile time by executing the real VM once,
        // instead of re-materializing the same value on every state.
        // Refusals fall through to the normal arms. Constants-change safety:
        // BytecodeCache::sync_resolved_constants (tla-eval) rebuilds this
        // compiler AND clears compiled results whenever the resolved-
        // constants key changes, so folded constants cannot go stale.
        if is_const_fold_candidate(&expr.node) {
            if let Some(rd) = self.try_const_fold_set_expr(expr)? {
                return Ok(rd);
            }
        }
        match &expr.node {
            // === Constants ===
            TirExpr::Const { value, .. } => self.compile_const(value),

            // === Variables (support module) ===
            TirExpr::Name(name_ref) => self.compile_name_expr(name_ref),

            // === Arithmetic ===
            TirExpr::ArithBinOp { left, op, right } => {
                let r1 = self.compile_expr(left)?;
                let r2 = self.compile_expr(right)?;
                let rd = self.alloc_reg()?;
                let opcode = match op {
                    TirArithOp::Add => Opcode::AddInt { rd, r1, r2 },
                    TirArithOp::Sub => Opcode::SubInt { rd, r1, r2 },
                    TirArithOp::Mul => Opcode::MulInt { rd, r1, r2 },
                    TirArithOp::Div => Opcode::DivInt { rd, r1, r2 },
                    TirArithOp::IntDiv => Opcode::IntDiv { rd, r1, r2 },
                    TirArithOp::Mod => Opcode::ModInt { rd, r1, r2 },
                    TirArithOp::Pow => Opcode::PowInt { rd, r1, r2 },
                };
                self.func.emit(opcode);
                Ok(rd)
            }

            TirExpr::ArithNeg(inner) => {
                let rs = self.compile_expr(inner)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::NegInt { rd, rs });
                Ok(rd)
            }

            // === Boolean ===
            TirExpr::BoolBinOp { left, op, right } => self.compile_bool_binop(left, *op, right),

            TirExpr::BoolNot(inner) => {
                let rs = self.compile_expr(inner)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::Not { rd, rs });
                Ok(rd)
            }

            // === Comparison ===
            TirExpr::Cmp { left, op, right } => {
                // VM-only exact-shape fusion for Sailfish's TypeOK predicate
                // `e = <<e[1], e[2]>>`. The matcher proves that all three
                // identifiers are the same current lexical binding before
                // eliding either projection or the temporary tuple.
                if matches!(op, TirCmpOp::Eq) && self.tuple2_self_eq {
                    if let Some(value) = self.match_tuple2_self_eq(left, right) {
                        let rd = self.alloc_reg()?;
                        self.func.emit(Opcode::Tuple2SelfEq { rd, value });
                        return Ok(rd);
                    }
                }
                let r1 = self.compile_expr(left)?;
                let r2 = self.compile_expr(right)?;
                // Eq-fusion peephole (implied-action term compile only): when
                // the right operand's producer is the immediately preceding
                // FuncExcept/RecordNew writing the expression temp `r2`, fuse
                // producer + Eq into one non-materializing superinstruction.
                // `r2` is a pure expression temp (its only consumer is this
                // Eq), and the fused opcode replaces the producer at ITS pc,
                // so any jump landing there still executes the full
                // producer-then-Eq semantics. Fail-closed guards live in
                // `try_fuse_eq`.
                if matches!(op, TirCmpOp::Eq) && self.eq_fusion {
                    if let Some(fused_rd) = self.try_fuse_eq(r1, r2)? {
                        return Ok(fused_rd);
                    }
                }
                let rd = self.alloc_reg()?;
                let opcode = match op {
                    TirCmpOp::Eq => Opcode::Eq { rd, r1, r2 },
                    TirCmpOp::Neq => Opcode::Neq { rd, r1, r2 },
                    TirCmpOp::Lt => Opcode::LtInt { rd, r1, r2 },
                    TirCmpOp::Leq => Opcode::LeInt { rd, r1, r2 },
                    TirCmpOp::Gt => Opcode::GtInt { rd, r1, r2 },
                    TirCmpOp::Geq => Opcode::GeInt { rd, r1, r2 },
                };
                self.func.emit(opcode);
                Ok(rd)
            }

            // === Set Membership ===
            TirExpr::In { elem, set } => {
                // VM-only tuple-membership fusion. The syntactic/arity guard
                // makes the two component values directly available without
                // changing any general tuple expression. Components and the
                // set retain the historical left-to-right evaluation order.
                if self.tuple2_set_in {
                    if let TirExpr::Tuple(elements) = &elem.node {
                        if let [_, _] = elements.as_slice() {
                            let start = self.compile_exprs_into_consecutive(elements.iter())?;
                            let r_set = self.compile_expr(set)?;
                            let rd = self.alloc_reg()?;
                            self.func.emit(Opcode::Tuple2SetIn {
                                rd,
                                first: start,
                                second: start + 1,
                                set: r_set,
                            });
                            return Ok(rd);
                        }
                    }
                }
                let r_elem = self.compile_expr(elem)?;
                let r_set = self.compile_expr(set)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::SetIn {
                    rd,
                    elem: r_elem,
                    set: r_set,
                });
                Ok(rd)
            }

            TirExpr::Subseteq { left, right } => {
                // VM-only set-enum fusion. Compile the element expressions
                // first and the RHS second, preserving the historical
                // left-to-right evaluation order. The fused VM handler falls
                // back to ordinary SetEnum + Subseteq semantics unless the
                // evaluated RHS is a concrete Value::Set.
                if self.set_enum_subseteq {
                    if let TirExpr::SetEnum(elements) = &left.node {
                        // Keep the fusion bounded: direct membership is ideal
                        // for Sailfish's two-element shape, but can become
                        // quadratic for large enumerations over an
                        // unnormalized RHS set.
                        if elements.len() <= SET_ENUM_SUBSETEQ_FUSION_MAX_ARITY {
                            // Preserve the ordinary left-expression path's
                            // compile-time constant-set fold before bypassing
                            // `compile_expr(left)` with the fused opcode.
                            if let Some(r_left) = self.try_const_fold_set_expr(left)? {
                                let r_right = self.compile_expr(right)?;
                                let rd = self.alloc_reg()?;
                                self.func.emit(Opcode::Subseteq {
                                    rd,
                                    r1: r_left,
                                    r2: r_right,
                                });
                                return Ok(rd);
                            }

                            let start = self.compile_exprs_into_consecutive(elements.iter())?;
                            let r_set = self.compile_expr(right)?;
                            let rd = self.alloc_reg()?;
                            self.func.emit(Opcode::SetEnumSubseteq {
                                rd,
                                start,
                                count: elements.len() as u8,
                                set: r_set,
                            });
                            return Ok(rd);
                        }
                    }
                }
                let r1 = self.compile_expr(left)?;
                let r2 = self.compile_expr(right)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::Subseteq { rd, r1, r2 });
                Ok(rd)
            }

            // === If-Then-Else ===
            TirExpr::If { cond, then_, else_ } => self.compile_if(cond, then_, else_),

            // === Set Operations ===
            TirExpr::SetEnum(elements) => {
                if elements.is_empty() {
                    let rd = self.alloc_reg()?;
                    self.func.emit(Opcode::SetEnum {
                        rd,
                        start: 0,
                        count: 0,
                    });
                    return Ok(rd);
                }

                let start = self.compile_exprs_into_consecutive(elements.iter())?;
                let count = elements.len().min(255) as u8;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::SetEnum { rd, start, count });
                Ok(rd)
            }

            TirExpr::SetBinOp { left, op, right } => {
                let r1 = self.compile_expr(left)?;
                let r2 = self.compile_expr(right)?;
                let rd = self.alloc_reg()?;
                let opcode = match op {
                    TirSetOp::Union => Opcode::SetUnion { rd, r1, r2 },
                    TirSetOp::Intersect => Opcode::SetIntersect { rd, r1, r2 },
                    TirSetOp::Minus => Opcode::SetDiff { rd, r1, r2 },
                };
                self.func.emit(opcode);
                Ok(rd)
            }

            TirExpr::Powerset(inner) => {
                let rs = self.compile_expr(inner)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::Powerset { rd, rs });
                Ok(rd)
            }

            TirExpr::BigUnion(inner) => {
                let rs = self.compile_expr(inner)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::BigUnion { rd, rs });
                Ok(rd)
            }

            TirExpr::KSubset { base, k } => {
                let r_base = self.compile_expr(base)?;
                let r_k = self.compile_expr(k)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::KSubset {
                    rd,
                    base: r_base,
                    k: r_k,
                });
                Ok(rd)
            }

            TirExpr::Range { lo, hi } => {
                let r_lo = self.compile_expr(lo)?;
                let r_hi = self.compile_expr(hi)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::Range {
                    rd,
                    lo: r_lo,
                    hi: r_hi,
                });
                Ok(rd)
            }

            // === Quantifiers ===
            TirExpr::Forall { vars, body } => self.compile_forall(vars, body),
            TirExpr::Exists { vars, body } => self.compile_exists(vars, body),
            TirExpr::Choose { var, body } => self.compile_choose(var, body),

            // === Set Comprehensions ===
            TirExpr::SetFilter { var, body } => self.compile_set_filter(var, body),
            TirExpr::SetBuilder { body, vars } => self.compile_set_builder(body, vars),

            // === Functions ===
            TirExpr::FuncDef { vars, body } => self.compile_func_def(vars, body),

            TirExpr::FuncApply { func, arg } => {
                let r_func = self.compile_expr(func)?;
                let r_arg = self.compile_expr(arg)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::FuncApply {
                    rd,
                    func: r_func,
                    arg: r_arg,
                });
                Ok(rd)
            }

            TirExpr::FuncSet { domain, range } => {
                let r_domain = self.compile_expr(domain)?;
                let r_range = self.compile_expr(range)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::FuncSet {
                    rd,
                    domain: r_domain,
                    range: r_range,
                });
                Ok(rd)
            }

            TirExpr::Domain(inner) => {
                let rs = self.compile_expr(inner)?;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::Domain { rd, rs });
                Ok(rd)
            }

            // === EXCEPT (support module) ===
            TirExpr::Except { base, specs } => self.compile_except_expr(base, specs),

            // === Records (support module) ===
            TirExpr::Record(fields) => self.compile_record_expr(fields),
            TirExpr::RecordAccess { record, field } => {
                self.compile_record_access_expr(record, field)
            }
            TirExpr::RecordSet(fields) => self.compile_record_set_expr(fields),

            // === Tuples ===
            TirExpr::Tuple(elements) => {
                let start = self.compile_exprs_into_consecutive(elements.iter())?;
                let count = elements.len().min(255) as u8;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::TupleNew { rd, start, count });
                Ok(rd)
            }

            TirExpr::Times(components) => {
                let start = self.compile_exprs_into_consecutive(components.iter())?;
                let count = components.len().min(255) as u8;
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::Times { rd, start, count });
                Ok(rd)
            }

            // === Priming (support module) ===
            TirExpr::Prime(inner) => self.compile_prime_expr(inner),
            TirExpr::Unchanged(inner) => self.compile_unchanged_expr(inner),

            // === LET/IN ===
            TirExpr::Let { defs, body } => self.compile_let(defs, body),

            // === CASE ===
            TirExpr::Case { arms, other } => self.compile_case(arms, other.as_deref()),

            // === Labels (transparent) ===
            TirExpr::Label { body, .. } => self.compile_expr(body),

            // === ExceptAt (@) ===
            TirExpr::ExceptAt => {
                if let Some(at_reg) = self.except_at_register {
                    // @ resolves to the pre-computed base[key] value.
                    Ok(at_reg)
                } else {
                    Err(CompileError::Unsupported("standalone ExceptAt".to_string()))
                }
            }

            // === Operator References / Apply ===
            TirExpr::OperatorRef(op_ref) => self.compile_operator_ref(op_ref),
            TirExpr::Apply { op, args } => self.compile_apply(op, args),
            TirExpr::Lambda {
                params,
                body,
                ast_body,
            } => self.compile_lambda_expr(params, body, ast_body),
            TirExpr::OpRef(op) => self.compile_op_ref_expr(op, expr.span),
            TirExpr::ActionSubscript { .. } => {
                Err(CompileError::Unsupported("ActionSubscript".to_string()))
            }

            // Temporal operators — never evaluated at the value level.
            TirExpr::Always(_)
            | TirExpr::Eventually(_)
            | TirExpr::LeadsTo { .. }
            | TirExpr::WeakFair { .. }
            | TirExpr::StrongFair { .. }
            | TirExpr::Enabled(_) => {
                Err(CompileError::Unsupported("temporal operator".to_string()))
            }
        }
    }
}

impl<'a> FnCompileState<'a> {
    /// Match only
    /// `e = <<e[1], e[2]>> /\ {e[1], e[2]} \subseteq state_set`.
    ///
    /// The five `e` references must resolve to one bound identifier, and the
    /// right-hand set must resolve exactly as an ordinary state-variable
    /// load. Static prime contexts decline so the fused opcode retains the
    /// exact dynamic semantics of an ordinary unprimed `LoadVar`.
    pub(super) fn match_tuple2_self_subseteq(
        &self,
        left: &Spanned<TirExpr>,
        right: &Spanned<TirExpr>,
    ) -> Option<(Register, u16)> {
        if self.in_prime_context {
            return None;
        }

        let TirExpr::Cmp {
            left: equality_left,
            op: TirCmpOp::Eq,
            right: equality_right,
        } = &left.node
        else {
            return None;
        };
        let value = self.match_tuple2_self_eq(equality_left, equality_right)?;
        let TirExpr::Name(value_name) = &equality_left.node else {
            return None;
        };

        let TirExpr::Subseteq {
            left: subset_left,
            right: subset_right,
        } = &right.node
        else {
            return None;
        };
        let TirExpr::SetEnum(elements) = &subset_left.node else {
            return None;
        };
        let [first, second] = elements.as_slice() else {
            return None;
        };
        let first_name = exact_name_projection(first, 1)?;
        let second_name = exact_name_projection(second, 2)?;
        if !same_ident_identity(value_name, first_name)
            || !same_ident_identity(value_name, second_name)
            || self.lookup_binding(&first_name.name) != Some(value)
            || self.lookup_binding(&second_name.name) != Some(value)
        {
            return None;
        }

        let TirExpr::Name(set_name) = &subset_right.node else {
            return None;
        };
        let set_var_idx = self.resolve_name_state_var(set_name)?;
        Some((value, set_var_idx))
    }

    /// Match only `e = <<e[1], e[2]>>` for one already-bound local `e`.
    ///
    /// Orientation, tuple arity, projection order, literal indices, identifier
    /// kind, source identity, and lexical register resolution are all exact.
    /// Any mismatch retains ordinary compilation and evaluation.
    fn match_tuple2_self_eq(
        &self,
        left: &Spanned<TirExpr>,
        right: &Spanned<TirExpr>,
    ) -> Option<Register> {
        let TirExpr::Name(value_name) = &left.node else {
            return None;
        };
        let TirExpr::Tuple(elements) = &right.node else {
            return None;
        };
        let [first, second] = elements.as_slice() else {
            return None;
        };
        let first_name = exact_name_projection(first, 1)?;
        let second_name = exact_name_projection(second, 2)?;

        if !same_ident_identity(value_name, first_name)
            || !same_ident_identity(value_name, second_name)
        {
            return None;
        }

        let value = self.lookup_binding(&value_name.name)?;
        (self.lookup_binding(&first_name.name) == Some(value)
            && self.lookup_binding(&second_name.name) == Some(value))
        .then_some(value)
    }

    /// Eq-fusion peephole: try to fuse `Eq(r1, r2)` with the immediately
    /// preceding producer of `r2`.
    ///
    /// Fusable producers and their superinstructions:
    /// * `FuncExcept { rd: r2, func, path, val }` → `EqFuncExcept`
    /// * `RecordNew { rd: r2, fields_start, values_start, count }` → `EqRecordNew`
    ///
    /// Fail-closed guards (any failure → `Ok(None)`, the caller emits the
    /// plain `Eq`):
    /// * the producer must be the LAST emitted instruction and write exactly
    ///   `r2` (the expression temp produced for the Eq's right operand — its
    ///   only consumer by expression-tree construction);
    /// * `r1 != r2` (degenerate aliasing);
    /// * no patched jump may target a pc BEYOND the producer
    ///   (`max_patched_target <= producer_pc`): such a jump was patched to
    ///   land on the not-yet-emitted `Eq` slot and must keep the
    ///   two-instruction shape. Jumps landing ON the producer's pc remain
    ///   correct — the fused opcode executes the full producer+Eq semantics
    ///   at that pc.
    ///
    /// Returns the destination register of the fused instruction.
    fn try_fuse_eq(
        &mut self,
        r1: Register,
        r2: Register,
    ) -> Result<Option<Register>, CompileError> {
        if r1 == r2 || self.func.instructions.is_empty() {
            return Ok(None);
        }
        let producer_pc = self.func.instructions.len() - 1;
        if self.func.max_patched_target > producer_pc {
            return Ok(None);
        }
        let fused = match *self
            .func
            .instructions
            .last()
            .expect("non-empty checked above")
        {
            Opcode::FuncExcept {
                rd,
                func,
                path,
                val,
            } if rd == r2 => {
                let rd_out = self.alloc_reg()?;
                Some(Opcode::EqFuncExcept {
                    rd: rd_out,
                    lhs: r1,
                    func,
                    path,
                    val,
                })
            }
            Opcode::RecordNew {
                rd,
                fields_start,
                values_start,
                count,
            } if rd == r2 => {
                let rd_out = self.alloc_reg()?;
                Some(Opcode::EqRecordNew {
                    rd: rd_out,
                    lhs: r1,
                    fields_start,
                    values_start,
                    count,
                })
            }
            _ => None,
        };
        let Some(fused) = fused else {
            return Ok(None);
        };
        self.func.instructions.pop();
        self.func.emit(fused);
        let rd_out = fused
            .dest_register()
            .expect("fused Eq opcodes always have a destination");
        Ok(Some(rd_out))
    }
}

fn exact_name_projection(expr: &Spanned<TirExpr>, expected_index: i64) -> Option<&TirNameRef> {
    let TirExpr::FuncApply { func, arg } = &expr.node else {
        return None;
    };
    let TirExpr::Name(name) = &func.node else {
        return None;
    };
    let TirExpr::Const {
        value: tla_value::Value::SmallInt(index),
        ..
    } = &arg.node
    else {
        return None;
    };
    (*index == expected_index).then_some(name)
}

fn same_ident_identity(left: &TirNameRef, right: &TirNameRef) -> bool {
    matches!(left.kind, TirNameKind::Ident)
        && matches!(right.kind, TirNameKind::Ident)
        && left.name == right.name
        && left.name_id == right.name_id
}
