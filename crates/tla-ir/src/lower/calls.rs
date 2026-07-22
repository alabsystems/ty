// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Inter-function call lowering.

use crate::TrustIrError;
use tla_jit_abi::{JitStatus, SetBitmaskElement};
use tla_tir::bytecode::{BytecodeFunction, Opcode};
use tla_value::Value;
use trust_ir::inst::*;
use trust_ir::ty::Ty;
use trust_ir::value::ValueId;
use trust_ir::Constant;
use trust_ir::InstrNode;

use super::Ctx;

impl<'cp> Ctx<'cp> {
    // =====================================================================
    // Inter-function Call
    // =====================================================================

    /// Lower `Opcode::Call { rd, op_idx, args_start, argc }`.
    ///
    /// Loads `argc` arguments from consecutive registers starting at
    /// `args_start`, emits a trust-ir `Inst::Call` to the callee's FuncId,
    /// and stores either the scalar `i64` result or the encoded caller-owned
    /// fixed-width record/sequence/function return buffer pointer into
    /// register `rd`.
    pub(super) fn lower_call(
        &mut self,
        block_idx: usize,
        rd: u8,
        op_idx: u16,
        args_start: u8,
        argc: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        // Register the call target and get its trust-ir FuncId.
        let callee_func_id = self.register_call_target(op_idx);

        // WP-20: every callsite of a SELF-RECURSIVE callee normalizes each
        // argument to the ONE physical convention that callee's parameter seed
        // declares (`merge_self_recursive_callee_arg_shape`), so that the
        // external callsites (raw state scalars) and the recursive self-callsite
        // (union-INDEX values from e.g. a compact function apply) agree:
        //
        //   * declared `TaggedScalarUnion { universe, .. }` — the INDEX
        //     convention: the argument is ENCODED into that universe here, and
        //     an out-of-universe runtime value takes the encode's typed
        //     `TypeMismatch` fail-closed exit (recoverable interpreter
        //     fallback). This is what gives the callee's `Ret` of a parameter a
        //     DECLARED finite scalar domain, and hence lets the call RESULT be
        //     stored into a `TaggedScalarUnion` state slot.
        //   * anything else — the RAW convention: a union-index source is
        //     decoded to its raw member value (identity for every other tracked
        //     shape) and nothing is claimed about the return.
        //
        // The shape vector used for return-shape inference is normalized the
        // same way, so it describes the values the callee actually receives.
        let self_recursive_callee = self.self_recursive_ops.contains(&op_idx);
        let is_self_call = self_recursive_callee && self.current_callee_op_idx == Some(op_idx);

        let arg_function_domains = self.call_arg_function_domains(args_start, argc)?;
        let expected_arg_shapes = self
            .callee_arg_shapes
            .get(&op_idx)
            .cloned()
            .unwrap_or_default();
        // WP-28 diagnostic only: the PRE-normalization shapes, so a callsite's
        // decode/encode decision can be read against what the register actually
        // carried. Materialized only when the tracing gate is on.
        let raw_arg_shapes = if self_recursive_callee && super::wp20_debug() {
            self.call_arg_shapes(args_start, argc)?
        } else {
            Vec::new()
        };
        let arg_shapes = {
            let raw = self.call_arg_shapes(args_start, argc)?;
            if self_recursive_callee {
                raw.into_iter()
                    .enumerate()
                    .map(|(i, shape)| match expected_arg_shapes.get(i).cloned().flatten() {
                        Some(declared @ super::AggregateShape::TaggedScalarUnion { .. }) => {
                            Some(declared)
                        }
                        _ => super::decoded_self_recursive_callsite_arg_shape(shape),
                    })
                    .collect()
            } else {
                raw
            }
        };

        // Load user arguments before the call. Their shapes drive
        // callsite-sensitive return materialization below.
        let mut current_block = block_idx;
        let mut user_arg_values = Vec::with_capacity(usize::from(argc));
        for i in 0..argc {
            let reg = args_start.checked_add(i).ok_or_else(|| {
                TrustIrError::Emission(format!(
                    "Call argument register overflow: args_start={args_start} + i={i}"
                ))
            })?;
            let expected_arg_shape = expected_arg_shapes.get(usize::from(i)).cloned().flatten();
            let compact_arg_abi_shape = self.compact_helper_arg_abi_shape(
                op_idx,
                usize::from(i),
                expected_arg_shape.as_ref(),
            )?;
            if let Some(abi_shape) = compact_arg_abi_shape.as_ref() {
                self.record_callee_compact_arg_abi_shape(op_idx, usize::from(i), abi_shape)?;
                if matches!(
                    abi_shape,
                    super::AggregateShape::Function {
                        domain_lo: None,
                        ..
                    }
                ) {
                    let domain = arg_function_domains
                        .get(usize::from(i))
                        .and_then(Clone::clone)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "Call compact function argument {} for callee {op_idx} r{reg} requires explicit-domain metadata",
                                usize::from(i)
                            ))
                        })?;
                    self.record_callee_arg_function_domain(op_idx, usize::from(i), domain)?;
                }
            }
            let val = if let Some((next_block, materialized)) = self
                .materialize_compact_helper_call_arg(
                    current_block,
                    op_idx,
                    usize::from(i),
                    reg,
                    expected_arg_shape.as_ref(),
                    compact_arg_abi_shape.as_ref(),
                )? {
                current_block = next_block;
                materialized
            } else if self_recursive_callee {
                match expected_arg_shape.as_ref() {
                    // WP-20 INDEX convention: encode into the parameter's
                    // declared union universe. The encode's own guards are the
                    // fail-closed membership check — an out-of-universe runtime
                    // value takes a typed `TypeMismatch` exit, never a wrong
                    // index.
                    Some(super::AggregateShape::TaggedScalarUnion {
                        universe,
                        int_arm,
                        proof_source,
                    }) => {
                        let universe = universe.clone();
                        let int_arm = *int_arm;
                        let proof_source = *proof_source;
                        let context = format!(
                            "Call self-recursive callee {op_idx} argument {} r{reg}",
                            usize::from(i)
                        );
                        let (next_block, encoded) = self.encode_tagged_scalar_union_index(
                            current_block,
                            reg,
                            &universe,
                            int_arm,
                            proof_source,
                            &context,
                        )?;
                        current_block = next_block;
                        encoded
                    }
                    // WP-20 RAW convention: decode a union-index scalar to its
                    // raw member value (identity for every other tracked shape).
                    _ => {
                        // WP-28 fail-closed: the decode is shape-driven, so an
                        // argument whose physical convention is UNPROVEN (the
                        // untracked result of another call, which may itself be
                        // a tagged-scalar-union INDEX) would be handed to the
                        // callee verbatim and read as a raw member — the btree
                        // `GetValue` miscompile, where `FindLeafNode` re-entered
                        // on node `n-1`. Decline instead of emitting.
                        if self.untracked_callee_return_regs.contains(&reg) {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "Call self-recursive callee {op_idx} argument {} r{reg}: the \
                                 value is an untracked callee return whose physical convention \
                                 (raw member vs tagged-scalar-union INDEX) is unproven; the \
                                 raw-argument convention would consume a union index as a member \
                                 value",
                                usize::from(i)
                            )));
                        }
                        let raw = self.load_reg(current_block, reg)?;
                        self.decode_scalar_key_reg_raw_value(current_block, reg, raw)
                    }
                }
            } else {
                self.load_reg(current_block, reg)?
            };
            user_arg_values.push(val);
        }
        let callsite_shape = if argc == 0 {
            self.callee_return_shapes.get(&op_idx).cloned().flatten()
        } else if let Some(chunk) = self.config.source_chunk {
            super::infer_callee_return_shape_for_args(
                chunk,
                op_idx,
                &arg_shapes,
                self.config.state_layout.as_ref(),
            )
        } else {
            None
        };
        let callee_lowered_shape =
            self.inferred_callee_return_shape_for_lowered_args(op_idx, usize::from(argc));
        // WP-20 diagnostic: how a self-recursive callee's parameter convention
        // and RETURN domain were derived at this callsite. Opt-in, off by
        // default; the numbers it prints are the ones the admission dump's
        // decline reasons are read against.
        if self_recursive_callee && super::wp20_debug() {
            eprintln!(
                "[trust_cg-self-recursive] callee={op_idx} is_self_call={is_self_call} \
                 raw_arg_shapes={raw_arg_shapes:?} \
                 expected_arg_shapes={expected_arg_shapes:?} normalized_arg_shapes={arg_shapes:?} \
                 callsite_return_shape={callsite_shape:?} callee_lowered_return_shape={callee_lowered_shape:?}"
            );
        }
        let callsite_abi_shape = Self::compact_return_abi_shape(callsite_shape.clone());
        let completed_callee_lowered_shape = if let (Some(callee), Some(callsite_abi)) =
            (callee_lowered_shape.as_ref(), callsite_abi_shape.as_ref())
        {
            Some(
                Self::complete_inferred_compact_shape_from_expected(callee, callsite_abi)
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "Call compact compound return shape for callee {op_idx} differs between callsite ABI and callee lowering: callsite_abi={callsite_abi:?}, callee={callee:?}"
                        ))
                    })?,
            )
        } else {
            callee_lowered_shape.clone()
        };
        let callee_abi_shape =
            Self::compact_return_abi_shape(completed_callee_lowered_shape.clone());
        let aggregate_return_shape = match (callsite_abi_shape.as_ref(), callee_abi_shape.as_ref())
        {
            (Some(callsite_abi), Some(callee_abi)) => {
                if !Self::same_compact_physical_layout(callsite_abi, callee_abi) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Call compact compound return ABI for callee {op_idx} differs between caller and callee: callsite_abi={callsite_abi:?}, callee_abi={callee_abi:?}"
                    )));
                }
                self.record_callee_expected_return_abi_shape(op_idx, callsite_abi)?;
                Some(callsite_abi.clone())
            }
            (Some(callsite_abi), None) => {
                self.record_callee_expected_return_abi_shape(op_idx, callsite_abi)?;
                Some(callsite_abi.clone())
            }
            (None, Some(callee_abi)) => {
                self.record_callee_expected_return_abi_shape(op_idx, callee_abi)?;
                Some(callee_abi.clone())
            }
            _ => None,
        };
        let compact_model_value_set_return_shape = self
            .structurally_known_model_value_setdiff_return_shape(op_idx)
            .filter(|compact_shape| {
                aggregate_return_shape.as_ref().is_some_and(|abi_shape| {
                    Self::materialized_model_value_set_return_compatible(abi_shape, compact_shape)
                })
            });
        let aggregate_return = if let Some(shape) = aggregate_return_shape.as_ref() {
            Some(Self::caller_owned_return_slot_count(shape).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "Call compact compound return requires fixed-width shape for callee {op_idx}, got {shape:?}"
                ))
            })?)
        } else {
            None
        };
        let result_shape = compact_model_value_set_return_shape
            .clone()
            .or_else(|| aggregate_return_shape.clone())
            .or_else(|| callsite_shape.clone())
            .or_else(|| completed_callee_lowered_shape.clone());
        let result_function_domain = if matches!(
            result_shape.as_ref(),
            Some(super::AggregateShape::Function {
                domain_lo: None,
                ..
            })
        ) {
            let domain = self
                .infer_callee_return_function_domain_for_args(
                    op_idx,
                    &arg_shapes,
                    &arg_function_domains,
                )
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Call compact function return for callee {op_idx} requires explicit-domain metadata"
                    ))
                })?;
            self.record_callee_expected_return_function_domain(op_idx, domain.clone())?;
            Some(domain)
        } else {
            None
        };
        let return_ptr = self.alloc_aggregate(current_block, aggregate_return.unwrap_or(1));

        // Build the call arguments: context pointers first, then a hidden
        // caller-owned fixed-width aggregate return buffer, then user args.
        // Scalar callees ignore the buffer and keep returning i64 directly.
        let mut call_args = vec![self.out_ptr, self.state_in_ptr];
        if let Some(sop) = self.state_out_ptr {
            call_args.push(sop);
        }
        // state_len: we don't track a ValueId for it in the callee case,
        // so emit a dummy constant 0 for state_len (unused by callee ops,
        // but must be present to match the signature).
        let state_len_val = self.emit_with_result(
            current_block,
            Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(0),
            },
        );
        call_args.push(state_len_val);
        call_args.push(return_ptr);
        call_args.extend(user_arg_values);

        // WP-20: self-recursive callees take a hidden trailing depth argument.
        // External callsites seed it with 0. The self-callsite guards
        // `depth < SELF_RECURSION_DEPTH_LIMIT` — on exceed the action returns
        // a typed `TypeMismatch` runtime error (recoverable per-state
        // interpreter fallback), never a native stack overflow — and passes
        // `depth + 1`.
        if self_recursive_callee {
            let depth_arg = if is_self_call {
                let depth = self.callee_depth_param.ok_or_else(|| {
                    TrustIrError::Emission(format!(
                        "self-recursive callee {op_idx} is being lowered without a depth parameter"
                    ))
                })?;
                let limit =
                    self.emit_i64_const(current_block, super::SELF_RECURSION_DEPTH_LIMIT);
                let below_limit = self.emit_with_result(
                    current_block,
                    Inst::ICmp {
                        op: ICmpOp::Slt,
                        ty: Ty::I64,
                        lhs: depth,
                        rhs: limit,
                    },
                );
                let error_blk = self.new_aux_block("self_recursion_depth_error");
                let accept_blk = self.new_aux_block("self_recursion_depth_ok");
                let error_id = self.block_id_of(error_blk);
                let accept_id = self.block_id_of(accept_blk);
                self.emit(
                    current_block,
                    InstrNode::new(Inst::CondBr {
                        cond: below_limit,
                        then_target: accept_id,
                        then_args: vec![],
                        else_target: error_id,
                        else_args: vec![],
                    }),
                );
                self.emit_runtime_error_and_return(
                    error_blk,
                    tla_jit_abi::JitRuntimeErrorKind::TypeMismatch,
                );
                current_block = accept_blk;
                let one = self.emit_i64_const(current_block, 1);
                self.emit_with_result(
                    current_block,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: depth,
                        rhs: one,
                    },
                )
            } else {
                self.emit_i64_const(current_block, 0)
            };
            call_args.push(depth_arg);
        }

        // Helper calls share the entrypoint JitCallOut. Native fused callers
        // preseed status with RuntimeError as a fail-closed sentinel before
        // invoking a compiled callout, while successful helper callees return
        // their scalar result or encoded hidden compact compound return-buffer
        // pointer and do not rewrite status. Reset status
        // immediately before the helper call so the post-call status check sees
        // only this callee's failure signal.
        let status_ptr = self.emit_out_field_ptr(current_block, super::STATUS_OFFSET);
        let ok = self.emit_with_result(
            current_block,
            Inst::Const {
                ty: Ty::I8,
                value: Constant::Int(i128::from(JitStatus::Ok as u8)),
            },
        );
        self.emit(
            current_block,
            InstrNode::new(Inst::Store {
                ty: Ty::I8,
                ptr: status_ptr,
                value: ok,
                align: None,
                volatile: false,
            }),
        );

        // Emit the trust-ir Call instruction.
        let result = self.emit_with_result(
            current_block,
            Inst::Call {
                callee: callee_func_id,
                args: call_args,
            },
        );

        // Helper calls share the same JitCallOut as the entrypoint. If the
        // callee surfaced RuntimeError / FallbackNeeded / PartialPass, the
        // current function must stop immediately instead of continuing and
        // potentially overwriting that status with Ok.
        let status = self.emit_with_result(
            current_block,
            Inst::Load {
                ty: Ty::I8,
                ptr: status_ptr,
                align: None,
                volatile: false,
            },
        );
        let status_is_ok = self.emit_with_result(
            current_block,
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I8,
                lhs: status,
                rhs: ok,
            },
        );

        let continue_block = self.new_aux_block("call_ok");
        let propagate_block = self.new_aux_block("call_status");
        let continue_id = self.block_id_of(continue_block);
        let propagate_id = self.block_id_of(propagate_block);
        // Store the call result before splitting on status so the successor
        // block does not consume a ValueId defined in this predecessor block.
        // Compact compound callees return the caller-owned return-buffer
        // pointer as i64; make that returned ABI value the source of truth so
        // native direct-call lowering does not need to keep the pre-call
        // alloca pointer live across the call.
        if aggregate_return.is_some() {
            self.store_reg_value(current_block, rd, result)?;
            if aggregate_return_shape
                .as_ref()
                .is_some_and(Self::is_compact_compound_aggregate)
            {
                let result_ptr = self.emit_with_result(
                    current_block,
                    Inst::Cast {
                        op: CastOp::IntToPtr,
                        src_ty: Ty::I64,
                        dst_ty: Ty::Ptr,
                        operand: result,
                    },
                );
                self.compact_state_slots.insert(
                    rd,
                    super::CompactStateSlot::pointer_backed_in_block(result_ptr, 0, current_block),
                );
                self.aggregate_pointer_regs
                    .insert(rd, super::AggregatePointerKind::Compact);
            } else {
                self.compact_state_slots.remove(&rd);
                self.aggregate_pointer_regs
                    .insert(rd, super::AggregatePointerKind::Flat);
            }
        } else {
            self.store_reg_value(current_block, rd, result)?;
            self.compact_state_slots.remove(&rd);
            self.aggregate_pointer_regs.remove(&rd);
        }
        self.const_scalar_values.remove(&rd);

        self.emit(
            current_block,
            InstrNode::new(Inst::CondBr {
                cond: status_is_ok,
                then_target: continue_id,
                then_args: vec![],
                else_target: propagate_id,
                else_args: vec![],
            }),
        );

        self.emit_passthrough_status_return(propagate_block);

        if let Some(shape) = compact_model_value_set_return_shape.as_ref() {
            self.compact_materialized_model_value_set_return(continue_block, rd, shape)?;
        }

        if let Some(shape) = result_shape {
            if let Some(len) = shape.tracked_len() {
                self.const_set_sizes.insert(rd, len);
            } else {
                self.const_set_sizes.remove(&rd);
            }
            self.aggregate_shapes.insert(rd, shape);
            if let Some(domain) = result_function_domain {
                self.compact_function_domains.insert(rd, domain);
            } else {
                self.compact_function_domains.remove(&rd);
            }
            // WP-28: the return convention is proven by the tracked shape.
            self.untracked_callee_return_regs.remove(&rd);
        } else {
            self.aggregate_shapes.remove(&rd);
            self.compact_function_domains.remove(&rd);
            self.const_set_sizes.remove(&rd);
            // WP-28: no inferred return shape — remember that this register's
            // physical convention is unproven so the self-recursive callsite
            // (whose raw-argument convention is shape-driven) fails closed
            // rather than silently consuming a union INDEX as a member value.
            self.untracked_callee_return_regs.insert(rd);
        }

        Ok(Some(continue_block))
    }

    fn structurally_known_model_value_setdiff_return_shape(
        &self,
        op_idx: u16,
    ) -> Option<super::AggregateShape> {
        let chunk = self.config.source_chunk?;
        let func = chunk.functions.get(usize::from(op_idx))?;
        let mut result = None;
        for (pc, opcode) in func.instructions.iter().enumerate() {
            let Opcode::Ret { rs } = opcode else {
                continue;
            };
            let shape = self.model_value_setdiff_subset_shape_for_reg(func, *rs, pc, 0)?;
            if result.as_ref().is_some_and(|existing| existing != &shape) {
                return None;
            }
            result = Some(shape);
        }
        result
    }

    fn model_value_setdiff_subset_shape_for_reg(
        &self,
        func: &BytecodeFunction,
        reg: u8,
        before_pc: usize,
        depth: usize,
    ) -> Option<super::AggregateShape> {
        if depth > 16 {
            return None;
        }
        let (def_pc, opcode) = Self::find_register_definition_before(func, reg, before_pc)?;
        match opcode {
            Opcode::SetDiff { r1, .. } => {
                self.exact_model_value_set_shape_for_reg(func, *r1, def_pc, depth + 1)
            }
            Opcode::Move { rs, .. } => {
                self.model_value_setdiff_subset_shape_for_reg(func, *rs, def_pc, depth + 1)
            }
            _ => None,
        }
    }

    fn exact_model_value_set_shape_for_reg(
        &self,
        func: &BytecodeFunction,
        reg: u8,
        before_pc: usize,
        depth: usize,
    ) -> Option<super::AggregateShape> {
        if depth > 16 {
            return None;
        }
        let chunk = self.config.source_chunk?;
        let (def_pc, opcode) = Self::find_register_definition_before(func, reg, before_pc)?;
        match opcode {
            Opcode::LoadConst { idx, .. } => {
                Self::exact_model_value_set_shape(chunk.constants.get_value(*idx))
            }
            Opcode::Move { rs, .. } => {
                self.exact_model_value_set_shape_for_reg(func, *rs, def_pc, depth + 1)
            }
            _ => None,
        }
    }

    fn exact_model_value_set_shape(value: &Value) -> Option<super::AggregateShape> {
        let Value::Set(set) = value else {
            return None;
        };
        let mut elements = Vec::with_capacity(set.len());
        for value in set.iter() {
            let Value::ModelValue(name) = value else {
                return None;
            };
            elements.push(SetBitmaskElement::ModelValue(tla_core::intern_name(name)));
        }
        if elements.is_empty() || elements.len() > 63 {
            return None;
        }
        let universe_len = u32::try_from(elements.len()).ok()?;
        Some(super::AggregateShape::SetBitmask {
            universe_len,
            universe: super::SetBitmaskUniverse::Exact(elements),
        })
    }

    fn find_register_definition_before(
        func: &BytecodeFunction,
        reg: u8,
        before_pc: usize,
    ) -> Option<(usize, &Opcode)> {
        func.instructions
            .get(..before_pc)?
            .iter()
            .enumerate()
            .rev()
            .find(|(_, opcode)| Self::opcode_dest_reg(opcode) == Some(reg))
    }

    pub(super) fn opcode_dest_reg(opcode: &Opcode) -> Option<u8> {
        match opcode {
            Opcode::LoadConst { rd, .. }
            | Opcode::LoadImm { rd, .. }
            | Opcode::LoadBool { rd, .. }
            | Opcode::LoadVar { rd, .. }
            | Opcode::LoadPrime { rd, .. }
            | Opcode::Move { rd, .. }
            | Opcode::Call { rd, .. }
            | Opcode::ValueApply { rd, .. }
            | Opcode::CallExternal { rd, .. }
            | Opcode::CallBuiltin { rd, .. }
            | Opcode::RecordNew { rd, .. }
            | Opcode::RecordGet { rd, .. }
            | Opcode::Domain { rd, .. }
            | Opcode::TupleNew { rd, .. }
            | Opcode::TupleGet { rd, .. }
            | Opcode::SeqNew { rd, .. }
            | Opcode::Range { rd, .. }
            | Opcode::SetEnum { rd, .. }
            | Opcode::SetUnion { rd, .. }
            | Opcode::SetIntersect { rd, .. }
            | Opcode::SetDiff { rd, .. }
            | Opcode::SetIn { rd, .. }
            | Opcode::Tuple2SetIn { rd, .. }
            | Opcode::SetEnumSubseteq { rd, .. }
            | Opcode::Tuple2SelfEq { rd, .. }
            | Opcode::Tuple2SelfSubseteq { rd, .. }
            | Opcode::RoundStepEq { rd, .. }
            | Opcode::Subseteq { rd, .. }
            | Opcode::Powerset { rd, .. }
            | Opcode::BigUnion { rd, .. }
            | Opcode::KSubset { rd, .. }
            | Opcode::FuncSet { rd, .. }
            | Opcode::FuncApply { rd, .. }
            | Opcode::FuncExcept { rd, .. }
            | Opcode::FuncDef { rd, .. }
            | Opcode::FuncDefBegin { rd, .. }
            | Opcode::RecordSet { rd, .. }
            | Opcode::Times { rd, .. }
            | Opcode::SetFilterBegin { rd, .. }
            | Opcode::ForallBegin { rd, .. }
            | Opcode::ExistsBegin { rd, .. }
            | Opcode::ChooseBegin { rd, .. }
            | Opcode::SetBuilderBegin { rd, .. }
            | Opcode::CondMove { rd, .. }
            | Opcode::Unchanged { rd, .. }
            | Opcode::MakeClosure { rd, .. }
            | Opcode::Eq { rd, .. }
            | Opcode::Neq { rd, .. }
            | Opcode::LtInt { rd, .. }
            | Opcode::LeInt { rd, .. }
            | Opcode::GtInt { rd, .. }
            | Opcode::GeInt { rd, .. }
            | Opcode::AddInt { rd, .. }
            | Opcode::SubInt { rd, .. }
            | Opcode::MulInt { rd, .. }
            | Opcode::DivInt { rd, .. }
            | Opcode::IntDiv { rd, .. }
            | Opcode::ModInt { rd, .. }
            | Opcode::NegInt { rd, .. }
            | Opcode::PowInt { rd, .. }
            | Opcode::And { rd, .. }
            | Opcode::Or { rd, .. }
            | Opcode::Not { rd, .. }
            | Opcode::Implies { rd, .. }
            | Opcode::Equiv { rd, .. }
            | Opcode::StrConcat { rd, .. }
            | Opcode::Concat { rd, .. }
            // Fused Eq superinstructions (implied-action eval-VM compile
            // only; never present in bytecode handed to the trust-ir
            // lowering — the main lowering match rejects them as
            // unsupported if one ever appears).
            | Opcode::EqFuncExcept { rd, .. }
            | Opcode::EqRecordNew { rd, .. } => Some(*rd),
            Opcode::StoreVar { .. }
            | Opcode::Jump { .. }
            | Opcode::JumpTrue { .. }
            | Opcode::JumpFalse { .. }
            | Opcode::SetPrimeMode { .. }
            | Opcode::ForallNext { .. }
            | Opcode::ExistsNext { .. }
            | Opcode::ChooseNext { .. }
            | Opcode::LoopNext { .. }
            | Opcode::Ret { .. }
            | Opcode::Halt
            | Opcode::Nop => None,
        }
    }

    fn materialized_model_value_set_return_compatible(
        source: &super::AggregateShape,
        compact: &super::AggregateShape,
    ) -> bool {
        let super::AggregateShape::SetBitmask {
            universe_len,
            universe: super::SetBitmaskUniverse::Exact(elements),
        } = compact
        else {
            return false;
        };
        if usize::try_from(*universe_len).ok() != Some(elements.len()) {
            return false;
        }
        match source {
            super::AggregateShape::Set {
                len,
                element: Some(element),
            }
            | super::AggregateShape::BoundedSet {
                max_len: len,
                element: Some(element),
            } => {
                *len <= *universe_len
                    && matches!(
                        element.as_ref(),
                        super::AggregateShape::Scalar(super::ScalarShape::ModelValue)
                    )
            }
            _ => false,
        }
    }

    fn compact_materialized_model_value_set_return(
        &mut self,
        block_idx: usize,
        rd: u8,
        compact_shape: &super::AggregateShape,
    ) -> Result<(), TrustIrError> {
        let super::AggregateShape::SetBitmask {
            universe_len,
            universe: super::SetBitmaskUniverse::Exact(elements),
        } = compact_shape
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call model-value Set return compaction requires exact SetBitmask shape, got {compact_shape:?}"
            )));
        };
        if usize::try_from(*universe_len).ok() != Some(elements.len()) || *universe_len > 63 {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call model-value Set return compaction requires a valid exact SetBitmask universe, got {compact_shape:?}"
            )));
        }

        let ptr_i64 = self.load_reg(block_idx, rd)?;
        let source_ptr = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::IntToPtr,
                src_ty: Ty::I64,
                dst_ty: Ty::Ptr,
                operand: ptr_i64,
            },
        );
        let runtime_len = self.load_at_offset(block_idx, source_ptr, 0);
        let zero = self.emit_i64_const(block_idx, 0);
        let mut mask = zero;

        for (bit_idx, element) in elements.iter().enumerate() {
            let SetBitmaskElement::ModelValue(name) = element else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Call model-value Set return compaction requires model-value universe elements, got {compact_shape:?}"
                )));
            };
            let bit = self.emit_i64_const(block_idx, 1_i64 << bit_idx);
            let expected = self.emit_i64_const(block_idx, i64::from(name.0));
            let mut element_mask = zero;
            for slot in 1..=*universe_len {
                let slot_value = self.load_at_offset(block_idx, source_ptr, slot);
                let slot_matches = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: slot_value,
                        rhs: expected,
                    },
                );
                let slot_idx = self.emit_i64_const(block_idx, i64::from(slot));
                let slot_in_len = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Sle,
                        ty: Ty::I64,
                        lhs: slot_idx,
                        rhs: runtime_len,
                    },
                );
                let slot_matches_i64 = self.emit_with_result(
                    block_idx,
                    Inst::Cast {
                        op: CastOp::ZExt,
                        src_ty: Ty::Bool,
                        dst_ty: Ty::I64,
                        operand: slot_matches,
                    },
                );
                let slot_in_len_i64 = self.emit_with_result(
                    block_idx,
                    Inst::Cast {
                        op: CastOp::ZExt,
                        src_ty: Ty::Bool,
                        dst_ty: Ty::I64,
                        operand: slot_in_len,
                    },
                );
                let present_i64 = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: slot_matches_i64,
                        rhs: slot_in_len_i64,
                    },
                );
                let present = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Ne,
                        ty: Ty::I64,
                        lhs: present_i64,
                        rhs: zero,
                    },
                );
                let selected = self.emit_with_result(
                    block_idx,
                    Inst::Select {
                        ty: Ty::I64,
                        cond: present,
                        then_val: bit,
                        else_val: zero,
                    },
                );
                element_mask = self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Or,
                        ty: Ty::I64,
                        lhs: element_mask,
                        rhs: selected,
                    },
                );
            }
            mask = self.emit_with_result(
                block_idx,
                Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
                    lhs: mask,
                    rhs: element_mask,
                },
            );
        }

        self.store_reg_value(block_idx, rd, mask)?;
        Ok(())
    }

    fn compact_helper_arg_abi_shape(
        &self,
        op_idx: u16,
        arg_idx: usize,
        expected_shape: Option<&super::AggregateShape>,
    ) -> Result<Option<super::AggregateShape>, TrustIrError> {
        let Some(expected_shape) = expected_shape else {
            return Ok(None);
        };
        let use_compact = match expected_shape {
            super::AggregateShape::Record { .. } | super::AggregateShape::Sequence { .. } => true,
            super::AggregateShape::RecordSet { .. } => true,
            super::AggregateShape::Function { .. } => {
                self.callee_arg_flows_to_compact_operator(op_idx, arg_idx)
                    && !self.callee_arg_flows_to_generic_fold(op_idx, arg_idx)
            }
            _ => false,
        };
        if !use_compact {
            return Ok(None);
        }
        Self::compact_return_abi_shape(Some(expected_shape.clone()))
            .map(Some)
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "Call compact aggregate argument {arg_idx} for callee {op_idx} requires fixed-width ABI shape, got {expected_shape:?}"
                ))
            })
    }

    fn callee_arg_flows_to_compact_operator(&self, op_idx: u16, arg_idx: usize) -> bool {
        let Some(chunk) = self.config.source_chunk else {
            return true;
        };
        let Some(func) = chunk.functions.get(usize::from(op_idx)) else {
            return true;
        };
        let Ok(arg_reg) = u8::try_from(arg_idx) else {
            return false;
        };
        let mut aliases = std::collections::HashSet::from([arg_reg]);
        for opcode in &func.instructions {
            match *opcode {
                Opcode::FuncApply { func, .. } if aliases.contains(&func) => return true,
                Opcode::FuncExcept { func, .. } if aliases.contains(&func) => return true,
                Opcode::StoreVar { rs, .. } if aliases.contains(&rs) => return true,
                Opcode::Domain { rs, .. } if aliases.contains(&rs) => return true,
                Opcode::SetIn { set, .. } if aliases.contains(&set) => return true,
                Opcode::Ret { rs } if aliases.contains(&rs) => return true,
                Opcode::Call {
                    args_start, argc, ..
                } => {
                    for i in 0..argc {
                        if args_start
                            .checked_add(i)
                            .is_some_and(|reg| aliases.contains(&reg))
                        {
                            return true;
                        }
                    }
                }
                Opcode::Move { rd, rs } => {
                    if aliases.contains(&rs) {
                        aliases.insert(rd);
                    } else {
                        aliases.remove(&rd);
                    }
                    continue;
                }
                _ => {}
            }
            if let Some(rd) = Self::opcode_dest_reg(opcode) {
                aliases.remove(&rd);
            }
        }
        false
    }

    fn callee_arg_flows_to_generic_fold(&self, op_idx: u16, arg_idx: usize) -> bool {
        use tla_tir::bytecode::BuiltinOp;

        let Some(chunk) = self.config.source_chunk else {
            return false;
        };
        let Some(func) = chunk.functions.get(usize::from(op_idx)) else {
            return false;
        };
        let Ok(arg_reg) = u8::try_from(arg_idx) else {
            return false;
        };
        let mut aliases = std::collections::HashSet::from([arg_reg]);
        for opcode in &func.instructions {
            match *opcode {
                Opcode::CallBuiltin {
                    builtin: BuiltinOp::FoldFunctionOnSetSum,
                    args_start,
                    argc,
                    ..
                } => {
                    for i in 0..argc {
                        if args_start
                            .checked_add(i)
                            .is_some_and(|reg| aliases.contains(&reg))
                        {
                            return true;
                        }
                    }
                }
                Opcode::Move { rd, rs } => {
                    if aliases.contains(&rs) {
                        aliases.insert(rd);
                    } else {
                        aliases.remove(&rd);
                    }
                    continue;
                }
                _ => {}
            }
            if let Some(rd) = Self::opcode_dest_reg(opcode) {
                aliases.remove(&rd);
            }
        }
        false
    }

    pub(super) fn infer_callee_return_function_domain_for_args(
        &self,
        op_idx: u16,
        arg_shapes: &[Option<super::AggregateShape>],
        arg_domains: &[Option<super::CompactFunctionDomain>],
    ) -> Option<super::CompactFunctionDomain> {
        let chunk = self.config.source_chunk?;
        let func = chunk.functions.get(usize::from(op_idx))?;
        let mut shape_cache = std::collections::HashMap::new();
        let mut shape_visiting = std::collections::HashSet::new();
        let mut domain_visiting = std::collections::HashSet::new();
        self.infer_function_return_domain_with_params(
            op_idx,
            func,
            chunk,
            arg_shapes,
            arg_domains,
            &mut shape_cache,
            &mut shape_visiting,
            &mut domain_visiting,
        )
    }

    fn infer_function_return_domain_with_params(
        &self,
        op_idx: u16,
        func: &BytecodeFunction,
        chunk: &tla_tir::bytecode::BytecodeChunk,
        param_shapes: &[Option<super::AggregateShape>],
        param_domains: &[Option<super::CompactFunctionDomain>],
        shape_cache: &mut std::collections::HashMap<u16, Option<super::AggregateShape>>,
        shape_visiting: &mut std::collections::HashSet<u16>,
        domain_visiting: &mut std::collections::HashSet<u16>,
    ) -> Option<super::CompactFunctionDomain> {
        if super::has_unmodeled_shape_inference_loop(func) {
            return None;
        }
        if !domain_visiting.insert(op_idx) {
            return None;
        }
        let result = if super::uses_branch_return_shape_inference(func) {
            self.infer_function_return_domain_cfg(
                func,
                chunk,
                param_shapes,
                param_domains,
                shape_cache,
                shape_visiting,
                domain_visiting,
            )
        } else {
            let mut summary = super::seed_shape_summary(func, param_shapes, param_domains);
            let mut return_domain = None;
            for opcode in &func.instructions {
                match *opcode {
                    Opcode::Ret { rs } => {
                        return_domain = summary.compact_function_domains.get(&rs).cloned();
                        break;
                    }
                    opcode => {
                        let before = summary.clone();
                        super::apply_shape_transfer(
                            &mut summary,
                            opcode,
                            chunk,
                            self.config.state_layout.as_ref(),
                            shape_cache,
                            shape_visiting,
                        )?;
                        self.apply_function_domain_transfer(
                            &before,
                            &mut summary,
                            opcode,
                            chunk,
                            shape_cache,
                            shape_visiting,
                            domain_visiting,
                        );
                    }
                }
            }
            return_domain
        };
        domain_visiting.remove(&op_idx);
        result
    }

    fn infer_function_return_domain_cfg(
        &self,
        func: &BytecodeFunction,
        chunk: &tla_tir::bytecode::BytecodeChunk,
        param_shapes: &[Option<super::AggregateShape>],
        param_domains: &[Option<super::CompactFunctionDomain>],
        shape_cache: &mut std::collections::HashMap<u16, Option<super::AggregateShape>>,
        shape_visiting: &mut std::collections::HashSet<u16>,
        domain_visiting: &mut std::collections::HashSet<u16>,
    ) -> Option<super::CompactFunctionDomain> {
        if func.instructions.is_empty() {
            return None;
        }
        let len = func.instructions.len();
        let mut facts = vec![None; len];
        let mut worklist = std::collections::VecDeque::new();
        facts[0] = Some(super::seed_shape_summary(func, param_shapes, param_domains));
        worklist.push_back(0);

        let mut saw_return = false;
        let mut return_domain = None;
        while let Some(pc) = worklist.pop_front() {
            let Some(summary) = facts.get(pc).and_then(Clone::clone) else {
                continue;
            };
            let opcode = func.instructions[pc];
            match opcode {
                Opcode::Ret { rs } => {
                    let incoming = summary.compact_function_domains.get(&rs).cloned();
                    if !saw_return {
                        return_domain = incoming;
                        saw_return = true;
                    } else if return_domain != incoming {
                        return_domain = None;
                    }
                }
                Opcode::Jump { offset } => {
                    let target = super::shape_forward_target(pc, offset, len)?;
                    super::push_shape_fact(&mut facts, &mut worklist, target, summary)?;
                }
                Opcode::JumpTrue { offset, .. } | Opcode::JumpFalse { offset, .. } => {
                    let target = super::shape_forward_target(pc, offset, len)?;
                    super::push_shape_fact(&mut facts, &mut worklist, target, summary.clone())?;
                    let fallthrough = pc.checked_add(1)?;
                    if fallthrough < len {
                        super::push_shape_fact(&mut facts, &mut worklist, fallthrough, summary)?;
                    }
                }
                _ => {
                    let mut next = summary.clone();
                    super::apply_shape_transfer(
                        &mut next,
                        opcode,
                        chunk,
                        self.config.state_layout.as_ref(),
                        shape_cache,
                        shape_visiting,
                    )?;
                    self.apply_function_domain_transfer(
                        &summary,
                        &mut next,
                        opcode,
                        chunk,
                        shape_cache,
                        shape_visiting,
                        domain_visiting,
                    );
                    let fallthrough = pc.checked_add(1)?;
                    if fallthrough < len {
                        super::push_shape_fact(&mut facts, &mut worklist, fallthrough, next)?;
                    }
                }
            }
        }
        saw_return.then_some(return_domain).flatten()
    }

    fn apply_function_domain_transfer(
        &self,
        before: &super::ShapeSummary,
        after: &mut super::ShapeSummary,
        opcode: Opcode,
        chunk: &tla_tir::bytecode::BytecodeChunk,
        shape_cache: &mut std::collections::HashMap<u16, Option<super::AggregateShape>>,
        shape_visiting: &mut std::collections::HashSet<u16>,
        domain_visiting: &mut std::collections::HashSet<u16>,
    ) {
        match opcode {
            Opcode::LoadVar { rd, var_idx } | Opcode::LoadPrime { rd, var_idx } => {
                if let Some(domain) = self.compact_function_domain_from_state_var(var_idx) {
                    after.set_function_domain(rd, domain);
                }
            }
            Opcode::Move { rd, rs } => {
                if let Some(domain) = before.compact_function_domains.get(&rs).cloned() {
                    after.set_function_domain(rd, domain);
                }
            }
            Opcode::FuncApply { rd, func, .. } => {
                let domain = before
                    .compact_function_domains
                    .get(&func)
                    .and_then(|domain| {
                        before
                            .aggregate_shapes
                            .get(&func)
                            .and_then(super::AggregateShape::function_value_shape)
                            .and_then(|value_shape| {
                                Self::call_same_sized_explicit_function_value_domain(
                                    domain,
                                    Some(&value_shape),
                                )
                            })
                    })
                    .or_else(|| {
                        before.state_var_sources.get(&func).and_then(|var_idx| {
                            self.compact_function_value_domain_from_state_var(*var_idx)
                        })
                    });
                if let Some(domain) = domain {
                    after.set_function_domain(rd, domain);
                }
            }
            Opcode::FuncExcept { rd, func, .. } => {
                if let Some(domain) = before.compact_function_domains.get(&func).cloned() {
                    after.set_function_domain(rd, domain);
                }
            }
            Opcode::CondMove { rd, cond, rs } => {
                let selected = before
                    .const_scalar_values
                    .get(&cond)
                    .copied()
                    .and_then(|cond_value| {
                        if cond_value != 0 {
                            before.compact_function_domains.get(&rs).cloned()
                        } else {
                            before.compact_function_domains.get(&rd).cloned()
                        }
                    })
                    .or_else(|| {
                        let left = before.compact_function_domains.get(&rd)?;
                        let right = before.compact_function_domains.get(&rs)?;
                        (left == right).then(|| left.clone())
                    });
                if let Some(domain) = selected {
                    after.set_function_domain(rd, domain);
                }
            }
            Opcode::Call {
                rd,
                op_idx,
                args_start,
                argc,
            } => {
                let mut arg_shapes = Vec::with_capacity(usize::from(argc));
                let mut arg_domains = Vec::with_capacity(usize::from(argc));
                for i in 0..argc {
                    let Some(reg) = args_start.checked_add(i) else {
                        return;
                    };
                    arg_shapes.push(before.aggregate_shapes.get(&reg).cloned());
                    arg_domains.push(before.compact_function_domains.get(&reg).cloned());
                }
                let Some(callee) = chunk.functions.get(usize::from(op_idx)) else {
                    return;
                };
                let Some(domain) = self.infer_function_return_domain_with_params(
                    op_idx,
                    callee,
                    chunk,
                    &arg_shapes,
                    &arg_domains,
                    shape_cache,
                    shape_visiting,
                    domain_visiting,
                ) else {
                    return;
                };
                after.set_function_domain(rd, domain);
            }
            _ => {}
        }
    }

    fn compact_function_domain_from_state_var(
        &self,
        var_idx: u16,
    ) -> Option<super::CompactFunctionDomain> {
        let layout = self
            .config
            .state_layout
            .as_ref()?
            .var_layout(usize::from(var_idx))?;
        let tla_jit_abi::VarLayout::Compound(layout) = layout else {
            return None;
        };
        self.compact_function_domain_from_layout(layout)
    }

    fn compact_function_value_domain_from_state_var(
        &self,
        var_idx: u16,
    ) -> Option<super::CompactFunctionDomain> {
        let layout = self
            .config
            .state_layout
            .as_ref()?
            .var_layout(usize::from(var_idx))?;
        let tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
            value_layout,
            ..
        }) = layout
        else {
            return None;
        };
        self.compact_function_domain_from_layout(value_layout)
    }

    fn call_same_sized_explicit_function_value_domain(
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

    fn materialize_compact_helper_call_arg(
        &mut self,
        block_idx: usize,
        op_idx: u16,
        arg_idx: usize,
        reg: u8,
        expected_shape: Option<&super::AggregateShape>,
        compact_arg_abi_shape: Option<&super::AggregateShape>,
    ) -> Result<Option<(usize, ValueId)>, TrustIrError> {
        if let Some(abi_shape) = compact_arg_abi_shape {
            let materialized = self.materialize_compact_aggregate_call_arg_to_abi(
                block_idx, op_idx, arg_idx, reg, abi_shape,
            )?;
            return Ok(Some(materialized));
        }

        if let Some(value) = self.materialize_materialized_finite_set_call_arg(
            block_idx,
            op_idx,
            arg_idx,
            reg,
            expected_shape,
        )? {
            return Ok(Some(value));
        }
        if let Some(value) = self.materialize_compact_function_call_arg(block_idx, reg)? {
            return Ok(Some((block_idx, value)));
        }
        if let Some(value) = self.materialize_compact_sequence_call_arg(block_idx, reg)? {
            return Ok(Some(value));
        }
        self.materialize_compact_record_call_arg(block_idx, reg)
    }

    fn materialize_materialized_finite_set_call_arg(
        &mut self,
        block_idx: usize,
        op_idx: u16,
        arg_idx: usize,
        reg: u8,
        expected_shape: Option<&super::AggregateShape>,
    ) -> Result<Option<(usize, ValueId)>, TrustIrError> {
        let Some(expected_shape) = expected_shape else {
            return Ok(None);
        };
        if !expected_shape.is_finite_set_shape() {
            return Ok(None);
        }
        let Some(source_shape) = self.aggregate_shapes.get(&reg).cloned() else {
            return Ok(None);
        };
        if !source_shape.is_finite_set_shape()
            && !matches!(source_shape, super::AggregateShape::StateValue)
        {
            return Ok(None);
        }
        let Some(slot_count) = expected_shape.materialized_return_slot_count() else {
            if matches!(source_shape, super::AggregateShape::SetBitmask { .. })
                && !matches!(expected_shape, super::AggregateShape::SetBitmask { .. })
            {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Call compact SetBitmask argument {arg_idx} for callee {op_idx} r{reg} cannot be passed to unbounded finite-set ABI shape {expected_shape:?}"
                )));
            }
            return Ok(None);
        };
        let result_ptr = self.alloc_aggregate(block_idx, slot_count);
        let max_len = slot_count.checked_sub(1).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call finite-set argument {arg_idx} for callee {op_idx} materialized ABI has no len slot: {expected_shape:?}"
            ))
        })?;
        if let super::AggregateShape::SetBitmask {
            universe_len,
            universe,
        } = source_shape
        {
            let current_block = self.expand_setbitmask_call_arg_to_materialized_set(
                block_idx,
                op_idx,
                arg_idx,
                reg,
                result_ptr,
                max_len,
                universe_len,
                &universe,
            )?;
            return Ok(Some((
                current_block,
                self.ptr_to_i64(current_block, result_ptr),
            )));
        }

        let source_ptr = self.load_reg_as_ptr(block_idx, reg)?;
        let current_block = self
            .copy_bounded_materialized_return_buffer(block_idx, source_ptr, result_ptr, max_len)?;
        Ok(Some((
            current_block,
            self.ptr_to_i64(current_block, result_ptr),
        )))
    }

    fn expand_setbitmask_call_arg_to_materialized_set(
        &mut self,
        block_idx: usize,
        op_idx: u16,
        arg_idx: usize,
        reg: u8,
        result_ptr: ValueId,
        max_len: u32,
        universe_len: u32,
        universe: &super::SetBitmaskUniverse,
    ) -> Result<usize, TrustIrError> {
        if matches!(universe, super::SetBitmaskUniverse::Unknown) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call compact SetBitmask argument {arg_idx} for callee {op_idx} r{reg} cannot materialize an unknown universe as a finite-set buffer"
            )));
        }
        let mask = self.load_reg(block_idx, reg)?;
        let valid_mask =
            Self::compact_set_bitmask_valid_mask(universe_len, "Call compact SetBitmask argument")?;
        let valid_mask = self.emit_i64_const(block_idx, valid_mask);
        let mask = self.emit_with_result(
            block_idx,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: mask,
                rhs: valid_mask,
            },
        );

        let len_alloca = self.emit_with_result(
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
                ptr: len_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let mut current_block = block_idx;
        for bit_idx in 0..universe_len {
            let current_len = self.emit_with_result(
                current_block,
                Inst::Load {
                    ty: Ty::I64,
                    ptr: len_alloca,
                    align: None,
                    volatile: false,
                },
            );
            let bit = self.emit_i64_const(current_block, 1_i64 << bit_idx);
            let present_bits = self.emit_with_result(
                current_block,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: mask,
                    rhs: bit,
                },
            );
            let present = self.emit_with_result(
                current_block,
                Inst::ICmp {
                    op: ICmpOp::Ne,
                    ty: Ty::I64,
                    lhs: present_bits,
                    rhs: zero,
                },
            );
            let one = self.emit_i64_const(current_block, 1);
            let incremented_len = self.emit_with_result(
                current_block,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: current_len,
                    rhs: one,
                },
            );
            let next_len = self.emit_with_result(
                current_block,
                Inst::Select {
                    ty: Ty::I64,
                    cond: present,
                    then_val: incremented_len,
                    else_val: current_len,
                },
            );
            self.emit(
                current_block,
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: len_alloca,
                    value: next_len,
                    align: None,
                    volatile: false,
                }),
            );
        }

        let final_len = self.emit_with_result(
            current_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: len_alloca,
                align: None,
                volatile: false,
            },
        );
        current_block = self.guard_compact_sequence_len_in_bounds(
            current_block,
            final_len,
            max_len,
            "call_setbitmask_arg_len",
        );
        self.store_at_offset(current_block, result_ptr, 0, final_len);

        for slot in 1..=max_len {
            self.store_at_offset(current_block, result_ptr, slot, zero);
        }

        let one = self.emit_i64_const(current_block, 1);
        self.emit(
            current_block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: len_alloca,
                value: one,
                align: None,
                volatile: false,
            }),
        );

        for bit_idx in 0..universe_len {
            let bit = self.emit_i64_const(current_block, 1_i64 << bit_idx);
            let present_bits = self.emit_with_result(
                current_block,
                Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
                    lhs: mask,
                    rhs: bit,
                },
            );
            let present = self.emit_with_result(
                current_block,
                Inst::ICmp {
                    op: ICmpOp::Ne,
                    ty: Ty::I64,
                    lhs: present_bits,
                    rhs: zero,
                },
            );
            let store_block = self.new_aux_block("call_setbitmask_arg_store");
            let next_block = self.new_aux_block("call_setbitmask_arg_next");
            let store_id = self.block_id_of(store_block);
            let next_id = self.block_id_of(next_block);
            self.emit(
                current_block,
                InstrNode::new(Inst::CondBr {
                    cond: present,
                    then_target: store_id,
                    then_args: vec![],
                    else_target: next_id,
                    else_args: vec![],
                }),
            );

            let write_slot = self.emit_with_result(
                store_block,
                Inst::Load {
                    ty: Ty::I64,
                    ptr: len_alloca,
                    align: None,
                    volatile: false,
                },
            );
            let element = self.emit_setbitmask_element_value(store_block, universe, bit_idx)?;
            self.store_at_dynamic_offset(store_block, result_ptr, write_slot, element);
            let one = self.emit_i64_const(store_block, 1);
            let next_write_slot = self.emit_with_result(
                store_block,
                Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: write_slot,
                    rhs: one,
                },
            );
            self.emit(
                store_block,
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: len_alloca,
                    value: next_write_slot,
                    align: None,
                    volatile: false,
                }),
            );
            self.emit(
                store_block,
                InstrNode::new(Inst::Br {
                    target: next_id,
                    args: vec![],
                }),
            );
            current_block = next_block;
        }

        Ok(current_block)
    }

    fn emit_setbitmask_element_value(
        &mut self,
        block_idx: usize,
        universe: &super::SetBitmaskUniverse,
        bit_idx: u32,
    ) -> Result<ValueId, TrustIrError> {
        let value = match universe {
            super::SetBitmaskUniverse::IntRange { lo } => lo.checked_add(i64::from(bit_idx)),
            super::SetBitmaskUniverse::ExplicitInt(values) => values.get(bit_idx as usize).copied(),
            super::SetBitmaskUniverse::Exact(elements) => {
                elements.get(bit_idx as usize).map(|element| match element {
                    SetBitmaskElement::Int(value) => *value,
                    SetBitmaskElement::Bool(value) => i64::from(*value),
                    SetBitmaskElement::String(name) | SetBitmaskElement::ModelValue(name) => {
                        i64::from(name.0)
                    }
                })
            }
            super::SetBitmaskUniverse::Unknown => None,
        }
        .ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact SetBitmask argument cannot materialize unknown universe element {bit_idx}: {universe:?}"
            ))
        })?;
        Ok(self.emit_i64_const(block_idx, value))
    }

    fn materialize_compact_aggregate_call_arg_to_abi(
        &mut self,
        block_idx: usize,
        op_idx: u16,
        arg_idx: usize,
        reg: u8,
        expected_abi_shape: &super::AggregateShape,
    ) -> Result<(usize, ValueId), TrustIrError> {
        let expected_slots = expected_abi_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact aggregate argument {arg_idx} for callee {op_idx} requires fixed-width ABI shape, got {expected_abi_shape:?}"
            ))
        })?;
        let raw_source_shape = self.aggregate_shapes.get(&reg).cloned().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact aggregate argument {arg_idx} for callee {op_idx} requires tracked source shape for r{reg}, expected {expected_abi_shape:?}"
            ))
        })?;
        let source_shape = Self::complete_inferred_compact_shape_from_expected(
            &raw_source_shape,
            expected_abi_shape,
        )
        .unwrap_or(raw_source_shape);

        let result_ptr = self.alloc_aggregate(block_idx, expected_slots);
        let copied = if self.is_flat_funcdef_pair_list(reg) {
            if !Self::can_copy_flat_aggregate_to_compact_slots(&source_shape, expected_abi_shape) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Call compact aggregate argument {arg_idx} for callee {op_idx} requires compatible flat FuncDef source and ABI shapes for r{reg}, got {source_shape:?} -> {expected_abi_shape:?}"
                )));
            }
            let source_ptr = self.load_reg_as_ptr(block_idx, reg)?;
            self.copy_flat_aggregate_to_compact_slots(
                block_idx,
                source_ptr,
                &source_shape,
                expected_abi_shape,
                result_ptr,
                0,
                true,
            )?
        } else if let Some(source_slot) = self.compact_state_slots.get(&reg).copied() {
            if !Self::can_copy_compact_aggregate_to_compact_slots(&source_shape, expected_abi_shape)
            {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Call compact aggregate argument {arg_idx} for callee {op_idx} requires compatible compact source and ABI shapes for r{reg}, got {source_shape:?} -> {expected_abi_shape:?}"
                )));
            }
            let source_slot = if source_slot.requires_pointer_reload_in_block(block_idx) {
                let reloaded_ptr = self.load_reg_as_ptr(block_idx, reg)?;
                super::CompactStateSlot::pointer_backed_in_block(
                    reloaded_ptr,
                    source_slot.offset,
                    block_idx,
                )
            } else {
                source_slot
            };
            self.copy_compact_aggregate_to_compact_slots(
                block_idx,
                source_slot.source_ptr,
                source_slot.offset,
                &source_shape,
                expected_abi_shape,
                result_ptr,
                0,
            )?
        } else {
            if !Self::can_copy_flat_aggregate_to_compact_slots(&source_shape, expected_abi_shape) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Call compact aggregate argument {arg_idx} for callee {op_idx} requires compatible flat source and ABI shapes for r{reg}, got {source_shape:?} -> {expected_abi_shape:?}"
                )));
            }
            let source_ptr = self.load_reg_as_ptr(block_idx, reg)?;
            self.copy_flat_aggregate_to_compact_slots(
                block_idx,
                source_ptr,
                &source_shape,
                expected_abi_shape,
                result_ptr,
                0,
                false,
            )?
        };

        if copied.slots_written != expected_slots {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call compact aggregate argument {arg_idx} for callee {op_idx} copied {} slots for r{reg}, expected {expected_slots}",
                copied.slots_written
            )));
        }

        Ok((
            copied.block_idx,
            self.ptr_to_i64(copied.block_idx, result_ptr),
        ))
    }

    fn materialize_compact_sequence_call_arg(
        &mut self,
        block_idx: usize,
        reg: u8,
    ) -> Result<Option<(usize, ValueId)>, TrustIrError> {
        let Some(source_slot) = self.compact_state_slots.get(&reg).copied() else {
            return Ok(None);
        };
        let Some(source_shape) = self.aggregate_shapes.get(&reg).cloned() else {
            return Ok(None);
        };
        if !matches!(&source_shape, super::AggregateShape::Sequence { .. }) {
            return Ok(None);
        }

        let abi_shape = Self::compact_return_abi_shape(Some(source_shape.clone())).ok_or_else(
            || {
                TrustIrError::UnsupportedOpcode(format!(
                    "Call compact sequence argument r{reg} requires fixed-width ABI shape, got {source_shape:?}"
                ))
            },
        )?;
        let slot_count = abi_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact sequence argument r{reg} requires fixed-width ABI shape, got {abi_shape:?}"
            ))
        })?;

        let source_slot = if source_slot.requires_pointer_reload_in_block(block_idx) {
            let reloaded_ptr = self.load_reg_as_ptr(block_idx, reg)?;
            super::CompactStateSlot::pointer_backed_in_block(
                reloaded_ptr,
                source_slot.offset,
                block_idx,
            )
        } else {
            source_slot
        };

        let result_ptr = self.alloc_aggregate(block_idx, slot_count);
        let copied = self.copy_compact_aggregate_to_compact_slots(
            block_idx,
            source_slot.source_ptr,
            source_slot.offset,
            &source_shape,
            &abi_shape,
            result_ptr,
            0,
        )?;
        if copied.slots_written != slot_count {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call compact sequence argument r{reg} copied {} slots, expected {slot_count}",
                copied.slots_written
            )));
        }

        Ok(Some((
            copied.block_idx,
            self.ptr_to_i64(copied.block_idx, result_ptr),
        )))
    }

    fn materialize_compact_record_call_arg(
        &mut self,
        block_idx: usize,
        reg: u8,
    ) -> Result<Option<(usize, ValueId)>, TrustIrError> {
        let Some(source_slot) = self.compact_state_slots.get(&reg).copied() else {
            return Ok(None);
        };
        let Some(source_shape) = self.aggregate_shapes.get(&reg).cloned() else {
            return Ok(None);
        };
        if !matches!(&source_shape, super::AggregateShape::Record { .. }) {
            return Ok(None);
        }

        let abi_shape = Self::compact_return_abi_shape(Some(source_shape.clone())).ok_or_else(
            || {
                TrustIrError::UnsupportedOpcode(format!(
                    "Call compact record argument r{reg} requires fixed-width ABI shape, got {source_shape:?}"
                ))
            },
        )?;
        let slot_count = abi_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact record argument r{reg} requires fixed-width ABI shape, got {abi_shape:?}"
            ))
        })?;

        let source_slot = if source_slot.requires_pointer_reload_in_block(block_idx) {
            let reloaded_ptr = self.load_reg_as_ptr(block_idx, reg)?;
            super::CompactStateSlot::pointer_backed_in_block(
                reloaded_ptr,
                source_slot.offset,
                block_idx,
            )
        } else {
            source_slot
        };

        let result_ptr = self.alloc_aggregate(block_idx, slot_count);
        let copied = self.copy_compact_aggregate_to_compact_slots(
            block_idx,
            source_slot.source_ptr,
            source_slot.offset,
            &source_shape,
            &abi_shape,
            result_ptr,
            0,
        )?;
        if copied.slots_written != slot_count {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call compact record argument r{reg} copied {} slots, expected {slot_count}",
                copied.slots_written
            )));
        }

        Ok(Some((
            copied.block_idx,
            self.ptr_to_i64(copied.block_idx, result_ptr),
        )))
    }

    fn materialize_compact_function_call_arg(
        &mut self,
        block_idx: usize,
        reg: u8,
    ) -> Result<Option<ValueId>, TrustIrError> {
        if self.is_flat_funcdef_pair_list(reg) {
            return Ok(None);
        }
        let Some(source_slot) = self.compact_state_slots.get(&reg).copied() else {
            return Ok(None);
        };
        let Some(shape) = self.aggregate_shapes.get(&reg).cloned() else {
            return Ok(None);
        };
        let super::AggregateShape::Function {
            len,
            domain_lo,
            value,
            ..
        } = shape
        else {
            return Ok(None);
        };
        let explicit_domain = if domain_lo.is_none() {
            Some(
                self.compact_function_domains
                    .get(&reg)
                    .cloned()
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "Call compact function argument r{reg} requires explicit-domain metadata"
                        ))
                    })?,
            )
        } else {
            None
        };
        let value_shape = value.ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact function argument r{reg} requires tracked value shape"
            ))
        })?;
        if !value_shape.is_numeric_scalar_shape() {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call compact function argument r{reg} only supports Int values, got {value_shape:?}"
            )));
        }

        let value_stride = value_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact function argument r{reg} requires fixed-width value shape, got {value_shape:?}"
            ))
        })?;
        if value_stride != 1 {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call compact function argument r{reg} only supports single-slot values, got {value_shape:?}"
            )));
        }
        let source_slot = if source_slot.is_raw_compact_slot() {
            source_slot
        } else {
            let reloaded_ptr = self.load_reg_as_ptr(block_idx, reg)?;
            super::CompactStateSlot::pointer_backed_in_block(
                reloaded_ptr,
                source_slot.offset,
                block_idx,
            )
        };

        let pair_slots = len.checked_mul(2).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact function argument r{reg} slot count overflows: {len} * 2"
            ))
        })?;
        let total_slots = pair_slots.checked_add(1).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact function argument r{reg} slot count overflows: {pair_slots} + 1"
            ))
        })?;
        let result_ptr = self.alloc_aggregate(block_idx, total_slots);

        let pair_count = self.emit_i64_const(block_idx, i64::from(len));
        self.store_at_offset(block_idx, result_ptr, 0, pair_count);

        for idx in 0..len {
            let key_slot = idx
                .checked_mul(2)
                .and_then(|slot| slot.checked_add(1))
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "Call compact function key slot overflows u32".to_owned(),
                    )
                })?;
            let value_slot = key_slot.checked_add(1).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "Call compact function value slot overflows u32".to_owned(),
                )
            })?;
            let key_value = if let Some(domain_lo) = domain_lo {
                domain_lo.checked_add(i64::from(idx)).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Call compact function argument r{reg} key overflows: {domain_lo} + {idx}"
                    ))
                })?
            } else {
                Self::generic_function_arg_domain_key(
                    explicit_domain
                        .as_ref()
                        .expect("explicit domain was checked when domain_lo is None"),
                    idx,
                    reg,
                )?
            };
            let key = self.emit_i64_const(block_idx, key_value);
            self.store_at_offset(block_idx, result_ptr, key_slot, key);

            let source_offset = source_slot
                .offset
                .checked_add(idx.checked_mul(value_stride).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "Call compact function source slot overflows u32".to_owned(),
                    )
                })?)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "Call compact function source slot overflows u32".to_owned(),
                    )
                })?;
            let value = self.load_at_offset(block_idx, source_slot.source_ptr, source_offset);
            self.store_at_offset(block_idx, result_ptr, value_slot, value);
        }

        Ok(Some(self.ptr_to_i64(block_idx, result_ptr)))
    }

    fn generic_function_arg_domain_key(
        domain: &super::CompactFunctionDomain,
        idx: u32,
        reg: u8,
    ) -> Result<i64, TrustIrError> {
        match domain {
            super::CompactFunctionDomain::Raw(keys) => keys
                .get(usize::try_from(idx).expect("u32 index must fit in usize"))
                .copied()
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "Call compact function argument r{reg} explicit domain is missing key {idx}"
                    ))
                }),
            super::CompactFunctionDomain::Exact(keys) => {
                if !Self::call_exact_domain_all_one_shape(keys) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Call compact function argument r{reg} mixed explicit domain cannot be materialized as a generic raw-key function"
                    )));
                }
                keys.get(usize::try_from(idx).expect("u32 index must fit in usize"))
                    .map(|key| match key {
                        SetBitmaskElement::Int(value) => *value,
                        SetBitmaskElement::Bool(value) => i64::from(*value),
                        SetBitmaskElement::String(name) | SetBitmaskElement::ModelValue(name) => {
                            i64::from(name.0)
                        }
                    })
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "Call compact function argument r{reg} explicit domain is missing key {idx}"
                        ))
                    })
            }
        }
    }

    fn call_exact_domain_all_one_shape(keys: &[SetBitmaskElement]) -> bool {
        let Some(first) = keys.first().map(Self::call_setbitmask_element_scalar_tag) else {
            return true;
        };
        keys.iter()
            .all(|key| Self::call_setbitmask_element_scalar_tag(key) == first)
    }

    fn call_setbitmask_element_scalar_tag(key: &SetBitmaskElement) -> u8 {
        match key {
            SetBitmaskElement::Int(_) => 0,
            SetBitmaskElement::Bool(_) => 1,
            SetBitmaskElement::String(_) => 2,
            SetBitmaskElement::ModelValue(_) => 3,
        }
    }
}
