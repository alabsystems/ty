// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Native-eligibility / admission control for the trust-codegen dispatch path.
//!
//! These analyses decide whether an action's compiled bytecode is eligible for
//! the native path. The central concern is *layout sensitivity*: an action that
//! reads state through layout-dependent operations (function application,
//! sequence/set builtins on state-derived registers, etc.) must defer the
//! pre-layout native cache build until the concrete state layout is known.
//!
//! Pure analysis: no codegen, dispatch, or verdict effect — the result only
//! gates whether the native cache build is deferred.

use rustc_hash::FxHashSet;

pub(in crate::check) fn should_defer_pre_layout_trust_cg_cache_build(
    action_bytecode: Option<&tla_eval::bytecode_vm::CompiledBytecode>,
) -> bool {
    action_bytecode.is_some_and(action_bytecode_has_layout_sensitive_state_access)
}

fn action_bytecode_has_layout_sensitive_state_access(
    action_bytecode: &tla_eval::bytecode_vm::CompiledBytecode,
) -> bool {
    action_bytecode.op_indices.values().copied().any(|idx| {
        let mut visiting = FxHashSet::default();
        function_has_layout_sensitive_state_access(
            &action_bytecode.chunk,
            idx,
            RegisterMask::default(),
            &mut visiting,
        )
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct RegisterMask {
    lo: u128,
    hi: u128,
}

impl RegisterMask {
    fn insert(&mut self, reg: u8) {
        if reg < 128 {
            self.lo |= 1u128 << reg;
        } else {
            self.hi |= 1u128 << (reg - 128);
        }
    }

    fn contains(self, reg: u8) -> bool {
        if reg < 128 {
            self.lo & (1u128 << reg) != 0
        } else {
            self.hi & (1u128 << (reg - 128)) != 0
        }
    }

    fn is_empty(self) -> bool {
        self.lo == 0 && self.hi == 0
    }
}

fn function_has_layout_sensitive_state_access(
    chunk: &tla_tir::bytecode::BytecodeChunk,
    func_idx: u16,
    initial_state_regs: RegisterMask,
    visiting: &mut FxHashSet<(u16, RegisterMask)>,
) -> bool {
    if !visiting.insert((func_idx, initial_state_regs)) {
        return false;
    }

    let Some(func) = chunk.functions.get(usize::from(func_idx)) else {
        visiting.remove(&(func_idx, initial_state_regs));
        return false;
    };

    let mut state_regs = FxHashSet::default();
    for reg in 0..=u8::MAX {
        if initial_state_regs.contains(reg) {
            state_regs.insert(reg);
        }
    }

    for op in &func.instructions {
        use tla_tir::bytecode::Opcode;

        match *op {
            Opcode::LoadVar { rd, .. } | Opcode::LoadPrime { rd, .. } => {
                state_regs.insert(rd);
            }
            // VM-only fusions must never be treated as pre-layout native
            // candidates.
            Opcode::RoundStepEq { .. }
            | Opcode::SetEnumSubseteq { .. }
            | Opcode::Tuple2SelfEq { .. }
            | Opcode::Tuple2SelfSubseteq { .. } => {
                visiting.remove(&(func_idx, initial_state_regs));
                return true;
            }
            Opcode::Move { rd, rs } => {
                let state_derived = state_regs.contains(&rs);
                set_register_state_origin(&mut state_regs, rd, state_derived);
            }
            Opcode::CondMove { rd, rs, .. } => {
                let state_derived = state_regs.contains(&rd) || state_regs.contains(&rs);
                set_register_state_origin(&mut state_regs, rd, state_derived);
            }
            Opcode::FuncApply { rd, func, .. }
            | Opcode::ValueApply { rd, func, .. }
            | Opcode::Domain { rd, rs: func }
            | Opcode::RecordGet { rd, rs: func, .. }
            | Opcode::TupleGet { rd, rs: func, .. } => {
                if state_regs.contains(&func) {
                    visiting.remove(&(func_idx, initial_state_regs));
                    return true;
                }
                state_regs.remove(&rd);
            }
            Opcode::FuncExcept {
                rd,
                func,
                path,
                val,
                ..
            } => {
                if state_regs.contains(&func) {
                    visiting.remove(&(func_idx, initial_state_regs));
                    return true;
                }
                let state_derived = state_regs.contains(&path) || state_regs.contains(&val);
                set_register_state_origin(&mut state_regs, rd, state_derived);
            }
            Opcode::SetIn { rd, set, .. }
            | Opcode::Tuple2SetIn { rd, set, .. }
            | Opcode::Powerset { rd, rs: set }
            | Opcode::BigUnion { rd, rs: set } => {
                if state_regs.contains(&set) {
                    visiting.remove(&(func_idx, initial_state_regs));
                    return true;
                }
                state_regs.remove(&rd);
            }
            Opcode::SetUnion { rd, r1, r2 }
            | Opcode::SetIntersect { rd, r1, r2 }
            | Opcode::SetDiff { rd, r1, r2 }
            | Opcode::Subseteq { rd, r1, r2 }
            | Opcode::Concat { rd, r1, r2 } => {
                if state_regs.contains(&r1) || state_regs.contains(&r2) {
                    visiting.remove(&(func_idx, initial_state_regs));
                    return true;
                }
                state_regs.remove(&rd);
            }
            Opcode::KSubset { rd, base, k } => {
                if state_regs.contains(&base) || state_regs.contains(&k) {
                    visiting.remove(&(func_idx, initial_state_regs));
                    return true;
                }
                state_regs.remove(&rd);
            }
            Opcode::Call {
                rd,
                op_idx,
                args_start,
                argc,
            } => {
                let callee_state_regs =
                    state_derived_callee_arg_mask(&state_regs, args_start, argc);
                if function_has_layout_sensitive_state_access(
                    chunk,
                    op_idx,
                    callee_state_regs,
                    visiting,
                ) {
                    visiting.remove(&(func_idx, initial_state_regs));
                    return true;
                }
                set_register_state_origin(&mut state_regs, rd, !callee_state_regs.is_empty());
            }
            Opcode::CallBuiltin {
                rd,
                builtin,
                args_start,
                argc,
            } => {
                if builtin_is_layout_sensitive(builtin)
                    && register_range_has_state_origin(&state_regs, args_start, argc)
                {
                    visiting.remove(&(func_idx, initial_state_regs));
                    return true;
                }
                state_regs.remove(&rd);
            }
            Opcode::MakeClosure {
                rd,
                captures_start,
                capture_count,
                ..
            } => {
                let state_derived =
                    register_range_has_state_origin(&state_regs, captures_start, capture_count);
                set_register_state_origin(&mut state_regs, rd, state_derived);
            }
            Opcode::CallExternal {
                rd,
                args_start,
                argc,
                ..
            } => {
                let state_derived = register_range_has_state_origin(&state_regs, args_start, argc);
                set_register_state_origin(&mut state_regs, rd, state_derived);
            }
            _ => {
                if let Some(rd) = op.dest_register() {
                    state_regs.remove(&rd);
                }
            }
        }
    }

    visiting.remove(&(func_idx, initial_state_regs));
    false
}

fn set_register_state_origin(
    state_regs: &mut FxHashSet<tla_tir::bytecode::Register>,
    rd: tla_tir::bytecode::Register,
    state_derived: bool,
) {
    if state_derived {
        state_regs.insert(rd);
    } else {
        state_regs.remove(&rd);
    }
}

fn state_derived_callee_arg_mask(
    state_regs: &FxHashSet<tla_tir::bytecode::Register>,
    args_start: tla_tir::bytecode::Register,
    argc: u8,
) -> RegisterMask {
    let mut mask = RegisterMask::default();
    for offset in 0..argc {
        if state_regs.contains(&args_start.saturating_add(offset)) {
            mask.insert(offset);
        }
    }
    mask
}

fn register_range_has_state_origin(
    state_regs: &FxHashSet<tla_tir::bytecode::Register>,
    start: tla_tir::bytecode::Register,
    count: u8,
) -> bool {
    (0..count).any(|offset| state_regs.contains(&start.saturating_add(offset)))
}

fn builtin_is_layout_sensitive(builtin: tla_tir::bytecode::BuiltinOp) -> bool {
    use tla_tir::bytecode::BuiltinOp;

    matches!(
        builtin,
        BuiltinOp::Len
            | BuiltinOp::Head
            | BuiltinOp::Tail
            | BuiltinOp::Append
            | BuiltinOp::SubSeq
            | BuiltinOp::Seq
            | BuiltinOp::Cardinality
            | BuiltinOp::IsFiniteSet
            | BuiltinOp::FoldFunctionOnSetSum
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    #[test]
    fn round_step_eq_is_rejected_from_layout_sensitive_native_path() {
        let mut func = BytecodeFunction::new("RoundStep".to_string(), 2);
        func.emit(Opcode::RoundStepEq {
            rd: 2,
            child: 0,
            parent: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });
        let mut chunk = BytecodeChunk::new();
        let func_idx = chunk.add_function(func);

        assert!(function_has_layout_sensitive_state_access(
            &chunk,
            func_idx,
            RegisterMask::default(),
            &mut FxHashSet::default(),
        ));
    }

    #[test]
    fn set_enum_subseteq_is_rejected_from_layout_sensitive_native_path() {
        let mut func = BytecodeFunction::new("SetEnumSubseteq".to_string(), 3);
        func.emit(Opcode::SetEnumSubseteq {
            rd: 3,
            start: 0,
            count: 2,
            set: 2,
        });
        func.emit(Opcode::Ret { rs: 3 });
        let mut chunk = BytecodeChunk::new();
        let func_idx = chunk.add_function(func);

        assert!(function_has_layout_sensitive_state_access(
            &chunk,
            func_idx,
            RegisterMask::default(),
            &mut FxHashSet::default(),
        ));
    }

    #[test]
    fn tuple2_self_eq_is_rejected_from_layout_sensitive_native_path() {
        let mut func = BytecodeFunction::new("Tuple2SelfEq".to_string(), 1);
        func.emit(Opcode::Tuple2SelfEq { rd: 1, value: 0 });
        func.emit(Opcode::Ret { rs: 1 });
        let mut chunk = BytecodeChunk::new();
        let func_idx = chunk.add_function(func);

        assert!(function_has_layout_sensitive_state_access(
            &chunk,
            func_idx,
            RegisterMask::default(),
            &mut FxHashSet::default(),
        ));
    }

    #[test]
    fn tuple2_self_subseteq_is_rejected_from_layout_sensitive_native_path() {
        let mut func = BytecodeFunction::new("Tuple2SelfSubseteq".to_string(), 1);
        func.emit(Opcode::Tuple2SelfSubseteq {
            rd: 1,
            value: 0,
            set_var_idx: 3,
        });
        func.emit(Opcode::Ret { rs: 1 });
        let mut chunk = BytecodeChunk::new();
        let func_idx = chunk.add_function(func);

        assert!(function_has_layout_sensitive_state_access(
            &chunk,
            func_idx,
            RegisterMask::default(),
            &mut FxHashSet::default(),
        ));
    }
}
