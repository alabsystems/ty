// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fail-closed action-bytecode normalization for next-state compilation.
//!
//! Some action predicates compile to bytecode shapes the next-state transform
//! and the native backends cannot consume even though the underlying spec
//! semantics are simple:
//!
//! 1. A parameterized `LET`-operator lowered as a constant-pool closure applied
//!    via `ValueApply` (PaxosCommit's `Decide` / `Decided(rm, v)`).
//! 2. A successor-producing helper (`Send(m)` with `msgs' = msgs \cup {m}`)
//!    reached through a `Call`, so the `LoadPrime`/`Eq` pattern the action
//!    transform rewrites into `StoreVar` is hidden inside the callee.
//! 3. Guard quantifiers (`\A rm \in RM : ...`, `\E b \in Ballot, MS \in
//!    Majority : ...`) over compile-time-constant domains, which the native
//!    planner rejects (multiple `ExistsBegin` pairs / non-scalar witnesses).
//!
//! This module normalizes such functions with three semantics-preserving,
//! fail-closed rewrites, iterated to a bounded fixpoint:
//!
//! - [`rewrite_const_closure_applies`]: `ValueApply` whose callee register
//!   provably holds a constant-pool closure with a compiled bytecode
//!   sub-function becomes a direct `Call` (identical VM semantics: the VM
//!   executes exactly that sub-function for such closures).
//! - inline eligible `Call`s: single-`Ret` callees without nested calls or
//!   successor writes (`LoadPrime` IS allowed — exposing it to the action
//!   transform is the point) are spliced in with a uniform register shift.
//! - unroll BOOLEAN-POSITION quantifier loops (`ForallBegin`/`ExistsBegin`
//!   pairs whose bodies produce no successor effects) over provably-constant
//!   domains into short-circuit chains. This is a pure loop unroll of the same
//!   function — unlike per-witness expansion it cannot drop successors, and it
//!   is exact for guards regardless of surrounding disjunctions.
//!
//! Every rewrite either proves its preconditions structurally or leaves the
//! function untouched. Callers treat `None` as "keep the original bytecode".

use super::chunk::ConstantPool;
use tla_value::Rp;
use super::opcode::{Opcode, Register};
use super::BytecodeFunction;
use tla_value::Value;

/// Maximum callee size (in opcodes) eligible for inlining.
const MAX_INLINE_CALLEE_OPS: usize = 256;
/// Maximum unrolled domain size for a single boolean quantifier.
const MAX_UNROLL_DOMAIN: usize = 64;
/// Maximum function size (in opcodes) after any normalization step.
const MAX_NORMALIZED_OPS: usize = 4096;
/// Maximum normalization rounds. Each round runs the closure-apply rewrite
/// once and the inline/unroll passes to a (step-budgeted) fixpoint; a new
/// round is needed only when a pass unlocks work for an EARLIER pass (e.g.
/// an unroll exposing a newly-resolvable `ValueApply`).
const MAX_ROUNDS: usize = 16;
/// Maximum `Move`-chase depth when resolving constant registers.
const MAX_CHASE_DEPTH: usize = 8;

/// Normalize `func` for next-state compilation.
///
/// Returns `Some(normalized)` when at least one rewrite applied and all
/// bounds were respected; `None` when nothing applied (caller keeps the
/// original function). The constant pool may gain values (append-only), which
/// is safe: existing indices are never touched.
pub fn normalize_action_function(
    func: &BytecodeFunction,
    chunk_functions: &[BytecodeFunction],
    pool: &mut ConstantPool,
) -> Option<BytecodeFunction> {
    let mut current = func.clone();
    let mut changed = false;
    // Global step budget: every individual rewrite (one inline, one loop
    // unroll) consumes one step. Nested constant quantifiers multiply the
    // number of unrolls (each copy of an outer body carries its own inner
    // loops), so the budget is sized well above the per-function op cap.
    let mut steps_left: usize = 512;

    for _ in 0..MAX_ROUNDS {
        let mut round_changed = false;

        if rewrite_const_closure_applies(&mut current, chunk_functions, pool) {
            round_changed = true;
        }
        while steps_left > 0 && inline_one_call(&mut current, chunk_functions) {
            steps_left -= 1;
            round_changed = true;
            if current.instructions.len() > MAX_NORMALIZED_OPS {
                return None;
            }
        }
        while steps_left > 0 && unroll_one_bool_quantifier(&mut current, pool) {
            steps_left -= 1;
            round_changed = true;
            if current.instructions.len() > MAX_NORMALIZED_OPS {
                return None;
            }
        }

        if !round_changed || steps_left == 0 {
            changed = changed || round_changed;
            break;
        }
        changed = true;
        if current.instructions.len() > MAX_NORMALIZED_OPS {
            // Blowup guard: discard everything, keep the original.
            return None;
        }
    }

    if !changed {
        return None;
    }

    // Final cleanup: blank pure loads whose results are never read. This keeps
    // dead `LoadConst`s of natively-unloadable values (e.g. the now-unused
    // closure constant) from failing the whole function's native lowering.
    sweep_dead_pure_loads(&mut current);

    Some(current)
}

// =====================================================================
// Register helpers
// =====================================================================

/// Invoke `f` for every source (read) register of `op`.
fn for_each_source_register(op: &Opcode, mut f: impl FnMut(Register)) {
    match *op {
        Opcode::LoadImm { .. }
        | Opcode::LoadBool { .. }
        | Opcode::LoadConst { .. }
        | Opcode::LoadVar { .. }
        | Opcode::LoadPrime { .. }
        | Opcode::Jump { .. }
        | Opcode::SetPrimeMode { .. }
        | Opcode::Nop
        | Opcode::Halt
        | Opcode::Unchanged { .. } => {}

        Opcode::StoreVar { rs, .. }
        | Opcode::Move { rs, .. }
        | Opcode::NegInt { rs, .. }
        | Opcode::Not { rs, .. }
        | Opcode::Powerset { rs, .. }
        | Opcode::BigUnion { rs, .. }
        | Opcode::Domain { rs, .. }
        | Opcode::RecordGet { rs, .. }
        | Opcode::TupleGet { rs, .. }
        | Opcode::Tuple2SelfEq { value: rs, .. }
        | Opcode::Tuple2SelfSubseteq { value: rs, .. }
        | Opcode::Ret { rs }
        | Opcode::JumpTrue { rs, .. }
        | Opcode::JumpFalse { rs, .. } => f(rs),

        Opcode::RoundStepEq { child, parent, .. } => {
            f(child);
            f(parent);
        }

        Opcode::AddInt { r1, r2, .. }
        | Opcode::SubInt { r1, r2, .. }
        | Opcode::MulInt { r1, r2, .. }
        | Opcode::DivInt { r1, r2, .. }
        | Opcode::IntDiv { r1, r2, .. }
        | Opcode::ModInt { r1, r2, .. }
        | Opcode::PowInt { r1, r2, .. }
        | Opcode::Eq { r1, r2, .. }
        | Opcode::Neq { r1, r2, .. }
        | Opcode::LtInt { r1, r2, .. }
        | Opcode::LeInt { r1, r2, .. }
        | Opcode::GtInt { r1, r2, .. }
        | Opcode::GeInt { r1, r2, .. }
        | Opcode::And { r1, r2, .. }
        | Opcode::Or { r1, r2, .. }
        | Opcode::Implies { r1, r2, .. }
        | Opcode::Equiv { r1, r2, .. }
        | Opcode::SetUnion { r1, r2, .. }
        | Opcode::SetIntersect { r1, r2, .. }
        | Opcode::SetDiff { r1, r2, .. }
        | Opcode::Subseteq { r1, r2, .. }
        | Opcode::StrConcat { r1, r2, .. }
        | Opcode::Concat { r1, r2, .. } => {
            f(r1);
            f(r2);
        }

        Opcode::Range { lo, hi, .. } => {
            f(lo);
            f(hi);
        }
        Opcode::KSubset { base, k, .. } => {
            f(base);
            f(k);
        }
        Opcode::SetIn { elem, set, .. } => {
            f(elem);
            f(set);
        }
        Opcode::Tuple2SetIn {
            first, second, set, ..
        } => {
            f(first);
            f(second);
            f(set);
        }
        Opcode::SetEnumSubseteq {
            start, count, set, ..
        } => {
            for offset in 0..count {
                f(start.saturating_add(offset));
            }
            f(set);
        }
        Opcode::FuncApply { func, arg, .. } => {
            f(func);
            f(arg);
        }
        Opcode::FuncSet { domain, range, .. } => {
            f(domain);
            f(range);
        }
        Opcode::FuncExcept {
            func, path, val, ..
        } => {
            f(func);
            f(path);
            f(val);
        }
        Opcode::EqFuncExcept {
            lhs,
            func,
            path,
            val,
            ..
        } => {
            f(lhs);
            f(func);
            f(path);
            f(val);
        }
        Opcode::EqRecordNew {
            lhs,
            values_start,
            count,
            ..
        } => {
            f(lhs);
            for i in 0..count {
                f(values_start.saturating_add(i));
            }
        }
        Opcode::CondMove { cond, rs, .. } => {
            f(cond);
            f(rs);
        }

        Opcode::SetEnum { start, count, .. }
        | Opcode::TupleNew { start, count, .. }
        | Opcode::SeqNew { start, count, .. }
        | Opcode::Times { start, count, .. } => {
            for i in 0..count {
                f(start.saturating_add(i));
            }
        }
        Opcode::RecordNew {
            values_start,
            count,
            ..
        }
        | Opcode::RecordSet {
            values_start,
            count,
            ..
        } => {
            for i in 0..count {
                f(values_start.saturating_add(i));
            }
        }

        Opcode::FuncDef {
            r_domain,
            r_binding,
            ..
        } => {
            f(r_domain);
            f(r_binding);
        }

        Opcode::Call {
            args_start, argc, ..
        }
        | Opcode::CallExternal {
            args_start, argc, ..
        }
        | Opcode::CallBuiltin {
            args_start, argc, ..
        } => {
            for i in 0..argc {
                f(args_start.saturating_add(i));
            }
        }
        Opcode::ValueApply {
            func,
            args_start,
            argc,
            ..
        } => {
            f(func);
            for i in 0..argc {
                f(args_start.saturating_add(i));
            }
        }
        Opcode::MakeClosure {
            captures_start,
            capture_count,
            ..
        } => {
            for i in 0..capture_count {
                f(captures_start.saturating_add(i));
            }
        }

        Opcode::ForallBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::ExistsBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::ChooseBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::SetFilterBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::SetBuilderBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::FuncDefBegin {
            r_binding,
            r_domain,
            ..
        } => {
            f(r_binding);
            f(r_domain);
        }

        Opcode::ForallNext {
            r_binding, r_body, ..
        }
        | Opcode::ExistsNext {
            r_binding, r_body, ..
        }
        | Opcode::ChooseNext {
            r_binding, r_body, ..
        }
        | Opcode::LoopNext {
            r_binding, r_body, ..
        } => {
            f(r_binding);
            f(r_body);
        }
    }
}

/// PCs (over the whole function) whose instruction writes `reg` (destination
/// or loop-binding register).
fn writer_pcs(instrs: &[Opcode], reg: Register) -> Vec<usize> {
    instrs
        .iter()
        .enumerate()
        .filter(|(_, op)| op.dest_register() == Some(reg) || op.binding_register() == Some(reg))
        .map(|(pc, _)| pc)
        .collect()
}

/// The control-transfer offset of `op`, if it has one.
fn jump_offset(op: &Opcode) -> Option<i32> {
    match *op {
        Opcode::Jump { offset }
        | Opcode::JumpTrue { offset, .. }
        | Opcode::JumpFalse { offset, .. } => Some(offset),
        Opcode::ForallBegin { loop_end, .. }
        | Opcode::ExistsBegin { loop_end, .. }
        | Opcode::ChooseBegin { loop_end, .. }
        | Opcode::SetFilterBegin { loop_end, .. }
        | Opcode::SetBuilderBegin { loop_end, .. }
        | Opcode::FuncDefBegin { loop_end, .. } => Some(loop_end),
        Opcode::ForallNext { loop_begin, .. }
        | Opcode::ExistsNext { loop_begin, .. }
        | Opcode::ChooseNext { loop_begin, .. }
        | Opcode::LoopNext { loop_begin, .. } => Some(loop_begin),
        _ => None,
    }
}

/// Rebuild `op` with a new control-transfer offset. Must only be called for
/// opcodes where [`jump_offset`] returns `Some`.
fn with_jump_offset(op: Opcode, new_offset: i32) -> Opcode {
    match op {
        Opcode::Jump { .. } => Opcode::Jump { offset: new_offset },
        Opcode::JumpTrue { rs, .. } => Opcode::JumpTrue {
            rs,
            offset: new_offset,
        },
        Opcode::JumpFalse { rs, .. } => Opcode::JumpFalse {
            rs,
            offset: new_offset,
        },
        Opcode::ForallBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => Opcode::ForallBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: new_offset,
        },
        Opcode::ExistsBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => Opcode::ExistsBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: new_offset,
        },
        Opcode::ChooseBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => Opcode::ChooseBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: new_offset,
        },
        Opcode::SetFilterBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => Opcode::SetFilterBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: new_offset,
        },
        Opcode::SetBuilderBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => Opcode::SetBuilderBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: new_offset,
        },
        Opcode::FuncDefBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => Opcode::FuncDefBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: new_offset,
        },
        Opcode::ForallNext {
            rd,
            r_binding,
            r_body,
            ..
        } => Opcode::ForallNext {
            rd,
            r_binding,
            r_body,
            loop_begin: new_offset,
        },
        Opcode::ExistsNext {
            rd,
            r_binding,
            r_body,
            ..
        } => Opcode::ExistsNext {
            rd,
            r_binding,
            r_body,
            loop_begin: new_offset,
        },
        Opcode::ChooseNext {
            rd,
            r_binding,
            r_body,
            ..
        } => Opcode::ChooseNext {
            rd,
            r_binding,
            r_body,
            loop_begin: new_offset,
        },
        Opcode::LoopNext {
            r_binding, r_body, ..
        } => Opcode::LoopNext {
            r_binding,
            r_body,
            loop_begin: new_offset,
        },
        other => other,
    }
}

/// Rebuild `op` with every register field shifted up by `shift`.
///
/// Register *blocks* (`start`/`count`) remain contiguous under a uniform
/// shift, so this is exact. Returns `None` when any resulting register would
/// overflow `u8`, or for opcodes the inliner refuses to carry (calls,
/// closures, successor writes) — the caller fails closed.
fn shift_registers(op: &Opcode, shift: u16) -> Option<Opcode> {
    let s = |r: Register| -> Option<Register> {
        let v = u16::from(r) + shift;
        u8::try_from(v).ok()
    };
    Some(match *op {
        Opcode::Nop => Opcode::Nop,
        Opcode::LoadImm { rd, value } => Opcode::LoadImm { rd: s(rd)?, value },
        Opcode::LoadBool { rd, value } => Opcode::LoadBool { rd: s(rd)?, value },
        Opcode::LoadConst { rd, idx } => Opcode::LoadConst { rd: s(rd)?, idx },
        Opcode::LoadVar { rd, var_idx } => Opcode::LoadVar {
            rd: s(rd)?,
            var_idx,
        },
        Opcode::LoadPrime { rd, var_idx } => Opcode::LoadPrime {
            rd: s(rd)?,
            var_idx,
        },
        Opcode::Move { rd, rs } => Opcode::Move {
            rd: s(rd)?,
            rs: s(rs)?,
        },
        Opcode::NegInt { rd, rs } => Opcode::NegInt {
            rd: s(rd)?,
            rs: s(rs)?,
        },
        Opcode::Not { rd, rs } => Opcode::Not {
            rd: s(rd)?,
            rs: s(rs)?,
        },
        Opcode::Powerset { rd, rs } => Opcode::Powerset {
            rd: s(rd)?,
            rs: s(rs)?,
        },
        Opcode::BigUnion { rd, rs } => Opcode::BigUnion {
            rd: s(rd)?,
            rs: s(rs)?,
        },
        Opcode::Domain { rd, rs } => Opcode::Domain {
            rd: s(rd)?,
            rs: s(rs)?,
        },
        Opcode::AddInt { rd, r1, r2 } => Opcode::AddInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::SubInt { rd, r1, r2 } => Opcode::SubInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::MulInt { rd, r1, r2 } => Opcode::MulInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::DivInt { rd, r1, r2 } => Opcode::DivInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::IntDiv { rd, r1, r2 } => Opcode::IntDiv {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::ModInt { rd, r1, r2 } => Opcode::ModInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::PowInt { rd, r1, r2 } => Opcode::PowInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::Eq { rd, r1, r2 } => Opcode::Eq {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::Tuple2SelfEq { rd, value } => Opcode::Tuple2SelfEq {
            rd: s(rd)?,
            value: s(value)?,
        },
        Opcode::Tuple2SelfSubseteq {
            rd,
            value,
            set_var_idx,
        } => Opcode::Tuple2SelfSubseteq {
            rd: s(rd)?,
            value: s(value)?,
            set_var_idx,
        },
        Opcode::Neq { rd, r1, r2 } => Opcode::Neq {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::LtInt { rd, r1, r2 } => Opcode::LtInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::LeInt { rd, r1, r2 } => Opcode::LeInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::GtInt { rd, r1, r2 } => Opcode::GtInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::GeInt { rd, r1, r2 } => Opcode::GeInt {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::And { rd, r1, r2 } => Opcode::And {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::Or { rd, r1, r2 } => Opcode::Or {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::Implies { rd, r1, r2 } => Opcode::Implies {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::Equiv { rd, r1, r2 } => Opcode::Equiv {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::SetUnion { rd, r1, r2 } => Opcode::SetUnion {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::SetIntersect { rd, r1, r2 } => Opcode::SetIntersect {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::SetDiff { rd, r1, r2 } => Opcode::SetDiff {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::Subseteq { rd, r1, r2 } => Opcode::Subseteq {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::StrConcat { rd, r1, r2 } => Opcode::StrConcat {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::Concat { rd, r1, r2 } => Opcode::Concat {
            rd: s(rd)?,
            r1: s(r1)?,
            r2: s(r2)?,
        },
        Opcode::Jump { offset } => Opcode::Jump { offset },
        Opcode::JumpTrue { rs, offset } => Opcode::JumpTrue { rs: s(rs)?, offset },
        Opcode::JumpFalse { rs, offset } => Opcode::JumpFalse { rs: s(rs)?, offset },
        Opcode::Ret { rs } => Opcode::Ret { rs: s(rs)? },
        Opcode::SetEnum { rd, start, count } => Opcode::SetEnum {
            rd: s(rd)?,
            start: s(start)?,
            count,
        },
        Opcode::SetIn { rd, elem, set } => Opcode::SetIn {
            rd: s(rd)?,
            elem: s(elem)?,
            set: s(set)?,
        },
        Opcode::Tuple2SetIn {
            rd,
            first,
            second,
            set,
        } => Opcode::Tuple2SetIn {
            rd: s(rd)?,
            first: s(first)?,
            second: s(second)?,
            set: s(set)?,
        },
        Opcode::SetEnumSubseteq {
            rd,
            start,
            count,
            set,
        } => Opcode::SetEnumSubseteq {
            rd: s(rd)?,
            start: s(start)?,
            count,
            set: s(set)?,
        },
        Opcode::Range { rd, lo, hi } => Opcode::Range {
            rd: s(rd)?,
            lo: s(lo)?,
            hi: s(hi)?,
        },
        Opcode::KSubset { rd, base, k } => Opcode::KSubset {
            rd: s(rd)?,
            base: s(base)?,
            k: s(k)?,
        },
        Opcode::ForallBegin {
            rd,
            r_binding,
            r_domain,
            loop_end,
        } => Opcode::ForallBegin {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_domain: s(r_domain)?,
            loop_end,
        },
        Opcode::ForallNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => Opcode::ForallNext {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_body: s(r_body)?,
            loop_begin,
        },
        Opcode::ExistsBegin {
            rd,
            r_binding,
            r_domain,
            loop_end,
        } => Opcode::ExistsBegin {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_domain: s(r_domain)?,
            loop_end,
        },
        Opcode::ExistsNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => Opcode::ExistsNext {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_body: s(r_body)?,
            loop_begin,
        },
        Opcode::ChooseBegin {
            rd,
            r_binding,
            r_domain,
            loop_end,
        } => Opcode::ChooseBegin {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_domain: s(r_domain)?,
            loop_end,
        },
        Opcode::ChooseNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => Opcode::ChooseNext {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_body: s(r_body)?,
            loop_begin,
        },
        Opcode::SetBuilderBegin {
            rd,
            r_binding,
            r_domain,
            loop_end,
        } => Opcode::SetBuilderBegin {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_domain: s(r_domain)?,
            loop_end,
        },
        Opcode::SetFilterBegin {
            rd,
            r_binding,
            r_domain,
            loop_end,
        } => Opcode::SetFilterBegin {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_domain: s(r_domain)?,
            loop_end,
        },
        Opcode::FuncDefBegin {
            rd,
            r_binding,
            r_domain,
            loop_end,
        } => Opcode::FuncDefBegin {
            rd: s(rd)?,
            r_binding: s(r_binding)?,
            r_domain: s(r_domain)?,
            loop_end,
        },
        Opcode::LoopNext {
            r_binding,
            r_body,
            loop_begin,
        } => Opcode::LoopNext {
            r_binding: s(r_binding)?,
            r_body: s(r_body)?,
            loop_begin,
        },
        Opcode::RecordNew {
            rd,
            fields_start,
            values_start,
            count,
        } => Opcode::RecordNew {
            rd: s(rd)?,
            fields_start,
            values_start: s(values_start)?,
            count,
        },
        Opcode::RecordGet { rd, rs, field_idx } => Opcode::RecordGet {
            rd: s(rd)?,
            rs: s(rs)?,
            field_idx,
        },
        Opcode::RecordSet {
            rd,
            fields_start,
            values_start,
            count,
        } => Opcode::RecordSet {
            rd: s(rd)?,
            fields_start,
            values_start: s(values_start)?,
            count,
        },
        Opcode::FuncApply { rd, func, arg } => Opcode::FuncApply {
            rd: s(rd)?,
            func: s(func)?,
            arg: s(arg)?,
        },
        Opcode::FuncExcept {
            rd,
            func,
            path,
            val,
        } => Opcode::FuncExcept {
            rd: s(rd)?,
            func: s(func)?,
            path: s(path)?,
            val: s(val)?,
        },
        Opcode::EqFuncExcept {
            rd,
            lhs,
            func,
            path,
            val,
        } => Opcode::EqFuncExcept {
            rd: s(rd)?,
            lhs: s(lhs)?,
            func: s(func)?,
            path: s(path)?,
            val: s(val)?,
        },
        Opcode::EqRecordNew {
            rd,
            lhs,
            fields_start,
            values_start,
            count,
        } => Opcode::EqRecordNew {
            rd: s(rd)?,
            lhs: s(lhs)?,
            fields_start,
            values_start: s(values_start)?,
            count,
        },
        Opcode::TupleNew { rd, start, count } => Opcode::TupleNew {
            rd: s(rd)?,
            start: s(start)?,
            count,
        },
        Opcode::TupleGet { rd, rs, idx } => Opcode::TupleGet {
            rd: s(rd)?,
            rs: s(rs)?,
            idx,
        },
        Opcode::FuncDef {
            rd,
            r_domain,
            r_binding,
        } => Opcode::FuncDef {
            rd: s(rd)?,
            r_domain: s(r_domain)?,
            r_binding: s(r_binding)?,
        },
        Opcode::FuncSet { rd, domain, range } => Opcode::FuncSet {
            rd: s(rd)?,
            domain: s(domain)?,
            range: s(range)?,
        },
        Opcode::Times { rd, start, count } => Opcode::Times {
            rd: s(rd)?,
            start: s(start)?,
            count,
        },
        Opcode::SeqNew { rd, start, count } => Opcode::SeqNew {
            rd: s(rd)?,
            start: s(start)?,
            count,
        },
        Opcode::CondMove { rd, cond, rs } => Opcode::CondMove {
            rd: s(rd)?,
            cond: s(cond)?,
            rs: s(rs)?,
        },
        // Refused by callee eligibility: fail closed if ever reached.
        Opcode::StoreVar { .. }
        | Opcode::Unchanged { .. }
        | Opcode::SetPrimeMode { .. }
        | Opcode::RoundStepEq { .. }
        | Opcode::Halt
        | Opcode::Call { .. }
        | Opcode::ValueApply { .. }
        | Opcode::CallExternal { .. }
        | Opcode::CallBuiltin { .. }
        | Opcode::MakeClosure { .. } => return None,
    })
}

// =====================================================================
// Splice with offset fixup
// =====================================================================

/// Replace `instrs[start..end]` with `replacement`, remapping every
/// control-transfer offset of instructions OUTSIDE the region.
///
/// Fail-closed rules:
/// - An outside jump may target `start` (maps to the replacement's first
///   instruction — the replacement is the semantic equivalent of the removed
///   region entered at its top), `end` (maps just past the replacement), or
///   any pc outside `(start, end)`. A target strictly inside the region
///   returns `None`.
/// - Replacement instructions are emitted with final-position-relative
///   offsets by their constructors and are copied verbatim.
fn splice_with_offset_fixup(
    instrs: &[Opcode],
    start: usize,
    end: usize,
    replacement: &[Opcode],
) -> Option<Vec<Opcode>> {
    debug_assert!(start <= end && end <= instrs.len());
    let old_len = instrs.len() as i64;
    let delta = replacement.len() as i64 - (end - start) as i64;
    let map_pc = |pc: i64| -> Option<i64> {
        if pc <= start as i64 {
            Some(pc)
        } else if pc >= end as i64 {
            Some(pc + delta)
        } else {
            None
        }
    };

    let mut out = Vec::with_capacity((old_len + delta) as usize);
    for (pc, op) in instrs.iter().enumerate() {
        if pc >= start && pc < end {
            if pc == start {
                out.extend_from_slice(replacement);
            }
            continue;
        }
        if pc == start {
            // start == end (pure insertion before `start`).
            out.extend_from_slice(replacement);
        }
        let fixed = if let Some(off) = jump_offset(op) {
            let old_target = pc as i64 + i64::from(off);
            if old_target < 0 || old_target > old_len {
                return None;
            }
            let new_target = map_pc(old_target)?;
            let new_pc = map_pc(pc as i64)?;
            let new_off = i32::try_from(new_target - new_pc).ok()?;
            with_jump_offset(*op, new_off)
        } else {
            *op
        };
        out.push(fixed);
    }
    if start == instrs.len() {
        out.extend_from_slice(replacement);
    }
    Some(out)
}

// =====================================================================
// Constant register resolution
// =====================================================================

/// Control-flow successors of the instruction at `pc` (fallthrough and/or
/// jump target), for the reaching-definition walk. `Ret`/`Halt` terminate.
/// Conditional and loop opcodes conservatively yield BOTH edges.
fn control_successors(instrs: &[Opcode], pc: usize, out: &mut Vec<usize>) {
    let len = instrs.len();
    let op = &instrs[pc];
    match op {
        Opcode::Ret { .. } | Opcode::Halt => {}
        Opcode::Jump { offset } => {
            let t = pc as i64 + i64::from(*offset);
            if t >= 0 && (t as usize) < len {
                out.push(t as usize);
            }
        }
        _ => {
            if let Some(off) = jump_offset(op) {
                let t = pc as i64 + i64::from(off);
                if t >= 0 && (t as usize) < len {
                    out.push(t as usize);
                }
            }
            if pc + 1 < len {
                out.push(pc + 1);
            }
        }
    }
}

/// True when the ONLY definition of `reg` that can reach `at_pc` is the write
/// at `producer`.
///
/// Walks the CFG forward from the function entry (representing the
/// "uninitialized" definition) and from every writer of `reg` other than
/// `producer`. Each walk stops at any pc that (re)writes `reg` — the
/// definition changes there and is covered by that writer's own walk. If any
/// non-`producer` walk reaches `at_pc`, a different definition may be live
/// there and resolution fails closed.
fn only_producer_definition_reaches(
    instrs: &[Opcode],
    reg: Register,
    producer: usize,
    at_pc: usize,
) -> bool {
    let writers = writer_pcs(instrs, reg);
    let is_writer = |pc: usize| writers.contains(&pc);

    // Start set: entry (if it is not immediately a writer) plus the
    // successors of every non-producer writer.
    let mut pending: Vec<usize> = Vec::new();
    if !instrs.is_empty() && !is_writer(0) {
        pending.push(0);
    }
    for &w in &writers {
        if w != producer {
            control_successors(instrs, w, &mut pending);
        }
    }

    let mut visited = vec![false; instrs.len()];
    while let Some(pc) = pending.pop() {
        if pc >= instrs.len() || visited[pc] {
            continue;
        }
        visited[pc] = true;
        if pc == at_pc {
            return false;
        }
        if is_writer(pc) {
            // Definition redefined here; this walk ends (the writer's own
            // start state covers the continuation).
            continue;
        }
        control_successors(instrs, pc, &mut pending);
    }
    true
}

/// Resolve the constant [`Value`] held by `reg` when execution reaches
/// `at_pc`, chasing `Move` aliases through `LoadConst` producers.
///
/// Soundness: the linear "last write before `at_pc`" is accepted only when a
/// reaching-definition walk proves no other definition of `reg` (another
/// writer, or the uninitialized entry state) can reach `at_pc`
/// ([`only_producer_definition_reaches`]). This correctly accepts loop-body
/// reads whose producer sits inside the same loop iteration and per-copy
/// binding loads emitted by the quantifier unroll, while rejecting genuinely
/// ambiguous flows.
fn resolve_const_value_at(
    instrs: &[Opcode],
    pool: &ConstantPool,
    reg: Register,
    at_pc: usize,
    depth: usize,
) -> Option<Value> {
    if depth > MAX_CHASE_DEPTH {
        return None;
    }
    let writers = writer_pcs(instrs, reg);
    let producer = writers.iter().copied().filter(|&pc| pc < at_pc).max()?;

    if !only_producer_definition_reaches(instrs, reg, producer, at_pc) {
        return None;
    }

    match instrs[producer] {
        Opcode::LoadConst { idx, .. } => {
            if usize::from(idx) >= pool.value_count() {
                return None;
            }
            Some(pool.get_value(idx).clone())
        }
        Opcode::LoadImm { value, .. } => Some(Value::SmallInt(value)),
        Opcode::LoadBool { value, .. } => Some(Value::Bool(value)),
        Opcode::Move { rs, .. } => resolve_const_value_at(instrs, pool, rs, producer, depth + 1),
        _ => None,
    }
}

// =====================================================================
// Pass: ValueApply on constant closure -> Call
// =====================================================================

/// Rewrite every `ValueApply` whose callee register provably holds a
/// constant-pool closure carrying a compiled bytecode sub-function into a
/// direct `Call` of that sub-function. 1:1 opcode replacement (no splice).
///
/// The VM executes exactly `functions[bytecode_func_idx]` for such closures,
/// so the rewrite is semantics-preserving. Arity must match exactly — a
/// capturing closure's sub-function has extra capture parameters and is
/// rejected by the arity check.
fn rewrite_const_closure_applies(
    func: &mut BytecodeFunction,
    chunk_functions: &[BytecodeFunction],
    pool: &ConstantPool,
) -> bool {
    let mut changed = false;
    for pc in 0..func.instructions.len() {
        let Opcode::ValueApply {
            rd,
            func: func_reg,
            args_start,
            argc,
        } = func.instructions[pc]
        else {
            continue;
        };
        let debug = std::env::var_os("TY_NORMALIZE_DEBUG").is_some();
        let Some(resolved) = resolve_const_value_at(&func.instructions, pool, func_reg, pc, 0)
        else {
            if debug {
                eprintln!(
                    "[normalize] {}: ValueApply pc {pc}: callee register r{func_reg} did not resolve to a constant",
                    func.name
                );
            }
            continue;
        };
        let Value::Closure(ref closure) = resolved else {
            if debug {
                eprintln!(
                    "[normalize] {}: ValueApply pc {pc}: resolved constant is not a closure",
                    func.name
                );
            }
            continue;
        };
        let Some(op_idx) = closure.bytecode_func_idx() else {
            if debug {
                eprintln!(
                    "[normalize] {}: ValueApply pc {pc}: closure has no compiled bytecode sub-function",
                    func.name
                );
            }
            continue;
        };
        let Some(callee) = chunk_functions.get(usize::from(op_idx)) else {
            if debug {
                eprintln!(
                    "[normalize] {}: ValueApply pc {pc}: closure func idx {op_idx} out of range ({} functions)",
                    func.name,
                    chunk_functions.len()
                );
            }
            continue;
        };
        if callee.arity != argc {
            if debug {
                eprintln!(
                    "[normalize] {}: ValueApply pc {pc}: closure func '{}' arity {} != argc {argc}",
                    func.name, callee.name, callee.arity
                );
            }
            continue;
        }
        func.instructions[pc] = Opcode::Call {
            rd,
            op_idx,
            args_start,
            argc,
        };
        changed = true;
    }
    changed
}

// =====================================================================
// Pass: inline eligible calls
// =====================================================================

/// A callee is inline-eligible when its full body can be spliced into the
/// caller with a uniform register shift and no semantic change:
/// - exactly one `Ret`, as the final instruction (no `Halt`);
/// - no nested `Call`/`ValueApply`/`CallExternal`/`CallBuiltin`/`MakeClosure`;
/// - no successor writes (`StoreVar`/`Unchanged`/`SetPrimeMode`) — but
///   `LoadPrime` IS allowed (exposing the prime-equality pattern to the
///   action transform is the point of inlining);
/// - every control transfer stays within the body (targets in `[0, ret_pc]`;
///   a jump to the `Ret` lands on the substituted result-`Move`).
fn callee_inline_eligible(callee: &BytecodeFunction) -> Option<Register> {
    let n = callee.instructions.len();
    if n < 2 || n > MAX_INLINE_CALLEE_OPS {
        return None;
    }
    let ret_rs = match callee.instructions[n - 1] {
        Opcode::Ret { rs } => rs,
        _ => return None,
    };
    for (pc, op) in callee.instructions.iter().enumerate() {
        match op {
            Opcode::Ret { .. } if pc != n - 1 => return None,
            Opcode::Halt
            | Opcode::Call { .. }
            | Opcode::ValueApply { .. }
            | Opcode::CallExternal { .. }
            | Opcode::CallBuiltin { .. }
            | Opcode::MakeClosure { .. }
            | Opcode::StoreVar { .. }
            | Opcode::Unchanged { .. }
            | Opcode::SetPrimeMode { .. }
            | Opcode::RoundStepEq { .. } => return None,
            _ => {}
        }
        if let Some(off) = jump_offset(op) {
            let target = pc as i64 + i64::from(off);
            if target < 0 || target > (n - 1) as i64 {
                return None;
            }
        }
    }
    Some(ret_rs)
}

/// Inline the first eligible `Call` in `func`. Returns `true` when a call was
/// inlined (callers loop to a fixpoint).
fn inline_one_call(func: &mut BytecodeFunction, chunk_functions: &[BytecodeFunction]) -> bool {
    for pc in 0..func.instructions.len() {
        let Opcode::Call {
            rd,
            op_idx,
            args_start,
            argc,
        } = func.instructions[pc]
        else {
            continue;
        };
        let Some(callee) = chunk_functions.get(usize::from(op_idx)) else {
            continue;
        };
        if callee.arity != argc {
            continue;
        }
        let Some(ret_rs) = callee_inline_eligible(callee) else {
            continue;
        };

        // Uniform register shift: callee register r lives at `base + r`.
        let base = u16::from(func.max_register) + 1;
        if base + u16::from(callee.max_register) > u16::from(u8::MAX) {
            continue;
        }

        // Prologue: copy the argument block into the callee's parameter
        // registers (never write into the caller's argument registers).
        let mut replacement: Vec<Opcode> =
            Vec::with_capacity(usize::from(argc) + callee.instructions.len());
        let mut ok = true;
        for i in 0..argc {
            let Ok(dst) = u8::try_from(base + u16::from(i)) else {
                ok = false;
                break;
            };
            replacement.push(Opcode::Move {
                rd: dst,
                rs: args_start.saturating_add(i),
            });
        }
        if !ok {
            continue;
        }
        // Body with shifted registers; the trailing `Ret` becomes the result
        // `Move` at the same relative position, so intra-body jumps to the
        // `Ret` pc remain correct.
        let n = callee.instructions.len();
        for op in &callee.instructions[..n - 1] {
            match shift_registers(op, base) {
                Some(shifted) => replacement.push(shifted),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let Ok(result_src) = u8::try_from(base + u16::from(ret_rs)) else {
            continue;
        };
        replacement.push(Opcode::Move { rd, rs: result_src });

        let Some(new_instrs) =
            splice_with_offset_fixup(&func.instructions, pc, pc + 1, &replacement)
        else {
            continue;
        };
        if new_instrs.len() > MAX_NORMALIZED_OPS {
            continue;
        }

        let new_max = u8::try_from(base + u16::from(callee.max_register)).unwrap_or(u8::MAX);
        rebuild_function(func, new_instrs);
        if new_max > func.max_register {
            func.max_register = new_max;
        }
        return true;
    }
    false
}

/// Rebuild `func`'s instruction stream via `emit` so `max_register` is
/// re-derived from the actual opcodes (then callers may bump it further).
fn rebuild_function(func: &mut BytecodeFunction, instrs: Vec<Opcode>) {
    let mut rebuilt = BytecodeFunction::new(func.name.clone(), func.arity);
    rebuilt.max_register = func.max_register;
    for op in instrs {
        rebuilt.emit(op);
    }
    *func = rebuilt;
}

// =====================================================================
// Pass: unroll boolean-position constant-domain quantifiers
// =====================================================================

/// A matched quantifier loop.
struct QuantifierPair {
    begin_pc: usize,
    next_pc: usize,
    is_forall: bool,
    rd: Register,
    r_binding: Register,
    r_body: Register,
}

/// Find the first `ForallBegin`/`ExistsBegin` at or after `from_pc` whose
/// matching `*Next` is well-formed: the `Next`'s back-edge targets
/// `begin_pc + 1` and the `Begin`'s empty-domain edge targets `next_pc + 1`.
fn find_quantifier_pair(instrs: &[Opcode], from_pc: usize) -> Option<QuantifierPair> {
    for pc in from_pc..instrs.len() {
        let (is_forall, rd, r_binding, loop_end) = match instrs[pc] {
            Opcode::ForallBegin {
                rd,
                r_binding,
                loop_end,
                ..
            } => (true, rd, r_binding, loop_end),
            Opcode::ExistsBegin {
                rd,
                r_binding,
                loop_end,
                ..
            } => (false, rd, r_binding, loop_end),
            _ => continue,
        };
        let empty_target = pc as i64 + i64::from(loop_end);
        if empty_target <= pc as i64 || empty_target > instrs.len() as i64 {
            continue;
        }
        // Scan forward for the matching Next (back-edge to begin+1).
        let scan_hi = usize::try_from(empty_target).ok()?.min(instrs.len());
        let mut found = None;
        for next_pc in (pc + 1)..scan_hi {
            let (next_is_forall, n_rd, n_binding, n_body, loop_begin) = match instrs[next_pc] {
                Opcode::ForallNext {
                    rd,
                    r_binding,
                    r_body,
                    loop_begin,
                } => (true, rd, r_binding, r_body, loop_begin),
                Opcode::ExistsNext {
                    rd,
                    r_binding,
                    r_body,
                    loop_begin,
                } => (false, rd, r_binding, r_body, loop_begin),
                _ => continue,
            };
            if next_pc as i64 + i64::from(loop_begin) != (pc + 1) as i64 {
                continue;
            }
            if next_is_forall != is_forall || n_rd != rd || n_binding != r_binding {
                // Mismatched pair: fail closed for this Begin.
                found = None;
                break;
            }
            // The Begin's empty-domain edge must land exactly past the Next.
            if empty_target != (next_pc + 1) as i64 {
                found = None;
                break;
            }
            found = Some(QuantifierPair {
                begin_pc: pc,
                next_pc,
                is_forall,
                rd,
                r_binding,
                r_body: n_body,
            });
            break;
        }
        if let Some(pair) = found {
            return Some(pair);
        }
    }
    None
}

/// True when the loop body `[begin+1, next_pc)` is a pure boolean guard: no
/// successor effects, no calls, and every control transfer stays within
/// `[begin+1, next_pc]` (a jump to `next_pc` lands on the per-copy result
/// `Move`, which occupies the same relative slot as the original `*Next`).
fn body_is_pure_guard(instrs: &[Opcode], begin_pc: usize, next_pc: usize) -> bool {
    if next_pc <= begin_pc + 1 {
        return false;
    }
    for pc in (begin_pc + 1)..next_pc {
        match instrs[pc] {
            Opcode::StoreVar { .. }
            | Opcode::LoadPrime { .. }
            | Opcode::SetPrimeMode { .. }
            | Opcode::Unchanged { .. }
            | Opcode::Call { .. }
            | Opcode::ValueApply { .. }
            | Opcode::CallExternal { .. }
            | Opcode::CallBuiltin { .. }
            | Opcode::MakeClosure { .. }
            | Opcode::Ret { .. }
            | Opcode::Halt => return false,
            _ => {}
        }
        if let Some(off) = jump_offset(&instrs[pc]) {
            let target = pc as i64 + i64::from(off);
            if target < (begin_pc + 1) as i64 || target > next_pc as i64 {
                return false;
            }
        }
    }
    true
}

/// Emit the binding load for one unrolled element.
///
/// Scalar ints/bools use immediate loads; every other value (strings, model
/// values, sets, ...) is appended to the constant pool and loaded from there,
/// so the bound register holds EXACTLY the same [`Value`] the VM's iterator
/// would have produced.
fn bind_instr(r_binding: Register, elem: &Value, pool: &mut ConstantPool) -> Option<Opcode> {
    match elem {
        Value::SmallInt(v) => Some(Opcode::LoadImm {
            rd: r_binding,
            value: *v,
        }),
        Value::Bool(b) => Some(Opcode::LoadBool {
            rd: r_binding,
            value: *b,
        }),
        other => {
            if pool.value_count() >= usize::from(u16::MAX) {
                return None;
            }
            let idx = pool.add_value(other.clone());
            Some(Opcode::LoadConst { rd: r_binding, idx })
        }
    }
}

/// Unroll the first eligible boolean-position constant-domain quantifier.
/// Returns `true` when one loop was unrolled (callers loop to a fixpoint).
///
/// Replacement layout for `\E x \in {e_0..e_{n-1}} : body` (forall dual):
///
/// ```text
///   LoadBool rd <- false                 (VM's Begin seed; empty-domain result)
///   e_0:  <bind r_binding e_0> <body> Move rd <- r_body ; JumpTrue rd -> END
///   ...
///   e_n-1:<bind r_binding e_n-1> <body> Move rd <- r_body
///   END (== one past the replacement)
/// ```
///
/// Exactness vs the VM loop: the body always executes with `rd` equal to the
/// seed (a short-circuit would have jumped), `r_binding` is bound per element
/// in the VM's own `iter_set()` order, `rd` ends as "seed until proven
/// otherwise" exactly like `*Next`, and after the loop `r_binding` holds the
/// element the VM would have left there (short-circuit witness or last).
fn unroll_one_bool_quantifier(func: &mut BytecodeFunction, pool: &mut ConstantPool) -> bool {
    let mut from_pc = 0;
    while let Some(pair) = find_quantifier_pair(&func.instructions, from_pc) {
        from_pc = pair.begin_pc + 1;

        if !body_is_pure_guard(&func.instructions, pair.begin_pc, pair.next_pc) {
            continue;
        }
        // Resolve the constant domain.
        let r_domain = match func.instructions[pair.begin_pc] {
            Opcode::ForallBegin { r_domain, .. } | Opcode::ExistsBegin { r_domain, .. } => r_domain,
            _ => continue,
        };
        let Some(domain_value) =
            resolve_const_value_at(&func.instructions, pool, r_domain, pair.begin_pc, 0)
        else {
            continue;
        };
        let Some(iter) = domain_value.iter_set() else {
            continue;
        };
        let elems: Vec<Value> = iter.collect();
        if elems.len() > MAX_UNROLL_DOMAIN {
            continue;
        }

        let body = func.instructions[pair.begin_pc + 1..pair.next_pc].to_vec();
        let body_len = body.len();
        let n = elems.len();
        // Layout: 1 seed + per element (1 bind + body + 1 move + 1 jump). A jump
        // is emitted for EVERY element including the last: the jump's condition
        // read coerces the body result through as_bool, matching the VM loop
        // (forall_next/exists_next) which coerces every iteration incl. the last
        // and would raise a TypeError on a non-Bool final-element body. Without
        // the final jump the unrolled form silently drops that error. (audit3 N16)
        let total = 1 + n * (body_len + 3);
        if func.instructions.len() - (pair.next_pc + 1 - pair.begin_pc) + total > MAX_NORMALIZED_OPS
        {
            continue;
        }

        let seed = pair.is_forall;
        let mut replacement: Vec<Opcode> = Vec::with_capacity(total);
        replacement.push(Opcode::LoadBool {
            rd: pair.rd,
            value: seed,
        });
        let mut ok = true;
        for elem in &elems {
            let Some(bind) = bind_instr(pair.r_binding, elem, pool) else {
                ok = false;
                break;
            };
            replacement.push(bind);
            replacement.extend_from_slice(&body);
            replacement.push(Opcode::Move {
                rd: pair.rd,
                rs: pair.r_body,
            });
            // Emit the short-circuit jump after EVERY element (incl. the last).
            // For the last element offset computes to 1 (a fallthrough-equivalent
            // jump) whose sole effect is the VM's as_bool(rd) coercion — so a
            // non-Bool final-element body raises the same TypeError the VM loop
            // would. (audit3 N16)
            let jump_pc = replacement.len();
            let Ok(offset) = i32::try_from(total - jump_pc) else {
                ok = false;
                break;
            };
            // Short-circuit: exists on TRUE, forall on FALSE.
            replacement.push(if pair.is_forall {
                Opcode::JumpFalse {
                    rs: pair.rd,
                    offset,
                }
            } else {
                Opcode::JumpTrue {
                    rs: pair.rd,
                    offset,
                }
            });
        }
        if !ok {
            continue;
        }
        debug_assert_eq!(replacement.len(), total);

        let Some(new_instrs) = splice_with_offset_fixup(
            &func.instructions,
            pair.begin_pc,
            pair.next_pc + 1,
            &replacement,
        ) else {
            continue;
        };
        let saved_max = func.max_register;
        rebuild_function(func, new_instrs);
        if saved_max > func.max_register {
            func.max_register = saved_max;
        }
        return true;
    }
    false
}

// =====================================================================
// Pass: sweep dead pure loads
// =====================================================================

/// Replace `LoadConst`/`MakeClosure` instructions whose value is provably
/// dead (no read of the destination register is reachable before it is
/// rewritten) with `Nop`.
///
/// Purpose: after closure-apply rewriting and inlining, the original
/// `LoadConst <closure>` is dead, but native lowering materializes every
/// `LoadConst` value eagerly and cannot represent closures — the dead load
/// would fail the whole function. Deadness uses the same CFG walk as the
/// reaching-definition analysis: from the load's successors, stop at any
/// instruction that rewrites the register; if any visited instruction reads
/// it first, the load is live. Trivially semantics-preserving.
fn sweep_dead_pure_loads(func: &mut BytecodeFunction) {
    loop {
        let mut changed = false;
        for pc in 0..func.instructions.len() {
            let rd = match func.instructions[pc] {
                Opcode::LoadConst { rd, .. } | Opcode::MakeClosure { rd, .. } => rd,
                _ => continue,
            };
            if pure_load_is_dead(&func.instructions, pc, rd) {
                func.instructions[pc] = Opcode::Nop;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// True when no read of `rd` is reachable from the instruction after `pc`
/// before `rd` is redefined.
fn pure_load_is_dead(instrs: &[Opcode], pc: usize, rd: Register) -> bool {
    let mut pending: Vec<usize> = Vec::new();
    control_successors(instrs, pc, &mut pending);
    let mut visited = vec![false; instrs.len()];
    while let Some(cur) = pending.pop() {
        if cur >= instrs.len() || visited[cur] {
            continue;
        }
        visited[cur] = true;
        let op = &instrs[cur];
        // Reads are checked BEFORE writes: an op may read and write the same
        // register (e.g. `Move rd <- rd`-style flows through loop opcodes).
        let mut reads = false;
        for_each_source_register(op, |r| {
            if r == rd {
                reads = true;
            }
        });
        if reads {
            return false;
        }
        if op.dest_register() == Some(rd) || op.binding_register() == Some(rd) {
            // Redefined: this path is dead for the original value.
            continue;
        }
        control_successors(instrs, cur, &mut pending);
    }
    true
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::BytecodeChunk;

    fn set_value(elems: Vec<Value>) -> Value {
        Value::set(elems)
    }

    /// Execute a chunk function in a bare interpreter for pure boolean
    /// bytecode (no state vars). Mirrors the VM's quantifier semantics.
    /// Kept minimal: only the opcodes the fixtures use.
    fn run_bool(func: &BytecodeFunction, pool: &ConstantPool) -> bool {
        let mut regs: Vec<Value> = vec![Value::Bool(false); 256];
        #[derive(Clone)]
        struct Loop {
            elements: Vec<Value>,
            index: usize,
        }
        let mut stack: Vec<Loop> = Vec::new();
        let mut pc = 0usize;
        let instrs = &func.instructions;
        let mut steps = 0usize;
        while pc < instrs.len() {
            steps += 1;
            assert!(steps < 100_000, "runaway test interpreter");
            match instrs[pc] {
                Opcode::Nop => {}
                Opcode::LoadImm { rd, value } => regs[rd as usize] = Value::SmallInt(value),
                Opcode::LoadBool { rd, value } => regs[rd as usize] = Value::Bool(value),
                Opcode::LoadConst { rd, idx } => {
                    regs[rd as usize] = pool.get_value(idx).clone();
                }
                Opcode::Move { rd, rs } => regs[rd as usize] = regs[rs as usize].clone(),
                Opcode::Eq { rd, r1, r2 } => {
                    regs[rd as usize] = Value::Bool(regs[r1 as usize] == regs[r2 as usize]);
                }
                Opcode::LtInt { rd, r1, r2 } => {
                    let (Value::SmallInt(a), Value::SmallInt(b)) =
                        (&regs[r1 as usize], &regs[r2 as usize])
                    else {
                        panic!("LtInt on non-int");
                    };
                    regs[rd as usize] = Value::Bool(a < b);
                }
                Opcode::SetIn { rd, elem, set } => {
                    let contains = regs[set as usize]
                        .iter_set()
                        .expect("SetIn set operand")
                        .any(|v| v == regs[elem as usize]);
                    regs[rd as usize] = Value::Bool(contains);
                }
                Opcode::Jump { offset } => {
                    pc = (pc as i64 + i64::from(offset)) as usize;
                    continue;
                }
                Opcode::JumpTrue { rs, offset } => {
                    if regs[rs as usize] == Value::Bool(true) {
                        pc = (pc as i64 + i64::from(offset)) as usize;
                        continue;
                    }
                }
                Opcode::JumpFalse { rs, offset } => {
                    if regs[rs as usize] == Value::Bool(false) {
                        pc = (pc as i64 + i64::from(offset)) as usize;
                        continue;
                    }
                }
                Opcode::ForallBegin {
                    rd,
                    r_binding,
                    r_domain,
                    loop_end,
                } => {
                    let elements: Vec<Value> = regs[r_domain as usize]
                        .iter_set()
                        .expect("domain")
                        .collect();
                    if elements.is_empty() {
                        regs[rd as usize] = Value::Bool(true);
                        pc = (pc as i64 + i64::from(loop_end)) as usize;
                        continue;
                    }
                    regs[r_binding as usize] = elements[0].clone();
                    regs[rd as usize] = Value::Bool(true);
                    stack.push(Loop { elements, index: 0 });
                }
                Opcode::ExistsBegin {
                    rd,
                    r_binding,
                    r_domain,
                    loop_end,
                } => {
                    let elements: Vec<Value> = regs[r_domain as usize]
                        .iter_set()
                        .expect("domain")
                        .collect();
                    if elements.is_empty() {
                        regs[rd as usize] = Value::Bool(false);
                        pc = (pc as i64 + i64::from(loop_end)) as usize;
                        continue;
                    }
                    regs[r_binding as usize] = elements[0].clone();
                    regs[rd as usize] = Value::Bool(false);
                    stack.push(Loop { elements, index: 0 });
                }
                Opcode::ForallNext {
                    rd,
                    r_binding,
                    r_body,
                    loop_begin,
                } => {
                    let body = regs[r_body as usize] == Value::Bool(true);
                    if !body {
                        regs[rd as usize] = Value::Bool(false);
                        stack.pop();
                    } else {
                        let state = stack.last_mut().expect("loop state");
                        state.index += 1;
                        if state.index < state.elements.len() {
                            regs[r_binding as usize] = state.elements[state.index].clone();
                            pc = (pc as i64 + i64::from(loop_begin)) as usize;
                            continue;
                        }
                        regs[rd as usize] = Value::Bool(true);
                        stack.pop();
                    }
                }
                Opcode::ExistsNext {
                    rd,
                    r_binding,
                    r_body,
                    loop_begin,
                } => {
                    let body = regs[r_body as usize] == Value::Bool(true);
                    if body {
                        regs[rd as usize] = Value::Bool(true);
                        stack.pop();
                    } else {
                        let state = stack.last_mut().expect("loop state");
                        state.index += 1;
                        if state.index < state.elements.len() {
                            regs[r_binding as usize] = state.elements[state.index].clone();
                            pc = (pc as i64 + i64::from(loop_begin)) as usize;
                            continue;
                        }
                        regs[rd as usize] = Value::Bool(false);
                        stack.pop();
                    }
                }
                Opcode::Ret { rs } => {
                    return regs[rs as usize] == Value::Bool(true);
                }
                ref other => panic!("test interpreter: unsupported opcode {other:?}"),
            }
            pc += 1;
        }
        panic!("function fell off the end");
    }

    #[test]
    fn test_unroll_exists_truth_table() {
        for probe in 0..5 {
            let mut pool = ConstantPool::new();
            // Build with correct offsets: begin at 2, body at 3, next at 4,
            // empty-domain target must be next+1 = 5 -> loop_end = 3.
            let dom = pool.add_value(set_value(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ]));
            let mut f = BytecodeFunction::new("E".to_string(), 0);
            f.emit(Opcode::LoadImm {
                rd: 1,
                value: probe,
            });
            f.emit(Opcode::LoadConst { rd: 2, idx: dom });
            f.emit(Opcode::ExistsBegin {
                rd: 3,
                r_binding: 4,
                r_domain: 2,
                loop_end: 3,
            });
            f.emit(Opcode::Eq {
                rd: 5,
                r1: 4,
                r2: 1,
            });
            f.emit(Opcode::ExistsNext {
                rd: 3,
                r_binding: 4,
                r_body: 5,
                loop_begin: -1,
            });
            f.emit(Opcode::Ret { rs: 3 });

            let before = run_bool(&f, &pool);
            let normalized = normalize_action_function(&f, &[], &mut pool)
                .expect("constant-domain exists must unroll");
            assert!(
                !normalized
                    .instructions
                    .iter()
                    .any(|op| matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. })),
                "unrolled function must contain no exists loop"
            );
            let after = run_bool(&normalized, &pool);
            assert_eq!(before, after, "probe={probe}");
            assert_eq!(after, (1..=3).contains(&probe), "probe={probe}");
        }
    }

    #[test]
    fn test_unroll_nested_forall_exists_over_set_of_sets() {
        // \A ms \in {{1,2},{3}} : \E x \in ms : x = probe is FALSE for all
        // probes (no single x in every ms); instead test
        // \E ms \in {{1,2},{3}} : \A x \in ms : x = probe
        //   probe=3 -> true (ms={3}), probe=1 -> false ({1,2} needs both).
        for (probe, expected) in [(3i64, true), (1, false), (7, false)] {
            let mut pool = ConstantPool::new();
            let dom = pool.add_value(set_value(vec![
                set_value(vec![Value::SmallInt(1), Value::SmallInt(2)]),
                set_value(vec![Value::SmallInt(3)]),
            ]));
            let mut f = BytecodeFunction::new("NE".to_string(), 0);
            f.emit(Opcode::LoadImm {
                rd: 1,
                value: probe,
            }); // 0
            f.emit(Opcode::LoadConst { rd: 2, idx: dom }); // 1
            f.emit(Opcode::ExistsBegin {
                rd: 3,
                r_binding: 4,
                r_domain: 2,
                loop_end: 6,
            }); // 2, empty -> 8
            f.emit(Opcode::ForallBegin {
                rd: 5,
                r_binding: 6,
                r_domain: 4,
                loop_end: 3,
            }); // 3, empty -> 6
            f.emit(Opcode::Eq {
                rd: 7,
                r1: 6,
                r2: 1,
            }); // 4
            f.emit(Opcode::ForallNext {
                rd: 5,
                r_binding: 6,
                r_body: 7,
                loop_begin: -1,
            }); // 5 -> 4
            f.emit(Opcode::Move { rd: 8, rs: 5 }); // 6
            f.emit(Opcode::ExistsNext {
                rd: 3,
                r_binding: 4,
                r_body: 8,
                loop_begin: -4,
            }); // 7 -> 3
            f.emit(Opcode::Ret { rs: 3 }); // 8

            let before = run_bool(&f, &pool);
            assert_eq!(before, expected, "fixture semantics probe={probe}");
            let normalized = normalize_action_function(&f, &[], &mut pool)
                .expect("nested constant quantifiers must unroll");
            assert!(
                !normalized.instructions.iter().any(|op| matches!(
                    op,
                    Opcode::ExistsBegin { .. }
                        | Opcode::ExistsNext { .. }
                        | Opcode::ForallBegin { .. }
                        | Opcode::ForallNext { .. }
                )),
                "all constant quantifiers must be unrolled"
            );
            let after = run_bool(&normalized, &pool);
            assert_eq!(before, after, "probe={probe}");
        }
    }

    #[test]
    fn test_unroll_refuses_oversize_domain() {
        let mut pool = ConstantPool::new();
        let big: Vec<Value> = (0..(MAX_UNROLL_DOMAIN as i64 + 1))
            .map(Value::SmallInt)
            .collect();
        let dom = pool.add_value(set_value(big));
        let mut f = BytecodeFunction::new("Big".to_string(), 0);
        f.emit(Opcode::LoadImm { rd: 1, value: 1 });
        f.emit(Opcode::LoadConst { rd: 2, idx: dom });
        f.emit(Opcode::ExistsBegin {
            rd: 3,
            r_binding: 4,
            r_domain: 2,
            loop_end: 3,
        });
        f.emit(Opcode::Eq {
            rd: 5,
            r1: 4,
            r2: 1,
        });
        f.emit(Opcode::ExistsNext {
            rd: 3,
            r_binding: 4,
            r_body: 5,
            loop_begin: -1,
        });
        f.emit(Opcode::Ret { rs: 3 });
        assert!(
            normalize_action_function(&f, &[], &mut pool).is_none(),
            "oversize domains must fail closed"
        );
    }

    #[test]
    fn test_unroll_refuses_effectful_body() {
        let mut pool = ConstantPool::new();
        let dom = pool.add_value(set_value(vec![Value::SmallInt(1), Value::SmallInt(2)]));
        let mut f = BytecodeFunction::new("Eff".to_string(), 0);
        f.emit(Opcode::LoadConst { rd: 2, idx: dom });
        f.emit(Opcode::ExistsBegin {
            rd: 3,
            r_binding: 4,
            r_domain: 2,
            loop_end: 4,
        });
        f.emit(Opcode::StoreVar { var_idx: 0, rs: 4 });
        f.emit(Opcode::LoadBool { rd: 5, value: true });
        f.emit(Opcode::ExistsNext {
            rd: 3,
            r_binding: 4,
            r_body: 5,
            loop_begin: -2,
        });
        f.emit(Opcode::Ret { rs: 3 });
        assert!(
            normalize_action_function(&f, &[], &mut pool).is_none(),
            "successor-producing exists must NOT be unrolled (multi-successor semantics)"
        );
    }

    #[test]
    fn test_unroll_refuses_runtime_domain() {
        let mut pool = ConstantPool::new();
        let mut f = BytecodeFunction::new("RT".to_string(), 0);
        f.emit(Opcode::LoadVar { rd: 2, var_idx: 0 });
        f.emit(Opcode::ExistsBegin {
            rd: 3,
            r_binding: 4,
            r_domain: 2,
            loop_end: 3,
        });
        f.emit(Opcode::LoadBool { rd: 5, value: true });
        f.emit(Opcode::ExistsNext {
            rd: 3,
            r_binding: 4,
            r_body: 5,
            loop_begin: -1,
        });
        f.emit(Opcode::Ret { rs: 3 });
        assert!(
            normalize_action_function(&f, &[], &mut pool).is_none(),
            "state-dependent domains must fail closed"
        );
    }

    #[test]
    fn test_inline_call_with_load_prime() {
        // Callee: Send-like predicate `result = (prime(0) == arg0)`.
        let mut callee = BytecodeFunction::new("Send".to_string(), 1);
        callee.emit(Opcode::LoadPrime { rd: 1, var_idx: 0 }); // uses reg 1
        callee.emit(Opcode::Eq {
            rd: 2,
            r1: 1,
            r2: 0,
        });
        callee.emit(Opcode::Move { rd: 0, rs: 2 });
        callee.emit(Opcode::Ret { rs: 0 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(callee);

        let mut pool = ConstantPool::new();
        let mut caller = BytecodeFunction::new("A".to_string(), 0);
        caller.emit(Opcode::LoadImm { rd: 5, value: 42 });
        caller.emit(Opcode::Call {
            rd: 6,
            op_idx: 0,
            args_start: 5,
            argc: 1,
        });
        caller.emit(Opcode::Ret { rs: 6 });

        let normalized = normalize_action_function(&caller, &chunk.functions, &mut pool)
            .expect("call with LoadPrime callee must inline");
        assert!(
            normalized
                .instructions
                .iter()
                .any(|op| matches!(op, Opcode::LoadPrime { .. })),
            "LoadPrime must be exposed at top level"
        );
        assert!(
            !normalized
                .instructions
                .iter()
                .any(|op| matches!(op, Opcode::Call { .. })),
            "Call must be gone"
        );
        // The caller's argument register must not be written by the splice
        // except through the prologue copy into fresh registers.
        assert!(matches!(
            normalized.instructions[0],
            Opcode::LoadImm { rd: 5, value: 42 }
        ));
    }

    #[test]
    fn test_inline_refuses_multi_ret_callee() {
        let mut callee = BytecodeFunction::new("TwoRets".to_string(), 0);
        callee.emit(Opcode::LoadBool { rd: 0, value: true });
        callee.emit(Opcode::JumpTrue { rs: 0, offset: 2 });
        callee.emit(Opcode::Ret { rs: 0 });
        callee.emit(Opcode::Ret { rs: 0 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(callee);

        let mut pool = ConstantPool::new();
        let mut caller = BytecodeFunction::new("A".to_string(), 0);
        caller.emit(Opcode::Call {
            rd: 1,
            op_idx: 0,
            args_start: 0,
            argc: 0,
        });
        caller.emit(Opcode::Ret { rs: 1 });
        assert!(
            normalize_action_function(&caller, &chunk.functions, &mut pool).is_none(),
            "multi-Ret callees must fail closed"
        );
    }

    #[test]
    fn test_inline_refuses_round_step_eq_callee() {
        let mut callee = BytecodeFunction::new("VmOnlyRoundStep".to_string(), 2);
        callee.emit(Opcode::RoundStepEq {
            rd: 2,
            child: 0,
            parent: 1,
        });
        callee.emit(Opcode::Ret { rs: 2 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(callee);

        let mut pool = ConstantPool::new();
        let mut caller = BytecodeFunction::new("A".to_string(), 0);
        caller.emit(Opcode::LoadImm { rd: 0, value: 1 });
        caller.emit(Opcode::LoadImm { rd: 1, value: 2 });
        caller.emit(Opcode::Call {
            rd: 2,
            op_idx: 0,
            args_start: 0,
            argc: 2,
        });
        caller.emit(Opcode::Ret { rs: 2 });

        assert!(
            normalize_action_function(&caller, &chunk.functions, &mut pool).is_none(),
            "RoundStepEq callees must fail closed during action normalization"
        );
    }

    #[test]
    fn test_closure_apply_rewrite_and_inline() {
        use std::sync::Arc;
        use tla_core::ast::Expr;
        use tla_core::Spanned;

        // Sub-function: identity-ish bool check `arg0 = 1`.
        let mut sub = BytecodeFunction::new("<lambda>".to_string(), 1);
        sub.emit(Opcode::LoadImm { rd: 1, value: 1 });
        sub.emit(Opcode::Eq {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        sub.emit(Opcode::Move { rd: 0, rs: 2 });
        sub.emit(Opcode::Ret { rs: 0 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(sub);

        let closure = tla_value::ClosureValue::new(
            vec!["x".to_string()],
            Spanned {
                node: Expr::Bool(true),
                span: tla_core::Span::default(),
            },
            Arc::new(Default::default()),
            None,
        )
        .with_bytecode_func_idx(0);

        let mut pool = ConstantPool::new();
        let closure_idx = pool.add_value(Value::Closure(Rp::new(closure)));

        let mut caller = BytecodeFunction::new("A".to_string(), 0);
        caller.emit(Opcode::LoadConst {
            rd: 0,
            idx: closure_idx,
        });
        caller.emit(Opcode::LoadImm { rd: 1, value: 1 });
        caller.emit(Opcode::ValueApply {
            rd: 2,
            func: 0,
            args_start: 1,
            argc: 1,
        });
        caller.emit(Opcode::Ret { rs: 2 });

        let normalized = normalize_action_function(&caller, &chunk.functions, &mut pool)
            .expect("const-closure ValueApply must rewrite and inline");
        assert!(
            !normalized
                .instructions
                .iter()
                .any(|op| matches!(op, Opcode::ValueApply { .. } | Opcode::Call { .. })),
            "ValueApply must be rewritten to Call and inlined"
        );
        // The dead closure LoadConst must be swept (native lowering cannot
        // materialize closure constants).
        assert!(
            matches!(normalized.instructions[0], Opcode::Nop),
            "dead closure LoadConst must be Nop'd, got {:?}",
            normalized.instructions[0]
        );
        assert!(run_bool(&normalized, &pool), "1 = 1 must hold");
    }

    #[test]
    fn test_no_change_returns_none() {
        let mut pool = ConstantPool::new();
        let mut f = BytecodeFunction::new("Plain".to_string(), 0);
        f.emit(Opcode::LoadBool { rd: 0, value: true });
        f.emit(Opcode::Ret { rs: 0 });
        assert!(normalize_action_function(&f, &[], &mut pool).is_none());
    }
}
