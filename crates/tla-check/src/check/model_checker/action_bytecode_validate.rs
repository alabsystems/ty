// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Validation for next-state action bytecode eligibility.
//!
//! The action transform rewrites entry actions from predicate form into
//! next-state functions, but trust-codegen and the JIT may still traverse `Call`
//! edges through the shared bytecode chunk. To keep the next-state ABI
//! sound, retained entry actions must not reach helper functions that still
//! depend on primed-state evaluation machinery.

use rustc_hash::FxHashSet;
use std::collections::{BTreeMap, BTreeSet};
use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

/// Packed `Option<bool>` facts for all 256 bytecode registers.
///
/// Semantically identical to `[Option<bool>; 256]` (per-register tri-state),
/// but stored as two 256-bit bitmaps (known + value). The unknown-value bit is
/// kept normalized to 0 so `Eq`/`Hash` stay canonical. This keeps the
/// path-state hashing in the validators below at 64 bytes per state instead of
/// hashing a 256-element `Option<bool>` array (the dominant fixed cost of
/// `validate_next_state_action_chunk` on action-heavy specs).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
struct KnownBools {
    known: [u64; 4],
    value: [u64; 4],
}

impl KnownBools {
    #[inline]
    fn get(&self, reg: u8) -> Option<bool> {
        let w = (reg >> 6) as usize;
        let b = reg & 63;
        if (self.known[w] >> b) & 1 == 0 {
            None
        } else {
            Some((self.value[w] >> b) & 1 == 1)
        }
    }

    #[inline]
    fn set(&mut self, reg: u8, fact: Option<bool>) {
        let w = (reg >> 6) as usize;
        let b = reg & 63;
        match fact {
            None => {
                self.known[w] &= !(1u64 << b);
                self.value[w] &= !(1u64 << b);
            }
            Some(true) => {
                self.known[w] |= 1u64 << b;
                self.value[w] |= 1u64 << b;
            }
            Some(false) => {
                self.known[w] |= 1u64 << b;
                self.value[w] &= !(1u64 << b);
            }
        }
    }
}

/// Validate a transformed next-state action entry and every reachable callee.
///
/// Entry actions may write successor state via `StoreVar` and `Unchanged`,
/// but reachable helper functions must stay pure current-state computations.
/// Any primed-state opcode that survives in the closure is rejected before
/// trust_cg/JIT can compile it.
pub(crate) fn validate_next_state_action_chunk(
    entry_func_idx: u16,
    entry_instructions: &[Opcode],
    chunk: &BytecodeChunk,
    state_var_count: usize,
) -> Result<(), String> {
    validate_entry_shape(entry_instructions, &chunk.constants, state_var_count)?;
    validate_reachable_callees(entry_func_idx, entry_instructions, chunk)
}

/// Certify a transformed entry for the single-successor Value action VM.
///
/// This is deliberately stricter than [`validate_next_state_action_chunk`].
/// The shared validator also serves trust-codegen/JIT backends, whose runtime
/// contracts differ from the Value VM's transactional successor overlay. Keep
/// those existing admissions unchanged and layer the Value-VM-only rules here:
///
/// - the transformed entry has no runtime arguments;
/// - every potentially enabled return binds the complete successor state;
/// - successor bindings do not occur inside residual iteration bodies; and
/// - context-dependent opcodes that need an evaluator/closure environment are
///   absent from the reachable call closure.
pub(crate) fn validate_value_action_vm_eligibility(
    entry_func_idx: u16,
    entry_instructions: &[Opcode],
    chunk: &BytecodeChunk,
    state_var_count: usize,
) -> Result<(), String> {
    let entry = chunk
        .functions
        .get(entry_func_idx as usize)
        .ok_or_else(|| format!("Value-action VM entry function {entry_func_idx} is missing"))?;
    if entry.arity != 0 {
        return Err(format!(
            "Value-action VM entry {entry_func_idx} must have arity 0, got {}",
            entry.arity
        ));
    }

    // Do not duplicate or relax the validator shared with the native action
    // backends. Value-action eligibility is an additional certification layer.
    validate_next_state_action_chunk(entry_func_idx, entry_instructions, chunk, state_var_count)?;
    validate_value_action_vm_opcode_closure(entry_func_idx, entry_instructions, chunk)?;
    validate_value_action_vm_successor_bindings(
        entry_instructions,
        &chunk.constants,
        state_var_count,
    )
}

/// Certify that a Value-action entry can reuse a stale register frame.
///
/// This is an optimization-only certificate: failure must retain the ordinary
/// fully initialized VM path, not reject the Value-action plan. Reachable
/// entry-local loops and backedges are deliberately unsupported for now. A
/// direct `Call` is safe because the callee executes in its own fully reset
/// frame; only the caller's argument reads and result write matter here.
pub(crate) fn certify_value_action_vm_register_reuse(
    function: &BytecodeFunction,
) -> Result<(), String> {
    let max_register = function.max_register;
    if function.arity != 0 {
        return Err(format!(
            "register-reusing action entry must have arity 0, got {}",
            function.arity
        ));
    }

    let entry_defs = DefinedRegisters::default();

    let instructions = &function.instructions;
    if instructions.is_empty() {
        return require_defined_register(entry_defs, 0, 0, "implicit return");
    }

    // Every accepted edge is strictly forward, so a single PC-order pass sees
    // all reachable predecessors before processing a join. `None` means the PC
    // is unreachable; subsequent predecessors merge by must-def intersection.
    let mut incoming = vec![None; instructions.len()];
    incoming[0] = Some(entry_defs);

    for (pc, op) in instructions.iter().copied().enumerate() {
        let Some(mut defined) = incoming[pc] else {
            continue;
        };

        if let Some(rd) = op.dest_register() {
            require_declared_register(rd, max_register, pc, "destination")?;
        }
        if let Some(binding) = op.binding_register() {
            require_declared_register(binding, max_register, pc, "binding destination")?;
        }

        if matches!(
            op,
            Opcode::ForallBegin { .. }
                | Opcode::ForallNext { .. }
                | Opcode::ExistsBegin { .. }
                | Opcode::ExistsNext { .. }
                | Opcode::ChooseBegin { .. }
                | Opcode::ChooseNext { .. }
                | Opcode::SetBuilderBegin { .. }
                | Opcode::SetFilterBegin { .. }
                | Opcode::FuncDefBegin { .. }
                | Opcode::LoopNext { .. }
        ) {
            return Err(format!(
                "reachable entry-local loop opcode at pc {pc}: {op:?}"
            ));
        }

        validate_register_reuse_reads(op, defined, max_register, pc)?;

        // CondMove preserves the previous destination on its false arm, so it
        // cannot establish a new must-definition. All other destinations are
        // written before their successful continuation.
        if !matches!(op, Opcode::CondMove { .. }) {
            if let Some(rd) = op.dest_register() {
                defined.insert(rd);
            }
        }

        match op {
            Opcode::Jump { offset } => {
                let target = register_reuse_forward_target(pc, offset, instructions.len())?;
                merge_register_reuse_edge(&mut incoming[target], defined);
            }
            Opcode::JumpTrue { offset, .. } | Opcode::JumpFalse { offset, .. } => {
                let target = register_reuse_forward_target(pc, offset, instructions.len())?;
                merge_register_reuse_edge(&mut incoming[target], defined);
                propagate_register_reuse_fallthrough(&mut incoming, pc, defined)?;
            }
            Opcode::Ret { .. } | Opcode::Halt => {}
            _ => propagate_register_reuse_fallthrough(&mut incoming, pc, defined)?,
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DefinedRegisters([u64; 4]);

impl DefinedRegisters {
    #[inline]
    fn contains(self, reg: u8) -> bool {
        let word = usize::from(reg >> 6);
        let bit = reg & 63;
        self.0[word] & (1_u64 << bit) != 0
    }

    #[inline]
    fn insert(&mut self, reg: u8) {
        let word = usize::from(reg >> 6);
        let bit = reg & 63;
        self.0[word] |= 1_u64 << bit;
    }

    #[inline]
    fn intersect_assign(&mut self, other: Self) {
        for (word, other_word) in self.0.iter_mut().zip(other.0) {
            *word &= other_word;
        }
    }
}

fn require_declared_register(
    reg: u8,
    max_register: u8,
    pc: usize,
    role: &str,
) -> Result<(), String> {
    if reg > max_register {
        return Err(format!(
            "{role} r{reg} at pc {pc} exceeds declared max register r{max_register}"
        ));
    }
    Ok(())
}

fn require_defined_register(
    defined: DefinedRegisters,
    reg: u8,
    pc: usize,
    role: &str,
) -> Result<(), String> {
    if !defined.contains(reg) {
        return Err(format!(
            "{role} reads r{reg} before a definite assignment at pc {pc}"
        ));
    }
    Ok(())
}

fn validate_register_reuse_read(
    defined: DefinedRegisters,
    max_register: u8,
    pc: usize,
    reg: u8,
) -> Result<(), String> {
    require_declared_register(reg, max_register, pc, "source")?;
    require_defined_register(defined, reg, pc, "opcode")
}

fn validate_register_reuse_range(
    defined: DefinedRegisters,
    max_register: u8,
    pc: usize,
    start: u8,
    count: u8,
) -> Result<(), String> {
    for offset in 0..count {
        let reg = start.checked_add(offset).ok_or_else(|| {
            format!("source register range at pc {pc} overflows the 256-register file")
        })?;
        validate_register_reuse_read(defined, max_register, pc, reg)?;
    }
    Ok(())
}

/// Validate every register read performed by one opcode. Keep this match
/// exhaustive so additions to the bytecode ISA fail compilation until their
/// stale-frame semantics are classified.
fn validate_register_reuse_reads(
    op: Opcode,
    defined: DefinedRegisters,
    max_register: u8,
    pc: usize,
) -> Result<(), String> {
    let read = |reg| validate_register_reuse_read(defined, max_register, pc, reg);
    let range =
        |start, count| validate_register_reuse_range(defined, max_register, pc, start, count);

    match op {
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
        | Opcode::JumpFalse { rs, .. } => read(rs)?,
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
            read(r1)?;
            read(r2)?;
        }
        Opcode::Range { lo, hi, .. } => {
            read(lo)?;
            read(hi)?;
        }
        Opcode::KSubset { base, k, .. } => {
            read(base)?;
            read(k)?;
        }
        Opcode::SetIn { elem, set, .. } => {
            read(elem)?;
            read(set)?;
        }
        Opcode::Tuple2SetIn {
            first, second, set, ..
        } => {
            read(first)?;
            read(second)?;
            read(set)?;
        }
        Opcode::SetEnumSubseteq {
            start, count, set, ..
        } => {
            range(start, count)?;
            read(set)?;
        }
        Opcode::RoundStepEq { child, parent, .. } => {
            read(child)?;
            read(parent)?;
        }
        Opcode::FuncApply { func, arg, .. } => {
            read(func)?;
            read(arg)?;
        }
        Opcode::FuncSet {
            domain, range: r, ..
        } => {
            read(domain)?;
            read(r)?;
        }
        Opcode::FuncExcept {
            func, path, val, ..
        } => {
            read(func)?;
            read(path)?;
            read(val)?;
        }
        Opcode::EqFuncExcept {
            lhs,
            func,
            path,
            val,
            ..
        } => {
            read(lhs)?;
            read(func)?;
            read(path)?;
            read(val)?;
        }
        Opcode::EqRecordNew {
            lhs,
            values_start,
            count,
            ..
        } => {
            read(lhs)?;
            range(values_start, count)?;
        }
        Opcode::CondMove { cond, rs, .. } => {
            read(cond)?;
            read(rs)?;
        }
        Opcode::SetEnum { start, count, .. }
        | Opcode::TupleNew { start, count, .. }
        | Opcode::SeqNew { start, count, .. }
        | Opcode::Times { start, count, .. } => range(start, count)?,
        Opcode::RecordNew {
            values_start,
            count,
            ..
        }
        | Opcode::RecordSet {
            values_start,
            count,
            ..
        } => range(values_start, count)?,
        Opcode::FuncDef {
            r_domain,
            r_binding,
            ..
        } => {
            read(r_domain)?;
            read(r_binding)?;
        }
        Opcode::Call {
            args_start, argc, ..
        }
        | Opcode::CallExternal {
            args_start, argc, ..
        }
        | Opcode::CallBuiltin {
            args_start, argc, ..
        } => range(args_start, argc)?,
        Opcode::ValueApply {
            func,
            args_start,
            argc,
            ..
        } => {
            read(func)?;
            range(args_start, argc)?;
        }
        Opcode::MakeClosure {
            captures_start,
            capture_count,
            ..
        } => range(captures_start, capture_count)?,
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
            read(r_binding)?;
            read(r_domain)?;
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
            read(r_binding)?;
            read(r_body)?;
        }
    }
    Ok(())
}

fn register_reuse_forward_target(pc: usize, offset: i32, len: usize) -> Result<usize, String> {
    let target = (pc as i64)
        .checked_add(i64::from(offset))
        .and_then(|target| usize::try_from(target).ok())
        .filter(|&target| target < len)
        .ok_or_else(|| format!("invalid register-reuse jump target from pc {pc}: {offset}"))?;
    if target <= pc {
        return Err(format!(
            "register-reuse certificate rejects backedge from pc {pc} to pc {target}"
        ));
    }
    Ok(target)
}

fn merge_register_reuse_edge(slot: &mut Option<DefinedRegisters>, defined: DefinedRegisters) {
    if let Some(existing) = slot {
        existing.intersect_assign(defined);
    } else {
        *slot = Some(defined);
    }
}

fn propagate_register_reuse_fallthrough(
    incoming: &mut [Option<DefinedRegisters>],
    pc: usize,
    defined: DefinedRegisters,
) -> Result<(), String> {
    if pc + 1 < incoming.len() {
        merge_register_reuse_edge(&mut incoming[pc + 1], defined);
        Ok(())
    } else {
        require_defined_register(defined, 0, pc, "implicit return")
    }
}

fn validate_value_action_vm_opcode_closure(
    entry_func_idx: u16,
    entry_instructions: &[Opcode],
    chunk: &BytecodeChunk,
) -> Result<(), String> {
    let mut visited = FxHashSet::default();
    let mut pending = validate_value_action_vm_function_opcodes(
        &format!("entry {entry_func_idx}"),
        entry_instructions,
        chunk,
    )?;

    while let Some(func_idx) = pending.pop() {
        if !visited.insert(func_idx) {
            continue;
        }
        let func = chunk.functions.get(func_idx as usize).ok_or_else(|| {
            format!("Value-action VM entry {entry_func_idx} references missing callee {func_idx}")
        })?;
        pending.extend(validate_value_action_vm_function_opcodes(
            &format!("reachable callee {func_idx}"),
            &func.instructions,
            chunk,
        )?);
    }

    Ok(())
}

fn validate_value_action_vm_function_opcodes(
    location: &str,
    instructions: &[Opcode],
    chunk: &BytecodeChunk,
) -> Result<Vec<u16>, String> {
    let reachable = reachable_instruction_pcs(instructions)?;
    let mut callees = Vec::new();
    for pc in reachable {
        match instructions[pc] {
            Opcode::CallExternal { .. } => {
                return Err(format!(
                    "Value-action VM {location} contains unsupported CallExternal at pc {pc}"
                ));
            }
            Opcode::ValueApply { .. } => {
                return Err(format!(
                    "Value-action VM {location} contains unsupported ValueApply at pc {pc}"
                ));
            }
            Opcode::MakeClosure { .. } => {
                return Err(format!(
                    "Value-action VM {location} contains unsupported MakeClosure at pc {pc}"
                ));
            }
            Opcode::FuncDef { .. } => {
                return Err(format!(
                    "Value-action VM {location} contains unsupported FuncDef at pc {pc}"
                ));
            }
            Opcode::SetPrimeMode { .. } => {
                return Err(format!(
                    "Value-action VM {location} contains unsupported SetPrimeMode at pc {pc}"
                ));
            }
            Opcode::Call { op_idx, argc, .. } => {
                let callee = chunk.functions.get(op_idx as usize).ok_or_else(|| {
                    format!(
                        "Value-action VM {location} Call at pc {pc} references missing callee {op_idx}"
                    )
                })?;
                if callee.arity != argc {
                    return Err(format!(
                        "Value-action VM {location} Call at pc {pc} passes {argc} arguments to callee {op_idx} with arity {}",
                        callee.arity
                    ));
                }
                callees.push(op_idx);
            }
            _ => {}
        }
    }
    Ok(callees)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueActionLoopKind {
    Forall,
    Exists,
    Choose,
    SetBuilder,
    SetFilter,
    FuncDef,
}

fn validate_value_action_vm_successor_bindings(
    instructions: &[Opcode],
    constants: &tla_tir::bytecode::ConstantPool,
    state_var_count: usize,
) -> Result<(), String> {
    if state_var_count > usize::from(u16::MAX) + 1 {
        return Err(format!(
            "Value-action VM state has {state_var_count} variables, exceeding the u16 slot space"
        ));
    }

    for (pc, op) in instructions.iter().enumerate() {
        match *op {
            Opcode::StoreVar { var_idx, .. } => {
                if usize::from(var_idx) >= state_var_count {
                    return Err(format!(
                        "Value-action VM StoreVar at pc {pc} has out-of-range successor variable {var_idx}"
                    ));
                }
            }
            Opcode::Unchanged { start, count, .. } => {
                for var_idx in unchanged_var_indices(constants, start, count, pc)? {
                    if usize::from(var_idx) >= state_var_count {
                        return Err(format!(
                            "Value-action VM Unchanged at pc {pc} has out-of-range successor variable {var_idx}"
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    validate_no_successor_bindings_in_residual_loops(instructions)?;
    validate_full_successor_binding_on_enabled_returns(instructions, constants, state_var_count)
}

fn validate_no_successor_bindings_in_residual_loops(instructions: &[Opcode]) -> Result<(), String> {
    let len = instructions.len();
    for (begin_pc, op) in instructions.iter().enumerate() {
        let (kind, loop_end) = match *op {
            Opcode::ForallBegin { loop_end, .. } => (ValueActionLoopKind::Forall, loop_end),
            Opcode::ExistsBegin { loop_end, .. } => (ValueActionLoopKind::Exists, loop_end),
            Opcode::ChooseBegin { loop_end, .. } => (ValueActionLoopKind::Choose, loop_end),
            Opcode::SetBuilderBegin { loop_end, .. } => (ValueActionLoopKind::SetBuilder, loop_end),
            Opcode::SetFilterBegin { loop_end, .. } => (ValueActionLoopKind::SetFilter, loop_end),
            Opcode::FuncDefBegin { loop_end, .. } => (ValueActionLoopKind::FuncDef, loop_end),
            _ => continue,
        };

        let end = checked_action_loop_target(begin_pc, loop_end, len).ok_or_else(|| {
            format!(
                "Value-action VM {kind:?} loop at pc {begin_pc} has invalid loop_end {loop_end}"
            )
        })?;
        if end <= begin_pc + 1 {
            return Err(format!(
                "Value-action VM {kind:?} loop at pc {begin_pc} has an empty/backward lexical body"
            ));
        }
        let next_pc = end - 1;
        let loop_begin = match (kind, instructions[next_pc]) {
            (ValueActionLoopKind::Forall, Opcode::ForallNext { loop_begin, .. })
            | (ValueActionLoopKind::Exists, Opcode::ExistsNext { loop_begin, .. })
            | (ValueActionLoopKind::Choose, Opcode::ChooseNext { loop_begin, .. })
            | (
                ValueActionLoopKind::SetBuilder
                | ValueActionLoopKind::SetFilter
                | ValueActionLoopKind::FuncDef,
                Opcode::LoopNext { loop_begin, .. },
            ) => loop_begin,
            _ => {
                return Err(format!(
                    "Value-action VM {kind:?} loop at pc {begin_pc} has no matching terminator at pc {next_pc}"
                ));
            }
        };
        let body_target = checked_action_loop_target(next_pc, loop_begin, len).ok_or_else(|| {
            format!(
                "Value-action VM {kind:?} terminator at pc {next_pc} has invalid loop_begin {loop_begin}"
            )
        })?;
        if body_target <= begin_pc || body_target >= end {
            return Err(format!(
                "Value-action VM {kind:?} terminator at pc {next_pc} targets pc {body_target} outside its body"
            ));
        }

        for (body_pc, body_op) in instructions
            .iter()
            .enumerate()
            .take(next_pc)
            .skip(begin_pc + 1)
        {
            let binding = match body_op {
                Opcode::StoreVar { .. } => "StoreVar",
                // UNCHANGED also mutates the Value action VM's bound-slot
                // overlay. Repeating it is no safer than repeating StoreVar.
                Opcode::Unchanged { .. } => "Unchanged",
                _ => continue,
            };
            return Err(format!(
                "Value-action VM rejects {binding} at pc {body_pc} inside residual {kind:?} loop body (pc {begin_pc}..{next_pc})"
            ));
        }
    }
    Ok(())
}

fn checked_action_loop_target(pc: usize, offset: i32, len: usize) -> Option<usize> {
    let target = (pc as i64).checked_add(i64::from(offset))?;
    let target = usize::try_from(target).ok()?;
    (target <= len).then_some(target)
}

fn validate_full_successor_binding_on_enabled_returns(
    instructions: &[Opcode],
    constants: &tla_tir::bytecode::ConstantPool,
    state_var_count: usize,
) -> Result<(), String> {
    if instructions.is_empty() {
        // An empty zero-arity function implicitly returns the register file's
        // initial FALSE value, so it cannot emit a successor.
        return Ok(());
    }

    let reachable = reachable_instruction_pcs(instructions)?;
    let last_pc = instructions.len() - 1;
    if reachable.contains(&last_pc)
        && !matches!(instructions[last_pc], Opcode::Ret { .. } | Opcode::Halt)
    {
        return Err(format!(
            "Value-action VM entry has a reachable implicit return after pc {last_pc}"
        ));
    }

    let mut seen = FxHashSet::default();
    let mut pending = vec![MustWritePathState {
        pc: 0,
        written_vars: BTreeSet::new(),
        known_bools: KnownBools::default(),
    }];

    while let Some(state) = pending.pop() {
        if state.pc >= instructions.len() || !seen.insert(state.clone()) {
            continue;
        }
        let pc = state.pc;
        if let Opcode::Ret { rs } = instructions[pc] {
            if state.known_bools.get(rs) != Some(false) {
                let missing: Vec<usize> = (0..state_var_count)
                    .filter(|var_idx| {
                        let var_idx = u16::try_from(*var_idx)
                            .expect("state_var_count was checked against the u16 slot space");
                        !state.written_vars.contains(&var_idx)
                    })
                    .take(8)
                    .collect();
                if !missing.is_empty() {
                    return Err(format!(
                        "Value-action VM enabled return at pc {pc} can leave successor variables {missing:?} unbound"
                    ));
                }
            }
            continue;
        }
        pending.extend(must_write_instruction_successors(
            instructions,
            state,
            constants,
        )?);
    }

    Ok(())
}

fn validate_entry_shape(
    instructions: &[Opcode],
    constants: &tla_tir::bytecode::ConstantPool,
    state_var_count: usize,
) -> Result<(), String> {
    let mut stored_vars = BTreeSet::new();
    let mut stored_var_writes: BTreeMap<u16, Vec<usize>> = BTreeMap::new();
    let mut unchanged_vars = BTreeSet::new();
    let mut unchanged_var_proofs: BTreeMap<u16, Vec<usize>> = BTreeMap::new();
    let provable_unchanged = provably_true_unchanged_pcs(instructions, constants);

    for (pc, op) in instructions.iter().enumerate() {
        match *op {
            Opcode::LoadPrime { .. } => {}
            Opcode::RoundStepEq { .. } => {
                return Err(format!(
                    "VM-only RoundStepEq remains in action entry at pc {pc}"
                ));
            }
            Opcode::SetPrimeMode { .. } => {
                return Err(format!(
                    "SetPrimeMode remains after action rewrite at pc {pc}"
                ));
            }
            Opcode::StoreVar { var_idx, .. } => {
                if store_var_can_repeat_on_same_path(instructions, &provable_unchanged, pc)
                    || duplicate_write_on_same_path(
                        instructions,
                        &stored_var_writes,
                        var_idx,
                        &provable_unchanged,
                        pc,
                    )
                {
                    // Include the listing so the fail-closed reason is
                    // actionable (which StoreVars share a path / repeat in a
                    // loop, and what produced them).
                    let listing: Vec<String> = instructions
                        .iter()
                        .enumerate()
                        .map(|(p, op)| format!("pc {p}: {op:?}"))
                        .collect();
                    return Err(format!(
                        "duplicate writes to primed var {var_idx} (StoreVar at pc {pc}, \
                         loop-repeat={}). body: [{}]",
                        store_var_can_repeat_on_same_path(instructions, &provable_unchanged, pc),
                        listing.join("; ")
                    ));
                }
                stored_var_writes.entry(var_idx).or_default().push(pc);
                stored_vars.insert(var_idx);
            }
            Opcode::Unchanged { start, count, .. } => {
                for var_idx in unchanged_var_indices(constants, start, count, pc)? {
                    unchanged_vars.insert(var_idx);
                    unchanged_var_proofs.entry(var_idx).or_default().push(pc);
                }
            }
            _ => {}
        }
    }

    for var_idx in stored_vars.intersection(&unchanged_vars) {
        let store_pcs = &stored_var_writes[var_idx];
        let unchanged_pcs = &unchanged_var_proofs[var_idx];
        if store_pcs.iter().any(|&store_pc| {
            unchanged_pcs.iter().any(|&unchanged_pc| {
                writes_can_share_path(instructions, &provable_unchanged, store_pc, unchanged_pc)
            })
        }) {
            return Err(format!(
                "primed var {var_idx} is both written and UNCHANGED"
            ));
        }
    }

    if stored_vars.is_empty()
        && !unchanged_vars.is_empty()
        && unchanged_vars.len() != state_var_count
    {
        return Err(format!(
            "UNCHANGED-only action covers {} of {state_var_count} state variables",
            unchanged_vars.len()
        ));
    }

    validate_residual_load_primes_after_must_write(instructions, constants)?;

    Ok(())
}

fn unchanged_var_indices(
    constants: &tla_tir::bytecode::ConstantPool,
    start: u16,
    count: u8,
    pc: usize,
) -> Result<Vec<u16>, String> {
    let mut vars = Vec::with_capacity(count as usize);
    for offset in 0..count as u16 {
        let value = constants.get_value(start + offset);
        let tla_value::Value::SmallInt(raw_var_idx) = value else {
            return Err(format!(
                "Unchanged metadata at pc {pc} does not decode to SmallInt var indices"
            ));
        };
        let var_idx = u16::try_from(*raw_var_idx).map_err(|_| {
            format!("Unchanged metadata at pc {pc} has out-of-range var index {raw_var_idx}")
        })?;
        vars.push(var_idx);
    }
    Ok(vars)
}

fn duplicate_write_on_same_path(
    instructions: &[Opcode],
    stored_var_writes: &BTreeMap<u16, Vec<usize>>,
    var_idx: u16,
    provable_unchanged: &FxHashSet<usize>,
    write_pc: usize,
) -> bool {
    stored_var_writes.get(&var_idx).is_some_and(|writes| {
        writes
            .iter()
            .any(|&pc| writes_can_share_path(instructions, provable_unchanged, pc, write_pc))
    })
}

fn store_var_can_repeat_on_same_path(
    instructions: &[Opcode],
    provable_unchanged: &FxHashSet<usize>,
    write_pc: usize,
) -> bool {
    can_execute_after_write(instructions, provable_unchanged, write_pc, write_pc)
}

fn writes_can_share_path(
    instructions: &[Opcode],
    provable_unchanged: &FxHashSet<usize>,
    a_pc: usize,
    b_pc: usize,
) -> bool {
    can_execute_after_write(instructions, provable_unchanged, a_pc, b_pc)
        || can_execute_after_write(instructions, provable_unchanged, b_pc, a_pc)
}

/// PCs of `Unchanged` opcodes whose boolean result is PROVABLY true in the
/// transformed next-state generation context: the successor is seeded from the
/// current state before the action executes, so `UNCHANGED <vars>` over
/// variables that no `StoreVar` in this function ever writes always compares
/// equal. Used to refine the boolean path analysis — e.g. an exists loop whose
/// body truth value routes through such an `Unchanged` (PaxosCommit Phase1b's
/// `UNCHANGED rmState` trailing conjunct) provably terminates at `ExistsNext`
/// on the successful-write path, so the loop back-edge cannot re-execute the
/// primed `StoreVar`. Resolution failures simply leave the pc out (the result
/// stays unknown — conservative).
fn provably_true_unchanged_pcs(
    instructions: &[Opcode],
    constants: &tla_tir::bytecode::ConstantPool,
) -> FxHashSet<usize> {
    let stored: BTreeSet<u16> = instructions
        .iter()
        .filter_map(|op| match op {
            Opcode::StoreVar { var_idx, .. } => Some(*var_idx),
            _ => None,
        })
        .collect();
    let mut out = FxHashSet::default();
    for (pc, op) in instructions.iter().enumerate() {
        let Opcode::Unchanged { start, count, .. } = *op else {
            continue;
        };
        let Ok(vars) = unchanged_var_indices(constants, start, count, pc) else {
            continue;
        };
        if vars.iter().all(|v| !stored.contains(v)) {
            out.insert(pc);
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BoolPathState {
    pc: usize,
    seen_first_write: bool,
    known_bools: KnownBools,
}

fn can_execute_after_write(
    instructions: &[Opcode],
    provable_unchanged: &FxHashSet<usize>,
    first_write: usize,
    later_write: usize,
) -> bool {
    if first_write >= instructions.len() || later_write >= instructions.len() {
        return false;
    }

    let mut seen = FxHashSet::default();
    let mut pending = vec![BoolPathState {
        pc: 0,
        seen_first_write: false,
        known_bools: KnownBools::default(),
    }];

    while let Some(mut state) = pending.pop() {
        if state.pc >= instructions.len() {
            continue;
        }
        if state.seen_first_write && state.pc == later_write {
            return true;
        }
        if state.pc == first_write {
            state.seen_first_write = true;
        }
        if !seen.insert(state) {
            continue;
        }

        let Some(successors) = instruction_successors(instructions, provable_unchanged, state)
        else {
            return true;
        };
        pending.extend(successors);
    }

    false
}

fn instruction_successors(
    instructions: &[Opcode],
    provable_unchanged: &FxHashSet<usize>,
    mut state: BoolPathState,
) -> Option<Vec<BoolPathState>> {
    let len = instructions.len();
    let pc = state.pc;
    let mut successors = Vec::with_capacity(2);
    match instructions[pc] {
        Opcode::Jump { offset } => {
            successors.push(next_state(state, jump_target(pc, offset, len)?));
        }
        Opcode::JumpTrue { .. } | Opcode::JumpFalse { .. } => {
            push_conditional_successors(&mut successors, state, instructions[pc], len)?;
        }
        Opcode::ForallBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, Some(true));
            state.known_bools.set(r_binding, None);
            successors.push(next_state(state, jump_target(pc, loop_end, len)?));
            push_fallthrough(&mut successors, state, len);
        }
        Opcode::ExistsBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::ChooseBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, Some(false));
            state.known_bools.set(r_binding, None);
            successors.push(next_state(state, jump_target(pc, loop_end, len)?));
            push_fallthrough(&mut successors, state, len);
        }
        Opcode::SetFilterBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::SetBuilderBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::FuncDefBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, None);
            state.known_bools.set(r_binding, None);
            successors.push(next_state(state, jump_target(pc, loop_end, len)?));
            push_fallthrough(&mut successors, state, len);
        }
        Opcode::ForallNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(false) => {
                state.known_bools.set(rd, Some(false));
                push_fallthrough(&mut successors, state, len);
            }
            Some(true) => {
                state.known_bools.set(rd, Some(true));
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
            None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::ExistsNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(true) => {
                state.known_bools.set(rd, Some(true));
                push_fallthrough(&mut successors, state, len);
            }
            Some(false) => {
                state.known_bools.set(rd, Some(false));
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
            None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::ChooseNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(true) => {
                state.known_bools.set(rd, None);
                push_fallthrough(&mut successors, state, len);
            }
            Some(false) | None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::LoopNext {
            r_binding,
            loop_begin,
            ..
        } => {
            state.known_bools.set(r_binding, None);
            successors.push(next_state(state, jump_target(pc, loop_begin, len)?));
            push_fallthrough(&mut successors, state, len);
        }
        Opcode::Ret { .. } | Opcode::Halt => {}
        // In the generation context an `UNCHANGED <vars>` over never-stored
        // variables is provably true (see provably_true_unchanged_pcs); other
        // `Unchanged` results stay unknown.
        Opcode::Unchanged { rd, .. } => {
            state.known_bools.set(
                rd,
                if provable_unchanged.contains(&pc) {
                    Some(true)
                } else {
                    None
                },
            );
            push_fallthrough(&mut successors, state, len);
        }
        _ => {
            transfer_bool_facts(&mut state.known_bools, instructions[pc]);
            push_fallthrough(&mut successors, state, len);
        }
    }
    Some(successors)
}

fn push_conditional_successors(
    successors: &mut Vec<BoolPathState>,
    state: BoolPathState,
    op: Opcode,
    len: usize,
) -> Option<()> {
    let (rs, offset, jump_on) = match op {
        Opcode::JumpTrue { rs, offset } => (rs, offset, true),
        Opcode::JumpFalse { rs, offset } => (rs, offset, false),
        _ => unreachable!("conditional successor helper called for non-branch"),
    };
    let pc = state.pc;
    match state.known_bools.get(rs) {
        Some(value) if value == jump_on => {
            successors.push(next_state_with_bool_fact(
                state,
                jump_target(pc, offset, len)?,
                rs,
                jump_on,
            ));
        }
        Some(_) => push_fallthrough_with_bool_fact(successors, state, len, rs, !jump_on),
        None => {
            successors.push(next_state_with_bool_fact(
                state,
                jump_target(pc, offset, len)?,
                rs,
                jump_on,
            ));
            push_fallthrough_with_bool_fact(successors, state, len, rs, !jump_on);
        }
    }
    Some(())
}

fn transfer_bool_facts(known: &mut KnownBools, op: Opcode) {
    match op {
        Opcode::LoadBool { rd, value } => known.set(rd, Some(value)),
        Opcode::Move { rd, rs } => known.set(rd, known.get(rs)),
        Opcode::Not { rd, rs } => known.set(rd, known.get(rs).map(|value| !value)),
        Opcode::And { rd, r1, r2 } => {
            known.set(
                rd,
                match (known.get(r1), known.get(r2)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
            );
        }
        Opcode::Or { rd, r1, r2 } => {
            known.set(
                rd,
                match (known.get(r1), known.get(r2)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
            );
        }
        Opcode::Implies { rd, r1, r2 } => {
            known.set(
                rd,
                match (known.get(r1), known.get(r2)) {
                    (Some(false), _) | (_, Some(true)) => Some(true),
                    (Some(true), Some(false)) => Some(false),
                    _ => None,
                },
            );
        }
        Opcode::Equiv { rd, r1, r2 } => {
            known.set(rd, known.get(r1).zip(known.get(r2)).map(|(a, b)| a == b));
        }
        Opcode::Eq { rd, r1, r2 } if r1 == r2 => known.set(rd, Some(true)),
        _ => {
            if let Some(rd) = op.dest_register() {
                known.set(rd, None);
            }
            if let Some(r_binding) = op.binding_register() {
                known.set(r_binding, None);
            }
        }
    }
}

fn push_fallthrough(successors: &mut Vec<BoolPathState>, state: BoolPathState, len: usize) {
    if state.pc + 1 < len {
        let next_pc = state.pc + 1;
        successors.push(next_state(state, next_pc));
    }
}

fn push_fallthrough_with_bool_fact(
    successors: &mut Vec<BoolPathState>,
    state: BoolPathState,
    len: usize,
    reg: u8,
    value: bool,
) {
    if state.pc + 1 < len {
        let next_pc = state.pc + 1;
        successors.push(next_state_with_bool_fact(state, next_pc, reg, value));
    }
}

fn next_state(mut state: BoolPathState, pc: usize) -> BoolPathState {
    state.pc = pc;
    state
}

fn next_state_with_bool_fact(
    mut state: BoolPathState,
    pc: usize,
    reg: u8,
    value: bool,
) -> BoolPathState {
    state.known_bools.set(reg, Some(value));
    state.pc = pc;
    state
}

fn jump_target(pc: usize, offset: i32, len: usize) -> Option<usize> {
    let target = (pc as i64).checked_add(i64::from(offset))?;
    let target = usize::try_from(target).ok()?;
    (target < len).then_some(target)
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MustWritePathState {
    pc: usize,
    written_vars: BTreeSet<u16>,
    known_bools: KnownBools,
}

fn validate_residual_load_primes_after_must_write(
    instructions: &[Opcode],
    constants: &tla_tir::bytecode::ConstantPool,
) -> Result<(), String> {
    let mut seen = FxHashSet::default();
    let mut pending = vec![MustWritePathState {
        pc: 0,
        written_vars: BTreeSet::new(),
        known_bools: KnownBools::default(),
    }];

    while let Some(state) = pending.pop() {
        if state.pc >= instructions.len() {
            continue;
        }
        if !seen.insert(state.clone()) {
            continue;
        }

        let pc = state.pc;
        if let Opcode::LoadPrime { var_idx, .. } = instructions[pc] {
            if !state.written_vars.contains(&var_idx) {
                return Err(format!(
                    "residual LoadPrime for primed var {var_idx} at pc {pc} has no definite prior StoreVar/UNCHANGED proof"
                ));
            }
        }

        let successors = must_write_instruction_successors(instructions, state, constants)?;
        pending.extend(successors);
    }

    Ok(())
}

fn must_write_instruction_successors(
    instructions: &[Opcode],
    mut state: MustWritePathState,
    constants: &tla_tir::bytecode::ConstantPool,
) -> Result<Vec<MustWritePathState>, String> {
    let len = instructions.len();
    let pc = state.pc;
    let mut successors = Vec::with_capacity(2);
    match instructions[pc] {
        Opcode::Jump { offset } => {
            let target = jump_target(pc, offset, len)
                .ok_or_else(|| format!("invalid jump target from pc {pc} with offset {offset}"))?;
            successors.push(next_must_write_state(state, target));
        }
        Opcode::JumpTrue { .. } | Opcode::JumpFalse { .. } => {
            push_must_write_conditional_successors(&mut successors, state, instructions[pc], len)?;
        }
        Opcode::ForallBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, Some(true));
            state.known_bools.set(r_binding, None);
            let target = jump_target(pc, loop_end, len).ok_or_else(|| {
                format!("invalid ForallBegin loop_end from pc {pc} with offset {loop_end}")
            })?;
            successors.push(next_must_write_state(state.clone(), target));
            push_must_write_fallthrough(&mut successors, state, len);
        }
        Opcode::ExistsBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::ChooseBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, Some(false));
            state.known_bools.set(r_binding, None);
            let target = jump_target(pc, loop_end, len).ok_or_else(|| {
                format!("invalid quantifier loop_end from pc {pc} with offset {loop_end}")
            })?;
            successors.push(next_must_write_state(state.clone(), target));
            push_must_write_fallthrough(&mut successors, state, len);
        }
        Opcode::SetFilterBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::SetBuilderBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::FuncDefBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, None);
            state.known_bools.set(r_binding, None);
            let target = jump_target(pc, loop_end, len).ok_or_else(|| {
                format!("invalid builder loop_end from pc {pc} with offset {loop_end}")
            })?;
            successors.push(next_must_write_state(state.clone(), target));
            push_must_write_fallthrough(&mut successors, state, len);
        }
        Opcode::ForallNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(false) => {
                state.known_bools.set(rd, Some(false));
                push_must_write_fallthrough(&mut successors, state, len);
            }
            Some(true) => {
                state.known_bools.set(rd, Some(true));
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ForallNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
            None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ForallNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::ExistsNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(true) => {
                state.known_bools.set(rd, Some(true));
                push_must_write_fallthrough(&mut successors, state, len);
            }
            Some(false) => {
                state.known_bools.set(rd, Some(false));
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ExistsNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
            None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ExistsNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::ChooseNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(true) => {
                state.known_bools.set(rd, None);
                push_must_write_fallthrough(&mut successors, state, len);
            }
            Some(false) | None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ChooseNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::LoopNext {
            r_binding,
            loop_begin,
            ..
        } => {
            state.known_bools.set(r_binding, None);
            let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                format!("invalid LoopNext loop_begin from pc {pc} with offset {loop_begin}")
            })?;
            successors.push(next_must_write_state(state.clone(), target));
            push_must_write_fallthrough(&mut successors, state, len);
        }
        Opcode::Ret { .. } | Opcode::Halt => {}
        op => {
            apply_must_write_effect(&mut state, op, constants, pc)?;
            push_must_write_fallthrough(&mut successors, state, len);
        }
    }
    Ok(successors)
}

fn apply_must_write_effect(
    state: &mut MustWritePathState,
    op: Opcode,
    constants: &tla_tir::bytecode::ConstantPool,
    pc: usize,
) -> Result<(), String> {
    match op {
        Opcode::StoreVar { var_idx, .. } => {
            state.written_vars.insert(var_idx);
        }
        Opcode::Unchanged { start, count, .. } => {
            for var_idx in unchanged_var_indices(constants, start, count, pc)? {
                state.written_vars.insert(var_idx);
            }
        }
        _ => {}
    }
    transfer_bool_facts(&mut state.known_bools, op);
    Ok(())
}

fn push_must_write_conditional_successors(
    successors: &mut Vec<MustWritePathState>,
    state: MustWritePathState,
    op: Opcode,
    len: usize,
) -> Result<(), String> {
    let (rs, offset, jump_on) = match op {
        Opcode::JumpTrue { rs, offset } => (rs, offset, true),
        Opcode::JumpFalse { rs, offset } => (rs, offset, false),
        _ => unreachable!("conditional successor helper called for non-branch"),
    };
    let pc = state.pc;
    match state.known_bools.get(rs) {
        Some(value) if value == jump_on => {
            let target = jump_target(pc, offset, len).ok_or_else(|| {
                format!("invalid conditional jump from pc {pc} with offset {offset}")
            })?;
            successors.push(next_must_write_state_with_bool_fact(
                state, target, rs, jump_on,
            ));
        }
        Some(_) => push_must_write_fallthrough_with_bool_fact(successors, state, len, rs, !jump_on),
        None => {
            let target = jump_target(pc, offset, len).ok_or_else(|| {
                format!("invalid conditional jump from pc {pc} with offset {offset}")
            })?;
            successors.push(next_must_write_state_with_bool_fact(
                state.clone(),
                target,
                rs,
                jump_on,
            ));
            push_must_write_fallthrough_with_bool_fact(successors, state, len, rs, !jump_on);
        }
    }
    Ok(())
}

fn push_must_write_fallthrough(
    successors: &mut Vec<MustWritePathState>,
    state: MustWritePathState,
    len: usize,
) {
    if state.pc + 1 < len {
        let next_pc = state.pc + 1;
        successors.push(next_must_write_state(state, next_pc));
    }
}

fn push_must_write_fallthrough_with_bool_fact(
    successors: &mut Vec<MustWritePathState>,
    state: MustWritePathState,
    len: usize,
    reg: u8,
    value: bool,
) {
    if state.pc + 1 < len {
        let next_pc = state.pc + 1;
        successors.push(next_must_write_state_with_bool_fact(
            state, next_pc, reg, value,
        ));
    }
}

fn next_must_write_state(mut state: MustWritePathState, pc: usize) -> MustWritePathState {
    state.pc = pc;
    state
}

fn next_must_write_state_with_bool_fact(
    mut state: MustWritePathState,
    pc: usize,
    reg: u8,
    value: bool,
) -> MustWritePathState {
    state.known_bools.set(reg, Some(value));
    state.pc = pc;
    state
}

fn validate_reachable_callees(
    entry_func_idx: u16,
    entry_instructions: &[Opcode],
    chunk: &BytecodeChunk,
) -> Result<(), String> {
    let mut visited = FxHashSet::default();
    let mut pending = collect_reachable_callees(entry_instructions)?;

    while let Some(func_idx) = pending.pop() {
        if !visited.insert(func_idx) {
            continue;
        }

        let callee = chunk.functions.get(func_idx as usize).ok_or_else(|| {
            format!("entry action {entry_func_idx} references missing callee {func_idx}")
        })?;

        validate_helper_shape(func_idx, callee)?;
        pending.extend(collect_reachable_callees(&callee.instructions)?);
    }

    Ok(())
}

fn collect_reachable_callees(instructions: &[Opcode]) -> Result<Vec<u16>, String> {
    let reachable = reachable_instruction_pcs(instructions)?;
    Ok(reachable
        .into_iter()
        .filter_map(|pc| match instructions[pc] {
            Opcode::Call { op_idx, .. } => Some(op_idx),
            _ => None,
        })
        .collect())
}

pub(super) fn reachable_instruction_pcs(
    instructions: &[Opcode],
) -> Result<BTreeSet<usize>, String> {
    let mut reachable = BTreeSet::new();
    let mut seen = FxHashSet::default();
    let mut pending = vec![BoolPathState {
        pc: 0,
        seen_first_write: false,
        known_bools: KnownBools::default(),
    }];

    while let Some(state) = pending.pop() {
        if state.pc >= instructions.len() {
            continue;
        }
        if !seen.insert(state) {
            continue;
        }
        reachable.insert(state.pc);
        let pc = state.pc;
        // Pure reachability scan: no generation-context Unchanged provability
        // applies here (empty set keeps the result unknown, as before).
        let successors = instruction_successors(instructions, &FxHashSet::default(), state)
            .ok_or_else(|| format!("invalid control-flow target while scanning pc {pc}"))?;
        pending.extend(successors);
    }

    Ok(reachable)
}

fn validate_helper_shape(func_idx: u16, func: &BytecodeFunction) -> Result<(), String> {
    let reachable = reachable_instruction_pcs(&func.instructions)?;
    for pc in reachable {
        let op = func.instructions[pc];
        match op {
            Opcode::LoadPrime { .. } => {
                return Err(format!(
                    "reachable callee {func_idx} contains LoadPrime at pc {pc}"
                ));
            }
            Opcode::SetPrimeMode { .. } => {
                return Err(format!(
                    "reachable callee {func_idx} contains SetPrimeMode at pc {pc}"
                ));
            }
            Opcode::RoundStepEq { .. } => {
                return Err(format!(
                    "reachable callee {func_idx} contains VM-only RoundStepEq at pc {pc}"
                ));
            }
            Opcode::StoreVar { .. } => {
                return Err(format!(
                    "reachable callee {func_idx} writes successor state at pc {pc}"
                ));
            }
            Opcode::Unchanged { .. } => {
                return Err(format!(
                    "reachable callee {func_idx} contains Unchanged next-state check at pc {pc}"
                ));
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_tir::bytecode::BytecodeFunction;

    fn make_entry_with_call(op_idx: u16) -> Vec<Opcode> {
        vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::Call {
                rd: 1,
                op_idx,
                args_start: 0,
                argc: 0,
            },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::Ret { rs: 1 },
        ]
    }

    #[test]
    fn test_rejects_reachable_callee_with_load_prime() {
        let entry = make_entry_with_call(1);
        let mut helper = BytecodeFunction::new("BadPrimeHelper".to_string(), 0);
        helper.emit(Opcode::LoadPrime { rd: 0, var_idx: 1 });
        helper.emit(Opcode::Ret { rs: 0 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        chunk.add_function(helper);

        let reason =
            validate_next_state_action_chunk(0, &entry, &chunk, 2).expect_err("helper must reject");
        assert!(reason.contains("reachable callee 1 contains LoadPrime"));
    }

    #[test]
    fn test_rejects_round_step_eq_in_entry_and_reachable_callee() {
        let vm_only = Opcode::RoundStepEq {
            rd: 2,
            child: 0,
            parent: 1,
        };
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));

        let reason = validate_next_state_action_chunk(0, &[vm_only], &chunk, 1)
            .expect_err("VM-only RoundStepEq must reject in action entries");
        assert!(reason.contains("VM-only RoundStepEq remains in action entry"));

        let entry = make_entry_with_call(1);
        let mut helper = BytecodeFunction::new("BadRoundStepHelper".to_string(), 2);
        helper.emit(vm_only);
        helper.emit(Opcode::Ret { rs: 2 });
        chunk.add_function(helper);
        let reason = validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect_err("VM-only RoundStepEq must reject in reachable helpers");
        assert!(reason.contains("reachable callee 1 contains VM-only RoundStepEq"));
    }

    #[test]
    fn test_ignores_unreachable_callee_with_load_prime() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let mut helper = BytecodeFunction::new("DeadPrimeHelper".to_string(), 0);
        helper.emit(Opcode::LoadPrime { rd: 0, var_idx: 1 });
        helper.emit(Opcode::Ret { rs: 0 });
        chunk.add_function(helper);

        let entry = vec![
            Opcode::LoadBool { rd: 0, value: true },
            Opcode::JumpTrue { rs: 0, offset: 3 },
            Opcode::Call {
                rd: 1,
                op_idx: 1,
                args_start: 0,
                argc: 0,
            },
            Opcode::Ret { rs: 1 },
            Opcode::LoadImm { rd: 2, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 2 },
            Opcode::Ret { rs: 0 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 2)
            .expect("unreachable helper calls should not poison safe actions");
    }

    #[test]
    fn test_ignores_unreachable_load_prime_inside_reachable_callee() {
        let entry = make_entry_with_call(1);
        let mut helper = BytecodeFunction::new("DeadPrimeInsideHelper".to_string(), 0);
        helper.emit(Opcode::LoadBool { rd: 0, value: true });
        helper.emit(Opcode::JumpTrue { rs: 0, offset: 3 });
        helper.emit(Opcode::LoadPrime { rd: 1, var_idx: 1 });
        helper.emit(Opcode::Ret { rs: 1 });
        helper.emit(Opcode::LoadImm { rd: 2, value: 1 });
        helper.emit(Opcode::Ret { rs: 2 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        chunk.add_function(helper);

        validate_next_state_action_chunk(0, &entry, &chunk, 2)
            .expect("dead primed-state code inside helpers should not poison safe actions");
    }

    #[test]
    fn test_rejects_reachable_callee_with_set_prime_mode() {
        let entry = make_entry_with_call(1);
        let mut helper = BytecodeFunction::new("BadPrimeModeHelper".to_string(), 0);
        helper.emit(Opcode::SetPrimeMode { enable: true });
        helper.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
        helper.emit(Opcode::SetPrimeMode { enable: false });
        helper.emit(Opcode::Ret { rs: 0 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        chunk.add_function(helper);

        let reason =
            validate_next_state_action_chunk(0, &entry, &chunk, 1).expect_err("helper must reject");
        assert!(reason.contains("reachable callee 1 contains SetPrimeMode"));
    }

    #[test]
    fn test_accepts_pure_reachable_callee() {
        let entry = make_entry_with_call(1);
        let mut helper = BytecodeFunction::new("PureHelper".to_string(), 0);
        helper.emit(Opcode::LoadVar { rd: 0, var_idx: 1 });
        helper.emit(Opcode::LoadImm { rd: 1, value: 1 });
        helper.emit(Opcode::AddInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        helper.emit(Opcode::Ret { rs: 2 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        chunk.add_function(helper);

        validate_next_state_action_chunk(0, &entry, &chunk, 2)
            .expect("pure helper should remain eligible");
    }

    #[test]
    fn test_accepts_unchanged_only_full_state_action() {
        let mut chunk = BytecodeChunk::new();
        let start = chunk.constants.add_value(tla_value::Value::SmallInt(0));
        chunk.constants.add_value(tla_value::Value::SmallInt(1));
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::Unchanged {
                rd: 0,
                start,
                count: 2,
            },
            Opcode::Ret { rs: 0 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 2)
            .expect("full-state UNCHANGED action should be executable");
    }

    #[test]
    fn test_accepts_residual_load_prime_after_store_var() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::LoadPrime { rd: 1, var_idx: 0 },
            Opcode::Ret { rs: 1 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect("residual LoadPrime after StoreVar proof should be executable");
    }

    #[test]
    fn test_accepts_residual_load_prime_after_unchanged() {
        let mut chunk = BytecodeChunk::new();
        let start = chunk.constants.add_value(tla_value::Value::SmallInt(0));
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::Unchanged {
                rd: 0,
                start,
                count: 1,
            },
            Opcode::LoadPrime { rd: 1, var_idx: 0 },
            Opcode::Ret { rs: 1 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect("residual LoadPrime after UNCHANGED proof should be executable");
    }

    #[test]
    fn test_accepts_residual_load_prime_after_guarded_store() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::JumpFalse { rs: 0, offset: 13 },
            Opcode::LoadVar { rd: 1, var_idx: 1 },
            Opcode::Move { rd: 2, rs: 1 },
            Opcode::JumpFalse { rs: 2, offset: 10 },
            Opcode::LoadImm { rd: 3, value: 10 },
            Opcode::LoadBool { rd: 5, value: true },
            Opcode::StoreVar { var_idx: 2, rs: 3 },
            Opcode::Move { rd: 6, rs: 5 },
            Opcode::JumpFalse { rs: 6, offset: 5 },
            Opcode::LoadPrime { rd: 7, var_idx: 2 },
            Opcode::LoadBool { rd: 9, value: true },
            Opcode::StoreVar { var_idx: 3, rs: 7 },
            Opcode::Ret { rs: 9 },
            Opcode::Ret { rs: 0 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 4)
            .expect("branch-refined guard should prove residual LoadPrime is post-store");
    }

    #[test]
    fn test_rejects_residual_load_prime_without_prior_must_write() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::Ret { rs: 0 },
        ];

        let reason = validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect_err("residual LoadPrime before StoreVar proof must be rejected");
        assert!(reason.contains("residual LoadPrime for primed var 0 at pc 0"));
        assert!(reason.contains("no definite prior StoreVar/UNCHANGED proof"));
    }

    #[test]
    fn test_rejects_residual_load_prime_after_partial_branch_write() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::JumpFalse { rs: 0, offset: 3 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::LoadPrime { rd: 2, var_idx: 0 },
            Opcode::Ret { rs: 2 },
        ];

        let reason = validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect_err("branch-partial StoreVar proof must not allow residual LoadPrime");
        assert!(reason.contains("residual LoadPrime for primed var 0 at pc 4"));
    }

    #[test]
    fn test_accepts_branch_exclusive_duplicate_store() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadBool { rd: 0, value: true },
            Opcode::JumpFalse { rs: 0, offset: 4 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::Jump { offset: 3 },
            Opcode::LoadImm { rd: 2, value: 2 },
            Opcode::StoreVar { var_idx: 0, rs: 2 },
            Opcode::Ret { rs: 0 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect("branch-exclusive duplicate writes should be executable");
    }

    #[test]
    fn test_accepts_branch_exclusive_store_and_unchanged() {
        let mut chunk = BytecodeChunk::new();
        let start = chunk.constants.add_value(tla_value::Value::SmallInt(0));
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::JumpFalse { rs: 0, offset: 4 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::Jump { offset: 2 },
            Opcode::Unchanged {
                rd: 2,
                start,
                count: 1,
            },
            Opcode::Ret { rs: 0 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect("branch-exclusive StoreVar/UNCHANGED should be executable");
    }

    #[test]
    fn test_rejects_same_path_duplicate_store() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::LoadImm { rd: 1, value: 2 },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::Ret { rs: 0 },
        ];

        let reason = validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect_err("same-path duplicate writes must stay rejected");
        assert!(reason.contains("duplicate writes"));
    }

    #[test]
    fn test_rejects_same_path_store_and_unchanged() {
        let mut chunk = BytecodeChunk::new();
        let start = chunk.constants.add_value(tla_value::Value::SmallInt(0));
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::Unchanged {
                rd: 1,
                start,
                count: 1,
            },
            Opcode::Ret { rs: 1 },
        ];

        let reason = validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect_err("same-path StoreVar/UNCHANGED must stay rejected");
        assert!(reason.contains("primed var 0 is both written and UNCHANGED"));
    }

    #[test]
    fn test_accepts_exists_loop_duplicate_store_when_body_is_true() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::ExistsBegin {
                rd: 10,
                r_binding: 11,
                r_domain: 12,
                loop_end: 10,
            },
            Opcode::JumpFalse { rs: 0, offset: 5 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::LoadBool { rd: 9, value: true },
            Opcode::Jump { offset: 4 },
            Opcode::LoadImm { rd: 2, value: 2 },
            Opcode::StoreVar { var_idx: 0, rs: 2 },
            Opcode::LoadBool { rd: 9, value: true },
            Opcode::ExistsNext {
                rd: 10,
                r_binding: 11,
                r_body: 9,
                loop_begin: -8,
            },
            Opcode::Ret { rs: 10 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect("a provably-true EXISTS body cannot loop to a second write");
    }

    #[test]
    fn test_rejects_exists_loop_duplicate_store_when_body_can_be_false() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::ExistsBegin {
                rd: 10,
                r_binding: 11,
                r_domain: 12,
                loop_end: 10,
            },
            Opcode::JumpFalse { rs: 0, offset: 5 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::LoadBool {
                rd: 9,
                value: false,
            },
            Opcode::Jump { offset: 4 },
            Opcode::LoadImm { rd: 2, value: 2 },
            Opcode::StoreVar { var_idx: 0, rs: 2 },
            Opcode::LoadBool {
                rd: 9,
                value: false,
            },
            Opcode::ExistsNext {
                rd: 10,
                r_binding: 11,
                r_body: 9,
                loop_begin: -8,
            },
            Opcode::Ret { rs: 10 },
        ];

        let reason = validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect_err("a false EXISTS body can loop to a second write");
        assert!(reason.contains("duplicate writes"));
    }

    #[test]
    fn test_rejects_unchanged_only_partial_state_action() {
        let mut chunk = BytecodeChunk::new();
        let start = chunk.constants.add_value(tla_value::Value::SmallInt(0));
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::Unchanged {
                rd: 0,
                start,
                count: 1,
            },
            Opcode::Ret { rs: 0 },
        ];

        let reason = validate_next_state_action_chunk(0, &entry, &chunk, 2)
            .expect_err("partial-frame UNCHANGED-only action must stay uncompiled");
        assert!(reason.contains("UNCHANGED-only action covers 1 of 2 state variables"));
    }

    #[test]
    fn test_value_action_vm_accepts_straight_line_store_and_unchanged() {
        let mut chunk = BytecodeChunk::new();
        let unchanged_start = chunk.constants.add_value(tla_value::Value::SmallInt(1));
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadImm { rd: 0, value: 7 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::Unchanged {
                rd: 1,
                start: unchanged_start,
                count: 1,
            },
            Opcode::Ret { rs: 1 },
        ];

        validate_value_action_vm_eligibility(0, &entry, &chunk, 2)
            .expect("straight-line complete successor bindings should be Value-VM eligible");
    }

    #[test]
    fn test_value_action_vm_rejects_store_lexically_inside_residual_exists() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::ExistsBegin {
                rd: 10,
                r_binding: 11,
                r_domain: 12,
                loop_end: 10,
            },
            Opcode::JumpFalse { rs: 0, offset: 5 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::LoadBool { rd: 9, value: true },
            Opcode::Jump { offset: 4 },
            Opcode::LoadImm { rd: 2, value: 2 },
            Opcode::StoreVar { var_idx: 0, rs: 2 },
            Opcode::LoadBool { rd: 9, value: true },
            Opcode::ExistsNext {
                rd: 10,
                r_binding: 11,
                r_body: 9,
                loop_begin: -8,
            },
            Opcode::Ret { rs: 10 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect("the shared backend validator intentionally keeps this admission");
        let reason = validate_value_action_vm_eligibility(0, &entry, &chunk, 1)
            .expect_err("a single-output Value VM must reject residual-loop writes");
        assert!(reason.contains("StoreVar at pc 4 inside residual Exists loop body"));
    }

    #[test]
    fn test_value_action_vm_accepts_guard_only_residual_loop() {
        let mut chunk = BytecodeChunk::new();
        let domain = chunk.constants.add_value(tla_value::Value::set([
            tla_value::Value::SmallInt(1),
            tla_value::Value::SmallInt(2),
        ]));
        let unchanged_start = chunk.constants.add_value(tla_value::Value::SmallInt(1));
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadConst { rd: 0, idx: domain },
            Opcode::ForallBegin {
                rd: 1,
                r_binding: 2,
                r_domain: 0,
                loop_end: 3,
            },
            Opcode::LoadBool { rd: 3, value: true },
            Opcode::ForallNext {
                rd: 1,
                r_binding: 2,
                r_body: 3,
                loop_begin: -1,
            },
            Opcode::LoadImm { rd: 4, value: 7 },
            Opcode::StoreVar { var_idx: 0, rs: 4 },
            Opcode::Unchanged {
                rd: 5,
                start: unchanged_start,
                count: 1,
            },
            Opcode::And {
                rd: 6,
                r1: 1,
                r2: 5,
            },
            Opcode::Ret { rs: 6 },
        ];

        validate_value_action_vm_eligibility(0, &entry, &chunk, 2)
            .expect("current-state guard loops with successor writes after the loop are safe");
    }

    #[test]
    fn test_value_action_vm_rejects_context_dependent_opcodes() {
        let mut chunk = BytecodeChunk::new();
        let name_idx = chunk
            .constants
            .add_value(tla_value::Value::string("External"));
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let unsupported = [
            (
                "CallExternal",
                Opcode::CallExternal {
                    rd: 7,
                    name_idx,
                    args_start: 0,
                    argc: 0,
                    self_recursive: false,
                },
            ),
            (
                "ValueApply",
                Opcode::ValueApply {
                    rd: 7,
                    func: 0,
                    args_start: 0,
                    argc: 0,
                },
            ),
            (
                "MakeClosure",
                Opcode::MakeClosure {
                    rd: 7,
                    template_idx: name_idx,
                    captures_start: 0,
                    capture_count: 0,
                },
            ),
            (
                "FuncDef",
                Opcode::FuncDef {
                    rd: 7,
                    r_domain: 0,
                    r_binding: 1,
                },
            ),
        ];

        for (name, opcode) in unsupported {
            let entry = vec![
                opcode,
                Opcode::LoadImm { rd: 0, value: 7 },
                Opcode::StoreVar { var_idx: 0, rs: 0 },
                Opcode::LoadBool { rd: 1, value: true },
                Opcode::Ret { rs: 1 },
            ];
            let reason = validate_value_action_vm_eligibility(0, &entry, &chunk, 1)
                .expect_err("context-dependent opcodes need an explicit action-mode proof");
            assert!(reason.contains(name), "unexpected rejection: {reason}");
        }
    }

    #[test]
    fn test_value_action_vm_scans_chunk_entry_reached_through_call_cycle() {
        let mut chunk_entry = BytecodeFunction::new("ChunkEntry".to_string(), 0);
        chunk_entry.emit(Opcode::Call {
            rd: 0,
            op_idx: 1,
            args_start: 0,
            argc: 0,
        });
        chunk_entry.emit(Opcode::ValueApply {
            rd: 1,
            func: 0,
            args_start: 0,
            argc: 0,
        });
        chunk_entry.emit(Opcode::Ret { rs: 1 });

        let mut helper = BytecodeFunction::new("Helper".to_string(), 0);
        helper.emit(Opcode::Call {
            rd: 0,
            op_idx: 0,
            args_start: 0,
            argc: 0,
        });
        helper.emit(Opcode::Ret { rs: 0 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(chunk_entry);
        chunk.add_function(helper);

        // The candidate transformed entry differs from the chunk-resident
        // function at index 0. Its helper edge cycles back to that index, so
        // eligibility must scan the resident body instead of treating the
        // entry index as already visited.
        let transformed_entry = vec![
            Opcode::Call {
                rd: 7,
                op_idx: 1,
                args_start: 0,
                argc: 0,
            },
            Opcode::LoadImm { rd: 0, value: 7 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::LoadBool { rd: 1, value: true },
            Opcode::Ret { rs: 1 },
        ];

        validate_next_state_action_chunk(0, &transformed_entry, &chunk, 1)
            .expect("the shared validator permits pure dynamic helper evaluation");
        let reason = validate_value_action_vm_eligibility(0, &transformed_entry, &chunk, 1)
            .expect_err("the chunk-resident entry reached by the cycle must be rescanned");
        assert!(reason.contains("reachable callee 0 contains unsupported ValueApply"));
    }

    #[test]
    fn test_certified_value_action_vm_choose_preserves_tlc_witness_order() {
        let mut chunk = BytecodeChunk::new();
        let tuple_len_two = chunk.constants.add_value(tla_value::Value::tuple([
            tla_value::Value::int(1),
            tla_value::Value::int(1),
        ]));
        let tuple_len_one = chunk
            .constants
            .add_value(tla_value::Value::tuple([tla_value::Value::int(2)]));

        let mut entry = BytecodeFunction::new("ChooseAction".to_string(), 0);
        entry.emit(Opcode::LoadConst {
            rd: 0,
            idx: tuple_len_two,
        });
        entry.emit(Opcode::LoadConst {
            rd: 1,
            idx: tuple_len_one,
        });
        entry.emit(Opcode::SetEnum {
            rd: 2,
            start: 0,
            count: 2,
        });
        entry.emit(Opcode::ChooseBegin {
            rd: 3,
            r_binding: 4,
            r_domain: 2,
            loop_end: 3,
        });
        entry.emit(Opcode::LoadBool { rd: 5, value: true });
        entry.emit(Opcode::ChooseNext {
            rd: 3,
            r_binding: 4,
            r_body: 5,
            loop_begin: -1,
        });
        entry.emit(Opcode::StoreVar { var_idx: 0, rs: 3 });
        entry.emit(Opcode::LoadBool { rd: 6, value: true });
        entry.emit(Opcode::Ret { rs: 6 });
        let instructions = entry.instructions.clone();
        chunk.add_function(entry);

        validate_value_action_vm_eligibility(0, &instructions, &chunk, 1)
            .expect("CHOOSE with its successor write after the loop must certify");

        let parent = [tla_value::Value::Bool(false)];
        let outcome = tla_eval::bytecode_vm::BytecodeVm::new(&chunk, &parent, None)
            .execute_action_function(0)
            .expect("certified CHOOSE action must execute in Value action mode");
        assert_eq!(
            outcome,
            tla_eval::bytecode_vm::ActionVmOutcome::Enabled(
                [(0, tla_value::Value::tuple([tla_value::Value::int(2)]))]
                    .into_iter()
                    .collect()
            )
        );
    }

    #[test]
    fn test_value_action_vm_rejects_call_arity_mismatch() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let mut helper = BytecodeFunction::new("UnaryHelper".to_string(), 1);
        helper.emit(Opcode::Ret { rs: 0 });
        chunk.add_function(helper);

        let entry = vec![
            Opcode::Call {
                rd: 7,
                op_idx: 1,
                args_start: 0,
                argc: 0,
            },
            Opcode::LoadImm { rd: 0, value: 7 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::LoadBool { rd: 1, value: true },
            Opcode::Ret { rs: 1 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect("the shared validator does not certify Call argument layout");
        let reason = validate_value_action_vm_eligibility(0, &entry, &chunk, 1)
            .expect_err("Value action calls must match the bytecode callee arity");
        assert!(reason.contains("passes 0 arguments to callee 1 with arity 1"));
    }

    #[test]
    fn test_value_action_vm_accepts_and_executes_arity_one_pure_helper() {
        let entry_instructions = vec![
            Opcode::LoadImm { rd: 0, value: 41 },
            Opcode::Call {
                rd: 1,
                op_idx: 1,
                args_start: 0,
                argc: 1,
            },
            Opcode::StoreVar { var_idx: 0, rs: 1 },
            Opcode::LoadBool { rd: 2, value: true },
            Opcode::Ret { rs: 2 },
        ];
        let mut entry = BytecodeFunction::new("Entry".to_string(), 0);
        for op in entry_instructions.iter().copied() {
            entry.emit(op);
        }

        let mut helper = BytecodeFunction::new("Increment".to_string(), 1);
        helper.emit(Opcode::LoadImm { rd: 1, value: 1 });
        helper.emit(Opcode::AddInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        helper.emit(Opcode::Ret { rs: 2 });

        let mut chunk = BytecodeChunk::new();
        chunk.add_function(entry);
        chunk.add_function(helper);

        validate_value_action_vm_eligibility(0, &entry_instructions, &chunk, 1)
            .expect("a matching arity-1 pure helper must remain Value-action eligible");

        let parent = [tla_value::Value::int(0)];
        let outcome = tla_eval::bytecode_vm::BytecodeVm::new(&chunk, &parent, None)
            .execute_action_function(0)
            .expect("the certified helper call must execute in Value action mode");
        assert_eq!(
            outcome,
            tla_eval::bytecode_vm::ActionVmOutcome::Enabled(
                [(0, tla_value::Value::int(42))].into_iter().collect()
            )
        );
    }

    #[test]
    fn test_value_action_vm_rejects_func_def_in_reachable_callee() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let mut helper = BytecodeFunction::new("UnsupportedHelper".to_string(), 0);
        helper.emit(Opcode::FuncDef {
            rd: 0,
            r_domain: 1,
            r_binding: 2,
        });
        helper.emit(Opcode::Ret { rs: 0 });
        chunk.add_function(helper);

        let entry = vec![
            Opcode::Call {
                rd: 7,
                op_idx: 1,
                args_start: 0,
                argc: 0,
            },
            Opcode::LoadImm { rd: 0, value: 7 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::LoadBool { rd: 1, value: true },
            Opcode::Ret { rs: 1 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 1)
            .expect("the shared validator does not classify Value-VM opcode support");
        let reason = validate_value_action_vm_eligibility(0, &entry, &chunk, 1)
            .expect_err("non-loop FuncDef in a reachable helper must fail closed");
        assert!(reason.contains("reachable callee 1 contains unsupported FuncDef"));
    }

    #[test]
    fn test_value_action_vm_requires_zero_arity_entry() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 1));
        let entry = vec![
            Opcode::LoadImm { rd: 0, value: 7 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::LoadBool { rd: 1, value: true },
            Opcode::Ret { rs: 1 },
        ];

        let reason = validate_value_action_vm_eligibility(0, &entry, &chunk, 1)
            .expect_err("unspecialized action parameters must not enter the Value action VM");
        assert!(reason.contains("must have arity 0, got 1"));
    }

    #[test]
    fn test_value_action_vm_rejects_enabled_partial_successor() {
        let mut chunk = BytecodeChunk::new();
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        let entry = vec![
            Opcode::LoadImm { rd: 0, value: 7 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::LoadBool { rd: 1, value: true },
            Opcode::Ret { rs: 1 },
        ];

        validate_next_state_action_chunk(0, &entry, &chunk, 2)
            .expect("the shared native validator does not require full Value overlay coverage");
        let reason = validate_value_action_vm_eligibility(0, &entry, &chunk, 2)
            .expect_err("enabled Value action returns must bind every successor slot");
        assert!(reason.contains("successor variables [1] unbound"));
    }

    fn register_reuse_function(instructions: impl IntoIterator<Item = Opcode>) -> BytecodeFunction {
        let mut function = BytecodeFunction::new("ReuseEntry".to_string(), 0);
        for op in instructions {
            function.emit(op);
        }
        function
    }

    #[test]
    fn value_action_vm_register_reuse_accepts_straight_line_and_both_branch_definitions() {
        let straight_line = register_reuse_function([
            Opcode::LoadImm { rd: 0, value: 7 },
            Opcode::Move { rd: 1, rs: 0 },
            Opcode::Ret { rs: 1 },
        ]);
        certify_value_action_vm_register_reuse(&straight_line)
            .expect("straight-line reads are dominated by writes");

        let both_branches = register_reuse_function([
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::JumpFalse { rs: 0, offset: 3 },
            Opcode::LoadImm { rd: 1, value: 10 },
            Opcode::Jump { offset: 2 },
            Opcode::LoadImm { rd: 1, value: 20 },
            Opcode::Ret { rs: 1 },
        ]);
        certify_value_action_vm_register_reuse(&both_branches)
            .expect("both arms define the joined return register");
    }

    #[test]
    fn value_action_vm_register_reuse_rejects_partial_branch_and_condmove_definitions() {
        let partial_branch = register_reuse_function([
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::JumpFalse { rs: 0, offset: 3 },
            Opcode::LoadImm { rd: 1, value: 10 },
            Opcode::Jump { offset: 2 },
            Opcode::Nop,
            Opcode::Ret { rs: 1 },
        ]);
        let reason = certify_value_action_vm_register_reuse(&partial_branch)
            .expect_err("one-arm-only assignment must not certify");
        assert!(
            reason.contains("reads r1 before a definite assignment"),
            "{reason}"
        );

        let cond_move = register_reuse_function([
            Opcode::LoadBool {
                rd: 0,
                value: false,
            },
            Opcode::LoadImm { rd: 1, value: 9 },
            Opcode::CondMove {
                rd: 2,
                cond: 0,
                rs: 1,
            },
            Opcode::Ret { rs: 2 },
        ]);
        let reason = certify_value_action_vm_register_reuse(&cond_move)
            .expect_err("CondMove's false arm preserves an uninitialized destination");
        assert!(
            reason.contains("reads r2 before a definite assignment"),
            "{reason}"
        );

        let initialized_cond_move = register_reuse_function([
            Opcode::LoadBool {
                rd: 0,
                value: false,
            },
            Opcode::LoadImm { rd: 1, value: 9 },
            Opcode::LoadImm { rd: 2, value: 4 },
            Opcode::CondMove {
                rd: 2,
                cond: 0,
                rs: 1,
            },
            Opcode::Ret { rs: 2 },
        ]);
        certify_value_action_vm_register_reuse(&initialized_cond_move)
            .expect("a pre-defined CondMove destination stays definitely assigned");
    }

    #[test]
    fn value_action_vm_register_reuse_checks_implicit_returns_and_declared_max() {
        let arity_one = BytecodeFunction::new("ArityOne".to_string(), 1);
        let reason = certify_value_action_vm_register_reuse(&arity_one)
            .expect_err("the action executor does not install argument registers");
        assert!(reason.contains("must have arity 0, got 1"), "{reason}");

        let empty = BytecodeFunction::new("Empty".to_string(), 0);
        let reason = certify_value_action_vm_register_reuse(&empty)
            .expect_err("an empty zero-arity function implicitly reads initial FALSE from r0");
        assert!(reason.contains("implicit return reads r0"), "{reason}");

        let implicit_defined = register_reuse_function([Opcode::LoadBool { rd: 0, value: true }]);
        certify_value_action_vm_register_reuse(&implicit_defined)
            .expect("falling off after defining r0 is safe");

        let implicit_undefined = register_reuse_function([Opcode::LoadBool { rd: 1, value: true }]);
        let reason = certify_value_action_vm_register_reuse(&implicit_undefined)
            .expect_err("falling off still reads r0");
        assert!(reason.contains("implicit return reads r0"), "{reason}");

        let mut underreported = register_reuse_function([
            Opcode::LoadBool { rd: 1, value: true },
            Opcode::Ret { rs: 1 },
        ]);
        underreported.max_register = 0;
        let reason = certify_value_action_vm_register_reuse(&underreported)
            .expect_err("all referenced registers must fit the declared frame");
        assert!(reason.contains("destination r1"), "{reason}");
        assert!(
            reason.contains("exceeds declared max register r0"),
            "{reason}"
        );
    }

    #[test]
    fn value_action_vm_register_reuse_rejects_entry_loops_but_accepts_direct_calls() {
        let backedge = register_reuse_function([
            Opcode::LoadBool { rd: 0, value: true },
            Opcode::Jump { offset: -1 },
        ]);
        let reason = certify_value_action_vm_register_reuse(&backedge)
            .expect_err("entry backedges stay on the reset path");
        assert!(reason.contains("rejects backedge"), "{reason}");

        let loop_opcode = register_reuse_function([
            Opcode::LoadConst { rd: 0, idx: 0 },
            Opcode::ForallBegin {
                rd: 1,
                r_binding: 2,
                r_domain: 0,
                loop_end: 2,
            },
            Opcode::Ret { rs: 1 },
        ]);
        let reason = certify_value_action_vm_register_reuse(&loop_opcode)
            .expect_err("entry-local loop opcodes stay on the reset path");
        assert!(reason.contains("entry-local loop opcode"), "{reason}");

        let direct_call = register_reuse_function([
            Opcode::LoadImm { rd: 0, value: 41 },
            Opcode::Call {
                rd: 1,
                op_idx: 7,
                args_start: 0,
                argc: 1,
            },
            Opcode::Ret { rs: 1 },
        ]);
        certify_value_action_vm_register_reuse(&direct_call)
            .expect("Call reads defined arguments and defines its result on continuation");
    }

    #[test]
    fn value_action_vm_register_reuse_ignores_unreachable_unsafe_code() {
        let function = register_reuse_function([
            Opcode::LoadBool { rd: 0, value: true },
            Opcode::Ret { rs: 0 },
            Opcode::Move { rd: 1, rs: 2 },
            Opcode::Jump { offset: -1 },
        ]);
        certify_value_action_vm_register_reuse(&function)
            .expect("unreachable stale reads and backedges cannot affect execution");
    }
}
