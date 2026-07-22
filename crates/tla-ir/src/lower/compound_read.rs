// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compound-READ callout admission pre-scan (wishlist item 4 M1).
//!
//! # What this decides
//!
//! Under ty's hybrid flat view every non-flat-admissible state variable is
//! demoted to a [`CompoundLayout::Dynamic`] 1-slot placeholder whose buffer
//! slot carries no information; M0 hard-declines every access to one
//! (`Ctx::reject_hybrid_placeholder_var_access`). M1 lifts that decline for
//! **reads that terminate in a scalar leaf**, servicing them with one
//! `tla_hybrid_compound_apply{1,2}_i64` host call against the parent
//! `ArrayState` ty publishes for the duration of the dispatch.
//!
//! Emission is only sound when the compound value NEVER escapes as a whole.
//! `childOf[n, k]` is serviceable; `DOMAIN childOf`, `childOf = other`,
//! `Helper(childOf)`, `x' = childOf` are not — each of those needs the whole
//! aggregate, which the placeholder slot does not contain. A partially lowered
//! compound read would therefore be silently wrong, not merely slow.
//!
//! This module is that soundness boundary. It classifies **every** use of
//! every register rooted at a placeholder `LoadVar` and admits the variable
//! only when each use is the `func` operand of a chain-terminating `FuncApply`.
//! Anything else marks the variable escaped, no callout is planned for it, and
//! its `LoadVar` falls back into the M0 hard decline — so the whole action
//! routes to the interpreter.
//!
//! # Why an empty plan is the fail-closed state
//!
//! [`plan_compound_reads`] returns a plan, never an error. A variable that is
//! not admitted simply has no entry, which leaves the M0-G3 guard in place for
//! it. "Decline the action" and "plan nothing" are therefore the same
//! codepath, and any bug in this analysis that under-admits costs performance
//! while any bug that fails to run at all costs nothing.
//!
//! # Single source of truth for the declared footprint
//!
//! ty's M1 admission gate only lets a compiled read of a placeholder var
//! through when the artifact DECLARED that var as a callout read
//! (`TrustCgNativeActionEntry::compound_read_vars`). That declaration is
//! produced by calling [`compound_read_callout_vars`] — this same analysis, on
//! the same bytecode and layout — so the declaration cannot drift from what
//! the lowering actually emitted.

use std::collections::{BTreeSet, HashMap, HashSet};

use tla_jit_abi::{CompoundLayout, StateLayout as JitStateLayout, VarLayout};
use tla_tir::bytecode::{ConstantPool, Opcode};
use tla_value::Value;

use num_traits::ToPrimitive;

// Scalar kind codes — must match `tla_trust_cg::runtime_abi::compound_read`'s
// `CR_KIND_*`. String and ModelValue intern to the SAME NameId, so the kind is
// the only thing that distinguishes them at the boundary; it is passed
// explicitly for every key and for the expected leaf.
pub(super) const CR_KIND_INT: i64 = 0;
pub(super) const CR_KIND_BOOL: i64 = 1;
pub(super) const CR_KIND_STRING: i64 = 2;
pub(super) const CR_KIND_MODEL_VALUE: i64 = 3;

/// Host symbol for the fused single-key scalar apply (`var[k0]`).
pub(super) const CR_APPLY1_SYMBOL: &str = "tla_hybrid_compound_apply1_i64";
/// Host symbol for the fused two-key scalar apply (`var[k0, k1]` / `var[k0][k1]`).
pub(super) const CR_APPLY2_SYMBOL: &str = "tla_hybrid_compound_apply2_i64";

/// Environment gate. Default OFF: with the variable unset the plan is always
/// empty, no callout is ever emitted, no artifact declares a compound-read
/// footprint, and the M1 admission gate degrades exactly to M0.
pub(super) fn compound_read_emission_enabled() -> bool {
    std::env::var_os("TY_HYBRID_COMPOUND_READ").as_deref() == Some(std::ffi::OsStr::new("1"))
}

/// One planned callout: the placeholder variable to read and the bytecode
/// registers holding its scalar keys (one or two).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CompoundReadCallout {
    pub(super) var_idx: u16,
    /// Bytecode registers holding the scalar keys, outermost first. Length 1
    /// selects `apply1`, length 2 selects `apply2`.
    pub(super) key_regs: Vec<u8>,
    /// Declared kind of the scalar leaf, inferred from how the result is
    /// consumed. A wrong inference is fail-closed, not unsound: the callout
    /// returns `CR_ERR_KIND_MISMATCH`, latches the sticky status, and the
    /// dispatcher discards the whole native execution.
    pub(super) expect_kind: i64,
}

/// The admitted compound-read plan for one bytecode body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct CompoundReadPlan {
    /// PCs of chain-terminating `FuncApply`s that lower to a host callout.
    pub(super) callouts: HashMap<usize, CompoundReadCallout>,
    /// PCs that must emit NOTHING: placeholder `LoadVar` roots, intermediate
    /// curried `FuncApply`s, and elided tuple-key builds. Their register
    /// results are provably never read except by the chain that consumed them.
    pub(super) elided: HashSet<usize>,
    /// Placeholder variables serviced by the plan, sorted and deduped. This is
    /// the declared compound-read footprint ty's M1 gate checks.
    pub(super) vars: Vec<u16>,
}

/// The placeholder variables an action's compiled artifact reads through the
/// callout — the declared compound-read footprint.
///
/// This is the ty-side entry point. It runs the identical analysis the
/// lowering runs, so a declaration can never claim a var the lowering did not
/// actually service.
#[must_use]
pub fn compound_read_callout_vars(
    instructions: &[Opcode],
    layout: Option<&JitStateLayout>,
    const_pool: Option<&ConstantPool>,
) -> Vec<u16> {
    let Some(layout) = layout else {
        return Vec::new();
    };
    plan_compound_reads(instructions, layout, const_pool).vars
}

/// Registers an opcode WRITES.
///
/// Exhaustive by construction — there is deliberately no `_` arm, so adding a
/// bytecode opcode is a compile error here rather than a silent hole in the
/// escape analysis below.
fn opcode_writes(op: &Opcode) -> Vec<u8> {
    match *op {
        Opcode::LoadImm { rd, .. }
        | Opcode::LoadBool { rd, .. }
        | Opcode::LoadConst { rd, .. }
        | Opcode::LoadVar { rd, .. }
        | Opcode::LoadPrime { rd, .. }
        | Opcode::Move { rd, .. }
        | Opcode::AddInt { rd, .. }
        | Opcode::SubInt { rd, .. }
        | Opcode::MulInt { rd, .. }
        | Opcode::DivInt { rd, .. }
        | Opcode::IntDiv { rd, .. }
        | Opcode::ModInt { rd, .. }
        | Opcode::NegInt { rd, .. }
        | Opcode::PowInt { rd, .. }
        | Opcode::Eq { rd, .. }
        | Opcode::Neq { rd, .. }
        | Opcode::LtInt { rd, .. }
        | Opcode::LeInt { rd, .. }
        | Opcode::GtInt { rd, .. }
        | Opcode::GeInt { rd, .. }
        | Opcode::And { rd, .. }
        | Opcode::Or { rd, .. }
        | Opcode::Not { rd, .. }
        | Opcode::Implies { rd, .. }
        | Opcode::Equiv { rd, .. }
        | Opcode::Call { rd, .. }
        | Opcode::ValueApply { rd, .. }
        | Opcode::SetEnum { rd, .. }
        | Opcode::SetIn { rd, .. }
        | Opcode::Tuple2SetIn { rd, .. }
        | Opcode::SetEnumSubseteq { rd, .. }
        | Opcode::Tuple2SelfEq { rd, .. }
        | Opcode::Tuple2SelfSubseteq { rd, .. }
        | Opcode::SetUnion { rd, .. }
        | Opcode::SetIntersect { rd, .. }
        | Opcode::SetDiff { rd, .. }
        | Opcode::Subseteq { rd, .. }
        | Opcode::RoundStepEq { rd, .. }
        | Opcode::Powerset { rd, .. }
        | Opcode::BigUnion { rd, .. }
        | Opcode::KSubset { rd, .. }
        | Opcode::Range { rd, .. }
        | Opcode::RecordNew { rd, .. }
        | Opcode::RecordGet { rd, .. }
        | Opcode::FuncApply { rd, .. }
        | Opcode::Domain { rd, .. }
        | Opcode::FuncExcept { rd, .. }
        | Opcode::TupleNew { rd, .. }
        | Opcode::TupleGet { rd, .. }
        | Opcode::FuncSet { rd, .. }
        | Opcode::RecordSet { rd, .. }
        | Opcode::Times { rd, .. }
        | Opcode::SeqNew { rd, .. }
        | Opcode::StrConcat { rd, .. }
        | Opcode::CondMove { rd, .. }
        | Opcode::Unchanged { rd, .. }
        | Opcode::MakeClosure { rd, .. }
        | Opcode::CallExternal { rd, .. }
        | Opcode::Concat { rd, .. }
        | Opcode::CallBuiltin { rd, .. }
        | Opcode::EqFuncExcept { rd, .. }
        | Opcode::EqRecordNew { rd, .. } => vec![rd],

        // Loop constructs write their accumulator AND rebind the loop
        // variable on every iteration.
        Opcode::FuncDef { rd, r_binding, .. }
        | Opcode::ForallBegin { rd, r_binding, .. }
        | Opcode::ForallNext { rd, r_binding, .. }
        | Opcode::ExistsBegin { rd, r_binding, .. }
        | Opcode::ExistsNext { rd, r_binding, .. }
        | Opcode::ChooseBegin { rd, r_binding, .. }
        | Opcode::ChooseNext { rd, r_binding, .. }
        | Opcode::SetBuilderBegin { rd, r_binding, .. }
        | Opcode::SetFilterBegin { rd, r_binding, .. }
        | Opcode::FuncDefBegin { rd, r_binding, .. } => vec![rd, r_binding],
        Opcode::LoopNext { r_binding, .. } => vec![r_binding],

        Opcode::StoreVar { .. }
        | Opcode::Jump { .. }
        | Opcode::JumpTrue { .. }
        | Opcode::JumpFalse { .. }
        | Opcode::Ret { .. }
        | Opcode::SetPrimeMode { .. }
        | Opcode::Nop
        | Opcode::Halt => Vec::new(),
    }
}

/// Registers an opcode READS.
///
/// Exhaustive by construction (no `_` arm) for the same reason as
/// [`opcode_writes`]: a missed read is the one mistake in this file that could
/// admit an escaping compound value.
fn opcode_reads(op: &Opcode) -> Vec<u8> {
    fn block(start: u8, count: u8) -> Vec<u8> {
        (0..count).map(|i| start.saturating_add(i)).collect()
    }
    match *op {
        Opcode::LoadImm { .. }
        | Opcode::LoadBool { .. }
        | Opcode::LoadConst { .. }
        | Opcode::LoadVar { .. }
        | Opcode::LoadPrime { .. }
        | Opcode::Jump { .. }
        | Opcode::SetPrimeMode { .. }
        | Opcode::Unchanged { .. }
        | Opcode::Nop
        | Opcode::Halt => Vec::new(),

        Opcode::Move { rs, .. }
        | Opcode::StoreVar { rs, .. }
        | Opcode::NegInt { rs, .. }
        | Opcode::Not { rs, .. }
        | Opcode::JumpTrue { rs, .. }
        | Opcode::JumpFalse { rs, .. }
        | Opcode::Ret { rs, .. }
        | Opcode::Powerset { rs, .. }
        | Opcode::BigUnion { rs, .. }
        | Opcode::RecordGet { rs, .. }
        | Opcode::Domain { rs, .. }
        | Opcode::TupleGet { rs, .. } => vec![rs],

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
        | Opcode::Concat { r1, r2, .. } => vec![r1, r2],

        Opcode::SetIn { elem, set, .. } => vec![elem, set],
        Opcode::Tuple2SetIn {
            first, second, set, ..
        } => vec![first, second, set],
        Opcode::SetEnumSubseteq {
            start, count, set, ..
        } => {
            let mut regs = block(start, count);
            regs.push(set);
            regs
        }
        Opcode::Tuple2SelfEq { value, .. } | Opcode::Tuple2SelfSubseteq { value, .. } => {
            vec![value]
        }
        Opcode::RoundStepEq { child, parent, .. } => vec![child, parent],
        Opcode::KSubset { base, k, .. } => vec![base, k],
        Opcode::Range { lo, hi, .. } => vec![lo, hi],
        Opcode::FuncApply { func, arg, .. } => vec![func, arg],
        Opcode::FuncExcept {
            func, path, val, ..
        } => vec![func, path, val],
        Opcode::FuncSet { domain, range, .. } => vec![domain, range],
        Opcode::CondMove { cond, rs, .. } => vec![cond, rs],
        Opcode::EqFuncExcept {
            lhs,
            func,
            path,
            val,
            ..
        } => vec![lhs, func, path, val],

        Opcode::SetEnum { start, count, .. }
        | Opcode::TupleNew { start, count, .. }
        | Opcode::SeqNew { start, count, .. }
        | Opcode::Times { start, count, .. } => block(start, count),
        Opcode::RecordNew {
            values_start,
            count,
            ..
        }
        | Opcode::RecordSet {
            values_start,
            count,
            ..
        } => block(values_start, count),
        Opcode::EqRecordNew {
            lhs,
            values_start,
            count,
            ..
        } => {
            let mut regs = vec![lhs];
            regs.extend(block(values_start, count));
            regs
        }
        Opcode::MakeClosure {
            captures_start,
            capture_count,
            ..
        } => block(captures_start, capture_count),
        Opcode::Call {
            args_start, argc, ..
        }
        | Opcode::CallExternal {
            args_start, argc, ..
        }
        | Opcode::CallBuiltin {
            args_start, argc, ..
        } => block(args_start, argc),
        Opcode::ValueApply {
            func,
            args_start,
            argc,
            ..
        } => {
            let mut regs = vec![func];
            regs.extend(block(args_start, argc));
            regs
        }

        // Loop headers read their domain; the *Next forms read the loop body
        // result and the binding they are about to advance.
        Opcode::FuncDef {
            r_domain,
            r_binding,
            ..
        }
        | Opcode::ForallBegin {
            r_domain,
            r_binding,
            ..
        }
        | Opcode::ExistsBegin {
            r_domain,
            r_binding,
            ..
        }
        | Opcode::ChooseBegin {
            r_domain,
            r_binding,
            ..
        }
        | Opcode::SetBuilderBegin {
            r_domain,
            r_binding,
            ..
        }
        | Opcode::SetFilterBegin {
            r_domain,
            r_binding,
            ..
        }
        | Opcode::FuncDefBegin {
            r_domain,
            r_binding,
            ..
        } => vec![r_domain, r_binding],
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
        } => vec![r_binding, r_body],
    }
}

/// Scalar kind of a constant-pool value, if it is a scalar at all.
fn kind_of_const_value(value: &Value) -> Option<i64> {
    match value {
        Value::SmallInt(_) => Some(CR_KIND_INT),
        Value::Int(n) => n.to_i64().map(|_| CR_KIND_INT),
        Value::Bool(_) => Some(CR_KIND_BOOL),
        Value::String(_) => Some(CR_KIND_STRING),
        Value::ModelValue(_) => Some(CR_KIND_MODEL_VALUE),
        _ => None,
    }
}

/// Kind a register's value must have, judged from the single opcode that
/// produced it. Only the unambiguous producers are recognised.
fn kind_from_producer(
    instructions: &[Opcode],
    pc: usize,
    const_pool: Option<&ConstantPool>,
) -> Option<i64> {
    match instructions.get(pc)? {
        Opcode::LoadImm { .. }
        | Opcode::AddInt { .. }
        | Opcode::SubInt { .. }
        | Opcode::MulInt { .. }
        | Opcode::DivInt { .. }
        | Opcode::IntDiv { .. }
        | Opcode::ModInt { .. }
        | Opcode::NegInt { .. }
        | Opcode::PowInt { .. } => Some(CR_KIND_INT),
        Opcode::LoadBool { .. }
        | Opcode::Eq { .. }
        | Opcode::Neq { .. }
        | Opcode::LtInt { .. }
        | Opcode::LeInt { .. }
        | Opcode::GtInt { .. }
        | Opcode::GeInt { .. }
        | Opcode::And { .. }
        | Opcode::Or { .. }
        | Opcode::Not { .. }
        | Opcode::Implies { .. }
        | Opcode::Equiv { .. }
        | Opcode::SetIn { .. }
        | Opcode::Tuple2SetIn { .. }
        | Opcode::SetEnumSubseteq { .. }
        | Opcode::Tuple2SelfEq { .. }
        | Opcode::Tuple2SelfSubseteq { .. }
        | Opcode::Subseteq { .. }
        | Opcode::RoundStepEq { .. } => Some(CR_KIND_BOOL),
        Opcode::StrConcat { .. } => Some(CR_KIND_STRING),
        Opcode::LoadConst { idx, .. } => {
            let pool = const_pool?;
            if usize::from(*idx) >= pool.value_count() {
                return None;
            }
            kind_of_const_value(pool.get_value(*idx))
        }
        _ => None,
    }
}

/// Scalar kind of a flat state variable's slot, for `StoreVar`-directed leaf
/// kind inference. Only the two unambiguous whole-variable scalar layouts
/// participate; every compact/tagged carrier declines.
fn kind_from_var_layout(layout: &JitStateLayout, var_idx: u16) -> Option<i64> {
    match layout.var_layout(usize::from(var_idx))? {
        VarLayout::ScalarInt => Some(CR_KIND_INT),
        VarLayout::ScalarBool => Some(CR_KIND_BOOL),
        _ => None,
    }
}

/// Per-register write sites.
type WriteSites = HashMap<u8, Vec<usize>>;

/// Build the compound-read plan for one bytecode body.
///
/// Returns an empty plan — the fail-closed state, in which every placeholder
/// access stays on the M0 hard decline — unless the gate is on, the layout is
/// a hybrid flat view, and at least one placeholder variable is provably read
/// only through chain-terminating scalar applications.
pub(super) fn plan_compound_reads(
    instructions: &[Opcode],
    layout: &JitStateLayout,
    const_pool: Option<&ConstantPool>,
) -> CompoundReadPlan {
    let empty = CompoundReadPlan::default();
    if !compound_read_emission_enabled() || !layout.is_hybrid_flat_view() {
        return empty;
    }

    let dynamic_vars: HashSet<u16> = (0..layout.var_count())
        .filter(|&i| {
            matches!(
                layout.var_layout(i),
                Some(VarLayout::Compound(CompoundLayout::Dynamic))
            )
        })
        .filter_map(|i| u16::try_from(i).ok())
        .collect();
    if dynamic_vars.is_empty() {
        return empty;
    }

    let mut writes: WriteSites = HashMap::new();
    for (pc, op) in instructions.iter().enumerate() {
        for reg in opcode_writes(op) {
            writes.entry(reg).or_default().push(pc);
        }
    }

    // --- Roots: registers whose EVERY definition is a placeholder LoadVar of
    // one and the same variable. A register with any other definition is not a
    // root, so its placeholder LoadVar keeps the M0 decline.
    let mut roots: HashMap<u8, u16> = HashMap::new();
    for (&reg, pcs) in &writes {
        let mut var: Option<u16> = None;
        let mut all_placeholder_loads = true;
        for &pc in pcs {
            match instructions[pc] {
                Opcode::LoadVar { rd, var_idx } if rd == reg && dynamic_vars.contains(&var_idx) => {
                    if *var.get_or_insert(var_idx) != var_idx {
                        all_placeholder_loads = false;
                    }
                }
                _ => all_placeholder_loads = false,
            }
        }
        if all_placeholder_loads {
            if let Some(var_idx) = var {
                roots.insert(reg, var_idx);
            }
        }
    }
    if roots.is_empty() {
        return empty;
    }

    // Registers used in the `func` position of some FuncApply. A chain node
    // whose result feeds another application is an intermediate, not a leaf.
    let func_operand_regs: HashSet<u8> = instructions
        .iter()
        .filter_map(|op| match *op {
            Opcode::FuncApply { func, .. } => Some(func),
            _ => None,
        })
        .collect();

    // --- Curried intermediates: `t = root[k0]` with a single definition, later
    // applied again as `t[k1]`. A single definition is required so the key
    // register of the first application is unambiguous.
    let mut chain1: HashMap<u8, (u16, usize)> = HashMap::new();
    for (&reg, pcs) in &writes {
        if roots.contains_key(&reg) || pcs.len() != 1 || !func_operand_regs.contains(&reg) {
            continue;
        }
        let pc = pcs[0];
        if let Opcode::FuncApply { rd, func, .. } = instructions[pc] {
            if rd == reg {
                if let Some(&var_idx) = roots.get(&func) {
                    chain1.insert(reg, (var_idx, pc));
                }
            }
        }
    }

    // --- Escape analysis. A placeholder variable is admitted only when every
    // read of every register carrying it is the `func` operand of a FuncApply.
    let mut escaped: HashSet<u16> = HashSet::new();
    let mut read_counts: HashMap<u8, usize> = HashMap::new();
    for op in instructions {
        for reg in opcode_reads(op) {
            let owner = roots
                .get(&reg)
                .copied()
                .or_else(|| chain1.get(&reg).map(|&(v, _)| v));
            let Some(var_idx) = owner else { continue };
            *read_counts.entry(reg).or_default() += 1;
            let consumed_as_function =
                matches!(*op, Opcode::FuncApply { func, arg, .. } if func == reg && arg != reg);
            if !consumed_as_function {
                escaped.insert(var_idx);
            }
        }
    }

    // A placeholder variable that is written or re-read from the successor
    // buffer is out of M1's read-only contract entirely.
    for op in instructions {
        match *op {
            Opcode::StoreVar { var_idx, .. } | Opcode::LoadPrime { var_idx, .. } => {
                if dynamic_vars.contains(&var_idx) {
                    escaped.insert(var_idx);
                }
            }
            _ => {}
        }
    }

    // A root or intermediate with no reads at all is dead. Declining it keeps
    // the admission boundary "every use is a terminating apply" literally true
    // rather than vacuously true.
    for (&reg, &var_idx) in &roots {
        if read_counts.get(&reg).copied().unwrap_or(0) == 0 {
            escaped.insert(var_idx);
        }
    }
    for (&reg, &(var_idx, _)) in &chain1 {
        if read_counts.get(&reg).copied().unwrap_or(0) == 0 {
            escaped.insert(var_idx);
        }
    }

    let block_targets = collect_branch_target_pcs(instructions);

    // --- Emit plan entries.
    let mut callouts: HashMap<usize, CompoundReadCallout> = HashMap::new();
    // Elided PC -> the placeholder variable whose chain justifies eliding it.
    // Tracking the owner (rather than a bare set) is what lets a variable that
    // escapes later withdraw its elisions along with its callouts.
    let mut elided_owner: HashMap<usize, u16> = HashMap::new();
    let mut rejected: HashSet<u16> = escaped;

    for (pc, op) in instructions.iter().enumerate() {
        let Opcode::FuncApply { rd, func, arg } = *op else {
            continue;
        };
        let (var_idx, key_regs, elided_here) = if let Some(&var_idx) = roots.get(&func) {
            if chain1.get(&rd).is_some_and(|&(_, def_pc)| def_pc == pc) {
                // Intermediate of a curried two-key chain: the outer apply
                // emits the single fused callout.
                elided_owner.insert(pc, var_idx);
                continue;
            }
            if func_operand_regs.contains(&rd) {
                // The result feeds a further application this analysis did not
                // recognise as a curried intermediate (multiple definitions, or
                // a depth-3 chain). Servicing the inner apply as a scalar leaf
                // would hand the outer one a non-function.
                rejected.insert(var_idx);
                continue;
            }
            match tuple_key_regs(instructions, &writes, &read_counts, &block_targets, pc, arg) {
                Some((tuple_pc, k0, k1)) => (var_idx, vec![k0, k1], vec![tuple_pc]),
                None => (var_idx, vec![arg], Vec::new()),
            }
        } else if let Some(&(var_idx, def_pc)) = chain1.get(&func) {
            if func_operand_regs.contains(&rd) {
                rejected.insert(var_idx);
                continue;
            }
            let Opcode::FuncApply { arg: k0, .. } = instructions[def_pc] else {
                rejected.insert(var_idx);
                continue;
            };
            // The outer apply consumes the inner key register, so that register
            // must still hold the inner application's key here.
            if !value_survives(instructions, &block_targets, def_pc, pc, &[k0]) {
                rejected.insert(var_idx);
                continue;
            }
            (var_idx, vec![k0, arg], Vec::new())
        } else {
            continue;
        };

        // A key register that is itself a placeholder chain would need its own
        // callout first; that is a nested read this slice does not service.
        if key_regs
            .iter()
            .any(|k| roots.contains_key(k) || chain1.contains_key(k))
        {
            rejected.insert(var_idx);
            continue;
        }
        let Some(expect_kind) = infer_leaf_kind(instructions, layout, const_pool, &writes, rd)
        else {
            rejected.insert(var_idx);
            continue;
        };

        for elided_pc in elided_here {
            elided_owner.insert(elided_pc, var_idx);
        }
        callouts.insert(
            pc,
            CompoundReadCallout {
                var_idx,
                key_regs,
                expect_kind,
            },
        );
    }

    // Drop everything belonging to a variable that escaped anywhere. Admission
    // is per-variable and all-or-nothing: one escaping use forfeits every read
    // of that variable, and its LoadVar then hard-declines the action.
    callouts.retain(|_, c| !rejected.contains(&c.var_idx));
    let live_vars: BTreeSet<u16> = callouts.values().map(|c| c.var_idx).collect();
    if live_vars.is_empty() {
        return CompoundReadPlan::default();
    }
    // Roots of surviving variables emit nothing: their register is never read
    // except by the chain that the callouts above now answer directly.
    for (pc, op) in instructions.iter().enumerate() {
        if let Opcode::LoadVar { rd, var_idx } = *op {
            if roots.get(&rd) == Some(&var_idx) && live_vars.contains(&var_idx) {
                elided_owner.insert(pc, var_idx);
            }
        }
    }
    elided_owner.retain(|_, var_idx| live_vars.contains(var_idx));

    CompoundReadPlan {
        callouts,
        elided: elided_owner.into_keys().collect(),
        vars: live_vars.into_iter().collect(),
    }
}

/// PCs that some branch or loop construct can jump to. Used as a conservative
/// "a value defined at A may not reach B" test: any branch target inside
/// `(A, B]` means B is reachable without executing A.
fn collect_branch_target_pcs(instructions: &[Opcode]) -> BTreeSet<usize> {
    let mut targets = BTreeSet::new();
    let mut add = |pc: usize, offset: i32| {
        if let Ok(base) = i64::try_from(pc) {
            let t = base + i64::from(offset) + 1;
            if let Ok(t) = usize::try_from(t) {
                targets.insert(t);
            }
        }
    };
    for (pc, op) in instructions.iter().enumerate() {
        match *op {
            Opcode::Jump { offset }
            | Opcode::JumpTrue { offset, .. }
            | Opcode::JumpFalse { offset, .. } => add(pc, i32::from(offset)),
            Opcode::ForallBegin { loop_end, .. }
            | Opcode::ExistsBegin { loop_end, .. }
            | Opcode::ChooseBegin { loop_end, .. }
            | Opcode::SetBuilderBegin { loop_end, .. }
            | Opcode::SetFilterBegin { loop_end, .. }
            | Opcode::FuncDefBegin { loop_end, .. } => add(pc, i32::from(loop_end)),
            Opcode::ForallNext { loop_begin, .. }
            | Opcode::ExistsNext { loop_begin, .. }
            | Opcode::ChooseNext { loop_begin, .. }
            | Opcode::LoopNext { loop_begin, .. } => add(pc, i32::from(loop_begin)),
            _ => {}
        }
    }
    targets
}

/// True when a value defined at `def_pc` provably still holds `regs` at
/// `use_pc`: straight-line successor order, no intervening redefinition, and no
/// branch target in between that could reach `use_pc` without `def_pc`.
fn value_survives(
    instructions: &[Opcode],
    block_targets: &BTreeSet<usize>,
    def_pc: usize,
    use_pc: usize,
    regs: &[u8],
) -> bool {
    if def_pc >= use_pc {
        return false;
    }
    if block_targets.range(def_pc + 1..=use_pc).next().is_some() {
        return false;
    }
    for pc in (def_pc + 1)..use_pc {
        let written = opcode_writes(&instructions[pc]);
        if regs.iter().any(|r| written.contains(r)) {
            return false;
        }
    }
    true
}

/// Recognise a two-key application written as `var[<<k0, k1>>]`: the argument
/// register's sole definition is a 2-element tuple build whose only reader is
/// this apply, so the build itself can be elided and its element registers
/// passed as the two keys.
fn tuple_key_regs(
    instructions: &[Opcode],
    writes: &WriteSites,
    read_counts: &HashMap<u8, usize>,
    block_targets: &BTreeSet<usize>,
    apply_pc: usize,
    arg: u8,
) -> Option<(usize, u8, u8)> {
    let def_pcs = writes.get(&arg)?;
    if def_pcs.len() != 1 {
        return None;
    }
    let tuple_pc = def_pcs[0];
    let (start, count) = match instructions[tuple_pc] {
        Opcode::TupleNew { rd, start, count } | Opcode::SeqNew { rd, start, count }
            if rd == arg =>
        {
            (start, count)
        }
        _ => return None,
    };
    if count != 2 {
        return None;
    }
    // `read_counts` only ever counts placeholder-carrying registers, so a
    // non-zero entry here means the tuple register is itself a compound chain —
    // a nested read this slice does not service.
    if read_counts.get(&arg).copied().unwrap_or(0) != 0 {
        return None;
    }
    // Eliding the build is only sound when this apply is its ONLY consumer.
    let reader_count = instructions
        .iter()
        .filter(|op| opcode_reads(op).contains(&arg))
        .count();
    if reader_count != 1 {
        return None;
    }
    let k1 = start.checked_add(1)?;
    if !value_survives(
        instructions,
        block_targets,
        tuple_pc,
        apply_pc,
        &[start, k1],
    ) {
        return None;
    }
    Some((tuple_pc, start, k1))
}

/// Infer the scalar kind of the leaf a chain-terminating apply yields, from
/// how the result register is consumed.
///
/// Declining (returning `None`) costs a compile; guessing wrong costs a
/// discarded dispatch, never a wrong answer — the callout validates the kind
/// and latches `CR_ERR_KIND_MISMATCH`. Declining is still the right default:
/// an action whose every read mismatches would dispatch and be thrown away.
fn infer_leaf_kind(
    instructions: &[Opcode],
    layout: &JitStateLayout,
    const_pool: Option<&ConstantPool>,
    writes: &WriteSites,
    rd: u8,
) -> Option<i64> {
    let mut inferred: Option<i64> = None;
    let note = |kind: i64, inferred: &mut Option<i64>| -> bool {
        match *inferred {
            Some(existing) if existing != kind => false,
            _ => {
                *inferred = Some(kind);
                true
            }
        }
    };
    for op in instructions {
        if !opcode_reads(op).contains(&rd) {
            continue;
        }
        let kind = match *op {
            Opcode::AddInt { .. }
            | Opcode::SubInt { .. }
            | Opcode::MulInt { .. }
            | Opcode::DivInt { .. }
            | Opcode::IntDiv { .. }
            | Opcode::ModInt { .. }
            | Opcode::NegInt { .. }
            | Opcode::PowInt { .. }
            | Opcode::LtInt { .. }
            | Opcode::LeInt { .. }
            | Opcode::GtInt { .. }
            | Opcode::GeInt { .. }
            | Opcode::Range { .. } => CR_KIND_INT,
            Opcode::And { .. }
            | Opcode::Or { .. }
            | Opcode::Not { .. }
            | Opcode::Implies { .. }
            | Opcode::Equiv { .. }
            | Opcode::JumpTrue { .. }
            | Opcode::JumpFalse { .. } => CR_KIND_BOOL,
            Opcode::Eq { r1, r2, .. } | Opcode::Neq { r1, r2, .. } => {
                let other = if r1 == rd { r2 } else { r1 };
                let def_pcs = writes.get(&other)?;
                if def_pcs.len() != 1 {
                    return None;
                }
                kind_from_producer(instructions, def_pcs[0], const_pool)?
            }
            Opcode::StoreVar { var_idx, .. } => kind_from_var_layout(layout, var_idx)?,
            _ => return None,
        };
        if !note(kind, &mut inferred) {
            return None;
        }
    }
    inferred
}
