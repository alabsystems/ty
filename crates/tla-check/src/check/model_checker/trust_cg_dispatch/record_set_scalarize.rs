// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Record-set AGGREGATE scalarization: compile-time rewrite of a
//! `\E w \in <constant finite set> : <body over record-set aggregates>`
//! next-state action into per-witness, EXISTS-free scalar bytecode functions
//! that the existing single-successor native lowering compiles directly.
//!
//! Target shape (PaxosCommit `Phase2a`):
//!
//! ```tla
//! /\ ~\E m \in msgs : <per-record guard>                    \* boolean \A over msgs
//! /\ \E MS \in Majority :                                   \* constant domain (set-valued)
//!      LET mset   == {m \in msgs : <per-record filter>}     \* SetFilterBegin
//!          maxbal == Maximum({m.bal : m \in mset})          \* SetBuilderBegin + CallExternal
//!          val    == IF maxbal = -1 THEN "aborted"
//!                    ELSE (CHOOSE m \in mset : m.bal = maxbal).val
//!      IN /\ \A ac \in MS : \E m \in mset : m.acc = ac      \* boolean quantifiers over mset
//!         /\ msgs' = msgs \cup {[.. val |-> val]}           \* Send
//! /\ UNCHANGED <<rmState, aState>>
//! ```
//!
//! The rewrite exploits one invariant: the aggregate domain state variable
//! carries a PROVEN-CLOSED `RecordSetBitmask` layout, so its elements come
//! from a small compile-time universe of constant records whose fields are all
//! compile-time constants. Every aggregate over such a variable folds per
//! universe key:
//!
//! * boolean `\A m \in S : P(m)` / `\E m \in S : P(m)` with a per-key-constant
//!   body  ->  AND/OR chains over `SetIn(<constant record>, msgs)` membership
//!   tests (the exact `RecordNew`-of-constants + `SetIn` idiom the native
//!   record-set membership lowering already compiles);
//! * `{m \in msgs : P(m)}` -> a compile-time SUPPORT list of universe bits
//!   (never materialized at runtime);
//! * `{e(m) : m \in mset}` -> a compile-time list of (bit, constant) pairs;
//! * `CallExternal f({e(m) : m \in mset})` -> a decision tree over per-value
//!   presence bits with a plan-time-evaluated result table (the external is
//!   evaluated once per distinct-value subset via the fail-closed const-level
//!   evaluator supplied in [`RecordSetScalarizeEnv`]);
//! * `CHOOSE m \in mset : P(m, <runtime scalar>)` -> a first-match branch
//!   chain over the support keys in TLC-normalized order (the interpreter's
//!   CHOOSE order — which differs from the `Value::cmp` order every other
//!   quantifier uses), extracting the (constant) consumed field per match.
//!   A runtime no-match loads a SENTINEL value that provably matches no
//!   universe record and jumps straight to the successor construction: the
//!   strict record-set enum-fold then returns `FallbackNeeded`, the caller
//!   discards the native result, and the interpreter reproduces the exact
//!   CHOOSE-failure / out-of-universe-successor semantics for that state.
//!
//! Everything unrecognized fails CLOSED (`None`): the action simply stays on
//! its existing interpreter path.

use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::ops::Range;

use tla_tir::bytecode::{BytecodeFunction, ConstantPool, Opcode, Register};
use tla_value::Value;

/// Plan-time environment: the fail-closed constant-level evaluator used to
/// tabulate pure external operator calls.
///
/// The closure MUST evaluate `name(args...)` with NO state/next-state sources
/// (any state-variable access must error out -> `None`) against the same
/// operator environment the runtime `CallExternal` resolves. The intended
/// implementation is `try_eval_const_level` over a synthesized application:
/// dependency-tracked and state-source-free, so whatever it accepts is already
/// treated as referentially transparent by the evaluator's own const caches.
pub(in crate::check) struct RecordSetScalarizeEnv<'a> {
    #[allow(clippy::type_complexity)]
    pub eval_pure_op: &'a dyn Fn(&str, &[Value]) -> Option<Value>,
}

/// One per-witness expansion: the witness value bound to the successor-EXISTS
/// binding and the EXISTS-free scalar function specialized to it.
pub(in crate::check) struct ScalarizedWitnessExpansion {
    pub witness: Value,
    pub func: BytecodeFunction,
}

pub(in crate::check) struct ScalarizeOutcome {
    pub expansions: Vec<ScalarizedWitnessExpansion>,
    /// The source pool plus appended constants (shared by all expansions).
    pub pool: ConstantPool,
}

/// Hard caps (all fail-closed).
const MAX_WITNESSES: usize = 64;
const MAX_SUPPORT_BITS: usize = 64;
const MAX_EXTERNAL_DISTINCT_VALUES: usize = 4;
const MAX_OUTPUT_INSTRUCTIONS: usize = 4096;
const MAX_UNIVERSE: usize = 512;
const MAX_RECORD_FIELDS: usize = 16;

/// Sentinel field value that can never equal any universe record field: it
/// contains a control character, which no TLA+ string literal can.
const CHOOSE_SENTINEL: &str = "\u{1}__ty_record_set_scalarize_no_choose_witness__";

/// Try to scalarize `func` into per-witness EXISTS-free functions.
///
/// Fail-closed: any deviation from the recognized shape returns `None` and the
/// caller leaves the action on its existing (interpreter) path.
pub(in crate::check) fn scalarize_record_set_aggregate_action(
    func: &BytecodeFunction,
    pool: &ConstantPool,
    state_layout: &tla_jit_abi::StateLayout,
    env: &RecordSetScalarizeEnv<'_>,
) -> Option<ScalarizeOutcome> {
    if func.arity != 0 {
        return None;
    }
    let universes = reconstruct_record_universes(state_layout);
    if universes.is_empty() {
        return None;
    }

    // Locate the unique successor-producing EXISTS (body writes a primed var)
    // and require its domain to be a compile-time constant finite set.
    let pairs = quantifier_pairs(func)?;
    let successor = find_successor_exists(func, &pairs)?;
    let domain_value = chase_const_reg(func, successor.begin_pc, successor.r_domain, pool)?;
    let witnesses: Vec<Value> = iter_finite_set(&domain_value)?;
    if witnesses.is_empty() || witnesses.len() > MAX_WITNESSES {
        return None;
    }

    let mut shared_pool = pool.clone();
    let mut expansions = Vec::with_capacity(witnesses.len());
    for witness in witnesses {
        let mut walker = Walker {
            func,
            pool: &mut shared_pool,
            universes: &universes,
            env,
            successor: &successor,
            witness: witness.clone(),
            syms: vec![Sym::None; 256],
            out: Vec::new(),
            jump_fixups: Vec::new(),
            orig_labels: FxHashMap::default(),
            synth_labels: Vec::new(),
            memberships: BTreeMap::new(),
            membership_order: Vec::new(),
            next_membership_reg: 255,
            scratch_top: usize::from(func.max_register) + 1,
            max_scratch: usize::from(func.max_register) + 1,
        };
        let scalarized = walker.run()?;
        expansions.push(ScalarizedWitnessExpansion {
            witness,
            func: scalarized,
        });
    }
    Some(ScalarizeOutcome {
        expansions,
        pool: shared_pool,
    })
}

// ---------------------------------------------------------------------------
// Shape recovery
// ---------------------------------------------------------------------------

/// A matched Begin/Next loop pair (offsets validated both directions).
#[derive(Clone, Debug)]
struct LoopPair {
    begin_pc: usize,
    next_pc: usize,
}

#[derive(Clone, Debug)]
struct SuccessorExists {
    begin_pc: usize,
    next_pc: usize,
    r_domain: Register,
}

fn loop_end_of(op: &Opcode) -> Option<i32> {
    match op {
        Opcode::ForallBegin { loop_end, .. }
        | Opcode::ExistsBegin { loop_end, .. }
        | Opcode::ChooseBegin { loop_end, .. }
        | Opcode::SetFilterBegin { loop_end, .. }
        | Opcode::SetBuilderBegin { loop_end, .. } => Some(*loop_end),
        _ => None,
    }
}

fn matching_next_kind(begin: &Opcode, next: &Opcode) -> bool {
    matches!(
        (begin, next),
        (Opcode::ForallBegin { .. }, Opcode::ForallNext { .. })
            | (Opcode::ExistsBegin { .. }, Opcode::ExistsNext { .. })
            | (Opcode::ChooseBegin { .. }, Opcode::ChooseNext { .. })
            | (
                Opcode::SetFilterBegin { .. } | Opcode::SetBuilderBegin { .. },
                Opcode::LoopNext { .. }
            )
    )
}

fn next_back_edge(next: &Opcode) -> Option<i32> {
    match next {
        Opcode::ForallNext { loop_begin, .. }
        | Opcode::ExistsNext { loop_begin, .. }
        | Opcode::ChooseNext { loop_begin, .. }
        | Opcode::LoopNext { loop_begin, .. } => Some(*loop_begin),
        _ => None,
    }
}

fn pair_at(func: &BytecodeFunction, pc: usize) -> Option<LoopPair> {
    let op = &func.instructions[pc];
    let loop_end = loop_end_of(op)?;
    let end = pc.checked_add(usize::try_from(loop_end).ok()?)?;
    let next_pc = end.checked_sub(1)?;
    if next_pc <= pc || next_pc >= func.instructions.len() {
        return None;
    }
    if !matching_next_kind(op, &func.instructions[next_pc]) {
        return None;
    }
    let back = next_back_edge(&func.instructions[next_pc])?;
    if (next_pc as i64) + i64::from(back) != (pc as i64) + 1 {
        return None;
    }
    Some(LoopPair {
        begin_pc: pc,
        next_pc,
    })
}

/// Match every Begin with its Next; fail closed on any malformed pair.
fn quantifier_pairs(func: &BytecodeFunction) -> Option<Vec<LoopPair>> {
    let mut pairs = Vec::new();
    for pc in 0..func.instructions.len() {
        if loop_end_of(&func.instructions[pc]).is_some() {
            pairs.push(pair_at(func, pc)?);
        }
    }
    Some(pairs)
}

/// The unique top-level EXISTS pair whose body writes a primed variable.
fn find_successor_exists(func: &BytecodeFunction, pairs: &[LoopPair]) -> Option<SuccessorExists> {
    let instrs = &func.instructions;
    let mut found: Option<SuccessorExists> = None;
    for pair in pairs {
        let Opcode::ExistsBegin { r_domain, .. } = instrs[pair.begin_pc] else {
            continue;
        };
        let body = (pair.begin_pc + 1)..pair.next_pc;
        let writes_prime = instrs[body]
            .iter()
            .any(|op| matches!(op, Opcode::StoreVar { .. } | Opcode::LoadPrime { .. }));
        if !writes_prime {
            continue;
        }
        if found.is_some() {
            return None; // two successor-producing EXISTS pairs: fail closed
        }
        for other in pairs {
            if other.begin_pc != pair.begin_pc
                && other.begin_pc < pair.begin_pc
                && pair.next_pc < other.next_pc
            {
                return None; // nested inside another loop: fail closed
            }
        }
        found = Some(SuccessorExists {
            begin_pc: pair.begin_pc,
            next_pc: pair.next_pc,
            r_domain,
        });
    }
    found
}

/// Chase `reg` backwards from `before_pc` through `Move` aliases to a
/// `LoadConst`, returning the pooled value.
///
/// SOUNDNESS: each hop requires the register to have exactly ONE writer in
/// the ENTIRE function — a conditionally-reassigned register (two writers on
/// different branch arms) has no single compile-time value and must fail
/// closed rather than silently pick the lexically nearest arm.
fn chase_const_reg(
    func: &BytecodeFunction,
    before_pc: usize,
    reg: Register,
    pool: &ConstantPool,
) -> Option<Value> {
    let mut reg = reg;
    for _ in 0..8 {
        let mut writers = func
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, op)| op.dest_register() == Some(reg));
        let (writer_pc, writer) = writers.next()?;
        if writers.next().is_some() || writer_pc >= before_pc {
            return None;
        }
        match writer {
            Opcode::Move { rs, .. } => reg = *rs,
            Opcode::LoadConst { idx, .. } => return Some(pool.get_value(*idx).clone()),
            _ => return None,
        }
    }
    None
}

/// Deterministically enumerate a finite, materialized set value in canonical
/// `Value::cmp` order. Fails closed on lazy/infinite forms.
fn iter_finite_set(value: &Value) -> Option<Vec<Value>> {
    match value {
        Value::Set(set) => Some(set.iter().cloned().collect()),
        _ => None,
    }
}

/// Per-variable record universes (bit index -> constant record Value) for
/// every proven-closed `RecordSetBitmask` state variable.
fn reconstruct_record_universes(layout: &tla_jit_abi::StateLayout) -> FxHashMap<u16, Vec<Value>> {
    let mut out = FxHashMap::default();
    for var_idx in 0..layout.var_count() {
        let Some(tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::RecordSetBitmask {
            universe,
            is_proven_closed: true,
            ..
        })) = layout.var_layout(var_idx)
        else {
            continue;
        };
        if universe.is_empty() || universe.len() > MAX_UNIVERSE {
            continue;
        }
        let mut records = Vec::with_capacity(universe.len());
        let mut ok = true;
        for key in universe {
            if key.is_empty() || key.len() > MAX_RECORD_FIELDS {
                ok = false;
                break;
            }
            let entries: Vec<(tla_core::NameId, Value)> = key
                .iter()
                .map(|(name, element)| (*name, tla_jit_abi::set_bitmask_element_to_value(*element)))
                .collect();
            records.push(Value::Record(tla_value::RecordValue::from_entries(entries)));
        }
        let Ok(var_idx) = u16::try_from(var_idx) else {
            continue;
        };
        if ok {
            out.insert(var_idx, records);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The walker
// ---------------------------------------------------------------------------

/// Symbolic per-register state.
///
/// `Const` with `emitted: true`, `Emitted`, and `RecordSetVar` registers hold a
/// runtime value in the output program; `MaskSet` / `BuiltScalars` (and
/// `Const` compound values) are compile-time-only.
#[derive(Clone, Debug)]
enum Sym {
    None,
    /// Runtime value present in the output register (same index).
    Emitted,
    /// Compile-time constant. `frontier` snapshots the unpassed forward-branch
    /// targets at definition time; compile-time reads require the snapshot to
    /// be a subset of the reader's frontier (definition dominates read).
    Const {
        value: Value,
        emitted: bool,
        frontier: Vec<usize>,
    },
    /// `LoadVar` of a proven-closed record-set state variable (emitted).
    RecordSetVar(u16),
    /// Compile-time subset of a record-set variable's universe (NOT emitted).
    MaskSet {
        var: u16,
        support: Vec<u32>,
        frontier: Vec<usize>,
    },
    /// `{e(m) : m \in mset}` as compile-time (bit, constant) pairs (NOT
    /// emitted).
    BuiltScalars {
        var: u16,
        items: Vec<(u32, Value)>,
        frontier: Vec<usize>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    /// Original top-level pc (resolved through `orig_labels` at fixup time).
    Orig(usize),
    /// Synthetic label (decision trees, choose chains, unroll iteration ends).
    Synth(usize),
}

#[derive(Clone, Debug)]
enum SubstResult {
    Const(Value),
    Reg(Register),
}

struct Walker<'a> {
    func: &'a BytecodeFunction,
    pool: &'a mut ConstantPool,
    universes: &'a FxHashMap<u16, Vec<Value>>,
    env: &'a RecordSetScalarizeEnv<'a>,
    successor: &'a SuccessorExists,
    witness: Value,

    syms: Vec<Sym>,
    out: Vec<Opcode>,
    jump_fixups: Vec<(usize, Label)>,
    orig_labels: FxHashMap<usize, usize>,
    synth_labels: Vec<Option<usize>>,

    /// (var, bit) -> membership result register (allocated top-down from 255,
    /// emitted by the preamble, live for the whole function).
    memberships: BTreeMap<(u16, u32), Register>,
    membership_order: Vec<(u16, u32, Register)>,
    next_membership_reg: usize,

    /// Scratch registers: bump-allocated bottom-up above the original
    /// register file, released per emission region (stack discipline).
    scratch_top: usize,
    max_scratch: usize,
}

impl<'a> Walker<'a> {
    fn run(&mut self) -> Option<BytecodeFunction> {
        let len = self.func.instructions.len();
        let mut frontier: Vec<usize> = Vec::new();
        self.walk_span(0..len, &mut frontier)?;

        // Membership preamble (runs before everything; pure state reads, so
        // hoisting to the entry is safe and makes every membership register
        // dominate all its uses).
        let preamble = self.build_preamble()?;
        let preamble_len = preamble.len();
        let mut instructions = preamble;
        instructions.append(&mut self.out);
        if instructions.len() > MAX_OUTPUT_INSTRUCTIONS {
            return None;
        }

        for (at, label) in std::mem::take(&mut self.jump_fixups) {
            let target = match label {
                Label::Orig(pc) => *self.orig_labels.get(&pc)?,
                Label::Synth(id) => (*self.synth_labels.get(id)?)?,
            };
            let offset = i32::try_from(target as i64 - at as i64).ok()?;
            if offset <= 0 {
                return None; // only forward jumps are ever emitted
            }
            match &mut instructions[at + preamble_len] {
                Opcode::Jump { offset: o }
                | Opcode::JumpTrue { offset: o, .. }
                | Opcode::JumpFalse { offset: o, .. } => *o = offset,
                _ => return None,
            }
        }

        // Register file: originals + bottom-up scratch + top-down persistent
        // (memberships / var loads). Any overlap fails closed.
        if self.max_scratch > self.next_membership_reg {
            return None;
        }
        Some(BytecodeFunction {
            name: self.func.name.clone(),
            arity: 0,
            max_register: 255,
            instructions,
            max_patched_target: 0,
        })
    }

    // -- emission plumbing ---------------------------------------------------

    fn emit(&mut self, op: Opcode) {
        self.out.push(op);
    }

    fn emit_jump(&mut self, op: Opcode, label: Label) {
        self.jump_fixups.push((self.out.len(), label));
        self.out.push(op);
    }

    fn new_synth_label(&mut self) -> usize {
        self.synth_labels.push(None);
        self.synth_labels.len() - 1
    }

    fn bind_synth_label(&mut self, id: usize) {
        self.synth_labels[id] = Some(self.out.len());
    }

    fn alloc_scratch(&mut self) -> Option<Register> {
        let reg = self.scratch_top;
        if reg > 254 {
            return None;
        }
        self.scratch_top += 1;
        self.max_scratch = self.max_scratch.max(self.scratch_top);
        u8::try_from(reg).ok()
    }

    fn scratch_mark(&self) -> usize {
        self.scratch_top
    }

    fn scratch_release(&mut self, mark: usize) {
        self.scratch_top = mark;
    }

    fn membership_reg(&mut self, var: u16, bit: u32) -> Option<Register> {
        if let Some(reg) = self.memberships.get(&(var, bit)) {
            return Some(*reg);
        }
        if self.next_membership_reg == 0 {
            return None;
        }
        let reg = u8::try_from(self.next_membership_reg).ok()?;
        self.next_membership_reg -= 1;
        self.memberships.insert((var, bit), reg);
        self.membership_order.push((var, bit, reg));
        Some(reg)
    }

    /// Preamble: one `LoadVar` per referenced record-set variable, then per
    /// needed universe bit a `RecordNew` of the key's constant fields plus a
    /// `SetIn` into the pre-assigned membership register.
    fn build_preamble(&mut self) -> Option<Vec<Opcode>> {
        let mut pre = Vec::new();
        if self.membership_order.is_empty() {
            return Some(pre);
        }
        let mut var_regs: BTreeMap<u16, Register> = BTreeMap::new();
        for (var, _, _) in &self.membership_order {
            if !var_regs.contains_key(var) {
                if self.next_membership_reg == 0 {
                    return None;
                }
                let reg = u8::try_from(self.next_membership_reg).ok()?;
                self.next_membership_reg -= 1;
                var_regs.insert(*var, reg);
            }
        }
        let persistent_floor = self.next_membership_reg + 1;
        for (var, reg) in &var_regs {
            pre.push(Opcode::LoadVar {
                rd: *reg,
                var_idx: *var,
            });
        }
        // Field scratch window: right above the walker's high-water scratch
        // mark. These registers are dead once each SetIn retires, and the
        // window must stay below the persistent registers.
        let order = std::mem::take(&mut self.membership_order);
        for (var, bit, m_reg) in &order {
            let records = self.universes.get(var)?;
            let Value::Record(rec) = records.get(*bit as usize)? else {
                return None;
            };
            let fields: Vec<(std::sync::Arc<str>, Value)> = rec
                .iter_str()
                .map(|(name, value)| (name, value.clone()))
                .collect();
            let field_count = fields.len();
            if field_count == 0 || field_count > MAX_RECORD_FIELDS {
                return None;
            }
            if self.max_scratch + field_count + 1 > persistent_floor {
                return None;
            }
            // Field-name pool entries must be consecutive: append a fresh run.
            let mut fields_start: Option<u16> = None;
            for (i, (name, _)) in fields.iter().enumerate() {
                let idx = self.pool.add_value(Value::string(name.as_ref()));
                if i == 0 {
                    fields_start = Some(idx);
                }
            }
            let values_start = self.max_scratch;
            for (i, (_, field_value)) in fields.iter().enumerate() {
                let idx = self.pool.add_value(field_value.clone());
                pre.push(Opcode::LoadConst {
                    rd: u8::try_from(values_start + i).ok()?,
                    idx,
                });
            }
            let rec_reg = u8::try_from(values_start + field_count).ok()?;
            pre.push(Opcode::RecordNew {
                rd: rec_reg,
                fields_start: fields_start?,
                values_start: u8::try_from(values_start).ok()?,
                count: u8::try_from(field_count).ok()?,
            });
            pre.push(Opcode::SetIn {
                rd: *m_reg,
                elem: rec_reg,
                set: var_regs[var],
            });
        }
        self.membership_order = order;
        Some(pre)
    }

    // -- symbolic helpers ------------------------------------------------------

    fn sym(&self, reg: Register) -> &Sym {
        &self.syms[usize::from(reg)]
    }

    fn set_sym(&mut self, reg: Register, sym: Sym) {
        self.syms[usize::from(reg)] = sym;
    }

    /// Register readable by EMITTED code (holds a runtime value in the output).
    fn is_emitted(&self, reg: Register) -> bool {
        matches!(
            self.sym(reg),
            Sym::Emitted | Sym::Const { emitted: true, .. } | Sym::RecordSetVar(_)
        )
    }

    /// Compile-time constant whose definition dominates a reader at `frontier`.
    fn const_at(&self, reg: Register, frontier: &[usize]) -> Option<&Value> {
        match self.sym(reg) {
            Sym::Const {
                value,
                frontier: def,
                ..
            } if def.iter().all(|t| frontier.contains(t)) => Some(value),
            _ => None,
        }
    }

    fn note_forward_target(frontier: &mut Vec<usize>, target: usize) {
        if !frontier.contains(&target) {
            frontier.push(target);
            frontier.sort_unstable();
        }
    }

    // -- main walk ---------------------------------------------------------------

    /// Walk `range`, emitting witness-specialized scalar code.
    ///
    /// `frontier` holds the forward-branch targets not yet passed (dominance
    /// bookkeeping for compile-time constant reads). Jumps may target any pc in
    /// `range` or exactly `range.end` (short-circuit to the range end — the
    /// caller resolves that label: quantifier handlers record the Next pc's
    /// label right after the walk; unrolled iterations retarget it).
    fn walk_span(&mut self, range: Range<usize>, frontier: &mut Vec<usize>) -> Option<()> {
        let mut pc = range.start;
        while pc < range.end {
            frontier.retain(|t| *t > pc);
            self.orig_labels.insert(pc, self.out.len());
            let op = self.func.instructions[pc].clone();
            match op {
                Opcode::LoadVar { rd, var_idx } => {
                    if self.universes.contains_key(&var_idx) {
                        self.set_sym(rd, Sym::RecordSetVar(var_idx));
                    } else {
                        self.set_sym(rd, Sym::Emitted);
                    }
                    self.emit(op);
                    pc += 1;
                }
                Opcode::LoadConst { rd, idx } => {
                    let value = self.pool.get_value(idx).clone();
                    let scalar = matches!(
                        value,
                        Value::Bool(_)
                            | Value::SmallInt(_)
                            | Value::Int(_)
                            | Value::String(_)
                            | Value::ModelValue(_)
                    );
                    if scalar {
                        self.emit(op);
                    }
                    self.set_sym(
                        rd,
                        Sym::Const {
                            value,
                            emitted: scalar,
                            frontier: frontier.clone(),
                        },
                    );
                    pc += 1;
                }
                Opcode::LoadImm { rd, value } => {
                    self.set_sym(
                        rd,
                        Sym::Const {
                            value: Value::SmallInt(value),
                            emitted: true,
                            frontier: frontier.clone(),
                        },
                    );
                    self.emit(op);
                    pc += 1;
                }
                Opcode::LoadBool { rd, value } => {
                    self.set_sym(
                        rd,
                        Sym::Const {
                            value: Value::Bool(value),
                            emitted: true,
                            frontier: frontier.clone(),
                        },
                    );
                    self.emit(op);
                    pc += 1;
                }
                Opcode::Move { rd, rs } => {
                    let src = self.sym(rs).clone();
                    match src {
                        Sym::MaskSet { .. } | Sym::BuiltScalars { .. } => {
                            // Pure symbolic alias; nothing exists at runtime.
                            self.set_sym(rd, src);
                        }
                        Sym::None => return None,
                        _ => {
                            if !self.is_emitted(rs) {
                                return None;
                            }
                            self.set_sym(rd, src);
                            self.emit(op);
                        }
                    }
                    pc += 1;
                }

                // -- scalar ops: emit verbatim ------------------------------
                Opcode::Not { rd, rs } => {
                    if !self.is_emitted(rs) {
                        return None;
                    }
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }
                Opcode::And { rd, r1, r2 }
                | Opcode::Or { rd, r1, r2 }
                | Opcode::Eq { rd, r1, r2 }
                | Opcode::Neq { rd, r1, r2 }
                | Opcode::LtInt { rd, r1, r2 }
                | Opcode::LeInt { rd, r1, r2 }
                | Opcode::GtInt { rd, r1, r2 }
                | Opcode::GeInt { rd, r1, r2 } => {
                    if !self.is_emitted(r1) || !self.is_emitted(r2) {
                        return None;
                    }
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }
                Opcode::NegInt { rd, rs } => {
                    if !self.is_emitted(rs) {
                        return None;
                    }
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }

                // -- control flow ---------------------------------------------
                Opcode::Jump { offset } => {
                    let target = usize::try_from(pc as i64 + i64::from(offset)).ok()?;
                    if target <= pc || target > range.end {
                        return None;
                    }
                    Self::note_forward_target(frontier, target);
                    self.emit_jump(Opcode::Jump { offset: 0 }, Label::Orig(target));
                    pc += 1;
                }
                Opcode::JumpTrue { rs, offset } | Opcode::JumpFalse { rs, offset } => {
                    if !self.is_emitted(rs) {
                        return None;
                    }
                    let target = usize::try_from(pc as i64 + i64::from(offset)).ok()?;
                    if target <= pc || target > range.end {
                        return None;
                    }
                    Self::note_forward_target(frontier, target);
                    let stub = match op {
                        Opcode::JumpTrue { rs, .. } => Opcode::JumpTrue { rs, offset: 0 },
                        _ => Opcode::JumpFalse { rs, offset: 0 },
                    };
                    self.emit_jump(stub, Label::Orig(target));
                    pc += 1;
                }

                // -- quantifiers / loops ----------------------------------------
                Opcode::ForallBegin {
                    rd,
                    r_binding,
                    r_domain,
                    ..
                }
                | Opcode::ExistsBegin {
                    rd,
                    r_binding,
                    r_domain,
                    ..
                } => {
                    let pair = pair_at(self.func, pc)?;
                    let is_forall = matches!(op, Opcode::ForallBegin { .. });
                    if !is_forall && pc == self.successor.begin_pc {
                        // The successor EXISTS: bind THIS walker's witness and
                        // walk the body inline (single-witness semantics; the
                        // per-witness union is the caller's registration of
                        // one native key per witness).
                        self.set_sym(
                            r_binding,
                            Sym::Const {
                                value: self.witness.clone(),
                                emitted: false,
                                frontier: frontier.clone(),
                            },
                        );
                        self.walk_span((pc + 1)..pair.next_pc, frontier)?;
                        frontier.retain(|t| *t > pair.next_pc);
                        self.orig_labels.insert(pair.next_pc, self.out.len());
                        let Opcode::ExistsNext { r_body, .. } =
                            self.func.instructions[pair.next_pc]
                        else {
                            return None;
                        };
                        if !self.is_emitted(r_body) {
                            return None;
                        }
                        self.emit(Opcode::Move { rd, rs: r_body });
                        self.set_sym(rd, Sym::Emitted);
                        pc = pair.next_pc + 1;
                        continue;
                    }
                    match self.sym(r_domain).clone() {
                        Sym::RecordSetVar(var) => {
                            let support: Vec<u32> =
                                (0..self.universes.get(&var)?.len() as u32).collect();
                            self.emit_folded_bool_quantifier(
                                var, &support, &pair, r_binding, rd, is_forall, frontier,
                            )?;
                            pc = pair.next_pc + 1;
                        }
                        Sym::MaskSet {
                            var,
                            support,
                            frontier: def,
                        } => {
                            if !def.iter().all(|t| frontier.contains(t)) {
                                return None;
                            }
                            self.emit_folded_bool_quantifier(
                                var, &support, &pair, r_binding, rd, is_forall, frontier,
                            )?;
                            pc = pair.next_pc + 1;
                        }
                        Sym::Const { .. } => {
                            let value = self.const_at(r_domain, frontier)?.clone();
                            let elements = iter_finite_set(&value)?;
                            if elements.len() > MAX_SUPPORT_BITS {
                                return None;
                            }
                            self.emit_unrolled_const_quantifier(
                                &elements, &pair, r_binding, rd, is_forall, frontier,
                            )?;
                            pc = pair.next_pc + 1;
                        }
                        _ => return None,
                    }
                }
                Opcode::SetFilterBegin {
                    rd,
                    r_binding,
                    r_domain,
                    ..
                } => {
                    let pair = pair_at(self.func, pc)?;
                    let (var, base_support) = match self.sym(r_domain).clone() {
                        Sym::RecordSetVar(var) => {
                            let n = self.universes.get(&var)?.len() as u32;
                            (var, (0..n).collect::<Vec<u32>>())
                        }
                        Sym::MaskSet {
                            var,
                            support,
                            frontier: def,
                        } => {
                            if !def.iter().all(|t| frontier.contains(t)) {
                                return None;
                            }
                            (var, support)
                        }
                        _ => return None,
                    };
                    let body = (pc + 1)..pair.next_pc;
                    let mut support = Vec::new();
                    for bit in base_support {
                        let record = self.universes.get(&var)?[bit as usize].clone();
                        let keep =
                            self.fold_eval_bool_body(body.clone(), r_binding, &record, frontier)?;
                        if keep {
                            support.push(bit);
                        }
                    }
                    if support.len() > MAX_SUPPORT_BITS {
                        return None;
                    }
                    self.set_sym(
                        rd,
                        Sym::MaskSet {
                            var,
                            support,
                            frontier: frontier.clone(),
                        },
                    );
                    pc = pair.next_pc + 1;
                }
                Opcode::SetBuilderBegin {
                    rd,
                    r_binding,
                    r_domain,
                    ..
                } => {
                    let pair = pair_at(self.func, pc)?;
                    let Sym::MaskSet {
                        var,
                        support,
                        frontier: def,
                    } = self.sym(r_domain).clone()
                    else {
                        return None;
                    };
                    if !def.iter().all(|t| frontier.contains(t)) {
                        return None;
                    }
                    let body = (pc + 1)..pair.next_pc;
                    let mut items = Vec::new();
                    for bit in support {
                        let record = self.universes.get(&var)?[bit as usize].clone();
                        let value =
                            self.fold_eval_body_value(body.clone(), r_binding, &record, frontier)?;
                        items.push((bit, value));
                    }
                    self.set_sym(
                        rd,
                        Sym::BuiltScalars {
                            var,
                            items,
                            frontier: frontier.clone(),
                        },
                    );
                    pc = pair.next_pc + 1;
                }
                Opcode::ChooseBegin { .. } => {
                    pc = self.emit_choose(pc, frontier)?;
                }

                Opcode::CallExternal {
                    rd,
                    name_idx,
                    args_start,
                    argc,
                    ..
                } => {
                    if argc != 1 {
                        return None;
                    }
                    let Sym::BuiltScalars {
                        var,
                        items,
                        frontier: def,
                    } = self.sym(args_start).clone()
                    else {
                        return None;
                    };
                    if !def.iter().all(|t| frontier.contains(t)) {
                        return None;
                    }
                    let name = match self.pool.get_value(name_idx) {
                        Value::String(s) => s.to_string(),
                        _ => return None,
                    };
                    self.emit_external_table(rd, &name, var, &items)?;
                    pc += 1;
                }

                // -- successor construction / remaining verbatim ops ------------
                Opcode::RecordNew {
                    rd,
                    values_start,
                    count,
                    ..
                } => {
                    for i in 0..count {
                        if !self.is_emitted(values_start.checked_add(i)?) {
                            return None;
                        }
                    }
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }
                Opcode::SetEnum { rd, start, count } => {
                    for i in 0..count {
                        if !self.is_emitted(start.checked_add(i)?) {
                            return None;
                        }
                    }
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }
                Opcode::SetUnion { rd, r1, r2 } => {
                    if !self.is_emitted(r1) || !self.is_emitted(r2) {
                        return None;
                    }
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }
                Opcode::SetIn { rd, elem, set } => {
                    if !self.is_emitted(elem) || !self.is_emitted(set) {
                        return None;
                    }
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }
                Opcode::RecordGet { rd, rs, .. } => {
                    if !self.is_emitted(rs) {
                        return None;
                    }
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }
                Opcode::StoreVar { rs, .. } => {
                    if !self.is_emitted(rs) {
                        return None;
                    }
                    self.emit(op);
                    pc += 1;
                }
                Opcode::Unchanged { rd, .. } => {
                    self.set_sym(rd, Sym::Emitted);
                    self.emit(op);
                    pc += 1;
                }
                Opcode::Ret { rs } => {
                    if !self.is_emitted(rs) {
                        return None;
                    }
                    self.emit(op);
                    pc += 1;
                }
                _ => return None,
            }
        }
        Some(())
    }

    // -- folded boolean quantifier over a record-set support ---------------------

    #[allow(clippy::too_many_arguments)]
    fn emit_folded_bool_quantifier(
        &mut self,
        var: u16,
        support: &[u32],
        pair: &LoopPair,
        r_binding: Register,
        rd: Register,
        is_forall: bool,
        frontier: &[usize],
    ) -> Option<()> {
        let body = (pair.begin_pc + 1)..pair.next_pc;
        // Fold the body per key; keep the keys that DRIVE the runtime value:
        // forall -> the body-false keys (all must be absent), exists -> the
        // body-true keys (at least one must be present).
        let mut driving: Vec<u32> = Vec::new();
        for &bit in support {
            let record = self.universes.get(&var)?[bit as usize].clone();
            let result = self.fold_eval_bool_body(body.clone(), r_binding, &record, frontier)?;
            if result != is_forall {
                driving.push(bit);
            }
        }
        if driving.len() > MAX_SUPPORT_BITS {
            return None;
        }
        let mark = self.scratch_mark();
        if driving.is_empty() {
            self.emit(Opcode::LoadBool {
                rd,
                value: is_forall,
            });
        } else {
            let mut acc: Option<Register> = None;
            for bit in driving {
                let m = self.membership_reg(var, bit)?;
                let term = if is_forall {
                    let t = self.alloc_scratch()?;
                    self.emit(Opcode::Not { rd: t, rs: m });
                    t
                } else {
                    m
                };
                acc = Some(match acc {
                    None => term,
                    Some(prev) => {
                        let t = self.alloc_scratch()?;
                        if is_forall {
                            self.emit(Opcode::And {
                                rd: t,
                                r1: prev,
                                r2: term,
                            });
                        } else {
                            self.emit(Opcode::Or {
                                rd: t,
                                r1: prev,
                                r2: term,
                            });
                        }
                        t
                    }
                });
            }
            self.emit(Opcode::Move { rd, rs: acc? });
        }
        self.scratch_release(mark);
        self.set_sym(rd, Sym::Emitted);
        Some(())
    }

    // -- unrolled boolean quantifier over a constant finite set -------------------

    fn emit_unrolled_const_quantifier(
        &mut self,
        elements: &[Value],
        pair: &LoopPair,
        r_binding: Register,
        rd: Register,
        is_forall: bool,
        frontier: &mut Vec<usize>,
    ) -> Option<()> {
        let body = (pair.begin_pc + 1)..pair.next_pc;
        // Boolean position only: the body must not write successor state.
        if self.func.instructions[body.clone()]
            .iter()
            .any(|op| matches!(op, Opcode::StoreVar { .. } | Opcode::LoadPrime { .. }))
        {
            return None;
        }
        let (next_rd, r_body) = match self.func.instructions[pair.next_pc] {
            Opcode::ForallNext { rd, r_body, .. } | Opcode::ExistsNext { rd, r_body, .. } => {
                (rd, r_body)
            }
            _ => return None,
        };
        if next_rd != rd {
            return None;
        }
        if elements.is_empty() {
            self.emit(Opcode::LoadBool {
                rd,
                value: is_forall,
            });
            self.set_sym(rd, Sym::Emitted);
            return Some(());
        }
        let mark = self.scratch_mark();
        let acc = self.alloc_scratch()?;
        self.emit(Opcode::LoadBool {
            rd: acc,
            value: is_forall,
        });
        for element in elements {
            self.set_sym(
                r_binding,
                Sym::Const {
                    value: element.clone(),
                    emitted: false,
                    frontier: frontier.clone(),
                },
            );
            self.walk_iteration_body(body.clone(), frontier)?;
            if !self.is_emitted(r_body) {
                return None;
            }
            if is_forall {
                self.emit(Opcode::And {
                    rd: acc,
                    r1: acc,
                    r2: r_body,
                });
            } else {
                self.emit(Opcode::Or {
                    rd: acc,
                    r1: acc,
                    r2: r_body,
                });
            }
        }
        self.emit(Opcode::Move { rd, rs: acc });
        self.set_sym(rd, Sym::Emitted);
        self.scratch_release(mark);
        Some(())
    }

    /// Walk one unrolled quantifier-body iteration. Labels for body pcs are
    /// iteration-local: every fixup this iteration emits that targets a body
    /// pc (or the body end, i.e. the quantifier Next) is retargeted to a
    /// synthetic label bound within THIS iteration's emission, then the outer
    /// label table entries are restored.
    fn walk_iteration_body(&mut self, body: Range<usize>, frontier: &[usize]) -> Option<()> {
        let saved: Vec<(usize, Option<usize>)> = body
            .clone()
            .map(|p| (p, self.orig_labels.get(&p).copied()))
            .collect();
        let fixup_mark = self.jump_fixups.len();
        let mut inner_frontier = frontier.to_vec();
        self.walk_span(body.clone(), &mut inner_frontier)?;
        let end_index = self.out.len();

        // Retarget this iteration's intra-body fixups to synthetic labels.
        let mut retargets: Vec<(usize, usize)> = Vec::new();
        for (i, (_, label)) in self.jump_fixups.iter().enumerate().skip(fixup_mark) {
            if let Label::Orig(p) = label {
                if body.contains(p) {
                    let at = *self.orig_labels.get(p)?;
                    retargets.push((i, at));
                } else if *p == body.end {
                    retargets.push((i, end_index));
                }
            }
        }
        for (i, at) in retargets {
            let id = self.synth_labels.len();
            self.synth_labels.push(Some(at));
            self.jump_fixups[i].1 = Label::Synth(id);
        }
        for (p, old) in saved {
            match old {
                Some(v) => {
                    self.orig_labels.insert(p, v);
                }
                None => {
                    self.orig_labels.remove(&p);
                }
            }
        }
        Some(())
    }

    // -- CHOOSE -----------------------------------------------------------------

    /// Emit the first-match chain for
    /// `CHOOSE m \in <mask> : <body>(m, runtime scalars)` followed by exactly
    /// one immediate `RecordGet(field)` consumer. Returns the pc after the
    /// consumed `RecordGet`.
    fn emit_choose(&mut self, pc: usize, frontier: &mut Vec<usize>) -> Option<usize> {
        let pair = pair_at(self.func, pc)?;
        let Opcode::ChooseBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } = self.func.instructions[pc]
        else {
            return None;
        };
        let Sym::MaskSet {
            var,
            support,
            frontier: def,
        } = self.sym(r_domain).clone()
        else {
            return None;
        };
        if !def.iter().all(|t| frontier.contains(t)) {
            return None;
        }

        // Single immediate RecordGet consumer of the chosen record.
        let get_pc = pair.next_pc + 1;
        let Opcode::RecordGet {
            rd: get_rd,
            rs: get_rs,
            field_idx,
        } = *self.func.instructions.get(get_pc)?
        else {
            return None;
        };
        if get_rs != rd {
            return None;
        }
        for (other_pc, other) in self.func.instructions.iter().enumerate() {
            if other_pc == get_pc || other_pc == pc || other_pc == pair.next_pc {
                continue;
            }
            let mut uses = false;
            for_each_source_register(other, |r| {
                if r == rd {
                    uses = true;
                }
            });
            if uses {
                return None;
            }
        }
        let field_name = tla_core::NameId(self.pool.get_field_id(field_idx));

        // The extracted value may flow through an immediate Move-alias chain
        // (e.g. `RecordGet r52; Move r46, r52`). Follow it: the sentinel
        // bypass below must replicate it so the SEND reads the sentinel from
        // the register it actually consumes.
        let mut alias_chain: Vec<(Register, Register)> = Vec::new();
        let mut value_regs: Vec<Register> = vec![get_rd];
        let mut alias_end = get_pc + 1;
        while let Some(Opcode::Move { rd: a_rd, rs: a_rs }) =
            self.func.instructions.get(alias_end).copied()
        {
            if a_rs != *value_regs.last()? {
                break;
            }
            alias_chain.push((a_rd, a_rs));
            value_regs.push(a_rd);
            alias_end += 1;
        }

        // Bypass target for the no-witness path: the successor-construction
        // region starts right after the LAST pure guard (`JumpFalse` targeting
        // the successor ExistsNext). A CHOOSE with no witness is a hard
        // interpreter error; loading the sentinel and jumping THROUGH the
        // successor union (bypassing the remaining pure guards) makes the
        // strict record-set enum-fold surface `FallbackNeeded` at runtime, and
        // the interpreter then owns the state and reproduces the exact error
        // semantics.
        let succ = self.successor;
        if pc <= succ.begin_pc || pair.next_pc >= succ.next_pc {
            return None; // the choose must live inside the successor body
        }
        let mut send_start: Option<usize> = None;
        for p in alias_end..succ.next_pc {
            if let Opcode::JumpFalse { offset, .. } = self.func.instructions[p] {
                let target = usize::try_from(p as i64 + i64::from(offset)).ok()?;
                if target == succ.next_pc {
                    send_start = Some(p + 1);
                }
            }
        }
        let send_start = send_start?;
        // Between the alias chain and the send: pure guards only, and none may
        // read the extracted value's register (or any of its aliases) — the
        // bypass skips this whole region.
        for p in alias_end..send_start {
            let other = &self.func.instructions[p];
            if matches!(other, Opcode::StoreVar { .. } | Opcode::LoadPrime { .. }) {
                return None;
            }
            let mut reads_val = false;
            for_each_source_register(other, |r| {
                if value_regs.contains(&r) {
                    reads_val = true;
                }
            });
            if reads_val {
                return None;
            }
        }

        // Candidate order: the interpreter's CHOOSE iterates its domain in
        // TLC-normalized order (tlc_cmp), NOT the canonical Value::cmp order
        // the universe bits follow.
        let records: Vec<(u32, Value)> = support
            .iter()
            .map(|bit| {
                (
                    *bit,
                    self.universes.get(&var).unwrap()[*bit as usize].clone(),
                )
            })
            .collect();
        let ordered = tlc_order_candidates(&records)?;

        let body = (pc + 1)..pair.next_pc;
        let done = self.new_synth_label();
        let mark = self.scratch_mark();
        for (bit, record) in &ordered {
            let next_candidate = self.new_synth_label();
            let m = self.membership_reg(var, *bit)?;
            self.emit_jump(
                Opcode::JumpFalse { rs: m, offset: 0 },
                Label::Synth(next_candidate),
            );
            let pred = self.emit_body_substituted(body.clone(), r_binding, record, frontier)?;
            match pred {
                SubstResult::Const(Value::Bool(true)) => {}
                SubstResult::Const(Value::Bool(false)) => {
                    // This candidate can never match; drop it entirely. (The
                    // membership jump above becomes a jump-to-jump; harmless.)
                    self.bind_synth_label(next_candidate);
                    continue;
                }
                SubstResult::Const(_) => return None,
                SubstResult::Reg(r) => {
                    self.emit_jump(
                        Opcode::JumpFalse { rs: r, offset: 0 },
                        Label::Synth(next_candidate),
                    );
                }
            }
            // Match: load the consumed field of this candidate.
            let Value::Record(rec) = record else {
                return None;
            };
            let field_value = rec.get_by_id(field_name)?.clone();
            let idx = self.pool.add_value(field_value);
            self.emit(Opcode::LoadConst { rd: get_rd, idx });
            self.emit_jump(Opcode::Jump { offset: 0 }, Label::Synth(done));
            self.bind_synth_label(next_candidate);
        }
        // No witness: sentinel, replicate the alias chain, then bypass the
        // remaining pure guards straight to the successor construction.
        let sentinel_idx = self.pool.add_value(Value::string(CHOOSE_SENTINEL));
        self.emit(Opcode::LoadConst {
            rd: get_rd,
            idx: sentinel_idx,
        });
        for (a_rd, a_rs) in &alias_chain {
            self.emit(Opcode::Move {
                rd: *a_rd,
                rs: *a_rs,
            });
        }
        self.emit_jump(Opcode::Jump { offset: 0 }, Label::Orig(send_start));
        Self::note_forward_target(frontier, send_start);
        self.bind_synth_label(done);
        self.scratch_release(mark);
        self.set_sym(get_rd, Sym::Emitted);
        Some(get_pc + 1)
    }

    // -- external call table -------------------------------------------------------

    /// Emit `rd := f({v : (bit, v) present})` as a presence-bit decision tree
    /// over the DISTINCT constant values, each leaf a plan-time evaluation of
    /// the external on that exact value subset.
    fn emit_external_table(
        &mut self,
        rd: Register,
        name: &str,
        var: u16,
        items: &[(u32, Value)],
    ) -> Option<()> {
        let mut distinct: Vec<Value> = Vec::new();
        for (_, v) in items {
            if !distinct.contains(v) {
                distinct.push(v.clone());
            }
        }
        distinct.sort();
        if distinct.len() > MAX_EXTERNAL_DISTINCT_VALUES {
            return None;
        }

        let mark = self.scratch_mark();
        // presence[i] = OR over the bits whose built value is distinct[i].
        let mut presence: Vec<Register> = Vec::with_capacity(distinct.len());
        for v in &distinct {
            let mut acc: Option<Register> = None;
            for (bit, item_v) in items {
                if item_v != v {
                    continue;
                }
                let m = self.membership_reg(var, *bit)?;
                acc = Some(match acc {
                    None => m,
                    Some(prev) => {
                        let t = self.alloc_scratch()?;
                        self.emit(Opcode::Or {
                            rd: t,
                            r1: prev,
                            r2: m,
                        });
                        t
                    }
                });
            }
            presence.push(acc?);
        }

        // Plan-time table: evaluate the external on every distinct-value
        // subset (the argument is a SET, so only the distinct values matter).
        let subset_count = 1usize << distinct.len();
        let mut table: Vec<Value> = Vec::with_capacity(subset_count);
        for mask in 0..subset_count {
            let elems: Vec<Value> = distinct
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1usize << i) != 0)
                .map(|(_, v)| v.clone())
                .collect();
            let arg = Value::set(elems);
            let result = (self.env.eval_pure_op)(name, &[arg])?;
            if !matches!(
                result,
                Value::Bool(_)
                    | Value::SmallInt(_)
                    | Value::Int(_)
                    | Value::String(_)
                    | Value::ModelValue(_)
            ) {
                return None;
            }
            table.push(result);
        }

        let done = self.new_synth_label();
        self.emit_table_tree(rd, &presence, &table, 0, 0, done)?;
        self.bind_synth_label(done);
        self.scratch_release(mark);
        self.set_sym(rd, Sym::Emitted);
        Some(())
    }

    fn emit_table_tree(
        &mut self,
        rd: Register,
        presence: &[Register],
        table: &[Value],
        depth: usize,
        mask: usize,
        done: usize,
    ) -> Option<()> {
        if depth == presence.len() {
            let idx = self.pool.add_value(table[mask].clone());
            self.emit(Opcode::LoadConst { rd, idx });
            self.emit_jump(Opcode::Jump { offset: 0 }, Label::Synth(done));
            return Some(());
        }
        let absent = self.new_synth_label();
        self.emit_jump(
            Opcode::JumpFalse {
                rs: presence[depth],
                offset: 0,
            },
            Label::Synth(absent),
        );
        self.emit_table_tree(
            rd,
            presence,
            table,
            depth + 1,
            mask | (1usize << depth),
            done,
        )?;
        self.bind_synth_label(absent);
        self.emit_table_tree(rd, presence, table, depth + 1, mask, done)?;
        Some(())
    }

    // -- constant folding of pure per-record bodies -----------------------------------

    fn fold_eval_bool_body(
        &self,
        body: Range<usize>,
        r_binding: Register,
        record: &Value,
        frontier: &[usize],
    ) -> Option<bool> {
        match self.fold_eval_body_value(body, r_binding, record, frontier)? {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// Constant-evaluate a quantifier/filter/builder body with
    /// `r_binding := record`, following intra-body short-circuit jumps exactly
    /// as the VM would, and return the value the Next opcode reads. Any
    /// non-constant operand, unsupported opcode, jump escaping the body, or
    /// missing record field (an interpreter type error) fails closed.
    fn fold_eval_body_value(
        &self,
        body: Range<usize>,
        r_binding: Register,
        record: &Value,
        frontier: &[usize],
    ) -> Option<Value> {
        let instrs = &self.func.instructions;
        let next_pc = body.end;
        let r_body = match &instrs[next_pc] {
            Opcode::ForallNext { r_body, .. }
            | Opcode::ExistsNext { r_body, .. }
            | Opcode::ChooseNext { r_body, .. }
            | Opcode::LoopNext { r_body, .. } => *r_body,
            _ => return None,
        };
        let mut env: FxHashMap<Register, Value> = FxHashMap::default();
        env.insert(r_binding, record.clone());
        let read = |env: &FxHashMap<Register, Value>, r: Register| -> Option<Value> {
            if let Some(v) = env.get(&r) {
                return Some(v.clone());
            }
            self.const_at(r, frontier).cloned()
        };
        let mut pc = body.start;
        let mut steps = 0usize;
        while pc < next_pc {
            steps += 1;
            if steps > 4096 {
                return None;
            }
            match &instrs[pc] {
                Opcode::LoadConst { rd, idx } => {
                    env.insert(*rd, self.pool.get_value(*idx).clone());
                }
                Opcode::LoadImm { rd, value } => {
                    env.insert(*rd, Value::SmallInt(*value));
                }
                Opcode::LoadBool { rd, value } => {
                    env.insert(*rd, Value::Bool(*value));
                }
                Opcode::Move { rd, rs } => {
                    let v = read(&env, *rs)?;
                    env.insert(*rd, v);
                }
                Opcode::Not { rd, rs } => {
                    let Value::Bool(b) = read(&env, *rs)? else {
                        return None;
                    };
                    env.insert(*rd, Value::Bool(!b));
                }
                Opcode::And { rd, r1, r2 } | Opcode::Or { rd, r1, r2 } => {
                    let (Value::Bool(a), Value::Bool(b)) = (read(&env, *r1)?, read(&env, *r2)?)
                    else {
                        return None;
                    };
                    let is_and = matches!(instrs[pc], Opcode::And { .. });
                    env.insert(*rd, Value::Bool(if is_and { a && b } else { a || b }));
                }
                Opcode::Eq { rd, r1, r2 } => {
                    let (a, b) = (read(&env, *r1)?, read(&env, *r2)?);
                    env.insert(*rd, Value::Bool(a == b));
                }
                Opcode::Neq { rd, r1, r2 } => {
                    let (a, b) = (read(&env, *r1)?, read(&env, *r2)?);
                    env.insert(*rd, Value::Bool(a != b));
                }
                Opcode::NegInt { rd, rs } => {
                    let Value::SmallInt(v) = read(&env, *rs)? else {
                        return None;
                    };
                    env.insert(*rd, Value::SmallInt(v.checked_neg()?));
                }
                Opcode::LtInt { rd, r1, r2 }
                | Opcode::LeInt { rd, r1, r2 }
                | Opcode::GtInt { rd, r1, r2 }
                | Opcode::GeInt { rd, r1, r2 } => {
                    let (Value::SmallInt(a), Value::SmallInt(b)) =
                        (read(&env, *r1)?, read(&env, *r2)?)
                    else {
                        return None;
                    };
                    let result = match &instrs[pc] {
                        Opcode::LtInt { .. } => a < b,
                        Opcode::LeInt { .. } => a <= b,
                        Opcode::GtInt { .. } => a > b,
                        _ => a >= b,
                    };
                    env.insert(*rd, Value::Bool(result));
                }
                Opcode::RecordGet { rd, rs, field_idx } => {
                    let record = read(&env, *rs)?;
                    let Value::Record(rec) = &record else {
                        return None;
                    };
                    let field_id = tla_core::NameId(self.pool.get_field_id(*field_idx));
                    // Missing field = interpreter type error on any state where
                    // this key is present: fail the whole fold closed.
                    let v = rec.get_by_id(field_id)?.clone();
                    env.insert(*rd, v);
                }
                Opcode::SetIn { rd, elem, set } => {
                    let elem = read(&env, *elem)?;
                    let set = read(&env, *set)?;
                    let elements = iter_finite_set(&set)?;
                    env.insert(*rd, Value::Bool(elements.contains(&elem)));
                }
                Opcode::Jump { offset } => {
                    let target = usize::try_from(pc as i64 + i64::from(*offset)).ok()?;
                    if target <= pc || target > next_pc {
                        return None;
                    }
                    pc = target;
                    continue;
                }
                Opcode::JumpTrue { rs, offset } | Opcode::JumpFalse { rs, offset } => {
                    let Value::Bool(b) = read(&env, *rs)? else {
                        return None;
                    };
                    let jump_on = matches!(instrs[pc], Opcode::JumpTrue { .. });
                    if b == jump_on {
                        let target = usize::try_from(pc as i64 + i64::from(*offset)).ok()?;
                        if target <= pc || target > next_pc {
                            return None;
                        }
                        pc = target;
                        continue;
                    }
                }
                _ => return None,
            }
            pc += 1;
        }
        env.get(&r_body).cloned()
    }

    // -- substituted emission (the CHOOSE predicate) ------------------------------------

    /// Process a small pure body with the binding substituted by a constant
    /// record: constant subexpressions fold; comparisons touching runtime
    /// registers are emitted (constants materialized into scratch). Returns
    /// the body result as a constant or an emitted register.
    fn emit_body_substituted(
        &mut self,
        body: Range<usize>,
        r_binding: Register,
        record: &Value,
        frontier: &[usize],
    ) -> Option<SubstResult> {
        let instrs = &self.func.instructions;
        let next_pc = body.end;
        let Opcode::ChooseNext { r_body, .. } = instrs[next_pc] else {
            return None;
        };
        let mut env: FxHashMap<Register, SubstResult> = FxHashMap::default();
        env.insert(r_binding, SubstResult::Const(record.clone()));
        for pc in body {
            match instrs[pc].clone() {
                Opcode::LoadConst { rd, idx } => {
                    env.insert(rd, SubstResult::Const(self.pool.get_value(idx).clone()));
                }
                Opcode::LoadImm { rd, value } => {
                    env.insert(rd, SubstResult::Const(Value::SmallInt(value)));
                }
                Opcode::LoadBool { rd, value } => {
                    env.insert(rd, SubstResult::Const(Value::Bool(value)));
                }
                Opcode::Move { rd, rs } => {
                    let v = self.subst_read(&env, rs, frontier)?;
                    env.insert(rd, v);
                }
                Opcode::RecordGet { rd, rs, field_idx } => {
                    let source = self.subst_read(&env, rs, frontier)?;
                    let SubstResult::Const(Value::Record(rec)) = &source else {
                        return None;
                    };
                    let field_id = tla_core::NameId(self.pool.get_field_id(field_idx));
                    let v = rec.get_by_id(field_id)?.clone();
                    env.insert(rd, SubstResult::Const(v));
                }
                Opcode::Eq { rd, r1, r2 } | Opcode::Neq { rd, r1, r2 } => {
                    let a = self.subst_read(&env, r1, frontier)?;
                    let b = self.subst_read(&env, r2, frontier)?;
                    let negate = matches!(instrs[pc], Opcode::Neq { .. });
                    match (a, b) {
                        (SubstResult::Const(a), SubstResult::Const(b)) => {
                            env.insert(rd, SubstResult::Const(Value::Bool((a == b) != negate)));
                        }
                        (a, b) => {
                            let ra = self.subst_materialize(a)?;
                            let rb = self.subst_materialize(b)?;
                            let t = self.alloc_scratch()?;
                            if negate {
                                self.emit(Opcode::Neq {
                                    rd: t,
                                    r1: ra,
                                    r2: rb,
                                });
                            } else {
                                self.emit(Opcode::Eq {
                                    rd: t,
                                    r1: ra,
                                    r2: rb,
                                });
                            }
                            env.insert(rd, SubstResult::Reg(t));
                        }
                    }
                }
                _ => return None,
            }
        }
        env.get(&r_body).cloned()
    }

    fn subst_read(
        &self,
        env: &FxHashMap<Register, SubstResult>,
        reg: Register,
        frontier: &[usize],
    ) -> Option<SubstResult> {
        if let Some(v) = env.get(&reg) {
            return Some(v.clone());
        }
        if let Some(v) = self.const_at(reg, frontier) {
            return Some(SubstResult::Const(v.clone()));
        }
        if self.is_emitted(reg) {
            return Some(SubstResult::Reg(reg));
        }
        None
    }

    fn subst_materialize(&mut self, value: SubstResult) -> Option<Register> {
        match value {
            SubstResult::Reg(r) => Some(r),
            SubstResult::Const(v) => {
                let t = self.alloc_scratch()?;
                match v {
                    Value::SmallInt(i) => self.emit(Opcode::LoadImm { rd: t, value: i }),
                    Value::Bool(b) => self.emit(Opcode::LoadBool { rd: t, value: b }),
                    other @ (Value::String(_) | Value::ModelValue(_) | Value::Int(_)) => {
                        let idx = self.pool.add_value(other);
                        self.emit(Opcode::LoadConst { rd: t, idx });
                    }
                    _ => return None,
                }
                Some(t)
            }
        }
    }
}

/// Order candidate `(bit, record)` pairs by the interpreter's CHOOSE iteration
/// order: TLC-normalized (`tlc_cmp`) order over the record values.
fn tlc_order_candidates(records: &[(u32, Value)]) -> Option<Vec<(u32, Value)>> {
    if records.is_empty() {
        return Some(Vec::new());
    }
    let set = Value::set(records.iter().map(|(_, r)| r.clone()));
    let mut ordered = Vec::with_capacity(records.len());
    for value in set.iter_set_tlc_normalized().ok()? {
        let bit = records
            .iter()
            .find(|(_, r)| *r == value)
            .map(|(bit, _)| *bit)?;
        ordered.push((bit, value));
    }
    if ordered.len() != records.len() {
        return None; // duplicate records in one universe: malformed layout
    }
    Some(ordered)
}

/// Enumerate every source register an opcode reads (local mirror of the
/// normalize-module helper, restricted to the opcodes this module encounters;
/// the catch-all reports nothing, which is safe because unknown opcodes fail
/// the walk closed before any use-analysis result matters).
fn for_each_source_register(op: &Opcode, mut f: impl FnMut(Register)) {
    match *op {
        Opcode::Move { rs, .. }
        | Opcode::Not { rs, .. }
        | Opcode::NegInt { rs, .. }
        | Opcode::StoreVar { rs, .. }
        | Opcode::Ret { rs } => f(rs),
        Opcode::And { r1, r2, .. }
        | Opcode::Or { r1, r2, .. }
        | Opcode::Eq { r1, r2, .. }
        | Opcode::Neq { r1, r2, .. }
        | Opcode::LtInt { r1, r2, .. }
        | Opcode::LeInt { r1, r2, .. }
        | Opcode::GtInt { r1, r2, .. }
        | Opcode::GeInt { r1, r2, .. }
        | Opcode::SetUnion { r1, r2, .. } => {
            f(r1);
            f(r2);
        }
        Opcode::SetIn { elem, set, .. } => {
            f(elem);
            f(set);
        }
        Opcode::RecordGet { rs, .. } => f(rs),
        Opcode::RecordNew {
            values_start,
            count,
            ..
        } => {
            for i in 0..count {
                f(values_start.wrapping_add(i));
            }
        }
        Opcode::SetEnum { start, count, .. } => {
            for i in 0..count {
                f(start.wrapping_add(i));
            }
        }
        Opcode::JumpTrue { rs, .. } | Opcode::JumpFalse { rs, .. } => f(rs),
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
        Opcode::CallExternal {
            args_start, argc, ..
        } => {
            for i in 0..argc {
                f(args_start.wrapping_add(i));
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::intern_name;
    use tla_jit_abi::{CompoundLayout, SetBitmaskElement, StateLayout, VarLayout};
    use tla_value::RecordValue;

    fn rec(fields: &[(&str, Value)]) -> Value {
        Value::Record(RecordValue::from_entries(
            fields
                .iter()
                .map(|(name, value)| (intern_name(name), value.clone()))
                .collect(),
        ))
    }

    fn universe_layout(records: &[Value]) -> StateLayout {
        let universe: Vec<Vec<(tla_core::NameId, SetBitmaskElement)>> = records
            .iter()
            .map(|record| {
                let Value::Record(rec) = record else {
                    panic!("universe entry must be a record");
                };
                rec.iter_str()
                    .map(|(name, value)| {
                        let element = match value {
                            Value::SmallInt(v) => SetBitmaskElement::Int(*v),
                            Value::Bool(v) => SetBitmaskElement::Bool(*v),
                            Value::String(s) => SetBitmaskElement::String(intern_name(s.as_ref())),
                            other => panic!("unsupported field {other:?}"),
                        };
                        (intern_name(name.as_ref()), element)
                    })
                    .collect()
            })
            .collect();
        let slot_count = records.len().div_ceil(64);
        StateLayout::new(vec![VarLayout::Compound(
            CompoundLayout::RecordSetBitmask {
                universe,
                slot_count,
                is_proven_closed: true,
            },
        )])
    }

    /// Assemble the synthetic Phase2a-shaped transformed action over state var
    /// 0 (`msgs`):
    ///
    /// ```text
    ///   guard1: \A m \in msgs : ~(m.t = "p2")
    ///   \E MS \in <majority-const> :
    ///     mset  = {m \in msgs : m.t = "p1" /\ m.a \in MS}
    ///     maxb  = MaxOr({m.b : m \in mset})
    ///     val   = IF maxb = -1 THEN "AB" ELSE (CHOOSE m \in mset : m.b = maxb).v
    ///     /\ \A ac \in MS : \E m \in mset : m.a = ac
    ///     /\ msgs' = msgs \cup {[t |-> "p2s", v |-> val]}   (StoreVar form)
    /// ```
    fn build_phase2a_like(majority: Value) -> (BytecodeFunction, ConstantPool) {
        use Opcode::*;
        let mut pool = ConstantPool::new();
        let c_p2 = pool.add_value(Value::string("p2"));
        let c_maj = pool.add_value(majority);
        let c_p1 = pool.add_value(Value::string("p1"));
        let c_name = pool.add_value(Value::string("MaxOr"));
        let c_ab = pool.add_value(Value::string("AB"));
        let f_t_name = pool.add_value(Value::string("t"));
        let f_v_name = pool.add_value(Value::string("v"));
        let c_p2s = pool.add_value(Value::string("p2s"));
        assert_eq!(f_t_name + 1, f_v_name);
        let f_t = pool.add_field_id(intern_name("t").0);
        let f_a = pool.add_field_id(intern_name("a").0);
        let f_b = pool.add_field_id(intern_name("b").0);
        let f_v = pool.add_field_id(intern_name("v").0);

        let instructions = vec![
            // pc 0..8: guard1 = \A m \in msgs : ~(m.t = "p2")
            LoadVar { rd: 0, var_idx: 0 },
            ForallBegin {
                rd: 1,
                r_binding: 2,
                r_domain: 0,
                loop_end: 6,
            },
            RecordGet {
                rd: 3,
                rs: 2,
                field_idx: f_t,
            },
            LoadConst { rd: 4, idx: c_p2 },
            Eq {
                rd: 5,
                r1: 3,
                r2: 4,
            },
            Not { rd: 6, rs: 5 },
            ForallNext {
                rd: 1,
                r_binding: 2,
                r_body: 6,
                loop_begin: -4,
            },
            // pc 7: bail when guard1 false -> pc 49 (Move to r0)
            JumpFalse { rs: 1, offset: 42 },
            // pc 8: Majority const, pc 9: \E MS
            LoadConst { rd: 7, idx: c_maj },
            ExistsBegin {
                rd: 8,
                r_binding: 9,
                r_domain: 7,
                loop_end: 39,
            },
            // pc 10..20: mset = {m \in msgs : m.t = "p1" /\ m.a \in MS}
            LoadVar { rd: 10, var_idx: 0 },
            SetFilterBegin {
                rd: 11,
                r_binding: 12,
                r_domain: 10,
                loop_end: 9,
            },
            RecordGet {
                rd: 13,
                rs: 12,
                field_idx: f_t,
            },
            LoadConst { rd: 14, idx: c_p1 },
            Eq {
                rd: 15,
                r1: 13,
                r2: 14,
            },
            Move { rd: 16, rs: 15 },
            JumpFalse { rs: 16, offset: 3 },
            RecordGet {
                rd: 17,
                rs: 12,
                field_idx: f_a,
            },
            SetIn {
                rd: 16,
                elem: 17,
                set: 9,
            },
            LoopNext {
                r_binding: 12,
                r_body: 16,
                loop_begin: -7,
            },
            // pc 20..23: bals = {m.b : m \in mset}
            SetBuilderBegin {
                rd: 18,
                r_binding: 19,
                r_domain: 11,
                loop_end: 3,
            },
            RecordGet {
                rd: 20,
                rs: 19,
                field_idx: f_b,
            },
            LoopNext {
                r_binding: 19,
                r_body: 20,
                loop_begin: -1,
            },
            // pc 23: maxb = MaxOr(bals)
            CallExternal {
                rd: 21,
                name_idx: c_name,
                args_start: 18,
                argc: 1,
                self_recursive: false,
            },
            // pc 24..27: maxb = -1 ?
            LoadImm { rd: 22, value: 1 },
            NegInt { rd: 23, rs: 22 },
            Eq {
                rd: 24,
                r1: 21,
                r2: 23,
            },
            JumpFalse { rs: 24, offset: 4 },
            // pc 28..30 then-arm: val = "AB"; jump to guard2
            LoadConst { rd: 26, idx: c_ab },
            Move { rd: 25, rs: 26 },
            Jump { offset: 7 },
            // pc 31..34: CHOOSE m \in mset : m.b = maxb
            ChooseBegin {
                rd: 27,
                r_binding: 28,
                r_domain: 11,
                loop_end: 4,
            },
            RecordGet {
                rd: 29,
                rs: 28,
                field_idx: f_b,
            },
            Eq {
                rd: 30,
                r1: 29,
                r2: 21,
            },
            ChooseNext {
                rd: 27,
                r_binding: 28,
                r_body: 30,
                loop_begin: -2,
            },
            // pc 35..36: val = chosen.v
            RecordGet {
                rd: 31,
                rs: 27,
                field_idx: f_v,
            },
            Move { rd: 25, rs: 31 },
            // pc 37..43: guard2 = \A ac \in MS : \E m \in mset : m.a = ac
            ForallBegin {
                rd: 32,
                r_binding: 33,
                r_domain: 9,
                loop_end: 6,
            },
            ExistsBegin {
                rd: 34,
                r_binding: 35,
                r_domain: 11,
                loop_end: 4,
            },
            RecordGet {
                rd: 36,
                rs: 35,
                field_idx: f_a,
            },
            Eq {
                rd: 37,
                r1: 36,
                r2: 33,
            },
            ExistsNext {
                rd: 34,
                r_binding: 35,
                r_body: 37,
                loop_begin: -2,
            },
            ForallNext {
                rd: 32,
                r_binding: 33,
                r_body: 34,
                loop_begin: -4,
            },
            // pc 43: witness fails -> ExistsNext (pc 48)
            JumpFalse { rs: 32, offset: 5 },
            // pc 44..47: send: msgs' = msgs \cup {[t |-> "p2s", v |-> val]}
            LoadVar { rd: 38, var_idx: 0 },
            LoadConst { rd: 39, idx: c_p2s }, // value block starts at r39: [t, v]
            Move { rd: 40, rs: 25 },
            RecordNew {
                rd: 41,
                fields_start: f_t_name,
                values_start: 39,
                count: 2,
            },
            // pc 48 is inside the arithmetic below -- recompute indices!
            SetEnum {
                rd: 42,
                start: 41,
                count: 1,
            },
            SetUnion {
                rd: 43,
                r1: 38,
                r2: 42,
            },
            StoreVar { var_idx: 0, rs: 43 },
            LoadBool {
                rd: 32,
                value: true,
            },
            ExistsNext {
                rd: 8,
                r_binding: 9,
                r_body: 32,
                loop_begin: -43,
            },
            Move { rd: 0, rs: 8 },
            Ret { rs: 0 },
        ];
        let func = BytecodeFunction {
            name: "SyntheticPhase2a".to_string(),
            arity: 0,
            max_register: 43,
            instructions,
            max_patched_target: 0,
        };
        (func, pool)
    }

    /// Fix the jump offsets of the hand-built function (targets were sketched
    /// by comment; recompute them from structural positions so test edits stay
    /// maintainable).
    fn fixup_jumps(func: &mut BytecodeFunction) {
        // Find structural positions.
        let pcs: Vec<Opcode> = func.instructions.clone();
        let exists_next = pcs
            .iter()
            .rposition(|op| matches!(op, Opcode::ExistsNext { .. }))
            .unwrap();
        let move_to_r0 = exists_next + 1;
        let choose_begin = pcs
            .iter()
            .position(|op| matches!(op, Opcode::ChooseBegin { .. }))
            .unwrap();
        let guard2_begin = choose_begin + 6; // ForallBegin after choose+extract

        for (pc, op) in func.instructions.iter_mut().enumerate() {
            match op {
                Opcode::JumpFalse { rs: 1, offset } => {
                    *offset = (move_to_r0 as i64 - pc as i64) as i32;
                }
                Opcode::JumpFalse { rs: 24, offset } => {
                    *offset = (choose_begin as i64 - pc as i64) as i32;
                }
                Opcode::Jump { offset } => {
                    *offset = (guard2_begin as i64 - pc as i64) as i32;
                }
                Opcode::JumpFalse { rs: 32, offset } => {
                    *offset = (exists_next as i64 - pc as i64) as i32;
                }
                _ => {}
            }
        }
        // Verify the successor exists loop_end / back edge stay consistent.
        let begin_pc = pcs
            .iter()
            .position(|op| matches!(op, Opcode::ExistsBegin { rd: 8, .. }))
            .unwrap();
        if let Opcode::ExistsBegin { loop_end, .. } = &mut func.instructions[begin_pc] {
            *loop_end = (exists_next as i32 - begin_pc as i32) + 1;
        }
        if let Opcode::ExistsNext { loop_begin, .. } = &mut func.instructions[exists_next] {
            *loop_begin = (begin_pc as i32 + 1) - exists_next as i32;
        }
    }

    fn test_universe() -> Vec<Value> {
        let mut records = vec![
            rec(&[("t", Value::string("p2")), ("v", Value::string("A"))]),
            rec(&[
                ("t", Value::string("p1")),
                ("b", Value::SmallInt(-1)),
                ("a", Value::string("x")),
                ("v", Value::string("n")),
            ]),
            rec(&[
                ("t", Value::string("p1")),
                ("b", Value::SmallInt(0)),
                ("a", Value::string("x")),
                ("v", Value::string("A")),
            ]),
            rec(&[
                ("t", Value::string("p1")),
                ("b", Value::SmallInt(0)),
                ("a", Value::string("y")),
                ("v", Value::string("B")),
            ]),
            rec(&[
                ("t", Value::string("p1")),
                ("b", Value::SmallInt(1)),
                ("a", Value::string("y")),
                ("v", Value::string("A")),
            ]),
            rec(&[("t", Value::string("p2s")), ("v", Value::string("A"))]),
            rec(&[("t", Value::string("p2s")), ("v", Value::string("B"))]),
        ];
        records.sort();
        records
    }

    /// Test-side interpreter for the SCALARIZED (exists-free) output: plain
    /// Value semantics over the emitted opcode subset. Returns
    /// `(result_bool, next_msgs)` where `next_msgs` is the StoreVar'd value.
    fn run_scalarized(
        func: &BytecodeFunction,
        pool: &ConstantPool,
        msgs: &Value,
    ) -> (bool, Option<Value>) {
        let mut regs: Vec<Value> = vec![Value::Bool(false); 256];
        let mut next: Option<Value> = None;
        let mut pc = 0usize;
        let mut steps = 0usize;
        loop {
            assert!(pc < func.instructions.len(), "fell off function end");
            steps += 1;
            assert!(steps < 100_000, "runaway interpreter");
            match &func.instructions[pc] {
                Opcode::LoadVar { rd, var_idx } => {
                    assert_eq!(*var_idx, 0);
                    regs[*rd as usize] = msgs.clone();
                }
                Opcode::LoadConst { rd, idx } => {
                    regs[*rd as usize] = pool.get_value(*idx).clone();
                }
                Opcode::LoadImm { rd, value } => {
                    regs[*rd as usize] = Value::SmallInt(*value);
                }
                Opcode::LoadBool { rd, value } => {
                    regs[*rd as usize] = Value::Bool(*value);
                }
                Opcode::Move { rd, rs } => {
                    regs[*rd as usize] = regs[*rs as usize].clone();
                }
                Opcode::Not { rd, rs } => {
                    let Value::Bool(b) = regs[*rs as usize] else {
                        panic!("Not on non-bool");
                    };
                    regs[*rd as usize] = Value::Bool(!b);
                }
                Opcode::And { rd, r1, r2 } | Opcode::Or { rd, r1, r2 } => {
                    let (Value::Bool(a), Value::Bool(b)) =
                        (regs[*r1 as usize].clone(), regs[*r2 as usize].clone())
                    else {
                        panic!("And/Or on non-bool");
                    };
                    let is_and = matches!(func.instructions[pc], Opcode::And { .. });
                    regs[*rd as usize] = Value::Bool(if is_and { a && b } else { a || b });
                }
                Opcode::Eq { rd, r1, r2 } => {
                    regs[*rd as usize] = Value::Bool(regs[*r1 as usize] == regs[*r2 as usize]);
                }
                Opcode::Neq { rd, r1, r2 } => {
                    regs[*rd as usize] = Value::Bool(regs[*r1 as usize] != regs[*r2 as usize]);
                }
                Opcode::NegInt { rd, rs } => {
                    let Value::SmallInt(v) = regs[*rs as usize] else {
                        panic!("NegInt on non-int");
                    };
                    regs[*rd as usize] = Value::SmallInt(-v);
                }
                Opcode::Jump { offset } => {
                    pc = (pc as i64 + i64::from(*offset)) as usize;
                    continue;
                }
                Opcode::JumpTrue { rs, offset } | Opcode::JumpFalse { rs, offset } => {
                    let Value::Bool(b) = regs[*rs as usize] else {
                        panic!("jump on non-bool");
                    };
                    let jump_on = matches!(func.instructions[pc], Opcode::JumpTrue { .. });
                    if b == jump_on {
                        pc = (pc as i64 + i64::from(*offset)) as usize;
                        continue;
                    }
                }
                Opcode::RecordNew {
                    rd,
                    fields_start,
                    values_start,
                    count,
                } => {
                    let mut entries = Vec::new();
                    for i in 0..*count {
                        let Value::String(name) = pool.get_value(fields_start + u16::from(i))
                        else {
                            panic!("record field name must be a string");
                        };
                        entries.push((
                            intern_name(name.as_ref()),
                            regs[usize::from(*values_start) + usize::from(i)].clone(),
                        ));
                    }
                    regs[*rd as usize] = Value::Record(RecordValue::from_entries(entries));
                }
                Opcode::SetIn { rd, elem, set } => {
                    let elems = iter_finite_set(&regs[*set as usize]).expect("finite set");
                    regs[*rd as usize] = Value::Bool(elems.contains(&regs[*elem as usize]));
                }
                Opcode::SetEnum { rd, start, count } => {
                    let elems: Vec<Value> = (0..*count)
                        .map(|i| regs[usize::from(*start) + usize::from(i)].clone())
                        .collect();
                    regs[*rd as usize] = Value::set(elems);
                }
                Opcode::SetUnion { rd, r1, r2 } => {
                    let mut elems = iter_finite_set(&regs[*r1 as usize]).expect("set");
                    elems.extend(iter_finite_set(&regs[*r2 as usize]).expect("set"));
                    regs[*rd as usize] = Value::set(elems);
                }
                Opcode::StoreVar { var_idx, rs } => {
                    assert_eq!(*var_idx, 0);
                    next = Some(regs[*rs as usize].clone());
                }
                Opcode::Ret { rs } => {
                    let Value::Bool(b) = regs[*rs as usize] else {
                        panic!("Ret on non-bool");
                    };
                    return (b, next);
                }
                other => panic!("scalarized output contained unsupported opcode {other:?}"),
            }
            pc += 1;
        }
    }

    /// First-principles oracle for one witness: returns the successor msgs
    /// value when the witness fires.
    fn oracle(msgs: &[Value], witness: &Value) -> Option<Value> {
        let ms = iter_finite_set(witness).unwrap();
        let get = |record: &Value, field: &str| -> Value {
            let Value::Record(rec) = record else { panic!() };
            rec.get_by_id(intern_name(field)).unwrap().clone()
        };
        // guard1
        if msgs.iter().any(|m| get(m, "t") == Value::string("p2")) {
            return None;
        }
        let mset: Vec<Value> = msgs
            .iter()
            .filter(|m| get(m, "t") == Value::string("p1") && ms.contains(&get(m, "a")))
            .cloned()
            .collect();
        let bals: Vec<i64> = mset
            .iter()
            .map(|m| match get(m, "b") {
                Value::SmallInt(v) => v,
                _ => panic!(),
            })
            .collect();
        let maxb = bals.iter().copied().max().unwrap_or(-1);
        let val = if maxb == -1 {
            Value::string("AB")
        } else {
            // interpreter CHOOSE order: TLC-normalized over the mset.
            let mset_set = Value::set(mset.clone());
            let chosen = mset_set
                .iter_set_tlc_normalized()
                .unwrap()
                .find(|m| get(m, "b") == Value::SmallInt(maxb))
                .expect("choose witness");
            get(&chosen, "v")
        };
        // guard2
        if !ms.iter().all(|ac| mset.iter().any(|m| get(m, "a") == *ac)) {
            return None;
        }
        let sent = rec(&[("t", Value::string("p2s")), ("v", val)]);
        let mut next: Vec<Value> = msgs.to_vec();
        next.push(sent);
        Some(Value::set(next))
    }

    fn max_or_env(name: &str, args: &[Value]) -> Option<Value> {
        assert_eq!(name, "MaxOr");
        let elems = iter_finite_set(&args[0])?;
        let mut max: Option<i64> = None;
        for elem in elems {
            let Value::SmallInt(v) = elem else {
                return None;
            };
            max = Some(max.map_or(v, |m: i64| m.max(v)));
        }
        Some(Value::SmallInt(max.unwrap_or(-1)))
    }

    fn run_differential(majority: Value) {
        let (mut func, pool) = build_phase2a_like(majority.clone());
        fixup_jumps(&mut func);
        let universe = test_universe();
        let layout = universe_layout(&universe);
        let env = RecordSetScalarizeEnv {
            eval_pure_op: &max_or_env,
        };
        let outcome = scalarize_record_set_aggregate_action(&func, &pool, &layout, &env)
            .expect("scalarization must succeed on the recognized shape");
        let witnesses = iter_finite_set(&majority).unwrap();
        assert_eq!(outcome.expansions.len(), witnesses.len());
        for expansion in &outcome.expansions {
            assert!(
                !expansion.func.instructions.iter().any(|op| matches!(
                    op,
                    Opcode::ExistsBegin { .. }
                        | Opcode::ExistsNext { .. }
                        | Opcode::ForallBegin { .. }
                        | Opcode::ChooseBegin { .. }
                        | Opcode::SetFilterBegin { .. }
                        | Opcode::SetBuilderBegin { .. }
                        | Opcode::CallExternal { .. }
                )),
                "scalarized output must be loop- and external-free"
            );
        }

        // Differential: every msgs subset of the universe, every witness.
        let n = universe.len();
        for mask in 0..(1usize << n) {
            let msgs: Vec<Value> = (0..n)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| universe[i].clone())
                .collect();
            let msgs_value = Value::set(msgs.clone());
            for expansion in &outcome.expansions {
                let expected = oracle(&msgs, &expansion.witness);
                let (fired, next) = run_scalarized(&expansion.func, &outcome.pool, &msgs_value);
                match expected {
                    Some(expected_next) => {
                        assert!(
                            fired,
                            "witness {:?} must fire on msgs {msgs_value:?}",
                            expansion.witness
                        );
                        assert_eq!(
                            next.expect("successor written"),
                            expected_next,
                            "successor mismatch for witness {:?} on msgs {msgs_value:?}",
                            expansion.witness
                        );
                    }
                    None => {
                        assert!(
                            !fired,
                            "witness {:?} must NOT fire on msgs {msgs_value:?}",
                            expansion.witness
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn differential_single_set_witness() {
        run_differential(Value::set(vec![Value::set(vec![
            Value::string("x"),
            Value::string("y"),
        ])]));
    }

    #[test]
    fn differential_multiple_set_witnesses() {
        run_differential(Value::set(vec![
            Value::set(vec![Value::string("x")]),
            Value::set(vec![Value::string("y")]),
            Value::set(vec![Value::string("x"), Value::string("y")]),
        ]));
    }

    #[test]
    fn fail_closed_without_pure_op_eval() {
        let (mut func, pool) =
            build_phase2a_like(Value::set(vec![Value::set(vec![Value::string("x")])]));
        fixup_jumps(&mut func);
        let layout = universe_layout(&test_universe());
        let failing = |_: &str, _: &[Value]| -> Option<Value> { None };
        let env = RecordSetScalarizeEnv {
            eval_pure_op: &failing,
        };
        assert!(
            scalarize_record_set_aggregate_action(&func, &pool, &layout, &env).is_none(),
            "external evaluation failure must fail the whole scalarization closed"
        );
    }

    #[test]
    fn fail_closed_on_runtime_witness_domain() {
        let (mut func, pool) =
            build_phase2a_like(Value::set(vec![Value::set(vec![Value::string("x")])]));
        fixup_jumps(&mut func);
        // Turn the successor-EXISTS domain into a runtime value (LoadVar).
        let begin_pc = func
            .instructions
            .iter()
            .position(|op| matches!(op, Opcode::ExistsBegin { rd: 8, .. }))
            .unwrap();
        func.instructions[begin_pc - 1] = Opcode::LoadVar { rd: 7, var_idx: 0 };
        let layout = universe_layout(&test_universe());
        let env = RecordSetScalarizeEnv {
            eval_pure_op: &max_or_env,
        };
        assert!(
            scalarize_record_set_aggregate_action(&func, &pool, &layout, &env).is_none(),
            "runtime successor-EXISTS domain must fail closed"
        );
    }

    #[test]
    fn fail_closed_on_unclosed_universe() {
        let (mut func, pool) =
            build_phase2a_like(Value::set(vec![Value::set(vec![Value::string("x")])]));
        fixup_jumps(&mut func);
        let mut layout = universe_layout(&test_universe());
        if let Some(VarLayout::Compound(CompoundLayout::RecordSetBitmask {
            is_proven_closed,
            ..
        })) = layout.var_layout_mut(0)
        {
            *is_proven_closed = false;
        }
        let env = RecordSetScalarizeEnv {
            eval_pure_op: &max_or_env,
        };
        assert!(
            scalarize_record_set_aggregate_action(&func, &pool, &layout, &env).is_none(),
            "a non-proven-closed universe must fail closed"
        );
    }
}
