// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU StateSpace lane: device-resident explicit reachability for P/T nets.
//!
//! A Petri marking is already a fixed-width vector of non-negative token
//! counts (one slot per place) and a transition firing is a guard on input
//! places plus a subtract/add update — exactly the flat-state,
//! single-successor-per-action contract of the `tla-gpu` BFS engine. This
//! module hand-builds one trust-ir `JitNextStateFn` module per transition
//! (`fn(out, state_in, state_out, state_len)`; enabled bit in
//! `JitCallOut.value`, successor written into pre-seeded `state_out`) and runs
//! the shared device BFS, tracking the two marking statistics the MCC
//! StateSpace examination reports (max token in a place, max tokens in a
//! marking).
//!
//! Lane position: tried after the MDD / Tier-1 structural /
//! disconnected-component lanes, immediately BEFORE the CPU explicit BFS
//! fallback — it accelerates exactly that lane. Fail-closed: any probe,
//! build, emission, or engine error returns `None` and the CPU BFS runs
//! unchanged. `TY_MCC_DISABLE_GPU_STATESPACE` is a diagnostic kill-switch
//! (mirrors the sibling lanes' switches; not a semantic lever).
//!
//! Soundness notes:
//! - Parallel arcs are merged locally (guard on the SUM of input-arc weights,
//!   matching `apply_delta` / the DD lowering) — the lane does not assume
//!   [`PetriNet::canonicalize_parallel_arcs`] already ran on this net.
//! - Markings ride the engine's `i64` slots while the CPU carrier is `u64`.
//!   Any firing that would push a place past `i64::MAX` trips an in-kernel
//!   overflow trap (nonzero `JitCallOut` status → device error flag → engine
//!   error → lane declines), so a wrapped marking can never be published.

use std::collections::BTreeMap;

use tla_bignum::BigUint;
use trust_ir::constant::Constant;
use trust_ir::inst::{ICmpOp, Inst};
use trust_ir::ty::Ty;
use trust_ir::value::{BlockId, FuncId, ValueId};
use trust_ir::{Block, FuncTy, Function, InstrNode, Module};

use crate::examinations::state_space::StateSpaceStats;
use crate::petri_net::{PetriNet, TransitionInfo};
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

/// Per-place token cap enforced in-kernel (fail-closed status trap past it).
///
/// Every published marking satisfies `m[p] <= TOKEN_CAP` inductively: initial
/// markings are gated at admission, arc weights/deltas are gated in
/// [`firing_plan`], and every positive-delta update traps past the cap. This
/// is what makes the straight-line predicate arithmetic exact: a
/// `TokensCount` sum over ≤ 2^20 places stays below 2^60 — no i64 wrap — so
/// signed comparisons against (gated ≤ 2^60) constants are exact. A net that
/// genuinely exceeds the cap declines to the CPU lanes (such spaces are far
/// beyond explicit completion anyway).
const TOKEN_CAP: i64 = 1 << 40;

/// Largest formula constant admitted to the GPU predicate compiler; combined
/// with the [`TOKEN_CAP`] sum bound this keeps every signed i64 comparison
/// exact.
const PREDICATE_CONST_CAP: u64 = 1 << 60;

/// Diagnostic kill-switch (default OFF = lane runs). Mirrors
/// `TY_MCC_DISABLE_TIER1_STATESPACE`.
fn gpu_state_space_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_GPU_STATESPACE")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true" | "yes"))
}

/// Testing lever: skip the bounded CPU probe and go straight to the GPU
/// (used by the differential validation harness; small nets are otherwise
/// answered by the probe and never reach the device).
fn gpu_state_space_forced() -> bool {
    std::env::var("TY_MCC_GPU_STATESPACE_FORCE")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true" | "yes"))
}

/// Bounded CPU-probe cap for auto-escalation (mirrors the TLA+ engine's
/// probe-then-GPU tier). A space that completes within the cap is answered by
/// the probe — exact, and the device is never touched (unit tests and the
/// long tail of small MCC instances stay CPU-only). The cap must keep the
/// probe a CHEAP gate (~≤1 s): measured on the benchmark fixtures, a 2^20 cap
/// let the probe spend ~9 s completing a 690 K-marking net that the GPU
/// finishes in ~2 s after a sub-second tripped probe.
const GPU_AUTO_PROBE_STATE_CAP: usize = 1 << 18;

/// Cheap availability gates for the GPU lane (kill-switch, net shape, CUDA
/// probe). `true` means the ladder should run the bounded CPU probe and
/// escalate to [`state_space_stats_gpu`] when it trips.
pub(crate) fn gpu_lane_enabled(net: &PetriNet) -> bool {
    if gpu_state_space_disabled() {
        return false;
    }
    if net.num_places() == 0 || net.num_transitions() == 0 {
        return false;
    }
    tla_gpu::probe().is_ok()
}

/// The state cap for the pre-GPU bounded CPU probe, or `None` when the probe
/// is skipped (`TY_MCC_GPU_STATESPACE_FORCE`, testing only).
pub(crate) fn cpu_probe_cap(configured_max_states: usize) -> Option<usize> {
    if gpu_state_space_forced() {
        return None;
    }
    Some(GPU_AUTO_PROBE_STATE_CAP.min(configured_max_states))
}

/// One transition's firing semantics with parallel arcs merged, in the `i64`
/// carrier the GPU engine uses.
struct FiringPlan {
    /// `(place, summed input-arc weight)` — enabledness requires
    /// `m[place] >= weight` per entry.
    guards: Vec<(i64, i64)>,
    /// `(place, net token delta)` for touched places only (`delta != 0`).
    deltas: Vec<(i64, i64)>,
}

/// Merge parallel arcs and fold per-place net deltas, all in checked `i64`
/// bounded by [`TOKEN_CAP`]. `None` = weights don't fit the engine carrier →
/// net not GPU-admissible.
fn firing_plan(t: &TransitionInfo) -> Option<FiringPlan> {
    let mut in_w: BTreeMap<u32, i64> = BTreeMap::new();
    for arc in &t.inputs {
        let w = i64::try_from(arc.weight).ok()?;
        let e = in_w.entry(arc.place.0).or_insert(0);
        *e = e.checked_add(w).filter(|&s| s <= TOKEN_CAP)?;
    }
    let mut delta: BTreeMap<u32, i64> = BTreeMap::new();
    for (&p, &w) in &in_w {
        delta.insert(p, w.checked_neg()?);
    }
    for arc in &t.outputs {
        let w = i64::try_from(arc.weight).ok()?;
        let e = delta.entry(arc.place.0).or_insert(0);
        *e = e.checked_add(w)?;
    }
    // Bound every folded delta magnitude so `m[p] + delta` cannot wrap i64
    // even at the cap (|m| ≤ 2^40 inductively, |delta| ≤ 2^40 ⇒ |sum| ≤ 2^41).
    if delta
        .values()
        .any(|&d| d.checked_abs().is_none_or(|a| a > TOKEN_CAP))
    {
        return None;
    }
    Some(FiringPlan {
        guards: in_w.iter().map(|(&p, &w)| (i64::from(p), w)).collect(),
        deltas: delta
            .iter()
            .filter(|&(_, &d)| d != 0)
            .map(|(&p, &d)| (i64::from(p), d))
            .collect(),
    })
}

/// Small IR builder for one transition's `JitNextStateFn` module.
struct FnBuilder {
    blocks: Vec<Block>,
    current: usize,
    next_value: u32,
}

impl FnBuilder {
    fn new() -> Self {
        FnBuilder {
            blocks: vec![Block::new(BlockId::new(0))],
            current: 0,
            next_value: 0,
        }
    }

    fn fresh(&mut self) -> ValueId {
        let v = ValueId::new(self.next_value);
        self.next_value += 1;
        v
    }

    fn new_block(&mut self) -> BlockId {
        let id = BlockId::new(u32::try_from(self.blocks.len()).expect("block count"));
        self.blocks.push(Block::new(id));
        id
    }

    fn switch_to(&mut self, id: BlockId) {
        self.current = id.index() as usize;
    }

    fn push(&mut self, inst: Inst, results: Vec<ValueId>) {
        self.blocks[self.current]
            .body
            .push(InstrNode::new(inst).with_results(results));
    }

    fn const_i64(&mut self, v: i64) -> ValueId {
        let r = self.fresh();
        self.push(
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(i128::from(v)),
            },
            vec![r],
        );
        r
    }

    /// `base[idx]` address for an i64 array behind `base`.
    fn gep_i64(&mut self, base: ValueId, idx: i64) -> ValueId {
        let c = self.const_i64(idx);
        let r = self.fresh();
        self.push(
            Inst::GEP {
                pointee_ty: Ty::I64,
                base,
                indices: vec![c],
                inbounds: false,
            },
            vec![r],
        );
        r
    }

    fn load_i64(&mut self, ptr: ValueId) -> ValueId {
        let r = self.fresh();
        self.push(
            Inst::Load {
                ty: Ty::I64,
                ptr,
                volatile: false,
                align: None,
            },
            vec![r],
        );
        r
    }

    fn store_i64(&mut self, ptr: ValueId, value: ValueId) {
        self.push(
            Inst::Store {
                ty: Ty::I64,
                ptr,
                value,
                volatile: false,
                align: None,
            },
            vec![],
        );
    }

    fn icmp(&mut self, op: ICmpOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh();
        self.push(
            Inst::ICmp {
                op,
                ty: Ty::I64,
                lhs,
                rhs,
            },
            vec![r],
        );
        r
    }

    fn binop(&mut self, op: trust_ir::inst::BinOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh();
        self.push(
            Inst::BinOp {
                op,
                ty: Ty::I64,
                lhs,
                rhs,
            },
            vec![r],
        );
        r
    }

    fn cond_br(&mut self, cond: ValueId, then_target: BlockId, else_target: BlockId) {
        self.push(
            Inst::CondBr {
                cond,
                then_target,
                then_args: vec![],
                else_target,
                else_args: vec![],
            },
            vec![],
        );
    }
}

/// Build the `JitNextStateFn` trust-ir module for one transition.
///
/// The GPU driver pre-seeds `state_out` from `state_in`, so only touched
/// places are written. `out` is a zero-initialized `JitCallOut` (status byte
/// 0 already Ok, value already 0=disabled); the enabled path stores `1` into
/// `value` (i64 at byte offset 8), and the overflow trap stores a nonzero
/// status (byte offset 0) which the engine treats as a fail-closed hard stop.
fn transition_module(plan: &FiringPlan, symbol: &str) -> Module {
    let mut module = Module::new(symbol);
    let ft = module.add_func_type(FuncTy {
        params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::I32],
        returns: vec![],
        is_vararg: false,
    });

    let mut b = FnBuilder::new();
    let out_ptr = b.fresh();
    let state_in = b.fresh();
    let state_out = b.fresh();
    let state_len = b.fresh();
    b.blocks[0].params = vec![
        (out_ptr, Ty::Ptr),
        (state_in, Ty::Ptr),
        (state_out, Ty::Ptr),
        (state_len, Ty::I32),
    ];

    let disabled = b.new_block();
    let fault = b.new_block();

    // Guard chain: enabledness needs m[p] >= (summed) input weight per place.
    for &(place, weight) in &plan.guards {
        let addr = b.gep_i64(state_in, place);
        let tokens = b.load_i64(addr);
        let w = b.const_i64(weight);
        let cond = b.icmp(ICmpOp::Sge, tokens, w);
        let next = b.new_block();
        b.cond_br(cond, next, disabled);
        b.switch_to(next);
    }

    // Enabled: per touched place, state_out[p] = state_in[p] + delta. The
    // guard bounds every negative delta (new = old - in + out >= out >= 0),
    // so only positive deltas can grow a place — trap past TOKEN_CAP. No i64
    // wrap is possible before the check: |old| ≤ cap inductively and
    // |delta| ≤ cap by the firing-plan gate, so the sum stays within 2^41.
    for &(place, delta) in &plan.deltas {
        let in_addr = b.gep_i64(state_in, place);
        let tokens = b.load_i64(in_addr);
        let d = b.const_i64(delta);
        let new_tokens = b.fresh();
        b.push(
            Inst::BinOp {
                op: trust_ir::inst::BinOp::Add,
                ty: Ty::I64,
                lhs: tokens,
                rhs: d,
            },
            vec![new_tokens],
        );
        if delta > 0 {
            let cap = b.const_i64(TOKEN_CAP);
            let over_cap = b.icmp(ICmpOp::Sgt, new_tokens, cap);
            let ok = b.new_block();
            b.cond_br(over_cap, fault, ok);
            b.switch_to(ok);
        }
        let out_addr = b.gep_i64(state_out, place);
        b.store_i64(out_addr, new_tokens);
    }
    // out.value = 1 (JitCallOut value field: i64 at byte offset 8).
    let value_addr = b.gep_i64(out_ptr, 1);
    let one = b.const_i64(1);
    b.store_i64(value_addr, one);
    b.push(Inst::Return { values: vec![] }, vec![]);

    b.switch_to(disabled);
    b.push(Inst::Return { values: vec![] }, vec![]);

    // Overflow trap: nonzero JitCallOut status (u8 at byte offset 0; the
    // whole first word is padding past byte 0, so an i64 store of 1 sets
    // status=1 and leaves the padding zero).
    b.switch_to(fault);
    let status_addr = b.gep_i64(out_ptr, 0);
    let one = b.const_i64(1);
    b.store_i64(status_addr, one);
    b.push(Inst::Return { values: vec![] }, vec![]);

    let mut func = Function::new(FuncId::new(0), symbol.to_string(), ft, BlockId::new(0));
    func.blocks = b.blocks;
    module.functions.push(func);
    module
}

/// Compile a [`ResolvedPredicate`] to straight-line 0/1 i64 arithmetic.
///
/// Exactness rests on the run-wide token-cap invariant: every published
/// marking has `m[p] <= TOKEN_CAP` per place, so a `TokensCount` sum over
/// ≤ 2^20 places stays below 2^60 and never wraps; formula constants are
/// gated at [`PREDICATE_CONST_CAP`], so every signed comparison is exact.
/// Boolean structure is bitwise over 0/1 values (`ICmp` produces 0/1;
/// `And`/`Or` on 0/1 are the logical connectives; `Not` is `Xor 1`) — no
/// branches, so the device warp never diverges on formula shape.
///
/// `None` = the predicate is not GPU-admissible (oversized constant or an
/// unplannable fireability transition) → the whole lane declines.
fn emit_predicate(
    b: &mut FnBuilder,
    state_in: ValueId,
    pred: &ResolvedPredicate,
    net: &PetriNet,
) -> Option<ValueId> {
    match pred {
        ResolvedPredicate::True => Some(b.const_i64(1)),
        ResolvedPredicate::False => Some(b.const_i64(0)),
        ResolvedPredicate::And(children) => {
            let mut acc = b.const_i64(1);
            for child in children {
                let v = emit_predicate(b, state_in, child, net)?;
                acc = b.binop(trust_ir::inst::BinOp::And, acc, v);
            }
            Some(acc)
        }
        ResolvedPredicate::Or(children) => {
            let mut acc = b.const_i64(0);
            for child in children {
                let v = emit_predicate(b, state_in, child, net)?;
                acc = b.binop(trust_ir::inst::BinOp::Or, acc, v);
            }
            Some(acc)
        }
        ResolvedPredicate::Not(inner) => {
            let v = emit_predicate(b, state_in, inner, net)?;
            let one = b.const_i64(1);
            Some(b.binop(trust_ir::inst::BinOp::Xor, v, one))
        }
        ResolvedPredicate::IntLe(left, right) => {
            let lv = emit_int_expr(b, state_in, left)?;
            let rv = emit_int_expr(b, state_in, right)?;
            Some(b.icmp(ICmpOp::Sle, lv, rv))
        }
        ResolvedPredicate::IsFireable(transitions) => {
            // MCC is-fireable(t1..tn) = some listed transition is enabled.
            // Guards use the SUMMED input-arc weights (the caller gates nets
            // with unmerged parallel arcs so this matches `net.is_enabled`).
            let mut acc = b.const_i64(0);
            for &t in transitions {
                let plan = firing_plan(net.transitions.get(t.0 as usize)?)?;
                let mut enabled = b.const_i64(1);
                for &(place, weight) in &plan.guards {
                    let addr = b.gep_i64(state_in, place);
                    let tokens = b.load_i64(addr);
                    let w = b.const_i64(weight);
                    let cond = b.icmp(ICmpOp::Sge, tokens, w);
                    enabled = b.binop(trust_ir::inst::BinOp::And, enabled, cond);
                }
                acc = b.binop(trust_ir::inst::BinOp::Or, acc, enabled);
            }
            Some(acc)
        }
    }
}

/// Compile a [`ResolvedIntExpr`] to an i64 value (loads + adds; see
/// [`emit_predicate`] for the no-wrap argument).
fn emit_int_expr(b: &mut FnBuilder, state_in: ValueId, expr: &ResolvedIntExpr) -> Option<ValueId> {
    match expr {
        ResolvedIntExpr::Constant(v) => {
            if *v > PREDICATE_CONST_CAP {
                return None;
            }
            Some(b.const_i64(i64::try_from(*v).ok()?))
        }
        ResolvedIntExpr::TokensCount(places) => {
            // The sum bound (≤ places × TOKEN_CAP < 2^60) needs the place
            // count itself bounded.
            if places.len() > 1 << 20 {
                return None;
            }
            let mut acc = b.const_i64(0);
            for p in places {
                let addr = b.gep_i64(state_in, i64::from(p.0));
                let tokens = b.load_i64(addr);
                acc = b.binop(trust_ir::inst::BinOp::Add, acc, tokens);
            }
            Some(acc)
        }
    }
}

/// Build a `NativeInvariantFn` trust-ir module (`fn(out, state, state_len)`)
/// that stores the predicate's 0/1 truth into `JitCallOut.value`. The engine
/// treats value != 1 as an invariant violation: it publishes the violating
/// row and stops — the witness mechanism the reachability lane rides.
fn predicate_module(pred: &ResolvedPredicate, net: &PetriNet, symbol: &str) -> Option<Module> {
    let mut module = Module::new(symbol);
    let ft = module.add_func_type(FuncTy {
        params: vec![Ty::Ptr, Ty::Ptr, Ty::I32],
        returns: vec![],
        is_vararg: false,
    });

    let mut b = FnBuilder::new();
    let out_ptr = b.fresh();
    let state_in = b.fresh();
    let state_len = b.fresh();
    b.blocks[0].params = vec![
        (out_ptr, Ty::Ptr),
        (state_in, Ty::Ptr),
        (state_len, Ty::I32),
    ];

    let holds = emit_predicate(&mut b, state_in, pred, net)?;
    // out.value (i64 at byte offset 8) = 0/1 truth; status byte stays 0 (Ok).
    let value_addr = b.gep_i64(out_ptr, 1);
    b.store_i64(value_addr, holds);
    b.push(Inst::Return { values: vec![] }, vec![]);

    let mut func = Function::new(FuncId::new(0), symbol.to_string(), ft, BlockId::new(0));
    func.blocks = b.blocks;
    module.functions.push(func);
    Some(module)
}

/// Shared device-BFS core for the Petri GPU verdict lanes: exhaustively
/// explore the net's reachable markings on the GPU and return the raw engine
/// outcome, or `None` to fall through to the caller's CPU lane (fail-closed
/// on every probe/build/emission/capacity/engine error).
///
/// On success the outcome is EXHAUSTIVE: `distinct_states`/`transitions`
/// cover the full reachable set, `deadlock_states` counts markings with zero
/// enabled transitions, and (slot stats are always tracked here)
/// `max_slot_value`/`max_slot_sum`/`slot_maxima` are the marking-magnitude
/// maxima over all reachable markings.
///
/// `max_states` is the examination's configured exploration bound: the device
/// search declines once distinct markings exceed it, so an unbounded or
/// oversized net fails closed at the same bound the CPU lane honors instead
/// of grinding through the growth ladder. `lane` labels stderr diagnostics.
pub(crate) fn gpu_explore(
    net: &PetriNet,
    max_states: usize,
    lane: &str,
) -> Option<tla_gpu::GpuBfsOutcome> {
    let outcome = gpu_explore_core(net, max_states, lane, &[])?;
    if outcome.violation.is_some() {
        // No invariants are installed; a violation here is impossible and
        // indicates an engine fault — fail closed.
        eprintln!("[mcc] {lane} GPU lane declined: spurious violation (engine fault)");
        return None;
    }
    Some(outcome)
}

/// [`gpu_explore`] with formula predicates installed as engine invariants.
///
/// The engine checks every predicate on every DISTINCT marking as it is
/// claimed (initial markings included); the first marking where some
/// predicate's value != 1 is published as `outcome.violation` and the search
/// stops — an exact witness row. `outcome.violation == None` means the
/// exploration completed exhaustively with every predicate holding on every
/// reachable marking. On a violation outcome the aggregate statistics are
/// PARTIAL (the search stopped early) — callers must only consume the
/// witness row.
fn gpu_explore_core(
    net: &PetriNet,
    max_states: usize,
    lane: &str,
    invariant_preds: &[&ResolvedPredicate],
) -> Option<tla_gpu::GpuBfsOutcome> {
    if gpu_state_space_disabled() {
        return None;
    }
    if net.num_places() == 0 || net.num_transitions() == 0 {
        return None;
    }
    // Cheap availability probe before any build work.
    if let Err(e) = tla_gpu::probe() {
        eprintln!("[mcc] {lane} GPU lane declined (device probe): {e}");
        return None;
    }

    let init_rows: Vec<i64> = net
        .initial_marking
        .iter()
        .map(|&tokens| i64::try_from(tokens).ok().filter(|&t| t <= TOKEN_CAP))
        .collect::<Option<_>>()?;
    if init_rows.len() != net.num_places() {
        return None;
    }
    if net
        .transitions
        .iter()
        .flat_map(|t| t.inputs.iter().chain(&t.outputs))
        .any(|arc| (arc.place.0 as usize) >= net.num_places())
    {
        return None;
    }

    let modules: Vec<(String, String, Module)> = net
        .transitions
        .iter()
        .enumerate()
        .map(|(idx, t)| {
            let symbol = format!("ty_petri_transition_{idx}");
            let module = transition_module(&firing_plan(t)?, &symbol);
            Some((t.id.clone(), symbol, module))
        })
        .collect::<Option<_>>()?;
    let actions: Vec<(String, String, &Module)> = modules
        .iter()
        .map(|(name, symbol, module)| (name.clone(), symbol.clone(), module))
        .collect();

    let invariant_modules: Vec<(String, String, Module)> = invariant_preds
        .iter()
        .enumerate()
        .map(|(idx, pred)| {
            let symbol = format!("ty_petri_invariant_{idx}");
            let module = predicate_module(pred, net, &symbol)?;
            Some((format!("inv{idx}"), symbol, module))
        })
        .collect::<Option<_>>()?;
    let invariants: Vec<(String, String, &Module)> = invariant_modules
        .iter()
        .map(|(name, symbol, module)| (name.clone(), symbol.clone(), module))
        .collect();

    let source = match tla_gpu::emit_program(&actions, &invariants) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("[mcc] {lane} GPU lane declined (emission): {e}");
            return None;
        }
    };
    let spec = tla_gpu::GpuBfsSpec {
        slots: net.num_places(),
        action_count: source.action_count,
        actions_src: source.source,
        init_rows,
        track_slot_stats: true,
    };

    // The engine default config is sized for ~100M-state TLA+ runs with
    // narrow rows; for a wide net (row = places × 8 bytes) its 32Mi-row
    // arenas would commit tens of GB before the first level (measured: ~28 s
    // of silent allocation for a 128-place net). Start small — typical MCC
    // StateSpace instances that complete at all fit easily — and let the
    // grow-and-retry ladder expand on the fail-closed capacity errors, under
    // a hard arena-byte budget so a blow-up declines to the CPU lane instead
    // of pressuring the box.
    let row_bytes = (net.num_places() as u64) * 8;
    const ARENA_BYTE_BUDGET: u64 = 16 << 30;
    let init_count = (spec.init_rows.len() / net.num_places()) as u64;
    let mut config = tla_gpu::GpuBfsConfig {
        table_bits: 22,
        frontier_cap_rows: ((1u64 << 30) / row_bytes).min(1 << 20).max(init_count),
        max_distinct: u64::try_from(max_states).unwrap_or(u64::MAX),
        ..Default::default()
    };
    let mut attempts = 0;
    let outcome = loop {
        match tla_gpu::run_bfs(&spec, &config) {
            Ok(outcome) => break outcome,
            Err(tla_gpu::GpuError::CapacityExceeded {
                what,
                needed,
                capacity,
            }) if what == "distinct-state cap" => {
                // The examination's exploration bound, not an engine limit —
                // growing buffers cannot help. Same fail-closed outcome the
                // CPU lane reports at its cap.
                eprintln!(
                    "[mcc] {lane} GPU lane declined: distinct markings exceed the \
                     configured exploration bound ({needed} > {capacity})"
                );
                return None;
            }
            Err(tla_gpu::GpuError::CapacityExceeded { what, .. }) if attempts < 4 => {
                attempts += 1;
                if what == "fingerprint table" {
                    config.table_bits += 2;
                } else {
                    let grown = config.frontier_cap_rows.saturating_mul(4);
                    if grown.saturating_mul(row_bytes) > ARENA_BYTE_BUDGET {
                        eprintln!(
                            "[mcc] {lane} GPU lane declined: arena budget exhausted \
                             ({grown} rows x {row_bytes} B/row)"
                        );
                        return None;
                    }
                    config.frontier_cap_rows = grown;
                }
            }
            Err(e) => {
                eprintln!("[mcc] {lane} GPU lane declined (engine): {e}");
                return None;
            }
        }
    };
    eprintln!(
        "[mcc] {lane} GPU lane: {} states / {} edges in {:.3}s (nvrtc {:.0} ms, {} transitions as kernels{})",
        outcome.distinct_states,
        outcome.transitions,
        outcome.wall.as_secs_f64(),
        outcome.compile_wall.as_secs_f64() * 1e3,
        net.num_transitions(),
        if outcome.violation.is_some() {
            "; stopped at a predicate witness"
        } else {
            ""
        },
    );

    Some(outcome)
}

/// Outcome of a GPU reachability-formula exploration.
pub(crate) enum GpuReachabilityOutcome {
    /// Exhaustive completion: every installed predicate holds on every
    /// reachable marking.
    Exhausted,
    /// A reachable marking where some installed predicate fails (exact
    /// witness; the search stopped there, so nothing else is known).
    Witness(Vec<u64>),
}

/// One exhaustive GPU BFS with `invariant_preds` installed: `Exhausted` iff
/// all predicates hold everywhere; `Witness(marking)` = a reachable marking
/// falsifying at least one predicate. `None` = decline (fail-closed).
pub(crate) fn reachability_explore_gpu(
    net: &PetriNet,
    max_states: usize,
    invariant_preds: &[&ResolvedPredicate],
) -> Option<GpuReachabilityOutcome> {
    let outcome = gpu_explore_core(net, max_states, "Reachability", invariant_preds)?;
    match outcome.violation {
        None => Some(GpuReachabilityOutcome::Exhausted),
        Some(row) => {
            let marking: Vec<u64> = row
                .into_iter()
                .map(|tokens| u64::try_from(tokens).ok())
                .collect::<Option<_>>()?;
            if marking.len() != net.num_places() {
                return None;
            }
            Some(GpuReachabilityOutcome::Witness(marking))
        }
    }
}

/// GPU StateSpace lane entry: exact `StateSpaceStats` for the net, or `None`
/// to fall through to the CPU explicit BFS.
pub(crate) fn state_space_stats_gpu(net: &PetriNet, max_states: usize) -> Option<StateSpaceStats> {
    let outcome = gpu_explore(net, max_states, "StateSpace")?;
    Some(StateSpaceStats {
        states: BigUint::from(outcome.distinct_states),
        edges: BigUint::from(outcome.transitions),
        max_token_in_place: outcome.max_slot_value,
        max_token_sum: outcome.max_slot_sum,
    })
}

/// GPU ReachabilityDeadlock lane: `Some(true)` iff a reachable marking
/// enables zero transitions, `Some(false)` iff the EXHAUSTIVE exploration
/// found none. `None` = decline (CPU portfolio decides).
///
/// Exact in both directions: on success the device BFS covered the full
/// reachable set and every distinct marking was deadlock-checked exactly
/// once (each passes through a frontier once and the count kernel runs on
/// every frontier before the loop can exit).
pub(crate) fn deadlock_exists_gpu(net: &PetriNet, max_states: usize) -> Option<bool> {
    let outcome = gpu_explore(net, max_states, "ReachabilityDeadlock")?;
    Some(outcome.deadlock_states > 0)
}

/// GPU per-place token maxima over the EXHAUSTIVE reachable set (OneSafe /
/// UpperBounds carriers). `None` = decline. The returned vector has one
/// entry per place, indexed like the marking.
pub(crate) fn place_maxima_gpu(net: &PetriNet, max_states: usize, lane: &str) -> Option<Vec<u64>> {
    let outcome = gpu_explore(net, max_states, lane)?;
    if outcome.slot_maxima.len() != net.num_places() {
        return None;
    }
    Some(outcome.slot_maxima)
}

/// Lower a resolved CTL formula onto the device evaluator's op set,
/// interning atom predicates. Universal operators lower via the engine's
/// duality constructors (`AX/AG/AF/AU`), so the device core only evaluates
/// existential fixpoints. `None` = an atom is not GPU-admissible.
fn lower_ctl(
    formula: &crate::examinations::ctl::resolve::ResolvedCtl,
    net: &PetriNet,
    atoms: &mut Vec<ResolvedPredicate>,
) -> Option<tla_gpu::CtlOp> {
    use tla_gpu::CtlOp;
    type R = crate::examinations::ctl::resolve::ResolvedCtl;
    Some(match formula {
        R::Atom(pred) => {
            // Admissibility probe: the predicate must compile (constant caps,
            // fireability plans). Try a throwaway module build.
            predicate_module(pred, net, "ty_petri_atom_probe")?;
            let k = atoms.iter().position(|p| p == pred).unwrap_or_else(|| {
                atoms.push(pred.clone());
                atoms.len() - 1
            });
            CtlOp::Atom(k)
        }
        R::Not(a) => CtlOp::Not(Box::new(lower_ctl(a, net, atoms)?)),
        R::And(cs) => CtlOp::And(
            cs.iter()
                .map(|c| lower_ctl(c, net, atoms))
                .collect::<Option<_>>()?,
        ),
        R::Or(cs) => CtlOp::Or(
            cs.iter()
                .map(|c| lower_ctl(c, net, atoms))
                .collect::<Option<_>>()?,
        ),
        R::EX(a) => CtlOp::EX(Box::new(lower_ctl(a, net, atoms)?)),
        R::AX(a) => CtlOp::ax(lower_ctl(a, net, atoms)?),
        R::EF(a) => CtlOp::EF(Box::new(lower_ctl(a, net, atoms)?)),
        R::AF(a) => CtlOp::af(lower_ctl(a, net, atoms)?),
        R::EG(a) => CtlOp::EG(Box::new(lower_ctl(a, net, atoms)?)),
        R::AG(a) => CtlOp::ag(lower_ctl(a, net, atoms)?),
        R::EU(a, b) => CtlOp::EU(
            Box::new(lower_ctl(a, net, atoms)?),
            Box::new(lower_ctl(b, net, atoms)?),
        ),
        R::AU(a, b) => CtlOp::au(lower_ctl(a, net, atoms)?, lower_ctl(b, net, atoms)?),
        // E(GF a) fair-cycle carrier. `Not(EGF(Atom(¬p)))` then lowers to
        // exactly `CtlOp::afg(p)`, the GPU persistence dual.
        R::EGF(a) => CtlOp::EGF(Box::new(lower_ctl(a, net, atoms)?)),
    })
}

/// GPU deep-CTL batch check over the retained reachable set of the RAW net:
/// `Some(verdicts)` (one bool per formula, truth at the initial marking) or
/// `None` = decline (CPU checker decides). Semantics follow the CPU
/// checker's maximal-path convention (`EG` accepts a deadlocked φ-state; the
/// universal operators are the standard duals).
///
/// The `IsFireable` parity gate (summed vs per-arc weights) declines nets
/// with parallel input arcs from one place — mirroring the reachability
/// lane.
pub(crate) fn ctl_check_gpu(
    net: &PetriNet,
    formulas: &[crate::examinations::ctl::resolve::ResolvedCtl],
    max_states: usize,
    deadline: Option<std::time::Instant>,
) -> Option<Vec<bool>> {
    if gpu_state_space_disabled() || formulas.is_empty() {
        return None;
    }
    if net.num_places() == 0 || net.num_transitions() == 0 {
        return None;
    }
    if let Err(e) = tla_gpu::probe() {
        eprintln!("[mcc] CTL GPU lane declined (device probe): {e}");
        return None;
    }
    let has_parallel_input_arcs = net.transitions.iter().any(|t| {
        let mut seen = std::collections::HashSet::new();
        t.inputs.iter().any(|arc| !seen.insert(arc.place.0))
    });
    if has_parallel_input_arcs {
        return None;
    }

    let init_rows: Vec<i64> = net
        .initial_marking
        .iter()
        .map(|&tokens| i64::try_from(tokens).ok().filter(|&t| t <= TOKEN_CAP))
        .collect::<Option<_>>()?;
    if init_rows.len() != net.num_places()
        || net
            .transitions
            .iter()
            .flat_map(|t| t.inputs.iter().chain(&t.outputs))
            .any(|arc| (arc.place.0 as usize) >= net.num_places())
    {
        return None;
    }

    // Lower the batch, interning atoms.
    let mut atom_preds: Vec<ResolvedPredicate> = Vec::new();
    let ops: Vec<tla_gpu::CtlOp> = formulas
        .iter()
        .map(|f| lower_ctl(f, net, &mut atom_preds))
        .collect::<Option<_>>()?;

    // Actions.
    let modules: Vec<(String, String, Module)> = net
        .transitions
        .iter()
        .enumerate()
        .map(|(idx, t)| {
            let symbol = format!("ty_petri_transition_{idx}");
            let module = transition_module(&firing_plan(t)?, &symbol);
            Some((t.id.clone(), symbol, module))
        })
        .collect::<Option<_>>()?;
    let actions: Vec<(String, String, &Module)> = modules
        .iter()
        .map(|(name, symbol, module)| (name.clone(), symbol.clone(), module))
        .collect();
    let actions_source = match tla_gpu::emit_program(&actions, &[]) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("[mcc] CTL GPU lane declined (action emission): {e}");
            return None;
        }
    };

    // Atoms.
    let atom_modules: Vec<(String, String, Module)> = atom_preds
        .iter()
        .enumerate()
        .map(|(idx, pred)| {
            let symbol = format!("ty_petri_atom_{idx}");
            let module = predicate_module(pred, net, &symbol)?;
            Some((format!("atom{idx}"), symbol, module))
        })
        .collect::<Option<_>>()?;
    let atom_refs: Vec<(String, String, &Module)> = atom_modules
        .iter()
        .map(|(name, symbol, module)| (name.clone(), symbol.clone(), module))
        .collect();
    let atoms_source = match tla_gpu::emit_atom_adapters(&atom_refs, actions.len()) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("[mcc] CTL GPU lane declined (atom emission): {e}");
            return None;
        }
    };

    let spec = tla_gpu::GpuCtlSpec {
        slots: net.num_places(),
        action_count: actions_source.action_count,
        actions_src: actions_source.source,
        atoms_src: atoms_source.source,
        atom_count: atom_preds.len(),
        init_rows,
    };

    // Retained arena: states × row_bytes must fit a budget; grow-and-retry
    // through the fail-closed capacity errors like the sibling lanes.
    let row_bytes = (net.num_places() as u64) * 8;
    const CTL_ARENA_BYTE_BUDGET: u64 = 24 << 30;
    let mut config = tla_gpu::GpuCtlConfig {
        table_bits: 22,
        max_states: (u64::try_from(max_states).unwrap_or(u64::MAX))
            .min((2u64 << 30) / row_bytes.max(1))
            .max(1 << 16),
        deadline,
        ..Default::default()
    };
    let mut attempts = 0;
    let outcome = loop {
        match tla_gpu::run_ctl(&spec, &config, &ops) {
            Ok(outcome) => break outcome,
            Err(tla_gpu::GpuError::CapacityExceeded {
                what,
                needed,
                capacity,
            }) if attempts < 4 => {
                attempts += 1;
                if u64::try_from(max_states).map_or(false, |cap| needed > cap) {
                    eprintln!(
                        "[mcc] CTL GPU lane declined: reachable markings exceed the \
                         configured exploration bound ({needed} > {capacity})"
                    );
                    return None;
                }
                if what == "fingerprint table" {
                    config.table_bits += 2;
                } else {
                    let grown = config.max_states.saturating_mul(4);
                    if grown.saturating_mul(row_bytes) > CTL_ARENA_BYTE_BUDGET {
                        eprintln!("[mcc] CTL GPU lane declined: arena budget exhausted");
                        return None;
                    }
                    config.max_states = grown;
                }
            }
            Err(e) => {
                eprintln!("[mcc] CTL GPU lane declined (engine): {e}");
                return None;
            }
        }
    };
    eprintln!(
        "[mcc] CTL GPU lane: {} formulas decided over {} retained states in {:.3}s \
         (nvrtc {:.0} ms, {} atoms)",
        outcome.verdicts.len(),
        outcome.distinct_states,
        outcome.wall.as_secs_f64(),
        outcome.compile_wall.as_secs_f64() * 1e3,
        atom_preds.len(),
    );
    Some(outcome.verdicts)
}

/// GPU StableMarking (P/T): `Some(true)` iff some place's token count is
/// constant across the EXHAUSTIVE reachable set (per-place minimum equals
/// per-place maximum — and hence equals the initial marking, which is itself
/// reachable), `Some(false)` iff every place provably varies. `None` =
/// decline (CPU portfolio decides).
pub(crate) fn stable_marking_gpu(net: &PetriNet, max_states: usize) -> Option<bool> {
    let outcome = gpu_explore(net, max_states, "StableMarking")?;
    if outcome.slot_maxima.len() != net.num_places()
        || outcome.slot_minima.len() != net.num_places()
    {
        return None;
    }
    Some(
        outcome
            .slot_minima
            .iter()
            .zip(&outcome.slot_maxima)
            .any(|(min, max)| min == max),
    )
}
