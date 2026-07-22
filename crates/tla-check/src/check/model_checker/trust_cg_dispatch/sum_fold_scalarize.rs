// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Recursive comm-assoc Sum-fold plan-time scalarization: a pre-lowering
//! bytecode rewrite that replaces `Sum(f, {body(v) : v \in D})` — a
//! commutative-associative RECURSIVE fold over a set produced by a
//! constant-cardinality set-builder — with the UNROLLED fixed-length sum
//! `f[body(v_1)] + f[body(v_2)] + ... + f[body(v_K)]`.
//!
//! Target shape (GameOfLife `score`):
//!
//! ```tla
//! score(p) == LET nbrs   == {x \in {-1,0,1} \X {-1,0,1} : x /= <<0,0>>}  \* const-folded
//!                 points == {<<p[1] + x, p[2] + y>> : <<x,y>> \in nbrs}
//!             IN Sum(sc, points)
//!
//! RECURSIVE Sum(_, _)
//! Sum(f, S) == IF S = {} THEN 0 ELSE LET x == CHOOSE x \in S : TRUE
//!                                    IN f[x] + Sum(f, S \ {x})
//! ```
//!
//! `Sum` is a recursive fold that CANNOT lower to the non-recursive native
//! backend, and `points` is a SET-VALUED set-builder whose native lowering
//! requires structural deduplication (also unsupported). The rewrite
//! eliminates both: `points` becomes a fixed list of `K` runtime tuple keys,
//! and `Sum` becomes a straight-line `K`-term addition.
//!
//! # Soundness (this is a verifier — a wrong term count is catastrophic)
//!
//! The set-builder maps into a SET, so if two output tuples coincide the set
//! dedups them and the true term count is < K. The rewrite therefore emits `K`
//! terms ONLY after PROVING the map is injective:
//!
//! * `points == {body(v) : v \in D}` where `D` is a compile-time-CONSTANT
//!   finite set (a `LoadConst`ed `Value::Set`);
//! * each output tuple component is provably a componentwise unit translation
//!   `p[c] + off_i` (or a `v`-derived constant) with the p-index `c` IDENTICAL
//!   across all `v` (structural), so `body(v) = body(w)` iff the two elements'
//!   OFFSET VECTORS are equal;
//! * the `K` offset vectors are PAIRWISE DISTINCT, hence the translation is
//!   injective and `|points| = K`.
//!
//! `Sum` is recognized only when its bytecode proves it is exactly
//! `\Sigma_{x \in S} f[x]` (empty-set base case returning the additive identity
//! `0`, a `CHOOSE`-arbitrary element `x`, the single non-recursive term `f[x]`,
//! the exact-single-element remainder `S \ {x}`, a self-recursive call on the
//! remainder, and an `AddInt` combiner). Anything that deviates from any of
//! these — a non-constant / non-`Value::Set` domain, a body that is not a
//! componentwise unit translation, coincident offset vectors, or an
//! unrecognized fold — fails CLOSED (`None`): the function is left untouched
//! and stays on its interpreter path. NEVER emit a wrong term count.
//!
//! The rewrite operates on a CLONE of the bytecode chunk that feeds ONLY the
//! native (trust-cg) compile task; the interpreter oracle (and the compiled-BFS
//! interpreter crosscheck) evaluate the ORIGINAL AST, so the crosscheck
//! genuinely validates every rewrite.

use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, ConstantPool, Opcode, Register};
use tla_value::Value;

/// Hard caps (all fail-closed).
const MAX_FOLD_TERMS: usize = 512;
const MAX_TUPLE_ARITY: usize = 8;
const MAX_TRACE_HOPS: usize = 16;
/// Highest register index we will allocate in the rewritten body. Leaves a wide
/// margin below `Register::MAX` (255).
const MAX_REWRITE_REGISTER: usize = 240;

/// Rewrite every function in `chunk` that matches the injective-translation
/// recursive-Sum-fold pattern into an unrolled straight-line sum.
///
/// Returns `Some(new_chunk)` when at least one function was rewritten (the
/// returned chunk shares the source constant pool — the rewrite never adds or
/// mutates pool entries), or `None` when nothing matched (a strict no-op; the
/// caller keeps its existing chunk).
pub(in crate::check) fn rewrite_chunk_injective_sum_folds(
    chunk: &BytecodeChunk,
) -> Option<BytecodeChunk> {
    let mut new_functions = chunk.functions.clone();
    let mut any = false;
    for idx in 0..new_functions.len() {
        if let Some(rewritten) = try_rewrite_sum_fold_function(&new_functions[idx], chunk) {
            new_functions[idx] = rewritten;
            any = true;
        }
    }
    if !any {
        return None;
    }
    Some(BytecodeChunk {
        constants: chunk.constants.clone(),
        functions: new_functions,
    })
}

// ---------------------------------------------------------------------------
// Register tracing (single-writer / dominance discipline, mirroring
// record_set_scalarize::chase_const_reg).
// ---------------------------------------------------------------------------

/// The UNIQUE writer of `reg` in the ENTIRE function, required to occur before
/// `before_pc`. A register with two writers (conditional reassignment) has no
/// single compile-time producer and fails closed. Returns `(writer_pc, op)`.
fn unique_writer<'a>(
    func: &'a BytecodeFunction,
    reg: Register,
    before_pc: usize,
) -> Option<(usize, &'a Opcode)> {
    let mut writers = func
        .instructions
        .iter()
        .enumerate()
        .filter(|(_, op)| op.dest_register() == Some(reg));
    let (writer_pc, writer) = writers.next()?;
    if writers.next().is_some() || writer_pc >= before_pc {
        return None;
    }
    Some((writer_pc, writer))
}

/// Trace `reg` backwards through `Move` aliases to its non-`Move` producer.
/// Returns `(final_reg, producer_pc, producer_op)`.
fn trace_producer<'a>(
    func: &'a BytecodeFunction,
    reg: Register,
    before_pc: usize,
) -> Option<(Register, usize, &'a Opcode)> {
    let mut reg = reg;
    let mut before = before_pc;
    for _ in 0..MAX_TRACE_HOPS {
        let (pc, op) = unique_writer(func, reg, before)?;
        match op {
            Opcode::Move { rs, .. } => {
                reg = *rs;
                before = pc;
            }
            _ => return Some((reg, pc, op)),
        }
    }
    None
}

/// Follow `reg` backward through `Move` aliases using the LAST writer strictly
/// before the current position, returning the terminal non-`Move` producer
/// `(terminal_reg, producer_pc, op)`.
///
/// Used only for RETURN-value dataflow (`... ; Move r = v ; Ret r`), which is
/// straight-line in the recognized shapes. Unlike [`unique_writer`] this
/// tolerates a return register that is reused elsewhere in the function (e.g.
/// `sc` reuses r0 for both a domain-building temp and its final result).
fn last_producer<'a>(
    func: &'a BytecodeFunction,
    reg: Register,
    before_pc: usize,
) -> Option<(Register, usize, &'a Opcode)> {
    let mut reg = reg;
    let mut before = before_pc;
    for _ in 0..MAX_TRACE_HOPS {
        let (pc, op) = func
            .instructions
            .iter()
            .enumerate()
            .take(before)
            .filter(|(_, op)| op.dest_register() == Some(reg))
            .next_back()?;
        match op {
            Opcode::Move { rs, .. } => {
                reg = *rs;
                before = pc;
            }
            _ => return Some((reg, pc, op)),
        }
    }
    None
}

/// Trace `reg` to a `LoadConst` and return the pooled constant value.
fn trace_const_value(
    func: &BytecodeFunction,
    reg: Register,
    before_pc: usize,
    pool: &ConstantPool,
) -> Option<Value> {
    let (_reg, _pc, op) = trace_producer(func, reg, before_pc)?;
    match op {
        Opcode::LoadConst { idx, .. } => Some(pool.get_value(*idx).clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Sum-fold recognition
// ---------------------------------------------------------------------------

/// A recognized `Sum(f, S)` fold call inside the candidate function.
struct SumCallSite {
    /// Register holding `f` (the folded function).
    f_reg: Register,
    /// Register holding `S` (the folded set).
    s_reg: Register,
    /// PC of the fold call instruction.
    call_pc: usize,
}

/// Is `value` the empty set (in any finite representation)?
fn is_empty_set(value: &Value) -> bool {
    match value {
        Value::Set(set) => set.is_empty(),
        Value::Interval(iv) => iv.is_empty(),
        _ => false,
    }
}

/// Verify `func` is exactly the commutative-associative fold
/// `Sum(f, S) == IF S = {} THEN 0 ELSE f[CHOOSE x \in S : TRUE] + Sum(f, S\{x})`,
/// proving it computes `\Sigma_{x \in S} f[x]` with additive identity `0`.
///
/// `visited` guards mutual/self recursion (the compiler emits two Sum
/// compilations — one recursing via `Call`, one via `CallExternal "Sum"`).
fn is_comm_assoc_sum_fold(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    pool: &ConstantPool,
    visited: &mut Vec<String>,
) -> bool {
    if func.arity != 2 {
        return false;
    }
    if visited.iter().any(|n| n == &func.name) {
        // Already being verified up the recursion stack: accept as the genuine
        // self-recursion of this same operator.
        return true;
    }
    if visited.len() > 4 {
        return false;
    }
    visited.push(func.name.clone());
    let ok = verify_sum_fold_body(func, chunk, pool, visited);
    visited.pop();
    ok
}

fn verify_sum_fold_body(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    pool: &ConstantPool,
    visited: &mut Vec<String>,
) -> bool {
    let f_param: Register = 0;
    let s_param: Register = 1;
    let instrs = &func.instructions;

    // (a) Base case: `S = {}` compared against an empty-set constant, and the
    //     empty branch must yield the additive identity 0. We check both that
    //     an `Eq(S, <empty const>)` exists and that a `LoadImm 0` (the base
    //     value) is present.
    let mut has_empty_test = false;
    for (pc, op) in instrs.iter().enumerate() {
        if let Opcode::Eq { r1, r2, .. } = op {
            let (other, s_side) = if *r1 == s_param {
                (*r2, true)
            } else if *r2 == s_param {
                (*r1, true)
            } else {
                (0, false)
            };
            if s_side {
                if let Some(v) = trace_const_value(func, other, pc, pool) {
                    if is_empty_set(&v) {
                        has_empty_test = true;
                        break;
                    }
                }
            }
        }
    }
    if !has_empty_test {
        return false;
    }
    // Additive identity base value present.
    if !instrs
        .iter()
        .any(|op| matches!(op, Opcode::LoadImm { value: 0, .. }))
    {
        return false;
    }

    // (b) CHOOSE x \in S : TRUE — an arbitrary element of S.
    let mut choose: Option<(usize, Register)> = None; // (pc, chosen-element reg = rd)
    for (pc, op) in instrs.iter().enumerate() {
        if let Opcode::ChooseBegin { rd, r_domain, .. } = op {
            // Domain must be S (possibly via Move aliases).
            if trace_reg_is(func, *r_domain, pc, s_param) {
                if choose.is_some() {
                    return false; // two CHOOSEs: not the recognized shape
                }
                choose = Some((pc, *rd));
            }
        }
    }
    let Some((_choose_pc, x_reg)) = choose else {
        return false;
    };

    // (c) The non-recursive term is f[x]: FuncApply(func=f, arg=x).
    let mut term_ok = false;
    for (pc, op) in instrs.iter().enumerate() {
        if let Opcode::FuncApply { func: ff, arg, .. } = op {
            if trace_reg_is(func, *ff, pc, f_param) && trace_reg_is(func, *arg, pc, x_reg) {
                term_ok = true;
                break;
            }
        }
    }
    if !term_ok {
        return false;
    }

    // (d) Remainder S \ {x}: SetDiff(S, SetEnum{x}).
    let mut rest_ok = false;
    for (pc, op) in instrs.iter().enumerate() {
        if let Opcode::SetDiff { r1, r2, .. } = op {
            if !trace_reg_is(func, *r1, pc, s_param) {
                continue;
            }
            // r2 must be the singleton {x}.
            if let Some((_r, _pc2, prod)) = trace_producer(func, *r2, pc) {
                if let Opcode::SetEnum {
                    start, count: 1, ..
                } = prod
                {
                    if trace_reg_is(func, *start, pc, x_reg) {
                        rest_ok = true;
                        break;
                    }
                }
            }
        }
    }
    if !rest_ok {
        return false;
    }

    // (e) Self-recursive fold call on (f, remainder), combined via AddInt.
    let mut recursion_ok = false;
    for (pc, op) in instrs.iter().enumerate() {
        let (rd, args_start, argc, target_name): (Register, Register, u8, Option<String>) = match op
        {
            Opcode::Call {
                rd,
                op_idx,
                args_start,
                argc,
            } => (
                *rd,
                *args_start,
                *argc,
                chunk
                    .functions
                    .get(*op_idx as usize)
                    .map(|f| f.name.clone()),
            ),
            Opcode::CallExternal {
                rd,
                name_idx,
                args_start,
                argc,
                ..
            } => (
                *rd,
                *args_start,
                *argc,
                match pool.get_value(*name_idx) {
                    Value::String(s) => Some(s.to_string()),
                    _ => None,
                },
            ),
            _ => continue,
        };
        if argc != 2 {
            continue;
        }
        // arg0 = f, arg1 = remainder (= S \ {x}); confirm structurally.
        if !trace_reg_is(func, args_start, pc, f_param) {
            continue;
        }
        // Verify the recursion target is itself a Sum-fold (mutual induction).
        let target_is_fold = match op {
            Opcode::Call { op_idx, .. } => chunk
                .functions
                .get(*op_idx as usize)
                .is_some_and(|f| is_comm_assoc_sum_fold(f, chunk, pool, visited)),
            Opcode::CallExternal { .. } => {
                // A CallExternal recursion is accepted when it names an operator
                // already on the verification stack (genuine self-recursion) or
                // a chunk function that itself passes the fold check.
                match &target_name {
                    Some(name) if visited.iter().any(|n| n == name) => true,
                    Some(name) => chunk
                        .functions
                        .iter()
                        .find(|f| &f.name == name)
                        .is_some_and(|f| is_comm_assoc_sum_fold(f, chunk, pool, visited)),
                    None => false,
                }
            }
            _ => false,
        };
        if !target_is_fold {
            continue;
        }
        // The result must be combined with f[x] via AddInt (either operand
        // order — addition is commutative).
        let combined = instrs.iter().enumerate().any(|(add_pc, add)| {
            if let Opcode::AddInt { r1, r2, .. } = add {
                let uses_rec =
                    trace_reg_is(func, *r1, add_pc, rd) || trace_reg_is(func, *r2, add_pc, rd);
                uses_rec
            } else {
                false
            }
        });
        if combined {
            recursion_ok = true;
            break;
        }
    }

    recursion_ok
}

/// Does `reg` trace (through `Move` aliases / dominance) back to `target`?
fn trace_reg_is(
    func: &BytecodeFunction,
    reg: Register,
    before_pc: usize,
    target: Register,
) -> bool {
    if reg == target {
        return true;
    }
    let mut reg = reg;
    let mut before = before_pc;
    for _ in 0..MAX_TRACE_HOPS {
        if reg == target {
            return true;
        }
        let Some((pc, op)) = unique_writer(func, reg, before) else {
            return reg == target;
        };
        match op {
            Opcode::Move { rs, .. } => {
                reg = *rs;
                before = pc;
            }
            _ => return reg == target,
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Set-builder body: symbolic componentwise-translation decomposition
// ---------------------------------------------------------------------------

/// One output-tuple component of `body(v)`.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Comp {
    /// `p[index] + offset` (unit translation of a single element of the outer
    /// binder `p`).
    ParamApply { index: i64, offset: i64 },
    /// A `v`-derived (or literal) constant with no `p` dependence.
    Const(i64),
}

impl Comp {
    /// The per-element VARYING coordinate used for the injectivity check: the
    /// translation offset for a `ParamApply`, or the constant itself.
    fn offset_coord(&self) -> i64 {
        match self {
            Comp::ParamApply { offset, .. } => *offset,
            Comp::Const(c) => *c,
        }
    }

    /// The STRUCTURAL signature (identical across all elements when the map is a
    /// well-formed translation): the discriminant plus the p-index.
    fn structure(&self) -> (u8, i64) {
        match self {
            Comp::ParamApply { index, .. } => (0, *index),
            Comp::Const(_) => (1, 0),
        }
    }
}

/// Symbolic register value during set-builder body evaluation for one concrete
/// (constant) domain element.
#[derive(Clone, Debug)]
enum SymVal {
    /// A compile-time constant value.
    Const(Value),
    /// The outer binder `p` itself (usable only as the function in
    /// `FuncApply(p, <const index>)`).
    Param,
    /// `p[index] + offset`.
    ParamApply { index: i64, offset: i64 },
    /// The output tuple `<<c_1, ..., c_k>>`.
    Tuple(Vec<Comp>),
}

/// Extract an `i64` from a scalar integer value (fail closed on non-integers /
/// out-of-range big integers).
fn value_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(b) => Some(i64::from(*b)),
        other => other.as_i64(),
    }
}

/// Apply a constant container to a constant key: `container[key]` for a
/// 1-indexed tuple/sequence.
fn apply_const(container: &Value, key: &Value) -> Option<Value> {
    let idx = value_i64(key)?;
    if idx < 1 {
        return None;
    }
    let pos = usize::try_from(idx - 1).ok()?;
    match container {
        Value::Tuple(elems) => elems.get(pos).cloned(),
        Value::Seq(seq) => seq.iter().nth(pos).cloned(),
        _ => None,
    }
}

/// Symbolically evaluate the (straight-line) set-builder body for one concrete
/// domain element `elem`, returning the componentwise decomposition of the
/// output tuple, or `None` (fail closed) on any unrecognized construct.
fn eval_body_for_element(
    func: &BytecodeFunction,
    body_range: std::ops::Range<usize>,
    r_binding: Register,
    param_reg: Register,
    r_body: Register,
    elem: &Value,
    pool: &ConstantPool,
) -> Option<Vec<Comp>> {
    let mut syms: Vec<Option<SymVal>> = vec![None; 256];
    syms[param_reg as usize] = Some(SymVal::Param);
    syms[r_binding as usize] = Some(SymVal::Const(elem.clone()));

    for pc in body_range.clone() {
        let op = &func.instructions[pc];
        match op {
            Opcode::LoadImm { rd, value } => {
                syms[*rd as usize] = Some(SymVal::Const(Value::SmallInt(*value)));
            }
            Opcode::LoadBool { rd, value } => {
                syms[*rd as usize] = Some(SymVal::Const(Value::Bool(*value)));
            }
            Opcode::LoadConst { rd, idx } => {
                syms[*rd as usize] = Some(SymVal::Const(pool.get_value(*idx).clone()));
            }
            Opcode::Move { rd, rs } => {
                syms[*rd as usize] = syms[*rs as usize].clone();
            }
            Opcode::FuncApply { rd, func: ff, arg } => {
                let fv = syms[*ff as usize].clone();
                let av = syms[*arg as usize].clone();
                let result = match (fv, av) {
                    (Some(SymVal::Param), Some(SymVal::Const(k))) => {
                        let index = value_i64(&k)?;
                        SymVal::ParamApply { index, offset: 0 }
                    }
                    (Some(SymVal::Const(container)), Some(SymVal::Const(key))) => {
                        SymVal::Const(apply_const(&container, &key)?)
                    }
                    _ => return None,
                };
                syms[*rd as usize] = Some(result);
            }
            Opcode::AddInt { rd, r1, r2 } => {
                let a = syms[*r1 as usize].clone();
                let b = syms[*r2 as usize].clone();
                let result = match (a, b) {
                    (Some(SymVal::Const(x)), Some(SymVal::Const(y))) => {
                        SymVal::Const(Value::SmallInt(value_i64(&x)?.checked_add(value_i64(&y)?)?))
                    }
                    (Some(SymVal::ParamApply { index, offset }), Some(SymVal::Const(c)))
                    | (Some(SymVal::Const(c)), Some(SymVal::ParamApply { index, offset })) => {
                        SymVal::ParamApply {
                            index,
                            offset: offset.checked_add(value_i64(&c)?)?,
                        }
                    }
                    _ => return None,
                };
                syms[*rd as usize] = Some(result);
            }
            Opcode::SubInt { rd, r1, r2 } => {
                let a = syms[*r1 as usize].clone();
                let b = syms[*r2 as usize].clone();
                let result = match (a, b) {
                    (Some(SymVal::Const(x)), Some(SymVal::Const(y))) => {
                        SymVal::Const(Value::SmallInt(value_i64(&x)?.checked_sub(value_i64(&y)?)?))
                    }
                    // `p[i] + off - c` stays a unit translation.
                    (Some(SymVal::ParamApply { index, offset }), Some(SymVal::Const(c))) => {
                        SymVal::ParamApply {
                            index,
                            offset: offset.checked_sub(value_i64(&c)?)?,
                        }
                    }
                    // `c - p[i]` negates the coefficient: NOT a unit translation.
                    _ => return None,
                };
                syms[*rd as usize] = Some(result);
            }
            Opcode::TupleNew { rd, start, count } => {
                let count = *count as usize;
                if count == 0 || count > MAX_TUPLE_ARITY {
                    return None;
                }
                let mut comps = Vec::with_capacity(count);
                for j in 0..count {
                    let idx = usize::from(*start) + j;
                    match syms.get(idx).and_then(|s| s.clone()) {
                        Some(SymVal::ParamApply { index, offset }) => {
                            comps.push(Comp::ParamApply { index, offset });
                        }
                        Some(SymVal::Const(c)) => comps.push(Comp::Const(value_i64(&c)?)),
                        _ => return None,
                    }
                }
                syms[*rd as usize] = Some(SymVal::Tuple(comps));
            }
            // Any other opcode inside the body is unrecognized → fail closed.
            _ => return None,
        }
    }

    match syms.get(r_body as usize).and_then(|s| s.clone()) {
        Some(SymVal::Tuple(comps)) => Some(comps),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Set-builder loop-shape recovery
// ---------------------------------------------------------------------------

struct SetBuilderLoop {
    begin_pc: usize,
    loopnext_pc: usize,
    r_binding: Register,
    r_domain: Register,
    r_body: Register,
}

/// Recover the `SetBuilderBegin`/`LoopNext` pair whose result register is
/// `s_reg` (traced through `Move` aliases), validating the offsets both ways.
fn recover_set_builder_loop(func: &BytecodeFunction, s_reg: Register) -> Option<SetBuilderLoop> {
    // Locate the SetBuilderBegin that writes s_reg's producer.
    let (builder_reg, begin_pc, op) = trace_producer(func, s_reg, func.instructions.len())?;
    let Opcode::SetBuilderBegin {
        rd,
        r_binding,
        r_domain,
        loop_end,
    } = op
    else {
        return None;
    };
    if *rd != builder_reg {
        return None;
    }
    let end = begin_pc.checked_add(usize::try_from(*loop_end).ok()?)?;
    let loopnext_pc = end.checked_sub(1)?;
    if loopnext_pc <= begin_pc || loopnext_pc >= func.instructions.len() {
        return None;
    }
    let Opcode::LoopNext {
        r_binding: next_binding,
        r_body,
        loop_begin,
    } = func.instructions[loopnext_pc]
    else {
        return None;
    };
    if next_binding != *r_binding {
        return None;
    }
    // Validate the back edge points to begin_pc + 1.
    if (loopnext_pc as i64) + i64::from(loop_begin) != (begin_pc as i64) + 1 {
        return None;
    }
    Some(SetBuilderLoop {
        begin_pc,
        loopnext_pc,
        r_binding: *r_binding,
        r_domain: *r_domain,
        r_body,
    })
}

// ---------------------------------------------------------------------------
// The top-level per-function rewrite
// ---------------------------------------------------------------------------

/// Try to rewrite `func` (a `score`-like helper) into an unrolled sum. Fail
/// closed (`None`) on any deviation from the fully-recognized, fully-proven
/// shape.
fn try_rewrite_sum_fold_function(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
) -> Option<BytecodeFunction> {
    // The recognized outer helper takes a single parameter `p` in register 0.
    if func.arity != 1 {
        return None;
    }
    let param_reg: Register = 0;
    let pool = &chunk.constants;

    // 1. The returned value must be a Sum-fold call `Sum(f, S)`.
    let site = find_returned_sum_call(func, chunk, pool)?;

    // 2. `S` must be a constant-cardinality set-builder `{body(v) : v \in D}`.
    let loop_info = recover_set_builder_loop(func, site.s_reg)?;
    // Guard: the fold call must come after the set-builder completes.
    if site.call_pc <= loop_info.loopnext_pc {
        return None;
    }
    let domain = trace_const_value(func, loop_info.r_domain, loop_info.begin_pc, pool)?;
    let elements: Vec<Value> = match &domain {
        Value::Set(set) => set.iter().cloned().collect(),
        _ => return None,
    };
    if elements.is_empty() || elements.len() > MAX_FOLD_TERMS {
        return None;
    }

    // 3. Decompose the body per element; prove structural consistency + the
    //    injective-translation property.
    let body_range = (loop_info.begin_pc + 1)..loop_info.loopnext_pc;
    // The set-builder body must not overwrite the outer binder `p` (register 0)
    // — else the per-element decomposition would read a stale `p`.
    if body_range
        .clone()
        .any(|pc| func.instructions[pc].dest_register() == Some(param_reg))
    {
        return None;
    }
    let mut decompositions: Vec<Vec<Comp>> = Vec::with_capacity(elements.len());
    let mut structure: Option<Vec<(u8, i64)>> = None;
    let mut offset_vectors: Vec<Vec<i64>> = Vec::with_capacity(elements.len());
    for elem in &elements {
        let comps = eval_body_for_element(
            func,
            body_range.clone(),
            loop_info.r_binding,
            param_reg,
            loop_info.r_body,
            elem,
            pool,
        )?;
        if comps.is_empty() || comps.len() > MAX_TUPLE_ARITY {
            return None;
        }
        // Structural consistency: same discriminant + p-index per position.
        let this_structure: Vec<(u8, i64)> = comps.iter().map(Comp::structure).collect();
        match &structure {
            None => structure = Some(this_structure),
            Some(existing) if existing == &this_structure => {}
            Some(_) => return None,
        }
        offset_vectors.push(comps.iter().map(Comp::offset_coord).collect());
        decompositions.push(comps);
    }
    // INJECTIVITY PROOF: the offset vectors must be PAIRWISE DISTINCT, so the
    // translation `v |-> body(v)` maps the K distinct domain elements to K
    // distinct output tuples for EVERY value of `p`. Coincident vectors would
    // dedup in the target set and make the K-term sum over-count → catastrophic.
    {
        let mut sorted = offset_vectors.clone();
        sorted.sort();
        let unique = {
            let mut u = sorted.clone();
            u.dedup();
            u.len()
        };
        if unique != offset_vectors.len() {
            return None;
        }
    }

    // 4. `f` must be a 0-ary operator (`Call sc()`) whose body is a single
    //    `FuncDef`. A tuple-keyed compact function cannot be returned across the
    //    native call ABI (its domain has no scalar `CompactFunctionDomain`
    //    encoding), so rather than materialize the whole function and apply it
    //    per key we INLINE the FuncDef's body with the binding replaced by each
    //    key — the state reads inside the body (`grid[<<x,y>>]`) then lower
    //    through the already-supported state-tuple FuncApply.
    let (_f_reg, _f_pc, f_op) = trace_producer(func, site.f_reg, site.call_pc)?;
    let sc_idx = match f_op {
        Opcode::Call {
            op_idx, argc: 0, ..
        } => *op_idx,
        _ => return None,
    };
    let sc = chunk.functions.get(sc_idx as usize)?;
    let sc_body = extract_scalar_funcdef_body(sc)?;

    // 5. Build the unrolled + inlined function.
    build_unrolled_inline(func, &sc_body, &decompositions)
}

/// A `sc`-like 0-ary operator's `FuncDef` body, extracted for per-key inlining.
struct FuncDefBody {
    /// Verbatim body instructions (`FuncDefBegin+1 .. LoopNext`). Their internal
    /// jump offsets are relative, so a contiguous copy relocates correctly with
    /// no adjustment, and the terminal "skip else" jump to the (absent) LoopNext
    /// lands exactly on the instruction we append after the copy.
    body: Vec<Opcode>,
    /// The FuncDef binding register (`<<x,y>> \in DOM`), overwritten with the
    /// current key before each inlined copy.
    r_binding: Register,
    /// The register holding the body's result value (`LoopNext.r_body`).
    r_result: Register,
    /// Highest register index referenced anywhere in the body.
    max_reg: Register,
}

/// Extract the FuncDef body from a 0-ary `[k \in DOM |-> body]` operator, or
/// `None` (fail closed) if `sc` is not exactly one returned FuncDef with a
/// self-contained, relocatable straight-line-plus-forward-branch body.
fn extract_scalar_funcdef_body(sc: &BytecodeFunction) -> Option<FuncDefBody> {
    if sc.arity != 0 {
        return None;
    }
    let instrs = &sc.instructions;

    // Locate the unique FuncDefBegin and its matching LoopNext.
    let mut begin: Option<(usize, Register, Register, i32)> = None; // (pc, rd, r_binding, loop_end)
    for (pc, op) in instrs.iter().enumerate() {
        if let Opcode::FuncDefBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } = op
        {
            if begin.is_some() {
                return None; // more than one FuncDef: not the recognized shape
            }
            begin = Some((pc, *rd, *r_binding, *loop_end));
        }
    }
    let (begin_pc, fd_rd, r_binding, loop_end) = begin?;
    let end = begin_pc.checked_add(usize::try_from(loop_end).ok()?)?;
    let loopnext_pc = end.checked_sub(1)?;
    if loopnext_pc <= begin_pc + 1 || loopnext_pc >= instrs.len() {
        return None;
    }
    let Opcode::LoopNext {
        r_binding: next_binding,
        r_body,
        loop_begin,
    } = instrs[loopnext_pc]
    else {
        return None;
    };
    if next_binding != r_binding {
        return None;
    }
    if (loopnext_pc as i64) + i64::from(loop_begin) != (begin_pc as i64) + 1 {
        return None;
    }

    // The FuncDef must be the returned value: the unique Ret's value, traced
    // through its (straight-line) Move chain, is the FuncDef result register.
    let mut ret_reg: Option<Register> = None;
    for op in instrs {
        if let Opcode::Ret { rs } = op {
            if ret_reg.is_some() {
                return None;
            }
            ret_reg = Some(*rs);
        }
    }
    let (ret_term, _pc, _op) = last_producer(sc, ret_reg?, instrs.len())?;
    if ret_term != fd_rd {
        return None;
    }

    let body_start = begin_pc + 1;
    let body: Vec<Opcode> = instrs[body_start..loopnext_pc].to_vec();

    // Validate the body: relocatable, self-contained, side-effect-free.
    let mut written: Vec<bool> = vec![false; 256];
    written[r_binding as usize] = true; // the binding is provided per key
    let mut max_reg: Register = r_binding;
    for (i, op) in body.iter().enumerate() {
        // No nested loops / returns / state writes / next-state reads: these
        // either can't be relocated verbatim or aren't pure functions of the
        // binding + current state.
        match op {
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
            | Opcode::Ret { .. }
            | Opcode::Halt
            | Opcode::StoreVar { .. }
            | Opcode::LoadPrime { .. }
            | Opcode::SetPrimeMode { .. } => return None,
            _ => {}
        }
        // Forward/relocatable branch targets must stay within the body (or land
        // exactly on the continuation appended after the copy = loopnext_pc).
        if let Some(offset) = branch_offset(op) {
            let abs = body_start as i64 + i as i64;
            let target = abs + i64::from(offset);
            if target < body_start as i64 || target > loopnext_pc as i64 {
                return None;
            }
        }
        // Self-containment: every source register is the binding or is written
        // earlier within the body. An unrecognized opcode (whose source set we
        // cannot enumerate) fails closed.
        for src in source_registers(op)? {
            if !written[src as usize] {
                return None;
            }
            max_reg = max_reg.max(src);
        }
        // Every register the body WRITES must be >= the binding, so the inlined
        // copy stays within the window [binding, max_reg] and never clobbers
        // `score`'s param / accumulator / key-building scratch (all < binding).
        if let Some(rd) = op.dest_register() {
            if rd < r_binding {
                return None;
            }
            written[rd as usize] = true;
            max_reg = max_reg.max(rd);
        }
        if let Some(rb) = op.binding_register() {
            if rb < r_binding {
                return None;
            }
            written[rb as usize] = true;
            max_reg = max_reg.max(rb);
        }
    }

    Some(FuncDefBody {
        body,
        r_binding,
        r_result: r_body,
        max_reg,
    })
}

/// The relative branch offset of a jump opcode, if any.
fn branch_offset(op: &Opcode) -> Option<i32> {
    match op {
        Opcode::Jump { offset }
        | Opcode::JumpTrue { offset, .. }
        | Opcode::JumpFalse { offset, .. } => Some(*offset),
        _ => None,
    }
}

/// Every source (read) register of `op`, or `None` if `op` is not one of the
/// pure, relocatable opcodes we recognize (fail closed).
fn source_registers(op: &Opcode) -> Option<Vec<Register>> {
    let regs = match op {
        // No source registers.
        Opcode::LoadImm { .. }
        | Opcode::LoadBool { .. }
        | Opcode::LoadConst { .. }
        | Opcode::LoadVar { .. }
        | Opcode::Jump { .. }
        | Opcode::Nop => Vec::new(),
        Opcode::Move { rs, .. }
        | Opcode::NegInt { rs, .. }
        | Opcode::Not { rs, .. }
        | Opcode::Powerset { rs, .. }
        | Opcode::BigUnion { rs, .. }
        | Opcode::Domain { rs, .. }
        | Opcode::RecordGet { rs, .. }
        | Opcode::JumpTrue { rs, .. }
        | Opcode::JumpFalse { rs, .. }
        | Opcode::TupleGet { rs, .. } => vec![*rs],
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
        | Opcode::Concat { r1, r2, .. } => vec![*r1, *r2],
        Opcode::FuncApply { func, arg, .. } => vec![*func, *arg],
        Opcode::SetIn { elem, set, .. } => vec![*elem, *set],
        Opcode::SetEnumSubseteq {
            start, count, set, ..
        } => {
            let mut regs = (0..*count)
                .map(|i| start.wrapping_add(i))
                .collect::<Vec<_>>();
            regs.push(*set);
            regs
        }
        Opcode::Range { lo, hi, .. } => vec![*lo, *hi],
        Opcode::KSubset { base, k, .. } => vec![*base, *k],
        Opcode::CondMove { cond, rs, .. } => vec![*cond, *rs],
        Opcode::FuncExcept {
            func, path, val, ..
        } => vec![*func, *path, *val],
        Opcode::FuncSet { domain, range, .. } => vec![*domain, *range],
        // Contiguous source blocks.
        Opcode::TupleNew { start, count, .. }
        | Opcode::SeqNew { start, count, .. }
        | Opcode::SetEnum { start, count, .. }
        | Opcode::Times { start, count, .. }
        | Opcode::RecordNew {
            values_start: start,
            count,
            ..
        }
        | Opcode::RecordSet {
            values_start: start,
            count,
            ..
        } => (0..*count).map(|i| start.wrapping_add(i)).collect(),
        Opcode::ValueApply {
            func,
            args_start,
            argc,
            ..
        } => std::iter::once(*func)
            .chain((0..*argc).map(|i| args_start.wrapping_add(i)))
            .collect(),
        Opcode::Call {
            args_start, argc, ..
        }
        | Opcode::CallExternal {
            args_start, argc, ..
        }
        | Opcode::CallBuiltin {
            args_start, argc, ..
        } => (0..*argc).map(|i| args_start.wrapping_add(i)).collect(),
        Opcode::MakeClosure {
            captures_start,
            capture_count,
            ..
        } => (0..*capture_count)
            .map(|i| captures_start.wrapping_add(i))
            .collect(),
        // Anything else (loop Begin/Next, Ret, StoreVar, state-prime ops, Halt,
        // ...) is rejected by the caller's opcode filter before reaching here;
        // treat any remaining unrecognized opcode as fail-closed.
        _ => return None,
    };
    Some(regs)
}

/// Locate the fold call `Sum(f, S)` whose result is returned by `func`.
fn find_returned_sum_call(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    pool: &ConstantPool,
) -> Option<SumCallSite> {
    // Find the (unique) terminal `Ret` and trace its value to a fold call.
    let mut ret_reg: Option<Register> = None;
    for op in &func.instructions {
        if let Opcode::Ret { rs } = op {
            if ret_reg.is_some() {
                return None; // multiple returns: not the recognized shape
            }
            ret_reg = Some(*rs);
        }
    }
    let ret_reg = ret_reg?;
    let (_r, call_pc, op) = last_producer(func, ret_reg, func.instructions.len())?;
    let (args_start, argc, is_fold) = match op {
        Opcode::Call {
            op_idx,
            args_start,
            argc,
            ..
        } => (
            *args_start,
            *argc,
            chunk
                .functions
                .get(*op_idx as usize)
                .is_some_and(|f| is_comm_assoc_sum_fold(f, chunk, pool, &mut Vec::new())),
        ),
        Opcode::CallExternal {
            name_idx,
            args_start,
            argc,
            ..
        } => {
            let name = match pool.get_value(*name_idx) {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            };
            let is_fold = name.as_deref().is_some_and(|name| {
                chunk
                    .functions
                    .iter()
                    .find(|f| f.name == name)
                    .is_some_and(|f| is_comm_assoc_sum_fold(f, chunk, pool, &mut Vec::new()))
            });
            (*args_start, *argc, is_fold)
        }
        _ => return None,
    };
    if argc != 2 || !is_fold {
        return None;
    }
    Some(SumCallSite {
        f_reg: args_start,
        s_reg: args_start.checked_add(1)?,
        call_pc,
    })
}

/// Emit the unrolled + inlined function (see module docs):
///
/// ```text
/// ACC = 0
/// for comps in decompositions:      ; each = body(v_i), the i-th neighbour key
///     BINDING = << comps... >>       ; sc's FuncDef binding, set to this key
///     <inlined sc FuncDef body>      ; verbatim copy; computes sc[key] -> RESULT
///     ACC = ACC + RESULT
/// return ACC
/// ```
///
/// `score`'s own working registers occupy `[0, binding)`; the inlined body only
/// ever touches `[binding, max_reg]` (validated in `extract_scalar_funcdef_body`),
/// so the verbatim copy never collides with `p`, the accumulator, or the
/// key-building scratch.
fn build_unrolled_inline(
    orig: &BytecodeFunction,
    sc_body: &FuncDefBody,
    decompositions: &[Vec<Comp>],
) -> Option<BytecodeFunction> {
    let tuple_arity = decompositions.first()?.len();
    if tuple_arity == 0 || tuple_arity > MAX_TUPLE_ARITY {
        return None;
    }
    let binding = sc_body.r_binding;

    // Layout, strictly below `binding`: p=0, ACC=1, component block, SA, SB, SC.
    let acc: Register = 1;
    let cb_base: usize = 2;
    let sa = cb_base + tuple_arity;
    let sb = cb_base + tuple_arity + 1;
    let sc_scratch = cb_base + tuple_arity + 2;
    if sc_scratch >= usize::from(binding) || usize::from(sc_body.max_reg) > MAX_REWRITE_REGISTER {
        return None;
    }
    let sa = sa as Register;
    let sb = sb as Register;
    let sc_scratch = sc_scratch as Register;

    let mut out = BytecodeFunction::new(orig.name.clone(), 1);
    // Size the register file for the inlined body's registers even if a
    // particular copy path doesn't reference its very top one.
    out.max_register = out.max_register.max(sc_body.max_reg);

    out.emit(Opcode::LoadImm { rd: acc, value: 0 });

    for comps in decompositions {
        if comps.len() != tuple_arity {
            return None;
        }
        for (j, comp) in comps.iter().enumerate() {
            let slot = (cb_base + j) as Register;
            match comp {
                Comp::ParamApply { index, offset } => {
                    out.emit(Opcode::LoadImm {
                        rd: sa,
                        value: *index,
                    });
                    out.emit(Opcode::FuncApply {
                        rd: sb,
                        func: 0, // p
                        arg: sa,
                    });
                    if *offset != 0 {
                        out.emit(Opcode::LoadImm {
                            rd: sa,
                            value: *offset,
                        });
                        out.emit(Opcode::AddInt {
                            rd: sc_scratch,
                            r1: sb,
                            r2: sa,
                        });
                        out.emit(Opcode::Move {
                            rd: slot,
                            rs: sc_scratch,
                        });
                    } else {
                        out.emit(Opcode::Move { rd: slot, rs: sb });
                    }
                }
                Comp::Const(c) => {
                    out.emit(Opcode::LoadImm {
                        rd: slot,
                        value: *c,
                    });
                }
            }
        }
        // The key tuple becomes sc's FuncDef binding.
        out.emit(Opcode::TupleNew {
            rd: binding,
            start: cb_base as Register,
            count: u8::try_from(tuple_arity).ok()?,
        });
        // Inline sc's body verbatim. Its relative branch offsets relocate
        // correctly under a contiguous copy, and the terminal "skip else" jump
        // (to the absent LoopNext) lands exactly on the AddInt appended next.
        for op in &sc_body.body {
            out.emit(*op);
        }
        out.emit(Opcode::AddInt {
            rd: sc_scratch,
            r1: acc,
            r2: sc_body.r_result,
        });
        out.emit(Opcode::Move {
            rd: acc,
            rs: sc_scratch,
        });
    }

    out.emit(Opcode::Move { rd: 0, rs: acc });
    out.emit(Opcode::Ret { rs: 0 });
    Some(out)
}

#[cfg(test)]
mod tests;
