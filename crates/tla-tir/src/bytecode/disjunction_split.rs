// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Sound top-level action-disjunction split for JIT compilation.
//!
//! A TLA+ action of the shape
//!
//! ```text
//! A == guard /\ ( D1 \/ D2 \/ ... \/ Dn ) [ /\ trailing ]
//! ```
//!
//! produces, as its successor set, the UNION of the successor sets of the
//! per-disjunct sub-actions `guard /\ Di [ /\ trailing ]` (disjunction = set
//! union). When two or more `Di` each contain an inner `\E` quantifier the
//! single-successor native ABI cannot lower the monolithic action (the
//! [`super::static_expansion_drops_sibling_successor`] guard fails closed to
//! avoid silently dropping a sibling disjunct's successors). Splitting the
//! top-level disjunction into separate sub-actions gives each at most one inner
//! `\E` pair, re-enabling the existing single-pair expansion — soundly, because
//! the BFS engine already unions the successors of distinct native action
//! functions.
//!
//! # Soundness (union exactness)
//!
//! The split is performed structurally on the canonical OR-chain bytecode the
//! compiler emits for `D1 \/ ... \/ Dn` (see `compile_bool_binop`'s `Or` arm):
//!
//! ```text
//!   <D1 code -> r1> ; Move OR<-r1 ; JumpTrue OR -> end
//!   <D2 code -> r2> ; Move OR<-r2 ; JumpTrue OR -> end
//!   ...
//!   <Dn code -> rn> ; Move OR<-rn          ; (end is here)
//! ```
//!
//! For sub-action `i` we keep `Di` intact and *neutralize* every sibling
//! disjunct `j != i` by
//!
//! 1. replacing its join `Move { rd: OR, rs: r_j }` with `LoadBool { OR, false }`
//!    (so `Dj` contributes `false` to the disjunction), and
//! 2. replacing every `StoreVar` inside `Dj`'s span with `Nop` (so `Dj`
//!    produces no successor writes).
//!
//! The OR-result then evaluates to `false \/ ... \/ Di \/ ... \/ false = Di`,
//! and only `Di`'s `StoreVar`s execute. Hence sub-action `i` produces EXACTLY
//! the successor set of `guard /\ Di [ /\ trailing ]`. The guard and any
//! trailing conjuncts (e.g. `UNCHANGED <<...>>`) lie OUTSIDE the OR span, so
//! they are preserved identically in every sub-action. Therefore
//!
//! ```text
//!   successors(A) = U_i successors(guard /\ Di /\ trailing)
//!                 = U_i successors(sub-action i)
//! ```
//!
//! No successor is dropped and none is fabricated; duplicates across sub-actions
//! (when several `Di` hold for the same parent) are folded by the BFS engine's
//! fingerprint set, so an over-approximate union is still exact as a set.
//!
//! # Fail-closed
//!
//! The detector matches ONLY the exact canonical OR-chain pattern. Anything it
//! does not recognize returns `None`, leaving the action on its existing path
//! (which itself fails closed to the interpreter). Over-rejection only loses a
//! native-compilation opportunity; it never changes results.

use super::opcode::{Opcode, Register};

/// `TY_DEBUG_DISJ_SPLIT=1`: trace fail-closed points in the splitter.
fn split_debug() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var_os("TY_DEBUG_DISJ_SPLIT").is_some())
}
macro_rules! split_trace {
    ($($arg:tt)*) => {
        if split_debug() { eprintln!("[disj-split] {}", format!($($arg)*)); }
    };
}
use super::BytecodeFunction;

/// A detected top-level disjunction: the OR-result register, the end PC of the
/// chain, and the per-disjunct join-`Move` PCs (in source order).
#[derive(Debug, Clone)]
struct DisjunctionChain {
    /// The register that accumulates the disjunction result (`OR`).
    or_reg: Register,
    /// PC one past the last disjunct's join `Move` (the OR-chain's `end`).
    end_pc: usize,
    /// For each disjunct, the PC of its `Move { rd: OR, rs: r_i }` join.
    join_pcs: Vec<usize>,
    /// PC of the first join `Move` (used to rank the outermost chain).
    chain_start: usize,
    /// First body PC of disjunct 0 (resolved by [`resolve_body_start_d0`]).
    body_start_d0: usize,
}

/// Count the inner `ExistsBegin` opcodes in `func`.
fn exists_begin_count(func: &BytecodeFunction) -> usize {
    func.instructions
        .iter()
        .filter(|op| matches!(op, Opcode::ExistsBegin { .. }))
        .count()
}

/// Detect the OUTERMOST canonical OR chain in `func`, if exactly one exists and
/// it has two or more disjuncts.
///
/// The compiler emits `D1 \/ D2 \/ ... \/ Dn` left-nested as a flat sequence of
/// `Move { rd: OR, rs: r_i }` join writes to a single shared `OR` register,
/// each of the first `n-1` immediately followed by `JumpTrue { rs: OR, offset }`
/// that targets the common `end` (the PC right after the final join `Move`).
///
/// We locate the chain by its `JumpTrue` skeleton: a maximal run of
/// `Move OR<-ri ; JumpTrue OR -> end` pairs that share one `OR` register and one
/// `end` target, terminated by a final `Move OR<-rn` sitting at `end - 1`.
fn detect_top_level_disjunction(func: &BytecodeFunction) -> Option<DisjunctionChain> {
    let instrs = &func.instructions;
    let len = instrs.len();

    // Collect every `Move OR<-r ; JumpTrue OR -> tgt` adjacency. Group by
    // (OR, tgt). A real OR chain is one such group whose final disjunct's join
    // `Move OR<-rn` sits exactly at `tgt - 1`.
    let mut best: Option<DisjunctionChain> = None;

    for (pc, op) in instrs.iter().enumerate() {
        let Opcode::JumpTrue { rs: or_reg, offset } = *op else {
            continue;
        };
        if offset <= 0 {
            continue;
        }
        // The JumpTrue must be immediately preceded by `Move { rd: OR, rs: _ }`.
        if pc == 0 {
            continue;
        }
        if !matches!(instrs[pc - 1], Opcode::Move { rd, .. } if rd == or_reg) {
            continue;
        }
        let Some(end_pc) = pc.checked_add(offset as usize) else {
            continue;
        };
        if end_pc > len || end_pc == 0 {
            continue;
        }
        // The chain must terminate with the final disjunct's join
        // `Move { rd: OR, rs: rn }` at `end_pc - 1`.
        if !matches!(instrs[end_pc - 1], Opcode::Move { rd, .. } if rd == or_reg) {
            continue;
        }

        // Walk forward from this JumpTrue, collecting every join `Move OR<-r`
        // whose paired (or final) position belongs to this chain. We accept a
        // disjunct boundary as: a `Move { rd: OR, rs: _ }` that is either
        // (a) immediately followed by `JumpTrue { rs: OR, -> end_pc }`, or
        // (b) located exactly at `end_pc - 1` (the final disjunct).
        let Some(chain) = collect_chain(func, or_reg, pc - 1, end_pc) else {
            continue;
        };
        // Keep the OUTERMOST / longest chain. Ties: smallest start.
        let take = match &best {
            None => true,
            Some(prev) => {
                chain.join_pcs.len() > prev.join_pcs.len()
                    || (chain.join_pcs.len() == prev.join_pcs.len()
                        && chain.chain_start < prev.chain_start)
            }
        };
        if take {
            best = Some(chain);
        }
    }

    let chain = best?;
    // Require at least two disjuncts; a single-disjunct "chain" is not a
    // disjunction and needs no split.
    if chain.join_pcs.len() < 2 {
        return None;
    }
    Some(chain)
}

/// Given the OR register and the PC of the FIRST join `Move` (`first_join_pc`)
/// plus the common `end_pc`, collect all join PCs of the chain in source order.
///
/// The first join is at `first_join_pc`; subsequent joins are the `Move OR<-r`
/// opcodes between `first_join_pc` and `end_pc` that are followed by a matching
/// `JumpTrue OR -> end_pc`, plus the final `Move OR<-rn` at `end_pc - 1`.
fn collect_chain(
    func: &BytecodeFunction,
    or_reg: Register,
    first_join_pc: usize,
    end_pc: usize,
) -> Option<DisjunctionChain> {
    let instrs = &func.instructions;
    let mut join_pcs: Vec<usize> = Vec::new();

    // Walk the region [first_join_pc, end_pc) and record join boundaries.
    let mut pc = first_join_pc;
    while pc < end_pc {
        let is_join = matches!(instrs[pc], Opcode::Move { rd, .. } if rd == or_reg);
        if is_join {
            let final_join = pc == end_pc - 1;
            let followed_by_jump = pc + 1 < end_pc
                && matches!(
                    instrs[pc + 1],
                    Opcode::JumpTrue { rs, offset }
                        if rs == or_reg
                            && pc + 1 + (offset.max(0) as usize) == end_pc
                );
            if final_join {
                join_pcs.push(pc);
                pc += 1;
                continue;
            }
            if followed_by_jump {
                join_pcs.push(pc);
                // Skip past the join and its JumpTrue.
                pc += 2;
                continue;
            }
            // A `Move OR<-r` that is neither the final join nor followed by the
            // chain's JumpTrue means `OR` is reused outside the canonical chain
            // shape. Fail closed.
            return None;
        }
        pc += 1;
    }

    // The final join must land exactly on end_pc - 1.
    if join_pcs.last().copied() != Some(end_pc - 1) {
        return None;
    }

    // The first disjunct's code starts at the OR chain's beginning. We take the
    // chain_start as the PC after the previous structural boundary is unknown
    // here; the first disjunct's body precedes `first_join_pc`. For span
    // computation we only need each disjunct's [start, join] where start is the
    // instruction right after the previous disjunct's JumpTrue (or, for the
    // first disjunct, the join's own start is not needed because we never
    // neutralize the *guard*; we only neutralize sibling spans which are
    // bounded below by the previous boundary). We therefore record chain_start
    // as the first join pc's containing region start = the instruction after
    // the prior JumpTrue, which for disjunct 0 we treat as 0-safe (its span is
    // bounded above by its join and below by chain_start computed per-disjunct
    // in `neutralize`).
    let chain_start = first_join_pc;

    Some(DisjunctionChain {
        or_reg,
        end_pc,
        join_pcs,
        chain_start,
        body_start_d0: 0,
    })
}

/// Compute the half-open instruction span `[start, join]` (inclusive of the
/// join `Move`) owned by disjunct `idx`.
///
/// Disjunct 0's code begins at the chain start (the first disjunct body); each
/// later disjunct begins right after the previous disjunct's `JumpTrue`
/// (i.e. previous join PC + 2). The span ends at (and includes) the disjunct's
/// own join `Move`.
fn disjunct_span(chain: &DisjunctionChain, idx: usize) -> (usize, usize) {
    let join = chain.join_pcs[idx];
    let start = if idx == 0 {
        // Disjunct 0's body starts right after the guard prefix (resolved by
        // `resolve_body_start_d0`).
        chain.body_start_d0
    } else {
        // Right after the previous disjunct's `JumpTrue` (join + 1 is the
        // JumpTrue, join + 2 is the next disjunct's first body instruction).
        chain.join_pcs[idx - 1] + 2
    };
    (start, join)
}

/// Produce one sub-action bytecode function per disjunct of a detected
/// top-level disjunction, or `None` if `func` does not match the canonical
/// `guard /\ (D1 \/ ... \/ Dn)` shape, has no inner `\E` to free up, has an
/// inner `\E` outside the disjunction (so the split would not reduce the
/// per-disjunct `\E` count), or any kept disjunct would still carry more than
/// one inner `\E` pair.
///
/// Returned functions are named `"{base}#d{idx}"`. Each has its sibling
/// disjuncts neutralized as described in the module docs, so the union of their
/// successor sets equals `func`'s successor set exactly.
pub fn split_top_level_disjunction(func: &BytecodeFunction) -> Option<Vec<BytecodeFunction>> {
    split_top_level_disjunction_impl(func, true)
}

/// General top-level-disjunction split, WITHOUT the inner-`\E` precondition.
///
/// Same union-exact neutralization as [`split_top_level_disjunction`], but it
/// also splits pure-boolean disjunctions `guard /\ (D1 \/ ... \/ Dn)` that
/// carry no inner `\E`. This is what a single-successor next-state generator
/// needs: a disjunctive action is a relational fork (one parent can satisfy
/// several `Di` and yield several successors), which one generator cannot
/// represent; splitting into per-disjunct sub-actions restores exactness
/// because the BFS engine unions the successors of distinct action functions.
///
/// Returns `None` on any spec whose bytecode does not match the canonical
/// `guard /\ (OR-chain)` shape, so callers fail closed to the interpreter.
pub fn split_top_level_disjunction_general(
    func: &BytecodeFunction,
) -> Option<Vec<BytecodeFunction>> {
    if let Some(subs) = split_top_level_disjunction_impl(func, false) {
        return Some(subs);
    }
    // The compiler can also emit a top-level `\/` as a RIGHT-NESTED cascade
    // (PlusCal-generated multi-arm actions): each disjunct joins into its OWN
    // or-register and short-circuits into a pure Move-cascade tail
    // `r_{n-1}<-r_n ; ... ; OR<-r_1` that folds inner results outward. Flatten
    // that cascade in place (value-preserving, fail-closed) into the canonical
    // single-register chain, then split.
    let flattened = flatten_nested_or_cascade(func)?;
    split_top_level_disjunction_impl(&flattened, false)
}

/// Rewrite a right-nested OR cascade into the canonical flat OR chain.
///
/// Recognized shape (all conditions checked; `None` = fail closed, caller
/// falls back to the interpreter):
///   - a maximal run of `Move { rd: r_k, rs: r_{k+1} }` at PCs `[c, c+m)`
///     forming a linear register chain `OR <- r_1 <- r_2 <- ... <- r_m`
///     (the cascade tail), immediately followed by the function's result
///     consumption of `OR`;
///   - for every inner cascade register `r_k` there is exactly one
///     `Move { rd: r_k, rs: x_k } ; JumpTrue { rs: r_k, offset }` join pair
///     whose jump target is exactly `r_k`'s consumption PC in the cascade,
///     and `r_k` has NO other reads or writes in the function;
///   - the innermost register `r_m` is written exactly once (its disjunct's
///     join `Move { rd: r_m, rs: x_m }` at `c - 1`... i.e. the fall-through
///     into the cascade) and read only by the cascade.
///
/// Rewrite (same instruction count — all other jump offsets stay valid):
///   - each inner join `Move { rd: r_k, rs: x_k }` becomes
///     `Move { rd: OR, rs: x_k }`, and its `JumpTrue { rs: r_k }` becomes
///     `JumpTrue { rs: OR, -> end_pc }` where `end_pc` is the PC right after
///     the final join;
///   - the innermost (fall-through) join `Move { rd: r_m, rs: x_m }` becomes
///     the final join `Move { rd: OR, rs: x_m }`, relocated to `end_pc - 1`;
///   - all remaining cascade Moves become `Nop`.
///
/// Value preservation: the cascade is pure register Moves of the boolean OR
/// result; every path (each short-circuit target and the fall-through) ends
/// with the same boolean in `OR` at `end_pc`, exactly as before. Registers
/// `r_1..r_m` become dead, which the single-reference conditions above prove
/// is unobservable.
fn flatten_nested_or_cascade(func: &BytecodeFunction) -> Option<BytecodeFunction> {
    let instrs = &func.instructions;
    let len = instrs.len();

    // 1) Find the cascade tail: the LAST maximal run of chained Moves
    //    `Move { rd, rs }` where each Move's rd equals the NEXT Move's rs
    //    (folding inward-to-outward), length >= 2.
    let mut cascade_end = None; // exclusive
    for pc in (1..len).rev() {
        if let (Opcode::Move { rd: a_rd, .. }, Some(Opcode::Move { rs: b_rs, .. })) =
            (&instrs[pc - 1], instrs.get(pc))
        {
            if *a_rd == *b_rs {
                cascade_end = Some(pc + 1);
                break;
            }
        }
    }
    let cascade_end = cascade_end?;
    let mut cascade_start = cascade_end - 2;
    while cascade_start > 0 {
        if let (Opcode::Move { rd: a_rd, .. }, Opcode::Move { rs: b_rs, .. }) =
            (&instrs[cascade_start - 1], &instrs[cascade_start])
        {
            if *a_rd == *b_rs {
                cascade_start -= 1;
                continue;
            }
        }
        break;
    }
    // Cascade run is [cascade_start, cascade_end); need at least 2 moves.
    if cascade_end - cascade_start < 2 {
        return None;
    }
    // Extract the register chain: cascade[i] = Move { rd: chain[i+1], rs: chain[i] }
    // reading inner (chain[0]) toward outer (chain.last()).
    let mut inner_regs = Vec::new();
    let mut outer = None;
    for pc in cascade_start..cascade_end {
        let Opcode::Move { rd, rs } = instrs[pc] else {
            return None;
        };
        if let Some(prev_rd) = outer {
            if rs != prev_rd {
                return None;
            }
        } else {
            inner_regs.push(rs);
        }
        inner_regs.push(rd);
        outer = Some(rd);
    }
    // inner_regs = [r_m, r_{m-1}, ..., r_1, OR]; OR is the last.
    let or_reg = inner_regs.pop()?;
    // The registers consumed inside the cascade, innermost first.
    // consumption_pc(inner_regs[i]) = cascade_start + i.
    let cascade_regs = inner_regs; // r_m .. r_1

    // end_pc: PC right after the final join in the FLAT form. We place the
    // final join at cascade_end - 1 (the former outermost cascade move), so
    // end_pc = cascade_end.
    let end_pc = cascade_end;

    // 2) For each cascade register, find its unique join pair
    //    `Move { rd: r_k, rs: x } ; JumpTrue { rs: r_k, -> consumption pc }`
    //    (for the innermost r_m, the join is the fall-through write with NO
    //    JumpTrue — it must be the single write, and its only read is the
    //    cascade). Verify exclusivity of every other reference.
    let mut joins: Vec<(usize, Register, bool)> = Vec::new(); // (join_pc, src, has_jump)
    for (i, &rk) in cascade_regs.iter().enumerate() {
        let consumption_pc = cascade_start + i;
        let mut write_pc = None;
        for (pc, op) in instrs.iter().enumerate() {
            if pc >= cascade_start && pc < cascade_end {
                continue; // cascade itself
            }
            // Any write to rk outside the cascade must be the single join Move.
            if op.dest_register() == Some(rk) {
                if write_pc.is_some() {
                    return None;
                }
                if !matches!(op, Opcode::Move { .. }) {
                    return None;
                }
                write_pc = Some(pc);
            }
            // Any read of rk outside the cascade must be its own JumpTrue.
            if reads_register(op, rk) && !matches!(op, Opcode::JumpTrue { rs, .. } if *rs == rk) {
                return None;
            }
        }
        let join_pc = write_pc?;
        let Opcode::Move { rs: src, .. } = instrs[join_pc] else {
            return None;
        };
        // JumpTrue immediately after the join, targeting rk's consumption pc?
        let has_jump = match instrs.get(join_pc + 1) {
            Some(Opcode::JumpTrue { rs, offset }) if *rs == rk => {
                let tgt = (join_pc + 1).checked_add(*offset as usize)?;
                if tgt != consumption_pc {
                    return None;
                }
                true
            }
            _ => false,
        };
        // Innermost register (i == 0) is the fall-through join: no JumpTrue.
        // Every other cascade register must have its JumpTrue.
        if (i == 0) == has_jump {
            return None;
        }
        // No OTHER JumpTrue on rk elsewhere.
        for (pc, op) in instrs.iter().enumerate() {
            if pc == join_pc + 1 {
                continue;
            }
            if matches!(op, Opcode::JumpTrue { rs, .. } if *rs == rk) {
                return None;
            }
        }
        joins.push((join_pc, src, has_jump));
    }

    // The innermost join must be the LAST one before the cascade (fall-through
    // into it) — otherwise control could reach the cascade with r_m unset.
    let (inner_join_pc, inner_src, _) = joins[0];
    if inner_join_pc >= cascade_start || cascade_start - inner_join_pc != 1 {
        return None;
    }

    // 3) Rewrite in place (same length).
    let mut out = instrs.clone();
    // Inner joins (with jumps) -> flat OR joins targeting end_pc.
    for &(join_pc, src, has_jump) in joins.iter().skip(1) {
        out[join_pc] = Opcode::Move {
            rd: or_reg,
            rs: src,
        };
        if has_jump {
            let offset = i32::try_from(end_pc.checked_sub(join_pc + 1)?).ok()?;
            out[join_pc + 1] = Opcode::JumpTrue { rs: or_reg, offset };
        }
    }
    // Fall-through (innermost) join relocates to end_pc - 1; its original slot
    // and the rest of the cascade become Nops.
    out[inner_join_pc] = Opcode::Nop;
    for slot in out.iter_mut().take(cascade_end - 1).skip(cascade_start) {
        *slot = Opcode::Nop;
    }
    out[end_pc - 1] = Opcode::Move {
        rd: or_reg,
        rs: inner_src,
    };

    let mut flat = BytecodeFunction::new(func.name.clone(), func.arity);
    flat.max_register = func.max_register;
    for op in out {
        flat.emit(op);
    }
    Some(flat)
}

/// Whether `op` reads `reg`. Explicit per-shape membership test mirroring
/// `Opcode::max_source_register`'s arms; any opcode NOT enumerated here
/// reports `true` (conservative: the caller fails closed and the split is
/// declined — never unsound).
fn reads_register(op: &Opcode, reg: Register) -> bool {
    let in_range = |start: Register, count: Register| -> bool {
        count > 0 && reg >= start && u16::from(reg) < u16::from(start) + u16::from(count)
    };
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
        | Opcode::Unchanged { .. } => false,
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
        | Opcode::Ret { rs } => *rs == reg,
        Opcode::JumpTrue { rs, .. } | Opcode::JumpFalse { rs, .. } => *rs == reg,
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
        | Opcode::Concat { r1, r2, .. } => *r1 == reg || *r2 == reg,
        Opcode::Range { lo, hi, .. } => *lo == reg || *hi == reg,
        Opcode::KSubset { base, k, .. } => *base == reg || *k == reg,
        Opcode::SetIn { elem, set, .. } => *elem == reg || *set == reg,
        Opcode::Tuple2SetIn {
            first, second, set, ..
        } => *first == reg || *second == reg || *set == reg,
        Opcode::SetEnumSubseteq {
            start, count, set, ..
        } => *set == reg || in_range(*start, *count as Register),
        Opcode::RoundStepEq { child, parent, .. } => *child == reg || *parent == reg,
        Opcode::FuncApply { func, arg, .. } => *func == reg || *arg == reg,
        Opcode::FuncSet { domain, range, .. } => *domain == reg || *range == reg,
        Opcode::FuncExcept {
            func, path, val, ..
        } => *func == reg || *path == reg || *val == reg,
        Opcode::CondMove { cond, rs, .. } => *cond == reg || *rs == reg,
        Opcode::SetEnum { start, count, .. }
        | Opcode::TupleNew { start, count, .. }
        | Opcode::SeqNew { start, count, .. }
        | Opcode::Times { start, count, .. } => in_range(*start, *count as Register),
        Opcode::RecordNew {
            values_start,
            count,
            ..
        }
        | Opcode::RecordSet {
            values_start,
            count,
            ..
        } => in_range(*values_start, *count as Register),
        Opcode::FuncDef {
            r_domain,
            r_binding,
            ..
        } => *r_domain == reg || *r_binding == reg,
        Opcode::Call {
            args_start, argc, ..
        }
        | Opcode::CallExternal {
            args_start, argc, ..
        }
        | Opcode::CallBuiltin {
            args_start, argc, ..
        } => in_range(*args_start, *argc as Register),
        Opcode::ValueApply {
            func,
            args_start,
            argc,
            ..
        } => *func == reg || in_range(*args_start, *argc as Register),
        Opcode::MakeClosure {
            captures_start,
            capture_count,
            ..
        } => in_range(*captures_start, *capture_count as Register),
        // Loop Begin opcodes READ only the domain; r_binding is an OUTPUT
        // (the loop writes each element into it — see binding_register()).
        Opcode::ForallBegin { r_domain, .. }
        | Opcode::ExistsBegin { r_domain, .. }
        | Opcode::ChooseBegin { r_domain, .. }
        | Opcode::SetFilterBegin { r_domain, .. }
        | Opcode::SetBuilderBegin { r_domain, .. }
        | Opcode::FuncDefBegin { r_domain, .. } => *r_domain == reg,
        // Loop Next opcodes READ only the body result; r_binding is again an
        // OUTPUT (advanced to the next element).
        Opcode::ForallNext { r_body, .. }
        | Opcode::ExistsNext { r_body, .. }
        | Opcode::ChooseNext { r_body, .. }
        | Opcode::LoopNext { r_body, .. } => *r_body == reg,
        // Anything not enumerated: assume it reads (fail closed).
        _ => true,
    }
}

fn split_top_level_disjunction_impl(
    func: &BytecodeFunction,
    require_inner_exists: bool,
) -> Option<Vec<BytecodeFunction>> {
    // The `\E`-reduction caller only wants splits that free up an inner `\E`.
    // The general (next-state) caller wants every top-level disjunction split.
    if require_inner_exists && exists_begin_count(func) == 0 {
        return None;
    }

    let chain = match detect_top_level_disjunction(func) {
        Some(c) => c,
        None => {
            split_trace!("{}: detect_top_level_disjunction -> None", func.name);
            return None;
        }
    };
    split_trace!(
        "{}: chain or_reg={} joins={:?} end_pc={}",
        func.name,
        chain.or_reg,
        chain.join_pcs,
        chain.end_pc
    );
    let chain = match resolve_body_start_d0(func, chain) {
        Some(c) => c,
        None => {
            split_trace!("{}: resolve_body_start_d0 -> None", func.name);
            return None;
        }
    };
    split_trace!("{}: body_start_d0={}", func.name, chain.body_start_d0);

    let n = chain.join_pcs.len();

    // Pre-compute every disjunct's span and the StoreVar PCs inside it.
    let spans: Vec<(usize, usize)> = (0..n).map(|i| disjunct_span(&chain, i)).collect();

    // Validate the spans are well-ordered and non-overlapping; otherwise the
    // structural assumptions are violated -> fail closed.
    for i in 1..n {
        if spans[i].0 <= spans[i - 1].1 {
            return None;
        }
    }
    if spans[0].0 > spans[0].1 {
        return None;
    }

    // Every `ExistsBegin` must lie inside exactly one disjunct span; an
    // `ExistsBegin` outside the chain (e.g. in a trailing conjunct) means the
    // split would not actually reduce the per-disjunct inner-`\E` count, so the
    // split provides no benefit and we fail closed.
    for (pc, op) in func.instructions.iter().enumerate() {
        if matches!(op, Opcode::ExistsBegin { .. }) {
            let inside = spans.iter().any(|&(s, e)| pc >= s && pc <= e);
            if !inside {
                return None;
            }
        }
    }

    // After splitting, each per-disjunct function must have AT MOST one
    // `ExistsBegin` left live (the others are siblings, whose bodies stay
    // present in the bytecode but become dead via the neutralized join). The
    // downstream single-pair check operates over *all* ExistsBegin opcodes
    // present, so we must structurally remove sibling disjuncts' ExistsBegin
    // /ExistsNext by neutralizing them too. We replace sibling EXISTS opcodes
    // with Nop so only the kept disjunct's pair remains.
    let mut subs = Vec::with_capacity(n);
    for keep in 0..n {
        let mut instrs = func.instructions.clone();
        for (j, &(s, join_pc)) in spans.iter().enumerate() {
            if j == keep {
                continue;
            }
            // Neutralize sibling disjunct j: join Move -> LoadBool false; all
            // StoreVar and EXISTS loop opcodes in its span -> Nop.
            instrs[join_pc] = Opcode::LoadBool {
                rd: chain.or_reg,
                value: false,
            };
            // The sibling's short-circuit `JumpTrue OR -> end` (the opcode right
            // after its join, for every disjunct except the last) is now
            // unreachable-on-true (OR is forced false). Replace it with `Nop` so
            // the residual `JumpTrue` cannot STRUCTURALLY trip the inner-EXISTS
            // sibling-successor-drop guard in the kept sub-action (that guard is
            // purely structural and would otherwise still see the dead jump
            // skipping the kept disjunct's primed-variable stores).
            if j + 1 != n {
                let jt_pc = join_pc + 1;
                if matches!(
                    instrs.get(jt_pc),
                    Some(Opcode::JumpTrue { rs, .. }) if *rs == chain.or_reg
                ) {
                    instrs[jt_pc] = Opcode::Nop;
                }
            }
            // Nop the sibling disjunct's ENTIRE body span. The disjunct is dead
            // in this sub-action (its join is forced `false`), so every opcode
            // in `[s, join_pc)` is unreachable-on-success. Blanking the whole
            // span — not just `StoreVar`/EXISTS opcodes — is required for
            // soundness AND for backend lowering: a partially-blanked span would
            // leave reads (e.g. `FuncExcept val = <sibling \E binding>`) of a
            // now-uninitialized binding register, which the trust-ir backend
            // rejects. A disjunct's internal registers are local to the disjunct
            // (only its join register and its `StoreVar`s escape, both handled),
            // so Nop-ing the entire span produces no observable effect: the
            // sibling contributes `false` to the OR and writes nothing. Any
            // intra-span jumps become harmless fall-throughs over Nops to the
            // join.
            for op in &mut instrs[s..join_pc] {
                *op = Opcode::Nop;
            }
        }

        let mut sub = BytecodeFunction::new(format!("{}#d{keep}", func.name), func.arity);
        sub.max_register = func.max_register;
        for op in instrs {
            sub.emit(op);
        }

        // Defense-in-depth: the kept sub-action must now have at most one
        // ExistsBegin. If neutralization left more (unexpected structure), fail
        // closed for the whole split.
        if exists_begin_count(&sub) > 1 {
            return None;
        }
        subs.push(sub);
    }

    Some(subs)
}

/// Resolve disjunct 0's body start: the first instruction after the guard /
/// enclosing conjunction prefix and before disjunct 0's join.
///
/// The compiler emits `guard /\ (OR...)` as `<guard -> rg> ; Move rdAnd<-rg ;
/// JumpFalse rdAnd -> end_and ; <OR...>`, where `end_and` is at/after the OR
/// chain's `end_pc`. Disjunct 0's body therefore starts right after that guard
/// `JumpFalse`. We require that boundary to exist (fail closed otherwise) so we
/// never accidentally fold guard opcodes into disjunct 0's span — a guard
/// `ExistsBegin` neutralized in a sibling sub-action would corrupt the guard.
fn resolve_body_start_d0(
    func: &BytecodeFunction,
    mut chain: DisjunctionChain,
) -> Option<DisjunctionChain> {
    let first_join = chain.join_pcs[0];
    // Search backward from the first join for the nearest `JumpFalse` whose
    // forward target is at/after the OR chain's end — the guard short-circuit
    // that jumps past the entire disjunction body.
    let mut start = None;
    for pc in (0..first_join).rev() {
        if let Opcode::JumpFalse { offset, .. } = func.instructions[pc] {
            if offset > 0 {
                let tgt = pc + offset as usize;
                if tgt >= chain.end_pc.saturating_sub(1) {
                    start = Some(pc + 1);
                    break;
                }
            }
        }
    }
    // GUARD-LESS chains: an action of the shape `(D1 \/ ... \/ Dn) /\ trailing`
    // has no guard `JumpFalse` before the first join (PaxosCommit's `Decide`,
    // whose only pre-chain code is a LET-closure `LoadConst` shared by both
    // arms). Fall back to a register-flow boundary: everything before
    // `body_start_d0` is a SHARED PREFIX that stays live in every sub-action,
    // so it must be provably pure, and everything in disjunct 0's span must be
    // register-closed (no value computed there is read outside the span, where
    // sub-actions that neutralize the span would see a blanked register).
    let start = match start {
        Some(start) => start,
        None => resolve_guardless_body_start_d0(func, &chain)?,
    };
    chain.body_start_d0 = start;
    Some(chain)
}

/// Compute disjunct 0's body start for a chain with no guard prefix.
///
/// Soundness contract (see module docs): in sub-action `keep != 0`, the span
/// `[start, join_0)` is blanked to `Nop` and the join becomes `LoadBool
/// false`. That is observable-effect-free iff
///
/// 1. no instruction in `[start, join_0)` writes a register that is READ
///    outside `[0, join_0]` (a later disjunct or the trailing conjuncts would
///    see an uninitialized register) — instructions that DO feed later code
///    are hoisted into the shared prefix by moving `start` past them;
/// 2. the shared prefix `[0, start)` is pure: value-producing opcodes only
///    (no successor writes, no primed reads, no calls — a callee could hide
///    effects — and no control flow, so the prefix always runs in full); and
/// 3. control flow cannot ENTER `[start, join_0)` from outside (only the
///    fall-through edge from the prefix reaches the span), so blanking it
///    cannot be bypassed or re-entered. Jumps INSIDE the span are fine — in
///    neutralized sub-actions the whole span is `Nop`s.
///
/// Fail-closed: any violation returns `None` and the split is abandoned.
fn resolve_guardless_body_start_d0(
    func: &BytecodeFunction,
    chain: &DisjunctionChain,
) -> Option<usize> {
    let instrs = &func.instructions;
    let first_join = chain.join_pcs[0];

    // Registers read outside [0, first_join] (strictly after the join): reads
    // by later disjuncts, the chain end, and trailing conjuncts.
    let mut read_outside = [false; 256];
    for op in &instrs[first_join + 1..] {
        record_source_registers(op, &mut read_outside);
    }

    // Register-recycling exoneration: under the recycled compiler every
    // disjunct REUSES the same low temp registers (each arm rolls the
    // allocator back), so a later arm "reading" r is almost always reading its
    // OWN fresh write, not disjunct 0's value. Exonerate r when, in EVERY
    // later-arm region and in the tail, the first occurrence of r in pc order
    // is a WRITE (never a read), and no forward jump from before that write
    // targets a point after it inside the same region (which could skip the
    // write and reach a read). This is sound: every execution entering a
    // region flows from its entry; reads can only sit after the region's own
    // write, so blanking disjunct 0 can never expose a stale/blank r.
    // Fail-open here is impossible — exoneration only ever REMOVES registers
    // from read_outside when the write-first proof holds; anything unproven
    // stays escaping (fail closed).
    {
        // Region boundaries: each later disjunct k spans
        // (join_{k-1}+1, join_k]; the tail spans (end_pc-1, len).
        let mut regions: Vec<(usize, usize)> = Vec::new();
        for w in chain.join_pcs.windows(2) {
            regions.push((w[0] + 2, w[1] + 1)); // skip join + its JumpTrue
        }
        regions.push((chain.end_pc, instrs.len()));

        'reg: for r in 0..=func.max_register {
            if !read_outside[r as usize] {
                continue;
            }
            if r == chain.or_reg {
                // The chain accumulator is written by every join (each
                // sub-action keeps or neutralizes them); tail reads of it are
                // the chain output, not a disjunct-0 escape.
                read_outside[r as usize] = false;
                continue;
            }
            for &(rs, re) in &regions {
                if rs >= re {
                    continue;
                }
                let mut first_write: Option<usize> = None;
                for (off, op) in instrs[rs..re].iter().enumerate() {
                    let pc = rs + off;
                    if reads_register(op, r) {
                        // Read before any write in this region: genuinely
                        // escaping — keep it flagged.
                        if first_write.is_none() {
                            split_trace!(
                                "{}: r{} read-before-write at pc {} ({:?}) in region [{},{})",
                                func.name,
                                r,
                                pc,
                                op,
                                rs,
                                re
                            );
                            continue 'reg;
                        }
                        break;
                    }
                    if op.dest_register() == Some(r) || op.binding_register() == Some(r) {
                        first_write = Some(pc);
                        break;
                    }
                }
                let Some(w) = first_write else {
                    // r unused in this region — nothing to prove here.
                    continue;
                };
                // A forward jump from [rs, w) landing in (w, re) could skip
                // the write — reject ONLY if a read of r exists at/after the
                // landing point (otherwise the skipped write is irrelevant:
                // nothing beyond the target reads r).
                let last_read_pc = instrs[w + 1..re]
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, op)| reads_register(op, r))
                    .map(|(off, _)| w + 1 + off);
                if let Some(last_read) = last_read_pc {
                    for (off, op) in instrs[rs..w].iter().enumerate() {
                        let pc = rs + off;
                        let jump_off = match *op {
                            Opcode::Jump { offset }
                            | Opcode::JumpTrue { offset, .. }
                            | Opcode::JumpFalse { offset, .. } => offset,
                            _ => continue,
                        };
                        if jump_off <= 0 {
                            continue;
                        }
                        let tgt = pc + jump_off as usize;
                        if tgt > w && tgt <= last_read {
                            split_trace!(
                                "{}: r{} fwd-jump {}->{} skips write at {} with read at {} in [{},{})",
                                func.name, r, pc, tgt, w, last_read, rs, re
                            );
                            continue 'reg;
                        }
                    }
                }
            }
            // Every region either rewrites r first (jump-safe) or ignores it.
            read_outside[r as usize] = false;
        }
        // The chain accumulator itself is by definition threaded through the
        // chain; keep it non-escaping for the prefix computation (its writes
        // ARE the joins).
    }

    // The shared prefix must end after the LAST instruction whose destination
    // is read outside the chain's first span.
    let mut start = 0usize;
    for (pc, op) in instrs[..first_join].iter().enumerate() {
        let writes_escaping_register = op
            .dest_register()
            .is_some_and(|rd| read_outside[rd as usize])
            || op
                .binding_register()
                .is_some_and(|rb| read_outside[rb as usize]);
        if writes_escaping_register {
            start = pc + 1;
        }
    }
    split_trace!(
        "{}: guardless start={} first_join={} escaping={:?}",
        func.name,
        start,
        first_join,
        (0..=func.max_register)
            .filter(|&r| read_outside[r as usize])
            .collect::<Vec<_>>()
    );

    // Prefix purity: only pure value producers may execute unconditionally in
    // every sub-action. Control flow, calls, and successor/prime opcodes fail
    // closed. (`Nop` is fine.)
    for op in &instrs[..start] {
        match op {
            Opcode::StoreVar { .. }
            | Opcode::LoadPrime { .. }
            | Opcode::RoundStepEq { .. }
            | Opcode::SetPrimeMode { .. }
            | Opcode::Unchanged { .. }
            | Opcode::Call { .. }
            | Opcode::ValueApply { .. }
            | Opcode::CallExternal { .. }
            | Opcode::CallBuiltin { .. }
            | Opcode::MakeClosure { .. }
            | Opcode::Ret { .. }
            | Opcode::Halt
            | Opcode::Jump { .. }
            | Opcode::JumpTrue { .. }
            | Opcode::JumpFalse { .. }
            | Opcode::ForallBegin { .. }
            | Opcode::ForallNext { .. }
            | Opcode::ExistsBegin { .. }
            | Opcode::ExistsNext { .. }
            | Opcode::ChooseBegin { .. }
            | Opcode::ChooseNext { .. }
            | Opcode::SetFilterBegin { .. }
            | Opcode::SetBuilderBegin { .. }
            | Opcode::FuncDefBegin { .. }
            | Opcode::LoopNext { .. } => {
                split_trace!("{}: prefix purity failed at op {:?}", func.name, op);
                return None;
            }
            _ => {}
        }
    }

    if start > first_join {
        split_trace!("{}: start {} > first_join {}", func.name, start, first_join);
        return None;
    }

    // No control transfer anywhere in the function may target the span
    // interior `(start, first_join)`: the span must be enterable only by
    // falling through the prefix. (Targets at `start` itself are also
    // rejected — nothing in canonical compilation jumps to a disjunct 0 body
    // start that has no guard, and allowing it would let a neutralized span
    // be entered mid-way in the kept sub-action's siblings.)
    for (pc, op) in instrs.iter().enumerate() {
        let offset = match *op {
            Opcode::Jump { offset }
            | Opcode::JumpTrue { offset, .. }
            | Opcode::JumpFalse { offset, .. } => offset,
            Opcode::ForallBegin { loop_end, .. }
            | Opcode::ExistsBegin { loop_end, .. }
            | Opcode::ChooseBegin { loop_end, .. }
            | Opcode::SetFilterBegin { loop_end, .. }
            | Opcode::SetBuilderBegin { loop_end, .. }
            | Opcode::FuncDefBegin { loop_end, .. } => loop_end,
            Opcode::ForallNext { loop_begin, .. }
            | Opcode::ExistsNext { loop_begin, .. }
            | Opcode::ChooseNext { loop_begin, .. }
            | Opcode::LoopNext { loop_begin, .. } => loop_begin,
            _ => continue,
        };
        let target = pc as i64 + i64::from(offset);
        // Jumps from INSIDE the span may target the span (loops within
        // disjunct 0 are blanked together with it).
        if pc >= start && pc < first_join {
            continue;
        }
        if target >= start as i64 && target < first_join as i64 {
            split_trace!(
                "{}: control transfer from pc {} into span interior (target {})",
                func.name,
                pc,
                target
            );
            return None;
        }
    }

    Some(start)
}

/// Set `seen[r] = true` for every source register `op` reads.
fn record_source_registers(op: &Opcode, seen: &mut [bool; 256]) {
    let mut mark = |r: Register| seen[r as usize] = true;
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
        | Opcode::Ret { rs }
        | Opcode::JumpTrue { rs, .. }
        | Opcode::JumpFalse { rs, .. } => mark(rs),
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
            mark(r1);
            mark(r2);
        }
        Opcode::Range { lo, hi, .. } => {
            mark(lo);
            mark(hi);
        }
        Opcode::KSubset { base, k, .. } => {
            mark(base);
            mark(k);
        }
        Opcode::SetIn { elem, set, .. } => {
            mark(elem);
            mark(set);
        }
        Opcode::Tuple2SetIn {
            first, second, set, ..
        } => {
            mark(first);
            mark(second);
            mark(set);
        }
        Opcode::SetEnumSubseteq {
            start, count, set, ..
        } => {
            for i in 0..count {
                mark(start.saturating_add(i));
            }
            mark(set);
        }
        Opcode::Tuple2SelfEq { value, .. } | Opcode::Tuple2SelfSubseteq { value, .. } => {
            mark(value);
        }
        Opcode::RoundStepEq { child, parent, .. } => {
            mark(child);
            mark(parent);
        }
        Opcode::FuncApply { func, arg, .. } => {
            mark(func);
            mark(arg);
        }
        Opcode::FuncSet { domain, range, .. } => {
            mark(domain);
            mark(range);
        }
        Opcode::FuncExcept {
            func, path, val, ..
        } => {
            mark(func);
            mark(path);
            mark(val);
        }
        Opcode::EqFuncExcept {
            lhs,
            func,
            path,
            val,
            ..
        } => {
            mark(lhs);
            mark(func);
            mark(path);
            mark(val);
        }
        Opcode::EqRecordNew {
            lhs,
            values_start,
            count,
            ..
        } => {
            mark(lhs);
            for i in 0..count {
                mark(values_start.saturating_add(i));
            }
        }
        Opcode::CondMove { cond, rs, .. } => {
            mark(cond);
            mark(rs);
        }
        Opcode::SetEnum { start, count, .. }
        | Opcode::TupleNew { start, count, .. }
        | Opcode::SeqNew { start, count, .. }
        | Opcode::Times { start, count, .. } => {
            for i in 0..count {
                mark(start.saturating_add(i));
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
                mark(values_start.saturating_add(i));
            }
        }
        Opcode::FuncDef {
            r_domain,
            r_binding,
            ..
        } => {
            mark(r_domain);
            mark(r_binding);
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
                mark(args_start.saturating_add(i));
            }
        }
        Opcode::ValueApply {
            func,
            args_start,
            argc,
            ..
        } => {
            mark(func);
            for i in 0..argc {
                mark(args_start.saturating_add(i));
            }
        }
        Opcode::MakeClosure {
            captures_start,
            capture_count,
            ..
        } => {
            for i in 0..capture_count {
                mark(captures_start.saturating_add(i));
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
            mark(r_binding);
            mark(r_domain);
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
            mark(r_binding);
            mark(r_body);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_enum_subseteq_reads_terminal_register() {
        let opcode = Opcode::SetEnumSubseteq {
            rd: 0,
            start: 254,
            count: 2,
            set: 1,
        };
        assert!(reads_register(&opcode, 255));
    }

    /// Build a faithful e3/exit-shaped action:
    ///
    /// ```text
    /// A == guard
    ///      /\ ( (\E k \in {0,1} : v0' = k)        \* disjunct 0
    ///           \/ (\E i \in {0,1} : v1' = i) )   \* disjunct 1
    ///      /\ UNCHANGED <<v2>>                     \* trailing conjunct
    /// ```
    ///
    /// Mirrors the canonical compiled bytecode:
    /// - guard -> r5 ; JumpFalse r5 -> end_and
    /// - D0 exists (StoreVar v0) ; Move OR<-rD0 ; JumpTrue OR -> end
    /// - D1 exists (StoreVar v1) ; Move OR<-rD1               (end is here)
    /// - Move rAnd<-OR ; (UNCHANGED) ; Ret
    fn make_two_disjunct_exists_action() -> BytecodeFunction {
        let mut f = BytecodeFunction::new("A".to_string(), 0);

        // ---- guard: r5 = (something true), JumpFalse r5 -> end_and ----
        f.emit(Opcode::LoadBool { rd: 5, value: true }); // PC 0: guard truth
        f.emit(Opcode::JumpFalse { rs: 5, offset: 0 }); // PC 1: patched below
        let guard_jf = 1;

        // ---- disjunct 0: \E k \in {0,1} : v0' = k ----
        f.emit(Opcode::LoadImm { rd: 3, value: 0 }); // PC 2
        f.emit(Opcode::LoadImm { rd: 4, value: 1 }); // PC 3
        f.emit(Opcode::SetEnum {
            rd: 2,
            start: 3,
            count: 2,
        }); // PC 4
        f.emit(Opcode::ExistsBegin {
            rd: 9,
            r_binding: 10,
            r_domain: 2,
            loop_end: 4,
        }); // PC 5 -> next at PC 9
        f.emit(Opcode::StoreVar { var_idx: 0, rs: 10 }); // PC 6: v0' = k
        f.emit(Opcode::LoadBool {
            rd: 16,
            value: true,
        }); // PC 7: body truth
        f.emit(Opcode::ExistsNext {
            rd: 9,
            r_binding: 10,
            r_body: 16,
            loop_begin: -3,
        }); // PC 8 -> body PC 6
        f.emit(Opcode::Move { rd: 25, rs: 9 }); // PC 9: join OR<-rD0
        f.emit(Opcode::JumpTrue { rs: 25, offset: 0 }); // PC 10: patched -> end
        let d0_jt = 10;

        // ---- disjunct 1: \E i \in {0,1} : v1' = i ----
        f.emit(Opcode::LoadImm { rd: 26, value: 0 }); // PC 11
        f.emit(Opcode::LoadImm { rd: 27, value: 1 }); // PC 12
        f.emit(Opcode::SetEnum {
            rd: 28,
            start: 26,
            count: 2,
        }); // PC 13
        f.emit(Opcode::ExistsBegin {
            rd: 35,
            r_binding: 36,
            r_domain: 28,
            loop_end: 4,
        }); // PC 14 -> next PC 18
        f.emit(Opcode::StoreVar { var_idx: 1, rs: 36 }); // PC 15: v1' = i
        f.emit(Opcode::LoadBool {
            rd: 42,
            value: true,
        }); // PC 16
        f.emit(Opcode::ExistsNext {
            rd: 35,
            r_binding: 36,
            r_body: 42,
            loop_begin: -3,
        }); // PC 17 -> body PC 15
        f.emit(Opcode::Move { rd: 25, rs: 35 }); // PC 18: final join OR<-rD1 (end-1)

        // end of OR chain == PC 19
        f.emit(Opcode::Move { rd: 50, rs: 25 }); // PC 19: rAnd <- OR
                                                 // trailing conjunct UNCHANGED <<v2>>
        f.emit(Opcode::Unchanged {
            rd: 51,
            start: 0,
            count: 1,
        }); // PC 20
        f.emit(Opcode::Move { rd: 0, rs: 50 }); // PC 21: result
        f.emit(Opcode::Ret { rs: 0 }); // PC 22

        // Patch jumps: guard JumpFalse -> end_and (PC 19), d0 JumpTrue -> end (PC 19).
        f.patch_jump(guard_jf, 19);
        f.patch_jump(d0_jt, 19);
        f
    }

    fn store_var_indices(f: &BytecodeFunction) -> Vec<u16> {
        f.instructions
            .iter()
            .filter_map(|op| match op {
                Opcode::StoreVar { var_idx, .. } => Some(*var_idx),
                _ => None,
            })
            .collect()
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_split_two_disjunct_exists_action() {
        let f = make_two_disjunct_exists_action();
        assert_eq!(exists_begin_count(&f), 2, "fixture has two inner EXISTS");

        let subs = split_top_level_disjunction(&f).expect("e3-shaped action must split");
        assert_eq!(subs.len(), 2, "two disjuncts -> two sub-actions");

        // Instruction count preserved (offset-preserving rewrite).
        for sub in &subs {
            assert_eq!(sub.instructions.len(), f.instructions.len());
            assert_eq!(
                exists_begin_count(sub),
                1,
                "each sub-action keeps exactly one inner EXISTS pair"
            );
        }

        // Sub-action 0 keeps disjunct 0's StoreVar (v0) and drops v1.
        assert_eq!(
            store_var_indices(&subs[0]),
            vec![0],
            "sub-action 0 produces only disjunct 0's primed write"
        );
        // Sub-action 1 keeps disjunct 1's StoreVar (v1) and drops v0.
        assert_eq!(
            store_var_indices(&subs[1]),
            vec![1],
            "sub-action 1 produces only disjunct 1's primed write"
        );

        // The UNCHANGED trailing conjunct is preserved in BOTH sub-actions.
        for sub in &subs {
            assert!(
                sub.instructions
                    .iter()
                    .any(|op| matches!(op, Opcode::Unchanged { .. })),
                "trailing UNCHANGED preserved in every sub-action"
            );
        }

        // Sibling disjunct's join Move is replaced with LoadBool false.
        // In sub-action 0, disjunct 1's join (PC 18) becomes LoadBool OR=false.
        assert!(
            matches!(
                subs[0].instructions[18],
                Opcode::LoadBool {
                    rd: 25,
                    value: false
                }
            ),
            "sub-action 0 neutralizes disjunct 1's join to false"
        );
        // In sub-action 1, disjunct 0's join (PC 9) becomes LoadBool OR=false.
        assert!(
            matches!(
                subs[1].instructions[9],
                Opcode::LoadBool {
                    rd: 25,
                    value: false
                }
            ),
            "sub-action 1 neutralizes disjunct 0's join to false"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_no_split_for_single_exists() {
        // An action with a single inner EXISTS is not worth splitting.
        let mut f = BytecodeFunction::new("Single".to_string(), 0);
        f.emit(Opcode::LoadImm { rd: 3, value: 0 });
        f.emit(Opcode::LoadImm { rd: 4, value: 1 });
        f.emit(Opcode::SetEnum {
            rd: 2,
            start: 3,
            count: 2,
        });
        f.emit(Opcode::ExistsBegin {
            rd: 9,
            r_binding: 10,
            r_domain: 2,
            loop_end: 3,
        });
        f.emit(Opcode::StoreVar { var_idx: 0, rs: 10 });
        f.emit(Opcode::ExistsNext {
            rd: 9,
            r_binding: 10,
            r_body: 9,
            loop_begin: -2,
        });
        f.emit(Opcode::Ret { rs: 9 });
        assert!(
            split_top_level_disjunction(&f).is_none(),
            "single-EXISTS action must not split"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_no_split_when_exists_outside_chain() {
        // Two EXISTS but the second is in a trailing conjunct, not a disjunct —
        // splitting the disjunction would not reduce per-disjunct EXISTS count.
        // Build: guard /\ (\E .. \/ scalar) /\ (\E ..)
        let mut f = BytecodeFunction::new("Mixed".to_string(), 0);
        f.emit(Opcode::LoadBool { rd: 5, value: true }); // PC 0 guard
        f.emit(Opcode::JumpFalse { rs: 5, offset: 0 }); // PC 1
        let gjf = 1;
        // disjunct 0: \E
        f.emit(Opcode::LoadImm { rd: 3, value: 0 }); // PC 2
        f.emit(Opcode::SetEnum {
            rd: 2,
            start: 3,
            count: 1,
        }); // PC 3
        f.emit(Opcode::ExistsBegin {
            rd: 9,
            r_binding: 10,
            r_domain: 2,
            loop_end: 3,
        }); // PC 4
        f.emit(Opcode::StoreVar { var_idx: 0, rs: 10 }); // PC 5
        f.emit(Opcode::ExistsNext {
            rd: 9,
            r_binding: 10,
            r_body: 9,
            loop_begin: -2,
        }); // PC 6
        f.emit(Opcode::Move { rd: 25, rs: 9 }); // PC 7 join
        f.emit(Opcode::JumpTrue { rs: 25, offset: 0 }); // PC 8 -> end
        let d0jt = 8;
        // disjunct 1: scalar store
        f.emit(Opcode::LoadImm { rd: 11, value: 0 }); // PC 9
        f.emit(Opcode::StoreVar { var_idx: 0, rs: 11 }); // PC 10
        f.emit(Opcode::LoadBool {
            rd: 12,
            value: true,
        }); // PC 11
        f.emit(Opcode::Move { rd: 25, rs: 12 }); // PC 12 final join (end-1)
                                                 // end == PC 13
                                                 // trailing conjunct with a SECOND exists (outside the disjunction)
        f.emit(Opcode::LoadImm { rd: 13, value: 0 }); // PC 13
        f.emit(Opcode::SetEnum {
            rd: 14,
            start: 13,
            count: 1,
        }); // PC 14
        f.emit(Opcode::ExistsBegin {
            rd: 20,
            r_binding: 21,
            r_domain: 14,
            loop_end: 3,
        }); // PC 15
        f.emit(Opcode::StoreVar { var_idx: 1, rs: 21 }); // PC 16
        f.emit(Opcode::ExistsNext {
            rd: 20,
            r_binding: 21,
            r_body: 20,
            loop_begin: -2,
        }); // PC 17
        f.emit(Opcode::Ret { rs: 25 }); // PC 18
        f.patch_jump(gjf, 13);
        f.patch_jump(d0jt, 13);

        assert!(
            split_top_level_disjunction(&f).is_none(),
            "must fail closed: an EXISTS outside the disjunction would survive the split"
        );
    }
}
