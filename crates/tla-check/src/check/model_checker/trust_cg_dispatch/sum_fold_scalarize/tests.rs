// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Unit tests for the recursive Sum-fold plan-time scalarizer.

use super::*;
use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, ConstantPool, Opcode};
use tla_value::Value;

/// Which set-builder body to synthesize into `score`.
#[derive(Clone, Copy)]
enum BodyKind {
    /// `<<p[1] + v[1], p[2] + v[2]>>` — a genuine componentwise translation.
    Translation,
    /// `<<p[1] + v[1], p[2] + v[1]>>` — reads `v[1]` twice, so distinct `v`
    /// with equal first component collapse (non-injective).
    NonInjective,
}

/// Synthesize the recursive `Sum` fold function (arity 2), recursing on itself
/// via `Call` (op_idx == `self_idx`). When `well_formed` is false the base case
/// is dropped so it must NOT be recognized as a fold.
fn build_sum(self_idx: u16, empty_idx: u16, well_formed: bool) -> BytecodeFunction {
    let mut f = BytecodeFunction::new("Sum".to_string(), 2);
    if !well_formed {
        // Not a fold: just `Ret 0`.
        f.emit(Opcode::LoadImm { rd: 4, value: 0 });
        f.emit(Opcode::Move { rd: 0, rs: 4 });
        f.emit(Opcode::Ret { rs: 0 });
        return f;
    }
    // Sum(f=r0, S=r1) == IF S = {} THEN 0 ELSE f[CHOOSE x] + Sum(f, S\{x})
    f.emit(Opcode::LoadConst {
        rd: 2,
        idx: empty_idx,
    }); // pc 0
    f.emit(Opcode::Eq {
        rd: 3,
        r1: 1,
        r2: 2,
    }); // pc 1
    f.emit(Opcode::JumpFalse { rs: 3, offset: 4 }); // pc 2 -> pc 6
    f.emit(Opcode::LoadImm { rd: 5, value: 0 }); // pc 3
    f.emit(Opcode::Move { rd: 4, rs: 5 }); // pc 4
    f.emit(Opcode::Jump { offset: 13 }); // pc 5 -> pc 18
    f.emit(Opcode::ChooseBegin {
        rd: 6,
        r_binding: 7,
        r_domain: 1,
        loop_end: 3,
    }); // pc 6
    f.emit(Opcode::LoadBool { rd: 8, value: true }); // pc 7
    f.emit(Opcode::ChooseNext {
        rd: 6,
        r_binding: 7,
        r_body: 8,
        loop_begin: -1,
    }); // pc 8
    f.emit(Opcode::FuncApply {
        rd: 9,
        func: 0,
        arg: 6,
    }); // pc 9: f[x]
    f.emit(Opcode::Move { rd: 10, rs: 0 }); // pc 10: f
    f.emit(Opcode::Move { rd: 12, rs: 6 }); // pc 11: x
    f.emit(Opcode::SetEnum {
        rd: 13,
        start: 12,
        count: 1,
    }); // pc 12: {x}
    f.emit(Opcode::SetDiff {
        rd: 14,
        r1: 1,
        r2: 13,
    }); // pc 13: S \ {x}
    f.emit(Opcode::Move { rd: 11, rs: 14 }); // pc 14
    f.emit(Opcode::Call {
        rd: 15,
        op_idx: self_idx,
        args_start: 10,
        argc: 2,
    }); // pc 15: Sum(f, S\{x})
    f.emit(Opcode::AddInt {
        rd: 16,
        r1: 9,
        r2: 15,
    }); // pc 16: f[x] + rec
    f.emit(Opcode::Move { rd: 4, rs: 16 }); // pc 17
    f.emit(Opcode::Move { rd: 0, rs: 4 }); // pc 18
    f.emit(Opcode::Ret { rs: 0 }); // pc 19
    f
}

/// Synthesize a 0-ary `sc == [k \in 0..5 |-> k[1] + k[2]]` operator. Its body
/// (`k[1] + k[2]`) is inlined per key by the rewrite. The binding lives in a
/// high register (r14, matching GameOfLife's scale) so `score`'s own low
/// working registers fit below it. When `well_formed` is false it is NOT a
/// FuncDef (so the rewrite must fail closed at `sc` extraction).
fn build_sc(well_formed: bool) -> BytecodeFunction {
    let mut f = BytecodeFunction::new("sc".to_string(), 0);
    if !well_formed {
        f.emit(Opcode::LoadImm { rd: 0, value: 0 });
        f.emit(Opcode::Ret { rs: 0 });
        return f;
    }
    f.emit(Opcode::LoadImm { rd: 0, value: 0 }); // pc 0
    f.emit(Opcode::LoadImm { rd: 1, value: 5 }); // pc 1
    f.emit(Opcode::Range {
        rd: 2,
        lo: 0,
        hi: 1,
    }); // pc 2: dummy domain 0..5
    f.emit(Opcode::FuncDefBegin {
        rd: 13,
        r_binding: 14,
        r_domain: 2,
        loop_end: 7,
    }); // pc 3
    f.emit(Opcode::LoadImm { rd: 15, value: 1 }); // pc 4
    f.emit(Opcode::FuncApply {
        rd: 16,
        func: 14,
        arg: 15,
    }); // pc 5: k[1]
    f.emit(Opcode::LoadImm { rd: 17, value: 2 }); // pc 6
    f.emit(Opcode::FuncApply {
        rd: 18,
        func: 14,
        arg: 17,
    }); // pc 7: k[2]
    f.emit(Opcode::AddInt {
        rd: 19,
        r1: 16,
        r2: 18,
    }); // pc 8: k[1] + k[2]  (result)
    f.emit(Opcode::LoopNext {
        r_binding: 14,
        r_body: 19,
        loop_begin: -5,
    }); // pc 9
    f.emit(Opcode::Move { rd: 0, rs: 13 }); // pc 10
    f.emit(Opcode::Ret { rs: 0 }); // pc 11
    f
}

/// Synthesize `score(p)` — the outer helper that folds `sc` over the
/// set-builder image of `domain_idx`.
///
/// `const_domain`: when false, the set-builder domain is a `LoadVar` (not a
/// constant), so the rewrite must fail closed.
fn build_score(
    sc_idx: u16,
    sum_idx: u16,
    domain_idx: u16,
    body: BodyKind,
    const_domain: bool,
) -> BytecodeFunction {
    let mut f = BytecodeFunction::new("score".to_string(), 1);
    if const_domain {
        f.emit(Opcode::LoadConst {
            rd: 1,
            idx: domain_idx,
        }); // pc 0: D
    } else {
        f.emit(Opcode::LoadVar { rd: 1, var_idx: 0 }); // pc 0: non-constant domain
    }
    f.emit(Opcode::SetBuilderBegin {
        rd: 2,
        r_binding: 3,
        r_domain: 1,
        loop_end: 15,
    }); // pc 1
    f.emit(Opcode::LoadImm { rd: 4, value: 1 }); // pc 2
    f.emit(Opcode::FuncApply {
        rd: 5,
        func: 3,
        arg: 4,
    }); // pc 3: v[1]
    f.emit(Opcode::LoadImm { rd: 6, value: 2 }); // pc 4
    f.emit(Opcode::FuncApply {
        rd: 7,
        func: 3,
        arg: 6,
    }); // pc 5: v[2]
    f.emit(Opcode::LoadImm { rd: 10, value: 1 }); // pc 6
    f.emit(Opcode::FuncApply {
        rd: 11,
        func: 0,
        arg: 10,
    }); // pc 7: p[1]
    f.emit(Opcode::AddInt {
        rd: 12,
        r1: 11,
        r2: 5,
    }); // pc 8: p[1] + v[1]
    f.emit(Opcode::Move { rd: 8, rs: 12 }); // pc 9
    f.emit(Opcode::LoadImm { rd: 13, value: 2 }); // pc 10
    f.emit(Opcode::FuncApply {
        rd: 14,
        func: 0,
        arg: 13,
    }); // pc 11: p[2]
    let second_component_addend = match body {
        BodyKind::Translation => 7,  // v[2]
        BodyKind::NonInjective => 5, // v[1] (again)
    };
    f.emit(Opcode::AddInt {
        rd: 15,
        r1: 14,
        r2: second_component_addend,
    }); // pc 12: p[2] + v[2]  (or p[2] + v[1])
    f.emit(Opcode::Move { rd: 9, rs: 15 }); // pc 13
    f.emit(Opcode::TupleNew {
        rd: 16,
        start: 8,
        count: 2,
    }); // pc 14: <<p[1]+.., p[2]+..>>
    f.emit(Opcode::LoopNext {
        r_binding: 3,
        r_body: 16,
        loop_begin: -13,
    }); // pc 15
    f.emit(Opcode::Call {
        rd: 19,
        op_idx: sc_idx,
        args_start: 0,
        argc: 0,
    }); // pc 16: sc()
    f.emit(Opcode::Move { rd: 17, rs: 19 }); // pc 17
    f.emit(Opcode::Move { rd: 18, rs: 2 }); // pc 18: points
    f.emit(Opcode::Call {
        rd: 20,
        op_idx: sum_idx,
        args_start: 17,
        argc: 2,
    }); // pc 19: Sum(sc, points)
    f.emit(Opcode::Move { rd: 0, rs: 20 }); // pc 20
    f.emit(Opcode::Ret { rs: 0 }); // pc 21
    f
}

/// Build a full chunk. Function order: [0]=sc, [1]=Sum, [2]=score.
fn build_chunk(
    domain: Value,
    body: BodyKind,
    const_domain: bool,
    sum_well_formed: bool,
) -> BytecodeChunk {
    build_chunk_full(domain, body, const_domain, sum_well_formed, true)
}

fn build_chunk_full(
    domain: Value,
    body: BodyKind,
    const_domain: bool,
    sum_well_formed: bool,
    sc_well_formed: bool,
) -> BytecodeChunk {
    let mut pool = ConstantPool::new();
    let domain_idx = pool.add_value(domain);
    let empty_idx = pool.add_value(Value::set(std::iter::empty::<Value>()));

    let sc = build_sc(sc_well_formed);
    let sum = build_sum(1, empty_idx, sum_well_formed);
    let score = build_score(0, 1, domain_idx, body, const_domain);

    BytecodeChunk {
        constants: pool,
        functions: vec![sc, sum, score],
    }
}

fn tup(a: i64, b: i64) -> Value {
    Value::tuple([Value::SmallInt(a), Value::SmallInt(b)])
}

/// The 8 GameOfLife neighbour offsets (all pairs in {-1,0,1}^2 except <<0,0>>).
fn nbrs8() -> Value {
    let mut elems = Vec::new();
    for dx in [-1, 0, 1] {
        for dy in [-1, 0, 1] {
            if dx == 0 && dy == 0 {
                continue;
            }
            elems.push(tup(dx, dy));
        }
    }
    Value::set(elems)
}

fn count_op(func: &BytecodeFunction, pred: impl Fn(&Opcode) -> bool) -> usize {
    func.instructions.iter().filter(|op| pred(op)).count()
}

#[test]
fn positive_injective_fold_unrolls_to_k_terms() {
    let chunk = build_chunk(nbrs8(), BodyKind::Translation, true, true);
    let rewritten = rewrite_chunk_injective_sum_folds(&chunk)
        .expect("injective 8-neighbour fold must be recognized and rewritten");
    let score = &rewritten.functions[2];

    // The set-builder and every loop/call are gone (sc's body is inlined, not
    // called or materialized).
    assert_eq!(
        count_op(score, |op| matches!(op, Opcode::SetBuilderBegin { .. })),
        0,
        "rewritten score must not contain a set-builder"
    );
    assert_eq!(
        count_op(score, |op| matches!(op, Opcode::LoopNext { .. })),
        0,
        "rewritten score must not contain a loop"
    );
    assert_eq!(
        count_op(score, |op| matches!(
            op,
            Opcode::Call { .. } | Opcode::CallExternal { .. }
        )),
        0,
        "no Sum call and no sc() materialization call may remain"
    );
    // One binding tuple (sc's FuncDef key) per distinct neighbour term.
    assert_eq!(
        count_op(score, |op| matches!(op, Opcode::TupleNew { .. })),
        8,
        "exactly 8 inlined key bindings"
    );
    // sc's inlined body applies its binding twice (k[1], k[2]); with 8 copies
    // that is 16 body applications, plus the per-key p[1]/p[2] applications.
    assert!(
        count_op(score, |op| matches!(op, Opcode::FuncApply { .. })) >= 16,
        "each inlined sc body applies its binding"
    );
    // Arity is preserved.
    assert_eq!(score.arity, 1);
}

#[test]
fn positive_two_element_domain_emits_correct_keys() {
    // D = { <<0,0>>, <<1,2>> } -> keys << p[1]+0, p[2]+0 >> and
    // << p[1]+1, p[2]+2 >>. Offset vectors (0,0) and (1,2) are distinct.
    let domain = Value::set([tup(0, 0), tup(1, 2)]);
    let chunk = build_chunk(domain, BodyKind::Translation, true, true);
    let rewritten =
        rewrite_chunk_injective_sum_folds(&chunk).expect("distinct offsets must unroll");
    let score = &rewritten.functions[2];
    // One binding tuple per term (2 terms).
    assert_eq!(
        count_op(score, |op| matches!(op, Opcode::TupleNew { .. })),
        2,
    );
    // No calls / loops remain.
    assert_eq!(
        count_op(score, |op| matches!(
            op,
            Opcode::Call { .. } | Opcode::LoopNext { .. } | Opcode::SetBuilderBegin { .. }
        )),
        0,
    );
}

#[test]
fn fail_closed_sc_not_a_funcdef() {
    // `sc` is not a FuncDef (just `Ret 0`), so its body cannot be inlined.
    let chunk = build_chunk_full(nbrs8(), BodyKind::Translation, true, true, false);
    assert!(
        rewrite_chunk_injective_sum_folds(&chunk).is_none(),
        "a non-FuncDef sc must fail closed"
    );
}

#[test]
fn fail_closed_non_constant_domain() {
    // Domain is a LoadVar (state var), not a compile-time constant.
    let chunk = build_chunk(nbrs8(), BodyKind::Translation, false, true);
    assert!(
        rewrite_chunk_injective_sum_folds(&chunk).is_none(),
        "a non-constant-cardinality domain must fail closed"
    );
}

#[test]
fn fail_closed_non_injective_body() {
    // Body reads v[1] twice; distinct elements with equal first component map to
    // the same offset vector -> the target set dedups -> must fail closed.
    let domain = Value::set([tup(0, 1), tup(0, 2)]); // both -> offset (0, 0)
    let chunk = build_chunk(domain, BodyKind::NonInjective, true, true);
    assert!(
        rewrite_chunk_injective_sum_folds(&chunk).is_none(),
        "coincident offset vectors must fail closed"
    );
}

#[test]
fn fail_closed_unrecognized_fold() {
    // `Sum` is not a fold (just `Ret 0`), so the returned call is not a
    // recognized fold and the rewrite must not fire.
    let chunk = build_chunk(nbrs8(), BodyKind::Translation, true, false);
    assert!(
        rewrite_chunk_injective_sum_folds(&chunk).is_none(),
        "an unrecognized fold must fail closed"
    );
}

#[test]
fn non_injective_body_is_rejected_but_injective_variant_of_same_shape_accepted() {
    // Sanity: the ONLY difference between accept/reject here is injectivity of
    // the offset vectors, isolating that this is the deciding property.
    let injective = build_chunk(
        Value::set([tup(0, 1), tup(0, 2)]),
        BodyKind::Translation, // reads v[1] and v[2] -> offsets (0,1),(0,2) distinct
        true,
        true,
    );
    assert!(rewrite_chunk_injective_sum_folds(&injective).is_some());
    let non_injective = build_chunk(
        Value::set([tup(0, 1), tup(0, 2)]),
        BodyKind::NonInjective, // reads v[1] twice -> offsets (0,0),(0,0) coincide
        true,
        true,
    );
    assert!(rewrite_chunk_injective_sum_folds(&non_injective).is_none());
}
