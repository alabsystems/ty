// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! F1 (lever L2) validation: compile-time constant folding of set-constructor
//! subtrees, executed by the REAL VM via the injected executor.
//!
//! The fold on/off toggle uses `set_const_fold_override` (thread-local), NOT
//! env mutation — the `TY_BYTECODE_CONST_FOLD` env guard is OnceLock-cached,
//! so tests must never write it (this also sidesteps env-serialization).

use std::collections::HashMap;
use tla_value::Rp;

use tla_core::name_intern::intern_name;
use tla_core::{NameId, Span, Spanned};
use tla_tir::bytecode::{const_fold_count, set_const_fold_override, BytecodeCompiler, Opcode};
use tla_tir::{
    TirArithOp, TirBoundPattern, TirBoundVar, TirCmpOp, TirExpr, TirNameKind, TirNameRef, TirSetOp,
    TirType,
};
use tla_value::{boolean_set, FuncValue, Value};

use super::BytecodeVm;

// === TIR builders ===

fn spanned(expr: TirExpr) -> Spanned<TirExpr> {
    Spanned {
        node: expr,
        span: Span::default(),
    }
}

fn mv(name: &str) -> Value {
    Value::ModelValue(Rp::from(name))
}

fn name_expr(name: &str) -> Spanned<TirExpr> {
    spanned(TirExpr::Name(TirNameRef {
        name: name.to_string(),
        name_id: intern_name(name),
        kind: TirNameKind::Ident,
        ty: TirType::Dyn,
    }))
}

fn const_expr(value: Value) -> Spanned<TirExpr> {
    spanned(TirExpr::Const {
        value,
        ty: TirType::Dyn,
    })
}

fn int_expr(n: i64) -> Spanned<TirExpr> {
    const_expr(Value::SmallInt(n))
}

fn set_enum(elements: Vec<Spanned<TirExpr>>) -> Spanned<TirExpr> {
    spanned(TirExpr::SetEnum(elements))
}

fn union(left: Spanned<TirExpr>, right: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::SetBinOp {
        left: Box::new(left),
        op: TirSetOp::Union,
        right: Box::new(right),
    })
}

fn powerset(inner: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::Powerset(Box::new(inner)))
}

fn range(lo: i64, hi: i64) -> Spanned<TirExpr> {
    spanned(TirExpr::Range {
        lo: Box::new(int_expr(lo)),
        hi: Box::new(int_expr(hi)),
    })
}

fn func_set(domain: Spanned<TirExpr>, range: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::FuncSet {
        domain: Box::new(domain),
        range: Box::new(range),
    })
}

fn set_in(elem: Spanned<TirExpr>, set: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::In {
        elem: Box::new(elem),
        set: Box::new(set),
    })
}

fn subseteq(left: Spanned<TirExpr>, right: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::Subseteq {
        left: Box::new(left),
        right: Box::new(right),
    })
}

/// The MCDijkstra constant environment: `Proc = {p1, p2, p3}`, a cfg-assigned
/// `defaultInitValue` model value, and the stdlib `BOOLEAN`.
fn test_constants() -> HashMap<NameId, Value> {
    let mut constants = HashMap::new();
    constants.insert(
        intern_name("Proc"),
        Value::set([mv("p1"), mv("p2"), mv("p3")]),
    );
    constants.insert(intern_name("defaultInitValue"), mv("defaultInitValue"));
    constants.insert(intern_name("BOOLEAN"), boolean_set());
    constants
}

/// The measured MCTypeOK codomain: `(SUBSET Proc) \cup (Proc \cup {defaultInitValue})`.
fn mc_type_ok_codomain() -> Spanned<TirExpr> {
    union(
        powerset(name_expr("Proc")),
        union(
            name_expr("Proc"),
            set_enum(vec![name_expr("defaultInitValue")]),
        ),
    )
}

/// Guard that restores the previous fold override on drop.
struct FoldOverrideGuard(Option<bool>);

impl FoldOverrideGuard {
    fn set(enabled: bool) -> Self {
        Self(set_const_fold_override(Some(enabled)))
    }
}

impl Drop for FoldOverrideGuard {
    fn drop(&mut self) {
        set_const_fold_override(self.0);
    }
}

/// Compile `expr` as a standalone expression with the fold forced on/off,
/// then execute it. Returns the compiled instructions and the VM outcome.
fn compile_and_run(expr: &Spanned<TirExpr>, fold: bool) -> (Vec<Opcode>, Result<Value, String>) {
    crate::bytecode_vm::compile::ensure_const_fold_executor_installed();
    let _guard = FoldOverrideGuard::set(fold);
    let mut compiler = BytecodeCompiler::with_resolved_constants(test_constants());
    let idx = compiler
        .compile_expression("TestExpr", expr)
        .expect("expression should compile");
    let chunk = compiler.finish();
    let instructions = chunk.get_function(idx).instructions.clone();
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    let outcome = vm.execute_function(idx).map_err(|e| e.to_string());
    (instructions, outcome)
}

fn has_opcode(instructions: &[Opcode], pred: fn(&Opcode) -> bool) -> bool {
    instructions.iter().any(pred)
}

/// Differential harness (review H2(b)): compile TWICE (fold on/off), execute
/// both, assert result Values ==, same enum discriminant, and identical
/// fingerprints. Returns both instruction streams for shape assertions.
fn assert_differential(expr: &Spanned<TirExpr>) -> (Vec<Opcode>, Vec<Opcode>) {
    let (instr_on, v_on) = compile_and_run(expr, true);
    let (instr_off, v_off) = compile_and_run(expr, false);
    match (&v_on, &v_off) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "fold on/off values must be equal");
            assert_eq!(
                std::mem::discriminant(a),
                std::mem::discriminant(b),
                "fold on/off enum discriminants must match: {a:?} vs {b:?}"
            );
            assert_eq!(
                format!("{:?}", a.fingerprint_extend(0)),
                format!("{:?}", b.fingerprint_extend(0)),
                "fold on/off fingerprints must be identical"
            );
        }
        (Err(a), Err(b)) => {
            assert_eq!(a, b, "fold on/off errors must be identical");
        }
        other => panic!("fold on/off outcomes diverged: {other:?}"),
    }
    (instr_on, instr_off)
}

#[test]
fn set_enum_subseteq_fusion_preserves_left_constant_fold() {
    crate::bytecode_vm::compile::ensure_const_fold_executor_installed();
    let _guard = FoldOverrideGuard::set(true);
    let expr = subseteq(
        set_enum(vec![int_expr(1), int_expr(2)]),
        set_enum(vec![int_expr(1), int_expr(2), int_expr(3)]),
    );
    let mut compiler = BytecodeCompiler::with_resolved_constants(test_constants());
    compiler.enable_set_enum_subseteq();
    let idx = compiler
        .compile_expression("SetEnumSubseteqConstFold", &expr)
        .expect("expression should compile");
    let chunk = compiler.finish();
    let instructions = &chunk.get_function(idx).instructions;

    assert_eq!(
        instructions
            .iter()
            .filter(|op| matches!(op, Opcode::LoadConst { .. }))
            .count(),
        2,
        "both constant set operands must fold, got {instructions:?}"
    );
    assert!(!has_opcode(instructions, |op| matches!(
        op,
        Opcode::SetEnum { .. }
    )));
    assert!(has_opcode(instructions, |op| matches!(
        op,
        Opcode::Subseteq { .. }
    )));
    assert!(!has_opcode(instructions, |op| matches!(
        op,
        Opcode::SetEnumSubseteq { .. }
    )));
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    assert_eq!(vm.execute_function(idx).unwrap(), Value::Bool(true));
}

// === STEP-1 PRECONDITION (review H4) ===

/// `defaultInitValue` — a cfg-assigned CONSTANT model value — must resolve
/// through `resolved_constants` at the Name arm, or the headline MCTypeOK
/// codomain fold silently never fires. `Proc \cup {defaultInitValue}` must
/// fold to a single LoadConst.
#[test]
fn default_init_value_resolves_and_union_folds_to_load_const() {
    let expr = union(
        name_expr("Proc"),
        set_enum(vec![name_expr("defaultInitValue")]),
    );
    let (instructions, outcome) = compile_and_run(&expr, true);
    assert!(
        has_opcode(&instructions, |op| matches!(op, Opcode::LoadConst { .. })),
        "fold must produce a LoadConst, got {instructions:?}"
    );
    assert!(
        !has_opcode(&instructions, |op| matches!(
            op,
            Opcode::SetUnion { .. } | Opcode::SetEnum { .. }
        )),
        "fold must eliminate the per-state constructor opcodes, got {instructions:?}"
    );
    let value = outcome.expect("folded expression should execute");
    assert_eq!(
        value,
        Value::set([mv("p1"), mv("p2"), mv("p3"), mv("defaultInitValue")])
    );
}

/// The full measured codomain folds away its SetUnion/Powerset opcodes.
#[test]
fn mc_type_ok_codomain_folds_and_matches_unfolded_value() {
    let expr = mc_type_ok_codomain();
    let (instr_on, instr_off) = assert_differential(&expr);
    assert!(
        !has_opcode(&instr_on, |op| matches!(
            op,
            Opcode::SetUnion { .. } | Opcode::Powerset { .. } | Opcode::SetEnum { .. }
        )),
        "codomain must fold fully, got {instr_on:?}"
    );
    assert!(
        has_opcode(&instr_off, |op| matches!(op, Opcode::SetUnion { .. })),
        "fold-off baseline must keep the constructor opcodes"
    );
}

// === Differential matrix (review H2(b)) ===

/// `[Proc -> BOOLEAN]`: FuncSet is a LAZY O(1) constructor, so it keeps its
/// opcode at top level (folding it would only trade an Arc bump for a
/// trust-cg-visible opcode change); the differential still holds.
#[test]
fn differential_func_set_proc_to_boolean() {
    let expr = func_set(name_expr("Proc"), name_expr("BOOLEAN"));
    let (instr_on, _) = assert_differential(&expr);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::FuncSet { .. })),
        "[Proc -> BOOLEAN] keeps its lazy constructor opcode, got {instr_on:?}"
    );
}

#[test]
fn differential_nested_union() {
    let expr = union(
        union(set_enum(vec![int_expr(1)]), set_enum(vec![int_expr(2)])),
        range(1, 3),
    );
    let (instr_on, _) = assert_differential(&expr);
    assert!(
        !has_opcode(&instr_on, |op| matches!(op, Opcode::SetUnion { .. })),
        "nested union must fold, got {instr_on:?}"
    );
}

/// Top-level `SUBSET` is a LAZY O(1) constructor: it keeps its Powerset
/// opcode (never folds on its own), and runtime evaluation stays lazy. Full
/// differential (incl. the element-enumerating fingerprint) runs on the
/// small 2^10 case.
#[test]
fn differential_subset_of_interval_stays_lazy() {
    let expr = powerset(range(1, 10));
    let (instr_on, v_on) = compile_and_run(&expr, true);
    assert_differential(&expr);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::Powerset { .. })),
        "top-level SUBSET keeps its lazy constructor opcode, got {instr_on:?}"
    );
    let value = v_on.expect("SUBSET (1..10) should execute");
    assert!(
        matches!(value, Value::Subset(_)),
        "SUBSET must stay lazy, got discriminant of {value:?}"
    );
}

/// `SUBSET (1..20)` (2^20 elements — over-cap if it were ever materialized):
/// stays a lazy Powerset opcode with fold on and off. Fingerprinting a
/// 2^20-element powerset enumerates every subset (~90s), so this case
/// asserts value + discriminant equality only — SubsetValue equality
/// compares bases in O(1).
#[test]
fn differential_subset_of_large_interval_stays_lazy() {
    let expr = powerset(range(1, 20));
    let (instr_on, v_on) = compile_and_run(&expr, true);
    let (_, v_off) = compile_and_run(&expr, false);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::Powerset { .. })),
        "top-level SUBSET keeps its lazy constructor opcode, got {instr_on:?}"
    );
    let (a, b) = (
        v_on.expect("fold-on SUBSET (1..20) should execute"),
        v_off.expect("fold-off SUBSET (1..20) should execute"),
    );
    assert_eq!(a, b, "fold on/off values must be equal");
    assert_eq!(std::mem::discriminant(&a), std::mem::discriminant(&b));
    assert!(
        matches!(a, Value::Subset(_)),
        "SUBSET must stay lazy, got {a:?}"
    );
}

/// A quantifier binder shadowing the constant `Proc` must refuse the fold
/// (binding-scope check, last-binding-wins) — inside the loop `Proc` is the
/// bound variable, not the CONSTANT.
#[test]
fn differential_quantifier_shadowed_name_refuses_fold() {
    // \E Proc \in {{1,5},{2}} : 5 \in (Proc \cup {7})
    let domain = set_enum(vec![
        set_enum(vec![int_expr(1), int_expr(5)]),
        set_enum(vec![int_expr(2)]),
    ]);
    let body = set_in(
        int_expr(5),
        union(name_expr("Proc"), set_enum(vec![int_expr(7)])),
    );
    let expr = spanned(TirExpr::Exists {
        vars: vec![TirBoundVar {
            name: "Proc".to_string(),
            name_id: intern_name("Proc"),
            domain: Some(Box::new(domain)),
            pattern: None,
        }],
        body: Box::new(body),
    });
    let (instr_on, _) = assert_differential(&expr);
    // The shadowed `Proc \cup {7}` must NOT fold: its SetUnion survives.
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::SetUnion { .. })),
        "shadowed-name union must refuse the fold, got {instr_on:?}"
    );
    let (_, outcome) = compile_and_run(&expr, true);
    assert_eq!(outcome, Ok(Value::Bool(true)));
}

/// Work-fuse (review H1): a union that would materialize 200k elements at
/// compile time exceeds the ~100k budget and must refuse (the runtime never
/// pays for runtime-dead branches; compile time must not either).
///
/// The refusal is for the UNION node only: its children (`1..200000` — a lazy
/// O(1) Interval — and `{0}`) legitimately fold individually, so the
/// expensive SetUnion materialization still happens at runtime, never at
/// compile time.
#[test]
fn differential_over_budget_union_refuses_fold() {
    let expr = union(range(1, 200_000), set_enum(vec![int_expr(0)]));
    let (instr_on, instr_off) = assert_differential(&expr);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::SetUnion { .. })),
        "over-budget union must refuse the fold, got {instr_on:?}"
    );
    assert!(
        has_opcode(&instr_off, |op| matches!(op, Opcode::SetUnion { .. })),
        "fold-off baseline keeps SetUnion"
    );
}

// === Error-site fidelity ===

/// `SUBSET 5` must refuse to fold and produce the identical runtime error at
/// the same point (the Powerset opcode still executes per evaluation).
#[test]
fn error_fidelity_subset_of_non_set() {
    let expr = powerset(int_expr(5));
    let (instr_on, v_on) = compile_and_run(&expr, true);
    let (_, v_off) = compile_and_run(&expr, false);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::Powerset { .. })),
        "SUBSET 5 must keep its Powerset opcode, got {instr_on:?}"
    );
    let err_on = v_on.expect_err("SUBSET 5 must error at runtime");
    let err_off = v_off.expect_err("SUBSET 5 must error at runtime");
    assert_eq!(err_on, err_off, "identical runtime error required");
}

/// Union of a non-set must refuse to fold and keep its runtime error.
#[test]
fn error_fidelity_union_of_non_set() {
    let expr = union(int_expr(5), set_enum(vec![int_expr(1)]));
    let (instr_on, v_on) = compile_and_run(&expr, true);
    let (_, v_off) = compile_and_run(&expr, false);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::SetUnion { .. })),
        "5 \\cup {{1}} must keep its SetUnion opcode, got {instr_on:?}"
    );
    let err_on = v_on.expect_err("5 \\cup {1} must error at runtime");
    let err_off = v_off.expect_err("5 \\cup {1} must error at runtime");
    assert_eq!(err_on, err_off, "identical runtime error required");
}

// === Membership over folded values ===

/// SetIn over the folded MCTypeOK codomain constant gives identical verdicts
/// for in-universe and OUT-of-universe candidates — a violating TypeOK must
/// still violate.
#[test]
fn membership_verdicts_identical_over_folded_codomain() {
    for (candidate, expected) in [
        // {p1} \in SUBSET Proc — in-universe via the powerset branch.
        (Value::set([mv("p1")]), true),
        // p1 \in Proc — in-universe via the union's right branch.
        (mv("p1"), true),
        (mv("defaultInitValue"), true),
        // OUT of universe: an integer is in neither branch.
        (Value::SmallInt(42), false),
        // OUT of universe: a set containing a non-Proc element.
        (Value::set([mv("p1"), mv("zz")]), false),
    ] {
        let expr = set_in(const_expr(candidate), mc_type_ok_codomain());
        let (instr_on, v_on) = compile_and_run(&expr, true);
        let (_, v_off) = compile_and_run(&expr, false);
        assert!(
            !has_opcode(&instr_on, |op| matches!(op, Opcode::SetUnion { .. })),
            "codomain must be folded in the membership test, got {instr_on:?}"
        );
        assert_eq!(v_on, v_off, "fold on/off membership verdicts must match");
        assert_eq!(v_on, Ok(Value::Bool(expected)));
    }
}

/// SetIn over `[Proc -> BOOLEAN]` (lazy FuncSet opcode over LoadConst
/// operands) gives identical verdicts for in-universe and OUT-of-universe
/// candidates with the fold on and off.
#[test]
fn membership_verdicts_identical_over_func_set() {
    let in_universe = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (mv("p1"), Value::Bool(true)),
        (mv("p2"), Value::Bool(false)),
        (mv("p3"), Value::Bool(true)),
    ])));
    // Wrong domain: missing p3.
    let wrong_domain = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (mv("p1"), Value::Bool(true)),
        (mv("p2"), Value::Bool(false)),
    ])));
    // Wrong codomain: p1 maps outside BOOLEAN.
    let wrong_codomain = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (mv("p1"), Value::SmallInt(1)),
        (mv("p2"), Value::Bool(false)),
        (mv("p3"), Value::Bool(true)),
    ])));

    for (candidate, expected) in [
        (in_universe, true),
        (wrong_domain, false),
        (wrong_codomain, false),
    ] {
        let expr = set_in(
            const_expr(candidate),
            func_set(name_expr("Proc"), name_expr("BOOLEAN")),
        );
        let (_, v_on) = compile_and_run(&expr, true);
        let (_, v_off) = compile_and_run(&expr, false);
        assert_eq!(v_on, v_off, "fold on/off membership verdicts must match");
        assert_eq!(v_on, Ok(Value::Bool(expected)));
    }
}

// === Constant-domain set comprehensions (GameOfLife `nbrs`/`points`) ===

fn times(components: Vec<Spanned<TirExpr>>) -> Spanned<TirExpr> {
    spanned(TirExpr::Times(components))
}

fn tuple(elements: Vec<Spanned<TirExpr>>) -> Spanned<TirExpr> {
    spanned(TirExpr::Tuple(elements))
}

fn neq(left: Spanned<TirExpr>, right: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::Cmp {
        left: Box::new(left),
        op: TirCmpOp::Neq,
        right: Box::new(right),
    })
}

fn add(left: Spanned<TirExpr>, right: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::ArithBinOp {
        left: Box::new(left),
        op: TirArithOp::Add,
        right: Box::new(right),
    })
}

fn plain_var(name: &str, domain: Spanned<TirExpr>) -> TirBoundVar {
    TirBoundVar {
        name: name.to_string(),
        name_id: intern_name(name),
        domain: Some(Box::new(domain)),
        pattern: None,
    }
}

fn set_filter(var: TirBoundVar, body: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::SetFilter {
        var,
        body: Box::new(body),
    })
}

fn set_builder(body: Spanned<TirExpr>, vars: Vec<TirBoundVar>) -> Spanned<TirExpr> {
    spanned(TirExpr::SetBuilder {
        body: Box::new(body),
        vars,
    })
}

/// GameOfLife's `nbrs == {x \in {-1,0,1} \X {-1,0,1} : x /= <<0,0>>}`: a
/// constant `SetFilter` over a constant `Times` cross product. It must fold to
/// a single `LoadConst` (no `SetFilterBegin`/`Times` loop opcodes survive),
/// and the folded value must be the exact 8-element set the runtime builds.
#[test]
fn constant_set_filter_over_times_folds_to_load_const() {
    let domain = times(vec![
        set_enum(vec![int_expr(-1), int_expr(0), int_expr(1)]),
        set_enum(vec![int_expr(-1), int_expr(0), int_expr(1)]),
    ]);
    let var = plain_var("x", domain);
    let expr = set_filter(
        var,
        neq(name_expr("x"), tuple(vec![int_expr(0), int_expr(0)])),
    );

    let (instr_on, v_on) = compile_and_run(&expr, true);
    assert_differential(&expr);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::LoadConst { .. })),
        "constant nbrs SetFilter must fold to a LoadConst, got {instr_on:?}"
    );
    assert!(
        !has_opcode(&instr_on, |op| matches!(
            op,
            Opcode::SetFilterBegin { .. } | Opcode::Times { .. } | Opcode::LoopNext { .. }
        )),
        "no comprehension loop opcode should survive the fold, got {instr_on:?}"
    );
    let value = v_on.expect("nbrs should execute");
    let Value::Set(set) = &value else {
        panic!("expected a set, got {value:?}");
    };
    assert_eq!(
        set.len(),
        8,
        "nbrs has 8 offsets (9 - <<0,0>>), got {value:?}"
    );
}

/// A constant `SetBuilder` `{x + 1 : x \in {1,2,3}}` folds to `LoadConst`
/// `{2,3,4}`.
#[test]
fn constant_set_builder_folds_to_load_const() {
    let var = plain_var("x", set_enum(vec![int_expr(1), int_expr(2), int_expr(3)]));
    let expr = set_builder(add(name_expr("x"), int_expr(1)), vec![var]);

    let (instr_on, v_on) = compile_and_run(&expr, true);
    assert_differential(&expr);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::LoadConst { .. })),
        "constant SetBuilder must fold to a LoadConst, got {instr_on:?}"
    );
    assert!(
        !has_opcode(&instr_on, |op| matches!(
            op,
            Opcode::SetBuilderBegin { .. } | Opcode::LoopNext { .. }
        )),
        "no SetBuilder loop opcode should survive the fold, got {instr_on:?}"
    );
    assert_eq!(
        v_on.expect("builder should execute"),
        Value::set([Value::SmallInt(2), Value::SmallInt(3), Value::SmallInt(4)])
    );
}

/// FAIL-CLOSED: GameOfLife's `points == {<<p[1]+x, p[2]+y>> : <<x,y>> \in
/// nbrs}` reads the OUTER runtime binder `p`, so it is NOT a compile-time
/// constant and MUST NOT fold. Modelled as `\E p \in {10,20} : (5 \in {p + x :
/// x \in {1,2}})`: the inner `SetBuilder` reads the quantifier binder `p`.
/// Its `SetBuilderBegin` loop must survive with fold on, and the fold-on/off
/// verdicts must be identical.
#[test]
fn set_builder_reading_outer_binder_refuses_fold() {
    // {p + x : x \in {1,2}}
    let inner = set_builder(
        add(name_expr("p"), name_expr("x")),
        vec![plain_var("x", set_enum(vec![int_expr(1), int_expr(2)]))],
    );
    // 5 \in {p + x : x \in {1,2}}   (true iff p = 4 or p = 3; here p in {10,20} → false)
    let body = set_in(int_expr(5), inner);
    let expr = spanned(TirExpr::Exists {
        vars: vec![plain_var("p", set_enum(vec![int_expr(10), int_expr(20)]))],
        body: Box::new(body),
    });

    let (instr_on, _) = assert_differential(&expr);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::SetBuilderBegin { .. })),
        "a SetBuilder reading a runtime binder must refuse the fold, got {instr_on:?}"
    );
    let (_, outcome) = compile_and_run(&expr, true);
    assert_eq!(outcome, Ok(Value::Bool(false)));
}

/// FAIL-CLOSED: a constant `SetFilter` whose predicate reads an outer runtime
/// binder must not fold. `\E p \in {1,2} : (\E q \in {q0 \in {5,6} : q0 = p} :
/// TRUE)` — the inner filter's predicate `q0 = p` reads the quantifier binder
/// `p`, so its `SetFilterBegin` must survive.
#[test]
fn set_filter_reading_outer_binder_refuses_fold() {
    let inner_filter = set_filter(
        plain_var("q0", set_enum(vec![int_expr(5), int_expr(6)])),
        spanned(TirExpr::Cmp {
            left: Box::new(name_expr("q0")),
            op: TirCmpOp::Eq,
            right: Box::new(name_expr("p")),
        }),
    );
    let body = spanned(TirExpr::Exists {
        vars: vec![plain_var("q", inner_filter)],
        body: Box::new(const_expr(Value::Bool(true))),
    });
    let expr = spanned(TirExpr::Exists {
        vars: vec![plain_var("p", set_enum(vec![int_expr(1), int_expr(2)]))],
        body: Box::new(body),
    });

    let (instr_on, _) = assert_differential(&expr);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::SetFilterBegin { .. })),
        "a SetFilter reading a runtime binder must refuse the fold, got {instr_on:?}"
    );
}

/// A tuple-destructuring binder `{a + b : <<a, b>> \in {<<1,10>>, <<2,20>>}}`
/// folds: the pattern names `a`/`b` are bound by the comprehension, not free.
#[test]
fn constant_set_builder_tuple_pattern_folds() {
    let var = TirBoundVar {
        name: "t".to_string(),
        name_id: intern_name("t"),
        domain: Some(Box::new(set_enum(vec![
            tuple(vec![int_expr(1), int_expr(10)]),
            tuple(vec![int_expr(2), int_expr(20)]),
        ]))),
        pattern: Some(TirBoundPattern::Tuple(vec![
            ("a".to_string(), intern_name("a")),
            ("b".to_string(), intern_name("b")),
        ])),
    };
    let expr = set_builder(add(name_expr("a"), name_expr("b")), vec![var]);

    let (instr_on, v_on) = compile_and_run(&expr, true);
    assert_differential(&expr);
    assert!(
        has_opcode(&instr_on, |op| matches!(op, Opcode::LoadConst { .. })),
        "tuple-pattern constant SetBuilder must fold, got {instr_on:?}"
    );
    assert_eq!(
        v_on.expect("builder should execute"),
        Value::set([Value::SmallInt(11), Value::SmallInt(22)])
    );
}

// === Telemetry ===

/// Successful folds bump the process-wide fold counter (visible under
/// TY_BYTECODE_VM_STATS=1 as per-fold stderr lines).
#[test]
fn fold_count_increments_on_fold() {
    let before = const_fold_count();
    let expr = mc_type_ok_codomain();
    let (_, outcome) = compile_and_run(&expr, true);
    outcome.expect("codomain should execute");
    assert!(
        const_fold_count() > before,
        "const_fold_count must increase after a successful fold"
    );
}
