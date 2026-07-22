// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Differential VM tests for the opt-in exact RoundStepEq fusion.

use tla_value::Rp;

use super::BytecodeVm;
use num_bigint::BigInt;
use std::collections::HashMap;
use std::sync::Arc;
use tla_core::{intern_name, Span, Spanned};
use tla_tir::bytecode::{BytecodeChunk, BytecodeCompiler, BytecodeFunction, CalleeInfo, Opcode};
use tla_tir::{TirArithOp, TirBoundVar, TirCmpOp, TirExpr, TirNameKind, TirNameRef, TirType};
use tla_value::{FuncValue, IntIntervalFunc, Value};

fn spanned(node: TirExpr) -> Spanned<TirExpr> {
    Spanned {
        node,
        span: Span::default(),
    }
}

fn name(name: &str) -> Spanned<TirExpr> {
    spanned(TirExpr::Name(TirNameRef {
        name: name.to_string(),
        name_id: intern_name(name),
        kind: TirNameKind::Ident,
        ty: TirType::Dyn,
    }))
}

fn int(value: i64) -> Spanned<TirExpr> {
    spanned(TirExpr::Const {
        value: Value::SmallInt(value),
        ty: TirType::Int,
    })
}

fn round_call(arg: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::Apply {
        op: Box::new(name("Round")),
        args: vec![arg],
    })
}

fn round_callees() -> HashMap<String, CalleeInfo> {
    let body = spanned(TirExpr::If {
        cond: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(name("p")),
            op: TirCmpOp::Eq,
            right: Box::new(spanned(TirExpr::Tuple(vec![]))),
        })),
        then_: Box::new(int(0)),
        else_: Box::new(spanned(TirExpr::FuncApply {
            func: Box::new(name("p")),
            arg: Box::new(int(2)),
        })),
    });
    HashMap::from([(
        "Round".to_string(),
        CalleeInfo {
            params: vec!["p".to_string()],
            body: Arc::new(body),
            ast_body: None,
        },
    )])
}

fn predicate(domain: Value) -> Spanned<TirExpr> {
    let body = spanned(TirExpr::Cmp {
        left: Box::new(round_call(name("child"))),
        op: TirCmpOp::Eq,
        right: Box::new(spanned(TirExpr::ArithBinOp {
            left: Box::new(round_call(name("parent"))),
            op: TirArithOp::Sub,
            right: Box::new(int(1)),
        })),
    });
    spanned(TirExpr::Forall {
        vars: vec![TirBoundVar {
            name: "child".to_string(),
            name_id: intern_name("child"),
            domain: Some(Box::new(spanned(TirExpr::Const {
                value: domain,
                ty: TirType::Set(Box::new(TirType::Dyn)),
            }))),
            pattern: None,
        }],
        body: Box::new(body),
    })
}

fn compile(domain: Value, parent: Value, fused: bool) -> (BytecodeChunk, u16, u16) {
    let mut compiler = BytecodeCompiler::new();
    if fused {
        compiler.enable_round_step_eq();
    }
    let entry_idx = compiler
        .compile_expression_with_callees(
            "RoundStep",
            &["parent".to_string()],
            &predicate(domain),
            &round_callees(),
        )
        .expect("Round-step predicate bytecode");
    let mut chunk = compiler.finish();
    let parent_idx = chunk.constants.add_value(parent);
    let mut main = BytecodeFunction::new("Main".to_string(), 0);
    main.emit(Opcode::LoadConst {
        rd: 0,
        idx: parent_idx,
    });
    main.emit(Opcode::Call {
        rd: 1,
        op_idx: entry_idx,
        args_start: 0,
        argc: 1,
    });
    main.emit(Opcode::Ret { rs: 1 });
    let main_idx = chunk.add_function(main);
    (chunk, entry_idx, main_idx)
}

fn evaluate(domain: Value, parent: Value, fused: bool) -> Result<Value, String> {
    let (chunk, _, main_idx) = compile(domain, parent, fused);
    let result = BytecodeVm::new(&chunk, &[], None)
        .execute_function(main_idx)
        .map_err(|error| error.to_string());
    result
}

fn differential(child: Value, parent: Value) -> (Result<Value, String>, Result<Value, String>) {
    let domain = Value::set([child]);
    let ordinary = evaluate(domain.clone(), parent.clone(), false);
    let fused = evaluate(domain, parent, true);
    (ordinary, fused)
}

fn func(entries: impl IntoIterator<Item = (Value, Value)>) -> Value {
    Value::Func(Rp::new(FuncValue::from_sorted_entries(
        entries.into_iter().collect(),
    )))
}

fn int_func(min: i64, max: i64, values: impl IntoIterator<Item = Value>) -> Value {
    Value::IntFunc(Rp::new(IntIntervalFunc::new(
        min,
        max,
        values.into_iter().collect(),
    )))
}

#[test]
fn vm_round_step_eq_flag_requires_exactly_one() {
    for value in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some("01"),
        Some(" 1"),
        Some("1 "),
    ] {
        assert!(
            !crate::tir::vm_round_step_eq_enabled_from_env_value(value),
            "unexpected opt-in for {value:?}"
        );
    }
    assert!(crate::tir::vm_round_step_eq_enabled_from_env_value(Some(
        "1"
    )));
}

#[test]
fn exact_shape_activates_only_when_enabled() {
    let child = Value::tuple([Value::SmallInt(1), Value::SmallInt(3)]);
    let parent = Value::tuple([Value::SmallInt(1), Value::SmallInt(4)]);
    let domain = Value::set([child]);

    let (ordinary, ordinary_idx, _) = compile(domain.clone(), parent.clone(), false);
    assert!(ordinary
        .get_function(ordinary_idx)
        .instructions
        .iter()
        .all(|opcode| !matches!(opcode, Opcode::RoundStepEq { .. })));

    let (fused, fused_idx, _) = compile(domain, parent, true);
    assert_eq!(
        fused
            .get_function(fused_idx)
            .instructions
            .iter()
            .filter(|opcode| matches!(opcode, Opcode::RoundStepEq { .. }))
            .count(),
        1
    );
}

#[test]
fn fused_matches_round_representations_integer_widening_and_equality() {
    let widened = Value::big_int(BigInt::from(i64::MIN) - BigInt::from(1));
    let cases = vec![
        (
            "empty child",
            Value::tuple([]),
            Value::tuple([Value::SmallInt(9), Value::SmallInt(1)]),
            true,
        ),
        (
            "tuple and sequence",
            Value::tuple([Value::SmallInt(9), Value::SmallInt(3)]),
            Value::seq([Value::SmallInt(8), Value::SmallInt(4)]),
            true,
        ),
        (
            "sequence and function",
            Value::seq([Value::SmallInt(9), Value::SmallInt(10)]),
            func([
                (Value::SmallInt(1), Value::SmallInt(8)),
                (Value::SmallInt(2), Value::SmallInt(11)),
            ]),
            true,
        ),
        (
            "function and intfunc",
            func([
                (Value::SmallInt(1), Value::SmallInt(8)),
                (Value::SmallInt(2), Value::SmallInt(20)),
            ]),
            int_func(1, 2, [Value::SmallInt(7), Value::SmallInt(21)]),
            true,
        ),
        (
            "widened subtraction",
            Value::tuple([Value::SmallInt(0), widened]),
            Value::tuple([Value::SmallInt(0), Value::SmallInt(i64::MIN)]),
            true,
        ),
        (
            "ordinary false equality",
            Value::tuple([Value::SmallInt(0), Value::Bool(false)]),
            Value::tuple([Value::SmallInt(0), Value::SmallInt(1)]),
            false,
        ),
        (
            "different integer step",
            Value::tuple([Value::SmallInt(0), Value::SmallInt(3)]),
            Value::tuple([Value::SmallInt(0), Value::SmallInt(5)]),
            false,
        ),
    ];

    for (case, child, parent, expected) in cases {
        let (ordinary, fused) = differential(child, parent);
        assert_eq!(fused, ordinary, "{case}");
        assert_eq!(fused.unwrap(), Value::Bool(expected), "{case}");
    }
}

#[test]
fn child_parent_and_subtraction_errors_match_in_source_order() {
    let cases = [
        (
            "child fails first",
            Value::tuple([Value::SmallInt(1)]),
            Value::SmallInt(99),
        ),
        (
            "parent fails after child",
            Value::tuple([Value::SmallInt(1), Value::SmallInt(2)]),
            Value::tuple([Value::SmallInt(1)]),
        ),
        (
            "subtraction type error",
            Value::tuple([Value::SmallInt(1), Value::SmallInt(2)]),
            Value::tuple([Value::SmallInt(1), Value::Bool(true)]),
        ),
    ];

    for (case, child, parent) in cases {
        let (ordinary, fused) = differential(child, parent);
        assert_eq!(fused, ordinary, "{case}");
        assert!(fused.is_err(), "{case}");
    }

    let (ordinary, fused) = differential(Value::tuple([Value::SmallInt(1)]), Value::SmallInt(99));
    assert_eq!(fused, ordinary);
    let error = fused.unwrap_err();
    assert!(error.contains("<<1>>"), "child error must win: {error}");
}

#[test]
fn empty_forall_short_circuits_before_invalid_parent() {
    let ordinary = evaluate(Value::empty_set(), Value::SmallInt(99), false);
    let fused = evaluate(Value::empty_set(), Value::SmallInt(99), true);
    assert_eq!(fused, ordinary);
    assert_eq!(fused.unwrap(), Value::Bool(true));
}

#[test]
fn false_child_short_circuits_before_a_later_invalid_child() {
    // Canonical Value ordering visits <<0, 3>> before <<1>>. The first child
    // makes 3 = 5 - 1 FALSE; evaluating the later one would fail at [2].
    let domain = Value::set([
        Value::tuple([Value::SmallInt(0), Value::SmallInt(3)]),
        Value::tuple([Value::SmallInt(1)]),
    ]);
    let parent = Value::tuple([Value::SmallInt(0), Value::SmallInt(5)]);
    let ordinary = evaluate(domain.clone(), parent.clone(), false);
    let fused = evaluate(domain, parent, true);
    assert_eq!(fused, ordinary);
    assert_eq!(fused.unwrap(), Value::Bool(false));
}
