// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! VM differential tests for the opt-in exact Round-shape call fusion.

use tla_value::Rp;

use super::BytecodeVm;
use std::collections::HashMap;
use std::sync::Arc;
use tla_core::{NameId, Span, Spanned};
use tla_tir::bytecode::{
    BuiltinOp, BytecodeChunk, BytecodeCompiler, BytecodeFunction, CalleeInfo, Opcode,
};
use tla_tir::{TirCmpOp, TirExpr, TirNameKind, TirNameRef, TirType};
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
        name_id: NameId(0),
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

fn compile_round(input: Value, fused: bool) -> (BytecodeChunk, u16, u16) {
    let body = spanned(TirExpr::Apply {
        op: Box::new(name("Round")),
        args: vec![name("input")],
    });
    let mut compiler = BytecodeCompiler::new();
    if fused {
        compiler.enable_round_shape_apply();
    }
    let entry_idx = compiler
        .compile_expression_with_callees(
            "ApplyRound",
            &["input".to_string()],
            &body,
            &round_callees(),
        )
        .expect("Round entry bytecode");
    let mut chunk = compiler.finish();

    let input_idx = chunk.constants.add_value(input);
    let mut main = BytecodeFunction::new("Main".to_string(), 0);
    main.emit(Opcode::LoadConst {
        rd: 0,
        idx: input_idx,
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

fn execute_round(input: Value, fused: bool) -> Result<Value, String> {
    let (chunk, _, main_idx) = compile_round(input, fused);
    let result = BytecodeVm::new(&chunk, &[], None)
        .execute_function(main_idx)
        .map_err(|error| error.to_string());
    result
}

fn func(entries: Vec<(Value, Value)>) -> Value {
    Value::Func(Rp::new(FuncValue::from_sorted_entries(entries)))
}

fn int_func(min: i64, max: i64, values: Vec<Value>) -> Value {
    Value::IntFunc(Rp::new(IntIntervalFunc::new(min, max, values)))
}

#[test]
fn exact_shape_activates_only_when_enabled() {
    let input = Value::tuple([Value::SmallInt(10), Value::SmallInt(20)]);
    let (baseline, baseline_idx, _) = compile_round(input.clone(), false);
    assert!(baseline
        .get_function(baseline_idx)
        .instructions
        .iter()
        .all(|opcode| !matches!(
            opcode,
            Opcode::CallBuiltin {
                builtin: BuiltinOp::RoundApply,
                ..
            }
        )));

    let (fused, fused_idx, _) = compile_round(input, true);
    assert!(fused
        .get_function(fused_idx)
        .instructions
        .iter()
        .any(|opcode| matches!(
            opcode,
            Opcode::CallBuiltin {
                builtin: BuiltinOp::RoundApply,
                argc: 1,
                ..
            }
        )));
}

#[test]
fn fused_round_matches_ordinary_call_for_all_function_representations_and_errors() {
    let cases = vec![
        ("empty tuple", Value::tuple([])),
        (
            "tuple",
            Value::tuple([Value::SmallInt(10), Value::SmallInt(20)]),
        ),
        ("short tuple error", Value::tuple([Value::SmallInt(10)])),
        ("empty sequence", Value::seq([])),
        (
            "sequence",
            Value::seq([Value::SmallInt(30), Value::SmallInt(40)]),
        ),
        ("empty function", func(vec![])),
        (
            "function",
            func(vec![
                (Value::SmallInt(1), Value::SmallInt(50)),
                (Value::SmallInt(2), Value::SmallInt(60)),
            ]),
        ),
        (
            "function missing index error",
            func(vec![(Value::SmallInt(1), Value::SmallInt(50))]),
        ),
        ("empty intfunc", int_func(1, 0, vec![])),
        (
            "intfunc",
            int_func(1, 2, vec![Value::SmallInt(70), Value::SmallInt(80)]),
        ),
        (
            "shifted intfunc error",
            int_func(3, 4, vec![Value::SmallInt(70), Value::SmallInt(80)]),
        ),
        (
            "empty record",
            Value::record(std::iter::empty::<(&'static str, Value)>()),
        ),
        (
            "record index type error",
            Value::record([("two", Value::SmallInt(90))]),
        ),
        ("malformed scalar error", Value::SmallInt(123)),
    ];

    for (case, input) in cases {
        let ordinary = execute_round(input.clone(), false);
        let fused = execute_round(input, true);
        assert_eq!(fused, ordinary, "{case}");
    }
}
