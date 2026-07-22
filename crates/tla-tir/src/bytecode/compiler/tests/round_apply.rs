// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compiler proof-boundary tests for the VM-only Round-shape call fusion.

use super::*;
use crate::bytecode::BuiltinOp;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn name(name: &str) -> Spanned<TirExpr> {
    spanned(TirExpr::Name(ident_name(name)))
}

fn int(value: i64) -> Spanned<TirExpr> {
    spanned(TirExpr::Const {
        value: Value::SmallInt(value),
        ty: TirType::Int,
    })
}

fn round_body(param: &str) -> Spanned<TirExpr> {
    spanned(TirExpr::If {
        cond: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(name(param)),
            op: TirCmpOp::Eq,
            right: Box::new(spanned(TirExpr::Tuple(vec![]))),
        })),
        then_: Box::new(int(0)),
        else_: Box::new(spanned(TirExpr::FuncApply {
            func: Box::new(name(param)),
            arg: Box::new(int(2)),
        })),
    })
}

fn callee(name: &str, params: &[&str], body: Spanned<TirExpr>) -> (String, CalleeInfo) {
    (
        name.to_string(),
        CalleeInfo {
            params: params.iter().map(|param| (*param).to_string()).collect(),
            body: Arc::new(body),
            ast_body: None,
        },
    )
}

fn call(name_: &str, arg: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::Apply {
        op: Box::new(name(name_)),
        args: vec![arg],
    })
}

fn has_round_apply(function: &BytecodeFunction) -> bool {
    function.instructions.iter().any(|opcode| {
        matches!(
            opcode,
            Opcode::CallBuiltin {
                builtin: BuiltinOp::RoundApply,
                argc: 1,
                ..
            }
        )
    })
}

#[test]
fn round_shape_is_positive_opt_in_and_evaluates_argument_once() {
    let callees = HashMap::from([callee("Round", &["p"], round_body("p"))]);
    let argument = spanned(TirExpr::ArithBinOp {
        left: Box::new(int(1)),
        op: TirArithOp::Add,
        right: Box::new(int(2)),
    });
    let body = call("Round", argument);

    let mut baseline = BytecodeCompiler::new();
    let baseline_idx = baseline
        .compile_expression_with_callees("Main", &[], &body, &callees)
        .expect("ordinary Round call should compile");
    let baseline_chunk = baseline.finish();
    let baseline_func = baseline_chunk.get_function(baseline_idx);
    assert!(!has_round_apply(baseline_func));
    assert!(baseline_func
        .instructions
        .iter()
        .any(|opcode| matches!(opcode, Opcode::Call { argc: 1, .. })));

    let mut fused = BytecodeCompiler::new();
    fused.enable_round_shape_apply();
    let fused_idx = fused
        .compile_expression_with_callees("Main", &[], &body, &callees)
        .expect("exact Round shape should fuse");
    let fused_chunk = fused.finish();
    let fused_func = fused_chunk.get_function(fused_idx);
    assert!(has_round_apply(fused_func), "{:?}", fused_func.instructions);
    assert_eq!(
        fused_func
            .instructions
            .iter()
            .filter(|opcode| matches!(opcode, Opcode::AddInt { .. }))
            .count(),
        1,
        "the source argument must be compiled exactly once"
    );
}

#[test]
fn round_shape_refuses_replacement_and_forced_external() {
    let callees = HashMap::from([
        callee("Round", &["p"], round_body("p")),
        callee("Other", &["p"], round_body("p")),
    ]);
    let body = call("Round", int(7));

    let mut replacement = BytecodeCompiler::new();
    replacement.enable_round_shape_apply();
    replacement.set_op_replacements(HashMap::from([("Round".to_string(), "Other".to_string())]));
    let idx = replacement
        .compile_expression_with_callees("Main", &[], &body, &callees)
        .expect("replacement call should retain ordinary lowering");
    let chunk = replacement.finish();
    assert!(!has_round_apply(chunk.get_function(idx)));

    let mut external = BytecodeCompiler::new();
    external.enable_round_shape_apply();
    external.set_force_external_ops(HashSet::from(["Round".to_string()]));
    let idx = external
        .compile_expression_with_callees("Main", &[], &body, &callees)
        .expect("forced external call should compile");
    let chunk = external.finish();
    let function = chunk.get_function(idx);
    assert!(!has_round_apply(function));
    assert!(function
        .instructions
        .iter()
        .any(|opcode| matches!(opcode, Opcode::CallExternal { .. })));
}

#[test]
fn round_shape_refuses_local_and_higher_order_shadowing() {
    let callees = HashMap::from([callee("Round", &["p"], round_body("p"))]);

    let local_body = spanned(TirExpr::Let {
        defs: vec![TirLetDef {
            name: "Round".to_string(),
            name_id: tla_core::NameId(0),
            params: vec!["q".to_string()],
            body: round_body("q"),
        }],
        body: Box::new(call("Round", int(7))),
    });
    let mut local = BytecodeCompiler::new();
    local.enable_round_shape_apply();
    let idx = local
        .compile_expression_with_callees("Main", &[], &local_body, &callees)
        .expect("LET-local Round should compile normally");
    let chunk = local.finish();
    assert!(!has_round_apply(chunk.get_function(idx)));

    // The parameter named Round is an operator-valued runtime binding. It
    // shadows the complete global definition and must remain ValueApply.
    let mut higher_order = BytecodeCompiler::new();
    higher_order.enable_round_shape_apply();
    let body = call("Round", name("x"));
    let idx = higher_order
        .compile_expression_with_callees(
            "Main",
            &["Round".to_string(), "x".to_string()],
            &body,
            &callees,
        )
        .expect("higher-order shadow should compile as ValueApply");
    let chunk = higher_order.finish();
    let function = chunk.get_function(idx);
    assert!(!has_round_apply(function));
    assert!(function
        .instructions
        .iter()
        .any(|opcode| matches!(opcode, Opcode::ValueApply { argc: 1, .. })));
}

#[test]
fn round_shape_refuses_incomplete_or_nonexact_bodies() {
    let cases = [
        (
            "wrong then",
            spanned(TirExpr::If {
                cond: Box::new(spanned(TirExpr::Cmp {
                    left: Box::new(name("p")),
                    op: TirCmpOp::Eq,
                    right: Box::new(spanned(TirExpr::Tuple(vec![]))),
                })),
                then_: Box::new(int(1)),
                else_: Box::new(spanned(TirExpr::FuncApply {
                    func: Box::new(name("p")),
                    arg: Box::new(int(2)),
                })),
            }),
        ),
        ("incomplete body", int(0)),
    ];

    for (case, callee_body) in cases {
        let callees = HashMap::from([callee("Round", &["p"], callee_body)]);
        let mut compiler = BytecodeCompiler::new();
        compiler.enable_round_shape_apply();
        let idx = compiler
            .compile_expression_with_callees("Main", &[], &call("Round", int(7)), &callees)
            .unwrap_or_else(|error| panic!("{case}: {error}"));
        let chunk = compiler.finish();
        assert!(!has_round_apply(chunk.get_function(idx)), "{case}");
    }
}
