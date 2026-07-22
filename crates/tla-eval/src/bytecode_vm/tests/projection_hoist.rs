// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! VM parity tests for the opt-in SetFilter projection hoist.

use super::BytecodeVm;
use std::sync::Arc;
use tla_core::{NameId, Span, Spanned};
use tla_tir::bytecode::{BytecodeChunk, BytecodeCompiler, BytecodeFunction, CalleeInfo, Opcode};
use tla_tir::{TirBoundVar, TirExpr, TirNameKind, TirNameRef, TirType};
use tla_value::Value;

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

fn projection_callees() -> std::collections::HashMap<String, CalleeInfo> {
    let projection = spanned(TirExpr::FuncApply {
        func: Box::new(name("p")),
        arg: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(2),
            ty: TirType::Int,
        })),
    });
    std::collections::HashMap::from([(
        "Edges".to_string(),
        CalleeInfo {
            params: vec!["p".to_string()],
            body: Arc::new(projection),
            ast_body: None,
        },
    )])
}

fn filter_expr(domain: Value) -> Spanned<TirExpr> {
    let body = spanned(TirExpr::In {
        elem: Box::new(spanned(TirExpr::Tuple(vec![name("outer"), name("c")]))),
        set: Box::new(spanned(TirExpr::Apply {
            op: Box::new(name("Edges")),
            args: vec![name("graph")],
        })),
    });
    spanned(TirExpr::SetFilter {
        var: TirBoundVar {
            name: "c".to_string(),
            name_id: NameId(0),
            domain: Some(Box::new(spanned(TirExpr::Const {
                value: domain,
                ty: TirType::Dyn,
            }))),
            pattern: None,
        },
        body: Box::new(body),
    })
}

fn compile_filter(domain: Value, graph: Value, outer: Value, hoist: bool) -> (BytecodeChunk, u16) {
    let filter = filter_expr(domain);
    let callees = projection_callees();

    let mut compiler = BytecodeCompiler::new();
    compiler.enable_tuple2_set_in();
    compiler.enable_reg_recycling();
    if hoist {
        compiler.enable_set_filter_projection_hoist();
    }
    let filter_idx = compiler
        .compile_expression_with_callees(
            "Children",
            &["graph".to_string(), "outer".to_string()],
            &filter,
            &callees,
        )
        .expect("Children bytecode");
    let mut chunk = compiler.finish();

    let graph_idx = chunk.constants.add_value(graph);
    let outer_idx = chunk.constants.add_value(outer);
    let mut main = BytecodeFunction::new("Main".to_string(), 0);
    main.emit(Opcode::LoadConst {
        rd: 0,
        idx: graph_idx,
    });
    main.emit(Opcode::LoadConst {
        rd: 1,
        idx: outer_idx,
    });
    main.emit(Opcode::Call {
        rd: 2,
        op_idx: filter_idx,
        args_start: 0,
        argc: 2,
    });
    main.emit(Opcode::Ret { rs: 2 });
    let main_idx = chunk.add_function(main);
    (chunk, main_idx)
}

fn compile_nested_filter(graph: Value, hoist: bool) -> (BytecodeChunk, u16) {
    let filter = filter_expr(Value::set([
        Value::SmallInt(1),
        Value::SmallInt(2),
        Value::SmallInt(3),
    ]));
    let nested = spanned(TirExpr::SetBuilder {
        body: Box::new(filter),
        vars: vec![TirBoundVar {
            name: "outer".to_string(),
            name_id: NameId(0),
            domain: Some(Box::new(spanned(TirExpr::Const {
                value: Value::set([Value::SmallInt(9), Value::SmallInt(10)]),
                ty: TirType::Dyn,
            }))),
            pattern: None,
        }],
    });
    let callees = projection_callees();
    let mut compiler = BytecodeCompiler::new();
    compiler.enable_tuple2_set_in();
    compiler.enable_reg_recycling();
    if hoist {
        compiler.enable_set_filter_projection_hoist();
    }
    let nested_idx = compiler
        .compile_expression_with_callees(
            "NestedChildren",
            &["graph".to_string()],
            &nested,
            &callees,
        )
        .expect("nested Children bytecode");
    let mut chunk = compiler.finish();

    let graph_idx = chunk.constants.add_value(graph);
    let mut main = BytecodeFunction::new("Main".to_string(), 0);
    main.emit(Opcode::LoadConst {
        rd: 0,
        idx: graph_idx,
    });
    main.emit(Opcode::Call {
        rd: 1,
        op_idx: nested_idx,
        args_start: 0,
        argc: 1,
    });
    main.emit(Opcode::Ret { rs: 1 });
    let main_idx = chunk.add_function(main);
    (chunk, main_idx)
}

fn execute_filter(domain: Value, graph: Value, outer: Value, hoist: bool) -> Result<Value, String> {
    let (chunk, main_idx) = compile_filter(domain, graph, outer, hoist);
    let result = BytecodeVm::new(&chunk, &[], None)
        .execute_function(main_idx)
        .map_err(|err| err.to_string());
    result
}

#[test]
fn set_filter_projection_hoist_matches_call_path() {
    let outer = Value::SmallInt(9);
    let edges = Value::set([
        Value::tuple([outer.clone(), Value::SmallInt(1)]),
        Value::tuple([outer.clone(), Value::SmallInt(3)]),
        Value::tuple([Value::SmallInt(8), Value::SmallInt(2)]),
    ]);
    let graph = Value::tuple([Value::empty_set(), edges]);
    let domain = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);

    let baseline = execute_filter(domain.clone(), graph.clone(), outer.clone(), false);
    let hoisted = execute_filter(domain, graph, outer, true);
    assert_eq!(hoisted, baseline);
    assert_eq!(
        hoisted.unwrap(),
        Value::set([Value::SmallInt(1), Value::SmallInt(3)])
    );
}

#[test]
fn set_filter_projection_hoist_preserves_empty_and_error_order() {
    let invalid_graph = Value::SmallInt(7);
    let outer = Value::SmallInt(9);

    // Empty domains must jump over the projection, even though graph[2]
    // would fail if evaluated.
    let empty_baseline = execute_filter(
        Value::empty_set(),
        invalid_graph.clone(),
        outer.clone(),
        false,
    );
    let empty_hoisted = execute_filter(
        Value::empty_set(),
        invalid_graph.clone(),
        outer.clone(),
        true,
    );
    assert_eq!(empty_hoisted, empty_baseline);
    assert_eq!(empty_hoisted.unwrap(), Value::empty_set());

    // An invalid domain fails before the also-invalid projection.
    let invalid_domain = Value::SmallInt(11);
    let domain_baseline = execute_filter(
        invalid_domain.clone(),
        invalid_graph.clone(),
        outer.clone(),
        false,
    );
    let domain_hoisted = execute_filter(invalid_domain, invalid_graph.clone(), outer.clone(), true);
    assert_eq!(domain_hoisted, domain_baseline);
    assert!(domain_hoisted.unwrap_err().contains("set filter"));

    // For a non-empty valid domain, both paths surface the projection error.
    let nonempty = Value::set([Value::SmallInt(1)]);
    let projection_baseline = execute_filter(
        nonempty.clone(),
        invalid_graph.clone(),
        outer.clone(),
        false,
    );
    let projection_hoisted = execute_filter(nonempty, invalid_graph, outer, true);
    assert_eq!(projection_hoisted, projection_baseline);
    assert!(projection_hoisted.is_err());
}

#[test]
fn set_filter_projection_hoist_preserves_non_set_membership_error() {
    let outer = Value::SmallInt(9);
    let graph = Value::tuple([Value::empty_set(), Value::SmallInt(42)]);
    let domain = Value::set([Value::SmallInt(1)]);

    let baseline = execute_filter(domain.clone(), graph.clone(), outer.clone(), false);
    let hoisted = execute_filter(domain, graph, outer, true);
    assert_eq!(hoisted, baseline);
    assert!(hoisted
        .unwrap_err()
        .contains("expected enumerable set for \\in"));
}

#[test]
fn set_filter_projection_hoist_matches_nested_recycled_loops() {
    let edges = Value::set([
        Value::tuple([Value::SmallInt(9), Value::SmallInt(1)]),
        Value::tuple([Value::SmallInt(9), Value::SmallInt(3)]),
        Value::tuple([Value::SmallInt(10), Value::SmallInt(2)]),
    ]);
    let graph = Value::tuple([Value::empty_set(), edges]);

    let (baseline_chunk, baseline_idx) = compile_nested_filter(graph.clone(), false);
    let baseline = BytecodeVm::new(&baseline_chunk, &[], None)
        .execute_function(baseline_idx)
        .expect("baseline nested filter");
    let (hoisted_chunk, hoisted_idx) = compile_nested_filter(graph, true);
    let hoisted = BytecodeVm::new(&hoisted_chunk, &[], None)
        .execute_function(hoisted_idx)
        .expect("hoisted nested filter");

    assert_eq!(hoisted, baseline);
    assert_eq!(
        hoisted,
        Value::set([
            Value::set([Value::SmallInt(1), Value::SmallInt(3)]),
            Value::set([Value::SmallInt(2)]),
        ])
    );
}
