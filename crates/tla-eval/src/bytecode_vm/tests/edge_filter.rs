// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! VM parity tests for the VM-only `EdgeFilter` comprehension fusion.
//!
//! `EdgeFilter` replaces the whole `{c \in D : <<outer, c>> \in Edges(graph)}`
//! loop with one opcode that range-scans the sorted edge set. These tests drive
//! the exact same `Children`-shaped bytecode as the projection-hoist tests and
//! assert byte-identical results and errors versus the unfused baseline,
//! including the soundness-critical empty-domain / evaluation-order / non-set
//! and `c \notin domain` cases.

use super::BytecodeVm;
use std::sync::Arc;
use tla_core::{NameId, Span, Spanned};
use tla_tir::bytecode::{BytecodeChunk, BytecodeCompiler, BytecodeFunction, CalleeInfo, Opcode};
use tla_tir::{TirBoundVar, TirExpr, TirNameKind, TirNameRef, TirType};
use tla_value::{IntervalValue, Rp, Value};

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

/// `Edges(p) == p[2]` — the projection callee `Children` filters against.
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

/// `Children(graph, outer) == {c \in domain : <<outer, c>> \in Edges(graph)}`.
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

/// Fusion mode for a compiled `Children`.
#[derive(Clone, Copy)]
enum Mode {
    /// No VM fusion at all — the plain SetFilter loop over the domain.
    Baseline,
    /// The projection hoist (still iterates the domain).
    Hoist,
    /// The whole-comprehension `EdgeFilter` range-scan.
    Edge,
}

fn compile_filter(domain: Value, graph: Value, outer: Value, mode: Mode) -> (BytecodeChunk, u16) {
    let filter = filter_expr(domain);
    let callees = projection_callees();

    let mut compiler = BytecodeCompiler::new();
    compiler.enable_tuple2_set_in();
    compiler.enable_reg_recycling();
    match mode {
        Mode::Baseline => {}
        Mode::Hoist => compiler.enable_set_filter_projection_hoist(),
        Mode::Edge => {
            compiler.enable_set_filter_projection_hoist();
            compiler.enable_edge_filter();
        }
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

fn chunk_has_edge_filter(chunk: &BytecodeChunk) -> bool {
    (0..chunk.function_count()).any(|i| {
        chunk
            .get_function(i as u16)
            .instructions
            .iter()
            .any(|op| matches!(op, Opcode::EdgeFilter { .. }))
    })
}

fn execute(domain: Value, graph: Value, outer: Value, mode: Mode) -> Result<Value, String> {
    let (chunk, main_idx) = compile_filter(domain, graph, outer, mode);
    let result = BytecodeVm::new(&chunk, &[], None)
        .execute_function(main_idx)
        .map_err(|err| err.to_string());
    result
}

/// Assert the `EdgeFilter` path is byte-identical to the unfused baseline (and
/// the hoisted path) for the given inputs.
fn assert_edge_matches(domain: Value, graph: Value, outer: Value) {
    let baseline = execute(domain.clone(), graph.clone(), outer.clone(), Mode::Baseline);
    let hoisted = execute(domain.clone(), graph.clone(), outer.clone(), Mode::Hoist);
    let edge = execute(domain, graph, outer, Mode::Edge);
    assert_eq!(edge, baseline, "EdgeFilter diverged from unfused baseline");
    assert_eq!(edge, hoisted, "EdgeFilter diverged from projection hoist");
}

#[test]
fn edge_filter_is_emitted_for_the_children_shape() {
    let (chunk, _) = compile_filter(
        Value::set([Value::SmallInt(1)]),
        Value::tuple([Value::empty_set(), Value::empty_set()]),
        Value::SmallInt(9),
        Mode::Edge,
    );
    assert!(
        chunk_has_edge_filter(&chunk),
        "EdgeFilter must be emitted for {{c \\in D : <<v,c>> \\in Edges(graph)}}"
    );
    // The baseline and hoist paths must NOT contain the opcode.
    let (baseline, _) = compile_filter(
        Value::set([Value::SmallInt(1)]),
        Value::tuple([Value::empty_set(), Value::empty_set()]),
        Value::SmallInt(9),
        Mode::Baseline,
    );
    assert!(!chunk_has_edge_filter(&baseline));
    let (hoist, _) = compile_filter(
        Value::set([Value::SmallInt(1)]),
        Value::tuple([Value::empty_set(), Value::empty_set()]),
        Value::SmallInt(9),
        Mode::Hoist,
    );
    assert!(!chunk_has_edge_filter(&hoist));
}

#[test]
fn edge_filter_matches_call_path() {
    let outer = Value::SmallInt(9);
    let edges = Value::set([
        Value::tuple([outer.clone(), Value::SmallInt(1)]),
        Value::tuple([outer.clone(), Value::SmallInt(3)]),
        Value::tuple([Value::SmallInt(8), Value::SmallInt(2)]),
    ]);
    let graph = Value::tuple([Value::empty_set(), edges]);
    let domain = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);

    let edge = execute(domain.clone(), graph.clone(), outer.clone(), Mode::Edge);
    assert_eq!(
        edge.clone().unwrap(),
        Value::set([Value::SmallInt(1), Value::SmallInt(3)])
    );
    assert_edge_matches(domain, graph, outer);
}

#[test]
fn edge_filter_excludes_second_component_outside_domain() {
    // Edge `<<outer, 5>>` present but `5 \notin domain` must be dropped, exactly
    // as `Children == {c \in vs : <<v,c>> \in es}`.
    let outer = Value::SmallInt(9);
    let edges = Value::set([
        Value::tuple([outer.clone(), Value::SmallInt(1)]),
        Value::tuple([outer.clone(), Value::SmallInt(5)]), // 5 not in domain
    ]);
    let graph = Value::tuple([Value::empty_set(), edges]);
    let domain = Value::set([Value::SmallInt(1), Value::SmallInt(2)]);

    let edge = execute(domain.clone(), graph.clone(), outer.clone(), Mode::Edge);
    assert_eq!(edge.unwrap(), Value::set([Value::SmallInt(1)]));
    assert_edge_matches(domain, graph, outer);
}

#[test]
fn edge_filter_matches_vertex_tuple_edges() {
    // Edges are 2-tuples of vertices `<<node, round>>`, mirroring the real
    // dag-consensus DAG the opcode targets.
    let vertex = |n: i64, r: i64| Value::tuple([Value::SmallInt(n), Value::SmallInt(r)]);
    let outer = vertex(2, 3);
    let edges = Value::set([
        Value::tuple([outer.clone(), vertex(1, 2)]),
        Value::tuple([outer.clone(), vertex(3, 2)]),
        Value::tuple([vertex(1, 3), vertex(1, 2)]), // different first vertex
    ]);
    let graph = Value::tuple([Value::empty_set(), edges]);
    let domain = Value::set([vertex(1, 2), vertex(3, 2), vertex(9, 9)]);

    let edge = execute(domain.clone(), graph.clone(), outer.clone(), Mode::Edge);
    assert_eq!(
        edge.unwrap(),
        Value::set([vertex(1, 2), vertex(3, 2)]),
        "children are the second components of outer's edges that lie in domain"
    );
    assert_edge_matches(domain, graph, outer);
}

#[test]
fn edge_filter_preserves_empty_and_error_order() {
    let invalid_graph = Value::SmallInt(7);
    let outer = Value::SmallInt(9);

    // Empty domain must return {} WITHOUT evaluating graph[2] (which would
    // fail): matches the SetFilterBegin preheader skip.
    let empty = execute(
        Value::empty_set(),
        invalid_graph.clone(),
        outer.clone(),
        Mode::Edge,
    );
    assert_eq!(empty.clone().unwrap(), Value::empty_set());
    assert_edge_matches(Value::empty_set(), invalid_graph.clone(), outer.clone());

    // An invalid (non-set) domain fails before the also-invalid projection.
    let invalid_domain = Value::SmallInt(11);
    let domain_err = execute(
        invalid_domain.clone(),
        invalid_graph.clone(),
        outer.clone(),
        Mode::Edge,
    );
    assert!(domain_err.unwrap_err().contains("set filter"));
    assert_edge_matches(invalid_domain, invalid_graph.clone(), outer.clone());

    // Non-empty valid domain surfaces the projection error.
    let nonempty = Value::set([Value::SmallInt(1)]);
    let proj_err = execute(
        nonempty.clone(),
        invalid_graph.clone(),
        outer.clone(),
        Mode::Edge,
    );
    assert!(proj_err.is_err());
    assert_edge_matches(nonempty, invalid_graph, outer);
}

#[test]
fn edge_filter_preserves_non_set_membership_error() {
    // graph[2] resolves to a non-set: the membership test error must match.
    let outer = Value::SmallInt(9);
    let graph = Value::tuple([Value::empty_set(), Value::SmallInt(42)]);
    let domain = Value::set([Value::SmallInt(1)]);

    let edge = execute(domain.clone(), graph.clone(), outer.clone(), Mode::Edge);
    assert!(edge
        .unwrap_err()
        .contains("expected enumerable set for \\in"));
    assert_edge_matches(domain, graph, outer);
}

#[test]
fn edge_filter_matches_interval_domain() {
    // A non-`Value::Set` enumerable domain (an integer interval) takes the naive
    // fallback and must still match the baseline.
    let outer = Value::SmallInt(9);
    let edges = Value::set([
        Value::tuple([outer.clone(), Value::SmallInt(1)]),
        Value::tuple([outer.clone(), Value::SmallInt(4)]),
    ]);
    let graph = Value::tuple([Value::empty_set(), edges]);
    let domain = Value::Interval(Rp::new(IntervalValue::new(1.into(), 3.into()))); // 1..3, excludes 4
    let edge = execute(domain.clone(), graph.clone(), outer.clone(), Mode::Edge);
    assert_eq!(edge.unwrap(), Value::set([Value::SmallInt(1)]));
    assert_edge_matches(domain, graph, outer);
}
