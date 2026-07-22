// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tla_core::ast::Expr;

fn int(n: i64) -> Spanned<TirExpr> {
    spanned(TirExpr::Const {
        value: Value::SmallInt(n),
        ty: TirType::Int,
    })
}

fn name(name: &str) -> Spanned<TirExpr> {
    spanned(TirExpr::Name(ident_name(name)))
}

fn except(rhs: Spanned<TirExpr>, depth: usize) -> Spanned<TirExpr> {
    assert!(depth > 0);
    let mut base = int(10);
    for _ in 0..depth {
        base = spanned(TirExpr::Tuple(vec![base]));
    }
    spanned(TirExpr::Except {
        base: Box::new(base),
        specs: vec![TirExceptSpec {
            path: (0..depth)
                .map(|_| TirExceptPathElement::Index(Box::new(int(1))))
                .collect(),
            value: rhs,
        }],
    })
}

fn compile(
    mut compiler: BytecodeCompiler,
    rhs: Spanned<TirExpr>,
    params: &[String],
    callees: &HashMap<String, CalleeInfo>,
) -> Vec<Opcode> {
    let idx = compiler
        .compile_expression_with_callees("Main", params, &except(rhs, 1), callees)
        .expect("EXCEPT test expression should compile");
    compiler.finish().get_function(idx).instructions.clone()
}

fn apply_count(instructions: &[Opcode]) -> usize {
    instructions
        .iter()
        .filter(|opcode| matches!(opcode, Opcode::FuncApply { .. }))
        .count()
}

fn enabled_compiler() -> BytecodeCompiler {
    let mut compiler = BytecodeCompiler::new();
    compiler.enable_except_at_free_rhs();
    compiler
}

#[test]
fn except_at_free_is_opt_in_and_omits_only_the_at_apply() {
    let disabled = compile(BytecodeCompiler::new(), int(99), &[], &HashMap::new());
    let enabled = compile(enabled_compiler(), int(99), &[], &HashMap::new());

    assert_eq!(
        apply_count(&disabled),
        1,
        "default lowering must materialize @"
    );
    assert_eq!(
        apply_count(&enabled),
        0,
        "opt-in may omit the unused @ apply"
    );
    assert_eq!(
        enabled
            .iter()
            .filter(|opcode| matches!(opcode, Opcode::FuncExcept { .. }))
            .count(),
        1,
        "the update itself must remain present"
    );
}

#[test]
fn except_at_free_accepts_only_total_existing_value_names() {
    let bound = compile(
        enabled_compiler(),
        name("replacement"),
        &["replacement".to_string()],
        &HashMap::new(),
    );
    assert_eq!(apply_count(&bound), 0, "bound register should be certified");

    let mut explicit_state_compiler = enabled_compiler();
    explicit_state_compiler.set_state_vars(HashMap::from([("ignored".to_string(), 3)]));
    let explicit_state = compile(
        explicit_state_compiler,
        spanned(TirExpr::Name(TirNameRef {
            name: "state".to_string(),
            name_id: tla_core::NameId::INVALID,
            kind: TirNameKind::StateVar { index: 3 },
            ty: TirType::Dyn,
        })),
        &[],
        &HashMap::new(),
    );
    assert_eq!(apply_count(&explicit_state), 0);
    assert!(explicit_state
        .iter()
        .any(|opcode| matches!(opcode, Opcode::LoadVar { var_idx: 3, .. })));

    let mut unresolved_state_compiler = enabled_compiler();
    unresolved_state_compiler.set_state_vars(HashMap::from([("state".to_string(), 4)]));
    let unresolved_state = compile(
        unresolved_state_compiler,
        name("state"),
        &[],
        &HashMap::new(),
    );
    assert_eq!(apply_count(&unresolved_state), 0);
    assert!(unresolved_state
        .iter()
        .any(|opcode| matches!(opcode, Opcode::LoadVar { var_idx: 4, .. })));

    let constant_id = intern_name("ExceptAtFreeConstant");
    let mut constant_compiler = BytecodeCompiler::with_resolved_constants(HashMap::from([(
        constant_id,
        Value::SmallInt(77),
    )]));
    constant_compiler.enable_except_at_free_rhs();
    let resolved_constant = compile(
        constant_compiler,
        spanned(TirExpr::Name(TirNameRef {
            name: "ExceptAtFreeConstant".to_string(),
            name_id: constant_id,
            kind: TirNameKind::Ident,
            ty: TirType::Dyn,
        })),
        &[],
        &HashMap::new(),
    );
    assert_eq!(
        apply_count(&resolved_constant),
        0,
        "resolved constant should be certified"
    );
}

#[test]
fn except_at_free_captures_pure_index_mpmc_bound_rhs_shapes() {
    for rhs_name in ["to", "seq", "value", "next", "pc", "claimed", "read"] {
        let instructions = compile(
            enabled_compiler(),
            name(rhs_name),
            &[rhs_name.to_string()],
            &HashMap::new(),
        );
        assert_eq!(
            apply_count(&instructions),
            0,
            "pure-Index MPMC-style bound RHS '{rhs_name}' should omit @"
        );
    }
}

#[test]
fn except_at_free_preserves_every_navigation_apply() {
    for depth in 1..=4 {
        let mut compiler = enabled_compiler();
        let idx = compiler
            .compile_expression("Main", &except(int(99), depth))
            .expect("nested EXCEPT should compile");
        let chunk = compiler.finish();
        let instructions = &chunk.get_function(idx).instructions;
        assert_eq!(
            apply_count(instructions),
            depth - 1,
            "a depth-{depth} path must retain exactly its N-1 navigation applies: {instructions:?}"
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|opcode| matches!(opcode, Opcode::FuncExcept { .. }))
                .count(),
            depth,
            "a depth-{depth} update must retain every nested update"
        );
    }
}

#[test]
fn except_at_free_refuses_calls_apply_let_closure_external_and_arithmetic() {
    let arithmetic = spanned(TirExpr::ArithBinOp {
        left: Box::new(int(1)),
        op: TirArithOp::Add,
        right: Box::new(int(2)),
    });
    assert_eq!(
        apply_count(&compile(
            enabled_compiler(),
            arithmetic,
            &[],
            &HashMap::new()
        )),
        1,
        "arithmetic must retain the @ materialization"
    );

    let zero_callees = HashMap::from([(
        "ZeroArg".to_string(),
        CalleeInfo {
            params: vec![],
            body: Arc::new(spanned(TirExpr::Tuple(vec![int(1)]))),
            ast_body: None,
        },
    )]);
    let call = compile(enabled_compiler(), name("ZeroArg"), &[], &zero_callees);
    assert_eq!(apply_count(&call), 1, "Call RHS must be refused");
    assert!(call
        .iter()
        .any(|opcode| matches!(opcode, Opcode::Call { .. })));

    let apply_callees = HashMap::from([(
        "Identity".to_string(),
        CalleeInfo {
            params: vec!["x".to_string()],
            body: Arc::new(name("x")),
            ast_body: None,
        },
    )]);
    let apply = compile(
        enabled_compiler(),
        spanned(TirExpr::Apply {
            op: Box::new(name("Identity")),
            args: vec![int(5)],
        }),
        &[],
        &apply_callees,
    );
    assert_eq!(apply_count(&apply), 1, "Apply RHS must be refused");
    assert!(apply
        .iter()
        .any(|opcode| matches!(opcode, Opcode::Call { .. })));

    let let_rhs = spanned(TirExpr::Let {
        defs: vec![TirLetDef {
            name: "Local".to_string(),
            name_id: tla_core::NameId::INVALID,
            params: vec![],
            body: int(8),
        }],
        body: Box::new(name("Local")),
    });
    assert_eq!(
        apply_count(&compile(enabled_compiler(), let_rhs, &[], &HashMap::new())),
        1,
        "LET RHS must be refused"
    );

    assert_eq!(
        apply_count(&compile(
            enabled_compiler(),
            spanned(TirExpr::OpRef("+".to_string())),
            &[],
            &HashMap::new()
        )),
        1,
        "closure-producing RHS must be refused"
    );

    let mut external_compiler = enabled_compiler();
    external_compiler.set_force_external_ops(HashSet::from(["External".to_string()]));
    let external = compile(external_compiler, name("External"), &[], &HashMap::new());
    assert_eq!(
        apply_count(&external),
        1,
        "external callback RHS must be refused"
    );
    assert!(external
        .iter()
        .any(|opcode| matches!(opcode, Opcode::CallExternal { .. })));
}

#[test]
fn except_at_free_refuses_function_application_even_when_it_is_at_free() {
    let rhs = spanned(TirExpr::FuncApply {
        func: Box::new(spanned(TirExpr::Record(vec![(field_name("x"), int(4))]))),
        arg: Box::new(spanned(TirExpr::Const {
            value: Value::string("x"),
            ty: TirType::Str,
        })),
    });
    let instructions = compile(enabled_compiler(), rhs, &[], &HashMap::new());
    assert_eq!(
        apply_count(&instructions),
        2,
        "one apply must materialize @ and one must evaluate the RHS: {instructions:?}"
    );
}

#[test]
fn except_at_free_refuses_final_field_paths() {
    let expr = spanned(TirExpr::Except {
        base: Box::new(spanned(TirExpr::Record(vec![(field_name("x"), int(1))]))),
        specs: vec![TirExceptSpec {
            path: vec![TirExceptPathElement::Field(field_name("x"))],
            value: int(9),
        }],
    });
    let mut compiler = enabled_compiler();
    let idx = compiler
        .compile_expression("Main", &expr)
        .expect("record-field EXCEPT should compile");
    let chunk = compiler.finish();
    let instructions = &chunk.get_function(idx).instructions;
    assert_eq!(
        apply_count(instructions),
        1,
        "Field-vs-string-Index semantics require retaining @: {instructions:?}"
    );
}

#[test]
fn except_at_free_refuses_any_earlier_field_path_element() {
    let expr = spanned(TirExpr::Except {
        base: Box::new(spanned(TirExpr::Record(vec![(
            field_name("slots"),
            spanned(TirExpr::Tuple(vec![int(1)])),
        )]))),
        specs: vec![TirExceptSpec {
            path: vec![
                TirExceptPathElement::Field(field_name("slots")),
                TirExceptPathElement::Index(Box::new(int(1))),
            ],
            value: int(9),
        }],
    });
    let mut compiler = enabled_compiler();
    let idx = compiler
        .compile_expression("Main", &expr)
        .expect("mixed Field/Index EXCEPT should compile");
    let chunk = compiler.finish();
    let instructions = &chunk.get_function(idx).instructions;
    assert_eq!(
        apply_count(instructions),
        2,
        "mixed paths must retain navigation and @ applies: {instructions:?}"
    );
}

#[test]
fn except_at_free_refuses_parameterized_callee_values_and_replaced_names() {
    let parameterized_callees = HashMap::from([(
        "ParamOp".to_string(),
        CalleeInfo {
            params: vec!["x".to_string()],
            body: Arc::new(name("x")),
            ast_body: Some(PreservedAstBody(Arc::new(Spanned::dummy(Expr::Ident(
                "x".to_string(),
                tla_core::NameId::INVALID,
            ))))),
        },
    )]);
    let closure_value = compile(
        enabled_compiler(),
        name("ParamOp"),
        &[],
        &parameterized_callees,
    );
    assert_eq!(
        apply_count(&closure_value),
        1,
        "bare parameterized callee must retain @"
    );

    let replacement_callees = HashMap::from([(
        "ReplacementOp".to_string(),
        CalleeInfo {
            params: vec![],
            body: Arc::new(spanned(TirExpr::Tuple(vec![int(1)]))),
            ast_body: None,
        },
    )]);
    let mut replaced = enabled_compiler();
    replaced.set_state_vars(HashMap::from([("state".to_string(), 0)]));
    replaced.set_op_replacements(HashMap::from([(
        "state".to_string(),
        "ReplacementOp".to_string(),
    )]));
    let replaced_name = compile(replaced, name("state"), &[], &replacement_callees);
    assert_eq!(
        apply_count(&replaced_name),
        1,
        "operator replacement must win over unresolved-state certification"
    );
    assert!(replaced_name
        .iter()
        .any(|opcode| matches!(opcode, Opcode::Call { .. })));
}

#[test]
fn except_at_free_resolved_constant_precedes_external_name_resolution() {
    let constant_id = intern_name("ConstantBeforeExternal");
    let mut compiler = BytecodeCompiler::with_resolved_constants(HashMap::from([(
        constant_id,
        Value::SmallInt(12),
    )]));
    compiler.enable_except_at_free_rhs();
    compiler.set_force_external_ops(HashSet::from(["ConstantBeforeExternal".to_string()]));
    let instructions = compile(
        compiler,
        spanned(TirExpr::Name(TirNameRef {
            name: "ConstantBeforeExternal".to_string(),
            name_id: constant_id,
            kind: TirNameKind::Ident,
            ty: TirType::Dyn,
        })),
        &[],
        &HashMap::new(),
    );
    assert_eq!(apply_count(&instructions), 0);
    assert!(
        !instructions
            .iter()
            .any(|opcode| matches!(opcode, Opcode::CallExternal { .. })),
        "resolved constants must retain compile_name_expr precedence"
    );
}
