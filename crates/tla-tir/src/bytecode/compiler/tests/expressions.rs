// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[test]
fn test_compile_constant_bool() {
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Const {
        value: Value::Bool(true),
        ty: TirType::Bool,
    });
    let idx = compiler.compile_expression("test", &expr).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // Should be: LoadBool r0 true, Ret r0
    assert_eq!(func.instructions.len(), 2);
    assert!(matches!(
        func.instructions[0],
        Opcode::LoadBool { rd: 0, value: true }
    ));
    assert!(matches!(func.instructions[1], Opcode::Ret { rs: 0 }));
}

#[test]
fn test_compile_constant_int() {
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Const {
        value: Value::SmallInt(42),
        ty: TirType::Int,
    });
    let idx = compiler.compile_expression("test", &expr).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert_eq!(func.instructions.len(), 2);
    assert!(matches!(
        func.instructions[0],
        Opcode::LoadImm { rd: 0, value: 42 }
    ));
}

#[test]
fn test_compile_arithmetic() {
    let mut compiler = BytecodeCompiler::new();
    // 1 + 2
    let expr = spanned(TirExpr::ArithBinOp {
        left: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(1),
            ty: TirType::Int,
        })),
        op: TirArithOp::Add,
        right: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(2),
            ty: TirType::Int,
        })),
    });
    let idx = compiler.compile_expression("test", &expr).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // LoadImm r0, 1
    // LoadImm r1, 2
    // AddInt r2, r0, r1
    // Ret r2 -> Move r0, r2; Ret r0
    assert!(func.instructions.len() >= 4);
    assert!(matches!(
        func.instructions[2],
        Opcode::AddInt {
            rd: 2,
            r1: 0,
            r2: 1
        }
    ));
}

#[test]
fn test_compile_state_variable() {
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Name(TirNameRef {
        name: "x".to_string(),
        name_id: tla_core::NameId(0),
        kind: TirNameKind::StateVar { index: 3 },
        ty: TirType::Int,
    }));
    let idx = compiler.compile_expression("test", &expr).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(matches!(
        func.instructions[0],
        Opcode::LoadVar { rd: 0, var_idx: 3 }
    ));
}

#[test]
fn test_compile_set_enum() {
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::SetEnum(vec![
        spanned(TirExpr::Const {
            value: Value::SmallInt(1),
            ty: TirType::Int,
        }),
        spanned(TirExpr::Const {
            value: Value::SmallInt(2),
            ty: TirType::Int,
        }),
    ]));
    let idx = compiler.compile_expression("test", &expr).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // LoadImm r0, 1; LoadImm r1, 2; SetEnum r2, start=0, count=2; Move; Ret
    let has_set_enum = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::SetEnum { count: 2, .. }));
    assert!(has_set_enum);
}

fn tuple_membership_expr(arity: i64) -> Spanned<TirExpr> {
    let tuple = (0..arity)
        .map(|value| {
            spanned(TirExpr::Const {
                value: Value::SmallInt(value),
                ty: TirType::Int,
            })
        })
        .collect();
    spanned(TirExpr::In {
        elem: Box::new(spanned(TirExpr::Tuple(tuple))),
        set: Box::new(spanned(TirExpr::Const {
            value: Value::set([Value::tuple(
                (0..arity).map(Value::SmallInt).collect::<Vec<_>>(),
            )]),
            ty: TirType::Set(Box::new(TirType::Dyn)),
        })),
    })
}

#[test]
fn test_compile_tuple2_set_in_fusion_is_opt_in() {
    let expr = tuple_membership_expr(2);

    let mut compiler = BytecodeCompiler::new();
    let idx = compiler.compile_expression("plain", &expr).unwrap();
    let chunk = compiler.finish();
    let instructions = &chunk.get_function(idx).instructions;
    assert!(instructions
        .iter()
        .any(|op| matches!(op, Opcode::TupleNew { count: 2, .. })));
    assert!(instructions
        .iter()
        .any(|op| matches!(op, Opcode::SetIn { .. })));
    assert!(!instructions
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SetIn { .. })));
}

#[test]
fn test_compile_tuple2_set_in_fuses_only_exact_arity() {
    let mut compiler = BytecodeCompiler::new();
    compiler.enable_tuple2_set_in();
    let fused_idx = compiler
        .compile_expression("tuple2", &tuple_membership_expr(2))
        .unwrap();
    let tuple1_idx = compiler
        .compile_expression("tuple1", &tuple_membership_expr(1))
        .unwrap();
    let tuple3_idx = compiler
        .compile_expression("tuple3", &tuple_membership_expr(3))
        .unwrap();
    let chunk = compiler.finish();

    let fused = &chunk.get_function(fused_idx).instructions;
    assert!(fused
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SetIn { .. })));
    assert!(!fused
        .iter()
        .any(|op| matches!(op, Opcode::TupleNew { .. } | Opcode::SetIn { .. })));

    for idx in [tuple1_idx, tuple3_idx] {
        let instructions = &chunk.get_function(idx).instructions;
        assert!(instructions
            .iter()
            .any(|op| matches!(op, Opcode::TupleNew { .. })));
        assert!(instructions
            .iter()
            .any(|op| matches!(op, Opcode::SetIn { .. })));
        assert!(!instructions
            .iter()
            .any(|op| matches!(op, Opcode::Tuple2SetIn { .. })));
    }
}

fn set_enum_subseteq_expr() -> Spanned<TirExpr> {
    spanned(TirExpr::Subseteq {
        left: Box::new(spanned(TirExpr::SetEnum(vec![
            spanned(TirExpr::Const {
                value: Value::SmallInt(1),
                ty: TirType::Int,
            }),
            spanned(TirExpr::Const {
                value: Value::SmallInt(2),
                ty: TirType::Int,
            }),
        ]))),
        right: Box::new(spanned(TirExpr::Name(TirNameRef {
            name: "s".to_string(),
            name_id: tla_core::NameId(0),
            kind: TirNameKind::StateVar { index: 0 },
            ty: TirType::Set(Box::new(TirType::Int)),
        }))),
    })
}

#[test]
fn test_compile_set_enum_subseteq_fusion_is_opt_in() {
    let expr = set_enum_subseteq_expr();

    let mut plain = BytecodeCompiler::new();
    let plain_idx = plain.compile_expression("plain", &expr).unwrap();
    let plain_chunk = plain.finish();
    let plain_ops = &plain_chunk.get_function(plain_idx).instructions;
    assert!(plain_ops
        .iter()
        .any(|op| matches!(op, Opcode::SetEnum { count: 2, .. })));
    assert!(plain_ops
        .iter()
        .any(|op| matches!(op, Opcode::Subseteq { .. })));
    assert!(!plain_ops
        .iter()
        .any(|op| matches!(op, Opcode::SetEnumSubseteq { .. })));

    let mut fused = BytecodeCompiler::new();
    fused.enable_set_enum_subseteq();
    let fused_idx = fused.compile_expression("fused", &expr).unwrap();
    let fused_chunk = fused.finish();
    let fused_ops = &fused_chunk.get_function(fused_idx).instructions;
    assert!(fused_ops
        .iter()
        .any(|op| matches!(op, Opcode::SetEnumSubseteq { count: 2, .. })));
    assert!(!fused_ops
        .iter()
        .any(|op| matches!(op, Opcode::SetEnum { .. } | Opcode::Subseteq { .. })));
}

#[test]
fn test_compile_set_enum_subseteq_declines_non_enum_lhs() {
    let expr = spanned(TirExpr::Subseteq {
        left: Box::new(spanned(TirExpr::Name(TirNameRef {
            name: "left".to_string(),
            name_id: tla_core::NameId(1),
            kind: TirNameKind::StateVar { index: 1 },
            ty: TirType::Set(Box::new(TirType::Int)),
        }))),
        right: Box::new(spanned(TirExpr::Name(TirNameRef {
            name: "right".to_string(),
            name_id: tla_core::NameId(0),
            kind: TirNameKind::StateVar { index: 0 },
            ty: TirType::Set(Box::new(TirType::Int)),
        }))),
    });

    let mut compiler = BytecodeCompiler::new();
    compiler.enable_set_enum_subseteq();
    let idx = compiler.compile_expression("non-enum", &expr).unwrap();
    let chunk = compiler.finish();
    let ops = &chunk.get_function(idx).instructions;
    assert!(ops.iter().any(|op| matches!(op, Opcode::Subseteq { .. })));
    assert!(!ops
        .iter()
        .any(|op| matches!(op, Opcode::SetEnumSubseteq { .. })));
}

#[test]
fn test_compile_set_enum_subseteq_declines_large_enum() {
    let expr = spanned(TirExpr::Subseteq {
        left: Box::new(spanned(TirExpr::SetEnum(vec![
            spanned(TirExpr::Const {
                value: Value::SmallInt(1),
                ty: TirType::Int,
            }),
            spanned(TirExpr::Const {
                value: Value::SmallInt(2),
                ty: TirType::Int,
            }),
            spanned(TirExpr::Const {
                value: Value::SmallInt(3),
                ty: TirType::Int,
            }),
        ]))),
        right: Box::new(spanned(TirExpr::Name(TirNameRef {
            name: "s".to_string(),
            name_id: tla_core::NameId(0),
            kind: TirNameKind::StateVar { index: 0 },
            ty: TirType::Set(Box::new(TirType::Int)),
        }))),
    });

    let mut compiler = BytecodeCompiler::new();
    compiler.enable_set_enum_subseteq();
    let idx = compiler.compile_expression("large-enum", &expr).unwrap();
    let chunk = compiler.finish();
    let ops = &chunk.get_function(idx).instructions;
    assert!(ops
        .iter()
        .any(|op| matches!(op, Opcode::SetEnum { count: 3, .. })));
    assert!(ops.iter().any(|op| matches!(op, Opcode::Subseteq { .. })));
    assert!(!ops
        .iter()
        .any(|op| matches!(op, Opcode::SetEnumSubseteq { .. })));
}

fn tuple2_self_name(name: &str, name_id: tla_core::NameId) -> Spanned<TirExpr> {
    spanned(TirExpr::Name(TirNameRef {
        name: name.to_string(),
        name_id,
        kind: TirNameKind::Ident,
        ty: TirType::Dyn,
    }))
}

fn tuple2_self_state_name(name: &str, name_id: tla_core::NameId) -> Spanned<TirExpr> {
    spanned(TirExpr::Name(TirNameRef {
        name: name.to_string(),
        name_id,
        kind: TirNameKind::StateVar { index: 0 },
        ty: TirType::Dyn,
    }))
}

fn tuple2_self_projection(base: Spanned<TirExpr>, index: i64) -> Spanned<TirExpr> {
    spanned(TirExpr::FuncApply {
        func: Box::new(base),
        arg: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(index),
            ty: TirType::Int,
        })),
    })
}

fn tuple2_self_cmp(
    left: Spanned<TirExpr>,
    tuple_elements: Vec<Spanned<TirExpr>>,
) -> Spanned<TirExpr> {
    spanned(TirExpr::Cmp {
        left: Box::new(left),
        op: TirCmpOp::Eq,
        right: Box::new(spanned(TirExpr::Tuple(tuple_elements))),
    })
}

fn tuple2_self_bound_var(name: &str, name_id: tla_core::NameId) -> TirBoundVar {
    TirBoundVar {
        name: name.to_string(),
        name_id,
        domain: Some(Box::new(spanned(TirExpr::Const {
            value: Value::set([Value::tuple([Value::SmallInt(1), Value::SmallInt(2)])]),
            ty: TirType::Set(Box::new(TirType::Dyn)),
        }))),
        pattern: None,
    }
}

fn compile_tuple2_self_body(
    body: Spanned<TirExpr>,
    bind_second_name: bool,
    enable_fusion: bool,
) -> Vec<Opcode> {
    let e_id = tla_core::NameId(41);
    let mut vars = vec![tuple2_self_bound_var("e", e_id)];
    if bind_second_name {
        vars.push(tuple2_self_bound_var("f", tla_core::NameId(42)));
    }
    let expr = spanned(TirExpr::Forall {
        vars,
        body: Box::new(body),
    });
    let mut compiler = BytecodeCompiler::new();
    if enable_fusion {
        compiler.enable_tuple2_self_eq();
    }
    let idx = compiler
        .compile_expression("tuple2-self-eq", &expr)
        .expect("tuple self-equality expression should compile");
    compiler.finish().get_function(idx).instructions.clone()
}

fn exact_tuple2_self_body() -> Spanned<TirExpr> {
    exact_tuple2_self_body_with_id(tla_core::NameId(41))
}

fn exact_tuple2_self_body_with_id(e_id: tla_core::NameId) -> Spanned<TirExpr> {
    tuple2_self_cmp(
        tuple2_self_name("e", e_id),
        vec![
            tuple2_self_projection(tuple2_self_name("e", e_id), 1),
            tuple2_self_projection(tuple2_self_name("e", e_id), 2),
        ],
    )
}

#[test]
fn test_compile_tuple2_self_eq_fusion_is_exact_and_opt_in() {
    let plain = compile_tuple2_self_body(exact_tuple2_self_body(), false, false);
    assert!(plain.iter().any(|op| matches!(op, Opcode::Eq { .. })));
    assert!(plain
        .iter()
        .any(|op| matches!(op, Opcode::TupleNew { count: 2, .. })));
    assert_eq!(
        plain
            .iter()
            .filter(|op| matches!(op, Opcode::FuncApply { .. }))
            .count(),
        2
    );
    assert!(!plain
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfEq { .. })));

    let fused = compile_tuple2_self_body(exact_tuple2_self_body(), false, true);
    assert!(fused
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfEq { .. })));
    assert!(!fused.iter().any(|op| matches!(
        op,
        Opcode::Eq { .. } | Opcode::TupleNew { .. } | Opcode::FuncApply { .. }
    )));

    // Raw production TIR retains quantifier-body references as Ident with an
    // invalid NameId. Lexical register resolution is still exact by name.
    let production = compile_tuple2_self_body(
        exact_tuple2_self_body_with_id(tla_core::NameId::INVALID),
        false,
        true,
    );
    assert!(production
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfEq { .. })));
}

#[test]
fn test_compile_tuple2_self_eq_declines_identity_index_and_arity_variants() {
    let e_id = tla_core::NameId(41);
    let f_id = tla_core::NameId(42);
    let cases = [
        (
            tuple2_self_cmp(
                tuple2_self_name("e", e_id),
                vec![
                    tuple2_self_projection(tuple2_self_name("e", e_id), 1),
                    tuple2_self_projection(tuple2_self_name("f", f_id), 2),
                ],
            ),
            true,
        ),
        (
            tuple2_self_cmp(
                tuple2_self_name("e", e_id),
                vec![
                    tuple2_self_projection(tuple2_self_name("e", e_id), 2),
                    tuple2_self_projection(tuple2_self_name("e", e_id), 1),
                ],
            ),
            false,
        ),
        (
            tuple2_self_cmp(
                tuple2_self_name("e", e_id),
                vec![tuple2_self_projection(tuple2_self_name("e", e_id), 1)],
            ),
            false,
        ),
        (
            tuple2_self_cmp(
                tuple2_self_name("e", e_id),
                vec![
                    tuple2_self_projection(tuple2_self_name("e", e_id), 1),
                    tuple2_self_projection(tuple2_self_name("e", e_id), 2),
                    tuple2_self_projection(tuple2_self_name("e", e_id), 3),
                ],
            ),
            false,
        ),
        (
            tuple2_self_cmp(
                tuple2_self_name("e", e_id),
                vec![
                    tuple2_self_projection(tuple2_self_name("e", e_id), 1),
                    tuple2_self_projection(tuple2_self_name("e", f_id), 2),
                ],
            ),
            false,
        ),
        (
            tuple2_self_cmp(
                tuple2_self_name("e", e_id),
                vec![
                    tuple2_self_projection(tuple2_self_name("e", tla_core::NameId::INVALID), 1),
                    tuple2_self_projection(tuple2_self_name("e", tla_core::NameId::INVALID), 2),
                ],
            ),
            false,
        ),
    ];

    for (body, bind_second_name) in cases {
        let instructions = compile_tuple2_self_body(body, bind_second_name, true);
        assert!(!instructions
            .iter()
            .any(|op| matches!(op, Opcode::Tuple2SelfEq { .. })));
    }
}

#[test]
fn test_compile_tuple2_self_eq_declines_reversed_and_state_var_shapes() {
    let e_id = tla_core::NameId(41);
    let reversed = spanned(TirExpr::Cmp {
        left: Box::new(spanned(TirExpr::Tuple(vec![
            tuple2_self_projection(tuple2_self_name("e", e_id), 1),
            tuple2_self_projection(tuple2_self_name("e", e_id), 2),
        ]))),
        op: TirCmpOp::Eq,
        right: Box::new(tuple2_self_name("e", e_id)),
    });
    let reversed = compile_tuple2_self_body(reversed, false, true);
    assert!(!reversed
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfEq { .. })));

    let state = tuple2_self_cmp(
        tuple2_self_state_name("e", e_id),
        vec![
            tuple2_self_projection(tuple2_self_state_name("e", e_id), 1),
            tuple2_self_projection(tuple2_self_state_name("e", e_id), 2),
        ],
    );
    let mut compiler = BytecodeCompiler::new();
    compiler.enable_tuple2_self_eq();
    let idx = compiler
        .compile_expression("state-tuple2-self-eq", &state)
        .expect("state-variable shape should compile normally");
    let chunk = compiler.finish();
    assert!(!chunk
        .get_function(idx)
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfEq { .. })));
}

fn tuple2_self_subseteq(
    elements: Vec<Spanned<TirExpr>>,
    set: Spanned<TirExpr>,
) -> Spanned<TirExpr> {
    spanned(TirExpr::Subseteq {
        left: Box::new(spanned(TirExpr::SetEnum(elements))),
        right: Box::new(set),
    })
}

fn tuple2_self_and(left: Spanned<TirExpr>, right: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::BoolBinOp {
        left: Box::new(left),
        op: TirBoolOp::And,
        right: Box::new(right),
    })
}

fn tuple2_self_subseteq_state_name(
    name: &str,
    name_id: tla_core::NameId,
    index: u16,
) -> Spanned<TirExpr> {
    spanned(TirExpr::Name(TirNameRef {
        name: name.to_string(),
        name_id,
        kind: TirNameKind::StateVar { index },
        ty: TirType::Set(Box::new(TirType::Dyn)),
    }))
}

fn exact_tuple2_self_subseteq_body(set: Spanned<TirExpr>) -> Spanned<TirExpr> {
    exact_tuple2_self_subseteq_body_with_id(set, tla_core::NameId(41))
}

fn exact_tuple2_self_subseteq_body_with_id(
    set: Spanned<TirExpr>,
    e_id: tla_core::NameId,
) -> Spanned<TirExpr> {
    tuple2_self_and(
        exact_tuple2_self_body_with_id(e_id),
        tuple2_self_subseteq(
            vec![
                tuple2_self_projection(tuple2_self_name("e", e_id), 1),
                tuple2_self_projection(tuple2_self_name("e", e_id), 2),
            ],
            set,
        ),
    )
}

fn compile_tuple2_self_subseteq_body(
    body: Spanned<TirExpr>,
    extra_vars: &[(&str, tla_core::NameId)],
    enable_fusion: bool,
) -> Vec<Opcode> {
    compile_tuple2_self_subseteq_body_with_state_vars(body, extra_vars, enable_fusion, &[])
}

fn compile_tuple2_self_subseteq_body_with_state_vars(
    body: Spanned<TirExpr>,
    extra_vars: &[(&str, tla_core::NameId)],
    enable_fusion: bool,
    state_vars: &[(&str, u16)],
) -> Vec<Opcode> {
    let mut vars = vec![tuple2_self_bound_var("e", tla_core::NameId(41))];
    vars.extend(
        extra_vars
            .iter()
            .map(|(name, name_id)| tuple2_self_bound_var(name, *name_id)),
    );
    let expr = spanned(TirExpr::Forall {
        vars,
        body: Box::new(body),
    });
    let mut compiler = BytecodeCompiler::new();
    if enable_fusion {
        compiler.enable_tuple2_self_subseteq();
    }
    if !state_vars.is_empty() {
        compiler.set_state_vars(
            state_vars
                .iter()
                .map(|(name, index)| ((*name).to_string(), *index))
                .collect(),
        );
    }
    let idx = compiler
        .compile_expression("tuple2-self-subseteq", &expr)
        .expect("tuple self-subset conjunction should compile");
    compiler.finish().get_function(idx).instructions.clone()
}

#[test]
fn test_compile_tuple2_self_subseteq_fusion_is_exact_and_opt_in() {
    let vs = tuple2_self_subseteq_state_name("vs", tla_core::NameId(90), 7);
    let plain =
        compile_tuple2_self_subseteq_body(exact_tuple2_self_subseteq_body(vs.clone()), &[], false);
    assert!(!plain
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfSubseteq { .. })));
    assert_eq!(
        plain
            .iter()
            .filter(|op| matches!(op, Opcode::FuncApply { .. }))
            .count(),
        4
    );
    assert!(plain
        .iter()
        .any(|op| matches!(op, Opcode::TupleNew { count: 2, .. })));
    assert!(plain
        .iter()
        .any(|op| matches!(op, Opcode::SetEnum { count: 2, .. })));
    assert!(plain.iter().any(|op| matches!(op, Opcode::Subseteq { .. })));

    let fused = compile_tuple2_self_subseteq_body(exact_tuple2_self_subseteq_body(vs), &[], true);
    assert!(fused
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfSubseteq { set_var_idx: 7, .. })));
    assert!(!fused.iter().any(|op| matches!(
        op,
        Opcode::FuncApply { .. }
            | Opcode::TupleNew { .. }
            | Opcode::Eq { .. }
            | Opcode::SetEnum { .. }
            | Opcode::Subseteq { .. }
            | Opcode::Tuple2SelfEq { .. }
            | Opcode::SetEnumSubseteq { .. }
    )));

    let mut vs = ident_name("vs");
    vs.name_id = tla_core::NameId::INVALID;
    vs.ty = TirType::Set(Box::new(TirType::Dyn));
    let production = compile_tuple2_self_subseteq_body_with_state_vars(
        exact_tuple2_self_subseteq_body_with_id(
            spanned(TirExpr::Name(vs)),
            tla_core::NameId::INVALID,
        ),
        &[],
        true,
        &[("vs", 7)],
    );
    assert!(production
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfSubseteq { set_var_idx: 7, .. })));
}

#[test]
fn test_compile_tuple2_self_subseteq_declines_reversed_and_invalid_set_rhs() {
    let e_id = tla_core::NameId(41);
    let vs_id = tla_core::NameId(90);
    let subset = tuple2_self_subseteq(
        vec![
            tuple2_self_projection(tuple2_self_name("e", e_id), 1),
            tuple2_self_projection(tuple2_self_name("e", e_id), 2),
        ],
        tuple2_self_subseteq_state_name("vs", vs_id, 7),
    );
    let reversed = compile_tuple2_self_subseteq_body(
        tuple2_self_and(subset, exact_tuple2_self_body()),
        &[],
        true,
    );
    assert!(!reversed
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfSubseteq { .. })));

    let non_state = spanned(TirExpr::Const {
        value: Value::set([Value::SmallInt(1), Value::SmallInt(2)]),
        ty: TirType::Set(Box::new(TirType::Int)),
    });
    let non_state =
        compile_tuple2_self_subseteq_body(exact_tuple2_self_subseteq_body(non_state), &[], true);
    assert!(!non_state
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfSubseteq { .. })));

    let shadowed = compile_tuple2_self_subseteq_body(
        exact_tuple2_self_subseteq_body(tuple2_self_subseteq_state_name("vs", vs_id, 7)),
        &[("vs", vs_id)],
        true,
    );
    assert!(!shadowed
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfSubseteq { .. })));
}

#[test]
fn test_compile_tuple2_self_subseteq_declines_identity_index_and_order_mismatches() {
    let e_id = tla_core::NameId(41);
    let other_id = tla_core::NameId(42);
    let cases = [
        vec![
            tuple2_self_projection(tuple2_self_name("e", e_id), 1),
            tuple2_self_projection(tuple2_self_name("e", other_id), 2),
        ],
        vec![
            tuple2_self_projection(tuple2_self_name("e", e_id), 1),
            tuple2_self_projection(tuple2_self_name("e", e_id), 3),
        ],
        vec![
            tuple2_self_projection(tuple2_self_name("e", e_id), 2),
            tuple2_self_projection(tuple2_self_name("e", e_id), 1),
        ],
    ];

    for elements in cases {
        let body = tuple2_self_and(
            exact_tuple2_self_body(),
            tuple2_self_subseteq(
                elements,
                tuple2_self_subseteq_state_name("vs", tla_core::NameId(90), 7),
            ),
        );
        let instructions = compile_tuple2_self_subseteq_body(body, &[], true);
        assert!(!instructions
            .iter()
            .any(|op| matches!(op, Opcode::Tuple2SelfSubseteq { .. })));
    }
}

#[test]
fn test_compile_tuple2_self_subseteq_declines_static_prime_context() {
    let body = spanned(TirExpr::Prime(Box::new(exact_tuple2_self_subseteq_body(
        tuple2_self_subseteq_state_name("vs", tla_core::NameId(90), 7),
    ))));
    let instructions = compile_tuple2_self_subseteq_body(body, &[], true);
    assert!(!instructions
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SelfSubseteq { .. })));
}

#[test]
fn test_compile_range() {
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Range {
        lo: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(1),
            ty: TirType::Int,
        })),
        hi: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(10),
            ty: TirType::Int,
        })),
    });
    let idx = compiler.compile_expression("test", &expr).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let has_range = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::Range { .. }));
    assert!(has_range);
}

#[test]
fn test_compile_comparison() {
    let mut compiler = BytecodeCompiler::new();
    // x < y
    let expr = spanned(TirExpr::Cmp {
        left: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(1),
            ty: TirType::Int,
        })),
        op: TirCmpOp::Lt,
        right: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(2),
            ty: TirType::Int,
        })),
    });
    let idx = compiler.compile_expression("test", &expr).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let has_lt = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::LtInt { .. }));
    assert!(has_lt);
}

#[test]
fn test_compile_record_set_packs_field_results_into_reserved_slots() {
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::RecordSet(vec![
        (
            field_name("location"),
            spanned(TirExpr::ArithBinOp {
                left: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                })),
                op: TirArithOp::Add,
                right: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                })),
            }),
        ),
        (
            field_name("waiting"),
            spanned(TirExpr::ArithBinOp {
                left: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(3),
                    ty: TirType::Int,
                })),
                op: TirArithOp::Add,
                right: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(4),
                    ty: TirType::Int,
                })),
            }),
        ),
    ]));

    let idx = compiler.compile_expression("test", &expr).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);

    assert!(
        func.instructions
            .iter()
            .any(|op| matches!(op, Opcode::Move { rd: 0, rs } if *rs != 0)),
        "first field result should be moved into reserved slot r0: {:?}",
        func.instructions
    );
    assert!(
        func.instructions
            .iter()
            .any(|op| matches!(op, Opcode::Move { rd: 1, rs } if *rs != 1)),
        "second field result should be moved into reserved slot r1: {:?}",
        func.instructions
    );
    assert!(
        func.instructions.iter().any(|op| matches!(
            op,
            Opcode::RecordSet {
                values_start: 0,
                count: 2,
                ..
            }
        )),
        "record-set opcode should read from contiguous packed slots: {:?}",
        func.instructions
    );
}

#[test]
fn test_compile_multi_level_except_desugars_to_nested_func_except() {
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Except {
        base: Box::new(spanned(TirExpr::Record(vec![(
            field_name("a"),
            spanned(TirExpr::Record(vec![(
                field_name("b"),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
            )])),
        )]))),
        specs: vec![TirExceptSpec {
            path: vec![
                TirExceptPathElement::Field(field_name("a")),
                TirExceptPathElement::Field(field_name("b")),
            ],
            value: spanned(TirExpr::Const {
                value: Value::SmallInt(99),
                ty: TirType::Int,
            }),
        }],
    });

    let idx = compiler
        .compile_expression("test", &expr)
        .expect("multi-level EXCEPT should compile via nested FuncExcept");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);

    assert_eq!(
        func.instructions
            .iter()
            .filter(|op| matches!(op, Opcode::FuncExcept { .. }))
            .count(),
        2,
        "multi-level EXCEPT should lower to two nested FuncExcept ops: {:?}",
        func.instructions
    );
}

#[test]
fn test_compile_three_level_except_desugars_recursively() {
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Except {
        base: Box::new(spanned(TirExpr::Record(vec![(
            field_name("a"),
            spanned(TirExpr::Record(vec![(
                field_name("b"),
                spanned(TirExpr::Record(vec![(
                    field_name("c"),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(2),
                        ty: TirType::Int,
                    }),
                )])),
            )])),
        )]))),
        specs: vec![TirExceptSpec {
            path: vec![
                TirExceptPathElement::Field(field_name("a")),
                TirExceptPathElement::Field(field_name("b")),
                TirExceptPathElement::Field(field_name("c")),
            ],
            value: spanned(TirExpr::Const {
                value: Value::SmallInt(99),
                ty: TirType::Int,
            }),
        }],
    });

    let idx = compiler
        .compile_expression("test", &expr)
        .expect("three-level EXCEPT should compile via recursive nested FuncExcept");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);

    assert_eq!(
        func.instructions
            .iter()
            .filter(|op| matches!(op, Opcode::FuncExcept { .. }))
            .count(),
        3,
        "three-level EXCEPT should lower to three nested FuncExcept ops: {:?}",
        func.instructions
    );
}

#[test]
fn test_compile_identifier_from_resolved_constants() {
    let mut resolved_constants = std::collections::HashMap::new();
    resolved_constants.insert(intern_name("N"), Value::SmallInt(3));

    let mut compiler = BytecodeCompiler::with_resolved_constants(resolved_constants);
    let expr = spanned(TirExpr::ArithBinOp {
        left: Box::new(spanned(TirExpr::Name(TirNameRef {
            name: "N".to_string(),
            name_id: intern_name("N"),
            kind: TirNameKind::Ident,
            ty: TirType::Int,
        }))),
        op: TirArithOp::Add,
        right: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(1),
            ty: TirType::Int,
        })),
    });

    let idx = compiler
        .compile_expression("test", &expr)
        .expect("resolved constants should compile");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);

    assert!(matches!(
        func.instructions[0],
        Opcode::LoadImm { rd: 0, value: 3 }
    ));
    assert!(
        !func
            .instructions
            .iter()
            .any(|opcode| matches!(opcode, Opcode::Call { .. })),
        "resolved constants should lower directly, not through Call",
    );
}

#[test]
fn test_ident_resolved_as_state_var_via_state_vars_map() {
    let mut compiler = BytecodeCompiler::new();
    let mut state_vars = std::collections::HashMap::new();
    state_vars.insert("x".to_string(), 0u16);
    compiler.set_state_vars(state_vars);

    let expr = spanned(TirExpr::Name(ident_name("x")));
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("Ident should resolve as state var via state_vars map");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(
        matches!(func.instructions[0], Opcode::LoadVar { rd: 0, var_idx: 0 }),
        "Ident 'x' should emit LoadVar{{var_idx:0}}: {:?}",
        func.instructions
    );
}

#[test]
fn test_prime_ident_resolved_as_state_var_via_state_vars_map() {
    let mut compiler = BytecodeCompiler::new();
    let mut state_vars = std::collections::HashMap::new();
    state_vars.insert("x".to_string(), 2u16);
    compiler.set_state_vars(state_vars);

    let expr = spanned(TirExpr::Prime(Box::new(spanned(TirExpr::Name(
        ident_name("x"),
    )))));
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("Prime(Ident) should resolve as state var via state_vars map");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(
        matches!(
            func.instructions[0],
            Opcode::LoadPrime { rd: 0, var_idx: 2 }
        ),
        "Prime(Ident 'x') should emit LoadPrime{{var_idx:2}}: {:?}",
        func.instructions
    );
}

#[test]
fn test_unchanged_single_ident_via_state_vars_map() {
    let mut compiler = BytecodeCompiler::new();
    let mut state_vars = std::collections::HashMap::new();
    state_vars.insert("x".to_string(), 0u16);
    compiler.set_state_vars(state_vars);

    let inner = spanned(TirExpr::Name(ident_name("x")));
    let expr = spanned(TirExpr::Unchanged(Box::new(inner)));
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("UNCHANGED(Ident) should resolve via state_vars map");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let has_unchanged = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::Unchanged { count: 1, .. }));
    assert!(
        has_unchanged,
        "UNCHANGED on Ident state var should emit Unchanged opcode: {:?}",
        func.instructions
    );
}

#[test]
fn test_unchanged_tuple_ident_via_state_vars_map() {
    let mut compiler = BytecodeCompiler::new();
    let mut state_vars = std::collections::HashMap::new();
    state_vars.insert("x".to_string(), 0u16);
    state_vars.insert("y".to_string(), 1u16);
    compiler.set_state_vars(state_vars);

    let inner = spanned(TirExpr::Tuple(vec![
        spanned(TirExpr::Name(ident_name("x"))),
        spanned(TirExpr::Name(ident_name("y"))),
    ]));
    let expr = spanned(TirExpr::Unchanged(Box::new(inner)));
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("UNCHANGED(<<Ident, Ident>>) should resolve via state_vars map");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let has_unchanged = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::Unchanged { count: 2, .. }));
    assert!(
        has_unchanged,
        "UNCHANGED on tuple of Ident state vars should emit Unchanged{{count:2}}: {:?}",
        func.instructions
    );
}

#[test]
fn test_ident_binding_shadows_state_var() {
    // When a name is bound (e.g., quantifier variable), it should take
    // precedence over the state_vars map — bindings are checked first.
    let mut compiler = BytecodeCompiler::new();
    let mut state_vars = std::collections::HashMap::new();
    state_vars.insert("x".to_string(), 0u16);
    compiler.set_state_vars(state_vars);

    // Use a quantifier to bind "x" — the body's reference to "x" should
    // resolve to the binding register, not emit LoadVar.
    let expr = spanned(TirExpr::Exists {
        vars: vec![TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![spanned(
                TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                },
            )])))),
            pattern: None,
        }],
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
            op: TirCmpOp::Eq,
            right: Box::new(spanned(TirExpr::Const {
                value: Value::SmallInt(1),
                ty: TirType::Int,
            })),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("quantifier-bound x should shadow state var x");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // The body should NOT contain LoadVar — "x" resolves to the quantifier binding.
    let has_load_var = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::LoadVar { .. }));
    assert!(
        !has_load_var,
        "bound variable 'x' should shadow state var 'x', no LoadVar expected: {:?}",
        func.instructions
    );
}

#[test]
fn test_op_replacement_zero_arg_ident() {
    // Same scenario but Jug appears as a standalone Name(Ident), not via Apply.
    // This covers the compile_expr.rs zero-arg path.
    let mut compiler = BytecodeCompiler::new();
    let mut replacements = std::collections::HashMap::new();
    replacements.insert("Jug".to_string(), "MCJug".to_string());
    compiler.set_op_replacements(replacements);

    let mut callees = std::collections::HashMap::new();
    callees.insert(
        "MCJug".to_string(),
        CalleeInfo {
            params: vec![],
            body: std::sync::Arc::new(spanned(TirExpr::Const {
                value: Value::SmallInt(42),
                ty: TirType::Int,
            })),
            ast_body: None,
        },
    );

    // Entry: Check == Jug  (standalone ident, zero-arg)
    let body = spanned(TirExpr::Name(ident_name("Jug")));

    let idx = compiler
        .compile_expression_with_callees("Check", &[], &body, &callees)
        .expect("zero-arg ident should resolve via op_replacement");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let has_call = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::Call { argc: 0, .. }));
    assert!(has_call, "Check should Call MCJug via operator replacement");
}
