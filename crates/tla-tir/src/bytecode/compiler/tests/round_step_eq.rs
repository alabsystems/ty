// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Structural proof-boundary tests for the VM-only RoundStepEq fusion.

use super::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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

fn round_callee(name: &str, body: Spanned<TirExpr>) -> (String, CalleeInfo) {
    (
        name.to_string(),
        CalleeInfo {
            params: vec!["p".to_string()],
            body: Arc::new(body),
            ast_body: None,
        },
    )
}

fn call(callee: &str, arg: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::Apply {
        op: Box::new(name(callee)),
        args: vec![arg],
    })
}

fn exact_body(child_callee: &str, parent_callee: &str) -> Spanned<TirExpr> {
    spanned(TirExpr::Cmp {
        left: Box::new(call(child_callee, name("child"))),
        op: TirCmpOp::Eq,
        right: Box::new(spanned(TirExpr::ArithBinOp {
            left: Box::new(call(parent_callee, name("parent"))),
            op: TirArithOp::Sub,
            right: Box::new(int(1)),
        })),
    })
}

fn forall(body: Spanned<TirExpr>, pattern: Option<TirBoundPattern>) -> Spanned<TirExpr> {
    spanned(TirExpr::Forall {
        vars: vec![TirBoundVar {
            name: "child".to_string(),
            name_id: intern_name("child"),
            domain: Some(Box::new(spanned(TirExpr::Const {
                value: Value::set([Value::tuple([Value::SmallInt(1), Value::SmallInt(2)])]),
                ty: TirType::Set(Box::new(TirType::Dyn)),
            }))),
            pattern,
        }],
        body: Box::new(body),
    })
}

fn compile(
    expr: &Spanned<TirExpr>,
    callees: &HashMap<String, CalleeInfo>,
    enabled: bool,
) -> Vec<Opcode> {
    let mut compiler = BytecodeCompiler::new();
    if enabled {
        compiler.enable_round_step_eq();
    }
    let idx = compiler
        .compile_expression_with_callees("RoundStep", &["parent".to_string()], expr, callees)
        .expect("Round-step bytecode");
    compiler.finish().get_function(idx).instructions.clone()
}

fn assert_refused(expr: Spanned<TirExpr>, callees: &HashMap<String, CalleeInfo>) {
    let baseline = compile(&expr, callees, false);
    let enabled = compile(&expr, callees, true);
    assert_eq!(enabled, baseline, "refusal must preserve bytecode exactly");
    assert!(!enabled
        .iter()
        .any(|opcode| matches!(opcode, Opcode::RoundStepEq { .. })));
}

#[test]
fn exact_shape_is_default_off_and_emits_one_loop_body_opcode_when_enabled() {
    let callees = HashMap::from([round_callee("Round", round_body("p"))]);
    let expr = forall(exact_body("Round", "Round"), None);

    let baseline = compile(&expr, &callees, false);
    assert!(!baseline
        .iter()
        .any(|opcode| matches!(opcode, Opcode::RoundStepEq { .. })));

    let fused = compile(&expr, &callees, true);
    let begin = fused
        .iter()
        .position(|opcode| matches!(opcode, Opcode::ForallBegin { .. }))
        .expect("ForallBegin");
    let Opcode::ForallBegin { r_binding, .. } = fused[begin] else {
        unreachable!();
    };
    assert_eq!(
        fused[begin + 1],
        Opcode::RoundStepEq {
            rd: fused[begin + 1].dest_register().expect("destination"),
            child: r_binding,
            parent: 0,
        }
    );
    let Opcode::ForallNext { r_body, .. } = fused[begin + 2] else {
        panic!("fused body must be exactly one opcode: {fused:?}");
    };
    assert_eq!(Some(r_body), fused[begin + 1].dest_register());
    assert_eq!(
        fused
            .iter()
            .filter(|opcode| matches!(opcode, Opcode::RoundStepEq { .. }))
            .count(),
        1
    );
}

#[test]
fn orientation_literal_parent_and_callee_variants_are_refused() {
    let callees = HashMap::from([
        round_callee("Round", round_body("p")),
        round_callee("OtherRound", round_body("p")),
    ]);

    let flipped = spanned(TirExpr::Cmp {
        left: match exact_body("Round", "Round").node {
            TirExpr::Cmp { right, .. } => right,
            _ => unreachable!(),
        },
        op: TirCmpOp::Eq,
        right: Box::new(call("Round", name("child"))),
    });
    assert_refused(forall(flipped, None), &callees);

    let wrong_literal = spanned(TirExpr::Cmp {
        left: Box::new(call("Round", name("child"))),
        op: TirCmpOp::Eq,
        right: Box::new(spanned(TirExpr::ArithBinOp {
            left: Box::new(call("Round", name("parent"))),
            op: TirArithOp::Sub,
            right: Box::new(int(2)),
        })),
    });
    assert_refused(forall(wrong_literal, None), &callees);

    let wrong_parent = spanned(TirExpr::Cmp {
        left: Box::new(call("Round", name("child"))),
        op: TirCmpOp::Eq,
        right: Box::new(spanned(TirExpr::ArithBinOp {
            left: Box::new(call(
                "Round",
                spanned(TirExpr::FuncApply {
                    func: Box::new(name("parent")),
                    arg: Box::new(int(1)),
                }),
            )),
            op: TirArithOp::Sub,
            right: Box::new(int(1)),
        })),
    });
    assert_refused(forall(wrong_parent, None), &callees);

    assert_refused(forall(exact_body("Round", "OtherRound"), None), &callees);
}

#[test]
fn pattern_multi_binder_and_incomplete_round_definition_are_refused() {
    let callees = HashMap::from([round_callee("Round", round_body("p"))]);
    assert_refused(
        forall(
            exact_body("Round", "Round"),
            Some(TirBoundPattern::Var(
                "child".to_string(),
                intern_name("child"),
            )),
        ),
        &callees,
    );

    let multi = spanned(TirExpr::Forall {
        vars: vec![
            TirBoundVar {
                name: "ignored".to_string(),
                name_id: intern_name("ignored"),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![int(0)])))),
                pattern: None,
            },
            TirBoundVar {
                name: "child".to_string(),
                name_id: intern_name("child"),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![int(0)])))),
                pattern: None,
            },
        ],
        body: Box::new(exact_body("Round", "Round")),
    });
    assert_refused(multi, &callees);

    let incomplete = HashMap::from([round_callee("Round", int(0))]);
    assert_refused(forall(exact_body("Round", "Round"), None), &incomplete);

    let primed = spanned(TirExpr::Prime(Box::new(forall(
        exact_body("Round", "Round"),
        None,
    ))));
    assert_refused(primed, &callees);
}

#[test]
fn replacement_forced_external_and_shadowed_round_are_refused() {
    let callees = HashMap::from([
        round_callee("Round", round_body("p")),
        round_callee("Target", round_body("p")),
    ]);
    let expr = forall(exact_body("Round", "Round"), None);

    let compile_configured = |enabled: bool, external: bool| {
        let mut compiler = BytecodeCompiler::new();
        compiler.set_op_replacements(HashMap::from([("Round".to_string(), "Target".to_string())]));
        if external {
            compiler.set_force_external_ops(HashSet::from(["Target".to_string()]));
        }
        if enabled {
            compiler.enable_round_step_eq();
        }
        let idx = compiler
            .compile_expression_with_callees("RoundStep", &["parent".to_string()], &expr, &callees)
            .expect("configured Round-step bytecode");
        compiler.finish().get_function(idx).instructions.clone()
    };
    assert_eq!(
        compile_configured(false, false),
        compile_configured(true, false)
    );
    assert_eq!(
        compile_configured(false, true),
        compile_configured(true, true)
    );

    let shadowed = spanned(TirExpr::Let {
        defs: vec![TirLetDef {
            name: "Round".to_string(),
            name_id: intern_name("Round"),
            params: vec!["q".to_string()],
            body: round_body("q"),
        }],
        body: Box::new(expr.clone()),
    });
    assert_refused(shadowed, &callees);

    let compile_higher_order = |enabled: bool| {
        let mut compiler = BytecodeCompiler::new();
        if enabled {
            compiler.enable_round_step_eq();
        }
        let idx = compiler
            .compile_expression_with_callees(
                "RoundStepHigherOrder",
                &["parent".to_string(), "Round".to_string()],
                &expr,
                &callees,
            )
            .expect("higher-order shadow bytecode");
        compiler.finish().get_function(idx).instructions.clone()
    };
    assert_eq!(compile_higher_order(false), compile_higher_order(true));
}

#[test]
fn unreplaced_forced_external_round_is_refused() {
    let callees = HashMap::from([round_callee("Round", round_body("p"))]);
    let expr = forall(exact_body("Round", "Round"), None);

    let compile_forced_external = |enabled: bool| {
        let mut compiler = BytecodeCompiler::new();
        compiler.set_force_external_ops(HashSet::from(["Round".to_string()]));
        if enabled {
            compiler.enable_round_step_eq();
        }
        let idx = compiler
            .compile_expression_with_callees(
                "RoundStepForcedExternal",
                &["parent".to_string()],
                &expr,
                &callees,
            )
            .expect("forced-external Round-step bytecode");
        compiler.finish().get_function(idx).instructions.clone()
    };

    let baseline = compile_forced_external(false);
    let enabled = compile_forced_external(true);
    assert_eq!(enabled, baseline, "refusal must preserve bytecode exactly");
    assert!(enabled
        .iter()
        .any(|opcode| matches!(opcode, Opcode::CallExternal { .. })));
    assert!(!enabled
        .iter()
        .any(|opcode| matches!(opcode, Opcode::RoundStepEq { .. })));
}

#[test]
fn binder_name_id_mismatch_is_refused() {
    let callees = HashMap::from([round_callee("Round", round_body("p"))]);
    let mut body = exact_body("Round", "Round");
    let TirExpr::Cmp { left, .. } = &mut body.node else {
        unreachable!()
    };
    let TirExpr::Apply { args, .. } = &mut left.node else {
        unreachable!()
    };
    let TirExpr::Name(child) = &mut args[0].node else {
        unreachable!()
    };
    child.name_id = intern_name("different-child-id");
    assert_refused(forall(body, None), &callees);
}
