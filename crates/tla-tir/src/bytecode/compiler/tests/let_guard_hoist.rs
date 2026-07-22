// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! WP-21 guard-first LET pins.
//!
//! `LET defs IN /\ c1 .. /\ cn` must evaluate a provably pure leading-guard
//! prefix BEFORE the defs (short-circuiting to FALSE without touching them),
//! and must keep the eager defs-first order whenever the analysis cannot
//! prove the reorder safe.

use super::*;
use std::collections::HashMap;

fn state_var(name: &str) -> TirNameRef {
    TirNameRef {
        name: name.to_string(),
        name_id: tla_core::NameId(0),
        kind: TirNameKind::Ident,
        ty: TirType::Dyn,
    }
}

fn int_const(v: i64) -> Spanned<TirExpr> {
    spanned(TirExpr::Const {
        value: Value::SmallInt(v),
        ty: TirType::Int,
    })
}

/// `args[1]` — the btree def shape.
fn args_index(idx: i64) -> Spanned<TirExpr> {
    spanned(TirExpr::FuncApply {
        func: Box::new(spanned(TirExpr::Name(state_var("args")))),
        arg: Box::new(int_const(idx)),
    })
}

/// `state = <v>` on the `state` state variable.
fn state_eq(v: i64) -> Spanned<TirExpr> {
    spanned(TirExpr::Cmp {
        left: Box::new(spanned(TirExpr::Name(state_var("state")))),
        op: TirCmpOp::Eq,
        right: Box::new(int_const(v)),
    })
}

fn and(left: Spanned<TirExpr>, right: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::BoolBinOp {
        left: Box::new(left),
        op: TirBoolOp::And,
        right: Box::new(right),
    })
}

fn zero_arg_def(name: &str, body: Spanned<TirExpr>) -> TirLetDef {
    TirLetDef {
        name: name.to_string(),
        name_id: tla_core::NameId(0),
        params: vec![],
        body,
    }
}

fn compiler_with_state_vars(vars: &[(&str, u16)]) -> BytecodeCompiler {
    let mut compiler = BytecodeCompiler::new();
    let map: HashMap<String, u16> = vars
        .iter()
        .map(|(n, i)| ((*n).to_string(), *i))
        .collect();
    compiler.set_state_vars(map);
    compiler
}

fn opcode_positions(func: &BytecodeFunction) -> (Option<usize>, Option<usize>) {
    let first_jump_false = func
        .instructions
        .iter()
        .position(|op| matches!(op, Opcode::JumpFalse { .. }));
    let first_func_apply = func
        .instructions
        .iter()
        .position(|op| matches!(op, Opcode::FuncApply { .. }));
    (first_jump_false, first_func_apply)
}

/// The btree `UpdateLeaf` shape: pure defs reading `args`, first conjunct a
/// pure state guard. The guard's short-circuit MUST precede the defs'
/// `FuncApply` so a disabled parent never touches `args`.
#[test]
fn pure_state_guard_evaluates_before_let_defs() {
    let mut compiler = compiler_with_state_vars(&[("state", 0), ("args", 1)]);
    // LET key == args[1] IN /\ state = 7 /\ key = 1
    let body = spanned(TirExpr::Let {
        defs: vec![zero_arg_def("key", args_index(1))],
        body: Box::new(and(
            state_eq(7),
            spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(state_var("key")))),
                op: TirCmpOp::Eq,
                right: Box::new(int_const(1)),
            }),
        )),
    });
    let idx = compiler.compile_expression("GuardFirst", &body).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let (first_jump_false, first_func_apply) = opcode_positions(func);
    let jf = first_jump_false.expect("hoisted guard must short-circuit via JumpFalse");
    let fa = first_func_apply.expect("the def body must still compile (FuncApply on args)");
    assert!(
        jf < fa,
        "guard JumpFalse (pc={jf}) must precede the def's FuncApply (pc={fa}): {:?}",
        func.instructions
    );
}

/// A first conjunct that references a LET-bound name pins the eager order:
/// the def's FuncApply stays BEFORE any short-circuit.
#[test]
fn guard_referencing_def_name_is_not_hoisted() {
    let mut compiler = compiler_with_state_vars(&[("state", 0), ("args", 1)]);
    // LET key == args[1] IN /\ key = 1 /\ state = 7
    let body = spanned(TirExpr::Let {
        defs: vec![zero_arg_def("key", args_index(1))],
        body: Box::new(and(
            spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(state_var("key")))),
                op: TirCmpOp::Eq,
                right: Box::new(int_const(1)),
            }),
            state_eq(7),
        )),
    });
    let idx = compiler.compile_expression("NoHoist", &body).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let (first_jump_false, first_func_apply) = opcode_positions(func);
    let fa = first_func_apply.expect("def body compiles eagerly");
    if let Some(jf) = first_jump_false {
        assert!(
            fa < jf,
            "def must evaluate before the chain when conjunct 1 uses it: {:?}",
            func.instructions
        );
    }
}

/// A CHOOSE def is not provably pure: the whole LET keeps the eager order
/// (fail closed), even though the first conjunct is a hoistable guard.
#[test]
fn choose_def_blocks_hoist() {
    let mut compiler = compiler_with_state_vars(&[("state", 0)]);
    // LET c == CHOOSE x \in {1} : TRUE IN /\ state = 7 /\ c = 1
    let choose = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![int_const(1)])))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::Const {
            value: Value::Bool(true),
            ty: TirType::Bool,
        })),
    });
    let body = spanned(TirExpr::Let {
        defs: vec![zero_arg_def("c", choose)],
        body: Box::new(and(
            state_eq(7),
            spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(state_var("c")))),
                op: TirCmpOp::Eq,
                right: Box::new(int_const(1)),
            }),
        )),
    });
    let idx = compiler.compile_expression("ChooseDef", &body).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let first_choose = func
        .instructions
        .iter()
        .position(|op| matches!(op, Opcode::ChooseBegin { .. }))
        .expect("CHOOSE def compiles");
    let first_jump_false = func
        .instructions
        .iter()
        .position(|op| matches!(op, Opcode::JumpFalse { .. }))
        .expect("the And chain short-circuits");
    assert!(
        first_choose < first_jump_false,
        "impure def must keep the eager defs-first order: {:?}",
        func.instructions
    );
}

/// A primed first conjunct is not a hoistable guard, and the prefix rule
/// stops there: nothing is reordered.
#[test]
fn primed_guard_is_not_hoisted() {
    let mut compiler = compiler_with_state_vars(&[("state", 0), ("args", 1)]);
    // LET key == args[1] IN /\ state' = 7 /\ state = 7
    let primed_guard = spanned(TirExpr::Cmp {
        left: Box::new(spanned(TirExpr::Prime(Box::new(spanned(TirExpr::Name(
            state_var("state"),
        )))))),
        op: TirCmpOp::Eq,
        right: Box::new(int_const(7)),
    });
    let body = spanned(TirExpr::Let {
        defs: vec![zero_arg_def("key", args_index(1))],
        body: Box::new(and(primed_guard, state_eq(7))),
    });
    let idx = compiler.compile_expression("PrimedGuard", &body).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let (first_jump_false, first_func_apply) = opcode_positions(func);
    let fa = first_func_apply.expect("def body compiles eagerly");
    if let Some(jf) = first_jump_false {
        assert!(
            fa < jf,
            "primed first conjunct must keep the eager order: {:?}",
            func.instructions
        );
    }
}

/// Multi-guard prefix: both leading pure guards hoist; the def evaluates
/// after BOTH JumpFalse short-circuits.
#[test]
fn multiple_leading_guards_all_hoist() {
    let mut compiler = compiler_with_state_vars(&[("state", 0), ("op", 1), ("args", 2)]);
    // LET key == args[1] IN /\ state = 7 /\ op = 3 /\ key = 1
    let op_eq = spanned(TirExpr::Cmp {
        left: Box::new(spanned(TirExpr::Name(state_var("op")))),
        op: TirCmpOp::Eq,
        right: Box::new(int_const(3)),
    });
    let key_eq = spanned(TirExpr::Cmp {
        left: Box::new(spanned(TirExpr::Name(state_var("key")))),
        op: TirCmpOp::Eq,
        right: Box::new(int_const(1)),
    });
    let body = spanned(TirExpr::Let {
        defs: vec![zero_arg_def("key", args_index(1))],
        body: Box::new(and(state_eq(7), and(op_eq, key_eq))),
    });
    let idx = compiler.compile_expression("TwoGuards", &body).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let jump_false_pcs: Vec<usize> = func
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| matches!(op, Opcode::JumpFalse { .. }).then_some(pc))
        .collect();
    let first_func_apply = func
        .instructions
        .iter()
        .position(|op| matches!(op, Opcode::FuncApply { .. }))
        .expect("def body compiles");
    assert!(
        jump_false_pcs.len() >= 2 && jump_false_pcs[1] < first_func_apply,
        "both guards must short-circuit before the def evaluates \
         (jump_false_pcs={jump_false_pcs:?}, first_func_apply={first_func_apply}): {:?}",
        func.instructions
    );
}

/// The hoisted form must preserve VALUES on the enabled path: same guard,
/// same def, same final conjunct — the accumulator register holds the
/// conjunction result and every JumpFalse targets the function end.
#[test]
fn hoisted_chain_jumps_target_function_end() {
    let mut compiler = compiler_with_state_vars(&[("state", 0), ("args", 1)]);
    let body = spanned(TirExpr::Let {
        defs: vec![zero_arg_def("key", args_index(1))],
        body: Box::new(and(
            state_eq(7),
            spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(state_var("key")))),
                op: TirCmpOp::Eq,
                right: Box::new(int_const(1)),
            }),
        )),
    });
    let idx = compiler.compile_expression("JumpTargets", &body).unwrap();
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // Last instruction is Ret; every JumpFalse must land exactly on it (the
    // chain end), never inside the def evaluation.
    let ret_pc = func.instructions.len() - 1;
    assert!(
        matches!(func.instructions[ret_pc], Opcode::Ret { .. }),
        "chunk functions end in Ret: {:?}",
        func.instructions
    );
    for (pc, op) in func.instructions.iter().enumerate() {
        if let Opcode::JumpFalse { offset, .. } = op {
            let target = (pc as isize + *offset as isize) as usize;
            assert_eq!(
                target, ret_pc,
                "JumpFalse at pc={pc} must target the chain end {ret_pc}: {:?}",
                func.instructions
            );
        }
    }
}
