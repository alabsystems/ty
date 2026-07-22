// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for the eval implied-action bytecode fast path
//! (`attach_eval_implied_action_bytecode` + the VM execution contract).

use super::*;
use crate::Value;

/// Refinement-shaped term: a state-dependent zero-arg derived operator (`d`,
/// force-externalized because it mentions a state variable) referenced both
/// unprimed and PRIMED by an action-level PROPERTY term. Verifies:
///   * the term compiles and a VM handle is attached;
///   * the VM verdict matches the semantics on a true transition (primed
///     external `d'` must evaluate against the SUCCESSOR state via the
///     swapped-array discipline);
///   * a false transition yields `Bool(false)` (which production code then
///     routes to the interpreter for authoritative reporting).
#[test]
fn implied_bytecode_attach_and_vm_verdicts() {
    let src = r"
---- MODULE ImpliedBcAttach ----
EXTENDS Integers
VARIABLE x
d == x + 1
Prop == [][d' = d + 1]_<<x>>
====";
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let (op_defs, mut ctx, _root) = setup_for_classification(src);
    let result =
        crate::checker_ops::classify_property_safety_parts(&ctx, &["Prop".to_string()], &op_defs);
    let mut terms = result.eval_implied_actions;
    assert_eq!(terms.len(), 1, "expected one eval implied-action term");
    assert!(terms[0].vm.is_none(), "no VM handle before attach");

    crate::checker_ops::attach_eval_implied_action_bytecode(&ctx, &module, &[], &mut terms);
    let vm_spec = terms[0]
        .vm
        .as_ref()
        .expect("implied-action term should compile and attach a VM handle");

    // parent x=1 -> d=2, d+1=3; succ x=2 -> d'=3: term TRUE via the action.
    let parent = ArrayState::from_values(vec![Value::int(1)]);
    let succ_true = ArrayState::from_values(vec![Value::int(2)]);
    // succ x=5 -> d'=6 != 3 and x changed: term FALSE.
    let succ_false = ArrayState::from_values(vec![Value::int(5)]);

    for (succ, expected) in [(&succ_true, true), (&succ_false, false)] {
        let _next_guard = ctx.take_next_state_guard();
        let _state_guard = ctx.bind_state_env_guard(parent.env_ref());
        let _next_env_guard = ctx.bind_next_state_env_guard(succ.env_ref());
        crate::eval::clear_for_bound_state_eval_scope(&ctx);
        let mut vm = tla_eval::bytecode_vm::BytecodeVm::from_state_env(
            &vm_spec.compiled.chunk,
            parent.env_ref(),
            Some(succ.env_ref()),
        )
        .with_eval_ctx(&ctx)
        .with_zero_arg_external_memo();
        let verdict = vm
            .execute_function(vm_spec.func_idx)
            .expect("implied-action VM execution should succeed");
        assert_eq!(
            verdict,
            Value::Bool(expected),
            "VM verdict must match semantics (primed external d' evaluates on the successor)"
        );
    }
}

/// The same transition evaluated by the VM and by the interpreter must agree
/// — the in-process analog of the TY_IMPLIED_BC_XCHECK harness.
#[test]
fn implied_bytecode_vm_matches_interpreter() {
    let src = r"
---- MODULE ImpliedBcParity ----
EXTENDS Integers
VARIABLE x
d == x + 1
Prop == [][d' = d + 1]_<<x>>
====";
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let (op_defs, mut ctx, _root) = setup_for_classification(src);
    let result =
        crate::checker_ops::classify_property_safety_parts(&ctx, &["Prop".to_string()], &op_defs);
    let mut terms = result.eval_implied_actions;
    crate::checker_ops::attach_eval_implied_action_bytecode(&ctx, &module, &[], &mut terms);
    let term = &terms[0];
    let vm_spec = term.vm.as_ref().expect("VM handle attached");

    let parent = ArrayState::from_values(vec![Value::int(3)]);
    for succ_x in 0..8 {
        let succ = ArrayState::from_values(vec![Value::int(succ_x)]);
        let _next_guard = ctx.take_next_state_guard();
        let _state_guard = ctx.bind_state_env_guard(parent.env_ref());
        let _next_env_guard = ctx.bind_next_state_env_guard(succ.env_ref());
        crate::eval::clear_for_bound_state_eval_scope(&ctx);

        let vm_verdict = {
            let mut vm = tla_eval::bytecode_vm::BytecodeVm::from_state_env(
                &vm_spec.compiled.chunk,
                parent.env_ref(),
                Some(succ.env_ref()),
            )
            .with_eval_ctx(&ctx)
            .with_zero_arg_external_memo();
            vm.execute_function(vm_spec.func_idx)
                .expect("VM execution should succeed")
        };

        crate::eval::clear_for_state_eval_replay(&ctx);
        let interp_verdict =
            crate::eval::eval_entry(&ctx, &term.expr).expect("interpreter eval should succeed");
        assert_eq!(
            vm_verdict, interp_verdict,
            "VM and interpreter must agree for succ x={succ_x}"
        );
    }
}
