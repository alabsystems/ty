// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-engine parity for TLA+ real division `/`.
//!
//! `/` is division on the REALS; applied to integers it only has a value when
//! the divisor evenly divides the dividend. TY implements `/` on integers as
//! EXACT-OR-ERROR in every runtime engine — an inexact quotient is an
//! evaluation error, never a truncation (`7 / 2 = 3` has no TLA+ meaning).
//!
//! Each test drives the same operator through all three engines:
//! (a) the AST evaluator, (b) the TIR tree-walker, (c) the bytecode VM —
//! and asserts they agree exactly (same value, or same error variant AND
//! rendered message).

use super::*;
use tla_core::ast::Module;

fn parity_module() -> Module {
    parse_module(
        "\
---- MODULE RealDivisionParity ----
Inexact == 7 / 2
Exact == 8 / 2
NegExact == (0 - 7) / 7
====",
    )
}

/// (a) AST evaluator.
fn eval_via_ast(module: &Module, name: &str) -> Result<Value, EvalError> {
    clear_for_test_reset();
    let mut ctx = EvalCtx::new();
    ctx.load_module(module);
    ctx.eval_op(name)
}

/// (b) TIR tree-walker — bytecode VM disabled via the thread-local override,
/// and `eval_tir` called directly on the lowered body so a TIR error cannot
/// be masked by `eval_named_op`'s AST fallback.
fn eval_via_tir_tree_walker(module: &Module, name: &str) -> Result<Value, EvalError> {
    let _overrides = set_bytecode_vm_overrides_for_current_thread(false, None);
    clear_for_test_reset();
    let program = TirProgram::from_modules(module, &[]);
    let ctx = {
        let mut ctx = EvalCtx::new();
        ctx.load_module(module);
        ctx
    };
    let body = program
        .get_or_lower(name)
        .expect("TIR lowering should succeed");
    eval_tir(&ctx, &body)
}

/// (c) bytecode VM — `try_bytecode_eval` surfaces the VM's own outcome
/// (compile failure or Unsupported would return None and fail the test),
/// so a VM error cannot be masked by the tree-walker/AST fallback chain.
fn eval_via_bytecode_vm(module: &Module, name: &str) -> Result<Value, EvalError> {
    let _guard = enable_bytecode_vm_for_test();
    clear_for_test_reset();
    let mut ctx = EvalCtx::new();
    ctx.load_module(module);
    let state_values: Vec<Value> = vec![];
    let _state_guard = ctx.bind_state_array_guard(&state_values);
    let program = TirProgram::from_modules(module, &[]);
    let body = program
        .get_or_lower(name)
        .expect("TIR lowering should succeed");
    program
        .try_bytecode_eval(&ctx, name, &body)
        .expect("operator should compile and execute in the bytecode VM")
}

/// All three engines must produce the same exact value.
fn assert_cross_engine_value(module: &Module, name: &str, expected: i64) {
    let expected = Value::int(expected);
    for (engine, result) in [
        ("AST evaluator", eval_via_ast(module, name)),
        ("TIR tree-walker", eval_via_tir_tree_walker(module, name)),
        ("bytecode VM", eval_via_bytecode_vm(module, name)),
    ] {
        let value = result.unwrap_or_else(|e| panic!("{engine}: '{name}' should evaluate: {e}"));
        assert_eq!(value, expected, "{engine}: wrong value for '{name}'");
    }
}

#[test]
fn cross_engine_exact_division_agrees() {
    let module = parity_module();
    assert_cross_engine_value(&module, "Exact", 4); // 8 / 2 = 4
    assert_cross_engine_value(&module, "NegExact", -1); // (0 - 7) / 7 = -1
}

#[test]
fn cross_engine_inexact_division_errors_everywhere() {
    let module = parity_module();
    let outcomes = [
        ("AST evaluator", eval_via_ast(&module, "Inexact")),
        (
            "TIR tree-walker",
            eval_via_tir_tree_walker(&module, "Inexact"),
        ),
        ("bytecode VM", eval_via_bytecode_vm(&module, "Inexact")),
    ];
    let mut messages = Vec::new();
    for (engine, result) in outcomes {
        let err = match result {
            Ok(v) => panic!("{engine}: inexact `7 / 2` must error, got value: {v:?}"),
            Err(e) => e,
        };
        assert!(
            matches!(err, EvalError::ArgumentError { ref op, .. } if op == "/"),
            "{engine}: expected the shared ArgumentError for `/`, got: {err:?}"
        );
        messages.push((engine, err.to_string()));
    }
    // The rendered message must be byte-identical across engines.
    let (_, first) = &messages[0];
    for (engine, message) in &messages[1..] {
        assert_eq!(
            message, first,
            "{engine}: inexact-`/` error message diverges from the AST evaluator's"
        );
    }
}
