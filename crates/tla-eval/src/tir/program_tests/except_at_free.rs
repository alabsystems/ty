// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use tla_value::Rp;

use super::*;

struct VmOutcome {
    result: Result<Value, EvalError>,
    func_apply_count: usize,
    chunk_func_apply_count: usize,
}

fn eval_tree(module: &Module, name: &str) -> Result<Value, EvalError> {
    clear_for_test_reset();
    let mut ctx = EvalCtx::new();
    ctx.load_module(module);
    let program = TirProgram::from_modules(module, &[]);
    let body = program
        .get_or_lower(name)
        .expect("EXCEPT test operator should lower");
    eval_tir(&ctx, &body)
}

fn eval_vm(module: &Module, name: &str) -> VmOutcome {
    clear_for_test_reset();
    let mut ctx = EvalCtx::new();
    ctx.load_module(module);
    let state_values: Vec<Value> = vec![];
    let _state_guard = ctx.bind_state_array_guard(&state_values);
    let program = TirProgram::from_modules(module, &[]);
    let body = program
        .get_or_lower(name)
        .expect("EXCEPT test operator should lower");
    let result = program
        .try_bytecode_eval(&ctx, name, &body)
        .expect("EXCEPT test operator should compile and execute in the VM");
    let func_apply_count = program
        .compiled_func_apply_count(name)
        .expect("compiled EXCEPT function should remain cached");
    let chunk_func_apply_count = program.compiled_chunk_func_apply_count();
    VmOutcome {
        result,
        func_apply_count,
        chunk_func_apply_count,
    }
}

fn eval_tree_with_state(
    module: &Module,
    base_ctx: &EvalCtx,
    name: &str,
    current: &[Value],
    next: &[Value],
) -> Result<Value, EvalError> {
    clear_for_test_reset();
    let mut ctx = base_ctx.clone();
    let _state_guard = ctx.bind_state_array_guard(current);
    let _next_guard = ctx.bind_next_state_array_guard(next);
    let program = TirProgram::from_modules(module, &[]);
    let body = program
        .get_or_lower(name)
        .expect("stateful EXCEPT test operator should lower");
    eval_tir(&ctx, &body)
}

fn eval_vm_with_state(
    module: &Module,
    base_ctx: &EvalCtx,
    name: &str,
    current: &[Value],
    next: &[Value],
) -> VmOutcome {
    clear_for_test_reset();
    let mut ctx = base_ctx.clone();
    let _state_guard = ctx.bind_state_array_guard(current);
    let _next_guard = ctx.bind_next_state_array_guard(next);
    let program = TirProgram::from_modules(module, &[]);
    let body = program
        .get_or_lower(name)
        .expect("stateful EXCEPT test operator should lower");
    let result = program
        .try_bytecode_eval(&ctx, name, &body)
        .expect("stateful EXCEPT should compile and execute in the VM");
    let func_apply_count = program
        .compiled_func_apply_count(name)
        .expect("compiled stateful EXCEPT should remain cached");
    VmOutcome {
        result,
        func_apply_count,
        chunk_func_apply_count: program.compiled_chunk_func_apply_count(),
    }
}

fn assert_same_value(
    expected: Result<Value, EvalError>,
    actual: Result<Value, EvalError>,
    context: &str,
) -> Value {
    let expected = expected.unwrap_or_else(|err| panic!("{context}: reference errored: {err}"));
    let actual = actual.unwrap_or_else(|err| panic!("{context}: comparison errored: {err}"));
    assert_eq!(actual, expected, "{context}: values diverged");
    actual
}

#[test]
fn vm_except_at_free_existing_key_matches_tree_and_omits_at_by_default() {
    let module = parse_module(
        "\
---- MODULE VmExceptAtFreeExisting ----
Eval == [[i \\in {1, 2} |-> i * 10] EXCEPT ![1] = 99]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree = eval_tree(&module, "Eval");
    let vm = eval_vm(&module, "Eval");
    assert_same_value(tree, vm.result, "default VM vs tree");
    assert_eq!(vm.func_apply_count, 0, "default VM must omit unused @");
}

#[test]
fn vm_except_at_free_preserves_func_path_and_value_register_aliases() {
    let module = parse_module(
        "\
---- MODULE VmExceptAtFreeAliases ----
WithKey(f, k) == [f EXCEPT ![k] = k]
WithFunc(f, k) == [f EXCEPT ![k] = f]
Base == [i \\in {1, 2} |-> i * 10]
Eval == <<WithKey(Base, 1), WithFunc(Base, 2)>>
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree = eval_tree(&module, "Eval");
    let vm = eval_vm(&module, "Eval");
    assert_same_value(tree, vm.result, "alias default VM vs tree");
    assert_eq!(
        vm.chunk_func_apply_count, 0,
        "r_func/r_path reused as r_val must remain valid without @"
    );
}

#[test]
fn vm_except_at_free_state_loads_match_tree_in_current_and_prime_contexts() {
    let mut module = parse_module(
        "\
---- MODULE VmExceptAtFreeStateLoads ----
VARIABLE f, replacement
Current == [f EXCEPT ![1] = replacement]
Next == ([f EXCEPT ![1] = replacement])'
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let mut base_ctx = EvalCtx::new();
    base_ctx.register_var("f");
    base_ctx.register_var("replacement");
    base_ctx.load_module(&module);
    base_ctx.resolve_state_vars_in_loaded_ops();
    resolve_module_state_vars(&mut module, &base_ctx);

    let current = vec![
        Value::IntFunc(Rp::new(tla_value::IntIntervalFunc::new(
            1,
            2,
            vec![Value::int(10), Value::int(20)],
        ))),
        Value::int(99),
    ];
    let next = vec![
        Value::IntFunc(Rp::new(tla_value::IntIntervalFunc::new(
            1,
            2,
            vec![Value::int(30), Value::int(40)],
        ))),
        Value::int(88),
    ];

    for name in ["Current", "Next"] {
        let tree = eval_tree_with_state(&module, &base_ctx, name, &current, &next);
        let vm = eval_vm_with_state(&module, &base_ctx, name, &current, &next);
        assert_same_value(tree, vm.result, &format!("{name} state-load default VM"));
        assert_eq!(vm.func_apply_count, 0, "{name}: default VM @ shape");
    }
}

#[test]
fn vm_except_at_free_prime_state_load_falls_back_without_next_state_array() {
    let mut module = parse_module(
        "\
---- MODULE VmExceptAtFreePrimeFallback ----
VARIABLE f, replacement
Eval == ([f EXCEPT ![1] = replacement])'
====",
    );
    let _guard = enable_bytecode_vm_for_test();
    let mut base_ctx = EvalCtx::new();
    base_ctx.register_var("f");
    base_ctx.register_var("replacement");
    base_ctx.load_module(&module);
    base_ctx.resolve_state_vars_in_loaded_ops();
    resolve_module_state_vars(&mut module, &base_ctx);

    let current = vec![
        Value::IntFunc(Rp::new(tla_value::IntIntervalFunc::new(
            1,
            1,
            vec![Value::int(10)],
        ))),
        Value::int(99),
    ];
    let mut ctx = base_ctx.clone();
    let _state_guard = ctx.bind_state_array_guard(&current);
    let program = TirProgram::from_modules(&module, &[]);
    let body = program
        .get_or_lower("Eval")
        .expect("prime EXCEPT should lower");

    let tree_error = eval_tir(&ctx, &body).expect_err("tree-walker needs a next-state binding");
    assert!(matches!(
        tree_error,
        EvalError::PrimedVariableNotBound { .. }
    ));
    assert!(
        program.try_bytecode_eval(&ctx, "Eval", &body).is_none(),
        "VM must decline execution when LoadPrime has no next-state array"
    );
    assert_eq!(
        program
            .compiled_func_apply_count("Eval")
            .expect("declined compiled function should remain cached"),
        0,
        "the state-load certificate is still used at compile time"
    );
}

#[test]
fn vm_except_at_free_out_of_domain_matches_lazy_treewalker_noop() {
    let module = parse_module(
        "\
---- MODULE VmExceptAtFreeMissing ----
Eval == [[i \\in {1, 2} |-> i * 10] EXCEPT ![3] = 99]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree = eval_tree(&module, "Eval").expect("tree-walker EXCEPT should be a no-op");
    let vm = eval_vm(&module, "Eval");
    assert_same_value(
        Ok(tree),
        vm.result,
        "default VM out-of-domain no-op vs tree",
    );
    assert_eq!(vm.func_apply_count, 0);
}

#[test]
fn vm_except_at_free_out_of_domain_matrix_matches_treewalker_noops() {
    let modules = [
        parse_module(
            "\
---- MODULE VmExceptAtFreeMissingIntFunc ----
Eval == [[i \\in 1..2 |-> i] EXCEPT ![3] = 99]
====",
        ),
        parse_module(
            "\
---- MODULE VmExceptAtFreeMissingRecordIndex ----
Eval == [[a |-> 1] EXCEPT ![\"missing\"] = 99]
====",
        ),
        parse_module(
            "\
---- MODULE VmExceptAtFreeMissingTuple ----
Eval == [<<1, 2>> EXCEPT ![3] = 99]
====",
        ),
    ];
    let _guard = enable_bytecode_vm_for_test();

    for module in &modules {
        let tree = eval_tree(module, "Eval").expect("tree-walker update should be a no-op");
        let vm = eval_vm(module, "Eval");
        assert_same_value(Ok(tree), vm.result, "default out-of-domain matrix case");
        assert_eq!(vm.func_apply_count, 0);
    }
}

#[test]
fn vm_except_at_free_nested_path_preserves_navigation_and_update_order() {
    let module = parse_module(
        "\
---- MODULE VmExceptAtFreeNested ----
Eval == [[i \\in {1, 2} |-> [j \\in {1, 2} |-> i * 10 + j]] EXCEPT ![1][1] = 9, ![1][2] = 8]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree = eval_tree(&module, "Eval");
    let vm = eval_vm(&module, "Eval");
    assert_same_value(tree, vm.result, "nested default VM vs tree");
    assert_eq!(
        vm.func_apply_count, 2,
        "only the two navigation applies remain"
    );
}

#[test]
fn vm_except_at_free_nested_missing_boundary_is_exact() {
    let final_missing = parse_module(
        "\
---- MODULE VmExceptAtFreeFinalMissing ----
Eval == [[i \\in {1} |-> [j \\in {1} |-> 11]] EXCEPT ![1][2] = 99]
====",
    );
    let navigation_missing = parse_module(
        "\
---- MODULE VmExceptAtFreeNavigationMissing ----
Eval == [[i \\in {1} |-> [j \\in {1} |-> 11]] EXCEPT ![2][1] = 99]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree_final =
        eval_tree(&final_missing, "Eval").expect("missing final key should be a no-op");
    let vm_final = eval_vm(&final_missing, "Eval");
    assert_same_value(
        Ok(tree_final),
        vm_final.result,
        "missing final Index default-VM no-op",
    );
    assert_eq!(vm_final.func_apply_count, 1);

    let _tree_navigation = eval_tree(&navigation_missing, "Eval")
        .expect("tree-walker short-circuits a missing navigation key");
    let vm_navigation = eval_vm(&navigation_missing, "Eval");
    let vm_error = vm_navigation
        .result
        .expect_err("N-1 navigation apply must remain eager");
    assert!(matches!(vm_error, EvalError::NotInDomain { .. }));
    assert_eq!(vm_navigation.func_apply_count, 1);
}

#[test]
fn vm_except_at_free_preserves_duplicate_spec_order_when_later_rhs_uses_at() {
    let module = parse_module(
        "\
---- MODULE VmExceptAtFreeDuplicateOrder ----
Eval == [[i \\in {1, 2} |-> i * 10] EXCEPT ![1] = 9, ![1] = @ + 1]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree = eval_tree(&module, "Eval");
    let vm = eval_vm(&module, "Eval");
    let vm_value = assert_same_value(tree, vm.result, "duplicate-spec default VM");
    assert_eq!(
        vm_value
            .as_func()
            .and_then(|function| function.apply(&Value::int(1))),
        Some(&Value::int(10)),
        "the second @ must observe the first update's value 9"
    );
    assert_eq!(
        vm.func_apply_count, 1,
        "only the first at-free RHS may omit @"
    );
}

#[test]
fn vm_except_at_free_preserves_nested_except_at_binding_context() {
    let module = parse_module(
        "\
---- MODULE VmExceptAtFreeNestedAtContext ----
Base == [i \\in {1} |-> [j \\in {1} |-> 1]]
Eval == [Base EXCEPT ![1] = [@ EXCEPT ![1] = 9]]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree = eval_tree(&module, "Eval");
    let vm = eval_vm(&module, "Eval");
    assert_same_value(tree, vm.result, "nested-@ default VM");
    assert_eq!(
        vm.func_apply_count, 1,
        "outer @ remains while the inner at-free RHS omits only its own @"
    );
}

#[test]
fn vm_except_at_free_refuses_an_earlier_field_in_a_nested_path() {
    let module = parse_module(
        "\
---- MODULE VmExceptAtFreeEarlierFieldRefusal ----
Eval == [[slots |-> <<1, 2>>] EXCEPT !.slots[1] = 9]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree = eval_tree(&module, "Eval");
    let vm = eval_vm(&module, "Eval");
    assert_same_value(tree, vm.result, "earlier-Field default VM");
    assert_eq!(
        vm.func_apply_count, 2,
        "any Field in the path must refuse omission"
    );
}

#[test]
fn vm_except_at_free_refuses_final_field_on_function_value() {
    let module = parse_module(
        "\
---- MODULE VmExceptAtFreeFieldRefusal ----
Eval == [[i \\in {1, 2} |-> i] EXCEPT !.x = 9]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree_error = eval_tree(&module, "Eval").expect_err("Field on Func must error");
    let vm = eval_vm(&module, "Eval");
    let vm_error = vm
        .result
        .expect_err("final Field must be refused by the optimization");

    assert!(matches!(tree_error, EvalError::Internal { .. }));
    assert!(matches!(vm_error, EvalError::NotInDomain { .. }));
    assert_eq!(vm.func_apply_count, 1, "final Field must retain @");
}

#[test]
fn vm_except_at_free_refuses_direct_except_at_and_preserves_its_error() {
    let existing = parse_module(
        "\
---- MODULE VmExceptAtFreeDirectAt ----
Eval == [[a |-> 7] EXCEPT !.a = @]
====",
    );
    let missing = parse_module(
        "\
---- MODULE VmExceptAtFreeDirectAtMissing ----
Eval == [[a |-> 7] EXCEPT !.missing = @]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree_existing = eval_tree(&existing, "Eval");
    let vm = eval_vm(&existing, "Eval");
    assert_same_value(tree_existing, vm.result, "direct @ existing-key fallback");
    assert_eq!(vm.func_apply_count, 1, "direct @ must be refused");

    let tree_missing = eval_tree(&missing, "Eval")
        .expect("tree-walker must not evaluate @ when the field is absent");
    let vm_missing = eval_vm(&missing, "Eval");
    let vm_error = vm_missing
        .result
        .expect_err("refused RHS must retain generic VM behavior");
    assert!(matches!(vm_error, EvalError::NotInDomain { .. }));
    assert_eq!(vm_missing.func_apply_count, 1);
    assert_eq!(
        tree_missing,
        Value::Record(tla_value::RecordValue::from_sorted_entries(vec![(
            intern_name("a"),
            Value::int(7),
        )]))
    );
}

#[test]
fn vm_except_at_free_invalid_base_and_key_keep_errors_but_change_precedence() {
    let invalid_base = parse_module(
        "\
---- MODULE VmExceptAtFreeInvalidBase ----
Update(b) == [b EXCEPT ![1] = 9]
Eval == Update(42)
====",
    );
    let invalid_key = parse_module(
        "\
---- MODULE VmExceptAtFreeInvalidKey ----
Eval == [[a |-> 1] EXCEPT ![1] = 9]
====",
    );
    let _guard = enable_bytecode_vm_for_test();

    let tree_base = eval_tree(&invalid_base, "Eval").expect_err("invalid base must error");
    assert!(matches!(tree_base, EvalError::TypeError { .. }));
    let vm_base = eval_vm(&invalid_base, "Eval");
    let vm_base_error = vm_base
        .result
        .expect_err("FuncExcept must reject an invalid base");
    assert!(
        matches!(vm_base_error, EvalError::Internal { ref message, .. }
            if message == "bytecode VM type error: expected function-like value for EXCEPT, got 42"),
        "unexpected default-VM invalid-base error: {vm_base_error:?}"
    );
    assert_eq!(vm_base.chunk_func_apply_count, 0);

    let tree_key = eval_tree(&invalid_key, "Eval").expect_err("invalid key must error");
    assert!(matches!(tree_key, EvalError::TypeError { .. }));
    let vm_key = eval_vm(&invalid_key, "Eval");
    let vm_key_error = vm_key
        .result
        .expect_err("FuncExcept must reject an invalid record key");
    assert!(
        matches!(vm_key_error, EvalError::Internal { ref message, .. }
            if message == "bytecode VM type error: expected string field name for record EXCEPT, got 1"),
        "unexpected default-VM invalid-key error: {vm_key_error:?}"
    );
    assert_eq!(vm_key.func_apply_count, 0);
}
