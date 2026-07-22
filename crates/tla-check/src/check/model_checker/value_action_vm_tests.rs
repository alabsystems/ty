// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use tla_value::Rp;

use super::*;
use crate::config::Config;
use crate::test_support::parse_module;
use crate::value::{FuncBuilder, SetPredValue};
use crate::{CheckResult, Value};
use tla_core::ast::{BoundVar, Expr, ExprLabel};
use tla_core::{intern_name, Spanned};
use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

fn metadata(name: &str) -> ActionInstanceMeta {
    ActionInstanceMeta {
        name: Some(name.to_string()),
        bindings: Vec::new(),
        formal_bindings: Vec::new(),
        expr: None,
    }
}

fn request_value_action_vm_for_test(checker: &mut ModelChecker<'_>) {
    checker.value_action_vm.requested = true;
    checker.value_action_vm.auto_candidate = false;
    checker.value_action_vm.auto_selected = false;
    checker.value_action_vm.auto_activated = false;
    checker.value_action_vm.ctx_free_requested = true;
    checker.value_action_vm.register_reuse_requested = true;
    checker.value_action_vm.first_guard_requested = true;
}

#[test]
fn value_action_vm_flags_preserve_explicit_modes_and_auto_kill_switches() {
    use std::ffi::OsStr;

    assert!(resolve_value_action_vm_flag(None, true));
    assert!(!resolve_value_action_vm_flag(None, false));
    assert!(resolve_value_action_vm_flag(Some(OsStr::new("1")), false));
    for disabled in ["0", "false", "true", "", " 1 "] {
        assert!(
            !resolve_value_action_vm_flag(Some(OsStr::new(disabled)), true),
            "present non-exact-1 value {disabled:?} must override AUTO"
        );
    }
}

#[test]
fn dormant_auto_candidate_requires_selection_and_concrete_activation() {
    let mut dispatch = ValueActionVmDispatch::from_env();
    dispatch.requested = false;
    dispatch.auto_candidate = true;
    dispatch.auto_selected = false;
    dispatch.auto_activated = false;
    dispatch.install_plan(ValueActionVmPlan {
        entries: Vec::new(),
        split_instance_count: 0,
        linked_chunk: BytecodeChunk::new(),
        canonical_vars: Some(Vec::new()),
        uniform_slot_guard_index: None,
        self_recursive_helper_count: 0,
        self_recursive_call_site_count: 0,
    });

    assert!(dispatch.plan_requested());
    assert!(!dispatch.is_armed());
    assert!(!dispatch.activate_auto_candidate());
    dispatch.select_auto_candidate();
    assert!(dispatch.auto_selected());
    assert!(dispatch.activate_auto_candidate());
    assert!(dispatch.is_armed());
    assert!(!dispatch.activate_auto_candidate());
}

#[test]
fn discarding_auto_candidate_releases_only_dormant_state() {
    let install_empty_plan = |dispatch: &mut ValueActionVmDispatch| {
        dispatch.install_plan(ValueActionVmPlan {
            entries: Vec::new(),
            split_instance_count: 0,
            linked_chunk: BytecodeChunk::new(),
            canonical_vars: Some(Vec::new()),
            uniform_slot_guard_index: None,
            self_recursive_helper_count: 0,
            self_recursive_call_site_count: 0,
        });
    };

    let mut dormant = ValueActionVmDispatch::from_env();
    dormant.requested = false;
    dormant.auto_candidate = true;
    install_empty_plan(&mut dormant);
    dormant.select_auto_candidate();
    dormant.discard_auto_candidate();
    assert!(!dormant.plan_requested());
    assert!(!dormant.auto_selected());
    assert!(dormant.plan.is_none());

    let mut explicit = ValueActionVmDispatch::from_env();
    explicit.requested = true;
    explicit.auto_candidate = true;
    install_empty_plan(&mut explicit);
    explicit.discard_auto_candidate();
    assert!(explicit.requested());
    assert!(explicit.is_armed());
    assert!(explicit.plan.is_some());
}

const COVERAGE_ROUTE_SPEC: &str = r#"
---- MODULE ValueActionVmCoverageRoute ----
VARIABLE x

A == /\ x = 0
     /\ x' = 1

B == /\ x = 1
     /\ x' = 2

C == /\ x = 99
     /\ x' = 100

Init == x = 0
Next == A \/ B \/ C
====
"#;

fn coverage_route_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        check_deadlock: false,
        auto_por: Some(false),
        use_flat_state: Some(false),
        use_compiled_bfs: Some(false),
        ..Default::default()
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn default_dead_action_tracking_yields_to_armed_vm_in_full_bfs_route() {
    let module = parse_module(COVERAGE_ROUTE_SPEC);
    let config = coverage_route_config();
    let mut checker = ModelChecker::new(&module, &config);
    checker.set_force_explicit_bfs(true);
    checker.set_trust_cg_structural_veto();
    request_value_action_vm_for_test(&mut checker);
    checker.set_default_dead_action_coverage();

    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("Value VM coverage-route check should succeed, got {other:?}"),
    };

    assert_eq!(stats.states_found, 3);
    assert!(
        checker.value_action_vm.stats.candidate_parents > 0,
        "the production BFS route must deliver parents to an armed Value VM",
    );
    assert!(checker.value_action_vm.stats.entry_evals > 0);
    assert!(checker.value_action_vm.is_armed());
    assert_eq!(checker.value_action_vm.stats.shadow_mismatches, 0);
    assert!(!checker.coverage.collect);
    assert!(!checker.coverage.default_dead_action_tracking);
    assert!(stats.coverage.is_none());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn explicit_coverage_keeps_action_attribution_and_bypasses_vm() {
    let module = parse_module(COVERAGE_ROUTE_SPEC);
    let config = coverage_route_config();
    let mut checker = ModelChecker::new(&module, &config);
    checker.set_force_explicit_bfs(true);
    checker.set_trust_cg_structural_veto();
    request_value_action_vm_for_test(&mut checker);
    checker.set_collect_coverage(true);
    checker.set_default_dead_action_coverage();

    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("explicit coverage route should succeed, got {other:?}"),
    };
    assert_eq!(stats.states_found, 3);
    assert_eq!(checker.value_action_vm.stats.candidate_parents, 0);
    assert!(!checker.coverage.default_dead_action_tracking);

    let coverage = stats
        .coverage
        .expect("explicit coverage must retain per-action evidence");
    let transitions = |name: &str| {
        coverage
            .actions
            .values()
            .find(|action| action.name == name)
            .map(|action| action.transitions)
            .unwrap_or(0)
    };
    assert_eq!(transitions("A"), 1);
    assert_eq!(transitions("B"), 1);
    assert_eq!(transitions("C"), 0);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn track_only_coverage_keeps_dead_action_evidence_and_bypasses_vm() {
    let module = parse_module(COVERAGE_ROUTE_SPEC);
    let config = coverage_route_config();
    let mut checker = ModelChecker::new(&module, &config);
    checker.set_force_explicit_bfs(true);
    checker.set_trust_cg_structural_veto();
    request_value_action_vm_for_test(&mut checker);
    checker.set_default_dead_action_coverage();
    checker.set_track_coverage(true);

    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("track-only coverage route should succeed, got {other:?}"),
    };
    assert_eq!(stats.states_found, 3);
    assert_eq!(checker.value_action_vm.stats.candidate_parents, 0);
    assert!(!checker.coverage.default_dead_action_tracking);
    assert!(stats.coverage.is_none());

    let dead_actions = stats.vacuity_warnings.iter().find_map(|warning| match warning {
        crate::VacuityWarning::DeadActions(names) => Some(names.as_slice()),
        _ => None,
    });
    assert_eq!(dead_actions, Some(["C".to_string()].as_slice()));
}

#[test]
fn explicit_tracking_modes_clear_default_dead_action_provenance() {
    let module = parse_module(COVERAGE_ROUTE_SPEC);
    let config = coverage_route_config();
    let mut checker = ModelChecker::new(&module, &config);
    checker.set_trust_cg_structural_veto();

    checker.set_default_dead_action_coverage();
    assert!(checker.coverage.default_dead_action_tracking);
    checker.coverage.native_fast_path_skipped = true;
    checker.set_track_coverage(false);
    assert!(!checker.coverage.default_dead_action_tracking);
    assert!(!checker.coverage.native_fast_path_skipped);

    checker.set_default_dead_action_coverage();
    assert!(checker.coverage.default_dead_action_tracking);
    checker.coverage.native_fast_path_skipped = true;
    checker.set_collect_coverage(false);
    assert!(!checker.coverage.default_dead_action_tracking);
    assert!(!checker.coverage.native_fast_path_skipped);

    checker.set_default_dead_action_coverage();
    assert!(checker.coverage.default_dead_action_tracking);
    checker.coverage.native_fast_path_skipped = true;
    checker.set_coverage_guided(false, 8);
    assert!(!checker.coverage.default_dead_action_tracking);
    assert!(!checker.coverage.native_fast_path_skipped);

    checker.set_track_coverage(true);
    checker.set_default_dead_action_coverage();
    assert!(checker.coverage.collect);
    assert!(!checker.coverage.default_dead_action_tracking);
    assert!(!checker.coverage.native_fast_path_skipped);

    checker.set_coverage_guided(true, 8);
    checker.set_default_dead_action_coverage();
    assert!(checker.coverage.collect);
    assert!(checker.coverage.coverage_guided);
    assert!(!checker.coverage.default_dead_action_tracking);
    assert!(!checker.coverage.native_fast_path_skipped);
}

fn metadata_with_expr(name: &str, expr: Spanned<Expr>) -> ActionInstanceMeta {
    let mut metadata = metadata(name);
    metadata.expr = Some(expr);
    metadata
}

fn install_uniform_slot_guards_for_test(
    plan: &mut ValueActionVmPlan,
    guards: &[(VarIndex, Value)],
) {
    assert_eq!(plan.entries.len(), guards.len());
    for (entry, (var_idx, expected)) in plan.entries.iter_mut().zip(guards) {
        entry.first_guard = Some(ValueActionVmFirstGuard::SlotEq {
            var_idx: *var_idx,
            expected: expected.clone(),
        });
    }
    plan.uniform_slot_guard_index = ValueActionVmUniformSlotGuardIndex::build(&plan.entries);
}

fn certify_first_guard_for_test(
    action: &ActionInstanceMeta,
    ctx: &EvalCtx,
) -> Option<ValueActionVmFirstGuard> {
    let complete_bindings = action
        .bindings
        .iter()
        .chain(&action.formal_bindings)
        .cloned()
        .collect::<Vec<_>>();
    certify_first_guard_with_complete_bindings_for_test(action, &complete_bindings, ctx)
}

fn certify_first_guard_with_complete_bindings_for_test(
    action: &ActionInstanceMeta,
    complete_bindings: &[(std::sync::Arc<str>, Value)],
    ctx: &EvalCtx,
) -> Option<ValueActionVmFirstGuard> {
    let scopes = vec![complete_bindings.to_vec()];
    let globally_bindable_names =
        collect_first_guard_globally_bindable_names(std::slice::from_ref(action), &scopes, ctx);
    certify_value_action_vm_first_guard_exact_body(
        action,
        complete_bindings,
        ctx,
        &globally_bindable_names,
    )
}

fn conjunction(first: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::And(
        Box::new(first),
        Box::new(Spanned::dummy(Expr::Bool(true))),
    ))
}

fn direct_guard_expr(
    var_name: &str,
    raw_idx: u16,
    var_name_id: tla_core::NameId,
    expected_name: &str,
    reversed: bool,
) -> Spanned<Expr> {
    let state_var = Spanned::dummy(Expr::StateVar(var_name.to_string(), raw_idx, var_name_id));
    let expected = Spanned::dummy(Expr::Ident(
        expected_name.to_string(),
        intern_name(expected_name),
    ));
    let equality = if reversed {
        Expr::Eq(Box::new(expected), Box::new(state_var))
    } else {
        Expr::Eq(Box::new(state_var), Box::new(expected))
    };
    conjunction(Spanned::dummy(equality))
}

fn transformed_write(name: &str, value: i64, arity: u8) -> BytecodeFunction {
    let mut function = BytecodeFunction::new(name.to_string(), arity);
    function.emit(Opcode::LoadImm { rd: 0, value });
    function.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    function.emit(Opcode::LoadBool { rd: 1, value: true });
    function.emit(Opcode::Ret { rs: 1 });
    function
}

fn transformed_type_error(name: &str) -> BytecodeFunction {
    let mut function = BytecodeFunction::new(name.to_string(), 0);
    function.emit(Opcode::LoadBool { rd: 0, value: true });
    function.emit(Opcode::LoadImm { rd: 1, value: 1 });
    function.emit(Opcode::AddInt {
        rd: 2,
        r1: 0,
        r2: 1,
    });
    function.emit(Opcode::StoreVar { var_idx: 0, rs: 2 });
    function.emit(Opcode::LoadBool { rd: 3, value: true });
    function.emit(Opcode::Ret { rs: 3 });
    function
}

fn countdown_helper(name: &str, name_idx: u16) -> BytecodeFunction {
    let mut function = BytecodeFunction::new(name.to_string(), 1);
    function.emit(Opcode::LoadImm { rd: 1, value: 0 });
    function.emit(Opcode::Eq {
        rd: 2,
        r1: 0,
        r2: 1,
    });
    function.emit(Opcode::JumpFalse { rs: 2, offset: 3 });
    function.emit(Opcode::LoadImm { rd: 3, value: 7 });
    function.emit(Opcode::Ret { rs: 3 });
    function.emit(Opcode::LoadImm { rd: 1, value: 1 });
    function.emit(Opcode::SubInt {
        rd: 2,
        r1: 0,
        r2: 1,
    });
    function.emit(Opcode::CallExternal {
        rd: 3,
        name_idx,
        args_start: 2,
        argc: 1,
        self_recursive: true,
    });
    function.emit(Opcode::Ret { rs: 3 });
    function
}

fn transformed_helper_call(name: &str, helper_idx: u16, argument: i64) -> BytecodeFunction {
    let mut function = BytecodeFunction::new(name.to_string(), 0);
    function.emit(Opcode::LoadImm {
        rd: 0,
        value: argument,
    });
    function.emit(Opcode::Call {
        rd: 1,
        op_idx: helper_idx,
        args_start: 0,
        argc: 1,
    });
    function.emit(Opcode::StoreVar { var_idx: 0, rs: 1 });
    function.emit(Opcode::LoadBool { rd: 2, value: true });
    function.emit(Opcode::Ret { rs: 2 });
    function
}

fn context_equality_write(name: &str, value: i64) -> BytecodeFunction {
    let mut function = BytecodeFunction::new(name.to_string(), 0);
    function.emit(Opcode::LoadImm { rd: 0, value });
    function.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    function.emit(Opcode::LoadConst { rd: 1, idx: 0 });
    function.emit(Opcode::LoadConst { rd: 2, idx: 1 });
    function.emit(Opcode::Eq {
        rd: 3,
        r1: 1,
        r2: 2,
    });
    function.emit(Opcode::Ret { rs: 3 });
    function
}

fn parent_sensitive_setpred(expected: i64) -> Value {
    let predicate = Spanned::dummy(Expr::Eq(
        Box::new(Spanned::dummy(Expr::StateVar(
            "x".to_string(),
            0,
            intern_name("x"),
        ))),
        Box::new(Spanned::dummy(Expr::Int(expected.into()))),
    ));
    let bound = BoundVar {
        name: Spanned::dummy("elem".to_string()),
        domain: None,
        pattern: None,
    };
    Value::SetPred(Box::new(SetPredValue::new(
        Value::set([Value::int(1)]),
        bound,
        predicate,
        Default::default(),
        None,
        None,
    )))
}

fn compiled_bytecode(
    functions: Vec<BytecodeFunction>,
    indices: &[(&str, u16)],
) -> CompiledBytecode {
    let mut chunk = BytecodeChunk::new();
    for function in functions {
        chunk.add_function(function);
    }
    let mut op_indices = rustc_hash::FxHashMap::default();
    for &(name, func_idx) in indices {
        op_indices.insert(name.to_string(), func_idx);
    }
    CompiledBytecode {
        chunk,
        op_indices,
        failed: Vec::new(),
    }
}

#[test]
fn plan_follows_metadata_and_numeric_arm_order_not_map_or_function_order() {
    let metadata = vec![metadata("B"), metadata("A"), metadata("A"), metadata("C")];
    let bytecode = compiled_bytecode(
        vec![
            transformed_write("A#d0", 10, 0),
            transformed_write("C", 30, 0),
            transformed_write("B", 20, 0),
            transformed_write("A#d1", 11, 0),
        ],
        &[("A#d1", 3), ("C", 1), ("B", 2), ("A#d0", 0)],
    );

    let plan = ValueActionVmPlan::build(&metadata, &bytecode, 1)
        .expect("all four final entries should certify");
    assert_eq!(
        plan.entries
            .iter()
            .map(|entry| (entry.label.as_str(), entry.func_idx))
            .collect::<Vec<_>>(),
        vec![("B", 2), ("A#d0", 0), ("A#d1", 3), ("C", 1)]
    );
    assert_eq!(
        plan.entries
            .iter()
            .map(|entry| entry.metadata_idx)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3],
        "numeric function reconstruction must retain exact source occurrences"
    );
}

#[test]
fn plan_links_strict_helper_self_recursion_only_in_its_private_chunk() {
    let mut bytecode = compiled_bytecode(
        vec![
            countdown_helper("Countdown", 0),
            transformed_helper_call("A", 0, 3),
        ],
        &[("A", 1)],
    );
    let name_idx = bytecode
        .chunk
        .constants
        .add_value(Value::string("Countdown"));
    assert_eq!(name_idx, 0);

    let plan = ValueActionVmPlan::build(&[metadata("A")], &bytecode, 1)
        .expect("a pure strict-self helper should be linked and certified");
    assert_eq!(plan.self_recursive_helper_count, 1);
    assert_eq!(plan.self_recursive_call_site_count, 1);
    assert!(matches!(
        bytecode.chunk.functions[0].instructions[7],
        Opcode::CallExternal { .. }
    ));
    let linked = &plan.linked_chunk;
    assert!(matches!(
        linked.functions[0].instructions[7],
        Opcode::Call {
            op_idx: 0,
            argc: 1,
            ..
        }
    ));

    let module = parse_module(
        r#"
---- MODULE ValueActionVmRecursiveHelper ----
VARIABLE x
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let current = ArrayState::from_values(vec![Value::int(0)]);
    let (result, stats) = execute_value_action_vm_plan_attempt(
        &plan,
        &mut checker.ctx,
        &current,
        false,
        false,
        false,
    );
    let result = result.expect("linked helper recursion should execute in the Value VM");
    assert_eq!(stats.entry_evals, 1);
    assert_eq!(result.successors.len(), 1);
    assert_eq!(
        result.successors[0]
            .materialize(&current, checker.ctx.var_registry())
            .materialize_values(),
        vec![Value::int(7)]
    );
}

#[test]
fn plan_links_duplicate_same_named_helpers_to_their_containing_indices() {
    let mut first = BytecodeFunction::new("Rec".to_string(), 0);
    first.emit(Opcode::CallExternal {
        rd: 0,
        name_idx: 0,
        args_start: 0,
        argc: 0,
        self_recursive: true,
    });
    first.emit(Opcode::Ret { rs: 0 });
    let second = first.clone();

    let action = |name: &str, helper_idx: u16| {
        let mut function = BytecodeFunction::new(name.to_string(), 0);
        function.emit(Opcode::Call {
            rd: 0,
            op_idx: helper_idx,
            args_start: 0,
            argc: 0,
        });
        function.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
        function.emit(Opcode::LoadBool { rd: 1, value: true });
        function.emit(Opcode::Ret { rs: 1 });
        function
    };
    let mut bytecode = compiled_bytecode(
        vec![first, second, action("A", 0), action("B", 1)],
        &[("A", 2), ("B", 3)],
    );
    assert_eq!(bytecode.chunk.constants.add_value(Value::string("Rec")), 0);

    let plan = ValueActionVmPlan::build(&[metadata("A"), metadata("B")], &bytecode, 1)
        .expect("each same-named helper copy should link relative to itself");
    let linked = &plan.linked_chunk;
    assert_eq!(plan.self_recursive_helper_count, 2);
    assert!(matches!(
        linked.functions[0].instructions[0],
        Opcode::Call { op_idx: 0, .. }
    ));
    assert!(matches!(
        linked.functions[1].instructions[0],
        Opcode::Call { op_idx: 1, .. }
    ));
}

#[test]
fn plan_keeps_entry_and_nonexact_externals_ineligible() {
    let mut entry = BytecodeFunction::new("Entry".to_string(), 0);
    entry.emit(Opcode::CallExternal {
        rd: 0,
        name_idx: 0,
        args_start: 0,
        argc: 0,
        self_recursive: true,
    });
    entry.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    entry.emit(Opcode::LoadBool { rd: 1, value: true });
    entry.emit(Opcode::Ret { rs: 1 });
    let mut entry_bytecode = compiled_bytecode(vec![entry], &[("Entry", 0)]);
    assert_eq!(
        entry_bytecode
            .chunk
            .constants
            .add_value(Value::string("Entry")),
        0
    );
    let reason = ValueActionVmPlan::build(&[metadata("Entry")], &entry_bytecode, 1)
        .expect_err("strict self-recursion in an action entry must stay rejected");
    assert!(reason.contains("unsupported CallExternal"), "{reason}");

    let mut helper = BytecodeFunction::new("Helper".to_string(), 1);
    helper.emit(Opcode::CallExternal {
        rd: 0,
        name_idx: 0,
        args_start: 0,
        argc: 1,
        self_recursive: true,
    });
    helper.emit(Opcode::Ret { rs: 0 });
    let mut helper_bytecode = compiled_bytecode(
        vec![helper, transformed_helper_call("A", 0, 1)],
        &[("A", 1)],
    );
    assert_eq!(
        helper_bytecode
            .chunk
            .constants
            .add_value(Value::string("Different")),
        0
    );
    let reason = ValueActionVmPlan::build(&[metadata("A")], &helper_bytecode, 1)
        .expect_err("a different-name helper external must stay rejected");
    assert!(reason.contains("unsupported CallExternal"), "{reason}");

    for (case, constant, name_idx, external_arity, self_recursive) in [
        (
            "unauthenticated same name",
            Some(Value::string("Helper")),
            0,
            1,
            false,
        ),
        ("wrong arity", Some(Value::string("Helper")), 0, 0, true),
        ("non-string name", Some(Value::int(1)), 0, 1, true),
        ("out-of-range name", None, 17, 1, true),
    ] {
        let mut helper = BytecodeFunction::new("Helper".to_string(), 1);
        helper.emit(Opcode::CallExternal {
            rd: 0,
            name_idx,
            args_start: 0,
            argc: external_arity,
            self_recursive,
        });
        helper.emit(Opcode::Ret { rs: 0 });
        let mut bytecode = compiled_bytecode(
            vec![helper, transformed_helper_call("A", 0, 1)],
            &[("A", 1)],
        );
        if let Some(constant) = constant {
            assert_eq!(bytecode.chunk.constants.add_value(constant), 0);
        }
        let reason = ValueActionVmPlan::build(&[metadata("A")], &bytecode, 1)
            .expect_err("a nonexact helper external must stay rejected");
        assert!(
            reason.contains("unsupported CallExternal"),
            "{case}: {reason}"
        );
    }
}

#[test]
fn linked_recursive_helper_must_still_be_pure() {
    let mut helper = BytecodeFunction::new("Helper".to_string(), 1);
    helper.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    helper.emit(Opcode::CallExternal {
        rd: 0,
        name_idx: 0,
        args_start: 0,
        argc: 1,
        self_recursive: true,
    });
    helper.emit(Opcode::Ret { rs: 0 });
    let mut bytecode = compiled_bytecode(
        vec![helper, transformed_helper_call("A", 0, 1)],
        &[("A", 1)],
    );
    assert_eq!(
        bytecode.chunk.constants.add_value(Value::string("Helper")),
        0
    );

    let reason = ValueActionVmPlan::build(&[metadata("A")], &bytecode, 1)
        .expect_err("linking recursion must not bypass helper-purity validation");
    assert!(reason.contains("writes successor state"), "{reason}");
    assert!(matches!(
        bytecode.chunk.functions[0].instructions[1],
        Opcode::CallExternal { .. }
    ));
}

#[test]
fn unreachable_strict_self_external_is_left_unlinked() {
    let mut helper = BytecodeFunction::new("Helper".to_string(), 1);
    helper.emit(Opcode::LoadImm { rd: 0, value: 9 });
    helper.emit(Opcode::Ret { rs: 0 });
    helper.emit(Opcode::CallExternal {
        rd: 0,
        name_idx: 0,
        args_start: 0,
        argc: 1,
        self_recursive: true,
    });
    let mut bytecode = compiled_bytecode(
        vec![helper, transformed_helper_call("A", 0, 1)],
        &[("A", 1)],
    );
    assert_eq!(
        bytecode.chunk.constants.add_value(Value::string("Helper")),
        0
    );

    let plan = ValueActionVmPlan::build(&[metadata("A")], &bytecode, 1)
        .expect("dead dynamic code must not affect reachable eligibility");
    assert_eq!(plan.self_recursive_helper_count, 0);
    assert!(matches!(
        plan.linked_chunk.functions[0].instructions[2],
        Opcode::CallExternal { .. }
    ));
}

#[test]
fn first_guard_certifies_direct_and_reversed_constants_but_rejects_stale_slots() {
    let mut ctx = EvalCtx::new();
    ctx.register_vars(["state"]);
    let state_idx = ctx.var_registry().get("state").unwrap();
    let state_name_id = ctx.var_registry().name_id_at(state_idx);
    let ready = Value::ModelValue(Rp::from("ready"));
    std::sync::Arc::make_mut(ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("READY"), ready.clone());

    for reversed in [false, true] {
        let action = metadata_with_expr(
            "A",
            direct_guard_expr("state", state_idx.0, state_name_id, "READY", reversed),
        );
        let guard = certify_first_guard_for_test(&action, &ctx)
            .expect("both equality orientations should certify");
        assert!(!guard.mismatches(&ArrayState::from_values(vec![ready.clone()])));
        assert!(
            guard.mismatches(&ArrayState::from_values(vec![Value::ModelValue(
                Rp::from("busy")
            )]))
        );
    }

    let stale_slot = metadata_with_expr(
        "A",
        direct_guard_expr("state", 1, state_name_id, "READY", false),
    );
    assert!(certify_first_guard_for_test(&stale_slot, &ctx).is_none());

    let stale_name_id = metadata_with_expr(
        "A",
        direct_guard_expr(
            "state",
            state_idx.0,
            intern_name("not_state"),
            "READY",
            false,
        ),
    );
    assert!(certify_first_guard_for_test(&stale_name_id, &ctx).is_none());

    let not_first = metadata_with_expr(
        "A",
        Spanned::dummy(Expr::And(
            Box::new(Spanned::dummy(Expr::Bool(true))),
            Box::new(direct_guard_expr(
                "state",
                state_idx.0,
                state_name_id,
                "READY",
                false,
            )),
        )),
    );
    assert!(certify_first_guard_for_test(&not_first, &ctx).is_none());
}

#[test]
fn first_guard_uses_complete_split_binding_order_not_pruned_metadata_order() {
    let mut ctx = EvalCtx::new();
    ctx.register_vars(["x"]);
    let x_idx = ctx.var_registry().get("x").unwrap();
    let expr = direct_guard_expr(
        "x",
        x_idx.0,
        ctx.var_registry().name_id_at(x_idx),
        "phase",
        false,
    );

    let mut action = metadata_with_expr("A", expr);
    action.bindings = vec![
        (std::sync::Arc::from("phase"), Value::int(1)),
        (std::sync::Arc::from("phase"), Value::int(2)),
    ];
    action.formal_bindings = vec![(std::sync::Arc::from("phase"), Value::int(1))];
    let complete_bindings = vec![
        (std::sync::Arc::from("phase"), Value::int(1)),
        (std::sync::Arc::from("phase"), Value::int(2)),
    ];
    let guard =
        certify_first_guard_with_complete_bindings_for_test(&action, &complete_bindings, &ctx)
            .expect("the unpruned leaf scope should certify");
    assert!(!guard.mismatches(&ArrayState::from_values(vec![Value::int(2)])));
    assert!(
        guard.mismatches(&ArrayState::from_values(vec![Value::int(1)])),
        "the inner witness must shadow the same-named action formal"
    );
}

#[test]
fn first_guard_rejects_when_synthetic_alias_pruning_changes_leaf_lookup() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardAliasPruning ----
CONSTANT p
VARIABLE state
A(p, q) == \E p \in {2} : B(q)
B(b) ==
  /\ state = p
  /\ state' = state
Next == A(1, 2)
====
"#,
    );
    let config = Config::default();
    let checker = ModelChecker::new(&module, &config);
    let next = checker.ctx.get_op("Next").unwrap();
    let instances = crate::action_instance::split_action_instances(&checker.ctx, &next.body)
        .expect("the literal call and bounded EXISTS should split");
    assert_eq!(instances.len(), 1);
    let instance = &instances[0];
    assert_eq!(instance.name.as_deref(), Some("B"));
    assert_eq!(
        instance
            .bindings
            .iter()
            .filter(|(name, _)| name.as_ref() == "p")
            .map(|(_, value)| value.as_i64())
            .collect::<Vec<_>>(),
        vec![Some(1)],
        "dispatch-key pruning should demonstrate the missing inner alias"
    );
    assert_eq!(
        instance
            .complete_bindings
            .iter()
            .filter(|(name, _)| name.as_ref() == "p")
            .map(|(_, value)| value.as_i64())
            .collect::<Vec<_>>(),
        vec![Some(1), Some(2)],
        "the certificate sidecar must retain exact lexical order"
    );

    let action = ActionInstanceMeta {
        name: instance.name.clone(),
        bindings: instance.bindings.clone(),
        formal_bindings: instance.formal_bindings.clone(),
        expr: Some(instance.expr.clone()),
    };
    assert!(
        certify_first_guard_with_complete_bindings_for_test(
            &action,
            &instance.complete_bindings,
            &checker.ctx,
        )
        .is_none(),
        "a source/synthetic binding mismatch must disable the optimization"
    );
}

#[test]
fn first_guard_model_value_local_requires_uncaptured_exact_constant_roundtrip() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardModelCapture ----
VARIABLE state
A(actor) ==
  LET w1 == "captured"
  IN /\ state = actor
     /\ state' = state
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let actor = Value::ModelValue(Rp::from("w1"));
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("w1"), actor.clone());
    let mut action = metadata_with_expr("A", checker.ctx.get_op("A").unwrap().body.clone());
    action.formal_bindings = vec![(std::sync::Arc::from("actor"), actor)];
    assert!(
        certify_first_guard_for_test(&action, &checker.ctx).is_none(),
        "the synthetic Ident(w1) must not be captured by the source LET"
    );

    let mut state_ctx = EvalCtx::new();
    state_ctx.register_vars(["state", "w1"]);
    let state_idx = state_ctx.var_registry().get("state").unwrap();
    let actor = Value::ModelValue(Rp::from("w1"));
    std::sync::Arc::make_mut(state_ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("w1"), actor.clone());
    let mut action = metadata_with_expr(
        "A",
        direct_guard_expr(
            "state",
            state_idx.0,
            state_ctx.var_registry().name_id_at(state_idx),
            "actor",
            false,
        ),
    );
    action.formal_bindings = vec![(std::sync::Arc::from("actor"), actor)];
    assert!(
        certify_first_guard_for_test(&action, &state_ctx).is_none(),
        "state-var resolution must not capture the synthetic model atom"
    );
}

#[test]
fn first_guard_exact_body_provenance_ignores_only_rebuilt_root_span() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardExactBody ----
VARIABLE state
A ==
  LET unused == 0
  IN /\ state = "ready"
     /\ state' = state
Next == A
====
"#,
    );
    let config = Config::default();
    let checker = ModelChecker::new(&module, &config);
    let next = checker.ctx.get_op("Next").unwrap();
    let instances = crate::action_instance::split_action_instances(&checker.ctx, &next.body)
        .expect("the root action should split to one exact leaf");
    assert_eq!(instances.len(), 1);
    let instance = &instances[0];
    let shared = checker.ctx.get_op("A").unwrap();
    assert_ne!(instance.expr.span, shared.body.span);
    assert_eq!(instance.expr.node, shared.body.node);

    let action = ActionInstanceMeta {
        name: instance.name.clone(),
        bindings: instance.bindings.clone(),
        formal_bindings: instance.formal_bindings.clone(),
        expr: Some(instance.expr.clone()),
    };
    let scopes = vec![instance.complete_bindings.clone()];
    let bindable = collect_first_guard_globally_bindable_names(
        std::slice::from_ref(&action),
        &scopes,
        &checker.ctx,
    );
    assert!(
        certify_value_action_vm_first_guard(
            &action,
            &instance.complete_bindings,
            &checker.ctx,
            &bindable,
        )
        .is_some(),
        "the splitter's root-span-only LET reconstruction must retain provenance"
    );

    let mut wrapped = action.clone();
    wrapped.expr = Some(Spanned::dummy(Expr::And(
        Box::new(instance.expr.clone()),
        Box::new(Spanned::dummy(Expr::Bool(true))),
    )));
    let wrapped_scopes = vec![instance.complete_bindings.clone()];
    let wrapped_bindable = collect_first_guard_globally_bindable_names(
        std::slice::from_ref(&wrapped),
        &wrapped_scopes,
        &checker.ctx,
    );
    assert!(
        certify_value_action_vm_first_guard(
            &wrapped,
            &instance.complete_bindings,
            &checker.ctx,
            &wrapped_bindable,
        )
        .is_none(),
        "a real wrapper changes the Expr tree and must fail exact-body provenance"
    );
}

#[test]
fn first_guard_rejects_labeled_operator_expression_dispatch() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardLabeledOp ----
VARIABLE pc
Transition(t, from) ==
  /\ pc[t] = from
  /\ pc' = pc
====
"#,
    );
    let config = Config::default();
    let checker = ModelChecker::new(&module, &config);
    let labeled_op = Spanned::dummy(Expr::Label(ExprLabel {
        name: Spanned::dummy("op_label".to_string()),
        body: Box::new(Spanned::dummy(Expr::Ident(
            "Transition".to_string(),
            intern_name("Transition"),
        ))),
    }));
    let call = Spanned::dummy(Expr::Apply(
        Box::new(labeled_op),
        vec![
            Spanned::dummy(Expr::String("actor".to_string())),
            Spanned::dummy(Expr::String("ready".to_string())),
        ],
    ));
    let action = metadata_with_expr("Synthetic", conjunction(call));
    assert!(
        certify_first_guard_for_test(&action, &checker.ctx).is_none(),
        "canonical Apply dispatch accepts only a raw Ident operator expression"
    );
}

#[test]
fn first_guard_resolved_state_var_ignores_unrelated_binders_but_idents_stay_conservative() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardGlobalShadows ----
CONSTANT READY
VARIABLES pc, x
Transition(t, from, to) ==
  /\ pc[t] = from
  /\ pc' = [pc EXCEPT ![t] = to]
AliasSources(alias, pc, READY, Transition) == alias
TupleSources ==
  \E <<tuple_pc, tuple_ready, tuple_transition>>
      \in {<<"pc", "ready", "transition">>} : TRUE
StateLeaf ==
  /\ pc = "phase"
  /\ UNCHANGED <<pc, x>>
ConstantLeaf ==
  /\ x = READY
  /\ UNCHANGED <<pc, x>>
CallLeaf(actor) ==
  /\ Transition(actor, "from", "to")
  /\ UNCHANGED x
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("READY"), Value::string("ready"));

    let actions = ["StateLeaf", "ConstantLeaf", "CallLeaf"]
        .into_iter()
        .map(|name| metadata_with_expr(name, checker.ctx.get_op(name).unwrap().body.clone()))
        .collect::<Vec<_>>();
    let complete_bindings = vec![
        Vec::new(),
        Vec::new(),
        vec![(std::sync::Arc::from("actor"), Value::string("actor"))],
    ];
    let bindable =
        collect_first_guard_globally_bindable_names(&actions, &complete_bindings, &checker.ctx);
    for name in [
        "pc",
        "READY",
        "Transition",
        "tuple_pc",
        "tuple_ready",
        "tuple_transition",
    ] {
        assert!(bindable.contains(name), "missing binding source {name}");
    }

    assert!(
        certify_value_action_vm_first_guard(
            &actions[0],
            &complete_bindings[0],
            &checker.ctx,
            &bindable,
        )
        .is_some(),
        "an unrelated formal named pc cannot intercept an exact resolved StateVar(pc) read"
    );
    assert!(certify_value_action_vm_first_guard(
        &actions[1],
        &complete_bindings[1],
        &checker.ctx,
        &bindable,
    )
    .is_none());
    let mut call = actions[2].clone();
    call.formal_bindings = vec![(std::sync::Arc::from("actor"), Value::string("actor"))];
    assert!(certify_value_action_vm_first_guard(
        &call,
        &complete_bindings[2],
        &checker.ctx,
        &bindable,
    )
    .is_none());
}

#[test]
fn first_guard_resolved_state_var_still_rejects_action_local_binding_collision() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardLocalStateCollision ----
CONSTANT READY
VARIABLE state
A ==
  /\ state = READY
  /\ state' = state
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("READY"), Value::string("ready"));

    let mut action = metadata_with_expr("A", checker.ctx.get_op("A").unwrap().body.clone());
    action.bindings = vec![(std::sync::Arc::from("state"), Value::int(1))];
    let complete_bindings = vec![(std::sync::Arc::from("state"), Value::int(1))];
    let scopes = vec![complete_bindings.clone()];
    let bindable = collect_first_guard_globally_bindable_names(
        std::slice::from_ref(&action),
        &scopes,
        &checker.ctx,
    );

    assert!(
        certify_value_action_vm_first_guard(&action, &complete_bindings, &checker.ctx, &bindable,)
            .is_none(),
        "an action-local binding can rewrite StateVar(state) in the synthetic bytecode"
    );
}

#[test]
fn first_guard_rejects_active_instance_with_state_substitution() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardActiveInstance ----
CONSTANT READY
VARIABLES state, other
A ==
  /\ state = READY
  /\ UNCHANGED <<state, other>>
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("READY"), Value::string("ready"));

    let action = metadata_with_expr("A", checker.ctx.get_op("A").unwrap().body.clone());
    let complete_bindings = Vec::new();
    let scopes = vec![complete_bindings.clone()];
    let bindable = collect_first_guard_globally_bindable_names(
        std::slice::from_ref(&action),
        &scopes,
        &checker.ctx,
    );
    assert!(
        certify_value_action_vm_first_guard(&action, &complete_bindings, &checker.ctx, &bindable,)
            .is_some(),
        "the exact root action should certify before entering an INSTANCE scope"
    );

    let other_idx = checker.ctx.var_registry().get("other").unwrap();
    let instance_ctx = checker
        .ctx
        .with_instance_substitutions(vec![tla_core::ast::Substitution {
            from: Spanned::dummy("state".to_string()),
            to: Spanned::dummy(Expr::StateVar(
                "other".to_string(),
                other_idx.0,
                checker.ctx.var_registry().name_id_at(other_idx),
            )),
        }]);
    assert!(instance_ctx.instance_substitutions().is_some());
    let active_bindable = collect_first_guard_globally_bindable_names(
        std::slice::from_ref(&action),
        &scopes,
        &instance_ctx,
    );
    assert!(
        certify_value_action_vm_first_guard(
            &action,
            &complete_bindings,
            &instance_ctx,
            &active_bindable,
        )
        .is_none(),
        "an active INSTANCE substitution can redirect StateVar(state), so certification must fail"
    );
}

#[test]
fn first_guard_bound_specialization_ignores_unrelated_global_state_formal() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardBoundState ----
CONSTANT READY
VARIABLE state
Unrelated(state) == state
BoundAction(k) ==
  /\ state = READY
  /\ state' = state
Next == \E key \in {1} : BoundAction(key)
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(
            intern_name("READY"),
            Value::ModelValue(Rp::from("ready")),
        );
    let next = checker.ctx.get_op("Next").unwrap();
    let instances = crate::action_instance::split_action_instances(&checker.ctx, &next.body)
        .expect("the bounded action should split");
    assert_eq!(instances.len(), 1);
    let instance = &instances[0];
    assert_eq!(instance.name.as_deref(), Some("BoundAction"));
    assert!(!instance.bindings.is_empty());
    assert!(!instance.formal_bindings.is_empty());

    let action = ActionInstanceMeta {
        name: instance.name.clone(),
        bindings: instance.bindings.clone(),
        formal_bindings: instance.formal_bindings.clone(),
        expr: Some(instance.expr.clone()),
    };
    let complete_bindings = vec![instance.complete_bindings.clone()];
    let bindable = collect_first_guard_globally_bindable_names(
        std::slice::from_ref(&action),
        &complete_bindings,
        &checker.ctx,
    );
    assert!(bindable.contains("state"));
    assert!(
        certify_value_action_vm_first_guard(
            &action,
            &instance.complete_bindings,
            &checker.ctx,
            &bindable,
        )
        .is_some(),
        "an unrelated formal named state must not reject an exact bound-action StateVar read"
    );
}

#[test]
fn first_guard_follows_total_shared_transition_call_and_function_key() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardTransition ----
CONSTANTS Access, Advance
VARIABLE pc
Transition(t, from, to) ==
  /\ pc[t] = from
  /\ pc' = [pc EXCEPT ![t] = to]
End(actor) ==
  LET unused == 1 \div 0
  IN /\ Transition(actor, Access, Advance)
     /\ UNCHANGED pc
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    {
        let constants =
            std::sync::Arc::make_mut(checker.ctx.shared_arc_mut()).precomputed_constants_mut();
        constants.insert(intern_name("Access"), Value::string("Access"));
        constants.insert(intern_name("Advance"), Value::string("Advance"));
        constants.insert(
            intern_name("w1"),
            Value::ModelValue(Rp::from("w1")),
        );
    }
    let expr = checker.ctx.get_op("End").unwrap().body.clone();
    let mut action = metadata_with_expr("End", expr);
    let actor = Value::ModelValue(Rp::from("w1"));
    action.formal_bindings = vec![(std::sync::Arc::from("actor"), actor.clone())];

    let guard = certify_first_guard_for_test(&action, &checker.ctx)
        .expect("the first total Transition call should certify pc[actor]");
    let mut pc = FuncBuilder::new();
    pc.insert(actor.clone(), Value::string("Access"));
    let matching = ArrayState::from_values(vec![Value::Func(Rp::new(pc.build()))]);
    assert!(!guard.mismatches(&matching));

    let mut pc = FuncBuilder::new();
    pc.insert(actor, Value::string("Advance"));
    let mismatching = ArrayState::from_values(vec![Value::Func(Rp::new(pc.build()))]);
    assert!(guard.mismatches(&mismatching));
}

#[test]
fn first_guard_call_actuals_resolve_in_unchanged_caller_scope() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardCallerScope ----
VARIABLE pc
Transition(t, from, to) ==
  /\ pc[t] = from
  /\ pc' = [pc EXCEPT ![t] = to]
A(x, t) ==
  /\ Transition(x, t, "done")
  /\ UNCHANGED pc
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let actor = Value::ModelValue(Rp::from("w1"));
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("w1"), actor.clone());
    let access = Value::string("Access");
    let mut action = metadata_with_expr("A", checker.ctx.get_op("A").unwrap().body.clone());
    action.formal_bindings = vec![
        (std::sync::Arc::from("x"), actor.clone()),
        (std::sync::Arc::from("t"), access.clone()),
    ];

    let guard = certify_first_guard_for_test(&action, &checker.ctx)
        .expect("callee actual t must resolve to the caller's A.t binding");
    let mut pc = FuncBuilder::new();
    pc.insert(actor, access);
    let matching = ArrayState::from_values(vec![Value::Func(Rp::new(pc.build()))]);
    assert!(
        !guard.mismatches(&matching),
        "the first callee formal named t must not shadow the second actual t"
    );
}

#[test]
fn first_guard_shared_callee_cannot_inherit_unpassed_synthetic_caller_local() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardCalleeScope ----
CONSTANT p
VARIABLE state
B(q) ==
  /\ state = p
  /\ state' = state
A(p) ==
  /\ B(0)
  /\ UNCHANGED state
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("p"), Value::int(2));
    let mut action = metadata_with_expr("A", checker.ctx.get_op("A").unwrap().body.clone());
    action.formal_bindings = vec![(std::sync::Arc::from("p"), Value::int(1))];

    assert!(
        certify_first_guard_for_test(&action, &checker.ctx).is_none(),
        "compiled B sees global p=2; it cannot inherit synthetic A(p)=1 unless p is passed"
    );
}

#[test]
fn first_guard_skip_avoids_eager_vm_error_but_match_keeps_it_authoritative() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmFirstGuardEagerError ----
CONSTANT READY
VARIABLES state, args
Unrelated(state) == state
A ==
  LET eager == args[1]
  IN /\ state = READY
     /\ UNCHANGED <<state, args>>
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let ready = Value::ModelValue(Rp::from("ready"));
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut())
        .precomputed_constants_mut()
        .insert(intern_name("READY"), ready.clone());
    let action = metadata_with_expr("A", checker.ctx.get_op("A").unwrap().body.clone());

    let mut eager_error = BytecodeFunction::new("A".to_string(), 0);
    eager_error.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    eager_error.emit(Opcode::LoadImm { rd: 1, value: 1 });
    eager_error.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    eager_error.emit(Opcode::LoadVar { rd: 3, var_idx: 1 });
    eager_error.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    eager_error.emit(Opcode::StoreVar { var_idx: 1, rs: 3 });
    eager_error.emit(Opcode::LoadBool { rd: 4, value: true });
    eager_error.emit(Opcode::Ret { rs: 4 });
    let bytecode = compiled_bytecode(vec![eager_error], &[("A", 0)]);
    let complete_bindings = vec![Vec::new()];
    let plan = ValueActionVmPlan::build_with_first_guards(
        &[action],
        &bytecode,
        2,
        Some(&complete_bindings),
        &checker.ctx,
        &checker.module.vars,
    )
    .expect("the Value entry and source guard should certify");
    assert!(plan.entries[0].first_guard.is_some());

    let nil = Value::ModelValue(Rp::from("nil"));
    let wrong_phase = ArrayState::from_values(vec![
        nil.clone(),
        Value::ModelValue(Rp::from("busy")),
    ]);
    let (result, stats) = execute_value_action_vm_plan_attempt(
        &plan,
        &mut checker.ctx,
        &wrong_phase,
        false,
        false,
        true,
    );
    let result = result.expect("a false semantic first guard must skip eager bytecode");
    assert!(result.successors.is_empty());
    assert_eq!((stats.first_guard_checks, stats.first_guard_skips), (1, 1));
    assert_eq!(stats.entry_evals, 0);

    let matching_phase = ArrayState::from_values(vec![nil, ready]);
    let guard = plan.entries[0].first_guard.as_ref().unwrap();
    let actual_phase = matching_phase.get(VarIndex(1));
    assert!(
        !guard.mismatches(&matching_phase),
        "the matching phase must enter the eager bytecode: guard={guard:?}, actual={actual_phase:?}"
    );
    let (result, stats) = execute_value_action_vm_plan_attempt(
        &plan,
        &mut checker.ctx,
        &matching_phase,
        false,
        false,
        true,
    );
    let success_shape = result
        .as_ref()
        .ok()
        .map(|result| (result.successors.len(), result.had_raw_successors));
    assert!(
        result.is_err(),
        "a matching guard must retain the VM error: success={success_shape:?}, stats={stats:?}"
    );
    assert_eq!((stats.first_guard_checks, stats.first_guard_skips), (1, 0));
    assert_eq!(stats.entry_evals, 1);
}

#[test]
fn plan_rejects_holey_numeric_disjunction_suffixes() {
    let metadata = vec![metadata("A"), metadata("A")];
    let mut indices = rustc_hash::FxHashMap::default();
    indices.insert("A#d2".to_string(), 1);
    indices.insert("A#d0".to_string(), 0);

    let reason = resolve_value_action_vm_plan_entries(&metadata, &indices)
        .expect_err("a missing #d1 arm must reject the whole plan");
    assert!(reason.contains("missing contiguous arm #d1"), "{reason}");
}

#[test]
fn plan_rejects_exact_and_suffixed_ambiguity() {
    let metadata = vec![metadata("A")];
    let mut indices = rustc_hash::FxHashMap::default();
    indices.insert("A".to_string(), 0);
    indices.insert("A#d0".to_string(), 1);

    let reason = resolve_value_action_vm_plan_entries(&metadata, &indices)
        .expect_err("an exact entry plus a suffixed arm is ambiguous");
    assert!(reason.contains("both an exact function and #d arms"));
}

#[test]
fn plan_rejects_exact_entry_for_multiple_metadata_occurrences() {
    let metadata = vec![metadata("A"), metadata("A")];
    let mut indices = rustc_hash::FxHashMap::default();
    indices.insert("A".to_string(), 0);

    let reason = resolve_value_action_vm_plan_entries(&metadata, &indices)
        .expect_err("one exact generator cannot account for two split arms");
    assert!(reason.contains("ambiguously represents 2 metadata instances"));
}

#[test]
fn plan_rejects_noncontiguous_repeated_metadata_key() {
    let metadata = vec![metadata("A"), metadata("B"), metadata("A")];
    let mut indices = rustc_hash::FxHashMap::default();
    indices.insert("A".to_string(), 0);
    indices.insert("B".to_string(), 1);

    let reason = resolve_value_action_vm_plan_entries(&metadata, &indices)
        .expect_err("a transformed key cannot reconstruct noncontiguous source occurrences");
    assert!(reason.contains("noncontiguous metadata groups"));
}

#[test]
fn plan_rejects_key_collision_with_unequal_formal_bindings() {
    let mut first = metadata("A");
    first.formal_bindings = vec![(std::sync::Arc::from("p"), Value::int(1))];
    let mut second = metadata("A");
    second.formal_bindings = vec![(std::sync::Arc::from("p"), Value::int(2))];
    let mut indices = rustc_hash::FxHashMap::default();
    indices.insert("A#d0".to_string(), 0);
    indices.insert("A#d1".to_string(), 1);

    let reason = resolve_value_action_vm_plan_entries(&[first, second], &indices)
        .expect_err("one raw key must not merge unequal specializations");
    assert!(reason.contains("collides across unequal bindings or formal bindings"));
}

#[test]
fn plan_certification_is_all_or_nothing() {
    let metadata = vec![metadata("A"), metadata("B")];
    let bytecode = compiled_bytecode(
        vec![transformed_write("A", 1, 0), transformed_write("B", 2, 1)],
        &[("A", 0), ("B", 1)],
    );

    let reason = ValueActionVmPlan::build(&metadata, &bytecode, 1)
        .expect_err("the later arity-positive entry must reject the entire plan");
    assert!(reason.contains("entry 'B' is ineligible"), "{reason}");
    assert!(reason.contains("must have arity 0"), "{reason}");
}

#[test]
fn register_reuse_certificate_is_per_entry_and_mixed_plans_execute() {
    let mut guard_loop = BytecodeFunction::new("LoopGuard".to_string(), 0);
    guard_loop.emit(Opcode::LoadConst { rd: 0, idx: 0 });
    guard_loop.emit(Opcode::ForallBegin {
        rd: 1,
        r_binding: 2,
        r_domain: 0,
        loop_end: 3,
    });
    guard_loop.emit(Opcode::LoadBool { rd: 3, value: true });
    guard_loop.emit(Opcode::ForallNext {
        rd: 1,
        r_binding: 2,
        r_body: 3,
        loop_begin: -1,
    });
    guard_loop.emit(Opcode::LoadImm { rd: 4, value: 2 });
    guard_loop.emit(Opcode::StoreVar { var_idx: 0, rs: 4 });
    guard_loop.emit(Opcode::Ret { rs: 1 });

    let mut bytecode = compiled_bytecode(
        vec![
            transformed_write("A", 1, 0),
            guard_loop,
            transformed_write("C", 3, 0),
        ],
        &[("A", 0), ("LoopGuard", 1), ("C", 2)],
    );
    let domain = bytecode
        .chunk
        .constants
        .add_value(Value::set([Value::int(1)]));
    assert_eq!(domain, 0);

    let plan = ValueActionVmPlan::build(
        &[metadata("A"), metadata("LoopGuard"), metadata("C")],
        &bytecode,
        1,
    )
    .expect("an entry-local loop should disable only register reuse, not the Value plan");
    assert_eq!(
        plan.entries
            .iter()
            .map(|entry| entry.register_reuse_certified)
            .collect::<Vec<_>>(),
        vec![true, false, true]
    );

    let module = parse_module(
        r#"
---- MODULE ValueActionVmMixedRegisterReuse ----
VARIABLE x
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let current = ArrayState::from_values(vec![Value::int(0)]);
    let (result, stats) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, false, true, false);
    let result = result.expect("mixed reset/reuse entries should execute in plan order");
    assert_eq!(result.successors.len(), 3);
    assert_eq!(
        result
            .successors
            .iter()
            .map(|successor| {
                successor
                    .materialize(&current, checker.ctx.var_registry())
                    .materialize_values()
            })
            .collect::<Vec<_>>(),
        vec![
            vec![Value::int(1)],
            vec![Value::int(2)],
            vec![Value::int(3)]
        ],
        "mixed reset/reuse entries must preserve ordered concrete successors"
    );
    assert_eq!(stats.entry_evals, 3);
    assert_eq!(stats.register_reuse_entry_evals, 2);
}

#[test]
fn execution_preserves_duplicate_enabled_stutters() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmDuplicateStutters ----
VARIABLE x
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let bytecode = compiled_bytecode(
        vec![
            transformed_write("A#d0", 7, 0),
            transformed_write("A#d1", 7, 0),
        ],
        &[("A#d1", 1), ("A#d0", 0)],
    );
    let plan = ValueActionVmPlan::build(&[metadata("A"), metadata("A")], &bytecode, 1)
        .expect("both stuttering arms should certify");
    let current = ArrayState::from_values(vec![Value::int(7)]);

    let (result, _) = execute_value_action_vm_plan_attempt(
        &plan,
        &mut checker.ctx,
        &current,
        false,
        false,
        false,
    );
    let result = result.expect("both action entries should execute");
    assert!(result.had_raw_successors);
    assert_eq!(result.successors.len(), 2);
    assert!(result
        .successors
        .iter()
        .all(|successor| successor.changes.is_empty()));
}

#[test]
fn context_free_need_retries_whole_parent_once_then_latches_bound_mode() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmContextRetry ----
VARIABLE x
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let mut bytecode = compiled_bytecode(
        vec![
            transformed_write("Fast", 99, 0),
            context_equality_write("Ctx", 7),
        ],
        &[("Fast", 0), ("Ctx", 1)],
    );
    let lhs = bytecode
        .chunk
        .constants
        .add_value(parent_sensitive_setpred(42));
    let rhs = bytecode
        .chunk
        .constants
        .add_value(Value::set([Value::int(1)]));
    assert_eq!((lhs, rhs), (0, 1));
    let mut plan = ValueActionVmPlan::build(&[metadata("Fast"), metadata("Ctx")], &bytecode, 1)
        .expect("both context-retry entries should certify");
    plan.entries[0].first_guard = Some(ValueActionVmFirstGuard::SlotEq {
        var_idx: VarIndex(0),
        expected: Value::int(42),
    });

    checker.value_action_vm.requested = true;
    checker.value_action_vm.ctx_free_requested = true;
    checker.value_action_vm.register_reuse_requested = true;
    checker.value_action_vm.first_guard_requested = true;
    checker.value_action_vm.install_plan(plan);
    checker.value_action_vm.shadow_remaining = 0;
    checker.action_bytecode = Some(bytecode);
    let current = ArrayState::from_values(vec![Value::int(42)]);
    assert!(!checker.ctx.has_state_env());

    let first = checker
        .execute_value_action_vm_parent(&current)
        .expect("a context request must restart the complete parent in bound mode");
    assert!(
        !checker.ctx.has_state_env(),
        "the retry guard must restore EvalCtx"
    );
    assert_eq!(
        first
            .successors
            .iter()
            .map(|successor| {
                successor
                    .materialize(&current, checker.ctx.var_registry())
                    .materialize_values()
            })
            .collect::<Vec<_>>(),
        vec![vec![Value::int(99)], vec![Value::int(7)]],
        "the partial context-free successor must be discarded before retry"
    );
    assert!(checker.value_action_vm.ctx_required);
    assert_eq!(checker.value_action_vm.stats.ctx_free_parents, 1);
    assert_eq!(checker.value_action_vm.stats.ctx_bound_parents, 1);
    assert_eq!(checker.value_action_vm.stats.ctx_retries, 1);
    assert_eq!(checker.value_action_vm.stats.entry_evals, 2);
    assert_eq!(checker.value_action_vm.stats.enabled_entries, 2);
    assert_eq!(
        checker.value_action_vm.stats.register_reuse_entry_evals, 2,
        "discarded context-free probe work must not enter accepted-attempt telemetry"
    );
    assert_eq!(checker.value_action_vm.stats.first_guard_checks, 1);
    assert_eq!(checker.value_action_vm.stats.first_guard_skips, 0);

    let second = checker
        .execute_value_action_vm_parent(&current)
        .expect("the latched plan should execute directly in bound mode");
    assert_eq!(second.successors.len(), 2);
    assert!(
        !checker.ctx.has_state_env(),
        "the bound guard must restore EvalCtx"
    );
    assert_eq!(checker.value_action_vm.stats.ctx_free_parents, 1);
    assert_eq!(checker.value_action_vm.stats.ctx_bound_parents, 2);
    assert_eq!(checker.value_action_vm.stats.ctx_retries, 1);
    assert_eq!(checker.value_action_vm.stats.entry_evals, 4);
    assert_eq!(checker.value_action_vm.stats.enabled_entries, 4);
    assert_eq!(checker.value_action_vm.stats.register_reuse_entry_evals, 4);
    assert_eq!(checker.value_action_vm.stats.first_guard_checks, 2);
    assert_eq!(checker.value_action_vm.stats.first_guard_skips, 0);
}

#[test]
fn shadow_comparison_preserves_order_and_multiplicity() {
    let registry = VarRegistry::from_names(["x"]);
    let current = ArrayState::from_values(vec![Value::int(0)]);
    let successor = |value| {
        let mut changes = DiffChanges::new();
        changes.push((VarIndex(0), Value::int(value)));
        DiffSuccessor::from_changes(changes)
    };
    let candidate = SuccessorResult {
        successors: vec![successor(1), successor(1), successor(2)],
        had_raw_successors: true,
    };
    let identical = SuccessorResult {
        successors: vec![successor(1), successor(1), successor(2)],
        had_raw_successors: true,
    };
    let reordered = SuccessorResult {
        successors: vec![successor(1), successor(2), successor(1)],
        had_raw_successors: true,
    };
    let deduplicated = SuccessorResult {
        successors: vec![successor(1), successor(2)],
        had_raw_successors: true,
    };

    assert!(ordered_value_action_vm_shadow_match(
        &current, &registry, &candidate, &identical
    ));
    assert!(!ordered_value_action_vm_shadow_match(
        &current, &registry, &candidate, &reordered
    ));
    assert!(!ordered_value_action_vm_shadow_match(
        &current,
        &registry,
        &candidate,
        &deduplicated
    ));
}

#[test]
fn later_runtime_error_discards_whole_candidate_parent_and_disarms() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmWholeParentFallback ----
VARIABLE x
Init == x = 0
Next == x' = 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        check_deadlock: false,
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.trace.cached_next_name = Some("Next".to_string());
    checker.tir_parity = None;

    let fast = transformed_write("Fast", 99, 0);
    let mut late_error = BytecodeFunction::new("LateError".to_string(), 0);
    late_error.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    late_error.emit(Opcode::LoadImm { rd: 1, value: 1 });
    late_error.emit(Opcode::AddInt {
        rd: 2,
        r1: 0,
        r2: 1,
    });
    late_error.emit(Opcode::StoreVar { var_idx: 0, rs: 2 });
    late_error.emit(Opcode::LoadBool { rd: 3, value: true });
    late_error.emit(Opcode::Ret { rs: 3 });
    let bytecode = compiled_bytecode(vec![fast, late_error], &[("Fast", 0), ("LateError", 1)]);
    let metadata = vec![metadata("Fast"), metadata("LateError")];
    let plan = ValueActionVmPlan::build(&metadata, &bytecode, 1)
        .expect("the late type error is state-dependent, not a static eligibility failure");

    checker.value_action_vm.requested = true;
    checker.value_action_vm.ctx_free_requested = true;
    checker.value_action_vm.register_reuse_requested = true;
    checker.value_action_vm.install_plan(plan);
    checker.value_action_vm.shadow_remaining = 0;
    checker.action_bytecode = Some(bytecode);

    let current = ArrayState::from_values(vec![Value::Bool(true)]);
    let first = checker
        .generate_successors_as_diffs_raw(&current)
        .expect("runtime VM failure must fall back, not escape")
        .expect("canonical simple equality should produce diffs");
    assert_eq!(first.successors.len(), 1);
    assert_eq!(
        first.successors[0]
            .materialize(&current, checker.ctx.var_registry())
            .materialize_values(),
        vec![Value::int(1)],
        "the earlier speculative x'=99 result must not mix with canonical Next"
    );
    assert!(checker.value_action_vm.disabled);
    assert_eq!(checker.value_action_vm.stats.runtime_fallbacks, 1);
    assert_eq!(checker.value_action_vm.stats.candidate_parents, 1);
    assert_eq!(checker.value_action_vm.stats.ctx_free_parents, 1);
    assert_eq!(checker.value_action_vm.stats.ctx_bound_parents, 0);
    assert_eq!(checker.value_action_vm.stats.ctx_retries, 0);
    assert!(!checker.value_action_vm.ctx_required);
    assert_eq!(checker.value_action_vm.stats.entry_evals, 2);
    assert_eq!(checker.value_action_vm.stats.enabled_entries, 1);
    assert_eq!(checker.value_action_vm.stats.register_reuse_entry_evals, 2);

    let second = checker
        .generate_successors_as_diffs_raw(&current)
        .expect("disarmed dispatch should stay on the canonical path")
        .expect("canonical simple equality should still produce diffs");
    assert_eq!(second.successors.len(), 1);
    assert_eq!(checker.value_action_vm.stats.candidate_parents, 1);
    assert_eq!(checker.value_action_vm.stats.runtime_fallbacks, 1);
}

#[test]
fn source_compiled_lazy_choose_error_falls_back_for_parent_and_keeps_plan_armed() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmLazyChooseFallback ----
EXTENDS Naturals, Sequences
VARIABLES root, path, state

ParentOf(node) == CHOOSE parent \in 1..2 : parent = node + 10

A ==
    LET node == Head(path)
        parent == ParentOf(node)
        splitParent == parent = 2
    IN /\ state = "ready"
       /\ root' = CASE node = root -> root
                     [] splitParent -> parent
                     [] OTHER -> root
       /\ UNCHANGED <<path, state>>

Next == A
====
"#,
    );
    let config = Config {
        next: Some("Next".to_string()),
        check_deadlock: false,
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.trace.cached_next_name = Some("Next".to_string());

    let next = checker.ctx.get_op("Next").expect("Next operator");
    let instances = crate::action_instance::split_action_instances(&checker.ctx, &next.body)
        .expect("the single named action should split");
    assert_eq!(instances.len(), 1);
    checker.compiled.split_action_complete_bindings = Some(
        instances
            .iter()
            .map(|instance| instance.complete_bindings.clone())
            .collect(),
    );
    checker.compiled.split_action_meta = Some(
        instances
            .into_iter()
            .map(|instance| ActionInstanceMeta {
                name: instance.name,
                bindings: instance.bindings,
                formal_bindings: instance.formal_bindings,
                expr: Some(instance.expr),
            })
            .collect(),
    );

    checker.value_action_vm.requested = true;
    checker.value_action_vm.ctx_free_requested = true;
    checker.value_action_vm.register_reuse_requested = true;
    checker.value_action_vm.first_guard_requested = true;
    checker.compile_action_bytecode();
    assert!(checker.value_action_vm.is_armed());
    checker.value_action_vm.shadow_remaining = 0;
    checker.tir_parity = None;

    let registry = checker.ctx.var_registry();
    let mut values = vec![Value::Bool(false); registry.len()];
    values[registry.get("root").expect("root slot").as_usize()] = Value::int(1);
    values[registry.get("path").expect("path slot").as_usize()] = Value::tuple([Value::int(1)]);
    values[registry.get("state").expect("state slot").as_usize()] = Value::string("ready");
    let current = ArrayState::from_values(values);

    let result = checker
        .generate_successors_as_diffs_raw(&current)
        .expect("canonical retry should preserve evaluator success")
        .expect("the canonical diff path should handle the action");
    assert!(result.had_raw_successors);
    assert_eq!(result.successors.len(), 1);
    assert_eq!(
        result.successors[0]
            .materialize(&current, checker.ctx.var_registry())
            .materialize_values(),
        current.materialize_values(),
        "the node=root branch is an enabled stuttering successor"
    );
    assert!(checker.value_action_vm.is_armed());
    assert!(!checker.value_action_vm.disabled);
    assert!(checker.value_action_vm.ctx_required);
    assert_eq!(checker.value_action_vm.shadow_remaining, 64);
    assert_eq!(checker.value_action_vm.stats.candidate_parents, 1);
    assert_eq!(checker.value_action_vm.stats.runtime_fallbacks, 1);
    assert_eq!(checker.value_action_vm.stats.quarantined_entries, 1);
    assert_eq!(checker.value_action_vm.stats.quarantined_entry_replays, 0);
    assert_eq!(checker.value_action_vm.stats.entry_evals, 1);
    assert_eq!(checker.value_action_vm.stats.first_guard_checks, 1);
    assert_eq!(checker.value_action_vm.stats.first_guard_skips, 0);

    let mixed = checker
        .generate_successors_as_diffs_raw(&current)
        .expect("the quarantined entry must replay canonically")
        .expect("the mixed batch boundary should retain diff generation");
    assert_eq!(mixed.successors.len(), 1);
    assert_eq!(
        mixed.successors[0]
            .materialize(&current, checker.ctx.var_registry())
            .materialize_values(),
        current.materialize_values()
    );
    assert!(checker.value_action_vm.is_armed());
    assert_eq!(checker.value_action_vm.stats.candidate_parents, 2);
    assert_eq!(checker.value_action_vm.stats.ctx_bound_parents, 1);
    assert_eq!(checker.value_action_vm.stats.quarantined_entry_replays, 1);
    assert_eq!(checker.value_action_vm.stats.shadow_checks, 1);
    assert_eq!(checker.value_action_vm.shadow_remaining, 63);
}

#[test]
fn quarantined_entry_replay_uses_complete_unpruned_binding_order() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmExactReplayBindings ----
VARIABLE state
A == /\ state = phase
     /\ state' = state
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let mut action = metadata_with_expr("A", checker.ctx.get_op("A").unwrap().body.clone());
    action.bindings = vec![(std::sync::Arc::from("phase"), Value::int(1))];
    let complete_bindings = vec![vec![
        (std::sync::Arc::from("phase"), Value::int(1)),
        (std::sync::Arc::from("phase"), Value::int(2)),
    ]];
    let bytecode = compiled_bytecode(vec![transformed_write("A__1", 9, 0)], &[("A__1", 0)]);
    let mut plan = ValueActionVmPlan::build_with_first_guards(
        &[action],
        &bytecode,
        1,
        Some(&complete_bindings),
        &checker.ctx,
        &checker.module.vars,
    )
    .expect("exact replay provenance should be retained");
    plan.entries[0].quarantined = true;

    let current = ArrayState::from_values(vec![Value::int(2)]);
    let (result, stats) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, true, true, false);
    let result = result.expect("the newest complete binding must enable the action");
    assert_eq!(result.successors.len(), 1);
    assert_eq!(
        result.successors[0]
            .materialize(&current, checker.ctx.var_registry())
            .materialize_values(),
        current.materialize_values(),
        "replaying only alias-pruned metadata would bind phase=1 and lose this successor"
    );
    assert_eq!(stats.entry_evals, 1);
    assert_eq!(stats.quarantined_entry_replays, 1);
}

#[test]
fn quarantined_entry_replay_binds_the_current_parent_state() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmReplayCurrentState ----
VARIABLE x
Q == /\ x = 2
     /\ x' = x
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let action = metadata_with_expr("Q", checker.ctx.get_op("Q").unwrap().body.clone());
    let complete_bindings = vec![Vec::new()];
    let bytecode = compiled_bytecode(vec![transformed_write("Q", 9, 0)], &[("Q", 0)]);
    let mut plan = ValueActionVmPlan::build_with_first_guards(
        &[action],
        &bytecode,
        1,
        Some(&complete_bindings),
        &checker.ctx,
        &checker.module.vars,
    )
    .expect("the entry should retain exact replay provenance");
    plan.entries[0].quarantined = true;

    // Install a deliberately stale ambient state. The mixed attempt must bind
    // its own current parent for canonical replay instead of observing this
    // pre-existing EvalCtx state.
    let stale = ArrayState::from_values(vec![Value::int(99)]);
    let _stale_guard = checker.ctx.bind_state_env_guard(stale.env_ref());
    let current = ArrayState::from_values(vec![Value::int(2)]);
    let (result, stats) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, true, true, false);
    let result = result.expect("replay should read x=2 from the current parent");
    assert_eq!(result.successors.len(), 1);
    assert_eq!(
        result.successors[0]
            .materialize(&current, checker.ctx.var_registry())
            .materialize_values(),
        current.materialize_values(),
        "the canonical entry is an enabled stuttering action only for the current parent"
    );
    assert_eq!(stats.quarantined_entry_replays, 1);
}

#[test]
fn mixed_entry_replay_enforces_one_parent_successor_cap() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmReplayCombinedCap ----
EXTENDS Sequences
VARIABLE x
A == x' = 1
Q == x' \in {2, 3}
C == x' = 4
R == x' = 2 \/ x' = 3 \/ x' = 4 \/ Head(<<>>) = 0
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut()).per_state_successor_cap = Some(3);

    let metadata = ["A", "Q", "C"]
        .into_iter()
        .map(|name| metadata_with_expr(name, checker.ctx.get_op(name).unwrap().body.clone()))
        .collect::<Vec<_>>();
    let complete_bindings = vec![Vec::new(), Vec::new(), Vec::new()];
    let bytecode = compiled_bytecode(
        vec![
            transformed_write("A", 1, 0),
            transformed_write("Q", 9, 0),
            transformed_write("C", 4, 0),
        ],
        &[("A", 0), ("Q", 1), ("C", 2)],
    );
    let mut plan = ValueActionVmPlan::build_with_first_guards(
        &metadata,
        &bytecode,
        1,
        Some(&complete_bindings),
        &checker.ctx,
        &checker.module.vars,
    )
    .expect("all entries should retain exact replay provenance");
    plan.entries[1].quarantined = true;

    let current = ArrayState::from_values(vec![Value::int(0)]);
    let final_entry = plan.entries.pop().expect("C entry");
    let (at_cap, _) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, true, true, false);
    let at_cap = at_cap.expect("exactly three successors with no fourth must be accepted");
    assert_eq!(at_cap.successors.len(), 3);

    plan.entries.push(final_entry);
    let (over_cap, _) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, true, true, false);
    assert!(matches!(
        over_cap,
        Err(ValueActionVmExecutionError::Fatal {
            entry_idx: 2,
            error: crate::error::EvalError::SetTooLarge { .. },
            ..
        })
    ));

    // The canonical occurrence itself gets only the parent's remaining
    // budget. After A emits one successor, R may emit two more; its third
    // emission must stop enumeration with SetTooLarge before the later Head
    // error is evaluated. A fresh per-entry cap of three would evaluate that
    // error first and report the wrong failure.
    let capped_metadata = ["A", "R"]
        .into_iter()
        .map(|name| metadata_with_expr(name, checker.ctx.get_op(name).unwrap().body.clone()))
        .collect::<Vec<_>>();
    let capped_bytecode = compiled_bytecode(
        vec![transformed_write("A", 1, 0), transformed_write("R", 9, 0)],
        &[("A", 0), ("R", 1)],
    );
    let mut capped_plan = ValueActionVmPlan::build_with_first_guards(
        &capped_metadata,
        &capped_bytecode,
        1,
        Some(&[Vec::new(), Vec::new()]),
        &checker.ctx,
        &checker.module.vars,
    )
    .expect("the capped replay entries should retain exact provenance");
    capped_plan.entries[1].quarantined = true;
    let (canonical_over_cap, _) = execute_value_action_vm_plan_attempt(
        &capped_plan,
        &mut checker.ctx,
        &current,
        true,
        true,
        false,
    );
    assert!(matches!(
        canonical_over_cap,
        Err(ValueActionVmExecutionError::Fatal {
            entry_idx: 1,
            error: crate::error::EvalError::SetTooLarge { .. },
            ..
        })
    ));
}

#[test]
fn canonical_quarantined_entry_error_is_fatal_and_disarms_plan() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmReplayFatalError ----
EXTENDS Sequences
VARIABLE x
Q == x' = Head(<<>>)
====
"#,
    );
    let config = Config {
        next: Some("Q".to_string()),
        check_deadlock: false,
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.trace.cached_next_name = Some("Q".to_string());
    checker.tir_parity = None;

    let action = metadata_with_expr("Q", checker.ctx.get_op("Q").unwrap().body.clone());
    let complete_bindings = vec![Vec::new()];
    let bytecode = compiled_bytecode(vec![transformed_write("Q", 9, 0)], &[("Q", 0)]);
    let mut plan = ValueActionVmPlan::build_with_first_guards(
        &[action],
        &bytecode,
        1,
        Some(&complete_bindings),
        &checker.ctx,
        &checker.module.vars,
    )
    .expect("the entry should retain exact replay provenance");
    plan.entries[0].quarantined = true;

    checker.value_action_vm.requested = true;
    checker.value_action_vm.ctx_free_requested = true;
    checker.value_action_vm.install_plan(plan);
    checker.value_action_vm.ctx_required = true;
    checker.value_action_vm.shadow_remaining = 0;

    let current = ArrayState::from_values(vec![Value::int(0)]);
    assert!(checker.value_action_vm.is_armed());
    assert!(
        checker.generate_successors_as_diffs_raw(&current).is_err(),
        "a canonical entry evaluator error must escape as fatal"
    );
    assert!(checker.value_action_vm.disabled);
    assert!(!checker.value_action_vm.is_armed());
    assert_eq!(checker.value_action_vm.stats.candidate_parents, 1);
    assert_eq!(checker.value_action_vm.stats.runtime_fallbacks, 1);
    assert_eq!(checker.value_action_vm.stats.quarantined_entry_replays, 1);

    assert!(
        checker.generate_successors_as_diffs_raw(&current).is_err(),
        "after disarm the authoritative whole action should report the same semantic error"
    );
    assert_eq!(
        checker.value_action_vm.stats.candidate_parents, 1,
        "a fatal canonical entry error must not retry or re-arm the VM plan"
    );
    assert_eq!(checker.value_action_vm.stats.runtime_fallbacks, 1);
    assert_eq!(checker.value_action_vm.stats.quarantined_entry_replays, 1);
}

#[test]
fn entry_quarantine_preserves_metadata_order_and_duplicate_successors() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmQuarantineOrder ----
VARIABLE x
A == x' = 1
Q == x' = 2
C == x' = 3
Next == A \/ Q \/ Q \/ C
====
"#,
    );
    let config = Config {
        next: Some("Next".to_string()),
        check_deadlock: false,
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.trace.cached_next_name = Some("Next".to_string());
    checker.tir_parity = None;

    let next = checker.ctx.get_op("Next").expect("Next operator");
    let instances = crate::action_instance::split_action_instances(&checker.ctx, &next.body)
        .expect("the four source occurrences should split in order");
    assert_eq!(
        instances
            .iter()
            .map(|instance| instance.name.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("A"), Some("Q"), Some("Q"), Some("C")]
    );
    let complete_bindings = instances
        .iter()
        .map(|instance| instance.complete_bindings.clone())
        .collect::<Vec<_>>();
    let metadata = instances
        .into_iter()
        .map(|instance| ActionInstanceMeta {
            name: instance.name,
            bindings: instance.bindings,
            formal_bindings: instance.formal_bindings,
            expr: Some(instance.expr),
        })
        .collect::<Vec<_>>();
    let bytecode = compiled_bytecode(
        vec![
            transformed_write("A", 1, 0),
            transformed_type_error("Q#d0"),
            transformed_write("Q#d1", 2, 0),
            transformed_write("C", 3, 0),
        ],
        &[("A", 0), ("Q#d0", 1), ("Q#d1", 2), ("C", 3)],
    );
    let plan = ValueActionVmPlan::build_with_first_guards(
        &metadata,
        &bytecode,
        1,
        Some(&complete_bindings),
        &checker.ctx,
        &checker.module.vars,
    )
    .expect("every final entry should have exact replay provenance");

    checker.value_action_vm.requested = true;
    checker.value_action_vm.ctx_free_requested = true;
    checker.value_action_vm.register_reuse_requested = true;
    checker.value_action_vm.install_plan(plan);
    checker.value_action_vm.shadow_remaining = 0;

    let current = ArrayState::from_values(vec![Value::int(0)]);
    let expected = vec![Some(1), Some(2), Some(2), Some(3)];
    let first = checker
        .generate_successors_as_diffs_raw(&current)
        .expect("the first VM error should recover the whole parent")
        .expect("canonical diff generation should handle Next");
    let first_values = first
        .successors
        .iter()
        .map(|diff| {
            diff.materialize(&current, checker.ctx.var_registry())
                .get(VarIndex(0))
                .as_i64()
        })
        .collect::<Vec<_>>();
    assert_eq!(first_values, expected);
    assert!(checker.value_action_vm.plan.as_ref().unwrap().entries[1].quarantined);
    assert_eq!(checker.value_action_vm.stats.quarantined_entries, 1);
    assert!(checker.value_action_vm.ctx_required);
    assert_eq!(checker.value_action_vm.shadow_remaining, 64);

    let mixed = checker
        .generate_successors_as_diffs_raw(&current)
        .expect("the mixed VM/canonical parent should remain exact")
        .expect("mixed diff generation should stay available");
    let mixed_values = mixed
        .successors
        .iter()
        .map(|diff| {
            diff.materialize(&current, checker.ctx.var_registry())
                .get(VarIndex(0))
                .as_i64()
        })
        .collect::<Vec<_>>();
    assert_eq!(mixed_values, expected);
    assert!(checker.value_action_vm.is_armed());
    assert_eq!(checker.value_action_vm.stats.runtime_fallbacks, 1);
    assert_eq!(checker.value_action_vm.stats.quarantined_entry_replays, 1);
    assert_eq!(checker.value_action_vm.stats.shadow_checks, 1);
    assert_eq!(checker.value_action_vm.shadow_remaining, 63);
}

#[test]
fn uniform_slot_guard_index_selects_ordered_bucket_with_duplicates_and_stutter() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmUniformSlotIndex ----
VARIABLE x
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let bytecode = compiled_bytecode(
        vec![
            transformed_write("A", 1, 0),
            transformed_write("B", 9, 0),
            transformed_write("C", 2, 0),
            transformed_write("D", 2, 0),
        ],
        &[("A", 0), ("B", 1), ("C", 2), ("D", 3)],
    );
    let mut plan = ValueActionVmPlan::build(
        &[metadata("A"), metadata("B"), metadata("C"), metadata("D")],
        &bytecode,
        1,
    )
    .expect("the four transformed entries should certify");
    install_uniform_slot_guards_for_test(
        &mut plan,
        &[
            (VarIndex(0), Value::int(1)),
            (VarIndex(0), Value::int(9)),
            (VarIndex(0), Value::int(1)),
            (VarIndex(0), Value::int(1)),
        ],
    );
    let index = plan
        .uniform_slot_guard_index
        .as_ref()
        .expect("same-slot scalar certificates should build one index");

    let current = ArrayState::from_values(vec![Value::int(1)]);
    assert_eq!(index.candidates(&current), Some([0, 2, 3].as_slice()));
    let (result, stats) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, false, true, true);
    let result = result.expect("the matching bucket should execute");
    assert!(result.had_raw_successors);
    assert_eq!(
        result
            .successors
            .iter()
            .map(|successor| {
                successor
                    .materialize(&current, checker.ctx.var_registry())
                    .get(VarIndex(0))
                    .as_i64()
            })
            .collect::<Vec<_>>(),
        vec![Some(1), Some(2), Some(2)],
        "source order, the enabled stutter, and duplicate successors must survive bucket dispatch"
    );
    assert_eq!((stats.first_guard_checks, stats.first_guard_skips), (4, 1));
    assert_eq!((stats.entry_evals, stats.enabled_entries), (3, 3));

    let unmatched = ArrayState::from_values(vec![Value::int(5)]);
    assert_eq!(index.candidates(&unmatched), Some([].as_slice()));
    let (result, stats) = execute_value_action_vm_plan_attempt(
        &plan,
        &mut checker.ctx,
        &unmatched,
        false,
        true,
        true,
    );
    let result = result.expect("an unmatched certified scalar has an empty bucket");
    assert!(!result.had_raw_successors);
    assert!(result.successors.is_empty());
    assert_eq!((stats.first_guard_checks, stats.first_guard_skips), (4, 4));
    assert_eq!(stats.entry_evals, 0);

    let unexpected = ArrayState::from_values(vec![Value::tuple([Value::int(1)])]);
    assert_eq!(index.candidates(&unexpected), None);
    let (result, stats) = execute_value_action_vm_plan_attempt(
        &plan,
        &mut checker.ctx,
        &unexpected,
        false,
        true,
        true,
    );
    let result = result.expect("a non-scalar slot must retain the full VM scan");
    assert_eq!(result.successors.len(), 4);
    assert_eq!((stats.first_guard_checks, stats.first_guard_skips), (4, 0));
    assert_eq!(stats.entry_evals, 4);
}

#[test]
fn uniform_slot_guard_index_rejects_nonuniform_or_missing_certificates() {
    let bytecode = compiled_bytecode(
        vec![transformed_write("A", 1, 0), transformed_write("B", 2, 0)],
        &[("A", 0), ("B", 1)],
    );
    let mut plan = ValueActionVmPlan::build(&[metadata("A"), metadata("B")], &bytecode, 1)
        .expect("the two transformed entries should certify");

    plan.entries[0].first_guard = Some(ValueActionVmFirstGuard::SlotEq {
        var_idx: VarIndex(0),
        expected: Value::int(1),
    });
    plan.entries[1].first_guard = Some(ValueActionVmFirstGuard::SlotEq {
        var_idx: VarIndex(1),
        expected: Value::int(1),
    });
    assert!(ValueActionVmUniformSlotGuardIndex::build(&plan.entries).is_none());

    plan.entries[1].first_guard = None;
    assert!(ValueActionVmUniformSlotGuardIndex::build(&plan.entries).is_none());

    plan.entries[1].first_guard = Some(ValueActionVmFirstGuard::FuncSlotEq {
        var_idx: VarIndex(0),
        key: Value::int(0),
        expected: Value::int(1),
    });
    assert!(ValueActionVmUniformSlotGuardIndex::build(&plan.entries).is_none());

    plan.entries[1].first_guard = Some(ValueActionVmFirstGuard::SlotEq {
        var_idx: VarIndex(0),
        expected: Value::tuple([Value::int(1)]),
    });
    assert!(ValueActionVmUniformSlotGuardIndex::build(&plan.entries).is_none());
}

#[test]
fn uniform_slot_guard_index_stops_logical_stats_at_selected_error() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmUniformSlotErrorStats ----
VARIABLE x
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let bytecode = compiled_bytecode(
        vec![
            transformed_write("A", 0, 0),
            transformed_type_error("B"),
            transformed_write("C", 3, 0),
        ],
        &[("A", 0), ("B", 1), ("C", 2)],
    );
    let mut plan =
        ValueActionVmPlan::build(&[metadata("A"), metadata("B"), metadata("C")], &bytecode, 1)
            .expect("the transformed entries should certify");
    install_uniform_slot_guards_for_test(
        &mut plan,
        &[
            (VarIndex(0), Value::int(0)),
            (VarIndex(0), Value::int(1)),
            (VarIndex(0), Value::int(1)),
        ],
    );

    let current = ArrayState::from_values(vec![Value::int(1)]);
    let (result, stats) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, false, true, true);
    assert!(matches!(
        result,
        Err(ValueActionVmExecutionError::Vm { entry_idx: 1, .. })
    ));
    assert_eq!((stats.first_guard_checks, stats.first_guard_skips), (2, 1));
    assert_eq!(stats.entry_evals, 1);
}

#[test]
fn uniform_slot_guard_index_replays_selected_quarantine_and_keeps_global_cap() {
    let module = parse_module(
        r#"
---- MODULE ValueActionVmUniformSlotQuarantine ----
VARIABLE x
A == /\ x = 1
     /\ x' = 1
Q == /\ x = 2
     /\ x' = 2
C == /\ x = 1
     /\ x' = 3
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let metadata = ["A", "Q", "C"]
        .into_iter()
        .map(|name| metadata_with_expr(name, checker.ctx.get_op(name).unwrap().body.clone()))
        .collect::<Vec<_>>();
    let bytecode = compiled_bytecode(
        vec![
            transformed_write("A", 1, 0),
            transformed_write("Q", 2, 0),
            transformed_write("C", 3, 0),
        ],
        &[("A", 0), ("Q", 1), ("C", 2)],
    );
    let mut plan = ValueActionVmPlan::build_with_first_guards(
        &metadata,
        &bytecode,
        1,
        Some(&[Vec::new(), Vec::new(), Vec::new()]),
        &checker.ctx,
        &checker.module.vars,
    )
    .expect("source guards and replay provenance should certify");
    assert_eq!(
        plan.uniform_slot_guard_index
            .as_ref()
            .and_then(|index| index.candidates(&ArrayState::from_values(vec![Value::int(1)]))),
        Some([0, 2].as_slice())
    );
    plan.entries[2].quarantined = true;

    let current = ArrayState::from_values(vec![Value::int(1)]);
    let (result, stats) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, true, true, true);
    let result = result.expect("the selected mixed bucket should execute exactly");
    assert_eq!(
        result
            .successors
            .iter()
            .map(|successor| {
                successor
                    .materialize(&current, checker.ctx.var_registry())
                    .get(VarIndex(0))
                    .as_i64()
            })
            .collect::<Vec<_>>(),
        vec![Some(1), Some(3)]
    );
    assert_eq!((stats.first_guard_checks, stats.first_guard_skips), (3, 1));
    assert_eq!((stats.entry_evals, stats.quarantined_entry_replays), (2, 1));

    std::sync::Arc::make_mut(checker.ctx.shared_arc_mut()).per_state_successor_cap = Some(1);
    let (over_cap, stats) =
        execute_value_action_vm_plan_attempt(&plan, &mut checker.ctx, &current, true, true, true);
    assert!(matches!(
        over_cap,
        Err(ValueActionVmExecutionError::Fatal {
            entry_idx: 2,
            error: crate::error::EvalError::SetTooLarge { .. },
            ..
        })
    ));
    assert_eq!((stats.first_guard_checks, stats.first_guard_skips), (3, 1));
    assert_eq!(stats.quarantined_entry_replays, 1);
}
