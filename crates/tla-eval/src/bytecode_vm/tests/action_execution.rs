// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Isolated action-mode execution tests.

use super::{
    make_func, ActionVmOutcome, BytecodeChunk, BytecodeVm, ConstantPool, Opcode, Value, VmError,
};

fn chunk_with_functions(
    constants: ConstantPool,
    functions: impl IntoIterator<Item = (Vec<Opcode>, u8)>,
) -> BytecodeChunk {
    let mut chunk = BytecodeChunk::new();
    chunk.constants = constants;
    for (index, (instructions, max_register)) in functions.into_iter().enumerate() {
        chunk.add_function(make_func(
            format!("action_{index}"),
            0,
            instructions,
            max_register,
        ));
    }
    chunk
}

fn assert_unbound_prime(result: Result<ActionVmOutcome, super::VmError>) {
    let err = result.expect_err("the action entry must not retain a prior overlay");
    assert!(
        err.to_string().contains("unbound primed variable"),
        "unexpected error: {err}"
    );
}

#[test]
fn normal_execution_still_rejects_store_var() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [(
            vec![
                Opcode::LoadImm { rd: 0, value: 7 },
                Opcode::StoreVar { var_idx: 0, rs: 0 },
                Opcode::LoadBool { rd: 0, value: true },
                Opcode::Ret { rs: 0 },
            ],
            0,
        )],
    );
    let state = [Value::int(1)];
    let err = BytecodeVm::new(&chunk, &state, None)
        .execute_function(0)
        .expect_err("ordinary expression execution must reject StoreVar");
    assert!(err.to_string().contains("StoreVar"));
}

#[test]
fn action_store_and_bound_load_prime_share_the_overlay() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [(
            vec![
                Opcode::LoadImm { rd: 0, value: 11 },
                Opcode::StoreVar { var_idx: 0, rs: 0 },
                Opcode::LoadPrime { rd: 1, var_idx: 0 },
                Opcode::LoadImm { rd: 2, value: 11 },
                Opcode::Eq {
                    rd: 3,
                    r1: 1,
                    r2: 2,
                },
                Opcode::Ret { rs: 3 },
            ],
            3,
        )],
    );
    let state = [Value::int(4)];
    let outcome = BytecodeVm::new(&chunk, &state, None)
        .execute_action_function(0)
        .expect("action execution should succeed");
    assert_eq!(
        outcome,
        ActionVmOutcome::Enabled([(0, Value::int(11))].into_iter().collect())
    );
}

#[test]
fn nested_call_reads_the_callers_action_overlay() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [
            (
                vec![
                    Opcode::LoadPrime { rd: 0, var_idx: 0 },
                    Opcode::LoadImm { rd: 1, value: 9 },
                    Opcode::Eq {
                        rd: 2,
                        r1: 0,
                        r2: 1,
                    },
                    Opcode::Ret { rs: 2 },
                ],
                2,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 0, value: 9 },
                    Opcode::StoreVar { var_idx: 0, rs: 0 },
                    Opcode::Call {
                        rd: 1,
                        op_idx: 0,
                        args_start: 0,
                        argc: 0,
                    },
                    Opcode::Ret { rs: 1 },
                ],
                1,
            ),
        ],
    );
    let state = [Value::int(1)];
    assert_eq!(
        BytecodeVm::new(&chunk, &state, None)
            .execute_action_function(1)
            .expect("nested action call"),
        ActionVmOutcome::Enabled([(0, Value::int(9))].into_iter().collect())
    );
}

#[test]
fn action_passes_overlay_value_to_pure_helper_call() {
    let mut chunk = BytecodeChunk::new();
    chunk.constants = ConstantPool::new();
    chunk.add_function(make_func(
        "pure_helper".to_string(),
        1,
        vec![
            Opcode::LoadImm { rd: 1, value: 9 },
            Opcode::Eq {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    ));
    chunk.add_function(make_func(
        "entry".to_string(),
        0,
        vec![
            Opcode::LoadImm { rd: 0, value: 9 },
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::LoadPrime { rd: 1, var_idx: 0 },
            Opcode::Call {
                rd: 2,
                op_idx: 0,
                args_start: 1,
                argc: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    ));

    let state = [Value::int(1)];
    assert_eq!(
        BytecodeVm::new(&chunk, &state, None)
            .execute_action_function(1)
            .expect("certifiable pure helper call"),
        ActionVmOutcome::Enabled([(0, Value::int(9))].into_iter().collect())
    );
}

#[test]
fn nested_pure_helper_failures_discard_the_entry_overlay() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [
            (
                vec![
                    Opcode::LoadBool { rd: 0, value: true },
                    Opcode::LoadImm { rd: 1, value: 1 },
                    Opcode::AddInt {
                        rd: 2,
                        r1: 0,
                        r2: 1,
                    },
                    Opcode::Ret { rs: 2 },
                ],
                2,
            ),
            (
                vec![Opcode::LoadImm { rd: 0, value: 7 }, Opcode::Ret { rs: 0 }],
                0,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 0, value: 9 },
                    Opcode::StoreVar { var_idx: 0, rs: 0 },
                    Opcode::Call {
                        rd: 1,
                        op_idx: 0,
                        args_start: 0,
                        argc: 0,
                    },
                    Opcode::Ret { rs: 1 },
                ],
                1,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 0, value: 9 },
                    Opcode::StoreVar { var_idx: 0, rs: 0 },
                    Opcode::Call {
                        rd: 1,
                        op_idx: 1,
                        args_start: 0,
                        argc: 0,
                    },
                    Opcode::Ret { rs: 1 },
                ],
                1,
            ),
            (
                vec![
                    Opcode::LoadPrime { rd: 0, var_idx: 0 },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
        ],
    );
    let state = [Value::int(1)];
    let mut vm = BytecodeVm::new(&chunk, &state, None);

    assert!(vm.execute_action_function(2).is_err());
    assert_unbound_prime(vm.execute_action_function(4));
    assert!(vm.execute_action_function(3).is_err());
    assert_unbound_prime(vm.execute_action_function(4));
}

#[test]
fn unbound_action_prime_does_not_use_next_state_or_its_cache() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [
            (
                vec![
                    Opcode::LoadPrime { rd: 0, var_idx: 0 },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
            (
                vec![
                    Opcode::LoadPrime { rd: 0, var_idx: 0 },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
        ],
    );
    let state = [Value::int(10)];
    let next_state = [Value::int(99)];
    let mut vm = BytecodeVm::new(&chunk, &state, Some(&next_state));

    assert_eq!(
        vm.execute_function(1).expect("normal primed load"),
        Value::int(99)
    );
    assert_unbound_prime(vm.execute_action_function(0));
    assert_eq!(
        vm.execute_function(1)
            .expect("normal primed load after action"),
        Value::int(99)
    );
}

#[test]
fn unchanged_binds_parent_for_later_prime_load_without_emitting_a_change() {
    let mut constants = ConstantPool::new();
    let start = constants.add_value(Value::int(0));
    constants.add_value(Value::int(1));
    let chunk = chunk_with_functions(
        constants,
        [(
            vec![
                Opcode::Unchanged {
                    rd: 0,
                    start,
                    count: 2,
                },
                Opcode::LoadPrime { rd: 1, var_idx: 1 },
                Opcode::LoadImm { rd: 2, value: 20 },
                Opcode::Eq {
                    rd: 3,
                    r1: 1,
                    r2: 2,
                },
                Opcode::And {
                    rd: 4,
                    r1: 0,
                    r2: 3,
                },
                Opcode::Ret { rs: 4 },
            ],
            4,
        )],
    );
    let state = [Value::int(10), Value::int(20)];
    assert_eq!(
        BytecodeVm::new(&chunk, &state, None)
            .execute_action_function(0)
            .expect("UNCHANGED action"),
        ActionVmOutcome::Enabled(Default::default())
    );
}

#[test]
fn unchanged_enabled_stutter_crosses_bound_bitmap_word_boundary() {
    let mut constants = ConstantPool::new();
    let start = constants.add_value(Value::int(0));
    for var_idx in 1..65 {
        constants.add_value(Value::int(var_idx));
    }
    let chunk = chunk_with_functions(
        constants,
        [(
            vec![
                Opcode::Unchanged {
                    rd: 0,
                    start,
                    count: 65,
                },
                Opcode::Ret { rs: 0 },
            ],
            0,
        )],
    );
    let state: Vec<Value> = (0..65).map(Value::int).collect();

    assert_eq!(
        BytecodeVm::new(&chunk, &state, None)
            .execute_action_function(0)
            .expect("65-slot UNCHANGED stutter"),
        ActionVmOutcome::Enabled(Default::default())
    );
}

#[test]
fn unchanged_before_and_after_store_obey_bound_overlay_semantics() {
    let mut constants = ConstantPool::new();
    let start = constants.add_value(Value::int(0));
    let chunk = chunk_with_functions(
        constants,
        [
            (
                vec![
                    Opcode::Unchanged {
                        rd: 0,
                        start,
                        count: 1,
                    },
                    Opcode::LoadImm { rd: 1, value: 10 },
                    Opcode::StoreVar { var_idx: 0, rs: 1 },
                    Opcode::LoadBool { rd: 0, value: true },
                    Opcode::Ret { rs: 0 },
                ],
                1,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 1, value: 10 },
                    Opcode::StoreVar { var_idx: 0, rs: 1 },
                    Opcode::Unchanged {
                        rd: 0,
                        start,
                        count: 1,
                    },
                    Opcode::Ret { rs: 0 },
                ],
                1,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 1, value: 11 },
                    Opcode::StoreVar { var_idx: 0, rs: 1 },
                    Opcode::Unchanged {
                        rd: 0,
                        start,
                        count: 1,
                    },
                    Opcode::Ret { rs: 0 },
                ],
                1,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 1, value: 10 },
                    Opcode::StoreVar { var_idx: 0, rs: 1 },
                    Opcode::StoreVar { var_idx: 0, rs: 1 },
                    Opcode::LoadBool { rd: 0, value: true },
                    Opcode::Ret { rs: 0 },
                ],
                1,
            ),
        ],
    );
    let state = [Value::int(10)];
    let mut vm = BytecodeVm::new(&chunk, &state, None);
    let err = vm
        .execute_action_function(0)
        .expect_err("StoreVar after UNCHANGED must fail closed");
    assert!(err
        .to_string()
        .contains("duplicate action successor binding"));
    assert_eq!(
        vm.execute_action_function(1)
            .expect("UNCHANGED should compare an earlier equal write"),
        ActionVmOutcome::Enabled(Default::default())
    );
    assert_eq!(
        vm.execute_action_function(2)
            .expect("UNCHANGED should compare an earlier different write"),
        ActionVmOutcome::Disabled
    );
    let err = vm
        .execute_action_function(3)
        .expect_err("a second StoreVar for one slot must fail closed");
    assert!(err
        .to_string()
        .contains("duplicate action successor binding"));
}

#[test]
fn unchanged_binds_later_tuple_slots_even_after_an_earlier_mismatch() {
    let mut constants = ConstantPool::new();
    let start = constants.add_value(Value::int(0));
    constants.add_value(Value::int(1));
    let chunk = chunk_with_functions(
        constants,
        [(
            vec![
                Opcode::LoadImm { rd: 0, value: 1 },
                Opcode::StoreVar { var_idx: 0, rs: 0 },
                Opcode::Unchanged {
                    rd: 1,
                    start,
                    count: 2,
                },
                Opcode::LoadPrime { rd: 2, var_idx: 1 },
                Opcode::LoadImm { rd: 3, value: 20 },
                Opcode::Eq {
                    rd: 4,
                    r1: 2,
                    r2: 3,
                },
                Opcode::Ret { rs: 4 },
            ],
            4,
        )],
    );
    let state = [Value::int(0), Value::int(20)];
    assert_eq!(
        BytecodeVm::new(&chunk, &state, None)
            .execute_action_function(0)
            .expect("later UNCHANGED tuple slot must remain prime-readable"),
        ActionVmOutcome::Enabled([(0, Value::int(1))].into_iter().collect())
    );
}

#[test]
fn false_nonboolean_and_runtime_error_all_discard_the_overlay() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [
            (
                vec![
                    Opcode::LoadImm { rd: 0, value: 9 },
                    Opcode::StoreVar { var_idx: 0, rs: 0 },
                    Opcode::LoadBool {
                        rd: 1,
                        value: false,
                    },
                    Opcode::Ret { rs: 1 },
                ],
                1,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 0, value: 9 },
                    Opcode::StoreVar { var_idx: 0, rs: 0 },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 0, value: 9 },
                    Opcode::StoreVar { var_idx: 0, rs: 0 },
                    Opcode::LoadBool { rd: 1, value: true },
                    Opcode::LoadImm { rd: 2, value: 1 },
                    Opcode::AddInt {
                        rd: 3,
                        r1: 1,
                        r2: 2,
                    },
                    Opcode::Ret { rs: 3 },
                ],
                3,
            ),
            (
                vec![
                    Opcode::LoadPrime { rd: 0, var_idx: 0 },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
        ],
    );
    let state = [Value::int(1)];
    let mut vm = BytecodeVm::new(&chunk, &state, None);

    assert_eq!(
        vm.execute_action_function(0).expect("disabled action"),
        ActionVmOutcome::Disabled
    );
    assert_unbound_prime(vm.execute_action_function(3));
    let normal_err = vm
        .execute_function(1)
        .expect_err("retained action scratch must not enable normal StoreVar");
    assert!(normal_err.to_string().contains("StoreVar"));

    assert!(vm.execute_action_function(1).is_err());
    assert_unbound_prime(vm.execute_action_function(3));

    assert!(vm.execute_action_function(2).is_err());
    assert_unbound_prime(vm.execute_action_function(3));
}

#[test]
fn enabled_changes_are_sorted_and_parent_equal_writes_are_omitted() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [(
            vec![
                Opcode::LoadImm { rd: 0, value: 33 },
                Opcode::StoreVar { var_idx: 3, rs: 0 },
                Opcode::LoadImm { rd: 0, value: 10 },
                Opcode::StoreVar { var_idx: 0, rs: 0 },
                Opcode::LoadImm { rd: 0, value: 11 },
                Opcode::StoreVar { var_idx: 1, rs: 0 },
                Opcode::LoadImm { rd: 0, value: 30 },
                Opcode::StoreVar { var_idx: 2, rs: 0 },
                Opcode::LoadBool { rd: 0, value: true },
                Opcode::Ret { rs: 0 },
            ],
            0,
        )],
    );
    let state = [
        Value::int(10),
        Value::int(20),
        Value::int(30),
        Value::int(40),
    ];
    assert_eq!(
        BytecodeVm::new(&chunk, &state, None)
            .execute_action_function(0)
            .expect("enabled action"),
        ActionVmOutcome::Enabled(
            [(1, Value::int(11)), (3, Value::int(33))]
                .into_iter()
                .collect()
        )
    );
}

#[test]
fn enabled_compound_changes_drain_in_slot_order_across_repeated_spilling_runs() {
    let parent_equal = Value::tuple([Value::string("same"), Value::int(0)]);
    let changed_one = Value::set([Value::int(1), Value::int(2)]);
    let changed_two = Value::tuple([Value::string("tuple"), Value::int(2)]);
    let changed_three = Value::string("changed");
    let changed_four = Value::int(44);
    let changed_five = Value::set([Value::string("five")]);

    let mut constants = ConstantPool::new();
    let parent_equal_idx = constants.add_value(parent_equal.clone());
    let changed_one_idx = constants.add_value(changed_one.clone());
    let changed_two_idx = constants.add_value(changed_two.clone());
    let changed_three_idx = constants.add_value(changed_three.clone());
    let changed_four_idx = constants.add_value(changed_four.clone());
    let changed_five_idx = constants.add_value(changed_five.clone());
    let chunk = chunk_with_functions(
        constants,
        [(
            vec![
                // Deliberately bind out of order. The outcome must retain the
                // canonical slot order of the action overlay.
                Opcode::LoadConst {
                    rd: 0,
                    idx: changed_five_idx,
                },
                Opcode::StoreVar { var_idx: 5, rs: 0 },
                Opcode::LoadConst {
                    rd: 0,
                    idx: parent_equal_idx,
                },
                Opcode::StoreVar { var_idx: 0, rs: 0 },
                Opcode::LoadConst {
                    rd: 0,
                    idx: changed_three_idx,
                },
                Opcode::StoreVar { var_idx: 3, rs: 0 },
                Opcode::LoadConst {
                    rd: 0,
                    idx: changed_one_idx,
                },
                Opcode::StoreVar { var_idx: 1, rs: 0 },
                Opcode::LoadConst {
                    rd: 0,
                    idx: changed_four_idx,
                },
                Opcode::StoreVar { var_idx: 4, rs: 0 },
                Opcode::LoadConst {
                    rd: 0,
                    idx: changed_two_idx,
                },
                Opcode::StoreVar { var_idx: 2, rs: 0 },
                Opcode::LoadBool { rd: 0, value: true },
                Opcode::Ret { rs: 0 },
            ],
            0,
        )],
    );
    let state = [
        parent_equal,
        Value::Bool(false),
        Value::Bool(false),
        Value::Bool(false),
        Value::Bool(false),
        Value::Bool(false),
    ];
    let expected = ActionVmOutcome::Enabled(
        [
            (1, changed_one),
            (2, changed_two),
            (3, changed_three),
            (4, changed_four),
            (5, changed_five),
        ]
        .into_iter()
        .collect(),
    );
    let mut vm = BytecodeVm::new(&chunk, &state, None);

    for run in 0..2 {
        let outcome = vm
            .execute_action_function(0)
            .unwrap_or_else(|error| panic!("enabled action run {run} failed: {error}"));
        let ActionVmOutcome::Enabled(changes) = &outcome else {
            panic!("enabled action run {run} unexpectedly returned FALSE");
        };
        assert!(
            changes.spilled(),
            "five changes must exercise the outcome SmallVec spill path"
        );
        assert_eq!(outcome, expected);
    }
}

#[test]
fn enabled_action_requires_every_successor_slot_to_be_bound() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [(
            vec![
                Opcode::LoadImm { rd: 0, value: 9 },
                Opcode::StoreVar { var_idx: 0, rs: 0 },
                Opcode::LoadBool { rd: 0, value: true },
                Opcode::Ret { rs: 0 },
            ],
            0,
        )],
    );
    let state = [Value::int(1), Value::int(2)];
    let err = BytecodeVm::new(&chunk, &state, None)
        .execute_action_function(0)
        .expect_err("partial successors must fail closed");
    assert!(err
        .to_string()
        .contains("without binding all 2 successor slots"));
}

#[test]
fn disabled_action_may_leave_successor_slots_unbound() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [(
            vec![
                Opcode::LoadImm { rd: 0, value: 9 },
                Opcode::StoreVar { var_idx: 0, rs: 0 },
                Opcode::LoadBool {
                    rd: 0,
                    value: false,
                },
                Opcode::Ret { rs: 0 },
            ],
            0,
        )],
    );
    let state = [Value::int(1), Value::int(2)];

    assert_eq!(
        BytecodeVm::new(&chunk, &state, None)
            .execute_action_function(0)
            .expect("FALSE action with a partial binding"),
        ActionVmOutcome::Disabled
    );
}

#[test]
fn action_mode_rejects_dynamic_evaluation_opcodes() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [
            (
                vec![
                    Opcode::CallExternal {
                        rd: 0,
                        name_idx: 0,
                        args_start: 0,
                        argc: 0,
                        self_recursive: false,
                    },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
            (
                vec![
                    Opcode::ValueApply {
                        rd: 0,
                        func: 0,
                        args_start: 0,
                        argc: 0,
                    },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
            (
                vec![
                    Opcode::MakeClosure {
                        rd: 0,
                        template_idx: 0,
                        captures_start: 0,
                        capture_count: 0,
                    },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
        ],
    );
    let state = [];
    let mut vm = BytecodeVm::new(&chunk, &state, None);
    for (func_idx, opcode_name) in [(0, "CallExternal"), (1, "ValueApply"), (2, "MakeClosure")] {
        let err = vm
            .execute_action_function(func_idx)
            .expect_err("dynamic evaluation must fail closed in action mode");
        assert!(
            err.to_string().contains(opcode_name),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn set_prime_mode_fails_closed_and_cannot_leak_to_normal_execution() {
    let chunk = chunk_with_functions(
        ConstantPool::new(),
        [
            (
                vec![
                    Opcode::SetPrimeMode { enable: true },
                    Opcode::LoadBool { rd: 0, value: true },
                    Opcode::Ret { rs: 0 },
                ],
                0,
            ),
            (
                vec![Opcode::LoadVar { rd: 0, var_idx: 0 }, Opcode::Ret { rs: 0 }],
                0,
            ),
        ],
    );
    let state = [Value::int(1)];
    let next_state = [Value::int(2)];
    let mut vm = BytecodeVm::new(&chunk, &state, Some(&next_state));
    let err = vm
        .execute_action_function(0)
        .expect_err("SetPrimeMode must be rejected in action mode");
    assert!(err.to_string().contains("SetPrimeMode"));
    assert_eq!(
        vm.execute_function(1)
            .expect("normal load after action error"),
        Value::int(1)
    );
}

#[test]
fn action_exists_uses_tlc_normalized_domain_order() {
    let mut constants = ConstantPool::new();
    let tuple_len_two = constants.add_value(Value::tuple([Value::int(1), Value::int(1)]));
    let tuple_len_one = constants.add_value(Value::tuple([Value::int(2)]));
    let chunk = chunk_with_functions(
        constants,
        [(
            vec![
                Opcode::LoadConst {
                    rd: 0,
                    idx: tuple_len_two,
                },
                Opcode::LoadConst {
                    rd: 1,
                    idx: tuple_len_one,
                },
                Opcode::SetEnum {
                    rd: 2,
                    start: 0,
                    count: 2,
                },
                Opcode::ExistsBegin {
                    rd: 3,
                    r_binding: 4,
                    r_domain: 2,
                    loop_end: 4,
                },
                Opcode::StoreVar { var_idx: 0, rs: 4 },
                Opcode::LoadBool { rd: 5, value: true },
                Opcode::ExistsNext {
                    rd: 3,
                    r_binding: 4,
                    r_body: 5,
                    loop_begin: -2,
                },
                Opcode::Ret { rs: 3 },
            ],
            5,
        )],
    );
    let state = [Value::Bool(false)];
    assert_eq!(
        BytecodeVm::new(&chunk, &state, None)
            .execute_action_function(0)
            .expect("normalized EXISTS action"),
        ActionVmOutcome::Enabled([(0, Value::tuple([Value::int(2)]))].into_iter().collect())
    );
}

#[test]
fn context_free_action_set_diff_retries_only_for_non_materializable_values() {
    let mut constants = ConstantPool::new();
    let concrete_left = constants.add_value(Value::set([Value::int(1), Value::int(2)]));
    let concrete_right = constants.add_value(Value::set([Value::int(2)]));
    let lazy_left = constants.add_value(Value::StringSet);
    let empty_right = constants.add_value(Value::empty_set());
    let action = |left, right| {
        (
            vec![
                Opcode::LoadConst { rd: 0, idx: left },
                Opcode::LoadConst { rd: 1, idx: right },
                Opcode::SetDiff {
                    rd: 2,
                    r1: 0,
                    r2: 1,
                },
                Opcode::StoreVar { var_idx: 0, rs: 2 },
                Opcode::LoadBool { rd: 3, value: true },
                Opcode::Ret { rs: 3 },
            ],
            3,
        )
    };
    let chunk = chunk_with_functions(
        constants,
        [
            action(concrete_left, concrete_right),
            action(lazy_left, empty_right),
        ],
    );
    let state = [Value::empty_set()];
    let mut vm = BytecodeVm::new(&chunk, &state, None);
    let expected =
        ActionVmOutcome::Enabled([(0, Value::set([Value::int(1)]))].into_iter().collect());

    assert_eq!(
        vm.execute_action_function(0)
            .expect("concrete SetDiff must remain context-free"),
        expected
    );
    assert!(matches!(
        vm.execute_action_function(1),
        Err(VmError::NeedsEvalCtx("non-materializable set difference"))
    ));
    assert_eq!(
        vm.execute_action_function(0)
            .expect("NeedsEvalCtx must clear the transactional action overlay"),
        expected
    );
}

#[test]
fn certified_action_register_reuse_handles_grow_shrink_and_error_boundaries() {
    let mut constants = ConstantPool::new();
    let compound = constants.add_value(Value::tuple([
        Value::set([Value::int(1), Value::int(2)]),
        Value::string("stale scratch"),
    ]));
    let chunk = chunk_with_functions(
        constants,
        [
            (
                vec![
                    Opcode::LoadConst {
                        rd: 8,
                        idx: compound,
                    },
                    Opcode::LoadBool {
                        rd: 0,
                        value: false,
                    },
                    Opcode::Ret { rs: 0 },
                ],
                8,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 0, value: 9 },
                    Opcode::StoreVar { var_idx: 0, rs: 0 },
                    Opcode::LoadBool { rd: 1, value: true },
                    Opcode::Ret { rs: 1 },
                ],
                1,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 0, value: 12 },
                    Opcode::StoreVar { var_idx: 0, rs: 0 },
                    Opcode::LoadBool { rd: 9, value: true },
                    Opcode::LoadImm { rd: 10, value: 1 },
                    Opcode::AddInt {
                        rd: 11,
                        r1: 9,
                        r2: 10,
                    },
                    Opcode::Ret { rs: 11 },
                ],
                11,
            ),
            (
                vec![
                    Opcode::LoadImm { rd: 1, value: 15 },
                    Opcode::StoreVar { var_idx: 0, rs: 1 },
                    Opcode::LoadBool { rd: 0, value: true },
                    Opcode::Ret { rs: 0 },
                ],
                1,
            ),
            (vec![], 0),
        ],
    );
    let state = [Value::int(0)];
    let mut vm = BytecodeVm::new(&chunk, &state, None);

    assert_eq!(
        vm.execute_action_function_reusing_registers(0)
            .expect("large disabled certified entry"),
        ActionVmOutcome::Disabled
    );
    let expected = ActionVmOutcome::Enabled([(0, Value::int(9))].into_iter().collect());
    assert_eq!(
        vm.execute_action_function_reusing_registers(1)
            .expect("smaller certified entry after retained compound scratch"),
        expected
    );
    assert!(vm.execute_action_function_reusing_registers(2).is_err());
    assert_eq!(
        vm.execute_action_function_reusing_registers(1)
            .expect("an error must clear the overlay before the next certified entry"),
        expected
    );
    assert_eq!(
        vm.execute_action_function_reusing_registers(3)
            .expect("certified entry should leave r0 TRUE in retained scratch"),
        ActionVmOutcome::Enabled([(0, Value::int(15))].into_iter().collect())
    );
    assert_eq!(
        vm.execute_action_function(4)
            .expect("ordinary entry must reset stale r0 to its default FALSE"),
        ActionVmOutcome::Disabled
    );
}
