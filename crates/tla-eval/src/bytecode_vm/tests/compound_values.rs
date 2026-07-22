// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compound-value construction tests: sets, tuples, records.

use super::{
    exec, exec_simple, intern_name, BytecodeChunk, BytecodeVm, ConstantPool, Opcode, SortedSet,
    Value, VmError,
};
use tla_tir::bytecode::BytecodeFunction;
use tla_value::Rp;
use tla_value::{BagValue, FuncSetValue, FuncValue, IntIntervalFunc, IntervalValue, SeqSetValue};

fn execute_tuple2_membership(set_value: Value, fused: bool) -> Result<Value, VmError> {
    let mut pool = ConstantPool::new();
    let set_idx = pool.add_value(set_value);
    let mut instructions = vec![
        Opcode::LoadImm { rd: 0, value: 10 },
        Opcode::LoadImm { rd: 1, value: 20 },
    ];
    let result_reg = if fused {
        instructions.extend([
            Opcode::LoadConst {
                rd: 2,
                idx: set_idx,
            },
            Opcode::Tuple2SetIn {
                rd: 3,
                first: 0,
                second: 1,
                set: 2,
            },
        ]);
        3
    } else {
        instructions.extend([
            Opcode::TupleNew {
                rd: 2,
                start: 0,
                count: 2,
            },
            Opcode::LoadConst {
                rd: 3,
                idx: set_idx,
            },
            Opcode::SetIn {
                rd: 4,
                elem: 2,
                set: 3,
            },
        ]);
        4
    };
    instructions.push(Opcode::Ret { rs: result_reg });

    let mut func = BytecodeFunction::new("tuple2-membership".to_string(), 0);
    func.max_register = result_reg;
    func.instructions = instructions;
    let mut chunk = BytecodeChunk::new();
    chunk.constants = pool;
    chunk.add_function(func);
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    vm.execute_function(0)
}

fn execute_set_enum_subseteq(
    elements: &[i64],
    set_value: Value,
    fused: bool,
) -> Result<Value, VmError> {
    let mut pool = ConstantPool::new();
    let set_idx = pool.add_value(set_value);
    let count = elements.len() as u8;
    let mut instructions: Vec<Opcode> = elements
        .iter()
        .enumerate()
        .map(|(index, value)| Opcode::LoadImm {
            rd: index as u8,
            value: *value,
        })
        .collect();

    let result_reg = if fused {
        let set_reg = count;
        let rd = set_reg + 1;
        instructions.extend([
            Opcode::LoadConst {
                rd: set_reg,
                idx: set_idx,
            },
            Opcode::SetEnumSubseteq {
                rd,
                start: 0,
                count,
                set: set_reg,
            },
        ]);
        rd
    } else {
        let enum_reg = count;
        let set_reg = enum_reg + 1;
        let rd = set_reg + 1;
        instructions.extend([
            Opcode::SetEnum {
                rd: enum_reg,
                start: 0,
                count,
            },
            Opcode::LoadConst {
                rd: set_reg,
                idx: set_idx,
            },
            Opcode::Subseteq {
                rd,
                r1: enum_reg,
                r2: set_reg,
            },
        ]);
        rd
    };
    instructions.push(Opcode::Ret { rs: result_reg });

    let mut func = BytecodeFunction::new("set-enum-subseteq".to_string(), 0);
    func.max_register = result_reg;
    func.instructions = instructions;
    let mut chunk = BytecodeChunk::new();
    chunk.constants = pool;
    chunk.add_function(func);
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    vm.execute_function(0)
}

fn execute_tuple2_self_eq(value: Value, fused: bool) -> Result<Value, VmError> {
    let mut pool = ConstantPool::new();
    let value_idx = pool.add_value(value);
    let mut instructions = vec![Opcode::LoadConst {
        rd: 0,
        idx: value_idx,
    }];
    let result_reg = if fused {
        instructions.push(Opcode::Tuple2SelfEq { rd: 1, value: 0 });
        1
    } else {
        // Historical `e = <<e[1], e[2]>>`: projection one, projection two,
        // tuple construction, then equality. r1/r2 are the tuple operand block.
        instructions.extend([
            Opcode::LoadImm { rd: 3, value: 1 },
            Opcode::FuncApply {
                rd: 4,
                func: 0,
                arg: 3,
            },
            Opcode::Move { rd: 1, rs: 4 },
            Opcode::LoadImm { rd: 5, value: 2 },
            Opcode::FuncApply {
                rd: 6,
                func: 0,
                arg: 5,
            },
            Opcode::Move { rd: 2, rs: 6 },
            Opcode::TupleNew {
                rd: 7,
                start: 1,
                count: 2,
            },
            Opcode::Eq {
                rd: 8,
                r1: 0,
                r2: 7,
            },
        ]);
        8
    };
    instructions.push(Opcode::Ret { rs: result_reg });

    let mut func = BytecodeFunction::new("tuple2-self-eq".to_string(), 0);
    func.max_register = result_reg;
    func.instructions = instructions;
    let mut chunk = BytecodeChunk::new();
    chunk.constants = pool;
    chunk.add_function(func);
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    vm.execute_function(0)
}

fn execute_tuple2_self_subseteq(
    value: Value,
    state: &[Value],
    next_state: Option<&[Value]>,
    set_var_idx: u16,
    prime_mode: bool,
    fused: bool,
) -> Result<Value, VmError> {
    let mut pool = ConstantPool::new();
    let value_idx = pool.add_value(value);
    let mut func = BytecodeFunction::new("tuple2-self-subseteq".to_string(), 0);
    if prime_mode {
        func.emit(Opcode::SetPrimeMode { enable: true });
    }
    func.emit(Opcode::LoadConst {
        rd: 0,
        idx: value_idx,
    });

    if fused {
        func.emit(Opcode::Tuple2SelfSubseteq {
            rd: 1,
            value: 0,
            set_var_idx,
        });
    } else {
        func.emit(Opcode::Tuple2SelfEq { rd: 1, value: 0 });
        let jump = func.emit(Opcode::JumpFalse { rs: 1, offset: 0 });
        func.emit(Opcode::LoadImm { rd: 4, value: 1 });
        func.emit(Opcode::FuncApply {
            rd: 2,
            func: 0,
            arg: 4,
        });
        func.emit(Opcode::LoadImm { rd: 5, value: 2 });
        func.emit(Opcode::FuncApply {
            rd: 3,
            func: 0,
            arg: 5,
        });
        func.emit(Opcode::LoadVar {
            rd: 6,
            var_idx: set_var_idx,
        });
        func.emit(Opcode::SetEnumSubseteq {
            rd: 1,
            start: 2,
            count: 2,
            set: 6,
        });
        let ret = func.len();
        func.patch_jump(jump, ret);
    }
    func.emit(Opcode::Ret { rs: 1 });

    let mut chunk = BytecodeChunk::new();
    chunk.constants = pool;
    chunk.add_function(func);
    let mut vm = BytecodeVm::new(&chunk, state, next_state);
    vm.execute_function(0)
}

#[test]
fn test_vm_tuple2_self_eq_direct_tuple_and_seq_parity() {
    let cases = [
        Value::tuple([Value::SmallInt(10), Value::SmallInt(20)]),
        Value::tuple([
            Value::SmallInt(10),
            Value::SmallInt(20),
            Value::SmallInt(30),
        ]),
        Value::seq([Value::SmallInt(10), Value::SmallInt(20)]),
        Value::seq([
            Value::SmallInt(10),
            Value::SmallInt(20),
            Value::SmallInt(30),
        ]),
    ];

    for value in cases {
        let baseline = execute_tuple2_self_eq(value.clone(), false).unwrap();
        let fused = execute_tuple2_self_eq(value, true).unwrap();
        assert_eq!(fused, baseline);
    }
}

#[test]
fn test_vm_tuple2_self_eq_function_representation_fallback_parity() {
    let values = vec![
        (
            Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
                (Value::SmallInt(1), Value::SmallInt(10)),
                (Value::SmallInt(2), Value::SmallInt(20)),
            ]))),
            true,
        ),
        (
            Value::IntFunc(Rp::new(IntIntervalFunc::new(
                1,
                2,
                vec![Value::SmallInt(10), Value::SmallInt(20)],
            ))),
            true,
        ),
        (
            Value::Bag(Rp::new(
                BagValue::try_from_entries(vec![
                    (Value::SmallInt(1), Value::SmallInt(10)),
                    (Value::SmallInt(2), Value::SmallInt(20)),
                ])
                .unwrap(),
            )),
            true,
        ),
        (
            Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
                (Value::SmallInt(1), Value::SmallInt(10)),
                (Value::SmallInt(2), Value::SmallInt(20)),
                (Value::SmallInt(3), Value::SmallInt(30)),
            ]))),
            false,
        ),
        (
            Value::IntFunc(Rp::new(IntIntervalFunc::new(
                1,
                3,
                vec![
                    Value::SmallInt(10),
                    Value::SmallInt(20),
                    Value::SmallInt(30),
                ],
            ))),
            false,
        ),
    ];

    for (value, expected) in values {
        let baseline = execute_tuple2_self_eq(value.clone(), false).unwrap();
        let fused = execute_tuple2_self_eq(value, true).unwrap();
        assert_eq!(baseline, Value::Bool(expected));
        assert_eq!(fused, baseline);
    }
}

#[test]
fn test_vm_tuple2_self_eq_preserves_projection_errors() {
    let values = vec![
        Value::tuple(std::iter::empty::<Value>()),
        Value::tuple([Value::SmallInt(10)]),
        Value::seq(std::iter::empty::<Value>()),
        Value::seq([Value::SmallInt(10)]),
        Value::SmallInt(10),
        Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![(
            Value::SmallInt(1),
            Value::SmallInt(10),
        )]))),
    ];

    for value in values {
        let baseline = execute_tuple2_self_eq(value.clone(), false)
            .expect_err("historical expression must surface a projection error");
        let fused = execute_tuple2_self_eq(value, true)
            .expect_err("fused expression must preserve the projection error");
        assert_eq!(fused.to_string(), baseline.to_string());
    }
}

#[test]
fn test_vm_tuple2_self_subseteq_direct_value_parity() {
    let cases = [
        (
            Value::tuple([Value::int(10), Value::int(20)]),
            Value::set([Value::int(10), Value::int(20)]),
            true,
        ),
        (
            Value::seq([Value::int(10), Value::int(20)]),
            Value::set([Value::int(10)]),
            false,
        ),
        (
            Value::tuple([Value::int(10), Value::int(10)]),
            Value::set([Value::int(10)]),
            true,
        ),
    ];

    for (value, set, expected) in cases {
        let state = [set];
        let baseline =
            execute_tuple2_self_subseteq(value.clone(), &state, None, 0, false, false).unwrap();
        let fused = execute_tuple2_self_subseteq(value, &state, None, 0, false, true).unwrap();
        assert_eq!(baseline, Value::Bool(expected));
        assert_eq!(fused, baseline);
    }
}

#[test]
fn test_vm_tuple2_self_subseteq_function_value_parity() {
    let exact = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (Value::int(1), Value::int(10)),
        (Value::int(2), Value::int(20)),
    ])));
    let extra = Value::IntFunc(Rp::new(IntIntervalFunc::new(
        1,
        3,
        vec![Value::int(10), Value::int(20), Value::int(30)],
    )));
    let valid_state = [Value::set([Value::int(10), Value::int(20)])];
    let invalid_state = [Value::int(7)];

    for (value, state, expected) in [
        (exact, valid_state.as_slice(), Value::Bool(true)),
        (extra, invalid_state.as_slice(), Value::Bool(false)),
    ] {
        let baseline =
            execute_tuple2_self_subseteq(value.clone(), state, None, 0, false, false).unwrap();
        let fused = execute_tuple2_self_subseteq(value, state, None, 0, false, true).unwrap();
        assert_eq!(baseline, expected);
        assert_eq!(fused, baseline);
    }
}

#[test]
fn test_vm_tuple2_self_subseteq_preserves_errors_and_defers_state_read() {
    for value in [
        Value::tuple(std::iter::empty::<Value>()),
        Value::tuple([Value::int(10)]),
        Value::seq(std::iter::empty::<Value>()),
        Value::seq([Value::int(10)]),
        Value::int(10),
    ] {
        let baseline =
            execute_tuple2_self_subseteq(value.clone(), &[], None, 4, false, false).unwrap_err();
        let fused = execute_tuple2_self_subseteq(value, &[], None, 4, false, true).unwrap_err();
        assert_eq!(fused.to_string(), baseline.to_string());
    }

    // A false shape must not touch even an out-of-range state slot.
    let long = Value::tuple([Value::int(10), Value::int(20), Value::int(30)]);
    assert_eq!(
        execute_tuple2_self_subseteq(long, &[], None, 4, false, true).unwrap(),
        Value::Bool(false)
    );

    // A true shape reaches the RHS and preserves its ordinary type error.
    let pair = Value::tuple([Value::int(10), Value::int(20)]);
    let invalid_state = [Value::int(7)];
    let baseline =
        execute_tuple2_self_subseteq(pair.clone(), &invalid_state, None, 0, false, false)
            .unwrap_err();
    let fused =
        execute_tuple2_self_subseteq(pair, &invalid_state, None, 0, false, true).unwrap_err();
    assert_eq!(fused.to_string(), baseline.to_string());
}

#[test]
fn test_vm_tuple2_self_subseteq_honors_dynamic_prime_mode() {
    let value = Value::tuple([Value::int(10), Value::int(20)]);
    let current = [Value::set([Value::int(10)])];
    let next = [Value::set([Value::int(10), Value::int(20)])];

    let baseline =
        execute_tuple2_self_subseteq(value.clone(), &current, Some(&next), 0, true, false).unwrap();
    let fused = execute_tuple2_self_subseteq(value, &current, Some(&next), 0, true, true).unwrap();
    assert_eq!(baseline, Value::Bool(true));
    assert_eq!(fused, baseline);
}

#[test]
fn test_vm_set_enum() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::LoadImm { rd: 1, value: 2 },
            Opcode::LoadImm { rd: 2, value: 3 },
            Opcode::SetEnum {
                rd: 3,
                start: 0,
                count: 3,
            },
            Opcode::Ret { rs: 3 },
        ],
        3,
    );
    let expected = Value::Set(Rp::new(SortedSet::from_iter(vec![
        Value::SmallInt(1),
        Value::SmallInt(2),
        Value::SmallInt(3),
    ])));
    assert_eq!(result, expected);
}

#[test]
fn test_vm_set_in_true() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::LoadImm { rd: 1, value: 2 },
            Opcode::LoadImm { rd: 2, value: 3 },
            Opcode::SetEnum {
                rd: 3,
                start: 0,
                count: 3,
            },
            Opcode::LoadImm { rd: 4, value: 2 },
            Opcode::SetIn {
                rd: 5,
                elem: 4,
                set: 3,
            },
            Opcode::Ret { rs: 5 },
        ],
        5,
    );
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_vm_set_in_false() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::LoadImm { rd: 1, value: 2 },
            Opcode::SetEnum {
                rd: 2,
                start: 0,
                count: 2,
            },
            Opcode::LoadImm { rd: 3, value: 5 },
            Opcode::SetIn {
                rd: 4,
                elem: 3,
                set: 2,
            },
            Opcode::Ret { rs: 4 },
        ],
        4,
    );
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_vm_tuple2_set_in_concrete_cross_representation_hit_and_miss() {
    let hit_set = Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![Value::seq([
        Value::int(10),
        Value::int(20),
    ])])));
    assert_eq!(
        execute_tuple2_membership(hit_set, true).unwrap(),
        Value::Bool(true)
    );

    let miss_set = Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![Value::seq([
        Value::int(10),
        Value::int(21),
    ])])));
    assert_eq!(
        execute_tuple2_membership(miss_set, true).unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn test_vm_tuple2_set_in_lazy_fallback_matches_materialized_path() {
    let seq_set = Value::SeqSet(SeqSetValue::new(Value::set([
        Value::int(10),
        Value::int(20),
    ])));
    let fused = execute_tuple2_membership(seq_set.clone(), true).unwrap();
    let materialized = execute_tuple2_membership(seq_set, false).unwrap();
    assert_eq!(fused, materialized);
    assert_eq!(fused, Value::Bool(true));
}

#[test]
fn test_vm_tuple2_set_in_type_error_matches_materialized_path() {
    let fused = execute_tuple2_membership(Value::int(7), true).unwrap_err();
    let materialized = execute_tuple2_membership(Value::int(7), false).unwrap_err();
    assert_eq!(fused.to_string(), materialized.to_string());
}

#[test]
fn test_vm_set_enum_subseteq_concrete_matches_materialized_path() {
    let cases: &[(&[i64], bool)] = &[
        (&[1, 2], true),
        (&[1, 4], false),
        (&[1, 1], true),
        (&[], true),
    ];
    for (elements, expected) in cases {
        let set = Value::set([Value::int(1), Value::int(2), Value::int(3)]);
        let fused = execute_set_enum_subseteq(elements, set.clone(), true).unwrap();
        let materialized = execute_set_enum_subseteq(elements, set, false).unwrap();
        assert_eq!(fused, materialized);
        assert_eq!(fused, Value::Bool(*expected));
    }
}

#[test]
fn test_vm_set_enum_subseteq_lazy_fallback_matches_materialized_path() {
    let interval = Value::Interval(Rp::new(IntervalValue::new(1.into(), 3.into())));
    let fused = execute_set_enum_subseteq(&[1, 2], interval.clone(), true).unwrap();
    let materialized = execute_set_enum_subseteq(&[1, 2], interval, false).unwrap();
    assert_eq!(fused, materialized);
    assert_eq!(fused, Value::Bool(true));
}

#[test]
fn test_vm_set_enum_subseteq_large_fallback_matches_materialized_path() {
    let set = Value::set([Value::int(1), Value::int(2), Value::int(3), Value::int(4)]);
    let fused = execute_set_enum_subseteq(&[1, 2, 3], set.clone(), true).unwrap();
    let materialized = execute_set_enum_subseteq(&[1, 2, 3], set, false).unwrap();
    assert_eq!(fused, materialized);
    assert_eq!(fused, Value::Bool(true));
}

#[test]
fn test_vm_set_enum_subseteq_type_error_matches_materialized_path() {
    let fused = execute_set_enum_subseteq(&[1, 2], Value::int(7), true).unwrap_err();
    let materialized = execute_set_enum_subseteq(&[1, 2], Value::int(7), false).unwrap_err();
    assert_eq!(fused.to_string(), materialized.to_string());
}

#[test]
fn test_vm_powerset_preserves_lazy_base_without_enumerating() {
    let mut pool = ConstantPool::new();
    let nat_idx = pool.add_value(Value::ModelValue(Rp::from("Nat")));

    let mut func = BytecodeFunction::new("test".to_string(), 0);
    func.max_register = 1;
    func.instructions = vec![
        Opcode::LoadConst {
            rd: 0,
            idx: nat_idx,
        },
        Opcode::Powerset { rd: 1, rs: 0 },
        Opcode::Ret { rs: 1 },
    ];

    let result = exec(func, pool, &[]);
    let Value::Subset(ref subset) = result else {
        panic!("SUBSET should remain lazy in bytecode VM, got {result:?}");
    };
    assert!(
        matches!(subset.base(), Value::ModelValue(name) if name.as_ref() == "Nat"),
        "lazy SUBSET should preserve its base without materializing"
    );
}

#[test]
fn test_vm_set_union() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::LoadImm { rd: 1, value: 2 },
            Opcode::SetEnum {
                rd: 2,
                start: 0,
                count: 2,
            },
            Opcode::LoadImm { rd: 3, value: 2 },
            Opcode::LoadImm { rd: 4, value: 3 },
            Opcode::SetEnum {
                rd: 5,
                start: 3,
                count: 2,
            },
            Opcode::SetUnion {
                rd: 6,
                r1: 2,
                r2: 5,
            },
            Opcode::Ret { rs: 6 },
        ],
        6,
    );
    let expected = Value::Set(Rp::new(SortedSet::from_iter(vec![
        Value::SmallInt(1),
        Value::SmallInt(2),
        Value::SmallInt(3),
    ])));
    assert_eq!(result, expected);
}

#[test]
fn test_vm_big_union_singleton_preserves_lazy_inner_set() {
    let domain = Value::set([Value::SmallInt(0), Value::SmallInt(1)]);
    let inner = Value::FuncSet(FuncSetValue::new(
        domain,
        Value::ModelValue(Rp::from("Nat")),
    ));

    let mut pool = ConstantPool::new();
    let inner_idx = pool.add_value(inner);

    let mut func = BytecodeFunction::new("test".to_string(), 0);
    func.max_register = 2;
    func.instructions = vec![
        Opcode::LoadConst {
            rd: 0,
            idx: inner_idx,
        },
        Opcode::SetEnum {
            rd: 1,
            start: 0,
            count: 1,
        },
        Opcode::BigUnion { rd: 2, rs: 1 },
        Opcode::Ret { rs: 2 },
    ];

    let result = exec(func, pool, &[]);
    assert!(
        matches!(result, Value::FuncSet(_)),
        "singleton UNION should preserve the lazy inner set, got {result:?}"
    );
}

#[test]
fn test_vm_tuple_new() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 10 },
            Opcode::LoadImm { rd: 1, value: 20 },
            Opcode::TupleNew {
                rd: 2,
                start: 0,
                count: 2,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(
        result,
        Value::Tuple(vec![Value::SmallInt(10), Value::SmallInt(20)].into())
    );
}

#[test]
fn test_vm_empty_tuple_new_uses_canonical_pool() {
    let result = exec_simple(
        vec![
            Opcode::TupleNew {
                rd: 0,
                start: 0,
                count: 0,
            },
            Opcode::Ret { rs: 0 },
        ],
        0,
    );
    let canonical = Value::tuple(std::iter::empty::<Value>());
    assert_eq!(result, canonical);
    assert!(
        result.ptr_eq(&canonical),
        "TupleNew(count=0) should reuse the canonical empty tuple allocation"
    );
}

#[test]
fn test_vm_record_new_then_get_field() {
    let mut pool = ConstantPool::new();
    let fields_start = pool.add_value(Value::string("foo"));
    let field_idx = pool.add_field_id(intern_name("foo").0);

    let mut func = BytecodeFunction::new("test".to_string(), 0);
    func.max_register = 2;
    func.instructions = vec![
        Opcode::LoadImm { rd: 0, value: 42 },
        Opcode::RecordNew {
            rd: 1,
            fields_start,
            values_start: 0,
            count: 1,
        },
        Opcode::RecordGet {
            rd: 2,
            rs: 1,
            field_idx,
        },
        Opcode::Ret { rs: 2 },
    ];

    let result = exec(func, pool, &[]);
    assert_eq!(result, Value::SmallInt(42));
}

// ===========================================================================
// value_except (FuncExcept opcode helper) — TLC parity for out-of-domain
// updates: [f EXCEPT ![k] = v] with k \notin DOMAIN f is a no-op, and a
// record EXCEPT on a missing field must never grow the record.
// ===========================================================================

#[test]
fn test_vm_value_except_tuple_out_of_range_is_noop() {
    use crate::bytecode_vm::execute_helpers::value_except;

    let t = Value::Tuple(vec![Value::SmallInt(1), Value::SmallInt(2)].into());
    for idx in [0i64, 5, -1] {
        let r = value_except(t.clone(), Value::SmallInt(idx), Value::SmallInt(9))
            .expect("out-of-range tuple EXCEPT should be a no-op, not an error");
        assert_eq!(r, t, "tuple must be unchanged for index {idx}");
    }

    // In-range index still updates.
    let r = value_except(t.clone(), Value::SmallInt(2), Value::SmallInt(9)).unwrap();
    assert_eq!(
        r,
        Value::Tuple(vec![Value::SmallInt(1), Value::SmallInt(9)].into())
    );

    // A non-integer index is still a type error.
    assert!(value_except(t, Value::Bool(true), Value::SmallInt(9)).is_err());
}

#[test]
fn test_vm_value_except_seq_out_of_range_is_noop() {
    use crate::bytecode_vm::execute_helpers::value_except;

    let s = Value::Seq(Rp::new(vec![Value::SmallInt(7)].into()));
    for idx in [0i64, 2, -3] {
        let r = value_except(s.clone(), Value::SmallInt(idx), Value::SmallInt(9))
            .expect("out-of-range seq EXCEPT should be a no-op, not an error");
        assert_eq!(r, s, "sequence must be unchanged for index {idx}");
    }

    // In-range index still updates.
    let r = value_except(s.clone(), Value::SmallInt(1), Value::SmallInt(9)).unwrap();
    assert_eq!(r, Value::Seq(Rp::new(vec![Value::SmallInt(9)].into())));

    // A non-integer index is still a type error.
    assert!(value_except(s, Value::string("x"), Value::SmallInt(9)).is_err());
}

#[test]
fn test_vm_value_except_record_missing_field_is_noop() {
    use crate::bytecode_vm::execute_helpers::value_except;
    use tla_value::RecordValue;

    let rec = Value::Record(RecordValue::from_sorted_entries(vec![(
        intern_name("a"),
        Value::SmallInt(1),
    )]));

    // Missing field: no-op — must NOT insert/grow the record (TLC + the
    // tree-walker both leave the record unchanged).
    let r = value_except(rec.clone(), Value::string("b"), Value::SmallInt(2))
        .expect("missing-field record EXCEPT should be a no-op");
    assert_eq!(r, rec);
    let Value::Record(inner) = &r else {
        panic!("expected a record, got {r:?}");
    };
    assert_eq!(inner.len(), 1, "record must not grow a new field");
    assert!(inner.get("b").is_none());

    // Existing field still updates.
    let r = value_except(rec, Value::string("a"), Value::SmallInt(5)).unwrap();
    let Value::Record(inner) = &r else {
        panic!("expected a record, got {r:?}");
    };
    assert_eq!(inner.get("a"), Some(&Value::SmallInt(5)));
    assert_eq!(inner.len(), 1);

    // A non-string field name is still a type error.
    let rec2 = Value::Record(RecordValue::from_sorted_entries(vec![(
        intern_name("a"),
        Value::SmallInt(1),
    )]));
    assert!(value_except(rec2, Value::SmallInt(1), Value::SmallInt(5)).is_err());
}
