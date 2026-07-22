// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Scalar opcode smoke tests: LoadImm, LoadBool, arithmetic, comparison,
//! boolean/control flow, Move, division-by-zero, and constant-pool loading.

use super::{
    exec, exec_simple, BytecodeChunk, BytecodeFunction, BytecodeVm, ConstantPool, Opcode, Value,
};

#[test]
fn test_vm_load_imm_ret() {
    let result = exec_simple(
        vec![Opcode::LoadImm { rd: 0, value: 42 }, Opcode::Ret { rs: 0 }],
        0,
    );
    assert_eq!(result, Value::SmallInt(42));
}

#[test]
fn test_vm_load_bool() {
    let result = exec_simple(
        vec![
            Opcode::LoadBool { rd: 0, value: true },
            Opcode::Ret { rs: 0 },
        ],
        0,
    );
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_vm_add_int() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 10 },
            Opcode::LoadImm { rd: 1, value: 32 },
            Opcode::AddInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::SmallInt(42));
}

#[test]
fn test_vm_sub_int() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 50 },
            Opcode::LoadImm { rd: 1, value: 8 },
            Opcode::SubInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::SmallInt(42));
}

#[test]
fn test_vm_mul_int() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 6 },
            Opcode::LoadImm { rd: 1, value: 7 },
            Opcode::MulInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::SmallInt(42));
}

#[test]
fn test_vm_pow_int_basic() {
    // 2 ^ 10 = 1024 (SmallInt fast path).
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 2 },
            Opcode::LoadImm { rd: 1, value: 10 },
            Opcode::PowInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::SmallInt(1024));
}

#[test]
fn test_vm_pow_int_zero_zero_is_one() {
    // 0 ^ 0 = 1 (TLA+ / BigInt::pow convention, mirrored by const-prop).
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 0 },
            Opcode::LoadImm { rd: 1, value: 0 },
            Opcode::PowInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::SmallInt(1));
}

#[test]
fn test_vm_pow_int_negative_base_odd_exp() {
    // (-2) ^ 3 = -8.
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: -2 },
            Opcode::LoadImm { rd: 1, value: 3 },
            Opcode::PowInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::SmallInt(-8));
}

#[test]
fn test_vm_pow_int_overflows_to_bigint() {
    // 2 ^ 70 exceeds i64 and the VM uses arbitrary precision (BigInt). This is
    // the case the trust-cg direct-LLVM path must NOT compute in i64: it traps
    // instead of producing a divergent (wrapped/truncated) value.
    use num_bigint::BigInt;
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 2 },
            Opcode::LoadImm { rd: 1, value: 70 },
            Opcode::PowInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    let expected = BigInt::from(2u8).pow(70);
    assert_eq!(result, Value::big_int(expected));
    // And it is genuinely a big (non-i64) value.
    assert!(matches!(result, Value::Int(_)));
}

#[test]
fn test_vm_pow_int_negative_exponent_errors() {
    // Negative exponents are rejected by the interpreter (`to_u32()` fails);
    // the VM must surface an error, never a bogus value.
    let mut func = BytecodeFunction::new("test".to_string(), 0);
    func.max_register = 2;
    func.instructions = vec![
        Opcode::LoadImm { rd: 0, value: 2 },
        Opcode::LoadImm { rd: 1, value: -1 },
        Opcode::PowInt {
            rd: 2,
            r1: 0,
            r2: 1,
        },
        Opcode::Ret { rs: 2 },
    ];
    let mut chunk = BytecodeChunk::new();
    chunk.add_function(func);
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    let result = vm.execute_function(0);
    assert!(
        result.is_err(),
        "negative exponent must produce an error, got {result:?}"
    );
}

#[test]
fn test_vm_eq_true() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 5 },
            Opcode::LoadImm { rd: 1, value: 5 },
            Opcode::Eq {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_vm_eq_false() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 5 },
            Opcode::LoadImm { rd: 1, value: 6 },
            Opcode::Eq {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_vm_lt_int() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 3 },
            Opcode::LoadImm { rd: 1, value: 5 },
            Opcode::LtInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_vm_and_short_circuit() {
    // FALSE /\ anything = FALSE (via JumpFalse)
    let result = exec_simple(
        vec![
            Opcode::LoadBool {
                rd: 0,
                value: false,
            },
            Opcode::Move { rd: 1, rs: 0 },
            Opcode::JumpFalse { rs: 1, offset: 3 }, // skip to Ret
            Opcode::LoadBool { rd: 2, value: true },
            Opcode::Move { rd: 1, rs: 2 },
            // end:
            Opcode::Ret { rs: 1 },
        ],
        2,
    );
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_vm_if_then_else() {
    // IF TRUE THEN 1 ELSE 2
    let result = exec_simple(
        vec![
            Opcode::LoadBool { rd: 0, value: true },
            Opcode::JumpFalse { rs: 0, offset: 4 }, // jump to else
            // then:
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::Move { rd: 2, rs: 1 },
            Opcode::Jump { offset: 3 }, // jump to end
            // else:
            Opcode::LoadImm { rd: 3, value: 2 },
            Opcode::Move { rd: 2, rs: 3 },
            // end:
            Opcode::Ret { rs: 2 },
        ],
        3,
    );
    assert_eq!(result, Value::SmallInt(1));
}

#[test]
fn test_vm_neg_int() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 42 },
            Opcode::NegInt { rd: 1, rs: 0 },
            Opcode::Ret { rs: 1 },
        ],
        1,
    );
    assert_eq!(result, Value::SmallInt(-42));
}

#[test]
fn test_vm_not_bool() {
    let result = exec_simple(
        vec![
            Opcode::LoadBool { rd: 0, value: true },
            Opcode::Not { rd: 1, rs: 0 },
            Opcode::Ret { rs: 1 },
        ],
        1,
    );
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_vm_implies() {
    // FALSE => anything = TRUE
    let result = exec_simple(
        vec![
            Opcode::LoadBool {
                rd: 0,
                value: false,
            },
            Opcode::LoadBool {
                rd: 1,
                value: false,
            },
            Opcode::Implies {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_vm_move() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 99 },
            Opcode::Move { rd: 1, rs: 0 },
            Opcode::Ret { rs: 1 },
        ],
        1,
    );
    assert_eq!(result, Value::SmallInt(99));
}

#[test]
fn test_vm_division_by_zero() {
    let mut func = BytecodeFunction::new("test".to_string(), 0);
    func.max_register = 2;
    func.instructions = vec![
        Opcode::LoadImm { rd: 0, value: 10 },
        Opcode::LoadImm { rd: 1, value: 0 },
        Opcode::DivInt {
            rd: 2,
            r1: 0,
            r2: 1,
        },
        Opcode::Ret { rs: 2 },
    ];
    let mut chunk = BytecodeChunk::new();
    chunk.add_function(func);
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    let result = vm.execute_function(0);
    assert!(result.is_err());
}

/// `/` is TLA+ real division: on integers it is EXACT-OR-ERROR. Exact
/// quotients (positive and negative) must evaluate to the exact value.
#[test]
fn test_vm_real_division_exact() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: 8 },
            Opcode::LoadImm { rd: 1, value: 2 },
            Opcode::DivInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::SmallInt(4));

    let result = exec_simple(
        vec![
            Opcode::LoadImm { rd: 0, value: -7 },
            Opcode::LoadImm { rd: 1, value: 7 },
            Opcode::DivInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    assert_eq!(result, Value::SmallInt(-1));
}

/// An INEXACT `/` (7 / 2) is an evaluation error — never a truncation
/// (a truncated `7 / 2 = 3` has no TLA+ meaning). Must be the shared
/// ArgumentError so the VM agrees with the AST/TIR engines.
#[test]
fn test_vm_real_division_inexact_errors() {
    let mut func = BytecodeFunction::new("test".to_string(), 0);
    func.max_register = 2;
    func.instructions = vec![
        Opcode::LoadImm { rd: 0, value: 7 },
        Opcode::LoadImm { rd: 1, value: 2 },
        Opcode::DivInt {
            rd: 2,
            r1: 0,
            r2: 1,
        },
        Opcode::Ret { rs: 2 },
    ];
    let mut chunk = BytecodeChunk::new();
    chunk.add_function(func);
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    let err = vm
        .execute_function(0)
        .expect_err("7 / 2 is inexact and must error");
    match err {
        super::VmError::Eval(tla_value::error::EvalError::ArgumentError { ref op, .. }) => {
            assert_eq!(op, "/");
        }
        other => panic!("expected ArgumentError for inexact `/`, got: {other:?}"),
    }
}

/// i64::MIN / -1 must decline the SmallInt fast path (i64 overflow) and take
/// the BigInt path: -1 divides everything, so the division IS exact and the
/// result is the exact 2^63.
#[test]
fn test_vm_real_division_min_by_neg_one_promotes_to_bigint() {
    let result = exec_simple(
        vec![
            Opcode::LoadImm {
                rd: 0,
                value: i64::MIN,
            },
            Opcode::LoadImm { rd: 1, value: -1 },
            Opcode::DivInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ],
        2,
    );
    let expected = num_bigint::BigInt::from(i64::MAX) + 1;
    assert_eq!(result, Value::big_int(expected));
}

#[test]
fn test_vm_load_const() {
    let mut pool = ConstantPool::new();
    pool.add_value(Value::string("hello"));

    let mut func = BytecodeFunction::new("test".to_string(), 0);
    func.max_register = 0;
    func.instructions = vec![Opcode::LoadConst { rd: 0, idx: 0 }, Opcode::Ret { rs: 0 }];
    let result = exec(func, pool, &[]);
    assert_eq!(result, Value::string("hello"));
}

#[test]
fn test_vm_eq_set_values_compared_correctly() {
    // Value::PartialEq handles set comparison correctly via extensional
    // equality (eq_same_type/cmp delegation). No AST fallback needed.
    let mut pool = ConstantPool::new();
    let left = pool.add_value(Value::set([Value::int(1)]));
    let right = pool.add_value(Value::set(std::iter::empty::<Value>()));

    let mut func = BytecodeFunction::new("test".to_string(), 0);
    func.max_register = 2;
    func.instructions = vec![
        Opcode::LoadConst { rd: 0, idx: left },
        Opcode::LoadConst { rd: 1, idx: right },
        Opcode::Eq {
            rd: 2,
            r1: 0,
            r2: 1,
        },
        Opcode::Ret { rs: 2 },
    ];

    let result = exec(func, pool, &[]);
    assert_eq!(result, Value::Bool(false));
}
