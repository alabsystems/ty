// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Differential tests for the fused Eq superinstructions
//! (`EqFuncExcept` / `EqRecordNew`): every case executes BOTH the unfused
//! producer+Eq pair and the fused opcode over the same inputs and asserts
//! identical results — the fused opcodes' semantic contract.

use super::super::execute::{BytecodeVm, VmError};
use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};
use tla_value::Rp;
use tla_value::{IntIntervalFunc, RecordValue, Value};

/// Execute one function over `regs`-preloaded values by emitting LoadConst
/// prologs; returns the raw VM result.
fn run(instructions: Vec<Opcode>, constants: Vec<Value>) -> Result<Value, VmError> {
    let mut chunk = BytecodeChunk::new();
    for c in constants {
        chunk.constants.add_value(c);
    }
    let mut func = BytecodeFunction::new("t".to_string(), 0);
    func.max_register = 16;
    func.instructions = instructions;
    chunk.add_function(func);
    let mut vm = BytecodeVm::new(&chunk, &[], None);
    vm.execute_function(0)
}

/// Differential: unfused (FuncExcept r3 <- f,k,v; Eq r4 <- lhs,r3) vs fused
/// (EqFuncExcept r4 <- lhs,f,k,v) with operands loaded from the pool.
fn diff_func_except(lhs: Value, f: Value, k: Value, v: Value) {
    let constants = vec![lhs, f, k, v];
    let prolog = |()| {
        vec![
            Opcode::LoadConst { rd: 0, idx: 0 },
            Opcode::LoadConst { rd: 1, idx: 1 },
            Opcode::LoadConst { rd: 2, idx: 2 },
            Opcode::LoadConst { rd: 3, idx: 3 },
        ]
    };
    let mut unfused = prolog(());
    unfused.push(Opcode::FuncExcept {
        rd: 4,
        func: 1,
        path: 2,
        val: 3,
    });
    unfused.push(Opcode::Eq {
        rd: 5,
        r1: 0,
        r2: 4,
    });
    unfused.push(Opcode::Ret { rs: 5 });
    let mut fused = prolog(());
    fused.push(Opcode::EqFuncExcept {
        rd: 4,
        lhs: 0,
        func: 1,
        path: 2,
        val: 3,
    });
    fused.push(Opcode::Ret { rs: 4 });

    let unfused_result = run(unfused, constants.clone());
    let fused_result = run(fused, constants);
    match (&unfused_result, &fused_result) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "fused/unfused verdict divergence"),
        (Err(_), Err(_)) => {} // both error — same failure class
        other => panic!("fused/unfused outcome divergence: {other:?}"),
    }
}

fn intfunc(min: i64, vals: &[i64]) -> Value {
    Value::IntFunc(Rp::new(IntIntervalFunc::new(
        min,
        min + vals.len() as i64 - 1,
        vals.iter().map(|&v| Value::SmallInt(v)).collect(),
    )))
}

#[test]
fn eq_func_except_intfunc_matrix() {
    let f = intfunc(0, &[1, 2, 3]);
    // Equal: g = f EXCEPT ![1] = 9
    diff_func_except(
        intfunc(0, &[1, 9, 3]),
        f.clone(),
        Value::SmallInt(1),
        Value::SmallInt(9),
    );
    // Unequal at the except position.
    diff_func_except(
        intfunc(0, &[1, 8, 3]),
        f.clone(),
        Value::SmallInt(1),
        Value::SmallInt(9),
    );
    // Unequal at a non-except position.
    diff_func_except(
        intfunc(0, &[7, 9, 3]),
        f.clone(),
        Value::SmallInt(1),
        Value::SmallInt(9),
    );
    // Same backing buffer + except writes the same value back.
    diff_func_except(f.clone(), f.clone(), Value::SmallInt(1), Value::SmallInt(2));
    // Same backing buffer, except changes the value -> unequal.
    diff_func_except(f.clone(), f.clone(), Value::SmallInt(1), Value::SmallInt(9));
    // Domain mismatch (different min).
    diff_func_except(
        intfunc(1, &[1, 9, 3]),
        f.clone(),
        Value::SmallInt(1),
        Value::SmallInt(9),
    );
    // Domain length mismatch.
    diff_func_except(
        intfunc(0, &[1, 9]),
        f.clone(),
        Value::SmallInt(1),
        Value::SmallInt(9),
    );
    // Out-of-domain key (slow-path delegation).
    diff_func_except(
        intfunc(0, &[1, 2, 3]),
        f.clone(),
        Value::SmallInt(77),
        Value::SmallInt(9),
    );
    // Non-integer key (slow-path delegation).
    diff_func_except(
        intfunc(0, &[1, 2, 3]),
        f.clone(),
        Value::String("x".into()),
        Value::SmallInt(9),
    );
    // Non-function base (both paths must error identically).
    diff_func_except(
        intfunc(0, &[1, 2, 3]),
        Value::SmallInt(5),
        Value::SmallInt(0),
        Value::SmallInt(9),
    );
    // Cross-representation lhs (record vs func result) — slow path decides.
    diff_func_except(
        Value::Record(RecordValue::from_entries(vec![(
            tla_core::intern_name("a"),
            Value::SmallInt(1),
        )])),
        f,
        Value::SmallInt(1),
        Value::SmallInt(9),
    );
}

/// Differential for EqRecordNew: unfused (RecordNew; Eq) vs fused.
fn diff_record_new(lhs: Value, names: &[&str], vals: &[Value]) {
    assert_eq!(names.len(), vals.len());
    let mut constants = vec![lhs];
    // Field-name strings occupy pool slots 1..=n.
    for n in names {
        constants.push(Value::String(Rp::from(*n)));
    }
    let val_base = constants.len() as u16;
    for v in vals {
        constants.push(v.clone());
    }
    let mut prolog = vec![Opcode::LoadConst { rd: 0, idx: 0 }];
    for (i, _) in vals.iter().enumerate() {
        prolog.push(Opcode::LoadConst {
            rd: 1 + i as u8,
            idx: val_base + i as u16,
        });
    }
    let count = names.len() as u8;
    let mut unfused = prolog.clone();
    unfused.push(Opcode::RecordNew {
        rd: 10,
        fields_start: 1,
        values_start: 1,
        count,
    });
    unfused.push(Opcode::Eq {
        rd: 11,
        r1: 0,
        r2: 10,
    });
    unfused.push(Opcode::Ret { rs: 11 });
    let mut fused = prolog;
    fused.push(Opcode::EqRecordNew {
        rd: 10,
        lhs: 0,
        fields_start: 1,
        values_start: 1,
        count,
    });
    fused.push(Opcode::Ret { rs: 10 });

    let unfused_result = run(unfused, constants.clone());
    let fused_result = run(fused, constants);
    match (&unfused_result, &fused_result) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "fused/unfused verdict divergence"),
        (Err(_), Err(_)) => {}
        other => panic!("fused/unfused outcome divergence: {other:?}"),
    }
}

fn record(pairs: &[(&str, Value)]) -> Value {
    Value::Record(RecordValue::from_entries(
        pairs
            .iter()
            .map(|(n, v)| (tla_core::intern_name(n), v.clone()))
            .collect(),
    ))
}

#[test]
fn eq_record_new_matrix() {
    let token = record(&[
        ("pos", Value::SmallInt(2)),
        ("q", Value::SmallInt(0)),
        ("color", Value::String("white".into())),
    ]);
    // Equal.
    diff_record_new(
        token.clone(),
        &["pos", "q", "color"],
        &[
            Value::SmallInt(2),
            Value::SmallInt(0),
            Value::String("white".into()),
        ],
    );
    // Field value differs.
    diff_record_new(
        token.clone(),
        &["pos", "q", "color"],
        &[
            Value::SmallInt(1),
            Value::SmallInt(0),
            Value::String("white".into()),
        ],
    );
    // Field-name set differs.
    diff_record_new(
        token.clone(),
        &["pos", "q", "shade"],
        &[
            Value::SmallInt(2),
            Value::SmallInt(0),
            Value::String("white".into()),
        ],
    );
    // Field-count mismatch.
    diff_record_new(
        token.clone(),
        &["pos", "q"],
        &[Value::SmallInt(2), Value::SmallInt(0)],
    );
    // Non-record lhs (slow path decides; cross-representation semantics).
    diff_record_new(Value::SmallInt(3), &["pos"], &[Value::SmallInt(2)]);
    // Duplicate field names (slow path decides).
    diff_record_new(
        record(&[("a", Value::SmallInt(1))]),
        &["a", "a"],
        &[Value::SmallInt(1), Value::SmallInt(1)],
    );
    // Record-shaped Func lhs (cross-representation equality — slow path).
    let func_lhs = {
        let mut fb = tla_value::FuncBuilder::new();
        fb.insert(Value::String("pos".into()), Value::SmallInt(2));
        Value::Func(Rp::new(fb.build()))
    };
    diff_record_new(func_lhs, &["pos"], &[Value::SmallInt(2)]);
}
