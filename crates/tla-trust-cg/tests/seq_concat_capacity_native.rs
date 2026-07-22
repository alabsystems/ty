// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Native trust-codegen regression for the compact fixed-capacity Concat
//! with a capacity-NARROWING store-back: the self-grow write
//! `v' = <<x>> \o v` produces a Capacity(C+1) result that must store into
//! the Capacity(C) var under a runtime `len <= C` guard. In-bounds runs are
//! interpreter-exact; an over-capacity length is the typed runtime error
//! (per-state interpreter fallback) — never a truncated successor, never a
//! crash.

#![cfg(feature = "native")]

use std::ffi::c_void;

use tla_jit_abi::{
    CompoundLayout, JitCallOut, JitNextStateFn, JitRuntimeErrorKind, JitStatus, StateLayout,
    VarLayout,
};
use tla_tir::bytecode::{BytecodeFunction, ConstantPool, Opcode};

const SYMBOL: &str = "seq_concat_capacity_native";
const CAPACITY: i64 = 3;
const PREPENDED: i64 = 99;

fn layout() -> StateLayout {
    StateLayout::new(vec![VarLayout::Compound(CompoundLayout::Sequence {
        capacity_proven: true,
        element_layout: Box::new(CompoundLayout::Int),
        element_count: Some(CAPACITY as usize),
    })])
}

/// `v' = <<PREPENDED>> \o v` over a capacity-3 integer sequence var.
fn action() -> BytecodeFunction {
    let mut func = BytecodeFunction::new(SYMBOL.to_string(), 0);
    func.emit(Opcode::LoadImm {
        rd: 0,
        value: PREPENDED,
    });
    func.emit(Opcode::SeqNew {
        rd: 1,
        start: 0,
        count: 1,
    });
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 });
    func.emit(Opcode::Concat {
        rd: 3,
        r1: 1,
        r2: 2,
    });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 3 });
    func.emit(Opcode::LoadBool { rd: 4, value: true });
    func.emit(Opcode::Ret { rs: 4 });
    func
}

fn compile() -> (tla_trust_cg::NativeLibrary, JitNextStateFn) {
    let lib = tla_trust_cg::compile_next_state_native_with_constants_and_layout(
        &action(),
        SYMBOL,
        &ConstantPool::new(),
        &layout(),
        tla_trust_cg::OptLevel::O1,
    )
    .expect("self-grow compact Concat with narrowing store-back should compile natively");
    let f = unsafe {
        let raw = lib
            .get_symbol(SYMBOL)
            .expect("compiled self-grow Concat symbol should be present");
        std::mem::transmute::<*mut c_void, JitNextStateFn>(raw)
    };
    (lib, f)
}

fn eval(f: JitNextStateFn, state: &[i64]) -> (JitCallOut, Vec<i64>) {
    let mut out = JitCallOut::default();
    let mut state_out = state.to_vec();
    unsafe {
        f(
            &mut out,
            state.as_ptr(),
            state_out.as_mut_ptr(),
            state.len() as u32,
        )
    };
    (out, state_out)
}

/// Interpreter semantics of `<<PREPENDED>> \o v` on the flat
/// `[len, e1, e2, e3]` layout, for `Len(v) + 1 <= CAPACITY`.
fn interpreter_successor(state: &[i64]) -> Vec<i64> {
    let len = usize::try_from(state[0]).expect("test states carry valid lengths");
    let mut elements = vec![PREPENDED];
    elements.extend_from_slice(&state[1..=len]);
    let mut successor = vec![elements.len() as i64];
    elements.resize(CAPACITY as usize, 0);
    successor.extend_from_slice(&elements);
    successor
}

#[test]
fn native_self_grow_concat_matches_interpreter_within_capacity() {
    let (_lib, f) = compile();
    for state in [vec![0, 0, 0, 0], vec![1, 10, 0, 0], vec![2, 10, 20, 0]] {
        let (out, state_out) = eval(f, &state);
        assert_eq!(out.status, JitStatus::Ok, "state {state:?}: {out:?}");
        assert_eq!(out.value, 1, "state {state:?}: action guard should hold");
        assert_eq!(
            state_out,
            interpreter_successor(&state),
            "state {state:?}: native successor must be interpreter-exact"
        );
    }
}

#[test]
fn native_self_grow_concat_over_capacity_is_runtime_error_not_truncation() {
    let (_lib, f) = compile();
    let state = [3, 10, 20, 30];
    let (out, state_out) = eval(f, &state);
    assert_eq!(
        out.status,
        JitStatus::RuntimeError,
        "Len(v) + 1 > capacity must surface the runtime-error status: {out:?}"
    );
    assert_eq!(
        out.err_kind,
        JitRuntimeErrorKind::TypeMismatch,
        "narrowing guard failure must be the typed TypeMismatch error: {out:?}"
    );
    assert_eq!(
        state_out, state,
        "the error path must not write a truncated successor into state_out"
    );
}
