// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Native trust-codegen regression for model-value scalar StoreVar provenance.

#![cfg(feature = "native")]

use std::ffi::c_void;
use std::sync::Arc;

use tla_jit_abi::{CompoundLayout, JitCallOut, JitNextStateFn, JitStatus, StateLayout, VarLayout};
use tla_tir::bytecode::{BytecodeFunction, ConstantPool, Opcode};
use tla_value::Value;

const SYMBOL: &str = "scalar_string_store_var_native";

fn layout() -> StateLayout {
    StateLayout::new(vec![VarLayout::Compound(CompoundLayout::String)])
}

fn action_load_imm(value: i64) -> BytecodeFunction {
    let mut func = BytecodeFunction::new(SYMBOL.to_string(), 0);
    func.emit(Opcode::LoadImm { rd: 0, value });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    func.emit(Opcode::LoadBool { rd: 1, value: true });
    func.emit(Opcode::Ret { rs: 1 });
    func
}

fn action_load_const(idx: u16) -> BytecodeFunction {
    let mut func = BytecodeFunction::new(SYMBOL.to_string(), 0);
    func.emit(Opcode::LoadConst { rd: 0, idx });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    func.emit(Opcode::LoadBool { rd: 1, value: true });
    func.emit(Opcode::Ret { rs: 1 });
    func
}

fn compile_result(
    func: &BytecodeFunction,
    pool: &ConstantPool,
) -> Result<tla_trust_cg::NativeLibrary, tla_trust_cg::TrustCgError> {
    let layout = layout();
    tla_trust_cg::compile_next_state_native_with_constants_and_layout(
        func,
        SYMBOL,
        pool,
        &layout,
        tla_trust_cg::OptLevel::O1,
    )
}

fn compile(
    func: &BytecodeFunction,
    pool: &ConstantPool,
) -> (tla_trust_cg::NativeLibrary, JitNextStateFn) {
    let lib = compile_result(func, pool)
        .expect("proven model-value scalar StoreVar should compile natively");
    let f = unsafe {
        let raw = lib
            .get_symbol(SYMBOL)
            .expect("compiled scalar StoreVar symbol should be present");
        std::mem::transmute::<*mut c_void, JitNextStateFn>(raw)
    };
    (lib, f)
}

fn name(name: &str) -> i64 {
    i64::from(tla_core::intern_name(name).0)
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

#[test]
fn native_store_var_string_scalar_accepts_model_value_const_provenance() {
    let mut pool = ConstantPool::new();
    let p2_idx = pool.add_value(Value::ModelValue(Arc::from("p2")));
    let func = action_load_const(p2_idx);
    let (_lib, f) = compile(&func, &pool);
    let (out, state_out) = eval(f, &[name("p1")]);
    assert_eq!(out.status, JitStatus::Ok, "native callout: {out:?}");
    assert_eq!(out.value, 1);
    assert_eq!(state_out, [name("p2")]);
}

#[test]
fn native_store_var_string_scalar_rejects_bare_name_id_imm_without_provenance() {
    let pool = ConstantPool::new();
    let func = action_load_imm(name("p2"));
    let err = compile_result(&func, &pool)
        .expect_err("bare LoadImm NameId must not compile as a string/model-value store");
    let message = err.to_string();
    assert!(
        message.contains("requires compatible scalar source"),
        "unexpected bare NameId StoreVar rejection: {message}"
    );
}
