// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Native trust-codegen regression for compact model-value-keyed FuncExcept.

#![cfg(feature = "native")]

use std::ffi::c_void;

use tla_jit_abi::{CompoundLayout, JitCallOut, JitNextStateFn, JitStatus, StateLayout, VarLayout};
use tla_tir::bytecode::{BytecodeFunction, ConstantPool, Opcode};
use tla_value::{Rp, Value};

const SYMBOL: &str = "compact_model_value_func_except_native";

fn compact_proc_to_string() -> VarLayout {
    VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(CompoundLayout::String),
        value_layout: Box::new(CompoundLayout::String),
        pair_count: Some(4),
        domain_lo: None,
    })
}

fn layout() -> StateLayout {
    StateLayout::new(vec![compact_proc_to_string()])
}

fn constants() -> (ConstantPool, u16, u16) {
    let mut pool = ConstantPool::new();
    let _procs = pool.add_value(Value::set([
        Value::ModelValue(Rp::from("p1")),
        Value::ModelValue(Rp::from("p2")),
        Value::ModelValue(Rp::from("p3")),
        Value::ModelValue(Rp::from("p4")),
    ]));
    let p2 = pool.add_value(Value::ModelValue(Rp::from("p2")));
    let li0 = pool.add_value(Value::String("Li0".into()));
    (pool, p2, li0)
}

fn action(p2: u16, li0: u16) -> BytecodeFunction {
    let mut func = BytecodeFunction::new(SYMBOL.to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadConst { rd: 1, idx: p2 });
    func.emit(Opcode::LoadConst { rd: 2, idx: li0 });
    func.emit(Opcode::FuncExcept {
        rd: 3,
        func: 0,
        path: 1,
        val: 2,
    });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 3 });
    func.emit(Opcode::LoadBool { rd: 4, value: true });
    func.emit(Opcode::Ret { rs: 4 });
    func
}

fn compile() -> (tla_trust_cg::NativeLibrary, JitNextStateFn) {
    let layout = layout();
    let (pool, p2, li0) = constants();
    let func = action(p2, li0);
    let lib = tla_trust_cg::compile_next_state_native_with_constants_and_layout(
        &func,
        SYMBOL,
        &pool,
        &layout,
        tla_trust_cg::OptLevel::O1,
    )
    .expect("compact model-value-keyed FuncExcept should compile natively");
    let f = unsafe {
        let raw = lib
            .get_symbol(SYMBOL)
            .expect("compiled compact FuncExcept symbol should be present");
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
fn native_model_value_keyed_compact_func_except_updates_slot() {
    let (_lib, f) = compile();
    let state = [name("Li1"), name("Li1"), name("Li1"), name("Li1")];
    let (out, state_out) = eval(f, &state);
    assert_eq!(out.status, JitStatus::Ok, "native callout: {out:?}");
    assert_eq!(out.value, 1);
    assert_eq!(
        state_out,
        [name("Li1"), name("Li0"), name("Li1"), name("Li1")]
    );
}
