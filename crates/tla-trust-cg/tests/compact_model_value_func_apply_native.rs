// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Native trust-codegen regression for compact model-value-keyed FuncApply.

#![cfg(feature = "native")]

use std::ffi::c_void;
use std::sync::Arc;

use tla_jit_abi::{CompoundLayout, JitCallOut, JitInvariantFn, JitStatus, StateLayout, VarLayout};
use tla_tir::bytecode::{BytecodeFunction, ConstantPool, Opcode};
use tla_value::Value;

const SYMBOL: &str = "compact_model_value_func_apply_native";

fn compact_proc_to_string() -> VarLayout {
    VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(CompoundLayout::String),
        value_layout: Box::new(CompoundLayout::String),
        pair_count: Some(4),
        domain_lo: None,
    })
}

fn layout() -> StateLayout {
    StateLayout::new(vec![compact_proc_to_string(), compact_proc_to_string()])
}

fn constants() -> (ConstantPool, u16, u16) {
    let mut pool = ConstantPool::new();
    let _procs = pool.add_value(Value::set([
        Value::ModelValue(Arc::from("p1")),
        Value::ModelValue(Arc::from("p2")),
        Value::ModelValue(Arc::from("p3")),
        Value::ModelValue(Arc::from("p4")),
    ]));
    let p2 = pool.add_value(Value::ModelValue(Arc::from("p2")));
    let li0 = pool.add_value(Value::String("Li0".into()));
    (pool, p2, li0)
}

fn invariant(p2: u16, li0: u16) -> BytecodeFunction {
    let mut func = BytecodeFunction::new(SYMBOL.to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadConst { rd: 1, idx: p2 });
    func.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    func.emit(Opcode::LoadVar { rd: 3, var_idx: 1 });
    func.emit(Opcode::FuncApply {
        rd: 4,
        func: 3,
        arg: 2,
    });
    func.emit(Opcode::LoadConst { rd: 5, idx: li0 });
    func.emit(Opcode::Eq {
        rd: 6,
        r1: 4,
        r2: 5,
    });
    func.emit(Opcode::Ret { rs: 6 });
    func
}

fn compile() -> (tla_trust_cg::NativeLibrary, JitInvariantFn) {
    let layout = layout();
    let (pool, p2, li0) = constants();
    let func = invariant(p2, li0);
    let lib = tla_trust_cg::compile_invariant_native_with_constants_and_layout(
        &func,
        SYMBOL,
        &pool,
        &layout,
        tla_trust_cg::OptLevel::O1,
    )
    .expect("compact model-value-keyed FuncApply should compile natively");
    let f = unsafe {
        let raw = lib
            .get_symbol(SYMBOL)
            .expect("compiled compact FuncApply symbol should be present");
        std::mem::transmute::<*mut c_void, JitInvariantFn>(raw)
    };
    (lib, f)
}

fn name(name: &str) -> i64 {
    i64::from(tla_core::intern_name(name).0)
}

fn eval(f: JitInvariantFn, state: &[i64]) -> JitCallOut {
    let mut out = JitCallOut::default();
    unsafe { f(&mut out, state.as_ptr(), state.len() as u32) };
    out
}

#[test]
fn native_model_value_keyed_compact_func_apply_uses_metadata_domain() {
    let (_lib, f) = compile();
    let state_true = [
        name("p1"),
        name("p3"),
        name("p3"),
        name("p4"),
        name("Li1"),
        name("Li1"),
        name("Li0"),
        name("Li1"),
    ];
    let out = eval(f, &state_true);
    assert_eq!(out.status, JitStatus::Ok, "native callout: {out:?}");
    assert_eq!(out.value, 1);

    let mut state_false = state_true;
    state_false[6] = name("Li1");
    let out = eval(f, &state_false);
    assert_eq!(out.status, JitStatus::Ok, "native callout: {out:?}");
    assert_eq!(out.value, 0);
}
