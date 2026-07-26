// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Native trust-codegen regression for compact FuncApply over mixed scalar domains.

#![cfg(feature = "native")]

use std::ffi::c_void;

use tla_jit_abi::{
    CompoundLayout, JitCallOut, JitInvariantFn, JitStatus, SetBitmaskElement, StateLayout,
    VarLayout,
};
use tla_tir::bytecode::{BytecodeFunction, ConstantPool, Opcode};
use tla_value::{Rp, Value};

const SYMBOL: &str = "compact_mixed_domain_func_apply_native";

fn pc_layout() -> StateLayout {
    StateLayout::new(vec![VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(CompoundLayout::ExplicitScalarDomain {
            key_layout: Box::new(CompoundLayout::Dynamic),
            keys: vec![
                SetBitmaskElement::ModelValue(tla_core::intern_name("rm1")),
                SetBitmaskElement::ModelValue(tla_core::intern_name("rm2")),
                SetBitmaskElement::ModelValue(tla_core::intern_name("rm3")),
                SetBitmaskElement::Int(0),
                SetBitmaskElement::Int(10),
            ],
        }),
        value_layout: Box::new(CompoundLayout::String),
        pair_count: Some(5),
        domain_lo: None,
    })])
}

fn constants() -> (ConstantPool, u16, u16, u16) {
    let mut pool = ConstantPool::new();
    let rm2 = pool.add_value(Value::ModelValue(Rp::from("rm2")));
    let rs = pool.add_value(Value::String(Rp::from("RS")));
    let ts = pool.add_value(Value::String(Rp::from("TS")));
    (pool, rm2, rs, ts)
}

fn invariant(rm2: u16, rs: u16, ts: u16) -> BytecodeFunction {
    let mut func = BytecodeFunction::new(SYMBOL.to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadImm { rd: 1, value: 0 });
    func.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    func.emit(Opcode::LoadConst { rd: 3, idx: ts });
    func.emit(Opcode::Eq {
        rd: 4,
        r1: 2,
        r2: 3,
    });
    func.emit(Opcode::LoadConst { rd: 5, idx: rm2 });
    func.emit(Opcode::FuncApply {
        rd: 6,
        func: 0,
        arg: 5,
    });
    func.emit(Opcode::LoadConst { rd: 7, idx: rs });
    func.emit(Opcode::Eq {
        rd: 8,
        r1: 6,
        r2: 7,
    });
    func.emit(Opcode::And {
        rd: 9,
        r1: 4,
        r2: 8,
    });
    func.emit(Opcode::Ret { rs: 9 });
    func
}

fn compile() -> (tla_trust_cg::NativeLibrary, JitInvariantFn) {
    let layout = pc_layout();
    let (pool, rm2, rs, ts) = constants();
    let func = invariant(rm2, rs, ts);
    let lib = tla_trust_cg::compile_invariant_native_with_constants_and_layout(
        &func,
        SYMBOL,
        &pool,
        &layout,
        tla_trust_cg::OptLevel::O1,
    )
    .expect("compact mixed-domain FuncApply should compile natively");
    let f = unsafe {
        let raw = lib
            .get_symbol(SYMBOL)
            .expect("compiled compact mixed-domain FuncApply symbol should be present");
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
fn native_mixed_scalar_domain_func_apply_uses_explicit_domain_order() {
    let (_lib, f) = compile();
    let state_true = [name("RS"), name("RS"), name("RS"), name("TS"), name("BTS")];
    let out = eval(f, &state_true);
    assert_eq!(out.status, JitStatus::Ok, "native callout: {out:?}");
    assert_eq!(out.value, 1);

    let mut state_false = state_true;
    state_false[3] = name("TA");
    let out = eval(f, &state_false);
    assert_eq!(out.status, JitStatus::Ok, "native callout: {out:?}");
    assert_eq!(out.value, 0);
}
