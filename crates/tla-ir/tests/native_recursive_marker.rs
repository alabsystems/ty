// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use tla_ir::lower::{lower_module_invariant, LoweringOptions};
use tla_ir::trust_ir::inst::Inst;
use tla_ir::trust_ir::ty::Ty;
use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};
use tla_value::Value;

fn recursive_countdown_chunk(self_recursive: bool) -> (BytecodeChunk, u16) {
    let mut chunk = BytecodeChunk::new();
    let name_idx = chunk.constants.add_value(Value::String("Rec".into()));

    // Rec(n) == IF n = 0 THEN n ELSE Rec(n - 1)
    let mut rec = BytecodeFunction::new("Rec".to_string(), 1);
    rec.emit(Opcode::LoadImm { rd: 1, value: 0 });
    rec.emit(Opcode::Eq {
        rd: 2,
        r1: 0,
        r2: 1,
    });
    let recurse_jump = rec.emit(Opcode::JumpFalse { rs: 2, offset: 0 });
    rec.emit(Opcode::Ret { rs: 0 });
    let recurse_pc = rec.emit(Opcode::LoadImm { rd: 3, value: 1 });
    rec.emit(Opcode::SubInt {
        rd: 4,
        r1: 0,
        r2: 3,
    });
    rec.emit(Opcode::CallExternal {
        rd: 5,
        name_idx,
        args_start: 4,
        argc: 1,
        self_recursive,
    });
    rec.emit(Opcode::Ret { rs: 5 });
    rec.patch_jump(recurse_jump, recurse_pc);
    let rec_idx = chunk.add_function(rec);

    let mut entry = BytecodeFunction::new("Entry".to_string(), 0);
    entry.emit(Opcode::LoadImm { rd: 0, value: 3 });
    entry.emit(Opcode::Call {
        rd: 1,
        op_idx: rec_idx,
        args_start: 0,
        argc: 1,
    });
    entry.emit(Opcode::Ret { rs: 1 });
    let entry_idx = chunk.add_function(entry);

    (chunk, entry_idx)
}

#[test]
fn native_recursive_call_external_requires_authenticated_marker_end_to_end() {
    let (unmarked, entry_idx) = recursive_countdown_chunk(false);
    let err = lower_module_invariant(
        &unmarked,
        entry_idx,
        "unmarked_recursive",
        LoweringOptions::new(),
    )
    .expect_err("same-name/same-arity CallExternal without provenance must fail closed");
    assert!(
        err.to_string()
            .contains("only a strict self-recursive call"),
        "unexpected rejection: {err}"
    );

    let (marked, entry_idx) = recursive_countdown_chunk(true);
    let module = lower_module_invariant(
        &marked,
        entry_idx,
        "marked_recursive",
        LoweringOptions::new(),
    )
    .expect("authenticated strict self-recursion should lower");

    assert_eq!(module.functions.len(), 2);
    let helper = &module.functions[1];
    let helper_ty = &module.func_types[helper.ty.as_usize()];
    assert_eq!(helper_ty.params.len(), 6);
    assert_eq!(helper_ty.params.last(), Some(&Ty::I64));
    let helper_calls: Vec<_> = helper
        .blocks
        .iter()
        .flat_map(|block| &block.body)
        .filter_map(|node| match node.inst {
            Inst::Call { callee, .. } => Some(callee),
            _ => None,
        })
        .collect();
    assert_eq!(helper_calls, vec![helper.id]);
}
