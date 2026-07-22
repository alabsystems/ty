// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! F1 (lever L2) fail-open behavior inside `tla-tir` itself.
//!
//! This crate cannot execute the fold: the executor is injected by
//! `tla-eval`, and the `tla-tir` test binary never installs it. A perfectly
//! foldable constant subtree must therefore compile to the normal
//! per-state constructor opcodes — folding silently never fires.
//! (The positive fold tests, differential matrix, and error-fidelity tests
//! live in `tla-eval`'s `bytecode_vm/tests/const_fold.rs`, next to the VM.)

use super::*;

fn resolved_constants_with_proc() -> std::collections::HashMap<tla_core::NameId, Value> {
    let mut constants = std::collections::HashMap::new();
    constants.insert(
        intern_name("Proc"),
        Value::set([
            Value::ModelValue("p1".into()),
            Value::ModelValue("p2".into()),
            Value::ModelValue("p3".into()),
        ]),
    );
    constants
}

fn resolved_name(name: &str) -> Spanned<TirExpr> {
    spanned(TirExpr::Name(TirNameRef {
        name: name.to_string(),
        name_id: intern_name(name),
        kind: TirNameKind::Ident,
        ty: TirType::Dyn,
    }))
}

#[test]
fn fold_never_fires_without_installed_executor() {
    let expr = spanned(TirExpr::SetBinOp {
        left: Box::new(spanned(TirExpr::Powerset(Box::new(resolved_name("Proc"))))),
        op: TirSetOp::Union,
        right: Box::new(resolved_name("Proc")),
    });

    let mut compiler = BytecodeCompiler::with_resolved_constants(resolved_constants_with_proc());
    let idx = compiler
        .compile_expression("NoExecutor", &expr)
        .expect("compiles");
    let chunk = compiler.finish();
    let instructions = &chunk.get_function(idx).instructions;

    // Fail-open: without an executor the constructor opcodes must survive.
    assert!(
        instructions
            .iter()
            .any(|op| matches!(op, Opcode::SetUnion { .. })),
        "expected the normal SetUnion path, got {instructions:?}"
    );
    assert!(
        instructions
            .iter()
            .any(|op| matches!(op, Opcode::Powerset { .. })),
        "expected the normal Powerset path, got {instructions:?}"
    );
}
