// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Part of #4462: nested bare-INSTANCE bytecode namespace coverage.
//!
//! A spec that reaches its operators through a chain of substitution-free
//! imports (root `INSTANCE Mid`, Mid `INSTANCE Leaf`, Leaf `EXTENDS`
//! sibling modules) must expose those operators to the bytecode compiler:
//! previously only ONE level of unnamed INSTANCE was searched, so
//! dag-consensus-shaped specs (TLCSailfish1: root -> Sailfish -> BlockDag ->
//! Utils/Digraph) fell back to the AST interpreter for every invariant and
//! action ("unresolved identifier 'Node'/'Round'/'Max'").

use tla_core::ast::Module;
use tla_core::{lower, parse_to_syntax_tree, FileId};
use tla_eval::tir::TirProgram;

fn parse_module(src: &str, file_id: u32) -> Module {
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(file_id), &tree);
    assert!(
        lower_result.errors.is_empty(),
        "lowering errors: {:?}",
        lower_result.errors
    );
    lower_result.module.expect("module should parse")
}

const LEAF_SRC: &str = r"---- MODULE Leaf ----
EXTENDS Integers
LeafOp(x) == x + x
LeafBuiltin(s) == Cardinality(s)
====";

const MID_SRC: &str = r"---- MODULE Mid ----
INSTANCE Leaf
MidOp(x) == LeafOp(x)
====";

const ROOT_SRC: &str = r"---- MODULE Root ----
INSTANCE Mid
RootOp == MidOp(2)
====";

#[test]
fn nested_bare_instance_operators_resolve_and_export() {
    let leaf = parse_module(LEAF_SRC, 1);
    let mid = parse_module(MID_SRC, 2);
    let root = parse_module(ROOT_SRC, 3);

    let program = TirProgram::from_modules(&root, &[&mid, &leaf]);

    // Level-1 (Mid) worked before; level-2 (Leaf) is the #4462 fix.
    assert!(
        program.can_lower_operator("MidOp"),
        "level-1 INSTANCE operator must resolve"
    );
    assert!(
        program.can_lower_operator("LeafOp"),
        "level-2 nested bare INSTANCE operator must resolve"
    );

    program.seed_bytecode_namespace_cache();
    let callees = program.export_callee_bodies();

    assert!(
        callees.contains_key("MidOp"),
        "level-1 op must be exported as a bytecode callee"
    );
    assert!(
        callees.contains_key("LeafOp"),
        "nested op must be exported as a bytecode callee (strict lowering)"
    );
    // `LeafBuiltin` calls the parameterized builtin `Cardinality`, which the
    // STRICT lowerer rejects (TIR-interpreter protection). The bytecode-only
    // permissive export must still surface it (mirrors compile.rs Phase 1.75).
    assert!(
        callees.contains_key("LeafBuiltin"),
        "builtin-calling nested op must be exported via the permissive retry"
    );
}

#[test]
fn nested_instance_with_substitutions_is_not_traversed() {
    // A nested INSTANCE that carries an explicit WITH substitution must NOT
    // be traversed (substitution composition across levels is unimplemented;
    // fail closed to the AST interpreter).
    let leaf = parse_module(
        r"---- MODULE Leaf2 ----
CONSTANT C
LeafOp2(x) == x
LeafUsesC == C
====",
        10,
    );
    let mid = parse_module(
        r"---- MODULE Mid2 ----
INSTANCE Leaf2 WITH C <- 42
MidOp2(x) == LeafOp2(x)
====",
        11,
    );
    let root = parse_module(
        r"---- MODULE Root2 ----
INSTANCE Mid2
RootOp2 == MidOp2(2)
====",
        12,
    );

    let program = TirProgram::from_modules(&root, &[&mid, &leaf]);
    assert!(program.can_lower_operator("MidOp2"));
    assert!(
        !program.can_lower_operator("LeafUsesC"),
        "operators behind a WITH-substituted nested INSTANCE must stay \
         unresolved (fail closed): same-name fallback would miscompute C"
    );
}
