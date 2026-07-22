// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness gate for native compilation of the `Range` operator.
//!
//! `Range(f) == { f[x] : x \in DOMAIN f }` (Functions / SequencesExt). Before
//! native support, invariants referencing `Range` failed bytecode compilation
//! with `unresolved identifier 'Range'` and silently fell back to the
//! tree-walking interpreter. This test pins down two properties:
//!
//! 1. **Cross-backend agreement on a VIOLATED verdict** — the
//!    `RangeInvariantViolated` fixture has an invariant `0 \notin Range(f)`
//!    that is genuinely violated in a reachable state. Running the checker with
//!    the bytecode VM disabled (interpreter) and enabled (native bytecode VM)
//!    must produce the SAME `InvariantViolation` verdict.
//!
//! 2. **Load-bearingness (A/B at the bytecode-compilation layer)** — with the
//!    `"Range" => (BuiltinOp::Range, 1)` mapping present, the invariant
//!    compiles cleanly (`0` failed) and emits a `CallBuiltin(Range)` opcode,
//!    i.e. it takes the native path rather than falling back. A negative
//!    control shows what the pre-mapping failure looked like: an unknown
//!    operator name still lands in `failed` with an "unresolved identifier"
//!    diagnostic — the exact symptom `Range` used to exhibit.

mod common;

use std::path::{Path, PathBuf};

use tla_check::{check_module, CheckResult, Config};
use tla_eval::bytecode_vm::compile_operators_to_bytecode_with_constants;
use tla_eval::clear_for_test_reset;
use tla_tir::bytecode::{BuiltinOp, Opcode};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .to_path_buf()
}

fn read_fixture_spec() -> String {
    let path = repo_root()
        .join("test_specs")
        .join("RangeInvariantViolated.tla");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

/// Build the model-check config for the fixture, mirroring its `.cfg`
/// (`SPECIFICATION Spec` / `INVARIANT RangeExcludesZero`). We drive Init/Next
/// directly so the test does not depend on SPEC-resolution plumbing.
fn fixture_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["RangeExcludesZero".to_string()],
        constants: std::collections::HashMap::new(),
        constants_order: Vec::new(),
        check_deadlock: false,
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
enum BytecodeMode {
    /// Tree-walking interpreter (bytecode VM disabled).
    Interpreter,
    /// Native bytecode VM enabled.
    NativeVm,
}

fn run_check(mode: BytecodeMode) -> CheckResult {
    let _guard = match mode {
        BytecodeMode::Interpreter => common::EnvVarGuard::set("TY_BYTECODE_VM", Some("0")),
        BytecodeMode::NativeVm => common::EnvVarGuard::set("TY_BYTECODE_VM", Some("1")),
    };
    clear_for_test_reset();
    let module = common::parse_module(&read_fixture_spec());
    check_module(&module, &fixture_config())
}

fn assert_violated(label: &str, result: &CheckResult) {
    match result {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(
                invariant, "RangeExcludesZero",
                "{label}: violated invariant should be RangeExcludesZero"
            );
        }
        other => panic!("{label}: expected InvariantViolation (VIOLATED), got {other:?}"),
    }
}

/// Core soundness property: the interpreter and the native bytecode VM must
/// agree that the `Range`-based invariant is VIOLATED.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn range_invariant_violation_agrees_across_backends() {
    let interpreter = run_check(BytecodeMode::Interpreter);
    assert_violated("interpreter", &interpreter);

    let native = run_check(BytecodeMode::NativeVm);
    assert_violated("native-vm", &native);

    // Both backends reached the same verdict on the same invariant.
    assert!(
        matches!(interpreter, CheckResult::InvariantViolation { .. })
            && matches!(native, CheckResult::InvariantViolation { .. }),
        "both backends must report InvariantViolation"
    );
}

/// Load-bearingness: with the `Range` builtin mapping in place, an invariant
/// that uses `Range` compiles to bytecode (0 failed) and emits a
/// `CallBuiltin(Range)` opcode, proving it takes the native path.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn range_invariant_compiles_to_native_callbuiltin() {
    clear_for_test_reset();
    let module = common::parse_module(
        r#"
---- MODULE RangeCompilesNative ----
EXTENDS Naturals, Functions
RangeExcludesZero(f) == 0 \notin Range(f)
====
"#,
    );

    let op_names = vec!["RangeExcludesZero".to_string()];
    let resolved_constants = Default::default();
    let compiled =
        compile_operators_to_bytecode_with_constants(&module, &[], &op_names, &resolved_constants);

    assert!(
        compiled.failed.is_empty(),
        "Range-using operator should compile cleanly (1/1, 0 failed); failed = {:?}",
        compiled.failed
    );
    assert!(
        compiled.op_indices.contains_key("RangeExcludesZero"),
        "Range-using operator should be present in the compiled op table"
    );

    let emits_range_builtin = compiled.chunk.functions.iter().any(|func| {
        func.instructions.iter().any(|op| {
            matches!(
                op,
                Opcode::CallBuiltin {
                    builtin: BuiltinOp::Range,
                    ..
                }
            )
        })
    });
    assert!(
        emits_range_builtin,
        "compiled bytecode should emit CallBuiltin(BuiltinOp::Range) — i.e. take the native path, not fall back"
    );
}

/// Negative control documenting the pre-mapping symptom: an operator that
/// references a genuinely unknown identifier lands in `failed` with an
/// "unresolved identifier" diagnostic. This is exactly the failure mode that
/// `Range` exhibited before the `"Range" => (BuiltinOp::Range, 1)` mapping was
/// added, and confirms the mapping (not some unrelated path) is load-bearing.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn unresolved_operator_fails_compilation_with_unresolved_identifier() {
    clear_for_test_reset();
    let module = common::parse_module(
        r#"
---- MODULE UnresolvedControl ----
EXTENDS Naturals
UsesUnknown(f) == 0 \notin NotARealOperator(f)
====
"#,
    );

    let op_names = vec!["UsesUnknown".to_string()];
    let resolved_constants = Default::default();
    let compiled =
        compile_operators_to_bytecode_with_constants(&module, &[], &op_names, &resolved_constants);

    let failed_msg = compiled
        .failed
        .iter()
        .find(|(name, _)| name == "UsesUnknown")
        .map(|(_, err)| format!("{err}"))
        .unwrap_or_else(|| {
            panic!(
                "operator referencing an unknown identifier should fail to compile; failed = {:?}",
                compiled.failed
            )
        });
    assert!(
        failed_msg.contains("unresolved identifier"),
        "expected an 'unresolved identifier' diagnostic (the pre-mapping Range symptom), got: {failed_msg}"
    );
}
