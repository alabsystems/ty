// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness gate for native compilation of integer exponentiation (`^`).
//!
//! The TLA+ `^` operator lowers to the `PowInt` bytecode opcode. The bytecode
//! VM evaluates it in arbitrary precision (BigInt), while the trust-cg
//! direct-LLVM lowering computes it in i64 and traps (rather than diverging) on
//! a negative exponent or any result that does not fit in i64. For exponents
//! and bases small enough to fit i64 — the case exercised here — both backends
//! must compute the identical value. This test pins down two properties:
//!
//! 1. **Cross-backend agreement on a VIOLATED verdict** — the
//!    `PowInvariantViolated` fixture has the invariant `2 ^ x < 16`, which is
//!    genuinely violated once the reachable state `x = 4` is hit
//!    (`2 ^ 4 = 16`). Running the checker with the bytecode VM disabled
//!    (tree-walking interpreter) and enabled (native bytecode VM) must produce
//!    the SAME `InvariantViolation` verdict — no spurious overflow or
//!    divergence.
//!
//! 2. **Load-bearingness (A/B at the bytecode-compilation layer)** — an
//!    invariant using `^` compiles cleanly and emits a `PowInt` opcode, i.e.
//!    `^` really is the construct under test (the same opcode the trust-cg
//!    lowering now natively handles instead of falling back).

mod common;

use std::path::{Path, PathBuf};

use tla_check::{check_module, CheckResult, Config};
use tla_eval::bytecode_vm::compile_operators_to_bytecode_with_constants;
use tla_eval::clear_for_test_reset;
use tla_tir::bytecode::Opcode;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .to_path_buf()
}

fn read_fixture_spec() -> String {
    let path = repo_root()
        .join("test_specs")
        .join("PowInvariantViolated.tla");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

/// Build the model-check config for the fixture, mirroring its `.cfg`
/// (`SPECIFICATION Spec` / `INVARIANT PowBelow16`). We drive Init/Next directly
/// so the test does not depend on SPEC-resolution plumbing.
fn fixture_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["PowBelow16".to_string()],
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
                invariant, "PowBelow16",
                "{label}: violated invariant should be PowBelow16"
            );
        }
        other => panic!("{label}: expected InvariantViolation (VIOLATED), got {other:?}"),
    }
}

/// Core soundness property: the interpreter and the native bytecode VM must
/// agree that the `^`-based invariant is VIOLATED, with no spurious overflow.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn pow_invariant_violation_agrees_across_backends() {
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

/// Load-bearingness: an invariant that uses `^` compiles cleanly to bytecode
/// and emits a `PowInt` opcode, confirming `^` is genuinely the construct under
/// test (and the exact opcode the trust-cg direct-LLVM path now lowers).
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn pow_invariant_compiles_to_powint_opcode() {
    clear_for_test_reset();
    let module = common::parse_module(
        r#"
---- MODULE PowCompilesNative ----
EXTENDS Naturals
PowBelow16(x) == 2 ^ x < 16
====
"#,
    );

    let op_names = vec!["PowBelow16".to_string()];
    let resolved_constants = Default::default();
    let compiled =
        compile_operators_to_bytecode_with_constants(&module, &[], &op_names, &resolved_constants);

    assert!(
        compiled.failed.is_empty(),
        "`^`-using operator should compile cleanly (1/1, 0 failed); failed = {:?}",
        compiled.failed
    );
    assert!(
        compiled.op_indices.contains_key("PowBelow16"),
        "`^`-using operator should be present in the compiled op table"
    );

    let emits_pow = compiled.chunk.functions.iter().any(|func| {
        func.instructions
            .iter()
            .any(|op| matches!(op, Opcode::PowInt { .. }))
    });
    assert!(
        emits_pow,
        "compiled bytecode should emit a PowInt opcode for `^`"
    );
}
