// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness gate for native lowering of power-set quantifiers:
//!   `\E S \in SUBSET T : P(S)`   and   `\A S \in SUBSET T : P(S)`.
//!
//! The trust-ir/trust-cg backend lowers a quantifier whose domain is `SUBSET T`
//! (an exact, statically-known scalar/int set `T`) by enumerating the
//! `2^|T|` submasks of `T`'s element universe — a counter from `0..2^|T|`
//! where bit `i` decides whether element `i` of `T` is in the candidate
//! subset `S`. When `T` is too large (or not a materializable exact scalar
//! set), the lowering *declines* and the checker falls back to the
//! tree-walking interpreter (fail-closed).
//!
//! For `\E`/`\A` the verdict is independent of the subset visitation order,
//! so the native submask order (counter order) and the interpreter's
//! TLC-normalized cardinality-first order must agree on the boolean result.
//!
//! This test pins down three properties:
//!
//! 1. **Cross-backend agreement on a VIOLATED verdict + identical state
//!    counts** — the `SubsetQuantifierViolated` fixture has an invariant
//!    `\A S \in SUBSET T : x \notin S` that is genuinely violated in a
//!    reachable state (`x = 1`, witnessed by the subset `{1}`). Running the
//!    checker with the bytecode VM disabled (interpreter) and enabled
//!    (native bytecode VM) must produce the SAME `InvariantViolation`
//!    verdict and the SAME number of distinct states.
//!
//! 2. **Load-bearingness** — the invariant compiles to bytecode that emits a
//!    `Powerset` opcode feeding a `ForallBegin` quantifier domain, i.e. it
//!    exercises the SUBSET-quantifier path rather than being folded away.
//!
//! 3. **Fail-closed bound** — an oversized `T` (here `1..40`, far beyond the
//!    63-element compact-bitmask universe limit) still produces a correct,
//!    panic-free verdict on every backend. The native submask lowering must
//!    decline this domain and fall back to the interpreter.

mod common;

use std::collections::HashMap;
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
        .join("SubsetQuantifierViolated.tla");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

/// Build the model-check config for the fixture, mirroring its `.cfg`
/// (`SPECIFICATION Spec` / `INVARIANT NoMemberCovered`). We drive Init/Next
/// directly so the test does not depend on SPEC-resolution plumbing.
fn fixture_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["NoMemberCovered".to_string()],
        constants: HashMap::new(),
        constants_order: Vec::new(),
        check_deadlock: false,
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
enum BytecodeMode {
    /// Tree-walking interpreter (bytecode VM disabled).
    Interpreter,
    /// Native bytecode VM enabled (executes the same SUBSET-quantifier
    /// bytecode that the trust-cg backend lowers to native submask iteration).
    NativeVm,
}

fn run_check(spec: &str, config: &Config, mode: BytecodeMode) -> CheckResult {
    let _guard = match mode {
        BytecodeMode::Interpreter => common::EnvVarGuard::set("TY_BYTECODE_VM", Some("0")),
        BytecodeMode::NativeVm => common::EnvVarGuard::set("TY_BYTECODE_VM", Some("1")),
    };
    clear_for_test_reset();
    let module = common::parse_module(spec);
    check_module(&module, config)
}

fn assert_violated(label: &str, result: &CheckResult) {
    match result {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(
                invariant, "NoMemberCovered",
                "{label}: violated invariant should be NoMemberCovered"
            );
        }
        other => panic!("{label}: expected InvariantViolation (VIOLATED), got {other:?}"),
    }
}

/// Core soundness property: the interpreter and the native bytecode VM must
/// agree that the power-set-quantifier invariant is VIOLATED, and must
/// explore the same number of distinct states.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn subset_quantifier_violation_agrees_across_backends() {
    let spec = read_fixture_spec();
    let config = fixture_config();

    let interpreter = run_check(&spec, &config, BytecodeMode::Interpreter);
    assert_violated("interpreter", &interpreter);

    let native = run_check(&spec, &config, BytecodeMode::NativeVm);
    assert_violated("native-vm", &native);

    assert_eq!(
        interpreter.stats().states_found,
        native.stats().states_found,
        "interpreter and native bytecode VM must explore the same number of distinct states \
         (interpreter={}, native={})",
        interpreter.stats().states_found,
        native.stats().states_found,
    );
}

/// Load-bearingness: the SUBSET-quantifier invariant compiles to bytecode
/// that emits a `Powerset` opcode whose result is consumed as a `ForallBegin`
/// quantifier domain — i.e. it actually exercises the power-set quantifier
/// path rather than being constant-folded or rewritten away.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn subset_quantifier_compiles_to_powerset_forall_bytecode() {
    clear_for_test_reset();
    let module = common::parse_module(
        r#"
---- MODULE SubsetForallCompiles ----
EXTENDS Naturals
T == {1, 2}
NoMemberCovered(x) == \A S \in SUBSET T : x \notin S
====
"#,
    );

    let op_names = vec!["NoMemberCovered".to_string()];
    let resolved_constants = Default::default();
    let compiled =
        compile_operators_to_bytecode_with_constants(&module, &[], &op_names, &resolved_constants);

    assert!(
        compiled.failed.is_empty(),
        "SUBSET-quantifier operator should compile cleanly; failed = {:?}",
        compiled.failed
    );

    let emits_powerset = compiled.chunk.functions.iter().any(|func| {
        func.instructions
            .iter()
            .any(|op| matches!(op, Opcode::Powerset { .. }))
    });
    assert!(
        emits_powerset,
        "compiled bytecode should emit a Powerset opcode for the SUBSET T domain"
    );

    let emits_forall = compiled.chunk.functions.iter().any(|func| {
        func.instructions
            .iter()
            .any(|op| matches!(op, Opcode::ForallBegin { .. }))
    });
    assert!(
        emits_forall,
        "compiled bytecode should emit a ForallBegin over the SUBSET T domain"
    );

    // The Powerset result register must feed the ForallBegin domain — proving
    // the quantifier really ranges over the power set (not some other set).
    let powerset_feeds_forall = compiled.chunk.functions.iter().any(|func| {
        let powerset_dests: Vec<u8> = func
            .instructions
            .iter()
            .filter_map(|op| match op {
                Opcode::Powerset { rd, .. } => Some(*rd),
                _ => None,
            })
            .collect();
        func.instructions.iter().any(|op| match op {
            Opcode::ForallBegin { r_domain, .. } => powerset_dests.contains(r_domain),
            _ => false,
        })
    });
    assert!(
        powerset_feeds_forall,
        "the ForallBegin domain register should be produced by a Powerset opcode"
    );
}

/// Fail-closed bound: an oversized `T` (`1..40`, far beyond the 63-element
/// compact-bitmask universe limit, and 2^40 submasks) must still yield a
/// correct, panic-free verdict on every backend. The native submask lowering
/// *declines* this domain (its `2^40` element universe cannot be represented
/// as an i64 submask universe) and the checker falls back to the lazy
/// interpreter rather than emitting an unbounded/incorrect loop.
///
/// To keep the test fast we use an *existential* witness that the
/// cardinality-first interpreter finds early without enumerating `2^40`
/// subsets: `\E S \in SUBSET (1..40) : 1 \in S` is satisfied by the subset
/// `{1}`, which the interpreter visits as the second element (right after the
/// empty set). The invariant `NoBigMemberCovered == ~(\E S \in SUBSET BigT :
/// 1 \in S)` is therefore `~TRUE = FALSE`, i.e. genuinely VIOLATED in the
/// initial state. (An always-TRUE `\A`/always-FALSE `\E` over `SUBSET (1..40)`
/// would instead force full `2^40` enumeration — exactly the blow-up the
/// fail-closed bound exists to avoid emitting in native code.)
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn subset_quantifier_oversized_domain_falls_back_gracefully() {
    let spec = r#"
---- MODULE SubsetQuantifierOversized ----
EXTENDS Naturals
VARIABLE x
BigT == 1..40
Init == x = 0
Next == \/ x' = 0
        \/ x' = 1
NoBigMemberCovered == ~(\E S \in SUBSET BigT : 1 \in S)
====
"#;
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["NoBigMemberCovered".to_string()],
        constants: HashMap::new(),
        constants_order: Vec::new(),
        check_deadlock: false,
        ..Default::default()
    };

    let interpreter = run_check(spec, &config, BytecodeMode::Interpreter);
    let native = run_check(spec, &config, BytecodeMode::NativeVm);

    // Both backends must agree the invariant is violated by NoBigMemberCovered
    // (no panic, no hang on 2^40 submask enumeration), with identical state
    // counts.
    match (&interpreter, &native) {
        (
            CheckResult::InvariantViolation {
                invariant: i_inv, ..
            },
            CheckResult::InvariantViolation {
                invariant: n_inv, ..
            },
        ) => {
            assert_eq!(
                i_inv, "NoBigMemberCovered",
                "interpreter violated invariant"
            );
            assert_eq!(n_inv, "NoBigMemberCovered", "native-vm violated invariant");
        }
        other => panic!(
            "expected both backends to report InvariantViolation on the oversized SUBSET domain, \
             got {other:?}"
        ),
    }

    assert_eq!(
        interpreter.stats().states_found,
        native.stats().states_found,
        "fail-closed oversized SUBSET domain must still agree on distinct state count \
         (interpreter={}, native={})",
        interpreter.stats().states_found,
        native.stats().states_found,
    );
}
