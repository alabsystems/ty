// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-backend soundness for the native `SubSeq` trust-ir lowering.
//!
//! `SubSeqNativeFold.tla` drives a genuinely-violated invariant (`Len(s) > 1`)
//! by repeatedly applying `SubSeq(s, 2, Len(s))` (a length-changing sequence
//! op). The action also mutates a record field, which keeps it off the
//! all-scalar direct fast path and forces the compiled backend through the
//! trust-ir pipeline that now natively lowers `SubSeq`.
//!
//! Two guarantees:
//! - default-feature: the interpreter and the compiled-BFS backend agree on the
//!   same invariant violation.
//! - `trust-cg` feature: the in-process trust-codegen native backend reaches the
//!   same violation AND compiles every action instance (`compiled == total`),
//!   proving the native SubSeq arm actually fires (no VM fallback).

mod common;

use std::path::{Path, PathBuf};
use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

use tla_check::ModelChecker;

const SPEC_FILE: &str = "SubSeqNativeFold.tla";
const CFG_FILE: &str = "SubSeqNativeFold.cfg";
const VIOLATED_INVARIANT: &str = "InvLenGtOne";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .to_path_buf()
}

fn read_spec_source() -> String {
    let path = repo_root().join("test_specs").join(SPEC_FILE);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn read_cfg_source() -> String {
    let path = repo_root().join("test_specs").join(CFG_FILE);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

/// Assert the result is the expected reachable `InvLenGtOne` violation.
fn assert_len_violation(label: &str, result: CheckResult) {
    match result {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(
                invariant, VIOLATED_INVARIANT,
                "{label}: expected the {VIOLATED_INVARIANT} invariant to be violated"
            );
        }
        other => panic!("{label}: expected an InvariantViolation, got {other:?}"),
    }
}

// ============================================================================
// Default-feature: interpreter and compiled-BFS agree on the violation.
// ============================================================================

/// Interpreter baseline: SubSeq shrinking `s` to length 1 violates `Len(s) > 1`.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn subseq_native_fold_interpreter_reports_violation() {
    let _no_compiled = common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1"));
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    clear_for_test_reset();

    let module = common::parse_module(&read_spec_source());
    let config = Config::parse(&read_cfg_source()).expect("valid cfg");
    assert_len_violation("interpreter baseline", check_module(&module, &config));
}

/// The compiled-BFS backend (which exercises the trust-ir SubSeq lowering) must
/// agree with the interpreter on the same invariant violation.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn subseq_native_fold_compiled_bfs_agrees_on_violation() {
    let _compiled = common::EnvVarGuard::set("TY_COMPILED_BFS", Some("1"));
    let _no_flat = common::EnvVarGuard::remove("TY_NO_FLAT_BFS");
    clear_for_test_reset();

    let module = common::parse_module(&read_spec_source());
    let mut config = Config::parse(&read_cfg_source()).expect("valid cfg");
    config.use_compiled_bfs = Some(true);
    assert_len_violation("compiled-BFS run", check_module(&module, &config));
}

// ============================================================================
// trust-cg feature: native backend reaches the violation with full coverage.
// ============================================================================

/// Drive the trust-codegen native backend: it must reach the same violation
/// and natively compile every action instance, proving the native SubSeq arm
/// fired (no interpreter fallback for the SubSeq action).
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn subseq_native_fold_trust_cg_violation_with_full_action_coverage() {
    let _trust_cg = common::EnvVarGuard::set("TY_TRUST_CG", Some("1"));
    let _trust_cg_bfs = common::EnvVarGuard::remove("TY_TRUST_CG_BFS");
    let _no_compiled = common::EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _compiled_env = common::EnvVarGuard::remove("TY_COMPILED_BFS");
    let _no_flat = common::EnvVarGuard::remove("TY_NO_FLAT_BFS");
    let _flat_env = common::EnvVarGuard::remove("TY_FLAT_BFS");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(&read_spec_source());
    let mut config = Config::parse(&read_cfg_source()).expect("valid cfg");
    config.use_compiled_bfs = Some(true);

    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();

    let coverage = checker.trust_cg_action_coverage_for_testing();

    assert_len_violation("trust-cg native run", result);

    let (compiled, total) = coverage.expect("trust-cg run should record action coverage");
    assert!(
        compiled > 0 && compiled == total,
        "expected the native SubSeq action to compile fully (no VM fallback), got {compiled}/{total}"
    );
}
