// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness test for INSTANCE-namespaced name resolution in the native
//! action-bytecode compile path.
//!
//! A base module defines all state variables and named actions; an outer
//! module reaches them only through an (implicit) `INSTANCE` layer. Before the
//! resolution fix, such specs disqualified from the trust-codegen backend with
//! `no safe action bytecodes available` / `unresolved identifier`, because the
//! TIR-seeded callee bodies for the instance-imported actions carried the
//! inner module's unresolved state-variable references.
//!
//! The action splitter already produces the fully substituted and resolved
//! action expressions (the SAME AST the interpreter evaluates). The fix routes
//! those resolved expressions into the action-bytecode compile path. These
//! tests prove the resolution is SOUND end-to-end:
//!
//! (a) cross-backend: interpreter and the default compiled BFS find the same
//!     invariant violation through the instance layer; and
//! (b) native-admission fallback parity: trust-codegen prepares every
//!     instance-imported action, its currently unstaged install manifest is
//!     rejected fail-closed, and the fallback still finds the same violation.
//!     When the coordinated Trust-CG manifest work lands, this assertion must
//!     be replaced by an admitted-native coverage assertion.
//!
//! Mirrors the assertion structure of the flat-state / subseq native-fold
//! soundness tests.

mod common;

use tla_check::{Config, ModelChecker};
use tla_core::FileId;

/// Base module: declares all state, the named (all-scalar) actions, the `Next`
/// disjunction, and the invariant. Everything the outer module needs is reached
/// through an INSTANCE of this module.
fn base_module() -> tla_core::ast::Module {
    common::parse_module_strict_with_id(
        r#"
---- MODULE InstanceResolveBase ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

\* All-scalar named actions so every split disjunct is natively executable.
IncX == /\ x < 3
        /\ x' = x + 1
        /\ y' = y
IncY == /\ x = 3
        /\ y < 2
        /\ y' = y + 1
        /\ x' = x

Next == IncX \/ IncY

vars == <<x, y>>

Spec == Init /\ [][Next]_vars

\* Violated once the counter reaches x=3 /\ y=2.
NotDone == ~(x = 3 /\ y = 2)
====
"#,
        FileId(1),
    )
}

/// Outer module: an *unnamed* `INSTANCE InstanceResolveBase` (matching variable
/// names), mirroring the Apalache `AP*` wrapper structure (e.g. APclean's
/// `INSTANCE clean`). Because the instance is unnamed, `Init`/`Next`/`IncX`/
/// `IncY`/`NotDone` are imported DIRECTLY into this module's namespace — they
/// are NOT defined here, only reached through the instance layer. The native
/// action-bytecode compile path must therefore resolve the instance-imported
/// (substituted) action bodies rather than disqualify the spec.
fn outer_module() -> tla_core::ast::Module {
    common::parse_module_strict_with_id(
        r#"
---- MODULE InstanceResolveNativeFold ----
EXTENDS Naturals

VARIABLE x, y

INSTANCE InstanceResolveBase
====
"#,
        FileId(0),
    )
}

fn violation_config() -> Config {
    Config {
        // `Init`/`Next`/`NotDone` are reached only through the unnamed INSTANCE
        // layer — they are not defined in the outer (root) module.
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["NotDone".to_string()],
        check_deadlock: false,
        ..Default::default()
    }
}

/// (a) Default cross-backend agreement: the interpreter and the default
/// compiled BFS must both find the invariant violation through the INSTANCE
/// layer. A wrong resolution would compile a different action and either miss
/// the violation or explore a different state space.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn instance_resolve_native_fold_cross_backend_finds_violation() {
    // Hold the shared integration-test environment lock for the complete run
    // so this test cannot race the trust-cg-enabled fallback test below.
    let _no_trust_cg = common::EnvVarGuard::set("TY_TRUST_CG", Some("0"));
    let _no_trust_cg_bfs = common::EnvVarGuard::remove("TY_TRUST_CG_BFS");

    tla_eval::clear_for_test_reset();

    let base = base_module();
    let outer = outer_module();
    let config = violation_config();

    let mut checker = ModelChecker::new_with_extends(&outer, &[&base], &config);
    let result = checker.check();

    assert!(
        matches!(result, tla_check::CheckResult::InvariantViolation { .. }),
        "interpreter/compiled BFS must find the INSTANCE-resolved invariant violation, got {result:?}"
    );
}

/// (b) Fail-closed native-admission parity: trust-codegen must prepare every
/// instance-imported action, reject activation because the install manifest is
/// not yet staged, and preserve the SAME violation through its fallback.
///
/// This is deliberately an assertion on the production admission report, not
/// a boolean "manifest available" test flag. Once Trust-CG supplies a validated
/// install manifest, this test will fail on the stale `missing_manifest`
/// expectation and must graduate to admitted-native execution.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn instance_resolve_native_fold_trust_cg_fail_closed_fallback_preserves_violation() {
    let _trust_cg = common::EnvVarGuard::set("TY_TRUST_CG", Some("1"));
    let _trust_cg_bfs = common::EnvVarGuard::remove("TY_TRUST_CG_BFS");

    tla_eval::clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let base = base_module();
    let outer = outer_module();
    let config = violation_config();

    let mut checker = ModelChecker::new_with_extends(&outer, &[&base], &config);
    let result = checker.check();

    assert!(
        matches!(result, tla_check::CheckResult::InvariantViolation { .. }),
        "trust-cg fail-closed fallback must find the INSTANCE-resolved invariant violation, \
         got {result:?}"
    );

    let (compiled, total) = checker
        .trust_cg_action_coverage_for_testing()
        .expect("trust-cg preparation should record executable action coverage");
    assert!(
        compiled == total && total > 0,
        "instance-imported actions must all reach native preparation before admission, \
         got {compiled}/{total}"
    );

    let admission = checker
        .trust_cg_native_admission_evidence_report_json()
        .expect("trust-cg preparation should record structured native-admission evidence");
    assert_eq!(
        admission["fields"]["status_code"].as_str(),
        Some("rejected"),
        "unstaged native artifacts must be rejected fail-closed"
    );
    assert_eq!(
        admission["fields"]["reason_code"].as_str(),
        Some("missing_manifest"),
        "the current coordinated blocker must remain explicit"
    );
    assert_eq!(
        admission["fields"]["production_selected"].as_str(),
        Some("false"),
        "a missing manifest must never select native production execution"
    );
    assert_eq!(
        admission["fields"]["actions_ty_native_activate"].as_str(),
        Some("false"),
        "a missing manifest must deny TY native activation"
    );
}
