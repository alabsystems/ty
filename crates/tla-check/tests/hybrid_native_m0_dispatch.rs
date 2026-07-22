// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Item 4 M0 integration: hybrid per-action NATIVE flat-view dispatch.
//!
//! A 3-variable model — two flat scalars (`x`, `y`) plus one compound `Set`
//! (`data`) — where:
//!
//! - `IncX` / `BumpY` have entirely flat-admissible footprints, so under
//!   `TY_HYBRID_FLAT_VIEW=1` + `TY_HYBRID_NATIVE=1` + `TY_TRUST_CG=1` they are
//!   compiled against the HYBRID flat-view layout and executed natively per
//!   (parent, action); every routed successor is validated value- and
//!   fingerprint-equal against the interpreter successor by the fail-closed
//!   differential (M0 stays a validated shadow: the interpreter is
//!   authoritative on any divergence).
//! - `Touch` writes the compound `data` var, so it is NOT hybrid-eligible and
//!   stays on the interpreter (and its hybrid compile declines at the
//!   `Dynamic`-placeholder guard).
//!
//! The whole-state veto (`supports_flat_primary`) is false for this spec (the
//! `Set` var is un-flattenable), so WITHOUT the hybrid path zero actions could
//! dispatch natively — this test pins the first native dispatch on a compound
//! spec.

mod common;

use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

use tla_check::ModelChecker;

const HYBRID_TLA: &str = r#"
----------------------------- MODULE hybridm0 -----------------------------
EXTENDS Naturals

VARIABLES x, y, data

IncX == /\ x < 3
        /\ x' = x + 1
        /\ UNCHANGED <<y, data>>

BumpY == /\ y < 2
         /\ y' = y + 1
         /\ UNCHANGED <<x, data>>

Touch == /\ x > 0
         /\ data' = data \cup {x}
         /\ UNCHANGED <<x, y>>

Init == /\ x = 0
        /\ y = 0
        /\ data = {}

Next == \/ IncX
        \/ BumpY
        \/ Touch
=============================================================================
"#;

const HYBRID_CFG: &str = r#"
INIT Init
NEXT Next
"#;

fn states_found(label: &str, result: &CheckResult) -> usize {
    match result {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("{label}: expected Success, got {other:?}"),
    }
}

fn hybrid_env_guards() -> Vec<common::EnvVarGuard> {
    vec![
        common::EnvVarGuard::set("TY_TRUST_CG", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_FLAT_VIEW", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_NATIVE", Some("1")),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
        // Keep the run on the interpreter BFS loop: the hybrid per-action
        // native dispatch lives in the per-action successor path, and the
        // whole-state compiled-BFS loop is a different (fully-flat) engine
        // that must not race it in this test.
        common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1")),
        common::EnvVarGuard::remove("TY_TRUST_CG_BFS"),
        common::EnvVarGuard::remove("TY_COMPILED_BFS"),
        common::EnvVarGuard::remove("TY_NO_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_JIT"),
    ]
}

/// Interpreter baseline: reachable-state count with every native/hybrid switch
/// off.
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn hybrid_m0_interpreter_baseline_state_count() {
    let _no_trust_cg = common::EnvVarGuard::remove("TY_TRUST_CG");
    let _no_hybrid = common::EnvVarGuard::remove("TY_HYBRID_FLAT_VIEW");
    let _no_native = common::EnvVarGuard::remove("TY_HYBRID_NATIVE");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    clear_for_test_reset();

    let module = common::parse_module(HYBRID_TLA);
    let config = Config::parse(HYBRID_CFG).expect("valid cfg");
    let result = check_module(&module, &config);
    assert_eq!(
        states_found("interpreter baseline", &result),
        45,
        "interpreter reachable-state count (x in 0..3, y in 0..2, data any subset of 1..x)"
    );
}

/// The M0 native run: flat-footprint actions execute NATIVELY against the
/// hybrid layout; the compound-touching action stays interpreter; the routed
/// successor set is exactly the interpreter's (differential-exact,
/// `mismatch_fallback == 0`).
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn hybrid_m0_native_dispatch_is_differential_exact() {
    let _guards = hybrid_env_guards();
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(HYBRID_TLA);
    let config = Config::parse(HYBRID_CFG).expect("valid cfg");

    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();

    assert_eq!(
        states_found("hybrid native run", &result),
        45,
        "hybrid native run must reach the exact interpreter state count"
    );

    // The hybrid-layout cache compiled the flat-footprint actions.
    assert!(
        checker.hybrid_native_cache_ready_for_testing(),
        "the hybrid-layout native action cache must be built and non-empty"
    );

    // Exactly the two flat-footprint actions are hybrid-eligible; the
    // compound-touching `Touch` is not (interpreter-routed).
    assert_eq!(
        checker.hybrid_eligible_action_count_for_testing(),
        2,
        "IncX and BumpY are hybrid-eligible; Touch (compound Set footprint) is not"
    );

    let (
        routed,
        mismatch_fallback,
        _projection_declined,
        native_dispatched,
        native_matched,
        _native_declined,
        native_errors,
    ) = checker.hybrid_dispatch_stats_for_testing();
    assert!(
        native_dispatched > 0,
        "compiled hybrid artifacts must actually execute (native_dispatched > 0)"
    );
    assert!(
        native_matched > 0,
        "native successors must byte-exactly match interpreter successors (native_matched > 0)"
    );
    assert!(routed > 0, "hybrid-routed successors must be recorded");
    assert_eq!(
        mismatch_fallback, 0,
        "the fail-closed differential must find ZERO native/interpreter divergences"
    );
    assert_eq!(native_errors, 0, "native execution must not error");
}

/// The hybrid FLAT VIEW alone (native switch off) keeps the pre-existing
/// validated-shadow behavior: routed successors, no native execution.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn hybrid_m0_shadow_only_without_native_switch() {
    let _trust_cg = common::EnvVarGuard::set("TY_TRUST_CG", Some("1"));
    let _hybrid = common::EnvVarGuard::set("TY_HYBRID_FLAT_VIEW", Some("1"));
    let _no_native = common::EnvVarGuard::remove("TY_HYBRID_NATIVE");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(HYBRID_TLA);
    let config = Config::parse(HYBRID_CFG).expect("valid cfg");

    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();

    assert_eq!(states_found("hybrid shadow run", &result), 45);
    let (
        _routed,
        mismatch_fallback,
        _projection_declined,
        native_dispatched,
        native_matched,
        _native_declined,
        native_errors,
    ) = checker.hybrid_dispatch_stats_for_testing();
    assert_eq!(
        native_dispatched, 0,
        "without TY_HYBRID_NATIVE no native execution may happen"
    );
    assert_eq!(native_matched, 0);
    assert_eq!(native_errors, 0);
    assert_eq!(mismatch_fallback, 0);
}
