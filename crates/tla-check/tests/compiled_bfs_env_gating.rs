// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for compiled BFS env var gating.
//!
//! Compiled BFS auto-activates for all-scalar specs (no opt-in env var).
//! These tests verify that `TY_NO_COMPILED_BFS=1` still force-disables that
//! auto-activation, and that the auto-activated and interpreter paths produce
//! matching state counts.
//!
//! These tests use a simple all-scalar spec (Counter with 4 states) to
//! verify that both the interpreter and compiled BFS paths produce
//! matching state counts.
//!
//! Part of #4171: Wire compiled BFS into production BFS loop.

mod common;

use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

/// A simple all-scalar spec with 4 states, suitable for both interpreter
/// and compiled BFS paths. All state variables are integers (scalars).
const COUNTER_SPEC: &str = r#"
---- MODULE Counter ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == \/ (x = 0 /\ x' = 1 /\ y' = y)
        \/ (y = 0 /\ y' = 1 /\ x' = x)
        \/ (x = 1 /\ y = 1 /\ x' = 1 /\ y' = 1)
Inv == x \in {0, 1} /\ y \in {0, 1}
====
"#;

const COUNTER_CONFIG: &str = "INIT Init\nNEXT Next\nINVARIANT Inv\n";

/// Baseline: run without compiled BFS (interpreter path).
/// This is the reference state count all other runs must match.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_compiled_bfs_baseline_interpreter_path() {
    let _no_compiled = common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1"));
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    clear_for_test_reset();

    let module = common::parse_module(COUNTER_SPEC);
    let config = Config::parse(COUNTER_CONFIG).expect("valid cfg");
    let result = check_module(&module, &config);

    match result {
        CheckResult::Success(stats) => {
            // (0,0) -> (1,0), (0,1) -> (1,1): 4 distinct states
            assert_eq!(stats.states_found, 4, "baseline should find 4 states");
            assert_eq!(stats.initial_states, 1, "baseline should have 1 init state");
        }
        other => panic!("baseline interpreter run failed: {other:?}"),
    }
}

/// Verify that TY_NO_COMPILED_BFS=1 force-disables compiled BFS, falling back
/// to the interpreter path even though this all-scalar spec would otherwise
/// auto-activate compiled BFS. The state count must be unaffected.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_compiled_bfs_disable_suppresses_auto_activation() {
    let _disable = common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1"));
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    clear_for_test_reset();

    let module = common::parse_module(COUNTER_SPEC);
    let config = Config::parse(COUNTER_CONFIG).expect("valid cfg");
    let result = check_module(&module, &config);

    match result {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 4,
                "with disable flag, interpreter path should still find 4 states"
            );
        }
        other => panic!("disable run failed: {other:?}"),
    }
}

/// With no compiled-BFS env vars set, compiled BFS auto-activates for this
/// all-scalar spec. The result must match the interpreter baseline (4 states).
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_compiled_bfs_auto_activates_without_env_var() {
    let _no_disable = common::EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    clear_for_test_reset();

    let module = common::parse_module(COUNTER_SPEC);
    let config = Config::parse(COUNTER_CONFIG).expect("valid cfg");
    let result = check_module(&module, &config);

    match result {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 4,
                "auto-activated compiled BFS should find 4 states"
            );
        }
        other => panic!("auto-activation run failed: {other:?}"),
    }
}
