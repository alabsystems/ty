// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! End-to-end integration test for the flat_state_primary BFS path.
//!
//! Verifies that the full flat-state BFS pipeline works correctly:
//! - `flat_state_primary=true` detection for all-scalar specs
//! - FlatState fingerprinting (xxh3 on raw i64 buffers)
//! - `generate_successors_filtered_flat` action dispatch
//!
//! The test uses simple all-scalar specs (only Int/Bool state variables)
//! and verifies that the interpreter baseline state count is correct.
//!
//! Part of #3986: Flat i64 state as primary BFS representation.
//!
//! Note: retired-backend parity tests were removed as part of Stage 2c deletion
//! (#4266). trust-codegen parity coverage is retained for the native path.

mod common;

use tla_check::{check_module, CheckResult, Config};
#[cfg(feature = "testing")]
use tla_check::{CheckStats, ModelChecker};
use tla_eval::clear_for_test_reset;

#[cfg(feature = "testing")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SuccessSummary {
    states_found: usize,
    initial_states: usize,
    transitions: usize,
}

#[cfg(feature = "testing")]
impl SuccessSummary {
    fn from_stats(stats: &CheckStats) -> Self {
        Self {
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
        }
    }
}

#[cfg(feature = "testing")]
#[derive(Debug)]
struct NativeParityRun {
    summary: SuccessSummary,
    trust_cg_action_coverage: Option<(usize, usize)>,
}

#[cfg(feature = "testing")]
fn run_success(spec: &str, config: &str, trust_cg: bool, label: &str) -> NativeParityRun {
    let _trust_cg = common::EnvVarGuard::set("TY_trust_cg", trust_cg.then_some("1"));
    let _trust_cg_bfs = common::EnvVarGuard::remove("TY_TRUST_CG_BFS");
    let _trust_cg_exists = common::EnvVarGuard::remove("TY_TRUST_CG_EXISTS");
    let _no_compiled = common::EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _no_flat = common::EnvVarGuard::remove("TY_NO_FLAT_BFS");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(spec);
    let config = Config::parse(config).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    let summary = match result {
        CheckResult::Success(stats) => SuccessSummary::from_stats(&stats),
        other => panic!("{label} expected successful model check, got {other:?}"),
    };

    NativeParityRun {
        summary,
        trust_cg_action_coverage: checker.trust_cg_action_coverage_for_testing(),
    }
}

// ============================================================================
// Spec: DieHard water jug puzzle (all-scalar: two integer variables)
// ============================================================================

/// Classic DieHard water jug puzzle. State variables are `big` and `small`,
/// both integers in 0..5 and 0..3 respectively. This is the canonical
/// all-scalar spec for testing flat_state_primary detection.
///
/// State space: 16 distinct reachable states (full exploration with TypeOK).
/// All variables are Int — qualifies for flat_state_primary=true.
const DIEHARD_SPEC: &str = r#"
---- MODULE DieHardFlat ----
VARIABLE big, small
SmallCap == 3
BigCap == 5
Init ==
    /\ big = 0
    /\ small = 0
Min(m, n) == IF m < n THEN m ELSE n
FillSmallJug ==
    /\ small' = SmallCap
    /\ big' = big
FillBigJug ==
    /\ big' = BigCap
    /\ small' = small
EmptySmallJug ==
    /\ small' = 0
    /\ big' = big
EmptyBigJug ==
    /\ big' = 0
    /\ small' = small
SmallToBig ==
    /\ big' = Min(big + small, BigCap)
    /\ small' = small - (big' - big)
BigToSmall ==
    /\ small' = Min(big + small, SmallCap)
    /\ big' = big - (small' - small)
Next ==
    \/ FillSmallJug
    \/ FillBigJug
    \/ EmptySmallJug
    \/ EmptyBigJug
    \/ SmallToBig
    \/ BigToSmall
TypeOK == big \in 0..BigCap /\ small \in 0..SmallCap
====
"#;

const DIEHARD_CONFIG: &str = "INIT Init\nNEXT Next\nINVARIANT TypeOK\n";

// ============================================================================
// Spec: Scalar counter with complete trust-codegen action coverage
// ============================================================================

#[cfg(feature = "testing")]
const SCALAR_COUNTER_SPEC: &str = r#"
---- MODULE ScalarCounterFlat ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Inc == /\ x < 3
       /\ x' = x + 1
Stay == /\ x = 3
        /\ x' = x
Next == Inc \/ Stay
TypeOK == x \in 0..3
====
"#;

#[cfg(feature = "testing")]
const SCALAR_COUNTER_CONFIG: &str = "INIT Init\nNEXT Next\nINVARIANT TypeOK\n";

// ============================================================================
// Spec: Scalar state with mixed trust_cg/fallback action coverage
// ============================================================================

#[cfg(feature = "testing")]
const MIXED_FALLBACK_SPEC: &str = r#"
---- MODULE MixedFallbackFlat ----
EXTENDS Naturals, Sequences
VARIABLE x
Init == x = 0
FastInc == /\ x = 0
           /\ x' = 1
FallbackSequence == /\ x = 1
                    /\ x' = Len(Append(<<>>, x)) + 1
Stay == /\ x = 2
        /\ x' = x
Next == FastInc \/ FallbackSequence \/ Stay
TypeOK == x \in 0..2
====
"#;

#[cfg(feature = "testing")]
const MIXED_FALLBACK_CONFIG: &str = "INIT Init\nNEXT Next\nINVARIANT TypeOK\n";

// ============================================================================
// Test: Interpreter baseline for DieHard (establishes expected state count)
// ============================================================================

/// Run DieHard without JIT to establish the baseline state count (16 states).
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_flat_state_primary_diehard_interpreter_baseline() {
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    clear_for_test_reset();

    let module = common::parse_module(DIEHARD_SPEC);
    let config = Config::parse(DIEHARD_CONFIG).expect("valid cfg");
    let result = check_module(&module, &config);

    match result {
        CheckResult::Success(stats) => {
            // DieHard has 16 reachable states with TypeOK invariant (full exploration).
            assert_eq!(
                stats.states_found, 16,
                "DieHard baseline should find exactly 16 states, got {}",
                stats.states_found,
            );
            assert_eq!(stats.initial_states, 1, "DieHard has 1 init state");
            eprintln!(
                "[test] DieHard interpreter baseline: {} states, {} initial",
                stats.states_found, stats.initial_states
            );
        }
        other => panic!("DieHard interpreter baseline failed: {other:?}"),
    }
}

/// trust-codegen native dispatch should preserve exact counters for a simple
/// all-scalar flat-state-primary spec when every action is eligible.
#[cfg(feature = "testing")]
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_flat_state_primary_trust_cg_complete_action_coverage_matches_interpreter() {
    let baseline = run_success(
        SCALAR_COUNTER_SPEC,
        SCALAR_COUNTER_CONFIG,
        false,
        "scalar counter interpreter baseline",
    );
    let trust_cg = run_success(
        SCALAR_COUNTER_SPEC,
        SCALAR_COUNTER_CONFIG,
        true,
        "scalar counter trust-codegen run",
    );

    assert_eq!(
        baseline.summary,
        SuccessSummary {
            states_found: 4,
            initial_states: 1,
            transitions: 4,
        },
        "fixture should stay small and deterministic"
    );
    assert_eq!(
        trust_cg.summary, baseline.summary,
        "trust-cg complete native coverage must preserve flat-state counters"
    );

    let (compiled, total) = trust_cg
        .trust_cg_action_coverage
        .expect("trust-cg run should record action coverage");
    assert!(
        compiled > 0 && compiled == total,
        "expected complete trust-codegen action coverage, got {compiled}/{total}"
    );
}

/// Unsupported action shapes must stay interpreter-backed without changing
/// counters for the surrounding flat-state-primary run.
#[cfg(feature = "testing")]
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_flat_state_primary_trust_cg_mixed_action_fallback_matches_interpreter() {
    let baseline = run_success(
        MIXED_FALLBACK_SPEC,
        MIXED_FALLBACK_CONFIG,
        false,
        "mixed fallback interpreter baseline",
    );
    let trust_cg = run_success(
        MIXED_FALLBACK_SPEC,
        MIXED_FALLBACK_CONFIG,
        true,
        "mixed fallback trust-codegen run",
    );

    assert_eq!(
        baseline.summary,
        SuccessSummary {
            states_found: 3,
            initial_states: 1,
            transitions: 3,
        },
        "fixture should exercise one interpreted fallback transition"
    );
    assert_eq!(
        trust_cg.summary, baseline.summary,
        "trust-cg mixed native/interpreter coverage must preserve flat-state counters"
    );

    let (compiled, total) = trust_cg
        .trust_cg_action_coverage
        .expect("trust-cg run should record action coverage");
    assert!(
        compiled > 0 && compiled < total,
        "expected mixed trust-cg/fallback action coverage, got {compiled}/{total}"
    );
}
