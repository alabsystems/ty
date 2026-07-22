// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Baseline pass/fail/error semantics for `check_invariants_array`.

use super::*;
use crate::config::ConstantValue;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_invariants_all_pass() {
    let module = parse_module(
        r#"
---- MODULE InvPass ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Safety == x >= 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let state = State::from_pairs([("x", Value::int(0))]);
    // Part of #2484 Phase 3: use ArrayState path
    let registry = mc.ctx.var_registry().clone();
    let arr = ArrayState::from_state(&state, &registry);
    let result = mc.check_invariants_array(&arr);
    assert_eq!(result.unwrap(), None);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn invariant_verdict_cache_commits_only_after_whole_ordered_miss_set_passes() {
    let module = parse_module(
        r#"
---- MODULE InvVerdictCommitOrder ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x + 1
First == x >= 0
Second == x > 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["First".to_string(), "Second".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let registry = mc.ctx.var_registry().clone();
    let violating = ArrayState::from_state(&State::from_pairs([("x", Value::int(0))]), &registry);
    assert_eq!(
        mc.check_invariants_array(&violating).unwrap(),
        Some("Second".to_string())
    );
    assert_eq!(mc.invariant_verdict_cache.test_entry_count(), 0);

    let passing = ArrayState::from_state(&State::from_pairs([("x", Value::int(1))]), &registry);
    assert_eq!(mc.check_invariants_array(&passing).unwrap(), None);
    assert_eq!(mc.invariant_verdict_cache.test_entry_count(), 2);
    let hits_before = mc.invariant_verdict_cache.test_hit_count();
    assert_eq!(mc.check_invariants_array(&passing).unwrap(), None);
    assert_eq!(mc.invariant_verdict_cache.test_hit_count() - hits_before, 2);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn invariant_conjunct_cache_preserves_order_and_delays_commit() {
    let module = parse_module(
        r#"
---- MODULE InvConjunctOrder ----
EXTENDS Integers, TLC
VARIABLE guard, divisor, stable, z
Init == /\ guard = TRUE /\ divisor = 1 /\ stable = 1 /\ z = 1
Next == UNCHANGED <<guard, divisor, stable, z>>
First ==
    /\ TLCGet("level") = 0
    /\ guard
    /\ 10 \div divisor > 0
    /\ stable = 1
Second == z > 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["First".to_string(), "Second".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.tir_parity = None;
    mc.invariant_verdict_cache
        .test_enable_conjunct_cache(&mc.ctx, &config.invariants);
    assert_eq!(
        mc.invariant_verdict_cache.test_active_conjunct_plan_count(),
        3
    );
    let registry = mc.ctx.var_registry().clone();
    mc.ctx.set_tlc_level(0);

    let later_failure = ArrayState::from_state(
        &State::from_pairs([
            ("guard", Value::Bool(true)),
            ("divisor", Value::int(1)),
            ("stable", Value::int(1)),
            ("z", Value::int(0)),
        ]),
        &registry,
    );
    assert_eq!(
        mc.check_invariants_array(&later_failure).unwrap(),
        Some("Second".to_string())
    );
    assert_eq!(mc.invariant_verdict_cache.test_entry_count(), 0);

    let warm = ArrayState::from_state(
        &State::from_pairs([
            ("guard", Value::Bool(true)),
            ("divisor", Value::int(1)),
            ("stable", Value::int(1)),
            ("z", Value::int(1)),
        ]),
        &registry,
    );
    assert_eq!(mc.check_invariants_array(&warm).unwrap(), None);
    assert_eq!(mc.invariant_verdict_cache.test_entry_count(), 4);

    let false_before_error = ArrayState::from_state(
        &State::from_pairs([
            ("guard", Value::Bool(false)),
            ("divisor", Value::int(0)),
            ("stable", Value::int(1)),
            ("z", Value::int(1)),
        ]),
        &registry,
    );
    assert_eq!(
        mc.check_invariants_array(&false_before_error).unwrap(),
        Some("First".to_string())
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn invariant_conjunct_cache_runs_under_implicit_default_tir() {
    let module = parse_module(
        r#"
---- MODULE InvConjunctTirOwnership ----
EXTENDS Integers, TLC
VARIABLE x
Init == x = 0
Next == x' = x
Safety == /\ TLCGet("level") = 0 /\ x >= 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.invariant_verdict_cache
        .test_enable_conjunct_cache(&mc.ctx, &config.invariants);
    assert_eq!(
        mc.invariant_verdict_cache.test_active_conjunct_plan_count(),
        1
    );
    mc.bytecode = None;
    mc.ctx.set_tlc_level(0);
    let registry = mc.ctx.var_registry().clone();
    let state = ArrayState::from_state(&State::from_pairs([("x", Value::int(0))]), &registry);
    assert_eq!(mc.check_invariants_array(&state).unwrap(), None);
    assert_eq!(mc.check_invariants_array(&state).unwrap(), None);
    assert_eq!(mc.invariant_verdict_cache.test_hit_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn invariant_verdict_cache_rejects_closure_valued_config_constant() {
    let module = parse_module(
        r#"
---- MODULE InvVerdictConfigClosure ----
EXTENDS Integers, VectorClocks
CONSTANT Clock
VARIABLE x
Init == x = 0
Next == x' = x + 1
Safe == IsCausalOrder(<<1, 2>>, Clock)
====
"#,
    );
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safe".to_string()],
        ..Default::default()
    };
    config.add_constant(
        "Clock".to_string(),
        ConstantValue::Value("LAMBDA event : [n |-> IF event = 1 THEN x ELSE 0]".to_string()),
    );
    let mut mc = ModelChecker::new(&module, &config);
    mc.prepare_bfs_common().unwrap();
    let registry = mc.ctx.var_registry().clone();
    let passing = ArrayState::from_state(&State::from_pairs([("x", Value::int(0))]), &registry);
    let violating = ArrayState::from_state(&State::from_pairs([("x", Value::int(1))]), &registry);
    assert_eq!(mc.check_invariants_array(&passing).unwrap(), None);
    assert_eq!(mc.invariant_verdict_cache.test_entry_count(), 0);
    assert_eq!(
        mc.check_invariants_array(&violating).unwrap(),
        Some("Safe".to_string())
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn invariant_verdict_cache_rejects_non_concrete_precomputed_operator() {
    let module = parse_module(
        r#"
---- MODULE InvVerdictPrecomputedClosure ----
EXTENDS Integers, TLC, VectorClocks
VARIABLE x
Init == x = 0
Next == x' = x
Clock == LAMBDA event : [n |-> IF event = 1 THEN TLCGet("level") ELSE 0]
Safe == IsCausalOrder(<<1, 2>>, Clock)
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safe".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.prepare_bfs_common().unwrap();
    let clock_id = tla_core::name_intern::lookup_name_id("Clock").unwrap();
    assert!(!mc
        .ctx
        .precomputed_constants()
        .get(&clock_id)
        .unwrap()
        .is_concrete_data());

    let registry = mc.ctx.var_registry().clone();
    let state = ArrayState::from_state(&State::from_pairs([("x", Value::int(0))]), &registry);
    mc.ctx.set_tlc_level(0);
    assert_eq!(mc.check_invariants_array(&state).unwrap(), None);
    assert_eq!(mc.invariant_verdict_cache.test_entry_count(), 0);
    mc.ctx.set_tlc_level(1);
    assert_eq!(
        mc.check_invariants_array(&state).unwrap(),
        Some("Safe".to_string())
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn invariant_verdict_cache_is_disabled_during_jit_verification() {
    let module = parse_module(
        r#"
---- MODULE InvVerdictJitVerify ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x
Safe == x >= 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safe".to_string()],
        jit_verify: true,
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let registry = mc.ctx.var_registry().clone();
    let state = ArrayState::from_state(&State::from_pairs([("x", Value::int(0))]), &registry);
    assert_eq!(mc.check_invariants_array(&state).unwrap(), None);
    assert_eq!(mc.check_invariants_array(&state).unwrap(), None);
    assert_eq!(mc.invariant_verdict_cache.test_entry_count(), 0);
    assert_eq!(mc.invariant_verdict_cache.test_hit_count(), 0);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_invariants_clears_stale_action_scope() {
    let module = parse_module(
        r#"
---- MODULE InvActionScope ----
EXTENDS TLC
VARIABLE x
Init == x = 0
Next == x' = x + 1
ActionEmpty == TLCGet("action").name = ""
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["ActionEmpty".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let current = State::from_pairs([("x", Value::int(0))]);
    let next = State::from_pairs([("x", Value::int(1))]);
    let registry = mc.ctx.var_registry().clone();
    let current_arr = ArrayState::from_state(&current, &registry);
    let next_arr = ArrayState::from_state(&next, &registry);

    let _stale_next_guard = mc.ctx.bind_next_state_env_guard(next_arr.env_ref());

    let result = mc.check_invariants_array(&current_arr);
    assert_eq!(
        result.unwrap(),
        None,
        "state invariants must clear stale action scope before evaluating TLCGet(\"action\")"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_invariants_violation() {
    let module = parse_module(
        r#"
---- MODULE InvFail ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Positive == x > 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Positive".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    // x = 0 violates Positive (x > 0)
    let state = State::from_pairs([("x", Value::int(0))]);
    // Part of #2484 Phase 3: use ArrayState path
    let registry = mc.ctx.var_registry().clone();
    let arr = ArrayState::from_state(&state, &registry);
    let result = mc.check_invariants_array(&arr).unwrap();
    assert_eq!(result, Some("Positive".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_invariants_non_boolean_error() {
    let module = parse_module(
        r#"
---- MODULE InvNonBool ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
NotBool == x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["NotBool".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let state = State::from_pairs([("x", Value::int(0))]);
    // Part of #2484 Phase 3: use ArrayState path
    let registry = mc.ctx.var_registry().clone();
    let arr = ArrayState::from_state(&state, &registry);
    let result = mc.check_invariants_array(&arr);
    assert!(matches!(
        result,
        Err(CheckError::Eval(EvalCheckError::Eval(
            crate::EvalError::TypeError {
                expected: "BOOLEAN",
                ..
            }
        )))
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_invariants_empty_invariants_passes() {
    let module = parse_module(
        r#"
---- MODULE InvEmpty ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let state = State::from_pairs([("x", Value::int(0))]);
    // Part of #2484 Phase 3: use ArrayState path
    let registry = mc.ctx.var_registry().clone();
    let arr = ArrayState::from_state(&state, &registry);
    let result = mc.check_invariants_array(&arr);
    assert_eq!(result.unwrap(), None);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_invariants_first_fail_stops() {
    // When multiple invariants are configured, check stops at the first failure
    let module = parse_module(
        r#"
---- MODULE InvMulti ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
AlwaysFalse == FALSE
AlsoFalse == FALSE
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["AlwaysFalse".to_string(), "AlsoFalse".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let state = State::from_pairs([("x", Value::int(0))]);
    // Part of #2484 Phase 3: use ArrayState path
    let registry = mc.ctx.var_registry().clone();
    let arr = ArrayState::from_state(&state, &registry);
    let result = mc.check_invariants_array(&arr).unwrap();
    // Should report the first failing invariant
    assert_eq!(result, Some("AlwaysFalse".to_string()));
}

/// Test that cooperative invariant skip (full) causes `check_successor_invariant`
/// to return `Ok` even when an invariant is violated.
///
/// Part of #3810: verifies the full-skip path — when PDR proves ALL invariants,
/// the BFS lane skips per-state invariant evaluation entirely.
#[cfg(feature = "ay")]
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cooperative_full_invariant_skip_returns_ok() {
    use crate::checker_ops::InvariantOutcome;
    use crate::cooperative_state::SharedCooperativeState;
    use std::sync::Arc;

    let module = parse_module(
        r#"
---- MODULE CoopFullSkip ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
AlwaysFalse == FALSE
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["AlwaysFalse".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);

    // Without cooperative state, checking should report a violation.
    let registry = mc.ctx.var_registry().clone();
    let state = State::from_pairs([("x", Value::int(0))]);
    let arr = ArrayState::from_state(&state, &registry);
    let fp = Fingerprint(42);

    let outcome_before = mc.check_successor_invariant(Fingerprint(1), &arr, fp, 1);
    assert!(
        matches!(outcome_before, InvariantOutcome::Violation { ref invariant, .. } if invariant == "AlwaysFalse"),
        "without cooperative state, AlwaysFalse should be violated"
    );

    // Set cooperative state and mark all invariants proved.
    let coop = Arc::new(SharedCooperativeState::with_invariant_count(0, 1));
    coop.set_invariants_proved();
    mc.set_cooperative_state(coop);

    // Now check_successor_invariant should skip and return Ok.
    let outcome_after = mc.check_successor_invariant(Fingerprint(1), &arr, fp, 1);
    assert!(
        matches!(outcome_after, InvariantOutcome::Ok),
        "with all invariants proved by PDR, should skip and return Ok"
    );
}

/// Test that cooperative partial invariant skip only checks unproved invariants.
///
/// Part of #3810: verifies the partial-skip path — when PDR proves some
/// invariants, BFS only evaluates the unproved ones per-state.
#[cfg(feature = "ay")]
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cooperative_partial_invariant_skip_checks_only_unproved() {
    use crate::checker_ops::InvariantOutcome;
    use crate::cooperative_state::SharedCooperativeState;
    use std::sync::Arc;

    let module = parse_module(
        r#"
---- MODULE CoopPartialSkip ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
AlwaysFalse == FALSE
AlwaysTrue == TRUE
====
"#,
    );
    // Config with two invariants: AlwaysFalse (index 0) and AlwaysTrue (index 1).
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["AlwaysFalse".to_string(), "AlwaysTrue".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);

    let registry = mc.ctx.var_registry().clone();
    let state = State::from_pairs([("x", Value::int(0))]);
    let arr = ArrayState::from_state(&state, &registry);
    let fp = Fingerprint(42);

    // Without cooperative state, checking should report AlwaysFalse violation.
    let outcome_no_coop = mc.check_successor_invariant(Fingerprint(1), &arr, fp, 1);
    assert!(
        matches!(outcome_no_coop, InvariantOutcome::Violation { ref invariant, .. } if invariant == "AlwaysFalse"),
        "without cooperative, AlwaysFalse should be violated"
    );

    // Set cooperative state: mark invariant 0 (AlwaysFalse) as proved by PDR.
    // Only invariant 1 (AlwaysTrue) should be checked.
    let coop = Arc::new(SharedCooperativeState::with_invariant_count(0, 2));
    coop.mark_invariant_proved(0); // Prove AlwaysFalse
    assert!(coop.has_partial_proofs(), "should have partial proofs");
    mc.set_cooperative_state(coop);

    // Now check_successor_invariant should only check AlwaysTrue (which passes).
    let outcome_partial = mc.check_successor_invariant(Fingerprint(1), &arr, fp, 1);
    assert!(
        matches!(outcome_partial, InvariantOutcome::Ok),
        "with AlwaysFalse proved by PDR, only AlwaysTrue is checked and passes"
    );
}

/// Test that cooperative partial skip still detects violations in unproved invariants.
///
/// Part of #3810: when PDR proves one invariant but another (unproved) one is
/// violated, the violation must still be detected.
#[cfg(feature = "ay")]
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cooperative_partial_skip_detects_unproved_violation() {
    use crate::checker_ops::InvariantOutcome;
    use crate::cooperative_state::SharedCooperativeState;
    use std::sync::Arc;

    let module = parse_module(
        r#"
---- MODULE CoopPartialViolation ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
AlwaysTrue == TRUE
AlwaysFalse == FALSE
====
"#,
    );
    // Config: AlwaysTrue (index 0), AlwaysFalse (index 1).
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["AlwaysTrue".to_string(), "AlwaysFalse".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);

    let registry = mc.ctx.var_registry().clone();
    let state = State::from_pairs([("x", Value::int(0))]);
    let arr = ArrayState::from_state(&state, &registry);
    let fp = Fingerprint(42);

    // Prove AlwaysTrue (index 0), leave AlwaysFalse (index 1) unproved.
    let coop = Arc::new(SharedCooperativeState::with_invariant_count(0, 2));
    coop.mark_invariant_proved(0); // Prove AlwaysTrue
    mc.set_cooperative_state(coop);

    // AlwaysFalse is still unproved and violated — should be detected.
    let outcome = mc.check_successor_invariant(Fingerprint(1), &arr, fp, 1);
    assert!(
        matches!(outcome, InvariantOutcome::Violation { ref invariant, .. } if invariant == "AlwaysFalse"),
        "unproved AlwaysFalse should still be detected as violated"
    );
}
