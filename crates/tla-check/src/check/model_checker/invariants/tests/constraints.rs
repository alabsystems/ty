// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for `check_state_constraints_array`: empty, satisfied, violated.

use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_state_constraints_empty_passes() {
    let module = parse_module(
        r#"
---- MODULE ConstEmpty ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec![],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let state = State::from_pairs([("x", Value::int(5))]);
    let registry = mc.ctx.var_registry().clone();
    let arr = ArrayState::from_state(&state, &registry);
    assert!(matches!(mc.check_state_constraints_array(&arr), Ok(true)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_state_constraints_satisfied() {
    let module = parse_module(
        r#"
---- MODULE ConstPass ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Bound == x < 10
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["Bound".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let state = State::from_pairs([("x", Value::int(5))]);
    let registry = mc.ctx.var_registry().clone();
    let arr = ArrayState::from_state(&state, &registry);
    assert!(matches!(mc.check_state_constraints_array(&arr), Ok(true)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_state_constraints_violated() {
    let module = parse_module(
        r#"
---- MODULE ConstFail ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Bound == x < 10
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["Bound".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let state = State::from_pairs([("x", Value::int(15))]);
    let registry = mc.ctx.var_registry().clone();
    let arr = ArrayState::from_state(&state, &registry);
    assert!(matches!(mc.check_state_constraints_array(&arr), Ok(false)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn state_constraint_cache_uses_exact_inline_scalar_verdict_keys() {
    let module = parse_module(
        r#"
---- MODULE ConstraintVerdictInlineScalar ----
VARIABLES x, irrelevant
Init == /\ x = TRUE /\ irrelevant = 0
Next == UNCHANGED <<x, irrelevant>>
IsTrue == x = TRUE
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["IsTrue".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.tir_parity = None;
    let registry = mc.ctx.var_registry().clone();
    let make_state = |x: Value, irrelevant: i64| {
        ArrayState::from_state(
            &State::from_pairs([("x", x), ("irrelevant", Value::int(irrelevant))]),
            &registry,
        )
    };

    assert!(mc
        .check_state_constraints_array(&make_state(Value::Bool(true), 0))
        .unwrap());
    assert!(!mc
        .check_state_constraints_array(&make_state(Value::int(1), 0))
        .unwrap());
    assert!(mc
        .check_state_constraints_array(&make_state(Value::Bool(true), 99))
        .unwrap());
    assert!(!mc
        .check_state_constraints_array(&make_state(Value::int(1), 99))
        .unwrap());

    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 2);
    assert_eq!(
        mc.state_constraint_verdict_cache.test_inline_entry_count(),
        2
    );
    assert_eq!(mc.state_constraint_verdict_cache.test_hit_count(), 2);

    assert!(!mc
        .check_state_constraints_array(&make_state(Value::seq([Value::int(1), Value::int(2)]), 0,))
        .unwrap());
    assert!(!mc
        .check_state_constraints_array(&make_state(Value::seq([Value::int(1), Value::int(2)]), 99,))
        .unwrap());
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 3);
    assert_eq!(
        mc.state_constraint_verdict_cache.test_inline_entry_count(),
        2,
        "heap-backed values must use the exact projected-Value fallback"
    );
    assert_eq!(mc.state_constraint_verdict_cache.test_hit_count(), 3);

    let constraints = mc.config.constraints.clone();
    mc.state_constraint_verdict_cache
        .rebuild(&mc.ctx, &constraints);
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 0);
    assert!(mc.state_constraint_verdict_cache.test_is_enabled());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn state_constraint_cache_uses_the_exact_union_projection() {
    let module = parse_module(
        r#"
---- MODULE ConstraintVerdictUnion ----
EXTENDS Integers
VARIABLES x, y, irrelevant
Init == /\ x = 0 /\ y = 1 /\ irrelevant = 0
Next == UNCHANGED <<x, y, irrelevant>>
First == x < 10
Second == y > 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["First".to_string(), "Second".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.tir_parity = None;
    assert!(mc.state_constraint_verdict_cache.test_is_enabled());
    let registry = mc.ctx.var_registry().clone();
    let make_state = |x: i64, y: i64, irrelevant: i64| {
        ArrayState::from_state(
            &State::from_pairs([
                ("x", Value::int(x)),
                ("y", Value::int(y)),
                ("irrelevant", Value::int(irrelevant)),
            ]),
            &registry,
        )
    };

    assert!(mc
        .check_state_constraints_array(&make_state(5, 1, 0))
        .unwrap());
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 1);
    let hits_before = mc.state_constraint_verdict_cache.test_hit_count();
    assert!(mc
        .check_state_constraints_array(&make_state(5, 1, 99))
        .unwrap());
    assert_eq!(
        mc.state_constraint_verdict_cache.test_hit_count() - hits_before,
        1,
        "changing an unprojected slot should preserve the exact TRUE hit"
    );

    assert!(mc
        .check_state_constraints_array(&make_state(6, 1, 99))
        .unwrap());
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 2);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn state_constraint_cache_reuses_false_but_never_commits_error() {
    let module = parse_module(
        r#"
---- MODULE ConstraintVerdictNegative ----
EXTENDS Integers
VARIABLES x, y
Init == /\ x = 1 /\ y = 0
Next == UNCHANGED <<x, y>>
First == x > 0
Second == IF y = 0 THEN FALSE ELSE y
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["First".to_string(), "Second".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.tir_parity = None;
    let registry = mc.ctx.var_registry().clone();
    let false_state = ArrayState::from_state(
        &State::from_pairs([("x", Value::int(1)), ("y", Value::int(0))]),
        &registry,
    );

    assert!(!mc.check_state_constraints_array(&false_state).unwrap());
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 1);
    assert_eq!(
        mc.state_constraint_verdict_cache.test_inline_entry_count(),
        0
    );
    let hits_before = mc.state_constraint_verdict_cache.test_hit_count();
    assert!(!mc.check_state_constraints_array(&false_state).unwrap());
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 1);
    assert_eq!(
        mc.state_constraint_verdict_cache.test_hit_count() - hits_before,
        1,
        "the exact repeated FALSE union projection should be reused"
    );

    let short_circuit_false = ArrayState::from_state(
        &State::from_pairs([("x", Value::int(0)), ("y", Value::int(1))]),
        &registry,
    );
    assert!(!mc
        .check_state_constraints_array(&short_circuit_false)
        .unwrap());
    assert!(!mc
        .check_state_constraints_array(&short_circuit_false)
        .unwrap());
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 2);

    let non_boolean = ArrayState::from_state(
        &State::from_pairs([("x", Value::int(1)), ("y", Value::int(1))]),
        &registry,
    );
    let hits_before_error = mc.state_constraint_verdict_cache.test_hit_count();
    assert!(mc.check_state_constraints_array(&non_boolean).is_err());
    assert!(mc.check_state_constraints_array(&non_boolean).is_err());
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 2);
    assert_eq!(
        mc.state_constraint_verdict_cache.test_hit_count(),
        hits_before_error,
        "errors must execute canonically on every check"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn state_constraint_cache_handles_empty_and_compound_projections_exactly() {
    let module = parse_module(
        r#"
---- MODULE ConstraintVerdictProjectionShapes ----
EXTENDS Integers, Sequences
VARIABLES data, irrelevant
Init == /\ data = <<1, 2>> /\ irrelevant = 0
Next == UNCHANGED <<data, irrelevant>>
Always == TRUE
Never == FALSE
DataReflexive == data = data
====
"#,
    );
    let registry_config = |constraints| Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints,
        ..Default::default()
    };

    let config = registry_config(vec!["Always".to_string()]);
    let mut constant = ModelChecker::new(&module, &config);
    constant.tir_parity = None;
    let registry = constant.ctx.var_registry().clone();
    let first = ArrayState::from_state(
        &State::from_pairs([
            ("data", Value::seq([Value::int(1), Value::int(2)])),
            ("irrelevant", Value::int(0)),
        ]),
        &registry,
    );
    let second = ArrayState::from_state(
        &State::from_pairs([
            ("data", Value::seq([Value::int(9)])),
            ("irrelevant", Value::int(1)),
        ]),
        &registry,
    );
    assert!(constant.check_state_constraints_array(&first).unwrap());
    assert!(constant.check_state_constraints_array(&second).unwrap());
    assert_eq!(constant.state_constraint_verdict_cache.test_hit_count(), 1);

    let config = registry_config(vec!["Never".to_string()]);
    let mut never = ModelChecker::new(&module, &config);
    never.tir_parity = None;
    assert!(!never.check_state_constraints_array(&first).unwrap());
    assert!(!never.check_state_constraints_array(&second).unwrap());
    assert_eq!(never.state_constraint_verdict_cache.test_entry_count(), 1);
    assert_eq!(never.state_constraint_verdict_cache.test_hit_count(), 1);

    let config = registry_config(vec!["DataReflexive".to_string()]);
    let mut compound = ModelChecker::new(&module, &config);
    compound.tir_parity = None;
    let registry = compound.ctx.var_registry().clone();
    let equal_a = ArrayState::from_state(
        &State::from_pairs([
            ("data", Value::seq([Value::int(1), Value::int(2)])),
            ("irrelevant", Value::int(0)),
        ]),
        &registry,
    );
    let equal_b = ArrayState::from_state(
        &State::from_pairs([
            ("data", Value::seq([Value::int(1), Value::int(2)])),
            ("irrelevant", Value::int(99)),
        ]),
        &registry,
    );
    assert!(compound.check_state_constraints_array(&equal_a).unwrap());
    assert!(compound.check_state_constraints_array(&equal_b).unwrap());
    assert_eq!(compound.state_constraint_verdict_cache.test_hit_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn state_constraint_cache_rejects_contextual_union_and_observable_backends() {
    let contextual_module = parse_module(
        r#"
---- MODULE ConstraintVerdictContextual ----
EXTENDS Integers, TLC
VARIABLE x
Init == x = 0
Next == x' = x
Pure == x >= 0
Contextual == TLCGet("level") >= 0
====
"#,
    );
    let contextual_config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["Pure".to_string(), "Contextual".to_string()],
        ..Default::default()
    };
    let contextual = ModelChecker::new(&contextual_module, &contextual_config);
    assert!(!contextual.state_constraint_verdict_cache.test_is_enabled());

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["Pure".to_string()],
        jit_verify: true,
        ..Default::default()
    };
    let mut verified = ModelChecker::new(&contextual_module, &config);
    verified.tir_parity = None;
    let registry = verified.ctx.var_registry().clone();
    let state = ArrayState::from_state(&State::from_pairs([("x", Value::int(0))]), &registry);
    assert!(verified.check_state_constraints_array(&state).unwrap());
    assert!(verified.check_state_constraints_array(&state).unwrap());
    assert_eq!(
        verified.state_constraint_verdict_cache.test_entry_count(),
        0
    );
    assert_eq!(verified.state_constraint_verdict_cache.test_hit_count(), 0);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["Pure".to_string()],
        ..Default::default()
    };
    let mut tir = ModelChecker::new(&contextual_module, &config);
    tir.constraint_bytecode = None;
    tir.tir_parity = Some(
        super::super::super::tir_parity::TirParityState::test_eval_selected(
            contextual_module.clone(),
            Vec::new(),
            ["Pure"],
        ),
    );
    let registry = tir.ctx.var_registry().clone();
    let state = ArrayState::from_state(&State::from_pairs([("x", Value::int(0))]), &registry);
    assert!(tir.check_state_constraints_array(&state).unwrap());
    assert!(tir.check_state_constraints_array(&state).unwrap());
    assert_eq!(tir.state_constraint_verdict_cache.test_entry_count(), 0);
    assert_eq!(tir.state_constraint_verdict_cache.test_hit_count(), 0);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn state_constraint_cache_retires_on_the_boundary_without_hiding_false() {
    let module = parse_module(
        r#"
---- MODULE ConstraintVerdictRetire ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x
NonNegative == x >= 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constraints: vec!["NonNegative".to_string()],
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.tir_parity = None;
    let registry = mc.ctx.var_registry().clone();

    for value in 0..1_024 - 1 {
        let state = ArrayState::from_state(
            &State::from_pairs([("x", Value::int(value as i64))]),
            &registry,
        );
        assert!(mc.check_state_constraints_array(&state).unwrap());
    }
    assert!(mc.state_constraint_verdict_cache.test_is_enabled());

    let retiring_false =
        ArrayState::from_state(&State::from_pairs([("x", Value::int(-1))]), &registry);
    assert!(!mc.check_state_constraints_array(&retiring_false).unwrap());
    assert!(!mc.state_constraint_verdict_cache.test_is_enabled());
    assert_eq!(mc.state_constraint_verdict_cache.test_entry_count(), 0);
}
