// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! End-to-end PDR coverage for `\div`/`%` with a VARIABLE divisor.
//!
//! The CHC lane accepts a non-literal current-state divisor by recording a
//! `divisor > 0` side-condition that is conjoined into the safety obligation
//! (see `lower_div_mod` / `finalize_query_clauses` in tla-ay). These tests
//! pin the user-visible contract:
//! - divisor provably positive in every reachable state → PDR may prove Safe
//!   (and must never report Unsafe);
//! - divisor can reach 0 (TLC would raise a division error) → NEVER Safe.

use super::helpers::pdr_config;
use super::*;
use crate::test_support::parse_module;

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_pdr_variable_divisor_provably_positive() {
    // The divisor `window` is a state variable that invariantly stays 3, so
    // every `%` evaluation is well-defined and `q` stays within 0..3.
    let src = r#"
---- MODULE VarDivisorSafe ----
VARIABLES window, q
Init == window = 3 /\ q = 0
Next == window' = window /\ q' = (q + 1) % window
Safety == q >= 0 /\ q <= 3
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_pdr_with_config(&module, &config, &ctx, pdr_config(10, 100));
    match result {
        Ok(PdrResult::Safe { .. }) | Ok(PdrResult::Unknown { .. }) => {}
        Ok(PdrResult::Unsafe { trace }) => {
            panic!("false Unsafe on a safe variable-divisor spec: {trace:?}")
        }
        Err(e) => panic!("variable divisor must translate (not decline): {e}"),
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_pdr_variable_divisor_can_reach_zero_is_never_safe() {
    // The divisor starts at 0: TLC would raise a division error on the first
    // step. The divisor-positivity side-condition `window > 0` fails in the
    // initial state, so PDR must NOT prove the augmented property — and a
    // counterexample to it cannot be replay-confirmed as a genuine violation
    // of the original invariant (q >= 0 holds), so the result must be
    // Unknown: never a false Safe (PASS), never a division-error Unsafe.
    let src = r#"
---- MODULE VarDivisorZero ----
VARIABLES window, q
Init == window = 0 /\ q = 0
Next == window' = window /\ q' = (q + 1) % window
Safety == q >= 0
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_pdr_with_config(&module, &config, &ctx, pdr_config(10, 100));
    match result {
        Ok(PdrResult::Unknown { .. }) | Err(_) => {}
        Ok(PdrResult::Safe { invariant }) => {
            panic!("false Safe where TLC would raise a division error: {invariant}")
        }
        Ok(PdrResult::Unsafe { trace }) => {
            panic!("division-error CEX must not surface as Unsafe: {trace:?}")
        }
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_pdr_variable_divisor_genuine_violation_not_masked() {
    // Divisor stays positive (all divisions well-defined) but the invariant
    // is genuinely violated at q = 2: augmentation must not mask it as Safe.
    // Unsafe is expected (the CEX replay confirms the original violation);
    // Unknown is tolerated under solver budget variance.
    let src = r#"
---- MODULE VarDivisorViolated ----
VARIABLES window, q
Init == window = 3 /\ q = 0
Next == window' = window /\ q' = q + 1
Safety == q < 2 /\ (q % window) >= 0
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_pdr_with_config(&module, &config, &ctx, pdr_config(10, 100));
    match result {
        Ok(PdrResult::Unsafe { .. }) | Ok(PdrResult::Unknown { .. }) => {}
        Ok(PdrResult::Safe { invariant }) => {
            panic!("false Safe on a genuinely violated spec: {invariant}")
        }
        Err(e) => panic!("variable divisor must translate (not decline): {e}"),
    }
}
