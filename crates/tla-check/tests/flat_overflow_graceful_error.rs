// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Graceful flat-storage value-overflow handling (library level).
//!
//! An all-scalar spec is admitted to flat-primary storage (the init state
//! roundtrips fine), then mid-run produces a scalar integer beyond i64 that
//! the fixed flat i64 layout cannot represent.
//!
//! Contract under test (graceful flat overflow):
//! - The checker must NEVER panic ("invalid flat state serialization") or
//!   abort. Every flat-primary path either routes the state to a sound
//!   non-flat fallback (correct verdict directly) or surfaces the typed,
//!   catchable `RuntimeCheckError::FlatLayoutUnsupportedValue` error.
//! - Re-running with `Config::use_flat_state = Some(false)` (the programmatic
//!   flat disable the CLI retry uses) always produces the correct verdict.

mod common;

#[cfg(feature = "testing")]
use tla_check::ModelChecker;
use tla_check::{CheckResult, Config};
use tla_eval::clear_for_test_reset;

/// x: 2 -> 4 -> 16 -> 256 -> 65536 -> 2^32 -> 2^64 -> 2^128 -> (self-loop).
/// 2^64 > i64::MAX is the first state the flat layout cannot encode; the
/// guard literal 10^20 keeps the action on the interpreter (untranslatable
/// to native i64), so the overflow arrives as an interpreter BigInt.
const OVERFLOW_SPEC: &str = r#"
---- MODULE FlatOverflowLib ----
EXTENDS Integers
VARIABLE x
Init == x = 2
Next == x' = IF x < 100000000000000000000 THEN x * x ELSE x
Inv == x # 0
====
"#;

const OVERFLOW_CFG: &str = "INIT Init\nNEXT Next\nINVARIANT Inv\n";

const EXPECTED_STATES: usize = 8;

#[cfg(feature = "testing")]
fn run_overflow_spec(use_flat_state: Option<bool>) -> CheckResult {
    // Sanitize flat-related env so auto-detection is in its default state.
    let _no_flat = common::EnvVarGuard::remove("TY_NO_FLAT_BFS");
    let _no_compiled = common::EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    // Keep the flat-primary per-action successor path engaged (POR routes
    // successor generation through the interpreter batch dispatcher instead).
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    clear_for_test_reset();

    let module = common::parse_module(OVERFLOW_SPEC);
    let mut config = Config::parse(OVERFLOW_CFG).expect("valid cfg");
    config.use_flat_state = use_flat_state;
    let mut checker = ModelChecker::new(&module, &config);
    checker.check()
}

/// The overflow run must complete without panicking. Two sound outcomes are
/// permitted, and anything else is a regression:
/// - `Success` with the exact state count (a flat path routed the
///   unencodable states to its non-flat fallback), or
/// - the typed `FlatLayoutUnsupportedValue` error (a flat-primary path that
///   has no per-state fallback fails closed), which the CLI catches to
///   re-run without flat storage.
#[cfg(feature = "testing")]
#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn scalar_overflow_with_flat_auto_detection_is_graceful() {
    let result = run_overflow_spec(None);
    match result {
        CheckResult::Success(ref stats) => {
            assert_eq!(
                stats.states_found, EXPECTED_STATES,
                "sound fallback must still find the exact state count"
            );
        }
        CheckResult::Error { ref error, .. } => {
            let detail = error
                .flat_layout_unsupported_detail()
                .unwrap_or_else(|| panic!("expected FlatLayoutUnsupportedValue, got: {error}"));
            assert!(
                detail.contains("i64"),
                "detail should describe the unrepresentable value: {detail}"
            );
        }
        other => panic!("expected Success or the typed flat-layout error, got {other:?}"),
    }
}

/// The CLI retry recipe: `use_flat_state = Some(false)` disables every flat
/// path (checked before the env var and before auto-detection), and the
/// re-run yields the correct verdict with the exact state count.
#[cfg(feature = "testing")]
#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn scalar_overflow_retry_without_flat_reaches_correct_verdict() {
    let result = run_overflow_spec(Some(false));
    match result {
        CheckResult::Success(ref stats) => {
            assert_eq!(stats.states_found, EXPECTED_STATES);
            assert_eq!(stats.initial_states, 1);
        }
        other => panic!("expected Success with flat disabled, got {other:?}"),
    }
}

/// Forcing flat on (`use_flat_state = Some(true)`) must also stay graceful:
/// sound-fallback Success or the typed error — never a panic.
#[cfg(feature = "testing")]
#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn scalar_overflow_with_flat_forced_is_graceful() {
    let result = run_overflow_spec(Some(true));
    match result {
        CheckResult::Success(ref stats) => {
            assert_eq!(stats.states_found, EXPECTED_STATES);
        }
        CheckResult::Error { ref error, .. } => {
            assert!(
                error.flat_layout_unsupported_detail().is_some(),
                "expected FlatLayoutUnsupportedValue, got: {error}"
            );
        }
        other => panic!("expected Success or the typed flat-layout error, got {other:?}"),
    }
}
