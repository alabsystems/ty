// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fail-closed overflow backstop for the HEURISTIC sequence-capacity flat-primary
//! admission (`TY_SEQ_HEURISTIC_CAPACITY=1`).
//!
//! A growing `v \in Seq(U)` sequence with a SMALL element universe but repeated
//! elements can reach `Len(v) > |U|`. The heuristic capacity `= |U|` is then a
//! WRONG guess for the real max length. This is the exact scenario the heuristic
//! capacity is only sound against BECAUSE of the fail-closed overflow backstop:
//! the flat write path returns `SequenceLengthExceedsCapacity` on the first
//! over-capacity state, so the checker must NEVER silently undercount.
//!
//! Contract under test:
//! - Forcing flat on (`use_flat_state = Some(true)`) with the heuristic flag on:
//!   either a sound per-state fallback that finds the EXACT state count, or the
//!   typed `FlatLayoutUnsupportedValue` error — never a wrong (under)count, never
//!   a panic.
//! - The CLI retry recipe (`use_flat_state = Some(false)`) always yields the
//!   correct verdict and the EXACT state count.

mod common;

use tla_check::{CheckResult, Config};
#[cfg(feature = "testing")]
use tla_check::ModelChecker;
use tla_eval::clear_for_test_reset;

/// `v \in Seq({7})` grows `<<>> -> <<7>> -> <<7,7>> -> <<7,7,7>>` (self-loop).
/// The checked universe is `{7}` so the heuristic capacity is `|U| = 1`; the
/// state `v = <<7,7>>` (length 2) is the first the flat layout cannot encode.
/// Elements repeat, so the duplicate-free `Len(v) <= |U|` proof does NOT fire
/// (the append reuses an in-range element) and the sequence gets the heuristic
/// bound rather than a certified one.
const OVERFLOW_SPEC: &str = r#"
---- MODULE SeqHeuristicOverflow ----
EXTENDS Integers, Sequences
VARIABLE v
U == {7}
Init == v = <<>>
Next == v' = IF Len(v) < 3 THEN Append(v, 7) ELSE v
TypeOk == v \in Seq(U)
====
"#;

const OVERFLOW_CFG: &str = "INIT Init\nNEXT Next\nINVARIANT TypeOk\n";

/// v of length 0, 1, 2, 3 => 4 distinct states.
const EXPECTED_STATES: usize = 4;

#[cfg(feature = "testing")]
fn run_overflow_spec(use_flat_state: Option<bool>) -> CheckResult {
    // Enable the heuristic capacity feature under test; sanitize other flat env
    // so auto-detection is otherwise in its default state.
    let _heur = common::EnvVarGuard::set("TY_SEQ_HEURISTIC_CAPACITY", Some("1"));
    let _no_flat = common::EnvVarGuard::remove("TY_NO_FLAT_BFS");
    let _no_compiled = common::EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    clear_for_test_reset();

    let module = common::parse_module(OVERFLOW_SPEC);
    let mut config = Config::parse(OVERFLOW_CFG).expect("valid cfg");
    config.use_flat_state = use_flat_state;
    let mut checker = ModelChecker::new(&module, &config);
    checker.check()
}

/// Forcing flat on must stay graceful: a sound per-state fallback that still
/// finds the exact count, or the typed flat-layout error — never a wrong count,
/// never a panic. A silent undercount here (e.g. `states_found < 4`) would be a
/// soundness bug: the heuristic capacity dropped a successor.
#[cfg(feature = "testing")]
#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn heuristic_capacity_overflow_with_flat_forced_is_graceful() {
    let result = run_overflow_spec(Some(true));
    match result {
        CheckResult::Success(ref stats) => {
            assert_eq!(
                stats.states_found, EXPECTED_STATES,
                "a sound fallback must still find the EXACT state count (no undercount)"
            );
        }
        CheckResult::Error { ref error, .. } => {
            assert!(
                error.flat_layout_unsupported_detail().is_some(),
                "expected the typed FlatLayoutUnsupportedValue backstop error, got: {error}"
            );
        }
        other => panic!("expected Success or the typed flat-layout error, got {other:?}"),
    }
}

/// The CLI retry recipe: `use_flat_state = Some(false)` disables every flat path,
/// and the re-run yields the correct verdict with the exact state count. This is
/// what the CLI transparently does after catching the backstop error.
#[cfg(feature = "testing")]
#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn heuristic_capacity_overflow_retry_without_flat_reaches_correct_verdict() {
    let result = run_overflow_spec(Some(false));
    match result {
        CheckResult::Success(ref stats) => {
            assert_eq!(stats.states_found, EXPECTED_STATES);
            assert_eq!(stats.initial_states, 1);
        }
        other => panic!("expected Success with flat disabled, got {other:?}"),
    }
}

/// Auto-detection (`use_flat_state = None`) must also stay graceful and never
/// undercount.
#[cfg(feature = "testing")]
#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn heuristic_capacity_overflow_with_auto_detection_is_graceful() {
    let result = run_overflow_spec(None);
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
