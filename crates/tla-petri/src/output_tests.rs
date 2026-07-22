// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (negative tests construct the round-1 spaced literals at runtime.)

//! Tests for MCC output formatting contract.
//!
//! The MCC infrastructure parses `FORMULA <name> <verdict> TECHNIQUES <list>`
//! lines from stdout. Any formatting deviation produces invalid
//! competition output for ALL examinations. The qualification-1 (May
//! 2026) rejection was caused by emitting the spaced variant of
//! [`CANNOT_COMPUTE`](crate::mcc_keywords::CANNOT_COMPUTE) — see
//! `docs/mcc-2026/qualification-1/analysis.md`. These tests pin the
//! canonical underscored form.

use crate::output::{cannot_compute_line, formula_line, Verdict};

/// Build the round-1 spaced variants at runtime so an auto-fixer can't
/// silently rewrite the spaced literals (
/// mcc-keyword-guard: allow-spaced-mention
/// `CANNOT COMPUTE` etc.) to underscored ones and turn our negative
/// assertions into tautologies.
fn forbidden_cannot_compute_with_space() -> String {
    format!("CANNOT{}COMPUTE", " ")
}
fn forbidden_state_space_with_space() -> String {
    format!("STATE{}SPACE", " ")
}

#[test]
fn test_verdict_display_true() {
    assert_eq!(Verdict::True.to_string(), "TRUE");
}

#[test]
fn test_verdict_display_false() {
    assert_eq!(Verdict::False.to_string(), "FALSE");
}

#[test]
fn test_verdict_display_cannot_compute_is_underscored() {
    assert_eq!(Verdict::CannotCompute.to_string(), "CANNOT_COMPUTE");
    assert!(!Verdict::CannotCompute.to_string().contains(' '));
}

#[test]
fn test_formula_line_true_format() {
    let line = formula_line(
        "ModelA-PT-001",
        "ModelA-PT-001-ReachabilityFireability-00",
        Verdict::True,
    );
    assert_eq!(
        line,
        "FORMULA ModelA-PT-001-ReachabilityFireability-00 TRUE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn test_formula_line_false_format() {
    let line = formula_line(
        "ModelA-PT-001",
        "ModelA-PT-001-ReachabilityFireability-01",
        Verdict::False,
    );
    assert_eq!(
        line,
        "FORMULA ModelA-PT-001-ReachabilityFireability-01 FALSE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn test_state_space_cannot_compute_line_is_underscored() {
    let line = cannot_compute_line("ModelA-PT-001", "StateSpace");
    assert_eq!(line, "STATE_SPACE CANNOT_COMPUTE TECHNIQUES EXPLICIT");
    assert!(!line.contains(&forbidden_state_space_with_space()));
    assert!(!line.contains(&forbidden_cannot_compute_with_space()));
}

#[test]
fn test_formula_cannot_compute_line_is_underscored() {
    let line = cannot_compute_line("ModelA-PT-001", "ReachabilityDeadlock");
    assert_eq!(
        line,
        "FORMULA ReachabilityDeadlock CANNOT_COMPUTE TECHNIQUES EXPLICIT"
    );
    assert!(!line.contains(&forbidden_cannot_compute_with_space()));
}

#[test]
fn test_formula_line_contains_required_mcc_tokens() {
    let line = formula_line("X", "Y", Verdict::True);
    assert!(line.starts_with("FORMULA "));
    assert!(line.contains(" TECHNIQUES "));
}
