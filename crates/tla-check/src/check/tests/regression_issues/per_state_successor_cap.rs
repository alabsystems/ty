// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Regression test for the per-state successor materialization cap
//! (resource/soundness audit finding #12).
//!
//! The batch successor path collects the ENTIRE per-state successor set into one
//! `Vec<DiffSuccessor>` before any of it is processed. A pathological or
//! misconfigured action can make that Vec grow without bound and OOM-kill the
//! whole checker — a hard crash, not a verdict.
//!
//! The fix caps how many successors a single state may materialize on the batch
//! path. When the cap is exceeded the enumerator stops fail-closed and the run
//! terminates with `EvalError::SetTooLarge` (surfaced as
//! `CheckError::Eval(EvalCheckError::Eval(_))`) — never a panic, OOM, or a wrong
//! (under-counted) verdict.

use super::*;
use crate::{EvalCheckError, EvalError};
use tla_core::{lower, parse_to_syntax_tree, FileId};

/// Run the checker with an explicit per-state successor cap.
///
/// `cap` is written directly to the per-checker `Config::per_state_successor_cap`:
/// `None` uses the env/default cap, `Some(None)` disables it, `Some(Some(n))`
/// caps at `n`. Because the cap lives on this checker's own `Config` (never a
/// process-global), each call is fully isolated from any concurrently-running
/// checker — no quiescence lock or wall-clock timeout is needed.
// The `Option<Option<usize>>` mirrors `Config::per_state_successor_cap` exactly:
// the three states (default / disabled / capped-at-n) are load-bearing here.
#[allow(clippy::option_option)]
fn run_checker(src: &str, cap: Option<Option<usize>>) -> CheckResult {
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    assert!(
        lower_result.errors.is_empty(),
        "Errors: {:?}",
        lower_result.errors
    );
    let module = lower_result.module.expect("module should lower");

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        per_state_successor_cap: cap,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.check()
}

/// A single state whose `Next` action fans out to 50 successors. The exact count
/// is well-defined so the control phases can assert the cap does not false-trip.
const FANOUT_SPEC: &str = r#"
---- MODULE PerStateFanout ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' \in 1..50
====
"#;

/// The three phases each build their own checker with a distinct per-state
/// successor cap (fail-closed / no-false-trip / opt-out). Caps are per-checker
/// (carried on each `Config`), so the phases are independent and cannot leak
/// into — or be perturbed by — any concurrently-running checker. No global
/// quiescence lock and no wall-clock timeout are needed: the work is a fixed,
/// tiny state space and the assertions below fully establish correctness.
#[test]
fn test_finding12_per_state_successor_cap_fail_closed() {
    // ── Phase 1: fail-closed ─────────────────────────────────────────────────
    // Cap at 8 successors; the initial state alone produces 50. The batch path
    // must bail out fail-closed rather than panic, OOM, or under-count.
    match run_checker(FANOUT_SPEC, Some(Some(8))) {
        CheckResult::Error {
            error: CheckError::Eval(EvalCheckError::Eval(EvalError::SetTooLarge { .. })),
            ..
        } => {
            // Correct: one state's successor enumeration exceeded the cap and
            // bailed out fail-closed. No panic, no OOM, no wrong verdict.
        }
        CheckResult::Error { error, .. } => panic!(
            "finding #12: expected SetTooLarge (fail-closed cap), got a different \
             error: {error:?}"
        ),
        CheckResult::Success(_) => panic!(
            "finding #12: a state producing 50 successors under an 8-successor cap \
             must fail closed, but the checker reported success (the per-state \
             successor set was silently truncated — a wrong verdict)."
        ),
        other => panic!("finding #12: expected Error, got: {other:?}"),
    }

    // ── Phase 2: no false-trip ───────────────────────────────────────────────
    // The SAME spec must verify cleanly when the cap comfortably exceeds the
    // legitimate per-state fan-out. Proves the cap is a runaway-memory guard, not
    // a behavior change for normal specs, and that the bail-out above was caused
    // by the cap (not the spec itself).
    match run_checker(FANOUT_SPEC, Some(Some(1_000_000))) {
        CheckResult::Success(_) => {}
        other => panic!(
            "finding #12 control: spec with 50 per-state successors and a 1M cap \
             should succeed, got: {other:?}"
        ),
    }

    // ── Phase 3: explicit opt-out ────────────────────────────────────────────
    // Disabling the cap (`Some(None)`) restores the previous unbounded behavior.
    match run_checker(FANOUT_SPEC, Some(None)) {
        CheckResult::Success(_) => {}
        other => {
            panic!("finding #12: with the cap disabled the spec should succeed, got: {other:?}")
        }
    }
}
