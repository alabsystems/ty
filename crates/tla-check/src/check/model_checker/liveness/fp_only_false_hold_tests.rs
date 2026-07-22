// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Regression tests for the fingerprint-only liveness FALSE-HOLD bug
//! (#liveness-fp-only-false-hold).
//!
//! In fingerprint-only mode with the inline-bitmask fast path, the behavior
//! graph holds no concrete states. The witness re-verification gate
//! (#liveness-wf) used to treat "concrete state unavailable" as a refuted
//! witness, silently converting EVERY genuine liveness violation in this mode
//! into a HOLD — a missed-violation soundness bug. TLC reports VIOLATED for
//! the specs below; TY must agree.
//!
//! The fix: the gate reports `LivenessResult::CandidateStatesUnavailable`,
//! and the sequential checker materializes the fp-only replay state cache and
//! re-runs the check so the gate can authoritatively confirm or refute.
//!
//! Both directions are covered: a genuine violation must be reported
//! (no false HOLD) and a genuinely-holding property must stay holding
//! (no false VIOLATION after the retry).

use super::*;
use crate::config::Config;
use crate::{resolve_spec_from_config, CheckResult};
use tla_core::{lower, parse_to_syntax_tree, FileId};

/// `<>[](x = 0)` over a fair 3-cycle: the fair run cycles 0→1→2→0 forever, so
/// `x = 0` never becomes permanent. TLC: VIOLATED.
const FAIR_CYCLE_SPEC: &str = r#"
---- MODULE FairCycle ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Inc == x' = (x + 1) % 3
Spec == Init /\ [][Inc]_x /\ WF_x(Inc)
Prop == <>[](x = 0)
====
"#;

/// `[]<>(x = 5)` over the same fair 3-cycle: x never reaches 5 and the cycle
/// is weakly fair (Inc fires every step). TLC: VIOLATED.
const FAIR_CYCLE_AE_SPEC: &str = r#"
---- MODULE FairCycleAe ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Inc == x' = (x + 1) % 3
Spec == Init /\ [][Inc]_x /\ WF_x(Inc)
Prop == []<>(x = 5)
====
"#;

/// `<>[](x = 3)` over a saturating counter with WF: every fair run reaches and
/// stays at 3. TLC: HOLDS. Guards the opposite direction — the materialization
/// retry must not start accepting unfair witnesses.
const SATURATING_HOLD_SPEC: &str = r#"
---- MODULE SaturatingHold ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Inc == x' = IF x < 3 THEN x + 1 ELSE x
Spec == Init /\ [][Inc]_x /\ WF_x(Inc)
Prop == <>[](x = 3)
====
"#;

/// Run a SPECIFICATION/PROPERTY spec through the sequential checker in
/// fingerprint-only mode (`store_states = false`) and return the result.
fn run_fp_only_property_check(spec_source: &str, property: &str) -> CheckResult {
    run_fp_only_property_check_with_post_lower_hook(spec_source, property, || {})
}

fn run_fp_only_property_check_with_post_lower_hook(
    spec_source: &str,
    property: &str,
    post_lower_hook: impl FnOnce(),
) -> CheckResult {
    // Parsing interns names before `ModelChecker::check()` installs its nested
    // execution guard. Keep the caller-owned semantic input protected across
    // the whole parse/lower/check window so a concurrent run boundary cannot
    // invalidate the module's live NameIds.
    let _model_check_context = crate::enter_model_check_context();
    let tree = parse_to_syntax_tree(spec_source);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    post_lower_hook();
    let unresolved = Config {
        specification: Some("Spec".to_string()),
        ..Default::default()
    };
    let resolved =
        resolve_spec_from_config(&unresolved, &tree).expect("SPECIFICATION should resolve");
    let config = Config {
        init: Some(resolved.init.clone()),
        next: Some(resolved.next.clone()),
        specification: unresolved.specification.clone(),
        properties: vec![property.to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    // Fingerprint-only liveness: no full state retention during BFS.
    checker.set_store_states(false);
    checker.set_fairness(resolved.fairness);
    checker.set_stuttering_allowed(resolved.stuttering_allowed);
    checker.check()
}

#[test]
fn fp_only_property_helper_survives_concurrent_reset_after_lowering() {
    use std::sync::mpsc::sync_channel;
    use std::time::Duration;

    let (lowered_tx, lowered_rx) = sync_channel(0);
    let (reset_done_tx, reset_done_rx) = sync_channel(0);

    let result = std::thread::scope(|scope| {
        scope.spawn(move || {
            lowered_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("model should reach the post-lower reset window");
            crate::reset_global_state();
            reset_done_tx
                .send(())
                .expect("property checker should remain alive after reset");
        });

        run_fp_only_property_check_with_post_lower_hook(FAIR_CYCLE_SPEC, "Prop", || {
            lowered_tx
                .send(())
                .expect("reset worker should wait for the live lowered model");
            reset_done_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("guarded concurrent reset should complete");
        })
    });

    assert!(
        matches!(result, CheckResult::LivenessViolation { .. }),
        "a reset concurrent with a live lowered model must not turn a violation into HOLD: {result:?}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn fp_only_reports_violation_for_ea_property_over_fair_cycle() {
    match run_fp_only_property_check(FAIR_CYCLE_SPEC, "Prop") {
        CheckResult::LivenessViolation {
            property, cycle, ..
        } => {
            assert_eq!(property, "Prop");
            assert!(
                !cycle.states.is_empty(),
                "violation must come with a concrete counterexample cycle"
            );
        }
        other => panic!(
            "expected LivenessViolation for <>[](x=0) over a fair 3-cycle \
             (TLC: VIOLATED; fp-only false-HOLD regression), got: {other:?}"
        ),
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn fp_only_reports_violation_for_ae_property_over_fair_cycle() {
    match run_fp_only_property_check(FAIR_CYCLE_AE_SPEC, "Prop") {
        CheckResult::LivenessViolation { property, .. } => assert_eq!(property, "Prop"),
        other => panic!(
            "expected LivenessViolation for []<>(x=5) over a fair 3-cycle \
             (TLC: VIOLATED; fp-only false-HOLD regression), got: {other:?}"
        ),
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn fp_only_keeps_genuinely_holding_ea_property_holding() {
    match run_fp_only_property_check(SATURATING_HOLD_SPEC, "Prop") {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 4, "expected states 0..3");
        }
        other => panic!(
            "expected Success for <>[](x=3) over a saturating fair counter \
             (TLC: HOLDS; must not regress into a false VIOLATION), got: {other:?}"
        ),
    }
}
