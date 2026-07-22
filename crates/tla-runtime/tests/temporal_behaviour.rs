// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Behavioural tests for the temporal-property checker. The in-crate unit
//! tests cover the happy paths of `Always`/`Eventually`/`LeadsTo`/`BoxAction`/
//! `AngleAction`; this file exercises the *untested* branches: the fairness
//! variants (`WeakFairness`, `StrongFairness`), the violation paths of
//! `BoxAction` and `AngleAction`, the `TemporalCheckResult` predicate methods,
//! the `Inconclusive`-on-short-trace behaviour, and the `Debug` rendering.

use tla_runtime::{TemporalCheckResult, TemporalProp};

#[derive(Clone, Debug, PartialEq, Eq)]
struct S {
    x: i64,
}

fn trace(xs: &[i64]) -> Vec<S> {
    xs.iter().map(|&x| S { x }).collect()
}

// ----- TemporalCheckResult predicates -----

#[test]
fn check_result_predicates_are_mutually_exclusive() {
    assert!(TemporalCheckResult::Satisfied.is_satisfied());
    assert!(!TemporalCheckResult::Satisfied.is_violated());

    let v = TemporalCheckResult::Violated {
        index: 3,
        reason: "boom".to_string(),
    };
    assert!(v.is_violated());
    assert!(!v.is_satisfied());

    // Inconclusive is neither satisfied nor violated.
    assert!(!TemporalCheckResult::Inconclusive.is_satisfied());
    assert!(!TemporalCheckResult::Inconclusive.is_violated());
}

// ----- Always: violation index points at the first failing state -----

#[test]
fn always_violation_reports_first_failure_index() {
    let prop = TemporalProp::Always(Box::new(|s: &S| s.x < 2));
    let result = prop.check_trace(&trace(&[0, 1, 2, 3]));
    match result {
        TemporalCheckResult::Violated { index, reason } => {
            assert_eq!(index, 2, "first state failing x<2 is at index 2");
            assert!(reason.contains("state 2"));
        }
        other => panic!("expected violation, got {other:?}"),
    }
}

// ----- Eventually: violation reason and index on never-satisfied -----

#[test]
fn eventually_violation_index_is_last_state() {
    let prop = TemporalProp::Eventually(Box::new(|s: &S| s.x == 99));
    let result = prop.check_trace(&trace(&[0, 1, 2]));
    match result {
        TemporalCheckResult::Violated { index, reason } => {
            assert_eq!(index, 2, "index is the final trace position");
            assert!(reason.contains("never satisfied"));
        }
        other => panic!("expected violation, got {other:?}"),
    }
}

// ----- LeadsTo: P at the final state with no following Q -----

#[test]
fn leads_to_violated_when_p_holds_at_end_without_q() {
    // P (x==2) holds at the last state but Q (x==9) never follows.
    let prop = TemporalProp::LeadsTo(Box::new(|s: &S| s.x == 2), Box::new(|s: &S| s.x == 9));
    let result = prop.check_trace(&trace(&[0, 1, 2]));
    match result {
        TemporalCheckResult::Violated { index, .. } => assert_eq!(index, 2),
        other => panic!("expected violation, got {other:?}"),
    }
}

#[test]
fn leads_to_satisfied_when_q_equals_p_state() {
    // Q is allowed to hold at the same index as P (j >= i).
    let prop = TemporalProp::LeadsTo(Box::new(|s: &S| s.x == 2), Box::new(|s: &S| s.x == 2));
    assert!(prop.check_trace(&trace(&[0, 1, 2])).is_satisfied());
}

// ----- BoxAction: violation when action false AND subscript changed -----

#[test]
fn box_action_violated_on_non_stutter_disallowed_step() {
    // [A]_v is violated at a transition where the action is false AND the
    // subscript fingerprint changes (i.e. it is neither an A-step nor a stutter).
    // Action: x is even. Subscript fingerprint: x>0.
    // trace [-1, 1]: action(-1)=false (odd) and subscript flips false->true,
    // so the step is neither an A-step nor a stutter -> violation at index 0.
    let prop = TemporalProp::BoxAction(
        Box::new(|s: &S| s.x % 2 == 0), // action predicate
        Box::new(|s: &S| s.x > 0),      // subscript bool fingerprint
    );
    let result = prop.check_trace(&trace(&[-1, 1]));
    match result {
        TemporalCheckResult::Violated { index, reason } => {
            assert_eq!(index, 0);
            assert!(reason.contains("subscript changed"));
        }
        other => panic!("expected violation, got {other:?}"),
    }
}

#[test]
fn box_action_satisfied_when_action_holds_even_if_subscript_changes() {
    // Action always true -> any subscript change is permitted.
    let prop = TemporalProp::BoxAction(Box::new(|_s: &S| true), Box::new(|s: &S| s.x > 0));
    assert!(prop.check_trace(&trace(&[0, 1, 0, 1])).is_satisfied());
}

#[test]
fn box_action_single_state_trace_is_vacuously_satisfied() {
    // No consecutive pairs => no constraint to check.
    let prop = TemporalProp::BoxAction(Box::new(|_s: &S| false), Box::new(|s: &S| s.x > 0));
    assert!(prop.check_trace(&trace(&[5])).is_satisfied());
}

// ----- AngleAction: violation when no transition both acts and changes -----

#[test]
fn angle_action_violated_when_no_qualifying_transition() {
    // Subscript never changes -> <<A>>_v can never hold.
    let prop = TemporalProp::AngleAction(
        Box::new(|_s: &S| true),   // action always true
        Box::new(|s: &S| s.x > 0), // subscript constant (all states have x>0)
    );
    let result = prop.check_trace(&trace(&[1, 1, 1]));
    match result {
        TemporalCheckResult::Violated { index, reason } => {
            assert_eq!(index, 0);
            assert!(reason.contains("never satisfied"));
        }
        other => panic!("expected violation, got {other:?}"),
    }
}

#[test]
fn angle_action_violated_when_change_happens_but_action_false() {
    // Subscript changes but the action predicate is false at the changing step.
    let prop = TemporalProp::AngleAction(
        Box::new(|_s: &S| false),  // action never true
        Box::new(|s: &S| s.x > 0), // subscript flips 0->1
    );
    assert!(prop.check_trace(&trace(&[0, 1])).is_violated());
}

// ----- WeakFairness -----

#[test]
fn weak_fairness_short_trace_inconclusive() {
    let prop = TemporalProp::WeakFairness(Box::new(|_s: &S| true));
    matches_inconclusive(prop.check_trace(&trace(&[0])));
}

#[test]
fn weak_fairness_enabled_everywhere_is_inconclusive() {
    // Action "enabled" (predicate true) in every state of a >=2-length trace:
    // finite traces cannot tell whether it fired -> Inconclusive.
    let prop = TemporalProp::WeakFairness(Box::new(|_s: &S| true));
    matches_inconclusive(prop.check_trace(&trace(&[0, 1, 2])));
}

#[test]
fn weak_fairness_disabled_somewhere_is_satisfied() {
    // The action is not enabled in the final suffix (disabled at the last state),
    // so weak fairness is not obligated -> Satisfied.
    let prop = TemporalProp::WeakFairness(Box::new(|s: &S| s.x < 2));
    assert!(prop.check_trace(&trace(&[0, 1, 2])).is_satisfied());
}

// ----- StrongFairness -----

#[test]
fn strong_fairness_short_trace_inconclusive() {
    let prop = TemporalProp::StrongFairness(Box::new(|_s: &S| true));
    matches_inconclusive(prop.check_trace(&trace(&[0])));
}

#[test]
fn strong_fairness_enabled_everywhere_is_inconclusive() {
    let prop = TemporalProp::StrongFairness(Box::new(|_s: &S| true));
    matches_inconclusive(prop.check_trace(&trace(&[0, 1, 2, 3])));
}

#[test]
fn strong_fairness_intermittently_enabled_is_satisfied() {
    // Enabled in only some states (not all) -> Satisfied under the finite-trace
    // approximation.
    let prop = TemporalProp::StrongFairness(Box::new(|s: &S| s.x % 2 == 0));
    assert!(prop.check_trace(&trace(&[0, 1, 2, 3])).is_satisfied());
}

// ----- empty trace is always Inconclusive -----

#[test]
fn empty_trace_is_inconclusive_for_every_variant() {
    let empty: Vec<S> = vec![];
    matches_inconclusive(TemporalProp::Always(Box::new(|_s: &S| true)).check_trace(&empty));
    matches_inconclusive(TemporalProp::Eventually(Box::new(|_s: &S| true)).check_trace(&empty));
    matches_inconclusive(
        TemporalProp::LeadsTo(Box::new(|_s: &S| true), Box::new(|_s: &S| true)).check_trace(&empty),
    );
    matches_inconclusive(TemporalProp::WeakFairness(Box::new(|_s: &S| true)).check_trace(&empty));
    matches_inconclusive(TemporalProp::StrongFairness(Box::new(|_s: &S| true)).check_trace(&empty));
    matches_inconclusive(
        TemporalProp::BoxAction(Box::new(|_s: &S| true), Box::new(|_s: &S| true))
            .check_trace(&empty),
    );
    matches_inconclusive(
        TemporalProp::AngleAction(Box::new(|_s: &S| true), Box::new(|_s: &S| true))
            .check_trace(&empty),
    );
}

// ----- Debug rendering of the formula structure -----

#[test]
fn temporal_prop_debug_labels() {
    let f = || Box::new(|_s: &S| true) as Box<dyn Fn(&S) -> bool + Send + Sync>;
    assert_eq!(format!("{:?}", TemporalProp::Always(f())), "[]P");
    assert_eq!(format!("{:?}", TemporalProp::Eventually(f())), "<>P");
    assert_eq!(format!("{:?}", TemporalProp::LeadsTo(f(), f())), "P ~> Q");
    assert_eq!(format!("{:?}", TemporalProp::WeakFairness(f())), "WF");
    assert_eq!(format!("{:?}", TemporalProp::StrongFairness(f())), "SF");
    assert_eq!(format!("{:?}", TemporalProp::BoxAction(f(), f())), "[A]_v");
    assert_eq!(
        format!("{:?}", TemporalProp::AngleAction(f(), f())),
        "<<A>>_v"
    );
}

fn matches_inconclusive(r: TemporalCheckResult) {
    match r {
        TemporalCheckResult::Inconclusive => {}
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}
