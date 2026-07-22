// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use tla_runtime::{
    model_check, model_check_with_invariant, range_set, tla_set, MonitoredStateMachine,
    SpecViolation, StateMachine, TlaSet,
};

#[test]
fn test_tla_set_empty_set_algebra() {
    let empty: TlaSet<i32> = tla_set![];
    let a = tla_set![1, 2];

    let union = a.union(&empty);
    assert_eq!(
        union, a,
        "union with empty set should preserve all elements"
    );

    let intersect = a.intersect(&empty);
    assert!(
        intersect.is_empty(),
        "intersection with empty set should be empty"
    );

    let difference = a.difference(&empty);
    assert_eq!(
        difference, a,
        "difference against empty set should preserve all elements"
    );

    // Symmetric cases: empty on the left
    let union_rev = empty.union(&a);
    assert_eq!(
        union_rev, a,
        "empty union non-empty should equal the non-empty set"
    );

    let intersect_rev = empty.intersect(&a);
    assert!(
        intersect_rev.is_empty(),
        "empty intersect non-empty should be empty"
    );

    let diff_rev = empty.difference(&a);
    assert!(
        diff_rev.is_empty(),
        "empty difference non-empty should be empty"
    );

    // Both empty
    let empty2: TlaSet<i32> = tla_set![];
    let union_both = empty.union(&empty2);
    assert!(union_both.is_empty(), "empty union empty should be empty");
    let intersect_both = empty.intersect(&empty2);
    assert!(
        intersect_both.is_empty(),
        "empty intersect empty should be empty"
    );
    let diff_both = empty.difference(&empty2);
    assert!(
        diff_both.is_empty(),
        "empty difference empty should be empty"
    );
}

#[test]
fn test_range_set_singleton() {
    let r = range_set(5, 5);
    assert_eq!(r.len(), 1);
    assert!(r.contains(&5));
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CounterState {
    count: i64,
}

struct Counter {
    max: i64,
}

impl StateMachine for Counter {
    type State = CounterState;

    fn init(&self) -> Vec<Self::State> {
        vec![CounterState { count: 0 }]
    }

    fn next(&self, state: &Self::State) -> Vec<Self::State> {
        if state.count < self.max {
            vec![CounterState {
                count: state.count + 1,
            }]
        } else {
            Vec::new()
        }
    }
}

#[test]
fn test_model_check_max_states_zero() {
    let machine = Counter { max: 5 };
    let result = model_check(&machine, 0);

    assert_eq!(
        result.states_explored, 1,
        "max_states=0 currently explores exactly one state before cut-off"
    );
    assert_eq!(
        result.distinct_states, 1,
        "only the initial state should be considered before cut-off"
    );
    assert!(result.violation.is_none());
    assert!(result.deadlock.is_none());
    assert!(
        !result.complete,
        "max_states=0 truncates exploration, so the run is incomplete"
    );
    assert!(
        !result.is_ok(),
        "a truncated run verified nothing and must not report ok"
    );
}

/// Regression (finding: truncated-by-max_states was indistinguishable from a
/// complete pass): a spec with MORE reachable states than `max_states` must
/// yield `complete == false` and `is_ok() == false`, while the same spec under
/// a sufficient budget is `complete == true` and `is_ok() == true`.
#[test]
fn test_model_check_truncated_run_is_not_ok() {
    // A 10-state cycle (0..=9 wrapping), no deadlock, no invariant.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct WrapState(i64);

    struct WrapCounter;
    impl StateMachine for WrapCounter {
        type State = WrapState;

        fn init(&self) -> Vec<Self::State> {
            vec![WrapState(0)]
        }

        fn next(&self, state: &Self::State) -> Vec<Self::State> {
            vec![WrapState((state.0 + 1) % 10)]
        }
    }

    // Capped below the 10 reachable states: unverified, NOT a pass.
    let capped = model_check(&WrapCounter, 3);
    assert!(capped.violation.is_none());
    assert!(capped.deadlock.is_none());
    assert!(!capped.complete, "10 states > max_states=3 must truncate");
    assert!(
        !capped.is_ok(),
        "a capped, unverified run must not report is_ok()"
    );

    // Exhaustive run: complete and ok.
    let full = model_check(&WrapCounter, 1000);
    assert!(full.complete, "10 states < max_states=1000 must complete");
    assert!(full.is_ok());
    assert_eq!(full.distinct_states, 10);

    // Same contract for the custom-invariant variant.
    let capped_inv = model_check_with_invariant(&WrapCounter, 3, |_| true);
    assert!(!capped_inv.complete);
    assert!(!capped_inv.is_ok());
    let full_inv = model_check_with_invariant(&WrapCounter, 1000, |_| true);
    assert!(full_inv.complete);
    assert!(full_inv.is_ok());

    // A violation found under a cap IS definitive: complete=true, is_ok=false.
    let violated = model_check_with_invariant(&WrapCounter, 3, |s| s.0 < 1);
    assert!(violated.violation.is_some());
    assert!(violated.complete, "a found counterexample is definitive");
    assert!(!violated.is_ok());
}

#[test]
fn test_model_check_invariant_violation_in_initial_state() {
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct InitState {
        val: i64,
    }

    struct InitViolationMachine;
    impl StateMachine for InitViolationMachine {
        type State = InitState;

        fn init(&self) -> Vec<Self::State> {
            vec![InitState { val: -1 }]
        }

        fn next(&self, state: &Self::State) -> Vec<Self::State> {
            vec![InitState { val: state.val + 1 }]
        }

        fn check_invariant(&self, state: &Self::State) -> Option<bool> {
            Some(state.val >= 0)
        }
    }

    let result = model_check(&InitViolationMachine, 10);
    assert_eq!(result.states_explored, 1);
    let violation = result
        .violation
        .as_ref()
        .expect("initial state should fail invariant");
    assert_eq!(violation.state.val, -1);
    assert!(result.deadlock.is_none());
}

#[test]
fn test_model_check_multiple_initial_states() {
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct MultiInitState(i64);

    struct MultiInitMachine;
    impl StateMachine for MultiInitMachine {
        type State = MultiInitState;

        fn init(&self) -> Vec<Self::State> {
            vec![MultiInitState(0), MultiInitState(1)]
        }

        fn next(&self, state: &Self::State) -> Vec<Self::State> {
            vec![state.clone()]
        }
    }

    let result = model_check(&MultiInitMachine, 10);
    assert!(
        result.is_ok(),
        "self-looping initial states should terminate without deadlock or invariant violation"
    );
    assert_eq!(result.distinct_states, 2);
    assert_eq!(result.states_explored, 2);
}

#[test]
fn test_monitored_state_machine_empty_init() {
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct EmptyState;

    struct EmptyInitMachine;
    impl StateMachine for EmptyInitMachine {
        type State = EmptyState;

        fn init(&self) -> Vec<Self::State> {
            Vec::new()
        }

        fn next(&self, _state: &Self::State) -> Vec<Self::State> {
            Vec::new()
        }
    }

    assert!(matches!(
        MonitoredStateMachine::new(EmptyInitMachine),
        Err(SpecViolation::EmptyInit)
    ));
}
