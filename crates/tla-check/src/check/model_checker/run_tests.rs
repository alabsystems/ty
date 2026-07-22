// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for BFS setup behavior in `run.rs`.

use super::*;
use crate::config::Config;
use crate::state::{ArrayState, State};
use crate::test_support::parse_module;
use tla_value::Value;

#[cfg(feature = "ay")]
#[test]
fn explicit_coverage_vetoes_inductive_safety_certificate_route() {
    assert!(inductive_safety_certificate_route_allowed(
        false, false, false
    ));
    assert!(inductive_safety_certificate_route_allowed(
        false, true, true
    ));
    assert!(!inductive_safety_certificate_route_allowed(
        false, true, false
    ));
    assert!(!inductive_safety_certificate_route_allowed(
        true, false, false
    ));
}

/// Part of #3449: POR setup must include config-level INVARIANT variable reads
/// in the visibility set so that C3 does not reduce away invariant-relevant actions.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn setup_actions_and_por_includes_config_invariant_reads_in_visibility_set() {
    let module = parse_module(
        r#"
---- MODULE PorSetupVisibility ----
EXTENDS Naturals

VARIABLE x, y

Inv == y > 0

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ y' = y

IncY ==
    /\ y < 2
    /\ x' = x
    /\ y' = y + 1

Init == x = 0 /\ y = 1
Next == IncX \/ IncY
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        por_enabled: true,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    let next_name = checker
        .prepare_bfs_common()
        .expect("prepare_bfs_common should succeed for POR visibility test");

    checker.setup_actions_and_por(&next_name);

    let y_var = checker
        .ctx
        .var_registry()
        .get("y")
        .expect("test module should register state variable y");

    assert!(
        checker.por.independence.is_some(),
        "POR setup should compute independence when por_enabled is true",
    );
    assert!(
        checker.por.visibility.contains_var(&y_var),
        "visibility set should include y from config INVARIANT Inv == y > 0",
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn generate_successors_filtered_array_routes_por_through_per_action_dispatch() {
    let module = parse_module(
        r#"
---- MODULE PorArrayDispatch ----
EXTENDS Naturals

VARIABLE x, y

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ y' = y

IncY ==
    /\ y < 2
    /\ x' = x
    /\ y' = y + 1

Init == x = 0 /\ y = 0
Next == IncX \/ IncY
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        por_enabled: true,
        check_deadlock: false,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    let next_name = checker
        .prepare_bfs_common()
        .expect("prepare_bfs_common should succeed for POR array dispatch test");
    checker.trace.cached_next_name = Some(next_name.clone());
    checker.setup_actions_and_por(&next_name);

    let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(0))]);
    let current_array = ArrayState::from_state(&current_state, checker.ctx.var_registry());
    let result = checker
        .generate_successors_filtered_array(&current_array)
        .expect("POR array dispatch should enumerate successors");

    assert!(
        !result.successors.is_empty(),
        "test setup should produce enabled actions",
    );
    assert_eq!(
        checker.por.stats.total_states, 1,
        "POR-enabled array successor generation should record an ample-set decision",
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn generate_successors_as_diffs_falls_back_when_por_enabled() {
    let module = parse_module(
        r#"
---- MODULE PorDiffFallback ----
EXTENDS Naturals

VARIABLE x, y

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ y' = y

IncY ==
    /\ y < 2
    /\ x' = x
    /\ y' = y + 1

Init == x = 0 /\ y = 0
Next == IncX \/ IncY
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        por_enabled: true,
        check_deadlock: false,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    let next_name = checker
        .prepare_bfs_common()
        .expect("prepare_bfs_common should succeed for POR diff fallback test");
    checker.trace.cached_next_name = Some(next_name.clone());
    checker.setup_actions_and_por(&next_name);

    let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(0))]);
    let current_array = ArrayState::from_state(&current_state, checker.ctx.var_registry());

    assert!(
        checker
            .generate_successors_as_diffs(&current_array)
            .expect("POR diff fallback should not error")
            .is_none(),
        "POR should bypass the diff path so BFS can use the per-action reducer",
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn coverage_collection_routes_bfs_through_per_action_batch_consumer() {
    let module = parse_module(
        r#"
---- MODULE CoverageDiffBypass ----
EXTENDS Naturals

VARIABLE x

A ==
    /\ x = 0
    /\ x' = 1

B ==
    /\ x = 1
    /\ x' = 2

Init == x = 0
Next == A \/ B
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        check_deadlock: false,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_collect_coverage(true);

    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("coverage routing test should succeed, got {other:?}"),
    };
    let coverage = stats
        .coverage
        .expect("coverage stats should be present when collection is enabled");

    let action_transitions = |name: &str| {
        coverage
            .actions
            .values()
            .find(|action| action.name == name)
            .map(|action| action.transitions)
            .unwrap_or(0)
    };
    assert_eq!(
        action_transitions("A"),
        1,
        "coverage collection must bypass the monolithic diff path and record action A"
    );
    assert_eq!(
        action_transitions("B"),
        1,
        "coverage collection must bypass the monolithic diff path and record action B"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn prepare_bfs_common_disables_liveness_cache_for_fully_promoted_property() {
    let module = parse_module(
        r#"
---- MODULE FullyPromotedPropertyCache ----
EXTENDS Naturals

VARIABLE b
vars == <<b>>

Init == b = 0
Next ==
    /\ b < 1
    /\ b' = b + 1

Spec == Init /\ [][Next]_vars
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Spec".to_string()],
        check_deadlock: false,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker
        .prepare_bfs_common()
        .expect("prepare_bfs_common should succeed for fully promoted property");

    assert!(
        checker.is_property_fully_promoted("Spec"),
        "Spec should be fully handled by init/action BFS checks",
    );
    assert!(
        !checker.has_residual_liveness_properties(),
        "fully promoted properties should leave no residual liveness work",
    );
    assert!(
        !checker.liveness_cache.cache_for_liveness,
        "successor liveness cache should be disabled without residual liveness",
    );
    assert!(
        !checker.liveness_mode.is_active(),
        "liveness mode should be disabled without residual liveness",
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn flat_state_primary_reports_false_typeok_on_successor_in_no_trace_mode() {
    let module = parse_module(
        r#"
---- MODULE NoTraceFlatPrimaryTypeOkSuccessor ----
EXTENDS Naturals

VARIABLE x

Init == x = 0
Next == x < 1 /\ x' = x + 1
TypeOK == x \in {0}
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["TypeOK".to_string()],
        check_deadlock: false,
        use_flat_state: Some(true),
        use_compiled_bfs: Some(false),
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_store_states(false);

    match checker.check() {
        CheckResult::InvariantViolation {
            invariant, trace, ..
        } => {
            assert_eq!(invariant, "TypeOK");
            if !trace.states.is_empty() {
                assert_eq!(
                    trace.states.len(),
                    2,
                    "flat-primary no-trace violation should reconstruct init and successor"
                );
                assert_eq!(trace.states[0].get("x"), Some(&Value::int(0)));
                assert_eq!(trace.states[1].get("x"), Some(&Value::int(1)));
            }
        }
        other => panic!("expected TypeOK successor violation, got {other:?}"),
    }
    assert!(
        checker.is_flat_state_primary(),
        "scalar no-trace run should keep flat_state_primary active while reporting TypeOK"
    );
}
