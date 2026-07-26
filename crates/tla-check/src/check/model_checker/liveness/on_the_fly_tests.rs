// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::ModelChecker;
use super::runner::exact_otf_owned_cache_admitted;
use crate::check::{resolve_spec_from_config, CheckResult};
use crate::storage::TraceLocationStorage;
use crate::{Config, FingerprintSet, FingerprintStorage, LivenessExecutionMode, Value};
use std::sync::Arc;
use tla_core::{lower, parse_to_syntax_tree, FileId};

const ON_THE_FLY_SUCCESS_SPEC: &str = r#"
---- MODULE OnTheFlyLivenessSuccess ----
EXTENDS Integers, TLC

VARIABLE x
vars == <<x>>

Init == x = 0
Inc == x < 2 /\ x' = x + 1
Next == Inc \/ UNCHANGED x
Spec == Init /\ [][Next]_vars /\ WF_vars(Inc)
EventuallyTwo == <>(x = 2)
EventuallyThree == <>(x = 3)
Post == TLCGet("stats").distinct = 3
====
"#;

const ON_THE_FLY_VIOLATION_SPEC: &str = r#"
---- MODULE OnTheFlyLivenessViolation ----
EXTENDS Integers

VARIABLE x

Init == x = 0
Next == UNCHANGED x
EventuallyOne == <>(x = 1)
====
"#;

const ON_THE_FLY_UNSUPPORTED_SPEC: &str = r#"
---- MODULE OnTheFlyUnsupported ----
EXTENDS Integers

VARIABLE x

View == x
Init == x = 0
Next == UNCHANGED x
EventuallyZero == <>(x = 0)
====
"#;

const ON_THE_FLY_VIEW_SPEC: &str = r#"
---- MODULE OnTheFlyView ----
EXTENDS Integers

VARIABLES x, y

Init == x = 0 /\ y = 0
Next == /\ x' = x
        /\ y' = 1 - y
View == <<x>>
EventuallyOne == <>(x = 1)
====
"#;

const ON_THE_FLY_SYMMETRY_SPEC: &str = r#"
---- MODULE OnTheFlySymmetry ----
EXTENDS TLC, Integers

CONSTANT Procs
VARIABLE owner

Init == owner \in Procs
Next == UNCHANGED owner
StableOwner == <>[](owner \in Procs)
Sym == Permutations(Procs)
====
"#;

const ON_THE_FLY_MIXED_SAFETY_SPEC: &str = r#"
---- MODULE OnTheFlyMixedSafety ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0
Step == x < 1 /\ x' = x + 1
Next == Step \/ UNCHANGED x
InitMustBeOne == x = 1
StepMustSkip == [][x' = x + 2]_vars
MixedInitAndLive == (x = 1) /\ <>(x = 1)
====
"#;

const ON_THE_FLY_CROSS_GROUP_ROOTS_SPEC: &str = r#"
---- MODULE OnTheFlyCrossGroupRoots ----
EXTENDS Integers

VARIABLE x

Init == x \in {0, 1}
Next == UNCHANGED x
First == <>(x = 0 \/ x = 1)
Second == []<>(x = 0 \/ x = 1)
====
"#;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_liveness_succeeds_without_cached_successor_graph() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_SUCCESS_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let spec_config = Config {
        specification: Some("Spec".to_string()),
        ..Default::default()
    };
    let resolved = resolve_spec_from_config(&spec_config, &tree).expect("SPECIFICATION resolves");
    let config = Config {
        init: Some(resolved.init.clone()),
        next: Some(resolved.next.clone()),
        properties: vec!["EventuallyTwo".to_string()],
        postcondition: Some("Post".to_string()),
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_store_states(true);
    checker.set_fairness(resolved.fairness);

    let result = checker.check();
    match result {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 3, "expected x=0,1,2 to be reachable");
            assert_eq!(
                stats.storage_stats.memory_count, 3,
                "result storage stats must preserve the pre-release snapshot"
            );
        }
        other => panic!("expected success from on-the-fly liveness, got {other:?}"),
    }

    assert_eq!(
        checker.liveness_cache.successors.len(),
        0,
        "on-the-fly mode must not retain the BFS successor graph"
    );
    assert_eq!(
        checker.liveness_cache.successors.total_successors(),
        0,
        "on-the-fly mode must not record any cached liveness edges"
    );
    assert_eq!(
        checker.liveness_cache.init_states.len(),
        1,
        "compact initial roots must be restored after successful liveness checking"
    );
    assert!(
        checker.test_seen_is_empty(),
        "exact on-the-fly checking must release redundant full-state witnesses"
    );
    assert_eq!(
        checker.trace.trace_locs.len(),
        0,
        "exact on-the-fly checking must release the BFS trace-location index"
    );
    assert!(
        checker.trace.lazy_trace_index,
        "the retained trace file must remain available for cold index rebuilding"
    );
    assert_eq!(
        checker.test_seen_fps_len(),
        3,
        "logical fingerprint count must survive terminal membership release"
    );
    assert_eq!(checker.test_active_seen_fps_len(), 0);
    assert_eq!(checker.test_retired_seen_fps_len(), Some(3));

    let cached_stats = checker.stats.clone();
    match checker.with_current_storage_stats(CheckResult::Success(cached_stats)) {
        CheckResult::Success(stats) => assert_eq!(stats.storage_stats.memory_count, 3),
        other => panic!("expected rebound success stats, got {other:?}"),
    }
}

#[test]
fn exact_otf_owned_cache_admission_fails_closed() {
    let admitted = |partial_graph,
                    on_the_fly,
                    has_view,
                    has_symmetry,
                    cache_enabled,
                    init_count,
                    expected_init_count,
                    states_found| {
        exact_otf_owned_cache_admitted(
            partial_graph,
            on_the_fly,
            has_view,
            has_symmetry,
            cache_enabled,
            init_count,
            expected_init_count,
            states_found,
        )
    };

    assert!(admitted(false, true, false, false, true, 1, 1, 3));
    assert!(admitted(false, true, false, false, true, 0, 0, 0));
    assert!(!admitted(true, true, false, false, true, 1, 1, 3));
    assert!(!admitted(false, false, false, false, true, 1, 1, 3));
    assert!(!admitted(false, true, true, false, true, 1, 1, 3));
    assert!(!admitted(false, true, false, true, true, 1, 1, 3));
    assert!(!admitted(false, true, false, false, false, 1, 1, 3));
    assert!(!admitted(false, true, false, false, true, 1, 2, 3));
    assert!(!admitted(false, true, false, false, true, 1, 1, 0));
    assert!(!admitted(false, true, false, false, true, 2, 2, 1));
    assert!(!admitted(false, true, false, false, true, 0, 0, 3));
}

#[test]
fn liveness_terminal_fingerprint_release_requires_exact_count_and_exclusive_arc() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_VIOLATION_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);

    assert_eq!(
        checker
            .state_storage
            .try_release_terminal_seen_fps_entries(1),
        0
    );
    assert_eq!(checker.test_retired_seen_fps_len(), None);

    let weak_alias = Arc::downgrade(&checker.state_storage.seen_fps);
    assert_eq!(
        checker
            .state_storage
            .try_release_terminal_seen_fps_entries(0),
        0
    );
    assert_eq!(checker.test_retired_seen_fps_len(), None);

    drop(weak_alias);
    assert_eq!(
        checker
            .state_storage
            .try_release_terminal_seen_fps_entries(0),
        0
    );
    assert_eq!(checker.test_retired_seen_fps_len(), Some(0));

    let restored_fp = crate::Fingerprint(42);
    let mut checkpoint = crate::checkpoint::Checkpoint::new();
    checkpoint.fingerprints.push(restored_fp);
    checkpoint.depths.insert(restored_fp, 0);
    checkpoint.metadata.stats.states_found = 1;
    let frontier = checker
        .restore_from_checkpoint(checkpoint)
        .expect("retired membership backend should refresh before restore");
    assert!(frontier.is_empty());
    assert_eq!(checker.test_retired_seen_fps_len(), None);
    assert_eq!(checker.test_active_seen_fps_len(), 1);
    assert_eq!(checker.test_seen_fps_len(), 1);

    assert_eq!(
        checker
            .state_storage
            .try_release_terminal_seen_fps_entries(1),
        1
    );
    assert_eq!(checker.test_retired_seen_fps_len(), Some(1));
    checker.set_fingerprint_storage(Arc::new(FingerprintStorage::in_memory()));
    assert_eq!(checker.test_retired_seen_fps_len(), None);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_liveness_reports_violation_in_fp_only_mode() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_VIOLATION_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["EventuallyOne".to_string()],
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_store_states(false);

    let result = checker.check();
    match result {
        CheckResult::LivenessViolation {
            property,
            prefix: _,
            cycle,
            ..
        } => {
            assert_eq!(property, "EventuallyOne");
            assert!(
                !cycle.is_empty(),
                "violating on-the-fly run should include a witness cycle"
            );
            for state in &cycle.states {
                assert_eq!(state.get("x"), Some(&Value::int(0)));
            }
        }
        other => panic!("expected on-the-fly liveness violation, got {other:?}"),
    }

    assert_eq!(
        checker.liveness_cache.successors.len(),
        0,
        "on-the-fly violation runs must also avoid caching the BFS successor graph"
    );
    assert_eq!(
        checker.liveness_cache.init_states.len(),
        1,
        "compact initial roots must be restored after an early violation"
    );
    assert_eq!(checker.test_seen_fps_len(), 1);
    assert_eq!(checker.test_active_seen_fps_len(), 0);
    assert_eq!(checker.test_retired_seen_fps_len(), Some(1));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn exact_otf_shared_fingerprint_backend_declines_terminal_release() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_VIOLATION_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["EventuallyOne".to_string()],
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_store_states(true);
    let storage = Arc::new(FingerprintStorage::in_memory());
    checker.set_fingerprint_storage(storage.clone() as Arc<dyn FingerprintSet>);

    assert!(matches!(
        checker.check(),
        CheckResult::LivenessViolation { .. }
    ));
    assert_eq!(storage.len(), 1);
    assert_eq!(checker.test_seen_fps_len(), 1);
    assert_eq!(checker.test_active_seen_fps_len(), 1);
    assert_eq!(checker.test_retired_seen_fps_len(), None);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_liveness_reuses_exact_cache_across_properties_after_bfs_payload_release() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_SUCCESS_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let spec_config = Config {
        specification: Some("Spec".to_string()),
        ..Default::default()
    };
    let resolved = resolve_spec_from_config(&spec_config, &tree).expect("SPECIFICATION resolves");
    let config = Config {
        init: Some(resolved.init.clone()),
        next: Some(resolved.next.clone()),
        properties: vec!["EventuallyTwo".to_string(), "EventuallyThree".to_string()],
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_store_states(true);
    checker.set_fairness(resolved.fairness);

    match checker.check() {
        CheckResult::LivenessViolation {
            property,
            prefix,
            cycle,
            stats,
            ..
        } => {
            assert_eq!(property, "EventuallyThree");
            assert_eq!(
                stats.storage_stats.memory_count, 3,
                "violation result must preserve the pre-release storage snapshot"
            );
            assert!(
                !prefix.is_empty(),
                "violation must retain a concrete prefix"
            );
            assert!(!cycle.is_empty(), "violation must retain a concrete cycle");
            assert!(
                prefix
                    .states
                    .iter()
                    .chain(&cycle.states)
                    .all(|state| state.get("x").is_some()),
                "all counterexample states must remain materialized"
            );
            assert!(
                cycle
                    .states
                    .iter()
                    .all(|state| state.get("x") == Some(&Value::int(2))),
                "the terminal x=2 stutter must form the violating cycle"
            );
        }
        other => panic!("expected on-the-fly liveness violation, got {other:?}"),
    }

    assert!(checker.test_seen_is_empty());
    assert_eq!(checker.trace.trace_locs.len(), 0);
    assert_eq!(checker.test_seen_fps_len(), 3);
    assert_eq!(checker.test_active_seen_fps_len(), 0);
    assert_eq!(checker.test_retired_seen_fps_len(), Some(3));
    assert_eq!(checker.liveness_cache.init_states.len(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_frozen_exact_cache_accepts_roots_reached_by_a_later_group() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_CROSS_GROUP_ROOTS_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    // `First`'s negated eventual tableau rejects every Init and therefore
    // freezes an empty exact cache. `Second` plans a direct-traversal group,
    // which must extend that transferred cache with both roots rather than
    // treating the valid inserts as post-freeze mutations.
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["First".to_string(), "Second".to_string()],
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_store_states(true);

    match checker.check() {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 2,
                "both initial roots must remain reachable"
            );
        }
        other => panic!("expected cross-group on-the-fly success, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_liveness_supports_view_fingerprints() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_VIEW_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["EventuallyOne".to_string()],
        view: Some("View".to_string()),
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_store_states(true);

    match checker.check() {
        CheckResult::LivenessViolation {
            property, cycle, ..
        } => {
            assert_eq!(property, "EventuallyOne");
            assert!(
                !cycle.is_empty(),
                "VIEW on-the-fly liveness should produce a counterexample cycle"
            );
            for state in &cycle.states {
                assert_eq!(state.get("x"), Some(&Value::int(0)));
            }
        }
        other => panic!("expected VIEW on-the-fly liveness violation, got {other:?}"),
    }

    assert!(
        !checker.test_seen_is_empty(),
        "VIEW must keep BFS witnesses because exact raw ownership is unavailable"
    );
}

/// Part of #3706: Verify POR is accepted with on-the-fly liveness.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn model_checker_accepts_por_with_on_the_fly_liveness() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_UNSUPPORTED_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["EventuallyZero".to_string()],
        por_enabled: true,
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    // POR should no longer be rejected — the checker proceeds normally.
    // This spec has `UNCHANGED x` as Next, so x stays 0 forever, satisfying <>(x=0).
    match &result {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 1,
                "single-state UNCHANGED spec should have exactly 1 state"
            );
        }
        other => {
            panic!("POR with on-the-fly liveness should succeed (Part of #3706), got {other:?}")
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_liveness_supports_symmetry_configs() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_SYMMETRY_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["StableOwner".to_string()],
        symmetry: Some("Sym".to_string()),
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    match checker.check() {
        CheckResult::Success(stats) => {
            // Declared-SYMMETRY wrong-verdict fix: StableOwner is a genuine
            // temporal property, so declared SYMMETRY is ignored (the orbit
            // quotient is unsound for liveness) — expect the UNREDUCED count,
            // not 1 collapsed canonical state.
            assert_eq!(
                stats.states_found, 2,
                "declared SYMMETRY must be ignored under genuine liveness; expected the \
                 unreduced owner-state count"
            );
        }
        other => panic!("expected on-the-fly symmetry liveness success, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_safety_only_property_reports_state_level_violation() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_MIXED_SAFETY_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["InitMustBeOne".to_string()],
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    match result {
        CheckResult::PropertyViolation {
            property,
            kind,
            trace,
            ..
        } => {
            assert_eq!(property, "InitMustBeOne");
            assert_eq!(kind, crate::check::api::PropertyViolationKind::StateLevel);
            assert_eq!(
                trace.states.len(),
                1,
                "init violation should be single-state"
            );
            assert_eq!(trace.states[0].get("x"), Some(&Value::int(0)));
        }
        other => panic!("expected state-level property violation, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_safety_only_property_reports_action_level_violation() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_MIXED_SAFETY_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["StepMustSkip".to_string()],
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    match result {
        CheckResult::PropertyViolation {
            property,
            kind,
            trace,
            ..
        } => {
            assert_eq!(property, "StepMustSkip");
            assert_eq!(kind, crate::check::api::PropertyViolationKind::ActionLevel);
            assert_eq!(
                trace.states.len(),
                2,
                "action violation should include both states"
            );
            assert_eq!(trace.states[0].get("x"), Some(&Value::int(0)));
            assert_eq!(trace.states[1].get("x"), Some(&Value::int(1)));
        }
        other => panic!("expected action-level property violation, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn on_the_fly_mixed_property_checks_safety_parts_before_temporal_core() {
    let tree = parse_to_syntax_tree(ON_THE_FLY_MIXED_SAFETY_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["MixedInitAndLive".to_string()],
        liveness_execution: LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    match result {
        CheckResult::PropertyViolation {
            property,
            kind,
            trace,
            ..
        } => {
            assert_eq!(property, "MixedInitAndLive");
            assert_eq!(kind, crate::check::api::PropertyViolationKind::StateLevel);
            assert_eq!(
                trace.states.len(),
                1,
                "mixed init violation should short-circuit"
            );
            assert_eq!(trace.states[0].get("x"), Some(&Value::int(0)));
        }
        other => panic!("expected mixed state-level property violation, got {other:?}"),
    }
}
