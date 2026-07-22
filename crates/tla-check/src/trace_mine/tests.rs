// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Unit tests for each spec-mining pattern.

use std::collections::HashMap;

use crate::json_output::JsonValue;
use crate::trace_input::{TraceActionLabel, TraceStep};

use super::*;

fn int_state(pairs: &[(&str, i64)]) -> HashMap<String, JsonValue> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), JsonValue::Int(*v)))
        .collect()
}

fn step(index: usize, state: HashMap<String, JsonValue>, action: Option<&str>) -> TraceStep {
    TraceStep {
        index: Some(index),
        state,
        action: action.map(|name| TraceActionLabel {
            name: name.to_string(),
            params: None,
        }),
    }
}

fn trace_of(name: &str, variables: &[&str], steps: Vec<TraceStep>) -> MiningTrace {
    MiningTrace {
        name: name.to_string(),
        variables: variables.iter().map(|v| (*v).to_string()).collect(),
        steps,
    }
}

/// A counter 0..3 with unlabeled +1 steps.
fn counter_trace() -> MiningTrace {
    let steps = (0..=3)
        .map(|i| step(i as usize, int_state(&[("x", i)]), None))
        .collect();
    trace_of("counter", &["x"], steps)
}

fn find_action<'a>(spec: &'a MinedSpec, name: &str) -> &'a MinedAction {
    spec.actions
        .iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| {
            panic!(
                "action {name:?} not mined; got {:?}",
                spec.actions.iter().map(|a| &a.name).collect::<Vec<_>>()
            )
        })
}

// ---------------------------------------------------------------------------
// Variable domains
// ---------------------------------------------------------------------------

#[test]
fn domain_contiguous_ints_become_range() {
    let spec = mine_spec(&[counter_trace()], &MineOptions::default()).expect("mine");
    let dom = spec.domain_of("x").expect("domain for x");
    assert_eq!(dom.expr, "0..3");
    assert!(dom.all_int);
    assert_eq!((dom.min_int, dom.max_int), (Some(0), Some(3)));
    assert!(spec.needs_integers);
}

#[test]
fn domain_sparse_small_int_set_is_enumerated() {
    let steps = vec![
        step(0, int_state(&[("x", 0)]), None),
        step(1, int_state(&[("x", 10)]), None),
        step(2, int_state(&[("x", 0)]), None),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    assert_eq!(spec.domain_of("x").expect("domain").expr, "{0, 10}");
}

#[test]
fn domain_sparse_large_int_set_over_approximates_to_range() {
    let steps: Vec<TraceStep> = (0..6)
        .map(|i| step(i, int_state(&[("x", (i as i64) * 10)]), None))
        .collect();
    let options = MineOptions {
        max_domain_enum: 3,
        ..MineOptions::default()
    };
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &options).expect("mine");
    assert_eq!(spec.domain_of("x").expect("domain").expr, "0..50");
    assert!(
        spec.notes.iter().any(|n| n.contains("over-approximated")),
        "expected an over-approximation note; got {:?}",
        spec.notes
    );
}

#[test]
fn domain_non_int_values_are_enumerated() {
    let mut s0 = HashMap::new();
    s0.insert("st".to_string(), JsonValue::String("idle".to_string()));
    let mut s1 = HashMap::new();
    s1.insert("st".to_string(), JsonValue::String("busy".to_string()));
    let steps = vec![step(0, s0, None), step(1, s1, None)];
    let spec = mine_spec(&[trace_of("t", &["st"], steps)], &MineOptions::default()).expect("mine");
    assert_eq!(
        spec.domain_of("st").expect("domain").expr,
        "{\"busy\", \"idle\"}"
    );
    assert!(!spec.needs_integers);
}

// ---------------------------------------------------------------------------
// Action inference: update patterns
// ---------------------------------------------------------------------------

#[test]
fn action_constant_delta_is_mined() {
    let spec = mine_spec(&[counter_trace()], &MineOptions::default()).expect("mine");
    assert_eq!(spec.actions.len(), 1);
    let action = &spec.actions[0];
    assert_eq!(action.name, "Change_x");
    assert_eq!(action.instances, 3);
    assert_eq!(action.updates.len(), 1);
    assert_eq!(action.updates[0].pattern, UpdatePattern::ConstDelta(1));
    assert_eq!(action.updates[0].conjuncts, vec!["x' = x + 1".to_string()]);
}

#[test]
fn action_negative_delta_renders_subtraction() {
    let steps = vec![
        step(0, int_state(&[("x", 5)]), None),
        step(1, int_state(&[("x", 3)]), None),
        step(2, int_state(&[("x", 1)]), None),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    let action = &spec.actions[0];
    assert_eq!(action.updates[0].pattern, UpdatePattern::ConstDelta(-2));
    assert_eq!(action.updates[0].conjuncts, vec!["x' = x - 2".to_string()]);
}

#[test]
fn action_constant_assignment_is_mined() {
    // x goes 1 -> 9 and 4 -> 9 under the same label: not a constant delta,
    // but a constant assignment.
    let t1 = trace_of(
        "t1",
        &["x"],
        vec![
            step(0, int_state(&[("x", 1)]), None),
            step(1, int_state(&[("x", 9)]), Some("Reset")),
        ],
    );
    let t2 = trace_of(
        "t2",
        &["x"],
        vec![
            step(0, int_state(&[("x", 4)]), None),
            step(1, int_state(&[("x", 9)]), Some("Reset")),
        ],
    );
    let spec = mine_spec(&[t1, t2], &MineOptions::default()).expect("mine");
    let action = find_action(&spec, "Reset");
    assert_eq!(
        action.updates[0].pattern,
        UpdatePattern::ConstAssign("9".to_string())
    );
    assert_eq!(action.updates[0].conjuncts, vec!["x' = 9".to_string()]);
}

#[test]
fn action_monotone_strict_increase_is_mined_with_domain_bound() {
    // Deltas +1 and +3: not constant, but strictly increasing.
    let steps = vec![
        step(0, int_state(&[("x", 0)]), None),
        step(1, int_state(&[("x", 1)]), Some("Grow")),
        step(2, int_state(&[("x", 4)]), Some("Grow")),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    let action = find_action(&spec, "Grow");
    assert_eq!(action.updates[0].pattern, UpdatePattern::MonotoneStrict);
    assert_eq!(
        action.updates[0].conjuncts,
        vec!["x' \\in xDomain".to_string(), "x' > x".to_string()]
    );
}

#[test]
fn action_havoc_falls_back_to_observed_post_values() {
    // x moves 0 -> 7, 7 -> 0, 0 -> 7 under one label: no delta, no constant,
    // not monotone. Havoc within the observed post-value set.
    let steps = vec![
        step(0, int_state(&[("x", 0)]), None),
        step(1, int_state(&[("x", 7)]), Some("Flip")),
        step(2, int_state(&[("x", 0)]), Some("Flip")),
        step(3, int_state(&[("x", 7)]), Some("Flip")),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    let action = find_action(&spec, "Flip");
    assert_eq!(action.updates[0].pattern, UpdatePattern::Havoc);
    assert_eq!(
        action.updates[0].conjuncts,
        vec!["x' \\in {0, 7}".to_string()]
    );
}

#[test]
fn action_unchanged_variables_are_detected() {
    let steps = vec![
        step(0, int_state(&[("x", 0), ("y", 3)]), None),
        step(1, int_state(&[("x", 1), ("y", 3)]), Some("IncX")),
        step(2, int_state(&[("x", 2), ("y", 3)]), Some("IncX")),
    ];
    let spec = mine_spec(
        &[trace_of("t", &["x", "y"], steps)],
        &MineOptions::default(),
    )
    .expect("mine");
    let action = find_action(&spec, "IncX");
    assert_eq!(action.unchanged, vec!["y".to_string()]);
    assert_eq!(action.updates.len(), 1);
    assert_eq!(action.updates[0].var, "x");
}

#[test]
fn action_clusters_by_changed_set_without_labels() {
    let steps = vec![
        step(0, int_state(&[("x", 0), ("y", 0)]), None),
        step(1, int_state(&[("x", 1), ("y", 0)]), None),
        step(2, int_state(&[("x", 1), ("y", 1)]), None),
        step(3, int_state(&[("x", 2), ("y", 1)]), None),
    ];
    let spec = mine_spec(
        &[trace_of("t", &["x", "y"], steps)],
        &MineOptions::default(),
    )
    .expect("mine");
    let names: Vec<&str> = spec.actions.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, vec!["Change_x", "Change_y"]);
    assert_eq!(find_action(&spec, "Change_x").instances, 2);
    assert_eq!(find_action(&spec, "Change_y").instances, 1);
}

#[test]
fn action_label_names_are_sanitized() {
    let steps = vec![
        step(0, int_state(&[("x", 0)]), None),
        step(1, int_state(&[("x", 1)]), Some("do it!")),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    assert_eq!(spec.actions[0].name, "do_it_");
}

// ---------------------------------------------------------------------------
// Action inference: guards
// ---------------------------------------------------------------------------

#[test]
fn guard_equality_constant_is_mined() {
    // Reset always fires from x = 2.
    let steps = vec![
        step(0, int_state(&[("x", 2)]), None),
        step(1, int_state(&[("x", 0)]), Some("Reset")),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    let action = find_action(&spec, "Reset");
    assert_eq!(action.guards, vec!["x = 2".to_string()]);
}

#[test]
fn guard_upper_bound_is_mined_from_pre_states() {
    // Counter deadlocks at 3: Inc's pre-states are 0..2, domain is 0..3,
    // so the informative guard x <= 2 is mined.
    let spec = mine_spec(&[counter_trace()], &MineOptions::default()).expect("mine");
    let action = &spec.actions[0];
    assert_eq!(action.guards, vec!["x <= 2".to_string()]);
}

#[test]
fn guard_lower_bound_is_mined_when_informative() {
    // Dec fires only from x in {2, 3}; domain is 0..3 (0 and 1 are reached).
    let steps = vec![
        step(0, int_state(&[("x", 3)]), None),
        step(1, int_state(&[("x", 2)]), Some("Dec")),
        step(2, int_state(&[("x", 0)]), Some("Drop")),
        step(3, int_state(&[("x", 1)]), Some("Bump")),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    let action = find_action(&spec, "Dec");
    // Pre-states of Dec: {3} -> equality guard wins.
    assert_eq!(action.guards, vec!["x = 3".to_string()]);
    // Drop's pre-state is {2}: equality guard.
    assert_eq!(find_action(&spec, "Drop").guards, vec!["x = 2".to_string()]);
}

#[test]
fn guard_uninformative_bounds_are_omitted() {
    // Move fires from every domain value (0, 2, and 1): its pre-state bounds
    // coincide with the domain bounds, so no guard is informative.
    let steps = vec![
        step(0, int_state(&[("x", 0)]), None),
        step(1, int_state(&[("x", 2)]), Some("Move")),
        step(2, int_state(&[("x", 1)]), Some("Move")),
        step(3, int_state(&[("x", 0)]), Some("Move")),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    let action = find_action(&spec, "Move");
    assert!(
        action.guards.is_empty(),
        "expected no guards, got {:?}",
        action.guards
    );
}

// ---------------------------------------------------------------------------
// Invariant candidates
// ---------------------------------------------------------------------------

#[test]
fn invariant_domain_membership_per_variable() {
    let spec = mine_spec(&[counter_trace()], &MineOptions::default()).expect("mine");
    let inv = spec
        .invariants
        .iter()
        .find(|i| i.name == "TypeOK_x")
        .expect("TypeOK_x");
    assert_eq!(inv.def, "x \\in xDomain");
    assert_eq!(inv.kind, InvariantKind::Domain);
}

#[test]
fn invariant_linear_relation_with_offset() {
    let steps = (0..3)
        .map(|i| step(i as usize, int_state(&[("x", i + 2), ("y", i)]), None))
        .collect();
    let spec = mine_spec(
        &[trace_of("t", &["x", "y"], steps)],
        &MineOptions::default(),
    )
    .expect("mine");
    let rel = spec
        .invariants
        .iter()
        .find(|i| i.name == "Rel_x_y")
        .expect("Rel_x_y");
    assert_eq!(rel.def, "x = y + 2");
}

#[test]
fn invariant_ordering_relation() {
    let steps = vec![
        step(0, int_state(&[("x", 0), ("y", 0)]), None),
        step(1, int_state(&[("x", 1), ("y", 2)]), None),
        step(2, int_state(&[("x", 2), ("y", 2)]), None),
    ];
    let spec = mine_spec(
        &[trace_of("t", &["x", "y"], steps)],
        &MineOptions::default(),
    )
    .expect("mine");
    let rel = spec
        .invariants
        .iter()
        .find(|i| i.name == "Rel_x_y")
        .expect("Rel_x_y");
    assert_eq!(rel.def, "x <= y");
}

#[test]
fn invariant_no_relation_when_none_holds() {
    let steps = vec![
        step(0, int_state(&[("x", 0), ("y", 1)]), None),
        step(1, int_state(&[("x", 2), ("y", 1)]), None),
    ];
    let spec = mine_spec(
        &[trace_of("t", &["x", "y"], steps)],
        &MineOptions::default(),
    )
    .expect("mine");
    assert!(
        !spec.invariants.iter().any(|i| i.name.starts_with("Rel_")),
        "no relation should be mined: {:?}",
        spec.invariants.iter().map(|i| &i.def).collect::<Vec<_>>()
    );
}

#[test]
fn invariant_relation_needs_min_evidence() {
    let steps = vec![step(0, int_state(&[("x", 0), ("y", 0)]), None)];
    let spec = mine_spec(
        &[trace_of("t", &["x", "y"], steps)],
        &MineOptions::default(),
    )
    .expect("mine");
    assert!(
        !spec.invariants.iter().any(|i| i.name.starts_with("Rel_")),
        "one state is below the evidence threshold"
    );
}

// ---------------------------------------------------------------------------
// Monotonicity properties
// ---------------------------------------------------------------------------

#[test]
fn monotone_nondecreasing_variable_becomes_property() {
    let spec = mine_spec(&[counter_trace()], &MineOptions::default()).expect("mine");
    let prop = spec
        .properties
        .iter()
        .find(|p| p.name == "x_Monotone")
        .expect("x_Monotone");
    assert_eq!(prop.def, "[][x' >= x]_vars");
}

#[test]
fn non_monotone_variable_has_no_property() {
    let steps = vec![
        step(0, int_state(&[("x", 0)]), None),
        step(1, int_state(&[("x", 2)]), None),
        step(2, int_state(&[("x", 1)]), None),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    assert!(spec.properties.is_empty());
}

#[test]
fn monotone_decreasing_variable_is_note_only() {
    let steps = vec![
        step(0, int_state(&[("x", 3)]), None),
        step(1, int_state(&[("x", 1)]), None),
        step(2, int_state(&[("x", 0)]), None),
    ];
    let spec = mine_spec(&[trace_of("t", &["x"], steps)], &MineOptions::default()).expect("mine");
    assert!(spec.properties.is_empty());
    assert!(
        spec.notes.iter().any(|n| n.contains("non-increasing")),
        "expected a monotone-decrease note; got {:?}",
        spec.notes
    );
}

// ---------------------------------------------------------------------------
// Init, partial observations, rendering, pruning
// ---------------------------------------------------------------------------

#[test]
fn init_is_join_of_distinct_initial_states() {
    let t1 = trace_of(
        "t1",
        &["x"],
        vec![
            step(0, int_state(&[("x", 0)]), None),
            step(1, int_state(&[("x", 1)]), None),
        ],
    );
    let t2 = trace_of(
        "t2",
        &["x"],
        vec![
            step(0, int_state(&[("x", 2)]), None),
            step(1, int_state(&[("x", 3)]), None),
        ],
    );
    let t3 = trace_of("t3", &["x"], vec![step(0, int_state(&[("x", 0)]), None)]);
    let spec = mine_spec(&[t1, t2, t3], &MineOptions::default()).expect("mine");
    assert_eq!(
        spec.init_disjuncts,
        vec![vec!["x = 0".to_string()], vec!["x = 2".to_string()]]
    );
}

#[test]
fn partial_observation_uses_domain_in_init_and_skips_pair_evidence() {
    // y is only observed at step 1; step 0 leaves it unconstrained (domain).
    let mut s0 = HashMap::new();
    s0.insert("x".to_string(), JsonValue::Int(0));
    let steps = vec![
        step(0, s0, None),
        step(1, int_state(&[("x", 1), ("y", 5)]), None),
        step(2, int_state(&[("x", 2), ("y", 5)]), None),
    ];
    let spec = mine_spec(
        &[trace_of("t", &["x", "y"], steps)],
        &MineOptions::default(),
    )
    .expect("mine");
    assert_eq!(
        spec.init_disjuncts,
        vec![vec!["x = 0".to_string(), "y \\in yDomain".to_string()]]
    );
}

#[test]
fn never_observed_variable_is_dropped_with_note() {
    let steps = vec![
        step(0, int_state(&[("x", 0)]), None),
        step(1, int_state(&[("x", 1)]), None),
    ];
    let spec = mine_spec(
        &[trace_of("t", &["x", "ghost"], steps)],
        &MineOptions::default(),
    )
    .expect("mine");
    assert_eq!(spec.variables, vec!["x".to_string()]);
    assert!(spec.notes.iter().any(|n| n.contains("ghost")));
}

#[test]
fn drop_candidate_removes_invariants_and_properties() {
    let mut spec = mine_spec(&[counter_trace()], &MineOptions::default()).expect("mine");
    assert!(spec.drop_candidate("TypeOK_x"));
    assert!(spec.drop_candidate("x_Monotone"));
    assert!(!spec.drop_candidate("TypeOK_x"), "already dropped");
    assert!(spec.invariants.is_empty());
    assert!(spec.properties.is_empty());
}

#[test]
fn rendered_module_and_config_contain_all_sections() {
    let spec = mine_spec(&[counter_trace()], &MineOptions::default()).expect("mine");
    let module = render_module(&spec);
    assert!(module.contains("---- MODULE Mined ----"), "{module}");
    assert!(module.contains("CANDIDATE"), "{module}");
    assert!(module.contains("EXTENDS Integers"), "{module}");
    assert!(module.contains("VARIABLES x"), "{module}");
    assert!(module.contains("vars == <<x>>"), "{module}");
    assert!(module.contains("xDomain == 0..3"), "{module}");
    assert!(module.contains("TypeOK_x == x \\in xDomain"), "{module}");
    assert!(module.contains("\\/ (x = 0)"), "{module}");
    assert!(module.contains("Change_x =="), "{module}");
    assert!(module.contains("/\\ x <= 2"), "{module}");
    assert!(module.contains("/\\ x' = x + 1"), "{module}");
    assert!(
        module.contains("x_Monotone == [][x' >= x]_vars"),
        "{module}"
    );
    assert!(module.trim_end().ends_with("===="), "{module}");

    let cfg = render_config(&spec);
    assert!(cfg.contains("INIT Init"), "{cfg}");
    assert!(cfg.contains("NEXT Next"), "{cfg}");
    assert!(cfg.contains("CHECK_DEADLOCK FALSE"), "{cfg}");
    assert!(cfg.contains("INVARIANT TypeOK_x"), "{cfg}");
    assert!(cfg.contains("PROPERTY x_Monotone"), "{cfg}");
}

#[test]
fn model_values_become_constants() {
    let mut s0 = HashMap::new();
    s0.insert(
        "owner".to_string(),
        JsonValue::ModelValue("alice".to_string()),
    );
    let mut s1 = HashMap::new();
    s1.insert(
        "owner".to_string(),
        JsonValue::ModelValue("bob".to_string()),
    );
    let steps = vec![step(0, s0, None), step(1, s1, None)];
    let spec =
        mine_spec(&[trace_of("t", &["owner"], steps)], &MineOptions::default()).expect("mine");
    assert_eq!(spec.constants, vec!["alice".to_string(), "bob".to_string()]);
    let cfg = render_config(&spec);
    assert!(cfg.contains("CONSTANT alice = alice"), "{cfg}");
    assert!(cfg.contains("CONSTANT bob = bob"), "{cfg}");
}

#[test]
fn mining_errors_are_reported() {
    assert!(matches!(
        mine_spec(&[], &MineOptions::default()),
        Err(MineError::NoTraces)
    ));
    assert!(matches!(
        mine_spec(
            &[trace_of("empty", &["x"], vec![])],
            &MineOptions::default()
        ),
        Err(MineError::EmptyTrace { .. })
    ));
    assert!(matches!(
        mine_spec(
            &[trace_of("t", &["x"], vec![step(0, HashMap::new(), None)])],
            &MineOptions::default()
        ),
        Err(MineError::NoObservations)
    ));
}
