// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Checker creation, add_initial/successors, check_liveness basics, error edge cases
//!
//! Split from liveness/checker/tests.rs — Part of #2779

use super::helpers::state_pred_x_eq;
use super::*;
use crate::liveness::test_helpers::{
    constraints_to_grouped_plan, empty_successors, make_checker, make_checker_with_vars,
};
use crate::liveness::LiveExpr;
use crate::Value;
use std::cell::Cell;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_liveness_checker_new() {
    // Create a simple tableau for []P
    let checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    assert_eq!(checker.stats().graph_nodes, 0);
    assert_eq!(checker.stats().states_explored, 0);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_exact_raw_successor_csr_preserves_order_duplicates_and_empty_rows() {
    let a = Fingerprint(0x11);
    let b = Fingerprint(0x22);
    let c = Fingerprint(0x33);
    let d = Fingerprint(0x44);
    let expected_b = [c, a, c, b, b];
    let expected_d = [b, a, b];

    let mut rows = StateSuccessorFingerprints::default();
    rows.insert(a, Arc::new(Vec::new()));
    rows.insert(b, Arc::new(expected_b.to_vec()));
    rows.insert(c, Arc::new(vec![c, c]));

    assert!(rows.freeze());
    assert!(rows.is_frozen());
    assert_eq!(rows.get(&a).expect("empty source row").as_slice(), &[]);
    assert_eq!(
        rows.get(&b).expect("ordered duplicate row").as_slice(),
        expected_b
    );
    let owned_c = rows.get_owned(&c).expect("owned frozen row");
    assert_eq!(
        &*owned_c,
        &[c, c],
        "explicit and appended stuttering duplicates must remain distinct"
    );

    rows.insert(d, Arc::new(expected_d.to_vec()));
    assert!(
        !rows.is_frozen(),
        "a missing cross-group source should open a sparse extension"
    );
    assert_eq!(&*owned_c, &[c, c], "the old owned row must remain valid");
    assert!(rows.freeze(), "the sparse extension must refreeze");
    assert!(rows.is_frozen());
    assert_eq!(
        rows.get(&b).expect("row after repeated freeze").as_slice(),
        expected_b
    );
    assert_eq!(
        rows.get(&d)
            .expect("appended row after refreeze")
            .as_slice(),
        expected_d
    );
    assert_eq!(
        &*owned_c,
        &[c, c],
        "copy-on-write refreeze must preserve outstanding owned rows"
    );
    assert!(rows.freeze(), "repeated frozen freeze must be idempotent");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_exact_raw_csr_count_limits_are_checked_without_truncation() {
    assert!(StateSuccessorFingerprints::csr_counts_fit(3, 7));
    #[cfg(target_pointer_width = "64")]
    {
        let over_u32 = u32::MAX as usize + 1;
        assert!(!StateSuccessorFingerprints::csr_counts_fit(over_u32, 0));
        assert!(!StateSuccessorFingerprints::csr_counts_fit(1, over_u32));
    }
    assert!(!StateSuccessorFingerprints::csr_counts_fit(usize::MAX, 0));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_exact_raw_csr_freeze_requires_closed_payload_and_source_sets() {
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s0_fp = s0.fingerprint();
    let s1_fp = s1.fingerprint();
    let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    checker.enable_owned_behavior_graph_state_cache();
    checker.graph.cache_owned_state(s0_fp, &s0);
    checker.graph.cache_owned_state(s1_fp, &s1);
    checker
        .state_successor_fps
        .insert(s0_fp, Arc::new(vec![s1_fp]));

    assert!(!checker.freeze_complete_exact_raw_adjacency());
    assert!(
        !checker.state_successor_fps.is_frozen(),
        "a partial cache must remain mutable"
    );
    checker
        .state_successor_fps
        .insert(s1_fp, Arc::new(Vec::new()));
    assert!(checker.freeze_complete_exact_raw_adjacency());
    assert!(checker.state_successor_fps.is_frozen());

    let mut missing_endpoint = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    missing_endpoint.enable_owned_behavior_graph_state_cache();
    missing_endpoint.graph.cache_owned_state(s0_fp, &s0);
    missing_endpoint
        .state_successor_fps
        .insert(s0_fp, Arc::new(vec![Fingerprint(0xdead_beef)]));
    assert!(!missing_endpoint.freeze_complete_exact_raw_adjacency());
    assert!(!missing_endpoint.state_successor_fps.is_frozen());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_add_initial_state() {
    // Create a tableau where []TRUE (always true) - all states should be consistent
    let mut checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    let mut get_successors = empty_successors;

    let state = State::from_pairs([("x", Value::int(0))]);
    let added = checker
        .add_initial_state(&state, &mut get_successors, None)
        .unwrap();

    // Should have added at least one node
    assert!(!added.is_empty());
    assert_eq!(checker.graph().len(), added.len());
    // Verify the added nodes reference the correct state — a bug that inserted
    // a default/empty state would pass the count-only checks above.
    for node in &added {
        assert_eq!(
            node.state_fp,
            state.fingerprint(),
            "added node should reference the input state's fingerprint"
        );
        let stored = checker
            .graph()
            .get_state(node)
            .expect("node should exist in graph");
        assert_eq!(
            stored.get("x"),
            Some(&Value::int(0)),
            "stored state should preserve x=0"
        );
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_add_successors() {
    // Create a simple tableau
    let mut checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    let mut get_successors = empty_successors;

    // Add initial state
    let state0 = State::from_pairs([("x", Value::int(0))]);
    let added0 = checker
        .add_initial_state(&state0, &mut get_successors, None)
        .unwrap();
    assert!(!added0.is_empty());

    // Add successors
    let state1 = State::from_pairs([("x", Value::int(1))]);
    let added1 = checker
        .add_successors(
            added0[0],
            std::slice::from_ref(&state1),
            &mut get_successors,
            None,
        )
        .unwrap();

    // Verify successor nodes were actually created and connected — the old test
    // discarded `added1` with `let _ =` and only checked `graph().len() > 1`.
    assert!(
        !added1.is_empty(),
        "add_successors should have created new nodes for state1"
    );
    assert!(
        checker.graph().len() > 1,
        "graph should contain both initial and successor nodes"
    );
    assert!(
        checker.stats().graph_edges > 0,
        "edge count should be non-zero after adding successors"
    );
    // Verify the successor node stores the correct state value
    for node in &added1 {
        let stored = checker
            .graph()
            .get_state(node)
            .expect("successor node should exist in graph");
        assert_eq!(
            stored.get("x"),
            Some(&Value::int(1)),
            "successor stored state should have x=1"
        );
    }
    // Verify the initial node's adjacency list contains the actual successor node(s).
    // A non-emptiness-only check would miss a bug that creates edges pointing to the
    // wrong target (e.g., a self-loop on the initial node instead of an edge to state1).
    let init_info = checker
        .graph()
        .get_node_info(&added0[0])
        .expect("initial node should have info");
    assert!(
        !init_info.successors().is_empty(),
        "initial node should have at least one successor"
    );
    for succ_node in &added1 {
        assert!(
            init_info.successors().contains(succ_node),
            "initial node's adjacency list should contain successor {:?}",
            succ_node
        );
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_direct_compact_explore_records_ordered_successor_fingerprints() {
    let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    checker.enable_owned_behavior_graph_state_cache();

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s2 = State::from_pairs([("x", Value::int(2))]);
    let expected = vec![s2.fingerprint(), s1.fingerprint(), s2.fingerprint()];
    let mut get_successors = |state: &State| {
        if state == &s0 {
            Ok(vec![s2.clone(), s1.clone(), s2.clone()])
        } else {
            Ok(Vec::new())
        }
    };
    let mut raw_fp = |state: &State| Ok(state.fingerprint());

    checker
        .explore_state_graph_direct_with_state_fp(
            std::slice::from_ref(&s0),
            &mut get_successors,
            &mut raw_fp,
        )
        .unwrap();

    assert!(
        checker.state_successors.is_empty(),
        "compact direct exploration must not retain concrete successor vectors"
    );
    assert_eq!(
        checker
            .state_successor_fps
            .get(&s0.fingerprint())
            .expect("source successor fingerprints")
            .as_slice(),
        expected.as_slice(),
        "fingerprint adjacency must preserve successor order and duplicates"
    );
    assert_eq!(
        checker.graph.get_state_by_fp(s1.fingerprint()),
        Some(s1),
        "successor payload must remain resolvable after its temporary State is dropped"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_exact_raw_cache_transfer_reuses_direct_successors() {
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s2 = State::from_pairs([("x", Value::int(2))]);
    let expected = vec![s2.fingerprint(), s1.fingerprint(), s2.fingerprint()];

    let producer_calls = Cell::new(0usize);
    let mut producer_successors = |state: &State| {
        producer_calls.set(producer_calls.get() + 1);
        if state == &s0 {
            Ok(vec![s2.clone(), s1.clone(), s2.clone()])
        } else {
            Ok(Vec::new())
        }
    };
    let mut raw_fp = |state: &State| Ok(state.fingerprint());
    let mut producer = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    producer.enable_owned_behavior_graph_state_cache();
    producer
        .explore_state_graph_direct_with_state_fp(
            std::slice::from_ref(&s0),
            &mut producer_successors,
            &mut raw_fp,
        )
        .unwrap();
    assert_eq!(producer_calls.get(), 3);
    assert!(producer.freeze_complete_exact_raw_adjacency());
    assert!(producer.state_successor_fps.is_frozen());
    let expected_nodes = producer.stats.graph_nodes;
    let expected_edges = producer.stats.graph_edges;
    let cache = producer
        .take_exact_raw_state_graph_cache()
        .expect("producer exact raw cache");

    let consumer_calls = Cell::new(0usize);
    let mut consumer_successors = |_state: &State| {
        consumer_calls.set(consumer_calls.get() + 1);
        Ok(Vec::new())
    };
    let mut consumer = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    consumer.install_exact_raw_state_graph_cache(cache);
    consumer
        .explore_state_graph_direct_with_state_fp(
            std::slice::from_ref(&s0),
            &mut consumer_successors,
            &mut raw_fp,
        )
        .unwrap();

    assert_eq!(
        consumer_calls.get(),
        0,
        "warm direct exploration reran Next"
    );
    assert_eq!(consumer.stats.graph_nodes, expected_nodes);
    assert_eq!(consumer.stats.graph_edges, expected_edges);
    assert_eq!(
        consumer
            .state_successor_fps
            .get(&s0.fingerprint())
            .expect("transferred ordered adjacency")
            .as_slice(),
        expected.as_slice(),
        "transfer must preserve successor order and duplicates"
    );
    assert_eq!(consumer.graph.get_state_by_fp(s1.fingerprint()), Some(s1));
    assert_eq!(consumer.graph.get_state_by_fp(s2.fingerprint()), Some(s2));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_exact_raw_cache_transfer_reuses_successors_in_new_tableau() {
    let state = State::from_pairs([("x", Value::int(0))]);
    let state_fp = state.fingerprint();

    let mut producer = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    producer.enable_owned_behavior_graph_state_cache();
    let mut producer_successors = |current: &State| Ok(vec![current.clone(), current.clone()]);
    let mut raw_fp = |current: &State| Ok(current.fingerprint());
    producer
        .explore_state_graph_direct_with_state_fp(
            std::slice::from_ref(&state),
            &mut producer_successors,
            &mut raw_fp,
        )
        .unwrap();
    let cache = producer
        .take_exact_raw_state_graph_cache()
        .expect("producer exact raw cache");

    let formula = LiveExpr::eventually(LiveExpr::Bool(true));
    let warm_calls = Cell::new(0usize);
    let mut warm_successors = |_current: &State| {
        warm_calls.set(warm_calls.get() + 1);
        Ok(Vec::new())
    };
    let mut warm = make_checker_with_vars(formula.clone(), &["x"]);
    warm.install_exact_raw_state_graph_cache(cache);
    let warm_nodes = warm
        .explore_bfs(std::slice::from_ref(&state), &mut warm_successors, None)
        .unwrap();

    let cold_calls = Cell::new(0usize);
    let mut cold_successors = |current: &State| {
        cold_calls.set(cold_calls.get() + 1);
        Ok(vec![current.clone(), current.clone()])
    };
    let mut cold = make_checker_with_vars(formula, &["x"]);
    cold.enable_owned_behavior_graph_state_cache();
    let cold_nodes = cold
        .explore_bfs(std::slice::from_ref(&state), &mut cold_successors, None)
        .unwrap();

    assert_eq!(warm_calls.get(), 0, "warm tableau exploration reran Next");
    assert_eq!(cold_calls.get(), 1);
    assert_eq!(warm_nodes, cold_nodes);
    assert_eq!(warm.stats.graph_edges, cold.stats.graph_edges);
    assert_eq!(warm.stats.consistency_checks, cold.stats.consistency_checks);
    assert_eq!(
        warm.state_successor_fps
            .get(&state_fp)
            .expect("transferred self-loop adjacency")
            .as_slice(),
        &[state_fp, state_fp]
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_exact_raw_cache_transfer_falls_back_for_uncached_sources() {
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s0_fp = s0.fingerprint();
    let s1_fp = s1.fingerprint();

    let mut producer = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    producer.enable_owned_behavior_graph_state_cache();
    let mut empty = empty_successors;
    producer
        .explore_state_graph_direct(std::slice::from_ref(&s0), &mut empty)
        .unwrap();
    assert!(producer.freeze_complete_exact_raw_adjacency());
    assert!(producer.state_successor_fps.is_frozen());
    let cache = producer
        .take_exact_raw_state_graph_cache()
        .expect("group-complete exact raw cache");

    let calls = Cell::new(0usize);
    let fallback_successor = s0.clone();
    let mut get_successors = |_state: &State| {
        calls.set(calls.get() + 1);
        Ok(vec![fallback_successor.clone()])
    };
    let mut consumer = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    consumer.install_exact_raw_state_graph_cache(cache);
    consumer
        .explore_state_graph_direct(&[s0.clone(), s1.clone()], &mut get_successors)
        .unwrap();

    assert_eq!(
        calls.get(),
        1,
        "only the previously unseen source should regenerate successors"
    );
    assert!(consumer.state_successor_fps.contains_key(&s0_fp));
    assert!(consumer.state_successor_fps.contains_key(&s1_fp));
    assert_eq!(consumer.graph.get_state_by_fp(s0_fp), Some(s0));
    assert_eq!(consumer.graph.get_state_by_fp(s1_fp), Some(s1));
    assert!(consumer.freeze_complete_exact_raw_adjacency());
    assert!(consumer.state_successor_fps.is_frozen());
    assert_eq!(
        consumer
            .state_successor_fps
            .get(&s1_fp)
            .expect("new source row after refreeze")
            .as_slice(),
        &[s0_fp]
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_exact_raw_cache_transfer_missing_payload_fails_closed() {
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s1_fp = s1.fingerprint();

    let mut producer = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    producer.enable_owned_behavior_graph_state_cache();
    let mut producer_successors = |state: &State| {
        if state == &s0 {
            Ok(vec![s1.clone()])
        } else {
            Ok(Vec::new())
        }
    };
    producer
        .explore_state_graph_direct(std::slice::from_ref(&s0), &mut producer_successors)
        .unwrap();
    let mut cache = producer
        .take_exact_raw_state_graph_cache()
        .expect("producer exact raw cache");
    assert!(cache.state_payloads.remove(&s1_fp).is_some());

    let calls = Cell::new(0usize);
    let mut should_not_regenerate = |_state: &State| {
        calls.set(calls.get() + 1);
        Ok(Vec::new())
    };
    let mut consumer = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    consumer.install_exact_raw_state_graph_cache(cache);
    let error = consumer
        .explore_state_graph_direct(std::slice::from_ref(&s0), &mut should_not_regenerate)
        .expect_err("transferred adjacency with a missing payload must fail closed");

    assert_eq!(calls.get(), 0);
    assert!(
        error.to_string().contains("missing successor payload"),
        "unexpected invariant error: {error}"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_owned_tableau_explore_reuses_complete_successors_per_state() {
    let mut checker = make_checker_with_vars(LiveExpr::eventually(LiveExpr::Bool(true)), &["x"]);
    checker.enable_owned_behavior_graph_state_cache();

    let state = State::from_pairs([("x", Value::int(0))]);
    let state_fp = state.fingerprint();
    let successor_calls = Cell::new(0usize);
    let mut get_successors = |current: &State| {
        successor_calls.set(successor_calls.get() + 1);
        Ok(vec![current.clone(), current.clone()])
    };

    let nodes = checker
        .explore_bfs(std::slice::from_ref(&state), &mut get_successors, None)
        .unwrap();

    assert!(
        nodes > 1,
        "the tableau must contain multiple product nodes for the same state"
    );
    assert_eq!(
        successor_calls.get(),
        1,
        "Next should run once for a concrete state shared by tableau nodes"
    );
    assert!(
        checker.state_successors.is_empty(),
        "owned compact exploration must not retain concrete successor vectors"
    );
    assert_eq!(
        checker
            .state_successor_fps
            .get(&state_fp)
            .expect("complete successor fingerprints")
            .as_slice(),
        &[state_fp, state_fp],
        "reused adjacency must preserve generation order and duplicates"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_direct_raw_array_roots_ignore_foreign_cached_fingerprint() {
    let registry = crate::var_index::VarRegistry::from_names(["x"]);
    let state = State::from_pairs([("x", Value::int(7))]);
    let raw_fp = state.fingerprint();
    let foreign_fp = Fingerprint(raw_fp.0 ^ 0xd6e8_feb8_6659_fd93);
    let mut init_array = crate::ArrayState::from_state_with_fp(&state, &registry);
    init_array
        .fp_cache
        .as_mut()
        .expect("from_state_with_fp cache")
        .fingerprint = foreign_fp;

    let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
    checker.enable_owned_behavior_graph_state_cache();
    let mut get_successors = empty_successors;
    checker
        .explore_state_graph_direct_with_raw_array_init_states(
            std::iter::once(&init_array),
            &registry,
            &mut get_successors,
        )
        .unwrap();

    assert_eq!(checker.graph.get_state_by_fp(raw_fp), Some(state));
    assert_eq!(
        checker.graph.get_state_by_fp(foreign_fp),
        None,
        "the raw graph must ignore a foreign cached fingerprint"
    );
    assert!(
        checker
            .graph
            .get_array_state_by_fp(raw_fp)
            .expect("raw compact root")
            .fp_cache
            .is_none(),
        "the adopted payload must not retain the foreign cache"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_tableau_raw_array_roots_match_legacy_state_seeding() {
    let registry = crate::var_index::VarRegistry::from_names(["x"]);
    let states = vec![
        State::from_pairs([("x", Value::int(0))]),
        State::from_pairs([("x", Value::int(1))]),
    ];
    let arrays: Vec<_> = states
        .iter()
        .map(|state| crate::ArrayState::from_state_with_fp(state, &registry))
        .collect();

    let formula = LiveExpr::always(LiveExpr::Bool(true));
    let mut compact = make_checker_with_vars(formula.clone(), &["x"]);
    compact.enable_owned_behavior_graph_state_cache();
    let mut compact_successors = empty_successors;
    let compact_nodes = compact
        .explore_bfs_with_raw_array_init_states(
            arrays.iter(),
            &registry,
            &mut compact_successors,
            None,
        )
        .unwrap();

    let mut legacy = make_checker_with_vars(formula, &["x"]);
    let mut legacy_successors = empty_successors;
    let legacy_nodes = legacy
        .explore_bfs(&states, &mut legacy_successors, None)
        .unwrap();

    assert_eq!(compact_nodes, legacy_nodes);
    assert_eq!(compact.stats.graph_edges, legacy.stats.graph_edges);
    assert_eq!(compact.stats.states_explored, legacy.stats.states_explored);
    for state in states {
        assert_eq!(
            compact.graph.get_state_by_fp(state.fingerprint()),
            Some(state),
            "every compact root must remain reconstructible"
        );
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_liveness_no_cycle() {
    // Create a tableau where []TRUE - always satisfied
    let mut checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    let mut get_successors = empty_successors;

    // Add a linear chain of states (no cycle)
    let state0 = State::from_pairs([("x", Value::int(0))]);
    let state1 = State::from_pairs([("x", Value::int(1))]);
    let state2 = State::from_pairs([("x", Value::int(2))]);

    let added0 = checker
        .add_initial_state(&state0, &mut get_successors, None)
        .unwrap();
    assert!(
        !added0.is_empty(),
        "add_initial_state must return at least one node for the no-cycle test to be meaningful"
    );
    let added1 = checker
        .add_successors(
            added0[0],
            std::slice::from_ref(&state1),
            &mut get_successors,
            None,
        )
        .unwrap();
    let n1 = added1[0];
    let _ = checker
        .add_successors(n1, std::slice::from_ref(&state2), &mut get_successors, None)
        .unwrap();

    // Should be satisfied (no accepting cycle)
    assert!(
        checker.stats().graph_nodes >= 3,
        "checker must have explored all 3 states, got {} graph nodes",
        checker.stats().graph_nodes
    );
    let plan = constraints_to_grouped_plan(checker.constraints());
    let result = checker.check_liveness_grouped(&plan, 0);
    assert!(matches!(result, LivenessResult::Satisfied));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_liveness_single_state_without_self_loop_is_satisfied() {
    // A singleton SCC without a self-loop is not an accepting cycle.
    // Ensure we don't report a violation when the graph has no cycle.
    let p = state_pred_x_eq(1, 1);
    let mut checker = make_checker(LiveExpr::always(LiveExpr::not(p)));
    let mut get_successors = empty_successors;

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let init_nodes = checker
        .add_initial_state(&s0, &mut get_successors, None)
        .unwrap();
    assert_eq!(init_nodes.len(), 1);

    assert_eq!(
        checker.stats().graph_nodes,
        1,
        "single-state graph must have exactly 1 node"
    );
    let plan = constraints_to_grouped_plan(checker.constraints());
    let result = checker.check_liveness_grouped(&plan, 0);
    assert!(matches!(result, LivenessResult::Satisfied));
}

// ======== ERROR PATH TESTS ========

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_clone_state_for_bfs_missing_node_errors() {
    let checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    let missing = BehaviorGraphNode::new(crate::state::Fingerprint(0xBADC0FFE_u64), 0);

    let err = checker
        .clone_state_for_bfs(missing)
        .expect_err("missing BFS node must be an invariant error");
    assert!(
        matches!(err, EvalError::Internal { .. }),
        "expected internal invariant error, got {err:?}"
    );
    assert!(err.to_string().contains("BFS queue contains node"));
}

/// Test liveness on an empty graph (zero states). The checker should report
/// Satisfied since there are no SCCs (and hence no counterexample cycles).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_liveness_empty_graph() {
    let mut checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));

    assert_eq!(
        checker.graph().len(),
        0,
        "graph should be empty before any states added"
    );
    let plan = constraints_to_grouped_plan(checker.constraints());
    let result = checker.check_liveness_grouped(&plan, 0);
    assert!(
        matches!(result, LivenessResult::Satisfied),
        "empty graph should be Satisfied (no SCC, no cycle), got: {:?}",
        result
    );
}

/// Regression test for #1953: grouped liveness checking must surface missing
/// tableau-node invariants as RuntimeFailure via the SCC promise path.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_liveness_grouped_missing_tableau_node_returns_runtime_failure() {
    let mut checker = make_checker(LiveExpr::eventually(state_pred_x_eq(1, 1004)));

    // Same malformed graph setup as non-grouped path test.
    let s0 = State::from_pairs([("x", Value::int(0))]);
    assert!(
        checker.graph.add_init_node(&s0, usize::MAX),
        "malformed initial node should be inserted"
    );
    let malformed = BehaviorGraphNode::from_state(&s0, usize::MAX);
    let _ = checker
        .graph
        .add_successor(malformed, &s0, usize::MAX)
        .expect("self-loop insertion on malformed node should succeed");

    let plan = GroupedLivenessPlan {
        tf: LiveExpr::Bool(true),
        check_state: Vec::new(),
        check_action: Vec::new(),
        pems: vec![PemPlan {
            ae_state_idx: Vec::new(),
            ae_action_idx: Vec::new(),
            ea_state_idx: Vec::new(),
            ea_action_idx: Vec::new(),
        }],
    };

    let result = checker.check_liveness_grouped(&plan, 0);
    match result {
        LivenessResult::RuntimeFailure { reason } => {
            assert!(
                reason.contains("error checking SCC promises"),
                "grouped path should report SCC promise errors, got: {}",
                reason
            );
            assert!(
                reason.contains("tableau invariant violated"),
                "error should preserve tableau invariant context, got: {}",
                reason
            );
        }
        other => panic!(
            "check_liveness_grouped must return RuntimeFailure for missing tableau node, got {:?}",
            other
        ),
    }
}

/// Part of #2236: add_successors must return an error (not silently return
/// empty) when the source node's tableau index is invalid.
///
/// Before the fix, an invalid tableau index in add_successors returned
/// `Ok(added)` (empty vec), silently truncating the behavior graph at that
/// node. This could cause false negatives in liveness checking.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_add_successors_invalid_tableau_idx_returns_error() {
    let mut checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    let mut get_successors = empty_successors;

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);

    // Fabricate a BehaviorGraphNode with a tableau index that doesn't exist
    // in the tableau (the tableau for []TRUE has only 2 nodes: 0 and 1).
    let bad_node = BehaviorGraphNode::new(s0.fingerprint(), usize::MAX);

    let result = checker.add_successors(
        bad_node,
        std::slice::from_ref(&s1),
        &mut get_successors,
        None,
    );
    let err = result.expect_err(
        "add_successors must return error for invalid tableau index, not silently return empty",
    );
    assert!(
        matches!(err, EvalError::Internal { .. }),
        "expected Internal error for missing tableau node, got {err:?}"
    );
    assert!(
        err.to_string().contains("missing tableau node"),
        "error should mention missing tableau node, got: {}",
        err
    );
    assert!(
        err.to_string().contains("add_successors"),
        "error should identify the call site (add_successors), got: {}",
        err
    );
}

/// Regression for DNF overflow handling:
/// checker planning must fail with an explicit error instead of silently
/// returning zero clauses (which could mask liveness violations).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_from_formula_grouped_dnf_overflow_returns_error() {
    // 20 binary disjunctions crossed by conjunction produce 2^20 clauses
    // (1,048,576), which exceeds LiveExpr::MAX_DNF_CLAUSES (500,000).
    let fairness_like_terms: Vec<LiveExpr> = (0u32..20)
        .map(|i| {
            LiveExpr::or(vec![
                state_pred_x_eq(i as i64, 5000 + i * 2),
                state_pred_x_eq(i as i64 + 100, 5000 + i * 2 + 1),
            ])
        })
        .collect();
    let formula = LiveExpr::and(fairness_like_terms);

    let result = LivenessChecker::from_formula_grouped(&formula);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("DNF overflow must return an explicit error"),
    };
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("DNF clause count exceeded limit"),
        "error should report DNF overflow, got: {err_msg}"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_from_formula_dnf_overflow_returns_error() {
    let fairness_like_terms: Vec<LiveExpr> = (0u32..20)
        .map(|i| {
            LiveExpr::or(vec![
                state_pred_x_eq(i as i64, 6000 + i * 2),
                state_pred_x_eq(i as i64 + 100, 6000 + i * 2 + 1),
            ])
        })
        .collect();
    let formula = LiveExpr::and(fairness_like_terms);

    let result = LivenessChecker::from_formula(&formula, crate::eval::EvalCtx::new());
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("DNF overflow must return an explicit error"),
    };
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("DNF clause count exceeded limit"),
        "error should report DNF overflow, got: {err_msg}"
    );
}
