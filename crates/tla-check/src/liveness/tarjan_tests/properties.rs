// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use proptest::prelude::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn prop_tarjan_matches_reference_on_small_graphs() {
    let n = 4usize;
    let config = proptest::test_runner::Config {
        failure_persistence: None,
        ..proptest::test_runner::Config::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);
    let strategy = proptest::collection::vec(any::<bool>(), n * n);

    runner
        .run(&strategy, |adj| {
            let (graph, _states, nodes) = build_graph_from_adj(n, &adj);

            let result = find_sccs(&graph);
            prop_assert!(
                result.errors.is_empty(),
                "Tarjan invariant violation on random 4-node graph: {:?}",
                result.errors
            );
            let tarjan = canonicalize_sccs(result.sccs, &nodes);
            let reference = reference_sccs(n, &adj);

            prop_assert_eq!(tarjan, reference);
            Ok(())
        })
        .expect("small-graph Tarjan proptest should complete");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
#[allow(clippy::erasing_op, clippy::identity_op)]
fn test_1516_cross_scc_edges_separate_sccs() {
    let n = 6;
    let mut adj = vec![false; n * n];
    adj[0 * n + 1] = true;
    adj[1 * n + 2] = true;
    adj[2 * n + 0] = true;
    adj[3 * n + 4] = true;
    adj[4 * n + 3] = true;
    adj[2 * n + 3] = true;

    let (graph, _states, nodes) = build_graph_from_adj(n, &adj);
    let result = find_sccs(&graph);
    assert!(
        result.errors.is_empty(),
        "Tarjan invariant violation: {:?}",
        result.errors
    );
    let tarjan = canonicalize_sccs(result.sccs, &nodes);
    let reference = reference_sccs(n, &adj);

    assert_eq!(tarjan, reference, "Forward cross edge must not merge SCCs");
    assert!(
        tarjan.iter().any(|scc| scc == &[0, 1, 2]),
        "SCC A should be {{0,1,2}}"
    );
    assert!(
        tarjan.iter().any(|scc| scc == &[3, 4]),
        "SCC B should be {{3,4}}"
    );
}

/// Build the same adjacency into a DISK-backed behavior graph so the dense-id
/// Tarjan path (which resolves arena ids via the store's pointer table instead
/// of a `node_to_id` map) is exercised. Node identities match
/// `build_graph_from_adj`, so `canonicalize_sccs` compares across both.
fn build_disk_graph_from_adj(n: usize, adj: &[bool]) -> (BehaviorGraph, Vec<BehaviorGraphNode>) {
    assert_eq!(adj.len(), n * n);

    let mut graph = BehaviorGraph::new_disk_backed(64).expect("disk-backed graph");
    assert!(graph.supports_dense_ids());
    let states: Vec<State> = (0..n).map(|i| make_state(i as i64)).collect();

    for state in &states {
        graph.add_init_node(state, 0);
    }
    let nodes: Vec<BehaviorGraphNode> = states
        .iter()
        .map(|state| BehaviorGraphNode::from_state(state, 0))
        .collect();
    for from_idx in 0..n {
        for to_idx in 0..n {
            if adj[from_idx * n + to_idx] {
                graph
                    .add_successor(nodes[from_idx], &states[to_idx], 0)
                    .expect("disk adjacency builder should add successor");
            }
        }
    }
    (graph, nodes)
}

/// The dense-id path (disk-backed store, `TY_TARJAN_DENSE_IDS` default ON) must
/// produce SCCs bit-identical to the `node_to_id` path (in-memory store) on the
/// same topology. A dense-id bug would corrupt SCC membership → a liveness
/// verdict flip; this differential catches it. (G2 at the unit level.)
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn prop_dense_id_disk_matches_node_to_id_in_memory() {
    let n = 5usize;
    let config = proptest::test_runner::Config {
        failure_persistence: None,
        ..proptest::test_runner::Config::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);
    let strategy = proptest::collection::vec(any::<bool>(), n * n);

    runner
        .run(&strategy, |adj| {
            // node_to_id path (in-memory).
            let (mem_graph, _states, mem_nodes) = build_graph_from_adj(n, &adj);
            let mem_result = find_sccs(&mem_graph);
            prop_assert!(
                mem_result.errors.is_empty(),
                "in-memory Tarjan errors: {:?}",
                mem_result.errors
            );
            let mem_sccs = canonicalize_sccs(mem_result.sccs, &mem_nodes);

            // dense-id path (disk-backed).
            let (disk_graph, disk_nodes) = build_disk_graph_from_adj(n, &adj);
            let disk_result = find_sccs(&disk_graph);
            prop_assert!(
                disk_result.errors.is_empty(),
                "disk dense-id Tarjan errors: {:?}",
                disk_result.errors
            );
            let disk_sccs = canonicalize_sccs(disk_result.sccs, &disk_nodes);

            // Must be identical to each other AND to the reference oracle.
            let reference = reference_sccs(n, &adj);
            prop_assert_eq!(&mem_sccs, &reference);
            prop_assert_eq!(&disk_sccs, &reference);
            prop_assert_eq!(mem_sccs, disk_sccs);
            Ok(())
        })
        .expect("dense-id differential proptest should complete");
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn prop_tarjan_matches_reference_on_medium_graphs() {
    let n = 6usize;

    let config = proptest::test_runner::Config {
        failure_persistence: None,
        ..proptest::test_runner::Config::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);
    let strategy = proptest::collection::vec(any::<bool>(), n * n);

    runner
        .run(&strategy, |adj| {
            let (graph, _states, nodes) = build_graph_from_adj(n, &adj);

            let result = find_sccs(&graph);
            prop_assert!(
                result.errors.is_empty(),
                "Tarjan invariant violation on random 6-node graph: {:?}",
                result.errors
            );
            let tarjan = canonicalize_sccs(result.sccs, &nodes);
            let reference = reference_sccs(n, &adj);

            prop_assert_eq!(tarjan, reference);
            Ok(())
        })
        .expect("medium-graph Tarjan proptest should complete");
}
