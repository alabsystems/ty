// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for behavior graph construction and trace reconstruction.

use super::*;
use crate::error::EvalError;
use crate::var_index::VarRegistry;
use crate::Value;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_node_equality() {
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s2 = State::from_pairs([("x", Value::int(1))]);
    let s3 = State::from_pairs([("x", Value::int(2))]);

    // Same state, same tableau index -> equal
    let n1 = BehaviorGraphNode::from_state(&s1, 0);
    let n2 = BehaviorGraphNode::from_state(&s2, 0);
    assert_eq!(n1, n2);

    // Same state, different tableau index -> not equal
    let n3 = BehaviorGraphNode::from_state(&s1, 1);
    assert_ne!(n1, n3);

    // Different state, same tableau index -> not equal
    let n4 = BehaviorGraphNode::from_state(&s3, 0);
    assert_ne!(n1, n4);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_add_init() {
    let mut graph = BehaviorGraph::new();
    let s1 = State::from_pairs([("x", Value::int(0))]);

    // First add should succeed
    assert!(graph.add_init_node(&s1, 0));
    assert_eq!(graph.len(), 1);
    assert_eq!(graph.init_nodes().len(), 1);

    // Duplicate add should not increase size
    assert!(!graph.add_init_node(&s1, 0));
    assert_eq!(graph.len(), 1);

    // Same state, different tableau index should be added
    assert!(graph.add_init_node(&s1, 1));
    assert_eq!(graph.len(), 2);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_add_successor() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);

    graph.add_init_node(&s0, 0);
    let init_node = BehaviorGraphNode::from_state(&s0, 0);

    // Add successor
    assert!(graph.add_successor(init_node, &s1, 0).unwrap());
    assert_eq!(graph.len(), 2);

    let succ_node = BehaviorGraphNode::from_state(&s1, 0);
    let info = graph.get_node_info(&succ_node).unwrap();
    assert!(info.trace_parent.is_none());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_logical_successor_rows_preserve_order_duplicates_and_indices() {
    for mut graph in [
        BehaviorGraph::new(),
        BehaviorGraph::new_disk_backed(64).unwrap(),
    ] {
        let source_state = State::from_pairs([("x", Value::int(0))]);
        let first_state = State::from_pairs([("x", Value::int(1))]);
        let second_state = State::from_pairs([("x", Value::int(2))]);
        assert!(graph.try_add_init_node(&source_state, 0).unwrap());
        let source = BehaviorGraphNode::from_state(&source_state, 0);
        let first = BehaviorGraphNode::from_state(&first_state, 1);
        let second = BehaviorGraphNode::from_state(&second_state, 2);

        assert!(graph.add_successor(source, &first_state, 1).unwrap());
        assert!(graph.add_successor(source, &second_state, 2).unwrap());
        assert!(!graph.add_successor(source, &first_state, 1).unwrap());
        let expected = vec![first, second, first];

        let info = graph.try_get_node_info(&source).unwrap().unwrap();
        let row = info.successors();
        assert_eq!(row.len(), expected.len());
        assert!(!row.is_empty());
        assert_eq!(row.get(1), Some(&second));
        assert!(row.contains(&first));
        assert_eq!(row.position(&first), Some(0));
        assert_eq!(row.iter().copied().collect::<Vec<_>>(), expected);
        drop(info);

        let via_graph = graph
            .try_with_successors(&source, |row| row.iter().copied().collect::<Vec<_>>())
            .unwrap()
            .unwrap();
        assert_eq!(via_graph, expected);

        let update_row = graph
            .update_node_masks(&source, |row, state_check_mask, action_check_masks| {
                state_check_mask.set(4);
                *action_check_masks = vec![CheckMask::from_indices(&[5]); row.len()].into();
                row.iter().copied().collect::<Vec<_>>()
            })
            .unwrap()
            .unwrap();
        assert_eq!(update_row, expected);
        let updated = graph.try_get_node_info(&source).unwrap().unwrap();
        assert!(updated.state_check_mask.get(4));
        assert_eq!(updated.action_check_masks.len(), expected.len());
        assert!(updated.action_check_masks.iter().all(|mask| mask.get(5)));
        drop(updated);

        let terminal = graph.try_get_node_info(&first).unwrap().unwrap();
        assert!(terminal.successors().is_empty());
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_packed_inmemory_successors_preserve_topology_masks_and_trace() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s2 = State::from_pairs([("x", Value::int(2))]);
    let s3 = State::from_pairs([("x", Value::int(3))]);
    assert!(graph.try_add_init_node(&s0, 0).unwrap());
    let n0 = BehaviorGraphNode::from_state(&s0, 0);
    let n1 = BehaviorGraphNode::from_state(&s1, 1);
    let n2 = BehaviorGraphNode::from_state(&s2, 2);

    assert!(graph.add_successor(n0, &s1, 1).unwrap());
    assert!(!graph.add_successor(n0, &s0, 0).unwrap());
    assert!(!graph.add_successor(n0, &s1, 1).unwrap());
    assert!(graph.add_successor(n1, &s2, 2).unwrap());

    let expected_n0 = vec![n1, n0, n1];
    graph
        .update_node_masks(&n0, |row, state_mask, action_masks| {
            assert_eq!(row.iter().copied().collect::<Vec<_>>(), expected_n0);
            state_mask.set(7);
            *action_masks = ActionCheckMatrix::from_masks(
                8,
                [
                    CheckMask::from_indices(&[1]),
                    CheckMask::from_indices(&[3]),
                    CheckMask::from_indices(&[5]),
                ],
            );
        })
        .unwrap()
        .unwrap();
    let trace_before = graph.reconstruct_fingerprint_trace(n2).unwrap();

    assert!(graph.pack_inmemory_successors().unwrap());
    let packed = graph.packed_tarjan_view().expect("packed in-memory view");
    assert_eq!(packed.node_keys, &[n0, n1, n2]);
    assert_eq!(packed.offsets, &[0, 3, 4, 4]);
    assert_eq!(packed.targets, &[1, 0, 1, 2]);
    assert_eq!(packed.node_infos.len(), 3);
    assert!(packed
        .node_infos
        .iter()
        .all(|info| info.successors.is_empty()));

    let info = graph.try_get_node_info(&n0).unwrap().unwrap();
    assert_eq!(
        info.successors().iter().copied().collect::<Vec<_>>(),
        expected_n0
    );
    assert_eq!(info.successors().get(1), Some(&n0));
    assert_eq!(info.successors().position(&n1), Some(0));
    assert!(info.state_check_mask.get(7));
    assert!(info.action_check_masks.get(0).unwrap().get(1));
    assert!(info.action_check_masks.get(1).unwrap().get(3));
    assert!(info.action_check_masks.get(2).unwrap().get(5));
    drop(info);
    assert!(graph
        .try_get_node_info(&n2)
        .unwrap()
        .unwrap()
        .successors()
        .is_empty());
    assert_eq!(
        graph.reconstruct_fingerprint_trace(n2).unwrap(),
        trace_before
    );

    // Retry paths may repopulate masks after topology packing.
    graph
        .update_node_masks(&n0, |row, state_mask, action_masks| {
            assert_eq!(row.iter().copied().collect::<Vec<_>>(), expected_n0);
            *state_mask = CheckMask::from_indices(&[11]);
            *action_masks = ActionCheckMatrix::from_masks(
                13,
                row.iter()
                    .enumerate()
                    .map(|(edge_idx, _)| CheckMask::from_indices(&[edge_idx + 9])),
            );
        })
        .unwrap()
        .unwrap();
    let retried = graph.try_get_node_info(&n0).unwrap().unwrap();
    assert!(retried.state_check_mask.get(11));
    for edge_idx in 0..3 {
        assert!(retried
            .action_check_masks
            .get(edge_idx)
            .unwrap()
            .get(edge_idx + 9));
    }
    drop(retried);

    // An idempotent retry must still fail closed if a future mask writer ever
    // loses edge alignment after the topology has already been packed.
    graph
        .update_node_masks(&n0, |_row, _state_mask, action_masks| {
            *action_masks = ActionCheckMatrix::from_masks(1, [CheckMask::from_indices(&[0])]);
        })
        .unwrap()
        .unwrap();
    assert!(graph.pack_inmemory_successors().is_err());
    graph
        .update_node_masks(&n0, |row, _state_mask, action_masks| {
            *action_masks = ActionCheckMatrix::from_masks(
                0,
                std::iter::repeat_with(CheckMask::new).take(row.len()),
            );
        })
        .unwrap()
        .unwrap();

    // Packing is idempotent and completed topology is immutable.
    assert!(graph.pack_inmemory_successors().unwrap());
    assert_eq!(graph.packed_tarjan_view().unwrap().targets, &[1, 0, 1, 2]);
    let node_count = graph.len();
    let cached_state_count = graph.state_cache.len();
    assert!(graph.try_add_init_node(&s3, 3).is_err());
    assert!(graph.add_successor(n2, &s3, 3).is_err());
    assert_eq!(graph.len(), node_count);
    assert_eq!(graph.state_cache.len(), cached_state_count);
    assert!(graph.update_node_info(&n0, |_| ()).is_err());
    assert!(graph.get_node_info_mut(&n0).is_none());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_successor_packing_is_transactional_on_mask_misalignment() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s2 = State::from_pairs([("x", Value::int(2))]);
    assert!(graph.try_add_init_node(&s0, 0).unwrap());
    let n0 = BehaviorGraphNode::from_state(&s0, 0);
    assert!(graph.add_successor(n0, &s1, 0).unwrap());
    assert!(graph.add_successor(n0, &s2, 0).unwrap());
    let expected = vec![
        BehaviorGraphNode::from_state(&s1, 0),
        BehaviorGraphNode::from_state(&s2, 0),
    ];
    graph
        .update_node_info(&n0, |info| {
            info.action_check_masks =
                ActionCheckMatrix::from_masks(1, [CheckMask::from_indices(&[0])]);
        })
        .unwrap()
        .unwrap();

    let error = graph
        .pack_inmemory_successors()
        .expect_err("misaligned masks must reject packing");
    assert!(error.to_string().contains("do not align"));
    assert!(graph.packed_tarjan_view().is_none());
    assert_eq!(
        graph
            .try_get_node_info(&n0)
            .unwrap()
            .unwrap()
            .successors()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        expected
    );

    graph
        .update_node_info(&n0, |info| {
            info.action_check_masks = ActionCheckMatrix::new();
        })
        .unwrap()
        .unwrap();

    let missing = BehaviorGraphNode::new(Fingerprint(0xfeed_cafe), 7);
    graph
        .get_node_info_mut(&n0)
        .unwrap()
        .successors
        .push(missing);
    let error = graph
        .pack_inmemory_successors()
        .expect_err("missing successor target must reject packing");
    assert!(error.to_string().contains("missing target"));
    assert!(graph.packed_tarjan_view().is_none());
    assert_eq!(
        graph
            .get_node_info(&n0)
            .unwrap()
            .successors()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [expected.as_slice(), &[missing]].concat()
    );
    graph.get_node_info_mut(&n0).unwrap().successors.pop();

    assert!(graph.pack_inmemory_successors().unwrap());
    assert_eq!(
        graph
            .try_get_node_info(&n0)
            .unwrap()
            .unwrap()
            .successors()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        expected
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_successor_packing_is_disk_noop() {
    let mut graph = BehaviorGraph::new_disk_backed(64).unwrap();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    assert!(graph.try_add_init_node(&s0, 0).unwrap());
    let n0 = BehaviorGraphNode::from_state(&s0, 0);

    assert!(!graph.pack_inmemory_successors().unwrap());
    assert!(graph.packed_tarjan_view().is_none());
    assert!(graph.add_successor(n0, &s1, 1).unwrap());
    assert_eq!(
        graph
            .try_get_node_info(&n0)
            .unwrap()
            .unwrap()
            .successors()
            .len(),
        1
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_in_memory_dense_ids_follow_stable_node_key_order() {
    let mut graph = BehaviorGraph::new();
    assert!(graph.supports_dense_ids());

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s2 = State::from_pairs([("x", Value::int(2))]);
    assert!(graph.add_init_node(&s0, 1));
    let n0 = BehaviorGraphNode::from_state(&s0, 1);
    assert!(graph.add_successor(n0, &s1, 0).unwrap());
    let n1 = BehaviorGraphNode::from_state(&s1, 0);
    assert!(graph.add_successor(n1, &s2, 2).unwrap());
    let n2 = BehaviorGraphNode::from_state(&s2, 2);
    assert!(!graph.add_init_node(&s0, 1));

    let expected = vec![n0, n1, n2];
    assert_eq!(graph.node_keys(), expected);
    assert_eq!(graph.init_nodes(), vec![n0]);
    for (dense_id, node) in expected.iter().enumerate() {
        assert_eq!(graph.node_dense_id(node), Some(dense_id as u32));
    }

    // Existing-node edges and payload updates must not perturb dense ids.
    assert!(!graph.add_successor(n2, &s0, 1).unwrap());
    graph
        .update_node_info(&n1, |info| info.state_check_mask.set(7))
        .unwrap()
        .expect("existing in-memory node");
    assert_eq!(graph.node_keys(), expected);
    assert_eq!(graph.node_dense_id(&n1), Some(1));
    assert!(graph.get_node_info(&n1).unwrap().state_check_mask.get(7));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_trace_reconstruction() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s2 = State::from_pairs([("x", Value::int(2))]);

    graph.add_init_node(&s0, 0);
    let n0 = BehaviorGraphNode::from_state(&s0, 0);

    graph.add_successor(n0, &s1, 0).unwrap();
    let n1 = BehaviorGraphNode::from_state(&s1, 0);

    graph.add_successor(n1, &s2, 1).unwrap();
    let n2 = BehaviorGraphNode::from_state(&s2, 1);

    let trace = graph.reconstruct_trace(n2).unwrap();
    assert_eq!(trace.len(), 3);
    assert_eq!(trace[0].0, s0);
    assert_eq!(trace[0].1, 0);
    assert_eq!(trace[1].0, s1);
    assert_eq!(trace[1].1, 0);
    assert_eq!(trace[2].0, s2);
    assert_eq!(trace[2].1, 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_fingerprint_trace_reconstruction() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s2 = State::from_pairs([("x", Value::int(2))]);

    graph.add_init_node(&s0, 0);
    let n0 = BehaviorGraphNode::from_state(&s0, 0);

    graph.add_successor(n0, &s1, 0).unwrap();
    let n1 = BehaviorGraphNode::from_state(&s1, 0);

    graph.add_successor(n1, &s2, 1).unwrap();
    let n2 = BehaviorGraphNode::from_state(&s2, 1);

    let trace = graph.reconstruct_fingerprint_trace(n2).unwrap();
    assert_eq!(
        trace,
        vec![
            (s0.fingerprint(), 0),
            (s1.fingerprint(), 0),
            (s2.fingerprint(), 1),
        ]
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_branching_multi_init_trace_matches_in_memory_and_disk() {
    fn build_and_trace(mut graph: BehaviorGraph) -> Vec<(Fingerprint, usize)> {
        let a = State::from_pairs([("x", Value::int(0))]);
        let b = State::from_pairs([("x", Value::int(1))]);
        let branch = State::from_pairs([("x", Value::int(2))]);
        let end = State::from_pairs([("x", Value::int(3))]);

        assert!(graph.try_add_init_node(&a, 0).unwrap());
        assert!(graph.try_add_init_node(&b, 0).unwrap());
        let a_node = BehaviorGraphNode::from_state(&a, 0);
        let b_node = BehaviorGraphNode::from_state(&b, 0);

        // Match production multi-root BFS insertion order: both roots expand
        // before the child of the first root. The second root therefore first
        // discovers `end`, while the longer branch reaches it later.
        assert!(graph.add_successor(a_node, &branch, 0).unwrap());
        assert!(graph.add_successor(b_node, &end, 1).unwrap());
        let branch_node = BehaviorGraphNode::from_state(&branch, 0);
        assert!(!graph.add_successor(branch_node, &end, 1).unwrap());

        let end_node = BehaviorGraphNode::from_state(&end, 1);
        // In-memory reconstruction now walks dense target ids directly;
        // disk-backed packing is a no-op and retains its parent chain.
        graph.pack_inmemory_successors().unwrap();
        graph.reconstruct_fingerprint_trace(end_node).unwrap()
    }

    let in_memory = build_and_trace(BehaviorGraph::new());
    let disk = build_and_trace(BehaviorGraph::new_disk_backed(64).unwrap());
    assert_eq!(in_memory, disk);
    assert_eq!(in_memory.len(), 2);
}

/// Part of #3746: resolve_fingerprint_trace now tolerates partial missing
/// states by skipping them (filter_map). When some states are missing, the
/// result is a shorter trace. When ALL states are missing, it returns an error.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_resolve_fingerprint_trace_skips_missing_states() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);

    graph.add_init_node(&s0, 0);
    let n0 = BehaviorGraphNode::from_state(&s0, 0);
    graph.add_successor(n0, &s1, 0).unwrap();
    let n1 = BehaviorGraphNode::from_state(&s1, 0);

    let trace = graph.reconstruct_fingerprint_trace(n1).unwrap();
    assert_eq!(trace.len(), 2, "trace should have s0 and s1");

    // Remove s1 from state_cache — partial missing is tolerated (#3746).
    graph.state_cache.remove(&n1.state_fp);
    let resolved = graph
        .resolve_fingerprint_trace(&trace)
        .expect("partial missing states should be tolerated");
    assert_eq!(
        resolved.len(),
        1,
        "resolved trace should contain only s0 (s1 was skipped)"
    );
    assert_eq!(resolved[0].0.get("x"), Some(&Value::int(0)));

    // Remove s0 as well — ALL states missing produces an error.
    graph.state_cache.remove(&n0.state_fp);
    let err = graph
        .resolve_fingerprint_trace(&trace)
        .expect_err("all states missing must produce an error");
    assert!(
        matches!(err, EvalError::Internal { .. }),
        "expected internal invariant error, got {err:?}"
    );
    assert!(err.to_string().contains("all"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_add_successor_missing_source_errors() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let missing_from = BehaviorGraphNode::from_state(&s0, 0);

    let err = graph
        .add_successor(missing_from, &s1, 0)
        .expect_err("missing source node must be reported as invariant failure");
    assert!(
        matches!(err, EvalError::Internal { .. }),
        "expected internal invariant error, got {err:?}"
    );
    assert!(err.to_string().contains("source node is missing"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_trace_reconstruction_missing_node_errors() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);

    graph.add_init_node(&s0, 0);
    let n0 = BehaviorGraphNode::from_state(&s0, 0);
    graph.add_successor(n0, &s1, 0).unwrap();

    let missing = BehaviorGraphNode::new(Fingerprint(0xDEADBEEF), 0);

    let err = graph
        .reconstruct_trace(missing)
        .expect_err("missing endpoint must not produce a truncated trace");
    assert!(
        matches!(err, EvalError::Internal { .. }),
        "expected internal invariant error, got {err:?}"
    );
    assert!(err.to_string().contains("missing node"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_contains() {
    let mut graph = BehaviorGraph::new();
    let s1 = State::from_pairs([("x", Value::int(1))]);

    let node = BehaviorGraphNode::from_state(&s1, 0);
    assert!(!graph.contains(&node));

    graph.add_init_node(&s1, 0);
    assert!(graph.contains(&node));

    // Different tableau index should not be found
    let node2 = BehaviorGraphNode::from_state(&s1, 1);
    assert!(!graph.contains(&node2));
}

/// Approach I (#2364): states are deduplicated in state_cache by fingerprint.
/// Multiple behavior graph nodes with the same state but different tableau
/// indices share a single State entry in the cache.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_behavior_graph_state_deduplication_across_tableau_indices() {
    let mut graph = BehaviorGraph::new();
    let s0 = State::from_pairs([("x", Value::int(42))]);

    // Add the same state with two different tableau indices
    assert!(graph.add_init_node(&s0, 0));
    assert!(graph.add_init_node(&s0, 1));
    assert_eq!(graph.len(), 2, "two distinct BG nodes");

    // state_cache should have exactly one entry (deduplicated)
    assert_eq!(
        graph.state_cache.len(),
        1,
        "state_cache must deduplicate by fingerprint"
    );

    // Both nodes should return the same state
    let n0 = BehaviorGraphNode::from_state(&s0, 0);
    let n1 = BehaviorGraphNode::from_state(&s0, 1);
    let state0 = graph.get_state(&n0).expect("state for tableau 0");
    let state1 = graph.get_state(&n1).expect("state for tableau 1");
    assert_eq!(state0, state1);

    // add_successor with same state, different tableau idx should also dedup
    let s1 = State::from_pairs([("x", Value::int(99))]);
    graph.add_successor(n0, &s1, 0).unwrap();
    graph.add_successor(n1, &s1, 1).unwrap();
    assert_eq!(
        graph.state_cache.len(),
        2,
        "two unique states in cache after successor adds"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_owned_compact_state_cache_reconstructs_without_local_states() {
    let mut graph = BehaviorGraph::new();
    graph.enable_owned_state_cache(Arc::new(VarRegistry::from_names(["x"])));

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    assert!(graph.try_add_init_node(&s0, 0).unwrap());
    let n0 = BehaviorGraphNode::from_state(&s0, 0);
    assert!(graph.add_successor(n0, &s1, 0).unwrap());

    assert!(graph.has_compact_state_cache());
    assert!(
        !graph.has_shared_state_cache(),
        "owned payloads must not satisfy shared-cache-only by-fp APIs"
    );
    assert_eq!(
        graph.state_cache.len(),
        0,
        "full States must not be retained"
    );
    assert_eq!(
        graph.owned_state_cache.as_ref().map(|cache| cache.len()),
        Some(2),
        "one compact payload should be retained per exact fingerprint"
    );
    assert_eq!(
        graph.get_state_by_fp(s1.fingerprint()),
        Some(s1),
        "compact payload must reconstruct the exact concrete state"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_owned_array_root_shares_payload_and_drops_foreign_fingerprint_cache() {
    let registry = Arc::new(VarRegistry::from_names(["x"]));
    let state = State::from_pairs([("x", Value::int(7))]);
    let raw_fp = state.fingerprint();
    let mut array = ArrayState::from_state_with_fp(&state, &registry);
    let foreign_fp = Fingerprint(raw_fp.0 ^ 0x9e37_79b9_7f4a_7c15);
    array
        .fp_cache
        .as_mut()
        .expect("from_state_with_fp cache")
        .fingerprint = foreign_fp;
    let source_values = array.compact_values_arc();

    let mut graph = BehaviorGraph::new();
    graph.enable_owned_state_cache(Arc::clone(&registry));
    assert!(graph
        .try_add_init_array_state_with_fp(&array, raw_fp, 0)
        .unwrap());

    let stored = graph
        .get_array_state_by_fp(raw_fp)
        .expect("owned compact root");
    assert!(
        Arc::ptr_eq(&source_values, &stored.compact_values_arc()),
        "compact root insertion must share the value payload"
    );
    assert!(
        stored.fp_cache.is_none(),
        "a storage-domain fingerprint cache must not enter the raw graph"
    );
    assert!(
        graph.state_cache.is_empty(),
        "no full State may be retained"
    );
    assert_eq!(graph.get_state_by_fp(raw_fp), Some(state));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_owned_compact_state_cache_trace_fails_closed_on_missing_payload() {
    let mut graph = BehaviorGraph::new();
    graph.enable_owned_state_cache(Arc::new(VarRegistry::from_names(["x"])));

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);
    let s0_fp = s0.fingerprint();
    let s1_fp = s1.fingerprint();
    assert!(graph
        .try_add_init_node_with_fp(&s0, s0_fp, 0)
        .expect("initial owned node"));
    let n0 = BehaviorGraphNode::new(s0_fp, 0);
    assert!(graph
        .add_successor_with_fp(n0, &s1, s1_fp, 0)
        .expect("owned successor"));
    let n1 = BehaviorGraphNode::new(s1_fp, 0);

    let trace = graph
        .reconstruct_fingerprint_trace(n1)
        .expect("fingerprint trace");
    graph.remove_owned_state_for_test(s1_fp);
    let error = graph
        .resolve_fingerprint_trace(&trace)
        .expect_err("owned trace reconstruction must reject a missing payload");
    assert!(
        error.to_string().contains("missing trace payload"),
        "unexpected missing-trace error: {error}"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_disk_backed_behavior_graph_add_successor_and_trace() {
    let mut graph = BehaviorGraph::new_disk_backed(64).unwrap();
    assert!(graph.is_disk_backed());

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);

    assert!(graph.try_add_init_node(&s0, 0).unwrap());
    let n0 = BehaviorGraphNode::from_state(&s0, 0);
    assert!(graph.add_successor(n0, &s1, 0).unwrap());
    let n1 = BehaviorGraphNode::from_state(&s1, 0);

    let info = graph.try_get_node_info(&n1).unwrap().unwrap();
    assert_eq!(info.trace_parent.as_deref(), Some(&n0));

    graph
        .update_node_info(&n0, |info| {
            info.state_check_mask = crate::liveness::checker::CheckMask::from_u64(1);
        })
        .unwrap();

    let n0_info = graph.try_get_node_info(&n0).unwrap().unwrap();
    assert!(n0_info.state_check_mask.get(0));

    let trace = graph.reconstruct_fingerprint_trace(n1).unwrap();
    assert_eq!(trace, vec![(s0.fingerprint(), 0), (s1.fingerprint(), 0)]);
}
