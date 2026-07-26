// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for [`DiskGraphStore`].

use super::*;
use crate::liveness::checker::{ActionCheckMatrix, CheckMask};
use crate::state::Fingerprint;

fn make_node(fp: u64, tidx: usize) -> BehaviorGraphNode {
    BehaviorGraphNode::new(Fingerprint(fp), tidx)
}

fn make_info(successors: Vec<BehaviorGraphNode>, parent: Option<BehaviorGraphNode>) -> NodeInfo {
    NodeInfo {
        successors,
        trace_parent: parent.map(Box::new),
        state_check_mask: CheckMask::new(),
        action_check_masks: ActionCheckMatrix::new(),
    }
}

#[test]
fn test_append_and_read_single_node() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    let node = make_node(42, 0);
    let info = make_info(vec![], None);

    store.append_node(node, &info).unwrap();
    assert_eq!(store.node_count(), 1);
    assert!(store.contains(node));

    let read_info = store.read_node(node).unwrap().unwrap();
    assert!(read_info.successors.is_empty());
    assert!(read_info.trace_parent.is_none());
}

#[test]
fn test_append_and_read_with_successors() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    let parent = make_node(10, 0);
    let child = make_node(20, 1);
    let succ_a = make_node(30, 0);
    let succ_b = make_node(40, 2);

    let parent_info = make_info(vec![child], None);
    let child_info = make_info(vec![succ_a, succ_b], Some(parent));

    store.append_node(parent, &parent_info).unwrap();
    store.append_node(child, &child_info).unwrap();

    let read_child = store.read_node(child).unwrap().unwrap();
    assert_eq!(read_child.trace_parent.as_deref(), Some(&parent));
    assert_eq!(read_child.successors.len(), 2);
    assert_eq!(read_child.successors[0], succ_a);
    assert_eq!(read_child.successors[1], succ_b);
}

#[test]
fn test_missing_node_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    let node = make_node(99, 0);
    assert!(!store.contains(node));
    assert!(store.read_node(node).unwrap().is_none());
}

#[test]
fn test_multiple_nodes_random_access() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 256).unwrap();

    // Append 50 nodes.
    let nodes: Vec<BehaviorGraphNode> = (1..=50)
        .map(|i| make_node(i * 100, (i % 3) as usize))
        .collect();

    for (i, &node) in nodes.iter().enumerate() {
        let identity = make_node(10_000 + i as u64, i % 7);
        let info = make_info(vec![identity], None);
        store.append_node(node, &info).unwrap();
    }
    assert_eq!(store.node_count(), 50);

    // Read in reverse order (forces disk reads for cache misses).
    for (i, &node) in nodes.iter().enumerate().rev() {
        let read_info = store.read_node(node).unwrap().unwrap();
        assert_eq!(
            read_info.successors,
            vec![make_node(10_000 + i as u64, i % 7)]
        );
    }
}

#[test]
fn test_cache_hit_avoids_disk_read() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    let node = make_node(42, 0);
    let info = make_info(vec![make_node(99, 1)], None);
    store.append_node(node, &info).unwrap();

    // First read populates cache via append.
    // Second read should hit cache.
    let read1 = store.read_node(node).unwrap().unwrap();
    let read2 = store.read_node(node).unwrap().unwrap();
    assert_eq!(read1.successors, read2.successors);
}

#[test]
fn test_init_nodes_tracking() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    let node_a = make_node(1, 0);
    let node_b = make_node(2, 0);

    store.mark_init_node(node_a);
    store.mark_init_node(node_b);

    assert_eq!(store.init_nodes().len(), 2);
    assert_eq!(store.init_nodes()[0], node_a);
    assert_eq!(store.init_nodes()[1], node_b);
}

#[test]
fn test_with_check_masks() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    let node = make_node(42, 0);
    let succ = make_node(99, 1);

    let mut state_mask = CheckMask::new();
    state_mask.set(3);
    state_mask.set(64); // multi-word

    let mut action_mask = CheckMask::new();
    action_mask.set(7);

    let info = NodeInfo {
        successors: vec![succ],
        trace_parent: None,
        state_check_mask: state_mask,
        action_check_masks: vec![action_mask].into(),
    };

    store.append_node(node, &info).unwrap();

    // Invalidate cache to force disk read.
    store.cache[DiskGraphStore::cache_index(node)] = None;

    let read_info = store.read_node(node).unwrap().unwrap();
    assert!(read_info.state_check_mask.get(3));
    assert!(read_info.state_check_mask.get(64));
    assert!(!read_info.state_check_mask.get(4));
    assert!(read_info.action_check_masks.get(0).unwrap().get(7));
}

#[test]
fn test_update_node_rewrites_pointer_and_preserves_count() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    let node = make_node(42, 0);
    store.append_node(node, &make_info(vec![], None)).unwrap();
    let mut updated = make_info(vec![make_node(7, 1)], None);
    updated.state_check_mask.set(3);

    store.update_node(node, &updated).unwrap();

    assert_eq!(store.node_count(), 1);
    assert_eq!(store.all_nodes(), &[node]);

    let read_info = store.read_node(node).unwrap().unwrap();
    assert_eq!(read_info.successors, vec![make_node(7, 1)]);
    assert!(read_info.state_check_mask.get(3));
}

#[test]
fn test_update_node_succeeds_at_pointer_table_load_limit() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 4).unwrap();

    let node_a = make_node(11, 0);
    let node_b = make_node(22, 0);
    let node_c = make_node(33, 0);
    store.append_node(node_a, &make_info(vec![], None)).unwrap();
    store.append_node(node_b, &make_info(vec![], None)).unwrap();
    store.append_node(node_c, &make_info(vec![], None)).unwrap();

    let updated = make_info(vec![make_node(44, 1)], Some(node_a));
    store.update_node(node_b, &updated).unwrap();

    let read_info = store.read_node(node_b).unwrap().unwrap();
    assert_eq!(read_info.trace_parent.as_deref(), Some(&node_a));
    assert_eq!(read_info.successors, vec![make_node(44, 1)]);
    assert_eq!(store.node_count(), 3);
}

#[test]
fn test_flush() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    let node = make_node(42, 0);
    let info = make_info(vec![], None);
    store.append_node(node, &info).unwrap();

    // Flush should not error.
    store.flush().unwrap();
}

#[test]
fn test_same_fp_different_tidx() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    // Same state fingerprint, different tableau indices.
    let node_a = make_node(42, 0);
    let node_b = make_node(42, 1);
    let node_c = make_node(42, 2);

    store
        .append_node(node_a, &make_info(vec![make_node(100, 0)], None))
        .unwrap();
    store
        .append_node(node_b, &make_info(vec![make_node(101, 0)], None))
        .unwrap();
    store
        .append_node(node_c, &make_info(vec![make_node(102, 0)], None))
        .unwrap();

    assert_eq!(store.node_count(), 3);

    assert_eq!(
        store.read_node(node_a).unwrap().unwrap().successors,
        vec![make_node(100, 0)]
    );
    assert_eq!(
        store.read_node(node_b).unwrap().unwrap().successors,
        vec![make_node(101, 0)]
    );
    assert_eq!(
        store.read_node(node_c).unwrap().unwrap().successors,
        vec![make_node(102, 0)]
    );
}

#[test]
fn test_dense_id_matches_all_nodes_position() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 64).unwrap();

    // Mix distinct fingerprints and same-fp/different-tidx nodes.
    let nodes = [
        make_node(10, 0),
        make_node(10, 1),
        make_node(20, 0),
        make_node(30, 2),
        make_node(40, 0),
    ];
    for node in nodes {
        store.append_node(node, &make_info(vec![], None)).unwrap();
    }

    // dense_id_of(all_nodes()[i]) == Some(i) — the core Tarjan invariant.
    let all = store.all_nodes().to_vec();
    assert_eq!(all.len(), nodes.len());
    for (i, node) in all.iter().enumerate() {
        assert_eq!(store.dense_id_of(*node), Some(i as u32));
    }

    // Absent node → None.
    assert_eq!(store.dense_id_of(make_node(99, 0)), None);

    // Rewriting a record (offset moves) must NOT change the dense id.
    let node_c = make_node(20, 0); // dense id 2
    assert_eq!(store.dense_id_of(node_c), Some(2));
    store
        .update_node(node_c, &make_info(vec![make_node(40, 0)], None))
        .unwrap();
    assert_eq!(store.dense_id_of(node_c), Some(2));
    // Every other node's dense id is unchanged as well.
    for (i, node) in store.all_nodes().to_vec().iter().enumerate() {
        assert_eq!(store.dense_id_of(*node), Some(i as u32));
    }
}

#[test]
fn test_store_grows_ptr_table_and_preserves_reads_and_dense_ids() {
    // Start with a tiny ptr-table capacity so appends force several rehashes.
    // Every node must remain readable with its exact record AND keep a stable
    // dense id equal to its all_nodes position after the grows.
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskGraphStore::with_capacity(dir.path(), 4).unwrap();

    let n = 300u64;
    for i in 0..n {
        let node = make_node(i + 1, (i % 4) as usize);
        let identity = make_node(20_000 + i, (i % 5) as usize);
        store
            .append_node(node, &make_info(vec![identity], None))
            .unwrap();
    }
    assert_eq!(store.node_count(), n as usize);

    for i in 0..n {
        let node = make_node(i + 1, (i % 4) as usize);
        // dense id == append position, stable across all the intervening grows.
        assert_eq!(store.dense_id_of(node), Some(i as u32));
        let info = store.read_node(node).unwrap().unwrap();
        assert_eq!(
            info.successors,
            vec![make_node(20_000 + i, (i % 5) as usize)]
        );
    }

    // all_nodes ordering still matches dense ids after growth.
    for (i, node) in store.all_nodes().to_vec().iter().enumerate() {
        assert_eq!(store.dense_id_of(*node), Some(i as u32));
    }
}
