// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[test]
fn test_insert_and_get() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(4).unwrap();
    let fp_a = Fingerprint(100);
    let fp_b = Fingerprint(200);
    let fp_c = Fingerprint(300);

    graph.insert(fp_a, vec![fp_b, fp_c]);
    assert_eq!(graph.len(), 1);
    assert_eq!(graph.total_successors(), 2);

    let result = graph.get(&fp_a).unwrap();
    assert_eq!(result, vec![fp_b, fp_c]);

    assert!(graph.get(&fp_b).is_none());
}

#[test]
fn test_overwrite() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(4).unwrap();
    let fp_a = Fingerprint(100);
    let fp_b = Fingerprint(200);
    let fp_c = Fingerprint(300);

    graph.insert(fp_a, vec![fp_b]);
    graph.insert(fp_a, vec![fp_b, fp_c]);
    assert_eq!(graph.len(), 1);
    assert_eq!(
        graph.total_successors(),
        2,
        "live successor count should drop dead overwrite records"
    );

    let result = graph.get(&fp_a).unwrap();
    assert_eq!(result, vec![fp_b, fp_c]);
}

#[test]
fn test_empty_successors() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(4).unwrap();
    let fp_a = Fingerprint(100);

    graph.insert(fp_a, vec![]);
    assert_eq!(graph.len(), 1);

    let result = graph.get(&fp_a).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_direct_mapped_cache_collision_evicts_only_colliding_slot() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(2).unwrap();
    let fp_a = Fingerprint(10);
    let fp_b = Fingerprint(11);
    let fp_c = Fingerprint(12);

    graph.insert(fp_a, vec![Fingerprint(10)]);
    graph.insert(fp_b, vec![Fingerprint(20)]);
    assert!(graph.cache_contains(fp_a));
    assert!(graph.cache_contains(fp_b));
    assert_eq!(graph.cache_len(), 2);

    // fp_a and fp_c map to the same direct-mapped slot; fp_b maps elsewhere.
    graph.insert(fp_c, vec![Fingerprint(30)]);

    assert!(!graph.cache_contains(fp_a));
    assert!(graph.cache_contains(fp_b));
    assert!(graph.cache_contains(fp_c));

    // fp_a was evicted from its slot, so this read must come from mmap and
    // should evict fp_c without disturbing fp_b.
    let result = graph.get(&fp_a).unwrap();
    assert_eq!(result, vec![Fingerprint(10)]);
    assert!(graph.cache_contains(fp_a));
    assert!(graph.cache_contains(fp_b));
    assert!(!graph.cache_contains(fp_c));
}

#[test]
fn test_shrink_cache_drops_stale_entries_and_preserves_graph() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(4).unwrap();
    let fp_a = Fingerprint(10);
    let fp_b = Fingerprint(11);
    graph.insert(fp_a, vec![Fingerprint(100)]);
    graph.insert(fp_b, vec![Fingerprint(200)]);
    assert_eq!(graph.cache_slots(), 4);
    assert_eq!(graph.cache_len(), 2);

    graph.shrink_cache_to(1);
    assert_eq!(graph.cache_slots(), 1);
    assert_eq!(graph.cache_len(), 0);
    assert_eq!(graph.get(&fp_a), Some(vec![Fingerprint(100)]));
    assert_eq!(graph.get(&fp_b), Some(vec![Fingerprint(200)]));
    assert_eq!(graph.cache_len(), 1);

    graph.shrink_cache_to(3);
    assert_eq!(
        graph.cache_slots(),
        1,
        "the memory guard must not grow caches"
    );
    assert_eq!(graph.cache_len(), 1);
    assert_eq!(graph.get(&fp_a), Some(vec![Fingerprint(100)]));
}

#[test]
fn test_shrink_cache_preserves_overwrite_accounting_after_eviction() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(4).unwrap();
    let fp_a = Fingerprint(10);
    let fp_b = Fingerprint(11);
    graph.insert(fp_a, vec![Fingerprint(100), Fingerprint(101)]);
    graph.insert(fp_b, vec![Fingerprint(200)]);

    graph.shrink_cache_to(1);
    graph.insert(fp_a, vec![Fingerprint(102)]);
    assert_eq!(graph.len(), 2);
    assert_eq!(graph.total_successors(), 2);
    assert_eq!(graph.get(&fp_a), Some(vec![Fingerprint(102)]));
    assert_eq!(graph.get(&fp_b), Some(vec![Fingerprint(200)]));
}

#[test]
#[should_panic(expected = "cache_slots must be non-zero")]
fn test_shrink_cache_rejects_zero_slots() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(1).unwrap();
    graph.shrink_cache_to(0);
}

#[test]
fn test_overwrite_after_direct_mapped_eviction_counts_live_successors() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(1).unwrap();
    let fp_a = Fingerprint(10);
    let fp_b = Fingerprint(11);

    graph.insert(fp_a, vec![Fingerprint(100), Fingerprint(101)]);
    assert_eq!(graph.total_successors(), 2);

    graph.insert(fp_b, vec![Fingerprint(200)]);
    assert!(!graph.cache_contains(fp_a));
    assert!(graph.cache_contains(fp_b));

    graph.insert(fp_a, vec![Fingerprint(102)]);
    assert_eq!(
        graph.total_successors(),
        2,
        "overwrite accounting must read the old record after cache eviction"
    );
    assert_eq!(graph.get(&fp_a).unwrap(), vec![Fingerprint(102)]);
    assert_eq!(graph.get(&fp_b).unwrap(), vec![Fingerprint(200)]);
}

#[test]
fn test_clear() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(4).unwrap();
    graph.insert(Fingerprint(1), vec![Fingerprint(2)]);
    assert_eq!(graph.file_len(), 20);
    assert_eq!(graph.len(), 1);

    graph.clear();
    assert_eq!(graph.len(), 0);
    assert_eq!(graph.total_successors(), 0);
    assert_eq!(graph.file_len(), 0);
    assert_eq!(graph.mapped_len(), 0);
    assert!(graph.get(&Fingerprint(1)).is_none());

    graph.insert(Fingerprint(3), vec![Fingerprint(4)]);
    assert_eq!(graph.get(&Fingerprint(3)).unwrap(), vec![Fingerprint(4)]);
}

#[test]
fn test_many_entries() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(16).unwrap();
    for i in 0..1000u64 {
        let succs: Vec<Fingerprint> = (0..5).map(|j| Fingerprint(i * 10 + j)).collect();
        graph.insert(Fingerprint(i), succs);
    }
    assert_eq!(graph.len(), 1000);
    assert_eq!(graph.total_successors(), 5000);

    // Verify a sample of entries (will be a mix of cache hits and disk reads).
    for i in [0, 42, 500, 999] {
        let result = graph.get(&Fingerprint(i)).unwrap();
        let expected: Vec<Fingerprint> = (0..5).map(|j| Fingerprint(i * 10 + j)).collect();
        assert_eq!(result, expected, "mismatch at fp={i}");
    }
}

#[test]
fn test_get_remaps_after_new_write_extends_file() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(1).unwrap();
    let fp_a = Fingerprint(1);
    let fp_b = Fingerprint(2);

    graph.insert(fp_a, vec![Fingerprint(10)]);
    // Force the first read through the mmap path instead of the hot cache so
    // the test exercises true remap-after-extend behavior.
    graph.clear_cache_for_test();
    assert_eq!(graph.get(&fp_a).unwrap(), vec![Fingerprint(10)]);
    assert_eq!(graph.mapped_len(), 20);

    graph.insert(fp_b, vec![Fingerprint(20)]);
    assert!(graph.cache_contains(fp_b));
    assert!(!graph.cache_contains(fp_a));
    assert_eq!(graph.file_len(), 40);

    let result = graph.get(&fp_a).unwrap();
    assert_eq!(result, vec![Fingerprint(10)]);
    assert_eq!(graph.mapped_len(), 40);
}

#[test]
fn test_collect_all_fingerprints_includes_parents_and_successors() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(2).unwrap();
    let fp_a = Fingerprint(1);
    let fp_b = Fingerprint(2);
    let fp_c = Fingerprint(3);
    let fp_d = Fingerprint(4);

    graph.insert(fp_a, vec![fp_b, fp_c]);
    graph.insert(fp_d, vec![]);

    let all = graph.collect_all_fingerprints();
    assert_eq!(all.len(), 4);
    assert!(all.contains(&fp_a));
    assert!(all.contains(&fp_b));
    assert!(all.contains(&fp_c));
    assert!(all.contains(&fp_d));
}

#[test]
fn test_collect_all_fingerprints_ignores_overwritten_successors() {
    let mut graph = DiskSuccessorGraph::with_cache_slots(2).unwrap();
    let fp_a = Fingerprint(1);
    let stale_fp_b = Fingerprint(2);
    let stale_fp_c = Fingerprint(3);
    let live_fp_d = Fingerprint(4);

    graph.insert(fp_a, vec![stale_fp_b, stale_fp_c]);
    graph.insert(fp_a, vec![live_fp_d]);
    graph.clear_cache_for_test();

    let all = graph.collect_all_fingerprints();
    assert_eq!(all.len(), 2);
    assert!(all.contains(&fp_a));
    assert!(all.contains(&live_fp_d));
    assert!(
        !all.contains(&stale_fp_b),
        "collect_all_fingerprints should ignore dead overwrite records"
    );
    assert!(
        !all.contains(&stale_fp_c),
        "collect_all_fingerprints should ignore dead overwrite records"
    );
}
