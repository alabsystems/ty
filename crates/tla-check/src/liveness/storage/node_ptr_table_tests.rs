// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for [`NodePtrTable`].

#![allow(clippy::float_cmp)] // load_factor returns exact rational results (e.g. 3/4 = 0.75)

use super::*;

fn make_fp(v: u64) -> Fingerprint {
    Fingerprint(v)
}

#[test]
fn test_insert_and_get_single() {
    let mut table = NodePtrTable::new(64, None).unwrap();
    assert!(table.is_empty());

    let result = table.insert(make_fp(42), 0, 100, 0).unwrap();
    assert!(result); // newly inserted
    assert_eq!(table.len(), 1);
    assert_eq!(table.get(make_fp(42), 0), Some(100));
    assert_eq!(table.get_dense_id(make_fp(42), 0), Some(0));
}

#[test]
fn test_update_existing_key() {
    let mut table = NodePtrTable::new(64, None).unwrap();

    assert!(table.insert(make_fp(42), 0, 100, 7).unwrap());
    // Update: offset moves, dense id must be preserved (caller passes it back).
    assert!(!table.insert(make_fp(42), 0, 200, 7).unwrap()); // update
    assert_eq!(table.len(), 1);
    assert_eq!(table.get(make_fp(42), 0), Some(200));
    assert_eq!(table.get_dense_id(make_fp(42), 0), Some(7));
}

#[test]
fn test_same_fp_different_tidx() {
    let mut table = NodePtrTable::new(64, None).unwrap();

    assert!(table.insert(make_fp(42), 0, 100, 0).unwrap());
    assert!(table.insert(make_fp(42), 1, 200, 1).unwrap());
    assert!(table.insert(make_fp(42), 2, 300, 2).unwrap());

    assert_eq!(table.len(), 3);
    assert_eq!(table.get(make_fp(42), 0), Some(100));
    assert_eq!(table.get(make_fp(42), 1), Some(200));
    assert_eq!(table.get(make_fp(42), 2), Some(300));
    assert_eq!(table.get_dense_id(make_fp(42), 0), Some(0));
    assert_eq!(table.get_dense_id(make_fp(42), 1), Some(1));
    assert_eq!(table.get_dense_id(make_fp(42), 2), Some(2));
}

#[test]
fn test_different_fp_same_tidx() {
    let mut table = NodePtrTable::new(64, None).unwrap();

    assert!(table.insert(make_fp(10), 0, 100, 0).unwrap());
    assert!(table.insert(make_fp(20), 0, 200, 1).unwrap());
    assert!(table.insert(make_fp(30), 0, 300, 2).unwrap());

    assert_eq!(table.get(make_fp(10), 0), Some(100));
    assert_eq!(table.get(make_fp(20), 0), Some(200));
    assert_eq!(table.get(make_fp(30), 0), Some(300));
    assert_eq!(table.get_dense_id(make_fp(10), 0), Some(0));
    assert_eq!(table.get_dense_id(make_fp(20), 0), Some(1));
    assert_eq!(table.get_dense_id(make_fp(30), 0), Some(2));
}

#[test]
fn test_missing_key_returns_none() {
    let mut table = NodePtrTable::new(64, None).unwrap();
    table.insert(make_fp(42), 0, 100, 0).unwrap();

    assert_eq!(table.get(make_fp(42), 1), None); // wrong tidx
    assert_eq!(table.get(make_fp(99), 0), None); // wrong fp
    assert_eq!(table.get(make_fp(99), 1), None); // both wrong
    assert_eq!(table.get_dense_id(make_fp(42), 1), None);
    assert_eq!(table.get_dense_id(make_fp(99), 0), None);
}

#[test]
fn test_contains() {
    let mut table = NodePtrTable::new(64, None).unwrap();
    table.insert(make_fp(42), 0, 100, 0).unwrap();

    assert!(table.contains(make_fp(42), 0));
    assert!(!table.contains(make_fp(42), 1));
    assert!(!table.contains(make_fp(99), 0));
}

#[test]
fn test_zero_fingerprint() {
    let mut table = NodePtrTable::new(64, None).unwrap();

    // FP(0) goes through the side-channel.
    assert!(table.insert(make_fp(0), 0, 100, 0).unwrap());
    assert!(table.insert(make_fp(0), 1, 200, 1).unwrap());
    assert_eq!(table.len(), 2);

    assert_eq!(table.get(make_fp(0), 0), Some(100));
    assert_eq!(table.get(make_fp(0), 1), Some(200));
    assert_eq!(table.get(make_fp(0), 2), None);
    // Dense ids flow through the FP(0) side-channel too.
    assert_eq!(table.get_dense_id(make_fp(0), 0), Some(0));
    assert_eq!(table.get_dense_id(make_fp(0), 1), Some(1));
    assert_eq!(table.get_dense_id(make_fp(0), 2), None);
}

#[test]
fn test_zero_fingerprint_update() {
    let mut table = NodePtrTable::new(64, None).unwrap();

    assert!(table.insert(make_fp(0), 0, 100, 3).unwrap());
    assert!(!table.insert(make_fp(0), 0, 999, 3).unwrap()); // update
    assert_eq!(table.len(), 1);
    assert_eq!(table.get(make_fp(0), 0), Some(999));
    assert_eq!(table.get_dense_id(make_fp(0), 0), Some(3));
}

#[test]
fn test_zero_fingerprint_entries_do_not_consume_hash_table_capacity() {
    let mut table = NodePtrTable::new(4, None).unwrap();

    assert!(table.insert(make_fp(0), 0, 100, 0).unwrap());
    assert!(table.insert(make_fp(0), 1, 200, 1).unwrap());
    assert_eq!(table.load_factor(), 0.0); // FP(0) uses the side-channel, not slots

    assert!(table.insert(make_fp(1), 0, 10, 2).unwrap());
    assert!(table.insert(make_fp(2), 0, 20, 3).unwrap());
    assert!(table.insert(make_fp(3), 0, 30, 4).unwrap());
    assert_eq!(table.load_factor(), 0.75);
    assert_eq!(table.len(), 5); // 2 zero (side-channel) + 3 non-zero (slots)

    // A 4th non-zero insert now GROWS (rehash-exact) instead of erroring; the
    // FP(0) entries live in the side-channel and are carried across the grow.
    assert!(table.insert(make_fp(4), 0, 40, 5).unwrap());
    assert_eq!(table.len(), 6);
    assert_eq!(table.get(make_fp(0), 0), Some(100));
    assert_eq!(table.get(make_fp(0), 1), Some(200));
    assert_eq!(table.get_dense_id(make_fp(0), 0), Some(0));
    for i in 1u64..=4 {
        assert_eq!(table.get(make_fp(i), 0), Some(i * 10));
    }
}

#[test]
fn test_hash_collision_probe_chain() {
    // With a small capacity, hash collisions are guaranteed.
    let mut table = NodePtrTable::new(8, None).unwrap();

    // Insert several entries that will collide.
    for i in 1u64..=5 {
        table.insert(make_fp(i), 0, i * 10, (i - 1) as u32).unwrap();
    }

    assert_eq!(table.len(), 5);
    for i in 1u64..=5 {
        assert_eq!(table.get(make_fp(i), 0), Some(i * 10));
        // Dense ids survive probe chains (stored in the same slot).
        assert_eq!(table.get_dense_id(make_fp(i), 0), Some((i - 1) as u32));
    }
}

#[test]
fn test_grows_when_full_instead_of_erroring() {
    let mut table = NodePtrTable::new(4, None).unwrap();

    // With capacity 4 and load factor 0.75, 3 entries fill it.
    assert!(table.insert(make_fp(1), 0, 10, 0).unwrap());
    assert!(table.insert(make_fp(2), 0, 20, 1).unwrap());
    assert!(table.insert(make_fp(3), 0, 30, 2).unwrap());

    // The 4th insert now GROWS (rehash-exact) and succeeds instead of erroring.
    assert!(table.insert(make_fp(4), 0, 40, 3).unwrap());
    assert_eq!(table.len(), 4);

    // Every entry's offset AND dense id survive the rehash unchanged.
    for i in 1u64..=4 {
        assert_eq!(table.get(make_fp(i), 0), Some(i * 10));
        assert_eq!(table.get_dense_id(make_fp(i), 0), Some((i - 1) as u32));
    }
}

#[test]
fn test_grow_preserves_all_entries_across_multiple_rehashes() {
    // Start tiny and insert enough to force several doublings. A rehash bug that
    // drops or corrupts a slot's key/offset/dense id would surface here — and in
    // the liveness graph would be an SCC-corrupting, verdict-flipping bug.
    let mut table = NodePtrTable::new(4, None).unwrap();
    let n = 500u64;
    for i in 0..n {
        // Distinct fps → distinct keys; distinct offsets and dense ids.
        assert!(table
            .insert(make_fp(i + 1), (i % 3) as usize, i * 7 + 1, i as u32)
            .unwrap());
    }
    assert_eq!(table.len(), n as usize);
    for i in 0..n {
        assert_eq!(table.get(make_fp(i + 1), (i % 3) as usize), Some(i * 7 + 1));
        assert_eq!(
            table.get_dense_id(make_fp(i + 1), (i % 3) as usize),
            Some(i as u32)
        );
    }

    // An update after several grows preserves the dense id, moves the offset.
    assert!(!table.insert(make_fp(1), 0, 99_999, 0).unwrap());
    assert_eq!(table.get(make_fp(1), 0), Some(99_999));
    assert_eq!(table.get_dense_id(make_fp(1), 0), Some(0));
}

#[test]
fn test_update_existing_key_at_load_factor_threshold() {
    let mut table = NodePtrTable::new(4, None).unwrap();

    assert!(table.insert(make_fp(1), 0, 10, 0).unwrap());
    assert!(table.insert(make_fp(2), 0, 20, 1).unwrap());
    assert!(table.insert(make_fp(3), 0, 30, 2).unwrap());

    // Rewrites must succeed even after the table is full for new inserts,
    // preserving the node's dense id.
    assert!(!table.insert(make_fp(2), 0, 99, 1).unwrap());
    assert_eq!(table.get(make_fp(2), 0), Some(99));
    assert_eq!(table.get_dense_id(make_fp(2), 0), Some(1));
    assert_eq!(table.len(), 3);
}

#[test]
fn test_mixed_fp_and_tidx() {
    let mut table = NodePtrTable::new(128, None).unwrap();

    // Simulate a small liveness graph: 10 states × 3 tableau nodes.
    let mut dense = 0u32;
    for state in 1u64..=10 {
        for tidx in 0..3 {
            let offset = state * 1000 + tidx as u64;
            table.insert(make_fp(state), tidx, offset, dense).unwrap();
            dense += 1;
        }
    }

    assert_eq!(table.len(), 30);

    let mut dense = 0u32;
    for state in 1u64..=10 {
        for tidx in 0..3 {
            let expected = state * 1000 + tidx as u64;
            assert_eq!(table.get(make_fp(state), tidx), Some(expected));
            assert_eq!(table.get_dense_id(make_fp(state), tidx), Some(dense));
            dense += 1;
        }
    }
}

#[test]
fn test_file_backed_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let mut table = NodePtrTable::new(64, Some(dir.path().to_path_buf())).unwrap();

    table.insert(make_fp(42), 0, 100, 0).unwrap();
    table.insert(make_fp(42), 1, 200, 1).unwrap();
    table.flush().unwrap();

    assert_eq!(table.get(make_fp(42), 0), Some(100));
    assert_eq!(table.get(make_fp(42), 1), Some(200));
    assert_eq!(table.get_dense_id(make_fp(42), 0), Some(0));
    assert_eq!(table.get_dense_id(make_fp(42), 1), Some(1));
}

#[test]
fn test_load_factor_metric() {
    let mut table = NodePtrTable::new(100, None).unwrap();
    assert_eq!(table.load_factor(), 0.0);

    for i in 1u64..=10 {
        table.insert(make_fp(i), 0, i, (i - 1) as u32).unwrap();
    }
    let lf = table.load_factor();
    assert!((lf - 0.10).abs() < 0.001);
}

#[test]
fn test_large_tableau_index() {
    let mut table = NodePtrTable::new(64, None).unwrap();

    // Tableau indices can be large in theory (though typically small).
    table.insert(make_fp(1), 10_000, 999, 0).unwrap();
    assert_eq!(table.get(make_fp(1), 10_000), Some(999));
    assert_eq!(table.get(make_fp(1), 0), None);
    assert_eq!(table.get_dense_id(make_fp(1), 10_000), Some(0));
}

#[test]
fn test_dense_id_distinct_from_offset() {
    // Dense id and node offset are independent words: verify they do not alias
    // (a swapped-word bug would make them equal).
    let mut table = NodePtrTable::new(64, None).unwrap();

    table.insert(make_fp(500), 2, 8_192, 5).unwrap();
    assert_eq!(table.get(make_fp(500), 2), Some(8_192));
    assert_eq!(table.get_dense_id(make_fp(500), 2), Some(5));

    // Rewriting the offset (record moved) keeps the dense id stable.
    assert!(!table.insert(make_fp(500), 2, 16_384, 5).unwrap());
    assert_eq!(table.get(make_fp(500), 2), Some(16_384));
    assert_eq!(table.get_dense_id(make_fp(500), 2), Some(5));
}
