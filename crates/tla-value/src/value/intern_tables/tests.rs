// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for the counted intern-table size cap.
//!
//! These drive the production cap primitives (`record_counted_insert`,
//! `reset_counted_table`) against LOCAL tables. The previous versions
//! asserted exact entry counts on the process-global SET/INT_FUNC intern
//! tables, which every concurrently-running test mutates via `Value::set`
//! and friends — an inherently racy assertion (parallel-test flake:
//! "Table should have reset ... got N entries"). The global wrappers
//! (`intern_set_array`, `intern_int_func_array`) route their vacant-insert
//! path through this same `record_counted_insert` logic, so cap behavior
//! coverage is preserved without global state.

use super::shared::{record_counted_insert, reset_counted_table};
use crate::value::Value;
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::rp::Rp as Arc;

fn insert_counted(table: &DashMap<u64, Arc<[Value]>>, count: &AtomicUsize, cap: usize, key: u64) {
    let value: Arc<[Value]> = Arc::from(vec![Value::SmallInt(key as i64)]);
    table.insert(key, Arc::clone(&value));
    record_counted_insert(table, count, key, value, cap);
}

/// Verify that a counted intern table clears when it exceeds the cap and that
/// the boundary insert survives the reset.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn set_intern_table_respects_size_cap() {
    const CAP: usize = 8;
    let table: DashMap<u64, Arc<[Value]>> = DashMap::new();
    let count = AtomicUsize::new(0);

    // Insert one more than the cap. The boundary insert should clear the
    // table and leave only the newly inserted entry behind.
    for i in 0..=CAP as u64 {
        insert_counted(&table, &count, CAP, i);
    }

    assert_eq!(
        table.len(),
        1,
        "Table should have reset to the boundary insert"
    );
    assert!(
        table.contains_key(&(CAP as u64)),
        "the boundary insert must survive the cap reset"
    );
    assert_eq!(count.load(Ordering::Relaxed), 1);

    insert_counted(&table, &count, CAP, 1_000);
    assert_eq!(
        table.len(),
        2,
        "Table should keep accepting inserts after the reset"
    );
    assert_eq!(count.load(Ordering::Relaxed), 2);
}

/// Verify that `reset_counted_table` clears both the table and its counter,
/// and that inserts below the cap never trigger a reset.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn int_func_intern_table_respects_size_cap() {
    const CAP: usize = 8;
    let table: DashMap<u64, Arc<[Value]>> = DashMap::new();
    let count = AtomicUsize::new(0);

    // Stay strictly below the cap: nothing should be evicted.
    for i in 0..CAP as u64 {
        insert_counted(&table, &count, CAP, i);
    }
    assert_eq!(table.len(), CAP, "below-cap inserts must all be retained");
    assert_eq!(count.load(Ordering::Relaxed), CAP);

    // Explicit reset clears the table and the counter.
    reset_counted_table(&table, &count);
    assert_eq!(table.len(), 0);
    assert_eq!(count.load(Ordering::Relaxed), 0);

    // The table keeps accepting inserts after the reset.
    insert_counted(&table, &count, CAP, 42);
    assert_eq!(table.len(), 1);
    assert_eq!(count.load(Ordering::Relaxed), 1);
}
