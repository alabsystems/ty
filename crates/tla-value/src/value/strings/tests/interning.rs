// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::intern::intern_string_with_cap;
use super::super::{clear_string_intern_table, intern_string};
use crate::rp::Rp as Arc;
use dashmap::DashMap;
use std::sync::atomic::AtomicUsize;
/// Verify the intern-table size cap: the boundary insert clears the table and
/// leaves only the newly inserted string behind.
/// Part of #1331: memory safety audit -- unbounded intern tables.
///
/// Drives the production cap logic (`intern_string_with_cap`) against a LOCAL
/// table. The previous version asserted exact entry counts on the process-global
/// STRING_INTERN_TABLE, which every concurrently-running test mutates via
/// `Value::string`/`intern_string` — an inherently racy assertion (the
/// long-standing parallel-test flake: "Table should have reset ... got N
/// entries"). The global wrapper (`intern_string`) delegates to this same
/// logic, so cap behavior coverage is preserved without global state.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn string_intern_table_respects_size_cap() {
    const CAP: usize = 8;
    let table: DashMap<String, Arc<str>> = DashMap::new();
    let count = AtomicUsize::new(0);
    let mut token_calls = 0_usize;
    let token_counter = std::cell::Cell::new(0_usize);

    // Insert one more than the cap. The boundary insert should clear the
    // table and leave only the newly inserted string behind.
    for i in 0..=CAP {
        let _ =
            intern_string_with_cap(&table, &count, CAP, &format!("test_cap_string_{i}"), |_| {
                token_counter.set(token_counter.get() + 1);
            });
    }
    token_calls += token_counter.get();

    assert_eq!(
        table.len(),
        1,
        "Table should have reset to the boundary insert"
    );
    assert!(
        table.contains_key("test_cap_string_8"),
        "the boundary insert must survive the cap reset"
    );
    assert_eq!(
        token_calls,
        CAP + 1,
        "every new entry should run the on-new-entry hook (eager token assignment)"
    );

    // The table keeps accepting inserts after the reset.
    let _ = intern_string_with_cap(&table, &count, CAP, "test_cap_string_after_reset", |_| {});
    assert_eq!(
        table.len(),
        2,
        "Table should keep accepting inserts after the reset"
    );

    // Re-interning an existing string is a cache hit: no growth, no hook.
    let before = table.len();
    let hook_fired = std::cell::Cell::new(false);
    let a = intern_string_with_cap(&table, &count, CAP, "test_cap_string_after_reset", |_| {
        hook_fired.set(true);
    });
    let b = intern_string_with_cap(&table, &count, CAP, "test_cap_string_after_reset", |_| {
        hook_fired.set(true);
    });
    assert!(Arc::ptr_eq(&a, &b), "repeat interns share one Arc");
    assert_eq!(table.len(), before);
    assert!(!hook_fired.get(), "cache hits must not re-run the hook");
}

/// Verify that clearing the string intern table does not break equality.
/// After clearing, interning the same string should produce content-equal
/// values even though Arc pointers differ.
#[test]
fn string_intern_clear_preserves_equality() {
    let _lock = crate::value::lock_intern_state();
    clear_string_intern_table();

    let before = intern_string("hello");
    clear_string_intern_table();
    let after = intern_string("hello");

    // Arc pointers should differ (cleared table -> new allocation)
    assert!(!Arc::ptr_eq(&before, &after));
    // Content equality must still hold
    assert_eq!(*before, *after);

    clear_string_intern_table();
}
