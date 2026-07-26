// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::{parallel_intern, Value};
use super::shared::{record_counted_insert, reset_counted_table, MAX_INTERN_TABLE_ENTRIES};
use crate::rp::Rp as Arc;
use dashmap::DashMap;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::OnceLock;

/// Global intern table for IntIntervalFunc values.
/// Key: FNV-1a hash of (min, max, elements)
/// Value: Arc<Vec<Value>> - the interned values array
static INT_FUNC_INTERN_TABLE: OnceLock<DashMap<u64, Arc<Vec<Value>>>> = OnceLock::new();
static INT_FUNC_INTERN_TABLE_ENTRY_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Maximum IntIntervalFunc size for interning
pub(crate) const MAX_INTERN_INT_FUNC_SIZE: usize = 8;

// Part of #3316: Thread-local flag to skip IntIntervalFunc interning.
//
// Simulation mode generates unique states on random traces - interning
// provides little memory benefit but adds per-EXCEPT overhead:
// hash ALL values + DashMap lookup + potential re-intern.
// For 7-node EWD998ChanID, this is ~200+ hash ops per step.
thread_local! {
    static SKIP_INT_FUNC_INTERNING: Cell<bool> = const { Cell::new(false) };
}

/// Set whether IntIntervalFunc interning should be skipped on this thread.
pub fn set_skip_int_func_interning(skip: bool) {
    SKIP_INT_FUNC_INTERNING.with(|cell| cell.set(skip));
}

/// Swap the thread-local skip flag, returning the previous value.
/// Used by the scoped [`super::InterningSkipGuard`].
pub(crate) fn replace_skip_int_func_interning(skip: bool) -> bool {
    SKIP_INT_FUNC_INTERNING.with(|cell| cell.replace(skip))
}

/// Check if IntIntervalFunc interning should be skipped on this thread.
#[inline]
pub(crate) fn skip_int_func_interning() -> bool {
    SKIP_INT_FUNC_INTERNING.with(|cell| cell.get())
}

#[cfg(feature = "memory-stats")]
pub(crate) fn int_func_intern_table_len() -> Option<usize> {
    INT_FUNC_INTERN_TABLE.get().map(DashMap::len)
}

#[inline]
fn get_int_func_intern_table() -> &'static DashMap<u64, Arc<Vec<Value>>> {
    INT_FUNC_INTERN_TABLE.get_or_init(DashMap::new)
}

// ---------------------------------------------------------------------------
// Lever 5 (#EWD998PCal): thread-local front cache over the global DashMap.
//
// Every per-EXCEPT interning probe (`counter`/`color` IntFunc updates fire
// once per generated successor) pays a sharded-DashMap lookup. The front
// cache keeps recent fp -> Arc bindings in a plain thread-local FxHashMap.
// SOUNDNESS: identical contract to the DashMap layer -- a front-cache hit is
// only returned after FULL element-wise equality validation against the
// requested content, so an fp collision (or a stale entry from a cleared
// global table) degrades to a miss, never a wrong array. Bypassed while
// frozen parallel interning is active (worker overlays own the interning
// discipline there). Kill switch: TY_NO_INT_FUNC_TLS_FRONT=1.
// ---------------------------------------------------------------------------

const INT_FUNC_TLS_FRONT_MAX: usize = 65_536;

thread_local! {
    static INT_FUNC_TLS_FRONT: std::cell::RefCell<rustc_hash::FxHashMap<u64, Arc<Vec<Value>>>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

fn int_func_tls_front_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    !*FLAG.get_or_init(|| {
        std::env::var("TY_NO_INT_FUNC_TLS_FRONT").is_ok_and(|v| !v.trim().is_empty())
    })
}

#[inline]
fn tls_front_get(fp: u64) -> Option<Arc<Vec<Value>>> {
    INT_FUNC_TLS_FRONT.with(|c| c.borrow().get(&fp).cloned())
}

#[inline]
fn tls_front_insert(fp: u64, arc: &Arc<Vec<Value>>) {
    INT_FUNC_TLS_FRONT.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= INT_FUNC_TLS_FRONT_MAX {
            // Pure memoization front: wholesale drop only costs re-probes of
            // the global table, never correctness.
            c.clear();
        }
        c.insert(fp, Arc::clone(arc));
    });
}

/// Clear the thread-local front cache (table reset / test isolation).
pub(crate) fn clear_int_func_tls_front() {
    INT_FUNC_TLS_FRONT.with(|c| c.borrow_mut().clear());
}

/// Snapshot the int-function intern table into an FxHashMap for frozen parallel interning.
/// Part of #3285 Phase 2.
pub(crate) fn snapshot_int_func_intern_table() -> rustc_hash::FxHashMap<u64, Arc<Vec<Value>>> {
    match INT_FUNC_INTERN_TABLE.get() {
        Some(table) => table
            .iter()
            .map(|record| (*record.key(), Arc::clone(record.value())))
            .collect(),
        None => rustc_hash::FxHashMap::default(),
    }
}

/// Compute a fingerprint for an IntIntervalFunc modification.
/// Computes what the fingerprint would be after setting values[arr_idx] = new_value.
#[inline]
fn int_func_modified_fingerprint(
    min: i64,
    max: i64,
    values: &[Value],
    arr_idx: usize,
    new_value: &Value,
) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    min.hash(&mut hasher);
    max.hash(&mut hasher);
    for (i, value) in values.iter().enumerate() {
        if i == arr_idx {
            new_value.hash(&mut hasher);
        } else {
            value.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Compute a fingerprint for an IntIntervalFunc.
#[inline]
pub(crate) fn int_func_fingerprint(min: i64, max: i64, values: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    min.hash(&mut hasher);
    max.hash(&mut hasher);
    values.hash(&mut hasher);
    hasher.finish()
}

/// Try to find an interned IntIntervalFunc with a modification applied.
/// Returns the interned Arc if found, None if we need to create a new one.
///
/// Part of #3285: When parallel interning is active, checks the frozen snapshot
/// and worker-local overlay instead of the global DashMap.
#[inline]
pub(crate) fn try_get_interned_modified(
    min: i64,
    max: i64,
    values: &[Value],
    arr_idx: usize,
    new_value: &Value,
) -> Option<Arc<Vec<Value>>> {
    let fp = int_func_modified_fingerprint(min, max, values, arr_idx, new_value);

    if parallel_intern::is_parallel_intern_active() {
        if let Some(result) =
            parallel_intern::parallel_try_get_interned_modified(fp, values, arr_idx, new_value)
        {
            return result;
        }
    }

    // Lever 5 (#EWD998PCal): thread-local front probe with the SAME full
    // equality validation as the DashMap layer below.
    let use_front = int_func_tls_front_enabled() && !parallel_intern::is_parallel_intern_active();
    if use_front {
        if let Some(arc) = tls_front_get(fp) {
            if arc.len() == values.len() {
                let matches = arc.iter().enumerate().all(|(i, value)| {
                    if i == arr_idx {
                        value == new_value
                    } else {
                        value == &values[i]
                    }
                });
                if matches {
                    return Some(arc);
                }
            }
        }
    }

    let table = get_int_func_intern_table();
    if let Some(arc) = table.get(&fp) {
        if arc.len() == values.len() {
            let matches = arc.iter().enumerate().all(|(i, value)| {
                if i == arr_idx {
                    value == new_value
                } else {
                    value == &values[i]
                }
            });
            if matches {
                if use_front {
                    tls_front_insert(fp, arc.value());
                }
                return Some(Arc::clone(arc.value()));
            }
        }
    }
    None
}

/// Intern an IntIntervalFunc's values array.
///
/// Part of #3285: When parallel interning is active, uses the frozen snapshot
/// + worker-local overlay instead of the global DashMap.
#[inline]
pub(crate) fn intern_int_func_array(min: i64, max: i64, values: Vec<Value>) -> Arc<Vec<Value>> {
    if values.len() > MAX_INTERN_INT_FUNC_SIZE {
        return Arc::new(values);
    }

    let fp = int_func_fingerprint(min, max, &values);

    if parallel_intern::is_parallel_intern_active() {
        if let Some(arc) = parallel_intern::parallel_intern_int_func(fp, &values) {
            return arc;
        }
    }

    // Lever 5 (#EWD998PCal): thread-local front probe with the SAME full
    // equality validation as the DashMap layer below.
    let use_front = int_func_tls_front_enabled() && !parallel_intern::is_parallel_intern_active();
    if use_front {
        if let Some(arc) = tls_front_get(fp) {
            if arc.len() == values.len() && arc.iter().zip(values.iter()).all(|(a, b)| a == b) {
                return arc;
            }
        }
    }

    let table = get_int_func_intern_table();

    if let Some(arc) = table.get(&fp) {
        if arc.len() == values.len() && arc.iter().zip(values.iter()).all(|(a, b)| a == b) {
            if use_front {
                tls_front_insert(fp, arc.value());
            }
            return Arc::clone(arc.value());
        }
    }

    let arc = Arc::new(values);
    match table.entry(fp) {
        dashmap::mapref::entry::Entry::Occupied(mut entry) => {
            let interned = entry.get();
            if interned.len() == arc.len() && interned.iter().zip(arc.iter()).all(|(a, b)| a == b) {
                return Arc::clone(interned);
            }
            entry.insert(Arc::clone(&arc));
        }
        dashmap::mapref::entry::Entry::Vacant(entry) => {
            entry.insert(Arc::clone(&arc));
            record_counted_insert(
                table,
                &INT_FUNC_INTERN_TABLE_ENTRY_COUNT,
                fp,
                Arc::clone(&arc),
                MAX_INTERN_TABLE_ENTRIES,
            );
        }
    }
    if use_front {
        tls_front_insert(fp, &arc);
    }
    arc
}

/// Clear the IntIntervalFunc intern table.
pub fn clear_int_func_intern_table() {
    // Lever 5: drop the calling thread's front cache too. (Front entries are
    // equality-validated per hit, so stale entries on OTHER threads remain
    // sound — clearing here is about releasing memory and test hygiene.)
    clear_int_func_tls_front();
    if let Some(table) = INT_FUNC_INTERN_TABLE.get() {
        reset_counted_table(table, &INT_FUNC_INTERN_TABLE_ENTRY_COUNT);
    } else {
        INT_FUNC_INTERN_TABLE_ENTRY_COUNT.store(0, AtomicOrdering::Relaxed);
    }
}

#[cfg(test)]
mod tls_front_tests {
    use super::*;

    /// Lever 5 (#EWD998PCal): the TLS front returns content-identical arrays
    /// and NEVER serves an fp bucket whose content differs from the request
    /// (collision degrades to a miss into the global table).
    #[test]
    fn tls_front_validates_content_and_dedups() {
        clear_int_func_intern_table();
        let a1 = intern_int_func_array(0, 2, vec![Value::int(1), Value::int(2), Value::int(3)]);
        // Re-intern identical content: must dedup to the same Arc (served by
        // the TLS front or the global table — either way content-identical).
        let a2 = intern_int_func_array(0, 2, vec![Value::int(1), Value::int(2), Value::int(3)]);
        assert!(Arc::ptr_eq(&a1, &a2));

        // Different content must produce a different array even after the
        // front cache is warm.
        let b = intern_int_func_array(0, 2, vec![Value::int(1), Value::int(2), Value::int(4)]);
        assert!(!Arc::ptr_eq(&a1, &b));
        assert_ne!(a1.as_slice(), b.as_slice());

        // Modified-lookup soundness: whatever it returns (hit or miss — the
        // modified-fp keying is a different hash stream from the direct-fp
        // inserts, so hits are not guaranteed), a returned array MUST be
        // content-identical to the requested modification.
        for (new_val, expected) in [
            (
                Value::int(4),
                vec![Value::int(1), Value::int(2), Value::int(4)],
            ),
            (
                Value::int(5),
                vec![Value::int(1), Value::int(2), Value::int(5)],
            ),
        ] {
            if let Some(arc) = try_get_interned_modified(0, 2, a1.as_slice(), 2, &new_val) {
                assert_eq!(arc.as_slice(), expected.as_slice());
            }
        }
        clear_int_func_intern_table();
    }
}
