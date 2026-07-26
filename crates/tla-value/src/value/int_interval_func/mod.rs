// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `IntIntervalFunc` — integer-domain specialized function type.
//!
//! Represents functions `[min..max] -> Value` with a contiguous integer domain.
//! Uses COW (copy-on-write) and interning for memory efficiency during model
//! checking. Extracted from `functions.rs` as part of #3309.
//! Struct declaration co-located with impls as part of #3332.
//! Split into mod/except/ordering as part of #3443.

mod except;
mod ordering;

use super::functions::FP_UNSET;
use super::Value;
use crate::rp::Rp as Arc;
use num_traits::ToPrimitive;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
/// Array-backed function for small integer interval domains.
///
/// This is a performance optimization for functions with domain `a..b` (integer interval).
/// Instead of using OrdMap, values are stored in a contiguous array indexed by `(key - min)`.
/// This makes lookup O(1) and EXCEPT operations O(n) array clone instead of O(log n) B-tree ops,
/// which is faster for small domains due to eliminated allocation overhead.
///
/// Example: `pc = [i \in 1..4 |-> "V0"]` with N=4 creates:
/// - `min=1, max=4, values=["V0", "V0", "V0", "V0"]`
/// - `pc[2]` = `values[2-1]` = `values[1]` = "V0"
/// - `[pc EXCEPT ![3] = "AC"]` clones array and sets `values[2]` = "AC"
///
/// IntIntervalFunc uses `Arc<Vec<Value>>` to enable copy-on-write (COW) optimization.
/// When except() is called and we're the only owner, we can mutate in place
/// via Arc::make_mut() without cloning the entire array.
pub struct IntIntervalFunc {
    /// Minimum domain element (inclusive)
    pub(crate) min: i64,
    /// Maximum domain element (inclusive)
    pub(crate) max: i64,
    /// Values array: `values[(key - min) as usize] = f[key]`
    /// Uses `Arc<Vec>` instead of `Arc<[T]>` to enable `Arc::make_mut` COW optimization.
    pub(crate) values: Arc<Vec<Value>>,
    /// Cached commutative (additive) fingerprint for state-level dedup.
    /// Part of #3168: see FuncValue::additive_fp for rationale.
    /// Part of #3285: `cached_fp` (FP64) removed — recomputed on demand.
    additive_fp: AtomicU64,
}

impl Clone for IntIntervalFunc {
    fn clone(&self) -> Self {
        Self {
            min: self.min,
            max: self.max,
            values: Arc::clone(&self.values),
            additive_fp: AtomicU64::new(self.additive_fp.load(AtomicOrdering::Relaxed)),
        }
    }
}

impl IntIntervalFunc {
    /// Create a new int-interval function with given bounds and values.
    ///
    /// # Arguments
    /// * `min` - Minimum domain element (inclusive)
    /// * `max` - Maximum domain element (inclusive)
    /// * `values` - Values array where `values[i] = f[min + i]`
    pub fn new(min: i64, max: i64, values: Vec<Value>) -> Self {
        assert_eq!(
            Some(values.len()),
            super::checked_interval_len(min, max),
            "IntIntervalFunc::new: values.len() ({}) must equal interval length for [{}, {}]",
            values.len(),
            min,
            max,
        );
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_int_func(values.len());

        IntIntervalFunc {
            min,
            max,
            values: Arc::new(values),
            additive_fp: AtomicU64::new(FP_UNSET),
        }
    }

    /// Minimum domain element (inclusive).
    #[inline]
    pub fn min(&self) -> i64 {
        self.min
    }

    /// Maximum domain element (inclusive).
    #[inline]
    pub fn max(&self) -> i64 {
        self.max
    }

    /// Borrow the value array, where `values()[i]` is `f[min() + i]`.
    #[inline]
    pub fn values(&self) -> &[Value] {
        self.values.as_slice()
    }

    /// Strong reference count of the shared value array.
    ///
    /// A count of 1 means this function uniquely owns its array, so COW
    /// mutations (e.g. `EXCEPT`) can proceed in place without cloning.
    #[inline]
    pub fn values_refcount(&self) -> usize {
        Arc::strong_count(&self.values)
    }

    /// Mutable access to `f[key]`, ONLY when the value array is uniquely
    /// owned (never copies). Returns `None` when shared or out of domain.
    ///
    /// CONTRACT (record canonicalization walk): callers must only replace
    /// the value with a structurally EQUAL one — the cached additive
    /// fingerprint is NOT invalidated by this accessor.
    #[inline]
    pub(crate) fn value_mut_unique(&mut self, key: i64) -> Option<&mut Value> {
        let offset = key.checked_sub(self.min)?;
        if offset < 0 {
            return None;
        }
        Arc::get_mut(&mut self.values)?.get_mut(offset as usize)
    }

    /// Apply the function to an argument. Returns None if out of bounds.
    #[inline]
    pub fn apply(&self, arg: &Value) -> Option<&Value> {
        let idx = match arg {
            Value::SmallInt(n) => *n,
            Value::Int(n) => n.to_i64()?,
            _ => return None,
        };
        if idx >= self.min && idx <= self.max {
            Some(&self.values[(idx - self.min) as usize])
        } else {
            None
        }
    }

    /// Number of elements in the domain
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if domain is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the cached additive fingerprint if already computed.
    /// Part of #3168: commutative hash for state-level dedup.
    #[inline]
    pub fn get_additive_fp(&self) -> Option<u64> {
        let cached = self.additive_fp.load(AtomicOrdering::Relaxed);
        (cached != FP_UNSET).then_some(cached)
    }

    /// Cache the additive fingerprint. Returns the cached value.
    /// Part of #3168: commutative hash for state-level dedup.
    #[inline]
    pub fn cache_additive_fp(&self, fp: u64) -> u64 {
        let _ = self.additive_fp.compare_exchange(
            FP_UNSET,
            fp,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        );
        fp
    }
}
