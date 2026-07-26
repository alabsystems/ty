// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::Value;
use super::{DenseTag, FuncValue};
use crate::rp::Rp as Arc;

impl FuncValue {
    /// Maximum overlay entries before materialization is forced.
    /// Threshold of 4: typical TLA+ EXCEPT has 1-3 clauses.
    const MAX_OVERLAY_SIZE: usize = 4;

    /// Part of #3371: Lazy EXCEPT overlay — O(1) append instead of O(n) clone.
    fn except_existing(mut self, idx: usize, value: Value) -> Self {
        let old_value = self.get_value_at(idx);
        if *old_value == value {
            return self;
        }
        let updated_additive = self.cached_additive_after_replace(idx, old_value, &value);

        // Part of #3371: Use overlay path when below threshold.
        let overlay_len = self.overrides.as_ref().map_or(0, |v| v.len());
        if overlay_len >= Self::MAX_OVERLAY_SIZE {
            // Materialize overlay first, then mutate values directly.
            self.materialize();
            #[cfg(feature = "memory-stats")]
            if Arc::strong_count(&self.values) > 1 {
                crate::value::memory_stats::inc_func_except_clone();
            }
            // Part of #3964: Arc::get_mut fast path for non-atomic check.
            if let Some(values) = Arc::get_mut(&mut self.values) {
                values[idx] = value;
            } else {
                Arc::make_mut(&mut self.values)[idx] = value;
            }
        } else {
            // O(1) overlay append — no values Vec clone.
            let overrides = self
                .overrides
                .get_or_insert_with(|| Box::new(Vec::with_capacity(2)));
            overrides.push((idx, value));
        }

        self.store_additive_cache(updated_additive);
        // Note: tlc_normalized (domain index permutation) is NOT invalidated here.
        // EXCEPT changes values, not domain keys, so the TLC sort order is stable.
        self
    }

    /// Get the logical value at position `idx`, checking overlay first.
    /// Last override wins for duplicate indices.
    #[inline]
    pub fn get_value_at(&self, idx: usize) -> &Value {
        if let Some(ref overrides) = self.overrides {
            for &(oidx, ref val) in overrides.iter().rev() {
                if oidx == idx {
                    return val;
                }
            }
        }
        &self.values[idx]
    }

    /// Check if overlay is active.
    #[inline]
    pub fn has_overlay(&self) -> bool {
        self.overrides.is_some()
    }

    /// Collapse overlay into values Vec. Called by comparison, normalization,
    /// and any path that needs direct values array access.
    pub fn materialize(&mut self) {
        if let Some(overrides) = self.overrides.take() {
            // Part of #3964: Use Arc::get_mut (non-atomic check) when refcount == 1,
            // falling back to Arc::make_mut only when shared.
            let values = if let Some(v) = Arc::get_mut(&mut self.values) {
                v
            } else {
                Arc::make_mut(&mut self.values)
            };
            for (idx, val) in *overrides {
                values[idx] = val;
            }
        }
    }

    /// Apply the function to an argument (lookup by key).
    /// Part of #3371: overlay-aware.
    ///
    /// Fast path: when the domain is a dense integer interval or dense 2-D
    /// integer cross-product (see [`DenseTag`]), the slot is found by direct
    /// array index instead of a binary search with per-step `Value::cmp`. The
    /// index formula is verified against the actual sorted domain at detection
    /// time, so the result is identical to the binary-search path. Every domain
    /// that is not provably dense-integer falls back to the binary search.
    #[inline]
    pub fn apply(&self, arg: &Value) -> Option<&Value> {
        match self.dense_tag() {
            DenseTag::Dim1 => {
                // domain == { SmallInt(lo .. lo+len-1) }; direct index = arg-lo.
                let Value::SmallInt(lo) = self.domain.keys[0] else {
                    // Defensive: detection guarantees SmallInt; if not, fall back.
                    return self.apply_binary_search(arg);
                };
                // Non-integer args are never in an integer domain (no cross-type
                // equality exists between an int and a non-int), so `None` here
                // matches the binary-search result exactly.
                let n = arg.as_i64()?;
                let d = n.checked_sub(lo)?;
                if d >= 0 && (d as u64) < self.domain.keys.len() as u64 {
                    Some(self.get_value_at(d as usize))
                } else {
                    None
                }
            }
            DenseTag::Dim2 => {
                // Fast path only for the exact common shape `f[<<a,b>>]` with two
                // integer components. Any other argument shape (Seq, wrong arity,
                // non-int element) falls back to the binary search for an
                // identical result.
                if let Value::Tuple(t) = arg {
                    if t.len() == 2 {
                        if let (Some(a), Some(b)) = (t[0].as_i64(), t[1].as_i64()) {
                            return self.apply2_dense(a, b);
                        }
                    }
                }
                self.apply_binary_search(arg)
            }
            DenseTag::Sparse => self.apply_binary_search(arg),
        }
    }

    /// Binary-search lookup — the general (byte-identical) apply path.
    #[inline]
    fn apply_binary_search(&self, arg: &Value) -> Option<&Value> {
        self.domain
            .keys
            .binary_search_by(|k| k.cmp(arg))
            .ok()
            .map(|idx| self.get_value_at(idx))
    }

    /// Apply a dense 2-D function directly to the tuple components `<<a, b>>`
    /// without materializing a `Value::Tuple` argument.
    ///
    /// Returns `None` when the domain is not a recognized dense 2-D
    /// cross-product, or when `<<a, b>>` is outside it (i.e. not in the domain).
    /// The result is identical to `self.apply(&Value::Tuple([SmallInt(a),
    /// SmallInt(b)]))` because detection validated that the canonical row-major
    /// enumeration equals the actual sorted domain.
    #[inline]
    pub fn apply2_dense(&self, a: i64, b: i64) -> Option<&Value> {
        if self.dense_tag() != DenseTag::Dim2 {
            return None;
        }
        let len = self.domain.keys.len();
        // domain[0] == <<lo1, lo2>>, domain[len-1] == <<hi1, hi2>> (guaranteed
        // by detection). Re-derive the bounds from the cache-hot endpoints.
        let (lo1, lo2) = tuple2_ints(&self.domain.keys[0])?;
        let (hi1, hi2) = tuple2_ints(&self.domain.keys[len - 1])?;
        if a < lo1 || a > hi1 || b < lo2 || b > hi2 {
            return None;
        }
        let stride = hi2 - lo2 + 1; // > 0: hi2 >= lo2 in a nonempty cross-product
                                    // (a-lo1)*stride + (b-lo2) is in [0, len) given the bounds above.
        let idx = (a - lo1) * stride + (b - lo2);
        Some(self.get_value_at(idx as usize))
    }

    /// Whether this function has a dense 2-D integer cross-product domain, so
    /// callers may route `f[<<a,b>>]` through [`apply2_dense`] and skip the
    /// tuple allocation.
    #[inline]
    pub fn dense_is_dim2(&self) -> bool {
        self.dense_tag() == DenseTag::Dim2
    }

    /// Apply the function to a tuple key given only its components, without
    /// materializing a `Value::Tuple` argument (virtual-tuple apply).
    ///
    /// EXACTNESS: returns byte-identically what
    /// `self.apply(&Value::Tuple(elems.into()))` would return for any domain:
    /// - Dense 2-D integer domains take the same O(1) [`apply2_dense`] index
    ///   path `apply` takes for a materialized 2-tuple of integers.
    /// - Every other case binary-searches the sorted domain with
    ///   [`cmp_tuple_elements_with_value`], which is documented (and
    ///   sorted-set-membership-proven) to be exactly the ordering of
    ///   `Value::Tuple(elems).cmp(key)` — including the cross-representation
    ///   Seq/Func/IntFunc/Record/Bag arms — so the search lands on the same
    ///   slot (or the same miss) as `apply`'s `k.cmp(arg)` search.
    ///
    /// `None` means "not in domain", exactly as for [`FuncValue::apply`].
    #[inline]
    pub fn apply_tuple_elems(&self, elems: &[Value]) -> Option<&Value> {
        if self.dense_tag() == DenseTag::Dim2 {
            if let [a, b] = elems {
                if let (Some(ai), Some(bi)) = (a.as_i64(), b.as_i64()) {
                    return self.apply2_dense(ai, bi);
                }
            }
        }
        self.domain
            .keys
            .binary_search_by(|k| {
                super::super::cmp_helpers::cmp_tuple_elements_with_value(elems, k).reverse()
            })
            .ok()
            .map(|idx| self.get_value_at(idx))
    }

    /// Dense-domain shape (see [`DenseTag`]); shared with the immutable domain.
    #[inline]
    fn dense_tag(&self) -> DenseTag {
        self.domain.dense
    }

    /// Classify the (sorted, unique) domain as a dense integer interval, a dense
    /// 2-D integer cross-product, or neither. Called once per domain descriptor.
    ///
    /// Detection is fail-closed and self-validating: a cheap O(1) endpoint
    /// pre-check rejects the common non-dense case, then the full canonical
    /// enumeration is reconstructed and compared element-by-element against the
    /// actual domain. Only an exact match enables the O(1) index path.
    pub(super) fn compute_dense_tag(domain: &[Value]) -> DenseTag {
        let len = domain.len();
        if len == 0 {
            return DenseTag::Sparse;
        }

        // --- Dim1: domain == { SmallInt(lo), ..., SmallInt(lo+len-1) } ---
        if let (Value::SmallInt(lo), Value::SmallInt(hi)) = (&domain[0], &domain[len - 1]) {
            // Endpoint pre-check: for a sorted/unique integer domain this is a
            // necessary condition for full consecutiveness.
            if hi.checked_sub(*lo) == Some(len as i64 - 1) {
                let mut ok = true;
                for (i, k) in domain.iter().enumerate() {
                    match k {
                        Value::SmallInt(n) if *n == *lo + i as i64 => {}
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    return DenseTag::Dim1;
                }
            }
        }

        // --- Dim2: domain == { <<i,j>> : i in lo1..=hi1, j in lo2..=hi2 } ---
        if let (Some((lo1, lo2)), Some((hi1, hi2))) =
            (tuple2_ints(&domain[0]), tuple2_ints(&domain[len - 1]))
        {
            let (Some(rows), Some(cols)) = (
                hi1.checked_sub(lo1).map(|d| d + 1),
                hi2.checked_sub(lo2).map(|d| d + 1),
            ) else {
                return DenseTag::Sparse;
            };
            // Endpoint pre-check: |rows| * |cols| must equal the domain length.
            if rows >= 1 && cols >= 1 && (rows as i128) * (cols as i128) == len as i128 {
                let mut idx = 0usize;
                let mut ok = true;
                'outer: for n in lo1..=hi1 {
                    for k in lo2..=hi2 {
                        match tuple2_ints(&domain[idx]) {
                            Some((a, b)) if a == n && b == k => {}
                            _ => {
                                ok = false;
                                break 'outer;
                            }
                        }
                        idx += 1;
                    }
                }
                if ok {
                    return DenseTag::Dim2;
                }
            }
        }

        DenseTag::Sparse
    }

    /// Update the function at a single point: [f EXCEPT ![x] = y]
    ///
    /// Takes ownership for copy-on-write (COW) optimization: when the caller
    /// is the sole owner (Arc refcount == 1), modifies entries in place via
    /// `Arc::make_mut` instead of cloning. For chained EXCEPTs like
    /// `[f EXCEPT ![a]=1, ![b]=2, ![c]=3]`, the first creates a new FuncValue
    /// (clone required, original shared with state store), but subsequent
    /// EXCEPTs on the temporary modify in place — reducing O(k*n) to O(n).
    /// Part of #3073 Phase 2.
    pub fn except(self, arg: Value, value: Value) -> Self {
        #[cfg(feature = "memory-stats")]
        crate::value::memory_stats::inc_func_except();
        crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::FuncExcept);

        match self.domain.keys.binary_search_by(|k| k.cmp(&arg)) {
            Ok(idx) => self.except_existing(idx, value),
            Err(_) => {
                // Key not in domain - return unchanged (TLA+ function semantics)
                self
            }
        }
    }

    /// Check whether an EXCEPT operation would actually change the function.
    /// Returns `false` for out-of-domain keys and same-value updates.
    /// Part of #3386: used by compiled-guard no-op fast path to avoid cloning
    /// borrowed/shared function values for semantic no-ops.
    #[inline]
    pub fn would_except_change(&self, arg: &Value, new_val: &Value) -> bool {
        match self.domain.keys.binary_search_by(|k| k.cmp(arg)) {
            Ok(idx) => *self.get_value_at(idx) != *new_val,
            Err(_) => false,
        }
    }
}

/// Extract `(a, b)` if `v` is a 2-element tuple whose elements are both
/// integers (`SmallInt`/`Int`), else `None`. Used by dense 2-D detection and
/// the direct-index apply path.
#[inline]
fn tuple2_ints(v: &Value) -> Option<(i64, i64)> {
    if let Value::Tuple(t) = v {
        if t.len() == 2 {
            if let (Some(a), Some(b)) = (t[0].as_i64(), t[1].as_i64()) {
                return Some((a, b));
            }
        }
    }
    None
}
