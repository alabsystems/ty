// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Interval/domain helpers and set algebra for `SortedSet`.
//!
//! Extracted from `sorted_set/mod.rs` per #3408. Contains `equals_integer_interval`,
//! `equals_sequence_domain`, `insert`, `without`, `remove`, `union`, `intersection`,
//! `difference`, and `is_subset`.

use super::{SortedSet, Value};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::sync::OnceLock;

/// Raw RHS size at or below which a normalized LHS is cheaper to extend by
/// normalizing the RHS once and performing a linear merge. This covers the
/// small set-builder results common in `state_set \\cup {derived : ...}` while
/// retaining deferred normalization for genuinely large raw unions.
pub(super) const SMALL_RAW_UNION_RHS_MAX: usize = 8;

#[inline]
fn small_raw_rhs_union_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("TY_NO_SORTED_SET_SMALL_RHS_UNION")
            .is_ok_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true"))
    })
}

/// Add one merge-proven-new element to an incremental set-fingerprint delta.
///
/// `None` means either the LHS had no cached additive fingerprint or an element
/// could not be fingerprinted. In both cases the result remains uncached and
/// the normal full fingerprint path handles it later.
#[inline]
fn add_new_element_fingerprint(delta: &mut Option<u64>, value: &Value) {
    let Some(current) = *delta else {
        return;
    };
    *delta = crate::dedup_fingerprint::state_value_fingerprint(value)
        .ok()
        .map(|fp| current.wrapping_add(crate::dedup_fingerprint::splitmix64(fp)));
}

impl SortedSet {
    /// Check whether this set is extensionally equal to the integer interval `[min, max]`.
    ///
    /// Accepts either `SmallInt` or `Int` elements so long as their numeric
    /// value matches the expected interval entry.
    pub fn equals_integer_interval(&self, min: i64, max: i64) -> bool {
        let expected_len = if max < min {
            0
        } else {
            match max
                .checked_sub(min)
                .and_then(|diff| diff.checked_add(1))
                .and_then(|len| usize::try_from(len).ok())
            {
                Some(len) => len,
                None => return false,
            }
        };

        let elements = self.as_slice();
        if elements.len() != expected_len {
            return false;
        }

        for (offset, value) in elements.iter().enumerate() {
            let expected = match i64::try_from(offset)
                .ok()
                .and_then(|offset| min.checked_add(offset))
            {
                Some(expected) => expected,
                None => return false,
            };
            if value.as_i64() != Some(expected) {
                return false;
            }
        }
        true
    }

    /// Check whether this set is extensionally equal to `{1, 2, ..., len}`.
    pub fn equals_sequence_domain(&self, len: usize) -> bool {
        match len {
            0 => self.is_empty(),
            _ => i64::try_from(len)
                .ok()
                .is_some_and(|upper| self.equals_integer_interval(1, upper)),
        }
    }

    /// Insert a value, returning a new set (O(n) array copy, bounded fingerprint update).
    ///
    /// Part of #3246: When the parent set has a cached additive fingerprint, the new
    /// set's fingerprint is computed incrementally in O(1), except for a bounded
    /// recomputation when the result is small enough for interning.
    /// This is critical for specs with growing message sets (PaxosCommit pattern).
    pub fn insert(&self, v: Value) -> Self {
        let elements = self.normalized_slice();
        match elements.binary_search(&v) {
            Ok(_) => self.clone(), // Already present
            Err(pos) => {
                let mut vec = Vec::with_capacity(elements.len() + 1);
                vec.extend_from_slice(&elements[..pos]);
                vec.push(v.clone());
                vec.extend_from_slice(&elements[pos..]);
                let new_set = SortedSet::from_sorted_vec_canonical(vec);
                // Propagate the additive fingerprint from the actual returned
                // representation. Small results may be substituted by the set
                // interner with extensionally equal Value variants, so they
                // require a bounded full recompute; larger results cannot be
                // intern-substituted and retain the O(1) delta update.
                if let Some(old_fp) = self.get_additive_fp() {
                    let new_fp = if new_set.len()
                        <= crate::value::intern_tables::sets::MAX_INTERN_SET_SIZE
                    {
                        crate::dedup_fingerprint::compute_set_additive_fp(&new_set).ok()
                    } else {
                        crate::dedup_fingerprint::state_value_fingerprint(&v)
                            .ok()
                            .map(|elem_fp| {
                                let old_len = elements.len() as u64;
                                old_fp
                                    .wrapping_sub(crate::dedup_fingerprint::splitmix64(old_len))
                                    .wrapping_add(crate::dedup_fingerprint::splitmix64(old_len + 1))
                                    .wrapping_add(crate::dedup_fingerprint::splitmix64(elem_fp))
                            })
                    };
                    if let Some(new_fp) = new_fp {
                        new_set.cache_additive_fp(new_fp);
                    }
                }
                new_set
            }
        }
    }

    /// Remove a value, returning a new set (O(n))
    pub fn without(&self, v: &Value) -> Self {
        let elements = self.normalized_slice();
        match elements.binary_search(v) {
            Ok(pos) => {
                crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::SetWithout);
                let mut vec = Vec::with_capacity(elements.len() - 1);
                vec.extend_from_slice(&elements[..pos]);
                vec.extend_from_slice(&elements[pos + 1..]);
                if vec.is_empty() {
                    Self::new()
                } else {
                    SortedSet::from_sorted_vec_canonical(vec)
                }
            }
            Err(_) => self.clone(), // Not present
        }
    }

    /// Remove a value (alias for without, for OrdSet API compatibility)
    #[inline]
    pub fn remove(&self, v: &Value) -> Self {
        self.without(v)
    }

    /// Set union with smart dispatch based on normalization state.
    ///
    /// #3073: When both operands are already normalized (common in the model
    /// checking hot path — state variables are normalized by fingerprinting),
    /// uses O(n+m) sorted merge producing a Normalized result.
    ///
    /// A normalized LHS plus a modest raw RHS also takes the merge path: the RHS
    /// is sorted and deduplicated once in private scratch storage, then merged
    /// into the already-sorted LHS without populating the RHS normalization cache.
    /// This avoids sorting the entire growing result for patterns such as
    /// `edges \\cup {<<new_vertex, parent>> : parent \\in delivered}`. Larger raw
    /// operands retain the lazy concatenation behavior. The narrow fast path is
    /// killable with `TY_NO_SORTED_SET_SMALL_RHS_UNION=1` for A/B benchmarking.
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        if self.ptr_eq(other) {
            return self.clone();
        }

        if self.is_normalized() {
            // Fast path: both already normalized → O(n+m) sorted merge (optimal).
            if other.is_normalized() {
                return self.union_merge(other);
            }

            // Growing-set fast path: normalizing a bounded raw RHS and merging
            // it is O(m log m + n), instead of deferring an
            // O((n+m) log(n+m)) sort of the whole result to the next
            // fingerprint/ordered observation.
            if other.raw_slice().len() <= SMALL_RAW_UNION_RHS_MAX && small_raw_rhs_union_enabled() {
                return self.union_merge_raw_rhs(other);
            }
        }

        // Either operand is Unnormalized → concatenate raw and defer sort.
        // Avoids paying O(n log n) normalization cost at union time when
        // the result may be used for membership testing (linear scan) or
        // combined with further set operations before any ordered observation.
        let raw_a = self.raw_slice();
        let raw_b = other.raw_slice();
        let mut combined = Vec::with_capacity(raw_a.len() + raw_b.len());
        combined.extend_from_slice(raw_a);
        combined.extend_from_slice(raw_b);
        Self::from_unnormalized_vec(combined)
    }

    /// Sorted merge of two normalized sets. O(n+m).
    fn union_merge(&self, other: &Self) -> Self {
        crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::SetUnion);
        let a = self.as_slice();
        let b = other.as_slice();
        let mut result = Vec::with_capacity(a.len() + b.len());
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Less => {
                    result.push(a[i].clone());
                    i += 1;
                }
                Ordering::Greater => {
                    result.push(b[j].clone());
                    j += 1;
                }
                Ordering::Equal => {
                    result.push(a[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&a[i..]);
        result.extend_from_slice(&b[j..]);

        SortedSet::from_sorted_vec_canonical(result)
    }

    /// Normalize a bounded raw RHS in inline scratch storage, then merge it
    /// with the normalized LHS while advancing an existing LHS fingerprint.
    fn union_merge_raw_rhs(&self, other: &Self) -> Self {
        let mut rhs_scratch: SmallVec<[Value; SMALL_RAW_UNION_RHS_MAX]> = SmallVec::new();
        rhs_scratch.extend(other.raw_slice().iter().cloned());
        // Keep the same stable-sort/first-equal-representation behavior as
        // `normalized_elements_from_raw`.
        rhs_scratch.sort();
        rhs_scratch.dedup();

        let a = self.as_slice();
        let b = rhs_scratch.as_slice();
        let mut result = Vec::with_capacity(a.len() + b.len());
        let old_additive_fp = self.get_additive_fp();
        let mut additive_delta = old_additive_fp.map(|_| 0u64);
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Less => {
                    result.push(a[i].clone());
                    i += 1;
                }
                Ordering::Greater => {
                    result.push(b[j].clone());
                    add_new_element_fingerprint(&mut additive_delta, &b[j]);
                    j += 1;
                }
                Ordering::Equal => {
                    // Retain the LHS representation: its cached fingerprint
                    // was computed from this exact representative.
                    result.push(a[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&a[i..]);
        if additive_delta.is_some() {
            for value in &b[j..] {
                add_new_element_fingerprint(&mut additive_delta, value);
            }
        }
        result.extend_from_slice(&b[j..]);

        let old_len = a.len() as u64;
        let new_len = result.len() as u64;
        let new_set = SortedSet::from_sorted_vec_canonical(result);

        if let Some(old_fp) = old_additive_fp {
            // Small normalized sets pass through the interner, which may return
            // an extensionally equal Arc containing different Value variants
            // (for example Set vs Interval). Recompute from the actual returned
            // storage in that lane; state fingerprints are representation-aware
            // for some extensionally equal variants.
            let new_fp = if new_set.len() <= crate::value::intern_tables::sets::MAX_INTERN_SET_SIZE
            {
                crate::dedup_fingerprint::compute_set_additive_fp(&new_set).ok()
            } else {
                // Above the interning cutoff, the merge identifies precisely
                // which normalized RHS elements were absent from the LHS. If
                // any new element was not fingerprintable, the delta is None
                // and the cache deliberately remains unset.
                additive_delta.map(|delta| {
                    old_fp
                        .wrapping_sub(crate::dedup_fingerprint::splitmix64(old_len))
                        .wrapping_add(crate::dedup_fingerprint::splitmix64(new_len))
                        .wrapping_add(delta)
                })
            };
            if let Some(new_fp) = new_fp {
                new_set.cache_additive_fp(new_fp);
            }
        }

        new_set
    }

    /// Set intersection (O(n + m) merge)
    pub fn intersection(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::new();
        }
        if self.ptr_eq(other) {
            return self.clone();
        }
        crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::SetIntersection);

        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        let a = self.as_slice();
        let b = other.as_slice();

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    result.push(a[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }

        if result.is_empty() {
            Self::new()
        } else {
            SortedSet::from_sorted_vec_canonical(result)
        }
    }

    /// Set difference (self \ other) (O(n + m) merge)
    pub fn difference(&self, other: &Self) -> Self {
        if self.is_empty() || self.ptr_eq(other) {
            return Self::new();
        }
        if other.is_empty() {
            return self.clone();
        }
        crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::SetDifference);

        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        let a = self.as_slice();
        let b = other.as_slice();

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Less => {
                    result.push(a[i].clone());
                    i += 1;
                }
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&a[i..]);

        if result.is_empty() {
            Self::new()
        } else {
            SortedSet::from_sorted_vec_canonical(result)
        }
    }

    /// Check if self is a subset of other
    pub fn is_subset(&self, other: &Self) -> bool {
        if self.len() > other.len() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        if self.ptr_eq(other) {
            return true;
        }

        let mut j = 0;
        let b = other.as_slice();
        for v in self {
            while j < b.len() && b[j] < *v {
                j += 1;
            }
            if j >= b.len() || b[j] != *v {
                return false;
            }
            j += 1;
        }
        true
    }
}
