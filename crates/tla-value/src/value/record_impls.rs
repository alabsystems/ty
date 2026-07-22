// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! RecordBuilder, trait impls, and RecordIter for RecordValue.
//!
//! Extracted from the former `record.rs` as part of #3309 to keep each file
//! under the 500-line target. `RecordValue` core type and methods now live in
//! `record/mod.rs`.

use super::record::RecordValue;
use crate::rp::Rp;
use super::Value;
use smallvec::SmallVec;
use std::sync::Arc;
use tla_core::{intern_name, name_id_str_cmp, resolve_name_id, NameId};

/// Builder for constructing RecordValue incrementally.
///
/// Collects field-value pairs and sorts them when building the final RecordValue.
///
/// Part of #3805: Uses SmallVec<[(NameId, Value); 6]> to avoid heap allocation for
/// records with <= 6 fields (the vast majority in TLA+ -- most state records have
/// 3-6 fields like `[pc, stack, x, y]`).
pub struct RecordBuilder {
    entries: SmallVec<[(NameId, Value); 6]>,
}

impl RecordBuilder {
    /// Create a new empty builder
    pub fn new() -> Self {
        RecordBuilder {
            entries: SmallVec::new(),
        }
    }

    /// Create a new builder with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        RecordBuilder {
            entries: SmallVec::with_capacity(capacity),
        }
    }

    /// Add a field-value pair by NameId
    pub fn insert(&mut self, field: NameId, value: Value) {
        self.entries.push((field, value));
    }

    /// Add a field-value pair by string name (interned)
    pub fn insert_str(&mut self, field: &str, value: Value) {
        self.entries.push((intern_name(field), value));
    }

    /// Add a field-value pair by Arc<str> name (interned)
    pub fn insert_arc(&mut self, field: &Arc<str>, value: Value) {
        self.entries.push((intern_name(field), value));
    }

    /// Build the RecordValue, sorting entries into the canonical record field
    /// order (field-name string; see `record` module docs).
    ///
    /// Uses `into_vec()` for efficient Vec conversion: no-op when already
    /// spilled to heap, single alloc + copy when inline.
    pub fn build(self) -> RecordValue {
        RecordValue::from_entries(self.entries.into_vec())
    }
}

impl Default for RecordBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Trait impls for RecordValue
// ============================================================================

impl From<Vec<(String, Value)>> for RecordValue {
    fn from(entries: Vec<(String, Value)>) -> Self {
        let id_entries: Vec<(NameId, Value)> = entries
            .into_iter()
            .map(|(k, v)| (intern_name(&k), v))
            .collect();
        RecordValue::from_entries(id_entries)
    }
}

impl Default for RecordValue {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for RecordValue {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: pointer equality
        if crate::rp::Rp::ptr_eq(&self.entries, &other.entries) {
            return true;
        }
        self.entries == other.entries
    }
}

impl Eq for RecordValue {}

impl PartialOrd for RecordValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecordValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Fast path: pointer equality
        if crate::rp::Rp::ptr_eq(&self.entries, &other.entries) {
            return std::cmp::Ordering::Equal;
        }
        // Records are functions with string domains: compare (key, value)
        // pairs lexicographically in the canonical field order, with keys
        // compared by their NAME STRINGS — never by NameId numeric value
        // (interning order, run-dependent). This keeps the total order
        // consistent with `cmp_record_with_func` so mixed Record/Func sorted
        // sets stay transitive.
        //
        // Fast path per pair: identical NameIds (same field, O(1) integer
        // equality) skip the string compare entirely — the common case when
        // comparing same-shaped records.
        let mut a_iter = self.entries.iter();
        let mut b_iter = other.entries.iter();
        loop {
            match (a_iter.next(), b_iter.next()) {
                (Some((ak, av)), Some((bk, bv))) => {
                    if ak != bk {
                        match name_id_str_cmp(*ak, *bk) {
                            std::cmp::Ordering::Equal => {}
                            ord => return ord,
                        }
                    }
                    match av.cmp(bv) {
                        std::cmp::Ordering::Equal => {}
                        ord => return ord,
                    }
                }
                (None, Some(_)) => return std::cmp::Ordering::Less,
                (Some(_), None) => return std::cmp::Ordering::Greater,
                (None, None) => return std::cmp::Ordering::Equal,
            }
        }
    }
}

impl std::hash::Hash for RecordValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.entries.hash(state);
    }
}

impl std::fmt::Debug for RecordValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|(k, v)| {
                let name = resolve_name_id(*k);
                (name, v)
            }))
            .finish()
    }
}

impl FromIterator<(NameId, Value)> for RecordValue {
    fn from_iter<I: IntoIterator<Item = (NameId, Value)>>(iter: I) -> Self {
        RecordValue::from_entries(iter.into_iter().collect())
    }
}

impl FromIterator<(Arc<str>, Value)> for RecordValue {
    fn from_iter<I: IntoIterator<Item = (Arc<str>, Value)>>(iter: I) -> Self {
        let entries: Vec<_> = iter
            .into_iter()
            .map(|(k, v)| (intern_name(&k), v))
            .collect();
        RecordValue::from_entries(entries)
    }
}

impl<'a> IntoIterator for &'a RecordValue {
    type Item = (NameId, &'a Value);
    type IntoIter = RecordIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        RecordIter {
            inner: self.entries.iter(),
        }
    }
}

/// Iterator over RecordValue entries, yielding (NameId, &Value).
pub struct RecordIter<'a> {
    inner: std::slice::Iter<'a, (NameId, Value)>,
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = (NameId, &'a Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (*k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for RecordIter<'_> {}
