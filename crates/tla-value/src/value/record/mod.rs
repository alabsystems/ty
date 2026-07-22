// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Array-backed record type for TLA+ records with `NameId`-keyed field names.
//!
//! `RecordValue` keeps fields in a sorted contiguous array for fast lookup,
//! cache-friendly iteration, and copy-on-write `EXCEPT` updates.
//!
//! # Canonical field order (SOUNDNESS)
//!
//! Entries are sorted by field-name STRING (byte-lexicographic via
//! `tla_core::name_id_str_cmp`), NOT by NameId numeric value. In TLA+ records
//! ARE functions with string domains, and string-keyed `FuncValue`s sort their
//! domains by string content. Every order-observing consumer (cross-type
//! eq/cmp against `FuncValue`, fingerprinting, `Value::hash`, TLC-order
//! comparison, display) relies on record iteration agreeing with that order.
//!
//! NameId numeric order is interning order — run-dependent and semantically
//! meaningless. Sorting entries by NameId caused the BagsTest `h (+) h`
//! nondeterministic-verdict bug: `eq_record_with_func` zipped NameId-ordered
//! record entries pairwise against string-sorted func entries, so the verdict
//! flipped with the per-run interning order of the field names.

mod lookup;
mod mutation;

use super::functions::FP_UNSET;
use crate::rp::Rp;
use super::Value;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use tla_core::{intern_name, name_id_str_cmp, NameId};

/// Array-backed record for TLA+ records with `NameId`-keyed field names.
///
/// Fields stay sorted by field-name STRING (canonical record order; see module
/// docs) inside an `Arc<Vec<_>>`, which gives cache-friendly iteration and
/// copy-on-write updates. Field lookup by NameId still uses O(1) integer
/// equality per probe (linear scan) for small records.
pub struct RecordValue {
    /// Unique `(field_id, value)` pairs sorted by field-name string;
    /// `pub(super)` for `record_impls.rs`.
    pub(super) entries: crate::rp::Rp<Vec<(NameId, Value)>>,
    /// Additive fingerprint cache for nested `EXCEPT` / state dedup updates.
    additive_fp: AtomicU64,
}

/// Check that entries are sorted by the canonical record field order
/// (field-name string, see module docs). Strictly sorted: duplicate field
/// names are tolerated (adjacent) but never reordered.
#[inline]
pub(super) fn entries_canonically_sorted(entries: &[(NameId, Value)]) -> bool {
    entries
        .windows(2)
        .all(|w| name_id_str_cmp(w[0].0, w[1].0) != std::cmp::Ordering::Greater)
}

impl Clone for RecordValue {
    fn clone(&self) -> Self {
        RecordValue {
            entries: self.entries.clone(),
            additive_fp: AtomicU64::new(self.additive_fp.load(AtomicOrdering::Relaxed)),
        }
    }
}

impl RecordValue {
    /// Find the index of a field in the entries.
    /// Linear scan for <= 8 entries, binary search (by canonical string order)
    /// for larger. Part of #3073.
    #[inline]
    fn find_field_idx(&self, field: NameId) -> Option<usize> {
        if self.entries.len() <= 8 {
            self.entries.iter().position(|(k, _)| *k == field)
        } else {
            // Entries are sorted by field-name string, not NameId numeric value.
            self.entries
                .binary_search_by(|(k, _)| name_id_str_cmp(*k, field))
                .ok()
        }
    }

    /// Reset caches after a mutation, optionally seeding the additive cache.
    /// Part of #3221: mirrors FuncValue::reset_caches_with_additive.
    #[inline]
    fn reset_caches_with_additive(&mut self, additive_fp: Option<u64>) {
        self.additive_fp
            .store(additive_fp.unwrap_or(FP_UNSET), AtomicOrdering::Relaxed);
    }

    /// Create an empty record
    pub(crate) fn new() -> Self {
        RecordValue {
            entries: crate::rp::Rp::new(Vec::new()),
            additive_fp: AtomicU64::new(FP_UNSET),
        }
    }

    /// Create a record from pre-sorted field-value pairs (NameId keys).
    ///
    /// Caller should ensure entries are sorted by the CANONICAL record field
    /// order: field-name string (byte-lexicographic), NOT NameId numeric
    /// value. Unsorted input is detected and re-sorted (fail-safe), with a
    /// debug assertion so misuse is caught in debug builds.
    pub fn from_sorted_entries(mut entries: Vec<(NameId, Value)>) -> Self {
        if !entries_canonically_sorted(&entries) {
            debug_assert!(
                false,
                "RecordValue::from_sorted_entries: entries not sorted by field-name string"
            );
            entries.sort_by(|a, b| name_id_str_cmp(a.0, b.0));
        }
        RecordValue {
            entries: crate::rp::Rp::new(entries),
            additive_fp: AtomicU64::new(FP_UNSET),
        }
    }

    /// Create a record from field-value pairs in ANY order (NameId keys),
    /// sorting them into the canonical record field order (field-name string).
    pub fn from_entries(mut entries: Vec<(NameId, Value)>) -> Self {
        entries.sort_by(|a, b| name_id_str_cmp(a.0, b.0));
        RecordValue {
            entries: crate::rp::Rp::new(entries),
            additive_fp: AtomicU64::new(FP_UNSET),
        }
    }

    /// Create a record from pre-sorted field-value pairs (string keys, interned)
    /// Caller should ensure entries are sorted by field name; the order is
    /// PRESERVED (it already matches the canonical record field order).
    /// Unsorted input is detected and re-sorted (fail-safe).
    pub fn from_sorted_str_entries(mut entries: Vec<(Arc<str>, Value)>) -> Self {
        if !entries.windows(2).all(|w| w[0].0 <= w[1].0) {
            debug_assert!(
                false,
                "RecordValue::from_sorted_str_entries: entries not sorted by field name"
            );
            entries.sort_by(|a, b| a.0.cmp(&b.0));
        }
        let id_entries: Vec<(NameId, Value)> = entries
            .into_iter()
            .map(|(k, v)| (intern_name(&k), v))
            .collect();
        RecordValue {
            entries: crate::rp::Rp::new(id_entries),
            additive_fp: AtomicU64::new(FP_UNSET),
        }
    }

    /// Get the number of fields
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the record is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the cached additive fingerprint if already computed.
    /// Part of #3221: commutative hash for state-level dedup.
    #[inline]
    pub fn get_additive_fp(&self) -> Option<u64> {
        let cached = self.additive_fp.load(AtomicOrdering::Relaxed);
        (cached != FP_UNSET).then_some(cached)
    }

    /// Cache the additive fingerprint. Returns the fingerprint.
    /// Part of #3221: commutative hash for state-level dedup.
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

    /// Check if two RecordValues share the same underlying storage (pointer equality)
    #[inline]
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        crate::rp::Rp::ptr_eq(&self.entries, &other.entries)
    }

    /// Whether the backing `entries` Arc is shared (strong count > 1).
    ///
    /// When shared, dropping this `RecordValue` only decrements the Arc's
    /// refcount — no inner `Value` drops fire — so the iterative-drop machinery
    /// in `value::drop` is pure overhead and can be skipped. Part of
    /// value-churn-reduction: extends the `is_shared_arc` drop fast path to the
    /// inline (non-Arc-wrapped) `Value::Record` variant.
    #[inline]
    pub(crate) fn is_storage_shared(&self) -> bool {
        crate::rp::Rp::strong_count(&self.entries) > 1
    }

    /// Return a stable identity key for the underlying storage Arc.
    ///
    /// Part of #3337: Used by `WorkerFpMemo` to cache fingerprints by
    /// pointer identity without needing `AtomicU64` embedded caches.
    #[inline]
    pub fn storage_ptr_identity(&self) -> usize {
        crate::rp::Rp::as_ptr(&self.entries).cast::<()>() as usize
    }
}
