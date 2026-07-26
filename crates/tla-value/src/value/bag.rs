// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compact bag (multiset) representation — `BagValue`.
//!
//! A TLA+ bag is a function from elements to positive integer counts
//! (Bags.tla). The general representation is `Value::Func` with `Int` values,
//! which pays per-entry `Value` clone/drop Arc traffic, two array allocations,
//! and a full additive-fingerprint walk on EVERY `BagAdd`/`BagRemove`
//! (EWD998PCal: one bag update per successor, ~1.4M successors).
//!
//! `BagValue` stores the same mathematical value as:
//! - `elems`: the distinct elements, strictly sorted by `Value::cmp`
//!   (identical order to the equivalent `FuncValue`'s domain), shared via Arc
//!   across bag updates that do not change the element set;
//! - `counts`: plain `i64` counts (parallel array, all > 0);
//! - `additive_fp`: the precomputed state-dedup fingerprint, maintained
//!   incrementally and BIT-IDENTICAL to `compute_func_additive_fp` on the
//!   equivalent `FuncValue` (soundness: a state must fingerprint identically
//!   whether its bag is compact or general, or dedup splits/merges states).
//!
//! Fail-closed everywhere: any operation the compact representation cannot
//! answer exactly falls back to the lazily materialized general `FuncValue`
//! (`as_func_value`), which is cached per bag instance. Construction is gated
//! by the `TY_COMPACT_BAG` env kill switch (set to `0`/`off` to force the
//! general representation for differential validation).

use super::{FuncValue, SortedSet, Value};
use crate::dedup_fingerprint::{
    additive_entry_hash_from_fps, splitmix64, state_value_fingerprint, ADDITIVE_FUNC_SEED,
};
use crate::rp::Rp as Arc;
use crate::rp::Rp;
use std::sync::OnceLock;

/// Kill switch: `TY_COMPACT_BAG=0|off|false` forces the general Func
/// representation (constructors return `None`/`Err`). Read once per process.
pub fn compact_bags_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("TY_COMPACT_BAG").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// Compact multiset: sorted distinct elements + parallel positive counts.
///
/// Invariants (upheld by every constructor):
/// - `elems` strictly sorted by `Value::cmp`, no duplicates;
/// - `counts.len() == elems.len()`, every count `>= 1`;
/// - `additive_fp == ADDITIVE_FUNC_SEED + splitmix64(len) + Σ
///   additive_entry_hash(elem_i, SmallInt(count_i))` — the exact state-dedup
///   fingerprint of the equivalent `FuncValue`;
/// - every element fingerprints successfully via `state_value_fingerprint`
///   (checked at construction; fail-closed otherwise).
pub struct BagValue {
    /// Distinct elements, strictly sorted by `Value::cmp` (== the equivalent
    /// FuncValue's domain order, == DOMAIN iteration order).
    elems: Arc<[Value]>,
    /// counts[i] > 0 is the multiplicity of elems[i].
    counts: Box<[i64]>,
    /// Precomputed additive (state-dedup) fingerprint. See invariants above.
    additive_fp: u64,
    /// Cached `DOMAIN` result: `Value::Set` sharing `elems` (zero-copy).
    domain_set: OnceLock<Value>,
    /// Cached general representation for fail-closed operations.
    func: OnceLock<Rp<FuncValue>>,
}

impl std::fmt::Debug for BagValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BagValue(")?;
        for (i, (e, c)) in self.entries().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{e:?} :> {c}")?;
        }
        write!(f, ")")
    }
}

/// Compute the additive entry hash for `(elem, SmallInt(count))` from the
/// element's precomputed state fingerprint. Must stay bit-identical to
/// `additive_entry_hash(elem, &Value::SmallInt(count))`.
#[inline]
fn count_entry_hash(elem_fp: u64, count: i64) -> u64 {
    let count_fp = state_value_fingerprint(&Value::SmallInt(count))
        .expect("SmallInt fingerprint is total (lookup table + i64 fallback)");
    additive_entry_hash_from_fps(elem_fp, count_fp)
}

/// Full additive fingerprint over sorted (elem, count) pairs.
/// Bit-identical to `compute_func_additive_fp` on the equivalent FuncValue.
/// Returns `None` if any element exceeds TLC fingerprint limits (fail-closed).
fn compute_bag_additive_fp(elems: &[Value], counts: &[i64]) -> Option<u64> {
    let mut fp = ADDITIVE_FUNC_SEED;
    fp = fp.wrapping_add(splitmix64(elems.len() as u64));
    for (e, &c) in elems.iter().zip(counts.iter()) {
        let elem_fp = state_value_fingerprint(e).ok()?;
        fp = fp.wrapping_add(count_entry_hash(elem_fp, c));
    }
    Some(fp)
}

impl BagValue {
    fn from_parts(elems: Arc<[Value]>, counts: Box<[i64]>, additive_fp: u64) -> BagValue {
        debug_assert_eq!(elems.len(), counts.len());
        debug_assert!(counts.iter().all(|&c| c > 0), "bag counts must be positive");
        debug_assert!(
            elems.windows(2).all(|w| w[0] < w[1]),
            "bag elems must be strictly sorted by Value::cmp"
        );
        debug_assert_eq!(
            Some(additive_fp),
            compute_bag_additive_fp(&elems, &counts),
            "bag additive fingerprint must match full recomputation"
        );
        BagValue {
            elems,
            counts,
            additive_fp,
            domain_set: OnceLock::new(),
            func: OnceLock::new(),
        }
    }

    /// The shared empty bag (== the empty function).
    pub fn empty_arc() -> Rp<BagValue> {
        static EMPTY: OnceLock<Rp<BagValue>> = OnceLock::new();
        Arc::clone(EMPTY.get_or_init(|| {
            let fp = compute_bag_additive_fp(&[], &[]).expect("empty bag fingerprint is total");
            Arc::new(BagValue::from_parts(Arc::from([]), Box::from([]), fp))
        }))
    }

    /// Build a compact bag from `(key, count)` entries sorted strictly by key
    /// (`Value::cmp`). Fails closed (returns the entries back) when:
    /// - compact bags are disabled via the kill switch,
    /// - any count is not an integer in `1..=i64::MAX`,
    /// - any key exceeds TLC fingerprint limits.
    pub fn try_from_entries(entries: Vec<(Value, Value)>) -> Result<BagValue, Vec<(Value, Value)>> {
        if !compact_bags_enabled() {
            return Err(entries);
        }
        debug_assert!(
            entries.windows(2).all(|w| w[0].0 < w[1].0),
            "try_from_entries requires strictly-sorted keys"
        );
        let mut counts: Vec<i64> = Vec::with_capacity(entries.len());
        let mut fp = ADDITIVE_FUNC_SEED;
        fp = fp.wrapping_add(splitmix64(entries.len() as u64));
        for (k, v) in &entries {
            let c = match v.as_i64() {
                Some(c) if c > 0 => c,
                _ => return Err(entries),
            };
            let Ok(elem_fp) = state_value_fingerprint(k) else {
                return Err(entries);
            };
            fp = fp.wrapping_add(count_entry_hash(elem_fp, c));
            counts.push(c);
        }
        let elems: Arc<[Value]> = entries.into_iter().map(|(k, _)| k).collect();
        Ok(BagValue::from_parts(elems, counts.into(), fp))
    }

    /// Build a compact bag from a general function, if eligible.
    pub fn try_from_func(f: &FuncValue) -> Option<BagValue> {
        if !compact_bags_enabled() {
            return None;
        }
        let mut counts: Vec<i64> = Vec::with_capacity(f.domain_len());
        let mut fp = ADDITIVE_FUNC_SEED;
        fp = fp.wrapping_add(splitmix64(f.domain_len() as u64));
        for (k, v) in f.mapping_iter() {
            let c = match v.as_i64() {
                Some(c) if c > 0 => c,
                _ => return None,
            };
            let elem_fp = state_value_fingerprint(k).ok()?;
            fp = fp.wrapping_add(count_entry_hash(elem_fp, c));
            counts.push(c);
        }
        Some(BagValue::from_parts(f.domain_arc(), counts.into(), fp))
    }

    /// Number of DISTINCT elements (== domain size of the equivalent Func).
    #[inline]
    pub fn len(&self) -> usize {
        self.elems.len()
    }

    /// True when the bag has no elements (== the empty function).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    /// Distinct elements, sorted by `Value::cmp`.
    #[inline]
    pub fn elems(&self) -> &[Value] {
        &self.elems
    }

    /// Parallel positive counts.
    #[inline]
    pub fn counts(&self) -> &[i64] {
        &self.counts
    }

    /// Iterate `(elem, count)` pairs in domain (Value::cmp) order — identical
    /// to the equivalent FuncValue's `mapping_iter` order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&Value, i64)> + '_ {
        self.elems
            .iter()
            .zip(self.counts.iter())
            .map(|(e, &c)| (e, c))
    }

    /// True when the two bags share the same elems allocation (=> equal domains).
    #[inline]
    pub fn elems_ptr_eq(&self, other: &BagValue) -> bool {
        std::ptr::eq(
            Arc::as_ptr(&self.elems) as *const u8,
            Arc::as_ptr(&other.elems) as *const u8,
        )
    }

    /// Position of `key` in the element array, if present.
    #[inline]
    fn position(&self, key: &Value) -> Option<usize> {
        match self.elems.len() {
            0 => None,
            // Small bags: linear scan beats binary search dispatch.
            1..=4 => self.elems.iter().position(|e| e == key),
            _ => self.elems.binary_search_by(|e| e.cmp(key)).ok(),
        }
    }

    /// Multiplicity of `key` (0 when absent).
    #[inline]
    pub fn count_of(&self, key: &Value) -> i64 {
        self.position(key).map_or(0, |i| self.counts[i])
    }

    /// Function application `bag[key]` — `SmallInt(count)` or `None` when the
    /// key is not in the domain (matches Func's not-in-domain error contract).
    #[inline]
    pub fn apply(&self, key: &Value) -> Option<Value> {
        self.position(key).map(|i| Value::SmallInt(self.counts[i]))
    }

    /// True when `key` is in the bag's domain.
    #[inline]
    pub fn domain_contains(&self, key: &Value) -> bool {
        self.position(key).is_some()
    }

    /// Share the element array (sorted, unique — upholds the
    /// `SortedSet::Normalized` invariant, same as `FuncValue::domain_arc`).
    #[inline]
    pub fn domain_arc(&self) -> Arc<[Value]> {
        Arc::clone(&self.elems)
    }

    /// `DOMAIN bag` as a cached `Value::Set` sharing the element array.
    /// Iteration/CHOOSE order == `DOMAIN` of the equivalent Func (both are the
    /// same `Value::cmp`-sorted sequence).
    pub fn domain_set_value(&self) -> Value {
        self.domain_set
            .get_or_init(|| {
                Value::from_sorted_set(SortedSet::from_normalized_arc_shared(self.domain_arc()))
            })
            .clone()
    }

    /// Total number of copies: Σ counts (BagCardinality). `None` on overflow.
    pub fn cardinality(&self) -> Option<i64> {
        let mut total: i64 = 0;
        for &c in self.counts.iter() {
            total = total.checked_add(c)?;
        }
        Some(total)
    }

    /// The state-dedup (additive) fingerprint — bit-identical to the
    /// equivalent FuncValue's `compute_func_additive_fp`.
    #[inline]
    pub fn additive_fp(&self) -> u64 {
        self.additive_fp
    }

    /// The general representation, materialized once and cached.
    /// The returned FuncValue is the SAME mathematical value; its entries are
    /// `(elems[i], SmallInt(counts[i]))` in the same order.
    pub fn as_func_value(&self) -> &Rp<FuncValue> {
        self.func.get_or_init(|| {
            let entries: Vec<(Value, Value)> = self
                .entries()
                .map(|(e, c)| (e.clone(), Value::SmallInt(c)))
                .collect();
            let func = FuncValue::from_sorted_entries(entries);
            // Seed the additive cache so the materialized func fingerprints
            // in O(1) with the exact same value.
            func.cache_additive_fp(self.additive_fp);
            Arc::new(func)
        })
    }

    /// Owned general representation (for `to_func_coerced`).
    pub fn to_func_value(&self) -> FuncValue {
        (**self.as_func_value()).clone()
    }

    /// `BagAdd(self, elem)`: multiplicity of `elem` + 1.
    /// `None` when the compact rep cannot answer exactly (count overflow, or a
    /// new element that fails fingerprint limits) — caller falls back to the
    /// general path.
    pub fn bag_add(&self, elem: &Value) -> Option<BagValue> {
        match self.position(elem) {
            Some(i) => {
                let old = self.counts[i];
                let new = old.checked_add(1)?;
                let elem_fp = state_value_fingerprint(&self.elems[i]).ok()?;
                let fp = self
                    .additive_fp
                    .wrapping_sub(count_entry_hash(elem_fp, old))
                    .wrapping_add(count_entry_hash(elem_fp, new));
                let mut counts = self.counts.clone();
                counts[i] = new;
                let bag = BagValue {
                    elems: Arc::clone(&self.elems),
                    counts,
                    additive_fp: fp,
                    domain_set: OnceLock::new(),
                    func: OnceLock::new(),
                };
                // Domain unchanged — share the cached DOMAIN set.
                if let Some(ds) = self.domain_set.get() {
                    let _ = bag.domain_set.set(ds.clone());
                }
                debug_assert_eq!(
                    Some(bag.additive_fp),
                    compute_bag_additive_fp(&bag.elems, &bag.counts)
                );
                Some(bag)
            }
            None => {
                let elem_fp = state_value_fingerprint(elem).ok()?;
                let idx = self
                    .elems
                    .binary_search_by(|e| e.cmp(elem))
                    .expect_err("position() reported elem absent");
                let mut elems: Vec<Value> = Vec::with_capacity(self.elems.len() + 1);
                elems.extend_from_slice(&self.elems[..idx]);
                elems.push(elem.clone());
                elems.extend_from_slice(&self.elems[idx..]);
                let mut counts: Vec<i64> = Vec::with_capacity(self.counts.len() + 1);
                counts.extend_from_slice(&self.counts[..idx]);
                counts.push(1);
                counts.extend_from_slice(&self.counts[idx..]);
                let fp = self
                    .additive_fp
                    .wrapping_sub(splitmix64(self.elems.len() as u64))
                    .wrapping_add(splitmix64(elems.len() as u64))
                    .wrapping_add(count_entry_hash(elem_fp, 1));
                let bag = BagValue::from_parts(elems.into(), counts.into(), fp);
                Some(bag)
            }
        }
    }

    /// `BagRemove(self, elem)`: multiplicity - 1, entry dropped at zero.
    /// Returns `None` when `elem` is absent (result == self; caller reuses the
    /// existing value — matches the general path, which rebuilds identical
    /// entries).
    pub fn bag_remove(&self, elem: &Value) -> Option<BagValue> {
        let i = self.position(elem)?;
        let old = self.counts[i];
        let elem_fp = state_value_fingerprint(&self.elems[i])
            .expect("construction invariant: bag elems fingerprint successfully");
        if old > 1 {
            let new = old - 1;
            let fp = self
                .additive_fp
                .wrapping_sub(count_entry_hash(elem_fp, old))
                .wrapping_add(count_entry_hash(elem_fp, new));
            let mut counts = self.counts.clone();
            counts[i] = new;
            let bag = BagValue {
                elems: Arc::clone(&self.elems),
                counts,
                additive_fp: fp,
                domain_set: OnceLock::new(),
                func: OnceLock::new(),
            };
            if let Some(ds) = self.domain_set.get() {
                let _ = bag.domain_set.set(ds.clone());
            }
            debug_assert_eq!(
                Some(bag.additive_fp),
                compute_bag_additive_fp(&bag.elems, &bag.counts)
            );
            Some(bag)
        } else {
            Some(self.remove_entry_at(i, elem_fp, old))
        }
    }

    /// `BagRemoveAll(self, elem)`: drop the entry entirely.
    /// Returns `None` when `elem` is absent (result == self).
    pub fn bag_remove_all(&self, elem: &Value) -> Option<BagValue> {
        let i = self.position(elem)?;
        let elem_fp = state_value_fingerprint(&self.elems[i])
            .expect("construction invariant: bag elems fingerprint successfully");
        Some(self.remove_entry_at(i, elem_fp, self.counts[i]))
    }

    /// Remove the entry at position `i` (count `old`, element fp `elem_fp`).
    fn remove_entry_at(&self, i: usize, elem_fp: u64, old: i64) -> BagValue {
        let mut elems: Vec<Value> = Vec::with_capacity(self.elems.len() - 1);
        elems.extend_from_slice(&self.elems[..i]);
        elems.extend_from_slice(&self.elems[i + 1..]);
        let mut counts: Vec<i64> = Vec::with_capacity(self.counts.len() - 1);
        counts.extend_from_slice(&self.counts[..i]);
        counts.extend_from_slice(&self.counts[i + 1..]);
        let fp = self
            .additive_fp
            .wrapping_sub(splitmix64(self.elems.len() as u64))
            .wrapping_add(splitmix64(elems.len() as u64))
            .wrapping_sub(count_entry_hash(elem_fp, old));
        BagValue::from_parts(elems.into(), counts.into(), fp)
    }

    /// `BagCup(self, other)` — (+): merge with added counts.
    /// `None` on count overflow (fail-closed).
    pub fn bag_cup(&self, other: &BagValue) -> Option<BagValue> {
        // Fast path: identical domains — just add counts positionally.
        if self.elems_ptr_eq(other) {
            let mut counts: Vec<i64> = Vec::with_capacity(self.counts.len());
            for (&a, &b) in self.counts.iter().zip(other.counts.iter()) {
                counts.push(a.checked_add(b)?);
            }
            let fp = compute_bag_additive_fp(&self.elems, &counts)?;
            return Some(BagValue::from_parts(
                Arc::clone(&self.elems),
                counts.into(),
                fp,
            ));
        }
        // General sorted merge.
        let mut elems: Vec<Value> = Vec::with_capacity(self.elems.len() + other.elems.len());
        let mut counts: Vec<i64> = Vec::with_capacity(self.elems.len() + other.elems.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.elems.len() && j < other.elems.len() {
            match self.elems[i].cmp(&other.elems[j]) {
                std::cmp::Ordering::Less => {
                    elems.push(self.elems[i].clone());
                    counts.push(self.counts[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    elems.push(other.elems[j].clone());
                    counts.push(other.counts[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    elems.push(self.elems[i].clone());
                    counts.push(self.counts[i].checked_add(other.counts[j])?);
                    i += 1;
                    j += 1;
                }
            }
        }
        elems.extend_from_slice(&self.elems[i..]);
        counts.extend_from_slice(&self.counts[i..]);
        elems.extend_from_slice(&other.elems[j..]);
        counts.extend_from_slice(&other.counts[j..]);
        let fp = compute_bag_additive_fp(&elems, &counts)?;
        Some(BagValue::from_parts(elems.into(), counts.into(), fp))
    }

    /// `BagDiff(self, other)` — (-): subtract counts, keep positives only.
    pub fn bag_diff(&self, other: &BagValue) -> BagValue {
        let mut elems: Vec<Value> = Vec::with_capacity(self.elems.len());
        let mut counts: Vec<i64> = Vec::with_capacity(self.elems.len());
        for (e, c) in self.entries() {
            // saturating_sub is exact here: counts are >= 1, so a saturated
            // result (i64::MIN clamp) can only occur when the true difference
            // is also negative — and negative differences are dropped.
            let diff = c.saturating_sub(other.count_of(e));
            if diff > 0 {
                elems.push(e.clone());
                counts.push(diff);
            }
        }
        let fp = compute_bag_additive_fp(&elems, &counts)
            .expect("construction invariant: bag elems fingerprint successfully");
        BagValue::from_parts(elems.into(), counts.into(), fp)
    }
}

#[cfg(test)]
mod tests;
