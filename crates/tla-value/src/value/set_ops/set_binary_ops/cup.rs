// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Lazy set union (`SetCupValue`).

use super::super::super::*;
use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

/// A small direct-mapped cache for repeated record membership checks against
/// shared union trees. Entries pin both allocations, so pointer reuse cannot
/// turn an identity match into a stale result.
const MEMBERSHIP_CACHE_LEN: usize = 1 << 12;
/// Revalidate the cache often enough that a low-reuse phase cannot retain a
/// model's transient records for the remainder of the run.
const MEMBERSHIP_CACHE_SAMPLE_LEN: u64 = 1 << 11;
/// Stores are substantially more expensive than misses for cheap union trees.
/// Keep the cache only for the near-constant pairs it was designed for.
const MEMBERSHIP_CACHE_MIN_HIT_PERCENT: u64 = 90;

struct MembershipCacheEntry {
    cup: Rp<SetCupValue>,
    record: RecordValue,
    contains: bool,
}

struct MembershipCache {
    entries: Box<[Option<MembershipCacheEntry>]>,
    probes: u64,
    hits: u64,
    stores: u64,
    window_probes: u64,
    window_hits: u64,
}

impl MembershipCache {
    fn record_probe(&mut self, hit: bool) -> bool {
        self.probes += 1;
        self.hits += u64::from(hit);
        self.window_probes += 1;
        self.window_hits += u64::from(hit);
        if self.window_probes < MEMBERSHIP_CACHE_SAMPLE_LEN {
            return false;
        }

        let profitable =
            self.window_hits * 100 >= self.window_probes * MEMBERSHIP_CACHE_MIN_HIT_PERCENT;
        self.window_probes = 0;
        self.window_hits = 0;
        !profitable
    }
}

impl Drop for MembershipCache {
    fn drop(&mut self) {
        if membership_cache_stats_enabled() {
            eprintln!(
                "SetCup membership cache: probes={} hits={} stores={}",
                self.probes, self.hits, self.stores
            );
        }
    }
}

thread_local! {
    static MEMBERSHIP_CACHE: RefCell<Option<MembershipCache>> = const { RefCell::new(None) };
    static MEMBERSHIP_CACHE_REJECTED: Cell<bool> = const { Cell::new(false) };
}

#[inline(always)]
fn membership_cache_disabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_NO_SETCUP_MEMBERSHIP_CACHE")
            .is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

#[inline(always)]
fn membership_cache_stats_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_SETCUP_MEMBERSHIP_CACHE_STATS")
            .is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

#[inline(always)]
fn membership_cache_slot(cup_identity: usize, record_identity: usize) -> usize {
    let mixed =
        cup_identity.rotate_left(17) ^ record_identity.wrapping_mul(0x9e37_79b9_7f4a_7c15usize);
    (mixed ^ (mixed >> 29)) & (MEMBERSHIP_CACHE_LEN - 1)
}

pub(crate) fn clear_set_cup_membership_cache() {
    MEMBERSHIP_CACHE_REJECTED.with(|rejected| rejected.set(false));
    MEMBERSHIP_CACHE.with(|cache| {
        // Keep reset allocation-free for models that never use this cache and
        // release the bounded slot array between independent checking runs.
        *cache.borrow_mut() = None;
    });
}

fn new_membership_cache() -> MembershipCache {
    MembershipCache {
        entries: std::iter::repeat_with(|| None)
            .take(MEMBERSHIP_CACHE_LEN)
            .collect(),
        probes: 0,
        hits: 0,
        stores: 0,
        window_probes: 0,
        window_hits: 0,
    }
}

/// Lazy set union (S1 \cup S2)
///
/// Membership is computed lazily: v \in S1 \cup S2 iff v \in S1 OR v \in S2
/// Enumeration only happens when both operands are enumerable.
#[derive(Clone)]
pub struct SetCupValue {
    pub(crate) set1: Box<Value>,
    pub(crate) set2: Box<Value>,
}

impl SetCupValue {
    /// Create a reference-counted lazy set union.
    ///
    /// Union values frequently represent cached constant trees. Keeping the
    /// root behind `Rp` makes cloning such a tree constant-time instead of
    /// recursively cloning both boxed operands.
    pub fn new(set1: Value, set2: Value) -> Rp<Self> {
        Rp::new(SetCupValue {
            set1: Box::new(set1),
            set2: Box::new(set2),
        })
    }

    /// Borrow the left operand `S1` of `S1 \cup S2`.
    pub fn set1(&self) -> &Value {
        &self.set1
    }

    /// Borrow the right operand `S2` of `S1 \cup S2`.
    pub fn set2(&self) -> &Value {
        &self.set2
    }

    /// Check if a value is in this union set
    /// v \in S1 \cup S2 iff v \in S1 OR v \in S2
    /// Returns None if membership cannot be determined (e.g. SetPred operand).
    pub(crate) fn contains(&self, v: &Value) -> Option<bool> {
        let in1 = self.set1.set_contains(v)?;
        if in1 {
            return Some(true);
        }
        self.set2.set_contains(v)
    }

    /// Check membership through an allocation-identity cache when both the
    /// union and candidate record can be pinned. Only context-free `Some`
    /// answers are stored; indeterminate membership is always re-evaluated by
    /// the context-aware caller.
    pub(crate) fn contains_shared(cup: &Rp<Self>, v: &Value) -> Option<bool> {
        let Value::Record(record) = v else {
            return cup.contains(v);
        };
        if membership_cache_disabled() {
            return cup.contains(v);
        }
        if MEMBERSHIP_CACHE_REJECTED.with(Cell::get) {
            return cup.contains(v);
        }

        let cup_identity = Rp::as_ptr(cup) as usize;
        let record_identity = record.storage_ptr_identity();
        let slot = membership_cache_slot(cup_identity, record_identity);
        let (hit, reject) = MEMBERSHIP_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let cache = cache.get_or_insert_with(new_membership_cache);
            let hit = cache.entries[slot].as_ref().and_then(|entry| {
                (Rp::ptr_eq(&entry.cup, cup) && entry.record.ptr_eq(record))
                    .then_some(entry.contains)
            });
            let reject = cache.record_probe(hit.is_some());
            (hit, reject)
        });
        if reject {
            // Drop pinned records immediately. The rejection is run-scoped and
            // reset by the normal checker lifecycle.
            MEMBERSHIP_CACHE.with(|cache| *cache.borrow_mut() = None);
            MEMBERSHIP_CACHE_REJECTED.with(|rejected| rejected.set(true));
        }
        if hit.is_some() {
            return hit;
        }

        let contains = cup.contains(v);
        if !reject {
            if let Some(contains) = contains {
                MEMBERSHIP_CACHE.with(|cache| {
                    let mut cache = cache.borrow_mut();
                    let cache = cache.get_or_insert_with(new_membership_cache);
                    cache.entries[slot] = Some(MembershipCacheEntry {
                        cup: cup.clone(),
                        record: record.clone(),
                        contains,
                    });
                    cache.stores += 1;
                });
            }
        }
        contains
    }

    /// Check if the union is enumerable (both operands must be enumerable)
    #[allow(dead_code)] // called via Value enum dispatch
    pub(crate) fn is_enumerable(&self) -> bool {
        self.set1.iter_set().is_some() && self.set2.iter_set().is_some()
    }

    /// Check if the set is empty
    #[allow(dead_code)] // called via Value enum dispatch
    pub(crate) fn is_empty(&self) -> bool {
        // Empty iff both operands are empty
        let e1 = self.set1.set_len().is_some_and(|n| n.is_zero());
        let e2 = self.set2.set_len().is_some_and(|n| n.is_zero());
        e1 && e2
    }

    /// Materialize to a SortedSet with deferred normalization.
    ///
    /// #3073: Collects both operands' iterators into a Vec and defers sort+dedup
    /// to the first observation that requires canonical order (fingerprinting,
    /// comparison, iteration). This matches TLC's `SetEnumValue(vals, false)`.
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        let iter1 = self.set1.iter_set()?;
        let iter2 = self.set2.iter_set()?;
        Some(SortedSet::from_iter(iter1.chain(iter2)))
    }
}

impl fmt::Debug for SetCupValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SetCup({:?}, {:?})", self.set1, self.set2)
    }
}

impl Ord for SetCupValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.set1.cmp(&other.set1) {
            Ordering::Equal => self.set2.cmp(&other.set2),
            ord => ord,
        }
    }
}

impl PartialOrd for SetCupValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SetCupValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SetCupValue {}

impl Hash for SetCupValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "SetCup".hash(state);
        self.set1.hash(state);
        self.set2.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::intern_name;

    #[test]
    fn membership_cache_verifies_identity_and_clears() {
        clear_set_cup_membership_cache();
        let record = RecordValue::from_entries(vec![(intern_name("x"), Value::int(1))]);
        let candidate = Value::Record(record.clone());
        let matching = SetCupValue::new(
            Value::set([candidate.clone()]),
            Value::set(std::iter::empty::<Value>()),
        );
        let non_matching = SetCupValue::new(
            Value::set(std::iter::empty::<Value>()),
            Value::set(std::iter::empty::<Value>()),
        );

        // Put a result for a different union in the exact slot that will be
        // probed. The identity check must reject it rather than returning the
        // deliberately wrong cached answer.
        let slot = membership_cache_slot(
            Rp::as_ptr(&matching) as usize,
            record.storage_ptr_identity(),
        );
        MEMBERSHIP_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let cache = cache.get_or_insert_with(new_membership_cache);
            cache.entries[slot] = Some(MembershipCacheEntry {
                cup: non_matching,
                record: record.clone(),
                contains: false,
            });
        });
        assert_eq!(
            SetCupValue::contains_shared(&matching, &candidate),
            Some(true)
        );
        assert_eq!(
            SetCupValue::contains_shared(&matching, &candidate),
            Some(true)
        );

        clear_set_cup_membership_cache();
        MEMBERSHIP_CACHE.with(|cache| assert!(cache.borrow().is_none()));
    }

    #[test]
    fn membership_cache_policy_requires_near_constant_reuse() {
        let mut cache = new_membership_cache();
        for _ in 1..MEMBERSHIP_CACHE_SAMPLE_LEN {
            assert!(!cache.record_probe(false));
        }
        assert!(cache.record_probe(false));

        let mut cache = new_membership_cache();
        for _ in 1..MEMBERSHIP_CACHE_SAMPLE_LEN {
            assert!(!cache.record_probe(true));
        }
        assert!(!cache.record_probe(true));
        assert_eq!(cache.window_probes, 0);
        assert_eq!(cache.window_hits, 0);

        let required_hits =
            (MEMBERSHIP_CACHE_SAMPLE_LEN * MEMBERSHIP_CACHE_MIN_HIT_PERCENT).div_ceil(100);
        let mut cache = new_membership_cache();
        cache.window_probes = MEMBERSHIP_CACHE_SAMPLE_LEN - 1;
        cache.window_hits = required_hits - 2;
        assert!(cache.record_probe(true));

        let mut cache = new_membership_cache();
        cache.window_probes = MEMBERSHIP_CACHE_SAMPLE_LEN - 1;
        cache.window_hits = required_hits - 1;
        assert!(!cache.record_probe(true));
    }
}
