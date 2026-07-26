// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Per-allocation permutation memo for symmetry canonicalization (value-canon).
//!
//! Symmetry canonicalization permutes the SAME compound value allocations
//! (message records, node functions, growing message sets) once per
//! permutation for EVERY distinct state — on MultiPaxos ~20% of total wall
//! was `permute_cmp`/`permute_impl` rebuilding and dropping identical
//! permuted records millions of times. This memo caches
//! `permute_impl(value, perm)` results keyed by
//! `(stable allocation pointer, MVPerm serial)`.
//!
//! # Soundness contract (read before changing)
//!
//! A pointer-keyed cache is sound ONLY because every entry PINS its source:
//!
//! 1. **Key half 1 — allocation pointer.** Each entry stores a clone of the
//!    source `Value`, keeping the keyed allocation alive for the entry's
//!    lifetime. The pointer can therefore never be recycled (no ABA) while
//!    the entry exists; entries and pins drop together on `clear`.
//! 2. **Pinning also freezes content.** All in-place mutation of the keyed
//!    structures (`FuncValue` values array, `RecordValue` entries,
//!    `SeqValue` elements) goes through `Rp::make_mut`/`Rp::get_mut`, which
//!    only mutates when the refcount is 1. The pin holds a refcount, so any
//!    later "mutation" is forced copy-on-write into a NEW allocation — the
//!    pinned allocation's content is immutable for the entry's lifetime.
//!    (`SortedSet` storage is immutable after construction; its interior
//!    `AtomicU64`/`OnceLock` caches are content-neutral.)
//! 3. **Key half 2 — MVPerm serial.** Serials are process-unique and never
//!    reused (`MVPerm::memo_serial`), so an entry can only be observed by
//!    lookups against the exact permutation content it was computed with.
//! 4. **Purity.** `permute_impl` is a pure function of (value content,
//!    permutation content), so the memoized result is bit-equivalent to a
//!    recompute, including the `None` = "unchanged" case.
//!
//! The memo is thread-local (the symmetry canonicalization path runs on the
//! sequential BFS thread; other threads simply get their own maps) and
//! size-capped: on reaching the cap the whole map is cleared (it is a pure
//! cache — clearing is always sound and also releases the pins).
//!
//! Kill switch: `TY_NO_VALUE_FP_CACHE=1` disables the memo entirely.

use super::Value;
use crate::rp::Rp;
use rustc_hash::FxHashMap;
use std::cell::RefCell;

/// Entry: pinned source (keeps the key pointer alive and its content frozen)
/// plus the memoized `permute_impl` result.
struct MemoEntry {
    /// Pin. Never read; exists to hold refcounts. See module docs.
    _pin: Value,
    /// Memoized `permute_impl` result (`None` = permutation leaves the value
    /// unchanged, mirroring `permute_impl`'s contract).
    permuted: Option<Value>,
}

thread_local! {
    static PERMUTE_MEMO: RefCell<FxHashMap<(usize, u32), MemoEntry>> =
        RefCell::new(FxHashMap::default());
}

/// Master kill switch for the value-canon fast paths (permute memo and
/// cached-fingerprint equality fast-reject): `TY_NO_VALUE_FP_CACHE=1`.
#[inline(always)]
pub(crate) fn value_canon_disabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_NO_VALUE_FP_CACHE").map_or(false, |v| !v.is_empty() && v != "0")
    })
}

/// Entry cap. Each entry holds two `Value` handles (~small constant + map
/// overhead); the default cap bounds worst-case memo memory to tens of MB.
/// Reaching the cap clears the map (sound: pure cache).
fn memo_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("TY_PERMUTE_MEMO_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1 << 17)
    })
}

/// Stable content-identity pointer for `value`'s backing allocation, or
/// `None` for kinds the memo does not cover.
///
/// Covered kinds: Record / Seq — long-lived allocations with SMALL permuted
/// results and high cross-state reuse (message records recur across the
/// whole frontier; client-event sequences rarely change). Deliberately
/// EXCLUDED:
/// - Set: growing state sets get a fresh storage allocation per successor,
///   so set-level entries rarely hit while pinning a whole materialized
///   permuted set each — measured 2.3x wall / 2x RSS regression on
///   MultiPaxos. Set ELEMENTS (records) are memoized via the recursion.
/// - Func: state-variable functions (e.g. one-record-per-replica node maps)
///   mostly get fresh allocations per state, so whole-func entries are
///   low-reuse ~0.3-1KB pins that inflate the cap window; their function
///   VALUES (records) are memoized via the recursion instead.
/// The returned pointer uniquely identifies CONTENT while the source is
/// pinned (see module docs).
#[inline]
fn memo_key_ptr(value: &Value) -> Option<usize> {
    match value {
        Value::Record(r) => Some(r.storage_ptr_identity()),
        Value::Seq(s) => Some(Rp::as_ptr(s).cast::<()>() as usize),
        _ => None,
    }
}

impl Value {
    /// Memoized wrapper around [`Value::permute_impl_uncached`] for MVPerm
    /// permutations of compound values. Falls through to the uncached path
    /// for non-covered kinds, non-serial perms, or when killed by
    /// `TY_NO_VALUE_FP_CACHE=1`.
    #[inline]
    pub(in crate::value) fn permute_impl<P: super::permute::PermLookup>(
        &self,
        perm: &P,
    ) -> Option<Value> {
        if value_canon_disabled() {
            return self.permute_impl_uncached(perm);
        }
        let Some(serial) = perm.memo_serial() else {
            return self.permute_impl_uncached(perm);
        };
        let Some(ptr) = memo_key_ptr(self) else {
            return self.permute_impl_uncached(perm);
        };
        let key = (ptr, serial);

        // Fast path: hit. The borrow is dropped before any recursion.
        let hit = PERMUTE_MEMO.with(|m| m.borrow().get(&key).map(|entry| entry.permuted.clone()));
        if let Some(permuted) = hit {
            crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::PermuteMemoHit);
            return permuted;
        }

        // Miss: compute WITHOUT holding the borrow (the compute recurses into
        // child `permute_impl` calls, which take their own borrows).
        crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::PermuteMemoMiss);
        let permuted = self.permute_impl_uncached(perm);
        PERMUTE_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() >= memo_cap() {
                // Pure cache: clearing is sound and releases all pins.
                m.clear();
            }
            m.insert(
                key,
                MemoEntry {
                    _pin: self.clone(),
                    permuted: permuted.clone(),
                },
            );
        });
        permuted
    }
}

/// Clear the permutation memo (releases all pinned source values).
///
/// Serials are never reused, so stale entries can never be OBSERVED across
/// runs — clearing is purely a memory-release hook for run boundaries.
pub fn clear_permute_memo() {
    PERMUTE_MEMO.with(|m| m.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::super::{MVPerm, Value};
    use super::*;

    fn mvperm_swap(a: &str, b: &str) -> MVPerm {
        use crate::value::interned_model_value;
        let va = interned_model_value(a).unwrap();
        let vb = interned_model_value(b).unwrap();
        let mut fb = crate::value::FuncBuilder::new();
        fb.insert(va.clone(), vb.clone());
        fb.insert(vb, va);
        MVPerm::from_func_value(&fb.build()).unwrap()
    }

    #[test]
    fn memoized_permute_matches_uncached_and_hits_on_repeat() {
        // Serialize against other intern-state-mutating tests. Deliberately
        // does NOT clear the model value registry: unique names suffice, and
        // clearing would invalidate other tests' MVPerms if they race.
        let _guard = crate::value::lock_intern_state();
        let perm = mvperm_swap("mv_pm_a", "mv_pm_b");
        let a = crate::value::interned_model_value("mv_pm_a").unwrap();
        let b = crate::value::interned_model_value("mv_pm_b").unwrap();

        // A record set containing model values — the hot MultiPaxos shape.
        // The set itself is NOT memo-keyed (fresh-allocation churn), but its
        // record elements are memoized through the recursion.
        let rec = Value::record(vec![("x", a.clone()), ("y", Value::SmallInt(1))]);
        let set = Value::set([rec.clone(), Value::record(vec![("x", b.clone())])]);

        let uncached = set.permute_impl_uncached(&perm);
        let memo1 = set.permute_impl(&perm);
        let memo2 = set.permute_impl(&perm);
        assert_eq!(memo1, uncached, "memoized result must equal uncached");
        assert_eq!(memo2, uncached, "repeat lookup must be stable");

        // Top-level memo-keyed kind (Record): hit path returns the SAME
        // shared allocation and equal content.
        let rec_uncached = rec.permute_impl_uncached(&perm);
        let rec_memo1 = rec.permute_impl(&perm);
        let rec_memo2 = rec.permute_impl(&perm);
        assert_eq!(rec_memo1, rec_uncached);
        assert_eq!(rec_memo2, rec_uncached);
        if let (Some(m1), Some(m2)) = (&rec_memo1, &rec_memo2) {
            assert!(m1.ptr_eq(m2), "memo hit must share the cached allocation");
        }

        // Unchanged-value case memoizes None identically.
        let plain = Value::set([Value::SmallInt(3)]);
        assert_eq!(plain.permute_impl(&perm), None);
        assert_eq!(plain.permute_impl_uncached(&perm), None);
        clear_permute_memo();
    }
}
