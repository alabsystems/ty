// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compact bitset for a set of integers drawn from a small contiguous window.
//!
//! Part of the compound-value codegen program: TLA+ sets whose elements are
//! small integers over a bounded universe (`SUBSET (0..N)`, a piece over a
//! fixed board, a handful of process ids) are the common case in the
//! model-checker's hot loop, and a `SortedSet` backed by `Arc<[Value]>` pays an
//! allocation + O(n) merge for every `∪`/`∩`/`∖`/`∈`. When every element lands
//! in a 64-wide window `[base, base + 64)` those operations collapse to a single
//! machine word of bitwise logic — which is also what trust-cg can lower to
//! native code (unlike `Arc<[Value]>` set merges).
//!
//! This module is deliberately self-contained and `unsafe`-free so it can be
//! exhaustively unit-tested (and Miri-clean by construction) before it is wired
//! into `SetStorage`. Correctness contract that the integration relies on:
//!
//! * [`SmallIntBitset::iter`] yields elements in **ascending** order — identical
//!   to the canonical order of the equivalent sorted/deduped `Vec<Value>`, so a
//!   bitset set and a Vec set with the same members must be observationally
//!   indistinguishable (same iteration, same length, and therefore the same
//!   state fingerprint used for dedup). A divergence here would be a soundness
//!   bug (miscounted states), so it is covered by property tests.

/// A set of `i64` values, all within a single 64-wide window `[base, base+64)`,
/// stored as one machine word. Empty sets carry `base = 0`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SmallIntBitset {
    /// Lowest representable value; bit `i` (LSB-first) means `base + i` present.
    base: i64,
    /// Occupancy bitmask; `bits == 0` iff the set is empty.
    bits: u64,
}

impl SmallIntBitset {
    /// The empty set.
    #[inline]
    pub(crate) fn empty() -> Self {
        Self { base: 0, bits: 0 }
    }

    /// Build from an ascending, de-duplicated slice of integers, returning
    /// `None` if the span `max - min` does not fit in a 64-wide window (in which
    /// case the caller keeps the `Arc<[Value]>` representation).
    ///
    /// `sorted` MUST already be strictly ascending (the caller normalizes); this
    /// is asserted in debug builds. An empty slice yields the empty set.
    pub(crate) fn from_sorted_unique(sorted: &[i64]) -> Option<Self> {
        let (&min, &max) = match (sorted.first(), sorted.last()) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => return Some(Self::empty()),
        };
        debug_assert!(
            sorted.windows(2).all(|w| w[0] < w[1]),
            "SmallIntBitset::from_sorted_unique requires strictly ascending input"
        );
        // `max - min` can overflow i64 for pathological inputs; checked_sub keeps
        // us fail-closed (fall back to the Vec representation).
        let span = max.checked_sub(min)?;
        if span >= 64 {
            return None;
        }
        let mut bits = 0u64;
        for &v in sorted {
            // `v - min` is in `0..64` by the span check above.
            bits |= 1u64 << (v - min) as u32;
        }
        Some(Self { base: min, bits })
    }

    /// Cardinality (population count).
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.bits.count_ones() as usize
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Membership test.
    #[inline]
    pub(crate) fn contains(&self, value: i64) -> bool {
        match value.checked_sub(self.base) {
            Some(off) if (0..64).contains(&off) => (self.bits >> off) & 1 == 1,
            _ => false,
        }
    }

    /// Smallest element, or `None` if empty.
    #[inline]
    pub(crate) fn min(&self) -> Option<i64> {
        if self.bits == 0 {
            None
        } else {
            Some(self.base + self.bits.trailing_zeros() as i64)
        }
    }

    /// Largest element, or `None` if empty.
    #[inline]
    pub(crate) fn max(&self) -> Option<i64> {
        if self.bits == 0 {
            None
        } else {
            Some(self.base + (63 - self.bits.leading_zeros()) as i64)
        }
    }

    /// Re-express `self` and `other` against a common base so their bit columns
    /// line up, or `None` if the union of their spans exceeds a 64-wide window.
    /// Empty operands adopt the other's base (identity for the pointwise ops).
    fn aligned_bits(&self, other: &Self) -> Option<(i64, u64, u64)> {
        if self.bits == 0 {
            return Some((other.base, 0, other.bits));
        }
        if other.bits == 0 {
            return Some((self.base, self.bits, 0));
        }
        let base = self.base.min(other.base);
        let hi = self.max().unwrap().max(other.max().unwrap());
        if hi.checked_sub(base)? >= 64 {
            return None;
        }
        let a = self.bits << (self.base - base) as u32;
        let b = other.bits << (other.base - base) as u32;
        Some((base, a, b))
    }

    /// Set union (`self ∪ other`), or `None` if the result cannot fit one window.
    pub(crate) fn union(&self, other: &Self) -> Option<Self> {
        let (base, a, b) = self.aligned_bits(other)?;
        Some(Self::normalized(base, a | b))
    }

    /// Set intersection (`self ∩ other`). Always representable (subset of a
    /// 64-window operand), so never `None`.
    pub(crate) fn intersect(&self, other: &Self) -> Self {
        // Intersection never widens the span; align against whichever base is
        // safe. If they cannot align in one window the intersection is empty
        // (disjoint windows share no elements).
        match self.aligned_bits(other) {
            Some((base, a, b)) => Self::normalized(base, a & b),
            None => Self::empty(),
        }
    }

    /// Set difference (`self ∖ other`). Always representable.
    pub(crate) fn difference(&self, other: &Self) -> Self {
        match self.aligned_bits(other) {
            Some((base, a, b)) => Self::normalized(base, a & !b),
            None => *self, // disjoint windows: nothing removed
        }
    }

    /// `self ⊆ other`.
    pub(crate) fn is_subset(&self, other: &Self) -> bool {
        match self.aligned_bits(other) {
            Some((_, a, b)) => a & !b == 0,
            None => self.bits == 0,
        }
    }

    /// Canonicalize so an empty mask always carries `base = 0`, keeping the
    /// representation of a given set unique (needed for `Eq`/`Hash` to agree
    /// with extensional equality).
    #[inline]
    fn normalized(base: i64, bits: u64) -> Self {
        if bits == 0 {
            Self::empty()
        } else {
            // Re-anchor base at the lowest set bit so equal sets share one form.
            let shift = bits.trailing_zeros();
            Self {
                base: base + shift as i64,
                bits: bits >> shift,
            }
        }
    }

    /// Iterate members in ascending order (canonical sorted-Vec order).
    pub(crate) fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        let mut remaining = self.bits;
        let base = self.base;
        std::iter::from_fn(move || {
            if remaining == 0 {
                None
            } else {
                let off = remaining.trailing_zeros();
                remaining &= remaining - 1; // clear lowest set bit
                Some(base + off as i64)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(vals: &[i64]) -> SmallIntBitset {
        // vals must be ascending+unique for from_sorted_unique.
        SmallIntBitset::from_sorted_unique(vals).expect("fits in window")
    }

    fn collect(s: &SmallIntBitset) -> Vec<i64> {
        s.iter().collect()
    }

    #[test]
    fn empty_and_singletons() {
        let e = SmallIntBitset::empty();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
        assert_eq!(collect(&e), Vec::<i64>::new());
        assert_eq!(e.min(), None);
        assert_eq!(e.max(), None);

        let s = of(&[7]);
        assert_eq!(s.len(), 1);
        assert!(s.contains(7));
        assert!(!s.contains(6));
        assert_eq!(s.min(), Some(7));
        assert_eq!(s.max(), Some(7));
        assert_eq!(collect(&s), vec![7]);
    }

    #[test]
    fn iter_is_ascending_and_matches_input() {
        let s = of(&[-5, 0, 3, 40]);
        assert_eq!(collect(&s), vec![-5, 0, 3, 40]);
        assert_eq!(s.len(), 4);
        assert_eq!(s.min(), Some(-5));
        assert_eq!(s.max(), Some(40));
    }

    #[test]
    fn out_of_window_rejected() {
        assert!(SmallIntBitset::from_sorted_unique(&[0, 64]).is_none());
        assert!(SmallIntBitset::from_sorted_unique(&[0, 63]).is_some());
    }

    #[test]
    fn union_intersect_diff_subset() {
        let a = of(&[1, 2, 3, 10]);
        let b = of(&[2, 3, 4]);
        assert_eq!(collect(&a.union(&b).unwrap()), vec![1, 2, 3, 4, 10]);
        assert_eq!(collect(&a.intersect(&b)), vec![2, 3]);
        assert_eq!(collect(&a.difference(&b)), vec![1, 10]);
        assert!(!a.is_subset(&b));
        assert!(of(&[2, 3]).is_subset(&a));
        assert!(SmallIntBitset::empty().is_subset(&a));
    }

    #[test]
    fn ops_with_different_bases() {
        let a = of(&[100, 101]);
        let b = of(&[101, 102]);
        assert_eq!(collect(&a.union(&b).unwrap()), vec![100, 101, 102]);
        assert_eq!(collect(&a.intersect(&b)), vec![101]);
        // Disjoint far-apart windows.
        let far = of(&[1000]);
        assert_eq!(collect(&a.intersect(&far)), Vec::<i64>::new());
        assert_eq!(collect(&a.difference(&far)), vec![100, 101]);
    }

    #[test]
    fn eq_and_normalized_form_is_canonical() {
        // Same members built via different ops must be Eq (canonical base).
        let via_union = of(&[5]).union(&of(&[5])).unwrap();
        let direct = of(&[5]);
        assert_eq!(via_union, direct);
        // Difference emptying a set yields the canonical empty (base 0).
        let emptied = of(&[9]).difference(&of(&[9]));
        assert_eq!(emptied, SmallIntBitset::empty());
    }

    #[test]
    fn exhaustive_small_agreement_with_vec_semantics() {
        // Cross-check every subset of {0..6} against Vec set semantics for the
        // three pointwise ops — the soundness-critical invariant.
        for am in 0u32..128 {
            for bm in 0u32..128 {
                let av: Vec<i64> = (0..7).filter(|i| am & (1 << i) != 0).collect();
                let bv: Vec<i64> = (0..7).filter(|i| bm & (1 << i) != 0).collect();
                let a = of(&av);
                let b = of(&bv);
                let mut u: Vec<i64> = av.iter().chain(&bv).copied().collect();
                u.sort_unstable();
                u.dedup();
                let inter: Vec<i64> = av.iter().copied().filter(|x| bv.contains(x)).collect();
                let diff: Vec<i64> = av.iter().copied().filter(|x| !bv.contains(x)).collect();
                assert_eq!(collect(&a.union(&b).unwrap()), u);
                assert_eq!(collect(&a.intersect(&b)), inter);
                assert_eq!(collect(&a.difference(&b)), diff);
                assert_eq!(a.is_subset(&b), av.iter().all(|x| bv.contains(x)));
            }
        }
    }
}
