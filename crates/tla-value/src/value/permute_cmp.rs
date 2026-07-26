// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Lazy permute-compare for symmetry canonicalization.
//!
//! `permute_cmp` computes `self.permute(perm).cmp(other)` WITHOUT materializing
//! the fully-permuted value in the common cases. Symmetry canonicalization
//! compares every permutation of a state against the running lexicographic
//! minimum; most permutations lose at the first differing slot, so building the
//! whole permuted value (allocations, set re-sorts, function rebuilds) is
//! wasted work. This module streams the comparison through the value structure
//! and only materializes the small pieces it must (permuted set elements and
//! permuted function keys, which need re-sorting before they can be compared).
//!
//! # Exactness contract
//!
//! For every value `v`, permutation `p`, and comparand `o`:
//!
//! ```text
//! v.permute_cmp(p, o) == v.permute(p).cmp(o)
//! v.permute_cmp_fast(p, o) == v.permute_fast(p).cmp(o)
//! ```
//!
//! This must hold BIT-EXACTLY (the canonical representative selected by the
//! symmetry reduction must not change), so every arm below either:
//! - replicates the corresponding `permute_impl` + `Value::cmp` pair for a
//!   variant whose permutation preserves the variant and element order
//!   (scalars, tuples, sequences, records, int-functions),
//! - re-sorts only the small permuted portion exactly as the materialized path
//!   would (set elements: `sort` + `dedup`; function keys: stable
//!   `sort_by(key)` — permutations are injective so keys never tie), or
//! - FAILS CLOSED by materializing via `permute_impl` and comparing
//!   (`permute_then_cmp`), which is the contract by definition.
//!
//! Note `Subset` always falls back: permuting a changed `Subset` materializes
//! a concrete `Set`, which compares through different code paths
//! (`cmp_set_like`), so laziness there would have to replicate cross-variant
//! semantics — not worth it for a variant that effectively never appears in
//! reachable states.

use super::cmp_helpers::{cmp_i64_with_value, type_order};
use super::permute::PermLookup;
use super::{FuncValue, IntIntervalFunc, MVPerm, Value};
use crate::rp::Rp as Arc;
use smallvec::SmallVec;
use std::cmp::Ordering;

impl Value {
    /// Compute `self.permute(perm).cmp(other)` lazily (FuncValue lookup).
    pub fn permute_cmp(&self, perm: &FuncValue, other: &Value) -> Ordering {
        crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::PermuteCmp);
        permute_cmp_impl(self, perm, other)
    }

    /// Compute `self.permute_fast(perm).cmp(other)` lazily (MVPerm O(1) lookup).
    pub fn permute_cmp_fast(&self, perm: &MVPerm, other: &Value) -> Ordering {
        crate::churn_stats::churn_count(crate::churn_stats::ChurnSite::PermuteCmp);
        permute_cmp_impl(self, perm, other)
    }
}

/// Fail-closed fallback: materialize the permuted value and compare.
/// Exact by definition of the contract.
#[cold]
fn permute_then_cmp<P: PermLookup>(v: &Value, perm: &P, other: &Value) -> Ordering {
    match v.permute_impl(perm) {
        Some(p) => p.cmp(other),
        None => v.cmp(other),
    }
}

/// Compare a model value (by name) against an arbitrary `Value`, replicating
/// `Value::cmp(&Value::ModelValue(name), other)` without constructing the
/// left-hand `Value` (no `Arc` clone):
/// - `cmp_cross_type` has no ModelValue arms (returns `None`),
/// - `type_order(ModelValue) == 3` and only ModelValue maps to 3, so equal
///   orders imply `cmp_same_type(ModelValue, ModelValue)` — the ptr-eq /
///   name comparison below.
#[inline]
pub(in crate::value) fn cmp_model_value_name(name: &Arc<str>, other: &Value) -> Ordering {
    match other {
        Value::ModelValue(b) => {
            if Arc::ptr_eq(name, b) {
                Ordering::Equal
            } else {
                name.cmp(b)
            }
        }
        _ => 3u8.cmp(&type_order(other)),
    }
}

/// A permuted element that borrows the original when the permutation left it
/// unchanged and owns the rebuilt value only when it actually changed.
/// Avoids per-comparison `Arc` clone/drop churn on the canonicalization hot
/// path; comparisons go through [`Self::get`], so ordering and deduplication
/// behave exactly as on materialized (cloned) values.
enum PermutedElem<'a> {
    Borrowed(&'a Value),
    Owned(Value),
}

impl PermutedElem<'_> {
    #[inline]
    fn get(&self) -> &Value {
        match self {
            PermutedElem::Borrowed(v) => v,
            PermutedElem::Owned(v) => v,
        }
    }
}

fn permute_cmp_impl<P: PermLookup>(v: &Value, perm: &P, other: &Value) -> Ordering {
    match v {
        // Variants `permute_impl` never changes (returns None): the permuted
        // value IS `v`, so compare directly. This covers all primitives,
        // intervals, and the lazy set/function types.
        Value::Bool(_)
        | Value::SmallInt(_)
        | Value::Int(_)
        | Value::String(_)
        | Value::Interval(_)
        | Value::FuncSet(_)
        | Value::RecordSet(_)
        | Value::TupleSet(_)
        | Value::SetCup(_)
        | Value::SetCap(_)
        | Value::SetDiff(_)
        | Value::SetPred(_)
        | Value::KSubset(_)
        | Value::BigUnion(_)
        | Value::StringSet
        | Value::AnySet
        | Value::SeqSet(_)
        | Value::LazyFunc(_)
        | Value::Closure(_) => v.cmp(other),

        // Model values: O(1) permutation lookup, then a scalar compare
        // (allocation-free via the PermLookup hook — MVPerm avoids even the
        // Arc clone of the permuted name).
        Value::ModelValue(name) => perm.permute_cmp_model_value(name, other),

        // Tuples: permutation preserves length and element order.
        Value::Tuple(t) => match other {
            // cmp_same_type(Tuple, Tuple) → <[Value]>::cmp: lexicographic
            // element-wise, then length. (The Arc::ptr_eq fast path returns
            // Equal, consistent with element-wise equality.)
            Value::Tuple(o) => cmp_permuted_elems_then_len(t.as_ref(), perm, o.as_ref()),
            // cmp_cross_type(Tuple, Seq) → a.iter().cmp(b.iter()):
            // lexicographic with length tiebreak — same shape.
            Value::Seq(o) => cmp_permuted_elems_then_len(t.as_ref(), perm, o.flat_slice()),
            // cmp_cross_type(Tuple, Func/IntFunc) → cmp_tuple_with_func(_intfunc).
            Value::Func(o) => cmp_permuted_tuple_with_func(t.as_ref(), perm, o),
            Value::IntFunc(o) => cmp_permuted_tuple_with_intfunc(t.as_ref(), perm, o),
            // cmp_cross_type(Tuple, Record): both denote functions; the ordering
            // is decided by length + key-type (permutation-invariant), so defer
            // to the materialize-then-cmp oracle to stay in lockstep with
            // Value::cmp's cmp_intkeyed_with_record.
            Value::Record(_) => permute_then_cmp(v, perm, other),
            // Otherwise orders differ (Tuple is 4); fall through to type_order.
            _ => 4u8.cmp(&type_order(other)),
        },

        // Sequences: permutation preserves length and element order.
        Value::Seq(s) => match other {
            // cmp_same_type(Seq, Seq) → SeqValue::cmp: flat_slice zip
            // element-wise, then length. (ptr_eq fast path consistent.)
            Value::Seq(o) => cmp_permuted_elems_then_len(s.flat_slice(), perm, o.flat_slice()),
            // cmp_cross_type(Seq, Tuple) → a.iter().cmp(b.iter()).
            Value::Tuple(o) => cmp_permuted_elems_then_len(s.flat_slice(), perm, o.as_ref()),
            Value::Func(o) => cmp_permuted_tuple_with_func(s.flat_slice(), perm, o),
            Value::IntFunc(o) => cmp_permuted_tuple_with_intfunc(s.flat_slice(), perm, o),
            // cmp_cross_type(Seq, Record): decided by length + key-type; defer to
            // the oracle to match Value::cmp.
            Value::Record(_) => permute_then_cmp(v, perm, other),
            _ => 5u8.cmp(&type_order(other)),
        },

        // Records: NameId keys are unchanged by permutation; values permute
        // in place. Mirrors RecordValue::cmp: lexicographic over (key, value)
        // pairs in canonical field order, comparing keys by their NAME STRINGS
        // (never by NameId numeric value — interning order, run-dependent),
        // then length.
        Value::Record(r) => match other {
            Value::Record(o) => {
                for ((ka, va), (kb, vb)) in r.iter().zip(o.iter()) {
                    if ka != kb {
                        match tla_core::name_id_str_cmp(ka, kb) {
                            Ordering::Equal => {}
                            ord => return ord,
                        }
                    }
                    match permute_cmp_impl(va, perm, vb) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                }
                r.len().cmp(&o.len())
            }
            // cmp_cross_type(Record, Func) exists (string-keyed iteration);
            // rare in the symmetry loop (variants almost always match) —
            // fail closed.
            Value::Func(_) => permute_then_cmp(v, perm, other),
            // cmp_cross_type(Record, Tuple/Seq/IntFunc): decided by length +
            // key-type; defer to the oracle to match Value::cmp.
            Value::Tuple(_) | Value::Seq(_) | Value::IntFunc(_) => permute_then_cmp(v, perm, other),
            _ => 6u8.cmp(&type_order(other)),
        },

        // Sets: permuting elements changes their sort order, so the permuted
        // set compares by its re-sorted content. Materialize ONLY the
        // permuted elements (sort + dedup exactly like SortedSet
        // normalization), never the SortedSet/Value wrappers.
        Value::Set(s) => match other {
            Value::Set(o) => {
                // NOTE(value-canon): an earlier draft routed this arm through
                // memoized whole-set materialization. That was a measured
                // REGRESSION (2.3x wall, 2x RSS on MultiPaxos): growing
                // message sets get a FRESH storage allocation per successor,
                // so set-level memo entries almost never hit while pinning a
                // full materialized permuted set per (state, perm) — cap
                // thrash plus retention bloat. The streaming path below stays;
                // its per-ELEMENT `permute_impl` calls hit the memo for the
                // shared record/function elements, which is where the reuse
                // actually is.
                let raw = s.raw_slice();
                // Borrow unchanged elements; own only the rebuilt ones.
                let mut permuted: SmallVec<[PermutedElem<'_>; 16]> =
                    SmallVec::with_capacity(raw.len());
                let mut changed = false;
                for e in raw {
                    match e.permute_impl(perm) {
                        Some(p) => {
                            changed = true;
                            permuted.push(PermutedElem::Owned(p));
                        }
                        None => permuted.push(PermutedElem::Borrowed(e)),
                    }
                }
                if !changed {
                    // permute_impl returns None → permuted value is `v`
                    // itself (shares storage); Value::cmp applies its own
                    // ptr_eq fast path.
                    return v.cmp(other);
                }
                // Replicate SortedSet::normalized_elements_from_raw: stable
                // sort then dedup (keeps first of each equal run). Interning
                // does not affect comparison results (Ord is consistent
                // with Eq).
                permuted.sort_by(|a, b| a.get().cmp(b.get()));
                permuted.dedup_by(|a, b| a.get() == b.get());
                // Replicate cmp_same_type(Set, Set) element loop over
                // normalized slices (fresh storage — ptr_eq is false on the
                // materialized path whenever `changed`).
                let mut ai = permuted.iter();
                let mut bi = o.iter();
                loop {
                    match (ai.next(), bi.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.get().cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            // Set vs Interval/Subset/lazy-set/other: fail closed (rare —
            // the comparand in the symmetry loop has matching variant).
            _ => permute_then_cmp(v, perm, other),
        },

        // Functions: permutation can change KEYS (re-sort required) and
        // values. Materialize only the permuted keys; compare values lazily.
        Value::Func(f) => match other {
            Value::Func(o) => {
                // (permuted key — borrowed when unchanged, original value
                // ref), in original key order.
                let mut entries: SmallVec<[(PermutedElem<'_>, &Value); 16]> =
                    SmallVec::with_capacity(f.domain_len());
                let mut key_changed = false;
                for (k, val) in f.iter() {
                    match k.permute_impl(perm) {
                        Some(pk) => {
                            key_changed = true;
                            entries.push((PermutedElem::Owned(pk), val));
                        }
                        None => entries.push((PermutedElem::Borrowed(k), val)),
                    }
                }
                if key_changed {
                    // Replicate permute_impl's
                    // `permuted_entries.sort_by(|a, b| a.0.cmp(&b.0))`:
                    // stable sort by key; permutations are injective, so
                    // permuted keys are distinct and the order is unique.
                    entries.sort_by(|a, b| a.0.get().cmp(b.0.get()));
                }
                // Replicate FuncValue::cmp: zip in sorted order, key then
                // value, then domain length. (Its ptr_eq fast path returns
                // Equal, consistent with element-wise equality; the
                // materialized path takes the element loop whenever the
                // permutation changed anything.)
                for ((pk, pv), (ok, ov)) in entries.iter().zip(o.iter()) {
                    match pk.get().cmp(ok) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                    match permute_cmp_impl(pv, perm, ov) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                }
                entries.len().cmp(&o.domain_len())
            }
            // Func vs Tuple/Seq/Record/IntFunc cross-type: fail closed.
            _ => permute_then_cmp(v, perm, other),
        },

        // Int-interval functions: integer domain unchanged; values permute
        // in place.
        Value::IntFunc(f) => match other {
            Value::IntFunc(o) => {
                // Replicate cmp_same_type(IntFunc, IntFunc): min, max, then
                // values lexicographically with length tiebreak
                // (Iterator::cmp). The ptr_eq fast path returns Equal,
                // consistent with element-wise equality.
                match f.min.cmp(&o.min) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
                match f.max.cmp(&o.max) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
                for (a, b) in f.values.iter().zip(o.values.iter()) {
                    match permute_cmp_impl(a, perm, b) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                }
                f.values.len().cmp(&o.values.len())
            }
            // cmp_cross_type(IntFunc, Tuple/Seq/Func): fail closed.
            _ => permute_then_cmp(v, perm, other),
        },

        // SUBSET values change variant when permuted (Subset → Set) and
        // route through cmp_set_like — fail closed.
        Value::Subset(_) => permute_then_cmp(v, perm, other),

        // Bag: permutation may change both elements and variant (a changed
        // Bag permutes to a general Func) — fail closed via the
        // materialize-then-compare oracle.
        Value::Bag(_) => permute_then_cmp(v, perm, other),
    }
}

/// Replicates lexicographic element comparison with length tiebreak, with the
/// left side permuted lazily. Matches both `<[Value]>::cmp` (slice Ord) and
/// `Iterator::cmp` semantics, which agree.
fn cmp_permuted_elems_then_len<P: PermLookup>(lhs: &[Value], perm: &P, rhs: &[Value]) -> Ordering {
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        match permute_cmp_impl(a, perm, b) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }
    lhs.len().cmp(&rhs.len())
}

/// Replicates `cmp_helpers::cross_type::cmp_tuple_with_func` with the tuple
/// elements permuted lazily. Tuple keys are 1..n integers — unchanged by
/// permutation.
fn cmp_permuted_tuple_with_func<P: PermLookup>(
    tuple: &[Value],
    perm: &P,
    func: &FuncValue,
) -> Ordering {
    let min_len = tuple.len().min(func.domain_len());
    for (i, (func_key, func_val)) in func.mapping_iter().take(min_len).enumerate() {
        match cmp_i64_with_value((i + 1) as i64, func_key) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match permute_cmp_impl(&tuple[i], perm, func_val) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }
    tuple.len().cmp(&func.domain_len())
}

/// Replicates `cmp_helpers::cross_type::cmp_tuple_with_intfunc` with the
/// tuple elements permuted lazily.
fn cmp_permuted_tuple_with_intfunc<P: PermLookup>(
    tuple: &[Value],
    perm: &P,
    intfunc: &IntIntervalFunc,
) -> Ordering {
    let min_len = tuple.len().min(intfunc.values.len());
    for (i, (tuple_val, int_val)) in tuple
        .iter()
        .zip(intfunc.values.iter())
        .take(min_len)
        .enumerate()
    {
        let tuple_key = (i + 1) as i64;
        let intfunc_key = intfunc.min + i as i64;
        match tuple_key.cmp(&intfunc_key) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match permute_cmp_impl(tuple_val, perm, int_val) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }
    tuple.len().cmp(&intfunc.values.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rp::Rp;

    fn mv(name: &str) -> Value {
        Value::try_model_value(name).unwrap()
    }

    /// Build a permutation FuncValue + MVPerm pair from (from, to) name pairs.
    fn perm_pair(mapping: &[(&str, &str)]) -> (FuncValue, MVPerm) {
        let mut entries: Vec<(Value, Value)> = mapping
            .iter()
            .map(|(from, to)| (mv(from), mv(to)))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let fv = FuncValue::from_sorted_entries(entries);
        let mvp = MVPerm::from_func_value(&fv).expect("valid model-value permutation");
        (fv, mvp)
    }

    /// Exhaustive oracle check: lazy result must equal materialize-then-cmp
    /// for every (value, other) pair in the matrix, for both lookup kinds.
    fn assert_matches_oracle(values: &[Value], fv: &FuncValue, mvp: &MVPerm) {
        for v in values {
            for o in values {
                let oracle = v.permute(fv).cmp(o);
                assert_eq!(
                    v.permute_cmp(fv, o),
                    oracle,
                    "permute_cmp mismatch: v={v:?} other={o:?}"
                );
                let oracle_fast = v.permute_fast(mvp).cmp(o);
                assert_eq!(
                    v.permute_cmp_fast(mvp, o),
                    oracle_fast,
                    "permute_cmp_fast mismatch: v={v:?} other={o:?}"
                );
                assert_eq!(oracle, oracle_fast, "permute vs permute_fast disagree");
            }
        }
    }

    #[test]
    fn permute_cmp_matches_materialized_oracle_across_shapes() {
        // MVPerm::apply resolves model values through the global registry;
        // serialize with tests that clear/reset it.
        let _lock = crate::value::lock_intern_state();
        let (fv, mvp) = perm_pair(&[("a", "b"), ("b", "c"), ("c", "a")]);

        let set_abc = Value::set([mv("a"), mv("b"), mv("c")]);
        let set_ab = Value::set([mv("a"), mv("b")]);
        let set_bc = Value::set([mv("b"), mv("c")]);
        let set_ints = Value::set([Value::SmallInt(1), Value::SmallInt(2)]);
        let empty_set = Value::set(Vec::<Value>::new());

        let func_mv_keys = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (mv("a"), Value::SmallInt(1)),
            (mv("b"), Value::SmallInt(2)),
            (mv("c"), Value::SmallInt(3)),
        ])));
        let func_mv_both = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (mv("a"), mv("c")),
            (mv("b"), mv("a")),
        ])));
        let func_int_keys = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::SmallInt(1), mv("b")),
            (Value::SmallInt(2), mv("a")),
        ])));
        let func_set_vals = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (mv("a"), set_ab.clone()),
            (mv("b"), set_bc.clone()),
            (mv("c"), empty_set.clone()),
        ])));

        let tuple = Value::Tuple(vec![mv("a"), mv("b")].into());
        let tuple2 = Value::Tuple(vec![mv("b"), mv("c")].into());
        let tuple_long = Value::Tuple(vec![mv("a"), mv("b"), mv("c")].into());
        let seq = Value::seq(vec![mv("b"), mv("a")]);
        let seq2 = Value::seq(vec![mv("a"), mv("b")]);
        let intfunc = Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![mv("a"), mv("c")])));
        let intfunc2 = Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![mv("b"), mv("b")])));
        let interval = Value::Interval(Rp::new(super::super::IntervalValue::new(
            1.into(),
            3.into(),
        )));

        let rec = Value::record(vec![("x", mv("a")), ("y", set_ab.clone())]);
        let rec2 = Value::record(vec![("x", mv("c")), ("y", set_bc.clone())]);

        let nested = Value::set([tuple.clone(), tuple2.clone()]);
        let func_of_funcs = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (mv("a"), func_int_keys.clone()),
            (mv("b"), func_mv_both.clone()),
        ])));

        let values = [
            Value::Bool(true),
            Value::Bool(false),
            Value::SmallInt(-3),
            Value::SmallInt(7),
            Value::string("s"),
            mv("a"),
            mv("b"),
            mv("d"), // outside the permutation domain
            set_abc,
            set_ab,
            set_bc,
            set_ints,
            empty_set,
            func_mv_keys,
            func_mv_both,
            func_int_keys,
            func_set_vals,
            tuple,
            tuple2,
            tuple_long,
            seq,
            seq2,
            intfunc,
            intfunc2,
            interval,
            rec,
            rec2,
            nested,
            func_of_funcs,
        ];

        assert_matches_oracle(&values, &fv, &mvp);
    }

    #[test]
    fn permute_cmp_identity_perm_matches() {
        let _lock = crate::value::lock_intern_state();
        // Permutation that maps everything to itself (all identity lookups).
        let (fv, mvp) = perm_pair(&[("a", "a"), ("b", "b")]);
        let values = [
            mv("a"),
            Value::set([mv("a"), mv("b")]),
            Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![(
                mv("a"),
                mv("b"),
            )]))),
            Value::Tuple(vec![mv("b"), mv("a")].into()),
        ];
        assert_matches_oracle(&values, &fv, &mvp);
    }

    #[test]
    fn permute_cmp_unnormalized_set_matches() {
        let _lock = crate::value::lock_intern_state();
        let (fv, mvp) = perm_pair(&[("a", "b"), ("b", "a")]);
        // Build an unnormalized set with duplicates via from_vec
        // (Value::set defers normalization).
        let dup_set = Value::set([mv("b"), mv("a"), mv("b")]);
        let other_set = Value::set([mv("a"), mv("b")]);
        let values = [dup_set, other_set, Value::set([mv("a")])];
        assert_matches_oracle(&values, &fv, &mvp);
    }
}
