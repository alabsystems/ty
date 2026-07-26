// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `impl Value` set-like predicates and enumerable-set adapters.

use super::super::*;

impl Value {
    /// Return `true` if this value is a set (materialized, lazy, or an infinite
    /// set model value such as `Nat`/`Int`/`Real`).
    ///
    /// This is a structural check: it does not enumerate or evaluate. For a
    /// finiteness check (which excludes infinite sets), use
    /// [`Value::is_finite_set`].
    pub fn is_set(&self) -> bool {
        match self {
            Value::Set(_)
            | Value::Interval(_)
            | Value::Subset(_)
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
            | Value::SeqSet(_) => true,
            // ModelValue for infinite sets (Nat, Int, Real)
            Value::ModelValue(name) => matches!(name.as_ref(), "Nat" | "Int" | "Real"),
            _ => false,
        }
    }

    /// Check if this value is a finite set.
    ///
    /// Returns `true` for sets with finitely many elements, `false` for infinite
    /// sets (Nat, Int, Real, STRING, Seq(S), AnySet) and non-set values.
    /// Recurses on composite set types matching TLC's `isFinite()` semantics.
    ///
    /// Part of #1508: IsFiniteSet must check semantic finiteness, not just `is_set()`.
    pub fn is_finite_set(&self) -> bool {
        match self {
            // Materialized sets are always finite
            Value::Set(_) | Value::Interval(_) => true,
            // Composite sets: finite iff components are finite
            Value::Subset(sv) => sv.base.is_finite_set(),
            Value::FuncSet(fsv) => fsv.domain.is_finite_set() && fsv.codomain.is_finite_set(),
            Value::RecordSet(rsv) => rsv.fields.values().all(|v| v.is_finite_set()),
            Value::TupleSet(tsv) => tsv.components.iter().all(|c| c.is_finite_set()),
            Value::SetCup(scv) => scv.set1.is_finite_set() && scv.set2.is_finite_set(),
            Value::SetCap(scv) => scv.set1.is_finite_set() || scv.set2.is_finite_set(),
            Value::SetDiff(sdv) => sdv.set1.is_finite_set(),
            Value::SetPred(spv) => spv.source.is_finite_set(),
            Value::KSubset(ksv) => ksv.base.is_finite_set(),
            Value::BigUnion(_) => {
                // Conservative: we cannot recurse into elements without eval context.
                // TLC enumerates elements and checks each — we approximate by checking
                // if the outer set itself is finite (which it must be to enumerate).
                // A finite union of finite sets is finite, but we can't check inner
                // finiteness without evaluation. Return true for now — this matches
                // practical TLC behavior where UNION is only applied to enumerable sets.
                true
            }
            // Known infinite sets
            Value::StringSet | Value::AnySet | Value::SeqSet(_) => false,
            // Nat/Int/Real are infinite sets → false.
            // User model values (symmetry elements) are atoms, not sets → false.
            // TLC: ModelValue is not Enumerable, so isFinite() returns false.
            Value::ModelValue(_) => false,
            // Non-set values
            _ => false,
        }
    }

    /// Return `true` if this is a *lazy* set-like value (i.e. not an already
    /// materialized `Set`/`Interval`) that is enumerable and has an exact, cheaply
    /// known finite cardinality `<= cap`.
    ///
    /// Used by set-filter evaluation to decide whether to eagerly reduce a small
    /// finite domain (e.g. a Cartesian product `{-1,0,1} \X {-1,0,1}`) into a concrete
    /// `Set` instead of building a lazy `SetPredValue`. The lazy path deep-clones the
    /// filter's bound-variable and predicate AST on every evaluation *and* on every
    /// const-set-cache hit (`SetPredValue::clone` deep-copies the captured AST), so for
    /// a repeatedly-filtered *constant* domain it dominates allocation churn. Reducing
    /// yields an identical concrete set that is cheap (`Arc`-backed) to clone and cache.
    ///
    /// The `cap` gate keeps large enumerable domains (powersets, function sets) lazy so
    /// a short-circuiting `\E`/`\A` still avoids forcing predicate evaluation over the
    /// whole domain. Only variants with an O(1)/cheap exact `set_len` qualify.
    pub fn is_small_finite_lazy_set(&self, cap: u64) -> bool {
        match self {
            // Already reducible — the caller handles these directly; no need to
            // route through the lazy-set adapter.
            Value::Set(_) | Value::Interval(_) => false,
            // Keep powersets lazy: eagerly materializing `SUBSET S` risks a 2^n
            // blowup and defeats short-circuit `\E`/`\A` over a powerset. The
            // K-subset and nested-powerset rewrites already handle the important
            // small-powerset filter patterns before this check is consulted.
            Value::Subset(_) => false,
            other => match other.as_lazy_set() {
                Some(ls) if ls.is_enumerable() => {
                    ls.set_len().is_some_and(|n| n <= BigInt::from(cap))
                }
                _ => false,
            },
        }
    }

    /// Whether this value is a direct record-set filter chain supported by the
    /// Init bulk-streaming lane: `RecordSet` or one or more `SetPred` wrappers
    /// whose ultimate source is a `RecordSet`.
    ///
    /// This is deliberately narrower than "contains a RecordSet". Set algebra
    /// wrappers (`SetCup`, `SetCap`, `SetDiff`) retain their established eager
    /// small-set policy and are not claimed by the specialized streaming lane.
    /// Keeping this structural predicate shared prevents evaluator caching and
    /// Init extraction from silently disagreeing about which values are live.
    pub fn is_streamable_record_filter_domain(&self) -> bool {
        match self {
            Value::RecordSet(_) => true,
            Value::SetPred(spv) => spv.source().is_streamable_record_filter_domain(),
            _ => false,
        }
    }

    /// Whether a deferred value owns a snapshot of the current or next state.
    /// Such values must never enter a cross-state cache merely because creating
    /// the lazy wrapper performed no state reads.
    pub fn captures_state_environment(&self) -> bool {
        match self {
            Value::LazyFunc(value) => {
                value.captured_state().is_some() || value.captured_next_state().is_some()
            }
            Value::SetPred(value) => {
                value.captured_state().is_some() || value.captured_next_state().is_some()
            }
            _ => false,
        }
    }

    /// Check if this value is an empty set.
    /// Returns Some(true) for empty, Some(false) for non-empty, None for non-sets.
    /// Part of #343: Optimization for S /= {} patterns.
    pub fn is_empty_set(&self) -> Option<bool> {
        match self {
            Value::Set(s) => Some(s.is_empty()),
            Value::Interval(iv) => Some(iv.low > iv.high), // Empty if low > high
            // For other set types (SetPred, etc), check if first element exists.
            // Critical: use next().is_none() NOT count() == 0 to avoid iterating all elements!
            _ if self.is_set() => self.iter_set().map(|mut it| it.next().is_none()),
            _ => None, // Not a set
        }
    }

    /// Structural, **non-materializing** emptiness test for set-like values.
    ///
    /// Returns `Some(true)`/`Some(false)` when emptiness can be decided from the
    /// value's structure WITHOUT enumerating or materializing any operand;
    /// `None` when it cannot be decided cheaply (the caller must then fall back
    /// to a materializing check such as `set_len`). Every `Some` verdict is
    /// EXACT — it equals `set_len() == 0` for that shape.
    ///
    /// Motivation: `Value::set_len()` on a compound lazy set (`SetCup`, ...) with
    /// no cheap cardinality routes through `to_sorted_set()`, materializing +
    /// dedup-hashing the entire set. Emptiness alone never needs the full set:
    /// `A \cup B` is empty iff BOTH are empty; a record set `[f: S, ...]` is
    /// empty iff SOME field domain is empty; `[S -> T]` is empty iff `S` is
    /// non-empty and `T` is empty. Deciding these recursively over structure
    /// avoids the per-state materialization of large constant record-set unions
    /// in `TypeOK`-style invariants (e.g. `f \in [D -> Block \cup {NoBlock}]`).
    pub fn is_empty_set_structural(&self) -> Option<bool> {
        match self {
            Value::Set(s) => Some(s.is_empty()),
            Value::Interval(iv) => Some(iv.low > iv.high),
            // These are always non-empty:
            //   SUBSET S always contains {}; Seq(S) always contains <<>>;
            //   STRING/AnySet/Nat/Int/Real are (infinite and) non-empty.
            Value::Subset(_) | Value::SeqSet(_) | Value::StringSet | Value::AnySet => Some(false),
            Value::ModelValue(name) if matches!(name.as_ref(), "Nat" | "Int" | "Real") => {
                Some(false)
            }
            // [S -> T]: empty iff S is non-empty AND T is empty.
            // ([ {} -> T ] = { <<>> } is non-empty.)
            Value::FuncSet(fs) => match fs.domain().is_empty_set_structural()? {
                true => Some(false),
                false => fs.codomain().is_empty_set_structural(),
            },
            // [f: S, g: T, ...]: empty iff ANY field domain is empty.
            // The zero-field record set `[]` = { <<>> } (empty record) is non-empty.
            Value::RecordSet(rs) => {
                if rs.fields_len() == 0 {
                    return Some(false);
                }
                let mut any_unknown = false;
                for (_name, set) in rs.fields_iter() {
                    match set.is_empty_set_structural() {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => any_unknown = true,
                    }
                }
                if any_unknown {
                    None
                } else {
                    Some(false)
                }
            }
            // S \X T \X ...: empty iff ANY component is empty.
            Value::TupleSet(ts) => {
                let mut any_unknown = false;
                for comp in ts.components_iter() {
                    match comp.is_empty_set_structural() {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => any_unknown = true,
                    }
                }
                if any_unknown {
                    None
                } else {
                    Some(false)
                }
            }
            // A \cup B: empty iff BOTH are empty.
            Value::SetCup(c) => {
                let e1 = c.set1().is_empty_set_structural();
                let e2 = c.set2().is_empty_set_structural();
                match (e1, e2) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }
            // A \cap B: empty if EITHER is empty; otherwise undecidable here
            // (disjoint non-empty sets also intersect to {}).
            Value::SetCap(c) => {
                if c.set1().is_empty_set_structural() == Some(true)
                    || c.set2().is_empty_set_structural() == Some(true)
                {
                    Some(true)
                } else {
                    None
                }
            }
            // A \ B: empty if A is empty; otherwise undecidable here (A \subseteq B
            // also yields {}).
            Value::SetDiff(d) => {
                if d.set1().is_empty_set_structural() == Some(true) {
                    Some(true)
                } else {
                    None
                }
            }
            // SetPred (needs ctx), BigUnion, KSubset, other ModelValues, non-sets:
            // not cheaply decidable here.
            _ => None,
        }
    }

    /// Borrow the inner [`SortedSet`] if this is an eager `Value::Set`, else `None`.
    ///
    /// Matches only materialized sets; lazy set forms (intervals, powersets,
    /// predicate sets, etc.) return `None` and must be enumerated via
    /// `iter_set` to obtain their elements.
    pub fn as_set(&self) -> Option<&SortedSet> {
        match self {
            Value::Set(s) => Some(s),
            _ => None,
        }
    }

    /// Return a trait object for this value's lazy set implementation, if applicable.
    ///
    /// Covers: Interval, Subset, FuncSet, RecordSet, TupleSet, SetCup, SetCap,
    /// SetDiff, KSubset, BigUnion, SeqSet. Does NOT cover: Set (eager), SetPred
    /// (needs eval context), StringSet, AnySet, ModelValue (special cases).
    pub(crate) fn as_lazy_set(&self) -> Option<&dyn LazySet> {
        match self {
            Value::Interval(v) => Some(v.as_ref()),
            Value::Subset(v) => Some(v),
            Value::FuncSet(v) => Some(v),
            Value::RecordSet(v) => Some(v.as_ref()),
            Value::TupleSet(v) => Some(v.as_ref()),
            Value::SetCup(v) => Some(&**v),
            Value::SetCap(v) => Some(v),
            Value::SetDiff(v) => Some(v),
            Value::KSubset(v) => Some(v),
            Value::BigUnion(v) => Some(v),
            Value::SeqSet(v) => Some(v),
            _ => None,
        }
    }

    /// Check if this value (as a set) contains another value
    /// Works for Set, Interval, Subset, FuncSet, RecordSet, TupleSet, SetCup, SetCap, SetDiff, KSubset, BigUnion, SeqSet types
    pub fn set_contains(&self, v: &Value) -> Option<bool> {
        match self {
            Value::Set(s) => Some(s.contains(v)),
            Value::SetCup(cup) => SetCupValue::contains_shared(cup, v),
            Value::SetPred(_) => None, // Needs eval context
            // StringSet contains all strings
            Value::StringSet => Some(matches!(v, Value::String(_))),
            // AnySet contains all values
            Value::AnySet => Some(true),
            // ModelValue for infinite sets (Nat, Int, Real)
            Value::ModelValue(name) => match name.as_ref() {
                "Nat" => Some(match v {
                    Value::SmallInt(n) => *n >= 0,
                    Value::Int(n) => **n >= BigInt::zero(),
                    _ => false,
                }),
                "Int" => Some(matches!(v, Value::SmallInt(_) | Value::Int(_))),
                "Real" => Some(matches!(v, Value::SmallInt(_) | Value::Int(_))), // Int ⊆ Real
                _ => None, // Other model values are not sets
            },
            _ => self.as_lazy_set()?.set_contains(v),
        }
    }

    /// Try to answer membership for tuple elements without allocating a
    /// temporary `Value::Tuple`.
    ///
    /// `None` means the set representation needs the ordinary materialized
    /// candidate path; callers must then construct the tuple and delegate to
    /// [`Self::set_contains`] to preserve lazy-set and error semantics.
    pub fn try_set_contains_tuple_elements(&self, tuple: &[Value]) -> Option<bool> {
        match self {
            Value::Set(set) => Some(set.contains_tuple_elements(tuple)),
            Value::StringSet => Some(false),
            Value::AnySet => Some(true),
            Value::ModelValue(name) if matches!(name.as_ref(), "Nat" | "Int" | "Real") => {
                Some(false)
            }
            _ => None,
        }
    }

    /// Clone-free variant of [`Self::try_set_contains_tuple_elements`] for the
    /// virtual 2-tuple `<<first, second>>`. Returns `None` for lazy/compound
    /// set representations (caller falls back to the owned-tuple path).
    pub fn try_set_contains_tuple2_refs(&self, first: &Value, second: &Value) -> Option<bool> {
        match self {
            Value::Set(set) => Some(set.contains_tuple2_refs(first, second)),
            Value::StringSet => Some(false),
            Value::AnySet => Some(true),
            Value::ModelValue(name) if matches!(name.as_ref(), "Nat" | "Int" | "Real") => {
                Some(false)
            }
            _ => None,
        }
    }

    /// Convert this set-like value to a SortedSet
    /// Works for Set, Interval, Subset, FuncSet, RecordSet, TupleSet, SetCup, SetCap, SetDiff, KSubset, BigUnion types
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        match self {
            Value::Set(s) => Some((**s).clone()),
            _ => self.as_lazy_set()?.to_sorted_set(),
        }
    }

    /// O(1) check: is this set-like value equal to {1, 2, ..., n}?
    ///
    /// For `Interval` values, this is a direct comparison without materializing
    /// the entire set. Falls back to `to_sorted_set()` for other set types.
    pub fn is_sequence_domain(&self, n: usize) -> bool {
        match self {
            Value::Interval(iv) => {
                let expected_high = match i64::try_from(n) {
                    Ok(h) => h,
                    Err(_) => return false,
                };
                iv.low == BigInt::one() && iv.high == BigInt::from(expected_high)
            }
            Value::Set(s) => s.equals_sequence_domain(n),
            _ => self
                .to_sorted_set()
                .is_some_and(|s| s.equals_sequence_domain(n)),
        }
    }

    /// O(1) check: is this set-like value equal to the integer interval {min, min+1, ..., max}?
    ///
    /// For `Interval` values, this is a direct comparison without materializing
    /// the entire set. Falls back to `to_sorted_set()` for other set types.
    pub fn is_integer_interval(&self, min: i64, max: i64) -> bool {
        match self {
            Value::Interval(iv) => iv.low == BigInt::from(min) && iv.high == BigInt::from(max),
            Value::Set(s) => s.equals_integer_interval(min, max),
            _ => self
                .to_sorted_set()
                .is_some_and(|s| s.equals_integer_interval(min, max)),
        }
    }

    /// Get the number of elements in this set-like value
    pub fn set_len(&self) -> Option<BigInt> {
        match self {
            Value::Set(s) => Some(BigInt::from(s.len())),
            _ => self.as_lazy_set()?.set_len(),
        }
    }

    /// Structural check: does this set-like value have a **record set**
    /// (`[f: S, ...]`) anywhere in its lazy structure — directly, or as an
    /// operand of a `SetCup`/`SetCap`/`SetDiff`, the base of a `SUBSET`, the
    /// codomain/domain of a `[D -> R]`, or a `\X` component?
    ///
    /// Used to decide when eagerly materializing a set *union* would be
    /// expensive (materializing a record set enumerates `Π|field|` records).
    /// It does NOT recurse into the elements of an already-materialized
    /// `Value::Set` — that set is concrete, so re-materializing it is cheap and
    /// there is nothing to keep lazy. The check is O(structure), never O(set).
    pub fn references_record_set(&self) -> bool {
        match self {
            Value::RecordSet(_) => true,
            Value::SetCup(c) => {
                c.set1().references_record_set() || c.set2().references_record_set()
            }
            Value::SetCap(c) => {
                c.set1().references_record_set() || c.set2().references_record_set()
            }
            Value::SetDiff(d) => {
                d.set1().references_record_set() || d.set2().references_record_set()
            }
            Value::Subset(s) => s.base().references_record_set(),
            Value::FuncSet(f) => {
                f.domain().references_record_set() || f.codomain().references_record_set()
            }
            Value::TupleSet(t) => t.components_iter().any(Value::references_record_set),
            _ => false,
        }
    }

    /// Cardinality of this set-like value ONLY when it is O(1) to obtain, i.e.
    /// a materialized `Value::Set` (stored length) or an `Interval` (high−low+1).
    /// Returns `None` for every lazy/compound set (`SetCup`, `FuncSet`,
    /// `RecordSet`, `Subset`, ...) whose exact cardinality would require
    /// enumerating or materializing operands.
    ///
    /// Intended for ORDERING/heuristic callers that must never materialize (a
    /// wrong-but-cheap `None` only changes a tie-break order, never a result).
    /// Contrast with [`Self::set_len`], which materializes compound sets.
    pub fn set_len_if_cheap(&self) -> Option<BigInt> {
        match self {
            Value::Set(s) => Some(BigInt::from(s.len())),
            // IntervalValue::set_len is O(1) (high − low + 1); no materialization.
            Value::Interval(_) => self.as_lazy_set()?.set_len(),
            _ => None,
        }
    }

    /// Iterate over this set-like value.
    ///
    /// Returns boxed iterator for Set, Interval, Subset, FuncSet, RecordSet, TupleSet, SetCup,
    /// SetCap, SetDiff, KSubset, BigUnion, and (in the degenerate case) SeqSet types when
    /// enumerable.
    pub fn iter_set(&self) -> Option<Box<dyn Iterator<Item = Value> + '_>> {
        match self {
            Value::Set(s) => Some(Box::new(s.iter().cloned())),
            _ => self.as_lazy_set()?.set_iter(),
        }
    }

    /// Iterate over this set-like value, returning a fully owned iterator.
    ///
    /// Part of #3978: Unlike `iter_set()` which returns an iterator borrowing `self`,
    /// this returns a `'static` iterator that owns its data. This enables streaming
    /// iteration in `SetPredStreamIter` where the source iterator must outlive the
    /// source value reference.
    ///
    /// For types with native owned iterators (FuncSet's odometer, Interval, Subset,
    /// KSubset), delegates to `LazySet::set_iter_owned()` which returns a truly lazy
    /// iterator without collecting. For other types, falls back to collecting through
    /// `iter_set()`.
    pub fn iter_set_owned(&self) -> Option<Box<dyn Iterator<Item = Value>>> {
        match self {
            Value::Set(s) => {
                let elements: Vec<Value> = s.iter().cloned().collect();
                Some(Box::new(elements.into_iter()))
            }
            _ => self.as_lazy_set()?.set_iter_owned(),
        }
    }
}
