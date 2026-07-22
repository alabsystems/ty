// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `impl Value` tuple/sequence/function coercions.

use super::super::cmp_helpers::eq_tuple_elements_with_value;
use super::super::*;

impl Value {
    /// Test whether this value equals the TLA+ tuple described by `elements`
    /// without constructing a [`Value::Tuple`].
    ///
    /// This is the general form of [`Self::equals_empty_tuple`]. It preserves
    /// TLA+ function equality across tuple, sequence, function,
    /// integer-interval-function, record, and compact-bag representations.
    /// The comparison borrows both operands and does not allocate.
    #[inline]
    pub fn equals_tuple_elements(&self, elements: &[Value]) -> bool {
        eq_tuple_elements_with_value(elements, self)
    }

    /// Test whether this value equals the empty tuple without constructing one.
    ///
    /// This preserves TLA+ function equality across the tuple, sequence,
    /// function, integer-interval function, record, and compact-bag runtime
    /// representations. It borrows the value and neither allocates nor clones
    /// an `Arc`.
    #[inline]
    pub fn equals_empty_tuple(&self) -> bool {
        self.equals_tuple_elements(&[])
    }

    /// Extract elements from a Seq or Tuple.
    /// Both Seq and Tuple are indexed collections in TLA+.
    /// Returns Cow::Borrowed for Tuple (which uses Arc<[Value]>)
    /// and Cow::Owned for Seq (which uses im::Vector).
    #[inline]
    pub fn as_seq_or_tuple_elements(&self) -> Option<Cow<'_, [Value]>> {
        match self {
            Value::Seq(s) => Some(Cow::Owned(s.to_vec())),
            Value::Tuple(t) => Some(Cow::Borrowed(t.as_ref())),
            _ => None,
        }
    }

    /// Extract elements from a tuple-like value for tuple-pattern destructuring.
    ///
    /// TLC-parity: TLC's `toTuple()` converts sequence-like functions (domain 1..n)
    /// into tuples. This method implements the same coercion for:
    /// - `Value::Tuple` - direct extraction
    /// - `Value::Seq` - convert to slice
    /// - `Value::IntFunc` with min=1 - sequence-like function
    /// - `Value::Func` with keys exactly 1..n - sequence-like function
    ///
    /// Used by tuple-pattern binding in eval.rs and liveness/ast_to_live.rs.
    pub fn to_tuple_like_elements(&self) -> Option<Cow<'_, [Value]>> {
        match self {
            Value::Tuple(t) => Some(Cow::Borrowed(t.as_ref())),
            Value::Seq(s) => Some(Cow::Owned(s.to_vec())),
            Value::IntFunc(f) if f.min == 1 => {
                // IntFunc with domain 1..n is sequence-like
                Some(Cow::Owned(f.values.to_vec()))
            }
            Value::Func(f) => {
                if f.domain_is_empty() {
                    return Some(Cow::Owned(vec![]));
                }
                if !f.domain_is_sequence() {
                    return None;
                }
                Some(Cow::Owned(f.mapping_values().cloned().collect()))
            }
            // Compact bag: same coercion as its equivalent Func (fail-closed
            // via the cached materialization).
            Value::Bag(b) => {
                let f = b.as_func_value();
                if f.domain_is_empty() {
                    return Some(Cow::Owned(vec![]));
                }
                if !f.domain_is_sequence() {
                    return None;
                }
                Some(Cow::Owned(f.mapping_values().cloned().collect()))
            }
            _ => None,
        }
    }

    /// Extract the SeqValue from a Seq
    #[inline]
    pub fn as_seq_value(&self) -> Option<&SeqValue> {
        match self {
            Value::Seq(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow the inner [`FuncValue`] if this is a `Value::Func`, else `None`.
    ///
    /// Unlike [`Value::to_func_coerced`], this does not coerce tuples, sequences,
    /// or records; it only matches values already stored as explicit functions.
    /// A compact `Bag` IS an explicit function (alternate representation):
    /// it returns its cached materialized general form — fail-closed, so every
    /// `as_func` caller handles bags soundly without a dedicated arm.
    pub fn as_func(&self) -> Option<&FuncValue> {
        match self {
            Value::Func(f) => Some(f),
            Value::Bag(b) => Some(b.as_func_value()),
            _ => None,
        }
    }

    /// Borrow the inner [`BagValue`] if this is a compact `Value::Bag`.
    pub fn as_bag(&self) -> Option<&BagValue> {
        match self {
            Value::Bag(b) => Some(b),
            _ => None,
        }
    }

    /// Coerce function-like values (Func, Tuple, Seq, IntFunc, Record) to FuncValue.
    /// This is used for operations that accept any function-like type (e.g., Bags).
    /// Returns None for non-function-like types.
    pub fn to_func_coerced(&self) -> Option<FuncValue> {
        match self {
            Value::Func(f) => Some((**f).clone()),
            Value::Bag(b) => Some(b.to_func_value()),
            Value::IntFunc(f) => Some(f.to_func_value()),
            Value::Tuple(elems) => {
                // Convert to function with domain 1..n
                let entries: Vec<(Value, Value)> = elems
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (Value::SmallInt((i + 1) as i64), v.clone()))
                    .collect();
                Some(FuncValue::from_sorted_entries(entries))
            }
            Value::Seq(seq) => {
                // Convert to function with domain 1..n
                let entries: Vec<(Value, Value)> = seq
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (Value::SmallInt((i + 1) as i64), v.clone()))
                    .collect();
                Some(FuncValue::from_sorted_entries(entries))
            }
            Value::Record(rec) => {
                // Convert to function with string domain (resolve NameId to string)
                let mut entries: Vec<(Value, Value)> = rec
                    .iter()
                    .map(|(k, v)| (Value::string(tla_core::resolve_name_id(k)), v.clone()))
                    .collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                Some(FuncValue::from_sorted_entries(entries))
            }
            _ => None,
        }
    }

    /// View this value as a slice of sequence elements, if it is sequence-like.
    ///
    /// In TLA+ sequences and tuples share `<<...>>` syntax, and interning may
    /// substitute semantically-equal function variants (e.g. `Seq([])` ↔
    /// `Func([])`), so this also accepts tuples and `1..n`-domain functions.
    /// Returns a [`Cow`] to avoid copying when the elements are already
    /// contiguously stored. Returns `None` for non-sequence-like values.
    pub fn as_seq(&self) -> Option<Cow<'_, [Value]>> {
        // In TLA+, sequences and tuples share the same <<...>> syntax
        // and sequence operations work on both.
        // Also accept Func/IntFunc — the SET_INTERN_TABLE can substitute
        // semantically-equal variants (e.g., Seq([]) ↔ Func([])) (#1713)
        self.as_seq_or_tuple_elements()
            .or_else(|| self.to_tuple_like_elements())
    }

    /// Borrow the inner [`RecordValue`] if this is a `Value::Record`, else `None`.
    pub fn as_record(&self) -> Option<&RecordValue> {
        match self {
            Value::Record(r) => Some(r),
            _ => None,
        }
    }

    /// Borrow the inner element array if this is a `Value::Tuple`, else `None`.
    ///
    /// Matches only explicit tuples; use [`Value::as_seq`] to also accept
    /// sequence-like values.
    pub fn as_tuple(&self) -> Option<&Arc<[Value]>> {
        match self {
            Value::Tuple(t) => Some(t),
            _ => None,
        }
    }
}
