// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Concrete-data classification for `Value`.
//!
//! `Value::is_concrete_data()` returns `true` iff the value is *pure data*:
//! it contains, transitively, no stored expressions or captured evaluation
//! environments. For such values, structural equality (`Value::eq`) implies
//! full semantic interchangeability: every operation the evaluator or the
//! bytecode VM performs on the value is a deterministic function of the value
//! itself and never consults ambient evaluation state (`EvalCtx` state
//! bindings, definition scopes, ...).
//!
//! This is the fail-closed admission predicate for value-keyed verdict
//! caches (the implied-action verdict cache in `tla-check`): a cache keyed by
//! `Value` components may only admit components where equal keys guarantee
//! equal downstream behavior. Expression-bearing values break that guarantee:
//!
//! * `Closure` — evaluates its stored body through `EvalCtx` on application;
//!   two closures compare equal only by identity, but even identity equality
//!   does not pin down behavior when the body reads live state variables.
//! * `SetPred` — lazy `{x \in S : P(x)}` whose membership test evaluates the
//!   stored predicate through `EvalCtx`.
//! * `LazyFunc` — lazy function whose applications evaluate a stored body
//!   through `EvalCtx`.
//!
//! Everything else is data: scalars, model values, and the concrete/lazy
//! collection families whose payloads are nested `Value`s only (recursively
//! checked).
//!
//! The match below is intentionally EXHAUSTIVE (no `_` arm) inside the
//! defining crate: adding a new `Value` variant fails compilation here and
//! forces an explicit classification, so the cache admission predicate can
//! never silently admit a future expression-bearing variant.

use super::{lookup_model_value_index, lookup_tlc_string_token, Value};
use tla_core::resolve_name_id;

impl Value {
    /// `true` iff this value is pure data (no stored expressions or captured
    /// evaluation environments, transitively). See module docs.
    #[must_use]
    pub fn is_concrete_data(&self) -> bool {
        match self {
            // --- Scalar / leaf data ---
            Value::Bool(_)
            | Value::SmallInt(_)
            | Value::Int(_)
            | Value::String(_)
            | Value::ModelValue(_) => true,
            // Interval bounds are plain integers.
            Value::Interval(_) => true,
            // Infinite-but-parameterless lazy sets: pure data by construction.
            Value::StringSet | Value::AnySet => true,

            // --- Concrete collections: recurse into nested values ---
            Value::Set(set) => set.iter().all(Value::is_concrete_data),
            Value::Func(func) => func
                .iter()
                .all(|(k, v)| k.is_concrete_data() && v.is_concrete_data()),
            Value::Bag(bag) => bag.elems().iter().all(Value::is_concrete_data),
            Value::IntFunc(f) => f.values().iter().all(Value::is_concrete_data),
            Value::Seq(seq) => seq.iter().all(Value::is_concrete_data),
            Value::Record(record) => record.values().all(Value::is_concrete_data),
            Value::Tuple(elems) => elems.iter().all(Value::is_concrete_data),

            // --- Lazy collections whose payloads are nested values only ---
            Value::Subset(subset) => subset.base.is_concrete_data(),
            Value::FuncSet(fs) => fs.domain.is_concrete_data() && fs.codomain.is_concrete_data(),
            Value::RecordSet(rs) => rs.fields.values().all(|v| v.is_concrete_data()),
            Value::TupleSet(ts) => ts.components.iter().all(|v| v.is_concrete_data()),
            Value::SetCup(v) => v.set1.is_concrete_data() && v.set2.is_concrete_data(),
            Value::SetCap(v) => v.set1.is_concrete_data() && v.set2.is_concrete_data(),
            Value::SetDiff(v) => v.set1.is_concrete_data() && v.set2.is_concrete_data(),
            Value::KSubset(ks) => ks.base.is_concrete_data(),
            Value::BigUnion(u) => u.set.is_concrete_data(),
            Value::SeqSet(ss) => ss.base.is_concrete_data(),

            // --- Expression-bearing values: NOT pure data ---
            Value::SetPred(_) | Value::LazyFunc(_) | Value::Closure(_) => false,
        }
    }

    /// Whether every first-seen token that TLC ordering can demand while
    /// consuming this concrete value has already been assigned.
    ///
    /// This is a read-only admission predicate: it never interns a string or
    /// registers a model value. Besides explicit string/model leaves, record
    /// field names matter because TLC's cross-kind function comparison turns a
    /// record domain into string values before sorting it. A caller that may
    /// reorder otherwise-pure evaluations can therefore use this together with
    /// [`Self::is_concrete_data`] to prove that consuming a cached/config value
    /// cannot mutate globally observable TLC ordering.
    #[must_use]
    pub fn has_preassigned_tlc_order_tokens(&self) -> bool {
        match self {
            Value::Bool(_)
            | Value::SmallInt(_)
            | Value::Int(_)
            | Value::Interval(_)
            | Value::StringSet
            | Value::AnySet => true,
            Value::String(value) => lookup_tlc_string_token(value).is_some(),
            Value::ModelValue(value) => {
                lookup_tlc_string_token(value).is_some()
                    && lookup_model_value_index(value).is_some()
            }

            Value::Set(set) => set.iter().all(Self::has_preassigned_tlc_order_tokens),
            Value::Func(func) => func.iter().all(|(key, value)| {
                key.has_preassigned_tlc_order_tokens() && value.has_preassigned_tlc_order_tokens()
            }),
            Value::Bag(bag) => bag
                .elems()
                .iter()
                .all(Self::has_preassigned_tlc_order_tokens),
            Value::IntFunc(func) => func
                .values()
                .iter()
                .all(Self::has_preassigned_tlc_order_tokens),
            Value::Seq(seq) => seq.iter().all(Self::has_preassigned_tlc_order_tokens),
            Value::Record(record) => record.iter().all(|(field, value)| {
                lookup_tlc_string_token(resolve_name_id(field).as_ref()).is_some()
                    && value.has_preassigned_tlc_order_tokens()
            }),
            Value::Tuple(values) => values.iter().all(Self::has_preassigned_tlc_order_tokens),

            Value::Subset(subset) => subset.base.has_preassigned_tlc_order_tokens(),
            Value::FuncSet(func_set) => {
                func_set.domain.has_preassigned_tlc_order_tokens()
                    && func_set.codomain.has_preassigned_tlc_order_tokens()
            }
            Value::RecordSet(record_set) => record_set.fields_iter().all(|(field, values)| {
                lookup_tlc_string_token(field).is_some()
                    && values.has_preassigned_tlc_order_tokens()
            }),
            Value::TupleSet(tuple_set) => tuple_set
                .components
                .iter()
                .all(|value| value.has_preassigned_tlc_order_tokens()),
            Value::SetCup(value) => {
                value.set1.has_preassigned_tlc_order_tokens()
                    && value.set2.has_preassigned_tlc_order_tokens()
            }
            Value::SetCap(value) => {
                value.set1.has_preassigned_tlc_order_tokens()
                    && value.set2.has_preassigned_tlc_order_tokens()
            }
            Value::SetDiff(value) => {
                value.set1.has_preassigned_tlc_order_tokens()
                    && value.set2.has_preassigned_tlc_order_tokens()
            }
            Value::KSubset(subsets) => subsets.base.has_preassigned_tlc_order_tokens(),
            Value::BigUnion(union) => union.set.has_preassigned_tlc_order_tokens(),
            Value::SeqSet(sequences) => sequences.base.has_preassigned_tlc_order_tokens(),

            // These are rejected independently by `is_concrete_data`; keep the
            // token predicate fail-closed when called on its own as well.
            Value::SetPred(_) | Value::LazyFunc(_) | Value::Closure(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rp::Rp;
    use std::sync::Arc;

    fn small_set(vals: &[i64]) -> Value {
        Value::Set(Rp::new(crate::value::SortedSet::from_iter(
            vals.iter().map(|&v| Value::SmallInt(v)),
        )))
    }

    #[test]
    fn scalars_are_concrete() {
        assert!(Value::Bool(true).is_concrete_data());
        assert!(Value::SmallInt(7).is_concrete_data());
        assert!(Value::String(Rp::from("x")).is_concrete_data());
        assert!(Value::ModelValue(Rp::from("m1")).is_concrete_data());
    }

    #[test]
    fn concrete_collections_are_concrete() {
        assert!(small_set(&[1, 2, 3]).is_concrete_data());
        let tuple: Value = Value::Tuple(Rp::from(vec![Value::SmallInt(1), Value::Bool(false)]));
        assert!(tuple.is_concrete_data());
    }

    #[test]
    fn lazy_data_sets_are_concrete() {
        let subset = Value::Subset(crate::value::SubsetValue::new(small_set(&[1, 2])));
        assert!(subset.is_concrete_data());
    }

    #[test]
    fn closure_is_not_concrete() {
        // A closure nested inside a tuple must poison the whole value.
        let closure = Value::Closure(Rp::new(crate::value::ClosureValue::new(
            vec!["x".to_string()],
            tla_core::Spanned::dummy(tla_core::ast::Expr::Ident(
                "x".to_string(),
                tla_core::NameId::INVALID,
            )),
            Arc::new(Default::default()),
            None,
        )));
        assert!(!closure.is_concrete_data());
        let tuple: Value = Value::Tuple(Rp::from(vec![Value::SmallInt(1), closure]));
        assert!(!tuple.is_concrete_data());
    }

    #[test]
    fn record_field_tokens_are_required_without_assigning_them() {
        let _lock = crate::value::lock_intern_state();
        crate::clear_string_intern_table();
        crate::clear_tlc_string_tokens();

        let field = "concrete_data_unseen_record_field";
        let record = Value::Record(crate::RecordValue::from_entries(vec![(
            tla_core::intern_name(field),
            Value::SmallInt(1),
        )]));
        assert!(record.is_concrete_data());
        assert!(!record.has_preassigned_tlc_order_tokens());
        assert_eq!(crate::lookup_tlc_string_token(field), None);

        let _field_value = Value::string(field);
        assert!(record.has_preassigned_tlc_order_tokens());

        crate::clear_string_intern_table();
        crate::clear_tlc_string_tokens();
    }
}
