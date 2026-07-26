// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Trait implementations for Value ordering and equality.
//!
//! Implements `Ord`, `PartialOrd`, `PartialEq`, and `Eq` for `Value`.
//! Comparison helper functions live in `cmp_helpers/`.

use super::cmp_helpers::{cmp_cross_type, cmp_same_type, eq_cross_type, eq_same_type, type_order};
use super::Value;
use std::cmp::Ordering;

/// Returns true if this Value variant can participate in cross-type equality.
/// Cross-type groups: {Tuple, Seq, Func, IntFunc, Record, Bag}.
/// These are the only variants where `eq_cross_type` returns `Some`.
#[inline(always)]
fn is_cross_type_eligible(v: &Value) -> bool {
    matches!(
        v,
        Value::Tuple(_)
            | Value::Seq(_)
            | Value::Func(_)
            | Value::IntFunc(_)
            | Value::Record(_)
            | Value::Bag(_)
    )
}

/// Count a compound deep eq/cmp (diagnostic; no-op unless TY_CHURN_STATS).
/// "Compound" = heap-backed collection/structure kinds where a deep walk is
/// the expensive path a canonical-identity fast path would have avoided.
#[inline(always)]
fn count_compound_deep(site: crate::churn_stats::ChurnSite, v: &Value) {
    if crate::churn_stats::churn_stats_enabled()
        && matches!(
            v,
            Value::Set(_)
                | Value::Func(_)
                | Value::Record(_)
                | Value::Seq(_)
                | Value::Tuple(_)
                | Value::IntFunc(_)
                | Value::Bag(_)
        )
    {
        crate::churn_stats::churn_count(site);
    }
}

impl Ord for Value {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Fast path for scalar types — avoids the full cross-type match
        // and type_order calls for the most common comparison patterns.
        // Set sorting, fingerprint computation, and quantifier iteration
        // all hit Value::cmp millions of times, predominantly on SmallInts.
        match (self, other) {
            (Value::SmallInt(a), Value::SmallInt(b)) => return a.cmp(b),
            (Value::Bool(a), Value::Bool(b)) => return a.cmp(b),
            _ => {}
        }
        if let Some(ord) = cmp_cross_type(self, other) {
            return ord;
        }

        let ord = type_order(self).cmp(&type_order(other));
        if ord != Ordering::Equal {
            return ord;
        }

        if crate::churn_stats::churn_stats_enabled() && !self.ptr_eq(other) {
            count_compound_deep(crate::churn_stats::ChurnSite::CmpCompoundDeep, self);
        }
        cmp_same_type(self, other)
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Value {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Fast path for scalar types — the overwhelming majority of equality
        // checks during BFS are SmallInt==SmallInt or Bool==Bool (quantifier
        // guards, arithmetic comparisons, state variable checks). Checking
        // these first avoids the full eq_cross_type match (which handles
        // Tuple/Seq/Func/IntFunc/Record coercions) and the type_order calls.
        // This eliminates ~3 function calls and ~20 match arms for the
        // common case.
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => return *a == *b,
            (Value::SmallInt(a), Value::SmallInt(b)) => return *a == *b,
            _ => {}
        }
        // Arc pointer equality: for UNCHANGED evaluation in BFS, when a
        // variable was not modified by an action, the primed and unprimed
        // values share the same Arc allocation. This O(1) check avoids
        // deep comparison of arbitrarily large functions, records, sequences,
        // and sets. Part of #3805.
        if self.ptr_eq(other) {
            return true;
        }
        // Discriminant-based dispatch: avoids calling eq_cross_type (14-arm
        // match returning Option) and type_order (26-arm match) on every call.
        // Same discriminant → skip cross-type entirely, go to eq_same_type.
        // Different discriminant → only call eq_cross_type for the 5 eligible
        // variants, otherwise fast-reject via type_order.
        if std::mem::discriminant(self) == std::mem::discriminant(other) {
            // Same variant type: no cross-type coercion needed.
            // eq_same_type handles the remaining structural comparison.
            // (ptr_eq already missed above — this is a genuine deep compare.)
            count_compound_deep(crate::churn_stats::ChurnSite::EqCompoundDeep, self);
            return eq_same_type(self, other);
        }
        // Different discriminants. Most pairs are incompatible (e.g., Set vs
        // String). Only a few variant pairs support cross-type equality:
        // {Tuple, Seq, Func, IntFunc, Record} and {SmallInt, Int}.
        // Check the cheap eligibility predicate before the expensive match.
        if is_cross_type_eligible(self) && is_cross_type_eligible(other) {
            if let Some(result) = eq_cross_type(self, other) {
                return result;
            }
        }
        // SmallInt/Int cross-type: different discriminants but same type_order
        // group (both map to 1). eq_same_type handles SmallInt vs Int.
        // This is cheaper than calling type_order on both sides for the
        // general case, since we only reach here for non-scalar, non-cross-type
        // pairs.
        if type_order(self) != type_order(other) {
            return false;
        }
        eq_same_type(self, other)
    }
}

impl Eq for Value {}
