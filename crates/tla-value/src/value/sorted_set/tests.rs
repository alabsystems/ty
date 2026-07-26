// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Lazy normalization tests for `SortedSet`.
//!
//! Extracted from `sorted_set.rs` per #3326. Kept as a child module
//! of `sorted_set` to preserve access to private internals (`storage`,
//! `SetStorage::Unnormalized`, etc.) without widening visibility.

use crate::rp::Rp;

use super::algebra::SMALL_RAW_UNION_RHS_MAX;
use super::*;
use crate::value::clear_set_intern_table;
use crate::value::cmp_helpers::{cmp_tuple_elements_with_value, eq_tuple_elements_with_value};
use crate::value::{BagValue, FuncBuilder, IntIntervalFunc};
use crate::KSubsetValue;
use num_bigint::BigInt;
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hash_sorted_set(set: &SortedSet) -> u64 {
    let mut hasher = DefaultHasher::new();
    set.hash(&mut hasher);
    hasher.finish()
}

/// The pre-fast-path concrete union: concatenate raw storage and normalize only
/// when an ordered observer asks for it.
fn reference_raw_union(left: &SortedSet, right: &SortedSet) -> SortedSet {
    let mut combined = Vec::with_capacity(left.raw_slice().len() + right.raw_slice().len());
    combined.extend_from_slice(left.raw_slice());
    combined.extend_from_slice(right.raw_slice());
    SortedSet::from_unnormalized_vec(combined)
}

fn oversized_ksubset_value() -> Value {
    Value::KSubset(KSubsetValue::new(Value::StringSet, (i32::MAX as usize) + 1))
}

#[test]
fn from_iter_defers_normalization_until_ordered_observation() {
    let set = SortedSet::from_iter(vec![
        Value::int(3),
        Value::int(1),
        Value::int(3),
        Value::int(2),
    ]);

    match &set.storage {
        SetStorage::Unnormalized { normalized, .. } => {
            assert!(normalized.get().is_none(), "from_iter should stay lazy");
        }
        SetStorage::Normalized(_) => panic!("from_iter should not eagerly normalize"),
    }
    assert!(set.contains(&Value::int(2)));

    match &set.storage {
        SetStorage::Unnormalized { normalized, .. } => {
            assert!(
                normalized.get().is_none(),
                "membership-only checks should not populate normalized cache"
            );
        }
        SetStorage::Normalized(_) => panic!("from_iter should still be lazy after contains"),
    }

    // len() should NOT force normalization — it uses FxHashSet dedup counting.
    assert_eq!(set.len(), 3);

    match &set.storage {
        SetStorage::Unnormalized { normalized, .. } => {
            assert!(
                normalized.get().is_none(),
                "len() should not force normalization"
            );
        }
        SetStorage::Normalized(_) => panic!("from_iter should still be lazy after len()"),
    }

    let elements: Vec<_> = set.iter().cloned().collect();
    assert_eq!(elements, vec![Value::int(1), Value::int(2), Value::int(3)]);
    // After iter(), len() should still return the correct value.
    assert_eq!(set.len(), 3);

    match &set.storage {
        SetStorage::Unnormalized { normalized, .. } => {
            assert!(
                normalized.get().is_some(),
                "ordered observation should populate normalized cache"
            );
        }
        SetStorage::Normalized(_) => panic!("from_iter should keep raw storage variant"),
    }
}

fn equivalent_tuple2_representations() -> Vec<Value> {
    let values = vec![Value::int(10), Value::int(20)];
    let mut func_builder = FuncBuilder::new();
    func_builder.insert(Value::int(1), values[0].clone());
    func_builder.insert(Value::int(2), values[1].clone());

    let mut representations = vec![
        Value::Tuple(values.clone().into()),
        Value::seq(values.clone()),
        Value::Func(Rp::new(func_builder.build())),
        Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, values))),
    ];
    if let Ok(bag) = BagValue::try_from_entries(vec![
        (Value::int(1), Value::int(10)),
        (Value::int(2), Value::int(20)),
    ]) {
        representations.push(Value::Bag(Rp::new(bag)));
    }
    representations
}

#[test]
fn tuple_element_comparison_matches_materialized_tuple() {
    let elements = vec![Value::int(10), Value::int(20)];
    let materialized = Value::Tuple(elements.clone().into());
    let mut cases = equivalent_tuple2_representations();
    cases.extend([
        Value::Bool(false),
        Value::int(10),
        Value::string("tuple-order"),
        Value::record(Vec::<(&str, Value)>::new()),
        Value::record([("field", Value::int(1))]),
        Value::set([Value::int(1)]),
        Value::IntFunc(Rp::new(IntIntervalFunc::new(
            0,
            1,
            vec![Value::int(10), Value::int(20)],
        ))),
    ]);

    for candidate in cases {
        assert_eq!(
            cmp_tuple_elements_with_value(&elements, &candidate),
            materialized.cmp(&candidate),
            "allocation-free tuple ordering drifted for {candidate:?}"
        );
        assert_eq!(
            eq_tuple_elements_with_value(&elements, &candidate),
            materialized == candidate,
            "allocation-free tuple equality drifted for {candidate:?}"
        );
    }
}

#[test]
fn tuple_elements_membership_handles_all_equivalent_representations() {
    let elements = vec![Value::int(10), Value::int(20)];
    let miss = vec![Value::int(10), Value::int(21)];

    for representation in equivalent_tuple2_representations() {
        let mut normalized_elements = vec![
            Value::Bool(false),
            representation.clone(),
            Value::string("after-tuple"),
        ];
        normalized_elements.sort();
        let normalized = SortedSet::from_sorted_vec(normalized_elements);
        assert!(normalized.contains_tuple_elements(&elements));
        assert!(!normalized.contains_tuple_elements(&miss));

        let raw = SortedSet::from_iter(vec![
            Value::string("raw-first"),
            representation,
            Value::Bool(true),
        ]);
        assert!(raw.contains_tuple_elements(&elements));
        assert!(!raw.contains_tuple_elements(&miss));
        match &raw.storage {
            SetStorage::Unnormalized { normalized, .. } => assert!(
                normalized.get().is_none(),
                "tuple membership must not normalize raw set storage"
            ),
            SetStorage::Normalized(_) => panic!("from_iter should keep raw storage"),
        }
    }
}

#[test]
fn equality_and_hash_use_normalized_view() {
    let lazy = SortedSet::from_iter(vec![Value::int(2), Value::int(1), Value::int(2)]);
    let eager = SortedSet::from_sorted_vec(vec![Value::int(1), Value::int(2)]);

    assert_eq!(lazy, eager);
    assert_eq!(hash_sorted_set(&lazy), hash_sorted_set(&eager));
    assert_eq!(lazy.as_slice(), eager.as_slice());
}

#[test]
fn canonical_sorted_operations_reuse_small_set_interning() {
    // Clearing the process-global set intern table races with concurrently
    // running tests (e.g. the parallel-intern freeze tests) — serialize via
    // the shared intern-state test lock.
    let _lock = crate::value::lock_intern_state();
    clear_set_intern_table();

    let base = SortedSet::from_sorted_vec(vec![Value::int(1), Value::int(2)]);
    let inserted = base.insert(Value::int(3));
    let eager = SortedSet::from_sorted_vec(vec![Value::int(1), Value::int(2), Value::int(3)]);
    assert!(
        inserted.ptr_eq(&eager),
        "normalized insert results should reuse interned small-set storage"
    );

    let union = base.union(&SortedSet::from_sorted_vec(vec![
        Value::int(2),
        Value::int(3),
    ]));
    assert!(
        union.ptr_eq(&eager),
        "normalized union results should reuse interned small-set storage"
    );

    clear_set_intern_table();
}

#[test]
fn len_does_not_force_normalization_on_unnormalized_set() {
    // Construct an unnormalized set with duplicates.
    let set = SortedSet::from_iter(vec![
        Value::int(5),
        Value::int(3),
        Value::int(5),
        Value::int(1),
        Value::int(3),
    ]);

    // Verify storage is unnormalized.
    assert!(
        !matches!(set.storage, SetStorage::Normalized(_)),
        "from_iter should produce Unnormalized storage"
    );

    // len() should return the deduplicated count.
    assert_eq!(set.len(), 3, "len() should count unique elements");

    // Verify normalization was NOT triggered.
    match &set.storage {
        SetStorage::Unnormalized { normalized, .. } => {
            assert!(
                normalized.get().is_none(),
                "len() must not force normalization"
            );
        }
        SetStorage::Normalized(_) => panic!("len() should not have caused normalization"),
    }

    // Calling len() again should return the cached value.
    assert_eq!(set.len(), 3, "cached len should be stable");
}

#[test]
fn len_after_normalization_returns_correct_count() {
    let set = SortedSet::from_iter(vec![Value::int(2), Value::int(2), Value::int(1)]);

    // Force normalization by iterating.
    let _: Vec<_> = set.iter().cloned().collect();

    // len() should still return the correct deduplicated count.
    assert_eq!(set.len(), 2);
}

#[test]
fn equality_fast_exit_on_different_cardinality() {
    // Two sets with different cardinalities should fail equality quickly
    // via the len() check without needing full normalization of both.
    let small = SortedSet::from_iter(vec![Value::int(1), Value::int(2)]);
    let large = SortedSet::from_iter(vec![Value::int(1), Value::int(2), Value::int(3)]);

    assert_ne!(small, large);

    // At most one of them should have been normalized — the one whose len()
    // was computed first might trigger normalization on the other, but ideally
    // the cardinality mismatch is caught before any normalization.
    // We can at least verify the answer is correct.
}

#[test]
fn compute_dedup_len_small_set_quadratic_path() {
    // 2-8 elements use quadratic scan instead of HashSet.
    let elements = vec![Value::int(1), Value::int(2), Value::int(1), Value::int(3)];
    assert_eq!(SortedSet::compute_dedup_len(&elements), 3);
}

#[test]
fn compute_dedup_len_large_set_hashset_path() {
    // >8 elements use FxHashSet path.
    let elements: Vec<Value> = (0..20)
        .map(|i| Value::int(i % 7)) // 20 elements, 7 unique
        .collect();
    assert_eq!(SortedSet::compute_dedup_len(&elements), 7);
}

#[test]
fn normalized_set_has_dedup_len_set_eagerly() {
    let set = SortedSet::from_sorted_vec(vec![Value::int(1), Value::int(2), Value::int(3)]);

    // Normalized sets should have cached_dedup_len set eagerly.
    assert_eq!(
        set.cached_dedup_len.load(AtomicOrdering::Relaxed),
        3,
        "from_sorted_vec should eagerly set cached_dedup_len"
    );
    assert_eq!(set.len(), 3);
}

#[test]
fn normalized_lhs_small_raw_rhs_union_is_exact_and_normalized() {
    let left = SortedSet::from_sorted_vec(vec![Value::int(1), Value::int(3), Value::int(5)]);
    let right = SortedSet::from_iter(vec![
        Value::int(4),
        Value::int(3),
        Value::int(2),
        Value::int(2),
    ]);

    let result = left.union(&right);

    assert!(matches!(&result.storage, SetStorage::Normalized(_)));
    assert_eq!(
        result.as_slice(),
        &[
            Value::int(1),
            Value::int(2),
            Value::int(3),
            Value::int(4),
            Value::int(5),
        ]
    );
    match &right.storage {
        SetStorage::Unnormalized { normalized, .. } => assert!(
            normalized.get().is_none(),
            "the bounded raw RHS must be normalized only in local scratch storage"
        ),
        SetStorage::Normalized(_) => panic!("from_iter should produce raw RHS storage"),
    }
}

#[test]
fn small_raw_rhs_union_handles_empty_and_singleton_operands() {
    let left = SortedSet::from_sorted_vec(vec![Value::int(1), Value::int(3)]);
    let empty = SortedSet::new();
    assert_eq!(left.union(&empty), left);
    assert_eq!(empty.union(&left), left);

    let singleton = SortedSet::from_iter(vec![Value::int(2)]);
    let result = left.union(&singleton);
    assert!(matches!(&result.storage, SetStorage::Normalized(_)));
    assert_eq!(
        result.as_slice(),
        &[Value::int(1), Value::int(2), Value::int(3)]
    );
}

#[test]
fn small_raw_rhs_union_accepts_lhs_normalized_via_once_lock() {
    let left = SortedSet::from_iter(vec![
        Value::int(3),
        Value::int(1),
        Value::int(3),
        Value::int(2),
    ]);
    assert_eq!(
        left.as_slice(),
        &[Value::int(1), Value::int(2), Value::int(3)]
    );
    assert!(matches!(
        &left.storage,
        SetStorage::Unnormalized { normalized, .. } if normalized.get().is_some()
    ));

    let right = SortedSet::from_iter(vec![Value::int(4), Value::int(2)]);
    let result = left.union(&right);
    assert!(matches!(&result.storage, SetStorage::Normalized(_)));
    assert_eq!(
        result.as_slice(),
        &[Value::int(1), Value::int(2), Value::int(3), Value::int(4)]
    );
}

#[test]
fn small_raw_rhs_union_preserves_lhs_representation_for_equal_values() {
    let small_one = Value::SmallInt(1);
    let heap_one = Value::Int(Rp::new(BigInt::from(1)));
    assert_eq!(small_one, heap_one);
    assert_eq!(small_one.cmp(&heap_one), Ordering::Equal);

    let mut left_values = vec![
        small_one.clone(),
        Value::string("z"),
        Value::tuple([Value::int(1), Value::int(2)]),
    ];
    left_values.sort();
    left_values.dedup();
    let left = SortedSet::from_sorted_vec(left_values);
    let right = SortedSet::from_iter(vec![
        Value::string("a"),
        heap_one,
        Value::tuple([Value::int(1), Value::int(2)]),
        Value::string("a"),
    ]);

    let reference = reference_raw_union(&left, &right);
    let result = left.union(&right);
    assert_eq!(result, reference);
    assert_eq!(hash_sorted_set(&result), hash_sorted_set(&reference));
    assert_eq!(
        result.iter().filter(|value| value.is_int()).count(),
        1,
        "cross-representation equal integers must be deduplicated"
    );
    assert!(result
        .iter()
        .any(|value| matches!(value, Value::SmallInt(1))));
}

#[test]
fn small_raw_rhs_union_incremental_fingerprint_matches_full_recompute() {
    let left = SortedSet::from_sorted_vec(vec![Value::int(1), Value::int(3), Value::int(5)]);
    let left_fp = crate::dedup_fingerprint::compute_set_additive_fp(&left).unwrap();
    left.cache_additive_fp(left_fp);
    let right = SortedSet::from_iter(vec![
        Value::int(6),
        Value::int(3),
        Value::int(4),
        Value::int(4),
    ]);

    let reference = reference_raw_union(&left, &right);
    let result = left.union(&right);
    let recomputed = crate::dedup_fingerprint::compute_set_additive_fp(&result).unwrap();
    let reference_fp = crate::dedup_fingerprint::compute_set_additive_fp(&reference).unwrap();

    assert_eq!(result, reference);
    assert_eq!(recomputed, reference_fp);
    assert_eq!(result.get_additive_fp(), Some(recomputed));
    assert_eq!(
        crate::dedup_fingerprint::state_value_fingerprint(&Value::from_sorted_set(result.clone()))
            .unwrap(),
        crate::dedup_fingerprint::state_value_fingerprint(&Value::from_sorted_set(reference))
            .unwrap()
    );
}

#[test]
fn small_raw_rhs_union_fingerprints_actual_interned_representatives() {
    let _lock = crate::value::lock_intern_state();
    clear_set_intern_table();

    let explicit_singleton = Value::set([Value::int(1)]);
    let lazy_singleton = Value::SetCup(crate::value::SetCupValue::new(
        Value::set([Value::int(1)]),
        Value::empty_set(),
    ));
    assert_eq!(explicit_singleton, lazy_singleton);
    assert_ne!(
        crate::dedup_fingerprint::state_value_fingerprint(&explicit_singleton).unwrap(),
        crate::dedup_fingerprint::state_value_fingerprint(&lazy_singleton).unwrap(),
        "this regression requires equal values with representation-specific state fingerprints"
    );

    let preinterned = SortedSet::from_sorted_vec(vec![Value::int(0), explicit_singleton]);
    let left = SortedSet::from_sorted_vec(vec![Value::int(0)]);
    let left_fp = crate::dedup_fingerprint::compute_set_additive_fp(&left).unwrap();
    left.cache_additive_fp(left_fp);
    let right = SortedSet::from_iter(vec![lazy_singleton]);

    let result = left.union(&right);
    assert!(result.ptr_eq(&preinterned));
    let recomputed = crate::dedup_fingerprint::compute_set_additive_fp(&result).unwrap();
    assert_eq!(result.get_additive_fp(), Some(recomputed));

    clear_set_intern_table();
}

#[test]
fn interval_additive_fingerprint_matches_materialized_set_and_nested_recompute() {
    let interval = Value::Interval(Rp::new(crate::value::IntervalValue::new(
        BigInt::from(-1),
        BigInt::from(2),
    )));
    let materialized = Value::from_sorted_set(SortedSet::from_sorted_vec(vec![
        Value::int(-1),
        Value::int(0),
        Value::int(1),
        Value::int(2),
    ]));
    assert_eq!(interval, materialized);
    assert_eq!(
        crate::dedup_fingerprint::state_value_fingerprint(&interval).unwrap(),
        crate::dedup_fingerprint::state_value_fingerprint(&materialized).unwrap()
    );

    let nested_interval_set =
        SortedSet::from_normalized_arc_shared(Rp::<[Value]>::from(vec![interval]));
    let nested_materialized_set =
        SortedSet::from_normalized_arc_shared(Rp::<[Value]>::from(vec![materialized]));
    assert_eq!(nested_interval_set, nested_materialized_set);
    let nested_interval_fp =
        crate::dedup_fingerprint::compute_set_additive_fp(&nested_interval_set).unwrap();
    let nested_materialized_fp =
        crate::dedup_fingerprint::compute_set_additive_fp(&nested_materialized_set).unwrap();
    assert_eq!(nested_interval_fp, nested_materialized_fp);
}

#[test]
fn oversized_interval_additive_fingerprint_fails_before_iteration() {
    let interval = Value::Interval(Rp::new(crate::value::IntervalValue::new(
        BigInt::from(0),
        BigInt::from(i32::MAX),
    )));

    let error = crate::dedup_fingerprint::state_value_fingerprint(&interval)
        .expect_err("an interval longer than i32::MAX must fail immediately");
    assert!(matches!(
        error,
        crate::value::value_fingerprint::FingerprintError::I32Overflow {
            value,
            context: "interval length",
        } if value == "2147483648"
    ));
}

#[test]
fn insert_fingerprints_actual_interned_representatives() {
    let _lock = crate::value::lock_intern_state();
    clear_set_intern_table();

    let explicit_singleton = Value::set([Value::int(1)]);
    let lazy_singleton = Value::SetCup(crate::value::SetCupValue::new(
        Value::set([Value::int(1)]),
        Value::empty_set(),
    ));
    assert_eq!(explicit_singleton, lazy_singleton);
    assert_ne!(
        crate::dedup_fingerprint::state_value_fingerprint(&explicit_singleton).unwrap(),
        crate::dedup_fingerprint::state_value_fingerprint(&lazy_singleton).unwrap()
    );

    let preinterned = SortedSet::from_sorted_vec(vec![Value::int(0), explicit_singleton]);
    let left = SortedSet::from_sorted_vec(vec![Value::int(0)]);
    let left_fp = crate::dedup_fingerprint::compute_set_additive_fp(&left).unwrap();
    left.cache_additive_fp(left_fp);

    let result = left.insert(lazy_singleton);
    assert!(result.ptr_eq(&preinterned));
    let recomputed = crate::dedup_fingerprint::compute_set_additive_fp(&result).unwrap();
    assert_eq!(result.get_additive_fp(), Some(recomputed));

    clear_set_intern_table();
}

#[test]
fn raw_rhs_above_bound_retains_lazy_union_storage() {
    let left = SortedSet::from_sorted_vec(vec![Value::int(-1)]);
    let right =
        SortedSet::from_iter((0..=SMALL_RAW_UNION_RHS_MAX).map(|value| Value::int(value as i64)));

    let result = left.union(&right);
    assert!(matches!(
        &result.storage,
        SetStorage::Unnormalized { normalized, .. } if normalized.get().is_none()
    ));
    assert_eq!(result.len(), SMALL_RAW_UNION_RHS_MAX + 2);
}

#[test]
fn small_raw_rhs_union_leaves_fingerprint_unset_when_new_element_cannot_hash() {
    let left = SortedSet::from_sorted_vec((0..9).map(Value::int).collect());
    let left_fp = crate::dedup_fingerprint::compute_set_additive_fp(&left).unwrap();
    left.cache_additive_fp(left_fp);
    let oversized_ksubset = oversized_ksubset_value();
    assert!(crate::dedup_fingerprint::state_value_fingerprint(&oversized_ksubset).is_err());

    let result = left.union(&SortedSet::from_iter([oversized_ksubset]));

    assert_eq!(result.len(), 10);
    assert_eq!(result.get_additive_fp(), None);
    assert!(
        crate::dedup_fingerprint::state_value_fingerprint(&Value::from_sorted_set(result)).is_err()
    );
}

#[test]
fn insert_leaves_fingerprint_unset_when_new_element_cannot_hash() {
    let left = SortedSet::from_sorted_vec((0..9).map(Value::int).collect());
    let left_fp = crate::dedup_fingerprint::compute_set_additive_fp(&left).unwrap();
    left.cache_additive_fp(left_fp);

    let result = left.insert(oversized_ksubset_value());

    assert_eq!(result.len(), 10);
    assert_eq!(result.get_additive_fp(), None);
    assert!(
        crate::dedup_fingerprint::state_value_fingerprint(&Value::from_sorted_set(result)).is_err()
    );
}

proptest! {
    #[test]
    fn small_raw_rhs_union_matches_raw_concat_for_random_duplicates_and_order(
        left_raw in proptest::collection::vec(-32i16..=32, 1..32),
        right_raw in proptest::collection::vec(-32i16..=32, 0..=SMALL_RAW_UNION_RHS_MAX),
    ) {
        let mut left_values: Vec<_> = left_raw
            .into_iter()
            .map(|value| Value::int(i64::from(value)))
            .collect();
        left_values.sort();
        left_values.dedup();
        let left = SortedSet::from_sorted_vec(left_values);
        let left_fp = crate::dedup_fingerprint::compute_set_additive_fp(&left).unwrap();
        left.cache_additive_fp(left_fp);

        let right = SortedSet::from_iter(
            right_raw
                .into_iter()
                .map(|value| Value::int(i64::from(value))),
        );
        let reference = reference_raw_union(&left, &right);
        let result = left.union(&right);

        prop_assert_eq!(result.as_slice(), reference.as_slice());
        prop_assert_eq!(hash_sorted_set(&result), hash_sorted_set(&reference));
        let recomputed = crate::dedup_fingerprint::compute_set_additive_fp(&result).unwrap();
        let reference_fp = crate::dedup_fingerprint::compute_set_additive_fp(&reference).unwrap();
        prop_assert_eq!(recomputed, reference_fp);
        prop_assert_eq!(result.get_additive_fp(), Some(recomputed));
    }
}

// ---------------------------------------------------------------------------
// Differential coverage for the clone-free 2-tuple membership fast path
// (`contains_tuple2_refs` / `Value::try_set_contains_tuple2_refs`), which
// answers `<<a,b>> \in S` without materializing `[a.clone(), b.clone()]`.
// Every result must be byte-identical to the owned-tuple path it replaces.
// ---------------------------------------------------------------------------

#[cfg(test)]
fn pair(a: Value, b: Value) -> Value {
    Value::Tuple(vec![a, b].into())
}

#[test]
fn tuple2_refs_membership_matches_owned_path() {
    // Set of vertex-pairs `<<v, c>>` where v,c are themselves 2-tuples — the
    // exact `es` shape driving Sailfish's TypeOK `<<v,c>> \in es`.
    let members: Vec<Value> = vec![
        pair(
            pair(Value::int(1), Value::int(1)),
            pair(Value::int(0), Value::int(0)),
        ),
        pair(
            pair(Value::int(1), Value::int(2)),
            pair(Value::int(1), Value::int(1)),
        ),
        pair(
            pair(Value::int(2), Value::int(2)),
            pair(Value::int(1), Value::int(1)),
        ),
        pair(Value::int(3), Value::int(4)), // flat int-pair, different shape
    ];
    // Build both Normalized (from_sorted-ish via from_iter->normalize) and a
    // deliberately Unnormalized copy so both storage arms are exercised.
    let normalized = SortedSet::from_iter(members.clone());
    let _ = normalized.as_slice(); // force normalization
    let unnormalized = SortedSet::from_iter(members.clone());

    // Candidate (first, second) pairs: hits, near-misses, wrong types/arities.
    let candidates: Vec<(Value, Value)> = vec![
        // exact members
        (
            pair(Value::int(1), Value::int(1)),
            pair(Value::int(0), Value::int(0)),
        ),
        (
            pair(Value::int(1), Value::int(2)),
            pair(Value::int(1), Value::int(1)),
        ),
        (Value::int(3), Value::int(4)),
        // non-members
        (
            pair(Value::int(9), Value::int(9)),
            pair(Value::int(9), Value::int(9)),
        ),
        (
            pair(Value::int(1), Value::int(1)),
            pair(Value::int(0), Value::int(1)),
        ),
        (Value::int(3), Value::int(5)),
        (Value::int(1), pair(Value::int(1), Value::int(1))),
        (Value::string("x"), Value::int(1)),
        (Value::bool(true), Value::bool(false)),
    ];

    for set in [&normalized, &unnormalized] {
        for (first, second) in &candidates {
            let refs = set.contains_tuple2_refs(first, second);
            let owned = set.contains_tuple_elements(&[first.clone(), second.clone()]);
            assert_eq!(
                refs, owned,
                "contains_tuple2_refs disagreed with owned path for <<{first:?},{second:?}>>"
            );
            // And against the fully materialized candidate + generic membership.
            let materialized = Value::Tuple(vec![first.clone(), second.clone()].into());
            assert_eq!(refs, set.contains(&materialized));
        }
    }
}

#[test]
fn value_try_set_contains_tuple2_refs_matches_owned() {
    let members = vec![
        pair(Value::int(1), Value::int(2)),
        pair(Value::int(3), Value::int(4)),
    ];
    let set = Value::Set(Rp::new(SortedSet::from_iter(members)));
    let cases: Vec<(Value, Value)> = vec![
        (Value::int(1), Value::int(2)), // member
        (Value::int(3), Value::int(4)), // member
        (Value::int(1), Value::int(3)), // non-member
        (Value::int(5), Value::int(6)), // non-member
    ];
    for (a, b) in &cases {
        let owned = set.try_set_contains_tuple_elements(&[a.clone(), b.clone()]);
        let refs = set.try_set_contains_tuple2_refs(a, b);
        assert_eq!(refs, owned);
    }
    // Trivial-arm parity: Nat/Int/StringSet/AnySet and a non-set (None).
    for special in [
        Value::try_model_value("Nat").unwrap(),
        Value::try_model_value("Int").unwrap(),
        Value::StringSet,
        Value::AnySet,
        Value::int(7),
    ] {
        let a = Value::int(1);
        let b = Value::int(2);
        assert_eq!(
            special.try_set_contains_tuple2_refs(&a, &b),
            special.try_set_contains_tuple_elements(&[a.clone(), b.clone()]),
            "trivial-arm disagreement for {special:?}"
        );
    }
}

// ===================================================================
// EdgeFilter fast path: `{c \in domain : <<v, c>> \in edges}`
// ===================================================================

/// Vertex `<<node, round>>` — the element type of the real dag-consensus
/// edge set, where edges are 2-tuples of vertices.
fn vertex(node: &str, round: i64) -> Value {
    Value::Tuple(vec![Value::string(node), Value::int(round)].into())
}

fn edge(from: Value, to: Value) -> Value {
    Value::Tuple(vec![from, to].into())
}

/// Naive reference identical to the VM's fallback: iterate the domain in
/// sorted order, keep `c` where `<<v, c>>` is a member of `edges`.
fn naive_edge_children(v: &Value, edges: &SortedSet, domain: &SortedSet) -> Vec<Value> {
    domain
        .iter()
        .filter(|c| edges.contains_tuple2_refs(v, c))
        .cloned()
        .collect()
}

/// `edge_children` composed with its documented abort → naive fallback,
/// exactly as the VM opcode uses it.
fn fused_edge_children(v: &Value, edges: &SortedSet, domain: &SortedSet) -> Vec<Value> {
    match edges.edge_children(v, domain) {
        Some(children) => children,
        None => naive_edge_children(v, edges, domain),
    }
}

fn assert_fused_matches_naive(v: &Value, edges: &SortedSet, domain: &SortedSet) {
    let naive = naive_edge_children(v, edges, domain);
    let fused = fused_edge_children(v, edges, domain);
    // Both are strictly-ascending subsets of the domain; compare directly.
    assert_eq!(
        fused, naive,
        "edge_children (with fallback) diverged from naive for v={v:?}"
    );
}

#[test]
fn edge_children_pure_tuple_edges_take_fast_path_and_match_naive() {
    let v1 = vertex("n1", 1);
    let v2 = vertex("n2", 1);
    let v3 = vertex("n3", 1);
    let g = vertex("", 0);
    let domain = SortedSet::from_iter(vec![
        v1.clone(),
        v2.clone(),
        v3.clone(),
        g.clone(),
        vertex("n2", 2),
    ]);
    // v2's out-edges: v2 -> v1, v2 -> g ; plus unrelated edges.
    let edges = SortedSet::from_iter(vec![
        edge(v2.clone(), v1.clone()),
        edge(v2.clone(), g.clone()),
        edge(v1.clone(), g.clone()),
        edge(v3.clone(), v1.clone()),
        edge(vertex("n2", 2), v2.clone()),
    ]);
    // Fast path must apply (pure tuples).
    assert!(edges.edge_children(&v2, &domain).is_some());
    let children = edges.edge_children(&v2, &domain).unwrap();
    let mut expected = vec![v1.clone(), g.clone()];
    expected.sort();
    let mut got = children.clone();
    got.sort();
    assert_eq!(got, expected);
    for v in [&v1, &v2, &v3, &g, &vertex("n2", 2), &vertex("nX", 9)] {
        assert_fused_matches_naive(v, &edges, &domain);
    }
}

#[test]
fn edge_children_excludes_second_component_outside_domain() {
    // An edge `<<v, c>>` whose `c` is NOT in the domain must be excluded,
    // matching `Children == {c \in vs : <<v,c>> \in es}`.
    let v = vertex("n1", 2);
    let in_dom = vertex("n2", 1);
    let out_dom = vertex("n9", 1); // present in edges, absent from domain
    let domain = SortedSet::from_iter(vec![v.clone(), in_dom.clone()]);
    let edges = SortedSet::from_iter(vec![
        edge(v.clone(), in_dom.clone()),
        edge(v.clone(), out_dom.clone()),
    ]);
    let children = edges.edge_children(&v, &domain).expect("fast path");
    assert_eq!(children, vec![in_dom.clone()], "c \\notin domain must drop");
    assert_fused_matches_naive(&v, &edges, &domain);
}

#[test]
fn edge_children_skips_wrong_arity_and_scalars_in_edge_set() {
    let v = Value::int(5);
    let domain = SortedSet::from_iter(vec![Value::int(1), Value::int(2), Value::int(3)]);
    let edges = SortedSet::from_iter(vec![
        edge(Value::int(5), Value::int(1)),       // match -> c=1
        edge(Value::int(5), Value::int(2)),       // match -> c=2
        Value::Tuple(vec![Value::int(5)].into()), // 1-tuple <<5>> : skip
        Value::Tuple(vec![Value::int(5), Value::int(3), Value::int(9)].into()), // 3-tuple: skip
        Value::int(5),                            // bare scalar: skip
        Value::string("zzz"),                     // scalar: skip
        edge(Value::int(4), Value::int(1)),       // first != v: skip
        edge(Value::int(6), Value::int(1)),       // first != v: skip
    ]);
    let children = edges.edge_children(&v, &domain).expect("fast path");
    assert_eq!(children, vec![Value::int(1), Value::int(2)]);
    assert_fused_matches_naive(&v, &edges, &domain);
}

#[test]
fn edge_children_handles_seq_edges_on_the_fast_path() {
    // A Seq is a physical sequence rep the fast path extracts directly.
    let v = Value::int(2);
    let domain = SortedSet::from_iter(vec![Value::int(1), Value::int(7)]);
    let edges = SortedSet::from_iter(vec![
        Value::seq(vec![Value::int(2), Value::int(1)]), // <<2,1>> as Seq
        Value::seq(vec![Value::int(2), Value::int(7)]), // <<2,7>> as Seq
        Value::seq(vec![Value::int(3), Value::int(1)]), // first != v
    ]);
    let children = fused_edge_children(&v, &edges, &domain);
    let mut got = children;
    got.sort();
    assert_eq!(got, vec![Value::int(1), Value::int(7)]);
    assert_fused_matches_naive(&v, &edges, &domain);
}

#[test]
fn edge_children_aborts_on_funclike_edges_but_fallback_matches_naive() {
    // A Func representing `<<v, c>>` is a tuple-equivalent the fast path
    // refuses (abort). The naive fallback must still find the edge.
    let v = Value::int(1);
    let c = Value::int(20);
    let domain = SortedSet::from_iter(vec![Value::int(10), c.clone()]);
    let mut fb = FuncBuilder::new();
    fb.insert(Value::int(1), v.clone()); // <<v, c>> as a function 1->v, 2->c
    fb.insert(Value::int(2), c.clone());
    let func_edge = Value::Func(Rp::new(fb.build()));
    let edges = SortedSet::from_iter(vec![
        edge(Value::int(1), Value::int(10)), // ordinary tuple edge
        func_edge,
    ]);
    // The Func edge is <<1,20>>, a real member -> fast path must abort.
    assert!(
        edges.edge_children(&v, &domain).is_none(),
        "funclike edge in the v-block must abort to the naive fallback"
    );
    // Fallback recovers both children (10 via tuple, 20 via func).
    let mut got = fused_edge_children(&v, &edges, &domain);
    got.sort();
    assert_eq!(got, vec![Value::int(10), Value::int(20)]);
    assert_fused_matches_naive(&v, &edges, &domain);
}

#[test]
fn edge_children_empty_edges_and_empty_domain() {
    let v = Value::int(1);
    let empty = SortedSet::new();
    let domain = SortedSet::from_iter(vec![Value::int(1), Value::int(2)]);
    let edges = SortedSet::from_iter(vec![edge(Value::int(1), Value::int(2))]);
    assert_eq!(edges.edge_children(&v, &empty), Some(vec![]));
    assert_eq!(empty.edge_children(&v, &domain), Some(vec![]));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// Random pure-tuple edge sets and domains over a small vertex universe:
    /// the fast path (with fallback) must match the naive comprehension for
    /// every candidate first component.
    #[test]
    fn edge_children_matches_naive_over_random_tuple_edges(
        edge_pairs in prop::collection::vec((0i64..6, 0i64..6), 0..24),
        domain_ids in prop::collection::vec(0i64..6, 0..7),
    ) {
        let node = |i: i64| vertex(if i % 2 == 0 { "a" } else { "b" }, i / 2);
        let edges = SortedSet::from_iter(
            edge_pairs.iter().map(|(a, b)| edge(node(*a), node(*b))),
        );
        let domain = SortedSet::from_iter(domain_ids.iter().map(|i| node(*i)));
        for i in 0..6 {
            let v = node(i);
            // Pure tuples -> fast path must apply (never abort).
            prop_assert!(edges.edge_children(&v, &domain).is_some());
            let fused = fused_edge_children(&v, &edges, &domain);
            let naive = naive_edge_children(&v, &edges, &domain);
            prop_assert_eq!(fused, naive);
        }
    }
}
