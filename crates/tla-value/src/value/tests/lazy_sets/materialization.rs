// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compound lazy set materialization tests (RecordSet, TupleSet, KSubset).

use super::super::super::*;
use crate::rp::Rp;
use std::sync::Arc;
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_additional_lazy_sets_to_sorted_set_match_expected_contents() {
    let record_set = RecordSetValue::new([
        (Arc::from("a"), Value::set([Value::int(1), Value::int(2)])),
        (Arc::from("b"), Value::set([Value::string("x")])),
    ]);
    let record_expected = Value::set([
        Value::Record(RecordValue::from_sorted_str_entries(vec![
            (Arc::from("a"), Value::int(1)),
            (Arc::from("b"), Value::string("x")),
        ])),
        Value::Record(RecordValue::from_sorted_str_entries(vec![
            (Arc::from("a"), Value::int(2)),
            (Arc::from("b"), Value::string("x")),
        ])),
    ])
    .to_sorted_set()
    .expect("expected record set should materialize");
    assert_eq!(
        record_set
            .to_sorted_set()
            .expect("record set should materialize"),
        record_expected
    );

    let tuple_set = TupleSetValue::new([
        Value::set([Value::int(1), Value::int(2)]),
        Value::set([Value::string("x")]),
    ]);
    let tuple_expected = Value::set([
        Value::Tuple(vec![Value::int(1), Value::string("x")].into()),
        Value::Tuple(vec![Value::int(2), Value::string("x")].into()),
    ])
    .to_sorted_set()
    .expect("expected tuple set should materialize");
    assert_eq!(
        tuple_set
            .to_sorted_set()
            .expect("tuple set should materialize"),
        tuple_expected
    );

    let ksubset = KSubsetValue::new(Value::set([Value::int(1), Value::int(2), Value::int(3)]), 2);
    let ksubset_expected = Value::set([
        Value::set([Value::int(1), Value::int(2)]),
        Value::set([Value::int(1), Value::int(3)]),
        Value::set([Value::int(2), Value::int(3)]),
    ])
    .to_sorted_set()
    .expect("expected k-subset should materialize");
    assert_eq!(
        ksubset
            .to_sorted_set()
            .expect("k-subset should materialize"),
        ksubset_expected
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_small_finite_lazy_set_respects_exact_materialization_cap() {
    let interval = |high: i64| {
        Value::Interval(Rp::new(IntervalValue::new(
            BigInt::from(1),
            BigInt::from(high),
        )))
    };

    let exactly_at_cap = Value::tuple_set([interval(256), Value::set([Value::int(0)])]);
    let one_above_cap = Value::tuple_set([interval(257), Value::set([Value::int(0)])]);

    assert!(
        exactly_at_cap.is_small_finite_lazy_set(256),
        "an enumerable lazy set with exactly 256 elements should qualify at cap 256"
    );
    assert!(
        !exactly_at_cap.is_small_finite_lazy_set(255),
        "the cardinality comparison must be inclusive only at the configured cap"
    );
    assert!(
        !one_above_cap.is_small_finite_lazy_set(256),
        "an enumerable lazy set with 257 elements must remain lazy at cap 256"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_small_finite_lazy_set_excludes_eager_and_powerset_representations() {
    let eager_set = Value::set([Value::int(1), Value::int(2)]);
    let eager_interval = Value::Interval(Rp::new(IntervalValue::new(
        BigInt::from(1),
        BigInt::from(2),
    )));
    let small_subset = Value::Subset(SubsetValue::new(eager_set.clone()));

    assert!(!eager_set.is_small_finite_lazy_set(256));
    assert!(!eager_interval.is_small_finite_lazy_set(256));
    assert!(
        !small_subset.is_small_finite_lazy_set(u64::MAX),
        "SUBSET must stay lazy regardless of its exact finite cardinality or cap"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_small_finite_lazy_set_accepts_representative_enumerable_variants() {
    let func_set = Value::FuncSet(FuncSetValue::new(
        Value::set([Value::int(1), Value::int(2)]),
        Value::set([Value::Bool(false), Value::Bool(true)]),
    ));
    let record_set = Value::RecordSet(Rp::new(RecordSetValue::new([
        (
            Arc::from("flag"),
            Value::set([Value::Bool(false), Value::Bool(true)]),
        ),
        (Arc::from("tag"), Value::set([Value::string("only")])),
    ])));
    let tuple_set = Value::tuple_set([
        Value::set([Value::int(1), Value::int(2)]),
        Value::set([Value::string("x"), Value::string("y")]),
    ]);
    let ksubset = Value::KSubset(KSubsetValue::new(
        Value::set([Value::int(1), Value::int(2), Value::int(3), Value::int(4)]),
        2,
    ));

    for (name, value, exact_len) in [
        ("FuncSet", func_set, 4),
        ("RecordSet", record_set, 2),
        ("TupleSet", tuple_set, 4),
        ("KSubset", ksubset, 6),
    ] {
        assert!(
            value.is_small_finite_lazy_set(exact_len),
            "{name} should qualify at its exact finite cardinality"
        );
        assert!(
            !value.is_small_finite_lazy_set(exact_len - 1),
            "{name} should not qualify below its exact finite cardinality"
        );
    }
}
