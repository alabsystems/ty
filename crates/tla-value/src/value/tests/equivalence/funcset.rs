// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! FuncSet iterator semantics tests.

use super::super::super::*;
use crate::rp::Rp;
use std::cmp::Ordering;

fn reference_func(entries: impl IntoIterator<Item = (Value, Value)>) -> Value {
    let mut builder = FuncBuilder::new();
    for (key, value) in entries {
        builder.insert(key, value);
    }
    Value::Func(Rp::new(builder.build()))
}

fn assert_extensional_tlc_and_fingerprint_eq(actual: &Value, expected: &Value) {
    assert_eq!(actual, expected, "function mappings differ");
    assert_eq!(
        Value::tlc_cmp(actual, expected).expect("functions must be TLC-comparable"),
        Ordering::Equal,
        "TLC comparison differs"
    );
    assert_eq!(
        actual
            .fingerprint_extend(0x9e37_79b9_7f4a_7c15)
            .expect("actual function must fingerprint"),
        expected
            .fingerprint_extend(0x9e37_79b9_7f4a_7c15)
            .expect("reference function must fingerprint"),
        "TLC fingerprints differ"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_funcset_iterator_produces_seq_for_domain_1_n() {
    // FuncSetIterator should produce Seq values when domain is 1..n
    // (because in TLA+, functions with domain 1..n are semantically sequences)
    use crate::value::{FuncSetValue, IntervalValue, SortedSet};

    // Create [1..4 -> {"A", "B"}]
    let domain = Value::Interval(Rp::new(IntervalValue::new(
        BigInt::from(1),
        BigInt::from(4),
    )));
    let codomain = Value::Set(Rp::new(SortedSet::from_iter(vec![
        Value::String("A".into()),
        Value::String("B".into()),
    ])));

    let func_set = FuncSetValue::new(domain, codomain);
    let mut all_elems: Vec<Value> = func_set
        .iter()
        .expect("should be able to iterate")
        .collect();

    // [1..4 -> {"A", "B"}] should have 2^4 = 16 elements
    assert_eq!(
        all_elems.len(),
        16,
        "Expected 2^4 = 16 functions from 1..4 -> {{A, B}}"
    );

    all_elems.sort();
    all_elems.dedup();
    assert_eq!(all_elems.len(), 16, "All functions should be distinct");

    let symbols = [Value::String("A".into()), Value::String("B".into())];
    let mut expected = Vec::new();
    for e1 in &symbols {
        for e2 in &symbols {
            for e3 in &symbols {
                for e4 in &symbols {
                    expected.push(Value::seq([e1.clone(), e2.clone(), e3.clone(), e4.clone()]));
                }
            }
        }
    }
    expected.sort();
    expected.dedup();
    assert_eq!(
        expected.len(),
        16,
        "Expected set should contain 16 unique sequences"
    );

    // Verify exact set equality, not just count/type checks.
    assert_eq!(
        all_elems, expected,
        "FuncSet iterator did not enumerate 1..n sequences exactly"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_funcset_iterator_skips_shared_domain_for_specialized_or_empty_sets() {
    let int_domain = vec![Value::int(1), Value::int(2), Value::int(3)];
    let int_iter =
        FuncSetIterator::from_elems(int_domain, vec![Value::Bool(false), Value::Bool(true)]);
    assert!(!int_iter.has_shared_domain());

    let tuple_domain = vec![Value::tuple([Value::int(1), Value::int(1)])];
    let mut empty_iter = FuncSetIterator::from_elems(tuple_domain, Vec::new());
    assert!(!empty_iter.has_shared_domain());
    assert!(empty_iter.next().is_none());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_funcset_iterator_empty_domain_has_one_empty_function() {
    for codomain in [Vec::new(), vec![Value::Bool(false), Value::Bool(true)]] {
        let mut iterator = FuncSetIterator::from_elems(Vec::new(), codomain);
        assert!(!iterator.has_shared_domain());
        assert_eq!(
            iterator.next(),
            Some(Value::Func(Rp::new(FuncValue::from_sorted_entries(
                Vec::new()
            ))))
        );
        assert!(iterator.next().is_none());
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_funcset_iterator_produces_intfunc_for_non_one_start() {
    // FuncSetIterator should produce IntFunc when domain is NOT 1..n (e.g., 2..5)
    use crate::value::{FuncSetValue, IntervalValue, SortedSet};

    // Create [2..4 -> {"A", "B"}] - domain starts at 2, not 1
    let domain = Value::Interval(Rp::new(IntervalValue::new(
        BigInt::from(2),
        BigInt::from(4),
    )));
    let codomain = Value::Set(Rp::new(SortedSet::from_iter(vec![
        Value::String("A".into()),
        Value::String("B".into()),
    ])));

    let func_set = FuncSetValue::new(domain, codomain);
    let iter = func_set.iter().expect("should be able to iterate");

    // Check that all produced values are IntFunc (domain 2..n is NOT a sequence)
    let mut found_intfunc = false;
    for func in iter.take(5) {
        match func {
            Value::IntFunc(_) => found_intfunc = true,
            _ => panic!("FuncSetIterator produced unexpected type for domain 2..n: {func:?}"),
        }
    }

    assert!(
        found_intfunc,
        "FuncSetIterator should produce IntFunc for domain not starting at 1"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_funcset_iterator_shares_dense_function_domains() {
    let keys = vec![
        Value::tuple([Value::int(1), Value::int(1)]),
        Value::tuple([Value::int(1), Value::int(2)]),
        Value::tuple([Value::int(2), Value::int(1)]),
        Value::tuple([Value::int(2), Value::int(2)]),
    ];
    let iterator =
        FuncSetIterator::from_elems(keys.clone(), vec![Value::Bool(false), Value::Bool(true)]);
    assert!(iterator.has_shared_domain());
    let mut functions: Vec<_> = iterator.collect();

    assert_eq!(functions.len(), 16);
    let Value::Func(first) = &functions[0] else {
        panic!("non-integer function domain should produce Func values");
    };
    let shared_domain_ptr = first.domain_ptr();
    let shared_descriptor_ptr = first.domain_descriptor_ptr();
    for function in &functions {
        let Value::Func(function) = function else {
            panic!("non-integer function domain should produce Func values");
        };
        assert_eq!(function.domain_ptr(), shared_domain_ptr);
        assert_eq!(function.domain_descriptor_ptr(), shared_descriptor_ptr);
        assert!(function.dense_is_dim2());
        assert!(function.tlc_normalized_order().is_none());
    }
    for pair in functions.windows(2) {
        let (Value::Func(left), Value::Func(right)) = (&pair[0], &pair[1]) else {
            unreachable!("all iterator results were checked as Func values");
        };
        assert_ne!(left.values_ptr(), right.values_ptr());
    }

    // Tuple domains require TLC normalization. Computing it once must publish
    // the domain-only cache to every function produced by this iterator.
    functions[0]
        .fingerprint_extend(0)
        .expect("shared-domain function must fingerprint");
    for function in &functions {
        let Value::Func(function) = function else {
            unreachable!("all iterator results were checked as Func values");
        };
        assert!(function.tlc_normalized_order().is_some());
    }

    for (ordinal, actual) in functions.drain(..).enumerate() {
        let entries = keys.iter().enumerate().map(|(position, key)| {
            let shift = keys.len() - position - 1;
            (key.clone(), Value::Bool((ordinal & (1 << shift)) != 0))
        });
        let expected = reference_func(entries);
        assert_extensional_tlc_and_fingerprint_eq(&actual, &expected);

        let Value::Func(ref function) = actual else {
            unreachable!("all iterator results were checked as Func values");
        };
        for (position, key) in keys.iter().enumerate() {
            let shift = keys.len() - position - 1;
            assert_eq!(
                function.apply(key),
                Some(&Value::Bool((ordinal & (1 << shift)) != 0))
            );
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_funcset_iterator_preserves_tlc_odometer_order_when_storage_order_differs() {
    // TLC orders tuples length-first, while Value::cmp is lexicographic here.
    let short = Value::tuple([Value::int(2)]);
    let long = Value::tuple([Value::int(1), Value::int(2)]);
    assert!(long < short);
    assert_eq!(
        Value::tlc_cmp(&short, &long).expect("tuples must be TLC-comparable"),
        Ordering::Less
    );

    let function_set = Value::FuncSet(FuncSetValue::new(
        Value::set([short.clone(), long.clone()]),
        Value::set([Value::Bool(false), Value::Bool(true)]),
    ));
    let mut iter = function_set
        .iter_set_tlc_normalized()
        .expect("function set must support TLC-normalized iteration");
    let expected_values = [(false, false), (false, true), (true, false), (true, true)];
    let mut shared_domain_ptr = None;
    let mut shared_descriptor_ptr = None;

    for (short_value, long_value) in expected_values {
        let actual = iter.next().expect("expected all four functions");
        let expected = reference_func([
            (short.clone(), Value::Bool(short_value)),
            (long.clone(), Value::Bool(long_value)),
        ]);
        assert_extensional_tlc_and_fingerprint_eq(&actual, &expected);

        let Value::Func(ref function) = actual else {
            panic!("tuple domain should produce a Func value");
        };
        assert_eq!(function.apply(&short), Some(&Value::Bool(short_value)));
        assert_eq!(function.apply(&long), Some(&Value::Bool(long_value)));
        if let Some(pointer) = shared_domain_ptr {
            assert_eq!(function.domain_ptr(), pointer);
        } else {
            shared_domain_ptr = Some(function.domain_ptr());
        }
        if let Some(pointer) = shared_descriptor_ptr {
            assert_eq!(function.domain_descriptor_ptr(), pointer);
        } else {
            shared_descriptor_ptr = Some(function.domain_descriptor_ptr());
        }
    }
    assert!(iter.next().is_none());
}
