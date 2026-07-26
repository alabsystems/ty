// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Proof coverage for the dense-domain O(1) `FuncValue::apply` fast path.
//!
//! Every test compares the fast-path result against an INDEPENDENT linear-scan
//! reference (the semantic ground truth the binary search implements). Because
//! both the dense index and the binary search resolve "the value at the domain
//! key equal to `arg`", agreement with the linear reference across all in- and
//! out-of-domain arguments proves the dense path is byte-identical to the
//! binary-search path.

use super::super::*;
use crate::rp::Rp as Arc;
use num_bigint::BigInt;

/// Independent reference: linear scan over the original (key, value) entries,
/// using `Value`'s own equality. This does NOT go through `FuncValue::apply`.
fn reference_apply<'a>(entries: &'a [(Value, Value)], arg: &Value) -> Option<&'a Value> {
    entries.iter().find(|(k, _)| k == arg).map(|(_, v)| v)
}

/// Assert `fv.apply(arg)` equals the linear-scan reference for `arg`.
fn assert_apply_matches(fv: &FuncValue, entries: &[(Value, Value)], arg: &Value) {
    assert_eq!(
        fv.apply(arg),
        reference_apply(entries, arg),
        "apply({arg}) disagreed with linear reference"
    );
}

#[test]
fn dense_dim1_matches_reference_in_and_out_of_domain() {
    // Domain {1..8} -> distinct string values, exactly btree's `keysOf`/`isLeaf`
    // domain shape.
    let entries: Vec<(Value, Value)> = (1..=8)
        .map(|n| (Value::int(n), Value::string(format!("v{n}"))))
        .collect();
    let fv = FuncValue::from_sorted_entries(entries.clone());

    // Every in-domain key resolves to its value.
    for n in 1..=8 {
        assert_apply_matches(&fv, &entries, &Value::int(n));
    }
    // Out-of-domain integers on both sides and inside i64 range.
    for n in [-100, -1, 0, 9, 10, 1000] {
        assert_apply_matches(&fv, &entries, &Value::int(n));
    }
    // A BigInt-encoded in-domain key must resolve identically (SmallInt/Int
    // compare numerically), and a BigInt too large for i64 must miss.
    assert_apply_matches(&fv, &entries, &Value::big_int(BigInt::from(5)));
    let huge = Value::Int(Arc::new(BigInt::from(i64::MAX) + 1));
    assert_apply_matches(&fv, &entries, &huge);
    // Non-integer arguments are out of an integer domain.
    assert_apply_matches(&fv, &entries, &Value::string("nope"));
    assert_apply_matches(&fv, &entries, &Value::Bool(true));
    assert_apply_matches(&fv, &entries, &Value::tuple([Value::int(1), Value::int(2)]));
}

#[test]
fn dense_dim1_negative_and_singleton_domains() {
    // A domain that starts below zero and a single-element domain both classify
    // as dense Dim1 and must stay correct.
    let entries: Vec<(Value, Value)> = (-3..=2)
        .map(|n| (Value::int(n), Value::int(n * 10)))
        .collect();
    let fv = FuncValue::from_sorted_entries(entries.clone());
    for n in -6..=6 {
        assert_apply_matches(&fv, &entries, &Value::int(n));
    }

    let single = vec![(Value::int(42), Value::string("only"))];
    let fv1 = FuncValue::from_sorted_entries(single.clone());
    for n in [41, 42, 43] {
        assert_apply_matches(&fv1, &single, &Value::int(n));
    }
}

#[test]
fn sparse_int_domain_stays_correct() {
    // Gapped integer domain must NOT be treated as dense; binary search still
    // returns the right answers.
    let entries = vec![
        (Value::int(1), Value::string("a")),
        (Value::int(3), Value::string("b")),
        (Value::int(7), Value::string("c")),
    ];
    let fv = FuncValue::from_sorted_entries(entries.clone());
    assert!(!fv.dense_is_dim2());
    for n in -1..=9 {
        assert_apply_matches(&fv, &entries, &Value::int(n));
    }
}

/// Build the canonical (row-major, `Value::cmp`-sorted) cross-product
/// `[lo1..=hi1] x [lo2..=hi2]` mapped to distinct integer values.
fn cross_entries(lo1: i64, hi1: i64, lo2: i64, hi2: i64) -> Vec<(Value, Value)> {
    let mut out = Vec::new();
    let mut tag = 0i64;
    for n in lo1..=hi1 {
        for k in lo2..=hi2 {
            out.push((
                Value::tuple([Value::int(n), Value::int(k)]),
                Value::int(1000 + tag),
            ));
            tag += 1;
        }
    }
    out
}

#[test]
fn dense_dim2_matches_reference_over_full_grid() {
    // btree's childOf/valOf shape: Nodes(1..8) x Keys(1..4).
    let entries = cross_entries(1, 8, 1, 4);
    let fv = FuncValue::from_sorted_entries(entries.clone());
    assert!(fv.dense_is_dim2());

    // Every in-domain pair, plus out-of-range and wrong-shape arguments.
    for n in 0..=9 {
        for k in 0..=5 {
            let arg = Value::tuple([Value::int(n), Value::int(k)]);
            assert_apply_matches(&fv, &entries, &arg);
            // apply2_dense must agree with apply for integer components.
            assert_eq!(
                fv.apply2_dense(n, k),
                fv.apply(&arg),
                "apply2_dense({n},{k}) disagreed with apply"
            );
        }
    }

    // Wrong arity / non-integer element / non-tuple: all fall back and miss.
    assert_apply_matches(&fv, &entries, &Value::tuple([Value::int(1)]));
    assert_apply_matches(
        &fv,
        &entries,
        &Value::tuple([Value::int(1), Value::int(1), Value::int(1)]),
    );
    assert_apply_matches(
        &fv,
        &entries,
        &Value::tuple([Value::int(1), Value::string("x")]),
    );
    assert_apply_matches(&fv, &entries, &Value::int(1));
    assert_apply_matches(&fv, &entries, &Value::string("x"));
    // A BigInt-encoded component still resolves through the fast path.
    assert_apply_matches(
        &fv,
        &entries,
        &Value::tuple([Value::big_int(BigInt::from(3)), Value::int(2)]),
    );
}

#[test]
fn dense_dim2_asymmetric_and_offset_grid() {
    // Non-square, non-1-based grid stresses the stride/offset arithmetic.
    let entries = cross_entries(2, 5, 10, 12);
    let fv = FuncValue::from_sorted_entries(entries.clone());
    assert!(fv.dense_is_dim2());
    for n in 0..=7 {
        for k in 8..=14 {
            let arg = Value::tuple([Value::int(n), Value::int(k)]);
            assert_apply_matches(&fv, &entries, &arg);
            assert_eq!(fv.apply2_dense(n, k), fv.apply(&arg));
        }
    }
}

#[test]
fn incomplete_cross_product_is_not_dense_dim2() {
    // Drop one tuple from a full grid: it is no longer a dense cross-product, so
    // detection must fail closed and apply must still be exact via binary search.
    let mut entries = cross_entries(1, 3, 1, 3);
    // Remove <<2,2>> (a middle entry) and keep the rest sorted.
    entries.retain(|(k, _)| *k != Value::tuple([Value::int(2), Value::int(2)]));
    let fv = FuncValue::from_sorted_entries(entries.clone());
    assert!(!fv.dense_is_dim2());
    // apply2_dense must return None for a non-Dim2 function even at a real key.
    assert_eq!(fv.apply2_dense(1, 1), None);
    for n in 0..=4 {
        for k in 0..=4 {
            assert_apply_matches(&fv, &entries, &Value::tuple([Value::int(n), Value::int(k)]));
        }
    }
}

#[test]
fn apply2_dense_returns_none_for_dim1() {
    let entries: Vec<(Value, Value)> = (1..=5).map(|n| (Value::int(n), Value::int(n))).collect();
    let fv = FuncValue::from_sorted_entries(entries);
    assert!(!fv.dense_is_dim2());
    assert_eq!(fv.apply2_dense(1, 1), None);
    assert_eq!(fv.apply2_dense(3, 0), None);
}

#[test]
fn dense_classification_survives_except_overlay() {
    // EXCEPT keeps the domain, so the dense tag stays valid and reads through
    // the overlay must return the updated value.
    let entries = cross_entries(1, 4, 1, 4);
    let fv = FuncValue::from_sorted_entries(entries.clone());
    assert!(fv.dense_is_dim2());

    let key = Value::tuple([Value::int(3), Value::int(2)]);
    let new_val = Value::string("CHANGED");
    let updated = fv.clone().except(key.clone(), new_val.clone());
    assert!(updated.dense_is_dim2(), "EXCEPT must preserve dense tag");

    // Reference reflecting the single-point update.
    let mut ref_entries = entries.clone();
    for (k, v) in ref_entries.iter_mut() {
        if *k == key {
            *v = new_val.clone();
        }
    }
    for n in 1..=4 {
        for k in 1..=4 {
            let arg = Value::tuple([Value::int(n), Value::int(k)]);
            assert_eq!(
                updated.apply(&arg),
                reference_apply(&ref_entries, &arg),
                "post-EXCEPT apply({arg}) disagreed"
            );
            // Direct-index path must also see the overlay.
            assert_eq!(updated.apply2_dense(n, k), updated.apply(&arg));
        }
    }
}

#[test]
fn empty_function_is_sparse_and_misses() {
    let fv = FuncValue::from_sorted_entries(vec![]);
    assert!(!fv.dense_is_dim2());
    assert_eq!(fv.apply(&Value::int(1)), None);
    assert_eq!(fv.apply2_dense(1, 1), None);
}

// === Virtual-tuple apply (`apply_tuple_elems`) ===
//
// Contract under test: for ANY FuncValue and ANY component slice `elems`,
// `fv.apply_tuple_elems(&elems)` must equal `fv.apply(&Value::Tuple(elems))`.
// Agreement is checked against BOTH the materialized `apply` and the linear
// reference, across dense-2D, sparse-tuple, mixed-arity, mixed-type, and
// overlaid domains — the shapes the virtual apply is routed for.

/// Assert the virtual apply equals the materialized apply AND the reference.
fn assert_virtual_matches(fv: &FuncValue, entries: &[(Value, Value)], elems: &[Value]) {
    let materialized = Value::Tuple(Arc::from(elems.to_vec()));
    assert_eq!(
        fv.apply_tuple_elems(elems),
        fv.apply(&materialized),
        "apply_tuple_elems({materialized}) diverged from materialized apply"
    );
    assert_eq!(
        fv.apply_tuple_elems(elems),
        reference_apply(entries, &materialized),
        "apply_tuple_elems({materialized}) diverged from linear reference"
    );
}

#[test]
fn virtual_tuple2_matches_apply_on_dense_grid() {
    let entries = cross_entries(1, 4, 1, 3);
    let fv = FuncValue::from_sorted_entries(entries.clone());
    assert!(fv.dense_is_dim2());
    for n in 0..=5 {
        for k in 0..=4 {
            assert_virtual_matches(&fv, &entries, &[Value::int(n), Value::int(k)]);
        }
    }
    // Non-int components, wrong arity, non-tuple-ish shapes.
    assert_virtual_matches(&fv, &entries, &[Value::string("x"), Value::int(1)]);
    assert_virtual_matches(&fv, &entries, &[Value::int(1)]);
    assert_virtual_matches(
        &fv,
        &entries,
        &[Value::int(1), Value::int(2), Value::int(3)],
    );
}

#[test]
fn virtual_tuple2_matches_apply_on_sparse_tuple_domain() {
    // Sparse (incomplete) tuple grid + a hole: stays DenseTag::Sparse, so the
    // virtual apply exercises the component-wise binary search.
    let mut entries = cross_entries(1, 3, 1, 3);
    entries.remove(4); // punch a hole -> not a dense cross-product
    let fv = FuncValue::from_sorted_entries(entries.clone());
    assert!(!fv.dense_is_dim2());
    for n in 0..=4 {
        for k in 0..=4 {
            assert_virtual_matches(&fv, &entries, &[Value::int(n), Value::int(k)]);
        }
    }
}

#[test]
fn virtual_tuple_matches_apply_on_mixed_type_domain() {
    // Mixed domain: scalars, strings, and tuple keys of different arities in
    // one function. Virtual lookup must land exactly where the materialized
    // tuple's Value::cmp binary search lands.
    let mut entries: Vec<(Value, Value)> = vec![
        (Value::int(7), Value::string("scalar")),
        (Value::string("s"), Value::string("str")),
        (
            Value::tuple([Value::int(1), Value::int(2)]),
            Value::string("t12"),
        ),
        (
            Value::tuple([Value::int(1), Value::string("b")]),
            Value::string("t1b"),
        ),
        (
            Value::tuple([Value::int(1), Value::int(2), Value::int(3)]),
            Value::string("t123"),
        ),
        (Value::tuple([Value::int(9)]), Value::string("t9")),
    ];
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let fv = FuncValue::from_sorted_entries(entries.clone());

    // Hits at every arity, misses nearby.
    assert_virtual_matches(&fv, &entries, &[Value::int(1), Value::int(2)]);
    assert_virtual_matches(&fv, &entries, &[Value::int(1), Value::string("b")]);
    assert_virtual_matches(
        &fv,
        &entries,
        &[Value::int(1), Value::int(2), Value::int(3)],
    );
    assert_virtual_matches(&fv, &entries, &[Value::int(9)]);
    assert_virtual_matches(&fv, &entries, &[Value::int(1), Value::int(3)]);
    assert_virtual_matches(&fv, &entries, &[Value::int(2), Value::int(2)]);
    assert_virtual_matches(&fv, &entries, &[Value::string("s")]);
}

#[test]
fn virtual_tuple_sees_except_overlay() {
    // The virtual lookup goes through get_value_at, so a live EXCEPT overlay
    // must be visible exactly as through the materialized apply.
    let entries = cross_entries(1, 3, 1, 3);
    let fv = FuncValue::from_sorted_entries(entries.clone());
    let key = Value::tuple([Value::int(2), Value::int(3)]);
    let updated = fv.except(key.clone(), Value::string("patched"));
    let mut ref_entries = entries.clone();
    for (k, v) in ref_entries.iter_mut() {
        if *k == key {
            *v = Value::string("patched");
        }
    }
    for n in 1..=3 {
        for k in 1..=3 {
            assert_virtual_matches(&updated, &ref_entries, &[Value::int(n), Value::int(k)]);
        }
    }
}
