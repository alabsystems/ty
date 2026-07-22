// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::rp::Rp;

use super::*;
use crate::Value;

#[test]
fn test_size() {
    assert_eq!(std::mem::size_of::<CompactValue>(), 8);
}

#[test]
fn value_heap_encoding_query_matches_conversion_policy() {
    use num_bigint::BigInt;

    let max_inline = (1_i64 << 60) - 1;
    let cases = [
        (Value::Bool(true), false),
        (Value::SmallInt(max_inline), false),
        (Value::SmallInt(max_inline + 1), true),
        (Value::Int(Rp::new(BigInt::from(42))), false),
        (Value::Int(Rp::new(BigInt::from(max_inline + 1))), true),
        (Value::set([Value::int(1)]), true),
    ];

    for (value, expected_heap) in cases {
        assert_eq!(
            CompactValue::value_uses_heap_encoding(&value),
            expected_heap,
            "encoding query drifted for {value:?}"
        );
        assert_eq!(CompactValue::from(&value).is_heap(), expected_heap);
    }
}

#[test]
fn test_bool_values() {
    let f = CompactValue::from_bool(false);
    let t = CompactValue::from_bool(true);
    assert!(f.is_bool());
    assert!(t.is_bool());
    assert!(!f.as_bool());
    assert!(t.as_bool());
    assert_eq!(f, CompactValue::bool_false());
    assert_eq!(t, CompactValue::bool_true());
    assert_ne!(f, t);
}

#[test]
fn test_small_int() {
    let zero = CompactValue::from_int(0);
    assert!(zero.is_int());
    assert_eq!(zero.as_int(), 0);

    let pos = CompactValue::from_int(42);
    assert_eq!(pos.as_int(), 42);

    let neg = CompactValue::from_int(-100);
    assert_eq!(neg.as_int(), -100);

    let max = CompactValue::from_int(MAX_SMALL_INT);
    assert_eq!(max.as_int(), MAX_SMALL_INT);

    let min = CompactValue::from_int(MIN_SMALL_INT);
    assert_eq!(min.as_int(), MIN_SMALL_INT);
}

#[test]
fn test_int_roundtrip() {
    for &n in &[
        0_i64,
        1,
        -1,
        42,
        -42,
        1000000,
        -1000000,
        MAX_SMALL_INT,
        MIN_SMALL_INT,
    ] {
        let cv = CompactValue::from_int(n);
        assert_eq!(cv.as_int(), n, "roundtrip failed for {n}");
    }
}

#[test]
fn test_int_overflow() {
    assert!(CompactValue::try_from_int(MAX_SMALL_INT + 1).is_none());
    assert!(CompactValue::try_from_int(MIN_SMALL_INT - 1).is_none());
}

#[test]
fn test_string_id() {
    let s = CompactValue::from_interned_string(42);
    assert!(s.is_string());
    assert_eq!(s.as_string_id(), 42);
}

#[test]
fn test_model_value_id() {
    let mv = CompactValue::from_model_value(7);
    assert!(mv.is_model_value());
    assert_eq!(mv.as_model_value_id(), 7);
}

#[test]
fn test_nil() {
    let n = CompactValue::nil();
    assert!(n.is_nil());
    assert!(!n.is_bool());
    assert!(!n.is_int());
    assert!(!n.is_heap());
}

#[test]
fn test_heap_value_bool() {
    let cv = CompactValue::from(Value::Bool(true));
    assert!(cv.is_bool());
    assert!(cv.as_bool());
}

#[test]
fn test_heap_value_int() {
    let cv = CompactValue::from(Value::SmallInt(99));
    assert!(cv.is_int());
    assert_eq!(cv.as_int(), 99);
}

#[test]
fn test_small_bigint_normalizes_to_inline_int() {
    let value = Value::Int(Rp::new(num_bigint::BigInt::from(99)));

    let owned = CompactValue::from(value.clone());
    let borrowed = CompactValue::from(&value);
    assert!(owned.is_int());
    assert!(borrowed.is_int());
    assert_eq!(owned.as_int(), 99);
    assert_eq!(borrowed.as_int(), 99);
    assert!(owned.matches_value(&value));
    assert!(owned.matches_compact_encoding(&value));
    assert_eq!(owned, CompactValue::from(Value::SmallInt(99)));
}

#[test]
fn test_large_bigint_remains_heap_backed() {
    let huge = num_bigint::BigInt::from(1_u8) << 80;
    let value = Value::Int(Rp::new(huge));

    let compact = CompactValue::from(&value);
    assert!(compact.is_heap());
    assert_eq!(compact.as_heap_value(), &value);
    assert!(compact.matches_compact_encoding(&value));
}

#[test]
fn test_integer_compact_encoding_i61_boundaries() {
    for value in [MIN_SMALL_INT, MAX_SMALL_INT] {
        let small = Value::SmallInt(value);
        let big = Value::Int(Rp::new(num_bigint::BigInt::from(value)));
        let small_compact = CompactValue::from(&small);
        let big_compact = CompactValue::from(&big);
        assert!(small_compact.is_int());
        assert_eq!(small_compact, big_compact);
        assert!(small_compact.matches_compact_encoding(&small));
        assert!(small_compact.matches_compact_encoding(&big));
    }

    for value in [MIN_SMALL_INT - 1, MAX_SMALL_INT + 1] {
        let small = Value::SmallInt(value);
        let big = Value::Int(Rp::new(num_bigint::BigInt::from(value)));
        let small_compact = CompactValue::from(&small);
        let big_compact = CompactValue::from(&big);
        assert!(small_compact.is_heap());
        assert!(big_compact.is_heap());
        assert_eq!(small_compact, big_compact);
        assert!(small_compact.matches_compact_encoding(&small));
        assert!(big_compact.matches_compact_encoding(&big));
    }
}

#[test]
fn test_heap_value_compound() {
    let v = Value::set(vec![
        Value::SmallInt(1),
        Value::SmallInt(2),
        Value::SmallInt(3),
    ]);
    let cv = CompactValue::from(v.clone());
    assert!(cv.is_heap());
    assert_eq!(cv.as_heap_value(), &v);
}

#[test]
fn test_clone_inline() {
    let cv = CompactValue::from_int(42);
    let cv2 = cv.clone();
    assert_eq!(cv, cv2);
    assert_eq!(cv.as_int(), 42);
    assert_eq!(cv2.as_int(), 42);
}

#[test]
fn test_clone_heap() {
    let v = Value::set(vec![Value::SmallInt(1), Value::SmallInt(2)]);
    let cv = CompactValue::from(v.clone());
    let cv2 = cv.clone();
    assert_eq!(cv, cv2);
    assert_eq!(cv.as_heap_value(), &v);
    assert_eq!(cv2.as_heap_value(), &v);
}

#[test]
fn test_equality() {
    assert_eq!(CompactValue::from_bool(true), CompactValue::from_bool(true));
    assert_ne!(
        CompactValue::from_bool(true),
        CompactValue::from_bool(false)
    );
    assert_eq!(CompactValue::from_int(42), CompactValue::from_int(42));
    assert_ne!(CompactValue::from_int(42), CompactValue::from_int(43));
    // Different tags are never equal.
    assert_ne!(CompactValue::from_bool(true), CompactValue::from_int(1));
}

#[test]
fn test_value_to_compact_roundtrip_bool() {
    let v = Value::Bool(true);
    let cv = CompactValue::from(v.clone());
    let v2: Value = cv.into();
    assert_eq!(v, v2);
}

#[test]
fn test_value_to_compact_roundtrip_int() {
    let v = Value::SmallInt(42);
    let cv = CompactValue::from(v.clone());
    let v2: Value = cv.into();
    assert_eq!(v, v2);
}

#[test]
fn test_value_to_compact_roundtrip_set() {
    let v = Value::set(vec![
        Value::SmallInt(1),
        Value::SmallInt(2),
        Value::SmallInt(3),
    ]);
    let cv = CompactValue::from(v.clone());
    let v2: Value = cv.into();
    assert_eq!(v, v2);
}

#[test]
fn test_debug_format() {
    let cv = CompactValue::from_bool(true);
    assert_eq!(format!("{cv:?}"), "CompactValue::Bool(true)");

    let cv = CompactValue::from_int(-5);
    assert_eq!(format!("{cv:?}"), "CompactValue::Int(-5)");
}

#[test]
fn test_inline_check() {
    assert!(CompactValue::from_bool(true).is_inline());
    assert!(CompactValue::from_int(42).is_inline());
    assert!(CompactValue::from_interned_string(1).is_inline());
    assert!(CompactValue::from_model_value(1).is_inline());
    assert!(CompactValue::nil().is_inline());

    let v = Value::set(vec![Value::SmallInt(1)]);
    let cv = CompactValue::from(v);
    assert!(!cv.is_inline());
}
