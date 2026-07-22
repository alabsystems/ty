// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for allocation-free equality with the empty tuple.

use crate::rp::Rp;

use super::super::*;
use std::sync::Arc;
use tla_core::ast::{BoundVar, Expr};
use tla_core::kani_types::HashMap;
use tla_core::name_intern::NameId;

fn empty_func() -> Value {
    Value::Func(Rp::new(FuncValue::from_sorted_entries(Vec::new())))
}

fn nonempty_func() -> Value {
    Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![(
        Value::SmallInt(1),
        Value::SmallInt(10),
    )])))
}

fn nonempty_bag() -> Value {
    Value::Bag(Rp::new(
        BagValue::try_from_entries(vec![(Value::string("x"), Value::SmallInt(1))])
            .expect("compact bags are enabled in value tests"),
    ))
}

fn lazy_func() -> Value {
    let bound = BoundVar {
        name: Spanned::dummy("x".to_string()),
        domain: None,
        pattern: None,
    };
    let body = Spanned::dummy(Expr::Ident("x".to_string(), NameId::INVALID));
    let captures = LazyFuncCaptures::new(Arc::new(HashMap::default()), None, None, None);
    Value::LazyFunc(Rp::new(LazyFuncValue::new(
        None,
        LazyDomain::Int,
        bound,
        body,
        captures,
    )))
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn equals_empty_tuple_matches_value_equality_across_representations() {
    let cases = [
        ("empty tuple", Value::tuple([])),
        ("nonempty tuple", Value::tuple([Value::SmallInt(1)])),
        ("empty sequence", Value::seq([])),
        ("nonempty sequence", Value::seq([Value::SmallInt(1)])),
        ("empty function", empty_func()),
        ("nonempty function", nonempty_func()),
        (
            "empty integer-interval function",
            Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 0, Vec::new()))),
        ),
        (
            "nonempty integer-interval function",
            Value::IntFunc(Rp::new(IntIntervalFunc::new(
                1,
                1,
                vec![Value::SmallInt(1)],
            ))),
        ),
        (
            "empty record",
            Value::record(std::iter::empty::<(&'static str, Value)>()),
        ),
        (
            "nonempty record",
            Value::record([("x", Value::SmallInt(1))]),
        ),
        ("empty bag", Value::Bag(BagValue::empty_arc())),
        ("nonempty bag", nonempty_bag()),
        ("lazy function", lazy_func()),
        ("empty eager set", Value::empty_set()),
        ("lazy set", Value::tuple_set(std::iter::empty::<Value>())),
        ("boolean", Value::Bool(false)),
        ("integer", Value::SmallInt(0)),
        ("string", Value::string("")),
    ];
    let empty_tuple = Value::tuple([]);

    for (case, value) in cases {
        assert_eq!(value.equals_empty_tuple(), value == empty_tuple, "{case}");
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn only_empty_function_representations_match() {
    for value in [
        Value::tuple([]),
        Value::seq([]),
        empty_func(),
        Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 0, Vec::new()))),
        Value::record(std::iter::empty::<(&'static str, Value)>()),
        Value::Bag(BagValue::empty_arc()),
    ] {
        assert!(value.equals_empty_tuple(), "{value:?}");
    }

    for value in [
        nonempty_bag(),
        lazy_func(),
        Value::empty_set(),
        Value::tuple_set(std::iter::empty::<Value>()),
        Value::Bool(false),
        Value::SmallInt(0),
        Value::string(""),
    ] {
        assert!(!value.equals_empty_tuple(), "{value:?}");
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn equals_tuple_elements_matches_materialized_tuple_equality() {
    let elements = [Value::SmallInt(3), Value::SmallInt(4)];
    let materialized = Value::tuple(elements.clone());
    let equivalent_bag = Value::Bag(Rp::new(
        BagValue::try_from_entries(vec![
            (Value::SmallInt(1), Value::SmallInt(3)),
            (Value::SmallInt(2), Value::SmallInt(4)),
        ])
        .expect("positive compact bag counts"),
    ));
    let cases = [
        Value::tuple(elements.clone()),
        Value::seq(elements.clone()),
        Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::SmallInt(1), elements[0].clone()),
            (Value::SmallInt(2), elements[1].clone()),
        ]))),
        Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, elements.to_vec()))),
        equivalent_bag,
        Value::tuple([Value::SmallInt(3)]),
        Value::IntFunc(Rp::new(IntIntervalFunc::new(0, 1, elements.to_vec()))),
        Value::record([("one", Value::SmallInt(3)), ("two", Value::SmallInt(4))]),
        Value::SmallInt(3),
    ];

    for value in cases {
        assert_eq!(
            value.equals_tuple_elements(&elements),
            value == materialized,
            "{value:?}"
        );
    }
}
