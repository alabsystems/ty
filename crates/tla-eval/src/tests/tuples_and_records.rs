// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_eval_range() {
    assert_eq!(
        eval_str("1..3").unwrap(),
        Value::set([Value::int(1), Value::int(2), Value::int(3)])
    );
    // Empty range
    let empty = eval_str("5..3").unwrap();
    assert!(empty.as_set().unwrap().is_empty());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_eval_tuples() {
    assert_eq!(
        eval_str("<<1, 2, 3>>").unwrap(),
        Value::tuple([Value::int(1), Value::int(2), Value::int(3)])
    );
}

/// Regression test for #62: tuple subscript evaluation
///
/// Tests that tuples can be indexed using 1-based subscripting.
/// This is the standard TLA+ pattern for accessing tuple elements.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_eval_tuple_subscript() {
    // Basic 1-based indexing
    assert_eq!(eval_str("<<10, 20, 30>>[1]").unwrap(), Value::int(10));
    assert_eq!(eval_str("<<10, 20, 30>>[2]").unwrap(), Value::int(20));
    assert_eq!(eval_str("<<10, 20, 30>>[3]").unwrap(), Value::int(30));

    // Nested tuple access
    assert_eq!(
        eval_str("<<1, <<2, 3>>, 4>>[2]").unwrap(),
        Value::tuple([Value::int(2), Value::int(3)])
    );
    assert_eq!(eval_str("<<1, <<2, 3>>, 4>>[2][1]").unwrap(), Value::int(2));
    assert_eq!(eval_str("<<1, <<2, 3>>, 4>>[2][2]").unwrap(), Value::int(3));

    // Index out of bounds should return IndexOutOfBounds
    assert!(matches!(
        eval_str("<<1, 2, 3>>[0]"),
        Err(EvalError::IndexOutOfBounds {
            index: 0,
            len: 3,
            ..
        })
    ));
    assert!(matches!(
        eval_str("<<1, 2, 3>>[4]"),
        Err(EvalError::IndexOutOfBounds {
            index: 4,
            len: 3,
            ..
        })
    ));
    assert!(matches!(
        eval_str("<<>>[1]"),
        Err(EvalError::IndexOutOfBounds {
            index: 1,
            len: 0,
            ..
        })
    ));

    // Empty tuple has no valid indices
    assert!(matches!(
        eval_str("<<>>[0]"),
        Err(EvalError::IndexOutOfBounds {
            index: 0,
            len: 0,
            ..
        })
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_eval_records() {
    let r = eval_str("[x |-> 1, y |-> 2]").unwrap();
    let rec = r.as_record().unwrap();
    assert_eq!(rec.get(&Arc::from("x")), Some(&Value::int(1)));
    assert_eq!(rec.get(&Arc::from("y")), Some(&Value::int(2)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_eval_record_set_with_interval_field() {
    let v = eval_str("[x: 1..3]").unwrap();
    assert!(matches!(v, Value::RecordSet(_)));
    assert_eq!(
        v,
        Value::set([
            Value::record([("x", Value::int(1))]),
            Value::record([("x", Value::int(2))]),
            Value::record([("x", Value::int(3))]),
        ])
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_choose_over_record_set_is_deterministic() {
    let v = eval_str(r#"CHOOSE r \in [x: 1..3] : TRUE"#).unwrap();
    let rec = v.as_record().unwrap();
    assert_eq!(rec.get("x"), Some(&Value::int(1)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cardinality_of_record_set() {
    assert_eq!(
        eval_str("Cardinality([a: 1..3, b: {1, 2}])").unwrap(),
        Value::int(6)
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_eval_tuple_pattern_binding() {
    // Test tuple pattern destructuring: {x + y : <<x, y>> \in S}
    let src = r#"
---- MODULE Test ----
S == {<<1, 2>>, <<3, 4>>}
F == {x + y : <<x, y>> \in S}
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    eprintln!("=== CST for tuple pattern ===");
    // Print CST structure
    fn print_tree(node: &tla_core::SyntaxNode, indent: usize) {
        let text = node.text().to_string();
        let text_len = text.len();
        let kind = node.kind();
        if text_len < 80 {
            eprintln!("{:indent$}{:?}: {:?}", "", kind, text, indent = indent);
        } else {
            eprintln!(
                "{:indent$}{:?}: <{} chars>",
                "",
                kind,
                text_len,
                indent = indent
            );
        }
        for child in node.children() {
            print_tree(&child, indent + 2);
        }
    }
    print_tree(&tree, 0);

    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    eprintln!("Lower errors: {:?}", lower_result.errors);
    let module = lower_result.module.unwrap();
    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            eprintln!("Op '{}' body: {:?}", def.name.node, def.body.node);
        }
    }

    // Setup eval context
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let result = ctx.eval_op("F");
    eprintln!("Tuple pattern binding result: {:?}", result);
    let val = result.unwrap();
    let set = val.as_set().unwrap();
    // {1+2, 3+4} = {3, 7}
    assert_eq!(set.len(), 2);
    assert!(set.contains(&Value::int(3)));
    assert!(set.contains(&Value::int(7)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_record_set_membership() {
    // Test: is [smoking |-> FALSE] in [smoking: BOOLEAN]?
    // And: is f \in [S -> [smoking: BOOLEAN]] where f = [x \in S |-> [smoking |-> FALSE]]?
    let src = r#"
---- MODULE Test ----
S == {1, 2}
BoolSet == BOOLEAN
RecSet == [smoking: BOOLEAN]
F == [x \in S |-> [smoking |-> FALSE]]
FuncSet == [S -> [smoking: BOOLEAN]]
RecordInRecSet == [smoking |-> FALSE] \in RecSet
FuncInFuncSet == F \in FuncSet
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    eprintln!("Lower errors: {:?}", lower_result.errors);
    let module = lower_result.module.unwrap();

    // Print AST for RecSet
    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            eprintln!("Op '{}' body: {:?}", def.name.node, def.body.node);
        }
    }

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // First check BOOLEAN evaluates correctly
    let bool_set = ctx.eval_op("BoolSet").expect("BoolSet");
    eprintln!("BoolSet (BOOLEAN) = {:?}", bool_set);

    // Evaluate each piece for debugging
    let rec_set = ctx.eval_op("RecSet").expect("RecSet");
    eprintln!("RecSet = {:?}", rec_set);

    let f = ctx.eval_op("F").expect("F");
    eprintln!("F = {:?}", f);

    let func_set = ctx.eval_op("FuncSet").expect("FuncSet");
    eprintln!("FuncSet = {:?}", func_set);

    let rec_in_rec_set = ctx.eval_op("RecordInRecSet").expect("RecordInRecSet");
    eprintln!("RecordInRecSet = {:?}", rec_in_rec_set);
    assert_eq!(
        rec_in_rec_set,
        Value::Bool(true),
        "Record should be in record set"
    );

    let func_in_func_set = ctx.eval_op("FuncInFuncSet").expect("FuncInFuncSet");
    eprintln!("FuncInFuncSet = {:?}", func_in_func_set);
    assert_eq!(
        func_in_func_set,
        Value::Bool(true),
        "Function should be in function set"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_zero_arg_filtered_record_set_preserves_streaming_shape_and_members() {
    let src = r#"
---- MODULE FilteredRecordZeroArg ----
EXTENDS Integers
RS == [a : 0..1, b : 0..1]
Filtered == {r \in RS : r.a + r.b = 1}
Wrapped == RS \ {[a |-> 0, b |-> 0]}
WrappedFiltered == {r \in Wrapped : TRUE}
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let filtered = ctx.eval_op("Filtered").expect("Filtered");
    assert!(
        matches!(filtered, Value::SetPred(_)),
        "a zero-arg direct RecordSet filter must retain SetPred shape for Init streaming: {filtered:?}"
    );
    let members: Vec<Value> = eval_iter_set(&ctx, &filtered, None)
        .expect("stream filtered records")
        .collect();
    assert_eq!(
        Value::set(members),
        Value::set([
            Value::record([("a", Value::int(0)), ("b", Value::int(1))]),
            Value::record([("a", Value::int(1)), ("b", Value::int(0))]),
        ]),
        "lazy preservation must not change members"
    );

    // The specialized fragment is deliberately direct-only. Set-algebra
    // wrappers retain the established eager small-set policy rather than being
    // silently claimed by the bulk streamer.
    assert!(
        matches!(
            ctx.eval_op("WrappedFiltered").expect("WrappedFiltered"),
            Value::Set(_)
        ),
        "a wrapped RecordSet domain follows the documented eager parity path"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_zero_arg_filtered_record_set_does_not_cache_captured_state() {
    clear_for_test_reset();

    let src = r#"
---- MODULE StatefulFilteredRecordZeroArg ----
EXTENDS Integers
VARIABLE x
RS == [a : 0..1]
Filtered == {r \in RS : r.a = x}
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    ctx.register_var("x");

    let state_zero = vec![Value::int(0)];
    ctx.bind_state_array(&state_zero);
    let filtered_zero = ctx.eval_op("Filtered").expect("Filtered at x = 0");
    assert!(matches!(filtered_zero, Value::SetPred(_)));
    if let Value::SetPred(value) = &filtered_zero {
        assert_eq!(value.captured_state(), Some(state_zero.as_slice()));
    }
    let members_zero = eval_iter_set(&ctx, &filtered_zero, None)
        .expect("stream records at x = 0")
        .collect::<Vec<_>>();
    assert_eq!(
        Value::set(members_zero),
        Value::set([Value::record([("a", Value::int(0))])])
    );

    let state_one = vec![Value::int(1)];
    ctx.bind_state_array(&state_one);
    let filtered_one = ctx.eval_op("Filtered").expect("Filtered at x = 1");
    assert!(matches!(filtered_one, Value::SetPred(_)));
    if let Value::SetPred(value) = &filtered_one {
        assert_eq!(
            value.captured_state(),
            Some(state_one.as_slice()),
            "the new wrapper must capture the rebound state"
        );
    }
    let members_one = eval_iter_set(&ctx, &filtered_one, None)
        .expect("stream records at x = 1")
        .collect::<Vec<_>>();
    assert_eq!(
        Value::set(members_one),
        Value::set([Value::record([("a", Value::int(1))])]),
        "the zero-arg cache must not reuse a SetPred that captured x = 0"
    );

    clear_for_test_reset();
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_record_dot_access_in_filter() {
    // Test: {r \in S : F[r].smoking = TRUE} where F[r] = [smoking |-> FALSE] for all r
    let src = r#"
---- MODULE Test ----
EXTENDS Integers, FiniteSets
S == {1, 2}
F == [r \in S |-> [smoking |-> FALSE]]
FApply == F[1]
FApplySmoking == F[1].smoking
FilterSet == {r \in S : F[r].smoking = TRUE}
FilterSetFalse == {r \in S : F[r].smoking = FALSE}
FilterCount == Cardinality(FilterSet)
AtMostOne == Cardinality({r \in S : F[r].smoking}) <= 1
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    assert!(lower_result.errors.is_empty());
    let module = lower_result.module.unwrap();

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // Verify F[1].smoking evaluates correctly
    let f_apply_smoking = ctx.eval_op("FApplySmoking").expect("FApplySmoking");
    assert_eq!(
        f_apply_smoking,
        Value::Bool(false),
        "F[1].smoking should be FALSE"
    );

    // Verify {r \in S : F[r].smoking = TRUE} is empty
    let filter_set = ctx.eval_op("FilterSet").expect("FilterSet");
    assert_eq!(
        filter_set,
        Value::empty_set(),
        "No elements should satisfy smoking = TRUE"
    );

    // Verify Cardinality is 0
    let filter_count = ctx.eval_op("FilterCount").expect("FilterCount");
    assert_eq!(
        filter_count,
        Value::int(0),
        "Cardinality of empty set should be 0"
    );

    // Verify AtMostOne is TRUE
    let at_most_one = ctx.eval_op("AtMostOne").expect("AtMostOne");
    assert_eq!(at_most_one, Value::Bool(true), "0 <= 1 should be TRUE");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cartesian_product_membership_nat_int() {
    // Test <<2, -3>> \in Nat \X Int - should be TRUE since 2 \in Nat and -3 \in Int
    let src = r#"
---- MODULE Test ----
EXTENDS Naturals, Integers
CrossProduct == Nat \X Int
TupleVal == <<2, -3>>
Result == TupleVal \in CrossProduct
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    eprintln!("Lower errors: {:?}", lower_result.errors);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // Debug: check what CrossProduct evaluates to
    let cross_result = ctx.eval_op("CrossProduct");
    eprintln!("CrossProduct = {:?}", cross_result);

    // Debug: check what TupleVal evaluates to
    let tuple_result = ctx.eval_op("TupleVal");
    eprintln!("TupleVal = {:?}", tuple_result);

    // Check the membership result
    let result = ctx.eval_op("Result");
    eprintln!("Result = {:?}", result);
    assert!(result.is_ok(), "Should not error: {:?}", result);
    assert_eq!(
        result.unwrap(),
        Value::Bool(true),
        "<<2, -3>> should be in Nat \\X Int"
    );
}
