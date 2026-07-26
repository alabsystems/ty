// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Sort-directed model extraction and concrete-state round trips.

use super::*;

fn solve_step_zero(translator: &mut BmcTranslator) -> BmcState {
    assert_eq!(translator.try_check_sat().unwrap(), SolveResult::Sat);
    let model = translator.try_get_model().unwrap();
    translator
        .extract_trace(&model)
        .into_iter()
        .find(|state| state.step == 0)
        .expect("step zero")
}

#[test]
fn trace_extraction_keeps_intern_id_int_and_string_values_disjoint() {
    let mut translator = BmcTranslator::new(0).unwrap();
    let intern_id = translator.bmc_intern_string("literal");
    translator.declare_var("integer", TlaSort::Int).unwrap();
    translator.declare_var("string", TlaSort::String).unwrap();

    let integer = translator.get_var_at_step("integer", 0).unwrap();
    let string = translator.get_var_at_step("string", 0).unwrap();
    let raw = translator.solver.int_const(intern_id);
    let integer_eq = translator.solver.try_eq(integer, raw).unwrap();
    let string_eq = translator.solver.try_eq(string, raw).unwrap();
    translator.assert(integer_eq);
    translator.assert(string_eq);

    let state = solve_step_zero(&mut translator);
    assert_eq!(
        state.assignments.get("integer"),
        Some(&BmcValue::Int(intern_id))
    );
    assert_eq!(
        state.assignments.get("string"),
        Some(&BmcValue::String("literal".to_string()))
    );
}

#[test]
fn bool_sequence_and_function_round_trip_with_native_bool_carriers() {
    let mut translator = BmcTranslator::new_with_arrays(0).unwrap();
    translator
        .declare_seq_var("sequence", TlaSort::Bool, 3)
        .unwrap();
    translator
        .declare_func_var("function", TlaSort::Bool)
        .unwrap();

    let sequence = BmcValue::Sequence(vec![BmcValue::Bool(true), BmcValue::Bool(false)]);
    let function = BmcValue::Function(vec![(1, BmcValue::Bool(false)), (3, BmcValue::Bool(true))]);
    translator
        .assert_concrete_state(
            &[
                ("sequence".to_string(), sequence.clone()),
                ("function".to_string(), function.clone()),
            ],
            0,
        )
        .unwrap();

    let state = solve_step_zero(&mut translator);
    assert_eq!(state.assignments.get("sequence"), Some(&sequence));
    assert_eq!(state.assignments.get("function"), Some(&function));
}

#[test]
fn native_string_key_function_round_trips_without_integer_aliasing() {
    let mut translator = BmcTranslator::new_with_arrays(0).unwrap();
    translator
        .declare_func_var_with_key_sort("function", TlaSort::String, TlaSort::Bool)
        .unwrap();
    let function = BmcValue::StringFunction(vec![
        ("1".to_string(), BmcValue::Bool(true)),
        ("key__0".to_string(), BmcValue::Bool(false)),
    ]);
    translator
        .assert_concrete_state(&[("function".to_string(), function.clone())], 0)
        .unwrap();

    let state = solve_step_zero(&mut translator);
    assert_eq!(state.assignments.get("function"), Some(&function));
}

#[test]
fn symbolic_domain_bool_function_materializes_domain_from_rigid_bound() {
    let mut translator = BmcTranslator::new_with_arrays(0).unwrap();
    translator.declare_rigid_const("N", TlaSort::Int).unwrap();
    translator
        .declare_funcsym_var("function", 1, "N".to_string(), 0, TlaSort::Bool)
        .unwrap();
    let function = BmcValue::Function(vec![(1, BmcValue::Bool(true)), (2, BmcValue::Bool(false))]);
    translator
        .assert_concrete_state(
            &[
                ("N".to_string(), BmcValue::Int(2)),
                ("function".to_string(), function.clone()),
            ],
            0,
        )
        .unwrap();

    let state = solve_step_zero(&mut translator);
    assert_eq!(state.assignments.get("N"), Some(&BmcValue::Int(2)));
    assert_eq!(state.assignments.get("function"), Some(&function));
}

#[test]
fn record_set_string_and_mixed_tuple_round_trip_by_declared_sort() {
    let mut translator = BmcTranslator::new_with_arrays(0).unwrap();
    translator
        .declare_record_var(
            "record",
            vec![
                (
                    "names".to_string(),
                    TlaSort::Set {
                        element_sort: Box::new(TlaSort::String),
                    },
                ),
                ("ready".to_string(), TlaSort::Bool),
            ],
        )
        .unwrap();
    translator
        .declare_tuple_var("tuple", vec![TlaSort::Bool, TlaSort::String, TlaSort::Int])
        .unwrap();

    let record = BmcValue::Record(vec![
        (
            "names".to_string(),
            BmcValue::Set(vec![
                BmcValue::String("alice".to_string()),
                BmcValue::String("bob".to_string()),
            ]),
        ),
        ("ready".to_string(), BmcValue::Bool(true)),
    ]);
    let tuple = BmcValue::Tuple(vec![
        BmcValue::Bool(false),
        BmcValue::String("alice".to_string()),
        BmcValue::Int(-1_000_000_007),
    ]);
    translator
        .assert_concrete_state(
            &[
                ("record".to_string(), record.clone()),
                ("tuple".to_string(), tuple.clone()),
            ],
            0,
        )
        .unwrap();

    let state = solve_step_zero(&mut translator);
    assert_eq!(state.assignments.get("record"), Some(&record));
    assert_eq!(state.assignments.get("tuple"), Some(&tuple));
}

#[test]
fn wavefront_encoding_is_sort_directed_for_bool_and_string_compounds() {
    let mut translator = BmcTranslator::new_with_arrays(0).unwrap();
    translator
        .declare_seq_var("sequence", TlaSort::Bool, 2)
        .unwrap();
    translator
        .declare_record_var(
            "record",
            vec![(
                "names".to_string(),
                TlaSort::Set {
                    element_sort: Box::new(TlaSort::String),
                },
            )],
        )
        .unwrap();
    let sequence = BmcValue::Sequence(vec![BmcValue::Bool(true)]);
    let record = BmcValue::Record(vec![(
        "names".to_string(),
        BmcValue::Set(vec![BmcValue::String("alice".to_string())]),
    )]);

    translator
        .assert_wavefront_formula(
            &[
                ("sequence".to_string(), sequence.clone()),
                ("record".to_string(), record.clone()),
            ],
            &[],
            0,
        )
        .unwrap();

    let state = solve_step_zero(&mut translator);
    assert_eq!(state.assignments.get("sequence"), Some(&sequence));
    assert_eq!(state.assignments.get("record"), Some(&record));
}

#[test]
fn unknown_string_intern_is_omitted_instead_of_misdecoded_as_int() {
    let mut translator = BmcTranslator::new(0).unwrap();
    translator.declare_var("string", TlaSort::String).unwrap();
    let string = translator.get_var_at_step("string", 0).unwrap();
    let unknown_id = translator.solver.int_const(42);
    let equality = translator.solver.try_eq(string, unknown_id).unwrap();
    translator.assert(equality);

    let state = solve_step_zero(&mut translator);
    assert!(!state.assignments.contains_key("string"));
}
