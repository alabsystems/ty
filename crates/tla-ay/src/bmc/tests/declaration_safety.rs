// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fail-closed declaration-boundary regressions for BMC carrier namespaces.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestCarrier {
    Scalar,
    Rigid,
    Function,
    Sequence,
    Record,
    Tuple,
}

impl TestCarrier {
    const ALL: [Self; 6] = [
        Self::Scalar,
        Self::Rigid,
        Self::Function,
        Self::Sequence,
        Self::Record,
        Self::Tuple,
    ];

    fn declare(self, bmc: &mut BmcTranslator, name: &str) -> AYResult<()> {
        match self {
            Self::Scalar => bmc.declare_var(name, TlaSort::Int),
            Self::Rigid => bmc.declare_rigid_const(name, TlaSort::Int),
            Self::Function => bmc.declare_func_var(name, TlaSort::Int),
            Self::Sequence => bmc.declare_seq_var(name, TlaSort::Int, 3),
            Self::Record => bmc.declare_record_var(name, vec![("field".to_string(), TlaSort::Int)]),
            Self::Tuple => bmc.declare_tuple_var(name, vec![TlaSort::Int]),
        }
    }
}

fn carrier_count(bmc: &BmcTranslator, name: &str) -> usize {
    usize::from(bmc.vars.contains_key(name))
        + usize::from(bmc.func_vars.contains_key(name))
        + usize::from(bmc.seq_vars.contains_key(name))
        + usize::from(bmc.record_vars.contains_key(name))
        + usize::from(bmc.tuple_vars.contains_key(name))
}

#[test]
fn test_bmc_scalar_symbol_api_round_trips_adversarial_names() {
    let names = [
        "",
        "x",
        "x__0",
        "name_step_12",
        "__ty_bmc_state_1_78_step_0",
        "__ty_bmc_rigid_1_78",
        "snowman_☃",
        "embedded\0nul",
    ];
    let steps = [0, 1, 12, usize::MAX];
    let mut symbols = std::collections::HashSet::new();

    for name in names {
        let rigid = BmcTranslator::rigid_const_symbol(name);
        assert!(symbols.insert(rigid.clone()), "duplicate symbol {rigid}");
        assert_eq!(
            BmcTranslator::parse_scalar_symbol(&rigid),
            Some(BmcScalarSymbol::Rigid {
                name: name.to_string()
            })
        );

        for step in steps {
            let state = BmcTranslator::state_step_symbol(name, step);
            assert!(symbols.insert(state.clone()), "duplicate symbol {state}");
            assert_eq!(
                BmcTranslator::parse_scalar_symbol(&state),
                Some(BmcScalarSymbol::State {
                    name: name.to_string(),
                    step
                })
            );
        }
    }

    for malformed in [
        "x__0",
        "__ty_bmc_state_1_78_step_00",
        "__ty_bmc_state_01_78_step_0",
        "__ty_bmc_state_1_7A_step_0",
        "__ty_bmc_state_2_78_step_0",
        "__ty_bmc_rigid_1_7",
        "__ty_bmc_aux_0_purpose_1_78",
    ] {
        assert_eq!(
            BmcTranslator::parse_scalar_symbol(malformed),
            None,
            "malformed/non-scalar spelling was accepted: {malformed}"
        );
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_cross_carrier_name_collisions_reject_in_every_order() {
    for first in TestCarrier::ALL {
        for second in TestCarrier::ALL {
            if first == second {
                continue;
            }

            let mut bmc = BmcTranslator::new_with_arrays(1).unwrap();
            first.declare(&mut bmc, "shared").unwrap();
            assert_eq!(carrier_count(&bmc, "shared"), 1);

            let error = second
                .declare(&mut bmc, "shared")
                .expect_err("a name cannot acquire a second solver carrier");
            assert!(
                matches!(error, AYError::TypeMismatch { .. }),
                "unexpected {first:?} -> {second:?} collision error: {error}"
            );
            assert_eq!(
                carrier_count(&bmc, "shared"),
                1,
                "failed {first:?} -> {second:?} declaration mutated carrier maps"
            );

            first
                .declare(&mut bmc, "shared")
                .expect("collision rejection must preserve the original declaration");
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_same_carrier_redeclaration_requires_exact_shape() {
    let mut bmc = BmcTranslator::new_with_arrays(1).unwrap();

    bmc.declare_var("scalar", TlaSort::Int).unwrap();
    assert!(matches!(
        bmc.declare_var("scalar", TlaSort::Bool),
        Err(AYError::TypeMismatch { .. })
    ));
    bmc.declare_var("scalar", TlaSort::Int).unwrap();

    bmc.declare_rigid_const("rigid", TlaSort::Int).unwrap();
    assert!(matches!(
        bmc.declare_rigid_const("rigid", TlaSort::String),
        Err(AYError::TypeMismatch { .. })
    ));
    bmc.declare_rigid_const("rigid", TlaSort::Int).unwrap();

    bmc.declare_func_var_with_key_sort("function", TlaSort::Int, TlaSort::Int)
        .unwrap();
    assert!(matches!(
        bmc.declare_func_var_with_key_sort("function", TlaSort::Int, TlaSort::Bool),
        Err(AYError::TypeMismatch { .. })
    ));
    assert!(matches!(
        bmc.declare_func_var_with_key_sort("function", TlaSort::String, TlaSort::Int),
        Err(AYError::TypeMismatch { .. })
    ));
    bmc.declare_func_var_with_key_sort("function", TlaSort::Int, TlaSort::Int)
        .unwrap();

    bmc.declare_func_var("upgraded_function", TlaSort::Int)
        .unwrap();
    bmc.upgrade_func_key_sort_to_string("upgraded_function")
        .unwrap();
    bmc.declare_func_var("upgraded_function", TlaSort::Int)
        .expect("generic re-declaration must preserve a one-way String-key upgrade");
    assert!(matches!(
        bmc.declare_func_var_with_key_sort("upgraded_function", TlaSort::Int, TlaSort::Int),
        Err(AYError::TypeMismatch { .. })
    ));

    bmc.declare_funcsym_var("symbolic", 0, "N".to_string(), -1, TlaSort::Int)
        .unwrap();
    assert!(matches!(
        bmc.declare_funcsym_var("symbolic", 1, "N".to_string(), -1, TlaSort::Int),
        Err(AYError::TypeMismatch { .. })
    ));
    assert!(matches!(
        bmc.declare_func_var("symbolic", TlaSort::Int),
        Err(AYError::TypeMismatch { .. })
    ));

    bmc.declare_seq_var("sequence", TlaSort::Int, 3).unwrap();
    assert!(matches!(
        bmc.declare_seq_var("sequence", TlaSort::String, 3),
        Err(AYError::TypeMismatch { .. })
    ));
    assert!(matches!(
        bmc.declare_seq_var("sequence", TlaSort::Int, 4),
        Err(AYError::TypeMismatch { .. })
    ));
    bmc.declare_seq_var("sequence", TlaSort::Int, 3).unwrap();

    bmc.declare_record_var("record", vec![("field".to_string(), TlaSort::Int)])
        .unwrap();
    assert!(matches!(
        bmc.declare_record_var("record", vec![("field".to_string(), TlaSort::Bool)]),
        Err(AYError::TypeMismatch { .. })
    ));

    bmc.declare_tuple_var("tuple", vec![TlaSort::Int]).unwrap();
    assert!(matches!(
        bmc.declare_tuple_var("tuple", vec![TlaSort::String]),
        Err(AYError::TypeMismatch { .. })
    ));
}

/// Every old raw-concatenation collision remains independently assignable.
/// This pins both source-carrier namespaces and the disjoint auxiliary prefix.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_adversarial_generated_symbol_names_are_independent() {
    fn constrain_int_pair(bmc: &mut BmcTranslator, left: Term, right: Term) {
        assert_ne!(left, right, "logical carriers must have distinct TermIds");
        let zero = bmc.solver.int_const(0);
        let one = bmc.solver.int_const(1);
        let left_zero = bmc.solver.try_eq(left, zero).unwrap();
        let right_one = bmc.solver.try_eq(right, one).unwrap();
        bmc.assert(left_zero);
        bmc.assert(right_one);
    }

    fn constrain_array_pair(bmc: &mut BmcTranslator, left: Term, right: Term) {
        assert_ne!(left, right, "logical arrays must have distinct TermIds");
        let key = bmc.solver.int_const(0);
        let left_member = bmc.solver.try_select(left, key).unwrap();
        let right_member = bmc.solver.try_select(right, key).unwrap();
        let right_absent = bmc.solver.try_not(right_member).unwrap();
        bmc.assert(left_member);
        bmc.assert(right_absent);
    }

    let mut bmc = BmcTranslator::new_with_arrays(1).unwrap();

    // Old collision: scalar r__f_x at step 0 vs record r field x.
    bmc.declare_var("r__f_x", TlaSort::Int).unwrap();
    bmc.declare_record_var("r", vec![("x".to_string(), TlaSort::Int)])
        .unwrap();
    let scalar_record_stem = bmc.get_var_at_step("r__f_x", 0).unwrap();
    let record_field = bmc.get_record_field_at_step("r", "x", 0).unwrap();
    constrain_int_pair(&mut bmc, scalar_record_stem, record_field);

    // Old collision: scalar t__e_1 at step 0 vs tuple t element 1.
    bmc.declare_var("t__e_1", TlaSort::Int).unwrap();
    bmc.declare_tuple_var("t", vec![TlaSort::Int]).unwrap();
    let scalar_tuple_stem = bmc.get_var_at_step("t__e_1", 0).unwrap();
    let tuple_element = bmc.get_tuple_element_at_step("t", 1, 0).unwrap();
    constrain_int_pair(&mut bmc, scalar_tuple_stem, tuple_element);

    // Old collisions with the function domain and sequence array/length.
    let set_int = TlaSort::Set {
        element_sort: Box::new(TlaSort::Int),
    };
    bmc.declare_var("f__dom", set_int.clone()).unwrap();
    bmc.declare_func_var("f", TlaSort::Int).unwrap();
    let scalar_function_domain = bmc.get_var_at_step("f__dom", 0).unwrap();
    let function_domain = bmc.get_func_domain_at_step("f", 0).unwrap();
    constrain_array_pair(&mut bmc, scalar_function_domain, function_domain);

    bmc.declare_var("s__arr", set_int).unwrap();
    bmc.declare_var("s__len", TlaSort::Int).unwrap();
    bmc.declare_seq_var("s", TlaSort::Bool, 3).unwrap();
    let scalar_sequence_array = bmc.get_var_at_step("s__arr", 0).unwrap();
    let sequence_array = bmc.get_seq_array_at_step("s", 0).unwrap();
    constrain_array_pair(&mut bmc, scalar_sequence_array, sequence_array);
    let scalar_sequence_length = bmc.get_var_at_step("s__len", 0).unwrap();
    let sequence_length = bmc.get_seq_length_at_step("s", 0).unwrap();
    constrain_int_pair(&mut bmc, scalar_sequence_length, sequence_length);

    // Old collision: state x step 0 vs a rigid constant literally named x__0.
    bmc.declare_var("x", TlaSort::Int).unwrap();
    bmc.declare_rigid_const("x__0", TlaSort::Int).unwrap();
    let state_x = bmc.get_var_at_step("x", 0).unwrap();
    let rigid_step_spelling = bmc.get_var_at_step("x__0", 0).unwrap();
    constrain_int_pair(&mut bmc, state_x, rigid_step_spelling);

    // A source name spelling the exact next auxiliary symbol is encoded in the
    // rigid namespace; it cannot capture the internal declaration.
    let internal_spelling = "__ty_bmc_aux_0_purpose_1_78";
    bmc.declare_rigid_const(internal_spelling, TlaSort::Int)
        .unwrap();
    let source_internal_spelling = bmc.get_var_at_step(internal_spelling, 0).unwrap();
    let (internal_name, internal_term) = bmc.declare_internal_const("x", Sort::Int);
    assert_ne!(internal_name, internal_spelling);
    assert!(internal_name.starts_with("__ty_bmc_aux_1_"));
    constrain_int_pair(&mut bmc, source_internal_spelling, internal_term);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}
