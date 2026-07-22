// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for LP state-equation reachability pre-seeding.

use crate::lp_state_equation::MAX_FIREABILITY_CASE_SPLITS;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

use super::super::reachability::PropertyTracker;
use super::run_lp_seeding;

fn place(id: &str) -> PlaceInfo {
    PlaceInfo {
        id: id.to_string(),
        name: None,
    }
}

fn arc(place: u32, weight: u64) -> Arc {
    Arc {
        place: PlaceIdx(place),
        weight,
    }
}

fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
    TransitionInfo {
        id: id.to_string(),
        name: None,
        inputs,
        outputs,
    }
}

/// Token-conserving net: p0(3) → t0 → p1. Sum p0+p1 = 3 always.
fn conserving_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
        initial_marking: vec![3, 0],
    }
}

/// Mutex net: p_free <-> p_cs, so exactly one of enter/exit is enabled.
fn mutex_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p_free"), place("p_cs")],
        transitions: vec![
            trans("t_enter", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t_exit", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    }
}

/// Conserving net where t_both requires both places, but p0 + p1 = 1.
fn mutually_exclusive_input_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t01", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t10", vec![arc(1, 1)], vec![arc(0, 1)]),
            trans("t_both", vec![arc(0, 1), arc(1, 1)], vec![arc(0, 2)]),
        ],
        initial_marking: vec![1, 0],
    }
}

fn source_then_consume_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p")],
        transitions: vec![
            trans("t_source", vec![], vec![arc(0, 1)]),
            trans("t_consume", vec![arc(0, 1)], vec![]),
        ],
        initial_marking: vec![0],
    }
}

fn high_fan_in_fireability_net(input_count: usize) -> PetriNet {
    let mut transitions = Vec::with_capacity(input_count + 1);
    for place_index in 0..input_count {
        transitions.push(trans(
            &format!("source_{place_index}"),
            vec![],
            vec![arc(place_index as u32, 1)],
        ));
    }

    transitions.push(trans(
        "t_all",
        (0..input_count)
            .map(|place_index| arc(place_index as u32, 1))
            .collect(),
        vec![],
    ));

    PetriNet {
        name: None,
        places: (0..input_count)
            .map(|place_index| place(&format!("p{place_index}")))
            .collect(),
        transitions,
        initial_marking: vec![0; input_count],
    }
}

fn tracker(id: &str, quantifier: PathQuantifier, predicate: ResolvedPredicate) -> PropertyTracker {
    PropertyTracker {
        id: id.to_string(),
        quantifier,
        predicate,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }
}

#[test]
fn test_ef_infeasible_target_seeds_false() {
    let net = conserving_net();
    // EF(p0 >= 5): impossible since p0 + p1 = 3 and p1 >= 0 => p0 <= 3.
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(5),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(false),
        "EF of infeasible target should be FALSE"
    );
}

#[test]
fn test_ef_feasible_target_stays_unresolved() {
    let net = conserving_net();
    // EF(p1 >= 2): reachable by firing t0 twice.
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(2),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict, None,
        "LP-feasible EF should stay unresolved"
    );
}

#[test]
fn test_ag_conjunction_seeds_true() {
    let net = conserving_net();
    // AG(p0 <= 3 AND p1 <= 3): always true since p0 + p1 = 3.
    // Violating atoms: p0 >= 4 and p1 >= 4 — both LP-infeasible.
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::And(vec![
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
                ResolvedIntExpr::Constant(3),
            ),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(3),
            ),
        ]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "AG of invariant conjunction should be TRUE"
    );
}

#[test]
fn test_ag_violable_stays_unresolved() {
    let net = conserving_net();
    // AG(p0 <= 1): false since initial marking has p0 = 3.
    // But LP can't prove this false — it's the BFS/witness side.
    // The violating atom p0 >= 2 is LP-feasible, so AG stays unresolved.
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::Constant(1),
        ),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict, None,
        "AG with LP-feasible violation should stay unresolved"
    );
}

#[test]
fn test_feasible_fireability_ef_stays_unresolved() {
    let net = conserving_net();
    // EF(IsFireable(t0)): fireability is possible, but LP seeding only proves
    // false/true when decisive. It must not guess an EF=true witness.
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict, None,
        "LP-feasible fireability should stay unresolved"
    );
}

#[test]
fn test_ef_fireability_never_enabled_seeds_false() {
    let net = mutually_exclusive_input_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IsFireable(vec![TransitionIdx(2)]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(false),
        "EF of an LP-impossible fireability atom should seed FALSE"
    );
}

#[test]
fn test_ag_fireability_never_enabled_seeds_false() {
    let net = mutually_exclusive_input_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IsFireable(vec![TransitionIdx(2)]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(false),
        "AG of an always-false fireability atom should seed FALSE"
    );
}

/// 1-safe shuttle p0(1) <-> p1(0), plus self-loop probes ta on p0 and tb on
/// p1. ta and tb are never enabled simultaneously because p0 + p1 = 1.
fn mutex_fireability_conjunction_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t01", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t10", vec![arc(1, 1)], vec![arc(0, 1)]),
            trans("ta", vec![arc(0, 1)], vec![arc(0, 1)]),
            trans("tb", vec![arc(1, 1)], vec![arc(1, 1)]),
        ],
        initial_marking: vec![1, 0],
    }
}

#[test]
fn test_ef_conjunction_of_mutex_fireability_seeds_false() {
    // EF(AND(IsFireable(ta), IsFireable(tb))): never jointly enabled (p0+p1=1).
    // The joint state-equation relaxation proves the conjunction unreachable,
    // which the per-atom path could not. This is the fireability analogue of the
    // Cardinality conjunction LP seeding.
    let net = mutex_fireability_conjunction_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::And(vec![
            ResolvedPredicate::IsFireable(vec![TransitionIdx(2)]),
            ResolvedPredicate::IsFireable(vec![TransitionIdx(3)]),
        ]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(false),
        "EF of a mutually-exclusive fireability conjunction should seed FALSE"
    );
}

#[test]
fn test_ag_conjunction_of_mutex_fireability_seeds_false() {
    // AG(AND(IsFireable(ta), IsFireable(tb))): the conjunction never holds, so
    // it does not hold at the initial marking either → AG is FALSE.
    let net = mutex_fireability_conjunction_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::And(vec![
            ResolvedPredicate::IsFireable(vec![TransitionIdx(2)]),
            ResolvedPredicate::IsFireable(vec![TransitionIdx(3)]),
        ]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(false),
        "AG of an unreachable fireability conjunction should seed FALSE"
    );
}

#[test]
fn test_satisfiable_fireability_conjunction_stays_unresolved() {
    // AND(IsFireable(ta), p1 <= 1): reachable (ta enabled at M0, p1<=1 always),
    // so LP must not seed a verdict.
    let net = mutex_fireability_conjunction_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::And(vec![
            ResolvedPredicate::IsFireable(vec![TransitionIdx(2)]),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(1),
            ),
        ]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict, None,
        "a satisfiable fireability conjunction must stay unresolved"
    );
}

#[test]
fn test_ag_fireability_mutex_case_split_seeds_true() {
    let net = mutex_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IsFireable(vec![TransitionIdx(0), TransitionIdx(1)]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "LP should prove at least one mutex transition is always fireable"
    );
}

#[test]
fn test_ef_fireability_mutex_case_split_seeds_true() {
    let net = mutex_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IsFireable(vec![TransitionIdx(0), TransitionIdx(1)]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "EF of an always-true fireability atom should seed TRUE"
    );
}

#[test]
fn test_ag_fireability_feasible_disable_stays_unresolved() {
    let net = source_then_consume_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IsFireable(vec![TransitionIdx(1)]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict, None,
        "LP-feasible disabled marking must keep AG fireability unresolved"
    );
}

#[test]
fn test_ag_fireability_above_case_split_cap_stays_unresolved() {
    let input_count = MAX_FIREABILITY_CASE_SPLITS + 1;
    let net = high_fan_in_fireability_net(input_count);
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IsFireable(vec![TransitionIdx(input_count as u32)]),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict, None,
        "case splits above the cap should stay unresolved"
    );
}

#[test]
fn test_already_resolved_trackers_skipped() {
    let net = conserving_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(5),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ),
    )];
    // Pre-seed a verdict (as if BMC already resolved it).
    trackers[0].verdict = Some(true);

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "Pre-seeded verdict should not change"
    );
}

#[test]
fn test_mixed_trackers_partial_seeding() {
    let net = conserving_net();
    let mut trackers = vec![
        // EF(p0 >= 5): infeasible → FALSE
        tracker(
            "prop-00",
            PathQuantifier::EF,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(5),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ),
        ),
        // EF(p1 >= 2): feasible → stays None
        tracker(
            "prop-01",
            PathQuantifier::EF,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(2),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ),
        ),
        // EF(Or(...)): neither branch has decisive truth → stays None
        tracker(
            "prop-02",
            PathQuantifier::EF,
            ResolvedPredicate::Or(vec![
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(2),
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ),
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
            ]),
        ),
    ];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(trackers[0].verdict, Some(false));
    assert_eq!(trackers[1].verdict, None);
    assert_eq!(trackers[2].verdict, None);
}

#[test]
fn test_ag_with_token_rhs_seeds_true() {
    let net = conserving_net();
    // AG(p0 <= p0 + p1): always true because p1 >= 0 in every reachable marking.
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
        ),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "Token-vs-token AG invariant should be proven by strict LP violation"
    );
}

#[test]
fn test_ag_with_token_rhs_feasible_violation_stays_unresolved() {
    let net = conserving_net();
    // AG(p0 <= p1) is false at the initial marking, so LP must not seed TRUE.
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
    )];

    run_lp_seeding(&net, &mut trackers, None);
    assert_eq!(
        trackers[0].verdict, None,
        "LP-feasible token-vs-token violation must stay unresolved"
    );
}
