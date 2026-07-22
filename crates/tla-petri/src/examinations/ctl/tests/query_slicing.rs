// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::HashMap;

use super::super::check_ctl_properties;
use super::super::pipeline::with_experimental_ctl_shortcuts_for_test;
use super::support::{
    build_ctl_slice_candidate, check_ctl_properties_raw_oracle, check_ctl_properties_unsliced,
    component_a_ctl_props, disconnected_two_component_net, eg_a0_ge_1, livelock_disjoint_eg_net,
    make_ctl_prop,
};

use crate::examinations::query_support::{closure_on_reduced_net, ctl_support};
use crate::explorer::ExplorationConfig;
use crate::output::Verdict;
use crate::petri_net::{PlaceIdx, TransitionIdx};
use crate::property_xml::{CtlFormula, IntExpr, StatePredicate};
use crate::query_slice::build_query_slice;
use crate::reduction::ReducedNet;

#[test]
fn test_ctl_query_slicing_single_component() {
    // EF(tokens(a1) >= 1): only references component A.
    // Query slicing should explore only component A (2 states) not the full
    // 4-state product. The answer is TRUE (a0=1 fires ta to reach a1=1).
    let net = disconnected_two_component_net();
    let props = vec![make_ctl_prop(
        "slice-a",
        CtlFormula::EF(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec![String::from("a1")]),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ctl_query_slicing_both_components() {
    // AG(tokens(a0) + tokens(a1) + tokens(b0) + tokens(b1) <= 2):
    // references both components. No slicing possible. TRUE because each
    // component is conserving (1 token each, total always 2).
    let net = disconnected_two_component_net();
    let props = vec![make_ctl_prop(
        "no-slice",
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec![
                String::from("a0"),
                String::from("a1"),
                String::from("b0"),
                String::from("b1"),
            ]),
            IntExpr::Constant(2),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ctl_query_slicing_fireability() {
    let net = disconnected_two_component_net();
    let props = vec![make_ctl_prop(
        "slice-fire",
        CtlFormula::EF(Box::new(CtlFormula::Atom(StatePredicate::IsFireable(
            vec![String::from("ta")],
        )))),
    )];
    let (reduced, slice) = build_ctl_slice_candidate(&net, &props);
    let explore_net = slice.as_ref().map_or(&reduced.net, |slice| &slice.net);
    assert!(explore_net.num_places() < net.num_places());
    assert!(explore_net.num_transitions() < net.num_transitions());

    let config = ExplorationConfig::new(3);
    let results = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties(&net, &props, &config)
    });
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ctl_query_slicing_shrinks_disconnected_component() {
    let net = disconnected_two_component_net();
    let props = component_a_ctl_props();
    let (reduced, slice) = build_ctl_slice_candidate(&net, &props);
    let explore_net = slice.as_ref().map_or(&reduced.net, |slice| &slice.net);

    assert!(explore_net.num_places() < net.num_places());
    assert!(explore_net.num_transitions() < net.num_transitions());
}

#[test]
fn test_ctl_query_slicing_matches_unsliced_results() {
    let net = disconnected_two_component_net();
    let props = component_a_ctl_props();
    let config = ExplorationConfig::default();

    let sliced = check_ctl_properties(&net, &props, &config);
    let unsliced = check_ctl_properties_unsliced(&net, &props, &config);

    assert_eq!(sliced, unsliced);
}

/// Regression: the suffix `EG`/`AF` slice must not drop a disjoint livelock.
///
/// On `livelock_disjoint_eg_net`, `EG(tokens(a0) >= 1)` is TRUE on the full net
/// (fire the `tb` self-loop forever, keeping `a0` frozen at 1). The query slice
/// for the `a0` support drops the disjoint livelock `tb`, leaving an A-only
/// subnet whose only maximal path leaves the predicate region — which the
/// maximal-path suffix analysis would read as `EG = FALSE`.
///
/// The fail-closed gate detects that the slice dropped a transition and routes
/// to the full net, recovering the correct TRUE. This test would FAIL (assert
/// TRUE but get FALSE) against the pre-fix code, and the unsliced oracle proves
/// TRUE is the sound answer.
#[test]
fn test_ctl_suffix_slice_disjoint_livelock_not_dropped() {
    let net = livelock_disjoint_eg_net();
    let props = vec![make_ctl_prop("eg-livelock", eg_a0_ge_1())];

    // Mirror the production suffix-slice construction (identity reduced net +
    // closure on the raw net) and confirm it genuinely drops the disjoint
    // livelock transition `tb` — i.e. this net exercises the unsound path.
    let identity = ReducedNet::identity(&net);
    let place_map: HashMap<&str, PlaceIdx> = net
        .places
        .iter()
        .enumerate()
        .map(|(i, place)| (place.id.as_str(), PlaceIdx(i as u32)))
        .collect();
    let trans_map: HashMap<&str, TransitionIdx> = net
        .transitions
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.as_str(), TransitionIdx(i as u32)))
        .collect();
    let support = ctl_support(&identity, &props, &place_map, &trans_map).expect("support resolves");
    let closure = closure_on_reduced_net(&net, support);
    let suffix_slice = build_query_slice(&net, &closure).expect("slice is a strict subset");
    assert!(
        suffix_slice.net.num_transitions() < net.num_transitions(),
        "expected the suffix slice to drop the disjoint livelock transition tb"
    );

    // Sanity: the deep-slice candidate path is unaffected by this test.
    let _ = build_ctl_slice_candidate(&net, &props);

    // Ground-truth oracle on the full raw net (no reduction, no slicing):
    // EG(a0>=1) is TRUE because `tb` self-loops forever with a0 frozen at 1.
    let config = ExplorationConfig::default();
    let oracle = check_ctl_properties_raw_oracle(&net, &props, &config);
    assert_eq!(
        oracle[0].1,
        Verdict::True,
        "raw oracle: EG(a0>=1) is TRUE via the tb livelock"
    );

    // With shortcuts enabled the suffix-slice path is active. The slice on the
    // raw net drops `tb`; the OLD code evaluated EG on that A-only slice and
    // returned the unsound FALSE. The fail-closed gate now detects the dropped
    // transition and explores the full net, yielding the sound TRUE.
    let sliced = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties(&net, &props, &config)
    });
    assert_eq!(
        sliced[0].1,
        Verdict::True,
        "fail-closed suffix gate must route to the full net and report TRUE, \
         never the unsound sliced FALSE"
    );
}

/// Deep-shape + pure-sink-place differential: the sliced production path must
/// agree with the full-net path for every deep CTL shape on a net where the
/// query cone drops a pure-sink place. Pins BOTH the fair-cycle EGF fix (EGF is
/// excluded from slicing — commit f4c04867) AND the soundness of the still-sliced
/// non-fair-cycle deep shapes (nested EG/AF, AU, EU): dropping an unobserved sink
/// place is a bisimulation quotient over the observed atoms, so the verdict is
/// preserved. This is the one deep-shape × place-drop cell the slice battery did
/// not previously cover.
#[test]
fn test_deep_ctl_slice_with_sink_place_matches_full_net() {
    use crate::petri_net::{Arc, PetriNet, PlaceInfo, TransitionInfo};

    // p0 --t0--> p1 (pure sink). init p0=1. Deadlock [0,1]. Formulas over p0
    // have a cone {p0,t0} that DROPS the sink p1.
    let net = PetriNet {
        name: Some("sink-drop".to_string()),
        places: vec![
            PlaceInfo {
                id: "p0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p1".to_string(),
                name: None,
            },
        ],
        transitions: vec![TransitionInfo {
            id: "t0".to_string(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: vec![Arc {
                place: PlaceIdx(1),
                weight: 1,
            }],
        }],
        initial_marking: vec![1, 0],
    };
    let p0_le0 = StatePredicate::IntLe(
        IntExpr::TokensCount(vec!["p0".to_string()]),
        IntExpr::Constant(0),
    );
    let p0_ge1 = StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::TokensCount(vec!["p0".to_string()]),
    );
    let deep: Vec<(&str, CtlFormula)> = vec![
        // fair-cycle (excluded from slicing by the fix)
        (
            "egf",
            CtlFormula::EGF(Box::new(CtlFormula::Atom(p0_le0.clone()))),
        ),
        // non-fair-cycle deep shapes (still sliced; sound by the quotient)
        (
            "egef",
            CtlFormula::EG(Box::new(CtlFormula::EF(Box::new(CtlFormula::Atom(
                p0_ge1.clone(),
            ))))),
        ),
        (
            "afag",
            CtlFormula::AF(Box::new(CtlFormula::AG(Box::new(CtlFormula::Atom(
                p0_le0.clone(),
            ))))),
        ),
        (
            "eu",
            CtlFormula::EU(
                Box::new(CtlFormula::Atom(p0_ge1.clone())),
                Box::new(CtlFormula::Atom(p0_le0.clone())),
            ),
        ),
        (
            "au",
            CtlFormula::AU(
                Box::new(CtlFormula::Atom(p0_ge1.clone())),
                Box::new(CtlFormula::Atom(p0_le0.clone())),
            ),
        ),
    ];
    let config = ExplorationConfig::default();
    for (id, ctl) in deep {
        let props = vec![make_ctl_prop(id, ctl)];
        let sliced = with_experimental_ctl_shortcuts_for_test(true, || {
            check_ctl_properties(&net, &props, &config)
        });
        let full = check_ctl_properties_unsliced(&net, &props, &config);
        assert_eq!(
            sliced[0].1, full[0].1,
            "deep shape {id}: sliced production path must equal the full-net verdict \
             (sliced={:?} full={:?})",
            sliced[0].1, full[0].1
        );
    }
}
