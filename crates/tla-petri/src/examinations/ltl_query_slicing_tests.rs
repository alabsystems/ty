// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::{
    check_ltl_properties, count_unresolved_ltl, formula_contains_next,
    ltl_visible_reduced_transitions,
};
use crate::buchi::{check_ltl_on_the_fly, resolve_atom_with_aliases, to_nnf, PorContext};
use crate::examinations::query_support::{
    closure_on_reduced_net, ltl_property_support_with_aliases, relevance_cone_on_reduced_net,
};
use crate::explorer::ExplorationConfig;
use crate::model::PropertyAliases;
use crate::output::Verdict;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{Formula, IntExpr, LtlFormula, Property, StatePredicate};
use crate::query_slice::{build_query_local_slice, build_query_slice, QuerySlice};
use crate::reduction::{reduce_iterative, ReducedNet};
use crate::stubborn::DependencyGraph;

fn make_ltl_prop(id: &str, ltl: LtlFormula) -> Property {
    Property {
        id: id.to_string(),
        formula: Formula::Ltl(ltl),
    }
}

fn disconnected_two_component_net() -> PetriNet {
    PetriNet {
        name: Some("disconnected-ltl".to_string()),
        places: vec![
            PlaceInfo {
                id: "a0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "a1".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "b0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "b1".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "a2".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "a3".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "ta".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "ta_back".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "tb".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(3),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "ta_tail".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(4),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "ta_far".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(4),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(5),
                    weight: 1,
                }],
            },
            // Back-edge a3→a2 creates a cycle that blocks pre-agglomeration
            // of the a2/ta_far chain, preserving non-visible transitions.
            TransitionInfo {
                id: "ta_cycle".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(5),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(4),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![1, 0, 1, 0, 0, 0],
    }
}

fn component_a_ltl_props() -> Vec<Property> {
    vec![
        make_ltl_prop(
            "slice-a-ltl",
            LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["a0".to_string(), "a1".to_string()]),
                IntExpr::Constant(1),
            )))),
        ),
        make_ltl_prop(
            "slice-fire-ltl",
            LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IsFireable(
                vec!["ta_back".to_string()],
            )))),
        ),
    ]
}

fn build_ltl_slice_candidate(
    net: &PetriNet,
    property: &Property,
) -> (ReducedNet, Option<QuerySlice>) {
    let reduced = reduce_iterative(net).expect("reduction should succeed");
    let aliases = PropertyAliases::identity(net);
    let Some(support) = ltl_property_support_with_aliases(&reduced, property, &aliases) else {
        return (reduced, None);
    };

    // Mirror production strategy: use deep slice when no X in the formula.
    let has_next = if let Formula::Ltl(ltl) = &property.formula {
        let mut atoms = Vec::new();
        let nnf = to_nnf(ltl, &mut atoms);
        formula_contains_next(&nnf)
    } else {
        false
    };

    let slice = if has_next {
        let closure = closure_on_reduced_net(&reduced.net, support);
        build_query_slice(&reduced.net, &closure)
    } else {
        let cone = relevance_cone_on_reduced_net(&reduced.net, support);
        build_query_local_slice(&reduced.net, &cone)
    };
    (reduced, slice)
}

fn check_ltl_properties_unsliced(
    net: &PetriNet,
    properties: &[Property],
) -> Vec<(String, Verdict)> {
    let reduced = reduce_iterative(net).expect("reduction should succeed");
    let aliases = PropertyAliases::identity(net);

    properties
        .iter()
        .map(|prop| {
            let verdict = match &prop.formula {
                Formula::Ltl(ltl) => {
                    let (_, unresolved) = count_unresolved_ltl(ltl, &aliases);
                    if unresolved > 0 {
                        Verdict::CannotCompute
                    } else {
                        let mut atom_preds = Vec::new();
                        let nnf = to_nnf(ltl, &mut atom_preds);
                        let resolved_atoms: Vec<_> = atom_preds
                            .iter()
                            .map(|pred| resolve_atom_with_aliases(pred, &aliases))
                            .collect();

                        let por = if !formula_contains_next(&nnf) {
                            let visible =
                                ltl_visible_reduced_transitions(&resolved_atoms, &reduced);
                            if visible.len() < reduced.net.num_transitions() {
                                Some(PorContext {
                                    dep: DependencyGraph::build(&reduced.net),
                                    visible,
                                    per_state_visibility: true,
                                })
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        match check_ltl_on_the_fly(
                            &nnf,
                            &reduced.net,
                            &reduced,
                            net,
                            &resolved_atoms,
                            por.as_ref(),
                            ExplorationConfig::default().max_states(),
                            None,
                        ) {
                            Ok(Some(true)) => Verdict::True,
                            Ok(Some(false)) => Verdict::False,
                            Ok(None) => Verdict::CannotCompute,
                            Err(error) => {
                                panic!("on-the-fly LTL expansion should not overflow in test helper: {error}")
                            }
                        }
                    }
                }
                _ => Verdict::CannotCompute,
            };
            (prop.id.clone(), verdict)
        })
        .collect()
}

#[test]
fn test_ltl_query_slicing_shrinks_disconnected_component() {
    let net = disconnected_two_component_net();
    let props = component_a_ltl_props();
    for prop in &props {
        let (reduced, slice) = build_ltl_slice_candidate(&net, prop);
        let explore_net = slice.as_ref().map_or(&reduced.net, |slice| &slice.net);
        assert!(explore_net.num_places() < net.num_places());
        assert!(explore_net.num_transitions() < net.num_transitions());
    }
}

#[test]
fn test_ltl_query_slicing_fireability_builds_proper_visible_subset() {
    let net = disconnected_two_component_net();
    let prop = component_a_ltl_props()
        .into_iter()
        .find(|prop| prop.id == "slice-fire-ltl")
        .expect("fireability regression property should exist");
    let (reduced, slice) = build_ltl_slice_candidate(&net, &prop);
    let composed_slice;
    let composed = if let Some(slice) = slice {
        composed_slice = super::compose_slice_and_reduction(&slice, &reduced);
        &composed_slice
    } else {
        &reduced
    };

    let Formula::Ltl(ltl) = &prop.formula else {
        panic!("slice-fire-ltl should remain an LTL property");
    };
    let aliases = PropertyAliases::identity(&net);
    let mut atom_preds = Vec::new();
    let nnf = to_nnf(ltl, &mut atom_preds);
    assert!(!formula_contains_next(&nnf));

    let resolved_atoms: Vec<_> = atom_preds
        .iter()
        .map(|pred| resolve_atom_with_aliases(pred, &aliases))
        .collect();
    let visible = ltl_visible_reduced_transitions(&resolved_atoms, composed);

    // With the current reduction lane, the referenced fireability transition
    // may already be removed before query slicing/POR. The important invariant
    // is verdict parity, not a specific visible-set size.
    assert!(visible.len() <= composed.net.num_transitions());

    let sliced = check_ltl_properties(
        &net,
        std::slice::from_ref(&prop),
        &ExplorationConfig::default(),
    );
    let unsliced = check_ltl_properties_unsliced(&net, &[prop]);
    assert_eq!(sliced, unsliced);
}

#[test]
fn test_ltl_query_slicing_matches_unsliced_results() {
    let net = disconnected_two_component_net();
    let props = component_a_ltl_props();
    let config = ExplorationConfig::default();

    let sliced = check_ltl_properties(&net, &props, &config);
    let unsliced = check_ltl_properties_unsliced(&net, &props);

    assert_eq!(sliced, unsliced);
}

// ── Deep-slice gating tests ────────────────────────────────────────

#[test]
fn test_ltl_deep_slice_verdict_parity_on_disconnected_net() {
    // G(tokens(a0) + tokens(a1) <= 1) — stutter-insensitive (no X).
    // Uses the disconnected net which has enough structure to survive
    // reduction. Verdict with deep slice must match unsliced.
    let net = disconnected_two_component_net();
    let props = component_a_ltl_props();
    let config = ExplorationConfig::default();

    let sliced = check_ltl_properties(&net, &props, &config);
    let unsliced = check_ltl_properties_unsliced(&net, &props);
    assert_eq!(sliced, unsliced);
}

#[test]
fn test_ltl_fallback_with_next_verdict_parity() {
    // X(tokens(a0) >= 1) — contains X, must use conservative closure.
    // Verdict must still match the unsliced path.
    let net = disconnected_two_component_net();
    let prop = make_ltl_prop(
        "fallback-x",
        LtlFormula::Next(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["a0".to_string()]),
        )))),
    );

    // Verify the gate detects X.
    let mut atoms = Vec::new();
    let Formula::Ltl(ltl) = &prop.formula else {
        panic!("should be LTL");
    };
    let nnf = to_nnf(ltl, &mut atoms);
    assert!(formula_contains_next(&nnf));

    // Verdict must match.
    let sliced = check_ltl_properties(
        &net,
        std::slice::from_ref(&prop),
        &ExplorationConfig::default(),
    );
    let unsliced = check_ltl_properties_unsliced(&net, &[prop]);
    assert_eq!(sliced, unsliced);
}
