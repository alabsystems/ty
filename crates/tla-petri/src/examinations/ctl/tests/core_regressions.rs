// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::path::PathBuf;

use super::super::checker::CtlChecker;
use super::super::pipeline::reduce_ctl_batch_for_test;
use super::super::{check_ctl_properties, known_mcc_ctl_soundness_guards};
use super::support::{make_ctl_prop, reduce_with_ctl_protection, simple_net};

use crate::examinations::query_support::{
    closure_on_reduced_net, ctl_support_with_aliases, relevance_cone_on_reduced_net,
};
use crate::explorer::{explore_full, ExplorationConfig};
use crate::model::PropertyAliases;
use crate::output::Verdict;
use crate::parser::parse_pnml_dir;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::property_xml::{self, CtlFormula, Formula, IntExpr, Property, StatePredicate};
use crate::query_slice::{build_query_local_slice, build_query_slice};
use crate::reduction::{
    reduce_iterative, reduce_iterative_temporal_projection_candidate, ReducedNet,
};

const IBM5964_CTL_CARDINALITY_11_ID: &str = "IBM5964-PT-none-CTLCardinality-2024-11";

fn ctl_temporal_projection_net() -> PetriNet {
    PetriNet {
        name: Some("ctl-temporal-projection".into()),
        places: vec![
            PlaceInfo {
                id: "p_live".into(),
                name: None,
            },
            PlaceInfo {
                id: "p_sink".into(),
                name: None,
            },
            PlaceInfo {
                id: "p_const".into(),
                name: None,
            },
            PlaceInfo {
                id: "p_iso".into(),
                name: None,
            },
            PlaceInfo {
                id: "p_dead_src".into(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "t_live".into(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    },
                ],
                outputs: vec![
                    Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    },
                ],
            },
            TransitionInfo {
                id: "t_dead".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(4),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![1, 0, 3, 5, 0],
    }
}

fn check_ctl_property_on_reduced_net(
    net: &crate::petri_net::PetriNet,
    property: &Property,
    reduced: &ReducedNet,
    config: &ExplorationConfig,
) -> Verdict {
    let Formula::Ctl(ctl) = &property.formula else {
        return Verdict::CannotCompute;
    };

    let aliases = PropertyAliases::identity(net);
    let (_, unresolved) = super::super::pipeline::count_unresolved_ctl_with_aliases(ctl, &aliases);
    if unresolved > 0 {
        return Verdict::CannotCompute;
    }

    let properties = std::slice::from_ref(property);
    let Some(support) = ctl_support_with_aliases(reduced, properties, &aliases) else {
        return Verdict::CannotCompute;
    };

    let use_deep_slice = !super::super::ctl_batch_contains_next_step(properties);
    let slice = if use_deep_slice {
        let cone = relevance_cone_on_reduced_net(&reduced.net, support);
        build_query_local_slice(&reduced.net, &cone)
    } else {
        let closure = closure_on_reduced_net(&reduced.net, support);
        build_query_slice(&reduced.net, &closure)
    };

    let explore_net = slice.as_ref().map_or(&reduced.net, |slice| &slice.net);
    let config = config.refitted_for_net(explore_net);
    let mut full = explore_full(explore_net, &config);
    if !full.graph.completed {
        return Verdict::CannotCompute;
    }

    let expand_result = full.markings.expand_in_place(|marking| {
        if let Some(ref slice) = slice {
            let mut reduced_marking = vec![0u64; reduced.net.num_places()];
            for (sliced_idx, &tokens) in marking.iter().enumerate() {
                reduced_marking[slice.place_unmap[sliced_idx].0 as usize] = tokens;
            }
            reduced.expand_marking(&reduced_marking)
        } else {
            reduced.expand_marking(marking)
        }
    });
    if expand_result.is_err() {
        return Verdict::CannotCompute;
    }

    let checker = CtlChecker::new(&full, net);
    let resolved = super::super::resolve::resolve_ctl_with_aliases(ctl, &aliases);
    if checker.eval(&resolved)[0] {
        Verdict::True
    } else {
        Verdict::False
    }
}

#[test]
fn test_ctl_ag_atom() {
    let net = simple_net();
    let props = vec![make_ctl_prop(
        "ag-00",
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec![String::from("p0"), String::from("p1")]),
            IntExpr::Constant(2),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ctl_known_mcc_soundness_guards_are_retired() {
    let guarded_ids = known_mcc_ctl_soundness_guards();
    assert!(
        guarded_ids.is_empty(),
        "the deadlock/maximal-path fix should retire the historical MCC CTL soundness guards"
    );

    let net = simple_net();
    let config = ExplorationConfig::default();
    let historical_ids = [
        "AirplaneLD-PT-0010-CTLCardinality-2024-07",
        "AirplaneLD-PT-0010-CTLFireability-2024-09",
        "AirplaneLD-PT-0010-CTLFireability-2024-13",
        "Angiogenesis-PT-01-CTLFireability-2023-14",
        "CSRepetitions-PT-02-CTLFireability-2024-01",
    ];
    let props: Vec<_> = historical_ids
        .iter()
        .map(|id| {
            make_ctl_prop(
                id,
                CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
                    IntExpr::TokensCount(vec![String::from("p0"), String::from("p1")]),
                    IntExpr::Constant(2),
                )))),
            )
        })
        .collect();

    let results = check_ctl_properties(&net, &props, &config);
    for (_, verdict) in results {
        assert_eq!(
            verdict,
            Verdict::True,
            "historical guarded IDs should now execute normally on the public CTL path"
        );
    }
}

#[test]
fn test_ctl_adjacent_property_id_still_executes() {
    let net = simple_net();
    let props = vec![make_ctl_prop(
        "AirplaneLD-PT-0010-CTLCardinality-2024-06",
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec![String::from("p0"), String::from("p1")]),
            IntExpr::Constant(2),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ctl_ef_atom() {
    let net = simple_net();
    let props = vec![make_ctl_prop(
        "ef-00",
        CtlFormula::EF(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(2),
            IntExpr::TokensCount(vec![String::from("p1")]),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ctl_ex() {
    let net = simple_net();
    let props = vec![make_ctl_prop(
        "ex-00",
        CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec![String::from("p1")]),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ctl_source_place_referenced_by_formula_is_protected() {
    let net = simple_net();
    let props = vec![make_ctl_prop(
        "protect-source",
        CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec![String::from("p1")]),
        )))),
    )];

    let unprotected = reduce_iterative(&net).expect("unprotected reduction should succeed");
    assert_eq!(
        unprotected.place_map[1], None,
        "baseline reduction should still remove the producer-only sink place"
    );

    let reduced = reduce_with_ctl_protection(&net, &props);
    assert!(
        reduced.place_map[1].is_some(),
        "CTL reduction must keep formula-referenced source places alive"
    );
}

#[test]
fn test_ctl_eu() {
    let net = simple_net();
    let props = vec![make_ctl_prop(
        "eu-00",
        CtlFormula::EU(
            Box::new(CtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::Constant(1),
                IntExpr::TokensCount(vec![String::from("p0")]),
            ))),
            Box::new(CtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::Constant(2),
                IntExpr::TokensCount(vec![String::from("p1")]),
            ))),
        ),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ctl_unresolved_place_names_return_cannot_compute() {
    let net = simple_net();
    // Formula references "NONEXISTENT_PLACE" which is not in the net.
    // Without the resolution guard, this silently resolves to TokensCount([])
    // which evaluates to 0, producing a potentially wrong answer.
    let props = vec![make_ctl_prop(
        "unresolved-place",
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec![String::from("NONEXISTENT_PLACE")]),
            IntExpr::Constant(0),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(
        results[0].1,
        Verdict::CannotCompute,
        "Unresolved place names must trigger CANNOT_COMPUTE"
    );
}

#[test]
fn test_ctl_unresolved_transition_names_return_cannot_compute() {
    let net = simple_net();
    // Formula references "NONEXISTENT_TRANS" which is not in the net.
    // Without the resolution guard, this silently resolves to IsFireable([])
    // which evaluates to False, producing a potentially wrong answer.
    let props = vec![make_ctl_prop(
        "unresolved-trans",
        CtlFormula::EF(Box::new(CtlFormula::Atom(StatePredicate::IsFireable(
            vec![String::from("NONEXISTENT_TRANS")],
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(
        results[0].1,
        Verdict::CannotCompute,
        "Unresolved transition names must trigger CANNOT_COMPUTE"
    );
}

#[test]
fn test_ctl_resolved_names_still_evaluate() {
    // Ensure the resolution guard doesn't block valid formulas.
    let net = simple_net();
    let props = vec![make_ctl_prop(
        "resolved-ok",
        CtlFormula::EF(Box::new(CtlFormula::Atom(StatePredicate::IsFireable(
            vec![String::from("t0")],
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    // t0 is fireable in the initial state (p0 has 2 tokens), so EF(fireable(t0)) = true
    assert_eq!(
        results[0].1,
        Verdict::True,
        "Fully resolved formula must evaluate normally"
    );
}

#[test]
fn test_ctl_af_deadlock() {
    let net = simple_net();
    let props = vec![make_ctl_prop(
        "af-00",
        CtlFormula::AF(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec![String::from("p0")]),
            IntExpr::Constant(0),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_next_free_ctl_default_reduction_uses_next_free_mode() {
    // A next-free CTL batch (no EX/AX) reduces under `ReductionMode::NextFreeCTL`
    // — the declared safe mode for the CTL examinations. NextFreeCTL is strictly
    // stronger than the old marking-preserving `CTLWithNext` subset (it admits
    // source/dominated/self-loop/redundant-place rules) so it removes at least
    // as much, but it CRITICALLY still excludes the CTL-unsound agglomeration
    // rules (the IBM5964-CTLCardinality-11 wrong-verdict guard, below).
    let net = ctl_temporal_projection_net();
    let props = vec![make_ctl_prop(
        "next-free-safe-reduction",
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::True))),
    )];

    let reduced = reduce_ctl_batch_for_test(&net, &props);

    // Non-trivial reduction, never larger than the original.
    assert!(reduced.report.has_reductions());
    assert!(reduced.net.num_transitions() <= net.num_transitions());
    assert!(reduced.net.num_places() <= net.num_places());
    // At least as strong as the marking-preserving subset: the dead transition,
    // the constant place, and both isolated places are stripped.
    assert!(reduced.report.dead_transitions.contains(&TransitionIdx(1)));
    assert!(reduced.report.constant_places.contains(&PlaceIdx(2)));
    assert!(reduced.report.isolated_places.contains(&PlaceIdx(3)));
    assert!(reduced.report.isolated_places.contains(&PlaceIdx(4)));
    // SOUNDNESS GUARD: NextFreeCTL must never agglomerate — agglomeration is not
    // CTL-preserving (it merges transitions, changing path structure) and was
    // the root cause of the IBM5964 wrong FALSE verdict.
    assert!(reduced.report.pre_agglomerations.is_empty());
    assert!(reduced.report.post_agglomerations.is_empty());
    assert!(reduced.report.rule_r_agglomerations.is_empty());
    assert!(reduced.report.rule_s_agglomerations.is_empty());

    // Verdict-preservation: the sound reduction yields the same answer the
    // identity net would.
    let results = check_ctl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results,
        vec![("next-free-safe-reduction".to_string(), Verdict::True)]
    );
}

#[test]
fn test_ctl_next_step_batch_uses_marking_preserving_reduction() {
    // An EX/AX (next-step) batch reduces under `ReductionMode::CTLWithNext`,
    // which applies ONLY the marking-preserving rules (dead-transition,
    // constant-place, isolated-place removal) so the exact successor structure
    // that EX/AX observes is preserved. Previously this returned the identity
    // net (zero reduction) whenever any EX/AX appeared anywhere in the batch.
    let net = ctl_temporal_projection_net();
    let props = vec![make_ctl_prop(
        "next-step-marking-preserving",
        CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::True))),
    )];

    let reduced = reduce_ctl_batch_for_test(&net, &props);

    // Marking-preserving subset removes the dead transition, the constant place,
    // and both isolated places — leaving only `t_live` over {p_live, p_sink}.
    assert!(reduced.report.has_reductions());
    assert_eq!(reduced.net.num_transitions(), 1);
    assert_eq!(reduced.net.num_places(), 2);
    assert_eq!(reduced.report.dead_transitions, vec![TransitionIdx(1)]);
    assert!(reduced.report.constant_places.contains(&PlaceIdx(2)));
    assert!(reduced.report.isolated_places.contains(&PlaceIdx(3)));
    assert!(reduced.report.isolated_places.contains(&PlaceIdx(4)));
    // CTLWithNext applies NONE of the interleaving-changing or non-marking-
    // preserving rules — only dead/constant/isolated. In particular no
    // agglomeration (the EX/AX next-step soundness guard).
    assert!(reduced.report.source_places.is_empty());
    assert!(reduced.report.pre_agglomerations.is_empty());
    assert!(reduced.report.post_agglomerations.is_empty());
    assert!(reduced.report.rule_r_agglomerations.is_empty());
    assert!(reduced.report.rule_s_agglomerations.is_empty());
    assert!(reduced.report.duplicate_transitions.is_empty());
    assert!(reduced.report.dominated_transitions.is_empty());
    assert!(reduced.report.self_loop_arcs.is_empty());
    assert!(reduced.report.never_disabling_arcs.is_empty());

    // Verdict-preservation: EX(True) is TRUE (the initial state has a successor).
    let results = check_ctl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results,
        vec![("next-step-marking-preserving".to_string(), Verdict::True)]
    );
}

/// Regression: IBM5964 CTLCardinality-11 was wrong (FALSE instead of TRUE)
/// because structural reductions (agglomeration) are not CTL-preserving.
/// The fix: skip structural reductions for CTL Phase 1 evaluation.
#[test]
fn test_ibm5964_ctl_cardinality_11_regression() {
    let ws = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let model_dir = ws.join("benchmarks/mcc/2024/INPUTS/IBM5964-PT-none");
    if !model_dir.join("model.pnml").exists() {
        eprintln!("SKIP: IBM5964-PT-none benchmark not downloaded");
        return;
    }

    let net = parse_pnml_dir(&model_dir).expect("parse PNML");
    let properties =
        property_xml::parse_properties(&model_dir, "CTLCardinality").expect("parse XML");

    // Formula 11 — ground truth: TRUE (from c-t-l-cardinality-2024.csv)
    let prop = properties
        .iter()
        .find(|property| property.id == IBM5964_CTL_CARDINALITY_11_ID)
        .expect("IBM5964 CTLCardinality property 11 should exist");

    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, std::slice::from_ref(prop), &config);
    assert_eq!(
        results[0].1,
        Verdict::True,
        "IBM5964 CTLCardinality-11 should be TRUE (was wrong FALSE before #1421 fix)"
    );
}

#[test]
fn test_ibm5964_ctl_cardinality_11_temporal_projection_candidate_matches_identity() {
    let ws = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let model_dir = ws.join("benchmarks/mcc/2024/INPUTS/IBM5964-PT-none");
    if !model_dir.join("model.pnml").exists() {
        eprintln!("SKIP: IBM5964-PT-none benchmark not downloaded");
        return;
    }

    let net = parse_pnml_dir(&model_dir).expect("parse PNML");
    let properties =
        property_xml::parse_properties(&model_dir, "CTLCardinality").expect("parse XML");

    let prop = properties
        .iter()
        .find(|property| property.id == IBM5964_CTL_CARDINALITY_11_ID)
        .expect("IBM5964 CTLCardinality property 11 should exist");

    let config = ExplorationConfig::default();
    let production = check_ctl_properties(&net, std::slice::from_ref(prop), &config);
    assert_eq!(
        production[0].1,
        Verdict::True,
        "identity CTL path must stay blessed for IBM5964 CTLCardinality-11"
    );

    let candidate = reduce_iterative_temporal_projection_candidate(&net)
        .expect("temporal projection candidate reduction should succeed");
    let candidate_verdict = check_ctl_property_on_reduced_net(&net, prop, &candidate, &config);

    assert_eq!(
        candidate_verdict,
        production[0].1,
        "the quarantined dead/constant/isolated candidate should match the blessed IBM5964 identity verdict"
    );
    assert_eq!(
        candidate_verdict,
        Verdict::True,
        "IBM5964 should stay TRUE on the narrowed temporal candidate regression path"
    );
}

/// Regression: AirplaneLD-COL-0010-LTLFireability-08 answered FALSE against a
/// TRUE 4-tool consensus (the single corpus-wide consensus disagreement).
///
/// Root cause: `reduce_ctl_batch_for_mode` built its protected set from
/// `support.places` ONLY. An `is-fireable`-only batch (e.g. an LTLFireability
/// `A¬G(φ)` routed through the `AF(¬φ)` CTL lane) has an EMPTY place support —
/// its atoms live in `support.transitions` — so the NextFreeCTL structural
/// reduction ran completely unprotected and gutted the whole net (Airplane:
/// all 89 places + 88 transitions). The surviving single stuttering state
/// expanded back to the initial marking, where φ (some instance fireable)
/// holds, so `AF(¬φ)` evaluated FALSE.
///
/// The fix protects the pre/post places of every supported transition
/// (`protected_places_for_prefire`, the reachability pipeline's existing
/// contract), which keeps every fireability atom exact in the reduced net.
#[test]
fn test_ctl_fireability_only_batch_protects_prefire_places() {
    // Minimal Airplane shape: one token, one sink transition.
    //   src(1) --t_drain--> (no outputs)
    // φ = is-fireable(t_drain). AF(¬φ) is TRUE: the only maximal run fires
    // t_drain into the deadlock (which MCC-stutters), where φ is false.
    let net = PetriNet {
        name: Some("fireability-prefire-protection".into()),
        places: vec![PlaceInfo {
            id: "src".into(),
            name: None,
        }],
        transitions: vec![TransitionInfo {
            id: "t_drain".into(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: vec![],
        }],
        initial_marking: vec![1],
    };
    let props = vec![make_ctl_prop(
        "af-not-fireable",
        CtlFormula::AF(Box::new(CtlFormula::Not(Box::new(CtlFormula::Atom(
            StatePredicate::IsFireable(vec!["t_drain".into()]),
        ))))),
    )];

    // Structural guard: the fireability atom's transition and its input place
    // must survive the batch reduction. Under the places-only protected set
    // the sink-transition rule removed t_drain and the constant-place rule
    // then removed src, gutting the net to a single stuttering state.
    let reduced = reduce_ctl_batch_for_test(&net, &props);
    assert!(
        reduced.transition_map[0].is_some(),
        "the is-fireable target transition must survive a fireability-only \
         CTL batch reduction (protected via its pre/post places)"
    );
    assert!(
        reduced.place_map[0].is_some(),
        "the input place of an is-fireable target must be protected from \
         structural reduction"
    );

    // End-to-end verdict: AF(¬fireable(t_drain)) is TRUE on the real graph;
    // the gutted net answered FALSE (initial marking stutters with φ true).
    let results = check_ctl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results,
        vec![("af-not-fireable".to_string(), Verdict::True)],
        "fireability-only AF batch must not be flipped by unprotected reduction"
    );
}
