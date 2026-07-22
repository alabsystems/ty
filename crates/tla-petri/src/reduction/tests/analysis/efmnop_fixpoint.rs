// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::reduction::model::NeverDisablingProof;
use crate::reduction::ReductionMode;

use super::super::super::analysis::{analyze_efmnop_fixpoint, compute_efmnop_fixpoint_state};
use super::super::support::{arc, place, trans};

#[test]
fn test_efmnop_fixpoint_cascade_dead_to_orphan_exposes_isolated_places() {
    let net = PetriNet {
        name: Some("cascade-dead-to-orphan".into()),
        places: vec![
            place("p0"),
            place("p1"),
            place("p2"),
            place("p_live_in"),
            place("p_live_out"),
        ],
        transitions: vec![
            trans("t0_dead", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1_dead", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t_live", vec![arc(3, 1)], vec![arc(4, 1)]),
        ],
        initial_marking: vec![0, 0, 0, 1, 0],
    };

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(
        analysis.report.dead_transitions,
        vec![TransitionIdx(0), TransitionIdx(1)]
    );
    assert_eq!(
        analysis.report.isolated_places,
        vec![PlaceIdx(0), PlaceIdx(1), PlaceIdx(2)]
    );
    assert_eq!(analysis.dead_removed_by_cascade, 1);
    assert_eq!(analysis.per_rule_progress.rule_o_dead, 2);
    assert_eq!(analysis.per_rule_progress.rule_o_orphan, 3);
    assert_eq!(analysis.iterations, 2);
}

#[test]
fn test_efmnop_fixpoint_single_pass_simple_net_stabilizes() {
    let net = PetriNet {
        name: Some("single-pass-duplicate".into()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t_back", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    };

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(analysis.iterations, 1);
    assert!(analysis.report.dead_transitions.is_empty());
    assert_eq!(analysis.report.duplicate_transitions.len(), 1);
    assert_eq!(
        analysis.report.duplicate_transitions[0].duplicates,
        vec![TransitionIdx(1)]
    );
    assert_eq!(analysis.per_rule_progress.rule_m_duplicate, 1);
}

#[test]
fn test_efmnop_fixpoint_multi_pass_dead_removal_enables_agglomeration() {
    let net = PetriNet {
        name: Some("dead-enables-agglomeration".into()),
        places: vec![
            place("p_block"),
            place("p_in"),
            place("p_mid"),
            place("p_out"),
        ],
        transitions: vec![
            trans("t_dead", vec![arc(0, 1)], vec![arc(2, 2)]),
            trans("t_src", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t_sink", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![0, 1, 0, 0],
    };

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(analysis.report.dead_transitions, vec![TransitionIdx(0)]);
    assert_eq!(analysis.report.pre_agglomerations.len(), 1);
    assert_eq!(
        analysis.report.pre_agglomerations[0].transition,
        TransitionIdx(1)
    );
    assert_eq!(analysis.report.pre_agglomerations[0].place, PlaceIdx(2));
    assert_eq!(analysis.per_rule_progress.rule_p_pre_agglomeration, 1);
    assert_eq!(analysis.iterations, 2);
}

#[test]
fn test_efmnop_fixpoint_mode_gating_ctl_with_next_restricts_rules() {
    let net = PetriNet {
        name: Some("ctl-with-next-gating".into()),
        places: vec![
            place("p_dead"),
            place("p_dead_out"),
            place("p0"),
            place("p1"),
        ],
        transitions: vec![
            trans("t_dead", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t0", vec![arc(2, 1)], vec![arc(3, 1)]),
            trans("t1", vec![arc(2, 1)], vec![arc(3, 1)]),
            trans("t_back", vec![arc(3, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![0, 0, 1, 0],
    };

    let reachability = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);
    let ctl_with_next = analyze_efmnop_fixpoint(&net, &[], ReductionMode::CTLWithNext);

    assert_eq!(reachability.report.duplicate_transitions.len(), 1);
    assert_eq!(
        ctl_with_next.report.dead_transitions,
        vec![TransitionIdx(0)]
    );
    assert_eq!(
        ctl_with_next.report.isolated_places,
        vec![PlaceIdx(0), PlaceIdx(1)]
    );
    assert!(ctl_with_next.report.duplicate_transitions.is_empty());
    assert!(ctl_with_next.report.pre_agglomerations.is_empty());
    assert_eq!(ctl_with_next.iterations, 2);
}

#[test]
fn test_efmnop_fixpoint_empty_net_returns_zero_iterations() {
    let net = PetriNet {
        name: Some("empty".into()),
        places: Vec::new(),
        transitions: Vec::new(),
        initial_marking: Vec::new(),
    };

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(analysis.iterations, 0);
    assert_eq!(analysis.dead_removed_by_cascade, 0);
    assert!(!analysis.report.has_reductions());
}

#[test]
fn test_efmnop_fixpoint_combined_m_and_o_rules_both_reported() {
    let net = PetriNet {
        name: Some("combined-m-o".into()),
        places: vec![
            place("p_dead"),
            place("p_dead_out"),
            place("p0"),
            place("p1"),
        ],
        transitions: vec![
            trans("t_dead", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t0", vec![arc(2, 1)], vec![arc(3, 1)]),
            trans("t1", vec![arc(2, 1)], vec![arc(3, 1)]),
            trans("t_back", vec![arc(3, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![0, 0, 1, 0],
    };

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(analysis.report.dead_transitions, vec![TransitionIdx(0)]);
    assert_eq!(analysis.report.duplicate_transitions.len(), 1);
    assert_eq!(analysis.per_rule_progress.rule_o_dead, 1);
    assert_eq!(analysis.per_rule_progress.rule_m_duplicate, 1);
    assert_eq!(
        analysis.report.duplicate_transitions[0].duplicates,
        vec![TransitionIdx(2)]
    );
}

#[test]
fn test_efmnop_fixpoint_preserves_zero_trap_dead_transition_detection() {
    let net = PetriNet {
        name: Some("zero-trap-cycle".into()),
        places: vec![
            place("p_zero_a"),
            place("p_zero_b"),
            place("p_live_in"),
            place("p_live_out"),
        ],
        transitions: vec![
            trans("t_zero_a_to_b", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t_zero_b_to_a", vec![arc(1, 1)], vec![arc(0, 1)]),
            trans("t_live", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![0, 0, 1, 0],
    };

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(
        analysis.report.dead_transitions,
        vec![TransitionIdx(0), TransitionIdx(1)]
    );
    assert_eq!(
        analysis.report.isolated_places,
        vec![PlaceIdx(0), PlaceIdx(1)]
    );
    assert_eq!(analysis.per_rule_progress.rule_o_dead, 2);
    assert_eq!(analysis.per_rule_progress.rule_o_orphan, 2);
    assert_eq!(analysis.iterations, 2);
}

#[test]
fn test_efmnop_workqueue_marks_transitive_can_inc_can_dec() {
    let net = PetriNet {
        name: Some("workqueue-can-inc-dec".into()),
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };

    let state = compute_efmnop_fixpoint_state(&net);

    assert!(state.fired(TransitionIdx(0)));
    assert!(state.fired(TransitionIdx(1)));
    assert!(state.can_dec(PlaceIdx(0)));
    assert!(state.can_inc(PlaceIdx(1)));
    assert!(state.can_dec(PlaceIdx(1)));
    assert!(state.can_inc(PlaceIdx(2)));
    assert_eq!(state.lower_bound(PlaceIdx(0)), 0);
}

#[test]
fn test_efmnop_workqueue_seeds_zero_marked_trap_cycle_dead_transitions() {
    let net = PetriNet {
        name: Some("workqueue-zero-trap-cycle".into()),
        places: vec![
            place("p_zero_a"),
            place("p_zero_b"),
            place("p_escape"),
            place("p_live"),
        ],
        transitions: vec![
            trans("t_zero_a_to_b", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t_zero_b_to_a", vec![arc(1, 1)], vec![arc(0, 1)]),
            trans("t_escape", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t_live", vec![arc(3, 1)], vec![arc(3, 2)]),
        ],
        initial_marking: vec![0, 0, 0, 1],
    };

    let state = compute_efmnop_fixpoint_state(&net);
    assert!(!state.fired(TransitionIdx(0)));
    assert!(!state.fired(TransitionIdx(1)));
    assert!(!state.fired(TransitionIdx(2)));
    assert!(state.fired(TransitionIdx(3)));

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(
        analysis.report.dead_transitions,
        vec![TransitionIdx(0), TransitionIdx(1), TransitionIdx(2)]
    );
    assert_eq!(analysis.per_rule_progress.rule_e_workqueue_dead, 0);
    assert_eq!(analysis.per_rule_progress.rule_o_dead, 3);
}

#[test]
fn test_efmnop_workqueue_sums_duplicate_input_arcs_and_splits_dead_metrics() {
    let net = PetriNet {
        name: Some("workqueue-duplicate-input-weights".into()),
        places: vec![
            place("p_token"),
            place("p_workqueue_out"),
            place("p_rule_o_dead"),
            place("p_rule_o_out"),
        ],
        transitions: vec![
            trans("t_conserve", vec![arc(0, 1)], vec![arc(0, 1)]),
            trans(
                "t_needs_two_duplicate_inputs",
                vec![arc(0, 1), arc(0, 1)],
                vec![arc(1, 1)],
            ),
            trans("t_rule_o_dead", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![1, 0, 0, 0],
    };

    let state = compute_efmnop_fixpoint_state(&net);
    assert!(state.fired(TransitionIdx(0)));
    assert!(!state.fired(TransitionIdx(1)));
    assert!(!state.fired(TransitionIdx(2)));

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(
        analysis.report.dead_transitions,
        vec![TransitionIdx(1), TransitionIdx(2)]
    );
    assert_eq!(analysis.per_rule_progress.rule_e_workqueue_dead, 1);
    assert_eq!(analysis.per_rule_progress.rule_o_dead, 1);
}

#[test]
fn test_efmnop_fixpoint_emits_lower_bound_never_disabling_proof() {
    let net = PetriNet {
        name: Some("fixpoint-lower-bound-rule-n".into()),
        places: vec![place("p_guard"), place("p_out")],
        transitions: vec![trans(
            "t_grow_guard",
            vec![arc(0, 1)],
            vec![arc(0, 2), arc(1, 1)],
        )],
        initial_marking: vec![2, 0],
    };

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::Reachability);

    assert_eq!(analysis.report.never_disabling_arcs.len(), 1);
    let arc = &analysis.report.never_disabling_arcs[0];
    assert_eq!(arc.transition, TransitionIdx(0));
    assert_eq!(arc.place, PlaceIdx(0));
    assert_eq!(arc.weight, 1);
    assert_eq!(
        arc.proof,
        NeverDisablingProof::FixpointLowerBound { lower_bound: 1 }
    );
    assert_eq!(analysis.per_rule_progress.rule_n_fixpoint_lower_bound, 1);
}

#[test]
fn test_efmnop_fixpoint_skips_lower_bound_proofs_when_rule_n_is_disabled() {
    let net = PetriNet {
        name: Some("fixpoint-lower-bound-rule-n-disabled".into()),
        places: vec![place("p_guard"), place("p_out")],
        transitions: vec![trans(
            "t_grow_guard",
            vec![arc(0, 1)],
            vec![arc(0, 2), arc(1, 1)],
        )],
        initial_marking: vec![2, 0],
    };

    let analysis = analyze_efmnop_fixpoint(&net, &[], ReductionMode::CTLWithNext);

    assert!(analysis.report.never_disabling_arcs.is_empty());
    assert_eq!(analysis.per_rule_progress.rule_n_fixpoint_lower_bound, 0);
}
