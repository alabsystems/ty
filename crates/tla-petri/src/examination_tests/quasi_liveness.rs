// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::examination::{
    collect_examination_core, liveness_verdict, liveness_verdict_with_groups,
    quasi_liveness_verdict, quasi_liveness_verdict_with_groups, Examination, ExaminationValue,
};
use crate::examinations::quasi_liveness::QuasiLivenessObserver;
use crate::explorer::{explore, explore_observer};
use crate::model::PropertyAliases;
use crate::output::Verdict;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};

use super::fixtures::{
    cyclic_safe_net, default_config, immediate_deadlock_net, linear_deadlock_net,
};

#[test]
fn test_quasi_liveness_cyclic_net_all_fire() {
    let net = cyclic_safe_net();
    let config = default_config();
    let mut observer = QuasiLivenessObserver::new(net.num_transitions());
    let result = explore(&net, &config, &mut observer);

    assert!(observer.all_fired());
    assert!(result.completed || result.stopped_by_observer);
}

#[test]
fn test_quasi_liveness_linear_net_single_transition() {
    let net = linear_deadlock_net();
    let config = default_config();
    let mut observer = QuasiLivenessObserver::new(net.num_transitions());
    let result = explore(&net, &config, &mut observer);

    assert!(observer.all_fired());
    assert!(result.stopped_by_observer);
}

#[test]
fn test_quasi_liveness_no_transitions_vacuous_true() {
    let net = immediate_deadlock_net();
    let config = default_config();
    let mut observer = QuasiLivenessObserver::new(net.num_transitions());
    let _result = explore(&net, &config, &mut observer);

    assert!(observer.all_fired());
}

#[test]
fn test_quasi_liveness_unreachable_transition_false() {
    let net = PetriNet {
        name: Some("unreachable-trans".into()),
        places: vec![
            PlaceInfo {
                id: "P0".into(),
                name: None,
            },
            PlaceInfo {
                id: "P1".into(),
                name: None,
            },
            PlaceInfo {
                id: "P2".into(),
                name: None,
            },
            PlaceInfo {
                id: "P3".into(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "T0".into(),
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
                id: "T1".into(),
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
        ],
        initial_marking: vec![1, 0, 0, 0],
    };
    let config = default_config();
    let mut observer = QuasiLivenessObserver::new(net.num_transitions());
    let result = explore(&net, &config, &mut observer);

    assert!(!observer.all_fired());
    assert!(result.completed);
}

#[test]
fn test_quasi_liveness_parallel_matches_sequential_result() {
    let net = cyclic_safe_net();
    let sequential_config = default_config();
    let mut sequential = QuasiLivenessObserver::new(net.num_transitions());
    let sequential_result = explore_observer(&net, &sequential_config, &mut sequential);

    let parallel_config = default_config().with_workers(4);
    let mut parallel = QuasiLivenessObserver::new(net.num_transitions());
    let parallel_result = explore_observer(&net, &parallel_config, &mut parallel);

    assert_eq!(parallel.all_fired(), sequential.all_fired());
    assert_eq!(parallel_result.completed, sequential_result.completed);
}

fn colored_transition_binding_net() -> PetriNet {
    PetriNet {
        name: Some("colored-binding-proxy".into()),
        places: vec![
            PlaceInfo {
                id: "P0".into(),
                name: None,
            },
            PlaceInfo {
                id: "P1".into(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "T_live".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "T_dead_binding".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![1, 0],
    }
}

fn colored_transition_aliases() -> PropertyAliases {
    let mut transition_aliases = HashMap::new();
    transition_aliases.insert(
        "ColoredT".to_string(),
        vec![TransitionIdx(0), TransitionIdx(1)],
    );
    PropertyAliases {
        place_aliases: HashMap::new(),
        transition_aliases,
        colored_place_group_aliases: HashSet::new(),
    }
}

#[test]
fn test_colored_transition_groups_cover_live_binding_for_liveness_exams() {
    let net = colored_transition_binding_net();
    let config = default_config();
    let groups = vec![vec![0, 1]];

    assert_eq!(quasi_liveness_verdict(&net, &config), Verdict::False);
    assert_eq!(
        quasi_liveness_verdict_with_groups(&net, &config, &groups),
        Verdict::True
    );
    assert_eq!(liveness_verdict(&net, &config), Verdict::False);
    assert_eq!(
        liveness_verdict_with_groups(&net, &config, &groups),
        Verdict::True
    );
}

#[test]
fn test_examination_dispatch_uses_colored_transition_groups_for_liveness_exams() {
    let net = colored_transition_binding_net();
    let config = default_config();
    let aliases = colored_transition_aliases();

    for examination in [Examination::QuasiLiveness, Examination::Liveness] {
        let records = collect_examination_core(
            &net,
            "colored-binding-proxy",
            Path::new("."),
            &aliases,
            examination,
            &config,
            false,
        )
        .expect("non-property examination should collect");

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].value,
            ExaminationValue::Verdict(Verdict::True),
            "{examination:?} dispatch must ask the colored-transition question"
        );
    }
}
