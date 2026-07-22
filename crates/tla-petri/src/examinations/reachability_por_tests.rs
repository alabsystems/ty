// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::*;
use crate::explorer::{explore_observer, ExplorationConfig};
use crate::output::Verdict;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{
    Formula, IntExpr, PathQuantifier, Property, ReachabilityFormula, StatePredicate,
};
use crate::reduction::ParallelExpandingObserver;
use crate::stubborn::PorStrategy;

use crate::examinations::reachability_por::reachability_por_config;

fn make_ef_prop(id: &str, predicate: StatePredicate) -> Property {
    Property {
        id: id.to_string(),
        formula: Formula::Reachability(ReachabilityFormula {
            quantifier: PathQuantifier::EF,
            predicate,
        }),
    }
}

fn por_visible_subset_budget_net() -> PetriNet {
    // Keep disconnected irrelevant cycles alive after query-protected reduction.
    // One-shot irrelevant transitions are prefired away before POR sees them,
    // which collapses the reduced net and no longer exercises visible-subset POR.
    PetriNet {
        name: Some("por-visible-subset-budget".to_string()),
        places: vec![
            PlaceInfo {
                id: "r0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "r1".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "r2".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "r3".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "r4".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "i0_l".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "i0_r".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "i1_l".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "i1_r".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "i2_l".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "i2_r".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "i3_l".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "i3_r".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "i0_f".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(5),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(6),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "i0_b".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(6),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(5),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "i1_f".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(7),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(8),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "i1_b".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(8),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(7),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "i2_f".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(9),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(10),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "i2_b".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(10),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(9),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "i3_f".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(11),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(12),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "i3_b".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(12),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(11),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "a0".to_string(),
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
                id: "a1".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "a2".to_string(),
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
                id: "a3".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(3),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(4),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "t2".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![],
            },
        ],
        initial_marking: vec![1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    }
}

/// Same structure as `token_elimination_net()` in reduction/tests/support.rs,
/// but using explicit struct construction (no helper fns from test support).
///
/// Two P-invariants: p_cap + 2*p_mid = 10, p_mid + p_obs = 3.
/// Lower(p_cap) = 4 >= 2 (t_fwd consumption) → never-disabling proof.
fn query_token_elimination_net() -> PetriNet {
    PetriNet {
        name: Some("reachability-token-elimination".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_cap".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_mid".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_obs".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_aux".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            // t_fwd: [p_cap:2, p_obs:1] → [p_mid:1]
            TransitionInfo {
                id: "t_fwd".to_string(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(0),
                        weight: 2,
                    },
                    Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    },
                ],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            },
            // t_back: [p_mid:1] → [p_cap:2, p_obs:1, p_aux:1]
            TransitionInfo {
                id: "t_back".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![
                    Arc {
                        place: PlaceIdx(0),
                        weight: 2,
                    },
                    Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(3),
                        weight: 1,
                    },
                ],
            },
        ],
        initial_marking: vec![4, 3, 0, 0],
    }
}

fn agglomeration_net() -> PetriNet {
    PetriNet {
        name: Some("agglomeration".to_string()),
        places: vec![
            PlaceInfo {
                id: "p0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_mid".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p2".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
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
            },
            TransitionInfo {
                id: "t1".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![1, 0, 0],
    }
}

fn write_fake_solver_script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    let script = format!("#!/bin/sh\nset -eu\n{body}\n");
    fs::write(&path, script).expect("failed to write fake solver script");
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(&path)
            .expect("script metadata should exist")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).expect("failed to mark fake solver executable");
    }
    path
}

fn with_ay_path<T>(path: &Path, f: impl FnOnce() -> T) -> T {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();
    let previous = std::env::var("AY_PATH").ok();
    crate::env_guard::set_var("AY_PATH", path);
    let result = f();
    match previous {
        Some(value) => crate::env_guard::set_var("AY_PATH", value),
        None => crate::env_guard::remove_var("AY_PATH"),
    }
    result
}

#[test]
fn test_reachability_por_disables_when_reduction_leaves_no_invisible_subset() {
    let net = por_visible_subset_budget_net();
    let properties = vec![make_ef_prop(
        "goal",
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["r4".to_string()]),
        ),
    )];
    let (_, trackers) = prepare_trackers(&net, &properties);
    let reduced = reduce_reachability_queries(&net, &trackers)
        .expect("query-protected reduction should succeed");
    let base_config = ExplorationConfig::new(9);

    let por_config = reachability_por_config(&reduced, &trackers, &base_config);
    assert!(
        matches!(por_config.por_strategy, PorStrategy::None),
        "query-protected reduction already prunes to the single visible transition"
    );

    let mut por_observer = ReachabilityObserver::from_trackers(&net, trackers);
    let por_result = {
        let mut expanding = ParallelExpandingObserver::new(&reduced, &mut por_observer);
        explore_observer(&reduced.net, &por_config, &mut expanding)
    };
    assert_eq!(por_observer.results(), vec![("goal", Some(true))]);
    assert!(por_result.stopped_by_observer);
}

#[test]
fn test_reachability_por_preserves_config_invariants_when_disabled_after_reduction() {
    let net = por_visible_subset_budget_net();
    let properties = vec![make_ef_prop(
        "goal",
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["r4".to_string()]),
        ),
    )];
    let (_, trackers) = prepare_trackers(&net, &properties);
    let reduced = reduce_reachability_queries(&net, &trackers)
        .expect("query-protected reduction should succeed");

    let fingerprint_on = ExplorationConfig::auto_sized(&net, None, Some(0.01));
    let fingerprint_on_budget = fingerprint_on.max_states();
    let base_config = fingerprint_on.with_fingerprint_dedup(false).with_workers(3);
    let fingerprint_off_budget = base_config.max_states();

    let por_config = reachability_por_config(&reduced, &trackers, &base_config);
    assert!(
        matches!(por_config.por_strategy, PorStrategy::None),
        "POR should be disabled when reduced net has no strict invisible subset"
    );
    assert!(
        !por_config.fingerprint_dedup(),
        "derived POR config should preserve fingerprint-off mode"
    );
    assert_eq!(
        por_config.max_states(),
        fingerprint_off_budget,
        "derived POR config should preserve the active fingerprint-off budget"
    );
    assert_eq!(
        por_config.workers(),
        3,
        "derived POR config should preserve worker settings"
    );

    let restored = por_config.with_fingerprint_dedup(true);
    assert_eq!(
        restored.max_states(),
        fingerprint_on_budget,
        "derived POR config should retain auto-sizing basis for future recomputation"
    );
}

#[test]
fn test_reachability_por_keeps_protected_fireability_transition() {
    let net = agglomeration_net();
    let properties = vec![make_ef_prop(
        "fire-t0",
        StatePredicate::IsFireable(vec!["t0".to_string()]),
    )];
    let (_, trackers) = prepare_trackers(&net, &properties);
    let reduced = reduce_reachability_queries(&net, &trackers)
        .expect("query-protected reduction should succeed");
    let reduced_t0 = reduced.transition_map[0];
    assert!(
        reduced_t0.is_some(),
        "IsFireable support protects the referenced transition through reduction"
    );

    let por_config = reachability_por_config(&reduced, &trackers, &ExplorationConfig::new(32));
    match por_config.por_strategy {
        PorStrategy::SafetyPreserving { visible } => {
            assert!(
                visible.contains(&reduced_t0.expect("checked Some above")),
                "fireability-target transition must remain POR-visible"
            );
        }
        PorStrategy::None => {
            assert_eq!(
                reduced.net.num_transitions(),
                1,
                "POR may only disable here if no strict invisible subset remains"
            );
        }
        PorStrategy::DeadlockPreserving => panic!("reachability should not use deadlock POR"),
    }
}

#[test]
fn test_reachability_reduction_token_eliminates_unobserved_capacity_place() {
    let net = query_token_elimination_net();
    // Query: EF(p_obs >= 1). Protects only p_obs (PlaceIdx(2)).
    // p_cap (PlaceIdx(0)) is unobserved with lower_bound=4 >= 2 (t_fwd consumption)
    // → token elimination removes p_cap.
    let properties = vec![make_ef_prop(
        "goal",
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p_obs".to_string()]),
        ),
    )];

    let (_, trackers) = prepare_trackers(&net, &properties);
    let reduced = reduce_reachability_queries(&net, &trackers)
        .expect("query-protected reduction should succeed");

    assert_eq!(reduced.report.source_places, vec![PlaceIdx(3)]);
    assert_eq!(reduced.report.token_eliminated_places, vec![PlaceIdx(0)]);
    assert_eq!(reduced.place_map[0], None);

    // Verify the property is satisfiable on the original net:
    // fire t_back → p_obs goes from 0 to 1, satisfying EF(p_obs >= 1).
    assert_eq!(
        check_reachability_properties(&net, &properties, &ExplorationConfig::new(64)),
        vec![(String::from("goal"), Verdict::True)]
    );
}

#[test]
fn test_check_reachability_properties_uses_por_when_bmc_is_inconclusive() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay",
        "cat >/dev/null\nprintf 'unknown\\n'",
    );
    let net = por_visible_subset_budget_net();
    let properties = vec![make_ef_prop(
        "goal",
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["r4".to_string()]),
        ),
    )];

    let results = with_ay_path(&solver, || {
        check_reachability_properties(&net, &properties, &ExplorationConfig::new(9))
    });

    assert_eq!(results, vec![(String::from("goal"), Verdict::True)]);
}
