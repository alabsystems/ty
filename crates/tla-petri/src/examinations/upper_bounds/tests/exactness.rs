// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Structural and LP exactness tests for UpperBounds.

use crate::explorer::ExplorationConfig;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{Formula, Property};

use super::super::model::monotone_query_initial_bound;
use super::super::pipeline::check_upper_bounds_properties;
use super::fixtures::*;

#[test]
fn test_structural_bounds_resolve_at_initial_marking() {
    // Token-conserving net: p0 → t0 → p1 with initial [3, 0].
    // P-invariant [1,1] gives structural bound 3 for both places.
    // p0 initial = 3 = structural bound → confirmed exact.
    // p1 initial = 0 < 3 → needs exploration to find the witness state [0, 3].
    let net = simple_net();
    let props = vec![
        Property {
            id: "test-UpperBounds-00".to_string(),
            formula: Formula::PlaceBound(vec!["p0".to_string()]),
        },
        Property {
            id: "test-UpperBounds-01".to_string(),
            formula: Formula::PlaceBound(vec!["p1".to_string()]),
        },
    ];
    let config = ExplorationConfig::new(1);

    let results = check_upper_bounds_properties(&net, &props, &config);
    // p0: structural bound = 3, initial = 3 → confirmed exact
    assert_eq!(results[0], (String::from("test-UpperBounds-00"), Some(3)));
    // p1: structural bound = 3, initial = 0.
    //
    // Without `dd-backend`, BFS is the only path that can find the
    // witness state [0,3]; with `max_states=1` BFS is incomplete and
    // the result is CANNOT_COMPUTE.
    //
    // With `dd-backend`, the DD fast-path computes the *exact*
    // `max_{M ∈ R} m[p1]` symbolically — bypassing BFS entirely — so
    // p1 resolves to its true reachable maximum 3 regardless of
    // `max_states`. The DD verdict is authoritative (per
    // `dd_fastpath.rs` soundness contract) so reporting `Some(3)` is
    // correct, not a regression: it's a strict improvement over the
    // BFS-only path which would have given `None` here.
    #[cfg(not(feature = "dd-backend"))]
    assert_eq!(results[1], (String::from("test-UpperBounds-01"), None));
    #[cfg(feature = "dd-backend")]
    assert_eq!(results[1], (String::from("test-UpperBounds-01"), Some(3)));
}

#[test]
fn test_structural_bounds_all_resolve_without_bfs() {
    // When ALL properties are structurally resolved at the initial
    // marking, BFS is skipped entirely (even with max_states=1).
    let net = simple_net(); // p0 → t0 → p1, initial [3, 0]
    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        // Query: bound of p0 alone. Initial = 3 = structural bound.
        formula: Formula::PlaceBound(vec!["p0".to_string()]),
    }];
    let config = ExplorationConfig::new(1);

    let results = check_upper_bounds_properties(&net, &props, &config);
    assert_eq!(
        results,
        vec![(String::from("test-UpperBounds-00"), Some(3))]
    );
}

#[test]
fn test_structural_bounds_sum_property_conserving() {
    // Sum of p0+p1 in conserving net is always 3.
    // Structural bound for the set {p0, p1} is 3.
    // Initial marking sum = 3+0 = 3 = structural bound → resolved.
    let net = simple_net();
    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p0".to_string(), "p1".to_string()]),
    }];
    let config = ExplorationConfig::new(1);

    let results = check_upper_bounds_properties(&net, &props, &config);
    assert_eq!(
        results,
        vec![(String::from("test-UpperBounds-00"), Some(3))]
    );
}

#[test]
fn test_structural_bounds_with_full_exploration() {
    // Token-conserving net with default config (full exploration).
    // Structural bounds should match BFS-observed bounds exactly.
    let net = simple_net(); // p0 → t0 → p1, initial [3, 0]
    let props = vec![
        Property {
            id: "test-UpperBounds-00".to_string(),
            formula: Formula::PlaceBound(vec!["p0".to_string()]),
        },
        Property {
            id: "test-UpperBounds-01".to_string(),
            formula: Formula::PlaceBound(vec!["p1".to_string()]),
        },
    ];
    let config = default_config();
    let results = check_upper_bounds_properties(&net, &props, &config);

    assert_eq!(
        results,
        vec![
            (String::from("test-UpperBounds-00"), Some(3)),
            (String::from("test-UpperBounds-01"), Some(3)),
        ]
    );
}

#[test]
fn test_structural_bounds_mixed_covered_uncovered() {
    // Net with 3 places: p0 ↔ p1 (conserving), p2 unbounded (source).
    // Structural bound covers p0, p1 but NOT p2.
    let net = PetriNet {
        name: Some("mixed".to_string()),
        places: vec![
            PlaceInfo {
                id: "p0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p1".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p2".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            // T0: p0 → p1 (conserving pair)
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
            // T1: → p2 (source, unbounded)
            TransitionInfo {
                id: "t1".to_string(),
                name: None,
                inputs: vec![],
                outputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![4, 0, 0],
    };

    let props = vec![
        Property {
            id: "test-UpperBounds-00".to_string(),
            formula: Formula::PlaceBound(vec!["p0".to_string()]),
        },
        Property {
            id: "test-UpperBounds-01".to_string(),
            formula: Formula::PlaceBound(vec!["p2".to_string()]),
        },
    ];
    // Incomplete exploration: p0 gets structural bound, p2 does not.
    let config = ExplorationConfig::new(2);

    let results = check_upper_bounds_properties(&net, &props, &config);
    // p0: structural bound = 4, initial achieves it → resolved
    assert_eq!(results[0], (String::from("test-UpperBounds-00"), Some(4)));
    // p2: no structural bound, incomplete → CANNOT_COMPUTE
    assert_eq!(results[1], (String::from("test-UpperBounds-01"), None));
}

#[test]
fn test_upper_bounds_exact_after_later_witness() {
    // Bidirectional net: t0: p0→p1, t1: p1→p0.
    // Initial [1, 1]. P-invariant [1,1] → structural bound 2 for each.
    // Initial max for p1 = 1 (below cap). BFS discovers [0,2] where
    // p1 = 2 = structural bound → property proven exact mid-exploration.
    // State space: {[1,1], [0,2], [2,0]} — 3 states total.
    let net = PetriNet {
        name: Some("bidir".to_string()),
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
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![1, 1],
    };

    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p1".to_string()]),
    }];

    // Full exploration to verify the correct answer.
    let full_results = check_upper_bounds_properties(&net, &props, &default_config());
    assert_eq!(
        full_results,
        vec![(String::from("test-UpperBounds-00"), Some(2))],
        "Full exploration should find max p1 = 2"
    );

    // Now with limited exploration: only the single property tracking p1.
    // Observer's is_done() fires once p1 hits structural cap (2),
    // enabling exact result despite not visiting all states.
    let limited_config = ExplorationConfig::new(2);
    let limited_results = check_upper_bounds_properties(&net, &props, &limited_config);
    assert_eq!(
        limited_results,
        vec![(String::from("test-UpperBounds-00"), Some(2))],
        "Later witness (state [0,2]) should prove p1 exact on incomplete exploration"
    );
}

#[test]
fn test_duplicate_place_counts_do_not_resolve_early() {
    let net = simple_net(); // p0 -> p1, initial [3, 0]
    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p1".to_string(), "p1".to_string()]),
    }];

    let full_results = check_upper_bounds_properties(&net, &props, &default_config());
    assert_eq!(
        full_results,
        vec![(String::from("test-UpperBounds-00"), Some(6))],
        "Full exploration should count repeated places with multiplicity"
    );

    let limited_config = ExplorationConfig::new(3);
    let limited_results = check_upper_bounds_properties(&net, &props, &limited_config);
    // Without `dd-backend`, BFS truncates after 3 states and cannot
    // discover the witness for the multiplicity-2 weighted sum; result
    // is CANNOT_COMPUTE.
    //
    // With `dd-backend`, the DD fast-path computes the exact weighted
    // sum 6 = 2·3 symbolically off the reachable BDD — the multiplicity
    // is honoured by the per-tracker coefficient vector (see
    // `dd_fastpath.rs`). The DD verdict is authoritative; the result
    // upgrades from CANNOT_COMPUTE to the true reachable maximum.
    #[cfg(not(feature = "dd-backend"))]
    assert_eq!(
        limited_results,
        vec![(String::from("test-UpperBounds-00"), None)],
        "Repeated places must not certify exactness before the true weighted cap is reached"
    );
    #[cfg(feature = "dd-backend")]
    assert_eq!(
        limited_results,
        vec![(String::from("test-UpperBounds-00"), Some(6))],
        "DD fast-path honours place multiplicity in the weighted sum",
    );
}

fn trap_tightened_upper_bound_net() -> PetriNet {
    PetriNet {
        name: Some("trap-tightened-upper-bound".to_string()),
        places: vec![
            PlaceInfo {
                id: "p0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p1".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "q".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "kill_p1".to_string(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    },
                ],
                outputs: vec![
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    },
                ],
            },
            TransitionInfo {
                id: "kill_p0".to_string(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(1),
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
                id: "move_p1_to_p0".to_string(),
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
        ],
        initial_marking: vec![1, 1, 0],
    }
}

#[test]
fn test_trap_lp_ratifies_incomplete_bfs_observed_maximum() {
    let net = trap_tightened_upper_bound_net();
    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["q".to_string()]),
    }];

    let results = check_upper_bounds_properties(&net, &props, &ExplorationConfig::new(2));
    assert_eq!(
        results,
        vec![(String::from("test-UpperBounds-00"), Some(1))],
        "Trap-LP should prove that no marking with q >= 2 is reachable after BFS observes q = 1"
    );
}

#[test]
fn test_trap_lp_declines_to_ratify_loose_observed_maximum() {
    let net = trap_tightened_upper_bound_net();
    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["q".to_string()]),
    }];

    let results = check_upper_bounds_properties(&net, &props, &ExplorationConfig::new(1));
    #[cfg(not(feature = "dd-backend"))]
    assert_eq!(
        results,
        vec![(String::from("test-UpperBounds-00"), None)],
        "Trap-LP must not certify q = 0 because q = 1 is reachable"
    );
    #[cfg(feature = "dd-backend")]
    assert_eq!(
        results,
        vec![(String::from("test-UpperBounds-00"), Some(1))],
        "DD fast-path computes the exact bound before BFS/trap-LP"
    );
}

#[test]
fn test_trap_lp_declines_fractional_duplicate_gap() {
    let net = trap_tightened_upper_bound_net();
    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["q".to_string(), "q".to_string()]),
    }];

    let results = check_upper_bounds_properties(&net, &props, &ExplorationConfig::new(2));
    #[cfg(not(feature = "dd-backend"))]
    assert_eq!(
        results,
        vec![(String::from("test-UpperBounds-00"), None)],
        "The continuous trap-LP relaxation must decline the fractional 2*q >= 3 gap"
    );
    #[cfg(feature = "dd-backend")]
    assert_eq!(
        results,
        vec![(String::from("test-UpperBounds-00"), Some(2))],
        "DD fast-path computes the exact duplicate-place weighted bound"
    );
}

#[test]
fn test_monotone_query_resolves_at_initial_marking_without_bfs() {
    // Draining net: p0 can only decrease. The exact maximum of bound(p0)
    // is therefore the initial marking, even if exploration has no useful
    // budget.
    let net = PetriNet {
        name: Some("drain".to_string()),
        places: vec![PlaceInfo {
            id: "p0".to_string(),
            name: None,
        }],
        transitions: vec![TransitionInfo {
            id: "drain".to_string(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: vec![],
        }],
        initial_marking: vec![3],
    };
    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p0".to_string()]),
    }];
    let config = ExplorationConfig::new(0);

    let results = check_upper_bounds_properties(&net, &props, &config);
    assert_eq!(
        results,
        vec![(String::from("test-UpperBounds-00"), Some(3))]
    );
}

#[test]
fn test_monotone_query_honors_repeated_place_coefficients() {
    // p0 -> p1 increases bound(p1), but decreases the weighted query
    // bound(p0, p0, p1): delta is -2 + 1 <= 0.
    let net = PetriNet {
        name: Some("weighted-move".to_string()),
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
            id: "move".to_string(),
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
        initial_marking: vec![2, 0],
    };

    assert_eq!(
        monotone_query_initial_bound(&net, &[PlaceIdx(0), PlaceIdx(0), PlaceIdx(1)]),
        Some(4)
    );
    assert_eq!(monotone_query_initial_bound(&net, &[PlaceIdx(1)]), None);
}

// ── LP-only exactness tests ─────────────────────────────────────────

#[test]
fn test_lp_only_bound_resolves_without_p_invariant() {
    // Cycle-with-sink net: p0 ↔ p1 (cycle) + p0 → sink.
    //
    //   t0: p0 → p1   (move token from p0 to p1)
    //   t1: p1 → p0   (move token from p1 to p0)
    //   t2: p0 → ∅    (drain p0)
    //
    // Initial marking: [1, 1].
    //
    // P-invariant analysis: y·C = 0 where C columns are
    //   t0=[-1,1], t1=[1,-1], t2=[-1,0].
    // From t2: y0 = 0, then y1 = 0. No non-trivial P-invariant.
    // Therefore structural_bound = None for any place set.
    //
    // LP state equation: max(m1) = max(1 + x0 - x1) subject to
    //   1 - x0 + x1 - x2 ≥ 0, x ≥ 0.
    // At x1=0, x2=0: x0 ≤ 1 → max = 2. So lp_bound = Some(2).
    //
    // State space: {[1,1],[0,2],[2,0],[0,1],[1,0],[0,0]}.
    // max(p1) = 2 at [0,2].
    //
    // Both places have initial > 0, so agglomeration is blocked and
    // both survive reduction. The exactness proof comes from lp_bound,
    // NOT structural_bound.
    let net = PetriNet {
        name: Some("cycle-sink".to_string()),
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
        transitions: vec![
            // t0: p0 → p1
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
            // t1: p1 → p0
            TransitionInfo {
                id: "t1".to_string(),
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
            // t2: p0 → (sink)
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
        initial_marking: vec![1, 1],
    };

    let props = vec![Property {
        id: "test-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p1".to_string()]),
    }];

    // Full exploration: max(p1) = 2 at state [0, 2].
    let full_results = check_upper_bounds_properties(&net, &props, &default_config());
    assert_eq!(
        full_results,
        vec![(String::from("test-UpperBounds-00"), Some(2))],
        "Full exploration: max(p1) = 2"
    );

    // Incomplete exploration (2 states): visits [1,1] then [0,2]
    // (t0 fires first). Observed max = 2. LP cap = 2.
    // Since observed == lp_bound, the property is proven exact
    // even though exploration is incomplete and no P-invariant
    // covers p1.
    let limited_results = check_upper_bounds_properties(&net, &props, &ExplorationConfig::new(2));
    assert_eq!(
        limited_results,
        vec![(String::from("test-UpperBounds-00"), Some(2))],
        "LP cap proves exactness on incomplete exploration without P-invariant"
    );
}

// ── FINDING #11: trap-LP must not ratify an unwitnessed reduced-net max ──
//
// `confirm_observed_maxima_with_trap_lp` proves only the CEILING (`max_bound +
// 1` is unreachable). It must not certify `max_bound` itself as the exact
// reachable maximum unless `max_bound` is a confirmed reachable WITNESS. When
// the observation came from a reduced net whose reduction OVER-counted the
// query (e.g. GCD/arc-weight scaling, self-loop stripping), `max_bound` can
// exceed the true original-net maximum, and ratifying it would publish a
// too-high — i.e. WRONG — UpperBounds value.
//
// Constructed reproducer: a tiny net whose true reachable max of `q` is 1,
// paired with a reduced net carrying `place_scales[q] = 2` (modeling a scaling
// reduction). A reduced-net BFS value of 1 expands to an observed `max_bound =
// 2`. The original-net true max is 1, so trap-LP correctly proves `q >= 3` (=
// max_bound + 1) unreachable. The OLD all-witnessed ratification then ratifies
// the inflated 2; the NEW witness-gated ratification declines because `q`
// underwent a value-transforming reduction.
fn finding11_scaling_overcount_net() -> PetriNet {
    // q (idx 0, init 0), s (idx 1, init 1). `produce`: s -> q.
    // Reachable markings: [0,1] and [1,0]. True max of q is 1.
    PetriNet {
        name: Some("finding11-scaling-overcount".to_string()),
        places: vec![
            PlaceInfo {
                id: "q".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "s".to_string(),
                name: None,
            },
        ],
        transitions: vec![TransitionInfo {
            id: "produce".to_string(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(1),
                weight: 1,
            }],
            outputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
        }],
        initial_marking: vec![0, 1],
    }
}

#[test]
fn test_trap_lp_old_ratifies_inflated_reduced_max_new_declines() {
    use super::super::model::BoundTracker;
    use super::super::pipeline::{
        confirm_observed_maxima_with_trap_lp, confirm_observed_maxima_with_trap_lp_witnessed,
        reduced_observation_witnesses_max,
    };
    use crate::reduction::ReducedNet;

    let net = finding11_scaling_overcount_net();

    // Cross-check the GROUND TRUTH via exact full-net BFS: max(q) == 1.
    let props = vec![Property {
        id: "finding11-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["q".to_string()]),
    }];
    let oracle = check_upper_bounds_properties(&net, &props, &default_config());
    assert_eq!(
        oracle,
        vec![(String::from("finding11-UpperBounds-00"), Some(1))],
        "exact full-net BFS: true reachable max of q is 1",
    );

    // Reduced net modeling a scaling reduction on `q`: reduced value 1 expands
    // to original q = 2. This is the over-count vector. The tracker's seed
    // `lp_bound = 3` is deliberately loose (above the inflated 2), placing us in
    // the trap-LP refinement regime where the OLD code ratifies the inflated
    // value.
    let mut reduced = ReducedNet::identity(&net);
    reduced.place_scales[0] = 2; // q underwent scaling

    let make_tracker = || BoundTracker {
        id: "finding11-UpperBounds-00".to_string(),
        place_indices: vec![PlaceIdx(0)],
        max_bound: 2, // inflated reduced-net observation (true max is 1)
        structural_bound: None,
        lp_bound: Some(3), // loose seed, strictly above the inflated max
        monotone_bound: None,
    };

    // The scaled query place is NOT a reachability witness.
    let witnessed = reduced_observation_witnesses_max(&reduced, &net, &make_tracker());
    assert!(
        !witnessed,
        "a scaled (place_scales != 1) query place must not be treated as a reachable witness",
    );

    // OLD behavior (all observations treated as witnessed): trap-LP proves
    // `q >= 3` unreachable and ratifies the inflated max_bound = 2.
    let mut old_trackers = vec![make_tracker()];
    let old_ratified = confirm_observed_maxima_with_trap_lp(&net, &mut old_trackers);
    assert_eq!(old_ratified, 1, "OLD ratifies the unwitnessed observation");
    assert!(
        old_trackers[0].is_structurally_resolved(),
        "OLD marks the inflated tracker resolved",
    );
    assert_eq!(
        old_trackers[0].max_bound, 2,
        "OLD would publish the inflated bound 2 (WRONG — true max is 1)",
    );

    // NEW behavior (witness-gated): the scaled query is unwitnessed, so trap-LP
    // declines to ratify. The tracker stays unresolved with its loose cap, so
    // the inflated 2 is never published as exact; the downstream exact / safety
    // -net path resolves it at the true reachable max.
    let mut new_trackers = vec![make_tracker()];
    let new_ratified =
        confirm_observed_maxima_with_trap_lp_witnessed(&net, &mut new_trackers, |_| witnessed);
    assert_eq!(
        new_ratified, 0,
        "NEW declines to ratify the unwitnessed (inflated) observation",
    );
    assert!(
        !new_trackers[0].is_structurally_resolved(),
        "NEW leaves the inflated tracker unresolved (lp_bound stays 3 > max_bound 2)",
    );
}

// Regression: the TRIVIAL (initial-marking) witness early-return must also honor
// unit-scale. The existing finding-11 test exercises the per-place *passthrough*
// guard (it has `max_bound != initial_sum`, so it never hits the trivial path).
// The trivial witness returns `true` as soon as `max_bound == initial_sum`, and
// before the fix it did so WITHOUT checking `place_scales` — so a GCD-scaled
// query place whose reduced (scaled-unit) observation coincidentally equals the
// original-unit `initial_sum` was wrongly ratified as a reachable witness. This
// pins the unit-scale guard on the trivial path (both the `reduced_*` function
// and the `used_slice` inline check share the same predicate).
#[test]
fn test_trivial_witness_requires_unit_scale_on_scaled_query() {
    use super::super::model::BoundTracker;
    use super::super::pipeline::reduced_observation_witnesses_max;
    use crate::reduction::ReducedNet;

    // q (idx 0, init 2), s (idx 1, init 0); one transition keeps the net valid.
    let net = PetriNet {
        name: Some("trivial-witness-scale".to_string()),
        places: vec![
            PlaceInfo {
                id: "q".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "s".to_string(),
                name: None,
            },
        ],
        transitions: vec![TransitionInfo {
            id: "t".to_string(),
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
        initial_marking: vec![2, 0],
    };

    // Query = {q}. initial_sum = 2. Set max_bound == initial_sum so the trivial
    // witness is the path under test.
    let make_tracker = || BoundTracker {
        id: "trivial-UpperBounds-00".to_string(),
        place_indices: vec![PlaceIdx(0)],
        max_bound: 2,
        structural_bound: None,
        lp_bound: None,
        monotone_bound: None,
    };

    // Unit-scale control: an identity reduction (place_scales all 1) — the
    // trivial witness is sound and must still fire.
    let unit = ReducedNet::identity(&net);
    assert!(
        reduced_observation_witnesses_max(&unit, &net, &make_tracker()),
        "unit-scale query with max_bound == initial_sum is a valid trivial witness",
    );

    // Scaled query: q underwent GCD scaling (place_scales[q] = 2), so the
    // reduced-net observation is in *scaled* units and is NOT comparable to the
    // original-unit initial_sum. The trivial witness must be withheld.
    let mut scaled = ReducedNet::identity(&net);
    scaled.place_scales[0] = 2;
    assert!(
        !reduced_observation_witnesses_max(&scaled, &net, &make_tracker()),
        "a GCD-scaled query place must NOT be trivially witnessed even when \
         max_bound == initial_sum (different unit systems)",
    );
}

// A2 coverage lever: `tighten_lp_bounds_on_reduced_net` lowers an unresolved
// tracker's LP ceiling using the reduced-net LP when (and only when) every query
// place is a faithful unit-scale passthrough. Pins three properties: (1) a
// faithful tracker's loose ceiling is tightened to the reduced-net LP bound;
// (2) a GCD-scaled query is skipped (the gate that keeps the reduced ceiling a
// valid original-net ceiling); (3) it only ever LOWERS — a looser reduced bound
// is ignored. Sound because resolution emits `max_bound` exact once it reaches
// the (still-valid) ceiling.
#[test]
fn test_reduced_net_lp_tightens_only_faithful_unit_scale_queries() {
    use super::super::model::BoundTracker;
    use super::super::pipeline::tighten_lp_bounds_on_reduced_net;
    use crate::reduction::ReducedNet;

    // Token-conserving 2-place net: p0+p1 == 3 invariant, so the LP upper bound
    // of p0 is exactly 3.
    let net = PetriNet {
        name: Some("conserve3".to_string()),
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
        transitions: vec![
            TransitionInfo {
                id: "f".to_string(),
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
                id: "b".to_string(),
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
        ],
        initial_marking: vec![3, 0],
    };
    // Sanity: the reduced-net LP bound of p0 is 3.
    assert_eq!(
        crate::lp_state_equation::lp_upper_bound(&net, &[PlaceIdx(0)]),
        Some(3),
        "conserving net pins p0's LP upper bound at 3",
    );

    let make_tracker = |lp: Option<u64>| BoundTracker {
        id: "ub-00".to_string(),
        place_indices: vec![PlaceIdx(0)],
        max_bound: 0, // unresolved: observed below any ceiling
        structural_bound: None,
        lp_bound: lp,
        monotone_bound: None,
    };

    // (1) Faithful (identity) reduced net: a loose ceiling 5 tightens to 3.
    let faithful = ReducedNet::identity(&net);
    let mut trackers = vec![make_tracker(Some(5))];
    let n = tighten_lp_bounds_on_reduced_net(&faithful, &mut trackers);
    assert_eq!(
        n, 1,
        "faithful query with a tighter reduced LP must tighten"
    );
    assert_eq!(
        trackers[0].lp_bound,
        Some(3),
        "ceiling lowered to the reduced-net LP bound",
    );

    // (2) GCD-scaled query place: gate must skip it (reduced ceiling would be in
    // scaled units — not a valid original-net ceiling), leaving the ceiling as-is.
    let mut scaled = ReducedNet::identity(&net);
    scaled.place_scales[0] = 2;
    let mut trackers = vec![make_tracker(Some(5))];
    let n = tighten_lp_bounds_on_reduced_net(&scaled, &mut trackers);
    assert_eq!(n, 0, "a GCD-scaled query must be skipped");
    assert_eq!(
        trackers[0].lp_bound,
        Some(5),
        "scaled query ceiling untouched"
    );

    // (3) Only ever lowers: a tracker whose existing ceiling (2) is already
    // tighter than the reduced LP (3) is left unchanged.
    let mut trackers = vec![make_tracker(Some(2))];
    let n = tighten_lp_bounds_on_reduced_net(&faithful, &mut trackers);
    assert_eq!(n, 0, "a looser reduced LP must not raise the ceiling");
    assert_eq!(
        trackers[0].lp_bound,
        Some(2),
        "existing tighter ceiling kept"
    );
}
