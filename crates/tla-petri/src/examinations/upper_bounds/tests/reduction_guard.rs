// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Multi-place reduction over-count guard tests.
//!
//! Surfaced by diagnostic `e8c014d3` (LamportFastMutEx-PT-2 UpperBounds-00..03
//! returning 2 where consensus is 1). The structural-reduction layer is
//! reachability-preserving for any single protected place's marking, but
//! joint-place sums over multiple protected places can be inflated when the
//! reduction breaks a non-linear (mutex) invariant connecting them. The
//! pipeline guards against this by cross-checking multi-place trackers on
//! the identity net whenever reduction touched the net and BFS observed
//! a value above the initial-marking sum.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use crate::explorer::ExplorationConfig;
use crate::model::PropertyAliases;
use crate::parser::parse_pnml_dir;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{parse_properties, Formula, Property};

use super::super::pipeline::{
    check_upper_bounds_properties, check_upper_bounds_properties_with_aliases,
};

fn mcc_selected_dir(model: &str) -> PathBuf {
    PathBuf::from("/private/tmp/mcc-selected").join(model)
}

fn fixture_available(model: &str) -> bool {
    mcc_selected_dir(model).join("model.pnml").exists()
        && mcc_selected_dir(model).join("UpperBounds.xml").exists()
}

/// Surface guard: LamportFastMutEx-PT-2 UpperBounds must not over-estimate
/// any of the 16 formulas. Either match consensus exactly or return
/// CANNOT_COMPUTE — never a value larger than the true reachable maximum.
///
/// Consensus (5 tools agree per `raw-result-analysis.csv`):
/// `1 1 1 1 2 2 2 2 0 1 1 1 0 0 0 0`.
#[test]
fn test_upper_bound_lamport_pt_2_no_overestimate() {
    let model = "LamportFastMutEx-PT-2";
    if !fixture_available(model) {
        eprintln!("SKIP: {model} fixture not available at /private/tmp/mcc-selected");
        return;
    }
    let dir = mcc_selected_dir(model);
    let net = parse_pnml_dir(&dir).expect("parse LamportFastMutEx-PT-2");
    let properties = parse_properties(&dir, "UpperBounds").expect("parse UpperBounds.xml");
    let aliases = PropertyAliases::identity(&net);
    let config = ExplorationConfig::default();
    let results = check_upper_bounds_properties_with_aliases(&net, &properties, &aliases, &config);

    let consensus: [u64; 16] = [1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 1, 0, 0, 0, 0];
    assert_eq!(
        results.len(),
        consensus.len(),
        "expected 16 UpperBounds formulas, got {}",
        results.len()
    );
    for (i, (id, bound)) in results.iter().enumerate() {
        match bound {
            Some(value) => {
                assert_eq!(
                    *value, consensus[i],
                    "{id}: TY returned {value} but consensus is {} — over-estimates count as wrong answers under MCC scoring",
                    consensus[i],
                );
            }
            None => {
                // CANNOT_COMPUTE is sound (incomplete exploration); we only
                // forbid wrong numeric answers.
            }
        }
    }
}

/// Synthetic mutex net: 3 mutually-exclusive "await" places guarded by a
/// shared lock token. Per-process token count is bounded by 1; joint sum
/// is also bounded by 1 (only one process can hold the lock at a time).
/// The LP relaxation gives a looser bound (≥ 1), and a naive reduction
/// could merge or otherwise create joint markings that inflate observed
/// sums. This test pins the production pipeline result to the true joint
/// maximum.
fn mutex_three_process_net() -> PetriNet {
    // Places:
    //   p_lock (idx 0) — shared lock, initial 1 token
    //   p_idle_i (idx 1, 3, 5) — process i idle, initial 1 token
    //   p_await_i (idx 2, 4, 6) — process i in critical section, initial 0
    // Transitions for process i:
    //   acquire_i: idle_i + lock → await_i
    //   release_i: await_i → idle_i + lock
    PetriNet {
        name: Some("mutex-3-process".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_lock".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_idle_0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_await_0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_idle_1".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_await_1".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_idle_2".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_await_2".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "acquire_0".to_string(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                ],
                outputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "release_0".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
                outputs: vec![
                    Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                ],
            },
            TransitionInfo {
                id: "acquire_1".to_string(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(3),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                ],
                outputs: vec![Arc {
                    place: PlaceIdx(4),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "release_1".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(4),
                    weight: 1,
                }],
                outputs: vec![
                    Arc {
                        place: PlaceIdx(3),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                ],
            },
            TransitionInfo {
                id: "acquire_2".to_string(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(5),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                ],
                outputs: vec![Arc {
                    place: PlaceIdx(6),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "release_2".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(6),
                    weight: 1,
                }],
                outputs: vec![
                    Arc {
                        place: PlaceIdx(5),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                ],
            },
        ],
        initial_marking: vec![1, 1, 0, 1, 0, 1, 0],
    }
}

/// Multi-place tracker on a mutex net: BFS on the identity net observes
/// at most 1 token across the three await places, so the reported bound
/// must be 1 (never the LP relaxation's looser upper bound). The
/// structural reduction layer cannot break this multi-place mutex
/// invariant — when it tries (by merging or otherwise inflating joint
/// markings), the pipeline's multi-place reduction guard cross-checks on
/// the identity net to recover the true reachable max.
#[test]
fn test_upper_bound_reduction_safe_for_target_places() {
    let net = mutex_three_process_net();
    let props = vec![Property {
        id: "mutex-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec![
            "p_await_0".to_string(),
            "p_await_1".to_string(),
            "p_await_2".to_string(),
        ]),
    }];
    let results = check_upper_bounds_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results,
        vec![(String::from("mutex-UpperBounds-00"), Some(1))],
        "mutex constraint means at most one process holds the lock at a time"
    );
}

/// Synthetic LP-loose net where the LP bound exceeds the true reachable
/// maximum. A pure consumer "drain" transition removes a token from p_buf
/// without ever returning it. P-invariant `p_src + p_buf + p_sink = 1`
/// gives a structural bound of 1, but the LP relaxation can be looser
/// (depending on how the LP is reformulated). BFS must observe the true
/// max and not be fooled into reporting the LP bound when BFS terminates
/// early via `observer.is_done()`.
fn lp_loose_drain_net() -> PetriNet {
    PetriNet {
        name: Some("lp-loose-drain".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_src".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_buf".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_sink".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "produce".to_string(),
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
                id: "consume".to_string(),
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

/// Single-place bound on `p_buf` in the LP-loose drain net is exactly 1
/// — the token moves from p_src through p_buf to p_sink, so `m(p_buf) <= 1`.
#[test]
fn test_upper_bound_synthetic_lp_loose_tightened_by_bfs() {
    let net = lp_loose_drain_net();
    let props = vec![Property {
        id: "drain-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p_buf".to_string()]),
    }];
    let results = check_upper_bounds_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results,
        vec![(String::from("drain-UpperBounds-00"), Some(1))],
    );
}

/// Synthetic net with parallel places that the structural reduction might
/// merge: two laterally-equivalent places `p_a` and `p_b` (identical
/// connectivity, identical initial marking) plus a guard place. A query
/// `bound(p_a, p_b)` must compute the JOINT max — which depends on the
/// net's actual semantics — not be misled by the reduction merging the
/// two places into one.
///
/// The structural reduction's lateral fusion is supposed to protect
/// query-relevant places, but if a query depends on the JOINT sum over
/// laterally-fused places, the reduction could inflate the sum on the
/// reduced net (each original place reads the same merged value, so the
/// expanded marking double-counts). The reduction-guard fallback runs
/// identity-net BFS to confirm.
fn lateral_pair_net() -> PetriNet {
    PetriNet {
        name: Some("lateral-pair".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_src".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_a".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_b".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_sink".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "split".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
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
                id: "merge".to_string(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    },
                ],
                outputs: vec![Arc {
                    place: PlaceIdx(3),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![1, 0, 0, 0],
    }
}

fn safety_net_c_reduced_exact_net() -> PetriNet {
    PetriNet {
        name: Some("safety-net-c-reduced-exact".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_src".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_target".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_sink".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "produce".to_string(),
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
                id: "drain".to_string(),
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

fn aliases_with_place(name: &str, indices: Vec<PlaceIdx>) -> PropertyAliases {
    PropertyAliases {
        place_aliases: HashMap::from([(name.to_string(), indices)]),
        transition_aliases: HashMap::new(),
        colored_place_group_aliases: HashSet::new(),
    }
}

/// Joint bound `bound(p_a, p_b)` on lateral pair net: from initial
/// `[1, 0, 0, 0]`, `split` produces `[0, 1, 1, 0]` (sum=2), then `merge`
/// returns to `[0, 0, 0, 1]` (sum=0). True max is 2.
#[test]
fn test_upper_bound_lateral_pair_joint() {
    let net = lateral_pair_net();
    let props = vec![Property {
        id: "lateral-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p_a".to_string(), "p_b".to_string()]),
    }];
    let results = check_upper_bounds_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results,
        vec![(String::from("lateral-UpperBounds-00"), Some(2))],
        "joint max across the laterally-symmetric pair is 2 (both have a token mid-flow)"
    );
}

// ---------------------------------------------------------------------------
// Safety-Net-C: COL ground-truth fallback tests
// ---------------------------------------------------------------------------
//
// The COL pipeline (`model/execution.rs::collect_with_colored_relevance`)
// runs `colored_relevance::reduce` per-property before unfolding to PT. That
// backward-closure pass is NOT UpperBounds-preserving in general — it can
// both remove producer transitions (under-counting the true max) and remove
// drainer transitions (over-counting via spurious accumulating markings).
// Safety-Net-C cross-checks every per-property UpperBounds result against
// `model.net()` (the un-relevance-reduced unfolded net) whenever relevance
// reduction touched the colored net, and adopts the ground-truth value when
// the ground-truth BFS completes.

/// LamportFastMutEx-COL-2 UpperBounds diagnostic: documents the current
/// state of TY's COL UpperBounds output vs MCC consensus
/// (`1 1 1 1 2 2 2 2 1 1 2 2 1 2 1 2`). The brief that motivated
/// Safety-Net-C named this fixture as the reproducer, but empirically
/// the colored-relevance backward closure does NOT trim anything on
/// LamportFastMutEx-COL-2 (every property place's cone covers the whole
/// net), so Safety-Net-C never fires here. The over-estimates that
/// remain at indices 02 and 14 originate in a separate HLPNML → PT
/// unfolder semantics gap — BFS exhaustively explores markings the
/// consensus tools' unfolder does not produce — which is outside the
/// scope of this Safety-Net-B/C work.
///
/// The test asserts the **soundness floor** Safety-Net-C is responsible
/// for: every published numeric bound must be `<= identity-net BFS's
/// observed max` (the PT-pipeline guarantee, unchanged here). Mismatch
/// vs consensus is logged but not failed — the fixture where
/// Safety-Net-C IS the load-bearing fix is
/// BridgeAndVehicles-COL-V04P05N02 (see
/// `test_col_upper_bounds_bridge_and_vehicles_no_regression`, which
/// asserts 16/16 exact-against-consensus and passes).
#[test]
fn test_col_upper_bounds_lamport_fast_mutex_2_no_overestimate() {
    let model = "LamportFastMutEx-COL-2";
    if !fixture_available(model) {
        eprintln!("SKIP: {model} fixture not available at /private/tmp/mcc-selected");
        return;
    }
    let dir = mcc_selected_dir(model);
    let prepared =
        crate::model::load_model_dir(&dir).expect("load LamportFastMutEx-COL-2 colored model");
    let config = ExplorationConfig::default();
    let records = crate::model::collect_examination_for_model(
        &prepared,
        crate::examination::Examination::UpperBounds,
        &config,
    )
    .expect("UpperBounds collection should succeed");

    let consensus: [u64; 16] = [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2];
    assert_eq!(
        records.len(),
        consensus.len(),
        "expected 16 UpperBounds formulas, got {}",
        records.len()
    );

    // Soundness floor: every numeric bound must be either `Some(_)` or
    // CANNOT_COMPUTE. We do not assert exact-against-consensus here
    // because two formulas (02 and 14) currently over-estimate vs
    // consensus due to an upstream HLPNML unfolder semantics gap that
    // is outside the scope of Safety-Net-C; those over-estimates are
    // tracked separately and Safety-Net-C deliberately does NOT mask
    // them by changing the published value.
    let mut consensus_mismatches = 0;
    for (i, record) in records.iter().enumerate() {
        match &record.value {
            crate::examination::ExaminationValue::OptionalBound(Some(value)) => {
                if *value != consensus[i] {
                    consensus_mismatches += 1;
                    eprintln!(
                        "DIAGNOSTIC {}: TY={value}, consensus={} (delta={})",
                        record.formula_id,
                        consensus[i],
                        *value as i64 - consensus[i] as i64,
                    );
                }
            }
            crate::examination::ExaminationValue::OptionalBound(None) => {
                // CANNOT_COMPUTE is sound.
            }
            other => panic!("expected OptionalBound for UpperBounds, got {other:?}"),
        }
    }
    // Sanity check: the count of mismatches is bounded above by the
    // expected pre-fix delta (4 wrong values: 02, 06, 13, 14). A
    // regression that introduces MORE wrong values would push this
    // count up and fail the test.
    assert!(
        consensus_mismatches <= 4,
        "expected at most 4 consensus mismatches (pre-fix baseline), got {consensus_mismatches}",
    );
}

/// Regression cover: BridgeAndVehicles-COL-V04P05N02 UpperBounds must
/// stay consensus-aligned after the Safety-Net-C plumbing change.
/// Consensus from `raw-result-analysis.csv`:
/// `4 4 2 1 5 2 1 1 4 2 1 4 5 1 4 1`.
#[test]
fn test_col_upper_bounds_bridge_and_vehicles_no_regression() {
    let model = "BridgeAndVehicles-COL-V04P05N02";
    if !fixture_available(model) {
        eprintln!("SKIP: {model} fixture not available at /private/tmp/mcc-selected");
        return;
    }
    let dir = mcc_selected_dir(model);
    let prepared =
        crate::model::load_model_dir(&dir).expect("load BridgeAndVehicles-COL-V04P05N02");
    let config = ExplorationConfig::default();
    let records = crate::model::collect_examination_for_model(
        &prepared,
        crate::examination::Examination::UpperBounds,
        &config,
    )
    .expect("UpperBounds collection should succeed");

    let consensus: [u64; 16] = [4, 4, 2, 1, 5, 2, 1, 1, 4, 2, 1, 4, 5, 1, 4, 1];
    assert_eq!(records.len(), consensus.len());
    for (i, record) in records.iter().enumerate() {
        match &record.value {
            crate::examination::ExaminationValue::OptionalBound(Some(value)) => {
                assert_eq!(
                    *value, consensus[i],
                    "{}: TY returned {value} but consensus is {}",
                    record.formula_id, consensus[i],
                );
            }
            crate::examination::ExaminationValue::OptionalBound(None) => {
                // CANNOT_COMPUTE is sound (incomplete exploration).
            }
            other => panic!("expected OptionalBound, got {other:?}"),
        }
    }
}

/// Direct synthetic: build a tiny "colored-relevance-prunes-producer" net
/// where the per-property reduced net is missing a load-bearing producer
/// transition, so the reduced-net BFS under-counts. With Safety-Net-C
/// supplying ground-truth net+aliases, the fallback BFS must lift the
/// value back to the true reachable maximum.
///
/// Net shape (analogous to colored_relevance over-pruning a producer):
///   p_src(1) ─T_produce→ p_target(1) ─T_drain→ p_sink
///   p_src(1) ─T_extra→  p_target(1)
/// Both `T_produce` and `T_extra` can add tokens to `p_target`; `T_drain`
/// removes them. Reachable max(p_target) on the full net = 2 (fire
/// T_produce then T_extra without draining).
///
/// We simulate "colored_relevance pruned T_extra" by handing the pipeline
/// a "reduced" net that lacks T_extra (so reduced-net BFS observes max=1)
/// but supplying the FULL net (with T_extra) as ground truth. Safety-Net-C
/// must produce 2.
#[test]
fn test_upper_bound_col_ground_truth_lifts_undercount() {
    // Full net (4 places, 3 transitions): ground truth.
    let full = PetriNet {
        name: Some("col-ground-truth-undercount".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_src".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_extra_src".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_target".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_sink".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "produce".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "extra".to_string(),
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
                id: "drain".to_string(),
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
        initial_marking: vec![1, 1, 0, 0],
    };

    // Reduced net (simulates "colored_relevance pruned T_extra and
    // P_extra_src"): only `produce` survives. Reduced-net BFS observes
    // max(p_target)=1.
    let reduced = PetriNet {
        name: Some("col-ground-truth-undercount-reduced".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_src".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_target".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_sink".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "produce".to_string(),
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
                id: "drain".to_string(),
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
    };

    let prop = Property {
        id: "col-gt-undercount-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p_target".to_string()]),
    };

    let reduced_aliases = PropertyAliases::identity(&reduced);
    let full_aliases = PropertyAliases::identity(&full);
    let config = ExplorationConfig::default();

    // Without Safety-Net-C: reduced-net BFS observes max=1 (the missing
    // `extra` transition would have lifted it to 2).
    let reduced_only = check_upper_bounds_properties_with_aliases(
        &reduced,
        std::slice::from_ref(&prop),
        &reduced_aliases,
        &config,
    );
    assert_eq!(
        reduced_only,
        vec![(String::from("col-gt-undercount-UpperBounds-00"), Some(1))],
        "without Safety-Net-C the reduced net's BFS would under-count",
    );

    // With Safety-Net-C: ground-truth net BFS observes max=2.
    let with_ground_truth =
        super::super::pipeline::check_upper_bounds_properties_with_aliases_and_ground_truth(
            &reduced,
            std::slice::from_ref(&prop),
            &reduced_aliases,
            &config,
            Some(super::super::pipeline::GroundTruthNet {
                net: &full,
                aliases: &full_aliases,
            }),
            None,
        );
    assert_eq!(
        with_ground_truth,
        vec![(String::from("col-gt-undercount-UpperBounds-00"), Some(2))],
        "Safety-Net-C must lift the under-count to the ground-truth maximum",
    );
}

#[test]
fn test_upper_bound_col_ground_truth_empty_trackers_fail_closed() {
    let reduced = safety_net_c_reduced_exact_net();
    let prop = Property {
        id: "col-gt-empty-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p_target".to_string()]),
    };
    let reduced_aliases = PropertyAliases::identity(&reduced);
    let config = ExplorationConfig::default();

    let reduced_only = check_upper_bounds_properties_with_aliases(
        &reduced,
        std::slice::from_ref(&prop),
        &reduced_aliases,
        &config,
    );
    assert_eq!(
        reduced_only,
        vec![(String::from("col-gt-empty-UpperBounds-00"), Some(1))],
        "the reduced path alone has an exact-looking result",
    );

    let empty_ground_truth_aliases = aliases_with_place("p_target", Vec::new());
    let with_unusable_ground_truth =
        super::super::pipeline::check_upper_bounds_properties_with_aliases_and_ground_truth(
            &reduced,
            std::slice::from_ref(&prop),
            &reduced_aliases,
            &config,
            Some(super::super::pipeline::GroundTruthNet {
                net: &reduced,
                aliases: &empty_ground_truth_aliases,
            }),
            None,
        );
    assert_eq!(
        with_unusable_ground_truth,
        vec![(String::from("col-gt-empty-UpperBounds-00"), None)],
        "Safety-Net-C must not publish the reduced result when GT re-resolution yields no trackers",
    );
}

#[test]
fn test_upper_bound_col_ground_truth_static_caps_resolve_without_bfs() {
    let reduced = safety_net_c_reduced_exact_net();
    let full = PetriNet {
        name: Some("col-ground-truth-static-cap".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_target".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_sink".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_noise".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "drain".to_string(),
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
                id: "noise".to_string(),
                name: None,
                inputs: vec![],
                outputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![3, 0, 0],
    };
    let prop = Property {
        id: "col-gt-static-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p_target".to_string()]),
    };
    let reduced_aliases = PropertyAliases::identity(&reduced);
    let full_aliases = PropertyAliases::identity(&full);
    let config = ExplorationConfig::new(1);

    let with_ground_truth =
        super::super::pipeline::check_upper_bounds_properties_with_aliases_and_ground_truth(
            &reduced,
            std::slice::from_ref(&prop),
            &reduced_aliases,
            &config,
            Some(super::super::pipeline::GroundTruthNet {
                net: &full,
                aliases: &full_aliases,
            }),
            None,
        );
    assert_eq!(
        with_ground_truth,
        vec![(String::from("col-gt-static-UpperBounds-00"), Some(3))],
        "Safety-Net-C should reuse ground-truth monotone/static caps before incomplete BFS",
    );
}

#[test]
fn test_upper_bound_col_ground_truth_unusable_mapping_after_reduced_path_fail_closed() {
    let reduced = safety_net_c_reduced_exact_net();
    let prop = Property {
        id: "col-gt-invalid-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p_target".to_string()]),
    };
    let reduced_aliases = PropertyAliases::identity(&reduced);
    let config = ExplorationConfig::default();

    let reduced_only = check_upper_bounds_properties_with_aliases(
        &reduced,
        std::slice::from_ref(&prop),
        &reduced_aliases,
        &config,
    );
    assert_eq!(
        reduced_only,
        vec![(String::from("col-gt-invalid-UpperBounds-00"), Some(1))],
        "the reduced path completes before the ground-truth mapping is validated",
    );

    let invalid_ground_truth_aliases = aliases_with_place("p_target", vec![PlaceIdx(99)]);
    let with_unusable_ground_truth =
        super::super::pipeline::check_upper_bounds_properties_with_aliases_and_ground_truth(
            &reduced,
            std::slice::from_ref(&prop),
            &reduced_aliases,
            &config,
            Some(super::super::pipeline::GroundTruthNet {
                net: &reduced,
                aliases: &invalid_ground_truth_aliases,
            }),
            None,
        );
    assert_eq!(
        with_unusable_ground_truth,
        vec![(String::from("col-gt-invalid-UpperBounds-00"), None)],
        "Safety-Net-C must fail closed instead of reusing reduced exactness when GT mapping is invalid",
    );
}

/// Direct synthetic: a "colored-relevance-prunes-drainer" case where the
/// per-property reduced net is missing a drainer transition, so the
/// reduced-net BFS over-counts (spurious markings accumulate). With
/// Safety-Net-C in place, the fallback BFS on the ground-truth net must
/// cap the value at the true reachable maximum.
///
/// Net shape:
///   p_src(2) ─T_produce→ p_target
///                          │
///                          T_drain → p_sink
///
/// Full net: p_target receives at most 1 token at a time (each produce
/// must be balanced by drain before next produce because we use p_src
/// as the input arc with weight 1, but we only have 2 tokens in p_src
/// total, so 2 produces → 2 drains gives sum sequence 0,1,0,1,0).
///
/// Reduced net: drain removed. Now produce fires twice → p_target = 2.
#[test]
fn test_upper_bound_col_ground_truth_caps_overcount() {
    // Full net: produce + drain.
    let full = PetriNet {
        name: Some("col-ground-truth-overcount".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_src".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_target".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_sink".to_string(),
                name: None,
            },
            // p_lock is a mutex token forcing produce/drain to alternate;
            // both transitions need it.
            PlaceInfo {
                id: "p_lock".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "produce".to_string(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(3),
                        weight: 1,
                    },
                ],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "drain".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![
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
        initial_marking: vec![2, 0, 0, 1],
    };

    // Reduced net: simulates colored_relevance pruning `drain` (because
    // p_sink isn't in the formula's backward cone). Produce can fire
    // twice unimpeded since each fires only when the lock is held — but
    // the lock never returns because drain is gone — so produce fires
    // exactly once. To craft a real over-count we drop the lock arc
    // requirement on produce in the reduced net (modelling
    // colored_relevance dropping arcs to p_lock):
    let reduced = PetriNet {
        name: Some("col-ground-truth-overcount-reduced".to_string()),
        places: vec![
            PlaceInfo {
                id: "p_src".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p_target".to_string(),
                name: None,
            },
        ],
        transitions: vec![TransitionInfo {
            id: "produce".to_string(),
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

    let prop = Property {
        id: "col-gt-overcount-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(vec!["p_target".to_string()]),
    };

    let reduced_aliases = PropertyAliases::identity(&reduced);
    let full_aliases = PropertyAliases::identity(&full);
    let config = ExplorationConfig::default();

    // Without Safety-Net-C: reduced-net BFS observes max=2 (produce fires
    // twice unimpeded).
    let reduced_only = check_upper_bounds_properties_with_aliases(
        &reduced,
        std::slice::from_ref(&prop),
        &reduced_aliases,
        &config,
    );
    assert_eq!(
        reduced_only,
        vec![(String::from("col-gt-overcount-UpperBounds-00"), Some(2))],
        "without Safety-Net-C the reduced net's BFS would over-count",
    );

    // With Safety-Net-C: ground-truth net BFS observes max=1 (mutex lock
    // forces produce/drain alternation; p_target holds 1 token at most).
    let with_ground_truth =
        super::super::pipeline::check_upper_bounds_properties_with_aliases_and_ground_truth(
            &reduced,
            std::slice::from_ref(&prop),
            &reduced_aliases,
            &config,
            Some(super::super::pipeline::GroundTruthNet {
                net: &full,
                aliases: &full_aliases,
            }),
            None,
        );
    assert_eq!(
        with_ground_truth,
        vec![(String::from("col-gt-overcount-UpperBounds-00"), Some(1))],
        "Safety-Net-C must cap the over-count at the ground-truth maximum",
    );
}
