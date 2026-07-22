// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::examinations::stable_marking::StableMarkingObserver;
use crate::explorer::explore;
use crate::output::Verdict;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

use super::super::stable_marking_verdict;
use super::fixtures::{counting_net, cyclic_safe_net, default_config, immediate_deadlock_net};

#[test]
fn test_stable_marking_immediate_deadlock_all_stable() {
    let net = immediate_deadlock_net();
    let config = default_config();
    let mut observer = StableMarkingObserver::new(&net.initial_marking);
    let result = explore(&net, &config, &mut observer);

    assert!(!observer.all_unstable());
    assert!(result.completed);
}

#[test]
fn test_stable_marking_cyclic_net_all_unstable() {
    let net = cyclic_safe_net();
    let config = default_config();
    let mut observer = StableMarkingObserver::new(&net.initial_marking);
    let result = explore(&net, &config, &mut observer);

    assert!(observer.all_unstable());
    assert!(result.stopped_by_observer);
}

#[test]
fn test_stable_marking_counting_net_all_unstable() {
    let net = counting_net();
    let config = default_config();
    let mut observer = StableMarkingObserver::new(&net.initial_marking);
    let result = explore(&net, &config, &mut observer);

    assert!(observer.all_unstable());
    assert!(result.stopped_by_observer);
}

#[test]
fn test_stable_marking_with_isolated_stable_place() {
    let net = PetriNet {
        name: Some("isolated-stable".into()),
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
        ],
        transitions: vec![TransitionInfo {
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
        }],
        initial_marking: vec![1, 0, 5],
    };
    let config = default_config();
    let mut observer = StableMarkingObserver::new(&net.initial_marking);
    let result = explore(&net, &config, &mut observer);

    assert!(!observer.all_unstable());
    assert!(result.completed);
}

/// Verify the isolated-place shortcut works through `stable_marking_verdict`.
/// The net from `test_stable_marking_with_isolated_stable_place` has P2 isolated
/// with initial marking 5, which is structurally stable.
#[test]
fn test_stable_marking_verdict_with_isolated_place_returns_true() {
    let net = PetriNet {
        name: Some("isolated-stable".into()),
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
        ],
        transitions: vec![TransitionInfo {
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
        }],
        initial_marking: vec![1, 0, 5],
    };
    let config = default_config();
    let verdict = stable_marking_verdict(&net, &config, &[]);
    assert_eq!(
        verdict,
        Verdict::True,
        "StableMarking should be TRUE when an isolated place exists"
    );
}

/// Net with ONLY pre-agglomeration reduction (no constant or isolated places).
///
/// P0(1) → T0 → P1(0) → T1 → P2(0)
///
/// P1 is pre-agglomeratable (sole producer T0, initial=0, consumer T1 reads 1).
/// No constant places (P0 net effect -1, P1 net effect +1/-1, P2 net effect +1).
/// No isolated places (all connected to alive transitions).
///
/// In the original net, every place changes marking:
///   P0: 1→0, P1: 0→1→0, P2: 0→1
/// So StableMarking should be FALSE (no place has m(p) = m₀(p) for all reachable m).
///
/// Regression test: previously, `places_removed() > 0` short-circuited to TRUE
/// because the agglomerated place was counted as structurally stable, but
/// agglomerated places are NOT stable (their marking changes when transitions fire).
#[test]
fn test_stable_marking_agglomeration_only_not_false_positive() {
    let net = PetriNet {
        name: Some("agg-only-all-unstable".into()),
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

    let config = default_config();

    // Verify pre-agglomeration is detected: P1 has sole producer T0,
    // initial=0, consumer T1 reads 1, and T0's input (P0) does not
    // overlap with T1's output (P2). Berthelot condition 6 satisfied.
    let report = crate::reduction::analyze(&net);
    assert!(
        !report.pre_agglomerations.is_empty(),
        "net should have pre-agglomeration candidates"
    );
    assert!(
        report.constant_places.is_empty(),
        "net should have no constant places"
    );
    assert!(
        report.isolated_places.is_empty(),
        "net should have no isolated places"
    );

    // The verdict must be FALSE: all places change marking.
    // The old code returned TRUE here (wrong) because it checked
    // places_removed() which included agglomerated places.
    let verdict = stable_marking_verdict(&net, &config, &[]);
    assert_eq!(
        verdict,
        Verdict::False,
        "StableMarking should be FALSE when only agglomerated places are removed"
    );
}

/// Regression test for #1442: multi-round reduction reveals stable places
/// that single-round `analyze()` misses.
///
/// P_target(5) is connected to T_dead (dead, delta +1) and T_selfloop
/// (self-loop transition, delta 0). Single-round `analyze()` finds P_target
/// neither constant nor cascade-isolated. Multi-round `reduce_iterative`
/// removes T_dead (dead) and T_selfloop (Rule J) in round 1, then finds
/// P_target isolated in round 2. The fix checks the composed report and
/// returns TRUE; without it, BFS on the reduced net returns FALSE (wrong).
#[test]
fn test_stable_marking_multi_round_reduction_reveals_stable_place() {
    let net = PetriNet {
        name: Some("cascade-stable-multi-round".into()),
        places: vec![
            PlaceInfo {
                id: "P_target".into(),
                name: None,
            },
            PlaceInfo {
                id: "P_feeder".into(),
                name: None,
            },
            PlaceInfo {
                id: "P_c".into(),
                name: None,
            },
            PlaceInfo {
                id: "P_a".into(),
                name: None,
            },
            PlaceInfo {
                id: "P_b".into(),
                name: None,
            },
        ],
        transitions: vec![
            // T_dead: needs P_feeder(2) but initial=1, no producer → dead.
            // Output to P_target makes P_target non-constant in original net.
            TransitionInfo {
                id: "T_dead".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 2,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            },
            // T_selfloop: pure self-loop on P_target (Rule J removes it).
            TransitionInfo {
                id: "T_selfloop".into(),
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
            // T_alive: keeps P_feeder non-cascade-isolated after T_dead removal.
            TransitionInfo {
                id: "T_alive".into(),
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
            // T_live: makes P_a and P_b unstable.
            TransitionInfo {
                id: "T_live".into(),
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
        ],
        initial_marking: vec![5, 1, 0, 1, 0],
    };

    // Pre-check must NOT find P_target.
    let report = crate::reduction::analyze(&net);
    assert!(
        report.constant_places.is_empty(),
        "no constant places in original net"
    );
    assert!(
        report.isolated_places.is_empty(),
        "no isolated places in original net"
    );

    // Multi-round reduction must find P_target in composed report.
    let reduced = crate::reduction::reduce_iterative(&net).unwrap();
    assert!(
        !reduced.report.constant_places.is_empty() || !reduced.report.isolated_places.is_empty(),
        "composed report should have stable removed places from multi-round reduction"
    );

    // Verdict must be TRUE: P_target(5) is stable.
    let config = default_config();
    let verdict = stable_marking_verdict(&net, &config, &[]);
    assert_eq!(
        verdict,
        Verdict::True,
        "StableMarking should be TRUE: P_target is stable (multi-round reduction reveals it)"
    );
}

/// LP state-equation pinning catches a stable place that every structural test
/// misses. p0(1) and p2(1) each carry a self-loop (t_live0, t_live2) and feed a
/// transition t_dead with inputs p0(1) and p2(2). t_dead can never fire (p2 caps
/// at 1, below its demand of 2), so the only reachable marking is the initial
/// one and StableMarking is TRUE.
///
/// The simple structural tests all miss it:
/// - `find_constant_places`: t_dead's incidence row is nonzero on p0 and p2.
/// - `find_dead_transitions`: t_dead's input places both HAVE producers (the
///   self-loops), so the cheap "no producer + insufficient initial" rule never
///   flags it dead — hence no cascade-isolated places either.
///
/// The state-equation LP does prove it: `M[p2] = 1 - 2*x_dead >= 0` forces
/// `x_dead <= 0.5`, which pins `M[p0] = 1 - x_dead` to exactly 1.
fn lp_pinned_stable_place_net() -> PetriNet {
    PetriNet {
        name: Some("lp-pinned-stable".into()),
        places: vec![
            PlaceInfo {
                id: "p0".into(),
                name: None,
            },
            PlaceInfo {
                id: "p2".into(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "t_live0".into(),
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
                id: "t_live2".into(),
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
            TransitionInfo {
                id: "t_dead".into(),
                name: None,
                inputs: vec![
                    Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                    Arc {
                        place: PlaceIdx(1),
                        weight: 2,
                    },
                ],
                outputs: vec![],
            },
        ],
        initial_marking: vec![1, 1],
    }
}

#[test]
fn test_stable_marking_lp_pinning_returns_true() {
    let net = lp_pinned_stable_place_net();

    // The structural pre-checks must find nothing, so the LP pinning shortcut is
    // the path that decides TRUE.
    let report = crate::reduction::analyze(&net);
    assert!(
        report.constant_places.is_empty(),
        "no zero-incidence-row constant places"
    );
    assert!(
        report.isolated_places.is_empty(),
        "no isolated or cascade-isolated places (t_dead is not detected dead structurally)"
    );

    let config = default_config();
    let verdict = stable_marking_verdict(&net, &config, &[]);
    assert_eq!(
        verdict,
        Verdict::True,
        "StableMarking is TRUE: p0 is pinned to its initial marking by the state equation"
    );
}

/// Counterexample net for StableMarking place-swap canonicalization soundness.
///
/// Places `p0`, `p1` form a symmetric orbit fed by source `s`:
///   m0 = [p0=0, p1=0, s=1]
///   t0: s → p0      t1: s → p1
/// Reachable markings: [0,0,1] → [1,0,0] (fire t0); [0,0,1] → [0,1,0] (fire t1).
/// Every place is unstable on the true reachability graph (s drops 1→0, p0 rises
/// 0→1 on the t0 branch, p1 rises 0→1 on the t1 branch), so the StableMarking
/// truth is FALSE. With ascending orbit canonicalization, [1,0,0] and [0,1,0]
/// both collapse to canonical [0,1,0], so place index 0 only ever shows token 0
/// (its initial value) and the index-keyed observer wrongly reports it stable —
/// flipping the verdict to a false TRUE (−16 MCC).
fn stable_marking_orbit_counterexample_net() -> PetriNet {
    PetriNet {
        name: Some("stable-marking-orbit-counterexample".into()),
        places: vec![
            PlaceInfo {
                id: "p0".into(),
                name: None,
            },
            PlaceInfo {
                id: "p1".into(),
                name: None,
            },
            PlaceInfo {
                id: "s".into(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "t0".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "t1".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(2),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![0, 0, 1],
    }
}

/// Soundness regression: place-swap canonicalization must NOT be applied to
/// StableMarking exploration. The observer tracks per-place stability *by place
/// index*, which orbit canonicalization (sorting token counts within an orbit)
/// permutes — masquerading an unstable place as stable. Post-fix
/// (`canonicalization_is_sound(StableMarking) == false` plus
/// `StableMarkingObserver::canonicalization_safe() == false`) canon is disabled,
/// so the observer sees the true markings and reports every place unstable.
#[test]
fn test_stable_marking_canonicalization_does_not_fabricate_stable_place() {
    use crate::examination::Examination;
    use crate::explorer::symmetry::PetriCanonicalizer;
    use crate::explorer::ExplorationConfig;

    let net = stable_marking_orbit_counterexample_net();

    // Sanity: the {p0,p1} orbit must actually be discovered, else the test
    // would vacuously pass without exercising the canonicalization path.
    assert!(
        !PetriCanonicalizer::build(&net).is_empty(),
        "expected the p0<->p1 place orbit to be discovered",
    );

    let config = ExplorationConfig::new(10_000).with_examination(Some(Examination::StableMarking));
    let mut observer = StableMarkingObserver::new(&net.initial_marking);
    let _ = explore(&net, &config, &mut observer);

    assert!(
        observer.all_unstable(),
        "every place is unstable on the true reachability graph; place-swap \
         canonicalization is unsound for StableMarking because per-place \
         stability is tracked by index and the orbit sort permutes indices",
    );
}

/// End-to-end verdict on the same net through the real dispatcher. The net is
/// NOT stable (no place keeps its initial token count across all reachable
/// markings), so the verdict must be False. Pre-fix the canonicalizing observer
/// fabricated a stable place and returned True — a wrong MCC answer.
#[test]
fn test_stable_marking_orbit_counterexample_verdict_is_false() {
    use crate::examination::Examination;
    use crate::explorer::ExplorationConfig;

    let net = stable_marking_orbit_counterexample_net();
    let config = ExplorationConfig::new(10_000).with_examination(Some(Examination::StableMarking));
    assert_eq!(
        stable_marking_verdict(&net, &config, &[]),
        Verdict::False,
        "no place is stable across the reachable state space",
    );
}

/// Unit gate: the per-observer fail-closed hook must refuse canonicalization for
/// StableMarking regardless of the examination-level classification.
#[test]
fn test_stable_marking_observer_refuses_canonicalization() {
    use crate::explorer::ExplorationObserver;

    let observer = StableMarkingObserver::new(&[0, 0, 1]);
    assert!(
        !observer.canonicalization_safe(),
        "StableMarkingObserver tracks stability per place index; it must never \
         be paired with place-swap canonicalization",
    );
}
