// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Classifier unit tests + randomized differential soundness tests for the
//! exact net-class liveness certificates.
//!
//! The differential oracle (`brute_force_live`) is an INDEPENDENT
//! implementation of MCC L4-liveness: exhaustive BFS of the reachability
//! graph, then per-transition backward reachability of the enabling states
//! (`AG EF enabled(t)` ⟺ every reachable state can reach an enabling
//! state). It shares no code with the certificates, the mu-calculus
//! engine, or the SCC engine, so an agreement here is a genuine
//! cross-implementation check. A wrong Liveness verdict is catastrophic in
//! MCC scoring — these tests are the soundness pillar for the certificate
//! chain.

use super::*;
use crate::petri_net::{Arc, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::structural::structural_live;

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

// ---------------------------------------------------------------------------
// Independent brute-force L4-liveness oracle
// ---------------------------------------------------------------------------

/// Exhaustive L4-liveness: BFS the full reachability graph (up to
/// `max_states`), then check for every transition that EVERY reachable state
/// can reach a state enabling it (backward closure of the enabling set
/// covers all states). Returns `None` when the state space exceeds the cap
/// (unbounded / too large — skip the comparison).
fn brute_force_live(net: &PetriNet, max_states: usize) -> Option<bool> {
    use std::collections::HashMap;

    let nt = net.num_transitions();
    let mut index: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut states: Vec<Vec<u64>> = Vec::new();
    let mut edges: Vec<Vec<usize>> = Vec::new();
    index.insert(net.initial_marking.clone(), 0);
    states.push(net.initial_marking.clone());
    edges.push(Vec::new());

    let mut frontier = vec![0usize];
    while let Some(s) = frontier.pop() {
        let current = states[s].clone();
        for t in 0..nt {
            let tidx = TransitionIdx(t as u32);
            if !net.is_enabled(&current, tidx) {
                continue;
            }
            let succ = net.fire(&current, tidx).expect("fire (test)");
            let id = match index.get(&succ) {
                Some(&id) => id,
                None => {
                    if states.len() >= max_states {
                        return None;
                    }
                    let id = states.len();
                    index.insert(succ.clone(), id);
                    states.push(succ);
                    edges.push(Vec::new());
                    frontier.push(id);
                    id
                }
            };
            edges[s].push(id);
        }
    }

    let n = states.len();
    let mut reverse: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (s, succs) in edges.iter().enumerate() {
        for &d in succs {
            reverse[d].push(s);
        }
    }

    for t in 0..nt {
        let tidx = TransitionIdx(t as u32);
        let mut good = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        for (s, state) in states.iter().enumerate() {
            if net.is_enabled(state, tidx) {
                good[s] = true;
                stack.push(s);
            }
        }
        while let Some(s) = stack.pop() {
            for &p in &reverse[s] {
                if !good[p] {
                    good[p] = true;
                    stack.push(p);
                }
            }
        }
        if !good.iter().all(|&g| g) {
            return Some(false);
        }
    }
    Some(true)
}

// ---------------------------------------------------------------------------
// Classifier unit tests
// ---------------------------------------------------------------------------

#[test]
fn test_classify_state_machine_cycle() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    };
    let class = classify(&net);
    assert!(class.ordinary);
    assert!(class.free_choice);
    assert!(class.marked_graph, "a simple cycle is also a marked graph");
    assert!(class.state_machine);
    assert!(class.all_places_incident);
    assert!(!class.has_source_transition);
}

#[test]
fn test_classify_fork_join_is_marked_graph_not_state_machine() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("pa"), place("pb")],
        transitions: vec![
            trans("fork", vec![arc(0, 1)], vec![arc(1, 1), arc(2, 1)]),
            trans("join", vec![arc(1, 1), arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };
    let class = classify(&net);
    assert!(class.ordinary);
    assert!(
        class.free_choice,
        "join's input places have a single consumer each"
    );
    assert!(class.marked_graph);
    assert!(!class.state_machine);
}

#[test]
fn test_classify_choice_is_state_machine_not_marked_graph() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t2", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    };
    let class = classify(&net);
    assert!(class.state_machine);
    assert!(!class.marked_graph, "p0 has two consumers");
    assert!(class.free_choice);
}

#[test]
fn test_classify_non_free_choice_shared_place() {
    // p0 feeds both t0 (single-input) and t1 (two-input): asymmetric choice,
    // NOT free choice.
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(0, 1), arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 1, 0],
    };
    let class = classify(&net);
    assert!(class.ordinary);
    assert!(!class.free_choice);
    assert!(!class.state_machine);
    assert!(!class.marked_graph);
}

#[test]
fn test_classify_weighted_arc_not_ordinary() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0")],
        transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(0, 1)])],
        initial_marking: vec![2],
    };
    let class = classify(&net);
    assert!(!class.ordinary);
    assert!(!class.free_choice);
    assert!(!class.marked_graph);
    assert!(!class.state_machine);
}

#[test]
fn test_classify_parallel_unit_arcs_not_ordinary() {
    // Two unit arcs from the same place: effective weight 2 under the
    // firing semantics — must NOT be classified ordinary (and hence in no
    // class), otherwise the unit-weight theorems would be misapplied.
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1), arc(0, 1)], vec![arc(1, 1)])],
        initial_marking: vec![2, 0],
    };
    let class = classify(&net);
    assert!(!class.ordinary);
    assert!(!class.free_choice);
    assert!(!class.marked_graph);
    assert!(!class.state_machine);
    assert_eq!(
        structural_live(&net, None),
        None,
        "no certificate may fire on a non-ordinary net"
    );
}

#[test]
fn test_classify_source_transition_and_isolated_place() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p_isolated")],
        transitions: vec![trans("t_src", vec![], vec![arc(0, 1)])],
        initial_marking: vec![0, 0],
    };
    let class = classify(&net);
    assert!(class.has_source_transition);
    assert!(!class.all_places_incident);
    assert!(!class.state_machine, "a source transition has no input");
}

// ---------------------------------------------------------------------------
// Hand-built certificate checks against the independent oracle
// ---------------------------------------------------------------------------

#[test]
fn test_certificates_match_oracle_on_hand_built_nets() {
    // (net, name) pairs spanning: SM live/dead, MG live/dead (incl. the
    // fork/join and unmarked-circuit shapes), FC covered/uncovered, sink
    // transitions, jointly-dead joins.
    let nets: Vec<(PetriNet, &str)> = vec![
        (
            PetriNet {
                name: None,
                places: vec![place("p0"), place("p1")],
                transitions: vec![
                    trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                    trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
                ],
                initial_marking: vec![1, 0],
            },
            "sm-cycle-live",
        ),
        (
            PetriNet {
                name: None,
                places: vec![place("p0"), place("p1")],
                transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
                initial_marking: vec![1, 0],
            },
            "sm-chain-dead",
        ),
        (
            PetriNet {
                name: None,
                places: vec![place("p0"), place("pa"), place("pb")],
                transitions: vec![
                    trans("fork", vec![arc(0, 1)], vec![arc(1, 1), arc(2, 1)]),
                    trans("join", vec![arc(1, 1), arc(2, 1)], vec![arc(0, 1)]),
                ],
                initial_marking: vec![1, 0, 0],
            },
            "mg-fork-join-live",
        ),
        (
            PetriNet {
                name: None,
                places: vec![place("p0"), place("pa"), place("pb")],
                transitions: vec![
                    trans("fork", vec![arc(0, 1)], vec![arc(1, 1), arc(2, 1)]),
                    trans("join", vec![arc(1, 1), arc(2, 1)], vec![arc(0, 1)]),
                ],
                initial_marking: vec![0, 0, 0],
            },
            "mg-fork-join-unmarked-circuit",
        ),
        (
            PetriNet {
                name: None,
                places: vec![place("p0"), place("pa"), place("pb")],
                transitions: vec![
                    trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                    trans("t1", vec![arc(0, 1)], vec![arc(2, 1)]),
                    trans("t_join", vec![arc(1, 1), arc(2, 1)], vec![arc(0, 1)]),
                ],
                initial_marking: vec![1, 0, 0],
            },
            "fc-choice-join-dead",
        ),
        (
            PetriNet {
                name: None,
                places: vec![place("p0"), place("p1")],
                transitions: vec![
                    trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                    trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
                    trans("t_sink", vec![arc(1, 1)], vec![]),
                ],
                initial_marking: vec![1, 0],
            },
            "fc-sink-drain-dead",
        ),
    ];

    let mut decided = 0usize;
    for (net, name) in &nets {
        let certificate = structural_live(net, None);
        let oracle = brute_force_live(net, 100_000).expect("hand-built nets are tiny and bounded");
        if let Some(verdict) = certificate {
            decided += 1;
            assert_eq!(
                verdict, oracle,
                "certificate disagrees with the exhaustive oracle on {name}"
            );
        }
    }
    assert_eq!(
        decided,
        nets.len(),
        "every hand-built net is in an exact class — all must be decided"
    );
}

// ---------------------------------------------------------------------------
// Randomized differential soundness tests (deterministic seeds)
// ---------------------------------------------------------------------------

/// xorshift64* — deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_marking(rng: &mut Rng, np: usize, empty_bias: usize) -> Vec<u64> {
    (0..np)
        .map(|_| u64::from(rng.below(empty_bias) == 0))
        .collect()
}

/// Random state machine: every transition has exactly one input and one
/// output place; every place is the input of at least one transition
/// (incidence by construction).
fn random_state_machine(rng: &mut Rng) -> PetriNet {
    let np = 2 + rng.below(4);
    let extra = rng.below(5);
    let mut transitions = Vec::new();
    for p in 0..np {
        transitions.push(trans(
            &format!("t{p}"),
            vec![arc(p as u32, 1)],
            vec![arc(rng.below(np) as u32, 1)],
        ));
    }
    for i in 0..extra {
        transitions.push(trans(
            &format!("x{i}"),
            vec![arc(rng.below(np) as u32, 1)],
            vec![arc(rng.below(np) as u32, 1)],
        ));
    }
    PetriNet {
        name: None,
        places: (0..np).map(|p| place(&format!("p{p}"))).collect(),
        transitions,
        initial_marking: random_marking(rng, np, 3),
    }
}

/// Random marked graph: every place picks exactly one producer and one
/// consumer transition.
fn random_marked_graph(rng: &mut Rng) -> PetriNet {
    let nt = 1 + rng.below(4);
    let np = 1 + rng.below(6);
    let mut transitions: Vec<TransitionInfo> = (0..nt)
        .map(|t| trans(&format!("t{t}"), vec![], vec![]))
        .collect();
    for p in 0..np {
        let producer = rng.below(nt);
        let consumer = rng.below(nt);
        transitions[producer].outputs.push(arc(p as u32, 1));
        transitions[consumer].inputs.push(arc(p as u32, 1));
    }
    PetriNet {
        name: None,
        places: (0..np).map(|p| place(&format!("p{p}"))).collect(),
        transitions,
        initial_marking: random_marking(rng, np, 3),
    }
}

/// Random simple free-choice net: choice clusters (one shared place feeding
/// single-input transitions) and join transitions (private multi-place
/// presets), with arbitrary random postsets. Every place is some
/// transition's input (incidence), every transition has a nonempty preset
/// (no sources), all weights 1 (ordinary).
fn random_free_choice(rng: &mut Rng) -> PetriNet {
    let clusters = 1 + rng.below(4);
    let mut places: Vec<PlaceInfo> = Vec::new();
    // (inputs) per transition; outputs assigned after all places exist.
    let mut presets: Vec<Vec<u32>> = Vec::new();
    for _ in 0..clusters {
        if rng.below(2) == 0 {
            // Choice cluster: one place, 2-3 single-input consumers.
            let p = places.len() as u32;
            places.push(place(&format!("p{p}")));
            for _ in 0..(2 + rng.below(2)) {
                presets.push(vec![p]);
            }
        } else {
            // Join transition: 2-3 private input places.
            let mut preset = Vec::new();
            for _ in 0..(2 + rng.below(2)) {
                let p = places.len() as u32;
                places.push(place(&format!("p{p}")));
                preset.push(p);
            }
            presets.push(preset);
        }
    }
    let np = places.len();
    let transitions: Vec<TransitionInfo> = presets
        .iter()
        .enumerate()
        .map(|(i, preset)| {
            let mut outputs = Vec::new();
            // ~1 in 5 transitions is a SINK (empty postset): Commoner's
            // theorem must stay exact in the presence of token-draining
            // transitions — this is the corner the differential hammers.
            if rng.below(5) != 0 {
                let mut taken = vec![false; np];
                for _ in 0..=rng.below(2) {
                    let p = rng.below(np);
                    if !taken[p] {
                        taken[p] = true;
                        outputs.push(arc(p as u32, 1));
                    }
                }
            }
            trans(
                &format!("t{i}"),
                preset.iter().map(|&p| arc(p, 1)).collect(),
                outputs,
            )
        })
        .collect();
    PetriNet {
        name: None,
        places,
        transitions,
        initial_marking: random_marking(rng, np, 2),
    }
}

/// Differential harness: on every generated net where the certificate chain
/// fires AND the brute-force oracle completes, the verdicts must agree.
/// Returns (fired_and_compared, generated).
fn differential_run(
    rng: &mut Rng,
    iterations: usize,
    generate: impl Fn(&mut Rng) -> PetriNet,
) -> (usize, usize) {
    let mut compared = 0usize;
    for _ in 0..iterations {
        let net = generate(rng);
        let certificate = structural_live(&net, None);
        let Some(verdict) = certificate else {
            continue;
        };
        let Some(oracle) = brute_force_live(&net, 20_000) else {
            continue; // unbounded / too large — TRUE certificates remain
                      // theorem-backed; bounded cases dominate the sample.
        };
        assert_eq!(
            verdict,
            oracle,
            "certificate disagrees with the exhaustive oracle on {:?} \
             (places={}, transitions={:?}, marking={:?})",
            net.name,
            net.num_places(),
            net.transitions
                .iter()
                .map(|t| {
                    (
                        t.inputs.iter().map(|a| a.place.0).collect::<Vec<_>>(),
                        t.outputs.iter().map(|a| a.place.0).collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>(),
            net.initial_marking,
        );
        compared += 1;
    }
    (compared, iterations)
}

#[test]
fn test_random_state_machines_match_oracle() {
    let mut rng = Rng(0x5EED_0001);
    let (compared, total) = differential_run(&mut rng, 300, random_state_machine);
    assert!(
        compared >= total / 2,
        "differential coverage too low: {compared}/{total}"
    );
}

#[test]
fn test_random_marked_graphs_match_oracle() {
    let mut rng = Rng(0x5EED_0002);
    let (compared, _) = differential_run(&mut rng, 300, random_marked_graph);
    assert!(
        compared >= 50,
        "differential coverage too low: {compared}/300"
    );
}

#[test]
fn test_random_free_choice_nets_match_oracle() {
    let mut rng = Rng(0x5EED_0003);
    let (compared, _) = differential_run(&mut rng, 600, random_free_choice);
    assert!(
        compared >= 100,
        "differential coverage too low: {compared}/600"
    );
}

/// The generators must actually produce members of their advertised class —
/// otherwise the differential tests would be vacuous.
#[test]
fn test_generators_hit_their_classes() {
    let mut rng = Rng(0x5EED_0004);
    for _ in 0..50 {
        let sm = random_state_machine(&mut rng);
        let class = classify(&sm);
        assert!(class.state_machine && class.all_places_incident);

        let mg = random_marked_graph(&mut rng);
        assert!(classify(&mg).marked_graph);

        let fc = random_free_choice(&mut rng);
        let class = classify(&fc);
        assert!(
            class.free_choice && class.all_places_incident && !class.has_source_transition,
            "free-choice generator must satisfy every Commoner gate"
        );
    }
}
