// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::examination::Examination;
use crate::examinations::deadlock::DeadlockObserver;
use crate::examinations::one_safe::OneSafeObserver;
use crate::examinations::state_space::StateSpaceObserver;
use crate::explorer::{explore, explore_observer, ExplorationConfig};
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::stubborn::PorStrategy;

use super::fixtures::{
    counting_net, cyclic_safe_net, default_config, immediate_deadlock_net, linear_deadlock_net,
    not_safe_net,
};

/// Test helper: a `BigUint` from a small integer (the `StateSpaceStats` count
/// fields are arbitrary-precision; these fixtures compare against small exact
/// values).
fn big(n: u64) -> tla_bignum::BigUint {
    tla_bignum::BigUint::from(n)
}

#[test]
fn test_deadlock_observer_finds_deadlock_in_linear_net() {
    let net = linear_deadlock_net();
    let config = default_config();
    let mut observer = DeadlockObserver::new();
    let result = explore(&net, &config, &mut observer);

    assert!(observer.found_deadlock());
    assert!(result.stopped_by_observer);
}

#[test]
fn test_deadlock_observer_no_deadlock_in_cyclic_net() {
    let net = cyclic_safe_net();
    let config = default_config();
    let mut observer = DeadlockObserver::new();
    let result = explore(&net, &config, &mut observer);

    assert!(!observer.found_deadlock());
    assert!(result.completed);
    assert!(!result.stopped_by_observer);
}

#[test]
fn test_deadlock_observer_immediate_deadlock() {
    let net = immediate_deadlock_net();
    let config = default_config();
    let mut observer = DeadlockObserver::new();
    let result = explore(&net, &config, &mut observer);

    assert!(observer.found_deadlock());
    assert!(result.stopped_by_observer);
    assert_eq!(result.states_visited, 1);
}

#[test]
fn test_one_safe_observer_safe_cyclic_net() {
    let net = cyclic_safe_net();
    let config = default_config();
    let mut observer = OneSafeObserver::new();
    let result = explore(&net, &config, &mut observer);

    assert!(observer.is_safe());
    assert!(result.completed);
    assert_eq!(result.states_visited, 2);
}

#[test]
fn test_one_safe_observer_detects_unsafe_net() {
    let net = not_safe_net();
    let config = default_config();
    let mut observer = OneSafeObserver::new();
    let result = explore(&net, &config, &mut observer);

    assert!(!observer.is_safe());
    assert!(result.stopped_by_observer);
}

#[test]
fn test_one_safe_observer_initial_marking_safe() {
    let net = immediate_deadlock_net();
    let config = default_config();
    let mut observer = OneSafeObserver::new();
    let result = explore(&net, &config, &mut observer);

    assert!(observer.is_safe());
    assert!(result.completed);
}

#[test]
fn test_one_safe_observer_unsafe_at_initial() {
    let net = PetriNet {
        name: None,
        places: vec![PlaceInfo {
            id: "P0".into(),
            name: None,
        }],
        transitions: vec![],
        initial_marking: vec![2],
    };
    let config = default_config();
    let mut observer = OneSafeObserver::new();
    let result = explore(&net, &config, &mut observer);

    assert!(!observer.is_safe());
    assert!(result.stopped_by_observer);
    assert_eq!(result.states_visited, 1);
}

fn delayed_source_place_overflow_net() -> PetriNet {
    let mut places: Vec<PlaceInfo> = (0..=17)
        .map(|idx| PlaceInfo {
            id: format!("p{idx}"),
            name: None,
        })
        .collect();
    let accumulator = PlaceIdx(18);
    places.push(PlaceInfo {
        id: "p_acc".into(),
        name: None,
    });

    let transitions: Vec<TransitionInfo> = (0..=16)
        .map(|idx| {
            let mut outputs = vec![Arc {
                place: PlaceIdx((idx + 1) as u32),
                weight: 1,
            }];
            if idx >= 15 {
                outputs.push(Arc {
                    place: accumulator,
                    weight: 1,
                });
            }
            TransitionInfo {
                id: format!("t{idx}"),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(idx as u32),
                    weight: 1,
                }],
                outputs,
            }
        })
        .collect();

    let mut initial_marking = vec![0; places.len()];
    initial_marking[0] = 1;

    PetriNet {
        name: Some("delayed-source-place-overflow".into()),
        places,
        transitions,
        initial_marking,
    }
}

#[test]
fn test_state_space_observer_linear_net() {
    let net = linear_deadlock_net();
    let config = default_config();
    let mut observer = StateSpaceObserver::new(&net.initial_marking);
    let result = explore(&net, &config, &mut observer);

    assert!(result.completed);
    let stats = observer.stats();
    assert_eq!(stats.states, big(2));
    assert_eq!(stats.edges, big(1));
    assert_eq!(stats.max_token_in_place, 1);
    assert_eq!(stats.max_token_sum, 1);
}

#[test]
fn test_one_safe_verdict_detects_delayed_source_place_overflow() {
    let net = delayed_source_place_overflow_net();
    let config = ExplorationConfig::new(64);

    assert!(
        matches!(
            super::super::one_safe_verdict(&net, &config, &[]),
            crate::output::Verdict::False | crate::output::Verdict::CannotCompute
        ),
        "delayed source-place overflow must not report OneSafe = TRUE"
    );
}

#[test]
fn test_state_space_observer_counting_net() {
    let net = counting_net();
    let config = default_config();
    let mut observer = StateSpaceObserver::new(&net.initial_marking);
    let result = explore(&net, &config, &mut observer);

    assert!(result.completed);
    let stats = observer.stats();
    assert_eq!(stats.states, big(4));
    assert_eq!(stats.edges, big(3));
    assert_eq!(stats.max_token_in_place, 3);
    assert_eq!(stats.max_token_sum, 3);
}

#[test]
fn test_state_space_observer_immediate_deadlock() {
    let net = immediate_deadlock_net();
    let config = default_config();
    let mut observer = StateSpaceObserver::new(&net.initial_marking);
    let result = explore(&net, &config, &mut observer);

    assert!(result.completed);
    let stats = observer.stats();
    assert_eq!(stats.states, big(1));
    assert_eq!(stats.edges, big(0));
    assert_eq!(stats.max_token_in_place, 1);
    assert_eq!(stats.max_token_sum, 1);
}

#[test]
fn test_state_space_never_stops_early() {
    let net = cyclic_safe_net();
    let config = default_config();
    let mut observer = StateSpaceObserver::new(&net.initial_marking);
    let result = explore(&net, &config, &mut observer);

    assert!(!result.stopped_by_observer);
    assert!(result.completed);
}

#[test]
fn test_deadlock_observer_parallel_matches_sequential_verdict() {
    let net = linear_deadlock_net();
    let sequential_config = default_config();
    let mut sequential = DeadlockObserver::new();
    let sequential_result = explore_observer(&net, &sequential_config, &mut sequential);

    let parallel_config = default_config().with_workers(4);
    let mut parallel = DeadlockObserver::new();
    let parallel_result = explore_observer(&net, &parallel_config, &mut parallel);

    assert_eq!(parallel.found_deadlock(), sequential.found_deadlock());
    assert_eq!(
        parallel_result.stopped_by_observer,
        sequential_result.stopped_by_observer
    );
}

#[test]
fn test_state_space_observer_parallel_matches_sequential_stats() {
    let net = counting_net();
    let sequential_config = default_config();
    let mut sequential = StateSpaceObserver::new(&net.initial_marking);
    let sequential_result = explore_observer(&net, &sequential_config, &mut sequential);

    let parallel_config = default_config().with_workers(4);
    let mut parallel = StateSpaceObserver::new(&net.initial_marking);
    let parallel_result = explore_observer(&net, &parallel_config, &mut parallel);

    assert_eq!(parallel_result.completed, sequential_result.completed);
    let sequential_stats = sequential.stats();
    let parallel_stats = parallel.stats();
    assert_eq!(parallel_stats.states, sequential_stats.states);
    assert_eq!(parallel_stats.edges, sequential_stats.edges);
    assert_eq!(
        parallel_stats.max_token_in_place,
        sequential_stats.max_token_in_place
    );
    assert_eq!(parallel_stats.max_token_sum, sequential_stats.max_token_sum);
}

// ── Orbit-quotient exact StateSpace count (place-swap symmetry) ───────────

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

/// Two interchangeable places `{p0, p1}` (orbit = full S_2) each draining into
/// a shared sink `s`. m0 = [1,1,0].
///
/// Full reachable graph: {[1,1,0], [0,1,1], [1,0,1], [0,0,2]} ⇒ |R| = 4,
/// edges = 4 (two out of [1,1,0]; one each out of the two single-token
/// markings; deadlock at [0,0,2]). The orbit quotient explores 3 canonical
/// reps — [1,1,0] (orbit 1), [0,1,1] (orbit 2, the merged single-token pair),
/// [0,0,2] (orbit 1) — and must recover EXACTLY |R| = 1+2+1 = 4 and
/// edges = 1·2 + 2·1 + 1·0 = 4 using the SOURCE orbit size (the merged rep
/// contributes deg 1 weighted by its orbit size 2). A target-orbit-size bug
/// would mis-count, so this pins the source-orbit edge formula.
fn two_place_swap_sink_net() -> PetriNet {
    PetriNet {
        name: Some("two-place-swap-sink".into()),
        places: vec![place("p0"), place("p1"), place("s")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 1, 0],
    }
}

/// Three interchangeable places `{p0,p1,p2}` (orbit = full S_3) draining into
/// a shared sink. m0 = [1,1,1,0]. Reachable markings are the 8 subsets of
/// drained places (|R| = 2^3 = 8); the quotient collapses them by token
/// count on the orbit.
fn three_place_swap_sink_net() -> PetriNet {
    PetriNet {
        name: Some("three-place-swap-sink".into()),
        places: vec![place("p0"), place("p1"), place("p2"), place("s")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(3, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(3, 1)]),
            trans("t2", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![1, 1, 1, 0],
    }
}

fn state_space_config() -> ExplorationConfig {
    ExplorationConfig::new(10_000).with_examination(Some(Examination::StateSpace))
}

#[test]
fn orbit_quotient_state_space_exact_on_swap_net_sequential() {
    let net = two_place_swap_sink_net();

    // Exact ground truth via the un-reduced path (no examination ⇒ no symmetry).
    let mut exact = StateSpaceObserver::new(&net.initial_marking);
    let exact_result = explore(&net, &default_config(), &mut exact);
    assert!(exact_result.completed);
    let exact = exact.stats();
    assert_eq!(
        (exact.states.clone(), exact.edges.clone()),
        (big(4), big(4)),
        "un-reduced ground truth"
    );

    // Orbit-quotient path must recover the SAME exact |R|, |E| and maxima.
    let mut quot = StateSpaceObserver::new(&net.initial_marking);
    let quot_result = explore(&net, &state_space_config(), &mut quot);
    assert!(quot_result.completed);
    let quot = quot.stats();
    assert_eq!(quot.states, big(4), "orbit-quotient |R| must be exact");
    assert_eq!(
        quot.edges,
        big(4),
        "orbit-quotient |E| (source-orbit) must be exact"
    );
    assert_eq!(quot.max_token_in_place, exact.max_token_in_place);
    assert_eq!(quot.max_token_sum, exact.max_token_sum);
}

#[test]
fn orbit_quotient_state_space_exact_on_swap_net_parallel() {
    let net = two_place_swap_sink_net();

    let mut exact = StateSpaceObserver::new(&net.initial_marking);
    let _ = explore_observer(&net, &default_config().with_workers(4), &mut exact);
    let exact = exact.stats();
    assert_eq!(
        (exact.states.clone(), exact.edges.clone()),
        (big(4), big(4))
    );

    // Parallel orbit-quotient: the summary threads the multinomial weight
    // (the prior bug dropped it, counting reps not markings).
    let mut quot = StateSpaceObserver::new(&net.initial_marking);
    let quot_result = explore_observer(&net, &state_space_config().with_workers(4), &mut quot);
    assert!(quot_result.completed);
    let quot = quot.stats();
    assert_eq!(
        quot.states,
        big(4),
        "parallel orbit-quotient |R| must be exact"
    );
    assert_eq!(
        quot.edges,
        big(4),
        "parallel orbit-quotient |E| must be exact"
    );
}

#[test]
fn orbit_quotient_state_space_exact_on_three_place_swap_net() {
    let net = three_place_swap_sink_net();

    let mut exact = StateSpaceObserver::new(&net.initial_marking);
    assert!(explore(&net, &default_config(), &mut exact).completed);
    let exact = exact.stats();
    // 2^3 = 8 reachable markings (each of the three tokens may have drained).
    assert_eq!(
        exact.states,
        big(8),
        "un-reduced |R| for three-place swap net"
    );

    for workers in [1usize, 4] {
        let mut quot = StateSpaceObserver::new(&net.initial_marking);
        let cfg = if workers == 1 {
            state_space_config()
        } else {
            state_space_config().with_workers(workers)
        };
        let result = if workers == 1 {
            explore(&net, &cfg, &mut quot)
        } else {
            explore_observer(&net, &cfg, &mut quot)
        };
        assert!(result.completed);
        let quot = quot.stats();
        assert_eq!(
            quot.states, exact.states,
            "orbit-quotient |R| must equal exact for workers={workers}",
        );
        assert_eq!(
            quot.edges, exact.edges,
            "orbit-quotient |E| must equal exact for workers={workers}",
        );
        assert_eq!(quot.max_token_in_place, exact.max_token_in_place);
        assert_eq!(quot.max_token_sum, exact.max_token_sum);
    }
}

/// A full-Sₙ STAR: `n` source places (each initially marked 1) each draining
/// into a shared sink via its own transition (`p_i → t_i → sink`). Reachable
/// markings = which subset of sources has drained, so |R| = 2ⁿ — intractable to
/// enumerate un-reduced for n ≥ ~20. The full symmetric group Sₙ collapses this
/// to n+1 canonical reps (one per drained-count k, with orbit size C(n,k)), and
/// Σ_k C(n,k) = 2ⁿ recovers |R| exactly.
fn star_drain_net(n: u32) -> PetriNet {
    let mut places: Vec<PlaceInfo> = (0..n).map(|i| place(&format!("p{i}"))).collect();
    places.push(place("sink"));
    let sink = n;
    let transitions = (0..n)
        .map(|i| trans(&format!("t{i}"), vec![arc(i, 1)], vec![arc(sink, 1)]))
        .collect();
    let mut initial_marking = vec![1u64; n as usize];
    initial_marking.push(0);
    PetriNet {
        name: Some(format!("star-drain-{n}")),
        places,
        transitions,
        initial_marking,
    }
}

/// END-TO-END WIN: a LARGE full-Sₙ star (n=20, |R| = 2²⁰ = 1_048_576) whose
/// StateSpace examination is INTRACTABLE un-reduced but COMPLETES via the
/// orbit-quotient — *only because* the widened thorough budget's structural
/// seeder discovers the FULL S₂₀ orbit. (Under the old 64-generator budget the
/// orbit truncated to ~7 sources, leaving the quotient ≈ the 2²⁰ concrete space,
/// so a small max_states would NOT suffice and BFS would hit the cap.)
///
/// We cap `max_states` far below 2²⁰ so the un-reduced path provably could NOT
/// complete, then assert the orbit-quotient path completes within that cap and
/// recovers the EXACT |R| = 2²⁰ via the multinomial orbit-size weighting.
#[test]
fn orbit_quotient_state_space_completes_large_star_under_full_symmetry() {
    let n = 20u32;
    let net = star_drain_net(n);
    let exact_states: usize = 1usize << n; // 2^20 = 1_048_576

    // Cap exploration at n+8 states: enough for the n+1 canonical reps the
    // quotient explores, but ~131_000× too small to enumerate 2^20 un-reduced.
    let cap = (n as usize) + 8;
    let cfg = ExplorationConfig::new(cap).with_examination(Some(Examination::StateSpace));

    let mut quot = StateSpaceObserver::new(&net.initial_marking);
    let result = explore(&net, &cfg, &mut quot);
    assert!(
        result.completed,
        "orbit-quotient StateSpace must COMPLETE within {cap} reps (full S_{n} \
         collapses 2^{n} markings to {} reps); if this fails the widened budget \
         did not discover the full orbit",
        n + 1,
    );
    let quot = quot.stats();
    assert_eq!(
        quot.states,
        big(exact_states as u64),
        "orbit-quotient |R| must recover the EXACT 2^{n} via multinomial weights",
    );
    // Sanity: the quotient visited only the n+1 drained-count classes, not 2^n.
    assert!(
        result.states_visited <= cap,
        "quotient must explore ≤ {cap} canonical reps, visited {}",
        result.states_visited,
    );
}

// ── COUPLED (cyclic) group orbit-quotient via the BSGS path ──────────────

/// A directed RING `p_i → t_i → p_{(i+1)%n}` with `tokens` tokens circulating.
/// Its place-symmetry group is the CYCLIC `Z_n` rotation (coupled — all places
/// rotate together), NOT a full symmetric group, so the per-orbit multinomial
/// would OVER-count and the explorer takes the BSGS `GroupOrbit` path. The
/// orbit-quotient count must still recover EXACTLY the un-reduced `|R|`/`|E|`.
fn token_ring_net(n: u32, tokens: u64) -> PetriNet {
    let places = (0..n).map(|i| place(&format!("p{i}"))).collect();
    let transitions = (0..n)
        .map(|i| trans(&format!("t{i}"), vec![arc(i, 1)], vec![arc((i + 1) % n, 1)]))
        .collect();
    let mut initial_marking = vec![0u64; n as usize];
    // Place all tokens on p0 (a single rotating cluster keeps the net live and
    // the orbit non-trivial under Z_n).
    initial_marking[0] = tokens;
    PetriNet {
        name: Some(format!("token-ring-{n}-{tokens}")),
        places,
        transitions,
        initial_marking,
    }
}

#[test]
fn orbit_quotient_state_space_exact_on_coupled_cyclic_ring() {
    // Z_6 ring with 2 circulating tokens: a coupled group (|G|=6) the
    // multinomial cannot count. ON (BSGS GroupOrbit) must equal OFF exactly.
    let net = token_ring_net(6, 2);

    // Un-reduced ground truth (no examination ⇒ symmetry off).
    let mut exact = StateSpaceObserver::new(&net.initial_marking);
    assert!(explore(&net, &default_config(), &mut exact).completed);
    let exact = exact.stats();

    for workers in [1usize, 4] {
        let mut quot = StateSpaceObserver::new(&net.initial_marking);
        let cfg = if workers == 1 {
            state_space_config()
        } else {
            state_space_config().with_workers(workers)
        };
        let result = if workers == 1 {
            explore(&net, &cfg, &mut quot)
        } else {
            explore_observer(&net, &cfg, &mut quot)
        };
        assert!(result.completed);
        let quot = quot.stats();
        assert_eq!(
            quot.states, exact.states,
            "coupled cyclic orbit-quotient |R| must equal exact for workers={workers}",
        );
        assert_eq!(
            quot.edges, exact.edges,
            "coupled cyclic orbit-quotient |E| must equal exact for workers={workers}",
        );
        assert_eq!(quot.max_token_in_place, exact.max_token_in_place);
        assert_eq!(quot.max_token_sum, exact.max_token_sum);
    }
}

#[test]
fn orbit_quotient_state_space_exact_on_coupled_ring_varied_sizes() {
    // Sweep ring sizes/token counts so the differential covers several coupled
    // Z_n groups and orbit-size distributions.
    for (n, tokens) in [(4u32, 1u64), (4, 2), (5, 2), (6, 3), (7, 2)] {
        let net = token_ring_net(n, tokens);
        let mut exact = StateSpaceObserver::new(&net.initial_marking);
        assert!(
            explore(&net, &default_config(), &mut exact).completed,
            "exact must complete for ring({n},{tokens})",
        );
        let exact = exact.stats();

        let mut quot = StateSpaceObserver::new(&net.initial_marking);
        assert!(explore(&net, &state_space_config(), &mut quot).completed);
        let quot = quot.stats();
        assert_eq!(
            (quot.states, quot.edges),
            (exact.states, exact.edges),
            "coupled ring({n},{tokens}) orbit-quotient must equal exact",
        );
        assert_eq!(quot.max_token_in_place, exact.max_token_in_place);
        assert_eq!(quot.max_token_sum, exact.max_token_sum);
    }
}

// ── Structural deadlock-freedom shortcut ─────────────────────────────────

/// Cyclic net (token cycles p0→p1→p0) is structurally deadlock-free.
/// `deadlock_verdict` should return FALSE via siphon/trap without BFS.
#[test]
fn test_deadlock_verdict_structural_shortcut_cyclic_net() {
    let net = cyclic_safe_net();
    // Use a tiny budget — if the structural shortcut works, no BFS needed.
    let config = ExplorationConfig::new(1);
    assert_eq!(
        super::super::deadlock_verdict(&net, &config),
        crate::output::Verdict::False
    );
}

/// Linear net (p0→t0→p1, no cycle) has a siphon vulnerability.
/// Structural analysis returns `Some(false)` (inconclusive), so BFS runs
/// and finds the actual deadlock.
#[test]
fn test_deadlock_verdict_falls_through_to_bfs_on_linear_net() {
    let net = linear_deadlock_net();
    let config = default_config();
    assert_eq!(
        super::super::deadlock_verdict(&net, &config),
        crate::output::Verdict::True
    );
}

/// Immediate deadlock net (no transitions) — structural analysis returns
/// `Some(false)` (not deadlock-free), BFS confirms deadlock.
#[test]
fn test_deadlock_verdict_immediate_deadlock_structural_then_bfs() {
    let net = immediate_deadlock_net();
    let config = default_config();
    assert_eq!(
        super::super::deadlock_verdict(&net, &config),
        crate::output::Verdict::True
    );
}

/// Non-free-choice net with a reachable deadlock.
///
/// P0(1) and P1(1) feed two conflicting transitions:
///   T0: {P0} → {P2}        (needs only P0)
///   T1: {P0, P1} → {P3}    (needs both P0 and P1)
///
/// P0 is shared between T0 (input set {P0}) and T1 (input set {P0, P1}),
/// making this non-free-choice. Both nondeterministic paths deadlock:
///   - T0 fires → (0,1,1,0): nothing enabled
///   - T1 fires → (0,0,0,1): nothing enabled
///
/// Regression for 29f42ed79: `structural_deadlock_free` must return `None`
/// (non-free-choice guard), so `deadlock_verdict` falls through to BFS.
#[test]
fn test_deadlock_verdict_non_free_choice_net_falls_through_to_bfs() {
    let net = PetriNet {
        name: Some("non-free-choice-deadlock".into()),
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
                    place: PlaceIdx(2),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "T1".into(),
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
                outputs: vec![Arc {
                    place: PlaceIdx(3),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![1, 1, 0, 0],
    };
    let config = default_config();
    assert_eq!(
        super::super::deadlock_verdict(&net, &config),
        crate::output::Verdict::True
    );
}

// ── POR (stubborn set) integration tests ────────────────────────────────

/// Two independent processes (p0→t0→p1, p2→t1→p3) both leading to deadlock.
/// Full BFS explores 4 states (all interleavings). Deadlock-preserving POR
/// should explore fewer states (only one interleaving order) while still
/// finding the same deadlock.
fn two_independent_deadlocking_processes() -> PetriNet {
    PetriNet {
        name: Some("two-independent-deadlock".into()),
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
                id: "p2".into(),
                name: None,
            },
            PlaceInfo {
                id: "p3".into(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "t0".into(),
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
                id: "t1".into(),
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
        initial_marking: vec![1, 0, 1, 0],
    }
}

#[test]
fn test_por_deadlock_preserving_reduces_state_count() {
    let net = two_independent_deadlocking_processes();

    // Full BFS (no POR): explores all interleavings
    let full_config = ExplorationConfig::new(1000);
    let mut full_observer = DeadlockObserver::new();
    let full_result = explore(&net, &full_config, &mut full_observer);

    // POR BFS (deadlock-preserving): explores reduced state space
    let por_config = ExplorationConfig::new(1000).with_por(PorStrategy::DeadlockPreserving);
    let mut por_observer = DeadlockObserver::new();
    let por_result = explore(&net, &por_config, &mut por_observer);

    // Both find the deadlock (observer stops early once found)
    assert!(
        full_observer.found_deadlock(),
        "full BFS should find deadlock"
    );
    assert!(
        por_observer.found_deadlock(),
        "POR BFS should find deadlock"
    );

    // POR should explore strictly fewer states on independent processes.
    // Full BFS: 4 states (initial, t0→only, t1→only, both-fired deadlock).
    // POR: 3 states (one interleaving: initial → t0-fired → both-fired deadlock).
    assert!(
        por_result.states_visited < full_result.states_visited,
        "POR should explore fewer states: POR={}, full={}",
        por_result.states_visited,
        full_result.states_visited,
    );
}

#[test]
fn test_por_no_reduction_on_shared_resource_gives_same_deadlock() {
    // Two transitions competing for one token: t0 reads p0→p1, t1 reads p0→p2.
    // Since both share p0, the stubborn set = all enabled = no reduction.
    // POR and full BFS should give identical results.
    let net = PetriNet {
        name: Some("shared-resource".into()),
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
                id: "p2".into(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "t0".into(),
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
                id: "t1".into(),
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
        ],
        initial_marking: vec![1, 0, 0],
    };

    let full_config = ExplorationConfig::new(1000);
    let mut full_obs = DeadlockObserver::new();
    let full_result = explore(&net, &full_config, &mut full_obs);

    let por_config = ExplorationConfig::new(1000).with_por(PorStrategy::DeadlockPreserving);
    let mut por_obs = DeadlockObserver::new();
    let por_result = explore(&net, &por_config, &mut por_obs);

    assert_eq!(full_result.completed, por_result.completed);
    assert_eq!(full_obs.found_deadlock(), por_obs.found_deadlock());
    assert_eq!(full_result.states_visited, por_result.states_visited);
}

/// Net where P-invariants give no bounds but LP proves all places ≤ 1.
///
/// t0: p0 → p1, t1: p1 → (consumed, no output)
/// P-invariant: y^T·C = 0 has only trivial solution (y=0) because the
/// incidence matrix has rank 2 and both rows are linearly independent.
/// LP state equation: M_p0 = 1 - x_t0 ≥ 0 → x_t0 ≤ 1 → M_p0 ≤ 1;
/// M_p1 = x_t0 - x_t1 ≥ 0 and x_t0 ≤ 1 → M_p1 ≤ 1.
#[test]
fn test_one_safe_lp_structural_proof_when_p_invariants_insufficient() {
    let net = PetriNet {
        name: None,
        places: vec![
            PlaceInfo {
                id: "p0".into(),
                name: Some("p0".into()),
            },
            PlaceInfo {
                id: "p1".into(),
                name: Some("p1".into()),
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "t0".into(),
                name: Some("t0".into()),
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
                id: "t1".into(),
                name: Some("t1".into()),
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![],
            },
        ],
        initial_marking: vec![1, 0],
    };

    // Verify P-invariants are insufficient: no P-invariant bounds either place.
    let invariants = crate::invariant::compute_p_invariants(&net);
    let p0_bound = crate::invariant::structural_place_bound(&invariants, 0);
    let p1_bound = crate::invariant::structural_place_bound(&invariants, 1);
    assert!(
        p0_bound.is_none() || p1_bound.is_none(),
        "At least one place should not be bounded by P-invariants"
    );

    // But LP should prove both places ≤ 1.
    use crate::lp_state_equation::lp_upper_bound;
    let lp_p0 = lp_upper_bound(&net, &[PlaceIdx(0)]);
    let lp_p1 = lp_upper_bound(&net, &[PlaceIdx(1)]);
    assert_eq!(lp_p0, Some(1), "LP should bound p0 to 1");
    assert_eq!(lp_p1, Some(1), "LP should bound p1 to 1");

    // The verdict should be TRUE (1-safe) via the LP structural path.
    let config = ExplorationConfig::new(64);
    let verdict = super::super::one_safe_verdict(&net, &config, &[]);
    assert_eq!(verdict, crate::output::Verdict::True);
}

// ── Worker-budget contract parity tests (#1520) ──────────────────────

/// `deadlock_verdict` must give the same answer regardless of worker count.
/// On a deadlocking net, both sequential (workers=1) and portfolio (workers=4)
/// paths must return TRUE.
#[test]
fn test_deadlock_verdict_parity_linear_deadlock_workers_1_vs_4() {
    let net = linear_deadlock_net();
    let v1 = super::super::deadlock_verdict(&net, &ExplorationConfig::new(1024));
    let v4 = super::super::deadlock_verdict(&net, &ExplorationConfig::new(1024).with_workers(4));
    assert_eq!(v1, crate::output::Verdict::True);
    assert_eq!(v4, crate::output::Verdict::True);
}

/// Structurally deadlock-free cyclic net: both paths must return FALSE.
/// This is resolved by the structural shortcut before the budget split matters,
/// but the test confirms the full path is consistent.
#[test]
fn test_deadlock_verdict_parity_cyclic_safe_workers_1_vs_4() {
    let net = cyclic_safe_net();
    let v1 = super::super::deadlock_verdict(&net, &ExplorationConfig::new(1024));
    let v4 = super::super::deadlock_verdict(&net, &ExplorationConfig::new(1024).with_workers(4));
    assert_eq!(v1, crate::output::Verdict::False);
    assert_eq!(v4, crate::output::Verdict::False);
}

/// Immediate deadlock net: both paths must return TRUE.
#[test]
fn test_deadlock_verdict_parity_immediate_deadlock_workers_1_vs_4() {
    let net = immediate_deadlock_net();
    let v1 = super::super::deadlock_verdict(&net, &ExplorationConfig::new(1024));
    let v4 = super::super::deadlock_verdict(&net, &ExplorationConfig::new(1024).with_workers(4));
    assert_eq!(v1, crate::output::Verdict::True);
    assert_eq!(v4, crate::output::Verdict::True);
}
