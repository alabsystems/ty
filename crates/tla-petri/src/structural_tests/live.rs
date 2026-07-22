// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use std::time::{Duration, Instant};

#[test]
fn test_structural_live_free_choice_cycle() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    };

    assert_eq!(structural_live(&net, None), Some(true));
}

#[test]
fn test_structural_live_reports_uncovered_siphon_on_linear_net() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
        initial_marking: vec![1, 0],
    };

    assert_eq!(structural_live(&net, None), Some(false));
}

#[test]
fn test_structural_live_rejects_non_free_choice_net() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2"), place("p3")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(0, 1), arc(1, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![1, 1, 0, 0],
    };

    assert_eq!(structural_live(&net, None), None);
}

#[test]
fn test_structural_live_complete_enumeration_decides_multi_seed_siphon() {
    // ORDINARY FREE-CHOICE (and marked-graph) net whose only minimal siphon
    // {p0, p1} needs a multi-place seed: the OLD single-place-seed heuristic
    // missed it and had to decline (None). The COMPLETE branch-and-bound
    // enumeration finds it; it is unmarked and is its own (unmarked) maximal
    // trap, so Commoner's only-if direction yields the exact FALSE — the
    // initial marking is in fact already a deadlock. The marked-graph
    // certificate fires first ({p0,p1} is an unmarked circuit) and the
    // debug_assert cross-check verifies the two certificates agree.
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2"), place("p3")],
        transitions: vec![
            trans("t0", vec![arc(1, 1), arc(2, 1)], vec![arc(0, 1), arc(3, 1)]),
            trans("t1", vec![arc(0, 1), arc(3, 1)], vec![arc(1, 1), arc(2, 1)]),
        ],
        initial_marking: vec![0, 0, 1, 2],
    };
    assert_eq!(
        structural_live(&net, None),
        Some(false),
        "the unmarked minimal siphon {{p0,p1}} is an exact non-liveness witness"
    );
}

#[test]
fn test_structural_live_declines_uncovered_with_isolated_place() {
    // The NEGATIVE direction must fail-closed when a degenerate isolated place
    // manufactures a spurious uncovered singleton siphon. Here p1 is isolated
    // (no transition touches it) and initially unmarked, so {p1} is a siphon
    // with no marked trap — but it gates nothing. The single live cycle on
    // p0/p2 is genuinely live, so reporting `Some(false)` would be WRONG.
    // Every certificate is gated off: the state-machine certificate requires
    // every place incident, the marked-graph shape fails (p1 has no
    // producer/consumer), and the free-choice certificate requires every
    // place incident.
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };
    assert_ne!(
        structural_live(&net, None),
        Some(false),
        "an isolated unmarked place must not trigger a spurious non-liveness verdict"
    );
    assert_eq!(structural_live(&net, None), None);
}

#[test]
fn test_structural_live_rejects_weighted_net() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 2)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 2)]),
        ],
        initial_marking: vec![2, 0],
    };

    assert_eq!(structural_live(&net, None), None);
}

/// Free-choice net that is NOT a state machine (t_join has two inputs) and
/// NOT a marked graph (p0 has two consumers), so only the Commoner–Hack
/// certificate applies. p0 holds the single token; t0/t1 are a free choice
/// on p0; t_join needs pa AND pb simultaneously, but only one of them can
/// ever be marked — firing either choice strands the token. NOT live.
fn fc_only_choice_join_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0"), place("pa"), place("pb")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t_join", vec![arc(1, 1), arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    }
}

#[test]
fn test_structural_live_fc_commoner_uncovered_siphon_is_false() {
    // {p0, pa} is a siphon (every producer into it consumes from it) whose
    // maximal trap is empty (t1 consumes p0 producing only pb; t_join
    // consumes pa producing only p0 — the trap fixpoint drains both places),
    // so Commoner's only-if direction gives the exact FALSE.
    let net = fc_only_choice_join_net();
    assert_eq!(structural_live(&net, None), Some(false));
}

#[test]
fn test_structural_live_fc_expired_deadline_declines() {
    // An already-expired deadline must make the Commoner certificate decline
    // (None) — the enumeration bails before recording anything, so neither
    // direction may be claimed. Verdict-preserving by construction.
    let net = fc_only_choice_join_net();
    // `now - 1s` cannot underflow the monotonic clock in practice; this is a
    // test fixture constructing a deadline that is already in the past.
    #[allow(clippy::unchecked_time_subtraction)]
    let past = Instant::now() - Duration::from_secs(1);
    assert_eq!(
        structural_live(&net, Some(past)),
        None,
        "expired deadline must decline, never guess"
    );
}

#[test]
fn test_structural_live_fc_declines_source_transition() {
    // A source transition is outside the textbook free-choice "system"
    // setting; the FC certificate must decline. The net is also neither a
    // state machine (t_src has no input) nor a marked graph (p0 has two
    // producers), so the chain yields None.
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t_src", vec![], vec![arc(0, 1)]),
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![0, 0],
    };
    assert_eq!(structural_live(&net, None), None);
}

#[test]
fn test_structural_live_sink_transition_drain_is_false() {
    // Sink-transition shape: t_sink can drain the only token, after which
    // every transition is dead → NOT live. The maximal trap inside the
    // siphon {p0, p1} is empty (t_sink consumes p1 producing nothing), so
    // the Commoner certificate emits the exact FALSE. (Not a state machine:
    // t_sink has no output. Not a marked graph: p1 has two consumers.)
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            trans("t_sink", vec![arc(1, 1)], vec![]),
        ],
        initial_marking: vec![1, 0],
    };
    assert_eq!(structural_live(&net, None), Some(false));
}

#[test]
fn test_structural_live_fc_covered_choice_net_is_true() {
    // Free-choice (p0 has two single-input consumers), not a state machine
    // (tb has two outputs), not a marked graph (p0 has two consumers). Both
    // branches return the token to p0 (the pb branch also forks a side token
    // into pc which cycles back), so the net is live; every minimal siphon
    // contains a marked trap reaching through p0.
    //   p0 → t0 → pa → ta → p0
    //   p0 → t1 → pb → tb → {p0, pc};  pc → tc → pc (self-loop)
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("pa"), place("pb"), place("pc")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("ta", vec![arc(1, 1)], vec![arc(0, 1)]),
            trans("tb", vec![arc(2, 1)], vec![arc(0, 1), arc(3, 1)]),
            trans("tc", vec![arc(3, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![1, 0, 0, 0],
    };
    assert_eq!(structural_live(&net, None), Some(true));
}

#[test]
fn test_structural_live_state_machine_certificates() {
    // Strongly-connected state machine with one token → live (exact TRUE).
    let sc_marked = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t2", vec![arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![0, 0, 1],
    };
    assert_eq!(structural_live(&sc_marked, None), Some(true));

    // Same shape, token-free → every transition dead → exact FALSE.
    let sc_unmarked = PetriNet {
        initial_marking: vec![0, 0, 0],
        ..sc_marked.clone()
    };
    assert_eq!(structural_live(&sc_unmarked, None), Some(false));

    // Not strongly connected (chain): the source place drains → exact FALSE.
    let chain = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };
    assert_eq!(structural_live(&chain, None), Some(false));

    // Two disjoint SC components, both marked → live; unmark one → FALSE.
    let two_components = PetriNet {
        name: None,
        places: vec![place("a0"), place("a1"), place("b0"), place("b1")],
        transitions: vec![
            trans("ta0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("ta1", vec![arc(1, 1)], vec![arc(0, 1)]),
            trans("tb0", vec![arc(2, 1)], vec![arc(3, 1)]),
            trans("tb1", vec![arc(3, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 1, 0],
    };
    assert_eq!(structural_live(&two_components, None), Some(true));
    let one_empty_component = PetriNet {
        initial_marking: vec![1, 0, 0, 0],
        ..two_components.clone()
    };
    assert_eq!(structural_live(&one_empty_component, None), Some(false));
}

#[test]
fn test_structural_live_marked_graph_certificates() {
    // Marked-graph fork/join with every circuit marked → live (exact TRUE).
    // Not a state machine: t_fork has two outputs, t_join two inputs.
    //   t_fork: p0 → {pa, pb};  t_join: {pa, pb} → p0
    let mg_live = PetriNet {
        name: None,
        places: vec![place("p0"), place("pa"), place("pb")],
        transitions: vec![
            trans("t_fork", vec![arc(0, 1)], vec![arc(1, 1), arc(2, 1)]),
            trans("t_join", vec![arc(1, 1), arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };
    assert_eq!(structural_live(&mg_live, None), Some(true));

    // Unmark the only token: every circuit (through p0/pa and p0/pb) is
    // token-free → exact FALSE.
    let mg_dead = PetriNet {
        initial_marking: vec![0, 0, 0],
        ..mg_live.clone()
    };
    assert_eq!(structural_live(&mg_dead, None), Some(false));
}
