// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::{build_gba, GbaStateId, LtlNnf};

fn waiting_state(gba: &super::Gba) -> GbaStateId {
    (0..gba.num_states)
        .find(|state| !gba.acceptance[0].contains(state))
        .expect("expected a waiting state")
}

fn done_state(gba: &super::Gba) -> GbaStateId {
    *gba.acceptance[0]
        .iter()
        .next()
        .expect("expected an accepting state")
}

#[test]
fn test_gba_atom_only() {
    let gba = build_gba(&LtlNnf::Atom(0));

    assert_eq!(gba.num_states, 1);
    assert!(gba.acceptance.is_empty());
    assert_eq!(gba.initial_transitions.len(), 1);
    assert_eq!(gba.initial_transitions[0].pos_atoms, vec![0]);
    assert_eq!(gba.initial_transitions[0].successor, 0);
    assert_eq!(gba.transitions[0].len(), 1);
    assert_eq!(gba.transitions[0][0].successor, 0);
}

#[test]
fn test_gba_until_simple() {
    let formula = LtlNnf::Until(Box::new(LtlNnf::Atom(0)), Box::new(LtlNnf::Atom(1)));
    let gba = build_gba(&formula);
    let waiting = waiting_state(&gba);
    let done = done_state(&gba);

    assert_eq!(gba.num_states, 2);
    assert_eq!(gba.acceptance.len(), 1);
    assert_eq!(gba.acceptance[0].len(), 1);
    assert_ne!(waiting, done);
    assert!(gba
        .initial_transitions
        .iter()
        .any(|transition| transition.pos_atoms == vec![1] && transition.successor == done));
    assert!(gba
        .initial_transitions
        .iter()
        .any(|transition| transition.pos_atoms == vec![0] && transition.successor == waiting));
    assert_eq!(gba.transitions[waiting as usize].len(), 2);
}

#[test]
fn test_gba_release_simple() {
    let formula = LtlNnf::Release(Box::new(LtlNnf::Atom(0)), Box::new(LtlNnf::Atom(1)));
    let gba = build_gba(&formula);

    assert_eq!(gba.num_states, 2);
    assert!(gba.acceptance.is_empty());
    assert_eq!(gba.initial_transitions.len(), 2);
    assert!(gba
        .initial_transitions
        .iter()
        .any(|transition| transition.pos_atoms == vec![1, 0]));
    assert!(gba
        .initial_transitions
        .iter()
        .any(|transition| transition.pos_atoms == vec![1]));
}

#[test]
fn test_gba_globally() {
    let formula = LtlNnf::Release(Box::new(LtlNnf::False), Box::new(LtlNnf::Atom(0)));
    let gba = build_gba(&formula);

    assert_eq!(gba.num_states, 1);
    assert!(gba.acceptance.is_empty());
    assert_eq!(gba.initial_transitions.len(), 1);
    assert_eq!(gba.initial_transitions[0].pos_atoms, vec![0]);
    assert_eq!(gba.initial_transitions[0].successor, 0);
    assert_eq!(gba.transitions[0].len(), 1);
    assert_eq!(gba.transitions[0][0].pos_atoms, vec![0]);
    assert_eq!(gba.transitions[0][0].successor, 0);
}

#[test]
fn test_gba_finally() {
    let formula = LtlNnf::Until(Box::new(LtlNnf::True), Box::new(LtlNnf::Atom(0)));
    let gba = build_gba(&formula);
    let waiting = waiting_state(&gba);
    let done = done_state(&gba);

    assert_eq!(gba.num_states, 2);
    assert_eq!(gba.acceptance.len(), 1);
    assert!(gba
        .initial_transitions
        .iter()
        .any(|transition| transition.pos_atoms == vec![0] && transition.successor == done));
    assert!(gba
        .initial_transitions
        .iter()
        .any(|transition| transition.pos_atoms.is_empty() && transition.successor == waiting));
}

#[test]
fn test_gba_acceptance_condition_marks_done_state_only() {
    let formula = LtlNnf::Until(Box::new(LtlNnf::Atom(0)), Box::new(LtlNnf::Atom(1)));
    let gba = build_gba(&formula);
    let waiting = waiting_state(&gba);
    let done = done_state(&gba);

    assert!(gba.acceptance[0].contains(&done));
    assert!(!gba.acceptance[0].contains(&waiting));
}

/// Losslessness of the packed acceptance codec (audit sink S3): for every GBA
/// state and every outgoing edge, every packed bit must equal the `Gba`'s own
/// set-membership / `edge_accept` flag — the self-certifying invariant the
/// product builders' packed storage rests on.
#[test]
fn test_acceptance_masks_lossless_vs_gba_sets() {
    let a = || Box::new(LtlNnf::Atom(0));
    let b = || Box::new(LtlNnf::Atom(1));
    let c = || Box::new(LtlNnf::Atom(2));
    let formulas: Vec<LtlNnf> = vec![
        // 0 acceptance sets (no Until).
        LtlNnf::Atom(0),
        LtlNnf::Release(a(), b()),
        // 1 set.
        LtlNnf::Until(a(), b()),
        LtlNnf::Until(Box::new(LtlNnf::True), b()),
        // Nested / multiple Untils, incl. the G(X(F p)) edge-acceptance shape.
        LtlNnf::Until(Box::new(LtlNnf::Until(a(), b())), c()),
        LtlNnf::And(vec![
            LtlNnf::Until(a(), b()),
            LtlNnf::Until(b(), c()),
            LtlNnf::Release(
                Box::new(LtlNnf::False),
                Box::new(LtlNnf::Next(Box::new(LtlNnf::Until(
                    Box::new(LtlNnf::True),
                    a(),
                )))),
            ),
        ]),
    ];
    for formula in &formulas {
        let gba = build_gba(formula);
        let acc = super::AcceptanceMasks::from_gba(&gba);
        let num_accept = gba.acceptance.len();
        assert_eq!(acc.num_accept, num_accept);
        assert_eq!(acc.num_words, num_accept.div_ceil(64));
        for q in 0..gba.num_states {
            let words = acc.state(q);
            assert_eq!(words.len(), acc.num_words);
            for i in 0..num_accept {
                assert_eq!(
                    super::accept_bit(words, i),
                    gba.acceptance[i].contains(&q),
                    "state-acceptance bit drift: formula={formula:?} q={q} set={i}"
                );
            }
            for (e, t) in gba.transitions[q as usize].iter().enumerate() {
                let words = acc.edge(q, e);
                assert_eq!(words.len(), acc.num_words);
                for i in 0..num_accept {
                    assert_eq!(
                        super::accept_bit(words, i),
                        t.edge_accept.get(i).copied().unwrap_or(false),
                        "edge-acceptance bit drift: formula={formula:?} q={q} edge={e} set={i}"
                    );
                }
            }
        }
    }
}
