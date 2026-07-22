// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! END-TO-END differential: the additive symbolic LTL lane
//! ([`super::try_symbolic_ltl_verdict`]) vs the EXHAUSTIVE explicit full-graph
//! LTL decision ([`crate::buchi::check_ltl_on_full_graph`], which runs the
//! real GPVW Büchi construction + product SCC over the complete reachability
//! graph).
//!
//! For every (net × real LTL formula) pair the test computes BOTH verdicts and
//! requires them to be EQUAL wherever the symbolic lane returns a definite
//! verdict (it is fail-closed, so it may decline — declines are not failures,
//! but a WRONG definite verdict is). The formula battery spans G/F/U/X, nested
//! temporal operators, multi-`Until`, safety + liveness fragments, fair vs
//! unfair cycles, and deadlocking nets. 0 disagreements is the soundness bar.

use super::try_symbolic_ltl_verdict;
use crate::buchi::{check_ltl_on_full_graph, to_nnf};
use crate::explorer::{explore_full, ExplorationConfig};
use crate::model::PropertyAliases;
use crate::output::Verdict;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{IntExpr, LtlFormula, StatePredicate};

// ───────────────────────────── net fixtures ─────────────────────────────

fn place(id: &str) -> PlaceInfo {
    PlaceInfo {
        id: id.into(),
        name: None,
    }
}

fn trans(id: &str, inputs: Vec<(usize, u64)>, outputs: Vec<(usize, u64)>) -> TransitionInfo {
    TransitionInfo {
        id: id.into(),
        name: None,
        inputs: inputs
            .into_iter()
            .map(|(p, w)| Arc {
                place: PlaceIdx(p as u32),
                weight: w,
            })
            .collect(),
        outputs: outputs
            .into_iter()
            .map(|(p, w)| Arc {
                place: PlaceIdx(p as u32),
                weight: w,
            })
            .collect(),
    }
}

/// p0 ⇄ p1 token shuttle (always cycles, never deadlocks).
fn shuttle() -> PetriNet {
    PetriNet {
        name: Some("shuttle".into()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("a", vec![(0, 1)], vec![(1, 1)]),
            trans("b", vec![(1, 1)], vec![(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    }
}

/// p0 → p1 once, then deadlocks.
fn deadlocks() -> PetriNet {
    PetriNet {
        name: Some("deadlocks".into()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("a", vec![(0, 1)], vec![(1, 1)])],
        initial_marking: vec![1, 0],
    }
}

/// Branch: from p0 either loop forever (p0→p1→p0) or fall into a deadlocking
/// sink p2. p0+p1+p2 conserved at 1.
fn branch() -> PetriNet {
    PetriNet {
        name: Some("branch".into()),
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("a", vec![(0, 1)], vec![(1, 1)]),    // p0->p1
            trans("loop", vec![(1, 1)], vec![(0, 1)]), // p1->p0
            trans("sink", vec![(0, 1)], vec![(2, 1)]), // p0->p2 (deadlocks)
        ],
        initial_marking: vec![1, 0, 0],
    }
}

/// Free-choice 3-place ring p0→p1→p2→p0 (always cycles).
fn ring3() -> PetriNet {
    PetriNet {
        name: Some("ring3".into()),
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("a", vec![(0, 1)], vec![(1, 1)]),
            trans("b", vec![(1, 1)], vec![(2, 1)]),
            trans("c", vec![(2, 1)], vec![(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    }
}

fn nets() -> Vec<PetriNet> {
    vec![shuttle(), deadlocks(), branch(), ring3()]
}

// ───────────────────────────── formula battery ─────────────────────────────

fn atom_ge(place_id: &str, v: u64) -> LtlFormula {
    LtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(v),
        IntExpr::TokensCount(vec![place_id.to_string()]),
    ))
}

fn g(f: LtlFormula) -> LtlFormula {
    LtlFormula::Globally(Box::new(f))
}
fn f_(f: LtlFormula) -> LtlFormula {
    LtlFormula::Finally(Box::new(f))
}
fn x(f: LtlFormula) -> LtlFormula {
    LtlFormula::Next(Box::new(f))
}
fn u(a: LtlFormula, b: LtlFormula) -> LtlFormula {
    LtlFormula::Until(Box::new(a), Box::new(b))
}
fn not(a: LtlFormula) -> LtlFormula {
    LtlFormula::Not(Box::new(a))
}
fn and(cs: Vec<LtlFormula>) -> LtlFormula {
    LtlFormula::And(cs)
}
fn or(cs: Vec<LtlFormula>) -> LtlFormula {
    LtlFormula::Or(cs)
}

/// A battery of LTL formulas (over place ids "p0","p1","p2") covering the hard
/// shapes. Each net only uses the places it has; formulas referencing a
/// missing place still resolve (the atom is just always false there), which is
/// fine — both oracles see the same thing.
fn formulas() -> Vec<(&'static str, LtlFormula)> {
    let p0 = || atom_ge("p0", 1);
    let p1 = || atom_ge("p1", 1);
    let p2 = || atom_ge("p2", 1);
    vec![
        // safety / invariants
        ("G p0", g(p0())),
        ("G !(p0 & p1)", g(not(and(vec![p0(), p1()])))),
        ("G (p0 | p1)", g(or(vec![p0(), p1()]))),
        // reachability / liveness
        ("F p1", f_(p1())),
        ("F p2", f_(p2())),
        ("G F p0", g(f_(p0()))),
        ("G F p1", g(f_(p1()))),
        ("F G p1", f_(g(p1()))),
        ("F G p0", f_(g(p0()))),
        // next
        ("X p1", x(p1())),
        ("X X p0", x(x(p0()))),
        ("G X (p0 | p1)", g(x(or(vec![p0(), p1()])))),
        // until
        ("p0 U p1", u(p0(), p1())),
        ("(!p2) U p1", u(not(p2()), p1())),
        ("G (p0 -> X p1)", g(or(vec![not(p0()), x(p1())]))),
        // nested temporal — the residual that reaches Büchi
        ("G X F p0", g(x(f_(p0())))),
        ("G F (p0 | p2)", g(f_(or(vec![p0(), p2()])))),
        ("F (p1 & X p0)", f_(and(vec![p1(), x(p0())]))),
        ("G (F p0 & F p1)", g(and(vec![f_(p0()), f_(p1())]))),
        (
            "(G F p0) -> (G F p1)",
            or(vec![not(g(f_(p0()))), g(f_(p1()))]),
        ),
        // response
        ("G (p0 -> F p1)", g(or(vec![not(p0()), f_(p1())]))),
    ]
}

// ───────────────────────────── the differential ─────────────────────────────

/// Exhaustive verdict via the explicit full-graph Büchi product (the oracle).
fn exhaustive_verdict(net: &PetriNet, formula: &LtlFormula) -> Option<Verdict> {
    let aliases = PropertyAliases::identity(net);
    let mut atom_preds = Vec::new();
    let nnf = to_nnf(formula, &mut atom_preds);
    let resolved: Vec<_> = atom_preds
        .iter()
        .map(|p| crate::buchi::resolve_atom_with_aliases(p, &aliases))
        .collect();
    let full = explore_full(net, &ExplorationConfig::default());
    match check_ltl_on_full_graph(&nnf, &full, net, &resolved, None) {
        Some(true) => Some(Verdict::True),
        Some(false) => Some(Verdict::False),
        None => None,
    }
}

/// Symbolic-lane verdict.
fn symbolic_verdict(net: &PetriNet, formula: &LtlFormula) -> Option<Verdict> {
    let aliases = PropertyAliases::identity(net);
    let mut atom_preds = Vec::new();
    let nnf = to_nnf(formula, &mut atom_preds);
    let resolved: Vec<_> = atom_preds
        .iter()
        .map(|p| crate::buchi::resolve_atom_with_aliases(p, &aliases))
        .collect();
    try_symbolic_ltl_verdict(&nnf, net, &resolved, None)
}

#[test]
fn symbolic_lane_agrees_with_exhaustive_full_graph_zero_disagreements() {
    let mut pairs = 0usize;
    let mut decided = 0usize;
    let mut disagreements: Vec<String> = Vec::new();

    for net in nets() {
        let net_name = net.name.clone().unwrap_or_default();
        for (fname, formula) in formulas() {
            pairs += 1;
            let exhaustive = exhaustive_verdict(&net, &formula);
            let symbolic = symbolic_verdict(&net, &formula);
            match (symbolic, exhaustive) {
                // Symbolic declined: not a failure (fail-closed); the explicit
                // path would handle it.
                (None, _) => {}
                // Symbolic decided; exhaustive also decided ⇒ must agree.
                (Some(s), Some(e)) => {
                    decided += 1;
                    if s != e {
                        disagreements.push(format!(
                            "net={net_name} formula=[{fname}] symbolic={s:?} exhaustive={e:?}"
                        ));
                    }
                }
                // Symbolic decided but the exhaustive oracle could not — should
                // not happen on these tiny complete graphs; flag it so we never
                // silently trust an unvalidated symbolic verdict.
                (Some(s), None) => disagreements.push(format!(
                    "net={net_name} formula=[{fname}] symbolic={s:?} but exhaustive DECLINED \
                     (no oracle to validate against)"
                )),
            }
        }
    }

    eprintln!(
        "[ltl-symbolic-diff] {decided}/{pairs} pairs decided by the symbolic lane, 0 disagreements"
    );
    assert!(
        disagreements.is_empty(),
        "{} disagreement(s) over {pairs} pairs ({decided} symbolic-decided):\n{}",
        disagreements.len(),
        disagreements.join("\n"),
    );
    // The lane must actually FIRE on a meaningful fraction of the battery (else
    // the test is vacuous). These nets are all DD-eligible and small, so the
    // symbolic lane should decide the large majority.
    assert!(
        decided >= pairs / 2,
        "symbolic lane decided only {decided}/{pairs} — too few to be a meaningful differential",
    );
}
