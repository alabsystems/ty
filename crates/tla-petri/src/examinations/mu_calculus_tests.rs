// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for the unified mu-calculus solver
//! ([`super::mu_calculus::LocalMuSolver`]).
//!
//! Coverage:
//!
//! - Per-operator unit tests on small Petri nets:
//!   Atom, Not, And, Or, Diamond, Box, Mu (via μZ. p ∨ ◇Z = EF p),
//!   Nu (via νZ. p ∧ □Z = AG p), and the nested fixpoint
//!   νZ. μY. (p ∧ ◇Z) ∨ ◇Y is *rejected* by the alternation-free
//!   check (we assert the error path so the engine stays sound).
//! - μ-default-False / ν-default-True closure correctness on
//!   degenerate inputs (single-state nets, unresolved cycles).
//! - 20-formula differential battery against
//!   `tla_mc_core::CtlEngine` (the full-graph oracle), via the
//!   `ctl_to_mu` translation, on representative nets covering all
//!   eight CTL operators plus nesting.
//!
//! The differential battery is the soundness floor for the unified
//! solver: any mismatch indicates either a translation bug
//! (`ctl_to_mu` produced a wrong mu encoding) or a solver bug
//! (`LocalMuSolver` resolves the encoding incorrectly).

use std::collections::HashMap;

use super::ctl::resolve::{resolve_ctl_with_aliases, ResolvedCtl};
use super::mu_calculus::{ctl_to_mu, solve_local_mu, LocalMuSolver, MuAbort, MuFormula, VarId};
use crate::explorer::{explore_full, ExplorationConfig};
use crate::model::PropertyAliases;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{CtlFormula, IntExpr, StatePredicate};
use crate::resolved_predicate::{resolve_predicate_with_aliases, ResolvedPredicate};

use tla_mc_core::{build_predecessor_adjacency, CtlAtomEvaluator, CtlEngine, IndexedCtlGraph};

// ---------------------------------------------------------------------------
// Small nets, helpers
// ---------------------------------------------------------------------------

fn config() -> ExplorationConfig {
    ExplorationConfig::new(10_000)
}

/// Single deadlock state: place `p` with 0 tokens, transition `t`
/// needs `p >= 1`. No successors.
fn deadlock_net() -> PetriNet {
    PetriNet {
        name: Some("deadlock-single".to_string()),
        places: vec![PlaceInfo {
            id: "p".to_string(),
            name: None,
        }],
        transitions: vec![TransitionInfo {
            id: "t".to_string(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: vec![],
        }],
        initial_marking: vec![0],
    }
}

/// Two places p0, p1 with initial [2, 0]; transition t0 moves p0 → p1.
fn simple_net() -> PetriNet {
    PetriNet {
        name: Some("simple-2-place".to_string()),
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
        }],
        initial_marking: vec![2, 0],
    }
}

/// suffix_cycle_net: start →{to_loop, to_bad_from_start}; loop has
/// stay_loop (self-loop) and exit_bad (loop → bad); bad is a
/// deadlock.
fn suffix_cycle_net() -> PetriNet {
    PetriNet {
        name: Some("suffix-cycle".to_string()),
        places: vec![
            PlaceInfo {
                id: "start".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "loop".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "bad".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "to_loop".to_string(),
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
                id: "to_bad_from_start".to_string(),
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
                id: "stay_loop".to_string(),
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
                id: "exit_bad".to_string(),
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

/// suffix_exit_net: start → mid → bad (linear, bad is deadlock).
fn suffix_exit_net() -> PetriNet {
    PetriNet {
        name: Some("suffix-exit".to_string()),
        places: vec![
            PlaceInfo {
                id: "start".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "mid".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "bad".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "to_mid".to_string(),
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
                id: "to_bad".to_string(),
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

fn atom_ge(place: &str, value: u64) -> CtlFormula {
    CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(value),
        IntExpr::TokensCount(vec![place.to_string()]),
    ))
}

fn resolve(formula: &CtlFormula, net: &PetriNet) -> ResolvedCtl {
    let aliases = PropertyAliases::identity(net);
    resolve_ctl_with_aliases(formula, &aliases)
}

/// Resolve a `StatePredicate` as a `ResolvedPredicate` (so the
/// mu-calculus tests can build atoms directly without going through
/// `CtlFormula::Atom`).
fn resolve_pred(predicate: StatePredicate, net: &PetriNet) -> ResolvedPredicate {
    let aliases = PropertyAliases::identity(net);
    resolve_predicate_with_aliases(&predicate, &aliases)
}

fn pred_true() -> ResolvedPredicate {
    ResolvedPredicate::True
}

fn pred_false() -> ResolvedPredicate {
    ResolvedPredicate::False
}

// ---------------------------------------------------------------------------
// Per-operator unit tests on raw MuFormula
// ---------------------------------------------------------------------------

#[test]
fn test_mu_atom_true_at_initial() {
    let net = simple_net();
    // Initial marking has p0 = 2 ≥ 1.
    let predicate = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        ),
        &net,
    );
    let formula: MuFormula<ResolvedPredicate> = MuFormula::Atom(predicate);
    let verdict = solve_local_mu(&net, &formula, &config()).expect("atom should resolve");
    assert!(verdict, "p0 >= 1 holds at initial marking [2, 0]");
}

#[test]
fn test_mu_atom_false_at_initial() {
    let net = simple_net();
    let predicate = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p1".to_string()]),
        ),
        &net,
    );
    let formula: MuFormula<ResolvedPredicate> = MuFormula::Atom(predicate);
    let verdict = solve_local_mu(&net, &formula, &config()).expect("atom should resolve");
    assert!(!verdict, "p1 >= 1 fails at initial marking [2, 0]");
}

#[test]
fn test_mu_not_inverts() {
    let net = simple_net();
    let predicate = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        ),
        &net,
    );
    let formula = MuFormula::Not(Box::new(MuFormula::Atom(predicate)));
    let verdict = solve_local_mu(&net, &formula, &config()).expect("Not should resolve");
    assert!(!verdict, "Not(p0 >= 1) at initial should be FALSE");
}

#[test]
fn test_mu_and_short_circuits_false() {
    let net = simple_net();
    let p0 = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        ),
        &net,
    );
    let p1 = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p1".to_string()]),
        ),
        &net,
    );
    let formula = MuFormula::And(vec![MuFormula::Atom(p0), MuFormula::Atom(p1)]);
    let verdict = solve_local_mu(&net, &formula, &config()).expect("And should resolve");
    assert!(!verdict, "p0 >= 1 AND p1 >= 1 is FALSE initially (p1=0)");
}

#[test]
fn test_mu_or_short_circuits_true() {
    let net = simple_net();
    let p0 = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        ),
        &net,
    );
    let p1 = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p1".to_string()]),
        ),
        &net,
    );
    let formula = MuFormula::Or(vec![MuFormula::Atom(p1), MuFormula::Atom(p0)]);
    let verdict = solve_local_mu(&net, &formula, &config()).expect("Or should resolve");
    assert!(verdict, "p1 >= 1 OR p0 >= 1 is TRUE initially");
}

#[test]
fn test_mu_diamond_some_successor() {
    let net = simple_net();
    // After t0: p1 = 1. ◇(p1 ≥ 1) at initial state should be TRUE.
    let p1_ge_1 = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p1".to_string()]),
        ),
        &net,
    );
    let formula = MuFormula::Diamond(Box::new(MuFormula::Atom(p1_ge_1)));
    let verdict = solve_local_mu(&net, &formula, &config()).expect("Diamond should resolve");
    assert!(verdict, "Diamond(p1 >= 1) should be TRUE: t0 produces p1");
}

#[test]
fn test_mu_diamond_at_deadlock_is_false() {
    let net = deadlock_net();
    let formula = MuFormula::Diamond(Box::new(MuFormula::Atom(pred_true())));
    let verdict =
        solve_local_mu(&net, &formula, &config()).expect("Diamond at deadlock should resolve");
    assert!(!verdict, "Diamond(true) at deadlock must be FALSE");
}

#[test]
fn test_mu_box_at_deadlock_is_true() {
    let net = deadlock_net();
    let formula = MuFormula::Box(Box::new(MuFormula::Atom(pred_false())));
    let verdict =
        solve_local_mu(&net, &formula, &config()).expect("Box at deadlock should resolve");
    assert!(verdict, "Box(false) at deadlock must be TRUE vacuously");
}

#[test]
fn test_mu_least_fixpoint_ef_reachability() {
    let net = simple_net();
    // μZ. (p1 ≥ 1) ∨ ◇Z   ==   EF (p1 ≥ 1)
    let p1_ge_1 = resolve_pred(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p1".to_string()]),
        ),
        &net,
    );
    let z = VarId(0);
    let formula = MuFormula::Mu(
        z,
        Box::new(MuFormula::Or(vec![
            MuFormula::Atom(p1_ge_1),
            MuFormula::Diamond(Box::new(MuFormula::Var(z))),
        ])),
    );
    let verdict = solve_local_mu(&net, &formula, &config()).expect("mu (EF) should resolve");
    assert!(verdict, "EF (p1 >= 1) is TRUE: t0 produces p1");
}

#[test]
fn test_mu_least_fixpoint_defaults_to_false_on_no_witness() {
    // Pure-cycle net with no satisfying atom: μZ. False ∨ ◇Z = False
    // everywhere (no witness ever exists).
    let net = suffix_cycle_net();
    let z = VarId(0);
    let formula = MuFormula::Mu(
        z,
        Box::new(MuFormula::Or(vec![
            MuFormula::Atom(pred_false()),
            MuFormula::Diamond(Box::new(MuFormula::Var(z))),
        ])),
    );
    let verdict = solve_local_mu(&net, &formula, &config()).expect("mu should resolve");
    assert!(!verdict, "μ-default must be False on unresolved cycle");
}

#[test]
fn test_mu_greatest_fixpoint_ag_safety() {
    let net = simple_net();
    // νZ. true ∧ □Z   ==   AG true   ==   true
    let z = VarId(0);
    let formula = MuFormula::Nu(
        z,
        Box::new(MuFormula::And(vec![
            MuFormula::Atom(pred_true()),
            MuFormula::Box(Box::new(MuFormula::Var(z))),
        ])),
    );
    let verdict = solve_local_mu(&net, &formula, &config()).expect("nu (AG) should resolve");
    assert!(verdict, "AG true must be TRUE");
}

#[test]
fn test_mu_greatest_fixpoint_defaults_to_true_on_unbroken_cycle() {
    // suffix_cycle_net: loop self-loop never demotes ν; default is
    // True. νZ. true ∧ ◇Z holds on any state with at least one
    // outgoing transition leading back to a satisfying successor.
    let net = suffix_cycle_net();
    let z = VarId(0);
    let formula = MuFormula::Nu(
        z,
        Box::new(MuFormula::And(vec![
            MuFormula::Atom(pred_true()),
            MuFormula::Diamond(Box::new(MuFormula::Var(z))),
        ])),
    );
    let verdict = solve_local_mu(&net, &formula, &config()).expect("nu should resolve");
    assert!(
        verdict,
        "ν-default must keep an unbroken cycle inside the gfp"
    );
}

#[test]
fn test_mu_strict_alternation_is_rejected() {
    // Currently the alternation-free closure cannot soundly resolve
    // strict mu/nu alternation. Confirm the well-formedness gate
    // rejects νZ. μY. (◇Z ∨ ◇Y) with UnsupportedAlternation rather
    // than silently producing a possibly-unsound verdict.
    let z = VarId(0);
    let y = VarId(1);
    let formula: MuFormula<ResolvedPredicate> = MuFormula::Nu(
        z,
        Box::new(MuFormula::Mu(
            y,
            Box::new(MuFormula::Or(vec![
                MuFormula::Diamond(Box::new(MuFormula::Var(z))),
                MuFormula::Diamond(Box::new(MuFormula::Var(y))),
            ])),
        )),
    );
    let net = simple_net();
    let result = solve_local_mu(&net, &formula, &config());
    assert!(
        matches!(result, Err(MuAbort::UnsupportedAlternation)),
        "expected UnsupportedAlternation, got {result:?}"
    );
}

#[test]
fn test_mu_unbound_variable_is_rejected() {
    let z = VarId(0);
    let formula: MuFormula<ResolvedPredicate> = MuFormula::Var(z);
    let net = simple_net();
    let result = solve_local_mu(&net, &formula, &config());
    assert!(
        matches!(result, Err(MuAbort::UnboundVariable(VarId(0)))),
        "expected UnboundVariable, got {result:?}"
    );
}

#[test]
fn test_mu_negated_variable_is_rejected() {
    // μZ. Not(Z) — Z appears negatively, which violates the
    // positivity requirement of the alternation-free fragment.
    let z = VarId(0);
    let formula: MuFormula<ResolvedPredicate> =
        MuFormula::Mu(z, Box::new(MuFormula::Not(Box::new(MuFormula::Var(z)))));
    let net = simple_net();
    let result = solve_local_mu(&net, &formula, &config());
    assert!(
        matches!(result, Err(MuAbort::NegatedVariable)),
        "expected NegatedVariable, got {result:?}"
    );
}

#[test]
fn test_mu_node_cap_returns_abort() {
    let net = suffix_cycle_net();
    let z = VarId(0);
    let formula = MuFormula::Nu(
        z,
        Box::new(MuFormula::And(vec![
            MuFormula::Atom(pred_true()),
            MuFormula::Box(Box::new(MuFormula::Var(z))),
        ])),
    );

    let mut solver = LocalMuSolver::new(&net, &config()).with_node_cap(1);
    let result = solver.solve(&formula);
    assert!(
        matches!(result, Err(MuAbort::NodeCapReached)),
        "expected NodeCapReached, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// CTL-to-mu differential battery vs the full-graph oracle
// ---------------------------------------------------------------------------

/// Petri-net atom evaluator that bridges to the `tla-mc-core` engine.
struct PetriCtlAtomEval<'a> {
    net: &'a PetriNet,
}

impl<'a> CtlAtomEvaluator<Vec<u64>, ResolvedPredicate> for PetriCtlAtomEval<'a> {
    fn evaluate(&self, state: &Vec<u64>, atom: &ResolvedPredicate) -> bool {
        crate::resolved_predicate::eval_predicate(atom, state, self.net)
    }
}

/// Build the full-graph oracle. Returns the per-state satisfaction
/// vector for `formula` indexed by state id, plus the index of the
/// initial state (always 0).
///
/// Also differentially exercises the engine's early-exiting
/// `eval_root` against the exhaustive `eval` at the initial state, so
/// every oracle invocation is simultaneously an
/// eval_root-vs-full-fixpoint check (0 disagreements tolerated).
fn full_graph_eval(net: &PetriNet, formula: &ResolvedCtl) -> (Vec<bool>, usize) {
    let full = explore_full(net, &config().refitted_for_full_graph(net));
    assert!(
        full.graph.completed,
        "test net must explore fully for the oracle"
    );
    let predecessors = build_predecessor_adjacency(&full.graph.adj);
    let unpacked = full.markings.unpack_all();
    let graph = IndexedCtlGraph::new(&unpacked, &full.graph.adj, &predecessors);
    let engine = CtlEngine::new(graph, PetriCtlAtomEval { net });
    let sat = engine.eval(formula);
    let root = engine.eval_root(formula);
    assert_eq!(
        root, sat[0],
        "CtlEngine::eval_root early exit disagrees with the full fixpoint at the initial state"
    );
    (sat, 0)
}

/// Build the 21-formula differential battery over three place names.
fn differential_battery_for(p_a: &str, p_b: &str, p_c: &str) -> Vec<CtlFormula> {
    let ef = |inner| CtlFormula::EF(Box::new(inner));
    let ag = |inner| CtlFormula::AG(Box::new(inner));
    let ex = |inner| CtlFormula::EX(Box::new(inner));
    let ax = |inner| CtlFormula::AX(Box::new(inner));
    let eg = |inner| CtlFormula::EG(Box::new(inner));
    let af = |inner| CtlFormula::AF(Box::new(inner));
    let eu = |phi, psi| CtlFormula::EU(Box::new(phi), Box::new(psi));
    let au = |phi, psi| CtlFormula::AU(Box::new(phi), Box::new(psi));
    let not = |inner| CtlFormula::Not(Box::new(inner));

    let tautology = CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(0),
        IntExpr::Constant(0),
    ));
    let contradiction = CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::Constant(0),
    ));

    vec![
        // ---- atom + boolean ----
        atom_ge(p_a, 1),
        not(atom_ge(p_a, 1)),
        CtlFormula::And(vec![atom_ge(p_b, 1), atom_ge(p_a, 1)]),
        CtlFormula::Or(vec![atom_ge(p_b, 1), atom_ge(p_a, 1)]),
        // ---- EX / AX ----
        ex(atom_ge(p_b, 1)),
        ax(atom_ge(p_a, 1)),
        // ---- EF ----
        ef(atom_ge(p_a, 1)),
        ef(atom_ge(p_b, 1)),
        ef(atom_ge(p_c, 5)), // unreachable
        // ---- AG ----
        ag(tautology.clone()),
        ag(not(atom_ge(p_a, 1))),
        // ---- EG ----
        eg(not(atom_ge(p_a, 1))),
        eg(atom_ge(p_a, 1)),
        eg(tautology),
        // ---- AF ----
        af(atom_ge(p_a, 1)),
        af(CtlFormula::Or(vec![atom_ge(p_b, 1), atom_ge(p_a, 1)])),
        // ---- EU / AU ----
        eu(not(atom_ge(p_a, 1)), atom_ge(p_a, 1)),
        au(not(atom_ge(p_a, 1)), atom_ge(p_a, 1)),
        // ---- nested ----
        ag(ef(atom_ge(p_a, 1))),
        af(eg(not(atom_ge(p_a, 1)))),
        // ---- AG of a contradiction (always false on non-trivial nets) ----
        ag(contradiction),
    ]
}

/// The historical battery over the suffix-net place names.
fn differential_battery() -> Vec<CtlFormula> {
    differential_battery_for("bad", "loop", "start")
}

#[test]
fn test_unified_matches_full_graph_oracle_on_suffix_cycle() {
    let net = suffix_cycle_net();
    let formulas = differential_battery();

    for (idx, formula) in formulas.iter().enumerate() {
        let resolved = resolve(formula, &net);
        let (oracle_sat, initial_idx) = full_graph_eval(&net, &resolved);
        let oracle_verdict = oracle_sat[initial_idx];

        let mu_formula = ctl_to_mu(&resolved);
        let unified_verdict = solve_local_mu(&net, &mu_formula, &config())
            .unwrap_or_else(|e| panic!("formula #{idx}: unified solver aborted: {e:?}"));

        assert_eq!(
            unified_verdict, oracle_verdict,
            "formula #{idx} differential mismatch: unified={unified_verdict} oracle={oracle_verdict}\nformula: {formula:?}"
        );
    }
}

#[test]
fn test_unified_matches_full_graph_oracle_on_suffix_exit() {
    let net = suffix_exit_net();
    let formulas = differential_battery();

    for (idx, formula) in formulas.iter().enumerate() {
        let resolved = resolve(formula, &net);
        let (oracle_sat, initial_idx) = full_graph_eval(&net, &resolved);
        let oracle_verdict = oracle_sat[initial_idx];

        let mu_formula = ctl_to_mu(&resolved);
        let unified_verdict = solve_local_mu(&net, &mu_formula, &config())
            .unwrap_or_else(|e| panic!("formula #{idx}: unified solver aborted: {e:?}"));

        assert_eq!(
            unified_verdict, oracle_verdict,
            "formula #{idx} (suffix_exit) differential mismatch: unified={unified_verdict} oracle={oracle_verdict}\nformula: {formula:?}"
        );
    }
}

#[test]
fn test_unified_matches_full_graph_oracle_on_simple_net() {
    let net = simple_net();

    let formulas = [
        CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        )),
        CtlFormula::EF(Box::new(atom_ge("p1", 1))),
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(0),
        )))),
        CtlFormula::AF(Box::new(atom_ge("p1", 1))),
        CtlFormula::EX(Box::new(atom_ge("p1", 1))),
        CtlFormula::AX(Box::new(atom_ge("p1", 1))),
    ];

    for (idx, formula) in formulas.iter().enumerate() {
        let resolved = resolve(formula, &net);
        let (oracle_sat, initial_idx) = full_graph_eval(&net, &resolved);
        let oracle_verdict = oracle_sat[initial_idx];

        let mu_formula = ctl_to_mu(&resolved);
        let unified_verdict = solve_local_mu(&net, &mu_formula, &config())
            .unwrap_or_else(|e| panic!("simple formula #{idx}: unified aborted: {e:?}"));

        assert_eq!(
            unified_verdict, oracle_verdict,
            "simple formula #{idx}: unified={unified_verdict} oracle={oracle_verdict}"
        );
    }
}

// ---------------------------------------------------------------------------
// Nested-fixpoint closure soundness regression
// ---------------------------------------------------------------------------

/// Net for the nested-fixpoint closure regression:
///
/// ```text
///   a --t1--> b           (b is a deadlock; the "good" state)
///   a --t2--> c --t3--> c (self-loop trap; b unreachable from c)
/// ```
fn nested_fixpoint_trap_net() -> PetriNet {
    let arc = |place: u32| Arc {
        place: PlaceIdx(place),
        weight: 1,
    };
    PetriNet {
        name: Some("nested-fixpoint-trap".to_string()),
        places: ["a", "b", "c"]
            .iter()
            .map(|id| PlaceInfo {
                id: (*id).to_string(),
                name: None,
            })
            .collect(),
        transitions: vec![
            TransitionInfo {
                id: "t1".to_string(),
                name: None,
                inputs: vec![arc(0)],
                outputs: vec![arc(1)],
            },
            TransitionInfo {
                id: "t2".to_string(),
                name: None,
                inputs: vec![arc(0)],
                outputs: vec![arc(2)],
            },
            TransitionInfo {
                id: "t3".to_string(),
                name: None,
                inputs: vec![arc(2)],
                outputs: vec![arc(2)],
            },
        ],
        initial_marking: vec![1, 0, 0],
    }
}

/// Regression: `AG(EF p)` is `νZ. (μY. p ∨ ◇Y) ∧ □Z` — a μ component
/// nested inside a ν component. On `nested_fixpoint_trap_net` with
/// `p = (b >= 1)`:
///
/// - `EF p` is True at the initial state (fire t1) but False at the
///   trap state c (only t3 loops; b is unreachable), so `AG(EF p)`
///   must be **False**.
/// - The worklist drains with the root Unknown (both the inner μ and
///   the outer ν spin on the c self-loop), so the verdict is decided
///   entirely by the closure.
///
/// A closure that defaults μ-Vars (False) and ν-Vars (True)
/// *simultaneously* freezes the outer `Var(Z)@c` at True before the
/// inner refutation (`EF p` False at c) has propagated, and returns
/// the unsound True (this exact bug shipped until the component-
/// staged closure landed; marks are write-once so the wrong default
/// can never be corrected). The staged closure defaults the inner μ
/// component first, derives `EF p = False` at c, back-propagates it
/// into `Var(Z)@c` via Var-inherits-body, and correctly refutes the
/// root. The dual nesting (`EF(AG p)`, ν inside μ) is exercised by
/// the same battery below and by the randomized differential test.
#[test]
fn test_closure_nested_mu_inside_nu_is_sound() {
    let net = nested_fixpoint_trap_net();

    let ag_ef = CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(atom_ge("b", 1)))));
    let ef_ag = CtlFormula::EF(Box::new(CtlFormula::AG(Box::new(atom_ge("c", 1)))));

    for (formula, expected) in [(ag_ef, false), (ef_ag, true)] {
        let resolved = resolve(&formula, &net);
        let (oracle_sat, initial_idx) = full_graph_eval(&net, &resolved);
        assert_eq!(
            oracle_sat[initial_idx], expected,
            "oracle disagrees with the hand-computed expectation for {formula:?}"
        );

        let mu_formula = ctl_to_mu(&resolved);
        let unified = solve_local_mu(&net, &mu_formula, &config())
            .unwrap_or_else(|e| panic!("unified solver aborted: {e:?}"));
        assert_eq!(
            unified, expected,
            "nested-fixpoint closure produced a wrong verdict for {formula:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Randomized bounded-net differential battery: the certain-zero EDG
// (worklist + dependency-driven closure + root early-exit, in
// whichever combination fires for each formula) must agree with the
// exhaustive full-graph CtlEngine on every verdict. 0 disagreements.
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Deterministic random Petri net with a *non-increasing token count*
/// (every transition's total output weight <= total input weight, all
/// weights 1), which bounds the reachable state space, so the
/// full-graph oracle always completes. The construction produces
/// deadlocks, self-loops, token-conserving cycles, and concurrent
/// branches — the shapes that drive both the early-exit worklist path
/// and the exhaustion-shaped closure path of the EDG.
fn random_bounded_net(seed: u64) -> PetriNet {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let num_places = 2 + (lcg_next(&mut s) % 3) as usize; // 2..=4
    let num_transitions = 2 + (lcg_next(&mut s) % 4) as usize; // 2..=5

    let places: Vec<PlaceInfo> = (0..num_places)
        .map(|i| PlaceInfo {
            id: format!("p{i}"),
            name: None,
        })
        .collect();

    let mut transitions = Vec::new();
    for t in 0..num_transitions {
        let in_count = 1 + (lcg_next(&mut s) % 2) as usize; // 1..=2
        let mut input_places: Vec<usize> = Vec::new();
        while input_places.len() < in_count {
            let p = (lcg_next(&mut s) % num_places as u64) as usize;
            if !input_places.contains(&p) {
                input_places.push(p);
            }
        }
        // outputs: at most as many arcs as inputs (token-conserving
        // or token-consuming), possibly zero (a pure sink).
        let out_count = (lcg_next(&mut s) % (in_count as u64 + 1)) as usize;
        let mut output_places: Vec<usize> = Vec::new();
        while output_places.len() < out_count {
            let p = (lcg_next(&mut s) % num_places as u64) as usize;
            if !output_places.contains(&p) {
                output_places.push(p);
            }
        }
        transitions.push(TransitionInfo {
            id: format!("t{t}"),
            name: None,
            inputs: input_places
                .into_iter()
                .map(|p| Arc {
                    place: PlaceIdx(p as u32),
                    weight: 1,
                })
                .collect(),
            outputs: output_places
                .into_iter()
                .map(|p| Arc {
                    place: PlaceIdx(p as u32),
                    weight: 1,
                })
                .collect(),
        });
    }

    let initial_marking: Vec<u64> = (0..num_places).map(|_| lcg_next(&mut s) % 3).collect();

    PetriNet {
        name: Some(format!("random-bounded-{seed}")),
        places,
        transitions,
        initial_marking,
    }
}

#[test]
fn test_unified_matches_full_graph_oracle_on_random_bounded_nets() {
    let mut checked = 0usize;
    for seed in 0..40u64 {
        let net = random_bounded_net(seed);
        let last_place = format!("p{}", net.places.len() - 1);
        let formulas = differential_battery_for("p0", "p1", &last_place);

        for (idx, formula) in formulas.iter().enumerate() {
            let resolved = resolve(formula, &net);
            let (oracle_sat, initial_idx) = full_graph_eval(&net, &resolved);
            let oracle_verdict = oracle_sat[initial_idx];

            let mu_formula = ctl_to_mu(&resolved);
            let unified_verdict =
                solve_local_mu(&net, &mu_formula, &config()).unwrap_or_else(|e| {
                    panic!("seed {seed} formula #{idx}: unified solver aborted: {e:?}")
                });

            assert_eq!(
                unified_verdict, oracle_verdict,
                "seed {seed} formula #{idx} differential mismatch: \
                 unified={unified_verdict} oracle={oracle_verdict}\nformula: {formula:?}\nnet: {net:?}"
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 800,
        "differential battery too small: {checked} checks"
    );
    eprintln!("random-net differential: {checked} checks, 0 disagreements");
}

// ---------------------------------------------------------------------------
// Deadlock-specific CTL-to-mu translation correctness tests
// ---------------------------------------------------------------------------

#[test]
fn test_ctl_to_mu_eg_at_deadlock_with_predicate_true() {
    // EG(true) at deadlock should be TRUE (MCC max-path: deadlock
    // state belongs to the gfp). Verify the encoded
    // νZ. true ∧ (◇Z ∨ ¬◇true) form computes TRUE.
    let net = deadlock_net();
    let formula = CtlFormula::EG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(0),
        IntExpr::Constant(0),
    ))));
    let resolved = resolve(&formula, &net);
    let mu_formula = ctl_to_mu(&resolved);
    let verdict = solve_local_mu(&net, &mu_formula, &config()).expect("EG at deadlock resolves");
    assert!(
        verdict,
        "EG(true) at deadlock must be TRUE under max-path semantics"
    );
}

#[test]
fn test_ctl_to_mu_eg_at_deadlock_with_predicate_false() {
    // EG(false) at deadlock should be FALSE (no successor, predicate
    // fails locally).
    let net = deadlock_net();
    let formula = CtlFormula::EG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::Constant(0),
    ))));
    let resolved = resolve(&formula, &net);
    let mu_formula = ctl_to_mu(&resolved);
    let verdict = solve_local_mu(&net, &mu_formula, &config()).expect("EG at deadlock resolves");
    assert!(!verdict, "EG(false) at deadlock must be FALSE");
}

#[test]
fn test_ctl_to_mu_af_at_deadlock_with_predicate_true() {
    // AF(true) at deadlock should be TRUE.
    let net = deadlock_net();
    let formula = CtlFormula::AF(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(0),
        IntExpr::Constant(0),
    ))));
    let resolved = resolve(&formula, &net);
    let mu_formula = ctl_to_mu(&resolved);
    let verdict = solve_local_mu(&net, &mu_formula, &config()).expect("AF at deadlock resolves");
    assert!(verdict, "AF(true) at deadlock holds locally");
}

#[test]
fn test_ctl_to_mu_af_at_deadlock_with_predicate_false() {
    // AF(false) at deadlock should be FALSE — no path ever reaches
    // the impossible predicate.
    let net = deadlock_net();
    let formula = CtlFormula::AF(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::Constant(0),
    ))));
    let resolved = resolve(&formula, &net);
    let mu_formula = ctl_to_mu(&resolved);
    let verdict = solve_local_mu(&net, &mu_formula, &config()).expect("AF at deadlock resolves");
    assert!(!verdict, "AF(false) at deadlock must be FALSE");
}

#[test]
fn test_ctl_to_mu_ax_at_deadlock_is_true() {
    let net = deadlock_net();
    let formula = CtlFormula::AX(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::Constant(0),
    ))));
    let resolved = resolve(&formula, &net);
    let mu_formula = ctl_to_mu(&resolved);
    let verdict = solve_local_mu(&net, &mu_formula, &config()).expect("AX at deadlock resolves");
    assert!(verdict, "AX at deadlock is vacuously TRUE");
}

#[test]
fn test_ctl_to_mu_ex_at_deadlock_is_false() {
    let net = deadlock_net();
    let formula = CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(0),
        IntExpr::Constant(0),
    ))));
    let resolved = resolve(&formula, &net);
    let mu_formula = ctl_to_mu(&resolved);
    let verdict = solve_local_mu(&net, &mu_formula, &config()).expect("EX at deadlock resolves");
    assert!(!verdict, "EX at deadlock is FALSE (no successor)");
}

// ---------------------------------------------------------------------------
// Cross-formula sanity (ctl_to_mu produces the same verdict as the
// legacy direct EDG)
// ---------------------------------------------------------------------------

#[test]
fn test_ctl_to_mu_alpha_renames_uniquely() {
    // The variable counter should produce unique VarIds across an
    // EF-of-AG nesting. Build EF(AG(p)) and inspect the produced
    // MuFormula structure.
    let p = CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(0),
        IntExpr::Constant(0),
    ));
    let formula = CtlFormula::EF(Box::new(CtlFormula::AG(Box::new(p))));
    let net = simple_net();
    let resolved = resolve(&formula, &net);
    let mu = ctl_to_mu(&resolved);

    // Outer μZ. ... ∨ ◇Z, inner νZ'. p ∧ □Z'. Confirm we have two
    // distinct VarIds — the test will assert the structure rather
    // than the specific id values to stay robust to encoding
    // changes.
    fn collect_var_ids<A>(formula: &MuFormula<A>, ids: &mut HashMap<VarId, usize>) {
        match formula {
            MuFormula::Atom(_) => {}
            MuFormula::Var(v) => {
                *ids.entry(*v).or_insert(0) += 1;
            }
            MuFormula::Not(inner) | MuFormula::Diamond(inner) | MuFormula::Box(inner) => {
                collect_var_ids(inner, ids);
            }
            MuFormula::And(children) | MuFormula::Or(children) => {
                for c in children {
                    collect_var_ids(c, ids);
                }
            }
            MuFormula::Mu(v, body) | MuFormula::Nu(v, body) => {
                ids.entry(*v).or_insert(0);
                collect_var_ids(body, ids);
            }
        }
    }
    let mut ids = HashMap::new();
    collect_var_ids(&mu, &mut ids);
    assert!(
        ids.len() >= 2,
        "EF(AG(_)) should bind at least 2 distinct VarIds, got {ids:?}"
    );
}
