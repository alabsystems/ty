// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::local_edg::solve_local_edg;
use super::super::resolve::resolve_ctl_with_aliases;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::super::pipeline::{
    ctl_full_graph_deadline_for_test, pending_expensive_ctl_properties_for_test,
    suffix_ctl_properties_for_test, with_ctl_local_fallback_for_test,
    with_experimental_ctl_shortcuts_for_test,
};
use super::super::{
    check_ctl_properties, check_ctl_properties_with_flush, classify_shallow_ctl,
    classify_shallow_ctl_suffix, ctl_batch_contains_next_step, ctl_formula_contains_next_step,
};
use super::support::{
    atom_pred, check_ctl_properties_unsliced, component_a_ctl_props, ctl_budget_ag_fireable,
    ctl_budget_cycle_net, ctl_local_fallback_formula, ctl_local_fallback_net,
    disconnected_two_component_net, finite_ctl_local_fallback_net, make_ctl_prop, simple_net,
    suffix_cycle_net, suffix_exit_net,
};

use crate::explorer::{explore_full, ExplorationConfig};
use crate::model::PropertyAliases;
use crate::output::Verdict;
use crate::parser::parse_pnml_dir;
use crate::property_xml::{parse_properties, CtlFormula, Formula, IntExpr, StatePredicate};

/// The MCC Kanban net (4 stages, 16 places, 16 transitions), parameterised by
/// the number of tokens `n` initially in each of P1..P4. This is the exact
/// structure of the corpus `Kanban-PT-*` models; only the initial marking
/// scales. It is live and reversible, so `AG(EF(fireable t))` is TRUE for
/// every reachable state at every scale.
fn kanban_net(n: u64) -> crate::petri_net::PetriNet {
    use crate::petri_net::{Arc, PetriNet, PlaceInfo, TransitionInfo};
    let place_names = [
        "P1", "Pm1", "Pout1", "Pback1", "P2", "Pm2", "Pout2", "Pback2", "P3", "Pm3", "Pout3",
        "Pback3", "P4", "Pm4", "Pout4", "Pback4",
    ];
    let idx = |name: &str| {
        place_names
            .iter()
            .position(|p| *p == name)
            .expect("place exists") as u32
    };
    let places = place_names
        .iter()
        .map(|name| PlaceInfo {
            id: (*name).to_string(),
            name: None,
        })
        .collect();
    // (transition-id, input places, output places) — the MCC Kanban arcs.
    let spec: &[(&str, &[&str], &[&str])] = &[
        ("tback3", &["Pback3"], &["Pm3"]),
        ("tredo3", &["Pm3"], &["Pback3"]),
        ("tredo2", &["Pm2"], &["Pback2"]),
        ("tok3", &["Pm3"], &["Pout3"]),
        ("tredo4", &["Pm4"], &["Pback4"]),
        ("tin4", &["P4"], &["Pm4"]),
        ("tok4", &["Pm4"], &["Pout4"]),
        ("tback4", &["Pback4"], &["Pm4"]),
        (
            "tsynch1_23",
            &["Pout2", "Pout3", "P1"],
            &["P3", "Pm1", "P2"],
        ),
        ("tout1", &["Pout1"], &["P1"]),
        ("tok1", &["Pm1"], &["Pout1"]),
        ("tsynch4_23", &["P2", "Pout4", "P3"], &["P4", "Pm3", "Pm2"]),
        ("tredo1", &["Pm1"], &["Pback1"]),
        ("tback1", &["Pback1"], &["Pm1"]),
        ("tback2", &["Pback2"], &["Pm2"]),
        ("tok2", &["Pm2"], &["Pout2"]),
    ];
    let transitions = spec
        .iter()
        .map(|(id, ins, outs)| TransitionInfo {
            id: (*id).to_string(),
            name: None,
            inputs: ins
                .iter()
                .map(|p| Arc {
                    place: crate::petri_net::PlaceIdx(idx(p)),
                    weight: 1,
                })
                .collect(),
            outputs: outs
                .iter()
                .map(|p| Arc {
                    place: crate::petri_net::PlaceIdx(idx(p)),
                    weight: 1,
                })
                .collect(),
        })
        .collect();
    let mut initial_marking = vec![0u64; place_names.len()];
    for p in ["P1", "P2", "P3", "P4"] {
        initial_marking[idx(p) as usize] = n;
    }
    PetriNet {
        name: Some(format!("kanban-{n}")),
        places,
        transitions,
        initial_marking,
    }
}

/// Ground-truth pin: on the small fully-explorable `kanban-2` net (4600
/// reachable states) the exhaustive full-graph `CtlChecker` proves
/// `AG(EF(fireable tredo4)) = TRUE` (Kanban is live + reversible — `tredo4` can
/// always be re-enabled). The structure, and hence this verdict, is
/// scale-invariant; this is the scale-down of the corpus
/// `Kanban-PT-00200-CTLFireability-2025-05` whose TY verdict regressed to FALSE.
#[test]
fn test_kanban_ag_ef_fireable_is_true_exhaustively() {
    use super::super::checker::CtlChecker;
    use super::super::resolve::resolve_ctl;
    use crate::petri_net::{PlaceIdx, TransitionIdx};
    use std::collections::HashMap;

    let net = kanban_net(2);
    let place_map: HashMap<&str, PlaceIdx> = net
        .places
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id.as_str(), PlaceIdx(i as u32)))
        .collect();
    let trans_map: HashMap<&str, TransitionIdx> = net
        .transitions
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.as_str(), TransitionIdx(i as u32)))
        .collect();

    let formula = CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(CtlFormula::Atom(
        StatePredicate::IsFireable(vec![String::from("tredo4")]),
    )))));

    let full = explore_full(&net, &ExplorationConfig::new(1_000_000));
    assert!(full.graph.completed, "kanban-2 full graph must complete");
    let checker = CtlChecker::new(&full, &net);
    let resolved = resolve_ctl(&formula, &place_map, &trans_map);
    assert!(
        checker.eval(&resolved)[0],
        "AG(EF(fireable tredo4)) must be TRUE on kanban-2 (live + reversible net)"
    );
}

/// REGRESSION (Kanban-PT-00200 CTLFireability-2025-05 / CTLCardinality-2025-09):
/// when the batch full graph cannot complete, the CTL pipeline must NEVER emit
/// a wrong definite FALSE for a nested-fixpoint formula. The recursive
/// `LocalCtlChecker` fallback was producing exactly that — a confident
/// `Ok(false)` for `AG(EF(fireable tredo4))` (true answer TRUE) after interning
/// only a few hundred of the reachable states, because its single-pass DFS with
/// per-node cycle assumptions and write-once `Ready` memoization is not a valid
/// fixpoint iteration for alternating μ/ν. The fix routes an EDG abort straight
/// to CANNOT_COMPUTE. Under a tight `max_states` that truncates the full graph,
/// the verdict must be TRUE (if the sound EDG decides it) or CANNOT_COMPUTE —
/// FALSE would be the catastrophic regression.
#[test]
fn test_kanban_nested_fixpoint_never_wrongly_false_on_incomplete_graph() {
    let formula = CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(CtlFormula::Atom(
        StatePredicate::IsFireable(vec![String::from("tredo4")]),
    )))));
    let props = vec![make_ctl_prop("kanban-agef-tredo4", formula)];

    // Tight state budget forces the batch full graph to truncate, engaging the
    // post-full-graph local fallback path that previously returned a wrong
    // FALSE via LocalCtlChecker.
    let net = kanban_net(20);
    let results = with_ctl_local_fallback_for_test(true, || {
        check_ctl_properties(&net, &props, &ExplorationConfig::new(2000))
    });

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "kanban-agef-tredo4");
    assert!(
        matches!(results[0].1, Verdict::True | Verdict::CannotCompute),
        "kanban AG(EF(fireable tredo4)) on a truncated graph must be TRUE or \
         CANNOT_COMPUTE, never FALSE; got {:?} (LocalCtlChecker nested-fixpoint \
         unsoundness regression)",
        results[0].1
    );
}

/// Pins the μ/ν alternation classifier that gates the recursive
/// `LocalCtlChecker` fallback (the soundness fix for
/// Kanban-PT-00200-CTL*-2025-0{5,9}).
#[test]
fn test_ctl_alternation_classifier() {
    use super::super::pipeline::ctl_is_alternation_free_for_test as alt_free;
    use super::super::resolve::resolve_ctl_with_aliases;

    let net = ctl_budget_cycle_net();
    let aliases = PropertyAliases::identity(&net);
    let p = || CtlFormula::Atom(atom_pred("p0", 1));
    let resolve = |f: &CtlFormula| resolve_ctl_with_aliases(f, &aliases);

    // Alternation-FREE: single fixpoint class, or only EX/AX/boolean nesting.
    let ef = CtlFormula::EF(Box::new(p()));
    let ef_ef = CtlFormula::EF(Box::new(CtlFormula::EF(Box::new(p()))));
    let ex_ef = CtlFormula::EX(Box::new(CtlFormula::EF(Box::new(p()))));
    let ag_ag = CtlFormula::AG(Box::new(CtlFormula::AG(Box::new(p()))));
    assert!(alt_free(&resolve(&ef)), "EF p is alternation-free");
    assert!(
        alt_free(&resolve(&ef_ef)),
        "EF(EF p) is alternation-free (μ over μ)"
    );
    assert!(alt_free(&resolve(&ex_ef)), "EX(EF p) is alternation-free");
    assert!(
        alt_free(&resolve(&ag_ag)),
        "AG(AG p) is alternation-free (ν over ν)"
    );

    // ALTERNATING (μ/ν nesting): the LocalCtlChecker is unsound here.
    let ag_ef = CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(p())))); // ν⊃μ
    let ef_ag = CtlFormula::EF(Box::new(CtlFormula::AG(Box::new(p())))); // μ⊃ν
    let not_ef_ag = CtlFormula::Not(Box::new(ef_ag.clone())); // == Kanban-2025-09 shape
    assert!(
        !alt_free(&resolve(&ag_ef)),
        "AG(EF p) alternates (CTLFireability-2025-05)"
    );
    assert!(!alt_free(&resolve(&ef_ag)), "EF(AG p) alternates");
    assert!(
        !alt_free(&resolve(&not_ef_ag)),
        "NOT(EF(AG p)) alternates (CTLCardinality-2025-09)"
    );
}

/// ADJUDICATION (corpus-vs-consensus): exhaustive ground truth for the two
/// disputed Kanban-PT-00200 formulas, computed on the scale-down nets where the
/// full graph completes. Establishes the consensus verdicts (both TRUE) are
/// correct and that TY's FALSE was the bug. The Kanban structure is live +
/// reversible, so these CTL verdicts are scale-invariant; we confirm that by
/// re-deciding at N=2,3,4,5 and asserting the verdict is constant.
#[test]
fn test_kanban_disputed_formulas_exhaustive_ground_truth() {
    use super::super::checker::CtlChecker;
    use super::super::resolve::resolve_ctl;
    use crate::petri_net::{PlaceIdx, TransitionIdx};
    use std::collections::HashMap;

    // T2 (CTLFireability-2025-05): AG(EF(is-fireable(tredo4)))
    let t2 = CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(CtlFormula::Atom(
        StatePredicate::IsFireable(vec![String::from("tredo4")]),
    )))));
    // T1 (CTLCardinality-2025-09): NOT(EF(AG(P4 <= Pout3)))
    let p4_le_pout3 = CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::TokensCount(vec![String::from("P4")]),
        IntExpr::TokensCount(vec![String::from("Pout3")]),
    ));
    let inner_ef_ag = CtlFormula::EF(Box::new(CtlFormula::AG(Box::new(p4_le_pout3.clone()))));
    let t1 = CtlFormula::Not(Box::new(inner_ef_ag.clone()));

    let mut t2_verdicts = Vec::new();
    let mut t1_verdicts = Vec::new();
    let mut inner_verdicts = Vec::new();

    for n in [2u64, 3, 4, 5] {
        let net = kanban_net(n);
        let place_map: HashMap<&str, PlaceIdx> = net
            .places
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id.as_str(), PlaceIdx(i as u32)))
            .collect();
        let trans_map: HashMap<&str, TransitionIdx> = net
            .transitions
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id.as_str(), TransitionIdx(i as u32)))
            .collect();

        let full = explore_full(&net, &ExplorationConfig::new(50_000_000));
        assert!(
            full.graph.completed,
            "kanban-{n} full graph must complete for ground truth"
        );
        let checker = CtlChecker::new(&full, &net);

        let r_t2 = checker.eval(&resolve_ctl(&t2, &place_map, &trans_map))[0];
        let r_t1 = checker.eval(&resolve_ctl(&t1, &place_map, &trans_map))[0];
        let r_inner = checker.eval(&resolve_ctl(&inner_ef_ag, &place_map, &trans_map))[0];
        eprintln!(
            "kanban-{n}: states={} | AG(EF fireable tredo4)={r_t2} | \
             NOT(EF(AG(P4<=Pout3)))={r_t1} | inner EF(AG(P4<=Pout3))={r_inner}",
            full.markings.len()
        );
        t2_verdicts.push(r_t2);
        t1_verdicts.push(r_t1);
        inner_verdicts.push(r_inner);
    }

    // T2 consensus = TRUE; must hold at every scale (scale-invariant).
    assert!(
        t2_verdicts.iter().all(|&v| v),
        "AG(EF(fireable tredo4)) must be TRUE at every Kanban scale \
         (consensus=TRUE); got {t2_verdicts:?}"
    );
    // T1 consensus = TRUE; must hold at every scale (scale-invariant).
    assert!(
        t1_verdicts.iter().all(|&v| v),
        "NOT(EF(AG(P4<=Pout3))) must be TRUE at every Kanban scale \
         (consensus=TRUE); got {t1_verdicts:?}"
    );
    // The inner witness: T1=TRUE iff EF(AG(P4<=Pout3))=FALSE, i.e. no reachable
    // state has a future from which P4<=Pout3 holds forever.
    assert!(
        inner_verdicts.iter().all(|&v| !v),
        "EF(AG(P4<=Pout3)) must be FALSE at every scale (so its negation T1 is \
         TRUE); got {inner_verdicts:?}"
    );
}

/// ADVERSARIAL (mechanism proof): the unsound `LocalCtlChecker` returns a
/// confident WRONG verdict for BOTH disputed alternating formulas on a Kanban
/// net large enough that its single-pass DFS terminates via a cycle assumption
/// before the exhaustive truth is established. This pins the root cause to the
/// recursive local checker (NOT structural reduction) and justifies the
/// alternation gate. Ground truth (proven exhaustively above):
///   T2 AG(EF(fireable tredo4)) = TRUE   — checker must NOT return Ok(false)
///   T1 NOT(EF(AG(P4<=Pout3)))  = TRUE   — checker must NOT return Ok(false)
/// If the checker returns Ok(_), a `false` is the catastrophic wrong verdict the
/// gate exists to suppress; we assert that whenever it answers definitely, the
/// answer is wrong here — documenting exactly why the path is untrusted.
#[test]
fn test_local_checker_is_unsound_on_both_disputed_kanban_formulas() {
    use super::super::local_checker::LocalCtlChecker;
    use super::super::resolve::resolve_ctl;
    use crate::petri_net::{PlaceIdx, TransitionIdx};
    use std::collections::HashMap;

    let t2 = CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(CtlFormula::Atom(
        StatePredicate::IsFireable(vec![String::from("tredo4")]),
    )))));
    let t1 = CtlFormula::Not(Box::new(CtlFormula::EF(Box::new(CtlFormula::AG(
        Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec![String::from("P4")]),
            IntExpr::TokensCount(vec![String::from("Pout3")]),
        ))),
    )))));

    // Large scale: the reachable space (≈ tens of millions of states at N=20)
    // dwarfs any local budget, so the single-pass DFS commits an inner fixpoint
    // under a transient outer cycle assumption and halts with a wrong verdict.
    let net = kanban_net(20);
    let place_map: HashMap<&str, PlaceIdx> = net
        .places
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id.as_str(), PlaceIdx(i as u32)))
        .collect();
    let trans_map: HashMap<&str, TransitionIdx> = net
        .transitions
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.as_str(), TransitionIdx(i as u32)))
        .collect();

    let cfg = ExplorationConfig::new(5_000_000);
    for (label, formula, ground_truth) in [
        ("T2 AG(EF fireable)", &t2, true),
        ("T1 NOT(EF(AG))", &t1, true),
    ] {
        let resolved = resolve_ctl(formula, &place_map, &trans_map);
        let mut checker = LocalCtlChecker::new(&net, &cfg);
        match checker.eval_root(&resolved) {
            Ok(v) => {
                eprintln!(
                    "LocalCtlChecker {label}: returned Ok({v}) (ground truth={ground_truth})"
                );
                assert_ne!(
                    v, ground_truth,
                    "{label}: LocalCtlChecker returned the CORRECT answer here — if it \
                     became sound the alternation gate could be loosened, but the \
                     documented bug is that it returns the WRONG answer"
                );
            }
            Err(e) => {
                // Aborting (budget/deadline) is acceptable — that routes to
                // CANNOT_COMPUTE, never a wrong verdict.
                eprintln!("LocalCtlChecker {label}: aborted ({e}) — safe (CANNOT_COMPUTE)");
            }
        }
    }
}

fn mcc_benchmark_dir(model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("benchmarks")
        .join("mcc")
        .join("2024")
        .join("INPUTS")
        .join(model)
}

#[test]
fn test_ctl_full_graph_deadline_takes_majority_share_over_local_fallback() {
    let now = Instant::now();
    let deadline = now + Duration::from_secs(16);
    let full_graph = ctl_full_graph_deadline_for_test(Some(deadline), 16, true, now)
        .expect("finite global deadline should produce full-graph deadline");

    // The full-graph path is the all-properties-at-once exact oracle and
    // returns early when the net is enumerable, so it is given the majority of
    // the remaining budget (75%) regardless of the property count; the residual
    // is reserved for the per-property local fallback. Previously it was starved
    // to remaining/(unresolved+1) (1s here), which forced enumerable nets
    // through the slower fallback and produced spurious CANNOT_COMPUTE.
    assert_eq!(
        full_graph.saturating_duration_since(now),
        Duration::from_secs(12),
        "full-graph CTL should receive the 75% majority share (12s of 16s), not a 1/(N+1) lane"
    );
}

#[test]
fn test_ctl_full_graph_deadline_uses_global_deadline_without_local_fallback() {
    let now = Instant::now();
    let deadline = now + Duration::from_secs(17);

    assert_eq!(
        ctl_full_graph_deadline_for_test(Some(deadline), 16, false, now),
        Some(deadline),
        "without local fallback, full graph remains the only complete CTL lane"
    );
    assert_eq!(
        ctl_full_graph_deadline_for_test(None, 16, true, now),
        None,
        "unbounded CTL runs should remain unbounded"
    );
}

#[test]
fn test_ctl_budget_boundary_sufficient_resolves() {
    let props = vec![make_ctl_prop("ctl-budget-ok", ctl_budget_ag_fireable())];
    let results = check_ctl_properties(&ctl_budget_cycle_net(), &props, &ExplorationConfig::new(2));
    assert_eq!(
        results,
        vec![(String::from("ctl-budget-ok"), Verdict::True)]
    );
}

#[test]
fn test_ctl_budget_boundary_tight_fails_closed_or_resolves_by_shallow_routing() {
    // AG(is-fireable(t0 ∨ t1)) on a 2-state cycle is always TRUE.
    // In environments without ay, the shallow reachability route can remain
    // inconclusive under this tiny budget. Either an exact TRUE or a
    // fail-closed CANNOT_COMPUTE is acceptable; FALSE would be unsound.
    let props = vec![make_ctl_prop("ctl-budget-tight", ctl_budget_ag_fireable())];
    let results = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties(&ctl_budget_cycle_net(), &props, &ExplorationConfig::new(1))
    });
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "ctl-budget-tight");
    assert!(matches!(
        results[0].1,
        Verdict::True | Verdict::CannotCompute
    ));
}

#[test]
fn test_ctl_formula_contains_next_step_unit() {
    // EX and AX are immediate-successor operators -> true.
    assert!(ctl_formula_contains_next_step(&CtlFormula::EX(Box::new(
        CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(1),
        )),
    ))));
    assert!(ctl_formula_contains_next_step(&CtlFormula::AX(Box::new(
        CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(1),
        )),
    ))));

    // EF, AF, EG, AG, EU, AU without EX/AX -> false.
    let atom = CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(0),
        IntExpr::Constant(1),
    ));
    assert!(!ctl_formula_contains_next_step(&CtlFormula::EF(Box::new(
        atom.clone()
    ))));
    assert!(!ctl_formula_contains_next_step(&CtlFormula::AG(Box::new(
        atom.clone()
    ))));
    assert!(!ctl_formula_contains_next_step(&CtlFormula::EU(
        Box::new(atom.clone()),
        Box::new(atom.clone()),
    )));

    // Nested EX inside AG -> true.
    assert!(ctl_formula_contains_next_step(&CtlFormula::AG(Box::new(
        CtlFormula::EX(Box::new(atom.clone())),
    ))));
}

#[test]
fn test_ctl_batch_gate_detects_ex_in_mixed_batch() {
    let safe_prop = make_ctl_prop(
        "safe",
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(1),
        )))),
    );
    let ex_prop = make_ctl_prop(
        "has-ex",
        CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(1),
        )))),
    );

    // Pure stutter-insensitive batch -> eligible for deep slice.
    assert!(!ctl_batch_contains_next_step(std::slice::from_ref(
        &safe_prop
    )));

    // Mixed batch with EX -> not eligible.
    assert!(ctl_batch_contains_next_step(&[safe_prop, ex_prop]));
}

#[test]
fn test_ctl_batch_gate_ignores_already_qualified_ex_properties() {
    let next_free_prop = make_ctl_prop(
        "pending-next-free",
        CtlFormula::EU(
            Box::new(CtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::Constant(0),
                IntExpr::Constant(1),
            ))),
            Box::new(CtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::Constant(0),
                IntExpr::Constant(1),
            ))),
        ),
    );
    let skipped_ex_prop = make_ctl_prop(
        "skipped-ex",
        CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(1),
        )))),
    );

    let pending = pending_expensive_ctl_properties_for_test(
        &[next_free_prop.clone(), skipped_ex_prop],
        &["skipped-ex"],
    );

    assert_eq!(pending, vec![next_free_prop]);
    assert!(
        !ctl_batch_contains_next_step(&pending),
        "already-qualified EX properties must not force conservative routing for pending next-free CTL"
    );
}

#[test]
fn test_ctl_suffix_candidates_ignore_non_suffix_batch_properties() {
    let suffix_prop = make_ctl_prop(
        "suffix-af",
        CtlFormula::AF(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(1),
        )))),
    );
    let next_step_prop = make_ctl_prop(
        "next-step",
        CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(1),
        )))),
    );
    let deep_prop = make_ctl_prop(
        "deep-eu",
        CtlFormula::EU(
            Box::new(CtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::Constant(0),
                IntExpr::Constant(1),
            ))),
            Box::new(CtlFormula::EX(Box::new(CtlFormula::Atom(
                StatePredicate::IntLe(IntExpr::Constant(0), IntExpr::Constant(1)),
            )))),
        ),
    );

    let suffix_candidates =
        suffix_ctl_properties_for_test(&[suffix_prop.clone(), next_step_prop, deep_prop], &[]);

    assert_eq!(suffix_candidates, vec![suffix_prop]);
}

#[test]
fn test_ctl_deep_slice_verdict_parity_on_disconnected_net() {
    // Use the disconnected net with EF-only formulas (stutter-insensitive).
    // Production code uses the deep relevance cone; unsliced path doesn't.
    // Verdicts must match regardless.
    let net = disconnected_two_component_net();
    let props = component_a_ctl_props();
    let config = ExplorationConfig::default();

    let sliced = check_ctl_properties(&net, &props, &config);
    let unsliced = check_ctl_properties_unsliced(&net, &props, &config);
    assert_eq!(sliced, unsliced);
}

#[test]
fn test_ctl_fallback_with_ex_verdict_parity() {
    // EX(tokens(a1) >= 1) in the disconnected net. Contains EX, so the
    // batch stays on the conservative induced-subnet closure. Verdict
    // must still match the unsliced path.
    let net = disconnected_two_component_net();
    let props = vec![make_ctl_prop(
        "fallback-ex",
        CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec![String::from("a1")]),
        )))),
    )];

    assert!(ctl_batch_contains_next_step(&props));

    let config = ExplorationConfig::default();
    let sliced = check_ctl_properties(&net, &props, &config);
    let unsliced = check_ctl_properties_unsliced(&net, &props, &config);
    assert_eq!(sliced, unsliced);
}

#[test]
fn test_ctl_local_fallback_recovers_deep_formula_when_full_graph_truncates() {
    let net = ctl_local_fallback_net();
    let props = vec![make_ctl_prop("local-deep", ctl_local_fallback_formula())];
    let config = ExplorationConfig::new(3);

    let results =
        with_ctl_local_fallback_for_test(true, || check_ctl_properties(&net, &props, &config));
    assert_eq!(results, vec![(String::from("local-deep"), Verdict::True)]);
}

#[test]
fn test_ctl_local_fallback_disabled_after_full_graph_truncates() {
    let net = ctl_local_fallback_net();
    let props = vec![make_ctl_prop("local-deep", ctl_local_fallback_formula())];
    let config = ExplorationConfig::new(3);

    let results =
        with_ctl_local_fallback_for_test(false, || check_ctl_properties(&net, &props, &config));
    assert_eq!(
        results,
        vec![(String::from("local-deep"), Verdict::CannotCompute)]
    );
}

#[test]
fn test_ctl_local_fallback_tight_budget_matches_completed_full_graph_oracle() {
    use super::super::checker::CtlChecker;

    let net = finite_ctl_local_fallback_net();
    let formula = ctl_local_fallback_formula();
    let props = vec![make_ctl_prop("finite-local-deep", formula.clone())];
    let aliases = PropertyAliases::identity(&net);

    let full = explore_full(
        &net,
        &ExplorationConfig::new(16).refitted_for_full_graph(&net),
    );
    assert!(
        full.graph.completed,
        "finite fallback fixture must complete under the oracle budget"
    );
    let oracle =
        CtlChecker::new(&full, &net).eval(&resolve_ctl_with_aliases(&formula, &aliases))[0];

    let tight_config = ExplorationConfig::new(3);
    let tight_full = explore_full(&net, &tight_config.refitted_for_full_graph(&net));
    assert!(
        !tight_full.graph.completed,
        "tight budget must force the pipeline onto local fallback"
    );

    let results = check_ctl_properties(&net, &props, &tight_config);
    let expected = if oracle {
        Verdict::True
    } else {
        Verdict::False
    };
    assert_eq!(
        results,
        vec![(String::from("finite-local-deep"), expected)],
        "tight-budget local fallback must match the completed full-graph oracle"
    );
}

#[test]
fn test_ctl_local_fallback_flush_returns_only_unflushed_results() {
    let net = ctl_local_fallback_net();
    let props = vec![
        make_ctl_prop("local-deep-1", ctl_local_fallback_formula()),
        make_ctl_prop("local-deep-2", ctl_local_fallback_formula()),
    ];
    let config = ExplorationConfig::new(3);

    let results = with_ctl_local_fallback_for_test(true, || {
        check_ctl_properties_with_flush(&net, &props, &PropertyAliases::identity(&net), &config)
    });
    assert!(
        results.is_empty(),
        "flushing local fallback should not return already printed results"
    );
}

#[test]
fn test_local_edg_directly_proves_deep_formula_with_sufficient_budget() {
    let net = ctl_local_fallback_net();
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&ctl_local_fallback_formula(), &aliases);
    assert_eq!(
        solve_local_edg(&net, &resolved, &ExplorationConfig::new(10)),
        Ok(true)
    );
}

#[test]
fn test_ctl_local_fallback_recovers_real_mcc_formula_under_tight_budget() {
    const MODEL: &str = "AirplaneLD-PT-0010";
    const PROPERTY_ID: &str = "AirplaneLD-PT-0010-CTLFireability-2024-09";

    let model_dir = mcc_benchmark_dir(MODEL);
    if !model_dir.join("model.pnml").exists() {
        eprintln!("SKIP: {MODEL} benchmark not downloaded");
        return;
    }

    let net = parse_pnml_dir(&model_dir).expect("AirplaneLD-PT-0010 should parse");
    let properties =
        parse_properties(&model_dir, "CTLFireability").expect("CTLFireability XML should parse");
    let property = properties
        .into_iter()
        .find(|property| property.id == PROPERTY_ID)
        .expect("target AirplaneLD CTL property should exist");

    let ctl = match &property.formula {
        Formula::Ctl(ctl) => ctl,
        other => panic!("expected CTL formula, got {other:?}"),
    };
    assert!(
        ctl_formula_contains_next_step(ctl),
        "{PROPERTY_ID} should stay on the deep CTL lane, not a shallow shortcut"
    );
    assert!(
        classify_shallow_ctl(ctl).is_none() && classify_shallow_ctl_suffix(ctl).is_none(),
        "{PROPERTY_ID} should not be classified as a shallow reachability/suffix formula"
    );

    let config = ExplorationConfig::new(128);
    let full = explore_full(&net, &config);
    assert!(
        !full.graph.completed,
        "{PROPERTY_ID}: full-graph exploration should exceed the tight state budget"
    );

    let results = with_ctl_local_fallback_for_test(true, || {
        check_ctl_properties(&net, std::slice::from_ref(&property), &config)
    });
    assert_eq!(
        results,
        vec![(String::from(PROPERTY_ID), Verdict::False)],
        "{PROPERTY_ID}: local CTL fallback should recover the exact MCC verdict under a tight budget"
    );
}

/// Cross-validate that the local EDG solver agrees with the full-graph CtlChecker
/// on every CTL operator (EX, AX, EF, AF, EG, AG, EU, AU) across three
/// structurally different small nets.
#[test]
fn test_local_edg_agrees_with_full_graph_on_all_operators() {
    use super::super::checker::CtlChecker;
    use super::super::resolve::resolve_ctl;
    use crate::explorer::explore_full;
    use crate::petri_net::PlaceIdx;
    use std::collections::HashMap;

    let nets = [
        ("simple", simple_net()),
        ("suffix-cycle", suffix_cycle_net()),
        ("suffix-exit", suffix_exit_net()),
    ];

    for (net_name, net) in &nets {
        let place_map: HashMap<&str, PlaceIdx> = net
            .places
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id.as_str(), PlaceIdx(i as u32)))
            .collect();
        let trans_map: HashMap<&str, crate::petri_net::TransitionIdx> = net
            .transitions
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id.as_str(), crate::petri_net::TransitionIdx(i as u32)))
            .collect();
        let aliases = PropertyAliases::identity(net);

        let p0 = net.places[0].id.as_str();
        let atom = CtlFormula::Atom(atom_pred(p0, 1));
        let atom_false = CtlFormula::Atom(atom_pred(p0, 999));

        let formulas: Vec<(&str, CtlFormula)> = vec![
            ("EX-true", CtlFormula::EX(Box::new(atom.clone()))),
            ("AX-true", CtlFormula::AX(Box::new(atom.clone()))),
            ("EF-true", CtlFormula::EF(Box::new(atom.clone()))),
            ("AF-true", CtlFormula::AF(Box::new(atom.clone()))),
            ("EG-true", CtlFormula::EG(Box::new(atom.clone()))),
            ("AG-true", CtlFormula::AG(Box::new(atom.clone()))),
            ("EX-false", CtlFormula::EX(Box::new(atom_false.clone()))),
            ("EF-false", CtlFormula::EF(Box::new(atom_false.clone()))),
            ("AG-false", CtlFormula::AG(Box::new(atom_false.clone()))),
            (
                "EU",
                CtlFormula::EU(Box::new(atom.clone()), Box::new(atom_false.clone())),
            ),
            (
                "AU",
                CtlFormula::AU(Box::new(atom.clone()), Box::new(atom_false.clone())),
            ),
            (
                "nested-EF-AG",
                CtlFormula::EF(Box::new(CtlFormula::AG(Box::new(atom.clone())))),
            ),
        ];

        // Full-graph checker
        let config = ExplorationConfig::default();
        let full = explore_full(net, &config);
        assert!(
            full.graph.completed,
            "{net_name}: full graph must complete for cross-validation"
        );
        let full_checker = CtlChecker::new(&full, net);

        for (label, formula) in &formulas {
            let resolved_full = resolve_ctl(formula, &place_map, &trans_map);
            let full_result = full_checker.eval(&resolved_full)[0];

            let resolved_local = resolve_ctl_with_aliases(formula, &aliases);
            let local_result = solve_local_edg(net, &resolved_local, &config)
                .unwrap_or_else(|e| panic!("{net_name}/{label}: local EDG error: {e}"));

            assert_eq!(
                full_result, local_result,
                "{net_name}/{label}: full-graph={full_result}, local={local_result}"
            );
        }
    }
}

/// The 4-state ν-in-ν trigger net (finding N2): places `pr,p0,p1,p3`;
/// transitions t0:pr→p0, t1:p0→p1, t2:p0→p3, t3:p1→p0; init `[1,0,0,0]`.
fn nu_in_nu_trigger_net() -> crate::petri_net::PetriNet {
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

    let place_names = ["pr", "p0", "p1", "p3"];
    let places = place_names
        .iter()
        .map(|name| PlaceInfo {
            id: (*name).to_string(),
            name: None,
        })
        .collect();
    let idx = |n: &str| place_names.iter().position(|p| *p == n).unwrap() as u32;
    let spec: &[(&str, &str, &str)] = &[
        ("t0", "pr", "p0"),
        ("t1", "p0", "p1"),
        ("t2", "p0", "p3"),
        ("t3", "p1", "p0"),
    ];
    let transitions = spec
        .iter()
        .map(|(id, i, o)| TransitionInfo {
            id: (*id).to_string(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(idx(i)),
                weight: 1,
            }],
            outputs: vec![Arc {
                place: PlaceIdx(idx(o)),
                weight: 1,
            }],
        })
        .collect();
    PetriNet {
        name: Some("nu-in-nu-trigger".into()),
        places,
        transitions,
        initial_marking: vec![1, 0, 0, 0],
    }
}

/// The ν-in-ν trigger formula: `EG( AG(tokens(p3) <= 0) \/ (tokens(p1)+tokens(p3) <= 0) )`.
/// `EG` (ν) over `AG` (ν) nested through an `Or` — same-class nesting that the
/// old `ctl_is_alternation_free` gate WRONGLY admits to the `LocalCtlChecker`.
fn nu_in_nu_trigger_formula() -> CtlFormula {
    CtlFormula::EG(Box::new(CtlFormula::Or(vec![
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["p3".into()]),
            IntExpr::Constant(0),
        )))),
        CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["p1".into(), "p3".into()]),
            IntExpr::Constant(0),
        )),
    ])))
}

/// REGRESSION (finding N2 — CRITICAL): the single-pass `LocalCtlChecker`
/// returns a WRONG verdict for a same-class NESTED fixpoint (ν-in-ν). The OLD
/// `ctl_is_alternation_free` gate admitted this formula to the local lane; the
/// STRICT `ctl_has_single_fixpoint_layer` gate must REJECT it so it is instead
/// decided by the authoritative full-graph `CtlChecker` (or CANNOT_COMPUTE).
///
/// This pins all three facts at once:
///   1. the full-graph oracle verdict is FALSE (ground truth),
///   2. the raw `LocalCtlChecker` DISAGREES with the oracle (the unsoundness),
///   3. the new gate rejects the formula (so the unsound lane is never used),
/// while the old alternation-free gate wrongly admitted it.
#[test]
fn test_nu_in_nu_local_checker_disagrees_and_new_gate_rejects() {
    use super::super::checker::CtlChecker;
    use super::super::local_checker::LocalCtlChecker;
    use super::super::pipeline::{
        ctl_has_single_fixpoint_layer_for_test, ctl_is_alternation_free_for_test,
    };
    use super::super::resolve::resolve_ctl;
    use crate::petri_net::{PlaceIdx, TransitionIdx};
    use std::collections::HashMap;

    let net = nu_in_nu_trigger_net();
    let formula = nu_in_nu_trigger_formula();

    let place_map: HashMap<&str, PlaceIdx> = net
        .places
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id.as_str(), PlaceIdx(i as u32)))
        .collect();
    let trans_map: HashMap<&str, TransitionIdx> = net
        .transitions
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.as_str(), TransitionIdx(i as u32)))
        .collect();
    let resolved = resolve_ctl(&formula, &place_map, &trans_map);

    // (1) Full-graph oracle: the net is tiny (4 reachable states) so the
    // exhaustive `CtlChecker` is authoritative ground truth. EG(φ) is FALSE at
    // the initial state (every maximal path leaves the φ-set {pr,p0}).
    let full = explore_full(&net, &ExplorationConfig::new(1_000_000));
    assert!(full.graph.completed, "trigger full graph must complete");
    let oracle = CtlChecker::new(&full, &net).eval(&resolved)[0];
    assert!(
        !oracle,
        "full-graph oracle for the ν-in-ν trigger must be FALSE"
    );

    // (2) The raw single-pass LocalCtlChecker is UNSOUND on this ν-in-ν formula:
    // it disagrees with the oracle (returns a confident-but-wrong verdict). This
    // is precisely why it must be kept off the local lane. If a future rewrite
    // makes the local checker sound here, this assertion fires — revisit the gate.
    let mut local = LocalCtlChecker::new(&net, &ExplorationConfig::new(1_000_000));
    let local_val = local
        .eval_root(&resolved)
        .expect("local checker has ample budget on the 4-state trigger");
    assert_ne!(
        local_val, oracle,
        "LocalCtlChecker must DISAGREE with the oracle on the ν-in-ν trigger \
         (documented unsoundness that motivates the gate)"
    );

    // (3) THE FIX: the strict single-fixpoint-layer gate REJECTS the trigger, so
    // the pipeline never hands it to the unsound local lane — while the old
    // alternation-free gate wrongly admitted it (ν over ν is same-class).
    assert!(
        !ctl_has_single_fixpoint_layer_for_test(&resolved),
        "ctl_has_single_fixpoint_layer must REJECT the ν-in-ν trigger"
    );
    assert!(
        ctl_is_alternation_free_for_test(&resolved),
        "the old alternation-free gate WRONGLY admitted the ν-in-ν trigger \
         (same-class nesting) — this is the bug the strict gate fixes"
    );
}

/// Unit-pins the strict single-fixpoint-layer admission gate for the recursive
/// `LocalCtlChecker`. It must REJECT nested fixpoints (same OR opposite class)
/// and fixpoints under `EX`/`AX`, and ACCEPT a single fixpoint layer (with
/// sibling fixpoints and boolean/next-step nesting allowed above the layer).
#[test]
fn test_ctl_single_fixpoint_layer_gate() {
    use super::super::pipeline::ctl_has_single_fixpoint_layer_for_test as single_layer;
    use super::super::resolve::resolve_ctl_with_aliases;

    let net = ctl_budget_cycle_net();
    let aliases = PropertyAliases::identity(&net);
    let p = || CtlFormula::Atom(atom_pred("p0", 1));
    let q = || CtlFormula::Atom(atom_pred("p0", 2));
    let resolve = |f: &CtlFormula| resolve_ctl_with_aliases(f, &aliases);

    // ACCEPT: no fixpoint, or exactly one fixpoint layer.
    let atom = p();
    let ef = CtlFormula::EF(Box::new(p()));
    let ag = CtlFormula::AG(Box::new(p()));
    let eu = CtlFormula::EU(Box::new(p()), Box::new(q()));
    let ex_ax = CtlFormula::EX(Box::new(CtlFormula::AX(Box::new(p())))); // no fixpoint
    let and_siblings = CtlFormula::And(vec![ef.clone(), ag.clone()]); // sibling fixpoints
    let not_ef = CtlFormula::Not(Box::new(ef.clone())); // boolean above the layer
    let ex_atom = CtlFormula::EX(Box::new(p()));
    for (label, f) in [
        ("atom", &atom),
        ("EF p", &ef),
        ("AG p", &ag),
        ("E[p U q]", &eu),
        ("EX(AX p) — no fixpoint", &ex_ax),
        ("EF p /\\ AG p (siblings)", &and_siblings),
        ("NOT(EF p)", &not_ef),
        ("EX p", &ex_atom),
    ] {
        assert!(
            single_layer(&resolve(f)),
            "single-fixpoint-layer gate must ACCEPT {label}"
        );
    }

    // REJECT: two fixpoint layers (nesting), same OR opposite class, and any
    // fixpoint nested under a next-step modality.
    let nu_in_nu = nu_in_nu_trigger_formula(); // EG(AG p \/ atom) — the trigger
    let ag_ag = CtlFormula::AG(Box::new(CtlFormula::AG(Box::new(p())))); // ν-in-ν
    let ef_ef = CtlFormula::EF(Box::new(CtlFormula::EF(Box::new(p())))); // μ-in-μ
    let ag_ef = CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(p())))); // ν⊃μ (alternating)
    let ex_ef = CtlFormula::EX(Box::new(CtlFormula::EF(Box::new(p())))); // fixpoint under EX
    let ax_ag = CtlFormula::AX(Box::new(CtlFormula::AG(Box::new(p())))); // fixpoint under AX
    let egf = CtlFormula::EGF(Box::new(p())); // νμ fair cycle
    for (label, f) in [
        ("EG(AG p \\/ atom) — ν-in-ν trigger", &nu_in_nu),
        ("AG(AG p) — ν-in-ν", &ag_ag),
        ("EF(EF p) — μ-in-μ", &ef_ef),
        ("AG(EF p) — alternating", &ag_ef),
        ("EX(EF p) — fixpoint under EX", &ex_ef),
        ("AX(AG p) — fixpoint under AX", &ax_ag),
        ("EGF p — νμ fair cycle", &egf),
    ] {
        assert!(
            !single_layer(&resolve(f)),
            "single-fixpoint-layer gate must REJECT {label}"
        );
    }
}
