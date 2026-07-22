// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::path::{Path, PathBuf};
use std::sync::MutexGuard;
use std::time::{Duration, Instant};

use super::*;
use crate::buchi::{check_ltl_on_the_fly, resolve_atom_with_aliases};
use crate::model::PropertyAliases;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{
    parse_properties, CtlFormula, Formula, IntExpr, LtlFormula, Property, StatePredicate,
};
use crate::reduction::ReducedNet;

struct LtlRollingBudgetEnvGuard {
    _lock: MutexGuard<'static, ()>,
    prev_enable: Option<String>,
    prev_disable: Option<String>,
}

impl Drop for LtlRollingBudgetEnvGuard {
    fn drop(&mut self) {
        restore_env_var(
            "TY_MCC_ENABLE_LTL_ROLLING_BUDGET",
            self.prev_enable.as_deref(),
        );
        restore_env_var(
            "TY_MCC_DISABLE_LTL_ROLLING_BUDGET",
            self.prev_disable.as_deref(),
        );
    }
}

fn restore_env_var(key: &str, value: Option<&str>) {
    match value {
        Some(value) => crate::env_guard::set_var(key, value),
        None => crate::env_guard::remove_var(key),
    }
}

fn ltl_rolling_budget_env_guard(
    enable: Option<&str>,
    disable: Option<&str>,
) -> LtlRollingBudgetEnvGuard {
    // Single crate-wide env lock: serialize against every other module's
    // env-touching test, not just this file's rolling-budget tests.
    let lock = crate::env_test_lock();
    let prev_enable = std::env::var("TY_MCC_ENABLE_LTL_ROLLING_BUDGET").ok();
    let prev_disable = std::env::var("TY_MCC_DISABLE_LTL_ROLLING_BUDGET").ok();
    restore_env_var("TY_MCC_ENABLE_LTL_ROLLING_BUDGET", enable);
    restore_env_var("TY_MCC_DISABLE_LTL_ROLLING_BUDGET", disable);
    LtlRollingBudgetEnvGuard {
        _lock: lock,
        prev_enable,
        prev_disable,
    }
}

fn cyclic_net() -> PetriNet {
    // p0 -> t0 -> p1 -> t1 -> p0, initial marking [1, 0]
    // States: [1,0] <-> [0,1] — cycle
    PetriNet {
        name: Some("cyclic".to_string()),
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
        initial_marking: vec![1, 0],
    }
}

fn make_ltl_prop(id: &str, ltl: LtlFormula) -> Property {
    Property {
        id: id.to_string(),
        formula: Formula::Ltl(ltl),
    }
}

fn benchmark_model_dir(model: &str) -> PathBuf {
    Path::new("benchmarks/mcc/2024/INPUTS").join(model)
}

fn load_real_ltl_property(
    model: &str,
    examination: &str,
    property_id: &str,
) -> Option<(PetriNet, Property)> {
    let model_dir = benchmark_model_dir(model);
    if !model_dir.join("model.pnml").exists() {
        return None;
    }

    let net = crate::parser::parse_pnml_dir(&model_dir).expect("real benchmark PNML should parse");
    let property = parse_properties(&model_dir, examination)
        .expect("real benchmark property XML should parse")
        .into_iter()
        .find(|prop| prop.id == property_id)
        .expect("property id should exist in benchmark XML");
    Some((net, property))
}

fn lookup_registry_verdict(path: &Path, model: &str, formula_index: usize) -> Option<Verdict> {
    if !path.exists() {
        return None;
    }

    let needle = format!("{model}/{formula_index},");
    let contents = std::fs::read_to_string(path).expect("registry CSV should read");
    let line = contents.lines().find(|line| line.starts_with(&needle))?;
    let raw = line
        .split_once(',')
        .expect("registry line should contain a comma")
        .1;
    Some(match raw {
        "true" => Verdict::True,
        "false" => Verdict::False,
        other => panic!("unexpected registry verdict {other}"),
    })
}

fn check_ltl_property_unguarded(
    net: &PetriNet,
    property: &Property,
    config: &ExplorationConfig,
) -> Verdict {
    let Formula::Ltl(ltl) = &property.formula else {
        return Verdict::CannotCompute;
    };

    let aliases = PropertyAliases::identity(net);
    let mut atom_preds = Vec::new();
    let nnf = to_nnf(ltl, &mut atom_preds);
    let resolved_atoms: Vec<_> = atom_preds
        .iter()
        .map(|pred| resolve_atom_with_aliases(pred, &aliases))
        .collect();

    // Use identity reduction (no-op) since this helper tests on the raw net.
    let reduced = ReducedNet::identity(net);
    match check_ltl_on_the_fly(
        &nnf,
        net,
        &reduced,
        net,
        &resolved_atoms,
        None,
        config.max_states(),
        config.deadline(),
    ) {
        Ok(Some(true)) => Verdict::True,
        Ok(Some(false)) => Verdict::False,
        Ok(None) => Verdict::CannotCompute,
        Err(error) => {
            panic!("on-the-fly LTL expansion should not overflow in test helper: {error}")
        }
    }
}

#[test]
fn test_ltl_globally_atom() {
    // A(G(tokens(p0) + tokens(p1) <= 1)) — conserving net, always true
    let net = cyclic_net();
    let props = vec![make_ltl_prop(
        "g-00",
        LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
            IntExpr::Constant(1),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ltl_unguarded_tight_budget_returns_cannot_compute() {
    let net = cyclic_net();
    let property = make_ltl_prop(
        "tight-budget-ltl-unguarded",
        LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::True))),
    );
    let verdict = check_ltl_property_unguarded(&net, &property, &ExplorationConfig::new(1));
    assert_eq!(verdict, Verdict::CannotCompute);
}

#[test]
fn test_ltl_unguarded_expired_deadline_returns_cannot_compute() {
    let net = cyclic_net();
    let property = make_ltl_prop(
        "expired-deadline-ltl-unguarded",
        LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
            StatePredicate::IntLe(
                IntExpr::Constant(1),
                IntExpr::TokensCount(vec!["p0".to_string()]),
            ),
        ))))),
    );
    let config = ExplorationConfig::default().with_deadline(Some(
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
    ));
    let verdict = check_ltl_property_unguarded(&net, &property, &config);
    assert_eq!(verdict, Verdict::CannotCompute);
}

#[test]
fn test_fair_share_budget_divides_remaining_time() {
    assert_eq!(
        fair_share_budget(Duration::from_secs(9), 3),
        Duration::from_secs(3)
    );
    // Duration / u32 preserves sub-millisecond remainder: 7ms / 4 = 1.75ms
    assert_eq!(
        fair_share_budget(Duration::from_millis(7), 4),
        Duration::from_micros(1750)
    );
}

#[test]
fn test_fair_share_budget_reserves_virtual_lanes() {
    assert_eq!(
        fair_share_budget_with_virtual_lanes(Duration::from_secs(20), 4, 1),
        Duration::from_secs(4),
        "four active prefilter lanes plus one Buchi lane should split 20s five ways"
    );
    assert_eq!(
        fair_share_budget_with_virtual_lanes(Duration::from_secs(20), 1, 1),
        Duration::from_secs(10),
        "one lasso lane plus one Buchi lane should split the residual budget evenly"
    );
}

#[test]
fn test_buchi_property_deadline_keeps_global_deadline() {
    let deadline = Instant::now() + Duration::from_secs(60);

    assert_eq!(buchi_property_deadline(Some(deadline), 16), Some(deadline));
}

#[test]
fn test_ltl_deadline_policy_shares_prefilters_but_not_buchi() {
    let now = Instant::now();
    let deadline = now + Duration::from_secs(20);
    let prefilter_deadline =
        ltl_prefilter_deadline_at(Some(deadline), 4, now).expect("prefilter deadline should exist");
    let buchi_deadline =
        buchi_property_deadline(Some(deadline), 4).expect("Buchi deadline should exist");

    assert!(
        prefilter_deadline < deadline,
        "shallow reachability prefilters should stay opportunistic"
    );
    assert!(
        prefilter_deadline <= now + LTL_PREFILTER_PHASE_CAP,
        "prefilters should stay capped so Buchi keeps enough budget"
    );
    assert_eq!(
        buchi_deadline, deadline,
        "complete Buchi solving should keep the global examination deadline"
    );
    assert_eq!(ltl_prefilter_deadline_at(None, 4, now), None);
    assert_eq!(buchi_property_deadline(None, 4), None);
}

#[test]
fn test_ltl_prefilter_deadline_expires_when_fair_share_is_too_small() {
    let now = Instant::now();
    let deadline = now + (LTL_PREFILTER_MIN_BUDGET * 4);

    assert_eq!(ltl_prefilter_deadline_at(Some(deadline), 4, now), Some(now));
}

#[test]
fn test_deep_buchi_expired_deadline_fails_closed_in_pipeline() {
    let net = cyclic_net();
    let ltl = LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        ),
    )))));
    // The formula is either deep-unclassified or routes to the lasso BMC
    // liveness candidate lane; both lanes must fail-closed on an expired
    // deadline. The pre-Buchi reachability shortcuts (Invariant / Eventually)
    // are the only paths that can short-circuit to a TRUE/FALSE verdict
    // without solving — and neither shape applies here.
    assert!(
        !matches!(
            classify_shallow_ltl(&ltl),
            Some(ShallowLtl::Invariant(_) | ShallowLtl::Eventually(_))
        ),
        "test formula must not be routed through reachability shortcuts"
    );
    let props = vec![make_ltl_prop("deep-expired-buchi", ltl)];
    let config = ExplorationConfig::default().with_deadline(Some(
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
    ));

    let results = check_ltl_properties(&net, &props, &config);

    assert_eq!(
        results,
        vec![("deep-expired-buchi".to_string(), Verdict::CannotCompute)]
    );
}

#[test]
fn test_ltl_previously_guarded_ids_now_execute_normally() {
    // After the guard-timing fix (#1246), the 5 previously-guarded property
    // IDs now execute normally instead of returning CannotCompute.
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let formerly_guarded_ids = [
        "AirplaneLD-PT-0010-LTLCardinality-04",
        "AirplaneLD-PT-0010-LTLCardinality-09",
        "CSRepetitions-PT-02-LTLCardinality-03",
        "Anderson-PT-04-LTLFireability-02",
        "CSRepetitions-PT-02-LTLFireability-03",
    ];
    let props: Vec<_> = formerly_guarded_ids
        .iter()
        .map(|id| {
            make_ltl_prop(
                id,
                LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
                    IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
                    IntExpr::Constant(1),
                )))),
            )
        })
        .collect();

    let results = check_ltl_properties(&net, &props, &config);
    for (id, verdict) in &results {
        assert_ne!(
            *verdict,
            Verdict::CannotCompute,
            "{id} should execute normally after guard-timing fix"
        );
    }
}

#[test]
fn test_ltl_adjacent_property_id_still_executes() {
    let net = cyclic_net();
    let props = vec![make_ltl_prop(
        "AirplaneLD-PT-0010-LTLCardinality-05",
        LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
            IntExpr::Constant(1),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ltl_finally_atom() {
    // A(F(tokens(p1) >= 1)) — cyclic net always reaches p1=1
    let net = cyclic_net();
    let props = vec![make_ltl_prop(
        "f-00",
        LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p1".to_string()]),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ltl_globally_finally() {
    // A(G(F(tokens(p0) >= 1))) — on the cycle, p0=1 recurs infinitely
    let net = cyclic_net();
    let props = vec![make_ltl_prop(
        "gf-00",
        LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
            StatePredicate::IntLe(
                IntExpr::Constant(1),
                IntExpr::TokensCount(vec!["p0".to_string()]),
            ),
        ))))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_ltl_globally_false() {
    // A(G(tokens(p0) >= 1)) — false because state [0,1] has p0=0
    let net = cyclic_net();
    let props = vec![make_ltl_prop(
        "g-01",
        LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        )))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::False);
}

// ── Regression tests for the 5 formerly-wrong LTL properties ──────
//
// These properties previously returned wrong answers due to successor-state
// guard timing in the Buchi product construction (see #1246). After the fix,
// the production path now correctly uses current-state guards and all 5
// match ground truth.

#[test]
fn test_regression_airplane_ltl_cardinality_04_matches_ground_truth() {
    let Some((net, property)) = load_real_ltl_property(
        "AirplaneLD-PT-0010",
        "LTLCardinality",
        "AirplaneLD-PT-0010-LTLCardinality-04",
    ) else {
        return;
    };

    let expected = lookup_registry_verdict(
        Path::new("registry/mcc-labels/l-t-l-cardinality-2024.csv"),
        "AirplaneLD-PT-0010",
        4,
    )
    .expect("ground truth should contain AirplaneLD-PT-0010/4");
    let verdict = check_ltl_property_unguarded(&net, &property, &ExplorationConfig::default());

    assert_eq!(expected, Verdict::False);
    assert_eq!(
        verdict, expected,
        "formerly wrong — fixed by #1246 guard-timing fix"
    );
}

#[test]
fn test_regression_airplane_ltl_cardinality_05_matches_ground_truth() {
    let Some((net, property)) = load_real_ltl_property(
        "AirplaneLD-PT-0010",
        "LTLCardinality",
        "AirplaneLD-PT-0010-LTLCardinality-05",
    ) else {
        return;
    };

    let expected = lookup_registry_verdict(
        Path::new("registry/mcc-labels/l-t-l-cardinality-2024.csv"),
        "AirplaneLD-PT-0010",
        5,
    )
    .expect("ground truth should contain AirplaneLD-PT-0010/5");
    let verdict = check_ltl_property_unguarded(&net, &property, &ExplorationConfig::default());

    assert_eq!(expected, Verdict::True);
    assert_eq!(verdict, expected);
}

#[test]
fn test_regression_airplane_ltl_cardinality_09_matches_ground_truth() {
    let Some((net, property)) = load_real_ltl_property(
        "AirplaneLD-PT-0010",
        "LTLCardinality",
        "AirplaneLD-PT-0010-LTLCardinality-09",
    ) else {
        return;
    };

    let expected = lookup_registry_verdict(
        Path::new("registry/mcc-labels/l-t-l-cardinality-2024.csv"),
        "AirplaneLD-PT-0010",
        9,
    )
    .expect("ground truth should contain AirplaneLD-PT-0010/9");
    let verdict = check_ltl_property_unguarded(&net, &property, &ExplorationConfig::default());

    assert_eq!(expected, Verdict::False);
    assert_eq!(
        verdict, expected,
        "formerly wrong — fixed by #1246 guard-timing fix"
    );
}

#[test]
fn test_regression_csrepetitions_ltl_cardinality_03_matches_ground_truth() {
    let Some((net, property)) = load_real_ltl_property(
        "CSRepetitions-PT-02",
        "LTLCardinality",
        "CSRepetitions-PT-02-LTLCardinality-03",
    ) else {
        return;
    };

    let expected = lookup_registry_verdict(
        Path::new("registry/mcc-labels/l-t-l-cardinality-2024.csv"),
        "CSRepetitions-PT-02",
        3,
    )
    .expect("ground truth should contain CSRepetitions-PT-02/3");
    let verdict = check_ltl_property_unguarded(&net, &property, &ExplorationConfig::default());

    assert_eq!(expected, Verdict::False);
    assert_eq!(
        verdict, expected,
        "formerly wrong — fixed by #1246 guard-timing fix"
    );
}

#[test]
fn test_regression_anderson_ltl_fireability_02_matches_ground_truth() {
    let Some((net, property)) = load_real_ltl_property(
        "Anderson-PT-04",
        "LTLFireability",
        "Anderson-PT-04-LTLFireability-02",
    ) else {
        return;
    };

    let expected = lookup_registry_verdict(
        Path::new("registry/mcc-labels/l-t-l-fireability-2024.csv"),
        "Anderson-PT-04",
        2,
    )
    .expect("ground truth should contain Anderson-PT-04/2");
    let verdict = check_ltl_property_unguarded(&net, &property, &ExplorationConfig::default());

    assert_eq!(expected, Verdict::False);
    assert_eq!(
        verdict, expected,
        "formerly wrong — fixed by #1246 guard-timing fix"
    );
}

#[test]
fn test_regression_csrepetitions_ltl_fireability_03_matches_ground_truth() {
    let Some((net, property)) = load_real_ltl_property(
        "CSRepetitions-PT-02",
        "LTLFireability",
        "CSRepetitions-PT-02-LTLFireability-03",
    ) else {
        return;
    };

    let expected = lookup_registry_verdict(
        Path::new("registry/mcc-labels/l-t-l-fireability-2024.csv"),
        "CSRepetitions-PT-02",
        3,
    )
    .expect("ground truth should contain CSRepetitions-PT-02/3");
    let verdict = check_ltl_property_unguarded(&net, &property, &ExplorationConfig::default());

    assert_eq!(expected, Verdict::False);
    assert_eq!(
        verdict, expected,
        "formerly wrong — fixed by #1246 guard-timing fix"
    );
}

// ── Shallow LTL classification tests ──────────────────────────────

fn some_pred() -> StatePredicate {
    StatePredicate::IntLe(
        IntExpr::TokensCount(vec!["p0".to_string()]),
        IntExpr::Constant(1),
    )
}

#[test]
fn test_classify_g_atom_is_invariant() {
    let f = LtlFormula::Globally(Box::new(LtlFormula::Atom(some_pred())));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Invariant(_))
    ));
}

#[test]
fn test_classify_f_atom_is_eventually() {
    let f = LtlFormula::Finally(Box::new(LtlFormula::Atom(some_pred())));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Eventually(_))
    ));
}

#[test]
fn test_classify_not_f_atom_is_invariant() {
    // Not(F(atom)) = G(Not(atom)) = AG(Not(atom))
    let f = LtlFormula::Not(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        some_pred(),
    )))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Invariant(_))
    ));
}

#[test]
fn test_classify_not_g_atom_is_eventually() {
    // Not(G(atom)) = F(Not(atom)) = AF(Not(atom))
    let f = LtlFormula::Not(Box::new(LtlFormula::Globally(Box::new(LtlFormula::Atom(
        some_pred(),
    )))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Eventually(_))
    ));
}

#[test]
fn test_classify_not_ff_atom_is_invariant() {
    // Not(F(F(atom))) = G(Not(atom)) — keep idempotent F wrappers shallow.
    let f = LtlFormula::Not(Box::new(LtlFormula::Finally(Box::new(
        LtlFormula::Finally(Box::new(LtlFormula::Atom(some_pred()))),
    ))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Invariant(_))
    ));
}

#[test]
fn test_classify_not_gg_atom_is_eventually() {
    // Not(G(G(atom))) = F(Not(atom)) — keep idempotent G wrappers shallow.
    let f = LtlFormula::Not(Box::new(LtlFormula::Globally(Box::new(
        LtlFormula::Globally(Box::new(LtlFormula::Atom(some_pred()))),
    ))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Eventually(_))
    ));
}

#[test]
fn test_classify_double_negated_eventually_is_eventually() {
    // Not(Not(F(atom))) = F(atom), so avoid the deep Buchi path.
    let f = LtlFormula::Not(Box::new(LtlFormula::Not(Box::new(LtlFormula::Finally(
        Box::new(LtlFormula::Atom(some_pred())),
    )))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Eventually(_))
    ));
}

#[test]
fn test_initial_marking_forced_verdict_handles_boolean_temporal_prefixes() {
    let net = cyclic_net();
    let aliases = PropertyAliases::identity(&net);

    let false_conjunction = LtlFormula::And(vec![
        LtlFormula::Atom(p1_ge_one()),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(p0_ge_one()))),
    ]);
    assert_eq!(
        initial_marking_forced_verdict(&net, &aliases, &false_conjunction),
        Some(Verdict::False),
        "a false state predicate in a top-level conjunction forces the LTL verdict false"
    );

    let negated_failed_invariant = LtlFormula::Not(Box::new(LtlFormula::Globally(Box::new(
        LtlFormula::Atom(p1_ge_one()),
    ))));
    assert_eq!(
        initial_marking_forced_verdict(&net, &aliases, &negated_failed_invariant),
        Some(Verdict::True),
        "if G(p) is already false at the initial marking, !G(p) is true"
    );

    let until_false_now = LtlFormula::Until(
        Box::new(LtlFormula::Atom(p1_ge_one())),
        Box::new(LtlFormula::Atom(StatePredicate::False)),
    );
    assert_eq!(
        initial_marking_forced_verdict(&net, &aliases, &until_false_now),
        Some(Verdict::False),
        "p U q is false when q is false now and p is also false now"
    );

    let next_unknown = LtlFormula::Next(Box::new(LtlFormula::Atom(p0_ge_one())));
    assert_eq!(
        initial_marking_forced_verdict(&net, &aliases, &next_unknown),
        None,
        "X formulas are not forced by the current marking alone"
    );
}

#[test]
fn test_ltl_buchi_scheduler_runs_simpler_formulas_first() {
    let atom = LtlFormula::Atom(p0_ge_one());
    let until = LtlFormula::Until(
        Box::new(LtlFormula::Globally(Box::new(
            LtlFormula::Atom(p0_ge_one()),
        ))),
        Box::new(LtlFormula::Next(Box::new(LtlFormula::Finally(Box::new(
            LtlFormula::Atom(p1_ge_one()),
        ))))),
    );
    assert!(
        ltl_schedule_cost(&atom) < ltl_schedule_cost(&until),
        "nested temporal formulas should be scheduled after simple formulas"
    );

    let properties = vec![
        make_ltl_prop("hard", until),
        make_ltl_prop("easy", atom),
        make_ltl_prop(
            "medium",
            LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one()))),
        ),
    ];
    let order = sorted_ltl_buchi_indices(&properties, &[0, 1, 2]);
    assert_eq!(
        order,
        vec![1, 2, 0],
        "Buchi scheduling should give cheap formulas a chance before hard formulas"
    );
}

#[test]
fn test_classify_g_g_atom_is_invariant() {
    // G(G(atom)) = AG(atom) — idempotent
    let f = LtlFormula::Globally(Box::new(LtlFormula::Globally(Box::new(LtlFormula::Atom(
        some_pred(),
    )))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Invariant(_))
    ));
}

#[test]
fn test_classify_f_f_atom_is_eventually() {
    // F(F(atom)) = AF(atom) — idempotent
    let f = LtlFormula::Finally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        some_pred(),
    )))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Eventually(_))
    ));
}

#[test]
fn test_classify_f_g_is_lasso_candidate() {
    // F(G(atom)) — persistence, routed to the bounded lasso BMC lane.
    // (Was previously expected to be `None`; promoted to a lasso candidate
    // when the LassoBmcLivenessCandidate variant landed.)
    let f = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(LtlFormula::Atom(
        some_pred(),
    )))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::LassoBmcLivenessCandidate)
    ));
}

#[test]
fn test_classify_g_f_is_lasso_candidate() {
    // G(F(atom)) — recurrence, routed to the bounded lasso BMC lane.
    // (Was previously expected to be `None`; promoted to a lasso candidate
    // when the LassoBmcLivenessCandidate variant landed.)
    let f = LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        some_pred(),
    )))));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::LassoBmcLivenessCandidate)
    ));
}

#[test]
fn test_classify_boolean_liveness_is_lasso_candidate() {
    let recurrence = LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(
        LtlFormula::Atom(some_pred()),
    ))));
    let persistence = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(
        LtlFormula::Atom(some_pred()),
    ))));

    assert!(matches!(
        classify_shallow_ltl(&LtlFormula::And(vec![
            recurrence.clone(),
            persistence.clone()
        ])),
        Some(ShallowLtl::LassoBmcLivenessCandidate)
    ));
    assert!(matches!(
        classify_shallow_ltl(&LtlFormula::Or(vec![recurrence, persistence])),
        Some(ShallowLtl::LassoBmcLivenessCandidate)
    ));
}

#[test]
fn test_classify_until_is_deep() {
    let f = LtlFormula::Until(
        Box::new(LtlFormula::Atom(some_pred())),
        Box::new(LtlFormula::Atom(some_pred())),
    );
    assert!(classify_shallow_ltl(&f).is_none());
}

#[test]
fn test_universal_ctl_fallback_accepts_state_until() {
    let f = LtlFormula::Until(
        Box::new(LtlFormula::Atom(p0_ge_one())),
        Box::new(LtlFormula::Atom(p1_ge_one())),
    );
    assert!(matches!(
        ltl_universal_ctl_fallback(&f),
        Some(CtlFormula::AU(_, _))
    ));
}

#[test]
fn test_universal_ctl_fallback_accepts_eventually_state() {
    let f = LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one())));
    assert!(matches!(
        ltl_universal_ctl_fallback(&f),
        Some(CtlFormula::AF(_))
    ));

    let net = cyclic_net();
    let props = vec![make_ltl_prop("eventually-p1", f)];
    let results = check_ltl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_universal_ctl_fallback_accepts_globally_eventually_state() {
    let f = LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        p1_ge_one(),
    )))));

    match ltl_universal_ctl_fallback(&f) {
        Some(CtlFormula::AG(inner)) => {
            assert!(
                matches!(inner.as_ref(), CtlFormula::AF(_)),
                "G(F(p)) should map to AG(AF(p))"
            );
        }
        other => panic!("expected AG(AF(p)) fallback, got {other:?}"),
    }
}

#[test]
fn test_persistence_lowers_to_exact_ctl_egf() {
    // A(FG p) ≡ ¬EGF(¬p): the EXACT fair-cycle characterization, replacing the
    // old branching-unsound AF(AG p) approximation. F(G p) now lowers to
    // Not(EGF(Atom(¬p))), which rides the GPU/CPU CTL fair-cycle lane.
    let f_g = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(LtlFormula::Atom(
        p0_ge_one(),
    )))));
    match ltl_universal_ctl_fallback(&f_g) {
        Some(CtlFormula::Not(inner)) => {
            assert!(
                matches!(inner.as_ref(), CtlFormula::EGF(_)),
                "F(G p) should lower to ¬EGF(¬p), got {inner:?}"
            );
        }
        other => panic!("expected ¬EGF(¬p), got {other:?}"),
    }

    // Sibling shapes are unchanged: F(pure pred) -> AF, G(F q) -> AG(AF).
    assert!(matches!(
        ltl_universal_ctl_fallback(&LtlFormula::Finally(Box::new(
            LtlFormula::Atom(p1_ge_one())
        ))),
        Some(CtlFormula::AF(_))
    ));
    assert!(matches!(
        ltl_universal_ctl_fallback(&LtlFormula::Globally(Box::new(LtlFormula::Finally(
            Box::new(LtlFormula::Atom(p1_ge_one()))
        )))),
        Some(CtlFormula::AG(_))
    ));
}

#[test]
fn test_universal_ctl_fallback_accepts_stutter_corrected_next() {
    let x_atom = LtlFormula::Next(Box::new(LtlFormula::Atom(p0_ge_one())));
    assert!(
        matches!(
            ltl_universal_ctl_fallback(&x_atom),
            Some(CtlFormula::And(_))
        ),
        "X is encoded as AX(phi) plus a deadlock self-stutter guard"
    );

    let net = PetriNet {
        name: Some("deadlock".to_string()),
        places: vec![PlaceInfo {
            id: "p0".to_string(),
            name: None,
        }],
        transitions: Vec::new(),
        initial_marking: vec![0],
    };
    let props = vec![make_ltl_prop(
        "next-p0-at-deadlock",
        LtlFormula::Next(Box::new(LtlFormula::Atom(p0_ge_one()))),
    )];
    let results = check_ltl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results[0].1,
        Verdict::False,
        "the corrected AX encoding must not inherit vacuous CTL AX truth"
    );
}

#[test]
fn test_persistence_exact_lowering_supersedes_true_sufficient_afag() {
    // F(G p) previously fell through to the TRUE-only AF(AG p) sufficient
    // condition; it now has the EXACT fair-cycle lowering ¬EGF(¬p). Because
    // ltl_universal_ctl_true_sufficient tries the exact fallback first, it too
    // now returns the exact ¬EGF form (superseding AF(AG p)).
    let f_g = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(LtlFormula::Atom(
        p0_ge_one(),
    )))));
    assert!(
        matches!(ltl_universal_ctl_fallback(&f_g), Some(CtlFormula::Not(_))),
        "F(G p) now lowers exactly to ¬EGF(¬p)"
    );
    assert!(
        matches!(
            ltl_universal_ctl_true_sufficient(&f_g),
            Some(CtlFormula::Not(_))
        ),
        "the exact ¬EGF form is returned in preference to AF(AG p)"
    );
}

#[test]
fn test_universal_ctl_true_sufficient_accepts_nested_until_shape() {
    let formula = LtlFormula::Until(
        Box::new(LtlFormula::Until(
            Box::new(LtlFormula::Globally(Box::new(
                LtlFormula::Atom(p0_ge_one()),
            ))),
            Box::new(LtlFormula::Atom(p1_ge_one())),
        )),
        Box::new(LtlFormula::Globally(Box::new(
            LtlFormula::Atom(p1_ge_one()),
        ))),
    );

    assert!(
        ltl_universal_ctl_fallback(&formula).is_none(),
        "nested temporal children remain outside the exact CTL fallback"
    );
    assert!(
        matches!(
            ltl_universal_ctl_true_sufficient(&formula),
            Some(CtlFormula::AU(_, _))
        ),
        "nested universal-LTL shapes should receive a TRUE-only ACTL sufficient check"
    );
}

#[test]
fn test_universal_ctl_true_sufficient_accepts_negated_temporal_until() {
    let formula = LtlFormula::Globally(Box::new(LtlFormula::Not(Box::new(LtlFormula::Until(
        Box::new(LtlFormula::Atom(p0_ge_one())),
        Box::new(LtlFormula::Next(Box::new(LtlFormula::Atom(p1_ge_one())))),
    )))));

    assert!(
        ltl_universal_ctl_fallback(&formula).is_none(),
        "negated temporal formulas remain outside the exact CTL fallback"
    );
    assert!(
        matches!(
            ltl_universal_ctl_true_sufficient(&formula),
            Some(CtlFormula::AG(_)) | Some(CtlFormula::Not(_))
        ),
        "NNF Release should be representable as a TRUE-only CTL sufficient condition"
    );
}

#[test]
fn test_universal_ctl_fallback_accepts_gf_atom() {
    let f = LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        p0_ge_one(),
    )))));
    match ltl_universal_ctl_fallback(&f) {
        Some(CtlFormula::AG(inner)) => {
            assert!(matches!(inner.as_ref(), CtlFormula::AF(_)));
        }
        other => panic!("expected AG(AF(p)), got: {other:?}"),
    }

    let net = cyclic_net();
    let props = vec![make_ltl_prop("gf-p0", f)];
    let results = check_ltl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_classify_and_of_lasso_candidates_is_lasso_candidate() {
    let f1 = LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        p0_ge_one(),
    )))));
    let f2 = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(LtlFormula::Atom(
        p1_ge_one(),
    )))));
    let combined = LtlFormula::And(vec![f1, f2]);

    assert!(matches!(
        classify_shallow_ltl(&combined),
        Some(ShallowLtl::LassoBmcLivenessCandidate)
    ));
}

#[test]
fn test_classify_next_is_deep() {
    let f = LtlFormula::Next(Box::new(LtlFormula::Atom(some_pred())));
    assert!(classify_shallow_ltl(&f).is_none());
}

#[test]
fn test_classify_g_and_atoms_is_invariant() {
    // G(atom1 AND atom2) — conjunction of state preds
    let f = LtlFormula::Globally(Box::new(LtlFormula::And(vec![
        LtlFormula::Atom(some_pred()),
        LtlFormula::Atom(StatePredicate::True),
    ])));
    assert!(matches!(
        classify_shallow_ltl(&f),
        Some(ShallowLtl::Invariant(_))
    ));
}

#[test]
fn test_classify_g_nested_temporal_is_deep() {
    // G(atom AND F(atom)) — mixed: conjunction has temporal child
    let f = LtlFormula::Globally(Box::new(LtlFormula::And(vec![
        LtlFormula::Atom(some_pred()),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(some_pred()))),
    ])));
    assert!(classify_shallow_ltl(&f).is_none());
}

// ── Routing parity: G(atom) via reachability matches Buchi ──

#[test]
fn test_shallow_g_invariant_matches_buchi_on_cyclic_net() {
    // G(tokens(p0) + tokens(p1) <= 1) is TRUE on the cyclic net.
    // Verify that the shallow routing (reachability) gives same answer.
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![make_ltl_prop(
        "g-shallow",
        LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
            IntExpr::Constant(1),
        )))),
    )];
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_shallow_g_false_invariant_matches_buchi() {
    // G(tokens(p0) >= 1) is FALSE — state [0,1] violates it.
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![make_ltl_prop(
        "g-false-shallow",
        LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        )))),
    )];
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::False);
}

#[test]
fn test_shallow_f_prefilter_true_via_ag() {
    // F(tokens(p0) + tokens(p1) <= 1) where the predicate is an invariant.
    // AG(pred) holds, so F(pred) = AF(pred) should be TRUE.
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![make_ltl_prop(
        "f-prefilter-true",
        LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
            IntExpr::Constant(1),
        )))),
    )];
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_shallow_f_initial_marking_true_ignores_expired_buchi_budget() {
    let net = cyclic_net();
    let config = ExplorationConfig::default().with_deadline(Some(
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
    ));
    let props = vec![make_ltl_prop(
        "f-initial-true",
        LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        )))),
    )];

    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results, vec![("f-initial-true".to_string(), Verdict::True)]);
}

#[test]
fn test_mixed_shallow_and_deep_properties() {
    // Mix of G(atom) (shallow), G(F(atom)) (deep), and F(atom) (pre-filter).
    // All should produce correct results when processed together.
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![
        // G(atom) — shallow invariant, TRUE
        make_ltl_prop(
            "mix-g",
            LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
                IntExpr::Constant(1),
            )))),
        ),
        // G(F(atom)) — deep recurrence, TRUE on cycle
        make_ltl_prop(
            "mix-gf",
            LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
                StatePredicate::IntLe(
                    IntExpr::Constant(1),
                    IntExpr::TokensCount(vec!["p0".to_string()]),
                ),
            ))))),
        ),
        // F(atom) — pre-filterable, TRUE (atom is invariant on this net)
        make_ltl_prop(
            "mix-f",
            LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
                IntExpr::Constant(1),
            )))),
        ),
    ];
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].0, "mix-g");
    assert_eq!(results[0].1, Verdict::True);
    assert_eq!(results[1].0, "mix-gf");
    assert_eq!(results[1].1, Verdict::True);
    assert_eq!(results[2].0, "mix-f");
    assert_eq!(results[2].1, Verdict::True);
}

#[test]
fn test_shallow_ltl_flush_returns_only_unflushed_results() {
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![
        make_ltl_prop(
            "mix-g",
            LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
                IntExpr::Constant(1),
            )))),
        ),
        make_ltl_prop(
            "mix-gf",
            LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
                StatePredicate::IntLe(
                    IntExpr::Constant(1),
                    IntExpr::TokensCount(vec!["p0".to_string()]),
                ),
            ))))),
        ),
        make_ltl_prop(
            "mix-f",
            LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
                IntExpr::Constant(1),
            )))),
        ),
    ];

    let results =
        check_ltl_properties_with_flush(&net, &props, &PropertyAliases::identity(&net), &config);

    assert!(
        results.is_empty(),
        "flushed shallow and Buchi verdicts should not be returned for final printing"
    );
}

#[test]
fn test_ltl_simplification_preserves_unresolved_name_guard() {
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![make_ltl_prop(
        "ltl-missing-under-true",
        LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
            LtlFormula::Atom(StatePredicate::True),
            LtlFormula::Atom(StatePredicate::IsFireable(vec![String::from(
                "NONEXISTENT_TRANS",
            )])),
        ]))),
    )];

    let results = check_ltl_properties(&net, &props, &config);

    assert_eq!(
        results,
        vec![(
            String::from("ltl-missing-under-true"),
            Verdict::CannotCompute
        )],
        "unresolved names in the original LTL formula must fail closed even if simplification folds the branch away"
    );
}

#[test]
fn test_ltl_simplification_unresolved_guard_honors_flush_mode() {
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![make_ltl_prop(
        "ltl-missing-flush",
        LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
            LtlFormula::Atom(StatePredicate::True),
            LtlFormula::Atom(StatePredicate::IsFireable(vec![String::from(
                "NONEXISTENT_TRANS",
            )])),
        ]))),
    )];

    let results =
        check_ltl_properties_with_flush(&net, &props, &PropertyAliases::identity(&net), &config);

    assert!(
        results.is_empty(),
        "flush mode should emit the fail-closed unresolved LTL result immediately"
    );
}

#[test]
fn test_shallow_f_prefilter_false_via_ef() {
    // F(tokens(p0) >= 2) is FALSE — total tokens = 1, so tokens(p0) >= 2 is
    // unreachable. The EF shortcut should detect EF(pred)=FALSE and conclude
    // AF(pred)=FALSE without invoking the Buchi product.
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![make_ltl_prop(
        "f-prefilter-false",
        LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(2),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        )))),
    )];
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::False);
}

#[test]
fn test_shallow_f_falls_through_to_buchi_when_shortcuts_inconclusive() {
    // F(tokens(p0) <= 0) on the cyclic net:
    // - AG(p0<=0) = FALSE (initial [1,0] violates it) → no quick TRUE
    // - EF(p0<=0) = TRUE (state [0,1] satisfies it) → no quick FALSE
    // → Falls through to Buchi. Answer is TRUE because the net cycles and
    //   every path eventually reaches [0,1] where p0=0.
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let props = vec![make_ltl_prop(
        "f-buchi-fallthrough",
        LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["p0".to_string()]),
            IntExpr::Constant(0),
        )))),
    )];
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(results[0].1, Verdict::True);
}

// ===========================================================================
// Rolling-residual Buchi budget tests
// ===========================================================================
//
// These tests cover the default-on `TY_MCC_ENABLE_LTL_ROLLING_BUDGET` policy
// and the `TY_MCC_DISABLE_LTL_ROLLING_BUDGET` kill switch. The historical
// full-global-deadline helper is still covered above by
// `test_buchi_property_deadline_keeps_global_deadline`.

#[test]
fn test_buchi_rolling_share_divides_remaining_time() {
    // 16 formulas, 16 minutes remaining, plus one virtual full-graph retry
    // lane. Each Buchi attempt gets its fair share of the 17 lanes.
    let now = Instant::now();
    let deadline = now + Duration::from_secs(16 * 60);
    let share = buchi_rolling_share_deadline_at(Some(deadline), 16, now)
        .expect("share deadline must exist");
    let observed = share.saturating_duration_since(now);
    let expected = fair_share_budget(Duration::from_secs(16 * 60), 17);
    assert_eq!(
        observed, expected,
        "fair share for 16 formula lanes plus full-graph retry lane in 16m \
         should be {expected:?}, got {observed:?}"
    );
}

#[test]
fn test_buchi_rolling_share_residual_increases_for_later_formulas() {
    // Simulate: 4 formulas, 2m remaining, plus one virtual full-graph retry
    // lane. First gets 24s share (120/5). After 5s of work and one formula
    // done, 3 formulas plus retry lane have 115s left, so each share grows.
    let now = Instant::now();
    let deadline = now + Duration::from_secs(120);
    let first_share =
        buchi_rolling_share_deadline_at(Some(deadline), 4, now).expect("share must exist");
    let first_budget = first_share.saturating_duration_since(now);

    let after_first = now + Duration::from_secs(5);
    let second_share =
        buchi_rolling_share_deadline_at(Some(deadline), 3, after_first).expect("share must exist");
    let second_budget = second_share.saturating_duration_since(after_first);

    assert!(
        first_budget == Duration::from_secs(24),
        "first share should be 24s, got {first_budget:?}"
    );
    assert!(
        second_budget >= first_budget,
        "residual share for later formulas (after fast formulas finish) must be \
         at least the original share — got first={first_budget:?} second={second_budget:?}"
    );
}

#[test]
fn test_buchi_rolling_share_never_consumes_full_remaining_budget() {
    // 16 formulas, 16s remaining, plus one virtual full-graph retry lane. The
    // first formula must not consume the full remaining deadline.
    let now = Instant::now();
    let deadline = now + Duration::from_secs(16);
    let share = buchi_rolling_share_deadline_at(Some(deadline), 16, now).expect("share must exist");
    let budget = share.saturating_duration_since(now);
    let expected = fair_share_budget(Duration::from_secs(16), 17);
    assert_eq!(budget, expected, "unexpected fair-share budget");
    assert!(
        budget < Duration::from_secs(16),
        "share must leave time for the remaining solver lanes; got {budget:?}"
    );
}

#[test]
fn test_buchi_rolling_share_treats_full_graph_as_virtual_lane() {
    let now = Instant::now();
    let deadline = now + Duration::from_secs(120);
    let one = buchi_rolling_share_deadline_at(Some(deadline), 1, now).expect("share must exist");
    let four = buchi_rolling_share_deadline_at(Some(deadline), 4, now).expect("share must exist");

    assert_eq!(
        one.saturating_duration_since(now),
        Duration::from_secs(60),
        "one residual formula plus one full-graph retry lane should split 120s evenly"
    );
    assert_eq!(
        four.saturating_duration_since(now),
        Duration::from_secs(24),
        "four residual formulas plus one full-graph retry lane should split 120s five ways"
    );
}

#[test]
fn test_ltl_full_graph_retry_budget_only_requires_live_deadline() {
    let now = Instant::now();
    assert!(
        ltl_full_graph_retry_has_budget_at(Some(now + Duration::from_nanos(1)), now),
        "exact full-graph retry should run whenever the global deadline is still live"
    );
    assert!(
        !ltl_full_graph_retry_has_budget_at(Some(now), now),
        "exact full-graph retry should skip once the global deadline is exhausted"
    );
    assert!(
        ltl_full_graph_retry_has_budget_at(None, now),
        "unbounded runs should allow exact full-graph retry"
    );
}

#[test]
fn test_buchi_rolling_share_returns_global_deadline_when_expired() {
    let deadline = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let share = buchi_rolling_share_deadline_at(Some(deadline), 4, Instant::now())
        .expect("share must exist for an expired deadline");
    // When the deadline has expired, we return it unchanged so the Buchi
    // solver short-circuits on deadline check. Either deadline or a value
    // <= deadline is acceptable as long as it's in the past.
    assert!(
        share <= deadline + Duration::from_micros(1),
        "expired deadline must propagate unchanged or earlier"
    );
}

#[test]
fn test_buchi_rolling_share_passthrough_when_no_global_deadline() {
    assert_eq!(
        buchi_rolling_share_deadline_at(None, 4, Instant::now()),
        None,
        "no global deadline → no per-formula deadline"
    );
}

#[test]
fn test_ltl_rolling_budget_env_default_enabled() {
    let _guard = ltl_rolling_budget_env_guard(None, None);
    assert!(
        ltl_rolling_budget_enabled(),
        "MCC LTL rolling budget must be enabled by default"
    );
}

#[test]
fn test_ltl_rolling_budget_env_enable_zero_disables() {
    let _guard = ltl_rolling_budget_env_guard(Some("0"), None);
    assert!(
        !ltl_rolling_budget_enabled(),
        "TY_MCC_ENABLE_LTL_ROLLING_BUDGET=0 must preserve the conservative full-deadline policy"
    );
}

#[test]
fn test_ltl_rolling_budget_env_disable_kill_switch_wins() {
    let _guard = ltl_rolling_budget_env_guard(Some("1"), Some("1"));
    assert!(
        !ltl_rolling_budget_enabled(),
        "TY_MCC_DISABLE_LTL_ROLLING_BUDGET=1 must override the old enable switch"
    );
}

#[test]
fn test_buchi_per_formula_dispatcher_uses_rolling_by_default() {
    // Default MCC batch behavior: rolling budget on, so a multi-formula batch
    // gets an earlier per-formula Buchi deadline than the global deadline.
    let _guard = ltl_rolling_budget_env_guard(None, None);
    let now = Instant::now();
    let deadline = now + Duration::from_secs(16 * 60);
    let per_formula =
        buchi_per_formula_deadline(Some(deadline), 16).expect("per-formula deadline must exist");
    assert!(
        per_formula < deadline,
        "default rolling per-formula deadline must be earlier than the global deadline"
    );
    assert!(
        per_formula <= now + Duration::from_secs(61),
        "default rolling policy should allocate roughly the first 1m share; got {:?}",
        per_formula.saturating_duration_since(now)
    );
}

#[test]
fn test_buchi_per_formula_dispatcher_uses_global_deadline_when_disabled() {
    let deadline = Instant::now() + Duration::from_secs(60);
    let _guard = ltl_rolling_budget_env_guard(Some("0"), None);
    assert_eq!(
        buchi_per_formula_deadline(Some(deadline), 16),
        Some(deadline),
        "explicitly disabling rolling must preserve the historical full-global-deadline policy"
    );
}

#[test]
fn test_lasso_bmc_deadline_reserves_buchi_virtual_lane() {
    let now = Instant::now();
    let deadline = now + Duration::from_secs(10);
    let lasso_deadline =
        lasso_bmc_deadline_at(Some(deadline), now).expect("finite deadline should map");

    assert_eq!(
        lasso_deadline.saturating_duration_since(now),
        Duration::from_secs(5),
        "lasso prefilter must fair-share with the complete Buchi fallback lane"
    );
}

#[test]
fn test_lasso_bmc_deadline_is_rejected_when_fair_share_is_too_small() {
    let now = Instant::now();
    let deadline = now + LTL_LASSO_BMC_MIN_BUDGET;
    let lasso_deadline =
        lasso_bmc_deadline_at(Some(deadline), now).expect("finite deadline should map");

    assert!(
        !lasso_bmc_has_budget(Some(lasso_deadline)),
        "lasso prefilter must skip when its fair share is below the minimum useful solver budget"
    );
    assert_eq!(
        lasso_bmc_deadline_at(None, now),
        None,
        "unbounded runs should still allow lasso BMC without an artificial deadline"
    );
}

#[test]
fn test_buchi_per_formula_dispatcher_uses_rolling_when_enabled() {
    // With rolling enabled, per-formula deadline must be strictly earlier
    // than the global deadline for a multi-formula examination.
    let now = Instant::now();
    let deadline = now + Duration::from_secs(16 * 60);
    with_ltl_rolling_budget_for_test(true, || {
        let per_formula = buchi_per_formula_deadline(Some(deadline), 16)
            .expect("per-formula deadline must exist");
        assert!(
            per_formula < deadline,
            "rolling per-formula deadline must be earlier than the global deadline; \
             got {per_formula:?} vs deadline {deadline:?}"
        );
    });
}

#[test]
fn test_buchi_rolling_budget_pipeline_does_not_starve_remaining_formulas() {
    // Regression for the MCC qualification-2 scoring loss: under the historical
    // full-deadline policy, a single deep formula that hits the global deadline
    // returns CannotCompute for itself AND for all later formulas in the queue.
    // With the rolling-residual budget enabled, later formulas still get a fair
    // share and can resolve.
    //
    // We construct a 4-property batch where:
    //   - Property 0: A(G(true)) — trivially TRUE via simplification (resolves
    //     in the invariant prefilter, not Buchi). Picked so the test does not
    //     depend on Buchi timing of "easy" deep formulas.
    //   - Property 1..3: deep LTL on a cyclic net — these enter the Buchi loop.
    //     Under an expired global deadline, default behavior returns
    //     CannotCompute for every Buchi entry (correct fail-closed semantics).
    //     Under the rolling policy, the per-formula deadline is computed at
    //     loop entry; the test verifies the call path is exercised (no panic,
    //     no early return) and that the merge layer routes verdicts correctly.
    let net = cyclic_net();
    let deep = LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        ),
    )))));
    let easy = LtlFormula::Globally(Box::new(LtlFormula::Atom(StatePredicate::True)));
    let props = vec![
        make_ltl_prop("rolling-easy", easy),
        make_ltl_prop("rolling-deep-1", deep.clone()),
        make_ltl_prop("rolling-deep-2", deep.clone()),
        make_ltl_prop("rolling-deep-3", deep),
    ];
    let config = ExplorationConfig::default();

    let baseline =
        with_ltl_rolling_budget_for_test(false, || check_ltl_properties(&net, &props, &config));
    let rolling =
        with_ltl_rolling_budget_for_test(true, || check_ltl_properties(&net, &props, &config));

    // Soundness: every formula that resolves under the default also resolves
    // identically under rolling (no TRUE↔FALSE flips, no verdict→CannotCompute
    // regressions when the default already resolved it).
    assert_eq!(baseline.len(), rolling.len(), "result count must match");
    for ((id_b, v_b), (id_r, v_r)) in baseline.iter().zip(rolling.iter()) {
        assert_eq!(id_b, id_r, "result ordering must match");
        if matches!(v_b, Verdict::True | Verdict::False) {
            assert_eq!(
                v_b, v_r,
                "rolling budget must not change a resolved verdict for {id_b}"
            );
        }
    }
}

#[test]
fn test_buchi_rolling_budget_preserves_token_ring_verdicts() {
    // Real fixture parity: TokenRing LTLFireability has 2 formulas, both
    // resolve under default policy. Under rolling policy, both must still
    // resolve to identical verdicts. This is a regression gate against the
    // rolling budget changing real MCC verdicts.
    use crate::examination::{collect_examination_with_dir, Examination, ExaminationValue};

    let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("repo root")
        .join("tests")
        .join("mcc_benchmarks")
        .join("token_ring");
    if !model_dir.join("model.pnml").exists() {
        eprintln!("SKIP: token_ring fixture not present");
        return;
    }

    let net = crate::parser::parse_pnml_dir(&model_dir).expect("token_ring PNML should parse");
    let config = ExplorationConfig::new(100_000);

    let baseline = with_ltl_rolling_budget_for_test(false, || {
        collect_examination_with_dir(
            &net,
            "TokenRing",
            &model_dir,
            Examination::LTLFireability,
            &config,
        )
        .expect("default LTLFireability collection should succeed")
    });
    let rolling = with_ltl_rolling_budget_for_test(true, || {
        collect_examination_with_dir(
            &net,
            "TokenRing",
            &model_dir,
            Examination::LTLFireability,
            &config,
        )
        .expect("rolling LTLFireability collection should succeed")
    });

    assert_eq!(baseline.len(), rolling.len(), "record count must match");
    for (b, r) in baseline.iter().zip(rolling.iter()) {
        assert_eq!(b.formula_id, r.formula_id);
        // Compare verdicts directly so a structural change to ExaminationValue
        // does not silently weaken this gate.
        match (&b.value, &r.value) {
            (ExaminationValue::Verdict(vb), ExaminationValue::Verdict(vr)) => {
                if matches!(vb, Verdict::True | Verdict::False) {
                    assert_eq!(
                        vb, vr,
                        "rolling budget must preserve resolved verdict for {}",
                        b.formula_id
                    );
                }
            }
            other => panic!(
                "expected verdict value for {}, got {:?}",
                b.formula_id, other
            ),
        }
    }

    // Demonstration data: report how many verdicts each policy resolved.
    let count_resolved = |records: &[crate::examination::ExaminationRecord]| {
        records
            .iter()
            .filter(|r| {
                matches!(
                    r.value,
                    ExaminationValue::Verdict(Verdict::True | Verdict::False)
                )
            })
            .count()
    };
    let baseline_resolved = count_resolved(&baseline);
    let rolling_resolved = count_resolved(&rolling);
    eprintln!(
        "token_ring LTLFireability resolved verdicts — default: {baseline_resolved}/{}; \
         rolling: {rolling_resolved}/{}",
        baseline.len(),
        rolling.len()
    );
    assert!(
        rolling_resolved >= baseline_resolved,
        "rolling policy must not regress the resolved-verdict count: \
         baseline={baseline_resolved}, rolling={rolling_resolved}"
    );
}

// ===========================================================================
// Liveness-shape lasso-BMC classifier extension
// ===========================================================================
//
// The classifier promotes G F p, F G p, and G(p → F q) shapes to a
// dedicated LassoBmcLivenessCandidate variant. The Phase 2.5 pipeline
// stage runs a bounded lasso BMC on the property's NNF; any returned
// witness is replay-validated, so a Some(_) result produces a sound
// False. No witness ⇒ the property must fall through to the Büchi
// fallback (NEVER a bounded True from this stage).

fn p0_ge_one() -> StatePredicate {
    StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::TokensCount(vec!["p0".to_string()]),
    )
}

fn p1_ge_one() -> StatePredicate {
    StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::TokensCount(vec!["p1".to_string()]),
    )
}

/// A trap net: t0 fires once, removing the token from p0. After firing
/// the net is dead at marking [0, 1] — no transition is enabled.
///
/// Under stutter-extended LTL semantics, the only infinite execution is
/// [1,0] → [0,1] → [0,1] → … so G F (p0 ≥ 1) is FALSE (eventually p0
/// never holds again) and F G (p0 ≤ 0) is TRUE.
fn trap_net() -> PetriNet {
    PetriNet {
        name: Some("trap".to_string()),
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
        initial_marking: vec![1, 0],
    }
}

#[test]
fn test_classify_g_response_pattern_is_lasso_candidate() {
    // G(¬p ∨ F q) is the LTL encoding of G(p → F q) — must match.
    let response = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one()))),
    ])));
    assert!(matches!(
        classify_shallow_ltl(&response),
        Some(ShallowLtl::LassoBmcLivenessCandidate)
    ));

    // Order independence: the response body may appear with F first.
    let response_swapped = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one()))),
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
    ])));
    assert!(matches!(
        classify_shallow_ltl(&response_swapped),
        Some(ShallowLtl::LassoBmcLivenessCandidate)
    ));
}

#[test]
fn test_classify_g_or_three_children_is_not_response() {
    // Strict pattern: response body must have exactly two children
    // (¬p ∨ F q). A three-way Or is intentionally not classified to
    // avoid mis-recognising disjunctions that are not equivalent to
    // p → F q.
    let three_way = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one()))),
        LtlFormula::Atom(p0_ge_one()),
    ])));
    assert!(classify_shallow_ltl(&three_way).is_none());
}

#[test]
fn test_classify_g_or_temporal_body_is_not_response() {
    // The classifier rejects response bodies whose Not- or F-operand
    // contains additional temporal operators — the rewrite identity
    // G(p → F q) ≡ ¬F(p ∧ G ¬q) requires p and q to be pure state
    // predicates.
    let nested_not = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
            p0_ge_one(),
        ))))),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one()))),
    ])));
    assert!(classify_shallow_ltl(&nested_not).is_none());

    let nested_finally = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
        LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(LtlFormula::Atom(
            p1_ge_one(),
        ))))),
    ])));
    assert!(classify_shallow_ltl(&nested_finally).is_none());
}

#[test]
fn test_classify_response_rewrite_equivalence() {
    // For several G(p → F q) variants, the classifier-routed verdict
    // must match the direct Büchi verdict on the same formula. This
    // is the soundness contract for the response pattern: Phase 2.5
    // may upgrade an Unclassified/CannotCompute to False, but never
    // flips True ↔ False. We pin both verdicts on the cyclic net.
    let net = cyclic_net();
    let config = ExplorationConfig::default();

    // 1) G((p0 ≥ 1) → F (p1 ≥ 1)) — TRUE on the cycle [1,0] ↔ [0,1].
    let response_true = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one()))),
    ])));
    // 2) G((p0 ≥ 1) → F (p0 ≥ 2)) — FALSE: token count is conserved at
    // 1, so p0 ≥ 2 is unreachable; the implication fires at [1,0] but
    // F(p0 ≥ 2) is False there.
    let response_false = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(2),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        )))),
    ])));

    for (id, ltl) in [
        ("response-true", response_true),
        ("response-false", response_false),
    ] {
        let pipeline_verdict =
            check_ltl_properties(&net, &[make_ltl_prop(id, ltl.clone())], &config)[0].1;
        let direct_verdict = check_ltl_property_unguarded(&net, &make_ltl_prop(id, ltl), &config);
        assert_eq!(
            pipeline_verdict, direct_verdict,
            "{id}: classifier-routed verdict must match direct Büchi verdict — \
             pipeline={pipeline_verdict:?}, direct={direct_verdict:?}"
        );
    }
}

#[test]
fn test_gf_recurrence_matches_buchi_on_cyclic_net() {
    // G F (p0 ≥ 1) on the cyclic net is TRUE — p0 recurs every two
    // steps. Either the lasso BMC reports no counterexample within the
    // depth ladder (Unclassified ⇒ Phase 3 Büchi resolves True) or the
    // SMT solver is unavailable and Phase 3 takes over directly. Either
    // way, the verdict must be TRUE (not False — which would indicate a
    // wrong-answer soundness bug in the lasso replay).
    let net = cyclic_net();
    let props = vec![make_ltl_prop(
        "gf-recurrence-true",
        LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
            p0_ge_one(),
        ))))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(
        results[0].1,
        Verdict::True,
        "G F (p0 ≥ 1) on cyclic net must be TRUE; got {:?}",
        results[0].1
    );
}

#[test]
fn test_gf_recurrence_false_on_trap_net() {
    // G F (p0 ≥ 1) on the trap net is FALSE — after t0 fires we are
    // stuck at [0, 1] forever, so F(p0 ≥ 1) holds initially but is
    // False from [0, 1] onwards. The Büchi pipeline correctly returns
    // False; the lasso-BMC classifier extension must not flip this to
    // True. (When ay-chc is available the lasso BMC may find the witness
    // first — either way, the verdict is False.)
    let net = trap_net();
    let props = vec![make_ltl_prop(
        "gf-recurrence-false",
        LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
            p0_ge_one(),
        ))))),
    )];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(
        results[0].1,
        Verdict::False,
        "G F (p0 ≥ 1) on trap net must be FALSE; got {:?}",
        results[0].1
    );
}

#[test]
fn test_lasso_candidate_budget_exhausted_returns_cannot_compute() {
    // Soundness floor: when the lasso BMC depth ladder finds no
    // counterexample (here, simulated by an already-expired deadline
    // that prevents both the lasso BMC and the Büchi solver from
    // running) the pipeline MUST return CannotCompute, never False.
    // A bounded check cannot prove True for a liveness property and
    // must not be misread as evidence for False either.
    let net = cyclic_net();
    let props = vec![make_ltl_prop(
        "gf-budget-exhausted",
        LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
            p0_ge_one(),
        ))))),
    )];
    // Expired deadline starves Phase 2.5 lasso BMC AND Phase 3 Büchi.
    let config = ExplorationConfig::default().with_deadline(Some(
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
    ));
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(
        results[0].1,
        Verdict::CannotCompute,
        "no lasso witness within budget must NOT be reported as False; \
         got {:?}",
        results[0].1
    );
}

#[test]
fn test_response_satisfied_on_cyclic_net() {
    // G((p0 ≥ 1) → F (p1 ≥ 1)) on the cyclic net is TRUE — whenever
    // p0 holds, p1 holds at the next step.
    let net = cyclic_net();
    let response = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one()))),
    ])));
    let props = vec![make_ltl_prop("response-true", response)];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(
        results[0].1,
        Verdict::True,
        "satisfied response must be TRUE; got {:?}",
        results[0].1
    );
}

#[test]
fn test_response_violated_on_trap_net() {
    // G((p0 ≥ 1) → F (p0 ≥ 2)) on the trap net is FALSE — at the
    // initial marking p0 ≥ 1 holds, but tokens are conserved at 1
    // so p0 ≥ 2 is unreachable; F(p0 ≥ 2) is False.
    let net = trap_net();
    let response_false = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(2),
            IntExpr::TokensCount(vec!["p0".to_string()]),
        )))),
    ])));
    let props = vec![make_ltl_prop("response-false", response_false)];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(
        results[0].1,
        Verdict::False,
        "violated response must be FALSE; got {:?}",
        results[0].1
    );
}

#[test]
fn test_persistence_false_on_cyclic_net() {
    // A(FG (p1 ≥ 1)) on the cyclic net [1,0]<->[0,1] is FALSE: every path
    // revisits [1,0] (¬p1) infinitely often, so p1 is never "eventually
    // always". Routed through the exact ¬EGF(¬p1) fair-cycle lowering.
    let net = cyclic_net();
    let persistence = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(
        LtlFormula::Atom(p1_ge_one()),
    ))));
    let props = vec![make_ltl_prop("persistence-false", persistence)];
    let results = check_ltl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results[0].1,
        Verdict::False,
        "A(FG p1) must be FALSE on the toggling cycle; got {:?}",
        results[0].1
    );
}

#[test]
fn test_persistence_true_on_trap_net() {
    // A(FG (p1 ≥ 1)) on the trap net [1,0]->[0,1](deadlock) is TRUE: the single
    // path reaches [0,1] (p1) and stays there forever (the deadlock self-stutter
    // keeps p1 holding), so p1 is eventually always true.
    let net = trap_net();
    let persistence = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(
        LtlFormula::Atom(p1_ge_one()),
    ))));
    let props = vec![make_ltl_prop("persistence-true", persistence)];
    let results = check_ltl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results[0].1,
        Verdict::True,
        "A(FG p1) must be TRUE when the path settles into a p1 deadlock; got {:?}",
        results[0].1
    );
}

#[test]
fn test_persistence_deadlock_stutter_false_on_trap_net() {
    // A(FG (p0 ≥ 1)) on the trap net is FALSE: the path settles into the
    // deadlock [0,1] which is ¬p0, so p0 is NOT eventually always. This is the
    // deadlock-stutter soundness pin — EGF(¬p0) must treat the deadlocked
    // ¬p0-state as an infinite ¬p0-stutter witness (TRUE), making A(FG p0)
    // FALSE. Without the stutter term the deadlock would be dropped and the
    // verdict would flip to a WRONG TRUE.
    let net = trap_net();
    let persistence = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(
        LtlFormula::Atom(p0_ge_one()),
    ))));
    let props = vec![make_ltl_prop("persistence-stutter-false", persistence)];
    let results = check_ltl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(
        results[0].1,
        Verdict::False,
        "A(FG p0) must be FALSE (path ends in a ¬p0 deadlock); got {:?}",
        results[0].1
    );
}

#[test]
fn test_egf_not_query_sliced_on_deadlock_net() {
    // Direct CTL-pipeline pin for the fair-cycle slice-corruption fix. On the
    // trap net (t0: p0→p1, deadlock [0,1]), `EGF(p0 ≤ 0)` is TRUE (the
    // deadlock [0,1] is a p0≤0 fair-cycle witness). The deep relevance-cone
    // query slice used to drop the sink place p1 and corrupt the deadlock
    // marking to [1,0] (p0 not consumed), so `p0≤0` read false everywhere and
    // EGF wrongly returned FALSE — flipping the LTL persistence verdict. The
    // fix excludes fair-cycle (EGF) batches from slicing (and from structural
    // reduction), so the full net is explored.
    let net = trap_net();
    let egf = CtlFormula::EGF(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::TokensCount(vec!["p0".to_string()]),
        IntExpr::Constant(0),
    ))));
    let prop = Property {
        id: "egf-deadlock".to_string(),
        formula: Formula::Ctl(egf),
    };
    let out = crate::examinations::ctl::check_ctl_properties(
        &net,
        std::slice::from_ref(&prop),
        &ExplorationConfig::default(),
    );
    assert_eq!(
        out[0].1,
        Verdict::True,
        "EGF(p0≤0) must be TRUE (deadlock [0,1] is a p0≤0 fair-cycle witness); got {:?}",
        out[0].1
    );
}

#[test]
fn test_response_lowers_to_exact_ctl_ef_eg() {
    // The response body G(¬p ∨ F q) must lower to the EXACT CTL response
    // characterization ¬EF(p ∧ EG¬q) so it rides the CTL/GPU lane instead of
    // the Büchi lane. This proves the reduction FIRES (the two net-level
    // response tests above then confirm it produces the Büchi-equal verdict).
    let response = LtlFormula::Globally(Box::new(LtlFormula::Or(vec![
        LtlFormula::Not(Box::new(LtlFormula::Atom(p0_ge_one()))),
        LtlFormula::Finally(Box::new(LtlFormula::Atom(p1_ge_one()))),
    ])));
    let ctl = ltl_universal_ctl_fallback(&response)
        .expect("response shape must lower to a CTL formula, not fall through");
    match &ctl {
        CtlFormula::Not(inner) => match inner.as_ref() {
            CtlFormula::EF(ef) => match ef.as_ref() {
                CtlFormula::And(children) => {
                    assert_eq!(children.len(), 2, "expected p ∧ EG¬q");
                    assert!(
                        matches!(children[0], CtlFormula::Atom(_)),
                        "first conjunct must be the state predicate p"
                    );
                    assert!(
                        matches!(children[1], CtlFormula::EG(_)),
                        "second conjunct must be EG¬q"
                    );
                }
                other => panic!("expected And(p, EG¬q), got {other:?}"),
            },
            other => panic!("expected EF, got {other:?}"),
        },
        other => panic!("expected ¬EF(...), got {other:?}"),
    }

    // A plain safety G(p) and G(F q) must still lower to their existing exact
    // forms (not the response path) — no regression to the sibling shapes.
    let g_safety = LtlFormula::Globally(Box::new(LtlFormula::Atom(p0_ge_one())));
    assert!(matches!(
        ltl_universal_ctl_fallback(&g_safety),
        Some(CtlFormula::AG(_))
    ));
    let gf = LtlFormula::Globally(Box::new(LtlFormula::Finally(Box::new(LtlFormula::Atom(
        p1_ge_one(),
    )))));
    assert!(matches!(
        ltl_universal_ctl_fallback(&gf),
        Some(CtlFormula::AG(_)) // AG(AF ...)
    ));
}

#[test]
fn test_fg_persistence_classify_lasso_candidate() {
    // F G (p0 ≤ 0) on the trap net — persistence holds because we
    // settle into [0, 1] forever. The classifier promotes the F(G ...)
    // shape to LassoBmcLivenessCandidate; the verdict must be TRUE.
    let f_g = LtlFormula::Finally(Box::new(LtlFormula::Globally(Box::new(LtlFormula::Atom(
        StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["p0".to_string()]),
            IntExpr::Constant(0),
        ),
    )))));
    assert!(matches!(
        classify_shallow_ltl(&f_g),
        Some(ShallowLtl::LassoBmcLivenessCandidate)
    ));

    let net = trap_net();
    let props = vec![make_ltl_prop("fg-persistence-true", f_g)];
    let config = ExplorationConfig::default();
    let results = check_ltl_properties(&net, &props, &config);
    assert_eq!(
        results[0].1,
        Verdict::True,
        "F G (p0 ≤ 0) on trap net must be TRUE; got {:?}",
        results[0].1
    );
}

// ===========================================================================
// Default-on lasso BMC: budget and gating tests
// ===========================================================================
//
// `TY_MCC_DISABLE_LTL_LASSO_BMC` is the opt-out switch — default is ON. These
// tests pin the budget cap and the fail-closed behavior so a regression here
// would either eat into the Büchi fallback's wall budget, or silently disable
// the lasso lane entirely.

#[test]
fn test_lasso_bmc_deadline_caps_phase_when_remaining_budget_is_large() {
    // Ample remaining time: the cap is derived from one full depth ladder plus
    // a model query, rather than a machine-dependent wall-clock guess.
    let now = Instant::now();
    let deadline = now + Duration::from_secs(120);
    let lasso = lasso_bmc_deadline_at(Some(deadline), now).expect("lasso deadline exists");
    let observed = lasso.saturating_duration_since(now);
    let cap = ltl_lasso_bmc_phase_cap();
    assert!(
        observed <= cap,
        "lasso lane must cap at the full depth-ladder budget {cap:?}; got {observed:?}"
    );
    assert!(
        observed >= cap.saturating_sub(Duration::from_millis(1)),
        "lasso lane should use its full derived cap when budget is plentiful; got {observed:?}"
    );
}

#[test]
fn test_lasso_bmc_deadline_fair_shares_with_buchi_when_below_cap() {
    let now = Instant::now();
    let deadline = now + Duration::from_secs(10);
    let lasso = lasso_bmc_deadline_at(Some(deadline), now).expect("lasso deadline exists");
    assert_eq!(
        lasso,
        now + Duration::from_secs(5),
        "lasso lane must fair-share the residual per-formula budget with Buchi"
    );
}

#[test]
fn test_lasso_bmc_deadline_passthrough_when_no_global_deadline() {
    assert_eq!(
        lasso_bmc_deadline_at(None, Instant::now()),
        None,
        "no global deadline → no lasso deadline"
    );
}

#[test]
fn test_lasso_bmc_has_budget_rejects_expired_deadline() {
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(
        !lasso_bmc_has_budget(Some(past)),
        "expired deadline must skip lasso BMC to fail-closed"
    );
}

#[test]
fn test_lasso_bmc_has_budget_accepts_unbounded() {
    assert!(
        lasso_bmc_has_budget(None),
        "an unbounded deadline must not block lasso BMC"
    );
}

#[test]
fn test_lasso_bmc_default_enabled() {
    // The opt-out env var must be the only thing that disables lasso BMC.
    // Default behavior is ON.
    // Hold the crate-wide env lock so the remove/restore of the global
    // TY_MCC_DISABLE_LTL_LASSO_BMC cannot race a concurrent reader/mutator.
    let _env = crate::env_test_lock();
    let prev = std::env::var("TY_MCC_DISABLE_LTL_LASSO_BMC").ok();
    crate::env_guard::remove_var("TY_MCC_DISABLE_LTL_LASSO_BMC");
    let enabled = ltl_lasso_bmc_enabled();
    if let Some(value) = prev {
        crate::env_guard::set_var("TY_MCC_DISABLE_LTL_LASSO_BMC", value);
    } else {
        crate::env_guard::remove_var("TY_MCC_DISABLE_LTL_LASSO_BMC");
    }
    assert!(enabled, "lasso BMC must be on by default");
}

#[test]
fn test_lasso_bmc_only_runs_for_classified_liveness_candidates() {
    assert!(
        should_run_lasso_bmc(Some(&ShallowLtl::LassoBmcLivenessCandidate)),
        "classified liveness candidates should use the optional lasso lane"
    );
    assert!(
        !should_run_lasso_bmc(None),
        "unclassified formulas must preserve Buchi budget instead of spending it on lasso BMC"
    );
    assert!(
        !should_run_lasso_bmc(Some(&ShallowLtl::Invariant(StatePredicate::True))),
        "reachability-routable invariants must not spend fallback Buchi budget on lasso BMC"
    );
    assert!(
        !should_run_lasso_bmc(Some(&ShallowLtl::Eventually(StatePredicate::True))),
        "reachability-prefiltered eventual formulas must not spend fallback Buchi budget on lasso BMC"
    );
}
