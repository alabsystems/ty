// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::super::pipeline::with_experimental_ctl_shortcuts_for_test;
use super::super::{
    check_ctl_properties, check_ctl_properties_with_flush, classify_shallow_ctl,
    classify_shallow_ctl_suffix, ShallowCtl, ShallowCtlSuffix,
};
use super::support::{
    atom_pred, check_ctl_properties_unsliced, make_ctl_prop, simple_net, suffix_cycle_net,
    suffix_exit_net,
};

use crate::explorer::ExplorationConfig;
use crate::output::Verdict;
use crate::property_xml::{CtlFormula, IntExpr, StatePredicate};

#[test]
fn test_classify_shallow_ef_atom() {
    let formula = CtlFormula::EF(Box::new(CtlFormula::Atom(atom_pred("p0", 1))));
    let result = classify_shallow_ctl(&formula);
    assert!(matches!(result, Some(ShallowCtl::ExistsFinally(_))));
}

#[test]
fn test_classify_shallow_ag_atom() {
    let formula = CtlFormula::AG(Box::new(CtlFormula::Atom(atom_pred("p0", 1))));
    let result = classify_shallow_ctl(&formula);
    assert!(matches!(result, Some(ShallowCtl::AlwaysGlobally(_))));
}

#[test]
fn test_classify_shallow_ef_ef_atom_idempotent() {
    let formula = CtlFormula::EF(Box::new(CtlFormula::EF(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    let result = classify_shallow_ctl(&formula);
    assert!(matches!(result, Some(ShallowCtl::ExistsFinally(_))));
}

#[test]
fn test_classify_shallow_ag_ag_atom_idempotent() {
    let formula = CtlFormula::AG(Box::new(CtlFormula::AG(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    let result = classify_shallow_ctl(&formula);
    assert!(matches!(result, Some(ShallowCtl::AlwaysGlobally(_))));
}

#[test]
fn test_classify_shallow_not_ag_is_ef_negated() {
    let formula = CtlFormula::Not(Box::new(CtlFormula::AG(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    let result = classify_shallow_ctl(&formula);
    assert!(matches!(result, Some(ShallowCtl::ExistsFinally(_))));
}

#[test]
fn test_classify_shallow_not_ef_is_ag_negated() {
    let formula = CtlFormula::Not(Box::new(CtlFormula::EF(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    let result = classify_shallow_ctl(&formula);
    assert!(matches!(result, Some(ShallowCtl::AlwaysGlobally(_))));
}

#[test]
fn test_classify_shallow_ef_boolean_combo() {
    // EF(p0 >= 1 AND p1 >= 2) — boolean combination of atoms is still shallow.
    let formula = CtlFormula::EF(Box::new(CtlFormula::And(vec![
        CtlFormula::Atom(atom_pred("p0", 1)),
        CtlFormula::Atom(atom_pred("p1", 2)),
    ])));
    let result = classify_shallow_ctl(&formula);
    assert!(matches!(result, Some(ShallowCtl::ExistsFinally(_))));
}

#[test]
fn test_classify_shallow_eu_true_atom_as_ef() {
    let formula = CtlFormula::EU(
        Box::new(CtlFormula::Atom(StatePredicate::True)),
        Box::new(CtlFormula::Atom(atom_pred("p0", 1))),
    );
    let result = classify_shallow_ctl(&formula);
    assert!(matches!(result, Some(ShallowCtl::ExistsFinally(_))));
}

#[test]
fn test_classify_deep_ef_eu() {
    // EF(E[p U q]) — temporal inside means deep, not shallow.
    let formula = CtlFormula::EF(Box::new(CtlFormula::EU(
        Box::new(CtlFormula::Atom(atom_pred("p0", 1))),
        Box::new(CtlFormula::Atom(atom_pred("p1", 1))),
    )));
    assert!(classify_shallow_ctl(&formula).is_none());
}

#[test]
fn test_classify_deep_af() {
    // AF(atom) — liveness, not routable to reachability.
    let formula = CtlFormula::AF(Box::new(CtlFormula::Atom(atom_pred("p0", 1))));
    assert!(classify_shallow_ctl(&formula).is_none());
}

#[test]
fn test_classify_deep_eg() {
    // EG(atom) — needs SCC/cycle info, not shallow.
    let formula = CtlFormula::EG(Box::new(CtlFormula::Atom(atom_pred("p0", 1))));
    assert!(classify_shallow_ctl(&formula).is_none());
}

#[test]
fn test_classify_deep_ef_with_nested_ex() {
    // EF(EX(atom)) — next-step operator inside, not shallow.
    let formula = CtlFormula::EF(Box::new(CtlFormula::EX(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    assert!(classify_shallow_ctl(&formula).is_none());
}

#[test]
fn test_classify_shallow_suffix_af_atom() {
    let formula = CtlFormula::AF(Box::new(CtlFormula::Atom(atom_pred("p0", 1))));
    let result = classify_shallow_ctl_suffix(&formula);
    assert!(matches!(result, Some(ShallowCtlSuffix::AlwaysFinally(_))));
}

#[test]
fn test_classify_shallow_suffix_eg_atom() {
    let formula = CtlFormula::EG(Box::new(CtlFormula::Atom(atom_pred("p0", 1))));
    let result = classify_shallow_ctl_suffix(&formula);
    assert!(matches!(result, Some(ShallowCtlSuffix::ExistsGlobally(_))));
}

#[test]
fn test_classify_shallow_suffix_af_af_atom_idempotent() {
    let formula = CtlFormula::AF(Box::new(CtlFormula::AF(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    let result = classify_shallow_ctl_suffix(&formula);
    assert!(matches!(result, Some(ShallowCtlSuffix::AlwaysFinally(_))));
}

#[test]
fn test_classify_shallow_suffix_not_eg_is_af_negated() {
    let formula = CtlFormula::Not(Box::new(CtlFormula::EG(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    let result = classify_shallow_ctl_suffix(&formula);
    assert!(matches!(result, Some(ShallowCtlSuffix::AlwaysFinally(_))));
}

#[test]
fn test_classify_shallow_suffix_not_af_is_eg_negated() {
    let formula = CtlFormula::Not(Box::new(CtlFormula::AF(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    let result = classify_shallow_ctl_suffix(&formula);
    assert!(matches!(result, Some(ShallowCtlSuffix::ExistsGlobally(_))));
}

#[test]
fn test_classify_shallow_suffix_au_true_atom_as_af() {
    let formula = CtlFormula::AU(
        Box::new(CtlFormula::Atom(StatePredicate::True)),
        Box::new(CtlFormula::Atom(atom_pred("p0", 1))),
    );
    let result = classify_shallow_ctl_suffix(&formula);
    assert!(matches!(result, Some(ShallowCtlSuffix::AlwaysFinally(_))));
}

#[test]
fn test_classify_shallow_suffix_rejects_nested_temporal_inner_formula() {
    let formula = CtlFormula::EG(Box::new(CtlFormula::AF(Box::new(CtlFormula::Atom(
        atom_pred("p0", 1),
    )))));
    assert!(classify_shallow_ctl_suffix(&formula).is_none());
}

/// Verify that shallow routing produces the same verdicts as full CTL exploration
/// on a small net where both pipelines can complete.
#[test]
fn test_shallow_routing_parity_on_simple_net() {
    let net = simple_net();
    // EF(p1 >= 1) — should be TRUE (firing t0 moves token to p1).
    let ef_prop = make_ctl_prop(
        "ef-01",
        CtlFormula::EF(Box::new(CtlFormula::Atom(atom_pred("p1", 1)))),
    );
    // AG(p0 >= 0) — should be TRUE (always non-negative).
    let ag_prop = make_ctl_prop(
        "ag-01",
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::TokensCount(vec![String::from("p0")]),
        )))),
    );
    // AF(p1 >= 3) — deep, not shallow. On this net should be FALSE (max p1=2).
    let deep_prop = make_ctl_prop(
        "af-01",
        CtlFormula::AF(Box::new(CtlFormula::Atom(atom_pred("p1", 3)))),
    );

    let config = ExplorationConfig::default();
    let results = check_ctl_properties(&net, &[ef_prop, ag_prop, deep_prop], &config);

    let result_map: HashMap<&str, Verdict> = results
        .iter()
        .map(|(id, verdict)| (id.as_str(), *verdict))
        .collect();
    assert_eq!(result_map["ef-01"], Verdict::True);
    assert_eq!(result_map["ag-01"], Verdict::True);
    assert_eq!(result_map["af-01"], Verdict::False);
}

#[test]
fn test_shallow_routing_flush_returns_only_unflushed_results() {
    let net = simple_net();
    let ef_prop = make_ctl_prop(
        "ef-01",
        CtlFormula::EF(Box::new(CtlFormula::Atom(atom_pred("p1", 1)))),
    );
    let ag_prop = make_ctl_prop(
        "ag-01",
        CtlFormula::AG(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::TokensCount(vec![String::from("p0")]),
        )))),
    );
    // Use a predicate the simplifier can't resolve: tokens(p1) >= 2 is
    // reachable (state [0,2]) but not always true (initial [2,0] has p1=0).
    // AF(p1 >= 2) requires full state exploration and evaluates to TRUE.
    let deep_prop = make_ctl_prop(
        "af-01",
        CtlFormula::AF(Box::new(CtlFormula::Atom(atom_pred("p1", 2)))),
    );

    let config = ExplorationConfig::default();
    let results = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties_with_flush(
            &net,
            &[ef_prop, ag_prop, deep_prop],
            &crate::model::PropertyAliases::identity(&net),
            &config,
        )
    });

    assert!(
        results.is_empty(),
        "flushed shallow and suffix verdicts should not be returned for final printing"
    );
}

#[test]
fn test_initial_marking_forced_ctl_verdicts_ignore_expired_graph_budget() {
    let net = simple_net();
    let p0_present = CtlFormula::Atom(atom_pred("p0", 1));
    let p1_present = CtlFormula::Atom(atom_pred("p1", 1));
    let props = vec![
        make_ctl_prop(
            "ef-initial-true",
            CtlFormula::EF(Box::new(p0_present.clone())),
        ),
        make_ctl_prop(
            "ag-initial-false",
            CtlFormula::AG(Box::new(p1_present.clone())),
        ),
        make_ctl_prop(
            "af-initial-true",
            CtlFormula::AF(Box::new(p0_present.clone())),
        ),
        make_ctl_prop(
            "eg-initial-false",
            CtlFormula::EG(Box::new(p1_present.clone())),
        ),
        make_ctl_prop(
            "eu-initial-true",
            CtlFormula::EU(Box::new(p1_present.clone()), Box::new(p0_present.clone())),
        ),
        make_ctl_prop(
            "au-initial-false",
            CtlFormula::AU(
                Box::new(p1_present),
                Box::new(CtlFormula::Atom(atom_pred("p1", 2))),
            ),
        ),
    ];
    // Intentional past Instant; checked_sub would change the type to Option<Instant>.
    #[allow(clippy::unchecked_time_subtraction)]
    let expired_config =
        ExplorationConfig::new(0).with_deadline(Some(Instant::now() - Duration::from_secs(1)));

    let results = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties(&net, &props, &expired_config)
    });

    assert_eq!(
        results,
        vec![
            (String::from("ef-initial-true"), Verdict::True),
            (String::from("ag-initial-false"), Verdict::False),
            (String::from("af-initial-true"), Verdict::True),
            (String::from("eg-initial-false"), Verdict::False),
            (String::from("eu-initial-true"), Verdict::True),
            (String::from("au-initial-false"), Verdict::False),
        ],
        "initial-state CTL implications should resolve before graph exploration"
    );
}

#[test]
fn test_simplification_preserves_unresolved_name_guard() {
    let net = simple_net();
    let config = ExplorationConfig::default();
    let prop = make_ctl_prop(
        "missing-under-true",
        CtlFormula::AG(Box::new(CtlFormula::Or(vec![
            CtlFormula::Atom(StatePredicate::True),
            CtlFormula::Atom(StatePredicate::IsFireable(vec![String::from(
                "NONEXISTENT_TRANS",
            )])),
        ]))),
    );

    let results = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties(&net, &[prop], &config)
    });

    assert_eq!(
        results,
        vec![(String::from("missing-under-true"), Verdict::CannotCompute)],
        "unresolved names in the original CTL formula must fail closed even if simplification folds the branch away"
    );
}

#[test]
fn test_simplification_unresolved_guard_honors_flush_mode() {
    let net = simple_net();
    let config = ExplorationConfig::default();
    let prop = make_ctl_prop(
        "missing-under-true-flush",
        CtlFormula::AG(Box::new(CtlFormula::Or(vec![
            CtlFormula::Atom(StatePredicate::True),
            CtlFormula::Atom(StatePredicate::IsFireable(vec![String::from(
                "NONEXISTENT_TRANS",
            )])),
        ]))),
    );

    let results = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties_with_flush(
            &net,
            &[prop],
            &crate::model::PropertyAliases::identity(&net),
            &config,
        )
    });

    assert!(
        results.is_empty(),
        "flush mode should emit the fail-closed unresolved CTL result immediately"
    );
}

#[test]
fn test_suffix_routing_preserves_unresolved_name_guard() {
    let net = suffix_cycle_net();
    let config = ExplorationConfig::default();
    let props = vec![
        make_ctl_prop(
            "missing-eg",
            CtlFormula::EG(Box::new(CtlFormula::Atom(StatePredicate::IsFireable(
                vec![String::from("NONEXISTENT_TRANS")],
            )))),
        ),
        make_ctl_prop(
            "missing-af",
            CtlFormula::AF(Box::new(CtlFormula::Atom(StatePredicate::IntLe(
                IntExpr::Constant(1),
                IntExpr::TokensCount(vec![String::from("NONEXISTENT_PLACE")]),
            )))),
        ),
    ];

    let results = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties(&net, &props, &config)
    });

    assert_eq!(
        results,
        vec![
            (String::from("missing-eg"), Verdict::CannotCompute),
            (String::from("missing-af"), Verdict::CannotCompute),
        ],
        "CTL suffix shortcuts must not resolve formulas with missing original names"
    );
}

#[test]
fn test_suffix_routing_parity_on_non_bottom_cycle_net() {
    let net = suffix_cycle_net();
    let bad_reached = atom_pred("bad", 1);
    let not_bad = StatePredicate::IntLe(
        IntExpr::TokensCount(vec![String::from("bad")]),
        IntExpr::Constant(0),
    );
    let props = vec![
        make_ctl_prop(
            "eg-not-bad",
            CtlFormula::EG(Box::new(CtlFormula::Atom(not_bad.clone()))),
        ),
        make_ctl_prop(
            "af-bad",
            CtlFormula::AF(Box::new(CtlFormula::Atom(bad_reached.clone()))),
        ),
        make_ctl_prop(
            "not-eg-not-bad",
            CtlFormula::Not(Box::new(CtlFormula::EG(Box::new(CtlFormula::Atom(
                not_bad.clone(),
            ))))),
        ),
        make_ctl_prop(
            "not-af-bad",
            CtlFormula::Not(Box::new(CtlFormula::AF(Box::new(CtlFormula::Atom(
                bad_reached.clone(),
            ))))),
        ),
    ];

    let config = ExplorationConfig::default();
    let routed = check_ctl_properties(&net, &props, &config);

    // Note: check_ctl_properties_unsliced gives wrong answers on this net
    // because structural reduction removes the self-loop transition, which
    // breaks cycle detection for EG/AF. The suffix routing explores the
    // unreduced net and produces correct results.
    let result_map: HashMap<&str, Verdict> = routed
        .iter()
        .map(|(id, verdict)| (id.as_str(), *verdict))
        .collect();
    assert_eq!(result_map["eg-not-bad"], Verdict::True);
    assert_eq!(result_map["af-bad"], Verdict::False);
    assert_eq!(result_map["not-eg-not-bad"], Verdict::False);
    assert_eq!(result_map["not-af-bad"], Verdict::True);
}

#[test]
fn test_suffix_routing_parity_on_exit_only_net() {
    let net = suffix_exit_net();
    let bad_reached = atom_pred("bad", 1);
    let not_bad = StatePredicate::IntLe(
        IntExpr::TokensCount(vec![String::from("bad")]),
        IntExpr::Constant(0),
    );
    let props = vec![
        make_ctl_prop(
            "eg-not-bad",
            CtlFormula::EG(Box::new(CtlFormula::Atom(not_bad.clone()))),
        ),
        make_ctl_prop(
            "af-bad",
            CtlFormula::AF(Box::new(CtlFormula::Atom(bad_reached.clone()))),
        ),
    ];

    let config = ExplorationConfig::default();
    let routed = check_ctl_properties(&net, &props, &config);
    let unsliced = check_ctl_properties_unsliced(&net, &props, &config);
    assert_eq!(routed, unsliced);

    let result_map: HashMap<&str, Verdict> = routed
        .iter()
        .map(|(id, verdict)| (id.as_str(), *verdict))
        .collect();
    assert_eq!(result_map["eg-not-bad"], Verdict::False);
    assert_eq!(result_map["af-bad"], Verdict::True);
}

#[test]
fn test_suffix_routing_flush_returns_only_unflushed_results() {
    let net = suffix_cycle_net();
    let bad_reached = atom_pred("bad", 1);
    let not_bad = StatePredicate::IntLe(
        IntExpr::TokensCount(vec![String::from("bad")]),
        IntExpr::Constant(0),
    );
    let props = vec![
        make_ctl_prop(
            "eg-not-bad",
            CtlFormula::EG(Box::new(CtlFormula::Atom(not_bad))),
        ),
        make_ctl_prop(
            "af-bad",
            CtlFormula::AF(Box::new(CtlFormula::Atom(bad_reached))),
        ),
    ];

    let config = ExplorationConfig::default();
    let results = with_experimental_ctl_shortcuts_for_test(true, || {
        check_ctl_properties_with_flush(
            &net,
            &props,
            &crate::model::PropertyAliases::identity(&net),
            &config,
        )
    });

    assert!(
        results.is_empty(),
        "flushed suffix verdicts should not be returned for final printing"
    );
}
