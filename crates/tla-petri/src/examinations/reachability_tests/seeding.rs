// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::fs;
use std::time::{Duration, Instant};

use tempfile::TempDir;

use super::fixtures::*;
use super::*;

#[test]
fn test_unresolved_place_returns_cannot_compute() {
    // AG(tokens("NONEXISTENT") <= 0) — unresolved place → CANNOT_COMPUTE
    let net = simple_net();
    let props = vec![make_ag_prop(
        "unresolved-place",
        StatePredicate::IntLe(
            IntExpr::TokensCount(vec!["NONEXISTENT".to_string()]),
            IntExpr::Constant(0),
        ),
    )];

    let results = check_reachability_properties(&net, &props, &default_config());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "unresolved-place");
    assert_eq!(results[0].1, Verdict::CannotCompute);
}

#[test]
fn test_unresolved_transition_returns_cannot_compute() {
    // EF(is-fireable("NONEXISTENT")) — unresolved transition → CANNOT_COMPUTE
    let net = simple_net();
    let props = vec![make_ef_prop(
        "unresolved-trans",
        StatePredicate::IsFireable(vec!["NONEXISTENT".to_string()]),
    )];

    let results = check_reachability_properties(&net, &props, &default_config());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "unresolved-trans");
    assert_eq!(results[0].1, Verdict::CannotCompute);
}

#[test]
fn test_valid_formula_still_works() {
    // EF(is-fireable("t0")) — valid, t0 is enabled at initial marking
    let net = simple_net();
    let props = vec![make_ef_prop(
        "valid-ef",
        StatePredicate::IsFireable(vec!["t0".to_string()]),
    )];

    let results = check_reachability_properties(&net, &props, &default_config());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "valid-ef");
    assert_eq!(results[0].1, Verdict::True);
}

#[test]
fn test_unresolved_original_name_hidden_by_simplification_returns_cannot_compute() {
    let net = simple_net();
    let props = vec![make_ef_prop(
        "unresolved-hidden-by-true",
        StatePredicate::Or(vec![
            StatePredicate::True,
            StatePredicate::IsFireable(vec!["MISSING".to_string()]),
        ]),
    )];

    let results = check_reachability_properties(&net, &props, &default_config());

    assert_eq!(
        results,
        vec![(
            "unresolved-hidden-by-true".to_string(),
            Verdict::CannotCompute,
        )]
    );
}

#[test]
fn test_bmc_seeding_preserves_order_and_invalid_entries_when_bfs_incomplete() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay",
        &format!(
            "printf 'call\\n' >> \"{}\"\ncat >/dev/null\nprintf 'sat\\nsat\\nunsat\\nunknown\\n'",
            calls_path.display()
        ),
    );
    let net = simple_net();
    let props = vec![
        make_ag_prop(
            "inv-00",
            StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["NONEXISTENT".to_string()]),
                IntExpr::Constant(0),
            ),
        ),
        make_ef_prop(
            "ef-01",
            StatePredicate::IntLe(
                IntExpr::Constant(1),
                IntExpr::TokensCount(vec!["p1".to_string()]),
            ),
        ),
        make_ag_prop(
            "ag-02",
            StatePredicate::IntLe(
                IntExpr::Constant(3),
                IntExpr::TokensCount(vec!["p0".to_string()]),
            ),
        ),
        make_ef_prop(
            "ef-03",
            StatePredicate::IntLe(
                IntExpr::Constant(10),
                IntExpr::TokensCount(vec!["p1".to_string()]),
            ),
        ),
        make_ag_prop(
            "ag-04",
            StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
                IntExpr::Constant(3),
            ),
        ),
    ];
    let limited_config = ExplorationConfig::new(1);

    let results = with_ay_path(&solver, || {
        check_reachability_properties(&net, &props, &limited_config)
    });

    // Core invariants that hold regardless of whether the fake ay succeeds:
    assert_eq!(results.len(), 5, "order and count must be preserved");
    assert_eq!(
        results[0],
        ("inv-00".to_string(), Verdict::CannotCompute),
        "unresolved names → CannotCompute"
    );
    // ef-01 (EF p1>=1): BMC may seed True, or stays CannotCompute if solver fails.
    assert!(
        results[1] == ("ef-01".to_string(), Verdict::True)
            || results[1] == ("ef-01".to_string(), Verdict::CannotCompute),
        "ef-01 must be True (BMC witness) or CannotCompute (solver failed), got {:?}",
        results[1].1
    );
    // ag-02 (AG p0>=3): BMC may seed False, or stays CannotCompute if solver fails.
    assert!(
        results[2] == ("ag-02".to_string(), Verdict::False)
            || results[2] == ("ag-02".to_string(), Verdict::CannotCompute),
        "ag-02 must be False (BMC counterexample) or CannotCompute (solver failed), got {:?}",
        results[2].1
    );
    // ef-03: EF(p1 >= 10) — LP proves infeasible (p0+p1=3), always FALSE.
    assert_eq!(
        results[3],
        ("ef-03".to_string(), Verdict::False),
        "LP must prove EF(p1>=10) false on conserving net"
    );
    // ag-04: AG(p0+p1 <= 3) — LP proves violation (p0+p1>=4) infeasible, always TRUE.
    assert_eq!(
        results[4],
        ("ag-04".to_string(), Verdict::True),
        "LP must prove AG(p0+p1<=3) true on conserving net"
    );
    // The fake batch-only solver is probed by the incremental wrapper, then
    // used by the batch fallback. BMC internals may also retry/split pending
    // queries, but the externally visible verdict/order contract above should
    // remain stable.
    if calls_path.exists() {
        let call_count = fs::read_to_string(&calls_path)
            .expect("call log should exist")
            .lines()
            .count();
        assert!(
            call_count >= 2,
            "batch-only fake solver should be probed and used by fallback/retry, got {call_count}"
        );
    }
}

#[test]
fn test_witness_search_runs_before_kinduction_for_simple_ef() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-kinduction-guard",
        &format!(
            "input=$(cat)\n\
             if printf '%s' \"$input\" | grep -q 'parikh_'; then\n\
               printf 'kinduction\\n' >> \"{}\"\n\
               printf 'sat\\n'\n\
             else\n\
               printf 'bmc\\n' >> \"{}\"\n\
               printf 'unsat\\n'\n\
             fi",
            calls_path.display(),
            calls_path.display(),
        ),
    );
    let net = simple_net();
    let props = vec![make_ef_prop(
        "ef-needs-witness-search",
        StatePredicate::IntLe(
            IntExpr::Constant(1),
            IntExpr::TokensCount(vec!["p1".to_string()]),
        ),
    )];
    // 20s is intentionally below the old AIGER minimum-global gate. The
    // reachability scheduler now fair-shares the residual deadline, so the
    // pre-BFS lanes should still make useful progress before final BFS.
    let config =
        ExplorationConfig::new(1).with_deadline(Some(Instant::now() + Duration::from_secs(20)));

    let results = with_ay_path(&solver, || {
        check_reachability_properties(&net, &props, &config)
    });

    assert_eq!(
        results,
        vec![("ef-needs-witness-search".to_string(), Verdict::True)],
        "validated witness search should prove the EF formula before BFS"
    );
    let calls = fs::read_to_string(&calls_path).unwrap_or_default();
    assert!(
        !calls.lines().any(|line| line == "kinduction"),
        "witness search should run before k-induction for unresolved EF witnesses; calls={calls:?}"
    );
}

#[test]
fn test_post_smt_witness_bfs_resolves_residual_tracker_only() {
    let net = simple_net();
    let props = vec![
        make_ef_prop(
            "already-seeded",
            StatePredicate::IsFireable(vec!["t0".to_string()]),
        ),
        make_ef_prop(
            "residual-witness",
            StatePredicate::IntLe(
                IntExpr::Constant(1),
                IntExpr::TokensCount(vec!["p1".to_string()]),
            ),
        ),
    ];
    let (_, mut trackers) = prepare_trackers(&net, &props);
    resolve_tracker(
        &mut trackers[0],
        true,
        ReachabilityResolutionSource::Bmc,
        Some(1),
    );
    let targets =
        super::super::super::reachability_witness::validation_targets_from_trackers(&trackers);
    let validation =
        super::super::super::reachability_witness::WitnessValidationContext::new(&net, &targets);
    let config =
        ExplorationConfig::new(10).with_deadline(Some(Instant::now() + Duration::from_secs(20)));

    let report =
        super::super::pipeline::run_post_smt_witness_bfs(&net, &mut trackers, &validation, &config);
    let stats = report
        .stats
        .expect("deadline-bearing config should run post-SMT witness BFS");

    assert_eq!(report.residual_before, 1);
    assert_eq!(report.seeded, 1);
    assert_eq!(report.unresolved_after, 0);
    assert!(stats.visited_states >= 2);
    assert_eq!(stats.resolved, 1);
    assert_eq!(stats.stop_reason.code(), "all_resolved");
    assert_eq!(trackers[0].verdict, Some(true));
    assert_eq!(
        trackers[0].resolved_by.map(|resolution| resolution.source),
        Some(ReachabilityResolutionSource::Bmc)
    );
    assert_eq!(trackers[1].verdict, Some(true));
    assert_eq!(
        trackers[1].resolved_by.map(|resolution| resolution.source),
        Some(ReachabilityResolutionSource::BfsWitness)
    );
}

#[test]
fn test_mixed_valid_invalid_preserves_order() {
    // property A: unresolved AG(tokens("NONEXISTENT") <= 0) → CANNOT_COMPUTE
    // property B: valid EF(is-fireable("t0")) → TRUE
    // property C: unresolved EF(is-fireable("MISSING")) → CANNOT_COMPUTE
    // property D: valid AG(tokens("p0") + tokens("p1") <= 3) → TRUE
    let net = simple_net();
    let props = vec![
        make_ag_prop(
            "inv-00",
            StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["NONEXISTENT".to_string()]),
                IntExpr::Constant(0),
            ),
        ),
        make_ef_prop("val-01", StatePredicate::IsFireable(vec!["t0".to_string()])),
        make_ef_prop(
            "inv-02",
            StatePredicate::IsFireable(vec!["MISSING".to_string()]),
        ),
        make_ag_prop(
            "val-03",
            StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["p0".to_string(), "p1".to_string()]),
                IntExpr::Constant(3),
            ),
        ),
    ];

    let results = check_reachability_properties(&net, &props, &default_config());
    assert_eq!(results.len(), 4);
    assert_eq!(results[0], ("inv-00".to_string(), Verdict::CannotCompute));
    assert_eq!(results[1], ("val-01".to_string(), Verdict::True));
    assert_eq!(results[2], ("inv-02".to_string(), Verdict::CannotCompute));
    assert_eq!(results[3], ("val-03".to_string(), Verdict::True));
}

#[test]
fn test_invalid_does_not_affect_valid_early_termination() {
    // All properties invalid — BFS should still run (with empty observer)
    // and produce only CANNOT_COMPUTE results.
    let net = simple_net();
    let props = vec![
        make_ef_prop(
            "all-inv-00",
            StatePredicate::IsFireable(vec!["MISSING".to_string()]),
        ),
        make_ag_prop(
            "all-inv-01",
            StatePredicate::IntLe(
                IntExpr::TokensCount(vec!["GHOST".to_string()]),
                IntExpr::Constant(0),
            ),
        ),
    ];

    let results = check_reachability_properties(&net, &props, &default_config());
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].1, Verdict::CannotCompute);
    assert_eq!(results[1].1, Verdict::CannotCompute);
}

/// End-to-end pipeline gate: `AG(is-fireable(t_dead))` negated — i.e. the
/// dead-transition invariant `AG(¬is-fireable(t_dead))` — is resolved TRUE by
/// the integer dead-transition sweep (Phase 2b-int) wired into the reachability
/// pipeline, and the live-transition counterpart is FALSE (a witness exists). On
/// this tiny net the exhaustive BFS would also decide both; the point is that the
/// pipeline returns the verdicts that match BFS ground truth with the sweep in
/// the path.
#[test]
fn test_pipeline_resolves_ag_not_fireable_dead_transition_matches_bfs() {
    // p0(1), p1(0). t_live: p0->p1 (enabled at init, LIVE). t_dead: needs 2 in
    // p1 (never reachable; p0+p1 bounded) -> p0 (DEAD).
    let net = PetriNet {
        name: Some("one-dead-one-live".to_string()),
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
                id: "t_live".to_string(),
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
                id: "t_dead".to_string(),
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
        ],
        initial_marking: vec![1, 0],
    };

    let props = vec![
        // AG(¬is-fireable(t_dead)) — TRUE (t_dead is dead).
        make_ag_prop(
            "ag-not-fireable-dead",
            StatePredicate::Not(Box::new(StatePredicate::IsFireable(vec![
                "t_dead".to_string()
            ]))),
        ),
        // AG(¬is-fireable(t_live)) — FALSE (t_live fires at the initial marking).
        make_ag_prop(
            "ag-not-fireable-live",
            StatePredicate::Not(Box::new(StatePredicate::IsFireable(vec![
                "t_live".to_string()
            ]))),
        ),
    ];

    let results = check_reachability_properties(&net, &props, &default_config());
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "ag-not-fireable-dead");
    assert_eq!(
        results[0].1,
        Verdict::True,
        "AG(¬is-fireable(t_dead)) must be TRUE — t_dead is dead"
    );
    assert_eq!(results[1].0, "ag-not-fireable-live");
    assert_eq!(
        results[1].1,
        Verdict::False,
        "AG(¬is-fireable(t_live)) must be FALSE — t_live fires at the initial marking"
    );
}
