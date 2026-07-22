// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for PDR reachability pre-seeding.

use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

use super::super::reachability::PropertyTracker;
use super::super::reachability::ReachabilityResolutionSource;
use super::{env_flag_value_enabled, run_pdr_seeding};

struct EnvVarGuard<'a> {
    key: &'a str,
    prev: Option<String>,
}

impl<'a> EnvVarGuard<'a> {
    fn set(key: &'a str, value: Option<&str>) -> Self {
        let prev = std::env::var(key).ok();
        match value {
            Some(value) => crate::env_guard::set_var(key, value),
            None => crate::env_guard::remove_var(key),
        }
        Self { key, prev }
    }
}

impl Drop for EnvVarGuard<'_> {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            crate::env_guard::set_var(self.key, prev);
        } else {
            crate::env_guard::remove_var(self.key);
        }
    }
}

fn with_reachability_pdr_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    // Single crate-wide env lock: serialize against every other module's
    // env-touching test, not just this file's.
    let _lock = crate::env_test_lock();
    let _guard = EnvVarGuard::set("TY_MCC_ENABLE_REACHABILITY_PDR", value);
    f()
}

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

fn three_token_net() -> PetriNet {
    PetriNet {
        name: Some("three_token".to_string()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![3, 0],
    }
}

fn simple_net() -> PetriNet {
    PetriNet {
        name: Some("simple".to_string()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    }
}

fn tracker(id: &str, quantifier: PathQuantifier, predicate: ResolvedPredicate) -> PropertyTracker {
    PropertyTracker {
        id: id.to_string(),
        quantifier,
        predicate,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }
}

#[test]
fn test_reachability_pdr_env_flag_parser_defaults_and_truthy_values() {
    assert!(env_flag_value_enabled(None, true));
    assert!(!env_flag_value_enabled(None, false));
    assert!(env_flag_value_enabled(Some("1"), false));
    assert!(env_flag_value_enabled(Some(" true "), false));
    assert!(env_flag_value_enabled(Some("yes"), false));
    assert!(env_flag_value_enabled(Some("on"), false));
    assert!(!env_flag_value_enabled(Some("0"), true));
    assert!(!env_flag_value_enabled(Some("false"), true));
}

#[test]
fn test_pdr_seeds_ag_true_for_inductive_invariant() {
    let net = three_token_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
            ResolvedIntExpr::Constant(3),
        ),
    )];

    with_reachability_pdr_env(Some("1"), || run_pdr_seeding(&net, &mut trackers, None));
    assert_eq!(trackers[0].verdict, Some(true));
    assert_eq!(
        trackers[0].resolved_by.map(|resolution| resolution.source),
        Some(ReachabilityResolutionSource::Pdr)
    );
}

#[test]
fn test_pdr_seeds_ag_false_for_reachable_counterexample() {
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::Constant(0),
        ),
    )];

    with_reachability_pdr_env(Some("1"), || run_pdr_seeding(&net, &mut trackers, None));
    assert_eq!(trackers[0].verdict, Some(false));
    assert_eq!(
        trackers[0].resolved_by.map(|resolution| resolution.source),
        Some(ReachabilityResolutionSource::Pdr)
    );
}

#[test]
fn test_pdr_seeds_ef_true_when_target_is_reachable() {
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
    )];

    with_reachability_pdr_env(Some("1"), || run_pdr_seeding(&net, &mut trackers, None));
    assert_eq!(trackers[0].verdict, Some(true));
    assert_eq!(
        trackers[0].resolved_by.map(|resolution| resolution.source),
        Some(ReachabilityResolutionSource::Pdr)
    );
}

#[test]
fn test_pdr_seeds_ef_false_when_target_is_unreachable() {
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(2),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ),
    )];

    with_reachability_pdr_env(Some("1"), || run_pdr_seeding(&net, &mut trackers, None));
    assert_eq!(trackers[0].verdict, Some(false));
    assert_eq!(
        trackers[0].resolved_by.map(|resolution| resolution.source),
        Some(ReachabilityResolutionSource::Pdr)
    );
}

#[test]
fn test_pdr_leaves_preseeded_verdict_unchanged() {
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::True,
    )];
    trackers[0].verdict = Some(true);

    with_reachability_pdr_env(Some("1"), || run_pdr_seeding(&net, &mut trackers, None));
    assert_eq!(trackers[0].verdict, Some(true));
    assert_eq!(trackers[0].resolved_by, None);
}

#[test]
fn test_pdr_force_disabled_leaves_tracker_unresolved() {
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
    )];

    // PDR seeding is on by default; the explicit "0" override must still
    // suppress the phase so users can fall back to pure BFS.
    with_reachability_pdr_env(Some("0"), || run_pdr_seeding(&net, &mut trackers, None));
    assert_eq!(trackers[0].verdict, None);
    assert_eq!(trackers[0].resolved_by, None);
}

#[test]
fn test_pdr_deadline_expiry_leaves_tracker_unresolved() {
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::Constant(1),
        ),
    )];

    with_reachability_pdr_env(Some("1"), || {
        run_pdr_seeding(&net, &mut trackers, Some(std::time::Instant::now()))
    });
    assert_eq!(trackers[0].verdict, None);
    assert_eq!(trackers[0].resolved_by, None);
}

#[test]
fn test_pdr_skips_fireability_for_original_net_bfs_guard() {
    let net = simple_net();
    let mut trackers = vec![
        tracker(
            "ag-prop",
            PathQuantifier::AG,
            ResolvedPredicate::Not(Box::new(ResolvedPredicate::IsFireable(vec![
                TransitionIdx(0),
            ]))),
        ),
        tracker(
            "ef-prop",
            PathQuantifier::EF,
            ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
        ),
    ];

    with_reachability_pdr_env(Some("1"), || run_pdr_seeding(&net, &mut trackers, None));
    for tracker in trackers {
        assert_eq!(tracker.verdict, None);
        assert_eq!(tracker.resolved_by, None);
    }
}
