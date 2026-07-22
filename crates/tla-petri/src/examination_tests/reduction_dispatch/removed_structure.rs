// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests verifying verdict preservation after constant-place or dead-transition removal.

use crate::output::Verdict;

use crate::explorer::ExplorationConfig;

use super::super::super::{
    deadlock_verdict, liveness_verdict, one_safe_verdict, quasi_liveness_verdict,
    stable_marking_verdict, state_space_stats,
};
use super::super::fixtures::default_config;
use super::support::{
    agglomerable_chain_net, isolated_place_net, structurally_dead_transition_net,
};

#[test]
fn test_one_safe_verdict_preserves_removed_isolated_place_tokens() {
    let config = default_config();
    assert_eq!(
        one_safe_verdict(&isolated_place_net(5), &config, &[]),
        Verdict::False
    );
}

#[test]
fn test_stable_marking_verdict_short_circuits_when_reduction_finds_stable_place() {
    let config = default_config();
    assert_eq!(
        stable_marking_verdict(&isolated_place_net(5), &config, &[]),
        Verdict::True
    );
}

#[test]
fn test_state_space_stats_restore_removed_constant_contribution() {
    let config = default_config();
    let stats = state_space_stats(&isolated_place_net(5), &config)
        .expect("small net should finish exploration");

    assert_eq!(stats.states, tla_bignum::BigUint::from(2u32));
    assert_eq!(stats.edges, tla_bignum::BigUint::from(1u32));
    assert_eq!(stats.max_token_in_place, 5);
    assert_eq!(stats.max_token_sum, 6);
}

/// Regression (#1483): when the original net is too large to explore within
/// max_states, `state_space_stats` must return `None` (CANNOT_COMPUTE = 0 pts)
/// instead of wrong counts from a reduced net (-8 pts per wrong value).
#[test]
fn test_state_space_stats_returns_none_when_exploration_incomplete() {
    // The net must be unable to complete on BOTH the BFS path and the
    // sound DD path so the fail-closed contract is genuinely exercised.
    //
    // We use an *unbounded* net: a source transition with no input
    // perpetually adds tokens to an isolated place, so the reachable set
    // is infinite. BFS capped at 1 state is incomplete, and the DD path
    // declines because the place's LP upper bound is infinite (no finite
    // bound to encode) — independent of the per-place encoding cap, which
    // the binary encoding now raises to 2^20. (A merely *high-bound* but
    // finite net no longer forces a decline: the binary encoding handles
    // bounds up to 2^20 exactly, which is the intended coverage win.)
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
    let unbounded = PetriNet {
        name: Some("unbounded-source".into()),
        places: vec![PlaceInfo {
            id: "P0".into(),
            name: None,
        }],
        transitions: vec![TransitionInfo {
            id: "gen".into(),
            name: None,
            inputs: vec![],
            outputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
        }],
        initial_marking: vec![0],
    };
    let config = ExplorationConfig::new(1);
    let result = state_space_stats(&unbounded, &config);
    assert!(
        result.is_none(),
        "incomplete exploration must return None, got {result:?}"
    );
}

#[test]
fn test_quasi_liveness_verdict_false_when_reduction_removes_dead_transition() {
    let config = default_config();
    assert_eq!(
        quasi_liveness_verdict(&structurally_dead_transition_net(), &config),
        Verdict::False
    );
}

#[test]
fn test_liveness_verdict_false_when_reduction_removes_dead_transition() {
    let config = default_config();
    assert_eq!(
        liveness_verdict(&structurally_dead_transition_net(), &config),
        Verdict::False
    );
}

#[test]
fn test_deadlock_dead_transition_removed_no_deadlock() {
    let config = default_config();
    assert_eq!(
        deadlock_verdict(&structurally_dead_transition_net(), &config),
        Verdict::False
    );
}

#[test]
fn test_one_safe_dead_transition_removed_still_safe() {
    let config = default_config();
    assert_eq!(
        one_safe_verdict(&structurally_dead_transition_net(), &config, &[]),
        Verdict::True
    );
}

#[test]
fn test_stable_marking_dead_transition_removed_still_stable() {
    let config = default_config();
    assert_eq!(
        stable_marking_verdict(&structurally_dead_transition_net(), &config, &[]),
        Verdict::True
    );
}

#[test]
fn test_deadlock_isolated_place_removed_still_deadlocks() {
    let config = default_config();
    assert_eq!(
        deadlock_verdict(&isolated_place_net(1), &config),
        Verdict::True
    );
}

#[test]
fn test_liveness_isolated_place_removed_not_live() {
    let config = default_config();
    assert_eq!(
        liveness_verdict(&isolated_place_net(1), &config),
        Verdict::False
    );
}

#[test]
fn test_quasi_liveness_isolated_place_removed_still_quasi_live() {
    let config = default_config();
    assert_eq!(
        quasi_liveness_verdict(&isolated_place_net(1), &config),
        Verdict::True
    );
}

#[test]
fn test_one_safe_agglomerable_chain_preserves_verdict() {
    let config = default_config();
    assert_eq!(
        one_safe_verdict(&agglomerable_chain_net(), &config, &[]),
        Verdict::True
    );
}
