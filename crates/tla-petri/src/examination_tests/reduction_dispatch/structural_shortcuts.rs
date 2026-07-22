// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for shortcut reasoning: cyclic-safe, LP reduction, and agglomeration soundness.

use crate::explorer::ExplorationConfig;
use crate::output::Verdict;

use super::super::super::{deadlock_verdict, liveness_verdict, quasi_liveness_verdict};
use super::super::fixtures::{cyclic_safe_net, default_config, linear_deadlock_net};
use super::support::{agglomerable_chain_net, non_free_choice_cycle_net};

#[test]
fn test_quasi_liveness_structural_live_shortcut_with_tight_budget() {
    let config = ExplorationConfig::new(1);
    assert_eq!(
        quasi_liveness_verdict(&cyclic_safe_net(), &config),
        Verdict::True
    );
}

#[test]
fn test_liveness_structural_live_shortcut_with_tight_budget() {
    let config = ExplorationConfig::new(1);
    assert_eq!(liveness_verdict(&cyclic_safe_net(), &config), Verdict::True);
}

#[test]
fn test_liveness_structural_non_live_shortcut_with_tight_budget() {
    let config = ExplorationConfig::new(1);
    assert_eq!(
        liveness_verdict(&linear_deadlock_net(), &config),
        Verdict::False
    );
}

#[test]
fn test_quasi_liveness_does_not_use_structural_non_live_shortcut() {
    let config = default_config();
    assert_eq!(
        quasi_liveness_verdict(&linear_deadlock_net(), &config),
        Verdict::True
    );
}

#[test]
fn test_quasi_liveness_resolves_non_free_choice_net_after_lp_reduction() {
    // Quasi-liveness can still resolve this tiny non-free-choice net without
    // relying on quarantined structural reduction. Use enough explicit budget
    // that the test does not depend on ambient ay availability.
    let config = ExplorationConfig::new(3);
    assert_eq!(
        quasi_liveness_verdict(&non_free_choice_cycle_net(), &config),
        Verdict::True
    );
}

#[test]
fn test_liveness_fails_closed_on_non_free_choice_lp_reduction_with_tight_budget() {
    // Structural reductions are quarantined for Liveness because they can
    // suppress real firing behavior. With a budget too small for graph SCC
    // analysis, this non-free-choice net must fail closed instead of reviving
    // the old LP-reduction shortcut.
    //
    // NOTE: `ExplorationConfig::new(1)` caps the *state count*, not the wall
    // clock — `config.deadline()` is `None`. When the `dd-backend` feature is
    // enabled, the additive exact DD reachable-set nested-CTL lane (which runs
    // on its own deadline-less 5s budget, independent of the state-count cap)
    // SOUNDLY resolves this tiny bounded net to its true verdict TRUE (the same
    // value `test_liveness_resolves_non_free_choice_net_with_exact_budget`
    // confirms via the exact explicit pipeline) — a CANNOT_COMPUTE → CORRECT
    // lift, not a soundness break. Without the feature the explicit pipeline
    // still fails closed to CANNOT_COMPUTE under the tight state-count budget.
    let config = ExplorationConfig::new(1);
    let verdict = liveness_verdict(&non_free_choice_cycle_net(), &config);
    #[cfg(feature = "dd-backend")]
    assert_eq!(
        verdict,
        Verdict::True,
        "dd-backend: the exact DD lane resolves the tiny bounded net to its \
         true verdict TRUE (matches the exact-budget test)"
    );
    #[cfg(not(feature = "dd-backend"))]
    assert_eq!(verdict, Verdict::CannotCompute);
}

#[test]
fn test_liveness_resolves_non_free_choice_net_with_exact_budget() {
    let config = ExplorationConfig::new(3);
    assert_eq!(
        liveness_verdict(&non_free_choice_cycle_net(), &config),
        Verdict::True
    );
}

#[test]
fn test_quasi_liveness_not_false_when_agglomeration_removes_transition() {
    let config = default_config();
    // Both t0 and t1 fire once — the system IS quasi-live.
    // Agglomeration removes t0 but it's not dead.
    assert_eq!(
        quasi_liveness_verdict(&agglomerable_chain_net(), &config),
        Verdict::True
    );
}

#[test]
fn test_deadlock_agglomerable_chain_reaches_deadlock() {
    let config = default_config();
    // The chain fires once then reaches deadlock (p_out has a token, nothing consumes it).
    assert_eq!(
        deadlock_verdict(&agglomerable_chain_net(), &config),
        Verdict::True
    );
}
