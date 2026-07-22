// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Item 4 M1 integration: the compound-READ eligibility rule.
//!
//! M0 admitted an action only when `reads ∪ writes ⊆ flat-admissible`. That
//! rule declined every btree transition, because btree's 301,484 flat-write
//! transitions all READ compound vars (`childOf[n,k]`, `valOf[n,k]`,
//! `keysOf[n]`). M1 relaxes it to
//! `writes ⊆ flat-admissible ∧ reads ⊆ flat-admissible ∪ declared-callout-reads`.
//!
//! The model below is the minimal witness for that change:
//!
//! - `IncX` — reads/writes flat only. Eligible under BOTH rules.
//! - `ReadData` — **writes only the flat `y` while READING the compound `data`
//!   Set**. This is the M1 shape: declined by M0, admitted by M1.
//! - `Touch` — writes the compound `data`. Declined by BOTH rules, and must
//!   stay declined: reconstruction Arc-shares compound vars from the parent,
//!   which is only exact when no compound var is written. M1 is read-only.
//!
//! What is asserted:
//!
//! 1. The reachable-state set is IDENTICAL to the interpreter's under every
//!    switch combination (value- and fingerprint-exact, `mismatch_fallback == 0`).
//! 2. `ReadData` is hybrid-eligible under M1 and NOT under `TY_HYBRID_M1_READS=0`
//!    — the rule change is real and isolated to compound-reading actions.
//! 3. The compound-WRITING action still declines under M1.
//! 4. Publishing the parent context around the native call never destabilizes
//!    the differential.
//!
//! Note on scope: this file runs with `TY_HYBRID_COMPOUND_READ` UNSET, i.e.
//! with the lowering's callout emission switched off. `ReadData`'s native
//! admission therefore declines at the compiled-footprint dual gate (its
//! compiled read of a placeholder var is undeclared) and it runs through the
//! validated projection shadow. That is the fail-closed behaviour this test
//! pins: relaxing the AST rule widens the SHADOW, and can never widen the
//! NATIVE path ahead of an artifact that declared its callout reads.
//!
//! `ReadData` also reads `data` as a WHOLE (`1 \in data`), which the WP-03b
//! admission pre-scan classifies as an escape — so even with the gate on this
//! model would keep declining. The gate-on / actually-dispatching counterpart
//! lives in `hybrid_native_m1_compound_read_dispatch.rs`.

mod common;

use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

use tla_check::ModelChecker;

const M1_TLA: &str = r#"
----------------------------- MODULE hybridm1 -----------------------------
EXTENDS Naturals

VARIABLES x, y, data

IncX == /\ x < 3
        /\ x' = x + 1
        /\ UNCHANGED <<y, data>>

ReadData == /\ 1 \in data
            /\ y < 3
            /\ y' = y + 1
            /\ UNCHANGED <<x, data>>

Touch == /\ x > 0
         /\ data' = data \cup {x}
         /\ UNCHANGED <<x, y>>

Init == /\ x = 0
        /\ y = 0
        /\ data = {}

Next == \/ IncX
        \/ ReadData
        \/ Touch
=============================================================================
"#;

const M1_CFG: &str = r#"
INIT Init
NEXT Next
"#;

fn states_found(label: &str, result: &CheckResult) -> usize {
    match result {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("{label}: expected Success, got {other:?}"),
    }
}

fn base_guards() -> Vec<common::EnvVarGuard> {
    vec![
        common::EnvVarGuard::set("TY_TRUST_CG", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_FLAT_VIEW", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_NATIVE", Some("1")),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
        // Keep the run on the per-action interpreter BFS loop, where the
        // hybrid dispatch lives (mirrors the M0 integration test).
        common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1")),
        common::EnvVarGuard::remove("TY_TRUST_CG_BFS"),
        common::EnvVarGuard::remove("TY_COMPILED_BFS"),
        common::EnvVarGuard::remove("TY_NO_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_JIT"),
    ]
}

/// Interpreter ground truth for the differential.
fn interpreter_state_count() -> usize {
    let _no_trust_cg = common::EnvVarGuard::remove("TY_TRUST_CG");
    let _no_hybrid = common::EnvVarGuard::remove("TY_HYBRID_FLAT_VIEW");
    let _no_native = common::EnvVarGuard::remove("TY_HYBRID_NATIVE");
    let _no_m1 = common::EnvVarGuard::remove("TY_HYBRID_M1_READS");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    clear_for_test_reset();

    let module = common::parse_module(M1_TLA);
    let config = Config::parse(M1_CFG).expect("valid cfg");
    states_found("interpreter baseline", &check_module(&module, &config))
}

#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn m1_interpreter_baseline_is_stable() {
    let n = interpreter_state_count();
    assert!(n > 0, "the model must reach states");
    // Pin it so a spec edit cannot silently move the differential's ground
    // truth out from under the assertions below.
    assert_eq!(n, 36, "interpreter reachable-state count");
}

/// The rule change itself: an action that WRITES only flat vars while READING
/// a compound var becomes hybrid-eligible under M1 and is not under M0.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn m1_rule_admits_the_compound_reading_action_that_m0_declined() {
    let baseline = interpreter_state_count();

    // --- M0 rule (TY_HYBRID_M1_READS=0) ---
    let eligible_m0 = {
        let _guards = base_guards();
        let _m0 = common::EnvVarGuard::set("TY_HYBRID_M1_READS", Some("0"));
        clear_for_test_reset();
        tla_trust_cg::compile::clear_jit_cache();

        let module = common::parse_module(M1_TLA);
        let config = Config::parse(M1_CFG).expect("valid cfg");
        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.check();
        assert_eq!(
            states_found("M0 rule", &result),
            baseline,
            "the strict M0 rule must stay state-exact"
        );
        let (_, mismatch, _, _, _, _, _) = checker.hybrid_dispatch_stats_for_testing();
        assert_eq!(mismatch, 0, "M0 rule: zero divergences");
        checker.hybrid_eligible_action_count_for_testing()
    };

    // --- M1 rule (default) ---
    let eligible_m1 = {
        let _guards = base_guards();
        let _m1 = common::EnvVarGuard::remove("TY_HYBRID_M1_READS");
        clear_for_test_reset();
        tla_trust_cg::compile::clear_jit_cache();

        let module = common::parse_module(M1_TLA);
        let config = Config::parse(M1_CFG).expect("valid cfg");
        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.check();
        assert_eq!(
            states_found("M1 rule", &result),
            baseline,
            "the M1 rule must be state-exact vs the interpreter — relaxing \
             READ admission must never change the reachable-state set"
        );
        let (routed, mismatch, _, _, _, _, native_errors) =
            checker.hybrid_dispatch_stats_for_testing();
        assert_eq!(
            mismatch, 0,
            "M1 rule: the fail-closed differential must find ZERO \
             reconstructed/interpreter divergences"
        );
        assert_eq!(native_errors, 0, "no native runtime errors");
        assert!(routed > 0, "M1 rule must actually route successors");
        checker.hybrid_eligible_action_count_for_testing()
    };

    assert!(
        eligible_m1 > eligible_m0,
        "the M1 read rule must admit strictly more actions than M0 on a model \
         with a flat-write/compound-read action (M0={eligible_m0}, M1={eligible_m1})"
    );
}

/// The compound-WRITING action stays declined under M1. This is the soundness
/// boundary: M1 is read-only, because successor reconstruction Arc-shares
/// compound vars from the parent.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn m1_still_declines_the_compound_writing_action() {
    let baseline = interpreter_state_count();
    let _guards = base_guards();
    let _m1 = common::EnvVarGuard::remove("TY_HYBRID_M1_READS");
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(M1_TLA);
    let config = Config::parse(M1_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    assert_eq!(states_found("M1 compound-write", &result), baseline);

    // Three actions; `Touch` writes `data`, so at most two can be eligible.
    assert!(
        checker.hybrid_eligible_action_count_for_testing() <= 2,
        "an action writing a compound var must never be hybrid-eligible — \
         reconstruction Arc-shares compound vars from the parent"
    );
    let (_, mismatch, _, _, _, _, native_errors) = checker.hybrid_dispatch_stats_for_testing();
    assert_eq!(mismatch, 0);
    assert_eq!(native_errors, 0);
}

/// No compiled artifact declares a compound-read callout footprint yet, so the
/// native admission gate must stay at its strict M0 behaviour: relaxing the AST
/// rule widens the shadow, never the native path.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn m1_native_admission_requires_a_declared_callout_footprint() {
    let _guards = base_guards();
    let _m1 = common::EnvVarGuard::remove("TY_HYBRID_M1_READS");
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(M1_TLA);
    let config = Config::parse(M1_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    let _ = checker.check();

    for key in ["IncX", "ReadData", "Touch"] {
        assert!(
            checker
                .hybrid_declared_compound_read_vars_for_testing(key)
                .is_empty(),
            "{key}: no artifact may declare a compound-read footprint until the \
             lowering actually emits the callout"
        );
    }
    let (_, mismatch, _, _, _, _, native_errors) = checker.hybrid_dispatch_stats_for_testing();
    assert_eq!(
        mismatch, 0,
        "an undeclared compound read must decline native admission, not \
         produce a divergence"
    );
    assert_eq!(native_errors, 0);
}
