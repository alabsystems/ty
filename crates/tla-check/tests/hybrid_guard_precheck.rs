// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! WP-29 lever 1 / WP-34: the per-(parent, action) enabling PRE-CHECK, and the
//! WP-34 batch-consumer lazy-normalization elision.
//!
//! The pre-check evaluates a syntactically extracted state-only guard `g` with
//! `action => g` before entering the enumerator, and skips the enumeration when
//! `g` is FALSE. It is a strict UNDER-approximation of enabledness, so the only
//! thing these tests can (and must) pin is that it never changes a verdict:
//!
//! * the reachable-state count is EXACTLY the baseline's, including while POR
//!   is engaged (WP-34 admits the pre-check to the POR path, where the ample-set
//!   computation and the whole-Next parity self-check both consume the
//!   per-action successor sets);
//! * the lever is not inert — it actually decides "definitely disabled" on this
//!   model, so the exactness above is evidence rather than a vacuous pass.
//!
//! The model is a `pc`-discriminated state machine: every action leads with a
//! state-only `pc = "..."` conjunct, which is exactly the shape the extractor
//! accepts, and at most one action is enabled per state — so the other two are
//! provably disabled on every parent. `f` is a FUNCTION-valued variable written
//! with `EXCEPT`, which makes the spec lazy-producing
//! (`spec_may_produce_lazy_values`) and therefore routes every successor
//! through the WP-34 parent-delta normalization in the batch consumer.

mod common;

use tla_check::{check_module, CheckResult, Config, ModelChecker};
use tla_eval::clear_for_test_reset;

const GUARD_TLA: &str = r#"
----------------------------- MODULE guardprecheck -----------------------------
EXTENDS Naturals

VARIABLES pc, x, f

Idx == 0..2

Step == /\ pc = "step"
        /\ \E d \in {1, 2} : x' = (x + d) % 3
        /\ pc' = "store"
        /\ UNCHANGED f

Store == /\ pc = "store"
         /\ \E i \in Idx : f' = [f EXCEPT ![i] = x]
         /\ pc' = "reset"
         /\ UNCHANGED x

Reset == /\ pc = "reset"
         /\ pc' = "step"
         /\ UNCHANGED <<x, f>>

Init == /\ pc = "step"
        /\ x = 0
        /\ f = [i \in Idx |-> 0]

Next == Step \/ Store \/ Reset
================================================================================
"#;

const GUARD_CFG: &str = r#"
INIT Init
NEXT Next
"#;

fn states_found(label: &str, result: &CheckResult) -> usize {
    match result {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("{label}: expected Success, got {other:?}"),
    }
}

fn hybrid_env_guards() -> Vec<common::EnvVarGuard> {
    vec![
        common::EnvVarGuard::set("TY_TRUST_CG", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_FLAT_VIEW", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_NATIVE", Some("1")),
        common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1")),
        common::EnvVarGuard::remove("TY_TRUST_CG_BFS"),
        common::EnvVarGuard::remove("TY_COMPILED_BFS"),
        common::EnvVarGuard::remove("TY_NO_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_JIT"),
    ]
}

/// Baseline: reachable-state count with every hybrid/native switch and both
/// reductions off. This is the number every arm below must reproduce exactly.
fn baseline_state_count() -> usize {
    let _no_trust_cg = common::EnvVarGuard::remove("TY_TRUST_CG");
    let _no_hybrid = common::EnvVarGuard::remove("TY_HYBRID_FLAT_VIEW");
    let _no_native = common::EnvVarGuard::remove("TY_HYBRID_NATIVE");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    let _auto_sym = common::EnvVarGuard::set("TY_AUTO_SYMMETRY", Some("0"));
    clear_for_test_reset();

    let module = common::parse_module(GUARD_TLA);
    let config = Config::parse(GUARD_CFG).expect("valid cfg");
    states_found("interpreter baseline", &check_module(&module, &config))
}

/// The pre-check runs, decides "definitely disabled" on real instances, and the
/// reachable-state count is unchanged.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn guard_precheck_is_state_exact_and_not_inert() {
    let expected = baseline_state_count();

    let _guards = hybrid_env_guards();
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    let _auto_sym = common::EnvVarGuard::set("TY_AUTO_SYMMETRY", Some("0"));
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(GUARD_TLA);
    let config = Config::parse(GUARD_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    // Coverage tracking routes the run through the per-action batch engine
    // (the same engine POR and hybrid native dispatch force), which is where
    // the pre-check and the batch consumer live.
    checker.set_track_coverage(true);
    let result = checker.check();

    assert_eq!(
        states_found("pre-check run", &result),
        expected,
        "the enabling pre-check must not change the reachable-state count"
    );

    let (calls, skips) = checker.hybrid_guard_precheck_counters_for_testing();
    assert!(
        calls > 0,
        "the pre-check must actually run on this model (calls > 0)"
    );
    assert!(
        skips > 0,
        "the pre-check must actually decide 'definitely disabled' on this \
         pc-discriminated model, or the exactness above is vacuous"
    );
    assert!(
        skips <= calls,
        "skips ({skips}) cannot exceed calls ({calls})"
    );
}

/// WP-34: the same, with POR engaged. Under POR the per-action successor sets
/// feed the ample-set computation AND the fail-closed whole-Next parity
/// self-check, so this is the arm that pins the pre-check as admissible to that
/// protocol.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn guard_precheck_under_por_is_state_exact() {
    let expected = baseline_state_count();

    let _guards = hybrid_env_guards();
    let _auto_sym = common::EnvVarGuard::set("TY_AUTO_SYMMETRY", Some("0"));
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(GUARD_TLA);
    let mut config = Config::parse(GUARD_CFG).expect("valid cfg");
    config.por_enabled = true;

    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();

    assert_eq!(
        states_found("pre-check under POR", &result),
        expected,
        "the pre-check must not change the reachable-state count under POR"
    );
    let (calls, skips) = checker.hybrid_guard_precheck_counters_for_testing();
    assert!(
        calls > 0 && skips > 0,
        "the pre-check must run and decide under POR (calls={calls}, skips={skips})"
    );
}

/// WP-34 lever 2: the batch consumer normalizes lazy values by walking only the
/// variables a successor actually rewrote. `f` is function-valued and written
/// with `EXCEPT`, so the spec is lazy-producing and every successor goes through
/// the parent-delta path; the elided variables must not change the verdict.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn batch_consumer_parent_delta_normalization_is_state_exact() {
    let expected = baseline_state_count();

    let _guards = hybrid_env_guards();
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    let _auto_sym = common::EnvVarGuard::set("TY_AUTO_SYMMETRY", Some("0"));
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(GUARD_TLA);
    let config = Config::parse(GUARD_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    checker.set_track_coverage(true);
    let result = checker.check();

    assert_eq!(
        states_found("parent-delta normalization", &result),
        expected,
        "eliding the lazy-value scan for variables proven identical to a \
         lazy-free parent variable must not change the reachable-state count"
    );

    let (succ, scanned, total) = checker.hybrid_consume_lazy_counters_for_testing();
    assert!(
        succ > 0,
        "successors must flow through the batch consumer on this spec"
    );
    assert!(
        scanned < total,
        "the parent-delta elision must skip at least one variable scan \
         (scanned={scanned}, total={total})"
    );
}
