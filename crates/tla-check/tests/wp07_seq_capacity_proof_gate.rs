// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! WP-07 (wishlist item 1/5 closure): `TY_SEQ_CAPACITY_PROOF` end-to-end.
//!
//! A CapProofDemo-style micro-spec with a FUNCTION-RANGE growing sequence
//! (`q \in [1..2 -> Seq(Int)]`, grown by `Append` through `FuncExcept`) and a
//! CHECKED capacity invariant whose bound is a state-free *expression*
//! (`Len(q[p]) <= Cardinality(KSet)`), which is exactly the rule-(a)
//! `<= <bounded-expr>` case the `TY_SEQ_CAPACITY_PROOF` gate exists for:
//!
//! - gate OFF: the bound expression does not fold, the sequence capacity stays
//!   `Observed`, the function-range sequence var is not flat-admissible, and
//!   NO action is hybrid-eligible/compiled (the historical fail-closed
//!   surface);
//! - gate ON: the checked invariant folds to a proven capacity, the var
//!   becomes flat-admissible, the growing-sequence actions become
//!   hybrid-eligible AND compile+dispatch natively — the compiled-action
//!   count rises from zero;
//! - BOTH arms report the IDENTICAL reachable-state count (differential-exact,
//!   `mismatch_fallback == 0`).

mod common;

use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

use tla_check::ModelChecker;

const CAP_PROOF_TLA: &str = r#"
--------------------------- MODULE capproofdemo ---------------------------
EXTENDS Naturals, Sequences, FiniteSets

VARIABLES q, data

Procs == {1, 2}
KSet == {1, 2, 3}

CapOK == /\ q \in [Procs -> Seq(Nat)]
         /\ \A p \in Procs : Len(q[p]) <= Cardinality(KSet)

Push(p) == /\ Len(q[p]) < Cardinality(KSet)
           /\ q' = [q EXCEPT ![p] = Append(q[p], Len(q[p]) + 1)]
           /\ UNCHANGED data

Touch == /\ data' = data \cup {1}
         /\ UNCHANGED q

Init == /\ q = [p \in Procs |-> <<>>]
        /\ data = {}

Next == \/ \E p \in Procs : Push(p)
        \/ Touch
=============================================================================
"#;

const CAP_PROOF_CFG: &str = r#"
INIT Init
NEXT Next
INVARIANT CapOK
"#;

fn states_found(label: &str, result: &CheckResult) -> usize {
    match result {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("{label}: expected Success, got {other:?}"),
    }
}

/// Shared hybrid-native env surface (mirrors `hybrid_native_m0_dispatch`); the
/// capacity-proof gate itself is toggled per test arm.
fn hybrid_env_guards() -> Vec<common::EnvVarGuard> {
    vec![
        common::EnvVarGuard::set("TY_TRUST_CG", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_FLAT_VIEW", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_NATIVE", Some("1")),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
        // Keep the run on the interpreter BFS loop: the hybrid per-action
        // native dispatch lives in the per-action successor path, and the
        // whole-state compiled-BFS loop is a different (fully-flat) engine
        // that must not race it in this test.
        common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1")),
        common::EnvVarGuard::remove("TY_TRUST_CG_BFS"),
        common::EnvVarGuard::remove("TY_COMPILED_BFS"),
        common::EnvVarGuard::remove("TY_NO_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_JIT"),
    ]
}

/// Interpreter baseline with every native/hybrid switch off: the reachable
/// state count both gate arms must reproduce exactly.
///
/// q[p] grows deterministically (<<>>, <<1>>, <<1,2>>, <<1,2,3>>) — 4 states
/// per p, independently — and data is {} or {1}: 4 * 4 * 2 = 32 states.
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn wp07_interpreter_baseline_state_count() {
    let _no_trust_cg = common::EnvVarGuard::remove("TY_TRUST_CG");
    let _no_hybrid = common::EnvVarGuard::remove("TY_HYBRID_FLAT_VIEW");
    let _no_native = common::EnvVarGuard::remove("TY_HYBRID_NATIVE");
    let _no_cap = common::EnvVarGuard::remove("TY_SEQ_CAPACITY_PROOF");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    clear_for_test_reset();

    let module = common::parse_module(CAP_PROOF_TLA);
    let config = Config::parse(CAP_PROOF_CFG).expect("valid cfg");
    let result = check_module(&module, &config);
    assert_eq!(
        states_found("interpreter baseline", &result),
        32,
        "q[p] in 4 deterministic growth states each, data in {{}} or {{1}}"
    );
}

/// Gate OFF: the `Cardinality(KSet)` bound must NOT fold, so the
/// function-range growing sequence stays `Observed`-bounded, nothing is
/// hybrid-eligible, and nothing dispatches natively — while the state count
/// stays exact (fail-closed means slow, never wrong).
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn wp07_gate_off_stays_fail_closed_with_exact_states() {
    let _guards = hybrid_env_guards();
    let _no_cap = common::EnvVarGuard::remove("TY_SEQ_CAPACITY_PROOF");
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(CAP_PROOF_TLA);
    let config = Config::parse(CAP_PROOF_CFG).expect("valid cfg");

    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    assert_eq!(
        states_found("gate-off run", &result),
        32,
        "gate OFF must reproduce the interpreter state count exactly"
    );
    assert_eq!(
        checker.hybrid_eligible_action_count_for_testing(),
        0,
        "without the capacity proof the growing-sequence writes stay non-flat-admissible, \
         so no action may be hybrid-eligible (fail closed)"
    );
    let (_routed, mismatch_fallback, _decl, native_dispatched, _matched, _ndecl, native_errors) =
        checker.hybrid_dispatch_stats_for_testing();
    assert_eq!(
        native_dispatched, 0,
        "gate OFF: no compiled action exists, so nothing may dispatch natively"
    );
    assert_eq!(native_errors, 0);
    assert_eq!(mismatch_fallback, 0);
}

/// Gate ON: the checked `Len(q[p]) <= Cardinality(KSet)` invariant folds to a
/// proven capacity, the function-range sequence var becomes flat-admissible,
/// the compiled/hybrid-eligible action count RISES from zero, native dispatch
/// actually happens, and the state count is IDENTICAL to both the interpreter
/// baseline and the gate-off arm.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn wp07_gate_on_compiles_and_dispatches_with_exact_states() {
    let _guards = hybrid_env_guards();
    let _cap = common::EnvVarGuard::set("TY_SEQ_CAPACITY_PROOF", Some("1"));
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(CAP_PROOF_TLA);
    let config = Config::parse(CAP_PROOF_CFG).expect("valid cfg");

    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    assert_eq!(
        states_found("gate-on run", &result),
        32,
        "gate ON must reproduce the interpreter state count exactly"
    );
    assert!(
        checker.hybrid_eligible_action_count_for_testing() > 0,
        "the proven capacity must make the growing-sequence actions hybrid-eligible \
         (the compiled-action count rises from zero)"
    );
    assert!(
        checker.hybrid_native_cache_ready_for_testing(),
        "the hybrid-layout native action cache must be built and non-empty"
    );
    let (
        routed,
        mismatch_fallback,
        _decl,
        native_dispatched,
        native_matched,
        _ndecl,
        native_errors,
    ) = checker.hybrid_dispatch_stats_for_testing();
    assert!(
        native_dispatched > 0,
        "the compiled growing-sequence actions must actually execute natively"
    );
    assert!(native_matched > 0);
    assert!(routed > 0);
    assert_eq!(
        mismatch_fallback, 0,
        "the fail-closed differential must find ZERO native/interpreter divergences"
    );
    assert_eq!(native_errors, 0);
}
