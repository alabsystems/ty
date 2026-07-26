// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! WP-14 end-to-end: native dispatch becomes AUTHORITATIVE after per-action
//! burn-in (`TY_HYBRID_NATIVE_AUTHORITATIVE=1`).
//!
//! The M0/M1 shadow proved the native path is a complete per-instance
//! enumerator whose successor multiset byte-matches the interpreter's. This
//! file pins the conversion of that evidence into skipped interpreter work:
//!
//! 1. **State exactness with the interpreter actually skipped.** With a tiny
//!    burn-in (N=4) and heavy sampling (K=2), the run reaches exactly the
//!    interpreter's state set while `authoritative_dispatched > 0` (instances
//!    that never ran the interpreter) and `sampled_checks > 0` (post-flip
//!    instances that still ran the full differential) — with
//!    `sampled_mismatches == 0` and `mismatch_fallback == 0`.
//! 2. **The fail-back is real and permanent.** With the test-only corruption
//!    knob, the first sampled instance's native buffer is corrupted; the
//!    sampled differential must catch it (`sampled_mismatches == 1`), trip
//!    the permanent whole-run fail-back, and the run must STILL finish
//!    state-exact — the sampled instance kept the interpreter successors.
//! 3. **The gate is genuinely default-OFF and subordinate**: without
//!    `TY_HYBRID_NATIVE_AUTHORITATIVE=1` the identical binary never
//!    dispatches authoritatively; with an unreachable burn-in threshold no
//!    action ever flips.
//!
//! The model is the M1 compound-read model — the richest shape the hybrid
//! path services (flat writes + a compound read through the callout), so
//! authoritative dispatch is exercised over both flat and callout reads.

mod common;

use tla_check::ModelChecker;
use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

const WP14_TLA: &str = r#"
----------------------------- MODULE hybridwp14 -----------------------------
EXTENDS Naturals

VARIABLES x, y, f, data

IncX == /\ x < 3
        /\ x' = x + 1
        /\ UNCHANGED <<y, f, data>>

\* The M1 shape: writes only the flat `y`, reads the compound `f` through the
\* compound-read callout.
ReadF == /\ y < 6
         /\ y' = y + f[1][1]
         /\ UNCHANGED <<x, f, data>>

\* Writes the compound `f`. Never hybrid-eligible; keeps a compound-writing
\* interpreter action in the mix so authoritative dispatch must coexist with
\* interpreter-only actions in the same run.
Bump == /\ x > 0
        /\ f' = [f EXCEPT ![1] = [j \in 1..2 |-> 2]]
        /\ UNCHANGED <<x, y, data>>

\* An unbounded-universe Set keeping the whole-state layout off the fully-flat
\* path, so the run stays on the per-action hybrid dispatch this test is about.
Touch == /\ x > 0
         /\ data' = data \cup {x}
         /\ UNCHANGED <<x, y, f>>

Init == /\ x = 0
        /\ y = 0
        /\ f = [i \in 1..2 |-> [j \in 1..2 |-> 1]]
        /\ data = {}

Next == \/ IncX
        \/ ReadF
        \/ Bump
        \/ Touch
=============================================================================
"#;

const WP14_CFG: &str = r#"
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
        common::EnvVarGuard::set("TY_HYBRID_COMPOUND_READ", Some("1")),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
        common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1")),
        common::EnvVarGuard::remove("TY_HYBRID_M1_READS"),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE_AUTHORITATIVE"),
        common::EnvVarGuard::remove("TY_HYBRID_BURN_IN"),
        common::EnvVarGuard::remove("TY_HYBRID_SAMPLE"),
        common::EnvVarGuard::remove("TY_HYBRID_AUTHORITATIVE_INJECT_SAMPLED_CORRUPTION"),
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
    let _no_auth = common::EnvVarGuard::remove("TY_HYBRID_NATIVE_AUTHORITATIVE");
    let _no_callout = common::EnvVarGuard::remove("TY_HYBRID_COMPOUND_READ");
    let _no_m1 = common::EnvVarGuard::remove("TY_HYBRID_M1_READS");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    clear_for_test_reset();

    let module = common::parse_module(WP14_TLA);
    let config = Config::parse(WP14_CFG).expect("valid cfg");
    states_found("interpreter baseline", &check_module(&module, &config))
}

struct GatedOutcome {
    states: usize,
    mismatch: u64,
    native_dispatched: u64,
    native_errors: u64,
    authoritative_actions: u64,
    authoritative_dispatched: u64,
    sampled_checks: u64,
    sampled_mismatches: u64,
    burn_in_pending: u64,
    failback: bool,
}

/// One gated run with the given authoritative-mode env deltas applied on top
/// of the full hybrid gate stack.
fn gated_run(extra: &[(&'static str, Option<&str>)]) -> GatedOutcome {
    let _guards = base_guards();
    let _extra: Vec<common::EnvVarGuard> = extra
        .iter()
        .map(|(name, value)| common::EnvVarGuard::set(name, *value))
        .collect();
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(WP14_TLA);
    let config = Config::parse(WP14_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    let states = states_found("gated run", &result);
    let (_routed, mismatch, _projd, native_dispatched, _matched, _declined, native_errors) =
        checker.hybrid_dispatch_stats_for_testing();
    let (
        authoritative_actions,
        authoritative_dispatched,
        sampled_checks,
        sampled_mismatches,
        burn_in_pending,
        failback,
    ) = checker.hybrid_authoritative_stats_for_testing();
    GatedOutcome {
        states,
        mismatch,
        native_dispatched,
        native_errors,
        authoritative_actions,
        authoritative_dispatched,
        sampled_checks,
        sampled_mismatches,
        burn_in_pending,
        failback,
    }
}

/// The headline: with N=4 and K=2, actions flip after four clean
/// differentials, most later instances skip the interpreter, the sampled
/// remainder keeps the full differential — and the state set is EXACTLY the
/// interpreter's.
#[cfg_attr(test, ntest::timeout(180000))]
#[test]
fn authoritative_dispatch_after_burn_in_is_state_exact() {
    let baseline = interpreter_state_count();

    let out = gated_run(&[
        ("TY_HYBRID_NATIVE_AUTHORITATIVE", Some("1")),
        ("TY_HYBRID_BURN_IN", Some("4")),
        ("TY_HYBRID_SAMPLE", Some("2")),
    ]);

    assert_eq!(
        out.states, baseline,
        "authoritative native dispatch must not change the reachable-state set"
    );
    assert_eq!(
        out.mismatch, 0,
        "shadow-phase and sampled differentials must find ZERO divergences"
    );
    assert_eq!(out.native_errors, 0);
    assert_eq!(out.sampled_mismatches, 0);
    assert!(!out.failback, "no divergence, so no fail-back");
    assert!(
        out.authoritative_dispatched > 0,
        "THE WP-14 metric: some instances must have skipped the interpreter \
         entirely (native_dispatched={}, authoritative_dispatched={})",
        out.native_dispatched,
        out.authoritative_dispatched,
    );
    assert!(
        out.sampled_checks > 0,
        "with K=2, post-flip instances must keep sampling the full differential"
    );
    assert!(
        out.authoritative_actions > 0,
        "at least one action must have completed burn-in"
    );
    assert!(
        out.authoritative_dispatched < out.native_dispatched,
        "burn-in and sampled instances still run the differential, so the \
         authoritative subset must be strictly smaller than all native dispatches"
    );
}

/// The fail-back is real: a corrupted sampled buffer trips the permanent
/// whole-run fail-back — and the run STILL finishes state-exact, because the
/// sampled instance keeps the interpreter successors.
#[cfg_attr(test, ntest::timeout(180000))]
#[test]
fn injected_sampled_corruption_trips_permanent_failback_and_stays_state_exact() {
    let baseline = interpreter_state_count();

    let out = gated_run(&[
        ("TY_HYBRID_NATIVE_AUTHORITATIVE", Some("1")),
        ("TY_HYBRID_BURN_IN", Some("4")),
        ("TY_HYBRID_SAMPLE", Some("2")),
        (
            "TY_HYBRID_AUTHORITATIVE_INJECT_SAMPLED_CORRUPTION",
            Some("1"),
        ),
    ]);

    assert_eq!(
        out.states, baseline,
        "the run that trips the fail-back must STILL be state-exact — the \
         sampled instance's interpreter successors were the ones enqueued"
    );
    assert_eq!(
        out.sampled_mismatches, 1,
        "exactly one corruption is injected and the sampled differential must \
         catch exactly it"
    );
    assert!(
        out.failback,
        "a sampled mismatch must trip the permanent whole-run fail-back"
    );
    assert_eq!(
        out.authoritative_actions, 0,
        "after fail-back no action may remain authoritative"
    );
    assert!(
        out.mismatch > 0,
        "the corrupted buffer surfaces in the loud mismatch counter too"
    );
}

/// Default-off: the identical binary without `TY_HYBRID_NATIVE_AUTHORITATIVE`
/// never skips the interpreter and never samples — byte-identical M0/M1
/// shadow behavior.
#[cfg_attr(test, ntest::timeout(180000))]
#[test]
fn authoritative_gate_is_default_off() {
    let baseline = interpreter_state_count();

    let out = gated_run(&[]);

    assert_eq!(out.states, baseline, "gate-off must stay state-exact");
    assert_eq!(out.mismatch, 0);
    assert_eq!(
        out.authoritative_dispatched, 0,
        "without the gate nothing may skip the interpreter"
    );
    assert_eq!(out.sampled_checks, 0);
    assert_eq!(out.authoritative_actions, 0);
    assert_eq!(
        out.burn_in_pending, 0,
        "the burn-in machine must stay inert"
    );
    assert!(
        out.native_dispatched > 0,
        "the validated shadow itself must still dispatch natively"
    );
}

/// An unreachable burn-in threshold never flips: everything stays in the
/// validated shadow, `burn_in_pending` reports the actions still accumulating
/// evidence, and the run is state-exact.
#[cfg_attr(test, ntest::timeout(180000))]
#[test]
fn unreachable_burn_in_threshold_keeps_the_shadow_authoritative() {
    let baseline = interpreter_state_count();

    let out = gated_run(&[
        ("TY_HYBRID_NATIVE_AUTHORITATIVE", Some("1")),
        ("TY_HYBRID_BURN_IN", Some("1000000000")),
        ("TY_HYBRID_SAMPLE", Some("2")),
    ]);

    assert_eq!(out.states, baseline);
    assert_eq!(out.mismatch, 0);
    assert_eq!(out.authoritative_dispatched, 0);
    assert_eq!(out.authoritative_actions, 0);
    assert_eq!(out.sampled_checks, 0);
    assert!(
        out.burn_in_pending > 0,
        "natively-dispatched actions must be visible as burn-in pending"
    );
    assert!(out.native_dispatched > 0);
}
