// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Item 4 M1 end-to-end: an action that WRITES only flat vars while READING a
//! compound var compiles AND dispatches natively (WP-03b).
//!
//! `hybrid_native_m1_compound_read.rs` pins the *eligibility rule*: the M1 gate
//! admits the flat-write/compound-read shape, and — while the lowering emitted
//! no callout — such an action still declined at the compiled-footprint dual
//! gate and ran through the validated projection shadow.
//!
//! This file pins the other half. With `TY_HYBRID_COMPOUND_READ=1` the lowering
//! emits `tla_hybrid_compound_apply1_i64` for the admitted read, declares the
//! placeholder var in the artifact's compound-read footprint, and the action
//! actually reaches `native_dispatched`.
//!
//! `f` is a compound function variable that `Bump` writes, so the hybrid
//! projection cannot keep it in a flat slot: it is demoted to a
//! `CompoundLayout::Dynamic` placeholder whose buffer slot carries no
//! information. `ReadF` reads `f[1]` — a chain-terminating scalar leaf — and
//! writes only the flat `y`, which is exactly the shape M1 exists to service.
//!
//! What is asserted, in order of what would break first if the callout were
//! wrong:
//!
//! 1. **Value exactness.** The reachable-state set under every switch
//!    combination equals the interpreter's, with `mismatch_fallback == 0`. A
//!    callout returning the wrong leaf would show up here as a divergence, not
//!    a crash — the reconstructed successor is compared against the
//!    interpreter's.
//! 2. **`native_dispatched > 0`.** The number item 4 exists to move off zero.
//! 3. **The declared footprint names the placeholder var**, and is EMPTY with
//!    the gate off — the artifact cannot claim a callout read it did not emit.
//! 4. **The gate is genuinely default-OFF**: same binary, gate unset, zero
//!    declarations and the same state count.

mod common;

use tla_check::ModelChecker;
use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

const M1_TLA: &str = r#"
----------------------------- MODULE hybridm1cr -----------------------------
EXTENDS Naturals

VARIABLES x, y, f, data

IncX == /\ x < 3
        /\ x' = x + 1
        /\ UNCHANGED <<y, f, data>>

\* The M1 shape: writes only the flat `y`, reads the compound `f`. The read is
\* a CURRIED two-key chain terminating in an integer leaf, so it lowers to one
\* fused `tla_hybrid_compound_apply2_i64` — the intermediate `f[1]` is never
\* materialized and never crosses the FFI boundary.
ReadF == /\ y < 6
         /\ y' = y + f[1][1]
         /\ UNCHANGED <<x, f, data>>

\* Writes the compound `f`. Never M1-eligible itself: reconstruction
\* Arc-shares compound vars from the parent, which a compound write invalidates.
Bump == /\ x > 0
        /\ f' = [f EXCEPT ![1] = [j \in 1..2 |-> 2]]
        /\ UNCHANGED <<x, y, data>>

\* An unbounded-universe Set. Its only job is to keep the whole-state layout
\* off the fully-flat path, so the run stays on the per-action hybrid dispatch
\* this test is about rather than the whole-state compiled BFS engine.
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
        common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1")),
        common::EnvVarGuard::remove("TY_HYBRID_M1_READS"),
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
    let _no_callout = common::EnvVarGuard::remove("TY_HYBRID_COMPOUND_READ");
    let _no_m1 = common::EnvVarGuard::remove("TY_HYBRID_M1_READS");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));
    clear_for_test_reset();

    let module = common::parse_module(M1_TLA);
    let config = Config::parse(M1_CFG).expect("valid cfg");
    states_found("interpreter baseline", &check_module(&module, &config))
}

/// One gated run: returns (states, routed, mismatch, native_dispatched,
/// native_errors, declared compound-read vars for `ReadF`).
fn gated_run(callout: bool) -> (usize, u64, u64, u64, u64, Vec<u16>) {
    let _guards = base_guards();
    let _callout = common::EnvVarGuard::set(
        "TY_HYBRID_COMPOUND_READ",
        if callout { Some("1") } else { None },
    );
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(M1_TLA);
    let config = Config::parse(M1_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    let states = states_found("gated run", &result);
    let (routed, mismatch, _, native_dispatched, _, _, native_errors) =
        checker.hybrid_dispatch_stats_for_testing();
    let declared = checker.hybrid_declared_compound_read_vars_for_testing("ReadF");
    (
        states,
        routed,
        mismatch,
        native_dispatched,
        native_errors,
        declared,
    )
}

#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn m1_dispatch_interpreter_baseline_is_stable() {
    let n = interpreter_state_count();
    assert!(n > 0, "the model must reach states");
}

/// The headline: with the callout emitted, the compound-reading action leaves
/// the shadow and executes natively — value-exact, zero divergences.
#[cfg_attr(test, ntest::timeout(180000))]
#[test]
fn m1_compound_read_callout_dispatches_natively_and_stays_state_exact() {
    let baseline = interpreter_state_count();

    let (states, routed, mismatch, native_dispatched, native_errors, declared) = gated_run(true);

    assert_eq!(
        states, baseline,
        "emitting the compound-read callout must not change the reachable-state \
         set — a wrong leaf would surface here as a divergence"
    );
    assert_eq!(
        mismatch, 0,
        "the fail-closed differential must find ZERO reconstructed/interpreter \
         divergences"
    );
    assert_eq!(
        native_errors, 0,
        "a latched compound-read status (CR_ERR_*) voids the execution; there \
         must be none"
    );
    assert!(routed > 0, "the hybrid path must actually route successors");
    assert!(
        native_dispatched > 0,
        "THE item 4 metric: an action that writes only flat vars while reading a \
         compound var must now DISPATCH natively, not merely compile \
         (native_dispatched={native_dispatched})"
    );
    assert!(
        !declared.is_empty(),
        "the artifact must declare the placeholder var it services through the \
         callout — an undeclared compound read is a hard decline at the dual gate"
    );
}

/// The gate is genuinely default-OFF: unset, the identical binary declares
/// nothing and the M1 admission gate degrades exactly to M0.
#[cfg_attr(test, ntest::timeout(180000))]
#[test]
fn m1_compound_read_callout_is_default_off() {
    let baseline = interpreter_state_count();

    let (states, _routed, mismatch, _dispatched, native_errors, declared) = gated_run(false);

    assert_eq!(states, baseline, "gate-off must stay state-exact");
    assert_eq!(mismatch, 0);
    assert_eq!(native_errors, 0);
    assert!(
        declared.is_empty(),
        "with the gate off no artifact may declare a compound-read footprint, \
         so ty's M1 gate degrades to M0's strict flat-admissible read rule"
    );
}

/// M1 is read-only: the action that WRITES the compound var must never become
/// hybrid-eligible, gate or no gate.
#[cfg_attr(test, ntest::timeout(180000))]
#[test]
fn m1_compound_writing_action_stays_declined_with_the_callout_on() {
    let baseline = interpreter_state_count();
    let _guards = base_guards();
    let _callout = common::EnvVarGuard::set("TY_HYBRID_COMPOUND_READ", Some("1"));
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(M1_TLA);
    let config = Config::parse(M1_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    assert_eq!(states_found("compound-write", &result), baseline);

    assert!(
        checker
            .hybrid_declared_compound_read_vars_for_testing("Bump")
            .is_empty(),
        "an action writing the compound var must never declare a callout read"
    );
    let (_, mismatch, _, _, _, _, native_errors) = checker.hybrid_dispatch_stats_for_testing();
    assert_eq!(mismatch, 0);
    assert_eq!(native_errors, 0);
}
