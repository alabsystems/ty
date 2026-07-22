// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-backend soundness + multi-successor ("NextStateLoop") scaffold target
//! for the `glowingRaccoon/clean.tla` spec.
//!
//! The `anneal` action contains a runtime-domain inner existential:
//!
//! ```tla
//! anneal == /\ tee = "Hot" /\ tee' = "Warm" /\ UNCHANGED dna
//!           /\ \E k \in 1 .. natMin(primer, template) :
//!                /\ primer'   = primer - k
//!                /\ template' = template - k
//!                /\ hybrid'   = hybrid + k
//! natMin(i, j) == IF i < j THEN i ELSE j
//! ```
//!
//! The domain bound `natMin(primer, template)` is a *runtime* quantity, so the
//! domain `1 .. natMin(...)` is an integer `Range` whose upper bound is a
//! runtime `Call`. This action yields one successor per `k`, which the
//! single-successor native next-state ABI cannot express: it can only be
//! compile-time unrolled when the domain's values are known at compile time.
//!
//! This file pins two guarantees:
//!
//! 1. (active) Cross-backend parity: the interpreter and the trust-codegen
//!    native backend agree on the reachable-state count for `clean`. With the
//!    native multi-successor codegen still unimplemented, `anneal` is
//!    *recognized* as the NextStateLoop ABI target and routed to the
//!    interpreter (fail-closed) — the run must NOT drop or fabricate states.
//!    The other three actions (`heat`, `cool`, `extend`) compile natively.
//!
//! 2. (`#[ignore]`d) The eventual target: once the multi-successor
//!    `tla_jit_abi::NextStateLoopFn` codegen lands, `anneal` should compile
//!    natively and every action instance should be covered
//!    (`compiled == total`) while parity is preserved. This test documents the
//!    finish line; un-ignore it when the codegen is implemented.

mod common;

use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

use tla_check::ModelChecker;

/// The `clean.tla` source, embedded so the test is self-contained and does not
/// depend on `~/tlaplus-examples` being present.
const CLEAN_TLA: &str = r#"
----------------------------- MODULE clean -----------------------------
EXTENDS Naturals

CONSTANTS DNA, PRIMER

VARIABLES tee, primer, dna, template, hybrid

vars == << tee, primer, dna, template, hybrid >>

natMin(i,j) == IF i < j THEN i ELSE j

heat == /\ tee = "Hot"
        /\ tee' = "TooHot"
        /\ primer' = primer + hybrid
        /\ dna' = 0
        /\ template' = template + hybrid + 2 * dna
        /\ hybrid' = 0

cool == /\ tee = "TooHot"
        /\ tee' = "Hot"
        /\ UNCHANGED << primer, dna, template, hybrid >>

anneal == /\ tee = "Hot"
          /\ tee' = "Warm"
          /\ UNCHANGED dna
          /\ \E k \in 1..natMin(primer, template) :
             /\ primer' = primer - k
             /\ template' = template - k
             /\ hybrid' = hybrid + k

extend == /\ tee = "Warm"
            /\ tee' = "Hot"
            /\ UNCHANGED <<primer, template>>
            /\ dna' = dna + hybrid
            /\ hybrid' = 0

Init == /\ tee = "Hot"
        /\ primer = PRIMER
        /\ dna = DNA
        /\ template = 0
        /\ hybrid = 0

Next ==  \/ heat
         \/ cool
         \/ anneal
         \/ extend

Spec == /\ Init
        /\ [][Next]_vars

TypeOK ==
    /\ tee \in {"Warm", "Hot", "TooHot"}
    /\ primer \in Nat
    /\ dna \in Nat
    /\ template \in Nat
    /\ hybrid \in Nat

primerPositive == (primer >= 0)

preservationInvariant == template + primer + 2*(dna + hybrid) = PRIMER + 2 * DNA
=============================================================================
"#;

/// A no-PROPERTY config using direct `INIT`/`NEXT` (equivalent to
/// `SPECIFICATION Spec` for the safety-only reachable-state set, but resolvable
/// without spec-formula resolution helpers in the test harness).
///
/// The shipped `clean.cfg` declares `PROPERTY preservationProperty`, which
/// forces interpreter successor evaluation for implied-action checking and
/// disables native compilation wholesale (see
/// `implied_actions_require_interpreter_eval`). Dropping the property keeps the
/// same reachable-state set while letting the native action compiler actually
/// run, so the `anneal` NextStateLoop recognition fires.
const CLEAN_CFG: &str = r#"
CONSTANTS
  DNA = 5
  PRIMER = 5
INVARIANTS TypeOK primerPositive preservationInvariant
INIT Init
NEXT Next
"#;

/// Expected reachable-state count for `DNA = 5, PRIMER = 5` (verified with both
/// the interpreter and trust-cg backends).
const EXPECTED_STATES: usize = 63;

fn states_found(label: &str, result: &CheckResult) -> usize {
    match result {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("{label}: expected Success, got {other:?}"),
    }
}

// ============================================================================
// Active: interpreter / trust-cg parity. `anneal` falls back (fail-closed).
// ============================================================================

/// Interpreter baseline: `clean` has 63 reachable states with no violation.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn glowing_raccoon_interpreter_state_count() {
    let _no_compiled = common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1"));
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    clear_for_test_reset();

    let module = common::parse_module(CLEAN_TLA);
    let config = Config::parse(CLEAN_CFG).expect("valid cfg");
    let result = check_module(&module, &config);
    assert_eq!(
        states_found("interpreter baseline", &result),
        EXPECTED_STATES,
        "interpreter reachable-state count"
    );
}

/// trust-cg native backend: must reach the identical reachable-state count as
/// the interpreter. `anneal` is recognized as the runtime-domain
/// multi-successor (NextStateLoop) shape and routed to the interpreter
/// (fail-closed); the other three actions compile natively. The whole point is
/// that recognizing-but-falling-back does NOT change the state set.
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn glowing_raccoon_trust_cg_parity_with_anneal_fallback() {
    let _trust_cg = common::EnvVarGuard::set("TY_TRUST_CG", Some("1"));
    let _trust_cg_bfs = common::EnvVarGuard::remove("TY_TRUST_CG_BFS");
    let _no_compiled = common::EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _compiled_env = common::EnvVarGuard::remove("TY_COMPILED_BFS");
    let _no_flat = common::EnvVarGuard::remove("TY_NO_FLAT_BFS");
    let _flat_env = common::EnvVarGuard::remove("TY_FLAT_BFS");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(CLEAN_TLA);
    let mut config = Config::parse(CLEAN_CFG).expect("valid cfg");
    config.use_compiled_bfs = Some(true);

    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();

    let coverage = checker.trust_cg_action_coverage_for_testing();
    let recognized = checker.trust_cg_next_state_loop_recognized_for_testing();

    assert_eq!(
        states_found("trust-cg native run", &result),
        EXPECTED_STATES,
        "trust-cg reachable-state count must match the interpreter"
    );

    // Scaffold contract: `anneal` is recognized as a NextStateLoop target and
    // routed to the interpreter, so exactly 3 of 4 actions compile natively.
    let (compiled, total) = coverage.expect("trust-cg run should record action coverage");
    assert_eq!(total, 4, "clean has four top-level actions");
    assert_eq!(
        compiled, 3,
        "heat/cool/extend compile natively; anneal falls back (fail-closed)"
    );
    assert_eq!(
        recognized,
        Some(1),
        "anneal must be recognized as the runtime-domain multi-successor (NextStateLoop) target"
    );
}

// ============================================================================
// Target (ignored): full native multi-successor execution of `anneal`.
// ============================================================================

/// TARGET for the multi-successor `NextStateLoopFn` codegen.
///
/// When a sound native multi-successor lowering lands (a per-iteration loop
/// that re-seeds each successor from `state_in` and pushes one record per `k`
/// into a `tla_jit_abi::NextStateLoopSink`, distinct from the boolean
/// any-witness exists loop), `anneal` should compile natively: all four action
/// instances covered AND parity preserved. Remove `#[ignore]` then.
#[ignore = "multi-successor NextStateLoop codegen not yet implemented; scaffold falls back for anneal"]
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn glowing_raccoon_trust_cg_anneal_compiles_natively_target() {
    let _trust_cg = common::EnvVarGuard::set("TY_TRUST_CG", Some("1"));
    let _no_compiled = common::EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(CLEAN_TLA);
    let mut config = Config::parse(CLEAN_CFG).expect("valid cfg");
    config.use_compiled_bfs = Some(true);

    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    let coverage = checker.trust_cg_action_coverage_for_testing();

    assert_eq!(
        states_found("trust-cg native run (target)", &result),
        EXPECTED_STATES,
        "parity must be preserved once anneal compiles natively"
    );
    let (compiled, total) = coverage.expect("trust-cg run should record action coverage");
    assert_eq!(
        (compiled, total),
        (4, 4),
        "all four actions (including anneal) must compile natively via NextStateLoop"
    );
}
