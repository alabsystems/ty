// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-backend soundness + multi-successor ("NextStateLoop") coverage
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
//! 1. The interpreter and trust-codegen native backend agree on the exact
//!    reachable-state count for `clean`.
//!
//! 2. `anneal` compiles through the runtime-range `NextStateLoopFn`; all four
//!    actions are natively covered, so parity cannot be supplied by a hidden
//!    per-action interpreter fallback.

mod common;

use tla_check::{CheckResult, Config, ModelChecker, StateGraphSnapshot};
use tla_eval::clear_for_test_reset;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunSummary {
    states_found: usize,
    initial_states: usize,
    transitions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphRun {
    summary: RunSummary,
    graph: StateGraphSnapshot,
    trust_cg_action_coverage: Option<(usize, usize)>,
    trust_cg_next_state_loop_recognized: Option<usize>,
}

/// Run one backend with graph capture enabled. The `exhaustive_env` mode also
/// clears the alternative compiled/flat-engine selectors; the minimal mode
/// changes only backend-selection knobs shared by the paired runs.
fn run_graph(trust_cg: bool, exhaustive_env: bool) -> GraphRun {
    let _trust_cg = common::EnvVarGuard::set("TY_TRUST_CG", trust_cg.then_some("1"));
    let _legacy_trust_cg = common::EnvVarGuard::remove("TY_trust_cg");
    let _trust_cg_bfs = common::EnvVarGuard::remove("TY_TRUST_CG_BFS");
    let _no_compiled = common::EnvVarGuard::set("TY_NO_COMPILED_BFS", (!trust_cg).then_some("1"));
    let _compiled_env = exhaustive_env.then(|| common::EnvVarGuard::remove("TY_COMPILED_BFS"));
    let _no_flat = exhaustive_env.then(|| common::EnvVarGuard::remove("TY_NO_FLAT_BFS"));
    let _flat_env = exhaustive_env.then(|| common::EnvVarGuard::remove("TY_FLAT_BFS"));
    let _no_jit = common::EnvVarGuard::remove("TY_JIT");
    let _auto_por = common::EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(CLEAN_TLA);
    let mut config = Config::parse(CLEAN_CFG).expect("valid cfg");
    config.use_compiled_bfs = Some(trust_cg);

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_store_states(false);
    checker.enable_state_graph_capture_for_testing();

    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("expected successful model check, got {other:?}"),
    };
    let graph = checker.state_graph_snapshot_for_testing();
    assert_eq!(
        graph.nodes.len(),
        stats.states_found,
        "captured graph nodes must match the explored state count"
    );
    assert_eq!(
        graph.successor_count, stats.transitions,
        "captured graph edges must match the explored transition count"
    );

    GraphRun {
        summary: RunSummary {
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
        },
        graph,
        trust_cg_action_coverage: checker.trust_cg_action_coverage_for_testing(),
        trust_cg_next_state_loop_recognized: checker
            .trust_cg_next_state_loop_recognized_for_testing(),
    }
}

// ============================================================================
// Interpreter / trust-cg parity with native runtime-range execution.
// ============================================================================

/// Interpreter baseline: `clean` has 63 reachable states with no violation.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn glowing_raccoon_interpreter_state_count() {
    let baseline = run_graph(false, true);
    assert_eq!(
        baseline.summary.states_found, EXPECTED_STATES,
        "interpreter reachable-state count"
    );
}

/// trust-cg native backend: must reach the identical reachable-state count as
/// the interpreter. `anneal` executes as a runtime-domain multi-successor
/// NextStateLoop action; all four actions compile natively.
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn glowing_raccoon_trust_cg_parity_with_native_anneal() {
    let baseline = run_graph(false, true);
    let trust_cg = run_graph(true, true);

    assert_eq!(
        trust_cg.summary, baseline.summary,
        "trust-cg run summary must match the interpreter"
    );
    assert_eq!(
        trust_cg.graph, baseline.graph,
        "trust-cg must reproduce the interpreter's exact reachable state graph"
    );

    // Native contract: every action, including the dynamic-range `anneal`, is
    // executable through trust-codegen.
    let (compiled, total) = trust_cg
        .trust_cg_action_coverage
        .expect("trust-cg run should record action coverage");
    assert_eq!(total, 4, "clean has four top-level actions");
    assert_eq!(
        compiled, 4,
        "heat/cool/extend and runtime-range anneal must all compile natively"
    );
    assert_eq!(
        trust_cg.trust_cg_next_state_loop_recognized,
        Some(0),
        "no runtime-domain NextStateLoop action should remain recognized-but-unsupported"
    );
}

// ============================================================================
// Minimal-environment regression for full native execution of `anneal`.
// ============================================================================

/// The same native/parity guarantee with only the public trust-cg selection
/// knobs set, guarding against accidentally depending on the more exhaustive
/// environment cleanup in the test above.
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn glowing_raccoon_trust_cg_anneal_compiles_natively_minimal_env() {
    let baseline = run_graph(false, false);
    let trust_cg = run_graph(true, false);

    assert_eq!(
        trust_cg.summary, baseline.summary,
        "minimal-env trust-cg run summary must match the interpreter"
    );
    assert_eq!(
        trust_cg.graph, baseline.graph,
        "minimal-env trust-cg must reproduce the interpreter's exact reachable state graph"
    );
    let (compiled, total) = trust_cg
        .trust_cg_action_coverage
        .expect("trust-cg run should record action coverage");
    assert_eq!(
        (compiled, total),
        (4, 4),
        "all four actions (including anneal) must compile natively via NextStateLoop"
    );
}
