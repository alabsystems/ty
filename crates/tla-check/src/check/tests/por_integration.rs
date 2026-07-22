// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use tla_core::{lower, parse_to_syntax_tree, FileId};

const DISJOINT_COUNTERS_SPEC: &str = r#"
---- MODULE PorDisjointCounters ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ UNCHANGED y

IncY ==
    /\ y < 2
    /\ y' = y + 1
    /\ UNCHANGED x

Next == IncX \/ IncY
====
"#;

const DISJOINT_COUNTERS_WITH_INV_SPEC: &str = r#"
---- MODULE PorDisjointCountersInv ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ UNCHANGED y

IncY ==
    /\ y < 2
    /\ y' = y + 1
    /\ UNCHANGED x

Next == IncX \/ IncY
Inv == x + y < 4
====
"#;

const DEPENDENT_ACTIONS_SPEC: &str = r#"
---- MODULE PorDependentActions ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

ActionA ==
    /\ x < 2
    /\ x' = x + 1
    /\ UNCHANGED y

ActionB ==
    /\ x < 2
    /\ x' = x * 2
    /\ UNCHANGED y

Next == ActionA \/ ActionB
====
"#;

const THREE_DISJOINT_COUNTERS_SPEC: &str = r#"
---- MODULE PorThreeDisjointCounters ----
EXTENDS Naturals

VARIABLE x, y, z

Init == x = 0 /\ y = 0 /\ z = 0

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ UNCHANGED <<y, z>>

IncY ==
    /\ y < 2
    /\ y' = y + 1
    /\ UNCHANGED <<x, z>>

IncZ ==
    /\ z < 2
    /\ z' = z + 1
    /\ UNCHANGED <<x, y>>

Next == IncX \/ IncY \/ IncZ
====
"#;

const SIMPLE_POR_STATS_SPEC: &str = r#"
---- MODULE PorStatsSimple ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

IncX ==
    /\ x < 1
    /\ x' = x + 1
    /\ UNCHANGED y

IncY ==
    /\ y < 1
    /\ y' = y + 1
    /\ UNCHANGED x

Next == IncX \/ IncY
====
"#;

fn make_config(por_enabled: bool, invariants: &[&str]) -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: invariants.iter().map(|name| (*name).to_string()).collect(),
        por_enabled,
        check_deadlock: false,
        ..Default::default()
    }
}

/// Config with auto-POR explicitly disabled. Use this when you need a "truly no POR"
/// baseline (e.g., for comparing state counts with and without reduction).
fn make_config_no_por(invariants: &[&str]) -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: invariants.iter().map(|name| (*name).to_string()).collect(),
        por_enabled: false,
        auto_por: Some(false),
        check_deadlock: false,
        ..Default::default()
    }
}

/// Config with auto-POR EXPLICITLY opted in (`auto_por: Some(true)`). Auto-POR
/// is also ON by default (safe since the C3 BFS fresh-successor proviso landed
/// at all ample call sites — see `resolve_auto_por`); the explicit opt-in makes
/// reduction-behavior tests immune to the ambient `TY_AUTO_POR` env var and
/// pins the EXPLICIT-signal path (which the native-fused POR release must
/// honor).
fn make_config_auto_por(invariants: &[&str]) -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: invariants.iter().map(|name| (*name).to_string()).collect(),
        por_enabled: false,
        auto_por: Some(true),
        check_deadlock: false,
        ..Default::default()
    }
}

fn run_check(src: &str, config: Config) -> CheckResult {
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    assert!(
        lower_result.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lower_result.errors
    );
    let module = lower_result.module.unwrap();

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.check()
}

fn expect_success(result: CheckResult) -> CheckStats {
    match result {
        CheckResult::Success(stats) => stats,
        other => panic!("Expected success, got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_disjoint_counters_reduces_state_space() {
    let without_por = expect_success(run_check(DISJOINT_COUNTERS_SPEC, make_config_no_por(&[])));
    assert_eq!(without_por.states_found, 9);
    assert_eq!(without_por.transitions, 12);

    let with_por = expect_success(run_check(DISJOINT_COUNTERS_SPEC, make_config(true, &[])));
    assert_eq!(with_por.states_found, 5);
    assert!(with_por.states_found < without_por.states_found);
    assert!(with_por.transitions < without_por.transitions);
    assert_eq!(with_por.por_reduction.action_count, 2);
    assert_eq!(with_por.por_reduction.independent_pairs, 1);
    assert_eq!(with_por.por_reduction.total_pairs, 1);
    assert!(with_por.por_reduction.states_reduced > 0);
    assert!(with_por.por_reduction.actions_skipped > 0);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_preserves_invariant_detection() {
    let result = run_check(DISJOINT_COUNTERS_WITH_INV_SPEC, make_config(true, &["Inv"]));

    match result {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!("Expected invariant violation, got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_dependent_actions_no_reduction() {
    let stats = expect_success(run_check(DEPENDENT_ACTIONS_SPEC, make_config(true, &[])));

    assert_eq!(stats.por_reduction.action_count, 2);
    assert_eq!(stats.por_reduction.total_pairs, 1);
    assert_eq!(stats.por_reduction.independent_pairs, 0);
    assert!(stats.por_reduction.states_processed > 0);
    assert_eq!(stats.por_reduction.states_reduced, 0);
    assert_eq!(stats.por_reduction.actions_skipped, 0);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_three_disjoint_counters() {
    let without_por = expect_success(run_check(
        THREE_DISJOINT_COUNTERS_SPEC,
        make_config_no_por(&[]),
    ));
    assert_eq!(without_por.states_found, 27);

    let stats = expect_success(run_check(
        THREE_DISJOINT_COUNTERS_SPEC,
        make_config(true, &[]),
    ));

    assert_eq!(stats.states_found, 7);
    assert!(stats.states_found < without_por.states_found);
    assert_eq!(stats.por_reduction.action_count, 3);
    assert_eq!(stats.por_reduction.total_pairs, 3);
    assert_eq!(stats.por_reduction.independent_pairs, 3);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_statistics_populated() {
    let stats = expect_success(run_check(SIMPLE_POR_STATS_SPEC, make_config(true, &[])));
    let por = &stats.por_reduction;

    assert_eq!(por.action_count, 2);
    assert_eq!(por.total_pairs, 1);
    assert_eq!(por.independent_pairs, 1);
    assert!(por.states_processed > 0);
    assert!(por.states_reduced > 0);
    assert!(por.actions_skipped > 0);
    assert!(por.states_processed >= por.states_reduced);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_disabled_no_statistics() {
    let stats = expect_success(run_check(SIMPLE_POR_STATS_SPEC, make_config_no_por(&[])));
    let por = &stats.por_reduction;

    assert_eq!(por.action_count, 0);
    assert_eq!(por.total_pairs, 0);
    assert_eq!(por.independent_pairs, 0);
    assert_eq!(por.states_processed, 0);
    assert_eq!(por.states_reduced, 0);
    assert_eq!(por.actions_skipped, 0);
}

/// POR must be disabled when liveness properties are present.
/// The C3 BFS proviso is insufficient for liveness checking — liveness
/// requires the "ignoring proviso" (Peled 1996) or "strong proviso".
/// When POR is requested but liveness is present, the checker must fall
/// back to full exploration (no reduction, no POR stats).
const LIVENESS_SPEC: &str = r#"
---- MODULE PorLiveness ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ UNCHANGED y

IncY ==
    /\ y < 2
    /\ y' = y + 1
    /\ UNCHANGED x

Next == IncX \/ IncY
LiveProp == <>(x + y >= 0)
====
"#;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_disabled_with_liveness_properties() {
    let mut config = make_config(true, &[]);
    config.properties = vec!["LiveProp".to_string()];

    let stats = expect_success(run_check(LIVENESS_SPEC, config));
    let por = &stats.por_reduction;

    // POR should be fully disabled — no independence analysis, no reduction
    assert_eq!(por.action_count, 0);
    assert_eq!(por.independent_pairs, 0);
    assert_eq!(por.states_reduced, 0);
    assert_eq!(por.actions_skipped, 0);
    // Full state space should be explored (same as without POR)
    assert_eq!(stats.states_found, 9);
}

/// Identity assignment detection: `x' = x` should be treated as UNCHANGED x,
/// enabling independence when two actions only touch disjoint variables via
/// explicit identity writes rather than UNCHANGED keyword.
const IDENTITY_ASSIGNMENT_SPEC: &str = r#"
---- MODULE PorIdentityAssignment ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ y' = y

IncY ==
    /\ y < 2
    /\ y' = y + 1
    /\ x' = x

Next == IncX \/ IncY
====
"#;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_identity_assignment_enables_independence() {
    // `x' = x` is semantically identical to `UNCHANGED x` — POR should
    // detect this and treat the actions as independent.
    let stats = expect_success(run_check(IDENTITY_ASSIGNMENT_SPEC, make_config(true, &[])));
    let por = &stats.por_reduction;

    assert_eq!(por.action_count, 2);
    assert_eq!(por.total_pairs, 1);
    // The identity assignment detector recognizes `x' = x` → UNCHANGED x,
    // so the two actions should be independent.
    assert_eq!(por.independent_pairs, 1);
    // With independence, POR should reduce the state space
    assert!(por.states_reduced > 0);
    assert_eq!(stats.states_found, 5);
}

/// Visibility condition: when an invariant references a variable, any action
/// that writes to that variable is "visible" and must be included in the
/// ample set. This test verifies that POR correctly detects invariant
/// violations even when the violating action would otherwise be reduced.
const VISIBILITY_SPEC: &str = r#"
---- MODULE PorVisibility ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

IncX ==
    /\ x < 3
    /\ x' = x + 1
    /\ UNCHANGED y

IncY ==
    /\ y < 3
    /\ y' = y + 1
    /\ UNCHANGED x

Next == IncX \/ IncY
XBound == x <= 2
====
"#;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_visibility_preserves_invariant_checking() {
    // XBound references x, so IncX is visible. POR must not skip IncX
    // when it would violate the invariant.
    let result = run_check(VISIBILITY_SPEC, make_config(true, &["XBound"]));

    match result {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "XBound");
        }
        other => panic!("Expected invariant violation of XBound, got: {:?}", other),
    }
}

/// Safety net: POR must never change the set of reachable states for specs
/// where all actions are dependent (no reduction possible). The state count
/// with and without POR must be identical.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_same_state_count_as_without_por_when_dependent() {
    let without = expect_success(run_check(DEPENDENT_ACTIONS_SPEC, make_config_no_por(&[])));
    let with = expect_success(run_check(DEPENDENT_ACTIONS_SPEC, make_config(true, &[])));

    // When actions are dependent, POR cannot reduce — state counts must match exactly
    assert_eq!(with.states_found, without.states_found);
    assert_eq!(with.transitions, without.transitions);
}

// ==================== Auto-POR Tests ====================
//
// Part of #3993: Auto-POR enables partial order reduction automatically when
// the independence analysis detects independent action pairs. Users no longer
// need to pass --por explicitly.
//
// NOTE: Auto-POR is controlled by a OnceLock-cached env var (TY_AUTO_POR).
// The OnceLock reads the value once per process; env var tests that toggle it
// within a process are not feasible. Tests below rely on auto-POR being enabled
// by default (TY_AUTO_POR unset = true).

/// Auto-POR: disjoint counters spec gets reduced without --por.
///
/// This is the key auto-POR test: the spec has independent actions (IncX and
/// IncY touch disjoint variables), so auto-POR should detect this and enable
/// reduction automatically. The reduced state count should match explicit --por.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_auto_por_reduces_disjoint_counters() {
    // Auto-POR (default: por_enabled=false but auto-POR is on)
    let auto_por = expect_success(run_check(DISJOINT_COUNTERS_SPEC, make_config_auto_por(&[])));

    // Explicit --por
    let explicit_por = expect_success(run_check(DISJOINT_COUNTERS_SPEC, make_config(true, &[])));

    // Auto-POR should reduce to same state count as explicit --por
    assert_eq!(auto_por.states_found, explicit_por.states_found);
    assert_eq!(auto_por.states_found, 5);

    // Auto-POR stats should indicate auto-detection
    assert!(
        auto_por.por_reduction.auto_detected,
        "auto-POR should set auto_detected=true"
    );
    assert!(
        !explicit_por.por_reduction.auto_detected,
        "explicit --por should set auto_detected=false"
    );
    assert!(auto_por.por_reduction.independent_pairs > 0);
}

/// Auto-POR: dependent actions spec should NOT get POR overhead.
///
/// When all actions are dependent (no independent pairs), auto-POR should
/// not enable POR and the per-action enumeration path should be avoided.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_auto_por_skips_dependent_actions() {
    let stats = expect_success(run_check(DEPENDENT_ACTIONS_SPEC, make_config(false, &[])));

    // Auto-POR should detect no independent pairs and skip POR.
    // por_reduction should be empty (no POR active).
    assert_eq!(stats.por_reduction.action_count, 0);
    assert_eq!(stats.por_reduction.independent_pairs, 0);
    assert_eq!(stats.por_reduction.states_reduced, 0);
}

/// Auto-POR: three disjoint counters get full reduction.
///
/// With 3 mutually independent actions, auto-POR should achieve the same
/// reduction as explicit --por (3x state space reduction).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_auto_por_three_disjoint_counters() {
    let stats = expect_success(run_check(
        THREE_DISJOINT_COUNTERS_SPEC,
        make_config_auto_por(&[]),
    ));

    // Auto-POR should reduce to 7 states (same as explicit --por)
    assert_eq!(stats.states_found, 7);
    assert_eq!(stats.por_reduction.action_count, 3);
    assert_eq!(stats.por_reduction.independent_pairs, 3);
    assert!(stats.por_reduction.auto_detected);
}

/// Auto-POR preserves invariant violations.
///
/// Auto-POR must not suppress invariant detection. When IncX and IncY
/// are independent but the invariant x + y < 4 is violated by reaching
/// state (x=2, y=2), POR must still detect this.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_auto_por_preserves_invariant_detection() {
    let result = run_check(
        DISJOINT_COUNTERS_WITH_INV_SPEC,
        make_config(false, &["Inv"]),
    );

    match result {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!("Expected invariant violation, got: {:?}", other),
    }
}

/// Auto-POR: identity assignment detection works with auto-POR.
///
/// The spec uses explicit `x' = x` instead of `UNCHANGED x`, which should
/// still be detected as identity writes and enable independence.
///
/// NOTE: this test opts in EXPLICITLY (`auto_por: Some(true)`) to be immune to
/// the ambient `TY_AUTO_POR` env var; it exercises identity-write detection,
/// not the default-resolution path.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_auto_por_identity_assignment() {
    let cfg = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        por_enabled: false,
        auto_por: Some(true),
        check_deadlock: false,
        ..Default::default()
    };
    let stats = expect_success(run_check(IDENTITY_ASSIGNMENT_SPEC, cfg));

    // Auto-POR should detect identity assignments and reduce
    assert_eq!(stats.states_found, 5);
    assert_eq!(stats.por_reduction.independent_pairs, 1);
    assert!(stats.por_reduction.auto_detected);
    assert!(stats.por_reduction.states_reduced > 0);
}

const STUTTER_SELFLOOP_TWIN_SPEC: &str = r#"
---- MODULE PorStutterSelfloop ----
EXTENDS Integers

VARIABLE x

Init == x = 0

Next == (\E k \in 1..4 : x' = k) \/ (x' = x)

Inv == x <= 3
====
"#;

/// REGRESSION (2026-07-06 FALSE CLEAN): the invisible stutter disjunct formed a
/// singleton ample set, closed a SELF-LOOP in the reduced state graph, and the
/// visible writing action was ignored forever — 1 state explored, "No error has
/// been found", on a spec with a reachable violation (x = 4). Root cause: the
/// C3 cycle proviso is NOT "automatically satisfied by BFS" (that claim
/// confused exploration cycles with reduced-STATE-GRAPH cycles). This pins the
/// BEHAVIOR, not the mechanism: the DEFAULT configuration must find the
/// violation — originally guarded by the auto-POR containment default-off, and
/// NOW (default restored) guarded by the C3 BFS fresh-successor proviso: the
/// stutter-only ample expansion yields no fresh successor (its only successor
/// IS the expanding state), forcing full expansion, which fires the visible
/// writer immediately.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_stutter_selfloop_regression() {
    // (a) DEFAULT config (no explicit POR signals) must report the violation.
    let default_cfg = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        check_deadlock: false,
        ..Default::default()
    };
    match run_check(STUTTER_SELFLOOP_TWIN_SPEC, default_cfg) {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!(
            "DEFAULT check must find the x = 4 violation (the POR ignoring \
             false-clean), got: {other:?}"
        ),
    }
    // (b) Reductions explicitly off: the ground truth.
    match run_check(STUTTER_SELFLOOP_TWIN_SPEC, make_config_no_por(&["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!("no-POR ground truth must find the violation, got: {other:?}"),
    }
}

/// C3 proviso under EXPLICIT POR: the self-loop twin must report the violation
/// with `--por` (make_config(true, ..)) AND with the explicit auto-POR opt-in
/// (make_config_auto_por). Before the BFS fresh-successor proviso landed,
/// explicit POR still false-cleaned this spec (the containment only changed
/// the DEFAULT); the proviso is what makes reduced expansion sound here: the
/// stutter-only ample set's sole successor is the expanding state itself — no
/// fresh successor — so the state is fully expanded and the visible writer
/// (x' = 4) fires.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_stutter_selfloop_explicit_por_finds_violation() {
    // Explicit --por.
    match run_check(STUTTER_SELFLOOP_TWIN_SPEC, make_config(true, &["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!(
            "explicit --por must find the x = 4 violation via the C3 fresh-successor \
             proviso, got: {other:?}"
        ),
    }
    // Explicit auto-POR opt-in (config auto_por = Some(true)).
    match run_check(STUTTER_SELFLOOP_TWIN_SPEC, make_config_auto_por(&["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!(
            "explicit auto-POR must find the x = 4 violation via the C3 fresh-successor \
             proviso, got: {other:?}"
        ),
    }
}

/// INVISIBLE 2-CYCLE twin: unlike the self-loop twin (caught at the FIRST
/// state, whose ample successor equals the state itself), this spec only
/// trips the proviso at the SECOND cycle state. `Toggle` is invisible (real
/// write {y}; `Inv` reads only x) and independent of the visible `Write`, so
/// the ample set is {Toggle} at both s0=(0,0) and s1=(0,1):
///
///   s0 --Toggle--> s1   (s1 FRESH at s0's expansion => reduction stands,
///                        Write deferred)
///   s1 --Toggle--> s0   (s0 VISITED => NO fresh ample successor => the
///                        proviso forces FULL expansion => Write fires,
///                        x = 4 violates Inv)
///
/// Without the proviso, the s0 <-> s1 cycle would defer `Write` forever
/// (ignoring problem) and falsely report clean.
const INVISIBLE_TWO_CYCLE_TWIN_SPEC: &str = r#"
---- MODULE PorInvisibleTwoCycle ----
EXTENDS Integers

VARIABLE x, y

Init == x = 0 /\ y = 0

Toggle == y' = 1 - y /\ x' = x

Write == x' = 4 /\ y' = y

Next == Toggle \/ Write

Inv == x <= 3
====
"#;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_invisible_two_cycle_proviso_finds_violation() {
    // Ground truth (reductions off).
    match run_check(INVISIBLE_TWO_CYCLE_TWIN_SPEC, make_config_no_por(&["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!("no-POR ground truth must find the violation, got: {other:?}"),
    }
    // Explicit --por.
    match run_check(INVISIBLE_TWO_CYCLE_TWIN_SPEC, make_config(true, &["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!(
            "explicit --por must find the x = 4 violation (2-cycle ignoring), got: {other:?}"
        ),
    }
    // Explicit auto-POR opt-in.
    match run_check(
        INVISIBLE_TWO_CYCLE_TWIN_SPEC,
        make_config_auto_por(&["Inv"]),
    ) {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!(
            "explicit auto-POR must find the x = 4 violation (2-cycle ignoring), got: {other:?}"
        ),
    }
    // DEFAULT config (restored default-ON auto-POR resolves through the same
    // proviso-guarded ample path).
    let default_cfg = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        check_deadlock: false,
        ..Default::default()
    };
    match run_check(INVISIBLE_TWO_CYCLE_TWIN_SPEC, default_cfg) {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => {
            panic!("DEFAULT check must find the x = 4 violation (2-cycle ignoring), got: {other:?}")
        }
    }
}

const C1_DISABLED_DEPENDENT_TWIN_SPEC: &str = r#"
---- MODULE PorC1DisabledDependent ----
EXTENDS Integers

VARIABLES w, g, v, t

Init == w = 0 /\ g = 0 /\ v = 0 /\ t = 0

A == w = 0 /\ w' = 1 /\ UNCHANGED <<g, v, t>>

C == g = 0 /\ g' = 1 /\ UNCHANGED <<w, v, t>>

B == g = 1 /\ w = 0 /\ v' = 1 /\ UNCHANGED <<w, g, t>>

Tick == t' = 1 - t /\ UNCHANGED <<w, g, v>>

Next == A \/ C \/ B \/ Tick

Inv == v = 0
====
"#;

/// REGRESSION (2026-07-06 FALSE CLEAN #2, found by the adversarial verifier
/// AFTER the C3 fresh-successor proviso landed): ample condition C1 was closed
/// only over ENABLED dependents. At Init, ample {A} (A independent of the
/// enabled C and Tick) pruned the interleaving C;B — the DISABLED action B is
/// dependent on A (reads w, which A writes) and is awakened by the deferred C;
/// firing A first permanently disables B, and every reduced step makes
/// progress, so the C3 fresh-successor proviso never triggers. Fixed by the
/// conservative Valmari-style bail in `compute_ample_set` (any DISABLED action
/// dependent on a closure member rejects the seed); the surviving Tick-seed
/// reduction is then guarded by the C3 proviso on the toggle cycle. Every mode
/// must find the violation.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_c1_disabled_dependent_regression() {
    // Ground truth.
    match run_check(
        C1_DISABLED_DEPENDENT_TWIN_SPEC,
        make_config_no_por(&["Inv"]),
    ) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("no-POR ground truth must find the violation, got: {other:?}"),
    }
    // DEFAULT configuration.
    let default_cfg = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        check_deadlock: false,
        ..Default::default()
    };
    match run_check(C1_DISABLED_DEPENDENT_TWIN_SPEC, default_cfg) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("DEFAULT check must find the C1 violation, got: {other:?}"),
    }
    // Explicit --por and explicit auto-POR.
    match run_check(C1_DISABLED_DEPENDENT_TWIN_SPEC, make_config(true, &["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("explicit --por must find the C1 violation, got: {other:?}"),
    }
    match run_check(
        C1_DISABLED_DEPENDENT_TWIN_SPEC,
        make_config_auto_por(&["Inv"]),
    ) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("explicit auto-POR must find the C1 violation, got: {other:?}"),
    }
}

const HIDDEN_FUNCDEF_DEP_TWIN_SPEC: &str = r#"
---- MODULE PorHiddenFuncdefDep ----
EXTENDS Integers

VARIABLES w, g, v, t

Init == w = 0 /\ g = 0 /\ v = 0 /\ t = 0

WSel == [i \in {0} |-> w]

A == w = 0 /\ w' = 1 /\ UNCHANGED <<g, v, t>>

C == g = 0 /\ g' = 1 /\ UNCHANGED <<w, v, t>>

B == g = 1 /\ WSel[0] = 0 /\ v' = 1 /\ UNCHANGED <<w, g, t>>

Tick == t' = 1 - t /\ UNCHANGED <<w, g, v>>

Next == A \/ C \/ B \/ Tick

Inv == v = 0
====
"#;

/// REGRESSION (2026-07-06 FALSE CLEAN #3, found by the POR acceptance gate):
/// B's guard read of `w` is hidden behind the zero-arg FuncDef operator `WSel`,
/// which the expander declines to inline (the #2955 perf guard). The old POR
/// dependency extractor derived EMPTY deps from the un-inlined residue, the
/// (A, B) pair landed INDEPENDENT (off-diagonal Unknown never survives
/// `IndependenceMatrix::compute`), the C1 disabled-dependent bail was blind to
/// it, and ample {A} pruned the violating C;B interleaving — a false clean the
/// C3 proviso cannot see. FIXED (2026-07-07) by fail-closed extraction: the
/// `WSel` residue marks B OPAQUE (dependent on everything + visible), so the
/// C1 bail rejects every seed while B is disabled and C2 blocks every closure
/// once B is enabled — no interleaving is pruned. The auto-POR DEFAULT is
/// restored; this pin requires the violation in EVERY mode, with POR actually
/// ENGAGED (not silently skipped) under the explicit modes. The matrix-level
/// (A, B)-Dependent assertion lives in
/// `por::tests::test_hidden_funcdef_residue_marks_action_opaque_and_dependent`.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_hidden_funcdef_dep_regression() {
    // Ground truth (reductions off).
    match run_check(HIDDEN_FUNCDEF_DEP_TWIN_SPEC, make_config_no_por(&["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("no-POR ground truth must find the violation, got: {other:?}"),
    }
    // DEFAULT configuration (auto-POR on by default again).
    let default_cfg = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        check_deadlock: false,
        ..Default::default()
    };
    match run_check(HIDDEN_FUNCDEF_DEP_TWIN_SPEC, default_cfg) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("DEFAULT check must find the hidden-funcdef violation, got: {other:?}"),
    }
    // Explicit --por: POR engages (the spec has 3 analyzable independent
    // pairs among A/C/Tick, so the explicit path routes through the ample
    // machinery), and the opaque marking keeps B un-pruned: every seed is
    // rejected by the C1 disabled-dependent bail while B is disabled and by
    // C2 visibility once B is enabled — the violation must be found.
    // (Violation results snapshot stats BEFORE finalize populates
    // por_reduction, so engagement is pinned at the matrix level by the
    // por::tests unit test above, not via stats here.)
    match run_check(HIDDEN_FUNCDEF_DEP_TWIN_SPEC, make_config(true, &["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("explicit --por must find the violation, got: {other:?}"),
    }
    // Explicit auto-POR opt-in.
    match run_check(HIDDEN_FUNCDEF_DEP_TWIN_SPEC, make_config_auto_por(&["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("explicit auto-POR must find the violation, got: {other:?}"),
    }
}

const HIDDEN_CAPTURE_DEP_TWIN_SPEC: &str = r#"
---- MODULE PorHiddenCaptureDep ----
EXTENDS Integers

VARIABLES w, g, v, t

Init == w = 0 /\ g = 0 /\ v = 0 /\ t = 0

GuardW(a) == \E c \in {0} : w = 0 /\ a = c

A == w = 0 /\ w' = 1 /\ UNCHANGED <<g, v, t>>

C == g = 0 /\ g' = 1 /\ UNCHANGED <<w, v, t>>

B == g = 1 /\ (\E c \in {0} : GuardW(c)) /\ v' = 1 /\ UNCHANGED <<w, g, t>>

Tick == t' = 1 - t /\ UNCHANGED <<w, g, v>>

Next == A \/ C \/ B \/ Tick

Inv == v = 0
====
"#;

/// REGRESSION (2026-07-06 FALSE CLEAN #3, second vector): B's guard read of `w`
/// is hidden behind a CAPTURE-UNSAFE operator application, which the expander
/// keeps un-inlined — same empty-deps extraction, same unsound INDEPENDENT
/// verdict, same pruned interleaving. FIXED (2026-07-07) by the same
/// fail-closed opaque marking (`GuardW(c)` residue ⇒ B opaque); the matrix
/// pin lives in
/// `por::tests::test_hidden_capture_residue_marks_action_opaque_and_dependent`.
/// Behavior-level pin: every mode must find the violation with the restored
/// auto-POR default.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_hidden_capture_dep_regression() {
    match run_check(HIDDEN_CAPTURE_DEP_TWIN_SPEC, make_config_no_por(&["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("no-POR ground truth must find the violation, got: {other:?}"),
    }
    let default_cfg = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        check_deadlock: false,
        ..Default::default()
    };
    match run_check(HIDDEN_CAPTURE_DEP_TWIN_SPEC, default_cfg) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("DEFAULT check must find the hidden-capture violation, got: {other:?}"),
    }
    // Explicit --por and explicit auto-POR: POR engages, opaque B blocks
    // every prune, violation found.
    match run_check(HIDDEN_CAPTURE_DEP_TWIN_SPEC, make_config(true, &["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("explicit --por must find the violation, got: {other:?}"),
    }
    match run_check(HIDDEN_CAPTURE_DEP_TWIN_SPEC, make_config_auto_por(&["Inv"])) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("explicit auto-POR must find the violation, got: {other:?}"),
    }
}

/// REGRESSION (audit-2026-07 #11): coverage/POR action-decomposition mismatch.
/// `A` is a NAMED operator whose body directly contains primes plus a
/// TOP-LEVEL disjunction: the enumeration/coverage decomposition (non-primed
/// expansion) keeps `A` un-split (2 coverage actions), while the old POR
/// analysis re-detected actions on the with-primes expansion of the whole
/// `Next`, splitting `A` into its two disjuncts (3 actions).
/// `compute_ample_set` then fed coverage-space indices into the finer matrix
/// and read C1 dependencies / C2 visibility off the WRONG rows. A stopgap
/// skipped POR on the length mismatch; the real fix extracts one UNIONED
/// dependency set per coverage action, so the matrix is coverage-indexed by
/// construction (2x2 here — pinned in
/// `por::tests::test_coverage_indexed_matrix_unions_internal_disjuncts`) and
/// POR stays ON. The violation (z = 1 via B) must be found in EVERY mode.
const ACTION_DECOMPOSITION_MISMATCH_SPEC: &str = r#"
---- MODULE PorActionDecompositionMismatch ----
EXTENDS Integers

VARIABLE x, w, z

Init == x = 0 /\ w = 0 /\ z = 0

A == (x' = 1 /\ w' = w /\ z' = z) \/ (w' = 1 /\ x' = x /\ z' = z)

B == z' = 1 /\ x' = x /\ w' = w

Next == A \/ B

Inv == z # 1
====
"#;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_action_decomposition_mismatch_regression() {
    // Ground truth (reductions off).
    match run_check(
        ACTION_DECOMPOSITION_MISMATCH_SPEC,
        make_config_no_por(&["Inv"]),
    ) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("no-POR ground truth must find the violation, got: {other:?}"),
    }
    // DEFAULT configuration.
    let default_cfg = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        check_deadlock: false,
        ..Default::default()
    };
    match run_check(ACTION_DECOMPOSITION_MISMATCH_SPEC, default_cfg) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!(
            "DEFAULT check must find the z = 1 violation with the coverage-indexed \
             matrix, got: {other:?}"
        ),
    }
    // Explicit --por: POR engages on the 2x2 coverage-indexed matrix (the
    // A/B pair is independent, so ample sets DO form) and the C2 visibility
    // of B (writes z; Inv reads z) plus the C3 fresh-successor proviso keep
    // the violation reachable.
    match run_check(
        ACTION_DECOMPOSITION_MISMATCH_SPEC,
        make_config(true, &["Inv"]),
    ) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("explicit --por must find the violation, got: {other:?}"),
    }
    // Explicit auto-POR opt-in.
    match run_check(
        ACTION_DECOMPOSITION_MISMATCH_SPEC,
        make_config_auto_por(&["Inv"]),
    ) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, "Inv"),
        other => panic!("explicit auto-POR must find the violation, got: {other:?}"),
    }
}

/// Safe direction of audit-2026-07 #11: a named action with an INTERNAL
/// disjunction that is genuinely independent of its sibling must still get
/// POR reduction — the stopgap skipped POR entirely on the decomposition
/// length mismatch (2 coverage actions vs 3 with-primes actions), reporting
/// action_count == 0 and exploring the full product. With the coverage-indexed
/// unioned extraction, the matrix is 2x2, the pair is independent, and the
/// state space shrinks.
const INTERNAL_DISJUNCT_INDEPENDENT_SPEC: &str = r#"
---- MODULE PorInternalDisjunctIndependent ----
EXTENDS Integers

VARIABLE x, w

Init == x = 0 /\ w = 0

A == (x = 0 /\ x' = 1 /\ w' = w) \/ (x = 1 /\ x' = 2 /\ w' = w)

B == w = 0 /\ w' = 1 /\ x' = x

Next == A \/ B
====
"#;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_reduces_named_action_with_internal_disjuncts() {
    let without_por = expect_success(run_check(
        INTERNAL_DISJUNCT_INDEPENDENT_SPEC,
        make_config_no_por(&[]),
    ));
    assert_eq!(without_por.states_found, 6);

    let with_por = expect_success(run_check(
        INTERNAL_DISJUNCT_INDEPENDENT_SPEC,
        make_config(true, &[]),
    ));

    // Matrix is COVERAGE-indexed: 2 actions (A un-split), 1 independent pair.
    // Under the stopgap this was action_count == 0 (POR silently skipped).
    assert_eq!(with_por.por_reduction.action_count, 2);
    assert_eq!(with_por.por_reduction.total_pairs, 1);
    assert_eq!(with_por.por_reduction.independent_pairs, 1);

    // POR actually engaged and reduced: ample {A} defers B along the x-chain
    // (0,0) -> (1,0) -> (2,0), where A disables and B fires once.
    assert_eq!(with_por.states_found, 4);
    assert!(with_por.states_found < without_por.states_found);
    assert!(with_por.por_reduction.states_reduced > 0);
    assert!(with_por.por_reduction.actions_skipped > 0);
}

/// Trigger spec for the POR x trace-invariant soundness gap (audit finding:
/// C2 visibility is built ONLY from state invariants, so ample-set reduction
/// can prune the one history that violates a history-dependent trace
/// invariant). Full BFS reaches (x=2, y=0) via IncX;IncX and the trace
/// invariant TInv rejects any history containing that state; the ample set
/// at (0,0) may lawfully be the C2-invisible singleton {IncY} (TInv is not
/// consulted by visibility), after which (2,0) is never generated -> false
/// PASS.
const TRACE_INV_DISJOINT_COUNTERS_SPEC: &str = r#"
---- MODULE PorTraceInvCounters ----
EXTENDS Naturals

VARIABLE x, y

Init == x = 0 /\ y = 0

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ UNCHANGED y

IncY ==
    /\ y < 2
    /\ y' = y + 1
    /\ UNCHANGED x

Next == IncX \/ IncY
Inv == x >= 0
TInv(hist) == \A i \in DOMAIN hist : ~(hist[i].x = 2 /\ hist[i].y = 0)
====
"#;

fn expect_tinv_violation(result: CheckResult) {
    match result {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "TInv", "expected the TRACE invariant to fire");
        }
        other => panic!(
            "POR pruned the history that violates the trace invariant \
             (soundness regression): expected InvariantViolation for TInv, got: {:?}",
            other
        ),
    }
}

/// Pinned soundness regression: auto-POR must NOT mask a trace-invariant
/// violation. POR is disabled whenever `config.trace_invariants` is non-empty
/// (the C2 visibility set cannot see history-dependent properties), so the
/// violating history through (2,0) must still be found.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_trace_invariant_violation_not_masked_by_auto_por() {
    let mut config = make_config_auto_por(&["Inv"]);
    config.trace_invariants = vec!["TInv".to_string()];
    expect_tinv_violation(run_check(TRACE_INV_DISJOINT_COUNTERS_SPEC, config));
}

/// Same soundness gate for EXPLICIT `--por`: a user-requested reduction is
/// still unsound under trace invariants, so it must be disabled too.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_trace_invariant_violation_not_masked_by_explicit_por() {
    let mut config = make_config(true, &["Inv"]);
    config.trace_invariants = vec!["TInv".to_string()];
    expect_tinv_violation(run_check(TRACE_INV_DISJOINT_COUNTERS_SPEC, config));
}

/// Sanity baseline: with POR fully disabled the trace-invariant violation is
/// found (pins that the trigger spec actually violates TInv, so the two tests
/// above are testing POR gating and not the spec).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn por_trace_invariant_violation_found_without_por_baseline() {
    let mut config = make_config_no_por(&["Inv"]);
    config.trace_invariants = vec!["TInv".to_string()];
    expect_tinv_violation(run_check(TRACE_INV_DISJOINT_COUNTERS_SPEC, config));
}
