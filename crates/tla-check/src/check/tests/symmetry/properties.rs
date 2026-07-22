// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symmetry_large_permutation_group() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    // Test with 4 elements to verify scaling: 4! = 24 permutations
    let src = r#"
---- MODULE SymmetryLarge ----
EXTENDS TLC
CONSTANT Procs
VARIABLE leader

\* Select a leader from processes
Init == leader \in Procs
Next == leader' \in Procs /\ leader' /= leader

\* Symmetry
Sym == Permutations(Procs)
===="#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    // Config WITHOUT symmetry
    let mut config_no_sym = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config_no_sym.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec![
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
            "p4".to_string(),
        ]),
    );

    // Config WITH symmetry
    let mut config_sym = config_no_sym.clone();
    config_sym.symmetry = Some("Sym".to_string());

    // Check WITHOUT symmetry
    let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
    // Hermetic baseline: auto-symmetry is ON by default; this checker measures
    // the UNREDUCED state count, so disable it explicitly.
    checker_no_sym.set_auto_symmetry(false);
    checker_no_sym.set_deadlock_check(false);
    let result_no_sym = checker_no_sym.check();

    let states_no_sym = match result_no_sym {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("Expected Success without symmetry, got {:?}", other),
    };

    // Check WITH symmetry
    let mut checker_sym = ModelChecker::new(&module, &config_sym);
    checker_sym.set_deadlock_check(false);
    let result_sym = checker_sym.check();

    let states_sym = match result_sym {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("Expected Success with symmetry, got {:?}", other),
    };

    // Without symmetry: 4 states (leader = p1, p2, p3, or p4)
    // With symmetry: 1 canonical state
    assert_eq!(
        states_no_sym, 4,
        "Without symmetry should have 4 states (4 processes)"
    );
    assert_eq!(states_sym, 1, "With symmetry should have 1 canonical state");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symmetry_accepts_filtered_permutation_set() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE SymmetryFiltered ----
EXTENDS TLC
CONSTANT Procs
VARIABLE leader

Init == leader \in Procs
Next == leader' \in Procs /\ leader' /= leader

\* Part of #1918: filtered permutation sets stay lazy (SetPred) until iterated.
Sym == {p \in Permutations(Procs) : p = p}
===="#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        symmetry: Some("Sym".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec![
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ]),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    let states = match result {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!(
            "filtered permutation symmetry should succeed, got {:?}",
            other
        ),
    };

    assert_eq!(
        states, 1,
        "filtered permutation symmetry should still reduce to one canonical state"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symmetry_preserves_invariant_violation() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    // Ensure symmetry reduction still catches invariant violations
    let src = r#"
---- MODULE SymmetryInvariant ----
EXTENDS TLC
CONSTANT Procs
VARIABLE active, count

Init == active \in Procs /\ count = 0
Next == /\ active' \in Procs
    /\ count' = count + 1

\* Invariant: count < 3 (will be violated)
Safety == count < 3

\* Symmetry
Sym == Permutations(Procs)
===="#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    // Config with invariant
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );
    config.symmetry = Some("Sym".to_string());

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    let result = checker.check();

    // Should find invariant violation even with symmetry
    match result {
        CheckResult::InvariantViolation { .. } => {
            // Expected - symmetry doesn't hide violations
        }
        other => panic!("Expected InvariantViolation with symmetry, got {:?}", other),
    }
}

/// Part of #1963/#2227, updated for the declared-SYMMETRY wrong-verdict fix:
/// when both SYMMETRY and genuine temporal PROPERTIES are configured, the
/// checker now IGNORES the declared symmetry (prints a warning and installs no
/// permutations) instead of warn-and-continue with the unsound orbit quotient.
/// This test verifies the end-to-end code path completes without panic/error.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symmetry_with_liveness_property_emits_warning_and_completes() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    // Simple spec with a stuttering-allowed liveness property.
    // <>[] (x = 2) is eventually always satisfied once x reaches 2.
    let src = r#"
---- MODULE SymmetryLiveness ----
EXTENDS TLC, Integers
CONSTANT Procs
VARIABLE x

Init == x \in {0}
Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x

\* A liveness property: eventually x reaches 2 and stays there
Liveness == <>[](x = 2)

\* Symmetry (does not interact with x, but triggers the warning)
Sym == Permutations(Procs)
===="#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Liveness".to_string()],
        symmetry: Some("Sym".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_store_states(true); // required for liveness checking

    let result = checker.check();
    // The checker should complete (success or liveness result) — not panic
    match result {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 3, "x takes values 0, 1, 2");
        }
        CheckResult::LivenessViolation { .. } => {
            // Also acceptable — the spec may or may not satisfy liveness
            // depending on the liveness algorithm's handling of stuttering.
        }
        other => panic!(
            "Expected Success or LivenessViolation with symmetry+liveness, got {:?}",
            other
        ),
    }
}

/// Was Part of #3222 (SYMMETRY + liveness auto-upgraded to full-state mode).
/// Declared-SYMMETRY wrong-verdict fix: symmetry is now IGNORED for genuine
/// temporal properties, so no full-state upgrade happens — the run stays in
/// fp-only/no-trace mode (inline liveness recording works without symmetry)
/// and must still complete with the correct unreduced verdict.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symmetry_liveness_notrace_stays_fp_only_with_symmetry_ignored() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE SymLivenessNoTrace ----
EXTENDS TLC, Integers
CONSTANT Procs
VARIABLE x

Init == x \in {0}
Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x

Liveness == <>[](x = 2)
Sym == Permutations(Procs)
===="#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Liveness".to_string()],
        symmetry: Some("Sym".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_auto_create_trace_file(false);
    // Do NOT call set_store_states(true). Start from no-trace mode: with the
    // declared symmetry ignored for liveness, no full-state upgrade is needed.

    let result = checker.check();
    assert!(
        !checker.test_store_full_states(),
        "symmetry+liveness must NOT auto-upgrade to full-state storage anymore: declared \
         SYMMETRY is ignored for genuine temporal properties, and fp-only liveness needs \
         no full-state upgrade"
    );

    match result {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 3, "x takes values 0, 1, 2");
        }
        CheckResult::LivenessViolation { stats, .. } => {
            assert_eq!(stats.states_found, 3, "x takes values 0, 1, 2");
        }
        other => panic!(
            "Expected Success or LivenessViolation with symmetry ignored for liveness, got {:?}",
            other
        ),
    }
}

/// Part of #2227: Verify that pure safety properties (`[]P`) are NOT rejected
/// in SYMMETRY + notrace mode. The safety-temporal fast path handles these
/// correctly without needing full-state storage.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symmetry_safety_property_notrace_accepted() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    // Spec with a pure safety property expressed via PROPERTY keyword.
    // `[]P` where P is state-level — handled by safety-temporal fast path.
    let src = r#"
---- MODULE SymSafetyNoTrace ----
EXTENDS TLC, Integers
CONSTANT Procs
VARIABLE x

Init == x \in {0}
Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x

\* Pure safety property: x is always in range
Safety == [](x >= 0 /\ x <= 2)

\* Symmetry set
Sym == Permutations(Procs)
===="#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Safety".to_string()],
        symmetry: Some("Sym".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    // Do NOT call set_store_states(true) — leave in default notrace mode.
    // With #2227 fix, this should succeed (not error with ConfigError).

    let result = checker.check();
    match result {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 3, "x takes values 0, 1, 2");
        }
        CheckResult::Error { error, .. } => {
            let msg = format!("{error}");
            panic!("Pure safety property with symmetry+notrace should NOT be rejected, got: {msg}");
        }
        other => panic!(
            "Expected Success for symmetry+safety_property+notrace, got {:?}",
            other
        ),
    }
}

/// Spec where the orbit quotient actually matters: `leader` ranges over the
/// symmetric set, so declared SYMMETRY reduces 3 states to 1 canonical state.
const SYM_LEADER_SRC: &str = r#"
---- MODULE SymLeader ----
EXTENDS TLC
CONSTANT Procs
VARIABLE leader

Init == leader \in Procs
Next == leader' \in Procs /\ leader' /= leader

\* Genuine temporal property (requires the liveness checker; holds trivially).
Liveness == <>(leader \in Procs)

\* Pure safety property (safety-temporal fast path; keeps symmetry).
Safety == [](leader \in Procs)

Sym == Permutations(Procs)
===="#;

fn sym_leader_config(property: &str, with_symmetry: bool) -> Config {
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec![property.to_string()],
        ..Default::default()
    };
    if with_symmetry {
        config.symmetry = Some("Sym".to_string());
    }
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec![
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ]),
    );
    config
}

/// Declared-SYMMETRY wrong-verdict fix: with a genuine liveness PROPERTY the
/// checker must IGNORE the declared symmetry — the run explores the full
/// unreduced state space (3 states, not the 1-state orbit quotient) and
/// returns the same verdict as the unreduced run.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_declared_symmetry_ignored_for_genuine_liveness_matches_unreduced_run() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let tree = parse_to_syntax_tree(SYM_LEADER_SRC);
    let module = lower(FileId(0), &tree).module.unwrap();

    // Unreduced baseline: no SYMMETRY, auto-symmetry explicitly off.
    let config_baseline = sym_leader_config("Liveness", false);
    let mut checker_baseline = ModelChecker::new(&module, &config_baseline);
    checker_baseline.set_auto_symmetry(false);
    checker_baseline.set_deadlock_check(false);
    let baseline_states = match checker_baseline.check() {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("unreduced liveness baseline should hold, got {other:?}"),
    };
    assert_eq!(baseline_states, 3, "leader takes 3 values unreduced");

    // Declared SYMMETRY + genuine liveness: symmetry must be ignored.
    let config_sym = sym_leader_config("Liveness", true);
    let mut checker_sym = ModelChecker::new(&module, &config_sym);
    checker_sym.set_deadlock_check(false);
    match checker_sym.check() {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, baseline_states,
                "declared SYMMETRY must be ignored under genuine liveness: expected the \
                 unreduced state count (orbit quotient would give 1)"
            );
        }
        other => panic!(
            "declared SYMMETRY + liveness must return the unreduced verdict (HOLD), got {other:?}"
        ),
    }
}

/// Companion boundary test: a PURE-SAFETY `[]P` PROPERTY keeps declared
/// symmetry exactly as before — the orbit quotient still engages (1 canonical
/// state vs 3 unreduced).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_declared_symmetry_pure_safety_property_still_reduces() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let tree = parse_to_syntax_tree(SYM_LEADER_SRC);
    let module = lower(FileId(0), &tree).module.unwrap();

    let config_baseline = sym_leader_config("Safety", false);
    let mut checker_baseline = ModelChecker::new(&module, &config_baseline);
    checker_baseline.set_auto_symmetry(false);
    checker_baseline.set_deadlock_check(false);
    let baseline_states = match checker_baseline.check() {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("unreduced safety baseline should hold, got {other:?}"),
    };
    assert_eq!(baseline_states, 3, "leader takes 3 values unreduced");

    let config_sym = sym_leader_config("Safety", true);
    let mut checker_sym = ModelChecker::new(&module, &config_sym);
    checker_sym.set_deadlock_check(false);
    match checker_sym.check() {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 1,
                "pure-safety PROPERTY must keep declared symmetry (orbit quotient reduces \
                 3 states to 1 canonical state)"
            );
        }
        other => panic!("declared SYMMETRY + pure-safety PROPERTY should hold, got {other:?}"),
    }
}
