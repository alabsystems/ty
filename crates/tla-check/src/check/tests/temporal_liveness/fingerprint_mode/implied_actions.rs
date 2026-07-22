// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

fn assert_implied_action_property_stats_attached(result: &CheckResult) {
    let stats = result.stats();
    assert!(
        stats.property_check.implied_action_transition_checks > 0,
        "expected promoted implied-action transition telemetry on {result:?}",
    );
    // NOTE: the unchanged-trigger skip optimization (property_classify
    // `truth_if_unchanged`) can prove an implied-action term true with zero
    // term evals while still recording a transition check, so the former
    // `term_evals >= transition_checks` lower bound no longer holds.

    let output = crate::JsonOutput::new(
        std::path::Path::new("ImpliedActionTelemetry.tla"),
        None,
        "ImpliedActionTelemetry",
        1,
    )
    .with_check_result(result, std::time::Duration::from_secs(0));
    let json_stats = output
        .statistics
        .property_check
        .expect("JSON statistics should include property_check telemetry");
    assert_eq!(
        json_stats.implied_action_transition_checks,
        stats.property_check.implied_action_transition_checks,
    );
    assert_eq!(
        json_stats.implied_action_term_evals,
        stats.property_check.implied_action_term_evals,
    );
}

/// Part of #2670: `[][Bad]_vars` violations still surface in fingerprint-only mode.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_implied_action_violation_detected_in_fingerprint_only_mode() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    // Next increments x, but the property requires [][UNCHANGED x]_vars.
    // The transition x=0 -> x=1 violates the implied action.
    let src = r#"
---- MODULE ImpliedActionFpOnly ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0

Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x

Bad == UNCHANGED x
SpecProp == [][Bad]_vars
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        properties: vec!["SpecProp".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    // Do NOT call set_store_states(true) — stay in fingerprint-only mode (default).
    // Before #2670, this would return Success. After #2670, InvariantViolation.

    let result = checker.check();
    assert_implied_action_property_stats_attached(&result);
    match result {
        // Part of #2834: implied action violations from PROPERTIES are correctly
        // reported as PropertyViolation.
        CheckResult::PropertyViolation {
            property,
            trace: _,
            stats: _,
            kind: _,
        } => {
            assert_eq!(property, "SpecProp");
        }
        CheckResult::Success(_) => {
            panic!(
                "Implied action violation should be detected in fingerprint-only mode. \
                 Got Success, meaning the [][Bad]_vars check was silently skipped."
            );
        }
        other => panic!(
            "Expected PropertyViolation for implied action, got: {:?}",
            other
        ),
    }
}

/// A tuple `UNCHANGED` truth trigger is sound only when every tuple variable is
/// unchanged. Keeping one variable unchanged must not skip a violating action.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_tuple_unchanged_trigger_requires_all_vars() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE ImpliedActionTupleTrigger ----
EXTENDS Integers

VARIABLE x, y

Init == x = 0 /\ y = 0

Next == x' = x /\ y' = y + 1

SpecProp == [][FALSE]_<<x, y>>
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        properties: vec!["SpecProp".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    match result {
        CheckResult::PropertyViolation { property, .. } => {
            assert_eq!(property, "SpecProp");
        }
        other => panic!("Expected tuple-trigger PropertyViolation, got: {other:?}"),
    }
}

/// Implied-action evaluation errors leave the serial BFS loop through
/// `bfs_error_return`, not the ordinary terminal finalizer.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_implied_action_error_reports_property_telemetry() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE ImpliedActionErrorTelemetry ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0

Next == IF x < 1 THEN x' = x + 1 ELSE UNCHANGED x

Bad == x' = x + (1 \div 0)
SpecProp == [][Bad]_vars
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        properties: vec!["SpecProp".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    assert_implied_action_property_stats_attached(&result);
    match result {
        CheckResult::Error {
            error: crate::CheckError::Eval(crate::EvalCheckError::Eval(_)),
            ..
        } => {}
        other => panic!("Expected implied-action eval error, got: {other:?}"),
    }
}

/// Part of #2670: implied actions are checked even on transitions to seen states.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_implied_action_checked_for_seen_state_transitions() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    // x toggles between 0 and 1. The property forbids x changing (UNCHANGED x).
    // The first transition (x=0->x=1) is to a new state and should be caught.
    let src = r#"
---- MODULE ImpliedActionSeen ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x \in {0, 1}

Next == x' = 1 - x

Bad == UNCHANGED x
NeverChange == [][Bad]_vars
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        properties: vec!["NeverChange".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    match result {
        // Part of #2834: implied action violations from PROPERTIES are correctly
        // reported as PropertyViolation.
        CheckResult::PropertyViolation { property, .. } => {
            assert_eq!(property, "NeverChange");
        }
        other => panic!("Expected PropertyViolation, got: {:?}", other),
    }
}

/// Part of #2670: fully promoted action properties are omitted from the warning.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_fingerprint_warning_excludes_promoted_action_properties() {
    let config = Config {
        properties: vec!["ActionProp".to_string(), "TemporalProp".to_string()],
        ..Default::default()
    };

    // ActionProp is promoted (appears in promoted_names).
    // TemporalProp is not promoted (remains in warning).
    let promoted = vec!["ActionProp".to_string()];
    let warning = config
        .fingerprint_liveness_warning(false, &promoted)
        .expect("warning should exist for non-promoted temporal property");
    assert!(
        !warning.contains("ActionProp"),
        "Warning should not mention promoted action property: {warning}"
    );
    assert!(
        warning.contains("TemporalProp"),
        "Warning should mention non-promoted temporal property: {warning}"
    );
}

/// Part of #2670: No warning when all properties are fully promoted.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_fingerprint_warning_absent_when_all_promoted() {
    let config = Config {
        properties: vec!["ActionProp".to_string()],
        ..Default::default()
    };

    let promoted = vec!["ActionProp".to_string()];
    assert!(
        config
            .fingerprint_liveness_warning(false, &promoted)
            .is_none(),
        "No warning when all properties are fully promoted to BFS-phase checking"
    );
}

/// Part of #2670, Step 7 test case 4: mixed properties still BFS-check their action term.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_mixed_property_splitting_action_part_checked_in_bfs() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    // Next increments x, but the property says [][UNCHANGED x]_vars (Bad).
    // The property ALSO has WF_vars(Next) — a liveness/fairness constraint.
    // The action violation should be caught during BFS despite the liveness parts.
    let src = r#"
---- MODULE MixedPropertySplit ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0

Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x

Bad == UNCHANGED x

\* Mixed property: init + implied action + fairness
\* - Init is a state predicate (checked on initial states)
\* - [][Bad]_vars is an action property (checked during BFS on ALL transitions)
\* - WF_vars(Next) is a fairness constraint (requires liveness/SCC analysis)
MixedSpec == Init /\ [][Bad]_vars /\ WF_vars(Next)
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        properties: vec!["MixedSpec".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    // Fingerprint-only mode (default): the implied action part ([][Bad]_vars)
    // must still be checked during BFS before the liveness phase matters.

    let result = checker.check();
    match result {
        // Part of #2834: implied action violations from PROPERTIES are correctly
        // reported as PropertyViolation.
        CheckResult::PropertyViolation {
            property,
            trace: _,
            stats: _,
            kind: _,
        } => {
            assert_eq!(property, "MixedSpec");
        }
        CheckResult::Success(_) => {
            panic!(
                "Mixed property implied action violation should be detected during BFS. \
                 Got Success — the [][Bad]_vars part was not extracted for BFS checking \
                 when mixed with WF_vars(Next)."
            );
        }
        other => panic!(
            "Expected PropertyViolation for mixed property implied action, got: {:?}",
            other
        ),
    }
}

/// Part of #2670, Step 7 test case 4b: a satisfied mixed action term still succeeds.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_mixed_property_splitting_action_part_satisfied() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    // Next is the same action used in [][Next]_vars, so the implied action is satisfied.
    // The property also has WF_vars(Next), which is satisfied on this path.
    let src = r#"
---- MODULE MixedPropertySatSplit ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0

Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x

\* Mixed property where the action part matches Next (always satisfied)
MixedSpec == Init /\ [][Next]_vars /\ WF_vars(Next)
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        properties: vec!["MixedSpec".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    // Disable stuttering so the only behavior is 0→1→2→2→... which satisfies
    // WF_vars(Next). With stuttering_allowed=true (default), infinite stuttering
    // at x=0 or x=1 would violate WF_vars(Next) since <<Next>>_vars is enabled
    // but never taken on a pure stutter cycle.
    checker.set_stuttering_allowed(false);

    let result = checker.check();
    assert_implied_action_property_stats_attached(&result);
    match result {
        CheckResult::Success(stats) => {
            // x takes values 0, 1, 2
            assert_eq!(stats.states_found, 3);
        }
        other => panic!(
            "Expected Success for satisfied mixed property in fp-only mode, \
             got: {:?}",
            other
        ),
    }
}

/// Fingerprint-keyed transition cache: an action property whose `_v` subscript
/// and body are built from zero-arg DERIVED operators (a CHOOSE-based one,
/// mirroring EWD998PCal's `token`, and an arithmetic fold mirroring `pending`)
/// must produce the exact state count and verdict, and must actually populate
/// the implied-action transition cache (the derived values are memoized per
/// state fingerprint instead of re-evaluated per transition).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_implied_action_derived_op_transition_cache_success() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE ImpliedFpCacheDerived ----
EXTENDS Integers

VARIABLE f
vars == <<f>>

Init == f = [k \in 0..2 |-> 0]

Next == \E k \in 0..2 : f' = [f EXCEPT ![k] = (f[k] + 1) % 3]

\* Zero-arg derived state operators. `maxk` uses CHOOSE (kept off the
\* bytecode path, mirroring EWD998PCal's `token`); `total` mirrors `pending`.
maxk == CHOOSE k \in 0..2 : \A j \in 0..2 : f[j] <= f[k]
total == f[0] + f[1] + f[2]

\* Every step bumps one coordinate by +1 (mod 3), so total changes by +1 or
\* -2: the action body holds for every transition.
SpecProp == [][total' - total <= 1]_<<total, maxk>>
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        properties: vec!["SpecProp".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    assert_implied_action_property_stats_attached(&result);
    match result {
        CheckResult::Success(stats) => {
            // All 27 functions 0..2 -> 0..2 are reachable via single-coordinate
            // (+1 mod 3) increments from the all-zero function.
            assert_eq!(stats.states_found, 27);
            assert!(
                stats.property_check.implied_action_term_evals > 0,
                "derived-op action property must be term-evaluated"
            );
        }
        other => panic!(
            "Expected Success for derived-op implied action, got: {:?}",
            other
        ),
    }
    // The derived operators are state functions with pure single-sided state
    // deps, so the fingerprint-keyed transition cache must have been
    // populated during the run (sequential BFS runs on this thread).
    assert!(
        tla_eval::implied_transition_cache_len() > 0,
        "implied-action transition cache should hold derived-op entries"
    );
}

/// Fingerprint-keyed transition cache must never mask a violation: the same
/// derived-operator shape with a violated action body must still report a
/// PropertyViolation (in-process analog of the EWD998PCal known-violation
/// probe).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_implied_action_derived_op_violation_still_detected() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE ImpliedFpCacheDerivedViolation ----
EXTENDS Integers

VARIABLE f
vars == <<f>>

Init == f = [k \in 0..2 |-> 0]

Next == \E k \in 0..2 : f' = [f EXCEPT ![k] = (f[k] + 1) % 3]

maxk == CHOOSE k \in 0..2 : \A j \in 0..2 : f[j] <= f[k]
total == f[0] + f[1] + f[2]

\* Violated by every +1 step out of the initial state.
SpecProp == [][total' <= total]_<<total, maxk>>
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        properties: vec!["SpecProp".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);

    let result = checker.check();
    assert_implied_action_property_stats_attached(&result);
    match result {
        CheckResult::PropertyViolation { property, .. } => {
            assert_eq!(property, "SpecProp");
        }
        other => panic!(
            "Expected PropertyViolation for violated derived-op implied action, \
             got: {:?}",
            other
        ),
    }
}
