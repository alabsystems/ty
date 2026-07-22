// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for automatic symmetry detection from model value sets.
//!
//! IMPORTANT: these tests use `ModelChecker::set_auto_symmetry(..)` rather
//! than mutating the `TY_AUTO_SYMMETRY` environment variable. `std::env::set_var`
//! is process-global: toggling the var here used to enable auto-symmetry in
//! concurrently-running tests, silently collapsing their no-symmetry state
//! counts (the long-standing symmetry test flake).

use super::*;

/// Verify that auto-detection produces the same state count as explicit SYMMETRY.
///
/// This test runs twice: once with explicit SYMMETRY config, once with
/// auto-symmetry enabled (per-checker override) and no SYMMETRY config.
/// Both should produce the same canonical state count.
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_matches_explicit_symmetry() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE AutoDetectTest ----
EXTENDS TLC
CONSTANT Procs
VARIABLE active

Init == active \in Procs
Next == active' \in Procs /\ active' /= active

Sym == Permutations(Procs)
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    // Config WITH explicit symmetry.
    let mut config_explicit = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        symmetry: Some("Sym".to_string()),
        ..Default::default()
    };
    config_explicit.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec![
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ]),
    );

    let mut checker_explicit = ModelChecker::new(&module, &config_explicit);
    checker_explicit.set_deadlock_check(false);
    let result_explicit = checker_explicit.check();

    let states_explicit = match result_explicit {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("Expected Success with explicit symmetry, got {:?}", other),
    };

    // Config WITHOUT explicit symmetry, but with auto-symmetry enabled.
    let config_auto = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constants: config_explicit.constants.clone(),
        constants_order: config_explicit.constants_order.clone(),
        // No symmetry field!
        ..Default::default()
    };

    let mut checker_auto = ModelChecker::new(&module, &config_auto);
    checker_auto.set_auto_symmetry(true);
    checker_auto.set_deadlock_check(false);
    let result_auto = checker_auto.check();

    let (states_auto, sym_stats) = match result_auto {
        CheckResult::Success(stats) => {
            let sym = stats.symmetry_reduction.clone();
            (stats.states_found, sym)
        }
        other => panic!("Expected Success with auto symmetry, got {:?}", other),
    };

    // Both should produce the same state count.
    assert_eq!(
        states_explicit, states_auto,
        "auto-detected symmetry should produce same state count as explicit: explicit={}, auto={}",
        states_explicit, states_auto
    );

    // Auto-detected flag should be set.
    assert!(
        sym_stats.auto_detected,
        "symmetry should be marked as auto-detected"
    );

    // Permutation count: S3 has 6 group elements; the iteration list is
    // normalized to the 5 non-identity elements (the identity contributes
    // nothing to orbit minimization).
    assert_eq!(
        sym_stats.permutation_count, 5,
        "auto-detected S3 should iterate 5 non-identity permutations"
    );
}

/// Verify auto-detection works with multiple independent symmetric sets.
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_multi_group() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE AutoDetectMulti ----
EXTENDS TLC
CONSTANTS Acceptors, Values
VARIABLE votes

Init == votes \in [Acceptors -> Values \cup {"none"}]
Next == UNCHANGED votes

Sym == Permutations(Acceptors) \cup Permutations(Values)
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut config_explicit = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        symmetry: Some("Sym".to_string()),
        ..Default::default()
    };
    config_explicit.constants.insert(
        "Acceptors".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec![
            "a1".to_string(),
            "a2".to_string(),
            "a3".to_string(),
        ]),
    );
    config_explicit.constants.insert(
        "Values".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec!["v1".to_string(), "v2".to_string()]),
    );

    let mut checker_explicit = ModelChecker::new(&module, &config_explicit);
    checker_explicit.set_deadlock_check(false);
    let result_explicit = checker_explicit.check();

    let states_explicit = match result_explicit {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("Expected Success with explicit symmetry, got {:?}", other),
    };

    // Config with auto-detection.
    let config_auto = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        constants: config_explicit.constants.clone(),
        constants_order: config_explicit.constants_order.clone(),
        ..Default::default()
    };

    let mut checker_auto = ModelChecker::new(&module, &config_auto);
    checker_auto.set_auto_symmetry(true);
    checker_auto.set_deadlock_check(false);
    let result_auto = checker_auto.check();

    let (states_auto, sym_stats) = match result_auto {
        CheckResult::Success(stats) => {
            let sym = stats.symmetry_reduction.clone();
            (stats.states_found, sym)
        }
        other => panic!("Expected Success with auto symmetry, got {:?}", other),
    };

    assert_eq!(
        states_explicit, states_auto,
        "multi-group auto-detect should match explicit: explicit={}, auto={}",
        states_explicit, states_auto
    );

    // Should detect 2 groups.
    assert_eq!(
        sym_stats.symmetry_groups, 2,
        "should auto-detect 2 independent symmetry groups"
    );

    assert!(sym_stats.auto_detected);
}

/// Verify auto-detection does not activate when explicitly disabled via the
/// per-checker override (and, by extension, by default when TY_AUTO_SYMMETRY
/// is not set in the environment — see the
/// `auto_symmetry_enabled_from_value` unit tests for env parsing).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_auto_detect_disabled_by_default() {
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE AutoDetectDisabled ----
EXTENDS TLC
CONSTANT Procs
VARIABLE active

Init == active \in Procs
Next == active' \in Procs /\ active' /= active
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
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
    // Explicit per-checker override: disabled. This also keeps the test
    // hermetic if the ambient environment has TY_AUTO_SYMMETRY set.
    checker.set_auto_symmetry(false);
    checker.set_deadlock_check(false);
    let result = checker.check();

    let stats = match result {
        CheckResult::Success(stats) => stats,
        other => panic!("Expected Success, got {:?}", other),
    };

    // Without auto-detection or explicit symmetry, all 3 states should be found.
    assert_eq!(
        stats.states_found, 3,
        "without auto-detect, should find all 3 states"
    );

    // Symmetry stats should be empty.
    assert_eq!(stats.symmetry_reduction.permutation_count, 0);
    assert!(!stats.symmetry_reduction.auto_detected);
}

/// Shared module source for the `=`-form engagement and guard tests.
fn lower_module(src: &str) -> tla_core::ast::Module {
    use tla_core::{lower, parse_to_syntax_tree, FileId};
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    lower_result.module.expect("module should lower")
}

/// `Name = {m1, m2}`-form config sets must engage auto-symmetry exactly like
/// `Name <- {m1, m2}` model value sets (the MCKVS/btree/Disruptor cfg shape).
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_eq_form_matches_explicit_symmetry() {
    let module = lower_module(
        r#"
---- MODULE AutoDetectEqForm ----
EXTENDS TLC
CONSTANT Procs
VARIABLE active

Init == active \in Procs
Next == active' \in Procs /\ active' /= active

Sym == Permutations(Procs)
====
"#,
    );

    // Explicit declared symmetry: the parity oracle.
    let mut config_explicit = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        symmetry: Some("Sym".to_string()),
        ..Default::default()
    };
    config_explicit.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec![
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ]),
    );
    let mut checker_explicit = ModelChecker::new(&module, &config_explicit);
    checker_explicit.set_deadlock_check(false);
    let states_explicit = match checker_explicit.check() {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("Expected Success with explicit symmetry, got {:?}", other),
    };

    // `=`-form: Procs = {p1, p2, p3} parses as ConstantValue::Value.
    let mut config_auto = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config_auto.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::Value("{p1, p2, p3}".to_string()),
    );
    let mut checker_auto = ModelChecker::new(&module, &config_auto);
    checker_auto.set_auto_symmetry(true);
    checker_auto.set_deadlock_check(false);
    let (states_auto, sym_stats) = match checker_auto.check() {
        CheckResult::Success(stats) => {
            let sym = stats.symmetry_reduction.clone();
            (stats.states_found, sym)
        }
        other => panic!(
            "Expected Success with =-form auto symmetry, got {:?}",
            other
        ),
    };

    assert_eq!(
        states_explicit, states_auto,
        "=-form auto symmetry should match explicit SYMMETRY state count"
    );
    assert!(sym_stats.auto_detected, "symmetry should be auto-detected");
    // S3 normalized to its 5 non-identity elements.
    assert_eq!(
        sym_stats.permutation_count, 5,
        "S3 iterates 5 non-identity permutations"
    );
}

/// Guard (a): a constant binding that pins a member of the candidate set
/// (the SpanTreeRandom `Root = n1` shape) must prevent the FULL per-set group.
/// The phase-2 stabilizer construction then soundly engages the residual
/// subgroup that fixes the pinned member: with `Root = p1`, the states
/// reached via `p2` and `p3` are genuinely interchangeable (`(p2 p3)` fixes
/// every constant binding), so the orbit count is 2, not 3.
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_guard_a_pinned_member_disables() {
    let module = lower_module(
        r#"
---- MODULE AutoDetectPinned ----
CONSTANTS Procs, Root
VARIABLE active

Init == active = Root
Next == active' \in Procs
====
"#,
    );

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::Value("{p1, p2, p3}".to_string()),
    );
    config.constants.insert(
        "Root".to_string(),
        crate::config::ConstantValue::Value("p1".to_string()),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_auto_symmetry(true);
    checker.set_deadlock_check(false);
    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("Expected Success, got {:?}", other),
    };

    // Root pins p1, so the full S3 must NOT engage (that would fold p1 with
    // p2/p3 — unsound). The stabilizer subgroup {id, (p2 p3)} fixes Root and
    // every other binding, so it soundly folds the symmetric p2/p3 states.
    assert_eq!(
        stats.states_found, 2,
        "pinned member: stabilizer folds the symmetric non-pinned states"
    );
    assert_eq!(
        stats.symmetry_reduction.permutation_count, 1,
        "stabilizer of a pinned member is {{id, (p2 p3)}} → 1 non-identity perm"
    );
    assert!(stats.symmetry_reduction.auto_detected);
}

/// Guard (b): a bounded CHOOSE over the candidate set
/// (the MCNano `CHOOSE hash \in Hash` / Slush `CHOOSE n \in Node` shape)
/// must prevent the FULL per-set group. Because `Pick` here is a
/// constant-level zero-arity operator, it is PRECOMPUTED to a concrete value
/// (`p1`), and the phase-2 stabilizer verifies invariance of that VALUE:
/// permutations moving `p1` are rejected, and the residual `{id, (p2 p3)}`
/// subgroup soundly engages (re-evaluating a constant-level CHOOSE over the
/// unchanged constant environment is deterministic).
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_guard_b_bounded_choose_disables() {
    let module = lower_module(
        r#"
---- MODULE AutoDetectChoose ----
CONSTANT Procs
VARIABLE active

Pick == CHOOSE p \in Procs : TRUE

Init == active = Pick
Next == active' \in Procs
====
"#,
    );

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::Value("{p1, p2, p3}".to_string()),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_auto_symmetry(true);
    checker.set_deadlock_check(false);
    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("Expected Success, got {:?}", other),
    };

    // The CHOOSE pins p1 at the VALUE level (Pick precomputes to p1); the
    // stabilizer of the constant environment is {id, (p2 p3)}, folding the
    // genuinely symmetric p2/p3 states.
    assert_eq!(
        stats.states_found, 2,
        "bounded CHOOSE: stabilizer folds the non-chosen symmetric states"
    );
    assert_eq!(
        stats.symmetry_reduction.permutation_count, 1,
        "stabilizer of the precomputed CHOOSE value is {{id, (p2 p3)}}"
    );
}

/// The fresh-witness pattern `CHOOSE x : x \notin S` (KeyValueStore's NoVal,
/// btree's NIL/MISSING) is sound and must NOT prevent engagement.
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_unbounded_notin_choose_still_engages() {
    let module = lower_module(
        r#"
---- MODULE AutoDetectNotin ----
CONSTANT Procs
VARIABLE active

NoP == CHOOSE p : p \notin Procs

Init == active = NoP
Next == active' \in Procs \cup {NoP}
====
"#,
    );

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::Value("{p1, p2, p3}".to_string()),
    );
    // Production pattern (MCKVS `NoVal = NoVal`, btree `NIL = nil`): the config
    // constant overrides the spec's unbounded-CHOOSE definition with a model
    // value. The CHOOSE body remains in the op_defs that guard (b) scans.
    config.constants.insert(
        "NoP".to_string(),
        crate::config::ConstantValue::Value("NoP".to_string()),
    );

    // Unreduced baseline.
    let mut checker_off = ModelChecker::new(&module, &config);
    checker_off.set_auto_symmetry(false);
    checker_off.set_deadlock_check(false);
    let states_off = match checker_off.check() {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("Expected Success, got {:?}", other),
    };
    assert_eq!(states_off, 4, "unreduced: NoP + p1 + p2 + p3");

    // Auto-symmetry: the three Procs states collapse into one orbit.
    let mut checker_on = ModelChecker::new(&module, &config);
    checker_on.set_auto_symmetry(true);
    checker_on.set_deadlock_check(false);
    let stats_on = match checker_on.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("Expected Success, got {:?}", other),
    };
    assert!(
        stats_on.symmetry_reduction.auto_detected,
        "unbounded \\notin CHOOSE must not prevent engagement"
    );
    assert_eq!(stats_on.states_found, 2, "reduced: NoP orbit + Procs orbit");
}

/// Guard (c): genuine temporal properties hard-disable auto-symmetry
/// (the AllocatorImplementation / Disruptor_SPMC shape).
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_guard_c_temporal_property_disables() {
    let module = lower_module(
        r#"
---- MODULE AutoDetectTemporal ----
CONSTANT Procs
VARIABLE active

Init == active \in Procs
Next == active' \in Procs /\ active' /= active

Live == []<>(active \in Procs)
====
"#,
    );

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Live".to_string()],
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::Value("{p1, p2, p3}".to_string()),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_auto_symmetry(true);
    checker.set_deadlock_check(false);
    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("Expected Success, got {:?}", other),
    };

    assert_eq!(
        stats.symmetry_reduction.permutation_count, 0,
        "genuine temporal properties must hard-disable auto-symmetry"
    );
    assert_eq!(stats.states_found, 3, "full state count under liveness");
}

/// Canonicalization must not hide violations: an invariant that fails in
/// every orbit member must still be detected on the reduced state graph.
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_violation_still_detected() {
    let module = lower_module(
        r#"
---- MODULE AutoDetectViolation ----
CONSTANT Procs
VARIABLES active, count

Init == active \in Procs /\ count = 0
Next == count' = count + 1 /\ UNCHANGED active

Inv == count < 2
====
"#,
    );

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::Value("{p1, p2, p3}".to_string()),
    );

    // Sanity: the same spec/config DOES engage auto-symmetry (verified on the
    // success variant — symmetry stats are only finalized when BFS completes).
    let mut config_ok = config.clone();
    config_ok.invariants = vec!["InvOk".to_string()];
    let module_ok = lower_module(
        r#"
---- MODULE AutoDetectViolationOk ----
CONSTANT Procs
VARIABLES active, count

Init == active \in Procs /\ count = 0
Next == count < 2 /\ count' = count + 1 /\ UNCHANGED active

InvOk == count <= 2
====
"#,
    );
    let mut checker_ok = ModelChecker::new(&module_ok, &config_ok);
    checker_ok.set_auto_symmetry(true);
    checker_ok.set_deadlock_check(false);
    match checker_ok.check() {
        CheckResult::Success(stats) => {
            assert!(
                stats.symmetry_reduction.auto_detected,
                "auto-symmetry must engage on this spec/config shape"
            );
            // 3 init states fold into 1 orbit; count walks 0..=2 → 3 states.
            assert_eq!(stats.states_found, 3, "orbit-reduced state count");
        }
        other => panic!("Expected Success on the bounded variant, got {:?}", other),
    }

    // The violating variant: canonicalization must not hide the violation.
    let mut checker = ModelChecker::new(&module, &config);
    checker.set_auto_symmetry(true);
    checker.set_deadlock_check(false);
    match checker.check() {
        CheckResult::InvariantViolation {
            invariant, trace, ..
        } => {
            assert_eq!(invariant, "Inv");
            // The trace must be a concrete behavior reaching the violation:
            // count = 0 -> 1 -> 2.
            assert_eq!(
                trace.states.len(),
                3,
                "violation trace must be reconstructible under auto-symmetry"
            );
        }
        other => panic!(
            "Expected InvariantViolation under auto-symmetry, got {:?}",
            other
        ),
    }
}

/// Phase 2 (correlated-constant stabilizer subgroup), the SlushProtocol shape:
/// three interchangeable model value sets whose rows are correlated by a
/// `HostMapping`-style constant. Per-set groups are individually unsound
/// (a transposition of `a1`/`a2` alone maps `Pair` to a different value), so
/// phase 1 admits nothing — but the DIAGONAL subgroup acting consistently on
/// both sets fixes `Pair` setwise and is a true automorphism group.
///
/// State space: x : A -> {0,1}, reachable = all 8 assignments. Orbits under
/// the diagonal S3 (acting on A) are "number of 1s" classes: 4 canonical
/// states. The `HostOf`-style bounded CHOOSE inside a precomputed constant
/// operator exercises the relaxed guard (b).
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_correlated_diagonal_subgroup() {
    let module = lower_module(
        r#"
---- MODULE AutoDetectDiagonal ----
CONSTANTS A, B, Pair
VARIABLE x

HostOf == [b \in B |-> CHOOSE a \in A : \E p \in Pair : a \in p /\ b \in p]

Init == x = [a \in A |-> 0]
Next == \E a \in A : x' = [x EXCEPT ![a] = 1 - x[a]]
====
"#,
    );

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "A".to_string(),
        crate::config::ConstantValue::Value("{a1, a2, a3}".to_string()),
    );
    config.constants.insert(
        "B".to_string(),
        crate::config::ConstantValue::Value("{b1, b2, b3}".to_string()),
    );
    config.constants.insert(
        "Pair".to_string(),
        crate::config::ConstantValue::Value("{{a1, b1}, {a2, b2}, {a3, b3}}".to_string()),
    );

    // Unreduced oracle.
    let mut checker_off = ModelChecker::new(&module, &config);
    checker_off.set_auto_symmetry(false);
    checker_off.set_deadlock_check(false);
    match checker_off.check() {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 8, "unreduced: all 0/1 assignments");
        }
        other => panic!("Expected Success without symmetry, got {:?}", other),
    }

    // With auto-symmetry: the diagonal stabilizer subgroup must engage.
    let mut checker = ModelChecker::new(&module, &config);
    checker.set_auto_symmetry(true);
    checker.set_deadlock_check(false);
    match checker.check() {
        CheckResult::Success(stats) => {
            let sym = &stats.symmetry_reduction;
            assert!(sym.auto_detected, "diagonal subgroup must auto-engage");
            // Diagonal S3 has 6 elements; 5 non-identity perms iterated.
            assert_eq!(
                sym.permutation_count, 5,
                "diagonal S3 stabilizer iterates 5 non-identity permutations"
            );
            assert_eq!(
                stats.states_found, 4,
                "orbits of x : A -> {{0,1}} under diagonal S3 are the 4 popcount classes"
            );
        }
        other => panic!("Expected Success with stabilizer subgroup, got {:?}", other),
    }
}

/// Phase 2 must NOT engage when the correlating constant is asymmetric in a
/// way that kills every non-identity product element (here `Pair` correlates
/// `a1` with BOTH b's rows, pinning everything).
#[cfg_attr(test, ntest::timeout(15000))]
#[test]
fn test_auto_detect_correlated_asymmetric_no_engage() {
    let module = lower_module(
        r#"
---- MODULE AutoDetectDiagonalAsym ----
CONSTANTS A, B, Pair
VARIABLE x

Init == x = [a \in A |-> 0]
Next == \E a \in A : x' = [x EXCEPT ![a] = 1 - x[a]]
====
"#,
    );

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.constants.insert(
        "A".to_string(),
        crate::config::ConstantValue::Value("{a1, a2}".to_string()),
    );
    config.constants.insert(
        "B".to_string(),
        crate::config::ConstantValue::Value("{b1, b2}".to_string()),
    );
    // a1 pairs with both b1 and b2, a2 only with b1: the only product element
    // fixing Pair is the identity (swapping a's breaks {a2,b1} vs {a1,b1};
    // swapping b's breaks {a2,b1} vs {a2,b2}; swapping both maps {a2,b1} to
    // {a1,b2} which is not a member).
    config.constants.insert(
        "Pair".to_string(),
        crate::config::ConstantValue::Value("{{a1, b1}, {a1, b2}, {a2, b1}}".to_string()),
    );

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_auto_symmetry(true);
    checker.set_deadlock_check(false);
    match checker.check() {
        CheckResult::Success(stats) => {
            assert!(
                !stats.symmetry_reduction.auto_detected,
                "trivial stabilizer must not engage"
            );
            assert_eq!(stats.states_found, 4, "full unreduced state count");
        }
        other => panic!("Expected Success, got {:?}", other),
    }
}
