// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for basic state bookkeeping: counts, mark-seen, trace degradation.

use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_states_count_initially_zero() {
    let module = parse_module(
        r#"
---- MODULE Count0 ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mc = ModelChecker::new(&module, &config);
    assert_eq!(mc.states_count(), 0);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_mark_trace_degraded_sets_flag_once() {
    let module = parse_module(
        r#"
---- MODULE TraceDegradedFlag ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    assert!(!mc.trace.trace_degraded);

    let first = std::io::Error::other("synthetic trace write error");
    mc.mark_trace_degraded(&first);
    assert!(mc.trace.trace_degraded);

    // Second error should be a no-op for the flag (warning already emitted once).
    let second = std::io::Error::other("synthetic trace write error 2");
    mc.mark_trace_degraded(&second);
    assert!(mc.trace.trace_degraded);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_is_state_seen_false_for_unseen() {
    let module = parse_module(
        r#"
---- MODULE Unseen ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mc = ModelChecker::new(&module, &config);
    assert!(!mc.is_state_seen_checked(Fingerprint(12345)).unwrap());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_mark_state_seen_fp_only_then_is_seen() {
    let module = parse_module(
        r#"
---- MODULE MarkFp ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    let fp = Fingerprint(99999);
    assert!(!mc.is_state_seen_checked(fp).unwrap());

    mc.mark_state_seen_fp_only_checked(fp, None, 0).unwrap();

    assert!(mc.is_state_seen_checked(fp).unwrap());
    assert_eq!(mc.states_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_mark_multiple_states_counted() {
    let module = parse_module(
        r#"
---- MODULE MarkMulti ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.mark_state_seen_fp_only_checked(Fingerprint(1), None, 0)
        .unwrap();
    mc.mark_state_seen_fp_only_checked(Fingerprint(2), Some(Fingerprint(1)), 1)
        .unwrap();
    mc.mark_state_seen_fp_only_checked(Fingerprint(3), Some(Fingerprint(2)), 2)
        .unwrap();
    assert_eq!(mc.states_count(), 3);
}

fn assert_collision_rejected(result: CheckResult, expected_policy: &str) {
    assert_collision_rejected_with_details(result, expected_policy, &[]);
}

fn assert_collision_rejected_with_details(
    result: CheckResult,
    expected_policy: &str,
    expected_details: &[&str],
) {
    match result {
        CheckResult::Error { error, .. } => {
            let rendered = error.to_string();
            assert!(
                rendered.contains("prepared_fingerprint_admission")
                    && rendered.contains("reason_code=canonical_payload_mismatch"),
                "unexpected error: {rendered}"
            );
            assert!(
                rendered.contains(expected_policy),
                "unexpected error: {rendered}"
            );
            for expected_detail in expected_details {
                assert!(
                    rendered.contains(expected_detail),
                    "expected error to contain {expected_detail:?}, got: {rendered}"
                );
            }
        }
        other => panic!("expected CheckResult::Error, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_borrowed_state_admission_rejects_same_fingerprint_different_payload() {
    let module = parse_module(
        r#"
---- MODULE BorrowedCollisionDedup ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(55);
    let original = ArrayState::from_values(vec![Value::int(1)]);
    assert!(mc.mark_state_seen_checked(fp, &original, None, 0).unwrap());

    let err = mc
        .mark_state_seen_checked(fp, &ArrayState::from_values(vec![Value::int(2)]), None, 1)
        .expect_err("borrowed-state duplicate with different payload must fail closed");

    assert_collision_rejected(err, "collision_policy=canonical_payload_equality");
    assert_eq!(mc.states_count(), 1);
    assert_eq!(
        mc.state_storage
            .seen
            .get(&fp)
            .expect("original state must remain resident")
            .values(),
        original.values()
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_borrowed_state_admission_allows_payload_confirmed_duplicate() {
    let module = parse_module(
        r#"
---- MODULE BorrowedConfirmedDuplicate ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(56);
    let original = ArrayState::from_values(vec![Value::int(1)]);
    assert!(mc.mark_state_seen_checked(fp, &original, None, 0).unwrap());
    assert!(!mc.mark_state_seen_checked(fp, &original, None, 1).unwrap());
    assert_eq!(mc.states_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_fp_only_admission_rejects_duplicate_without_payload_proof() {
    let module = parse_module(
        r#"
---- MODULE FpOnlyDuplicateNoPayload ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(57);
    assert!(mc.mark_state_seen_fp_only_checked(fp, None, 0).unwrap());
    let err = mc
        .mark_state_seen_fp_only_checked(fp, None, 1)
        .expect_err("fp-only duplicate without payload proof must fail closed");

    assert_collision_rejected_with_details(
        err,
        "collision_policy=canonical_payload_equality",
        &[
            "payload_witness=tla_array_fp64",
            "dedup_identity=dedup:state_space:external:fingerprint:canonical_payload_equality:fingerprint_tla-explicit-state_canonical_domain_tla-array-state_v1_tla_fingerprint64_state_array-state-v1",
        ],
    );
    assert_eq!(mc.states_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_compiled_flat_fp_only_admission_rejects_duplicate_without_payload_proof() {
    let module = parse_module(
        r#"
---- MODULE CompiledFlatDuplicateNoPayload ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.flat_state_primary = true;

    let fp = Fingerprint(59);
    assert!(mc.mark_state_seen_fp_only_checked(fp, None, 0).unwrap());
    let err = mc
        .mark_state_seen_fp_only_checked(fp, None, 1)
        .expect_err("compiled-flat fp-only duplicate without payload proof must fail closed");

    assert_collision_rejected_with_details(
        err,
        "collision_policy=canonical_payload_equality",
        &[
            "payload_witness=compiled_flat_xxh3",
            "dedup_identity=dedup:state_space:external:fingerprint:canonical_payload_equality:fingerprint_tla-compiled-flat-state_canonical_domain_tla-flat-i64-state_v1_xxh3_u64_state_flat-i64-state-v1",
        ],
    );
    assert_eq!(mc.states_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_fp_only_admission_allows_current_payload_confirmed_duplicate() {
    let module = parse_module(
        r#"
---- MODULE FpOnlyCurrentDuplicate ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(58);
    assert!(mc.mark_state_seen_fp_only_checked(fp, None, 0).unwrap());
    assert!(!mc
        .mark_state_seen_fp_only_with_duplicate_payload_checked(fp, Some(fp), 1, true)
        .unwrap());
    assert_eq!(mc.states_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_full_state_owned_admission_rejects_unchecked_collision_policy_before_insert() {
    let module = parse_module(
        r#"
---- MODULE UnsafeDedupPolicy ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(42);
    let err = mc
        .mark_state_seen_owned_checked_with_collision_policy_for_test(
            fp,
            ArrayState::from_values(vec![Value::int(1)]),
            tla_mc_core::SharedCollisionPolicy::Unchecked,
        )
        .expect_err("unchecked collision policy must fail before admission");

    match err {
        CheckResult::Error { error, .. } => {
            let rendered = error.to_string();
            assert!(
                rendered.contains("non_fail_closed_collision_policy"),
                "unexpected error: {rendered}"
            );
        }
        other => panic!("expected CheckResult::Error, got {other:?}"),
    }
    assert_eq!(mc.states_count(), 0);
    assert!(!mc.state_storage.seen.contains_key(&fp));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_full_state_owned_admission_rejects_same_fingerprint_different_payload() {
    let module = parse_module(
        r#"
---- MODULE CollisionDedup ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(99);
    let original = ArrayState::from_values(vec![Value::int(1)]);
    assert!(mc
        .mark_state_seen_owned_checked(fp, original.clone(), None, 0)
        .unwrap());

    let err = mc
        .mark_state_seen_owned_checked(fp, ArrayState::from_values(vec![Value::int(2)]), None, 1)
        .expect_err("same fingerprint with different payload must fail closed");

    assert_collision_rejected_with_details(
        err,
        "collision_policy=canonical_payload_equality",
        &[
            "payload_witness=tla_array_fp64",
            "dedup_identity=dedup:state_space:external:explicit_state:canonical_payload_equality:fingerprint_tla-explicit-state_canonical_domain_tla-array-state_v1_tla_fingerprint64_state_array-state-v1",
        ],
    );
    assert_eq!(mc.states_count(), 1);
    assert_eq!(
        mc.state_storage
            .seen
            .get(&fp)
            .expect("original state must remain resident")
            .values(),
        original.values()
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_view_full_state_admission_allows_same_view_duplicate_payload() {
    let module = parse_module(
        r#"
---- MODULE ViewDuplicatePayload ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
View == 0
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        view: Some("View".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);
    mc.compiled.cached_view_name = Some("View".to_string());

    let fp = Fingerprint(88);
    assert!(mc
        .mark_state_seen_checked(fp, &ArrayState::from_values(vec![Value::int(1)]), None, 0)
        .unwrap());
    assert!(!mc
        .mark_state_seen_checked(fp, &ArrayState::from_values(vec![Value::int(2)]), None, 1)
        .unwrap());
    assert_eq!(mc.states_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_view_full_state_admission_rejects_same_fingerprint_different_view() {
    let module = parse_module(
        r#"
---- MODULE ViewCollisionDedup ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
View == x
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        view: Some("View".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);
    mc.compiled.cached_view_name = Some("View".to_string());

    let fp = Fingerprint(89);
    assert!(mc
        .mark_state_seen_checked(fp, &ArrayState::from_values(vec![Value::int(1)]), None, 0)
        .unwrap());
    let err = mc
        .mark_state_seen_checked(fp, &ArrayState::from_values(vec![Value::int(2)]), None, 1)
        .expect_err("VIEW duplicate with different canonical view must fail closed");

    assert_collision_rejected_with_details(
        err,
        "collision_policy=canonical_payload_equality",
        &[
            "payload_witness=tla_array_fp64",
            "dedup_identity=dedup:state_space:external:explicit_state:canonical_payload_equality:fingerprint_tla-view-state_canonical_domain_tla-view-value_v1_tla_fingerprint64_state_tla-view-value-v1",
        ],
    );
    assert_eq!(mc.states_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symmetry_full_state_admission_rejects_uncanonical_collision() {
    let module = parse_module(
        r#"
---- MODULE SymmetryCollisionDedup ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);
    mc.symmetry
        .perms
        .push(crate::value::FuncValue::from_sorted_entries(Vec::<(
            Value,
            Value,
        )>::new(
        )));

    let fp = Fingerprint(90);
    assert!(mc
        .mark_state_seen_checked(fp, &ArrayState::from_values(vec![Value::int(1)]), None, 0)
        .unwrap());
    let err = mc
        .mark_state_seen_checked(fp, &ArrayState::from_values(vec![Value::int(2)]), None, 1)
        .expect_err("symmetry-domain duplicate with different canonical payload must fail closed");

    assert_collision_rejected_with_details(
        err,
        "collision_policy=canonical_payload_equality",
        &[
            "payload_witness=tla_array_fp64",
            "dedup_identity=dedup:state_space:external:explicit_state:canonical_payload_equality:fingerprint_tla-symmetry-canonical-state_canonical_domain_tla-symmetry-canonical-array-state_v1_tla_fingerprint64_state_array-state-v1",
        ],
    );
    assert_eq!(mc.states_count(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_full_state_owned_admission_allows_payload_confirmed_duplicate() {
    let module = parse_module(
        r#"
---- MODULE ConfirmedDuplicate ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(7);
    assert!(mc
        .mark_state_seen_owned_checked(fp, ArrayState::from_values(vec![Value::int(1)]), None, 0,)
        .unwrap());
    assert!(!mc
        .mark_state_seen_owned_checked(fp, ArrayState::from_values(vec![Value::int(1)]), None, 1,)
        .unwrap());
    assert_eq!(mc.states_count(), 1);
}
