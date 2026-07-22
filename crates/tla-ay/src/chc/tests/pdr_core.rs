// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::helpers::*;
use super::*;
use tla_core::ast::Expr;
use tla_core::Spanned;

use crate::TlaSort;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_simple_counter_safe() {
    // Simple counter: count starts at 0, increments, stays <= 5
    // Init: count = 0
    // Next: count' = count + 1 /\ count < 5
    // Safety: count <= 5
    //
    // This is SAFE because count can only reach 5 (then transition is disabled)

    let mut trans = ChcTranslator::new(&[("count", TlaSort::Int)]).unwrap();

    // Init: count = 0
    let init = eq_expr(var_expr("count"), int_expr(0));
    trans.add_init(&init).unwrap();

    // Next: count < 5 /\ count' = count + 1
    let guard = lt_expr(var_expr("count"), int_expr(5));
    let update = eq_expr(
        prime_expr("count"),
        add_expr(var_expr("count"), int_expr(1)),
    );
    let next = and_expr(guard, update);
    trans.add_next(&next).unwrap();

    // Safety: count <= 5
    let safety = le_expr(var_expr("count"), int_expr(5));
    trans.add_safety(&safety).unwrap();

    // Solve
    let result = trans.solve_pdr(pdr_test_config()).unwrap();

    // Should be Safe
    match result {
        PdrCheckResult::Safe { .. } => {
            // Expected
        }
        PdrCheckResult::Unknown { .. } => {
            // Acceptable for skeleton PDR
        }
        PdrCheckResult::Unsafe { .. } => {
            panic!("Expected Safe or Unknown, got Unsafe for safe spec");
        }
    }
}

#[test]
fn test_chc_proof_replay_boundary_missing_evidence_fails_closed() {
    let evidence = render_chc_proof_replay_boundary_evidence("TLA", None);

    assert!(evidence.contains("TLA ay_chc_proof_replay_boundary"));
    assert!(evidence.contains("status=Unavailable"));
    assert!(evidence.contains("status_code=missing_typed_chc_proof_transcript"));
    assert!(evidence.contains("typed_consumer=false"));
    assert!(evidence.contains("expected_schema=ay.chc-proof-transcript/v1"));
    assert!(evidence.contains("expected_schema_version=1"));
    assert!(evidence.contains("expected_normalized_input_schema=ay.chc.normalized-input/v1"));
    assert!(evidence.contains("upstream_api=ay_chc::engines::solve_pdr_proof"));
    assert!(evidence.contains("production_selected=false"));
    assert!(evidence.contains("fail_closed=true"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_chc_proof_replay_boundary_uses_ay_typed_consumer_evidence() {
    let mut trans = ChcTranslator::new(&[]).unwrap();
    trans.add_init(&Spanned::dummy(Expr::Bool(true))).unwrap();
    trans.add_next(&Spanned::dummy(Expr::Bool(true))).unwrap();
    trans.add_safety(&Spanned::dummy(Expr::Bool(true))).unwrap();

    let checked = trans
        .solve_pdr_with_proof_evidence(pdr_config(5, 50))
        .unwrap();
    let evidence = &checked.proof_replay_evidence;
    let consumer = checked
        .proof_consumer_evidence
        .as_ref()
        .expect("AY typed CHC consumer evidence should be preserved for TLA consumers");

    assert!(evidence.contains("TLA ay_chc_proof_replay_boundary"));
    assert!(evidence.contains("status=Available"));
    assert!(evidence.contains("status_code=typed_chc_proof_transcript"));
    assert!(evidence.contains("schema=ay.chc-proof-transcript/v1"));
    assert!(evidence.contains("schema_version=1"));
    assert!(evidence.contains("normalized_input_schema=ay.chc.normalized-input/v1"));
    assert!(evidence.contains("engine=pdr"));
    assert!(evidence.contains("typed_consumer=true"));
    assert!(evidence.contains("trust_full_verifier_admissible=false"));
    assert!(evidence.contains("production_selected=false"));
    assert!(evidence.contains("fail_closed=true"));
    assert_eq!(
        consumer.schema,
        "ay.chc-proof-transcript-consumer-evidence/v1"
    );
    assert_eq!(consumer.schema_version, 1);
    assert_eq!(
        consumer.normalized_input_schema,
        "ay.chc.normalized-input/v1"
    );
    assert_eq!(consumer.engine, "pdr");
    assert_eq!(consumer.backend_code, "ay_chc_pdr");
    assert_eq!(consumer.trust_status, "trust_full_verifier_rejected");
    assert!(!consumer.trust_full_verifier_admissible);
    assert!(consumer.property_id.contains(&consumer.query_id));
    assert_eq!(consumer.property_sha256.len(), 64);

    match checked.result {
        PdrCheckResult::Safe { .. } => {
            assert!(evidence.contains("result=safe"));
            assert!(evidence.contains("proof_status=verified-invariant"));
            assert!(evidence.contains("accepted_as_proof=true"));
            assert!(evidence
                .contains("trust_full_verifier_non_admission_reason=metadata_only_missing_checked_replay_artifacts"));
            assert!(evidence.contains("unknown_reason=none"));
            assert_eq!(consumer.verdict_code, "safe");
            assert_eq!(consumer.proof_status, "verified-invariant");
            assert!(consumer.accepted_for_consumer);
            assert_eq!(consumer.consumer_rejection_code, None);
            assert!(consumer.model_validated);
            assert_eq!(consumer.model_validation_status, "validated");
            assert_eq!(
                consumer.verification_level_code,
                "ay_chc_verified_invariant"
            );
            assert_eq!(
                consumer.trust_full_verifier_non_admission_reason.as_deref(),
                Some("metadata_only_missing_checked_replay_artifacts")
            );
            assert_eq!(consumer.unknown_reason_code, None);
        }
        PdrCheckResult::Unknown { .. } => {
            assert!(evidence.contains("result=unknown"));
            assert!(evidence.contains("proof_status=non-proof"));
            assert!(evidence.contains("accepted_as_proof=false"));
            assert_eq!(consumer.verdict_code, "unknown");
            assert_eq!(consumer.proof_status, "non-proof");
            assert!(!consumer.accepted_for_consumer);
            assert_eq!(consumer.model_validation_status, "not_validated");
            assert_eq!(consumer.verification_level_code, "ay_chc_non_proof");
            assert!(consumer
                .consumer_rejection_code
                .as_deref()
                .is_some_and(|code| code.starts_with("ay_chc_unknown_")));
            assert!(consumer.unknown_reason_code.is_some());
        }
        PdrCheckResult::Unsafe { .. } => {
            panic!("TRUE safety with no state should not produce a CHC counterexample");
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_simple_counter_unsafe() {
    // Counter that grows unboundedly but claims count <= 5
    // Init: count = 0
    // Next: count' = count + 1 (no guard)
    // Safety: count <= 5
    //
    // This is UNSAFE because count eventually exceeds 5

    let mut trans = ChcTranslator::new(&[("count", TlaSort::Int)]).unwrap();

    // Init: count = 0
    let init = eq_expr(var_expr("count"), int_expr(0));
    trans.add_init(&init).unwrap();

    // Next: count' = count + 1
    let next = eq_expr(
        prime_expr("count"),
        add_expr(var_expr("count"), int_expr(1)),
    );
    trans.add_next(&next).unwrap();

    // Safety: count <= 5
    let safety = le_expr(var_expr("count"), int_expr(5));
    trans.add_safety(&safety).unwrap();

    // Solve
    let result = trans.solve_pdr(pdr_test_config()).unwrap();

    // Should be Unsafe (or Unknown if PDR can't prove it)
    match result {
        PdrCheckResult::Unsafe { trace } => {
            // Expected: found counterexample
            // Trace should show count going from 0 to 6
            assert!(!trace.is_empty(), "counterexample should have states");
        }
        PdrCheckResult::Unknown { .. } => {
            // Acceptable: PDR may not find the counterexample in limited iterations
        }
        PdrCheckResult::Safe { .. } => {
            panic!("Expected Unsafe or Unknown, got Safe for unsafe spec");
        }
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_two_variables() {
    // Two-variable spec: x increments, y = 2*x, invariant y >= 0
    // Init: x = 0 /\ y = 0
    // Next: x' = x + 1 /\ y' = y + 2
    // Safety: y >= 0
    //
    // This is SAFE: y starts at 0 and only increases

    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int), ("y", TlaSort::Int)]).unwrap();

    // Init: x = 0 /\ y = 0
    let init = and_expr(
        eq_expr(var_expr("x"), int_expr(0)),
        eq_expr(var_expr("y"), int_expr(0)),
    );
    trans.add_init(&init).unwrap();

    // Next: x' = x + 1 /\ y' = y + 2
    let next = and_expr(
        eq_expr(prime_expr("x"), add_expr(var_expr("x"), int_expr(1))),
        eq_expr(prime_expr("y"), add_expr(var_expr("y"), int_expr(2))),
    );
    trans.add_next(&next).unwrap();

    // Safety: y >= 0
    let safety = Spanned::dummy(Expr::Geq(Box::new(var_expr("y")), Box::new(int_expr(0))));
    trans.add_safety(&safety).unwrap();

    let result = trans.solve_pdr(pdr_test_config()).unwrap();

    match result {
        PdrCheckResult::Safe { .. } | PdrCheckResult::Unknown { .. } => {
            // Expected
        }
        PdrCheckResult::Unsafe { .. } => {
            panic!("Expected Safe or Unknown, got Unsafe for safe spec");
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_disjunctive_next() {
    // Spec with disjunctive Next (two actions)
    // Init: x = 0
    // Next: (x < 5 /\ x' = x + 1) \/ (x >= 5 /\ x' = x)  (saturates at 5)
    // Safety: x <= 10
    //
    // This is SAFE: x can reach at most 5

    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int)]).unwrap();

    // Init: x = 0
    let init = eq_expr(var_expr("x"), int_expr(0));
    trans.add_init(&init).unwrap();

    // Action 1: x < 5 /\ x' = x + 1
    let action1 = and_expr(
        lt_expr(var_expr("x"), int_expr(5)),
        eq_expr(prime_expr("x"), add_expr(var_expr("x"), int_expr(1))),
    );

    // Action 2: x >= 5 /\ x' = x (saturate)
    let action2 = and_expr(
        Spanned::dummy(Expr::Geq(Box::new(var_expr("x")), Box::new(int_expr(5)))),
        eq_expr(prime_expr("x"), var_expr("x")),
    );

    // Next: Action1 \/ Action2
    let next = or_expr(action1, action2);
    trans.add_next(&next).unwrap();

    // Safety: x <= 10
    let safety = le_expr(var_expr("x"), int_expr(10));
    trans.add_safety(&safety).unwrap();

    let result = trans.solve_pdr(pdr_test_config()).unwrap();

    match result {
        PdrCheckResult::Safe { .. } | PdrCheckResult::Unknown { .. } => {
            // Expected
        }
        PdrCheckResult::Unsafe { .. } => {
            panic!("Expected Safe or Unknown, got Unsafe for safe spec");
        }
    }
}
