// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! End-to-end PDR tests: basic counter and operator specs
//!
//! Split from ay_pdr/tests.rs — Part of #3692

use super::helpers::pdr_config;
use super::*;
use crate::shared_verdict::{SharedVerdict, Verdict};
use crate::test_support::parse_module;
use std::sync::Arc;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_pdr_end_to_end_safe_counter() {
    // Safe counter: Init: count = 0, Next: count < 5 /\ count' = count + 1
    // Safety: count <= 5
    let src = r#"
---- MODULE SafeCounter ----
VARIABLE count
Init == count = 0
Next == count < 5 /\ count' = count + 1
Safety == count <= 5
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_pdr(&module, &config, &ctx);
    match result {
        Ok(PdrResult::Safe { .. }) | Ok(PdrResult::Unknown { .. }) => {
            // Expected for safe spec
        }
        Ok(PdrResult::Unsafe { .. }) => {
            panic!("Expected Safe or Unknown for safe counter spec");
        }
        Err(e) => {
            panic!("PDR failed unexpectedly: {}", e);
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_pdr_with_config_exposes_chc_proof_replay_evidence() {
    let src = r#"
---- MODULE PdrEvidenceStable ----
VARIABLE x
Init == x = 0
Next == x' = x
Safety == x = 0
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let run = check_pdr_with_config_and_evidence(&module, &config, &ctx, pdr_config(10, 100))
        .expect("PDR evidence run should complete");
    let evidence = run.proof_replay_evidence();
    let boundary = run.proof_replay_boundary();

    assert!(evidence.contains("TLA ay_chc_proof_replay_boundary"));
    assert!(evidence.contains("status=Available"));
    assert!(evidence.contains("status_code=typed_chc_proof_transcript"));
    assert!(evidence.contains("schema=ay.chc-proof-transcript/v1"));
    assert!(evidence.contains("normalized_input_schema=ay.chc.normalized-input/v1"));
    assert!(evidence.contains("typed_consumer=true"));
    assert!(evidence.contains("production_selected=false"));
    assert!(evidence.contains("fail_closed=true"));
    assert!(run
        .shared_engine_lane_evidence()
        .contains("TLA ay_shared_engine_lane_metadata"));
    assert!(run.shared_engine_lane_evidence().contains("lane=pdr"));
    assert!(run.shared_engine_lane_evidence().contains(
        "compatible_frontends=TLA,Quint,MCC/Petri,AIGER,BTOR2,AY-only,VMT/replay,witness/replay"
    ));
    assert_eq!(boundary.status(), "Available");
    assert_eq!(boundary.status_code(), "typed_chc_consumer_evidence");
    assert_eq!(boundary.row_status_code(), "typed_chc_proof_transcript");
    assert!(boundary.typed_consumer());
    assert!(boundary.accepted_as_proof());
    assert!(boundary.accepted_for_consumer());
    assert!(!boundary.trust_full_verifier_admissible());
    assert_eq!(
        boundary.trust_full_verifier_non_admission_reason(),
        "metadata_only_missing_checked_replay_artifacts"
    );
    assert!(boundary.fail_closed());
    assert!(!boundary.accepts_proof_for_tla_boundary());
    let consumer = boundary
        .consumer_evidence()
        .expect("typed AY CHC consumer evidence should be available");
    assert_eq!(
        consumer.schema,
        "ay.chc-proof-transcript-consumer-evidence/v1"
    );
    assert_eq!(consumer.schema_version, 1);
    assert_eq!(consumer.verdict_code, "safe");
    assert_eq!(consumer.backend_code, "ay_chc_pdr");
    assert_eq!(consumer.proof_status, "verified-invariant");
    assert!(consumer.accepted_for_consumer);
    assert_eq!(consumer.consumer_rejection_code, None);
    assert!(consumer.model_validated);
    assert_eq!(consumer.model_validation_status, "validated");
    assert_eq!(
        consumer.verification_level_code,
        "ay_chc_verified_invariant"
    );
    assert_eq!(consumer.trust_status, "trust_full_verifier_rejected");
    assert_eq!(
        consumer.trust_full_verifier_non_admission_reason.as_deref(),
        Some("metadata_only_missing_checked_replay_artifacts")
    );
    assert_eq!(consumer.replay_status, "replay-artifacts-required");
    assert_eq!(consumer.transcript_status, "metadata-only");
    assert_eq!(consumer.unknown_reason_code, None);
    assert_eq!(consumer.unsafe_trace, None);
    assert!(run.proof_consumer_evidence().is_some());
}

#[test]
fn test_pdr_boundary_without_typed_metadata_stays_fail_closed_without_row_parsing() {
    let row = "TLA ay_chc_proof_replay_boundary status=Available status_code=typed_chc_proof_transcript accepted_as_proof=true trust_full_verifier_admissible=true fail_closed=false";
    let boundary = AYChcProofReplayEvidence::from_observable_proof_replay_row(row.to_string());

    assert_eq!(boundary.evidence_row(), row);
    assert_eq!(boundary.status(), "Unavailable");
    assert_eq!(boundary.status_code(), "missing_public_chc_metadata");
    assert_eq!(boundary.row_status_code(), "typed_chc_proof_transcript");
    assert!(!boundary.typed_consumer());
    assert!(!boundary.accepted_as_proof());
    assert!(!boundary.trust_full_verifier_admissible());
    assert_eq!(
        boundary.trust_full_verifier_non_admission_reason(),
        "missing_public_chc_metadata"
    );
    assert!(boundary.consumer_evidence().is_none());
    assert!(boundary.fail_closed());
    assert!(!boundary.accepts_proof_for_tla_boundary());
}

#[test]
fn test_pdr_portfolio_evidence_fails_closed_when_lane_exits_before_ay() {
    let src = r#"
---- MODULE PdrEvidenceSkipped ----
VARIABLE x
Init == x = 0
Next == x' = x
Safety == x = 0
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);
    let verdict = Arc::new(SharedVerdict::new());
    verdict.publish(Verdict::Satisfied);

    let run = check_pdr_with_portfolio_and_evidence(
        &module,
        &config,
        &ctx,
        pdr_config(5, 50),
        Some(verdict),
    )
    .expect("PDR early-exit evidence should still be returned");
    let evidence = run.proof_replay_evidence();
    let boundary = run.proof_replay_boundary();

    assert!(matches!(&run.result, PdrResult::Unknown { .. }));
    assert!(evidence.contains("TLA ay_chc_proof_replay_boundary"));
    assert!(evidence.contains("status=Unavailable"));
    assert!(evidence.contains("status_code=missing_typed_chc_proof_transcript"));
    assert!(evidence.contains("typed_consumer=false"));
    assert!(evidence.contains("expected_schema=ay.chc-proof-transcript/v1"));
    assert!(evidence.contains("expected_normalized_input_schema=ay.chc.normalized-input/v1"));
    assert!(evidence.contains("production_selected=false"));
    assert!(evidence.contains("fail_closed=true"));
    assert_eq!(boundary.status(), "Unavailable");
    assert_eq!(boundary.status_code(), "missing_typed_chc_proof_transcript");
    assert_eq!(
        boundary.row_status_code(),
        "missing_typed_chc_proof_transcript"
    );
    assert!(!boundary.typed_consumer());
    assert!(!boundary.accepted_as_proof());
    assert!(!boundary.trust_full_verifier_admissible());
    assert_eq!(
        boundary.trust_full_verifier_non_admission_reason(),
        "missing_typed_chc_proof_transcript"
    );
    assert!(boundary.consumer_evidence().is_none());
    assert!(boundary.fail_closed());
    assert!(!boundary.accepts_proof_for_tla_boundary());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_pdr_end_to_end_unsafe_counter() {
    // Unsafe counter: grows unboundedly but claims count <= 5
    let src = r#"
---- MODULE UnsafeCounter ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 5
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_pdr(&module, &config, &ctx);
    match result {
        Ok(PdrResult::Unsafe { trace }) => {
            // Expected: counterexample found
            assert!(!trace.is_empty(), "counterexample should have states");
        }
        Ok(PdrResult::Unknown { .. }) => {
            // Acceptable: PDR may not find counterexample in limited iterations
        }
        Ok(PdrResult::Safe { .. }) => {
            panic!("Expected Unsafe or Unknown for unsafe counter spec");
        }
        Err(e) => {
            panic!("PDR failed unexpectedly: {}", e);
        }
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_pdr_end_to_end_two_vars_safe() {
    // Two-variable spec with invariant
    let src = r#"
---- MODULE TwoVars ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ y' = y + 2
Safety == y >= 0
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_pdr_with_config(&module, &config, &ctx, pdr_config(10, 100));
    match result {
        Ok(PdrResult::Safe { .. }) | Ok(PdrResult::Unknown { .. }) => {
            // Expected for safe spec (y only increases from 0)
        }
        Ok(PdrResult::Unsafe { .. }) => {
            panic!("Expected Safe or Unknown for safe two-var spec");
        }
        Err(e) => {
            panic!("PDR failed unexpectedly: {}", e);
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_pdr_end_to_end_with_operator_expansion() {
    // Test that operator expansion works with Next containing operator calls
    let src = r#"
---- MODULE OperatorExpansion ----
VARIABLE count
Init == count = 0
Inc == count' = count + 1
Next == count < 5 /\ Inc
Safety == count <= 5
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_pdr(&module, &config, &ctx);
    match result {
        Ok(PdrResult::Safe { .. }) | Ok(PdrResult::Unknown { .. }) => {
            // Expected: operator expansion should inline Inc
        }
        Ok(PdrResult::Unsafe { .. }) => {
            panic!("Expected Safe or Unknown for safe spec with operator expansion");
        }
        Err(e) => {
            panic!("PDR failed unexpectedly: {}", e);
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_pdr_end_to_end_with_unchanged() {
    // Test UNCHANGED support
    let src = r#"
---- MODULE UnchangedTest ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ UNCHANGED y
Safety == y = 0
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_pdr(&module, &config, &ctx);
    match result {
        Ok(PdrResult::Safe { .. }) | Ok(PdrResult::Unknown { .. }) => {
            // Expected: y is always 0 because UNCHANGED y
        }
        Ok(PdrResult::Unsafe { .. }) => {
            panic!("Expected Safe or Unknown for spec with UNCHANGED");
        }
        Err(e) => {
            panic!("PDR failed unexpectedly: {}", e);
        }
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_pdr_with_zero_solve_timeout_returns_unknown() {
    use std::time::Duration;

    let src = r#"
---- MODULE TimeoutCounter ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 5
====
"#;
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let mut pdr_config = pdr_config(10, 100);
    pdr_config.solve_timeout = Some(Duration::ZERO);

    let result = check_pdr_with_config(&module, &config, &ctx, pdr_config);
    match result {
        // Portfolio solver may return Unknown (timeout) or Unsafe (BMC finds counterexample
        // instantly even with zero timeout since this spec is trivially unsafe).
        Ok(PdrResult::Unknown { .. }) | Ok(PdrResult::Unsafe { .. }) => {}
        Ok(other) => panic!("expected Unknown or Unsafe under zero solve_timeout, got {other:?}"),
        Err(e) => panic!("expected Unknown or Unsafe under zero solve_timeout, got error: {e}"),
    }
}
