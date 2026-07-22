// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::time::Duration;

use super::*;
use crate::bind_constants_from_config;
use crate::test_support::parse_module;
use tla_ay::BmcValue;

fn check_spec(src: &str, max_depth: usize) -> Result<BmcResult, BmcError> {
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(max_depth))
}

fn check_spec_with_evidence(src: &str, max_depth: usize) -> Result<BmcRunResult, BmcError> {
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    check_bmc_with_evidence(&module, &config, &ctx, BmcConfig::with_max_depth(max_depth))
}

/// Check a spec using incremental BMC (reuses solver across depths). Part of #3724.
fn check_spec_incremental(src: &str, max_depth: usize) -> Result<BmcResult, BmcError> {
    let module = parse_module(src);
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    check_bmc(
        &module,
        &config,
        &ctx,
        BmcConfig {
            max_depth,
            incremental: true,
            ..BmcConfig::default()
        },
    )
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_end_to_end_violation_depth() {
    let src = r#"
---- MODULE UnsafeCounter ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 5
====
"#;

    let result = check_spec(src, 10).expect("BMC should succeed");
    match result {
        BmcResult::Violation { depth, trace } => {
            assert_eq!(depth, 6, "violation should be discovered at depth 6");
            assert_eq!(trace.len(), 7, "trace should contain states 0 through 6");
            assert!(matches!(
                trace[6].assignments.get("count"),
                Some(BmcValue::Int(6))
            ));
        }
        other => panic!("expected Violation, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_evidence_exposes_ay_consumer_accepted_sat_boundary() {
    let src = r#"
---- MODULE UnsafeCounterEvidence ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 1
====
"#;

    let run = check_spec_with_evidence(src, 3).expect("BMC with evidence should succeed");

    match run.result {
        BmcResult::Violation { depth, .. } => assert_eq!(depth, 2),
        other => panic!("expected Violation, got {other:?}"),
    }
    assert!(run.solver_decision_profile.accepts_model_for_tla_boundary());
    assert!(!run.solver_decision_profile.fail_closed());
    assert!(run
        .solver_decision_profile_evidence()
        .contains("accepted_for_consumer=true"));
    assert!(run
        .solver_decision_profile_evidence()
        .contains("model_validated=true"));
    assert!(run
        .solver_decision_profile_evidence()
        .contains("fail_closed=false"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_end_to_end_bound_reached() {
    let src = r#"
---- MODULE StableCounter ----
VARIABLE x
Init == x \in {0, 1}
Next == x' = x
Safety == x >= 0
====
"#;

    let result = check_spec(src, 5).expect("BMC should succeed");
    match result {
        BmcResult::BoundReached { max_depth } => assert_eq!(max_depth, 5),
        other => panic!("expected BoundReached, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_end_to_end_init_violation_depth_zero() {
    let src = r#"
---- MODULE InitViolation ----
VARIABLE count
Init == count = 10
Next == count' = count
Safety == count <= 5
====
"#;

    let result = check_spec(src, 5).expect("BMC should succeed");
    match result {
        BmcResult::Violation { depth, trace } => {
            assert_eq!(
                depth, 0,
                "initial-state violation should be discovered at depth 0"
            );
            assert_eq!(
                trace.len(),
                1,
                "depth-0 violation should only report the init state"
            );
        }
        other => panic!("expected Violation, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_end_to_end_with_operator_expansion() {
    let src = r#"
---- MODULE OperatorExpansion ----
VARIABLE count
Inc == count' = count + 1
Init == count = 0
Next == count < 5 /\ Inc
Safety == count <= 5
====
"#;

    // `Next == count < 5 /\ Inc` becomes disabled at count=5 (the guard fails,
    // so no successor exists). With sound symbolic deadlock detection (Fix A),
    // BMC now detects this reachable deadlock — agreeing with explicit BFS,
    // which also reports a deadlock here. (Previously BMC ignored deadlock and
    // reported BoundReached; this is the corrected behavior.)
    let result = check_spec(src, 5).expect("BMC should succeed");
    match result {
        BmcResult::Deadlock { depth, trace } => {
            assert_eq!(depth, 5, "deadlock reached at count=5 (depth 5)");
            assert!(matches!(
                trace[5].assignments.get("count"),
                Some(BmcValue::Int(5))
            ));
        }
        other => panic!("expected Deadlock at depth 5, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_end_to_end_with_unchanged() {
    let src = r#"
---- MODULE UnchangedTest ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ UNCHANGED y
Safety == y = 0
====
"#;

    let result = check_spec(src, 5).expect("BMC should succeed");
    match result {
        BmcResult::BoundReached { max_depth } => assert_eq!(max_depth, 5),
        other => panic!("expected BoundReached, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_resolves_replacement_routed_next() {
    let src = r#"
---- MODULE BmcReplacementNext ----
VARIABLE x
Init == x = 0
Next == x' = x
MCNext == IF x < 2 THEN x' = x + 1 ELSE x' = x
Safety == x <= 1
====
"#;

    let module = parse_module(src);
    let mut config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    config.constants.insert(
        "Next".to_string(),
        crate::ConstantValue::Replacement("MCNext".to_string()),
    );

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);

    let result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(3))
        .expect("replacement-routed NEXT should be accepted by symbolic BMC");
    match result {
        BmcResult::Violation { depth, trace } => {
            assert_eq!(
                depth, 2,
                "replacement-routed next should reach x = 2 at depth 2"
            );
            assert!(matches!(
                trace.last().and_then(|state| state.assignments.get("x")),
                Some(BmcValue::Int(2))
            ));
        }
        other => panic!("expected Violation via replacement-routed NEXT, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_with_zero_timeout_returns_unknown() {
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

    let result = check_bmc(
        &module,
        &config,
        &ctx,
        BmcConfig {
            max_depth: 10,
            solve_timeout: Some(Duration::ZERO),
            debug: false,
            incremental: false,
            check_deadlock: false,
        },
    )
    .expect("zero-timeout BMC should return Unknown, not error");

    match result {
        BmcResult::Unknown { reason, .. } => {
            assert!(
                reason.contains("timed out") || reason.contains("unknown"),
                "unexpected unknown reason: {reason}"
            );
        }
        other => panic!("expected Unknown under zero timeout, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_evidence_fails_closed_for_unknown_ay_summary() {
    let src = r#"
---- MODULE BmcTimeoutEvidence ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Safety == x <= 10
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

    let run = check_bmc_with_evidence(
        &module,
        &config,
        &ctx,
        BmcConfig {
            max_depth: 10,
            solve_timeout: Some(Duration::ZERO),
            debug: false,
            incremental: false,
            check_deadlock: false,
        },
    )
    .expect("zero-timeout BMC should return Unknown with evidence, not error");

    match &run.result {
        BmcResult::Unknown { reason, .. } => {
            assert!(
                reason.contains("timed out") || reason.contains("unknown"),
                "unexpected unknown reason: {reason}"
            );
        }
        other => panic!("expected Unknown under zero timeout, got {other:?}"),
    }
    assert!(run.solver_decision_profile.fail_closed());
    assert!(!run.solver_decision_profile.accepts_model_for_tla_boundary());
    assert!(run
        .solver_decision_profile_evidence()
        .contains("decision_code=unknown"));
    assert!(run
        .solver_decision_profile_evidence()
        .contains("unknown_reason_code=timeout"));
    assert!(run
        .solver_decision_profile_evidence()
        .contains("fail_closed=true"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_reifies_config_constants_in_shared_expansion() {
    let src = r#"
---- MODULE BmcConfigConstant ----
CONSTANT N
VARIABLE x
Init == x \in 0..N
Next == x' = x
Safety == x <= N
====
"#;

    let module = parse_module(src);
    let mut config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    config.constants.insert(
        "N".to_string(),
        crate::ConstantValue::Value("3".to_string()),
    );

    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);
    bind_constants_from_config(&mut ctx, &config).expect("config constants should bind");

    let result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(2))
        .expect("BMC should accept config-constant ranges");
    match result {
        BmcResult::BoundReached { max_depth } => assert_eq!(max_depth, 2),
        other => panic!("expected BoundReached, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Incremental BMC tests (Part of #3724)
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_incremental_violation_depth() {
    let src = r#"
---- MODULE IncrUnsafeCounter ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 5
====
"#;

    let result = check_spec_incremental(src, 10).expect("incremental BMC should succeed");
    match result {
        BmcResult::Violation { depth, trace } => {
            assert_eq!(
                depth, 6,
                "incremental: violation should be discovered at depth 6"
            );
            assert_eq!(
                trace.len(),
                7,
                "incremental: trace should contain states 0 through 6"
            );
            assert!(matches!(
                trace[6].assignments.get("count"),
                Some(BmcValue::Int(6))
            ));
        }
        other => panic!("expected Violation, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_incremental_bound_reached() {
    let src = r#"
---- MODULE IncrStableCounter ----
VARIABLE x
Init == x \in {0, 1}
Next == x' = x
Safety == x >= 0
====
"#;

    let result = check_spec_incremental(src, 5).expect("incremental BMC should succeed");
    match result {
        BmcResult::BoundReached { max_depth } => assert_eq!(max_depth, 5),
        other => panic!("expected BoundReached, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_incremental_init_violation_depth_zero() {
    let src = r#"
---- MODULE IncrInitViolation ----
VARIABLE count
Init == count = 10
Next == count' = count
Safety == count <= 5
====
"#;

    let result = check_spec_incremental(src, 5).expect("incremental BMC should succeed");
    match result {
        BmcResult::Violation { depth, trace } => {
            assert_eq!(
                depth, 0,
                "incremental: initial-state violation should be discovered at depth 0"
            );
            assert_eq!(
                trace.len(),
                1,
                "incremental: depth-0 violation should only report the init state"
            );
        }
        other => panic!("expected Violation, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_incremental_with_unchanged() {
    let src = r#"
---- MODULE IncrUnchangedTest ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ UNCHANGED y
Safety == y = 0
====
"#;

    let result = check_spec_incremental(src, 5).expect("incremental BMC should succeed");
    match result {
        BmcResult::BoundReached { max_depth } => assert_eq!(max_depth, 5),
        other => panic!("expected BoundReached, got {other:?}"),
    }
}

/// Verify that incremental and non-incremental BMC produce identical results. Part of #3724.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_incremental_matches_per_depth() {
    let src = r#"
---- MODULE IncrMatchCheck ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 5
====
"#;

    let result_per_depth = check_spec(src, 10).expect("per-depth BMC should succeed");
    let result_incremental =
        check_spec_incremental(src, 10).expect("incremental BMC should succeed");

    match (&result_per_depth, &result_incremental) {
        (
            BmcResult::Violation {
                depth: d1,
                trace: t1,
            },
            BmcResult::Violation {
                depth: d2,
                trace: t2,
            },
        ) => {
            assert_eq!(
                d1, d2,
                "violation depth must match between per-depth and incremental"
            );
            assert_eq!(t1.len(), t2.len(), "trace length must match");
        }
        _ => panic!(
            "expected both Violation, got per_depth={result_per_depth:?}, incr={result_incremental:?}"
        ),
    }
}

// ---------------------------------------------------------------------------
// Compound type end-to-end tests (sets, functions, sequences)
// Part of #3778 (sets), #3786 (functions), #3793 (sequences).
// ---------------------------------------------------------------------------

/// Test: Set-typed variable stays safe when x remains in the set.
///
/// `x \in {1,2,3}` across 3 steps — safety `x <= 3` always holds.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_e2e_set_membership_safe() {
    let src = r#"
---- MODULE SetMemberSafe ----
VARIABLE x
Init == x \in {1, 2, 3}
Next == x' \in {1, 2, 3}
Safety == x <= 3
====
"#;

    let result = check_spec(src, 3).expect("BMC should succeed");
    match result {
        BmcResult::BoundReached { max_depth } => assert_eq!(max_depth, 3),
        other => panic!("expected BoundReached, got {other:?}"),
    }
}

/// Test: x increments past the safe range — violation detected by BMC.
///
/// Init: x = 0, Next: x' = x + 1, Safety: x \in {0,1,2,3,4,5}
/// At depth 6, x = 6 which violates Safety.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_e2e_set_membership_violation() {
    let src = r#"
---- MODULE SetMemberViolation ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Safety == x \in {0, 1, 2, 3, 4, 5}
====
"#;

    let result = check_spec(src, 10).expect("BMC should succeed");
    match result {
        BmcResult::Violation { depth, trace } => {
            assert_eq!(depth, 6, "violation at depth 6 when x = 6 leaves the set");
            assert_eq!(trace.len(), 7);
        }
        other => panic!("expected Violation, got {other:?}"),
    }
}

/// Test: Range membership violation — x exceeds the range.
///
/// Init: x = 0, Next: x' = x + 1, Safety: x \in 0..5
/// At depth 6, x = 6 is NOT in 0..5.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_e2e_range_membership_violation() {
    let src = r#"
---- MODULE RangeMemberViolation ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Safety == x \in 0..5
====
"#;

    let result = check_spec(src, 10).expect("BMC should succeed");
    match result {
        BmcResult::Violation { depth, trace } => {
            assert_eq!(depth, 6, "violation at depth 6 when x = 6 leaves 0..5");
            assert_eq!(trace.len(), 7);
        }
        other => panic!("expected Violation, got {other:?}"),
    }
}

/// Test: Multiple variables with set membership safety.
///
/// x increments, y decrements. Safety: x <= 5 /\ y >= 0.
/// y reaches -1 at depth 4.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_e2e_multi_var_set_membership() {
    let src = r#"
---- MODULE MultiVarSet ----
VARIABLES x, y
Init == x = 0 /\ y = 3
Next == x' = x + 1 /\ y' = y - 1
Safety == x <= 5 /\ y >= 0
====
"#;

    let result = check_spec(src, 10).expect("BMC should succeed");
    match result {
        BmcResult::Violation { depth, trace } => {
            assert_eq!(depth, 4, "y becomes -1 at depth 4");
            assert_eq!(trace.len(), 5);
        }
        other => panic!("expected Violation, got {other:?}"),
    }
}

/// Test: Disjunctive Next with safety violation.
///
/// x starts at 0, can increment by 1 or 2 each step.
/// Safety: x <= 3. Fastest violation: 0 -> 2 -> 4 at depth 2.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_e2e_disjunctive_next_violation() {
    let src = r#"
---- MODULE DisjunctiveNext ----
VARIABLE x
Init == x = 0
Next == x' = x + 1 \/ x' = x + 2
Safety == x <= 3
====
"#;

    let result = check_spec(src, 5).expect("BMC should succeed");
    match result {
        BmcResult::Violation { depth, .. } => {
            // Shortest path: 0 -> 2 -> 4 (depth 2)
            assert!(
                depth <= 4,
                "violation must be reachable within 4 steps; got depth {depth}"
            );
        }
        other => panic!("expected Violation, got {other:?}"),
    }
}

/// Test: Two variables with UNCHANGED — only one variable evolves.
///
/// x increments, y stays at 0 via UNCHANGED.
/// Safety: y = 0 always holds (BoundReached).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_e2e_unchanged_compound_safe() {
    let src = r#"
---- MODULE UnchangedCompoundSafe ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ UNCHANGED y
Safety == y = 0
====
"#;

    let result = check_spec(src, 5).expect("BMC should succeed");
    match result {
        BmcResult::BoundReached { max_depth } => assert_eq!(max_depth, 5),
        other => panic!("expected BoundReached, got {other:?}"),
    }
}

/// MEASUREMENT (certifying verification, AY proof-artifact leg): does AY produce
/// and STRICT-CHECK its own proof for each of the three inductive-safety
/// obligations of the Accumulator? Prints the per-obligation verdict and asserts
/// every obligation is UNSAT (the implication holds); strict-coverage is reported.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_certificate_obligation_proofs_accumulator() {
    let src = "---- MODULE Accumulator ----\n\
               EXTENDS Integers\n\
               VARIABLE x\n\
               Init == x = 0\n\
               Next == x' = x + 1\n\
               Safety == x >= 0\n\
               ====\n";
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    let proofs = certificate_obligation_proofs(src, &config, "x >= 0")
        .expect("obligations must re-derive + discharge");
    assert_eq!(proofs.len(), 4); // init, consecution, safety, deadlock_freedom
    let mut verified = 0;
    for p in &proofs {
        eprintln!(
            "[obligation-proof] {:<12} unsat={} strict_verified={} clean={} lrat={} alethe_len={}",
            p.name,
            p.unsat,
            p.strict_verified,
            p.clean_supported,
            p.lrat_present,
            p.alethe.len()
        );
        assert!(
            p.unsat,
            "obligation `{}` must be UNSAT (the implication holds)",
            p.name
        );
        if p.strict_verified {
            verified += 1;
        }
    }
    eprintln!("[obligation-proof] AY strict-verified {verified}/4 obligation proofs");
    // Regression guard: after the negate_normalized fix, AY strict-verifies all
    // three obligations (Not(comparison) was the only blocker — see negate_normalized).
    assert_eq!(
        verified, 4,
        "all four obligation proofs (incl. deadlock-freedom) must be AY strict-verified"
    );
}

#[test]
fn strict_safety_certificate_state_verifies_an_inductive_spec() {
    use crate::shared_verdict::CertificateVerification;
    // `x` stays 0, so the safety invariant `Safety == x = 0` is 1-inductive and its strict
    // certificate re-discharges (all obligations UNSAT + strict-verified) -> Verified. This
    // is exactly what lets a fused symbolic `Satisfied` RESOLVE the cooperative verdict: a
    // re-checkable proof, not trust. A non-re-dischargeable proof yields MissingVerifier and
    // publish_analytical leaves the slot unresolved (BFS authoritative).
    let module = parse_module(
        "---- MODULE Ind ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x' = x\nSafety == x = 0\n====\n",
    );
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        check_deadlock: false,
        ..Default::default()
    };
    let mut ctx = crate::EvalCtx::new();
    ctx.load_module(&module);
    let vars = crate::ay_shared::collect_state_vars(&module, &ctx);
    assert_eq!(
        strict_safety_certificate_state(&ctx, &config, &vars),
        CertificateVerification::Verified,
        "an inductive safety invariant must re-discharge a strict certificate (Verified)"
    );
}

#[test]
fn safety_certificate_module_records_discharged_obligations_with_smt_evidence() {
    use trust_ir::{ObligationKind, ProofEvidence, ProofStatus};
    // The three discharged Hoare obligations (as produced by `discharge_obligations_with_proofs`).
    let mk = |name: &'static str, clean: bool, proof: &str| super::ObligationProof {
        name,
        unsat: true,
        strict_verified: true,
        clean_supported: clean,
        alethe: format!("(alethe {name})"),
        lrat_present: false,
        bundle_json: Some(proof.to_string()),
    };
    let obs = vec![
        mk("initiation", true, "{\"bundle\":\"init\"}"),
        mk("consecution", false, "{\"bundle\":\"consec\"}"),
        mk("safety", true, "{\"bundle\":\"safety\"}"),
    ];

    // Non-reflexive obligations (distinct antecedent/consequent) → all stay Discharged here.
    let reflexive_pairs: Vec<Option<(tla_core::ast::Expr, tla_core::ast::Expr)>> =
        vec![None, None, None];
    let module =
        super::build_safety_certificate_module("ty.test_inductive_safety", &obs, &reflexive_pairs);

    // J (LoopInvariant) + the three Hoare obligations (TemporalSafety), all Discharged.
    assert_eq!(module.proof_obligations.len(), 4);
    assert_eq!(
        module
            .proof_obligations
            .iter()
            .filter(|o| o.kind == ObligationKind::LoopInvariant)
            .count(),
        1,
        "the inductive invariant J is recorded as one LoopInvariant obligation"
    );
    assert_eq!(
        module
            .proof_obligations
            .iter()
            .filter(|o| o.kind == ObligationKind::TemporalSafety)
            .count(),
        3,
        "initiation/consecution/safety are the three TemporalSafety obligations"
    );
    assert!(
        module
            .proof_obligations
            .iter()
            .all(|o| o.status == ProofStatus::Discharged),
        "every obligation is Discharged (proved by ay SMT)"
    );
    // HONEST tier: Discharged via SMT, never Certified (which needs a CleanCic kernel term).
    assert!(
        module
            .proof_obligations
            .iter()
            .all(|o| o.status != ProofStatus::Certified),
        "no obligation is Certified — that tier requires a CleanCic de-Bruijn proof term"
    );

    // Each Hoare obligation carries its re-checkable SMT proof bundle as evidence.
    assert_eq!(module.proof_certificates.len(), 3);
    for cert in &module.proof_certificates {
        assert_eq!(cert.prover, "ty.ay");
        match &cert.evidence {
            ProofEvidence::SmtProof(bytes) => assert!(
                !bytes.is_empty(),
                "the SMT proof bundle is embedded verbatim (re-checkable offline)"
            ),
            other => panic!("expected SmtProof evidence, got {other:?}"),
        }
    }

    // The module's proof summary accounts for all four obligations.
    let summary = module.proof_summary();
    assert_eq!(
        summary.total(),
        4,
        "all four obligations are accounted for in the proof summary"
    );
}

#[cfg(feature = "clean-cic")]
#[test]
fn reflexive_safety_obligation_reaches_kernel_certified() {
    use trust_ir::{ProofEvidence, ProofStatus};
    let mk = |name: &'static str, proof: &str| super::ObligationProof {
        name,
        unsat: true,
        strict_verified: true,
        clean_supported: true,
        alethe: format!("(alethe {name})"),
        lrat_present: false,
        bundle_json: Some(proof.to_string()),
    };
    let obs = vec![
        mk("initiation", "i"),
        mk("consecution", "c"),
        mk("safety", "s"),
    ];
    // Mark only the safety obligation reflexive with a GENUINE φ ≡ φ pair (J ≡ Safety = x≥0) — the
    // Clean kernel checks the identity at Π(x). embed(x≥0)→embed(x≥0) and accepts.
    use tla_core::ast::Expr as E;
    let sp = |e: E| Box::new(tla_core::Spanned::dummy(e));
    let ge0 = || {
        E::Geq(
            sp(E::Ident("x".to_string(), tla_core::NameId::INVALID)),
            sp(E::Int(num_bigint::BigInt::from(0))),
        )
    };
    let reflexive_pairs: Vec<Option<(E, E)>> = vec![None, None, Some((ge0(), ge0()))];
    let module =
        super::build_safety_certificate_module("ty.test_certified", &obs, &reflexive_pairs);

    // The reflexive safety obligation is promoted to the kernel-checked CERTIFIED tier — the Clean
    // kernel actually accepted the faithful proof term at Π(x). embed(J)→embed(Safety).
    let safety = module
        .proof_obligations
        .iter()
        .find(|o| o.description.contains(": safety"))
        .expect("safety obligation present");
    assert_eq!(
        safety.status,
        ProofStatus::Certified,
        "a reflexive, SMT-discharged safety obligation reaches kernel-Certified"
    );
    assert!(
        module
            .proof_certificates
            .iter()
            .any(|c| c.obligation == safety.id
                && matches!(c.evidence, ProofEvidence::CleanCic { .. })),
        "the Certified safety obligation carries a CleanCic certificate"
    );

    // Non-reflexive obligations stay at the honest Discharged (SMT) tier — fail-closed: the kernel
    // is never asked to (and never does) certify them.
    let consec = module
        .proof_obligations
        .iter()
        .find(|o| o.description.ends_with("consecution"))
        .expect("consecution obligation present");
    assert_eq!(
        consec.status,
        ProofStatus::Discharged,
        "non-reflexive obligations stay Discharged (no false Certified)"
    );
}

// ===========================================================================
// T2 decomposition widenings (docs/cert/alln-fragment-widening.md): unit tests
// for the Enabled derivation — DNF distribution (fail-closed cap), primed
// Int range/comparison guards (same-assignment substitution), ITE lifting
// (unprimed conditions only), nested-UNCHANGED flattening. These pin the
// DERIVED Enabled AST shape (the SOUND substitution — the disjunct's own
// assignment, never a fresh existential) without any solving.
// ===========================================================================

/// Re-derive the obligation inputs for a widening test spec (J = `x >= 0`).
/// `None` == the honest structural decline (NotRederivable upstream).
fn rederive_widening(spec: &str) -> Option<ObligationInputs> {
    let config = crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    rederive_obligation_inputs(spec, &config, "x >= 0")
}

/// Linearize an expression to a structural S-expr (idents by name, spans and
/// NameIds ignored) so tests can PIN the derived Enabled shape exactly.
fn sexpr(e: &Spanned<Expr>) -> String {
    match &e.node {
        Expr::And(a, b) => format!("(and {} {})", sexpr(a), sexpr(b)),
        Expr::Or(a, b) => format!("(or {} {})", sexpr(a), sexpr(b)),
        Expr::Not(a) => format!("(not {})", sexpr(a)),
        Expr::Eq(a, b) => format!("(= {} {})", sexpr(a), sexpr(b)),
        Expr::Leq(a, b) => format!("(<= {} {})", sexpr(a), sexpr(b)),
        Expr::Lt(a, b) => format!("(< {} {})", sexpr(a), sexpr(b)),
        Expr::Geq(a, b) => format!("(>= {} {})", sexpr(a), sexpr(b)),
        Expr::Gt(a, b) => format!("(> {} {})", sexpr(a), sexpr(b)),
        Expr::Add(a, b) => format!("(+ {} {})", sexpr(a), sexpr(b)),
        Expr::Sub(a, b) => format!("(- {} {})", sexpr(a), sexpr(b)),
        Expr::Mul(a, b) => format!("(* {} {})", sexpr(a), sexpr(b)),
        Expr::In(a, b) => format!("(in {} {})", sexpr(a), sexpr(b)),
        Expr::Range(a, b) => format!("(range {} {})", sexpr(a), sexpr(b)),
        Expr::Prime(a) => format!("(prime {})", sexpr(a)),
        Expr::Ident(n, _) | Expr::StateVar(n, ..) => n.clone(),
        Expr::Int(n) => n.to_string(),
        Expr::Bool(b) => b.to_string(),
        Expr::String(s) => format!("\"{s}\""),
        _ => "?".to_string(),
    }
}

/// DNF distribution + primed range guard: `(x'=x+1 \/ x'=x-1) /\ x' \in 1..N`
/// derives `Enabled = (1<=x+1 /\ x+1<=N) \/ (1<=x-1 /\ x-1<=N)` — the guard is
/// substituted with EACH disjunct's OWN assignment (T2 rail: never a fresh
/// existential `\E v': v' \in 1..N`, which would weaken ~Enabled unsoundly).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_dnf_distribution_substitutes_own_assignment() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLES x, y\n\
                Init == x = 1 /\\ y = 0\n\
                Next == (x' = x + 1 \\/ x' = x - 1) /\\ x' \\in 1..N /\\ UNCHANGED y\n\
                Safety == x >= 0\n\
                ====\n";
    let inputs = rederive_widening(spec).expect("DNF + primed-range shape must decompose");
    assert_eq!(
        sexpr(&inputs.enabled),
        "(or (and (<= 1 (+ x 1)) (<= (+ x 1) N)) (and (<= 1 (- x 1)) (<= (- x 1) N)))",
        "Enabled must substitute the disjunct's own assignment into the range guard"
    );
}

/// A primed COMPARISON guard resolves the same way (orientation preserved),
/// including against an UNCHANGED witness (`v' := v`).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_primed_cmp_guard_and_unchanged_witness() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLES x, y\n\
                Init == x = 1 /\\ y = 0\n\
                Next == x' = x + 1 /\\ x' <= N /\\ UNCHANGED y /\\ N >= y'\n\
                Safety == x >= 0\n\
                ====\n";
    let inputs = rederive_widening(spec).expect("primed cmp guards must decompose");
    assert_eq!(
        sexpr(&inputs.enabled),
        "(and (<= (+ x 1) N) (>= N y))",
        "cmp guards must substitute the Eq witness (x+1) and the UNCHANGED witness (y)"
    );
}

/// ITE lift: an unprimed condition splits the action; the else-branch enters
/// Enabled as an ordinary `~g` guard.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_ite_lift_unprimed_condition() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                VARIABLES x, y\n\
                Init == x = 1 /\\ y = 0\n\
                Next == x' = (IF y = 0 THEN x + 1 ELSE x + 2) /\\ UNCHANGED y\n\
                Safety == x >= 0\n\
                ====\n";
    let inputs = rederive_widening(spec).expect("unprimed-condition ITE must lift");
    assert_eq!(
        sexpr(&inputs.enabled),
        "(or (= y 0) (not (= y 0)))",
        "the lifted condition and its negation must be the two disjuncts' guards"
    );
}

/// TWIN — a PRIMED ITE condition must NOT lift (assignment-order semantics):
/// the conjunct stays Opaque and the analysis declines.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_ite_primed_condition_declines() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                VARIABLES x, y\n\
                Init == x = 1 /\\ y = 0\n\
                Next == x' = x + 1 /\\ (IF x' > x THEN y' = 0 ELSE y' = 1)\n\
                Safety == x >= 0\n\
                ====\n";
    assert!(
        rederive_widening(spec).is_none(),
        "a primed ITE condition must decline the Enabled derivation"
    );
}

/// TWIN — a primed guard on a var assigned by SET MEMBERSHIP (`w' \in S`) has
/// no deterministic witness to substitute: decline.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_primed_guard_on_membership_assigned_var_declines() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                VARIABLES x, w\n\
                Init == x = 1 /\\ w = 1\n\
                Next == x' = x + 1 /\\ w' \\in {1, 2} /\\ w' <= 5\n\
                Safety == x >= 0\n\
                ====\n";
    assert!(
        rederive_widening(spec).is_none(),
        "a primed guard on a membership-assigned var must decline (no witness)"
    );
}

/// TWIN — a primed guard on an ∃k-assigned var: substituting the k-dependent
/// RHS would leak the skolem into ~Enabled (out of QF): decline.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_primed_guard_on_exists_assigned_var_declines() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLES x, y\n\
                Init == x = 1 /\\ y = 0\n\
                Next == (\\E k \\in 1..5 : x' = x + k) /\\ x' <= N /\\ UNCHANGED y\n\
                Safety == x >= 0\n\
                ====\n";
    assert!(
        rederive_widening(spec).is_none(),
        "a primed guard on an exists-assigned var must decline (no witness)"
    );
}

/// TWIN — DNF cap is FAIL-CLOSED: 2^7 = 128 distributed disjuncts (> cap 64)
/// must DECLINE the derivation, never truncate.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_dnf_cap_fail_closed() {
    let vars = ["x", "b", "c", "d", "e", "f", "g"];
    let next: Vec<String> = vars
        .iter()
        .map(|v| format!("({v}' = 1 \\/ {v}' = 2)"))
        .collect();
    let spec = format!(
        "---- MODULE M ----\n\
         EXTENDS Integers\n\
         VARIABLES {}\n\
         Init == x = 1\n\
         Next == {}\n\
         Safety == x >= 0\n\
         ====\n",
        vars.join(", "),
        next.join(" /\\ ")
    );
    assert!(
        rederive_widening(&spec).is_none(),
        "128 distributed disjuncts exceed the cap and must decline (fail-closed)"
    );
    // The 2^6 = 64 sibling sits EXACTLY at the cap and must still decompose.
    let next6: Vec<String> = vars[..6]
        .iter()
        .map(|v| format!("({v}' = 1 \\/ {v}' = 2)"))
        .collect();
    let spec6 = format!(
        "---- MODULE M ----\n\
         EXTENDS Integers\n\
         VARIABLES {}\n\
         Init == x = 1\n\
         Next == {}\n\
         Safety == x >= 0\n\
         ====\n",
        vars[..6].join(", "),
        next6.join(" /\\ ")
    );
    let inputs = rederive_widening(&spec6).expect("64 disjuncts == cap must decompose");
    // 64 unguarded total disjuncts: Enabled is a disjunction of TRUEs.
    assert!(sexpr(&inputs.enabled).contains("true"));
}

/// Nested UNCHANGED tuples flatten (`UNCHANGED <<y, <<z>>>>` == both members
/// unchanged): the spec decomposes and Enabled is TRUE.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_nested_unchanged_flattens() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                VARIABLES x, y, z\n\
                Init == x = 1 /\\ y = 0 /\\ z = 0\n\
                Next == x' = x + 1 /\\ UNCHANGED <<y, <<z>>>>\n\
                Safety == x >= 0\n\
                ====\n";
    let inputs = rederive_widening(spec).expect("nested UNCHANGED must flatten");
    assert_eq!(sexpr(&inputs.enabled), "true");
}

/// LET inlining (in `expand_operators_for_chc`): parameterless AND
/// parameterized non-recursive LET defs inline, so the action classifies.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_let_inlined_action_decomposes() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                VARIABLES x, y\n\
                Init == x = 1 /\\ y = 0\n\
                Next == LET delta == 1\n\
                            bump(v) == v + delta\n\
                        IN x' = bump(x) /\\ UNCHANGED y\n\
                Safety == x >= 0\n\
                ====\n";
    let inputs = rederive_widening(spec).expect("non-recursive LET must inline");
    assert_eq!(sexpr(&inputs.enabled), "true");
    assert_eq!(
        sexpr(&inputs.next),
        "(and (= (prime x) (+ x 1)) ?)",
        "the expanded Next must carry the beta-reduced assignment (no LET node)"
    );
}

/// TWIN — a RECURSIVE LET def stays verbatim (the wrapper is kept) and the
/// action honestly declines.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enabled_recursive_let_declines() {
    let spec = "---- MODULE M ----\n\
                EXTENDS Integers\n\
                VARIABLES x, y\n\
                Init == x = 1 /\\ y = 0\n\
                Next == LET RECURSIVE f(_)\n\
                            f(n) == IF n <= 0 THEN 0 ELSE f(n - 1)\n\
                        IN x' = x + f(1) /\\ UNCHANGED y\n\
                Safety == x >= 0\n\
                ====\n";
    assert!(
        rederive_widening(spec).is_none(),
        "a recursive LET must keep the wrapper and decline"
    );
}


// ===========================================================================
// Function-state all-N: symbolic-domain FunctionSym + pointwise-∀ discipline.
// Fast unit tests (NO solver): sort recognition, obligation-transform shape,
// index collection, and mint/verify determinism. Solver-backed accept/decline
// twins live in cert_all_n.rs. See docs/cert/function-state-alln-design.md.
//
// NOTE: the function-set membership is written as a leading `/\` conjunct
// (`TypeOK == /\ f \in [D -> R]`), matching real corpus specs (ewd426); a bare
// top-level `f \in [S -> T]` operator body is misparsed by the front end as an
// action subscript `[A]_e`, unrelated to this feature.
// ===========================================================================

#[cfg(test)]
fn fs_config() -> crate::Config {
    crate::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["TypeOK".to_string()],
        ..Default::default()
    }
}

/// Build `ObligationInputs` for a FunctionSym spec, then pin `J = Safety` (the
/// invariant) — the certify path does the same when covering a TypeOK conjunct.
/// We inject a scalar `TRUE` as the placeholder invariant (a function-set body
/// injected as the trailing cert operator hits the `[A]_e` parse ambiguity).
#[cfg(test)]
fn fs_inputs(spec: &str, config: &crate::Config) -> ObligationInputs {
    let mut inputs =
        rederive_obligation_inputs(spec, config, "TRUE").expect("FunctionSym spec must rederive");
    inputs.j = inputs.safety.clone();
    inputs
}

/// A `f \in [1..N -> 0..1]` type constraint over an UNBOUND `N` must infer the
/// state var as a symbolic-domain `FunctionSym` (Layer-1/2 recognition, end to
/// end through `rederive`), not a scalar Int nor a finite `Function`.
#[test]
fn funcstate_symbolic_domain_inferred_as_functionsym() {
    let spec = "---- MODULE FSInfer ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLE f\n\
                Init == /\\ f \\in [1..N -> 0..1]\n\
                Next == f' = [f EXCEPT ![1] = 0]\n\
                TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                ====\n";
    let inputs = fs_inputs(spec, &fs_config());
    let f_sort = inputs
        .var_sorts
        .iter()
        .find(|(n, _)| n == "f")
        .map(|(_, s)| s.clone())
        .expect("state var f present");
    assert!(
        matches!(
            &f_sort,
            TlaSort::FunctionSym { domain_lo: 1, domain_hi_const, domain_hi_offset: 0, range }
                if domain_hi_const == "N" && matches!(**range, TlaSort::Int)
        ),
        "expected FunctionSym{{1..N -> Int}}, got {f_sort:?}"
    );
    assert!(funcsym_present(&inputs.var_sorts));
}

/// The `0..N-1` (affine `N-1`) domain of ewd426/TokenRing must be recognised
/// with `domain_lo = 0` and `domain_hi_offset = -1`, and only `N` symbolic (M
/// config-bound).
#[test]
fn funcstate_affine_upper_bound_recognized() {
    let spec = "---- MODULE FSAffine ----\n\
                EXTENDS Integers\n\
                CONSTANTS N, M\n\
                VARIABLE c\n\
                Node == 0 .. N-1\n\
                Init == /\\ c \\in [ Node -> 0 .. M-1 ]\n\
                Next == c' = [ c EXCEPT ![0] = 0 ]\n\
                TypeOK == /\\ c \\in [ Node -> 0 .. M-1 ]\n\
                ====\n";
    let mut config = fs_config();
    config
        .constants
        .insert("M".to_string(), crate::config::ConstantValue::Value("6".to_string()));
    let inputs = fs_inputs(spec, &config);
    let c_sort = inputs
        .var_sorts
        .iter()
        .find(|(n, _)| n == "c")
        .map(|(_, s)| s.clone())
        .expect("state var c present");
    assert!(
        matches!(
            &c_sort,
            TlaSort::FunctionSym { domain_lo: 0, domain_hi_const, domain_hi_offset: -1, .. }
                if domain_hi_const == "N"
        ),
        "expected FunctionSym{{0..N-1}}, got {c_sort:?}"
    );
}

/// The consecution transform must (a) skolemize the negated pointwise goal to a
/// fresh `__ty_pw_0` rigid index, and (b) instantiate the hypothesis (no longer
/// a raw membership). The Next selector `\E p` also skolemizes.
#[test]
fn funcstate_consecution_transform_skolemizes_and_instantiates() {
    let spec = "---- MODULE FSTrans ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLE f\n\
                Init == /\\ f \\in [1..N -> 0..1]\n\
                Next == \\E p \\in 1..N : f' = [f EXCEPT ![p] = 0]\n\
                TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                ====\n";
    let inputs = fs_inputs(spec, &fs_config());
    let pt = transform_all_n_pointwise(SmtObligation::Consecution, &inputs);
    assert!(
        pt.skolem_consts.iter().any(|c| c == "__ty_pw_0"),
        "goal skolem __ty_pw_0 must be minted, got {:?}",
        pt.skolem_consts
    );
    assert!(
        pt.skolem_consts.iter().any(|c| c.starts_with("__ty_skolem_")),
        "Next selector skolem must be minted, got {:?}",
        pt.skolem_consts
    );
    assert!(
        !matches!(pt.j.node, Expr::In(..)),
        "hypothesis J must be instantiated, not a raw membership"
    );
}

/// `transform_all_n_pointwise` is deterministic (identical skolem names and
/// structure across repeated calls) — the property that makes the mint/verify
/// render binding match term-for-term.
#[test]
fn funcstate_transform_is_deterministic() {
    let spec = "---- MODULE FSDet ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLE f\n\
                Init == /\\ f \\in [1..N -> 0..1]\n\
                Next == \\E p \\in 1..N : f' = [f EXCEPT ![p] = 0]\n\
                TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                ====\n";
    let inputs = fs_inputs(spec, &fs_config());
    for ob in [
        SmtObligation::Initiation,
        SmtObligation::Consecution,
        SmtObligation::Safety,
    ] {
        let a = transform_all_n_pointwise(ob, &inputs);
        let b = transform_all_n_pointwise(ob, &inputs);
        assert_eq!(a.skolem_consts, b.skolem_consts, "skolem naming must match");
        assert!(eq_ignore_span(&a.j, &b.j));
        assert!(eq_ignore_span(&a.not_j, &b.not_j));
        assert!(eq_ignore_span(&a.init, &b.init));
        assert!(eq_ignore_span(&a.next, &b.next));
    }
}

/// A SET-valued codomain (`[1..N -> SUBSET (1..3)]`) is NON-scalar, so it must
/// NOT be recognised as `FunctionSym` (rail: set/sequence codomains are OUT —
/// they need array extensionality, which is fail-closed at strict re-check).
#[test]
fn funcstate_set_codomain_not_functionsym() {
    let spec = "---- MODULE FSSetCod ----\n\
                EXTENDS Integers, FiniteSets\n\
                CONSTANT N\n\
                VARIABLE f\n\
                Init == /\\ f \\in [1..N -> SUBSET (1..3)]\n\
                Next == UNCHANGED f\n\
                TypeOK == /\\ f \\in [1..N -> SUBSET (1..3)]\n\
                ====\n";
    let inputs =
        rederive_obligation_inputs(spec, &fs_config(), "TRUE").expect("must rederive");
    let f_sort = inputs
        .var_sorts
        .iter()
        .find(|(n, _)| n == "f")
        .map(|(_, s)| s.clone())
        .expect("f present");
    assert!(
        !matches!(f_sort, TlaSort::FunctionSym { .. }),
        "a set codomain must not be a FunctionSym, got {f_sort:?}"
    );
}
