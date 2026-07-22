// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Prepare-phase tests for parallel action bytecode validation.

use super::*;
use crate::config::Config;
use crate::test_support::parse_module;
use tla_tir::bytecode::CompileError;

fn unsupported_failure_message(
    compiled: &tla_eval::bytecode_vm::CompiledBytecode,
    action_name: &str,
) -> String {
    match compiled
        .failed
        .iter()
        .find(|(name, _)| name == action_name)
        .map(|(_, err)| err)
    {
        Some(CompileError::Unsupported(message)) => message.clone(),
        Some(other) => panic!("expected Unsupported failure for {action_name}, got {other:?}"),
        None => panic!("missing failed compile entry for {action_name}"),
    }
}

fn assert_failed_message_contains(
    compiled: &tla_eval::bytecode_vm::CompiledBytecode,
    action_name: &str,
    expected_fragment: &str,
) {
    let message = unsupported_failure_message(compiled, action_name);
    assert!(
        message.contains(expected_fragment),
        "{action_name} should report {expected_fragment:?}, got: {message}",
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn compile_action_bytecode_for_workers_prunes_unsafe_reachable_callees() {
    let _serial = crate::test_utils::acquire_interner_lock();
    let module = parse_module(
        r#"
---- MODULE ParallelPrepareActionCalleeClosure ----
VARIABLES x, y

Good ==
    /\ x' = x + 1
    /\ y' = y

PureHelper == x + 1

SafeWrapped ==
    /\ x' = PureHelper
    /\ y' = y

UnsafeHelper == x' = x + 1

UnsafeWrapped ==
    /\ UnsafeHelper
    /\ y' = y

====
"#,
    );
    let checker = ParallelChecker::new(&module, &Config::default(), 2);
    let ctx = checker
        .prepare_base_ctx()
        .expect("parallel prepare should build EvalCtx");
    let var_registry = std::sync::Arc::new(ctx.var_registry().clone());

    let compiled = checker
        .compile_action_bytecode_for_workers(
            &ctx,
            &var_registry,
            "Good",
            &["SafeWrapped".to_string(), "UnsafeWrapped".to_string()],
        )
        .expect("safe actions should remain available after pruning");

    assert!(
        compiled.op_indices.contains_key("Good"),
        "baseline action should remain available for parallel next-state compilation",
    );
    assert!(
        compiled.op_indices.contains_key("SafeWrapped"),
        "actions calling pure helpers should stay eligible",
    );
    assert!(
        !compiled.op_indices.contains_key("UnsafeWrapped"),
        "actions reaching primed helper callees must be pruned",
    );

    assert_failed_message_contains(compiled.as_ref(), "UnsafeWrapped", "reachable callee");
    assert_failed_message_contains(compiled.as_ref(), "UnsafeWrapped", "LoadPrime");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn compile_action_bytecode_for_workers_prunes_reachable_set_prime_mode_helpers() {
    let _serial = crate::test_utils::acquire_interner_lock();
    let module = parse_module(
        r#"
---- MODULE ParallelPrepareActionPrimeModeClosure ----
VARIABLES x, y

Good ==
    /\ x' = x + 1
    /\ y' = y

CurrentX == x
PrimeCurrentX == CurrentX'

PrimeModeWrapped ==
    /\ x' = PrimeCurrentX
    /\ y' = y

====
"#,
    );
    let checker = ParallelChecker::new(&module, &Config::default(), 2);
    let ctx = checker
        .prepare_base_ctx()
        .expect("parallel prepare should build EvalCtx");
    let var_registry = std::sync::Arc::new(ctx.var_registry().clone());

    let compiled = checker
        .compile_action_bytecode_for_workers(
            &ctx,
            &var_registry,
            "Good",
            &["PrimeModeWrapped".to_string()],
        )
        .expect("baseline safe action should remain available after pruning");

    assert!(
        compiled.op_indices.contains_key("Good"),
        "baseline action should remain available for parallel next-state compilation",
    );
    assert!(
        !compiled.op_indices.contains_key("PrimeModeWrapped"),
        "actions reaching SetPrimeMode helpers must be pruned",
    );

    assert_failed_message_contains(compiled.as_ref(), "PrimeModeWrapped", "reachable callee");
    assert_failed_message_contains(compiled.as_ref(), "PrimeModeWrapped", "SetPrimeMode");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn compile_action_bytecode_for_workers_prunes_multi_hop_unsafe_helper_closures() {
    let _serial = crate::test_utils::acquire_interner_lock();
    let module = parse_module(
        r#"
---- MODULE ParallelPrepareActionMultiHopClosure ----
VARIABLES x, y

Good ==
    /\ x' = x + 1
    /\ y' = y

SafeLeaf == x + 1
SafeMid == SafeLeaf

SafeTwoHopWrapped ==
    /\ x' = SafeMid
    /\ y' = y

UnsafePrimeLeaf == x' = x + 1
UnsafePrimeMid == UnsafePrimeLeaf

UnsafePrimeWrapped ==
    /\ UnsafePrimeMid
    /\ y' = y

CurrentX == x
PrimeCurrentX == CurrentX'
PrimeModeMid == PrimeCurrentX

PrimeModeTwoHopWrapped ==
    /\ x' = PrimeModeMid
    /\ y' = y

====
"#,
    );
    let checker = ParallelChecker::new(&module, &Config::default(), 2);
    let ctx = checker
        .prepare_base_ctx()
        .expect("parallel prepare should build EvalCtx");
    let var_registry = std::sync::Arc::new(ctx.var_registry().clone());

    let compiled = checker
        .compile_action_bytecode_for_workers(
            &ctx,
            &var_registry,
            "Good",
            &[
                "SafeTwoHopWrapped".to_string(),
                "UnsafePrimeWrapped".to_string(),
                "PrimeModeTwoHopWrapped".to_string(),
            ],
        )
        .expect("safe actions should remain available after pruning");

    assert!(
        compiled.op_indices.contains_key("Good"),
        "baseline action should remain available for parallel next-state compilation",
    );
    assert!(
        compiled.op_indices.contains_key("SafeTwoHopWrapped"),
        "pure multi-hop helper chains should stay eligible",
    );
    assert!(
        !compiled.op_indices.contains_key("UnsafePrimeWrapped"),
        "multi-hop LoadPrime helper chains must be pruned",
    );
    assert!(
        !compiled.op_indices.contains_key("PrimeModeTwoHopWrapped"),
        "multi-hop SetPrimeMode helper chains must be pruned",
    );

    assert_failed_message_contains(compiled.as_ref(), "UnsafePrimeWrapped", "reachable callee");
    assert_failed_message_contains(compiled.as_ref(), "UnsafePrimeWrapped", "LoadPrime");
    assert_failed_message_contains(
        compiled.as_ref(),
        "PrimeModeTwoHopWrapped",
        "reachable callee",
    );
    assert_failed_message_contains(compiled.as_ref(), "PrimeModeTwoHopWrapped", "SetPrimeMode");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn compile_action_bytecode_for_workers_keeps_multi_hop_pure_helper_closures_eligible() {
    let _serial = crate::test_utils::acquire_interner_lock();
    let module = parse_module(
        r#"
---- MODULE ParallelPrepareActionPureMultiHopClosure ----
VARIABLES x, y

Good ==
    /\ x' = x + 1
    /\ y' = y

PureLeaf == x + 1
PureMid == PureLeaf
PureTop == PureMid

PureThreeHopWrapped ==
    /\ x' = PureTop
    /\ y' = y

====
"#,
    );
    let checker = ParallelChecker::new(&module, &Config::default(), 2);
    let ctx = checker
        .prepare_base_ctx()
        .expect("parallel prepare should build EvalCtx");
    let var_registry = std::sync::Arc::new(ctx.var_registry().clone());

    let compiled = checker
        .compile_action_bytecode_for_workers(
            &ctx,
            &var_registry,
            "Good",
            &["PureThreeHopWrapped".to_string()],
        )
        .expect("pure helper chains should remain available after pruning");

    assert!(
        compiled.op_indices.contains_key("Good"),
        "baseline action should remain available for parallel next-state compilation",
    );
    assert!(
        compiled.op_indices.contains_key("PureThreeHopWrapped"),
        "pure multi-hop helper chains should stay eligible",
    );
    assert!(
        compiled.failed.is_empty(),
        "pure multi-hop helper chains should not be reported as transform failures: {:?}",
        compiled.failed,
    );
}

/// REGRESSION (audit-2026-07 #11), parallel engine: `prepare_por_state` must
/// build the independence matrix over the ENUMERATION (coverage) action
/// decomposition — the same `por_actions` list the workers iterate — not over
/// a whole-Next with-primes re-detection that splits a named action at its
/// internal top-level disjunction into a finer list. The stopgap returned
/// `independence = None` (POR skipped) on the 2-vs-3 length mismatch; with
/// the per-coverage-action unioned extraction, POR stays ON with a 2x2
/// coverage-indexed matrix.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn prepare_por_state_matrix_is_coverage_indexed() {
    let _serial = crate::test_utils::acquire_interner_lock();
    let module = parse_module(
        r#"
---- MODULE ParallelPreparePorCoverageIndexed ----
VARIABLE x, w, z

Init == x = 0 /\ w = 0 /\ z = 0

A == (x' = 1 /\ w' = w /\ z' = z) \/ (w' = 1 /\ x' = x /\ z' = z)

B == z' = 1 /\ x' = x /\ w' = w

Next == A \/ B
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        por_enabled: true,
        check_deadlock: false,
        ..Default::default()
    };
    let checker = ParallelChecker::new(&module, &config, 2);
    let ctx = checker
        .prepare_base_ctx()
        .expect("parallel prepare should build EvalCtx");

    let detected = checker.detected_actions_info_for_resolved_next("Next");
    assert_eq!(
        detected.len(),
        2,
        "enumeration decomposition keeps the named A un-split"
    );

    let (por_actions, independence, _visibility) = checker.prepare_por_state(&ctx, detected, &[]);
    assert_eq!(por_actions.len(), 2);

    let matrix =
        independence.expect("POR must stay ON — no fail-closed decomposition-mismatch skip");
    assert_eq!(
        matrix.action_count(),
        2,
        "independence matrix must be indexed by the enumeration decomposition"
    );
    assert_eq!(
        matrix.count_independent_pairs(),
        1,
        "A (union real-writes x and w) and B (writes z) are independent"
    );
}
