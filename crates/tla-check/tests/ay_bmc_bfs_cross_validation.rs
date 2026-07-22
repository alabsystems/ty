// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AY BMC vs BFS cross-validation integration tests.
//!
//! These tests run both the explicit-state BFS checker (`check_module`) and the
//! ay symbolic BMC engine (`check_bmc`) on the same TLA+ specs, verifying that
//! both engines agree on whether a spec is safe or unsafe.
//!
//! Agreement is defined as:
//! - If BFS finds an invariant violation, BMC must find a `Violation`.
//! - If BFS finds `Success` (all states explored, no violation), BMC must find
//!   `BoundReached` (no violation within the bound).
//! - Counterexample depths must be consistent (BMC depth <= BFS trace length).
//!
//! Part of #3744: ay integration test -- verified ay improves TY symbolic
//! backend.

#![cfg(feature = "ay")]

mod common;

use common::parse_module;
use tla_check::{
    bind_constants_from_config, check_bmc, check_module, BmcConfig, BmcResult, BmcValue,
    CheckResult, Config, ConstantValue, EvalCtx,
};

// ---------------------------------------------------------------------------
// Helper: classify BFS result as safe/unsafe for comparison
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Safe,
    Unsafe,
}

fn bfs_verdict(result: &CheckResult) -> Verdict {
    match result {
        CheckResult::Success(_) => Verdict::Safe,
        CheckResult::InvariantViolation { .. } => Verdict::Unsafe,
        CheckResult::Deadlock { .. } => Verdict::Unsafe,
        other => panic!("unexpected BFS result: {other:?}"),
    }
}

fn bmc_verdict(result: &BmcResult) -> Verdict {
    match result {
        BmcResult::Violation { .. } => Verdict::Unsafe,
        // A reachable deadlock state is Unsafe, mirroring explicit-BFS Deadlock.
        BmcResult::Deadlock { .. } => Verdict::Unsafe,
        BmcResult::BoundReached { .. } => Verdict::Safe,
        BmcResult::Unknown { reason, .. } => {
            panic!("BMC returned Unknown (cannot compare): {reason}")
        }
    }
}

/// Run both BFS and BMC on a spec and assert they agree.
///
/// `bmc_depth` must be large enough for BMC to find any violation that BFS
/// finds. For finite-state specs where BFS exhausts the state space within
/// `bmc_depth` steps, this guarantees complete agreement.
fn assert_bfs_bmc_agree(src: &str, bmc_depth: usize) {
    assert_bfs_bmc_agree_with_config(src, bmc_depth, Config::default());
}

fn assert_bfs_bmc_agree_with_config(src: &str, bmc_depth: usize, mut config: Config) {
    config.init = Some("Init".to_string());
    config.next = Some("Next".to_string());
    config.invariants = vec!["Safety".to_string()];

    let module = parse_module(src);
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    bind_constants_from_config(&mut ctx, &config).expect("constants should bind");

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(bmc_depth))
        .expect("BMC should not error");

    let bfs_v = bfs_verdict(&bfs_result);
    let bmc_v = bmc_verdict(&bmc_result);

    assert_eq!(
        bfs_v, bmc_v,
        "BFS and BMC disagree! BFS={bfs_v:?} but BMC={bmc_v:?}\n\
         BFS result: {bfs_result:?}\n\
         BMC result: {bmc_result:?}"
    );
}

// ============================================================================
// Test 1: Simple safe counter -- both engines agree it is safe
// ============================================================================
//
// Counter increments only while count < 3. Safety: count <= 3.
// Finite state space: {0, 1, 2, 3}. Both BFS and BMC(k=10) should find Safe.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_safe_bounded_counter() {
    let src = r#"
---- MODULE SafeBoundedCounter ----
VARIABLE count
Init == count = 0
Next == IF count < 3 THEN count' = count + 1 ELSE count' = count
Safety == count <= 3
====
"#;
    assert_bfs_bmc_agree(src, 10);
}

// ============================================================================
// Test 2: Unsafe counter -- both engines agree on violation
// ============================================================================
//
// Counter increments without bound. Safety: count <= 5.
// BFS finds violation when count reaches 6. BMC finds it at depth 6.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_unsafe_unbounded_counter() {
    let src = r#"
---- MODULE UnsafeBoundedCounter ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 5
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    // Both must find violation
    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    // BMC violation should be at depth 6 (count goes 0,1,2,3,4,5,6)
    if let BmcResult::Violation { depth, trace } = &bmc_result {
        assert_eq!(*depth, 6, "BMC should find violation at depth 6");
        // Trace should start at 0 and end at 6
        assert!(
            matches!(trace[0].assignments.get("count"), Some(BmcValue::Int(0))),
            "trace should start at count=0"
        );
        assert!(
            matches!(
                trace[*depth].assignments.get("count"),
                Some(BmcValue::Int(6))
            ),
            "trace should end at count=6"
        );
    }
}

// ============================================================================
// Soundness guard: inductive-bound injection must NOT clip a real violation
// ============================================================================
//
// `count' = count + 1` is unbounded, so the candidate interval bound derived
// from the literals {0, 5} ([0,5]) is NOT inductive (B /\ Next => B' fails:
// from count=5 you reach count=6 \notin [0,5]). The inductiveness gate must
// therefore REJECT it and assert nothing — leaving the depth-6 violation
// (count: 0,1,2,3,4,5,6 vs Safety count<=5) fully reachable.
//
// If a future regression were to assert a non-inductive bound, this trace
// would be clipped at count=5 and BMC would wrongly report Safe. This test
// is the direct guard on the inductiveness gate.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_inductive_bound_does_not_clip_unsafe_violation() {
    let src = r#"
---- MODULE UnsafeUnboundedGuard ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 5
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // Exercise BOTH BMC paths (per-depth and incremental): the non-inductive
    // candidate must be skipped by each, leaving the violation at depth 6.
    for incremental in [false, true] {
        let bmc_result = check_bmc(
            &module,
            &config,
            &ctx,
            BmcConfig {
                max_depth: 10,
                incremental,
                ..BmcConfig::default()
            },
        )
        .expect("BMC should not error");

        match &bmc_result {
            BmcResult::Violation { depth, .. } => assert_eq!(
                *depth, 6,
                "inductive-bound injection must not clip violation (incremental={incremental})"
            ),
            other => {
                panic!("expected Violation at depth 6 (incremental={incremental}), got {other:?}")
            }
        }
    }
}

// ============================================================================
// Test 3: Init-state violation -- both engines detect depth-0 bug
// ============================================================================

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_init_violation() {
    let src = r#"
---- MODULE InitViolationCross ----
VARIABLE x
Init == x = 100
Next == x' = x
Safety == x <= 50
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(5))
        .expect("BMC should not error");

    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    // BMC should find it at depth 0
    if let BmcResult::Violation { depth, .. } = &bmc_result {
        assert_eq!(*depth, 0, "init violation should be at depth 0");
    }
}

// ============================================================================
// Test 4: Two-variable safe spec with UNCHANGED
// ============================================================================
//
// x increments, y stays at 0. Safety: y = 0.
// Tests that BMC correctly handles UNCHANGED and multi-variable specs.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_two_var_unchanged_safe() {
    let src = r#"
---- MODULE TwoVarUnchangedSafe ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ UNCHANGED y
Safety == y = 0
====
"#;
    assert_bfs_bmc_agree(src, 10);
}

// ============================================================================
// Test 5: Two-variable unsafe spec
// ============================================================================
//
// Both x and y increment. Safety: x + y <= 8.
// Violation at depth 5: x=5, y=5, x+y=10 > 8.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_two_var_sum_unsafe() {
    let src = r#"
---- MODULE TwoVarSumUnsafe ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ y' = y + 1
Safety == x + y <= 8
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    // At depth 5, x=5, y=5, sum=10 > 8
    if let BmcResult::Violation { depth, trace } = &bmc_result {
        assert_eq!(*depth, 5, "violation at x+y > 8 should be at depth 5");
        let last = &trace[*depth];
        let x_val = match last.assignments.get("x") {
            Some(BmcValue::Int(v)) => *v,
            other => panic!("expected Int for x, got {other:?}"),
        };
        let y_val = match last.assignments.get("y") {
            Some(BmcValue::Int(v)) => *v,
            other => panic!("expected Int for y, got {other:?}"),
        };
        assert!(
            x_val + y_val > 8,
            "violation state should have x + y > 8, got x={x_val}, y={y_val}"
        );
    }
}

// ============================================================================
// Test 6: Mutual exclusion -- both processes never in critical section
// ============================================================================
//
// Two processes with a simple turn-based protocol.
// pc1, pc2 in {0, 1} where 1 = critical section.
// Protocol: only enter CS when turn = self.
// Safety: NOT (pc1 = 1 /\ pc2 = 1).

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_mutex_safe() {
    let src = r#"
---- MODULE MutexSafe ----
VARIABLES pc1, pc2, turn
Init == pc1 = 0 /\ pc2 = 0 /\ turn = 1
Next ==
    \/ (pc1 = 0 /\ turn = 1 /\ pc1' = 1 /\ UNCHANGED <<pc2, turn>>)
    \/ (pc1 = 1 /\ pc1' = 0 /\ turn' = 2 /\ UNCHANGED pc2)
    \/ (pc2 = 0 /\ turn = 2 /\ pc2' = 1 /\ UNCHANGED <<pc1, turn>>)
    \/ (pc2 = 1 /\ pc2' = 0 /\ turn' = 1 /\ UNCHANGED pc1)
Safety == ~(pc1 = 1 /\ pc2 = 1)
====
"#;
    assert_bfs_bmc_agree(src, 15);
}

// ============================================================================
// Test 7: Broken mutex -- both engines detect violation
// ============================================================================
//
// Same as above but processes can enter CS without checking turn.
// Safety: NOT (pc1 = 1 /\ pc2 = 1). This should be violated.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_mutex_broken() {
    let src = r#"
---- MODULE MutexBroken ----
VARIABLES pc1, pc2, turn
Init == pc1 = 0 /\ pc2 = 0 /\ turn = 1
Next ==
    \/ (pc1 = 0 /\ pc1' = 1 /\ UNCHANGED <<pc2, turn>>)
    \/ (pc1 = 1 /\ pc1' = 0 /\ turn' = 2 /\ UNCHANGED pc2)
    \/ (pc2 = 0 /\ pc2' = 1 /\ UNCHANGED <<pc1, turn>>)
    \/ (pc2 = 1 /\ pc2' = 0 /\ turn' = 1 /\ UNCHANGED pc1)
Safety == ~(pc1 = 1 /\ pc2 = 1)
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    // Both should find violation (both processes in CS simultaneously)
    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    // Verify BMC counterexample shows both in CS
    if let BmcResult::Violation { depth, trace } = &bmc_result {
        let last = &trace[*depth];
        let pc1 = last.assignments.get("pc1");
        let pc2 = last.assignments.get("pc2");
        assert!(
            matches!((pc1, pc2), (Some(BmcValue::Int(1)), Some(BmcValue::Int(1)))),
            "violation state should have pc1=1 and pc2=1, got pc1={pc1:?}, pc2={pc2:?}"
        );
    }
}

// ============================================================================
// Test 8: Token ring -- safe with N=3
// ============================================================================
//
// A token circulates among 3 nodes. Only the token holder can act.
// Safety: exactly one node holds the token at all times.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_token_ring_safe() {
    let src = r#"
---- MODULE TokenRingSafe ----
VARIABLES t1, t2, t3
Init == t1 = 1 /\ t2 = 0 /\ t3 = 0
Next ==
    \/ (t1 = 1 /\ t1' = 0 /\ t2' = 1 /\ UNCHANGED t3)
    \/ (t2 = 1 /\ t2' = 0 /\ t3' = 1 /\ UNCHANGED t1)
    \/ (t3 = 1 /\ t3' = 0 /\ t1' = 1 /\ UNCHANGED t2)
Safety == t1 + t2 + t3 = 1
====
"#;
    assert_bfs_bmc_agree(src, 15);
}

// ============================================================================
// Test 9: Broken token ring -- token can be duplicated
// ============================================================================
//
// Bug: passing the token does not clear the sender's token.
// Safety: exactly one node holds the token. Should be violated.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_token_ring_broken() {
    let src = r#"
---- MODULE TokenRingBroken ----
VARIABLES t1, t2, t3
Init == t1 = 1 /\ t2 = 0 /\ t3 = 0
Next ==
    \/ (t1 = 1 /\ t2' = 1 /\ UNCHANGED <<t1, t3>>)
    \/ (t2 = 1 /\ t3' = 1 /\ UNCHANGED <<t1, t2>>)
    \/ (t3 = 1 /\ t1' = 1 /\ UNCHANGED <<t2, t3>>)
Safety == t1 + t2 + t3 = 1
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(5))
        .expect("BMC should not error");

    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    // After one step, token is duplicated: t1=1, t2=1
    if let BmcResult::Violation { trace, .. } = &bmc_result {
        let last = trace.last().unwrap();
        let sum: i64 = ["t1", "t2", "t3"]
            .iter()
            .filter_map(|name| match last.assignments.get(*name) {
                Some(BmcValue::Int(v)) => Some(*v),
                _ => None,
            })
            .sum();
        assert!(
            sum != 1,
            "violation state should have token count != 1, got sum={sum}"
        );
    }
}

// ============================================================================
// Test 10: Conditional branching -- IF/THEN/ELSE safe spec
// ============================================================================
//
// x oscillates between 0 and 1. Safety: x \in {0, 1}.
// Tests BMC handling of IF-THEN-ELSE in Next.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_if_then_else_safe() {
    let src = r#"
---- MODULE IfThenElseSafe ----
VARIABLE x
Init == x \in {0, 1}
Next == IF x = 0 THEN x' = 1 ELSE x' = 0
Safety == x >= 0 /\ x <= 1
====
"#;
    assert_bfs_bmc_agree(src, 10);
}

// ============================================================================
// Test 11: Multiple initial states -- both engines explore all inits
// ============================================================================
//
// x starts in {0, 1, 2, 3}. Next: x stays. Safety: x <= 3.
// All initial states satisfy safety. Both should agree: Safe.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_multiple_init_states_safe() {
    let src = r#"
---- MODULE MultiInitSafe ----
VARIABLE x
Init == x \in {0, 1, 2, 3}
Next == x' = x
Safety == x <= 3
====
"#;
    assert_bfs_bmc_agree(src, 5);
}

// ============================================================================
// Test 12: Multiple initial states -- one init violates invariant
// ============================================================================

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_multiple_init_states_one_unsafe() {
    let src = r#"
---- MODULE MultiInitOneUnsafe ----
VARIABLE x
Init == x \in {0, 1, 2, 10}
Next == x' = x
Safety == x <= 5
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(3))
        .expect("BMC should not error");

    // Both should find the init-state violation (x=10 > 5)
    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    if let BmcResult::Violation { depth, .. } = &bmc_result {
        assert_eq!(*depth, 0, "init violation should be found at depth 0");
    }
}

// ============================================================================
// Test 13: Disjunctive Next -- multiple enabled actions
// ============================================================================
//
// x can increase by 1 or decrease by 1 (but not below 0).
// Safety: x <= 4. Starting from 0, can reach 4 at depth 4 but never 5
// since decrease is also always available. BFS exhausts small space.
// Actually x CAN reach 5 via 0->1->2->3->4->5 in 5 steps.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_disjunctive_next_unsafe() {
    let src = r#"
---- MODULE DisjunctiveNextUnsafe ----
VARIABLE x
Init == x = 0
Next ==
    \/ x' = x + 1
    \/ (x > 0 /\ x' = x - 1)
Safety == x <= 4
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    // Both should find violation (x can reach 5 via 0->1->2->3->4->5)
    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    if let BmcResult::Violation { depth, .. } = &bmc_result {
        assert!(
            *depth <= 5,
            "BMC should find violation at depth <= 5, got {depth}"
        );
    }
}

// ============================================================================
// Test 14: Config constants -- BMC respects CONSTANT bindings
// ============================================================================
//
// Uses CONSTANT N to parameterize the spec. With N=3, spec is safe.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_config_constants_safe() {
    let src = r#"
---- MODULE ConfigConstSafe ----
CONSTANT N
VARIABLE x
Init == x \in 0..N
Next == x' = x
Safety == x <= N
====
"#;
    let mut config = Config::default();
    config
        .constants
        .insert("N".to_string(), ConstantValue::Value("3".to_string()));
    assert_bfs_bmc_agree_with_config(src, 5, config);
}

// ============================================================================
// Test 15: Config constants -- BMC detects violation with parameterized bound
// ============================================================================

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_config_constants_unsafe() {
    let src = r#"
---- MODULE ConfigConstUnsafe ----
CONSTANT Limit
VARIABLE x
Init == x = 0
Next == x' = x + 1
Safety == x <= Limit
====
"#;
    let module = parse_module(src);
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };
    config
        .constants
        .insert("Limit".to_string(), ConstantValue::Value("3".to_string()));

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    bind_constants_from_config(&mut ctx, &config).expect("constants should bind");

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    // Violation at depth 4: count goes 0,1,2,3,4 and 4 > Limit=3
    if let BmcResult::Violation { depth, .. } = &bmc_result {
        assert_eq!(*depth, 4, "violation should be at depth 4 (x=4 > Limit=3)");
    }
}

// ============================================================================
// Test 16: Operator definitions -- BMC expands user operators
// ============================================================================
//
// Tests that BMC correctly expands operator definitions in Init/Next/Safety.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_operator_expansion() {
    let src = r#"
---- MODULE OperatorExpansionCross ----
VARIABLE count
Inc == count' = count + 1
Init == count = 0
Next == count < 3 /\ Inc
Safety == count <= 3
====
"#;
    assert_bfs_bmc_agree(src, 8);
}

// ============================================================================
// Test 17: Incremental vs per-depth BMC agreement
// ============================================================================
//
// Runs both incremental and per-depth BMC modes on the same unsafe spec and
// verifies they find the violation at the same depth.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_incremental_vs_per_depth_agree() {
    let src = r#"
---- MODULE IncrPerDepthAgree ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Safety == x <= 7
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let per_depth = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(15))
        .expect("per-depth BMC should succeed");

    let incremental = check_bmc(
        &module,
        &config,
        &ctx,
        BmcConfig {
            max_depth: 15,
            incremental: true,
            ..BmcConfig::default()
        },
    )
    .expect("incremental BMC should succeed");

    match (&per_depth, &incremental) {
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
            assert_eq!(d1, d2, "per-depth and incremental must find same depth");
            assert_eq!(t1.len(), t2.len(), "trace lengths must match");
        }
        _ => panic!(
            "both modes should find Violation, got per_depth={per_depth:?}, incr={incremental:?}"
        ),
    }
}

// ============================================================================
// Test 18: Three-variable pipeline -- data flows through stages
// ============================================================================
//
// a -> b -> c pipeline with one-step delay.
// Safety: c >= 0 (always true since a starts at 0 and increments).

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_pipeline_safe() {
    let src = r#"
---- MODULE PipelineSafe ----
VARIABLES a, b, c
Init == a = 0 /\ b = 0 /\ c = 0
Next == a' = a + 1 /\ b' = a /\ c' = b
Safety == c >= 0
====
"#;
    assert_bfs_bmc_agree(src, 10);
}

// ============================================================================
// Test 19: Swap spec -- x and y swap each step
// ============================================================================
//
// Init: x=0, y=1. Next: swap. Safety: x >= 0 /\ y >= 0.
// Finite 2-state space: {(0,1), (1,0)}. Always safe.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_swap_safe() {
    let src = r#"
---- MODULE SwapSafe ----
VARIABLES x, y
Init == x = 0 /\ y = 1
Next == x' = y /\ y' = x
Safety == x >= 0 /\ y >= 0
====
"#;
    assert_bfs_bmc_agree(src, 10);
}

// ============================================================================
// Test 20: BFS and BMC BOTH detect deadlock (now in agreement)
// ============================================================================
//
// `Next == x < 3 /\ x' = x + 1` becomes disabled once x reaches 3 (the guard
// `x < 3` fails, so no successor exists). Both engines must report Unsafe:
//   - BFS reports `Deadlock { .. }` (no enabled Next at x=3).
//   - BMC NOW detects the reachable deadlock too, via SOUND concrete-state
//     enumeration (Fix A): it enumerates the reachable frontier state x=3 and
//     proves `EXISTS x': Next(x=3, x')` is UNSAT (no successor), i.e.
//     `~Enabled(Next)(x=3)`. This is the lynchpin soundness encoding — the
//     successor query fixes the source state CONCRETELY, so the existential is
//     answerable in QF_LIA and its UNSAT genuinely certifies deadlock.
//
// This records CORRECTED behavior: BMC formerly stopped at invariant safety
// (`x <= 10` never violated => BoundReached). It now agrees with BFS that a
// reachable deadlock makes the spec Unsafe.

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_deadlock_spec_complementary() {
    let src = r#"
---- MODULE DeadlockSpec ----
VARIABLE x
Init == x = 0
Next == x < 3 /\ x' = x + 1
Safety == x <= 10
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    // BFS detects deadlock at x=3 (no Next enabled).
    assert!(
        matches!(bfs_result, CheckResult::Deadlock { .. }),
        "BFS should detect deadlock, got {bfs_result:?}"
    );

    // BMC now ALSO detects the reachable deadlock (Fix A). The engines agree:
    // both report Unsafe.
    assert!(
        matches!(bmc_result, BmcResult::Deadlock { .. }),
        "BMC should now detect the reachable deadlock, got {bmc_result:?}"
    );
    assert_eq!(
        bmc_verdict(&bmc_result),
        Verdict::Unsafe,
        "BMC deadlock => Unsafe, agreeing with BFS"
    );

    // The deadlocked frontier state is x=3 (reached after 3 increments).
    if let BmcResult::Deadlock { depth, trace } = &bmc_result {
        assert_eq!(*depth, 3, "deadlock reached at depth 3 (x: 0,1,2,3)");
        assert!(
            matches!(trace[*depth].assignments.get("x"), Some(BmcValue::Int(3))),
            "deadlocked state should have x=3, got {:?}",
            trace[*depth].assignments.get("x")
        );
    }
}

// ============================================================================
// Adversarial deadlock soundness guards (Fix A)
// ============================================================================
//
// These four tests pin the SOUNDNESS of symbolic deadlock detection: it must
// fire ONLY on genuinely reachable, successor-free states, and must NEVER
// false-positive on a live (non-total but enabled) Next.

// Guard 1: a non-total, always-ENABLED Next must NOT be flagged deadlock.
//
// `Next == IF x = 0 THEN x' = 1 ELSE x' = 0` is non-total (for a given x only
// ONE successor value satisfies it, so `~Next(x, ghost)` is SAT for some ghost)
// yet ALWAYS enabled (every reachable x has a successor). The naive
// `EXISTS s': ~Next` encoding would mislabel this deadlocked; the sound
// concrete-state encoding finds a successor for every reachable x. The state
// space is finite ({0,1}), so BFS terminates Safe and BMC must NOT report a
// (false) deadlock. Both engines agree: Safe.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_deadlock_unreachable_stays_safe() {
    let src = r#"
---- MODULE DeadlockUnreachableSafe ----
VARIABLE x
Init == x = 0
Next == IF x = 0 THEN x' = 1 ELSE x' = 0
Safety == x <= 1000
====
"#;
    assert_bfs_bmc_agree(src, 5);
}

// Guard 2: an intended terminal that SELF-LOOPS must NOT be flagged deadlock.
//
// At count=3 the ELSE branch fires `count' = count` (a self-loop), so the
// state HAS a successor — it is not deadlocked. BMC must report BoundReached,
// not Deadlock. This traps a probe that confuses "guard failed" with "no
// successor".
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_deadlock_self_loop_stays_safe() {
    let src = r#"
---- MODULE DeadlockSelfLoopSafe ----
VARIABLE count
Init == count = 0
Next == IF count < 3 THEN count' = count + 1 ELSE count' = count
Safety == count <= 3
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    assert!(
        matches!(bmc_result, BmcResult::BoundReached { .. }),
        "self-looping terminal count=3 is NOT a deadlock; expected BoundReached, got {bmc_result:?}"
    );
}

// Guard 3: deadlock at depth 0 (the Init state itself is stuck).
//
// Init x=5 fails the guard `x < 3`, so Init has no successor. Both engines must
// report Unsafe with the deadlock detected at depth 0.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_deadlock_at_depth0_unsafe() {
    let src = r#"
---- MODULE DeadlockAtDepth0 ----
VARIABLE x
Init == x = 5
Next == x < 3 /\ x' = x + 1
Safety == x <= 10
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(5))
        .expect("BMC should not error");

    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    if let BmcResult::Deadlock { depth, .. } = &bmc_result {
        assert_eq!(*depth, 0, "Init x=5 is stuck => deadlock at depth 0");
    } else {
        panic!("expected BMC Deadlock at depth 0, got {bmc_result:?}");
    }
}

// Guard 4: deadlock with an UNCHANGED variable in the transition.
//
// `Next == x < 2 /\ x' = x + 1 /\ UNCHANGED y` becomes disabled at x=2. The
// reachable state (x=2, y=0) has no successor. This guards that the successor
// test correctly accounts for UNCHANGED-variable framing when computing
// Enabled(Next). Both engines must report Unsafe.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_deadlock_unchanged_guard_unsafe() {
    let src = r#"
---- MODULE DeadlockUnchangedGuard ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x < 2 /\ x' = x + 1 /\ UNCHANGED y
Safety == y = 0
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);

    if let BmcResult::Deadlock { depth, trace } = &bmc_result {
        assert_eq!(*depth, 2, "deadlock reached at x=2 (depth 2)");
        assert!(
            matches!(trace[*depth].assignments.get("x"), Some(BmcValue::Int(2))),
            "deadlocked state should have x=2"
        );
    } else {
        panic!("expected BMC Deadlock, got {bmc_result:?}");
    }
}

// ============================================================================
// FIX B: SOUND inductive infinite-state safety certificate (adversarial tests)
// ============================================================================
//
// These tests pin the FIX-B certificate: an INFINITE-state spec that is
// inductively safe AND deadlock-free must TERMINATE Safe (instead of hanging
// in unbounded BFS), while a spec that is inductively safe but DEADLOCKS — or
// a finite unsafe accumulating spec — must NOT be falsely certified Safe.

// FIX B / Test 1: the two BFS-hangers must TERMINATE Safe via the certificate.
//
// Both have an UNBOUNDED reachable state space (x / a accumulate without bound)
// so explicit BFS never terminates. The certificate proves them safe + deadlock-
// free (no guards => Enabled==TRUE) and returns Safe; BMC reaches the bound
// (BoundReached => Safe). Both run inside `assert_bfs_bmc_agree`, which would
// panic if BFS or BMC disagreed or skipped. The ntest timeout guards against a
// regression that re-introduces the hang.
//
//   two_var_unchanged_safe: x'=x+1 /\ UNCHANGED y; Safety y=0 (1-inductive).
//   pipeline_safe:          a'=a+1 /\ b'=a /\ c'=b; Safety c>=0 (needs
//                           strengthening to a>=0 /\ b>=0 /\ c>=0, which IS
//                           inductive). Both have NO guards => deadlock-free.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_inductive_safe_unbounded_terminates() {
    let two_var = r#"
---- MODULE TwoVarUnchangedTerminates ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ UNCHANGED y
Safety == y = 0
====
"#;
    assert_bfs_bmc_agree(two_var, 10);

    let pipeline = r#"
---- MODULE PipelineTerminates ----
VARIABLES a, b, c
Init == a = 0 /\ b = 0 /\ c = 0
Next == a' = a + 1 /\ b' = a /\ c' = b
Safety == c >= 0
====
"#;
    assert_bfs_bmc_agree(pipeline, 10);
}

// FIX B / Test 2: inductively-safe-but-DEADLOCKING spec must stay Unsafe.
//
// `Next == count < 3 /\ count' = count + 1`. Safety `count <= 3` IS 1-inductive,
// but the guard `count < 3` becomes disabled at count=3 => the reachable state
// count=3 DEADLOCKS. The certificate MUST NOT fire Safe: its trigger requires an
// UNGUARDED total Next (Enabled==TRUE), and this Next carries the guard
// `count < 3`, so `analyze_deadlock_freedom` returns Decomposed with a NON-empty
// guard list => the certificate declines at the trigger and falls through. BFS
// then finds the deadlock => Unsafe; BMC (Fix A) detects it too.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_inductive_safe_but_deadlocks_is_unsafe() {
    let src = r#"
---- MODULE InductiveSafeButDeadlocks ----
VARIABLE count
Init == count = 0
Next == count < 3 /\ count' = count + 1
Safety == count <= 3
====
"#;
    let module = parse_module(src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    // The certificate must NOT have falsely certified Safe: BFS reports Deadlock.
    assert!(
        matches!(bfs_result, CheckResult::Deadlock { .. }),
        "certificate must NOT fire Safe on a deadlocking spec; expected BFS Deadlock, got {bfs_result:?}"
    );
    assert_eq!(bfs_verdict(&bfs_result), Verdict::Unsafe);
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);
}

// FIX B / Test 3: a true-but-not-1-inductive safety that becomes Safe ONLY via
// strengthening (or BFS), plus a finite UNSAFE accumulating spec that must stay
// Unsafe (never certificate-Safe).
//
// Part A: Init a=0/\b=0; Next a'=a+1/\b'=a; Safety b>=0. `b>=0` is NOT
// 1-inductive alone (consecution: b'=a, and a>=0 is not in the hypothesis), but
// strengthening to a>=0/\b>=0 IS inductive. No guards => deadlock-free. Must end
// Safe and NOT hang.
//
// Part B: count'=count+1; Safety count<=5 — finite-bug accumulating spec. The
// candidate bound [0,5] is NOT inductive (from 5 you reach 6 \notin [0,5]), so
// the certificate falls through and BFS finds the depth-6 violation. Must stay
// Unsafe — never certificate-Safe.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn ay_cross_inductive_true_but_not_inductive_falls_through() {
    // Part A: needs strengthening; ends Safe (BFS+BMC agree) without hanging.
    let strengthen = r#"
---- MODULE NeedsStrengthening ----
VARIABLES a, b
Init == a = 0 /\ b = 0
Next == a' = a + 1 /\ b' = a
Safety == b >= 0
====
"#;
    assert_bfs_bmc_agree(strengthen, 10);

    // Part B: accumulating but UNSAFE — certificate must fall through, leaving
    // the violation reachable. Both engines report Unsafe.
    let unsafe_src = r#"
---- MODULE FiniteUnsafeAccumulating ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
Safety == count <= 5
====
"#;
    let module = parse_module(unsafe_src);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Safety".to_string()],
        ..Default::default()
    };

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let bfs_result = check_module(&module, &config);
    let bmc_result = check_bmc(&module, &config, &ctx, BmcConfig::with_max_depth(10))
        .expect("BMC should not error");

    assert_eq!(
        bfs_verdict(&bfs_result),
        Verdict::Unsafe,
        "certificate must NOT fire Safe on a finite unsafe accumulating spec; got {bfs_result:?}"
    );
    assert_eq!(bmc_verdict(&bmc_result), Verdict::Unsafe);
}
