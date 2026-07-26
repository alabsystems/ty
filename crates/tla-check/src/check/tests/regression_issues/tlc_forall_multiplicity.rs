// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::EvalError;
use tla_core::{lower, parse_to_syntax_tree, FileId};

/// TLC expands bounded FORALL bodies into its action-item continuation. With
/// three committed resource managers, each of the three universal instances
/// has three successful EXISTS witnesses: 3 * 3 * 3 = 27 raw generations of
/// the same endpoint. This reduced model is taken from 2PCwithBTM's canCommit
/// guard, where collapsing the FORALL Boolean caused a 550-generation deficit.
const MULTIPLICITY_SPEC: &str = r#"
---- MODULE TlcForallMultiplicity ----
VARIABLES x, f

S == {"a", "b", "c"}

Guard ==
    \A i \in S :
        f[i] \in {"prepared"}
        \/ \E j \in S : f[j] \in {"committed"}

Init ==
    /\ x = 0
    /\ f = [i \in S |-> "committed"]

Next ==
    /\ x = 0
    /\ Guard
    /\ x' = 1
    /\ UNCHANGED f
====
"#;

#[allow(clippy::option_option)]
fn check(src: &str, successor_cap: Option<Option<usize>>) -> CheckResult {
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    assert!(
        lower_result.errors.is_empty(),
        "lowering errors: {:?}",
        lower_result.errors
    );
    let module = lower_result.module.expect("module should lower");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        per_state_successor_cap: successor_cap,
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.check()
}

fn assert_raw_successors(src: &str, expected: usize) {
    match check(src, None) {
        CheckResult::Success(stats) => {
            assert_eq!(stats.raw_initial_states_generated, 1);
            assert_eq!(stats.raw_successors_generated, expected);
        }
        other => panic!("expected success, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn action_free_forall_preserves_tlc_raw_proof_multiplicity() {
    match check(MULTIPLICITY_SPEC, None) {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 2);
            assert_eq!(stats.raw_initial_states_generated, 1);
            assert_eq!(
                stats.raw_successors_generated, 27,
                "TLC generates one proof path for every EXISTS witness in every FORALL instance"
            );
            assert_eq!(stats.states_generated(), 28);
        }
        other => panic!("expected success, got {other:?}"),
    }
}

#[test]
fn forall_before_assignment_expands_but_after_assignment_collapses() {
    let before = r#"
---- MODULE TlcForallBeforeAssignment ----
VARIABLE x
Init == x = 0
Next ==
    /\ x = 0
    /\ \A i \in {1, 2} : TRUE \/ TRUE
    /\ x' = 1
====
"#;
    let after = r#"
---- MODULE TlcForallAfterAssignment ----
VARIABLE x
Init == x = 0
Next ==
    /\ x = 0
    /\ x' = 1
    /\ \A i \in {1, 2} : TRUE \/ TRUE
====
"#;

    assert_raw_successors(before, 4);
    assert_raw_successors(after, 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn forall_large_domain_uses_stack_safe_proof_tail() {
    let src = r#"
---- MODULE TlcForallStackSafe ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next ==
    /\ x = 0
    /\ \A i \in 1..2048 : TRUE
    /\ x' = 1
====
"#;
    assert_raw_successors(src, 1);
}

/// TLC proof DFS must still honor the batch sink's per-state materialization
/// cap. The ninth proof path makes the sink signal early stop; enumeration must
/// unwind and fail closed instead of returning eight states.
#[test]
fn forall_proof_dfs_honors_batch_cap_and_early_stop() {
    use crate::enumerate::enumerate_successors_array_as_diffs_body_with_cap;
    use crate::eval::EvalCtx;
    use crate::state::{ArrayState, State};
    use std::sync::Arc;

    let src = r#"
---- MODULE TlcForallMultiplicityCap ----
VARIABLE x
S == {1, 2, 3}
Guard == \A i \in S : x = 0 \/ \E j \in S : x = 0
Init == x = 0
Next == /\ Guard
        /\ x' = 1
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    assert!(lower_result.errors.is_empty());
    let module = lower_result.module.expect("module should lower");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    let x: Arc<str> = Arc::from("x");
    ctx.register_var(Arc::clone(&x));
    ctx.resolve_state_vars_in_loaded_ops();
    let next = Arc::clone(ctx.get_op("Next").expect("Next definition"));
    let vars = vec![x];
    let registry = ctx.var_registry().clone();
    let current =
        ArrayState::from_state(&State::from_pairs([("x", crate::Value::int(0))]), &registry);

    let result = enumerate_successors_array_as_diffs_body_with_cap(
        &mut ctx, &next.body, &current, &vars, None, 8,
    );
    assert!(
        matches!(result, Err(EvalError::SetTooLarge { .. })),
        "the explicit eight-successor batch cap must fail closed, got {result:?}"
    );

    let exact = enumerate_successors_array_as_diffs_body_with_cap(
        &mut ctx, &next.body, &current, &vars, None, 64,
    )
    .expect("the exact cap must succeed")
    .expect("batch enumeration returns a successor vector");
    assert_eq!(
        exact.len(),
        64,
        "a cap equal to the TLC proof-path count is inclusive"
    );
}

/// Streaming sinks can stop after the first proof path without first computing
/// or overflowing the full 2^64 multiplicity.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn forall_proof_dfs_honors_streaming_break_without_counting_all_paths() {
    use crate::enumerate::{enumerate_successors_array_as_diffs_into, ClosureSink};
    use crate::eval::EvalCtx;
    use crate::state::{ArrayState, State};
    use std::ops::ControlFlow;
    use std::sync::Arc;

    let src = r#"
---- MODULE TlcForallStreamingBreak ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next ==
    /\ x = 0
    /\ \A i \in 1..64 : TRUE \/ TRUE
    /\ x' = 1
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    assert!(lower_result.errors.is_empty());
    let module = lower_result.module.expect("module should lower");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    let x: Arc<str> = Arc::from("x");
    ctx.register_var(Arc::clone(&x));
    ctx.resolve_state_vars_in_loaded_ops();
    let next = Arc::clone(ctx.get_op("Next").expect("Next definition"));
    let vars = vec![x];
    let registry = ctx.var_registry().clone();
    let current =
        ArrayState::from_state(&State::from_pairs([("x", crate::Value::int(0))]), &registry);

    let mut seen = 0usize;
    let mut sink = ClosureSink::new(|_| {
        seen += 1;
        ControlFlow::Break(())
    });
    enumerate_successors_array_as_diffs_into(&mut ctx, &next, &current, &vars, &mut sink, None)
        .expect("streaming proof enumeration should stop cleanly");
    drop(sink);
    assert_eq!(seen, 1);
}

const AND_GUARD_CHILD_ENV: &str = "TY_TEST_FORALL_AND_GUARD_CHILD";
const AND_GUARD_CHILD_SENTINEL: &str = "TY_FORALL_AND_GUARD_CHILD_OK";

/// The optional whole-AND guard precheck is reject-only. Its old prefix skip
/// confused "defer" with "proven true": FALSE \/ FALSE fabricated a successor,
/// and true OR/EXISTS/FORALL prefixes lost TLC raw proof paths.
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn and_guard_precheck_preserves_false_and_raw_proof_paths() {
    if std::env::var_os(AND_GUARD_CHILD_ENV).is_some() {
        let false_or = r#"
---- MODULE GuardPrecheckFalseOr ----
VARIABLE x
Init == x = 0
Next == /\ (FALSE \/ FALSE)
        /\ x' = 1
====
"#;
        let true_or = r#"
---- MODULE GuardPrecheckTrueOr ----
VARIABLE x
Init == x = 0
Next == /\ x = 0
        /\ (TRUE \/ TRUE)
        /\ x' = 1
====
"#;
        let exists = r#"
---- MODULE GuardPrecheckExists ----
VARIABLE x
Init == x = 0
Next == /\ x = 0
        /\ \E i \in {1, 2} : TRUE
        /\ x' = 1
====
"#;
        let forall = r#"
---- MODULE GuardPrecheckForall ----
VARIABLE x
Init == x = 0
Next == /\ x = 0
        /\ \A i \in {1, 2} : TRUE \/ TRUE
        /\ x' = 1
====
"#;
        let forall_operator = r#"
---- MODULE GuardPrecheckForallOperator ----
VARIABLE x
Guard == \A i \in {1, 2} : TRUE \/ TRUE
Init == x = 0
Next == /\ x = 0
        /\ Guard
        /\ x' = 1
====
"#;

        assert_raw_successors(false_or, 0);
        assert_raw_successors(true_or, 2);
        assert_raw_successors(exists, 2);
        assert_raw_successors(forall, 4);
        assert_raw_successors(forall_operator, 4);
        eprintln!("{AND_GUARD_CHILD_SENTINEL}");
        return;
    }

    let exe = std::env::current_exe().expect("current test executable");
    let child_module = module_path!()
        .split_once("::")
        .map_or(module_path!(), |(_, path)| path);
    let child_test =
        format!("{child_module}::and_guard_precheck_preserves_false_and_raw_proof_paths");
    let output = std::process::Command::new(exe)
        .env(AND_GUARD_CHILD_ENV, "1")
        .env("TY_AND_GUARD_PRECHECK", "1")
        .arg("--exact")
        .arg(child_test)
        .arg("--nocapture")
        .output()
        .expect("failed to spawn guard-precheck child");

    assert!(
        output.status.success(),
        "guard-precheck child failed (status={:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(AND_GUARD_CHILD_SENTINEL),
        "guard-precheck child did not run the expected test:\n{stderr}"
    );
}
