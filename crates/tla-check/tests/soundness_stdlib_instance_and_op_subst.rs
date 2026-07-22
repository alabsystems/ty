// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness-triage regressions (2026-07 corpus sweep, ty-vs-TLC verdict
//! disagreements):
//!
//! 1. **MCEWD687a** (ewd687a): a PROPERTY containing a LET-local
//!    `G == INSTANCE Graphs` (builtin stdlib module) hard-errored during
//!    liveness conversion ("Unsupported temporal expression") because
//!    (a) the level classifier's `level_module_ref_with_ctx` could not resolve
//!        stdlib-module operators (never in `instance_ops`) and conservatively
//!        classified the leaf as Temporal, and
//!    (b) the evaluator's `eval_module_ref` had no builtin fallback for
//!        instanced stdlib modules ("Undefined operator: G!Transpose").
//!    TLC checks the property fine (Java module overrides).
//!
//! 2. **BlockDagTest** (dag-consensus): an operator-valued INSTANCE
//!    substitution (`CONSTANTS Leader(_)` instantiated `WITH Leader <- Leader`)
//!    failed with "Arity mismatch: <closure#..> expects 0 arguments, got 1".
//!    Unnamed-INSTANCE materialization LET-wraps each substitution as a
//!    zero-arg def (`__ty_subst_Leader == Leader`); applying that zero-arg
//!    thunk to arguments must resolve the thunk body as an operator in its
//!    captured context (TLC parity), not fail on closure arity.
//!
//! Each fix has a "still falsifiable" companion test so the pass verdicts are
//! demonstrably not vacuous.

mod common;

use tla_check::{CheckResult, Config, ModelChecker};
use tla_core::FileId;
use tla_eval::clear_for_test_reset;

// ---------------------------------------------------------------------------
// Case 1: LET-local INSTANCE of a builtin stdlib module (Graphs) in a PROPERTY
// ---------------------------------------------------------------------------

/// Mirrors EWD687a's TreeWithRoot: a `[]`-property whose body resolves
/// operators through a LET-local `G == INSTANCE Graphs`.
fn graphs_property_module(property_body: &str) -> tla_core::ast::Module {
    let src = format!(
        r#"
---- MODULE LetInstGraphs ----
EXTENDS Integers

VARIABLE x

Init == x = 0
Next == x' = (x + 1) % 3
vars == << x >>

Prop ==
    LET E == IF x = 1 THEN {{<<1, 2>>}} ELSE {{}}
        N == {{e[1] : e \in E}} \cup {{e[2] : e \in E}}
        G == INSTANCE Graphs
        O == G!Transpose([edge |-> E, node |-> N])
    IN []({property_body})
====
"#
    );
    common::parse_module_strict_with_id(&src, FileId(0))
}

fn graphs_property_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Prop".to_string()],
        check_deadlock: false,
        ..Default::default()
    }
}

/// The EWD687a/TreeWithRoot shape must check successfully (TLC parity),
/// not hard-error in liveness conversion or evaluation.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn let_local_instance_graphs_property_checks_clean() {
    clear_for_test_reset();
    let module = graphs_property_module("O.edge # {} => G!IsDirectedGraph(O)");
    let config = graphs_property_config();
    let mut checker = ModelChecker::new_with_extends(&module, &[], &config);
    match checker.check() {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 3,
                "expected 3 states for mod-3 counter, got {}",
                stats.states_found
            );
        }
        other => panic!(
            "LET-local INSTANCE Graphs property must check clean \
             (was: liveness conversion hard error / G!Transpose undefined), got {other:?}"
        ),
    }
}

/// Companion falsifiability check: the same property shape with a body that is
/// false at x=1 must be reported violated — proving the property is genuinely
/// evaluated, not skipped.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn let_local_instance_graphs_property_still_falsifiable() {
    clear_for_test_reset();
    let module = graphs_property_module("O.edge = {}");
    let config = graphs_property_config();
    let mut checker = ModelChecker::new_with_extends(&module, &[], &config);
    match checker.check() {
        CheckResult::LivenessViolation { .. } | CheckResult::PropertyViolation { .. } => {}
        other => panic!(
            "violated []-property through LET-local INSTANCE Graphs must be detected, \
             got {other:?}"
        ),
    }
}

// ---------------------------------------------------------------------------
// Case 2: operator-valued INSTANCE WITH substitution (BlockDagTest shape)
// ---------------------------------------------------------------------------

/// Inner module with an operator-valued constant, like BlockDag's
/// `CONSTANTS Leader(_)`.
fn op_const_module() -> tla_core::ast::Module {
    common::parse_module_strict_with_id(
        r#"
---- MODULE OpParamConst ----
EXTENDS Integers
CONSTANTS Foo(_)
Baz(r) == IF r > 0 THEN Foo(r) + 1 ELSE 0
ASSUME Baz(1) = 3
====
"#,
        FileId(1),
    )
}

fn op_subst_host_module(invariant_body: &str) -> tla_core::ast::Module {
    let src = format!(
        r#"
---- MODULE OpParamHost ----
EXTENDS Integers

VARIABLE x

Foo(v) == v + 1

INSTANCE OpParamConst WITH Foo <- Foo

Init == x = 0
Next == x' = (x + 1) % 3

Inv == {invariant_body}
====
"#
    );
    common::parse_module_strict_with_id(&src, FileId(0))
}

fn op_subst_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        check_deadlock: false,
        ..Default::default()
    }
}

/// `Baz(x + 1)` routes through the materialized zero-arg LET def
/// `__ty_subst_Foo == Foo` — applying it with 1 argument must resolve the
/// outer unary operator, not fail with a closure arity mismatch. The inner
/// module's ASSUME (`Baz(1) = 3`) exercises the same path at prepare time.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn operator_valued_instance_substitution_applies() {
    clear_for_test_reset();
    let inner = op_const_module();
    // Baz(x+1) = Foo(x+1) + 1 = x + 3
    let host = op_subst_host_module("Baz(x + 1) = x + 3");
    let config = op_subst_config();
    let mut checker = ModelChecker::new_with_extends(&host, &[&inner], &config);
    match checker.check() {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 3,
                "expected 3 states for mod-3 counter, got {}",
                stats.states_found
            );
        }
        other => panic!(
            "operator-valued INSTANCE WITH substitution must evaluate \
             (was: Arity mismatch: <closure#..> expects 0 arguments, got 1), got {other:?}"
        ),
    }
}

/// Companion falsifiability check: a wrong invariant through the same
/// operator-valued substitution must still be reported violated.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn operator_valued_instance_substitution_still_falsifiable() {
    clear_for_test_reset();
    let inner = op_const_module();
    let host = op_subst_host_module("Baz(x + 1) = x + 4");
    let config = op_subst_config();
    let mut checker = ModelChecker::new_with_extends(&host, &[&inner], &config);
    match checker.check() {
        CheckResult::InvariantViolation { invariant, .. } => {
            assert_eq!(invariant, "Inv");
        }
        other => panic!(
            "violated invariant through operator-valued substitution must be detected, \
             got {other:?}"
        ),
    }
}

/// Companion falsifiability check for ASSUME: a false assumption evaluated
/// through the operator-valued substitution must abort with an error (this is
/// exactly the BlockDagTest ASSUME-only unit-test module shape).
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn operator_valued_substitution_false_assume_is_detected() {
    clear_for_test_reset();
    let inner = common::parse_module_strict_with_id(
        r#"
---- MODULE OpParamConstBad ----
EXTENDS Integers
CONSTANTS Foo(_)
Baz(r) == IF r > 0 THEN Foo(r) + 1 ELSE 0
ASSUME Baz(1) = 999
====
"#,
        FileId(1),
    );
    let host = common::parse_module_strict_with_id(
        r#"
---- MODULE OpParamHostBad ----
EXTENDS Integers

VARIABLE x

Foo(v) == v + 1

INSTANCE OpParamConstBad WITH Foo <- Foo

Init == x = 0
Next == x' = (x + 1) % 3
====
"#,
        FileId(0),
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        check_deadlock: false,
        ..Default::default()
    };
    let mut checker = ModelChecker::new_with_extends(&host, &[&inner], &config);
    match checker.check() {
        CheckResult::Error { error, .. } => {
            let msg = format!("{error:?}");
            assert!(
                msg.to_lowercase().contains("assum"),
                "expected an assumption-false error, got {msg}"
            );
        }
        other => panic!("false ASSUME must abort checking with an error, got {other:?}"),
    }
}
