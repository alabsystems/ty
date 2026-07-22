// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Regression: a refinement / implied-action violation must be REPORTED as an
//! action-level `PropertyViolation`, never panic.
//!
//! The interpreter full-state successor loop batches admitted (new, unseen)
//! successors and flushes the batch lazily. `handle_implied_action_outcome`
//! consumes the per-iteration `BfsIterState` via `return_to` on a terminal
//! violation. The former code then called `flush_admission_batch`, which
//! re-borrows `iter_state.array()` AFTER it had been returned — panicking with
//! "BfsIterState: array already returned" on the very first
//! refinement/implied-action violation whenever an earlier sibling successor of
//! the same parent had already been batched.
//!
//! This is the exact shape of `EWD998PCal` with an abstract action removed (so a
//! concrete step has no matching abstract image): the abstract-derived
//! `pending`/`token` VARIABLES map to concrete zero-arg operators, which keeps
//! the implied action on the interpreter `eval_implied_actions` path (not the
//! native compiled-BFS path, which reports violations through a separate
//! mechanism that never had the bug).

mod common;

use tla_check::{CheckResult, Config, ModelChecker};
use tla_core::FileId;

/// Abstract module with a SINGLE real action (`GoodMsg`). There is deliberately
/// NO abstract image for the concrete `BadStep`, so a concrete `BadStep`
/// transition is neither an abstract step nor an abstract stutter — the implied
/// action `[][Next]_vars` is violated. `pending`/`token` are VARIABLES here;
/// in the concrete module they are supplied by zero-arg operators.
fn abstract_broken_module() -> tla_core::ast::Module {
    common::parse_module_strict_with_id(
        r#"
---- MODULE AbstractBroken ----
EXTENDS Integers

Node == 0 .. 2

VARIABLE active, counter, pending, token, step

Init ==
    /\ active = [i \in Node |-> i = 2]
    /\ counter = [i \in Node |-> IF i = 2 THEN 1 ELSE 0]
    /\ pending = [i \in Node |-> IF i = 0 THEN 1 ELSE 0]
    /\ token = [pos |-> 0, q |-> 0, color |-> "black"]
    /\ step = 0

GoodMsg(i) ==
    /\ step = 0
    /\ counter' = [counter EXCEPT ![i] = @ + 1]
    /\ pending' = [pending EXCEPT ![(i + 2) % 3] = @ + 1]
    /\ UNCHANGED <<active, token>>
    /\ step' = 1

\* NOTE: no abstract image for the concrete `BadStep`.
Next == GoodMsg(2)

vars == <<active, counter, pending, token, step>>

Spec == Init /\ [][Next]_vars
====
"#,
        FileId(11),
    )
}

fn property_config(property: &str) -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec![property.to_string()],
        check_deadlock: false,
        ..Default::default()
    }
}

/// A concrete step whose abstract image is `GoodMsg(2)` is enumerated FIRST, so
/// it is admitted into the (lazily flushed) admission batch. The second
/// concrete step, `BadStep`, has no abstract image and violates the implied
/// action while the batch is non-empty — the exact condition that made the old
/// code re-borrow the already-returned `iter_state` and panic.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn implied_action_violation_reports_action_level_not_panic() {
    let abstract_mod = abstract_broken_module();

    let concrete = common::parse_module_strict_with_id(
        r#"
---- MODULE ConcreteBroken ----
EXTENDS Integers

Node == 0 .. 2

VARIABLES active, counter, raw, net, step

\* Zero-arg operators supply the abstract `pending`/`token` VARIABLES; this keeps
\* the implied action on the interpreter eval path (mirrors EWD998PCal).
pending == raw
token == net

Init ==
    /\ active = [i \in Node |-> i = 2]
    /\ counter = [i \in Node |-> IF i = 2 THEN 1 ELSE 0]
    /\ raw = [i \in Node |-> IF i = 0 THEN 1 ELSE 0]
    /\ net = [pos |-> 0, q |-> 0, color |-> "black"]
    /\ step = 0

\* Maps to abstract GoodMsg(2): counter[2]+1, pending[(2+2)%3=1]+1, token/active unchanged.
GoodStep ==
    /\ step = 0
    /\ counter' = [counter EXCEPT ![2] = @ + 1]
    /\ raw' = [raw EXCEPT ![1] = @ + 1]
    /\ step' = 1
    /\ UNCHANGED <<active, net>>

\* No abstract image: changes counter[0] and moves to step 2.
BadStep ==
    /\ step = 0
    /\ counter' = [counter EXCEPT ![0] = @ + 1]
    /\ step' = 2
    /\ UNCHANGED <<active, raw, net>>

\* GoodStep first so its (valid) successor is batched before BadStep violates.
Next == GoodStep \/ BadStep

vars == <<active, counter, raw, net, step>>

A == INSTANCE AbstractBroken

ASpec == A!Spec
====
"#,
        FileId(10),
    );

    let config = property_config("ASpec");
    let mut checker = ModelChecker::new_with_extends(&concrete, &[&abstract_mod], &config);
    checker.set_store_states(true);
    match checker.check() {
        CheckResult::PropertyViolation { property, kind, .. } => {
            assert_eq!(
                property, "ASpec",
                "expected the refinement property ASpec to be the violated property"
            );
            assert_eq!(
                kind,
                tla_check::PropertyViolationKind::ActionLevel,
                "a `[][A!Next]_A!vars` refinement failure is an action-level property violation"
            );
        }
        other => panic!(
            "expected an action-level PropertyViolation for the removed abstract action, got: {other:?}"
        ),
    }
}
