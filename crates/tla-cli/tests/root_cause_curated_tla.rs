// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::process::Command;

mod common;
use common::TempDir;

fn run_ty(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ty"))
        .args(args)
        .output()
        .expect("run ty")
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn parallel_check_accepts_assume_only_toolbox_expression_models() {
    let dir = TempDir::new("tla-cli-assume-only-parallel");
    let spec = dir.path.join("MC.tla");
    let cfg = dir.path.join("MC.cfg");
    std::fs::write(
        &spec,
        r#"---- MODULE MC ----
CONSTANT N
ASSUME N = 40
====
"#,
    )
    .expect("write spec");
    std::fs::write(&cfg, "CONSTANT N = 40\n").expect("write cfg");

    let out = run_ty(&[
        "check",
        spec.to_str().expect("utf-8 spec"),
        "--config",
        cfg.to_str().expect("utf-8 cfg"),
        "--workers",
        "4",
        "--output",
        "json",
    ]);

    assert!(
        out.status.success(),
        "expected success\nstatus: {:?}\nstderr:\n{}\nstdout:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(json["result"]["status"], "ok");
    assert_eq!(json["statistics"]["states_found"], 0);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn primed_zero_arg_let_uses_primed_definition_body() {
    let dir = TempDir::new("tla-cli-primed-zero-arg-let");
    let spec = dir.path.join("LetPrime.tla");
    let cfg = dir.path.join("LetPrime.cfg");
    std::fs::write(
        &spec,
        r#"---- MODULE LetPrime ----
EXTENDS Naturals, TLC

VARIABLE x, y

Init == /\ x = 1
        /\ y = 3

Next ==
    \/ /\ x = 1
       /\ x' = 2
       /\ y' = y
       /\ IF LET a == x IN a' = 2
             THEN TRUE
             ELSE Assert(FALSE, "primed LET used current value")
    \/ LET b == x IN
       /\ x = 2
       /\ x' = 3
       /\ y' = y
       /\ IF b' = 3
             THEN TRUE
             ELSE Assert(FALSE, "primed structural LET used current value")
    \/ /\ x = 3
       /\ UNCHANGED <<x, y>>

Inv == TRUE
====
"#,
    )
    .expect("write spec");
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("write cfg");

    let out = run_ty(&[
        "check",
        spec.to_str().expect("utf-8 spec"),
        "--config",
        cfg.to_str().expect("utf-8 cfg"),
        "--workers",
        "4",
        "--no-deadlock",
        "--output",
        "json",
    ]);

    assert!(
        out.status.success(),
        "expected success\nstatus: {:?}\nstderr:\n{}\nstdout:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(json["result"]["status"], "ok");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn disabled_angle_action_does_not_evaluate_later_assert_conjunct() {
    let dir = TempDir::new("tla-cli-angle-action-disabled-assert");
    let spec = dir.path.join("AngleAction.tla");
    let cfg = dir.path.join("AngleAction.cfg");
    std::fs::write(
        &spec,
        r#"---- MODULE AngleAction ----
EXTENDS TLC

VARIABLE x, y

Init == /\ x = 1
        /\ y = 1

Act1 == /\ x' = 1
        /\ y' = 1

Next ==
    \/ /\ UNCHANGED <<x, y>>
       /\ IF [x # x]_<<x, y>>
             THEN Print("Test1 OK", TRUE)
             ELSE Assert(FALSE, "Test 1 Failed")
       /\ IF <<Act1>>_<<x, y>>
             THEN Assert(FALSE, "Test 2 Failed")
             ELSE Print("Test2 OK", TRUE)
    \/ /\ <<x' = 1 /\ y' = 1>>_y
       /\ Assert(FALSE, "Test 3 Failed")

Inv == TRUE
====
"#,
    )
    .expect("write spec");
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("write cfg");

    let out = run_ty(&[
        "check",
        spec.to_str().expect("utf-8 spec"),
        "--config",
        cfg.to_str().expect("utf-8 cfg"),
        "--workers",
        "4",
        "--output",
        "json",
    ]);

    assert!(
        out.status.success(),
        "expected disabled angle action to prune later assert\nstatus: {:?}\nstderr:\n{}\nstdout:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(json["result"]["status"], "ok");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn parallel_liveness_splits_named_instance_spec_before_planning() {
    let dir = TempDir::new("tla-cli-named-instance-spec-liveness-split");
    let inner = dir.path.join("InnerSpec.tla");
    let outer = dir.path.join("InstanceRefines.tla");
    let cfg = dir.path.join("InstanceRefines.cfg");
    std::fs::write(
        &inner,
        r#"---- MODULE InnerSpec ----
VARIABLE x

vars == <<x>>
Init == x = 0
Next == UNCHANGED x
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
====
"#,
    )
    .expect("write inner spec");
    std::fs::write(
        &outer,
        r#"---- MODULE InstanceRefines ----
VARIABLE x

Init == x = 0
Next == UNCHANGED x
I == INSTANCE InnerSpec
Refines == I!Spec
====
"#,
    )
    .expect("write outer spec");
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nPROPERTY Refines\n").expect("write cfg");

    let out = run_ty(&[
        "check",
        outer.to_str().expect("utf-8 spec"),
        "--config",
        cfg.to_str().expect("utf-8 cfg"),
        "--workers",
        "4",
        "--output",
        "json",
    ]);

    assert!(
        out.status.success(),
        "expected named-instance Spec liveness split to succeed\nstatus: {:?}\nstderr:\n{}\nstdout:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(json["result"]["status"], "ok");
}
