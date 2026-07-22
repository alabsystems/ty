// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for `ty verdict-emit` + `ty verdict-check` — the self-service
//! `ty.verdict/v1` round-trip. The re-check is eval-only: it re-parses the embedded
//! spec and replays the trace through the evaluator, so the SOUNDNESS basis is the
//! replay leg, not the digest. The adversarial test below confirms that: tampering a
//! field AND recomputing the digest is still REJECTED.

use std::path::PathBuf;
use std::process::Command;

use tla_check::verdict::VerdictEnvelope;

fn ty() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ty"))
}

struct Spec {
    dir: PathBuf,
    tla: PathBuf,
    cfg: PathBuf,
}

impl Drop for Spec {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn write_spec(tag: &str, module: &str, tla_src: &str, cfg_src: &str) -> Spec {
    // `tag` makes the directory unique PER TEST (tests run in parallel, so a shared dir
    // would race and the Drop cleanup of one test would delete another's files);
    // `module` is the spec's module name and must match its `.tla` filename.
    let dir = std::env::temp_dir().join(format!("ty-verdict-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let tla = dir.join(format!("{module}.tla"));
    let cfg = dir.join(format!("{module}.cfg"));
    std::fs::write(&tla, tla_src).expect("write tla");
    std::fs::write(&cfg, cfg_src).expect("write cfg");
    Spec { dir, tla, cfg }
}

fn emit(spec: &Spec, out: &std::path::Path) -> (i32, String) {
    let o = ty()
        .arg("verdict-emit")
        .arg(&spec.tla)
        .arg("--config")
        .arg(&spec.cfg)
        .arg("--out")
        .arg(out)
        .output()
        .expect("run ty verdict-emit");
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}

fn check(envelope: &std::path::Path) -> (i32, String) {
    let o = ty()
        .arg("verdict-check")
        .arg(envelope)
        .output()
        .expect("run ty verdict-check");
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}

// A spec that violates `Inv` (x reaches 5, Inv is x < 2). `Always` (x >= 0) is a
// defined operator that always HOLDS — used by the adversarial test.
const VIOLATING_SPEC: &str = "---- MODULE BadV ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x < 5 /\\ x' = x + 1\nInv == x < 2\nAlways == x >= 0\n====\n";
const VIOLATING_CFG: &str = "INIT Init\nNEXT Next\nINVARIANT Inv\n";

#[test]
fn verdict_emit_then_check_round_trips_verified() {
    let spec = write_spec("rt", "BadV", VIOLATING_SPEC, VIOLATING_CFG);
    let env_path = spec.dir.join("v.json");

    let (ecode, eout) = emit(&spec, &env_path);
    assert_eq!(
        ecode, 0,
        "verdict-emit should succeed on a violation; got {ecode}\n{eout}"
    );
    assert!(env_path.is_file(), "envelope file should be written");
    assert!(
        eout.contains("invariant-violation"),
        "emit should report the kind:\n{eout}"
    );

    let (ccode, cout) = check(&env_path);
    assert_eq!(
        ccode, 0,
        "verdict-check should VERIFY a genuine envelope; got {ccode}\n{cout}"
    );
    assert!(cout.contains("VERIFIED"), "expected VERIFIED:\n{cout}");
    // The re-check must actually replay (not just check the digest).
    assert!(
        cout.contains("replayed"),
        "expected the replay leg to run:\n{cout}"
    );
}

#[test]
fn verdict_check_rejects_a_byte_tampered_envelope() {
    let spec = write_spec("byte", "BadV", VIOLATING_SPEC, VIOLATING_CFG);
    let env_path = spec.dir.join("v.json");
    assert_eq!(emit(&spec, &env_path).0, 0);

    // Flip one character in a trace value WITHOUT recomputing the digest → the digest
    // leg catches it.
    let json = std::fs::read_to_string(&env_path).unwrap();
    let tampered = json.replacen("\"value\": 1", "\"value\": 7", 1);
    assert_ne!(
        tampered, json,
        "test setup: expected a trace int value to tamper"
    );
    let tampered_path = spec.dir.join("v_byte.json");
    std::fs::write(&tampered_path, tampered).unwrap();

    let (code, out) = check(&tampered_path);
    assert_eq!(
        code, 1,
        "byte-tampered envelope must be REJECTED; got {code}\n{out}"
    );
    assert!(out.contains("REJECTED"), "expected REJECTED:\n{out}");
}

#[test]
fn verdict_check_rejects_tamper_even_with_recomputed_digest() {
    // The KEY soundness property: the trust is in the REPLAY leg, not the digest. Point
    // `violated` at `Always` (which HOLDS at the final state), recompute the digest so
    // the tamper-evidence passes, and confirm the re-check still REJECTS because the
    // evaluator finds the named invariant true.
    let spec = write_spec("recomp", "BadV", VIOLATING_SPEC, VIOLATING_CFG);
    let env_path = spec.dir.join("v.json");
    assert_eq!(emit(&spec, &env_path).0, 0);

    let json = std::fs::read_to_string(&env_path).unwrap();
    let mut env = VerdictEnvelope::from_json(&json).expect("parse envelope");
    env.violated = Some("Always".to_string());
    env.digest = String::new();
    env.digest = env.compute_digest(); // re-seal so the digest leg passes

    let recomputed_path = spec.dir.join("v_recomputed.json");
    std::fs::write(&recomputed_path, env.to_json()).unwrap();

    let (code, out) = check(&recomputed_path);
    assert_eq!(
        code, 1,
        "tamper with a recomputed digest must STILL be REJECTED by the replay/eval leg; got {code}\n{out}"
    );
    assert!(
        out.contains("REJECTED") && out.contains("HOLDS"),
        "expected REJECTED because the invariant actually HOLDS:\n{out}"
    );
}

#[test]
fn verdict_emit_reports_no_violation_for_a_safe_spec() {
    // A deadlock-free safe cycle: no violation, so there is nothing to certify.
    let spec = write_spec(
        "safe",
        "SafeV",
        "---- MODULE SafeV ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x' = (x + 1) % 3\nInv == x < 3\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let env_path = spec.dir.join("v.json");
    let (code, out) = emit(&spec, &env_path);
    assert_eq!(
        code, 2,
        "a safe spec yields no violation envelope (exit 2); got {code}\n{out}"
    );
    assert!(
        out.contains("NO VIOLATION"),
        "expected a NO VIOLATION message:\n{out}"
    );
}

#[test]
fn verdict_property_violation_emits_then_check_is_inconclusive() {
    // An ACTION-LEVEL temporal property violation: `ty check` emits error_type
    // "property_violation" (NOT the dead "state_level"/"action_level" strings). emit must
    // package it (exit 0); check replays the trace (valid Init/Next) but honestly reports
    // INCONCLUSIVE (exit 2) — v1 cannot re-confirm a transition-level property.
    let spec = write_spec(
        "prop",
        "PropV",
        "---- MODULE PropV ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 0\nNext == x > -3 /\\ x' = x - 1\nInc == [][x' > x]_x\n====\n",
        "INIT Init\nNEXT Next\nPROPERTY Inc\n",
    );
    let env_path = spec.dir.join("v.json");
    let (ecode, eout) = emit(&spec, &env_path);
    assert_eq!(
        ecode, 0,
        "verdict-emit should package a property violation; got {ecode}\n{eout}"
    );
    assert!(
        eout.contains("property-violation"),
        "expected the property-violation kind:\n{eout}"
    );

    let (ccode, cout) = check(&env_path);
    assert_eq!(
        ccode, 2,
        "a temporal property cannot be re-checked in v1 -> INCONCLUSIVE; got {ccode}\n{cout}"
    );
    assert!(
        cout.contains("INCONCLUSIVE") && cout.contains("replay OK"),
        "expected an honest inconclusive that confirms the replay:\n{cout}"
    );
}

#[test]
fn verdict_multivar_round_trips_verified_deterministically() {
    // Multi-variable spec: each trace state is a 3-key variable map (HashMap). The
    // content-address digest must be DETERMINISTIC across processes (sorted-key
    // canonicalization), so re-check VERIFIES every time. A single-variable spec cannot
    // catch the HashMap-ordering digest bug this guards.
    let spec = write_spec(
        "multi",
        "MultiV",
        "---- MODULE MultiV ----\nEXTENDS Naturals\nVARIABLES a, b, c\nInit == a = 0 /\\ b = 0 /\\ c = 0\nNext == a < 4 /\\ a' = a + 1 /\\ b' = b + 1 /\\ c' = c + 1\nInv == a < 3\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let env_path = spec.dir.join("v.json");
    assert_eq!(emit(&spec, &env_path).0, 0, "emit a multi-var violation");
    // Re-check several times in separate processes: all must VERIFY (stable digest).
    for _ in 0..3 {
        let (code, out) = check(&env_path);
        assert_eq!(
            code, 0,
            "multi-var envelope must VERIFY deterministically; got {code}\n{out}"
        );
        assert!(out.contains("VERIFIED"), "expected VERIFIED:\n{out}");
    }
}
