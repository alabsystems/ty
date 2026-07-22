// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for `ty selfcheck` — the self-service trust report that composes
//! cross-mode parity with an independent verdict re-check.

use std::path::PathBuf;
use std::process::Command;

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

fn write_spec(name: &str, tla: &str, cfg: &str) -> Spec {
    let dir = std::env::temp_dir().join(format!("ty-selfcheck-it-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let t = dir.join(format!("{name}.tla"));
    let c = dir.join(format!("{name}.cfg"));
    std::fs::write(&t, tla).unwrap();
    std::fs::write(&c, cfg).unwrap();
    Spec {
        dir,
        tla: t,
        cfg: c,
    }
}

fn selfcheck(spec: &Spec, extra: &[&str]) -> (i32, String) {
    let o = ty()
        .arg("selfcheck")
        .arg(&spec.tla)
        .arg("--config")
        .arg(&spec.cfg)
        .args(extra)
        .output()
        .expect("run ty selfcheck");
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}

#[test]
fn selfcheck_violation_composes_parity_and_counterexample_recheck() {
    let spec = write_spec(
        "SCViol",
        "---- MODULE SCViol ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x < 5 /\\ x' = x + 1\nInv == x < 2\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let (code, out) = selfcheck(&spec, &[]);
    assert_eq!(
        code, 0,
        "all checks consistent -> exit 0; got {code}\n{out}"
    );
    assert!(
        out.contains("VIOLATION"),
        "expected the verdict row:\n{out}"
    );
    assert!(
        out.contains("cross-mode parity"),
        "expected the parity row:\n{out}"
    );
    assert!(
        out.contains("VERIFIED") && out.contains("counterexample re-check"),
        "expected the counterexample re-check to VERIFY:\n{out}"
    );
    assert!(
        out.contains("consistent"),
        "expected a consistent trust report:\n{out}"
    );
}

#[test]
fn selfcheck_safe_spec_reports_parity_and_a_caveat() {
    let spec = write_spec(
        "SCSafe",
        "---- MODULE SCSafe ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x' = (x + 1) % 3\nInv == x < 3\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let (code, out) = selfcheck(&spec, &[]);
    assert_eq!(
        code, 0,
        "a safe consistent spec -> exit 0; got {code}\n{out}"
    );
    assert!(out.contains("SAFE"), "expected a SAFE verdict row:\n{out}");
    assert!(
        out.contains("cross-mode parity"),
        "expected the parity row:\n{out}"
    );
}

#[test]
fn selfcheck_json_reports_trusted_and_checks() {
    let spec = write_spec(
        "SCJson",
        "---- MODULE SCJson ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x < 5 /\\ x' = x + 1\nInv == x < 2\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let (code, out) = selfcheck(&spec, &["--format", "json"]);
    assert_eq!(code, 0, "got {code}\n{out}");
    assert!(
        out.contains("\"trusted\": true"),
        "expected trusted=true:\n{out}"
    );
    assert!(
        out.contains("\"schema\": \"ty.selfcheck/v1\""),
        "expected schema tag:\n{out}"
    );
    assert!(
        out.contains("\"check\": \"cross-mode parity\""),
        "expected the parity check:\n{out}"
    );
}

#[test]
fn selfcheck_native_row_is_honest_not_a_false_pass() {
    // Since the AUTO engine flip (ea5b2986) admits the fused compiled-BFS level for
    // unconstrained flat-safety specs, the default-on SAMPLED native↔interpreter
    // successor crosscheck (a126daab) genuinely runs for this spec — every level has
    // exactly one parent, so every explored parent is compared fail-closed against the
    // interpreter. The row must therefore be a PASS backed by crosscheck-ran evidence
    // (the positive "[compiled-bfs-xcheck] active" marker), NOT the old honest-n/a
    // text, and its wording must claim only a SAMPLED comparison (the fused check
    // samples parents per level and covers only the first few levels on large runs).
    let spec = write_spec(
        "SCNative",
        "---- MODULE SCNative ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x < 4 /\\ x' = x + 1\nInv == x <= 4\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let (code, out) = selfcheck(&spec, &[]);
    assert_eq!(code, 0, "got {code}\n{out}");
    assert!(
        out.contains("native ≡ interpreter"),
        "the native row must be present:\n{out}"
    );
    assert!(
        !out.contains("crosscheck not exercised"),
        "the sampled native↔interpreter crosscheck runs by default on the fused path since \
         ea5b2986/a126daab: the native row must not claim the crosscheck was skipped:\n{out}"
    );
    assert!(
        out.contains("sampled per-parent native↔interpreter crosscheck ran"),
        "the native row must be a PASS backed by crosscheck-ran evidence:\n{out}"
    );
    assert!(
        !out.contains("at every explored state"),
        "the crosscheck is SAMPLED — the PASS row must not overclaim exhaustive per-state \
         coverage:\n{out}"
    );
}
