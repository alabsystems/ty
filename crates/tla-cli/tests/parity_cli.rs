// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for `ty parity` — the self-service cross-mode verdict-parity
//! check. Each test runs the real `ty` binary, which internally shells out to
//! `ty check` under the interpreter oracle, the fused default, and the native
//! trust-cg backend, and asserts they agree. (The native engine degrades to
//! `not run` / fallback on a non-ay build; parity still holds via oracle+fused.)

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

fn write_spec(name: &str, tla_src: &str, cfg_src: &str) -> Spec {
    let dir = std::env::temp_dir().join(format!("ty-parity-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let tla = dir.join(format!("{name}.tla"));
    let cfg = dir.join(format!("{name}.cfg"));
    std::fs::write(&tla, tla_src).expect("write tla");
    std::fs::write(&cfg, cfg_src).expect("write cfg");
    Spec { dir, tla, cfg }
}

fn run_parity(spec: &Spec, extra: &[&str]) -> (i32, String) {
    let out = ty()
        .arg("parity")
        .arg(&spec.tla)
        .arg("--config")
        .arg(&spec.cfg)
        .args(extra)
        .output()
        .expect("run `ty parity`");
    let mut combined = String::from_utf8_lossy(&out.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), combined)
}

#[test]
fn parity_safe_spec_all_engines_agree() {
    let spec = write_spec(
        "ParitySafe",
        "---- MODULE ParitySafe ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x' = (x + 1) % 3\nInv == x < 3\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let (code, out) = run_parity(&spec, &[]);
    assert_eq!(code, 0, "expected exit 0 (parity); got {code}\n{out}");
    assert!(out.contains("PARITY"), "expected a PARITY verdict:\n{out}");
    assert!(
        out.contains("SAFE"),
        "expected the agreed verdict to be SAFE:\n{out}"
    );
    assert!(
        out.contains("agrees with oracle"),
        "expected at least one engine to agree with the oracle:\n{out}"
    );
}

#[test]
fn parity_safe_invariant_with_deadlock_all_engines_agree_deadlock() {
    // The exact masking-bug scenario: an inductive safety invariant (x <= 3) PLUS a
    // reachable deadlock at x = 3. Every engine — including the fused default whose
    // symbolic-safe lane proves the invariant — must report DEADLOCK, not a masked
    // SAFE. A regression of the symbolic-safe-masking fix would surface here as a
    // DISAGREEMENT (exit 1).
    let spec = write_spec(
        "ParityDeadlock",
        "---- MODULE ParityDeadlock ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x < 3 /\\ x' = x + 1\nInv == x <= 3\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let (code, out) = run_parity(&spec, &[]);
    assert_eq!(code, 0, "expected exit 0 (parity); got {code}\n{out}");
    assert!(
        out.contains("DEADLOCK"),
        "expected DEADLOCK across engines:\n{out}"
    );
    assert!(out.contains("PARITY"), "expected a PARITY verdict:\n{out}");
}

#[test]
fn parity_json_reports_status_and_per_engine_agreement() {
    let spec = write_spec(
        "ParityJson",
        "---- MODULE ParityJson ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x < 5 /\\ x' = x + 1\nInv == x < 2\n====\n",
        "INIT Init\nNEXT Next\nINVARIANT Inv\n",
    );
    let (code, out) = run_parity(&spec, &["--format", "json"]);
    assert_eq!(
        code, 0,
        "all engines agree on the invariant violation -> parity, exit 0; got {code}\n{out}"
    );
    assert!(
        out.contains("\"status\": \"parity\""),
        "expected parity status in JSON:\n{out}"
    );
    assert!(
        out.contains("\"verdict\": \"invariant-violation\""),
        "expected the invariant-violation verdict in JSON:\n{out}"
    );
    assert!(
        out.contains("\"schema\": \"ty.parity/v1\""),
        "expected the ty.parity/v1 schema tag:\n{out}"
    );
}
