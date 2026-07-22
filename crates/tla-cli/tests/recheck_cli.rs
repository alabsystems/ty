// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for `ty recheck` — the unified minimal re-checker.

use std::path::PathBuf;
use std::process::Command;

fn ty() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ty"))
}

fn out_of(mut c: Command) -> (i32, String) {
    let o = c.output().expect("run ty");
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}

struct Tmp {
    dir: PathBuf,
}
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
fn tmp(name: &str) -> Tmp {
    let dir = std::env::temp_dir().join(format!("ty-recheck-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    Tmp { dir }
}

#[test]
fn recheck_tcb_prints_trusted_base() {
    let mut c = ty();
    c.arg("recheck").arg("--tcb");
    let (code, out) = out_of(c);
    assert_eq!(code, 0, "--tcb should exit 0; got {code}\n{out}");
    assert!(
        out.contains("TRUSTED COMPUTING BASE"),
        "expected the TCB declaration:\n{out}"
    );
    assert!(
        out.contains("tla-eval"),
        "expected the evaluator in the TCB:\n{out}"
    );
}

#[test]
fn recheck_dispatches_a_verdict_envelope() {
    let t = tmp("verdict");
    let tla = t.dir.join("R.tla");
    let cfg = t.dir.join("R.cfg");
    std::fs::write(
        &tla,
        "---- MODULE R ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x < 5 /\\ x' = x + 1\nInv == x < 2\n====\n",
    )
    .unwrap();
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nINVARIANT Inv\n").unwrap();
    let env = t.dir.join("v.json");

    let mut emit = ty();
    emit.arg("verdict-emit")
        .arg(&tla)
        .arg("--config")
        .arg(&cfg)
        .arg("--out")
        .arg(&env);
    assert_eq!(out_of(emit).0, 0, "verdict-emit should succeed");

    let mut rc = ty();
    rc.arg("recheck").arg(&env);
    let (code, out) = out_of(rc);
    assert_eq!(
        code, 0,
        "recheck of a genuine verdict envelope should VERIFY; got {code}\n{out}"
    );
    assert!(out.contains("VERIFIED"), "expected VERIFIED:\n{out}");
}

#[test]
fn recheck_rejects_unknown_schema() {
    let t = tmp("bad");
    let bad = t.dir.join("bad.json");
    std::fs::write(&bad, "{\"schema\": \"nope/v9\"}").unwrap();
    let mut rc = ty();
    rc.arg("recheck").arg(&bad);
    let (code, out) = out_of(rc);
    assert_ne!(
        code, 0,
        "an unknown-schema artifact must not be accepted; got {code}\n{out}"
    );
    assert!(
        out.contains("unknown artifact schema"),
        "expected a clear error:\n{out}"
    );
}
