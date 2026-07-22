// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration test for `ty prove` — prove + immediate re-check.

use std::process::Command;

#[test]
fn prove_inductive_spec_proves_and_rechecks() {
    let dir = std::env::temp_dir().join(format!("ty-prove-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let tla = dir.join("P.tla");
    let cfg = dir.join("P.cfg");
    std::fs::write(
        &tla,
        "---- MODULE P ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x' = x\nInv == x = 0\n====\n",
    )
    .unwrap();
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nINVARIANT Inv\n").unwrap();

    let o = Command::new(env!("CARGO_BIN_EXE_ty"))
        .arg("prove")
        .arg(&tla)
        .arg("--config")
        .arg(&cfg)
        .output()
        .expect("run ty prove");
    let mut out = String::from_utf8_lossy(&o.stdout).into_owned();
    out.push_str(&String::from_utf8_lossy(&o.stderr));
    let _ = std::fs::remove_dir_all(&dir);

    if out.contains("requires the `ay` feature") {
        // Non-ay build: the prover is unavailable; nothing to assert.
        return;
    }
    assert_eq!(
        o.status.code(),
        Some(0),
        "ty prove should exit 0 on an inductive spec:\n{out}"
    );
    assert!(out.contains("PROVED"), "expected PROVED:\n{out}");
    assert!(
        out.contains("VERIFIED"),
        "expected the immediate re-check to VERIFY:\n{out}"
    );
}
