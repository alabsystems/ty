// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration test for `ty fuzz` — the differential fuzzer.

use std::process::Command;

#[test]
fn fuzz_small_run_is_deterministic_and_clean() {
    let o = Command::new(env!("CARGO_BIN_EXE_ty"))
        .arg("fuzz")
        .arg("--seed")
        .arg("7")
        .arg("--count")
        .arg("8")
        .output()
        .expect("run ty fuzz");
    let mut out = String::from_utf8_lossy(&o.stdout).into_owned();
    out.push_str(&String::from_utf8_lossy(&o.stderr));
    // The engines are correct, so a small fuzz run must find no divergence (exit 0).
    assert_eq!(
        o.status.code(),
        Some(0),
        "fuzz should be CLEAN (no divergence); got:\n{out}"
    );
    assert!(
        out.contains("8 specs fuzzed"),
        "expected the run summary:\n{out}"
    );
    assert!(out.contains("CLEAN"), "expected a CLEAN result:\n{out}");
    // The differential must actually have RUN on comparable specs — guard against a
    // generator/harness regression that yields only un-comparable (inconclusive) specs,
    // which would make the "CLEAN" result vacuous.
    assert!(
        !out.contains(": 0 parity,"),
        "the generator must produce specs the engines can actually compare (>0 parity), else \
         the differential is a no-op:\n{out}"
    );
}
