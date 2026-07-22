// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (test narrative names the round-1 spaced literal.)

//! Cargo-test entry point for the repo-wide MCC keyword regression guard.
//!
//! This test invokes `scripts/mcc_keyword_guard.sh` from the workspace
//! root and fails if any production source file emits a spaced variant
//! of an MCC protocol keyword. The May 2026 qualification-1 rejection
//! was caused by exactly that drift — see
//! `docs/mcc-2026/qualification-1/analysis.md`.
//!
//! Running the guard from `cargo test` means a developer can't merge a
//! change that re-introduces the spaced qual-1 variants
//! mcc-keyword-guard: allow-spaced-mention
//! (`CANNOT COMPUTE` etc.) even if they skip `pre-commit`.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    // crates/tla-petri/tests -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn mcc_keyword_guard_passes() {
    let root = workspace_root();
    let script = root.join("scripts").join("mcc_keyword_guard.sh");
    if !script.exists() {
        panic!(
            "mcc_keyword_guard.sh missing at {} — the regression guard \
             must ship with the source tree.",
            script.display()
        );
    }
    let output = Command::new("bash")
        .arg(&script)
        .current_dir(&root)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke mcc_keyword_guard.sh: {e}"));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "MCC keyword regression guard failed (exit={:?}).\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
        output.status.code()
    );
}
