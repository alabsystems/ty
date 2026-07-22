// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cargo-test entry point for the MCC examination-name parity guard.
//!
//! The 13 MCC examination names are the protocol vocabulary. Rust has
//! `Examination` as the source of truth. All former Python drift sites
//! have been replaced by in-tree Rust binaries (`ty-mcc-history`,
//! `ty-mcc-sweep`, `ty-mcc-validate`) that route every examination name
//! through `Examination`.
//!
//! The parity guard now acts as a structural fence: any reintroduction
//! of a Python `EXAMS = [...]` list under crates/, scripts/, mcc/, or
//! tests/ fails the build. Same class of bug as the qualification-1
//! keyword drift (see `docs/mcc-2026/qualification-1/analysis.md`).
//!
//! This test invokes `scripts/mcc_examination_parity.sh` so a developer
//! can't merge a Python-side drift even by skipping pre-commit.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn mcc_examination_parity_passes() {
    let root = workspace_root();
    let script = root.join("scripts").join("mcc_examination_parity.sh");
    assert!(
        script.exists(),
        "mcc_examination_parity.sh missing at {} — the parity guard \
         must ship with the source tree.",
        script.display()
    );
    let output = Command::new("bash")
        .arg(&script)
        .current_dir(&root)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke mcc_examination_parity.sh: {e}"));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "MCC examination-name parity guard failed (exit={:?}).\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
        output.status.code()
    );
}
