// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cargo-test entry point for the cross-repo Cargo dep drift guard.
//!
//! Invokes the proof-grade `ty-mcc-drift-guard` binary (a Rust
//! replacement for the legacy regex shell script
//! `scripts/cargo_dep_drift_guard.sh`) and fails the build if any
//! sibling package (trust-ir / trust-codegen / ay family) resolves to a different
//! `(source, version)` pair across the five sibling repos.
//!
//! The binary in turn drives `cargo metadata --locked --all-features
//! --format-version 1` for
//! each repo, so this test exercises the real resolver path — not a
//! Cargo.toml regex.
//!
//! Skipped when no sibling repos are present (single-repo CI clones).
//! Otherwise the default five-repository topology is fail-closed: a missing
//! configured repository, an existing manifest whose metadata fails, or an
//! existing repository with no cross-repo contribution prevents a clean
//! verdict. `--allow-missing` is available only for an explicitly partial
//! checkout; this canonical integration test does not use it.

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

fn sibling_root() -> PathBuf {
    workspace_root()
        .parent()
        .expect("~/root parent of ty")
        .to_path_buf()
}

#[test]
fn cargo_dep_drift_guard_passes() {
    let ty_root = workspace_root();
    let root = sibling_root();

    // Skip if no sibling repos exist (single-repo CI environment).
    let any_sibling = ["trust-ir", "trust-cg", "clean", "ay"]
        .iter()
        .any(|r| root.join(r).is_dir());
    if !any_sibling {
        eprintln!(
            "ty-mcc-drift-guard: no sibling repos under {} — skipping.",
            root.display()
        );
        return;
    }

    // env!("CARGO_BIN_EXE_<name>") gives the path to the built binary
    // for tests in the same crate. This avoids any PATH hunting and
    // guarantees we run the exact binary cargo built for this test.
    let binary = env!("CARGO_BIN_EXE_ty-mcc-drift-guard");

    let output = Command::new(binary)
        .arg("--root")
        .arg(&root)
        .current_dir(&ty_root)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke ty-mcc-drift-guard: {e}"));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "cross-repo dep drift guard failed (exit={:?}).\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
        output.status.code()
    );
}
