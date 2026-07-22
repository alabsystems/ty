// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// Regression test for #976: TLC runs from our scripts/binaries must use an
// ephemeral `-metadir` by default, and only write `./states` when the user
// opts in via `TY_KEEP_STATES=1`. Three source groups need `-metadir`
// AND `TY_PRESERVE_STATES_DIR=1` support so existing `./states` trees
// are not clobbered:
//
//   * verify_correctness (shell)
//   * compare_with_tlc (shell)
//   * collect_tlc_baseline (the Rust port — `ty-tlc-baseline`)
//
// Two more source groups need `-metadir` + `TY_KEEP_STATES` only:
//
//   * differential_test (v1 and v2, shell)
//   * test_all_liveness (shell)
//
// This test was ported from `tests/regression/test_tlc_metadir_usage.py`
// when the Python collect_tlc_baseline package was rewritten in Rust.

use std::fs;
use std::path::{Path, PathBuf};

fn project_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<root>/crates/tla-petri`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("project root above CARGO_MANIFEST_DIR")
        .to_path_buf()
}

fn read_group(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| fs::read_to_string(p).unwrap_or_else(|err| panic!("read {}: {err}", p.display())))
        .collect::<Vec<_>>()
        .join("\n")
}

struct SourceGroup {
    name: &'static str,
    paths: Vec<PathBuf>,
    preserves_states_dir: bool,
}

fn source_groups() -> Vec<SourceGroup> {
    let root = project_root();
    vec![
        SourceGroup {
            name: "verify_correctness",
            paths: vec![
                root.join("scripts/verify_correctness.sh"),
                root.join("scripts/lib/verify_correctness/runners.sh"),
            ],
            preserves_states_dir: true,
        },
        SourceGroup {
            name: "compare_with_tlc",
            paths: vec![root.join("scripts/compare_with_tlc.sh")],
            preserves_states_dir: true,
        },
        SourceGroup {
            name: "collect_tlc_baseline",
            paths: vec![root.join("crates/tla-petri/src/bin/ty-tlc-baseline.rs")],
            preserves_states_dir: true,
        },
        SourceGroup {
            name: "differential_test",
            paths: vec![root.join("scripts/differential_test.sh")],
            preserves_states_dir: false,
        },
        SourceGroup {
            name: "differential_test_v2",
            paths: vec![root.join("scripts/differential_test_v2.sh")],
            preserves_states_dir: false,
        },
        SourceGroup {
            name: "test_all_liveness",
            paths: vec![root.join("scripts/test_all_liveness.sh")],
            preserves_states_dir: false,
        },
    ]
}

#[test]
fn tlc_runs_use_ephemeral_metadir_by_default() {
    for group in source_groups() {
        let text = read_group(&group.paths);
        assert!(
            text.contains("-metadir"),
            "{} should pass -metadir to TLC by default",
            group.name
        );
        assert!(
            text.contains("TY_KEEP_STATES"),
            "{} should allow opting into keeping ./states via TY_KEEP_STATES=1",
            group.name
        );
        if group.preserves_states_dir {
            assert!(
                text.contains("TY_PRESERVE_STATES_DIR"),
                "{} should allow preserving an existing ./states via TY_PRESERVE_STATES_DIR=1",
                group.name
            );
        }
    }
}
