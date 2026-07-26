// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Build script: embeds `TY_GIT_COMMIT` at compile time.

fn main() {
    // Embed the full source identity so strict evidence can attribute the
    // exact executable rather than accepting an ambiguous short prefix.
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        );

    println!("cargo:rustc-env=TY_GIT_COMMIT={commit}");
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain=v1", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or("unknown", |output| {
            if output.stdout.is_empty() {
                "false"
            } else {
                "true"
            }
        });
    println!("cargo:rustc-env=TY_GIT_DIRTY={dirty}");

    // Re-run if HEAD or the index changes. `git rev-parse --git-path` is
    // required here because `.git` is a file in a linked worktree, while the
    // per-worktree HEAD/index and shared refs live elsewhere.
    for path in ["HEAD", "refs/heads", "index"] {
        let resolved = std::process::Command::new("git")
            .args(["rev-parse", "--git-path", path])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string());
        if let Some(resolved) = resolved.filter(|path| !path.is_empty()) {
            println!("cargo:rerun-if-changed={resolved}");
        }
    }
}
