// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const BUILD_GIT_HEAD_ENV: &str = "TY_MCC_BUILD_GIT_HEAD";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={BUILD_GIT_HEAD_ENV}");

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"),
    );
    emit_git_rerun_paths(&manifest_dir);

    let build_git_head = match env::var(BUILD_GIT_HEAD_ENV) {
        Ok(value) => {
            let value = value.trim().to_owned();
            assert!(
                is_full_git_rev(&value),
                "{BUILD_GIT_HEAD_ENV} must be a 40-hex Git revision"
            );
            value
        }
        Err(_) => git_output(&manifest_dir, &["rev-parse", "HEAD"])
            .filter(|value| is_full_git_rev(value))
            .unwrap_or_else(|| "unknown".to_owned()),
    };
    let build_git_head_short = build_git_head
        .get(..8)
        .filter(|_| is_full_git_rev(&build_git_head))
        .unwrap_or("unknown");

    println!("cargo:rustc-env=TY_MCC_BUILD_GIT_HEAD={build_git_head}");
    println!("cargo:rustc-env=TY_MCC_BUILD_GIT_HEAD_SHORT={build_git_head_short}");
}

fn emit_git_rerun_paths(manifest_dir: &Path) {
    let Some(git_dir_raw) = git_output(manifest_dir, &["rev-parse", "--git-dir"]) else {
        return;
    };
    let git_dir = manifest_dir.join(git_dir_raw);
    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());

    let Ok(head) = std::fs::read_to_string(git_dir.join("HEAD")) else {
        return;
    };
    let Some(ref_name) = head.trim().strip_prefix("ref: ") else {
        return;
    };
    println!(
        "cargo:rerun-if-changed={}",
        git_dir.join(ref_name).display()
    );
}

fn git_output(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_owned())
    }
}

fn is_full_git_rev(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
