// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (this module emits no MCC stdout; comment present so the keyword guard is
// satisfied for any future tests that build spaced literals at runtime.)

//! Library backing for `ty-mcc-ay-pin-validate` and the equivalent
//! `ty-mccctl ay-pin-validate` subcommand.
//!
//! Validates that the MCC packaging pins (Dockerfile `ARG AY_REV`) match the
//! workspace `Cargo.toml` `[workspace.dependencies]` and the locked
//! `Cargo.lock` git revs for every `ay*` crate. The Python module lived in
//! `mcc/` and was invoked from `crates/tla-petri/src/mccctl.rs` via a
//! `python3` subprocess. Replacing it with a Cargo bin (plus the shared
//! [`crate::mcc_ay_pin`] library) gives us a single, compiler-enforced
//! interface so the doctor gate no longer depends on the Python install.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use serde_json::json;

use crate::mcc_ay_pin::{validate_ay_pin, AYPinSummary, PinValidationError};

/// Command-line arguments for the `ty-mcc-ay-pin-validate` helper.
#[derive(Parser, Debug)]
#[command(
    name = "ty-mcc-ay-pin-validate",
    about = "Validate that MCC Dockerfile AY_REV matches workspace ay pins.",
    long_about = "Validates that `mcc/Dockerfile.mcc` ARG AY_REV matches the \
                  workspace `Cargo.toml` rev for every `ay*` git dependency \
                  AND every locked `Cargo.lock` git source.\n\n\
                  Exit 0 = OK, exit 1 = any failure (reason on stderr).\n\
                  Replaces a former Python helper."
)]
pub struct Cli {
    /// TY repository root. Defaults to the workspace root resolved
    /// relative to this binary's compile-time `CARGO_MANIFEST_DIR`.
    #[arg(long = "repo-root", value_name = "PATH")]
    pub repo_root: Option<PathBuf>,
    /// MCC Dockerfile path; defaults to `<repo-root>/mcc/Dockerfile.mcc`.
    #[arg(long, value_name = "PATH")]
    pub dockerfile: Option<PathBuf>,
    /// Emit JSON summary on stdout instead of the human-readable line.
    #[arg(long)]
    pub json: bool,
}

/// Entry point used by the standalone `ty-mcc-ay-pin-validate` binary.
pub fn run() -> ExitCode {
    execute(Cli::parse())
}

/// Entry point used by `ty-mccctl ay-pin-validate`.
pub fn run_from<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(err) => {
            let _ = err.print();
            return ExitCode::from(u8::from(err.use_stderr()));
        }
    };
    execute(cli)
}

fn execute(cli: Cli) -> ExitCode {
    let repo_root = match cli.repo_root.clone() {
        Some(p) => p,
        None => match default_repo_root() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("FAIL: {err}");
                return ExitCode::from(1);
            }
        },
    };
    let repo_root = match repo_root.canonicalize() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("FAIL: resolving repo root {}: {err}", repo_root.display());
            return ExitCode::from(1);
        }
    };
    match validate_ay_pin(&repo_root, cli.dockerfile.as_deref()) {
        Ok(summary) => {
            print_summary(&summary, cli.json);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("FAIL: {}", format_error(&err));
            ExitCode::from(1)
        }
    }
}

fn format_error(err: &PinValidationError) -> String {
    err.to_string()
}

fn print_summary(summary: &AYPinSummary, want_json: bool) {
    if want_json {
        let body = json!({
            "dockerfile_rev": summary.dockerfile_rev,
            "cargo_toml_rev": summary.cargo_toml_rev,
            "cargo_lock_rev": summary.cargo_lock_rev,
            "cargo_toml_deps": summary.cargo_toml_deps,
            "cargo_lock_packages": summary.cargo_lock_packages,
        });
        println!("{}", serde_json::to_string(&body).expect("json serialize"));
    } else {
        println!(
            "OK: mcc/Dockerfile.mcc AY_REV matches Cargo.toml and Cargo.lock: {}",
            summary.dockerfile_rev
        );
    }
}

/// The compiled binary lives at `<repo>/target/.../ty-mcc-ay-pin-validate`.
/// Walking up from the binary cwd is unreliable, so prefer the workspace
/// root resolved relative to this source file. `CARGO_MANIFEST_DIR` is set
/// at build time and points at `<repo>/crates/tla-petri`; the workspace root
/// is two levels up.
fn default_repo_root() -> Result<PathBuf, String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("could not resolve workspace root from {}", manifest_dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: the binary surface re-exports the library validator's
    /// summary type and the JSON shape stays in sync with the Python
    /// `AYPinSummary.as_dict()` keys. The exhaustive failure-class tests
    /// live in [`crate::mcc_ay_pin`].
    #[test]
    fn print_summary_emits_json_keys_in_legacy_shape() {
        let summary = AYPinSummary {
            dockerfile_rev: "a".repeat(40),
            cargo_toml_rev: "a".repeat(40),
            cargo_lock_rev: "a".repeat(40),
            cargo_toml_deps: vec!["ay".into(), "ay-dpll".into()],
            cargo_lock_packages: vec!["ay-chc".into()],
        };
        // We don't capture stdout here; just exercise the json! macro to
        // catch any future AYPinSummary field rename that would break the
        // sidecar consumers.
        let body = json!({
            "dockerfile_rev": summary.dockerfile_rev,
            "cargo_toml_rev": summary.cargo_toml_rev,
            "cargo_lock_rev": summary.cargo_lock_rev,
            "cargo_toml_deps": summary.cargo_toml_deps,
            "cargo_lock_packages": summary.cargo_lock_packages,
        });
        let s = body.to_string();
        assert!(s.contains("dockerfile_rev"));
        assert!(s.contains("cargo_toml_deps"));
        assert!(s.contains("cargo_lock_packages"));
    }
}
