// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use tla_petri::cli::{run_cli, PetriCommandMode, PetriRunArgs};

const BACKEND_EVIDENCE_JSONL_ENV: &str = "TY_MCC_BACKEND_EVIDENCE_JSONL";
const BUILD_PROVENANCE_SCHEMA: &str = "mcc.ty_mcc.build_provenance.v1";
const BUILD_GIT_HEAD: &str = env!("TY_MCC_BUILD_GIT_HEAD");
const BUILD_GIT_HEAD_SHORT: &str = env!("TY_MCC_BUILD_GIT_HEAD_SHORT");

/// Stack size for the MCC worker thread.
///
/// CTL/LTL model checking can recurse through formula depth and state-space
/// successors. Some MCC fireability formulas are deeply nested enough to
/// overflow a 64 MiB worker stack before the normal CANNOT_COMPUTE guards can
/// run, so reserve a larger single worker stack for the competition binary.
const WORKER_STACK_SIZE: usize = 256 * 1024 * 1024;

/// Dedicated MCC entrypoint for the BenchKit wrapper.
#[derive(Debug, Parser)]
#[command(name = "ty-mcc", version, about = "Run TY MCC Petri model checking")]
struct Cli {
    /// Model directory or a direct path to `model.pnml`.
    ///
    /// If omitted, uses `BK_INPUT` or the current directory.
    model_dir: Option<PathBuf>,

    /// MCC examination name.
    ///
    /// If omitted, uses `BK_EXAMINATION`.
    #[arg(long)]
    examination: Option<String>,

    /// Write backend routing evidence to a JSONL sidecar without changing MCC stdout.
    #[arg(long, value_name = "PATH")]
    backend_evidence_jsonl: Option<PathBuf>,

    /// Print machine-readable build provenance and exit.
    #[arg(long, hide = true)]
    build_provenance_json: bool,

    #[command(flatten)]
    args: PetriRunArgs,
}

#[derive(Debug, Serialize)]
struct BuildProvenance {
    schema: &'static str,
    schema_version: u32,
    binary: &'static str,
    package: &'static str,
    cargo_pkg_version: &'static str,
    build_git_head: &'static str,
    build_git_head_short: &'static str,
    target_arch: &'static str,
    target_os: &'static str,
}

impl BuildProvenance {
    fn current() -> Self {
        Self {
            schema: BUILD_PROVENANCE_SCHEMA,
            schema_version: 1,
            binary: "ty-mcc",
            package: env!("CARGO_PKG_NAME"),
            cargo_pkg_version: env!("CARGO_PKG_VERSION"),
            build_git_head: BUILD_GIT_HEAD,
            build_git_head_short: BUILD_GIT_HEAD_SHORT,
            target_arch: std::env::consts::ARCH,
            target_os: std::env::consts::OS,
        }
    }

    fn evidence_row(&self) -> String {
        format!(
            "MCC ty_mcc_build_provenance schema={schema} schema_version={schema_version} \
             binary={binary} package={package} cargo_pkg_version={cargo_pkg_version} \
             build_git_head={build_git_head} build_git_head_short={build_git_head_short} \
             target_arch={target_arch} target_os={target_os} current_head_gate=required \
             production_selected=true fail_closed=false",
            schema = self.schema,
            schema_version = self.schema_version,
            binary = self.binary,
            package = self.package,
            cargo_pkg_version = self.cargo_pkg_version,
            build_git_head = self.build_git_head,
            build_git_head_short = self.build_git_head_short,
            target_arch = self.target_arch,
            target_os = self.target_os,
        )
    }
}

#[derive(Debug, Serialize)]
struct BuildProvenanceSidecar {
    schema_version: u32,
    model: &'static str,
    examination: &'static str,
    report: BuildProvenanceReport,
}

#[derive(Debug, Serialize)]
struct BuildProvenanceReport {
    evidence: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.build_provenance_json {
        serde_json::to_writer_pretty(std::io::stdout(), &BuildProvenance::current())?;
        println!();
        return Ok(());
    }

    // Capability gate. If the operator opts in to native via env var
    // (TY_MCC_REQUIRE_NATIVE=1) AND the host arch has no working
    // trust-codegen codegen backend (everything except aarch64 as of May 2026),
    // refuse with the tool-level `CANNOT_COMPUTE` keyword alone on a
    // single line. This is the Rust-side mirror of the same check in
    // `mcc/BenchKit_head.sh` — the shell wrapper was the only fence
    // until codex audit finding #7 caught that a direct binary
    // invocation bypassed it. Failing closed here means a future
    // automated harness that skips the wrapper can't ship silent
    // wrong native code targeting the "unknown-host" sentinel.
    if require_native_from_env() && !host_has_native_codegen() {
        eprintln!(
            "TY_MCC_REQUIRE_NATIVE=1 but trust-codegen has no codegen backend \
             for host arch '{}' (only aarch64 supported as of \
             2026-05-17). Emitting tool-level CANNOT_COMPUTE.",
            std::env::consts::ARCH,
        );
        println!("CANNOT_COMPUTE");
        return Ok(());
    }

    let builder = std::thread::Builder::new()
        .name("ty-mcc-main".into())
        .stack_size(WORKER_STACK_SIZE);
    let handle = builder.spawn(move || run(cli))?;
    handle.join().expect("ty-mcc worker thread panicked")
}

fn require_native_from_env() -> bool {
    match std::env::var("TY_MCC_REQUIRE_NATIVE") {
        Ok(value) => matches!(
            value.trim(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        ),
        Err(_) => false,
    }
}

const fn host_has_native_codegen() -> bool {
    // Mirrors `tla_trust_cg::compile::host_has_trust_cg_codegen_backend()`.
    // Kept const + inline so the binary doesn't gain a cross-crate
    // dep just for the capability check.
    cfg!(target_arch = "aarch64") || cfg!(target_arch = "x86_64")
}

fn run(cli: Cli) -> Result<()> {
    if let Some(path) = cli.backend_evidence_jsonl {
        tla_petri::env_guard::set_var(BACKEND_EVIDENCE_JSONL_ENV, path);
    }

    maybe_emit_build_provenance_sidecar();

    run_cli(
        PetriCommandMode::Mcc,
        cli.model_dir,
        cli.examination,
        cli.args,
    )
}

fn maybe_emit_build_provenance_sidecar() {
    let Some(path) = std::env::var_os(BACKEND_EVIDENCE_JSONL_ENV) else {
        return;
    };
    if path.is_empty() {
        return;
    }

    let provenance = BuildProvenance::current();
    let record = BuildProvenanceSidecar {
        schema_version: 1,
        model: "ty-mcc-build-provenance",
        examination: "build_provenance",
        report: BuildProvenanceReport {
            evidence: vec![provenance.evidence_row()],
        },
    };

    let path = std::path::Path::new(&path);
    let write_result = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        serde_json::to_writer(&mut file, &record)?;
        use std::io::Write;
        file.write_all(b"\n")
    })();
    if let Err(error) = write_result {
        eprintln!(
            "Warning: failed to write ty-mcc build provenance to {}: {error}",
            path.display()
        );
    }
}
