// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (doc narrative below describes the round-1 spaced literals; the
// regression-fence tests construct them at runtime via `format!` so an
// auto-fixer cannot rewrite them into tautologies.)

//! MCC backend-evidence sidecar smoke harness. Single binary with two
//! subcommands:
//!
//! * `wrapper` — Per-case BenchKit shim. Resolves the repo-built
//!   `ty-mcc` / `ty` binary, runs it under a sidecar gate, and falls
//!   back to fail-closed `CANNOT_COMPUTE` stdout when the candidate is
//!   missing, times out, returns non-MCC output, or the sidecar fails the
//!   freshness diagnostic. The default subcommand mirrors the Python
//!   wrapper's positional CLI so existing BenchKit_head.sh invocations
//!   (`ty-mcc-smoke <input> --examination <name>`) work unchanged.
//! * `competition` — Run the focused MCC competition sidecar smoke
//!   matrix. Spawns `BenchKit_head.sh` per case with the wrapper
//!   subcommand wired in as `TY_MCC_BIN`, captures stdout/stderr/sidecar,
//!   validates the JSONL, and writes a per-case readiness matrix plus
//!   `matrix.json`.
//!
//! Routes every MCC keyword through [`tla_petri::mcc_keywords`] and every
//! examination name through [`tla_petri::examination::Examination`] —
//! the Rust enum is the only authority for the 13-examination vocabulary.
//! Killing the Python entry point removes the last cross-language drift
//! site that produced the qualification-1 keyword bug.
//!
//! ## Backend validation delegation
//!
//! Deep schema validation of the backend evidence JSONL is delegated to
//! the in-tree Rust binary `ty-mcc-backend-evidence-validate`. The
//! competition subcommand invokes it with `--json` and parses the JSON
//! summary; failures there surface as `evidence_error` in the per-case
//! report. All Python entry points have been removed (see #4509).

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::examination::Examination;
use crate::mcc_ay_pin::validate_ay_pin as ay_pin_validate;
use crate::mcc_backend_evidence_smoke::{generated_replay_smoke_rows, write_jsonl};
use crate::mcc_keywords::{
    CANNOT_COMPUTE, FORMULA, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING, STATES, STATE_SPACE,
    TECHNIQUES, TRANSITIONS,
};

// ---------- Environment / constants shared with Python ----------

const SMOKE_REAL_BIN_ENV: &str = "TY_MCC_SMOKE_REAL_BIN";
const SMOKE_DISABLE_REAL_BIN_ENV: &str = "TY_MCC_SMOKE_DISABLE_REAL_BIN";
const SMOKE_TIMEOUT_ENV: &str = "TY_MCC_SMOKE_TIMEOUT";
const BACKEND_EVIDENCE_ENV: &str = "TY_MCC_BACKEND_EVIDENCE_JSONL";
const MCC_BACKEND_EVIDENCE_ENV: &str = "MCC_BACKEND_EVIDENCE_JSONL";
const TY_MCC_BIN_ENV: &str = "TY_MCC_BIN";
const TY_MCC_STORAGE_DIR_ENV: &str = "TY_MCC_STORAGE_DIR";
const BK_EXAMINATION_ENV: &str = "BK_EXAMINATION";
const BK_INPUT_ENV: &str = "BK_INPUT";

const DEFAULT_TIMEOUT_SECONDS: f64 = 10.0;
const DEFAULT_COMPETITION_TIMEOUT_SECONDS: u64 = 15;

const BUILD_PROVENANCE_FLAG: &str = "--build-provenance-json";
const BUILD_PROVENANCE_SCHEMA: &str = "mcc.ty_mcc.build_provenance.v1";

const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA: &str =
    "ay.petri.successor.chc_model_acceptance.v1";

/// Markers / surfaces the competition harness scans for in the sidecar.
const EXECUTION_PLAN_SURFACE: &str = "petri_native_successor_execution_plan_from_trust_ir_bundle";
const CALL_PACKET_SURFACE: &str = "petri_native_successor_call_packet_from_trust_ir_bundle";
const SEMANTIC_SUCCESSOR_BRIDGE_SURFACE: &str =
    "ty.petri.native.successor.plan_cache_equivalence.v1";
const SEMANTIC_SUCCESSOR_BRIDGE_MARKER: &str = "Petri native_jit semantic_successor_bridge";
const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_MARKER: &str =
    "trust_mc_petri_successor_chc_model_acceptance";
const TRUST_CG_CALL_PACKET_PRIMITIVE_MARKER: &str = "primitive=petri_native_successor_call_packet";
const AY_TYPED_TRACE_ASSIGNMENT_PRIMITIVE_MARKER: &str = "primitive=chc_typed_trace_assignments";
const HARDWARE_REPLAY_PRIMITIVE_SCHEMA: &str = "hardware_replay_primitive/v1";
const HARDWARE_REPLAY_DECISION_MARKER: &str = "BTOR2 hardware_replay_decision";
const BLOCKER_ACTION_MARKER: &str = "MCC blocker_action";
const PORTFOLIO_ROUTE_MARKER: &str = "MCC portfolio_route";
const PORTFOLIO_ROUTE_SCHEMA: &str = "mcc.portfolio_route.v1";
const COMPILE_ARTIFACT_HANDOFF_SCHEMA: &str =
    "trust-cg.petri.native_successor.compile_artifact_handoff.v1";

const SCHEMA_DIAGNOSTIC_SUMMARY_FILENAME: &str = "sidecar-schema-diagnostic-summary.json";

/// 40-hex git revision regex (validated with `is_git_rev`). Matches the
/// Python `^[0-9a-f]{40}$` (lowercase-only) regex.
fn is_git_rev(value: &str) -> bool {
    value.len() == 40 && value.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

// ---------- CLI ----------

/// Command-line arguments for the `ty-mcc-smoke` harness.
#[derive(Parser, Debug)]
#[command(
    name = "ty-mcc-smoke",
    about = "MCC backend-evidence sidecar smoke harness (wrapper + competition).",
    long_about = "Single Rust entry point for the MCC backend-evidence sidecar smoke. \
                  \n\nWhen no subcommand is given the binary takes a positional \
                  <input> + --examination, delegates to a repo-built ty-mcc binary, \
                  writes a backend evidence sidecar, and falls back to fail-closed \
                  CANNOT_COMPUTE stdout when the \
                  candidate is missing, times out, returns non-MCC stdout, or the \
                  sidecar fails the freshness diagnostic. Use the `competition` \
                  subcommand to run the per-case BenchKit smoke matrix."
)]
pub struct Cli {
    /// Optional subcommand; absent means wrapper-compatibility mode using the
    /// top-level positional `input`/`examination`.
    #[command(subcommand)]
    pub command: Option<Cmd>,

    /// Wrapper-mode positional input (PNML model file or directory).
    ///
    /// When no subcommand is given, this argument is forwarded to the
    /// repo-built ty-mcc binary alongside `--examination`.
    #[arg(value_name = "INPUT", required = false)]
    pub input: Option<PathBuf>,

    /// Examination name (matches `Examination::from_name`).
    #[arg(long, default_value = "ReachabilityFireability")]
    pub examination: String,

    /// Pass-through arguments forwarded verbatim to the delegated
    /// candidate binary. Use `--` to separate wrapper flags from
    /// pass-through flags, mirroring the Python argparse behaviour.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub passthrough: Vec<String>,
}

/// The two `ty-mcc-smoke` operating modes.
#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Per-case BenchKit wrapper (Python wrapper.py compatibility).
    Wrapper(WrapperArgs),
    /// Run the MCC competition sidecar smoke matrix.
    Competition(CompetitionArgs),
}

/// Arguments for the per-case BenchKit `wrapper` subcommand.
#[derive(Args, Debug)]
pub struct WrapperArgs {
    /// PNML model file or directory.
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,
    /// Examination name (defaults to ReachabilityFireability).
    #[arg(long, default_value = "ReachabilityFireability")]
    pub examination: String,
    /// Pass-through arguments for the delegated binary.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub passthrough: Vec<String>,
}

/// Arguments for the `competition` sidecar smoke-matrix subcommand.
#[derive(Args, Debug)]
pub struct CompetitionArgs {
    /// MCC fixture case key (e.g. "mutex:ReachabilityFireability"). May
    /// be repeated. Defaults to `mutex:ReachabilityFireability`.
    #[arg(long)]
    pub case: Vec<String>,

    /// Directory for stdout, stderr, sidecar, and matrix.json artifacts.
    #[arg(long, value_name = "DIR")]
    pub output_dir: PathBuf,

    /// MCC command wrapper invoked through BenchKit_head.sh. Defaults to
    /// invoking this binary with the `wrapper` subcommand.
    #[arg(long)]
    pub wrapper: Option<PathBuf>,

    /// Optional repo-built ty-mcc/ty binary for the wrapper to
    /// delegate to.
    #[arg(long, value_name = "PATH")]
    pub real_bin: Option<PathBuf>,

    /// Force the wrapper's missing-binary path (fail-closed coverage).
    #[arg(long)]
    pub disable_real_binary: bool,

    /// Per-case subprocess timeout in seconds.
    #[arg(long, default_value_t = DEFAULT_COMPETITION_TIMEOUT_SECONDS)]
    pub timeout: u64,
}

// ---------- Entry points ----------

/// Entry point used by the standalone `ty-mcc-smoke` binary.
pub fn run() -> ExitCode {
    execute(Cli::parse())
}

/// Entry point used by `ty-mccctl smoke` (subcommand delegation).
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
    match dispatch(cli) {
        Ok(code) => code,
        Err(error) => {
            let _ = writeln!(io::stderr(), "ty-mcc-smoke: {error:#}");
            ExitCode::from(2)
        }
    }
}

fn dispatch(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Some(Cmd::Wrapper(args)) => wrapper_main(WrapperInvocation {
            input: args.input,
            examination: args.examination,
            passthrough: args.passthrough,
        }),
        Some(Cmd::Competition(args)) => competition_main(args),
        None => {
            // Wrapper compatibility mode: positional <input> required.
            let input = cli
                .input
                .ok_or_else(|| anyhow!("missing required <input> argument"))?;
            wrapper_main(WrapperInvocation {
                input,
                examination: cli.examination,
                passthrough: cli.passthrough,
            })
        }
    }
}

// ============================================================================
// Wrapper subcommand
// ============================================================================

struct WrapperInvocation {
    input: PathBuf,
    examination: String,
    passthrough: Vec<String>,
}

fn wrapper_main(invocation: WrapperInvocation) -> Result<ExitCode> {
    // Resolve & validate examination.
    let exam_kind = Examination::from_name(&invocation.examination)
        .map_err(|err| anyhow!("unknown examination {:?}: {}", invocation.examination, err))?;

    // Step 1: write the generated backend evidence smoke rows.
    let sidecar = backend_evidence_path()?;
    if let Err(error) = write_backend_evidence(&sidecar) {
        let _ = writeln!(
            io::stderr(),
            "failed to write backend evidence sidecar: {error}"
        );
        return Ok(ExitCode::from(2));
    }

    // Step 2: choose a real binary (or fall back).
    let real_binary = executable_candidate();
    let Some(real_binary) = real_binary else {
        let _ = writeln!(
            io::stderr(),
            "MCC sidecar smoke: repo-built MCC binary unavailable; failing closed"
        );
        write_fail_closed_stdout(&invocation.input, exam_kind)?;
        return Ok(ExitCode::SUCCESS);
    };

    match run_delegate(&real_binary, &invocation, &sidecar) {
        Ok(Some(stdout)) => {
            io::stdout().write_all(stdout.as_bytes())?;
            Ok(ExitCode::SUCCESS)
        }
        Ok(None) => {
            write_fail_closed_stdout(&invocation.input, exam_kind)?;
            Ok(ExitCode::SUCCESS)
        }
        Err(error) => {
            let _ = writeln!(io::stderr(), "MCC sidecar smoke: {error}");
            write_fail_closed_stdout(&invocation.input, exam_kind)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn backend_evidence_path() -> Result<PathBuf> {
    if let Ok(value) = env::var(BACKEND_EVIDENCE_ENV) {
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
    if let Ok(value) = env::var(MCC_BACKEND_EVIDENCE_ENV) {
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
    bail!(
        "{} or {} is required",
        BACKEND_EVIDENCE_ENV,
        MCC_BACKEND_EVIDENCE_ENV
    );
}

/// Append the generated replay smoke rows to the sidecar JSONL via the
/// in-process `tla_petri::mcc_backend_evidence_smoke` library — the same
/// source the standalone `ty-mcc-evidence-generate` binary uses. No
/// subprocess fork, no Python fallback.
fn write_backend_evidence(sidecar: &Path) -> Result<()> {
    if let Some(parent) = sidecar.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create sidecar dir {}", parent.display()))?;
    }
    let rows = generated_replay_smoke_rows();
    let mut handle = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(sidecar)
        .with_context(|| format!("open sidecar {}", sidecar.display()))?;
    write_jsonl(&rows, &mut handle)
        .with_context(|| format!("append sidecar {}", sidecar.display()))?;
    Ok(())
}

fn repo_root_from_env_or_cwd() -> Result<PathBuf> {
    if let Ok(value) = env::var("TY_MCC_SMOKE_REPO_ROOT") {
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
    // Walk up looking for a `mcc/` directory.
    let cwd = env::current_dir().context("read current dir")?;
    for ancestor in cwd.ancestors() {
        if ancestor.join("mcc").is_dir() && ancestor.join("Cargo.toml").is_file() {
            return Ok(ancestor.to_path_buf());
        }
    }
    Ok(cwd)
}

/// Resolve the repo-built `ty-mcc` / `ty` binary candidate, or
/// `None` if no candidate exists with current-HEAD provenance.
fn executable_candidate() -> Option<PathBuf> {
    if env_enabled(SMOKE_DISABLE_REAL_BIN_ENV) {
        return None;
    }
    let expected_head = repo_git_head();
    let wrapper = env::current_exe().ok().and_then(|p| p.canonicalize().ok());

    for candidate in candidate_binaries() {
        let resolved = candidate.canonicalize().unwrap_or(candidate.clone());
        if wrapper.as_ref().is_some_and(|w| w == &resolved) {
            continue;
        }
        if !is_executable(&resolved) {
            continue;
        }
        match build_provenance_diagnostic(&resolved, expected_head.as_deref()) {
            None => return Some(resolved),
            Some(diag) => {
                let _ = writeln!(
                    io::stderr(),
                    "MCC sidecar smoke: candidate build provenance rejected: {diag}"
                );
            }
        }
    }
    None
}

fn candidate_binaries() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    let push = |path: PathBuf, out: &mut Vec<PathBuf>, seen: &mut BTreeSet<PathBuf>| {
        let resolved = path.canonicalize().unwrap_or(path);
        if seen.insert(resolved.clone()) {
            out.push(resolved);
        }
    };

    if let Ok(explicit) = env::var(SMOKE_REAL_BIN_ENV) {
        if !explicit.is_empty() {
            push(PathBuf::from(explicit), &mut out, &mut seen);
            return out;
        }
    }

    let repo_root = repo_root_from_env_or_cwd().unwrap_or_else(|_| PathBuf::from("."));
    let relative = [
        "target/user/debug/ty-mcc",
        "target/user/agent/ty-mcc",
        "target/user/release/ty-mcc",
        "target/debug/ty-mcc",
        "target/agent/ty-mcc",
        "target/release/ty-mcc",
        "target/user/debug/ty",
        "target/user/release/ty",
        "target/debug/ty",
        "target/release/ty",
    ];
    for rel in relative {
        push(repo_root.join(rel), &mut out, &mut seen);
    }
    for name in ["ty-mcc", "ty"] {
        if let Some(found) = which(name) {
            push(found, &mut out, &mut seen);
        }
    }
    out
}

fn which(name: &str) -> Option<PathBuf> {
    let path_env = env::var_os("PATH")?;
    for entry in env::split_paths(&path_env) {
        let candidate = entry.join(name);
        if is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match fs::metadata(path) {
        Ok(meta) => meta.is_file() && (meta.permissions().mode() & 0o111) != 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

fn env_enabled(name: &str) -> bool {
    match env::var(name) {
        Ok(value) => {
            let trimmed = value.trim().to_ascii_lowercase();
            matches!(trimmed.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn repo_git_head() -> Option<String> {
    let output = Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let head = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if is_git_rev(&head) {
        Some(head)
    } else {
        None
    }
}

/// Verify a candidate binary advertises current-HEAD provenance via
/// `--build-provenance-json`. Returns a human-readable diagnostic on
/// mismatch, or `None` when the candidate is OK.
fn build_provenance_diagnostic(binary: &Path, expected_head: Option<&str>) -> Option<String> {
    let expected_head = expected_head.map(str::to_string).or_else(repo_git_head);
    let Some(expected_head) = expected_head else {
        return Some("unable to resolve current repository HEAD".into());
    };

    let output = match Command::new(binary)
        .arg(BUILD_PROVENANCE_FLAG)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(o) => o,
        Err(err) => {
            return Some(format!(
                "{} build provenance probe failed: {err}",
                binary.display()
            ))
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        let detail = if stderr.is_empty() {
            String::new()
        } else {
            format!(": {stderr}")
        };
        return Some(format!(
            "{} does not expose {BUILD_PROVENANCE_FLAG} (exit {:?}){detail}",
            binary.display(),
            output.status.code(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let provenance: Value = match serde_json::from_str(stdout.trim()) {
        Ok(v) => v,
        Err(err) => {
            return Some(format!(
                "{} emitted invalid build provenance JSON: {err}",
                binary.display()
            ));
        }
    };
    if !provenance.is_object() {
        return Some(format!(
            "{} build provenance JSON is not an object",
            binary.display()
        ));
    }
    let schema = provenance
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if schema != BUILD_PROVENANCE_SCHEMA {
        return Some(format!(
            "{} build provenance schema mismatch: expected {BUILD_PROVENANCE_SCHEMA}, observed {}",
            binary.display(),
            if schema.is_empty() { "missing" } else { schema },
        ));
    }
    let observed_head = provenance
        .get("build_git_head")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !is_git_rev(observed_head) {
        return Some(format!(
            "{} build provenance is missing a 40-hex build_git_head",
            binary.display()
        ));
    }
    if observed_head != expected_head {
        return Some(format!(
            "{} build_git_head is not current HEAD: expected {expected_head}, observed {observed_head}",
            binary.display(),
        ));
    }
    None
}

// ---------- delegate ----------

const DIRECT_MCC_TOOL_NAMES: &[&str] = &["ty-mcc", "pnml-tools"];

fn run_delegate(
    binary: &Path,
    invocation: &WrapperInvocation,
    sidecar: &Path,
) -> Result<Option<String>> {
    let _ = writeln!(
        io::stderr(),
        "MCC sidecar smoke: delegating to repo-built binary {}",
        binary.display()
    );
    let mut cmd = Command::new(binary);
    let bin_name = binary
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    if !DIRECT_MCC_TOOL_NAMES.contains(&bin_name) {
        cmd.arg("mcc");
    }
    cmd.arg(&invocation.input)
        .arg("--examination")
        .arg(&invocation.examination);
    for arg in &invocation.passthrough {
        cmd.arg(arg);
    }
    cmd.env(BACKEND_EVIDENCE_ENV, sidecar);
    cmd.env(MCC_BACKEND_EVIDENCE_ENV, sidecar);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let timeout_seconds = timeout_seconds();
    let (status_code, timed_out, stdout, stderr) = match run_with_timeout(cmd, timeout_seconds) {
        Ok(v) => v,
        Err(err) => bail!("invoke {}: {err}", binary.display()),
    };
    if !stderr.is_empty() {
        let _ = io::stderr().write_all(&stderr);
    }
    if timed_out {
        let _ = writeln!(
            io::stderr(),
            "MCC sidecar smoke: delegated binary timed out; failing closed"
        );
        return Ok(None);
    }
    if status_code != 0 {
        let _ = writeln!(
            io::stderr(),
            "MCC sidecar smoke: delegated binary exited {status_code}; failing closed"
        );
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&stdout).into_owned();
    if !official_stdout(&stdout) {
        let _ = writeln!(
            io::stderr(),
            "MCC sidecar smoke: delegated binary produced non-MCC stdout; failing closed"
        );
        return Ok(None);
    }
    if let Some(diag) = sidecar_freshness_diagnostic(sidecar, None, repo_git_head().as_deref()) {
        let _ = writeln!(
            io::stderr(),
            "MCC sidecar smoke: sidecar freshness gate failed: {diag}; failing closed"
        );
        return Ok(None);
    }
    Ok(Some(stdout))
}

fn timeout_seconds() -> Duration {
    let raw = match env::var(SMOKE_TIMEOUT_ENV) {
        Ok(v) => v,
        Err(_) => return Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS),
    };
    let parsed: f64 = match raw.parse() {
        Ok(v) => v,
        Err(_) => return Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS),
    };
    if parsed <= 0.0 {
        Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS)
    } else {
        Duration::from_secs_f64(parsed)
    }
}

/// Run a `Command` with a wall-clock timeout. Returns
/// `(exit_code, timed_out, stdout, stderr)`. On timeout the child is
/// killed.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<(i32, bool, Vec<u8>, Vec<u8>)> {
    let mut child = cmd.spawn().context("spawn delegated binary")?;
    let start = Instant::now();
    let interval = Duration::from_millis(50);
    let exit_status = loop {
        match child.try_wait().context("wait on delegated binary")? {
            Some(status) => break Some(status),
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(interval);
            }
        }
    };

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(mut handle) = child.stdout.take() {
        let _ = handle.read_to_end(&mut stdout);
    }
    if let Some(mut handle) = child.stderr.take() {
        let _ = handle.read_to_end(&mut stderr);
    }

    let (code, timed_out) = match exit_status {
        Some(status) => (status.code().unwrap_or(-1), false),
        None => (-1, true),
    };
    Ok((code, timed_out, stdout, stderr))
}

// ---------- official stdout shape ----------

/// Returns true iff every non-empty line in `stdout` matches one of the
/// canonical MCC answer forms: FORMULA, STATE_SPACE metric, or
/// `STATE_SPACE CANNOT_COMPUTE`. Mirrors the Python `official_stdout`.
pub(crate) fn official_stdout(stdout: &str) -> bool {
    let mut any = false;
    for raw in stdout.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        any = true;
        if !is_state_space_cannot_compute_line(line)
            && !is_state_space_metric_line(line)
            && !is_formula_line(line)
        {
            return false;
        }
    }
    any
}

fn split_with_techniques(line: &str) -> Option<(&str, &str)> {
    let key = TECHNIQUES;
    let idx = line.find(key)?;
    let head = line[..idx].trim_end();
    let tail = line[idx + key.len()..].trim();
    if tail.is_empty() {
        None
    } else {
        Some((head, tail))
    }
}

fn is_formula_line(line: &str) -> bool {
    let Some((head, _techniques)) = split_with_techniques(line) else {
        return false;
    };
    let tokens: Vec<&str> = head.split_whitespace().collect();
    if tokens.len() != 3 {
        return false;
    }
    if tokens[0] != FORMULA {
        return false;
    }
    let answer = tokens[2];
    answer == "TRUE"
        || answer == "FALSE"
        || answer == CANNOT_COMPUTE
        || answer.chars().all(|c| c.is_ascii_digit()) && !answer.is_empty()
}

fn is_state_space_metric_line(line: &str) -> bool {
    let Some((head, _)) = split_with_techniques(line) else {
        return false;
    };
    let tokens: Vec<&str> = head.split_whitespace().collect();
    if tokens.len() != 3 {
        return false;
    }
    if tokens[0] != STATE_SPACE {
        return false;
    }
    let metric_ok = matches!(
        tokens[1],
        STATES | TRANSITIONS | MAX_TOKEN_IN_PLACE | MAX_TOKEN_PER_MARKING
    );
    if !metric_ok {
        return false;
    }
    !tokens[2].is_empty() && tokens[2].chars().all(|c| c.is_ascii_digit())
}

fn is_state_space_cannot_compute_line(line: &str) -> bool {
    let Some((head, _)) = split_with_techniques(line) else {
        return false;
    };
    let tokens: Vec<&str> = head.split_whitespace().collect();
    tokens.len() == 2 && tokens[0] == STATE_SPACE && tokens[1] == CANNOT_COMPUTE
}

// ---------- fail-closed stdout ----------

fn write_fail_closed_stdout(input: &Path, exam: Examination) -> Result<()> {
    let text = fail_closed_stdout(input, exam);
    io::stdout().write_all(text.as_bytes())?;
    Ok(())
}

fn fail_closed_stdout(input: &Path, exam: Examination) -> String {
    if matches!(exam, Examination::StateSpace) {
        return format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n");
    }
    let formula_ids = property_ids_for_examination(input, exam);
    let ids = if formula_ids.is_empty() {
        vec![default_formula_id(exam).to_string()]
    } else {
        formula_ids
    };
    let mut out = String::new();
    for id in ids {
        out.push_str(&format!(
            "{FORMULA} {id} {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n"
        ));
    }
    out
}

fn default_formula_id(exam: Examination) -> &'static str {
    match exam {
        Examination::ReachabilityFireability => "Mutex-ReachabilityFireability-00",
        other => other.as_str(),
    }
}

fn model_directory(input: &Path) -> PathBuf {
    if input.is_dir() {
        return input.to_path_buf();
    }
    if input.is_file() {
        return input
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| input.to_path_buf());
    }
    if input.extension().is_some() {
        return input
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| input.to_path_buf());
    }
    input.to_path_buf()
}

fn property_ids_for_examination(input: &Path, exam: Examination) -> Vec<String> {
    let dir = model_directory(input);
    let path = dir.join(format!("{}.xml", exam.as_str()));
    let Ok(text) = fs::read_to_string(&path) else {
        return Vec::new();
    };
    // Minimal property-id parse: scan for <id>…</id> within <property> blocks.
    parse_property_ids(&text)
}

fn parse_property_ids(text: &str) -> Vec<String> {
    let mut ids = Vec::new();
    let mut depth_in_property: i32 = 0;
    let mut idx = 0;
    let bytes = text.as_bytes();
    while idx < bytes.len() {
        if depth_in_property == 0 {
            if let Some(open) = find_tag(&text[idx..], "property", false) {
                depth_in_property = 1;
                idx += open.end;
                continue;
            }
            break;
        }
        // Look for </property> closer OR a child <id> tag.
        if let Some(close) = find_tag(&text[idx..], "property", true) {
            if let Some(id_open) = find_tag(&text[idx..], "id", false) {
                if id_open.start < close.start {
                    let after = idx + id_open.end;
                    if let Some(end_id) = find_tag(&text[after..], "id", true) {
                        let value = text[after..after + end_id.start].trim();
                        if !value.is_empty() {
                            ids.push(value.to_string());
                        }
                        idx = after + end_id.end;
                        depth_in_property = 0; // first <id> in property wins
                        continue;
                    }
                }
            }
            idx += close.end;
            depth_in_property = 0;
        } else {
            break;
        }
    }
    ids
}

#[derive(Debug, Clone, Copy)]
struct TagSpan {
    start: usize,
    end: usize,
}

/// Find the next opening (`<name`) or closing (`</name>`) tag with the
/// matching local name, ignoring namespace prefixes (so `<x:id>` matches
/// `name="id"`). Returns the byte span of the tag.
fn find_tag(text: &str, name: &str, closing: bool) -> Option<TagSpan> {
    let bytes = text.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'<' {
            let mut cursor = idx + 1;
            let is_close = bytes.get(cursor) == Some(&b'/');
            if is_close {
                cursor += 1;
            }
            if is_close != closing {
                idx = cursor;
                continue;
            }
            // skip namespace prefix
            let local_start = cursor;
            while cursor < bytes.len() && bytes[cursor] != b':' && !is_tag_terminator(bytes[cursor])
            {
                cursor += 1;
            }
            let local_name = if cursor < bytes.len() && bytes[cursor] == b':' {
                cursor += 1;
                let local_start2 = cursor;
                while cursor < bytes.len() && !is_tag_terminator(bytes[cursor]) {
                    cursor += 1;
                }
                &text[local_start2..cursor]
            } else {
                &text[local_start..cursor]
            };
            if local_name == name {
                // Find end of tag '>'.
                while cursor < bytes.len() && bytes[cursor] != b'>' {
                    cursor += 1;
                }
                if cursor < bytes.len() {
                    cursor += 1;
                }
                return Some(TagSpan {
                    start: idx,
                    end: cursor,
                });
            }
            idx = cursor.max(idx + 1);
        } else {
            idx += 1;
        }
    }
    None
}

fn is_tag_terminator(byte: u8) -> bool {
    byte == b' ' || byte == b'>' || byte == b'/' || byte == b'\t' || byte == b'\n' || byte == b'\r'
}

// ---------- sidecar freshness diagnostic ----------

const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA_PATTERN: &str =
    AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA;

/// Return a fail-closed diagnostic when sidecar evidence is not current.
///
/// The AY revision is resolved in-process via
/// [`tla_petri::mcc_ay_pin::validate_ay_pin`] — same source the standalone
/// `ty-mcc-ay-pin-validate` binary uses. No Python shell-out.
fn sidecar_freshness_diagnostic(
    sidecar: &Path,
    expected_ay_rev: Option<&str>,
    expected_build_head: Option<&str>,
) -> Option<String> {
    let metadata = match fs::metadata(sidecar) {
        Ok(m) => m,
        Err(_) => {
            return Some(format!(
                "backend evidence sidecar is missing or empty: {}",
                sidecar.display()
            ));
        }
    };
    if metadata.len() == 0 {
        return Some(format!(
            "backend evidence sidecar is missing or empty: {}",
            sidecar.display()
        ));
    }

    let expected_ay_rev_owned;
    let expected_ay_rev = if let Some(rev) = expected_ay_rev {
        rev
    } else {
        match pinned_ay_rev() {
            Ok(rev) => {
                expected_ay_rev_owned = rev;
                expected_ay_rev_owned.as_str()
            }
            Err(error) => {
                return Some(format!("unable to resolve pinned AY revision: {error}"));
            }
        }
    };

    let text = match fs::read_to_string(sidecar) {
        Ok(t) => t,
        Err(error) => {
            return Some(format!(
                "unable to read backend evidence sidecar {}: {error}",
                sidecar.display()
            ));
        }
    };

    let mut current_ay_revs: Vec<String> = Vec::new();
    let mut build_git_heads: Vec<String> = Vec::new();
    let mut acceptance_rows: usize = 0;
    for (line_number, raw_line) in text.lines().enumerate() {
        let line_number = line_number + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let record: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(error) => {
                return Some(format!(
                    "backend evidence sidecar has invalid JSON at line {line_number}: {error}"
                ));
            }
        };
        let Value::Object(_) = record else {
            return Some(format!(
                "backend evidence sidecar line {line_number} is not a JSON object"
            ));
        };
        for evidence_line in collect_evidence_lines(&record) {
            let data = parse_evidence_tokens(&evidence_line);
            if let Some(schema) = data.get("schema") {
                if schema == AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA_PATTERN {
                    acceptance_rows += 1;
                }
            }
            if let Some(rev) = data.get("current_ay_rev") {
                let trimmed = rev.trim();
                if !trimmed.is_empty() && trimmed != "missing" && trimmed != "none" {
                    current_ay_revs.push(trimmed.to_string());
                }
            }
            let is_build_provenance = data.get("schema").map(String::as_str)
                == Some(BUILD_PROVENANCE_SCHEMA)
                || (data.get("scope").map(String::as_str) == Some("MCC")
                    && data.get("component").map(String::as_str)
                        == Some("ty_mcc_build_provenance"));
            if is_build_provenance {
                if let Some(head) = data.get("build_git_head") {
                    let trimmed = head.trim();
                    if !trimmed.is_empty() && trimmed != "missing" && trimmed != "none" {
                        build_git_heads.push(trimmed.to_string());
                    }
                }
            }
        }
    }

    if acceptance_rows == 0 {
        return Some(format!(
            "backend evidence sidecar is missing {AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA_PATTERN} schema evidence"
        ));
    }
    if current_ay_revs.is_empty() {
        return Some("backend evidence sidecar is missing current_ay_rev evidence".into());
    }
    let mut unique_revs: Vec<String> = current_ay_revs;
    unique_revs.sort();
    unique_revs.dedup();
    if unique_revs.len() != 1 {
        return Some(format!(
            "backend evidence sidecar has mixed current_ay_rev values: {}",
            unique_revs.join(", "),
        ));
    }
    let observed = &unique_revs[0];
    if observed != expected_ay_rev {
        return Some(format!(
            "backend evidence sidecar current_ay_rev is stale: expected {expected_ay_rev}, observed {observed}"
        ));
    }

    if let Some(expected_head) = expected_build_head {
        if !is_git_rev(expected_head) {
            return Some(format!("invalid expected build_git_head: {expected_head}"));
        }
        if build_git_heads.is_empty() {
            return Some("backend evidence sidecar is missing ty-mcc build provenance".into());
        }
        let mut invalid: Vec<String> = build_git_heads
            .iter()
            .filter(|head| !is_git_rev(head))
            .cloned()
            .collect();
        invalid.sort();
        invalid.dedup();
        if !invalid.is_empty() {
            return Some(format!(
                "backend evidence sidecar has invalid build_git_head values: {}",
                invalid.join(", ")
            ));
        }
        let mut unique_heads: Vec<String> = build_git_heads;
        unique_heads.sort();
        unique_heads.dedup();
        if unique_heads.as_slice() != [expected_head.to_string()].as_slice() {
            return Some(format!(
                "backend evidence sidecar build_git_head is not current HEAD: expected {expected_head}, observed {}",
                unique_heads.join(", ")
            ));
        }
    }

    None
}

fn pinned_ay_rev() -> Result<String> {
    let repo_root = repo_root_from_env_or_cwd()?;
    let summary =
        ay_pin_validate(&repo_root, None).map_err(|err| anyhow!("validate_ay_pin: {err}"))?;
    Ok(summary.dockerfile_rev)
}

fn collect_evidence_lines(record: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let push_from = |container: &Value, out: &mut Vec<String>| {
        let Some(value) = container.get("evidence") else {
            return;
        };
        match value {
            Value::Array(rows) => {
                for row in rows {
                    if let Some(s) = row.as_str() {
                        if !s.trim().is_empty() {
                            out.push(s.to_string());
                        }
                    } else {
                        let s = row.to_string();
                        if !s.trim().is_empty() {
                            out.push(s);
                        }
                    }
                }
            }
            Value::String(s) if !s.trim().is_empty() => out.push(s.clone()),
            _ => {}
        }
    };
    push_from(record, &mut out);
    for key in [
        "report",
        "capability_report",
        "backend_capability_report",
        "backendCapabilityReport",
        "capability",
    ] {
        if let Some(child) = record.get(key) {
            if child.is_object() {
                push_from(child, &mut out);
            }
        }
    }
    out
}

/// Tokenize an evidence line via shlex-compatible parsing.
///
/// Tokens without `=` become `scope` and `component`, every `k=v` token
/// becomes a map entry.
fn parse_evidence_tokens(row: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let tokens = shlex_split(row);
    if let Some(first) = tokens.first() {
        out.insert("scope".into(), first.clone());
    }
    if let Some(second) = tokens.get(1) {
        if !second.contains('=') {
            out.insert("component".into(), second.clone());
        }
    }
    for token in &tokens {
        if let Some((k, v)) = token.split_once('=') {
            if !k.is_empty() {
                out.insert(k.to_string(), v.to_string());
            }
        }
    }
    out
}

/// Minimal shell-style tokenizer compatible with `shlex.split` for the
/// subset of inputs this harness sees (whitespace-separated, optionally
/// single- or double-quoted, backslash escapes inside double quotes).
fn shlex_split(input: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = input.chars().peekable();
    let mut have_token = false;
    while let Some(ch) = chars.next() {
        if !in_single && !in_double && ch.is_whitespace() {
            if have_token {
                out.push(std::mem::take(&mut buf));
                have_token = false;
            }
            continue;
        }
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                have_token = true;
            }
            '"' if !in_single => {
                in_double = !in_double;
                have_token = true;
            }
            '\\' if in_double => {
                if let Some(next) = chars.next() {
                    buf.push(next);
                }
                have_token = true;
            }
            other => {
                buf.push(other);
                have_token = true;
            }
        }
    }
    if have_token {
        out.push(buf);
    }
    out
}

// ============================================================================
// Competition subcommand
// ============================================================================

/// Built-in MCC fixture cases, mirroring the Python `CASES` dict.
fn competition_cases(repo_root: &Path) -> BTreeMap<&'static str, CompetitionCase> {
    let mut out = BTreeMap::new();
    out.insert(
        "mutex:ReachabilityFireability",
        CompetitionCase {
            code: "mutex-ReachabilityFireability",
            input_path: repo_root
                .join("tests")
                .join("mcc_benchmarks")
                .join("mutex")
                .join("model.pnml"),
            examination: Examination::ReachabilityFireability,
        },
    );
    out
}

#[derive(Debug, Clone)]
struct CompetitionCase {
    code: &'static str,
    input_path: PathBuf,
    examination: Examination,
}

fn competition_main(args: CompetitionArgs) -> Result<ExitCode> {
    let repo_root = repo_root_from_env_or_cwd()?;
    let cases_index = competition_cases(&repo_root);

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create output dir {}", args.output_dir.display()))?;

    let case_keys: Vec<String> = if args.case.is_empty() {
        vec!["mutex:ReachabilityFireability".into()]
    } else {
        args.case.clone()
    };

    let wrapper_path = match &args.wrapper {
        Some(path) => path.clone(),
        None => self_invocation_wrapper_target()?,
    };

    let mut case_results: Vec<Value> = Vec::new();
    for key in &case_keys {
        let case = cases_index
            .get(key.as_str())
            .ok_or_else(|| anyhow!("unknown case key: {key}"))?
            .clone();
        let case_result = run_case(&args, &wrapper_path, &case, &repo_root)?;
        case_results.push(case_result);
    }

    let status = if case_results
        .iter()
        .all(|c| c.get("status").and_then(Value::as_str) == Some("ok"))
    {
        "ok"
    } else {
        "failed"
    };

    let matrix = json!({
        "status": status,
        "required_evidence": required_evidence(),
        "required_evidence_markers": required_evidence_markers(),
        "cases": case_results,
    });
    let matrix_path = args.output_dir.join("matrix.json");
    let matrix_text = serde_json::to_string_pretty(&matrix)? + "\n";
    fs::write(&matrix_path, matrix_text.as_bytes())
        .with_context(|| format!("write {}", matrix_path.display()))?;

    let mut combined = matrix.as_object().cloned().unwrap_or_default();
    combined.insert(
        "matrix".into(),
        Value::String(matrix_path.display().to_string()),
    );
    println!("{}", serde_json::to_string(&Value::Object(combined))?);

    Ok(if status == "ok" {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

/// Returns the wrapper invocation target the competition subcommand
/// hands to BenchKit_head.sh. Defaults to `<this-binary> wrapper`,
/// matching the user's "single interface" requirement.
fn self_invocation_wrapper_target() -> Result<PathBuf> {
    let exe = env::current_exe().context("resolve current_exe")?;
    Ok(exe)
}

fn required_evidence() -> Vec<&'static str> {
    vec![
        "mcc_ay_symbolic_execution",
        "native_jit_fail_closed_gate",
        "trust_ir_transport_identity",
        "trust_cg_native_admission",
        "ay_solve_decision_profile",
        "hardware_proof_replay_boundary",
        "hardware_replay_decision",
        "trust_cg_compile_artifact_cache_telemetry",
        "trust_cg_host_jit_pgo_provenance",
        "trust_cg_call_packet_contract_descriptor",
        "production_selector_decision",
        "portfolio_route",
        "ay_solver_capability_descriptor",
        "ay_symbolic_execution_contract_manifest",
        "trust_ir_native_evidence_artifact_resolution",
        "trust_ir_native_verification_bundle_handoff",
        "trust_ir_native_semantic_bridge_proof_identity",
        "trust_ir_petri_proof_evidence_identity",
        "trust_ir_native_verification_bundle_handoff_replay_contract_surface",
        "trust_ir_native_verification_bundle_handoff_replay_contract_report_identity",
        "trust_ir_native_verification_bundle_handoff_replay_contract_json_manifest_binding",
        "semantic_successor_bridge",
        "petri_trust_mc_model_acceptance",
    ]
}

fn required_evidence_markers() -> Vec<&'static str> {
    vec![
        EXECUTION_PLAN_SURFACE,
        CALL_PACKET_SURFACE,
        COMPILE_ARTIFACT_HANDOFF_SCHEMA,
        SEMANTIC_SUCCESSOR_BRIDGE_MARKER,
        SEMANTIC_SUCCESSOR_BRIDGE_SURFACE,
        AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_MARKER,
        AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA,
        TRUST_CG_CALL_PACKET_PRIMITIVE_MARKER,
        AY_TYPED_TRACE_ASSIGNMENT_PRIMITIVE_MARKER,
        HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
        HARDWARE_REPLAY_DECISION_MARKER,
        BLOCKER_ACTION_MARKER,
        PORTFOLIO_ROUTE_MARKER,
        PORTFOLIO_ROUTE_SCHEMA,
    ]
}

fn run_case(
    args: &CompetitionArgs,
    wrapper_path: &Path,
    case: &CompetitionCase,
    repo_root: &Path,
) -> Result<Value> {
    let case_dir = args.output_dir.join(case.code);
    fs::create_dir_all(&case_dir).with_context(|| format!("create {}", case_dir.display()))?;
    let sidecar = case_dir.join("backend-capability.jsonl");
    let stdout_path = case_dir.join("stdout.txt");
    let stderr_path = case_dir.join("stderr.txt");
    if sidecar.exists() {
        let _ = fs::remove_file(&sidecar);
    }

    let env_map = build_case_env(args, wrapper_path, case, &case_dir, &sidecar);
    let mut cmd = Command::new("bash");
    cmd.arg(repo_root.join("mcc").join("BenchKit_head.sh"));
    cmd.current_dir(&case_dir);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    for (k, v) in &env_map {
        cmd.env(k, v);
    }

    let timeout = Duration::from_secs(args.timeout + 5);
    let (rc, timed_out, stdout, stderr) = run_with_timeout(cmd, timeout)?;
    let stdout_text = String::from_utf8_lossy(&stdout).into_owned();
    let stderr_text = String::from_utf8_lossy(&stderr).into_owned();
    fs::write(&stdout_path, &stdout_text)?;
    fs::write(&stderr_path, &stderr_text)?;

    let mut errors: Vec<String> = Vec::new();
    if rc != 0 {
        errors.push(format!("BenchKit_head.sh exited {rc}"));
    }
    if timed_out {
        errors.push("BenchKit_head.sh timed out".into());
    }
    let stdout_is_official = official_stdout(&stdout_text);
    if !stdout_is_official {
        errors.push("stdout is not official MCC output".into());
    }

    // Validate sidecar (delegated to python module until ported).
    let (evidence_summary, evidence_error) = validate_sidecar(&sidecar, repo_root);
    if let Some(ref msg) = evidence_error {
        errors.push(format!("backend evidence validation failed: {msg}"));
    }

    // Marker scan over the sidecar.
    let sidecar_text = fs::read_to_string(&sidecar).unwrap_or_default();
    let marker_present = |needle: &str| sidecar_text.contains(needle);
    let execution_plan_acknowledged = marker_present(EXECUTION_PLAN_SURFACE);
    let call_packet_acknowledged = marker_present(CALL_PACKET_SURFACE);
    let trust_cg_call_packet_primitive_acknowledged =
        marker_present(TRUST_CG_CALL_PACKET_PRIMITIVE_MARKER);
    let ay_typed_trace_assignment_acknowledged =
        marker_present(AY_TYPED_TRACE_ASSIGNMENT_PRIMITIVE_MARKER);
    let hardware_replay_primitive_acknowledged = marker_present(HARDWARE_REPLAY_PRIMITIVE_SCHEMA);
    let semantic_successor_bridge_acknowledged = marker_present(SEMANTIC_SUCCESSOR_BRIDGE_MARKER);
    let blocker_action_acknowledged = marker_present(BLOCKER_ACTION_MARKER);
    let portfolio_route_acknowledged = marker_present(PORTFOLIO_ROUTE_MARKER);
    let ay_model_acceptance_acknowledged =
        marker_present(AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_MARKER);

    if !execution_plan_acknowledged {
        errors.push(
            "backend evidence missing trust-codegen Petri native successor execution-plan surface"
                .into(),
        );
    }
    if !call_packet_acknowledged {
        errors.push(
            "backend evidence missing trust-codegen Petri native successor call-packet surface"
                .into(),
        );
    }
    if !trust_cg_call_packet_primitive_acknowledged {
        errors.push(
            "backend evidence missing trust-codegen call-packet shared primitive availability"
                .into(),
        );
    }
    if !ay_typed_trace_assignment_acknowledged {
        errors.push(
            "backend evidence missing AY typed trace assignment shared primitive availability"
                .into(),
        );
    }
    if !hardware_replay_primitive_acknowledged {
        errors.push("backend evidence missing hardware replay primitive schema".into());
    }
    if !semantic_successor_bridge_acknowledged {
        errors.push("backend evidence missing Petri semantic successor bridge".into());
    }
    if !blocker_action_acknowledged {
        errors.push("backend evidence missing MCC blocker action gate".into());
    }
    if !portfolio_route_acknowledged {
        errors.push("backend evidence missing MCC portfolio route evidence".into());
    }

    let schema_diagnostic_summary_path = case_dir.join(SCHEMA_DIAGNOSTIC_SUMMARY_FILENAME);
    let schema_diagnostic_summary =
        write_schema_diagnostic_summary(&sidecar_text, &schema_diagnostic_summary_path)?;

    let readiness = build_readiness_matrix(
        &stdout_text,
        evidence_summary.as_ref(),
        evidence_error.as_deref(),
        &MarkerAcknowledgements {
            execution_plan: execution_plan_acknowledged,
            call_packet: call_packet_acknowledged,
            trust_cg_call_packet_primitive: trust_cg_call_packet_primitive_acknowledged,
            ay_typed_trace_assignment: ay_typed_trace_assignment_acknowledged,
            hardware_replay_primitive: hardware_replay_primitive_acknowledged,
            semantic_successor_bridge: semantic_successor_bridge_acknowledged,
            blocker_action: blocker_action_acknowledged,
            portfolio_route: portfolio_route_acknowledged,
            ay_model_acceptance: ay_model_acceptance_acknowledged,
        },
    );

    let status = if errors.is_empty() { "ok" } else { "failed" };

    Ok(json!({
        "case": case.code,
        "input": case.input_path.display().to_string(),
        "examination": case.examination.as_str(),
        "returncode": rc,
        "status": status,
        "errors": errors,
        "stdout_official": stdout_is_official,
        "fail_closed": stdout_text.contains(CANNOT_COMPUTE),
        "stdout_path": stdout_path.display().to_string(),
        "stderr_path": stderr_path.display().to_string(),
        "backend_evidence_jsonl": sidecar.display().to_string(),
        "schema_diagnostic_summary_path": schema_diagnostic_summary_path.display().to_string(),
        "schema_diagnostic_summary": schema_diagnostic_summary,
        "evidence_summary": evidence_summary,
        "readiness": readiness,
        "trust_cg_execution_plan_surface":
            if execution_plan_acknowledged { Value::String(EXECUTION_PLAN_SURFACE.into()) } else { Value::Null },
        "trust_cg_call_packet_surface":
            if call_packet_acknowledged { Value::String(CALL_PACKET_SURFACE.into()) } else { Value::Null },
        "trust_cg_call_packet_primitive":
            if trust_cg_call_packet_primitive_acknowledged {
                Value::String("petri_native_successor_call_packet".into())
            } else { Value::Null },
        "ay_typed_trace_assignment_surface":
            if ay_typed_trace_assignment_acknowledged {
                Value::String("ay_chc_typed_trace_assignments".into())
            } else { Value::Null },
        "ay_petri_trust_mc_model_acceptance_schema":
            if ay_model_acceptance_acknowledged {
                Value::String(AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA.into())
            } else { Value::Null },
        "hardware_replay_primitive_schema":
            if hardware_replay_primitive_acknowledged {
                Value::String(HARDWARE_REPLAY_PRIMITIVE_SCHEMA.into())
            } else { Value::Null },
    }))
}

fn build_case_env(
    args: &CompetitionArgs,
    wrapper_path: &Path,
    case: &CompetitionCase,
    case_dir: &Path,
    sidecar: &Path,
) -> Vec<(OsString, OsString)> {
    let mut out: Vec<(OsString, OsString)> = vec![
        (BK_EXAMINATION_ENV.into(), case.examination.as_str().into()),
        (BK_INPUT_ENV.into(), case.input_path.as_os_str().to_owned()),
        (TY_MCC_BIN_ENV.into(), wrapper_path.as_os_str().to_owned()),
        (
            TY_MCC_STORAGE_DIR_ENV.into(),
            case_dir.join("storage").into_os_string(),
        ),
        (BACKEND_EVIDENCE_ENV.into(), sidecar.as_os_str().to_owned()),
        (
            MCC_BACKEND_EVIDENCE_ENV.into(),
            sidecar.as_os_str().to_owned(),
        ),
        (SMOKE_TIMEOUT_ENV.into(), args.timeout.to_string().into()),
    ];
    if let Some(real_bin) = &args.real_bin {
        out.push((SMOKE_REAL_BIN_ENV.into(), real_bin.as_os_str().to_owned()));
    } else if args.disable_real_binary {
        // The Python harness pops SMOKE_REAL_BIN_ENV; in our env vector we
        // simply do not set it.
    }
    if args.disable_real_binary {
        out.push((SMOKE_DISABLE_REAL_BIN_ENV.into(), "1".into()));
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct MarkerAcknowledgements {
    execution_plan: bool,
    call_packet: bool,
    trust_cg_call_packet_primitive: bool,
    ay_typed_trace_assignment: bool,
    hardware_replay_primitive: bool,
    semantic_successor_bridge: bool,
    blocker_action: bool,
    portfolio_route: bool,
    ay_model_acceptance: bool,
}

/// Validate the sidecar via the in-tree
/// `ty-mcc-backend-evidence-validate` Rust binary. Returns
/// `(summary, error)` so failures surface as `evidence_error` in the
/// per-case report. Fails closed when the binary is not on PATH or in
/// `TY_MCC_BACKEND_EVIDENCE_VALIDATE_BIN`.
fn validate_sidecar(sidecar: &Path, repo_root: &Path) -> (Option<Value>, Option<String>) {
    if !sidecar.is_file() {
        return (
            None,
            Some(format!("sidecar missing: {}", sidecar.display())),
        );
    }
    let pinned = env::var_os("TY_MCC_BACKEND_EVIDENCE_VALIDATE_BIN")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let bin = match pinned.or_else(|| which("ty-mcc-backend-evidence-validate")) {
        Some(b) => b,
        None => {
            return (
                None,
                Some(
                    "ty-mcc-backend-evidence-validate binary not found on PATH \
                     (set TY_MCC_BACKEND_EVIDENCE_VALIDATE_BIN)"
                        .to_string(),
                ),
            );
        }
    };
    let mut cmd = Command::new(&bin);
    cmd.arg("--json");
    for check in required_evidence() {
        cmd.arg("--require").arg(check);
    }
    cmd.arg(sidecar);
    cmd.current_dir(repo_root);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let output = match cmd.output() {
        Ok(o) => o,
        Err(error) => {
            return (None, Some(format!("invoke {}: {error}", bin.display())));
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return (None, Some(stderr));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    match serde_json::from_str::<Value>(stdout.trim()) {
        Ok(v) => (Some(v), None),
        Err(error) => (None, Some(format!("parse {} JSON: {error}", bin.display()))),
    }
}

fn build_readiness_matrix(
    stdout: &str,
    evidence_summary: Option<&Value>,
    evidence_error: Option<&str>,
    markers: &MarkerAcknowledgements,
) -> Value {
    let stdout_official = official_stdout(stdout);
    let rows = parse_official_stdout_rows(stdout);
    let cannot_compute_rows: Vec<String> = rows
        .iter()
        .filter(|row| row.answer.as_deref() == Some(CANNOT_COMPUTE))
        .filter_map(|row| row.id.clone())
        .collect();

    let counts = evidence_summary
        .and_then(|v| v.get("counts"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let count_of = |name: &str| -> u64 { counts.get(name).and_then(Value::as_u64).unwrap_or(0) };

    let replay_count = count_of("hardware_proof_replay_boundary");
    let replay_decision_count = count_of("hardware_replay_decision");
    let semantic_bridge_count = count_of("semantic_successor_bridge");
    let portfolio_route_count = count_of("portfolio_route");
    let mcc_symbolic_count = count_of("mcc_ay_symbolic_execution");
    let ay_solve_decision_count = count_of("ay_solve_decision_profile");

    let sidecar_replay_status =
        if evidence_error.is_none() && replay_count >= 2 && replay_decision_count >= 1 {
            "pass"
        } else {
            "failed"
        };

    let ay_evidence_status = if mcc_symbolic_count > 0 && ay_solve_decision_count > 0 {
        "pass"
    } else {
        "failed"
    };

    let actual_status = if !stdout_official {
        "failed"
    } else if !cannot_compute_rows.is_empty() {
        "blocked"
    } else if !markers.execution_plan || !markers.call_packet {
        "blocked"
    } else {
        "ready"
    };

    json!({
        "official_stdout": {
            "status": if stdout_official { "pass" } else { "failed" },
            "row_count": rows.len(),
            "rows": rows.iter().map(StdoutRow::to_json).collect::<Vec<_>>(),
        },
        "sidecar_replay": {
            "status": sidecar_replay_status,
            "error": evidence_error,
            "rows": evidence_summary
                .and_then(|v| v.get("rows").cloned())
                .unwrap_or(Value::from(0u64)),
            "hardware_proof_replay_boundary": replay_count,
            "hardware_replay_decision": replay_decision_count,
            "semantic_successor_bridge": semantic_bridge_count,
            "portfolio_route": portfolio_route_count,
            "required": evidence_summary.and_then(|v| v.get("required").cloned()).unwrap_or(Value::Array(vec![])),
        },
        "ay_evidence": {
            "status": ay_evidence_status,
            "mcc_symbolic_execution": mcc_symbolic_count,
            "ay_solve_decision_profile": ay_solve_decision_count,
        },
        "semantic_successor_bridge_acknowledged": markers.semantic_successor_bridge,
        "ay_model_acceptance_acknowledged": markers.ay_model_acceptance,
        "execution_plan_acknowledged": markers.execution_plan,
        "call_packet_acknowledged": markers.call_packet,
        "trust_cg_call_packet_primitive_acknowledged": markers.trust_cg_call_packet_primitive,
        "ay_typed_trace_assignment_acknowledged": markers.ay_typed_trace_assignment,
        "hardware_replay_primitive_acknowledged": markers.hardware_replay_primitive,
        "blocker_action_acknowledged": markers.blocker_action,
        "portfolio_route_acknowledged": markers.portfolio_route,
        "actual_answers": {
            "status": actual_status,
            "blocked_rows": cannot_compute_rows,
        },
    })
}

#[derive(Debug, Clone, Serialize)]
struct StdoutRow {
    index: usize,
    kind: String,
    id: Option<String>,
    answer: Option<String>,
    techniques: Option<String>,
    raw: String,
}

impl StdoutRow {
    fn to_json(&self) -> Value {
        json!({
            "index": self.index,
            "kind": self.kind,
            "id": self.id,
            "answer": self.answer,
            "techniques": self.techniques,
            "raw": self.raw,
        })
    }
}

fn parse_official_stdout_rows(stdout: &str) -> Vec<StdoutRow> {
    let mut out: Vec<StdoutRow> = Vec::new();
    for (idx, raw) in stdout.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let line = raw.trim();
        if let Some((head, techniques)) = split_with_techniques(line) {
            let tokens: Vec<&str> = head.split_whitespace().collect();
            if tokens.first() == Some(&FORMULA) && tokens.len() == 3 {
                out.push(StdoutRow {
                    index: idx,
                    kind: "FORMULA".into(),
                    id: Some(tokens[1].to_string()),
                    answer: Some(tokens[2].to_string()),
                    techniques: Some(techniques.to_string()),
                    raw: line.to_string(),
                });
                continue;
            }
            if tokens.first() == Some(&STATE_SPACE) {
                if tokens.len() == 2 && tokens[1] == CANNOT_COMPUTE {
                    out.push(StdoutRow {
                        index: idx,
                        kind: "STATE_SPACE".into(),
                        id: Some(STATE_SPACE.into()),
                        answer: Some(CANNOT_COMPUTE.into()),
                        techniques: Some(techniques.to_string()),
                        raw: line.to_string(),
                    });
                    continue;
                }
                if tokens.len() == 3 {
                    let metric = tokens[1];
                    let value = tokens[2];
                    if matches!(
                        metric,
                        STATES | TRANSITIONS | MAX_TOKEN_IN_PLACE | MAX_TOKEN_PER_MARKING
                    ) && value.chars().all(|c| c.is_ascii_digit())
                    {
                        out.push(StdoutRow {
                            index: idx,
                            kind: "STATE_SPACE".into(),
                            id: Some(format!("{STATE_SPACE}:{metric}")),
                            answer: Some(value.to_string()),
                            techniques: Some(techniques.to_string()),
                            raw: line.to_string(),
                        });
                        continue;
                    }
                }
            }
        }
        out.push(StdoutRow {
            index: idx,
            kind: "UNKNOWN".into(),
            id: None,
            answer: None,
            techniques: None,
            raw: line.to_string(),
        });
    }
    out
}

fn write_schema_diagnostic_summary(sidecar_text: &str, path: &Path) -> Result<Value> {
    let mut record_count: u64 = 0;
    let mut evidence_count: u64 = 0;
    let mut schema_counts: BTreeMap<String, u64> = BTreeMap::new();

    for line in sidecar_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        if !value.is_object() {
            continue;
        }
        record_count += 1;
        for evidence_line in collect_evidence_lines(&value) {
            evidence_count += 1;
            let data = parse_evidence_tokens(&evidence_line);
            if let Some(schema) = data.get("schema") {
                *schema_counts.entry(schema.clone()).or_default() += 1;
            }
        }
    }

    let mut counts_map: Map<String, Value> = Map::new();
    for (k, v) in schema_counts {
        counts_map.insert(k, Value::from(v));
    }
    let summary = json!({
        "records": record_count,
        "evidence_rows": evidence_count,
        "schema_counts": Value::Object(counts_map),
    });
    fs::write(path, serde_json::to_string_pretty(&summary)? + "\n")?;
    Ok(summary)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn official_stdout_accepts_formula_lines() {
        let stdout = format!(
            "{FORMULA} F0 TRUE {TECHNIQUES} EXPLICIT\n\
             {FORMULA} F1 FALSE {TECHNIQUES} EXPLICIT\n\
             {FORMULA} F2 {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n"
        );
        assert!(official_stdout(&stdout));
    }

    #[test]
    fn official_stdout_accepts_state_space_metrics() {
        let stdout = format!(
            "{STATE_SPACE} {STATES} 1024 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} {TRANSITIONS} 2048 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} {MAX_TOKEN_IN_PLACE} 5 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} {MAX_TOKEN_PER_MARKING} 10 {TECHNIQUES} EXPLICIT\n"
        );
        assert!(official_stdout(&stdout));
    }

    #[test]
    fn official_stdout_accepts_state_space_cannot_compute() {
        let stdout = format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n");
        assert!(official_stdout(&stdout));
    }

    #[test]
    fn official_stdout_rejects_malformed_rows() {
        let bad_rows = [
            "STATE_SPACE BOGUS TECHNIQUES EXPLICIT",
            "STATE_SPACE STATES TECHNIQUES EXPLICIT",
            "STATE_SPACE MAX_TOKEN_IN_PLACE MANY TECHNIQUES EXPLICIT",
        ];
        for bad in bad_rows {
            let stdout = format!("{bad}\n");
            assert!(!official_stdout(&stdout), "expected rejection for: {bad}");
        }
    }

    #[test]
    fn official_stdout_rejects_legacy_spaced_state_space() {
        // Construct the 2025-archive spaced literal at runtime so the
        // keyword guard never sees the spaced form in source. The
        // wrapper rejects this on the wire because it's not the
        // canonical underscored form.
        let sp = " ";
        // mcc-keyword-guard: allow-spaced-mention
        let stdout = format!(
            "STATE{sp}SPACE MAX{sp}TOKEN{sp}IN{sp}PLACE 5 TECHNIQUES EXPLICIT\n",
            sp = sp
        );
        assert!(!official_stdout(&stdout));
    }

    #[test]
    fn official_stdout_rejects_empty_input() {
        assert!(!official_stdout(""));
        assert!(!official_stdout("\n\n  \n"));
    }

    #[test]
    fn fail_closed_stdout_state_space_emits_canonical_keywords() {
        let line = fail_closed_stdout(Path::new("/nonexistent"), Examination::StateSpace);
        assert_eq!(
            line,
            format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n")
        );
    }

    #[test]
    fn fail_closed_stdout_reachability_uses_default_formula_id() {
        let line = fail_closed_stdout(
            Path::new("/nonexistent"),
            Examination::ReachabilityFireability,
        );
        assert!(line.contains("Mutex-ReachabilityFireability-00"));
        assert!(line.contains(CANNOT_COMPUTE));
        assert!(line.contains(FORMULA));
    }

    #[test]
    fn fail_closed_stdout_reads_property_xml() {
        let dir = tempdir().unwrap();
        let xml = dir.path().join("ReachabilityFireability.xml");
        fs::write(
            &xml,
            r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>P-One</id>
    <formula><true/></formula>
  </property>
  <property>
    <id>P-Two</id>
    <formula><true/></formula>
  </property>
</property-set>"#,
        )
        .unwrap();
        let line = fail_closed_stdout(&xml, Examination::ReachabilityFireability);
        assert!(line.contains("P-One"));
        assert!(line.contains("P-Two"));
    }

    #[test]
    fn examination_round_trips_through_enum() {
        // All MCC examinations must be reachable through the enum — no
        // hardcoded EXAMS list.
        for exam in Examination::ALL {
            assert_eq!(Examination::from_name(exam.as_str()).unwrap(), exam);
        }
    }

    #[test]
    fn shlex_split_parses_evidence_rows() {
        let row = r#"MCC blocker_action priority_rank=10 owner_project="trust_cg" action_code='do_thing'"#;
        let tokens = shlex_split(row);
        assert_eq!(tokens[0], "MCC");
        assert_eq!(tokens[1], "blocker_action");
        assert_eq!(tokens[2], "priority_rank=10");
        // shlex_split is a faithful POSIX-style tokenizer: it strips quoting
        // but never rewrites token *values*, so the quoted "trust_cg" keeps its
        // underscore (just like the unrelated 'do_thing' value below). Project
        // name canonicalization, if ever needed, belongs in a semantic layer,
        // not the tokenizer.
        assert_eq!(tokens[3], "owner_project=trust_cg");
        assert_eq!(tokens[4], "action_code=do_thing");
    }

    #[test]
    fn parse_evidence_tokens_extracts_scope_component_and_kv() {
        let row = "AY trust_mc_petri_successor_chc_model_acceptance current_ay_rev=abc";
        let data = parse_evidence_tokens(row);
        assert_eq!(data.get("scope").map(String::as_str), Some("AY"));
        assert_eq!(
            data.get("component").map(String::as_str),
            Some("trust_mc_petri_successor_chc_model_acceptance"),
        );
        assert_eq!(data.get("current_ay_rev").map(String::as_str), Some("abc"));
    }

    #[test]
    fn sidecar_freshness_diagnostic_flags_missing_sidecar() {
        let dir = tempdir().unwrap();
        let sidecar = dir.path().join("nope.jsonl");
        let diag = sidecar_freshness_diagnostic(&sidecar, Some("deadbeef"), None);
        assert!(diag.is_some());
        assert!(diag.unwrap().contains("missing or empty"));
    }

    #[test]
    fn sidecar_freshness_diagnostic_flags_empty_sidecar() {
        let dir = tempdir().unwrap();
        let sidecar = dir.path().join("empty.jsonl");
        fs::write(&sidecar, "").unwrap();
        let diag = sidecar_freshness_diagnostic(&sidecar, Some("deadbeef"), None);
        assert!(diag.is_some());
        assert!(diag.unwrap().contains("missing or empty"));
    }

    #[test]
    fn sidecar_freshness_diagnostic_flags_missing_acceptance_schema() {
        let dir = tempdir().unwrap();
        let sidecar = dir.path().join("sc.jsonl");
        let record = json!({
            "report": {
                "evidence": ["Unrelated row"],
            }
        });
        fs::write(&sidecar, serde_json::to_string(&record).unwrap() + "\n").unwrap();
        let diag = sidecar_freshness_diagnostic(&sidecar, Some("deadbeef"), None);
        assert!(diag.is_some());
        assert!(diag.unwrap().contains("schema evidence"));
    }

    #[test]
    fn sidecar_freshness_diagnostic_flags_stale_ay_rev() {
        let dir = tempdir().unwrap();
        let sidecar = dir.path().join("sc.jsonl");
        let row = format!(
            "AY trust_mc_petri_successor_chc_model_acceptance schema={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA} current_ay_rev=oldrev"
        );
        let record = json!({ "report": { "evidence": [row] } });
        fs::write(&sidecar, serde_json::to_string(&record).unwrap() + "\n").unwrap();
        let diag = sidecar_freshness_diagnostic(&sidecar, Some("expectedrev"), None);
        assert!(diag.is_some());
        let msg = diag.unwrap();
        assert!(msg.contains("current_ay_rev is stale"), "{msg}");
    }

    #[test]
    fn sidecar_freshness_diagnostic_accepts_current_revision() {
        let dir = tempdir().unwrap();
        let sidecar = dir.path().join("sc.jsonl");
        let row = format!(
            "AY trust_mc_petri_successor_chc_model_acceptance schema={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA} current_ay_rev=goodrev"
        );
        let record = json!({ "report": { "evidence": [row] } });
        fs::write(&sidecar, serde_json::to_string(&record).unwrap() + "\n").unwrap();
        let diag = sidecar_freshness_diagnostic(&sidecar, Some("goodrev"), None);
        assert!(diag.is_none(), "unexpected diagnostic: {diag:?}");
    }

    #[test]
    fn parse_official_stdout_rows_handles_mixed_lines() {
        let stdout = format!(
            "{FORMULA} F1 TRUE {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} {STATES} 42 {TECHNIQUES} EXPLICIT\n\
             garbage\n"
        );
        let rows = parse_official_stdout_rows(&stdout);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].kind, "FORMULA");
        assert_eq!(rows[0].id.as_deref(), Some("F1"));
        assert_eq!(rows[0].answer.as_deref(), Some("TRUE"));
        assert_eq!(rows[1].kind, "STATE_SPACE");
        assert_eq!(rows[1].id.as_deref(), Some("STATE_SPACE:STATES"));
        assert_eq!(rows[1].answer.as_deref(), Some("42"));
        assert_eq!(rows[2].kind, "UNKNOWN");
    }

    #[test]
    fn parse_property_ids_extracts_first_id_per_property() {
        let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>Alpha</id>
    <formula><true/></formula>
  </property>
  <property>
    <id>Beta</id>
  </property>
</property-set>"#;
        let ids = parse_property_ids(xml);
        assert_eq!(ids, vec!["Alpha".to_string(), "Beta".to_string()]);
    }

    #[test]
    fn is_git_rev_validates_lowercase_hex_40() {
        assert!(is_git_rev("0123456789abcdef0123456789abcdef01234567"));
        assert!(!is_git_rev("0123"));
        assert!(!is_git_rev("0123456789ABCDEF0123456789ABCDEF01234567")); // upper-case rejected by Python `re` too.
    }

    #[test]
    fn env_enabled_handles_truthy_values() {
        for val in ["1", "true", "TRUE", "yes", "on"] {
            // SAFETY: scoped env mutation for testing.
            crate::env_guard::set_var("TY_MCC_SMOKE_TEST_FLAG", val);
            assert!(env_enabled("TY_MCC_SMOKE_TEST_FLAG"), "value={val}");
        }
        crate::env_guard::set_var("TY_MCC_SMOKE_TEST_FLAG", "");
        assert!(!env_enabled("TY_MCC_SMOKE_TEST_FLAG"));
        crate::env_guard::remove_var("TY_MCC_SMOKE_TEST_FLAG");
        assert!(!env_enabled("TY_MCC_SMOKE_TEST_FLAG"));
    }

    #[test]
    fn required_evidence_includes_petri_trust_mc_model_acceptance() {
        assert!(required_evidence().contains(&"petri_trust_mc_model_acceptance"));
    }

    #[test]
    fn required_evidence_markers_include_portfolio_route_schema() {
        let markers = required_evidence_markers();
        assert!(markers.contains(&PORTFOLIO_ROUTE_SCHEMA));
        assert!(markers.contains(&AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA));
    }
}
