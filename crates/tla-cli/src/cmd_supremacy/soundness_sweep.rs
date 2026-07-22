// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness regression sweep for `ty supremacy soundness-sweep`.
//!
//! This is the native port of the throwaway `por_change_soundness_sweep.py` and
//! `triage_mismatches.py` dev scripts, restructured to use the CORRECT
//! soundness criterion under auto partial-order reduction (auto-POR).
//!
//! The recorded `ty.states` baselines are FULL state-space counts captured with
//! POR OFF (they equal the TLC ground-truth counts). Auto-POR legitimately and
//! soundly REDUCES the number of distinct states explored for safety checking,
//! so comparing a production-default (auto-POR-on) count against a POR-off
//! baseline would false-flag a sound reduction as a regression. Under POR,
//! exact-count-matching is the WRONG soundness criterion; verdict-matching is
//! the right one.
//!
//! Each spec is therefore run TWICE and classified along two independent axes:
//!
//! 1. EXACT-COUNT CHECK (primary). The candidate is run with every sound
//!    state-count-reducing default FORCED OFF (the `--no-reduction` CLI flag —
//!    auto-POR + auto-symmetry off; every `TY_*` knob stripped, `--workers 1
//!    --backend trust-cg`). Its distinct-state count must equal the raw
//!    `ty.states` baseline. (Auto-symmetry, like auto-POR, soundly reduces
//!    distinct-state counts — orbit-reduced counts match
//!    TLC-with-declared-SYMMETRY, not the raw baseline, so the count axis
//!    must pin it off; the verdict axis below exercises it.)
//!    Equal -> `COUNT_OK`. A violation-halting spec (whose
//!    run halts on the first violated invariant/property, making the count an
//!    order-dependent count-to-first-violation) -> `VIOLATION_DELTA` (benign).
//!    Otherwise -> `COUNT_REGRESSION` (a genuine reachability regression).
//!
//!    A run that fails BEFORE completing a model-check -- i.e. it exits non-zero
//!    without finding a violation (a frontend/load/eval failure: missing file,
//!    false `ASSUME`, INSTANCE arity mismatch, parse error) -> `ERROR`. This is
//!    distinguished from a genuine safety violation, which also exits non-zero
//!    but prints a violation marker. `ERROR` is NOT a regression: the checker
//!    never produced a count to compare, so it cannot have miscounted. (Such
//!    failures typically mean the spec corpus has drifted away from the baseline
//!    snapshot, not that the checker regressed.)
//!
//! 2. VERDICT CHECK (secondary). The candidate is run in PRODUCTION DEFAULT
//!    (every `TY_*` knob stripped, auto-POR and auto-symmetry free to fire).
//!    ONLY the safety verdict (holds vs violated/deadlock/counterexample) is
//!    compared against the baseline's expected verdict. Counts are NOT compared
//!    here, because POR and symmetry orbit reduction may
//!    legitimately explore fewer states. Verdict matches -> `VERDICT_OK`;
//!    verdict differs -> `VERDICT_REGRESSION` (a real POR-soundness regression).
//!    A production-default run that errors before completing (as above) -> the
//!    verdict `ERROR` class, never `VERDICT_REGRESSION`.
//!
//! For transparency the production-default distinct count is also recorded, along
//! with whether it differs from the POR-off count (showing POR's per-spec
//! effect). Such a difference is INFORMATIONAL, never a regression.
//!
//! Overall sweep verdict: PASS iff there are zero `COUNT_REGRESSION` and zero
//! `VERDICT_REGRESSION`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::Deserialize;

use crate::cli_schema::SupremacySoundnessSweepArgs;

/// Terminal verdict phrases in a run's stdout+stderr blob that are emitted ONLY
/// on a genuine safety violation. Absence of every marker is read as "holds".
///
/// These are matched against ty's authoritative human-readable verdict lines
/// (`crates/tla-cli/src/check_report/human.rs` and the parallel path in
/// `cmd_check/runner.rs`):
///   - InvariantViolation / PropertyViolation: `Error: <X> is violated.`
///     (headline always ends `... is violated.`)
///   - LivenessViolation:                      `Error: Temporal properties were violated.`
///   - Deadlock:                               `Error: Deadlock reached.`
///
/// They MUST be precise, anchored phrases — NOT bare substrings like
/// `"Invariant"`, `"property"`, `"Temporal"`, or `"FALSE"`. A bare `"Invariant"`
/// matches the invariant *name* echo of any spec that merely declares one (e.g.
/// `TypeInvariant` contains the substring `Invariant`), `"property"`/`"Temporal"`
/// match informational "Checking N properties"/"Temporal properties:" lines, and
/// `"FALSE"` matches the TLA+ boolean literal that appears throughout ordinary
/// state output. A prior version used those broad substrings and false-flagged
/// every spec with a declared invariant (e.g. SlushSmall, `INVARIANTS:
/// TypeInvariant`) as VIOLATED while it actually held. The trailing-period
/// anchoring here ensures we only match a completed violation verdict, never a
/// declaration, a progress line, or a benign `[trust-cg] failed to compile ...`
/// interpreter-fallback warning.
const VIOLATION_MARKERS: &[&str] = &[
    "is violated.",      // Invariant / Action property violation
    "were violated.",    // Temporal (liveness) properties violation
    "Deadlock reached.", // deadlock
];

/// Substrings in a baseline record's `error_type`/`status` that indicate the
/// recorded run halted on a violated invariant/property/deadlock. Used both to
/// classify a spec as violation-halting (benign count delta) and to derive the
/// baseline's expected safety verdict.
const BASELINE_VIOLATION_TOKENS: &[&str] = &[
    "invariant",
    "violation",
    "violated",
    "deadlock",
    "counterexample",
    "liveness",
    "assume_violation",
    "property",
    "temporal",
];

// ---------- Baseline JSON shape (minimal subset) ----------

#[derive(Debug, Deserialize)]
struct Baseline {
    #[serde(default)]
    specs: BTreeMap<String, BaselineSpec>,
}

#[derive(Debug, Deserialize)]
struct BaselineSpec {
    #[serde(default)]
    source: Option<BaselineSource>,
    #[serde(default)]
    ty: Option<BaselineRecord>,
}

#[derive(Debug, Deserialize)]
struct BaselineSource {
    #[serde(default)]
    tla_path: Option<String>,
    #[serde(default)]
    cfg_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BaselineRecord {
    #[serde(default)]
    states: Option<i64>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    error_type: Option<String>,
}

impl BaselineSpec {
    fn ty_states(&self) -> Option<i64> {
        self.ty.as_ref().and_then(|m| m.states)
    }

    /// The baseline's expected safety verdict, derived from the recorded `ty`
    /// record's `status`/`error_type`. A record advertising a violation
    /// (invariant/property/deadlock/...) is expected-VIOLATED; everything else
    /// (including plain timeouts, which are not safety violations) is
    /// expected-HOLDS.
    fn expected_verdict(&self) -> SafetyVerdict {
        match &self.ty {
            Some(rec) if record_indicates_violation(rec) => SafetyVerdict::Violated,
            _ => SafetyVerdict::Holds,
        }
    }

    /// Whether the baseline marks this spec as halting on a violation (so its
    /// POR-off count is an order-dependent count-to-first-violation).
    fn baseline_is_violation(&self) -> bool {
        self.ty
            .as_ref()
            .map(record_indicates_violation)
            .unwrap_or(false)
    }
}

/// True when a baseline `ty`/`tlc` record advertises a safety violation through
/// its `error_type` or `status` (as opposed to a plain pass or a timeout).
fn record_indicates_violation(rec: &BaselineRecord) -> bool {
    // A bare "timeout"/"timeout after Ns" is a failure but NOT a safety
    // violation, so it must not flip the verdict. Any of the violation tokens in
    // either `error_type` or `status` marks a genuine violation halt.
    let mentions_violation = |s: &str| {
        let lower = s.to_ascii_lowercase();
        BASELINE_VIOLATION_TOKENS
            .iter()
            .any(|tok| lower.contains(tok))
    };
    rec.error_type.as_deref().is_some_and(mentions_violation)
        || rec.status.as_deref().is_some_and(mentions_violation)
}

// ---------- Verdicts and classifications ----------

/// The safety verdict captured from a run (or expected from a baseline).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SafetyVerdict {
    Holds,
    Violated,
}

impl SafetyVerdict {
    fn as_str(self) -> &'static str {
        match self {
            SafetyVerdict::Holds => "HOLDS",
            SafetyVerdict::Violated => "VIOLATED",
        }
    }
}

/// Outcome of the primary EXACT-COUNT check (POR forced off vs POR-off baseline).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CountClass {
    /// No source, or the tla/cfg file is missing on disk.
    Skip,
    /// The POR-off run exceeded the per-spec timeout.
    Timeout,
    /// The POR-off run failed before completing a model-check: it exited
    /// non-zero without finding a violation (a frontend/load/eval failure such
    /// as a missing file, a false ASSUME, an INSTANCE arity mismatch, or a
    /// parse error), or it produced no parseable state count. NOT a soundness
    /// regression -- the checker never produced a count to compare.
    Error,
    /// The baseline has no recorded `ty.states` to compare against.
    NoBaseline,
    /// POR-off count equals the baseline full count.
    CountOk,
    /// Count differs, but the spec halts on a violation: benign order-dependent
    /// count-to-first-violation.
    ViolationDelta,
    /// Count differs on a full-exploration spec: a genuine reachability
    /// regression -- the thing we must catch.
    CountRegression,
}

impl CountClass {
    fn as_str(self) -> &'static str {
        match self {
            CountClass::Skip => "SKIP",
            CountClass::Timeout => "TIMEOUT",
            CountClass::Error => "ERROR",
            CountClass::NoBaseline => "NO_BASELINE",
            CountClass::CountOk => "COUNT_OK",
            CountClass::ViolationDelta => "VIOLATION_DELTA",
            CountClass::CountRegression => "COUNT_REGRESSION",
        }
    }
}

/// Outcome of the secondary VERDICT check (production default vs baseline
/// verdict). Counts are deliberately NOT part of this axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerdictClass {
    /// Not evaluated (the spec was skipped before the verdict run).
    Skip,
    /// The production-default run exceeded the per-spec timeout.
    Timeout,
    /// The production-default run failed before completing a model-check (exited
    /// non-zero without finding a violation: a frontend/load/eval failure). The
    /// run produced no trustworthy verdict, so this is NOT a verdict regression.
    Error,
    /// The baseline has no usable expected verdict (no `ty` record at all).
    NoBaseline,
    /// Candidate verdict matches the baseline's expected verdict.
    VerdictOk,
    /// Candidate verdict differs: a real POR-soundness regression.
    VerdictRegression,
}

impl VerdictClass {
    fn as_str(self) -> &'static str {
        match self {
            VerdictClass::Skip => "SKIP",
            VerdictClass::Timeout => "TIMEOUT",
            VerdictClass::Error => "ERROR",
            VerdictClass::NoBaseline => "NO_BASELINE",
            VerdictClass::VerdictOk => "VERDICT_OK",
            VerdictClass::VerdictRegression => "VERDICT_REGRESSION",
        }
    }
}

// ---------- Pure helpers (unit-tested) ----------

/// Parse the FINAL distinct-state count from a captured stdout+stderr blob.
///
/// Reads the `States found: <N>` field of the terminal `Statistics:` block.
/// We must NOT parse the `Progress(<n>) at <t>s: <G> states generated, <D>
/// distinct states found, ...` checkpoint line: that is an intermediate value
/// printed mid-exploration, and for any spec slow enough to emit a progress
/// line it is far below the final total. A prior version preferred the
/// `distinct states` phrasing and thereby false-flagged every slow spec
/// (>10s) as a soundness regression while its true final count matched the
/// baseline exactly. `States found:` (capital S, trailing colon) is emitted
/// only in the final statistics, never in a progress line, so matching it is
/// unambiguous. Returns `None` if no final total is present (run errored or
/// timed out before completing) — a missing total is treated as "no count"
/// rather than silently substituting a partial checkpoint.
fn parse_states(blob: &str) -> Option<i64> {
    static FOUND: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let found = FOUND.get_or_init(|| Regex::new(r"States found:\s*([\d,]+)").unwrap());
    let cap = found.captures_iter(blob).last()?;
    cap[1].replace(',', "").parse::<i64>().ok()
}

fn blob_has_violation_marker(blob: &str) -> bool {
    VIOLATION_MARKERS.iter().any(|m| blob.contains(m))
}

/// The safety verdict implied by a run's stdout+stderr blob: VIOLATED if any
/// violation marker is present, else HOLDS.
fn verdict_from_blob(blob: &str) -> SafetyVerdict {
    if blob_has_violation_marker(blob) {
        SafetyVerdict::Violated
    } else {
        SafetyVerdict::Holds
    }
}

/// Classify the PRIMARY exact-count check for one spec.
///
/// `por_off_states` is the candidate's distinct-state count from the run with
/// auto-POR forced off; `baseline_ty_states` is the recorded POR-off full count.
fn classify_count(
    timed_out: bool,
    errored: bool,
    por_off_states: Option<i64>,
    baseline_ty_states: Option<i64>,
    violation_halting: bool,
) -> CountClass {
    if timed_out {
        return CountClass::Timeout;
    }
    if errored {
        return CountClass::Error;
    }
    let Some(states) = por_off_states else {
        return CountClass::Error;
    };
    let Some(baseline) = baseline_ty_states else {
        return CountClass::NoBaseline;
    };
    if states == baseline {
        CountClass::CountOk
    } else if violation_halting {
        CountClass::ViolationDelta
    } else {
        CountClass::CountRegression
    }
}

/// Classify the SECONDARY verdict check for one spec. Counts are intentionally
/// ignored here: production default may explore fewer states under POR.
fn classify_verdict(
    timed_out: bool,
    errored: bool,
    have_baseline_record: bool,
    observed: SafetyVerdict,
    expected: SafetyVerdict,
) -> VerdictClass {
    if timed_out {
        return VerdictClass::Timeout;
    }
    if errored {
        return VerdictClass::Error;
    }
    if !have_baseline_record {
        return VerdictClass::NoBaseline;
    }
    if observed == expected {
        VerdictClass::VerdictOk
    } else {
        VerdictClass::VerdictRegression
    }
}

/// Expand a leading `~` (or `~/...`) against `home`. Mirrors the dev scripts'
/// `os.path.expanduser` behavior for the base dir. Pure: the home dir is passed
/// in so this is unit-testable without mutating the process environment.
fn expand_home_with(path: &Path, home: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s == "~" {
        return home.to_path_buf();
    }
    if let Some(rest) = s.strip_prefix("~/") {
        return home.join(rest);
    }
    path.to_path_buf()
}

fn expand_home(path: &Path) -> PathBuf {
    expand_home_with(path, &home_dir())
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Resolve a baseline-relative tla/cfg path against the base dir. Absolute
/// paths are returned unchanged.
fn resolve_path(base_dir: &Path, raw: &str) -> PathBuf {
    if raw.starts_with('/') {
        PathBuf::from(raw)
    } else {
        base_dir.join(raw)
    }
}

// ---------- Subprocess execution ----------

struct RunOutcome {
    blob: String,
    states: Option<i64>,
    timed_out: bool,
    /// Whether the child exited with a success status (code 0). A frontend/load
    /// failure (missing file, false ASSUME, INSTANCE arity mismatch, parse error)
    /// exits non-zero while still gracefully printing a `States found: 0`
    /// statistics block, so the exit status -- not the parsed count -- is what
    /// distinguishes "checker errored before exploring" from "explored and got a
    /// (wrong) count". Note a genuine safety violation also exits non-zero, so
    /// this must be combined with the absence of a violation marker; see
    /// [`run_errored`].
    exit_success: bool,
    wall: Duration,
}

/// Whether a completed run failed at the frontend (load/eval) rather than
/// finishing a model-check. True iff the child exited non-zero, did not time
/// out, and produced no violation marker. A genuine violation also exits
/// non-zero but carries a [`VIOLATION_MARKERS`] substring, so it is excluded
/// here and flows on to the normal verdict/count classification. A crash
/// (panic/SIGSEGV) likewise exits non-zero with no marker and is treated as an
/// error -- the same outcome it already gets today via a missing state count.
fn run_errored(outcome: &RunOutcome) -> bool {
    !outcome.timed_out && !outcome.exit_success && !blob_has_violation_marker(&outcome.blob)
}

/// Build the `ty check` command for a spec under a clean production environment:
/// every `TY_*` variable is removed, all other inherited vars are preserved, and
/// any `extra_args` (e.g. the primary run's `--no-reduction`) is appended to the
/// argv. Semantic levers (auto-POR/auto-symmetry) are CLI flags — the child
/// `ty check` ignores ambient TY_AUTO_POR / TY_AUTO_SYMMETRY env.
fn build_command(ty_bin: &Path, tla: &Path, cfg: &Path, extra_args: &[&str]) -> Command {
    let mut cmd = Command::new(ty_bin);
    cmd.arg("check")
        .arg(tla)
        .arg("--config")
        .arg(cfg)
        .arg("--workers")
        .arg("1")
        .arg("--backend")
        .arg("trust-cg");
    cmd.args(extra_args);
    // Clean production env: strip ALL TY_* knobs so nothing is forced, while
    // preserving the rest of the inherited environment (PATH, HOME, ...).
    cmd.env_clear();
    for (key, value) in std::env::vars() {
        if !key.starts_with("TY_") {
            cmd.env(key, value);
        }
    }
    cmd
}

/// Spawn `cmd`, enforce `timeout`, and return the merged stdout+stderr blob,
/// parsed state count, and whether it timed out. Mirrors the spawn+poll pattern
/// used by `tla-petri`'s `mccctl_cmd::sweep::run_with_timeout`.
///
/// The child's stdout/stderr pipes are drained CONCURRENTLY on background
/// threads while the poll loop waits. Draining only after exit deadlocks any
/// child whose output exceeds the OS pipe buffer (64 KiB on macOS): the child
/// blocks on a full pipe, never exits, and gets killed at the timeout — a
/// false `TIMEOUT` classification. Barriers (78 trust-cg action callouts)
/// emits ~82 KiB of compile logging to stderr in the production env and was
/// misclassified exactly this way while the actual check completes in ~3s.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<RunOutcome> {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let start = Instant::now();
    let mut child = cmd.spawn().context("spawn ty check")?;
    let stdout_reader = spawn_pipe_reader(child.stdout.take());
    let stderr_reader = spawn_pipe_reader(child.stderr.take());
    let finish = |timed_out: bool, exit_success: bool, wall: Duration| {
        // Pipe writers are closed once the child has exited (or been killed),
        // so the reader threads terminate promptly; join cannot hang.
        let stdout = join_pipe_reader(stdout_reader);
        let stderr = join_pipe_reader(stderr_reader);
        let mut blob = String::with_capacity(stderr.len() + stdout.len());
        // stderr first, matching the dev scripts' `stderr + stdout` ordering.
        blob.push_str(&stderr);
        blob.push_str(&stdout);
        let states = parse_states(&blob);
        Ok(RunOutcome {
            blob,
            states,
            timed_out,
            exit_success,
            wall,
        })
    };
    loop {
        match child.try_wait()? {
            Some(status) => {
                return finish(false, status.success(), start.elapsed());
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    // Killed on timeout; the dedicated `timed_out` flag wins
                    // in classification, so `exit_success` is never consulted.
                    return finish(true, false, start.elapsed());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Start a background thread that reads a child pipe to EOF.
fn spawn_pipe_reader<R: Read + Send + 'static>(
    pipe: Option<R>,
) -> Option<std::thread::JoinHandle<String>> {
    pipe.map(|mut pipe| {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = pipe.read_to_string(&mut buf);
            buf
        })
    })
}

/// Join a pipe-reader thread, tolerating both an absent pipe and a panicked
/// reader (either yields an empty capture, never an error).
fn join_pipe_reader(reader: Option<std::thread::JoinHandle<String>>) -> String {
    reader
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default()
}

// ---------- Row + summary bookkeeping ----------

struct SweepRow {
    spec: String,
    baseline: Option<i64>,
    /// Distinct-state count from the POR-off (primary) run.
    por_off_states: Option<i64>,
    /// Distinct-state count from the production-default (verdict) run.
    prod_states: Option<i64>,
    /// Whether `prod_states` differs from `por_off_states` (POR effect;
    /// informational only).
    por_reduced: bool,
    expected_verdict: SafetyVerdict,
    observed_verdict: SafetyVerdict,
    wall_display: String,
    count_class: CountClass,
    verdict_class: VerdictClass,
    note: String,
}

fn fmt_opt(value: Option<i64>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "None".to_string(),
    }
}

fn por_effect_str(por_reduced: bool, por_off: Option<i64>, prod: Option<i64>) -> String {
    if por_reduced {
        format!("yes ({}->{})", fmt_opt(por_off), fmt_opt(prod))
    } else {
        "no".to_string()
    }
}

// ---------- Entry point ----------

pub fn run(args: SupremacySoundnessSweepArgs) -> Result<()> {
    let baseline_text = fs::read_to_string(&args.baseline)
        .with_context(|| format!("read baseline {}", args.baseline.display()))?;
    let baseline: Baseline = serde_json::from_str(&baseline_text)
        .with_context(|| format!("parse baseline {}", args.baseline.display()))?;

    let base_dir = expand_home(&args.base_dir);
    let timeout = Duration::from_secs(args.timeout_secs);
    let triage_timeout = Duration::from_secs(args.triage_timeout_secs);

    let output = args.output.clone().unwrap_or_else(default_output_path);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }

    // Select spec names: explicit filter (preserving baseline membership) or all
    // baseline specs sorted by name.
    let names: Vec<String> = if args.specs.is_empty() {
        baseline.specs.keys().cloned().collect()
    } else {
        args.specs
            .iter()
            .filter(|name| baseline.specs.contains_key(*name))
            .cloned()
            .collect()
    };
    let total = names.len();

    let mut rows: Vec<SweepRow> = Vec::with_capacity(total);
    let mut count_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for key in [
        "COUNT_OK",
        "COUNT_REGRESSION",
        "VIOLATION_DELTA",
        "TIMEOUT",
        "ERROR",
        "NO_BASELINE",
        "SKIP",
    ] {
        count_counts.insert(key, 0);
    }
    let mut verdict_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for key in [
        "VERDICT_OK",
        "VERDICT_REGRESSION",
        "TIMEOUT",
        "ERROR",
        "NO_BASELINE",
        "SKIP",
    ] {
        verdict_counts.insert(key, 0);
    }

    let stdout = std::io::stdout();
    for (i, name) in names.iter().enumerate() {
        let spec = &baseline.specs[name];
        let baseline_ty_states = spec.ty_states();
        let have_baseline_record = spec.ty.is_some();
        let expected_verdict = spec.expected_verdict();
        let violation_halting = spec.baseline_is_violation();

        // SKIP: no source or the tla/cfg file does not exist on disk.
        let Some(source) = &spec.source else {
            record_skip(
                &mut rows,
                &mut count_counts,
                &mut verdict_counts,
                name,
                "no source",
            );
            print_progress(
                &stdout,
                i + 1,
                total,
                name,
                CountClass::Skip,
                VerdictClass::Skip,
            );
            continue;
        };
        let (Some(tla_raw), Some(cfg_raw)) = (&source.tla_path, &source.cfg_path) else {
            record_skip(
                &mut rows,
                &mut count_counts,
                &mut verdict_counts,
                name,
                "no source paths",
            );
            print_progress(
                &stdout,
                i + 1,
                total,
                name,
                CountClass::Skip,
                VerdictClass::Skip,
            );
            continue;
        };
        let tla = resolve_path(&base_dir, tla_raw);
        let cfg = resolve_path(&base_dir, cfg_raw);
        if !(tla.is_file() && cfg.is_file()) {
            record_skip(
                &mut rows,
                &mut count_counts,
                &mut verdict_counts,
                name,
                "missing files",
            );
            print_progress(
                &stdout,
                i + 1,
                total,
                name,
                CountClass::Skip,
                VerdictClass::Skip,
            );
            continue;
        }

        // --- PRIMARY: exact-count check with every sound count-reducing
        // default forced OFF (auto-POR and auto-symmetry): raw reachability
        // parity against the recorded full state-space baseline. ---
        let por_off = run_with_timeout(
            build_command(&args.ty_bin, &tla, &cfg, &["--no-reduction"]),
            timeout,
        )?;
        let count_class = classify_count(
            por_off.timed_out,
            run_errored(&por_off),
            por_off.states,
            baseline_ty_states,
            violation_halting,
        );

        // --- SECONDARY: verdict check in production default (auto-POR free). ---
        // This re-run reuses the (typically more generous) triage timeout, since
        // it is the analog of the old mismatch re-run phase.
        let prod = run_with_timeout(build_command(&args.ty_bin, &tla, &cfg, &[]), triage_timeout)?;
        let observed_verdict = verdict_from_blob(&prod.blob);
        let verdict_class = classify_verdict(
            prod.timed_out,
            run_errored(&prod),
            have_baseline_record,
            observed_verdict,
            expected_verdict,
        );

        // POR effect is informational only: a production-default count that is
        // lower than the POR-off count is a sound reduction, never a regression.
        let por_reduced = match (por_off.states, prod.states) {
            (Some(a), Some(b)) => a != b,
            _ => false,
        };

        let note = build_note(
            count_class,
            verdict_class,
            por_off.states,
            baseline_ty_states,
            expected_verdict,
            observed_verdict,
            por_reduced,
            por_off.states,
            prod.states,
        );

        *count_counts.entry(count_class.as_str()).or_insert(0) += 1;
        *verdict_counts.entry(verdict_class.as_str()).or_insert(0) += 1;

        let wall_display = {
            let total_wall = por_off.wall + prod.wall;
            if por_off.timed_out {
                format!(">{}", args.timeout_secs)
            } else if prod.timed_out {
                format!(">{}", args.triage_timeout_secs)
            } else {
                format!("{:.2}", total_wall.as_secs_f64())
            }
        };
        print_progress(&stdout, i + 1, total, name, count_class, verdict_class);
        rows.push(SweepRow {
            spec: name.clone(),
            baseline: baseline_ty_states,
            por_off_states: por_off.states,
            prod_states: prod.states,
            por_reduced,
            expected_verdict,
            observed_verdict,
            wall_display,
            count_class,
            verdict_class,
            note,
        });
    }

    write_report(&output, &rows, &count_counts, &verdict_counts, total)
        .with_context(|| format!("write report {}", output.display()))?;

    print_summary(
        &stdout,
        &output,
        &count_counts,
        &verdict_counts,
        total,
        &rows,
    );

    // Propagate the sweep verdict to the process exit code so CI fails loudly on
    // a regression instead of silently exiting 0 with "OVERALL: FAIL" buried in
    // the report. The full report and summary have already been printed above.
    if !sweep_passed(&rows) {
        let count_regs = count_regressions(&rows);
        let verdict_regs = verdict_regressions(&rows);
        bail!(
            "soundness sweep FAILED: {} count regression(s), {} verdict regression(s) (see {})",
            count_regs.len(),
            verdict_regs.len(),
            output.display()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_note(
    count_class: CountClass,
    verdict_class: VerdictClass,
    por_off_states: Option<i64>,
    baseline_ty_states: Option<i64>,
    expected: SafetyVerdict,
    observed: SafetyVerdict,
    por_reduced: bool,
    por_off: Option<i64>,
    prod: Option<i64>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    match count_class {
        CountClass::Error => parts.push(
            "run errored before completing a check (frontend/load failure or no state count)"
                .to_string(),
        ),
        CountClass::ViolationDelta => parts.push(
            "violation-halting: count-to-first-violation is order-dependent (benign)".to_string(),
        ),
        CountClass::CountRegression => parts.push(format!(
            "reachability regression: POR-off {} != baseline {}",
            fmt_opt(por_off_states),
            fmt_opt(baseline_ty_states)
        )),
        _ => {}
    }
    // Only annotate a verdict-run error when the count axis did not already say
    // the same thing (avoids a duplicate "run errored" note for the common case
    // where both runs of a drifted spec fail identically).
    if verdict_class == VerdictClass::Error && count_class != CountClass::Error {
        parts.push("verdict run errored before completing a check".to_string());
    }
    if verdict_class == VerdictClass::VerdictRegression {
        parts.push(format!(
            "verdict regression: production default {} != expected {}",
            observed.as_str(),
            expected.as_str()
        ));
    }
    if por_reduced && count_class != CountClass::CountRegression {
        parts.push(format!(
            "POR reduced states {}->{} (informational)",
            fmt_opt(por_off),
            fmt_opt(prod)
        ));
    }
    parts.join("; ")
}

fn record_skip(
    rows: &mut Vec<SweepRow>,
    count_counts: &mut BTreeMap<&'static str, usize>,
    verdict_counts: &mut BTreeMap<&'static str, usize>,
    name: &str,
    note: &str,
) {
    *count_counts.entry("SKIP").or_insert(0) += 1;
    *verdict_counts.entry("SKIP").or_insert(0) += 1;
    rows.push(SweepRow {
        spec: name.to_string(),
        baseline: None,
        por_off_states: None,
        prod_states: None,
        por_reduced: false,
        expected_verdict: SafetyVerdict::Holds,
        observed_verdict: SafetyVerdict::Holds,
        wall_display: "-".to_string(),
        count_class: CountClass::Skip,
        verdict_class: VerdictClass::Skip,
        note: note.to_string(),
    });
}

fn print_progress(
    stdout: &std::io::Stdout,
    index: usize,
    total: usize,
    name: &str,
    count_class: CountClass,
    verdict_class: VerdictClass,
) {
    let mut handle = stdout.lock();
    let _ = writeln!(
        handle,
        "[{index}/{total}] {name}: count={} verdict={}",
        count_class.as_str(),
        verdict_class.as_str(),
    );
    let _ = handle.flush();
}

fn default_output_path() -> PathBuf {
    Path::new("reports")
        .join("perf")
        .join(format!(
            "{}-soundness-sweep",
            chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
        ))
        .join("results.tsv")
}

fn count_regressions(rows: &[SweepRow]) -> Vec<&SweepRow> {
    rows.iter()
        .filter(|r| r.count_class == CountClass::CountRegression)
        .collect()
}

fn verdict_regressions(rows: &[SweepRow]) -> Vec<&SweepRow> {
    rows.iter()
        .filter(|r| r.verdict_class == VerdictClass::VerdictRegression)
        .collect()
}

/// Overall sweep verdict: PASS iff zero COUNT_REGRESSION and zero
/// VERDICT_REGRESSION across all rows.
fn sweep_passed(rows: &[SweepRow]) -> bool {
    count_regressions(rows).is_empty() && verdict_regressions(rows).is_empty()
}

fn write_report(
    output: &Path,
    rows: &[SweepRow],
    count_counts: &BTreeMap<&'static str, usize>,
    verdict_counts: &BTreeMap<&'static str, usize>,
    total: usize,
) -> Result<()> {
    let mut body = String::new();
    body.push_str(
        "spec\tty_baseline\tpor_off_states\tcount_class\tprod_states\texp_verdict\tobs_verdict\tverdict_class\tpor_effect\twall_s\tnote\n",
    );
    for row in rows {
        let _ = writeln!(
            body,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            row.spec,
            fmt_opt(row.baseline),
            fmt_opt(row.por_off_states),
            row.count_class.as_str(),
            fmt_opt(row.prod_states),
            row.expected_verdict.as_str(),
            row.observed_verdict.as_str(),
            row.verdict_class.as_str(),
            por_effect_str(row.por_reduced, row.por_off_states, row.prod_states),
            row.wall_display,
            row.note,
        );
    }

    body.push_str("\n# COUNT-CHECK SUMMARY (primary: POR-off vs full baseline)\n");
    for (key, count) in count_counts {
        let _ = writeln!(body, "# {key}\t{count}");
    }
    body.push_str("\n# VERDICT-CHECK SUMMARY (secondary: production-default verdict)\n");
    for (key, count) in verdict_counts {
        let _ = writeln!(body, "# {key}\t{count}");
    }
    let _ = writeln!(body, "# total\t{total}");

    let count_regs = count_regressions(rows);
    let verdict_regs = verdict_regressions(rows);
    body.push_str("\n# REGRESSIONS\n");
    if count_regs.is_empty() && verdict_regs.is_empty() {
        body.push_str("# (none)\n");
    } else {
        for row in &count_regs {
            let _ = writeln!(
                body,
                "# COUNT_REGRESSION\t{}\tbaseline={}\tpor_off={}",
                row.spec,
                fmt_opt(row.baseline),
                fmt_opt(row.por_off_states),
            );
        }
        for row in &verdict_regs {
            let _ = writeln!(
                body,
                "# VERDICT_REGRESSION\t{}\texpected={}\tobserved={}",
                row.spec,
                row.expected_verdict.as_str(),
                row.observed_verdict.as_str(),
            );
        }
    }

    let _ = writeln!(
        body,
        "\n# OVERALL\t{}",
        if sweep_passed(rows) { "PASS" } else { "FAIL" }
    );

    fs::write(output, body)?;
    Ok(())
}

fn print_summary(
    stdout: &std::io::Stdout,
    output: &Path,
    count_counts: &BTreeMap<&'static str, usize>,
    verdict_counts: &BTreeMap<&'static str, usize>,
    total: usize,
    rows: &[SweepRow],
) {
    let mut handle = stdout.lock();
    let _ = writeln!(handle, "\n=== COUNT-CHECK SUMMARY (primary) ===");
    for (key, count) in count_counts {
        let _ = writeln!(handle, "{key}: {count}");
    }
    let _ = writeln!(handle, "\n=== VERDICT-CHECK SUMMARY (secondary) ===");
    for (key, count) in verdict_counts {
        let _ = writeln!(handle, "{key}: {count}");
    }
    let _ = writeln!(handle, "total: {total}");

    let por_specs: Vec<&str> = rows
        .iter()
        .filter(|r| r.por_reduced)
        .map(|r| r.spec.as_str())
        .collect();
    let _ = writeln!(
        handle,
        "\nPOR reduced state count on {} spec(s) (informational): {}",
        por_specs.len(),
        if por_specs.is_empty() {
            "-".to_string()
        } else {
            por_specs.join(", ")
        }
    );

    let count_regs = count_regressions(rows);
    let verdict_regs = verdict_regressions(rows);
    let _ = writeln!(handle, "\nreport: {}", output.display());
    if count_regs.is_empty() && verdict_regs.is_empty() {
        let _ = writeln!(handle, "OVERALL: PASS (no count or verdict regressions)");
    } else {
        let _ = writeln!(handle, "OVERALL: FAIL");
        if !count_regs.is_empty() {
            let names: Vec<&str> = count_regs.iter().map(|r| r.spec.as_str()).collect();
            let _ = writeln!(handle, "!! COUNT_REGRESSION: {}", names.join(", "));
        }
        if !verdict_regs.is_empty() {
            let names: Vec<&str> = verdict_regs.iter().map(|r| r.spec.as_str()).collect();
            let _ = writeln!(handle, "!! VERDICT_REGRESSION: {}", names.join(", "));
        }
    }
    let _ = handle.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_states_ignores_progress_checkpoint_uses_final_total() {
        // Regression guard: the `Progress(N)` checkpoint reports a mid-run
        // `distinct states found` value that is far below the final total. The
        // parser must return the terminal `States found:` total (32293), NOT
        // the 26213 progress checkpoint. This exact confusion previously
        // false-flagged 9 slow specs as soundness regressions.
        let blob = "Progress(10) at 10.4s: 50462 states generated, 26213 distinct states found, 2513 states/sec\n\
                    Model checking complete: No errors found (exhaustive).\n\
                    Statistics:\n  States found: 32293\n  Transitions: 63081\n";
        assert_eq!(parse_states(blob), Some(32293));
    }

    #[test]
    fn parse_states_strips_commas_in_final_total() {
        let blob = "Statistics:\n  States found: 1,234,567\n";
        assert_eq!(parse_states(blob), Some(1_234_567));
    }

    #[test]
    fn parse_states_takes_last_states_found_match() {
        let blob = "States found: 7\nStates found: 13\n";
        assert_eq!(parse_states(blob), Some(13));
    }

    #[test]
    fn parse_states_none_for_distinct_phrasing_without_final_total() {
        // A run that only emitted progress lines (no final Statistics block,
        // e.g. timed out) yields no trustworthy total — return None rather
        // than the partial checkpoint.
        let blob = "Progress(3) at 10.0s: 99 states generated, 42 distinct states found, ...";
        assert_eq!(parse_states(blob), None);
    }

    #[test]
    fn parse_states_none_when_no_match() {
        assert_eq!(parse_states("no counts here"), None);
    }

    // ----- Primary exact-count classification -----

    #[test]
    fn classify_count_timeout_wins() {
        assert_eq!(
            classify_count(true, false, Some(10), Some(10), false),
            CountClass::Timeout
        );
    }

    #[test]
    fn classify_count_error_when_no_states_parsed() {
        assert_eq!(
            classify_count(false, false, None, Some(10), false),
            CountClass::Error
        );
    }

    #[test]
    fn classify_count_no_baseline_when_ty_states_missing() {
        assert_eq!(
            classify_count(false, false, Some(10), None, false),
            CountClass::NoBaseline
        );
    }

    #[test]
    fn classify_count_ok_on_equal_counts() {
        assert_eq!(
            classify_count(false, false, Some(10), Some(10), false),
            CountClass::CountOk
        );
    }

    #[test]
    fn classify_count_violation_delta_for_violation_halting_spec() {
        // POR-off count differs from baseline, but the spec halts on a
        // violation -> benign order-dependent count.
        assert_eq!(
            classify_count(false, false, Some(5), Some(10), true),
            CountClass::ViolationDelta
        );
    }

    #[test]
    fn classify_count_regression_for_full_exploration_delta() {
        // Full-exploration spec whose POR-off count differs from the full
        // baseline -> a genuine reachability regression.
        assert_eq!(
            classify_count(false, false, Some(5), Some(10), false),
            CountClass::CountRegression
        );
    }

    #[test]
    fn classify_count_error_when_errored_overrides_zero_count() {
        // A frontend/load failure gracefully prints "States found: 0" and exits
        // non-zero, so parse_states yields Some(0). The `errored` flag must win
        // so this is ERROR, not a COUNT_REGRESSION against the baseline 28.
        assert_eq!(
            classify_count(false, true, Some(0), Some(28), false),
            CountClass::Error
        );
    }

    #[test]
    fn classify_count_timeout_beats_errored() {
        // A timed-out, killed child is also non-success, but the dedicated
        // timeout classification must take precedence over the error path.
        assert_eq!(
            classify_count(true, true, None, Some(28), false),
            CountClass::Timeout
        );
    }

    // ----- Secondary verdict classification -----

    #[test]
    fn classify_verdict_timeout_wins() {
        assert_eq!(
            classify_verdict(
                true,
                false,
                true,
                SafetyVerdict::Holds,
                SafetyVerdict::Holds
            ),
            VerdictClass::Timeout
        );
    }

    #[test]
    fn classify_verdict_no_baseline_when_record_missing() {
        assert_eq!(
            classify_verdict(
                false,
                false,
                false,
                SafetyVerdict::Holds,
                SafetyVerdict::Holds
            ),
            VerdictClass::NoBaseline
        );
    }

    #[test]
    fn classify_verdict_ok_when_verdicts_agree() {
        assert_eq!(
            classify_verdict(
                false,
                false,
                true,
                SafetyVerdict::Holds,
                SafetyVerdict::Holds
            ),
            VerdictClass::VerdictOk
        );
        assert_eq!(
            classify_verdict(
                false,
                false,
                true,
                SafetyVerdict::Violated,
                SafetyVerdict::Violated
            ),
            VerdictClass::VerdictOk
        );
    }

    #[test]
    fn classify_verdict_regression_when_verdicts_differ() {
        assert_eq!(
            classify_verdict(
                false,
                false,
                true,
                SafetyVerdict::Holds,
                SafetyVerdict::Violated
            ),
            VerdictClass::VerdictRegression
        );
    }

    #[test]
    fn classify_verdict_error_when_errored_overrides_verdict_mismatch() {
        // A frontend/load failure makes the spec un-checkable, so its observed
        // verdict (Holds, since 0 states were explored) must NOT be compared
        // against the baseline's expected Violated -- that would be a spurious
        // VERDICT_REGRESSION. The `errored` flag routes it to ERROR instead.
        assert_eq!(
            classify_verdict(
                false,
                true,
                true,
                SafetyVerdict::Holds,
                SafetyVerdict::Violated
            ),
            VerdictClass::Error
        );
    }

    #[test]
    fn classify_verdict_timeout_beats_errored() {
        // As on the count axis, the dedicated timeout classification outranks the
        // error path when a run both timed out and exited non-zero.
        assert_eq!(
            classify_verdict(
                true,
                true,
                true,
                SafetyVerdict::Holds,
                SafetyVerdict::Violated
            ),
            VerdictClass::Timeout
        );
    }

    // ----- The frontend-error discriminator (run_errored) -----

    fn outcome(timed_out: bool, exit_success: bool, blob: &str) -> RunOutcome {
        RunOutcome {
            blob: blob.to_string(),
            states: None,
            timed_out,
            exit_success,
            wall: Duration::from_secs(0),
        }
    }

    #[test]
    fn run_errored_false_on_success_exit() {
        // A clean exit (HOLDS) is never an error, regardless of marker content.
        assert!(!run_errored(&outcome(false, true, "States found: 28\n")));
    }

    #[test]
    fn run_errored_true_on_nonzero_exit_without_marker() {
        // The frontend-failure case: non-zero exit, not timed out, no violation
        // marker (e.g. an INSTANCE arity mismatch or a false ASSUME).
        assert!(run_errored(&outcome(
            false,
            false,
            "Arity mismatch: P!pc_translation expects 1 arguments, got 3\nStates found: 0\n"
        )));
    }

    #[test]
    fn run_errored_false_on_violation_marker() {
        // A genuine safety violation also exits non-zero, but carries a marker,
        // so it must flow on to normal verdict/count classification, not ERROR.
        assert!(!run_errored(&outcome(
            false,
            false,
            "Invariant Inv is violated.\n"
        )));
    }

    #[test]
    fn run_errored_false_when_timed_out() {
        // A killed (timed-out) child is non-success, but timeout is handled by a
        // dedicated class upstream, so run_errored must not also claim it.
        assert!(!run_errored(&outcome(true, false, "")));
    }

    // ----- The key POR case (Step 3) -----

    /// A spec whose production-default count is LESS than its baseline (because
    /// auto-POR soundly reduced the explored states) but whose POR-off count
    /// EQUALS the baseline and whose verdict is preserved must classify as PASS:
    /// COUNT_OK on the primary axis and VERDICT_OK on the secondary axis -- NOT
    /// a regression.
    #[test]
    fn por_reduction_with_preserved_verdict_is_pass_not_regression() {
        let baseline = Some(100);
        // POR-off run reproduces the full baseline exactly.
        let por_off_states = Some(100);
        // Production default explores fewer states under POR.
        let prod_states = Some(60);

        // Primary: POR-off count matches the full baseline -> COUNT_OK.
        let count_class = classify_count(false, false, por_off_states, baseline, false);
        assert_eq!(count_class, CountClass::CountOk);

        // Secondary: the safety verdict is preserved (holds vs holds) -> VERDICT_OK.
        // Counts are NOT consulted here, so the POR reduction is irrelevant.
        let verdict_class = classify_verdict(
            false,
            false,
            true,
            SafetyVerdict::Holds,
            SafetyVerdict::Holds,
        );
        assert_eq!(verdict_class, VerdictClass::VerdictOk);

        // The per-spec POR effect is recorded but informational only.
        let row = SweepRow {
            spec: "ReducibleSafe".to_string(),
            baseline,
            por_off_states,
            prod_states,
            por_reduced: por_off_states != prod_states,
            expected_verdict: SafetyVerdict::Holds,
            observed_verdict: SafetyVerdict::Holds,
            wall_display: "0.00".to_string(),
            count_class,
            verdict_class,
            note: String::new(),
        };
        assert!(row.por_reduced, "POR effect should be recorded");
        assert!(
            sweep_passed(std::slice::from_ref(&row)),
            "a sound POR reduction with preserved verdict must PASS"
        );
        assert!(count_regressions(std::slice::from_ref(&row)).is_empty());
        assert!(verdict_regressions(std::slice::from_ref(&row)).is_empty());
    }

    #[test]
    fn sweep_fails_on_count_regression() {
        let row = SweepRow {
            spec: "BrokenReach".to_string(),
            baseline: Some(100),
            por_off_states: Some(80),
            prod_states: Some(80),
            por_reduced: false,
            expected_verdict: SafetyVerdict::Holds,
            observed_verdict: SafetyVerdict::Holds,
            wall_display: "0.00".to_string(),
            count_class: CountClass::CountRegression,
            verdict_class: VerdictClass::VerdictOk,
            note: String::new(),
        };
        assert!(!sweep_passed(std::slice::from_ref(&row)));
    }

    #[test]
    fn sweep_fails_on_verdict_regression() {
        let row = SweepRow {
            spec: "PorUnsound".to_string(),
            baseline: Some(100),
            por_off_states: Some(100),
            prod_states: Some(60),
            por_reduced: true,
            expected_verdict: SafetyVerdict::Violated,
            observed_verdict: SafetyVerdict::Holds,
            wall_display: "0.00".to_string(),
            count_class: CountClass::CountOk,
            verdict_class: VerdictClass::VerdictRegression,
            note: String::new(),
        };
        assert!(!sweep_passed(std::slice::from_ref(&row)));
    }

    // ----- Violation detection (data-driven, from baseline record) -----

    #[test]
    fn record_violation_detection() {
        let viol = BaselineRecord {
            states: Some(16),
            status: Some("pass".to_string()),
            error_type: Some("invariant_violation".to_string()),
        };
        assert!(record_indicates_violation(&viol));

        let deadlock = BaselineRecord {
            states: Some(4),
            status: Some("error".to_string()),
            error_type: Some("deadlock".to_string()),
        };
        assert!(record_indicates_violation(&deadlock));

        // A plain timeout is a failure but NOT a safety violation.
        let timeout = BaselineRecord {
            states: None,
            status: Some("fail".to_string()),
            error_type: Some("timeout after 120s".to_string()),
        };
        assert!(!record_indicates_violation(&timeout));

        // A clean pass is not a violation.
        let pass = BaselineRecord {
            states: Some(1245),
            status: Some("pass".to_string()),
            error_type: None,
        };
        assert!(!record_indicates_violation(&pass));
    }

    #[test]
    fn baseline_spec_expected_verdict_tracks_record() {
        let violated = BaselineSpec {
            source: None,
            ty: Some(BaselineRecord {
                states: Some(16),
                status: Some("pass".to_string()),
                error_type: Some("invariant_violation".to_string()),
            }),
        };
        assert_eq!(violated.expected_verdict(), SafetyVerdict::Violated);
        assert!(violated.baseline_is_violation());

        let holds = BaselineSpec {
            source: None,
            ty: Some(BaselineRecord {
                states: Some(1245),
                status: Some("pass".to_string()),
                error_type: None,
            }),
        };
        assert_eq!(holds.expected_verdict(), SafetyVerdict::Holds);
        assert!(!holds.baseline_is_violation());
    }

    #[test]
    fn verdict_from_blob_detects_violation() {
        // Each genuine terminal verdict line ty emits on a real violation.
        assert_eq!(
            verdict_from_blob("Error: Invariant Inv is violated."),
            SafetyVerdict::Violated
        );
        assert_eq!(
            verdict_from_blob("Error: Action property Next is violated."),
            SafetyVerdict::Violated
        );
        assert_eq!(
            verdict_from_blob("Error: Temporal properties were violated."),
            SafetyVerdict::Violated
        );
        assert_eq!(
            verdict_from_blob("Error: Deadlock reached."),
            SafetyVerdict::Violated
        );
        assert_eq!(
            verdict_from_blob("12 distinct states found."),
            SafetyVerdict::Holds
        );
    }

    #[test]
    fn blob_marker_detection() {
        assert!(blob_has_violation_marker(
            "Error: Invariant Inv is violated."
        ));
        assert!(blob_has_violation_marker(
            "Error: Temporal properties were violated."
        ));
        assert!(blob_has_violation_marker("Error: Deadlock reached."));
        assert!(!blob_has_violation_marker("12 distinct states found."));
    }

    /// Regression guard for the SlushSmall false positive: a spec that HOLDS but
    /// declares an invariant named `TypeInvariant` and emits benign trust-cg
    /// interpreter-fallback warnings must NOT be classified as VIOLATED. The
    /// earlier broad-substring markers (`"Invariant"`, `"property"`, `"FALSE"`,
    /// `"Temporal"`) matched the declaration/echo and the literal `FALSE` in
    /// state output, falsely flipping the verdict.
    #[test]
    fn verdict_from_blob_holds_for_declared_invariant_and_fallbacks() {
        let holds_blob = "\
Config: INVARIANTS: TypeInvariant
Checking 1 invariant(s), 0 propertie(s)
[trust-cg] failed to compile action SendMsg: unsupported shape; using interpreter
[trust-cg] failed to compile invariant TypeInvariant: error in compact materialization
Some value was FALSE during evaluation
Temporal properties: none configured
Model checking complete: No errors found (exhaustive).

Statistics:
  States found: 274678
";
        assert_eq!(verdict_from_blob(holds_blob), SafetyVerdict::Holds);
        assert!(!blob_has_violation_marker(holds_blob));
    }

    // ----- Path helpers -----

    #[test]
    fn expand_home_handles_tilde_prefix() {
        let home = Path::new("/home/test");
        assert_eq!(
            expand_home_with(Path::new("~"), home),
            PathBuf::from("/home/test")
        );
        assert_eq!(
            expand_home_with(Path::new("~/tlaplus-examples/specifications"), home),
            PathBuf::from("/home/test/tlaplus-examples/specifications")
        );
        // Non-tilde paths pass through unchanged.
        assert_eq!(
            expand_home_with(Path::new("/abs/path"), home),
            PathBuf::from("/abs/path")
        );
    }

    #[test]
    fn resolve_path_respects_absolute_and_relative() {
        let base = Path::new("/base/dir");
        assert_eq!(
            resolve_path(base, "/abs/spec.tla"),
            PathBuf::from("/abs/spec.tla")
        );
        assert_eq!(
            resolve_path(base, "rel/spec.tla"),
            PathBuf::from("/base/dir/rel/spec.tla")
        );
    }
}
