// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (regression-fence tests below construct the round-1 spaced literals at
// runtime so an auto-fixer cannot rewrite them into tautologies; the
// legacy-data parser intentionally accepts the spaced 2025-archive form.)

//! Direct MCC sweep harness for `ty mcc` / `ty-mcc`.
//!
//! Replaces a former Python helper (#4341). Runs
//! model/examination cases directly against a candidate binary, parses
//! MCC stdout, compares against either the MCC raw-result-analysis.csv
//! consensus column or local `expected.json` fixtures, and writes
//! TSV/JSON/Markdown evidence.
//!
//! Routes every MCC keyword through [`tla_petri::mcc_keywords`] and
//! every examination name through
//! [`tla_petri::examination::Examination`]. Killing the Python script
//! eliminates the last cross-language drift site that produced the
//! qualification-1 keyword bug. See `docs/mcc-2026/qualification-1/`.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use serde_json::{json, Value};

use crate::examination::Examination;
use crate::mcc_keywords::{
    CANNOT_COMPUTE, FORMULA, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING, STATES, STATE_SPACE,
    TECHNIQUES, TRANSITIONS,
};

// ---------- Constants ----------

/// StateSpace metric names in canonical report order.
const STATE_SPACE_METRICS: [&str; 4] = [
    STATES,
    TRANSITIONS,
    MAX_TOKEN_IN_PLACE,
    MAX_TOKEN_PER_MARKING,
];

/// Examinations that produce exactly one FORMULA line, with the formula
/// id equal to the examination name.
const SINGLE_FORMULA_EXAMS: [Examination; 5] = [
    Examination::ReachabilityDeadlock,
    Examination::OneSafe,
    Examination::QuasiLiveness,
    Examination::StableMarking,
    Examination::Liveness,
];

fn is_single_formula(exam: Examination) -> bool {
    SINGLE_FORMULA_EXAMS.contains(&exam)
}

/// Examinations whose verdicts come from per-formula property XML files
/// (each model directory must ship the matching `<exam>.xml`).
fn is_property_exam(exam: Examination) -> bool {
    exam.needs_property_xml()
}

/// Canonical examination report order — mirrors the Python `EXAMS`
/// tuple. Routes through [`Examination`] so the Rust enum is the only
/// authority for the 13 MCC examinations.
fn exams_in_report_order() -> [Examination; 13] {
    [
        Examination::StateSpace,
        Examination::ReachabilityDeadlock,
        Examination::OneSafe,
        Examination::QuasiLiveness,
        Examination::StableMarking,
        Examination::Liveness,
        Examination::UpperBounds,
        Examination::ReachabilityCardinality,
        Examination::ReachabilityFireability,
        Examination::CTLCardinality,
        Examination::CTLFireability,
        Examination::LTLCardinality,
        Examination::LTLFireability,
    ]
}

fn exam_order(name: &str) -> usize {
    exams_in_report_order()
        .iter()
        .position(|e| e.as_str() == name)
        .unwrap_or(999)
}

fn default_exams_string() -> String {
    exams_in_report_order()
        .iter()
        .map(|e| e.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

/// Local `expected.json` StateSpace key aliases mapped to canonical metric names.
fn local_state_metric_canon(key: &str) -> Option<&'static str> {
    match key {
        "states" => Some(STATES),
        "transitions" => Some(TRANSITIONS),
        "max_token_in_place" => Some(MAX_TOKEN_IN_PLACE),
        "max_token_sum" | "max_token_per_marking" => Some(MAX_TOKEN_PER_MARKING),
        _ => None,
    }
}

/// Normalize a StateSpace metric token from MCC stdout, mapping the
/// legacy spaced 2025-archive variants to the canonical underscored
/// form. The Python parser accepts both forms; we keep parity.
fn normalize_state_space_metric(metric: &str) -> String {
    match metric.trim() {
        // mcc-keyword-guard: allow-spaced-mention
        "MAX TOKEN IN PLACE" => MAX_TOKEN_IN_PLACE.to_string(),
        // mcc-keyword-guard: allow-spaced-mention
        "MAX TOKEN PER MARKING" => MAX_TOKEN_PER_MARKING.to_string(),
        other => other.to_string(),
    }
}

// ---------- CLI ----------

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CommandMode {
    /// Auto: treat a binary named `ty` as `ty mcc`, else as `ty-mcc`.
    Auto,
    /// Run the candidate as `ty mcc <model_dir> ...`.
    Ty,
    /// Run the candidate as `ty-mcc <model_dir> ...`.
    #[clap(name = "ty-mcc")]
    TyMcc,
}

impl CommandMode {
    fn resolve(self, binary: &Path) -> &'static str {
        match self {
            Self::Ty => "ty",
            Self::TyMcc => "ty-mcc",
            Self::Auto => {
                let name = binary
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                if name == "ty" {
                    "ty"
                } else {
                    "ty-mcc"
                }
            }
        }
    }
}

/// Command-line arguments for the `ty-mcc-sweep` harness.
#[derive(Parser, Debug)]
#[command(
    name = "ty-mcc-sweep",
    about = "Direct MCC sweep harness: run ty-mcc against MCC benchmarks and compare to expected verdicts.",
    long_about = "Runs the local `ty-mcc` (or `ty mcc`) binary across MCC \
                  benchmark cases (model x examination), parses each run's \
                  stdout, and compares verdicts against either the MCC \
                  raw-result-analysis.csv consensus column or per-model \
                  `expected.json` fixtures.\n\n\
                  Routes every MCC keyword through `tla_petri::mcc_keywords` \
                  and every examination name through \
                  `tla_petri::examination::Examination`, so the Rust enum is \
                  the single authority for the 13 examination vocabulary.\n\n\
                  Outputs results.tsv, summary.json, optional skipped.tsv, \
                  and a markdown report.\n\n\
                  Exit 0 = no wrong-answer units (and, in --strict mode, no \
                  timeouts / nonzero exits / malformed output / CANNOT_COMPUTE \
                  / missing known units).\n\
                  Exit 1 = any wrong/strict failure."
)]
pub struct Cli {
    /// MCC year (selects default inputs/answer-key paths under --root).
    #[arg(long, default_value_t = 2024)]
    year: u32,

    /// MCC fetch root; used for default inputs and answer-key paths.
    #[arg(long, value_name = "DIR")]
    root: Option<PathBuf>,

    /// Directory of model dirs, directory of per-model `.tgz` archives, or one model dir.
    #[arg(long, alias = "bench-dir", value_name = "PATH")]
    inputs_path: Option<PathBuf>,

    /// `raw-result-analysis.csv` or `raw-result-analysis.csv.zip`; omit for `expected.json` fixtures.
    #[arg(long, value_name = "PATH")]
    answer_key: Option<PathBuf>,

    /// Candidate binary. Falls back to `TY_MCC_BIN`, `MCC_BINARY`, `TY_BIN`, then PATH.
    #[arg(long, value_name = "PATH")]
    binary: Option<PathBuf>,

    /// `auto` treats a binary named `ty` as `ty mcc`, otherwise `ty-mcc`.
    #[arg(long, value_enum, default_value_t = CommandMode::Auto)]
    command_mode: CommandMode,

    /// Comma-separated input/model names (whitelist).
    #[arg(long)]
    models: Option<String>,

    /// Limit the number of selected models (0 = no limit).
    #[arg(long, default_value_t = 0)]
    limit_models: usize,

    /// Comma-separated MCC examinations (default: all 13 in canonical order).
    #[arg(long)]
    exams: Option<String>,

    /// Worker threads passed to the candidate.
    #[arg(long, default_value_t = 1)]
    threads: u32,

    /// Memory fraction passed to the candidate.
    #[arg(long, default_value_t = 0.25)]
    memory_fraction: f64,

    /// Storage mode passed to the candidate.
    #[arg(long, default_value = "memory")]
    storage: String,

    /// Max-state limit passed to the candidate.
    #[arg(long, default_value_t = 1_000_000)]
    max_states: u64,

    /// MCC candidate (`--timeout`) in seconds.
    #[arg(long, default_value_t = 60)]
    mcc_timeout: u64,

    /// Outer harness timeout in seconds (kills the child).
    #[arg(long, default_value_t = 75)]
    outer_timeout: u64,

    /// Run artifacts directory.
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Markdown report path.
    #[arg(long, value_name = "PATH")]
    report: Option<PathBuf>,

    /// Also fail on timeouts, nonzero exits, malformed output, missing, or CANNOT_COMPUTE known units.
    #[arg(long)]
    strict: bool,

    /// Run cases even if neither the answer key nor `expected.json` has an expected verdict.
    #[arg(long)]
    allow_no_expected: bool,
}

/// Entry point used by the standalone `ty-mcc-sweep` binary.
pub fn run() -> ExitCode {
    execute(Cli::parse())
}

/// Entry point used by `ty-mccctl sweep`.
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
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(2)
        }
    }
}

// ---------- Domain types ----------

#[derive(Debug, Clone)]
struct ExpectedUnit {
    /// Canonical value (e.g. "T", "F", "D", "?" or a normalized integer).
    value: String,
    /// True when the value is non-empty, non-"?" and non-infinite.
    known: bool,
    /// True when the value was wrapped in `(...)` indicating a soft-consensus.
    soft: bool,
}

#[derive(Debug, Clone)]
struct ExpectedCase {
    ids: Vec<String>,
    units: Vec<ExpectedUnit>,
    source: String,
}

#[derive(Debug, Clone)]
struct InputSpec {
    name: String,
    model_dir: Option<PathBuf>,
    archive: Option<PathBuf>,
}

#[derive(Debug, Clone, Default)]
struct ParsedOutput {
    format_ok: bool,
    note: String,
    values: BTreeMap<String, String>,
}

#[derive(Debug, Default, Clone)]
struct CaseResult {
    input_name: String,
    exam: String,
    category: String,
    rc: i32,
    timeout: bool,
    format_ok: bool,
    elapsed_ms: u128,
    expected: String,
    actual: String,
    source: String,
    known_units: u64,
    soft_units: u64,
    unknown_units: u64,
    exact_units: u64,
    soft_exact_units: u64,
    cannot_compute_units: u64,
    missing_units: u64,
    wrong_units: u64,
    soft_mismatch_units: u64,
    extra_output_units: u64,
    note: String,
}

// ---------- Run orchestration ----------

fn dispatch(mut cli: Cli) -> Result<ExitCode> {
    let exams_raw = cli.exams.clone().unwrap_or_else(default_exams_string);
    let exams: Vec<Examination> = exams_raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| Examination::from_name(s).map_err(|e| anyhow!("unknown examination {s:?}: {e}")))
        .collect::<Result<Vec<_>>>()?;
    if exams.is_empty() {
        bail!("--exams is empty after parsing");
    }

    let binary = resolve_binary(cli.binary.clone())?;
    cli.binary = Some(binary.clone());
    let command_mode_label = cli.command_mode.resolve(&binary);

    let root = cli.root.clone().unwrap_or_else(default_sweep_root);
    if cli.inputs_path.is_none() {
        cli.inputs_path = Some(default_inputs_root(&root, cli.year));
    }
    if cli.answer_key.is_none() {
        cli.answer_key = default_answer_key(&root);
    }

    let answer_key_source = cli.answer_key.as_ref().map(|p| p.display().to_string());
    let answer_key = if let Some(path) = &cli.answer_key {
        Some(read_answer_key(path)?)
    } else {
        None
    };

    let inputs_path = cli
        .inputs_path
        .clone()
        .context("--inputs-path was not provided and no default could be derived")?;
    let mut inputs = discover_inputs(&inputs_path)?;

    if let Some(wanted) = parse_csv_set(cli.models.as_deref()) {
        inputs.retain(|spec| wanted.contains(&spec.name));
    }
    if cli.limit_models > 0 && inputs.len() > cli.limit_models {
        inputs.truncate(cli.limit_models);
    }
    if inputs.is_empty() {
        bail!(
            "no benchmark inputs selected under {}",
            inputs_path.display()
        );
    }

    let output_dir = cli.output_dir.clone().unwrap_or_else(default_output_dir);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("create output dir {}", output_dir.display()))?;
    let artifacts_dir = output_dir.join("artifacts");
    fs::create_dir_all(&artifacts_dir)
        .with_context(|| format!("create artifacts dir {}", artifacts_dir.display()))?;

    let report_path = cli.report.clone().unwrap_or_else(default_report_path);

    let total_planned = inputs.len() * exams.len();
    let mut case_index = 0usize;
    let mut rows: Vec<CaseResult> = Vec::new();
    let mut skipped: Vec<SkippedCase> = Vec::new();

    for spec in &inputs {
        let (model_dir, _holder) = materialize_model(spec)?;

        for exam in &exams {
            let expected = expected_for_case(
                &model_dir,
                &spec.name,
                *exam,
                answer_key.as_ref(),
                answer_key_source.as_deref(),
            )?;
            let expected_available = expected.is_some() || cli.allow_no_expected;
            if let Some(reason) = should_skip_exam(&model_dir, *exam, expected_available) {
                skipped.push(SkippedCase {
                    input: spec.name.clone(),
                    examination: exam.as_str().to_string(),
                    reason: reason.to_string(),
                });
                continue;
            }
            case_index += 1;
            let case_dir =
                artifacts_dir.join(format!("{}__{}", clean_name(&spec.name), exam.as_str()));
            let result = run_one_case(
                &cli,
                &binary,
                command_mode_label,
                &spec.name,
                &model_dir,
                *exam,
                expected.as_ref(),
                &case_dir,
            )?;
            println!(
                "[{case_index}/{total}] {input} {exam}: {category} rc={rc} wrong={wrong} elapsed_ms={elapsed}",
                total = total_planned,
                input = spec.name,
                exam = exam.as_str(),
                category = result.category,
                rc = result.rc,
                wrong = result.wrong_units,
                elapsed = result.elapsed_ms,
            );
            rows.push(result);
        }
    }

    if rows.is_empty() {
        bail!("no cases ran; all selected cases were skipped");
    }

    let results_tsv = output_dir.join("results.tsv");
    let summary_json = output_dir.join("summary.json");
    let skipped_tsv = output_dir.join("skipped.tsv");

    write_results_tsv(&results_tsv, &rows)?;
    let summary = summarize(&rows, &skipped);
    fs::write(
        &summary_json,
        serde_json::to_string_pretty(&summary)? + "\n",
    )
    .with_context(|| format!("write {}", summary_json.display()))?;
    if !skipped.is_empty() {
        write_skipped_tsv(&skipped_tsv, &skipped)?;
    }

    let report_ctx = ReportContext {
        binary: binary.clone(),
        command_mode: command_mode_label,
        inputs_path: inputs_path.clone(),
        answer_key: cli.answer_key.clone(),
        exams: exams.iter().map(|e| e.as_str().to_string()).collect(),
        mcc_timeout: cli.mcc_timeout,
        outer_timeout: cli.outer_timeout,
        threads: cli.threads,
        storage: cli.storage.clone(),
        max_states: cli.max_states,
        results_tsv: results_tsv.clone(),
        summary_json: summary_json.clone(),
        skipped_count: skipped.len(),
    };
    write_report(&report_path, &report_ctx, &rows, &summary)?;

    let totals = &summary.totals;
    let stdout_summary = json!({
        "report": report_path.display().to_string(),
        "results_tsv": results_tsv.display().to_string(),
        "summary_json": summary_json.display().to_string(),
        "cases": totals.cases,
        "wrong_units": totals.wrong_units,
        "timeouts": totals.timeouts,
        "nonzero_exits": totals.nonzero_exits,
        "malformed_output": totals.malformed_output,
        "cannot_compute_units": totals.cannot_compute_units,
        "missing_units": totals.missing_units,
        "skipped": skipped.len(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&sort_json(&stdout_summary))?
    );

    if totals.wrong_units > 0 {
        return Ok(ExitCode::from(1));
    }
    if cli.strict
        && (totals.timeouts > 0
            || totals.nonzero_exits > 0
            || totals.malformed_output > 0
            || totals.cannot_compute_units > 0
            || totals.missing_units > 0)
    {
        return Ok(ExitCode::from(1));
    }
    Ok(ExitCode::SUCCESS)
}

// ---------- Defaults ----------

fn default_sweep_root() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("mcc-benchmarks")
        .join("2024")
}

fn default_inputs_root(root: &Path, year: u32) -> PathBuf {
    let candidates = [
        root.join("inputs").join(format!("INPUTS-{year}")),
        root.join(format!("INPUTS-{year}")),
        root.join("inputs"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    candidates[0].clone()
}

fn default_answer_key(root: &Path) -> Option<PathBuf> {
    let candidates = [
        root.join("results")
            .join("extracted")
            .join("raw-result-analysis.csv"),
        root.join("archives").join("raw-result-analysis.csv.zip"),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.clone());
        }
    }
    None
}

fn default_output_dir() -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    PathBuf::from("/tmp").join(format!("ty-mcc-sweep-{ts}"))
}

fn default_report_path() -> PathBuf {
    let date = format_today();
    PathBuf::from("reports")
        .join("mcc")
        .join(format!("{date}-wave1-sweep.md"))
}

fn format_today() -> String {
    // YYYY-MM-DD via the system clock; we avoid pulling chrono just for this.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let days = (now.as_secs() / 86_400) as i64;
    // Civil-from-days (Howard Hinnant) — public-domain algorithm.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

// ---------- Binary resolution ----------

fn resolve_binary(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p);
    }
    for key in ["TY_MCC_BIN", "MCC_BINARY", "TY_BIN"] {
        if let Some(v) = std::env::var_os(key) {
            if !v.is_empty() {
                return Ok(PathBuf::from(v));
            }
        }
    }
    for name in ["ty-mcc", "ty"] {
        if let Some(found) = which(name) {
            return Ok(found);
        }
    }
    bail!("--binary is required when ty-mcc/ty is not on PATH")
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn parse_csv_set(raw: Option<&str>) -> Option<BTreeSet<String>> {
    let raw = raw?;
    let values: BTreeSet<String> = raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

// ---------- Input discovery ----------

fn discover_inputs(path: &Path) -> Result<Vec<InputSpec>> {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if path.is_file() && name.ends_with(".tar.gz") {
        let stem = name.trim_end_matches(".tar.gz").to_string();
        return Ok(vec![InputSpec {
            name: stem,
            model_dir: None,
            archive: Some(path.to_path_buf()),
        }]);
    }
    if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("tgz") {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| name.to_string());
        return Ok(vec![InputSpec {
            name: stem,
            model_dir: None,
            archive: Some(path.to_path_buf()),
        }]);
    }
    if path.is_dir() && path.join("model.pnml").is_file() {
        return Ok(vec![InputSpec {
            name: path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string(),
            model_dir: Some(path.to_path_buf()),
            archive: None,
        }]);
    }
    if !path.is_dir() {
        bail!("benchmark input path does not exist: {}", path.display());
    }

    let mut archives: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("read dir {}", path.display()))? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("tgz") {
            archives.push(p);
        }
    }
    archives.sort();
    if !archives.is_empty() {
        return Ok(archives
            .into_iter()
            .map(|archive| {
                let stem = archive
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                    .unwrap_or_default();
                InputSpec {
                    name: stem,
                    model_dir: None,
                    archive: Some(archive),
                }
            })
            .collect());
    }

    let mut model_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    walk_for_pnml(path, &mut model_dirs)?;
    let mut inputs: Vec<InputSpec> = model_dirs
        .into_iter()
        .map(|dir| {
            let name = dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string();
            InputSpec {
                name,
                model_dir: Some(dir),
                archive: None,
            }
        })
        .collect();
    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(inputs)
}

fn walk_for_pnml(root: &Path, out: &mut BTreeSet<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root).with_context(|| format!("read dir {}", root.display()))? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            walk_for_pnml(&p, out)?;
        } else if p.file_name().and_then(|n| n.to_str()) == Some("model.pnml") {
            if let Some(parent) = p.parent() {
                out.insert(parent.to_path_buf());
            }
        }
    }
    Ok(())
}

/// Materialize a model directory (either the on-disk dir directly, or
/// the extracted archive). The returned `_holder` keeps any temp dir
/// alive for the lifetime of the model.
fn materialize_model(spec: &InputSpec) -> Result<(PathBuf, Option<TempHolder>)> {
    if let Some(dir) = &spec.model_dir {
        return Ok((dir.clone(), None));
    }
    let archive = spec
        .archive
        .as_ref()
        .ok_or_else(|| anyhow!("input has neither model dir nor archive: {:?}", spec.name))?;
    let holder = TempHolder::new(&format!("ty-mcc-{}-", clean_name(&spec.name)))?;
    let model_dir = safe_extract_tgz(archive, &holder.path)?;
    Ok((model_dir, Some(holder)))
}

struct TempHolder {
    path: PathBuf,
}

impl TempHolder {
    fn new(prefix: &str) -> Result<Self> {
        let mut base = std::env::temp_dir();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        base.push(format!("{prefix}{stamp}"));
        fs::create_dir_all(&base).with_context(|| format!("create temp dir {}", base.display()))?;
        Ok(Self { path: base })
    }
}

impl Drop for TempHolder {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn safe_extract_tgz(archive: &Path, destination: &Path) -> Result<PathBuf> {
    fs::create_dir_all(destination)?;
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(destination)
        .status()
        .with_context(|| format!("invoke tar to extract {}", archive.display()))?;
    if !status.success() {
        bail!(
            "tar exited with status {status:?} extracting {}",
            archive.display()
        );
    }
    let mut model_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    walk_for_pnml(destination, &mut model_dirs)?;
    if model_dirs.is_empty() {
        bail!("no model.pnml found after extracting {}", archive.display());
    }
    if model_dirs.len() > 1 {
        bail!(
            "multiple model.pnml files found after extracting {}",
            archive.display()
        );
    }
    Ok(model_dirs.into_iter().next().unwrap())
}

fn clean_name(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_underscore = false;
    for ch in value.chars() {
        let ok = ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == '-';
        if ok {
            out.push(ch);
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    out
}

// ---------- Answer key (CSV / CSV.zip) ----------

type AnswerKey = BTreeMap<(String, String), Vec<ExpectedUnit>>;

fn read_answer_key(path: &Path) -> Result<AnswerKey> {
    let rows = open_csv_rows(path)?;
    let mut result: AnswerKey = BTreeMap::new();
    for row in rows {
        let row = clean_row(&row);
        let input_name = row.get("Input").map(String::as_str).unwrap_or("").trim();
        let exam = row
            .get("Examination")
            .map(String::as_str)
            .unwrap_or("")
            .trim();
        let estimate = row
            .get("estimated result")
            .map(String::as_str)
            .unwrap_or("")
            .trim();
        if input_name.is_empty() || exam.is_empty() {
            continue;
        }
        let units = expected_units_from_raw(exam, estimate);
        let key = (input_name.to_string(), exam.to_string());
        if let Some(existing) = result.get(&key) {
            if !expected_units_equal(existing, &units) {
                bail!("conflicting estimated result for {} {}", input_name, exam);
            }
        }
        result.insert(key, units);
    }
    Ok(result)
}

fn expected_units_equal(a: &[ExpectedUnit], b: &[ExpectedUnit]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.value == y.value && x.known == y.known && x.soft == y.soft)
}

fn open_csv_rows(path: &Path) -> Result<Vec<BTreeMap<String, String>>> {
    let suffix = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    if suffix.as_deref() == Some("zip") {
        let csv_bytes = extract_first_csv_from_zip(path)
            .with_context(|| format!("scan CSV member in {}", path.display()))?;
        return parse_csv_records(&csv_bytes[..]);
    }
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    parse_csv_records(file)
}

fn clean_row(row: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (k, v) in row {
        let key = k.trim().trim_start_matches('#').trim().to_string();
        out.insert(key, v.clone());
    }
    out
}

fn parse_csv_records<R: Read>(reader: R) -> Result<Vec<BTreeMap<String, String>>> {
    let mut buf = BufReader::new(reader);
    let mut header_line = String::new();
    if buf.read_line(&mut header_line)? == 0 {
        return Ok(Vec::new());
    }
    let headers: Vec<String> = parse_csv_line(header_line.trim_end_matches(['\r', '\n']));
    let mut rows = Vec::new();
    for line in buf.lines() {
        let line = line?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }
        let fields = parse_csv_line(trimmed);
        let mut row = BTreeMap::new();
        for (i, value) in fields.into_iter().enumerate() {
            if let Some(key) = headers.get(i) {
                row.insert(key.clone(), value);
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quote {
            if c == '"' {
                if matches!(chars.peek(), Some('"')) {
                    current.push('"');
                    chars.next();
                } else {
                    in_quote = false;
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quote = true;
        } else if c == ',' {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    fields.push(current);
    fields
}

// ---------- Zip extraction (single CSV member, via system `unzip`) ----------

/// Extract the first `.csv` member from a ZIP archive via the system
/// `unzip` binary. Used for the MCC `raw-result-analysis.csv.zip`
/// distribution. Avoiding a new third-party Rust dep keeps the
/// dependency graph minimal; `unzip` is universally available.
fn extract_first_csv_from_zip(path: &Path) -> Result<Vec<u8>> {
    // List members to find the first .csv.
    let list = Command::new("unzip")
        .arg("-Z1")
        .arg(path)
        .output()
        .with_context(|| format!("invoke unzip -Z1 {}", path.display()))?;
    if !list.status.success() {
        bail!(
            "unzip -Z1 failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&list.stderr)
        );
    }
    let listing = String::from_utf8_lossy(&list.stdout);
    let csv_member = listing
        .lines()
        .map(str::trim)
        .find(|line| line.to_ascii_lowercase().ends_with(".csv"))
        .ok_or_else(|| anyhow!("no CSV member in {}", path.display()))?;
    let extracted = Command::new("unzip")
        .arg("-p")
        .arg(path)
        .arg(csv_member)
        .output()
        .with_context(|| format!("invoke unzip -p {} {}", path.display(), csv_member))?;
    if !extracted.status.success() {
        bail!(
            "unzip -p failed for {} member {}: {}",
            path.display(),
            csv_member,
            String::from_utf8_lossy(&extracted.stderr)
        );
    }
    Ok(extracted.stdout)
}

// ---------- Expected verdict construction ----------

fn expected_for_case(
    model_dir: &Path,
    input_name: &str,
    exam: Examination,
    answer_key: Option<&AnswerKey>,
    answer_key_source: Option<&str>,
) -> Result<Option<ExpectedCase>> {
    if let Some(key_map) = answer_key {
        let key = (input_name.to_string(), exam.as_str().to_string());
        if let Some(units) = key_map.get(&key) {
            let ids = if exam == Examination::StateSpace {
                STATE_SPACE_METRICS
                    .iter()
                    .take(units.len())
                    .map(|s| (*s).to_string())
                    .collect::<Vec<_>>()
            } else if is_single_formula(exam) {
                vec![exam.as_str().to_string()]
            } else {
                sorted_property_ids(model_dir, exam)?
            };
            return Ok(Some(ExpectedCase {
                ids,
                units: units.clone(),
                source: answer_key_source
                    .unwrap_or("raw-result-analysis.csv")
                    .to_string(),
            }));
        }
    }
    expected_from_json(model_dir, exam)
}

fn sorted_property_ids(model_dir: &Path, exam: Examination) -> Result<Vec<String>> {
    let xml_path = model_dir.join(format!("{}.xml", exam.as_str()));
    if !xml_path.exists() {
        bail!("missing property XML: {}", xml_path.display());
    }
    let content =
        fs::read_to_string(&xml_path).with_context(|| format!("read {}", xml_path.display()))?;
    let doc = roxmltree::Document::parse(&content)
        .map_err(|e| anyhow!("invalid property XML: {}: {e}", xml_path.display()))?;
    let mut ids: Vec<String> = Vec::new();
    for node in doc.descendants() {
        if node.tag_name().name() != "property" {
            continue;
        }
        let prop_ids: Vec<String> = node
            .children()
            .filter(|c| c.is_element() && c.tag_name().name() == "id")
            .map(|c| c.text().unwrap_or("").trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if prop_ids.len() != 1 {
            bail!("property without exactly one id in {}", xml_path.display());
        }
        ids.push(prop_ids[0].clone());
    }
    if ids.is_empty() {
        bail!("no property ids in {}", xml_path.display());
    }
    let mut seen: BTreeSet<&String> = BTreeSet::new();
    for id in &ids {
        if !seen.insert(id) {
            bail!("duplicate property ids in {}", xml_path.display());
        }
    }
    ids.sort();
    Ok(ids)
}

fn expected_from_json(model_dir: &Path, exam: Examination) -> Result<Option<ExpectedCase>> {
    let path = model_dir.join("expected.json");
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let data: Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let raw = match data.get(exam.as_str()) {
        Some(v) => v,
        None => return Ok(None),
    };
    let source = path.display().to_string();

    if is_single_formula(exam) {
        return Ok(Some(ExpectedCase {
            ids: vec![exam.as_str().to_string()],
            units: vec![normalize_unit_from_json(raw)],
            source,
        }));
    }
    if exam == Examination::StateSpace {
        if let Some(map) = raw.as_object() {
            let mut by_metric: BTreeMap<&'static str, &Value> = BTreeMap::new();
            for (k, v) in map {
                if let Some(metric) = local_state_metric_canon(k.as_str()) {
                    by_metric.insert(metric, v);
                }
            }
            let mut ids: Vec<String> = Vec::new();
            let mut units: Vec<ExpectedUnit> = Vec::new();
            for metric in STATE_SPACE_METRICS {
                ids.push(metric.to_string());
                let value = by_metric.get(metric).copied();
                units.push(
                    value.map_or_else(|| normalize_unit_from_str("?"), normalize_unit_from_json),
                );
            }
            return Ok(Some(ExpectedCase { ids, units, source }));
        }
        return Ok(None);
    }

    if let Some(map) = raw.as_object() {
        let xml_path = model_dir.join(format!("{}.xml", exam.as_str()));
        let ids: Vec<String> = if xml_path.exists() {
            let property_ids = sorted_property_ids(model_dir, exam)?;
            property_ids
                .into_iter()
                .filter(|id| map.contains_key(id))
                .collect()
        } else {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            keys
        };
        let units: Vec<ExpectedUnit> = ids
            .iter()
            .map(|id| {
                map.get(id)
                    .map_or_else(|| normalize_unit_from_str("?"), normalize_unit_from_json)
            })
            .collect();
        return Ok(Some(ExpectedCase { ids, units, source }));
    }
    Ok(None)
}

// ---------- Unit normalization ----------

fn normalize_unit_from_json(value: &Value) -> ExpectedUnit {
    match value {
        Value::String(s) => normalize_unit_from_str(s),
        Value::Bool(b) => normalize_unit_from_str(if *b { "TRUE" } else { "FALSE" }),
        Value::Number(n) => normalize_unit_from_str(&n.to_string()),
        Value::Null => normalize_unit_from_str("?"),
        other => normalize_unit_from_str(&other.to_string()),
    }
}

fn normalize_unit_from_str(raw: &str) -> ExpectedUnit {
    let raw = raw.trim();
    let (token_text, soft) = strip_soft_marker(raw);
    let upper = token_text.trim().to_ascii_uppercase().replace('_', " ");
    let value = if matches!(upper.as_str(), "TRUE" | "T") {
        "T".to_string()
    } else if matches!(upper.as_str(), "FALSE" | "F") {
        "F".to_string()
    // mcc-keyword-guard: allow-spaced-mention
    } else if matches!(
        upper.as_str(),
        "CANNOT COMPUTE" | "CANNOTCOMPUTE" | "DNC" | "D"
    ) {
        "D".to_string()
    } else if upper == "?" || upper.is_empty() {
        "?".to_string()
    } else {
        normalize_number(token_text)
    };
    let lower = value.to_ascii_lowercase();
    let known =
        !value.is_empty() && value != "?" && !matches!(lower.as_str(), "inf" | "+inf" | "-inf");
    ExpectedUnit { value, known, soft }
}

fn strip_soft_marker(token: &str) -> (&str, bool) {
    let t = token.trim();
    if t.len() >= 2 && t.starts_with('(') && t.ends_with(')') {
        (t[1..t.len() - 1].trim(), true)
    } else {
        (t, false)
    }
}

fn normalize_number(token: &str) -> String {
    let t = token.trim();
    if t.is_empty() {
        return String::new();
    }
    if let Ok(n) = t.parse::<i128>() {
        return n.to_string();
    }
    if let Ok(f) = t.parse::<f64>() {
        if !f.is_finite() {
            return t.to_string();
        }
        if f == f.trunc() && f.abs() < 1e18 {
            return format!("{}", f as i64);
        }
        // Strip trailing zeros / decimal point.
        let formatted = format!("{f:.18}");
        let mut s = formatted;
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        return s;
    }
    t.to_string()
}

fn expected_units_from_raw(exam: &str, raw: &str) -> Vec<ExpectedUnit> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Vec::new();
    }
    if exam == "StateSpace" || exam == "UpperBounds" {
        return raw
            .split_whitespace()
            .map(normalize_unit_from_str)
            .collect();
    }
    parse_bool_vector(raw)
        .into_iter()
        .map(|t| normalize_unit_from_str(&t))
        .collect()
}

/// Split a TF/(F)-style consensus string into per-formula tokens. Each
/// token is either a single character or a parenthesised group.
fn parse_bool_vector(raw: &str) -> Vec<String> {
    let compact: String = raw.split_whitespace().collect();
    let bytes = compact.as_bytes();
    let mut tokens: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            if let Some(end_rel) = compact[i + 1..].find(')') {
                let end = i + 1 + end_rel;
                tokens.push(compact[i..=end].to_string());
                i = end + 1;
            } else {
                tokens.push(compact[i..].to_string());
                break;
            }
        } else {
            tokens.push(compact[i..=i].to_string());
            i += 1;
        }
    }
    tokens
}

// ---------- MCC output parsing ----------

fn parse_mcc_output(exam: Examination, stdout: &str) -> ParsedOutput {
    let lines: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if lines.is_empty() {
        return ParsedOutput {
            format_ok: false,
            note: "empty stdout".to_string(),
            values: BTreeMap::new(),
        };
    }
    if exam == Examination::StateSpace {
        return parse_state_space_lines(&lines);
    }
    parse_formula_lines(exam, &lines)
}

fn parse_state_space_lines(lines: &[&str]) -> ParsedOutput {
    if lines.len() == 1 {
        let l = lines[0];
        let canonical_prefix = format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} ");
        // mcc-keyword-guard: allow-spaced-mention
        let legacy_prefix = "STATE SPACE CANNOT COMPUTE TECHNIQUES ";
        if l == CANNOT_COMPUTE || l.starts_with(&canonical_prefix) || l.starts_with(legacy_prefix) {
            let mut values = BTreeMap::new();
            values.insert(STATE_SPACE.to_string(), "D".to_string());
            return ParsedOutput {
                format_ok: true,
                note: "cannot-compute".to_string(),
                values,
            };
        }
    }
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(tech_index) = parts.iter().position(|p| *p == TECHNIQUES) else {
            return ParsedOutput {
                format_ok: false,
                note: format!("invalid StateSpace line: {line}"),
                values: BTreeMap::new(),
            };
        };
        if parts.first().copied() == Some(STATE_SPACE) {
            if tech_index < 3 {
                return ParsedOutput {
                    format_ok: false,
                    note: format!("invalid StateSpace metric: {line}"),
                    values: BTreeMap::new(),
                };
            }
            let metric = normalize_state_space_metric(parts[1]);
            let value_token = parts[tech_index - 1];
            values.insert(metric, normalize_unit_from_str(value_token).value);
        } else if parts.len() >= 2
            && parts[0] == "STATE"
            // mcc-keyword-guard: allow-spaced-mention
            && parts[1] == "SPACE"
        {
            if tech_index < 4 {
                return ParsedOutput {
                    format_ok: false,
                    note: format!("invalid StateSpace metric: {line}"),
                    values: BTreeMap::new(),
                };
            }
            let joined = parts[2..tech_index - 1].join(" ");
            let metric = normalize_state_space_metric(&joined);
            let value_token = parts[tech_index - 1];
            values.insert(metric, normalize_unit_from_str(value_token).value);
        } else {
            return ParsedOutput {
                format_ok: false,
                note: format!("invalid StateSpace line: {line}"),
                values: BTreeMap::new(),
            };
        }
    }
    ParsedOutput {
        format_ok: true,
        note: "ok".to_string(),
        values,
    }
}

fn parse_formula_lines(exam: Examination, lines: &[&str]) -> ParsedOutput {
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    for line in lines {
        let Some((id, raw_value)) = parse_formula_line(line) else {
            return ParsedOutput {
                format_ok: false,
                note: format!("invalid FORMULA line: {line}"),
                values: BTreeMap::new(),
            };
        };
        if values.contains_key(&id) {
            return ParsedOutput {
                format_ok: false,
                note: format!("duplicate FORMULA id: {id}"),
                values: BTreeMap::new(),
            };
        }
        values.insert(id, normalize_unit_from_str(&raw_value).value);
    }
    if is_single_formula(exam) {
        let expected_id = exam.as_str();
        let only_expected = values.len() == 1 && values.contains_key(expected_id);
        if !only_expected {
            let mut ids: Vec<&str> = values.keys().map(String::as_str).collect();
            ids.sort();
            return ParsedOutput {
                format_ok: false,
                note: format!(
                    "unexpected FORMULA ids for {}: {}",
                    exam.as_str(),
                    ids.join(",")
                ),
                values,
            };
        }
    }
    ParsedOutput {
        format_ok: true,
        note: "ok".to_string(),
        values,
    }
}

fn parse_formula_line(line: &str) -> Option<(String, String)> {
    // FORMULA <id> <verdict-tokens...> TECHNIQUES <techniques...>
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.first().copied() != Some(FORMULA) {
        return None;
    }
    // Per-formula inability form (MCC manual p7): no TECHNIQUES suffix is
    // emitted because no technique was applied. Shape: FORMULA <id>
    // CANNOT_COMPUTE. The verdict canonicalizes to "D" downstream via
    // normalize_unit_from_str. Mirrors the bare-line StateSpace path in
    // parse_state_space_lines.
    if tokens.len() == 3 && tokens[2] == CANNOT_COMPUTE {
        return Some((tokens[1].to_string(), CANNOT_COMPUTE.to_string()));
    }
    if tokens.len() < 5 {
        return None;
    }
    let tech_index = tokens.iter().position(|t| *t == TECHNIQUES)?;
    if tech_index < 3 {
        return None;
    }
    let id = tokens[1].to_string();
    let verdict = tokens[2..tech_index].join(" ");
    Some((id, verdict))
}

// ---------- Comparison ----------

fn canonical_expected(expected: Option<&ExpectedCase>) -> String {
    expected.map_or_else(String::new, |e| {
        e.units
            .iter()
            .map(|u| u.value.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    })
}

fn canonical_actual(expected: Option<&ExpectedCase>, parsed: &ParsedOutput) -> String {
    if let Some(e) = expected {
        e.ids
            .iter()
            .filter_map(|id| parsed.values.get(id))
            .filter(|v| !v.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        // Sorted by key for determinism.
        parsed
            .values
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn compare_output(
    input_name: &str,
    exam: Examination,
    expected: Option<&ExpectedCase>,
    parsed: &ParsedOutput,
    rc: i32,
    timed_out: bool,
    elapsed_ms: u128,
) -> CaseResult {
    let mut result = CaseResult {
        input_name: input_name.to_string(),
        exam: exam.as_str().to_string(),
        category: "no_expected".to_string(),
        rc,
        timeout: timed_out,
        format_ok: parsed.format_ok,
        elapsed_ms,
        expected: canonical_expected(expected),
        actual: canonical_actual(expected, parsed),
        source: expected.map(|e| e.source.clone()).unwrap_or_default(),
        note: parsed.note.clone(),
        ..CaseResult::default()
    };

    if timed_out {
        result.category = "timeout".to_string();
    } else if rc != 0 {
        result.category = "nonzero_exit".to_string();
    } else if !parsed.format_ok {
        result.category = "malformed_output".to_string();
    } else if expected.is_none() {
        result.category = "no_expected".to_string();
    }

    let Some(expected) = expected else {
        return result;
    };

    if expected.ids.len() != expected.units.len() {
        result.format_ok = false;
        result.category = "harness_expected_shape_error".to_string();
        result.note = format!(
            "expected ids/units mismatch: ids={} units={}",
            expected.ids.len(),
            expected.units.len()
        );
        return result;
    }

    for (unit_id, expected_unit) in expected.ids.iter().zip(expected.units.iter()) {
        if !expected_unit.known {
            result.unknown_units += 1;
            continue;
        }
        if expected_unit.soft {
            result.soft_units += 1;
        } else {
            result.known_units += 1;
        }

        let mut actual = parsed.values.get(unit_id).cloned();
        if actual.is_none()
            && exam == Examination::StateSpace
            && parsed.values.get(STATE_SPACE).map(String::as_str) == Some("D")
        {
            actual = Some("D".to_string());
        }
        let actual = actual.unwrap_or_default();
        if actual.is_empty() {
            if expected_unit.soft {
                result.soft_mismatch_units += 1;
            } else {
                result.missing_units += 1;
            }
            continue;
        }
        if actual == expected_unit.value {
            if expected_unit.soft {
                result.soft_exact_units += 1;
            } else {
                result.exact_units += 1;
            }
        } else if actual == "D" {
            if expected_unit.soft {
                result.soft_mismatch_units += 1;
            } else {
                result.cannot_compute_units += 1;
            }
        } else if expected_unit.soft {
            result.soft_mismatch_units += 1;
        } else {
            result.wrong_units += 1;
        }
    }

    let expected_ids: BTreeSet<&String> = expected.ids.iter().collect();
    let mut actual_ids: BTreeSet<&String> = parsed.values.keys().collect();
    let state_space_key = STATE_SPACE.to_string();
    if exam == Examination::StateSpace
        && parsed.values.get(STATE_SPACE).map(String::as_str) == Some("D")
    {
        // The single STATE_SPACE D row stands in for every metric — don't
        // count it as extra output.
        actual_ids = expected.ids.iter().collect();
        let _ = state_space_key;
    }
    result.extra_output_units = actual_ids.difference(&expected_ids).count() as u64;

    if matches!(
        result.category.as_str(),
        "timeout" | "nonzero_exit" | "malformed_output"
    ) {
        return result;
    }
    if result.wrong_units > 0 {
        result.category = "wrong".to_string();
    } else if result.missing_units > 0 || result.cannot_compute_units > 0 {
        result.category = "incomplete".to_string();
    } else if result.extra_output_units > 0 {
        result.category = "extra_output".to_string();
    } else if result.soft_mismatch_units > 0 {
        result.category = "soft_consensus_mismatch".to_string();
    } else if result.known_units > 0 && result.exact_units == result.known_units {
        result.category = "exact".to_string();
    } else if result.soft_units > 0 && result.soft_exact_units == result.soft_units {
        result.category = "soft_exact".to_string();
    } else if result.unknown_units > 0 {
        result.category = "unknown_only".to_string();
    } else {
        result.category = "empty_expected".to_string();
    }
    result
}

// ---------- Per-case orchestration ----------

fn build_command(
    cli: &Cli,
    binary: &Path,
    mode_label: &str,
    model_dir: &Path,
    exam: Examination,
    storage_dir: &Path,
) -> Command {
    let mut cmd = Command::new(binary);
    if mode_label == "ty" {
        cmd.arg("mcc");
    }
    cmd.arg(model_dir)
        .arg("--examination")
        .arg(exam.as_str())
        .arg("--threads")
        .arg(cli.threads.to_string())
        .arg("--memory-fraction")
        .arg(cli.memory_fraction.to_string())
        .arg("--storage")
        .arg(&cli.storage)
        .arg("--storage-dir")
        .arg(storage_dir)
        .arg("--max-states")
        .arg(cli.max_states.to_string())
        .arg("--timeout")
        .arg(cli.mcc_timeout.to_string());
    cmd
}

fn run_one_case(
    cli: &Cli,
    binary: &Path,
    mode_label: &str,
    input_name: &str,
    model_dir: &Path,
    exam: Examination,
    expected: Option<&ExpectedCase>,
    case_dir: &Path,
) -> Result<CaseResult> {
    fs::create_dir_all(case_dir)
        .with_context(|| format!("create case dir {}", case_dir.display()))?;
    let storage_dir = case_dir.join("storage");
    fs::create_dir_all(&storage_dir)
        .with_context(|| format!("create storage dir {}", storage_dir.display()))?;

    let cmd = build_command(cli, binary, mode_label, model_dir, exam, &storage_dir);
    let cmd_json = command_to_json(&cmd, binary, mode_label, model_dir, exam, &storage_dir, cli);
    fs::write(
        case_dir.join("command.json"),
        serde_json::to_string_pretty(&cmd_json)?,
    )?;

    let timeout = Duration::from_secs(cli.outer_timeout);
    let started = Instant::now();
    let (rc, timed_out, stdout, stderr) = run_with_timeout(cmd, timeout)?;
    let elapsed_ms = started.elapsed().as_millis();

    fs::write(case_dir.join("stdout.txt"), &stdout)?;
    fs::write(case_dir.join("stderr.txt"), &stderr)?;
    let parsed = parse_mcc_output(exam, &stdout);
    Ok(compare_output(
        input_name, exam, expected, &parsed, rc, timed_out, elapsed_ms,
    ))
}

fn command_to_json(
    _cmd: &Command,
    binary: &Path,
    mode_label: &str,
    model_dir: &Path,
    exam: Examination,
    storage_dir: &Path,
    cli: &Cli,
) -> Value {
    let mut parts: Vec<String> = Vec::new();
    parts.push(binary.display().to_string());
    if mode_label == "ty" {
        parts.push("mcc".to_string());
    }
    parts.push(model_dir.display().to_string());
    parts.extend([
        "--examination".to_string(),
        exam.as_str().to_string(),
        "--threads".to_string(),
        cli.threads.to_string(),
        "--memory-fraction".to_string(),
        cli.memory_fraction.to_string(),
        "--storage".to_string(),
        cli.storage.clone(),
        "--storage-dir".to_string(),
        storage_dir.display().to_string(),
        "--max-states".to_string(),
        cli.max_states.to_string(),
        "--timeout".to_string(),
        cli.mcc_timeout.to_string(),
    ]);
    Value::Array(parts.into_iter().map(Value::String).collect())
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<(i32, bool, String, String)> {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn().context("spawn candidate")?;
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = s.read_to_string(&mut stdout);
                }
                if let Some(mut s) = child.stderr.take() {
                    let _ = s.read_to_string(&mut stderr);
                }
                return Ok((status.code().unwrap_or(-1), false, stdout, stderr));
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let mut stdout = String::new();
                    let mut stderr = String::new();
                    if let Some(mut s) = child.stdout.take() {
                        let _ = s.read_to_string(&mut stdout);
                    }
                    if let Some(mut s) = child.stderr.take() {
                        let _ = s.read_to_string(&mut stderr);
                    }
                    return Ok((124, true, stdout, stderr));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn should_skip_exam(
    model_dir: &Path,
    exam: Examination,
    expected_available: bool,
) -> Option<String> {
    if is_property_exam(exam) {
        let xml = model_dir.join(format!("{}.xml", exam.as_str()));
        if !xml.is_file() {
            return Some(format!("missing {}.xml", exam.as_str()));
        }
    }
    if !expected_available {
        return Some("no expected verdict".to_string());
    }
    None
}

// ---------- Reporting ----------

#[derive(Debug, Serialize)]
struct SummaryOutput {
    totals: SummaryTotals,
    per_examination: BTreeMap<String, PerExamination>,
    skipped: Vec<SkippedCase>,
}

#[derive(Debug, Default, Serialize)]
struct SummaryTotals {
    cases: u64,
    wrong_units: u64,
    known_units: u64,
    exact_units: u64,
    cannot_compute_units: u64,
    missing_units: u64,
    soft_units: u64,
    soft_mismatch_units: u64,
    unknown_units: u64,
    extra_output_units: u64,
    timeouts: u64,
    nonzero_exits: u64,
    malformed_output: u64,
    categories: BTreeMap<String, u64>,
}

#[derive(Debug, Default, Serialize)]
struct PerExamination {
    cases: u64,
    wrong_units: u64,
    known_units: u64,
    exact_units: u64,
    cannot_compute_units: u64,
    missing_units: u64,
    timeouts: u64,
    nonzero_exits: u64,
}

#[derive(Debug, Clone, Serialize)]
struct SkippedCase {
    input: String,
    examination: String,
    reason: String,
}

fn summarize(rows: &[CaseResult], skipped: &[SkippedCase]) -> SummaryOutput {
    let mut totals = SummaryTotals {
        cases: rows.len() as u64,
        ..SummaryTotals::default()
    };
    for row in rows {
        totals.wrong_units += row.wrong_units;
        totals.known_units += row.known_units;
        totals.exact_units += row.exact_units;
        totals.cannot_compute_units += row.cannot_compute_units;
        totals.missing_units += row.missing_units;
        totals.soft_units += row.soft_units;
        totals.soft_mismatch_units += row.soft_mismatch_units;
        totals.unknown_units += row.unknown_units;
        totals.extra_output_units += row.extra_output_units;
        if row.timeout {
            totals.timeouts += 1;
        }
        if row.rc != 0 {
            totals.nonzero_exits += 1;
        }
        if !row.format_ok {
            totals.malformed_output += 1;
        }
        *totals.categories.entry(row.category.clone()).or_default() += 1;
    }

    let mut per_exam: BTreeMap<String, PerExamination> = BTreeMap::new();
    for row in rows {
        let bucket = per_exam.entry(row.exam.clone()).or_default();
        bucket.cases += 1;
        bucket.wrong_units += row.wrong_units;
        bucket.known_units += row.known_units;
        bucket.exact_units += row.exact_units;
        bucket.cannot_compute_units += row.cannot_compute_units;
        bucket.missing_units += row.missing_units;
        if row.timeout {
            bucket.timeouts += 1;
        }
        if row.rc != 0 {
            bucket.nonzero_exits += 1;
        }
    }

    SummaryOutput {
        totals,
        per_examination: per_exam,
        skipped: skipped.to_vec(),
    }
}

struct ReportContext {
    binary: PathBuf,
    command_mode: &'static str,
    inputs_path: PathBuf,
    answer_key: Option<PathBuf>,
    exams: Vec<String>,
    mcc_timeout: u64,
    outer_timeout: u64,
    threads: u32,
    storage: String,
    max_states: u64,
    results_tsv: PathBuf,
    summary_json: PathBuf,
    skipped_count: usize,
}

fn write_report(
    path: &Path,
    ctx: &ReportContext,
    rows: &[CaseResult],
    summary: &SummaryOutput,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut content = String::new();
    let generated = format_today();
    let totals = &summary.totals;

    content.push_str("# MCC Direct Sweep Report\n\n");
    content.push_str(&format!("Generated: {generated}\n\n"));
    content.push_str("Issue: #4341\n\n");
    content.push_str(&format!(
        "This is the degraded direct sweep path. It invokes `{}` directly\nand does not require the BenchKit Docker/VM path from #4340.\n\n",
        ctx.command_mode
    ));
    content.push_str("## Configuration\n\n");
    content.push_str(&format!("- Binary: `{}`\n", ctx.binary.display()));
    content.push_str(&format!("- Command mode: `{}`\n", ctx.command_mode));
    content.push_str(&format!("- Inputs: `{}`\n", ctx.inputs_path.display()));
    content.push_str(&format!(
        "- Answer key: `{}`\n",
        ctx.answer_key
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "local expected.json fixtures".to_string())
    ));
    content.push_str(&format!("- Examinations: `{}`\n", ctx.exams.join(",")));
    content.push_str(&format!("- Per-run timeout: `{}s`\n", ctx.mcc_timeout));
    content.push_str(&format!(
        "- Outer harness timeout: `{}s`\n",
        ctx.outer_timeout
    ));
    content.push_str(&format!(
        "- Threads/storage/max-states: `{}` / `{}` / `{}`\n",
        ctx.threads, ctx.storage, ctx.max_states
    ));
    content.push_str(&format!("- Results TSV: `{}`\n", ctx.results_tsv.display()));
    content.push_str(&format!(
        "- Summary JSON: `{}`\n",
        ctx.summary_json.display()
    ));
    content.push_str(&format!(
        "- Skipped selected cases: `{}`\n\n",
        ctx.skipped_count
    ));
    content.push_str(
        "The MCC raw-result-analysis source is treated as a consensus oracle, not\n\
         ground truth. Unknown `?` units are excluded from pass/fail accounting, and\n\
         parenthesized soft-consensus units are reported separately from hard wrong\n\
         answers.\n\n",
    );
    content.push_str("## Summary\n\n");
    content.push_str(&markdown_table(
        &[
            "Cases",
            "Known Units",
            "Exact",
            "Wrong",
            "DNC",
            "Missing",
            "Soft Units",
            "Soft Mismatch",
            "Unknown",
            "Timeouts",
            "Nonzero",
        ],
        &[vec![
            totals.cases.to_string(),
            totals.known_units.to_string(),
            totals.exact_units.to_string(),
            totals.wrong_units.to_string(),
            totals.cannot_compute_units.to_string(),
            totals.missing_units.to_string(),
            totals.soft_units.to_string(),
            totals.soft_mismatch_units.to_string(),
            totals.unknown_units.to_string(),
            totals.timeouts.to_string(),
            totals.nonzero_exits.to_string(),
        ]],
    ));
    content.push_str("\n\n## Per Examination\n\n");
    let mut per_exam_rows: Vec<(usize, Vec<String>)> = summary
        .per_examination
        .iter()
        .map(|(exam, data)| {
            (
                exam_order(exam),
                vec![
                    exam.clone(),
                    data.cases.to_string(),
                    data.known_units.to_string(),
                    data.exact_units.to_string(),
                    data.wrong_units.to_string(),
                    data.cannot_compute_units.to_string(),
                    data.missing_units.to_string(),
                    data.timeouts.to_string(),
                ],
            )
        })
        .collect();
    per_exam_rows.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1[0].cmp(&b.1[0])));
    let per_exam_rendered: Vec<Vec<String>> = per_exam_rows.into_iter().map(|(_, r)| r).collect();
    content.push_str(&markdown_table(
        &[
            "Examination",
            "Cases",
            "Known Units",
            "Exact",
            "Wrong",
            "DNC",
            "Missing",
            "Timeouts",
        ],
        &per_exam_rendered,
    ));
    content.push_str("\n\n## Wrong Answers\n\n");
    let wrong_rows: Vec<Vec<String>> = rows
        .iter()
        .filter(|r| r.wrong_units > 0)
        .take(50)
        .map(|r| {
            vec![
                r.input_name.clone(),
                r.exam.clone(),
                if r.expected.is_empty() {
                    "-".to_string()
                } else {
                    r.expected.clone()
                },
                if r.actual.is_empty() {
                    "-".to_string()
                } else {
                    r.actual.clone()
                },
                if r.note.is_empty() {
                    r.category.clone()
                } else {
                    r.note.clone()
                },
            ]
        })
        .collect();
    if wrong_rows.is_empty() {
        content.push_str("No hard wrong-answer units found.\n");
    } else {
        content.push_str(&markdown_table(
            &["Input", "Examination", "Expected", "Actual", "Note"],
            &wrong_rows,
        ));
        content.push('\n');
    }

    content.push_str("\n## Incomplete Known Units\n\n");
    let incomplete_rows: Vec<Vec<String>> = rows
        .iter()
        .filter(|r| r.cannot_compute_units > 0 || r.missing_units > 0 || r.timeout)
        .take(50)
        .map(|r| {
            vec![
                r.input_name.clone(),
                r.exam.clone(),
                r.category.clone(),
                r.cannot_compute_units.to_string(),
                r.missing_units.to_string(),
                r.timeout.to_string(),
            ]
        })
        .collect();
    if incomplete_rows.is_empty() {
        content.push_str("No hard known units were missing, CANNOT_COMPUTE, or timed out.\n");
    } else {
        content.push_str(&markdown_table(
            &[
                "Input",
                "Examination",
                "Category",
                "DNC",
                "Missing",
                "Timeout",
            ],
            &incomplete_rows,
        ));
        content.push('\n');
    }

    fs::write(path, content).with_context(|| format!("write report {}", path.display()))?;
    Ok(())
}

fn markdown_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    out.push_str("| ");
    out.push_str(&headers.join(" | "));
    out.push_str(" |\n| ");
    out.push_str(&vec!["---"; headers.len()].join(" | "));
    out.push_str(" |\n");
    for row in rows {
        out.push_str("| ");
        out.push_str(&row.join(" | "));
        out.push_str(" |\n");
    }
    out
}

fn write_results_tsv(path: &Path, rows: &[CaseResult]) -> Result<()> {
    let fields = [
        "input",
        "examination",
        "category",
        "rc",
        "timeout",
        "format_ok",
        "elapsed_ms",
        "known_units",
        "soft_units",
        "unknown_units",
        "exact_units",
        "soft_exact_units",
        "cannot_compute_units",
        "missing_units",
        "wrong_units",
        "soft_mismatch_units",
        "extra_output_units",
        "expected",
        "actual",
        "source",
        "note",
    ];
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(file, "{}", fields.join("\t"))?;
    for row in rows {
        let cells: Vec<String> = vec![
            tsv_clean(&row.input_name),
            tsv_clean(&row.exam),
            tsv_clean(&row.category),
            row.rc.to_string(),
            (row.timeout as u8).to_string(),
            (row.format_ok as u8).to_string(),
            row.elapsed_ms.to_string(),
            row.known_units.to_string(),
            row.soft_units.to_string(),
            row.unknown_units.to_string(),
            row.exact_units.to_string(),
            row.soft_exact_units.to_string(),
            row.cannot_compute_units.to_string(),
            row.missing_units.to_string(),
            row.wrong_units.to_string(),
            row.soft_mismatch_units.to_string(),
            row.extra_output_units.to_string(),
            tsv_clean(&row.expected),
            tsv_clean(&row.actual),
            tsv_clean(&row.source),
            tsv_clean(&row.note),
        ];
        writeln!(file, "{}", cells.join("\t"))?;
    }
    Ok(())
}

fn write_skipped_tsv(path: &Path, rows: &[SkippedCase]) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(file, "input\texamination\treason")?;
    for row in rows {
        writeln!(
            file,
            "{}\t{}\t{}",
            tsv_clean(&row.input),
            tsv_clean(&row.examination),
            tsv_clean(&row.reason),
        )?;
    }
    Ok(())
}

fn tsv_clean(s: &str) -> String {
    s.replace(['\t', '\r', '\n'], " ")
}

/// Recursively sort JSON object keys for deterministic stdout.
fn sort_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = serde_json::Map::new();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for k in keys {
                sorted.insert(k.clone(), sort_json(&map[k]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_json).collect()),
        other => other.clone(),
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit(value: &str, known: bool, soft: bool) -> ExpectedUnit {
        ExpectedUnit {
            value: value.to_string(),
            known,
            soft,
        }
    }

    #[test]
    fn examination_from_name_is_authoritative() {
        // The CLI must accept exactly the 13 canonical names — no hardcoded list.
        for exam in Examination::ALL {
            let parsed = Examination::from_name(exam.as_str()).expect("parse");
            assert_eq!(parsed.as_str(), exam.as_str());
        }
        assert!(Examination::from_name("NotAnExam").is_err());
    }

    #[test]
    fn report_order_covers_all_examinations() {
        let report = exams_in_report_order();
        for exam in Examination::ALL {
            assert!(report.contains(&exam), "missing {}", exam.as_str());
        }
    }

    #[test]
    fn state_space_parser_accepts_canonical_keywords() {
        // Build the canonical underscored line via the keyword constants.
        let line = format!(
            "{STATE_SPACE} {STATES} 42 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} {TRANSITIONS} 17 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} {MAX_TOKEN_IN_PLACE} 1 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} {MAX_TOKEN_PER_MARKING} 1 {TECHNIQUES} EXPLICIT\n"
        );
        let parsed = parse_mcc_output(Examination::StateSpace, &line);
        assert!(parsed.format_ok, "{}", parsed.note);
        assert_eq!(parsed.values[STATES], "42");
        assert_eq!(parsed.values[TRANSITIONS], "17");
        assert_eq!(parsed.values[MAX_TOKEN_IN_PLACE], "1");
        assert_eq!(parsed.values[MAX_TOKEN_PER_MARKING], "1");
    }

    #[test]
    fn state_space_parser_accepts_legacy_spaced_keywords() {
        // Construct spaced literals at runtime so the keyword guard never
        // sees them in source. This is the 2025-archive legacy spelling
        // we must continue to parse for replay.
        let sp = " ";
        let state_space_word = format!("STATE{sp}SPACE");
        let max_in_place = format!("MAX{sp}TOKEN{sp}IN{sp}PLACE");
        let max_per_marking = format!("MAX{sp}TOKEN{sp}PER{sp}MARKING");
        let lines = format!(
            "{state_space_word} STATES 7 TECHNIQUES EXPLICIT\n\
             {state_space_word} TRANSITIONS 9 TECHNIQUES EXPLICIT\n\
             {state_space_word} {max_in_place} 1 TECHNIQUES EXPLICIT\n\
             {state_space_word} {max_per_marking} 1 TECHNIQUES EXPLICIT\n"
        );
        let parsed = parse_mcc_output(Examination::StateSpace, &lines);
        assert!(parsed.format_ok, "{}", parsed.note);
        assert_eq!(parsed.values[STATES], "7");
        assert_eq!(parsed.values[MAX_TOKEN_IN_PLACE], "1");
        assert_eq!(parsed.values[MAX_TOKEN_PER_MARKING], "1");
    }

    #[test]
    fn state_space_cannot_compute_canonical() {
        let line = format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n");
        let parsed = parse_mcc_output(Examination::StateSpace, &line);
        assert!(parsed.format_ok);
        assert_eq!(
            parsed.values.get(STATE_SPACE).map(String::as_str),
            Some("D")
        );
    }

    #[test]
    fn state_space_cannot_compute_legacy_spaced() {
        let sp = " ";
        // mcc-keyword-guard: allow-spaced-mention
        let line = format!(
            "STATE{sp}SPACE CANNOT{sp}COMPUTE TECHNIQUES EXPLICIT\n",
            sp = sp
        );
        let parsed = parse_mcc_output(Examination::StateSpace, &line);
        assert!(parsed.format_ok);
        assert_eq!(
            parsed.values.get(STATE_SPACE).map(String::as_str),
            Some("D")
        );
    }

    #[test]
    fn formula_parser_routes_through_keywords() {
        let line = format!(
            "{FORMULA} F0 TRUE {TECHNIQUES} EXPLICIT\n\
             {FORMULA} F1 FALSE {TECHNIQUES} EXPLICIT\n"
        );
        let parsed = parse_mcc_output(Examination::ReachabilityFireability, &line);
        assert!(parsed.format_ok, "{}", parsed.note);
        assert_eq!(parsed.values["F0"], "T");
        assert_eq!(parsed.values["F1"], "F");
    }

    #[test]
    fn formula_cannot_compute_four_token_form_is_well_formed() {
        // Verdict canonicalizes to "D" via normalize_unit_from_str.
        let line =
            format!("{FORMULA} ReachabilityDeadlock {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n");
        let parsed = parse_mcc_output(Examination::ReachabilityDeadlock, &line);
        assert!(parsed.format_ok, "note: {}", parsed.note);
        assert_eq!(
            parsed
                .values
                .get("ReachabilityDeadlock")
                .map(String::as_str),
            Some("D")
        );
    }

    #[test]
    fn state_space_cannot_compute_bare_line_is_well_formed() {
        // MCC 2026 protocol (Fabrice Kordon 2026-05-09): StateSpace inability
        // emits a single line `CANNOT_COMPUTE` with no prefix or technique tail.
        // Embedding it inside `STATE_SPACE CANNOT_COMPUTE TECHNIQUES …` was the
        // qual-1 reject class. The sweep parser accepts both forms so it can
        // score historical and current runs without flagging valid output as
        // malformed.
        let parsed = parse_mcc_output(Examination::StateSpace, "CANNOT_COMPUTE\n");
        assert!(parsed.format_ok, "note: {}", parsed.note);
        assert_eq!(
            parsed.values.get(STATE_SPACE).map(String::as_str),
            Some("D")
        );
    }

    #[test]
    fn unknown_examination_name_is_error_via_enum() {
        // The CLI parser feeds straight into Examination::from_name — no
        // separate Python EXAMS list to drift against.
        let result = Examination::from_name("NotAnExam");
        assert!(result.is_err(), "expected error for unknown examination");
    }

    #[test]
    fn normalize_unit_handles_soft_and_dnc() {
        let u = normalize_unit_from_str("TRUE");
        assert_eq!(u.value, "T");
        assert!(u.known && !u.soft);

        let u = normalize_unit_from_str("(FALSE)");
        assert_eq!(u.value, "F");
        assert!(u.known && u.soft);

        // mcc-keyword-guard: allow-spaced-mention
        let u = normalize_unit_from_str("CANNOT COMPUTE");
        assert_eq!(u.value, "D");
        assert!(u.known);

        let u = normalize_unit_from_str("?");
        assert_eq!(u.value, "?");
        assert!(!u.known);
    }

    #[test]
    fn comparator_matches_python_exact_match() {
        let expected = ExpectedCase {
            ids: vec!["F0".to_string(), "F1".to_string()],
            units: vec![make_unit("T", true, false), make_unit("F", true, false)],
            source: "test".to_string(),
        };
        let mut values = BTreeMap::new();
        values.insert("F0".to_string(), "T".to_string());
        values.insert("F1".to_string(), "F".to_string());
        let parsed = ParsedOutput {
            format_ok: true,
            note: "ok".to_string(),
            values,
        };
        let result = compare_output(
            "Mutex",
            Examination::ReachabilityFireability,
            Some(&expected),
            &parsed,
            0,
            false,
            10,
        );
        assert_eq!(result.category, "exact");
        assert_eq!(result.exact_units, 2);
        assert_eq!(result.wrong_units, 0);
    }

    #[test]
    fn comparator_flags_wrong_unit() {
        let expected = ExpectedCase {
            ids: vec!["F0".to_string()],
            units: vec![make_unit("T", true, false)],
            source: "test".to_string(),
        };
        let mut values = BTreeMap::new();
        values.insert("F0".to_string(), "F".to_string());
        let parsed = ParsedOutput {
            format_ok: true,
            note: "ok".to_string(),
            values,
        };
        let result = compare_output(
            "Mutex",
            Examination::ReachabilityFireability,
            Some(&expected),
            &parsed,
            0,
            false,
            10,
        );
        assert_eq!(result.category, "wrong");
        assert_eq!(result.wrong_units, 1);
    }

    #[test]
    fn comparator_flags_cannot_compute() {
        let expected = ExpectedCase {
            ids: vec!["F0".to_string()],
            units: vec![make_unit("T", true, false)],
            source: "test".to_string(),
        };
        let mut values = BTreeMap::new();
        values.insert("F0".to_string(), "D".to_string());
        let parsed = ParsedOutput {
            format_ok: true,
            note: "ok".to_string(),
            values,
        };
        let result = compare_output(
            "Mutex",
            Examination::ReachabilityFireability,
            Some(&expected),
            &parsed,
            0,
            false,
            10,
        );
        assert_eq!(result.category, "incomplete");
        assert_eq!(result.cannot_compute_units, 1);
    }

    #[test]
    fn parse_bool_vector_handles_parentheses() {
        let tokens = parse_bool_vector("T F (T) F");
        assert_eq!(tokens, vec!["T", "F", "(T)", "F"]);
    }

    #[test]
    fn expected_units_from_raw_state_space_splits_on_whitespace() {
        let units = expected_units_from_raw("StateSpace", "3 5 1 3");
        assert_eq!(units.len(), 4);
        assert_eq!(units[0].value, "3");
        assert_eq!(units[3].value, "3");
    }

    #[test]
    fn clean_name_replaces_unsafe_chars() {
        assert_eq!(clean_name("Mutex/PT 01"), "Mutex_PT_01");
        assert_eq!(clean_name("simple-name.tla"), "simple-name.tla");
    }

    #[test]
    fn local_state_metric_aliases_resolve() {
        assert_eq!(local_state_metric_canon("states"), Some(STATES));
        assert_eq!(local_state_metric_canon("transitions"), Some(TRANSITIONS));
        assert_eq!(
            local_state_metric_canon("max_token_in_place"),
            Some(MAX_TOKEN_IN_PLACE)
        );
        assert_eq!(
            local_state_metric_canon("max_token_sum"),
            Some(MAX_TOKEN_PER_MARKING)
        );
        assert_eq!(
            local_state_metric_canon("max_token_per_marking"),
            Some(MAX_TOKEN_PER_MARKING)
        );
        assert_eq!(local_state_metric_canon("unrelated"), None);
    }

    #[test]
    fn csv_line_parser_handles_quoted_commas() {
        let parsed = parse_csv_line("a,b,\"c,d\",e");
        assert_eq!(parsed, vec!["a", "b", "c,d", "e"]);
    }

    #[test]
    fn format_today_returns_iso_date_shape() {
        let s = format_today();
        assert_eq!(s.len(), 10);
        assert_eq!(s.as_bytes()[4], b'-');
        assert_eq!(s.as_bytes()[7], b'-');
    }
}
