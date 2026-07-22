// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLC-vs-TY comparison gate for `ty supremacy compare`.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::parse;
use super::policy;
use super::runner::{run_command, CommandResult, CommandSpec};
#[cfg(test)]
use super::tlc_java_single_thread_args;
use super::tlc_java_single_thread_base_argv;
use crate::cli_schema::{
    SupremacyCompareArgs, SupremacyCompareBackend, SupremacyComparePolicy,
    SupremacyCompareSpecSource, SupremacyMode, SupremacyOutputFormat,
};

const COMPARE_REPORT_SCHEMA: &str = "ty.supremacy.compare.v1";
const DEFAULT_TLC_JAR: &str = "tlaplus/tytools.jar";
const DEFAULT_COMMUNITY_MODULES_JAR: &str = "tlaplus/CommunityModules.jar";
const DEFAULT_TLA_LIBRARY: &str = "test_specs/tla_library";
const ENV_TLC_BIN: &str = "TLC_BIN";
const ENV_TYTOOLS_JAR: &str = "TYTOOLS_JAR";
const ENV_TLC_JAR: &str = "TLC_JAR";
const ENV_COMMUNITY_MODULES: &str = "COMMUNITY_MODULES";
const ENV_TLA_LIBRARY: &str = "TLA_LIBRARY";
const ENV_TLA_PLUS_LIBRARY: &str = "TLA_PLUS_LIBRARY";
const DEFAULT_CASE: &str = "default";
const ALLOWED_COMPARE_CASE_ENV_KEYS: &[&str] = &["TY_PARALLEL_READONLY_VALUE_CACHES"];
pub(super) fn run(args: SupremacyCompareArgs) -> Result<()> {
    validate_args(&args)?;
    let cases = resolve_cases(&args)?;
    let repo_root = env::current_dir().context("resolve current working directory")?;
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir("compare"));
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("create output dir {}", output_dir.display()))?;

    let specs = resolve_specs(&args, &repo_root)?;
    let tlc_runner = resolve_tlc_runner(&args, &repo_root)?;
    let ty_bin = args
        .ty_bin
        .clone()
        .map(Ok)
        .unwrap_or_else(env::current_exe)
        .context("resolve ty binary")?;
    validate_file(&ty_bin).with_context(|| format!("validate --ty-bin {}", ty_bin.display()))?;

    let mut rows = Vec::new();
    for spec in &specs {
        for workers in &args.workers {
            for case in &cases {
                if matches!(args.format, SupremacyOutputFormat::Human) {
                    eprintln!(
                        "[supremacy] compare {} case={} backend={} workers={}",
                        spec.name,
                        case.name,
                        backend_cli_name(args.backend),
                        workers
                    );
                }
                let row = run_compare_row(
                    spec,
                    *workers,
                    case,
                    &args,
                    &repo_root,
                    &output_dir,
                    &tlc_runner,
                    &ty_bin,
                )?;
                rows.push(row);
            }
        }
    }

    let report = CompareReport::new(&args, output_dir.clone(), cases, rows);
    let report_json =
        serde_json::to_string_pretty(&report).context("serialize supremacy compare report")?;
    fs::write(output_dir.join("compare.json"), report_json + "\n")
        .with_context(|| format!("write {}", output_dir.join("compare.json").display()))?;
    fs::write(output_dir.join("compare.md"), report.to_markdown())
        .with_context(|| format!("write {}", output_dir.join("compare.md").display()))?;

    print_report(&report, args.format)?;

    if !report.passed && args.mode == SupremacyMode::Enforce {
        bail!(
            "ty supremacy compare failed {} row(s); see {}",
            report.failed_rows,
            output_dir.join("compare.json").display()
        );
    }
    Ok(())
}

fn validate_args(args: &SupremacyCompareArgs) -> Result<()> {
    if args.timeout == 0 {
        bail!("--timeout must be >= 1");
    }
    if args.workers.is_empty() {
        bail!("--workers must list at least one worker count");
    }
    if args.workers.contains(&0) {
        bail!("--workers values must be >= 1");
    }
    if !args.min_speedup.is_finite() || args.min_speedup <= 0.0 {
        bail!("--min-speedup must be finite and > 0");
    }
    if !args.max_memory_ratio.is_finite() || args.max_memory_ratio <= 0.0 {
        bail!("--max-memory-ratio must be finite and > 0");
    }
    if args.mode == SupremacyMode::Enforce
        && policy_checks_speed(args.policy)
        && (args.tlc_bin.is_some() || non_empty_env_path(ENV_TLC_BIN).is_some())
    {
        bail!(
            "enforced performance compare requires the auditable Java TLC runner; unset {ENV_TLC_BIN} and omit --tlc-bin so single-thread JVM controls are recorded in command artifacts"
        );
    }
    if args.mode == SupremacyMode::Enforce && policy_checks_speed(args.policy) {
        if args.workers.iter().any(|workers| *workers != 1) {
            bail!(
                "enforced single-thread performance compare requires --workers 1; use --mode warn for diagnostic multi-worker comparisons"
            );
        }
        if !args.ty_flag.is_empty() {
            bail!(
                "enforced performance compare does not allow TY-only --ty-flag values; use shared TLA+/cfg settings or --mode warn diagnostics"
            );
        }
    }
    match args.spec_source {
        SupremacyCompareSpecSource::Baseline => {
            if args.tla.is_some() || args.config.is_some() {
                bail!("--tla/--config require --spec-source explicit");
            }
        }
        SupremacyCompareSpecSource::Explicit => {
            if args.tla.is_none() || args.config.is_none() {
                bail!("--spec-source explicit requires --tla and --config");
            }
        }
    }
    Ok(())
}

fn policy_checks_speed(policy: SupremacyComparePolicy) -> bool {
    matches!(
        policy,
        SupremacyComparePolicy::ParityAndSpeed | SupremacyComparePolicy::ParityAndSpeedAndMemory
    )
}

fn policy_checks_memory(policy: SupremacyComparePolicy) -> bool {
    policy == SupremacyComparePolicy::ParityAndSpeedAndMemory
}

#[derive(Clone, Debug)]
struct CompareSpec {
    name: String,
    tla_path: PathBuf,
    cfg_path: PathBuf,
    expected_tlc_states: Option<u64>,
    expected_backend_states: Option<u64>,
    expected_tlc_error: Option<String>,
    expected_backend_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct EnvCase {
    name: String,
    env_overrides: BTreeMap<String, String>,
}

fn resolve_cases(args: &SupremacyCompareArgs) -> Result<Vec<EnvCase>> {
    let case_names = if args.cases.is_empty() {
        vec![DEFAULT_CASE.to_string()]
    } else {
        args.cases.clone()
    };

    let mut seen = BTreeSet::new();
    for name in &case_names {
        validate_case_name(name)?;
        if !seen.insert(name.clone()) {
            bail!("duplicate --case {name:?}");
        }
    }

    let protected_keys = protected_ty_env_keys(args.backend);
    let global_env = parse_env_assignments(&args.ty_env, "--ty-env", &protected_keys)?;
    let mut case_env = BTreeMap::<String, BTreeMap<String, String>>::new();
    for value in &args.case_env {
        let (case_name, key, env_value) =
            parse_case_env_assignment(value, "--case-env", &protected_keys)?;
        if !seen.contains(&case_name) {
            bail!("--case-env references unknown case {case_name:?}");
        }
        case_env
            .entry(case_name)
            .or_default()
            .insert(key, env_value);
    }

    Ok(case_names
        .into_iter()
        .map(|name| {
            let mut env_overrides = global_env.clone();
            if let Some(overrides) = case_env.remove(&name) {
                env_overrides.extend(overrides);
            }
            EnvCase {
                name,
                env_overrides,
            }
        })
        .collect())
}

fn validate_case_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("--case names must not be empty");
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.')
    {
        bail!(
            "--case {name:?} contains unsupported characters; use ASCII letters, digits, '.', '-', or '_'"
        );
    }
    Ok(())
}

fn parse_env_assignments(
    values: &[String],
    flag: &str,
    protected_keys: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for value in values {
        let (key, env_value) = parse_env_assignment(value, flag, protected_keys)?;
        result.insert(key, env_value);
    }
    Ok(result)
}

fn parse_case_env_assignment(
    value: &str,
    flag: &str,
    protected_keys: &BTreeSet<String>,
) -> Result<(String, String, String)> {
    let Some((case_name, assignment)) = value.split_once(':') else {
        bail!("{flag} must be NAME:KEY=VALUE");
    };
    validate_case_name(case_name)?;
    let (key, env_value) = parse_env_assignment(assignment, flag, protected_keys)?;
    Ok((case_name.to_string(), key, env_value))
}

fn parse_env_assignment(
    value: &str,
    flag: &str,
    protected_keys: &BTreeSet<String>,
) -> Result<(String, String)> {
    let Some((key, env_value)) = value.split_once('=') else {
        bail!("{flag} must be KEY=VALUE");
    };
    validate_user_ty_env_key(key, flag, protected_keys)?;
    validate_user_ty_env_value(key, env_value, flag)?;
    Ok((key.to_string(), env_value.to_string()))
}

fn validate_user_ty_env_key(
    key: &str,
    flag: &str,
    protected_keys: &BTreeSet<String>,
) -> Result<()> {
    if key.is_empty() {
        bail!("{flag} env key must not be empty");
    }
    if protected_keys.contains(key) {
        bail!("{flag} cannot override protected backend env key {key}");
    }
    if !key.starts_with("TY_") {
        bail!("{flag} env key {key} is not allowed; only TY_* keys may be varied");
    }
    if !key
        .chars()
        .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
    {
        bail!("{flag} env key {key} must use ASCII uppercase letters, digits, and '_'");
    }
    if !ALLOWED_COMPARE_CASE_ENV_KEYS.contains(&key) {
        bail!(
            "{flag} env key {key} is not allowed for compare env cases; allowed keys: {}",
            ALLOWED_COMPARE_CASE_ENV_KEYS.join(", ")
        );
    }
    Ok(())
}

fn validate_user_ty_env_value(key: &str, value: &str, flag: &str) -> Result<()> {
    match key {
        "TY_PARALLEL_READONLY_VALUE_CACHES" => {
            if matches!(value, "" | "0" | "1") {
                Ok(())
            } else {
                bail!(
                    "{flag} env key {key} accepts only \"\", \"0\", or \"1\" for compare env cases"
                );
            }
        }
        _ => Ok(()),
    }
}

fn protected_ty_env_keys(backend: SupremacyCompareBackend) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    keys.extend(backend_env(backend).into_keys());
    keys.extend(
        policy::full_native_fused_protected_env()
            .into_keys()
            .collect::<Vec<_>>(),
    );
    keys.extend([
        "TY_CACHE_DIR".to_string(),
        "TLA_LIBRARY".to_string(),
        "TLA_PLUS_LIBRARY".to_string(),
    ]);
    keys
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SpecBaseline {
    #[serde(default)]
    inputs: BaselineInputs,
    specs: BTreeMap<String, SpecBaselineEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct BaselineInputs {
    #[serde(default)]
    examples_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
struct SpecBaselineEntry {
    source: Option<SpecBaselineSource>,
    #[serde(default)]
    tlc: SpecBaselineMode,
    #[serde(default)]
    ty: SpecBaselineMode,
    #[serde(default)]
    ty_expected_states: Option<u64>,
    #[serde(default)]
    verified_match: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
struct SpecBaselineSource {
    tla_path: PathBuf,
    cfg_path: PathBuf,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SpecBaselineMode {
    #[serde(default)]
    states: Option<u64>,
    #[serde(default)]
    error_type: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

fn resolve_specs(args: &SupremacyCompareArgs, repo_root: &Path) -> Result<Vec<CompareSpec>> {
    match args.spec_source {
        SupremacyCompareSpecSource::Baseline => resolve_baseline_specs(args, repo_root),
        SupremacyCompareSpecSource::Explicit => resolve_explicit_spec(args, repo_root),
    }
}

fn resolve_baseline_specs(
    args: &SupremacyCompareArgs,
    repo_root: &Path,
) -> Result<Vec<CompareSpec>> {
    let text = fs::read_to_string(&args.baseline)
        .with_context(|| format!("read baseline {}", args.baseline.display()))?;
    let baseline: SpecBaseline = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline {}", args.baseline.display()))?;
    let examples_dir = baseline
        .inputs
        .examples_dir
        .clone()
        .or_else(default_examples_dir)
        .context("baseline inputs.examples_dir is absent and HOME is not set")?;
    let explicit_names = !args.specs.is_empty();
    let names = if explicit_names {
        args.specs.clone()
    } else {
        baseline.specs.keys().cloned().collect::<Vec<_>>()
    };

    let mut specs = Vec::new();
    for name in names {
        let entry = baseline.specs.get(&name).with_context(|| {
            format!(
                "baseline spec {name:?} not found in {}",
                args.baseline.display()
            )
        })?;
        if !explicit_names && !entry.verified_match.unwrap_or(false) {
            continue;
        }
        let Some(source) = entry.source.as_ref() else {
            if explicit_names {
                bail!("baseline spec {name:?} has no source paths");
            }
            continue;
        };
        let mode = source.mode.as_deref().unwrap_or("check");
        if mode != "check" {
            if explicit_names {
                bail!(
                    "baseline spec {name:?} source mode is {mode:?}; supremacy compare supports only check-mode specs"
                );
            }
            continue;
        }
        let tla_path = resolve_source_path(repo_root, &examples_dir, &source.tla_path);
        let cfg_path = resolve_source_path(repo_root, &examples_dir, &source.cfg_path);
        validate_spec_files(&tla_path, &cfg_path)?;
        specs.push(CompareSpec {
            name,
            tla_path,
            cfg_path,
            expected_tlc_states: entry.tlc.states,
            expected_backend_states: entry
                .ty_expected_states
                .or(entry.ty.states)
                .or(entry.tlc.states),
            expected_tlc_error: expected_error_type(&entry.tlc),
            expected_backend_error: expected_error_type(&entry.ty),
        });
    }
    if specs.is_empty() {
        bail!(
            "no check-mode specs selected from {}",
            args.baseline.display()
        );
    }
    Ok(specs)
}

fn resolve_explicit_spec(
    args: &SupremacyCompareArgs,
    repo_root: &Path,
) -> Result<Vec<CompareSpec>> {
    let tla_path = absolutize(repo_root, args.tla.as_ref().expect("validated --tla"));
    let cfg_path = absolutize(repo_root, args.config.as_ref().expect("validated --config"));
    validate_spec_files(&tla_path, &cfg_path)?;
    let name = args
        .specs
        .first()
        .cloned()
        .or_else(|| {
            tla_path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "explicit".to_string());
    Ok(vec![CompareSpec {
        name,
        tla_path,
        cfg_path,
        expected_tlc_states: None,
        expected_backend_states: None,
        expected_tlc_error: None,
        expected_backend_error: None,
    }])
}

fn run_compare_row(
    spec: &CompareSpec,
    workers: usize,
    case: &EnvCase,
    args: &SupremacyCompareArgs,
    repo_root: &Path,
    output_dir: &Path,
    tlc_runner: &TlcRunner,
    ty_bin: &Path,
) -> Result<CompareRow> {
    let spec_dir = output_dir
        .join(safe_name(&spec.name))
        .join(format!("workers-{workers}"))
        .join(safe_name(&case.name));
    let tlc_result = run_tlc(
        spec,
        workers,
        args.timeout,
        repo_root,
        &spec_dir,
        tlc_runner,
    )?;
    let backend_result = run_ty_backend(spec, workers, case, args, repo_root, &spec_dir, ty_bin)?;
    Ok(CompareRow::classify(
        spec,
        workers,
        &case.name,
        args.backend,
        args.policy,
        args.min_speedup,
        args.max_memory_ratio,
        tlc_result,
        backend_result,
    ))
}

fn run_tlc(
    spec: &CompareSpec,
    workers: usize,
    timeout_seconds: u64,
    repo_root: &Path,
    spec_dir: &Path,
    tlc_runner: &TlcRunner,
) -> Result<RunObservation> {
    let artifact_dir = spec_dir.join("tlc");
    let metadir = artifact_dir.join("tlc-metadir");
    let mut env_overrides = BTreeMap::new();
    let mut argv = match tlc_runner {
        TlcRunner::Executable {
            tlc_bin,
            tla_library,
        } => {
            if let Some(tla_library) = tla_library {
                env_overrides.insert(
                    "JAVA_TOOL_OPTIONS".to_string(),
                    format!("-DTLA-Library={}", tla_library.display()),
                );
            }
            vec![
                absolutize(repo_root, tlc_bin).display().to_string(),
                "-workers".to_string(),
                workers.to_string(),
            ]
        }
        TlcRunner::Java {
            tlc_jar,
            community_modules,
            tla_library,
        } => {
            let mut argv = tlc_java_single_thread_base_argv();
            if let Some(tla_library) = tla_library {
                argv.push(format!("-DTLA-Library={}", tla_library.display()));
            }
            argv.extend([
                "-cp".to_string(),
                tlc_classpath(tlc_jar, community_modules.as_deref())?,
                "tlc2.TLC".to_string(),
                "-workers".to_string(),
                workers.to_string(),
            ]);
            argv
        }
    };
    argv.extend([
        "-config".to_string(),
        spec.cfg_path.display().to_string(),
        "-metadir".to_string(),
        metadir.display().to_string(),
        spec.tla_path.display().to_string(),
    ]);
    let result = run_command(CommandSpec {
        argv,
        cwd: spec
            .tla_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        env_overrides,
        timeout_seconds,
        artifact_dir,
    })?;
    Ok(observe_tlc_run(result))
}

fn run_ty_backend(
    spec: &CompareSpec,
    workers: usize,
    case: &EnvCase,
    args: &SupremacyCompareArgs,
    repo_root: &Path,
    spec_dir: &Path,
    ty_bin: &Path,
) -> Result<RunObservation> {
    let artifact_dir = spec_dir.join(backend_cli_name(args.backend));
    let mut argv = vec![
        ty_bin.display().to_string(),
        "check".to_string(),
        spec.tla_path.display().to_string(),
        "--config".to_string(),
        spec.cfg_path.display().to_string(),
        "--workers".to_string(),
        workers.to_string(),
        "--force".to_string(),
    ];
    if args.backend == SupremacyCompareBackend::TrustCg {
        // Count-parity lever (was the TY_AUTO_POR/TY_AUTO_SYMMETRY env pins in
        // the protected trust-cg env): the child `ty check` ignores ambient
        // env for these semantic levers, so the flag is the only control.
        argv.push("--no-reduction".to_string());
    }
    argv.extend(args.ty_flag.iter().cloned());
    argv.extend([
        "--backend".to_string(),
        backend_cli_name(args.backend).to_string(),
    ]);
    let mut env_overrides = backend_env(args.backend);
    if args.backend == SupremacyCompareBackend::TrustCg {
        env_overrides.insert(
            "TY_CACHE_DIR".to_string(),
            artifact_dir
                .join("trust_cg-artifact-cache")
                .display()
                .to_string(),
        );
    }
    if let Some(tla_library) = resolve_tla_library(args, repo_root) {
        env_overrides.insert("TLA_LIBRARY".to_string(), tla_library.display().to_string());
    }
    env_overrides.extend(case.env_overrides.clone());
    let result = run_command(CommandSpec {
        argv,
        cwd: repo_root.to_path_buf(),
        env_overrides,
        timeout_seconds: args.timeout,
        artifact_dir,
    })?;
    Ok(observe_ty_run(result, args.backend))
}

#[derive(Clone, Debug, Serialize)]
struct RunObservation {
    tool: String,
    mode: String,
    status: String,
    elapsed_seconds: f64,
    peak_rss_bytes: Option<u64>,
    states_found: Option<u64>,
    transitions: Option<u64>,
    states_generated: Option<u64>,
    returncode: i32,
    timed_out: bool,
    error_type: Option<String>,
    error: Option<String>,
    artifact_dir: String,
}

fn observe_tlc_run(result: CommandResult) -> RunObservation {
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_tlc_final_counts(&stdout, &stderr);
    let error_type = run_error_type(result.returncode, result.timed_out, &stdout, &stderr);
    let error = run_error(
        result.returncode,
        result.timed_out,
        &stderr,
        counts.states_found,
    );
    RunObservation {
        tool: "tlc".to_string(),
        mode: "tlc".to_string(),
        status: status_for_error(error_type.as_deref()).to_string(),
        elapsed_seconds: result.elapsed_seconds,
        peak_rss_bytes: result.peak_rss_bytes,
        states_found: counts.states_found,
        transitions: counts.transitions,
        states_generated: counts.states_generated,
        returncode: result.returncode,
        timed_out: result.timed_out,
        error_type,
        error,
        artifact_dir: result.artifact_dir.display().to_string(),
    }
}

fn observe_ty_run(result: CommandResult, backend: SupremacyCompareBackend) -> RunObservation {
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_ty_final_counts(&stdout, &stderr);
    let error_type = run_error_type(result.returncode, result.timed_out, &stdout, &stderr);
    let error = run_error(
        result.returncode,
        result.timed_out,
        &stderr,
        counts.states_found,
    );
    RunObservation {
        tool: "ty".to_string(),
        mode: backend_cli_name(backend).to_string(),
        status: status_for_error(error_type.as_deref()).to_string(),
        elapsed_seconds: result.elapsed_seconds,
        peak_rss_bytes: result.peak_rss_bytes,
        states_found: counts.states_found,
        transitions: counts.transitions,
        states_generated: None,
        returncode: result.returncode,
        timed_out: result.timed_out,
        error_type,
        error,
        artifact_dir: result.artifact_dir.display().to_string(),
    }
}

fn run_error(
    returncode: i32,
    timed_out: bool,
    stderr: &str,
    required_states: Option<u64>,
) -> Option<String> {
    if timed_out {
        return Some("timeout".to_string());
    }
    if returncode != 0 {
        return Some(
            first_error_line(stderr).unwrap_or_else(|| format!("returncode {returncode}")),
        );
    }
    if required_states.is_none() {
        return Some("missing states_found".to_string());
    }
    None
}

fn first_error_line(stderr: &str) -> Option<String> {
    stderr
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.chars().take(240).collect())
}

fn run_error_type(returncode: i32, timed_out: bool, stdout: &str, stderr: &str) -> Option<String> {
    if timed_out {
        return Some("timeout".to_string());
    }
    if returncode == 0 {
        return None;
    }
    Some(classify_output_error_type(stdout, stderr))
}

fn classify_output_error_type(stdout: &str, stderr: &str) -> String {
    let output = format!("{stdout}\n{stderr}");
    let lower = output.to_ascii_lowercase();
    if has_liveness_violation_marker(&lower) {
        "liveness".to_string()
    } else if has_invariant_violation_marker(&lower) {
        "invariant".to_string()
    } else if lower.contains("deadlock") {
        "deadlock".to_string()
    } else if lower.contains("parse") || lower.contains("syntax") {
        "parse".to_string()
    } else if lower.contains("unsupported") || lower.contains("not supported") {
        "unsupported".to_string()
    } else if lower.contains("action") && lower.contains("failed") {
        "action".to_string()
    } else if lower.contains("safety") {
        "safety".to_string()
    } else {
        "unknown".to_string()
    }
}

fn has_liveness_violation_marker(output_lower: &str) -> bool {
    output_lower.lines().any(|line| {
        let line = line.trim();
        line.contains("temporal properties were violated")
            || line.contains("liveness violation")
            || (line.contains("liveness property") && line.contains("violated"))
    })
}

fn has_invariant_violation_marker(output_lower: &str) -> bool {
    output_lower.lines().any(|line| {
        let line = line.trim();
        line.contains("invariant") && line.contains("violated")
    })
}

fn status_for_error(error_type: Option<&str>) -> &'static str {
    match error_type {
        None => "pass",
        Some("timeout") => "timeout",
        Some(_) => "fail",
    }
}

fn expected_error_type(mode: &SpecBaselineMode) -> Option<String> {
    if let Some(error_type) = &mode.error_type {
        return Some(normalize_error_type(error_type));
    }
    match mode.status.as_deref() {
        Some("timeout") => Some("timeout".to_string()),
        Some("error" | "fail") => Some("unknown".to_string()),
        _ => None,
    }
}

fn normalize_error_type(error_type: &str) -> String {
    match error_type {
        "invariant_violation" => "invariant".to_string(),
        "liveness_violation" => "liveness".to_string(),
        value if value.starts_with("timeout") => "timeout".to_string(),
        value => value.to_string(),
    }
}

fn error_types_compatible(tlc: Option<&str>, backend: Option<&str>) -> bool {
    tlc == backend
        || matches!(
            (tlc, backend),
            (Some("invariant"), Some("safety")) | (Some("safety"), Some("invariant"))
        )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CompareClass {
    Pass,
    TlcFailed,
    BackendFailed,
    ExpectedStateMismatch,
    ExpectedErrorMismatch,
    ErrorMismatch,
    ParityFail,
    MissingRuntime,
    SpeedFail,
    MissingMemory,
    MemoryFail,
}

#[derive(Clone, Debug, Serialize)]
struct CompareRow {
    spec: String,
    workers: usize,
    case: String,
    backend: String,
    class: CompareClass,
    passed: bool,
    reason: String,
    tlc: RunObservation,
    backend_run: RunObservation,
    parity_states: bool,
    parity_transitions: bool,
    expected_tlc_states: Option<u64>,
    expected_backend_states: Option<u64>,
    expected_tlc_error: Option<String>,
    expected_backend_error: Option<String>,
    speedup_tlc_vs_backend: Option<f64>,
    memory_ratio_backend_vs_tlc: Option<f64>,
    policy: String,
    min_speedup: f64,
    max_memory_ratio: f64,
}

impl CompareRow {
    fn classify(
        spec: &CompareSpec,
        workers: usize,
        case: &str,
        backend: SupremacyCompareBackend,
        policy: SupremacyComparePolicy,
        min_speedup: f64,
        max_memory_ratio: f64,
        tlc: RunObservation,
        backend_run: RunObservation,
    ) -> Self {
        let speedup = speedup(tlc.elapsed_seconds, backend_run.elapsed_seconds);
        let memory_ratio = memory_ratio(backend_run.peak_rss_bytes, tlc.peak_rss_bytes);
        let classification = classify_observations_with_limits(
            spec.expected_tlc_states,
            spec.expected_backend_states,
            spec.expected_tlc_error.as_deref(),
            spec.expected_backend_error.as_deref(),
            policy,
            min_speedup,
            max_memory_ratio,
            &tlc,
            &backend_run,
            speedup,
            memory_ratio,
        );
        Self {
            spec: spec.name.clone(),
            workers,
            case: case.to_string(),
            backend: backend_cli_name(backend).to_string(),
            passed: classification.class == CompareClass::Pass,
            class: classification.class,
            reason: classification.reason,
            tlc,
            backend_run,
            parity_states: classification.parity_states,
            parity_transitions: classification.parity_transitions,
            expected_tlc_states: spec.expected_tlc_states,
            expected_backend_states: spec.expected_backend_states,
            expected_tlc_error: spec.expected_tlc_error.clone(),
            expected_backend_error: spec.expected_backend_error.clone(),
            speedup_tlc_vs_backend: speedup,
            memory_ratio_backend_vs_tlc: memory_ratio,
            policy: policy_name(policy).to_string(),
            min_speedup,
            max_memory_ratio,
        }
    }
}

#[derive(Clone, Debug)]
struct Classification {
    class: CompareClass,
    reason: String,
    parity_states: bool,
    parity_transitions: bool,
}

fn classify_observations_with_limits(
    expected_tlc_states: Option<u64>,
    expected_backend_states: Option<u64>,
    expected_tlc_error: Option<&str>,
    expected_backend_error: Option<&str>,
    policy: SupremacyComparePolicy,
    min_speedup: f64,
    max_memory_ratio: f64,
    tlc: &RunObservation,
    backend: &RunObservation,
    speedup: Option<f64>,
    memory_ratio: Option<f64>,
) -> Classification {
    if expected_tlc_error.is_some_and(|expected| tlc.error_type.as_deref() != Some(expected)) {
        return classified(
            CompareClass::ExpectedErrorMismatch,
            format!(
                "TLC error {:?} did not match expected {:?}",
                tlc.error_type, expected_tlc_error
            ),
            false,
        );
    }
    if expected_backend_error
        .is_some_and(|expected| backend.error_type.as_deref() != Some(expected))
    {
        return classified(
            CompareClass::ExpectedErrorMismatch,
            format!(
                "backend error {:?} did not match expected {:?}",
                backend.error_type, expected_backend_error
            ),
            false,
        );
    }
    if expected_tlc_states.is_some_and(|expected| tlc.states_found != Some(expected)) {
        return classified(
            CompareClass::ExpectedStateMismatch,
            format!(
                "TLC states {:?} did not match expected {:?}",
                tlc.states_found, expected_tlc_states
            ),
            false,
        );
    }
    if expected_backend_states.is_some_and(|expected| backend.states_found != Some(expected)) {
        return classified(
            CompareClass::ExpectedStateMismatch,
            format!(
                "backend states {:?} did not match expected {:?}",
                backend.states_found, expected_backend_states
            ),
            false,
        );
    }

    let tlc_has_error = tlc.error_type.is_some();
    let backend_has_error = backend.error_type.is_some();
    if tlc_has_error || backend_has_error {
        if !tlc_has_error || !backend_has_error {
            return classified(
                CompareClass::ErrorMismatch,
                format!(
                    "error detection mismatch: TLC={:?} backend={:?}",
                    tlc.error_type, backend.error_type
                ),
                false,
            );
        }
        if !error_types_compatible(tlc.error_type.as_deref(), backend.error_type.as_deref()) {
            return classified(
                CompareClass::ErrorMismatch,
                format!(
                    "error type mismatch: TLC={:?} backend={:?}",
                    tlc.error_type, backend.error_type
                ),
                false,
            );
        }
        if let Some(failure) =
            classify_performance(policy, min_speedup, max_memory_ratio, speedup, memory_ratio)
        {
            return failure;
        }
        return classified(
            CompareClass::Pass,
            "compatible error outcome".to_string(),
            true,
        );
    }

    if let Some(error) = &tlc.error {
        return classified(
            CompareClass::TlcFailed,
            format!("TLC failed: {error}"),
            false,
        );
    }
    if let Some(error) = &backend.error {
        return classified(
            CompareClass::BackendFailed,
            format!("backend failed: {error}"),
            false,
        );
    }
    let parity_states = tlc.states_found == backend.states_found;
    if !parity_states {
        return classified(
            CompareClass::ParityFail,
            format!(
                "state-count parity failed: TLC={:?} backend={:?}",
                tlc.states_found, backend.states_found
            ),
            false,
        );
    }
    let parity_transitions = tlc.transitions == backend.transitions;
    if !parity_transitions && (tlc.transitions.is_some() || backend.transitions.is_some()) {
        return Classification {
            class: CompareClass::ParityFail,
            reason: format!(
                "transition-count parity failed: TLC={:?} backend={:?}",
                tlc.transitions, backend.transitions
            ),
            parity_states: true,
            parity_transitions: false,
        };
    }
    if let Some(failure) =
        classify_performance(policy, min_speedup, max_memory_ratio, speedup, memory_ratio)
    {
        return failure;
    }
    classified(CompareClass::Pass, "passed".to_string(), true)
}

fn classify_performance(
    policy: SupremacyComparePolicy,
    min_speedup: f64,
    max_memory_ratio: f64,
    speedup: Option<f64>,
    memory_ratio: Option<f64>,
) -> Option<Classification> {
    if policy_checks_speed(policy) {
        let Some(speedup) = speedup else {
            return Some(classified(
                CompareClass::MissingRuntime,
                "missing finite positive runtime for speed policy".to_string(),
                true,
            ));
        };
        if speedup < min_speedup {
            return Some(classified(
                CompareClass::SpeedFail,
                format!("speedup {speedup:.6}x below required {min_speedup:.6}x"),
                true,
            ));
        }
    }
    if policy_checks_memory(policy) {
        let Some(memory_ratio) = memory_ratio else {
            return Some(classified(
                CompareClass::MissingMemory,
                "missing positive peak RSS for memory policy".to_string(),
                true,
            ));
        };
        if memory_ratio > max_memory_ratio {
            return Some(classified(
                CompareClass::MemoryFail,
                format!(
                    "TY/TLC peak-RSS ratio {memory_ratio:.6}x above allowed {max_memory_ratio:.6}x"
                ),
                true,
            ));
        }
    }
    None
}

#[cfg(test)]
fn classify_observations(
    expected_tlc_states: Option<u64>,
    expected_backend_states: Option<u64>,
    expected_tlc_error: Option<&str>,
    expected_backend_error: Option<&str>,
    policy: SupremacyComparePolicy,
    min_speedup: f64,
    tlc: &RunObservation,
    backend: &RunObservation,
    speedup: Option<f64>,
) -> Classification {
    classify_observations_with_limits(
        expected_tlc_states,
        expected_backend_states,
        expected_tlc_error,
        expected_backend_error,
        policy,
        min_speedup,
        1.0,
        tlc,
        backend,
        speedup,
        memory_ratio(backend.peak_rss_bytes, tlc.peak_rss_bytes),
    )
}

fn classified(class: CompareClass, reason: String, parity_states: bool) -> Classification {
    Classification {
        class,
        reason,
        parity_states,
        parity_transitions: true,
    }
}

#[derive(Clone, Debug, Serialize)]
struct CompareReport {
    schema: &'static str,
    timestamp: String,
    backend: String,
    policy: String,
    mode: String,
    min_speedup: f64,
    max_memory_ratio: f64,
    passed: bool,
    total_rows: usize,
    failed_rows: usize,
    output_dir: PathBuf,
    workers: Vec<usize>,
    cases: Vec<EnvCase>,
    rows: Vec<CompareRow>,
}

impl CompareReport {
    fn new(
        args: &SupremacyCompareArgs,
        output_dir: PathBuf,
        cases: Vec<EnvCase>,
        rows: Vec<CompareRow>,
    ) -> Self {
        let failed_rows = rows.iter().filter(|row| !row.passed).count();
        Self {
            schema: COMPARE_REPORT_SCHEMA,
            timestamp: chrono::Utc::now().to_rfc3339(),
            backend: backend_cli_name(args.backend).to_string(),
            policy: policy_name(args.policy).to_string(),
            mode: mode_name(args.mode).to_string(),
            min_speedup: args.min_speedup,
            max_memory_ratio: args.max_memory_ratio,
            passed: failed_rows == 0,
            total_rows: rows.len(),
            failed_rows,
            output_dir,
            workers: args.workers.clone(),
            cases,
            rows,
        }
    }

    fn to_human(&self) -> String {
        let mut out = String::new();
        let status = if self.passed { "PASS" } else { "FAIL" };
        let _ = writeln!(
            out,
            "Supremacy compare {status}: {} rows, {} failed",
            self.total_rows, self.failed_rows
        );
        let _ = writeln!(
            out,
            "backend={} policy={} min_speedup={} max_memory_ratio={} cases={} output_dir={}",
            self.backend,
            self.policy,
            self.min_speedup,
            self.max_memory_ratio,
            self.cases
                .iter()
                .map(|case| case.name.as_str())
                .collect::<Vec<_>>()
                .join(","),
            self.output_dir.display()
        );
        for row in &self.rows {
            let row_status = if row.passed { "PASS" } else { "FAIL" };
            let speedup = row
                .speedup_tlc_vs_backend
                .map(|value| format!("{value:.3}x"))
                .unwrap_or_else(|| "n/a".to_string());
            let memory_ratio = row
                .memory_ratio_backend_vs_tlc
                .map(|value| format!("{value:.3}x"))
                .unwrap_or_else(|| "n/a".to_string());
            let _ = writeln!(
                out,
                "- {row_status} {} case={} workers={} class={:?} tlc_states={:?} backend_states={:?} tlc_peak_rss={} backend_peak_rss={} speedup={} memory_ratio={} reason={}",
                row.spec,
                row.case,
                row.workers,
                row.class,
                row.tlc.states_found,
                row.backend_run.states_found,
                fmt_bytes(row.tlc.peak_rss_bytes),
                fmt_bytes(row.backend_run.peak_rss_bytes),
                speedup,
                memory_ratio,
                row.reason
            );
        }
        out
    }

    fn to_markdown(&self) -> String {
        let mut lines = vec![
            "# Supremacy Compare".to_string(),
            String::new(),
            format!("Verdict: **{}**", if self.passed { "PASS" } else { "FAIL" }),
            format!("Backend: `{}`", self.backend),
            format!("Policy: `{}`", self.policy),
            format!("Minimum TLC/TY speedup: `{}`", self.min_speedup),
            format!(
                "Maximum TY/TLC peak-RSS ratio: `{}`",
                self.max_memory_ratio
            ),
            format!(
                "Cases: `{}`",
                self.cases
                    .iter()
                    .map(|case| case.name.as_str())
                    .collect::<Vec<_>>()
                    .join("`, `")
            ),
            format!("Output dir: `{}`", self.output_dir.display()),
            String::new(),
            "| Spec | Case | Workers | Class | TLC states | Backend states | TLC seconds | Backend seconds | TLC peak RSS | Backend peak RSS | Speedup | TY/TLC RSS | Reason |".to_string(),
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |".to_string(),
        ];
        for row in &self.rows {
            lines.push(format!(
                "| {} | {} | {} | {:?} | {} | {} | {:.3} | {:.3} | {} | {} | {} | {} | {} |",
                row.spec,
                row.case,
                row.workers,
                row.class,
                fmt_opt_u64(row.tlc.states_found),
                fmt_opt_u64(row.backend_run.states_found),
                row.tlc.elapsed_seconds,
                row.backend_run.elapsed_seconds,
                fmt_bytes(row.tlc.peak_rss_bytes),
                fmt_bytes(row.backend_run.peak_rss_bytes),
                row.speedup_tlc_vs_backend
                    .map(|value| format!("{value:.3}x"))
                    .unwrap_or_else(|| "n/a".to_string()),
                row.memory_ratio_backend_vs_tlc
                    .map(|value| format!("{value:.3}x"))
                    .unwrap_or_else(|| "n/a".to_string()),
                row.reason.replace('|', "\\|")
            ));
        }
        lines.push(String::new());
        lines.join("\n")
    }
}

fn print_report(report: &CompareReport, format: SupremacyOutputFormat) -> Result<()> {
    match format {
        SupremacyOutputFormat::Json => println!("{}", serde_json::to_string_pretty(report)?),
        SupremacyOutputFormat::Markdown => println!("{}", report.to_markdown()),
        SupremacyOutputFormat::Human => print!("{}", report.to_human()),
    }
    Ok(())
}

fn speedup(tlc_seconds: f64, backend_seconds: f64) -> Option<f64> {
    if tlc_seconds.is_finite()
        && backend_seconds.is_finite()
        && tlc_seconds > 0.0
        && backend_seconds > 0.0
    {
        Some(tlc_seconds / backend_seconds)
    } else {
        None
    }
}

fn memory_ratio(backend_bytes: Option<u64>, tlc_bytes: Option<u64>) -> Option<f64> {
    let backend_bytes = backend_bytes?;
    let tlc_bytes = tlc_bytes?;
    if backend_bytes == 0 || tlc_bytes == 0 {
        return None;
    }
    Some(backend_bytes as f64 / tlc_bytes as f64)
}

fn backend_env(backend: SupremacyCompareBackend) -> BTreeMap<String, String> {
    match backend {
        SupremacyCompareBackend::Interpreter => BTreeMap::from([
            ("TY_trust_cg".to_string(), "0".to_string()),
            ("TY_TRUST_CG_BFS".to_string(), "0".to_string()),
        ]),
        SupremacyCompareBackend::TrustCg => policy::full_native_fused_protected_env(),
    }
}

#[derive(Clone, Debug)]
enum TlcRunner {
    Executable {
        tlc_bin: PathBuf,
        tla_library: Option<PathBuf>,
    },
    Java {
        tlc_jar: PathBuf,
        community_modules: Option<PathBuf>,
        tla_library: Option<PathBuf>,
    },
}

fn resolve_tlc_runner(args: &SupremacyCompareArgs, repo_root: &Path) -> Result<TlcRunner> {
    let tla_library = resolve_tla_library(args, repo_root);
    if let Some(tlc_bin) = args
        .tlc_bin
        .clone()
        .or_else(|| non_empty_env_path(ENV_TLC_BIN))
    {
        validate_file(&tlc_bin)
            .with_context(|| format!("validate TLC executable {}", tlc_bin.display()))?;
        return Ok(TlcRunner::Executable {
            tlc_bin,
            tla_library,
        });
    }

    let tlc_jar = args
        .tlc_jar
        .clone()
        .map(Ok)
        .unwrap_or_else(default_tlc_jar)?;
    validate_file(&tlc_jar).with_context(|| format!("validate TLC jar {}", tlc_jar.display()))?;
    let community_modules = args
        .community_modules
        .clone()
        .or_else(|| non_empty_env_path(ENV_COMMUNITY_MODULES))
        .or_else(default_community_modules_jar);
    if let Some(community_modules) = &community_modules {
        validate_file(community_modules).with_context(|| {
            format!(
                "validate CommunityModules jar {}",
                community_modules.display()
            )
        })?;
    }
    Ok(TlcRunner::Java {
        tlc_jar,
        community_modules,
        tla_library,
    })
}

fn resolve_tla_library(args: &SupremacyCompareArgs, repo_root: &Path) -> Option<PathBuf> {
    args.tla_library
        .clone()
        .or_else(|| non_empty_env_path(ENV_TLA_LIBRARY))
        .or_else(|| non_empty_env_path(ENV_TLA_PLUS_LIBRARY))
        .or_else(|| {
            let repo_library = repo_root.join(DEFAULT_TLA_LIBRARY);
            repo_library.is_dir().then_some(repo_library)
        })
}

fn default_community_modules_jar() -> Option<PathBuf> {
    env::var_os("HOME")
        .map(|home| PathBuf::from(home).join(DEFAULT_COMMUNITY_MODULES_JAR))
        .filter(|path| path.is_file())
}

fn tlc_classpath(tlc_jar: &Path, community_modules: Option<&Path>) -> Result<String> {
    let mut paths = vec![tlc_jar.to_path_buf()];
    if let Some(community_modules) = community_modules {
        paths.push(community_modules.to_path_buf());
    }
    let classpath = env::join_paths(paths).context("build TLC classpath")?;
    Ok(classpath.to_string_lossy().to_string())
}

fn backend_cli_name(backend: SupremacyCompareBackend) -> &'static str {
    match backend {
        SupremacyCompareBackend::Interpreter => "interpreter",
        SupremacyCompareBackend::TrustCg => "trust-cg",
    }
}

fn policy_name(policy: SupremacyComparePolicy) -> &'static str {
    match policy {
        SupremacyComparePolicy::Parity => "parity",
        SupremacyComparePolicy::ParityAndSpeed => "parity-and-speed",
        SupremacyComparePolicy::ParityAndSpeedAndMemory => "parity-and-speed-and-memory",
    }
}

fn mode_name(mode: SupremacyMode) -> &'static str {
    match mode {
        SupremacyMode::Warn => "warn",
        SupremacyMode::Enforce => "enforce",
    }
}

fn resolve_source_path(repo_root: &Path, examples_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        let examples_path = examples_dir.join(path);
        if examples_path.exists() {
            examples_path
        } else {
            repo_root.join(path)
        }
    }
}

fn default_examples_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join("tlaplus-examples")
            .join("specifications")
    })
}

fn default_tlc_jar() -> Result<PathBuf> {
    if let Some(path) = non_empty_env_path(ENV_TYTOOLS_JAR) {
        return Ok(path);
    }
    if let Some(path) = non_empty_env_path(ENV_TLC_JAR) {
        return Ok(path);
    }
    let home = env::var_os("HOME").context("HOME is not set; pass --tlc-jar")?;
    Ok(PathBuf::from(home).join(DEFAULT_TLC_JAR))
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    let value = env::var_os(name)?;
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value))
    }
}

fn validate_file(path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        bail!("required file not found: {}", path.display())
    }
}

fn validate_spec_files(tla_path: &Path, cfg_path: &Path) -> Result<()> {
    validate_file(tla_path).with_context(|| format!("validate TLA file {}", tla_path.display()))?;
    validate_file(cfg_path).with_context(|| format!("validate config file {}", cfg_path.display()))
}

fn absolutize(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn default_output_dir(command: &str) -> PathBuf {
    Path::new("reports").join("perf").join(format!(
        "{}-supremacy-{}",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        command
    ))
}

fn safe_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "spec".to_string()
    } else {
        out
    }
}

fn fmt_opt_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_bytes(value: Option<u64>) -> String {
    value
        .map(|bytes| {
            if bytes >= 1024 * 1024 * 1024 {
                format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            } else if bytes >= 1024 * 1024 {
                format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
            } else if bytes >= 1024 {
                format!("{:.1} KiB", bytes as f64 / 1024.0)
            } else {
                format!("{bytes} B")
            }
        })
        .unwrap_or_else(|| "n/a".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> SupremacyCompareArgs {
        SupremacyCompareArgs {
            spec_source: SupremacyCompareSpecSource::Baseline,
            baseline: PathBuf::from("tests/tlc_comparison/spec_baseline.json"),
            specs: vec![],
            tla: None,
            config: None,
            backend: SupremacyCompareBackend::Interpreter,
            workers: vec![1],
            mode: SupremacyMode::Enforce,
            policy: SupremacyComparePolicy::Parity,
            min_speedup: 1.0,
            max_memory_ratio: 1.0,
            output_dir: None,
            ty_bin: None,
            tlc_jar: None,
            tlc_bin: None,
            community_modules: None,
            tla_library: None,
            timeout: 300,
            ty_flag: vec![],
            cases: vec![],
            ty_env: vec![],
            case_env: vec![],
            format: SupremacyOutputFormat::Human,
        }
    }

    fn obs(states_found: Option<u64>, elapsed_seconds: f64, error: Option<&str>) -> RunObservation {
        obs_with_transitions(states_found, None, elapsed_seconds, error)
    }

    fn obs_with_transitions(
        states_found: Option<u64>,
        transitions: Option<u64>,
        elapsed_seconds: f64,
        error: Option<&str>,
    ) -> RunObservation {
        let error_type = error.map(str::to_string);
        RunObservation {
            tool: "test".to_string(),
            mode: "test".to_string(),
            status: status_for_error(error).to_string(),
            elapsed_seconds,
            peak_rss_bytes: Some(1024),
            states_found,
            transitions,
            states_generated: None,
            returncode: i32::from(error.is_some()),
            timed_out: false,
            error_type,
            error: error.map(str::to_string),
            artifact_dir: "artifact".to_string(),
        }
    }

    #[test]
    fn parity_policy_passes_matching_states_without_speed_requirement() {
        let tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let backend = obs_with_transitions(Some(10), Some(20), 10.0, None);

        let result = classify_observations(
            Some(10),
            Some(10),
            None,
            None,
            SupremacyComparePolicy::Parity,
            2.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::Pass);
        assert!(result.parity_states);
        assert!(result.parity_transitions);
    }

    #[test]
    fn parity_policy_rejects_state_drift() {
        let tlc = obs(Some(10), 1.0, None);
        let backend = obs(Some(11), 0.5, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(!result.parity_states);
    }

    #[test]
    fn parity_policy_rejects_transition_drift_after_state_parity() {
        let tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let backend = obs_with_transitions(Some(10), Some(21), 0.5, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(result.parity_states);
        assert!(!result.parity_transitions);
        assert!(result.reason.contains("transition-count parity failed"));
    }

    #[test]
    fn parity_policy_rejects_one_sided_missing_transition_count() {
        let tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let backend = obs_with_transitions(Some(10), None, 0.5, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(result.parity_states);
        assert!(!result.parity_transitions);
        assert!(result.reason.contains("backend=None"));
    }

    #[test]
    fn speed_policy_rejects_transition_drift_before_speed_check() {
        let tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let backend = obs_with_transitions(Some(10), Some(21), 2.0, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeed,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(!result.parity_transitions);
    }

    #[test]
    fn speed_policy_rejects_below_min_speedup() {
        let tlc = obs(Some(10), 1.0, None);
        let backend = obs(Some(10), 2.0, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeed,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::SpeedFail);
        assert!(result.parity_states);
    }

    #[test]
    fn speed_policy_accepts_min_speedup_boundary() {
        let tlc = obs_with_transitions(Some(10), Some(20), 2.0, None);
        let backend = obs_with_transitions(Some(10), Some(20), 2.0, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeed,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::Pass);
    }

    #[test]
    fn memory_policy_rejects_peak_rss_above_limit() {
        let mut tlc = obs(Some(10), 2.0, None);
        let mut backend = obs(Some(10), 1.0, None);
        tlc.peak_rss_bytes = Some(1000);
        backend.peak_rss_bytes = Some(1100);

        let result = classify_observations_with_limits(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.0,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
            memory_ratio(backend.peak_rss_bytes, tlc.peak_rss_bytes),
        );

        assert_eq!(result.class, CompareClass::MemoryFail);
        assert!(result.reason.contains("peak-RSS ratio"));
    }

    #[test]
    fn memory_policy_accepts_configured_ratio_boundary() {
        let mut tlc = obs(Some(10), 2.0, None);
        let mut backend = obs(Some(10), 1.0, None);
        tlc.peak_rss_bytes = Some(1000);
        backend.peak_rss_bytes = Some(900);

        let result = classify_observations_with_limits(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.0,
            0.9,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
            memory_ratio(backend.peak_rss_bytes, tlc.peak_rss_bytes),
        );

        assert_eq!(result.class, CompareClass::Pass);
    }

    #[test]
    fn memory_policy_rejects_missing_peak_rss() {
        let tlc = obs(Some(10), 2.0, None);
        let mut backend = obs(Some(10), 1.0, None);
        backend.peak_rss_bytes = None;

        let result = classify_observations_with_limits(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.0,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
            memory_ratio(backend.peak_rss_bytes, tlc.peak_rss_bytes),
        );

        assert_eq!(result.class, CompareClass::MissingMemory);
    }

    #[test]
    fn compatible_error_outcomes_pass_before_state_parity() {
        let tlc = obs(None, 1.0, Some("invariant"));
        let backend = obs(None, 0.5, Some("safety"));

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::Pass);
    }

    #[test]
    fn compatible_error_outcomes_still_enforce_efficiency_policy() {
        let mut tlc = obs(None, 1.0, Some("invariant"));
        let mut backend = obs(None, 2.0, Some("safety"));
        tlc.peak_rss_bytes = Some(1000);
        backend.peak_rss_bytes = Some(2000);

        let result = classify_observations_with_limits(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.0,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
            memory_ratio(backend.peak_rss_bytes, tlc.peak_rss_bytes),
        );

        assert_eq!(result.class, CompareClass::SpeedFail);
    }

    #[test]
    fn compatible_error_outcomes_still_enforce_expected_counts() {
        let tlc = obs(Some(10), 1.0, Some("invariant"));
        let backend = obs(Some(11), 0.5, Some("safety"));

        let result = classify_observations(
            Some(10),
            Some(10),
            Some("invariant"),
            Some("safety"),
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ExpectedStateMismatch);
    }

    #[test]
    fn baseline_error_aliases_normalize_before_comparison() {
        let tlc = SpecBaselineMode {
            states: Some(10),
            error_type: Some("invariant".to_string()),
            status: Some("fail".to_string()),
        };
        let backend = SpecBaselineMode {
            states: Some(10),
            error_type: Some("invariant_violation".to_string()),
            status: Some("pass".to_string()),
        };

        assert_eq!(expected_error_type(&tlc).as_deref(), Some("invariant"));
        assert_eq!(expected_error_type(&backend).as_deref(), Some("invariant"));
    }

    #[test]
    fn tool_specific_expected_counts_do_not_override_cross_tool_parity() {
        let tlc = obs(Some(87898), 20.0, None);
        let backend = obs(Some(27242), 10.0, None);

        let result = classify_observations(
            Some(87898),
            Some(27242),
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(!result.parity_states);
    }

    #[test]
    fn java_tlc_runner_uses_auditable_single_thread_jvm_profile() {
        let argv = tlc_java_single_thread_base_argv();

        assert_eq!(argv[0], "java");
        for arg in tlc_java_single_thread_args() {
            assert!(argv.contains(&(*arg).to_string()), "{arg}");
        }
        assert!(!argv.contains(&"-XX:+UseParallelGC".to_string()));
    }

    #[test]
    fn enforced_speed_compare_rejects_opaque_tlc_executable_runner() {
        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeed;
        args.tlc_bin = Some(PathBuf::from("/tmp/tlc-wrapper"));

        let error = validate_args(&args).expect_err("opaque TLC runner should be rejected");

        assert!(error.to_string().contains("auditable Java TLC runner"));
    }

    #[test]
    fn enforced_speed_compare_rejects_multiworker_or_ty_only_flag_claims() {
        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeed;
        args.workers = vec![1, 2];

        let error = validate_args(&args).expect_err("multiworker speed claim should be rejected");
        assert!(error.to_string().contains("--workers 1"));

        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeed;
        args.ty_flag = vec!["--max-depth".to_string(), "3".to_string()];

        let error = validate_args(&args).expect_err("TY-only flags should be rejected");
        assert!(error.to_string().contains("--ty-flag"));
    }

    #[test]
    fn compare_rejects_invalid_memory_ratio() {
        let mut args = test_args();
        args.max_memory_ratio = 0.0;

        let error = validate_args(&args).expect_err("zero memory ratio should be rejected");

        assert!(error.to_string().contains("--max-memory-ratio"));
    }

    #[test]
    fn trust_cg_backend_env_uses_strict_native_fused_launch_controls() {
        let env = backend_env(SupremacyCompareBackend::TrustCg);

        for (key, value) in [
            ("TY_trust_cg", "1"),
            ("TY_TRUST_CG_BFS", "1"),
            ("TY_TRUST_CG_EXISTS", "1"),
            ("TY_BYTECODE_VM", "1"),
            ("TY_BYTECODE_VM_STATS", "1"),
            ("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST", "strict"),
            ("TY_TRUST_CG_NATIVE_FUSED_STRICT", "1"),
            ("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS", "27"),
            ("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP", "1"),
            ("TY_DISABLE_ARTIFACT_CACHE", "1"),
        ] {
            assert_eq!(env.get(key).map(String::as_str), Some(value), "{key}");
        }
        // Count-parity (auto-POR/auto-symmetry off) is the `--no-reduction`
        // CLI flag in the child argv, never an env pin: the child `ty check`
        // ignores ambient TY_AUTO_POR / TY_AUTO_SYMMETRY.
        assert_eq!(env.get("TY_AUTO_POR"), None);
        assert_eq!(env.get("TY_AUTO_SYMMETRY"), None);
    }

    #[test]
    fn compare_cases_default_to_single_default_case() {
        let args = test_args();

        let cases = resolve_cases(&args).expect("default case should resolve");

        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].name, DEFAULT_CASE);
        assert!(cases[0].env_overrides.is_empty());
    }

    #[test]
    fn compare_cases_apply_global_env_to_every_case() {
        let mut args = test_args();
        args.cases = vec!["control".to_string(), "treatment".to_string()];
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()];

        let cases = resolve_cases(&args).expect("global env should resolve");

        assert_eq!(cases.len(), 2);
        for case in cases {
            assert_eq!(
                case.env_overrides
                    .get("TY_PARALLEL_READONLY_VALUE_CACHES")
                    .map(String::as_str),
                Some("1")
            );
        }
    }

    #[test]
    fn compare_case_env_applies_to_named_case_and_overrides_global() {
        let mut args = test_args();
        args.cases = vec!["control".to_string(), "treatment".to_string()];
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES=0".to_string()];
        args.case_env = vec!["treatment:TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()];

        let cases = resolve_cases(&args).expect("case env should resolve");
        let control = cases.iter().find(|case| case.name == "control").unwrap();
        let treatment = cases.iter().find(|case| case.name == "treatment").unwrap();

        assert_eq!(
            control
                .env_overrides
                .get("TY_PARALLEL_READONLY_VALUE_CACHES")
                .map(String::as_str),
            Some("0")
        );
        assert_eq!(
            treatment
                .env_overrides
                .get("TY_PARALLEL_READONLY_VALUE_CACHES")
                .map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn compare_cases_reject_duplicate_case_names() {
        let mut args = test_args();
        args.cases = vec!["same".to_string(), "same".to_string()];

        let error = resolve_cases(&args).expect_err("duplicate cases should fail");

        assert!(error.to_string().contains("duplicate --case"));
    }

    #[test]
    fn compare_case_env_rejects_unknown_case() {
        let mut args = test_args();
        args.cases = vec!["control".to_string()];
        args.case_env = vec!["treatment:TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()];

        let error = resolve_cases(&args).expect_err("unknown case should fail");

        assert!(error.to_string().contains("unknown case"));
    }

    #[test]
    fn compare_env_rejects_malformed_assignment() {
        let mut args = test_args();
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES".to_string()];

        let error = resolve_cases(&args).expect_err("malformed env should fail");

        assert!(error.to_string().contains("KEY=VALUE"));
    }

    #[test]
    fn compare_env_rejects_protected_backend_keys() {
        let mut args = test_args();
        args.backend = SupremacyCompareBackend::TrustCg;
        args.ty_env = vec!["TY_trust_cg=0".to_string()];

        let error = resolve_cases(&args).expect_err("protected env should fail");

        assert!(error.to_string().contains("protected backend env key"));
    }

    #[test]
    fn compare_env_accepts_allowed_non_semantic_case_key() {
        let mut args = test_args();
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()];

        let cases = resolve_cases(&args).expect("allowed env should resolve");

        assert_eq!(
            cases[0]
                .env_overrides
                .get("TY_PARALLEL_READONLY_VALUE_CACHES")
                .map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn compare_env_rejects_disallowed_ty_keys() {
        let mut args = test_args();
        args.ty_env = vec!["TY_EXPERIMENT=1".to_string()];

        let error = resolve_cases(&args).expect_err("disallowed TY env should fail");

        assert!(error
            .to_string()
            .contains("not allowed for compare env cases"));
    }

    #[test]
    fn compare_env_rejects_inverse_backend_flags() {
        for key in [
            "TY_NO_FLAT_BFS",
            "TY_NO_COMPILED_BFS",
            "TY_TRUST_CG_DISABLE_COMPILED_BFS_LEVEL",
            "TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP",
            "TY_TRUST_CG_ENTRY_COUNTER_GATE",
        ] {
            let mut args = test_args();
            args.backend = SupremacyCompareBackend::TrustCg;
            args.ty_env = vec![format!("{key}=1")];

            let error = resolve_cases(&args).expect_err("inverse backend env should fail");

            assert!(
                error
                    .to_string()
                    .contains("not allowed for compare env cases")
                    || error.to_string().contains("protected backend env key"),
                "{key}: {error}"
            );
        }
    }

    #[test]
    fn compare_case_env_rejects_inverse_backend_flags() {
        for key in [
            "TY_NO_FLAT_BFS",
            "TY_NO_COMPILED_BFS",
            "TY_TRUST_CG_DISABLE_COMPILED_BFS_LEVEL",
            "TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP",
            "TY_TRUST_CG_ENTRY_COUNTER_GATE",
        ] {
            let mut args = test_args();
            args.backend = SupremacyCompareBackend::TrustCg;
            args.cases = vec!["control".to_string()];
            args.case_env = vec![format!("control:{key}=1")];

            let error = resolve_cases(&args).expect_err("inverse backend case env should fail");

            assert!(
                error
                    .to_string()
                    .contains("not allowed for compare env cases")
                    || error.to_string().contains("protected backend env key"),
                "{key}: {error}"
            );
        }
    }

    #[test]
    fn compare_env_rejects_invalid_allowed_case_key_value() {
        let mut args = test_args();
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES=true".to_string()];

        let error = resolve_cases(&args).expect_err("invalid allowed env value should fail");

        assert!(error.to_string().contains("accepts only"));
    }

    #[test]
    fn compare_env_rejects_non_ty_keys() {
        let mut args = test_args();
        args.ty_env = vec!["JAVA_TOOL_OPTIONS=-Xmx1g".to_string()];

        let error = resolve_cases(&args).expect_err("non-TY env should fail");

        assert!(error.to_string().contains("only TY_* keys"));
    }

    #[test]
    fn compare_report_serializes_case_inventory_and_rows() {
        let args = test_args();
        let case = EnvCase {
            name: "control".to_string(),
            env_overrides: BTreeMap::from([(
                "TY_PARALLEL_READONLY_VALUE_CACHES".to_string(),
                "1".to_string(),
            )]),
        };
        let spec = CompareSpec {
            name: "Spec".to_string(),
            tla_path: PathBuf::from("Spec.tla"),
            cfg_path: PathBuf::from("Spec.cfg"),
            expected_tlc_states: None,
            expected_backend_states: None,
            expected_tlc_error: None,
            expected_backend_error: None,
        };
        let row = CompareRow::classify(
            &spec,
            1,
            &case.name,
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::Parity,
            1.0,
            1.0,
            obs(Some(1), 1.0, None),
            obs(Some(1), 0.5, None),
        );
        let report = CompareReport::new(&args, PathBuf::from("out"), vec![case], vec![row]);
        let json = serde_json::to_value(&report).expect("report serializes");

        assert_eq!(json["cases"][0]["name"], "control");
        assert_eq!(
            json["cases"][0]["env_overrides"]["TY_PARALLEL_READONLY_VALUE_CACHES"],
            "1"
        );
        assert_eq!(json["rows"][0]["case"], "control");
        assert!(report.to_markdown().contains("| Spec | control | 1 |"));
    }
}
