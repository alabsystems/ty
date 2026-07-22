// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty supremacy` command family.
//!
//! This module owns the active single-thread supremacy smoke, benchmark, and
//! gate CLI. The benchmark runner executes TLC, interpreter, and trust-codegen
//! subprocesses, and the gate can either run benchmark collection itself or
//! evaluate an existing `summary.json`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::cli_schema::{
    SupremacyAntiOverfitArgs, SupremacyArgs, SupremacyBenchmarkArgs, SupremacyCommand,
    SupremacyCommonArgs, SupremacyCompareArgs, SupremacyGateArgs, SupremacyGateMode,
    SupremacyMatrixArgs, SupremacyMatrixFullSuiteArgs, SupremacyMatrixRuntimeScope, SupremacyMode,
    SupremacyOutputFormat, SupremacyReproduceArgs, SupremacySmokeArgs, SupremacySoundnessSweepArgs,
};

mod anti_overfit;
mod benchmark;
mod compare;
mod matrix;
mod matrix_refresh;
#[cfg(test)]
mod matrix_runtime_cli_tests;
mod parse;
mod policy;
mod reproduce;
mod runner;
mod smoke;
mod soundness_sweep;
mod summary;
mod verdict;

use policy::{PlannedGate, SupremacyPolicy};

const DEFAULT_POLICY_FILE: &str = "tests/tlc_comparison/single_thread_supremacy_gate.json";
const DEFAULT_SPECS: &[&str] = &["CoffeeCan1000BeansSafety", "EWD998Small", "MCLamportMutex"];
const ENV_GATE_MODE: &str = "TY_SINGLE_THREAD_SUPREMACY_GATE_MODE";
const ENV_POLICY_FILE: &str = "TY_SINGLE_THREAD_SUPREMACY_POLICY_FILE";
const ENV_RUNS: &str = "TY_SINGLE_THREAD_SUPREMACY_RUNS";
const ENV_SKIP: &str = "TY_SINGLE_THREAD_SUPREMACY_SKIP";
const ENV_TY_BIN: &str = "TY_SINGLE_THREAD_SUPREMACY_TY_BIN";
const TLC_JAVA_SINGLE_THREAD_ARGS: &[&str] = &[
    "-XX:ActiveProcessorCount=1",
    "-XX:+UseSerialGC",
    "-Xms64m",
    "-Xmx4g",
];

fn tlc_java_single_thread_args() -> &'static [&'static str] {
    TLC_JAVA_SINGLE_THREAD_ARGS
}

fn tlc_java_single_thread_base_argv() -> Vec<String> {
    let mut argv = vec!["java".to_string()];
    argv.extend(
        tlc_java_single_thread_args()
            .iter()
            .map(|arg| (*arg).to_string()),
    );
    argv
}

pub(crate) fn cmd_supremacy(args: SupremacyArgs) -> Result<()> {
    match args.command {
        SupremacyCommand::Smoke(args) => cmd_smoke(args),
        SupremacyCommand::Benchmark(args) => cmd_benchmark(args),
        SupremacyCommand::Gate(args) => cmd_gate(args),
        SupremacyCommand::Compare(args) => cmd_compare(args),
        SupremacyCommand::Reproduce(args) => cmd_reproduce(args),
        SupremacyCommand::AntiOverfit(args) => cmd_anti_overfit(args),
        SupremacyCommand::Matrix(args) => cmd_matrix(args),
        SupremacyCommand::MatrixFullSuite(args) => cmd_matrix_full_suite(args),
        SupremacyCommand::SoundnessSweep(args) => cmd_soundness_sweep(args),
    }
}

fn cmd_soundness_sweep(args: SupremacySoundnessSweepArgs) -> Result<()> {
    soundness_sweep::run(args)
}

fn cmd_smoke(args: SupremacySmokeArgs) -> Result<()> {
    let prepared = PreparedSupremacy::prepare("smoke", &args.common, None, None, None)?;
    smoke::run_smoke(prepared)
}

fn cmd_benchmark(args: SupremacyBenchmarkArgs) -> Result<()> {
    if args.runs == 0 {
        bail!("--runs must be >= 1");
    }
    let prepared =
        PreparedSupremacy::prepare("benchmark", &args.common, Some(args.runs), None, None)?;
    let report = benchmark::run_benchmark(&prepared)?;
    if !report.passed {
        bail!(
            "ty supremacy benchmark completed but parity/run checks failed; see {}",
            report.summary_path.display()
        );
    }
    Ok(())
}

fn cmd_gate(args: SupremacyGateArgs) -> Result<()> {
    let Some(args) = ResolvedGateArgs::resolve(args)? else {
        return Ok(());
    };
    let mut prepared = PreparedSupremacy::prepare(
        "gate",
        &args.common,
        Some(args.runs),
        args.gate_mode,
        Some(args.mode),
    )?;
    if args.mode == SupremacyMode::Enforce {
        prepared.validate_enforce_env_overrides()?;
    }
    run_gate_anti_overfit(&prepared, args.mode)?;
    if let Some(summary_json) = args.summary_json {
        if args.common.output_dir.is_none() {
            if let Some(parent) = summary_json.parent() {
                prepared.output_dir = parent.to_path_buf();
            }
        }
        let passed = verdict::evaluate_and_write(&prepared, &summary_json)?;
        match prepared.format {
            SupremacyOutputFormat::Human | SupremacyOutputFormat::Markdown => {
                let status = if passed { "PASS" } else { "FAIL" };
                eprintln!(
                    "[supremacy] {status}: policy verdict written to {}",
                    prepared.output_dir.join("policy_verdict.json").display()
                );
            }
            SupremacyOutputFormat::Json => {
                println!(
                    "{}",
                    json!({
                        "status": if passed { "pass" } else { "fail" },
                        "policy_verdict": prepared.output_dir.join("policy_verdict.json"),
                    })
                );
            }
        }
        if !passed && args.mode == SupremacyMode::Enforce {
            bail!("ty supremacy gate policy verdict failed")
        }
        return Ok(());
    }
    let benchmark_report = match benchmark::run_benchmark(&prepared) {
        Ok(report) => report,
        Err(err) => {
            if args.mode == SupremacyMode::Enforce {
                return Err(err).context("ty supremacy gate benchmark execution failed");
            }
            eprintln!("[supremacy] WARNING: benchmark execution failed: {err:#}");
            return Ok(());
        }
    };
    let passed = verdict::evaluate_and_write(&prepared, &benchmark_report.summary_path)?;
    match prepared.format {
        SupremacyOutputFormat::Human | SupremacyOutputFormat::Markdown => {
            let status = if passed { "PASS" } else { "FAIL" };
            eprintln!(
                "[supremacy] {status}: benchmark summary and policy verdict written to {}",
                prepared.output_dir.display()
            );
        }
        SupremacyOutputFormat::Json => {
            println!(
                "{}",
                json!({
                    "status": if passed { "pass" } else { "fail" },
                    "summary": benchmark_report.summary_path,
                    "policy_verdict": prepared.output_dir.join("policy_verdict.json"),
                })
            );
        }
    }
    if !passed && args.mode == SupremacyMode::Enforce {
        bail!("ty supremacy gate policy verdict failed")
    }
    Ok(())
}

fn cmd_anti_overfit(args: SupremacyAntiOverfitArgs) -> Result<()> {
    anti_overfit::run(args)
}

fn cmd_compare(args: SupremacyCompareArgs) -> Result<()> {
    compare::run(args)
}

fn cmd_reproduce(args: SupremacyReproduceArgs) -> Result<()> {
    reproduce::run(args)
}

fn run_gate_anti_overfit(prepared: &PreparedSupremacy, mode: SupremacyMode) -> Result<()> {
    run_gate_anti_overfit_scan(prepared, mode, None)
}

fn run_gate_anti_overfit_scan(
    prepared: &PreparedSupremacy,
    mode: SupremacyMode,
    baseline_path: Option<&Path>,
) -> Result<()> {
    let report = match anti_overfit::scan(anti_overfit::AntiOverfitScanInput {
        policy_path: &prepared.policy_path,
        policy: &prepared.policy,
        baseline_path,
        scan_roots: &[],
        include_comments: false,
    }) {
        Ok(report) => report,
        Err(err) if mode == SupremacyMode::Warn => {
            eprintln!("[supremacy] WARNING: anti-overfit scan failed: {err:#}");
            return Ok(());
        }
        Err(err) => return Err(err).context("ty supremacy gate anti-overfit scan failed"),
    };
    if !report.has_findings() {
        return Ok(());
    }

    let findings = report.finding_count();
    if mode == SupremacyMode::Enforce {
        bail!("ty supremacy gate anti-overfit scan found {findings} forbidden corpus references");
    }
    eprintln!(
        "[supremacy] WARNING: anti-overfit scan found {findings} forbidden corpus references; continuing because --mode warn"
    );
    Ok(())
}

fn matrix_output_summary<'a>(
    initial: &'a matrix::SupremacyMatrixSummary,
    refreshed: Option<&'a matrix::SupremacyMatrixSummary>,
) -> &'a matrix::SupremacyMatrixSummary {
    refreshed.unwrap_or(initial)
}

fn cmd_matrix(args: SupremacyMatrixArgs) -> Result<()> {
    matrix::validate_matrix_enforce_inputs(&args)?;
    let matrix_policy = match &args.policy {
        Some(path) => policy::load_matrix_policy(path)?,
        None => policy::MatrixPolicy::default(),
    };
    let summary = matrix::classify_baseline_path_with_policy(&args.baseline, &matrix_policy)?;
    let refreshed_summary = matrix::collect_missing_runtime_path(&args, &summary, &matrix_policy)?;
    let enforce_summary = matrix_output_summary(&summary, refreshed_summary.as_ref());
    matrix::print_summary(enforce_summary, args.format)?;
    let blockers = enforce_summary.enforce_blocker_count();
    if args.mode == SupremacyMode::Enforce && blockers > 0 {
        bail!(
            "all-runnable supremacy matrix has {blockers} non-passing rows out of {}",
            enforce_summary.total_rows()
        );
    }
    Ok(())
}

fn cmd_matrix_full_suite(args: SupremacyMatrixFullSuiteArgs) -> Result<()> {
    cmd_matrix(SupremacyMatrixArgs {
        baseline: args.baseline,
        policy: args.policy,
        mode: args.mode,
        format: args.format,
        refresh_runtime: true,
        runtime_scope: SupremacyMatrixRuntimeScope::AllRunnable,
        runtime_output_dir: args.runtime_output_dir,
        runtime_limit: None,
        runtime_specs: Vec::new(),
        runtime_timeout: args.runtime_timeout,
        production_runtime: args.production_runtime,
        runtime_ty_bin: args.runtime_ty_bin,
        allow_debug_runtime: args.allow_debug_runtime,
        runtime_tlc_jar: args.runtime_tlc_jar,
        runtime_community_modules: args.runtime_community_modules,
        runtime_tla_library: args.runtime_tla_library,
    })
}

#[derive(Debug)]
struct ResolvedGateArgs {
    common: SupremacyCommonArgs,
    mode: SupremacyMode,
    gate_mode: Option<SupremacyGateMode>,
    runs: usize,
    summary_json: Option<PathBuf>,
}

impl ResolvedGateArgs {
    fn resolve(mut args: SupremacyGateArgs) -> Result<Option<Self>> {
        let mode = args.mode.unwrap_or(SupremacyMode::Enforce);
        let explicit_ty_bin = args.common.ty_bin.is_some();
        let legacy_env_ty_bin = non_empty_env_path(ENV_TY_BIN);
        if mode == SupremacyMode::Enforce && env::var_os(ENV_POLICY_FILE).is_some() {
            bail!("{ENV_POLICY_FILE} is not allowed in enforce mode");
        }
        if mode == SupremacyMode::Enforce && args.summary_json.is_none() {
            if explicit_ty_bin {
                bail!(
                    "--ty-bin is not allowed for enforce-mode gate collection; omit it so the Rust gate builds a fresh child binary"
                );
            }
            if legacy_env_ty_bin.is_some() {
                bail!(
                    "{ENV_TY_BIN} is not allowed for enforce-mode gate collection; unset it so the Rust gate builds a fresh child binary"
                );
            }
            if !args.common.ty_flag.is_empty() {
                bail!(
                    "--ty-flag is not allowed for enforce-mode gate collection; encode shared model-checking settings in the TLA+/cfg files instead"
                );
            }
            if !args.common.interp_env.is_empty() {
                bail!(
                    "--interp-env is not allowed for enforce-mode gate collection; use the policy-controlled launch settings"
                );
            }
        }
        if mode != SupremacyMode::Enforce {
            if let Some(policy_file) = env::var_os(ENV_POLICY_FILE) {
                args.common.policy = Some(PathBuf::from(policy_file));
            }
        }
        if env::var(ENV_SKIP).ok().as_deref() == Some("1") {
            if mode == SupremacyMode::Enforce {
                bail!("{ENV_SKIP} is not allowed in enforce mode");
            }
            return Ok(None);
        }

        let runs = resolve_gate_runs(args.runs, mode)?;
        if args.common.ty_bin.is_none() {
            if let Some(ty_bin) = legacy_env_ty_bin {
                args.common.ty_bin = Some(ty_bin);
            }
        }
        let gate_mode = match args.gate_mode {
            Some(mode) => Some(mode),
            None => match env::var(ENV_GATE_MODE) {
                Ok(value) if !value.is_empty() => Some(parse_gate_mode_env(&value)?),
                _ => None,
            },
        };

        Ok(Some(Self {
            common: args.common,
            mode,
            gate_mode,
            runs,
            summary_json: args.summary_json,
        }))
    }
}

#[derive(Debug)]
struct PreparedSupremacy {
    command: &'static str,
    policy_path: PathBuf,
    output_dir: PathBuf,
    specs: Vec<String>,
    trust_cg_env_overrides: BTreeMap<String, String>,
    interp_env_overrides: BTreeMap<String, String>,
    format: SupremacyOutputFormat,
    timeout_seconds: u64,
    ty_bin: Option<PathBuf>,
    target_dir: Option<PathBuf>,
    cargo_profile: String,
    ty_flags: Vec<String>,
    runs: Option<usize>,
    policy: SupremacyPolicy,
    gate_plan: Option<PlannedGate>,
}

impl PreparedSupremacy {
    fn prepare(
        command: &'static str,
        common: &SupremacyCommonArgs,
        runs: Option<usize>,
        gate_mode: Option<SupremacyGateMode>,
        run_mode: Option<SupremacyMode>,
    ) -> Result<Self> {
        let policy_path = common.policy.clone().unwrap_or_else(default_policy_path);
        let policy = SupremacyPolicy::load(&policy_path)?;
        if run_mode.is_some() {
            policy.validate_gate_ready()?;
        }
        let specs = if common.specs.is_empty() {
            if policy.specs.is_empty() {
                default_specs()
            } else {
                policy.specs.clone()
            }
        } else {
            common.specs.clone()
        };
        let trust_cg_env_overrides = parse_env_overrides("--trust_cg-env", &common.trust_cg_env)?;
        let interp_env_overrides = parse_env_overrides("--interp-env", &common.interp_env)?;
        let gate_plan = if let Some(run_mode) = run_mode {
            Some(PlannedGate::from_resolved(
                policy.resolve_gate_mode(run_mode, gate_mode)?,
            ))
        } else {
            None
        };
        let output_dir = common
            .output_dir
            .clone()
            .unwrap_or_else(|| default_output_dir(command));
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("create output dir {}", output_dir.display()))?;
        Ok(Self {
            command,
            policy_path,
            output_dir,
            specs,
            trust_cg_env_overrides,
            interp_env_overrides,
            format: common.format,
            timeout_seconds: common.timeout,
            ty_bin: common.ty_bin.clone(),
            target_dir: common.target_dir.clone(),
            cargo_profile: common.cargo_profile.clone(),
            ty_flags: common.ty_flag.clone(),
            runs,
            policy,
            gate_plan,
        })
    }

    fn validate_enforce_env_overrides(&self) -> Result<()> {
        let Some(gate_plan) = &self.gate_plan else {
            return Ok(());
        };
        if !self.interp_env_overrides.is_empty() {
            bail!(
                "enforce/{} does not allow --interp-env overrides",
                gate_plan.gate_mode
            );
        }
        if !self.ty_flags.is_empty() {
            bail!(
                "enforce/{} does not allow --ty-flag overrides",
                gate_plan.gate_mode
            );
        }
        let required = gate_plan.enforce_required_env();
        for (key, value) in &self.trust_cg_env_overrides {
            let Some(expected) = required.get(key) else {
                bail!(
                    "enforce/{} does not allow extra --trust_cg-env override {}",
                    gate_plan.gate_mode,
                    key
                );
            };
            if value != expected {
                bail!(
                    "enforce/{} requires {}={}; --trust_cg-env supplied {}={}",
                    gate_plan.gate_mode,
                    key,
                    expected,
                    key,
                    value
                );
            }
        }
        Ok(())
    }
}

fn default_policy_path() -> PathBuf {
    let repo_relative = PathBuf::from(DEFAULT_POLICY_FILE);
    if repo_relative.exists() {
        return repo_relative;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(DEFAULT_POLICY_FILE)
}

fn default_policy_path_near(anchor: &Path) -> PathBuf {
    let anchor_dir = if anchor.is_dir() {
        anchor
    } else {
        anchor.parent().unwrap_or_else(|| Path::new("."))
    };
    for ancestor in anchor_dir.ancestors() {
        let candidate = ancestor.join(DEFAULT_POLICY_FILE);
        if candidate.exists() {
            return candidate;
        }
    }
    default_policy_path()
}

fn default_output_dir(command: &str) -> PathBuf {
    Path::new("reports").join("perf").join(format!(
        "{}-supremacy-{}",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        command
    ))
}

fn resolve_gate_runs(cli_runs: Option<usize>, mode: SupremacyMode) -> Result<usize> {
    let (runs, source) = if let Some(runs) = cli_runs {
        (runs, "--runs")
    } else if let Ok(value) = env::var(ENV_RUNS) {
        (
            value
                .parse::<usize>()
                .with_context(|| format!("{ENV_RUNS} must be a non-negative integer"))?,
            ENV_RUNS,
        )
    } else {
        (3, "default")
    };
    if runs == 0 {
        bail!("{source} must be >= 1");
    }
    if mode == SupremacyMode::Enforce && runs < 3 {
        if source == ENV_RUNS {
            bail!("{ENV_RUNS} must be at least 3 in enforce mode");
        }
        bail!("--runs must be at least 3 in enforce mode");
    }
    Ok(runs)
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    let value = env::var_os(name)?;
    if value.is_empty() {
        return None;
    }
    Some(PathBuf::from(value))
}

fn parse_gate_mode_env(value: &str) -> Result<SupremacyGateMode> {
    match value {
        "full-native-fused" | "full_native_fused" => Ok(SupremacyGateMode::FullNativeFused),
        "interim-action-only-native-fused" | "interim_action_only_native_fused" => {
            Ok(SupremacyGateMode::InterimActionOnlyNativeFused)
        }
        _ => bail!(
            "{ENV_GATE_MODE} must be one of full-native-fused, full_native_fused, interim-action-only-native-fused, interim_action_only_native_fused"
        ),
    }
}

fn default_specs() -> Vec<String> {
    DEFAULT_SPECS
        .iter()
        .map(|spec| (*spec).to_string())
        .collect()
}

fn parse_env_overrides(flag_name: &str, items: &[String]) -> Result<BTreeMap<String, String>> {
    let mut env = BTreeMap::new();
    for item in items {
        let Some((key, value)) = item.split_once('=') else {
            bail!("{flag_name} expects KEY=VALUE, got {item:?}");
        };
        if key.is_empty() {
            bail!("{flag_name} key must not be empty");
        }
        env.insert(key.to_string(), value.to_string());
    }
    Ok(env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};

    fn repo_policy_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/tlc_comparison/single_thread_supremacy_gate.json")
    }

    fn gate_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_gate_env() {
        for key in [
            ENV_GATE_MODE,
            ENV_POLICY_FILE,
            ENV_RUNS,
            ENV_SKIP,
            ENV_TY_BIN,
        ] {
            // These tests serialize access to the process environment with
            // gate_env_lock; the raw mutation routes through the crate's single
            // env choke point.
            crate::env_guard::remove_var(key);
        }
    }

    fn set_gate_env(key: &str, value: &str) {
        // These tests serialize access to the process environment with
        // gate_env_lock; the raw mutation routes through the crate's single
        // env choke point.
        crate::env_guard::set_var(key, value);
    }

    fn common_for_test(output_dir: Option<PathBuf>) -> SupremacyCommonArgs {
        SupremacyCommonArgs {
            policy: Some(repo_policy_path()),
            output_dir,
            ty_bin: None,
            target_dir: None,
            cargo_profile: "release".to_string(),
            ty_flag: Vec::new(),
            timeout: 300,
            specs: Vec::new(),
            interp_env: Vec::new(),
            trust_cg_env: Vec::new(),
            format: SupremacyOutputFormat::Human,
        }
    }

    fn gate_args_for_test(output_dir: Option<PathBuf>) -> SupremacyGateArgs {
        SupremacyGateArgs {
            common: common_for_test(output_dir),
            mode: None,
            gate_mode: None,
            runs: None,
            summary_json: None,
        }
    }

    fn anti_overfit_policy_for_test() -> SupremacyPolicy {
        SupremacyPolicy {
            specs: vec!["LaunchSpec".to_string()],
            engine_selection_contract: None,
            matrix_policy: policy::MatrixPolicy::default(),
            expected_state_counts: BTreeMap::from([("LaunchSpec".to_string(), 100_000)]),
            expected_generated_state_counts: BTreeMap::from([("LaunchSpec".to_string(), 200_000)]),
            required_trust_cg_gate_flags: Vec::new(),
            default_gate_mode: None,
            final_gate_mode: None,
            gate_modes: BTreeMap::new(),
            thresholds: BTreeMap::new(),
        }
    }

    fn anti_overfit_gate_fixture(source: &str) -> (tempfile::TempDir, PreparedSupremacy, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::create_dir_all(dir.path().join("crates/tla-check/src")).unwrap();
        fs::create_dir_all(dir.path().join("crates/tla-trust-cg/src")).unwrap();
        fs::create_dir_all(dir.path().join("tests/tlc_comparison")).unwrap();
        fs::write(dir.path().join("crates/tla-check/src/runtime.rs"), source).unwrap();

        let policy = anti_overfit_policy_for_test();
        let policy_path = dir
            .path()
            .join("tests/tlc_comparison/single_thread_supremacy_gate.json");
        fs::write(&policy_path, serde_json::to_string(&policy).unwrap()).unwrap();
        let baseline_path = dir.path().join("tests/tlc_comparison/spec_baseline.json");
        fs::write(&baseline_path, r#"{"rows":[{"spec":"BaselineOnly"}]}"#).unwrap();

        let prepared = PreparedSupremacy {
            command: "gate",
            policy_path,
            output_dir: dir.path().join("reports"),
            specs: policy.specs.clone(),
            trust_cg_env_overrides: BTreeMap::new(),
            interp_env_overrides: BTreeMap::new(),
            format: SupremacyOutputFormat::Human,
            timeout_seconds: 300,
            ty_bin: None,
            target_dir: None,
            cargo_profile: "release".to_string(),
            ty_flags: Vec::new(),
            runs: Some(3),
            policy,
            gate_plan: None,
        };

        (dir, prepared, baseline_path)
    }

    #[test]
    fn parses_env_overrides() {
        let parsed = parse_env_overrides(
            "--trust_cg-env",
            &["TY_trust_cg=1".to_string(), "A=".to_string()],
        )
        .unwrap();

        assert_eq!(parsed.get("TY_trust_cg").map(String::as_str), Some("1"));
        assert_eq!(parsed.get("A").map(String::as_str), Some(""));
    }

    #[test]
    fn rejects_malformed_env_override() {
        let error = parse_env_overrides("--interp-env", &["TY_trust_cg".to_string()]).unwrap_err();

        assert!(error.to_string().contains("KEY=VALUE"));
        assert!(error.to_string().contains("--interp-env"));
    }

    #[test]
    fn enforce_gate_blocks_on_anti_overfit_findings() {
        let (_dir, prepared, baseline_path) =
            anti_overfit_gate_fixture(r#"const BAD: &str = "LaunchSpec";"#);

        let error =
            run_gate_anti_overfit_scan(&prepared, SupremacyMode::Enforce, Some(&baseline_path))
                .unwrap_err();

        assert!(error
            .to_string()
            .contains("ty supremacy gate anti-overfit scan found 1 forbidden corpus references"));
    }

    #[test]
    fn warn_gate_keeps_anti_overfit_findings_non_blocking() {
        let (_dir, prepared, baseline_path) =
            anti_overfit_gate_fixture(r#"const BAD: &str = "LaunchSpec";"#);

        run_gate_anti_overfit_scan(&prepared, SupremacyMode::Warn, Some(&baseline_path)).unwrap();
    }

    #[test]
    fn warn_gate_keeps_anti_overfit_scan_errors_non_blocking() {
        let (_dir, prepared, baseline_path) =
            anti_overfit_gate_fixture(r#"const SAFE: &str = "structural";"#);
        fs::remove_file(&baseline_path).unwrap();

        run_gate_anti_overfit_scan(&prepared, SupremacyMode::Warn, Some(&baseline_path)).unwrap();
    }

    #[test]
    fn matrix_output_summary_prefers_refreshed_matrix_when_available() {
        let initial = matrix::classify_baseline_str(
            r#"{
              "specs": {
                "NeedsRuntime": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "states": 1, "error_type": null},
                  "ty": {"status": "pass", "states": 1, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();
        let refreshed = matrix::classify_baseline_str(
            r#"{
              "specs": {
                "NeedsRuntime": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 1, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 1, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            matrix_output_summary(&initial, None).counts.missing_runtime,
            1
        );
        let output = matrix_output_summary(&initial, Some(&refreshed));
        assert_eq!(output.counts.missing_runtime, 0);
        assert_eq!(output.counts.pass, 1);
    }

    #[test]
    fn reads_specs_from_policy() {
        let policy = SupremacyPolicy::load(&repo_policy_path()).unwrap();

        assert_eq!(
            policy.specs,
            vec![
                "CoffeeCan1000BeansSafety".to_string(),
                "EWD998Small".to_string(),
                "MCLamportMutex".to_string()
            ]
        );
    }

    #[test]
    fn enforce_rejects_protected_env_override() {
        let mut common = common_for_test(None);
        common
            .trust_cg_env
            .push("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS=1".to_string());
        let prepared = PreparedSupremacy::prepare(
            "gate",
            &common,
            Some(3),
            Some(SupremacyGateMode::FullNativeFused),
            Some(SupremacyMode::Enforce),
        )
        .unwrap();

        let error = prepared.validate_enforce_env_overrides().unwrap_err();
        assert!(error
            .to_string()
            .contains("requires TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS=27"));
    }

    #[test]
    fn enforce_rejects_extra_env_overrides_outside_policy_controls() {
        let mut common = common_for_test(None);
        common
            .trust_cg_env
            .push("TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string());
        let prepared = PreparedSupremacy::prepare(
            "gate",
            &common,
            Some(3),
            Some(SupremacyGateMode::FullNativeFused),
            Some(SupremacyMode::Enforce),
        )
        .unwrap();

        let error = prepared.validate_enforce_env_overrides().unwrap_err();
        assert!(error.to_string().contains("extra --trust_cg-env override"));

        let mut common = common_for_test(None);
        common.interp_env.push("TY_BYTECODE_VM=0".to_string());
        let prepared = PreparedSupremacy::prepare(
            "gate",
            &common,
            Some(3),
            Some(SupremacyGateMode::FullNativeFused),
            Some(SupremacyMode::Enforce),
        )
        .unwrap();

        let error = prepared.validate_enforce_env_overrides().unwrap_err();
        assert!(error.to_string().contains("--interp-env"));
    }

    #[test]
    fn enforce_rejects_ty_only_flags_for_summary_json_too() {
        let mut common = common_for_test(None);
        common.ty_flag.push("--max-depth".to_string());
        let prepared = PreparedSupremacy::prepare(
            "gate",
            &common,
            Some(3),
            Some(SupremacyGateMode::FullNativeFused),
            Some(SupremacyMode::Enforce),
        )
        .unwrap();

        let error = prepared.validate_enforce_env_overrides().unwrap_err();
        assert!(error.to_string().contains("--ty-flag"));
    }

    #[test]
    fn gate_defaults_to_enforce_three_runs() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        let args = gate_args_for_test(None);

        let resolved = ResolvedGateArgs::resolve(args).unwrap().unwrap();

        assert_eq!(resolved.mode, SupremacyMode::Enforce);
        assert_eq!(resolved.runs, 3);
        assert_eq!(resolved.gate_mode, None);
        clear_gate_env();
    }

    #[test]
    fn gate_uses_legacy_warn_env_when_cli_absent() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        let policy_path = repo_policy_path();
        set_gate_env(ENV_POLICY_FILE, policy_path.to_str().unwrap());
        set_gate_env(ENV_RUNS, "5");
        set_gate_env(ENV_GATE_MODE, "interim_action_only_native_fused");
        set_gate_env(ENV_TY_BIN, "/tmp/ty-from-env");
        let mut args = gate_args_for_test(None);
        args.common.policy = None;
        args.mode = Some(SupremacyMode::Warn);

        let resolved = ResolvedGateArgs::resolve(args).unwrap().unwrap();

        assert_eq!(resolved.mode, SupremacyMode::Warn);
        assert_eq!(resolved.runs, 5);
        assert_eq!(
            resolved.gate_mode,
            Some(SupremacyGateMode::InterimActionOnlyNativeFused)
        );
        assert_eq!(resolved.common.policy, Some(policy_path));
        assert_eq!(
            resolved.common.ty_bin.as_deref(),
            Some(Path::new("/tmp/ty-from-env"))
        );
        clear_gate_env();
    }

    #[test]
    fn gate_cli_values_override_legacy_env_defaults() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        set_gate_env(ENV_RUNS, "5");
        set_gate_env(ENV_GATE_MODE, "interim_action_only_native_fused");
        set_gate_env(ENV_TY_BIN, "/tmp/ty-from-env");
        let mut args = gate_args_for_test(None);
        args.mode = Some(SupremacyMode::Warn);
        args.runs = Some(4);
        args.gate_mode = Some(SupremacyGateMode::FullNativeFused);
        args.common.ty_bin = Some(PathBuf::from("/tmp/ty-from-cli"));

        let resolved = ResolvedGateArgs::resolve(args).unwrap().unwrap();

        assert_eq!(resolved.runs, 4);
        assert_eq!(resolved.gate_mode, Some(SupremacyGateMode::FullNativeFused));
        assert_eq!(
            resolved.common.ty_bin.as_deref(),
            Some(Path::new("/tmp/ty-from-cli"))
        );
        clear_gate_env();
    }

    #[test]
    fn gate_skip_is_warn_only() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        set_gate_env(ENV_SKIP, "1");
        let mut warn_args = gate_args_for_test(None);
        warn_args.mode = Some(SupremacyMode::Warn);
        assert!(ResolvedGateArgs::resolve(warn_args).unwrap().is_none());

        let enforce_error = ResolvedGateArgs::resolve(gate_args_for_test(None)).unwrap_err();
        assert!(enforce_error.to_string().contains(ENV_SKIP));
        clear_gate_env();
    }

    #[test]
    fn gate_policy_env_is_warn_only() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        set_gate_env(ENV_POLICY_FILE, "/tmp/policy.json");

        let enforce_error = ResolvedGateArgs::resolve(gate_args_for_test(None)).unwrap_err();
        assert!(enforce_error.to_string().contains(ENV_POLICY_FILE));
        clear_gate_env();
    }

    #[test]
    fn gate_enforce_collection_rejects_explicit_ty_bin() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        let mut args = gate_args_for_test(None);
        args.common.ty_bin = Some(PathBuf::from("/tmp/stale-ty"));

        let enforce_error = ResolvedGateArgs::resolve(args).unwrap_err();
        assert!(enforce_error.to_string().contains("--ty-bin"));
        clear_gate_env();
    }

    #[test]
    fn gate_enforce_collection_rejects_ty_only_flags_and_interp_env() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        let mut args = gate_args_for_test(None);
        args.common.ty_flag = vec!["--max-depth".to_string(), "5".to_string()];

        let enforce_error = ResolvedGateArgs::resolve(args).unwrap_err();
        assert!(enforce_error.to_string().contains("--ty-flag"));

        let mut args = gate_args_for_test(None);
        args.common.interp_env = vec!["TY_BYTECODE_VM=0".to_string()];

        let enforce_error = ResolvedGateArgs::resolve(args).unwrap_err();
        assert!(enforce_error.to_string().contains("--interp-env"));
        clear_gate_env();
    }

    #[test]
    fn gate_enforce_collection_rejects_legacy_ty_bin_env() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        set_gate_env(ENV_TY_BIN, "/tmp/stale-ty");

        let enforce_error = ResolvedGateArgs::resolve(gate_args_for_test(None)).unwrap_err();
        assert!(enforce_error.to_string().contains(ENV_TY_BIN));
        clear_gate_env();
    }

    #[test]
    fn gate_enforce_summary_recheck_allows_existing_summary_and_ty_bin_env() {
        let _guard = gate_env_lock().lock().unwrap();
        clear_gate_env();
        set_gate_env(ENV_TY_BIN, "/tmp/ty-from-env");
        let mut args = gate_args_for_test(None);
        args.summary_json = Some(PathBuf::from("/tmp/summary.json"));

        let resolved = ResolvedGateArgs::resolve(args).unwrap().unwrap();

        assert_eq!(
            resolved.common.ty_bin.as_deref(),
            Some(Path::new("/tmp/ty-from-env"))
        );
        clear_gate_env();
    }

    #[test]
    fn warn_gate_requires_explicit_gate_mode_when_policy_has_modes() {
        let dir = tempfile::tempdir().unwrap();
        let error = PreparedSupremacy::prepare(
            "gate",
            &common_for_test(Some(dir.path().join("out"))),
            Some(1),
            None,
            Some(SupremacyMode::Warn),
        )
        .unwrap_err();

        assert!(error.to_string().contains("--gate-mode is required"));
    }

    #[test]
    fn prepare_defaults_policy_path() {
        let dir = tempfile::tempdir().unwrap();
        let mut common = common_for_test(Some(dir.path().join("out")));
        common.policy = None;

        let prepared =
            PreparedSupremacy::prepare("benchmark", &common, Some(1), None, None).unwrap();

        assert!(prepared.policy_path.ends_with(DEFAULT_POLICY_FILE));
    }
}
