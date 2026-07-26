// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Runner for `ty supremacy benchmark`.

use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::parse;
use super::runner::{create_fresh_artifact_dir, run_command, CommandResult, CommandSpec};
use super::summary::{
    BackendControls, BenchmarkBuildIdentity, BenchmarkGateFlags, BenchmarkRow, BenchmarkSummary,
    LaunchControls, TlcLaunchControls, TlcRunResult, TrustCgTelemetry, TyLaunchControls,
    TyModeLaunchControls, TyRunResult,
};
use super::{tlc_java_single_thread_args, tlc_java_single_thread_base_argv, PreparedSupremacy};
use crate::cli_schema::SupremacyOutputFormat;

const COFFEECAN_SAFETY_SPEC_NAME: &str = "CoffeeCan1000BeansSafety";
const COFFEECAN_SAFETY_BEANS: usize = 1000;
const DEFAULT_SPEC_BASELINE: &str = "tests/tlc_comparison/spec_baseline.json";
const DEFAULT_SPEC_TIMEOUT_SECONDS: u64 = 300;
const DEFAULT_TLC_JAR: &str = "tlaplus/tytools.jar";
const ENV_TYTOOLS_JAR: &str = "TYTOOLS_JAR";
const ENV_TLC_JAR: &str = "TLC_JAR";
const TLC_BENCHMARK_WORKERS: usize = 1;
const TY_BENCHMARK_WORKERS: usize = 1;
const TY_CACHE_DIR_ENV: &str = "TY_CACHE_DIR";
const TY_DISABLE_ARTIFACT_CACHE_ENV: &str = "TY_DISABLE_ARTIFACT_CACHE";
const TY_NATIVE_CALLOUT_COMPILE_JOBS_ENV: &str = "TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS";
// Auto-POR/auto-symmetry are NOT pinned here: those semantic levers are
// controlled by CLI flags only (the child `ty check` ignores ambient
// TY_AUTO_POR / TY_AUTO_SYMMETRY). The count-parity `--no-reduction` flag is
// injected into every planned `ty check` argv by `plan_ty_run` instead.
const GATE_BENCHMARK_TRUST_CG_ENV: &[(&str, &str)] = &[
    ("TY_trust_cg", "1"),
    ("TY_TRUST_CG_BFS", "1"),
    ("TY_TRUST_CG_EXISTS", "1"),
    ("TY_BYTECODE_VM", "1"),
    ("TY_BYTECODE_VM_STATS", "1"),
];

pub(super) struct BenchmarkRunReport {
    pub(super) summary_path: PathBuf,
    pub(super) passed: bool,
}

pub(super) fn run_benchmark(prepared: &PreparedSupremacy) -> Result<BenchmarkRunReport> {
    let repo_root = env::current_dir().context("resolve current working directory")?;
    let examples_dir = tlaplus_examples_dir()?;
    let ty_binary = resolve_ty_binary(prepared, &repo_root)?;
    let tlc_jar = default_tlc_jar()?;
    validate_binary(&tlc_jar)?;
    let run_count = prepared.runs.unwrap_or(1);
    let trust_cg_overrides = trust_cg_env_overrides(prepared)?;

    let mut rows = Vec::new();
    for spec_name in &prepared.specs {
        let spec =
            resolve_benchmark_spec(spec_name, &prepared.output_dir, &examples_dir, &repo_root)?;
        let spec_artifact_dir = prepared.output_dir.join(&spec.name);
        let timeout_seconds = prepared.timeout_seconds.max(spec.timeout_seconds);
        let mut tlc_runs = Vec::new();
        let mut interp_runs = Vec::new();
        let mut trust_cg_runs = Vec::new();
        for run_index in 1..=run_count {
            print_run_start(prepared, &spec.name, "tlc", run_index, run_count);
            let tlc_run = plan_tlc_run(&spec, run_index, &repo_root, &tlc_jar, &spec_artifact_dir)?;
            tlc_runs.push(execute_tlc_run(
                &spec,
                tlc_run,
                timeout_seconds,
                &repo_root,
            )?);

            print_run_start(prepared, &spec.name, "interp", run_index, run_count);
            let interp_run = plan_ty_run(
                &spec,
                "interp",
                run_index,
                &repo_root,
                &ty_binary,
                &prepared.interp_env_overrides,
                &prepared.ty_flags,
                &spec_artifact_dir,
            )?;
            interp_runs.push(execute_ty_run(
                &spec,
                interp_run,
                timeout_seconds,
                &repo_root,
            )?);

            print_run_start(prepared, &spec.name, "trust-cg", run_index, run_count);
            let trust_cg_run = plan_ty_run(
                &spec,
                "trust-cg",
                run_index,
                &repo_root,
                &ty_binary,
                &trust_cg_overrides,
                &prepared.ty_flags,
                &spec_artifact_dir,
            )?;
            trust_cg_runs.push(execute_ty_run(
                &spec,
                trust_cg_run,
                timeout_seconds,
                &repo_root,
            )?);
        }
        rows.push(BenchmarkRow::from_runs(
            spec.name.clone(),
            spec.expected_states,
            tlc_runs,
            interp_runs,
            trust_cg_runs,
        ));
    }

    let gate_flags = benchmark_gate_flags(prepared);
    let backend_controls = BackendControls {
        interp_env: prepared.interp_env_overrides.clone(),
        trust_cg_env: trust_cg_overrides.clone(),
    };
    let launch_controls = benchmark_launch_controls(&backend_controls);
    let summary = BenchmarkSummary::new(
        chrono::Local::now().format("%Y-%m-%dT%H%M%S").to_string(),
        git_commit_short(&repo_root),
        repo_relative(&repo_root, &prepared.output_dir),
        invocation(prepared),
        BenchmarkBuildIdentity::new(
            prepared.cargo_profile.clone(),
            repo_relative(&repo_root, &ty_binary),
            sha256_file(&ty_binary)
                .with_context(|| format!("hash ty binary {}", ty_binary.display()))?,
        ),
        backend_controls,
        launch_controls,
        gate_flags,
        rows,
    );
    let passed = summary_passed(&summary);
    let summary_json =
        serde_json::to_string_pretty(&summary).context("serialize benchmark summary")?;
    let summary_path = prepared.output_dir.join("summary.json");
    fs::write(&summary_path, summary_json + "\n")
        .with_context(|| format!("write {}", summary_path.display()))?;
    let markdown = render_markdown(&summary);
    fs::write(prepared.output_dir.join("summary.md"), &markdown)
        .with_context(|| format!("write {}", prepared.output_dir.join("summary.md").display()))?;

    match prepared.format {
        SupremacyOutputFormat::Human | SupremacyOutputFormat::Markdown => {
            let status = if passed { "PASS" } else { "FAIL" };
            eprintln!(
                "[supremacy] benchmark {status}: wrote {}",
                summary_path.display()
            );
        }
        SupremacyOutputFormat::Json => {
            println!(
                "{}",
                json!({
                    "status": if passed { "pass" } else { "fail" },
                    "summary": summary_path,
                })
            );
        }
    }

    Ok(BenchmarkRunReport {
        summary_path,
        passed,
    })
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkSpec {
    name: String,
    tla_path: PathBuf,
    cfg_path: PathBuf,
    category: String,
    expected_states: Option<u64>,
    /// Expected pre-constraint successor count (excludes initial states).
    expected_raw_successors_generated: Option<u64>,
    /// Expected TLC-style total generated count (raw Init + raw successors).
    expected_states_generated: Option<u64>,
    timeout_seconds: u64,
    notes: String,
}

#[derive(Clone, Debug, Serialize)]
struct PlannedRun {
    spec: String,
    mode: String,
    run_index: usize,
    artifact_dir: PathBuf,
    command: PlannedCommand,
}

#[derive(Clone, Debug, Serialize)]
struct PlannedCommand {
    argv: Vec<String>,
    cwd: PathBuf,
    env_overrides: BTreeMap<String, String>,
}

fn resolve_benchmark_spec(
    spec_name: &str,
    output_dir: &Path,
    examples_dir: &Path,
    repo_root: &Path,
) -> Result<BenchmarkSpec> {
    match spec_name {
        COFFEECAN_SAFETY_SPEC_NAME => stage_coffecan_safety_spec(output_dir, examples_dir),
        "EWD998Small" => catalog_spec(
            "EWD998Small",
            "ewd998/EWD998.tla",
            "ewd998/EWD998Small.cfg",
            "ewd998",
            Some(1_520_618),
            Some(9_630_813),
            examples_dir,
        ),
        "MCLamportMutex" => catalog_spec(
            "MCLamportMutex",
            "lamport_mutex/MCLamportMutex.tla",
            "lamport_mutex/MCLamportMutex.cfg",
            "lamport_mutex",
            Some(724_274),
            Some(2_496_350),
            examples_dir,
        ),
        other => baseline_catalog_spec(other, examples_dir, &repo_root.join(DEFAULT_SPEC_BASELINE)),
    }
}

fn catalog_spec(
    name: &str,
    tla_path: &str,
    cfg_path: &str,
    category: &str,
    expected_states: Option<u64>,
    expected_raw_successors_generated: Option<u64>,
    examples_dir: &Path,
) -> Result<BenchmarkSpec> {
    let tla_path = examples_dir.join(tla_path);
    let cfg_path = examples_dir.join(cfg_path);
    validate_spec_files(&tla_path, &cfg_path)?;
    Ok(BenchmarkSpec {
        name: name.to_string(),
        tla_path,
        cfg_path,
        category: category.to_string(),
        expected_states,
        expected_raw_successors_generated,
        expected_states_generated: None,
        timeout_seconds: DEFAULT_SPEC_TIMEOUT_SECONDS,
        notes: String::new(),
    })
}

fn stage_coffecan_safety_spec(output_dir: &Path, examples_dir: &Path) -> Result<BenchmarkSpec> {
    let spec_dir = output_dir
        .join("generated-specs")
        .join("CoffeeCanSafety1000");
    fs::create_dir_all(&spec_dir).with_context(|| format!("create {}", spec_dir.display()))?;
    let source = examples_dir.join("CoffeeCan/CoffeeCan.tla");
    if !source.is_file() {
        bail!("CoffeeCan source not found: {}", source.display());
    }
    fs::copy(&source, spec_dir.join("CoffeeCan.tla"))
        .with_context(|| format!("copy {}", source.display()))?;

    let wrapper_tla = spec_dir.join("CoffeeCanSafetyBench.tla");
    let wrapper_cfg = spec_dir.join("CoffeeCanSafetyBench.cfg");
    fs::write(
        &wrapper_tla,
        format!(
            "---- MODULE CoffeeCanSafetyBench ----\n\
             EXTENDS Naturals\n\n\
             VARIABLE can\n\n\
             Can == [black : 0..{COFFEECAN_SAFETY_BEANS}, white : 0..{COFFEECAN_SAFETY_BEANS}]\n\n\
             TypeInvariant == can \\in Can\n\n\
             SafetyInit == can \\in {{c \\in Can : c.black + c.white = {COFFEECAN_SAFETY_BEANS}}}\n\n\
             BeanCount == can.black + can.white\n\n\
             PickSameColorBlack ==\n\
             \t/\\ BeanCount > 1\n\
             \t/\\ can.black >= 2\n\
             \t/\\ can' = [can EXCEPT !.black = @ - 1]\n\n\
             PickSameColorWhite ==\n\
             \t/\\ BeanCount > 1\n\
             \t/\\ can.white >= 2\n\
             \t/\\ can' = [can EXCEPT !.black = @ + 1, !.white = @ - 2]\n\n\
             PickDifferentColor ==\n\
             \t/\\ BeanCount > 1\n\
             \t/\\ can.black >= 1\n\
             \t/\\ can.white >= 1\n\
             \t/\\ can' = [can EXCEPT !.black = @ - 1]\n\n\
             Termination ==\n\
             \t/\\ BeanCount = 1\n\
             \t/\\ UNCHANGED can\n\n\
             Next ==\n\
             \t\\/ PickSameColorWhite\n\
             \t\\/ PickSameColorBlack\n\
             \t\\/ PickDifferentColor\n\
             \t\\/ Termination\n\n\
             ====\n"
        ),
    )
    .with_context(|| format!("write {}", wrapper_tla.display()))?;
    fs::write(
        &wrapper_cfg,
        "INIT\n    SafetyInit\n\nNEXT\n    Next\n\nINVARIANTS\n    TypeInvariant\n",
    )
    .with_context(|| format!("write {}", wrapper_cfg.display()))?;

    Ok(BenchmarkSpec {
        name: COFFEECAN_SAFETY_SPEC_NAME.to_string(),
        tla_path: wrapper_tla,
        cfg_path: wrapper_cfg,
        category: "CoffeeCan".to_string(),
        expected_states: Some(501_500),
        expected_raw_successors_generated: Some(1_498_502),
        expected_states_generated: Some(1_499_503),
        timeout_seconds: DEFAULT_SPEC_TIMEOUT_SECONDS,
        notes: "Generated safety-only CoffeeCan1000 model: exact-1000-bean initial frontier, Next, TypeInvariant, no temporal properties or fairness.".to_string(),
    })
}

#[derive(Debug, Deserialize)]
struct SpecBaseline {
    specs: BTreeMap<String, SpecBaselineEntry>,
}

#[derive(Debug, Deserialize)]
struct SpecBaselineEntry {
    category: String,
    source: Option<SpecBaselineSource>,
    tlc: SpecBaselineTlc,
    #[serde(default)]
    ty_expected_states: Option<u64>,
    #[serde(default)]
    diagnose_timeout_seconds: Option<u64>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpecBaselineSource {
    tla_path: String,
    cfg_path: String,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpecBaselineTlc {
    #[serde(default)]
    states: Option<u64>,
    #[serde(default)]
    states_generated: Option<u64>,
    #[serde(default)]
    generated_states: Option<u64>,
}

fn baseline_catalog_spec(
    name: &str,
    examples_dir: &Path,
    baseline_path: &Path,
) -> Result<BenchmarkSpec> {
    let baseline = load_spec_baseline(baseline_path)?;
    let entry = baseline.specs.get(name).with_context(|| {
        format!(
            "unsupported supremacy benchmark spec {name:?}; not found in {}",
            baseline_path.display()
        )
    })?;
    let source = entry.source.as_ref().with_context(|| {
        format!("unsupported supremacy benchmark spec {name:?}; baseline entry has no source paths")
    })?;
    let mode = source.mode.as_deref().unwrap_or("check");
    if mode != "check" {
        bail!(
            "unsupported supremacy benchmark spec {name:?}; baseline source mode is {mode:?}, but supremacy benchmark only supports check-mode specs"
        );
    }

    let tla_path = examples_dir.join(&source.tla_path);
    let cfg_path = examples_dir.join(&source.cfg_path);
    validate_spec_files(&tla_path, &cfg_path)?;
    Ok(BenchmarkSpec {
        name: name.to_string(),
        tla_path,
        cfg_path,
        category: entry.category.clone(),
        expected_states: entry.ty_expected_states.or(entry.tlc.states),
        expected_raw_successors_generated: None,
        expected_states_generated: entry.tlc.states_generated.or(entry.tlc.generated_states),
        timeout_seconds: entry
            .diagnose_timeout_seconds
            .unwrap_or(DEFAULT_SPEC_TIMEOUT_SECONDS),
        notes: entry
            .notes
            .clone()
            .unwrap_or_else(|| format!("Resolved from {}", DEFAULT_SPEC_BASELINE)),
    })
}

fn load_spec_baseline(path: &Path) -> Result<SpecBaseline> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read baseline {}", path.display()))?;
    serde_json::from_str(&text).context("parse spec_baseline.json")
}

fn validate_spec_files(tla_path: &Path, cfg_path: &Path) -> Result<()> {
    if !tla_path.is_file() {
        bail!("TLA file not found: {}", tla_path.display());
    }
    if !cfg_path.is_file() {
        bail!("config file not found: {}", cfg_path.display());
    }
    Ok(())
}

fn plan_tlc_run(
    spec: &BenchmarkSpec,
    run_index: usize,
    repo_root: &Path,
    tlc_jar: &Path,
    spec_artifact_dir: &Path,
) -> Result<PlannedRun> {
    let artifact_dir = absolute_path(
        repo_root,
        &spec_artifact_dir.join(format!("tlc-run{run_index}")),
    );
    let metadir = artifact_dir.join("tlc-metadir");
    let tlc_jar = absolute_path(repo_root, tlc_jar);
    let tla_path = absolute_path(repo_root, &spec.tla_path);
    let cfg_path = absolute_path(repo_root, &spec.cfg_path);
    let metadir = absolute_path(repo_root, &metadir);
    let mut argv = tlc_java_single_thread_base_argv();
    argv.extend([
        "-jar".to_string(),
        tlc_jar.display().to_string(),
        tla_path.display().to_string(),
        "-config".to_string(),
        cfg_path.display().to_string(),
        "-metadir".to_string(),
        metadir.display().to_string(),
        "-workers".to_string(),
        "1".to_string(),
    ]);
    let command = PlannedCommand {
        argv,
        cwd: tla_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        env_overrides: BTreeMap::new(),
    };
    write_planned_artifacts(repo_root, &artifact_dir, &command)?;
    Ok(PlannedRun {
        spec: spec.name.clone(),
        mode: "tlc".to_string(),
        run_index,
        artifact_dir,
        command,
    })
}

fn plan_ty_run(
    spec: &BenchmarkSpec,
    mode: &str,
    run_index: usize,
    repo_root: &Path,
    ty_binary: &Path,
    cli_overrides: &BTreeMap<String, String>,
    ty_flags: &[String],
    spec_artifact_dir: &Path,
) -> Result<PlannedRun> {
    let artifact_dir = absolute_path(
        repo_root,
        &spec_artifact_dir.join(format!("{mode}-run{run_index}")),
    );
    let env_overrides = ty_env(mode, cli_overrides)?;
    let mut argv = vec![
        ty_binary.display().to_string(),
        "check".to_string(),
        spec.tla_path.display().to_string(),
        "--config".to_string(),
        spec.cfg_path.display().to_string(),
        "--workers".to_string(),
        "1".to_string(),
        "--force".to_string(),
        // Count-parity lever: these gates compare exact distinct-state counts
        // against unreduced TLC baselines, so auto-POR and auto-symmetry (both
        // sound, default-on count reducers) are disabled via the CLI flag —
        // the child ignores ambient TY_AUTO_POR / TY_AUTO_SYMMETRY env.
        "--no-reduction".to_string(),
    ];
    argv.extend(ty_flags.iter().cloned());
    argv.extend(["--backend".to_string(), backend_name(mode)?.to_string()]);
    let command = PlannedCommand {
        argv,
        cwd: repo_root.to_path_buf(),
        env_overrides,
    };
    write_planned_artifacts(repo_root, &artifact_dir, &command)?;
    Ok(PlannedRun {
        spec: spec.name.clone(),
        mode: mode.to_string(),
        run_index,
        artifact_dir,
        command,
    })
}

fn backend_name(mode: &str) -> Result<&'static str> {
    match mode {
        "interp" => Ok("interpreter"),
        "trust-cg" => Ok("trust-cg"),
        other => bail!("unsupported benchmark mode {other:?}"),
    }
}

fn benchmark_launch_controls(backend_controls: &BackendControls) -> LaunchControls {
    let jvm_args = tlc_java_single_thread_args()
        .iter()
        .map(|arg| (*arg).to_string())
        .collect::<Vec<_>>();
    LaunchControls {
        tlc: TlcLaunchControls {
            workers: TLC_BENCHMARK_WORKERS,
            heap_xms: jvm_arg_value(&jvm_args, "-Xms"),
            heap_xmx: jvm_arg_value(&jvm_args, "-Xmx"),
            jvm_args,
        },
        ty: TyLaunchControls {
            interp: ty_mode_launch_controls(&backend_controls.interp_env),
            trust_cg: ty_mode_launch_controls(&backend_controls.trust_cg_env),
        },
    }
}

fn jvm_arg_value(args: &[String], prefix: &str) -> Option<String> {
    args.iter()
        .find_map(|arg| arg.strip_prefix(prefix).map(str::to_string))
}

fn ty_mode_launch_controls(env: &BTreeMap<String, String>) -> TyModeLaunchControls {
    TyModeLaunchControls {
        workers: TY_BENCHMARK_WORKERS,
        cache_dir: env.get(TY_CACHE_DIR_ENV).cloned(),
        artifact_cache_disabled_env: env.get(TY_DISABLE_ARTIFACT_CACHE_ENV).cloned(),
        native_callout_compile_jobs: env.get(TY_NATIVE_CALLOUT_COMPILE_JOBS_ENV).cloned(),
    }
}

fn ty_env(
    mode: &str,
    cli_overrides: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    // Count-parity (auto-POR/auto-symmetry off) is requested via the
    // `--no-reduction` CLI flag in `plan_ty_run`, not env: the child `ty check`
    // ignores ambient TY_AUTO_POR / TY_AUTO_SYMMETRY.
    let mut env = BTreeMap::from([("TY_BYTECODE_VM".to_string(), "1".to_string())]);
    match mode {
        "interp" => {
            env.insert("TY_trust_cg".to_string(), "0".to_string());
            env.insert("TY_TRUST_CG_BFS".to_string(), "0".to_string());
        }
        "trust-cg" => {
            env.insert("TY_trust_cg".to_string(), "1".to_string());
            env.insert("TY_TRUST_CG_BFS".to_string(), "1".to_string());
            env.insert("TY_TRUST_CG_EXISTS".to_string(), "1".to_string());
        }
        other => bail!("unsupported benchmark mode {other:?}"),
    }
    env.extend(cli_overrides.clone());
    Ok(env)
}

fn trust_cg_env_overrides(prepared: &PreparedSupremacy) -> Result<BTreeMap<String, String>> {
    let mut overrides = if prepared.gate_plan.is_some() {
        gate_benchmark_trust_cg_env()
    } else {
        BTreeMap::new()
    };
    if let Some(plan) = &prepared.gate_plan {
        overrides.extend(plan.enforce_required_env());
    }
    overrides.insert(
        "TY_CACHE_DIR".to_string(),
        prepared
            .output_dir
            .join("trust_cg-artifact-cache")
            .display()
            .to_string(),
    );
    overrides.extend(prepared.trust_cg_env_overrides.clone());
    Ok(overrides)
}

fn gate_benchmark_trust_cg_env() -> BTreeMap<String, String> {
    GATE_BENCHMARK_TRUST_CG_ENV
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

fn execute_tlc_run(
    spec: &BenchmarkSpec,
    run: PlannedRun,
    timeout_seconds: u64,
    repo_root: &Path,
) -> Result<TlcRunResult> {
    let result = execute_planned_run(&run, timeout_seconds)?;
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_tlc_final_counts(&stdout, &stderr);
    let error = run_error(
        result.returncode,
        &stderr,
        [
            (counts.states_found, "states_found"),
            (
                counts.raw_initial_states_generated,
                "raw_initial_states_generated",
            ),
            (counts.raw_successors_generated, "raw_successors_generated"),
            (counts.states_generated, "states_generated"),
        ],
    );
    Ok(TlcRunResult {
        tool: "tlc".to_string(),
        spec_name: spec.name.clone(),
        run_index: Some(run.run_index),
        workers: 1,
        elapsed_seconds: result.elapsed_seconds,
        peak_rss_bytes: result.peak_rss_bytes,
        states_found: counts.states_found,
        distinct_states: counts.distinct_states,
        transitions: counts.transitions,
        raw_initial_states_generated: counts.raw_initial_states_generated,
        raw_successors_generated: counts.raw_successors_generated,
        states_generated: counts.states_generated,
        returncode: result.returncode,
        error,
        artifact_dir: Some(repo_relative(repo_root, &result.artifact_dir)),
    })
}

fn execute_ty_run(
    spec: &BenchmarkSpec,
    run: PlannedRun,
    timeout_seconds: u64,
    repo_root: &Path,
) -> Result<TyRunResult> {
    let mode = run.mode.clone();
    let result = execute_planned_run(&run, timeout_seconds)?;
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_ty_final_counts(&stdout, &stderr);
    let error = run_error(
        result.returncode,
        &stderr,
        [
            (counts.states_found, "states_found"),
            (
                counts.raw_initial_states_generated,
                "raw_initial_states_generated",
            ),
            (counts.raw_successors_generated, "raw_successors_generated"),
            (counts.states_generated, "states_generated"),
        ],
    );
    let trust_cg_telemetry = if mode == "trust-cg" {
        let mut telemetry = summary_telemetry(parse::parse_trust_cg_telemetry(&stdout, &stderr));
        telemetry.transitions = counts.transitions;
        Some(telemetry)
    } else {
        None
    };
    Ok(TyRunResult {
        tool: "ty".to_string(),
        mode,
        spec_name: spec.name.clone(),
        run_index: run.run_index,
        elapsed_seconds: result.elapsed_seconds,
        peak_rss_bytes: result.peak_rss_bytes,
        states_found: counts.states_found,
        transitions: counts.transitions,
        raw_initial_states_generated: counts.raw_initial_states_generated,
        raw_successors_generated: counts.raw_successors_generated,
        states_generated: counts.states_generated,
        returncode: result.returncode,
        error,
        artifact_dir: Some(repo_relative(repo_root, &result.artifact_dir)),
        workers: 1,
        env_overrides: Some(result.env_overrides),
        trust_cg_telemetry,
    })
}

fn execute_planned_run(run: &PlannedRun, timeout_seconds: u64) -> Result<CommandResult> {
    run_command(CommandSpec {
        argv: run.command.argv.clone(),
        cwd: run.command.cwd.clone(),
        env_overrides: run.command.env_overrides.clone(),
        timeout_seconds,
        capture_limits: None,
        artifact_dir: run.artifact_dir.clone(),
        payload_dir: None,
        observation_storage_contract: None,
        observation_storage_binding: None,
        tlc_metadir: None,
    })
}

fn run_error<const N: usize>(
    returncode: i32,
    stderr: &str,
    required_fields: [(Option<u64>, &str); N],
) -> Option<String> {
    if returncode != 0 {
        let tail = stderr
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("command failed");
        return Some(tail.to_string());
    }
    let missing = required_fields
        .into_iter()
        .filter_map(|(value, field)| value.is_none().then_some(field))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        None
    } else {
        Some(format!("missing parsed {}", missing.join(", ")))
    }
}

fn summary_telemetry(value: parse::TrustCgTelemetry) -> TrustCgTelemetry {
    TrustCgTelemetry {
        trust_cg_actions_compiled: value.trust_cg_actions_compiled,
        trust_cg_actions_total: value.trust_cg_actions_total,
        trust_cg_invariants_compiled: value.trust_cg_invariants_compiled,
        trust_cg_invariants_total: value.trust_cg_invariants_total,
        trust_cg_state_constraints_compiled: value.trust_cg_state_constraints_compiled,
        trust_cg_state_constraints_total: value.trust_cg_state_constraints_total,
        compiled_bfs_step_active: value.compiled_bfs_step_active,
        compiled_bfs_level_active: value.compiled_bfs_level_active,
        compiled_bfs_level_loop_started: value.compiled_bfs_level_loop_started,
        compiled_bfs_level_loop_initial_states: value.compiled_bfs_level_loop_initial_states,
        compiled_bfs_level_loop_fused: value.compiled_bfs_level_loop_fused,
        compiled_bfs_levels_completed: value.compiled_bfs_levels_completed,
        compiled_bfs_parents_processed: value.compiled_bfs_parents_processed,
        compiled_bfs_successors_generated: value.compiled_bfs_successors_generated,
        compiled_bfs_successors_new: value.compiled_bfs_successors_new,
        compiled_bfs_total_states: value.compiled_bfs_total_states,
        compiled_bfs_zero_work: value.compiled_bfs_zero_work,
        compiled_bfs_execution_nanos: value.compiled_bfs_execution_nanos,
        compiled_bfs_execution_seconds: value.compiled_bfs_execution_seconds,
        trust_cg_bfs_level_active: value.trust_cg_bfs_level_active,
        trust_cg_native_fused_level_built: value.trust_cg_native_fused_level_built,
        trust_cg_native_fused_level_active: value.trust_cg_native_fused_level_active,
        trust_cg_native_fused_regular_invariants_checked: value
            .trust_cg_native_fused_regular_invariants_checked,
        trust_cg_native_fused_mode: value.trust_cg_native_fused_mode,
        trust_cg_native_fused_invariant_count: value.trust_cg_native_fused_invariant_count,
        trust_cg_native_fused_state_constraint_count: value
            .trust_cg_native_fused_state_constraint_count,
        trust_cg_native_fused_state_len: value.trust_cg_native_fused_state_len,
        trust_cg_native_fused_local_dedup: value.trust_cg_native_fused_local_dedup,
        trust_cg_native_bfs_trace_generated: value.trust_cg_native_bfs_trace_generated,
        trust_cg_native_bfs_trace_state_count: value.trust_cg_native_bfs_trace_state_count,
        trust_cg_native_bfs_trace_parents_processed: value
            .trust_cg_native_bfs_trace_parents_processed,
        trust_cg_bfs_level_loop_kind: value.trust_cg_bfs_level_loop_kind,
        trust_cg_native_fused_flat_frontier_admission_active: value
            .trust_cg_native_fused_flat_frontier_admission_active,
        compiled_bfs_flat_frontier_admitted: value.compiled_bfs_flat_frontier_admitted,
        flat_state_primary: value.flat_state_primary,
        flat_bfs_frontier_active: value.flat_bfs_frontier_active,
        flat_bfs_frontier_fallbacks: value.flat_bfs_frontier_fallbacks,
        native_action_callout_batch_artifact_identity_source: value
            .native_action_callout_batch_artifact_identity_source,
        native_action_callout_batch_artifact_identity: value
            .native_action_callout_batch_artifact_identity,
        native_action_callout_batch_artifact_cache_digest: value
            .native_action_callout_batch_artifact_cache_digest,
        native_action_callout_batch_cache_key: value.native_action_callout_batch_cache_key,
        native_action_callout_batch_artifact_cacheable: value
            .native_action_callout_batch_artifact_cacheable,
        native_action_callout_batch_artifact_cache_disabled_by_env: value
            .native_action_callout_batch_artifact_cache_disabled_by_env,
        native_action_callout_batch_shard_count: value.native_action_callout_batch_shard_count,
        native_action_callout_batch_warm_cache_enabled: value
            .native_action_callout_batch_warm_cache_enabled,
        native_action_callout_batch_warm_cache_lookup_attempted: value
            .native_action_callout_batch_warm_cache_lookup_attempted,
        native_action_callout_batch_warm_cache_hits: value
            .native_action_callout_batch_warm_cache_hits,
        native_action_callout_batch_warm_cache_misses: value
            .native_action_callout_batch_warm_cache_misses,
        native_action_callout_batch_warm_cache_stores: value
            .native_action_callout_batch_warm_cache_stores,
        native_action_callout_batch_setup_ms: value.native_action_callout_batch_setup_ms,
        native_action_callout_batch_lowering_ms: value.native_action_callout_batch_lowering_ms,
        native_action_callout_batch_assembly_ms: value.native_action_callout_batch_assembly_ms,
        native_action_callout_batch_compile_ms: value.native_action_callout_batch_compile_ms,
        native_action_callout_batch_warm_cache_lookup_ms: value
            .native_action_callout_batch_warm_cache_lookup_ms,
        native_action_callout_batch_artifact_materialization_ms: value
            .native_action_callout_batch_artifact_materialization_ms,
        native_action_callout_batch_fallback_per_action_compile_ms: value
            .native_action_callout_batch_fallback_per_action_compile_ms,
        native_action_callout_batch_shard_warm_cache_statuses: value
            .native_action_callout_batch_shard_warm_cache_statuses,
        fallback_reasons: value.fallback_reasons,
        transitions: None,
    }
}

fn benchmark_gate_flags(prepared: &PreparedSupremacy) -> BenchmarkGateFlags {
    if let Some(plan) = &prepared.gate_plan {
        return BenchmarkGateFlags::from_names(
            &plan.benchmark_flags,
            &plan.forbidden_benchmark_flags,
        );
    }
    BenchmarkGateFlags::default()
}

fn summary_passed(summary: &BenchmarkSummary) -> bool {
    summary.rows.iter().all(|row| {
        row.tlc.all_ok
            && row.interp.all_ok
            && row.trust_cg.all_ok
            && row.parity_interp_vs_tlc
            && row.parity_trust_cg_vs_tlc
            && row.trust_cg_gate_failures.is_empty()
    })
}

fn validate_binary(path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        bail!("required executable/artifact not found: {}", path.display())
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolve_ty_binary(prepared: &PreparedSupremacy, repo_root: &Path) -> Result<PathBuf> {
    if let Some(binary) = &prepared.ty_bin {
        validate_binary(binary)?;
        return Ok(binary.clone());
    }
    let target_dir = resolve_target_dir(prepared, repo_root);
    build_ty_binary(repo_root, &target_dir, &prepared.cargo_profile)?;
    let binary = target_dir
        .join(profile_binary_dir(&prepared.cargo_profile))
        .join("ty");
    validate_binary(&binary)?;
    Ok(binary)
}

fn resolve_target_dir(prepared: &PreparedSupremacy, repo_root: &Path) -> PathBuf {
    if let Some(target_dir) = &prepared.target_dir {
        return absolutize(repo_root, target_dir);
    }
    if let Some(target_dir) = env::var_os("CARGO_TARGET_DIR") {
        return absolutize(repo_root, Path::new(&target_dir));
    }
    repo_root.join("target/user")
}

fn absolutize(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn profile_binary_dir(profile: &str) -> &str {
    if profile == "dev" {
        "debug"
    } else {
        profile
    }
}

fn build_ty_binary(repo_root: &Path, target_dir: &Path, cargo_profile: &str) -> Result<()> {
    let status = Command::new("cargo")
        .arg("build")
        .arg("--profile")
        .arg(cargo_profile)
        .arg("-p")
        .arg("tla-cli")
        .arg("--target-dir")
        .arg(target_dir)
        .arg("--bin")
        .arg("ty")
        .current_dir(repo_root)
        .status()
        .with_context(|| "run cargo build")?;
    if !status.success() {
        bail!(
            "cargo build --profile {cargo_profile} -p tla-cli --target-dir {} --bin ty failed with {status}",
            target_dir.display()
        );
    }
    Ok(())
}

fn git_commit_short(repo_root: &Path) -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(repo_root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn invocation(prepared: &PreparedSupremacy) -> String {
    let mut parts = vec![
        "ty".to_string(),
        "supremacy".to_string(),
        prepared.command.to_string(),
    ];
    if let Some(runs) = prepared.runs {
        parts.extend(["--runs".to_string(), runs.to_string()]);
    }
    if prepared.cargo_profile != "release" {
        parts.extend([
            "--cargo-profile".to_string(),
            prepared.cargo_profile.clone(),
        ]);
    }
    if let Some(target_dir) = &prepared.target_dir {
        parts.extend(["--target-dir".to_string(), target_dir.display().to_string()]);
    }
    if let Some(ty_bin) = &prepared.ty_bin {
        parts.extend(["--ty-bin".to_string(), ty_bin.display().to_string()]);
    }
    for flag in &prepared.ty_flags {
        parts.extend(["--ty-flag".to_string(), flag.clone()]);
    }
    for (key, value) in &prepared.interp_env_overrides {
        parts.extend(["--interp-env".to_string(), format!("{key}={value}")]);
    }
    for (key, value) in &prepared.trust_cg_env_overrides {
        parts.extend(["--trust_cg-env".to_string(), format!("{key}={value}")]);
    }
    for spec in &prepared.specs {
        parts.extend(["--specs".to_string(), spec.clone()]);
    }
    parts.join(" ")
}

fn print_run_start(
    prepared: &PreparedSupremacy,
    spec: &str,
    mode: &str,
    run_index: usize,
    run_count: usize,
) {
    if matches!(prepared.format, SupremacyOutputFormat::Human) {
        eprintln!("[supremacy] {spec}: {mode} run {run_index}/{run_count}");
    }
}

fn write_planned_artifacts(
    repo_root: &Path,
    artifact_dir: &Path,
    command: &PlannedCommand,
) -> Result<()> {
    create_fresh_artifact_dir(artifact_dir)?;
    fs::write(artifact_dir.join("stdout.txt"), "")
        .with_context(|| format!("write {}", artifact_dir.join("stdout.txt").display()))?;
    fs::write(artifact_dir.join("stderr.txt"), "")
        .with_context(|| format!("write {}", artifact_dir.join("stderr.txt").display()))?;
    let report = json!({
        "schema": "ty.supremacy.planned_command.v1",
        "argv": command.argv,
        "cwd": command.cwd,
        "repo_relative_artifact_dir": repo_relative(repo_root, artifact_dir),
        "env_overrides": command.env_overrides,
        "status": "planned",
    });
    fs::write(
        artifact_dir.join("command.json"),
        serde_json::to_string_pretty(&report).context("serialize planned command")? + "\n",
    )
    .with_context(|| format!("write {}", artifact_dir.join("command.json").display()))?;
    Ok(())
}

fn render_markdown(summary: &BenchmarkSummary) -> String {
    let mut lines = vec![
        "# Single-Thread Supremacy Benchmark".to_string(),
        String::new(),
        format!("Timestamp: {}", summary.timestamp),
        format!("Commit: {}", summary.git_commit),
        format!("Artifact bundle: `{}`", summary.artifact_bundle),
        format!(
            "TLC controls: workers={}, JVM args=`{}`, heap_xms={}, heap_xmx={}",
            summary.launch_controls.tlc.workers,
            summary.launch_controls.tlc.jvm_args.join(" "),
            fmt_control(summary.launch_controls.tlc.heap_xms.as_deref()),
            fmt_control(summary.launch_controls.tlc.heap_xmx.as_deref()),
        ),
        format!(
            "TY controls: interp workers={}, interp cache={}, trust-cg workers={}, trust-cg cache={}, artifact_cache_disabled_env={}, native_callout_compile_jobs={}",
            summary.launch_controls.ty.interp.workers,
            fmt_control(summary.launch_controls.ty.interp.cache_dir.as_deref()),
            summary.launch_controls.ty.trust_cg.workers,
            fmt_control(summary.launch_controls.ty.trust_cg.cache_dir.as_deref()),
            fmt_control(
                summary
                    .launch_controls
                    .ty
                    .trust_cg
                    .artifact_cache_disabled_env
                    .as_deref(),
            ),
            fmt_control(
                summary
                    .launch_controls
                    .ty
                    .trust_cg
                    .native_callout_compile_jobs
                    .as_deref(),
            ),
        ),
        String::new(),
        "| Spec | TLC median | Interp median | Trust-CG wall median | TLC peak RSS | Interp peak RSS | Trust-CG peak RSS | Trust-CG cold setup | Trust-CG exec median | Batch compile | Fallback compile | Cache lookup | Materialize | Interp/TLC | Trust-CG/TLC | Trust-CG exec/TLC | Parity | Winner | Outcome |".to_string(),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |".to_string(),
    ];
    for row in &summary.rows {
        let parity = if row.parity_interp_vs_tlc && row.parity_trust_cg_vs_tlc {
            "PASS"
        } else {
            "FAIL"
        };
        let phase = row.trust_cg.phase_median_seconds.as_ref();
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.spec,
            fmt_seconds(row.tlc.median_seconds),
            fmt_seconds(row.interp.median_seconds),
            fmt_seconds(row.trust_cg.median_seconds),
            fmt_bytes(row.tlc.median_peak_rss_bytes),
            fmt_bytes(row.interp.median_peak_rss_bytes),
            fmt_bytes(row.trust_cg.median_peak_rss_bytes),
            fmt_seconds(phase.and_then(|phase| phase.cold_setup)),
            fmt_seconds(row.trust_cg.execution_median_seconds),
            fmt_seconds(phase.and_then(|phase| phase.batch_compile)),
            fmt_seconds(phase.and_then(|phase| phase.batch_fallback_per_action_compile)),
            fmt_seconds(phase.and_then(|phase| phase.batch_warm_cache_lookup)),
            fmt_seconds(phase.and_then(|phase| phase.batch_artifact_materialization)),
            fmt_ratio(row.speedup_interp_vs_tlc),
            fmt_ratio(row.speedup_trust_cg_vs_tlc),
            fmt_ratio(row.speedup_trust_cg_execution_vs_tlc),
            parity,
            row.trust_cg_evidence.winner,
            row.trust_cg_outcome,
        ));
    }
    lines.push(String::new());
    lines.push(String::new());
    lines.join("\n")
}

fn fmt_control(value: Option<&str>) -> String {
    value
        .map(|value| format!("`{value}`"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_seconds(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_ratio(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}x"))
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

fn tlaplus_examples_dir() -> Result<PathBuf> {
    let home = env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join("tlaplus-examples/specifications"))
}

fn default_tlc_jar() -> Result<PathBuf> {
    let home = env::var_os("HOME").context("HOME is not set")?;
    Ok(resolve_tlc_jar_path(
        PathBuf::from(home),
        env::var_os(ENV_TYTOOLS_JAR),
        env::var_os(ENV_TLC_JAR),
    ))
}

fn resolve_tlc_jar_path(
    home: PathBuf,
    tytools_jar: Option<OsString>,
    tlc_jar: Option<OsString>,
) -> PathBuf {
    non_empty_env_path(tytools_jar)
        .or_else(|| non_empty_env_path(tlc_jar))
        .unwrap_or_else(|| home.join(DEFAULT_TLC_JAR))
}

fn non_empty_env_path(value: Option<OsString>) -> Option<PathBuf> {
    value.and_then(|value| {
        if value.is_empty() {
            None
        } else {
            Some(PathBuf::from(value))
        }
    })
}

fn repo_relative(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .map(Path::display)
        .map(|display| display.to_string())
        .unwrap_or_else(|_| path.display().to_string())
}

fn absolute_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_schema::{
        SupremacyCommonArgs, SupremacyGateMode, SupremacyMode, SupremacyOutputFormat,
    };

    fn repo_policy_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/tlc_comparison/single_thread_supremacy_gate.json")
    }

    fn prepared_for_test(output_dir: PathBuf) -> PreparedSupremacy {
        let common = SupremacyCommonArgs {
            policy: Some(repo_policy_path()),
            output_dir: Some(output_dir),
            ty_bin: None,
            target_dir: None,
            cargo_profile: "release".to_string(),
            ty_flag: Vec::new(),
            timeout: 300,
            specs: Vec::new(),
            interp_env: Vec::new(),
            trust_cg_env: Vec::new(),
            format: SupremacyOutputFormat::Human,
        };
        PreparedSupremacy::prepare("benchmark", &common, Some(1), None, None).unwrap()
    }

    #[test]
    fn stages_coffecan_safety_spec() {
        let dir = tempfile::tempdir().unwrap();
        let examples = dir.path().join("examples");
        let coffee_dir = examples.join("CoffeeCan");
        fs::create_dir_all(&coffee_dir).unwrap();
        fs::write(
            coffee_dir.join("CoffeeCan.tla"),
            "---- MODULE CoffeeCan ----\n====\n",
        )
        .unwrap();
        let output = dir.path().join("out");

        let spec = stage_coffecan_safety_spec(&output, &examples).unwrap();

        assert_eq!(spec.name, COFFEECAN_SAFETY_SPEC_NAME);
        assert_eq!(spec.expected_states, Some(501_500));
        assert_eq!(spec.expected_raw_successors_generated, Some(1_498_502));
        assert_eq!(spec.expected_states_generated, Some(1_499_503));
        assert!(spec.tla_path.is_file());
        assert!(spec.cfg_path.is_file());
        assert!(spec.tla_path.with_file_name("CoffeeCan.tla").is_file());
        let wrapper = fs::read_to_string(&spec.tla_path).unwrap();
        assert!(wrapper.contains("Can == [black : 0..1000, white : 0..1000]"));
        assert!(wrapper.contains("TypeInvariant == can \\in Can"));
        assert!(wrapper.contains("Next =="));
        assert!(wrapper.contains("c.black + c.white = 1000"));
        let cfg = fs::read_to_string(&spec.cfg_path).unwrap();
        assert!(cfg.contains("INIT\n    SafetyInit"));
        assert!(cfg.contains("INVARIANTS\n    TypeInvariant"));
    }

    #[test]
    fn resolve_benchmark_spec_preserves_hardcoded_catalog_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let examples = dir.path().join("examples");
        fs::create_dir_all(examples.join("ewd998")).unwrap();
        fs::write(
            examples.join("ewd998/EWD998.tla"),
            "---- MODULE EWD998 ----\n====\n",
        )
        .unwrap();
        fs::write(
            examples.join("ewd998/EWD998Small.cfg"),
            "INIT Init\nNEXT Next\n",
        )
        .unwrap();

        let spec =
            resolve_benchmark_spec("EWD998Small", dir.path(), &examples, dir.path()).unwrap();

        assert_eq!(spec.name, "EWD998Small");
        assert_eq!(spec.category, "ewd998");
        assert_eq!(spec.expected_states, Some(1_520_618));
        assert_eq!(spec.expected_raw_successors_generated, Some(9_630_813));
        assert_eq!(spec.expected_states_generated, None);
        assert_eq!(spec.timeout_seconds, DEFAULT_SPEC_TIMEOUT_SECONDS);
    }

    #[test]
    fn resolve_benchmark_spec_falls_back_to_check_mode_baseline_entry() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("repo");
        let baseline_dir = repo.join("tests/tlc_comparison");
        fs::create_dir_all(&baseline_dir).unwrap();
        fs::write(
            baseline_dir.join("spec_baseline.json"),
            r#"{
  "specs": {
    "ABCorrectness": {
      "category": "small",
      "diagnose_timeout_seconds": 123,
      "notes": "fixture baseline note",
      "source": {
        "tla_path": "SpecifyingSystems/TLC/ABCorrectness.tla",
        "cfg_path": "SpecifyingSystems/TLC/ABCorrectness.cfg"
      },
      "tlc": {
        "states": 20,
        "states_generated": 34
      }
    }
  }
}"#,
        )
        .unwrap();
        let examples = dir.path().join("examples");
        fs::create_dir_all(examples.join("SpecifyingSystems/TLC")).unwrap();
        fs::write(
            examples.join("SpecifyingSystems/TLC/ABCorrectness.tla"),
            "---- MODULE ABCorrectness ----\n====\n",
        )
        .unwrap();
        fs::write(
            examples.join("SpecifyingSystems/TLC/ABCorrectness.cfg"),
            "INIT Init\nNEXT Next\n",
        )
        .unwrap();

        let spec = resolve_benchmark_spec("ABCorrectness", dir.path(), &examples, &repo).unwrap();

        assert_eq!(spec.name, "ABCorrectness");
        assert_eq!(spec.category, "small");
        assert_eq!(
            spec.tla_path,
            examples.join("SpecifyingSystems/TLC/ABCorrectness.tla")
        );
        assert_eq!(
            spec.cfg_path,
            examples.join("SpecifyingSystems/TLC/ABCorrectness.cfg")
        );
        assert_eq!(spec.expected_states, Some(20));
        assert_eq!(spec.expected_raw_successors_generated, None);
        assert_eq!(spec.expected_states_generated, Some(34));
        assert_eq!(spec.timeout_seconds, 123);
        assert_eq!(spec.notes, "fixture baseline note");
    }

    #[test]
    fn resolve_benchmark_spec_rejects_non_check_baseline_mode() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("repo");
        let baseline_dir = repo.join("tests/tlc_comparison");
        fs::create_dir_all(&baseline_dir).unwrap();
        fs::write(
            baseline_dir.join("spec_baseline.json"),
            r#"{
  "specs": {
    "SimOnly": {
      "category": "small",
      "source": {
        "tla_path": "sim/SimOnly.tla",
        "cfg_path": "sim/SimOnly.cfg",
        "mode": "simulate"
      },
      "tlc": {
        "states": null
      }
    }
  }
}"#,
        )
        .unwrap();

        let error =
            resolve_benchmark_spec("SimOnly", dir.path(), &dir.path().join("examples"), &repo)
                .unwrap_err()
                .to_string();

        assert!(error.contains("source mode is \"simulate\""));
        assert!(error.contains("only supports check-mode specs"));
    }

    #[test]
    fn builds_mode_specific_ty_env() {
        let overrides = BTreeMap::from([
            ("TY_DISABLE_ARTIFACT_CACHE".to_string(), "1".to_string()),
            ("TY_TRUST_CG_BFS".to_string(), "forced".to_string()),
        ]);

        let interp = ty_env("interp", &overrides).unwrap();
        assert_eq!(interp.get("TY_trust_cg").map(String::as_str), Some("0"));
        assert_eq!(
            interp.get("TY_TRUST_CG_BFS").map(String::as_str),
            Some("forced")
        );

        let trust_cg = ty_env("trust-cg", &overrides).unwrap();
        assert_eq!(trust_cg.get("TY_trust_cg").map(String::as_str), Some("1"));
        assert_eq!(
            trust_cg.get("TY_TRUST_CG_EXISTS").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            trust_cg
                .get("TY_DISABLE_ARTIFACT_CACHE")
                .map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn gate_benchmark_trust_cg_env_matches_gate_base_env() {
        let gate_env = gate_benchmark_trust_cg_env();

        for (key, value) in GATE_BENCHMARK_TRUST_CG_ENV {
            assert_eq!(gate_env.get(*key).map(String::as_str), Some(*value));
        }

        let trust_cg = ty_env("trust-cg", &gate_env).unwrap();
        // Flat/compiled BFS are now auto-admitted from the inferred layout, so
        // the gate env must NOT carry the retired force-enable knobs.
        assert_eq!(trust_cg.get("TY_COMPILED_BFS"), None);
        assert_eq!(trust_cg.get("TY_FLAT_BFS"), None);
        assert_eq!(
            trust_cg.get("TY_BYTECODE_VM_STATS").map(String::as_str),
            Some("1")
        );
        // Count-parity is a CLI flag (`--no-reduction` in the planned argv),
        // not an env pin: the child ignores ambient TY_AUTO_POR/TY_AUTO_SYMMETRY.
        assert_eq!(trust_cg.get("TY_AUTO_POR"), None);
        assert_eq!(trust_cg.get("TY_AUTO_SYMMETRY"), None);
    }

    #[test]
    fn writes_planned_command_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        let command = PlannedCommand {
            argv: vec!["ty".to_string(), "check".to_string()],
            cwd: dir.path().to_path_buf(),
            env_overrides: BTreeMap::from([("TY_trust_cg".to_string(), "1".to_string())]),
        };

        write_planned_artifacts(dir.path(), &artifact_dir, &command).unwrap();

        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            ""
        );
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stderr.txt")).unwrap(),
            ""
        );
        let command_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(artifact_dir.join("command.json")).unwrap())
                .unwrap();
        assert_eq!(command_json["schema"], "ty.supremacy.planned_command.v1");
        assert_eq!(command_json["status"], "planned");
        assert_eq!(command_json["repo_relative_artifact_dir"], "artifact");
        assert_eq!(command_json["env_overrides"]["TY_trust_cg"], "1");
    }

    #[test]
    fn planned_command_artifacts_reject_existing_run_dir() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir_all(artifact_dir.join("tlc-metadir")).unwrap();
        let command = PlannedCommand {
            argv: vec!["ty".to_string(), "check".to_string()],
            cwd: dir.path().to_path_buf(),
            env_overrides: BTreeMap::new(),
        };

        let err = write_planned_artifacts(dir.path(), &artifact_dir, &command).unwrap_err();

        assert!(
            err.to_string()
                .contains("supremacy artifact dir already exists"),
            "{err:#}"
        );
        assert!(!artifact_dir.join("command.json").exists());
        assert!(artifact_dir.join("tlc-metadir").is_dir());
    }

    #[test]
    fn planned_ty_run_uses_expected_backend_and_artifact_dir() {
        let dir = tempfile::tempdir().unwrap();
        let spec = BenchmarkSpec {
            name: "Example".to_string(),
            tla_path: dir.path().join("Example.tla"),
            cfg_path: dir.path().join("Example.cfg"),
            category: "example".to_string(),
            expected_states: Some(1),
            expected_raw_successors_generated: Some(2),
            expected_states_generated: None,
            timeout_seconds: 3,
            notes: String::new(),
        };

        let run = plan_ty_run(
            &spec,
            "trust-cg",
            7,
            dir.path(),
            &PathBuf::from("target/user/release/ty"),
            &BTreeMap::new(),
            &[],
            &dir.path().join("Example"),
        )
        .unwrap();

        assert_eq!(run.mode, "trust-cg");
        assert_eq!(run.run_index, 7);
        assert!(run
            .command
            .argv
            .ends_with(&["--backend".to_string(), "trust-cg".to_string()]));
        // Count-parity is requested via the flag, never via env pins.
        assert!(run.command.argv.contains(&"--no-reduction".to_string()));
        assert_eq!(run.command.env_overrides.get("TY_AUTO_POR"), None);
        assert_eq!(run.command.env_overrides.get("TY_AUTO_SYMMETRY"), None);
        assert_eq!(
            run.command
                .env_overrides
                .get("TY_trust_cg")
                .map(String::as_str),
            Some("1")
        );
        // Artifact dir naming uses the hyphenated mode ("trust-cg-run7"), matching
        // `run.mode == "trust-cg"` above (was the underscored "trust_cg-run7").
        assert!(run.artifact_dir.ends_with("trust-cg-run7"));
        assert!(run.artifact_dir.join("command.json").is_file());
    }

    #[test]
    fn planned_tlc_run_uses_absolute_paths_for_relative_generated_specs() {
        let dir = tempfile::tempdir().unwrap();
        let repo_root = dir.path();
        let spec = BenchmarkSpec {
            name: "Example".to_string(),
            tla_path: PathBuf::from("reports/generated/Example.tla"),
            cfg_path: PathBuf::from("reports/generated/Example.cfg"),
            category: "example".to_string(),
            expected_states: Some(1),
            expected_raw_successors_generated: Some(2),
            expected_states_generated: None,
            timeout_seconds: 3,
            notes: String::new(),
        };

        let run = plan_tlc_run(
            &spec,
            1,
            repo_root,
            &PathBuf::from("tlaplus/tytools.jar"),
            &PathBuf::from("reports/out/Example"),
        )
        .unwrap();

        assert_eq!(run.command.argv[0], "java");
        for arg in tlc_java_single_thread_args() {
            assert!(run.command.argv.contains(&(*arg).to_string()), "{arg}");
        }
        let jar_index = run
            .command
            .argv
            .iter()
            .position(|arg| arg == "-jar")
            .expect("TLC command should use -jar");
        assert_eq!(
            run.command.argv[jar_index + 1],
            repo_root.join("tlaplus/tytools.jar").display().to_string()
        );
        assert_eq!(
            run.command.argv[jar_index + 2],
            repo_root
                .join("reports/generated/Example.tla")
                .display()
                .to_string()
        );
        let config_index = run
            .command
            .argv
            .iter()
            .position(|arg| arg == "-config")
            .expect("TLC command should pass -config");
        assert_eq!(
            run.command.argv[config_index + 1],
            repo_root
                .join("reports/generated/Example.cfg")
                .display()
                .to_string()
        );
        let metadir_index = run
            .command
            .argv
            .iter()
            .position(|arg| arg == "-metadir")
            .expect("TLC command should pass -metadir");
        assert_eq!(
            run.command.argv[metadir_index + 1],
            repo_root
                .join("reports/out/Example/tlc-run1/tlc-metadir")
                .display()
                .to_string()
        );
        assert_eq!(run.command.cwd, repo_root.join("reports/generated"));
        assert!(run.artifact_dir.join("command.json").is_file());
    }

    #[test]
    fn tlc_jar_resolution_honors_tytools_then_tlc_env() {
        let home = PathBuf::from("/home/tester");

        assert_eq!(
            resolve_tlc_jar_path(
                home.clone(),
                Some(OsString::from("/opt/tytools.jar")),
                Some(OsString::from("/opt/legacy-tlc.jar")),
            ),
            PathBuf::from("/opt/tytools.jar")
        );
        assert_eq!(
            resolve_tlc_jar_path(
                home.clone(),
                Some(OsString::new()),
                Some(OsString::from("/opt/legacy-tlc.jar")),
            ),
            PathBuf::from("/opt/legacy-tlc.jar")
        );
        assert_eq!(
            resolve_tlc_jar_path(home.clone(), None, None),
            home.join(DEFAULT_TLC_JAR)
        );
    }

    #[test]
    fn planned_ty_run_keeps_user_flags_before_forced_backend() {
        let dir = tempfile::tempdir().unwrap();
        let spec = BenchmarkSpec {
            name: "Example".to_string(),
            tla_path: dir.path().join("Example.tla"),
            cfg_path: dir.path().join("Example.cfg"),
            category: "example".to_string(),
            expected_states: Some(1),
            expected_raw_successors_generated: Some(2),
            expected_states_generated: None,
            timeout_seconds: 3,
            notes: String::new(),
        };
        let flags = vec!["--max-depth".to_string(), "2".to_string()];

        let run = plan_ty_run(
            &spec,
            "interp",
            1,
            dir.path(),
            &PathBuf::from("target/user/release/ty"),
            &BTreeMap::from([("TY_TRUST_CG_BFS".to_string(), "custom".to_string())]),
            &flags,
            &dir.path().join("Example"),
        )
        .unwrap();

        let backend_index = run
            .command
            .argv
            .iter()
            .position(|arg| arg == "--backend")
            .unwrap();
        let max_depth_index = run
            .command
            .argv
            .iter()
            .position(|arg| arg == "--max-depth")
            .unwrap();
        assert!(max_depth_index < backend_index);
        assert!(run
            .command
            .argv
            .ends_with(&["--backend".to_string(), "interpreter".to_string()]));
        assert_eq!(
            run.command
                .env_overrides
                .get("TY_trust_cg")
                .map(String::as_str),
            Some("0")
        );
        assert_eq!(
            run.command
                .env_overrides
                .get("TY_TRUST_CG_BFS")
                .map(String::as_str),
            Some("custom")
        );
    }

    #[test]
    fn resolve_target_dir_honors_explicit_and_profile_layout() {
        let dir = tempfile::tempdir().unwrap();
        let mut prepared = prepared_for_test(dir.path().join("out"));
        prepared.target_dir = Some(PathBuf::from("target/custom"));
        prepared.cargo_profile = "release-canary".to_string();

        assert_eq!(
            resolve_target_dir(&prepared, dir.path()),
            dir.path().join("target/custom")
        );
        assert_eq!(
            profile_binary_dir(&prepared.cargo_profile),
            "release-canary"
        );
        assert_eq!(profile_binary_dir("dev"), "debug");
    }

    #[test]
    fn trust_cg_overrides_are_mode_scoped_and_include_cache_dir() {
        let dir = tempfile::tempdir().unwrap();
        let mut prepared = prepared_for_test(dir.path().join("out"));
        prepared
            .trust_cg_env_overrides
            .insert("TY_DISABLE_ARTIFACT_CACHE".to_string(), "1".to_string());
        prepared
            .interp_env_overrides
            .insert("TY_TRUST_CG_BFS".to_string(), "interp-only".to_string());

        let trust_cg = trust_cg_env_overrides(&prepared).unwrap();
        let expected_cache_dir = prepared
            .output_dir
            .join("trust_cg-artifact-cache")
            .display()
            .to_string();

        assert_eq!(
            trust_cg
                .get("TY_DISABLE_ARTIFACT_CACHE")
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(trust_cg.get("TY_TRUST_CG_BFS"), None);
        assert_eq!(
            trust_cg.get("TY_CACHE_DIR").map(String::as_str),
            Some(expected_cache_dir.as_str())
        );
    }

    #[test]
    fn full_native_fused_gate_overrides_include_strict_launch_env() {
        let dir = tempfile::tempdir().unwrap();
        let common = SupremacyCommonArgs {
            policy: Some(repo_policy_path()),
            output_dir: Some(dir.path().join("out")),
            ty_bin: None,
            target_dir: None,
            cargo_profile: "release".to_string(),
            ty_flag: Vec::new(),
            timeout: 300,
            specs: Vec::new(),
            interp_env: Vec::new(),
            trust_cg_env: Vec::new(),
            format: SupremacyOutputFormat::Human,
        };
        let prepared = PreparedSupremacy::prepare(
            "gate",
            &common,
            Some(1),
            Some(SupremacyGateMode::FullNativeFused),
            Some(SupremacyMode::Warn),
        )
        .unwrap();

        let trust_cg = trust_cg_env_overrides(&prepared).unwrap();

        assert_eq!(
            trust_cg
                .get("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS")
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(
            trust_cg
                .get("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST")
                .map(String::as_str),
            Some("strict")
        );
        assert_eq!(
            trust_cg
                .get("TY_TRUST_CG_NATIVE_FUSED_STRICT")
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(
            trust_cg
                .get("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP")
                .map(String::as_str),
            Some("1")
        );
        assert!(!trust_cg.contains_key("TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP"));
    }
}
