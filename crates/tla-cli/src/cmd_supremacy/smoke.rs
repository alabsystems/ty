// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::Serialize;
use serde_json::{json, Value};

use super::PreparedSupremacy;
use crate::cli_schema::SupremacyOutputFormat;

const COFFEECAN_SAFETY_SPEC_NAME: &str = "CoffeeCan1000BeansSafety";
const COFFEECAN_SAFETY_BEANS: usize = 1000;
const STRICT_SELFTEST_ENV: &str = "TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST";
const STRICT_NATIVE_FUSED_ENV: &str = "TY_TRUST_CG_NATIVE_FUSED_STRICT";
const REPLAY_ARTIFACT_DIR_ENV: &str = "TY_TRUST_CG_REPLAY_ARTIFACT_DIR";
const REPLAY_ARTIFACT_FILTER_ENV: &str = "TY_TRUST_CG_REPLAY_ARTIFACT_FILTER";
const REPLAY_TY_GIT_COMMIT_ENV: &str = "TY_TRUST_CG_REPLAY_TY_GIT_COMMIT";
const CRASH_PACKET_SCHEMA: &str = "ty.trust_cg.native_crash_packet.v1";
const CRASH_REPORT_POLL_ATTEMPTS: usize = 20;

const FLAT_PRIMARY_REBUILD_MARKER: &str =
    "[compiled-bfs] clearing layout-sensitive compiled artifacts before rebuild: \
     reason=flat_state_primary layout promotion";

// Auto-POR/auto-symmetry are NOT pinned here: those semantic levers are
// controlled by CLI flags only (the child `ty check` ignores ambient
// TY_AUTO_POR / TY_AUTO_SYMMETRY). The count-parity `--no-reduction` flag is
// injected into the smoke argv by `build_check_command` instead.
const TRUST_CG_SMOKE_ENV: &[(&str, &str)] = &[
    ("TY_trust_cg", "1"),
    ("TY_TRUST_CG_BFS", "1"),
    ("TY_TRUST_CG_EXISTS", "1"),
    ("TY_BYTECODE_VM", "1"),
    ("TY_BYTECODE_VM_STATS", "1"),
    ("TY_SKIP_LIVENESS", "1"),
    ("TY_DISABLE_ARTIFACT_CACHE", "1"),
    (STRICT_SELFTEST_ENV, "strict"),
    (STRICT_NATIVE_FUSED_ENV, "1"),
    ("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS", "1"),
    ("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP", "1"),
];

const COMMON_REQUIRED_SUBSTRINGS: &[&str] = &[
    "Search completeness: bounded",
    "[trust-cg] trust-codegen native compilation enabled (default engine under AUTO selection; opt out via --backend interpreter)",
    "[trust-cg] CompiledBfsStep built:",
    "trust_cg_bfs_level_active=true",
    "trust_cg_native_fused_level_active=true",
    "trust_cg_bfs_level_loop_kind=native_fused_trust_cg_parent_loop",
    "trust_cg_native_fused_regular_invariants_checked=true",
    "[compiled-bfs] activating compiled BFS loop",
    "flat_bfs_frontier_active=true",
    "[trust_cg-selftest] native fused callout selftest complete",
];

const COMMON_FORBIDDEN_SUBSTRINGS: &[&str] = &[
    "CompiledBfsStep not eligible",
    "CompiledBfsStep requested interpreter fallback",
    "requested interpreter fallback",
    "native fused CompiledBfsLevel unavailable",
    "compiled BFS fallback",
    "compiled BFS requested interpreter fallback",
    "compiled BFS disabled",
    "compiled BFS step available but not enabled",
    "fused level error",
    "step error",
    "native fused fallback",
    "native fused requested interpreter fallback",
    "falling back to prototype",
    "prototype Rust parent loop",
    "compiled_bfs_step_active=false",
    "compiled_bfs_level_active=false",
    "trust_cg_native_fused_level_active=false",
    "trust_cg_native_fused_regular_invariants_checked=false",
    "trust_cg_native_fused_mode=action_only",
    "flat_bfs_frontier_active=false",
    "native fused CompiledBfsLevel using action-only fallback",
    "native fused level checks invariants in Rust after flat dedup",
    "state constraints require trust-codegen native fused constraint support",
    "state constraints require native fused constraint pruning",
    "native fused CompiledBfsLevel not eligible for state constraints",
    "native fused level does not report active state constraints",
    "constrained native fused BFS not eligible",
    "state constraints missing native entries",
    "failed to compile action",
    "failed to compile invariant",
    "failed to compile state constraint",
    "missing bytecode for state constraint",
    "missing native code",
    "unsupported opcode for trust-ir backend",
    "register allocation failed",
    "native BFS level generation is blocked",
];

const COMMON_FORBIDDEN_FULL_OUTPUT_SUBSTRINGS: &[&str] = &[
    "[trust_cg-selftest] native fused callout selftest failed",
    "[trust_cg-selftest] failing closed",
];

#[derive(Clone)]
struct SmokeSpec {
    name: &'static str,
    tla_path: Cow<'static, str>,
    cfg_path: Cow<'static, str>,
    bound_args: &'static [&'static str],
    native_fused_action_count: usize,
    native_fused_invariant_count: usize,
    required_substrings: &'static [&'static str],
    native_fused_state_constraint_count: Option<usize>,
    native_fused_mode: &'static str,
    native_fused_loop_label: &'static str,
    min_transitions: usize,
    expected_native_state_len: Option<usize>,
}

impl SmokeSpec {
    fn absolute_paths(&self, examples_dir: &Path) -> (PathBuf, PathBuf) {
        let tla_path = Path::new(self.tla_path.as_ref());
        let cfg_path = Path::new(self.cfg_path.as_ref());
        let tla_path = if tla_path.is_absolute() {
            tla_path.to_path_buf()
        } else {
            examples_dir.join(tla_path)
        };
        let cfg_path = if cfg_path.is_absolute() {
            cfg_path.to_path_buf()
        } else {
            examples_dir.join(cfg_path)
        };
        (tla_path, cfg_path)
    }
}

#[derive(Serialize)]
struct CommandResult {
    argv: Vec<String>,
    cwd: String,
    pid: Option<u32>,
    returncode: i32,
    signal: Option<i32>,
    timed_out: bool,
    started_unix_ms: Option<u64>,
    ended_unix_ms: Option<u64>,
    elapsed_seconds: f64,
    stdout: String,
    stderr: String,
    env_overrides: BTreeMap<String, String>,
    crash_packet: Option<CrashPacketSummary>,
}

#[derive(Serialize)]
struct SmokeRun {
    spec: String,
    artifact_dir: String,
    pid: Option<u32>,
    returncode: i32,
    signal: Option<i32>,
    timed_out: bool,
    elapsed_seconds: f64,
    crash_packet: Option<CrashPacketSummary>,
    failures: Vec<String>,
    command: Vec<String>,
    cwd: String,
    env_overrides: BTreeMap<String, String>,
    stderr_tail: Vec<String>,
    compiled_bfs_level_loop_initial_states: Option<usize>,
    compiled_bfs_level_loop_fused: Option<bool>,
    compiled_bfs_levels_completed: Option<usize>,
    compiled_bfs_parents_processed: Option<usize>,
    compiled_bfs_successors_generated: Option<usize>,
    compiled_bfs_successors_new: Option<usize>,
    compiled_bfs_total_states: Option<usize>,
    ok: bool,
}

#[derive(Clone, Serialize)]
struct CrashJitSymbolRangeSummary {
    start: String,
    end: String,
    code_len: u64,
}

#[derive(Clone, Serialize)]
struct CrashPacketSummary {
    artifact_root: String,
    replay_artifact_dir: String,
    path: String,
    os_crash_report: String,
    fault_signal: Option<String>,
    fault_signal_code: Option<i32>,
    fault_pc: Option<String>,
    fault_pc_decimal: Option<u64>,
    fault_address: Option<String>,
    fault_address_decimal: Option<u64>,
    jit_symbol: Option<String>,
    jit_symbol_range: Option<CrashJitSymbolRangeSummary>,
    pc_map_offset: Option<String>,
    pc_map_offset_decimal: Option<u64>,
    pc_map_block: Option<String>,
    metadata_path: Option<String>,
    non_promoting: bool,
    status_reason: String,
    diagnostics: Vec<String>,
}

#[derive(Clone)]
struct MacCrashReport {
    path: PathBuf,
    header: Value,
    body: Value,
    modified_unix_ms: Option<u64>,
    pid: Option<u64>,
    proc_name: Option<String>,
    capture_time: Option<String>,
    incident: Option<String>,
    exception_signal: Option<String>,
    termination_code: Option<i32>,
    fault_pc: Option<u64>,
    fault_address: Option<u64>,
    faulting_thread_index: Option<usize>,
    faulting_thread: Option<Value>,
}

#[derive(Clone)]
struct JitFaultResolution {
    metadata_path: PathBuf,
    module_name: String,
    stage: String,
    function_name: String,
    runtime_start: u64,
    code_len: u64,
    symbol_offset: Option<String>,
    pc_map_offset: u64,
    nearest_block: Option<Value>,
    next_block: Option<Value>,
    diagnostics: Vec<String>,
}

pub(super) fn run_smoke(prepared: PreparedSupremacy) -> Result<()> {
    if prepared.timeout_seconds == 0 {
        bail!("--timeout must be >= 1");
    }
    let repo_root = env::current_dir().context("resolve current working directory")?;
    let examples_dir = tlaplus_examples_dir()?;
    let binary = resolve_binary(
        &repo_root,
        prepared.ty_bin.as_deref(),
        prepared.target_dir.as_deref(),
        &prepared.cargo_profile,
    )?;
    let env_overrides = smoke_env_with_overrides(&prepared.trust_cg_env_overrides)?;
    let timeout = Duration::from_secs(prepared.timeout_seconds);

    let mut runs = Vec::new();
    for spec_name in &prepared.specs {
        let spec =
            smoke_spec(spec_name).with_context(|| format!("unknown smoke spec {spec_name:?}"))?;
        let preview = build_check_command(&binary, &spec, &examples_dir);
        print_smoke_start(&prepared, &spec, &preview);
        let run = run_spec(
            spec.clone(),
            &binary,
            timeout,
            &prepared.output_dir,
            &repo_root,
            &examples_dir,
            &env_overrides,
        )?;
        print_smoke_result(&spec, &run);
        runs.push(run);
    }

    let summary = json!({
        "schema": "ty.trust_cg_native_fused_smoke.summary.v1",
        "timestamp": chrono::Local::now().format("%Y-%m-%dT%H%M%S").to_string(),
        "binary": binary,
        "artifact_bundle": repo_relative(&repo_root, &prepared.output_dir),
        "env_overrides": env_overrides,
        "runs": runs,
    });
    let summary_json = serde_json::to_string_pretty(&summary).context("serialize smoke summary")?;
    fs::write(
        prepared.output_dir.join("summary.json"),
        summary_json + "\n",
    )
    .with_context(|| {
        format!(
            "write {}",
            prepared.output_dir.join("summary.json").display()
        )
    })?;
    let markdown = render_markdown(&summary);
    fs::write(prepared.output_dir.join("summary.md"), &markdown)
        .with_context(|| format!("write {}", prepared.output_dir.join("summary.md").display()))?;

    match prepared.format {
        SupremacyOutputFormat::Human => {
            eprintln!(
                "[trust-cg-native-fused-smoke] wrote {}",
                prepared.output_dir.display()
            );
        }
        SupremacyOutputFormat::Json => println!("{}", summary),
        SupremacyOutputFormat::Markdown => println!("{markdown}"),
    }

    let any_failed = summary
        .get("runs")
        .and_then(|runs| runs.as_array())
        .is_some_and(|runs| runs.iter().any(|run| run.get("ok") != Some(&json!(true))));
    if any_failed {
        bail!(
            "ty supremacy smoke failed; see {}",
            prepared.output_dir.display()
        );
    }
    Ok(())
}

fn smoke_specs() -> &'static [SmokeSpec] {
    &[
        SmokeSpec {
            name: COFFEECAN_SAFETY_SPEC_NAME,
            tla_path: Cow::Borrowed("generated-stage/CoffeeCanSafetyBench.tla"),
            cfg_path: Cow::Borrowed("generated-stage/CoffeeCanSafetyBench.cfg"),
            bound_args: &["--max-depth", "1"],
            native_fused_action_count: 4,
            native_fused_invariant_count: 1,
            native_fused_state_constraint_count: Some(0),
            expected_native_state_len: Some(2),
            native_fused_mode: "invariant_checking",
            native_fused_loop_label: "invariant-checking native fused Trust-CG parent loop",
            required_substrings: &[
                "Model checking stopped: depth limit reached.",
                "Max depth: 1",
                "States found: 2001",
                "Initial states: 1001",
                "Transitions: 2997",
                "[trust-cg] executable action coverage: trust_cg_actions_compiled=4 trust_cg_actions_total=4",
                "[trust-cg] compilation complete: 4/4 actions, 1/1 invariants",
            ],
            min_transitions: 1,
        },
        SmokeSpec {
            name: "EWD998Small",
            tla_path: Cow::Borrowed("ewd998/EWD998.tla"),
            cfg_path: Cow::Borrowed("ewd998/EWD998Small.cfg"),
            bound_args: &["--max-depth", "3"],
            native_fused_action_count: 15,
            native_fused_invariant_count: 3,
            native_fused_state_constraint_count: Some(1),
            expected_native_state_len: Some(15),
            native_fused_mode: "state_constraint_checking",
            native_fused_loop_label: "state-constrained native fused Trust-CG parent loop",
            required_substrings: &[
                "Model checking stopped: depth limit reached.",
                "Max depth: 3",
                "States found: 9404",
                "[trust-cg] executable action coverage: trust_cg_actions_compiled=15 trust_cg_actions_total=15",
                "[trust-cg] compilation complete: 15/15 actions, 3/3 invariants, 1/1 state constraints compiled",
            ],
            min_transitions: 1,
        },
        SmokeSpec {
            name: "MCLamportMutex",
            tla_path: Cow::Borrowed("lamport_mutex/MCLamportMutex.tla"),
            cfg_path: Cow::Borrowed("lamport_mutex/MCLamportMutex.cfg"),
            bound_args: &["--max-depth", "1"],
            native_fused_action_count: 27,
            native_fused_invariant_count: 3,
            native_fused_state_constraint_count: Some(1),
            expected_native_state_len: Some(89),
            native_fused_mode: "state_constraint_checking",
            native_fused_loop_label: "state-constrained native fused Trust-CG parent loop",
            required_substrings: &[
                "Model checking stopped: depth limit reached.",
                "Max depth: 1",
                "States found: 4",
                "Transitions: 3",
                "[trust-cg] executable action coverage: trust_cg_actions_compiled=27 trust_cg_actions_total=27",
                "[trust-cg] compilation complete: 27/27 actions, 3/3 invariants, 1/1 state constraints compiled",
            ],
            min_transitions: 0,
        },
    ]
}

fn smoke_spec(name: &str) -> Option<SmokeSpec> {
    smoke_specs().iter().find(|spec| spec.name == name).cloned()
}

fn tlaplus_examples_dir() -> Result<PathBuf> {
    let home = env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join("tlaplus-examples/specifications"))
}

fn resolve_binary(
    repo_root: &Path,
    explicit: Option<&Path>,
    target_dir: Option<&Path>,
    cargo_profile: &str,
) -> Result<PathBuf> {
    if let Some(binary) = explicit {
        if binary.is_file() {
            return Ok(binary.to_path_buf());
        }
        bail!("ty binary not found: {}", binary.display());
    }
    let target_dir = match target_dir {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => repo_root.join(path),
        None => repo_root.join("target/user"),
    };
    let binary_name = if cfg!(windows) { "ty.exe" } else { "ty" };
    let binary = target_dir
        .join(profile_binary_dir(cargo_profile))
        .join(binary_name);
    if binary.is_file() {
        return Ok(binary);
    }
    bail!(
        "benchmark binary not found: {}\nBuild it with:\n  cargo build --profile {} -p tla-cli --target-dir {} --bin ty\nor pass --ty-bin /path/to/ty",
        binary.display(),
        cargo_profile,
        target_dir.display()
    );
}

fn profile_binary_dir(profile: &str) -> &str {
    if profile == "dev" {
        "debug"
    } else {
        profile
    }
}

fn git_commit(repo_root: &Path) -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn smoke_env_with_overrides(
    cli_overrides: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    let mut env: BTreeMap<String, String> = TRUST_CG_SMOKE_ENV
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    env.extend(cli_overrides.clone());
    require_env_value(&env, STRICT_SELFTEST_ENV, "strict")?;
    require_env_value(&env, STRICT_NATIVE_FUSED_ENV, "1")?;
    require_env_value(&env, "TY_DISABLE_ARTIFACT_CACHE", "1")?;
    require_env_value(&env, "TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS", "1")?;
    require_env_value(&env, "TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP", "1")?;
    Ok(env)
}

fn require_env_value(env: &BTreeMap<String, String>, key: &str, value: &str) -> Result<()> {
    if env.get(key).map(String::as_str) != Some(value) {
        bail!("native fused smoke requires {key}={value}");
    }
    Ok(())
}

fn build_check_command(binary: &Path, spec: &SmokeSpec, examples_dir: &Path) -> Vec<String> {
    let (tla_path, cfg_path) = spec.absolute_paths(examples_dir);
    let mut argv = vec![
        binary.display().to_string(),
        "check".to_string(),
        tla_path.display().to_string(),
        "--config".to_string(),
        cfg_path.display().to_string(),
        "--workers".to_string(),
        "1".to_string(),
        "--force".to_string(),
        // Count-parity lever (was the TY_AUTO_POR/TY_AUTO_SYMMETRY env pins):
        // the child `ty check` ignores ambient env for these semantic levers.
        "--no-reduction".to_string(),
    ];
    argv.extend(spec.bound_args.iter().map(|arg| (*arg).to_string()));
    argv.extend(["--backend".to_string(), "trust-cg".to_string()]);
    argv
}

fn print_smoke_start(prepared: &PreparedSupremacy, spec: &SmokeSpec, command: &[String]) {
    if matches!(prepared.format, SupremacyOutputFormat::Human) {
        eprintln!(
            "[trust-cg-native-fused-smoke] {}: {}",
            spec.name,
            shell_join(command)
        );
    }
}

fn print_smoke_result(spec: &SmokeSpec, run: &SmokeRun) {
    let status = if run.ok { "PASS" } else { "FAIL" };
    eprintln!(
        "[trust-cg-native-fused-smoke] {}: {} ({})",
        spec.name, status, run.artifact_dir
    );
    for failure in &run.failures {
        eprintln!("  - {failure}");
    }
}

fn run_spec(
    mut spec: SmokeSpec,
    binary: &Path,
    timeout: Duration,
    output_dir: &Path,
    repo_root: &Path,
    examples_dir: &Path,
    env_overrides: &BTreeMap<String, String>,
) -> Result<SmokeRun> {
    let spec_dir = output_dir.join(spec.name);
    fs::create_dir_all(&spec_dir).with_context(|| format!("create {}", spec_dir.display()))?;
    let mut run_env_overrides = env_overrides.clone();
    run_env_overrides.insert(
        REPLAY_ARTIFACT_DIR_ENV.to_string(),
        spec_dir.join("trust_cg-replay").display().to_string(),
    );
    run_env_overrides
        .entry(REPLAY_ARTIFACT_FILTER_ENV.to_string())
        .or_insert_with(|| "all".to_string());
    run_env_overrides.insert(REPLAY_TY_GIT_COMMIT_ENV.to_string(), git_commit(repo_root));

    if spec.name == COFFEECAN_SAFETY_SPEC_NAME {
        match stage_coffecan_safety_spec(&spec_dir, examples_dir) {
            Ok(staged) => spec = staged,
            Err(err) => {
                let command = build_check_command(binary, &spec, examples_dir);
                let result = CommandResult {
                    argv: command.clone(),
                    cwd: repo_root.display().to_string(),
                    pid: None,
                    returncode: 1,
                    signal: None,
                    timed_out: false,
                    started_unix_ms: None,
                    ended_unix_ms: None,
                    elapsed_seconds: 0.0,
                    stdout: String::new(),
                    stderr: err.to_string(),
                    env_overrides: run_env_overrides.clone(),
                    crash_packet: None,
                };
                write_artifacts(&spec_dir, &result)?;
                return Ok(smoke_run_from_failure(
                    &spec,
                    &spec_dir,
                    repo_root,
                    result,
                    vec![err.to_string()],
                ));
            }
        }
    }

    let command = build_check_command(binary, &spec, examples_dir);
    let file_failures = validate_spec_files(&spec, examples_dir);
    if !file_failures.is_empty() {
        let result = CommandResult {
            argv: command,
            cwd: repo_root.display().to_string(),
            pid: None,
            returncode: 1,
            signal: None,
            timed_out: false,
            started_unix_ms: None,
            ended_unix_ms: None,
            elapsed_seconds: 0.0,
            stdout: String::new(),
            stderr: file_failures.join("\n"),
            env_overrides: run_env_overrides.clone(),
            crash_packet: None,
        };
        write_artifacts(&spec_dir, &result)?;
        return Ok(smoke_run_from_failure(
            &spec,
            &spec_dir,
            repo_root,
            result,
            file_failures,
        ));
    }

    let mut result = match run_command(command.clone(), repo_root, timeout, &run_env_overrides) {
        Ok(result) => result,
        Err(err) => CommandResult {
            argv: command,
            cwd: repo_root.display().to_string(),
            pid: None,
            returncode: 1,
            signal: None,
            timed_out: false,
            started_unix_ms: None,
            ended_unix_ms: None,
            elapsed_seconds: 0.0,
            stdout: String::new(),
            stderr: err.to_string(),
            env_overrides: run_env_overrides.clone(),
            crash_packet: None,
        },
    };
    maybe_capture_crash_packet(&spec_dir, repo_root, &mut result);
    write_artifacts(&spec_dir, &result)?;
    let mut failures = validate_output(&spec, &result.stdout, &result.stderr);
    if result.returncode != 0 {
        failures.insert(0, format!("command exited with {}", result.returncode));
    }
    Ok(smoke_run_from_result(
        &spec, &spec_dir, repo_root, result, failures,
    ))
}

fn stage_coffecan_safety_spec(spec_dir: &Path, examples_dir: &Path) -> Result<SmokeSpec> {
    let generated_dir = spec_dir.join("generated-specs").join("CoffeeCanSafety1000");
    fs::create_dir_all(&generated_dir)
        .with_context(|| format!("create {}", generated_dir.display()))?;
    let source = examples_dir.join("CoffeeCan/CoffeeCan.tla");
    if !source.is_file() {
        bail!("CoffeeCan source not found: {}", source.display());
    }
    fs::copy(&source, generated_dir.join("CoffeeCan.tla"))
        .with_context(|| format!("copy {}", source.display()))?;
    let wrapper_tla = generated_dir.join("CoffeeCanSafetyBench.tla");
    let wrapper_cfg = generated_dir.join("CoffeeCanSafetyBench.cfg");
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
    let wrapper_tla = fs::canonicalize(&wrapper_tla)
        .with_context(|| format!("canonicalize {}", wrapper_tla.display()))?;
    let wrapper_cfg = fs::canonicalize(&wrapper_cfg)
        .with_context(|| format!("canonicalize {}", wrapper_cfg.display()))?;
    Ok(SmokeSpec {
        tla_path: Cow::Owned(wrapper_tla.to_string_lossy().into_owned()),
        cfg_path: Cow::Owned(wrapper_cfg.to_string_lossy().into_owned()),
        ..smoke_spec(COFFEECAN_SAFETY_SPEC_NAME).expect("CoffeeCan smoke spec")
    })
}

fn validate_spec_files(spec: &SmokeSpec, examples_dir: &Path) -> Vec<String> {
    let (tla_path, cfg_path) = spec.absolute_paths(examples_dir);
    let mut failures = Vec::new();
    if !tla_path.is_file() {
        failures.push(format!("TLA file not found: {}", tla_path.display()));
    }
    if !cfg_path.is_file() {
        failures.push(format!("config file not found: {}", cfg_path.display()));
    }
    failures
}

fn run_command(
    argv: Vec<String>,
    repo_root: &Path,
    timeout: Duration,
    env_overrides: &BTreeMap<String, String>,
) -> Result<CommandResult> {
    let started = Instant::now();
    let started_unix_ms = system_time_millis(SystemTime::now());
    let mut command = Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .current_dir(repo_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command.env_clear();
    for (key, value) in env::vars_os() {
        if !key.to_string_lossy().starts_with("TY_") {
            command.env(key, value);
        }
    }
    command.envs(env_overrides);
    let mut child = command
        .spawn()
        .with_context(|| format!("spawn {}", shell_join(&argv)))?;
    let pid = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_handle = stdout.map(|mut stream| {
        thread::spawn(move || {
            let mut text = String::new();
            let _ = stream.read_to_string(&mut text);
            text
        })
    });
    let stderr_handle = stderr.map(|mut stream| {
        thread::spawn(move || {
            let mut text = String::new();
            let _ = stream.read_to_string(&mut text);
            text
        })
    });

    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait().context("poll smoke subprocess")? {
            break status;
        }
        if started.elapsed() >= timeout {
            timed_out = true;
            let _ = child.kill();
            break child.wait().context("wait for killed smoke subprocess")?;
        }
        thread::sleep(Duration::from_millis(100));
    };

    let stdout = join_reader(stdout_handle);
    let mut stderr = join_reader(stderr_handle);
    let ended_unix_ms = system_time_millis(SystemTime::now());
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let signal = if timed_out {
        None
    } else {
        exit_signal(&status)
    };
    let returncode = if timed_out {
        if stderr.is_empty() {
            stderr = format!("Timeout after {} seconds", timeout.as_secs());
        } else {
            let _ = write!(stderr, "\n\nTimeout after {} seconds", timeout.as_secs());
        }
        124
    } else {
        status.code().unwrap_or(1)
    };

    Ok(CommandResult {
        argv,
        cwd: repo_root.display().to_string(),
        pid: Some(pid),
        returncode,
        signal,
        timed_out,
        started_unix_ms: Some(started_unix_ms),
        ended_unix_ms: Some(ended_unix_ms),
        elapsed_seconds,
        stdout,
        stderr,
        env_overrides: env_overrides.clone(),
        crash_packet: None,
    })
}

#[cfg(unix)]
fn exit_signal(status: &ExitStatus) -> Option<i32> {
    status.signal()
}

#[cfg(not(unix))]
fn exit_signal(_status: &ExitStatus) -> Option<i32> {
    None
}

fn join_reader(handle: Option<thread::JoinHandle<String>>) -> String {
    handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default()
}

fn system_time_millis(time: SystemTime) -> u64 {
    let millis = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}

fn maybe_capture_crash_packet(spec_dir: &Path, repo_root: &Path, result: &mut CommandResult) {
    match capture_crash_packet(spec_dir, repo_root, result) {
        Ok(Some(packet)) => {
            eprintln!(
                "[trust-cg-native-fused-smoke] captured crash packet: {}",
                packet.path
            );
            result.crash_packet = Some(packet);
        }
        Ok(None) => {}
        Err(err) => {
            eprintln!("[trust-cg-native-fused-smoke] failed to capture crash packet: {err:#}");
        }
    }
}

fn capture_crash_packet(
    spec_dir: &Path,
    repo_root: &Path,
    result: &CommandResult,
) -> Result<Option<CrashPacketSummary>> {
    capture_crash_packet_from_report_dirs(
        spec_dir,
        repo_root,
        result,
        &macos_diagnostic_report_dirs(),
        CRASH_REPORT_POLL_ATTEMPTS,
    )
}

fn capture_crash_packet_from_report_dirs(
    spec_dir: &Path,
    repo_root: &Path,
    result: &CommandResult,
    report_dirs: &[PathBuf],
    poll_attempts: usize,
) -> Result<Option<CrashPacketSummary>> {
    if result.timed_out || result.signal.is_none() {
        return Ok(None);
    }
    let Some(pid) = result.pid else {
        return Ok(None);
    };
    let Some(replay_dir) = result
        .env_overrides
        .get(REPLAY_ARTIFACT_DIR_ENV)
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
    else {
        return Ok(None);
    };

    let report = wait_for_matching_diagnostic_report(
        pid,
        &result.argv,
        result.started_unix_ms,
        result.ended_unix_ms,
        report_dirs,
        poll_attempts,
    )?;
    let Some(report) = report else {
        return Ok(None);
    };

    let resolution = report
        .fault_pc
        .and_then(|fault_pc| resolve_jit_fault(&replay_dir, fault_pc).transpose())
        .transpose()?;

    write_crash_packet(repo_root, spec_dir, result, &replay_dir, report, resolution)
}

fn macos_diagnostic_report_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = env::var_os("HOME") {
        dirs.push(
            PathBuf::from(home)
                .join("Library")
                .join("Logs")
                .join("DiagnosticReports"),
        );
    }
    dirs.push(PathBuf::from("/Library/Logs/DiagnosticReports"));
    dirs
}

fn wait_for_matching_diagnostic_report(
    pid: u32,
    argv: &[String],
    started_unix_ms: Option<u64>,
    ended_unix_ms: Option<u64>,
    report_dirs: &[PathBuf],
    poll_attempts: usize,
) -> Result<Option<MacCrashReport>> {
    for attempt in 0..=poll_attempts {
        if let Some(report) =
            find_matching_diagnostic_report(pid, argv, started_unix_ms, ended_unix_ms, report_dirs)?
        {
            return Ok(Some(report));
        }
        if attempt < poll_attempts {
            thread::sleep(Duration::from_millis(250));
        }
    }
    Ok(None)
}

fn find_matching_diagnostic_report(
    pid: u32,
    argv: &[String],
    started_unix_ms: Option<u64>,
    ended_unix_ms: Option<u64>,
    report_dirs: &[PathBuf],
) -> Result<Option<MacCrashReport>> {
    let expected_proc = argv
        .first()
        .and_then(|arg| Path::new(arg).file_name())
        .map(|name| name.to_string_lossy().into_owned());
    let mut reports = Vec::new();
    for dir in report_dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries {
            let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("ips") {
                continue;
            }
            let Some(report) = parse_macos_ips_report(&path)? else {
                continue;
            };
            if report.pid != Some(u64::from(pid)) {
                continue;
            }
            if let (Some(expected), Some(actual)) = (&expected_proc, &report.proc_name) {
                if actual != expected {
                    continue;
                }
            }
            if !report_matches_time_window(&report, started_unix_ms, ended_unix_ms) {
                continue;
            }
            reports.push(report);
        }
    }
    reports.sort_by_key(|report| report.modified_unix_ms.unwrap_or_default());
    Ok(reports.pop())
}

fn report_matches_time_window(
    report: &MacCrashReport,
    started_unix_ms: Option<u64>,
    ended_unix_ms: Option<u64>,
) -> bool {
    let Some(modified) = report.modified_unix_ms else {
        return true;
    };
    let lower = started_unix_ms.unwrap_or_default().saturating_sub(10_000);
    let upper = ended_unix_ms.unwrap_or(u64::MAX).saturating_add(60_000);
    modified >= lower && modified <= upper
}

fn parse_macos_ips_report(path: &Path) -> Result<Option<MacCrashReport>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let Some((header_text, body_text)) = text.split_once('\n') else {
        return Ok(None);
    };
    let header: Value = match serde_json::from_str(header_text.trim()) {
        Ok(header) => header,
        Err(_) => return Ok(None),
    };
    let body: Value = match serde_json::from_str(body_text.trim()) {
        Ok(body) => body,
        Err(_) => return Ok(None),
    };
    let metadata = fs::metadata(path).ok();
    let modified_unix_ms = metadata
        .and_then(|metadata| metadata.modified().ok())
        .map(system_time_millis);
    let faulting_thread_index = body
        .get("faultingThread")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let faulting_thread = faulting_thread_index
        .and_then(|idx| body.get("threads").and_then(Value::as_array)?.get(idx))
        .cloned();
    let thread_pc = faulting_thread
        .as_ref()
        .and_then(|thread| value_at_path_u64(thread, &["threadState", "pc", "value"]));
    let exception_pc = body
        .get("exception")
        .and_then(|exception| exception.get("rawCodes"))
        .and_then(Value::as_array)
        .and_then(|codes| codes.get(1))
        .and_then(Value::as_u64);
    let fault_address = faulting_thread
        .as_ref()
        .and_then(|thread| value_at_path_u64(thread, &["threadState", "far", "value"]))
        .filter(|value| *value != 0)
        .or(exception_pc);

    Ok(Some(MacCrashReport {
        path: path.to_path_buf(),
        header: header.clone(),
        body: body.clone(),
        modified_unix_ms,
        pid: body.get("pid").and_then(Value::as_u64),
        proc_name: body
            .get("procName")
            .and_then(Value::as_str)
            .map(str::to_string),
        capture_time: body
            .get("captureTime")
            .and_then(Value::as_str)
            .or_else(|| header.get("timestamp").and_then(Value::as_str))
            .map(str::to_string),
        incident: body
            .get("incident")
            .and_then(Value::as_str)
            .or_else(|| header.get("incident_id").and_then(Value::as_str))
            .map(str::to_string),
        exception_signal: body
            .get("exception")
            .and_then(|exception| exception.get("signal"))
            .and_then(Value::as_str)
            .map(str::to_string),
        termination_code: body
            .get("termination")
            .and_then(|termination| termination.get("code"))
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok()),
        fault_pc: thread_pc.or(exception_pc),
        fault_address,
        faulting_thread_index,
        faulting_thread,
    }))
}

fn value_at_path_u64(value: &Value, path: &[&str]) -> Option<u64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_u64()
}

fn resolve_jit_fault(replay_dir: &Path, fault_pc: u64) -> Result<Option<JitFaultResolution>> {
    let modules_dir = replay_dir.join("trust-ir-modules");
    let Ok(entries) = fs::read_dir(&modules_dir) else {
        return Ok(None);
    };
    let mut best: Option<JitFaultResolution> = None;
    for entry in entries {
        let entry = entry.with_context(|| format!("read entry in {}", modules_dir.display()))?;
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".metadata.json"))
        {
            continue;
        }
        let metadata_text =
            fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let metadata: Value = serde_json::from_str(&metadata_text)
            .with_context(|| format!("parse {}", path.display()))?;
        let Some(functions) = metadata
            .get("jit_pc_map")
            .and_then(|map| map.get("functions"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        for function in functions {
            let Some(runtime_start) = function.get("runtime_start").and_then(parse_hex_value_u64)
            else {
                continue;
            };
            let Some(code_len) = function.get("code_len").and_then(Value::as_u64) else {
                continue;
            };
            let Some(range_end) = runtime_start.checked_add(code_len) else {
                continue;
            };
            if fault_pc < runtime_start || fault_pc >= range_end {
                continue;
            }
            let pc_map_offset = fault_pc - runtime_start;
            let (nearest_block, next_block) = pc_map_blocks_around(function, pc_map_offset);
            let resolution = JitFaultResolution {
                metadata_path: path.clone(),
                module_name: metadata
                    .get("module_name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                stage: metadata
                    .get("stage")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                function_name: function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                runtime_start,
                code_len,
                symbol_offset: function
                    .get("symbol_offset")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                pc_map_offset,
                nearest_block,
                next_block,
                diagnostics: Vec::new(),
            };
            if best
                .as_ref()
                .is_none_or(|current| resolution.code_len < current.code_len)
            {
                best = Some(resolution);
            }
        }
    }
    Ok(best)
}

fn parse_hex_value_u64(value: &Value) -> Option<u64> {
    let text = value.as_str()?.trim();
    let text = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    u64::from_str_radix(text, 16).ok()
}

fn pc_map_blocks_around(function: &Value, pc_map_offset: u64) -> (Option<Value>, Option<Value>) {
    let mut blocks: Vec<(u64, Value)> = function
        .get("blocks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|block| {
            let offset = block.get("offset").and_then(parse_hex_value_u64)?;
            Some((offset, block.clone()))
        })
        .collect();
    blocks.sort_by_key(|(offset, _)| *offset);
    let mut nearest = None;
    let mut next = None;
    for (offset, block) in blocks {
        if offset <= pc_map_offset {
            nearest = Some(block);
        } else {
            next = Some(block);
            break;
        }
    }
    (nearest, next)
}

fn write_crash_packet(
    repo_root: &Path,
    spec_dir: &Path,
    result: &CommandResult,
    replay_dir: &Path,
    report: MacCrashReport,
    resolution: Option<JitFaultResolution>,
) -> Result<Option<CrashPacketSummary>> {
    let crash_dir = replay_dir.join("crash");
    fs::create_dir_all(&crash_dir).with_context(|| format!("create {}", crash_dir.display()))?;
    let signal_code = result.signal.or(report.termination_code);
    let packet_name = format!(
        "pid-{}-signal-{}-fault-{}.json",
        result.pid.unwrap_or_default(),
        signal_code.unwrap_or_default(),
        report
            .fault_pc
            .map(format_hex)
            .unwrap_or_else(|| "unknown".to_string())
    );
    let packet_path = crash_dir.join(packet_name);
    let artifact_root_display = repo_relative(repo_root, spec_dir);
    let replay_dir_display = repo_relative(repo_root, replay_dir);
    let packet_path_display = repo_relative(repo_root, &packet_path);
    let report_path_display = repo_relative(repo_root, &report.path);
    let fault_pc = report.fault_pc.map(format_hex);
    let fault_address = report.fault_address.map(format_hex);
    let fault_signal = crash_fault_signal(result.signal, report.exception_signal.as_deref());
    let mut diagnostics = Vec::new();
    if report.fault_pc.is_none() {
        diagnostics.push("missing_fault_pc".to_string());
    }
    if resolution.is_none() {
        diagnostics.push("missing_jit_pc_map_resolution_for_fault_pc".to_string());
    }
    if let Some(resolution) = &resolution {
        diagnostics.extend(resolution.diagnostics.clone());
    }

    let jit_json = resolution.as_ref().map(|resolution| {
        let range_end = resolution.runtime_start.saturating_add(resolution.code_len);
        json!({
            "matched": true,
            "metadata_path": repo_relative(repo_root, &resolution.metadata_path),
            "module_name": &resolution.module_name,
            "stage": &resolution.stage,
            "function": {
                "name": &resolution.function_name,
                "runtime_start": format_hex(resolution.runtime_start),
                "symbol_offset": &resolution.symbol_offset,
                "code_len": resolution.code_len,
                "range": {
                    "start": format_hex(resolution.runtime_start),
                    "end": format_hex(range_end),
                },
            },
            "pc_map_offset": format_hex(resolution.pc_map_offset),
            "pc_map_offset_decimal": resolution.pc_map_offset,
            "nearest_block": &resolution.nearest_block,
            "next_block": &resolution.next_block,
        })
    });

    let packet = json!({
        "schema": CRASH_PACKET_SCHEMA,
        "generated_at": chrono::Local::now().format("%Y-%m-%dT%H%M%S").to_string(),
        "artifact_root": &artifact_root_display,
        "replay_artifact_dir": &replay_dir_display,
        "packet_path": &packet_path_display,
        "status": {
            "non_promoting": true,
            "native_dispatch_promoted": false,
            "reason": "native_smoke_child_failed",
        },
        "child": {
            "argv": &result.argv,
            "cwd": &result.cwd,
            "pid": result.pid,
            "returncode": result.returncode,
            "signal": result.signal,
            "timed_out": result.timed_out,
            "started_unix_ms": result.started_unix_ms,
            "ended_unix_ms": result.ended_unix_ms,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "os_crash_report": {
            "path": &report_path_display,
            "header": &report.header,
            "incident": &report.incident,
            "capture_time": &report.capture_time,
            "pid": report.pid,
            "proc_name": &report.proc_name,
            "proc_path": report.body.get("procPath").cloned(),
            "parent_proc": report.body.get("parentProc").cloned(),
            "parent_pid": report.body.get("parentPid").cloned(),
            "fault_signal": &fault_signal,
            "fault_signal_code": signal_code,
            "fault_pc": fault_pc,
            "fault_pc_decimal": report.fault_pc,
            "fault_address": fault_address,
            "fault_address_decimal": report.fault_address,
            "faulting_thread": report.faulting_thread_index,
            "exception": report.body.get("exception").cloned(),
            "termination": report.body.get("termination").cloned(),
            "vm_region_info": report
                .body
                .get("vmRegionInfo")
                .or_else(|| report.body.get("vmregioninfo"))
                .cloned(),
            "thread": &report.faulting_thread,
        },
        "jit": jit_json.unwrap_or_else(|| json!({
            "matched": false,
            "replay_artifact_dir": &replay_dir_display,
        })),
        "diagnostics": &diagnostics,
    });
    let packet_json = serde_json::to_string_pretty(&packet).context("serialize crash packet")?;
    fs::write(&packet_path, packet_json + "\n")
        .with_context(|| format!("write {}", packet_path.display()))?;

    let summary = CrashPacketSummary {
        artifact_root: artifact_root_display,
        replay_artifact_dir: replay_dir_display,
        path: repo_relative(repo_root, &packet_path),
        os_crash_report: report_path_display,
        fault_signal,
        fault_signal_code: signal_code,
        fault_pc: report.fault_pc.map(format_hex),
        fault_pc_decimal: report.fault_pc,
        fault_address: report.fault_address.map(format_hex),
        fault_address_decimal: report.fault_address,
        jit_symbol: resolution
            .as_ref()
            .map(|resolution| resolution.function_name.clone()),
        jit_symbol_range: resolution
            .as_ref()
            .map(|resolution| CrashJitSymbolRangeSummary {
                start: format_hex(resolution.runtime_start),
                end: format_hex(resolution.runtime_start.saturating_add(resolution.code_len)),
                code_len: resolution.code_len,
            }),
        pc_map_offset: resolution
            .as_ref()
            .map(|resolution| format_hex(resolution.pc_map_offset)),
        pc_map_offset_decimal: resolution
            .as_ref()
            .map(|resolution| resolution.pc_map_offset),
        pc_map_block: resolution.as_ref().and_then(|resolution| {
            resolution
                .nearest_block
                .as_ref()
                .and_then(|block| block.get("block"))
                .and_then(Value::as_str)
                .map(str::to_string)
        }),
        metadata_path: resolution
            .as_ref()
            .map(|resolution| repo_relative(repo_root, &resolution.metadata_path)),
        non_promoting: true,
        status_reason: "native_smoke_child_failed".to_string(),
        diagnostics,
    };
    Ok(Some(summary))
}

fn crash_fault_signal(signal: Option<i32>, report_signal: Option<&str>) -> Option<String> {
    report_signal
        .map(str::to_string)
        .or_else(|| signal.and_then(unix_signal_name).map(str::to_string))
}

fn unix_signal_name(signal: i32) -> Option<&'static str> {
    match signal {
        4 => Some("SIGILL"),
        6 => Some("SIGABRT"),
        8 => Some("SIGFPE"),
        10 => Some("SIGBUS"),
        11 => Some("SIGSEGV"),
        _ => None,
    }
}

fn format_hex(value: u64) -> String {
    format!("0x{value:x}")
}

fn write_artifacts(output_dir: &Path, result: &CommandResult) -> Result<()> {
    fs::write(output_dir.join("stdout.txt"), &result.stdout)
        .with_context(|| format!("write {}", output_dir.join("stdout.txt").display()))?;
    fs::write(output_dir.join("stderr.txt"), &result.stderr)
        .with_context(|| format!("write {}", output_dir.join("stderr.txt").display()))?;
    fs::write(
        output_dir.join("command.json"),
        serde_json::to_string_pretty(result).context("serialize smoke command")? + "\n",
    )
    .with_context(|| format!("write {}", output_dir.join("command.json").display()))?;
    Ok(())
}

fn smoke_run_from_failure(
    spec: &SmokeSpec,
    spec_dir: &Path,
    repo_root: &Path,
    result: CommandResult,
    failures: Vec<String>,
) -> SmokeRun {
    smoke_run_from_result(spec, spec_dir, repo_root, result, failures)
}

fn smoke_run_from_result(
    spec: &SmokeSpec,
    spec_dir: &Path,
    repo_root: &Path,
    result: CommandResult,
    failures: Vec<String>,
) -> SmokeRun {
    let combined = format!("{}\n{}", result.stdout, result.stderr);
    let backend_segment = latest_flat_primary_backend_segment(&combined);
    let evidence_segment = latest_native_fused_evidence_segment(&backend_segment);
    let loop_start = compiled_bfs_loop_start(&evidence_segment);
    let completion = compiled_bfs_completion(&evidence_segment);
    let ok = result.returncode == 0 && failures.is_empty();
    SmokeRun {
        spec: spec.name.to_string(),
        artifact_dir: repo_relative(repo_root, spec_dir),
        pid: result.pid,
        returncode: result.returncode,
        signal: result.signal,
        timed_out: result.timed_out,
        elapsed_seconds: result.elapsed_seconds,
        crash_packet: result.crash_packet,
        failures,
        command: result.argv,
        cwd: result.cwd,
        env_overrides: result.env_overrides,
        stderr_tail: evidence_tail(&result.stderr),
        compiled_bfs_level_loop_initial_states: loop_start.map(|value| value.0),
        compiled_bfs_level_loop_fused: loop_start.map(|value| value.1),
        compiled_bfs_levels_completed: completion.map(|value| value.0),
        compiled_bfs_parents_processed: completion.map(|value| value.1),
        compiled_bfs_successors_generated: completion.map(|value| value.2),
        compiled_bfs_successors_new: completion.map(|value| value.3),
        compiled_bfs_total_states: completion.map(|value| value.4),
        ok,
    }
}

fn validate_output(spec: &SmokeSpec, stdout: &str, stderr: &str) -> Vec<String> {
    let combined = format!("{stdout}\n{stderr}");
    let backend_segment = latest_flat_primary_backend_segment(&combined);
    let evidence_segment = latest_native_fused_evidence_segment(&backend_segment);
    let final_summary = final_summary_segment(&combined);
    let mut failures = Vec::new();
    let state_constrained = spec.native_fused_state_constraint_count.unwrap_or(0) > 0;
    let state_constrained_proven = proves_state_constrained_native_fused(spec, &evidence_segment);

    for needle in required_needles(spec, state_constrained) {
        if requires_final_summary_result_line(&needle) {
            if Some(needle.trim()) != final_summary_result_line(&final_summary).as_deref() {
                failures.push(format!("missing required substring: {needle:?}"));
            }
            continue;
        }
        if let Some(label) = required_final_summary_count_label(&needle) {
            if Some(needle.trim()) != final_summary_count_line(&final_summary, label).as_deref() {
                failures.push(format!("missing required substring: {needle:?}"));
            }
            continue;
        }
        let haystack =
            required_substring_haystack(&needle, &combined, &backend_segment, &evidence_segment);
        if !haystack.contains(&needle) {
            failures.push(format!("missing required substring: {needle:?}"));
        }
    }

    validate_native_fused_layout(spec, &backend_segment, &evidence_segment, &mut failures);
    validate_compiled_loop(&evidence_segment, &final_summary, &mut failures);
    failures.extend(forbidden_substring_failures(
        &backend_segment,
        state_constrained_proven,
    ));
    for needle in COMMON_FORBIDDEN_FULL_OUTPUT_SUBSTRINGS {
        if combined.contains(needle) {
            failures.push(format!("found forbidden substring: {needle:?}"));
        }
    }
    validate_selftest(spec, &evidence_segment, &mut failures);
    validate_level_counts(spec, &evidence_segment, &mut failures);
    validate_state_constraint_skip(spec, &evidence_segment, state_constrained, &mut failures);
    validate_exact_telemetry(spec, &evidence_segment, state_constrained, &mut failures);
    validate_forbidden_patterns(&backend_segment, &evidence_segment, &mut failures);
    validate_transition_floor(spec, &final_summary, &mut failures);
    validate_flat_frontier(&evidence_segment, &mut failures);
    failures
}

fn required_needles(spec: &SmokeSpec, state_constrained: bool) -> Vec<String> {
    let mut needles = Vec::new();
    needles.extend(
        COMMON_REQUIRED_SUBSTRINGS
            .iter()
            .copied()
            .filter(|needle| !state_constrained || *needle != "[trust-cg] CompiledBfsStep built:")
            .map(str::to_string),
    );
    needles.push(format!(
        "CompiledBfsLevel built ({})",
        spec.native_fused_loop_label
    ));
    needles.push(format!(
        "trust_cg_native_fused_mode={}",
        spec.native_fused_mode
    ));
    needles.extend(
        spec.required_substrings
            .iter()
            .map(|needle| (*needle).to_string()),
    );
    needles
}

fn required_substring_haystack<'a>(
    needle: &str,
    combined: &'a str,
    backend_segment: &'a str,
    rebuild_segment: &'a str,
) -> &'a str {
    if needle.starts_with("[trust-cg] trust-codegen native compilation enabled") {
        combined
    } else if needle.starts_with("[trust_cg-selftest]")
        || needle.starts_with("[compiled-bfs]")
        || needle.starts_with("compiled_bfs_")
        || needle.starts_with("flat_bfs_")
        || needle.starts_with("trust_cg_bfs_level_")
        || needle.starts_with("trust_cg_native_fused_")
        || needle.starts_with("[trust-cg]")
    {
        rebuild_segment
    } else if needle.starts_with("flat_state_primary") {
        backend_segment
    } else {
        combined
    }
}

fn validate_compiled_loop(rebuild_segment: &str, final_summary: &str, failures: &mut Vec<String>) {
    let final_transitions = final_summary_count(final_summary, "Transitions");
    let final_states = final_summary_count(final_summary, "States found");
    match compiled_bfs_loop_start(rebuild_segment) {
        Some((initial_states, fused)) => {
            if initial_states == 0 {
                failures.push(format!(
                    "compiled BFS level loop started without initial work (initial_states={initial_states})"
                ));
            }
            if !fused {
                failures.push("compiled BFS level loop start was not fused".to_string());
            }
        }
        None => failures.push(
            "missing required substring: '[compiled-bfs] starting compiled BFS level loop'"
                .to_string(),
        ),
    }
    match compiled_bfs_completion(rebuild_segment) {
        Some((levels, parents, generated, new, total_states)) => {
            if levels == 0 || parents == 0 || generated == 0 || total_states == 0 {
                failures.push(format!(
                    "compiled BFS loop completed without positive native-fused work \
                     (levels={levels}, parents={parents}, generated={generated}, total_states={total_states})"
                ));
            }
            if new == 0 {
                failures.push(format!(
                    "compiled BFS loop completed with zero new states (new={new}, total_states={total_states})"
                ));
            }
            if final_states.is_some_and(|states| states != total_states) {
                failures.push(format!(
                    "compiled BFS total_states did not match final States found \
                     (total_states={total_states}, states_found={})",
                    final_states.unwrap_or_default()
                ));
            }
            if final_transitions.is_some_and(|transitions| transitions != generated) {
                failures.push(format!(
                    "compiled BFS generated successors did not match final Transitions \
                     (generated={generated}, transitions={})",
                    final_transitions.unwrap_or_default()
                ));
            }
        }
        None => {
            failures.push("missing required substring: '[compiled-bfs] completed:'".to_string())
        }
    }
    let (nanos, seconds) = compiled_bfs_execution_timing(rebuild_segment);
    if !seconds.is_some_and(|seconds| seconds > 0.0) && !nanos.is_some_and(|nanos| nanos > 0) {
        failures.push(format!(
            "compiled BFS execution timing telemetry was missing or non-positive \
             (compiled_bfs_execution_seconds={seconds:?}, compiled_bfs_execution_nanos={nanos:?})"
        ));
    }
}

fn validate_selftest(spec: &SmokeSpec, rebuild_segment: &str, failures: &mut Vec<String>) {
    for (description, pattern) in strict_selftest_marker_patterns(spec) {
        if !pattern.is_match(rebuild_segment) {
            failures.push(format!(
                "missing strict native callout selftest marker: {description}"
            ));
        }
    }
    let missing_expected = Regex::new(r"\bmissing_expected=(\d[\d,]*)\b").unwrap();
    for captures in missing_expected.captures_iter(rebuild_segment) {
        let missing = parse_usize(&captures[1]).unwrap_or_default();
        if missing > 0 {
            failures.push(format!(
                "native fused callout selftest reported missing expected callouts: missing_expected={missing}"
            ));
        }
    }
    failures.extend(strict_selftest_false_result_failures(rebuild_segment));
}

fn validate_level_counts(spec: &SmokeSpec, rebuild_segment: &str, failures: &mut Vec<String>) {
    match compiled_bfs_level_built_counts(rebuild_segment, spec.native_fused_loop_label) {
        Some((actions, invariants, state_len)) => {
            if actions != spec.native_fused_action_count
                || invariants != spec.native_fused_invariant_count
            {
                failures.push(format!(
                    "CompiledBfsLevel built counts did not match expected spec counts: \
                     expected actions={}, invariants={}; observed actions={actions}, invariants={invariants}",
                    spec.native_fused_action_count, spec.native_fused_invariant_count
                ));
            }
            if spec.expected_native_state_len.is_some()
                && state_len != spec.expected_native_state_len
            {
                failures.push(format!(
                    "CompiledBfsLevel built state_len did not match expected spec state length: \
                     expected state_len={}; observed state_len={state_len:?}",
                    spec.expected_native_state_len.unwrap_or_default()
                ));
            }
        }
        None => failures.push(format!(
            "missing exact CompiledBfsLevel built counts: actions={}, invariants={}",
            spec.native_fused_action_count, spec.native_fused_invariant_count
        )),
    }
}

fn validate_state_constraint_skip(
    spec: &SmokeSpec,
    rebuild_segment: &str,
    state_constrained: bool,
    failures: &mut Vec<String>,
) {
    if !state_constrained {
        return;
    }
    let mut saw_expected_skip = false;
    for line in rebuild_segment.lines() {
        if is_allowed_state_constraint_step_skip(line) {
            saw_expected_skip = true;
        }
        if line.contains("CompiledBfsStep not eligible")
            && !is_allowed_state_constraint_step_skip(line)
        {
            failures.push(format!(
                "found non-state-constraint CompiledBfsStep ineligibility: {line:?}"
            ));
        }
        if line.contains("state constraints require native fused constraint pruning")
            && !is_allowed_state_constraint_step_skip(line)
        {
            failures.push(format!(
                "found state-constraint pruning text outside the allowed CompiledBfsStep diagnostic: {line:?}"
            ));
        }
    }
    if !saw_expected_skip {
        failures.push(format!(
            "missing CompiledBfsStep skip diagnostic; expected one of: {:?} or {:?}",
            "[trust-cg] CompiledBfsStep not eligible: state constraints require native fused constraint pruning",
            "[trust-cg] CompiledBfsStep skipped: native fused level is the only admissible compiled BFS path for this run"
        ));
    }
    if !proves_state_constrained_native_fused(spec, rebuild_segment) {
        failures.push("missing state-constrained native-fused proof telemetry".to_string());
    }
}

fn validate_exact_telemetry(
    spec: &SmokeSpec,
    rebuild_segment: &str,
    state_constrained: bool,
    failures: &mut Vec<String>,
) {
    let mut expected = vec![
        (
            "trust_cg_native_fused_mode",
            spec.native_fused_mode.to_string(),
        ),
        (
            "trust_cg_native_fused_invariant_count",
            spec.native_fused_invariant_count.to_string(),
        ),
        ("trust_cg_native_fused_local_dedup", "true".to_string()),
        (
            "compiled_bfs_step_active",
            if state_constrained { "false" } else { "true" }.to_string(),
        ),
        ("compiled_bfs_level_active", "true".to_string()),
    ];
    if let Some(count) = spec.native_fused_state_constraint_count {
        expected.push((
            "trust_cg_native_fused_state_constraint_count",
            count.to_string(),
        ));
    }
    for (key, value) in expected {
        if !has_exact_telemetry(rebuild_segment, key, &value) {
            failures.push(format!("missing exact telemetry: {key}={value}"));
        }
    }
    if has_exact_telemetry(
        rebuild_segment,
        "trust_cg_native_fused_local_dedup",
        "false",
    ) {
        failures.push(
            "native fused telemetry reported local dedup disabled; smoke requires policy local dedup: \
             trust_cg_native_fused_local_dedup=true"
                .to_string(),
        );
    }
    if rebuild_segment
        .lines()
        .any(|line| line.contains("[trust-cg-native-bfs]") && line.contains("local_dedup=false"))
    {
        failures.push(
            "native BFS trace reported local dedup disabled; smoke requires policy local dedup: local_dedup=true"
                .to_string(),
        );
    }
}

fn validate_forbidden_patterns(
    backend_segment: &str,
    rebuild_segment: &str,
    failures: &mut Vec<String>,
) {
    let patterns = [
        Regex::new(r"\b[1-9]\d*\s+fallback\b").unwrap(),
        Regex::new(
            r"(?i)\bcompiled[- ]bfs\b.*\b(fallback|falling back|error|failed|disabled|disabling|not enabled|became unavailable|interpreter path used)\b",
        )
        .unwrap(),
        Regex::new(r"(?i)\bnative[- ]fused\b.*\bfallback\b").unwrap(),
        Regex::new(r"(?i)\bper[- ]parent\b.*\bfallback\b").unwrap(),
        Regex::new(r"(?i)\brequested interpreter fallback\b").unwrap(),
    ];
    for pattern in patterns {
        if let Some(matched) = pattern.find(backend_segment) {
            failures.push(format!(
                "matched forbidden pattern {:?}: {:?}",
                pattern.as_str(),
                matched.as_str()
            ));
        }
    }
    let level_blocker = Regex::new(
        r"(?i)\b(?:CompiledBfsLevel\b.*\b(requested interpreter fallback|falling back|not eligible|skipped|fallback)\b|(requested interpreter fallback|falling back|not eligible|skipped|fallback)\b.*\bCompiledBfsLevel\b)",
    )
    .unwrap();
    for line in rebuild_segment.lines() {
        if let Some(captures) = level_blocker.captures(line) {
            let blocker = captures
                .get(1)
                .or_else(|| captures.get(2))
                .map(|m| m.as_str())
                .unwrap_or("blocker");
            failures.push(format!(
                "found post-rebuild CompiledBfsLevel blocker: {blocker:?}: {line:?}"
            ));
        }
    }
    let zero_work = Regex::new(
        r"(?i)\[compiled-bfs\]\s+completed:\s+0\s+levels,\s+0\s+parents,\s+0\s+generated,\s+0\s+new\b",
    )
    .unwrap();
    if let Some(matched) = zero_work.find(rebuild_segment) {
        failures.push(format!(
            "compiled BFS loop completed without processing work: {:?}",
            matched.as_str()
        ));
    }
}

fn validate_transition_floor(spec: &SmokeSpec, final_summary: &str, failures: &mut Vec<String>) {
    if spec.min_transitions == 0 {
        return;
    }
    match final_summary_count(final_summary, "Transitions") {
        Some(transitions) if transitions >= spec.min_transitions => {}
        Some(transitions) => failures.push(format!(
            "transition count did not exercise work: expected at least {}, observed {transitions}",
            spec.min_transitions
        )),
        None => failures.push(format!(
            "missing transition count telemetry: expected at least {}",
            spec.min_transitions
        )),
    }
}

fn validate_native_fused_layout(
    spec: &SmokeSpec,
    backend_segment: &str,
    evidence_segment: &str,
    failures: &mut Vec<String>,
) {
    let Some(flat_state_line) = latest_flat_state_line(backend_segment) else {
        failures.push("missing current flat-state layout telemetry line".to_string());
        return;
    };
    if has_exact_telemetry(flat_state_line, "flat_state_primary", "true") {
        return;
    }

    let fused_loop = compiled_bfs_loop_start(evidence_segment).is_some_and(|(_, fused)| fused);
    let Some(flat_frontier_line) = latest_flat_frontier_line(evidence_segment) else {
        failures.push("missing current flat-frontier telemetry line".to_string());
        return;
    };
    let flat_frontier = parse_flat_frontier_line(flat_frontier_line);
    let Some(admission_line) = latest_native_fused_flat_frontier_admission_line(evidence_segment)
    else {
        failures.push(
            "missing current native-fused flat-frontier admission telemetry line".to_string(),
        );
        return;
    };
    let expected_bytes_per_state = spec.expected_native_state_len.map(|slots| slots * 8);
    let bytes_match = expected_bytes_per_state
        .zip(flat_frontier.bytes_per_state)
        .is_some_and(|(expected, observed)| expected == observed);

    if !(has_exact_telemetry(flat_state_line, "flat_state_primary", "false")
        && has_exact_telemetry(flat_state_line, "roundtrip_ok", "true")
        && has_exact_telemetry(flat_state_line, "fully_flat", "true")
        && has_exact_telemetry(flat_state_line, "flat_bfs", "true")
        && has_exact_telemetry(flat_state_line, "full_state_storage", "false")
        && has_exact_telemetry(flat_state_line, "view", "false")
        && has_exact_telemetry(flat_state_line, "symmetry", "false")
        && has_exact_telemetry(
            evidence_segment,
            "trust_cg_native_fused_level_active",
            "true",
        )
        && has_exact_telemetry(
            evidence_segment,
            "trust_cg_bfs_level_loop_kind",
            "native_fused_trust_cg_parent_loop",
        )
        && has_exact_telemetry(
            admission_line,
            "trust_cg_native_fused_flat_frontier_admission_active",
            "true",
        )
        && has_exact_telemetry(
            admission_line,
            "compiled_bfs_flat_frontier_admitted",
            "true",
        )
        && fused_loop
        && flat_frontier.active == Some(true)
        && flat_frontier.fallbacks == Some(0)
        && bytes_match)
    {
        failures.push(
            "native fused smoke requires flat_state_primary=true or a fully-flat flat-BFS \
             frontier with same-line layout proof, native-fused level execution, zero fallback, \
             and bytes/state matching the fused native state width"
                .to_string(),
        );
    }
}

fn validate_flat_frontier(rebuild_segment: &str, failures: &mut Vec<String>) {
    let Some(line) = latest_flat_frontier_line(rebuild_segment) else {
        failures.push("missing flat-frontier active telemetry line".to_string());
        return;
    };
    let telemetry = parse_flat_frontier_line(line);
    if telemetry.active != Some(true) {
        failures.push("latest flat-frontier telemetry line did not report active=true".to_string());
    }
    if telemetry.fallbacks != Some(0) {
        failures
            .push("latest flat-frontier telemetry line did not report zero fallback".to_string());
    }
}

fn latest_flat_state_line(text: &str) -> Option<&str> {
    text.lines()
        .rev()
        .find(|line| line.contains("[flat_state]") && line.contains("flat_state_primary="))
}

fn latest_flat_frontier_line(text: &str) -> Option<&str> {
    text.lines()
        .rev()
        .find(|line| line.contains("[flat-frontier]") && line.contains("flat_bfs_frontier_active="))
}

fn latest_native_fused_flat_frontier_admission_line(text: &str) -> Option<&str> {
    text.lines().rev().find(|line| {
        line.contains("trust_cg_native_fused_flat_frontier_admission_active=")
            || line.contains("compiled_bfs_flat_frontier_admitted=")
    })
}

#[derive(Debug, Default, PartialEq, Eq)]
struct FlatFrontierTelemetry {
    active: Option<bool>,
    fallbacks: Option<usize>,
    bytes_per_state: Option<usize>,
}

fn parse_flat_frontier_line(line: &str) -> FlatFrontierTelemetry {
    let active = if line.contains("flat_bfs_frontier_active=true") {
        Some(true)
    } else if line.contains("flat_bfs_frontier_active=false") {
        Some(false)
    } else {
        None
    };
    let fallbacks = Regex::new(r"\b(\d[\d,]*)\s+fallback\b")
        .unwrap()
        .captures(line)
        .and_then(|captures| parse_usize(&captures[1]));
    let bytes_per_state = Regex::new(r"\b(\d[\d,]*)\s+bytes/state\b")
        .unwrap()
        .captures(line)
        .and_then(|captures| parse_usize(&captures[1]));
    FlatFrontierTelemetry {
        active,
        fallbacks,
        bytes_per_state,
    }
}

fn forbidden_substring_failures(
    backend_segment: &str,
    state_constrained_native_fused_proven: bool,
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut in_rebuild_segment = !backend_segment.contains(FLAT_PRIMARY_REBUILD_MARKER);
    for line in backend_segment.lines() {
        if line.contains(FLAT_PRIMARY_REBUILD_MARKER) {
            in_rebuild_segment = true;
        }
        for needle in COMMON_FORBIDDEN_SUBSTRINGS {
            if !line.contains(needle) {
                continue;
            }
            if is_allowed_state_constraint_forbidden_line(
                line,
                needle,
                in_rebuild_segment,
                state_constrained_native_fused_proven,
            ) {
                continue;
            }
            failures.push(format!("found forbidden substring: {needle:?}: {line:?}"));
        }
    }
    failures
}

fn is_allowed_state_constraint_forbidden_line(
    line: &str,
    needle: &str,
    in_rebuild_segment: bool,
    state_constrained_native_fused_proven: bool,
) -> bool {
    if !in_rebuild_segment || !state_constrained_native_fused_proven {
        return false;
    }
    match needle {
        "CompiledBfsStep not eligible"
        | "state constraints require native fused constraint pruning" => {
            is_allowed_state_constraint_step_skip(line)
        }
        "state constraints require trust-codegen native fused constraint support" => {
            is_allowed_state_constraint_fused_level_skip(line)
        }
        "compiled_bfs_step_active=false" => {
            has_exact_telemetry(line, "compiled_bfs_step_active", "false")
        }
        _ => false,
    }
}

fn strict_selftest_marker_patterns(spec: &SmokeSpec) -> Vec<(&'static str, Regex)> {
    let state_constraint_count = spec.native_fused_state_constraint_count.unwrap_or(0);
    let state_len = spec
        .expected_native_state_len
        .map(|value| value.to_string())
        .unwrap_or_else(|| r"\d+".to_string());
    vec![
        (
            "prepared native fused callout selftest",
            Regex::new(&format!(
                r"\[trust_cg-selftest\]\s+prepared native fused callout selftest:\s+actions={},\s+state_constraints={},\s+invariants={},\s+missing_expected=0,\s+fail_closed=true\b",
                spec.native_fused_action_count,
                state_constraint_count,
                spec.native_fused_invariant_count
            ))
            .unwrap(),
        ),
        (
            "running native fused callout selftest on first real parent",
            Regex::new(&format!(
                r"\[trust_cg-selftest\]\s+running native fused callout selftest on first real parent:\s+state_len={},\s+actions={},\s+state_constraints={},\s+invariants={},\s+fail_closed=true\b",
                state_len,
                spec.native_fused_action_count,
                state_constraint_count,
                spec.native_fused_invariant_count
            ))
            .unwrap(),
        ),
        (
            "native fused callout selftest complete",
            Regex::new(r"\[trust_cg-selftest\]\s+native fused callout selftest complete\b").unwrap(),
        ),
    ]
}

fn strict_selftest_false_result_failures(text: &str) -> Vec<String> {
    let mut failures = Vec::new();
    for line in text.lines() {
        let Some((kind, status, value)) = parse_selftest_callout_result(line) else {
            continue;
        };
        if !matches!(kind.as_str(), "invariant" | "state_constraint") {
            continue;
        }
        if status != "Ok" || value == 0 {
            failures.push(format!(
                "native fused callout selftest reported failed strict check: \
                 kind={kind} status={status} value={value} line={line:?}"
            ));
        }
    }
    failures
}

fn parse_selftest_callout_result(line: &str) -> Option<(String, String, i64)> {
    if !line.contains("[trust_cg-selftest]")
        || !line.contains("status=")
        || !line.contains("value=")
    {
        return None;
    }
    let kv = Regex::new(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=([^,\s]+)").unwrap();
    let mut kind = None;
    let mut status = None;
    let mut value = None;
    for captures in kv.captures_iter(line) {
        match &captures[1] {
            "kind" => kind = Some(captures[2].trim_end_matches(',').to_string()),
            "status" => status = Some(captures[2].trim_end_matches(',').to_string()),
            "value" => value = captures[2].trim_end_matches(',').parse::<i64>().ok(),
            _ => {}
        }
    }
    if kind.is_none() {
        let leading =
            Regex::new(r"^\s*\[trust_cg-selftest\]\s+([A-Za-z_][A-Za-z0-9_]*)\s+callout\b")
                .unwrap();
        if let Some(captures) = leading.captures(line) {
            kind = Some(captures[1].to_string());
        }
    }
    Some((kind?, status?, value?))
}

fn latest_flat_primary_backend_segment(combined: &str) -> String {
    let lines: Vec<&str> = combined.lines().collect();
    let Some(marker_line_index) = lines
        .iter()
        .rposition(|line| line.contains(FLAT_PRIMARY_REBUILD_MARKER))
    else {
        return combined.to_string();
    };
    for idx in (0..marker_line_index).rev() {
        if lines[idx].contains("flat_state_primary=true") {
            return lines[idx..].join("\n");
        }
    }
    lines[marker_line_index..].join("\n")
}

fn latest_flat_primary_rebuild_segment(backend_segment: &str) -> String {
    backend_segment
        .rfind(FLAT_PRIMARY_REBUILD_MARKER)
        .map(|idx| backend_segment[idx..].to_string())
        .unwrap_or_default()
}

fn latest_native_fused_evidence_segment(backend_segment: &str) -> String {
    let rebuild_segment = latest_flat_primary_rebuild_segment(backend_segment);
    if rebuild_segment.is_empty() {
        backend_segment.to_string()
    } else {
        rebuild_segment
    }
}

fn final_summary_segment(combined: &str) -> String {
    let result = Regex::new(r"(?m)^\s*Model checking (?:complete|stopped):").unwrap();
    result
        .find_iter(combined)
        .last()
        .map(|matched| combined[matched.start()..].to_string())
        .unwrap_or_default()
}

fn requires_final_summary_result_line(needle: &str) -> bool {
    Regex::new(r"^Model checking (?:complete|stopped):.+$")
        .unwrap()
        .is_match(needle.trim())
}

fn final_summary_result_line(final_summary: &str) -> Option<String> {
    let result = Regex::new(r"^Model checking (?:complete|stopped):.+$").unwrap();
    final_summary
        .lines()
        .map(str::trim)
        .find(|line| result.is_match(line))
        .map(str::to_string)
}

fn required_final_summary_count_label(needle: &str) -> Option<&'static str> {
    for label in ["Max depth", "States found", "Initial states", "Transitions"] {
        let pattern = Regex::new(&format!(r"^{}:\s+\d[\d,]*$", regex::escape(label))).unwrap();
        if pattern.is_match(needle.trim()) {
            return Some(label);
        }
    }
    None
}

fn final_summary_count_line(final_summary: &str, label: &str) -> Option<String> {
    let prefix = format!("{label}:");
    final_summary
        .lines()
        .map(str::trim)
        .rfind(|line| line.starts_with(&prefix))
        .map(str::to_string)
}

fn final_summary_count(final_summary: &str, label: &str) -> Option<usize> {
    let line = final_summary_count_line(final_summary, label)?;
    let pattern = Regex::new(&format!(r"^{}:\s+(\d[\d,]*)$", regex::escape(label))).unwrap();
    let captures = pattern.captures(&line)?;
    parse_usize(&captures[1])
}

fn compiled_bfs_loop_start(combined: &str) -> Option<(usize, bool)> {
    let pattern = Regex::new(
        r"\[compiled-bfs\]\s+starting compiled BFS level loop \(([\d,]+) initial states in arena, fused=(true|false)\)",
    )
    .unwrap();
    let captures = pattern.captures_iter(combined).last()?;
    Some((parse_usize(&captures[1])?, &captures[2] == "true"))
}

fn compiled_bfs_completion(combined: &str) -> Option<(usize, usize, usize, usize, usize)> {
    let pattern = Regex::new(
        r"(?i)\[compiled-bfs\]\s+completed:\s+(\d[\d,]*)\s+levels,\s+(\d[\d,]*)\s+parents,\s+(\d[\d,]*)\s+generated,\s+(\d[\d,]*)\s+new,\s+(\d[\d,]*)\s+total states\b",
    )
    .unwrap();
    let captures = pattern.captures_iter(combined).last()?;
    Some((
        parse_usize(&captures[1])?,
        parse_usize(&captures[2])?,
        parse_usize(&captures[3])?,
        parse_usize(&captures[4])?,
        parse_usize(&captures[5])?,
    ))
}

fn compiled_bfs_execution_timing(combined: &str) -> (Option<usize>, Option<f64>) {
    let nanos_re = Regex::new(
        r"\b(?:compiled_bfs_execution_nanos|execution_time_ns|execution_time_nanos)\s*[:=]\s*(\d[\d,]*)\b",
    )
    .unwrap();
    let seconds_re = Regex::new(
        r"\b(?:compiled_bfs_execution_seconds|execution_time_seconds)\s*[:=]\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b",
    )
    .unwrap();
    let completion_re = Regex::new(r"(?i)\[compiled-bfs\]\s+completed:").unwrap();
    let mut nanos = None;
    let mut seconds = None;
    for line in combined.lines() {
        let timing_line =
            line.contains("[compiled-bfs]") && line.contains("compiled_bfs_execution_");
        if !timing_line && !completion_re.is_match(line) {
            continue;
        }
        for captures in nanos_re.captures_iter(line) {
            nanos = parse_usize(&captures[1]);
        }
        for captures in seconds_re.captures_iter(line) {
            seconds = captures[1].parse::<f64>().ok();
        }
    }
    (nanos, seconds)
}

fn compiled_bfs_level_built_counts(
    segment: &str,
    loop_label: &str,
) -> Option<(usize, usize, Option<usize>)> {
    let pattern = Regex::new(
        r"\[trust[_-]cg\]\s+CompiledBfsLevel built \(([^)]+)\):\s+(\d[\d,]*)\s+action instances,\s+(\d[\d,]*)\s+invariants(?:,\s+\d[\d,]*\s+state constraints)?(?:,\s+state_len=(\d[\d,]*))?\b",
    )
    .unwrap();
    let captures = pattern
        .captures_iter(segment)
        .filter(|captures| {
            captures
                .get(1)
                .is_some_and(|label| label.as_str() == loop_label)
        })
        .last()?;
    Some((
        parse_usize(&captures[2])?,
        parse_usize(&captures[3])?,
        captures
            .get(4)
            .and_then(|value| parse_usize(value.as_str())),
    ))
}

fn is_allowed_state_constraint_step_skip(line: &str) -> bool {
    let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
    let not_eligible = Regex::new(
        r"^\[trust[_-]cg\] CompiledBfsStep not eligible: state constraints require native fused constraint pruning(?: \(first state constraint: [^)]+\))?$",
    )
    .unwrap();
    let native_fused_only = Regex::new(
        r"^\[trust[_-]cg\] CompiledBfsStep skipped: native fused level is the only admissible compiled BFS path for this run$",
    )
    .unwrap();
    not_eligible.is_match(&normalized) || native_fused_only.is_match(&normalized)
}

fn is_allowed_state_constraint_fused_level_skip(line: &str) -> bool {
    let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
    Regex::new(
        r"(?i)^\[compiled-bfs\] fused level skipped: state constraints require trust-codegen native fused constraint support(?: \(first state constraint: [^)]+\))?$",
    )
    .unwrap()
    .is_match(&normalized)
}

fn proves_state_constrained_native_fused(spec: &SmokeSpec, segment: &str) -> bool {
    let Some(count) = spec.native_fused_state_constraint_count else {
        return false;
    };
    count > 0
        && spec.native_fused_mode == "state_constraint_checking"
        && spec
            .native_fused_loop_label
            .contains("state-constrained native fused")
        && segment.contains(&format!(
            "CompiledBfsLevel built ({})",
            spec.native_fused_loop_label
        ))
        && has_exact_telemetry(
            segment,
            "trust_cg_bfs_level_loop_kind",
            "native_fused_trust_cg_parent_loop",
        )
        && has_exact_telemetry(segment, "trust_cg_native_fused_level_active", "true")
        && has_exact_telemetry(
            segment,
            "trust_cg_native_fused_mode",
            "state_constraint_checking",
        )
        && has_exact_telemetry(
            segment,
            "trust_cg_native_fused_state_constraint_count",
            &count.to_string(),
        )
}

fn has_exact_telemetry(text: &str, key: &str, value: &str) -> bool {
    let expected = format!("{key}={value}");
    text.split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ';')
        .map(|token| token.trim_matches(|ch: char| matches!(ch, '(' | ')' | '[' | ']' | ':')))
        .any(|token| token == expected)
}

fn parse_usize(value: &str) -> Option<usize> {
    value.replace(',', "").parse::<usize>().ok()
}

fn evidence_tail(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect();
    if lines.len() > 24 {
        lines.drain(0..lines.len() - 24);
    }
    lines
}

fn repo_relative(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .map(Path::display)
        .map(|display| display.to_string())
        .unwrap_or_else(|_| path.display().to_string())
}

fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| {
            if arg.chars().all(|ch| {
                ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '/' | '.' | ':' | '=')
            }) {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_markdown(summary: &serde_json::Value) -> String {
    let mut lines = vec![
        "# trust-codegen Native-Fused Bounded Smoke".to_string(),
        String::new(),
        format!("**Timestamp:** {}", summary["timestamp"].as_str().unwrap_or("")),
        format!("**Binary:** `{}`", summary["binary"].as_str().unwrap_or("")),
        format!(
            "**Artifact bundle:** `{}`",
            summary["artifact_bundle"].as_str().unwrap_or("")
        ),
        String::new(),
        "## Backend Controls".to_string(),
        String::new(),
        "```json".to_string(),
        serde_json::to_string_pretty(&summary["env_overrides"]).unwrap_or_else(|_| "{}".to_string()),
        "```".to_string(),
        String::new(),
        "| Spec | Result | Seconds | Loop fused | Initial | Levels | Parents | Generated | Total states | Artifact | Crash packet | Failures |".to_string(),
        "|------|--------|---------|------------|---------|--------|---------|-----------|--------------|----------|--------------|----------|".to_string(),
    ];
    if let Some(runs) = summary["runs"].as_array() {
        for row in runs {
            let result = if row["ok"].as_bool().unwrap_or(false) {
                "PASS"
            } else {
                "FAIL"
            };
            let failures = row["failures"]
                .as_array()
                .map(|failures| {
                    failures
                        .iter()
                        .filter_map(|failure| failure.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_default();
            lines.push(format!(
                "| {} | {result} | {:.3} | {} | {} | {} | {} | {} | {} | `{}` | {} | {} |",
                row["spec"].as_str().unwrap_or(""),
                row["elapsed_seconds"].as_f64().unwrap_or(0.0),
                fmt_bool(row["compiled_bfs_level_loop_fused"].as_bool()),
                fmt_usize(row["compiled_bfs_level_loop_initial_states"].as_u64()),
                fmt_usize(row["compiled_bfs_levels_completed"].as_u64()),
                fmt_usize(row["compiled_bfs_parents_processed"].as_u64()),
                fmt_usize(row["compiled_bfs_successors_generated"].as_u64()),
                fmt_usize(row["compiled_bfs_total_states"].as_u64()),
                row["artifact_dir"].as_str().unwrap_or(""),
                fmt_crash_packet(row.get("crash_packet")),
                failures
            ));
        }
        lines.extend([String::new(), "## Commands".to_string(), String::new()]);
        for row in runs {
            let command = row["command"]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            lines.extend([
                format!("### {}", row["spec"].as_str().unwrap_or("")),
                String::new(),
                format!("`cwd: {}`", row["cwd"].as_str().unwrap_or("")),
                String::new(),
                "```sh".to_string(),
                shell_join(&command),
                "```".to_string(),
                String::new(),
                "```json".to_string(),
                serde_json::to_string_pretty(&row["env_overrides"])
                    .unwrap_or_else(|_| "{}".to_string()),
                "```".to_string(),
                String::new(),
            ]);
        }
    }
    lines.join("\n")
}

fn fmt_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "n/a",
    }
}

fn fmt_usize(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_crash_packet(value: Option<&Value>) -> String {
    value
        .and_then(|value| value.get("path"))
        .and_then(Value::as_str)
        .map(|path| format!("`{path}`"))
        .unwrap_or_else(|| "n/a".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_env_rejects_protected_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS".to_string(),
            "27".to_string(),
        );

        let error = smoke_env_with_overrides(&overrides).unwrap_err();

        assert!(error
            .to_string()
            .contains("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS=1"));
    }

    #[test]
    fn smoke_env_rejects_strict_selftest_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert(STRICT_SELFTEST_ENV.to_string(), "0".to_string());

        let error = smoke_env_with_overrides(&overrides).unwrap_err();

        assert!(error
            .to_string()
            .contains("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST=strict"));
    }

    #[test]
    fn smoke_env_requires_policy_local_dedup() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP".to_string(),
            "0".to_string(),
        );

        let error = smoke_env_with_overrides(&overrides).unwrap_err();

        assert!(error
            .to_string()
            .contains("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP=1"));
    }

    #[test]
    fn exact_telemetry_requires_policy_local_dedup() {
        let spec = smoke_spec(COFFEECAN_SAFETY_SPEC_NAME).unwrap();
        let segment = [
            "trust_cg_native_fused_mode=invariant_checking",
            "trust_cg_native_fused_invariant_count=1",
            "trust_cg_native_fused_local_dedup=false",
            "compiled_bfs_step_active=true",
            "compiled_bfs_level_active=true",
            "trust_cg_native_fused_state_constraint_count=0",
            "[trust-cg-native-bfs] generated=1 state_count=1 parents_processed=1 local_dedup=false",
        ]
        .join("\n");
        let mut failures = Vec::new();

        validate_exact_telemetry(&spec, &segment, false, &mut failures);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("trust_cg_native_fused_local_dedup=true")),
            "{failures:?}"
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("local_dedup=true")),
            "{failures:?}"
        );
    }

    #[test]
    fn latest_segment_starts_at_flat_primary_rebuild() {
        let text = format!(
            "old fallback\nflat_state_primary=true\n{FLAT_PRIMARY_REBUILD_MARKER}\ntrust_cg_native_fused_level_active=true"
        );

        let segment = latest_flat_primary_backend_segment(&text);

        assert!(!segment.contains("old fallback"));
        assert!(segment.contains("flat_state_primary=true"));
        assert!(segment.contains("trust_cg_native_fused_level_active=true"));
    }

    #[test]
    fn marker_absent_native_fused_evidence_uses_current_segment() {
        let backend = [
            "[flat_state] flat_state_primary=false: roundtrip_ok=true, fully_flat=true, flat_primary_safe=false, view=false, symmetry=false, flat_bfs=true, full_state_storage=false",
            "[compiled-bfs] starting compiled BFS level loop (192 initial states in arena, fused=true)",
            "[compiled-bfs] completed: 4 levels, 3064 parents, 22532 generated, 9212 new, 9404 total states, execution_time_ns=1, execution_time_seconds=0.001",
        ]
        .join("\n");

        let segment = latest_native_fused_evidence_segment(&backend);

        assert!(segment.contains("flat_state_primary=false"));
        assert!(segment.contains("[compiled-bfs] starting compiled BFS level loop"));
        assert_eq!(compiled_bfs_loop_start(&segment), Some((192, true)));
    }

    #[test]
    fn state_constrained_native_fused_without_flat_primary_marker_passes() {
        let spec = smoke_spec("EWD998Small").unwrap();
        let (stdout, stderr) = synthetic_ewd_native_fused_output(false, true);

        let failures = validate_output(&spec, &stdout, &stderr);

        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn stale_native_fused_evidence_before_rebuild_marker_is_rejected() {
        let spec = smoke_spec("EWD998Small").unwrap();
        let (stdout, mut stderr) = synthetic_ewd_native_fused_output(false, true);
        stderr.push('\n');
        stderr.push_str(FLAT_PRIMARY_REBUILD_MARKER);
        stderr.push('\n');
        stderr.push_str("[flat_state] flat_state_primary=true: roundtrip_ok=true, fully_flat=true, flat_bfs=true, full_state_storage=false\n");

        let failures = validate_output(&spec, &stdout, &stderr);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("[compiled-bfs] starting compiled BFS level loop")),
            "{failures:#?}"
        );
    }

    #[test]
    fn flat_primary_false_without_flat_frontier_telemetry_is_rejected() {
        let spec = smoke_spec("EWD998Small").unwrap();
        let (stdout, stderr) = synthetic_ewd_native_fused_output(false, false);

        let failures = validate_output(&spec, &stdout, &stderr);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("missing current flat-frontier telemetry line")),
            "{failures:#?}"
        );
    }

    #[test]
    fn flat_primary_false_requires_matching_frontier_width() {
        let spec = smoke_spec("EWD998Small").unwrap();
        let (stdout, mut stderr) = synthetic_ewd_native_fused_output(false, false);
        stderr.push_str(
            "\n[flat-frontier] flat_bfs_frontier_active=true: 9404 total pushed, 9404 flat \
             (100.0%), 0 fallback, 112 bytes/state",
        );

        let failures = validate_output(&spec, &stdout, &stderr);

        assert!(
            failures.iter().any(|failure| failure.contains(
                "native fused smoke requires flat_state_primary=true or a fully-flat flat-BFS"
            )),
            "{failures:#?}"
        );
    }

    #[test]
    fn flat_primary_false_uses_latest_native_fused_admission_line() {
        let spec = smoke_spec("EWD998Small").unwrap();
        let (stdout, mut stderr) = synthetic_ewd_native_fused_output(false, true);
        stderr.push_str(
            "\n[trust-cg] trust_cg_native_fused_flat_frontier_admission_active=false \
             compiled_bfs_flat_frontier_admitted=false",
        );

        let failures = validate_output(&spec, &stdout, &stderr);

        assert!(
            failures.iter().any(|failure| failure.contains(
                "native fused smoke requires flat_state_primary=true or a fully-flat flat-BFS"
            )),
            "{failures:#?}"
        );
    }

    #[test]
    fn flat_frontier_line_parser_extracts_active_fallbacks_and_width() {
        let parsed = parse_flat_frontier_line(
            "[flat-frontier] flat_bfs_frontier_active=true: 9,404 total pushed, 9,404 flat \
             (100.0%), 0 fallback, 120 bytes/state",
        );

        assert_eq!(
            parsed,
            FlatFrontierTelemetry {
                active: Some(true),
                fallbacks: Some(0),
                bytes_per_state: Some(120),
            }
        );
    }

    #[test]
    fn parses_loop_completion() {
        let completion = compiled_bfs_completion(
            "[compiled-bfs] completed: 3 levels, 12 parents, 2,997 generated, 1,000 new, 2,001 total states",
        )
        .unwrap();

        assert_eq!(completion, (3, 12, 2997, 1000, 2001));
    }

    fn synthetic_ewd_native_fused_output(
        flat_primary: bool,
        include_flat_frontier: bool,
    ) -> (String, String) {
        let stdout = [
            "Model checking: /tmp/EWD998.tla",
            "Search completeness: bounded (max_depth=3)",
            "Model checking stopped: depth limit reached.",
            "Statistics:",
            "  States found: 9404",
            "  Initial states: 192",
            "  Transitions: 22532",
            "  Max depth: 3",
        ]
        .join("\n");
        let mut stderr = vec![
            "[trust-cg] trust-codegen native compilation enabled (default engine under AUTO selection; opt out via --backend interpreter)".to_string(),
            format!(
                "[flat_state] flat_state_primary={}: roundtrip_ok=true, fully_flat=true, flat_primary_safe={}, view=false, symmetry=false, flat_bfs=true, full_state_storage=false",
                if flat_primary { "true" } else { "false" },
                if flat_primary { "true" } else { "false" },
            ),
            "[trust-cg] executable action coverage: trust_cg_actions_compiled=15 trust_cg_actions_total=15".to_string(),
            "[trust-cg] compilation complete: 15/15 actions, 3/3 invariants, 1/1 state constraints compiled in 1ms".to_string(),
            "[trust-cg] CompiledBfsStep not eligible: state constraints require native fused constraint pruning (first state constraint: StateConstraint)".to_string(),
            "[trust_cg-selftest] prepared native fused callout selftest: actions=15, state_constraints=1, invariants=3, missing_expected=0, fail_closed=true".to_string(),
            "[trust-cg] CompiledBfsLevel built (state-constrained native fused Trust-CG parent loop): 15 action instances, 3 invariants, state_len=15".to_string(),
            "[trust-cg] trust_cg_bfs_level_active=true trust_cg_native_fused_level_active=true trust_cg_bfs_level_loop_kind=native_fused_trust_cg_parent_loop trust_cg_native_fused_mode=state_constraint_checking trust_cg_native_fused_state_constraint_count=1 trust_cg_native_fused_invariant_count=3 trust_cg_native_fused_regular_invariants_checked=true trust_cg_native_fused_local_dedup=true".to_string(),
            "[trust-cg] trust_cg_native_fused_flat_frontier_admission_active=true compiled_bfs_flat_frontier_admitted=true".to_string(),
            "[compiled-bfs] activating compiled BFS loop (auto-detected (all-scalar, fully JIT-compiled))".to_string(),
            "[trust_cg-selftest] running native fused callout selftest on first real parent: state_len=15, actions=15, state_constraints=1, invariants=3, fail_closed=true".to_string(),
            "[trust_cg-selftest] state_constraint callout index=0 symbol=trust_cg_state_constraint_0 name=StateConstraint status=Ok value=1".to_string(),
            "[trust_cg-selftest] invariant callout index=0 symbol=trust_cg_inv_0 name=TerminationDetection status=Ok value=1".to_string(),
            "[trust_cg-selftest] invariant callout index=1 symbol=trust_cg_inv_1 name=Inv status=Ok value=1".to_string(),
            "[trust_cg-selftest] invariant callout index=2 symbol=trust_cg_inv_2 name=TypeOK status=Ok value=1".to_string(),
            "[trust_cg-selftest] native fused callout selftest complete".to_string(),
            "[compiled-bfs] starting compiled BFS level loop (192 initial states in arena, fused=true)".to_string(),
            "[compiled-bfs] compiled_bfs_step_active=false compiled_bfs_level_active=true".to_string(),
            "[compiled-bfs] completed: 4 levels, 3064 parents, 22532 generated, 9212 new, 9404 total states, execution_time_ns=1, execution_time_seconds=0.001".to_string(),
        ];
        if include_flat_frontier {
            stderr.push("[flat-frontier] flat_bfs_frontier_active=true: 9404 total pushed, 9404 flat (100.0%), 0 fallback, 120 bytes/state".to_string());
        }
        (stdout, stderr.join("\n"))
    }

    #[test]
    fn check_command_forces_trust_cg_backend() {
        let spec = smoke_spec("MCLamportMutex").unwrap();
        let command = build_check_command(
            Path::new("target/user/release/ty"),
            &spec,
            Path::new("/tmp/examples"),
        );

        assert!(command
            .windows(2)
            .any(|window| window[0] == "--backend" && window[1] == "trust-cg"));
    }

    #[test]
    fn resolve_binary_honors_target_dir_and_profile() {
        let temp = tempfile::tempdir().unwrap();
        let target_dir = temp.path().join("custom-target");
        let binary_name = if cfg!(windows) { "ty.exe" } else { "ty" };
        let binary = target_dir.join("release-canary").join(binary_name);
        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::write(&binary, "").unwrap();

        let resolved = resolve_binary(
            temp.path(),
            None,
            Some(Path::new("custom-target")),
            "release-canary",
        )
        .unwrap();

        assert_eq!(resolved, binary);
    }

    #[test]
    fn resolve_binary_maps_dev_profile_to_debug_dir() {
        let temp = tempfile::tempdir().unwrap();
        let target_dir = temp.path().join("custom-target");
        let binary_name = if cfg!(windows) { "ty.exe" } else { "ty" };
        let binary = target_dir.join("debug").join(binary_name);
        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::write(&binary, "").unwrap();

        let resolved =
            resolve_binary(temp.path(), None, Some(Path::new("custom-target")), "dev").unwrap();

        assert_eq!(resolved, binary);
        assert_eq!(profile_binary_dir("release-canary"), "release-canary");
        assert_eq!(profile_binary_dir("dev"), "debug");
    }

    #[test]
    fn strict_selftest_non_ok_result_fails() {
        let failures = strict_selftest_false_result_failures(
            "[trust_cg-selftest] invariant callout kind=invariant status=Err value=1",
        );

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("status=Err"));
    }

    #[test]
    fn crash_packet_resolves_macos_ips_fault_pc_to_jit_pc_map_offset() {
        let temp = tempfile::tempdir().unwrap();
        let repo_root = temp.path();
        let spec_dir = repo_root.join("reports/run/MCLamportMutex");
        let replay_dir = spec_dir.join("trust_cg-replay");
        let modules_dir = replay_dir.join("trust-ir-modules");
        let diagnostic_dir = repo_root.join("DiagnosticReports");
        fs::create_dir_all(&modules_dir).unwrap();
        fs::create_dir_all(&diagnostic_dir).unwrap();

        fs::write(
            modules_dir.join("000056-compile_module_native.linked-Request__1_1.metadata.json"),
            serde_json::to_string_pretty(&json!({
                "schema": "ty.trust_cg.native_replay_trust_ir.v1",
                "stage": "compile_module_native.linked",
                "module_name": "Request__1_1",
                "jit_pc_map": {
                    "available": true,
                    "functions": [{
                        "name": "Request__1_1",
                        "runtime_start": "0x1036fc000",
                        "symbol_offset": "0x0",
                        "code_len": 64,
                        "blocks": [
                            {"block": "bb0", "offset": "0x0", "pc": "0x1036fc000"},
                            {"block": "bb1", "offset": "0x20", "pc": "0x1036fc020"}
                        ]
                    }]
                }
            }))
            .unwrap()
                + "\n",
        )
        .unwrap();

        let fault_pc = 0x1036fc000_u64;
        let ips_header = json!({
            "app_name": "ty",
            "timestamp": "2026-04-27 20:22:53.00 -0700",
            "incident_id": "incident-123",
            "name": "ty",
        });
        let ips_body = json!({
            "pid": 4242,
            "procName": "ty",
            "procPath": "/USERDIR/*/ty",
            "parentProc": "ty",
            "parentPid": 4241,
            "captureTime": "2026-04-27 20:22:53.2982 -0700",
            "incident": "incident-123",
            "vmRegionInfo": "0x1036fc000 is in 0x1036fc000-0x103708000",
            "exception": {
                "type": "EXC_BAD_ACCESS",
                "signal": "SIGBUS",
                "subtype": "KERN_PROTECTION_FAILURE at 0x00000001036fc000",
                "rawCodes": [2, fault_pc],
            },
            "termination": {
                "namespace": "SIGNAL",
                "code": 10,
                "indicator": "Bus error: 10",
            },
            "faultingThread": 0,
            "threads": [{
                "triggered": true,
                "name": "ty-main",
                "threadState": {
                    "pc": {"value": fault_pc, "matchesCrashFrame": 1},
                    "far": {"value": fault_pc},
                    "lr": {"value": 0},
                    "esr": {"description": "(Instruction Abort) Permission fault"},
                }
            }],
        });
        fs::write(
            diagnostic_dir.join("ty-2026-04-27-202253.ips"),
            format!("{ips_header}\n{ips_body}\n"),
        )
        .unwrap();

        let mut env_overrides = BTreeMap::new();
        env_overrides.insert(
            REPLAY_ARTIFACT_DIR_ENV.to_string(),
            replay_dir.display().to_string(),
        );
        let now = system_time_millis(SystemTime::now());
        let result = CommandResult {
            argv: vec!["target/user/release/ty".to_string()],
            cwd: repo_root.display().to_string(),
            pid: Some(4242),
            returncode: 1,
            signal: Some(10),
            timed_out: false,
            started_unix_ms: Some(now.saturating_sub(1_000)),
            ended_unix_ms: Some(now.saturating_add(1_000)),
            elapsed_seconds: 1.0,
            stdout: String::new(),
            stderr: String::new(),
            env_overrides,
            crash_packet: None,
        };

        let summary = capture_crash_packet_from_report_dirs(
            &spec_dir,
            repo_root,
            &result,
            &[diagnostic_dir],
            0,
        )
        .unwrap()
        .unwrap();

        assert_eq!(summary.fault_signal.as_deref(), Some("SIGBUS"));
        assert_eq!(summary.fault_signal_code, Some(10));
        assert_eq!(summary.fault_pc.as_deref(), Some("0x1036fc000"));
        assert_eq!(summary.fault_address.as_deref(), Some("0x1036fc000"));
        assert_eq!(summary.artifact_root, "reports/run/MCLamportMutex");
        assert_eq!(
            summary.replay_artifact_dir,
            "reports/run/MCLamportMutex/trust_cg-replay"
        );
        assert!(summary.non_promoting);
        assert_eq!(summary.status_reason, "native_smoke_child_failed");
        assert_eq!(summary.jit_symbol.as_deref(), Some("Request__1_1"));
        assert_eq!(summary.pc_map_offset.as_deref(), Some("0x0"));
        assert_eq!(summary.pc_map_offset_decimal, Some(0));
        assert_eq!(summary.pc_map_block.as_deref(), Some("bb0"));

        let packet_path = repo_root.join(&summary.path);
        let packet: Value =
            serde_json::from_str(&fs::read_to_string(packet_path).unwrap()).unwrap();
        assert_eq!(packet["schema"].as_str(), Some(CRASH_PACKET_SCHEMA));
        assert_eq!(
            packet["artifact_root"].as_str(),
            Some("reports/run/MCLamportMutex")
        );
        assert_eq!(
            packet["status"]["native_dispatch_promoted"].as_bool(),
            Some(false)
        );
        assert_eq!(packet["status"]["non_promoting"].as_bool(), Some(true));
        assert_eq!(
            packet
                .pointer("/os_crash_report/fault_pc")
                .and_then(Value::as_str),
            Some("0x1036fc000")
        );
        assert_eq!(
            packet
                .pointer("/os_crash_report/fault_address")
                .and_then(Value::as_str),
            Some("0x1036fc000")
        );
        assert_eq!(
            packet.pointer("/jit/pc_map_offset").and_then(Value::as_str),
            Some("0x0")
        );
        assert_eq!(
            packet
                .pointer("/jit/function/range/start")
                .and_then(Value::as_str),
            Some("0x1036fc000")
        );
    }
}
