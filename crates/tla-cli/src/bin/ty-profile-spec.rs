// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Profile a single spec from the baseline catalog with a deterministic
//! per-role binary path.
//!
//! Rust port of `scripts/profile_spec.py` (and its helpers
//! `scripts/perf_harness.py`, `scripts/profile_spec_workflow.py`, and
//! `scripts/profile_sample_capture.py`). Builds `tla-cli` into a per-role
//! target dir, runs `ty check` against one spec, optionally attaches
//! macOS `sample` for a bounded window, and writes `summary.json`,
//! `build.json`, and captured stdout/stderr under `reports/perf/`.

use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use regex::Regex;
use serde_json::{json, Value};

const REPORTS_SUBDIR: &str = "reports/perf";
const TLAPLUS_EXAMPLES_SUBDIR: &str = "tlaplus-examples/specifications";
const SPEC_BASELINE_PATH: &str = "tests/tlc_comparison/spec_baseline.json";

#[derive(Parser, Debug)]
#[command(
    name = "ty-profile-spec",
    about = "Profile a single spec from the baseline catalog"
)]
struct Cli {
    /// Spec name from `tests/tlc_comparison/spec_baseline.json` (e.g., MCBakery).
    spec_name: Option<String>,
    /// Checker timeout in seconds (default: spec's `diagnose_timeout_seconds`, or 120).
    #[arg(long)]
    timeout: Option<u64>,
    /// Additional flags to pass to `ty check` (e.g. "--max-states 100000").
    #[arg(long, default_value = "")]
    extra_flags: String,
    /// Override output directory (default: `reports/perf/<timestamp>/<spec_name>`).
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Explicit cargo target dir (default: `target/profile_spec/<role-scope>`).
    #[arg(long)]
    target_dir: Option<PathBuf>,
    /// Cargo profile used to build the profiling binary.
    #[arg(long, default_value = "release")]
    cargo_profile: String,
    /// Worker count passed to `ty check`.
    #[arg(long, default_value_t = 1)]
    workers: u32,
    /// Attach macOS `sample` for N seconds while the checker runs.
    #[arg(long)]
    sample_seconds: Option<u64>,
    /// Sampling interval in milliseconds for macOS `sample`.
    #[arg(long, default_value_t = 1)]
    sample_interval_ms: u64,
    /// Seconds to wait after launch before attaching `sample`.
    #[arg(long, default_value_t = 0.0)]
    warmup_seconds: f64,
    /// List all available spec names and exit.
    #[arg(long)]
    list: bool,
    /// Enable detailed enumeration profiling (`--profile-enum-detail`).
    #[arg(long)]
    detailed: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(code) => ExitCode::from(code as u8),
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(1)
        }
    }
}

#[derive(Clone, Debug)]
struct SpecInfo {
    name: String,
    tla_path: String,
    cfg_path: String,
    category: Option<String>,
    expected_states: Option<i64>,
    timeout_seconds: u64,
}

fn run(cli: &Cli) -> Result<i32> {
    let repo_root = repo_root()?;
    let specs = load_specs(&repo_root)?;

    if cli.list {
        list_specs(&specs);
        return Ok(0);
    }
    let spec_name = match &cli.spec_name {
        Some(name) => name.clone(),
        None => {
            eprintln!("usage: ty-profile-spec <SPEC> [...]");
            return Ok(1);
        }
    };
    validate_cli(cli)?;

    let spec = specs
        .iter()
        .find(|s| s.name == spec_name)
        .cloned()
        .ok_or_else(|| anyhow!("Spec '{spec_name}' not found in baseline catalog"))?;

    let timeout = cli.timeout.unwrap_or(spec.timeout_seconds);
    let timestamp = current_timestamp();
    let output_dir = cli.output_dir.clone().unwrap_or_else(|| {
        repo_root
            .join(REPORTS_SUBDIR)
            .join(&timestamp)
            .join(&spec.name)
    });
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating {}", output_dir.display()))?;

    let target_dir = cli
        .target_dir
        .clone()
        .map(|p| {
            if p.is_absolute() {
                p
            } else {
                repo_root.join(p)
            }
        })
        .unwrap_or_else(|| {
            repo_root
                .join("target/profile_spec")
                .join(target_scope_name())
        });
    let binary_path = resolve_binary_path(&target_dir, &cli.cargo_profile, "ty");

    let (tla_path, cfg_path) = validate_spec_files(&repo_root, &spec)?;
    let build_command = build_cargo_build_command(&repo_root, &cli.cargo_profile, &target_dir);
    let run_command = build_ty_check_command(
        &binary_path,
        &tla_path,
        &cfg_path,
        &cli.extra_flags,
        cli.detailed,
        cli.workers,
        &repo_root,
    )?;

    show_run_header(
        &spec,
        &timestamp,
        cli,
        timeout,
        &target_dir,
        &binary_path,
        &output_dir,
    );

    eprintln!("Building profiling binary...");
    let build_result = run_command_capture(&build_command, None)?;
    let build_stdout_path = output_dir.join("build.stdout.txt");
    let build_stderr_path = output_dir.join("build.stderr.txt");
    fs::write(&build_stdout_path, &build_result.stdout)?;
    fs::write(&build_stderr_path, &build_result.stderr)?;

    let mut build_json = build_metadata(
        cli,
        timeout,
        &cli.cargo_profile,
        &target_dir,
        &binary_path,
        &build_command,
        &run_command,
        &build_result,
        &repo_root,
    );

    if build_result.returncode != 0 || !binary_path.exists() {
        return persist_build_failure(
            &output_dir,
            &binary_path,
            &build_result,
            &mut build_json,
            &build_command,
            &run_command,
            cli.workers,
        );
    }

    eprintln!("Running checker...");
    let sample_output_path = cli.sample_seconds.map(|_| output_dir.join("sample.txt"));
    let (run_result, sample_meta) = run_command_with_optional_sample(
        &run_command,
        timeout,
        cli.sample_seconds,
        cli.sample_interval_ms,
        cli.warmup_seconds,
        sample_output_path.as_deref(),
    )?;

    fs::write(output_dir.join("stdout.txt"), &run_result.stdout)?;
    fs::write(output_dir.join("stderr.txt"), &run_result.stderr)?;
    if let Some(meta) = &sample_meta {
        build_json["sample"] = meta.clone();
    } else {
        build_json["sample"] = Value::Null;
    }
    fs::write(
        output_dir.join("build.json"),
        serde_json::to_string_pretty(&build_json)? + "\n",
    )?;
    write_command_log(
        &output_dir,
        cli.workers,
        &build_command,
        &run_command,
        sample_meta.as_ref(),
    )?;

    let summary = build_summary(
        &spec,
        &timestamp,
        cli,
        timeout,
        &cli.cargo_profile,
        &target_dir,
        &binary_path,
        &build_result,
        &run_result,
        sample_meta.as_ref(),
        &repo_root,
    );
    fs::write(
        output_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    show_summary(
        &build_result,
        &run_result,
        &summary,
        sample_meta.as_ref(),
        &output_dir,
    );
    Ok(run_result.returncode)
}

fn validate_cli(cli: &Cli) -> Result<()> {
    if cli.workers == 0 {
        bail!("--workers must be positive");
    }
    if extra_flags_override_workers(&cli.extra_flags) {
        bail!("pass worker count via --workers, not --extra-flags");
    }
    if let Some(s) = cli.sample_seconds {
        if s == 0 {
            bail!("--sample-seconds must be positive");
        }
    }
    if cli.sample_interval_ms == 0 {
        bail!("--sample-interval-ms must be positive");
    }
    if cli.warmup_seconds < 0.0 {
        bail!("--warmup-seconds must be non-negative");
    }
    Ok(())
}

fn extra_flags_override_workers(extra_flags: &str) -> bool {
    for tok in shlex_split(extra_flags) {
        if tok == "--workers" || tok.starts_with("--workers=") {
            return true;
        }
        if tok == "-w" || tok.starts_with("-w=") {
            return true;
        }
        if tok.starts_with("-w") && tok[2..].chars().all(|c| c.is_ascii_digit()) && tok.len() > 2 {
            return true;
        }
    }
    false
}

fn shlex_split(text: &str) -> Vec<String> {
    // Conservative shell-like splitter mirroring `shlex.split` for the common
    // cases used in the perf harness CLI (whitespace, single/double quotes).
    let mut out = Vec::<String>::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escape_next = false;
    for ch in text.chars() {
        if escape_next {
            current.push(ch);
            escape_next = false;
            continue;
        }
        match ch {
            '\\' if !in_single => {
                escape_next = true;
            }
            '\'' if !in_double => {
                in_single = !in_single;
            }
            '"' if !in_single => {
                in_double = !in_double;
            }
            c if c.is_whitespace() && !in_single && !in_double => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn shlex_join(argv: &[String]) -> String {
    argv.iter()
        .map(|a| {
            if a.is_empty() || a.contains(' ') || a.contains('"') || a.contains('\'') {
                format!("'{}'", a.replace('\'', "'\\''"))
            } else {
                a.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn repo_root() -> Result<PathBuf> {
    if let Ok(cwd) = env::current_dir() {
        let mut dir = Some(cwd);
        while let Some(p) = dir {
            if p.join("Cargo.lock").exists() && p.join("crates").exists() {
                return Ok(p);
            }
            dir = p.parent().map(Path::to_path_buf);
        }
    }
    Err(anyhow!("could not locate workspace root from cwd"))
}

fn load_specs(repo_root: &Path) -> Result<Vec<SpecInfo>> {
    let path = repo_root.join(SPEC_BASELINE_PATH);
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let baseline: Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let specs_obj = baseline
        .get("specs")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("baseline JSON missing 'specs' object"))?;

    let mut out = Vec::new();
    for (name, entry) in specs_obj {
        let source = entry.get("source").and_then(|v| v.as_object());
        let tla = source
            .and_then(|s| s.get("tla_path"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let cfg = source
            .and_then(|s| s.get("cfg_path"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if tla.is_empty() || cfg.is_empty() {
            continue;
        }
        let category = entry
            .get("category")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let expected_states = entry
            .get("tlc")
            .and_then(|v| v.get("states_found"))
            .and_then(|v| v.as_i64());
        let timeout_seconds = entry
            .get("diagnose_timeout_seconds")
            .and_then(|v| v.as_u64())
            .map(|t| if t < 120 { 120 } else { t })
            .unwrap_or(120);
        out.push(SpecInfo {
            name: name.clone(),
            tla_path: tla.to_string(),
            cfg_path: cfg.to_string(),
            category,
            expected_states,
            timeout_seconds,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn list_specs(specs: &[SpecInfo]) {
    println!("Available specs:");
    for s in specs {
        println!("  {}", s.name);
    }
}

fn current_timestamp() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H%M%S").to_string()
}

fn target_scope_name() -> String {
    "user".to_string()
}

fn resolve_binary_path(target_dir: &Path, cargo_profile: &str, bin_name: &str) -> PathBuf {
    target_dir.join(cargo_profile).join(bin_name)
}

fn tlaplus_examples_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(TLAPLUS_EXAMPLES_SUBDIR)
}

fn validate_spec_files(_repo_root: &Path, spec: &SpecInfo) -> Result<(PathBuf, PathBuf)> {
    let raw_tla = PathBuf::from(&spec.tla_path);
    let raw_cfg = PathBuf::from(&spec.cfg_path);
    let base = tlaplus_examples_dir();
    let tla = if raw_tla.is_absolute() {
        raw_tla
    } else {
        base.join(&raw_tla)
    };
    let cfg = if raw_cfg.is_absolute() {
        raw_cfg
    } else {
        base.join(&raw_cfg)
    };
    if !tla.exists() {
        bail!("TLA file not found: {}", tla.display());
    }
    if !cfg.exists() {
        bail!("config file not found: {}", cfg.display());
    }
    Ok((tla, cfg))
}

#[derive(Clone, Debug)]
struct CommandSpec {
    argv: Vec<String>,
    cwd: PathBuf,
}

fn build_cargo_build_command(
    repo_root: &Path,
    cargo_profile: &str,
    target_dir: &Path,
) -> CommandSpec {
    CommandSpec {
        argv: vec![
            "cargo".to_string(),
            "build".to_string(),
            "--profile".to_string(),
            cargo_profile.to_string(),
            "-p".to_string(),
            "tla-cli".to_string(),
            "--target-dir".to_string(),
            target_dir.to_string_lossy().to_string(),
            "--bin".to_string(),
            "ty".to_string(),
        ],
        cwd: repo_root.to_path_buf(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_ty_check_command(
    binary_path: &Path,
    tla_path: &Path,
    cfg_path: &Path,
    extra_flags: &str,
    detailed: bool,
    workers: u32,
    repo_root: &Path,
) -> Result<CommandSpec> {
    let mut argv = vec![
        binary_path.to_string_lossy().to_string(),
        "check".to_string(),
        tla_path.to_string_lossy().to_string(),
        "--config".to_string(),
        cfg_path.to_string_lossy().to_string(),
        "--profile-enum".to_string(),
        "--profile-eval".to_string(),
    ];
    if detailed {
        argv.push("--profile-enum-detail".to_string());
    }
    argv.push("--workers".to_string());
    argv.push(workers.to_string());
    argv.push("--force".to_string());
    argv.extend(shlex_split(extra_flags));
    Ok(CommandSpec {
        argv,
        cwd: repo_root.to_path_buf(),
    })
}

#[derive(Clone, Debug)]
struct CommandResult {
    argv: Vec<String>,
    cwd: String,
    returncode: i32,
    elapsed_seconds: f64,
    stdout: String,
    stderr: String,
}

fn run_command_capture(cmd: &CommandSpec, timeout: Option<u64>) -> Result<CommandResult> {
    let mut command = Command::new(&cmd.argv[0]);
    command.args(&cmd.argv[1..]).current_dir(&cmd.cwd);
    let start = Instant::now();
    if let Some(_secs) = timeout {
        // We rely on the caller to enforce timeouts via the sample workflow.
        // For pure `cargo build` invocations there is no built-in timeout in
        // std::process::Command; this matches Python's `subprocess.run` for
        // builds where the script also lacks a graceful kill.
    }
    let output = command
        .output()
        .with_context(|| format!("running {:?}", cmd.argv))?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok(CommandResult {
        argv: cmd.argv.clone(),
        cwd: cmd.cwd.to_string_lossy().to_string(),
        returncode: output.status.code().unwrap_or(-1),
        elapsed_seconds: elapsed,
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn run_command_with_optional_sample(
    command: &CommandSpec,
    timeout_seconds: u64,
    sample_seconds: Option<u64>,
    sample_interval_ms: u64,
    warmup_seconds: f64,
    sample_output_path: Option<&Path>,
) -> Result<(CommandResult, Option<Value>)> {
    let Some(sample_seconds) = sample_seconds else {
        let result = run_with_timeout(command, timeout_seconds)?;
        return Ok((result, None));
    };
    let sample_output_path = sample_output_path
        .ok_or_else(|| anyhow!("sample_output_path required when --sample-seconds is set"))?;

    let started = Instant::now();
    let mut child = Command::new(&command.argv[0])
        .args(&command.argv[1..])
        .current_dir(&command.cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {:?}", command.argv))?;

    if warmup_seconds > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(warmup_seconds));
    }

    let pid = child.id();
    let sample_meta = if process_still_running(&mut child) {
        run_sample_capture(
            command,
            pid,
            sample_seconds,
            sample_interval_ms,
            sample_output_path,
        )?
    } else {
        record_sample_skip(sample_output_path, warmup_seconds)?
    };

    let total_budget = Duration::from_secs(timeout_seconds);
    let result = wait_with_timeout(command, &mut child, started, total_budget)?;
    Ok((result, Some(sample_meta)))
}

fn run_with_timeout(command: &CommandSpec, timeout_seconds: u64) -> Result<CommandResult> {
    let started = Instant::now();
    let mut child = Command::new(&command.argv[0])
        .args(&command.argv[1..])
        .current_dir(&command.cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {:?}", command.argv))?;
    wait_with_timeout(
        command,
        &mut child,
        started,
        Duration::from_secs(timeout_seconds),
    )
}

fn wait_with_timeout(
    command: &CommandSpec,
    child: &mut Child,
    started: Instant,
    budget: Duration,
) -> Result<CommandResult> {
    let poll_interval = Duration::from_millis(100);
    loop {
        match child.try_wait()? {
            Some(status) => {
                let elapsed = started.elapsed().as_secs_f64();
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut s) = child.stdout.take() {
                    use std::io::Read;
                    let _ = s.read_to_string(&mut stdout);
                }
                if let Some(mut s) = child.stderr.take() {
                    use std::io::Read;
                    let _ = s.read_to_string(&mut stderr);
                }
                return Ok(CommandResult {
                    argv: command.argv.clone(),
                    cwd: command.cwd.to_string_lossy().to_string(),
                    returncode: status.code().unwrap_or(-1),
                    elapsed_seconds: elapsed,
                    stdout,
                    stderr,
                });
            }
            None => {
                if started.elapsed() >= budget {
                    let _ = child.kill();
                    let _ = child.wait();
                    let mut stdout = String::new();
                    let mut stderr = String::new();
                    if let Some(mut s) = child.stdout.take() {
                        use std::io::Read;
                        let _ = s.read_to_string(&mut stdout);
                    }
                    if let Some(mut s) = child.stderr.take() {
                        use std::io::Read;
                        let _ = s.read_to_string(&mut stderr);
                    }
                    let message = format!("Timeout after {} seconds", budget.as_secs());
                    let stderr = if stderr.is_empty() {
                        message
                    } else {
                        format!("{stderr}\n\n{message}")
                    };
                    return Ok(CommandResult {
                        argv: command.argv.clone(),
                        cwd: command.cwd.to_string_lossy().to_string(),
                        returncode: 124,
                        elapsed_seconds: started.elapsed().as_secs_f64(),
                        stdout,
                        stderr,
                    });
                }
                std::thread::sleep(poll_interval);
            }
        }
    }
}

fn process_still_running(child: &mut Child) -> bool {
    match child.try_wait() {
        Ok(Some(_)) => false,
        Ok(None) => true,
        Err(err) if err.kind() == ErrorKind::Interrupted => true,
        Err(_) => false,
    }
}

fn run_sample_capture(
    command: &CommandSpec,
    pid: u32,
    sample_seconds: u64,
    sample_interval_ms: u64,
    sample_output_path: &Path,
) -> Result<Value> {
    let sample_bin =
        find_sample_binary().ok_or_else(|| anyhow!("macOS `sample` binary not found on PATH"))?;
    let sample_command: Vec<String> = vec![
        sample_bin.to_string_lossy().to_string(),
        pid.to_string(),
        sample_seconds.to_string(),
        sample_interval_ms.to_string(),
        "-mayDie".to_string(),
        "-fullPaths".to_string(),
        "-file".to_string(),
        sample_output_path.to_string_lossy().to_string(),
    ];

    let started = Instant::now();
    let sample_timeout = Duration::from_secs(std::cmp::max(sample_seconds + 5, 10));
    let mut cmd = Command::new(&sample_command[0]);
    cmd.args(&sample_command[1..]).current_dir(&command.cwd);
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {:?}", sample_command))?;
    let result = wait_with_timeout(
        &CommandSpec {
            argv: sample_command.clone(),
            cwd: command.cwd.clone(),
        },
        &mut child,
        started,
        sample_timeout,
    )?;

    if result.returncode == 124 {
        let message = format!(
            "sample timed out after {} seconds",
            sample_timeout.as_secs()
        );
        ensure_sample_artifact_text(sample_output_path, &message)?;
        return Ok(build_sample_meta(
            Some(&sample_command),
            "timed_out",
            sample_output_path,
            result.elapsed_seconds,
            Some(124),
            &result.stdout,
            &message,
        ));
    }
    let artifact_bytes = sample_output_path.metadata().map(|m| m.len()).unwrap_or(0);
    if result.returncode != 0 {
        let message = if !result.stderr.trim().is_empty() {
            result.stderr.trim().to_string()
        } else if !result.stdout.trim().is_empty() {
            result.stdout.trim().to_string()
        } else {
            format!(
                "sample exited with return code {} without producing output",
                result.returncode
            )
        };
        ensure_sample_artifact_text(sample_output_path, &message)?;
        let stderr = if result.stderr.is_empty() {
            message.clone()
        } else {
            result.stderr.clone()
        };
        return Ok(build_sample_meta(
            Some(&sample_command),
            "failed",
            sample_output_path,
            result.elapsed_seconds,
            Some(result.returncode),
            &result.stdout,
            &stderr,
        ));
    }
    if artifact_bytes == 0 {
        let message = if !result.stderr.trim().is_empty() {
            result.stderr.trim().to_string()
        } else if !result.stdout.trim().is_empty() {
            result.stdout.trim().to_string()
        } else {
            "sample exited without producing output".to_string()
        };
        ensure_sample_artifact_text(sample_output_path, &message)?;
        let stderr = if result.stderr.is_empty() {
            message.clone()
        } else {
            result.stderr.clone()
        };
        return Ok(build_sample_meta(
            Some(&sample_command),
            "empty_output",
            sample_output_path,
            result.elapsed_seconds,
            Some(result.returncode),
            &result.stdout,
            &stderr,
        ));
    }
    Ok(build_sample_meta(
        Some(&sample_command),
        "succeeded",
        sample_output_path,
        result.elapsed_seconds,
        Some(result.returncode),
        &result.stdout,
        &result.stderr,
    ))
}

fn record_sample_skip(sample_output_path: &Path, warmup_seconds: f64) -> Result<Value> {
    let message = format!(
        "process exited before sampling after {:.3}s warmup",
        warmup_seconds
    );
    fs::write(sample_output_path, format!("{message}\n"))?;
    Ok(build_sample_meta(
        None,
        "skipped",
        sample_output_path,
        0.0,
        None,
        "",
        &message,
    ))
}

fn ensure_sample_artifact_text(sample_output_path: &Path, message: &str) -> Result<()> {
    if let Ok(meta) = sample_output_path.metadata() {
        if meta.len() > 0 {
            return Ok(());
        }
    }
    fs::write(sample_output_path, format!("{}\n", message.trim_end()))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_sample_meta(
    argv: Option<&[String]>,
    status: &str,
    sample_output_path: &Path,
    elapsed_seconds: f64,
    returncode: Option<i32>,
    stdout: &str,
    stderr: &str,
) -> Value {
    let artifact_bytes = sample_output_path.metadata().map(|m| m.len()).unwrap_or(0);
    json!({
        "status": status,
        "argv": argv.map(|a| a.to_vec()),
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout": stdout,
        "stderr": stderr,
        "artifact": repo_relative(sample_output_path),
        "artifact_bytes": artifact_bytes,
    })
}

fn repo_relative(path: &Path) -> String {
    if let Ok(root) = repo_root() {
        if let Ok(rel) = path.strip_prefix(&root) {
            return rel.to_string_lossy().to_string();
        }
    }
    path.to_string_lossy().to_string()
}

fn find_sample_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("PATH") {
        for entry in path.split(':') {
            let candidate = Path::new(entry).join("sample");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    let fallback = PathBuf::from("/usr/bin/sample");
    if fallback.exists() {
        Some(fallback)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn build_metadata(
    cli: &Cli,
    timeout: u64,
    cargo_profile: &str,
    target_dir: &Path,
    binary_path: &Path,
    build_command: &CommandSpec,
    run_command: &CommandSpec,
    build_result: &CommandResult,
    _repo_root: &Path,
) -> Value {
    let build_section = json!({
        "command": build_result.argv,
        "cwd": build_result.cwd,
        "returncode": build_result.returncode,
        "elapsed_seconds": build_result.elapsed_seconds,
        "stdout_artifact": repo_relative(Path::new("build.stdout.txt")),
        "stderr_artifact": repo_relative(Path::new("build.stderr.txt")),
    });
    let _ = build_section;
    let _ = build_command;
    json!({
        "workers": cli.workers,
        "cargo_profile": cargo_profile,
        "target_dir": repo_relative(target_dir),
        "binary_path": repo_relative(binary_path),
        "build": {
            "command": build_result.argv,
            "cwd": build_result.cwd,
            "returncode": build_result.returncode,
            "elapsed_seconds": build_result.elapsed_seconds,
        },
        "run_command": run_command.argv,
        "run_cwd": run_command.cwd.to_string_lossy(),
        "sample_seconds": cli.sample_seconds,
        "sample_interval_ms": cli.sample_interval_ms,
        "warmup_seconds": cli.warmup_seconds,
        "timeout_used": timeout,
    })
}

fn persist_build_failure(
    output_dir: &Path,
    binary_path: &Path,
    build_result: &CommandResult,
    build_json: &mut Value,
    build_command: &CommandSpec,
    run_command: &CommandSpec,
    workers: u32,
) -> Result<i32> {
    if build_result.returncode == 0 {
        build_json["error"] = json!(format!(
            "built binary not found at {}",
            binary_path.display()
        ));
    }
    fs::write(
        output_dir.join("build.json"),
        serde_json::to_string_pretty(build_json)? + "\n",
    )?;
    fs::write(output_dir.join("stdout.txt"), "")?;
    fs::write(output_dir.join("stderr.txt"), &build_result.stderr)?;
    write_command_log(output_dir, workers, build_command, run_command, None)?;
    if !build_result.stderr.is_empty() {
        eprintln!("{}", build_result.stderr);
    }
    if build_result.returncode == 0 {
        eprintln!("Error: built binary not found at {}", binary_path.display());
        return Ok(1);
    }
    Ok(build_result.returncode)
}

fn write_command_log(
    output_dir: &Path,
    workers: u32,
    build_command: &CommandSpec,
    run_command: &CommandSpec,
    sample_meta: Option<&Value>,
) -> Result<()> {
    let mut lines = vec![
        format!("workers: {workers}"),
        format!("build: {}", shlex_join(&build_command.argv)),
        format!("build_cwd: {}", build_command.cwd.display()),
        format!("run: {}", shlex_join(&run_command.argv)),
        format!("run_cwd: {}", run_command.cwd.display()),
    ];
    if let Some(meta) = sample_meta {
        let status = meta
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        lines.push(format!("sample_status: {status}"));
        if let Some(argv) = meta.get("argv").and_then(|v| v.as_array()) {
            let argv_str: Vec<String> = argv
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            lines.push(format!("sample: {}", shlex_join(&argv_str)));
        } else {
            let stderr = meta
                .get("stderr")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            lines.push(format!("sample: skipped ({stderr})"));
        }
    }
    fs::write(output_dir.join("command.txt"), lines.join("\n") + "\n")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_summary(
    spec: &SpecInfo,
    timestamp: &str,
    cli: &Cli,
    timeout: u64,
    cargo_profile: &str,
    target_dir: &Path,
    binary_path: &Path,
    build_result: &CommandResult,
    run_result: &CommandResult,
    sample_meta: Option<&Value>,
    _repo_root: &Path,
) -> Value {
    let parsed = parse_profiling_output(&run_result.stdout, &run_result.stderr);
    let extra_flags_value = if cli.extra_flags.is_empty() {
        Value::Null
    } else {
        Value::String(cli.extra_flags.clone())
    };
    json!({
        "states_found": parsed.states_found,
        "transitions": parsed.transitions,
        "runtime_sec": parsed.runtime_sec,
        "profile_enum": parsed.profile_enum,
        "profile_eval": parsed.profile_eval,
        "error": parsed.error,
        "spec_name": spec.name,
        "workers": cli.workers,
        "tla_path": spec.tla_path,
        "cfg_path": spec.cfg_path,
        "category": spec.category,
        "expected_states": spec.expected_states,
        "timeout_used": timeout,
        "extra_flags": extra_flags_value,
        "timestamp": timestamp,
        "returncode": run_result.returncode,
        "cargo_profile": cargo_profile,
        "target_dir": repo_relative(target_dir),
        "binary_path": repo_relative(binary_path),
        "command": run_result.argv,
        "command_cwd": run_result.cwd,
        "build": {
            "command": build_result.argv,
            "cwd": build_result.cwd,
            "returncode": build_result.returncode,
            "elapsed_seconds": build_result.elapsed_seconds,
        },
        "sample": sample_meta.cloned().unwrap_or(Value::Null),
    })
}

#[derive(Default, Debug)]
struct ProfilingParseResult {
    states_found: Option<i64>,
    transitions: Option<i64>,
    runtime_sec: Option<f64>,
    profile_enum: Value,
    profile_eval: Value,
    error: Option<String>,
}

fn parse_profiling_output(stdout: &str, stderr: &str) -> ProfilingParseResult {
    let combined = format!("{stdout}\n{stderr}");
    let states_found = last_capture_int(r"(?m)^\s*States found:\s+(\d+)\s*$", &combined);
    let transitions = last_capture_int(r"(?m)^\s*Transitions:\s+(\d+)\s*$", &combined);
    let runtime_sec = first_capture_float(r"Time:\s*([\d.]+)s", &combined);
    let profile_enum = parse_enumeration_profile(&combined);
    let profile_eval = parse_eval_profile(&combined);
    let error = extract_first_error(&combined);
    ProfilingParseResult {
        states_found,
        transitions,
        runtime_sec,
        profile_enum,
        profile_eval,
        error,
    }
}

fn last_capture_int(pattern: &str, hay: &str) -> Option<i64> {
    let re = Regex::new(pattern).ok()?;
    let mut last = None;
    for caps in re.captures_iter(hay) {
        if let Some(m) = caps.get(1) {
            last = m.as_str().parse::<i64>().ok();
        }
    }
    last
}

fn first_capture_float(pattern: &str, hay: &str) -> Option<f64> {
    let re = Regex::new(pattern).ok()?;
    let caps = re.captures(hay)?;
    caps.get(1)?.as_str().parse::<f64>().ok()
}

fn parse_enumeration_profile(combined: &str) -> Value {
    let re_block =
        Regex::new(r"(?s)=== Enumeration Profile ===(.*?)(?:===|Model checking complete|\z)")
            .unwrap();
    let mut obj = serde_json::Map::new();
    if let Some(block) = re_block.captures(combined).and_then(|c| c.get(1)) {
        let block = block.as_str();
        let line_re = Regex::new(r"(?m)^\s*([A-Za-z ]+):\s*([\d.]+)s\s*\(\s*([\d.]+)%\)").unwrap();
        for caps in line_re.captures_iter(block) {
            let key = caps[1].trim().to_lowercase().replace(' ', "_");
            let seconds: f64 = caps[2].parse().unwrap_or(0.0);
            let percent: f64 = caps[3].parse().unwrap_or(0.0);
            obj.insert(key, json!({"seconds": seconds, "percent": percent}));
        }
        if let Some(c) = Regex::new(r"Total successors:\s*(\d+)")
            .unwrap()
            .captures(block)
        {
            if let Ok(v) = c[1].parse::<i64>() {
                obj.insert("total_successors".to_string(), json!(v));
            }
        }
        if let Some(c) = Regex::new(r"New states:\s*(\d+)").unwrap().captures(block) {
            if let Ok(v) = c[1].parse::<i64>() {
                obj.insert("new_states".to_string(), json!(v));
            }
        }
    }
    Value::Object(obj)
}

fn parse_eval_profile(combined: &str) -> Value {
    let re_block =
        Regex::new(r"(?s)=== Eval Profile ===(.*?)(?:===|Model checking complete|\z)").unwrap();
    let mut obj = serde_json::Map::new();
    if let Some(block) = re_block.captures(combined).and_then(|c| c.get(1)) {
        let block = block.as_str();
        if let Some(c) = Regex::new(r"Total eval\(\) calls:\s*(\d+)")
            .unwrap()
            .captures(block)
        {
            if let Ok(v) = c[1].parse::<i64>() {
                obj.insert("total_calls".to_string(), json!(v));
            }
        }
    }
    Value::Object(obj)
}

fn extract_first_error(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let re = Regex::new(r"(?s)(Error:.*?)(?:\n\n|\z)").unwrap();
    if let Some(c) = re.captures(text) {
        return Some(c[1].trim().to_string());
    }
    let re2 = Regex::new(r"(?im)^(?:error:|exception:).*$").unwrap();
    if let Some(m) = re2.find(text) {
        return Some(m.as_str().trim().to_string());
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn show_run_header(
    spec: &SpecInfo,
    _timestamp: &str,
    cli: &Cli,
    timeout: u64,
    target_dir: &Path,
    binary_path: &Path,
    output_dir: &Path,
) {
    eprintln!("Profiling: {}", spec.name);
    eprintln!("  TLA: {}", spec.tla_path);
    eprintln!("  Config: {}", spec.cfg_path);
    eprintln!("  Timeout: {timeout}s");
    eprintln!("  Workers: {}", cli.workers);
    eprintln!("  Cargo profile: {}", cli.cargo_profile);
    eprintln!("  Target dir: {}", target_dir.display());
    eprintln!("  Binary path: {}", binary_path.display());
    eprintln!("  Output: {}", output_dir.display());
    if let Some(s) = cli.sample_seconds {
        eprintln!(
            "  Sample: {}s at {}ms (warmup {:.3}s)",
            s, cli.sample_interval_ms, cli.warmup_seconds
        );
    }
    eprintln!();
}

fn show_summary(
    build_result: &CommandResult,
    run_result: &CommandResult,
    summary: &Value,
    sample_meta: Option<&Value>,
    output_dir: &Path,
) {
    eprintln!("=== Profiling Summary ===");
    eprintln!("  Return code: {}", run_result.returncode);
    eprintln!("  Build time: {:.3}s", build_result.elapsed_seconds);
    eprintln!("  Checker time: {:.3}s", run_result.elapsed_seconds);
    if let Some(states) = summary.get("states_found").and_then(|v| v.as_i64()) {
        eprintln!("  States found: {states}");
    }
    if let Some(runtime) = summary.get("runtime_sec").and_then(|v| v.as_f64()) {
        eprintln!("  Runtime: {runtime:.3}s");
    }
    if let Some(meta) = sample_meta {
        let status = meta
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let artifact = meta.get("artifact").and_then(|v| v.as_str()).unwrap_or("");
        eprintln!("  Sample status: {status}");
        eprintln!("  Sample artifact: {artifact}");
        if status != "succeeded" {
            if let Some(detail) = meta.get("stderr").and_then(|v| v.as_str()) {
                if !detail.is_empty() {
                    let preview: String = detail.chars().take(100).collect();
                    eprintln!("  Sample detail: {preview}");
                }
            }
        }
    }
    if let Some(err) = summary.get("error").and_then(|v| v.as_str()) {
        if !err.is_empty() {
            let preview: String = err.chars().take(100).collect();
            eprintln!("  Error: {preview}...");
        }
    }
    eprintln!();
    eprintln!("Output saved to: {}", output_dir.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_flags_override_workers_detects_long_short_and_packed_forms() {
        for flag in [
            "--workers 9",
            "--workers=9",
            "-w 9",
            "-w=9",
            "-w9",
            "--max-states 100 --workers 4",
        ] {
            assert!(
                extra_flags_override_workers(flag),
                "should reject worker override in {flag:?}"
            );
        }
        for flag in ["", "--max-states 100", "--workers-foo"] {
            assert!(
                !extra_flags_override_workers(flag),
                "should accept {flag:?}"
            );
        }
    }

    #[test]
    fn shlex_split_handles_basic_quoting() {
        assert_eq!(shlex_split(""), Vec::<String>::new());
        assert_eq!(
            shlex_split("--workers 4 --force"),
            vec!["--workers", "4", "--force"]
        );
        assert_eq!(
            shlex_split("--name 'my value' --x \"q u\""),
            vec!["--name", "my value", "--x", "q u"]
        );
    }

    #[test]
    fn shlex_join_quotes_arguments_with_spaces() {
        let argv = vec![
            "bin".to_string(),
            "with space".to_string(),
            "ok".to_string(),
        ];
        let joined = shlex_join(&argv);
        assert_eq!(joined, "bin 'with space' ok");
    }

    #[test]
    fn parse_profiling_output_recovers_states_and_runtime() {
        let text = "
States found: 1234
Transitions: 5678
Time: 0.123s
=== Enumeration Profile ===
Successor generation:     0.10s ( 50.0%)
New states:               42
Total successors:         100
=== Eval Profile ===
Total eval() calls: 99
Model checking complete
";
        let parsed = parse_profiling_output(text, "");
        assert_eq!(parsed.states_found, Some(1234));
        assert_eq!(parsed.transitions, Some(5678));
        assert!((parsed.runtime_sec.unwrap() - 0.123).abs() < 1e-6);
        assert_eq!(
            parsed
                .profile_enum
                .get("successor_generation")
                .and_then(|v| v.get("seconds"))
                .and_then(|v| v.as_f64()),
            Some(0.10)
        );
        assert_eq!(
            parsed
                .profile_enum
                .get("total_successors")
                .and_then(|v| v.as_i64()),
            Some(100)
        );
        assert_eq!(
            parsed
                .profile_eval
                .get("total_calls")
                .and_then(|v| v.as_i64()),
            Some(99)
        );
        assert!(parsed.error.is_none());
    }

    #[test]
    fn extract_first_error_returns_first_block() {
        let text = "Some context\nError: oh no\nmore details\n\nNext block";
        let err = extract_first_error(text).unwrap();
        assert!(err.contains("Error: oh no"));
        assert!(err.contains("more details"));
        assert!(!err.contains("Next block"));
    }

    #[test]
    fn target_scope_name_returns_user() {
        assert_eq!(target_scope_name(), "user");
    }

    #[test]
    fn build_cargo_build_command_targets_tla_cli_package() {
        let cmd = build_cargo_build_command(
            Path::new("/tmp/repo"),
            "profiling",
            Path::new("/tmp/target"),
        );
        assert!(cmd.argv.iter().any(|s| s == "-p"));
        let pkg_idx = cmd.argv.iter().position(|s| s == "-p").unwrap();
        assert_eq!(cmd.argv[pkg_idx + 1], "tla-cli");
        assert!(cmd.argv.iter().any(|s| s == "--profile"));
        assert!(cmd.argv.iter().any(|s| s == "ty"));
    }

    #[test]
    fn build_ty_check_command_uses_requested_workers() {
        let cmd = build_ty_check_command(
            Path::new("/tmp/bin"),
            Path::new("/tmp/a.tla"),
            Path::new("/tmp/a.cfg"),
            "--max-states 100",
            true,
            4,
            Path::new("/tmp/repo"),
        )
        .unwrap();
        let idx = cmd.argv.iter().position(|s| s == "--workers").unwrap();
        assert_eq!(cmd.argv[idx + 1], "4");
        assert!(cmd.argv.iter().any(|s| s == "--profile-enum-detail"));
        assert!(cmd.argv.iter().any(|s| s == "--max-states"));
    }

    #[test]
    fn validate_cli_rejects_non_positive_workers() {
        let cli = Cli {
            spec_name: None,
            timeout: None,
            extra_flags: String::new(),
            output_dir: None,
            target_dir: None,
            cargo_profile: "release".into(),
            workers: 0,
            sample_seconds: None,
            sample_interval_ms: 1,
            warmup_seconds: 0.0,
            list: false,
            detailed: false,
        };
        assert!(validate_cli(&cli).is_err());
    }
}
