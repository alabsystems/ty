// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty system-health-gate` -- Rust implementation of the system health contract.
//!
//! The former `scripts/system_health_check.py` Python facade and its
//! `system_health_check_support/` helpers have been deleted in favor of
//! this single, compiler-enforced interface. This module owns the
//! observable health-check contract: status-prefixed lines, optional JSON
//! manifest, and fail-closed behavior in enforce mode.

mod cargo_timings;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use regex::{Regex, RegexBuilder};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::time;

use self::cargo_timings::check_parse_cargo_timings_ndjson_self_test;
use crate::cli_schema::{SystemHealthGateArgs, SystemHealthGateMode};

const TLAPLUS_DIR_NAME: &str = "tlaplus";
const TLAPLUS_EXAMPLES_DIR_NAME: &str = "tlaplus-examples";
const TYTOOLS_JAR_NAME: &str = "tytools.jar";
const CANONICAL_AY_REPO_ID: &str = "github.com/alabsystems/ay";
const AY_DEP_SECTIONS: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];
const CURRENT_DOC_ROUTING_CHECK_NAME: &str = "cmd:check_current_doc_routing.py";
const CARGO_TIMINGS_SELF_TEST_CHECK_NAME: &str = "cmd:parse_cargo_timings_ndjson.py --self-test";
const TLC_DOT_SMOKE_CHECK_NAME: &str = "cmd:test_parse_tlc_dot_smoke.py";

#[derive(Clone, Copy, Debug)]
struct DocTextGuard {
    path: &'static str,
    required_substrings: &'static [&'static str],
    forbidden_patterns: &'static [&'static str],
}

const CURRENT_DOC_ROUTING_GUARDS: &[DocTextGuard] = &[
    DocTextGuard {
        path: "reports/research/coverage-gap-decomposition-current.md",
        required_substrings: &[
            "do not treat `MCBoulanger` as an active current timeout gap",
            "Category C calibration lane",
        ],
        forbidden_patterns: &[
            r"^\| MCBoulanger \| 7,866,982 \| 136s \| timeout \| timeout \| Too large for 120s even at TLC parity \|$",
            r"^\*\*3 are genuinely too large\*\* \(dijkstra-mutex, SlushMedium, MCBoulanger\)",
        ],
    },
    DocTextGuard {
        path: "reports/research/2026-03-11-coverage-gap-audit-current.md",
        required_substrings: &[
            "do not use `MCBoulanger` as an active current timeout canary",
            "Category C calibration lane",
        ],
        forbidden_patterns: &[
            r"^\| MCBoulanger \| 136s \| 7,866,982 \| safety, CONSTRAINT \|$",
            r"^### TLC completes but needs >120s \(11 specs\)$",
        ],
    },
    DocTextGuard {
        path: "reports/research/vision-gap-analysis-current.md",
        required_substrings: &["profile the Category C calibration lane"],
        forbidden_patterns: &[r"profile Category C timeouts"],
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HealthCheck {
    pub name: String,
    pub ok: bool,
    pub level: Option<String>,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CommandOutput {
    return_code: i32,
    output: String,
}

#[derive(Debug, Serialize)]
struct HealthManifest {
    schema_version: &'static str,
    generated_at: String,
    git_commit: String,
    project: String,
    summary: HealthSummary,
    checks: Vec<ManifestCheck>,
}

#[derive(Debug, Serialize)]
struct HealthSummary {
    status: String,
    passed: usize,
    warnings: usize,
    errors: usize,
}

#[derive(Debug, Serialize)]
struct ManifestCheck {
    name: String,
    ok: bool,
    level: String,
    detail: Option<String>,
}

impl HealthCheck {
    fn ok(name: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            name: name.into(),
            ok: true,
            level: None,
            detail,
        }
    }

    fn err(name: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            name: name.into(),
            ok: false,
            level: None,
            detail,
        }
    }

    fn warn(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ok: true,
            level: Some("warn".to_string()),
            detail: Some(detail.into()),
        }
    }
}

pub(crate) async fn cmd_system_health_gate(args: SystemHealthGateArgs) -> Result<()> {
    let project_root = args
        .project_root
        .unwrap_or(std::env::current_dir().context("failed to resolve current directory")?);
    let project_root = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.clone());
    let env_vars: BTreeMap<String, String> = std::env::vars().collect();
    let checks = run_system_health_checks(&project_root, &env_vars).await;
    let manifest = manifest(&project_root, &checks).await;

    if let Some(path) = &args.json_output {
        let text = serde_json::to_string_pretty(&manifest)
            .context("failed to serialize system health manifest")?;
        std::fs::write(path, format!("{text}\n"))
            .with_context(|| format!("failed to write {}", path.display()))?;
    }

    for check in &checks {
        println!("{}", format_check_line(check));
    }

    if manifest.summary.errors != 0 && args.mode == SystemHealthGateMode::Enforce {
        std::process::exit(1);
    }
    Ok(())
}

async fn run_system_health_checks(
    project_root: &Path,
    env_vars: &BTreeMap<String, String>,
) -> Vec<HealthCheck> {
    let mut checks = Vec::new();
    let home = home_dir();
    let tlaplus_dir = home.join(TLAPLUS_DIR_NAME);
    let tlaplus_examples_dir = home.join(TLAPLUS_EXAMPLES_DIR_NAME);
    let tytools_jar = tlaplus_dir.join(TYTOOLS_JAR_NAME);

    let untracked_before = git_untracked_files(project_root).await;

    checks.push(check_git_worktree_stable(project_root, env_vars).await);
    checks.push(check_exists(&project_root.join("Cargo.toml")));
    checks.push(check_spec_baseline(project_root));
    checks.push(check_baseline_provenance(project_root));
    checks.push(check_baseline_specs_digest(project_root));
    checks.push(check_baseline_drift(project_root, &tlaplus_examples_dir, &tytools_jar).await);
    checks.push(check_ay_pins(project_root));
    checks.push(check_spec_coverage_freshness(project_root).await);
    checks.push(check_script_pipefail(project_root));

    checks.push(check_parse_cargo_timings_ndjson_self_test());
    checks.push(check_parse_tlc_dot_smoke(project_root));
    checks.push(check_current_doc_routing(project_root));

    let untracked_after = git_untracked_files(project_root).await;
    if let (Some(before), Some(after)) = (untracked_before, untracked_after) {
        let removed: Vec<_> = before.difference(&after).cloned().collect();
        if removed.is_empty() {
            checks.push(HealthCheck::ok(
                "git:untracked_files_preserved",
                Some(format!("untracked={}", after.len())),
            ));
        } else {
            checks.push(HealthCheck::err(
                "git:untracked_files_preserved",
                Some(trim(&format!("removed:\n{}", removed.join("\n")), 350)),
            ));
        }
    }

    let metadata = run_command(
        &["cargo", "metadata", "--no-deps", "--format-version", "1"],
        project_root,
        20,
    )
    .await;
    let metadata_detail = if metadata.return_code == 0 {
        match serde_json::from_str::<Value>(&metadata.output) {
            Ok(value) => {
                let packages = value
                    .get("packages")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                let workspace_members = value
                    .get("workspace_members")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                format!("packages={packages} workspace_members={workspace_members}")
            }
            Err(err) => format!("ok (failed to parse json: {err})"),
        }
    } else {
        trim(&metadata.output, 400)
    };
    checks.push(HealthCheck {
        name: "cmd:cargo metadata".to_string(),
        ok: metadata.return_code == 0,
        level: None,
        detail: non_empty(metadata_detail),
    });

    checks
}

async fn run_command(command: &[&str], cwd: &Path, timeout_sec: u64) -> CommandOutput {
    if command.is_empty() {
        return CommandOutput {
            return_code: 127,
            output: "empty command".to_string(),
        };
    }

    let mut child = match Command::new(command[0])
        .args(&command[1..])
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return CommandOutput {
                return_code: 127,
                output: err.to_string(),
            };
        }
    };

    let stdout_task = child
        .stdout
        .take()
        .map(|stdout| tokio::spawn(read_stream(stdout)));
    let stderr_task = child
        .stderr
        .take()
        .map(|stderr| tokio::spawn(read_stream(stderr)));

    let wait = time::timeout(Duration::from_secs(timeout_sec), child.wait()).await;
    let return_code = match wait {
        Ok(Ok(status)) => status.code().unwrap_or(1),
        Ok(Err(err)) => {
            return CommandOutput {
                return_code: 1,
                output: err.to_string(),
            };
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            124
        }
    };

    let stdout = join_output_task(stdout_task).await;
    let stderr = join_output_task(stderr_task).await;
    let output = if return_code == 124 {
        format!("timeout after {timeout_sec}s")
    } else {
        combine_output(&stdout, &stderr)
    };

    CommandOutput {
        return_code,
        output,
    }
}

async fn read_stream<R>(mut reader: R) -> String
where
    R: AsyncRead + Unpin,
{
    let mut bytes = Vec::new();
    if reader.read_to_end(&mut bytes).await.is_err() {
        return String::new();
    }
    String::from_utf8_lossy(&bytes).trim().to_string()
}

async fn join_output_task(task: Option<tokio::task::JoinHandle<String>>) -> String {
    match task {
        Some(task) => task.await.unwrap_or_default(),
        None => String::new(),
    }
}

fn combine_output(stdout: &str, stderr: &str) -> String {
    let stdout = stdout.trim();
    let stderr = stderr.trim();
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => String::new(),
        (false, true) => stdout.to_string(),
        (true, false) => format!("STDERR:\n{stderr}"),
        (false, false) => format!("{stdout}\nSTDERR:\n{stderr}"),
    }
}

async fn git_untracked_files(project_root: &Path) -> Option<BTreeSet<String>> {
    let result = run_command(
        &["git", "ls-files", "--others", "--exclude-standard"],
        project_root,
        5,
    )
    .await;
    if result.return_code != 0 {
        return None;
    }
    Some(
        result
            .output
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

async fn git_status_porcelain(project_root: &Path) -> Option<String> {
    let result = run_command(&["git", "status", "--porcelain"], project_root, 5).await;
    (result.return_code == 0).then(|| result.output.trim_end_matches('\n').to_string())
}

async fn check_git_worktree_stable(
    project_root: &Path,
    _env_vars: &BTreeMap<String, String>,
) -> HealthCheck {
    let mut samples: Vec<Vec<String>> = Vec::new();
    for index in 0..2 {
        let Some(status) = git_status_porcelain(project_root).await else {
            return HealthCheck::err(
                "git:worktree_stable",
                Some("unable to run git status".into()),
            );
        };
        samples.push(disallowed_status_lines(&status, &["reports/"]));
        if index == 0 {
            time::sleep(Duration::from_millis(200)).await;
        }
    }

    if samples[0].is_empty() && samples[1].is_empty() {
        return HealthCheck::ok(
            "git:worktree_stable",
            Some("clean (reports/ allowed)".to_string()),
        );
    }

    if samples[0] == samples[1] {
        let paths = status_paths(samples[0].iter());
        return HealthCheck::warn(
            "git:worktree_stable",
            trim(
                &format!(
                    "dirty worktree ({} file(s))\n{}",
                    paths.len(),
                    paths
                        .iter()
                        .take(20)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n")
                ),
                350,
            ),
        );
    }

    let paths = status_paths(samples.iter().flatten());

    HealthCheck {
        name: "git:worktree_stable".to_string(),
        ok: false,
        level: Some("err".to_string()),
        detail: Some(trim(
            &format!(
                "unstable git status (concurrent mutation suspected)\n{}",
                paths
                    .iter()
                    .take(20)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            350,
        )),
    }
}

fn disallowed_status_lines(status_output: &str, allowed_prefixes: &[&str]) -> Vec<String> {
    status_output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter(|line| {
            let path = status_path(line);
            !allowed_prefixes
                .iter()
                .any(|prefix| path.starts_with(prefix))
        })
        .map(str::to_string)
        .collect()
}

fn status_paths<'a>(lines: impl Iterator<Item = &'a String>) -> BTreeSet<String> {
    lines.map(|line| status_path(line)).collect()
}

fn status_path(line: &str) -> String {
    let path = if line.len() >= 4 { &line[3..] } else { line };
    let path = path.split(" -> ").last().unwrap_or(path);
    path.trim().trim_matches('"').to_string()
}

fn check_exists(path: &Path) -> HealthCheck {
    if path.exists() {
        HealthCheck::ok(format!("exists:{}", path.display()), None)
    } else {
        HealthCheck::err(
            format!("exists:{}", path.display()),
            Some("missing".to_string()),
        )
    }
}

fn check_spec_baseline(project_root: &Path) -> HealthCheck {
    let path = spec_baseline_path(project_root);
    let Some(data) = read_json_file(&path) else {
        return HealthCheck::err(
            "tlc_baseline:spec_baseline.json",
            Some("missing".to_string()),
        );
    };
    match data.get("specs").and_then(Value::as_object) {
        Some(specs) => HealthCheck::ok(
            "tlc_baseline:spec_baseline.json",
            Some(format!("specs={}", specs.len())),
        ),
        None => HealthCheck::err(
            "tlc_baseline:spec_baseline.json",
            Some("expected dict at .specs".to_string()),
        ),
    }
}

fn check_baseline_provenance(project_root: &Path) -> HealthCheck {
    let path = spec_baseline_path(project_root);
    let Some(data) = read_json_file(&path) else {
        return HealthCheck::err("baseline_provenance", Some("baseline missing".to_string()));
    };
    let schema_version = data
        .get("schema_version")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    if schema_version == 0 {
        return HealthCheck::err(
            "baseline_provenance",
            Some("missing schema_version".to_string()),
        );
    }
    if schema_version < 2 {
        return HealthCheck::err(
            "baseline_provenance",
            Some(format!("schema_version={schema_version} (need >=2)")),
        );
    }

    let mut missing = Vec::new();
    if data
        .pointer("/inputs/examples_git/head")
        .and_then(Value::as_str)
        .is_none()
    {
        missing.push("inputs.examples_git.head");
    }
    if data
        .pointer("/tlc/tlc_version")
        .and_then(Value::as_str)
        .is_none()
    {
        missing.push("tlc.tlc_version");
    }
    if data
        .pointer("/tlc/jar_sha256")
        .and_then(Value::as_str)
        .is_none()
    {
        missing.push("tlc.jar_sha256");
    }
    if !missing.is_empty() {
        return HealthCheck::err(
            "baseline_provenance",
            Some(format!("missing: {}", missing.join(", "))),
        );
    }

    let tlc_version = data
        .pointer("/tlc/tlc_version")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .chars()
        .take(20)
        .collect::<String>();
    let examples_commit = data
        .pointer("/inputs/examples_git/head_short")
        .and_then(Value::as_str)
        .unwrap_or("?");
    HealthCheck::ok(
        "baseline_provenance",
        Some(format!("tlc={tlc_version}, examples={examples_commit}")),
    )
}

fn check_baseline_specs_digest(project_root: &Path) -> HealthCheck {
    let path = spec_baseline_path(project_root);
    let Some(data) = read_json_file(&path) else {
        return HealthCheck::err(
            "baseline_specs_digest",
            Some("baseline missing".to_string()),
        );
    };
    if data
        .get("schema_version")
        .and_then(Value::as_i64)
        .unwrap_or_default()
        < 2
    {
        return HealthCheck::ok(
            "baseline_specs_digest",
            Some("skipped (schema_version < 2)".to_string()),
        );
    }
    let Some(specs) = data.get("specs") else {
        return HealthCheck::err(
            "baseline_specs_digest",
            Some("missing/invalid .specs".to_string()),
        );
    };
    let Some(stored) = data.get("specs_jcs_sha256").and_then(Value::as_str) else {
        return HealthCheck::err(
            "baseline_specs_digest",
            Some("missing specs_jcs_sha256".to_string()),
        );
    };
    let computed = sha256_jcs(specs);
    if computed != stored {
        return HealthCheck::err("baseline_specs_digest", Some("digest mismatch".to_string()));
    }
    HealthCheck::ok(
        "baseline_specs_digest",
        Some(stored.chars().take(16).collect::<String>()),
    )
}

async fn check_baseline_drift(
    project_root: &Path,
    tlaplus_examples_dir: &Path,
    tytools_jar: &Path,
) -> HealthCheck {
    let path = spec_baseline_path(project_root);
    let Some(data) = read_json_file(&path) else {
        return HealthCheck::err("baseline_drift", Some("baseline missing".to_string()));
    };
    if data
        .get("schema_version")
        .and_then(Value::as_i64)
        .unwrap_or_default()
        < 2
    {
        return HealthCheck::ok(
            "baseline_drift",
            Some("skipped (no provenance)".to_string()),
        );
    }

    let mut warnings = Vec::new();
    if let Some(baseline_head) = data
        .pointer("/inputs/examples_git/head")
        .and_then(Value::as_str)
    {
        match git_head(tlaplus_examples_dir).await {
            None => warnings.push("examples: repo missing/not git".to_string()),
            Some(current) if current != baseline_head => warnings.push(format!(
                "examples: {} -> {}",
                short_hex(baseline_head, 8),
                short_hex(&current, 8)
            )),
            Some(_) => {}
        }
    }
    if let Some(baseline_sha) = data.pointer("/tlc/jar_sha256").and_then(Value::as_str) {
        if baseline_sha != "unknown" {
            match file_sha256(tytools_jar) {
                None => warnings.push("tlc: jar missing".to_string()),
                Some(current) if current != baseline_sha => warnings.push(format!(
                    "tlc: jar changed ({} -> {})",
                    short_hex(baseline_sha, 8),
                    short_hex(&current, 8)
                )),
                Some(_) => {}
            }
        }
    }

    if warnings.is_empty() {
        HealthCheck::ok("baseline_drift", Some("no drift".to_string()))
    } else {
        HealthCheck::err("baseline_drift", Some(warnings.join("; ")))
    }
}

async fn git_head(repo_path: &Path) -> Option<String> {
    if !repo_path.join(".git").exists() {
        return None;
    }
    let result = run_command(&["git", "rev-parse", "HEAD"], repo_path, 10).await;
    (result.return_code == 0).then(|| result.output.trim().to_string())
}

fn file_sha256(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Some(format!("{:x}", hasher.finalize()))
}

async fn check_spec_coverage_freshness(project_root: &Path) -> HealthCheck {
    check_spec_coverage_freshness_with_now(project_root, Utc::now()).await
}

async fn check_spec_coverage_freshness_with_now(
    project_root: &Path,
    now: DateTime<Utc>,
) -> HealthCheck {
    let name = "spec_coverage_freshness";
    let refresh = "cargo run --release --bin ty -- diagnose --output-metrics";
    let path = project_root.join("metrics/spec_coverage.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == ErrorKind::NotFound => {
            return HealthCheck::warn(name, format!("missing; run: {refresh}"));
        }
        Err(err) => {
            return HealthCheck::err(name, Some(format!("could not read spec coverage: {err}")));
        }
    };
    let data: Value = match serde_json::from_str(&text) {
        Ok(data) => data,
        Err(err) => return HealthCheck::err(name, Some(format!("invalid json: {err}"))),
    };
    let Some(generated_at) = data.get("generated_at").and_then(Value::as_str) else {
        return HealthCheck::err(name, Some("missing generated_at field".to_string()));
    };
    let gen_time = match DateTime::parse_from_rfc3339(generated_at) {
        Ok(time) => time.with_timezone(&Utc),
        Err(err) => return HealthCheck::err(name, Some(format!("bad generated_at: {err}"))),
    };
    let age_hours = (now - gen_time).num_seconds() as f64 / 3600.0;
    let git_commit = data
        .get("git_commit_short")
        .and_then(Value::as_str)
        .or_else(|| {
            data.pointer("/binary_info/git_commit")
                .and_then(Value::as_str)
        });
    let drift = match git_commit {
        Some(commit) => git_commit_distance(project_root, commit, "HEAD").await,
        None => None,
    };
    let age_level = threshold_level(age_hours, 24.0, 168.0);
    let drift_level = match (git_commit, drift) {
        (_, Some(value)) => threshold_level(value as f64, 100.0, 500.0),
        (Some(_), None) => "warn",
        (None, None) => "ok",
    };
    let combined = max_level(age_level, drift_level);
    let mut detail = match drift {
        Some(value) => format!("age={:.0}h, drift={value} commits", age_hours),
        None => format!("age={:.0}h, drift=unknown", age_hours),
    };
    if combined != "ok" {
        let _ = write!(detail, "; refresh: {refresh}");
    }
    HealthCheck {
        name: name.to_string(),
        ok: combined != "err",
        level: Some(combined.to_string()),
        detail: Some(detail),
    }
}

async fn git_commit_distance(
    project_root: &Path,
    old_commit: &str,
    new_ref: &str,
) -> Option<usize> {
    let range = format!("{old_commit}..{new_ref}");
    let result = run_command(&["git", "rev-list", "--count", &range], project_root, 10).await;
    if result.return_code != 0 {
        return None;
    }
    result.output.trim().parse().ok()
}

fn threshold_level(value: f64, warn: f64, err: f64) -> &'static str {
    if value >= err {
        "err"
    } else if value >= warn {
        "warn"
    } else {
        "ok"
    }
}

fn max_level(left: &str, right: &str) -> &'static str {
    if level_rank(left) >= level_rank(right) {
        match left {
            "err" => "err",
            "warn" => "warn",
            _ => "ok",
        }
    } else {
        match right {
            "err" => "err",
            "warn" => "warn",
            _ => "ok",
        }
    }
}

fn level_rank(level: &str) -> usize {
    match level {
        "err" => 2,
        "warn" => 1,
        _ => 0,
    }
}

fn check_script_pipefail(project_root: &Path) -> HealthCheck {
    let scripts_dir = project_root.join("scripts");
    if !scripts_dir.exists() {
        return HealthCheck::ok(
            "script_pipefail",
            Some("scripts/ not found (ok)".to_string()),
        );
    }
    let mut violations = Vec::new();
    let set_e = Regex::new(r"(?m)^\s*set\s+-[a-z]*e").expect("valid regex");
    if let Ok(entries) = std::fs::read_dir(&scripts_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("sh") {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            if set_e.is_match(&content) && !content.contains("pipefail") {
                if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                    violations.push(name.to_string());
                }
            }
        }
    }
    violations.sort();
    if violations.is_empty() {
        HealthCheck::ok("script_pipefail", Some("all scripts ok".to_string()))
    } else {
        let suffix = if violations.len() > 5 { "..." } else { "" };
        HealthCheck::err(
            "script_pipefail",
            Some(format!(
                "missing pipefail: {}{suffix}",
                violations
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        )
    }
}

fn check_current_doc_routing(project_root: &Path) -> HealthCheck {
    let failures = check_doc_text_guards(project_root, CURRENT_DOC_ROUTING_GUARDS);
    if failures.is_empty() {
        HealthCheck::ok(
            CURRENT_DOC_ROUTING_CHECK_NAME,
            Some("routing guards passed".to_string()),
        )
    } else {
        HealthCheck::err(
            CURRENT_DOC_ROUTING_CHECK_NAME,
            Some(trim(
                &format!(
                    "=== Current Doc Routing Failures ===\n  {}\n\nTotal failures: {}",
                    failures.join("\n  "),
                    failures.len()
                ),
                400,
            )),
        )
    }
}

fn check_doc_text_guards(project_root: &Path, guards: &[DocTextGuard]) -> Vec<String> {
    let mut failures = Vec::new();
    for guard in guards {
        let doc_path = project_root.join(guard.path);
        let content = match std::fs::read_to_string(&doc_path) {
            Ok(content) => content,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                failures.push(format!("{}: file not found", guard.path));
                continue;
            }
            Err(err) => {
                failures.push(format!("{}: read failed: {err}", guard.path));
                continue;
            }
        };
        for failure in check_guard_content(&content, guard) {
            failures.push(format!("{}: {failure}", guard.path));
        }
    }
    failures
}

fn check_guard_content(content: &str, guard: &DocTextGuard) -> Vec<String> {
    let mut failures = Vec::new();
    let normalized_content = normalize_whitespace(content);
    for required in guard.required_substrings {
        if !normalized_content.contains(&normalize_whitespace(required)) {
            failures.push(format!("missing required text: '{required}'"));
        }
    }
    for pattern in guard.forbidden_patterns {
        if RegexBuilder::new(pattern)
            .multi_line(true)
            .build()
            .expect("valid current-doc routing guard regex")
            .is_match(content)
        {
            failures.push(format!("matched forbidden pattern: {pattern}"));
        }
    }
    failures
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TlcState {
    fingerprint: i64,
    label: String,
    is_initial: bool,
    depth: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TlcTransition {
    src_fp: i64,
    dst_fp: i64,
    action: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TlcStateGraph {
    states: BTreeMap<i64, TlcState>,
    transitions: Vec<TlcTransition>,
    depth_groups: BTreeMap<usize, BTreeSet<i64>>,
    initial_states: BTreeSet<i64>,
}

fn check_parse_tlc_dot_smoke(project_root: &Path) -> HealthCheck {
    match parse_tlc_dot_smoke(project_root) {
        Ok(detail) => HealthCheck::ok(TLC_DOT_SMOKE_CHECK_NAME, Some(detail)),
        Err(detail) => HealthCheck::err(TLC_DOT_SMOKE_CHECK_NAME, Some(detail)),
    }
}

fn parse_tlc_dot_smoke(project_root: &Path) -> std::result::Result<String, String> {
    let fixture = project_root.join("test_data/tlc_dot/DieHard.dot");
    let graph = parse_tlc_dot(&fixture)?;

    if graph.initial_states.is_empty() {
        return Err("expected at least 1 initial state (style=filled)".to_string());
    }
    if graph.transitions.is_empty() {
        return Err("expected non-empty edge list".to_string());
    }

    let mut missing = Vec::new();
    for transition in &graph.transitions {
        if !graph.states.contains_key(&transition.src_fp) {
            missing.push(format!("src missing: {}", transition.src_fp));
        }
        if !graph.states.contains_key(&transition.dst_fp) {
            missing.push(format!("dst missing: {}", transition.dst_fp));
        }
        if !missing.is_empty() && missing.len() >= 5 {
            break;
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "edge endpoint(s) missing from nodes: {}",
            missing.join(", ")
        ));
    }

    for fp in &graph.initial_states {
        let Some(state) = graph.states.get(fp) else {
            return Err(format!("initial state missing from nodes: {fp}"));
        };
        if state.depth != Some(0) {
            return Err(format!(
                "expected initial state depth=0, got fp={fp} depth={}",
                format_tlc_depth(state.depth)
            ));
        }
    }

    Ok(format!(
        "OK parse_tlc_dot_smoke: states={} edges={} initials={}",
        graph.states.len(),
        graph.transitions.len(),
        graph.initial_states.len()
    ))
}

fn parse_tlc_dot(path: &Path) -> std::result::Result<TlcStateGraph, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    parse_tlc_dot_text(&text)
}

fn parse_tlc_dot_text(text: &str) -> std::result::Result<TlcStateGraph, String> {
    let node_re = Regex::new(r"^(-?\d+)\s+\[(.+)\];?$").expect("valid TLC DOT node regex");
    let edge_with_attrs_re =
        Regex::new(r"^(-?\d+)\s+->\s+(-?\d+)\s+\[(.+)\];?$").expect("valid TLC DOT edge regex");
    let edge_no_attrs_re =
        Regex::new(r"^(-?\d+)\s+->\s+(-?\d+)\s*;?$").expect("valid TLC DOT edge regex");
    let rank_re = Regex::new(r"^\{rank\s*=\s*same;\s*(.+)\}$").expect("valid TLC DOT rank regex");
    let mut states = BTreeMap::new();
    let mut transitions = Vec::new();
    let mut initial_states = BTreeSet::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || rank_re.is_match(line) {
            continue;
        }

        if let Some(captures) = node_re.captures(line) {
            let fp = parse_tlc_fingerprint(&captures[1])?;
            let attrs = &captures[2];
            let label = extract_tlc_quoted_attr(attrs, "label")?;
            let is_initial = (attrs.contains("style = filled") || attrs.contains("style=filled"))
                && !attrs.contains("fillcolor");
            if is_initial {
                initial_states.insert(fp);
            }
            states.insert(
                fp,
                TlcState {
                    fingerprint: fp,
                    label,
                    is_initial,
                    depth: None,
                },
            );
            continue;
        }

        if let Some(captures) = edge_with_attrs_re.captures(line) {
            let src_fp = parse_tlc_fingerprint(&captures[1])?;
            let dst_fp = parse_tlc_fingerprint(&captures[2])?;
            let attrs = &captures[3];
            let action = if attrs.contains("label=") {
                Some(extract_tlc_quoted_attr(attrs, "label")?)
            } else {
                None
            };
            transitions.push(TlcTransition {
                src_fp,
                dst_fp,
                action,
            });
            continue;
        }

        if let Some(captures) = edge_no_attrs_re.captures(line) {
            transitions.push(TlcTransition {
                src_fp: parse_tlc_fingerprint(&captures[1])?,
                dst_fp: parse_tlc_fingerprint(&captures[2])?,
                action: None,
            });
        }
    }

    let (depth_map, depth_groups) = compute_tlc_depths(&initial_states, &transitions);
    for state in states.values_mut() {
        state.depth = depth_map.get(&state.fingerprint).copied();
    }

    Ok(TlcStateGraph {
        states,
        transitions,
        depth_groups,
        initial_states,
    })
}

fn parse_tlc_fingerprint(raw: &str) -> std::result::Result<i64, String> {
    raw.parse::<i64>()
        .map_err(|err| format!("invalid TLC state fingerprint {raw:?}: {err}"))
}

fn extract_tlc_quoted_attr(attrs: &str, key: &str) -> std::result::Result<String, String> {
    let key_re = Regex::new(&format!(r"{}\s*=", regex::escape(key))).expect("valid attr key regex");
    let Some(found) = key_re.find(attrs) else {
        return Err(format!("Missing {key}= in attrs: {attrs}"));
    };
    let mut index = found.end();

    while let Some(ch) = attrs[index..].chars().next() {
        if !ch.is_whitespace() {
            break;
        }
        index += ch.len_utf8();
    }

    if !attrs[index..].starts_with('"') {
        return Err(format!(
            "Expected {key} to be a quoted string in attrs: {attrs}"
        ));
    }
    index += 1;

    let mut raw = String::new();
    while index < attrs.len() {
        let ch = attrs[index..]
            .chars()
            .next()
            .expect("index is on a char boundary");
        if ch == '"' {
            return Ok(tlc_dot_unescape(&raw));
        }
        if ch == '\\' {
            raw.push('\\');
            index += ch.len_utf8();
            if index < attrs.len() {
                let next = attrs[index..]
                    .chars()
                    .next()
                    .expect("index is on a char boundary");
                raw.push(next);
                index += next.len_utf8();
            }
            continue;
        }
        raw.push(ch);
        index += ch.len_utf8();
    }

    Err(format!(
        "Unterminated quoted string for {key}= in attrs: {attrs}"
    ))
}

fn tlc_dot_unescape(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }

        match chars.next() {
            Some('n') => out.push('\n'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some(next) => {
                out.push('\\');
                out.push(next);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn compute_tlc_depths(
    initial_states: &BTreeSet<i64>,
    transitions: &[TlcTransition],
) -> (BTreeMap<i64, usize>, BTreeMap<usize, BTreeSet<i64>>) {
    let mut adj: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for transition in transitions {
        adj.entry(transition.src_fp)
            .or_default()
            .push(transition.dst_fp);
    }

    let mut depth = BTreeMap::new();
    let mut queue = VecDeque::new();
    for fp in initial_states {
        if depth.contains_key(fp) {
            continue;
        }
        depth.insert(*fp, 0);
        queue.push_back(*fp);
    }

    while let Some(cur) = queue.pop_front() {
        let cur_depth = depth[&cur];
        for next in adj.get(&cur).into_iter().flatten() {
            if depth.contains_key(next) {
                continue;
            }
            depth.insert(*next, cur_depth + 1);
            queue.push_back(*next);
        }
    }

    let mut groups: BTreeMap<usize, BTreeSet<i64>> = BTreeMap::new();
    for (fp, state_depth) in &depth {
        groups.entry(*state_depth).or_default().insert(*fp);
    }

    (depth, groups)
}

fn format_tlc_depth(depth: Option<usize>) -> String {
    depth
        .map(|value| value.to_string())
        .unwrap_or_else(|| "None".to_string())
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn check_ay_pins(project_root: &Path) -> HealthCheck {
    let cargo_toml_path = project_root.join("Cargo.toml");
    let tla_ay_manifest_path = project_root.join("crates/tla-ay/Cargo.toml");
    if !cargo_toml_path.exists() {
        return HealthCheck::err("ay_pin:workspace", Some("missing Cargo.toml".to_string()));
    }
    if !tla_ay_manifest_path.exists() {
        return HealthCheck::err(
            "ay_pin:workspace",
            Some("missing crates/tla-ay/Cargo.toml".to_string()),
        );
    }
    let cargo_toml = match read_toml_file(&cargo_toml_path) {
        Ok(value) => value,
        Err(err) => {
            return HealthCheck::err("ay_pin:workspace", Some(format!("invalid toml: {err}")));
        }
    };

    let required = match required_workspace_ay_deps(&cargo_toml, project_root) {
        Ok(required) => required,
        Err(err) => return HealthCheck::err("ay_pin:workspace", Some(err)),
    };
    if required.is_empty() {
        return HealthCheck::err(
            "ay_pin:workspace",
            Some("no workspace ay deps found".to_string()),
        );
    }

    let ay_specs = match workspace_ay_specs(&cargo_toml, &required) {
        Ok(specs) => specs,
        Err(err) => return HealthCheck::err("ay_pin:workspace", Some(err)),
    };
    let rev = match validate_workspace_ay_specs(&ay_specs) {
        Ok(rev) => rev,
        Err(err) => return HealthCheck::err("ay_pin:workspace", Some(err)),
    };
    let lock_sources = match load_required_ay_lock_sources(project_root, &required) {
        Ok(sources) => sources,
        Err(err) => return HealthCheck::err("ay_pin:workspace", Some(err)),
    };
    if let Err(err) = validate_required_ay_lock_sources(&lock_sources, &rev) {
        return HealthCheck::err("ay_pin:workspace", Some(err));
    }

    HealthCheck::ok(
        "ay_pin:workspace",
        Some(format!(
            "rev={} ({} deps)",
            short_hex(&rev, 12),
            required.len()
        )),
    )
}

fn read_toml_file(path: &Path) -> Result<toml::Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    text.parse::<toml::Value>()
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn required_workspace_ay_deps(
    cargo_toml: &toml::Value,
    project_root: &Path,
) -> std::result::Result<BTreeSet<String>, String> {
    let manifests = workspace_member_manifest_paths(cargo_toml, project_root)?;
    let mut required = BTreeSet::new();
    for manifest_path in manifests {
        let manifest = read_toml_file(&manifest_path).map_err(|err| {
            format!(
                "invalid {}: {err}",
                relative_display(&manifest_path, project_root)
            )
        })?;
        required.extend(manifest_workspace_ay_deps(&manifest));
    }
    Ok(required)
}

fn workspace_member_manifest_paths(
    cargo_toml: &toml::Value,
    project_root: &Path,
) -> std::result::Result<Vec<PathBuf>, String> {
    let workspace = cargo_toml
        .get("workspace")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| "workspace table missing".to_string())?;
    let members = workspace
        .get("members")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| "workspace.members missing".to_string())?;
    let mut manifests = BTreeSet::new();
    for member in members {
        let member = member
            .as_str()
            .ok_or_else(|| format!("invalid workspace member: {member:?}"))?;
        let matches = expand_workspace_member(project_root, member);
        if matches.is_empty() {
            return Err(format!("workspace member matched nothing: {member}"));
        }
        for matched in matches {
            let manifest_path =
                if matched.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml") {
                    matched
                } else {
                    matched.join("Cargo.toml")
                };
            if !manifest_path.exists() {
                return Err(format!(
                    "missing workspace member manifest: {}",
                    relative_display(&manifest_path, project_root)
                ));
            }
            manifests.insert(manifest_path);
        }
    }
    Ok(manifests.into_iter().collect())
}

fn expand_workspace_member(project_root: &Path, member: &str) -> Vec<PathBuf> {
    if !member.contains('*') {
        return vec![project_root.join(member)];
    }
    let Some((prefix, suffix)) = member.split_once('*') else {
        return vec![project_root.join(member)];
    };
    let base = project_root.join(prefix.trim_end_matches('/'));
    let suffix = suffix.trim_start_matches('/');
    let Ok(entries) = std::fs::read_dir(base) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path().join(suffix))
        .filter(|path| path.exists())
        .collect()
}

fn manifest_workspace_ay_deps(manifest: &toml::Value) -> BTreeSet<String> {
    let mut required = BTreeSet::new();
    for section in AY_DEP_SECTIONS {
        collect_ay_deps(manifest.get(*section), &mut required);
    }
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for target in targets.values() {
            for section in AY_DEP_SECTIONS {
                collect_ay_deps(target.get(*section), &mut required);
            }
        }
    }
    required
}

fn collect_ay_deps(section: Option<&toml::Value>, required: &mut BTreeSet<String>) {
    let Some(table) = section.and_then(toml::Value::as_table) else {
        return;
    };
    for (name, spec) in table {
        if (name == "ay" || name.starts_with("ay-"))
            && spec
                .as_table()
                .and_then(|table| table.get("workspace"))
                .and_then(toml::Value::as_bool)
                == Some(true)
        {
            required.insert(name.clone());
        }
    }
}

fn workspace_ay_specs(
    cargo_toml: &toml::Value,
    required: &BTreeSet<String>,
) -> std::result::Result<BTreeMap<String, toml::value::Table>, String> {
    let deps = cargo_toml
        .get("workspace")
        .and_then(|value| value.get("dependencies"))
        .and_then(toml::Value::as_table)
        .ok_or_else(|| "workspace.dependencies missing".to_string())?;
    let missing = required
        .iter()
        .filter(|name| !deps.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!("missing deps: {}", missing.join(", ")));
    }
    let mut specs = BTreeMap::new();
    for name in required {
        let Some(spec) = deps.get(name).and_then(toml::Value::as_table) else {
            return Err(format!("{name} is not a git dependency table"));
        };
        specs.insert(name.clone(), spec.clone());
    }
    Ok(specs)
}

fn validate_workspace_ay_specs(
    specs: &BTreeMap<String, toml::value::Table>,
) -> std::result::Result<String, String> {
    let rev_re = Regex::new(r"^[0-9a-f]{40}$").expect("valid regex");
    let mut bad_urls = Vec::new();
    let mut invalid_revs = Vec::new();
    let mut revs = BTreeSet::new();
    for (name, spec) in specs {
        let git_url = spec.get("git").and_then(toml::Value::as_str);
        if git_url.and_then(normalize_github_repo_identity)
            != Some(CANONICAL_AY_REPO_ID.to_string())
        {
            bad_urls.push(format!("{name}={git_url:?}"));
        }
        let rev = spec.get("rev").and_then(toml::Value::as_str);
        match rev {
            Some(rev) if rev_re.is_match(rev) => {
                revs.insert(rev.to_string());
            }
            _ => invalid_revs.push(format!("{name}={rev:?}")),
        }
    }
    if !bad_urls.is_empty() {
        return Err(format!("non-canonical ay url(s): {}", bad_urls.join(", ")));
    }
    if !invalid_revs.is_empty() {
        return Err(format!("invalid ay rev(s): {}", invalid_revs.join(", ")));
    }
    if revs.len() != 1 {
        let detail = specs
            .iter()
            .map(|(name, spec)| {
                format!(
                    "{name}={}",
                    spec.get("rev")
                        .and_then(toml::Value::as_str)
                        .map(|rev| short_hex(rev, 8))
                        .unwrap_or_else(|| "?".to_string())
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("mismatched revs ({detail})"));
    }
    Ok(revs.into_iter().next().expect("one rev"))
}

fn load_required_ay_lock_sources(
    project_root: &Path,
    required: &BTreeSet<String>,
) -> std::result::Result<BTreeMap<String, String>, String> {
    let lock_path = project_root.join("Cargo.lock");
    if !lock_path.exists() {
        return Err("missing Cargo.lock".to_string());
    }
    let lock = read_toml_file(&lock_path).map_err(|err| format!("invalid Cargo.lock: {err}"))?;
    let packages = lock
        .get("package")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| "Cargo.lock missing package list".to_string())?;
    let mut sources = BTreeMap::new();
    for package in packages {
        let Some(table) = package.as_table() else {
            continue;
        };
        let Some(name) = table.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let Some(source) = table.get("source").and_then(toml::Value::as_str) else {
            continue;
        };
        if required.contains(name) {
            sources.insert(name.to_string(), source.to_string());
        }
    }
    let missing = required
        .iter()
        .filter(|name| !sources.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Cargo.lock missing ay package(s): {}",
            missing.join(", ")
        ));
    }
    Ok(sources)
}

fn validate_required_ay_lock_sources(
    sources: &BTreeMap<String, String>,
    expected_rev: &str,
) -> std::result::Result<(), String> {
    let mut bad_urls = Vec::new();
    let mut lock_revs = BTreeSet::new();
    for (name, source) in sources {
        if normalize_github_repo_identity(source) != Some(CANONICAL_AY_REPO_ID.to_string()) {
            bad_urls.push(format!("{name}={source:?}"));
            continue;
        }
        let Some(rev) = parse_lock_source_rev(source) else {
            return Err(format!("Cargo.lock missing ay rev for {name}"));
        };
        lock_revs.insert(rev);
    }
    if !bad_urls.is_empty() {
        return Err(format!(
            "Cargo.lock has non-canonical ay source(s): {}",
            bad_urls.join(", ")
        ));
    }
    if lock_revs.len() != 1 {
        return Err(format!(
            "Cargo.lock has multiple ay revs ({})",
            lock_revs
                .iter()
                .map(|rev| short_hex(rev, 8))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let actual = lock_revs.into_iter().next().expect("one lock rev");
    if actual != expected_rev {
        return Err(format!(
            "Cargo.lock rev {} != workspace {}",
            short_hex(&actual, 12),
            short_hex(expected_rev, 12)
        ));
    }
    Ok(())
}

fn normalize_github_repo_identity(url: &str) -> Option<String> {
    let base = url
        .trim_start_matches("git+")
        .split(['?', '#'])
        .next()
        .unwrap_or(url);
    let patterns = [
        r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$",
        r"^ssh://git@github\.com/([^/]+)/([^/]+?)(?:\.git)?$",
        r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$",
    ];
    for pattern in patterns {
        let regex = Regex::new(pattern).expect("valid regex");
        if let Some(captures) = regex.captures(base) {
            return Some(format!(
                "{}/{}",
                "github.com",
                [&captures[1], &captures[2]].join("/")
            ));
        }
    }
    None
}

fn parse_lock_source_rev(source: &str) -> Option<String> {
    let regex = Regex::new(r"[?&]rev=([0-9a-f]{40})").expect("valid regex");
    regex
        .captures(source)
        .and_then(|captures| captures.get(1))
        .map(|capture| capture.as_str().to_string())
}

fn read_json_file(path: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn spec_baseline_path(project_root: &Path) -> PathBuf {
    project_root.join("tests/tlc_comparison/spec_baseline.json")
}

fn sha256_jcs(value: &Value) -> String {
    let canonical = dumps_canonical(value);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn dumps_canonical(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => serde_json::to_string(text).expect("string serializes"),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(dumps_canonical)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            let fields = keys
                .into_iter()
                .map(|key| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("key serializes"),
                        dumps_canonical(&map[key])
                    )
                })
                .collect::<Vec<_>>();
            format!("{{{}}}", fields.join(","))
        }
    }
}

fn check_level(check: &HealthCheck) -> &str {
    check
        .level
        .as_deref()
        .unwrap_or(if check.ok { "ok" } else { "err" })
}

fn format_check_line(check: &HealthCheck) -> String {
    let prefix = match check_level(check) {
        "ok" => "OK ",
        "warn" => "WARN",
        _ => "ERR",
    };
    match &check.detail {
        Some(detail) => format!("{prefix} {} ({detail})", check.name),
        None => format!("{prefix} {}", check.name),
    }
}

async fn manifest(project_root: &Path, checks: &[HealthCheck]) -> HealthManifest {
    let levels = checks.iter().map(check_level).collect::<Vec<_>>();
    let errors = levels.iter().filter(|level| **level == "err").count();
    let warnings = levels.iter().filter(|level| **level == "warn").count();
    let passed = levels.iter().filter(|level| **level == "ok").count();
    let status = if errors != 0 {
        "fail"
    } else if warnings != 0 {
        "warn"
    } else {
        "pass"
    };
    HealthManifest {
        schema_version: "1.0",
        generated_at: Utc::now().to_rfc3339(),
        git_commit: git_commit(project_root).await,
        project: project_root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string(),
        summary: HealthSummary {
            status: status.to_string(),
            passed,
            warnings,
            errors,
        },
        checks: checks
            .iter()
            .map(|check| ManifestCheck {
                name: check.name.clone(),
                ok: check.ok,
                level: check_level(check).to_string(),
                detail: check.detail.clone(),
            })
            .collect(),
    }
}

async fn git_commit(project_root: &Path) -> String {
    let result = run_command(&["git", "rev-parse", "HEAD"], project_root, 5).await;
    if result.return_code != 0 {
        "unknown".to_string()
    } else {
        short_hex(result.output.trim(), 12)
    }
}

fn trim(output: &str, max_chars: usize) -> String {
    let output = output.trim();
    if output.chars().count() <= max_chars {
        return output.to_string();
    }
    let mut trimmed = output.chars().take(max_chars).collect::<String>();
    trimmed = trimmed.trim_end().to_string();
    format!("{trimmed}\n... (truncated)")
}

fn non_empty(text: String) -> Option<String> {
    (!text.is_empty()).then_some(text)
}

fn short_hex(text: &str, len: usize) -> String {
    text.chars().take(len).collect()
}

fn relative_display(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
#[path = "cmd_system_health_gate_tests.rs"]
mod tests;
