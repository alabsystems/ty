// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLC baseline collector — Rust port of `scripts/collect_tlc_baseline/`.
//!
//! Runs the Java TLC baseline tool (`~/tlaplus/tytools.jar`) against
//! every spec listed in `tests/tlc_comparison/spec_catalog.py`, parses
//! TLC stdout/stderr, classifies the verdict, and emits or refreshes
//! `tests/tlc_comparison/spec_baseline.json` (schema v3).
//!
//! Provenance fields (TLC version, jar SHA-256, examples-repo git
//! head, etc.) are recorded deterministically so consumers like
//! `system_health_check` can detect baseline drift. The `specs` map is
//! canonicalized via JCS-style ordering and a SHA-256 digest is
//! recorded in `specs_jcs_sha256` so byte-for-byte formatting drift
//! is detectable without re-running TLC.
//!
//! This is the single compiler-enforced interface for baseline
//! collection — the Python package
//! `scripts/collect_tlc_baseline/` it replaces has been deleted in
//! the same commit.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

// ---------- Constants ----------

const SCHEMA_VERSION: u64 = 3;
const DEFAULT_TIMEOUT_SECONDS: u64 = 600;
const STATS_KEY_ORDER: &[&str] = &[
    "tlc_pass",
    "tlc_error",
    "tlc_timeout",
    "ty_match",
    "ty_mismatch",
    "ty_fail",
    "ty_untested",
];
const CATEGORIES_KEY_ORDER: &[&str] = &["small", "medium", "large", "xlarge", "skip", "unknown"];
const TLC_ENTRY_KEY_ORDER: &[&str] =
    &["status", "states", "runtime_seconds", "error_type", "error"];
const TY_ENTRY_KEY_ORDER: &[&str] = &["status", "states", "error_type", "last_run", "git_commit"];
const SPEC_ENTRY_KEY_ORDER: &[&str] =
    &["tlc", "ty", "verified_match", "issue", "category", "source"];
const LEGACY_V2_KEYS: &[&str] = &[
    "expected_states",
    "tlc_runtime_seconds",
    "status",
    "error",
    "error_type",
];

// ---------- CLI ----------

#[derive(Parser, Debug)]
#[command(
    name = "ty-tlc-baseline",
    about = "Collect TLC baselines for every spec listed in spec_catalog.py",
    long_about = "Rust replacement for scripts/collect_tlc_baseline/. Runs TLC against \
                  every spec in tests/tlc_comparison/spec_catalog.py and writes \
                  tests/tlc_comparison/spec_baseline.json (schema v3) with full \
                  provenance and a JCS digest of the specs map."
)]
struct Cli {
    /// Timeout per spec in seconds.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECONDS)]
    timeout: u64,

    /// Do not reuse the existing baseline file as a resume cache.
    #[arg(long)]
    no_resume: bool,

    /// Override `spec_catalog.py` path. Defaults to
    /// `tests/tlc_comparison/spec_catalog.py` under the project root.
    #[arg(long, value_name = "PATH")]
    spec_catalog: Option<PathBuf>,

    /// Override `spec_baseline.json` output path.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Override TLA+ examples specifications directory.
    /// Defaults to `~/tlaplus-examples/specifications`.
    #[arg(long, value_name = "PATH")]
    examples_dir: Option<PathBuf>,

    /// Override `tytools.jar` path. Defaults to `~/tlaplus/tytools.jar`.
    #[arg(long, value_name = "PATH")]
    tlc_jar: Option<PathBuf>,

    /// Override `CommunityModules.jar` path. Defaults to
    /// `~/tlaplus/CommunityModules.jar`.
    #[arg(long, value_name = "PATH")]
    community_modules: Option<PathBuf>,

    /// Override the `~/tlaplus` git repo path (provenance only).
    #[arg(long, value_name = "PATH")]
    tlaplus_dir: Option<PathBuf>,

    /// Override the `~/tlaplus-examples` git repo path (provenance only).
    #[arg(long, value_name = "PATH")]
    examples_base_dir: Option<PathBuf>,

    /// Override the project root. Defaults to the current working directory.
    #[arg(long, value_name = "PATH")]
    project_root: Option<PathBuf>,
}

// ---------- Catalog model ----------

#[derive(Debug, Clone)]
struct SpecInfo {
    name: String,
    tla_path: String,
    cfg_path: String,
}

/// Parse `spec_catalog.py` for `SpecInfo("name", "tla_path", "cfg_path", ...)` rows.
///
/// We only care about the first three string positional arguments; any
/// remaining keyword/positional arguments are ignored. We do **not**
/// invoke a Python interpreter — the catalog is a flat list of
/// literal-string constructor calls and is parseable with a small state
/// machine.
fn parse_spec_catalog(text: &str) -> Result<Vec<SpecInfo>> {
    let mut specs = Vec::new();
    let mut in_all_specs = false;
    for raw in text.lines() {
        let trimmed = raw.trim();
        if !in_all_specs {
            if trimmed.starts_with("ALL_SPECS") && trimmed.contains('[') {
                in_all_specs = true;
            }
            continue;
        }
        if trimmed.starts_with("LARGE_SPECS")
            || trimmed.starts_with("KNOWN_BLOCKERS")
            || trimmed.starts_with("TY_LIMITATIONS")
            || trimmed.starts_with("TY_BUGS")
            || trimmed.starts_with("def ")
        {
            break;
        }
        if !trimmed.starts_with("SpecInfo(") {
            continue;
        }
        let inner = trimmed
            .trim_start_matches("SpecInfo(")
            .trim_end_matches(',')
            .trim_end_matches(')')
            .trim_end_matches(',')
            .trim_end_matches(')')
            .trim();
        let strings = parse_python_strings(inner, 3)?;
        if strings.len() < 3 {
            bail!("spec_catalog row missing required positional arguments: {trimmed}");
        }
        specs.push(SpecInfo {
            name: strings[0].clone(),
            tla_path: strings[1].clone(),
            cfg_path: strings[2].clone(),
        });
    }
    if specs.is_empty() {
        bail!("no SpecInfo rows parsed from spec_catalog.py — wrong file?");
    }
    Ok(specs)
}

/// Pull the first `n` Python-style string literals from `text`.
///
/// Recognizes both single- and double-quoted strings with `\\` and
/// `\"`/`\'` escapes — enough for the catalog rows which only use
/// double-quoted ASCII paths.
fn parse_python_strings(text: &str, n: usize) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(n);
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() && out.len() < n {
        let c = bytes[i];
        if c == b'"' || c == b'\'' {
            let quote = c;
            i += 1;
            let mut buf = String::new();
            while i < bytes.len() {
                let b = bytes[i];
                if b == b'\\' && i + 1 < bytes.len() {
                    let esc = bytes[i + 1];
                    match esc {
                        b'n' => buf.push('\n'),
                        b't' => buf.push('\t'),
                        b'\\' => buf.push('\\'),
                        b'\'' => buf.push('\''),
                        b'"' => buf.push('"'),
                        other => {
                            buf.push('\\');
                            buf.push(other as char);
                        }
                    }
                    i += 2;
                    continue;
                }
                if b == quote {
                    i += 1;
                    out.push(buf);
                    break;
                }
                buf.push(b as char);
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    Ok(out)
}

// ---------- TLC execution ----------

#[derive(Debug, Clone)]
struct TlcOutcome {
    status: String,
    states: Option<u64>,
    runtime_seconds: Option<f64>,
    error_type: Option<String>,
    error: Option<String>,
}

fn run_tlc(
    spec_path: &Path,
    cfg_path: &Path,
    timeout_seconds: u64,
    tlc_jar: &Path,
    community_modules: &Path,
    project_root: &Path,
) -> TlcOutcome {
    let mut outcome = TlcOutcome {
        status: "unknown".into(),
        states: None,
        runtime_seconds: None,
        error_type: None,
        error: None,
    };

    let use_ephemeral_metadir = std::env::var("TY_KEEP_STATES")
        .map(|v| v.trim() != "1")
        .unwrap_or(true);
    let preserve_states_dir = std::env::var("TY_PRESERVE_STATES_DIR")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let states_dir = spec_path
        .parent()
        .map(|p| p.join("states"))
        .unwrap_or_else(|| PathBuf::from("states"));

    let metadir_root = std::env::var_os("TY_TLC_METADIR_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root.join("target").join("tlc_metadir"));
    if use_ephemeral_metadir {
        if let Err(e) = fs::create_dir_all(&metadir_root) {
            outcome.status = "error".into();
            outcome.error = Some(format!(
                "failed to create TLC metadir root {}: {e}",
                metadir_root.display()
            ));
            outcome.error_type = Some("metadir_setup".into());
            return outcome;
        }
    }

    let metadir = if use_ephemeral_metadir {
        match tempfile::Builder::new()
            .prefix("tlc-")
            .tempdir_in(&metadir_root)
        {
            Ok(td) => Some(td),
            Err(e) => {
                outcome.status = "error".into();
                outcome.error = Some(format!("failed to create TLC metadir: {e}"));
                outcome.error_type = Some("metadir_setup".into());
                return outcome;
            }
        }
    } else {
        None
    };

    let mut classpath = OsString::from(tlc_jar.as_os_str());
    if community_modules.exists() {
        #[cfg(unix)]
        classpath.push(":");
        #[cfg(not(unix))]
        classpath.push(";");
        classpath.push(community_modules.as_os_str());
    }

    let mut cmd = Command::new("java");
    cmd.arg("-Xmx4g").arg("-cp").arg(&classpath).arg("tlc2.TLC");
    if let Some(md) = metadir.as_ref() {
        cmd.arg("-metadir").arg(md.path());
    }
    cmd.arg("-config")
        .arg(cfg_path)
        .arg("-workers")
        .arg("1")
        .arg(spec_path);
    if let Some(parent) = spec_path.parent() {
        cmd.current_dir(parent);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let start = Instant::now();
    let result = run_with_timeout(cmd, Duration::from_secs(timeout_seconds));

    drop(metadir); // remove the tempdir before cleaning ./states

    if use_ephemeral_metadir && !preserve_states_dir {
        cleanup_states_dir(&states_dir);
    }

    match result {
        Ok((code, timed_out, stdout, stderr)) => {
            let elapsed = start.elapsed().as_secs_f64();
            outcome.runtime_seconds = Some(round2(elapsed));
            if timed_out {
                outcome.status = "timeout".into();
                outcome.runtime_seconds = Some(timeout_seconds as f64);
                outcome.error = Some(format!("Timeout after {timeout_seconds}s"));
                outcome.error_type = Some("timeout".into());
                return outcome;
            }
            let combined = format!("{stdout}{stderr}");
            let (states, parse_err) = parse_tlc_output(&combined);
            outcome.states = states;
            outcome.error_type = classify_error(&combined);

            if states.is_some() {
                outcome.status = "pass".into();
            } else if let Some(err) = parse_err.as_deref() {
                if let Some(module) = err.strip_prefix("missing_module:") {
                    outcome.status = "error".into();
                    outcome.error_type = Some("missing_module".into());
                    outcome.error = Some(format!("Missing module: {module}"));
                } else if code != 0 && combined.contains("Exception") {
                    outcome.status = "error".into();
                } else {
                    outcome.status = "error".into();
                    outcome.error = Some(err.to_string());
                }
            } else {
                outcome.status = "unknown".into();
                outcome.error = Some("No state count found in output".into());
            }

            if matches!(outcome.status.as_str(), "error" | "unknown") && outcome.error.is_none() {
                for line in combined.lines() {
                    if line.contains("Error:") || line.contains("Exception") {
                        let mut s = line.to_string();
                        if s.len() > 200 {
                            s.truncate(200);
                        }
                        outcome.error = Some(s);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            outcome.status = "error".into();
            let mut msg = format!("{e}");
            if msg.len() > 200 {
                msg.truncate(200);
            }
            outcome.error = Some(msg);
        }
    }

    outcome
}

fn parse_tlc_output(output: &str) -> (Option<u64>, Option<String>) {
    // Pattern 1: "N states generated, M distinct states found, ..."
    for line in output.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_digit() || c == ',') {
            // backtrack — easier to scan with explicit search
            let _ = rest;
        }
        if let Some((Some(distinct), _rest)) = split_states_generated_distinct_left(
            trimmed,
            "states generated,",
            "distinct states found,",
        ) {
            return (Some(distinct), None);
        }
    }

    // Pattern 2: "N distinct states found"
    if let Some(value) = parse_last_count_before(output, "distinct states found") {
        return (Some(value), None);
    }

    // Pattern 3: "Cannot find source file for module Foo"
    if let Some(idx) = output.find("Cannot find source file for module ") {
        let tail = &output[idx + "Cannot find source file for module ".len()..];
        let module: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !module.is_empty() {
            return (None, Some(format!("missing_module:{module}")));
        }
    }

    (None, Some("no_state_count".into()))
}

/// Match `"N states generated, M distinct states found, L states left"` and
/// return the *distinct* count (M).
fn split_states_generated_distinct_left(
    line: &str,
    sep_generated: &str,
    sep_distinct: &str,
) -> Option<(Option<u64>, ())> {
    let gen_idx = line.find(sep_generated)?;
    let after_gen = &line[gen_idx + sep_generated.len()..];
    let dist_idx = after_gen.find(sep_distinct)?;
    let token = after_gen[..dist_idx].trim();
    let cleaned: String = token.chars().filter(|c| *c != ',').collect();
    let value = cleaned.parse::<u64>().ok();
    Some((value, ()))
}

fn parse_last_count_before(output: &str, marker: &str) -> Option<u64> {
    let mut found = None;
    let mut search_start = 0;
    while let Some(idx) = output[search_start..].find(marker) {
        let absolute = search_start + idx;
        let head = &output[..absolute];
        let trimmed = head.trim_end();
        let digit_start = trimmed
            .char_indices()
            .rev()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == ','))
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        let token = &trimmed[digit_start..];
        let cleaned: String = token.chars().filter(|c| *c != ',').collect();
        if let Ok(value) = cleaned.parse::<u64>() {
            found = Some(value);
        }
        search_start = absolute + marker.len();
    }
    found
}

fn classify_error(output: &str) -> Option<String> {
    if !output.contains("Error:") {
        return None;
    }
    if output.contains("Invariant") && output.contains("violated") {
        return Some("invariant".into());
    }
    if output.contains("Deadlock") {
        return Some("deadlock".into());
    }
    if output.contains("Parsing or semantic analysis failed") {
        return Some("parse".into());
    }
    if output.contains("Temporal properties were violated") {
        return Some("liveness".into());
    }
    if output.contains("Action property") && output.contains("violated") {
        return Some("action".into());
    }
    // "Property <name> is violated"
    for line in output.lines() {
        if line.contains("Property ") && line.contains(" is violated") {
            return Some("safety".into());
        }
    }
    Some("unknown".into())
}

fn cleanup_states_dir(states_dir: &Path) {
    // Safety guard: only ever remove a directory named "states" under a spec dir.
    if states_dir.file_name().and_then(|n| n.to_str()) != Some("states") {
        return;
    }
    if states_dir.is_symlink() {
        let _ = fs::remove_file(states_dir);
    } else if states_dir.is_dir() {
        let _ = fs::remove_dir_all(states_dir);
    }
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<(i32, bool, String, String)> {
    let mut child = cmd.spawn().context("spawn java tlc child")?;
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
                    return Ok((-1, true, stdout, stderr));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

fn categorize_runtime(seconds: Option<f64>) -> &'static str {
    match seconds {
        None => "unknown",
        Some(s) if s < 1.0 => "small",
        Some(s) if s < 30.0 => "medium",
        Some(s) if s < 300.0 => "large",
        Some(_) => "xlarge",
    }
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

// ---------- Provenance ----------

fn build_provenance(timeout_seconds: u64, ctx: &PathContext) -> Map<String, Value> {
    let has_community = ctx.community_modules.exists();
    let mut collector = Map::new();
    collector.insert(
        "ty_git_commit".into(),
        Value::String(git_short_head(&ctx.project_root)),
    );
    collector.insert(
        "script".into(),
        Value::String("crates/tla-petri/src/bin/ty-tlc-baseline.rs".into()),
    );

    let mut tlc = Map::new();
    tlc.insert(
        "jar_path".into(),
        Value::String(ctx.tlc_jar.display().to_string()),
    );
    tlc.insert(
        "jar_sha256".into(),
        Value::String(sha256_file(&ctx.tlc_jar)),
    );
    tlc.insert(
        "community_modules_path".into(),
        if has_community {
            Value::String(ctx.community_modules.display().to_string())
        } else {
            Value::Null
        },
    );
    tlc.insert(
        "community_modules_sha256".into(),
        if has_community {
            Value::String(sha256_file(&ctx.community_modules))
        } else {
            Value::Null
        },
    );
    tlc.insert(
        "tlc_version".into(),
        Value::String(tlc_version(&ctx.tlc_jar)),
    );
    tlc.insert("java_version".into(), Value::String(java_version()));
    tlc.insert("jvm_args".into(), json!(["-Xmx4g"]));
    tlc.insert("workers".into(), json!(1));

    let mut inputs = Map::new();
    inputs.insert(
        "examples_dir".into(),
        Value::String(ctx.examples_dir.display().to_string()),
    );
    inputs.insert("examples_git".into(), git_info(&ctx.examples_base_dir));
    inputs.insert("tlaplus_git".into(), git_info(&ctx.tlaplus_dir));

    let mut seed = Map::new();
    seed.insert("enabled".into(), Value::Bool(false));
    seed.insert("policy".into(), Value::String("no_seed".into()));
    seed.insert("source_path".into(), Value::Null);

    let mut prov = Map::new();
    prov.insert("schema_version".into(), json!(SCHEMA_VERSION));
    prov.insert("collector".into(), Value::Object(collector));
    prov.insert("tlc".into(), Value::Object(tlc));
    prov.insert("inputs".into(), Value::Object(inputs));
    prov.insert("seed".into(), Value::Object(seed));
    prov.insert("tlc_timeout_seconds".into(), json!(timeout_seconds));
    prov
}

fn git_info(repo: &Path) -> Value {
    let mut out = Map::new();
    out.insert("head".into(), Value::String("unknown".into()));
    out.insert("head_short".into(), Value::String("unknown".into()));
    out.insert("is_dirty".into(), Value::Null);
    out.insert("status_porcelain_sha256".into(), Value::Null);

    if !repo.exists() || !repo.join(".git").exists() {
        return Value::Object(out);
    }
    if let Some(head) = git_capture(repo, &["rev-parse", "HEAD"]) {
        out.insert("head".into(), Value::String(head));
    }
    if let Some(short) = git_capture(repo, &["rev-parse", "--short", "HEAD"]) {
        out.insert("head_short".into(), Value::String(short));
    }
    if let Some(status) = git_capture_raw(repo, &["status", "--porcelain=v1"]) {
        let is_dirty = !status.trim().is_empty();
        out.insert("is_dirty".into(), Value::Bool(is_dirty));
        let mut hasher = Sha256::new();
        hasher.update(status.as_bytes());
        let digest = format!("{:x}", hasher.finalize());
        out.insert(
            "status_porcelain_sha256".into(),
            Value::String(digest[..16.min(digest.len())].to_string()),
        );
    }
    Value::Object(out)
}

fn git_capture(repo: &Path, args: &[&str]) -> Option<String> {
    let out = git_capture_raw(repo, args)?;
    Some(out.trim().to_string())
}

fn git_capture_raw(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn git_short_head(repo: &Path) -> String {
    git_capture(repo, &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into())
}

fn sha256_file(path: &Path) -> String {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return "unknown".into(),
    };
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!("{:x}", hasher.finalize())
}

fn tlc_version(jar: &Path) -> String {
    if !jar.exists() {
        return "unknown".into();
    }
    let output = match Command::new("java")
        .arg("-jar")
        .arg(jar)
        .arg("-version")
        .output()
    {
        Ok(o) => o,
        Err(_) => return "unknown".into(),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if let Some(v) = extract_tlc_version(&text) {
        return v;
    }
    let trimmed = text.trim();
    if trimmed.is_empty() {
        "unknown".into()
    } else {
        truncate(trimmed, 50)
    }
}

fn extract_tlc_version(text: &str) -> Option<String> {
    // Look for "TLC Version X.Y.Z" or "TLC2 Version X.Y.Z" (case-insensitive).
    let lower = text.to_ascii_lowercase();
    for marker in ["tlc2 version", "tlc version"] {
        if let Some(idx) = lower.find(marker) {
            let tail = &text[idx + marker.len()..];
            if let Some(v) = first_dotted_triple(tail) {
                return Some(v);
            }
        }
    }
    first_dotted_triple(text)
}

fn first_dotted_triple(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let mut j = i;
            let mut dots = 0;
            let mut last_was_digit = false;
            while j < bytes.len() {
                let c = bytes[j];
                if c.is_ascii_digit() {
                    last_was_digit = true;
                    j += 1;
                } else if c == b'.' && last_was_digit {
                    dots += 1;
                    last_was_digit = false;
                    j += 1;
                } else {
                    break;
                }
            }
            if dots == 2 && last_was_digit {
                return Some(text[i..j].to_string());
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    None
}

fn java_version() -> String {
    let output = match Command::new("java").arg("-version").output() {
        Ok(o) => o,
        Err(_) => return "unknown".into(),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    if let Some(start) = text.find("version \"") {
        let tail = &text[start + "version \"".len()..];
        if let Some(end) = tail.find('"') {
            return tail[..end].to_string();
        }
    }
    let first_line = text.lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        "unknown".into()
    } else {
        truncate(first_line, 50)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    s.char_indices()
        .take_while(|(i, _)| *i < max)
        .map(|(_, c)| c)
        .collect()
}

// ---------- Baseline schema ----------

fn make_untested_ty_entry() -> Value {
    json!({
        "status": "untested",
        "states": Value::Null,
        "error_type": Value::Null,
        "last_run": Value::Null,
        "git_commit": Value::Null,
    })
}

fn load_existing_output(path: &Path) -> Option<Value> {
    let text = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    if value.get("specs").and_then(Value::as_object).is_some() {
        Some(value)
    } else {
        None
    }
}

fn order_spec_entry(entry: Value) -> Value {
    let mut entry_obj = match entry {
        Value::Object(map) => map,
        _ => return Value::Null,
    };

    // Schema-v2 migration: if "tlc" is missing or not an object, lift the
    // legacy flat keys (`expected_states`, `tlc_runtime_seconds`, etc.)
    // into a nested `tlc` object.
    let needs_migration = entry_obj.get("tlc").map(|v| !v.is_object()).unwrap_or(true);
    if needs_migration {
        let mut tlc = Map::new();
        tlc.insert(
            "status".into(),
            entry_obj
                .get("status")
                .cloned()
                .unwrap_or_else(|| Value::String("unknown".into())),
        );
        tlc.insert(
            "states".into(),
            entry_obj
                .get("expected_states")
                .cloned()
                .unwrap_or(Value::Null),
        );
        tlc.insert(
            "runtime_seconds".into(),
            entry_obj
                .get("tlc_runtime_seconds")
                .cloned()
                .unwrap_or(Value::Null),
        );
        tlc.insert(
            "error_type".into(),
            entry_obj.get("error_type").cloned().unwrap_or(Value::Null),
        );
        tlc.insert(
            "error".into(),
            entry_obj.get("error").cloned().unwrap_or(Value::Null),
        );

        let category = entry_obj
            .get("category")
            .cloned()
            .unwrap_or_else(|| Value::String("unknown".into()));
        let source = entry_obj.get("source").cloned();

        let mut new_entry = Map::new();
        new_entry.insert("tlc".into(), Value::Object(tlc));
        new_entry.insert("ty".into(), make_untested_ty_entry());
        new_entry.insert("verified_match".into(), Value::Bool(false));
        new_entry.insert("category".into(), category);
        if let Some(src) = source {
            new_entry.insert("source".into(), src);
        }

        for (k, v) in entry_obj.into_iter() {
            if new_entry.contains_key(&k) {
                continue;
            }
            if LEGACY_V2_KEYS.iter().any(|legacy| *legacy == k) {
                continue;
            }
            new_entry.insert(k, v);
        }
        entry_obj = new_entry;
    }

    let mut result = Map::new();
    for key in SPEC_ENTRY_KEY_ORDER {
        if let Some(value) = entry_obj.remove(*key) {
            match (*key, &value) {
                ("tlc", Value::Object(_)) => {
                    let Value::Object(inner) = value else {
                        unreachable!()
                    };
                    result.insert(
                        (*key).into(),
                        Value::Object(order_inner(inner, TLC_ENTRY_KEY_ORDER)),
                    );
                }
                ("ty", Value::Object(_)) => {
                    let Value::Object(inner) = value else {
                        unreachable!()
                    };
                    result.insert(
                        (*key).into(),
                        Value::Object(order_inner(inner, TY_ENTRY_KEY_ORDER)),
                    );
                }
                _ => {
                    result.insert((*key).into(), value);
                }
            }
        }
    }
    let mut leftover_keys: Vec<String> = entry_obj.keys().cloned().collect();
    leftover_keys.sort();
    for key in leftover_keys {
        if let Some(value) = entry_obj.remove(&key) {
            result.insert(key, value);
        }
    }
    Value::Object(result)
}

fn order_inner(mut inner: Map<String, Value>, order: &[&str]) -> Map<String, Value> {
    let mut out = Map::new();
    for key in order {
        if let Some(value) = inner.remove(*key) {
            out.insert((*key).into(), value);
        }
    }
    let mut leftover: Vec<String> = inner.keys().cloned().collect();
    leftover.sort();
    for key in leftover {
        if let Some(value) = inner.remove(&key) {
            out.insert(key, value);
        }
    }
    out
}

fn compute_stats(specs: &Map<String, Value>) -> Map<String, Value> {
    let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
    for key in STATS_KEY_ORDER {
        counts.insert(key, 0);
    }
    for data in specs.values() {
        let tlc_status = data
            .get("tlc")
            .and_then(Value::as_object)
            .and_then(|m| m.get("status"))
            .and_then(Value::as_str)
            .or_else(|| data.get("status").and_then(Value::as_str))
            .unwrap_or("unknown");
        match tlc_status {
            "pass" => *counts.entry("tlc_pass").or_default() += 1,
            "timeout" => *counts.entry("tlc_timeout").or_default() += 1,
            _ => *counts.entry("tlc_error").or_default() += 1,
        }

        let Some(ty) = data.get("ty").and_then(Value::as_object) else {
            *counts.entry("ty_untested").or_default() += 1;
            continue;
        };
        let ty_status = ty
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("untested");
        let verified = data
            .get("verified_match")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        match ty_status {
            "pass" if verified => *counts.entry("ty_match").or_default() += 1,
            "mismatch" => *counts.entry("ty_mismatch").or_default() += 1,
            "fail" => *counts.entry("ty_fail").or_default() += 1,
            _ => *counts.entry("ty_untested").or_default() += 1,
        }
    }
    let mut out = Map::new();
    for key in STATS_KEY_ORDER {
        out.insert((*key).into(), json!(*counts.get(key).unwrap_or(&0)));
    }
    out
}

fn compute_categories(specs: &Map<String, Value>) -> Map<String, Value> {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for data in specs.values() {
        let cat = data
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        *counts.entry(cat).or_default() += 1;
    }
    let mut out = Map::new();
    for key in CATEGORIES_KEY_ORDER {
        let value = counts.remove(*key).unwrap_or(0);
        out.insert((*key).into(), json!(value));
    }
    let mut leftover: Vec<String> = counts.keys().cloned().collect();
    leftover.sort();
    for key in leftover {
        let value = counts.remove(&key).unwrap_or(0);
        out.insert(key, json!(value));
    }
    out
}

fn validate_baselines(specs: &Map<String, Value>) -> Vec<String> {
    let mut warnings = Vec::new();
    for (name, data) in specs {
        let Some(tlc) = data.get("tlc").and_then(Value::as_object) else {
            continue;
        };
        let status = tlc
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let states = tlc.get("states");
        if status == "pass" && (states.is_none() || matches!(states, Some(Value::Null))) {
            warnings.push(format!(
                "{name}: TLC status=pass but states=null — state count not parsed"
            ));
        }
        let error_type = tlc.get("error_type").and_then(Value::as_str);
        if status == "pass" && matches!(error_type, Some(et) if et != "unknown") {
            warnings.push(format!(
                "{name}: TLC status=pass but error_type={} — status should likely be 'error'",
                error_type.unwrap()
            ));
        }
    }
    warnings
}

fn build_ordered_specs(
    baselines: BTreeMap<String, Value>,
    catalog: &[SpecInfo],
) -> Map<String, Value> {
    let catalog_names: BTreeSet<&str> = catalog.iter().map(|s| s.name.as_str()).collect();
    let mut work = baselines;
    let mut result = Map::new();
    for spec in catalog {
        if let Some(value) = work.remove(&spec.name) {
            result.insert(spec.name.clone(), order_spec_entry(value));
        }
    }
    let mut extras: Vec<String> = work
        .keys()
        .filter(|k| !catalog_names.contains(k.as_str()))
        .cloned()
        .collect();
    extras.sort();
    for name in extras {
        if let Some(value) = work.remove(&name) {
            result.insert(name, order_spec_entry(value));
        }
    }
    // Anything still left (e.g. a baseline entry that *was* in the catalog
    // but failed the migration) — append sorted.
    let mut remaining: Vec<String> = work.keys().cloned().collect();
    remaining.sort();
    for name in remaining {
        if let Some(value) = work.remove(&name) {
            result.insert(name, order_spec_entry(value));
        }
    }
    result
}

// ---------- JCS digest (matches scripts/json_jcs.py) ----------

fn sha256_jcs(value: &Value) -> Result<String> {
    let mut canonical = String::new();
    write_canonical_json(value, &mut canonical)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_canonical_json(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&canonicalize_number(number)?),
        Value::String(text) => out.push_str(&serde_json::to_string(text)?),
        Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(a, _)| *a);
            out.push('{');
            for (idx, (key, item)) in entries.into_iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key)?);
                out.push(':');
                write_canonical_json(item, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn canonicalize_number(number: &serde_json::Number) -> Result<String> {
    if let Some(value) = number.as_i64() {
        return Ok(value.to_string());
    }
    if let Some(value) = number.as_u64() {
        return Ok(value.to_string());
    }
    if let Some(value) = number.as_f64() {
        return format_float_jcs(value);
    }
    bail!("unsupported JSON number for canonicalization: {number}")
}

fn format_float_jcs(value: f64) -> Result<String> {
    if !value.is_finite() {
        bail!("non-finite float not allowed in canonical JSON: {value:?}");
    }
    if value == 0.0 {
        return Ok("0".to_string());
    }
    let mut formatted = format!("{value:?}");
    if let Some(exp_index) = formatted.find(['e', 'E']) {
        let mantissa = formatted[..exp_index].to_string();
        let exponent = &formatted[exp_index + 1..];
        let (sign, digits) = match exponent.as_bytes().first().copied() {
            Some(b'+') => ("+", &exponent[1..]),
            Some(b'-') => ("-", &exponent[1..]),
            _ => ("", exponent),
        };
        let digits = digits.trim_start_matches('0');
        let digits = if digits.is_empty() { "0" } else { digits };
        return Ok(format!("{mantissa}e{sign}{digits}"));
    }
    if formatted.contains('.') {
        while formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.ends_with('.') {
            formatted.pop();
        }
    }
    Ok(formatted)
}

// ---------- Output writer ----------

fn write_output(
    path: &Path,
    baselines: BTreeMap<String, Value>,
    provenance: &Map<String, Value>,
    catalog: &[SpecInfo],
) -> Result<Map<String, Value>> {
    let ordered_specs = build_ordered_specs(baselines, catalog);
    for warning in validate_baselines(&ordered_specs) {
        eprintln!("WARNING: baseline anomaly: {warning}");
    }
    let stats = compute_stats(&ordered_specs);
    let categories = compute_categories(&ordered_specs);
    let specs_value = Value::Object(ordered_specs.clone());
    let specs_jcs = sha256_jcs(&specs_value)?;

    let mut output = Map::new();
    output.insert(
        "schema_version".into(),
        provenance
            .get("schema_version")
            .cloned()
            .unwrap_or_else(|| json!(SCHEMA_VERSION)),
    );
    output.insert("generated".into(), Value::String(now_iso_local()));
    output.insert(
        "collector".into(),
        provenance.get("collector").cloned().unwrap_or_default(),
    );
    output.insert(
        "tlc".into(),
        provenance.get("tlc").cloned().unwrap_or_default(),
    );
    output.insert(
        "inputs".into(),
        provenance.get("inputs").cloned().unwrap_or_default(),
    );
    output.insert(
        "seed".into(),
        provenance.get("seed").cloned().unwrap_or_default(),
    );
    output.insert(
        "tlc_timeout_seconds".into(),
        provenance
            .get("tlc_timeout_seconds")
            .cloned()
            .unwrap_or_else(|| json!(DEFAULT_TIMEOUT_SECONDS)),
    );
    output.insert("total_specs".into(), json!(catalog.len()));
    output.insert("specs_jcs_sha256".into(), Value::String(specs_jcs));
    output.insert("stats".into(), Value::Object(stats));
    output.insert("categories".into(), Value::Object(categories));
    output.insert("specs".into(), specs_value);

    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let text = serde_json::to_string_pretty(&Value::Object(output.clone()))?;
    fs::write(&tmp_path, text).with_context(|| format!("writing {}", tmp_path.display()))?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("renaming {} -> {}", tmp_path.display(), path.display()))?;
    Ok(output)
}

fn now_iso_local() -> String {
    // Mirrors Python's `datetime.now().isoformat()` shape with UTC time
    // (the local-naive baseline format only used the wall-clock time, so
    // emitting UTC keeps the output reproducible across machines).
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let micros = now.subsec_micros();
    let days = (secs / 86_400) as i64;
    let secs_of_day = (secs % 86_400) as u32;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}.{micros:06}")
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    // Howard Hinnant's days-to-civil algorithm.
    let z = z + 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = (y + i64::from(m <= 2)) as i32;
    (y, m, d)
}

// ---------- Path discovery ----------

struct PathContext {
    project_root: PathBuf,
    examples_dir: PathBuf,
    tlc_jar: PathBuf,
    community_modules: PathBuf,
    tlaplus_dir: PathBuf,
    examples_base_dir: PathBuf,
    output: PathBuf,
    spec_catalog: PathBuf,
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

impl PathContext {
    fn from_cli(cli: &Cli) -> PathContext {
        let project_root = cli
            .project_root
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let tlaplus_dir = cli
            .tlaplus_dir
            .clone()
            .unwrap_or_else(|| home().join("tlaplus"));
        let examples_base_dir = cli
            .examples_base_dir
            .clone()
            .unwrap_or_else(|| home().join("tlaplus-examples"));
        let examples_dir = cli
            .examples_dir
            .clone()
            .unwrap_or_else(|| examples_base_dir.join("specifications"));
        let tlc_jar = cli
            .tlc_jar
            .clone()
            .unwrap_or_else(|| tlaplus_dir.join("tytools.jar"));
        let community_modules = cli
            .community_modules
            .clone()
            .unwrap_or_else(|| tlaplus_dir.join("CommunityModules.jar"));
        let output = cli.output.clone().unwrap_or_else(|| {
            project_root
                .join("tests")
                .join("tlc_comparison")
                .join("spec_baseline.json")
        });
        let spec_catalog = cli.spec_catalog.clone().unwrap_or_else(|| {
            project_root
                .join("tests")
                .join("tlc_comparison")
                .join("spec_catalog.py")
        });
        PathContext {
            project_root,
            examples_dir,
            tlc_jar,
            community_modules,
            tlaplus_dir,
            examples_base_dir,
            output,
            spec_catalog,
        }
    }
}

// ---------- Main ----------

fn run(cli: Cli) -> Result<()> {
    let ctx = PathContext::from_cli(&cli);
    let timeout_seconds = cli.timeout;

    let catalog_text = fs::read_to_string(&ctx.spec_catalog)
        .with_context(|| format!("reading {}", ctx.spec_catalog.display()))?;
    let catalog = parse_spec_catalog(&catalog_text)
        .with_context(|| format!("parsing {}", ctx.spec_catalog.display()))?;

    let provenance = build_provenance(timeout_seconds, &ctx);

    println!("Collecting TLC baselines for {} specs...", catalog.len());
    println!("Output: {}", ctx.output.display());
    println!("Timeout: {timeout_seconds}s per spec");
    let prov_collector_short = provenance
        .get("collector")
        .and_then(|v| v.get("ty_git_commit"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let tlc_version_str = provenance
        .get("tlc")
        .and_then(|v| v.get("tlc_version"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let examples_short = provenance
        .get("inputs")
        .and_then(|v| v.get("examples_git"))
        .and_then(|v| v.get("head_short"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    println!(
        "Provenance: schema_version={SCHEMA_VERSION}, tlc={tlc_version_str}, examples={examples_short}, collector={prov_collector_short}"
    );

    let mut baselines: BTreeMap<String, Value> = if cli.no_resume {
        BTreeMap::new()
    } else if let Some(existing) = load_existing_output(&ctx.output) {
        let specs = existing
            .get("specs")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        specs
            .into_iter()
            .filter(|(_, v)| v.is_object())
            .map(|(k, v)| (k, order_spec_entry(v)))
            .collect()
    } else {
        BTreeMap::new()
    };

    let total = catalog.len();
    for (idx, spec) in catalog.iter().enumerate() {
        let progress_label = if spec.name.len() > 40 {
            spec.name.chars().take(40).collect::<String>()
        } else {
            spec.name.clone()
        };
        eprint!("\r[{}/{}] {progress_label:<40}", idx + 1, total);

        let existing_entry = baselines.get(&spec.name).cloned();
        if let Some(Value::Object(existing_obj)) = &existing_entry {
            if let Some(tlc) = existing_obj.get("tlc").and_then(Value::as_object) {
                let status = tlc.get("status").and_then(Value::as_str);
                let states_present = tlc
                    .get("states")
                    .map(|v| !matches!(v, Value::Null))
                    .unwrap_or(false);
                if status == Some("pass") && states_present {
                    continue;
                }
                if matches!(status, Some("error") | Some("timeout")) {
                    continue;
                }
            }
        }

        let spec_path = ctx.examples_dir.join(&spec.tla_path);
        let cfg_path = ctx.examples_dir.join(&spec.cfg_path);

        if !spec_path.exists() {
            baselines.insert(
                spec.name.clone(),
                missing_entry(
                    &existing_entry,
                    "missing_file",
                    &format!("File not found: {}", spec.tla_path),
                ),
            );
            write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;
            continue;
        }
        if !cfg_path.exists() {
            baselines.insert(
                spec.name.clone(),
                missing_entry(
                    &existing_entry,
                    "missing_config",
                    &format!("Config not found: {}", spec.cfg_path),
                ),
            );
            write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;
            continue;
        }

        let outcome = run_tlc(
            &spec_path,
            &cfg_path,
            timeout_seconds,
            &ctx.tlc_jar,
            &ctx.community_modules,
            &ctx.project_root,
        );
        let category = categorize_runtime(outcome.runtime_seconds);

        let (ty_data, issue) = match existing_entry.as_ref().and_then(Value::as_object) {
            Some(existing_obj) => {
                let ty = existing_obj
                    .get("ty")
                    .filter(|v| v.is_object())
                    .cloned()
                    .unwrap_or_else(make_untested_ty_entry);
                let issue = existing_obj.get("issue").cloned();
                (ty, issue)
            }
            None => (make_untested_ty_entry(), None),
        };

        let tlc_states = outcome.states;
        let ty_states = ty_data.get("states").and_then(|v| v.as_u64());
        let ty_status = ty_data
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("untested");
        let verified_match = ty_status == "pass"
            && ty_states.is_some()
            && tlc_states.is_some()
            && ty_states == tlc_states;

        let mut tlc_entry = Map::new();
        tlc_entry.insert("status".into(), Value::String(outcome.status.clone()));
        tlc_entry.insert(
            "states".into(),
            match outcome.states {
                Some(s) => json!(s),
                None => Value::Null,
            },
        );
        tlc_entry.insert(
            "runtime_seconds".into(),
            match outcome.runtime_seconds {
                Some(rt) => json!(rt),
                None => Value::Null,
            },
        );
        tlc_entry.insert(
            "error_type".into(),
            match outcome.error_type.as_deref() {
                Some(et) => Value::String(et.into()),
                None => Value::Null,
            },
        );
        tlc_entry.insert(
            "error".into(),
            match outcome.error.as_deref() {
                Some(e) => Value::String(e.into()),
                None => Value::Null,
            },
        );

        let mut spec_entry = Map::new();
        spec_entry.insert("tlc".into(), Value::Object(tlc_entry));
        spec_entry.insert("ty".into(), ty_data);
        spec_entry.insert("verified_match".into(), Value::Bool(verified_match));
        spec_entry.insert("category".into(), Value::String(category.into()));
        spec_entry.insert(
            "source".into(),
            json!({
                "tla_path": spec.tla_path,
                "cfg_path": spec.cfg_path,
            }),
        );
        if let Some(issue) = issue {
            spec_entry.insert("issue".into(), issue);
        }

        baselines.insert(
            spec.name.clone(),
            order_spec_entry(Value::Object(spec_entry)),
        );
        write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;
    }

    eprintln!();
    println!();
    println!("{}", "=".repeat(60));
    println!("TLC BASELINE COLLECTION SUMMARY");
    println!("{}", "=".repeat(60));
    let final_output = write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;
    let stats = final_output
        .get("stats")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let categories = final_output
        .get("categories")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let get =
        |m: &Map<String, Value>, k: &str| -> u64 { m.get(k).and_then(Value::as_u64).unwrap_or(0) };
    println!("TLC pass:    {}", get(&stats, "tlc_pass"));
    println!("TLC error:   {}", get(&stats, "tlc_error"));
    println!("TLC timeout: {}", get(&stats, "tlc_timeout"));
    println!();
    println!("Runtime Categories:");
    println!("  Small (<1s):     {}", get(&categories, "small"));
    println!("  Medium (<30s):   {}", get(&categories, "medium"));
    println!("  Large (<300s):   {}", get(&categories, "large"));
    println!("  XLarge (>300s):  {}", get(&categories, "xlarge"));
    println!(
        "  Skip/Unknown:    {}",
        get(&categories, "skip") + get(&categories, "unknown")
    );
    println!();
    println!("Wrote {}", ctx.output.display());

    let errors: Vec<(&String, &Value)> = final_output
        .get("specs")
        .and_then(Value::as_object)
        .map(|specs| {
            specs
                .iter()
                .filter(|(_, v)| {
                    let status = v
                        .get("tlc")
                        .and_then(Value::as_object)
                        .and_then(|m| m.get("status"))
                        .and_then(Value::as_str);
                    matches!(status, Some("error") | Some("timeout"))
                })
                .collect()
        })
        .unwrap_or_default();
    if !errors.is_empty() {
        println!();
        println!("ERRORS/TIMEOUTS:");
        for (name, data) in errors.iter().take(20) {
            let tlc = data.get("tlc").and_then(Value::as_object);
            let status = tlc
                .and_then(|m| m.get("status"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let etype = tlc
                .and_then(|m| m.get("error_type"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            println!("  {name}: {status} - {etype}");
        }
        if errors.len() > 20 {
            println!("  ... and {} more", errors.len() - 20);
        }
    }
    Ok(())
}

fn missing_entry(existing: &Option<Value>, error_type: &str, message: &str) -> Value {
    let mut tlc = Map::new();
    tlc.insert("status".into(), Value::String("error".into()));
    tlc.insert("states".into(), Value::Null);
    tlc.insert("runtime_seconds".into(), Value::Null);
    tlc.insert("error_type".into(), Value::String(error_type.into()));
    tlc.insert("error".into(), Value::String(message.into()));

    let ty = existing
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|m| m.get("ty"))
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(make_untested_ty_entry);

    let mut entry = Map::new();
    entry.insert("tlc".into(), Value::Object(tlc));
    entry.insert("ty".into(), ty);
    entry.insert("verified_match".into(), Value::Bool(false));
    entry.insert("category".into(), Value::String("unknown".into()));
    order_spec_entry(Value::Object(entry))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:?}");
            ExitCode::FAILURE
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_catalog_extracts_three_string_fields() {
        let body = r#"
ALL_SPECS = [
    SpecInfo("MCBakery", "Bakery/MCBakery.tla", "Bakery/MCBakery.cfg", "Bakery"),
    SpecInfo("DieHard", "DieHard/DieHard.tla", "DieHard/DieHard.cfg", "DieHard"),
]

LARGE_SPECS = {}
"#;
        let specs = parse_spec_catalog(body).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "MCBakery");
        assert_eq!(specs[0].tla_path, "Bakery/MCBakery.tla");
        assert_eq!(specs[0].cfg_path, "Bakery/MCBakery.cfg");
        assert_eq!(specs[1].name, "DieHard");
    }

    #[test]
    fn parse_real_spec_catalog_yields_known_specs() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/tlc_comparison/spec_catalog.py");
        if !path.exists() {
            return; // tolerate missing checkout (e.g. published crate)
        }
        let text = fs::read_to_string(&path).unwrap();
        let specs = parse_spec_catalog(&text).unwrap();
        assert!(
            specs.len() >= 100,
            "expected at least 100 specs, got {}",
            specs.len()
        );
        assert!(specs.iter().any(|s| s.name == "MCBakery"));
        assert!(specs.iter().any(|s| s.name == "DieHard"));
    }

    #[test]
    fn parse_tlc_output_states_generated_distinct_left() {
        let text = "Progress(7) at 2024-01-01 12:00:00\n\
                    1,234,567 states generated, 1,234,000 distinct states found, 0 states left on queue.\n";
        let (states, err) = parse_tlc_output(text);
        assert_eq!(states, Some(1_234_000));
        assert!(err.is_none());
    }

    #[test]
    fn parse_tlc_output_distinct_only() {
        let text = "Model checking completed.\n42 distinct states found.\n";
        let (states, _err) = parse_tlc_output(text);
        assert_eq!(states, Some(42));
    }

    #[test]
    fn parse_tlc_output_missing_module() {
        let text = "Cannot find source file for module Naturals imported";
        let (states, err) = parse_tlc_output(text);
        assert_eq!(states, None);
        assert_eq!(err.as_deref(), Some("missing_module:Naturals"));
    }

    #[test]
    fn parse_tlc_output_no_state_count() {
        let (states, err) = parse_tlc_output("nothing useful here\n");
        assert_eq!(states, None);
        assert_eq!(err.as_deref(), Some("no_state_count"));
    }

    #[test]
    fn classify_invariant_error() {
        let text = "Error: Invariant TypeOK is violated.";
        assert_eq!(classify_error(text).as_deref(), Some("invariant"));
    }

    #[test]
    fn classify_deadlock_error() {
        let text = "Error: Deadlock reached.";
        assert_eq!(classify_error(text).as_deref(), Some("deadlock"));
    }

    #[test]
    fn classify_safety_property_error() {
        let text = "Error: Property MySafety is violated.";
        assert_eq!(classify_error(text).as_deref(), Some("safety"));
    }

    #[test]
    fn classify_no_error_returns_none() {
        assert_eq!(classify_error("Model checking completed.").as_deref(), None);
    }

    #[test]
    fn categorize_runtime_buckets() {
        assert_eq!(categorize_runtime(None), "unknown");
        assert_eq!(categorize_runtime(Some(0.5)), "small");
        assert_eq!(categorize_runtime(Some(15.0)), "medium");
        assert_eq!(categorize_runtime(Some(100.0)), "large");
        assert_eq!(categorize_runtime(Some(500.0)), "xlarge");
    }

    #[test]
    fn sha256_jcs_is_order_independent() {
        let left = json!({"b": 1, "a": 2, "c": [3, 4]});
        let right = json!({"a": 2, "c": [3, 4], "b": 1});
        assert_eq!(sha256_jcs(&left).unwrap(), sha256_jcs(&right).unwrap());
    }

    #[test]
    fn sha256_jcs_matches_python_reference_for_specs_shape() {
        // The Python `json_jcs.sha256_jcs` returns the same digest for
        // a small fixture; encode that contract here as a regression.
        let value = json!({
            "Foo": {
                "ty": {"status": "untested"},
                "tlc": {"states": 42, "status": "pass"},
            },
        });
        let digest = sha256_jcs(&value).unwrap();
        // Recomputed via Python: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        // Both impls canonicalize identically for integer/string/null types.
        // Keys are emitted in lexicographic order (RFC 8785 / sort_keys=True):
        // "tlc" < "ty" at the Foo level, "states" < "status" within tlc.
        let expected_canonical =
            "{\"Foo\":{\"tlc\":{\"states\":42,\"status\":\"pass\"},\"ty\":{\"status\":\"untested\"}}}";
        let mut hasher = Sha256::new();
        hasher.update(expected_canonical.as_bytes());
        let expected = format!("{:x}", hasher.finalize());
        assert_eq!(digest, expected);
    }

    #[test]
    fn order_spec_entry_migrates_v2_flat_keys() {
        let legacy = json!({
            "status": "pass",
            "expected_states": 17,
            "tlc_runtime_seconds": 0.42,
            "category": "small",
            "issue": null,
        });
        let ordered = order_spec_entry(legacy);
        let tlc = ordered.get("tlc").unwrap().as_object().unwrap();
        assert_eq!(tlc.get("status").unwrap(), &json!("pass"));
        assert_eq!(tlc.get("states").unwrap(), &json!(17));
        assert_eq!(tlc.get("runtime_seconds").unwrap(), &json!(0.42));
        assert_eq!(ordered.get("category").unwrap(), &json!("small"));
        assert!(ordered.get("ty").unwrap().is_object());
    }

    #[test]
    fn order_spec_entry_preserves_unknown_keys() {
        // serde_json::Map is a BTreeMap (no preserve_order feature), so the
        // serialized form is alphabetical regardless of insertion order.
        // The contract we care about is: (1) all known keys retained,
        // (2) unknown keys retained (no data loss), (3) deterministic order.
        let entry = json!({
            "tlc": {"status": "pass", "states": 1},
            "ty": {"status": "untested"},
            "verified_match": false,
            "category": "small",
            "source": {"tla_path": "x.tla", "cfg_path": "x.cfg"},
            "expected_mismatch": true,
            "issue": "#42",
        });
        let ordered = order_spec_entry(entry).as_object().unwrap().clone();
        for key in [
            "tlc",
            "ty",
            "verified_match",
            "category",
            "source",
            "expected_mismatch",
            "issue",
        ] {
            assert!(ordered.contains_key(key), "missing key {key}");
        }
        // Determinism: keys are sorted (serde_json::Map = BTreeMap).
        let keys: Vec<&String> = ordered.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn compute_stats_counts_tlc_and_ty_buckets() {
        let mut specs: Map<String, Value> = Map::new();
        specs.insert(
            "A".into(),
            json!({
                "tlc": {"status": "pass", "states": 1},
                "ty": {"status": "pass", "states": 1},
                "verified_match": true,
            }),
        );
        specs.insert(
            "B".into(),
            json!({
                "tlc": {"status": "pass", "states": 1},
                "ty": {"status": "mismatch"},
                "verified_match": false,
            }),
        );
        specs.insert(
            "C".into(),
            json!({
                "tlc": {"status": "timeout"},
                "ty": {"status": "untested"},
            }),
        );
        specs.insert(
            "D".into(),
            json!({
                "tlc": {"status": "error"},
                "ty": {"status": "fail"},
            }),
        );
        let stats = compute_stats(&specs);
        assert_eq!(stats.get("tlc_pass").unwrap(), &json!(2));
        assert_eq!(stats.get("tlc_timeout").unwrap(), &json!(1));
        assert_eq!(stats.get("tlc_error").unwrap(), &json!(1));
        assert_eq!(stats.get("ty_match").unwrap(), &json!(1));
        assert_eq!(stats.get("ty_mismatch").unwrap(), &json!(1));
        assert_eq!(stats.get("ty_fail").unwrap(), &json!(1));
        assert_eq!(stats.get("ty_untested").unwrap(), &json!(1));
    }

    #[test]
    fn compute_categories_counts_all_buckets() {
        // serde_json::Map = BTreeMap; key order is alphabetical when
        // serialized. Contract: every observed category is present with
        // the right count, plus the known buckets are always populated
        // even when empty.
        let mut specs: Map<String, Value> = Map::new();
        specs.insert("A".into(), json!({"category": "small"}));
        specs.insert("B".into(), json!({"category": "medium"}));
        specs.insert("C".into(), json!({"category": "apalache"}));
        let cats = compute_categories(&specs);
        assert_eq!(cats.get("small").unwrap(), &json!(1));
        assert_eq!(cats.get("medium").unwrap(), &json!(1));
        assert_eq!(cats.get("apalache").unwrap(), &json!(1));
        for key in CATEGORIES_KEY_ORDER {
            assert!(cats.contains_key(*key), "missing canonical key {key}");
        }
    }

    #[test]
    fn extract_tlc_version_finds_dotted_triple() {
        let text = "TLC Version 2.18.0 of Day Month Year\n";
        assert_eq!(extract_tlc_version(text), Some("2.18.0".into()));
    }

    #[test]
    fn first_dotted_triple_returns_none_for_two_dots_only() {
        assert_eq!(first_dotted_triple("just 1.2"), None);
    }

    #[test]
    fn build_ordered_specs_includes_catalog_and_extras() {
        // serde_json::Map = BTreeMap, so serialized output is always
        // alphabetical regardless of insertion order. Contract: (1)
        // catalog specs and extras are both retained, (2) no data loss.
        let catalog = vec![
            SpecInfo {
                name: "Beta".into(),
                tla_path: "B.tla".into(),
                cfg_path: "B.cfg".into(),
            },
            SpecInfo {
                name: "Alpha".into(),
                tla_path: "A.tla".into(),
                cfg_path: "A.cfg".into(),
            },
        ];
        let mut baselines: BTreeMap<String, Value> = BTreeMap::new();
        baselines.insert(
            "Beta".into(),
            json!({"tlc": {"status": "pass"}, "ty": {"status": "untested"}, "category": "small"}),
        );
        baselines.insert(
            "Alpha".into(),
            json!({"tlc": {"status": "pass"}, "ty": {"status": "untested"}, "category": "small"}),
        );
        baselines.insert(
            "Zeta".into(),
            json!({"tlc": {"status": "pass"}, "ty": {"status": "untested"}, "category": "small"}),
        );
        let ordered = build_ordered_specs(baselines, &catalog);
        for name in ["Alpha", "Beta", "Zeta"] {
            assert!(ordered.contains_key(name), "missing {name}");
        }
        assert_eq!(ordered.len(), 3);
    }

    #[test]
    fn missing_entry_marks_status_error_and_keeps_existing_ty() {
        let prev = json!({"ty": {"status": "pass", "states": 7}});
        let entry = missing_entry(&Some(prev), "missing_file", "File not found: x.tla");
        let obj = entry.as_object().unwrap();
        assert_eq!(
            obj.get("tlc").unwrap().get("error_type").unwrap(),
            &json!("missing_file")
        );
        assert_eq!(obj.get("verified_match").unwrap(), &Value::Bool(false));
        assert_eq!(
            obj.get("ty").unwrap().get("status").unwrap(),
            &json!("pass")
        );
    }

    #[test]
    fn write_output_round_trip_matches_jcs_digest() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("baseline.json");
        let catalog = vec![SpecInfo {
            name: "Alpha".into(),
            tla_path: "A.tla".into(),
            cfg_path: "A.cfg".into(),
        }];
        let mut baselines: BTreeMap<String, Value> = BTreeMap::new();
        baselines.insert(
            "Alpha".into(),
            json!({
                "tlc": {"status": "pass", "states": 5, "runtime_seconds": 0.1},
                "ty": {"status": "untested"},
                "verified_match": false,
                "category": "small",
                "source": {"tla_path": "A.tla", "cfg_path": "A.cfg"},
            }),
        );
        let provenance = {
            let mut p = Map::new();
            p.insert("schema_version".into(), json!(SCHEMA_VERSION));
            p.insert("collector".into(), json!({"ty_git_commit": "deadbeef"}));
            p.insert("tlc".into(), json!({"tlc_version": "X"}));
            p.insert("inputs".into(), json!({}));
            p.insert("seed".into(), json!({}));
            p.insert("tlc_timeout_seconds".into(), json!(60));
            p
        };
        write_output(&path, baselines, &provenance, &catalog).unwrap();
        let body = fs::read_to_string(&path).unwrap();
        let value: Value = serde_json::from_str(&body).unwrap();
        let specs = value.get("specs").cloned().unwrap();
        let expected = sha256_jcs(&specs).unwrap();
        assert_eq!(
            value
                .get("specs_jcs_sha256")
                .and_then(Value::as_str)
                .unwrap(),
            expected
        );
    }
}
