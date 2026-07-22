// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Liveness verdict parity matrix generator.
//!
//! Ported from `scripts/liveness_verdict_matrix.py`. Discovers temporal
//! specs from the TLC baseline (`tests/tlc_comparison/spec_baseline.json`)
//! and from `test_specs/`, optionally runs TLC and TY on each, and
//! emits a JSON + Markdown parity matrix.
//!
//! Verdict classification and trace structure parsing both route
//! through [`tla_petri::liveness_verdict`] so the BenchKit-style
//! parsers stay aligned with the bash harness in
//! `scripts/test_all_liveness.sh`.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};

use tla_petri::liveness_verdict::{
    classify_tlc_status, classify_ty_status, parse_tlc_states, parse_trace_info, parse_ty_states,
    prepend_to_tla_path, temporal_markers, SpecSource, SpecTarget, Tool, TraceInfo, VerdictStatus,
};

const DEFAULT_TIMEOUT_SECONDS: u64 = 120;

#[derive(Parser, Debug)]
#[command(
    name = "ty-liveness-matrix",
    about = "Generate the TLC↔TY liveness verdict parity matrix",
    long_about = "Discovers temporal specs from spec_baseline.json and test_specs/, \
                  optionally runs TLC and TY on each, and writes JSON + Markdown \
                  parity reports. Verdict and trace parsers come from \
                  `tla_petri::liveness_verdict` so the parity matrix and the \
                  shell harness use the same classification."
)]
struct Cli {
    /// Per-tool timeout in seconds.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECONDS)]
    timeout: u64,

    /// Spec source filter.
    #[arg(long, default_value = "all", value_parser = ["all", "baseline", "tests"])]
    source: String,

    /// Substring filter against discovered spec names.
    #[arg(long)]
    name_filter: Option<String>,

    /// Limit number of discovered specs.
    #[arg(long)]
    limit: Option<usize>,

    /// Only print discovered specs (tab-separated).
    #[arg(long, default_value_t = true)]
    dry_run: bool,

    /// Disable dry-run (actually run TLC + TY subprocesses).
    #[arg(long)]
    no_dry_run: bool,

    /// Repository root. Defaults to current working directory.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// JSON output path (only used outside dry-run).
    #[arg(long)]
    json_output: Option<PathBuf>,

    /// Markdown output path (only used outside dry-run).
    #[arg(long)]
    md_output: Option<PathBuf>,

    /// TY binary path. Defaults to `<repo>/target/release/tla`.
    #[arg(long)]
    ty_bin: Option<PathBuf>,

    /// TLC tools jar path. Defaults to `$HOME/tlaplus/tytools.jar`.
    #[arg(long)]
    tlc_jar: Option<PathBuf>,

    /// CommunityModules jar path (optional).
    #[arg(long)]
    community_modules: Option<PathBuf>,

    /// `tlaplus-examples/specifications` directory. Defaults to
    /// `$HOME/tlaplus-examples/specifications`.
    #[arg(long)]
    examples_dir: Option<PathBuf>,
}

fn main() -> ExitCode {
    match Cli::parse().run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(1)
        }
    }
}

impl Cli {
    fn run(&self) -> Result<()> {
        let repo_root = self
            .repo_root
            .canonicalize()
            .with_context(|| format!("canonicalizing --repo-root {}", self.repo_root.display()))?;
        let examples_dir = self
            .examples_dir
            .clone()
            .unwrap_or_else(default_examples_dir);
        let test_specs_dir = repo_root.join("test_specs");

        let mut discovered: Vec<SpecTarget> = Vec::new();
        discovered.extend(discover_baseline(&repo_root, &examples_dir)?);
        discovered.extend(discover_test_specs(&repo_root, &test_specs_dir)?);
        discovered = dedupe(discovered);
        discovered = filter(discovered, self);

        let dry_run = self.dry_run && !self.no_dry_run;

        if dry_run {
            let mut stdout = std::io::stdout().lock();
            for target in &discovered {
                let markers = target.temporal_markers.join(",");
                writeln!(
                    stdout,
                    "{}\t{}\t{}\t{}\t{}",
                    target.source.as_str(),
                    target.name,
                    markers,
                    target.spec_path.display(),
                    target.cfg_path.display()
                )?;
            }
            return Ok(());
        }

        // Live mode: run both tools per spec. Mirrors the Python script.
        let ty_bin = self
            .ty_bin
            .clone()
            .unwrap_or_else(|| repo_root.join("target/release/tla"));
        let tlc_jar = self.tlc_jar.clone().unwrap_or_else(default_tlc_jar);
        let community_modules = self
            .community_modules
            .clone()
            .unwrap_or_else(default_community_modules);

        if !ty_bin.exists() {
            bail!("missing TY binary: {}", ty_bin.display());
        }
        if !tlc_jar.exists() {
            bail!("missing TLC jar: {}", tlc_jar.display());
        }
        let tla_library = repo_root.join("test_specs/tla_library");

        let mut records: Vec<Value> = Vec::new();
        for (idx, target) in discovered.iter().enumerate() {
            eprintln!("[{}/{}] {}", idx + 1, discovered.len(), target.name);
            let tlc = run_tlc(
                target,
                self.timeout,
                &tlc_jar,
                &community_modules,
                &tla_library,
            )?;
            let ty = run_ty(target, self.timeout, &ty_bin, &tla_library)?;
            records.push(build_record(target, &tlc, &ty));
        }

        let summary = summarize(&records, &discovered);
        let generated_at = utc_now_iso();
        let payload = json!({
            "generated_at": generated_at,
            "criteria": {
                "temporal_spec": "cfg has PROPERTY or spec contains WF_/SF_",
                "verdict_match": "tlc_status == ty_status",
                "trace_structure_match": "trace state-count parity + stuttering parity when both runs are errors",
            },
            "summary": summary,
            "records": records,
        });
        let json_text = serde_json::to_string_pretty(&payload).context("encoding JSON output")?;
        let md_text = render_markdown(&records, &generated_at, &summary);

        let json_path = self.json_output.clone().unwrap_or_else(|| {
            repo_root.join("reports/research/issue-1518-liveness-verdict-matrix-current.json")
        });
        let md_path = self.md_output.clone().unwrap_or_else(|| {
            repo_root.join("reports/research/issue-1518-liveness-verdict-matrix-current.md")
        });
        write_with_parents(&json_path, format!("{json_text}\n").as_bytes())?;
        write_with_parents(&md_path, format!("{md_text}\n").as_bytes())?;
        eprintln!("Wrote {}", json_path.display());
        eprintln!("Wrote {}", md_path.display());
        Ok(())
    }
}

#[derive(Debug)]
struct ToolRunOutcome {
    cmd: String,
    rc: i32,
    elapsed_seconds: f64,
    status: VerdictStatus,
    states: Option<u64>,
    trace: TraceInfo,
    /// Captured stdout+stderr. Kept for diagnostics but the JSON
    /// matrix emits only the parsed verdict/state-count/trace fields.
    #[allow(dead_code)]
    output: String,
}

fn discover_baseline(repo_root: &Path, examples_dir: &Path) -> Result<Vec<SpecTarget>> {
    let baseline_path = repo_root.join("tests/tlc_comparison/spec_baseline.json");
    if !baseline_path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&baseline_path)
        .with_context(|| format!("reading {}", baseline_path.display()))?;
    let baseline: Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", baseline_path.display()))?;
    let Some(specs) = baseline.get("specs").and_then(|v| v.as_object()) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (spec_name, payload) in specs {
        let Some(source) = payload.get("source").and_then(|v| v.as_object()) else {
            continue;
        };
        let tla_rel = source.get("tla_path").and_then(|v| v.as_str());
        let cfg_rel = source.get("cfg_path").and_then(|v| v.as_str());
        let (Some(tla_rel), Some(cfg_rel)) = (tla_rel, cfg_rel) else {
            continue;
        };
        let spec_path = examples_dir.join(tla_rel);
        let cfg_path = examples_dir.join(cfg_rel);
        if !spec_path.exists() || !cfg_path.exists() {
            continue;
        }
        let spec_text = fs::read_to_string(&spec_path).unwrap_or_default();
        let cfg_text = fs::read_to_string(&cfg_path).unwrap_or_default();
        let markers = temporal_markers(&spec_text, &cfg_text);
        if markers.is_empty() {
            continue;
        }
        out.push(SpecTarget {
            name: spec_name.clone(),
            source: SpecSource::Baseline,
            spec_path: spec_path.canonicalize().unwrap_or(spec_path),
            cfg_path: cfg_path.canonicalize().unwrap_or(cfg_path),
            temporal_markers: markers,
        });
    }
    Ok(out)
}

fn discover_test_specs(repo_root: &Path, dir: &Path) -> Result<Vec<SpecTarget>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut cfg_paths = Vec::new();
    collect_cfg_files(dir, &mut cfg_paths)?;
    cfg_paths.sort();
    for cfg_path in cfg_paths {
        let Some(spec_path) = resolve_test_spec_path(&cfg_path) else {
            continue;
        };
        let spec_text = fs::read_to_string(&spec_path).unwrap_or_default();
        let cfg_text = fs::read_to_string(&cfg_path).unwrap_or_default();
        let markers = temporal_markers(&spec_text, &cfg_text);
        if markers.is_empty() {
            continue;
        }
        let rel = cfg_path.strip_prefix(repo_root).unwrap_or(&cfg_path);
        let name = rel.with_extension("").to_string_lossy().to_string();
        out.push(SpecTarget {
            name,
            source: SpecSource::Tests,
            spec_path: spec_path.canonicalize().unwrap_or(spec_path),
            cfg_path: cfg_path.canonicalize().unwrap_or(cfg_path),
            temporal_markers: markers,
        });
    }
    Ok(out)
}

fn collect_cfg_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_cfg_files(&path, out)?;
        } else if path.extension().map(|e| e == "cfg").unwrap_or(false) {
            out.push(path);
        }
    }
    Ok(())
}

fn resolve_test_spec_path(cfg_path: &Path) -> Option<PathBuf> {
    let direct = cfg_path.with_extension("tla");
    if direct.exists() {
        return Some(direct);
    }
    let stem = cfg_path.file_stem()?.to_string_lossy().to_string();
    let mut prefixes: Vec<String> = Vec::new();
    if stem.contains('_') {
        let parts: Vec<&str> = stem.split('_').collect();
        for idx in (1..parts.len()).rev() {
            prefixes.push(parts[..idx].join("_"));
        }
    }
    if stem.contains('-') {
        let parts: Vec<&str> = stem.split('-').collect();
        for idx in (1..parts.len()).rev() {
            prefixes.push(parts[..idx].join("-"));
        }
    }
    prefixes.push(stem.clone());

    let mut seen: BTreeSet<String> = BTreeSet::new();
    let parent = cfg_path.parent()?;
    for prefix in prefixes {
        if !seen.insert(prefix.clone()) {
            continue;
        }
        let candidate = parent.join(format!("{prefix}.tla"));
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let mut tla_files: Vec<PathBuf> = Vec::new();
    if let Ok(read_dir) = fs::read_dir(parent) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "tla").unwrap_or(false) {
                tla_files.push(path);
            }
        }
    }
    if tla_files.len() == 1 {
        return Some(tla_files.remove(0));
    }
    None
}

fn dedupe(targets: Vec<SpecTarget>) -> Vec<SpecTarget> {
    let mut seen: BTreeSet<(PathBuf, PathBuf)> = BTreeSet::new();
    let mut out: Vec<SpecTarget> = Vec::new();
    for target in targets {
        let key = (target.spec_path.clone(), target.cfg_path.clone());
        if !seen.insert(key) {
            continue;
        }
        out.push(target);
    }
    out.sort_by_key(|a| (a.source, a.name.clone()));
    out
}

fn filter(targets: Vec<SpecTarget>, cli: &Cli) -> Vec<SpecTarget> {
    let mut out: Vec<SpecTarget> = targets
        .into_iter()
        .filter(|t| match cli.source.as_str() {
            "baseline" => matches!(t.source, SpecSource::Baseline),
            "tests" => matches!(t.source, SpecSource::Tests),
            _ => true,
        })
        .filter(|t| match cli.name_filter.as_deref() {
            Some(needle) => t.name.contains(needle),
            None => true,
        })
        .collect();
    if let Some(limit) = cli.limit {
        out.truncate(limit);
    }
    out
}

fn default_examples_dir() -> PathBuf {
    home()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tlaplus-examples/specifications")
}

fn default_tlc_jar() -> PathBuf {
    home()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tlaplus/tytools.jar")
}

fn default_community_modules() -> PathBuf {
    home()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tlaplus/CommunityModules.jar")
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn write_with_parents(path: &Path, body: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    fs::write(path, body).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn run_tlc(
    target: &SpecTarget,
    timeout_seconds: u64,
    tlc_jar: &Path,
    community_modules: &Path,
    tla_library: &Path,
) -> Result<ToolRunOutcome> {
    let mut classpath = OsString::from(tlc_jar.as_os_str());
    if community_modules.exists() {
        classpath.push(separator_os());
        classpath.push(community_modules.as_os_str());
    }
    let mut args: Vec<OsString> = vec![
        OsString::from("java"),
        OsString::from(format!("-DTLA-Library={}", tla_library.display())),
        OsString::from("-cp"),
        classpath,
        OsString::from("tlc2.TLC"),
        OsString::from("-workers"),
        OsString::from("1"),
        OsString::from("-config"),
        target.cfg_path.clone().into_os_string(),
        target.spec_path.clone().into_os_string(),
    ];

    let env = build_env_for_spec(&target.spec_path, tla_library);
    let started = Instant::now();
    let (rc, output) =
        run_with_timeout(&mut args, &env, target.spec_path.parent(), timeout_seconds)?;
    let elapsed = started.elapsed().as_secs_f64();
    let elapsed = (elapsed * 1000.0).round() / 1000.0;
    Ok(ToolRunOutcome {
        cmd: args
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(" "),
        rc,
        elapsed_seconds: elapsed,
        status: classify_tlc_status(&output, rc),
        states: parse_tlc_states(&output),
        trace: parse_trace_info(&output, Tool::Tlc),
        output,
    })
}

fn run_ty(
    target: &SpecTarget,
    timeout_seconds: u64,
    ty_bin: &Path,
    tla_library: &Path,
) -> Result<ToolRunOutcome> {
    let mut args: Vec<OsString> = vec![
        ty_bin.into(),
        OsString::from("check"),
        target.spec_path.clone().into_os_string(),
        OsString::from("--config"),
        target.cfg_path.clone().into_os_string(),
        OsString::from("--workers"),
        OsString::from("1"),
    ];
    let env = build_env_for_spec(&target.spec_path, tla_library);
    let started = Instant::now();
    let (rc, output) = run_with_timeout(&mut args, &env, None, timeout_seconds)?;
    let elapsed = started.elapsed().as_secs_f64();
    let elapsed = (elapsed * 1000.0).round() / 1000.0;
    Ok(ToolRunOutcome {
        cmd: args
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(" "),
        rc,
        elapsed_seconds: elapsed,
        status: classify_ty_status(&output, rc),
        states: parse_ty_states(&output),
        trace: parse_trace_info(&output, Tool::Ty),
        output,
    })
}

fn build_env_for_spec(spec_path: &Path, tla_library: &Path) -> Vec<(OsString, OsString)> {
    let existing = std::env::var("TLA_PATH").ok();
    let with_library = prepend_to_tla_path(existing.as_deref(), tla_library);
    let parent = spec_path.parent().unwrap_or(spec_path);
    let final_value = prepend_to_tla_path(Some(&with_library), parent);
    vec![(OsString::from("TLA_PATH"), OsString::from(final_value))]
}

fn run_with_timeout(
    args: &mut Vec<OsString>,
    env: &[(OsString, OsString)],
    cwd: Option<&Path>,
    timeout_seconds: u64,
) -> Result<(i32, String)> {
    if args.is_empty() {
        bail!("empty command");
    }
    let program = args.remove(0);
    let mut command = Command::new(&program);
    command.args(args.iter());
    for (k, v) in env {
        command.env(k, v);
    }
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = command.spawn()?;
    let start = Instant::now();

    // Std `Child::wait_with_output` has no timeout; emulate by polling.
    loop {
        if let Some(status) = child.try_wait()? {
            let output = child.wait_with_output()?;
            let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
            combined.push_str(&String::from_utf8_lossy(&output.stderr));
            let rc = status.code().unwrap_or(-1);
            // Rebuild args for caller diagnostics.
            args.insert(0, program);
            return Ok((rc, combined));
        }
        if start.elapsed().as_secs() >= timeout_seconds {
            let _ = child.kill();
            let _ = child.wait();
            args.insert(0, program);
            let note = format!(
                "\n[TIMEOUT] after {timeout_seconds}s: {}\n",
                args.iter()
                    .map(|s| s.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            return Ok((124, note));
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn separator_os() -> OsString {
    if cfg!(windows) {
        OsString::from(";")
    } else {
        OsString::from(":")
    }
}

fn build_record(target: &SpecTarget, tlc: &ToolRunOutcome, ty: &ToolRunOutcome) -> Value {
    let verdict_match = tlc.status == ty.status;
    let state_match = if tlc.status == VerdictStatus::Success && ty.status == VerdictStatus::Success
    {
        match (tlc.states, ty.states) {
            (Some(a), Some(b)) => Some(a == b),
            _ => Some(false),
        }
    } else {
        None
    };
    let trace_structure_match = if tlc.status != VerdictStatus::Success
        && ty.status != VerdictStatus::Success
        && (tlc.trace.state_count > 0 || ty.trace.state_count > 0)
    {
        Some(
            tlc.trace.state_count == ty.trace.state_count
                && tlc.trace.has_stuttering == ty.trace.has_stuttering,
        )
    } else {
        None
    };
    let overall_match =
        if tlc.status == VerdictStatus::Success && ty.status == VerdictStatus::Success {
            matches!(state_match, Some(true))
        } else if tlc.status == VerdictStatus::Liveness && ty.status == VerdictStatus::Liveness {
            !matches!(trace_structure_match, Some(false))
        } else {
            false
        };

    json!({
        "name": target.name,
        "source": target.source.as_str(),
        "spec": target.spec_path,
        "config": target.cfg_path,
        "temporal_markers": target.temporal_markers,
        "tlc_cmd": tlc.cmd,
        "ty_cmd": ty.cmd,
        "tlc_rc": tlc.rc,
        "ty_rc": ty.rc,
        "tlc_status": tlc.status.as_str(),
        "ty_status": ty.status.as_str(),
        "tlc_states": tlc.states,
        "ty_states": ty.states,
        "verdict_match": verdict_match,
        "state_match": state_match,
        "trace_structure_match": trace_structure_match,
        "tlc_trace_states": tlc.trace.state_count,
        "ty_trace_states": ty.trace.state_count,
        "tlc_trace_stuttering": tlc.trace.has_stuttering,
        "ty_trace_stuttering": ty.trace.has_stuttering,
        "tlc_trace_signature": tlc.trace.signature,
        "ty_trace_signature": ty.trace.signature,
        "overall_match": overall_match,
        "tlc_elapsed_seconds": tlc.elapsed_seconds,
        "ty_elapsed_seconds": ty.elapsed_seconds,
    })
}

fn summarize(records: &[Value], targets: &[SpecTarget]) -> Value {
    let baseline_specs = targets
        .iter()
        .filter(|t| matches!(t.source, SpecSource::Baseline))
        .count();
    let test_specs = targets
        .iter()
        .filter(|t| matches!(t.source, SpecSource::Tests))
        .count();
    let overall_match_count = records
        .iter()
        .filter(|r| {
            r.get("overall_match")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .count();
    let verdict_mismatch_count = records
        .iter()
        .filter(|r| {
            !r.get("verdict_match")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .count();
    json!({
        "total_specs": targets.len(),
        "baseline_specs": baseline_specs,
        "test_specs": test_specs,
        "overall_match_count": overall_match_count,
        "verdict_mismatch_count": verdict_mismatch_count,
    })
}

fn render_markdown(records: &[Value], generated_at: &str, summary: &Value) -> String {
    let mut lines: Vec<String> = vec![
        "# Issue #1518 Liveness Verdict Matrix".to_string(),
        String::new(),
        format!("- Generated at: `{generated_at}`"),
        format!(
            "- Temporal specs discovered: `{}`",
            summary["total_specs"]
        ),
        format!("- Baseline temporal specs: `{}`", summary["baseline_specs"]),
        format!("- Test-spec temporal specs: `{}`", summary["test_specs"]),
        format!(
            "- Overall parity matches: `{}/{}`",
            summary["overall_match_count"], summary["total_specs"]
        ),
        format!(
            "- Verdict mismatches: `{}`",
            summary["verdict_mismatch_count"]
        ),
        String::new(),
        "## Matrix".to_string(),
        String::new(),
        "| Name | Source | Markers | TLC | TY | States (TLC/TY) | Trace states (TLC/TY) | Overall |".to_string(),
        "|---|---|---|---:|---:|---:|---:|---:|".to_string(),
    ];
    for record in records {
        let markers = record
            .get("temporal_markers")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        let states = format!(
            "{}/{}",
            render_optional_u64(&record["tlc_states"]),
            render_optional_u64(&record["ty_states"])
        );
        let trace_states = format!(
            "{}/{}",
            record["tlc_trace_states"], record["ty_trace_states"]
        );
        let overall = if record["overall_match"].as_bool().unwrap_or(false) {
            "yes"
        } else {
            "NO"
        };
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            string_field(record, "name"),
            string_field(record, "source"),
            markers,
            string_field(record, "tlc_status"),
            string_field(record, "ty_status"),
            states,
            trace_states,
            overall
        ));
    }
    lines.push(String::new());
    lines.push("## Notes".to_string());
    lines.push(String::new());
    lines.push("- `overall=yes` means: success runs matched state counts, or liveness violations matched trace structure.".to_string());
    lines.push("- Trace structure comparison uses state-count parity and stuttering-marker parity when both tools report errors.".to_string());
    lines.push(String::new());
    lines.join("\n")
}

fn render_optional_u64(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn string_field(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

fn utc_now_iso() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_utc(now)
}

/// Best-effort UTC timestamp formatter that doesn't pull `chrono`.
/// Produces `YYYY-MM-DDTHH:MM:SSZ`.
fn format_utc(secs: u64) -> String {
    // Days since 1970-01-01.
    let days = (secs / 86_400) as i64;
    let secs_of_day = (secs % 86_400) as u32;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}-{second:02}Z")
        .replacen('-', "-", 1)
        .replace("--", "-")
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    // Howard Hinnant's algorithm.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupe_drops_duplicate_paths() {
        let a = SpecTarget {
            name: "A".into(),
            source: SpecSource::Baseline,
            spec_path: PathBuf::from("/x.tla"),
            cfg_path: PathBuf::from("/x.cfg"),
            temporal_markers: vec!["PROPERTY".into()],
        };
        let b = SpecTarget {
            name: "B".into(),
            source: SpecSource::Tests,
            spec_path: PathBuf::from("/x.tla"),
            cfg_path: PathBuf::from("/x.cfg"),
            temporal_markers: vec!["PROPERTY".into()],
        };
        let out = dedupe(vec![a.clone(), b]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "A");
    }

    #[test]
    fn temporal_markers_helpers_pass_through() {
        // Smoke that the re-exported library functions stay connected.
        let markers = temporal_markers(
            "Spec == Init /\\ [][Next]_v /\\ WF_v(Next)",
            "PROPERTY Live\n",
        );
        assert_eq!(markers, vec!["PROPERTY".to_string(), "WF_/SF_".to_string()]);
    }

    #[test]
    fn build_record_marks_state_match() {
        let target = SpecTarget {
            name: "ok".into(),
            source: SpecSource::Tests,
            spec_path: PathBuf::from("/a.tla"),
            cfg_path: PathBuf::from("/a.cfg"),
            temporal_markers: vec!["PROPERTY".into()],
        };
        let tlc = ToolRunOutcome {
            cmd: "tlc …".into(),
            rc: 0,
            elapsed_seconds: 0.1,
            status: VerdictStatus::Success,
            states: Some(7),
            trace: TraceInfo::default(),
            output: String::new(),
        };
        let ty = ToolRunOutcome {
            cmd: "tla …".into(),
            rc: 0,
            elapsed_seconds: 0.1,
            status: VerdictStatus::Success,
            states: Some(7),
            trace: TraceInfo::default(),
            output: String::new(),
        };
        let record = build_record(&target, &tlc, &ty);
        assert_eq!(record["state_match"], Value::Bool(true));
        assert_eq!(record["overall_match"], Value::Bool(true));
        assert_eq!(record["tlc_states"], Value::from(7u64));
    }

    #[test]
    fn render_markdown_includes_summary_block() {
        let target = SpecTarget {
            name: "ok".into(),
            source: SpecSource::Tests,
            spec_path: PathBuf::from("/a.tla"),
            cfg_path: PathBuf::from("/a.cfg"),
            temporal_markers: vec!["PROPERTY".into()],
        };
        let tlc = ToolRunOutcome {
            cmd: "tlc".into(),
            rc: 0,
            elapsed_seconds: 0.1,
            status: VerdictStatus::Success,
            states: Some(1),
            trace: TraceInfo::default(),
            output: String::new(),
        };
        let ty = tlc.clone();
        let records = vec![build_record(&target, &tlc, &ty)];
        let summary = summarize(&records, std::slice::from_ref(&target));
        let md = render_markdown(&records, "2026-01-01T00:00:00Z", &summary);
        assert!(md.contains("# Issue #1518 Liveness Verdict Matrix"));
        assert!(md.contains("| ok |"));
        assert!(md.contains("Overall parity matches"));
    }

    impl Clone for ToolRunOutcome {
        fn clone(&self) -> Self {
            ToolRunOutcome {
                cmd: self.cmd.clone(),
                rc: self.rc,
                elapsed_seconds: self.elapsed_seconds,
                status: self.status,
                states: self.states,
                trace: self.trace.clone(),
                output: self.output.clone(),
            }
        }
    }
}
