// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty parity` — self-service cross-mode verdict-parity check.
//!
//! Runs one spec under several independent engines — the tree-walking interpreter
//! BFS (the trusted ORACLE), the default cooperative fused engine, and the native
//! trust-cg compiled backend — and asserts they reach the SAME verdict. This is the
//! differential soundness check that catches a single mode silently diverging: e.g.
//! a symbolic-safe lane masking a reachable deadlock, or native codegen miscompiling
//! a successor relation. No CI required — a user runs `ty parity MySpec.tla` on demand
//! and gets a one-shot, self-contained assurance that every engine agrees, or a precise
//! disagreement report naming the divergent engine.
//!
//! Exit 0 on agreement (or all-inconclusive), 1 on a disagreement between conclusive
//! engines (a soundness alert).

use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::cli_schema::AuditOutputFormat;

/// The engines compared, in report order. The first is the trusted oracle.
const MODES: &[(&str, &[&str])] = &[
    (
        "interpreter-oracle",
        &["--bfs-only", "--backend", "interpreter"],
    ),
    ("fused-default", &[]),
    ("native-trust-cg", &["--bfs-only", "--backend", "trust-cg"]),
];

/// Canonical verdict for cross-engine comparison. Conclusive variants must agree;
/// `Inconclusive` / `Failed` engines are reported but excluded from the parity check.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Verdict {
    Safe,
    InvariantViolation,
    PropertyViolation,
    LivenessViolation,
    AssumeViolation,
    Deadlock,
    Vacuous,
    /// Bounded / timed out / interrupted — no verdict to compare.
    Inconclusive(String),
    /// The engine itself failed to run (runtime error, backend unavailable, parse).
    Failed(String),
}

impl Verdict {
    fn is_conclusive(&self) -> bool {
        !matches!(self, Verdict::Inconclusive(_) | Verdict::Failed(_))
    }

    /// The comparison key: two conclusive verdicts agree iff their keys match.
    fn key(&self) -> Option<&'static str> {
        match self {
            Verdict::Safe => Some("safe"),
            Verdict::InvariantViolation => Some("invariant-violation"),
            Verdict::PropertyViolation => Some("property-violation"),
            Verdict::LivenessViolation => Some("liveness-violation"),
            Verdict::AssumeViolation => Some("assume-violation"),
            Verdict::Deadlock => Some("deadlock"),
            Verdict::Vacuous => Some("vacuous"),
            Verdict::Inconclusive(_) | Verdict::Failed(_) => None,
        }
    }

    fn label(&self) -> String {
        match self {
            Verdict::Safe => "SAFE (no error found)".to_string(),
            Verdict::InvariantViolation => "INVARIANT VIOLATED".to_string(),
            Verdict::PropertyViolation => "PROPERTY VIOLATED".to_string(),
            Verdict::LivenessViolation => "LIVENESS VIOLATED".to_string(),
            Verdict::AssumeViolation => "ASSUME VIOLATED".to_string(),
            Verdict::Deadlock => "DEADLOCK".to_string(),
            Verdict::Vacuous => "VACUOUS".to_string(),
            Verdict::Inconclusive(r) => format!("inconclusive ({r})"),
            Verdict::Failed(r) => format!("not run ({r})"),
        }
    }
}

#[derive(Deserialize)]
struct CheckJson {
    result: ResultJson,
    #[serde(default)]
    statistics: Option<StatsJson>,
}

#[derive(Deserialize)]
struct ResultJson {
    status: String,
    #[serde(default)]
    error_type: Option<String>,
    #[serde(default)]
    error_code: Option<String>,
}

#[derive(Deserialize)]
struct StatsJson {
    #[serde(default)]
    states_distinct: Option<u64>,
    #[serde(default)]
    states_found: Option<u64>,
}

struct ModeResult {
    name: &'static str,
    verdict: Verdict,
    states: Option<u64>,
    elapsed: Duration,
}

/// Run a spec under every engine and report whether their verdicts agree.
pub(crate) fn cmd_parity(
    file: Option<&Path>,
    config: Option<&Path>,
    timeout_secs: u64,
    max_states: usize,
    corpus: Option<&Path>,
    format: AuditOutputFormat,
) -> Result<()> {
    let exe = std::env::current_exe().context("could not resolve the `ty` executable path")?;
    let timeout = Duration::from_secs(timeout_secs.max(1));

    if let Some(dir) = corpus {
        return run_corpus(&exe, dir, timeout, max_states, format);
    }

    let file = file.context("provide a spec file, or use --corpus <dir> to sweep a directory")?;
    if !file.is_file() {
        bail!("spec file not found: {}", file.display());
    }
    let cfg = resolve_config(file, config);
    let (results, disagreement, conclusive) = parity_one(&exe, file, &cfg, timeout, max_states);

    match format {
        AuditOutputFormat::Json => print_json(file, &results, disagreement, conclusive),
        AuditOutputFormat::Human => print_human(file, &results, disagreement, conclusive),
    }

    if disagreement {
        // SOUNDNESS ALERT: engines diverged on a definitive verdict.
        std::process::exit(1);
    }
    Ok(())
}

/// Run one spec under every engine. Returns the per-engine results, whether the
/// conclusive engines disagreed, and how many were conclusive.
fn parity_one(
    exe: &Path,
    file: &Path,
    cfg: &Option<PathBuf>,
    timeout: Duration,
    max_states: usize,
) -> (Vec<ModeResult>, bool, usize) {
    let mut results = Vec::with_capacity(MODES.len());
    for (name, flags) in MODES {
        let mut args: Vec<String> = vec!["check".to_string(), file.display().to_string()];
        if let Some(c) = cfg {
            args.push("--config".to_string());
            args.push(c.display().to_string());
        }
        args.extend(flags.iter().map(|f| (*f).to_string()));
        if max_states > 0 {
            args.push("--max-states".to_string());
            args.push(max_states.to_string());
        }
        args.push("--output".to_string());
        args.push("json".to_string());

        let started = Instant::now();
        let (verdict, states) = run_mode(exe, &args, timeout);
        results.push(ModeResult {
            name,
            verdict,
            states,
            elapsed: started.elapsed(),
        });
    }
    let keys: BTreeSet<&str> = results.iter().filter_map(|r| r.verdict.key()).collect();
    let conclusive = results.iter().filter(|r| r.verdict.is_conclusive()).count();
    (results, keys.len() >= 2, conclusive)
}

/// Sweep every `<name>.tla` (with a sibling `<name>.cfg`) in `dir` and report a
/// cross-mode parity scorecard. Exits 1 if ANY spec shows a disagreement.
fn run_corpus(
    exe: &Path,
    dir: &Path,
    timeout: Duration,
    max_states: usize,
    format: AuditOutputFormat,
) -> Result<()> {
    if !dir.is_dir() {
        bail!("corpus path is not a directory: {}", dir.display());
    }
    let mut specs: Vec<(PathBuf, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("tla") {
            let cfg = path.with_extension("cfg");
            if cfg.is_file() {
                specs.push((path, cfg));
            }
        }
    }
    specs.sort();
    if specs.is_empty() {
        bail!(
            "no `<name>.tla` + `<name>.cfg` pairs found in {}",
            dir.display()
        );
    }

    let mut rows = Vec::new();
    let mut disagreements = 0usize;
    for (tla, cfg) in &specs {
        let (results, disagreement, conclusive) =
            parity_one(exe, tla, &Some(cfg.clone()), timeout, max_states);
        if disagreement {
            disagreements += 1;
        }
        let status = if disagreement {
            "DISAGREEMENT"
        } else if conclusive == 0 {
            "inconclusive"
        } else {
            "parity"
        };
        let verdict = results
            .first()
            .map(|r| r.verdict.label())
            .unwrap_or_default();
        rows.push((
            tla.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string(),
            status,
            verdict,
        ));
    }

    match format {
        AuditOutputFormat::Json => {
            let items: Vec<serde_json::Value> = rows
                .iter()
                .map(|(name, status, verdict)| {
                    serde_json::json!({ "spec": name, "status": status, "oracle_verdict": verdict })
                })
                .collect();
            let doc = serde_json::json!({
                "tool": "ty parity --corpus",
                "schema": "ty.parity-corpus/v1",
                "dir": dir.display().to_string(),
                "specs": specs.len(),
                "disagreements": disagreements,
                "results": items,
            });
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        }
        AuditOutputFormat::Human => {
            println!(
                "Cross-mode parity sweep: {} ({} specs)\n",
                dir.display(),
                specs.len()
            );
            let w = rows.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
            for (name, status, verdict) in &rows {
                println!("  {:<w$}  {:<13} {}", name, status, verdict, w = w);
            }
            println!();
            if disagreements == 0 {
                println!(
                    "SWEEP CLEAN — all {} specs show cross-mode parity.",
                    specs.len()
                );
            } else {
                println!(
                    "SWEEP FOUND {disagreements} DISAGREEMENT(S) of {} specs — soundness alert.",
                    specs.len()
                );
            }
        }
    }

    if disagreements > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn resolve_config(file: &Path, config: Option<&Path>) -> Option<PathBuf> {
    if let Some(c) = config {
        return Some(c.to_path_buf());
    }
    let derived = file.with_extension("cfg");
    derived.is_file().then_some(derived)
}

/// Spawn `ty check ...` and classify its verdict, killing it after `timeout`.
/// stderr (telemetry) is discarded; only the JSON on stdout is parsed.
fn run_mode(exe: &Path, args: &[String], timeout: Duration) -> (Verdict, Option<u64>) {
    let mut child = match Command::new(exe)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return (Verdict::Failed(format!("spawn failed: {e}")), None),
    };
    let Some(mut stdout) = child.stdout.take() else {
        let _ = child.kill();
        return (Verdict::Failed("no stdout pipe".to_string()), None);
    };
    // Drain stdout on a background thread so the pipe can never fill and wedge the
    // child (which would defeat the try_wait timeout below).
    let (tx, rx) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut buf = String::new();
        let _ = stdout.read_to_string(&mut buf);
        let _ = tx.send(buf);
    });

    let start = Instant::now();
    let timed_out = loop {
        match child.try_wait() {
            Ok(Some(_status)) => break false,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break true;
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                let _ = child.kill();
                return (Verdict::Failed(format!("wait failed: {e}")), None);
            }
        }
    };
    let stdout_str = rx.recv_timeout(Duration::from_secs(3)).unwrap_or_default();
    let _ = reader.join();

    if timed_out {
        return (
            Verdict::Inconclusive(format!("timeout after {}s", timeout.as_secs())),
            None,
        );
    }
    classify(&stdout_str)
}

/// Map a `ty check --output json` result to a canonical [`Verdict`].
fn classify(stdout: &str) -> (Verdict, Option<u64>) {
    // The whole JSON document is on stdout; be defensive and also accept the
    // last JSON-parseable line (jsonl/aggregator shapes).
    let parsed = serde_json::from_str::<CheckJson>(stdout.trim())
        .ok()
        .or_else(|| {
            stdout
                .lines()
                .rev()
                .find_map(|l| serde_json::from_str::<CheckJson>(l.trim()).ok())
        });
    let Some(j) = parsed else {
        return (
            Verdict::Failed("no parseable JSON verdict".to_string()),
            None,
        );
    };
    let states = j
        .statistics
        .as_ref()
        .and_then(|s| s.states_distinct.or(s.states_found));
    let et = j.result.error_type.as_deref().unwrap_or("");
    let verdict = match j.result.status.as_str() {
        "ok" => Verdict::Safe,
        "vacuous" => Verdict::Vacuous,
        "limit_reached" => Verdict::Inconclusive("state/depth limit reached".to_string()),
        "timeout" => Verdict::Inconclusive("engine timeout".to_string()),
        "interrupted" => Verdict::Inconclusive("interrupted".to_string()),
        "error" => match et {
            "deadlock" => Verdict::Deadlock,
            "invariant_violation" => Verdict::InvariantViolation,
            "liveness_violation" => Verdict::LivenessViolation,
            "property_violation" => Verdict::PropertyViolation,
            "assume_violation" => Verdict::AssumeViolation,
            other => {
                let code = j
                    .result
                    .error_code
                    .as_deref()
                    .map(|c| format!(" [{c}]"))
                    .unwrap_or_default();
                let kind = if other.is_empty() {
                    "runtime error"
                } else {
                    other
                };
                Verdict::Failed(format!("{kind}{code}"))
            }
        },
        other => Verdict::Failed(format!("unexpected status `{other}`")),
    };
    (verdict, states)
}

fn print_human(file: &Path, results: &[ModeResult], disagreement: bool, conclusive: usize) {
    println!("Cross-mode verdict parity: {}", file.display());
    println!();
    // The oracle's verdict (if conclusive) is the trusted reference.
    let oracle_key = results.first().and_then(|r| r.verdict.key());
    for r in results {
        let states = r
            .states
            .map(|s| format!("  ·  {s} states"))
            .unwrap_or_default();
        let agreement = match (r.verdict.key(), oracle_key) {
            (Some(k), Some(o)) if r.name == "interpreter-oracle" => {
                let _ = (k, o);
                "  (oracle / reference)".to_string()
            }
            (Some(k), Some(o)) if k == o => "  ✓ agrees with oracle".to_string(),
            (Some(_), Some(_)) => "  ✗ DIFFERS FROM ORACLE".to_string(),
            _ => String::new(),
        };
        println!(
            "  {:<20} {}{}{}  ({:.2}s)",
            r.name,
            r.verdict.label(),
            states,
            agreement,
            r.elapsed.as_secs_f64()
        );
    }
    println!();
    if disagreement {
        println!("DISAGREEMENT — engines reached different verdicts. This is a SOUNDNESS ALERT:");
        println!("  the tree-walking interpreter is the trusted oracle; any engine that differs");
        println!("  from it has a soundness bug (e.g. a symbolic-safe lane masking a deadlock, or");
        println!("  a native-codegen divergence). Re-run the diverging engine to reproduce.");
    } else if conclusive == 0 {
        println!(
            "INCONCLUSIVE — no engine reached a definitive verdict (all bounded / timed out /"
        );
        println!("  unavailable). Raise --timeout or --max-states, or check backend availability.");
    } else {
        println!("PARITY — all {conclusive} conclusive engine(s) agree. No cross-mode divergence.");
    }
}

fn print_json(file: &Path, results: &[ModeResult], disagreement: bool, conclusive: usize) {
    let oracle_key = results.first().and_then(|r| r.verdict.key());
    let engines: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "engine": r.name,
                "verdict": r.verdict.key().unwrap_or("inconclusive"),
                "detail": r.verdict.label(),
                "conclusive": r.verdict.is_conclusive(),
                "agrees_with_oracle": match (r.verdict.key(), oracle_key) {
                    (Some(k), Some(o)) => Some(k == o),
                    _ => None,
                },
                "states": r.states,
                "elapsed_seconds": r.elapsed.as_secs_f64(),
            })
        })
        .collect();
    let status = if disagreement {
        "disagreement"
    } else if conclusive == 0 {
        "inconclusive"
    } else {
        "parity"
    };
    let doc = serde_json::json!({
        "tool": "ty parity",
        "schema": "ty.parity/v1",
        "spec_file": file.display().to_string(),
        "status": status,
        "conclusive_engines": conclusive,
        "engines": engines,
    });
    match serde_json::to_string_pretty(&doc) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("error: failed to serialize parity JSON: {e}"),
    }
}
