// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty selfcheck` — self-service trust report for one spec.
//!
//! Composes every independent re-check TY can perform into a single command — the
//! self-service equivalent of a "required gate" (TY has no CI by design): a user runs
//! `ty selfcheck MySpec.tla` and gets a scorecard of orthogonal confirmations of the
//! verdict, each from a different trust angle:
//!   * cross-mode verdict parity (`ty parity`) — independent engines agree;
//!   * verdict re-check — a VIOLATED verdict's counterexample replays (`ty verdict-check`,
//!     eval-only), or a SAFE verdict carries a re-checkable inductive proof
//!     (`ty certify` + `ty cert-check`).
//! Incompleteness is printed as a caveat, never silent confidence. Exits 0 when every
//! applicable check is consistent, 1 when any check disagrees/rejects.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::cli_schema::AuditOutputFormat;

#[derive(Deserialize)]
struct CheckJson {
    result: CheckResultJson,
}
#[derive(Deserialize)]
struct CheckResultJson {
    status: String,
    #[serde(default)]
    error_type: Option<String>,
}
#[derive(Deserialize)]
struct ParityJson {
    status: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail,
    Caveat,
    NotApplicable,
}

impl Outcome {
    fn symbol(self) -> &'static str {
        match self {
            Outcome::Pass => "PASS",
            Outcome::Fail => "FAIL",
            Outcome::Caveat => "CAVEAT",
            Outcome::NotApplicable => "n/a",
        }
    }
}

struct Row {
    name: &'static str,
    outcome: Outcome,
    detail: String,
}

/// Run `ty selfcheck`.
pub(crate) fn cmd_selfcheck(
    file: &Path,
    config: Option<&Path>,
    format: AuditOutputFormat,
) -> Result<()> {
    if !file.is_file() {
        bail!("spec file not found: {}", file.display());
    }
    let cfg = resolve_config(file, config);
    let exe = std::env::current_exe().context("could not resolve the `ty` executable path")?;
    let scratch = ScratchDir::new()?;

    let mut rows: Vec<Row> = Vec::new();

    // --- 1. Determine the verdict via `ty check --output json`. ---
    let mut check_args = vec!["check".to_string(), path(file)];
    push_config(&mut check_args, &cfg);
    check_args.push("--output".into());
    check_args.push("json".into());
    let (_c, check_out) = run(&exe, &check_args);
    let verdict = classify_check(&check_out);
    rows.push(Row {
        name: "verdict",
        outcome: Outcome::NotApplicable,
        detail: verdict.label().to_string(),
    });

    // --- 2. Cross-mode verdict parity (`ty parity`). ---
    let mut parity_args = vec!["parity".to_string(), path(file)];
    push_config(&mut parity_args, &cfg);
    parity_args.push("--format".into());
    parity_args.push("json".into());
    let (_p, parity_out) = run(&exe, &parity_args);
    let parity_status = serde_json::from_str::<ParityJson>(&parity_out)
        .map(|p| p.status)
        .unwrap_or_else(|_| "unknown".to_string());
    rows.push(match parity_status.as_str() {
        "parity" => Row {
            name: "cross-mode parity",
            outcome: Outcome::Pass,
            detail: "independent engines (interpreter / fused / native) agree".into(),
        },
        "disagreement" => Row {
            name: "cross-mode parity",
            outcome: Outcome::Fail,
            detail: "engines DISAGREE — soundness alert (run `ty parity` for detail)".into(),
        },
        _ => Row {
            name: "cross-mode parity",
            outcome: Outcome::Caveat,
            detail: "no engine reached a conclusive comparable verdict".into(),
        },
    });

    // --- 3. Verdict re-check, per direction. ---
    match verdict {
        Verdict::Violation => {
            let env_path = scratch.path.join("verdict.json");
            let mut emit_args = vec!["verdict-emit".to_string(), path(file)];
            push_config(&mut emit_args, &cfg);
            emit_args.push("--out".into());
            emit_args.push(path(&env_path));
            let (emit_code, _emit_out) = run(&exe, &emit_args);
            if emit_code == 0 && env_path.is_file() {
                let (code, out) = run(&exe, &["verdict-check".to_string(), path(&env_path)]);
                rows.push(recheck_row("counterexample re-check", code, &out));
            } else {
                rows.push(Row {
                    name: "counterexample re-check",
                    outcome: Outcome::Caveat,
                    detail: "could not emit a verdict envelope for this violation".into(),
                });
            }
        }
        Verdict::Safe => {
            let cert_path = scratch.path.join("cert.json");
            let mut cert_args = vec!["certify".to_string(), path(file)];
            push_config(&mut cert_args, &cfg);
            cert_args.push("--out".into());
            cert_args.push(path(&cert_path));
            let (cert_code, _cert_out) = run(&exe, &cert_args);
            if cert_code == 0 && cert_path.is_file() {
                let (code, out) = run(&exe, &["cert-check".to_string(), path(&cert_path)]);
                rows.push(recheck_row("inductive-proof re-check", code, &out));
            } else {
                // No inductive proof — the safe verdict rests on exhaustive BFS +
                // cross-mode parity, which is sound but not a re-checkable proof object.
                rows.push(Row {
                    name: "inductive-proof re-check",
                    outcome: Outcome::Caveat,
                    detail: "spec not in the inductive-provable class; safety rests on \
                             exhaustive search + cross-mode parity (no standalone proof object)"
                        .into(),
                });
            }
        }
        Verdict::Vacuous => rows.push(Row {
            name: "verdict re-check",
            outcome: Outcome::Caveat,
            detail:
                "VACUOUS — the verdict rests on an empty/unexercised basis; re-examine the spec"
                    .into(),
        }),
        Verdict::Bounded => rows.push(Row {
            name: "verdict re-check",
            outcome: Outcome::Caveat,
            detail: "search was BOUNDED (limit reached) — absence of error is not exhaustive"
                .into(),
        }),
        Verdict::Other(_) => rows.push(Row {
            name: "verdict re-check",
            outcome: Outcome::NotApplicable,
            detail: "no re-check applies to this verdict".into(),
        }),
    }

    // --- 4. Native↔interpreter translation validation ("compiles in Trust"). ---
    // Run the native trust-cg compiled-BFS backend with the per-parent interpreter
    // crosscheck on; key off the crosscheck markers (NOT the spec's own verdict).
    rows.push(native_equivalence_row(&exe, file, &cfg));

    let any_fail = rows.iter().any(|r| r.outcome == Outcome::Fail);
    match format {
        AuditOutputFormat::Json => print_json(file, &rows, any_fail),
        AuditOutputFormat::Human => print_human(file, &rows, any_fail),
    }
    if any_fail {
        std::process::exit(1);
    }
    Ok(())
}

enum Verdict {
    Safe,
    Violation,
    Vacuous,
    Bounded,
    Other(String),
}

impl Verdict {
    fn label(&self) -> &str {
        match self {
            Verdict::Safe => "SAFE (no error found)",
            Verdict::Violation => "VIOLATION",
            Verdict::Vacuous => "VACUOUS",
            Verdict::Bounded => "bounded (limit reached)",
            Verdict::Other(s) => s,
        }
    }
}

fn classify_check(stdout: &str) -> Verdict {
    let Ok(j) = serde_json::from_str::<CheckJson>(stdout.trim()) else {
        return Verdict::Other("unparseable check output".to_string());
    };
    let et = j.result.error_type.as_deref().unwrap_or("");
    match (j.result.status.as_str(), et) {
        ("ok", _) => Verdict::Safe,
        ("vacuous", _) => Verdict::Vacuous,
        ("limit_reached", _) => Verdict::Bounded,
        ("error", "invariant_violation" | "deadlock" | "property_violation") => Verdict::Violation,
        (other, _) => Verdict::Other(other.to_string()),
    }
}

/// Trust-internal env enabling the native trust-cg compiled-BFS path + the per-parent
/// interpreter crosscheck.
const NATIVE_CROSSCHECK_ENV: &[(&str, &str)] = &[
    ("TY_COMPILED_BFS_INTERPRETER_CROSSCHECK", "1"),
    ("TY_trust_cg", "1"),
    ("TY_TRUST_CG_BFS", "1"),
    ("TY_TRUST_CG_EXISTS", "1"),
    ("TY_BYTECODE_VM", "1"),
];

/// Run the native backend with the interpreter crosscheck and report whether the
/// sampled per-parent crosscheck found any native↔interpreter successor divergence.
fn native_equivalence_row(exe: &Path, file: &Path, cfg: &Option<PathBuf>) -> Row {
    const NAME: &str = "native ≡ interpreter";
    let mut args = vec![
        "check".to_string(),
        path(file),
        "--backend".into(),
        "trust-cg".into(),
        "--workers".into(),
        "1".into(),
    ];
    push_config(&mut args, cfg);

    let mut cmd = Command::new(exe);
    cmd.args(&args);
    for (k, v) in NATIVE_CROSSCHECK_ENV {
        cmd.env(k, v);
    }
    let stderr = {
        let child = cmd.stdout(Stdio::null()).stderr(Stdio::piped()).spawn();
        match child {
            Ok(mut c) => {
                let mut s = String::new();
                if let Some(mut se) = c.stderr.take() {
                    let _ = se.read_to_string(&mut s);
                }
                let _ = c.wait();
                s
            }
            Err(e) => {
                return Row {
                    name: NAME,
                    outcome: Outcome::Caveat,
                    detail: format!("could not run native crosscheck: {e}"),
                }
            }
        }
    };
    // SOUNDNESS: only the per-parent compiled-BFS loop runs the interpreter crosscheck;
    // the native FUSED fast path does NOT. So require the POSITIVE crosscheck-active
    // marker — NOT merely "compiled BFS active" — before claiming native ≡ interpreter,
    // or a fused-fast-path run would be falsely reported as validated.
    let crosscheck_ran = stderr.contains("[compiled-bfs-xcheck] active");
    let diverged = stderr.contains("crosscheck found")
        || stderr.contains("[compiled-bfs-crosscheck] missing-edge")
        || stderr.contains("[compiled-bfs-crosscheck] extra-edge");
    if diverged {
        Row {
            name: NAME,
            outcome: Outcome::Fail,
            detail: "native trust-cg computed a DIFFERENT successor set than the interpreter \
                     (native-codegen soundness bug)"
                .into(),
        }
    } else if crosscheck_ran {
        // HONESTY: the default-on fused crosscheck is SAMPLED — it compares a
        // subset of parents per level (stride-sampled, capped per level) on the
        // first few BFS levels only — so it must NOT claim "every explored
        // state" (only tiny specs get full coverage as a degenerate case).
        Row {
            name: NAME,
            outcome: Outcome::Pass,
            detail: "sampled per-parent native↔interpreter crosscheck ran and found no \
                     successor-set divergence (sampled parents on early BFS levels; not \
                     exhaustive for large runs) — \"compiles in Trust\""
                .into(),
        }
    } else {
        Row {
            name: NAME,
            outcome: Outcome::NotApplicable,
            detail: "per-state crosscheck not exercised (native fused/JIT fast path, not \
                     compiled-BFS-eligible, or non-ay) — the cross-mode parity row above still \
                     validates the native backend's verdict + state count against the interpreter"
                .into(),
        }
    }
}

fn recheck_row(name: &'static str, code: i32, out: &str) -> Row {
    let first = out.lines().next().unwrap_or("").trim().to_string();
    match code {
        0 => Row {
            name,
            outcome: Outcome::Pass,
            detail: if first.is_empty() {
                "VERIFIED".into()
            } else {
                first
            },
        },
        2 => Row {
            name,
            outcome: Outcome::Caveat,
            detail: if first.is_empty() {
                "INCONCLUSIVE".into()
            } else {
                first
            },
        },
        _ => Row {
            name,
            outcome: Outcome::Fail,
            detail: if first.is_empty() {
                "REJECTED".into()
            } else {
                first
            },
        },
    }
}

fn print_human(file: &Path, rows: &[Row], any_fail: bool) {
    println!("Self-check trust report: {}\n", file.display());
    let width = rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
    for r in rows {
        println!(
            "  [{:^6}] {:<width$}  {}",
            r.outcome.symbol(),
            r.name,
            r.detail,
            width = width
        );
    }
    println!();
    if any_fail {
        println!(
            "TRUST REPORT: FAILED — an independent check disagreed; do not trust this verdict."
        );
    } else {
        println!(
            "TRUST REPORT: consistent — every applicable independent check agreed (see CAVEATs \
             for the limits of what was confirmed)."
        );
    }
}

fn print_json(file: &Path, rows: &[Row], any_fail: bool) {
    let checks: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "check": r.name,
                "outcome": r.outcome.symbol(),
                "detail": r.detail,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "tool": "ty selfcheck",
        "schema": "ty.selfcheck/v1",
        "spec_file": file.display().to_string(),
        "trusted": !any_fail,
        "checks": checks,
    });
    match serde_json::to_string_pretty(&doc) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("error: failed to serialize selfcheck JSON: {e}"),
    }
}

fn resolve_config(file: &Path, config: Option<&Path>) -> Option<PathBuf> {
    if let Some(c) = config {
        return Some(c.to_path_buf());
    }
    let derived = file.with_extension("cfg");
    derived.is_file().then_some(derived)
}

fn push_config(args: &mut Vec<String>, cfg: &Option<PathBuf>) {
    if let Some(c) = cfg {
        args.push("--config".into());
        args.push(path(c));
    }
}

fn path(p: &Path) -> String {
    p.display().to_string()
}

/// Run a `ty` subcommand, capturing exit code + stdout (stderr, which carries
/// telemetry, is discarded so it cannot pollute JSON parsing). The detail lines we
/// surface (`VERIFIED`/`REJECTED`/JSON) are all written to stdout.
fn run(exe: &Path, args: &[String]) -> (i32, String) {
    let mut child = match Command::new(exe)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return (-1, format!("spawn failed: {e}")),
    };
    let mut out = String::new();
    if let Some(mut so) = child.stdout.take() {
        let _ = so.read_to_string(&mut out);
    }
    let status = child.wait().ok();
    (status.and_then(|s| s.code()).unwrap_or(-1), out)
}

/// A temp directory that cleans itself up.
struct ScratchDir {
    path: PathBuf,
}
impl ScratchDir {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!("ty-selfcheck-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).with_context(|| "create scratch dir")?;
        Ok(ScratchDir { path })
    }
}
impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
