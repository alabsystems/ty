// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Differential oracle harness: interpreter vs trust-codegen permanent parity gate.
//!
//! Part of #4252 (Stream 6). The tree-walking interpreter is the permanent
//! correctness oracle for the TY model checker. Every compiled backend
//! (trust_cg AOT today, more later) MUST match the interpreter on every spec.
//! Any divergence is a compiler bug and blocks the build.
//!
//! # Current scope
//!
//! Rollout Step 1 of the differential-oracle harness:
//!
//! - Wire `--oracle-mode {off,compare,fail-closed}` and `TY_ORACLE` env var
//!   to `ty diagnose`.
//! - Wire `--backend {interpreter,trust_cg}` to `ty check`. `--backend trust_cg`
//!   is prototype/gated: eligible specs may compile native artifacts, while
//!   ineligible or unavailable paths can fall back for soundness or emit
//!   `backend_unavailable`.
//! - For each spec, run `ty check` once with each backend, collect both
//!   `SubprocessOutput`s, diff, classify as `OraclePass`, `OracleDivergence`,
//!   `OracleInfra`, or `OracleSkip`.
//! - Emit `metrics/oracle_parity.json` with per-spec verdicts.
//! - Fail the process with exit code 1 in `fail-closed` mode on any
//!   `OracleDivergence`. `OracleInfra` does NOT fail (pre-existing crash,
//!   missing trust-codegen feature, etc.).
//!
//! # Design references
//!
//! - #4252 Stream 6 (differential-oracle harness)
//! - trust_cg / trust-ir supremacy plan, §11 Stream 6

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;

use super::runner::run_one_spec_with_env;
use super::types::{Baseline, BaselineSpec, DiagnoseCheckPolicy, SpecResult, SpecVerdict};

/// Per-spec verdict for the differential oracle harness.
// The `Oracle` prefix is load-bearing: with `rename_all = "snake_case"` it
// produces the stable serialized verdict strings (`oracle_pass`, ...) consumed
// by `metrics/oracle_parity.json` and `label()`. Stripping it would change the
// on-disk report format, so keep the shared prefix.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum OracleVerdict {
    /// Both backends completed with identical state count, status, and error type.
    OraclePass,
    /// Backends disagree on at least one axis — this is a P0 compiler bug.
    OracleDivergence,
    /// trust-codegen backend reported `backend_unavailable` or crashed for
    /// infrastructure reasons while the interpreter completed. Not a
    /// divergence but also not a pass — it means the oracle could not run.
    OracleInfra,
    /// Interpreter side failed (pre-existing TY bug to fix separately).
    /// The oracle cannot make a judgment because the reference is broken.
    OracleInterpreterBroken,
    /// The spec was not eligible for the harness (missing source paths, etc.).
    /// Part of the documented verdict set (Step 1 design scope); the skip path
    /// is not yet exercised by `classify_oracle` but is wired through
    /// `label()`/tally/summary, so keep the variant.
    #[allow(dead_code)]
    OracleSkip,
}

impl OracleVerdict {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::OraclePass => "oracle_pass",
            Self::OracleDivergence => "oracle_divergence",
            Self::OracleInfra => "oracle_infra",
            Self::OracleInterpreterBroken => "oracle_interpreter_broken",
            Self::OracleSkip => "oracle_skip",
        }
    }
}

/// A single differing axis between interpreter and trust-codegen results.
#[derive(Clone, Debug, Serialize)]
pub(super) struct Divergence {
    pub(super) axis: DivergenceAxis,
    pub(super) interpreter: Option<String>,
    pub(super) trust_cg: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DivergenceAxis {
    StateCount,
    Status,
    ErrorType,
}

/// Full result of one oracle run on one spec.
#[derive(Clone, Debug, Serialize)]
pub(super) struct OracleSpecResult {
    pub(super) name: String,
    pub(super) verdict_label: &'static str,
    pub(super) interpreter_status: Option<String>,
    pub(super) interpreter_states: Option<u64>,
    pub(super) interpreter_error: Option<String>,
    pub(super) trust_cg_status: Option<String>,
    pub(super) trust_cg_states: Option<u64>,
    pub(super) trust_cg_error: Option<String>,
    pub(super) divergences: Vec<Divergence>,
    pub(super) interpreter_time_seconds: f64,
    pub(super) trust_cg_time_seconds: f64,
    /// Verdict as an enum (for tally computation); not serialized — the
    /// stable string form is `verdict_label`.
    #[serde(skip_serializing)]
    pub(super) verdict: OracleVerdict,
    /// Optional note: infra-gap reason, skip rationale, etc.
    pub(super) note: Option<String>,
}

/// Aggregate tally across all specs.
#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct OracleTally {
    pub(super) pass: usize,
    pub(super) divergence: usize,
    pub(super) infra: usize,
    pub(super) interpreter_broken: usize,
    pub(super) skip: usize,
}

impl OracleTally {
    fn from_results(results: &[OracleSpecResult]) -> Self {
        let mut t = OracleTally::default();
        for r in results {
            match r.verdict {
                OracleVerdict::OraclePass => t.pass += 1,
                OracleVerdict::OracleDivergence => t.divergence += 1,
                OracleVerdict::OracleInfra => t.infra += 1,
                OracleVerdict::OracleInterpreterBroken => t.interpreter_broken += 1,
                OracleVerdict::OracleSkip => t.skip += 1,
            }
        }
        t
    }
}

/// Top-level JSON report written to `metrics/oracle_parity.json`.
#[derive(Serialize)]
pub(super) struct OracleReport {
    pub(super) schema_version: u32,
    pub(super) generated_at: String,
    pub(super) mode: &'static str,
    pub(super) tally: OracleTally,
    pub(super) specs: Vec<OracleSpecResult>,
}

/// Compare an interpreter-run `SpecResult` against an trust_cg-run one and
/// produce an `OracleSpecResult`.
pub(super) fn classify_oracle(
    name: &str,
    interp: &SpecResult,
    trust_cg: &SpecResult,
) -> OracleSpecResult {
    // Interpreter side broken: we cannot trust the oracle.
    if matches!(
        interp.verdict,
        SpecVerdict::Crash | SpecVerdict::Timeout | SpecVerdict::OomKilled
    ) {
        return OracleSpecResult {
            name: name.to_string(),
            verdict_label: OracleVerdict::OracleInterpreterBroken.label(),
            interpreter_status: interp.ty_status.clone(),
            interpreter_states: interp.ty_states,
            interpreter_error: interp.error_details.clone(),
            trust_cg_status: trust_cg.ty_status.clone(),
            trust_cg_states: trust_cg.ty_states,
            trust_cg_error: trust_cg.error_details.clone(),
            divergences: Vec::new(),
            interpreter_time_seconds: interp.time_seconds,
            trust_cg_time_seconds: trust_cg.time_seconds,
            verdict: OracleVerdict::OracleInterpreterBroken,
            note: Some(format!(
                "interpreter verdict {}: oracle reference is broken — file as pre-existing bug",
                interp.verdict.label()
            )),
        };
    }

    // trust-codegen side failed for infra reasons (backend_unavailable, crash, skip).
    // Not a divergence — just means the compiled path could not run.
    let trust_cg_infra = matches!(
        trust_cg.verdict,
        SpecVerdict::Crash | SpecVerdict::Timeout | SpecVerdict::OomKilled | SpecVerdict::Skip
    ) || trust_cg
        .ty_status
        .as_deref()
        .is_some_and(|s| s == "backend_unavailable");

    if trust_cg_infra {
        return OracleSpecResult {
            name: name.to_string(),
            verdict_label: OracleVerdict::OracleInfra.label(),
            interpreter_status: interp.ty_status.clone(),
            interpreter_states: interp.ty_states,
            interpreter_error: interp.error_details.clone(),
            trust_cg_status: trust_cg.ty_status.clone(),
            trust_cg_states: trust_cg.ty_states,
            trust_cg_error: trust_cg.error_details.clone(),
            divergences: Vec::new(),
            interpreter_time_seconds: interp.time_seconds,
            trust_cg_time_seconds: trust_cg.time_seconds,
            verdict: OracleVerdict::OracleInfra,
            note: Some(format!(
                "trust-cg side unavailable/crashed: {}",
                trust_cg.ty_status.as_deref().unwrap_or("unknown")
            )),
        };
    }

    // Both sides completed. Diff each axis.
    let mut divs = Vec::new();
    if interp.ty_states != trust_cg.ty_states {
        divs.push(Divergence {
            axis: DivergenceAxis::StateCount,
            interpreter: interp.ty_states.map(|v| v.to_string()),
            trust_cg: trust_cg.ty_states.map(|v| v.to_string()),
        });
    }
    if interp.ty_status != trust_cg.ty_status {
        divs.push(Divergence {
            axis: DivergenceAxis::Status,
            interpreter: interp.ty_status.clone(),
            trust_cg: trust_cg.ty_status.clone(),
        });
    }
    if interp.error_details != trust_cg.error_details {
        // Only flag when both sides agree the spec is erroring. If only one
        // side reports an error, that's already captured by status mismatch.
        let both_error = interp.ty_status.as_deref().map_or(false, |s| s != "ok")
            && trust_cg.ty_status.as_deref().map_or(false, |s| s != "ok");
        if both_error {
            divs.push(Divergence {
                axis: DivergenceAxis::ErrorType,
                interpreter: interp.error_details.clone(),
                trust_cg: trust_cg.error_details.clone(),
            });
        }
    }

    let verdict = if divs.is_empty() {
        OracleVerdict::OraclePass
    } else {
        OracleVerdict::OracleDivergence
    };

    OracleSpecResult {
        name: name.to_string(),
        verdict_label: verdict.label(),
        interpreter_status: interp.ty_status.clone(),
        interpreter_states: interp.ty_states,
        interpreter_error: interp.error_details.clone(),
        trust_cg_status: trust_cg.ty_status.clone(),
        trust_cg_states: trust_cg.ty_states,
        trust_cg_error: trust_cg.error_details.clone(),
        divergences: divs,
        interpreter_time_seconds: interp.time_seconds,
        trust_cg_time_seconds: trust_cg.time_seconds,
        verdict,
        note: None,
    }
}

/// Entry point: run the oracle harness across the selected specs.
///
/// Returns (results, tally). In `FailClosed` mode with any divergences,
/// the caller is responsible for exiting non-zero.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_oracle_harness(
    spec_names: &[String],
    baseline: &Baseline,
    exe: &Path,
    examples_dir: &Path,
    cli_timeout_default: u64,
    retries: u32,
    checker_policy: DiagnoseCheckPolicy,
    mode_label: &'static str,
) -> Vec<OracleSpecResult> {
    let total = spec_names.len();
    let done_count = AtomicUsize::new(0);
    let mut results = Vec::with_capacity(total);
    for name in spec_names {
        let spec = match baseline.specs.get(name) {
            Some(s) => s,
            None => continue,
        };
        let timeout = effective_timeout(cli_timeout_default, spec);
        let result = run_one_oracle(
            name,
            spec,
            exe,
            examples_dir,
            timeout,
            retries,
            checker_policy,
        );
        let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
        eprint!(
            "\r[{}/{}] {} => {}     ",
            done, total, name, result.verdict_label
        );
        let _ = std::io::stderr().flush();
        results.push(result);
    }
    eprintln!();
    let _ = mode_label;
    results
}

fn effective_timeout(cli_default: u64, spec: &BaselineSpec) -> Duration {
    let secs = spec
        .diagnose_timeout_seconds
        .filter(|&t| t > cli_default)
        .unwrap_or(cli_default);
    Duration::from_secs(secs)
}

fn run_one_oracle(
    name: &str,
    spec: &BaselineSpec,
    exe: &Path,
    examples_dir: &Path,
    timeout: Duration,
    retries: u32,
    checker_policy: DiagnoseCheckPolicy,
) -> OracleSpecResult {
    // Run interpreter side first (the oracle).
    let start = Instant::now();
    let interp = run_one_spec_with_env(
        name,
        spec,
        exe,
        examples_dir,
        timeout,
        retries,
        checker_policy,
        "interpreter",
    );
    let interp_elapsed = start.elapsed().as_secs_f64();
    let _ = interp_elapsed;

    // Run trust-codegen side second. Depending on feature/build/spec eligibility, this
    // can compile native artifacts, fall back, or report backend_unavailable.
    let trust_cg = run_one_spec_with_env(
        name,
        spec,
        exe,
        examples_dir,
        timeout,
        retries,
        checker_policy,
        "trust-cg",
    );

    classify_oracle(name, &interp, &trust_cg)
}

/// Write the oracle parity report to disk.
pub(super) fn write_report(
    path: &Path,
    results: Vec<OracleSpecResult>,
    mode: &'static str,
) -> Result<OracleTally> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create parent directory {}", parent.display()))?;
    }
    let tally = OracleTally::from_results(&results);
    let report = OracleReport {
        schema_version: 1,
        generated_at: chrono::Utc::now().to_rfc3339(),
        mode,
        tally: tally.clone(),
        specs: results,
    };
    let json = serde_json::to_string_pretty(&report).context("serialize oracle report")?;
    std::fs::write(path, json)
        .with_context(|| format!("write oracle report {}", path.display()))?;
    Ok(tally)
}

/// Convenience: emit a human-readable summary line to stderr.
pub(super) fn emit_summary(tally: &OracleTally, path: &Path) {
    eprintln!("Oracle harness summary (#4252 Stream 6):");
    eprintln!("  pass:                {}", tally.pass);
    eprintln!("  divergence:          {}", tally.divergence);
    eprintln!("  infra (trust-cg unavailable/crash): {}", tally.infra);
    eprintln!("  interpreter_broken:  {}", tally.interpreter_broken);
    eprintln!("  skip:                {}", tally.skip);
    eprintln!("  report:              {}", path.display());
}

/// Resolve the oracle mode from the CLI flag and/or environment variable.
/// CLI flag takes precedence over env var.
pub(super) fn resolve_oracle_mode(
    cli: Option<crate::cli_schema::DiagnoseOracleMode>,
) -> crate::cli_schema::DiagnoseOracleMode {
    use crate::cli_schema::DiagnoseOracleMode;
    if let Some(m) = cli {
        return m;
    }
    match std::env::var("TY_ORACLE").ok().as_deref() {
        Some("off") | Some("") | None => DiagnoseOracleMode::Off,
        Some("compare") => DiagnoseOracleMode::Compare,
        Some("fail-closed") | Some("fail_closed") | Some("failclosed") => {
            DiagnoseOracleMode::FailClosed
        }
        Some(other) => {
            eprintln!(
                "warning: TY_ORACLE={other:?} not recognized; use off|compare|fail-closed. Defaulting to off."
            );
            DiagnoseOracleMode::Off
        }
    }
}

/// Resolve the oracle output path. Defaults to `metrics/oracle_parity.json`
/// if the CLI flag was not provided.
pub(super) fn resolve_output_path(cli: Option<PathBuf>) -> PathBuf {
    cli.unwrap_or_else(|| PathBuf::from("metrics/oracle_parity.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd_diagnose::types::{SpecResult, SpecVerdict, TimeoutSource};

    fn mk_result(states: Option<u64>, status: Option<&str>, verdict: SpecVerdict) -> SpecResult {
        SpecResult {
            name: "Spec".to_string(),
            verdict,
            ty_status: status.map(|s| s.to_string()),
            ty_states: states,
            tlc_status: "pass".to_string(),
            tlc_states: states,
            tlc_error_type: None,
            error_details: None,
            expected_mismatch_reason: None,
            time_seconds: 0.0,
            timeout_seconds: 60,
            timeout_source: TimeoutSource::Cli,
        }
    }

    #[test]
    fn oracle_pass_when_both_sides_match() {
        let a = mk_result(Some(100), Some("ok"), SpecVerdict::Pass);
        let b = mk_result(Some(100), Some("ok"), SpecVerdict::Pass);
        let r = classify_oracle("Spec", &a, &b);
        assert_eq!(r.verdict, OracleVerdict::OraclePass);
        assert!(r.divergences.is_empty());
    }

    #[test]
    fn oracle_divergence_on_state_mismatch() {
        let a = mk_result(Some(100), Some("ok"), SpecVerdict::Pass);
        let b = mk_result(Some(99), Some("ok"), SpecVerdict::Pass);
        let r = classify_oracle("Spec", &a, &b);
        assert_eq!(r.verdict, OracleVerdict::OracleDivergence);
        assert_eq!(r.divergences.len(), 1);
        assert!(matches!(r.divergences[0].axis, DivergenceAxis::StateCount));
    }

    #[test]
    fn oracle_infra_when_trust_cg_backend_unavailable() {
        let interp = mk_result(Some(50), Some("ok"), SpecVerdict::Pass);
        let mut trust_cg = mk_result(None, Some("backend_unavailable"), SpecVerdict::Crash);
        trust_cg.error_details = Some("trust-cg unavailable".to_string());
        let r = classify_oracle("Spec", &interp, &trust_cg);
        assert_eq!(r.verdict, OracleVerdict::OracleInfra);
    }

    #[test]
    fn oracle_interpreter_broken_propagates() {
        let interp = mk_result(None, None, SpecVerdict::Crash);
        let trust_cg = mk_result(Some(10), Some("ok"), SpecVerdict::Pass);
        let r = classify_oracle("Spec", &interp, &trust_cg);
        assert_eq!(r.verdict, OracleVerdict::OracleInterpreterBroken);
    }

    #[test]
    fn resolve_mode_env_var_compare() {
        crate::env_guard::set_var("TY_ORACLE", "compare");
        let m = resolve_oracle_mode(None);
        crate::env_guard::remove_var("TY_ORACLE");
        assert_eq!(m, crate::cli_schema::DiagnoseOracleMode::Compare);
    }

    #[test]
    fn resolve_mode_cli_beats_env() {
        crate::env_guard::set_var("TY_ORACLE", "compare");
        let m = resolve_oracle_mode(Some(crate::cli_schema::DiagnoseOracleMode::FailClosed));
        crate::env_guard::remove_var("TY_ORACLE");
        assert_eq!(m, crate::cli_schema::DiagnoseOracleMode::FailClosed);
    }
}
