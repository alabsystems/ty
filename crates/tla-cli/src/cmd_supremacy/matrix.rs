// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Baseline-backed all-runnable supremacy matrix classification.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::anti_overfit;
use super::matrix_refresh;
use super::parse;
use super::policy::{MatrixPolicy, SupremacyPolicy};
use super::runner::{run_command, CommandSpec};
#[cfg(test)]
use super::tlc_java_single_thread_args;
use super::tlc_java_single_thread_base_argv;
use crate::cli_schema::{SupremacyMatrixArgs, SupremacyMatrixRuntimeScope, SupremacyMode};

#[derive(Clone, Debug, Deserialize)]
struct SpecBaseline {
    // Deserialize-only schema fields: present in the baseline JSON for
    // documentation/validation parity but not read off this struct directly.
    #[allow(dead_code)]
    #[serde(default)]
    total_specs: Option<usize>,
    #[allow(dead_code)]
    #[serde(default)]
    specs_jcs_sha256: Option<String>,
    #[serde(default)]
    inputs: BaselineInputs,
    #[serde(default)]
    ty_refresh: Option<BaselineTyRefresh>,
    specs: BTreeMap<String, BaselineSpec>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct BaselineInputs {
    #[serde(default)]
    examples_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
struct BaselineSpec {
    tlc: BaselineMode,
    ty: BaselineMode,
    #[serde(default)]
    verified_match: bool,
    #[serde(default)]
    source: Option<BaselineSource>,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct BaselineMode {
    status: String,
    #[serde(default)]
    error_type: Option<String>,
    #[serde(default)]
    runtime_seconds: Option<f64>,
    #[serde(default)]
    states: Option<u64>,
    /// Production-default measurement axis (auto-POR/auto-symmetry free to engage),
    /// recorded alongside the pinned count-verify run by `--refresh-runtime
    /// --production-runtime true`. Backward-compatible: absent in older baselines.
    /// `production_status` is the presence marker for the whole production axis.
    #[serde(default)]
    production_status: Option<String>,
    #[serde(default)]
    production_error_type: Option<String>,
    #[serde(default)]
    production_runtime_seconds: Option<f64>,
    /// Informational only: production-default state counts are reduced by
    /// auto-POR/auto-symmetry and are never compared against TLC; the pinned
    /// `states` field owns count parity.
    #[serde(default)]
    #[allow(dead_code)]
    production_states: Option<u64>,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct BaselineSource {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    tla_path: Option<PathBuf>,
    #[serde(default)]
    cfg_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
struct BaselineTyRefresh {
    #[serde(default)]
    git_commit: Option<String>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    binary_path: Option<String>,
    #[serde(default)]
    binary_sha256: Option<String>,
    #[serde(default)]
    allow_debug_runtime: bool,
}

const DEFAULT_TLC_JAR: &str = "tlaplus/tytools.jar";
const DEFAULT_COMMUNITY_MODULES_JAR: &str = "tlaplus/CommunityModules.jar";
const DEFAULT_TLA_LIBRARY: &str = "test_specs/tla_library";
const ENV_TLA_LIBRARY: &str = "TLA_LIBRARY";
const ENV_TLA_PLUS_LIBRARY: &str = "TLA_PLUS_LIBRARY";
const MATRIX_SUMMARY_SCHEMA: &str = "ty.supremacy.matrix_summary.v1";
const RUNTIME_EVIDENCE_SCHEMA: &str = "ty.supremacy.matrix_runtime_evidence.v1";
const RUNTIME_BATCH_PLAN_SCHEMA: &str = "ty.supremacy.matrix_runtime_batch_plan.v1";
const RUNTIME_METADATA_WARNING_FIELD: &str = "matrix_runtime_refresh_metadata_warning";
const MISSING_RUNTIME_MEANING: &str = "missing_runtime means the row is a runnable, parity-verified check or simulation spec, but the baseline lacks finite positive runtime_seconds for TLC, TY, or both, or runtime evidence was collected with an undersized per-spec timeout budget";
const MISSING_RUNTIME_LAUNCH_GATE_POLICY: &str = "missing_runtime is a strict launch-gate blocker, not an unsupported spec and not a win; refresh runtime evidence before enforcing all-runnable supremacy";
const RUNTIME_REFRESH_COMPILE_JOBS_ENV: &str = "TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS";
const DEFAULT_RUNTIME_REFRESH_COMPILE_JOBS: &str = "27";
const MISSING_RUNTIME_REFRESH_COMMAND_ARGS: &[&str] = &[
    "ty",
    "supremacy",
    "matrix",
    "--baseline",
    "<baseline.json>",
    "--mode",
    "warn",
    "--format",
    "json",
    "--refresh-runtime",
    "--runtime-output-dir",
    "<output-dir>",
];
// Runtime evidence is currently a single wall-clock sample. Treat sub-10ms
// non-faster deltas as measurement noise instead of actionable regressions.
const PERF_TIE_TOLERANCE_SECONDS: f64 = 0.010;
// Below 50ms, process startup and scheduler noise dominate the checker runtime.
const PERF_TIE_TINY_RUNTIME_FLOOR_SECONDS: f64 = 0.050;
const MATRIX_SIMULATION_TRACES: u64 = 1_000;
const MATRIX_SIMULATION_DEPTH: u64 = 100;
const MATRIX_SIMULATION_SEED: u64 = 1;
const RANDOMIZED_COUNT_POLICY_REASON_PREFIX: &str = "randomized_count_policy";
const RANDOMIZED_EXTERNAL_OPERATORS: &[&str] = &["RandomElement"];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SupremacyMatrixSummary {
    pub(super) schema: &'static str,
    pub(super) verdict: SupremacyMatrixVerdict,
    pub(super) strict_pass: bool,
    pub(super) strict_blockers: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) policy: Option<SupremacyMatrixPolicySummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) build_identity: Option<SupremacyMatrixBuildIdentity>,
    pub(super) corpus: SupremacyMatrixCorpusIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) missing_runtime_diagnostics: Option<SupremacyMatrixMissingRuntimeDiagnostics>,
    pub(super) counts: SupremacyMatrixCounts,
    pub(super) next_action_counts: BTreeMap<&'static str, usize>,
    pub(super) rows: Vec<SupremacyMatrixRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SupremacyMatrixCorpusIdentity {
    pub(super) total_specs: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) specs_jcs_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SupremacyMatrixBuildIdentity {
    pub(super) git_commit: String,
    pub(super) timestamp: String,
    pub(super) ty_binary_path: String,
    pub(super) ty_binary_sha256: String,
    pub(super) allow_debug_runtime: bool,
}

impl SupremacyMatrixBuildIdentity {
    fn from_refresh(refresh: Option<&BaselineTyRefresh>) -> Option<Self> {
        let refresh = refresh?;
        Some(Self {
            git_commit: refresh.git_commit.clone()?,
            timestamp: refresh.timestamp.clone()?,
            ty_binary_path: refresh.binary_path.clone()?,
            ty_binary_sha256: refresh.binary_sha256.clone()?,
            allow_debug_runtime: refresh.allow_debug_runtime,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(super) struct SupremacyMatrixCounts {
    pub(super) unsupported: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub(super) expected_violation_match: usize,
    pub(super) tlc_error: usize,
    pub(super) tlc_timeout: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub(super) runtime_to_error: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub(super) timeout_dominance: usize,
    pub(super) ty_timeout: usize,
    pub(super) parity_fail: usize,
    pub(super) missing_runtime: usize,
    pub(super) perf_tie: usize,
    pub(super) perf_loser: usize,
    pub(super) pass: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SupremacyMatrixPolicySummary {
    pub(super) allow_runtime_to_error: bool,
    pub(super) allow_timeout_dominance: bool,
    pub(super) comparable_outcomes: usize,
    pub(super) pass: bool,
    pub(super) blockers: usize,
    pub(super) verdict: SupremacyMatrixVerdict,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SupremacyMatrixMissingRuntimeDiagnostics {
    pub(super) meaning: &'static str,
    pub(super) launch_gate_policy: &'static str,
    pub(super) specs_needing_measurement: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) missing_tlc_runtime_specs: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) missing_ty_runtime_specs: Vec<String>,
    pub(super) specs_needing_measurement_details: Vec<SupremacyMatrixMissingRuntimeDetail>,
    pub(super) refresh_command: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SupremacyMatrixMissingRuntimeDetail {
    pub(super) spec: String,
    pub(super) missing_tlc_runtime: bool,
    pub(super) missing_ty_runtime: bool,
    pub(super) reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SupremacyMatrixRow {
    pub(super) spec: String,
    pub(super) class: SupremacyMatrixClass,
    pub(super) next_action: SupremacyMatrixNextAction,
    pub(super) reason: String,
    #[serde(skip_serializing)]
    pub(super) missing_tlc_runtime: bool,
    #[serde(skip_serializing)]
    pub(super) missing_ty_runtime: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) perf_loser_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tlc_seconds: Option<f64>,
    /// TY runtime used for the speed axis: the production-default measurement
    /// when present, otherwise the pinned count-verify runtime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ty_seconds: Option<f64>,
    /// Pinned count-verify TY runtime, reported alongside `ty_seconds` whenever
    /// the speed axis used the production-default measurement instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ty_pinned_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) speedup_tlc_vs_ty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) slowdown_ty_vs_tlc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) seconds_lost_vs_tlc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) perf_loser_follow_up: Option<String>,
}

/// Canonical JSON status strings for supremacy comparisons.
///
/// These are the exact tokens emitted in baseline JSON and parsed by every
/// downstream consumer: Rust call sites (here, `cmd_diagnose`,
/// `tests/spec_regression.rs`, the Rust port at
/// `crates/tla-petri/src/bin/ty-validate-codegen-state-counts.rs`, and
/// the `ty-tlc-baseline` collector), plus any remaining Python tooling
/// (`scripts/compare_test_results.py`). Use these constants — never
/// inline `"pass"` / `"fail"` literals — so changing the wire form
/// requires touching one place. Drift here is the same class of bug as
/// the MCC qualification-1 keyword issue (see
/// `docs/mcc-2026/qualification-1/analysis.md`).
pub(crate) const SUPREMACY_STATUS_PASS: &str = "pass";
// Wire-form constant only referenced via `as_wire_str`/tests in non-test builds.
#[allow(dead_code)]
pub(crate) const SUPREMACY_STATUS_FAIL: &str = "fail";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SupremacyMatrixVerdict {
    Pass,
    Fail,
}

impl SupremacyMatrixVerdict {
    /// Canonical JSON-wire-form name for the verdict. Pinned by the
    /// `supremacy_status_wire_consts_match_enum_serialization` test below.
    #[must_use]
    #[allow(dead_code)] // only referenced from tests in non-test builds
    pub(super) fn as_wire_str(self) -> &'static str {
        match self {
            Self::Pass => SUPREMACY_STATUS_PASS,
            Self::Fail => SUPREMACY_STATUS_FAIL,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SupremacyMatrixClass {
    Unsupported,
    ExpectedViolationMatch,
    TlcError,
    TlcTimeout,
    RuntimeToError,
    TimeoutDominance,
    TyTimeout,
    ParityFail,
    MissingRuntime,
    PerfTie,
    PerfLoser,
    Pass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SupremacyMatrixNextAction {
    None,
    TriageUnsupported,
    TriageTlcError,
    RebaselineTlcTimeout,
    FixTyTimeout,
    FixParity,
    RefreshRuntime,
    RemeasurePerfTie,
    FixPerfRegression,
}

impl SupremacyMatrixNextAction {
    fn from_class(class: SupremacyMatrixClass) -> Self {
        match class {
            SupremacyMatrixClass::Pass | SupremacyMatrixClass::ExpectedViolationMatch => Self::None,
            SupremacyMatrixClass::Unsupported => Self::TriageUnsupported,
            SupremacyMatrixClass::TlcError | SupremacyMatrixClass::RuntimeToError => {
                Self::TriageTlcError
            }
            SupremacyMatrixClass::TlcTimeout | SupremacyMatrixClass::TimeoutDominance => {
                Self::RebaselineTlcTimeout
            }
            SupremacyMatrixClass::TyTimeout => Self::FixTyTimeout,
            SupremacyMatrixClass::ParityFail => Self::FixParity,
            SupremacyMatrixClass::MissingRuntime => Self::RefreshRuntime,
            SupremacyMatrixClass::PerfTie => Self::RemeasurePerfTie,
            SupremacyMatrixClass::PerfLoser => Self::FixPerfRegression,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::TriageUnsupported => "triage_unsupported",
            Self::TriageTlcError => "triage_tlc_error",
            Self::RebaselineTlcTimeout => "rebaseline_tlc_timeout",
            Self::FixTyTimeout => "fix_ty_timeout",
            Self::FixParity => "fix_parity",
            Self::RefreshRuntime => "refresh_runtime",
            Self::RemeasurePerfTie => "remeasure_perf_tie",
            Self::FixPerfRegression => "fix_perf_regression",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::TriageUnsupported => "triage unsupported row",
            Self::TriageTlcError => "triage TLC error/comparable-error policy",
            Self::RebaselineTlcTimeout => "rebaseline TLC timeout or policy",
            Self::FixTyTimeout => "fix TY timeout",
            Self::FixParity => "fix parity",
            Self::RefreshRuntime => "refresh runtime",
            Self::RemeasurePerfTie => "remeasure perf tie",
            Self::FixPerfRegression => "fix perf regression",
        }
    }
}

#[allow(dead_code)] // default-policy convenience wrapper used only from tests
pub(super) fn classify_baseline_path(path: &Path) -> Result<SupremacyMatrixSummary> {
    classify_baseline_path_with_policy(path, &MatrixPolicy::default())
}

pub(super) fn classify_baseline_path_with_policy(
    path: &Path,
    matrix_policy: &MatrixPolicy,
) -> Result<SupremacyMatrixSummary> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read baseline {}", path.display()))?;
    classify_baseline_str_with_policy(&text, matrix_policy)
        .with_context(|| format!("parse baseline {}", path.display()))
}

pub(super) fn validate_enforceable_baseline_path(path: &Path) -> Result<()> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read baseline {}", path.display()))?;
    let baseline: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline {}", path.display()))?;
    validate_enforceable_baseline_value(&baseline)
        .with_context(|| format!("validate enforceable baseline {}", path.display()))
}

pub(super) fn enforceable_baseline_corpus_identity_path(
    path: &Path,
) -> Result<SupremacyMatrixCorpusIdentity> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read baseline {}", path.display()))?;
    let baseline: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline {}", path.display()))?;
    validate_enforceable_baseline_value(&baseline)
        .with_context(|| format!("validate enforceable baseline {}", path.display()))?;
    corpus_identity_from_baseline_value(&baseline)
}

fn validate_enforceable_baseline_value(baseline: &Value) -> Result<()> {
    if baseline.get(RUNTIME_METADATA_WARNING_FIELD).is_some() {
        bail!(
            "matrix baseline carries {RUNTIME_METADATA_WARNING_FIELD}; refresh runtime evidence with a schema-supported baseline before --mode enforce"
        );
    }
    if baseline
        .pointer("/ty_refresh/allow_debug_runtime")
        .and_then(Value::as_bool)
        == Some(true)
    {
        bail!(
            "matrix baseline was refreshed with --allow-debug-runtime; debug runtime evidence is not allowed in --mode enforce"
        );
    }
    validate_baseline_promotion_metadata_value(baseline)?;
    Ok(())
}

fn validate_baseline_promotion_metadata_value(baseline: &Value) -> Result<()> {
    let Some(root) = baseline.as_object() else {
        bail!("baseline root is not an object");
    };
    let specs_obj = root
        .get("specs")
        .and_then(Value::as_object)
        .context("baseline has no 'specs' object")?;
    let existing_category_keys = root
        .get("categories")
        .and_then(Value::as_object)
        .map(|categories| categories.keys().cloned().collect::<BTreeSet<_>>())
        .unwrap_or_default();

    let expected_total_specs = json!(specs_obj.len());
    let expected_categories = Value::Object(compute_baseline_categories(
        specs_obj,
        existing_category_keys,
    ));
    let expected_stats = Value::Object(compute_baseline_stats(specs_obj));
    let expected_digest = Value::String(sha256_jcs_value(&Value::Object(specs_obj.clone()))?);

    let mut stale_fields = Vec::new();
    collect_stale_metadata_field(
        &mut stale_fields,
        root,
        "total_specs",
        &expected_total_specs,
    );
    collect_stale_metadata_field(&mut stale_fields, root, "categories", &expected_categories);
    collect_stale_metadata_field(&mut stale_fields, root, "stats", &expected_stats);
    collect_stale_metadata_field(
        &mut stale_fields,
        root,
        "specs_jcs_sha256",
        &expected_digest,
    );

    if stale_fields.is_empty() {
        return Ok(());
    }
    bail!(
        "matrix baseline promotion metadata is stale: {}",
        stale_fields.join("; ")
    )
}

fn collect_stale_metadata_field(
    stale_fields: &mut Vec<String>,
    root: &Map<String, Value>,
    field: &str,
    expected: &Value,
) {
    match root.get(field) {
        Some(actual) if actual == expected => {}
        Some(actual) => stale_fields.push(format!(
            "{field} expected {} but found {}",
            compact_json(expected),
            compact_json(actual)
        )),
        None => stale_fields.push(format!(
            "{field} missing; expected {}",
            compact_json(expected)
        )),
    }
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

pub(super) fn validate_matrix_enforce_inputs(args: &SupremacyMatrixArgs) -> Result<()> {
    validate_matrix_runtime_refresh_policy(args.mode, args.allow_debug_runtime)?;
    run_matrix_anti_overfit_scan(args)?;
    if args.mode == SupremacyMode::Enforce {
        validate_enforceable_baseline_path(&args.baseline)?;
    }
    Ok(())
}

fn run_matrix_anti_overfit_scan(args: &SupremacyMatrixArgs) -> Result<()> {
    if args.mode == SupremacyMode::Warn && args.policy.is_none() {
        return Ok(());
    }
    let Some((policy_path, policy)) = load_matrix_anti_overfit_policy(args)? else {
        return Ok(());
    };
    let report = match anti_overfit::scan(anti_overfit::AntiOverfitScanInput {
        policy_path: &policy_path,
        policy: &policy,
        baseline_path: Some(&args.baseline),
        scan_roots: &[],
        include_comments: false,
    }) {
        Ok(report) => report,
        Err(err) if args.mode == SupremacyMode::Warn => {
            eprintln!("[supremacy] WARNING: anti-overfit scan failed: {err:#}");
            return Ok(());
        }
        Err(err) => return Err(err).context("ty supremacy matrix anti-overfit scan failed"),
    };
    if !report.has_findings() {
        return Ok(());
    }

    let findings = report.finding_count();
    if args.mode == SupremacyMode::Enforce {
        bail!("ty supremacy matrix anti-overfit scan found {findings} forbidden corpus references");
    }
    eprintln!(
        "[supremacy] WARNING: anti-overfit scan found {findings} forbidden corpus references; continuing because --mode warn"
    );
    Ok(())
}

fn load_matrix_anti_overfit_policy(
    args: &SupremacyMatrixArgs,
) -> Result<Option<(PathBuf, SupremacyPolicy)>> {
    let requested_policy_path = args.policy.as_ref();
    if let Some(policy_path) = requested_policy_path {
        match SupremacyPolicy::load(policy_path) {
            Ok(policy) => return Ok(Some((policy_path.clone(), policy))),
            Err(err) if args.mode == SupremacyMode::Warn => {
                eprintln!(
                    "[supremacy] WARNING: anti-overfit policy load failed for {}: {err:#}",
                    policy_path.display(),
                );
                return Ok(None);
            }
            Err(err) => {
                if !is_matrix_only_policy_document(policy_path).with_context(|| {
                    format!("inspect matrix-only policy shape {}", policy_path.display())
                })? {
                    return Err(err).with_context(|| {
                        format!(
                            "load supremacy anti-overfit policy {}",
                            policy_path.display()
                        )
                    });
                }
                let default_policy_path = super::default_policy_path_near(policy_path);
                if policy_path == &default_policy_path {
                    return Err(err).with_context(|| {
                        format!(
                            "load supremacy anti-overfit policy {}",
                            policy_path.display()
                        )
                    });
                }
                return SupremacyPolicy::load(&default_policy_path)
                    .map(|policy| Some((default_policy_path, policy)))
                    .with_context(|| {
                        format!(
                            "load default supremacy anti-overfit policy after {} was not a full launch policy: {err:#}",
                            policy_path.display(),
                        )
                    });
            }
        }
    }

    let policy_path = super::default_policy_path();
    match SupremacyPolicy::load(&policy_path) {
        Ok(policy) => Ok(Some((policy_path, policy))),
        Err(err) if args.mode == SupremacyMode::Warn => {
            eprintln!(
                "[supremacy] WARNING: anti-overfit policy load failed for {}: {err:#}",
                policy_path.display(),
            );
            Ok(None)
        }
        Err(err) => Err(err).with_context(|| {
            format!(
                "load supremacy anti-overfit policy {}",
                policy_path.display()
            )
        }),
    }
}

fn is_matrix_only_policy_document(policy_path: &Path) -> Result<bool> {
    let text = fs::read_to_string(policy_path)
        .with_context(|| format!("read policy {}", policy_path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse policy {}", policy_path.display()))?;
    Ok(value.get("matrix_policy").is_some() && value.get("specs").is_none())
}

fn validate_matrix_runtime_refresh_policy(
    mode: SupremacyMode,
    allow_debug_runtime: bool,
) -> Result<()> {
    if mode == SupremacyMode::Enforce && allow_debug_runtime {
        bail!("--allow-debug-runtime is not allowed with --mode enforce");
    }
    Ok(())
}

pub(super) fn collect_missing_runtime_path(
    args: &SupremacyMatrixArgs,
    summary: &SupremacyMatrixSummary,
    matrix_policy: &MatrixPolicy,
) -> Result<Option<SupremacyMatrixSummary>> {
    if !args.refresh_runtime {
        return Ok(None);
    }
    validate_matrix_runtime_refresh_policy(args.mode, args.allow_debug_runtime)?;

    let baseline_path = &args.baseline;
    let text = fs::read_to_string(baseline_path)
        .with_context(|| format!("read baseline {}", baseline_path.display()))?;
    let baseline: SpecBaseline = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline {}", baseline_path.display()))?;
    let mut baseline_value: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline value {}", baseline_path.display()))?;
    let repo_root = env::current_dir().context("resolve current working directory")?;
    let config = RuntimeCollectionConfig::from_args(args, &repo_root)?;
    let examples_dir = baseline
        .inputs
        .examples_dir
        .clone()
        .or_else(default_examples_dir)
        .context("baseline inputs.examples_dir is absent and HOME is not set")?;
    let refresh_scope = runtime_refresh_scope(args.runtime_scope, &args.runtime_specs);
    let refresh_plan = matrix_refresh::plan_runtime_refresh_str(
        &text,
        baseline_path,
        Some(&examples_dir),
        refresh_scope,
    )
    .with_context(|| {
        format!(
            "plan matrix runtime refresh for {}",
            baseline_path.display()
        )
    })?;
    let selection = runtime_batch_selection(
        summary,
        &refresh_plan,
        &args.runtime_specs,
        args.runtime_limit,
    )?;
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("create {}", config.output_dir.display()))?;
    let runtime_batch_plan = write_runtime_batch_plan(
        baseline_path,
        &config.output_dir,
        args.runtime_limit,
        &args.runtime_specs,
        &selection,
        &refresh_plan,
    )?;
    eprintln!(
        "[supremacy] matrix runtime batch plan: {} selected, {} batchable selected runtime rows, {} blocked; wrote {}",
        selection.selected_specs.len(),
        refresh_plan.counts.batchable_runtime_rows,
        refresh_plan.counts.blocked_runtime_rows,
        runtime_batch_plan.display()
    );
    let tlc_jar = config.tlc_jar.clone();
    validate_file(&tlc_jar)?;
    let tlc_classpath = tlc_classpath(&tlc_jar, config.community_modules.as_deref())?;
    if let Some(tla_library) = &config.tla_library {
        validate_dir(tla_library)
            .with_context(|| format!("validate --runtime-tla-library {}", tla_library.display()))?;
    }
    validate_file(&config.ty_bin)?;
    if !selection.selected_rows.is_empty() {
        validate_runtime_ty_binary_for_refresh(&config.ty_bin, config.allow_debug_runtime)?;
        if selection_needs_trust_cg_preflight(&selection, &baseline) {
            preflight_runtime_ty_trust_cg_for_refresh(
                &config.ty_bin,
                &config.output_dir,
                &repo_root,
                config.timeout_seconds,
                &config.ty_base_env,
            )?;
        }
    }
    let metadata = RuntimeEvidenceMetadata::collect(
        &repo_root,
        &examples_dir,
        &config.ty_bin,
        &tlc_jar,
        chrono::Utc::now().to_rfc3339(),
    )
    .context("collect matrix runtime provenance")?;
    let baseline_provenance =
        RuntimeBaselineProvenance::from_metadata(&metadata, config.allow_debug_runtime);

    let mut rows = Vec::new();
    let mut checkpoint = None;
    for row in &selection.selected_rows {
        let Some(entry) = baseline.specs.get(&row.spec) else {
            continue;
        };
        let planned_row = refresh_plan.row(&row.spec);
        let collected = match collect_spec_runtime(
            &row.spec,
            entry,
            &config,
            &repo_root,
            &examples_dir,
            &tlc_classpath,
            planned_row,
        ) {
            Ok(collected) => collected,
            Err(error) => {
                eprintln!(
                    "[supremacy] {}: runtime collection failed; recording row error and continuing: {error:#}",
                    row.spec
                );
                runtime_collection_error_row(&row.spec, &config.output_dir, &error)
            }
        };
        apply_runtime_row(&mut baseline_value, &collected, &baseline_provenance);
        rows.push(collected);
        checkpoint = Some(write_runtime_refresh_checkpoint(
            baseline_path,
            &config,
            &baseline_value,
            matrix_policy,
            &runtime_batch_plan,
            &selection,
            &refresh_plan,
            &metadata,
            &rows,
            &baseline_provenance,
        )?);
    }
    let checkpoint = match checkpoint {
        Some(checkpoint) => checkpoint,
        None => write_runtime_refresh_checkpoint(
            baseline_path,
            &config,
            &baseline_value,
            matrix_policy,
            &runtime_batch_plan,
            &selection,
            &refresh_plan,
            &metadata,
            &rows,
            &baseline_provenance,
        )?,
    };
    if checkpoint.metadata_refresh == BaselineMetadataRefresh::WarningInserted {
        eprintln!(
            "[supremacy] matrix runtime metadata warning: refreshed baseline carries {RUNTIME_METADATA_WARNING_FIELD}"
        );
    }
    eprintln!(
        "[supremacy] matrix runtime evidence: wrote {}, {}, and {}",
        checkpoint.report_path.display(),
        checkpoint.refreshed_baseline.display(),
        checkpoint.matrix_after_refresh.display()
    );
    if args.mode == SupremacyMode::Enforce {
        validate_runtime_refresh_rows_promoted(&rows)?;
    }
    if args.mode == SupremacyMode::Enforce
        && checkpoint.metadata_refresh == BaselineMetadataRefresh::WarningInserted
    {
        bail!(
            "refreshed runtime baseline carries {RUNTIME_METADATA_WARNING_FIELD}; it is not valid enforce-mode evidence"
        );
    }
    Ok(Some(checkpoint.summary))
}

fn selected_runtime_rows<'a>(
    summary: &'a SupremacyMatrixSummary,
    runtime_specs: &[String],
    limit: Option<usize>,
    scope: matrix_refresh::MatrixRefreshScope,
) -> Result<Vec<&'a SupremacyMatrixRow>> {
    let mut rows = Vec::new();
    if runtime_specs.is_empty() {
        for row in summary
            .rows
            .iter()
            .filter(|row| runtime_class_in_scope(scope, row.class))
        {
            if let Some(limit) = limit {
                if rows.len() >= limit {
                    break;
                }
            }
            rows.push(row);
        }
        return Ok(rows);
    }

    for spec in runtime_specs {
        let row = summary
            .rows
            .iter()
            .find(|row| row.spec == *spec)
            .with_context(|| format!("--runtime-spec {spec}: no baseline row found"))?;
        match row.class {
            SupremacyMatrixClass::MissingRuntime
            | SupremacyMatrixClass::PerfTie
            | SupremacyMatrixClass::PerfLoser
            | SupremacyMatrixClass::ExpectedViolationMatch
            | SupremacyMatrixClass::TlcError
            | SupremacyMatrixClass::TlcTimeout
            | SupremacyMatrixClass::RuntimeToError
            | SupremacyMatrixClass::TimeoutDominance
            | SupremacyMatrixClass::TyTimeout
            | SupremacyMatrixClass::ParityFail
            | SupremacyMatrixClass::Pass => {
                if rows.iter().all(|selected| selected.spec != row.spec) {
                    rows.push(row);
                }
            }
            class => {
                bail!(
                    "--runtime-spec {} has class {:?}; unsupported rows cannot be refreshed",
                    spec,
                    class
                );
            }
        }
    }
    if let Some(limit) = limit {
        rows.truncate(limit);
    }
    Ok(rows)
}

#[derive(Clone, Debug)]
struct RuntimeBatchSelection<'a> {
    selected_rows: Vec<&'a SupremacyMatrixRow>,
    selected_specs: Vec<String>,
    skipped_batchable_runtime_specs_by_limit: Vec<String>,
}

fn runtime_refresh_scope(
    cli_scope: SupremacyMatrixRuntimeScope,
    runtime_specs: &[String],
) -> matrix_refresh::MatrixRefreshScope {
    if !runtime_specs.is_empty() {
        return matrix_refresh::MatrixRefreshScope::AllRunnable;
    }
    match cli_scope {
        SupremacyMatrixRuntimeScope::MissingRuntime => {
            matrix_refresh::MatrixRefreshScope::MissingRuntime
        }
        SupremacyMatrixRuntimeScope::AllRunnable => matrix_refresh::MatrixRefreshScope::AllRunnable,
    }
}

fn runtime_class_in_scope(
    scope: matrix_refresh::MatrixRefreshScope,
    class: SupremacyMatrixClass,
) -> bool {
    match scope {
        matrix_refresh::MatrixRefreshScope::MissingRuntime => {
            class == SupremacyMatrixClass::MissingRuntime
        }
        matrix_refresh::MatrixRefreshScope::AllRunnable => {
            class != SupremacyMatrixClass::Unsupported
        }
    }
}

fn selection_needs_trust_cg_preflight(
    selection: &RuntimeBatchSelection<'_>,
    baseline: &SpecBaseline,
) -> bool {
    selection
        .selected_rows
        .iter()
        .any(|row| baseline.specs.get(&row.spec).is_some_and(is_check_source))
}

fn runtime_batch_selection<'a>(
    summary: &'a SupremacyMatrixSummary,
    refresh_plan: &matrix_refresh::MatrixRefreshPlan,
    runtime_specs: &[String],
    limit: Option<usize>,
) -> Result<RuntimeBatchSelection<'a>> {
    if runtime_specs.is_empty() {
        let (selected_specs, skipped_batchable_runtime_specs_by_limit) =
            simulation_first_batchable_specs_limited(refresh_plan, limit);
        let selected_rows = if selected_specs.is_empty() {
            Vec::new()
        } else {
            selected_runtime_rows(summary, &selected_specs, None, refresh_plan.scope)?
        };
        return Ok(RuntimeBatchSelection {
            selected_rows,
            selected_specs,
            skipped_batchable_runtime_specs_by_limit,
        });
    }

    let mut selected_rows = selected_runtime_rows(
        summary,
        runtime_specs,
        None,
        matrix_refresh::MatrixRefreshScope::AllRunnable,
    )?;
    for row in &selected_rows {
        let Some(planned) = refresh_plan.row(&row.spec) else {
            bail!(
                "--runtime-spec {} is not present in the runtime refresh plan",
                row.spec
            );
        };
        if !runtime_plan_row_is_batchable(planned) {
            bail!(
                "--runtime-spec {} is not batchable for runtime refresh: {:?}",
                row.spec,
                planned.readiness
            );
        }
    }
    let skipped_batchable_runtime_specs_by_limit = limit
        .and_then(|limit| selected_rows.get(limit..))
        .map(|rows| rows.iter().map(|row| row.spec.clone()).collect())
        .unwrap_or_default();
    if let Some(limit) = limit {
        selected_rows.truncate(limit);
    }
    let selected_specs = selected_rows
        .iter()
        .map(|row| row.spec.clone())
        .collect::<Vec<_>>();
    Ok(RuntimeBatchSelection {
        selected_rows,
        selected_specs,
        skipped_batchable_runtime_specs_by_limit,
    })
}

fn runtime_plan_row_is_batchable(row: &matrix_refresh::MatrixRefreshRow) -> bool {
    matches!(
        row.readiness,
        matrix_refresh::MatrixRefreshReadiness::RunnableWithConfig
            | matrix_refresh::MatrixRefreshReadiness::NeedsNoConfigCliFlags { .. }
    )
}

fn simulation_first_batchable_specs_limited(
    refresh_plan: &matrix_refresh::MatrixRefreshPlan,
    limit: Option<usize>,
) -> (Vec<String>, Vec<String>) {
    let ordered_batchable_rows = limited_batchable_rows(refresh_plan, limit);
    let selected_limit = limit.unwrap_or(ordered_batchable_rows.len());
    let selected = ordered_batchable_rows
        .iter()
        .take(selected_limit)
        .map(|row| row.spec.clone())
        .collect::<Vec<_>>();
    let skipped = if limit.is_some() {
        ordered_batchable_rows
            .iter()
            .skip(selected_limit)
            .map(|row| row.spec.clone())
            .collect()
    } else {
        Vec::new()
    };
    (selected, skipped)
}

fn limited_batchable_rows<'a>(
    refresh_plan: &'a matrix_refresh::MatrixRefreshPlan,
    limit: Option<usize>,
) -> Vec<&'a matrix_refresh::MatrixRefreshRow> {
    let mut rows = refresh_plan
        .batchable_runtime_specs
        .iter()
        .filter_map(|spec| refresh_plan.row(spec))
        .collect::<Vec<_>>();
    if limit.is_some() {
        rows.sort_by_key(|row| limited_runtime_batch_key(row));
    }
    rows
}

fn limited_runtime_batch_key(row: &matrix_refresh::MatrixRefreshRow) -> LimitedAllRunnableBatchKey {
    LimitedAllRunnableBatchKey {
        source_mode_rank: source_mode_rank(row.source.mode.as_deref()),
        class_rank: runtime_class_rank(row.class),
        readiness_rank: runtime_readiness_rank(&row.readiness),
        known_runtime_key: known_runtime_key(row.tlc_seconds, row.ty_seconds),
        missing_runtime_side_rank: missing_runtime_side_rank(row.tlc_seconds, row.ty_seconds),
        path_kind_rank: path_kind_rank(row.source.path_kind),
        category: row.category.clone().unwrap_or_default(),
        tla_path: row
            .source
            .tla_path
            .as_deref()
            .map(normalized_path_text)
            .unwrap_or_default(),
        cfg_path: row
            .source
            .cfg_path
            .as_deref()
            .map(normalized_path_text)
            .unwrap_or_default(),
        expected_tlc_states: row.expected_tlc_states.unwrap_or(u64::MAX),
        expected_ty_states: row.expected_ty_states.unwrap_or(u64::MAX),
        spec: row.spec.clone(),
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct LimitedAllRunnableBatchKey {
    source_mode_rank: u8,
    class_rank: u8,
    readiness_rank: u8,
    known_runtime_key: u64,
    missing_runtime_side_rank: u8,
    path_kind_rank: u8,
    category: String,
    tla_path: String,
    cfg_path: String,
    expected_tlc_states: u64,
    expected_ty_states: u64,
    spec: String,
}

fn runtime_class_rank(class: SupremacyMatrixClass) -> u8 {
    match class {
        SupremacyMatrixClass::MissingRuntime => 0,
        SupremacyMatrixClass::PerfLoser => 1,
        SupremacyMatrixClass::PerfTie => 2,
        SupremacyMatrixClass::RuntimeToError => 3,
        SupremacyMatrixClass::TimeoutDominance => 4,
        SupremacyMatrixClass::TyTimeout => 5,
        SupremacyMatrixClass::ParityFail => 6,
        SupremacyMatrixClass::ExpectedViolationMatch => 7,
        SupremacyMatrixClass::TlcError => 8,
        SupremacyMatrixClass::TlcTimeout => 9,
        SupremacyMatrixClass::Pass => 10,
        SupremacyMatrixClass::Unsupported => 11,
    }
}

fn runtime_readiness_rank(readiness: &matrix_refresh::MatrixRefreshReadiness) -> u8 {
    match readiness {
        matrix_refresh::MatrixRefreshReadiness::RunnableWithConfig => 0,
        matrix_refresh::MatrixRefreshReadiness::NeedsNoConfigCliFlags { .. } => 1,
        matrix_refresh::MatrixRefreshReadiness::MissingSourceFiles { .. } => 2,
        matrix_refresh::MatrixRefreshReadiness::MissingSourceMetadata { .. } => 3,
    }
}

fn source_mode_rank(mode: Option<&str>) -> u8 {
    match mode {
        Some("generate") => 0,
        Some("simulate") => 1,
        None | Some("check") => 2,
        Some("no_config" | "no-config" | "config_free" | "config-free") => 3,
        Some(_) => 4,
    }
}

fn path_kind_rank(path_kind: matrix_refresh::MatrixRefreshPathKind) -> u8 {
    match path_kind {
        matrix_refresh::MatrixRefreshPathKind::ExamplesRelative => 0,
        matrix_refresh::MatrixRefreshPathKind::Absolute => 1,
        matrix_refresh::MatrixRefreshPathKind::Mixed => 2,
        matrix_refresh::MatrixRefreshPathKind::Missing => 3,
    }
}

fn normalized_path_text(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn known_runtime_key(tlc_seconds: Option<f64>, ty_seconds: Option<f64>) -> u64 {
    [tlc_seconds, ty_seconds]
        .into_iter()
        .filter_map(|seconds| {
            seconds
                .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
                .map(|seconds| (seconds * 1_000_000.0).round() as u64)
        })
        .min()
        .unwrap_or(u64::MAX)
}

fn missing_runtime_side_rank(tlc_seconds: Option<f64>, ty_seconds: Option<f64>) -> u8 {
    let missing = usize::from(!has_finite_positive_runtime(tlc_seconds))
        + usize::from(!has_finite_positive_runtime(ty_seconds));
    match missing {
        1 => 0,
        2 => 1,
        _ => 2,
    }
}

#[derive(Serialize)]
struct RuntimeBatchPlanReport<'a> {
    schema: &'static str,
    baseline: &'a Path,
    output_dir: &'a Path,
    runtime_limit: Option<usize>,
    explicit_runtime_specs: &'a [String],
    selected_runtime_specs: &'a [String],
    skipped_batchable_runtime_specs_by_limit: &'a [String],
    refresh_plan: &'a matrix_refresh::MatrixRefreshPlan,
}

fn write_runtime_batch_plan(
    baseline: &Path,
    output_dir: &Path,
    runtime_limit: Option<usize>,
    explicit_runtime_specs: &[String],
    selection: &RuntimeBatchSelection<'_>,
    refresh_plan: &matrix_refresh::MatrixRefreshPlan,
) -> Result<PathBuf> {
    let path = output_dir.join("runtime_batch_plan.json");
    let report = RuntimeBatchPlanReport {
        schema: RUNTIME_BATCH_PLAN_SCHEMA,
        baseline,
        output_dir,
        runtime_limit,
        explicit_runtime_specs,
        selected_runtime_specs: &selection.selected_specs,
        skipped_batchable_runtime_specs_by_limit: &selection
            .skipped_batchable_runtime_specs_by_limit,
        refresh_plan,
    };
    fs::write(&path, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

pub(super) fn classify_baseline_str(text: &str) -> Result<SupremacyMatrixSummary> {
    classify_baseline_str_with_policy(text, &MatrixPolicy::default())
}

pub(super) fn classify_baseline_str_with_policy(
    text: &str,
    matrix_policy: &MatrixPolicy,
) -> Result<SupremacyMatrixSummary> {
    let value: Value = serde_json::from_str(text)?;
    classify_baseline_value_with_policy(value, matrix_policy)
}

#[allow(dead_code)] // default-policy convenience wrapper used only from tests
fn classify_baseline_value(value: Value) -> Result<SupremacyMatrixSummary> {
    classify_baseline_value_with_policy(value, &MatrixPolicy::default())
}

fn classify_baseline_value_with_policy(
    value: Value,
    matrix_policy: &MatrixPolicy,
) -> Result<SupremacyMatrixSummary> {
    let corpus = corpus_identity_from_baseline_value(&value)?;
    let baseline: SpecBaseline = serde_json::from_value(value)?;
    Ok(classify_baseline(baseline, matrix_policy, corpus))
}

fn classify_baseline(
    baseline: SpecBaseline,
    matrix_policy: &MatrixPolicy,
    corpus: SupremacyMatrixCorpusIdentity,
) -> SupremacyMatrixSummary {
    let build_identity = SupremacyMatrixBuildIdentity::from_refresh(baseline.ty_refresh.as_ref());
    let examples_dir = baseline.inputs.examples_dir.clone();
    let mut counts = SupremacyMatrixCounts::default();
    let mut rows = Vec::with_capacity(baseline.specs.len());

    for (spec, entry) in baseline.specs {
        let row = classify_spec(spec, &entry, matrix_policy, examples_dir.as_deref());
        counts.add(row.class);
        rows.push(row);
    }
    assign_perf_loser_ranks(&mut rows);
    let strict_blockers = counts.strict_blocker_count();
    let strict_pass = strict_blockers == 0;
    let missing_runtime_diagnostics = SupremacyMatrixMissingRuntimeDiagnostics::from_rows(&rows);
    let next_action_counts = next_action_counts(&rows);
    let policy = matrix_policy
        .has_comparable_outcome_opt_in()
        .then(|| SupremacyMatrixPolicySummary::from_counts(&counts, matrix_policy));
    let verdict = policy
        .as_ref()
        .map(|policy| policy.verdict)
        .unwrap_or_else(|| {
            if strict_pass {
                SupremacyMatrixVerdict::Pass
            } else {
                SupremacyMatrixVerdict::Fail
            }
        });

    SupremacyMatrixSummary {
        schema: MATRIX_SUMMARY_SCHEMA,
        verdict,
        strict_pass,
        strict_blockers,
        policy,
        build_identity,
        corpus,
        missing_runtime_diagnostics,
        counts,
        next_action_counts,
        rows,
    }
}

fn next_action_counts(rows: &[SupremacyMatrixRow]) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for row in rows {
        if row.next_action == SupremacyMatrixNextAction::None {
            continue;
        }
        *counts.entry(row.next_action.as_str()).or_insert(0) += 1;
    }
    counts
}

fn classify_spec(
    spec: String,
    entry: &BaselineSpec,
    matrix_policy: &MatrixPolicy,
    examples_dir: Option<&Path>,
) -> SupremacyMatrixRow {
    let tlc_seconds = entry.tlc.runtime_seconds;
    // Two-axis TY runtime evidence (mirrors the soundness sweep's design):
    // - COUNT-VERIFY axis (`runtime_seconds`): auto-POR/auto-symmetry pinned off so
    //   `states` is unreduced-parity comparable with TLC; owns verified_match.
    // - SPEED axis (`production_runtime_seconds` when present): production-default
    //   configuration — what users actually get — owns the perf classification.
    let pinned_ty_seconds = entry.ty.runtime_seconds;
    let production_ty_seconds = entry
        .ty
        .production_runtime_seconds
        .filter(|seconds| has_finite_positive_runtime(Some(*seconds)));
    let ty_seconds = production_ty_seconds.or(pinned_ty_seconds);
    let speedup_tlc_vs_ty = speedup(tlc_seconds, ty_seconds);
    let slowdown_ty_vs_tlc = slowdown(tlc_seconds, ty_seconds);
    let seconds_lost_vs_tlc = seconds_lost(tlc_seconds, ty_seconds);
    let randomized_count_policy = randomized_count_policy(entry, examples_dir);

    let source_mode = matrix_source_mode(entry);
    let (class, reason) = if let MatrixSourceMode::Unsupported(mode) = source_mode {
        (
            SupremacyMatrixClass::Unsupported,
            format!("baseline source mode `{mode}` is not runnable by the supremacy matrix"),
        )
    } else if let Some(reason) = tlc_impossible_reason(&entry.tlc) {
        (SupremacyMatrixClass::Unsupported, reason)
    } else if let Some(reason) = production_verdict_mismatch_reason(&entry.ty) {
        // Verdict-consistency guard: the production-default run reaching a different
        // VERDICT than the pinned count-verify run is a soundness signal, never a
        // perf number. Hard-fail the row before any win/perf classification.
        (SupremacyMatrixClass::ParityFail, reason)
    } else if source_mode == MatrixSourceMode::Simulation {
        classify_simulation_spec(entry, matrix_policy, tlc_seconds, ty_seconds)
    } else if is_timeout(&entry.tlc) {
        if let Some(reason) = timeout_dominance_reason(matrix_policy, entry) {
            (SupremacyMatrixClass::TimeoutDominance, reason)
        } else {
            (
                SupremacyMatrixClass::TlcTimeout,
                "TLC baseline timed out".to_string(),
            )
        }
    } else if let Some(expected_bmc_error) = classify_bmc_only_matching_error(entry) {
        expected_bmc_error
    } else if is_tlc_error(&entry.tlc) {
        if let Some(expected_violation) =
            classify_expected_violation_match(entry, matrix_policy, tlc_seconds, ty_seconds)
        {
            expected_violation
        } else if let Some(reason) = runtime_to_error_reason(matrix_policy, entry) {
            (SupremacyMatrixClass::RuntimeToError, reason)
        } else {
            (
                SupremacyMatrixClass::TlcError,
                "TLC baseline records a model/checker error".to_string(),
            )
        }
    } else if let Some(reason) = undersized_ty_timeout_reason(entry) {
        (SupremacyMatrixClass::MissingRuntime, reason)
    } else if is_timeout(&entry.ty) {
        (
            SupremacyMatrixClass::TyTimeout,
            "TY baseline timed out".to_string(),
        )
    } else if entry.ty.status != "pass" || entry.ty.error_type.is_some() {
        (
            SupremacyMatrixClass::ParityFail,
            "TY baseline did not verify against TLC".to_string(),
        )
    } else if let Some(reason) =
        successful_check_parity_failure_reason(entry, randomized_count_policy.as_ref())
    {
        (SupremacyMatrixClass::ParityFail, reason)
    } else if !has_finite_positive_runtime(tlc_seconds)
        || !has_finite_positive_runtime(pinned_ty_seconds)
    {
        // Missing-runtime gates on the pinned count-verify run: production-only
        // evidence can never substitute for the parity axis.
        (
            SupremacyMatrixClass::MissingRuntime,
            reason_with_randomized_count_policy(
                missing_runtime_reason(tlc_seconds, pinned_ty_seconds),
                randomized_count_policy.as_ref(),
            ),
        )
    } else if let Some(tie_reason) = perf_tie_reason(tlc_seconds, ty_seconds) {
        (
            SupremacyMatrixClass::PerfTie,
            reason_with_randomized_count_policy(tie_reason, randomized_count_policy.as_ref()),
        )
    } else if is_perf_loser(tlc_seconds, ty_seconds) {
        (
            SupremacyMatrixClass::PerfLoser,
            reason_with_randomized_count_policy(
                "TY runtime is not faster than TLC runtime".to_string(),
                randomized_count_policy.as_ref(),
            ),
        )
    } else {
        (
            SupremacyMatrixClass::Pass,
            reason_with_randomized_count_policy(
                "baseline is runnable, parity-verified, and TY is faster than TLC".to_string(),
                randomized_count_policy.as_ref(),
            ),
        )
    };
    let perf_loser_follow_up = perf_loser_follow_up(class, &spec).map(str::to_string);
    let next_action = SupremacyMatrixNextAction::from_class(class);
    let (missing_tlc_runtime, missing_ty_runtime) =
        missing_runtime_modes_for_row(class, entry, tlc_seconds, pinned_ty_seconds);

    SupremacyMatrixRow {
        spec,
        class,
        next_action,
        reason,
        missing_tlc_runtime,
        missing_ty_runtime,
        perf_loser_rank: None,
        tlc_seconds,
        ty_seconds,
        ty_pinned_seconds: if production_ty_seconds.is_some() {
            pinned_ty_seconds
        } else {
            None
        },
        speedup_tlc_vs_ty,
        slowdown_ty_vs_tlc,
        seconds_lost_vs_tlc,
        perf_loser_follow_up,
    }
}

fn classify_simulation_spec(
    entry: &BaselineSpec,
    matrix_policy: &MatrixPolicy,
    tlc_seconds: Option<f64>,
    ty_seconds: Option<f64>,
) -> (SupremacyMatrixClass, String) {
    let has_ty_runtime = has_finite_positive_runtime(ty_seconds);
    if let Some(reason) = undersized_ty_timeout_reason(entry) {
        return (SupremacyMatrixClass::MissingRuntime, reason);
    }
    if is_timeout(&entry.ty) {
        return (
            SupremacyMatrixClass::TyTimeout,
            "TY simulation baseline timed out".to_string(),
        );
    }
    if entry.ty.status != "pass" {
        return (
            SupremacyMatrixClass::ParityFail,
            "TY simulation baseline did not verify as a runnable simulation".to_string(),
        );
    }
    if is_timeout(&entry.tlc) {
        if !has_ty_runtime {
            return (
                SupremacyMatrixClass::MissingRuntime,
                missing_runtime_reason(tlc_seconds, ty_seconds),
            );
        }
        if let Some(reason) = timeout_dominance_reason(matrix_policy, entry) {
            return (SupremacyMatrixClass::TimeoutDominance, reason);
        }
        return (
            SupremacyMatrixClass::TlcTimeout,
            "TLC simulation baseline timed out".to_string(),
        );
    }
    if is_tlc_error(&entry.tlc) {
        if let Some(expected_violation) =
            classify_expected_violation_match(entry, matrix_policy, tlc_seconds, ty_seconds)
        {
            return expected_violation;
        }
        if !has_ty_runtime || !has_finite_positive_runtime(tlc_seconds) {
            return (
                SupremacyMatrixClass::MissingRuntime,
                missing_runtime_reason(tlc_seconds, ty_seconds),
            );
        }
        if let Some(reason) = runtime_to_error_reason(matrix_policy, entry) {
            return (SupremacyMatrixClass::RuntimeToError, reason);
        }
        return (
            SupremacyMatrixClass::TlcError,
            "TLC simulation baseline records a model/checker error".to_string(),
        );
    }
    if entry.ty.error_type.is_some() {
        return (
            SupremacyMatrixClass::ParityFail,
            "TY simulation baseline reported an unexpected checker error".to_string(),
        );
    }
    if !entry.verified_match {
        return (
            SupremacyMatrixClass::ParityFail,
            "TY simulation baseline did not verify as a runnable simulation".to_string(),
        );
    }
    if !has_finite_positive_runtime(tlc_seconds) || !has_ty_runtime {
        return (
            SupremacyMatrixClass::MissingRuntime,
            missing_runtime_reason(tlc_seconds, ty_seconds),
        );
    }
    if let Some(tie_reason) = perf_tie_reason(tlc_seconds, ty_seconds) {
        return (SupremacyMatrixClass::PerfTie, tie_reason);
    }
    if is_perf_loser(tlc_seconds, ty_seconds) {
        return (
            SupremacyMatrixClass::PerfLoser,
            "TY simulation runtime is not faster than TLC simulation runtime".to_string(),
        );
    }
    (
        SupremacyMatrixClass::Pass,
        "simulation baseline is runnable, verified, and TY is faster than TLC".to_string(),
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RandomizedCountPolicy {
    tlc_states: u64,
    ty_states: u64,
    operators: Vec<String>,
    evidence_sources: Vec<String>,
}

impl RandomizedCountPolicy {
    fn reason_fragment(&self) -> String {
        format!(
            "both tools passed but state counts differ because randomized external operator(s) {} were detected in {}; exact state-count parity is not enforced (TLC states={}, TY states={})",
            self.operators.join(", "),
            self.evidence_sources.join(", "),
            self.tlc_states,
            self.ty_states
        )
    }
}

fn randomized_count_policy(
    entry: &BaselineSpec,
    examples_dir: Option<&Path>,
) -> Option<RandomizedCountPolicy> {
    let (tlc_states, ty_states) = state_count_mismatch(entry)?;
    if !baseline_modes_are_clean_pass(entry) {
        return None;
    }
    let (operators, evidence_sources) = randomized_external_operator_evidence(entry, examples_dir)?;
    Some(RandomizedCountPolicy {
        tlc_states,
        ty_states,
        operators,
        evidence_sources,
    })
}

fn successful_check_parity_failure_reason(
    entry: &BaselineSpec,
    randomized_count_policy: Option<&RandomizedCountPolicy>,
) -> Option<String> {
    if let Some((tlc_states, ty_states)) = state_count_mismatch(entry) {
        if randomized_count_policy.is_some() {
            return None;
        }
        return Some(format!(
            "TLC and TY both passed but exact state counts differ (TLC states={tlc_states}, TY states={ty_states}); no randomized external operator evidence was found, so state-count parity is required"
        ));
    }
    (!entry.verified_match).then(|| "TY baseline did not verify against TLC".to_string())
}

fn reason_with_randomized_count_policy(
    base_reason: String,
    randomized_count_policy: Option<&RandomizedCountPolicy>,
) -> String {
    let Some(policy) = randomized_count_policy else {
        return base_reason;
    };
    format!(
        "{RANDOMIZED_COUNT_POLICY_REASON_PREFIX}: {}; {base_reason}",
        policy.reason_fragment()
    )
}

fn baseline_modes_are_clean_pass(entry: &BaselineSpec) -> bool {
    mode_is_clean_pass(&entry.tlc) && mode_is_clean_pass(&entry.ty)
}

fn mode_is_clean_pass(mode: &BaselineMode) -> bool {
    mode.status.eq_ignore_ascii_case("pass") && mode.error_type.is_none()
}

fn state_count_mismatch(entry: &BaselineSpec) -> Option<(u64, u64)> {
    match (entry.tlc.states, entry.ty.states) {
        (Some(tlc_states), Some(ty_states)) if tlc_states != ty_states => {
            Some((tlc_states, ty_states))
        }
        _ => None,
    }
}

fn randomized_external_operator_evidence(
    entry: &BaselineSpec,
    examples_dir: Option<&Path>,
) -> Option<(Vec<String>, Vec<String>)> {
    let mut operators = BTreeSet::new();
    let mut evidence_sources = BTreeSet::new();

    collect_randomized_operator_evidence_from_metadata(
        "baseline spec metadata",
        &entry.metadata,
        &mut operators,
        &mut evidence_sources,
    );
    collect_randomized_operator_evidence_from_metadata(
        "TLC baseline metadata",
        &entry.tlc.metadata,
        &mut operators,
        &mut evidence_sources,
    );
    collect_randomized_operator_evidence_from_metadata(
        "TY baseline metadata",
        &entry.ty.metadata,
        &mut operators,
        &mut evidence_sources,
    );

    if let Some(source) = &entry.source {
        if let Some(path) = &source.tla_path {
            let resolved = resolve_baseline_source_path(examples_dir, path);
            collect_randomized_operator_evidence_from_source_path(
                &resolved,
                &format!("source {}", resolved.display()),
                &mut operators,
                &mut evidence_sources,
            );
        }
        if let Some(path) = &source.cfg_path {
            let resolved = resolve_baseline_source_path(examples_dir, path);
            collect_randomized_operator_evidence_from_source_path(
                &resolved,
                &format!("source {}", resolved.display()),
                &mut operators,
                &mut evidence_sources,
            );
        }
    }

    (!operators.is_empty()).then(|| {
        (
            operators.into_iter().collect(),
            evidence_sources.into_iter().collect(),
        )
    })
}

fn resolve_baseline_source_path(examples_dir: Option<&Path>, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    examples_dir
        .map(|base| base.join(path))
        .unwrap_or_else(|| path.to_path_buf())
}

fn collect_randomized_operator_evidence_from_metadata(
    label: &str,
    metadata: &BTreeMap<String, Value>,
    operators: &mut BTreeSet<String>,
    evidence_sources: &mut BTreeSet<String>,
) {
    for (key, value) in metadata {
        if !metadata_key_may_report_randomized_operator(key) {
            continue;
        }
        for operator in RANDOMIZED_EXTERNAL_OPERATORS {
            if json_value_contains_operator(value, operator) {
                operators.insert((*operator).to_string());
                evidence_sources.insert(format!("{label}.{key}"));
            }
        }
    }
}

fn metadata_key_may_report_randomized_operator(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "randomized_external_operator"
            | "randomized_external_operators"
            | "random_external_operator"
            | "random_external_operators"
            | "runtime_randomized_operator"
            | "runtime_randomized_operators"
    )
}

fn json_value_contains_operator(value: &Value, operator: &str) -> bool {
    match value {
        Value::String(text) => contains_tla_identifier(text, operator),
        Value::Array(items) => items
            .iter()
            .any(|item| json_value_contains_operator(item, operator)),
        Value::Object(map) => map
            .values()
            .any(|item| json_value_contains_operator(item, operator)),
        _ => false,
    }
}

fn collect_randomized_operator_evidence_from_source_path(
    path: &Path,
    label: &str,
    operators: &mut BTreeSet<String>,
    evidence_sources: &mut BTreeSet<String>,
) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    let searchable = strip_tla_comments(&text);
    for operator in RANDOMIZED_EXTERNAL_OPERATORS {
        if contains_tla_identifier(&searchable, operator) {
            operators.insert((*operator).to_string());
            evidence_sources.insert(label.to_string());
        }
    }
}

fn strip_tla_comments(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_line_comment = false;
    let mut block_comment_depth = 0usize;

    while let Some(ch) = chars.next() {
        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
                stripped.push('\n');
            }
            continue;
        }

        if block_comment_depth > 0 {
            if ch == '(' && chars.peek() == Some(&'*') {
                chars.next();
                block_comment_depth += 1;
                continue;
            }
            if ch == '*' && chars.peek() == Some(&')') {
                chars.next();
                block_comment_depth -= 1;
                continue;
            }
            if ch == '\n' {
                stripped.push('\n');
            } else {
                stripped.push(' ');
            }
            continue;
        }

        if ch == '\\' && chars.peek() == Some(&'*') {
            chars.next();
            in_line_comment = true;
            continue;
        }
        if ch == '(' && chars.peek() == Some(&'*') {
            chars.next();
            block_comment_depth = 1;
            continue;
        }
        stripped.push(ch);
    }

    stripped
}

fn contains_tla_identifier(text: &str, identifier: &str) -> bool {
    let mut search_start = 0usize;
    while let Some(offset) = text[search_start..].find(identifier) {
        let start = search_start + offset;
        let end = start + identifier.len();
        let previous = text[..start].chars().next_back();
        let next = text[end..].chars().next();
        if !previous.is_some_and(is_tla_identifier_char)
            && !next.is_some_and(is_tla_identifier_char)
        {
            return true;
        }
        search_start = end;
    }
    false
}

fn is_tla_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn assign_perf_loser_ranks(rows: &mut [SupremacyMatrixRow]) {
    let mut ranked: Vec<(usize, f64, f64)> = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            if row.class != SupremacyMatrixClass::PerfLoser {
                return None;
            }
            Some((index, row.slowdown_ty_vs_tlc?, row.seconds_lost_vs_tlc?))
        })
        .collect();

    ranked.sort_by(
        |(_, left_slowdown, left_loss), (_, right_slowdown, right_loss)| {
            right_slowdown
                .total_cmp(left_slowdown)
                .then_with(|| right_loss.total_cmp(left_loss))
        },
    );

    for (rank, (index, _, _)) in ranked.into_iter().enumerate() {
        rows[index].perf_loser_rank = Some(rank + 1);
    }
}

#[derive(Clone, Debug)]
struct RuntimeCollectionConfig {
    output_dir: PathBuf,
    timeout_seconds: u64,
    limit: Option<usize>,
    ty_bin: PathBuf,
    ty_base_env: BTreeMap<String, String>,
    allow_debug_runtime: bool,
    /// Collect a second TY measurement per check-mode row under the
    /// production-default configuration (count-parity pins removed).
    production_runtime: bool,
    tlc_jar: PathBuf,
    community_modules: Option<PathBuf>,
    tla_library: Option<PathBuf>,
}

impl RuntimeCollectionConfig {
    fn from_args(args: &SupremacyMatrixArgs, repo_root: &Path) -> Result<Self> {
        let output_dir = args
            .runtime_output_dir
            .clone()
            .unwrap_or_else(default_runtime_output_dir);
        let timeout_seconds = args.runtime_timeout;
        if timeout_seconds == 0 {
            bail!("--runtime-timeout must be >= 1");
        }
        let limit = args.runtime_limit;
        let ty_bin = args
            .runtime_ty_bin
            .clone()
            .unwrap_or_else(|| env::current_exe().unwrap_or_else(|_| PathBuf::from("ty")));
        let ty_base_env = matrix_runtime_refresh_base_env();
        let allow_debug_runtime = args.allow_debug_runtime;
        let production_runtime = args.production_runtime;
        let tlc_jar = args.runtime_tlc_jar.clone().unwrap_or_else(default_tlc_jar);
        let community_modules = args
            .runtime_community_modules
            .clone()
            .or_else(default_community_modules_jar);
        let tla_library = resolve_runtime_tla_library(args, repo_root);
        if let Some(community_modules) = &community_modules {
            validate_file(community_modules).with_context(|| {
                format!(
                    "validate --runtime-community-modules {}",
                    community_modules.display()
                )
            })?;
        }
        Ok(Self {
            output_dir,
            timeout_seconds,
            limit,
            ty_bin,
            ty_base_env,
            allow_debug_runtime,
            production_runtime,
            tlc_jar,
            community_modules,
            tla_library,
        })
    }
}

struct RuntimeRefreshCheckpoint {
    summary: SupremacyMatrixSummary,
    metadata_refresh: BaselineMetadataRefresh,
    report_path: PathBuf,
    refreshed_baseline: PathBuf,
    matrix_after_refresh: PathBuf,
}

fn write_runtime_refresh_checkpoint(
    baseline_path: &Path,
    config: &RuntimeCollectionConfig,
    baseline_value: &Value,
    matrix_policy: &MatrixPolicy,
    runtime_batch_plan: &Path,
    selection: &RuntimeBatchSelection<'_>,
    refresh_plan: &matrix_refresh::MatrixRefreshPlan,
    metadata: &RuntimeEvidenceMetadata,
    rows: &[RuntimeEvidenceRow],
    baseline_provenance: &RuntimeBaselineProvenance,
) -> Result<RuntimeRefreshCheckpoint> {
    let mut checkpoint_baseline = baseline_value.clone();
    let metadata_refresh =
        refresh_runtime_baseline_metadata(&mut checkpoint_baseline, rows, baseline_provenance)
            .context("refresh runtime baseline metadata")?;

    let refreshed_baseline = config.output_dir.join("spec_baseline.refreshed.json");
    fs::write(
        &refreshed_baseline,
        serde_json::to_string_pretty(&checkpoint_baseline)? + "\n",
    )
    .with_context(|| format!("write {}", refreshed_baseline.display()))?;

    let summary = classify_baseline_value_with_policy(checkpoint_baseline, matrix_policy)
        .context("classify refreshed runtime baseline")?;
    let matrix_after_refresh = config.output_dir.join("matrix_after_refresh.json");
    fs::write(
        &matrix_after_refresh,
        serde_json::to_string_pretty(&summary)? + "\n",
    )
    .with_context(|| format!("write {}", matrix_after_refresh.display()))?;

    let collected_runtime_specs = rows.iter().map(|row| row.spec.clone()).collect::<Vec<_>>();
    let uncollected_selected_runtime_specs =
        uncollected_selected_runtime_specs(&selection.selected_specs, rows);
    let attempted_all_selected_runtime_specs = uncollected_selected_runtime_specs.is_empty();
    let incomplete_runtime_specs =
        incomplete_selected_runtime_specs(&selection.selected_specs, rows);
    let report_path = config.output_dir.join("runtime_evidence.json");
    let report = RuntimeEvidenceReport {
        schema: RUNTIME_EVIDENCE_SCHEMA,
        baseline: baseline_path.to_path_buf(),
        output_dir: config.output_dir.clone(),
        refreshed_baseline: refreshed_baseline.clone(),
        matrix_after_refresh: matrix_after_refresh.clone(),
        timeout_seconds: config.timeout_seconds,
        limit: config.limit,
        runtime_batch_plan: runtime_batch_plan.to_path_buf(),
        selected_runtime_specs: selection.selected_specs.clone(),
        selected_runtime_spec_count: selection.selected_specs.len(),
        collected_runtime_specs,
        collected_runtime_spec_count: rows.len(),
        attempted_all_selected_runtime_specs,
        complete: attempted_all_selected_runtime_specs && incomplete_runtime_specs.is_empty(),
        uncollected_selected_runtime_specs,
        incomplete_runtime_specs,
        blocked_runtime_specs: refresh_plan.blocked_runtime_specs.clone(),
        allow_debug_runtime: config.allow_debug_runtime,
        metadata: metadata.clone(),
        rows: rows.to_vec(),
        errors: runtime_evidence_errors(rows),
    };
    fs::write(&report_path, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("write {}", report_path.display()))?;

    Ok(RuntimeRefreshCheckpoint {
        summary,
        metadata_refresh,
        report_path,
        refreshed_baseline,
        matrix_after_refresh,
    })
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct RuntimeEvidenceReport {
    schema: &'static str,
    baseline: PathBuf,
    output_dir: PathBuf,
    refreshed_baseline: PathBuf,
    matrix_after_refresh: PathBuf,
    timeout_seconds: u64,
    limit: Option<usize>,
    runtime_batch_plan: PathBuf,
    selected_runtime_specs: Vec<String>,
    selected_runtime_spec_count: usize,
    collected_runtime_specs: Vec<String>,
    collected_runtime_spec_count: usize,
    attempted_all_selected_runtime_specs: bool,
    complete: bool,
    uncollected_selected_runtime_specs: Vec<String>,
    incomplete_runtime_specs: Vec<String>,
    blocked_runtime_specs: Vec<String>,
    allow_debug_runtime: bool,
    metadata: RuntimeEvidenceMetadata,
    rows: Vec<RuntimeEvidenceRow>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    errors: Vec<RuntimeEvidenceError>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeEvidenceError {
    spec: String,
    error: String,
}

fn uncollected_selected_runtime_specs(
    selected_specs: &[String],
    rows: &[RuntimeEvidenceRow],
) -> Vec<String> {
    let collected = rows
        .iter()
        .map(|row| row.spec.as_str())
        .collect::<BTreeSet<_>>();
    selected_specs
        .iter()
        .filter(|spec| !collected.contains(spec.as_str()))
        .cloned()
        .collect()
}

fn incomplete_selected_runtime_specs(
    selected_specs: &[String],
    rows: &[RuntimeEvidenceRow],
) -> Vec<String> {
    selected_specs
        .iter()
        .filter(|spec| match rows.iter().find(|row| row.spec == **spec) {
            Some(row) => !runtime_row_has_complete_fresh_evidence(row),
            None => true,
        })
        .cloned()
        .collect()
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeEvidenceMetadata {
    generated_at: String,
    ty: RuntimeTyProvenance,
    tlc: RuntimeTlcProvenance,
    java: RuntimeCommandVersionProvenance,
    examples_checkout: RuntimeGitCheckoutProvenance,
}

impl RuntimeEvidenceMetadata {
    fn collect(
        repo_root: &Path,
        examples_dir: &Path,
        ty_bin: &Path,
        tlc_jar: &Path,
        generated_at: String,
    ) -> Result<Self> {
        Ok(Self {
            generated_at,
            ty: RuntimeTyProvenance {
                git_commit: current_ty_git_commit(repo_root),
                workspace_git_commit: git_command_text(repo_root, &["rev-parse", "HEAD"]).ok(),
                binary: RuntimeFileProvenance::new(ty_bin)
                    .with_context(|| format!("hash TY binary {}", ty_bin.display()))?,
            },
            tlc: RuntimeTlcProvenance {
                jar: RuntimeFileProvenance::new(tlc_jar)
                    .with_context(|| format!("hash TLC jar {}", tlc_jar.display()))?,
            },
            java: java_version_provenance(),
            examples_checkout: git_checkout_provenance(examples_dir),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeTyProvenance {
    git_commit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_git_commit: Option<String>,
    binary: RuntimeFileProvenance,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeTlcProvenance {
    jar: RuntimeFileProvenance,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeFileProvenance {
    path: PathBuf,
    sha256: String,
}

impl RuntimeFileProvenance {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            path: canonicalize_lossy(path),
            sha256: sha256_file(path)?,
        })
    }
}

#[derive(Clone, Debug)]
struct RuntimeBaselineProvenance {
    timestamp: String,
    ty_git_commit: String,
    ty_binary: RuntimeFileProvenance,
    allow_debug_runtime: bool,
}

impl RuntimeBaselineProvenance {
    fn from_metadata(metadata: &RuntimeEvidenceMetadata, allow_debug_runtime: bool) -> Self {
        Self {
            timestamp: metadata.generated_at.clone(),
            ty_git_commit: metadata.ty.git_commit.clone(),
            ty_binary: metadata.ty.binary.clone(),
            allow_debug_runtime,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeCommandVersionProvenance {
    argv: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    output: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeGitCheckoutProvenance {
    path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    worktree_root: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    head: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    head_short: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_dirty: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status_porcelain_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeEvidenceRow {
    spec: String,
    tlc: RuntimeModeEvidence,
    ty: RuntimeModeEvidence,
    verified_match: bool,
    refreshed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    required_flags: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct RuntimeModeEvidence {
    status: String,
    runtime_seconds: Option<f64>,
    states: Option<u64>,
    error_type: Option<String>,
    artifact_dir: PathBuf,
    /// Production-default measurement axis (auto-POR/auto-symmetry free to
    /// engage), collected only for the TY side of check-mode rows when
    /// `--production-runtime true`. `production_status` marks presence.
    #[serde(skip_serializing_if = "Option::is_none")]
    production_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    production_error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    production_runtime_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    production_states: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    production_artifact_dir: Option<PathBuf>,
}

/// Fold a production-default run's result into the pinned TY evidence's
/// production axis fields.
fn attach_production_runtime_evidence(
    mut ty: RuntimeModeEvidence,
    production: RuntimeModeEvidence,
) -> RuntimeModeEvidence {
    ty.production_status = Some(production.status);
    ty.production_error_type = production.error_type;
    ty.production_runtime_seconds = production.runtime_seconds;
    ty.production_states = production.states;
    ty.production_artifact_dir = Some(production.artifact_dir);
    ty
}

const RUNTIME_COLLECTION_FAILED_ERROR_TYPE: &str = "runtime_collection_failed";

fn runtime_collection_error_row(
    spec_name: &str,
    output_dir: &Path,
    error: &anyhow::Error,
) -> RuntimeEvidenceRow {
    RuntimeEvidenceRow {
        spec: spec_name.to_string(),
        tlc: runtime_collection_error_mode(output_dir, spec_name, "tlc"),
        ty: runtime_collection_error_mode(output_dir, spec_name, "ty"),
        verified_match: false,
        refreshed: false,
        note: Some(format!("runtime evidence collection failed: {error:#}")),
        required_flags: Vec::new(),
    }
}

fn runtime_collection_error_mode(
    output_dir: &Path,
    spec_name: &str,
    mode: &str,
) -> RuntimeModeEvidence {
    RuntimeModeEvidence {
        status: "fail".to_string(),
        runtime_seconds: None,
        states: None,
        error_type: Some(RUNTIME_COLLECTION_FAILED_ERROR_TYPE.to_string()),
        artifact_dir: output_dir
            .join(spec_name)
            .join("collection-failed")
            .join(mode),
        ..RuntimeModeEvidence::default()
    }
}

fn runtime_evidence_errors(rows: &[RuntimeEvidenceRow]) -> Vec<RuntimeEvidenceError> {
    rows.iter()
        .filter(|row| runtime_row_is_collection_error(row))
        .map(|row| RuntimeEvidenceError {
            spec: row.spec.clone(),
            error: row.note.clone().unwrap_or_else(|| {
                "runtime evidence collection failed without a recorded note".to_string()
            }),
        })
        .collect()
}

fn runtime_row_is_collection_error(row: &RuntimeEvidenceRow) -> bool {
    row.tlc.error_type.as_deref() == Some(RUNTIME_COLLECTION_FAILED_ERROR_TYPE)
        || row.ty.error_type.as_deref() == Some(RUNTIME_COLLECTION_FAILED_ERROR_TYPE)
}

fn collect_spec_runtime(
    spec_name: &str,
    entry: &BaselineSpec,
    config: &RuntimeCollectionConfig,
    repo_root: &Path,
    examples_dir: &Path,
    tlc_classpath: &str,
    planned_row: Option<&matrix_refresh::MatrixRefreshRow>,
) -> Result<RuntimeEvidenceRow> {
    let source = entry
        .source
        .as_ref()
        .with_context(|| format!("{spec_name}: baseline source is missing"))?;
    // `--runtime-timeout` is an authoritative hard cap. A single known-slow spec
    // (e.g. diagnose_timeout_seconds=50000 ≈ 13.9h) must never balloon a broad sweep
    // into hours; to grant a spec more time, raise --runtime-timeout explicitly. The
    // diagnose budget is surfaced as an annotation (undersized_ty_timeout_reason), not
    // used to escalate the per-spec run timeout.
    let timeout_seconds = config.timeout_seconds;
    let tla_path = planned_row
        .and_then(|row| row.source.resolved_tla_path.clone())
        .or_else(|| {
            source
                .tla_path
                .as_ref()
                .map(|path| absolutize(examples_dir, path))
        })
        .with_context(|| format!("{spec_name}: baseline source.tla_path is missing"))?;
    validate_file(&tla_path)?;

    if let Some(no_config_flags) = no_config_runtime_flags(planned_row) {
        return collect_no_config_spec_runtime(
            spec_name,
            &tla_path,
            config,
            repo_root,
            tlc_classpath,
            no_config_flags,
            timeout_seconds,
        );
    }

    let cfg_path = planned_row
        .and_then(|row| row.source.resolved_cfg_path.clone())
        .or_else(|| {
            source
                .cfg_path
                .as_ref()
                .map(|path| absolutize(examples_dir, path))
        })
        .with_context(|| format!("{spec_name}: baseline source.cfg_path is missing"))?;
    validate_file(&cfg_path)?;

    if is_simulation_source(entry) {
        return collect_simulation_spec_runtime(
            spec_name,
            &tla_path,
            &cfg_path,
            config,
            repo_root,
            tlc_classpath,
            timeout_seconds,
        );
    }

    eprintln!("[supremacy] {spec_name}: collecting matrix runtime evidence");
    let spec_dir = config.output_dir.join(spec_name);
    let tlc = run_tlc_runtime(
        spec_name,
        &tla_path,
        &cfg_path,
        tlc_classpath,
        config.tla_library.as_deref(),
        repo_root,
        &spec_dir,
        timeout_seconds,
    )?;
    let ty = run_ty_runtime(
        spec_name,
        &tla_path,
        &cfg_path,
        &config.ty_bin,
        repo_root,
        &spec_dir,
        timeout_seconds,
        &config.ty_base_env,
        config.allow_debug_runtime,
    )?;
    let ty = if config.production_runtime {
        eprintln!("[supremacy] {spec_name}: collecting production-default runtime evidence");
        let production = run_ty_production_runtime(
            spec_name,
            &tla_path,
            &cfg_path,
            &config.ty_bin,
            repo_root,
            &spec_dir,
            timeout_seconds,
            &config.ty_base_env,
            config.allow_debug_runtime,
        )?;
        attach_production_runtime_evidence(ty, production)
    } else {
        ty
    };
    let verified_match = runtime_modes_verified_match(&tlc, &ty);
    let refreshed = runtime_modes_have_fresh_evidence(&tlc, &ty);
    Ok(RuntimeEvidenceRow {
        spec: spec_name.to_string(),
        tlc,
        ty,
        verified_match,
        refreshed,
        note: None,
        required_flags: Vec::new(),
    })
}

fn collect_simulation_spec_runtime(
    spec_name: &str,
    tla_path: &Path,
    cfg_path: &Path,
    config: &RuntimeCollectionConfig,
    repo_root: &Path,
    tlc_classpath: &str,
    timeout_seconds: u64,
) -> Result<RuntimeEvidenceRow> {
    eprintln!("[supremacy] {spec_name}: collecting simulation runtime evidence");
    let spec_dir = config.output_dir.join(spec_name);
    let simulation_cfg_path = write_simulation_runtime_config(cfg_path, repo_root, &spec_dir)?;
    let tlc = run_tlc_simulation_runtime(
        spec_name,
        tla_path,
        &simulation_cfg_path,
        tlc_classpath,
        config.tla_library.as_deref(),
        repo_root,
        &spec_dir,
        timeout_seconds,
    )?;
    let ty = run_ty_simulation_runtime(
        spec_name,
        tla_path,
        &simulation_cfg_path,
        &config.ty_bin,
        repo_root,
        &spec_dir,
        timeout_seconds,
        config.allow_debug_runtime,
    )?;
    let verified_match = ty.status == SUPREMACY_STATUS_PASS;
    let has_ty_runtime = has_finite_positive_runtime(ty.runtime_seconds);
    let has_required_tlc_evidence = has_finite_positive_runtime(tlc.runtime_seconds);
    let refreshed = has_ty_runtime && has_required_tlc_evidence;
    Ok(RuntimeEvidenceRow {
        spec: spec_name.to_string(),
        tlc,
        ty,
        verified_match,
        refreshed,
        note: Some(
            "Simulation-mode row: TY simulation completion is required; TLC status is recorded separately and state-count parity is intentionally ignored"
                .to_string(),
        ),
        required_flags: vec!["simulate".to_string()],
    })
}

#[allow(dead_code)] // predicate exercised only from tests in non-test builds
fn should_run_no_config_runtime(
    _spec_name: &str,
    planned_row: Option<&matrix_refresh::MatrixRefreshRow>,
) -> bool {
    no_config_runtime_flags(planned_row).is_some()
}

fn no_config_runtime_flags(
    planned_row: Option<&matrix_refresh::MatrixRefreshRow>,
) -> Option<&[String]> {
    planned_row.and_then(|row| match &row.readiness {
        matrix_refresh::MatrixRefreshReadiness::NeedsNoConfigCliFlags { flags, .. } => {
            Some(flags.as_slice())
        }
        _ => None,
    })
}

fn collect_no_config_spec_runtime(
    spec_name: &str,
    tla_path: &Path,
    config: &RuntimeCollectionConfig,
    repo_root: &Path,
    tlc_classpath: &str,
    no_config_flags: &[String],
    timeout_seconds: u64,
) -> Result<RuntimeEvidenceRow> {
    eprintln!("[supremacy] {spec_name}: collecting config-free matrix runtime evidence");
    let spec_dir = config.output_dir.join(spec_name);
    let generated_cfg_path = write_no_config_tlc_config(repo_root, &spec_dir, no_config_flags)?;
    let tlc = run_tlc_runtime(
        spec_name,
        tla_path,
        &generated_cfg_path,
        tlc_classpath,
        config.tla_library.as_deref(),
        repo_root,
        &spec_dir,
        timeout_seconds,
    )?;
    let ty = run_ty_no_config_runtime(
        spec_name,
        tla_path,
        &config.ty_bin,
        repo_root,
        &spec_dir,
        timeout_seconds,
        &config.ty_base_env,
        config.allow_debug_runtime,
        no_config_flags,
    )?;
    let ty = if config.production_runtime {
        eprintln!("[supremacy] {spec_name}: collecting production-default runtime evidence");
        let production = run_ty_no_config_production_runtime(
            spec_name,
            tla_path,
            &config.ty_bin,
            repo_root,
            &spec_dir,
            timeout_seconds,
            &config.ty_base_env,
            config.allow_debug_runtime,
            no_config_flags,
        )?;
        attach_production_runtime_evidence(ty, production)
    } else {
        ty
    };
    let verified_match = runtime_modes_verified_match(&tlc, &ty);
    let refreshed = runtime_modes_have_fresh_evidence(&tlc, &ty);
    Ok(RuntimeEvidenceRow {
        spec: spec_name.to_string(),
        tlc,
        ty,
        verified_match,
        refreshed,
        note: Some("TLC used a generated config; TY used config-free CLI flags".to_string()),
        required_flags: no_config_runtime_required_flags(no_config_flags),
    })
}

fn no_config_runtime_required_flags(no_config_flags: &[String]) -> Vec<String> {
    no_config_flags.to_vec()
}

fn runtime_modes_verified_match(tlc: &RuntimeModeEvidence, ty: &RuntimeModeEvidence) -> bool {
    if tlc.status == SUPREMACY_STATUS_PASS
        && ty.status == SUPREMACY_STATUS_PASS
        && tlc.error_type.is_none()
        && ty.error_type.is_none()
    {
        return match (tlc.states, ty.states) {
            (Some(tlc_states), Some(ty_states)) if tlc_states == ty_states => true,
            (Some(tlc_states), Some(ty_states)) if tlc_states != ty_states => {
                runtime_artifacts_have_randomized_count_policy(tlc, ty)
            }
            _ => false,
        };
    }

    let Some(tlc_identity) = expected_violation_identity(tlc) else {
        return false;
    };
    let Some(ty_identity) = expected_violation_identity(ty) else {
        return false;
    };
    tlc_identity.matches(&ty_identity, tlc.states, ty.states)
}

fn runtime_states_are_compatible(tlc_states: Option<u64>, ty_states: Option<u64>) -> bool {
    matches!((tlc_states, ty_states), (Some(tlc_states), Some(ty_states)) if tlc_states == ty_states)
}

fn expected_violation_identity(
    evidence: &RuntimeModeEvidence,
) -> Option<ExpectedViolationIdentity> {
    let kind = expected_violation_kind(evidence.error_type.as_deref())?;
    let name = runtime_artifact_text(evidence)
        .as_deref()
        .and_then(|text| expected_violation_name(kind, text));
    Some(ExpectedViolationIdentity { kind, name })
}

fn runtime_artifact_text(evidence: &RuntimeModeEvidence) -> Option<String> {
    let mut text = String::new();
    for file_name in ["stdout.txt", "stderr.txt"] {
        let path = evidence.artifact_dir.join(file_name);
        let Ok(part) = fs::read_to_string(path) else {
            continue;
        };
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&part);
    }
    (!text.is_empty()).then_some(text)
}

fn runtime_artifacts_have_randomized_count_policy(
    tlc: &RuntimeModeEvidence,
    ty: &RuntimeModeEvidence,
) -> bool {
    let mut operators = BTreeSet::new();
    let mut evidence_sources = BTreeSet::new();
    collect_randomized_operator_evidence_from_runtime_artifact(
        &tlc.artifact_dir,
        &mut operators,
        &mut evidence_sources,
    );
    collect_randomized_operator_evidence_from_runtime_artifact(
        &ty.artifact_dir,
        &mut operators,
        &mut evidence_sources,
    );
    !operators.is_empty()
}

fn collect_randomized_operator_evidence_from_runtime_artifact(
    artifact_dir: &Path,
    operators: &mut BTreeSet<String>,
    evidence_sources: &mut BTreeSet<String>,
) {
    let Ok(text) = fs::read_to_string(artifact_dir.join("command.json")) else {
        return;
    };
    let Ok(command) = serde_json::from_str::<Value>(&text) else {
        return;
    };
    collect_randomized_operator_evidence_from_metadata_value(
        &format!("runtime artifact {}", artifact_dir.display()),
        &command,
        operators,
        evidence_sources,
    );

    let cwd = command
        .get("cwd")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| artifact_dir.to_path_buf());
    let Some(argv) = command.get("argv").and_then(Value::as_array) else {
        return;
    };
    for arg in argv.iter().filter_map(Value::as_str) {
        let path = Path::new(arg);
        if path.extension().and_then(|extension| extension.to_str()) != Some("tla") {
            continue;
        }
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            cwd.join(path)
        };
        collect_randomized_operator_evidence_from_source_path(
            &resolved,
            &format!("runtime source {}", resolved.display()),
            operators,
            evidence_sources,
        );
    }
}

fn collect_randomized_operator_evidence_from_metadata_value(
    label: &str,
    value: &Value,
    operators: &mut BTreeSet<String>,
    evidence_sources: &mut BTreeSet<String>,
) {
    let Some(map) = value.as_object() else {
        return;
    };
    for (key, item) in map {
        if !metadata_key_may_report_randomized_operator(key) {
            continue;
        }
        for operator in RANDOMIZED_EXTERNAL_OPERATORS {
            if json_value_contains_operator(item, operator) {
                operators.insert((*operator).to_string());
                evidence_sources.insert(format!("{label}.{key}"));
            }
        }
    }
}

fn runtime_modes_have_fresh_evidence(tlc: &RuntimeModeEvidence, ty: &RuntimeModeEvidence) -> bool {
    runtime_mode_has_fresh_evidence(tlc) && runtime_mode_has_fresh_evidence(ty)
}

fn runtime_mode_has_fresh_evidence(evidence: &RuntimeModeEvidence) -> bool {
    has_finite_positive_runtime(evidence.runtime_seconds)
}

fn write_no_config_tlc_config(
    repo_root: &Path,
    spec_dir: &Path,
    no_config_flags: &[String],
) -> Result<PathBuf> {
    let cfg_path = absolutize(repo_root, &spec_dir.join("config_free.generated.cfg"));
    if let Some(parent) = cfg_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let config_text = matrix_refresh::no_config_tlc_config_text_from_flags(no_config_flags)
        .with_context(|| format!("derive generated TLC config for {}", cfg_path.display()))?;
    fs::write(&cfg_path, config_text).with_context(|| format!("write {}", cfg_path.display()))?;
    Ok(cfg_path)
}

fn run_tlc_runtime(
    spec_name: &str,
    tla_path: &Path,
    cfg_path: &Path,
    tlc_classpath: &str,
    tla_library: Option<&Path>,
    repo_root: &Path,
    spec_dir: &Path,
    timeout_seconds: u64,
) -> Result<RuntimeModeEvidence> {
    let artifact_dir = spec_dir.join("tlc-run1");
    let metadir = artifact_dir.join("tlc-metadir");
    let mut argv = tlc_matrix_runtime_base_argv(tlc_classpath, tla_library);
    argv.extend([
        tla_path.display().to_string(),
        "-config".to_string(),
        cfg_path.display().to_string(),
        "-metadir".to_string(),
        metadir.display().to_string(),
        "-workers".to_string(),
        "1".to_string(),
    ]);
    let result = run_command(CommandSpec {
        argv,
        cwd: tla_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        env_overrides: BTreeMap::new(),
        timeout_seconds,
        artifact_dir: absolutize(repo_root, &artifact_dir),
    })
    .with_context(|| format!("{spec_name}: run TLC"))?;
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_tlc_final_counts(&stdout, &stderr);
    let error_type =
        runtime_error_with_output(result.returncode, result.timed_out, &stdout, &stderr).or_else(
            || {
                counts
                    .states_found
                    .is_none()
                    .then(|| "missing_state_count".to_string())
            },
        );
    let runtime_seconds = runtime_seconds_for_evidence(&error_type, result.elapsed_seconds);
    Ok(RuntimeModeEvidence {
        status: status_for_result(result.returncode, result.timed_out),
        runtime_seconds,
        states: counts.states_found,
        error_type,
        artifact_dir: result.artifact_dir,
        ..RuntimeModeEvidence::default()
    })
}

fn run_tlc_simulation_runtime(
    spec_name: &str,
    tla_path: &Path,
    simulation_cfg_path: &Path,
    tlc_classpath: &str,
    tla_library: Option<&Path>,
    repo_root: &Path,
    spec_dir: &Path,
    timeout_seconds: u64,
) -> Result<RuntimeModeEvidence> {
    let artifact_dir = spec_dir.join("tlc-simulate-run1");
    let metadir = artifact_dir.join("tlc-metadir");
    let mut argv = tlc_matrix_runtime_base_argv(tlc_classpath, tla_library);
    argv.extend([
        tlc_simulation_mode_arg(tla_path),
        format!("num={MATRIX_SIMULATION_TRACES}"),
        "-config".to_string(),
        simulation_cfg_path.display().to_string(),
        "-depth".to_string(),
        MATRIX_SIMULATION_DEPTH.to_string(),
        "-seed".to_string(),
        MATRIX_SIMULATION_SEED.to_string(),
        "-metadir".to_string(),
        metadir.display().to_string(),
        "-workers".to_string(),
        "1".to_string(),
        tla_path.display().to_string(),
    ]);
    let result = run_command(CommandSpec {
        argv,
        cwd: tla_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        env_overrides: BTreeMap::new(),
        timeout_seconds,
        artifact_dir: absolutize(repo_root, &artifact_dir),
    })
    .with_context(|| format!("{spec_name}: run TLC simulation"))?;
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_tlc_final_counts(&stdout, &stderr);
    let error_type =
        runtime_error_with_output(result.returncode, result.timed_out, &stdout, &stderr);
    let runtime_seconds = runtime_seconds_for_evidence(&error_type, result.elapsed_seconds);
    Ok(RuntimeModeEvidence {
        status: status_for_result(result.returncode, result.timed_out),
        runtime_seconds,
        states: counts.states_found,
        error_type,
        artifact_dir: result.artifact_dir,
        ..RuntimeModeEvidence::default()
    })
}

fn tlc_matrix_runtime_base_argv(tlc_classpath: &str, tla_library: Option<&Path>) -> Vec<String> {
    let mut argv = tlc_java_single_thread_base_argv();
    if let Some(tla_library) = tla_library {
        argv.push(format!("-DTLA-Library={}", tla_library.display()));
    }
    argv.extend([
        "-cp".to_string(),
        tlc_classpath.to_string(),
        "tlc2.TLC".to_string(),
    ]);
    argv
}

fn write_simulation_runtime_config(
    cfg_path: &Path,
    repo_root: &Path,
    spec_dir: &Path,
) -> Result<PathBuf> {
    let config_text = fs::read_to_string(cfg_path)
        .with_context(|| format!("read simulation cfg {}", cfg_path.display()))?;
    let generated_cfg = spec_dir.join("simulation.generated.cfg");
    let generated_cfg = absolutize(repo_root, &generated_cfg);
    if let Some(parent) = generated_cfg.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(
        &generated_cfg,
        strip_simulation_checker_clauses(&config_text),
    )
    .with_context(|| format!("write {}", generated_cfg.display()))?;
    Ok(generated_cfg)
}

fn strip_simulation_checker_clauses(config_text: &str) -> String {
    let mut stripped = Vec::new();
    let mut dropping = false;
    for line in config_text.lines() {
        let keyword = config_section_keyword(line);
        if keyword.is_some_and(is_simulation_checker_clause) {
            dropping = true;
            continue;
        }
        if dropping && keyword.is_some_and(is_config_section_keyword) {
            dropping = false;
        }
        if !dropping {
            stripped.push(line);
        }
    }
    let mut text = stripped.join("\n");
    text.push('\n');
    text
}

fn config_section_keyword(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() || trimmed.starts_with("\\*") {
        return None;
    }
    trimmed.split_whitespace().next()
}

fn is_simulation_checker_clause(keyword: &str) -> bool {
    matches!(
        keyword,
        "INVARIANT" | "INVARIANTS" | "PROPERTY" | "PROPERTIES" | "POSTCONDITION" | "POSTCONDITIONS"
    )
}

fn is_config_section_keyword(keyword: &str) -> bool {
    matches!(
        keyword,
        "ACTION_CONSTRAINT"
            | "ACTION_CONSTRAINTS"
            | "ALIAS"
            | "CHECK_DEADLOCK"
            | "CONSTANT"
            | "CONSTANTS"
            | "CONSTRAINT"
            | "CONSTRAINTS"
            | "INIT"
            | "NEXT"
            | "SPECIFICATION"
            | "SYMMETRY"
            | "VIEW"
    ) || is_simulation_checker_clause(keyword)
}

fn tlc_simulation_mode_arg(tla_path: &Path) -> String {
    if source_text_requires_tlc_generate(tla_path) {
        "-generate".to_string()
    } else {
        "-simulate".to_string()
    }
}

fn source_text_requires_tlc_generate(tla_path: &Path) -> bool {
    let Ok(text) = fs::read_to_string(tla_path) else {
        return false;
    };
    let compact_text = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    compact_text.contains("TLCGet(\"config\").mode=\"generate\"")
}

fn run_ty_runtime(
    spec_name: &str,
    tla_path: &Path,
    cfg_path: &Path,
    ty_bin: &Path,
    repo_root: &Path,
    spec_dir: &Path,
    timeout_seconds: u64,
    ty_base_env: &BTreeMap<String, String>,
    allow_debug_runtime: bool,
) -> Result<RuntimeModeEvidence> {
    run_ty_check_runtime_command(
        spec_name,
        with_count_verify_flag(ty_config_runtime_argv(ty_bin, tla_path, cfg_path)),
        ty_matrix_runtime_refresh_env(spec_dir, ty_base_env),
        repo_root,
        &spec_dir.join("ty-trust_cg-run1"),
        timeout_seconds,
        allow_debug_runtime,
        "run TY trust-cg",
    )
}

/// Production-default speed measurement: same env as the pinned count-verify
/// run; only the count-parity `--no-reduction` flag is removed from the argv,
/// so auto-POR/auto-symmetry are free to engage — this run measures what users
/// actually get.
fn run_ty_production_runtime(
    spec_name: &str,
    tla_path: &Path,
    cfg_path: &Path,
    ty_bin: &Path,
    repo_root: &Path,
    spec_dir: &Path,
    timeout_seconds: u64,
    ty_base_env: &BTreeMap<String, String>,
    allow_debug_runtime: bool,
) -> Result<RuntimeModeEvidence> {
    run_ty_check_runtime_command(
        spec_name,
        ty_config_runtime_argv(ty_bin, tla_path, cfg_path),
        ty_matrix_runtime_refresh_env(spec_dir, ty_base_env),
        repo_root,
        &spec_dir.join("ty-trust_cg-production-run1"),
        timeout_seconds,
        allow_debug_runtime,
        "run TY trust-cg production-default",
    )
}

#[allow(clippy::too_many_arguments)]
fn run_ty_check_runtime_command(
    spec_name: &str,
    argv: Vec<String>,
    env_overrides: BTreeMap<String, String>,
    repo_root: &Path,
    artifact_dir: &Path,
    timeout_seconds: u64,
    allow_debug_runtime: bool,
    context_label: &str,
) -> Result<RuntimeModeEvidence> {
    let result = run_command(CommandSpec {
        argv,
        cwd: repo_root.to_path_buf(),
        env_overrides,
        timeout_seconds,
        artifact_dir: absolutize(repo_root, artifact_dir),
    })
    .with_context(|| format!("{spec_name}: {context_label}"))?;
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_ty_final_counts(&stdout, &stderr);
    let error_type = ty_runtime_error(
        result.returncode,
        result.timed_out,
        &stdout,
        &stderr,
        allow_debug_runtime,
    )
    .or_else(|| {
        counts
            .states_found
            .is_none()
            .then(|| "missing_state_count".to_string())
    });
    let runtime_seconds = runtime_seconds_for_evidence(&error_type, result.elapsed_seconds);
    Ok(RuntimeModeEvidence {
        status: status_for_result(result.returncode, result.timed_out),
        runtime_seconds,
        states: counts.states_found,
        error_type,
        artifact_dir: result.artifact_dir,
        ..RuntimeModeEvidence::default()
    })
}

fn run_ty_simulation_runtime(
    spec_name: &str,
    tla_path: &Path,
    cfg_path: &Path,
    ty_bin: &Path,
    repo_root: &Path,
    spec_dir: &Path,
    timeout_seconds: u64,
    allow_debug_runtime: bool,
) -> Result<RuntimeModeEvidence> {
    let artifact_dir = spec_dir.join("ty-simulate-run1");
    let result = run_command(CommandSpec {
        argv: ty_simulation_runtime_argv(ty_bin, tla_path, cfg_path),
        cwd: repo_root.to_path_buf(),
        env_overrides: BTreeMap::new(),
        timeout_seconds,
        artifact_dir: absolutize(repo_root, &artifact_dir),
    })
    .with_context(|| format!("{spec_name}: run TY simulation"))?;
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let error_type = ty_runtime_error(
        result.returncode,
        result.timed_out,
        &stdout,
        &stderr,
        allow_debug_runtime,
    );
    let runtime_seconds = runtime_seconds_for_evidence(&error_type, result.elapsed_seconds);
    Ok(RuntimeModeEvidence {
        status: status_for_result(result.returncode, result.timed_out),
        runtime_seconds,
        states: None,
        error_type,
        artifact_dir: result.artifact_dir,
        ..RuntimeModeEvidence::default()
    })
}

fn run_ty_no_config_runtime(
    spec_name: &str,
    tla_path: &Path,
    ty_bin: &Path,
    repo_root: &Path,
    spec_dir: &Path,
    timeout_seconds: u64,
    ty_base_env: &BTreeMap<String, String>,
    allow_debug_runtime: bool,
    no_config_flags: &[String],
) -> Result<RuntimeModeEvidence> {
    run_ty_check_runtime_command(
        spec_name,
        ty_no_config_runtime_argv(ty_bin, tla_path, no_config_flags),
        ty_matrix_runtime_refresh_env(spec_dir, ty_base_env),
        repo_root,
        &spec_dir.join("ty-trust_cg-no-config-run1"),
        timeout_seconds,
        allow_debug_runtime,
        "run TY trust-codegen no-config",
    )
}

/// Production-default speed measurement for config-free rows: same argv as the
/// pinned count-verify run; only the count-parity env pins are removed.
fn run_ty_no_config_production_runtime(
    spec_name: &str,
    tla_path: &Path,
    ty_bin: &Path,
    repo_root: &Path,
    spec_dir: &Path,
    timeout_seconds: u64,
    ty_base_env: &BTreeMap<String, String>,
    allow_debug_runtime: bool,
    no_config_flags: &[String],
) -> Result<RuntimeModeEvidence> {
    run_ty_check_runtime_command(
        spec_name,
        ty_no_config_runtime_argv(ty_bin, tla_path, no_config_flags),
        ty_matrix_runtime_refresh_env(spec_dir, ty_base_env),
        repo_root,
        &spec_dir.join("ty-trust_cg-no-config-production-run1"),
        timeout_seconds,
        allow_debug_runtime,
        "run TY trust-codegen no-config production-default",
    )
}

fn ty_config_runtime_argv(ty_bin: &Path, tla_path: &Path, cfg_path: &Path) -> Vec<String> {
    vec![
        ty_bin.display().to_string(),
        "check".to_string(),
        tla_path.display().to_string(),
        "--config".to_string(),
        cfg_path.display().to_string(),
        "--workers".to_string(),
        "1".to_string(),
        "--force".to_string(),
        "--backend".to_string(),
        "trust-cg".to_string(),
    ]
}

fn ty_simulation_runtime_argv(ty_bin: &Path, tla_path: &Path, cfg_path: &Path) -> Vec<String> {
    vec![
        ty_bin.display().to_string(),
        "simulate".to_string(),
        tla_path.display().to_string(),
        "--config".to_string(),
        cfg_path.display().to_string(),
        "--no-invariants".to_string(),
        "--allow-io".to_string(),
        "--num-traces".to_string(),
        MATRIX_SIMULATION_TRACES.to_string(),
        "--max-trace-length".to_string(),
        MATRIX_SIMULATION_DEPTH.to_string(),
        "--seed".to_string(),
        MATRIX_SIMULATION_SEED.to_string(),
    ]
}

fn ty_no_config_runtime_argv(
    ty_bin: &Path,
    tla_path: &Path,
    no_config_flags: &[String],
) -> Vec<String> {
    let mut argv = vec![
        ty_bin.display().to_string(),
        "check".to_string(),
        tla_path.display().to_string(),
    ];
    argv.extend(no_config_flags.iter().cloned());
    argv.extend([
        "--workers".to_string(),
        "1".to_string(),
        "--force".to_string(),
        "--backend".to_string(),
        "trust-cg".to_string(),
    ]);
    argv
}

fn preflight_runtime_ty_trust_cg_for_refresh(
    ty_bin: &Path,
    output_dir: &Path,
    repo_root: &Path,
    timeout_seconds: u64,
    ty_base_env: &BTreeMap<String, String>,
) -> Result<()> {
    let preflight_dir = output_dir.join("runtime-ty-trust_cg-preflight");
    let spec_path = preflight_dir.join("SupremacyMatrixRuntimePreflight.tla");
    fs::create_dir_all(&preflight_dir)
        .with_context(|| format!("create {}", preflight_dir.display()))?;
    fs::write(&spec_path, runtime_ty_trust_cg_preflight_spec())
        .with_context(|| format!("write {}", spec_path.display()))?;

    let command = ty_runtime_preflight_command_spec(
        ty_bin,
        &spec_path,
        repo_root,
        &preflight_dir,
        timeout_seconds,
        ty_base_env,
    );
    let result = run_command(command).with_context(|| {
        format!(
            "preflight --runtime-ty-bin {} for trust-codegen matrix runtime refresh",
            ty_bin.display()
        )
    })?;
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    if runtime_ty_preflight_reports_backend_unavailable(&stdout, &stderr) {
        bail!(
            "matrix runtime refresh preflight failed: selected runtime TY binary {} reports backend_unavailable for `ty check --backend trust_cg`. Rebuild it with `cargo build -p tla-cli --bin ty` and pass that binary, or pass a proper --runtime-ty-bin",
            ty_bin.display()
        );
    }
    if result.timed_out {
        bail!(
            "matrix runtime refresh preflight timed out after {}s for --runtime-ty-bin {}; refusing to run row batch",
            result.elapsed_seconds,
            ty_bin.display()
        );
    }
    if result.returncode != 0 {
        bail!(
            "matrix runtime refresh preflight failed for --runtime-ty-bin {} with exit code {}; refusing to run row batch. See {}",
            ty_bin.display(),
            result.returncode,
            result.artifact_dir.display()
        );
    }
    Ok(())
}

fn ty_runtime_preflight_command_spec(
    ty_bin: &Path,
    spec_path: &Path,
    repo_root: &Path,
    preflight_dir: &Path,
    timeout_seconds: u64,
    ty_base_env: &BTreeMap<String, String>,
) -> CommandSpec {
    CommandSpec {
        argv: ty_no_config_preflight_argv(ty_bin, spec_path),
        cwd: repo_root.to_path_buf(),
        env_overrides: ty_matrix_runtime_refresh_env(preflight_dir, ty_base_env),
        timeout_seconds: timeout_seconds.clamp(1, 10),
        artifact_dir: absolutize(repo_root, &preflight_dir.join("run")),
    }
}

fn ty_no_config_preflight_argv(ty_bin: &Path, tla_path: &Path) -> Vec<String> {
    let mut argv = vec![
        ty_bin.display().to_string(),
        "check".to_string(),
        tla_path.display().to_string(),
    ];
    argv.extend(matrix_refresh::no_config_cli_flags());
    argv.extend([
        "--workers".to_string(),
        "1".to_string(),
        "--force".to_string(),
        "--output".to_string(),
        "json".to_string(),
        "--backend".to_string(),
        "trust-cg".to_string(),
    ]);
    argv
}

fn runtime_ty_trust_cg_preflight_spec() -> &'static str {
    concat!(
        "---- MODULE SupremacyMatrixRuntimePreflight ----\n",
        "VARIABLE x\n",
        "MyInit == x = 0\n",
        "MyNext == UNCHANGED x\n",
        "TypeOK == x = 0\n",
        "====\n",
    )
}

fn runtime_ty_preflight_reports_backend_unavailable(stdout: &str, stderr: &str) -> bool {
    stdout.contains("backend_unavailable") || stderr.contains("backend_unavailable")
}

fn matrix_runtime_refresh_base_env() -> BTreeMap<String, String> {
    let compile_jobs = runtime_refresh_compile_jobs_value();
    matrix_runtime_refresh_base_env_with_compile_jobs(&compile_jobs)
}

// The runner drops inherited TY_* variables; capture this operator override
// before building CommandSpec env_overrides so shell-style CLI env assignment survives.
fn runtime_refresh_compile_jobs_value() -> String {
    runtime_refresh_compile_jobs_value_from_env(env::var(RUNTIME_REFRESH_COMPILE_JOBS_ENV).ok())
}

fn runtime_refresh_compile_jobs_value_from_env(value: Option<String>) -> String {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_RUNTIME_REFRESH_COMPILE_JOBS.to_string())
}

fn matrix_runtime_refresh_base_env_with_compile_jobs(
    compile_jobs: &str,
) -> BTreeMap<String, String> {
    // Auto-POR/auto-symmetry are NOT pinned here: those semantic levers are
    // controlled by CLI flags only (the child `ty check` ignores ambient
    // TY_AUTO_POR / TY_AUTO_SYMMETRY). The pinned count-verify runs pass the
    // `--no-reduction` flag (see `COUNT_VERIFY_FLAG`); production-default runs
    // omit it.
    BTreeMap::from([
        ("TY_trust_cg".to_string(), "1".to_string()),
        ("TY_TRUST_CG_BFS".to_string(), "1".to_string()),
        ("TY_TRUST_CG_EXISTS".to_string(), "1".to_string()),
        ("TY_BYTECODE_VM".to_string(), "1".to_string()),
        ("TY_BYTECODE_VM_STATS".to_string(), "1".to_string()),
        (
            RUNTIME_REFRESH_COMPILE_JOBS_ENV.to_string(),
            compile_jobs.to_string(),
        ),
        (
            "TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP".to_string(),
            "1".to_string(),
        ),
        ("TY_DISABLE_ARTIFACT_CACHE".to_string(), "1".to_string()),
    ])
}

fn ty_matrix_runtime_refresh_env(
    spec_dir: &Path,
    base_env: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut env = base_env.clone();
    env.insert(
        "TY_CACHE_DIR".to_string(),
        spec_dir
            .join("trust_cg-artifact-cache")
            .display()
            .to_string(),
    );
    env
}

/// CLI flag passed ONLY for the count-verify axis: it forces off the sound
/// state-count-reducing production defaults (auto-POR, auto-symmetry) so the
/// pinned run's `States found` stays unreduced-parity comparable with TLC.
/// This is a CLI flag (not the retired TY_AUTO_POR/TY_AUTO_SYMMETRY env pins)
/// because the child `ty check` ignores ambient env for these semantic levers.
const COUNT_VERIFY_FLAG: &str = "--no-reduction";

/// Append the count-verify lever to a planned `ty check` argv, keeping the
/// trailing `--backend <name>` pair last for readability of recorded commands.
fn with_count_verify_flag(mut argv: Vec<String>) -> Vec<String> {
    let backend_position = argv
        .iter()
        .position(|arg| arg == "--backend")
        .unwrap_or(argv.len());
    argv.insert(backend_position, COUNT_VERIFY_FLAG.to_string());
    argv
}

fn runtime_error_with_output(
    returncode: i32,
    timed_out: bool,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    if timed_out {
        return Some("timeout".to_string());
    }
    if let Some(error_type) = classified_runtime_error(returncode, stdout, stderr) {
        return Some(error_type);
    }
    if returncode == 0 {
        return None;
    }
    Some(
        stderr
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .or_else(|| stdout.lines().rev().find(|line| !line.trim().is_empty()))
            .unwrap_or("command failed")
            .to_string(),
    )
}

fn classified_runtime_error(returncode: i32, stdout: &str, stderr: &str) -> Option<String> {
    let lower_stdout = stdout.to_ascii_lowercase();
    let lower_stderr = stderr.to_ascii_lowercase();
    let has_marker = |marker: &str| lower_stdout.contains(marker) || lower_stderr.contains(marker);
    let combined = format!("{lower_stdout}\n{lower_stderr}");
    if has_marker("backend_unavailable") {
        return Some("backend_unavailable".to_string());
    }
    if has_marker("state is not completely specified by the initial predicate") {
        return Some("incomplete_initial_state".to_string());
    }
    if has_marker("cannot handle the temporal formula") {
        return Some("unsupported".to_string());
    }
    if has_marker("deadlock reached")
        || has_marker("deadlock found")
        || has_marker("error: deadlock")
    {
        return Some("deadlock".to_string());
    }
    if has_liveness_violation_marker(&combined) {
        return Some("liveness".to_string());
    }
    if has_invariant_violation_marker(&combined) {
        return Some("invariant".to_string());
    }
    if has_assume_violation_marker(&combined) {
        return Some("assume_violation".to_string());
    }
    if has_marker("action property") && has_marker("violated") {
        return Some("property".to_string());
    }
    if (combined.contains("property '") && has_marker("violated"))
        || combined.contains("property is violated")
        || (combined.contains("property ") && combined.contains(" is violated"))
    {
        return Some("property".to_string());
    }
    if has_marker("parse error")
        || has_marker("syntax error")
        || has_fatal_semantic_error_marker(returncode, &combined)
        || has_marker("semantic analysis failed")
        || has_marker("parsing or semantic analysis failed")
        || has_marker("failed to load extended modules")
        || has_marker("failed to load instanced modules")
    {
        return Some("parse".to_string());
    }
    None
}

fn has_fatal_semantic_error_marker(returncode: i32, combined_lower: &str) -> bool {
    let mut has_semantic_error_output = false;
    for line in combined_lower.lines() {
        let line = line.trim();
        if line_has_semantic_error_output(line) {
            has_semantic_error_output = true;
        }
        let Some(rest) = line.strip_prefix("*** errors:") else {
            continue;
        };
        if rest
            .split_whitespace()
            .next()
            .and_then(|count| {
                count
                    .trim_matches(|ch: char| !ch.is_ascii_digit())
                    .parse::<u64>()
                    .ok()
            })
            .is_some_and(|count| count > 0)
        {
            return true;
        }
    }
    returncode != 0 && has_semantic_error_output
}

fn line_has_semantic_error_output(line: &str) -> bool {
    let line = line
        .strip_prefix("error:")
        .map(str::trim_start)
        .unwrap_or(line);
    matches!(line, "semantic error" | "semantic errors")
        || line.starts_with("semantic error:")
        || line.starts_with("semantic errors:")
}

fn has_assume_violation_marker(combined_lower: &str) -> bool {
    combined_lower.lines().any(|line| {
        let line = line.trim();
        line.contains("assume_violation")
            || line.contains("assume false")
            || line.contains("assume_false")
            || (line.contains("assumption") && line.contains(" is false"))
            || (line.contains("assumption") && line.contains("violated"))
    })
}

fn has_liveness_violation_marker(combined_lower: &str) -> bool {
    combined_lower.lines().any(|line| {
        let line = line.trim();
        line.contains("temporal properties were violated")
            || line.contains("liveness violation")
            || (line.contains("liveness property") && line.contains("violated"))
    })
}

fn has_invariant_violation_marker(combined_lower: &str) -> bool {
    combined_lower.lines().any(|line| {
        let line = line.trim();
        line.contains("invariant") && line.contains("violated")
    })
}

fn ty_runtime_error(
    returncode: i32,
    timed_out: bool,
    stdout: &str,
    stderr: &str,
    allow_debug_runtime: bool,
) -> Option<String> {
    if !allow_debug_runtime {
        if let Some(contamination) = ty_runtime_output_contamination(stdout, stderr) {
            return Some(contamination);
        }
    }
    runtime_error_with_output(returncode, timed_out, stdout, stderr)
}

fn ty_runtime_output_contamination(stdout: &str, stderr: &str) -> Option<String> {
    let has_marker = |marker: &str| stdout.contains(marker) || stderr.contains(marker);
    if has_marker("Note: running an unoptimized debug build") {
        return Some("debug_build_runtime_evidence".to_string());
    }
    if [
        "=== Enumeration Profile ===",
        "=== Eval Profile ===",
        "=== Enumeration Detail Profile ===",
    ]
    .into_iter()
    .any(has_marker)
    {
        return Some("profile_runtime_evidence".to_string());
    }
    None
}

fn runtime_seconds_for_evidence(error_type: &Option<String>, elapsed_seconds: f64) -> Option<f64> {
    if error_type
        .as_deref()
        .is_some_and(runtime_error_invalidates_runtime_evidence)
    {
        None
    } else {
        Some(elapsed_seconds)
    }
}

fn runtime_error_invalidates_runtime_evidence(error_type: &str) -> bool {
    matches!(
        error_type,
        "backend_unavailable" | "debug_build_runtime_evidence" | "profile_runtime_evidence"
    )
}

fn status_for_result(returncode: i32, timed_out: bool) -> String {
    if timed_out {
        "timeout".to_string()
    } else if returncode == 0 {
        "pass".to_string()
    } else {
        "fail".to_string()
    }
}

fn apply_runtime_row(
    baseline: &mut Value,
    row: &RuntimeEvidenceRow,
    provenance: &RuntimeBaselineProvenance,
) {
    if !runtime_row_has_complete_fresh_evidence(row) {
        return;
    }
    let Some(spec) = baseline
        .get_mut("specs")
        .and_then(Value::as_object_mut)
        .and_then(|specs| specs.get_mut(&row.spec))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    update_mode_value(spec.get_mut("tlc"), &row.tlc, None);
    update_mode_value(spec.get_mut("ty"), &row.ty, Some(provenance));
    if runtime_row_can_update_verified_match(row) {
        spec.insert("verified_match".to_string(), json!(row.verified_match));
    }
}

fn update_mode_value(
    value: Option<&mut Value>,
    evidence: &RuntimeModeEvidence,
    provenance: Option<&RuntimeBaselineProvenance>,
) {
    let Some(mode) = value.and_then(Value::as_object_mut) else {
        return;
    };
    mode.insert("status".to_string(), json!(evidence.status));
    mode.insert("error_type".to_string(), json!(evidence.error_type));
    mode.insert(
        "runtime_seconds".to_string(),
        json!(evidence.runtime_seconds),
    );
    mode.insert("states".to_string(), json!(evidence.states));
    // Production-default axis (schema addition, backward-compatible):
    // `production_status` marks presence. When this refresh did not collect a
    // production run, remove any stale production fields so an older production
    // measurement never overlays freshly pinned numbers.
    if evidence.production_status.is_some() {
        mode.insert(
            "production_status".to_string(),
            json!(evidence.production_status),
        );
        mode.insert(
            "production_error_type".to_string(),
            json!(evidence.production_error_type),
        );
        mode.insert(
            "production_runtime_seconds".to_string(),
            json!(evidence.production_runtime_seconds),
        );
        mode.insert(
            "production_states".to_string(),
            json!(evidence.production_states),
        );
    } else {
        for field in [
            "production_status",
            "production_error_type",
            "production_runtime_seconds",
            "production_states",
        ] {
            mode.remove(field);
        }
    }
    if let Some(provenance) = provenance {
        mode.insert(
            "last_run".to_string(),
            Value::String(provenance.timestamp.clone()),
        );
        mode.insert(
            "git_commit".to_string(),
            Value::String(provenance.ty_git_commit.clone()),
        );
    }
}

fn runtime_row_can_update_verified_match(row: &RuntimeEvidenceRow) -> bool {
    !runtime_mode_is_timeout_evidence(&row.tlc) && !runtime_mode_is_timeout_evidence(&row.ty)
}

fn runtime_row_has_complete_fresh_evidence(row: &RuntimeEvidenceRow) -> bool {
    runtime_modes_have_fresh_evidence(&row.tlc, &row.ty)
}

fn validate_runtime_refresh_rows_promoted(rows: &[RuntimeEvidenceRow]) -> Result<()> {
    let failed = rows
        .iter()
        .filter(|row| !runtime_row_has_complete_fresh_evidence(row))
        .collect::<Vec<_>>();
    if failed.is_empty() {
        return Ok(());
    }

    let sample = failed
        .iter()
        .take(5)
        .map(|row| row.spec.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let suffix = if failed.len() > 5 { ", ..." } else { "" };
    bail!(
        "runtime refresh produced {} selected row(s) without promotable fresh evidence: {sample}{suffix}; see runtime_evidence.json errors and per-row artifacts",
        failed.len()
    )
}

fn runtime_mode_is_timeout_evidence(evidence: &RuntimeModeEvidence) -> bool {
    evidence.status.eq_ignore_ascii_case("timeout")
        || evidence
            .error_type
            .as_deref()
            .is_some_and(|error_type| error_type.to_ascii_lowercase().contains("timeout"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaselineMetadataRefresh {
    PromotionReady,
    WarningInserted,
}

fn refresh_runtime_baseline_metadata(
    baseline: &mut Value,
    rows: &[RuntimeEvidenceRow],
    provenance: &RuntimeBaselineProvenance,
) -> Result<BaselineMetadataRefresh> {
    let Some(root) = baseline.as_object_mut() else {
        bail!("baseline root is not an object");
    };
    if !root.get("specs").is_some_and(Value::is_object) {
        bail!("baseline has no 'specs' object");
    }

    if !baseline_has_supported_promotion_metadata(root) {
        root.insert(
            RUNTIME_METADATA_WARNING_FIELD.to_string(),
            runtime_metadata_warning(root, rows),
        );
        return Ok(BaselineMetadataRefresh::WarningInserted);
    }

    let existing_category_keys = root
        .get("categories")
        .and_then(Value::as_object)
        .map(|categories| categories.keys().cloned().collect::<BTreeSet<_>>())
        .unwrap_or_default();
    let has_categories = root.contains_key("categories");
    let (total_specs, stats, categories, specs_digest) = {
        let specs_obj = root
            .get("specs")
            .and_then(Value::as_object)
            .expect("validated specs object");
        let stats = compute_baseline_stats(specs_obj);
        let categories = compute_baseline_categories(specs_obj, existing_category_keys);
        let specs_digest = sha256_jcs_value(&Value::Object(specs_obj.clone()))?;
        (specs_obj.len(), stats, categories, specs_digest)
    };

    root.insert("total_specs".to_string(), json!(total_specs));
    root.insert("stats".to_string(), Value::Object(stats));
    if has_categories {
        root.insert("categories".to_string(), Value::Object(categories));
    }
    root.insert("specs_jcs_sha256".to_string(), Value::String(specs_digest));
    root.insert(
        "ty_refresh".to_string(),
        Value::Object(runtime_ty_refresh(provenance, rows)),
    );
    root.remove(RUNTIME_METADATA_WARNING_FIELD);
    Ok(BaselineMetadataRefresh::PromotionReady)
}

fn runtime_ty_refresh(
    provenance: &RuntimeBaselineProvenance,
    rows: &[RuntimeEvidenceRow],
) -> Map<String, Value> {
    let mut refresh = Map::new();
    refresh.insert(
        "git_commit".to_string(),
        Value::String(provenance.ty_git_commit.clone()),
    );
    refresh.insert(
        "script".to_string(),
        Value::String(runtime_refresh_script(provenance.allow_debug_runtime)),
    );
    refresh.insert("specs_ran".to_string(), json!(rows.len()));
    refresh.insert(
        "specs_updated".to_string(),
        json!(rows
            .iter()
            .filter(|row| runtime_row_has_complete_fresh_evidence(row))
            .count()),
    );
    refresh.insert(
        "allow_debug_runtime".to_string(),
        json!(provenance.allow_debug_runtime),
    );
    refresh.insert(
        "timestamp".to_string(),
        Value::String(provenance.timestamp.clone()),
    );
    refresh.insert(
        "binary_path".to_string(),
        Value::String(provenance.ty_binary.path.display().to_string()),
    );
    refresh.insert(
        "binary_sha256".to_string(),
        Value::String(provenance.ty_binary.sha256.clone()),
    );
    refresh
}

fn runtime_refresh_script(allow_debug_runtime: bool) -> String {
    if allow_debug_runtime {
        "ty supremacy matrix --refresh-runtime --allow-debug-runtime".to_string()
    } else {
        "ty supremacy matrix --refresh-runtime".to_string()
    }
}

fn baseline_has_supported_promotion_metadata(root: &Map<String, Value>) -> bool {
    root.get("schema_version")
        .and_then(Value::as_u64)
        .is_some_and(|schema_version| schema_version >= 3)
        || root.contains_key("stats")
        || root.contains_key("specs_jcs_sha256")
}

fn runtime_metadata_warning(root: &Map<String, Value>, rows: &[RuntimeEvidenceRow]) -> Value {
    let known_fields = ["specs_jcs_sha256", "stats", "categories"];
    let mut fields = known_fields
        .iter()
        .copied()
        .filter(|field| root.contains_key(*field))
        .collect::<Vec<_>>();
    if fields.is_empty() {
        fields.extend(known_fields);
    }

    json!({
        "promotion_ready": false,
        "reason": "baseline schema support for recomputing top-level metadata was not detected",
        "required_action": "refresh baseline metadata with the canonical baseline updater before promoting this file",
        "schema_version": root.get("schema_version").cloned().unwrap_or(Value::Null),
        "specs_collected": rows.len(),
        "specs_refreshed": rows
            .iter()
            .filter(|row| runtime_row_has_complete_fresh_evidence(row))
            .count(),
        "stale_or_unverified_top_level_fields": fields,
    })
}

fn compute_baseline_stats(specs_obj: &Map<String, Value>) -> Map<String, Value> {
    let mut tlc_pass = 0usize;
    let mut tlc_timeout = 0usize;
    let mut tlc_error = 0usize;
    let mut ty_match = 0usize;
    let mut ty_mismatch = 0usize;
    let mut ty_fail = 0usize;
    let mut ty_untested = 0usize;

    for spec in specs_obj.values() {
        let tlc_status = spec
            .get("tlc")
            .and_then(|tlc| tlc.get("status"))
            .and_then(Value::as_str)
            .or_else(|| spec.get("status").and_then(Value::as_str))
            .unwrap_or("unknown");
        match tlc_status {
            "pass" => tlc_pass += 1,
            "timeout" => tlc_timeout += 1,
            _ => tlc_error += 1,
        }

        let Some(ty) = spec.get("ty").and_then(Value::as_object) else {
            ty_untested += 1;
            continue;
        };

        let ty_status = ty
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("untested");
        let verified_match = spec
            .get("verified_match")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        match ty_status {
            "pass" if verified_match => ty_match += 1,
            "mismatch" => ty_mismatch += 1,
            "fail" => ty_fail += 1,
            _ => ty_untested += 1,
        }
    }

    let mut stats = Map::new();
    stats.insert("ty_fail".to_string(), json!(ty_fail));
    stats.insert("ty_match".to_string(), json!(ty_match));
    stats.insert("ty_mismatch".to_string(), json!(ty_mismatch));
    stats.insert("ty_untested".to_string(), json!(ty_untested));
    stats.insert("tlc_error".to_string(), json!(tlc_error));
    stats.insert("tlc_pass".to_string(), json!(tlc_pass));
    stats.insert("tlc_timeout".to_string(), json!(tlc_timeout));
    stats
}

fn compute_baseline_categories(
    specs_obj: &Map<String, Value>,
    mut category_keys: BTreeSet<String>,
) -> Map<String, Value> {
    let mut counts = BTreeMap::<String, usize>::new();
    for spec in specs_obj.values() {
        let category = spec
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        *counts.entry(category.to_string()).or_default() += 1;
    }
    category_keys.extend(counts.keys().cloned());

    let mut categories = Map::new();
    for category in category_keys {
        let count = counts.get(&category).copied().unwrap_or(0);
        categories.insert(category, json!(count));
    }
    categories
}

fn corpus_identity_from_baseline_value(value: &Value) -> Result<SupremacyMatrixCorpusIdentity> {
    let specs = value
        .get("specs")
        .context("baseline has no 'specs' object")?;
    let specs_obj = specs
        .as_object()
        .context("baseline 'specs' field is not an object")?;
    Ok(SupremacyMatrixCorpusIdentity {
        total_specs: specs_obj.len(),
        specs_jcs_sha256: Some(sha256_jcs_value(specs)?),
    })
}

fn sha256_jcs_value(value: &Value) -> Result<String> {
    let mut canonical = String::new();
    write_canonical_json(value, &mut canonical)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn canonicalize_lossy(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn validate_runtime_ty_binary_for_refresh(ty_bin: &Path, allow_debug_runtime: bool) -> Result<()> {
    if allow_debug_runtime || !is_debug_runtime_binary(ty_bin) {
        return Ok(());
    }
    bail!(
        "--runtime-ty-bin {} appears to be a debug-profile binary; rebuild and pass a release binary for promotable runtime evidence, or pass --allow-debug-runtime for development-only smoke evidence",
        ty_bin.display()
    )
}

fn is_debug_runtime_binary(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "debug")
}

fn current_ty_git_commit(repo_root: &Path) -> String {
    option_env!("TY_GIT_COMMIT")
        .filter(|commit| !commit.trim().is_empty() && *commit != "unknown")
        .map(ToOwned::to_owned)
        .or_else(|| git_command_text(repo_root, &["rev-parse", "--short", "HEAD"]).ok())
        .unwrap_or_else(|| "unknown".to_string())
}

fn java_version_provenance() -> RuntimeCommandVersionProvenance {
    command_version_provenance(vec!["java".to_string(), "-version".to_string()])
}

fn command_version_provenance(argv: Vec<String>) -> RuntimeCommandVersionProvenance {
    if argv.is_empty() {
        return RuntimeCommandVersionProvenance {
            argv,
            version: None,
            output: Vec::new(),
            status: None,
            error: Some("empty argv".to_string()),
        };
    }

    match Command::new(&argv[0]).args(&argv[1..]).output() {
        Ok(output) => {
            let output_lines = command_output_lines(&output.stdout, &output.stderr);
            RuntimeCommandVersionProvenance {
                argv,
                version: output_lines.first().cloned(),
                output: output_lines,
                status: output.status.code(),
                error: None,
            }
        }
        Err(err) => RuntimeCommandVersionProvenance {
            argv,
            version: None,
            output: Vec::new(),
            status: None,
            error: Some(err.to_string()),
        },
    }
}

fn command_output_lines(stdout: &[u8], stderr: &[u8]) -> Vec<String> {
    let combined = [stdout, stderr].concat();
    String::from_utf8_lossy(&combined)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn git_checkout_provenance(path: &Path) -> RuntimeGitCheckoutProvenance {
    let path = canonicalize_lossy(path);
    let mut errors = Vec::new();
    let worktree_root = match git_command_text(&path, &["rev-parse", "--show-toplevel"]) {
        Ok(root) => Some(PathBuf::from(root)),
        Err(err) => {
            errors.push(err);
            None
        }
    };
    let head = match git_command_text(&path, &["rev-parse", "HEAD"]) {
        Ok(head) => Some(head),
        Err(err) => {
            errors.push(err);
            None
        }
    };
    let head_short = match git_command_text(&path, &["rev-parse", "--short", "HEAD"]) {
        Ok(head) => Some(head),
        Err(err) => {
            errors.push(err);
            None
        }
    };
    let status = match git_command_text(&path, &["status", "--porcelain=v1"]) {
        Ok(status) => Some(status),
        Err(err) => {
            errors.push(err);
            None
        }
    };

    RuntimeGitCheckoutProvenance {
        path,
        worktree_root,
        head,
        head_short,
        is_dirty: status.as_ref().map(|status| !status.is_empty()),
        status_porcelain_sha256: status
            .as_ref()
            .map(|status| sha256_bytes(status.as_bytes())),
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

fn git_command_text(cwd: &Path, args: &[&str]) -> std::result::Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|err| format!("git {}: {err}", args.join(" ")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let message = stderr.trim();
        return Err(if message.is_empty() {
            format!("git {} exited with {}", args.join(" "), output.status)
        } else {
            format!("git {}: {message}", args.join(" "))
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
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
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(left, _)| *left);
            out.push('{');
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
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
    bail!("unsupported JSON number for canonicalization: {number}");
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

fn validate_file(path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        bail!("required file not found: {}", path.display())
    }
}

fn validate_dir(path: &Path) -> Result<()> {
    if path.is_dir() {
        Ok(())
    } else {
        bail!("required directory not found: {}", path.display())
    }
}

fn default_examples_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| PathBuf::from(home).join("tlaplus-examples/specifications"))
}

fn default_tlc_jar() -> PathBuf {
    env::var_os("TYTOOLS_JAR")
        .or_else(|| env::var_os("TLC_JAR"))
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(DEFAULT_TLC_JAR)))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_TLC_JAR))
}

fn default_community_modules_jar() -> Option<PathBuf> {
    env::var_os("COMMUNITY_MODULES")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME").map(|home| PathBuf::from(home).join(DEFAULT_COMMUNITY_MODULES_JAR))
        })
        .filter(|path| path.is_file())
}

fn resolve_runtime_tla_library(args: &SupremacyMatrixArgs, repo_root: &Path) -> Option<PathBuf> {
    resolve_runtime_tla_library_from(
        args.runtime_tla_library.clone(),
        repo_root,
        non_empty_env_path(ENV_TLA_LIBRARY),
        non_empty_env_path(ENV_TLA_PLUS_LIBRARY),
        env::var_os("HOME").map(PathBuf::from),
    )
}

fn resolve_runtime_tla_library_from(
    explicit: Option<PathBuf>,
    repo_root: &Path,
    tla_library_env: Option<PathBuf>,
    tla_plus_library_env: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path);
    }
    if let Some(path) = tla_library_env {
        return Some(path);
    }
    if let Some(path) = tla_plus_library_env {
        return Some(path);
    }

    let repo_library = repo_root.join(DEFAULT_TLA_LIBRARY);
    if repo_library.is_dir() {
        return Some(repo_library);
    }

    home.map(|home| home.join("tlapm/library"))
        .filter(|path| path.is_dir())
}

fn non_empty_env_path(key: &str) -> Option<PathBuf> {
    env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn tlc_classpath(tlc_jar: &Path, community_modules: Option<&Path>) -> Result<String> {
    let mut paths = vec![tlc_jar.to_path_buf()];
    if let Some(community_modules) = community_modules {
        paths.push(community_modules.to_path_buf());
    }
    let classpath = env::join_paths(paths).context("build TLC classpath")?;
    Ok(classpath.to_string_lossy().to_string())
}

fn default_runtime_output_dir() -> PathBuf {
    Path::new("reports").join("perf").join(format!(
        "{}-supremacy-matrix-runtime",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
    ))
}

fn absolutize(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

impl SupremacyMatrixCounts {
    fn add(&mut self, class: SupremacyMatrixClass) {
        match class {
            SupremacyMatrixClass::Unsupported => self.unsupported += 1,
            SupremacyMatrixClass::ExpectedViolationMatch => self.expected_violation_match += 1,
            SupremacyMatrixClass::TlcError => self.tlc_error += 1,
            SupremacyMatrixClass::TlcTimeout => self.tlc_timeout += 1,
            SupremacyMatrixClass::RuntimeToError => self.runtime_to_error += 1,
            SupremacyMatrixClass::TimeoutDominance => self.timeout_dominance += 1,
            SupremacyMatrixClass::TyTimeout => self.ty_timeout += 1,
            SupremacyMatrixClass::ParityFail => self.parity_fail += 1,
            SupremacyMatrixClass::MissingRuntime => self.missing_runtime += 1,
            SupremacyMatrixClass::PerfTie => self.perf_tie += 1,
            SupremacyMatrixClass::PerfLoser => self.perf_loser += 1,
            SupremacyMatrixClass::Pass => self.pass += 1,
        }
    }

    fn strict_blocker_count(&self) -> usize {
        self.unsupported
            + self.tlc_error
            + self.tlc_timeout
            + self.runtime_to_error
            + self.timeout_dominance
            + self.ty_timeout
            + self.parity_fail
            + self.missing_runtime
            + self.perf_tie
            + self.perf_loser
    }

    fn comparable_outcome_count(&self) -> usize {
        self.runtime_to_error + self.timeout_dominance
    }

    fn policy_blocker_count(&self, matrix_policy: &MatrixPolicy) -> usize {
        let allowed_comparable_outcomes = usize::from(matrix_policy.allow_runtime_to_error)
            * self.runtime_to_error
            + usize::from(matrix_policy.allow_timeout_dominance) * self.timeout_dominance;
        self.strict_blocker_count() - allowed_comparable_outcomes
    }
}

impl SupremacyMatrixPolicySummary {
    fn from_counts(counts: &SupremacyMatrixCounts, matrix_policy: &MatrixPolicy) -> Self {
        let blockers = counts.policy_blocker_count(matrix_policy);
        let pass = blockers == 0;
        Self {
            allow_runtime_to_error: matrix_policy.allow_runtime_to_error,
            allow_timeout_dominance: matrix_policy.allow_timeout_dominance,
            comparable_outcomes: counts.comparable_outcome_count(),
            pass,
            blockers,
            verdict: if pass {
                SupremacyMatrixVerdict::Pass
            } else {
                SupremacyMatrixVerdict::Fail
            },
        }
    }
}

impl SupremacyMatrixMissingRuntimeDiagnostics {
    fn from_rows(rows: &[SupremacyMatrixRow]) -> Option<Self> {
        let missing_rows = rows
            .iter()
            .filter(|row| row.class == SupremacyMatrixClass::MissingRuntime)
            .collect::<Vec<_>>();
        let specs_needing_measurement = missing_rows
            .iter()
            .map(|row| row.spec.clone())
            .collect::<Vec<_>>();
        if specs_needing_measurement.is_empty() {
            return None;
        }
        let missing_tlc_runtime_specs = missing_rows
            .iter()
            .filter(|row| row.missing_tlc_runtime)
            .map(|row| row.spec.clone())
            .collect::<Vec<_>>();
        let missing_ty_runtime_specs = missing_rows
            .iter()
            .filter(|row| row.missing_ty_runtime)
            .map(|row| row.spec.clone())
            .collect::<Vec<_>>();
        let specs_needing_measurement_details = missing_rows
            .iter()
            .map(|row| SupremacyMatrixMissingRuntimeDetail {
                spec: row.spec.clone(),
                missing_tlc_runtime: row.missing_tlc_runtime,
                missing_ty_runtime: row.missing_ty_runtime,
                reason: row.reason.clone(),
            })
            .collect::<Vec<_>>();
        let refresh_command = missing_runtime_refresh_command(&specs_needing_measurement_details);
        Some(Self {
            meaning: MISSING_RUNTIME_MEANING,
            launch_gate_policy: MISSING_RUNTIME_LAUNCH_GATE_POLICY,
            specs_needing_measurement,
            missing_tlc_runtime_specs,
            missing_ty_runtime_specs,
            specs_needing_measurement_details,
            refresh_command,
        })
    }

    fn refresh_command_text(&self) -> String {
        self.refresh_command.join(" ")
    }
}

fn missing_runtime_refresh_command(details: &[SupremacyMatrixMissingRuntimeDetail]) -> Vec<String> {
    let mut command = MISSING_RUNTIME_REFRESH_COMMAND_ARGS
        .iter()
        .map(|part| (*part).to_string())
        .collect::<Vec<_>>();
    for detail in details {
        command.push("--runtime-spec".to_string());
        command.push(detail.spec.clone());
    }
    command
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MatrixSourceMode<'a> {
    Check,
    Simulation,
    Unsupported(&'a str),
}

fn matrix_source_mode(entry: &BaselineSpec) -> MatrixSourceMode<'_> {
    match entry
        .source
        .as_ref()
        .and_then(|source| source.mode.as_deref())
    {
        None | Some("check") | Some("no_config") | Some("no-config") | Some("config_free")
        | Some("config-free") => MatrixSourceMode::Check,
        Some("simulate" | "generate") => MatrixSourceMode::Simulation,
        Some(mode) if is_bmc_only_source_mode(mode) => MatrixSourceMode::Check,
        Some(mode) => MatrixSourceMode::Unsupported(mode),
    }
}

fn is_simulation_source(entry: &BaselineSpec) -> bool {
    matrix_source_mode(entry) == MatrixSourceMode::Simulation
}

fn is_check_source(entry: &BaselineSpec) -> bool {
    matrix_source_mode(entry) == MatrixSourceMode::Check
}

fn is_bmc_only_source(entry: &BaselineSpec) -> bool {
    entry
        .source
        .as_ref()
        .and_then(|source| source.mode.as_deref())
        .is_some_and(is_bmc_only_source_mode)
}

fn is_bmc_only_source_mode(mode: &str) -> bool {
    let normalized = mode.to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "bmc" | "bmc-only" | "bounded-model-check" | "bounded-model-checking"
    )
}

fn tlc_impossible_reason(mode: &BaselineMode) -> Option<String> {
    let mut evidence = Vec::new();
    if text_is_not_runnable_marker(&mode.status) {
        evidence.push(format!("status={}", mode.status));
    }
    if let Some(error_type) = mode
        .error_type
        .as_deref()
        .filter(|error_type| text_is_not_runnable_marker(error_type))
    {
        evidence.push(format!("error_type={error_type}"));
    }
    (!evidence.is_empty()).then(|| {
        format!(
            "TLC-impossible: TLC baseline records unsupported/not-runnable evidence ({})",
            evidence.join(", ")
        )
    })
}

fn text_is_not_runnable_marker(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let normalized = lower.replace(['_', '-'], " ");
    lower.contains("unsupported")
        || normalized.contains("not supported")
        || normalized.contains("not runnable")
}

fn is_timeout(mode: &BaselineMode) -> bool {
    mode.status.eq_ignore_ascii_case("timeout")
        || mode
            .error_type
            .as_deref()
            .is_some_and(|error_type| error_type.to_ascii_lowercase().contains("timeout"))
}

fn production_evidence_is_timeout(mode: &BaselineMode) -> bool {
    mode.production_status
        .as_deref()
        .is_some_and(|status| status.eq_ignore_ascii_case("timeout"))
        || mode
            .production_error_type
            .as_deref()
            .is_some_and(|error_type| error_type.to_ascii_lowercase().contains("timeout"))
}

/// Verdict-consistency guard between the two TY measurement axes.
///
/// The pinned count-verify run and the production-default run check the same
/// spec with the same binary; any divergence in checker VERDICT (status +
/// classified error type) means a sound-by-default reduction (auto-POR /
/// auto-symmetry) changed the model-checking outcome — a soundness signal that
/// must hard-fail the row instead of feeding either runtime into the speed
/// axis. Timeouts on either axis are budget exhaustion, not verdicts, and are
/// excluded.
fn production_verdict_mismatch_reason(mode: &BaselineMode) -> Option<String> {
    let production_status = mode.production_status.as_deref()?;
    if is_timeout(mode) || production_evidence_is_timeout(mode) {
        return None;
    }
    let pinned_error = mode.error_type.as_deref();
    let production_error = mode.production_error_type.as_deref();
    if mode.status.eq_ignore_ascii_case(production_status) && pinned_error == production_error {
        return None;
    }
    Some(format!(
        "TY production-default verdict (status={production_status}, error_type={}) differs from the pinned count-verify verdict (status={}, error_type={}); verdict divergence between configurations is a soundness signal, not a perf number",
        production_error.unwrap_or("none"),
        mode.status,
        pinned_error.unwrap_or("none"),
    ))
}

fn is_tlc_error(tlc: &BaselineMode) -> bool {
    !tlc.status.eq_ignore_ascii_case("pass")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpectedViolationKind {
    Invariant,
    Property,
    Liveness,
    Assume,
    Deadlock,
}

impl ExpectedViolationKind {
    fn label(self) -> &'static str {
        match self {
            Self::Invariant => "invariant",
            Self::Property => "property",
            Self::Liveness => "liveness",
            Self::Assume => "assume_violation",
            Self::Deadlock => "deadlock",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedViolationIdentity {
    kind: ExpectedViolationKind,
    name: Option<String>,
}

impl ExpectedViolationIdentity {
    fn matches(&self, other: &Self, tlc_states: Option<u64>, ty_states: Option<u64>) -> bool {
        if self.kind != other.kind {
            return false;
        }
        match (&self.name, &other.name) {
            (Some(tlc_name), Some(ty_name)) => tlc_name == ty_name,
            (Some(_), None) | (None, Some(_)) => false,
            (None, None) => runtime_states_are_compatible(tlc_states, ty_states),
        }
    }
}

fn classify_expected_violation_match(
    entry: &BaselineSpec,
    _matrix_policy: &MatrixPolicy,
    tlc_seconds: Option<f64>,
    ty_seconds: Option<f64>,
) -> Option<(SupremacyMatrixClass, String)> {
    let tlc_kind = expected_violation_kind(entry.tlc.error_type.as_deref())?;
    if is_timeout(&entry.ty) {
        return Some((
            SupremacyMatrixClass::TyTimeout,
            "TY timed out before matching TLC's expected violation".to_string(),
        ));
    }
    let Some(ty_kind) = expected_violation_kind(entry.ty.error_type.as_deref()) else {
        return Some((
            SupremacyMatrixClass::ParityFail,
            format!(
                "TLC found expected {} violation, but TY did not report a matching violation",
                tlc_kind.label()
            ),
        ));
    };
    if !entry.verified_match {
        return Some((
            SupremacyMatrixClass::ParityFail,
            format!(
                "TLC and TY both reported {} violations, but baseline parity is not verified",
                tlc_kind.label()
            ),
        ));
    }
    if tlc_kind != ty_kind {
        return Some((
            SupremacyMatrixClass::ParityFail,
            format!(
                "TLC found expected {} violation, but TY reported {} violation",
                tlc_kind.label(),
                ty_kind.label()
            ),
        ));
    }
    Some((
        SupremacyMatrixClass::ExpectedViolationMatch,
        expected_violation_match_reason(tlc_kind, tlc_seconds, ty_seconds),
    ))
}

fn expected_violation_match_reason(
    kind: ExpectedViolationKind,
    tlc_seconds: Option<f64>,
    ty_seconds: Option<f64>,
) -> String {
    if let (Some(tlc_seconds), Some(ty_seconds)) = (tlc_seconds, ty_seconds) {
        if has_finite_positive_runtime(Some(tlc_seconds))
            && has_finite_positive_runtime(Some(ty_seconds))
            && ty_seconds < tlc_seconds
        {
            return format!(
                "TY reached the matching expected {} violation faster than TLC",
                kind.label()
            );
        }
    }
    format!(
        "TLC and TY reported matching expected {} violation; expected invalid/error rows are excluded from runtime supremacy comparisons",
        kind.label()
    )
}

fn classify_bmc_only_matching_error(
    entry: &BaselineSpec,
) -> Option<(SupremacyMatrixClass, String)> {
    if !is_bmc_only_source(entry) || !entry.verified_match || is_timeout(&entry.ty) {
        return None;
    }
    let tlc_error = entry.tlc.error_type.as_deref()?;
    let ty_error = entry.ty.error_type.as_deref()?;
    if !equivalent_error_types(tlc_error, ty_error) {
        return None;
    }
    Some((
        SupremacyMatrixClass::ExpectedViolationMatch,
        format!(
            "BMC-only fixture reported matching checker error `{}` under normal check; expected invalid/error rows are excluded from runtime supremacy comparisons",
            normalized_error_label(tlc_error, ty_error)
        ),
    ))
}

fn runtime_to_error_reason(matrix_policy: &MatrixPolicy, entry: &BaselineSpec) -> Option<String> {
    if !matrix_policy.allow_runtime_to_error
        || !entry.verified_match
        || entry.ty.status != "pass"
        || !comparable_error_types(&entry.tlc, &entry.ty)
    {
        return None;
    }
    let (Some(tlc_seconds), Some(ty_seconds)) =
        (entry.tlc.runtime_seconds, entry.ty.runtime_seconds)
    else {
        return None;
    };
    if !has_finite_positive_runtime(Some(tlc_seconds))
        || !has_finite_positive_runtime(Some(ty_seconds))
        || ty_seconds >= tlc_seconds
    {
        return None;
    }
    Some(format!(
        "policy permits runtime-to-error comparison: TY reached the matching error in {ty_seconds:.3}s before TLC reached it in {tlc_seconds:.3}s"
    ))
}

fn timeout_dominance_reason(matrix_policy: &MatrixPolicy, entry: &BaselineSpec) -> Option<String> {
    if !matrix_policy.allow_timeout_dominance
        || !entry.verified_match
        || entry.ty.status != "pass"
        || entry.ty.error_type.is_some()
    {
        return None;
    }
    let (Some(tlc_seconds), Some(ty_seconds)) =
        (entry.tlc.runtime_seconds, entry.ty.runtime_seconds)
    else {
        return None;
    };
    if !has_finite_positive_runtime(Some(tlc_seconds))
        || !has_finite_positive_runtime(Some(ty_seconds))
    {
        return None;
    }
    Some(format!(
        "policy permits timeout-dominance comparison: TY completed in {ty_seconds:.3}s while TLC timed out after {tlc_seconds:.3}s"
    ))
}

fn comparable_error_types(tlc: &BaselineMode, ty: &BaselineMode) -> bool {
    let Some(tlc_error) = tlc
        .error_type
        .as_deref()
        .and_then(normalized_comparable_error_type)
    else {
        return false;
    };
    let Some(ty_error) = ty
        .error_type
        .as_deref()
        .and_then(normalized_comparable_error_type)
    else {
        return false;
    };
    tlc_error == ty_error
}

fn equivalent_error_types(left: &str, right: &str) -> bool {
    if let (Some(left), Some(right)) = (
        normalized_comparable_error_type(left),
        normalized_comparable_error_type(right),
    ) {
        left == right
    } else {
        left.eq_ignore_ascii_case(right)
    }
}

fn normalized_error_label<'a>(left: &'a str, right: &str) -> &'a str {
    normalized_comparable_error_type(left)
        .or_else(|| normalized_comparable_error_type(right))
        .unwrap_or(left)
}

fn normalized_comparable_error_type(error_type: &str) -> Option<&'static str> {
    match error_type.to_ascii_lowercase().as_str() {
        "invariant" | "invariant_violation" => Some("invariant"),
        "liveness" | "liveness_violation" => Some("liveness"),
        "assume_violation" => Some("assume_violation"),
        "deadlock" => Some("deadlock"),
        _ => None,
    }
}

fn expected_violation_kind(error_type: Option<&str>) -> Option<ExpectedViolationKind> {
    match error_type?.to_ascii_lowercase().as_str() {
        "invariant" | "invariant_violation" => Some(ExpectedViolationKind::Invariant),
        "property" | "property_violation" | "action_property" | "action_property_violation" => {
            Some(ExpectedViolationKind::Property)
        }
        "liveness" | "liveness_violation" => Some(ExpectedViolationKind::Liveness),
        "assume_violation" => Some(ExpectedViolationKind::Assume),
        "deadlock" => Some(ExpectedViolationKind::Deadlock),
        _ => None,
    }
}

fn expected_violation_name(kind: ExpectedViolationKind, text: &str) -> Option<String> {
    match kind {
        ExpectedViolationKind::Invariant => violation_name_after_marker(text, "invariant ")
            .or_else(|| violation_name_after_marker(text, "invariant '")),
        ExpectedViolationKind::Property => violation_name_after_marker(text, "action property ")
            .or_else(|| violation_name_after_marker(text, "property "))
            .or_else(|| violation_name_after_marker(text, "property '")),
        ExpectedViolationKind::Liveness => violation_name_after_marker(text, "temporal property ")
            .or_else(|| violation_name_after_marker(text, "liveness property "))
            .or_else(|| violation_name_after_marker(text, "property ")),
        ExpectedViolationKind::Assume => violation_name_after_marker(text, "assumption ")
            .or_else(|| violation_name_after_marker(text, "assume ")),
        ExpectedViolationKind::Deadlock => None,
    }
}

fn violation_name_after_marker(text: &str, marker: &str) -> Option<String> {
    let marker = marker.to_ascii_lowercase();
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        let Some(marker_idx) = lower.find(&marker) else {
            continue;
        };
        let name_start = marker_idx + marker.len();
        let Some(name_end) = violation_name_end(&lower, name_start) else {
            continue;
        };
        if name_end <= name_start {
            continue;
        }
        let name = normalize_violation_name(&line[name_start..name_end]);
        if !name.is_empty() && !is_generic_violation_auxiliary(&name) {
            return Some(name);
        }
    }
    None
}

fn violation_name_end(lower_line: &str, name_start: usize) -> Option<usize> {
    let rest = &lower_line[name_start..];
    let mut candidates = [
        rest.find(" is violated").map(|idx| name_start + idx),
        rest.find(" violated").map(|idx| name_start + idx),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.into_iter().next()
}

fn normalize_violation_name(raw: &str) -> String {
    raw.trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '\'' | '"' | '`' | ':' | ';' | ',' | '.' | '(' | ')' | '[' | ']'
            )
        })
        .trim()
        .to_string()
}

fn is_generic_violation_auxiliary(name: &str) -> bool {
    matches!(name.to_ascii_lowercase().as_str(), "is" | "was")
}

#[cfg(test)]
fn expected_violation_kind_for_tlc_tool_error(
    code: i32,
    class: i32,
) -> Option<ExpectedViolationKind> {
    if class != crate::tlc_codes::mp::ERROR {
        return None;
    }
    match code {
        crate::tlc_codes::ec::TLC_INVARIANT_VIOLATED_BEHAVIOR => {
            Some(ExpectedViolationKind::Invariant)
        }
        crate::tlc_codes::ec::TLC_ACTION_PROPERTY_VIOLATED_BEHAVIOR => {
            Some(ExpectedViolationKind::Property)
        }
        crate::tlc_codes::ec::TLC_TEMPORAL_PROPERTY_VIOLATED => {
            Some(ExpectedViolationKind::Liveness)
        }
        crate::tlc_codes::ec::TLC_DEADLOCK_REACHED => Some(ExpectedViolationKind::Deadlock),
        _ => None,
    }
}

fn is_perf_loser(tlc_seconds: Option<f64>, ty_seconds: Option<f64>) -> bool {
    let (Some(tlc_seconds), Some(ty_seconds)) = (tlc_seconds, ty_seconds) else {
        return false;
    };
    tlc_seconds.is_finite()
        && ty_seconds.is_finite()
        && tlc_seconds > 0.0
        && ty_seconds >= tlc_seconds
        && perf_tie_reason(Some(tlc_seconds), Some(ty_seconds)).is_none()
}

fn perf_tie_reason(tlc_seconds: Option<f64>, ty_seconds: Option<f64>) -> Option<String> {
    let (Some(tlc_seconds), Some(ty_seconds)) = (tlc_seconds, ty_seconds) else {
        return None;
    };
    if !tlc_seconds.is_finite()
        || !ty_seconds.is_finite()
        || tlc_seconds <= 0.0
        || ty_seconds < tlc_seconds
    {
        return None;
    }

    let delta = ty_seconds - tlc_seconds;
    if delta <= PERF_TIE_TOLERANCE_SECONDS + f64::EPSILON {
        return Some(format!(
            "TY runtime is not strictly faster than TLC runtime, but the delta is within the {:.3}s tie tolerance",
            PERF_TIE_TOLERANCE_SECONDS
        ));
    }
    if tlc_seconds <= PERF_TIE_TINY_RUNTIME_FLOOR_SECONDS
        && ty_seconds <= PERF_TIE_TINY_RUNTIME_FLOOR_SECONDS
    {
        return Some(format!(
            "TLC and TY runtimes are both below the {:.3}s tiny-runtime tie floor",
            PERF_TIE_TINY_RUNTIME_FLOOR_SECONDS
        ));
    }
    None
}

fn missing_runtime_reason(tlc_seconds: Option<f64>, ty_seconds: Option<f64>) -> String {
    let missing_modes = match (
        has_finite_positive_runtime(tlc_seconds),
        has_finite_positive_runtime(ty_seconds),
    ) {
        (false, false) => "TLC and TY",
        (false, true) => "TLC",
        (true, false) => "TY",
        (true, true) => "TLC or TY",
    };
    format!(
        "baseline lacks finite positive {missing_modes} runtime_seconds; runtime_seconds must be present, finite, and greater than zero"
    )
}

fn missing_runtime_modes_for_row(
    class: SupremacyMatrixClass,
    entry: &BaselineSpec,
    tlc_seconds: Option<f64>,
    ty_seconds: Option<f64>,
) -> (bool, bool) {
    if class != SupremacyMatrixClass::MissingRuntime {
        return (false, false);
    }
    (
        !has_finite_positive_runtime(tlc_seconds),
        !has_finite_positive_runtime(ty_seconds) || undersized_ty_timeout_evidence(entry).is_some(),
    )
}

fn undersized_ty_timeout_reason(entry: &BaselineSpec) -> Option<String> {
    let (runtime_seconds, timeout_budget_seconds) = undersized_ty_timeout_evidence(entry)?;
    Some(format!(
        "TY runtime evidence timed out at {runtime_seconds:.3}s below spec-specific diagnose_timeout_seconds={timeout_budget_seconds}s; refresh runtime with --runtime-timeout >= {timeout_budget_seconds} before treating this as a TY timeout"
    ))
}

fn undersized_ty_timeout_evidence(entry: &BaselineSpec) -> Option<(f64, u64)> {
    if !is_timeout(&entry.ty) {
        return None;
    }
    let runtime_seconds = entry.ty.runtime_seconds?;
    if !runtime_seconds.is_finite() || runtime_seconds <= 0.0 {
        return None;
    }
    let timeout_budget_seconds = diagnose_timeout_seconds(entry)?;
    if runtime_seconds + 0.5 < timeout_budget_seconds as f64 {
        Some((runtime_seconds, timeout_budget_seconds))
    } else {
        None
    }
}

fn diagnose_timeout_seconds(entry: &BaselineSpec) -> Option<u64> {
    match entry.metadata.get("diagnose_timeout_seconds")? {
        Value::Number(number) => number
            .as_u64()
            .or_else(|| number.as_f64().and_then(positive_f64_to_u64_ceil)),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
    .filter(|seconds| *seconds > 0)
}

fn positive_f64_to_u64_ceil(value: f64) -> Option<u64> {
    if value.is_finite() && value > 0.0 && value <= u64::MAX as f64 {
        Some(value.ceil() as u64)
    } else {
        None
    }
}

fn has_finite_positive_runtime(seconds: Option<f64>) -> bool {
    seconds.is_some_and(|seconds| seconds.is_finite() && seconds > 0.0)
}

fn speedup(tlc_seconds: Option<f64>, ty_seconds: Option<f64>) -> Option<f64> {
    let (Some(tlc_seconds), Some(ty_seconds)) = (tlc_seconds, ty_seconds) else {
        return None;
    };
    if !tlc_seconds.is_finite()
        || !ty_seconds.is_finite()
        || tlc_seconds <= 0.0
        || ty_seconds <= 0.0
    {
        return None;
    }
    Some(tlc_seconds / ty_seconds)
}

fn slowdown(tlc_seconds: Option<f64>, ty_seconds: Option<f64>) -> Option<f64> {
    let (Some(tlc_seconds), Some(ty_seconds)) = (tlc_seconds, ty_seconds) else {
        return None;
    };
    if !tlc_seconds.is_finite()
        || !ty_seconds.is_finite()
        || tlc_seconds <= 0.0
        || ty_seconds <= 0.0
        || ty_seconds < tlc_seconds
    {
        return None;
    }
    Some(ty_seconds / tlc_seconds)
}

fn seconds_lost(tlc_seconds: Option<f64>, ty_seconds: Option<f64>) -> Option<f64> {
    let (Some(tlc_seconds), Some(ty_seconds)) = (tlc_seconds, ty_seconds) else {
        return None;
    };
    if !tlc_seconds.is_finite()
        || !ty_seconds.is_finite()
        || tlc_seconds <= 0.0
        || ty_seconds < tlc_seconds
    {
        return None;
    }
    Some(ty_seconds - tlc_seconds)
}

fn perf_loser_follow_up(class: SupremacyMatrixClass, spec: &str) -> Option<&'static str> {
    if class != SupremacyMatrixClass::PerfLoser {
        return None;
    }

    match spec {
        "MCReachabilityTestAllGraphs" => Some("alabsystems/ty#4391"),
        "dijkstra-mutex_Safety-4-processors" => Some("alabsystems/ty#4392"),
        _ => None,
    }
}

pub(super) fn print_summary(
    summary: &SupremacyMatrixSummary,
    format: crate::cli_schema::SupremacyOutputFormat,
) -> Result<()> {
    match format {
        crate::cli_schema::SupremacyOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(summary)?);
        }
        crate::cli_schema::SupremacyOutputFormat::Markdown => {
            println!("{}", summary.to_markdown());
        }
        crate::cli_schema::SupremacyOutputFormat::Human => {
            println!("{}", summary.to_human());
        }
    }
    Ok(())
}

impl SupremacyMatrixSummary {
    pub(super) fn total_rows(&self) -> usize {
        self.rows.len()
    }

    pub(super) fn strict_pass_count(&self) -> usize {
        self.counts.pass
    }

    pub(super) fn strict_blocker_count(&self) -> usize {
        self.strict_blockers
    }

    pub(super) fn enforce_blocker_count(&self) -> usize {
        self.policy
            .as_ref()
            .map(|policy| policy.blockers)
            .unwrap_or(self.strict_blockers)
    }

    fn comparable_outcome_count(&self) -> usize {
        self.counts.comparable_outcome_count()
    }

    fn policy_pass_count(&self) -> usize {
        self.strict_pass_count() + self.comparable_outcome_count()
    }

    fn to_human(&self) -> String {
        let mut out = String::new();
        out.push_str("All-runnable TLC supremacy matrix\n");
        let _ = writeln!(out, "rows: {}", self.total_rows());
        let _ = writeln!(out, "pass: {}", self.strict_pass_count());
        if let Some(policy) = &self.policy {
            let _ = writeln!(
                out,
                "policy_pass: {} (comparable_outcomes={}, blockers={})",
                self.policy_pass_count(),
                policy.comparable_outcomes,
                policy.blockers
            );
        }
        if self.policy.is_some() {
            let _ = writeln!(
	                out,
	                "strict_blocked: {} (unsupported={}, tlc_error={}, tlc_timeout={}, runtime_to_error={}, timeout_dominance={}, ty_timeout={}, parity_fail={}, missing_runtime={}, perf_tie={}, perf_loser={})",
	                self.strict_blocker_count(),
	                self.counts.unsupported,
	                self.counts.tlc_error,
	                self.counts.tlc_timeout,
	                self.counts.runtime_to_error,
                self.counts.timeout_dominance,
                self.counts.ty_timeout,
                self.counts.parity_fail,
                self.counts.missing_runtime,
                self.counts.perf_tie,
                self.counts.perf_loser,
            );
        } else {
            let _ = writeln!(
	                out,
	                "blocked: {} (unsupported={}, tlc_error={}, tlc_timeout={}, ty_timeout={}, parity_fail={}, missing_runtime={}, perf_tie={}, perf_loser={})",
	                self.strict_blocker_count(),
	                self.counts.unsupported,
	                self.counts.tlc_error,
	                self.counts.tlc_timeout,
	                self.counts.ty_timeout,
                self.counts.parity_fail,
                self.counts.missing_runtime,
                self.counts.perf_tie,
                self.counts.perf_loser,
            );
        }
        if !self.next_action_counts.is_empty() {
            out.push_str("next_actions:");
            for (action, count) in &self.next_action_counts {
                let _ = write!(out, " {action}={count}");
            }
            out.push('\n');
        }
        if let Some(diagnostics) = &self.missing_runtime_diagnostics {
            let _ = writeln!(out, "missing_runtime_meaning: {}", diagnostics.meaning);
            let _ = writeln!(
                out,
                "missing_runtime_policy: {}",
                diagnostics.launch_gate_policy
            );
            let _ = writeln!(
                out,
                "missing_runtime_specs: {}",
                diagnostics.specs_needing_measurement.join(", ")
            );
            if !diagnostics.missing_tlc_runtime_specs.is_empty() {
                let _ = writeln!(
                    out,
                    "missing_runtime_tlc_specs: {}",
                    diagnostics.missing_tlc_runtime_specs.join(", ")
                );
            }
            if !diagnostics.missing_ty_runtime_specs.is_empty() {
                let _ = writeln!(
                    out,
                    "missing_runtime_ty_specs: {}",
                    diagnostics.missing_ty_runtime_specs.join(", ")
                );
            }
            let _ = writeln!(
                out,
                "missing_runtime_refresh: {}",
                diagnostics.refresh_command_text()
            );
        }
        for row in self
            .rows
            .iter()
            .filter(|row| row.class != SupremacyMatrixClass::Pass)
        {
            let _ = write!(
                out,
                "- {}: {:?}: next_action={}: {}",
                row.spec,
                row.class,
                row.next_action.as_str(),
                row.reason
            );
            if let Some(speedup) = row.speedup_tlc_vs_ty {
                let _ = write!(out, " (speedup={speedup:.3}x)");
            }
            out.push('\n');
        }
        let perf_losers = self.ranked_perf_losers();
        if !perf_losers.is_empty() {
            out.push_str("ranked perf losers:\n");
            for row in perf_losers {
                let _ = writeln!(
                    out,
                    "- #{} {}: slowdown={}x lost={}s follow_up={}",
                    row.perf_loser_rank.expect("perf loser rank"),
                    row.spec,
                    format_optional_ratio(row.slowdown_ty_vs_tlc),
                    format_optional_seconds(row.seconds_lost_vs_tlc),
                    row.perf_loser_follow_up.as_deref().unwrap_or("-"),
                );
            }
        }
        out
    }

    fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# All-Runnable TLC Supremacy Matrix\n\n");
        out.push_str("| Metric | Count |\n|---|---:|\n");
        let _ = writeln!(out, "| Rows | {} |", self.total_rows());
        let _ = writeln!(out, "| Pass | {} |", self.strict_pass_count());
        if let Some(policy) = &self.policy {
            let _ = writeln!(out, "| Policy pass | {} |", self.policy_pass_count());
            let _ = writeln!(out, "| Policy blocked | {} |", policy.blockers);
            let _ = writeln!(
                out,
                "| Comparable outcomes | {} |",
                policy.comparable_outcomes
            );
        }
        let blocked_label = if self.policy.is_some() {
            "Strict blocked"
        } else {
            "Blocked"
        };
        let _ = writeln!(out, "| {blocked_label} | {} |", self.strict_blocker_count());
        let _ = writeln!(out, "| Unsupported | {} |", self.counts.unsupported);
        if self.counts.expected_violation_match > 0 {
            let _ = writeln!(
                out,
                "| Expected violation match | {} |",
                self.counts.expected_violation_match
            );
        }
        let _ = writeln!(out, "| TLC error | {} |", self.counts.tlc_error);
        let _ = writeln!(out, "| TLC timeout | {} |", self.counts.tlc_timeout);
        if self.counts.runtime_to_error > 0 {
            let _ = writeln!(
                out,
                "| Runtime-to-error | {} |",
                self.counts.runtime_to_error
            );
        }
        if self.counts.timeout_dominance > 0 {
            let _ = writeln!(
                out,
                "| Timeout dominance | {} |",
                self.counts.timeout_dominance
            );
        }
        let _ = writeln!(out, "| TY timeout | {} |", self.counts.ty_timeout);
        let _ = writeln!(out, "| Parity fail | {} |", self.counts.parity_fail);
        let _ = writeln!(out, "| Missing runtime | {} |", self.counts.missing_runtime);
        let _ = writeln!(out, "| Perf tie | {} |", self.counts.perf_tie);
        let _ = writeln!(out, "| Perf loser | {} |", self.counts.perf_loser);

        if !self.next_action_counts.is_empty() {
            out.push_str("\n## Next Actions\n\n");
            out.push_str("| Next action | Rows |\n");
            out.push_str("|---|---:|\n");
            for (action, count) in &self.next_action_counts {
                let _ = writeln!(out, "| {} | {} |", action, count);
            }
        }

        if let Some(diagnostics) = &self.missing_runtime_diagnostics {
            out.push_str("\n## Missing Runtime\n\n");
            let _ = write!(out, "{}\n\n", diagnostics.meaning);
            let _ = write!(out, "{}\n\n", diagnostics.launch_gate_policy);
            out.push_str("Refresh with the Rust CLI:\n\n");
            out.push_str("```bash\n");
            out.push_str(&diagnostics.refresh_command_text());
            out.push_str("\n```\n\n");
            out.push_str("| Spec needing measurement | Missing TLC runtime | Missing TY runtime | Reason |\n");
            out.push_str("|---|---:|---:|---|\n");
            for detail in &diagnostics.specs_needing_measurement_details {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} |",
                    detail.spec,
                    detail.missing_tlc_runtime,
                    detail.missing_ty_runtime,
                    detail.reason.replace('|', "\\|"),
                );
            }
        }

        let perf_losers = self.ranked_perf_losers();
        if !perf_losers.is_empty() {
            out.push_str("\n## Ranked Perf Losers\n\n");
            out.push_str("| Rank | Spec | TY/TLC slowdown | Seconds lost | TLC seconds | TY seconds | Follow-up |\n");
            out.push_str("|---:|---|---:|---:|---:|---:|---|\n");
            for row in perf_losers {
                let _ = writeln!(
                    out,
                    "| {} | {} | {}x | {} | {} | {} | {} |",
                    row.perf_loser_rank.expect("perf loser rank"),
                    row.spec,
                    format_optional_ratio(row.slowdown_ty_vs_tlc),
                    format_optional_seconds(row.seconds_lost_vs_tlc),
                    format_optional_seconds(row.tlc_seconds),
                    format_optional_seconds(row.ty_seconds),
                    row.perf_loser_follow_up.as_deref().unwrap_or("-"),
                );
            }
        }

        out.push_str(
            "\n| Spec | Class | Next action | TLC seconds | TY seconds | Speedup | Reason |\n",
        );
        out.push_str("|---|---|---|---:|---:|---:|---|\n");
        for row in self
            .rows
            .iter()
            .filter(|row| row.class != SupremacyMatrixClass::Pass)
        {
            let speedup = row
                .speedup_tlc_vs_ty
                .map(|speedup| format!("{speedup:.3}x"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "| {} | {:?} | {} | {} | {} | {} | {} |",
                row.spec,
                row.class,
                row.next_action.label(),
                format_optional_seconds(row.tlc_seconds),
                format_optional_seconds(row.ty_seconds),
                speedup,
                row.reason,
            );
        }
        out
    }

    fn ranked_perf_losers(&self) -> Vec<&SupremacyMatrixRow> {
        let mut rows: Vec<_> = self
            .rows
            .iter()
            .filter(|row| row.class == SupremacyMatrixClass::PerfLoser)
            .collect();
        rows.sort_by_key(|row| row.perf_loser_rank.unwrap_or(usize::MAX));
        rows
    }
}

fn format_optional_seconds(seconds: Option<f64>) -> String {
    seconds
        .filter(|seconds| seconds.is_finite())
        .map(|seconds| format!("{seconds:.3}"))
        .unwrap_or_else(|| "-".to_string())
}

fn format_optional_ratio(ratio: Option<f64>) -> String {
    ratio
        .filter(|ratio| ratio.is_finite())
        .map(|ratio| format!("{ratio:.3}"))
        .unwrap_or_else(|| "-".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supremacy_status_wire_consts_match_enum_serialization() {
        // Pin the JSON wire form against the constants so a future change
        // to either side fails the build. See SUPREMACY_STATUS_PASS doc
        // comment for why this matters.
        let pass_json = serde_json::to_string(&SupremacyMatrixVerdict::Pass)
            .expect("serialize Pass verdict to JSON");
        let fail_json = serde_json::to_string(&SupremacyMatrixVerdict::Fail)
            .expect("serialize Fail verdict to JSON");
        assert_eq!(pass_json, format!("\"{SUPREMACY_STATUS_PASS}\""));
        assert_eq!(fail_json, format!("\"{SUPREMACY_STATUS_FAIL}\""));
        assert_eq!(
            SupremacyMatrixVerdict::Pass.as_wire_str(),
            SUPREMACY_STATUS_PASS
        );
        assert_eq!(
            SupremacyMatrixVerdict::Fail.as_wire_str(),
            SUPREMACY_STATUS_FAIL
        );
        // The wire form is documented to be lowercase ASCII; ensure no
        // typos slipped in.
        assert_eq!(SUPREMACY_STATUS_PASS, "pass");
        assert_eq!(SUPREMACY_STATUS_FAIL, "fail");
    }

    fn repo_baseline_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/tlc_comparison/spec_baseline.json")
    }

    fn class_for(summary: &SupremacyMatrixSummary, spec: &str) -> SupremacyMatrixClass {
        summary
            .rows
            .iter()
            .find(|row| row.spec == spec)
            .unwrap_or_else(|| panic!("missing matrix row for {spec}"))
            .class
    }

    fn row_for<'a>(summary: &'a SupremacyMatrixSummary, spec: &str) -> &'a SupremacyMatrixRow {
        summary
            .rows
            .iter()
            .find(|row| row.spec == spec)
            .unwrap_or_else(|| panic!("missing matrix row for {spec}"))
    }

    fn reason_for<'a>(summary: &'a SupremacyMatrixSummary, spec: &str) -> &'a str {
        row_for(summary, spec).reason.as_str()
    }

    #[test]
    fn expected_violation_kind_matches_true_tlc_tool_error_codes() {
        use crate::tlc_codes::{ec, mp};

        assert_eq!(
            expected_violation_kind_for_tlc_tool_error(
                ec::TLC_INVARIANT_VIOLATED_BEHAVIOR,
                mp::ERROR
            ),
            Some(ExpectedViolationKind::Invariant)
        );
        assert_eq!(
            expected_violation_kind_for_tlc_tool_error(
                ec::TLC_ACTION_PROPERTY_VIOLATED_BEHAVIOR,
                mp::ERROR
            ),
            Some(ExpectedViolationKind::Property)
        );
        assert_eq!(
            expected_violation_kind_for_tlc_tool_error(
                ec::TLC_TEMPORAL_PROPERTY_VIOLATED,
                mp::ERROR
            ),
            Some(ExpectedViolationKind::Liveness)
        );
        assert_eq!(
            expected_violation_kind_for_tlc_tool_error(ec::TLC_DEADLOCK_REACHED, mp::ERROR),
            Some(ExpectedViolationKind::Deadlock)
        );
        assert_eq!(
            expected_violation_kind_for_tlc_tool_error(ec::GENERAL, mp::ERROR),
            None
        );
        assert_eq!(
            expected_violation_kind_for_tlc_tool_error(
                ec::TLC_INVARIANT_VIOLATED_BEHAVIOR,
                mp::NONE
            ),
            None
        );
    }

    fn write_file(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, text).unwrap();
    }

    fn randomized_count_policy_baseline(
        examples_dir: &Path,
        tla_text: &str,
        verified_match: bool,
    ) -> String {
        let tla_path = examples_dir.join("RandomizedFixture.tla");
        write_file(&tla_path, tla_text);
        format!(
            r#"{{
              "inputs": {{"examples_dir": "{}"}},
              "specs": {{
                "randomized_fixture": {{
                  "source": {{"tla_path": "RandomizedFixture.tla"}},
                  "tlc": {{"status": "pass", "runtime_seconds": 2.0, "states": 10, "error_type": null}},
                  "ty": {{"status": "pass", "runtime_seconds": 1.0, "states": 8, "error_type": null}},
                  "verified_match": {verified_match}
                }}
              }}
            }}"#,
            examples_dir.display()
        )
    }

    #[test]
    fn randomized_count_policy_allows_both_pass_state_mismatch_with_source_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let baseline = randomized_count_policy_baseline(
            dir.path(),
            "---- MODULE RandomizedFixture ----\nEXTENDS TLC\nEdges == RandomElement({1, 2})\n====\n",
            false,
        );

        let summary = classify_baseline_str(&baseline).unwrap();

        assert_eq!(
            class_for(&summary, "randomized_fixture"),
            SupremacyMatrixClass::Pass
        );
        assert_eq!(summary.strict_blockers, 0);
        assert_eq!(summary.counts.pass, 1);
        let reason = reason_for(&summary, "randomized_fixture");
        assert!(reason.contains(RANDOMIZED_COUNT_POLICY_REASON_PREFIX));
        assert!(reason.contains("RandomElement"));
        assert!(reason.contains("TLC states=10"));
        assert!(reason.contains("TY states=8"));
    }

    #[test]
    fn randomized_count_policy_is_reported_on_missing_runtime_rows() {
        let dir = tempfile::tempdir().unwrap();
        let tla_path = dir.path().join("RandomizedFixture.tla");
        write_file(
            &tla_path,
            "---- MODULE RandomizedFixture ----\nEXTENDS TLC\nPick == RandomElement({1, 2})\n====\n",
        );
        let baseline = format!(
            r#"{{
              "inputs": {{"examples_dir": "{}"}},
              "specs": {{
                "randomized_missing_runtime": {{
                  "source": {{"tla_path": "RandomizedFixture.tla"}},
                  "tlc": {{"status": "pass", "runtime_seconds": 2.0, "states": 10, "error_type": null}},
                  "ty": {{"status": "pass", "states": 8, "error_type": null}},
                  "verified_match": false
                }}
              }}
            }}"#,
            dir.path().display()
        );

        let summary = classify_baseline_str(&baseline).unwrap();

        assert_eq!(
            class_for(&summary, "randomized_missing_runtime"),
            SupremacyMatrixClass::MissingRuntime
        );
        let reason = reason_for(&summary, "randomized_missing_runtime");
        assert!(reason.contains(RANDOMIZED_COUNT_POLICY_REASON_PREFIX));
        assert!(reason.contains("TY runtime_seconds"));
    }

    #[test]
    fn non_random_both_pass_state_mismatch_remains_parity_failure() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "non_random_mismatch": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 10, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 8, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "non_random_mismatch"),
            SupremacyMatrixClass::ParityFail
        );
        assert!(reason_for(&summary, "non_random_mismatch")
            .contains("no randomized external operator evidence"));
    }

    #[test]
    fn speed_classification_prefers_production_runtime_over_pinned() {
        // Pinned count-verify runtime (43.7s) loses to TLC (12.0s), but the
        // production-default runtime (8.5s, auto-symmetry engaged) wins: the
        // speed axis must classify on production. The differing
        // production_states (reduced orbit count) must NOT affect parity.
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "symmetric_row": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 12.0, "states": 100, "error_type": null},
                  "ty": {
                    "status": "pass", "runtime_seconds": 43.7, "states": 100, "error_type": null,
                    "production_status": "pass", "production_error_type": null,
                    "production_runtime_seconds": 8.5, "production_states": 10
                  },
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "symmetric_row"),
            SupremacyMatrixClass::Pass
        );
        let row = row_for(&summary, "symmetric_row");
        assert_eq!(row.ty_seconds, Some(8.5));
        assert_eq!(row.ty_pinned_seconds, Some(43.7));
        assert!(row
            .speedup_tlc_vs_ty
            .is_some_and(|speedup| (speedup - 12.0 / 8.5).abs() < 1e-9));
    }

    #[test]
    fn speed_classification_falls_back_to_pinned_runtime_without_production_evidence() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "pinned_only_row": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 12.0, "states": 100, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 43.7, "states": 100, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "pinned_only_row"),
            SupremacyMatrixClass::PerfLoser
        );
        let row = row_for(&summary, "pinned_only_row");
        assert_eq!(row.ty_seconds, Some(43.7));
        assert_eq!(row.ty_pinned_seconds, None);
    }

    #[test]
    fn production_runtime_slower_than_pinned_classifies_as_perf_loser() {
        // The production number owns the speed axis in BOTH directions: a row
        // whose pinned run wins but whose production-default run loses is a
        // perf loser (that is what users experience).
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "production_loser_row": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 12.0, "states": 100, "error_type": null},
                  "ty": {
                    "status": "pass", "runtime_seconds": 1.0, "states": 100, "error_type": null,
                    "production_status": "pass", "production_error_type": null,
                    "production_runtime_seconds": 30.0, "production_states": 100
                  },
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "production_loser_row"),
            SupremacyMatrixClass::PerfLoser
        );
        let row = row_for(&summary, "production_loser_row");
        assert_eq!(row.ty_seconds, Some(30.0));
        assert_eq!(row.ty_pinned_seconds, Some(1.0));
    }

    #[test]
    fn production_verdict_mismatch_is_a_hard_error_not_a_perf_number() {
        // Pinned run passed; production-default run reported an invariant
        // violation. A sound-by-default reduction changing the verdict is a
        // soundness signal: the row must hard-fail, never count as a win or
        // feed either runtime into the speed axis.
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "verdict_divergent_row": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 12.0, "states": 100, "error_type": null},
                  "ty": {
                    "status": "pass", "runtime_seconds": 4.0, "states": 100, "error_type": null,
                    "production_status": "fail", "production_error_type": "invariant",
                    "production_runtime_seconds": 0.3, "production_states": 7
                  },
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "verdict_divergent_row"),
            SupremacyMatrixClass::ParityFail
        );
        let reason = reason_for(&summary, "verdict_divergent_row");
        assert!(reason.contains("production-default verdict"));
        assert!(reason.contains("soundness signal"));
        assert_eq!(summary.counts.parity_fail, 1);
        assert_eq!(summary.counts.pass, 0);
    }

    #[test]
    fn production_verdict_mismatch_outranks_expected_violation_match() {
        // Pinned TY matched TLC's expected invariant violation, but the
        // production-default run PASSED (missed the violation): the scariest
        // divergence. The row must not be excused as ExpectedViolationMatch.
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "missed_violation_row": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {
                    "status": "fail", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant",
                    "production_status": "pass", "production_error_type": null,
                    "production_runtime_seconds": 0.9, "production_states": 12
                  },
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "missed_violation_row"),
            SupremacyMatrixClass::ParityFail
        );
        assert!(reason_for(&summary, "missed_violation_row").contains("soundness signal"));
    }

    #[test]
    fn production_timeout_is_a_budget_signal_not_a_verdict_mismatch() {
        // A production-default run that exhausts the timeout budget is not a
        // verdict divergence; its elapsed seconds still own the speed axis
        // (users would wait at least that long), so the row is a perf loser.
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "production_timeout_row": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 12.0, "states": 100, "error_type": null},
                  "ty": {
                    "status": "pass", "runtime_seconds": 4.0, "states": 100, "error_type": null,
                    "production_status": "timeout", "production_error_type": "timeout",
                    "production_runtime_seconds": 300.0, "production_states": null
                  },
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "production_timeout_row"),
            SupremacyMatrixClass::PerfLoser
        );
        assert_eq!(summary.counts.parity_fail, 0);
    }

    #[test]
    fn production_verdict_mismatch_reason_compares_status_and_error_type() {
        let mode = |status: &str,
                    error_type: Option<&str>,
                    production_status: Option<&str>,
                    production_error_type: Option<&str>| BaselineMode {
            status: status.to_string(),
            error_type: error_type.map(str::to_string),
            runtime_seconds: Some(1.0),
            states: Some(10),
            production_status: production_status.map(str::to_string),
            production_error_type: production_error_type.map(str::to_string),
            production_runtime_seconds: Some(0.5),
            production_states: Some(5),
            metadata: BTreeMap::new(),
        };

        // No production evidence: guard never fires.
        assert!(production_verdict_mismatch_reason(&mode("pass", None, None, None)).is_none());
        // Equal verdicts: no mismatch.
        assert!(
            production_verdict_mismatch_reason(&mode("pass", None, Some("pass"), None)).is_none()
        );
        assert!(production_verdict_mismatch_reason(&mode(
            "fail",
            Some("invariant"),
            Some("fail"),
            Some("invariant")
        ))
        .is_none());
        // Status divergence and error-type divergence both fire.
        assert!(production_verdict_mismatch_reason(&mode(
            "pass",
            None,
            Some("fail"),
            Some("invariant")
        ))
        .is_some());
        assert!(production_verdict_mismatch_reason(&mode(
            "fail",
            Some("invariant"),
            Some("fail"),
            Some("deadlock")
        ))
        .is_some());
        // Timeouts on either axis are excluded.
        assert!(production_verdict_mismatch_reason(&mode(
            "pass",
            None,
            Some("timeout"),
            Some("timeout")
        ))
        .is_none());
        assert!(production_verdict_mismatch_reason(&mode(
            "timeout",
            Some("timeout"),
            Some("pass"),
            None
        ))
        .is_none());
    }

    #[test]
    fn runtime_env_is_reducer_pin_free_and_count_verify_uses_the_flag() {
        // The retired design pinned TY_AUTO_POR/TY_AUTO_SYMMETRY=0 in the env;
        // semantic levers are now CLI-flag-only (the child `ty check` ignores
        // ambient env), so the runtime env must carry NO reducer pins and the
        // count-verify axis must get its lever via `--no-reduction` in argv.
        let base = matrix_runtime_refresh_base_env();
        let spec_dir = Path::new("/tmp/production-env-spec");
        let env = ty_matrix_runtime_refresh_env(spec_dir, &base);
        for key in ["TY_AUTO_POR", "TY_AUTO_SYMMETRY"] {
            assert!(
                !env.contains_key(key),
                "runtime env must not pin {key} (semantic levers are CLI flags now)"
            );
        }

        // Count-verify inserts the flag before the trailing `--backend` pair.
        let argv = vec![
            "ty".to_string(),
            "check".to_string(),
            "Spec.tla".to_string(),
            "--backend".to_string(),
            "trust-cg".to_string(),
        ];
        let with_flag = with_count_verify_flag(argv);
        assert_eq!(
            with_flag,
            vec![
                "ty".to_string(),
                "check".to_string(),
                "Spec.tla".to_string(),
                COUNT_VERIFY_FLAG.to_string(),
                "--backend".to_string(),
                "trust-cg".to_string(),
            ],
            "count-verify lever must be the {COUNT_VERIFY_FLAG} CLI flag, inserted before --backend"
        );

        // Without a --backend pair the flag is appended.
        let with_flag_no_backend =
            with_count_verify_flag(vec!["ty".to_string(), "check".to_string()]);
        assert_eq!(
            with_flag_no_backend.last().map(String::as_str),
            Some(COUNT_VERIFY_FLAG)
        );
    }

    #[test]
    fn apply_runtime_row_records_and_clears_production_axis_fields() {
        let mut baseline = json!({
            "specs": {
                "Row": {
                    "source": {},
                    "tlc": {"status": "pass", "states": null, "error_type": null},
                    "ty": {"status": "pass", "states": null, "error_type": null},
                    "verified_match": false
                }
            }
        });
        let provenance = runtime_baseline_provenance();
        let mut row = runtime_evidence_row("Row", 2.0, 1.0, 10);
        row.ty = attach_production_runtime_evidence(
            row.ty,
            RuntimeModeEvidence {
                status: "pass".to_string(),
                runtime_seconds: Some(0.5),
                states: Some(4),
                error_type: None,
                artifact_dir: PathBuf::from("Row/ty-trust_cg-production-run1"),
                ..RuntimeModeEvidence::default()
            },
        );

        apply_runtime_row(&mut baseline, &row, &provenance);
        let ty = &baseline["specs"]["Row"]["ty"];
        assert_eq!(ty["runtime_seconds"], json!(1.0));
        assert_eq!(ty["production_status"], json!("pass"));
        assert_eq!(ty["production_runtime_seconds"], json!(0.5));
        assert_eq!(ty["production_states"], json!(4));
        assert_eq!(ty["production_error_type"], json!(null));
        assert!(baseline["specs"]["Row"]["tlc"]
            .get("production_status")
            .is_none());

        // A later refresh WITHOUT a production run must clear the stale
        // production axis so old production numbers never overlay fresh
        // pinned evidence.
        let row = runtime_evidence_row("Row", 2.0, 1.5, 10);
        apply_runtime_row(&mut baseline, &row, &provenance);
        let ty = &baseline["specs"]["Row"]["ty"];
        assert_eq!(ty["runtime_seconds"], json!(1.5));
        assert!(ty.get("production_status").is_none());
        assert!(ty.get("production_runtime_seconds").is_none());
        assert!(ty.get("production_states").is_none());
        assert!(ty.get("production_error_type").is_none());
    }

    #[test]
    fn randomized_count_policy_ignores_comment_only_random_element_mentions() {
        let dir = tempfile::tempdir().unwrap();
        let baseline = randomized_count_policy_baseline(
            dir.path(),
            "---- MODULE RandomizedFixture ----\n\\* RandomElement({1, 2}) is mentioned only in a comment.\nPick == 1\n====\n",
            false,
        );

        let summary = classify_baseline_str(&baseline).unwrap();

        assert_eq!(
            class_for(&summary, "randomized_fixture"),
            SupremacyMatrixClass::ParityFail
        );
        assert!(!reason_for(&summary, "randomized_fixture")
            .contains(RANDOMIZED_COUNT_POLICY_REASON_PREFIX));
    }

    fn write_runtime_command_artifact(artifact_dir: &Path, source_path: &Path) {
        fs::create_dir_all(artifact_dir).unwrap();
        let source_arg = source_path.display().to_string();
        let cwd = source_path.parent().unwrap().display().to_string();
        write_file(
            &artifact_dir.join("command.json"),
            &(serde_json::to_string_pretty(&json!({
                "argv": ["ty", "check", source_arg],
                "cwd": cwd,
                "returncode": 0,
                "elapsed_seconds": 1.0,
                "env_overrides": {},
                "timed_out": false
            }))
            .unwrap()
                + "\n"),
        );
    }

    #[test]
    fn runtime_evidence_match_allows_randomized_source_state_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("RandomizedFixture.tla");
        write_file(
            &source_path,
            "---- MODULE RandomizedFixture ----\nEXTENDS TLC\nPick == RandomElement({1, 2})\n====\n",
        );
        let tlc_dir = dir.path().join("tlc");
        let ty_dir = dir.path().join("ty");
        write_runtime_command_artifact(&tlc_dir, &source_path);
        write_runtime_command_artifact(&ty_dir, &source_path);
        let mut tlc = runtime_mode_evidence("pass", Some(2.0), Some(10), None, "Row", "tlc");
        tlc.artifact_dir = tlc_dir;
        let mut ty = runtime_mode_evidence("pass", Some(1.0), Some(8), None, "Row", "ty");
        ty.artifact_dir = ty_dir;

        assert!(runtime_modes_verified_match(&tlc, &ty));
    }

    #[test]
    fn runtime_evidence_match_rejects_non_random_state_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("PlainFixture.tla");
        write_file(
            &source_path,
            "---- MODULE PlainFixture ----\nPick == 1\n====\n",
        );
        let tlc_dir = dir.path().join("tlc");
        let ty_dir = dir.path().join("ty");
        write_runtime_command_artifact(&tlc_dir, &source_path);
        write_runtime_command_artifact(&ty_dir, &source_path);
        let mut tlc = runtime_mode_evidence("pass", Some(2.0), Some(10), None, "Row", "tlc");
        tlc.artifact_dir = tlc_dir;
        let mut ty = runtime_mode_evidence("pass", Some(1.0), Some(8), None, "Row", "ty");
        ty.artifact_dir = ty_dir;

        assert!(!runtime_modes_verified_match(&tlc, &ty));
    }

    fn runtime_refresh_baseline(specs: &str, examples_dir: &Path) -> String {
        format!(
            r#"{{
              "inputs": {{"examples_dir": "{}"}},
              "specs": {{{specs}}}
            }}"#,
            examples_dir.display()
        )
    }

    fn missing_runtime_spec_json(name: &str, tla_path: &Path, cfg_path: Option<&Path>) -> String {
        let source = match cfg_path {
            Some(cfg_path) => format!(
                r#""source": {{"tla_path": "{}", "cfg_path": "{}"}}"#,
                tla_path.display(),
                cfg_path.display()
            ),
            None => format!(r#""source": {{"tla_path": "{}"}}"#, tla_path.display()),
        };
        format!(
            r#""{name}": {{
              "category": "small",
              {source},
              "tlc": {{"status": "pass", "states": 3, "error_type": null}},
              "ty": {{"status": "pass", "states": 3, "error_type": null}},
              "verified_match": true
            }}"#
        )
    }

    fn missing_runtime_simulation_spec_json(
        name: &str,
        mode: &str,
        tla_path: &Path,
        cfg_path: &Path,
    ) -> String {
        format!(
            r#""{name}": {{
              "category": "small",
              "source": {{"mode": "{mode}", "tla_path": "{}", "cfg_path": "{}"}},
              "tlc": {{"status": "pass", "states": null, "error_type": null}},
              "ty": {{"status": "pass", "states": null, "error_type": null}},
              "verified_match": true
            }}"#,
            tla_path.display(),
            cfg_path.display()
        )
    }

    fn config_free_tla(module: &str) -> String {
        format!(
            r#"---- MODULE {module} ----
\* Config-free checking: --no-config --init --next --inv.
VARIABLE counter
MyInit == counter = 0
MyNext == counter' = IF counter < 2 THEN counter + 1 ELSE counter
TypeOK == counter \in {{0, 1, 2}}
====
"#
        )
    }

    fn runtime_evidence_row(
        spec: &str,
        tlc_seconds: f64,
        ty_seconds: f64,
        states: u64,
    ) -> RuntimeEvidenceRow {
        RuntimeEvidenceRow {
            spec: spec.to_string(),
            tlc: runtime_mode_evidence("pass", Some(tlc_seconds), Some(states), None, spec, "tlc"),
            ty: runtime_mode_evidence("pass", Some(ty_seconds), Some(states), None, spec, "ty"),
            verified_match: true,
            refreshed: true,
            note: None,
            required_flags: Vec::new(),
        }
    }

    fn runtime_mode_evidence(
        status: &str,
        runtime_seconds: Option<f64>,
        states: Option<u64>,
        error_type: Option<&str>,
        spec: &str,
        mode: &str,
    ) -> RuntimeModeEvidence {
        RuntimeModeEvidence {
            status: status.to_string(),
            runtime_seconds,
            states,
            error_type: error_type.map(str::to_string),
            artifact_dir: PathBuf::from(format!("{spec}/{mode}")),
            ..RuntimeModeEvidence::default()
        }
    }

    #[test]
    fn runtime_evidence_reports_uncollected_selected_specs_after_interruption() {
        let rows = vec![
            runtime_evidence_row("A", 2.0, 1.0, 3),
            runtime_evidence_row("C", 2.0, 1.0, 3),
        ];
        let selected = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        assert_eq!(
            uncollected_selected_runtime_specs(&selected, &rows),
            vec!["B".to_string()]
        );
    }

    fn runtime_baseline_provenance() -> RuntimeBaselineProvenance {
        RuntimeBaselineProvenance {
            timestamp: "2026-04-28T20:30:00Z".to_string(),
            ty_git_commit: "abc1234".to_string(),
            ty_binary: RuntimeFileProvenance {
                path: PathBuf::from("/tmp/ty"),
                sha256: "0123456789abcdef".to_string(),
            },
            allow_debug_runtime: false,
        }
    }

    fn promotion_ready_baseline() -> Value {
        let mut baseline = json!({
            "schema_version": 3,
            "total_specs": 2,
            "specs_jcs_sha256": "",
            "categories": {
                "small": 1,
                "xlarge": 1
            },
            "stats": {
                "ty_fail": 0,
                "ty_match": 1,
                "ty_mismatch": 0,
                "ty_untested": 1,
                "tlc_error": 0,
                "tlc_pass": 1,
                "tlc_timeout": 1
            },
            "specs": {
                "RuntimeSpec": {
                    "category": "small",
                    "source": {},
                    "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 10, "error_type": null},
                    "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 10, "error_type": null},
                    "verified_match": true
                },
                "TimeoutSpec": {
                    "category": "xlarge",
                    "source": {},
                    "tlc": {"status": "timeout", "states": null, "error_type": "timeout"},
                    "ty": {"status": "untested", "states": null, "error_type": null},
                    "verified_match": false
                }
            }
        });
        let digest = sha256_jcs_value(&baseline["specs"]).unwrap();
        baseline["specs_jcs_sha256"] = json!(digest);
        baseline
    }

    fn matrix_anti_overfit_fixture(
        source: &str,
        baseline_text: &str,
    ) -> (tempfile::TempDir, SupremacyMatrixArgs) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::create_dir_all(dir.path().join("crates/tla-check/src")).unwrap();
        fs::create_dir_all(dir.path().join("crates/tla-trust-cg/src")).unwrap();
        fs::create_dir_all(dir.path().join("tests/tlc_comparison")).unwrap();
        fs::write(dir.path().join("crates/tla-check/src/runtime.rs"), source).unwrap();

        let policy_path = dir
            .path()
            .join("tests/tlc_comparison/single_thread_supremacy_gate.json");
        fs::write(
            &policy_path,
            serde_json::to_string(&json!({
                "specs": ["LaunchSpec"],
                "expected_state_counts": {"LaunchSpec": 100000},
                "expected_generated_state_counts": {"LaunchSpec": 200000}
            }))
            .unwrap(),
        )
        .unwrap();
        let baseline_path = dir.path().join("tests/tlc_comparison/spec_baseline.json");
        fs::write(&baseline_path, baseline_text).unwrap();

        let args = SupremacyMatrixArgs {
            baseline: baseline_path,
            policy: Some(policy_path),
            mode: SupremacyMode::Enforce,
            format: crate::cli_schema::SupremacyOutputFormat::Human,
            refresh_runtime: false,
            runtime_scope: SupremacyMatrixRuntimeScope::MissingRuntime,
            runtime_output_dir: None,
            runtime_limit: None,
            runtime_specs: Vec::new(),
            runtime_timeout: 300,
            production_runtime: true,
            runtime_ty_bin: None,
            allow_debug_runtime: false,
            runtime_tlc_jar: None,
            runtime_community_modules: None,
            runtime_tla_library: None,
        };

        (dir, args)
    }

    #[test]
    fn json_contract_classifies_python_gate_retirement_buckets() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "pass_spec": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "error_type": null},
                  "verified_match": true
                },
                "perf_loser_spec": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 2.0, "error_type": null},
                  "verified_match": true
                },
                "missing_runtime_spec": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "error_type": null},
                  "ty": {"status": "pass", "error_type": null},
                  "verified_match": true
                },
                "unsupported_spec": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "unsupported", "error_type": "unsupported operator"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "error_type": null},
                  "verified_match": false
                },
                "tlc_error_spec": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "error", "error_type": "TLCError"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "error_type": null},
                  "verified_match": false
                },
                "tlc_timeout_spec": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "timeout", "error_type": "timeout"},
                  "ty": {"status": "untested", "error_type": null},
                  "verified_match": false
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(summary.schema, MATRIX_SUMMARY_SCHEMA);
        assert_eq!(summary.verdict, SupremacyMatrixVerdict::Fail);
        assert!(!summary.strict_pass);
        assert_eq!(summary.strict_blockers, 5);
        assert_eq!(summary.strict_blocker_count(), 5);
        assert_eq!(class_for(&summary, "pass_spec"), SupremacyMatrixClass::Pass);
        assert_eq!(
            class_for(&summary, "perf_loser_spec"),
            SupremacyMatrixClass::PerfLoser
        );
        assert_eq!(
            class_for(&summary, "missing_runtime_spec"),
            SupremacyMatrixClass::MissingRuntime
        );
        assert_eq!(
            class_for(&summary, "unsupported_spec"),
            SupremacyMatrixClass::Unsupported
        );
        assert_eq!(
            class_for(&summary, "tlc_error_spec"),
            SupremacyMatrixClass::TlcError
        );
        assert_eq!(
            class_for(&summary, "tlc_timeout_spec"),
            SupremacyMatrixClass::TlcTimeout
        );

        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["schema"], MATRIX_SUMMARY_SCHEMA);
        assert_eq!(json["verdict"], "fail");
        assert_eq!(json["strict_pass"], false);
        assert_eq!(json["strict_blockers"], 5);
        assert_eq!(json["counts"]["perf_loser"], 1);
    }

    #[test]
    fn tlc_not_supported_markers_are_unsupported_not_tlc_errors() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "tlc_not_supported_error": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "error", "runtime_seconds": 2.0, "states": null, "error_type": "not supported by TLC"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": null, "error_type": null},
                  "verified_match": false
                },
                "tlc_not_runnable_status": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "not-runnable", "states": null, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": null, "error_type": null},
                  "verified_match": false
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(summary.counts.unsupported, 2);
        assert_eq!(summary.counts.tlc_error, 0);
        assert_eq!(
            class_for(&summary, "tlc_not_supported_error"),
            SupremacyMatrixClass::Unsupported
        );
        assert_eq!(
            class_for(&summary, "tlc_not_runnable_status"),
            SupremacyMatrixClass::Unsupported
        );
        assert!(reason_for(&summary, "tlc_not_supported_error").contains("TLC-impossible"));
        assert!(reason_for(&summary, "tlc_not_supported_error")
            .contains("error_type=not supported by TLC"));
        assert!(reason_for(&summary, "tlc_not_runnable_status").contains("status=not-runnable"));
        let selected = selected_runtime_rows(
            &summary,
            &[],
            None,
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap();
        assert!(selected.is_empty());
    }

    #[test]
    fn ty_unsupported_for_tlc_runnable_rows_is_parity_fail() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "ty_status_error": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
                  "ty": {"status": "error", "states": null, "error_type": "unsupported operator"},
                  "verified_match": false
                },
                "ty_pass_with_error": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": "not supported by trust-cg"},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(summary.counts.unsupported, 0);
        assert_eq!(summary.counts.parity_fail, 2);
        assert_eq!(
            class_for(&summary, "ty_status_error"),
            SupremacyMatrixClass::ParityFail
        );
        assert_eq!(
            class_for(&summary, "ty_pass_with_error"),
            SupremacyMatrixClass::ParityFail
        );
        let selected = selected_runtime_rows(
            &summary,
            &[],
            None,
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|row| row.spec.as_str())
                .collect::<Vec<_>>(),
            vec!["ty_pass_with_error", "ty_status_error"]
        );
    }

    #[test]
    fn comparable_tlc_outcomes_remain_strict_without_policy_opt_in() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "error_runtime": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "timeout_dominated": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "timeout", "runtime_seconds": 60.0, "states": null, "error_type": "timeout"},
                  "ty": {"status": "pass", "runtime_seconds": 5.0, "states": 100, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "error_runtime"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "timeout_dominated"),
            SupremacyMatrixClass::TlcTimeout
        );
        assert_eq!(summary.verdict, SupremacyMatrixVerdict::Fail);
        assert_eq!(summary.strict_blockers, 1);
        assert_eq!(summary.enforce_blocker_count(), 1);
        assert!(summary.policy.is_none());
    }

    #[test]
    fn matrix_policy_runtime_to_error_opt_in_does_not_forgive_timeout_dominance() {
        let counts = SupremacyMatrixCounts {
            runtime_to_error: 1,
            timeout_dominance: 1,
            ..SupremacyMatrixCounts::default()
        };
        let policy = MatrixPolicy {
            allow_runtime_to_error: true,
            allow_timeout_dominance: false,
        };

        let policy_summary = SupremacyMatrixPolicySummary::from_counts(&counts, &policy);

        assert_eq!(policy_summary.comparable_outcomes, 2);
        assert_eq!(policy_summary.blockers, 1);
        assert_eq!(policy_summary.verdict, SupremacyMatrixVerdict::Fail);
    }

    #[test]
    fn matrix_policy_timeout_dominance_opt_in_does_not_forgive_runtime_to_error() {
        let counts = SupremacyMatrixCounts {
            runtime_to_error: 1,
            timeout_dominance: 1,
            ..SupremacyMatrixCounts::default()
        };
        let policy = MatrixPolicy {
            allow_runtime_to_error: false,
            allow_timeout_dominance: true,
        };

        let policy_summary = SupremacyMatrixPolicySummary::from_counts(&counts, &policy);

        assert_eq!(policy_summary.comparable_outcomes, 2);
        assert_eq!(policy_summary.blockers, 1);
        assert_eq!(policy_summary.verdict, SupremacyMatrixVerdict::Fail);
    }

    #[test]
    fn matrix_policy_promotes_only_supported_comparable_tlc_outcomes() {
        let policy = MatrixPolicy {
            allow_runtime_to_error: true,
            allow_timeout_dominance: true,
        };
        let summary = classify_baseline_str_with_policy(
            r#"{
              "specs": {
                "error_runtime": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "timeout_dominated": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "timeout", "runtime_seconds": 60.0, "states": null, "error_type": "timeout"},
                  "ty": {"status": "pass", "runtime_seconds": 5.0, "states": 100, "error_type": null},
                  "verified_match": true
                },
                "unsupported_error": {
                  "source": {"mode": "template"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                }
              }
            }"#,
            &policy,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "error_runtime"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "timeout_dominated"),
            SupremacyMatrixClass::TimeoutDominance
        );
        assert_eq!(
            class_for(&summary, "unsupported_error"),
            SupremacyMatrixClass::Unsupported
        );
        assert_eq!(summary.counts.runtime_to_error, 0);
        assert_eq!(summary.counts.timeout_dominance, 1);
        assert_eq!(summary.strict_blockers, 2);
        assert!(!summary.strict_pass);
        let policy_summary = summary.policy.as_ref().expect("policy summary");
        assert_eq!(policy_summary.comparable_outcomes, 1);
        assert_eq!(policy_summary.blockers, 1);
        assert_eq!(summary.enforce_blocker_count(), 1);
        assert_eq!(summary.verdict, SupremacyMatrixVerdict::Fail);
    }

    #[test]
    fn missing_runtime_diagnostics_explain_strict_blockers_and_refresh_path() {
        let policy = MatrixPolicy {
            allow_runtime_to_error: true,
            allow_timeout_dominance: true,
        };
        let summary = classify_baseline_str_with_policy(
            r#"{
              "specs": {
                "missing_both": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "states": 1, "error_type": null},
                  "ty": {"status": "pass", "states": 1, "error_type": null},
                  "verified_match": true
                },
                "missing_ty": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 2, "error_type": null},
                  "ty": {"status": "pass", "states": 2, "error_type": null},
                  "verified_match": true
                },
                "missing_tlc": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "verified_match": true
                },
                "expected_violation_match": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 4, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 4, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "unsupported": {
                  "source": {"mode": "template"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 5, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 5, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
            &policy,
        )
        .unwrap();

        assert_eq!(summary.counts.missing_runtime, 3);
        assert_eq!(summary.counts.runtime_to_error, 0);
        assert_eq!(summary.counts.unsupported, 1);
        assert_eq!(summary.next_action_counts.get("refresh_runtime"), Some(&3));
        assert_eq!(
            summary.next_action_counts.get("triage_unsupported"),
            Some(&1)
        );
        assert_eq!(summary.strict_blockers, 4);
        assert_eq!(summary.enforce_blocker_count(), 4);
        assert_eq!(summary.verdict, SupremacyMatrixVerdict::Fail);
        assert_eq!(summary.policy.as_ref().unwrap().comparable_outcomes, 0);
        assert_eq!(summary.policy.as_ref().unwrap().blockers, 4);
        assert!(row_for(&summary, "missing_both")
            .reason
            .contains("TLC and TY runtime_seconds"));
        assert!(row_for(&summary, "missing_ty")
            .reason
            .contains("TY runtime_seconds"));
        assert!(row_for(&summary, "missing_tlc")
            .reason
            .contains("TLC runtime_seconds"));

        let diagnostics = summary
            .missing_runtime_diagnostics
            .as_ref()
            .expect("missing runtime diagnostics");
        // Order-independent: `specs_needing_measurement` is derived from row
        // iteration, so assert set-equality (sort both) rather than a brittle,
        // stale element order.
        let mut needing = diagnostics.specs_needing_measurement.clone();
        needing.sort();
        assert_eq!(
            needing,
            vec![
                "missing_both".to_string(),
                "missing_tlc".to_string(),
                "missing_ty".to_string()
            ]
        );
        assert!(diagnostics
            .meaning
            .contains("finite positive runtime_seconds"));
        assert!(diagnostics.launch_gate_policy.contains("not a win"));
        assert_eq!(
            diagnostics.refresh_command,
            missing_runtime_refresh_command(&diagnostics.specs_needing_measurement_details)
        );
        assert_eq!(
            diagnostics.missing_tlc_runtime_specs,
            vec!["missing_both".to_string(), "missing_tlc".to_string()]
        );
        assert_eq!(
            diagnostics.missing_ty_runtime_specs,
            vec!["missing_both".to_string(), "missing_ty".to_string()]
        );
        // Order-independent (sorted by spec name): set-equality, not a stale order.
        let mut details = diagnostics
            .specs_needing_measurement_details
            .iter()
            .map(|detail| {
                (
                    detail.spec.as_str(),
                    detail.missing_tlc_runtime,
                    detail.missing_ty_runtime,
                )
            })
            .collect::<Vec<_>>();
        details.sort();
        assert_eq!(
            details,
            vec![
                ("missing_both", true, true),
                ("missing_tlc", true, false),
                ("missing_ty", false, true)
            ]
        );

        let human = summary.to_human();
        assert!(human.contains("next_actions: refresh_runtime=3 triage_unsupported=1"));
        assert!(human.contains("- missing_both: MissingRuntime: next_action=refresh_runtime:"));
        assert!(human.contains("missing_runtime_meaning: missing_runtime means"));
        assert!(human.contains("missing_runtime_specs: missing_both, missing_tlc, missing_ty"));
        assert!(human.contains("missing_runtime_tlc_specs: missing_both, missing_tlc"));
        assert!(human.contains("missing_runtime_ty_specs: missing_both, missing_ty"));
        assert!(human.contains(
            "--runtime-spec missing_both --runtime-spec missing_tlc --runtime-spec missing_ty"
        ));
        assert!(human.contains("ty supremacy matrix --baseline <baseline.json>"));

        let markdown = summary.to_markdown();
        assert!(markdown.contains("## Next Actions"));
        assert!(markdown.contains("| refresh_runtime | 3 |"));
        assert!(markdown.contains("| triage_unsupported | 1 |"));
        assert!(markdown.contains("## Missing Runtime"));
        assert!(markdown.contains(
            "| Spec needing measurement | Missing TLC runtime | Missing TY runtime | Reason |"
        ));
        assert!(markdown.contains("| missing_both | true | true |"));
        assert!(markdown.contains("Refresh with the Rust CLI"));

        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(
            json["missing_runtime_diagnostics"]["specs_needing_measurement"],
            json!(["missing_both", "missing_tlc", "missing_ty"])
        );
        assert_eq!(
            json["missing_runtime_diagnostics"]["missing_tlc_runtime_specs"],
            json!(["missing_both", "missing_tlc"])
        );
        assert_eq!(
            json["missing_runtime_diagnostics"]["missing_ty_runtime_specs"],
            json!(["missing_both", "missing_ty"])
        );
        // Details are emitted in sorted (BTreeMap) order: index 2 is "missing_ty".
        assert_eq!(
            json["missing_runtime_diagnostics"]["specs_needing_measurement_details"][2]
                ["missing_ty_runtime"],
            json!(true)
        );
        assert_eq!(
            json["missing_runtime_diagnostics"]["refresh_command"],
            json!(diagnostics.refresh_command.clone())
        );
        assert_eq!(json["next_action_counts"]["refresh_runtime"], json!(3));
        // Locate one of the missing-runtime rows by spec name (rows are emitted
        // in BTreeMap order, not classification order).
        let rows = json["rows"].as_array().expect("rows array");
        let missing_both_row = rows
            .iter()
            .find(|row| row["spec"] == json!("missing_both"))
            .expect("missing_both row in JSON");
        assert_eq!(missing_both_row["next_action"], json!("refresh_runtime"));
    }

    #[test]
    fn non_pass_rows_carry_next_action_classifications() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "missing_runtime": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "states": 1, "error_type": null},
                  "ty": {"status": "pass", "states": 1, "error_type": null},
                  "verified_match": true
                },
                "parity_fail": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 2, "error_type": null},
                  "ty": {"status": "fail", "runtime_seconds": 1.0, "states": 3, "error_type": "mismatch"},
                  "verified_match": false
                },
                "perf_tie": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.005, "states": 3, "error_type": null},
                  "verified_match": true
                },
                "perf_loser": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 4, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.5, "states": 4, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            row_for(&summary, "missing_runtime").next_action,
            SupremacyMatrixNextAction::RefreshRuntime
        );
        assert_eq!(
            row_for(&summary, "parity_fail").next_action,
            SupremacyMatrixNextAction::FixParity
        );
        assert_eq!(
            row_for(&summary, "perf_tie").next_action,
            SupremacyMatrixNextAction::RemeasurePerfTie
        );
        assert_eq!(
            row_for(&summary, "perf_loser").next_action,
            SupremacyMatrixNextAction::FixPerfRegression
        );
        assert_eq!(summary.next_action_counts["refresh_runtime"], 1);
        assert_eq!(summary.next_action_counts["fix_parity"], 1);
        assert_eq!(summary.next_action_counts["remeasure_perf_tie"], 1);
        assert_eq!(summary.next_action_counts["fix_perf_regression"], 1);
    }

    #[test]
    fn matrix_policy_can_pass_on_comparable_rows_without_strict_win() {
        let policy = MatrixPolicy {
            allow_runtime_to_error: true,
            allow_timeout_dominance: true,
        };
        let summary = classify_baseline_str_with_policy(
            r#"{
              "specs": {
                "error_runtime": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "timeout_dominated": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "timeout", "runtime_seconds": 60.0, "states": null, "error_type": "timeout"},
                  "ty": {"status": "pass", "runtime_seconds": 5.0, "states": 100, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
            &policy,
        )
        .unwrap();

        assert_eq!(summary.strict_blockers, 1);
        assert!(!summary.strict_pass);
        assert_eq!(summary.enforce_blocker_count(), 0);
        assert_eq!(summary.verdict, SupremacyMatrixVerdict::Pass);
        assert_eq!(
            summary.policy.as_ref().unwrap().verdict,
            SupremacyMatrixVerdict::Pass
        );
    }

    #[test]
    fn matrix_policy_requires_comparable_evidence_before_promotion() {
        let policy = MatrixPolicy {
            allow_runtime_to_error: true,
            allow_timeout_dominance: true,
        };
        let summary = classify_baseline_str_with_policy(
            r#"{
              "specs": {
                "slower_error": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 4.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "unknown_error": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "unknown"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": null},
                  "verified_match": true
                },
                "unverified_timeout": {
                  "source": {"mode": "check"},
                  "tlc": {"status": "timeout", "runtime_seconds": 60.0, "states": null, "error_type": "timeout"},
                  "ty": {"status": "pass", "runtime_seconds": 5.0, "states": 100, "error_type": null},
                  "verified_match": false
                }
              }
            }"#,
            &policy,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "slower_error"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "unknown_error"),
            SupremacyMatrixClass::TlcError
        );
        assert_eq!(
            class_for(&summary, "unverified_timeout"),
            SupremacyMatrixClass::TlcTimeout
        );
        assert_eq!(summary.counts.runtime_to_error, 0);
        assert_eq!(summary.counts.timeout_dominance, 0);
        assert_eq!(summary.enforce_blocker_count(), 2);
        assert_eq!(summary.verdict, SupremacyMatrixVerdict::Fail);
    }

    #[test]
    fn matrix_enforce_blocks_on_gate_policy_anti_overfit_findings() {
        let (_dir, args) = matrix_anti_overfit_fixture(
            r#"const BAD: &str = "LaunchSpec";"#,
            r#"{"rows":[{"spec":"BaselineOnly"}]}"#,
        );

        let err = run_matrix_anti_overfit_scan(&args)
            .expect_err("enforce matrix must block policy corpus references");

        assert!(err
            .to_string()
            .contains("ty supremacy matrix anti-overfit scan found 1 forbidden corpus references"));
    }

    #[test]
    fn matrix_enforce_blocks_on_baseline_corpus_anti_overfit_findings() {
        let (_dir, args) = matrix_anti_overfit_fixture(
            r#"const BAD: &str = "BaselineOnly";"#,
            r#"{"rows":[{"spec":"BaselineOnly"}]}"#,
        );

        let err = run_matrix_anti_overfit_scan(&args)
            .expect_err("enforce matrix must block baseline corpus references");

        assert!(err
            .to_string()
            .contains("ty supremacy matrix anti-overfit scan found 1 forbidden corpus references"));
    }

    #[test]
    fn matrix_warn_keeps_anti_overfit_findings_non_blocking() {
        let (_dir, mut args) = matrix_anti_overfit_fixture(
            r#"const BAD: &str = "LaunchSpec";"#,
            r#"{"rows":[{"spec":"BaselineOnly"}]}"#,
        );
        args.mode = SupremacyMode::Warn;

        run_matrix_anti_overfit_scan(&args).unwrap();
    }

    #[test]
    fn matrix_warn_without_explicit_policy_skips_default_anti_overfit_scan() {
        let (_dir, mut args) = matrix_anti_overfit_fixture(
            r#"const BAD: &str = "BaselineOnly";"#,
            r#"{"rows":[{"spec":"BaselineOnly"}]}"#,
        );
        args.mode = SupremacyMode::Warn;
        args.policy = None;
        fs::remove_file(&args.baseline).unwrap();

        run_matrix_anti_overfit_scan(&args).unwrap();
    }

    #[test]
    fn matrix_warn_keeps_anti_overfit_scan_errors_non_blocking() {
        let (_dir, mut args) = matrix_anti_overfit_fixture(
            r#"const SAFE: &str = "structural";"#,
            r#"{"rows":[{"spec":"BaselineOnly"}]}"#,
        );
        args.mode = SupremacyMode::Warn;
        fs::remove_file(&args.baseline).unwrap();

        run_matrix_anti_overfit_scan(&args).unwrap();
    }

    #[test]
    fn matrix_warn_keeps_matrix_only_policy_non_blocking() {
        let (dir, mut args) = matrix_anti_overfit_fixture(
            r#"const SAFE: &str = "structural";"#,
            r#"{"rows":[{"spec":"BaselineOnly"}]}"#,
        );
        let matrix_only_policy = dir.path().join("matrix-only-policy.json");
        fs::write(
            &matrix_only_policy,
            r#"{"matrix_policy":{"allow_runtime_to_error":true}}"#,
        )
        .unwrap();
        args.mode = SupremacyMode::Warn;
        args.policy = Some(matrix_only_policy);

        run_matrix_anti_overfit_scan(&args).unwrap();
    }

    #[test]
    fn matrix_enforce_uses_default_anti_overfit_policy_for_matrix_only_policy() {
        let (dir, mut args) = matrix_anti_overfit_fixture(
            r#"const BAD: &str = "LaunchSpec";"#,
            r#"{"rows":[{"spec":"BaselineOnly"}]}"#,
        );
        let matrix_only_policy = dir.path().join("matrix-only-policy.json");
        fs::write(
            &matrix_only_policy,
            r#"{"matrix_policy":{"allow_runtime_to_error":true}}"#,
        )
        .unwrap();
        args.policy = Some(matrix_only_policy);

        let err = run_matrix_anti_overfit_scan(&args)
            .expect_err("enforce matrix must still scan the default anti-overfit policy");

        assert!(err
            .to_string()
            .contains("ty supremacy matrix anti-overfit scan found 1 forbidden corpus references"));
    }

    #[test]
    fn matrix_enforce_rejects_allow_debug_runtime() {
        let err = validate_matrix_runtime_refresh_policy(SupremacyMode::Enforce, true)
            .expect_err("enforce mode must reject debug runtime refresh evidence");

        assert!(err.to_string().contains("--allow-debug-runtime"));
        validate_matrix_runtime_refresh_policy(SupremacyMode::Warn, true)
            .expect("warn mode may use debug runtime smoke evidence");
    }

    #[test]
    fn matrix_enforce_rejects_debug_refreshed_baseline() {
        let baseline = json!({
            "schema_version": 3,
            "specs": {},
            "ty_refresh": {
                "allow_debug_runtime": true
            }
        });

        let err = validate_enforceable_baseline_value(&baseline)
            .expect_err("debug-refreshed baseline must not be enforceable");

        assert!(err.to_string().contains("--allow-debug-runtime"));
    }

    #[test]
    fn matrix_enforce_rejects_warning_refreshed_baseline() {
        let baseline = json!({
            "specs": {},
            RUNTIME_METADATA_WARNING_FIELD: {
                "promotion_ready": false
            }
        });

        let err = validate_enforceable_baseline_value(&baseline)
            .expect_err("warning-refreshed baseline must not be enforceable");

        assert!(err.to_string().contains(RUNTIME_METADATA_WARNING_FIELD));
    }

    #[test]
    fn matrix_enforce_accepts_fresh_promotion_metadata() {
        validate_enforceable_baseline_value(&promotion_ready_baseline())
            .expect("fresh baseline promotion metadata should be enforceable");
    }

    #[test]
    fn matrix_enforce_rejects_stale_promotion_metadata() {
        let mut baseline = promotion_ready_baseline();
        baseline["total_specs"] = json!(1);
        baseline["categories"]["small"] = json!(99);
        baseline["stats"]["ty_match"] = json!(0);
        baseline["specs_jcs_sha256"] = json!("stale");

        let err = validate_enforceable_baseline_value(&baseline)
            .expect_err("stale promotion metadata must not be enforceable");
        let message = err
            .chain()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            message.contains("matrix baseline promotion metadata is stale"),
            "{message}"
        );
        assert!(message.contains("total_specs"), "{message}");
        assert!(message.contains("categories"), "{message}");
        assert!(message.contains("stats"), "{message}");
        assert!(message.contains("specs_jcs_sha256"), "{message}");
    }

    #[test]
    fn matrix_classification_recomputes_corpus_digest_from_specs() {
        let mut baseline = promotion_ready_baseline();
        baseline["specs_jcs_sha256"] = json!("stale");
        let expected_digest = sha256_jcs_value(&baseline["specs"]).expect("digest should compute");

        let summary = classify_baseline_value(baseline).unwrap();

        assert_eq!(summary.corpus.total_specs, 2);
        assert_eq!(
            summary.corpus.specs_jcs_sha256.as_deref(),
            Some(expected_digest.as_str())
        );
    }

    #[test]
    fn matrix_enforceable_corpus_identity_rejects_stale_promotion_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let mut baseline = promotion_ready_baseline();
        baseline["specs_jcs_sha256"] = json!("stale");

        let validation_err = validate_enforceable_baseline_value(&baseline)
            .expect_err("stale metadata should reject enforceable baseline validation");
        let validation_message = validation_err.to_string();
        assert!(
            validation_message.contains("matrix baseline promotion metadata is stale")
                && validation_message.contains("specs_jcs_sha256"),
            "{validation_message}"
        );

        let path = dir.path().join("baseline.json");
        fs::write(&path, serde_json::to_string(&baseline).unwrap()).unwrap();
        let path_err = enforceable_baseline_corpus_identity_path(&path)
            .expect_err("stale metadata should reject enforceable corpus identity");
        let path_message = path_err.to_string();

        assert!(
            path_message.contains("validate enforceable baseline"),
            "{path_message}"
        );
    }

    #[test]
    fn runtime_file_provenance_hashes_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tytools.jar");
        fs::write(&file, b"tlc jar bytes").unwrap();

        let provenance = RuntimeFileProvenance::new(&file).unwrap();

        assert_eq!(provenance.path, file.canonicalize().unwrap());
        assert_eq!(provenance.sha256, sha256_bytes(b"tlc jar bytes"));
    }

    #[test]
    fn refresh_runtime_rejects_debug_ty_binary_unless_allowed() {
        let debug_bin = Path::new("target/user/debug/ty");
        let release_bin = Path::new("target/user/release/ty");

        let err = validate_runtime_ty_binary_for_refresh(debug_bin, false)
            .expect_err("debug binary should not be promotable runtime evidence");
        assert!(err.to_string().contains("--allow-debug-runtime"), "{err:?}");

        validate_runtime_ty_binary_for_refresh(debug_bin, true)
            .expect("explicit debug override should allow development smoke evidence");
        validate_runtime_ty_binary_for_refresh(release_bin, false)
            .expect("release binary should be promotable runtime evidence");
    }

    #[test]
    fn refresh_runtime_preflight_command_forces_json_trust_cg_backend() {
        let argv = ty_no_config_preflight_argv(
            Path::new("target/user/release/ty"),
            Path::new("/tmp/SupremacyMatrixRuntimePreflight.tla"),
        );

        assert_eq!(
            argv,
            [
                "target/user/release/ty",
                "check",
                "/tmp/SupremacyMatrixRuntimePreflight.tla",
                "--no-config",
                "--init",
                "MyInit",
                "--next",
                "MyNext",
                "--inv",
                "TypeOK",
                "--workers",
                "1",
                "--force",
                "--output",
                "json",
                "--backend",
                "trust-cg"
            ]
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn refresh_runtime_preflight_detects_backend_unavailable_output() {
        assert!(runtime_ty_preflight_reports_backend_unavailable(
            r#"{"result":{"status":"backend_unavailable"}}"#,
            "",
        ));
        assert!(runtime_ty_preflight_reports_backend_unavailable(
            "",
            "error_type=backend_unavailable",
        ));
        assert!(!runtime_ty_preflight_reports_backend_unavailable(
            r#"{"result":{"status":"ok"}}"#,
            "",
        ));
    }

    #[cfg(unix)]
    #[test]
    fn refresh_runtime_preflight_rejects_backend_unavailable_binary() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("ty");
        fs::write(
            &bin,
            "#!/bin/sh\nprintf '%s\\n' '{\"result\":{\"status\":\"backend_unavailable\",\"error_type\":\"backend_unavailable\"},\"statistics\":{\"states_found\":0}}'\nexit 3\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&bin).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&bin, permissions).unwrap();

        let err = preflight_runtime_ty_trust_cg_for_refresh(
            &bin,
            dir.path(),
            dir.path(),
            5,
            &BTreeMap::new(),
        )
        .expect_err("backend_unavailable preflight must fail closed");
        let message = err.to_string();

        assert!(message.contains("backend_unavailable"), "{message}");
        assert!(
            message.contains("cargo build -p tla-cli --bin ty"),
            "{message}"
        );
        assert!(message.contains("--runtime-ty-bin"), "{message}");
    }

    #[test]
    fn refresh_runtime_blocks_debug_or_profile_output_without_override() {
        assert_eq!(
            ty_runtime_output_contamination(
                "Note: running an unoptimized debug build\n",
                "Model checking\n",
            )
            .as_deref(),
            Some("debug_build_runtime_evidence")
        );
        assert_eq!(
            ty_runtime_output_contamination("", "=== Enumeration Profile ===\n").as_deref(),
            Some("profile_runtime_evidence")
        );
        assert_eq!(
            ty_runtime_error(
                0,
                false,
                "",
                "=== Eval Profile ===\nStates found: 1\n",
                false,
            )
            .as_deref(),
            Some("profile_runtime_evidence")
        );
        assert_eq!(
            ty_runtime_error(
                0,
                false,
                "Error: Invariant TypeOK is violated.\n",
                "=== Eval Profile ===\nStates found: 1\n",
                false,
            )
            .as_deref(),
            Some("profile_runtime_evidence")
        );
        assert_eq!(
            ty_runtime_error(
                1,
                false,
                "Note: running an unoptimized debug build\n",
                "backend failed\n",
                false,
            )
            .as_deref(),
            Some("debug_build_runtime_evidence")
        );
        assert_eq!(
            ty_runtime_error(
                0,
                true,
                "Note: running an unoptimized debug build\n",
                "timeout\n",
                false,
            )
            .as_deref(),
            Some("debug_build_runtime_evidence")
        );
        assert_eq!(
            ty_runtime_error(
                0,
                false,
                "",
                "=== Eval Profile ===\nStates found: 1\n",
                true,
            ),
            None
        );
        assert_eq!(
            runtime_refresh_script(true),
            "ty supremacy matrix --refresh-runtime --allow-debug-runtime"
        );
    }

    #[test]
    fn java_version_provenance_captures_stderr_version_line() {
        let provenance = command_version_provenance(vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            "printf 'openjdk version \"21.0.2\"\\n' >&2".to_string(),
        ]);

        assert_eq!(provenance.status, Some(0));
        assert_eq!(
            provenance.version.as_deref(),
            Some("openjdk version \"21.0.2\"")
        );
        assert_eq!(
            provenance.output,
            vec!["openjdk version \"21.0.2\"".to_string()]
        );
        assert!(provenance.error.is_none());
    }

    #[test]
    fn examples_checkout_provenance_records_commit_and_dirty_state() {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .expect("git init should run");
        fs::write(dir.path().join("spec.tla"), "---- MODULE spec ----\n====\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "spec.tla"])
            .current_dir(dir.path())
            .output()
            .expect("git add should run");
        let commit = std::process::Command::new("git")
            .args([
                "-c",
                "user.name=Matrix Test",
                "-c",
                "user.email=matrix@example.com",
                "commit",
                "-m",
                "initial",
            ])
            .current_dir(dir.path())
            .output()
            .expect("git commit should run");
        assert!(
            commit.status.success(),
            "git commit failed: {}",
            String::from_utf8_lossy(&commit.stderr)
        );
        fs::write(
            dir.path().join("spec.tla"),
            "---- MODULE spec ----\nEXTENDS Naturals\n====\n",
        )
        .unwrap();

        let provenance = git_checkout_provenance(dir.path());

        assert_eq!(provenance.path, dir.path().canonicalize().unwrap());
        assert!(provenance.error.is_none(), "{:?}", provenance.error);
        assert_eq!(provenance.head.as_deref().map(str::len), Some(40));
        assert!(provenance
            .head_short
            .as_deref()
            .is_some_and(|head| !head.is_empty()));
        assert_eq!(provenance.is_dirty, Some(true));
        assert!(provenance.status_porcelain_sha256.is_some());
    }

    #[test]
    fn no_config_runtime_generates_tlc_config_and_ty_cli_flags() {
        let dir = tempfile::tempdir().unwrap();
        let spec_dir = Path::new("runtime").join("ConfigFreeCounter");
        let flags = matrix_refresh::no_config_cli_flags();

        let cfg_path = write_no_config_tlc_config(dir.path(), &spec_dir, &flags).unwrap();

        assert_eq!(
            cfg_path,
            dir.path()
                .join("runtime")
                .join("ConfigFreeCounter")
                .join("config_free.generated.cfg")
        );
        assert_eq!(
            fs::read_to_string(&cfg_path).unwrap(),
            "INIT MyInit\nNEXT MyNext\nINVARIANT TypeOK\n"
        );
        let argv = ty_no_config_runtime_argv(
            Path::new("target/user/ty"),
            Path::new("/tmp/ConfigFreeCounter.tla"),
            &flags,
        );
        assert_eq!(
            argv,
            [
                "target/user/ty",
                "check",
                "/tmp/ConfigFreeCounter.tla",
                "--no-config",
                "--init",
                "MyInit",
                "--next",
                "MyNext",
                "--inv",
                "TypeOK",
                "--workers",
                "1",
                "--force",
                "--backend",
                "trust-cg"
            ]
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
        );
        assert!(!argv.iter().any(|arg| arg == "--config"));
    }

    #[test]
    fn no_config_runtime_uses_metadata_cli_flags() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/AltNoConfig.tla"),
            r#"---- MODULE AltNoConfig ----
\* Config-free checking with metadata-provided flags.
VARIABLE counter
AltInit == counter = 0
AltNext == counter' = counter
AltOK == counter = 0
====
"#,
        );
        let text = runtime_refresh_baseline(
            r#""AltNoConfig": {
              "category": "small",
              "source": {
                "tla_path": "specs/AltNoConfig.tla",
                "required_flags": ["--no-config", "--init", "AltInit", "--next", "AltNext", "--inv", "AltOK"]
              },
              "tlc": {"status": "pass", "states": 1, "error_type": null},
              "ty": {"status": "pass", "states": 1, "error_type": null},
              "verified_match": true
            }"#,
            &examples_dir,
        );
        let plan = matrix_refresh::plan_missing_runtime_refresh_str(
            &text,
            Path::new("baseline.json"),
            None,
        )
        .unwrap();
        let planned_row = plan.row("AltNoConfig").unwrap();
        let flags = no_config_runtime_flags(Some(planned_row)).unwrap();

        let cfg_path =
            write_no_config_tlc_config(dir.path(), Path::new("runtime/AltNoConfig"), flags)
                .unwrap();
        assert_eq!(
            fs::read_to_string(&cfg_path).unwrap(),
            "INIT AltInit\nNEXT AltNext\nINVARIANT AltOK\n"
        );
        let argv = ty_no_config_runtime_argv(
            Path::new("target/user/ty"),
            Path::new("/tmp/AltNoConfig.tla"),
            flags,
        );
        assert!(argv
            .windows(2)
            .any(|window| window == ["--init", "AltInit"]));
        assert!(argv
            .windows(2)
            .any(|window| window == ["--next", "AltNext"]));
        assert!(argv.windows(2).any(|window| window == ["--inv", "AltOK"]));
        assert_eq!(
            no_config_runtime_required_flags(flags),
            [
                "--no-config",
                "--init",
                "AltInit",
                "--next",
                "AltNext",
                "--inv",
                "AltOK"
            ]
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn refresh_runtime_ty_env_uses_non_strict_trust_cg_overrides() {
        let base_env =
            matrix_runtime_refresh_base_env_with_compile_jobs(DEFAULT_RUNTIME_REFRESH_COMPILE_JOBS);
        let spec_dir = Path::new("reports/perf/runtime/SpecA");
        let env = ty_matrix_runtime_refresh_env(spec_dir, &base_env);
        let expected_cache_dir = spec_dir
            .join("trust_cg-artifact-cache")
            .display()
            .to_string();

        for (key, value) in [
            ("TY_trust_cg", "1"),
            ("TY_TRUST_CG_BFS", "1"),
            ("TY_TRUST_CG_EXISTS", "1"),
            ("TY_BYTECODE_VM", "1"),
            ("TY_BYTECODE_VM_STATS", "1"),
            (
                RUNTIME_REFRESH_COMPILE_JOBS_ENV,
                DEFAULT_RUNTIME_REFRESH_COMPILE_JOBS,
            ),
            ("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP", "1"),
            ("TY_DISABLE_ARTIFACT_CACHE", "1"),
        ] {
            assert_eq!(env.get(key).map(String::as_str), Some(value), "{key}");
        }
        assert_eq!(
            env.get("TY_CACHE_DIR").map(String::as_str),
            Some(expected_cache_dir.as_str())
        );
        assert!(!env.contains_key("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST"));
        assert!(!env.contains_key("TY_TRUST_CG_NATIVE_FUSED_STRICT"));
        assert!(!env.contains_key("TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP"));
        // Semantic reducer levers are CLI flags now (`--no-reduction` in the
        // count-verify argv); the env must not pin them.
        assert!(!env.contains_key("TY_AUTO_POR"));
        assert!(!env.contains_key("TY_AUTO_SYMMETRY"));
        assert!(!base_env.contains_key("TY_CACHE_DIR"));
    }

    #[test]
    fn refresh_runtime_compile_jobs_value_defaults_and_trims_override() {
        assert_eq!(
            runtime_refresh_compile_jobs_value_from_env(None),
            DEFAULT_RUNTIME_REFRESH_COMPILE_JOBS
        );
        assert_eq!(
            runtime_refresh_compile_jobs_value_from_env(Some(String::new())),
            DEFAULT_RUNTIME_REFRESH_COMPILE_JOBS
        );
        assert_eq!(
            runtime_refresh_compile_jobs_value_from_env(Some(" 4 ".to_string())),
            "4"
        );
    }

    #[test]
    fn refresh_runtime_ty_env_allows_compile_jobs_override() {
        let base_env = matrix_runtime_refresh_base_env_with_compile_jobs("4");
        let spec_dir = Path::new("reports/perf/runtime/SpecA");
        let env = ty_matrix_runtime_refresh_env(spec_dir, &base_env);

        assert_eq!(
            env.get(RUNTIME_REFRESH_COMPILE_JOBS_ENV)
                .map(String::as_str),
            Some("4")
        );
    }

    #[test]
    fn runtime_collection_config_uses_non_strict_matrix_refresh_env() {
        let dir = tempfile::tempdir().unwrap();
        let args = SupremacyMatrixArgs {
            baseline: PathBuf::from("tests/tlc_comparison/spec_baseline.json"),
            policy: None,
            mode: SupremacyMode::Warn,
            format: crate::cli_schema::SupremacyOutputFormat::Json,
            refresh_runtime: true,
            runtime_scope: SupremacyMatrixRuntimeScope::MissingRuntime,
            runtime_output_dir: Some(dir.path().join("runtime")),
            runtime_limit: Some(1),
            runtime_specs: Vec::new(),
            runtime_timeout: 30,
            production_runtime: true,
            runtime_ty_bin: Some(PathBuf::from("target/user/release/ty")),
            allow_debug_runtime: false,
            runtime_tlc_jar: Some(PathBuf::from("tlaplus/tytools.jar")),
            runtime_community_modules: None,
            runtime_tla_library: None,
        };

        let config = RuntimeCollectionConfig::from_args(&args, dir.path()).unwrap();

        assert_eq!(
            config.ty_base_env.get("TY_trust_cg").map(String::as_str),
            Some("1")
        );
        assert!(!config
            .ty_base_env
            .contains_key("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST"));
        assert!(!config
            .ty_base_env
            .contains_key("TY_TRUST_CG_NATIVE_FUSED_STRICT"));
    }

    #[test]
    fn runtime_tla_library_resolution_prefers_explicit_then_env_then_repo_then_home() {
        let dir = tempfile::tempdir().unwrap();
        let repo_root = dir.path().join("repo");
        let repo_library = repo_root.join(DEFAULT_TLA_LIBRARY);
        let home = dir.path().join("home");
        let home_library = home.join("tlapm/library");
        fs::create_dir_all(&repo_library).unwrap();
        fs::create_dir_all(&home_library).unwrap();

        let explicit = dir.path().join("explicit-library");
        let tla_library_env = dir.path().join("env-tla-library");
        let tla_plus_library_env = dir.path().join("env-tla-plus-library");

        assert_eq!(
            resolve_runtime_tla_library_from(
                Some(explicit.clone()),
                &repo_root,
                Some(tla_library_env.clone()),
                Some(tla_plus_library_env.clone()),
                Some(home.clone()),
            ),
            Some(explicit)
        );
        assert_eq!(
            resolve_runtime_tla_library_from(
                None,
                &repo_root,
                Some(tla_library_env.clone()),
                Some(tla_plus_library_env.clone()),
                Some(home.clone()),
            ),
            Some(tla_library_env)
        );
        assert_eq!(
            resolve_runtime_tla_library_from(
                None,
                &repo_root,
                None,
                Some(tla_plus_library_env.clone()),
                Some(home.clone()),
            ),
            Some(tla_plus_library_env)
        );
        assert_eq!(
            resolve_runtime_tla_library_from(None, &repo_root, None, None, Some(home.clone())),
            Some(repo_library.clone())
        );

        fs::remove_dir_all(&repo_library).unwrap();
        assert_eq!(
            resolve_runtime_tla_library_from(None, &repo_root, None, None, Some(home)),
            Some(home_library)
        );
    }

    #[test]
    fn refresh_runtime_preflight_uses_non_strict_env() {
        let base_env =
            matrix_runtime_refresh_base_env_with_compile_jobs(DEFAULT_RUNTIME_REFRESH_COMPILE_JOBS);
        let dir = tempfile::tempdir().unwrap();
        let repo_root = dir.path().join("repo");
        let output_dir = dir.path().join("out");
        let spec_path = output_dir
            .join("runtime-ty-trust_cg-preflight")
            .join("SupremacyMatrixRuntimePreflight.tla");
        let preflight_dir = output_dir.join("runtime-ty-trust_cg-preflight");

        let command = ty_runtime_preflight_command_spec(
            Path::new("target/user/release/ty"),
            &spec_path,
            &repo_root,
            &preflight_dir,
            300,
            &base_env,
        );

        assert_eq!(command.timeout_seconds, 10);
        assert_eq!(
            command
                .env_overrides
                .get(RUNTIME_REFRESH_COMPILE_JOBS_ENV)
                .map(String::as_str),
            Some(DEFAULT_RUNTIME_REFRESH_COMPILE_JOBS)
        );
        assert!(!command
            .env_overrides
            .contains_key("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST"));
        assert!(!command
            .env_overrides
            .contains_key("TY_TRUST_CG_NATIVE_FUSED_STRICT"));
        assert!(command
            .argv
            .windows(2)
            .any(|window| window == ["--backend", "trust-cg"]));
    }

    #[test]
    fn refresh_runtime_preflight_carries_compile_jobs_override() {
        let base_env = matrix_runtime_refresh_base_env_with_compile_jobs("8");
        let dir = tempfile::tempdir().unwrap();
        let repo_root = dir.path().join("repo");
        let output_dir = dir.path().join("out");
        let spec_path = output_dir
            .join("runtime-ty-trust_cg-preflight")
            .join("SupremacyMatrixRuntimePreflight.tla");
        let preflight_dir = output_dir.join("runtime-ty-trust_cg-preflight");

        let command = ty_runtime_preflight_command_spec(
            Path::new("target/user/release/ty"),
            &spec_path,
            &repo_root,
            &preflight_dir,
            300,
            &base_env,
        );

        assert_eq!(
            command
                .env_overrides
                .get(RUNTIME_REFRESH_COMPILE_JOBS_ENV)
                .map(String::as_str),
            Some("8")
        );
    }

    #[test]
    fn runtime_selection_defaults_to_missing_runtime_rows() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "missing_a": {
                  "source": {},
                  "tlc": {"status": "pass", "states": 5, "error_type": null},
                  "ty": {"status": "pass", "states": 5, "error_type": null},
                  "verified_match": true
                },
                "perf_loser": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 6, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.5, "states": 6, "error_type": null},
                  "verified_match": true
                },
                "missing_b": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 7, "error_type": null},
                  "ty": {"status": "pass", "states": 7, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        let selected = selected_runtime_rows(
            &summary,
            &[],
            Some(1),
            matrix_refresh::MatrixRefreshScope::MissingRuntime,
        )
        .unwrap();

        assert_eq!(
            selected
                .iter()
                .map(|row| row.spec.as_str())
                .collect::<Vec<_>>(),
            vec!["missing_a"]
        );
    }

    #[test]
    fn runtime_batch_selection_defaults_to_batchable_missing_runtime_rows() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        for spec in ["A", "B", "Blocked"] {
            write_file(
                &examples_dir.join(format!("specs/{spec}.tla")),
                &format!("---- MODULE {spec} ----\n====\n"),
            );
        }
        write_file(&examples_dir.join("specs/A.cfg"), "INIT Init\n");
        write_file(&examples_dir.join("specs/B.cfg"), "INIT Init\n");
        let text = runtime_refresh_baseline(
            &[
                missing_runtime_spec_json(
                    "A",
                    Path::new("specs/A.tla"),
                    Some(Path::new("specs/A.cfg")),
                ),
                missing_runtime_spec_json(
                    "B",
                    Path::new("specs/B.tla"),
                    Some(Path::new("specs/B.cfg")),
                ),
                missing_runtime_spec_json(
                    "Blocked",
                    Path::new("specs/Blocked.tla"),
                    Some(Path::new("specs/Blocked.cfg")),
                ),
            ]
            .join(","),
            &examples_dir,
        );
        let summary = classify_baseline_str(&text).unwrap();
        let plan = matrix_refresh::plan_missing_runtime_refresh_str(
            &text,
            Path::new("baseline.json"),
            None,
        )
        .unwrap();

        let selection = runtime_batch_selection(&summary, &plan, &[], None).unwrap();

        assert_eq!(selection.selected_specs, vec!["A", "B"]);
        assert_eq!(plan.blocked_runtime_specs, vec!["Blocked"]);
        assert!(selection
            .skipped_batchable_runtime_specs_by_limit
            .is_empty());

        let limited = runtime_batch_selection(&summary, &plan, &[], Some(1)).unwrap();
        assert_eq!(limited.selected_specs, vec!["A"]);
        assert_eq!(limited.skipped_batchable_runtime_specs_by_limit, vec!["B"]);

        let explicit_limited = runtime_batch_selection(
            &summary,
            &plan,
            &["A".to_string(), "B".to_string()],
            Some(1),
        )
        .unwrap();
        assert_eq!(explicit_limited.selected_specs, vec!["A"]);
        assert_eq!(
            explicit_limited.skipped_batchable_runtime_specs_by_limit,
            vec!["B"]
        );

        let err = runtime_batch_selection(
            &summary,
            &plan,
            &["A".to_string(), "Blocked".to_string()],
            Some(1),
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("--runtime-spec Blocked is not batchable"));

        let output_dir = dir.path().join("out");
        fs::create_dir_all(&output_dir).unwrap();
        let batch_plan_path = write_runtime_batch_plan(
            Path::new("baseline.json"),
            &output_dir,
            Some(1),
            &[],
            &limited,
            &plan,
        )
        .unwrap();
        let batch_plan_json: Value =
            serde_json::from_str(&fs::read_to_string(batch_plan_path).unwrap()).unwrap();

        assert_eq!(batch_plan_json["schema"], json!(RUNTIME_BATCH_PLAN_SCHEMA));
        assert_eq!(batch_plan_json["selected_runtime_specs"], json!(["A"]));
        assert_eq!(
            batch_plan_json["skipped_batchable_runtime_specs_by_limit"],
            json!(["B"])
        );
        assert_eq!(
            batch_plan_json["refresh_plan"]["blocked_runtime_specs"],
            json!(["Blocked"])
        );
    }

    #[test]
    fn runtime_batch_selection_hard_caps_simulation_rows_at_runtime_limit() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        for spec in ["A", "B", "Sim"] {
            write_file(
                &examples_dir.join(format!("specs/{spec}.tla")),
                &format!("---- MODULE {spec} ----\n====\n"),
            );
            write_file(
                &examples_dir.join(format!("specs/{spec}.cfg")),
                "INIT Init\n",
            );
        }
        let text = runtime_refresh_baseline(
            &[
                missing_runtime_spec_json(
                    "A",
                    Path::new("specs/A.tla"),
                    Some(Path::new("specs/A.cfg")),
                ),
                missing_runtime_spec_json(
                    "B",
                    Path::new("specs/B.tla"),
                    Some(Path::new("specs/B.cfg")),
                ),
                missing_runtime_simulation_spec_json(
                    "Sim",
                    "simulate",
                    Path::new("specs/Sim.tla"),
                    Path::new("specs/Sim.cfg"),
                ),
            ]
            .join(","),
            &examples_dir,
        );
        let summary = classify_baseline_str(&text).unwrap();
        let plan = matrix_refresh::plan_missing_runtime_refresh_str(
            &text,
            Path::new("baseline.json"),
            None,
        )
        .unwrap();

        let limited = runtime_batch_selection(&summary, &plan, &[], Some(1)).unwrap();

        assert_eq!(limited.selected_specs, vec!["Sim"]);
        assert_eq!(limited.selected_specs.len(), 1);
        assert_eq!(
            limited.skipped_batchable_runtime_specs_by_limit,
            vec!["A", "B"]
        );
    }

    #[test]
    fn no_config_runtime_detection_uses_structural_refresh_plan() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/ConfigFreeCounter.tla"),
            &config_free_tla("ConfigFreeCounter"),
        );
        let text = runtime_refresh_baseline(
            &missing_runtime_spec_json(
                "ConfigFreeCounter",
                Path::new("specs/ConfigFreeCounter.tla"),
                None,
            ),
            &examples_dir,
        );
        let summary = classify_baseline_str(&text).unwrap();
        let plan = matrix_refresh::plan_missing_runtime_refresh_str(
            &text,
            Path::new("baseline.json"),
            None,
        )
        .unwrap();
        let planned_row = plan.row("ConfigFreeCounter");

        let selection = runtime_batch_selection(&summary, &plan, &[], None).unwrap();

        assert_eq!(selection.selected_specs, vec!["ConfigFreeCounter"]);
        assert!(should_run_no_config_runtime(
            "ConfigFreeCounter",
            planned_row
        ));
        assert!(!should_run_no_config_runtime("Other", None));
    }

    #[test]
    fn runtime_batch_selection_full_suite_refreshes_all_batchable_runnable_rows() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        for spec in [
            "ExpectedViolation",
            "Missing",
            "Winner",
            "Tie",
            "Loser",
            "Blocked",
            "TyTimeout",
            "TlcError",
            "TlcTimeout",
            "Unsupported",
        ] {
            write_file(
                &examples_dir.join(format!("specs/{spec}.tla")),
                &format!("---- MODULE {spec} ----\n====\n"),
            );
        }
        for spec in [
            "ExpectedViolation",
            "Missing",
            "Winner",
            "Tie",
            "Loser",
            "TyTimeout",
            "TlcError",
            "TlcTimeout",
            "Unsupported",
        ] {
            write_file(
                &examples_dir.join(format!("specs/{spec}.cfg")),
                "INIT Init\n",
            );
        }
        let text = runtime_refresh_baseline(
            &[
                missing_runtime_spec_json(
                    "Missing",
                    Path::new("specs/Missing.tla"),
                    Some(Path::new("specs/Missing.cfg")),
                ),
                r#""Winner": {
                  "category": "small",
                  "source": {"tla_path": "specs/Winner.tla", "cfg_path": "specs/Winner.cfg"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "verified_match": true
                }"#
                .to_string(),
                r#""Tie": {
                  "category": "small",
                  "source": {"tla_path": "specs/Tie.tla", "cfg_path": "specs/Tie.cfg"},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "verified_match": true
                }"#
                .to_string(),
                r#""Loser": {
                  "category": "small",
                  "source": {"tla_path": "specs/Loser.tla", "cfg_path": "specs/Loser.cfg"},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
                  "verified_match": true
                }"#
                .to_string(),
                r#""ExpectedViolation": {
                  "category": "small",
                  "source": {"tla_path": "specs/ExpectedViolation.tla", "cfg_path": "specs/ExpectedViolation.cfg"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                }"#
                .to_string(),
                r#""TyTimeout": {
                  "category": "small",
                  "source": {"tla_path": "specs/TyTimeout.tla", "cfg_path": "specs/TyTimeout.cfg"},
                  "tlc": {"status": "pass", "runtime_seconds": 4.0, "states": 3, "error_type": null},
                  "ty": {"status": "timeout", "runtime_seconds": 300.0, "states": null, "error_type": "timeout"},
                  "verified_match": false
                }"#
                .to_string(),
                r#""TlcError": {
                  "category": "small",
                  "source": {"tla_path": "specs/TlcError.tla", "cfg_path": "specs/TlcError.cfg"},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": null, "error_type": "runtime_error"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": null, "error_type": null},
                  "verified_match": true
                }"#
                .to_string(),
                r#""TlcTimeout": {
                  "category": "small",
                  "source": {"tla_path": "specs/TlcTimeout.tla", "cfg_path": "specs/TlcTimeout.cfg"},
                  "tlc": {"status": "timeout", "runtime_seconds": 60.0, "states": null, "error_type": "timeout"},
                  "ty": {"status": "pass", "runtime_seconds": 5.0, "states": 100, "error_type": null},
                  "verified_match": true
                }"#
                .to_string(),
                r#""Blocked": {
                  "category": "small",
                  "source": {"tla_path": "specs/Blocked.tla", "cfg_path": "specs/Blocked.cfg"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "verified_match": true
                }"#
                .to_string(),
                r#""Unsupported": {
                  "category": "small",
                  "source": {"mode": "unsupported", "tla_path": "specs/Unsupported.tla", "cfg_path": "specs/Unsupported.cfg"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "verified_match": true
                }"#
                .to_string(),
            ]
            .join(","),
            &examples_dir,
        );
        let summary = classify_baseline_str(&text).unwrap();
        assert_eq!(class_for(&summary, "Winner"), SupremacyMatrixClass::Pass);
        assert_eq!(class_for(&summary, "Tie"), SupremacyMatrixClass::PerfTie);
        assert_eq!(
            class_for(&summary, "Loser"),
            SupremacyMatrixClass::PerfLoser
        );
        assert_eq!(
            class_for(&summary, "ExpectedViolation"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "TyTimeout"),
            SupremacyMatrixClass::TyTimeout
        );
        assert_eq!(
            class_for(&summary, "TlcError"),
            SupremacyMatrixClass::TlcError
        );
        assert_eq!(
            class_for(&summary, "TlcTimeout"),
            SupremacyMatrixClass::TlcTimeout
        );

        let plan = matrix_refresh::plan_runtime_refresh_str(
            &text,
            Path::new("baseline.json"),
            None,
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap();
        let selection = runtime_batch_selection(&summary, &plan, &[], None).unwrap();

        // Order-independent: the full selected set is rendered in a deterministic
        // (sorted) order; assert set-equality rather than a stale element order.
        let mut selected = selection.selected_specs.clone();
        selected.sort();
        assert_eq!(
            selected,
            vec![
                "ExpectedViolation",
                "Loser",
                "Missing",
                "Tie",
                "TlcError",
                "TlcTimeout",
                "TyTimeout",
                "Winner"
            ]
        );
        assert_eq!(plan.blocked_runtime_specs, vec!["Blocked"]);
        assert!(!selection
            .selected_specs
            .iter()
            .any(|spec| spec == "Unsupported"));

        let limited = runtime_batch_selection(&summary, &plan, &[], Some(2)).unwrap();
        assert_eq!(limited.selected_specs, vec!["Missing", "Loser"]);
        assert_eq!(
            limited.skipped_batchable_runtime_specs_by_limit,
            vec![
                "Tie",
                "TyTimeout",
                "ExpectedViolation",
                "TlcError",
                "TlcTimeout",
                "Winner"
            ]
        );
    }

    #[test]
    fn limited_all_runnable_selection_uses_source_order_before_spec_name() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        for (module, path) in [
            ("FirstBySource", "specs/001.tla"),
            ("SecondBySource", "specs/002.tla"),
        ] {
            write_file(
                &examples_dir.join(path),
                &format!("---- MODULE {module} ----\n====\n"),
            );
        }
        write_file(&examples_dir.join("specs/001.cfg"), "INIT Init\n");
        write_file(&examples_dir.join("specs/002.cfg"), "INIT Init\n");
        let text = runtime_refresh_baseline(
            r#""ASecondBySpecName": {
              "category": "small",
              "source": {"tla_path": "specs/002.tla", "cfg_path": "specs/002.cfg"},
              "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
              "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
              "verified_match": true
            },
            "ZFirstBySource": {
              "category": "small",
              "source": {"tla_path": "specs/001.tla", "cfg_path": "specs/001.cfg"},
              "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
              "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
              "verified_match": true
            }"#,
            &examples_dir,
        );
        let summary = classify_baseline_str(&text).unwrap();
        let plan = matrix_refresh::plan_runtime_refresh_str(
            &text,
            Path::new("baseline.json"),
            None,
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap();

        let full = runtime_batch_selection(&summary, &plan, &[], None).unwrap();
        assert_eq!(
            full.selected_specs,
            vec!["ASecondBySpecName", "ZFirstBySource"]
        );

        let limited = runtime_batch_selection(&summary, &plan, &[], Some(1)).unwrap();
        assert_eq!(limited.selected_specs, vec!["ZFirstBySource"]);
        assert_eq!(
            limited.skipped_batchable_runtime_specs_by_limit,
            vec!["ASecondBySpecName"]
        );
    }

    #[test]
    fn runtime_selection_allows_named_perf_blocker_rows() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "MCReachabilityTestAllGraphs": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 6, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 10.0, "states": 6, "error_type": null},
                  "verified_match": true
                },
                "near_equal": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 6, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.005, "states": 6, "error_type": null},
                  "verified_match": true
                },
                "already_pass": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 7, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 7, "error_type": null},
                  "verified_match": true
                },
                "unsupported_mode": {
                  "source": {"mode": "template"},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 7, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 7, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        let selected = selected_runtime_rows(
            &summary,
            &["MCReachabilityTestAllGraphs".to_string()],
            None,
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap();

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].spec, "MCReachabilityTestAllGraphs");
        assert_eq!(selected[0].class, SupremacyMatrixClass::PerfLoser);
        assert_eq!(
            selected[0].next_action,
            SupremacyMatrixNextAction::FixPerfRegression
        );
        let selected = selected_runtime_rows(
            &summary,
            &["near_equal".to_string()],
            None,
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].spec, "near_equal");
        assert_eq!(selected[0].class, SupremacyMatrixClass::PerfTie);
        assert_eq!(
            selected[0].next_action,
            SupremacyMatrixNextAction::RemeasurePerfTie
        );
        let selected = selected_runtime_rows(
            &summary,
            &["MCReachabilityTestAllGraphs".to_string()],
            Some(0),
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap();
        assert!(selected.is_empty());
        let err = selected_runtime_rows(
            &summary,
            &[
                "already_pass".to_string(),
                "missing_after_limit".to_string(),
            ],
            Some(1),
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("--runtime-spec missing_after_limit: no baseline row found"));
        let err = selected_runtime_rows(
            &summary,
            &["already_pass".to_string(), "unsupported_mode".to_string()],
            Some(1),
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported rows cannot be refreshed"));
        let selected = selected_runtime_rows(
            &summary,
            &["already_pass".to_string()],
            None,
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].class, SupremacyMatrixClass::Pass);
        let err = selected_runtime_rows(
            &summary,
            &["unsupported_mode".to_string()],
            None,
            matrix_refresh::MatrixRefreshScope::AllRunnable,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported rows cannot be refreshed"));
    }

    #[test]
    fn classifies_non_faster_ties_separately_from_perf_losers() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "strict_faster_inside_tolerance": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 1, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 0.999, "states": 1, "error_type": null},
                  "verified_match": true
                },
                "exact_equal": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 2, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 2, "error_type": null},
                  "verified_match": true
                },
                "near_equal_slow": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.005, "states": 3, "error_type": null},
                  "verified_match": true
                },
                "tiny_runtime_slow": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 0.020, "states": 4, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 0.045, "states": 4, "error_type": null},
                  "verified_match": true
                },
                "true_perf_loser": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 5, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.05, "states": 5, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "strict_faster_inside_tolerance"),
            SupremacyMatrixClass::Pass
        );
        assert_eq!(
            class_for(&summary, "exact_equal"),
            SupremacyMatrixClass::PerfTie
        );
        assert_eq!(
            class_for(&summary, "near_equal_slow"),
            SupremacyMatrixClass::PerfTie
        );
        assert_eq!(
            class_for(&summary, "tiny_runtime_slow"),
            SupremacyMatrixClass::PerfTie
        );
        assert_eq!(
            class_for(&summary, "true_perf_loser"),
            SupremacyMatrixClass::PerfLoser
        );
        assert_eq!(summary.counts.pass, 1);
        assert_eq!(summary.counts.perf_tie, 3);
        assert_eq!(summary.counts.perf_loser, 1);
        assert!(row_for(&summary, "near_equal_slow")
            .reason
            .contains("tie tolerance"));
        assert!(row_for(&summary, "tiny_runtime_slow")
            .reason
            .contains("tiny-runtime tie floor"));
        assert_eq!(row_for(&summary, "exact_equal").perf_loser_rank, None);
        assert_eq!(
            row_for(&summary, "true_perf_loser").perf_loser_rank,
            Some(1)
        );
    }

    #[test]
    fn classifies_expected_violation_matches_separately_from_tlc_errors() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "invariant_match": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "property_match": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "property"},
                  "ty": {"status": "error", "runtime_seconds": 1.0, "states": 12, "error_type": "property_violation"},
                  "verified_match": true
                },
                "liveness_match": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "liveness"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "liveness_violation"},
                  "verified_match": true
                },
                "assume_match": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 0, "error_type": "assume_violation"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 0, "error_type": "assume_violation"},
                  "verified_match": true
                },
                "deadlock_match": {
                  "source": {},
                  "tlc": {"status": "error", "runtime_seconds": 3.0, "states": 2, "error_type": "deadlock"},
                  "ty": {"status": "error", "runtime_seconds": 1.0, "states": 2, "error_type": "deadlock"},
                  "verified_match": true
                },
                "tool_error": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": null, "error_type": "runtime_error"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": null, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        for spec in [
            "invariant_match",
            "property_match",
            "liveness_match",
            "assume_match",
            "deadlock_match",
        ] {
            assert_eq!(
                class_for(&summary, spec),
                SupremacyMatrixClass::ExpectedViolationMatch
            );
            assert!(row_for(&summary, spec).reason.contains("expected"));
        }
        assert_eq!(
            class_for(&summary, "tool_error"),
            SupremacyMatrixClass::TlcError
        );
        assert_eq!(summary.counts.expected_violation_match, 5);
        assert_eq!(summary.counts.tlc_error, 1);
        assert_eq!(summary.strict_blockers, 1);
    }

    #[test]
    fn tlc_pass_status_overrides_stale_error_type_for_matrix_classification() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "warning_only_tlc": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 45, "error_type": "parse"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 45, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "warning_only_tlc"),
            SupremacyMatrixClass::Pass
        );
        assert_eq!(summary.counts.tlc_error, 0);
        assert_eq!(summary.counts.pass, 1);
    }

    #[test]
    fn expected_violation_mismatches_remain_parity_failures() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "missing_ty_violation": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": null},
                  "verified_match": true
                },
                "different_violation": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "error", "runtime_seconds": 1.0, "states": 12, "error_type": "deadlock"},
                  "verified_match": true
                },
                "unverified_same_violation": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": false
                }
              }
            }"#,
        )
        .unwrap();

        for spec in [
            "missing_ty_violation",
            "different_violation",
            "unverified_same_violation",
        ] {
            assert_eq!(class_for(&summary, spec), SupremacyMatrixClass::ParityFail);
        }
        assert_eq!(summary.counts.expected_violation_match, 0);
        assert_eq!(summary.counts.tlc_error, 0);
        assert_eq!(summary.counts.parity_fail, 3);
    }

    #[test]
    fn expected_violation_matches_are_non_blocking_without_runtime_evidence() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "missing_ty_runtime": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "slower_violation": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 4.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "tie_violation": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant_violation"},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "missing_ty_runtime"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "slower_violation"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "tie_violation"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(summary.counts.expected_violation_match, 3);
        assert_eq!(summary.counts.missing_runtime, 0);
        assert_eq!(summary.counts.perf_loser, 0);
        assert_eq!(summary.counts.perf_tie, 0);
        assert_eq!(summary.strict_blockers, 0);
        assert!(row_for(&summary, "missing_ty_runtime")
            .reason
            .contains("excluded from runtime supremacy"));
    }

    #[test]
    fn bmc_only_matching_checker_errors_are_non_blocking() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "bmc_only_fixture": {
                  "source": {"mode": "bmc-only"},
                  "tlc": {"status": "error", "states": null, "error_type": "unknown_operator"},
                  "ty": {"status": "error", "states": null, "error_type": "unknown_operator"},
                  "verified_match": true
                },
                "regular_unknown_error": {
                  "source": {},
                  "tlc": {"status": "error", "states": null, "error_type": "unknown_operator"},
                  "ty": {"status": "error", "states": null, "error_type": "unknown_operator"},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            class_for(&summary, "bmc_only_fixture"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "regular_unknown_error"),
            SupremacyMatrixClass::TlcError
        );
        assert_eq!(summary.counts.expected_violation_match, 1);
        assert_eq!(summary.counts.tlc_error, 1);
        assert_eq!(summary.strict_blockers, 1);
        assert!(row_for(&summary, "bmc_only_fixture")
            .reason
            .contains("BMC-only fixture"));
    }

    #[test]
    fn classifies_each_baseline_bucket() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "unsupported": {
                  "source": {"mode": "export"},
                  "tlc": {"status": "pass", "states": 1, "error_type": null},
                  "ty": {"status": "pass", "states": 1, "error_type": null},
                  "verified_match": true
                },
                "simulation_missing_runtime": {
                  "source": {"mode": "simulate"},
                  "tlc": {"status": "error", "states": null, "error_type": "unknown"},
                  "ty": {"status": "pass", "states": null, "error_type": null},
                  "verified_match": true
                },
                "simulation_tlc_error_after_ty_runtime": {
                  "source": {"mode": "simulate"},
                  "tlc": {"status": "error", "states": null, "error_type": "unknown"},
                  "ty": {"status": "pass", "runtime_seconds": 0.5, "states": null, "error_type": null},
                  "verified_match": true
                },
                "tlc_error": {
                  "source": {},
                  "tlc": {"status": "fail", "states": 2, "error_type": "runtime_error"},
                  "ty": {"status": "pass", "states": 2, "error_type": null},
                  "verified_match": true
                },
                "expected_violation_match": {
                  "source": {},
                  "tlc": {"status": "fail", "runtime_seconds": 2.0, "states": 2, "error_type": "invariant"},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 2, "error_type": "invariant_violation"},
                  "verified_match": true
                },
                "tlc_timeout": {
                  "source": {},
                  "tlc": {"status": "timeout", "states": null, "error_type": "timeout"},
                  "ty": {"status": "fail", "states": null, "error_type": "timeout after 120s"},
                  "verified_match": true
                },
                "ty_timeout": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 4.0, "states": 3, "error_type": null},
                  "ty": {"status": "fail", "states": null, "error_type": "timeout after 120s"},
                  "verified_match": false
                },
                "parity_fail": {
                  "source": {},
                  "tlc": {"status": "pass", "states": 4, "error_type": null},
                  "ty": {"status": "pass", "states": 5, "error_type": null},
                  "verified_match": false
                },
                "missing_runtime": {
                  "source": {},
                  "tlc": {"status": "pass", "states": 5, "error_type": null},
                  "ty": {"status": "pass", "states": 5, "error_type": null},
                  "verified_match": true
                },
                "perf_tie": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 6, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.005, "states": 6, "error_type": null},
                  "verified_match": true
                },
                "perf_loser": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 6, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.5, "states": 6, "error_type": null},
                  "verified_match": true
                },
                "pass": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 7, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 7, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(summary.counts.unsupported, 1);
        assert_eq!(summary.counts.expected_violation_match, 1);
        assert_eq!(summary.counts.tlc_error, 1);
        assert_eq!(summary.counts.tlc_timeout, 1);
        assert_eq!(summary.counts.ty_timeout, 1);
        assert_eq!(summary.counts.parity_fail, 1);
        assert_eq!(summary.counts.missing_runtime, 3);
        assert_eq!(summary.counts.perf_tie, 1);
        assert_eq!(summary.counts.perf_loser, 1);
        assert_eq!(summary.counts.pass, 1);
        assert_eq!(
            class_for(&summary, "simulation_missing_runtime"),
            SupremacyMatrixClass::MissingRuntime
        );
        assert_eq!(
            class_for(&summary, "simulation_tlc_error_after_ty_runtime"),
            SupremacyMatrixClass::MissingRuntime
        );
        assert_eq!(
            class_for(&summary, "expected_violation_match"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
    }

    #[test]
    fn undersized_ty_timeout_budget_is_missing_runtime_evidence() {
        let baseline = r#"{
          "specs": {
            "policy_artifact": {
              "diagnose_timeout_seconds": 450,
              "source": {"mode": "check"},
              "tlc": {"status": "pass", "runtime_seconds": 152.0, "states": 901692, "error_type": null},
              "ty": {"status": "timeout", "runtime_seconds": 300.02, "states": null, "error_type": "timeout"},
              "verified_match": true
            },
            "true_timeout": {
              "diagnose_timeout_seconds": 300,
              "source": {"mode": "check"},
              "tlc": {"status": "pass", "runtime_seconds": 152.0, "states": 901692, "error_type": null},
              "ty": {"status": "timeout", "runtime_seconds": 300.02, "states": null, "error_type": "timeout"},
              "verified_match": true
            }
          }
        }"#;

        let summary = classify_baseline_str(baseline).unwrap();

        assert_eq!(
            class_for(&summary, "policy_artifact"),
            SupremacyMatrixClass::MissingRuntime
        );
        assert_eq!(
            class_for(&summary, "true_timeout"),
            SupremacyMatrixClass::TyTimeout
        );
        assert_eq!(summary.counts.missing_runtime, 1);
        assert_eq!(summary.counts.ty_timeout, 1);
        assert!(reason_for(&summary, "policy_artifact").contains("diagnose_timeout_seconds=450s"));
        let diagnostics = summary
            .missing_runtime_diagnostics
            .as_ref()
            .expect("missing runtime diagnostics");
        assert_eq!(
            diagnostics.missing_ty_runtime_specs,
            vec!["policy_artifact".to_string()]
        );
        assert_eq!(diagnostics.missing_tlc_runtime_specs, Vec::<String>::new());

        // The diagnose budget is annotation-only: it surfaces in the missing-runtime
        // reason but must NOT escalate the per-spec run timeout (see collect_spec_runtime,
        // where --runtime-timeout is an authoritative hard cap).
        let parsed: SpecBaseline = serde_json::from_str(baseline).unwrap();
        let entry = parsed.specs.get("policy_artifact").unwrap();
        assert_eq!(diagnose_timeout_seconds(entry), Some(450));
    }

    #[test]
    fn simulation_runtime_argv_uses_simulation_mode_without_trust_cg_backend() {
        let argv = super::ty_simulation_runtime_argv(
            Path::new("target/ty"),
            Path::new("specs/Sim.tla"),
            Path::new("specs/Sim.cfg"),
        );

        assert_eq!(argv[1], "simulate");
        assert!(argv.contains(&"--no-invariants".to_string()));
        assert!(argv.contains(&"--allow-io".to_string()));
        assert!(argv.contains(&"--num-traces".to_string()));
        assert!(argv.contains(&"--max-trace-length".to_string()));
        assert!(!argv.contains(&"--backend".to_string()));
        assert!(!argv.contains(&"trust-cg".to_string()));
    }

    #[test]
    fn matrix_tlc_runtime_base_argv_uses_shared_single_thread_jvm_profile() {
        let argv = super::tlc_matrix_runtime_base_argv("tytools.jar", Some(Path::new("lib")));

        assert_eq!(argv[0], "java");
        for arg in super::tlc_java_single_thread_args() {
            assert!(argv.contains(&(*arg).to_string()), "{arg}");
        }
        assert!(argv.contains(&"-DTLA-Library=lib".to_string()));
        assert!(argv.contains(&"tlc2.TLC".to_string()));
        assert!(!argv.contains(&"-XX:+UseParallelGC".to_string()));
    }

    #[test]
    fn tlc_simulation_runtime_uses_generate_for_generate_mode_specs() {
        let dir = tempfile::tempdir().unwrap();
        let generate_spec = dir.path().join("SimGenerate.tla");
        fs::write(
            &generate_spec,
            r#"---- MODULE SimGenerate ----
ASSUME TLCGet("config").mode = "generate"
====
"#,
        )
        .unwrap();
        let simulate_spec = dir.path().join("SimPlain.tla");
        fs::write(&simulate_spec, "---- MODULE SimPlain ----\n====\n").unwrap();
        let spaced_generate_spec = dir.path().join("SimSpacedGenerate.tla");
        fs::write(
            &spaced_generate_spec,
            "---- MODULE SimSpacedGenerate ----\nASSUME TLCGet(\"config\").mode     =\n    \"generate\"\n====\n",
        )
        .unwrap();

        assert_eq!(super::tlc_simulation_mode_arg(&generate_spec), "-generate");
        assert_eq!(
            super::tlc_simulation_mode_arg(&spaced_generate_spec),
            "-generate"
        );
        assert_eq!(super::tlc_simulation_mode_arg(&simulate_spec), "-simulate");
    }

    #[test]
    fn runtime_error_reads_tlc_stdout_for_incomplete_initial_state() {
        let error = super::runtime_error_with_output(
            12,
            false,
            "Error: State is not completely specified by the initial predicate.\n",
            "",
        );

        assert_eq!(error.as_deref(), Some("incomplete_initial_state"));
    }

    #[test]
    fn runtime_error_ignores_tlc_semantic_warnings_only() {
        let stdout = "\
Semantic errors:

*** Warnings: 10

Warning: the definition of 'Restrict' conflicts with its definition.

Model checking completed. No error has been found.
227344 states generated, 14424 distinct states found, 0 states left on queue.
";

        assert_eq!(super::runtime_error_with_output(0, false, stdout, ""), None);
    }

    #[test]
    fn runtime_error_ignores_tlc_fairness_warning_after_semantic_warnings() {
        let stdout = "\
Semantic errors:

*** Warnings: 10

Starting...
Warning: Temporal properties (PROPERTY or PROPERTIES) are being verified without a fairness constraint.
Model checking completed. No error has been found.
90 states generated, 45 distinct states found, 0 states left on queue.
";

        assert_eq!(super::runtime_error_with_output(0, false, stdout, ""), None);
    }

    #[test]
    fn runtime_error_classifies_nonzero_tlc_semantic_heading_without_error_count_as_parse() {
        let plural_stdout = "\
Semantic errors:

line 1, col 1 to line 1, col 5 of module Foo
";
        let singular_stderr = "Error: Semantic error: Unknown operator Foo\n";

        assert_eq!(
            super::runtime_error_with_output(12, false, plural_stdout, "").as_deref(),
            Some("parse")
        );
        assert_eq!(
            super::runtime_error_with_output(12, false, "", singular_stderr).as_deref(),
            Some("parse")
        );
    }

    #[test]
    fn runtime_error_prefers_timeout_and_expected_violations_over_semantic_heading() {
        let invariant_stdout = "\
Semantic errors:

Error: Invariant TypeOK is violated.
";
        let deadlock_stdout = "\
Semantic errors:

Error: Deadlock reached.
";

        assert_eq!(
            super::runtime_error_with_output(12, true, invariant_stdout, "").as_deref(),
            Some("timeout")
        );
        assert_eq!(
            super::runtime_error_with_output(12, false, invariant_stdout, "").as_deref(),
            Some("invariant")
        );
        assert_eq!(
            super::runtime_error_with_output(12, false, deadlock_stdout, "").as_deref(),
            Some("deadlock")
        );
    }

    #[test]
    fn runtime_error_prefers_liveness_over_invariant_compile_noise() {
        let stderr = "\
[trust_cg] invariant 'BigNext' not yet compiled
Error: Temporal properties were violated.
Error: Liveness violation detected.
";

        assert_eq!(
            super::runtime_error_with_output(12, false, "", stderr).as_deref(),
            Some("liveness")
        );
    }

    #[test]
    fn runtime_error_classifies_tlc_semantic_errors_count_as_parse() {
        let stdout = "\
Semantic errors:

*** Errors: 1

line 1, col 1 to line 1, col 5 of module Foo
";

        assert_eq!(
            super::runtime_error_with_output(0, false, stdout, "").as_deref(),
            Some("parse")
        );
    }

    #[test]
    fn runtime_error_ignores_assume_named_ops_and_backend_fallback_noise_on_success() {
        let stderr = "\
[trust_cg] callee lower failed for FastAssume
[trust_cg] failed to compile invariant FastAssume
";

        assert_eq!(super::runtime_error_with_output(0, false, "", stderr), None);
    }

    #[test]
    fn runtime_error_classifies_real_assumption_false_output() {
        let stdout = "Error: Assumption line 4, col 1 to line 4, col 12 is false.\n";

        assert_eq!(
            super::runtime_error_with_output(0, false, stdout, "").as_deref(),
            Some("assume_violation")
        );
    }

    #[test]
    fn runtime_error_classifies_expected_violations_from_successful_ty_output() {
        let error = super::runtime_error_with_output(
            0,
            false,
            "States found: 12\nError: Invariant TypeOK is violated.\n",
            "",
        );

        assert_eq!(error.as_deref(), Some("invariant"));
        assert_eq!(status_for_result(0, false), "pass");
    }

    #[test]
    fn runtime_error_falls_back_to_stdout_when_stderr_is_empty() {
        let error = super::runtime_error_with_output(12, false, "last useful line\n", "");

        assert_eq!(error.as_deref(), Some("last useful line"));
    }

    #[test]
    fn runtime_evidence_keeps_elapsed_seconds_for_blocker_outcomes() {
        assert_eq!(
            runtime_seconds_for_evidence(&Some("invariant".to_string()), 1.25),
            Some(1.25)
        );
        assert_eq!(
            runtime_seconds_for_evidence(&Some("timeout".to_string()), 300.0),
            Some(300.0)
        );
        assert_eq!(
            runtime_seconds_for_evidence(&Some("debug_build_runtime_evidence".to_string()), 1.25),
            None
        );
        assert_eq!(
            runtime_seconds_for_evidence(&Some("backend_unavailable".to_string()), 1.25),
            None
        );
        assert_eq!(
            runtime_seconds_for_evidence(&Some("profile_runtime_evidence".to_string()), 1.25),
            None
        );
    }

    #[test]
    fn runtime_refresh_evidence_is_independent_from_verified_match() {
        let tlc = runtime_mode_evidence("pass", Some(2.0), Some(10), None, "ParityRow", "tlc");
        let ty = runtime_mode_evidence("pass", Some(1.0), Some(11), None, "ParityRow", "ty");

        assert!(!runtime_modes_verified_match(&tlc, &ty));
        assert!(runtime_modes_have_fresh_evidence(&tlc, &ty));

        let tlc_violation = runtime_mode_evidence(
            "fail",
            Some(3.0),
            Some(12),
            Some("invariant"),
            "ViolationRow",
            "tlc",
        );
        let ty_violation = runtime_mode_evidence(
            "pass",
            Some(1.0),
            Some(12),
            Some("invariant_violation"),
            "ViolationRow",
            "ty",
        );

        assert!(runtime_modes_verified_match(&tlc_violation, &ty_violation));
        assert!(runtime_modes_have_fresh_evidence(
            &tlc_violation,
            &ty_violation
        ));
    }

    #[test]
    fn runtime_expected_violation_match_uses_reported_name_not_state_count() {
        let dir = tempfile::tempdir().unwrap();
        let tlc_dir = dir.path().join("tlc");
        let ty_dir = dir.path().join("ty");
        fs::create_dir_all(&tlc_dir).unwrap();
        fs::create_dir_all(&ty_dir).unwrap();
        fs::write(
            tlc_dir.join("stdout.txt"),
            "Error: Invariant NotSolved is violated.\n14 states generated\n",
        )
        .unwrap();
        fs::write(
            ty_dir.join("stderr.txt"),
            "Error: Invariant NotSolved is violated.\nError: Invariant violation detected\n",
        )
        .unwrap();

        let mut tlc =
            runtime_mode_evidence("fail", Some(3.0), Some(14), Some("invariant"), "Row", "tlc");
        tlc.artifact_dir = tlc_dir;
        let mut ty = runtime_mode_evidence(
            "pass",
            Some(1.0),
            Some(13),
            Some("invariant_violation"),
            "Row",
            "ty",
        );
        ty.artifact_dir = ty_dir;

        assert!(runtime_modes_verified_match(&tlc, &ty));
    }

    #[test]
    fn runtime_expected_violation_different_names_do_not_match() {
        let dir = tempfile::tempdir().unwrap();
        let tlc_dir = dir.path().join("tlc");
        let ty_dir = dir.path().join("ty");
        fs::create_dir_all(&tlc_dir).unwrap();
        fs::create_dir_all(&ty_dir).unwrap();
        fs::write(
            tlc_dir.join("stdout.txt"),
            "Error: Invariant AC1 is violated.\n",
        )
        .unwrap();
        fs::write(
            ty_dir.join("stderr.txt"),
            "Error: Invariant AC2 is violated.\n",
        )
        .unwrap();

        let mut tlc =
            runtime_mode_evidence("fail", Some(3.0), Some(12), Some("invariant"), "Row", "tlc");
        tlc.artifact_dir = tlc_dir;
        let mut ty = runtime_mode_evidence(
            "pass",
            Some(1.0),
            Some(12),
            Some("invariant_violation"),
            "Row",
            "ty",
        );
        ty.artifact_dir = ty_dir;

        assert!(!runtime_modes_verified_match(&tlc, &ty));
    }

    #[test]
    fn runtime_expected_violation_generic_message_keeps_state_count_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let tlc_dir = dir.path().join("tlc");
        let ty_dir = dir.path().join("ty");
        fs::create_dir_all(&tlc_dir).unwrap();
        fs::create_dir_all(&ty_dir).unwrap();
        fs::write(
            tlc_dir.join("stdout.txt"),
            "Error: Invariant is violated.\n",
        )
        .unwrap();
        fs::write(ty_dir.join("stderr.txt"), "Error: Invariant is violated.\n").unwrap();

        let mut tlc =
            runtime_mode_evidence("fail", Some(3.0), Some(14), Some("invariant"), "Row", "tlc");
        tlc.artifact_dir = tlc_dir;
        let mut ty = runtime_mode_evidence(
            "pass",
            Some(1.0),
            Some(13),
            Some("invariant_violation"),
            "Row",
            "ty",
        );
        ty.artifact_dir = ty_dir;

        assert_eq!(
            expected_violation_name(
                ExpectedViolationKind::Invariant,
                "Error: Invariant is violated.\n"
            ),
            None
        );
        assert!(!runtime_modes_verified_match(&tlc, &ty));
    }

    #[test]
    fn runtime_expected_violation_without_name_requires_real_state_count() {
        let tlc = runtime_mode_evidence("fail", Some(3.0), None, Some("invariant"), "Row", "tlc");
        let ty = runtime_mode_evidence(
            "pass",
            Some(1.0),
            None,
            Some("invariant_violation"),
            "Row",
            "ty",
        );

        assert!(!runtime_modes_verified_match(&tlc, &ty));
    }

    #[test]
    fn simulation_tlc_error_with_runtimes_is_evidence_not_missing() {
        let baseline = r#"{
          "specs": {
            "simulation_tlc_error_with_runtime": {
              "source": {"mode": "simulate"},
              "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": null, "error_type": "invariant"},
              "ty": {"status": "pass", "runtime_seconds": 1.0, "states": null, "error_type": "invariant_violation"},
              "verified_match": true
            }
          }
        }"#;

        let strict_summary = classify_baseline_str(baseline).unwrap();
        assert_eq!(
            class_for(&strict_summary, "simulation_tlc_error_with_runtime"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );

        let policy_summary = classify_baseline_str_with_policy(
            baseline,
            &MatrixPolicy {
                allow_runtime_to_error: true,
                allow_timeout_dominance: false,
            },
        )
        .unwrap();
        assert_eq!(
            class_for(&policy_summary, "simulation_tlc_error_with_runtime"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
    }

    #[test]
    fn simulation_runtime_config_strips_checker_clauses() {
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("Sim.cfg");
        let artifact_dir = dir.path().join("artifact");
        fs::write(
            &cfg_path,
            r#"CONSTANT
    N = 3
SPECIFICATION Spec
INVARIANT Inv
PROPERTY
    TDSpec
POSTCONDITION
    PostCondition
CONSTRAINT StopAfter
CHECK_DEADLOCK FALSE
"#,
        )
        .unwrap();

        let generated =
            super::write_simulation_runtime_config(&cfg_path, dir.path(), &artifact_dir).unwrap();
        let stripped = fs::read_to_string(&generated).unwrap();

        assert!(generated.ends_with("simulation.generated.cfg"));
        assert!(stripped.contains("SPECIFICATION Spec"));
        assert!(stripped.contains("CONSTRAINT StopAfter"));
        assert!(stripped.contains("CHECK_DEADLOCK FALSE"));
        assert!(!stripped.contains("INVARIANT"));
        assert!(!stripped.contains("PROPERTY"));
        assert!(!stripped.contains("POSTCONDITION"));
        assert!(!stripped.contains("TDSpec"));
        assert!(!stripped.contains("PostCondition"));
    }

    #[test]
    fn classifies_current_spec_baseline_counts() {
        // Counts track the live checked-in `tests/tlc_comparison/spec_baseline.json`
        // and drift as baselines are refreshed. Per-class invariants we still pin:
        //  - `unsupported`, `tlc_error`, `ty_timeout`, `perf_tie` stay at 0,
        //  - `pass` stays >= 1 (`MCReachabilityTestAllGraphs`),
        //  - `perf_loser` stays >= 1 (`dijkstra-mutex_Safety-4-processors`),
        //  - the total row count matches the baseline file's spec count.
        let summary = classify_baseline_path(&repo_baseline_path()).unwrap();

        assert_eq!(summary.counts.unsupported, 0);
        assert_eq!(summary.counts.expected_violation_match, 8);
        assert_eq!(summary.counts.tlc_error, 0);
        assert_eq!(summary.counts.tlc_timeout, 19);
        assert_eq!(summary.counts.ty_timeout, 0);
        assert_eq!(summary.counts.parity_fail, 14);
        assert_eq!(summary.counts.missing_runtime, 181);
        assert_eq!(summary.counts.perf_tie, 0);
        assert_eq!(summary.counts.perf_loser, 1);
        assert_eq!(summary.counts.pass, 1);
        assert_eq!(summary.rows.len(), 224);

        assert_eq!(
            class_for(&summary, "SimTokenRing"),
            SupremacyMatrixClass::MissingRuntime
        );
        assert_eq!(
            class_for(&summary, "BPConProof"),
            SupremacyMatrixClass::TlcTimeout
        );
        assert_eq!(
            class_for(&summary, "ACP_NB_WRONG_TLC"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "BmcUnsafeCounter"),
            SupremacyMatrixClass::ExpectedViolationMatch
        );
        assert_eq!(
            class_for(&summary, "MCReachabilityTestAllGraphs"),
            SupremacyMatrixClass::Pass
        );
        assert_eq!(
            class_for(&summary, "dijkstra-mutex_Safety-4-processors"),
            SupremacyMatrixClass::PerfLoser
        );
        assert_eq!(
            class_for(&summary, "ABCorrectness"),
            SupremacyMatrixClass::MissingRuntime
        );

        let reachability = row_for(&summary, "MCReachabilityTestAllGraphs");
        assert_eq!(reachability.perf_loser_follow_up, None);
        assert_eq!(reachability.perf_loser_rank, None);
        assert_eq!(reachability.seconds_lost_vs_tlc, None);

        let dijkstra = row_for(&summary, "dijkstra-mutex_Safety-4-processors");
        assert_eq!(
            dijkstra.perf_loser_follow_up.as_deref(),
            Some("alabsystems/ty#4392")
        );
        assert_eq!(dijkstra.perf_loser_rank, Some(1));
        assert!(dijkstra
            .seconds_lost_vs_tlc
            .is_some_and(|seconds_lost| seconds_lost > 0.0));
    }

    #[test]
    fn current_spec_baseline_promotion_metadata_is_fresh() {
        validate_enforceable_baseline_path(&repo_baseline_path())
            .expect("checked-in spec_baseline.json promotion metadata should be fresh");
    }

    #[test]
    fn refresh_runtime_recomputes_promotion_metadata_for_supported_baseline() {
        let mut baseline = json!({
            "schema_version": 3,
            "total_specs": 99,
            "specs_jcs_sha256": "stale",
            "stats": {
                "ty_fail": 99,
                "ty_match": 99,
                "ty_mismatch": 99,
                "ty_untested": 99,
                "tlc_error": 99,
                "tlc_pass": 99,
                "tlc_timeout": 99
            },
            "categories": {
                "medium": 99,
                "xlarge": 0
            },
            "specs": {
                "RuntimeSpec": {
                    "category": "medium",
                    "source": {},
                    "tlc": {"status": "pass", "states": null, "error_type": null},
                    "ty": {
                        "status": "pass",
                        "states": null,
                        "error_type": null,
                        "last_run": "2026-01-01T00:00:00Z",
                        "git_commit": "stale"
                    },
                    "verified_match": true
                },
                "TimeoutSpec": {
                    "category": "xlarge",
                    "source": {},
                    "tlc": {"status": "timeout", "states": null, "error_type": "timeout"},
                    "ty": {"status": "untested", "states": null, "error_type": null},
                    "verified_match": false
                }
            }
        });
        let row = runtime_evidence_row("RuntimeSpec", 2.0, 1.0, 10);
        let provenance = runtime_baseline_provenance();

        apply_runtime_row(&mut baseline, &row, &provenance);
        let metadata = refresh_runtime_baseline_metadata(&mut baseline, &[row], &provenance)
            .expect("metadata refresh should succeed");

        assert_eq!(metadata, BaselineMetadataRefresh::PromotionReady);
        assert!(baseline.get(RUNTIME_METADATA_WARNING_FIELD).is_none());
        assert_eq!(
            baseline["specs"]["RuntimeSpec"]["tlc"]["runtime_seconds"],
            json!(2.0)
        );
        assert_eq!(
            baseline["specs"]["RuntimeSpec"]["ty"]["runtime_seconds"],
            json!(1.0)
        );
        assert_eq!(
            baseline["specs"]["RuntimeSpec"]["ty"]["last_run"],
            json!("2026-04-28T20:30:00Z")
        );
        assert_eq!(
            baseline["specs"]["RuntimeSpec"]["ty"]["git_commit"],
            json!("abc1234")
        );
        assert_eq!(baseline["stats"]["tlc_pass"], json!(1));
        assert_eq!(baseline["stats"]["tlc_timeout"], json!(1));
        assert_eq!(baseline["stats"]["tlc_error"], json!(0));
        assert_eq!(baseline["stats"]["ty_match"], json!(1));
        assert_eq!(baseline["stats"]["ty_untested"], json!(1));
        assert_eq!(baseline["total_specs"], json!(2));
        assert_eq!(baseline["categories"]["medium"], json!(1));
        assert_eq!(baseline["categories"]["xlarge"], json!(1));
        assert_eq!(
            baseline["ty_refresh"]["script"],
            json!("ty supremacy matrix --refresh-runtime")
        );
        assert_eq!(baseline["ty_refresh"]["git_commit"], json!("abc1234"));
        assert_eq!(
            baseline["ty_refresh"]["binary_sha256"],
            json!("0123456789abcdef")
        );

        let expected_digest = sha256_jcs_value(&baseline["specs"]).expect("digest should compute");
        assert_eq!(baseline["specs_jcs_sha256"], json!(expected_digest));
        validate_enforceable_baseline_value(&baseline)
            .expect("refreshed promotion metadata should be enforceable");

        let summary = classify_baseline_value(baseline).expect("baseline should classify");
        assert_eq!(
            class_for(&summary, "RuntimeSpec"),
            SupremacyMatrixClass::Pass
        );
        assert_eq!(
            class_for(&summary, "TimeoutSpec"),
            SupremacyMatrixClass::TlcTimeout
        );
    }

    #[test]
    fn refresh_runtime_updates_timeout_rows_for_timeout_dominance_policy() {
        let mut baseline = json!({
            "schema_version": 3,
            "total_specs": 1,
            "specs_jcs_sha256": "stale",
            "stats": {
                "ty_fail": 0,
                "ty_match": 1,
                "ty_mismatch": 0,
                "ty_untested": 0,
                "tlc_error": 0,
                "tlc_pass": 0,
                "tlc_timeout": 1
            },
            "specs": {
                "TimeoutRow": {
                    "category": "xlarge",
                    "source": {},
                    "tlc": {"status": "timeout", "states": null, "error_type": "timeout"},
                    "ty": {
                        "status": "pass",
                        "states": 100,
                        "error_type": null,
                        "last_run": "2026-01-01T00:00:00Z",
                        "git_commit": "stale"
                    },
                    "verified_match": true
                }
            }
        });
        let policy = MatrixPolicy {
            allow_runtime_to_error: false,
            allow_timeout_dominance: true,
        };
        let before =
            classify_baseline_value_with_policy(baseline.clone(), &policy).expect("baseline");
        assert_eq!(
            class_for(&before, "TimeoutRow"),
            SupremacyMatrixClass::TlcTimeout
        );

        let tlc = runtime_mode_evidence(
            "timeout",
            Some(60.0),
            None,
            Some("timeout"),
            "TimeoutRow",
            "tlc",
        );
        let ty = runtime_mode_evidence("pass", Some(5.0), Some(100), None, "TimeoutRow", "ty");
        assert!(!runtime_modes_verified_match(&tlc, &ty));
        assert!(runtime_modes_have_fresh_evidence(&tlc, &ty));
        let row = RuntimeEvidenceRow {
            spec: "TimeoutRow".to_string(),
            verified_match: runtime_modes_verified_match(&tlc, &ty),
            refreshed: runtime_modes_have_fresh_evidence(&tlc, &ty),
            tlc,
            ty,
            note: None,
            required_flags: Vec::new(),
        };
        let provenance = runtime_baseline_provenance();

        apply_runtime_row(&mut baseline, &row, &provenance);
        let metadata = refresh_runtime_baseline_metadata(&mut baseline, &[row], &provenance)
            .expect("metadata refresh should succeed");

        assert_eq!(metadata, BaselineMetadataRefresh::PromotionReady);
        assert_eq!(
            baseline["specs"]["TimeoutRow"]["tlc"]["runtime_seconds"],
            json!(60.0)
        );
        assert_eq!(
            baseline["specs"]["TimeoutRow"]["ty"]["runtime_seconds"],
            json!(5.0)
        );
        assert_eq!(
            baseline["specs"]["TimeoutRow"]["verified_match"],
            json!(true)
        );
        assert_eq!(baseline["ty_refresh"]["specs_updated"], json!(1));

        let summary = classify_baseline_value_with_policy(baseline, &policy)
            .expect("baseline should classify");
        assert_eq!(
            class_for(&summary, "TimeoutRow"),
            SupremacyMatrixClass::TimeoutDominance
        );
        assert_eq!(summary.counts.timeout_dominance, 1);
        assert_eq!(summary.counts.missing_runtime, 0);
        assert_eq!(summary.enforce_blocker_count(), 0);
    }

    #[test]
    fn refresh_runtime_removes_missing_runtime_from_ty_parity_failures() {
        let mut baseline = json!({
            "schema_version": 3,
            "total_specs": 1,
            "specs_jcs_sha256": "stale",
            "stats": {
                "ty_fail": 0,
                "ty_match": 0,
                "ty_mismatch": 0,
                "ty_untested": 1,
                "tlc_error": 0,
                "tlc_pass": 1,
                "tlc_timeout": 0
            },
            "specs": {
                "ParityRow": {
                    "category": "small",
                    "source": {},
                    "tlc": {"status": "pass", "states": 10, "error_type": null},
                    "ty": {
                        "status": "pass",
                        "states": 10,
                        "error_type": null,
                        "last_run": "2026-01-01T00:00:00Z",
                        "git_commit": "stale"
                    },
                    "verified_match": true
                }
            }
        });
        let before = classify_baseline_value(baseline.clone()).expect("baseline should classify");
        assert_eq!(
            class_for(&before, "ParityRow"),
            SupremacyMatrixClass::MissingRuntime
        );
        let tlc = runtime_mode_evidence("pass", Some(2.0), Some(10), None, "ParityRow", "tlc");
        let ty = runtime_mode_evidence("pass", Some(1.0), Some(11), None, "ParityRow", "ty");
        let row = RuntimeEvidenceRow {
            spec: "ParityRow".to_string(),
            verified_match: runtime_modes_verified_match(&tlc, &ty),
            refreshed: runtime_modes_have_fresh_evidence(&tlc, &ty),
            tlc,
            ty,
            note: None,
            required_flags: Vec::new(),
        };
        let provenance = runtime_baseline_provenance();

        apply_runtime_row(&mut baseline, &row, &provenance);
        let metadata = refresh_runtime_baseline_metadata(&mut baseline, &[row], &provenance)
            .expect("metadata refresh should succeed");

        assert_eq!(metadata, BaselineMetadataRefresh::PromotionReady);
        assert_eq!(
            baseline["specs"]["ParityRow"]["verified_match"],
            json!(false)
        );
        assert_eq!(
            baseline["specs"]["ParityRow"]["ty"]["runtime_seconds"],
            json!(1.0)
        );
        assert_eq!(
            baseline["specs"]["ParityRow"]["ty"]["last_run"],
            json!("2026-04-28T20:30:00Z")
        );
        assert_eq!(baseline["ty_refresh"]["specs_updated"], json!(1));

        let summary = classify_baseline_value(baseline).expect("baseline should classify");
        assert_eq!(
            class_for(&summary, "ParityRow"),
            SupremacyMatrixClass::ParityFail
        );
        assert_eq!(summary.counts.missing_runtime, 0);
    }

    #[test]
    fn refresh_runtime_does_not_promote_invalid_runtime_evidence_markers() {
        for error_type in [
            "backend_unavailable",
            "debug_build_runtime_evidence",
            "profile_runtime_evidence",
        ] {
            let mut baseline = json!({
                "schema_version": 3,
                "total_specs": 1,
                "specs_jcs_sha256": "stale",
                "stats": {
                    "ty_fail": 0,
                    "ty_match": 0,
                    "ty_mismatch": 0,
                    "ty_untested": 1,
                    "tlc_error": 0,
                    "tlc_pass": 1,
                    "tlc_timeout": 0
                },
                "specs": {
                    "InvalidEvidence": {
                        "category": "small",
                        "source": {},
                        "tlc": {"status": "pass", "states": 3, "error_type": null},
                        "ty": {
                            "status": "pass",
                            "states": 3,
                            "error_type": null,
                            "last_run": "2026-01-01T00:00:00Z",
                            "git_commit": "stale"
                        },
                        "verified_match": true
                    }
                }
            });
            let before =
                classify_baseline_value(baseline.clone()).expect("baseline should classify");
            assert_eq!(
                class_for(&before, "InvalidEvidence"),
                SupremacyMatrixClass::MissingRuntime
            );
            let original_spec = baseline["specs"]["InvalidEvidence"].clone();
            let tlc =
                runtime_mode_evidence("pass", Some(2.0), Some(3), None, "InvalidEvidence", "tlc");
            let invalid_error = Some(error_type.to_string());
            let ty = runtime_mode_evidence(
                if error_type == "backend_unavailable" {
                    "fail"
                } else {
                    "pass"
                },
                runtime_seconds_for_evidence(&invalid_error, 1.0),
                Some(3),
                Some(error_type),
                "InvalidEvidence",
                "ty",
            );
            let row = RuntimeEvidenceRow {
                spec: "InvalidEvidence".to_string(),
                verified_match: runtime_modes_verified_match(&tlc, &ty),
                refreshed: runtime_modes_have_fresh_evidence(&tlc, &ty),
                tlc,
                ty,
                note: None,
                required_flags: Vec::new(),
            };
            assert!(!row.refreshed, "{error_type}");
            let provenance = runtime_baseline_provenance();

            apply_runtime_row(&mut baseline, &row, &provenance);
            let metadata = refresh_runtime_baseline_metadata(
                &mut baseline,
                std::slice::from_ref(&row),
                &provenance,
            )
            .expect("metadata refresh should succeed");

            assert_eq!(metadata, BaselineMetadataRefresh::PromotionReady);
            assert_eq!(baseline["specs"]["InvalidEvidence"], original_spec);
            assert_eq!(baseline["ty_refresh"]["specs_updated"], json!(0));

            let summary = classify_baseline_value(baseline).expect("baseline should classify");
            assert_eq!(
                class_for(&summary, "InvalidEvidence"),
                SupremacyMatrixClass::MissingRuntime,
                "{error_type}"
            );
            assert_eq!(summary.counts.missing_runtime, 1, "{error_type}");
        }
    }

    #[test]
    fn refresh_runtime_does_not_mix_fresh_tlc_with_stale_ty_and_verified_match() {
        let mut baseline = json!({
            "schema_version": 3,
            "total_specs": 1,
            "specs_jcs_sha256": "stale",
            "stats": {
                "ty_fail": 0,
                "ty_match": 1,
                "ty_mismatch": 0,
                "ty_untested": 0,
                "tlc_error": 0,
                "tlc_pass": 1,
                "tlc_timeout": 0
            },
            "specs": {
                "MixedRow": {
                    "category": "small",
                    "source": {},
                    "tlc": {"status": "pass", "states": 10, "error_type": null},
                    "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 10, "error_type": null},
                    "verified_match": true
                }
            }
        });
        let original_spec = baseline["specs"]["MixedRow"].clone();
        let before = classify_baseline_value(baseline.clone()).expect("baseline should classify");
        assert_eq!(
            class_for(&before, "MixedRow"),
            SupremacyMatrixClass::MissingRuntime
        );

        let tlc = runtime_mode_evidence("pass", Some(2.0), Some(11), None, "MixedRow", "tlc");
        let invalid = Some("backend_unavailable".to_string());
        let ty = runtime_mode_evidence(
            "fail",
            runtime_seconds_for_evidence(&invalid, 1.0),
            Some(11),
            Some("backend_unavailable"),
            "MixedRow",
            "ty",
        );
        let row = RuntimeEvidenceRow {
            spec: "MixedRow".to_string(),
            verified_match: runtime_modes_verified_match(&tlc, &ty),
            refreshed: runtime_modes_have_fresh_evidence(&tlc, &ty),
            tlc,
            ty,
            note: None,
            required_flags: Vec::new(),
        };
        let provenance = runtime_baseline_provenance();

        apply_runtime_row(&mut baseline, &row, &provenance);
        let metadata = refresh_runtime_baseline_metadata(&mut baseline, &[row], &provenance)
            .expect("metadata refresh should succeed");

        assert_eq!(metadata, BaselineMetadataRefresh::PromotionReady);
        assert_eq!(baseline["specs"]["MixedRow"], original_spec);
        assert_eq!(baseline["ty_refresh"]["specs_updated"], json!(0));

        let summary = classify_baseline_value(baseline).expect("baseline should classify");
        assert_eq!(
            class_for(&summary, "MixedRow"),
            SupremacyMatrixClass::MissingRuntime
        );
    }

    #[test]
    fn runtime_collection_errors_are_recorded_as_evidence_rows() {
        let mut baseline = json!({
            "schema_version": 3,
            "total_specs": 1,
            "specs_jcs_sha256": "stale",
            "stats": {
                "ty_fail": 0,
                "ty_match": 0,
                "ty_mismatch": 0,
                "ty_untested": 1,
                "tlc_error": 0,
                "tlc_pass": 1,
                "tlc_timeout": 0
            },
            "specs": {
                "CollectionError": {
                    "category": "small",
                    "source": {},
                    "tlc": {"status": "pass", "states": 3, "error_type": null},
                    "ty": {"status": "pass", "states": 3, "error_type": null},
                    "verified_match": true
                }
            }
        });
        let original_spec = baseline["specs"]["CollectionError"].clone();
        let error = anyhow::anyhow!("run TLC failed");
        let row = runtime_collection_error_row("CollectionError", Path::new("out"), &error);

        assert_eq!(row.spec, "CollectionError");
        assert_eq!(
            row.tlc.error_type.as_deref(),
            Some(RUNTIME_COLLECTION_FAILED_ERROR_TYPE)
        );
        assert_eq!(row.tlc.runtime_seconds, None);
        assert!(row.tlc.states.is_none());
        assert!(!row.refreshed);
        assert!(row
            .note
            .as_deref()
            .is_some_and(|note| note.contains("run TLC failed")));
        let errors = runtime_evidence_errors(std::slice::from_ref(&row));
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].spec, "CollectionError");

        let provenance = runtime_baseline_provenance();
        apply_runtime_row(&mut baseline, &row, &provenance);
        let metadata = refresh_runtime_baseline_metadata(&mut baseline, &[row], &provenance)
            .expect("metadata refresh should succeed");

        assert_eq!(metadata, BaselineMetadataRefresh::PromotionReady);
        assert_eq!(baseline["specs"]["CollectionError"], original_spec);
        assert_eq!(baseline["ty_refresh"]["specs_updated"], json!(0));
    }

    #[test]
    fn enforce_runtime_refresh_rejects_unpromoted_selected_rows() {
        let passing = runtime_evidence_row("Passing", 2.0, 1.0, 3);
        validate_runtime_refresh_rows_promoted(std::slice::from_ref(&passing))
            .expect("complete fresh evidence should be promotable");

        let collection_error = runtime_collection_error_row(
            "CollectionError",
            Path::new("out"),
            &anyhow::anyhow!("TLC failed"),
        );
        let err = validate_runtime_refresh_rows_promoted(std::slice::from_ref(&collection_error))
            .expect_err("collection errors must fail enforce-mode refresh");
        assert!(err.to_string().contains("CollectionError"));
        assert!(err
            .to_string()
            .contains("without promotable fresh evidence"));

        let tlc = runtime_mode_evidence("pass", Some(2.0), Some(3), None, "DebugRow", "tlc");
        let invalid_error = Some("debug_build_runtime_evidence".to_string());
        let ty = runtime_mode_evidence(
            "pass",
            runtime_seconds_for_evidence(&invalid_error, 1.0),
            Some(3),
            Some("debug_build_runtime_evidence"),
            "DebugRow",
            "ty",
        );
        let debug_row = RuntimeEvidenceRow {
            spec: "DebugRow".to_string(),
            verified_match: true,
            refreshed: runtime_modes_have_fresh_evidence(&tlc, &ty),
            tlc,
            ty,
            note: None,
            required_flags: Vec::new(),
        };

        let err = validate_runtime_refresh_rows_promoted(&[passing, debug_row])
            .expect_err("invalid runtime markers must fail enforce-mode refresh");
        assert!(err.to_string().contains("DebugRow"));
    }

    #[test]
    fn refresh_runtime_marks_unknown_schema_as_not_promotion_ready() {
        let mut baseline = json!({
            "specs": {
                "RuntimeSpec": {
                    "source": {},
                    "tlc": {"status": "pass", "states": null, "error_type": null},
                    "ty": {"status": "pass", "states": null, "error_type": null},
                    "verified_match": true
                }
            }
        });
        let row = runtime_evidence_row("RuntimeSpec", 3.0, 1.0, 10);
        let provenance = runtime_baseline_provenance();

        apply_runtime_row(&mut baseline, &row, &provenance);
        let metadata = refresh_runtime_baseline_metadata(&mut baseline, &[row], &provenance)
            .expect("metadata refresh should add warning");

        assert_eq!(metadata, BaselineMetadataRefresh::WarningInserted);
        assert!(baseline.get("stats").is_none());
        assert!(baseline.get("specs_jcs_sha256").is_none());
        assert_eq!(
            baseline[RUNTIME_METADATA_WARNING_FIELD]["promotion_ready"],
            json!(false)
        );
        assert_eq!(
            baseline[RUNTIME_METADATA_WARNING_FIELD]["specs_collected"],
            json!(1)
        );
        assert_eq!(
            baseline[RUNTIME_METADATA_WARNING_FIELD]["specs_refreshed"],
            json!(1)
        );
        assert_eq!(
            baseline["specs"]["RuntimeSpec"]["ty"]["runtime_seconds"],
            json!(1.0)
        );
    }

    #[test]
    fn sha256_jcs_value_normalizes_float_lexemes_for_matrix_metadata() {
        let value: Value = serde_json::from_str(
            r#"{
                "integral_float": 5.0,
                "small_float": 1e-06,
                "zero_float": 0.0
            }"#,
        )
        .expect("test JSON should parse");

        let digest = sha256_jcs_value(&value).expect("digest should compute");
        let expected = r#"{"integral_float":5,"small_float":1e-6,"zero_float":0}"#;
        let expected_digest = format!("{:x}", Sha256::digest(expected.as_bytes()));

        assert_eq!(digest, expected_digest);
    }

    #[test]
    fn markdown_reports_ranked_perf_losers_with_followups() {
        let summary = classify_baseline_str(
            r#"{
              "specs": {
                "MCReachabilityTestAllGraphs": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 6, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 10.0, "states": 6, "error_type": null},
                  "verified_match": true
                },
                "dijkstra-mutex_Safety-4-processors": {
                  "source": {},
                  "tlc": {"status": "pass", "runtime_seconds": 5.0, "states": 6, "error_type": null},
                  "ty": {"status": "pass", "runtime_seconds": 7.5, "states": 6, "error_type": null},
                  "verified_match": true
                }
              }
            }"#,
        )
        .unwrap();

        let markdown = summary.to_markdown();
        assert!(markdown.contains("## Ranked Perf Losers"));
        assert!(markdown.contains("alabsystems/ty#4391"));
        assert!(markdown.contains("alabsystems/ty#4392"));
        assert_eq!(
            row_for(&summary, "MCReachabilityTestAllGraphs").perf_loser_rank,
            Some(1)
        );
        assert_eq!(
            row_for(&summary, "dijkstra-mutex_Safety-4-processors").perf_loser_rank,
            Some(2)
        );
    }
}
