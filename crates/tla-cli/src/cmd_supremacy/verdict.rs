// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Rust policy verdict evaluator for `ty supremacy gate`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::matrix;
use super::policy::{PlannedGate, SelftestRequirement, TelemetryRequirement};
use super::runner::{
    COMMAND_SCOPED_ENV_KEYS, COMMAND_SCRATCH_DIR_NAME, DISK_SCOPE_CONTRACT_SCHEMA,
    DISK_USAGE_SAMPLE_INTERVAL, DISK_USAGE_SCAN_BUDGET, DISK_USAGE_SCAN_ENTRY_LIMIT,
};
use super::summary::SUMMARY_SCHEMA;
use super::PreparedSupremacy;

const VERDICT_SCHEMA: &str = "ty.single_thread_supremacy.policy_verdict.v1";
const MATRIX_SUMMARY_SCHEMA: &str = "ty.supremacy.matrix_summary.v1";
const FLAT_PRIMARY_REBUILD_MARKER: &str = "[compiled-bfs] clearing layout-sensitive compiled artifacts before rebuild: reason=flat_state_primary layout promotion";
const STRICT_SELFTEST_FALSE_RESULT_KINDS: &[&str] = &["invariant", "state_constraint"];
const COMMAND_ARTIFACT_SCHEMA: &str = "ty.supremacy.command.v4";
const TY_CACHE_DIR_ENV: &str = "TY_CACHE_DIR";
const MATRIX_STRICT_BLOCKER_COUNT_FIELDS: &[&str] = &[
    "unsupported",
    "tlc_error",
    "tlc_timeout",
    "runtime_to_error",
    "timeout_dominance",
    "ty_timeout",
    "parity_fail",
    "missing_runtime",
    "perf_tie",
    "perf_loser",
];
const OPTIONAL_ZERO_MATRIX_COUNT_FIELDS: &[&str] = &[
    "expected_violation_match",
    "runtime_to_error",
    "timeout_dominance",
];
const DEFAULT_MATRIX_BASELINE: &str = "tests/tlc_comparison/spec_baseline.json";

#[derive(Debug, Serialize)]
pub(super) struct PolicyVerdict {
    schema: &'static str,
    verdict: &'static str,
    gate_mode: Option<String>,
    expected_run_count: Option<usize>,
    errors: Vec<String>,
    policy_file: PathBuf,
    raw_benchmark_summary: SummaryReference,
    policy_fields: BTreeMap<&'static str, &'static str>,
    anti_overfit_evidence: AntiOverfitEvidence,
    required_trust_cg_env: BTreeMap<String, String>,
    generated_state_count_sources: BTreeMap<&'static str, &'static str>,
    planned_gate: Option<PlannedGate>,
}

#[derive(Debug, Serialize)]
struct SummaryReference {
    path: PathBuf,
}

#[derive(Debug, Serialize)]
struct AntiOverfitEvidence {
    launch_corpus: LaunchCorpusEvidence,
    engine_selection_contract: EngineSelectionEvidence,
    matrix_holdout: MatrixHoldoutEvidence,
    cold_single_thread_wall: ColdSingleThreadWallEvidence,
}

#[derive(Debug, Serialize)]
struct LaunchCorpusEvidence {
    specs: Vec<String>,
    spec_count: usize,
    role: &'static str,
}

#[derive(Debug, Serialize)]
struct EngineSelectionEvidence {
    selection_basis: String,
    forbidden_selector_inputs: Vec<String>,
    permitted_future_engines: Vec<String>,
}

#[derive(Debug, Serialize)]
struct MatrixHoldoutEvidence {
    baseline_path: PathBuf,
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_specs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    specs_jcs_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covers_more_than_launch_canary: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unavailable_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct ColdSingleThreadWallEvidence {
    required_runs: Option<usize>,
    required_workers: u64,
    artifact_cache_disabled: bool,
    native_fused_strict: bool,
    native_fused_local_dedup: bool,
    wall_clock_source: &'static str,
    trust_cg_execution_source: &'static str,
}

impl PolicyVerdict {
    fn passed(&self) -> bool {
        self.errors.is_empty()
    }
}

pub(super) fn evaluate_and_write(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
) -> Result<bool> {
    let verdict = evaluate(prepared, summary_path)?;
    let verdict_path = prepared.output_dir.join("policy_verdict.json");
    fs::write(
        &verdict_path,
        serde_json::to_string_pretty(&verdict).context("serialize policy verdict")? + "\n",
    )
    .with_context(|| format!("write {}", verdict_path.display()))?;

    let markdown_path = prepared.output_dir.join("policy_verdict.md");
    fs::write(&markdown_path, render_markdown(&verdict))
        .with_context(|| format!("write {}", markdown_path.display()))?;

    Ok(verdict.passed())
}

fn evaluate(prepared: &PreparedSupremacy, summary_path: &Path) -> Result<PolicyVerdict> {
    let text = fs::read_to_string(summary_path)
        .with_context(|| format!("read benchmark summary {}", summary_path.display()))?;
    let summary: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse benchmark summary {}", summary_path.display()))?;
    let mut errors = Vec::new();
    let gate_plan = prepared.gate_plan.as_ref();

    match admit_summary_for_launch(prepared, summary_path, &summary, &mut errors) {
        SummaryAdmission::Benchmark => {
            require_gate_flags(&summary, gate_plan, &mut errors);

            let rows_by_spec = rows_by_spec(&summary, &mut errors);
            require_benchmark_rows_match_policy(prepared, &rows_by_spec, &mut errors);
            for spec in &prepared.policy.specs {
                let Some(row) = rows_by_spec.get(spec.as_str()) else {
                    errors.push(format!("{spec}: summary row missing"));
                    continue;
                };
                evaluate_row(prepared, summary_path, gate_plan, spec, row, &mut errors);
            }
        }
        SummaryAdmission::Matrix => {}
    }

    let required_trust_cg_env = gate_plan
        .map(PlannedGate::enforce_required_env)
        .unwrap_or_default();
    let anti_overfit_evidence = anti_overfit_evidence(prepared, &required_trust_cg_env);
    Ok(PolicyVerdict {
        schema: VERDICT_SCHEMA,
        verdict: if errors.is_empty() { "pass" } else { "fail" },
        gate_mode: gate_plan.map(|plan| plan.gate_mode.clone()),
        expected_run_count: prepared.runs,
        errors,
        policy_file: prepared.policy_path.clone(),
        raw_benchmark_summary: SummaryReference {
            path: summary_path.to_path_buf(),
        },
        policy_fields: BTreeMap::from([
            ("generated_state_counts", "expected_generated_state_counts"),
            ("trust_cg_env", "gate_modes.*.required_trust_cg_env"),
        ]),
        anti_overfit_evidence,
        required_trust_cg_env,
        generated_state_count_sources: BTreeMap::from([
            ("tlc", "runs[].raw_successors_generated"),
            ("interp", "runs[].raw_successors_generated"),
            ("trust-cg", "runs[].raw_successors_generated"),
        ]),
        planned_gate: gate_plan.cloned(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SummaryAdmission {
    Benchmark,
    Matrix,
}

fn admit_summary_for_launch(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    summary: &Value,
    errors: &mut Vec<String>,
) -> SummaryAdmission {
    if is_matrix_summary(summary) {
        require_matrix_summary_launch_evidence(prepared, summary_path, summary, errors);
        SummaryAdmission::Matrix
    } else {
        require_benchmark_summary_launch_evidence(prepared, summary_path, summary, errors);
        SummaryAdmission::Benchmark
    }
}

fn is_matrix_summary(summary: &Value) -> bool {
    summary.get("schema").and_then(Value::as_str) == Some(MATRIX_SUMMARY_SCHEMA)
        || summary.get("strict_blockers").is_some()
        || summary
            .get("rows")
            .and_then(Value::as_array)
            .is_some_and(|rows| rows.iter().any(|row| row.get("class").is_some()))
}

fn require_benchmark_summary_launch_evidence(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    summary: &Value,
    errors: &mut Vec<String>,
) {
    match summary.get("schema").and_then(Value::as_str) {
        Some(SUMMARY_SCHEMA) => {}
        Some(other) => errors.push(format!(
            "summary.schema was {other:?}, expected {SUMMARY_SCHEMA:?}"
        )),
        None => errors.push(format!(
            "summary.schema missing; expected {SUMMARY_SCHEMA:?}"
        )),
    }
    require_non_empty_summary_string(summary, "timestamp", "measurement timestamp", errors);
    require_non_empty_summary_string(summary, "artifact_bundle", "artifact identity", errors);
    require_non_empty_summary_string(summary, "invocation", "launch invocation", errors);
    if !summary
        .get("backend_controls")
        .is_some_and(Value::is_object)
    {
        errors.push("summary.backend_controls missing or not an object".to_string());
    }
    require_backend_control_env(summary, prepared.gate_plan.as_ref(), errors);
    require_launch_controls(summary, prepared.gate_plan.as_ref(), errors);
    require_build_identity(prepared, summary_path, summary, errors);
    match summary
        .pointer("/build_identity/cargo_profile")
        .and_then(Value::as_str)
    {
        Some("release" | "release-canary") => {}
        Some(other) => errors.push(format!(
            "summary.build_identity.cargo_profile was {other:?}; enforce-mode summary evidence requires release or release-canary"
        )),
        None => errors.push(
            "summary.build_identity.cargo_profile missing; absent release build identity".to_string(),
        ),
    }
}

fn require_launch_controls(
    summary: &Value,
    gate_plan: Option<&PlannedGate>,
    errors: &mut Vec<String>,
) {
    if !summary.get("launch_controls").is_some_and(Value::is_object) {
        errors.push("summary.launch_controls missing or not an object".to_string());
        return;
    }
    require_usize_field(
        "summary.launch_controls.tlc.workers",
        summary.pointer("/launch_controls/tlc/workers"),
        1,
        errors,
    );
    let expected_jvm_args = super::tlc_java_single_thread_args().to_vec();
    require_string_array_field(
        "summary.launch_controls.tlc.jvm_args",
        summary.pointer("/launch_controls/tlc/jvm_args"),
        &expected_jvm_args,
        errors,
    );
    require_string_field(
        "summary.launch_controls.tlc.heap_xms",
        summary.pointer("/launch_controls/tlc/heap_xms"),
        "64m",
        errors,
    );
    require_string_field(
        "summary.launch_controls.tlc.heap_xmx",
        summary.pointer("/launch_controls/tlc/heap_xmx"),
        "4g",
        errors,
    );
    for mode in ["interp", "trust_cg"] {
        require_usize_field(
            &format!("summary.launch_controls.ty.{mode}.workers"),
            summary.pointer(&format!("/launch_controls/ty/{mode}/workers")),
            1,
            errors,
        );
    }
    let Some(plan) = gate_plan else {
        return;
    };
    let required_env = plan.enforce_required_env();
    if let Some(expected) = required_env.get("TY_DISABLE_ARTIFACT_CACHE") {
        require_string_field(
            "summary.launch_controls.ty.trust_cg.artifact_cache_disabled_env",
            summary.pointer("/launch_controls/ty/trust_cg/artifact_cache_disabled_env"),
            expected,
            errors,
        );
    }
    if let Some(expected) = required_env.get("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS") {
        require_string_field(
            "summary.launch_controls.ty.trust_cg.native_callout_compile_jobs",
            summary.pointer("/launch_controls/ty/trust_cg/native_callout_compile_jobs"),
            expected,
            errors,
        );
    }
    if summary
        .pointer("/launch_controls/ty/trust_cg/cache_dir")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        errors.push(
            "summary.launch_controls.ty.trust_cg.cache_dir missing or empty; absent cold-cache launch identity"
                .to_string(),
        );
    }
    if let Some(expected) = summary
        .pointer("/backend_controls/trust_cg_env")
        .and_then(Value::as_object)
        .and_then(|env| env.get(TY_CACHE_DIR_ENV))
        .and_then(Value::as_str)
    {
        require_string_field(
            "summary.launch_controls.ty.trust_cg.cache_dir",
            summary.pointer("/launch_controls/ty/trust_cg/cache_dir"),
            expected,
            errors,
        );
    }
}

fn require_usize_field(
    label: &str,
    value: Option<&Value>,
    expected: usize,
    errors: &mut Vec<String>,
) {
    match non_negative_integer_value(value) {
        Some(actual) if actual == expected as i64 => {}
        _ => errors.push(format!(
            "{label} was {}, expected {expected}",
            display_value(value)
        )),
    }
}

fn require_string_field(
    label: &str,
    value: Option<&Value>,
    expected: &str,
    errors: &mut Vec<String>,
) {
    match value.and_then(Value::as_str) {
        Some(actual) if actual == expected => {}
        _ => errors.push(format!(
            "{label} was {}, expected {expected:?}",
            display_value(value)
        )),
    }
}

fn require_string_array_field(
    label: &str,
    value: Option<&Value>,
    expected: &[&str],
    errors: &mut Vec<String>,
) {
    let Some(actual) = value.and_then(Value::as_array) else {
        errors.push(format!(
            "{label} was {}, expected {:?}",
            display_value(value),
            expected
        ));
        return;
    };
    let actual = actual.iter().map(Value::as_str).collect::<Option<Vec<_>>>();
    match actual {
        Some(actual) if actual == expected => {}
        _ => errors.push(format!(
            "{label} was {}, expected {:?}",
            display_value(value),
            expected
        )),
    }
}

fn require_backend_control_env(
    summary: &Value,
    gate_plan: Option<&PlannedGate>,
    errors: &mut Vec<String>,
) {
    let Some(plan) = gate_plan else {
        return;
    };
    let Some(env) = summary
        .pointer("/backend_controls/trust_cg_env")
        .and_then(Value::as_object)
    else {
        errors.push("summary.backend_controls.trust_cg_env missing or not an object".to_string());
        return;
    };
    require_env_map("summary.backend_controls.trust_cg_env", plan, env, errors);
}

fn require_matrix_summary_launch_evidence(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    summary: &Value,
    errors: &mut Vec<String>,
) {
    if prepared.gate_plan.is_some() {
        errors.push(
            "matrix summary artifacts are all-runnable diagnostic evidence and cannot satisfy the final launch gate; run `ty supremacy gate --mode enforce --gate-mode full-native-fused --runs 3` to produce benchmark launch evidence".to_string(),
        );
    }
    match summary.get("schema").and_then(Value::as_str) {
        Some(MATRIX_SUMMARY_SCHEMA) => {}
        Some(other) => errors.push(format!(
            "matrix summary schema was {other:?}, expected {MATRIX_SUMMARY_SCHEMA:?}"
        )),
        None => errors.push(format!(
            "matrix summary.schema missing; expected {MATRIX_SUMMARY_SCHEMA:?}"
        )),
    }
    require_build_identity(prepared, summary_path, summary, errors);
    match summary
        .pointer("/build_identity/allow_debug_runtime")
        .and_then(Value::as_bool)
    {
        Some(false) => {}
        Some(true) => {
            errors.push(
                "matrix summary build_identity.allow_debug_runtime was true; debug runtime evidence is not launch evidence"
                    .to_string(),
            );
        }
        None => errors.push(
            "matrix summary build_identity.allow_debug_runtime missing; launch evidence must explicitly be non-debug"
                .to_string(),
        ),
    }
    let missing_matrix_timestamp = summary
        .pointer("/build_identity/timestamp")
        .and_then(Value::as_str)
        .map(str::is_empty)
        .unwrap_or(true);
    if missing_matrix_timestamp {
        errors.push(
            "matrix summary build_identity.timestamp missing or empty; absent runtime evidence timestamp"
                .to_string(),
        );
    }
    let counts = summary.get("counts");
    let strict_blockers = matrix_strict_blocker_count(counts);
    require_matrix_strict_fields(summary, strict_blockers, errors);
    let policy_blockers = require_matrix_policy_blocker_counts(prepared, counts, errors);
    if let Some(policy_blockers) = policy_blockers {
        if policy_blockers != 0 {
            errors.push(format!(
                "matrix summary strict_blockers was {}, with {policy_blockers} policy blockers after allowed comparable outcomes; expected 0 policy blockers",
                display_value(summary.get("strict_blockers"))
            ));
        }
    }
    if summary.get("verdict").and_then(Value::as_str) != Some("pass") {
        errors.push(format!(
            "matrix summary verdict was {}, expected \"pass\"",
            display_value(summary.get("verdict"))
        ));
    }
    let Some(rows) = summary.get("rows").and_then(Value::as_array) else {
        errors.push("matrix summary.rows missing or not an array".to_string());
        return;
    };
    require_matrix_corpus_identity(prepared, summary, rows, errors);
    require_matrix_counts_match_rows(summary, rows, errors);
    require_matrix_row_specs_match_baseline(prepared, rows, errors);
    for row in rows
        .iter()
        .filter(|row| {
            !row.get("class")
                .and_then(Value::as_str)
                .is_some_and(|class| matrix_row_class_accepted_by_policy(prepared, class))
        })
        .take(10)
    {
        errors.push(format!(
            "matrix summary row {} has class {}; policy verdict requires pass, expected_violation_match, or an explicitly allowed comparable outcome",
            display_value(row.get("spec")),
            display_value(row.get("class"))
        ));
    }
}

fn require_matrix_strict_fields(
    summary: &Value,
    strict_blockers_from_counts: Option<u64>,
    errors: &mut Vec<String>,
) {
    match (
        summary.get("strict_pass").and_then(Value::as_bool),
        strict_blockers_from_counts,
    ) {
        (Some(strict_pass), Some(strict_blockers)) if strict_pass == (strict_blockers == 0) => {}
        (Some(strict_pass), Some(strict_blockers)) => errors.push(format!(
            "matrix summary strict_pass was {strict_pass}, but counts imply {}",
            strict_blockers == 0
        )),
        (Some(_), None) => {}
        (None, _) => errors.push(format!(
            "matrix summary strict_pass was {}, expected a boolean",
            display_value(summary.get("strict_pass"))
        )),
    }

    match (
        non_negative_u64_value(summary.get("strict_blockers")),
        strict_blockers_from_counts,
    ) {
        (Some(declared), Some(expected)) if declared == expected => {}
        (Some(declared), Some(expected)) => errors.push(format!(
            "matrix summary strict_blockers was {declared}, but counts imply {expected}"
        )),
        (Some(_), None) => {}
        (None, _) => errors.push(format!(
            "matrix summary strict_blockers was {}, expected a non-negative integer",
            display_value(summary.get("strict_blockers"))
        )),
    }
}

fn require_matrix_policy_blocker_counts(
    prepared: &PreparedSupremacy,
    counts: Option<&Value>,
    errors: &mut Vec<String>,
) -> Option<u64> {
    let mut policy_blockers = 0u64;
    let mut all_counts_valid = true;
    for field in MATRIX_STRICT_BLOCKER_COUNT_FIELDS {
        let count = matrix_count_value(counts, field);
        let Some(count) = count else {
            all_counts_valid = false;
            errors.push(format!(
                "matrix summary counts[{field}] was {}, expected a non-negative integer",
                display_value(counts.and_then(|counts| counts.get(*field)))
            ));
            continue;
        };
        if !matrix_count_field_accepted_by_policy(prepared, field) {
            policy_blockers += count;
            if count != 0 {
                errors.push(format!(
                    "matrix summary contains {field}={count}; matrix policy requires 0"
                ));
            }
        }
    }
    all_counts_valid.then_some(policy_blockers)
}

fn matrix_strict_blocker_count(counts: Option<&Value>) -> Option<u64> {
    MATRIX_STRICT_BLOCKER_COUNT_FIELDS
        .iter()
        .map(|field| matrix_count_value(counts, field))
        .sum()
}

fn matrix_count_field_accepted_by_policy(prepared: &PreparedSupremacy, field: &str) -> bool {
    match field {
        "runtime_to_error" => prepared.policy.matrix_policy.allow_runtime_to_error,
        "timeout_dominance" => prepared.policy.matrix_policy.allow_timeout_dominance,
        _ => false,
    }
}

fn matrix_row_class_accepted_by_policy(prepared: &PreparedSupremacy, class: &str) -> bool {
    match class {
        "pass" | "expected_violation_match" => true,
        "runtime_to_error" => prepared.policy.matrix_policy.allow_runtime_to_error,
        "timeout_dominance" => prepared.policy.matrix_policy.allow_timeout_dominance,
        _ => false,
    }
}

fn require_matrix_row_specs_match_baseline(
    prepared: &PreparedSupremacy,
    rows: &[Value],
    errors: &mut Vec<String>,
) {
    let baseline_path = matrix_baseline_path(prepared);
    let expected_summary = match matrix::classify_baseline_path_with_policy(
        &baseline_path,
        &prepared.policy.matrix_policy,
    ) {
        Ok(summary) => summary,
        Err(err) => {
            errors.push(format!(
                "matrix summary rows could not be verified against baseline {}: {err}",
                baseline_path.display()
            ));
            return;
        }
    };

    let expected_classes = expected_summary
        .rows
        .iter()
        .map(|row| (row.spec.clone(), matrix_class_value(row.class)))
        .collect::<BTreeMap<_, _>>();
    let expected_rows = expected_classes.keys().cloned().collect::<BTreeSet<_>>();
    let mut observed_counts = BTreeMap::<String, usize>::new();
    for row in rows {
        let Some(spec) = row.get("spec").and_then(Value::as_str) else {
            errors.push("matrix summary row missing string spec".to_string());
            continue;
        };
        *observed_counts.entry(spec.to_string()).or_default() += 1;
    }
    for (spec, count) in observed_counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .take(10)
    {
        errors.push(format!(
            "matrix summary contains duplicate row for spec {spec:?} ({count} copies)"
        ));
    }
    let observed_set = observed_counts.keys().cloned().collect::<BTreeSet<_>>();
    let missing = expected_rows
        .difference(&observed_set)
        .take(10)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        errors.push(format!(
            "matrix summary rows missing baseline specs: {}",
            missing.join(", ")
        ));
    }
    let unexpected = observed_set
        .difference(&expected_rows)
        .take(10)
        .cloned()
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        errors.push(format!(
            "matrix summary rows contain specs not present in baseline: {}",
            unexpected.join(", ")
        ));
    }

    let mut mismatch_count = 0usize;
    for row in rows {
        let Some(spec) = row.get("spec").and_then(Value::as_str) else {
            continue;
        };
        let Some(expected_class) = expected_classes.get(spec) else {
            continue;
        };
        let observed_class = row.get("class");
        if observed_class != Some(expected_class) {
            if mismatch_count < 10 {
                errors.push(format!(
                    "matrix summary row {spec:?} class was {}, but recomputed baseline policy class was {}",
                    display_value(row.get("class")),
                    display_value(Some(expected_class))
                ));
            }
            mismatch_count += 1;
        }
    }
}

fn matrix_class_value(class: matrix::SupremacyMatrixClass) -> Value {
    serde_json::to_value(class).expect("SupremacyMatrixClass serializes as JSON")
}

fn require_matrix_corpus_identity(
    prepared: &PreparedSupremacy,
    summary: &Value,
    rows: &[Value],
    errors: &mut Vec<String>,
) {
    let Some(corpus) = summary.get("corpus").and_then(Value::as_object) else {
        errors.push("matrix summary.corpus missing; absent corpus identity".to_string());
        return;
    };
    let total_specs = corpus
        .get("total_specs")
        .and_then(|value| non_negative_u64_value(Some(value)));
    match total_specs {
        Some(total) if total as usize == rows.len() => {}
        Some(total) => errors.push(format!(
            "matrix summary corpus.total_specs was {total}, but rows has {} entries",
            rows.len()
        )),
        None => errors.push(
            "matrix summary corpus.total_specs missing or invalid; absent corpus size".to_string(),
        ),
    }
    if rows.len() <= prepared.policy.specs.len() {
        errors.push(format!(
            "matrix summary rows has only {} entries; all-runnable matrix evidence must cover more than the {}-spec launch canary",
            rows.len(),
            prepared.policy.specs.len()
        ));
    }
    match corpus.get("specs_jcs_sha256").and_then(Value::as_str) {
        Some(value) if is_sha256_hex(value) => {
            require_matrix_corpus_matches_baseline(prepared, value, total_specs, errors);
        }
        Some(value) => errors.push(format!(
            "matrix summary corpus.specs_jcs_sha256 was {value:?}, expected a SHA-256 hex digest"
        )),
        None => errors.push(
            "matrix summary corpus.specs_jcs_sha256 missing; absent corpus digest".to_string(),
        ),
    }
}

fn require_matrix_corpus_matches_baseline(
    prepared: &PreparedSupremacy,
    summary_digest: &str,
    summary_total_specs: Option<u64>,
    errors: &mut Vec<String>,
) {
    let baseline_path = matrix_baseline_path(prepared);
    let baseline_identity = match matrix::enforceable_baseline_corpus_identity_path(&baseline_path)
    {
        Ok(identity) => identity,
        Err(err) => {
            errors.push(format!(
                "matrix summary corpus could not be verified against enforceable baseline {}: {err}",
                baseline_path.display()
            ));
            return;
        }
    };
    let baseline_digest = baseline_identity.specs_jcs_sha256.as_deref().unwrap_or("");
    if baseline_digest != summary_digest {
        errors.push(format!(
            "matrix summary corpus.specs_jcs_sha256 was {summary_digest:?}, but baseline {} has {baseline_digest:?}",
            baseline_path.display()
        ));
    }
    let baseline_total = baseline_identity.total_specs as u64;
    if Some(baseline_total) != summary_total_specs {
        errors.push(format!(
            "matrix summary corpus.total_specs was {summary_total_specs:?}, but baseline {} has {baseline_total}",
            baseline_path.display()
        ));
    }
}

fn matrix_baseline_path(prepared: &PreparedSupremacy) -> PathBuf {
    let relative = PathBuf::from(DEFAULT_MATRIX_BASELINE);
    if let Some(repo_root) = policy_repo_root(prepared) {
        let rooted = repo_root.join(DEFAULT_MATRIX_BASELINE);
        if rooted.exists() {
            return rooted;
        }
    }
    relative
}

fn anti_overfit_evidence(
    prepared: &PreparedSupremacy,
    required_trust_cg_env: &BTreeMap<String, String>,
) -> AntiOverfitEvidence {
    let (selection_basis, forbidden_selector_inputs, permitted_future_engines) = prepared
        .policy
        .engine_selection_contract
        .as_ref()
        .map(|contract| {
            (
                contract.selection_basis.clone(),
                contract.forbidden_selector_inputs.clone(),
                contract.permitted_future_engines.clone(),
            )
        })
        .unwrap_or_else(|| ("absent".to_string(), Vec::new(), Vec::new()));
    AntiOverfitEvidence {
        launch_corpus: LaunchCorpusEvidence {
            specs: prepared.policy.specs.clone(),
            spec_count: prepared.policy.specs.len(),
            role: "cold launch canary",
        },
        engine_selection_contract: EngineSelectionEvidence {
            selection_basis,
            forbidden_selector_inputs,
            permitted_future_engines,
        },
        matrix_holdout: matrix_holdout_evidence(prepared),
        cold_single_thread_wall: cold_single_thread_wall_evidence(prepared, required_trust_cg_env),
    }
}

fn matrix_holdout_evidence(prepared: &PreparedSupremacy) -> MatrixHoldoutEvidence {
    let baseline_path = matrix_baseline_path(prepared);
    match matrix::enforceable_baseline_corpus_identity_path(&baseline_path) {
        Ok(identity) => MatrixHoldoutEvidence {
            baseline_path,
            role: "diagnostic holdout; cannot satisfy final launch gate",
            total_specs: Some(identity.total_specs),
            specs_jcs_sha256: identity.specs_jcs_sha256,
            covers_more_than_launch_canary: Some(
                identity.total_specs > prepared.policy.specs.len(),
            ),
            unavailable_reason: None,
        },
        Err(err) => MatrixHoldoutEvidence {
            baseline_path,
            role: "diagnostic holdout; cannot satisfy final launch gate",
            total_specs: None,
            specs_jcs_sha256: None,
            covers_more_than_launch_canary: None,
            unavailable_reason: Some(err.to_string()),
        },
    }
}

fn cold_single_thread_wall_evidence(
    prepared: &PreparedSupremacy,
    required_trust_cg_env: &BTreeMap<String, String>,
) -> ColdSingleThreadWallEvidence {
    ColdSingleThreadWallEvidence {
        required_runs: prepared.runs,
        required_workers: 1,
        artifact_cache_disabled: required_trust_cg_env
            .get("TY_DISABLE_ARTIFACT_CACHE")
            .is_some_and(|value| value == "1"),
        native_fused_strict: required_trust_cg_env
            .get("TY_TRUST_CG_NATIVE_FUSED_STRICT")
            .is_some_and(|value| value == "1"),
        native_fused_local_dedup: required_trust_cg_env
            .get("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP")
            .is_some_and(|value| value == "1"),
        wall_clock_source: "runs[].elapsed_seconds median with workers=1",
        trust_cg_execution_source: "compiled_bfs_execution_nanos median",
    }
}

fn require_matrix_counts_match_rows(summary: &Value, rows: &[Value], errors: &mut Vec<String>) {
    let Some(counts) = summary.get("counts").and_then(Value::as_object) else {
        errors.push("matrix summary.counts missing or not an object".to_string());
        return;
    };
    let mut observed = BTreeMap::<String, usize>::new();
    for row in rows {
        let class = row
            .get("class")
            .and_then(Value::as_str)
            .unwrap_or("<missing>")
            .to_string();
        *observed.entry(class).or_default() += 1;
    }
    let mut declared_total = 0usize;
    for (class, value) in counts {
        let Some(count) = non_negative_integer_value(Some(value)) else {
            errors.push(format!(
                "matrix summary counts[{class}] was {}, expected a non-negative integer",
                display_value(Some(value))
            ));
            continue;
        };
        declared_total += count as usize;
        let observed_count = observed.get(class).copied().unwrap_or_default();
        if observed_count != count as usize {
            errors.push(format!(
                "matrix summary counts[{class}] was {count}, but rows contain {observed_count}"
            ));
        }
    }
    if declared_total != rows.len() {
        errors.push(format!(
            "matrix summary counts total was {declared_total}, but rows has {} entries",
            rows.len()
        ));
    }
}

fn matrix_count_value(counts: Option<&Value>, field: &str) -> Option<u64> {
    let value = counts.and_then(|counts| counts.get(field));
    if value.is_none() && OPTIONAL_ZERO_MATRIX_COUNT_FIELDS.contains(&field) {
        return Some(0);
    }
    value.and_then(|value| non_negative_u64_value(Some(value)))
}

fn require_non_empty_summary_string(
    summary: &Value,
    field: &str,
    identity: &str,
    errors: &mut Vec<String>,
) {
    if summary
        .get(field)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty())
    {
        return;
    }
    errors.push(format!(
        "summary.{field} missing or empty; absent {identity}"
    ));
}

fn require_build_identity(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    summary: &Value,
    errors: &mut Vec<String>,
) {
    let Some(identity) = summary.get("build_identity").and_then(Value::as_object) else {
        errors.push("summary.build_identity missing; absent binary/build identity".to_string());
        return;
    };
    let summary_git = summary
        .get("git_commit")
        .and_then(Value::as_str)
        .or_else(|| identity.get("git_commit").and_then(Value::as_str));
    match summary_git {
        Some(value) if !value.is_empty() && value != "unknown" => {
            if value.len() < 7 {
                errors.push(format!(
                    "summary git_commit {value:?} is too short; expected at least 7 hex characters"
                ));
            }
            match current_git_commit(prepared) {
                Some(current) if git_commit_matches(value, &current) => {}
                Some(current) => errors.push(format!(
                    "summary git_commit {value:?} is stale for current checkout {current:?}; rerun launch evidence on this build"
                )),
                None => errors.push(
                    "current git commit could not be resolved; refusing summary-json launch evidence"
                        .to_string(),
                ),
            }
        }
        _ => {
            errors.push("summary git_commit missing or unknown; absent build identity".to_string())
        }
    }
    let binary_path = identity.get("ty_binary_path").and_then(Value::as_str);
    let resolved_binary_path = match binary_path {
        Some(value) if !value.is_empty() => {
            Some(resolve_summary_artifact_path(prepared, summary_path, value))
        }
        _ => {
            errors.push(
                "summary.build_identity.ty_binary_path missing or empty; absent binary identity"
                    .to_string(),
            );
            None
        }
    };
    match identity.get("ty_binary_sha256").and_then(Value::as_str) {
        Some(value) if is_sha256_hex(value) => {
            if let Some(path) = resolved_binary_path {
                match sha256_file(&path) {
                    Ok(actual) if actual == value => {}
                    Ok(actual) => errors.push(format!(
                        "summary.build_identity.ty_binary_sha256 was {value:?}, but {} hashes to {actual:?}",
                        path.display()
                    )),
                    Err(err) => errors.push(format!(
                        "summary.build_identity.ty_binary_path {} could not be hashed: {err}",
                        path.display()
                    )),
                }
            }
        }
        Some(value) => errors.push(format!(
            "summary.build_identity.ty_binary_sha256 was {value:?}, expected a SHA-256 hex digest"
        )),
        None => errors.push(
            "summary.build_identity.ty_binary_sha256 missing; absent binary identity".to_string(),
        ),
    }
}

fn resolve_summary_artifact_path(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    raw_path: &str,
) -> PathBuf {
    let path = PathBuf::from(raw_path);
    if path.is_absolute() {
        return path;
    }
    let summary_base = summary_path.parent().unwrap_or_else(|| Path::new("."));
    let mut candidates = Vec::new();
    if let Some(repo_root) = policy_repo_root(prepared) {
        candidates.push(repo_root.join(&path));
    }
    candidates.push(summary_base.join(&path));
    candidates
        .iter()
        .find(|candidate| candidate.exists())
        .cloned()
        .unwrap_or_else(|| summary_base.join(path))
}

fn current_git_commit(prepared: &PreparedSupremacy) -> Option<String> {
    // `PATH` is process-global. Coordinate this reader with the crate's
    // restore-on-exit environment editors so a concurrent fake-tool test (or
    // other scoped launcher setup) cannot make `git` transiently disappear.
    let _env_lock = crate::env_guard::lock_env();
    let repo_root = policy_repo_root(prepared).unwrap_or_else(|| PathBuf::from("."));
    // Large parallel test or gate processes can briefly exhaust the host's
    // process slots. Do not turn that transient `spawn(2)` condition into an
    // "unknown" build identity. Retry only interrupt/resource errors; a real
    // Git failure (not a repository, invalid HEAD, permission error) remains an
    // immediate fail-closed `None`.
    for attempt in 0..3 {
        match Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&repo_root)
            .output()
        {
            Ok(output) if output.status.success() => {
                let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
                return (!value.is_empty()).then_some(value);
            }
            Ok(_) => return None,
            Err(error)
                if attempt < 2
                    && matches!(
                        error.kind(),
                        std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock
                    ) =>
            {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(_) => return None,
        }
    }
    None
}

fn git_commit_matches(summary: &str, current: &str) -> bool {
    summary == current || (summary.len() >= 7 && current.starts_with(summary))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_file(path: &Path) -> Result<String> {
    fs::read(path)
        .map(|bytes| sha256_bytes(&bytes))
        .with_context(|| format!("read {}", path.display()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn require_gate_flags(summary: &Value, gate_plan: Option<&PlannedGate>, errors: &mut Vec<String>) {
    let Some(gate_plan) = gate_plan else {
        return;
    };
    let Some(flags) = summary.get("gate_flags").and_then(Value::as_object) else {
        errors.push("summary.gate_flags missing or not an object".to_string());
        return;
    };
    for flag in &gate_plan.benchmark_flags {
        if flags.get(flag).and_then(Value::as_bool) != Some(true) {
            errors.push(format!("required benchmark flag was not enabled: {flag}"));
        }
    }
    for flag in &gate_plan.forbidden_benchmark_flags {
        if flags.get(flag).and_then(Value::as_bool) != Some(false) {
            errors.push(format!("forbidden benchmark flag was not disabled: {flag}"));
        }
    }
}

fn rows_by_spec<'a>(summary: &'a Value, errors: &mut Vec<String>) -> BTreeMap<&'a str, &'a Value> {
    let mut rows = BTreeMap::new();
    let Some(items) = summary.get("rows").and_then(Value::as_array) else {
        errors.push("summary.rows missing or not an array".to_string());
        return rows;
    };
    for row in items {
        let Some(spec) = row.get("spec").and_then(Value::as_str) else {
            errors.push("summary row missing string spec".to_string());
            continue;
        };
        if rows.insert(spec, row).is_some() {
            errors.push(format!("{spec}: duplicate summary row"));
        }
    }
    rows
}

fn require_benchmark_rows_match_policy(
    prepared: &PreparedSupremacy,
    rows_by_spec: &BTreeMap<&str, &Value>,
    errors: &mut Vec<String>,
) {
    let expected = prepared
        .policy
        .specs
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let observed = rows_by_spec.keys().copied().collect::<BTreeSet<_>>();
    let missing = expected.difference(&observed).copied().collect::<Vec<_>>();
    if !missing.is_empty() {
        errors.push(format!(
            "benchmark summary rows missing pinned policy specs: {}",
            missing.join(", ")
        ));
    }
    let extra = observed.difference(&expected).copied().collect::<Vec<_>>();
    if !extra.is_empty() {
        errors.push(format!(
            "benchmark summary rows contain unpinned spec(s): {}; final launch evidence must be exactly the policy corpus",
            extra.join(", ")
        ));
    }
}

fn evaluate_row(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    gate_plan: Option<&PlannedGate>,
    spec: &str,
    row: &Value,
    errors: &mut Vec<String>,
) {
    let expected_states = prepared.policy.expected_state_counts.get(spec).copied();
    let expected_generated = prepared
        .policy
        .expected_generated_state_counts
        .get(spec)
        .copied();

    require_no_row_gate_failures(spec, row, errors);

    let tlc = evaluate_mode(
        prepared,
        summary_path,
        gate_plan,
        spec,
        "tlc",
        row.get("tlc"),
        prepared.runs,
        expected_states,
        expected_generated,
        errors,
    );
    let interp = evaluate_mode(
        prepared,
        summary_path,
        gate_plan,
        spec,
        "interp",
        row.get("interp"),
        prepared.runs,
        expected_states,
        expected_generated,
        errors,
    );
    let trust_cg = evaluate_mode(
        prepared,
        summary_path,
        gate_plan,
        spec,
        "trust-cg",
        row.get("trust_cg"),
        prepared.runs,
        expected_states,
        expected_generated,
        errors,
    );

    if let Some(plan) = gate_plan {
        require_state_parity_flags(spec, row, errors);
        require_generated_parity(spec, plan, &tlc, &interp, &trust_cg, errors);
        require_trust_cg_runs(prepared, summary_path, plan, spec, row, errors);
    }

    if let Some(thresholds) = prepared.policy.thresholds.get(spec) {
        let skip_interpreter_thresholds =
            gate_plan.is_some_and(|plan| plan.gate_mode == "full_native_fused");
        if let (Some(tlc_median), Some(interp_median)) = (tlc.median, interp.median) {
            if let Some(min) = thresholds.min_speedup_interp_vs_tlc {
                let speedup = tlc_median / interp_median;
                if !skip_interpreter_thresholds && speedup <= min {
                    errors.push(format!(
                        "{spec}: speedup_interp_vs_tlc was {speedup:.6}, expected > {min:.6}"
                    ));
                }
            }
        }
        if let (Some(tlc_median), Some(trust_cg_median)) = (tlc.median, trust_cg.median) {
            if let Some(min) = thresholds.min_speedup_trust_cg_vs_tlc {
                let speedup = tlc_median / trust_cg_median;
                require_advertised_speedup(spec, row, "speedup_trust_cg_vs_tlc", speedup, errors);
                if speedup <= min {
                    errors.push(format!(
                        "{spec}: speedup_trust_cg_vs_tlc was {speedup:.6}, expected > {min:.6}"
                    ));
                }
            }
        }
        if let (Some(interp_median), Some(trust_cg_median)) = (interp.median, trust_cg.median) {
            if let Some(min) = thresholds.min_trust_cg_vs_interp_ratio {
                let ratio = interp_median / trust_cg_median;
                if !skip_interpreter_thresholds && ratio <= min {
                    errors.push(format!(
                        "{spec}: trust_cg_vs_interp_ratio was {ratio:.6}, expected > {min:.6}"
                    ));
                }
            }
        }
    }
}

fn require_no_row_gate_failures(spec: &str, row: &Value, errors: &mut Vec<String>) {
    let Some(failures) = row.get("trust_cg_gate_failures") else {
        errors.push(format!("{spec}: trust_cg_gate_failures missing"));
        return;
    };
    let Some(failures) = failures.as_array() else {
        errors.push(format!(
            "{spec}: trust_cg_gate_failures was {}, expected an array",
            display_value(row.get("trust_cg_gate_failures"))
        ));
        return;
    };
    if !failures.is_empty() {
        errors.push(format!(
            "{spec}: trust_cg_gate_failures contains {} failure(s): {}",
            failures.len(),
            failures
                .iter()
                .map(|failure| display_value(Some(failure)))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
}

fn require_state_parity_flags(spec: &str, row: &Value, errors: &mut Vec<String>) {
    if row.get("parity_interp_vs_tlc").and_then(Value::as_bool) != Some(true) {
        errors.push(format!("{spec}: interp parity drift vs TLC"));
    }
    if row.get("parity_trust_cg_vs_tlc").and_then(Value::as_bool) != Some(true) {
        errors.push(format!("{spec}: trust-cg parity drift vs TLC"));
    }
}

#[derive(Default)]
struct ModeFacts {
    median: Option<f64>,
    generated_by_run: BTreeMap<u64, RawGeneratedCounts>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RawGeneratedCounts {
    initial: u64,
    successors: u64,
    total: u64,
}

fn evaluate_mode(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    gate_plan: Option<&PlannedGate>,
    spec: &str,
    mode_name: &str,
    mode: Option<&Value>,
    expected_run_count: Option<usize>,
    expected_states: Option<u64>,
    expected_generated: Option<u64>,
    errors: &mut Vec<String>,
) -> ModeFacts {
    let mut facts = ModeFacts::default();
    let Some(mode) = mode else {
        errors.push(format!("{spec}: {mode_name} summary missing"));
        return facts;
    };
    if mode.get("all_ok").and_then(Value::as_bool) != Some(true) {
        errors.push(format!("{spec}: {mode_name} run failed"));
    }
    let Some(runs) = mode.get("runs").and_then(Value::as_array) else {
        errors.push(format!("{spec}: {mode_name}.runs missing or not an array"));
        return facts;
    };

    let mut elapsed = Vec::new();
    let mut indexes = Vec::new();
    let mut seen_indexes = BTreeSet::new();
    for (position, run) in runs.iter().enumerate() {
        let label = run_label(run, position);
        let run_index = integer_field(run, "run_index");
        if let Some(index) = run_index {
            if !seen_indexes.insert(index) {
                errors.push(format!(
                    "{spec}: {mode_name} run_index {index} was duplicated"
                ));
            }
            indexes.push(index);
        } else {
            errors.push(format!(
                "{spec}: {mode_name} successful run at position {}: run_index was {}, expected an integer",
                position + 1,
                display_value(run.get("run_index")),
            ));
        }

        if gate_plan.is_some() {
            require_run_identity(spec, mode_name, &label, run, errors);
            require_command_artifact(prepared, summary_path, spec, mode_name, &label, run, errors);
        }
        require_integer_equals(spec, mode_name, &label, run, "returncode", 0, errors);
        require_integer_equals(spec, mode_name, &label, run, "workers", 1, errors);
        if !run.get("error").map(Value::is_null).unwrap_or(true) {
            errors.push(format!(
                "{spec}: {mode_name} {label}: error was {}, expected null/missing",
                display_value(run.get("error"))
            ));
        }
        if let Some(expected) = expected_states {
            require_integer_equals(
                spec,
                mode_name,
                &label,
                run,
                "states_found",
                expected,
                errors,
            );
        }
        if let Some(expected) = expected_generated {
            require_integer_equals(
                spec,
                mode_name,
                &label,
                run,
                "raw_successors_generated",
                expected,
                errors,
            );
        }
        if gate_plan.is_some() {
            if let (Some(index), Some(generated)) = (
                run_index.and_then(|value| u64::try_from(value).ok()),
                require_raw_generated_counts(spec, mode_name, &label, run, errors),
            ) {
                facts.generated_by_run.insert(index, generated);
            }
        }
        match finite_float_field(run, "elapsed_seconds") {
            Some(value) if value >= 0.0 => elapsed.push(value),
            _ => errors.push(format!(
                "{spec}: {mode_name} {label}: elapsed_seconds was {}, expected a non-negative finite number",
                display_value(run.get("elapsed_seconds"))
            )),
        }
    }

    if let Some(expected) = expected_run_count {
        let expected_indexes = (1..=expected as i64).collect::<Vec<_>>();
        if indexes != expected_indexes {
            errors.push(format!(
                "{spec}: {mode_name} successful run_index values were {indexes:?}, expected sequential {expected_indexes:?}"
            ));
        }
    }

    facts.median = median(&mut elapsed);
    match mode.get("median_seconds") {
        Some(advertised) => match (facts.median, advertised.as_f64()) {
            (Some(actual), Some(value)) if close(actual, value) => {}
            (Some(actual), _) => errors.push(format!(
                "{spec}: {mode_name} advertised median_seconds {} did not match recomputed median {actual:?}",
                display_value(Some(advertised))
            )),
            (None, _) if advertised.is_null() => {}
            (None, _) => errors.push(format!(
                "{spec}: {mode_name} advertised median_seconds {} did not match recomputed median None",
                display_value(Some(advertised))
            )),
        },
        None => errors.push(format!(
            "{spec}: {mode_name} advertised median_seconds was missing; wall-clock median evidence is required"
        )),
    }
    facts
}

fn require_raw_generated_counts(
    spec: &str,
    mode_name: &str,
    label: &str,
    run: &Value,
    errors: &mut Vec<String>,
) -> Option<RawGeneratedCounts> {
    let read = |field: &str, errors: &mut Vec<String>| {
        non_negative_u64_value(run.get(field)).or_else(|| {
            errors.push(format!(
                "{spec}: {mode_name} {label}: {field} was {}, expected a non-negative integer",
                display_value(run.get(field))
            ));
            None
        })
    };
    let initial = read("raw_initial_states_generated", errors);
    let successors = read("raw_successors_generated", errors);
    let total = read("states_generated", errors);
    let (Some(initial), Some(successors), Some(total)) = (initial, successors, total) else {
        return None;
    };
    match initial.checked_add(successors) {
        Some(recomputed) if recomputed == total => Some(RawGeneratedCounts {
            initial,
            successors,
            total,
        }),
        Some(recomputed) => {
            errors.push(format!(
                "{spec}: {mode_name} {label}: states_generated was {total}, expected raw_initial_states_generated + raw_successors_generated = {recomputed}"
            ));
            None
        }
        None => {
            errors.push(format!(
                "{spec}: {mode_name} {label}: raw generated-state counts overflowed while recomputing the total"
            ));
            None
        }
    }
}

fn require_run_identity(
    spec: &str,
    mode_name: &str,
    label: &str,
    run: &Value,
    errors: &mut Vec<String>,
) {
    let expected_tool = if mode_name == "tlc" { "tlc" } else { "ty" };
    require_run_string_field(spec, mode_name, label, run, "tool", expected_tool, errors);
    require_run_string_field(spec, mode_name, label, run, "spec_name", spec, errors);
    if mode_name != "tlc" {
        require_run_string_field(spec, mode_name, label, run, "mode", mode_name, errors);
    }
}

fn require_run_string_field(
    spec: &str,
    mode_name: &str,
    label: &str,
    run: &Value,
    field: &str,
    expected: &str,
    errors: &mut Vec<String>,
) {
    match run.get(field).and_then(Value::as_str) {
        Some(actual) if actual == expected => {}
        _ => errors.push(format!(
            "{spec}: {mode_name} {label}: {field} was {}, expected {expected:?}",
            display_value(run.get(field))
        )),
    }
}

fn require_command_artifact(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    spec: &str,
    mode_name: &str,
    label: &str,
    run: &Value,
    errors: &mut Vec<String>,
) {
    let Some(artifact_dir) = run.get("artifact_dir").and_then(Value::as_str) else {
        errors.push(format!("{spec}: {mode_name} {label}: artifact_dir missing"));
        return;
    };
    let artifact_dir = resolve_artifact_dir(prepared, summary_path, artifact_dir);
    let command_path = artifact_dir.join("command.json");
    let command = match fs::read_to_string(&command_path)
        .with_context(|| format!("read {}", command_path.display()))
        .and_then(|text| {
            serde_json::from_str::<Value>(&text)
                .with_context(|| format!("parse {}", command_path.display()))
        }) {
        Ok(command) => command,
        Err(err) => {
            errors.push(format!(
                "{spec}: {mode_name} {label}: command artifact unavailable: {err:#}"
            ));
            return;
        }
    };
    match command.get("schema").and_then(Value::as_str) {
        Some(COMMAND_ARTIFACT_SCHEMA) => {}
        Some(other) => errors.push(format!(
            "{spec}: {mode_name} {label}: command schema was {other:?}, expected {COMMAND_ARTIFACT_SCHEMA:?}"
        )),
        None => errors.push(format!(
            "{spec}: {mode_name} {label}: command schema missing, expected {COMMAND_ARTIFACT_SCHEMA:?}"
        )),
    }
    match command.get("cwd").and_then(Value::as_str) {
        Some(value) if !value.is_empty() => {}
        _ => errors.push(format!(
            "{spec}: {mode_name} {label}: command cwd was {}, expected a non-empty string",
            display_value(command.get("cwd"))
        )),
    }
    require_command_disk_evidence(spec, mode_name, label, &artifact_dir, &command, errors);
    require_command_returncode(spec, mode_name, label, run, &command, errors);
    require_command_env(spec, mode_name, label, run, &command, errors);
    require_command_argv(spec, mode_name, label, &command, errors);
}

fn require_command_disk_evidence(
    spec: &str,
    mode_name: &str,
    label: &str,
    artifact_dir: &Path,
    command: &Value,
    errors: &mut Vec<String>,
) {
    let context = format!("{spec}: {mode_name} {label}: command");
    let Some(resource) = command.get("resource_evidence").and_then(Value::as_object) else {
        errors.push(format!(
            "{context} resource_evidence missing or not an object"
        ));
        return;
    };
    require_exact_bool(
        &context,
        resource.get("strict_qualified"),
        "resource_evidence.strict_qualified",
        true,
        errors,
    );
    require_empty_array(
        &context,
        resource.get("qualification_failures"),
        "resource_evidence.qualification_failures",
        errors,
    );

    let Some(disk) = resource.get("disk").and_then(Value::as_object) else {
        errors.push(format!(
            "{context} resource_evidence.disk missing or not an object"
        ));
        return;
    };

    for (field, expected) in [
        ("contract_schema", DISK_SCOPE_CONTRACT_SCHEMA),
        ("scope", "command_artifact_and_scratch_tree"),
        ("method", "recursive_filesystem_metadata_polling"),
        ("sampling_execution", "inline_runner_poll_loop"),
    ] {
        require_exact_string(
            &context,
            disk.get(field),
            &format!("disk.{field}"),
            expected,
            errors,
        );
    }
    require_exact_bool(
        &context,
        disk.get("peak_exact"),
        "disk.peak_exact",
        false,
        errors,
    );
    require_exact_bool(
        &context,
        disk.get("sampling_can_perturb_elapsed"),
        "disk.sampling_can_perturb_elapsed",
        true,
        errors,
    );

    let sampling_interval_ms =
        u64::try_from(DISK_USAGE_SAMPLE_INTERVAL.as_millis()).expect("interval fits u64");
    let scan_budget_ms =
        u64::try_from(DISK_USAGE_SCAN_BUDGET.as_millis()).expect("budget fits u64");
    for (field, expected) in [
        ("sampling_interval_ms", sampling_interval_ms),
        ("scan_budget_ms", scan_budget_ms),
        ("scan_entry_limit", DISK_USAGE_SCAN_ENTRY_LIMIT),
        ("samples_partial", 0),
    ] {
        require_exact_u64(
            &context,
            disk.get(field),
            &format!("disk.{field}"),
            expected,
            errors,
        );
    }

    let total_scan_nanoseconds = require_u64(
        &context,
        disk.get("total_scan_nanoseconds"),
        "disk.total_scan_nanoseconds",
        errors,
    );
    let max_scan_nanoseconds = require_u64(
        &context,
        disk.get("max_scan_nanoseconds"),
        "disk.max_scan_nanoseconds",
        errors,
    );
    if let (Some(total), Some(maximum)) = (total_scan_nanoseconds, max_scan_nanoseconds) {
        if maximum > total {
            errors.push(format!(
                "{context} disk.max_scan_nanoseconds was {maximum}, greater than total_scan_nanoseconds {total}"
            ));
        }
    }
    let samples_attempted = require_u64(
        &context,
        disk.get("samples_attempted"),
        "disk.samples_attempted",
        errors,
    );
    let samples_complete = require_u64(
        &context,
        disk.get("samples_complete"),
        "disk.samples_complete",
        errors,
    );
    if let Some(attempted) = samples_attempted {
        if attempted < 2 {
            errors.push(format!(
                "{context} disk.samples_attempted was {attempted}, expected at least 2"
            ));
        }
        if samples_complete != Some(attempted) {
            errors.push(format!(
                "{context} disk.samples_complete was {samples_complete:?}, expected {attempted}"
            ));
        }
    }
    for field in [
        "peak_allocated_bytes",
        "peak_apparent_bytes",
        "peak_entries_observed",
    ] {
        let value = require_u64(&context, disk.get(field), &format!("disk.{field}"), errors);
        if field == "peak_entries_observed" && value == Some(0) {
            errors.push(format!(
                "{context} disk.peak_entries_observed was 0, expected a positive integer"
            ));
        }
    }

    for field in [
        "initial_sample_complete",
        "final_sample_complete",
        "setup_complete",
        "environment_confinement_complete",
        "scope_identity_stable",
        "ownership_verified",
        "accounting_complete",
        "polling_complete",
        "process_tree_lifetime_complete",
        "complete",
        "strict_qualified",
    ] {
        require_exact_bool(
            &context,
            disk.get(field),
            &format!("disk.{field}"),
            true,
            errors,
        );
    }
    require_empty_array(
        &context,
        disk.get("diagnostics"),
        "disk.diagnostics",
        errors,
    );
    require_empty_array(
        &context,
        disk.get("qualification_failures"),
        "disk.qualification_failures",
        errors,
    );

    let canonical_artifact =
        require_canonical_directory(&context, artifact_dir, "artifact directory", errors);
    let Some(canonical_artifact) = canonical_artifact else {
        return;
    };
    let canonical_scope = canonical_artifact.display().to_string();
    require_exact_string(
        &context,
        disk.get("scope_root"),
        "disk.scope_root",
        &canonical_scope,
        errors,
    );

    let scratch = canonical_artifact.join(COMMAND_SCRATCH_DIR_NAME);
    let canonical_scratch =
        require_canonical_directory(&context, &scratch, "command scratch directory", errors);
    let Some(canonical_scratch) = canonical_scratch else {
        return;
    };
    if canonical_scratch.parent() != Some(canonical_artifact.as_path()) {
        errors.push(format!(
            "{context} command scratch directory was not a direct child of the canonical artifact directory"
        ));
    }
    let canonical_scratch_text = canonical_scratch.display().to_string();
    require_exact_string(
        &context,
        disk.get("scratch_root"),
        "disk.scratch_root",
        &canonical_scratch_text,
        errors,
    );

    let Some(confinement) = disk
        .get("environment_confinement")
        .and_then(Value::as_object)
    else {
        errors.push(format!(
            "{context} disk.environment_confinement missing or not an object"
        ));
        return;
    };
    let expected_keys = COMMAND_SCOPED_ENV_KEYS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let actual_keys = confinement
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual_keys != expected_keys {
        errors.push(format!(
            "{context} disk.environment_confinement keys were {actual_keys:?}, expected {expected_keys:?}"
        ));
    }
    for key in COMMAND_SCOPED_ENV_KEYS {
        require_exact_string(
            &context,
            confinement.get(*key),
            &format!("disk.environment_confinement.{key}"),
            &canonical_scratch_text,
            errors,
        );
    }
}

fn require_canonical_directory(
    context: &str,
    path: &Path,
    description: &str,
    errors: &mut Vec<String>,
) -> Option<PathBuf> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            errors.push(format!(
                "{context} {description} {} was a symlink",
                path.display()
            ));
            return None;
        }
        Ok(metadata) if !metadata.file_type().is_dir() => {
            errors.push(format!(
                "{context} {description} {} was not a directory",
                path.display()
            ));
            return None;
        }
        Ok(_) => {}
        Err(err) => {
            errors.push(format!(
                "{context} could not inspect {description} {}: {err}",
                path.display()
            ));
            return None;
        }
    }
    match fs::canonicalize(path) {
        Ok(canonical) => Some(canonical),
        Err(err) => {
            errors.push(format!(
                "{context} could not canonicalize {description} {}: {err}",
                path.display()
            ));
            None
        }
    }
}

fn require_exact_string(
    context: &str,
    value: Option<&Value>,
    field: &str,
    expected: &str,
    errors: &mut Vec<String>,
) {
    if value.and_then(Value::as_str) != Some(expected) {
        errors.push(format!(
            "{context} {field} was {}, expected {expected:?}",
            display_value(value)
        ));
    }
}

fn require_exact_bool(
    context: &str,
    value: Option<&Value>,
    field: &str,
    expected: bool,
    errors: &mut Vec<String>,
) {
    if value.and_then(Value::as_bool) != Some(expected) {
        errors.push(format!(
            "{context} {field} was {}, expected {expected}",
            display_value(value)
        ));
    }
}

fn require_u64(
    context: &str,
    value: Option<&Value>,
    field: &str,
    errors: &mut Vec<String>,
) -> Option<u64> {
    let parsed = value.and_then(Value::as_u64);
    if parsed.is_none() {
        errors.push(format!(
            "{context} {field} was {}, expected a non-negative integer",
            display_value(value)
        ));
    }
    parsed
}

fn require_exact_u64(
    context: &str,
    value: Option<&Value>,
    field: &str,
    expected: u64,
    errors: &mut Vec<String>,
) {
    if value.and_then(Value::as_u64) != Some(expected) {
        errors.push(format!(
            "{context} {field} was {}, expected {expected}",
            display_value(value)
        ));
    }
}

fn require_empty_array(
    context: &str,
    value: Option<&Value>,
    field: &str,
    errors: &mut Vec<String>,
) {
    match value.and_then(Value::as_array) {
        Some(items) if items.is_empty() => {}
        Some(items) => errors.push(format!(
            "{context} {field} contained {} item(s), expected []",
            items.len()
        )),
        None => errors.push(format!(
            "{context} {field} was {}, expected []",
            display_value(value)
        )),
    }
}

fn require_command_returncode(
    spec: &str,
    mode_name: &str,
    label: &str,
    run: &Value,
    command: &Value,
    errors: &mut Vec<String>,
) {
    let expected = integer_field(run, "returncode").unwrap_or(0);
    match integer_field(command, "returncode") {
        Some(actual) if actual == expected && actual == 0 => {}
        _ => errors.push(format!(
            "{spec}: {mode_name} {label}: command returncode was {}, expected {expected}",
            display_value(command.get("returncode"))
        )),
    }
}

fn require_command_env(
    spec: &str,
    mode_name: &str,
    label: &str,
    run: &Value,
    command: &Value,
    errors: &mut Vec<String>,
) {
    let Some(command_env) = command.get("env_overrides").and_then(Value::as_object) else {
        errors.push(format!(
            "{spec}: {mode_name} {label}: command env_overrides missing or not an object"
        ));
        return;
    };
    if mode_name == "tlc" {
        if !command_env.is_empty() {
            errors.push(format!(
                "{spec}: tlc {label}: command env_overrides was {}, expected {{}}",
                display_value(command.get("env_overrides"))
            ));
        }
        return;
    }
    let Some(run_env) = run.get("env_overrides").and_then(Value::as_object) else {
        errors.push(format!(
            "{spec}: {mode_name} {label}: env_overrides missing for command comparison"
        ));
        return;
    };
    if command_env != run_env {
        errors.push(format!(
            "{spec}: {mode_name} {label}: command env_overrides did not match summary run env_overrides"
        ));
    }
}

fn require_command_argv(
    spec: &str,
    mode_name: &str,
    label: &str,
    command: &Value,
    errors: &mut Vec<String>,
) {
    let Some(argv) = string_array(command.get("argv")) else {
        errors.push(format!(
            "{spec}: {mode_name} {label}: command argv was {}, expected an array of strings",
            display_value(command.get("argv"))
        ));
        return;
    };
    if mode_name == "tlc" {
        require_tlc_command_argv(spec, label, &argv, errors);
    } else {
        require_ty_command_argv(spec, mode_name, label, &argv, errors);
    }
}

fn require_tlc_command_argv(spec: &str, label: &str, argv: &[String], errors: &mut Vec<String>) {
    let expected_len = 14;
    if argv.len() != expected_len {
        errors.push(format!(
            "{spec}: tlc {label}: command argv length was {}, expected {expected_len}",
            argv.len()
        ));
        return;
    }
    let expected_positions: &[(usize, &str)] = &[
        (0, "java"),
        (5, "-jar"),
        (8, "-config"),
        (10, "-metadir"),
        (12, "-workers"),
        (13, "1"),
    ];
    for (offset, expected) in expected_positions {
        if argv.get(*offset).map(String::as_str) != Some(*expected) {
            errors.push(format!(
                "{spec}: tlc {label}: command argv[{offset}] was {}, expected {expected:?}",
                argv.get(*offset)
                    .map(|value| format!("{value:?}"))
                    .unwrap_or_else(|| "missing".to_string())
            ));
        }
    }
    for (offset, expected) in super::tlc_java_single_thread_args().iter().enumerate() {
        let offset = offset + 1;
        if argv.get(offset).map(String::as_str) != Some(*expected) {
            errors.push(format!(
                "{spec}: tlc {label}: command JVM arg {offset} was {}, expected {expected:?}",
                argv.get(offset)
                    .map(|value| format!("{value:?}"))
                    .unwrap_or_else(|| "missing".to_string())
            ));
        }
    }
}

fn require_ty_command_argv(
    spec: &str,
    mode_name: &str,
    label: &str,
    argv: &[String],
    errors: &mut Vec<String>,
) {
    let expected_backend = match mode_name {
        "interp" => "interpreter",
        "trust-cg" => "trust-cg",
        _ => mode_name,
    };
    let expected_len = 11;
    if argv.len() != expected_len {
        errors.push(format!(
            "{spec}: {mode_name} {label}: command argv length was {}, expected {expected_len}; enforce mode does not permit TY-only flags",
            argv.len()
        ));
        return;
    }
    let expected_positions: &[(usize, &str)] = &[
        (1, "check"),
        (3, "--config"),
        (5, "--workers"),
        (6, "1"),
        (7, "--force"),
        // Count-parity lever: auto-POR/auto-symmetry off via CLI flag (the
        // child ignores ambient TY_AUTO_POR / TY_AUTO_SYMMETRY env pins).
        (8, "--no-reduction"),
        (9, "--backend"),
        (10, expected_backend),
    ];
    for (offset, expected) in expected_positions {
        if argv.get(*offset).map(String::as_str) != Some(*expected) {
            errors.push(format!(
                "{spec}: {mode_name} {label}: command argv[{offset}] was {}, expected {expected:?}",
                argv.get(*offset)
                    .map(|value| format!("{value:?}"))
                    .unwrap_or_else(|| "missing".to_string())
            ));
        }
    }
}

fn string_array(value: Option<&Value>) -> Option<Vec<String>> {
    value.and_then(Value::as_array).and_then(|items| {
        items
            .iter()
            .map(|item| item.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
    })
}

fn require_generated_parity(
    spec: &str,
    plan: &PlannedGate,
    tlc: &ModeFacts,
    interp: &ModeFacts,
    trust_cg: &ModeFacts,
    errors: &mut Vec<String>,
) {
    if !plan.require_generated_state_parity_by_run_index {
        return;
    }
    for (run_index, tlc_generated) in &tlc.generated_by_run {
        let interp_generated = interp.generated_by_run.get(run_index);
        let trust_cg_generated = trust_cg.generated_by_run.get(run_index);
        if interp_generated != Some(tlc_generated) || trust_cg_generated != Some(tlc_generated) {
            errors.push(format!(
                "{spec}: raw generated-state parity failed at run {run_index}: tlc={tlc_generated:?}, interp={interp_generated:?}, trust_cg={trust_cg_generated:?}"
            ));
        }
    }
}

fn require_trust_cg_runs(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    plan: &PlannedGate,
    spec: &str,
    row: &Value,
    errors: &mut Vec<String>,
) {
    let Some(runs) = row
        .get("trust_cg")
        .and_then(|mode| mode.get("runs"))
        .and_then(Value::as_array)
    else {
        return;
    };
    let mut execution_seconds = Vec::new();
    let require_execution_speedup = plan
        .benchmark_flags
        .iter()
        .any(|flag| flag == "require_trust_cg_execution_faster_than_tlc");
    for (position, run) in runs.iter().enumerate() {
        let label = run_label(run, position);
        require_env(spec, &label, plan, run, errors);
        require_telemetry(spec, &label, plan, run, errors);
        if let Some(seconds) = compiled_bfs_execution_seconds(run.get("trust_cg_telemetry")) {
            execution_seconds.push(seconds);
        }
        if let Some(requirement) = plan.required_trust_cg_selftest_by_spec.get(spec) {
            require_selftest(
                prepared,
                summary_path,
                spec,
                &label,
                run,
                requirement,
                errors,
            );
        }
    }

    let actual_execution_median = median(&mut execution_seconds);
    match row
        .get("trust_cg")
        .and_then(|mode| mode.get("execution_median_seconds"))
    {
        Some(advertised) => match (actual_execution_median, advertised.as_f64()) {
            (Some(actual), Some(value)) if close(actual, value) => {}
            (Some(actual), _) => errors.push(format!(
                "{spec}: trust-cg advertised execution_median_seconds {} did not match recomputed median {actual:?}",
                display_value(Some(advertised))
            )),
            (None, _) if advertised.is_null() && !require_execution_speedup => {}
            (None, _) => errors.push(format!(
                "{spec}: trust-cg advertised execution_median_seconds {} did not match recomputed median None",
                display_value(Some(advertised))
            )),
        },
        None if actual_execution_median.is_some() || require_execution_speedup => errors.push(
            format!(
                "{spec}: trust-cg advertised execution_median_seconds was missing, expected recomputed median {actual_execution_median:?}"
            ),
        ),
        None => {}
    }

    if let (Some(tlc_median), Some(execution_median)) = (
        row.get("tlc")
            .and_then(|mode| mode.get("median_seconds"))
            .and_then(Value::as_f64),
        actual_execution_median,
    ) {
        let speedup = tlc_median / execution_median;
        if require_execution_speedup {
            require_advertised_speedup(
                spec,
                row,
                "speedup_trust_cg_execution_vs_tlc",
                speedup,
                errors,
            );
            if speedup <= 1.0 {
                errors.push(format!(
                    "{spec}: speedup_trust_cg_execution_vs_tlc was {speedup:.6}, expected > 1.000000"
                ));
            }
        }
    }
}

fn require_env(spec: &str, label: &str, plan: &PlannedGate, run: &Value, errors: &mut Vec<String>) {
    let Some(env) = run.get("env_overrides").and_then(Value::as_object) else {
        errors.push(format!("{spec}: trust-cg {label}: env_overrides missing"));
        return;
    };
    require_env_map(
        &format!("{spec}: trust_cg {label}: env_overrides"),
        plan,
        env,
        errors,
    );
}

fn require_env_map(
    label: &str,
    plan: &PlannedGate,
    env: &serde_json::Map<String, Value>,
    errors: &mut Vec<String>,
) {
    let required = plan.enforce_required_env();
    for (key, expected) in required {
        if env.get(&key).and_then(Value::as_str) != Some(expected.as_str()) {
            errors.push(format!(
                "{label}[{key}] was {}, expected {expected:?}",
                display_value(env.get(&key))
            ));
        }
    }
    let unexpected = plan.unexpected_enforce_env_keys(env.keys().map(String::as_str));
    if !unexpected.is_empty() {
        errors.push(format!(
            "{label} contains unexpected gate-control env key(s): {}",
            unexpected.join(", ")
        ));
    }
}

fn require_telemetry(
    spec: &str,
    label: &str,
    plan: &PlannedGate,
    run: &Value,
    errors: &mut Vec<String>,
) {
    let Some(telemetry) = run.get("trust_cg_telemetry").and_then(Value::as_object) else {
        errors.push(format!(
            "{spec}: trust-cg {label}: trust_cg_telemetry missing"
        ));
        return;
    };
    if plan
        .benchmark_flags
        .iter()
        .any(|flag| flag == "require_no_trust_cg_fallbacks")
        && !plan
            .benchmark_flags
            .iter()
            .any(|flag| flag == "allow_trust_cg_invariant_rust_fallbacks")
    {
        require_no_fallback_markers(spec, label, telemetry, errors);
    }
    for name in &plan.required_trust_cg_compilation_total_matches {
        require_compilation_total_match(spec, label, telemetry, name, errors);
    }
    if plan
        .benchmark_flags
        .iter()
        .any(|flag| flag == "require_native_fused_flat_frontier_admission")
    {
        require_native_fused_flat_frontier_admission(spec, label, telemetry, errors);
    }
    let mut requirements = plan.required_trust_cg_run_telemetry.clone();
    if let Some(per_spec) = plan.required_trust_cg_run_telemetry_by_spec.get(spec) {
        requirements.extend(per_spec.clone());
    }
    for (field, requirement) in requirements {
        let value = telemetry.get(&field);
        match requirement {
            TelemetryRequirement::Bool(expected) => {
                if value.and_then(Value::as_bool) != Some(expected) {
                    errors.push(format!(
                        "{spec}: trust-cg {label}: telemetry[{field}] was {}, expected {expected}",
                        display_value(value)
                    ));
                }
            }
            TelemetryRequirement::Integer(expected) => {
                if integer_value(value) != Some(expected) {
                    errors.push(format!(
                        "{spec}: trust-cg {label}: telemetry[{field}] was {}, expected {expected}",
                        display_value(value)
                    ));
                }
            }
            TelemetryRequirement::Text(expected) => {
                require_text_telemetry(
                    spec, label, telemetry, &field, &expected, value, run, errors,
                );
            }
        }
    }
}

fn require_native_fused_flat_frontier_admission(
    spec: &str,
    label: &str,
    telemetry: &serde_json::Map<String, Value>,
    errors: &mut Vec<String>,
) {
    let flat_primary = telemetry.get("flat_state_primary").and_then(Value::as_bool);
    let admission_active = telemetry
        .get("trust_cg_native_fused_flat_frontier_admission_active")
        .and_then(Value::as_bool);
    let frontier_admitted = telemetry
        .get("compiled_bfs_flat_frontier_admitted")
        .and_then(Value::as_bool);

    if flat_primary == Some(true) {
        return;
    }
    if admission_active == Some(true) && frontier_admitted == Some(true) {
        return;
    }

    errors.push(format!(
        "{spec}: trust-cg {label}: native fused flat frontier proof was \
         flat_state_primary={}, trust_cg_native_fused_flat_frontier_admission_active={}, \
         compiled_bfs_flat_frontier_admitted={}, expected flat_state_primary=true or \
         non-primary native-fused admission active with compiled BFS flat frontier admitted",
        display_value(telemetry.get("flat_state_primary")),
        display_value(telemetry.get("trust_cg_native_fused_flat_frontier_admission_active")),
        display_value(telemetry.get("compiled_bfs_flat_frontier_admitted"))
    ));
}

fn require_compilation_total_match(
    spec: &str,
    label: &str,
    telemetry: &serde_json::Map<String, Value>,
    name: &str,
    errors: &mut Vec<String>,
) {
    let (compiled_key, total_key) = match name {
        "actions" => ("trust_cg_actions_compiled", "trust_cg_actions_total"),
        "invariants" => ("trust_cg_invariants_compiled", "trust_cg_invariants_total"),
        other => {
            errors.push(format!(
                "{spec}: trust-cg {label}: unknown trust-codegen compilation total policy check {other:?}"
            ));
            return;
        }
    };
    let total = telemetry.get(total_key);
    let Some(total_value) = non_negative_integer_value(total) else {
        errors.push(format!(
            "{spec}: trust-cg {label}: telemetry[{total_key}] was {}, expected a non-negative integer",
            display_value(total)
        ));
        return;
    };
    let compiled = telemetry.get(compiled_key);
    let Some(compiled_value) = non_negative_integer_value(compiled) else {
        errors.push(format!(
            "{spec}: trust-cg {label}: telemetry[{compiled_key}] was {}, expected telemetry[{total_key}] ({total_value})",
            display_value(compiled)
        ));
        return;
    };
    if compiled_value != total_value {
        errors.push(format!(
            "{spec}: trust-cg {label}: telemetry[{compiled_key}] was {compiled_value}, expected telemetry[{total_key}] ({total_value})"
        ));
    }
}

fn require_no_fallback_markers(
    spec: &str,
    label: &str,
    telemetry: &serde_json::Map<String, Value>,
    errors: &mut Vec<String>,
) {
    match telemetry.get("fallback_reasons").and_then(Value::as_array) {
        Some(reasons) if reasons.is_empty() => {}
        Some(reasons) => errors.push(format!(
            "{spec}: trust-cg {label}: trust-codegen fallback reasons observed ({})",
            reasons.len()
        )),
        None => errors.push(format!(
            "{spec}: trust-cg {label}: fallback_reasons missing or not an array"
        )),
    }
}

fn require_text_telemetry(
    spec: &str,
    label: &str,
    telemetry: &serde_json::Map<String, Value>,
    field: &str,
    expected: &str,
    value: Option<&Value>,
    run: &Value,
    errors: &mut Vec<String>,
) {
    match expected {
        "positive" => {
            if !positive_integer(value) {
                errors.push(format!(
                    "{spec}: trust-cg {label}: telemetry[{field}] was {}, expected positive integer",
                    display_value(value)
                ));
            }
        }
        "transitions" => {
            if integer_value(value) != integer_field(run, "transitions") {
                errors.push(format!(
                    "{spec}: trust-cg {label}: telemetry[{field}] was {}, expected transitions {}",
                    display_value(value),
                    display_value(run.get("transitions"))
                ));
            }
        }
        "states_found" => {
            if integer_value(value) != integer_field(run, "states_found") {
                errors.push(format!(
                    "{spec}: trust-cg {label}: telemetry[{field}] was {}, expected states_found {}",
                    display_value(value),
                    display_value(run.get("states_found"))
                ));
            }
        }
        "all" => {
            let total = telemetry.get("trust_cg_invariants_total");
            let actual = integer_value(value);
            let expected = integer_value(total);
            if actual.is_none() || expected.is_none() || actual != expected {
                errors.push(format!(
                    "{spec}: trust-cg {label}: telemetry[{field}] was {}, expected trust_cg_invariants_total {}",
                    display_value(value),
                    display_value(total)
                ));
            }
        }
        exact => {
            if value.and_then(Value::as_str) != Some(exact) {
                errors.push(format!(
                    "{spec}: trust-cg {label}: telemetry[{field}] was {}, expected {exact:?}",
                    display_value(value)
                ));
            }
        }
    }
}

fn require_selftest(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    spec: &str,
    label: &str,
    run: &Value,
    requirement: &SelftestRequirement,
    errors: &mut Vec<String>,
) {
    let Some(artifact_dir) = run.get("artifact_dir").and_then(Value::as_str) else {
        errors.push(format!("{spec}: trust-cg {label}: artifact_dir missing"));
        return;
    };
    let artifact_dir = resolve_artifact_dir(prepared, summary_path, artifact_dir);
    let text = ["stdout.txt", "stderr.txt"]
        .into_iter()
        .filter_map(|name| read_lossy(artifact_dir.join(name)).ok())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        errors.push(format!(
            "{spec}: trust-cg {label}: selftest artifacts missing under {}",
            artifact_dir.display()
        ));
        return;
    }
    let relevant = if let Some(marker_index) = text.rfind(FLAT_PRIMARY_REBUILD_MARKER) {
        &text[marker_index..]
    } else {
        // Some artifacts are flat-primary from the first build and do not rebuild
        // after layout promotion, so the whole artifact is the current segment.
        text.as_str()
    };
    let prepared = relevant.contains(&format!(
        "prepared native fused callout selftest: actions={}, state_constraints={}, invariants={}, missing_expected=0, fail_closed=true",
        requirement.actions, requirement.state_constraints, requirement.invariants
    ));
    let running = relevant.contains(&format!(
        "running native fused callout selftest on first real parent: state_len={}, actions={}, state_constraints={}, invariants={}, fail_closed=true",
        requirement.state_len, requirement.actions, requirement.state_constraints, requirement.invariants
    ));
    let complete = relevant.contains("native fused callout selftest complete");
    if !prepared || !running || !complete {
        errors.push(format!(
            "{spec}: trust-cg {label}: strict native fused selftest markers missing or mismatched"
        ));
    }
    for line in relevant.lines() {
        if line.contains("[trust_cg-selftest]")
            && line.contains("prepared native fused callout selftest:")
        {
            if let Some(missing_expected) = selftest_integer_field(line, "missing_expected") {
                if missing_expected != 0 {
                    errors.push(format!(
                        "{spec}: trust-cg {label}: strict native callout selftest reported missing expected callouts: {missing_expected}"
                    ));
                }
            }
        }
        if let Some((kind, status, value)) = parse_selftest_callout_result(line) {
            if STRICT_SELFTEST_FALSE_RESULT_KINDS.contains(&kind.as_str())
                && status == "Ok"
                && value == 0
            {
                errors.push(format!(
                    "{spec}: trust-cg {label}: native fused callout selftest reported false strict check: kind={kind} status=Ok value=0 line={line:?}"
                ));
            }
        }
    }
    if text.lines().any(|line| {
        line.contains("[trust_cg-selftest]")
            && (line.contains("native fused callout selftest failed")
                || line.contains("failing closed"))
    }) {
        errors.push(format!(
            "{spec}: trust-cg {label}: strict native callout selftest failure marker was present"
        ));
    }
}

fn resolve_artifact_dir(
    prepared: &PreparedSupremacy,
    summary_path: &Path,
    artifact_dir: &str,
) -> PathBuf {
    let path = PathBuf::from(artifact_dir);
    if path.is_absolute() {
        return path;
    }
    let summary_base = summary_path.parent().unwrap_or_else(|| Path::new("."));
    let mut candidates = vec![summary_base.join(&path)];
    if let Some(repo_root) = policy_repo_root(prepared) {
        candidates.push(repo_root.join(&path));
    }
    candidates
        .iter()
        .find(|candidate| candidate.exists())
        .cloned()
        .unwrap_or_else(|| summary_base.join(path))
}

fn policy_repo_root(prepared: &PreparedSupremacy) -> Option<PathBuf> {
    let root = prepared
        .policy_path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)?;
    if root.as_os_str().is_empty() {
        Some(PathBuf::from("."))
    } else {
        Some(root)
    }
}

fn read_lossy(path: PathBuf) -> std::io::Result<String> {
    fs::read(path).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

fn parse_selftest_callout_result(line: &str) -> Option<(String, String, i64)> {
    if !line.contains("[trust_cg-selftest]")
        || !line.contains("status=")
        || !line.contains("value=")
    {
        return None;
    }
    let kind = selftest_field(line, "kind").or_else(|| {
        line.trim_start()
            .strip_prefix("[trust_cg-selftest]")
            .and_then(|rest| rest.split_whitespace().next())
            .map(str::to_string)
    })?;
    let status = selftest_field(line, "status")?;
    let value = selftest_integer_field(line, "value")?;
    Some((kind, status, value))
}

fn selftest_integer_field(line: &str, key: &str) -> Option<i64> {
    selftest_field(line, key).and_then(|value| value.replace(',', "").parse().ok())
}

fn selftest_field(line: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    line.split_whitespace().find_map(|part| {
        part.trim_end_matches(',')
            .strip_prefix(&prefix)
            .map(|value| value.trim_end_matches(',').to_string())
    })
}

fn require_integer_equals(
    spec: &str,
    mode_name: &str,
    label: &str,
    run: &Value,
    field: &str,
    expected: u64,
    errors: &mut Vec<String>,
) {
    if integer_field(run, field).map(|value| value as u64) != Some(expected) {
        errors.push(format!(
            "{spec}: {mode_name} {label}: {field} was {}, expected {expected}",
            display_value(run.get(field))
        ));
    }
}

fn run_label(run: &Value, position: usize) -> String {
    integer_field(run, "run_index")
        .map(|index| format!("run {index}"))
        .unwrap_or_else(|| format!("run at position {}", position + 1))
}

fn integer_field(value: &Value, field: &str) -> Option<i64> {
    integer_value(value.get(field))
}

fn integer_value(value: Option<&Value>) -> Option<i64> {
    let value = value?;
    if value.is_boolean() {
        return None;
    }
    value.as_i64()
}

fn non_negative_integer_value(value: Option<&Value>) -> Option<i64> {
    integer_value(value).filter(|value| *value >= 0)
}

fn non_negative_u64_value(value: Option<&Value>) -> Option<u64> {
    non_negative_integer_value(value).and_then(|value| u64::try_from(value).ok())
}

fn finite_float_field(value: &Value, field: &str) -> Option<f64> {
    let value = value.get(field)?;
    if value.is_boolean() {
        return None;
    }
    value.as_f64().filter(|number| number.is_finite())
}

fn positive_integer(value: Option<&Value>) -> bool {
    integer_value(value).is_some_and(|value| value > 0)
}

fn require_advertised_speedup(
    spec: &str,
    row: &Value,
    field: &str,
    recomputed: f64,
    errors: &mut Vec<String>,
) {
    match row.get(field).and_then(Value::as_f64) {
        Some(advertised) if close(advertised, recomputed) => {}
        Some(advertised) => errors.push(format!(
            "{spec}: advertised {field} {advertised:?} did not match recomputed speedup {recomputed:?}"
        )),
        None => errors.push(format!(
            "{spec}: advertised {field} was {}, expected recomputed speedup {recomputed:?}",
            display_value(row.get(field))
        )),
    }
}

fn compiled_bfs_execution_seconds(telemetry: Option<&Value>) -> Option<f64> {
    let telemetry = telemetry?;
    let nanos = telemetry.get("compiled_bfs_execution_nanos");
    if let Some(nanos) = integer_value(nanos).filter(|value| *value > 0) {
        return Some(nanos as f64 / 1_000_000_000.0);
    }
    telemetry
        .get("compiled_bfs_execution_seconds")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) / 2.0)
    }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 || (left - right).abs() <= 1e-9 * left.abs().max(right.abs())
}

fn display_value(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => format!("{value:?}"),
        Some(value) => value.to_string(),
        None => "missing".to_string(),
    }
}

fn render_markdown(verdict: &PolicyVerdict) -> String {
    let mut markdown = format!(
        "# TY Supremacy Policy Verdict\n\nVerdict: **{}**\n\nSummary: `{}`\n",
        verdict.verdict,
        verdict.raw_benchmark_summary.path.display()
    );
    if let Some(gate_mode) = &verdict.gate_mode {
        let _ = write!(markdown, "\nGate mode: `{gate_mode}`\n");
    }
    markdown.push_str("\n## Anti-Overfit Evidence\n");
    let evidence = &verdict.anti_overfit_evidence;
    let _ = writeln!(
        markdown,
        "- Launch canary: {} specs ({})",
        evidence.launch_corpus.spec_count, evidence.launch_corpus.role
    );
    let _ = writeln!(
        markdown,
        "- Engine selection: `{}`; forbidden selector inputs: `{}`",
        evidence.engine_selection_contract.selection_basis,
        evidence
            .engine_selection_contract
            .forbidden_selector_inputs
            .join("`, `")
    );
    let matrix = &evidence.matrix_holdout;
    match matrix.total_specs {
        Some(total_specs) => {
            let _ = writeln!(
                markdown,
                "- Matrix holdout: {total_specs} specs at `{}`; covers more than launch canary: `{}`",
                matrix.baseline_path.display(),
                matrix.covers_more_than_launch_canary.unwrap_or(false)
            );
        }
        None => {
            let _ = writeln!(
                markdown,
                "- Matrix holdout: unavailable at `{}` ({})",
                matrix.baseline_path.display(),
                matrix
                    .unavailable_reason
                    .as_deref()
                    .unwrap_or("unknown error")
            );
        }
    }
    let _ = writeln!(
        markdown,
        "- Cold wall controls: artifact cache disabled `{}`, workers `{}`, native fused strict `{}`",
        evidence.cold_single_thread_wall.artifact_cache_disabled,
        evidence.cold_single_thread_wall.required_workers,
        evidence.cold_single_thread_wall.native_fused_strict
    );
    if !verdict.errors.is_empty() {
        markdown.push_str("\n## Errors\n");
        for error in &verdict.errors {
            let _ = writeln!(markdown, "- {error}");
        }
    }
    markdown
}

#[cfg(test)]
#[path = "verdict_focused_tests.rs"]
mod focused_tests;

#[cfg(test)]
mod verdict_write_tests {
    use std::collections::BTreeMap;

    use serde_json::{json, Value};

    use super::*;
    use crate::cli_schema::{SupremacyGateMode, SupremacyMode, SupremacyOutputFormat};

    fn repo_policy_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/tlc_comparison/single_thread_supremacy_gate.json")
    }

    fn prepared(output_dir: &Path) -> PreparedSupremacy {
        let policy_path = repo_policy_path();
        let policy = super::super::policy::SupremacyPolicy::load(&policy_path).unwrap();
        let gate_plan = policy
            .resolve_gate_mode(
                SupremacyMode::Enforce,
                Some(SupremacyGateMode::FullNativeFused),
            )
            .map(super::super::policy::PlannedGate::from_resolved)
            .unwrap();
        PreparedSupremacy {
            command: "gate",
            policy_path,
            output_dir: output_dir.to_path_buf(),
            specs: policy.specs.clone(),
            trust_cg_env_overrides: BTreeMap::new(),
            interp_env_overrides: BTreeMap::new(),
            format: SupremacyOutputFormat::Human,
            timeout_seconds: 300,
            ty_bin: None,
            target_dir: None,
            cargo_profile: "release".to_string(),
            ty_flags: Vec::new(),
            runs: Some(3),
            policy,
            gate_plan: Some(gate_plan),
        }
    }

    fn required_env() -> BTreeMap<&'static str, &'static str> {
        // No TY_AUTO_POR / TY_AUTO_SYMMETRY pins: count-parity is the
        // `--no-reduction` CLI flag in the recorded argv, not env.
        BTreeMap::from([
            ("TY_trust_cg", "1"),
            ("TY_TRUST_CG_BFS", "1"),
            ("TY_TRUST_CG_EXISTS", "1"),
            ("TY_BYTECODE_VM", "1"),
            ("TY_BYTECODE_VM_STATS", "1"),
            ("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST", "strict"),
            ("TY_TRUST_CG_NATIVE_FUSED_STRICT", "1"),
            ("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS", "1"),
            ("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP", "1"),
            ("TY_DISABLE_ARTIFACT_CACHE", "1"),
        ])
    }

    fn test_binary_identity(root: &Path) -> (String, String) {
        let path = root.join("test-bin").join("ty");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"verdict-write-test-ty-binary").unwrap();
        let digest = sha256_file(&path).unwrap();
        (path.display().to_string(), digest)
    }

    fn interp_env() -> BTreeMap<&'static str, &'static str> {
        // No TY_AUTO_POR / TY_AUTO_SYMMETRY pins: count-parity is the
        // `--no-reduction` CLI flag in the recorded argv, not env.
        BTreeMap::from([
            ("TY_BYTECODE_VM", "1"),
            ("TY_trust_cg", "0"),
            ("TY_TRUST_CG_BFS", "0"),
        ])
    }

    fn required_env_with_cache(output_dir: &Path) -> BTreeMap<String, String> {
        let mut env = required_env()
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect::<BTreeMap<_, _>>();
        env.insert(
            TY_CACHE_DIR_ENV.to_string(),
            output_dir
                .join("trust_cg-artifact-cache")
                .display()
                .to_string(),
        );
        env
    }

    fn mode(
        output_dir: &Path,
        spec: &str,
        kind: &str,
        states: u64,
        generated: u64,
        elapsed: f64,
    ) -> Value {
        let runs = [1, 2, 3]
            .into_iter()
            .map(|index| {
                let artifact_dir = format!("artifacts/{spec}/{kind}-{index}");
                let mut run = json!({
                    "tool": if kind == "tlc" { "tlc" } else { "ty" },
                    "spec_name": spec,
                    "run_index": index,
                    "states_found": states,
                    "elapsed_seconds": elapsed,
                    "workers": 1,
                    "returncode": 0,
                    "artifact_dir": artifact_dir,
                });
                if kind == "tlc" {
                    run["raw_initial_states_generated"] = json!(1);
                    run["raw_successors_generated"] = json!(generated);
                    run["states_generated"] = json!(generated + 1);
                    run["transitions"] = json!(generated);
                } else {
                    run["mode"] = json!(kind);
                    run["env_overrides"] = json!(interp_env());
                    run["transitions"] = json!(generated);
                    run["raw_initial_states_generated"] = json!(1);
                    run["raw_successors_generated"] = json!(generated);
                    run["states_generated"] = json!(generated + 1);
                }
                write_command_artifact(output_dir, spec, kind, &run);
                run
            })
            .collect::<Vec<_>>();
        json!({
            "all_ok": true,
            "median_seconds": elapsed,
            "expected_states": states,
            "expected_states_match": true,
            "runs": runs,
        })
    }

    fn write_command_artifact(output_dir: &Path, spec: &str, mode: &str, run: &Value) {
        let artifact_path = output_dir.join(run["artifact_dir"].as_str().unwrap());
        fs::create_dir_all(&artifact_path).unwrap();
        fs::create_dir(artifact_path.join(COMMAND_SCRATCH_DIR_NAME)).unwrap();
        fs::write(
            artifact_path.join("command.json"),
            serde_json::to_string_pretty(&command_artifact(spec, mode, run, &artifact_path))
                .unwrap()
                + "\n",
        )
        .unwrap();
        if !artifact_path.join("stdout.txt").exists() {
            fs::write(artifact_path.join("stdout.txt"), "").unwrap();
        }
        fs::write(artifact_path.join("stderr.txt"), "").unwrap();
    }

    fn command_artifact(spec: &str, mode: &str, run: &Value, artifact_path: &Path) -> Value {
        let argv = if mode == "tlc" {
            let mut argv = vec!["java".to_string()];
            argv.extend(
                super::super::tlc_java_single_thread_args()
                    .iter()
                    .map(|arg| (*arg).to_string()),
            );
            argv.extend([
                "-jar".to_string(),
                "tlaplus/tytools.jar".to_string(),
                format!("{spec}.tla"),
                "-config".to_string(),
                format!("{spec}.cfg"),
                "-metadir".to_string(),
                "tlc-metadir".to_string(),
                "-workers".to_string(),
                "1".to_string(),
            ]);
            argv
        } else {
            vec![
                "ty".to_string(),
                "check".to_string(),
                format!("{spec}.tla"),
                "--config".to_string(),
                format!("{spec}.cfg"),
                "--workers".to_string(),
                "1".to_string(),
                "--force".to_string(),
                "--no-reduction".to_string(),
                "--backend".to_string(),
                if mode == "interp" {
                    "interpreter"
                } else {
                    "trust-cg"
                }
                .to_string(),
            ]
        };
        let artifact_path = fs::canonicalize(artifact_path).unwrap();
        let scratch_path = fs::canonicalize(artifact_path.join(COMMAND_SCRATCH_DIR_NAME)).unwrap();
        let scratch_text = scratch_path.display().to_string();
        let environment_confinement = COMMAND_SCOPED_ENV_KEYS
            .iter()
            .map(|key| ((*key).to_string(), json!(scratch_text.clone())))
            .collect::<serde_json::Map<_, _>>();
        let disk_evidence = json!({
            "contract_schema": DISK_SCOPE_CONTRACT_SCHEMA,
            "scope_root": artifact_path.display().to_string(),
            "scratch_root": scratch_text,
            "scope": "command_artifact_and_scratch_tree",
            "method": "recursive_filesystem_metadata_polling",
            "peak_exact": false,
            "sampling_execution": "inline_runner_poll_loop",
            "sampling_can_perturb_elapsed": true,
            "peak_allocated_bytes": 4096,
            "peak_apparent_bytes": 1024,
            "sampling_interval_ms": DISK_USAGE_SAMPLE_INTERVAL.as_millis() as u64,
            "scan_budget_ms": DISK_USAGE_SCAN_BUDGET.as_millis() as u64,
            "scan_entry_limit": DISK_USAGE_SCAN_ENTRY_LIMIT,
            "total_scan_nanoseconds": 2000,
            "max_scan_nanoseconds": 1000,
            "samples_attempted": 2,
            "samples_complete": 2,
            "samples_partial": 0,
            "peak_entries_observed": 2,
            "initial_sample_complete": true,
            "final_sample_complete": true,
            "setup_complete": true,
            "environment_confinement": environment_confinement,
            "environment_confinement_complete": true,
            "scope_identity_stable": true,
            "ownership_verified": true,
            "accounting_complete": true,
            "polling_complete": true,
            "process_tree_lifetime_complete": true,
            "complete": true,
            "strict_qualified": true,
            "diagnostics": [],
            "qualification_failures": [],
        });
        json!({
            "schema": COMMAND_ARTIFACT_SCHEMA,
            "argv": argv,
            "cwd": "/tmp",
            "returncode": run["returncode"],
            "elapsed_seconds": run["elapsed_seconds"],
            "env_overrides": run.get("env_overrides").cloned().unwrap_or_else(|| json!({})),
            "timed_out": false,
            "peak_rss_bytes": null,
            "resource_evidence": {
                "strict_qualified": true,
                "qualification_failures": [],
                "disk": disk_evidence,
            },
        })
    }

    fn valid_summary(prepared: &PreparedSupremacy, output_dir: &Path) -> Value {
        let (binary_path, binary_sha256) = test_binary_identity(output_dir);
        let expected_states = BTreeMap::from([
            ("CoffeeCan1000BeansSafety", 501500_u64),
            ("EWD998Small", 1520618),
            ("MCLamportMutex", 724274),
        ]);
        let expected_generated = BTreeMap::from([
            ("CoffeeCan1000BeansSafety", 1498502_u64),
            ("EWD998Small", 9630813),
            ("MCLamportMutex", 2496350),
        ]);
        let per_spec = BTreeMap::from([
            (
                "CoffeeCan1000BeansSafety",
                json!({
                    "trust_cg_native_fused_mode": "invariant_checking",
                    "trust_cg_native_fused_state_len": 2,
                    "trust_cg_native_fused_state_constraint_count": 0,
                }),
            ),
            (
                "EWD998Small",
                json!({
                    "trust_cg_native_fused_mode": "state_constraint_checking",
                    "trust_cg_native_fused_state_len": 15,
                    "trust_cg_native_fused_state_constraint_count": 1,
                    "trust_cg_state_constraints_compiled": 1,
                    "trust_cg_state_constraints_total": 1,
                }),
            ),
            (
                "MCLamportMutex",
                json!({
                    "trust_cg_native_fused_mode": "state_constraint_checking",
                    "trust_cg_native_fused_state_len": 89,
                    "trust_cg_native_fused_state_constraint_count": 1,
                    "trust_cg_state_constraints_compiled": 1,
                    "trust_cg_state_constraints_total": 1,
                }),
            ),
        ]);
        let selftest = BTreeMap::from([
            ("CoffeeCan1000BeansSafety", (4_u64, 0_u64, 1_u64, 2_u64)),
            ("EWD998Small", (15, 1, 3, 15)),
            ("MCLamportMutex", (27, 1, 3, 89)),
        ]);

        let mut rows = Vec::new();
        for spec in ["CoffeeCan1000BeansSafety", "EWD998Small", "MCLamportMutex"] {
            let states = expected_states[spec];
            let generated = expected_generated[spec];
            let mut telemetry = json!({
                "compiled_bfs_level_loop_started": true,
                "compiled_bfs_level_loop_fused": true,
                "compiled_bfs_level_loop_initial_states": 1,
                "compiled_bfs_levels_completed": 1,
                "compiled_bfs_parents_processed": 1,
                "compiled_bfs_successors_generated": generated,
                "compiled_bfs_successors_new": 1,
                "compiled_bfs_execution_nanos": 500000000,
                "compiled_bfs_total_states": states,
                "trust_cg_native_fused_level_built": true,
                "trust_cg_native_fused_level_active": true,
                "trust_cg_bfs_level_loop_kind": "native_fused_trust_cg_parent_loop",
                "transitions": generated,
                "trust_cg_native_fused_regular_invariants_checked": true,
                "trust_cg_native_fused_invariant_count": 3,
                "trust_cg_invariants_total": 3,
                "trust_cg_invariants_compiled": 3,
                "trust_cg_actions_total": 4,
                "trust_cg_actions_compiled": 4,
                "trust_cg_native_fused_local_dedup": true,
                "trust_cg_native_fused_flat_frontier_admission_active": true,
                "compiled_bfs_flat_frontier_admitted": true,
                "flat_state_primary": true,
                "flat_bfs_frontier_active": true,
                "flat_bfs_frontier_fallbacks": 0,
                "fallback_reasons": [],
            });
            telemetry.as_object_mut().unwrap().extend(
                per_spec[spec]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );

            let (actions, state_constraints, invariants, state_len) = selftest[spec];

            rows.push(json!({
                "spec": spec,
                "tlc": mode(output_dir, spec, "tlc", states, generated, 3.0),
                "interp": mode(output_dir, spec, "interp", states, generated, 2.0),
                "trust_cg": {
                    "all_ok": true,
                    "median_seconds": 2.0,
                    "execution_median_seconds": 0.5,
                    "expected_states": states,
                    "expected_states_match": true,
                    "runs": ([1, 2, 3].into_iter().map(|index| {
                        let artifact_dir = format!("artifacts/{spec}/trust-cg-{index}");
                        let artifact_path = output_dir.join(&artifact_dir);
                        fs::create_dir_all(&artifact_path).unwrap();
                        fs::write(
                            artifact_path.join("stdout.txt"),
                            format!(
                                "{FLAT_PRIMARY_REBUILD_MARKER}\n[trust_cg-selftest] prepared native fused callout selftest: actions={actions}, state_constraints={state_constraints}, invariants={invariants}, missing_expected=0, fail_closed=true\n[trust_cg-selftest] running native fused callout selftest on first real parent: state_len={state_len}, actions={actions}, state_constraints={state_constraints}, invariants={invariants}, fail_closed=true\n[trust_cg-selftest] native fused callout selftest complete\n"
                            ),
                        )
                        .unwrap();
                        let run = json!({
                            "tool": "ty",
                            "mode": "trust-cg",
                            "spec_name": spec,
                            "run_index": index,
                            "states_found": states,
                            "elapsed_seconds": 2.0,
                            "workers": 1,
                            "returncode": 0,
                            "transitions": generated,
                            "raw_initial_states_generated": 1,
                            "raw_successors_generated": generated,
                            "states_generated": generated + 1,
                            "artifact_dir": artifact_dir,
                            "trust_cg_telemetry": telemetry.clone(),
                            "env_overrides": required_env_with_cache(output_dir),
                        });
                        write_command_artifact(output_dir, spec, "trust-cg", &run);
                        run
                    }).collect::<Vec<_>>()),
                },
                "parity_interp_vs_tlc": true,
                "parity_trust_cg_vs_tlc": true,
                "trust_cg_gate_failures": [],
                "speedup_interp_vs_tlc": 1.5,
                "speedup_trust_cg_vs_tlc": 1.5,
                "speedup_trust_cg_execution_vs_tlc": 6.0,
            }));
        }
        json!({
            "schema": SUMMARY_SCHEMA,
            "timestamp": "2026-04-29T120000",
            "git_commit": current_git_commit(prepared).unwrap_or_else(|| "unknown".to_string()),
            "artifact_bundle": output_dir.display().to_string(),
            "invocation": "ty supremacy gate --mode enforce --runs 3",
            "build_identity": {
                "cargo_profile": "release",
                "ty_binary_path": binary_path,
                "ty_binary_sha256": binary_sha256,
            },
            "backend_controls": {
                "interp_env": {},
                "trust_cg_env": required_env_with_cache(output_dir),
            },
            "launch_controls": {
                "tlc": {
                    "workers": 1,
                    "jvm_args": [
                        "-XX:ActiveProcessorCount=1",
                        "-XX:+UseSerialGC",
                        "-Xms64m",
                        "-Xmx4g"
                    ],
                    "heap_xms": "64m",
                    "heap_xmx": "4g",
                },
                "ty": {
                    "interp": {
                        "workers": 1,
                    },
                    "trust_cg": {
                        "workers": 1,
                        "cache_dir": output_dir.join("trust_cg-artifact-cache").display().to_string(),
                        "artifact_cache_disabled_env": "1",
                        "native_callout_compile_jobs": "1",
                    },
                },
            },
            "gate_flags": {
                "require_trust_cg_compiled_actions": true,
                "require_trust_cg_all_actions": true,
                "require_trust_cg_compiled_invariants": true,
                "require_trust_cg_compiled_bfs": true,
                "require_trust_cg_fused_level": true,
                "require_trust_cg_native_fused_level": true,
                "require_trust_cg_successor_telemetry": true,
                "require_native_fused_flat_frontier_admission": true,
                "require_flat_bfs_frontier": true,
                "require_no_trust_cg_fallbacks": true,
                "require_trust_cg_faster_than_tlc": true,
                "require_trust_cg_execution_faster_than_tlc": true,
                "allow_trust_cg_invariant_rust_fallbacks": false
            },
            "rows": rows,
        })
    }

    #[test]
    fn writes_passing_policy_verdict() {
        let dir = tempfile::tempdir().unwrap();
        let prepared = prepared(dir.path());
        let summary_path = dir.path().join("summary.json");
        fs::write(
            &summary_path,
            serde_json::to_string(&valid_summary(&prepared, dir.path())).unwrap(),
        )
        .unwrap();

        let passed = evaluate_and_write(&prepared, &summary_path).unwrap();

        assert!(passed);
        let verdict: Value = serde_json::from_str(
            &fs::read_to_string(dir.path().join("policy_verdict.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(verdict["schema"], VERDICT_SCHEMA);
        assert_eq!(verdict["verdict"], "pass");
        assert_eq!(
            verdict["generated_state_count_sources"]["trust-cg"],
            "runs[].raw_successors_generated"
        );
        assert_eq!(
            verdict["generated_state_count_sources"]["tlc"],
            "runs[].raw_successors_generated"
        );
    }

    #[test]
    fn writes_failing_policy_verdict_with_machine_readable_errors() {
        let dir = tempfile::tempdir().unwrap();
        let prepared = prepared(dir.path());
        let mut summary = valid_summary(&prepared, dir.path());
        summary["rows"][0]["trust_cg"]["median_seconds"] = json!(4.0);
        for run in summary["rows"][0]["trust_cg"]["runs"]
            .as_array_mut()
            .unwrap()
        {
            run["elapsed_seconds"] = json!(4.0);
        }
        summary["rows"][0]["speedup_trust_cg_vs_tlc"] = json!(0.75);
        let summary_path = dir.path().join("summary.json");
        fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

        let passed = evaluate_and_write(&prepared, &summary_path).unwrap();

        assert!(!passed);
        let verdict: Value = serde_json::from_str(
            &fs::read_to_string(dir.path().join("policy_verdict.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(verdict["schema"], VERDICT_SCHEMA);
        assert_eq!(verdict["verdict"], "fail");
        let errors = verdict["errors"].as_array().unwrap();
        assert!(errors.iter().any(|error| {
            error
                .as_str()
                .is_some_and(|error| error.contains("speedup_trust_cg_vs_tlc was 0.750000"))
        }));
    }

    #[test]
    fn rejects_wrong_fixed_state_count() {
        let dir = tempfile::tempdir().unwrap();
        let prepared = prepared(dir.path());
        let mut summary = valid_summary(&prepared, dir.path());
        summary["rows"][0]["trust_cg"]["runs"][0]["states_found"] = json!(501499);
        let summary_path = dir.path().join("summary.json");
        fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

        let verdict = evaluate(&prepared, &summary_path).unwrap();

        assert!(!verdict.passed());
        assert!(verdict
            .errors
            .iter()
            .any(|error| error.contains("states_found was 501499, expected 501500")));
    }
}
