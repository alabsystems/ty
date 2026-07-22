// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::*;
use crate::cli_schema::{
    SupremacyCommonArgs, SupremacyGateMode, SupremacyMode, SupremacyOutputFormat,
};

fn repo_policy_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/tlc_comparison/single_thread_supremacy_gate.json")
}

fn prepared(output_dir: PathBuf) -> PreparedSupremacy {
    let common = SupremacyCommonArgs {
        policy: Some(repo_policy_path()),
        output_dir: Some(output_dir),
        ty_bin: None,
        target_dir: None,
        cargo_profile: "release".to_string(),
        ty_flag: Vec::new(),
        timeout: 300,
        specs: Vec::new(),
        interp_env: Vec::new(),
        trust_cg_env: Vec::new(),
        format: SupremacyOutputFormat::Human,
    };
    PreparedSupremacy::prepare(
        "gate",
        &common,
        Some(3),
        Some(SupremacyGateMode::FullNativeFused),
        Some(SupremacyMode::Enforce),
    )
    .unwrap()
}

// Test scaffolding: a `prepared()` variant taking an explicit policy path, kept
// for focused tests that need a non-default policy.
#[allow(dead_code)]
fn prepared_with_policy(output_dir: PathBuf, policy: PathBuf) -> PreparedSupremacy {
    let common = SupremacyCommonArgs {
        policy: Some(policy),
        output_dir: Some(output_dir),
        ty_bin: None,
        target_dir: None,
        cargo_profile: "release".to_string(),
        ty_flag: Vec::new(),
        timeout: 300,
        specs: Vec::new(),
        interp_env: Vec::new(),
        trust_cg_env: Vec::new(),
        format: SupremacyOutputFormat::Human,
    };
    PreparedSupremacy::prepare(
        "gate",
        &common,
        Some(3),
        Some(SupremacyGateMode::FullNativeFused),
        Some(SupremacyMode::Enforce),
    )
    .unwrap()
}

fn default_matrix_corpus_identity(prepared: &PreparedSupremacy) -> (usize, String) {
    let baseline_path = matrix_baseline_path(prepared);
    let identity =
        matrix::classify_baseline_path_with_policy(&baseline_path, &prepared.policy.matrix_policy)
            .unwrap()
            .corpus;
    (identity.total_specs, identity.specs_jcs_sha256.unwrap())
}

fn default_matrix_spec_names(prepared: &PreparedSupremacy) -> Vec<String> {
    let baseline_path = matrix_baseline_path(prepared);
    let baseline: Value =
        serde_json::from_str(&fs::read_to_string(&baseline_path).unwrap()).unwrap();
    baseline["specs"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

fn repo_text(relative: &str) -> String {
    fs::read_to_string(repo_path(relative))
        .unwrap_or_else(|err| panic!("failed to read {relative}: {err}"))
}

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn normalized_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn retired_python_benchmark_helper_paths() -> [&'static str; 2] {
    [
        "scripts/benchmark_jit_codegen.py",
        "scripts/lib/verify_correctness/python_fast_subset.sh",
    ]
}

fn expected_states(prepared: &PreparedSupremacy, spec: &str) -> u64 {
    *prepared.policy.expected_state_counts.get(spec).unwrap()
}

fn expected_generated(prepared: &PreparedSupremacy, spec: &str) -> u64 {
    *prepared
        .policy
        .expected_generated_state_counts
        .get(spec)
        .unwrap()
}

fn test_binary_identity(root: &Path) -> (String, String) {
    let path = root.join("test-bin").join("ty");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"focused-test-ty-binary").unwrap();
    let digest = sha256_file(&path).unwrap();
    (path.display().to_string(), digest)
}

fn prepared_with_matrix_opt_in(
    output_dir: PathBuf,
    allow_runtime_to_error: bool,
    allow_timeout_dominance: bool,
) -> PreparedSupremacy {
    let mut prepared = prepared(output_dir);
    prepared.policy.matrix_policy.allow_runtime_to_error = allow_runtime_to_error;
    prepared.policy.matrix_policy.allow_timeout_dominance = allow_timeout_dominance;
    prepared
}

fn attach_temp_matrix_baseline(
    prepared: &mut PreparedSupremacy,
    repo_root: &Path,
    mut baseline: Value,
) {
    let policy_path = repo_root
        .join("tests")
        .join("tlc_comparison")
        .join("single_thread_supremacy_gate.json");
    let baseline_path = repo_root
        .join("tests")
        .join("tlc_comparison")
        .join("spec_baseline.json");
    fs::create_dir_all(policy_path.parent().unwrap()).unwrap();
    finalize_temp_matrix_baseline(&mut baseline);
    fs::write(&policy_path, "{}").unwrap();
    fs::write(&baseline_path, serde_json::to_string(&baseline).unwrap()).unwrap();
    prepared.policy_path = policy_path;
}

fn finalize_temp_matrix_baseline(baseline: &mut Value) {
    let specs = baseline["specs"].as_object().unwrap().clone();
    let root = baseline.as_object_mut().unwrap();
    root.insert("schema_version".to_string(), json!(3));
    root.insert("total_specs".to_string(), json!(specs.len()));
    root.insert("stats".to_string(), temp_matrix_stats(&specs));
    root.insert("categories".to_string(), temp_matrix_categories(&specs));
    root.insert(
        "specs_jcs_sha256".to_string(),
        json!(temp_sha256_jcs_value(&Value::Object(specs))),
    );
}

fn temp_matrix_stats(specs: &serde_json::Map<String, Value>) -> Value {
    let mut ty_fail = 0usize;
    let mut ty_match = 0usize;
    let mut ty_mismatch = 0usize;
    let mut ty_untested = 0usize;
    let mut tlc_error = 0usize;
    let mut tlc_pass = 0usize;
    let mut tlc_timeout = 0usize;

    for spec in specs.values() {
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

    json!({
        "ty_fail": ty_fail,
        "ty_match": ty_match,
        "ty_mismatch": ty_mismatch,
        "ty_untested": ty_untested,
        "tlc_error": tlc_error,
        "tlc_pass": tlc_pass,
        "tlc_timeout": tlc_timeout,
    })
}

fn temp_matrix_categories(specs: &serde_json::Map<String, Value>) -> Value {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for spec in specs.values() {
        let category = spec
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        *counts.entry(category.to_string()).or_default() += 1;
    }
    let categories = counts
        .into_iter()
        .map(|(category, count)| (category, json!(count)))
        .collect();
    Value::Object(categories)
}

fn temp_sha256_jcs_value(value: &Value) -> String {
    let mut canonical = String::new();
    temp_write_canonical_json(value, &mut canonical);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn temp_write_canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => out.push_str(&serde_json::to_string(text).unwrap()),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                temp_write_canonical_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(left, _)| *left);
            out.push('{');
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap());
                out.push(':');
                temp_write_canonical_json(item, out);
            }
            out.push('}');
        }
    }
}

fn pass_only_matrix_baseline() -> Value {
    json!({
        "specs": {
            "FastPassA": {
                "source": {"mode": "check"},
                "tlc": {"status": "pass", "runtime_seconds": 3, "states": 10, "error_type": null},
                "ty": {"status": "pass", "runtime_seconds": 1, "states": 10, "error_type": null},
                "verified_match": true
            },
            "FastPassB": {
                "source": {"mode": "check"},
                "tlc": {"status": "pass", "runtime_seconds": 4, "states": 11, "error_type": null},
                "ty": {"status": "pass", "runtime_seconds": 1, "states": 11, "error_type": null},
                "verified_match": true
            },
            "FastPassC": {
                "source": {"mode": "check"},
                "tlc": {"status": "pass", "runtime_seconds": 5, "states": 12, "error_type": null},
                "ty": {"status": "pass", "runtime_seconds": 1, "states": 12, "error_type": null},
                "verified_match": true
            },
            "FastPassD": {
                "source": {"mode": "check"},
                "tlc": {"status": "pass", "runtime_seconds": 6, "states": 13, "error_type": null},
                "ty": {"status": "pass", "runtime_seconds": 1, "states": 13, "error_type": null},
                "verified_match": true
            }
        }
    })
}

fn matrix_row_json(row: &matrix::SupremacyMatrixRow) -> Value {
    json!({
        "spec": row.spec.clone(),
        "class": row.class,
        "reason": row.reason.clone(),
    })
}

// Test scaffolding: builds an all-pass matrix-row set for the default corpus.
#[allow(dead_code)]
fn default_passing_matrix_rows(prepared: &PreparedSupremacy) -> Vec<Value> {
    default_matrix_spec_names(prepared)
        .into_iter()
        .map(|spec| json!({"spec": spec, "class": "pass", "reason": "faster"}))
        .collect()
}

fn matrix_counts(pass: usize, overrides: &[(&str, usize)]) -> Value {
    let mut counts = json!({
        "unsupported": 0,
        "expected_violation_match": 0,
        "tlc_error": 0,
        "tlc_timeout": 0,
        "runtime_to_error": 0,
        "timeout_dominance": 0,
        "ty_timeout": 0,
        "parity_fail": 0,
        "missing_runtime": 0,
        "perf_tie": 0,
        "perf_loser": 0,
        "pass": pass,
    });
    for (field, count) in overrides {
        counts[*field] = json!(*count);
    }
    counts
}

fn matrix_summary(
    prepared: &PreparedSupremacy,
    binary_root: &Path,
    rows: Vec<Value>,
    counts: Value,
    strict_pass: bool,
    strict_blockers: u64,
    verdict: &str,
    policy_summary: Option<Value>,
) -> Value {
    let (binary_path, binary_sha256) = test_binary_identity(binary_root);
    let (total_specs, specs_jcs_sha256) = default_matrix_corpus_identity(prepared);
    assert_eq!(rows.len(), total_specs);
    let mut summary = json!({
        "schema": MATRIX_SUMMARY_SCHEMA,
        "verdict": verdict,
        "strict_pass": strict_pass,
        "strict_blockers": strict_blockers,
        "corpus": {
            "total_specs": total_specs,
            "specs_jcs_sha256": specs_jcs_sha256,
        },
        "build_identity": {
            "git_commit": current_git_commit(prepared).unwrap_or_else(|| "unknown".to_string()),
            "timestamp": "2026-04-29T120000Z",
            "ty_binary_path": binary_path,
            "ty_binary_sha256": binary_sha256,
            "allow_debug_runtime": false,
        },
        "counts": counts,
        "rows": rows,
    });
    if let Some(policy_summary) = policy_summary {
        summary["policy"] = policy_summary;
    }
    summary
}

fn assert_no_temp_matrix_fixture_noise(verdict: &PolicyVerdict) {
    for forbidden in [
        "all-runnable matrix evidence must cover more than",
        "matrix baseline promotion metadata is stale",
        "could not be verified against enforceable baseline",
        "matrix summary corpus.specs_jcs_sha256",
        "matrix summary corpus.total_specs",
    ] {
        assert!(
            !verdict.errors.iter().any(|error| error.contains(forbidden)),
            "unexpected fixture noise {forbidden:?} in {:?}",
            verdict.errors
        );
    }
}

fn telemetry(prepared: &PreparedSupremacy, spec: &str) -> Value {
    let expected_states = expected_states(prepared, spec);
    let expected_generated = expected_generated(prepared, spec);
    let (actions, invariants, mode, state_len, state_constraints) = match spec {
        "CoffeeCan1000BeansSafety" => (4, 1, "invariant_checking", 2, 0),
        "EWD998Small" => (15, 3, "state_constraint_checking", 15, 1),
        "MCLamportMutex" => (27, 3, "state_constraint_checking", 89, 1),
        other => panic!("unexpected spec {other}"),
    };
    let mut telemetry = json!({
        "trust_cg_actions_compiled": actions,
        "trust_cg_actions_total": actions,
        "trust_cg_invariants_compiled": invariants,
        "trust_cg_invariants_total": invariants,
        "compiled_bfs_level_loop_started": true,
        "compiled_bfs_level_loop_fused": true,
        "compiled_bfs_level_loop_initial_states": 1,
        "compiled_bfs_levels_completed": 1,
        "compiled_bfs_parents_processed": 1,
        "compiled_bfs_successors_generated": expected_generated,
        "compiled_bfs_successors_new": expected_states.saturating_sub(1).max(1),
        "compiled_bfs_execution_nanos": 1_000_000_000u64,
        "compiled_bfs_execution_seconds": 1.0,
        "compiled_bfs_total_states": expected_states,
        "trust_cg_native_fused_level_built": true,
        "trust_cg_native_fused_level_active": true,
        "trust_cg_bfs_level_loop_kind": "native_fused_trust_cg_parent_loop",
        "transitions": expected_generated,
        "trust_cg_native_fused_regular_invariants_checked": true,
        "trust_cg_native_fused_invariant_count": invariants,
        "trust_cg_native_fused_mode": mode,
        "trust_cg_native_fused_state_len": state_len,
        "trust_cg_native_fused_state_constraint_count": state_constraints,
        "trust_cg_native_fused_local_dedup": true,
        "trust_cg_native_fused_flat_frontier_admission_active": false,
        "compiled_bfs_flat_frontier_admitted": true,
        "flat_state_primary": true,
        "flat_bfs_frontier_active": true,
        "flat_bfs_frontier_fallbacks": 0,
        "fallback_reasons": [],
    });
    if state_constraints > 0 {
        telemetry["trust_cg_state_constraints_compiled"] = json!(state_constraints);
        telemetry["trust_cg_state_constraints_total"] = json!(state_constraints);
    }
    telemetry
}

fn runs(prepared: &PreparedSupremacy, spec: &str, mode: &str) -> Vec<Value> {
    let expected_states = expected_states(prepared, spec);
    let expected_generated = expected_generated(prepared, spec);
    let env = required_env_with_cache(prepared);
    (1..=3)
        .map(|run_index| {
            let mut run = json!({
                "tool": if mode == "tlc" { "tlc" } else { "ty" },
                "spec_name": spec,
                "run_index": run_index,
                "states_found": expected_states,
                "elapsed_seconds": if mode == "tlc" { 3.0 } else { 2.0 },
                "returncode": 0,
                "error": null,
                "workers": 1,
                "artifact_dir": format!("artifacts/{spec}/{mode}-{run_index}"),
            });
            if mode == "tlc" {
                run["states_generated"] = json!(expected_generated + 1);
                run["transitions"] = json!(expected_generated);
            } else {
                run["transitions"] = json!(expected_generated);
            }
            if mode != "tlc" {
                run["mode"] = json!(mode);
            }
            if mode == "interp" {
                run["env_overrides"] = json!(interp_env());
            }
            if mode == "trust-cg" {
                run["env_overrides"] = json!(env);
                run["trust_cg_telemetry"] = telemetry(prepared, spec);
            }
            run
        })
        .collect()
}

fn interp_env() -> BTreeMap<String, String> {
    // No TY_AUTO_POR / TY_AUTO_SYMMETRY pins: count-parity is the
    // `--no-reduction` CLI flag in the recorded argv, not env.
    BTreeMap::from([
        ("TY_BYTECODE_VM".to_string(), "1".to_string()),
        ("TY_trust_cg".to_string(), "0".to_string()),
        ("TY_TRUST_CG_BFS".to_string(), "0".to_string()),
    ])
}

fn required_env_with_cache(prepared: &PreparedSupremacy) -> BTreeMap<String, String> {
    let mut env = prepared.gate_plan.as_ref().unwrap().enforce_required_env();
    env.insert(
        "TY_CACHE_DIR".to_string(),
        prepared
            .output_dir
            .join("trust_cg-artifact-cache")
            .display()
            .to_string(),
    );
    env
}

fn row(prepared: &PreparedSupremacy, spec: &str) -> Value {
    let expected_states = expected_states(prepared, spec);
    json!({
        "spec": spec,
        "tlc": {
            "all_ok": true,
            "median_seconds": 3.0,
            "expected_states": expected_states,
            "runs": runs(prepared, spec, "tlc"),
        },
        "interp": {
            "all_ok": true,
            "median_seconds": 2.0,
            "expected_states": expected_states,
            "runs": runs(prepared, spec, "interp"),
        },
        "trust_cg": {
            "all_ok": true,
            "median_seconds": 2.0,
            "execution_median_seconds": 1.0,
            "expected_states": expected_states,
            "runs": runs(prepared, spec, "trust-cg"),
        },
        "parity_interp_vs_tlc": true,
        "parity_trust_cg_vs_tlc": true,
        "trust_cg_gate_failures": [],
        "speedup_interp_vs_tlc": 1.5,
        "speedup_trust_cg_vs_tlc": 1.5,
        "speedup_trust_cg_execution_vs_tlc": 3.0,
    })
}

fn summary(prepared: &PreparedSupremacy) -> Value {
    let plan = prepared.gate_plan.as_ref().unwrap();
    let (binary_path, binary_sha256) = test_binary_identity(&prepared.output_dir);
    let mut flags = serde_json::Map::new();
    for flag in &plan.benchmark_flags {
        flags.insert(flag.clone(), Value::Bool(true));
    }
    for flag in &plan.forbidden_benchmark_flags {
        flags.insert(flag.clone(), Value::Bool(false));
    }
    json!({
        "schema": SUMMARY_SCHEMA,
        "timestamp": "2026-04-29T120000",
        "git_commit": current_git_commit(prepared).unwrap_or_else(|| "unknown".to_string()),
        "artifact_bundle": prepared.output_dir.display().to_string(),
        "invocation": "ty supremacy gate --mode enforce --runs 3",
        "build_identity": {
            "cargo_profile": "release",
            "ty_binary_path": binary_path,
            "ty_binary_sha256": binary_sha256,
        },
        "backend_controls": {
            "interp_env": {},
            "trust_cg_env": required_env_with_cache(prepared),
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
                    "cache_dir": prepared.output_dir.join("trust_cg-artifact-cache").display().to_string(),
                    "artifact_cache_disabled_env": "1",
                    "native_callout_compile_jobs": "27",
                },
            },
        },
        "gate_flags": flags,
        "rows": prepared
            .policy
            .specs
            .iter()
            .map(|spec| row(prepared, spec))
            .collect::<Vec<_>>(),
    })
}

fn write_selftest_artifacts(prepared: &PreparedSupremacy, summary_path: &Path, summary: &Value) {
    for row in summary["rows"].as_array().unwrap() {
        let spec = row["spec"].as_str().unwrap();
        let requirement = &prepared
            .gate_plan
            .as_ref()
            .unwrap()
            .required_trust_cg_selftest_by_spec[spec];
        for (row_key, mode) in [
            ("tlc", "tlc"),
            ("interp", "interp"),
            ("trust_cg", "trust-cg"),
        ] {
            for run in row[row_key]["runs"].as_array().unwrap() {
                let artifact_dir = summary_path
                    .parent()
                    .unwrap()
                    .join(run["artifact_dir"].as_str().unwrap());
                fs::create_dir_all(&artifact_dir).unwrap();
                fs::write(
                    artifact_dir.join("command.json"),
                    serde_json::to_string_pretty(&command_artifact(spec, mode, run)).unwrap()
                        + "\n",
                )
                .unwrap();
                let stdout = if mode == "trust-cg" {
                    format!(
                        "{FLAT_PRIMARY_REBUILD_MARKER}\n\
                         [trust_cg-selftest] prepared native fused callout selftest: actions={}, state_constraints={}, invariants={}, missing_expected=0, fail_closed=true\n\
                         [trust_cg-selftest] running native fused callout selftest on first real parent: state_len={}, actions={}, state_constraints={}, invariants={}, fail_closed=true\n\
                         [trust_cg-selftest] native fused callout selftest complete\n",
                        requirement.actions,
                        requirement.state_constraints,
                        requirement.invariants,
                        requirement.state_len,
                        requirement.actions,
                        requirement.state_constraints,
                        requirement.invariants,
                    )
                } else {
                    String::new()
                };
                fs::write(artifact_dir.join("stdout.txt"), stdout).unwrap();
                fs::write(artifact_dir.join("stderr.txt"), "").unwrap();
            }
        }
    }
}

fn command_artifact(spec: &str, mode: &str, run: &Value) -> Value {
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
    json!({
        "schema": "ty.supremacy.command.v1",
        "argv": argv,
        "cwd": "/tmp",
        "returncode": run["returncode"],
        "elapsed_seconds": run["elapsed_seconds"],
        "env_overrides": run.get("env_overrides").cloned().unwrap_or_else(|| json!({})),
        "timed_out": false,
        "peak_rss_bytes": null,
    })
}

fn evaluate_summary(summary: &Value) -> PolicyVerdict {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, summary);
    evaluate(&prepared, &summary_path).unwrap()
}

fn evaluate_summary_after_artifact_edit(
    summary: &Value,
    edit: impl FnOnce(&Path, &Value),
) -> PolicyVerdict {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, summary);
    edit(&summary_path, summary);
    evaluate(&prepared, &summary_path).unwrap()
}

fn first_artifact_stdout(summary_path: &Path, summary: &Value) -> PathBuf {
    summary_path
        .parent()
        .unwrap()
        .join(
            summary["rows"][0]["trust_cg"]["runs"][0]["artifact_dir"]
                .as_str()
                .unwrap(),
        )
        .join("stdout.txt")
}

#[test]
fn benchmarking_doc_keeps_rust_cli_as_launch_gate_without_total_supremacy_claim() {
    let doc = normalized_text(&repo_text("benchmarking.md"));

    for required in [
        "documented launch path for trust-codegen single-thread evidence and launch gates",
        "only `ty supremacy gate --mode enforce --gate-mode full-native-fused` is the final single-thread launch gate",
        "Python/JQ all-runnable classifiers, spec allowlist gates, ad hoc benchmark-verdict scripts, and Python TLC-vs-TY perf gates have been removed as gates",
        "Retired Python benchmark and fast-subset helper paths have been deleted and must not be recreated, sourced, or treated as launch evidence",
        "Historical reports and superseded designs may still quote those deleted command paths as archived provenance, not runnable guidance",
        "Current deletion audit: no Python or JQ source path is accepted as a canary, supremacy, or launch gate",
        "The remaining blockers to deleting the shell compatibility wrappers are operational callers",
        "Generated bytecode caches or archived report text are not executable gate surfaces",
        "Final benchmark collection and policy verdict must run through the Rust CLI",
        "this document does not claim that all-runnable matrix enforcement currently passes",
        "A matrix summary cannot replace the final three-spec single-thread launch command",
    ] {
        assert!(
            doc.contains(required),
            "benchmarking.md missing {required:?}"
        );
    }

    for forbidden in [
        "all-test supremacy has already been achieved",
        "all-test supremacy has been achieved",
        "all-runnable supremacy has already been achieved",
        "Python/JQ allowlists are acceptance gates",
    ] {
        assert!(
            !doc.contains(forbidden),
            "benchmarking.md must not claim {forbidden:?}"
        );
    }
}

#[test]
fn retired_python_benchmark_helpers_do_not_exist() {
    for relative in retired_python_benchmark_helper_paths() {
        assert!(
            !repo_path(relative).exists(),
            "{relative} is retired; use `ty supremacy ...`"
        );
    }
}

#[test]
fn active_benchmark_docs_do_not_link_retired_python_helpers() {
    for doc_path in [
        "benchmarking.md",
        "trust-cg-native-jit-launch-program.md",
    ] {
        let doc = repo_text(doc_path);
        for retired_path in retired_python_benchmark_helper_paths() {
            assert!(
                !doc.contains(retired_path),
                "{doc_path} must not point readers at retired Python gate path {retired_path}"
            );
        }
    }
}

#[test]
fn final_gate_summary_policy_facts_pass() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.passed(), "{:?}", verdict.errors);
}

#[test]
fn final_gate_verdict_reports_holdout_and_cold_wall_controls() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.passed(), "{:?}", verdict.errors);
    let evidence = &verdict.anti_overfit_evidence;
    assert_eq!(
        evidence.launch_corpus.spec_count,
        prepared.policy.specs.len()
    );
    assert_eq!(
        evidence.engine_selection_contract.selection_basis,
        "structural"
    );
    assert!(evidence
        .engine_selection_contract
        .forbidden_selector_inputs
        .contains(&"exact_spec_name_allowlist".to_string()));
    assert!(evidence
        .matrix_holdout
        .covers_more_than_launch_canary
        .unwrap());
    assert!(
        evidence.matrix_holdout.total_specs.unwrap() > evidence.launch_corpus.spec_count,
        "{:?}",
        evidence.matrix_holdout
    );
    assert_eq!(
        evidence
            .matrix_holdout
            .specs_jcs_sha256
            .as_deref()
            .unwrap()
            .len(),
        64
    );
    assert!(evidence.cold_single_thread_wall.artifact_cache_disabled);
    assert_eq!(evidence.cold_single_thread_wall.required_workers, 1);
    assert!(evidence.cold_single_thread_wall.native_fused_strict);
}

#[test]
fn final_gate_rejects_missing_strict_launch_env() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["rows"][0]["trust_cg"]["runs"][0]["env_overrides"]
        .as_object_mut()
        .unwrap()
        .remove("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST");
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict
        .errors
        .iter()
        .any(|error| { error.contains("env_overrides[TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST]") }));
}

#[test]
fn final_gate_accepts_exact_required_launch_env_keys() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);

    let verdict = evaluate_summary(&summary);

    assert!(verdict.passed(), "{:?}", verdict.errors);
}

#[test]
fn final_gate_rejects_missing_or_mismatched_launch_controls() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));

    let mut summary_without_controls = summary(&prepared);
    summary_without_controls
        .as_object_mut()
        .unwrap()
        .remove("launch_controls");
    let verdict = evaluate_summary(&summary_without_controls);
    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("summary.launch_controls missing")));

    let mut summary_with_mismatch = summary(&prepared);
    summary_with_mismatch["launch_controls"]["tlc"]["workers"] = json!(2);
    summary_with_mismatch["launch_controls"]["tlc"]["jvm_args"][0] =
        json!("-XX:ActiveProcessorCount=2");
    summary_with_mismatch["launch_controls"]["ty"]["trust_cg"]["native_callout_compile_jobs"] =
        json!("1");
    summary_with_mismatch["launch_controls"]["ty"]["trust_cg"]["artifact_cache_disabled_env"] =
        json!("0");
    summary_with_mismatch["launch_controls"]["ty"]["trust_cg"]["cache_dir"] = json!("");

    let verdict = evaluate_summary(&summary_with_mismatch);
    for expected in [
        "summary.launch_controls.tlc.workers",
        "summary.launch_controls.tlc.jvm_args",
        "summary.launch_controls.ty.trust_cg.artifact_cache_disabled_env",
        "summary.launch_controls.ty.trust_cg.native_callout_compile_jobs",
        "summary.launch_controls.ty.trust_cg.cache_dir missing or empty",
    ] {
        assert!(
            verdict.errors.iter().any(|error| error.contains(expected)),
            "missing {expected:?} in {:?}",
            verdict.errors
        );
    }
}

#[test]
fn final_gate_rejects_mutated_command_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);

    let verdict = evaluate_summary_after_artifact_edit(&summary, |summary_path, summary| {
        let artifact_dir = summary_path.parent().unwrap().join(
            summary["rows"][0]["interp"]["runs"][0]["artifact_dir"]
                .as_str()
                .unwrap(),
        );
        let command_path = artifact_dir.join("command.json");
        let mut command: Value =
            serde_json::from_str(&fs::read_to_string(&command_path).unwrap()).unwrap();
        command["argv"]
            .as_array_mut()
            .unwrap()
            .insert(8, json!("--max-depth"));
        fs::write(
            command_path,
            serde_json::to_string_pretty(&command).unwrap() + "\n",
        )
        .unwrap();
    });
    assert!(verdict.errors.iter().any(|error| {
        error.contains("interp run 1: command argv length")
            && error.contains("does not permit TY-only flags")
    }));

    let verdict = evaluate_summary_after_artifact_edit(&summary, |summary_path, summary| {
        let artifact_dir = summary_path.parent().unwrap().join(
            summary["rows"][0]["tlc"]["runs"][0]["artifact_dir"]
                .as_str()
                .unwrap(),
        );
        let command_path = artifact_dir.join("command.json");
        let mut command: Value =
            serde_json::from_str(&fs::read_to_string(&command_path).unwrap()).unwrap();
        command["argv"][13] = json!("2");
        fs::write(
            command_path,
            serde_json::to_string_pretty(&command).unwrap() + "\n",
        )
        .unwrap();
    });
    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("tlc run 1: command argv[13]")));
}

#[test]
fn final_gate_rejects_extra_ty_launch_env_keys() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["backend_controls"]["trust_cg_env"]["TY_PARALLEL_READONLY_VALUE_CACHES"] = json!("1");
    summary["rows"][0]["trust_cg"]["runs"][0]["env_overrides"]
        ["TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP"] = json!("1");

    let verdict = evaluate_summary(&summary);

    for expected in [
        "summary.backend_controls.trust_cg_env contains unexpected gate-control env key(s): TY_PARALLEL_READONLY_VALUE_CACHES",
        "env_overrides contains unexpected gate-control env key(s): TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP",
    ] {
        assert!(
            verdict.errors.iter().any(|error| error.contains(expected)),
            "missing {expected:?} in {:?}",
            verdict.errors
        );
    }
}

#[test]
fn final_gate_rejects_summary_without_build_identity() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary.as_object_mut().unwrap().remove("build_identity");
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.errors.iter().any(|error| {
        error.contains("summary.build_identity missing") && error.contains("binary/build identity")
    }));
}

#[test]
fn final_gate_rejects_stale_summary_git_commit() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["git_commit"] = json!("0000000");
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("is stale for current checkout")));
}

#[test]
fn final_gate_summary_json_reports_matrix_blockers_loudly() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = json!({
        "schema": MATRIX_SUMMARY_SCHEMA,
        "verdict": "fail",
        "strict_pass": false,
        "strict_blockers": 3,
        "counts": {
            "unsupported": 1,
            "tlc_error": 0,
            "tlc_timeout": 0,
            "runtime_to_error": 0,
            "timeout_dominance": 0,
            "ty_timeout": 0,
            "parity_fail": 0,
            "missing_runtime": 1,
            "perf_tie": 0,
            "perf_loser": 1,
            "pass": 1
        },
        "rows": [
            {"spec": "UnsupportedSpec", "class": "unsupported", "reason": "unsupported"},
            {"spec": "MissingRuntimeSpec", "class": "missing_runtime", "reason": "missing"},
            {"spec": "PerfLoserSpec", "class": "perf_loser", "reason": "slow"}
        ]
    });
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    for expected in [
        "summary.build_identity missing",
        "unsupported=1",
        "missing_runtime=1",
        "perf_loser=1",
        "strict_blockers was 3",
    ] {
        assert!(
            verdict.errors.iter().any(|error| error.contains(expected)),
            "missing {expected:?} in {:?}",
            verdict.errors
        );
    }
}

#[test]
fn final_gate_allows_expected_violation_match_matrix_rows_as_non_blockers() {
    let dir = tempfile::tempdir().unwrap();
    let mut prepared = prepared(dir.path().join("out"));
    let mut baseline = pass_only_matrix_baseline();
    baseline["specs"]["ExpectedViolation"] = json!({
        "source": {"mode": "check"},
        "tlc": {"status": "fail", "runtime_seconds": 3, "states": 12, "error_type": "invariant"},
        "ty": {"status": "pass", "runtime_seconds": 1, "states": 12, "error_type": "invariant_violation"},
        "verified_match": true
    });
    attach_temp_matrix_baseline(&mut prepared, &dir.path().join("repo"), baseline);
    let (total_specs, _) = default_matrix_corpus_identity(&prepared);
    let baseline_path = matrix_baseline_path(&prepared);
    let expected_summary =
        matrix::classify_baseline_path_with_policy(&baseline_path, &prepared.policy.matrix_policy)
            .unwrap();
    let rows = expected_summary
        .rows
        .iter()
        .map(matrix_row_json)
        .collect::<Vec<_>>();
    let summary = matrix_summary(
        &prepared,
        dir.path(),
        rows,
        matrix_counts(total_specs - 1, &[("expected_violation_match", 1)]),
        true,
        0,
        "pass",
        None,
    );
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(
        !verdict
            .errors
            .iter()
            .any(|error| error.contains("policy verdict requires pass")),
        "{:?}",
        verdict.errors
    );
    assert_no_temp_matrix_fixture_noise(&verdict);
    assert!(
        verdict.errors.iter().any(|error| {
            error.contains("matrix summary artifacts are all-runnable diagnostic evidence")
                && error.contains("cannot satisfy the final launch gate")
        }),
        "{:?}",
        verdict.errors
    );
}

#[test]
fn final_gate_rejects_forged_comparable_row_classes_despite_policy_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let mut prepared = prepared_with_matrix_opt_in(dir.path().join("out"), true, true);
    attach_temp_matrix_baseline(
        &mut prepared,
        &dir.path().join("repo"),
        pass_only_matrix_baseline(),
    );
    let (total_specs, _) = default_matrix_corpus_identity(&prepared);
    let baseline_path = matrix_baseline_path(&prepared);
    let expected_summary =
        matrix::classify_baseline_path_with_policy(&baseline_path, &prepared.policy.matrix_policy)
            .unwrap();
    let mut rows = expected_summary
        .rows
        .iter()
        .map(matrix_row_json)
        .collect::<Vec<_>>();
    assert!(rows.len() >= 2);
    let expected_first_class = serde_json::to_value(expected_summary.rows[0].class).unwrap();
    let expected_second_class = serde_json::to_value(expected_summary.rows[1].class).unwrap();
    rows[0]["class"] = json!("runtime_to_error");
    rows[0]["reason"] = json!("policy permits runtime-to-error comparison");
    rows[1]["class"] = json!("timeout_dominance");
    rows[1]["reason"] = json!("policy permits timeout-dominance comparison");
    let summary = matrix_summary(
        &prepared,
        dir.path(),
        rows,
        matrix_counts(
            total_specs - 2,
            &[("runtime_to_error", 1), ("timeout_dominance", 1)],
        ),
        false,
        2,
        "pass",
        Some(json!({
            "allow_runtime_to_error": true,
            "allow_timeout_dominance": true,
            "comparable_outcomes": 2,
            "pass": true,
            "blockers": 0,
            "verdict": "pass",
        })),
    );
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    for expected in [
        format!(
            "matrix summary row \"FastPassA\" class was \"runtime_to_error\", but recomputed baseline policy class was {}",
            display_value(Some(&expected_first_class))
        ),
        format!(
            "matrix summary row \"FastPassB\" class was \"timeout_dominance\", but recomputed baseline policy class was {}",
            display_value(Some(&expected_second_class))
        ),
    ] {
        assert!(
            verdict.errors.iter().any(|error| error.contains(&expected)),
            "missing {expected:?} in {:?}",
            verdict.errors
        );
    }
    assert!(
        verdict.errors.iter().any(|error| {
            error.contains("matrix summary artifacts are all-runnable diagnostic evidence")
                && error.contains("cannot satisfy the final launch gate")
        }),
        "{:?}",
        verdict.errors
    );
    for forbidden in [
        "runtime_to_error=1",
        "timeout_dominance=1",
        "strict_pass",
        "strict_blockers was 2",
        "policy blockers",
        "matrix summary corpus",
        "promotion metadata",
        "could not be verified against enforceable baseline",
    ] {
        assert!(
            !verdict.errors.iter().any(|error| error.contains(forbidden)),
            "unexpected {forbidden:?} in {:?}",
            verdict.errors
        );
    }
    assert_no_temp_matrix_fixture_noise(&verdict);
}

#[test]
fn final_gate_rejects_forged_comparable_row_classes_without_policy_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let mut prepared = prepared(dir.path().join("out"));
    attach_temp_matrix_baseline(
        &mut prepared,
        &dir.path().join("repo"),
        pass_only_matrix_baseline(),
    );
    let (total_specs, _) = default_matrix_corpus_identity(&prepared);
    let baseline_path = matrix_baseline_path(&prepared);
    let expected_summary =
        matrix::classify_baseline_path_with_policy(&baseline_path, &prepared.policy.matrix_policy)
            .unwrap();
    let mut rows = expected_summary
        .rows
        .iter()
        .map(matrix_row_json)
        .collect::<Vec<_>>();
    assert!(rows.len() >= 2);
    let expected_first_class = serde_json::to_value(expected_summary.rows[0].class).unwrap();
    let expected_second_class = serde_json::to_value(expected_summary.rows[1].class).unwrap();
    rows[0]["class"] = json!("runtime_to_error");
    rows[0]["reason"] = json!("forged runtime-to-error without policy opt-in");
    rows[1]["class"] = json!("timeout_dominance");
    rows[1]["reason"] = json!("forged timeout-dominance without policy opt-in");
    let summary = matrix_summary(
        &prepared,
        dir.path(),
        rows,
        matrix_counts(
            total_specs - 2,
            &[("runtime_to_error", 1), ("timeout_dominance", 1)],
        ),
        false,
        2,
        "pass",
        None,
    );
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    for expected in [
        format!(
            "matrix summary row \"FastPassA\" class was \"runtime_to_error\", but recomputed baseline policy class was {}",
            display_value(Some(&expected_first_class))
        ),
        format!(
            "matrix summary row \"FastPassB\" class was \"timeout_dominance\", but recomputed baseline policy class was {}",
            display_value(Some(&expected_second_class))
        ),
    ] {
        assert!(
            verdict.errors.iter().any(|error| error.contains(&expected)),
            "missing {expected:?} in {:?}",
            verdict.errors
        );
    }
    for expected in [
        "runtime_to_error=1",
        "timeout_dominance=1",
        "policy blockers",
        "class \"runtime_to_error\"",
        "class \"timeout_dominance\"",
    ] {
        assert!(
            verdict.errors.iter().any(|error| error.contains(expected)),
            "missing {expected:?} in {:?}",
            verdict.errors
        );
    }
    assert_no_temp_matrix_fixture_noise(&verdict);
}

#[test]
fn final_gate_rejects_forged_strict_blocker_row_classes_despite_comparable_policy_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let mut prepared = prepared_with_matrix_opt_in(dir.path().join("out"), true, true);
    attach_temp_matrix_baseline(
        &mut prepared,
        &dir.path().join("repo"),
        pass_only_matrix_baseline(),
    );
    let (total_specs, _) = default_matrix_corpus_identity(&prepared);
    let baseline_path = matrix_baseline_path(&prepared);
    let expected_summary =
        matrix::classify_baseline_path_with_policy(&baseline_path, &prepared.policy.matrix_policy)
            .unwrap();
    let mut rows = expected_summary
        .rows
        .iter()
        .map(matrix_row_json)
        .collect::<Vec<_>>();
    assert!(rows.len() >= 4);
    for (idx, class) in [
        "unsupported",
        "parity_fail",
        "missing_runtime",
        "perf_loser",
    ]
    .into_iter()
    .enumerate()
    {
        rows[idx]["class"] = json!(class);
        rows[idx]["reason"] = json!("real blocker");
    }
    let summary = matrix_summary(
        &prepared,
        dir.path(),
        rows,
        matrix_counts(
            total_specs - 4,
            &[
                ("unsupported", 1),
                ("parity_fail", 1),
                ("missing_runtime", 1),
                ("perf_loser", 1),
            ],
        ),
        false,
        4,
        "pass",
        Some(json!({
            "allow_runtime_to_error": true,
            "allow_timeout_dominance": true,
            "comparable_outcomes": 0,
            "pass": false,
            "blockers": 4,
            "verdict": "fail",
        })),
    );
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    for (idx, class) in [
        "unsupported",
        "parity_fail",
        "missing_runtime",
        "perf_loser",
    ]
    .into_iter()
    .enumerate()
    {
        let spec = expected_summary.rows[idx].spec.as_str();
        let expected_class = serde_json::to_value(expected_summary.rows[idx].class).unwrap();
        let expected = format!(
            "matrix summary row {spec:?} class was {class:?}, but recomputed baseline policy class was {}",
            display_value(Some(&expected_class))
        );
        assert!(
            verdict.errors.iter().any(|error| error.contains(&expected)),
            "missing {expected:?} in {:?}",
            verdict.errors
        );
    }
    for expected in [
        "unsupported=1",
        "parity_fail=1",
        "missing_runtime=1",
        "perf_loser=1",
        "policy blockers",
        "class \"unsupported\"",
        "class \"parity_fail\"",
        "class \"missing_runtime\"",
        "class \"perf_loser\"",
    ] {
        assert!(
            verdict.errors.iter().any(|error| error.contains(expected)),
            "missing {expected:?} in {:?}",
            verdict.errors
        );
    }
    assert_no_temp_matrix_fixture_noise(&verdict);
}

#[test]
fn final_gate_rejects_matrix_rows_not_matching_baseline_specs() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let (binary_path, binary_sha256) = test_binary_identity(dir.path());
    let (total_specs, specs_jcs_sha256) = default_matrix_corpus_identity(&prepared);
    let rows = (1..=total_specs)
        .map(
            |idx| json!({"spec": format!("PassingSpec{idx}"), "class": "pass", "reason": "faster"}),
        )
        .collect::<Vec<_>>();
    let summary = json!({
        "schema": MATRIX_SUMMARY_SCHEMA,
        "verdict": "pass",
        "strict_pass": true,
        "strict_blockers": 0,
        "corpus": {
            "total_specs": total_specs,
            "specs_jcs_sha256": specs_jcs_sha256,
        },
        "build_identity": {
            "git_commit": current_git_commit(&prepared).unwrap_or_else(|| "unknown".to_string()),
            "timestamp": "2026-04-29T120000Z",
            "ty_binary_path": binary_path,
            "ty_binary_sha256": binary_sha256,
            "allow_debug_runtime": false,
        },
        "counts": {
            "unsupported": 0,
            "tlc_error": 0,
            "tlc_timeout": 0,
            "ty_timeout": 0,
            "parity_fail": 0,
            "missing_runtime": 0,
            "perf_tie": 0,
            "perf_loser": 0,
            "pass": total_specs
        },
        "rows": rows
    });
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(
        verdict.errors.iter().any(|error| {
            error.contains("matrix summary rows missing baseline specs")
                || error.contains("matrix summary rows contain specs not present in baseline")
        }),
        "{:?}",
        verdict.errors
    );
}

#[test]
fn final_gate_rejects_duplicate_matrix_row_specs() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let (binary_path, binary_sha256) = test_binary_identity(dir.path());
    let (total_specs, specs_jcs_sha256) = default_matrix_corpus_identity(&prepared);
    let duplicate_spec = default_matrix_spec_names(&prepared).remove(0);
    let rows = (0..total_specs)
        .map(|_| json!({"spec": duplicate_spec.clone(), "class": "pass", "reason": "faster"}))
        .collect::<Vec<_>>();
    let summary = json!({
        "schema": MATRIX_SUMMARY_SCHEMA,
        "verdict": "pass",
        "strict_pass": true,
        "strict_blockers": 0,
        "corpus": {
            "total_specs": total_specs,
            "specs_jcs_sha256": specs_jcs_sha256,
        },
        "build_identity": {
            "git_commit": current_git_commit(&prepared).unwrap_or_else(|| "unknown".to_string()),
            "timestamp": "2026-04-29T120000Z",
            "ty_binary_path": binary_path,
            "ty_binary_sha256": binary_sha256,
            "allow_debug_runtime": false,
        },
        "counts": {
            "unsupported": 0,
            "tlc_error": 0,
            "tlc_timeout": 0,
            "ty_timeout": 0,
            "parity_fail": 0,
            "missing_runtime": 0,
            "perf_tie": 0,
            "perf_loser": 0,
            "pass": total_specs
        },
        "rows": rows
    });
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(
        verdict
            .errors
            .iter()
            .any(|error| error.contains("duplicate row for spec")),
        "{:?}",
        verdict.errors
    );
}

#[test]
fn final_gate_rejects_all_pass_matrix_summary_as_launch_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let (binary_path, binary_sha256) = test_binary_identity(dir.path());
    let (total_specs, specs_jcs_sha256) = default_matrix_corpus_identity(&prepared);
    let rows = default_matrix_spec_names(&prepared)
        .into_iter()
        .map(|spec| json!({"spec": spec, "class": "pass", "reason": "faster"}))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), total_specs);
    let summary = json!({
        "schema": MATRIX_SUMMARY_SCHEMA,
        "verdict": "pass",
        "strict_pass": true,
        "strict_blockers": 0,
        "corpus": {
            "total_specs": total_specs,
            "specs_jcs_sha256": specs_jcs_sha256,
        },
        "build_identity": {
            "git_commit": current_git_commit(&prepared).unwrap_or_else(|| "unknown".to_string()),
            "timestamp": "2026-04-29T120000Z",
            "ty_binary_path": binary_path,
            "ty_binary_sha256": binary_sha256,
            "allow_debug_runtime": false,
        },
        "counts": {
            "unsupported": 0,
            "tlc_error": 0,
            "tlc_timeout": 0,
            "runtime_to_error": 0,
            "timeout_dominance": 0,
            "ty_timeout": 0,
            "parity_fail": 0,
            "missing_runtime": 0,
            "perf_tie": 0,
            "perf_loser": 0,
            "pass": total_specs
        },
        "rows": rows
    });
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(
        verdict.errors.iter().any(|error| {
            error.contains("matrix summary artifacts are all-runnable diagnostic evidence")
                && error.contains("cannot satisfy the final launch gate")
        }),
        "{:?}",
        verdict.errors
    );
}

#[test]
fn final_gate_rejects_matrix_digest_that_does_not_match_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let (binary_path, binary_sha256) = test_binary_identity(dir.path());
    let (total_specs, _) = default_matrix_corpus_identity(&prepared);
    let rows = (1..=total_specs)
        .map(|idx| json!({"spec": format!("PassingSpec{idx}"), "class": "pass"}))
        .collect::<Vec<_>>();
    let summary = json!({
        "schema": MATRIX_SUMMARY_SCHEMA,
        "verdict": "pass",
        "strict_pass": true,
        "strict_blockers": 0,
        "corpus": {
            "total_specs": total_specs,
            "specs_jcs_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        },
        "build_identity": {
            "git_commit": current_git_commit(&prepared).unwrap_or_else(|| "unknown".to_string()),
            "timestamp": "2026-04-29T120000Z",
            "ty_binary_path": binary_path,
            "ty_binary_sha256": binary_sha256,
            "allow_debug_runtime": false,
        },
        "counts": {
            "unsupported": 0,
            "tlc_error": 0,
            "tlc_timeout": 0,
            "ty_timeout": 0,
            "parity_fail": 0,
            "missing_runtime": 0,
            "perf_tie": 0,
            "perf_loser": 0,
            "pass": total_specs
        },
        "rows": rows
    });
    let summary_path = dir.path().join("matrix_summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.errors.iter().any(|error| {
        error.contains("matrix summary corpus.specs_jcs_sha256") && error.contains("but baseline")
    }));
}

#[test]
fn relative_summary_binary_path_prefers_repo_root_over_artifact_shadow() {
    let dir = tempfile::tempdir().unwrap();
    let mut prepared = prepared(dir.path().join("out"));
    let repo_root = dir.path().join("repo");
    let policy_path = repo_root
        .join("tests")
        .join("tlc_comparison")
        .join("single_thread_supremacy_gate.json");
    fs::create_dir_all(policy_path.parent().unwrap()).unwrap();
    fs::write(&policy_path, "{}").unwrap();
    prepared.policy_path = policy_path;

    let relative = PathBuf::from("target/user/release/ty");
    let repo_binary = repo_root.join(&relative);
    let summary_dir = dir.path().join("artifact");
    let shadow_binary = summary_dir.join(&relative);
    fs::create_dir_all(repo_binary.parent().unwrap()).unwrap();
    fs::create_dir_all(shadow_binary.parent().unwrap()).unwrap();
    fs::write(&repo_binary, b"repo binary").unwrap();
    fs::write(&shadow_binary, b"artifact shadow").unwrap();
    let summary_path = summary_dir.join("summary.json");

    let resolved =
        resolve_summary_artifact_path(&prepared, &summary_path, relative.to_str().unwrap());

    assert_eq!(resolved, repo_binary);
}

#[test]
fn default_relative_policy_path_resolves_current_git_commit() {
    let dir = tempfile::tempdir().unwrap();
    let mut prepared = prepared(dir.path().join("out"));
    prepared.policy_path = PathBuf::from("tests/tlc_comparison/single_thread_supremacy_gate.json");

    assert!(current_git_commit(&prepared).is_some());
}

#[test]
fn final_gate_rejects_row_gate_failure_markers() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["rows"][0]["trust_cg_gate_failures"] = json!(["unsupported_native_callout"]);

    let verdict = evaluate_summary(&summary);

    assert!(verdict.errors.iter().any(|error| {
        error.contains("trust_cg_gate_failures contains 1 failure")
            && error.contains("unsupported_native_callout")
    }));
}

#[test]
fn final_gate_rejects_unpinned_extra_summary_rows() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary_path = dir.path().join("summary.json");
    let mut summary = summary(&prepared);
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let mut extra = row(&prepared, "CoffeeCan1000BeansSafety");
    extra["spec"] = json!("UnpinnedBroadClaimSpec");
    summary["rows"].as_array_mut().unwrap().push(extra);
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.errors.iter().any(|error| {
        error.contains("benchmark summary rows contain unpinned spec(s): UnpinnedBroadClaimSpec")
            && error.contains("exactly the policy corpus")
    }));
}

#[test]
fn final_gate_rejects_missing_advertised_wall_median() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["rows"][0]["tlc"]
        .as_object_mut()
        .unwrap()
        .remove("median_seconds");

    let verdict = evaluate_summary(&summary);

    assert!(verdict.errors.iter().any(|error| {
        error.contains(
            "CoffeeCan1000BeansSafety: tlc advertised median_seconds was missing; wall-clock median evidence is required",
        )
    }));
}

#[test]
fn final_gate_does_not_treat_tlc_generated_states_as_ty_transitions() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    for row in summary["rows"].as_array_mut().unwrap() {
        for run in row["tlc"]["runs"].as_array_mut().unwrap() {
            run.as_object_mut().unwrap().remove("transitions");
            run["states_generated"] = json!(99_999_999_u64);
        }
    }
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.passed(), "{:?}", verdict.errors);
}

#[test]
fn final_gate_accepts_native_fused_flat_frontier_admission_when_flat_primary_false() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    for row in summary["rows"].as_array_mut().unwrap() {
        for run in row["trust_cg"]["runs"].as_array_mut().unwrap() {
            run["trust_cg_telemetry"]["flat_state_primary"] = json!(false);
            run["trust_cg_telemetry"]["trust_cg_native_fused_flat_frontier_admission_active"] =
                json!(true);
            run["trust_cg_telemetry"]["compiled_bfs_flat_frontier_admitted"] = json!(true);
        }
    }
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.passed(), "{:?}", verdict.errors);
}

#[test]
fn final_gate_allows_state_constrained_native_generated_counter_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    let ewd = summary["rows"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["spec"].as_str() == Some("EWD998Small"))
        .unwrap();
    for run in ewd["trust_cg"]["runs"].as_array_mut().unwrap() {
        run["transitions"] = json!(8_900_429_u64);
        run["trust_cg_telemetry"]["transitions"] = json!(8_900_429_u64);
        run["trust_cg_telemetry"]["compiled_bfs_successors_generated"] = json!(8_900_429_u64);
    }
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.passed(), "{:?}", verdict.errors);
}

#[test]
fn final_gate_rejects_spec_name_only_generated_count_waiver_without_telemetry() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let expected_generated = expected_generated(&prepared, "EWD998Small");
    let mismatched_generated = 8_900_429_u64;
    let mut summary = summary(&prepared);
    let ewd = summary["rows"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["spec"].as_str() == Some("EWD998Small"))
        .unwrap();
    for run in ewd["trust_cg"]["runs"].as_array_mut().unwrap() {
        run["transitions"] = json!(mismatched_generated);
        run["trust_cg_telemetry"]["transitions"] = json!(mismatched_generated);
        run["trust_cg_telemetry"]["compiled_bfs_successors_generated"] =
            json!(mismatched_generated);
        run["trust_cg_telemetry"]["trust_cg_native_fused_mode"] = json!("invariant_checking");
        run["trust_cg_telemetry"]["trust_cg_native_fused_state_constraint_count"] = json!(0);
    }
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.errors.iter().any(|error| {
            error.contains(&format!(
                "EWD998Small: trust-cg run 1: transitions was {mismatched_generated}, expected {expected_generated}"
            ))
        }));
    assert!(verdict.errors.iter().any(|error| {
        error.contains(&format!(
            "EWD998Small: generated-state parity failed at run 1: interp={expected_generated}"
        ))
    }));
}

#[test]
fn final_gate_rejects_state_constraint_waiver_without_compiled_constraint_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let expected_generated = expected_generated(&prepared, "EWD998Small");
    let mismatched_generated = 8_900_429_u64;
    let mut summary = summary(&prepared);
    let ewd = summary["rows"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["spec"].as_str() == Some("EWD998Small"))
        .unwrap();
    for run in ewd["trust_cg"]["runs"].as_array_mut().unwrap() {
        run["transitions"] = json!(mismatched_generated);
        let telemetry = run["trust_cg_telemetry"].as_object_mut().unwrap();
        telemetry.insert("transitions".to_string(), json!(mismatched_generated));
        telemetry.insert(
            "compiled_bfs_successors_generated".to_string(),
            json!(mismatched_generated),
        );
        telemetry.insert(
            "trust_cg_native_fused_mode".to_string(),
            json!("state_constraint_checking"),
        );
        telemetry.insert(
            "trust_cg_native_fused_state_constraint_count".to_string(),
            json!(1),
        );
        telemetry.insert("trust_cg_state_constraints_compiled".to_string(), json!(0));
        telemetry.insert("trust_cg_state_constraints_total".to_string(), json!(1));
    }
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.errors.iter().any(|error| {
        error.contains(
            "state-constrained generated-count waiver requires active native fused state constraints",
        ) && error.contains(
            "trust_cg_state_constraints_compiled did not match native state constraint count",
        )
    }));
    assert!(verdict.errors.iter().any(|error| {
        error.contains(&format!(
            "EWD998Small: trust-cg run 1: transitions was {mismatched_generated}, expected {expected_generated}"
        ))
    }));
    assert!(verdict.errors.iter().any(|error| {
        error.contains(&format!(
            "EWD998Small: generated-state parity failed at run 1: interp={expected_generated}"
        ))
    }));
}

#[test]
fn final_gate_rejects_worker_state_and_generated_drift() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["rows"][0]["tlc"]["runs"][0]["workers"] = json!(2);
    summary["rows"][0]["parity_trust_cg_vs_tlc"] = json!(false);
    summary["rows"][0]["interp"]["runs"][1]["transitions"] = json!(42);
    let summary_path = dir.path().join("summary.json");
    fs::write(&summary_path, serde_json::to_string(&summary).unwrap()).unwrap();
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("tlc run 1: workers was 2, expected 1")));
    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("trust-cg parity drift vs TLC")));
    assert!(verdict
        .errors
        .iter()
        .any(|error| { error.contains("interp run 2: transitions was 42") }));
}

#[test]
fn final_gate_rejects_native_flat_fallback_and_speedup_drift() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    let telemetry = &mut summary["rows"][0]["trust_cg"]["runs"][0]["trust_cg_telemetry"];
    telemetry["trust_cg_native_fused_level_active"] = json!(false);
    telemetry["flat_state_primary"] = json!(false);
    telemetry["trust_cg_native_fused_flat_frontier_admission_active"] = json!(false);
    telemetry["compiled_bfs_flat_frontier_admitted"] = json!(false);
    telemetry["flat_bfs_frontier_fallbacks"] = json!(1);
    telemetry["fallback_reasons"] = json!(["[trust-cg] requested interpreter fallback"]);
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
    write_selftest_artifacts(&prepared, &summary_path, &summary);

    let verdict = evaluate(&prepared, &summary_path).unwrap();

    assert!(verdict.errors.iter().any(|error| {
        error.contains("trust_cg_native_fused_level_active] was false, expected true")
    }));
    assert!(verdict.errors.iter().any(|error| {
        error.contains("native fused flat frontier proof was flat_state_primary=false")
    }));
    assert!(verdict.errors.iter().any(|error| {
        error.contains("compiled_bfs_flat_frontier_admitted] was false, expected true")
    }));
    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("flat_bfs_frontier_fallbacks] was 1, expected 0")));
    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("trust-codegen fallback reasons observed (1)")));
    assert!(verdict.errors.iter().any(|error| {
        error.contains("speedup_trust_cg_vs_tlc was 0.750000, expected > 1.000000")
    }));
}

#[test]
fn final_gate_rejects_missing_required_compilation_totals() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    let telemetry = &mut summary["rows"][0]["trust_cg"]["runs"][0]["trust_cg_telemetry"];
    telemetry
        .as_object_mut()
        .unwrap()
        .remove("trust_cg_actions_total");
    telemetry
        .as_object_mut()
        .unwrap()
        .remove("trust_cg_actions_compiled");

    let verdict = evaluate_summary(&summary);

    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("telemetry[trust_cg_actions_total] was missing")));
}

#[test]
fn final_gate_rejects_missing_all_requirement_actual_and_total() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    let telemetry = &mut summary["rows"][0]["trust_cg"]["runs"][0]["trust_cg_telemetry"];
    telemetry
        .as_object_mut()
        .unwrap()
        .remove("trust_cg_native_fused_invariant_count");
    telemetry
        .as_object_mut()
        .unwrap()
        .remove("trust_cg_invariants_total");

    let verdict = evaluate_summary(&summary);

    assert!(verdict.errors.iter().any(|error| {
            error.contains(
                "telemetry[trust_cg_native_fused_invariant_count] was missing, expected trust_cg_invariants_total missing",
            )
        }));
}

#[test]
fn final_gate_allows_full_native_low_trust_cg_interp_ratio() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["rows"][0]["trust_cg"]["median_seconds"] = json!(2.9);
    for run in summary["rows"][0]["trust_cg"]["runs"]
        .as_array_mut()
        .unwrap()
    {
        run["elapsed_seconds"] = json!(2.9);
    }
    summary["rows"][0]["speedup_trust_cg_vs_tlc"] = json!(3.0 / 2.9);

    let verdict = evaluate_summary(&summary);

    assert!(
        !verdict.errors.iter().any(|error| {
            error.contains("speedup_interp_vs_tlc") || error.contains("trust_cg_vs_interp_ratio")
        }),
        "{:?}",
        verdict.errors
    );
}

#[test]
fn final_gate_rejects_advertised_trust_cg_execution_speedup_drift() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["rows"][0]["speedup_trust_cg_execution_vs_tlc"] = json!(99.0);

    let verdict = evaluate_summary(&summary);

    assert!(verdict.errors.iter().any(|error| {
        error.contains("advertised speedup_trust_cg_execution_vs_tlc 99.0 did not match")
    }));
}

#[test]
fn final_gate_rejects_missing_advertised_trust_cg_execution_median_when_required() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let mut summary = summary(&prepared);
    summary["rows"][0]["trust_cg"]
        .as_object_mut()
        .unwrap()
        .remove("execution_median_seconds");

    let verdict = evaluate_summary(&summary);

    assert!(verdict
        .errors
        .iter()
        .any(|error| { error.contains("advertised execution_median_seconds was missing") }));
}

#[test]
fn final_gate_accepts_selftest_markers_without_rebuild_marker() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);

    let verdict = evaluate_summary_after_artifact_edit(&summary, |summary_path, summary| {
        let stdout = first_artifact_stdout(summary_path, summary);
        let text = fs::read_to_string(&stdout).unwrap();
        fs::write(stdout, text.replace(FLAT_PRIMARY_REBUILD_MARKER, "")).unwrap();
    });

    assert!(verdict.passed(), "{:?}", verdict.errors);
}

#[test]
fn final_gate_rejects_stale_pre_rebuild_selftest_markers() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);

    let verdict = evaluate_summary_after_artifact_edit(&summary, |summary_path, summary| {
        let stdout = first_artifact_stdout(summary_path, summary);
        let text = fs::read_to_string(&stdout).unwrap();
        let stale_only = format!("{text}\n{FLAT_PRIMARY_REBUILD_MARKER}\n");
        fs::write(stdout, stale_only).unwrap();
    });

    assert!(verdict.errors.iter().any(|error| {
        error.contains("strict native fused selftest markers missing or mismatched")
    }));
}

#[test]
fn final_gate_rejects_nonzero_missing_expected_selftest_marker() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);

    let verdict = evaluate_summary_after_artifact_edit(&summary, |summary_path, summary| {
        let stdout = first_artifact_stdout(summary_path, summary);
        let text = fs::read_to_string(&stdout)
            .unwrap()
            .replace("missing_expected=0", "missing_expected=2");
        fs::write(stdout, text).unwrap();
    });

    assert!(verdict
        .errors
        .iter()
        .any(|error| { error.contains("reported missing expected callouts: 2") }));
}

#[test]
fn final_gate_rejects_false_strict_selftest_result() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);

    let verdict = evaluate_summary_after_artifact_edit(&summary, |summary_path, summary| {
        let stdout = first_artifact_stdout(summary_path, summary);
        let mut text = fs::read_to_string(&stdout).unwrap();
        text.push_str("[trust_cg-selftest] state_constraint callout status=Ok value=0\n");
        fs::write(stdout, text).unwrap();
    });

    assert!(verdict
        .errors
        .iter()
        .any(|error| { error.contains("reported false strict check: kind=state_constraint") }));
}

#[test]
fn final_gate_rejects_selftest_failure_marker() {
    let dir = tempfile::tempdir().unwrap();
    let prepared = prepared(dir.path().join("out"));
    let summary = summary(&prepared);

    let verdict = evaluate_summary_after_artifact_edit(&summary, |summary_path, summary| {
        let stdout = first_artifact_stdout(summary_path, summary);
        let mut text = fs::read_to_string(&stdout).unwrap();
        text.push_str("[trust_cg-selftest] native fused callout selftest failed: failing closed\n");
        fs::write(stdout, text).unwrap();
    });

    assert!(verdict
        .errors
        .iter()
        .any(|error| error.contains("failure marker was present")));
}
