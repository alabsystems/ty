// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{SystemTime, UNIX_EPOCH};
use tla_core::{lower, parse_to_syntax_tree, FileId};

const SPEC_SRC: &str = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = (x + 1) % 3
TypeOK == x \in {0, 1, 2}
====
"#;

const CFG_SRC: &str = "INIT Init\nNEXT Next\nINVARIANT TypeOK\n";
const PROGRESS_SPEC_SRC: &str = r#"
---- MODULE ProgressStates ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x < 1500 /\ x' = x + 1
TypeOK == x \in 0..1500
====
"#;
const PROGRESS_CFG_SRC: &str = "INIT Init\nNEXT Next\nINVARIANT TypeOK\nCHECK_DEADLOCK FALSE\n";
const AY_ACCEPTED_ANALYTICAL_RECEIPT_ROW: &str = "TY shared_engine_validation_receipt receipt_role=analytical_solve lane=bmc backend_code=ay_smt solver_family=ay validator_kind=ay_proof receipt_identity=shared_engine.validation_receipt:ok digest_algorithm=ay_fingerprint_identity digest=ay.proof.fingerprint:ok receipt_status=accepted receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready";

fn parse_module(src: &str) -> tla_core::ast::Module {
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    lower_result.module.expect("module should lower")
}

fn make_temp_base_dir() -> PathBuf {
    let mut base_dir = std::env::temp_dir();
    let unique_suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after unix epoch")
        .as_nanos();
    base_dir.push(format!(
        "tla-cli-runner-test-{}-{unique_suffix}",
        std::process::id()
    ));
    fs::create_dir_all(&base_dir).expect("create temp base directory");
    base_dir
}

fn write_spec_and_cfg(base_dir: &Path, stem: &str) -> (PathBuf, PathBuf) {
    let spec_path = base_dir.join(format!("{stem}.tla"));
    let cfg_path = base_dir.join(format!("{stem}.cfg"));
    fs::write(&spec_path, SPEC_SRC).expect("write spec");
    fs::write(&cfg_path, CFG_SRC).expect("write cfg");
    (spec_path, cfg_path)
}

fn make_progress_case(base_dir: &Path) -> (tla_core::ast::Module, Config, PathBuf, PathBuf) {
    let module = parse_module(PROGRESS_SPEC_SRC);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["TypeOK".to_string()],
        check_deadlock: false,
        ..Default::default()
    };
    let spec_path = base_dir.join("ProgressStates.tla");
    let cfg_path = base_dir.join("ProgressStates.cfg");
    fs::write(&spec_path, PROGRESS_SPEC_SRC).expect("write spec");
    fs::write(&cfg_path, PROGRESS_CFG_SRC).expect("write cfg");
    (module, config, spec_path, cfg_path)
}

struct RunCase<'a> {
    file: &'a Path,
    config_path: &'a Path,
    checkpoint_dir: Option<PathBuf>,
    resume_from: Option<PathBuf>,
    max_states: usize,
}

fn run_once(
    module: &tla_core::ast::Module,
    config: &Config,
    case: RunCase<'_>,
) -> Result<(CheckResult, Option<String>)> {
    let no_progress: Box<dyn Fn(&Progress) + Send + Sync> = Box::new(|_| {});
    run_once_with_output(module, config, case, OutputFormat::Json, no_progress)
}

fn run_once_with_output(
    module: &tla_core::ast::Module,
    config: &Config,
    case: RunCase<'_>,
    output_format: OutputFormat,
    progress_callback: Box<dyn Fn(&Progress) + Send + Sync>,
) -> Result<(CheckResult, Option<String>)> {
    let checker_modules: [&tla_core::ast::Module; 0] = [];
    let no_storage: Option<std::sync::Arc<dyn tla_check::FingerprintSet>> = None;
    let resolved_spec: Option<tla_check::ResolvedSpec> = None;
    let fairness: Vec<tla_check::FairnessConstraint> = Vec::new();
    let mut tool_out: Option<tlc_tool::TlcToolOutput> = None;

    run_model_checker(ModelCheckerRunCfg {
        module,
        checker_modules: &checker_modules,
        config,
        workers: 1,
        file: case.file,
        file_paths: Vec::new(),
        resolved_spec: &resolved_spec,
        check_deadlock: false,
        show_coverage: false,
        strict_vacuity: false,
        continue_on_error: false,
        store_states: false,
        no_trace: false,
        fingerprint_storage: &no_storage,
        trace_file: None,
        trace_locs_storage: None,
        resolved_fairness: &fairness,
        max_states: case.max_states,
        max_depth: 0,
        memory_limit: 0,
        disk_limit: 0,
        output_format,
        progress_callback,
        checkpoint_dir: &case.checkpoint_dir,
        checkpoint_interval: 0,
        resume_from: &case.resume_from,
        config_path: case.config_path,
        tool_out: &mut tool_out,
        collision_check_mode: Default::default(),
    })
}

fn run_once_with_workers(
    module: &tla_core::ast::Module,
    config: &Config,
    case: RunCase<'_>,
    workers: usize,
) -> Result<(CheckResult, Option<String>)> {
    run_once_with_workers_and_strict(module, config, case, workers, false)
}

fn run_once_with_workers_and_strict(
    module: &tla_core::ast::Module,
    config: &Config,
    case: RunCase<'_>,
    workers: usize,
    strict_vacuity: bool,
) -> Result<(CheckResult, Option<String>)> {
    let checker_modules: [&tla_core::ast::Module; 0] = [];
    let no_storage: Option<std::sync::Arc<dyn tla_check::FingerprintSet>> = None;
    let resolved_spec: Option<tla_check::ResolvedSpec> = None;
    let fairness: Vec<tla_check::FairnessConstraint> = Vec::new();
    let mut tool_out: Option<tlc_tool::TlcToolOutput> = None;
    let no_progress: Box<dyn Fn(&Progress) + Send + Sync> = Box::new(|_| {});

    run_model_checker(ModelCheckerRunCfg {
        module,
        checker_modules: &checker_modules,
        config,
        workers,
        file: case.file,
        file_paths: Vec::new(),
        resolved_spec: &resolved_spec,
        check_deadlock: false,
        show_coverage: false,
        strict_vacuity,
        continue_on_error: false,
        store_states: false,
        no_trace: false,
        fingerprint_storage: &no_storage,
        trace_file: None,
        trace_locs_storage: None,
        resolved_fairness: &fairness,
        max_states: case.max_states,
        max_depth: 0,
        memory_limit: 0,
        disk_limit: 0,
        output_format: OutputFormat::Json,
        progress_callback: no_progress,
        checkpoint_dir: &case.checkpoint_dir,
        checkpoint_interval: 0,
        resume_from: &case.resume_from,
        config_path: case.config_path,
        tool_out: &mut tool_out,
        collision_check_mode: Default::default(),
    })
}

#[test]
fn portfolio_json_includes_frontend_neutral_evidence_for_routing() {
    let module = parse_module(
        r#"
---- MODULE PortfolioJsonEvidence ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        ..Default::default()
    };
    let strategies = vec!["analytical".to_string()];
    let mut result = tla_check::PortfolioResult::run_with_frontend_source(
        &module,
        &[],
        &config,
        &strategies,
        true,
    );
    result.shared_engine_validation_receipts.push("Quint shared_engine_validation_receipt source_kind=quint payload_kind=quint receipt_role=analytical_solve model_check_search=false lane=bmc backend_code=ay_smt solver_family=ay validator_kind=ay_proof receipt_identity=shared_engine.validation_receipt:quint digest_algorithm=ay_fingerprint_identity digest=ay.proof.fingerprint:quint receipt_status=accepted receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready".to_string());
    let stats_snapshot = match &mut result.bfs_result {
        CheckResult::Success(stats) => {
            stats.backend_capability_report = Some(serde_json::json!({
                "evidence": [
                    "Quint trust_cg_native_jit_route source_kind=quint frontend_kind=quint"
                ],
                "fields": {
                    "source_kind": "quint",
                    "frontend_kind": "quint",
                    "prepared_admission_receipt": "prepared_frontier_admission_observed",
                    "prepared_admission_status": "accepted",
                    "fingerprint_evidence_label": "prepared_program_fingerprint_chain",
                    "prepared_program_fingerprint": "prepared-program-fp:quint",
                    "storage_layout_fingerprint": "storage-layout-fp:quint",
                    "artifact_fingerprint": "artifact-fp:quint",
                    "proof_or_witness_fingerprint": "proof-fp:quint",
                    "prepared_program_identity": "prepared-program-id:quint",
                    "frontend_payload_identity": "frontend-payload-id:quint",
                    "artifact_identity": "artifact-id:quint",
                    "storage_policy_identity": "storage-policy-id:quint",
                    "fingerprint_policy_identity": "fingerprint-policy-id:quint",
                    "fingerprint_identity": "fingerprint-id:quint",
                    "native_callable_receipt": "native_action_callout_batch_observed",
                    "native_callable_receipt_readiness": "ready",
                    "current_head_freshness": "current_head_evidence",
                    "evidence_json_freshness": "current_head_evidence",
                    "cold_warm_cache_label": "warm_cache_hit",
                    "cache_temperature": "warm",
                    "benchmark_gate_status": "not_claimed",
                    "faster_than_tlc_claim_supported": "false"
                }
            }));
            stats.clone()
        }
        other => panic!("expected analytical portfolio success, got {other:?}"),
    };

    let value = portfolio_success_json_value(
        &result,
        &stats_snapshot,
        std::time::Duration::from_millis(7),
        true,
    );
    let rendered = render_structured_json_value(OutputFormat::Json, &value)
        .expect("portfolio JSON should serialize");

    assert_eq!(value["mode"], "portfolio");
    assert_eq!(value["frontend_source"], "quint");
    assert_eq!(value["analytical_eligibility"], "verified_execution_model");
    assert!(value["analytical_solve_evidence"]
        .as_array()
        .expect("analytical evidence array")
        .iter()
        .any(|row| row.as_str().is_some_and(|row| {
            row.contains("Quint analytical_solve_decision")
                && row.contains("source_kind=quint")
                && row.contains("payload_kind=quint")
        })));
    assert!(value["shared_engine_validation_receipts"]
        .as_array()
        .expect("shared engine validation receipt array")
        .iter()
        .any(|row| row.as_str().is_some_and(|row| {
            row.contains("Quint shared_engine_validation_receipt")
                && row.contains("source_kind=quint")
                && row.contains("payload_kind=quint")
                && row.contains("receipt_role=analytical_solve")
                && row.contains("model_check_search=false")
                && row.contains("validator_kind=ay_proof")
                && row.contains("receipt_status=accepted")
                && row.contains("publication_readiness=ready")
        })));
    assert_eq!(
        value["backend_capability_report"]["fields"]["frontend_kind"],
        "quint"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["prepared_admission_receipt"],
        "prepared_frontier_admission_observed"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["prepared_admission_status"],
        "accepted"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["fingerprint_evidence_label"],
        "prepared_program_fingerprint_chain"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["fingerprint_chain_current_status"],
        "observed"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["analytical_solve_receipt"],
        "analytical_ay_solve_receipt_observed"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["analytical_solve_receipt_readiness"],
        "ready"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["ay_solve_receipt_readiness"],
        "ready"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["native_callable_receipt"],
        "native_action_callout_batch_observed"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["current_head_freshness"],
        "current_head_evidence"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["evidence_json_freshness"],
        "current_head_evidence"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["cold_warm_cache_label"],
        "warm_cache_hit"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["cache_temperature"],
        "warm"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["benchmark_gate_status"],
        "not_claimed"
    );
    assert_eq!(
        value["backend_capability_report"]["fields"]["faster_than_tlc_claim_supported"],
        "false"
    );
    assert!(value["backend_capability_report"]["evidence"]
        .as_array()
        .expect("backend evidence array")
        .iter()
        .any(|row| row.as_str().is_some_and(|row| {
            row.contains("cli_shared_engine_current_status")
                && row.contains("origin_frontend=quint")
                && row.contains("prepared_admission_receipt=prepared_frontier_admission_observed")
                && row.contains("faster_than_tlc_claim_supported=false")
        })));
    assert!(rendered.contains("Quint trust_cg_native_jit_route"));
    assert!(!rendered.contains("TY analytical_solve_decision"));

    #[cfg(feature = "ay")]
    {
        assert!(value["ay_shared_engine_evidence"]
            .as_array()
            .expect("ay shared engine evidence array")
            .iter()
            .any(|row| row.as_str().is_some_and(|row| {
                row.contains("Quint ay_shared_engine_lane_admission")
                    && row.contains("source_kind=quint")
                    && row.contains("payload_kind=quint")
            })));
    }
}

#[test]
fn shared_engine_report_fails_closed_when_receipts_are_missing() {
    let value = strict_shared_engine_report_json(None, false, &[]);
    let fields = value["fields"].as_object().expect("strict fields");

    assert_eq!(fields["origin_frontend"], "tla_plus");
    assert_eq!(
        fields["prepared_admission_receipt"],
        "blocked_missing_prepared_admission_receipt"
    );
    assert_eq!(
        fields["prepared_admission_status"],
        "blocked_missing_prepared_admission_receipt"
    );
    assert_eq!(
        fields["fingerprint_chain_current_status"],
        "blocked_missing_current_identity"
    );
    assert_eq!(
        fields["analytical_solve_receipt_readiness"],
        "blocked_missing_analytical_solve_receipt"
    );
    assert_eq!(
        fields["ay_solve_receipt_readiness"],
        "blocked_missing_ay_solve_receipt"
    );
    assert_eq!(
        fields["native_callable_receipt"],
        "blocked_missing_native_callable_receipt"
    );
    assert_eq!(fields["current_head_freshness"], "not_checked");
    assert_eq!(fields["evidence_json_freshness"], "not_checked");
    assert_eq!(fields["cold_warm_cache_label"], "not_timed");
    assert_eq!(fields["benchmark_gate_status"], "not_claimed");
    assert_eq!(fields["faster_than_tlc_claim_supported"], "false");

    let row = value["evidence"]
        .as_array()
        .expect("evidence rows")
        .iter()
        .find_map(serde_json::Value::as_str)
        .expect("strict evidence row");
    assert!(row.contains("cli_shared_engine_current_status"));
    assert!(row.contains("origin_frontend=tla_plus"));
    assert!(row.contains("prepared_admission_receipt=blocked_missing_prepared_admission_receipt"));
    assert!(row.contains("native_callable_receipt=blocked_missing_native_callable_receipt"));
    assert!(row.contains("faster_than_tlc_claim_supported=false"));
    assert!(!row.contains("analytical_solve_receipt_readiness=ready"));
}

#[test]
fn analytical_receipt_readiness_rejects_contradictory_receipts() {
    let receipts = vec![
        "TY shared_engine_validation_receipt receipt_status=accepted receipt_status=rejected receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready"
            .to_string(),
        "TY shared_engine_validation_receipt receipt_status=accepted receipt_validation=valid failure_reason=solver_failed publication_blocker=none publication_readiness=ready"
            .to_string(),
        "TY analytical_solve_decision receipt_role=analytical_solve receipt_status=accepted receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready"
            .to_string(),
        "TY shared_engine_validation_receipt receipt_role=analytical_solve lane=bmc backend_code=ay_smt solver_family=ay validator_kind=structural_proof receipt_identity=shared_engine.validation_receipt:bad digest_algorithm=ay_fingerprint_identity digest=ay.proof.fingerprint:bad receipt_status=accepted receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready"
            .to_string(),
        "TY shared_engine_validation_receipt receipt_role=analytical_solve lane=bmc backend_code=ay_smt solver_family=ay validator_kind=ay_proof receipt_identity=shared_engine.validation_receipt:bad digest_algorithm=fnv1a64 digest=ay.proof.fingerprint:bad receipt_status=accepted receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready"
            .to_string(),
        "producer=malformed shared_engine_validation_receipt receipt_role=analytical_solve receipt_status=accepted receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready"
            .to_string(),
        "TY shared_engine_validation_receipt receipt_role=analytical_solve receipt_status=accepted malformed_token receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready"
            .to_string(),
    ];
    let value = strict_shared_engine_report_json(None, false, &receipts);
    let fields = value["fields"].as_object().expect("strict fields");

    assert_eq!(
        fields["analytical_solve_receipt_readiness"],
        "blocked_missing_analytical_solve_receipt"
    );
    assert_eq!(
        fields["ay_solve_receipt_readiness"],
        "blocked_missing_ay_solve_receipt"
    );
    assert_ne!(fields["analytical_solve_receipt_readiness"], "ready");
}

#[test]
fn analytical_receipt_readiness_requires_strict_validation_receipt_shape() {
    let accepted = vec![AY_ACCEPTED_ANALYTICAL_RECEIPT_ROW.to_string()];
    let value = strict_shared_engine_report_json(None, false, &accepted);
    assert_eq!(
        value["fields"]["analytical_solve_receipt_readiness"],
        "ready"
    );
    assert_eq!(value["fields"]["ay_solve_receipt_readiness"], "ready");

    let wrong_role = vec![
        "TY shared_engine_validation_receipt receipt_role=model_check_search lane=bmc backend_code=ay_smt solver_family=ay validator_kind=ay_proof receipt_identity=shared_engine.validation_receipt:bad digest_algorithm=ay_fingerprint_identity digest=ay.proof.fingerprint:bad receipt_status=accepted receipt_validation=valid failure_reason=none publication_blocker=none publication_readiness=ready"
            .to_string(),
    ];
    let value = strict_shared_engine_report_json(None, false, &wrong_role);
    assert_eq!(
        value["fields"]["analytical_solve_receipt_readiness"],
        "blocked_missing_analytical_solve_receipt"
    );
}

#[test]
fn faster_than_tlc_claim_requires_passing_benchmark_gate() {
    let blocked = strict_shared_engine_report_json(
        Some(serde_json::json!({
            "fields": {
                "faster_than_tlc_claim_supported": "true",
                "benchmark_gate_status": "stale"
            }
        })),
        false,
        &[],
    );
    assert_eq!(
        blocked["fields"]["faster_than_tlc_claim_supported"],
        "false"
    );
    assert_eq!(
        blocked["fields"]["faster_than_tlc_claim_blocker"],
        "blocked_benchmark_gate_not_passed"
    );

    let passed = strict_shared_engine_report_json(
        Some(serde_json::json!({
            "fields": {
                "faster_than_tlc_claim_supported": "true",
                "benchmark_gate_status": "enforced",
                "current_head_freshness": "current_head_evidence",
                "tlc_wall_seconds": "2.0",
                "trust_cg_cold_wall_seconds": "1.0"
            }
        })),
        false,
        &[],
    );
    assert_eq!(passed["fields"]["faster_than_tlc_claim_supported"], "true");
    assert!(passed["fields"]
        .as_object()
        .expect("strict fields")
        .get("faster_than_tlc_claim_blocker")
        .is_none());

    let missing_current_head = strict_shared_engine_report_json(
        Some(serde_json::json!({
            "fields": {
                "faster_than_tlc_claim_supported": "true",
                "benchmark_gate_status": "enforced",
                "tlc_wall_seconds": "2.0",
                "trust_cg_cold_wall_seconds": "1.0"
            }
        })),
        false,
        &[],
    );
    assert_eq!(
        missing_current_head["fields"]["faster_than_tlc_claim_supported"],
        "false"
    );
    assert_eq!(
        missing_current_head["fields"]["faster_than_tlc_claim_blocker"],
        "blocked_current_head_evidence_missing"
    );

    let missing_cold_wall = strict_shared_engine_report_json(
        Some(serde_json::json!({
            "fields": {
                "faster_than_tlc_claim_supported": "true",
                "benchmark_gate_status": "enforced",
                "current_head_freshness": "current_head_evidence"
            }
        })),
        false,
        &[],
    );
    assert_eq!(
        missing_cold_wall["fields"]["faster_than_tlc_claim_supported"],
        "false"
    );
    assert_eq!(
        missing_cold_wall["fields"]["faster_than_tlc_claim_blocker"],
        "blocked_cold_wall_win_missing"
    );
}

#[test]
fn resume_only_mode_still_validates_checkpoint_metadata_paths() {
    let module = parse_module(SPEC_SRC);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["TypeOK".to_string()],
        ..Default::default()
    };

    let base_dir = make_temp_base_dir();
    let (spec_a, cfg_a) = write_spec_and_cfg(&base_dir, "SpecA");
    let (spec_b, cfg_b) = write_spec_and_cfg(&base_dir, "SpecB");
    let checkpoint_dir = base_dir.join("checkpoint");

    let save_result = run_once(
        &module,
        &config,
        RunCase {
            file: &spec_a,
            config_path: &cfg_a,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            resume_from: None,
            max_states: 1,
        },
    )
    .expect("checkpoint save run should succeed");
    assert!(
        matches!(
            save_result.0,
            CheckResult::LimitReached {
                limit_type: tla_check::LimitType::States,
                ..
            }
        ),
        "expected checkpoint-producing limit run, got {:?}",
        save_result.0
    );

    let err = run_once(
        &module,
        &config,
        RunCase {
            file: &spec_b,
            config_path: &cfg_b,
            checkpoint_dir: None,
            resume_from: Some(checkpoint_dir),
            max_states: 0,
        },
    )
    .expect_err("resume-only run must reject mismatched checkpoint metadata paths");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("checkpoint spec path mismatch"),
        "expected spec-path mismatch in error chain, got: {msg}"
    );

    let _ = fs::remove_dir_all(base_dir);
}

#[test]
fn workers_one_fp_only_path_runs_liveness_without_store_states() {
    let module = parse_module(
        r#"
---- MODULE FpOnlyLiveness ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == UNCHANGED x
Progress == <>(x = 1)
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Progress".to_string()],
        check_deadlock: false,
        ..Default::default()
    };

    let base_dir = make_temp_base_dir();
    let spec_path = base_dir.join("FpOnlyLiveness.tla");
    let cfg_path = base_dir.join("FpOnlyLiveness.cfg");
    fs::write(
        &spec_path,
        r#"
---- MODULE FpOnlyLiveness ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == UNCHANGED x
Progress == <>(x = 1)
====
"#,
    )
    .expect("write spec");
    fs::write(
        &cfg_path,
        "INIT Init\nNEXT Next\nPROPERTY Progress\nCHECK_DEADLOCK FALSE\n",
    )
    .expect("write cfg");

    let result = run_once(
        &module,
        &config,
        RunCase {
            file: &spec_path,
            config_path: &cfg_path,
            checkpoint_dir: None,
            resume_from: None,
            max_states: 0,
        },
    )
    .expect("workers=1 fp-only liveness run should complete");

    assert!(
        matches!(result.0, CheckResult::LivenessViolation { .. }),
        "expected workers=1 fp-only path to report LivenessViolation, got {:?}",
        result.0
    );

    let _ = fs::remove_dir_all(base_dir);
}

#[test]
fn human_output_wires_progress_callback_without_cli_flag() {
    let base_dir = make_temp_base_dir();
    let (module, config, spec_path, cfg_path) = make_progress_case(&base_dir);
    let progress_hits = Arc::new(AtomicUsize::new(0));
    let progress_hits_clone = Arc::clone(&progress_hits);
    let result = run_once_with_output(
        &module,
        &config,
        RunCase {
            file: &spec_path,
            config_path: &cfg_path,
            checkpoint_dir: None,
            resume_from: None,
            max_states: 0,
        },
        OutputFormat::Human,
        Box::new(move |_| {
            progress_hits_clone.fetch_add(1, Ordering::Relaxed);
        }),
    )
    .expect("human-output run should succeed");

    assert!(
        matches!(result.0, CheckResult::Success(_)),
        "expected successful counter exploration, got {:?}",
        result.0
    );
    assert!(
        progress_hits.load(Ordering::Relaxed) > 0,
        "expected human output to wire progress callback without --progress"
    );

    let _ = fs::remove_dir_all(base_dir);
}

/// Part of #3706: POR is now accepted in auto mode (routed to sequential or parallel).
#[test]
fn auto_mode_accepts_por_flag() {
    let module = parse_module(SPEC_SRC);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["TypeOK".to_string()],
        por_enabled: true,
        ..Default::default()
    };

    let base_dir = make_temp_base_dir();
    let (spec_path, cfg_path) = write_spec_and_cfg(&base_dir, "AutoPor");

    let (result, _strategy) = run_once_with_workers(
        &module,
        &config,
        RunCase {
            file: &spec_path,
            config_path: &cfg_path,
            checkpoint_dir: None,
            resume_from: None,
            max_states: 0,
        },
        0,
    )
    .expect("auto POR run should reach the checker");

    assert!(
        matches!(result, CheckResult::Success(_)),
        "auto mode with --por should succeed, got: {result:?}",
    );

    let _ = fs::remove_dir_all(base_dir);
}

#[test]
fn strict_vacuity_auto_mode_uses_exhaustive_sequential_action_evidence() {
    let spec = r#"
---- MODULE StrictVacuityAuto ----
VARIABLE x
Init == x = 0
A == /\ x = 0 /\ x' = 1
B == /\ x = 1 /\ x' = 2
C == /\ x = 99 /\ x' = 100
Next == A \/ B \/ C
====
"#;
    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        check_deadlock: false,
        ..Default::default()
    };

    let base_dir = make_temp_base_dir();
    let spec_path = base_dir.join("StrictVacuityAuto.tla");
    let cfg_path = base_dir.join("StrictVacuityAuto.cfg");
    fs::write(&spec_path, spec).expect("write strict-vacuity spec");
    fs::write(
        &cfg_path,
        "INIT Init\nNEXT Next\nCHECK_DEADLOCK FALSE\n",
    )
    .expect("write strict-vacuity cfg");

    let (result, strategy) = run_once_with_workers_and_strict(
        &module,
        &config,
        RunCase {
            file: &spec_path,
            config_path: &cfg_path,
            checkpoint_dir: None,
            resume_from: None,
            max_states: 0,
        },
        0,
        true,
    )
    .expect("strict-vacuity auto run should use sequential BFS");

    let stats = match result {
        CheckResult::Success(stats) => stats,
        other => panic!("strict-vacuity evidence run should succeed, got {other:?}"),
    };
    assert_eq!(stats.states_found, 3);
    assert!(strategy
        .as_deref()
        .is_some_and(|line| line.contains("strict-vacuity exhaustive action evidence")));
    let dead_actions = stats.vacuity_warnings.iter().find_map(|warning| match warning {
        tla_check::VacuityWarning::DeadActions(names) => Some(names.as_slice()),
        _ => None,
    });
    assert_eq!(dead_actions, Some(["C".to_string()].as_slice()));

    let _ = fs::remove_dir_all(base_dir);
}

#[test]
fn strict_vacuity_rejects_parallel_workers() {
    let module = parse_module(SPEC_SRC);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let base_dir = make_temp_base_dir();
    let (spec_path, cfg_path) = write_spec_and_cfg(&base_dir, "StrictVacuityParallel");

    let err = run_once_with_workers_and_strict(
        &module,
        &config,
        RunCase {
            file: &spec_path,
            config_path: &cfg_path,
            checkpoint_dir: None,
            resume_from: None,
            max_states: 0,
        },
        2,
        true,
    )
    .expect_err("strict-vacuity must reject parallel workers");
    assert!(err.to_string().contains("exhaustive sequential BFS"));

    let _ = fs::remove_dir_all(base_dir);
}

/// Part of #3706: POR is now accepted in parallel mode.
#[test]
fn parallel_mode_accepts_por_flag() {
    let module = parse_module(SPEC_SRC);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["TypeOK".to_string()],
        por_enabled: true,
        ..Default::default()
    };

    let base_dir = make_temp_base_dir();
    let (spec_path, cfg_path) = write_spec_and_cfg(&base_dir, "ParallelPor");

    let (result, _strategy) = run_once_with_workers(
        &module,
        &config,
        RunCase {
            file: &spec_path,
            config_path: &cfg_path,
            checkpoint_dir: None,
            resume_from: None,
            max_states: 0,
        },
        2,
    )
    .expect("parallel POR run should reach the checker");

    assert!(
        matches!(result, CheckResult::Success(_)),
        "parallel mode with --por should succeed, got: {result:?}",
    );

    let _ = fs::remove_dir_all(base_dir);
}

#[test]
fn parallel_mode_runs_on_the_fly_liveness() {
    let module = parse_module(
        r#"
---- MODULE ParallelOnTheFly ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == UNCHANGED x
Progress == <>(x = 1)
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Progress".to_string()],
        liveness_execution: tla_check::LivenessExecutionMode::OnTheFly,
        ..Default::default()
    };

    let base_dir = make_temp_base_dir();
    let spec_path = base_dir.join("ParallelOnTheFly.tla");
    let cfg_path = base_dir.join("ParallelOnTheFly.cfg");
    fs::write(
        &spec_path,
        r#"
---- MODULE ParallelOnTheFly ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == UNCHANGED x
Progress == <>(x = 1)
====
"#,
    )
    .expect("write spec");
    fs::write(
        &cfg_path,
        "INIT Init\nNEXT Next\nPROPERTY Progress\nCHECK_DEADLOCK FALSE\n",
    )
    .expect("write cfg");

    let (result, _strategy) = run_once_with_workers(
        &module,
        &config,
        RunCase {
            file: &spec_path,
            config_path: &cfg_path,
            checkpoint_dir: None,
            resume_from: None,
            max_states: 0,
        },
        2,
    )
    .expect("parallel on-the-fly liveness should run");
    match result {
        CheckResult::LivenessViolation { property, .. } => {
            assert_eq!(property, "Progress");
        }
        other => panic!("expected parallel on-the-fly liveness violation, got: {other:?}"),
    }

    let _ = fs::remove_dir_all(base_dir);
}
