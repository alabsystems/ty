// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::path::Path;

use super::*;

fn sample_expected_mismatch_result() -> SpecResult {
    SpecResult {
        name: "ExampleExpectedMismatch".to_string(),
        verdict: SpecVerdict::ExpectedMismatch,
        ty_status: Some("ok".to_string()),
        ty_states: Some(0),
        tlc_status: "error".to_string(),
        tlc_states: None,
        tlc_error_type: Some("unknown".to_string()),
        error_details: None,
        expected_mismatch_reason: Some(
            "TLC smoke harness requires IOEnvExec/CSVWrite side effects.".to_string(),
        ),
        time_seconds: 1.0,
        timeout_seconds: 120,
        timeout_source: TimeoutSource::Cli,
    }
}

fn sample_pass_result() -> SpecResult {
    SpecResult {
        name: "SmokeEWD998_SC".to_string(),
        verdict: SpecVerdict::Pass,
        ty_status: Some("ok".to_string()),
        ty_states: Some(0),
        tlc_status: "pass".to_string(),
        tlc_states: Some(0),
        tlc_error_type: None,
        error_details: None,
        expected_mismatch_reason: None,
        time_seconds: 1.0,
        timeout_seconds: 120,
        timeout_source: TimeoutSource::Cli,
    }
}

fn sample_timeout_result() -> SpecResult {
    SpecResult {
        name: "CarTalkPuzzle_M1".to_string(),
        verdict: SpecVerdict::Timeout,
        ty_status: Some("timeout".to_string()),
        ty_states: None,
        tlc_status: "pass".to_string(),
        tlc_states: Some(0),
        tlc_error_type: None,
        error_details: Some("timeout after 1s".to_string()),
        expected_mismatch_reason: None,
        time_seconds: 1.0,
        timeout_seconds: 1,
        timeout_source: TimeoutSource::Cli,
    }
}

#[test]
fn test_build_json_report_includes_expected_mismatch_fields() {
    let results = vec![sample_expected_mismatch_result()];
    let tally = Tally::from_results(&results, 10);
    let report = build_json_report(
        &results,
        &tally,
        10,
        1,
        Path::new("/tmp/ty"),
        RunConditions {
            cpu_count: 1,
            load_avg_1m: 0.0,
            load_avg_5m: 0.0,
            load_avg_15m: 0.0,
            timeout_floor_seconds: 120,
            timeout_seconds: 120,
            retries: 0,
            checker_policy: "baseline_parity",
            checker_workers: None,
        },
    );

    assert_eq!(report.schema_version, 7);
    assert_eq!(report.run_conditions.timeout_floor_seconds, 120);
    assert_eq!(report.run_conditions.timeout_seconds, 120);
    assert_eq!(report.summary.expected_mismatch, 1);
    assert_eq!(report.run_conditions.checker_policy, "baseline_parity");
    assert_eq!(report.run_conditions.checker_workers, None);
    assert_eq!(
        report.specs["ExampleExpectedMismatch"].status,
        "expected_mismatch"
    );
    assert_eq!(
        report.specs["ExampleExpectedMismatch"]
            .expected_mismatch_reason
            .as_deref(),
        Some("TLC smoke harness requires IOEnvExec/CSVWrite side effects.")
    );
}

#[test]
fn test_write_metrics_file_records_partial_timeout_result() {
    let results = vec![sample_timeout_result()];
    let tally = Tally::from_results(&results, 99);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("diagnose-progress.json");

    write_metrics_file(
        &results,
        &tally,
        99,
        results.len(),
        Path::new("/tmp/ty"),
        RunConditions {
            cpu_count: 1,
            load_avg_1m: 0.0,
            load_avg_5m: 0.0,
            load_avg_15m: 0.0,
            timeout_floor_seconds: 1,
            timeout_seconds: 1,
            retries: 0,
            checker_policy: "baseline_parity",
            checker_workers: None,
        },
        &path,
    )
    .expect("write metrics");

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read metrics"))
            .expect("parse metrics");
    assert_eq!(json["summary"]["specs_ran"], 1);
    assert_eq!(json["summary"]["timeout"], 1);
    assert_eq!(json["specs"]["CarTalkPuzzle_M1"]["status"], "timeout");
    assert_eq!(
        json["specs"]["CarTalkPuzzle_M1"]["ty_error"],
        "timeout after 1s"
    );
}

#[test]
fn test_build_json_report_records_explicit_checker_worker_override() {
    let results = vec![sample_pass_result()];
    let tally = Tally::from_results(&results, 1);
    let report = build_json_report(
        &results,
        &tally,
        1,
        1,
        Path::new("/tmp/ty"),
        RunConditions {
            cpu_count: 8,
            load_avg_1m: 1.0,
            load_avg_5m: 0.5,
            load_avg_15m: 0.25,
            timeout_floor_seconds: 300,
            timeout_seconds: 300,
            retries: 2,
            checker_policy: "checker_workers",
            checker_workers: Some(4),
        },
    );

    assert_eq!(report.run_conditions.checker_policy, "checker_workers");
    assert_eq!(report.run_conditions.checker_workers, Some(4));
    assert_eq!(report.specs["SmokeEWD998_SC"].status, "pass");
    assert!(
        report.specs["SmokeEWD998_SC"]
            .expected_mismatch_reason
            .is_none(),
        "passing results should not carry an expected mismatch reason"
    );
}
