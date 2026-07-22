// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::cmd_diagnose::types::TimeoutSource;
use sha2::{Digest, Sha256};

fn make_result(
    name: &str,
    verdict: SpecVerdict,
    ty_states: Option<u64>,
    error_details: Option<&str>,
) -> SpecResult {
    SpecResult {
        name: name.to_string(),
        verdict,
        ty_status: None,
        ty_states,
        tlc_status: "pass".to_string(),
        tlc_states: None,
        tlc_error_type: None,
        error_details: error_details.map(str::to_string),
        expected_mismatch_reason: None,
        time_seconds: 0.0,
        timeout_seconds: 120,
        timeout_source: TimeoutSource::Cli,
    }
}

fn make_timed_result(
    name: &str,
    verdict: SpecVerdict,
    ty_states: Option<u64>,
    error_details: Option<&str>,
    time_seconds: f64,
) -> SpecResult {
    let mut result = make_result(name, verdict, ty_states, error_details);
    result.time_seconds = time_seconds;
    result
}

fn sample_baseline() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 3,
        "specs_jcs_sha256": "old-digest",
        "stats": {
            "ty_fail": 99,
            "ty_match": 99,
            "ty_mismatch": 99,
            "ty_untested": 99,
            "tlc_error": 99,
            "tlc_pass": 99,
            "tlc_timeout": 99
        },
        "ty_refresh": {
            "git_commit": "old",
            "script": "old",
            "specs_ran": 0,
            "specs_updated": 0,
            "timestamp": "old"
        },
        "specs": {
            "PassSpec": spec_entry("pass", 3, None),
            "MismatchSpec": {
                "tlc": {
                    "status": "pass",
                    "states": 5,
                    "error_type": null
                },
                "ty": {
                    "status": "untested",
                    "states": null,
                    "error_type": null,
                    "last_run": null,
                    "git_commit": null
                },
                "verified_match": false,
                "category": "small",
                "expected_mismatch": true,
                "expected_mismatch_reason": "known non-comparable baseline"
            },
            "FailSpec": spec_entry("error", 0, Some("invariant")),
            "UntouchedSpec": spec_entry("timeout", 0, Some("timeout"))
        }
    })
}

fn spec_entry(status: &str, states: u64, error_type: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "tlc": {
            "status": status,
            "states": if status == "timeout" { serde_json::Value::Null } else { serde_json::json!(states) },
            "error_type": error_type
        },
        "ty": {
            "status": "untested",
            "states": null,
            "error_type": null,
            "last_run": null,
            "git_commit": null
        },
        "verified_match": false,
        "category": "small"
    })
}

fn refreshed_sample() -> serde_json::Value {
    refresh_baseline_value(
        sample_baseline(),
        &[
            make_result("PassSpec", SpecVerdict::Pass, Some(3), None),
            make_result(
                "MismatchSpec",
                SpecVerdict::ExpectedMismatch,
                Some(4),
                Some("state mismatch"),
            ),
            make_result("FailSpec", SpecVerdict::Crash, None, Some("runtime error")),
            make_result("UntouchedSpec", SpecVerdict::Skip, None, None),
        ],
    )
    .expect("baseline refresh should succeed")
}

#[test]
fn test_refresh_baseline_value_records_pass_runtime() {
    let refreshed = refresh_baseline_value(
        sample_baseline(),
        &[make_timed_result(
            "PassSpec",
            SpecVerdict::Pass,
            Some(3),
            None,
            1.234,
        )],
    )
    .expect("baseline refresh should record runtime");

    assert_eq!(
        refreshed["specs"]["PassSpec"]["ty"]["runtime_seconds"],
        serde_json::json!(1.234)
    );
}

#[test]
fn test_refresh_baseline_value_removes_stale_non_pass_runtime() {
    let mut raw = sample_baseline();
    raw["specs"]["FailSpec"]["ty"]["runtime_seconds"] = serde_json::json!(99.0);

    let refreshed = refresh_baseline_value(
        raw,
        &[make_timed_result(
            "FailSpec",
            SpecVerdict::Crash,
            None,
            Some("runtime error"),
            2.0,
        )],
    )
    .expect("baseline refresh should drop stale runtime");

    let ty = refreshed["specs"]["FailSpec"]["ty"]
        .as_object()
        .expect("ty entry should stay an object");
    assert!(
        !ty.contains_key("runtime_seconds"),
        "non-pass refresh must not leave stale speed evidence"
    );
}

#[test]
fn test_refresh_baseline_value_updates_spec_entries() {
    let refreshed = refreshed_sample();

    assert_eq!(refreshed["specs"]["PassSpec"]["ty"]["status"], "pass");
    assert_eq!(refreshed["specs"]["PassSpec"]["verified_match"], true);
    assert_eq!(
        refreshed["specs"]["MismatchSpec"]["ty"]["status"],
        "mismatch"
    );
    assert_eq!(refreshed["specs"]["MismatchSpec"]["verified_match"], false);
    assert_eq!(
        refreshed["specs"]["MismatchSpec"]["expected_mismatch_reason"],
        "known non-comparable baseline"
    );
    assert_eq!(refreshed["specs"]["FailSpec"]["ty"]["status"], "fail");
    assert_eq!(
        refreshed["specs"]["FailSpec"]["ty"]["error_type"],
        "runtime error"
    );
    assert_eq!(
        refreshed["specs"]["UntouchedSpec"]["ty"]["status"],
        "untested"
    );
}

#[test]
fn test_refresh_baseline_value_updates_summary_metadata() {
    let refreshed = refreshed_sample();

    assert_eq!(refreshed["stats"]["ty_match"], serde_json::json!(1));
    assert_eq!(refreshed["stats"]["ty_mismatch"], serde_json::json!(1));
    assert_eq!(refreshed["stats"]["ty_fail"], serde_json::json!(1));
    assert_eq!(refreshed["stats"]["ty_untested"], serde_json::json!(1));
    assert_eq!(refreshed["stats"]["tlc_pass"], serde_json::json!(2));
    assert_eq!(refreshed["stats"]["tlc_error"], serde_json::json!(1));
    assert_eq!(refreshed["stats"]["tlc_timeout"], serde_json::json!(1));

    assert_eq!(
        refreshed["ty_refresh"]["script"],
        serde_json::Value::String("ty diagnose --update-baseline".to_string())
    );
    assert_eq!(refreshed["ty_refresh"]["specs_ran"], serde_json::json!(4));
    assert_eq!(
        refreshed["ty_refresh"]["specs_updated"],
        serde_json::json!(3)
    );
    assert_ne!(refreshed["ty_refresh"]["timestamp"], "old");
}

#[test]
fn test_refresh_baseline_value_recomputes_specs_digest() {
    let refreshed = refreshed_sample();
    let expected_digest = sha256_jcs_value(&refreshed["specs"]).expect("digest should compute");

    assert_eq!(
        refreshed["specs_jcs_sha256"],
        serde_json::Value::String(expected_digest)
    );
}

#[test]
fn test_refresh_baseline_value_preserves_first_error_override() {
    let mut raw = sample_baseline();
    raw.as_object_mut()
        .expect("baseline root should be object")
        .get_mut("specs")
        .and_then(serde_json::Value::as_object_mut)
        .expect("baseline specs should be object")
        .insert(
            "FirstErrorSpec".to_string(),
            serde_json::json!({
                "tlc": {
                    "status": "error",
                    "states": 11285,
                    "error_type": "invariant"
                },
                "ty": {
                    "status": "untested",
                    "states": null,
                    "error_type": null,
                    "last_run": null,
                    "git_commit": null
                },
                "verified_match": false,
                "category": "small",
                "continue_on_error": false
            }),
        );

    let refreshed = refresh_baseline_value(
        raw,
        &[make_result(
            "FirstErrorSpec",
            SpecVerdict::Crash,
            None,
            Some("runtime error"),
        )],
    )
    .expect("baseline refresh should preserve per-spec execution mode");

    let spec = refreshed["specs"]["FirstErrorSpec"]
        .as_object()
        .expect("refreshed spec should remain an object");
    assert_eq!(
        spec.get("continue_on_error"),
        Some(&serde_json::Value::Bool(false))
    );
}

#[test]
fn test_refresh_baseline_value_keeps_default_continue_mode_implicit() {
    let refreshed = refreshed_sample();

    let spec = refreshed["specs"]["PassSpec"]
        .as_object()
        .expect("refreshed spec should remain an object");
    assert!(
        spec.get("continue_on_error").is_none(),
        "baseline refresh should not churn default continue-on-error metadata"
    );
}

#[test]
fn test_sha256_jcs_value_normalizes_float_lexemes() {
    let value: serde_json::Value = serde_json::from_str(
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
