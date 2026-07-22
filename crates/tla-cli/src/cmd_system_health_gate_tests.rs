// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::path::{Path, PathBuf};

use chrono::TimeZone;

use super::cargo_timings::{
    check_parse_cargo_timings_ndjson_self_test, parse_cargo_timings_ndjson_text,
};

use super::{
    check_current_doc_routing, check_doc_text_guards, check_guard_content, check_level,
    check_parse_tlc_dot_smoke, check_spec_coverage_freshness_with_now, combine_output,
    dumps_canonical, extract_tlc_quoted_attr, format_check_line, manifest, parse_tlc_dot_text,
    sha256_jcs, tlc_dot_unescape, DocTextGuard, HealthCheck,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn check_line_format_matches_legacy_prefix_contract() {
    let ok = HealthCheck::ok("exists:/repo/Cargo.toml", None);
    let warn = HealthCheck::warn("spec_coverage_freshness", "missing; run: diagnose");
    let err = HealthCheck::err("baseline_drift", Some("examples changed".to_string()));

    assert_eq!(format_check_line(&ok), "OK  exists:/repo/Cargo.toml");
    assert_eq!(
        format_check_line(&warn),
        "WARN spec_coverage_freshness (missing; run: diagnose)"
    );
    assert_eq!(
        format_check_line(&err),
        "ERR baseline_drift (examples changed)"
    );
}

#[test]
fn combine_output_preserves_python_runner_stderr_label() {
    assert_eq!(combine_output("out", ""), "out");
    assert_eq!(combine_output("", "err"), "STDERR:\nerr");
    assert_eq!(combine_output("out", "err"), "out\nSTDERR:\nerr");
}

#[test]
fn canonical_json_digest_is_order_independent() {
    let left = serde_json::json!({"b": [2, 3], "a": {"z": true, "m": null}});
    let right = serde_json::json!({"a": {"m": null, "z": true}, "b": [2, 3]});

    assert_eq!(dumps_canonical(&left), dumps_canonical(&right));
    assert_eq!(sha256_jcs(&left), sha256_jcs(&right));
}

#[tokio::test]
async fn manifest_summary_counts_error_warning_and_ok_levels() {
    let checks = vec![
        HealthCheck::ok("ok", None),
        HealthCheck::warn("warn", "heads up"),
        HealthCheck::err("err", Some("broken".to_string())),
    ];
    let manifest = manifest(&repo_root(), &checks).await;

    assert_eq!(manifest.summary.status, "fail");
    assert_eq!(manifest.summary.passed, 1);
    assert_eq!(manifest.summary.warnings, 1);
    assert_eq!(manifest.summary.errors, 1);
    assert_eq!(check_level(&checks[1]), "warn");
}

#[test]
fn shell_quality_gate_uses_rust_system_health_command() {
    let shell = std::fs::read_to_string(repo_root().join("scripts/check_code_quality_gate.sh"))
        .expect("read code quality gate");

    assert!(shell.contains("system-health-gate \"$@\""));
    assert!(shell.contains("run_system_health_gate --mode warn"));
    assert!(!shell.contains("python3 scripts/system_health_check.py"));
}

#[test]
fn current_doc_routing_guard_matches_python_text_rules() {
    let guard = DocTextGuard {
        path: "doc.md",
        required_substrings: &["alpha beta"],
        forbidden_patterns: &[r"^\| stale \| row \|$"],
    };

    assert!(check_guard_content("alpha\n\nbeta\n| fresh | row |\n", &guard).is_empty());

    let failures = check_guard_content("alpha beta\n| stale | row |\n", &guard);
    assert_eq!(
        failures,
        vec!["matched forbidden pattern: ^\\| stale \\| row \\|$"]
    );
}

#[test]
fn current_doc_routing_reports_missing_files_and_content_failures() {
    let temp = tempfile::tempdir().expect("tempdir");
    let existing = temp.path().join("reports/research");
    std::fs::create_dir_all(&existing).expect("reports dir");
    std::fs::write(existing.join("current.md"), "old current guidance").expect("write doc");
    let guards = [
        DocTextGuard {
            path: "reports/research/current.md",
            required_substrings: &["new current guidance"],
            forbidden_patterns: &[],
        },
        DocTextGuard {
            path: "reports/research/missing.md",
            required_substrings: &[],
            forbidden_patterns: &[],
        },
    ];

    let failures = check_doc_text_guards(temp.path(), &guards);

    assert_eq!(
        failures,
        vec![
            "reports/research/current.md: missing required text: 'new current guidance'",
            "reports/research/missing.md: file not found",
        ]
    );
}

#[test]
fn current_doc_routing_success_uses_legacy_check_name() {
    let temp = tempfile::tempdir().expect("tempdir");
    for guard in super::CURRENT_DOC_ROUTING_GUARDS {
        let path = temp.path().join(guard.path);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("parent dir");
        std::fs::write(&path, guard.required_substrings.join("\n")).expect("write doc");
    }

    let check = check_current_doc_routing(temp.path());

    assert!(check.ok);
    assert_eq!(check.name, "cmd:check_current_doc_routing.py");
    assert_eq!(check.detail.as_deref(), Some("routing guards passed"));
}

#[test]
fn cargo_timings_parser_matches_python_self_test_fixtures() {
    let empty = parse_cargo_timings_ndjson_text("");

    assert!(empty.timings.is_empty());
    assert_eq!(empty.skipped_non_json_lines, 0);

    let mixed = parse_cargo_timings_ndjson_text(
        r#"
Compiling foo v0.1.0
{"reason":"compiler-artifact","duration":1.0}
{"reason":"timing-info","package_id":"p#0.1.0","duration":0.5}
{not json}
{"reason":"timing-info","package_id":"q#0.1.0","duration":"1.25"}
"#,
    );

    assert_eq!(
        mixed.timings,
        vec![("q#0.1.0".to_string(), 1.25), ("p#0.1.0".to_string(), 0.5),]
    );
    assert_eq!(mixed.skipped_non_json_lines, 2);
}

#[test]
fn cargo_timings_self_test_success_uses_legacy_check_name() {
    let check = check_parse_cargo_timings_ndjson_self_test();

    assert!(check.ok);
    assert_eq!(check.name, "cmd:parse_cargo_timings_ndjson.py --self-test");
    assert_eq!(check.detail.as_deref(), Some("self-test: ok"));
}

#[test]
fn tlc_dot_unescape_matches_python_escape_rules() {
    assert_eq!(
        tlc_dot_unescape("left\\nright\\\\slash\\\"quote\\q"),
        "left\nright\\slash\"quote\\q"
    );
    assert_eq!(tlc_dot_unescape("trail\\"), "trail\\");
}

#[test]
fn tlc_dot_extract_quoted_attrs_handles_escapes_and_commas() {
    let attrs =
        r#"tooltip="ignored", label="/\\ name = \"small\"\n/\\ path = C:\\tmp",style = filled"#;

    assert_eq!(
        extract_tlc_quoted_attr(attrs, "label").expect("label attr"),
        "/\\ name = \"small\"\n/\\ path = C:\\tmp"
    );
    assert_eq!(
        extract_tlc_quoted_attr(attrs, "tooltip").expect("tooltip attr"),
        "ignored"
    );
}

#[test]
fn tlc_dot_diehard_fixture_counts_and_depths_match_smoke_contract() {
    let fixture = std::fs::read_to_string(repo_root().join("test_data/tlc_dot/DieHard.dot"))
        .expect("read DieHard DOT fixture");
    let graph = parse_tlc_dot_text(&fixture).expect("parse DieHard DOT");
    let initial_fp = 1_317_622_219_392_791_164_i64;
    let initial_state = graph.states.get(&initial_fp).expect("initial state");
    let depth_sizes = graph
        .depth_groups
        .iter()
        .map(|(depth, fps)| (*depth, fps.len()))
        .collect::<Vec<_>>();

    assert_eq!(graph.states.len(), 14);
    assert_eq!(graph.transitions.len(), 72);
    assert_eq!(
        graph.initial_states,
        std::collections::BTreeSet::from([initial_fp])
    );
    assert_eq!(initial_state.fingerprint, initial_fp);
    assert_eq!(initial_state.label, "/\\ big = 0\n/\\ small = 0");
    assert!(initial_state.is_initial);
    assert_eq!(initial_state.depth, Some(0));
    assert_eq!(
        depth_sizes,
        vec![(0, 1), (1, 2), (2, 3), (3, 2), (4, 2), (5, 2), (6, 2)]
    );
    assert_eq!(
        graph
            .states
            .get(&7_056_248_354_844_844_581_i64)
            .expect("depth 6 state")
            .depth,
        Some(6)
    );
}

#[test]
fn tlc_dot_parser_skips_rank_and_unknown_lines() {
    let graph = parse_tlc_dot_text(
        r#"
strict digraph DiskGraph {
node [shape=box,style=rounded]
1 [label="one",style = filled]
{rank = same; 1;2;}
unknown text
1 -> 2;
2 [label="two",tooltip="two"];
}
"#,
    )
    .expect("parse DOT");

    assert_eq!(graph.states.len(), 2);
    assert_eq!(graph.transitions.len(), 1);
    assert_eq!(graph.states.get(&1).expect("state 1").depth, Some(0));
    assert_eq!(graph.states.get(&2).expect("state 2").depth, Some(1));
}

#[test]
fn tlc_dot_edge_label_with_space_keeps_action_none() {
    let graph = parse_tlc_dot_text(
        r#"
1 [label="one",style = filled]
2 [label="two"]
1 -> 2 [label = "Next",color="black"];
2 -> 1 [label="Back",color="black"];
"#,
    )
    .expect("parse DOT");

    assert_eq!(graph.transitions[0].action, None);
    assert_eq!(graph.transitions[1].action.as_deref(), Some("Back"));
}

#[test]
fn tlc_dot_smoke_success_uses_legacy_check_name_and_detail() {
    let check = check_parse_tlc_dot_smoke(&repo_root());

    assert!(check.ok, "{check:?}");
    assert_eq!(check.name, "cmd:test_parse_tlc_dot_smoke.py");
    assert_eq!(
        check.detail.as_deref(),
        Some("OK parse_tlc_dot_smoke: states=14 edges=72 initials=1")
    );
}

#[tokio::test]
async fn stale_coverage_thresholds_are_error_level() {
    let temp = tempfile::tempdir().expect("tempdir");
    let metrics = temp.path().join("metrics");
    std::fs::create_dir(&metrics).expect("metrics dir");
    std::fs::write(
        metrics.join("spec_coverage.json"),
        serde_json::json!({
            "generated_at": "2026-01-01T00:00:00+00:00",
            "binary_info": {"git_commit": "not-a-real-commit"}
        })
        .to_string(),
    )
    .expect("write coverage");

    let now = chrono::Utc
        .with_ymd_and_hms(2026, 1, 9, 0, 0, 0)
        .single()
        .expect("valid datetime");
    let check = super::check_spec_coverage_freshness_with_now(temp.path(), now).await;

    assert_eq!(check_level(&check), "err");
    assert!(!check.ok);
    assert!(check
        .detail
        .as_deref()
        .expect("detail")
        .contains("age=192h"));
}

fn write_spec_coverage(project_root: &Path, value: serde_json::Value) {
    let metrics = project_root.join("metrics");
    std::fs::create_dir(&metrics).expect("metrics dir");
    std::fs::write(metrics.join("spec_coverage.json"), value.to_string()).expect("write coverage");
}

#[tokio::test]
async fn missing_coverage_snapshot_is_warning_with_refresh_hint() {
    let temp = tempfile::tempdir().expect("tempdir");
    let now = chrono::Utc
        .with_ymd_and_hms(2026, 1, 2, 0, 0, 0)
        .single()
        .expect("valid datetime");

    let check = check_spec_coverage_freshness_with_now(temp.path(), now).await;

    assert_eq!(check.level.as_deref(), Some("warn"));
    assert!(check.ok);
    assert!(check
        .detail
        .as_deref()
        .expect("detail")
        .contains("missing; run: cargo run --release --bin ty -- diagnose --output-metrics"));
}

#[tokio::test]
async fn invalid_coverage_snapshot_json_is_error() {
    let temp = tempfile::tempdir().expect("tempdir");
    let metrics = temp.path().join("metrics");
    std::fs::create_dir(&metrics).expect("metrics dir");
    std::fs::write(metrics.join("spec_coverage.json"), "NOT JSON").expect("write coverage");
    let now = chrono::Utc
        .with_ymd_and_hms(2026, 1, 2, 0, 0, 0)
        .single()
        .expect("valid datetime");

    let check = check_spec_coverage_freshness_with_now(temp.path(), now).await;

    assert_eq!(check_level(&check), "err");
    assert!(!check.ok);
    assert!(check
        .detail
        .as_deref()
        .expect("detail")
        .contains("invalid json"));
}

#[tokio::test]
async fn missing_coverage_generated_at_is_error() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_spec_coverage(
        temp.path(),
        serde_json::json!({"schema_version": 7, "summary": {"pass": 93}}),
    );
    let now = chrono::Utc
        .with_ymd_and_hms(2026, 1, 2, 0, 0, 0)
        .single()
        .expect("valid datetime");

    let check = check_spec_coverage_freshness_with_now(temp.path(), now).await;

    assert_eq!(check_level(&check), "err");
    assert!(!check.ok);
    assert_eq!(check.detail.as_deref(), Some("missing generated_at field"));
}

#[tokio::test]
async fn fresh_coverage_without_commit_field_is_ok() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_spec_coverage(
        temp.path(),
        serde_json::json!({
            "schema_version": 7,
            "generated_at": "2026-01-01T23:00:00+00:00",
            "summary": {"pass": 93}
        }),
    );
    let now = chrono::Utc
        .with_ymd_and_hms(2026, 1, 2, 0, 0, 0)
        .single()
        .expect("valid datetime");

    let check = check_spec_coverage_freshness_with_now(temp.path(), now).await;

    assert_eq!(check.level.as_deref(), Some("ok"));
    assert!(check.ok);
    assert_eq!(check.detail.as_deref(), Some("age=1h, drift=unknown"));
}

#[tokio::test]
async fn fresh_coverage_with_unresolvable_commit_is_warning() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_spec_coverage(
        temp.path(),
        serde_json::json!({
            "schema_version": 7,
            "generated_at": "2026-01-01T23:00:00+00:00",
            "binary_info": {"git_commit": "deadbeef"}
        }),
    );
    let now = chrono::Utc
        .with_ymd_and_hms(2026, 1, 2, 0, 0, 0)
        .single()
        .expect("valid datetime");

    let check = check_spec_coverage_freshness_with_now(temp.path(), now).await;

    assert_eq!(check.level.as_deref(), Some("warn"));
    assert!(check.ok);
    assert!(check.detail.as_deref().expect("detail").contains(
        "drift=unknown; refresh: cargo run --release --bin ty -- diagnose --output-metrics"
    ));
}
