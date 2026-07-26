// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! End-to-end validation of the backend-evidence CLI pipeline.
//!
//! This test lives in `tla-petri`, which owns all three binaries. Cargo can
//! therefore provide their exact paths without a nested build that deadlocks
//! on the outer test process's target-directory lock.

use std::path::Path;
use std::process::Command;

const SIMPLE_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="TestNet" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <name><text>TestNet</text></name>
    <page id="p1">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <place id="P1"/>
      <transition id="T0"><name><text>fire</text></name></transition>
      <arc id="a1" source="P0" target="T0"/>
      <arc id="a2" source="T0" target="P1"/>
    </page>
  </net>
</pnml>"#;

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("tla-petri should live two levels below the workspace root")
}

#[test]
fn ty_mcc_backend_evidence_jsonl_can_be_summarized_without_stdout_contamination() {
    let dir = tempfile::tempdir().expect("create test directory");
    std::fs::write(dir.path().join("model.pnml"), SIMPLE_PNML).expect("write model.pnml");
    let evidence_path = dir.path().join("backend-evidence.jsonl");
    let empty_bin_dir = dir.path().join("empty-bin");
    std::fs::create_dir(&empty_bin_dir).expect("empty bin dir should be created");

    let mcc_output = Command::new(env!("CARGO_BIN_EXE_ty-mcc"))
        .arg(dir.path())
        .arg("--examination")
        .arg("ReachabilityDeadlock")
        .arg("--backend-evidence-jsonl")
        .arg(&evidence_path)
        .env("HOME", dir.path())
        .env("PATH", &empty_bin_dir)
        .env_remove("AY_PATH")
        .output()
        .expect("run ty-mcc");
    let mcc_stdout = String::from_utf8_lossy(&mcc_output.stdout).to_string();
    let mcc_stderr = String::from_utf8_lossy(&mcc_output.stderr).to_string();
    assert!(
        mcc_output.status.success(),
        "ty-mcc should succeed.\nstdout: {mcc_stdout}\nstderr: {mcc_stderr}"
    );
    assert!(
        mcc_stdout.lines().any(|line| line.starts_with("FORMULA ")),
        "ty-mcc should emit FORMULA output.\nstdout: {mcc_stdout}"
    );
    for line in mcc_stdout.lines().filter(|line| !line.trim().is_empty()) {
        assert!(
            line.trim().starts_with("FORMULA "),
            "non-FORMULA output on ty-mcc stdout (format violation): '{line}'"
        );
    }

    let validation_output = Command::new(env!("CARGO_BIN_EXE_ty-mcc-backend-evidence-validate"))
        .current_dir(workspace_root())
        .arg("--require")
        .arg("portfolio_route")
        .arg("--require")
        .arg("ay_solver_capability_descriptor")
        .arg("--require")
        .arg("ay_symbolic_execution_contract_manifest")
        .arg(&evidence_path)
        .output()
        .expect("validate backend evidence portfolio_route and AY rows");
    let validation_stdout = String::from_utf8_lossy(&validation_output.stdout).to_string();
    let validation_stderr = String::from_utf8_lossy(&validation_output.stderr).to_string();
    assert!(
        validation_output.status.success(),
        "runtime ty-mcc sidecar should pass promoted portfolio_route and AY validation.\nstdout: {validation_stdout}\nstderr: {validation_stderr}"
    );
    assert!(
        validation_stdout.contains("portfolio_route=6"),
        "validator should count exactly six canonical portfolio routes: {validation_stdout}"
    );
    assert!(
        validation_stdout.contains("ay_solver_capability_descriptor=1"),
        "validator should count the AY solver capability descriptor: {validation_stdout}"
    );
    assert!(
        validation_stdout.contains("ay_symbolic_execution_contract_manifest=1"),
        "validator should accept the AY symbolic contract manifest and health rows: {validation_stdout}"
    );

    let summary_output = Command::new(env!("CARGO_BIN_EXE_ty-mcc-summarize-evidence"))
        .arg("--summary-json")
        .arg(&evidence_path)
        .output()
        .expect("run backend evidence summarizer");
    let summary_stdout = String::from_utf8_lossy(&summary_output.stdout).to_string();
    let summary_stderr = String::from_utf8_lossy(&summary_output.stderr).to_string();
    assert!(
        summary_output.status.success(),
        "summarizer should succeed.\nstdout: {summary_stdout}\nstderr: {summary_stderr}"
    );

    let summary: serde_json::Value =
        serde_json::from_str(&summary_stdout).expect("summary JSON should parse");
    let routing_counts = summary["counts"]["production_routing_status"]
        .as_array()
        .expect("summary should include production routing counts");
    assert_eq!(
        routing_counts
            .iter()
            .find(|row| row["production_routing_status"] == "JustifiedLocalFallback")
            .and_then(|row| row["count"].as_u64()),
        Some(1),
        "summary should count the completed fallback-routed MCC run: {summary_stdout}"
    );

    let lane_counts = summary["counts"]["backend_lane_status"]
        .as_array()
        .expect("summary should include backend lane status counts");
    assert_eq!(
        lane_counts
            .iter()
            .find(|row| {
                row["lane"] == "selected"
                    && row["backend"] == "explicit_state"
                    && row["role"] == "production"
                    && row["status"] == "available"
            })
            .and_then(|row| row["count"].as_u64()),
        Some(1),
        "summary should count the explicit-state production lane: {summary_stdout}"
    );

    let reason_counts = summary["counts"]["reason_code"]
        .as_array()
        .expect("summary should include reason-code counts");
    assert_eq!(
        reason_counts
            .iter()
            .find(|row| {
                row["lane"] == "rejected"
                    && row["backend"] == "external_ay_binary"
                    && row["reason_code"] == "missing_binary"
            })
            .and_then(|row| row["count"].as_u64()),
        Some(1),
        "summary should count the missing AY reason code: {summary_stdout}"
    );

    let rows = summary["rows"]
        .as_array()
        .expect("summary should include row details");
    let exam_rows: Vec<&serde_json::Value> = rows
        .iter()
        .filter(|row| {
            row.get("selected_lanes")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|lanes| lanes != "-")
        })
        .collect();
    assert_eq!(
        exam_rows.len(),
        1,
        "summary should contain exactly one MCC examination evidence row (excluding the build-provenance sidecar): {summary_stdout}"
    );
    let exam_row = exam_rows[0];
    assert!(
        exam_row["selected_lanes"]
            .as_str()
            .is_some_and(|lanes| lanes.contains("explicit_state:production:available")),
        "summary row should render the selected explicit-state lane: {summary_stdout}"
    );
    assert!(
        exam_row["unsupported_reason_codes"]
            .as_str()
            .is_some_and(|codes| codes.contains("missing_binary")),
        "summary row should render the missing_binary reason code: {summary_stdout}"
    );
}
