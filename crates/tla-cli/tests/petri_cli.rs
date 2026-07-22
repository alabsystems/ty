// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for `ty petri`, `ty mcc`, and `ty petri-simplify`
//! CLI subcommands.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

mod common;

/// Minimal P/T net: two places, one transition (P0 -> T0 -> P1), initial marking [1, 0].
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

/// ReachabilityFireability property XML with one formula: EF is-fireable(T0).
const REACHABILITY_FIREABILITY_XML: &str = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/pnml">
  <property>
    <id>TestNet-ReachabilityFireability-0</id>
    <description>Can T0 fire?</description>
    <formula>
      <exists-path>
        <finally>
          <is-fireable>
            <transition>T0</transition>
          </is-fireable>
        </finally>
      </exists-path>
    </formula>
  </property>
</property-set>"#;

fn write_model_dir(dir: &common::TempDir) {
    std::fs::write(dir.path.join("model.pnml"), SIMPLE_PNML).expect("write model.pnml");
}

fn write_model_dir_with_properties(dir: &common::TempDir) {
    write_model_dir(dir);
    std::fs::write(
        dir.path.join("ReachabilityFireability.xml"),
        REACHABILITY_FIREABILITY_XML,
    )
    .expect("write ReachabilityFireability.xml");
}

fn run_petri(args: &[&str]) -> (i32, String, String) {
    common::run_tla_parsed_with_timeout(args, Duration::from_secs(30))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("tla-cli should live two levels below the workspace root")
        .to_path_buf()
}

fn target_debug_dir() -> PathBuf {
    let ty_bin = PathBuf::from(env!("CARGO_BIN_EXE_ty"));
    let parent = ty_bin
        .parent()
        .expect("CARGO_BIN_EXE_ty should have a parent directory");
    if parent.file_name().is_some_and(|name| name == "deps") {
        parent
            .parent()
            .expect("target debug deps dir should have a parent")
            .to_path_buf()
    } else {
        parent.to_path_buf()
    }
}

fn ty_mcc_binary() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_ty-mcc") {
        return PathBuf::from(path);
    }

    let binary = target_debug_dir().join(format!("ty-mcc{}", std::env::consts::EXE_SUFFIX));
    if binary.exists() {
        return binary;
    }

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .current_dir(workspace_root())
        .args(["build", "--quiet", "-p", "tla-petri", "--bin", "ty-mcc"])
        .output()
        .expect("build ty-mcc binary");
    assert!(
        output.status.success(),
        "cargo build -p tla-petri --bin ty-mcc failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        binary.exists(),
        "built ty-mcc binary should exist at {}",
        binary.display()
    );
    binary
}

/// Locate the in-tree `ty-mcc-summarize-evidence` Rust binary (the
/// Replaces a former Python helper).
/// Mirrors `ty_mcc_binary()` — builds on demand so the test is
/// self-contained.
fn ty_mcc_summarize_evidence_binary() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_ty-mcc-summarize-evidence") {
        return PathBuf::from(path);
    }
    let binary = target_debug_dir().join(format!(
        "ty-mcc-summarize-evidence{}",
        std::env::consts::EXE_SUFFIX
    ));
    if binary.exists() {
        return binary;
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .current_dir(workspace_root())
        .args([
            "build",
            "--quiet",
            "-p",
            "tla-petri",
            "--bin",
            "ty-mcc-summarize-evidence",
        ])
        .output()
        .expect("build ty-mcc-summarize-evidence binary");
    assert!(
        output.status.success(),
        "cargo build -p tla-petri --bin ty-mcc-summarize-evidence failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        binary.exists(),
        "built ty-mcc-summarize-evidence binary should exist at {}",
        binary.display()
    );
    binary
}

/// Locate the in-tree `ty-mcc-backend-evidence-validate` Rust binary
/// (the Replaces a former Python helper).
/// Mirrors `ty_mcc_binary()` — builds on demand so the test is
/// self-contained.
fn ty_mcc_backend_evidence_validate_binary() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_ty-mcc-backend-evidence-validate") {
        return PathBuf::from(path);
    }
    let binary = target_debug_dir().join(format!(
        "ty-mcc-backend-evidence-validate{}",
        std::env::consts::EXE_SUFFIX
    ));
    if binary.exists() {
        return binary;
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .current_dir(workspace_root())
        .args([
            "build",
            "--quiet",
            "-p",
            "tla-petri",
            "--bin",
            "ty-mcc-backend-evidence-validate",
        ])
        .output()
        .expect("build ty-mcc-backend-evidence-validate binary");
    assert!(
        output.status.success(),
        "cargo build -p tla-petri --bin ty-mcc-backend-evidence-validate failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        binary.exists(),
        "built ty-mcc-backend-evidence-validate binary should exist at {}",
        binary.display()
    );
    binary
}

// ---------------------------------------------------------------------------
// ty petri
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_reachability_deadlock_succeeds() {
    let dir = common::TempDir::new("petri-deadlock");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "ReachabilityDeadlock",
    ]);
    assert_eq!(
        code, 0,
        "petri ReachabilityDeadlock should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("FORMULA") && stdout.contains("TECHNIQUES"),
        "output should contain MCC FORMULA line.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_state_space_succeeds() {
    let dir = common::TempDir::new("petri-statespace");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "StateSpace",
    ]);
    assert_eq!(
        code, 0,
        "petri StateSpace should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    // StateSpace prints STATE_SPACE lines, not FORMULA.
    assert!(
        stdout.contains("STATE_SPACE") || stdout.contains("FORMULA"),
        "output should contain state-space or formula info.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_reachability_fireability_succeeds() {
    let dir = common::TempDir::new("petri-reach-fire");
    write_model_dir_with_properties(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "ReachabilityFireability",
    ]);
    assert_eq!(
        code, 0,
        "petri ReachabilityFireability should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("FORMULA"),
        "output should contain FORMULA line.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_one_safe_succeeds() {
    let dir = common::TempDir::new("petri-onesafe");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "OneSafe",
    ]);
    assert_eq!(
        code, 0,
        "petri OneSafe should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("FORMULA"),
        "output should contain FORMULA line.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_quasi_liveness_succeeds() {
    let dir = common::TempDir::new("petri-quasilive");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "QuasiLiveness",
    ]);
    assert_eq!(
        code, 0,
        "petri QuasiLiveness should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("FORMULA"),
        "output should contain FORMULA line.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_stable_marking_succeeds() {
    let dir = common::TempDir::new("petri-stablemark");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "StableMarking",
    ]);
    assert_eq!(
        code, 0,
        "petri StableMarking should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("FORMULA"),
        "output should contain FORMULA line.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_liveness_succeeds() {
    let dir = common::TempDir::new("petri-liveness");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "Liveness",
    ]);
    assert_eq!(
        code, 0,
        "petri Liveness should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("FORMULA"),
        "output should contain FORMULA line.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_rejects_unknown_examination() {
    let dir = common::TempDir::new("petri-unknown-exam");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, _stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "BogusExamination",
    ]);
    assert_ne!(
        code, 0,
        "petri BogusExamination should fail.\nstderr: {stderr}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_rejects_missing_model() {
    let (code, _stdout, stderr) = run_petri(&[
        "petri",
        "/tmp/nonexistent-model-dir/model.pnml",
        "--examination",
        "ReachabilityDeadlock",
    ]);
    assert_ne!(
        code, 0,
        "petri with missing model should fail.\nstderr: {stderr}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_threads_flag_accepted() {
    let dir = common::TempDir::new("petri-threads");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "ReachabilityDeadlock",
        "--threads",
        "2",
    ]);
    assert_eq!(
        code, 0,
        "petri with --threads 2 should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_storage_memory_flag_accepted() {
    let dir = common::TempDir::new("petri-storage");
    write_model_dir(&dir);
    let model_path = dir.path.join("model.pnml");

    let (code, stdout, stderr) = run_petri(&[
        "petri",
        model_path.to_str().unwrap(),
        "--examination",
        "ReachabilityDeadlock",
        "--storage",
        "memory",
    ]);
    assert_eq!(
        code, 0,
        "petri with --storage memory should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// ty mcc
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_with_dir_and_examination_succeeds() {
    let dir = common::TempDir::new("mcc-dir-exam");
    write_model_dir(&dir);

    let (code, stdout, stderr) = run_petri(&[
        "mcc",
        dir.path.to_str().unwrap(),
        "--examination",
        "ReachabilityDeadlock",
    ]);
    assert_eq!(
        code, 0,
        "mcc with dir and --examination should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("FORMULA"),
        "output should contain FORMULA line.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_uses_bk_env_vars() {
    let dir = common::TempDir::new("mcc-bk-env");
    write_model_dir(&dir);
    let dir_str = dir.path.to_str().unwrap();

    let (code, stdout, stderr) = common::run_tla_parsed_with_env_timeout(
        &["mcc"],
        &[
            ("BK_INPUT", dir_str),
            ("BK_EXAMINATION", "ReachabilityDeadlock"),
        ],
        &[],
        Duration::from_secs(30),
    );
    assert_eq!(
        code, 0,
        "mcc with BK_INPUT/BK_EXAMINATION should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("FORMULA"),
        "output should contain FORMULA line.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_fails_without_examination() {
    let dir = common::TempDir::new("mcc-no-exam");
    write_model_dir(&dir);

    // Remove BK_EXAMINATION from env to ensure it's truly missing.
    let (code, _stdout, stderr) = common::run_tla_parsed_with_env_timeout(
        &["mcc", dir.path.to_str().unwrap()],
        &[],
        &["BK_EXAMINATION"],
        Duration::from_secs(10),
    );
    assert_ne!(
        code, 0,
        "mcc without examination should fail.\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("examination not specified"),
        "error should mention missing examination.\nstderr: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// MCC format compliance: all non-property examinations produce valid output
// ---------------------------------------------------------------------------

/// Validate that a stdout line matches the MCC FORMULA output format.
/// Returns true if valid, false otherwise.
fn validate_mcc_formula_line(line: &str) -> bool {
    // MCC examination output: property examinations emit `FORMULA <id> <verdict>`
    // lines, while the StateSpace examination emits
    // `STATE_SPACE <MEASURE> <value>` lines. Both carry a non-empty TECHNIQUES
    // suffix and are valid MCC output.
    if !line.starts_with("FORMULA ") && !line.starts_with("STATE_SPACE ") {
        return false;
    }
    if !line.contains(" TECHNIQUES ") {
        return false;
    }
    // Must end with technique name(s) — uppercase letters and underscores
    let after_techniques = line.split(" TECHNIQUES ").last().unwrap_or("");
    !after_techniques.trim().is_empty()
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_format_compliance_all_non_property_examinations() {
    // All 6 non-property examinations must produce valid FORMULA lines
    let non_property_exams = [
        "ReachabilityDeadlock",
        "StateSpace",
        "OneSafe",
        "QuasiLiveness",
        "StableMarking",
        "Liveness",
    ];

    let dir = common::TempDir::new("mcc-format-all");
    write_model_dir(&dir);

    for exam in non_property_exams {
        let (code, stdout, stderr) =
            run_petri(&["mcc", dir.path.to_str().unwrap(), "--examination", exam]);
        assert_eq!(
            code, 0,
            "mcc {exam} should succeed.\nstdout: {stdout}\nstderr: {stderr}"
        );

        // Every non-empty stdout line must be a valid FORMULA line
        let formula_lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
        assert!(
            !formula_lines.is_empty(),
            "mcc {exam} should produce at least one FORMULA line.\nstdout: {stdout}"
        );
        for line in &formula_lines {
            assert!(
                validate_mcc_formula_line(line),
                "mcc {exam}: invalid MCC line format: '{line}'\nfull stdout: {stdout}"
            );
        }

        // Verify no debug/log output leaked to stdout
        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            assert!(
                trimmed.starts_with("FORMULA ") || trimmed.starts_with("STATE_SPACE"),
                "mcc {exam}: non-FORMULA output on stdout: '{trimmed}'"
            );
        }
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_format_compliance_property_examination_without_xml_produces_cannot_compute() {
    // Property examinations without XML should produce CANNOT_COMPUTE
    let property_exams = [
        "ReachabilityCardinality",
        "ReachabilityFireability",
        "CTLCardinality",
        "CTLFireability",
        "LTLCardinality",
        "LTLFireability",
        "UpperBounds",
    ];

    let dir = common::TempDir::new("mcc-format-noprop");
    write_model_dir(&dir);
    // No property XML files written — just model.pnml

    for exam in property_exams {
        let (code, stdout, stderr) =
            run_petri(&["mcc", dir.path.to_str().unwrap(), "--examination", exam]);
        // Should either succeed with CANNOT_COMPUTE or fail gracefully
        if code == 0 {
            // If it succeeds, stdout should contain CANNOT_COMPUTE
            let has_formula = stdout.contains("FORMULA");
            let has_cannot = stdout.contains("CANNOT_COMPUTE");
            assert!(
                has_formula || has_cannot,
                "mcc {exam}: succeeded but no FORMULA/CANNOT_COMPUTE output.\nstdout: {stdout}\nstderr: {stderr}"
            );
        }
        // Non-zero exit is also acceptable for missing XML (error reported on stderr)
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_format_compliance_reachability_fireability_with_xml() {
    // With property XML, should produce valid FORMULA lines with TRUE/FALSE verdicts
    let dir = common::TempDir::new("mcc-format-props");
    write_model_dir_with_properties(&dir);

    let (code, stdout, stderr) = run_petri(&[
        "mcc",
        dir.path.to_str().unwrap(),
        "--examination",
        "ReachabilityFireability",
    ]);
    assert_eq!(
        code, 0,
        "mcc ReachabilityFireability with XML should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let formula_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("FORMULA "))
        .collect();
    assert!(
        !formula_lines.is_empty(),
        "should produce at least one FORMULA line"
    );
    for line in &formula_lines {
        assert!(
            validate_mcc_formula_line(line),
            "invalid MCC format: {line}"
        );
        // Property verdicts should be TRUE, FALSE, or CANNOT_COMPUTE
        assert!(
            line.contains(" TRUE TECHNIQUES ")
                || line.contains(" FALSE TECHNIQUES ")
                || line.contains(" CANNOT_COMPUTE TECHNIQUES "),
            "property verdict should be TRUE/FALSE/CANNOT_COMPUTE: {line}"
        );
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_bk_time_confinement_is_respected() {
    // BK_TIME_CONFINEMENT should be read and used as a deadline.
    // With a very short timeout, the tool should still exit cleanly (not crash).
    let dir = common::TempDir::new("mcc-timeout");
    write_model_dir(&dir);

    let (code, stdout, stderr) = common::run_tla_parsed_with_env_timeout(
        &[
            "mcc",
            dir.path.to_str().unwrap(),
            "--examination",
            "ReachabilityDeadlock",
        ],
        &[("BK_TIME_CONFINEMENT", "10")],
        &[],
        Duration::from_secs(30),
    );
    // Should exit cleanly (0) even with a tight timeout budget
    assert_eq!(
        code, 0,
        "mcc with BK_TIME_CONFINEMENT=10 should exit cleanly.\nstdout: {stdout}\nstderr: {stderr}"
    );
    // Should still produce FORMULA output (either a real answer or CANNOT_COMPUTE)
    assert!(
        stdout.contains("FORMULA"),
        "should produce FORMULA output even with timeout.\nstdout: {stdout}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_stdout_only_formula_lines_no_debug() {
    // MCC infrastructure only parses FORMULA lines from stdout.
    // Any other output on stdout is a format violation.
    let dir = common::TempDir::new("mcc-clean-stdout");
    write_model_dir(&dir);

    let (code, stdout, _stderr) = run_petri(&[
        "mcc",
        dir.path.to_str().unwrap(),
        "--examination",
        "ReachabilityDeadlock",
    ]);
    assert_eq!(code, 0);

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Only FORMULA lines should appear on stdout
        assert!(
            trimmed.starts_with("FORMULA "),
            "non-FORMULA output on stdout (format violation): '{trimmed}'"
        );
    }
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn mcc_backend_evidence_sidecar_keeps_stdout_clean() {
    let dir = common::TempDir::new("mcc-backend-evidence");
    write_model_dir(&dir);
    let evidence_path = dir.path.join("backend-evidence.jsonl");
    let evidence_path_string = evidence_path.to_string_lossy().into_owned();
    let empty_bin_dir = dir.path.join("empty-bin");
    std::fs::create_dir(&empty_bin_dir).expect("empty bin dir should be created");
    let empty_bin_dir_string = empty_bin_dir.to_string_lossy().into_owned();
    let home_string = dir.path.to_string_lossy().into_owned();

    let (code, stdout, stderr) = common::run_tla_parsed_with_env_timeout(
        &[
            "mcc",
            dir.path.to_str().unwrap(),
            "--examination",
            "ReachabilityDeadlock",
        ],
        &[
            ("TY_MCC_BACKEND_EVIDENCE_JSONL", &evidence_path_string),
            ("HOME", &home_string),
            ("PATH", &empty_bin_dir_string),
        ],
        &["AY_PATH"],
        Duration::from_secs(30),
    );
    assert_eq!(
        code, 0,
        "mcc should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );

    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        assert!(
            line.trim().starts_with("FORMULA "),
            "non-FORMULA output on stdout (format violation): '{line}'"
        );
    }

    let evidence = std::fs::read_to_string(&evidence_path).expect("read evidence sidecar");
    let record: serde_json::Value = serde_json::from_str(evidence.trim()).expect("json evidence");
    assert_eq!(
        record["model"],
        dir.path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("model directory name")
    );
    assert_eq!(record["examination"], "ReachabilityDeadlock");
    assert_eq!(record["run_status"]["status"], "completed");
    assert_eq!(
        record["report"]["production_routing_status"],
        "JustifiedLocalFallback"
    );
    assert_eq!(record["report"]["ay_selected_for_production"], false);
    assert_eq!(record["report"]["has_unjustified_local_production"], false);

    let selected = record["report"]["selected"]
        .as_array()
        .expect("evidence should include selected backend lanes");
    let explicit = selected
        .iter()
        .find(|lane| lane["backend_code"] == "explicit_state")
        .expect("explicit-state lane should be selected");
    assert_eq!(explicit["role"], "production");
    assert_eq!(explicit["status"], "available");

    let rejected = record["report"]["rejected"]
        .as_array()
        .expect("evidence should include rejected backend lanes");
    let ay = rejected
        .iter()
        .find(|lane| lane["backend_code"] == "external_ay_binary")
        .expect("external AY lane should be rejected when no ay is discoverable");
    assert_eq!(ay["reason_code"], "missing_binary");
    assert!(
        record["report"]["evidence"]
            .as_array()
            .expect("report evidence should be an array")
            .iter()
            .any(|entry| entry
                .as_str()
                .is_some_and(|entry| entry.contains("mcc run completed records=1"))),
        "evidence should include completed run status: {record}"
    );
}

#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn ty_mcc_backend_evidence_jsonl_can_be_summarized_without_stdout_contamination() {
    let dir = common::TempDir::new("ty-mcc-backend-summary");
    write_model_dir(&dir);
    let evidence_path = dir.path.join("backend-evidence.jsonl");
    let empty_bin_dir = dir.path.join("empty-bin");
    std::fs::create_dir(&empty_bin_dir).expect("empty bin dir should be created");

    let mcc_output = Command::new(ty_mcc_binary())
        .arg(&dir.path)
        .arg("--examination")
        .arg("ReachabilityDeadlock")
        .arg("--backend-evidence-jsonl")
        .arg(&evidence_path)
        .env("HOME", &dir.path)
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

    let validation_output = Command::new(ty_mcc_backend_evidence_validate_binary())
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

    let summary_output = Command::new(ty_mcc_summarize_evidence_binary())
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
    // ty-mcc also appends a build-provenance sidecar row to the backend-evidence
    // file, so the detail rows include it alongside the examination evidence.
    // Select the examination evidence row (the one carrying lane routing) rather
    // than asserting a fixed total row count.
    // The build-provenance sidecar row renders `selected_lanes` as "-"; the real
    // examination evidence row carries actual lane routing. Select the latter.
    let exam_rows: Vec<&serde_json::Value> = rows
        .iter()
        .filter(|r| {
            r.get("selected_lanes")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|s| s != "-")
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

// ---------------------------------------------------------------------------
// ty petri-simplify
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_simplify_produces_json() {
    let dir = common::TempDir::new("petri-simplify");
    write_model_dir_with_properties(&dir);

    let (code, stdout, stderr) = run_petri(&[
        "petri-simplify",
        dir.path.to_str().unwrap(),
        "--examination",
        "ReachabilityFireability",
    ]);
    assert_eq!(
        code, 0,
        "petri-simplify should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    );

    // Verify output is valid JSON with expected fields.
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("output should be valid JSON");
    assert!(
        json.get("model_name").is_some(),
        "JSON should have model_name field.\nstdout: {stdout}"
    );
    assert!(
        json.get("examination").is_some(),
        "JSON should have examination field.\nstdout: {stdout}"
    );
    assert!(
        json.get("summary").is_some(),
        "JSON should have summary field.\nstdout: {stdout}"
    );
    assert!(
        json.get("properties").is_some(),
        "JSON should have properties field.\nstdout: {stdout}"
    );
    assert_eq!(
        json["examination"].as_str(),
        Some("ReachabilityFireability"),
        "examination field should match."
    );
    assert_eq!(
        json["source_kind"].as_str(),
        Some("ptnet"),
        "source_kind should be ptnet."
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_simplify_rejects_non_property_examination() {
    let dir = common::TempDir::new("petri-simplify-bad");
    write_model_dir(&dir);

    let (code, _stdout, stderr) = run_petri(&[
        "petri-simplify",
        dir.path.to_str().unwrap(),
        "--examination",
        "ReachabilityDeadlock",
    ]);
    assert_ne!(
        code, 0,
        "petri-simplify with ReachabilityDeadlock should fail (no property XML).\nstderr: {stderr}"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn petri_simplify_rejects_missing_model() {
    let (code, _stdout, stderr) = run_petri(&[
        "petri-simplify",
        "/tmp/nonexistent-model-dir",
        "--examination",
        "ReachabilityFireability",
    ]);
    assert_ne!(
        code, 0,
        "petri-simplify with missing model should fail.\nstderr: {stderr}"
    );
}
