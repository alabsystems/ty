// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Subprocess smoke test for `ty-mcc-csv-compare`.
//!
//! Writes a 2-row toy CSV (consensus column form), invokes the binary
//! against the in-tree `tests/mcc_benchmarks/mutex/` fixture, and
//! asserts the per-row TSV has the expected columns and that the
//! Wrong_Units count is zero on a known-good fixture.
//!
//! This is the CI fast-gate canary contract: if `ty-mcc` regresses on a
//! mutex-class verdict, this test fails in <30s instead of waiting for
//! the full `mcc_benchmarks` integration suite (or worse, the next
//! external sweep cycle).

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use tla_petri::mcc_unit_compare::numeric_units_equal;

fn workspace_root() -> PathBuf {
    // crates/tla-petri -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn csv_compare_mutex_smoke_passes_on_known_good_fixture() {
    let bin = env!("CARGO_BIN_EXE_ty-mcc-csv-compare");
    let ty_mcc = env!("CARGO_BIN_EXE_ty-mcc");

    let root = workspace_root();
    let models_root = root.join("tests").join("mcc_benchmarks");
    assert!(
        models_root.join("mutex").join("model.pnml").is_file(),
        "missing in-tree fixture: tests/mcc_benchmarks/mutex/model.pnml"
    );

    // Two-row toy CSV in MCC raw-result-analysis.csv schema (Input,
    // Examination, estimated result). The header columns must match
    // `clean_row` keys: `Input`, `Examination`, `estimated result`.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let csv_path = tmp.path().join("toy.csv");
    fs::write(
        &csv_path,
        "Input,Examination,estimated result\nmutex,ReachabilityDeadlock,F\nmutex,StateSpace,3 4 1 3\n",
    )
    .expect("write toy CSV");

    let tsv_path = tmp.path().join("results.tsv");
    let summary_path = tmp.path().join("summary.json");

    let status = Command::new(bin)
        .arg("--csv-path")
        .arg(&csv_path)
        .arg("--models-root")
        .arg(&models_root)
        .arg("--binary")
        .arg(ty_mcc)
        .arg("--exams")
        .arg("StateSpace,ReachabilityDeadlock")
        .arg("--subset")
        .arg("mutex")
        .arg("--threads")
        .arg("1")
        .arg("--timeout")
        .arg("60")
        .arg("--results-tsv")
        .arg(&tsv_path)
        .arg("--summary-json")
        .arg(&summary_path)
        .status()
        .expect("spawn ty-mcc-csv-compare");

    assert!(
        status.success(),
        "ty-mcc-csv-compare exited non-zero (code={:?}): expected zero wrong_units on mutex fixture",
        status.code()
    );

    let tsv =
        fs::read_to_string(&tsv_path).expect("ty-mcc-csv-compare did not write --results-tsv");
    let lines: Vec<&str> = tsv.lines().collect();
    assert!(
        lines.len() >= 3,
        "expected TSV header + 2 data rows; got {} lines:\n{}",
        lines.len(),
        tsv
    );
    let header = lines[0];
    // Canonical column set per the per-row schema in the task spec.
    for expected_col in [
        "Input",
        "Examination",
        "Expected",
        "Actual",
        "Category",
        "Exact_Units",
        "Wrong_Units",
        "Cannot_Compute_Units",
        "Missing_Units",
        "Time_Ms",
    ] {
        assert!(
            header.contains(expected_col),
            "TSV header missing column {expected_col:?}: {header}"
        );
    }

    // Every data row's Wrong_Units cell (column index 6) must be 0.
    // If this fails, ty-mcc regressed on a mutex verdict — investigate before
    // dismissing as a flake.
    let wrong_col_idx = header.split('\t').position(|c| c == "Wrong_Units").unwrap();
    for row in &lines[1..] {
        let cells: Vec<&str> = row.split('\t').collect();
        let wrong: u64 = cells[wrong_col_idx].parse().unwrap_or_else(|e| {
            panic!(
                "non-integer Wrong_Units cell {:?}: {e}",
                cells[wrong_col_idx]
            )
        });
        assert_eq!(
            wrong, 0,
            "ty-mcc regressed on mutex fixture — Wrong_Units > 0 in row: {row}"
        );
    }

    // Summary JSON should also exist and report wrong_units == 0.
    let summary_raw = fs::read_to_string(&summary_path).expect("summary.json not written");
    let summary: serde_json::Value =
        serde_json::from_str(&summary_raw).expect("summary.json not valid JSON");
    assert_eq!(
        summary["wrong_units"].as_u64(),
        Some(0),
        "summary.json reports wrong_units > 0: {summary_raw}"
    );

    // Dedup field invariants on a 2-distinct-case CSV: unique cases evaluated
    // equals row count, dedup factor = 1.0 (no replication).
    assert_eq!(
        summary["row_count"].as_u64(),
        Some(2),
        "row_count should match 2-row CSV: {summary_raw}"
    );
    assert_eq!(
        summary["unique_cases_evaluated"].as_u64(),
        Some(2),
        "unique_cases_evaluated should equal 2 on distinct-case CSV: {summary_raw}"
    );
}

/// Regression test for the (Input, Examination) dedup added 2026-05-23:
/// when `raw-result-analysis.csv` has 2 rows for the same (model, exam)
/// (the common case — one row per evaluating tool) the binary must
/// invoke `ty-mcc` exactly once and fan the cached verdict back out
/// across both original CSV rows in the TSV.
///
/// Mechanism: wrap `ty-mcc` in a shell script that bumps a counter file
/// before exec'ing the real binary, then assert the counter is `1`.
#[test]
fn csv_compare_dedups_repeated_input_exam_invocations() {
    let bin = env!("CARGO_BIN_EXE_ty-mcc-csv-compare");
    let ty_mcc = env!("CARGO_BIN_EXE_ty-mcc");

    let root = workspace_root();
    let models_root = root.join("tests").join("mcc_benchmarks");
    assert!(
        models_root.join("mutex").join("model.pnml").is_file(),
        "missing in-tree fixture: tests/mcc_benchmarks/mutex/model.pnml"
    );

    let tmp = tempfile::tempdir().expect("create tempdir");

    // Counter-bumping wrapper: each invocation appends a line to `counter`
    // then exec's the real ty-mcc binary with the original args.
    let wrapper_path = tmp.path().join("ty-mcc-counter.sh");
    let counter_path = tmp.path().join("counter");
    let wrapper_body = format!(
        "#!/usr/bin/env bash\nset -e\necho x >> {counter}\nexec {real} \"$@\"\n",
        counter = counter_path.display(),
        real = ty_mcc,
    );
    fs::write(&wrapper_path, wrapper_body).expect("write wrapper script");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&wrapper_path)
            .expect("stat wrapper")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&wrapper_path, perms).expect("chmod wrapper");
    }

    // Two rows, same (Input, Examination): simulates two tool-rows for the
    // same case on `raw-result-analysis.csv`. With the dedup fix the
    // wrapper must run exactly once; without it, twice.
    let csv_path = tmp.path().join("toy_dup.csv");
    fs::write(
        &csv_path,
        "Input,Examination,estimated result\nmutex,ReachabilityDeadlock,F\nmutex,ReachabilityDeadlock,F\n",
    )
    .expect("write toy CSV");

    let tsv_path = tmp.path().join("results.tsv");
    let summary_path = tmp.path().join("summary.json");

    let status = Command::new(bin)
        .arg("--csv-path")
        .arg(&csv_path)
        .arg("--models-root")
        .arg(&models_root)
        .arg("--binary")
        .arg(&wrapper_path)
        .arg("--exams")
        .arg("ReachabilityDeadlock")
        .arg("--subset")
        .arg("mutex")
        .arg("--threads")
        .arg("1")
        .arg("--timeout")
        .arg("60")
        .arg("--results-tsv")
        .arg(&tsv_path)
        .arg("--summary-json")
        .arg(&summary_path)
        .status()
        .expect("spawn ty-mcc-csv-compare");

    assert!(
        status.success(),
        "ty-mcc-csv-compare exited non-zero (code={:?}) on duplicate-row CSV",
        status.code()
    );

    // Counter file must have exactly 1 line — i.e. the wrapper (and thus
    // ty-mcc) was invoked once despite 2 input rows.
    let counter_contents = fs::read_to_string(&counter_path).unwrap_or_default();
    let invocations = counter_contents.lines().count();
    assert_eq!(
        invocations, 1,
        "expected ty-mcc to be invoked exactly once on duplicate (Input, Examination) rows; got {invocations} (counter contents: {counter_contents:?})"
    );

    // TSV must still contain BOTH original rows (per-row granularity
    // preserved) — header + 2 data rows.
    let tsv =
        fs::read_to_string(&tsv_path).expect("ty-mcc-csv-compare did not write --results-tsv");
    let data_rows: Vec<&str> = tsv.lines().skip(1).filter(|l| !l.is_empty()).collect();
    assert_eq!(
        data_rows.len(),
        2,
        "expected 2 data rows (one per original CSV row) — got {}:\n{}",
        data_rows.len(),
        tsv
    );

    // Summary JSON: backward-compatible fields preserved + new dedup fields.
    let summary_raw = fs::read_to_string(&summary_path).expect("summary.json not written");
    let summary: serde_json::Value =
        serde_json::from_str(&summary_raw).expect("summary.json not valid JSON");
    // Preserved field.
    assert_eq!(
        summary["rows"].as_u64(),
        Some(2),
        "rows field (backward-compat) should be 2: {summary_raw}"
    );
    // New fields.
    assert_eq!(
        summary["row_count"].as_u64(),
        Some(2),
        "row_count should be 2: {summary_raw}"
    );
    assert_eq!(
        summary["unique_cases_evaluated"].as_u64(),
        Some(1),
        "unique_cases_evaluated should be 1 on duplicate-row CSV: {summary_raw}"
    );
    // dedup_factor = 2.0 (2 rows / 1 unique case).
    let dedup = summary["dedup_factor"].as_f64().unwrap_or(0.0);
    assert!(
        (dedup - 2.0).abs() < 1e-9,
        "dedup_factor should be 2.0; got {dedup} ({summary_raw})"
    );
}

// ----------------------------------------------------------------------------
// Numeric-equivalence regression tests (see crates/tla-petri/src/mcc_unit_compare.rs).
//
// Surfaced by ab23e6e4: the MCC consensus CSV truncates large state counts
// into 5-significant-digit scientific notation (e.g. `1.3391E+6`), while
// TY emits full decimals (e.g. `1339104`). The verdict gate used to flag
// these as `wrong` despite numerical equivalence, corrupting every
// measurement run on COL StateSpace rows. The fix routes unit comparison
// through `numeric_units_equal` BEFORE falling back to string equality.
// These tests lock in the contract per the task spec.
// ----------------------------------------------------------------------------

#[test]
fn test_numeric_equivalence_scientific_vs_decimal() {
    // The canonical case from ab23e6e4's report: AirplaneLD-COL-0020
    // StateSpace transitions = `1.3391E+6` (CSV) vs `1339104` (TY).
    // Must be reported equal - these differ by ~3e-6 relative, well
    // inside the 5-sig-digit truncation envelope.
    assert!(
        numeric_units_equal("1.3391E+6", "1339104"),
        "AirplaneLD-COL-0020 transitions: 1.3391E+6 (CSV) must match 1339104 (TY)"
    );
    assert!(
        numeric_units_equal("1339104", "1.3391E+6"),
        "comparison must be symmetric"
    );
}

#[test]
fn test_numeric_equivalence_exact_match() {
    assert!(numeric_units_equal("42", "42"));
}

#[test]
fn test_numeric_equivalence_string_fallback() {
    // Non-numeric verdicts (`T`/`F`/`?`/`D`) must NOT match via the
    // numeric path - the helper returns false and the caller falls
    // back to string equality. This contract is what lets the binary
    // still compare boolean verdicts correctly via raw `==` after the
    // numeric check returns false.
    assert!(
        !numeric_units_equal("T", "T"),
        "string-typed verdicts must not pass via numeric path"
    );
}

#[test]
fn test_numeric_equivalence_genuine_mismatch() {
    // The Murphy-COL-D1N010 shape from ab23e6e4: TY pre-fix emitted 10,
    // consensus was 39780. The comparator MUST still catch real wrong
    // answers - that bug was masked by the scientific-notation noise
    // this commit fixes, so a regression here would re-mask it.
    assert!(
        !numeric_units_equal("10", "39780"),
        "Murphy-shape genuine wrong answer must remain flagged"
    );
}

#[test]
fn test_numeric_equivalence_negative() {
    // Negative-number parse path (i128 integer fast path on the LHS,
    // f64 fallback once one side has a decimal point).
    assert!(numeric_units_equal("-5", "-5.0"));
}

// ----------------------------------------------------------------------------
// Per-formula-examination dispatch regression tests.
//
// Pre-fix: `compare_output` constructed `expected.ids = vec![exam.as_str()]`
// for every non-`StateSpace` examination. For the 7 newer examinations
// (CTL/LTL/Reachability {Cardinality,Fireability} + UpperBounds) the CSV's
// `estimated result` is a per-formula positional vector (e.g. `T T T` for
// 3 formulas) so `ids.len() == 1 != units.len() == 3` always triggered
// `harness_expected_shape_error`. This blocked verdict-correctness
// measurement on every formula-based exam (1862 rows / 266 per exam on the
// 2026-05-24 13-exam sweep — see the report block in this commit).
//
// Post-fix: the harness reads `<model_dir>/<exam>.xml` to get the per-
// formula id vector (matching the CSV's positional order), then routes
// through the existing per-formula parser. The tests below pin one
// synthetic 2-row CSV per shape and assert the category is anything BUT
// `harness_expected_shape_error`.
// ----------------------------------------------------------------------------

/// Helper: invoke `ty-mcc-csv-compare` with a single-row CSV against a
/// fixture model dir and return the per-row TSV contents.
fn run_csv_compare_single(
    fixture: &str,
    exam: &str,
    estimated_result: &str,
) -> (std::process::ExitStatus, String) {
    let bin = env!("CARGO_BIN_EXE_ty-mcc-csv-compare");
    let ty_mcc = env!("CARGO_BIN_EXE_ty-mcc");

    let root = workspace_root();
    let models_root = root.join("tests").join("mcc_benchmarks");
    assert!(
        models_root.join(fixture).join("model.pnml").is_file(),
        "missing fixture: tests/mcc_benchmarks/{fixture}/model.pnml"
    );

    let tmp = tempfile::tempdir().expect("create tempdir");
    let csv_path = tmp.path().join("toy.csv");
    fs::write(
        &csv_path,
        format!("Input,Examination,estimated result\n{fixture},{exam},{estimated_result}\n"),
    )
    .expect("write toy CSV");

    let tsv_path = tmp.path().join("results.tsv");

    let status = Command::new(bin)
        .arg("--csv-path")
        .arg(&csv_path)
        .arg("--models-root")
        .arg(&models_root)
        .arg("--binary")
        .arg(ty_mcc)
        .arg("--exams")
        .arg(exam)
        .arg("--subset")
        .arg(fixture)
        .arg("--threads")
        .arg("1")
        .arg("--timeout")
        .arg("60")
        .arg("--results-tsv")
        .arg(&tsv_path)
        .status()
        .expect("spawn ty-mcc-csv-compare");

    let tsv =
        fs::read_to_string(&tsv_path).expect("ty-mcc-csv-compare did not write --results-tsv");
    (status, tsv)
}

/// Assert that the single data row in `tsv` is NOT a
/// `harness_expected_shape_error`. The point of these tests is to lock in
/// the dispatch fix: post-fix the category may be `exact`, `incomplete`,
/// `wrong`, `timeout`, `nonzero_exit`, or `malformed_output` depending on
/// what `ty-mcc` actually produces for the fixture — but it must NEVER
/// fall through to the harness-shape gate.
fn assert_not_shape_error(tsv: &str, context: &str) {
    let header = tsv.lines().next().expect("TSV header");
    let cat_idx = header
        .split('\t')
        .position(|c| c == "Category")
        .expect("Category column");
    for row in tsv.lines().skip(1).filter(|l| !l.is_empty()) {
        let cells: Vec<&str> = row.split('\t').collect();
        let cat = cells[cat_idx];
        assert_ne!(
            cat, "harness_expected_shape_error",
            "{context}: row regressed to harness_expected_shape_error (row: {row})",
        );
    }
}

#[test]
fn csv_compare_ctl_cardinality_no_shape_error() {
    // mutex/CTLCardinality.xml has 2 properties — synthetic CSV with 2
    // `?` units (unknown expected) so the post-fix harness produces a
    // categorized row regardless of ty-mcc's actual verdict.
    let (_status, tsv) = run_csv_compare_single("mutex", "CTLCardinality", "? ?");
    assert_not_shape_error(&tsv, "mutex/CTLCardinality");
}

#[test]
fn csv_compare_ctl_fireability_no_shape_error() {
    let (_status, tsv) = run_csv_compare_single("mutex", "CTLFireability", "? ?");
    assert_not_shape_error(&tsv, "mutex/CTLFireability");
}

#[test]
fn csv_compare_reachability_fireability_no_shape_error() {
    // mutex/ReachabilityFireability.xml has 3 properties.
    let (_status, tsv) = run_csv_compare_single("mutex", "ReachabilityFireability", "? ? ?");
    assert_not_shape_error(&tsv, "mutex/ReachabilityFireability");
}

#[test]
fn csv_compare_reachability_cardinality_no_shape_error() {
    // producer_consumer/ReachabilityCardinality.xml has 2 properties.
    let (_status, tsv) =
        run_csv_compare_single("producer_consumer", "ReachabilityCardinality", "? ?");
    assert_not_shape_error(&tsv, "producer_consumer/ReachabilityCardinality");
}

#[test]
fn csv_compare_ltl_cardinality_no_shape_error() {
    // token_ring/LTLCardinality.xml has 2 properties.
    let (_status, tsv) = run_csv_compare_single("token_ring", "LTLCardinality", "? ?");
    assert_not_shape_error(&tsv, "token_ring/LTLCardinality");
}

#[test]
fn csv_compare_ltl_fireability_no_shape_error() {
    let (_status, tsv) = run_csv_compare_single("token_ring", "LTLFireability", "? ?");
    assert_not_shape_error(&tsv, "token_ring/LTLFireability");
}

#[test]
fn csv_compare_upper_bounds_no_shape_error() {
    // producer_consumer/UpperBounds.xml has 2 properties; expected
    // values are integer-typed bounds (here `?` so any verdict is
    // categorized without `wrong_units` noise).
    let (_status, tsv) = run_csv_compare_single("producer_consumer", "UpperBounds", "? ?");
    assert_not_shape_error(&tsv, "producer_consumer/UpperBounds");
}

// ----------------------------------------------------------------------------
// Corpus-version mismatch guard tests (2026-05-25 incident remediation).
//
// On 2026-05-25 a `ty-mcc-csv-compare` run produced 1071 "wrong" rows that
// were actually a corpus hygiene issue: the harness read April-2024 property
// XMLs while the consensus CSV referenced May-2025 XML formula IDs. Same
// formula names, different formulas at each positional index. No
// wrong-answer signal — just stale data. The guard added in this commit
// detects the mismatch and categorizes affected rows as
// `harness_corpus_mismatch` (NEW category, distinct from `wrong`) plus
// emits a remediation hint to stderr.
//
// The tests below set up a synthetic `model_dir` with a stale-vintage XML
// (`-2024-` tags) and a CSV whose `run id` epoch parses to 2025 — the
// minimum reproducer of the incident. The CSV expected vector is built so
// the comparator would otherwise have produced false `wrong` rows.
// ----------------------------------------------------------------------------

/// Build a synthetic per-model directory under `parent` containing:
///   * `model.pnml` copied from the in-tree `mutex` fixture (so `ty-mcc`
///     succeeds and emits FORMULA lines).
///   * `CTLCardinality.xml` with 2 properties whose `<id>` tags embed
///     `vintage_year` (e.g. `-2024-` or `-2025-`). The PNML formulas are
///     intentionally toy (constant `0 <= 1` and `1 <= 0`) so the verdict
///     comparator path is exercised regardless of `ty-mcc` semantics.
fn write_year_tagged_ctl_xml(parent: &std::path::Path, model: &str, vintage_year: &str) {
    let model_dir = parent.join(model);
    fs::create_dir_all(&model_dir).expect("mkdir model dir");
    let root = workspace_root();
    let src_pnml = root.join("tests/mcc_benchmarks/mutex/model.pnml");
    fs::copy(&src_pnml, model_dir.join("model.pnml")).expect("copy model.pnml");
    let xml = format!(
        r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>{model}-CTLCardinality-{vintage_year}-00</id>
    <formula><all-paths><globally>
      <integer-le><integer-constant>0</integer-constant><integer-constant>1</integer-constant></integer-le>
    </globally></all-paths></formula>
  </property>
  <property>
    <id>{model}-CTLCardinality-{vintage_year}-01</id>
    <formula><all-paths><globally>
      <integer-le><integer-constant>1</integer-constant><integer-constant>0</integer-constant></integer-le>
    </globally></all-paths></formula>
  </property>
</property-set>
"#
    );
    fs::write(model_dir.join("CTLCardinality.xml"), xml).expect("write CTLCardinality.xml");
}

/// Write a MCC-shape `raw-result-analysis.csv` with one row whose `run id`
/// column embeds a Unix epoch corresponding to `vintage_year`. Uses the
/// canonical MCC column order so the bin's CSV loader populates `run id`
/// per the production format.
fn write_csv_with_run_id_vintage(
    csv_path: &std::path::Path,
    model: &str,
    exam: &str,
    estimated_result: &str,
    vintage_year: u32,
) {
    // Unix-epoch midpoint of each year (well within the year boundaries for
    // the simple civil_from_days math used by `vintage_year_from_run_id`).
    let epoch = match vintage_year {
        2024 => 1_717_000_000_u64,
        2025 => 1_748_537_266_u64,
        2026 => 1_780_000_000_u64,
        _ => panic!("unsupported vintage_year {vintage_year} in test helper"),
    };
    let run_id = format!("r001-tall-{epoch}00008");
    let csv = format!(
        "### tool,Input,Examination,nb cores,time flag,memory flag,results,techniques,max memory  (MB),CPU (ms),Time (ms),i/o wait (ms),Status,run id,flags:bonus:scores:mask,estimated result,# tools computing estimated results\n\
         ToolX,{model},{exam},1,OK,OK,FT,-,1000,100,100,0,normal,{run_id},FFFF:--:0:?,{estimated_result},1.0 1.0\n"
    );
    fs::write(csv_path, csv).expect("write CSV");
}

fn invoke_csv_compare(
    csv_path: &std::path::Path,
    models_root: &std::path::Path,
    binary: &str,
    exam: &str,
    subset: &str,
    tsv_path: &std::path::Path,
    summary_path: Option<&std::path::Path>,
    strict_corpus: bool,
) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_ty-mcc-csv-compare");
    let mut cmd = Command::new(bin);
    cmd.arg("--csv-path")
        .arg(csv_path)
        .arg("--models-root")
        .arg(models_root)
        .arg("--binary")
        .arg(binary)
        .arg("--exams")
        .arg(exam)
        .arg("--subset")
        .arg(subset)
        .arg("--threads")
        .arg("1")
        .arg("--timeout")
        .arg("60")
        .arg("--results-tsv")
        .arg(tsv_path);
    if let Some(p) = summary_path {
        cmd.arg("--summary-json").arg(p);
    }
    if strict_corpus {
        cmd.arg("--strict-corpus");
    }
    cmd.output().expect("spawn ty-mcc-csv-compare")
}

#[test]
fn test_corpus_mismatch_detected_and_categorized() {
    let ty_mcc = env!("CARGO_BIN_EXE_ty-mcc");
    let tmp = tempfile::tempdir().expect("create tempdir");

    // Stale XML: 2024-tagged IDs.
    let models_root = tmp.path().join("models");
    write_year_tagged_ctl_xml(&models_root, "synthetic_mutex", "2024");

    // CSV with 2025-vintage `run id`. Expected vector `TF` for the 2
    // properties; values irrelevant — the test asserts the row is BUCKETED
    // as `harness_corpus_mismatch` regardless of what TY emits.
    let csv_path = tmp.path().join("toy.csv");
    write_csv_with_run_id_vintage(&csv_path, "synthetic_mutex", "CTLCardinality", "TF", 2025);

    let tsv_path = tmp.path().join("results.tsv");
    let summary_path = tmp.path().join("summary.json");
    let out = invoke_csv_compare(
        &csv_path,
        &models_root,
        ty_mcc,
        "CTLCardinality",
        "synthetic_mutex",
        &tsv_path,
        Some(&summary_path),
        /* strict_corpus */ false,
    );

    // Default (non-strict) exit code: the binary still returns 0 or 1 per
    // wrong_units, never 2, even when mismatch is detected. The mismatch
    // rows are explicitly NOT counted as `wrong`.
    let code = out.status.code().unwrap_or(-1);
    assert_ne!(
        code,
        2,
        "default (non-strict) mode must not exit with code 2 on corpus mismatch (stderr: {})",
        String::from_utf8_lossy(&out.stderr)
    );

    // The categorized TSV row must be `harness_corpus_mismatch`, NEVER
    // `wrong`. This is the soundness floor of the fix: stale-corpus runs
    // must not inflate the wrong-count and look like a TY regression.
    let tsv = fs::read_to_string(&tsv_path).expect("results.tsv missing");
    let header = tsv.lines().next().expect("TSV header");
    let cat_idx = header
        .split('\t')
        .position(|c| c == "Category")
        .expect("Category column");
    let wrong_idx = header
        .split('\t')
        .position(|c| c == "Wrong_Units")
        .expect("Wrong_Units column");
    let data_rows: Vec<&str> = tsv.lines().skip(1).filter(|l| !l.is_empty()).collect();
    assert_eq!(data_rows.len(), 1, "expected 1 data row, got {}", tsv);
    let cells: Vec<&str> = data_rows[0].split('\t').collect();
    assert_eq!(
        cells[cat_idx], "harness_corpus_mismatch",
        "stale-corpus row must be categorized as `harness_corpus_mismatch`, not `wrong` (full row: {})",
        data_rows[0]
    );
    assert_eq!(
        cells[wrong_idx], "0",
        "stale-corpus row must report 0 wrong_units (would have been inflated pre-fix); row: {}",
        data_rows[0]
    );

    // Stderr must carry the remediation hint so a human (or AI) skimming
    // the log can fix the root cause immediately.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Re-point --models-root"),
        "stderr missing remediation hint `Re-point --models-root ...`; got:\n{stderr}"
    );
    assert!(
        stderr.contains("harness_corpus_mismatch") || stderr.contains("CORPUS MISMATCH"),
        "stderr missing corpus-mismatch signal; got:\n{stderr}"
    );
}

#[test]
fn test_corpus_match_passes_through() {
    let ty_mcc = env!("CARGO_BIN_EXE_ty-mcc");
    let tmp = tempfile::tempdir().expect("create tempdir");

    // Matching XML/CSV vintages: 2025/2025. Expected behavior: no
    // corpus-mismatch row, normal verdict comparison proceeds.
    let models_root = tmp.path().join("models");
    write_year_tagged_ctl_xml(&models_root, "synthetic_mutex", "2025");

    let csv_path = tmp.path().join("toy.csv");
    write_csv_with_run_id_vintage(&csv_path, "synthetic_mutex", "CTLCardinality", "? ?", 2025);

    let tsv_path = tmp.path().join("results.tsv");
    let out = invoke_csv_compare(
        &csv_path,
        &models_root,
        ty_mcc,
        "CTLCardinality",
        "synthetic_mutex",
        &tsv_path,
        None,
        /* strict_corpus */ false,
    );

    let tsv = fs::read_to_string(&tsv_path).expect("results.tsv missing");
    let header = tsv.lines().next().expect("TSV header");
    let cat_idx = header
        .split('\t')
        .position(|c| c == "Category")
        .expect("Category column");
    let data_rows: Vec<&str> = tsv.lines().skip(1).filter(|l| !l.is_empty()).collect();
    assert_eq!(data_rows.len(), 1, "expected 1 data row, got: {}", tsv);
    let cells: Vec<&str> = data_rows[0].split('\t').collect();
    assert_ne!(
        cells[cat_idx], "harness_corpus_mismatch",
        "matching vintages must NOT trigger corpus mismatch; row: {}",
        data_rows[0]
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("CORPUS MISMATCH"),
        "matching-vintage run must not emit corpus-mismatch warning; got:\n{stderr}"
    );
}

#[test]
fn test_corpus_mismatch_with_strict_flag_exits_nonzero() {
    let ty_mcc = env!("CARGO_BIN_EXE_ty-mcc");
    let tmp = tempfile::tempdir().expect("create tempdir");

    // Same stale-XML + 2025-CSV setup as the first mismatch test, but
    // with `--strict-corpus`. Expected exit code is 2 (hard fail per
    // the documented harness-error semantics).
    let models_root = tmp.path().join("models");
    write_year_tagged_ctl_xml(&models_root, "synthetic_mutex", "2024");

    let csv_path = tmp.path().join("toy.csv");
    write_csv_with_run_id_vintage(&csv_path, "synthetic_mutex", "CTLCardinality", "TF", 2025);

    let tsv_path = tmp.path().join("results.tsv");
    let out = invoke_csv_compare(
        &csv_path,
        &models_root,
        ty_mcc,
        "CTLCardinality",
        "synthetic_mutex",
        &tsv_path,
        None,
        /* strict_corpus */ true,
    );

    let code = out.status.code().unwrap_or(-1);
    assert_eq!(
        code,
        2,
        "--strict-corpus must exit with code 2 on detected mismatch (stderr: {})",
        String::from_utf8_lossy(&out.stderr)
    );
}
