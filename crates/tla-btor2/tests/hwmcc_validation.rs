// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! HWMCC'25 validation test harness.
//!
//! Reads the HWMCC'25 word-level BV results CSV to determine ground-truth
//! expected results (consensus across all participating solvers), then
//! validates that the TY BTOR2 parser can handle every benchmark file.
//!
//! ## Modes
//!
//! **Parse-only (default):** For every benchmark with a consensus result,
//! parse the BTOR2 file and verify that parsing succeeds. This validates
//! the parser covers all opcodes and constructs used in real-world
//! hardware model checking benchmarks.
//!
//! **Full solve (opt-in):** Set `TY_HWMCC_SOLVE=1` to additionally run
//! the BTOR2 model checker on each benchmark and compare the result against
//! the HWMCC consensus. This is expensive and intended for CI or manual
//! validation runs.
//!
//! ## Data sources
//!
//! - Results CSV: `~/hwmcc/results/hwmcc25-wordlevel-bv.csv`
//!   Columns: benchmark, config, result, time_real, time_cpu, memory
//!   Results: sat, unsat, none, unknown, timeout, memout, error
//!
//! - Benchmark files: `~/hwmcc/benchmarks/wordlevel/bv/<benchmark_path>`
//!
//! Run with: cargo test -p tla-btor2 --test hwmcc_validation

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use tla_btor2::parser::parse_file;
use tla_btor2::PortfolioConfig;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// Expected result for a benchmark (consensus across all HWMCC solvers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HwmccResult {
    /// All properties are safe (unreachable bad states).
    Unsat,
    /// At least one property is violated (reachable bad state).
    Sat,
}

/// A benchmark entry with its expected result and filesystem path.
#[derive(Debug)]
struct BenchmarkEntry {
    /// Relative path as it appears in the CSV (e.g., "2019/beem/bakery.3.prop1-func-interl.btor2").
    relative_path: String,
    /// Consensus result from HWMCC.
    expected: HwmccResult,
    /// Absolute path to the BTOR2 file.
    absolute_path: PathBuf,
}

// ---------------------------------------------------------------------------
// CSV loading
// ---------------------------------------------------------------------------

/// HWMCC'25 word-level track (each has its own benchmark tree + results CSV).
#[derive(Debug, Clone, Copy)]
enum Track {
    /// Bit-vector track (`wordlevel/bv`).
    Bv,
    /// Array track (`wordlevel/array`) — Track 2, the array-engine target.
    Array,
}

impl Track {
    fn dir_name(self) -> &'static str {
        match self {
            Track::Bv => "bv",
            Track::Array => "array",
        }
    }

    fn csv_name(self) -> &'static str {
        match self {
            Track::Bv => "hwmcc25-wordlevel-bv.csv",
            Track::Array => "hwmcc25-wordlevel-array.csv",
        }
    }
}

/// Root directory for a track's benchmarks.
fn hwmcc_track_dir(track: Track) -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home)
        .join("hwmcc/benchmarks/wordlevel")
        .join(track.dir_name())
}

/// Path to a track's results CSV.
fn hwmcc_track_csv_path(track: Track) -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home)
        .join("hwmcc/results")
        .join(track.csv_name())
}

/// Root directory for HWMCC'25 word-level BV benchmarks.
fn hwmcc_bv_dir() -> PathBuf {
    hwmcc_track_dir(Track::Bv)
}

/// Path to the HWMCC'25 results CSV.
fn hwmcc_csv_path() -> PathBuf {
    hwmcc_track_csv_path(Track::Bv)
}

/// Skip-guard: the HWMCC benchmark corpus is an external, machine-local data
/// set (not vendored in this repo). When the results CSV is absent, the parse
/// validation tests have no ground truth to run against, so they early-return
/// rather than panic. Returns `true` (and prints a SKIP line) when the corpus
/// is missing. Guards ONLY on explicit absence of the CSV — a present-but-
/// malformed CSV still surfaces as a real parser/loader failure.
fn hwmcc_corpus_absent() -> bool {
    let csv_path = hwmcc_csv_path();
    if !csv_path.exists() {
        eprintln!(
            "SKIP hwmcc_validation: benchmark corpus/CSV absent at {}",
            csv_path.display()
        );
        return true;
    }
    false
}

/// Load the HWMCC results CSV and compute consensus results (BV track).
fn load_consensus_results() -> Vec<BenchmarkEntry> {
    load_consensus_results_for(Track::Bv)
}

/// Load a track's HWMCC results CSV and compute consensus results.
///
/// A benchmark has consensus when all tools that returned a definitive
/// answer (sat or unsat) agree. Benchmarks where no tool returned a
/// definitive answer, or where tools disagree, are excluded. The CSV
/// consensus is the ONLY ground truth used (never directory-label
/// inference).
fn load_consensus_results_for(track: Track) -> Vec<BenchmarkEntry> {
    let csv_path = hwmcc_track_csv_path(track);
    if !csv_path.exists() {
        panic!(
            "HWMCC results CSV not found at {}. Set up ~/hwmcc/ to run these tests.",
            csv_path.display()
        );
    }

    let bv_dir = hwmcc_track_dir(track);

    // Collect all definitive results per benchmark.
    let contents = std::fs::read_to_string(&csv_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", csv_path.display()));

    // Validate header.
    let header = contents.lines().next().expect("CSV must have header");
    assert!(
        header.contains("benchmark"),
        "CSV header should contain 'benchmark', got: {header}"
    );

    // Parse all data lines (skip header).
    let csv_lines: Vec<&str> = contents.lines().skip(1).collect();
    let mut results_map: HashMap<String, Vec<String>> = HashMap::new();

    for line in &csv_lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 3 {
            continue;
        }
        let benchmark = fields[0].to_string();
        let result = fields[2].trim().to_string();
        results_map.entry(benchmark).or_default().push(result);
    }

    // Compute consensus: all sat/unsat results must agree.
    let mut entries = Vec::new();
    for (benchmark, results) in &results_map {
        let definitive: Vec<&str> = results
            .iter()
            .filter(|r| *r == "sat" || *r == "unsat")
            .map(|s| s.as_str())
            .collect();

        if definitive.is_empty() {
            // No tool returned a definitive result.
            continue;
        }

        // Check consensus: all definitive results must be the same.
        let first = definitive[0];
        if !definitive.iter().all(|r| *r == first) {
            // Conflict among solvers — skip this benchmark.
            eprintln!("WARNING: conflicting results for {benchmark}, skipping");
            continue;
        }

        let expected = match first {
            "sat" => HwmccResult::Sat,
            "unsat" => HwmccResult::Unsat,
            _ => unreachable!(),
        };

        let absolute_path = bv_dir.join(benchmark);
        if !absolute_path.exists() {
            continue;
        }

        entries.push(BenchmarkEntry {
            relative_path: benchmark.clone(),
            expected,
            absolute_path,
        });
    }

    // Sort for deterministic test ordering.
    entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    entries
}

/// Load only BEEM protocol benchmarks (small, directly relevant to TY).
fn load_beem_benchmarks() -> Vec<BenchmarkEntry> {
    load_consensus_results()
        .into_iter()
        .filter(|e| e.relative_path.contains("/beem/"))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests: Parse-only validation
// ---------------------------------------------------------------------------

/// Validate that every BEEM benchmark with consensus parses successfully.
///
/// BEEM benchmarks are small protocol models (bakery, collision, brp2, exit,
/// pgm_protocol) — directly relevant to TY's domain.
#[test]
fn test_hwmcc_beem_benchmarks_parse() {
    if hwmcc_corpus_absent() {
        return;
    }
    let benchmarks = load_beem_benchmarks();
    assert!(
        !benchmarks.is_empty(),
        "expected at least one BEEM benchmark with consensus"
    );

    let mut failures = Vec::new();
    let mut success_count = 0;

    for entry in &benchmarks {
        match parse_file(&entry.absolute_path) {
            Ok(prog) => {
                success_count += 1;
                // Sanity: every BEEM benchmark should have at least one bad property.
                assert!(
                    !prog.bad_properties.is_empty(),
                    "{}: parsed but has no bad properties",
                    entry.relative_path
                );
            }
            Err(e) => {
                failures.push(format!("{}: {e}", entry.relative_path));
            }
        }
    }

    eprintln!(
        "BEEM parse validation: {success_count}/{} passed",
        benchmarks.len()
    );

    assert!(
        failures.is_empty(),
        "failed to parse {} BEEM benchmark(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// True iff a parse error is the KNOWN, fail-closed design limit: TY
/// represents bitvectors as `u128`, so any sort wider than 128 bits is
/// declined at parse time (no verdict is ever produced for such a net — the
/// benchmark is outside TY's current class, honestly reported, never
/// guessed). Anything else is a REAL parser gap and fails the test.
fn is_known_width_limit(err: &impl std::fmt::Display) -> bool {
    err.to_string()
        .contains("exceeds maximum supported width 128")
}

/// Validate that ALL benchmarks with consensus parse successfully.
///
/// This covers all 291 benchmarks where HWMCC solvers agreed on the result,
/// spanning multiple categories: BEEM protocols, Goel industrial circuits,
/// Mann designs, Wolf designs, HKUST arithmetic, SosyLab, and YosysHQ.
///
/// Corpus reality (measured 2026-07): 63/291 BV-track consensus benchmarks
/// use bitvector sorts wider than 128 bits — the known `u128` design limit,
/// counted separately and reported honestly; the test fails on any OTHER
/// parse error (a genuine opcode/format gap).
#[test]
fn test_hwmcc_all_consensus_benchmarks_parse() {
    if hwmcc_corpus_absent() {
        return;
    }
    let benchmarks = load_consensus_results();
    assert!(
        benchmarks.len() > 100,
        "expected >100 consensus benchmarks, got {}",
        benchmarks.len()
    );

    let mut failures = Vec::new();
    let mut width_unsupported = 0;
    let mut success_count = 0;
    let mut sat_count = 0;
    let mut unsat_count = 0;

    for entry in &benchmarks {
        match parse_file(&entry.absolute_path) {
            Ok(_prog) => {
                success_count += 1;
                match entry.expected {
                    HwmccResult::Sat => sat_count += 1,
                    HwmccResult::Unsat => unsat_count += 1,
                }
            }
            Err(e) if is_known_width_limit(&e) => width_unsupported += 1,
            Err(e) => {
                failures.push(format!("{}: {e}", entry.relative_path));
            }
        }
    }

    eprintln!(
        "HWMCC parse validation: {success_count}/{} parsed (sat={sat_count}, unsat={unsat_count}), \
         {width_unsupported} outside the 128-bit width limit (declined fail-closed)",
        benchmarks.len()
    );

    assert!(
        success_count >= 220,
        "parsed-benchmark floor regressed: {success_count}"
    );
    assert!(
        failures.is_empty(),
        "REAL parse gap in {} benchmark(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Verify expected result distribution matches HWMCC'25 data.
#[test]
fn test_hwmcc_result_distribution() {
    if hwmcc_corpus_absent() {
        return;
    }
    let benchmarks = load_consensus_results();

    let sat_count = benchmarks
        .iter()
        .filter(|b| b.expected == HwmccResult::Sat)
        .count();
    let unsat_count = benchmarks
        .iter()
        .filter(|b| b.expected == HwmccResult::Unsat)
        .count();

    // From the CSV analysis: 103 sat, 188 unsat, 0 conflicts.
    // Allow some tolerance in case benchmarks are added/removed.
    assert!(
        sat_count >= 90,
        "expected >= 90 sat benchmarks, got {sat_count}"
    );
    assert!(
        unsat_count >= 170,
        "expected >= 170 unsat benchmarks, got {unsat_count}"
    );
    assert_eq!(
        sat_count + unsat_count,
        benchmarks.len(),
        "all benchmarks should be sat or unsat"
    );

    eprintln!(
        "HWMCC result distribution: {} total ({sat_count} sat, {unsat_count} unsat)",
        benchmarks.len()
    );
}

/// Validate structural properties of parsed BEEM benchmarks.
///
/// Verifies that each benchmark has the expected number of states, inputs,
/// and bad properties based on manual inspection.
#[test]
fn test_hwmcc_beem_structural_properties() {
    if hwmcc_corpus_absent() {
        return;
    }
    let _ = load_beem_benchmarks(); // ensures data is available

    // Expected values from grep -c on the benchmark files.
    let expected: &[(&str, usize, usize, usize)] = &[
        // (relative_path, num_states, num_inputs, num_bad)
        ("2019/beem/bakery.3.prop1-func-interl.btor2", 28, 24, 1),
        ("2019/beem/brp2.3.prop3-func-interl.btor2", 44, 33, 1),
        ("2019/beem/collision.1.prop1-func-interl.btor2", 27, 31, 1),
        ("2019/beem/exit.3.prop1-back-serstep.btor2", 52, 139, 1),
        (
            "2019/beem/pgm_protocol.8.prop6-func-interl.btor2",
            178,
            120,
            1,
        ),
    ];

    let bv_dir = hwmcc_bv_dir();
    for &(relative, exp_states, exp_inputs, exp_bad) in expected {
        let path = bv_dir.join(relative);
        if !path.exists() {
            continue;
        }
        let prog = parse_file(&path).unwrap_or_else(|e| panic!("failed to parse {relative}: {e}"));

        assert_eq!(
            prog.num_states, exp_states,
            "{relative}: expected {exp_states} states, got {}",
            prog.num_states
        );
        assert_eq!(
            prog.num_inputs, exp_inputs,
            "{relative}: expected {exp_inputs} inputs, got {}",
            prog.num_inputs
        );
        assert_eq!(
            prog.bad_properties.len(),
            exp_bad,
            "{relative}: expected {exp_bad} bad properties, got {}",
            prog.bad_properties.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: Full solve validation (opt-in via TY_HWMCC_SOLVE=1)
// ---------------------------------------------------------------------------

/// Run the full BTOR2 model checker on BEEM benchmarks and compare results.
///
/// This test is gated by the `TY_HWMCC_SOLVE` environment variable.
/// Set `TY_HWMCC_SOLVE=1` to enable.
#[test]
fn test_hwmcc_beem_solve() {
    if std::env::var("TY_HWMCC_SOLVE").unwrap_or_default() != "1" {
        eprintln!("SKIP: set TY_HWMCC_SOLVE=1 to run full solve validation");
        return;
    }

    let benchmarks = load_beem_benchmarks();
    assert!(!benchmarks.is_empty());

    let config = PortfolioConfig {
        time_budget: Some(Duration::from_secs(30)),
        enable_coi: true,
        enable_simplify: true,
        enable_bmc: true,
        bmc_budget_fraction: 0.2,
        bmc_max_depth: 20,
        verbose: false,
    };

    let mut correct = 0;
    let mut wrong = 0;
    let mut timeout = 0;
    let mut errors = Vec::new();

    for entry in &benchmarks {
        let prog = match parse_file(&entry.absolute_path) {
            Ok(p) => p,
            Err(e) => {
                errors.push(format!("{}: parse error: {e}", entry.relative_path));
                wrong += 1;
                continue;
            }
        };

        let (results, stats) = match tla_btor2::check_btor2_portfolio(&prog, &config) {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("{}: solver error: {e}", entry.relative_path));
                wrong += 1;
                continue;
            }
        };

        // Determine our verdict: any sat result means sat, all unsat means unsat.
        let any_sat = results
            .iter()
            .any(|r| matches!(r, tla_btor2::Btor2CheckResult::Sat { .. }));
        let any_unknown = results
            .iter()
            .any(|r| matches!(r, tla_btor2::Btor2CheckResult::Unknown { .. }));

        if any_unknown && !any_sat {
            timeout += 1;
            eprintln!(
                "  {} -> TIMEOUT (expected {:?}, phase={:?})",
                entry.relative_path, entry.expected, stats.result_phase
            );
            continue;
        }

        let our_result = if any_sat {
            HwmccResult::Sat
        } else {
            HwmccResult::Unsat
        };

        if our_result == entry.expected {
            correct += 1;
            eprintln!(
                "  {} -> CORRECT ({:?}, phase={:?}, COI {}/{})",
                entry.relative_path,
                our_result,
                stats.result_phase,
                stats.states_after_coi,
                stats.states_before_coi,
            );
        } else {
            wrong += 1;
            errors.push(format!(
                "{}: expected {:?}, got {:?}",
                entry.relative_path, entry.expected, our_result
            ));
            eprintln!(
                "  {} -> WRONG (expected {:?}, got {:?})",
                entry.relative_path, entry.expected, our_result
            );
        }
    }

    eprintln!(
        "BEEM solve validation: {correct} correct, {wrong} wrong, {timeout} timeout / {} total",
        benchmarks.len()
    );
    assert!(
        errors.is_empty(),
        "WRONG answers ({wrong}):\n{}",
        errors.join("\n")
    );
}

/// Run the full BTOR2 model checker on ALL consensus benchmarks.
///
/// Gated by `TY_HWMCC_SOLVE=all`. Uses the portfolio pipeline
/// (COI + BMC preprocessing + full CHC) with a per-benchmark timeout.
#[test]
fn test_hwmcc_all_solve() {
    if std::env::var("TY_HWMCC_SOLVE").unwrap_or_default() != "all" {
        eprintln!("SKIP: set TY_HWMCC_SOLVE=all to run full solve on all benchmarks");
        return;
    }

    let timeout_secs: u64 = std::env::var("TY_HWMCC_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    let benchmarks = load_consensus_results();
    let config = PortfolioConfig {
        time_budget: Some(Duration::from_secs(timeout_secs)),
        enable_coi: true,
        enable_simplify: true,
        enable_bmc: true,
        bmc_budget_fraction: 0.2,
        bmc_max_depth: 20,
        verbose: false,
    };

    let mut correct = 0;
    let mut wrong = 0;
    let mut timeout = 0;
    let mut error_count = 0;
    let mut wrong_details = Vec::new();

    for (idx, entry) in benchmarks.iter().enumerate() {
        let prog = match parse_file(&entry.absolute_path) {
            Ok(p) => p,
            Err(e) => {
                error_count += 1;
                eprintln!(
                    "  [{}/{}] {} -> ERROR (parse: {e})",
                    idx + 1,
                    benchmarks.len(),
                    entry.relative_path,
                );
                continue;
            }
        };

        let (results, stats) = match tla_btor2::check_btor2_portfolio(&prog, &config) {
            Ok(r) => r,
            Err(e) => {
                error_count += 1;
                eprintln!(
                    "  [{}/{}] {} -> ERROR (solver: {e})",
                    idx + 1,
                    benchmarks.len(),
                    entry.relative_path,
                );
                continue;
            }
        };

        let any_sat = results
            .iter()
            .any(|r| matches!(r, tla_btor2::Btor2CheckResult::Sat { .. }));
        let any_unknown = results
            .iter()
            .any(|r| matches!(r, tla_btor2::Btor2CheckResult::Unknown { .. }));

        if any_unknown && !any_sat {
            timeout += 1;
            eprintln!(
                "  [{}/{}] {} -> TIMEOUT (expected {:?})",
                idx + 1,
                benchmarks.len(),
                entry.relative_path,
                entry.expected,
            );
            continue;
        }

        let our_result = if any_sat {
            HwmccResult::Sat
        } else {
            HwmccResult::Unsat
        };

        if our_result == entry.expected {
            correct += 1;
            eprintln!(
                "  [{}/{}] {} -> CORRECT ({:?}, phase={:?}, COI {}/{}, {:.1}s)",
                idx + 1,
                benchmarks.len(),
                entry.relative_path,
                our_result,
                stats.result_phase,
                stats.states_after_coi,
                stats.states_before_coi,
                stats.total_time.as_secs_f64(),
            );
        } else {
            wrong += 1;
            wrong_details.push(format!(
                "{}: expected {:?}, got {:?}",
                entry.relative_path, entry.expected, our_result
            ));
            eprintln!(
                "  [{}/{}] {} -> WRONG (expected {:?}, got {:?})",
                idx + 1,
                benchmarks.len(),
                entry.relative_path,
                entry.expected,
                our_result,
            );
        }
    }

    let total = benchmarks.len();
    eprintln!(
        "\nHWMCC full solve: {correct} correct, {wrong} wrong, {timeout} timeout, {error_count} error / {total} total"
    );
    eprintln!(
        "Accuracy: {correct}/{} ({:.1}% of resolved)",
        correct + wrong,
        if correct + wrong > 0 {
            100.0 * correct as f64 / (correct + wrong) as f64
        } else {
            0.0
        }
    );

    assert!(
        wrong == 0,
        "WRONG answers ({wrong}):\n{}",
        wrong_details.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Tests: Array track (Track 2) — parse validation + flag-gated lane smoke
// ---------------------------------------------------------------------------

/// Skip-guard for the array track (mirrors `hwmcc_corpus_absent`).
fn hwmcc_array_corpus_absent() -> bool {
    let csv_path = hwmcc_track_csv_path(Track::Array);
    if !csv_path.exists() {
        eprintln!(
            "SKIP hwmcc_validation (array track): CSV absent at {}",
            csv_path.display()
        );
        return true;
    }
    false
}

/// Every array-track benchmark with solver consensus must either parse or
/// hit the KNOWN 128-bit width design limit (declined fail-closed at parse —
/// corpus reality measured 2026-07: 118/178 consensus array benchmarks use
/// >128-bit scalar sorts; the remaining 60 parse). Any other parse error is
/// a genuine gap and fails the test.
#[test]
fn test_hwmcc_array_benchmarks_parse() {
    if hwmcc_array_corpus_absent() {
        return;
    }
    let benchmarks = load_consensus_results_for(Track::Array);
    assert!(
        benchmarks.len() > 100,
        "expected >100 consensus array benchmarks, got {}",
        benchmarks.len()
    );

    let mut failures = Vec::new();
    let mut width_unsupported = 0;
    let mut success_count = 0;
    let mut sat_count = 0;
    let mut unsat_count = 0;

    for entry in &benchmarks {
        match parse_file(&entry.absolute_path) {
            Ok(_prog) => {
                success_count += 1;
                match entry.expected {
                    HwmccResult::Sat => sat_count += 1,
                    HwmccResult::Unsat => unsat_count += 1,
                }
            }
            Err(e) if is_known_width_limit(&e) => width_unsupported += 1,
            Err(e) => failures.push(format!("{}: {e}", entry.relative_path)),
        }
    }

    eprintln!(
        "HWMCC array-track parse validation: {success_count}/{} parsed (sat={sat_count}, unsat={unsat_count}), \
         {width_unsupported} outside the 128-bit width limit (declined fail-closed)",
        benchmarks.len()
    );
    assert!(
        success_count >= 55,
        "parsed-benchmark floor regressed: {success_count}"
    );
    assert!(
        failures.is_empty(),
        "REAL parse gap in {} array benchmark(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Consensus distribution for the array track (from the triage of the
/// downloaded CSV: 178 consensus benchmarks — 60 sat, 118 unsat, zero
/// cross-solver disagreements). Tolerant bounds in case of corpus updates.
#[test]
fn test_hwmcc_array_result_distribution() {
    if hwmcc_array_corpus_absent() {
        return;
    }
    let benchmarks = load_consensus_results_for(Track::Array);
    let sat_count = benchmarks
        .iter()
        .filter(|b| b.expected == HwmccResult::Sat)
        .count();
    let unsat_count = benchmarks
        .iter()
        .filter(|b| b.expected == HwmccResult::Unsat)
        .count();
    assert!(sat_count >= 50, "expected >= 50 sat, got {sat_count}");
    assert!(
        unsat_count >= 100,
        "expected >= 100 unsat, got {unsat_count}"
    );
    assert_eq!(sat_count + unsat_count, benchmarks.len());
    eprintln!(
        "HWMCC array-track distribution: {} total ({sat_count} sat, {unsat_count} unsat)",
        benchmarks.len()
    );
}

/// Flag-gated REAL-corpus smoke for the lazy-array BMC lane: the "run first"
/// triage list, verdicts compared ONLY against the CSV consensus.
///
/// Gated by `TY_HWMCC_ARRAY_SMOKE=1` (runs solvers; minutes of wall time).
///
/// Soundness contract asserted here:
/// * an `Unsafe` lane verdict (replay-gated by construction) must match a
///   `sat` consensus — an Unsafe on an `unsat` consensus is a WRONG-UNSAFE
///   and fails the test loudly;
/// * `BoundedNoCex` / `Declined` are honest NON-verdicts (reported in the
///   table, never counted as answers).
/// FULL 2020/mann FAMILY sweep for the array-IC3 lane — the lane's design-target
/// class (lt200/pred1 are its members). Gated by `TY_HWMCC_ARRAY_FAMILY=1`.
/// Consensus-checked: a wrong verdict FAILS the test; Proved/Unsafe tallied,
/// declines/timeouts reported honestly.
#[test]
fn test_hwmcc_array_mann_family_sweep() {
    if std::env::var("TY_HWMCC_ARRAY_FAMILY").unwrap_or_default() != "1" {
        return;
    }
    if hwmcc_array_corpus_absent() {
        return;
    }
    let consensus: HashMap<String, HwmccResult> = load_consensus_results_for(Track::Array)
        .into_iter()
        .map(|e| (e.relative_path, e.expected))
        .collect();
    let dir = hwmcc_track_dir(Track::Array);
    let fam = dir.join("2020/mann");
    let budget = Duration::from_secs(
        std::env::var("TY_HWMCC_ARRAY_FAMILY_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120),
    );
    let mut names: Vec<String> = std::fs::read_dir(&fam)
        .expect("family dir")
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().and_then(|x| x.to_str()) == Some("btor2"))
                .then(|| format!("2020/mann/{}", p.file_name().unwrap().to_string_lossy()))
        })
        .collect();
    names.sort();
    if let Ok(f) = std::env::var("TY_HWMCC_ARRAY_FAMILY_ONLY") {
        names.retain(|n| n.contains(f.as_str()));
    }
    let (mut proved, mut unsafe_found, mut other) = (0usize, 0usize, 0usize);
    let mut wrong: Vec<String> = Vec::new();
    for rel in &names {
        let expected = consensus.get(rel).copied();
        let src = std::fs::read_to_string(dir.join(rel)).expect("read net");
        let prog = match tla_btor2::parse_btor2(&src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{rel}: PARSE-FAIL ({e}) — skipped");
                other += 1;
                continue;
            }
        };
        let t = std::time::Instant::now();
        let ic3 = tla_btor2::check_array_ic3(
            &prog,
            &tla_btor2::ArrayIc3Config {
                time_budget: Some(budget),
                ..Default::default()
            },
        );
        let dt = t.elapsed().as_secs_f64();
        match ic3 {
            tla_btor2::ArrayIc3Outcome::ProvedSafe { tier, .. } => {
                let ok = expected.map(|e| e == HwmccResult::Unsat);
                eprintln!("{rel}: PROVED-SAFE ({tier:?}) vs {expected:?} [{dt:.1}s]");
                if ok == Some(false) {
                    wrong.push(rel.clone());
                }
                proved += 1;
            }
            tla_btor2::ArrayIc3Outcome::Unsafe { depth, .. } => {
                let ok = expected.map(|e| e == HwmccResult::Sat);
                eprintln!("{rel}: UNSAFE@{depth} (replay-gated) vs {expected:?} [{dt:.1}s]");
                if ok == Some(false) {
                    wrong.push(rel.clone());
                }
                unsafe_found += 1;
            }
            outcome => {
                eprintln!("{rel}: {outcome:?} — honest non-verdict [{dt:.1}s]");
                other += 1;
            }
        }
    }
    eprintln!(
        "[mann-family] {} nets: {proved} PROVED, {unsafe_found} UNSAFE, {other} non-verdicts",
        names.len()
    );
    assert!(wrong.is_empty(), "WRONG verdicts vs consensus: {wrong:?}");
}

#[test]
fn test_hwmcc_array_runfirst_lane_smoke() {
    if std::env::var("TY_HWMCC_ARRAY_SMOKE").unwrap_or_default() != "1" {
        eprintln!("SKIP: set TY_HWMCC_ARRAY_SMOKE=1 to run the array-lane corpus smoke");
        return;
    }
    if hwmcc_array_corpus_absent() {
        return;
    }
    let consensus: HashMap<String, HwmccResult> = load_consensus_results_for(Track::Array)
        .into_iter()
        .map(|e| (e.relative_path, e.expected))
        .collect();

    // The triage "run first" list (see the phase-2 plan): in-slice smoke,
    // smallest first, then the wide-iw bit-blast-INELIGIBLE pair that is the
    // lane's true class.
    let run_first = [
        "2020/mann/array_lt200.btor2",
        "2020/mann/simple-stack-pred1.btor2",
        "2025/sosylab/safety-func/array-fpi/ifcompf.btor2",
        "2025/sosylab/safety-func/array-fpi/ifcomp.btor2",
    ];

    let dir = hwmcc_track_dir(Track::Array);
    let budget = Duration::from_secs(
        std::env::var("TY_HWMCC_ARRAY_SMOKE_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60),
    );

    // Optional spec filter (TY_HWMCC_ARRAY_SMOKE_ONLY=substring) for targeted
    // long-budget runs of a single benchmark without paying for the others.
    let only = std::env::var("TY_HWMCC_ARRAY_SMOKE_ONLY").ok();
    let mut rows = Vec::new();
    let mut wrong = Vec::new();
    for rel in run_first {
        if let Some(f) = &only {
            if !rel.contains(f.as_str()) {
                continue;
            }
        }
        let Some(&expected) = consensus.get(rel) else {
            rows.push(format!("{rel}: NO CONSENSUS — skipped"));
            continue;
        };
        let path = dir.join(rel);
        let t0 = std::time::Instant::now();
        let prog = match parse_file(&path) {
            Ok(p) => p,
            Err(e) => {
                rows.push(format!("{rel}: PARSE ERROR: {e}"));
                wrong.push(format!("{rel}: parse error: {e}"));
                continue;
            }
        };
        let outcome = tla_btor2::check_array_bmc(
            &prog,
            &tla_btor2::ArrayBmcConfig {
                time_budget: Some(budget),
                ..tla_btor2::ArrayBmcConfig::default()
            },
        );
        let dt = t0.elapsed().as_secs_f64();
        match outcome {
            tla_btor2::ArrayBmcOutcome::Unsafe { depth, .. } => {
                let ok = expected == HwmccResult::Sat;
                rows.push(format!(
                    "{rel}: bmc-lane UNSAFE@{depth} (replay-gated) vs consensus {expected:?} -> {} [{dt:.1}s]",
                    if ok { "MATCH" } else { "WRONG-UNSAFE" }
                ));
                if !ok {
                    wrong.push(format!(
                        "{rel}: lane replay-gated Unsafe@{depth} contradicts unsat consensus"
                    ));
                }
            }
            tla_btor2::ArrayBmcOutcome::BoundedNoCex { depth_reached } => {
                rows.push(format!(
                    "{rel}: bmc-lane BoundedNoCex(depth {depth_reached}) — non-verdict (consensus {expected:?}) [{dt:.1}s]"
                ));
            }
            tla_btor2::ArrayBmcOutcome::Declined { reason } => {
                // Phase-3 F1/F2 regression gate: a refinement stall
                // ("no new axiom") is a completeness bug, not an honest
                // budget decline — it must never reappear.
                if reason.contains("no new axiom") {
                    wrong.push(format!(
                        "{rel}: bmc-lane refinement STALL ({reason}) — F1/F2 regression"
                    ));
                }
                rows.push(format!(
                    "{rel}: bmc-lane DECLINED ({reason}) — non-verdict (consensus {expected:?}) [{dt:.1}s]"
                ));
            }
        }

        // K-induction lane (the phase-2 stepping stone) on the same net.
        let t1 = std::time::Instant::now();
        let kind = tla_btor2::check_array_kinduction(
            &prog,
            &tla_btor2::ArrayKindConfig {
                time_budget: Some(budget),
                ..tla_btor2::ArrayKindConfig::default()
            },
        );
        let dt1 = t1.elapsed().as_secs_f64();
        match kind {
            tla_btor2::ArrayKindOutcome::ProvedSafe { k } => {
                let ok = expected == HwmccResult::Unsat;
                rows.push(format!(
                    "{rel}: kind-lane PROVED-SAFE(k={k}, LRAT-discharged) vs consensus {expected:?} -> {} [{dt1:.1}s]",
                    if ok { "MATCH" } else { "WRONG-SAFE" }
                ));
                if !ok {
                    wrong.push(format!(
                        "{rel}: kind ProvedSafe(k={k}) contradicts sat consensus — WRONG-SAFE"
                    ));
                }
            }
            tla_btor2::ArrayKindOutcome::Unsafe { depth, .. } => {
                let ok = expected == HwmccResult::Sat;
                rows.push(format!(
                    "{rel}: kind-lane UNSAFE@{depth} (replay-gated) vs consensus {expected:?} -> {} [{dt1:.1}s]",
                    if ok { "MATCH" } else { "WRONG-UNSAFE" }
                ));
                if !ok {
                    wrong.push(format!(
                        "{rel}: kind replay-gated Unsafe@{depth} contradicts unsat consensus"
                    ));
                }
            }
            tla_btor2::ArrayKindOutcome::BoundedNoCex { depth_reached } => {
                rows.push(format!(
                    "{rel}: kind-lane BoundedNoCex(depth {depth_reached}) — non-verdict (consensus {expected:?}) [{dt1:.1}s]"
                ));
            }
            tla_btor2::ArrayKindOutcome::Declined { reason } => {
                if reason.contains("no new axiom") {
                    wrong.push(format!(
                        "{rel}: kind-lane refinement STALL ({reason}) — F1/F2 regression"
                    ));
                }
                rows.push(format!(
                    "{rel}: kind-lane DECLINED ({reason}) — non-verdict (consensus {expected:?}) [{dt1:.1}s]"
                ));
            }
        }

        // IC3 frames lane (phase 3) on the same net.
        let t2 = std::time::Instant::now();
        let ic3 = tla_btor2::check_array_ic3(
            &prog,
            &tla_btor2::ArrayIc3Config {
                time_budget: Some(budget),
                ..tla_btor2::ArrayIc3Config::default()
            },
        );
        let dt2 = t2.elapsed().as_secs_f64();
        match ic3 {
            tla_btor2::ArrayIc3Outcome::ProvedSafe {
                converged_at,
                ref invariant,
                tier,
            } => {
                let ok = expected == HwmccResult::Unsat;
                rows.push(format!(
                    "{rel}: ic3-lane PROVED-SAFE(level={converged_at}, {} clauses/{} probes, {tier:?}) vs consensus {expected:?} -> {} [{dt2:.1}s]",
                    invariant.clauses.len(),
                    invariant.probes.len(),
                    if ok { "MATCH" } else { "WRONG-SAFE" }
                ));
                if !ok {
                    wrong.push(format!(
                        "{rel}: ic3 ProvedSafe contradicts sat consensus — WRONG-SAFE"
                    ));
                }
            }
            tla_btor2::ArrayIc3Outcome::Unsafe { depth, .. } => {
                let ok = expected == HwmccResult::Sat;
                rows.push(format!(
                    "{rel}: ic3-lane UNSAFE@{depth} (replay-gated) vs consensus {expected:?} -> {} [{dt2:.1}s]",
                    if ok { "MATCH" } else { "WRONG-UNSAFE" }
                ));
                if !ok {
                    wrong.push(format!(
                        "{rel}: ic3 replay-gated Unsafe@{depth} contradicts unsat consensus"
                    ));
                }
            }
            tla_btor2::ArrayIc3Outcome::BoundedNoCex { frames_completed } => {
                rows.push(format!(
                    "{rel}: ic3-lane BoundedNoCex({frames_completed} frames) — non-verdict (consensus {expected:?}) [{dt2:.1}s]"
                ));
            }
            tla_btor2::ArrayIc3Outcome::Declined { reason } => {
                if reason.contains("no new axiom") {
                    wrong.push(format!(
                        "{rel}: ic3-lane refinement STALL ({reason}) — F1/F2 regression"
                    ));
                }
                rows.push(format!(
                    "{rel}: ic3-lane DECLINED ({reason}) — non-verdict (consensus {expected:?}) [{dt2:.1}s]"
                ));
            }
        }
    }

    eprintln!("\n== array-lane run-first smoke ==");
    for r in &rows {
        eprintln!("  {r}");
    }
    assert!(
        wrong.is_empty(),
        "SOUNDNESS violation(s) in the lane smoke:\n{}",
        wrong.join("\n")
    );
}
