// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (legacy archive parser accepts spaced 2025 archive variants by design.)

//! MCC 2025 history harness.
//!
//! Reads the MCC 2025 archive (CSV per-tool results, per-input tarballs),
//! parses each tool's stdout per the MCC protocol, and produces normalized
//! score / blocker / comparison reports. Routes every MCC keyword through
//! [`tla_petri::mcc_keywords`] and every examination name through
//! [`tla_petri::examination::Examination`] so the canonical Rust types are
//! the single source of truth.
//!
//! Legacy 2025-archive output emitted some MCC keywords with embedded
//! mcc-keyword-guard: allow-spaced-mention
//! spaces (`CANNOT COMPUTE`, `STATE SPACE`, `MAX TOKEN IN PLACE`, ...).
//! Those spaced forms are normalized to the canonical underscored variants
//! at parse time. New code in this crate must emit only the underscored
//! forms — see `crates/tla-petri/src/output.rs`.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::examination::Examination;
use crate::mcc_keywords::{
    CANNOT_COMPUTE, FORMULA, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING, STATES, STATE_SPACE,
    TECHNIQUES, TRANSITIONS,
};

// ---------- Constants ----------

/// Default MCC archive root: `$HOME/mcc-prev/2025`.
fn default_root() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("mcc-prev/2025")
}

/// Default `probe-macos` output cache dir, derived from the home directory.
fn default_probe_output_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        home.join("Library/Caches/ty-clean")
    } else {
        home.join(".cache/ty-clean")
    }
}
const BACKEND_EVIDENCE_JSONL_ENV: &str = "TY_MCC_BACKEND_EVIDENCE_JSONL";
const MCC_BACKEND_EVIDENCE_JSONL_ENV: &str = "MCC_BACKEND_EVIDENCE_JSONL";

/// Canonical MCC examination order, sourced from
/// [`Examination::ALL`] but reordered to the legacy 2025-archive order
/// the historical reports emit. Source of truth for membership is the
/// enum.
fn exams_in_report_order() -> [Examination; 13] {
    [
        Examination::StateSpace,
        Examination::ReachabilityDeadlock,
        Examination::OneSafe,
        Examination::QuasiLiveness,
        Examination::StableMarking,
        Examination::Liveness,
        Examination::UpperBounds,
        Examination::ReachabilityCardinality,
        Examination::ReachabilityFireability,
        Examination::CTLCardinality,
        Examination::CTLFireability,
        Examination::LTLCardinality,
        Examination::LTLFireability,
    ]
}

fn exam_order(name: &str) -> usize {
    exams_in_report_order()
        .iter()
        .position(|e| e.as_str() == name)
        .unwrap_or(999)
}

fn parse_exam_name(name: &str) -> Result<Examination> {
    Examination::from_name(name).map_err(|e| anyhow!("{e}"))
}

// ---------- Bucket specification ----------

#[derive(Debug, Clone)]
struct BucketSpec {
    description: &'static str,
    inputs: Option<&'static [&'static str]>,
    exams: Vec<Examination>,
}

fn all_exams() -> Vec<Examination> {
    exams_in_report_order().to_vec()
}

fn temporal_exams() -> Vec<Examination> {
    vec![
        Examination::CTLFireability,
        Examination::CTLCardinality,
        Examination::LTLFireability,
        Examination::LTLCardinality,
    ]
}

fn reachability_exams() -> Vec<Examination> {
    vec![
        Examination::ReachabilityDeadlock,
        Examination::OneSafe,
        Examination::StableMarking,
        Examination::ReachabilityCardinality,
        Examination::ReachabilityFireability,
    ]
}

fn ctl_exams() -> Vec<Examination> {
    vec![Examination::CTLCardinality, Examination::CTLFireability]
}

const SMALL_CORRECTNESS_INPUTS: &[&str] = &[
    "Sudoku-PT-AN01",
    "Sudoku-PT-BN01",
    "ResAllocation-PT-R002C002",
    "ERK-PT-000001",
    "ResAllocation-PT-R003C002",
    "Eratosthenes-PT-010",
    "TwoPhaseLocking-PT-nC00004vD",
    "ShieldRVt-PT-001A",
];

const TEMPORAL_RISK_INPUTS: &[&str] = &[
    "ParamProductionCell-PT-2",
    "Planning-PT-none",
    "PolyORBLF-PT-S02J04T06",
    "PolyORBLF-PT-S04J04T06",
    "RobotManipulation-PT-10000",
];

const REACHABILITY_AIGER_INPUTS: &[&str] = &[
    "ResAllocation-PT-R002C002",
    "ResAllocation-PT-R003C002",
    "ResAllocation-PT-R003C003",
    "ResAllocation-PT-R005C002",
    "LamportFastMutEx-PT-2",
    "Philosophers-PT-000005",
    "TokenRing-PT-005",
    "NQueens-PT-05",
    "Railroad-PT-005",
    "CircularTrains-PT-012",
    "RobotManipulation-PT-00001",
    "RobotManipulation-PT-00002",
    "StigmergyElection-PT-02a",
    "TwoPhaseLocking-PT-nC00004vD",
    "TwoPhaseLocking-PT-nC00004vN",
];

const CTL_PLAIN_60_INPUTS: &[&str] = &[
    "Angiogenesis-PT-01",
    "AutoFlight-PT-01a",
    "AutonomousCar-PT-01a",
    "AutonomousCar-PT-02a",
    "CircadianClock-PT-000001",
    "CircularTrains-PT-012",
    "CopsAndRobbers-PT-CRL005X001",
    "CopsAndRobbers-PT-KFL010N005K002X001",
    "DatabaseWithMutex-PT-02",
    "DoubleExponent-PT-001",
    "DrinkVendingMachine-PT-02",
    "ERK-PT-000001",
    "Eratosthenes-PT-010",
    "Eratosthenes-PT-020",
    "GPUForwardProgress-PT-04a",
    "HouseConstruction-PT-00002",
    "IBM319-PT-none",
    "IOTPpurchase-PT-C01M01P01D01",
    "LamportFastMutEx-PT-2",
    "NQueens-PT-05",
    "NeoElection-PT-2",
    "Philosophers-PT-000005",
    "PhilosophersDyn-PT-03",
    "QuasiCertifProtocol-PT-02",
    "Railroad-PT-005",
    "ResAllocation-PT-R002C002",
    "ResAllocation-PT-R003C002",
    "ResAllocation-PT-R003C003",
    "ResAllocation-PT-R003C005",
    "ResAllocation-PT-R005C002",
    "RingSingleMessageInMbox-PT-d0m005",
    "RobotManipulation-PT-00001",
    "RobotManipulation-PT-00002",
    "RwMutex-PT-r0010w0010",
    "RwMutex-PT-r0010w0020",
    "RwMutex-PT-r0010w0050",
    "RwMutex-PT-r0010w0100",
    "RwMutex-PT-r0010w0500",
    "RwMutex-PT-r0010w1000",
    "ServersAndClients-PT-100020",
    "SharedMemory-PT-000005",
    "ShieldIIPt-PT-001A",
    "ShieldRVs-PT-001A",
    "ShieldRVt-PT-001A",
    "ShieldRVt-PT-002A",
    "SieveSingleMsgMbox-PT-d0m04",
    "SimpleLoadBal-PT-02",
    "StigmergyCommit-PT-02a",
    "StigmergyElection-PT-02a",
    "StigmergyElection-PT-03a",
    "StigmergyElection-PT-04a",
    "Sudoku-PT-AN01",
    "Sudoku-PT-AN02",
    "Sudoku-PT-BN01",
    "TokenRing-PT-005",
    "TwoPhaseLocking-PT-nC00004vD",
    "TwoPhaseLocking-PT-nC00004vN",
    "TwoPhaseLocking-PT-nC00010vD",
    "TwoPhaseLocking-PT-nC00010vN",
    "UtilityControlRoom-PT-Z2T4N02",
];

fn bucket_table() -> BTreeMap<&'static str, BucketSpec> {
    let mut buckets: BTreeMap<&'static str, BucketSpec> = BTreeMap::new();
    buckets.insert(
        "small-correctness",
        BucketSpec {
            description: "Fast PT-only correctness smoke covering all MCC output shapes.",
            inputs: Some(SMALL_CORRECTNESS_INPUTS),
            exams: all_exams(),
        },
    );
    buckets.insert(
        "temporal-risk",
        BucketSpec {
            description: "High-risk CTL/LTL bucket called out by the MCC 2026 plan.",
            inputs: Some(TEMPORAL_RISK_INPUTS),
            exams: temporal_exams(),
        },
    );
    buckets.insert(
        "reachability-aiger",
        BucketSpec {
            description: "PT-only bounded-looking safety/reachability spread for symbolic lanes.",
            inputs: Some(REACHABILITY_AIGER_INPUTS),
            exams: reachability_exams(),
        },
    );
    buckets.insert(
        "ctl-plain-60",
        BucketSpec {
            description: "Fixed 60-input PT CTL bucket from the clean 2026-04-30 local run.",
            inputs: Some(CTL_PLAIN_60_INPUTS),
            exams: ctl_exams(),
        },
    );
    buckets.insert(
        "full-2025-soak",
        BucketSpec {
            description: "Full 2025 historical corpus; final promotion soak only.",
            inputs: None,
            exams: all_exams(),
        },
    );
    buckets
}

// ---------- Spaced-keyword normalization ----------
//
// mcc-keyword-guard: allow-spaced-mention
// 2025-archive data emits both `STATE_SPACE` (canonical) and the spaced
// `STATE SPACE` form. The set below is the legacy normalization layer.
// Direct hardcoded spaced literals follow with the allow-spaced-mention
// directive so the MCC keyword guard accepts them.
//
// New emit sites in this crate must never write the spaced form; they
// must use the constants from [`tla_petri::mcc_keywords`].

fn is_cannot_compute_token(s: &str) -> bool {
    let s = s.trim();
    if s == CANNOT_COMPUTE {
        return true;
    }
    // mcc-keyword-guard: allow-spaced-mention
    matches!(s, "D" | "DNC" | "CANNOT COMPUTE" | "CANNOTCOMPUTE")
}

fn cannot_compute_unit_set() -> BTreeSet<&'static str> {
    let mut s = BTreeSet::new();
    s.insert("D");
    s.insert("DNC");
    // mcc-keyword-guard: allow-spaced-mention
    s.insert("CANNOT COMPUTE");
    s.insert(CANNOT_COMPUTE);
    s.insert("CANNOTCOMPUTE");
    s
}

const STATE_SPACE_METRICS: [&str; 4] = [
    STATES,
    TRANSITIONS,
    MAX_TOKEN_IN_PLACE,
    MAX_TOKEN_PER_MARKING,
];

/// Normalize a StateSpace metric token, mapping spaced legacy variants
/// to the canonical underscored form.
fn normalize_state_space_metric(metric: &str) -> String {
    match metric.trim() {
        // mcc-keyword-guard: allow-spaced-mention
        "MAX TOKEN IN PLACE" => MAX_TOKEN_IN_PLACE.to_string(),
        // mcc-keyword-guard: allow-spaced-mention
        "MAX TOKEN PER MARKING" => MAX_TOKEN_PER_MARKING.to_string(),
        other => other.to_string(),
    }
}

const SINGLE_FORMULA_EXAMS: [Examination; 5] = [
    Examination::ReachabilityDeadlock,
    Examination::OneSafe,
    Examination::QuasiLiveness,
    Examination::StableMarking,
    Examination::Liveness,
];

fn is_single_formula(exam: Examination) -> bool {
    SINGLE_FORMULA_EXAMS.contains(&exam)
}

// ---------- Number normalization ----------

/// Normalize a decimal token. Strips trailing zeros / scientific form,
/// otherwise preserves the input.
fn normalize_number(token: &str) -> String {
    let t = token.trim();
    if t.is_empty() {
        return String::new();
    }
    // Try integer first.
    if let Ok(n) = t.parse::<i128>() {
        return n.to_string();
    }
    // Try float; preserve original if not a finite number.
    if let Ok(f) = t.parse::<f64>() {
        if !f.is_finite() {
            return t.to_string();
        }
        if f == f.trunc() && f.abs() < 1e18 {
            return format!("{}", f as i64);
        }
        return format_finite_decimal(f, t);
    }
    t.to_string()
}

/// Format a finite f64 without scientific notation.
fn format_finite_decimal(value: f64, fallback: &str) -> String {
    let formatted = format!("{value:.18}");
    if let Some(trimmed) = formatted.strip_suffix('0') {
        let mut s = formatted.clone();
        // Strip trailing zeros (preserve one digit after the decimal point).
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        if !s.is_empty() {
            return s;
        }
        return trimmed.to_string();
    }
    if formatted.is_empty() {
        return fallback.to_string();
    }
    formatted
}

fn normalize_expected(exam: &str, raw: &str) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return String::new();
    }
    match exam {
        "StateSpace" | "UpperBounds" => raw
            .split_whitespace()
            .map(normalize_number)
            .collect::<Vec<_>>()
            .join(" "),
        _ => raw.replace(' ', ""),
    }
}

// ---------- CSV parsing (lightweight) ----------
//
// MCC archive CSVs use `,` separators with no embedded quotes/newlines
// in the columns we care about. A full RFC 4180 parser is overkill; we
// implement the minimum needed for the archive shape.

fn read_csv_records<R: Read>(reader: R) -> Result<Vec<BTreeMap<String, String>>> {
    let mut buf = BufReader::new(reader);
    let mut header_line = String::new();
    if buf.read_line(&mut header_line)? == 0 {
        return Ok(Vec::new());
    }
    let headers: Vec<String> = parse_csv_line(header_line.trim_end_matches(['\r', '\n']))
        .into_iter()
        .map(|h| {
            let h = h.trim();
            h.trim_start_matches('#').trim().to_string()
        })
        .collect();
    let mut rows = Vec::new();
    for line in buf.lines() {
        let line = line?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }
        let fields = parse_csv_line(trimmed);
        let mut row = BTreeMap::new();
        for (i, value) in fields.iter().enumerate() {
            if let Some(key) = headers.get(i) {
                row.insert(key.clone(), value.clone());
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quote {
            if c == '"' {
                if matches!(chars.peek(), Some('"')) {
                    current.push('"');
                    chars.next();
                } else {
                    in_quote = false;
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quote = true;
        } else if c == ',' {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    fields.push(current);
    fields
}

fn parse_tsv_records<R: Read>(reader: R) -> Result<Vec<BTreeMap<String, String>>> {
    let mut buf = BufReader::new(reader);
    let mut header_line = String::new();
    if buf.read_line(&mut header_line)? == 0 {
        return Ok(Vec::new());
    }
    let headers: Vec<String> = header_line
        .trim_end_matches(['\r', '\n'])
        .split('\t')
        .map(|h| h.trim().trim_start_matches('#').trim().to_string())
        .collect();
    let mut rows = Vec::new();
    for line in buf.lines() {
        let line = line?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split('\t').collect();
        let mut row = BTreeMap::new();
        for (i, value) in fields.iter().enumerate() {
            if let Some(key) = headers.get(i) {
                row.insert(key.clone(), (*value).to_string());
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

// ---------- Result row & analysis ----------

#[derive(Debug, Clone)]
struct ResultRow {
    input_name: String,
    exam: String,
    rc: i32,
    timed_out: bool,
    format_ok: bool,
    matched: bool,
    elapsed_ms: i64,
    expected: String,
    actual: String,
    note: String,
}

#[derive(Debug, Clone)]
struct RowAnalysis {
    category: String,
    known_units: usize,
    exact_units: usize,
    historical_unknown_units: usize,
    cannot_compute_known_units: usize,
    wrong_known_units: usize,
    missing_known_units: usize,
    extra_output_units: usize,
    outcomes: Vec<String>,
    expected_units: usize,
    actual_units: usize,
}

fn parse_bool_token(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

fn split_result_units(exam: &str, value: &str) -> Vec<String> {
    let value = value.trim();
    if value.is_empty() {
        return Vec::new();
    }
    if cannot_compute_unit_set().contains(value) {
        return vec![value.to_string()];
    }
    if exam == "StateSpace" || exam == "UpperBounds" {
        return value.split_whitespace().map(String::from).collect();
    }
    value.chars().map(|c| c.to_string()).collect()
}

fn is_known_expected_unit(unit: &str) -> bool {
    let unit = unit.trim();
    if unit.is_empty() || unit.contains('?') {
        return false;
    }
    let lower = unit.to_ascii_lowercase();
    if matches!(lower.as_str(), "inf" | "+inf" | "-inf") {
        return false;
    }
    !(unit.starts_with("+Inf") || unit.starts_with("-Inf"))
}

fn analyze_result_row(row: &ResultRow) -> RowAnalysis {
    let expected_units = split_result_units(&row.exam, &row.expected);
    let historical_unknown_units = expected_units
        .iter()
        .filter(|u| !is_known_expected_unit(u))
        .count();
    let known_units = expected_units.len() - historical_unknown_units;

    if row.timed_out || row.rc != 0 || !row.format_ok {
        let category = if row.timed_out {
            "timeout"
        } else if row.rc != 0 {
            "nonzero_exit"
        } else {
            "malformed_output"
        };
        let outcomes: Vec<String> = expected_units
            .iter()
            .map(|u| {
                if is_known_expected_unit(u) {
                    "blocked".to_string()
                } else {
                    "historical_unknown".to_string()
                }
            })
            .collect();
        return RowAnalysis {
            category: category.to_string(),
            known_units,
            exact_units: 0,
            historical_unknown_units,
            cannot_compute_known_units: 0,
            wrong_known_units: 0,
            missing_known_units: 0,
            extra_output_units: 0,
            outcomes,
            expected_units: expected_units.len(),
            actual_units: 0,
        };
    }

    let mut actual_units = split_result_units(&row.exam, &row.actual);
    if actual_units.len() == 1
        && is_cannot_compute_token(&actual_units[0])
        && expected_units.len() > 1
    {
        let cc = actual_units[0].clone();
        actual_units = (0..expected_units.len()).map(|_| cc.clone()).collect();
    }

    let mut exact_units = 0;
    let mut cannot_compute_known_units = 0;
    let mut wrong_known_units = 0;
    let mut missing_known_units = 0;
    let mut outcomes = Vec::new();
    for (index, expected) in expected_units.iter().enumerate() {
        if !is_known_expected_unit(expected) {
            outcomes.push("historical_unknown".to_string());
            continue;
        }
        let Some(actual) = actual_units.get(index) else {
            missing_known_units += 1;
            outcomes.push("missing".to_string());
            continue;
        };
        if actual == expected {
            exact_units += 1;
            outcomes.push("exact".to_string());
        } else if is_cannot_compute_token(actual) {
            cannot_compute_known_units += 1;
            outcomes.push("cannot_compute".to_string());
        } else {
            wrong_known_units += 1;
            outcomes.push("wrong".to_string());
        }
    }

    let extra_output_units = actual_units.len().saturating_sub(expected_units.len());
    let category = if wrong_known_units > 0 {
        "wrong_known"
    } else if cannot_compute_known_units > 0 || missing_known_units > 0 {
        "cannot_compute_against_known"
    } else if extra_output_units > 0 {
        "extra_output"
    } else if known_units > 0 && exact_units == known_units && historical_unknown_units > 0 {
        "exact_known_with_historical_unknowns"
    } else if known_units > 0 && exact_units == known_units {
        "exact_match"
    } else if historical_unknown_units > 0 {
        "historical_unknown_only"
    } else {
        "empty_expected"
    };

    RowAnalysis {
        category: category.to_string(),
        known_units,
        exact_units,
        historical_unknown_units,
        cannot_compute_known_units,
        wrong_known_units,
        missing_known_units,
        extra_output_units,
        outcomes,
        expected_units: expected_units.len(),
        actual_units: actual_units.len(),
    }
}

// ---------- MCC output parsing ----------

/// Parse MCC stdout into a canonical `(format_ok, actual, note)` triple.
///
/// `expected_formula_ids` must be provided for multi-formula examinations
/// (CTL/LTL/Reachability/UpperBounds); when supplied, parsed ids must
/// match this set exactly.
pub fn parse_mcc_output(
    exam: &str,
    stdout: &str,
    expected_formula_ids: Option<&[String]>,
) -> (bool, String, String) {
    let lines: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return (false, String::new(), "empty stdout".to_string());
    }

    if exam == "StateSpace" {
        return parse_state_space_output(&lines);
    }

    parse_formula_output(exam, &lines, expected_formula_ids)
}

fn parse_state_space_output(lines: &[&str]) -> (bool, String, String) {
    // Tool-level cannot-compute single line plus canonical StateSpace
    // cannot-compute and legacy spaced variants.
    let cannot_compute_canonical = format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT");
    if lines.len() == 1
        && (lines[0] == CANNOT_COMPUTE
            || lines[0] == cannot_compute_canonical
            || lines[0].starts_with(&format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} "))
            // mcc-keyword-guard: allow-spaced-mention
            || lines[0].starts_with("STATE SPACE CANNOT COMPUTE TECHNIQUES "))
    {
        return (true, "DNC".to_string(), "cannot-compute".to_string());
    }
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(tech_index) = parts.iter().position(|p| *p == TECHNIQUES) else {
            return (false, String::new(), format!("missing TECHNIQUES: {line}"));
        };
        if parts.first().copied() == Some(STATE_SPACE) {
            if tech_index < 3 {
                return (
                    false,
                    String::new(),
                    format!("invalid StateSpace metric: {line}"),
                );
            }
            let metric = normalize_state_space_metric(parts[1]);
            let value = parts[tech_index - 1];
            values.insert(metric, normalize_number(value));
        } else if parts.len() >= 2
            && parts[0] == "STATE"
            // mcc-keyword-guard: allow-spaced-mention
            && parts[1] == "SPACE"
        {
            if tech_index < 4 {
                return (
                    false,
                    String::new(),
                    format!("invalid StateSpace metric: {line}"),
                );
            }
            let joined = parts[2..tech_index - 1].join(" ");
            let metric = normalize_state_space_metric(&joined);
            let value = parts[tech_index - 1];
            values.insert(metric, normalize_number(value));
        } else {
            return (
                false,
                String::new(),
                format!("invalid StateSpace line: {line}"),
            );
        }
    }
    let missing: Vec<&str> = STATE_SPACE_METRICS
        .iter()
        .copied()
        .filter(|m| !values.contains_key(*m))
        .collect();
    if !missing.is_empty() {
        return (
            false,
            String::new(),
            format!("missing StateSpace metrics: {}", missing.join(",")),
        );
    }
    let joined = STATE_SPACE_METRICS
        .iter()
        .map(|m| values[*m].clone())
        .collect::<Vec<_>>()
        .join(" ");
    (true, joined, "ok".to_string())
}

fn parse_formula_line(line: &str) -> Option<(String, String)> {
    // Format: FORMULA <id> <verdict-tokens...> TECHNIQUES <techniques...>
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 5 || tokens[0] != FORMULA {
        return None;
    }
    let tech_index = tokens.iter().position(|t| *t == TECHNIQUES)?;
    if tech_index < 3 {
        return None;
    }
    let id = tokens[1].to_string();
    let verdict = tokens[2..tech_index].join(" ");
    Some((id, verdict))
}

fn parse_formula_output(
    exam: &str,
    lines: &[&str],
    expected_formula_ids: Option<&[String]>,
) -> (bool, String, String) {
    let mut parsed: Vec<(String, String)> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for line in lines {
        let Some((formula_id, value)) = parse_formula_line(line) else {
            return (
                false,
                String::new(),
                format!("invalid FORMULA line: {line}"),
            );
        };
        if !seen.insert(formula_id.clone()) {
            return (
                false,
                String::new(),
                format!("duplicate FORMULA id: {formula_id}"),
            );
        }
        // mcc-keyword-guard: allow-spaced-mention
        let valid = matches!(value.as_str(), "TRUE" | "FALSE" | "CANNOT COMPUTE")
            || value == CANNOT_COMPUTE
            || value.chars().all(|c| c.is_ascii_digit());
        if !valid {
            return (
                false,
                String::new(),
                format!("invalid FORMULA value: {value}"),
            );
        }
        parsed.push((formula_id, value));
    }

    let Ok(exam_kind) = parse_exam_name(exam) else {
        return (false, String::new(), format!("unknown examination: {exam}"));
    };

    if is_single_formula(exam_kind) {
        if parsed.len() != 1 {
            return (
                false,
                String::new(),
                format!("expected 1 FORMULA line, saw {}", parsed.len()),
            );
        }
        let (id, value) = &parsed[0];
        if id != exam {
            return (
                false,
                String::new(),
                format!("unexpected FORMULA id for {exam}: {id}"),
            );
        }
        return match value.as_str() {
            "TRUE" => (true, "T".to_string(), "ok".to_string()),
            "FALSE" => (true, "F".to_string(), "ok".to_string()),
            _ => (true, "DNC".to_string(), "cannot-compute".to_string()),
        };
    }

    let final_order: Vec<String> = if let Some(expected) = expected_formula_ids {
        let unique_count = expected.iter().collect::<BTreeSet<_>>().len();
        if unique_count != expected.len() {
            return (
                false,
                String::new(),
                "duplicate expected FORMULA ids".to_string(),
            );
        }
        let expected_set: BTreeSet<&String> = expected.iter().collect();
        let actual_set: BTreeSet<&String> = parsed.iter().map(|(id, _)| id).collect();
        if expected_set != actual_set {
            return (
                false,
                String::new(),
                formula_id_set_mismatch_note(expected, &actual_set),
            );
        }
        let mut ordered = expected.to_vec();
        ordered.sort();
        ordered
    } else {
        let mut ids: Vec<String> = parsed.iter().map(|(id, _)| id.clone()).collect();
        ids.sort();
        ids
    };

    let values_by_id: BTreeMap<String, String> = parsed
        .iter()
        .map(|(id, v)| (id.clone(), v.clone()))
        .collect();

    if exam == "UpperBounds" {
        let values: Vec<String> = final_order
            .iter()
            .map(|id| {
                let v = &values_by_id[id];
                if v.chars().all(|c| c.is_ascii_digit()) {
                    v.clone()
                } else {
                    "DNC".to_string()
                }
            })
            .collect();
        return (true, values.join(" "), "ok".to_string());
    }

    let mut out = String::new();
    for id in &final_order {
        match values_by_id[id].as_str() {
            "TRUE" => out.push('T'),
            "FALSE" => out.push('F'),
            _ => out.push('D'),
        }
    }
    (true, out, "ok".to_string())
}

fn formula_id_set_mismatch_note(
    expected_formula_ids: &[String],
    actual_formula_ids: &BTreeSet<&String>,
) -> String {
    let expected_set: BTreeSet<&String> = expected_formula_ids.iter().collect();
    let missing: Vec<&String> = expected_formula_ids
        .iter()
        .filter(|id| !actual_formula_ids.contains(id))
        .collect();
    let mut extra: Vec<&String> = actual_formula_ids
        .iter()
        .filter(|id| !expected_set.contains(*id))
        .copied()
        .collect();
    extra.sort();
    let mut details = Vec::new();
    if !missing.is_empty() {
        let s: Vec<String> = missing.iter().map(|s| s.to_string()).collect();
        details.push(format!("missing FORMULA ids: {}", s.join(",")));
    }
    if !extra.is_empty() {
        let s: Vec<String> = extra.iter().map(|s| s.to_string()).collect();
        details.push(format!("unexpected FORMULA ids: {}", s.join(",")));
    }
    if details.is_empty() {
        "FORMULA id mismatch".to_string()
    } else {
        details.join("; ")
    }
}

// ---------- Archive history loading ----------

#[derive(Debug, Clone)]
struct RootPaths {
    inputs_archive: PathBuf,
    inputs_dir: PathBuf,
    raw_history: PathBuf,
    case_list: PathBuf,
}

fn root_paths(root: &Path) -> RootPaths {
    RootPaths {
        inputs_archive: root.join("INPUTS-2025.tar.gz"),
        inputs_dir: root.join("inputs").join("INPUTS-2025"),
        raw_history: root
            .join("results")
            .join("extracted")
            .join("raw-result-analysis.csv"),
        case_list: root.join("run-all").join("cases.tsv"),
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_history(
    raw_history: &Path,
) -> Result<(
    Vec<BTreeMap<String, String>>,
    BTreeMap<(String, String), String>,
)> {
    let file =
        File::open(raw_history).with_context(|| format!("open {}", raw_history.display()))?;
    let rows = read_csv_records(file)?;
    let mut expected: BTreeMap<(String, String), String> = BTreeMap::new();
    for row in &rows {
        let input_name = row
            .get("Input")
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let exam = row
            .get("Examination")
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let raw = row.get("estimated result").cloned().unwrap_or_default();
        let normalized = normalize_expected(&exam, &raw);
        let key = (input_name, exam);
        if let Some(prev) = expected.get(&key) {
            if prev != &normalized {
                bail!(
                    "conflicting historical estimates for {} {}: {prev:?} vs {normalized:?}",
                    key.0,
                    key.1
                );
            }
        }
        expected.insert(key, normalized);
    }
    Ok((rows, expected))
}

fn history_summary(root: &Path, expected_inputs_sha256: Option<&str>) -> Result<Value> {
    let paths = root_paths(root);
    let (rows, _) = read_history(&paths.raw_history)?;
    let inputs: BTreeSet<String> = rows
        .iter()
        .filter_map(|r| r.get("Input").map(|s| s.trim().to_string()))
        .collect();
    let cases: BTreeSet<(String, String)> = rows
        .iter()
        .map(|r| {
            (
                r.get("Input")
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default(),
                r.get("Examination")
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default(),
            )
        })
        .collect();
    let tools: BTreeSet<String> = rows
        .iter()
        .filter_map(|r| r.get("tool").map(|s| s.trim().to_string()))
        .collect();
    let mut exams_count: BTreeMap<String, usize> = BTreeMap::new();
    for ex in exams_in_report_order() {
        exams_count.insert(ex.as_str().to_string(), 0);
    }
    for (_, exam) in &cases {
        *exams_count.entry(exam.clone()).or_insert(0) += 1;
    }
    let tarball_count = if paths.inputs_dir.is_dir() {
        fs::read_dir(&paths.inputs_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("tgz"))
            })
            .count()
    } else {
        0
    };
    let actual_inputs_sha256 = sha256_file(&paths.inputs_archive)?;
    let expected_inputs_sha256 = expected_inputs_sha256
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if let Some(expected) = expected_inputs_sha256 {
        if actual_inputs_sha256 != expected {
            bail!(
                "INPUTS-2025.tar.gz SHA-256 mismatch: expected {expected}, got {actual_inputs_sha256}"
            );
        }
    }
    Ok(json!({
        "root": root.display().to_string(),
        "inputs_archive": paths.inputs_archive.display().to_string(),
        "inputs_archive_sha256": actual_inputs_sha256,
        "inputs_archive_sha256_expected": expected_inputs_sha256,
        "inputs_archive_sha256_verified": expected_inputs_sha256.is_some(),
        "input_tarballs": tarball_count,
        "raw_rows": rows.len(),
        "unique_inputs": inputs.len(),
        "unique_cases": cases.len(),
        "tools": tools.iter().collect::<Vec<_>>(),
        "exams": exams_count,
    }))
}

fn write_case_list(root: &Path) -> Result<PathBuf> {
    let paths = root_paths(root);
    let (_, expected) = read_history(&paths.raw_history)?;
    let mut keys: Vec<(String, String)> = expected.keys().cloned().collect();
    keys.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| exam_order(&a.1).cmp(&exam_order(&b.1)))
            .then_with(|| a.1.cmp(&b.1))
    });
    if let Some(parent) = paths.case_list.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(&paths.case_list)?;
    for key in &keys {
        let value = &expected[key];
        writeln!(file, "{}\t{}\t{}", key.0, key.1, value)?;
    }
    Ok(paths.case_list)
}

fn load_cases(root: &Path) -> Result<Vec<(String, String, String)>> {
    let paths = root_paths(root);
    if !paths.case_list.exists() {
        write_case_list(root)?;
    }
    let mut cases = Vec::new();
    let file = File::open(&paths.case_list)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        if parts.len() < 3 {
            bail!("invalid case-list line: {line}");
        }
        cases.push((
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2].to_string(),
        ));
    }
    Ok(cases)
}

fn state_size_index(root: &Path) -> Result<BTreeMap<String, i64>> {
    let (_, expected) = read_history(&root_paths(root).raw_history)?;
    let mut sizes = BTreeMap::new();
    for ((input_name, exam), value) in &expected {
        if exam != "StateSpace" {
            continue;
        }
        let first = value.split_whitespace().next().unwrap_or("");
        if let Ok(n) = first.parse::<i64>() {
            sizes.insert(input_name.clone(), n);
        }
    }
    Ok(sizes)
}

// ---------- Property XML formula-id extraction ----------

fn load_expected_formula_ids(model_dir: &Path, exam: &str) -> Result<Option<Vec<String>>> {
    let exam_kind = parse_exam_name(exam)?;
    if exam_kind == Examination::StateSpace {
        return Ok(None);
    }
    if is_single_formula(exam_kind) {
        return Ok(Some(vec![exam.to_string()]));
    }
    let xml_path = model_dir.join(format!("{exam}.xml"));
    if !xml_path.exists() {
        bail!("missing property XML for {exam}: {}", xml_path.display());
    }
    let content =
        fs::read_to_string(&xml_path).with_context(|| format!("read {}", xml_path.display()))?;
    let doc = roxmltree::Document::parse(&content).map_err(|e| {
        anyhow!(
            "invalid property XML for {exam}: {}: {e}",
            xml_path.display()
        )
    })?;
    let mut formula_ids: Vec<String> = Vec::new();
    for node in doc.descendants() {
        if node.tag_name().name() != "property" {
            continue;
        }
        let ids: Vec<String> = node
            .children()
            .filter(|c| c.is_element() && c.tag_name().name() == "id")
            .map(|c| c.text().unwrap_or("").trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if ids.len() != 1 {
            bail!(
                "property without exactly one non-empty <id> in {}",
                xml_path.display()
            );
        }
        formula_ids.push(ids[0].clone());
    }
    if formula_ids.is_empty() {
        bail!("no property <id> values found in {}", xml_path.display());
    }
    let mut seen = BTreeSet::new();
    let mut dups: Vec<String> = Vec::new();
    for id in &formula_ids {
        if !seen.insert(id.clone()) && !dups.contains(id) {
            dups.push(id.clone());
        }
    }
    if !dups.is_empty() {
        bail!(
            "duplicate property <id> values in {}: {}",
            xml_path.display(),
            dups.join(",")
        );
    }
    Ok(Some(formula_ids))
}

// ---------- Score / blocker / comparison reports ----------

fn percentile_nearest(values: &mut [i64], percentile: usize) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    values.sort();
    let n = values.len();
    let idx = ((percentile as f64 / 100.0) * n as f64).ceil() as usize;
    let idx = idx.saturating_sub(1).min(n - 1);
    Some(values[idx])
}

#[derive(Default, Debug, Clone)]
struct ScoreBucket {
    cases: i64,
    format_ok: i64,
    matched_cases: i64,
    mismatched_cases: i64,
    malformed_output: i64,
    nonzero_exits: i64,
    timeouts: i64,
    known_units: i64,
    exact_units: i64,
    historical_unknown_units: i64,
    cannot_compute_known_units: i64,
    wrong_known_units: i64,
    missing_known_units: i64,
    extra_output_units: i64,
    runtime_ms: Vec<i64>,
    categories: BTreeMap<String, i64>,
}

impl ScoreBucket {
    fn add(&mut self, row: &ResultRow, analysis: &RowAnalysis) {
        self.cases += 1;
        self.format_ok += i64::from(row.format_ok);
        self.matched_cases += i64::from(row.matched);
        self.mismatched_cases += i64::from(!row.matched);
        self.malformed_output += i64::from(!row.format_ok);
        self.nonzero_exits += i64::from(row.rc != 0);
        self.timeouts += i64::from(row.timed_out);
        self.known_units += analysis.known_units as i64;
        self.exact_units += analysis.exact_units as i64;
        self.historical_unknown_units += analysis.historical_unknown_units as i64;
        self.cannot_compute_known_units += analysis.cannot_compute_known_units as i64;
        self.wrong_known_units += analysis.wrong_known_units as i64;
        self.missing_known_units += analysis.missing_known_units as i64;
        self.extra_output_units += analysis.extra_output_units as i64;
        self.runtime_ms.push(row.elapsed_ms);
        *self
            .categories
            .entry(analysis.category.clone())
            .or_insert(0) += 1;
    }

    fn finalize(mut self) -> Value {
        let times = std::mem::take(&mut self.runtime_ms);
        let p50 = percentile_nearest(&mut times.clone(), 50);
        let p95 = percentile_nearest(&mut times.clone(), 95);
        let max = times.iter().copied().max();
        let mut categories = Map::new();
        for (k, v) in self.categories {
            categories.insert(k, Value::from(v));
        }
        json!({
            "cases": self.cases,
            "format_ok": self.format_ok,
            "matched_cases": self.matched_cases,
            "mismatched_cases": self.mismatched_cases,
            "malformed_output": self.malformed_output,
            "nonzero_exits": self.nonzero_exits,
            "timeouts": self.timeouts,
            "known_units": self.known_units,
            "exact_units": self.exact_units,
            "historical_unknown_units": self.historical_unknown_units,
            "cannot_compute_known_units": self.cannot_compute_known_units,
            "wrong_known_units": self.wrong_known_units,
            "missing_known_units": self.missing_known_units,
            "extra_output_units": self.extra_output_units,
            "runtime_p50_ms": p50,
            "runtime_p95_ms": p95,
            "runtime_max_ms": max,
            "categories": categories,
        })
    }
}

fn read_result_rows(path: &Path) -> Result<Vec<ResultRow>> {
    let results_path = resolve_results_path(path)?;
    let file = File::open(&results_path)?;
    let raw = parse_tsv_records(file)?;
    let mut rows = Vec::new();
    for r in raw {
        let rc: i32 = r.get("rc").and_then(|s| s.parse().ok()).unwrap_or(0);
        let elapsed_ms: i64 = r
            .get("elapsed_ms")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        rows.push(ResultRow {
            input_name: r.get("input").cloned().unwrap_or_default(),
            exam: r.get("examination").cloned().unwrap_or_default(),
            rc,
            timed_out: parse_bool_token(r.get("timeout").map(String::as_str).unwrap_or("0")),
            format_ok: parse_bool_token(r.get("format_ok").map(String::as_str).unwrap_or("0")),
            matched: parse_bool_token(r.get("match").map(String::as_str).unwrap_or("0")),
            elapsed_ms,
            expected: r.get("expected").cloned().unwrap_or_default(),
            actual: r.get("actual").cloned().unwrap_or_default(),
            note: r.get("note").cloned().unwrap_or_default(),
        });
    }
    Ok(rows)
}

fn resolve_results_path(path: &Path) -> Result<PathBuf> {
    let target = if path.is_dir() {
        path.join("results.tsv")
    } else {
        path.to_path_buf()
    };
    if !target.exists() {
        bail!("results.tsv does not exist: {}", target.display());
    }
    Ok(target)
}

fn build_score_report(rows: &[ResultRow]) -> (Value, Vec<Map<String, Value>>) {
    let mut totals = ScoreBucket::default();
    let mut per_exam: BTreeMap<String, ScoreBucket> = BTreeMap::new();
    let mut case_rows: Vec<Map<String, Value>> = Vec::new();
    for row in rows {
        let analysis = analyze_result_row(row);
        totals.add(row, &analysis);
        per_exam
            .entry(row.exam.clone())
            .or_default()
            .add(row, &analysis);
        let mut case_row = Map::new();
        case_row.insert("input".into(), Value::from(row.input_name.clone()));
        case_row.insert("examination".into(), Value::from(row.exam.clone()));
        case_row.insert("category".into(), Value::from(analysis.category.clone()));
        case_row.insert("rc".into(), Value::from(row.rc));
        case_row.insert("timeout".into(), Value::from(i32::from(row.timed_out)));
        case_row.insert("format_ok".into(), Value::from(i32::from(row.format_ok)));
        case_row.insert("match".into(), Value::from(i32::from(row.matched)));
        case_row.insert("elapsed_ms".into(), Value::from(row.elapsed_ms));
        case_row.insert("known_units".into(), Value::from(analysis.known_units));
        case_row.insert("exact_units".into(), Value::from(analysis.exact_units));
        case_row.insert(
            "historical_unknown_units".into(),
            Value::from(analysis.historical_unknown_units),
        );
        case_row.insert(
            "cannot_compute_known_units".into(),
            Value::from(analysis.cannot_compute_known_units),
        );
        case_row.insert(
            "wrong_known_units".into(),
            Value::from(analysis.wrong_known_units),
        );
        case_row.insert(
            "missing_known_units".into(),
            Value::from(analysis.missing_known_units),
        );
        case_row.insert(
            "extra_output_units".into(),
            Value::from(analysis.extra_output_units),
        );
        case_row.insert("expected".into(), Value::from(row.expected.clone()));
        case_row.insert("actual".into(), Value::from(row.actual.clone()));
        case_row.insert("note".into(), Value::from(row.note.clone()));
        case_rows.push(case_row);
    }
    let mut per_exam_sorted: Vec<(String, ScoreBucket)> = per_exam.into_iter().collect();
    per_exam_sorted.sort_by(|a, b| {
        exam_order(&a.0)
            .cmp(&exam_order(&b.0))
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut per_exam_map = Map::new();
    for (exam, bucket) in per_exam_sorted {
        per_exam_map.insert(exam, bucket.finalize());
    }
    let report = json!({
        "totals": totals.finalize(),
        "per_examination": per_exam_map,
    });
    (report, case_rows)
}

fn tsv_cell(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn write_tsv(path: &Path, rows: &[Map<String, Value>], fields: &[&str]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{}", fields.join("\t"))?;
    for row in rows {
        let cols: Vec<String> = fields
            .iter()
            .map(|f| tsv_cell(row.get(*f).unwrap_or(&Value::Null)))
            .collect();
        writeln!(file, "{}", cols.join("\t"))?;
    }
    Ok(())
}

fn lost_points_for_analysis(a: &RowAnalysis) -> i64 {
    (a.known_units as i64 - a.exact_units as i64).max(0)
}

fn default_blocker_for_row(row: &ResultRow, a: &RowAnalysis) -> &'static str {
    if row.timed_out {
        return "timeout";
    }
    if row.rc != 0 {
        return "nonzero_exit";
    }
    if !row.format_ok {
        return "malformed_output";
    }
    if a.wrong_known_units > 0 {
        return "wrong_answer";
    }
    if a.missing_known_units > 0 {
        return "missing_output";
    }
    if a.cannot_compute_known_units > 0 {
        return "cannot_compute";
    }
    if a.extra_output_units > 0 {
        return "extra_output";
    }
    // Fallback: examine category (string-equal); allocate to leak-proof &'static via match.
    match a.category.as_str() {
        "timeout" => "timeout",
        "nonzero_exit" => "nonzero_exit",
        "malformed_output" => "malformed_output",
        "wrong_known" => "wrong_known",
        "cannot_compute_against_known" => "cannot_compute_against_known",
        "extra_output" => "extra_output",
        "exact_known_with_historical_unknowns" => "exact_known_with_historical_unknowns",
        "exact_match" => "exact_match",
        "historical_unknown_only" => "historical_unknown_only",
        "empty_expected" => "empty_expected",
        _ => "unknown",
    }
}

fn default_reason_for_row(row: &ResultRow, a: &RowAnalysis) -> String {
    if row.timed_out {
        return "timeout".to_string();
    }
    if row.rc != 0 {
        return "nonzero_exit".to_string();
    }
    if !row.format_ok {
        return if row.note.is_empty() {
            "malformed_output".to_string()
        } else {
            row.note.clone()
        };
    }
    if a.wrong_known_units > 0 {
        return "wrong_known_answer".to_string();
    }
    if a.missing_known_units > 0 {
        return "missing_known_output".to_string();
    }
    if a.cannot_compute_known_units > 0 {
        return "cannot_compute_known_answer".to_string();
    }
    if a.extra_output_units > 0 {
        return "extra_output".to_string();
    }
    a.category.clone()
}

// ---------- Route-cause sidecar (best effort; degrades gracefully) ----------

#[derive(Debug, Default, Clone)]
struct RouteCause {
    input_name: String,
    exam: String,
    route_status: String,
    selected_lane: String,
    next_answer_lane: String,
    fallback_lane: String,
    lane_family: String,
    route_reason_code: String,
    blocker_piece: String,
    blocker_gate: String,
    blocker_reason_code: String,
    blocker_action_code: String,
    owner_project: String,
    owner_primitive: String,
}

fn evidence_containers(record: &Value) -> Vec<&Value> {
    let mut containers = vec![record];
    for key in [
        "report",
        "capability_report",
        "backend_capability_report",
        "backendCapabilityReport",
        "capability",
    ] {
        if let Some(v) = record.get(key) {
            if v.is_object() {
                containers.push(v);
            }
        }
    }
    containers
}

fn first_text_field(containers: &[&Value], keys: &[&str]) -> String {
    for c in containers {
        for k in keys {
            if let Some(v) = c.get(*k) {
                let s = match v {
                    Value::String(s) => s.trim().to_string(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
                if !s.is_empty() {
                    return s;
                }
            }
        }
    }
    String::new()
}

fn evidence_lines_from_record(record: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    for c in evidence_containers(record) {
        if let Some(v) = c.get("evidence") {
            match v {
                Value::Array(arr) => {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            if !s.trim().is_empty() {
                                lines.push(s.to_string());
                            }
                        }
                    }
                }
                Value::String(s) if !s.trim().is_empty() => {
                    lines.push(s.clone());
                }
                _ => {}
            }
        }
    }
    lines
}

fn parse_evidence_line(line: &str) -> BTreeMap<String, String> {
    let tokens = shell_split(line);
    let mut data = BTreeMap::new();
    if tokens.len() < 2 {
        return data;
    }
    data.insert("scope".to_string(), tokens[0].clone());
    data.insert("component".to_string(), tokens[1].clone());
    for tok in tokens.iter().skip(2) {
        if let Some((k, v)) = tok.split_once('=') {
            if !k.is_empty() {
                data.insert(k.to_string(), v.to_string());
            }
        }
    }
    data
}

/// Minimal shell-like tokenizer: split on whitespace, honor double/single quotes.
fn shell_split(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut had_token = false;
    for c in input.chars() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            } else {
                current.push(c);
            }
            had_token = true;
        } else if c == '"' || c == '\'' {
            quote = Some(c);
            had_token = true;
        } else if c.is_whitespace() {
            if had_token {
                out.push(std::mem::take(&mut current));
                had_token = false;
            }
        } else {
            current.push(c);
            had_token = true;
        }
    }
    if had_token {
        out.push(current);
    }
    out
}

fn parse_int_token(value: Option<&String>, default: i64) -> i64 {
    value.and_then(|s| s.parse::<i64>().ok()).unwrap_or(default)
}

fn parse_bool_evidence(s: Option<&String>) -> bool {
    s.is_some_and(|s| parse_bool_token(s))
}

fn selected_blocker_action(rows: &[BTreeMap<String, String>]) -> BTreeMap<String, String> {
    let actions: Vec<&BTreeMap<String, String>> = rows
        .iter()
        .filter(|r| r.get("scope").map(String::as_str) == Some("MCC"))
        .filter(|r| r.get("component").map(String::as_str) == Some("blocker_action"))
        .collect();
    let selected: Vec<&BTreeMap<String, String>> = actions
        .iter()
        .copied()
        .filter(|r| parse_bool_evidence(r.get("selected")))
        .collect();
    let pool: &[&BTreeMap<String, String>] = if !selected.is_empty() {
        &selected
    } else {
        &actions
    };
    let mut sorted: Vec<&BTreeMap<String, String>> = pool.to_vec();
    sorted.sort_by(|a, b| {
        let pa = parse_int_token(a.get("priority_rank"), 999);
        let pb = parse_int_token(b.get("priority_rank"), 999);
        pa.cmp(&pb).then_with(|| {
            a.get("blocker_piece")
                .cloned()
                .unwrap_or_default()
                .cmp(&b.get("blocker_piece").cloned().unwrap_or_default())
        })
    });
    sorted.first().map(|r| (*r).clone()).unwrap_or_default()
}

fn route_cause_from_record(record: &Value) -> Option<RouteCause> {
    let containers = evidence_containers(record);
    let input_name = first_text_field(
        &containers,
        &[
            "input",
            "input_name",
            "model",
            "model_name",
            "benchmark",
            "case",
            "spec",
        ],
    );
    let exam = first_text_field(&containers, &["examination", "exam", "property"]);
    let evidence_rows: Vec<BTreeMap<String, String>> = evidence_lines_from_record(record)
        .iter()
        .map(|line| parse_evidence_line(line))
        .filter(|m| !m.is_empty())
        .collect();
    let selector = evidence_rows
        .iter()
        .rfind(|r| {
            r.get("scope").map(String::as_str) == Some("MCC")
                && r.get("component").map(String::as_str) == Some("production_selector_decision")
        })
        .cloned()
        .unwrap_or_default();
    let symbolic = evidence_rows
        .iter()
        .rfind(|r| {
            r.get("scope").map(String::as_str) == Some("MCC")
                && r.get("component").map(String::as_str) == Some("symbolic_execution")
        })
        .cloned()
        .unwrap_or_default();
    let ltl_route = evidence_rows
        .iter()
        .rfind(|r| {
            r.get("scope").map(String::as_str) == Some("MCC")
                && matches!(
                    r.get("component").map(String::as_str),
                    Some("ltl_answer_lane_summary") | Some("ltl_route_admission")
                )
        })
        .cloned()
        .unwrap_or_default();
    let blocker = selected_blocker_action(&evidence_rows);
    let admission = evidence_rows
        .iter()
        .rfind(|r| r.get("component").map(String::as_str) == Some("trust_cg_admission_blocker"))
        .cloned()
        .unwrap_or_default();

    let pick = |maps: &[&BTreeMap<String, String>], keys: &[&str]| -> String {
        for m in maps {
            for k in keys {
                if let Some(v) = m.get(*k) {
                    if !v.is_empty() {
                        return v.clone();
                    }
                }
            }
        }
        String::new()
    };

    let selected_lane = pick(
        &[&selector, &ltl_route, &symbolic],
        &[
            "selected_lane",
            "preferred_backend_code",
            "preferred_backend",
        ],
    );
    let mut blocker_piece = blocker.get("blocker_piece").cloned().unwrap_or_default();
    if blocker_piece.is_empty() {
        if let Some(lp) = ltl_route.get("blocker_piece") {
            if lp != "none" {
                blocker_piece = lp.clone();
            }
        }
    }
    if blocker_piece.is_empty() && !admission.is_empty() {
        blocker_piece = admission.get("component").cloned().unwrap_or_default();
    }
    if input_name.is_empty()
        && exam.is_empty()
        && selected_lane.is_empty()
        && blocker_piece.is_empty()
        && selector.is_empty()
        && ltl_route.is_empty()
        && symbolic.is_empty()
        && admission.is_empty()
    {
        return None;
    }
    let route_status = pick(
        &[&selector, &ltl_route, &symbolic],
        &["selector_status", "status", "route_status", "status_code"],
    );
    let next_answer_lane = pick(&[&selector, &ltl_route, &blocker], &["next_answer_lane"]);
    let fallback_lane = pick(&[&selector, &ltl_route], &["fallback_lane"]);
    let lane_family = blocker.get("lane_family").cloned().unwrap_or_default();
    let route_reason_code = pick(
        &[&selector, &ltl_route, &symbolic],
        &[
            "selected_reason_code",
            "reason_code",
            "cannot_compute_reason_code",
            "status_code",
        ],
    );
    let blocker_gate = pick(&[&blocker, &ltl_route], &["blocker_gate"]);
    let blocker_reason_code = pick(
        &[&blocker, &ltl_route, &selector, &admission],
        &[
            "reason_code",
            "cannot_compute_reason_code",
            "trust_cg_native_jit_reason_code",
            "fallback_reason_code",
            "rejection_code",
        ],
    );

    Some(RouteCause {
        input_name,
        exam,
        route_status,
        selected_lane,
        next_answer_lane,
        fallback_lane,
        lane_family,
        route_reason_code,
        blocker_piece,
        blocker_gate,
        blocker_reason_code,
        blocker_action_code: blocker.get("action_code").cloned().unwrap_or_default(),
        owner_project: blocker.get("owner_project").cloned().unwrap_or_default(),
        owner_primitive: blocker.get("owner_primitive").cloned().unwrap_or_default(),
    })
}

type RouteCauseIndex = BTreeMap<(String, String), RouteCause>;

fn load_route_cause_index(path: Option<&Path>) -> RouteCauseIndex {
    let mut index = BTreeMap::new();
    let Some(p) = path else { return index };
    let Ok(file) = File::open(p) else {
        return index;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if !record.is_object() {
            continue;
        }
        if let Some(cause) = route_cause_from_record(&record) {
            if !cause.input_name.is_empty() {
                index.insert((cause.input_name.clone(), cause.exam.clone()), cause);
            }
        }
    }
    index
}

fn route_cause_for_row(index: Option<&RouteCauseIndex>, row: &ResultRow) -> RouteCause {
    let Some(idx) = index else {
        return RouteCause {
            input_name: row.input_name.clone(),
            exam: row.exam.clone(),
            ..Default::default()
        };
    };
    if let Some(c) = idx.get(&(row.input_name.clone(), row.exam.clone())) {
        return c.clone();
    }
    if let Some(c) = idx.get(&(row.input_name.clone(), String::new())) {
        return c.clone();
    }
    RouteCause {
        input_name: row.input_name.clone(),
        exam: row.exam.clone(),
        ..Default::default()
    }
}

// ---------- Blocker-loss report ----------

fn case_blocker_row(
    row: &ResultRow,
    analysis: &RowAnalysis,
    cause: &RouteCause,
) -> Map<String, Value> {
    let blocker = if cause.blocker_piece.is_empty() {
        default_blocker_for_row(row, analysis).to_string()
    } else {
        cause.blocker_piece.clone()
    };
    let reason_code = if !cause.blocker_reason_code.is_empty() {
        cause.blocker_reason_code.clone()
    } else if !cause.route_reason_code.is_empty() {
        cause.route_reason_code.clone()
    } else {
        default_reason_for_row(row, analysis)
    };
    let mut map = Map::new();
    map.insert("input".into(), Value::from(row.input_name.clone()));
    map.insert("examination".into(), Value::from(row.exam.clone()));
    map.insert("category".into(), Value::from(analysis.category.clone()));
    map.insert("blocker".into(), Value::from(blocker));
    map.insert(
        "lost_points".into(),
        Value::from(lost_points_for_analysis(analysis)),
    );
    map.insert("known_units".into(), Value::from(analysis.known_units));
    map.insert("exact_units".into(), Value::from(analysis.exact_units));
    map.insert(
        "cannot_compute_known_units".into(),
        Value::from(analysis.cannot_compute_known_units),
    );
    map.insert(
        "missing_known_units".into(),
        Value::from(analysis.missing_known_units),
    );
    map.insert(
        "wrong_known_units".into(),
        Value::from(analysis.wrong_known_units),
    );
    map.insert(
        "extra_output_units".into(),
        Value::from(analysis.extra_output_units),
    );
    map.insert("timeout".into(), Value::from(i32::from(row.timed_out)));
    map.insert("nonzero_exit".into(), Value::from(i32::from(row.rc != 0)));
    map.insert(
        "malformed_output".into(),
        Value::from(i32::from(!row.format_ok)),
    );
    map.insert(
        "selected_lane".into(),
        Value::from(cause.selected_lane.clone()),
    );
    map.insert(
        "next_answer_lane".into(),
        Value::from(cause.next_answer_lane.clone()),
    );
    map.insert(
        "fallback_lane".into(),
        Value::from(cause.fallback_lane.clone()),
    );
    map.insert("lane_family".into(), Value::from(cause.lane_family.clone()));
    map.insert("reason_code".into(), Value::from(reason_code));
    map.insert(
        "route_status".into(),
        Value::from(cause.route_status.clone()),
    );
    map.insert(
        "blocker_gate".into(),
        Value::from(cause.blocker_gate.clone()),
    );
    map.insert(
        "action_code".into(),
        Value::from(cause.blocker_action_code.clone()),
    );
    map.insert(
        "owner_project".into(),
        Value::from(cause.owner_project.clone()),
    );
    map.insert(
        "owner_primitive".into(),
        Value::from(cause.owner_primitive.clone()),
    );
    map.insert("elapsed_ms".into(), Value::from(row.elapsed_ms));
    map.insert("expected".into(), Value::from(row.expected.clone()));
    map.insert("actual".into(), Value::from(row.actual.clone()));
    map.insert("note".into(), Value::from(row.note.clone()));
    map
}

fn build_blocker_loss_report(
    rows: &[ResultRow],
    route_cause_index: Option<&RouteCauseIndex>,
) -> (Value, Vec<Map<String, Value>>) {
    let mut case_rows = Vec::new();
    #[derive(Default)]
    struct Agg {
        blocker: String,
        lane_family: String,
        selected_lane: String,
        next_answer_lane: String,
        fallback_lane: String,
        reason_code: String,
        route_status: String,
        blocker_gate: String,
        action_code: String,
        owner_project: String,
        owner_primitive: String,
        cases: i64,
        lost_points: i64,
        known_units: i64,
        exact_units: i64,
        cannot_compute_known_units: i64,
        missing_known_units: i64,
        wrong_known_units: i64,
        extra_output_units: i64,
        timeouts: i64,
        nonzero_exits: i64,
        malformed_outputs: i64,
        affected_examinations: BTreeSet<String>,
        examples: Vec<String>,
    }
    type Key = (String, String, String, String, String, String);
    let mut aggregates: BTreeMap<Key, Agg> = BTreeMap::new();
    for row in rows {
        let analysis = analyze_result_row(row);
        let cause = route_cause_for_row(route_cause_index, row);
        let cr = case_blocker_row(row, &analysis, &cause);
        let has_loss = cr.get("lost_points").and_then(Value::as_i64).unwrap_or(0) > 0
            || cr
                .get("wrong_known_units")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                > 0
            || cr
                .get("extra_output_units")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                > 0
            || cr.get("timeout").and_then(Value::as_i64).unwrap_or(0) > 0
            || cr.get("nonzero_exit").and_then(Value::as_i64).unwrap_or(0) > 0
            || cr
                .get("malformed_output")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                > 0;
        if !has_loss {
            continue;
        }
        let key: Key = (
            cr.get("blocker")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            cr.get("lane_family")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            cr.get("selected_lane")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            cr.get("next_answer_lane")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            cr.get("fallback_lane")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            cr.get("reason_code")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        );
        let agg = aggregates.entry(key.clone()).or_insert_with(|| Agg {
            blocker: key.0.clone(),
            lane_family: key.1.clone(),
            selected_lane: key.2.clone(),
            next_answer_lane: key.3.clone(),
            fallback_lane: key.4.clone(),
            reason_code: key.5.clone(),
            route_status: cr
                .get("route_status")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            blocker_gate: cr
                .get("blocker_gate")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            action_code: cr
                .get("action_code")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            owner_project: cr
                .get("owner_project")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            owner_primitive: cr
                .get("owner_primitive")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            ..Default::default()
        });
        agg.cases += 1;
        let int = |k: &str| cr.get(k).and_then(Value::as_i64).unwrap_or(0);
        agg.lost_points += int("lost_points");
        agg.known_units += int("known_units");
        agg.exact_units += int("exact_units");
        agg.cannot_compute_known_units += int("cannot_compute_known_units");
        agg.missing_known_units += int("missing_known_units");
        agg.wrong_known_units += int("wrong_known_units");
        agg.extra_output_units += int("extra_output_units");
        agg.timeouts += int("timeout");
        agg.nonzero_exits += int("nonzero_exit");
        agg.malformed_outputs += int("malformed_output");
        agg.affected_examinations.insert(
            cr.get("examination")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        );
        if agg.examples.len() < 5 {
            agg.examples.push(format!(
                "{}:{}({})",
                cr.get("input").and_then(Value::as_str).unwrap_or(""),
                cr.get("examination").and_then(Value::as_str).unwrap_or(""),
                int("lost_points")
            ));
        }
        case_rows.push(cr);
    }
    let mut ranked: Vec<Map<String, Value>> = Vec::new();
    for agg in aggregates.into_values() {
        let mut m = Map::new();
        m.insert("blocker".into(), Value::from(agg.blocker));
        m.insert("lane_family".into(), Value::from(agg.lane_family));
        m.insert("selected_lane".into(), Value::from(agg.selected_lane));
        m.insert("next_answer_lane".into(), Value::from(agg.next_answer_lane));
        m.insert("fallback_lane".into(), Value::from(agg.fallback_lane));
        m.insert("reason_code".into(), Value::from(agg.reason_code));
        m.insert("route_status".into(), Value::from(agg.route_status));
        m.insert("blocker_gate".into(), Value::from(agg.blocker_gate));
        m.insert("action_code".into(), Value::from(agg.action_code));
        m.insert("owner_project".into(), Value::from(agg.owner_project));
        m.insert("owner_primitive".into(), Value::from(agg.owner_primitive));
        m.insert("cases".into(), Value::from(agg.cases));
        m.insert("lost_points".into(), Value::from(agg.lost_points));
        m.insert("known_units".into(), Value::from(agg.known_units));
        m.insert("exact_units".into(), Value::from(agg.exact_units));
        m.insert(
            "cannot_compute_known_units".into(),
            Value::from(agg.cannot_compute_known_units),
        );
        m.insert(
            "missing_known_units".into(),
            Value::from(agg.missing_known_units),
        );
        m.insert(
            "wrong_known_units".into(),
            Value::from(agg.wrong_known_units),
        );
        m.insert(
            "extra_output_units".into(),
            Value::from(agg.extra_output_units),
        );
        m.insert("timeouts".into(), Value::from(agg.timeouts));
        m.insert("nonzero_exits".into(), Value::from(agg.nonzero_exits));
        m.insert(
            "malformed_outputs".into(),
            Value::from(agg.malformed_outputs),
        );
        let exams: Vec<String> = agg.affected_examinations.into_iter().collect();
        m.insert("affected_examinations".into(), Value::from(exams.join(",")));
        m.insert(
            "examples".into(),
            Value::Array(agg.examples.into_iter().map(Value::from).collect()),
        );
        ranked.push(m);
    }
    ranked.sort_by(|a, b| {
        let la = a.get("lost_points").and_then(Value::as_i64).unwrap_or(0);
        let lb = b.get("lost_points").and_then(Value::as_i64).unwrap_or(0);
        let wa = a
            .get("wrong_known_units")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let wb = b
            .get("wrong_known_units")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let ca = a.get("cases").and_then(Value::as_i64).unwrap_or(0);
        let cb = b.get("cases").and_then(Value::as_i64).unwrap_or(0);
        lb.cmp(&la)
            .then_with(|| wb.cmp(&wa))
            .then_with(|| cb.cmp(&ca))
            .then_with(|| {
                a.get("blocker")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .cmp(b.get("blocker").and_then(Value::as_str).unwrap_or(""))
            })
    });
    let totals = json!({
        "case_blockers": case_rows.len(),
        "ranked_blockers": ranked.len(),
        "lost_points": case_rows.iter().map(|r| r.get("lost_points").and_then(Value::as_i64).unwrap_or(0)).sum::<i64>(),
        "wrong_known_units": case_rows.iter().map(|r| r.get("wrong_known_units").and_then(Value::as_i64).unwrap_or(0)).sum::<i64>(),
        "cannot_compute_known_units": case_rows.iter().map(|r| r.get("cannot_compute_known_units").and_then(Value::as_i64).unwrap_or(0)).sum::<i64>(),
        "missing_known_units": case_rows.iter().map(|r| r.get("missing_known_units").and_then(Value::as_i64).unwrap_or(0)).sum::<i64>(),
        "extra_output_units": case_rows.iter().map(|r| r.get("extra_output_units").and_then(Value::as_i64).unwrap_or(0)).sum::<i64>(),
    });
    let report = json!({
        "totals": totals,
        "ranked_blockers": ranked,
    });
    (report, case_rows)
}

fn write_blocker_loss_report(
    rows: &[ResultRow],
    output_dir: &Path,
    route_cause_index: Option<&RouteCauseIndex>,
) -> Result<Value> {
    fs::create_dir_all(output_dir)?;
    let (report, case_rows) = build_blocker_loss_report(rows, route_cause_index);
    fs::write(
        output_dir.join("blocker-loss.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    let blocker_fields = &[
        "blocker",
        "lane_family",
        "selected_lane",
        "next_answer_lane",
        "fallback_lane",
        "reason_code",
        "route_status",
        "blocker_gate",
        "action_code",
        "owner_project",
        "owner_primitive",
        "cases",
        "lost_points",
        "known_units",
        "exact_units",
        "cannot_compute_known_units",
        "missing_known_units",
        "wrong_known_units",
        "extra_output_units",
        "timeouts",
        "nonzero_exits",
        "malformed_outputs",
        "affected_examinations",
        "examples",
    ];
    let ranked = report
        .get("ranked_blockers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|v| v.as_object().cloned())
        .collect::<Vec<_>>();
    write_tsv(
        &output_dir.join("blocker-loss.tsv"),
        &ranked,
        blocker_fields,
    )?;
    let case_fields = &[
        "input",
        "examination",
        "category",
        "blocker",
        "lost_points",
        "known_units",
        "exact_units",
        "cannot_compute_known_units",
        "missing_known_units",
        "wrong_known_units",
        "extra_output_units",
        "timeout",
        "nonzero_exit",
        "malformed_output",
        "selected_lane",
        "next_answer_lane",
        "fallback_lane",
        "lane_family",
        "reason_code",
        "route_status",
        "blocker_gate",
        "action_code",
        "owner_project",
        "owner_primitive",
        "elapsed_ms",
        "expected",
        "actual",
        "note",
    ];
    write_tsv(
        &output_dir.join("case-blockers.tsv"),
        &case_rows,
        case_fields,
    )?;
    Ok(report)
}

fn write_score_report(
    rows: &[ResultRow],
    output_dir: &Path,
    route_cause_index: Option<&RouteCauseIndex>,
) -> Result<Value> {
    fs::create_dir_all(output_dir)?;
    let (report, case_rows) = build_score_report(rows);
    fs::write(
        output_dir.join("score-loss.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    let mut per_exam_rows: Vec<Map<String, Value>> = Vec::new();
    if let Some(per_exam) = report.get("per_examination").and_then(Value::as_object) {
        for (exam, bucket) in per_exam {
            let mut map = Map::new();
            map.insert("examination".into(), Value::from(exam.clone()));
            if let Some(obj) = bucket.as_object() {
                for (k, v) in obj {
                    map.insert(k.clone(), v.clone());
                }
            }
            per_exam_rows.push(map);
        }
    }
    if let Some(totals) = report.get("totals").and_then(Value::as_object) {
        let mut map = Map::new();
        map.insert("examination".into(), Value::from("TOTAL"));
        for (k, v) in totals {
            map.insert(k.clone(), v.clone());
        }
        per_exam_rows.push(map);
    }
    let score_fields = &[
        "examination",
        "cases",
        "format_ok",
        "matched_cases",
        "mismatched_cases",
        "malformed_output",
        "nonzero_exits",
        "timeouts",
        "known_units",
        "exact_units",
        "historical_unknown_units",
        "cannot_compute_known_units",
        "wrong_known_units",
        "missing_known_units",
        "extra_output_units",
        "runtime_p50_ms",
        "runtime_p95_ms",
        "runtime_max_ms",
        "categories",
    ];
    write_tsv(
        &output_dir.join("score-loss.tsv"),
        &per_exam_rows,
        score_fields,
    )?;
    let case_fields = &[
        "input",
        "examination",
        "category",
        "rc",
        "timeout",
        "format_ok",
        "match",
        "elapsed_ms",
        "known_units",
        "exact_units",
        "historical_unknown_units",
        "cannot_compute_known_units",
        "wrong_known_units",
        "missing_known_units",
        "extra_output_units",
        "expected",
        "actual",
        "note",
    ];
    write_tsv(&output_dir.join("case-loss.tsv"), &case_rows, case_fields)?;
    write_blocker_loss_report(rows, output_dir, route_cause_index)?;
    Ok(report)
}

// ---------- Comparison ----------

fn compare_unit_outcomes(baseline: &RowAnalysis, candidate: &RowAnalysis) -> BTreeMap<String, i64> {
    let mut totals: BTreeMap<String, i64> = BTreeMap::new();
    let keys = [
        "gained_exact_units",
        "lost_exact_units",
        "recovered_from_cannot_units",
        "new_cannot_compute_units",
        "new_wrong_units",
        "new_missing_known_units",
        "new_missing_output_units",
        "new_extra_output_units",
        "wrong_after_exact_units",
        "fixed_wrong_units",
    ];
    for k in keys {
        totals.insert(k.to_string(), 0);
    }
    for (b, c) in baseline.outcomes.iter().zip(candidate.outcomes.iter()) {
        if c == "exact" && b != "exact" {
            *totals.get_mut("gained_exact_units").unwrap() += 1;
        }
        if b == "exact" && c != "exact" {
            *totals.get_mut("lost_exact_units").unwrap() += 1;
        }
        if b == "cannot_compute" && c == "exact" {
            *totals.get_mut("recovered_from_cannot_units").unwrap() += 1;
        }
        if b == "exact" && c == "cannot_compute" {
            *totals.get_mut("new_cannot_compute_units").unwrap() += 1;
        }
        if c == "wrong" && b != "wrong" {
            *totals.get_mut("new_wrong_units").unwrap() += 1;
        }
        if c == "missing" && b != "missing" {
            *totals.get_mut("new_missing_known_units").unwrap() += 1;
        }
        if b == "exact" && c == "wrong" {
            *totals.get_mut("wrong_after_exact_units").unwrap() += 1;
        }
        if b == "wrong" && c != "wrong" {
            *totals.get_mut("fixed_wrong_units").unwrap() += 1;
        }
    }
    if candidate.extra_output_units > baseline.extra_output_units {
        *totals.get_mut("new_extra_output_units").unwrap() +=
            (candidate.extra_output_units - baseline.extra_output_units) as i64;
    }
    let baseline_missing = (baseline.expected_units as i64 - baseline.actual_units as i64).max(0);
    let candidate_missing =
        (candidate.expected_units as i64 - candidate.actual_units as i64).max(0);
    if candidate_missing > baseline_missing {
        *totals.get_mut("new_missing_output_units").unwrap() +=
            candidate_missing - baseline_missing;
    }
    totals
}

fn result_map(rows: &[ResultRow]) -> Result<BTreeMap<(String, String), &ResultRow>> {
    let mut map = BTreeMap::new();
    for row in rows {
        let key = (row.input_name.clone(), row.exam.clone());
        if map.contains_key(&key) {
            bail!("duplicate result row for {} {}", row.input_name, row.exam);
        }
        map.insert(key, row);
    }
    Ok(map)
}

fn write_comparison_report(
    baseline_rows: &[ResultRow],
    candidate_rows: &[ResultRow],
    output_dir: &Path,
) -> Result<Value> {
    fs::create_dir_all(output_dir)?;
    let baseline_map = result_map(baseline_rows)?;
    let candidate_map = result_map(candidate_rows)?;
    let baseline_keys: BTreeSet<&(String, String)> = baseline_map.keys().collect();
    let candidate_keys: BTreeSet<&(String, String)> = candidate_map.keys().collect();
    let mut common: Vec<&(String, String)> = baseline_keys
        .intersection(&candidate_keys)
        .copied()
        .collect();
    common.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| exam_order(&a.1).cmp(&exam_order(&b.1)))
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut candidate_only: Vec<&(String, String)> =
        candidate_keys.difference(&baseline_keys).copied().collect();
    candidate_only.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| exam_order(&a.1).cmp(&exam_order(&b.1)))
            .then_with(|| a.1.cmp(&b.1))
    });
    let baseline_only: Vec<&(String, String)> =
        baseline_keys.difference(&candidate_keys).copied().collect();
    let mut totals: BTreeMap<String, i64> = BTreeMap::new();
    let total_keys = [
        "exact_unit_delta",
        "gained_exact_units",
        "lost_exact_units",
        "recovered_from_cannot_units",
        "new_cannot_compute_units",
        "new_wrong_units",
        "candidate_only_wrong_units",
        "new_missing_known_units",
        "candidate_only_missing_known_units",
        "new_missing_output_units",
        "candidate_only_missing_output_units",
        "new_extra_output_units",
        "candidate_only_extra_output_units",
        "wrong_after_exact_units",
        "fixed_wrong_units",
        "new_timeouts",
        "candidate_only_timeouts",
        "fixed_timeouts",
        "new_nonzero_exits",
        "candidate_only_nonzero_exits",
        "fixed_nonzero_exits",
        "new_malformed_output",
        "candidate_only_malformed_output",
        "fixed_malformed_output",
    ];
    for k in total_keys {
        totals.insert(k.to_string(), 0);
    }
    totals.insert("baseline_cases".into(), baseline_rows.len() as i64);
    totals.insert("candidate_cases".into(), candidate_rows.len() as i64);
    totals.insert("common_cases".into(), common.len() as i64);
    totals.insert("baseline_only_cases".into(), baseline_only.len() as i64);
    totals.insert("candidate_only_cases".into(), candidate_only.len() as i64);
    let mut case_rows: Vec<Map<String, Value>> = Vec::new();
    for key in &common {
        let b = baseline_map[key];
        let c = candidate_map[key];
        let ba = analyze_result_row(b);
        let ca = analyze_result_row(c);
        let unit_delta = compare_unit_outcomes(&ba, &ca);
        let exact_delta = ca.exact_units as i64 - ba.exact_units as i64;
        *totals.get_mut("exact_unit_delta").unwrap() += exact_delta;
        for (k, v) in &unit_delta {
            *totals.entry(k.clone()).or_insert(0) += v;
        }
        *totals.get_mut("new_timeouts").unwrap() += i64::from(c.timed_out && !b.timed_out);
        *totals.get_mut("fixed_timeouts").unwrap() += i64::from(b.timed_out && !c.timed_out);
        *totals.get_mut("new_nonzero_exits").unwrap() += i64::from(c.rc != 0 && b.rc == 0);
        *totals.get_mut("fixed_nonzero_exits").unwrap() += i64::from(b.rc != 0 && c.rc == 0);
        *totals.get_mut("new_malformed_output").unwrap() += i64::from(!c.format_ok && b.format_ok);
        *totals.get_mut("fixed_malformed_output").unwrap() +=
            i64::from(!b.format_ok && c.format_ok);
        let mut row = Map::new();
        row.insert("input".into(), Value::from(key.0.clone()));
        row.insert("examination".into(), Value::from(key.1.clone()));
        row.insert("baseline_category".into(), Value::from(ba.category.clone()));
        row.insert(
            "candidate_category".into(),
            Value::from(ca.category.clone()),
        );
        row.insert("exact_unit_delta".into(), Value::from(exact_delta));
        for (k, v) in &unit_delta {
            row.insert(k.clone(), Value::from(*v));
        }
        row.insert(
            "elapsed_ms_delta".into(),
            Value::from(c.elapsed_ms - b.elapsed_ms),
        );
        row.insert("baseline_elapsed_ms".into(), Value::from(b.elapsed_ms));
        row.insert("candidate_elapsed_ms".into(), Value::from(c.elapsed_ms));
        row.insert("baseline_actual".into(), Value::from(b.actual.clone()));
        row.insert("candidate_actual".into(), Value::from(c.actual.clone()));
        row.insert("expected".into(), Value::from(c.expected.clone()));
        case_rows.push(row);
    }
    let mut candidate_only_rows: Vec<Map<String, Value>> = Vec::new();
    for key in &candidate_only {
        let c = candidate_map[key];
        let ca = analyze_result_row(c);
        *totals.get_mut("candidate_only_wrong_units").unwrap() += ca.wrong_known_units as i64;
        *totals.get_mut("new_wrong_units").unwrap() += ca.wrong_known_units as i64;
        *totals
            .get_mut("candidate_only_missing_known_units")
            .unwrap() += ca.missing_known_units as i64;
        *totals.get_mut("new_missing_known_units").unwrap() += ca.missing_known_units as i64;
        let candidate_missing = (ca.expected_units as i64 - ca.actual_units as i64).max(0);
        *totals
            .get_mut("candidate_only_missing_output_units")
            .unwrap() += candidate_missing;
        *totals.get_mut("new_missing_output_units").unwrap() += candidate_missing;
        *totals.get_mut("candidate_only_extra_output_units").unwrap() +=
            ca.extra_output_units as i64;
        *totals.get_mut("new_extra_output_units").unwrap() += ca.extra_output_units as i64;
        *totals.get_mut("candidate_only_timeouts").unwrap() += i64::from(c.timed_out);
        *totals.get_mut("candidate_only_nonzero_exits").unwrap() += i64::from(c.rc != 0);
        *totals.get_mut("candidate_only_malformed_output").unwrap() += i64::from(!c.format_ok);
        let mut row = Map::new();
        row.insert("input".into(), Value::from(key.0.clone()));
        row.insert("examination".into(), Value::from(key.1.clone()));
        row.insert(
            "candidate_category".into(),
            Value::from(ca.category.clone()),
        );
        row.insert("rc".into(), Value::from(c.rc));
        row.insert("timeout".into(), Value::from(i32::from(c.timed_out)));
        row.insert("format_ok".into(), Value::from(i32::from(c.format_ok)));
        row.insert("elapsed_ms".into(), Value::from(c.elapsed_ms));
        row.insert("known_units".into(), Value::from(ca.known_units));
        row.insert("exact_units".into(), Value::from(ca.exact_units));
        row.insert(
            "cannot_compute_known_units".into(),
            Value::from(ca.cannot_compute_known_units),
        );
        row.insert(
            "wrong_known_units".into(),
            Value::from(ca.wrong_known_units),
        );
        row.insert(
            "missing_known_units".into(),
            Value::from(ca.missing_known_units),
        );
        row.insert(
            "missing_output_units".into(),
            Value::from(candidate_missing),
        );
        row.insert(
            "extra_output_units".into(),
            Value::from(ca.extra_output_units),
        );
        row.insert("expected".into(), Value::from(c.expected.clone()));
        row.insert("actual".into(), Value::from(c.actual.clone()));
        row.insert("note".into(), Value::from(c.note.clone()));
        candidate_only_rows.push(row);
        // Mirror Python: append the candidate-only summary to case_rows too.
        let mut case = Map::new();
        case.insert("input".into(), Value::from(key.0.clone()));
        case.insert("examination".into(), Value::from(key.1.clone()));
        case.insert(
            "baseline_category".into(),
            Value::from("baseline_missing".to_string()),
        );
        case.insert(
            "candidate_category".into(),
            Value::from(ca.category.clone()),
        );
        case.insert(
            "exact_unit_delta".into(),
            Value::from(ca.exact_units as i64),
        );
        case.insert(
            "gained_exact_units".into(),
            Value::from(ca.exact_units as i64),
        );
        case.insert("lost_exact_units".into(), Value::from(0));
        case.insert("recovered_from_cannot_units".into(), Value::from(0));
        case.insert("new_cannot_compute_units".into(), Value::from(0));
        case.insert(
            "new_wrong_units".into(),
            Value::from(ca.wrong_known_units as i64),
        );
        case.insert(
            "candidate_only_wrong_units".into(),
            Value::from(ca.wrong_known_units as i64),
        );
        case.insert(
            "new_missing_known_units".into(),
            Value::from(ca.missing_known_units as i64),
        );
        case.insert(
            "candidate_only_missing_known_units".into(),
            Value::from(ca.missing_known_units as i64),
        );
        case.insert(
            "new_missing_output_units".into(),
            Value::from(candidate_missing),
        );
        case.insert(
            "candidate_only_missing_output_units".into(),
            Value::from(candidate_missing),
        );
        case.insert(
            "new_extra_output_units".into(),
            Value::from(ca.extra_output_units as i64),
        );
        case.insert(
            "candidate_only_extra_output_units".into(),
            Value::from(ca.extra_output_units as i64),
        );
        case.insert("wrong_after_exact_units".into(), Value::from(0));
        case.insert("fixed_wrong_units".into(), Value::from(0));
        case.insert("elapsed_ms_delta".into(), Value::from(c.elapsed_ms));
        case.insert("baseline_elapsed_ms".into(), Value::from(""));
        case.insert("candidate_elapsed_ms".into(), Value::from(c.elapsed_ms));
        case.insert("baseline_actual".into(), Value::from(""));
        case.insert("candidate_actual".into(), Value::from(c.actual.clone()));
        case.insert("expected".into(), Value::from(c.expected.clone()));
        case_rows.push(case);
    }
    let baseline_only_rows: Vec<Value> = {
        let mut v = baseline_only
            .into_iter()
            .map(|(input_name, exam)| json!({"input": input_name, "examination": exam}))
            .collect::<Vec<_>>();
        v.sort_by(|a, b| {
            a.get("input")
                .and_then(Value::as_str)
                .unwrap_or("")
                .cmp(b.get("input").and_then(Value::as_str).unwrap_or(""))
        });
        v
    };
    let mut totals_obj = Map::new();
    for (k, v) in totals {
        totals_obj.insert(k, Value::from(v));
    }
    let report = json!({
        "totals": totals_obj,
        "baseline_only": baseline_only_rows,
        "candidate_only": candidate_only_rows,
    });
    fs::write(
        output_dir.join("comparison.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    let fields = &[
        "input",
        "examination",
        "baseline_category",
        "candidate_category",
        "exact_unit_delta",
        "gained_exact_units",
        "lost_exact_units",
        "recovered_from_cannot_units",
        "new_cannot_compute_units",
        "new_wrong_units",
        "candidate_only_wrong_units",
        "new_missing_known_units",
        "candidate_only_missing_known_units",
        "new_missing_output_units",
        "candidate_only_missing_output_units",
        "new_extra_output_units",
        "candidate_only_extra_output_units",
        "wrong_after_exact_units",
        "fixed_wrong_units",
        "elapsed_ms_delta",
        "baseline_elapsed_ms",
        "candidate_elapsed_ms",
        "baseline_actual",
        "candidate_actual",
        "expected",
    ];
    write_tsv(&output_dir.join("comparison.tsv"), &case_rows, fields)?;
    Ok(report)
}

// ---------- Archive helpers (used by `run`) ----------

fn clean_name(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut prev_underscore = false;
    for c in value.chars() {
        let ok = c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-';
        if ok {
            out.push(c);
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    out
}

/// Extract `archive.tgz` into `destination` using the system `tar` binary,
/// then locate the unique model.pnml directory. This shells out because
/// no archive crate is available in the workspace.
fn safe_extract_tgz(archive: &Path, destination: &Path) -> Result<PathBuf> {
    fs::create_dir_all(destination)?;
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(destination)
        .status()
        .with_context(|| format!("invoke tar to extract {}", archive.display()))?;
    if !status.success() {
        bail!(
            "tar exited with status {status:?} extracting {}",
            archive.display()
        );
    }
    let mut model_dirs: Vec<PathBuf> = Vec::new();
    walk_for_pnml(destination, &mut model_dirs)?;
    if model_dirs.is_empty() {
        bail!("no model.pnml found after extracting {}", archive.display());
    }
    if model_dirs.len() > 1 {
        bail!(
            "multiple model.pnml files found after extracting {}",
            archive.display()
        );
    }
    Ok(model_dirs.remove(0))
}

fn walk_for_pnml(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_for_pnml(&path, out)?;
        } else if path.file_name().and_then(|n| n.to_str()) == Some("model.pnml") {
            if let Some(parent) = path.parent() {
                out.push(parent.to_path_buf());
            }
        }
    }
    Ok(())
}

// ---------- Provenance gate (best-effort) ----------
//
// `repo_git_head` and `build_provenance_diagnostic` are native to the
// Rust `ty-mcc-smoke` binary (`src/bin/ty-mcc-smoke.rs`). The
// implementation below performs the same git-HEAD vs
// `--build-provenance-json` reconciliation inline; the `ty-mcc-smoke`
// binary remains the canonical entry point for the
// BenchKit sidecar flow.

fn repo_git_head_from_env_or_git() -> Option<String> {
    if let Ok(value) = std::env::var("TY_MCC_BUILD_GIT_HEAD") {
        if !value.is_empty() {
            return Some(value);
        }
    }
    let output = Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let head = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if head.is_empty() {
        None
    } else {
        Some(head)
    }
}

/// Invoke the candidate binary with `--build-provenance-json` and check
/// the reported build-git-head matches the current repo HEAD. Returns
/// `Ok(head)` on success, `Err` with a diagnostic on mismatch.
fn require_current_head_binary(binary: &Path) -> Result<String> {
    let expected_head = repo_git_head_from_env_or_git()
        .context("could not determine repository HEAD for provenance check")?;
    let output = Command::new(binary)
        .arg("--build-provenance-json")
        .output()
        .with_context(|| format!("invoke {} --build-provenance-json", binary.display()))?;
    if !output.status.success() {
        bail!(
            "binary build provenance gate failed: {} --build-provenance-json exited {:?}",
            binary.display(),
            output.status.code()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let provenance: Value = serde_json::from_str(stdout.trim())
        .with_context(|| format!("parse build-provenance JSON from {}", binary.display()))?;
    let actual_head = provenance
        .get("build_git_head")
        .and_then(Value::as_str)
        .unwrap_or("");
    if actual_head != expected_head {
        bail!(
            "binary build provenance gate failed: build_git_head={actual_head} but repo HEAD={expected_head}"
        );
    }
    Ok(expected_head)
}

// ---------- Run-cases helpers ----------

#[derive(Debug)]
struct CaseRunOutcome {
    rc: i32,
    timed_out: bool,
    elapsed_ms: i64,
    format_ok: bool,
    actual: String,
    format_note: String,
}

fn run_case(
    binary: &Path,
    model_dir: &Path,
    exam: &str,
    args: &RunCliArgs,
    case_artifact_dir: &Path,
) -> Result<CaseRunOutcome> {
    let expected_formula_ids = load_expected_formula_ids(model_dir, exam)?;
    let storage_dir = case_artifact_dir.join("storage");
    fs::create_dir_all(&storage_dir)?;
    let mut cmd = Command::new(binary);
    cmd.arg(model_dir)
        .arg("--examination")
        .arg(exam)
        .arg("--threads")
        .arg(args.threads.to_string())
        .arg("--memory-fraction")
        .arg(args.memory_fraction.to_string())
        .arg("--storage")
        .arg(&args.storage)
        .arg("--storage-dir")
        .arg(&storage_dir)
        .arg("--max-states")
        .arg(args.max_states.to_string())
        .arg("--timeout")
        .arg(args.mcc_timeout.to_string());
    if let Some(p) = &args.backend_evidence_jsonl {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).ok();
        }
        cmd.env(BACKEND_EVIDENCE_JSONL_ENV, p);
        cmd.env(MCC_BACKEND_EVIDENCE_JSONL_ENV, p);
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let started = Instant::now();
    let timeout = Duration::from_secs(args.outer_timeout);
    let outcome = run_with_timeout(cmd, timeout)?;
    let elapsed_ms = started.elapsed().as_millis() as i64;
    let (rc, timed_out, stdout, stderr) = outcome;
    fs::write(case_artifact_dir.join("stdout.txt"), &stdout)?;
    fs::write(case_artifact_dir.join("stderr.txt"), &stderr)?;
    let (format_ok, actual, format_note) =
        parse_mcc_output(exam, &stdout, expected_formula_ids.as_deref());
    Ok(CaseRunOutcome {
        rc,
        timed_out,
        elapsed_ms,
        format_ok,
        actual,
        format_note,
    })
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<(i32, bool, String, String)> {
    let mut child = cmd.spawn().context("spawn child process")?;
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = s.read_to_string(&mut stdout);
                }
                if let Some(mut s) = child.stderr.take() {
                    let _ = s.read_to_string(&mut stderr);
                }
                return Ok((status.code().unwrap_or(-1), false, stdout, stderr));
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let mut stdout = String::new();
                    let mut stderr = String::new();
                    if let Some(mut s) = child.stdout.take() {
                        let _ = s.read_to_string(&mut stdout);
                    }
                    if let Some(mut s) = child.stderr.take() {
                        let _ = s.read_to_string(&mut stderr);
                    }
                    return Ok((124, true, stdout, stderr));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

// ---------- Case selection ----------

fn comma_items(value: Option<&str>) -> BTreeSet<String> {
    let Some(v) = value else {
        return BTreeSet::new();
    };
    v.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn select_cases(run: &RunCliArgs, root: &Path) -> Result<Vec<(String, String, String)>> {
    let (wanted_exams, wanted_inputs): (BTreeSet<String>, Option<BTreeSet<String>>) =
        if let Some(bucket_name) = &run.bucket {
            let buckets = bucket_table();
            let Some(bucket) = buckets.get(bucket_name.as_str()) else {
                bail!("unknown bucket {bucket_name:?}");
            };
            if run.inputs.is_some()
                || run.exams.is_some()
                || run.small_inputs.is_some()
                || run.offset != 0
                || run.limit != 0
            {
                bail!(
                "--bucket cannot be combined with --inputs/--exams/--small-inputs/--offset/--limit"
            );
            }
            let exams: BTreeSet<String> = bucket
                .exams
                .iter()
                .map(|e| e.as_str().to_string())
                .collect();
            let inputs = bucket.inputs.map(|inputs| {
                inputs
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect::<BTreeSet<_>>()
            });
            (exams, inputs)
        } else {
            let exams = if run.exams.is_some() {
                comma_items(run.exams.as_deref())
            } else {
                exams_in_report_order()
                    .iter()
                    .map(|e| e.as_str().to_string())
                    .collect()
            };
            let inputs = run.inputs.as_deref().map(|v| comma_items(Some(v)));
            (exams, inputs)
        };
    let cases = load_cases(root)?;
    let mut filtered: Vec<(String, String, String)> = cases
        .into_iter()
        .filter(|(_input, exam, _expected)| wanted_exams.contains(exam))
        .collect();
    if let Some(inputs) = &wanted_inputs {
        filtered.retain(|(input, _, _)| inputs.contains(input));
    }
    if run.bucket.is_none() {
        if let Some(small) = run.small_inputs {
            let sizes = state_size_index(root)?;
            let mut pairs: Vec<(String, i64)> = sizes.into_iter().collect();
            pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let selected: BTreeSet<String> =
                pairs.into_iter().take(small).map(|(k, _)| k).collect();
            filtered.retain(|(input, _, _)| selected.contains(input));
        }
    }
    if run.offset > 0 {
        filtered = filtered.into_iter().skip(run.offset).collect();
    }
    if run.limit > 0 {
        filtered.truncate(run.limit);
    }
    Ok(filtered)
}

fn print_buckets(root: &Path) -> Result<()> {
    let buckets = bucket_table();
    for (name, bucket) in &buckets {
        let input_count = match bucket.inputs {
            Some(list) => list.len().to_string(),
            None => "all".to_string(),
        };
        let cases = match bucket.inputs {
            Some(list) => list.len() * bucket.exams.len(),
            None => load_cases(root).map(|c| c.len()).unwrap_or(0),
        };
        println!(
            "{name}\tinputs={input_count}\texams={}\tcases={cases}\t{}",
            bucket.exams.len(),
            bucket.description
        );
    }
    Ok(())
}

// ---------- CLI ----------

/// Command-line arguments for the `ty-mcc-history` harness.
#[derive(Debug, Parser)]
#[command(
    name = "ty-mcc-history",
    version,
    about = "MCC 2025 history harness — validates archive shape, runs cases against a candidate \
             binary, and produces score/blocker/comparison reports.",
    long_about = "MCC 2025 history harness for the TY MCC submission.\n\n\
        The official MCC archives are too large to keep in the repository. \
        This binary expects them under ~/mcc-prev/2025 by default, \
        verifies the archive/result shape, and can run historical input/examination \
        cases through a candidate `ty-mcc` binary."
)]
pub struct Cli {
    /// MCC archive root directory.
    #[arg(long, default_value_os_t = default_root(), global = true)]
    pub root: PathBuf,

    /// Expected `INPUTS-2025.tar.gz` SHA-256 (or set via `MCC_2025_INPUTS_SHA256`).
    #[arg(long, global = true)]
    pub inputs_sha256: Option<String>,

    /// The history subcommand to run.
    #[command(subcommand)]
    pub command: Cmd,
}

/// Subcommands of the `ty-mcc-history` harness.
//
// Clap-derived subcommands; Args structs are pub(crate) for internal organization.
// The enum is pub for the binary shim entry point but external users won't
// pattern-match on its variants.
#[allow(private_interfaces)]
#[derive(Debug, Subcommand)]
pub enum Cmd {
    /// Verify downloaded MCC 2025 history shape and write the case list.
    Summary,
    /// Copy and codesign a candidate binary, then run --version.
    ProbeMacos(ProbeMacosArgs),
    /// Write score-loss reports from a historical run results.tsv.
    Report(ReportArgs),
    /// Compare baseline and candidate historical run results.tsv files.
    Compare(CompareArgs),
    /// Run selected MCC 2025 cases against a candidate binary.
    Run(RunCliArgs),
    /// Parse a single MCC stdout payload and print the canonical verdict.
    ///
    /// Helpful debugging shim around [`parse_mcc_output`]. Reads stdout from
    /// either the `--stdin` flag or a `--input FILE` path and emits the
    /// (format_ok, actual, note) triple as JSON.
    ValidateOutput(ValidateOutputArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ProbeMacosArgs {
    /// Candidate binary to copy / codesign / probe.
    #[arg(long)]
    binary: PathBuf,
    /// Destination directory; defaults to a cache under the user's home directory.
    #[arg(long, default_value_os_t = default_probe_output_dir())]
    output_dir: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ReportArgs {
    /// Run directory or path to `results.tsv`.
    results: PathBuf,
    /// Report output directory; defaults to the results.tsv parent.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Optional route/lane evidence JSONL sidecar to enrich blocker-loss reports.
    #[arg(long)]
    backend_evidence_jsonl: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct CompareArgs {
    /// Baseline run directory or results.tsv.
    #[arg(long, alias = "baseline")]
    baseline_run: PathBuf,
    /// Candidate run directory or results.tsv.
    #[arg(long, alias = "candidate")]
    candidate_run: PathBuf,
    /// Comparison output directory; defaults to `<candidate-run>/comparison`.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Exit non-zero if the candidate introduces any wrong, missing, malformed,
    /// or failing known-case output relative to the baseline.
    #[arg(long)]
    reject_on_wrong: bool,
}

#[derive(Debug, Args, Clone)]
pub(crate) struct RunCliArgs {
    /// Candidate `ty-mcc` binary to execute per case.
    #[arg(long)]
    binary: Option<PathBuf>,
    /// Output directory; defaults to `<root>/run-history/<timestamp>`.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Fixed target bucket for reproducible candidate gates.
    #[arg(long)]
    bucket: Option<String>,
    /// List fixed target buckets and exit.
    #[arg(long)]
    list_buckets: bool,
    /// Comma-separated input names to include.
    #[arg(long)]
    inputs: Option<String>,
    /// Comma-separated examination names to include.
    #[arg(long)]
    exams: Option<String>,
    /// Select the N inputs with the smallest StateSpace history.
    #[arg(long)]
    small_inputs: Option<usize>,
    /// Skip the first N selected cases.
    #[arg(long, default_value_t = 0)]
    offset: usize,
    /// Limit the run to the first N selected cases (after offset).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Threads for the candidate binary.
    #[arg(long, default_value_t = 4)]
    threads: u32,
    /// Memory budget fraction for the candidate binary.
    #[arg(long, default_value_t = 0.25)]
    memory_fraction: f64,
    /// Storage backend.
    #[arg(long, default_value = "memory")]
    storage: String,
    /// State-space limit passed to the candidate binary.
    #[arg(long, default_value_t = 1_000_000)]
    max_states: u64,
    /// Per-case timeout (seconds) passed to the candidate binary.
    #[arg(long, default_value_t = 20)]
    mcc_timeout: u64,
    /// Outer (wall-clock) timeout per case.
    #[arg(long, default_value_t = 40)]
    outer_timeout: u64,
    /// Write backend evidence JSONL sidecar to this path.
    #[arg(long)]
    backend_evidence_jsonl: Option<PathBuf>,
    /// Exit non-zero on any mismatch / timeout / nonzero exit.
    #[arg(long)]
    strict: bool,
    /// Skip the build-provenance gate (TY_MCC_BUILD_GIT_HEAD identity check).
    #[arg(long)]
    skip_provenance_gate: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ValidateOutputArgs {
    /// MCC examination name (e.g. `ReachabilityFireability`).
    #[arg(long)]
    examination: String,
    /// Optional path to a model directory containing the property XML so
    /// formula-id parity is enforced.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Read stdout from this file (mutually exclusive with `--stdin`).
    #[arg(long)]
    input: Option<PathBuf>,
    /// Read stdout from this process's stdin.
    #[arg(long)]
    stdin: bool,
}

// ---------- Subcommand dispatch ----------

fn cmd_summary(root: &Path, inputs_sha256: Option<&str>) -> Result<()> {
    let mut summary = history_summary(root, inputs_sha256)?;
    let case_list = write_case_list(root)?;
    let lines = std::io::BufReader::new(File::open(&case_list)?)
        .lines()
        .map_while(Result::ok)
        .count();
    if let Some(obj) = summary.as_object_mut() {
        obj.insert(
            "case_list".to_string(),
            Value::from(case_list.display().to_string()),
        );
        obj.insert("case_list_lines".to_string(), Value::from(lines));
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn cmd_probe_macos(args: &ProbeMacosArgs) -> Result<()> {
    fs::create_dir_all(&args.output_dir)?;
    let name = args
        .binary
        .file_name()
        .ok_or_else(|| anyhow!("binary path has no filename: {}", args.binary.display()))?;
    let signed = args.output_dir.join(name);
    let cp_status = Command::new("cp")
        .arg("-X")
        .arg(&args.binary)
        .arg(&signed)
        .status()
        .context("invoke cp")?;
    if !cp_status.success() {
        bail!("cp exited {cp_status:?}");
    }
    let sign_status = Command::new("codesign")
        .args(["--force", "--sign", "-"])
        .arg(&signed)
        .status()
        .context("invoke codesign")?;
    if !sign_status.success() {
        bail!("codesign exited {sign_status:?}");
    }
    let output = Command::new(&signed).arg("--version").output()?;
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    std::process::exit(output.status.code().unwrap_or(0));
}

fn cmd_report(args: &ReportArgs) -> Result<()> {
    let results_path = resolve_results_path(&args.results)?;
    let output_dir = args.output_dir.clone().unwrap_or_else(|| {
        results_path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    });
    let route_cause_index = load_route_cause_index(args.backend_evidence_jsonl.as_deref());
    let rows = read_result_rows(&results_path)?;
    let report = write_score_report(&rows, &output_dir, Some(&route_cause_index))?;
    let blocker_report: Value =
        serde_json::from_str(&fs::read_to_string(output_dir.join("blocker-loss.json"))?)?;
    let blocker_totals = blocker_report.get("totals").cloned().unwrap_or(Value::Null);
    let summary = json!({
        "results": results_path.display().to_string(),
        "output_dir": output_dir.display().to_string(),
        "score_loss_json": output_dir.join("score-loss.json").display().to_string(),
        "score_loss_tsv": output_dir.join("score-loss.tsv").display().to_string(),
        "case_loss_tsv": output_dir.join("case-loss.tsv").display().to_string(),
        "blocker_loss_json": output_dir.join("blocker-loss.json").display().to_string(),
        "blocker_loss_tsv": output_dir.join("blocker-loss.tsv").display().to_string(),
        "case_blockers_tsv": output_dir.join("case-blockers.tsv").display().to_string(),
        "totals": report.get("totals"),
        "blocker_totals": blocker_totals,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn cmd_compare(args: &CompareArgs) -> Result<i32> {
    let baseline_path = resolve_results_path(&args.baseline_run)?;
    let candidate_path = resolve_results_path(&args.candidate_run)?;
    let output_dir = args.output_dir.clone().unwrap_or_else(|| {
        candidate_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("comparison")
    });
    let baseline_rows = read_result_rows(&baseline_path)?;
    let candidate_rows = read_result_rows(&candidate_path)?;
    let report = write_comparison_report(&baseline_rows, &candidate_rows, &output_dir)?;
    let summary = json!({
        "baseline_results": baseline_path.display().to_string(),
        "candidate_results": candidate_path.display().to_string(),
        "output_dir": output_dir.display().to_string(),
        "comparison_json": output_dir.join("comparison.json").display().to_string(),
        "comparison_tsv": output_dir.join("comparison.tsv").display().to_string(),
        "totals": report.get("totals"),
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if args.reject_on_wrong {
        let reject_keys = [
            "new_wrong_units",
            "wrong_after_exact_units",
            "new_missing_known_units",
            "new_missing_output_units",
            "new_extra_output_units",
            "new_timeouts",
            "candidate_only_timeouts",
            "new_nonzero_exits",
            "candidate_only_nonzero_exits",
            "new_malformed_output",
            "candidate_only_malformed_output",
        ];
        if let Some(totals) = report.get("totals").and_then(Value::as_object) {
            for k in reject_keys {
                if totals.get(k).and_then(Value::as_i64).unwrap_or(0) != 0 {
                    return Ok(1);
                }
            }
        }
    }
    Ok(0)
}

#[derive(Debug, Serialize)]
struct RunSummary {
    cases: i64,
    format_ok: i64,
    matched: i64,
    mismatched: i64,
    timeouts: i64,
    nonzero: i64,
}

fn cmd_run(root: &Path, args: &mut RunCliArgs) -> Result<i32> {
    if args.list_buckets {
        print_buckets(root)?;
        return Ok(0);
    }
    let Some(binary) = args.binary.clone() else {
        bail!("--binary is required unless --list-buckets is used");
    };
    let binary = binary
        .canonicalize()
        .with_context(|| format!("resolve binary path {}", binary.display()))?;
    let binary_build_git_head = if args.skip_provenance_gate {
        repo_git_head_from_env_or_git().unwrap_or_default()
    } else {
        require_current_head_binary(&binary)?
    };
    let cases = select_cases(args, root)?;
    if cases.is_empty() {
        bail!("no cases selected");
    }
    let inputs_dir = root_paths(root).inputs_dir;
    let output_dir = args.output_dir.clone().unwrap_or_else(|| {
        let stamp = format_timestamp(SystemTime::now());
        root.join("run-history").join(stamp)
    });
    fs::create_dir_all(&output_dir)?;
    let artifacts_dir = output_dir.join("artifacts");
    fs::create_dir_all(&artifacts_dir)?;
    let results_path = output_dir.join("results.tsv");
    let backend_evidence_jsonl_explicit = args.backend_evidence_jsonl.is_some();
    if args.backend_evidence_jsonl.is_none() {
        args.backend_evidence_jsonl = Some(output_dir.join("backend-capability.jsonl"));
    }
    let sidecar = args.backend_evidence_jsonl.as_ref().unwrap().clone();
    if let Some(parent) = sidecar.parent() {
        fs::create_dir_all(parent)?;
    }
    let config = json!({
        "root": root.display().to_string(),
        "binary": binary.display().to_string(),
        "cases_selected": cases.len(),
        "threads": args.threads,
        "storage": args.storage,
        "max_states": args.max_states,
        "mcc_timeout": args.mcc_timeout,
        "outer_timeout": args.outer_timeout,
        "backend_evidence_jsonl": sidecar.display().to_string(),
        "backend_evidence_jsonl_explicit": backend_evidence_jsonl_explicit,
        "binary_build_git_head": binary_build_git_head,
    });
    fs::write(
        output_dir.join("run-config.json"),
        serde_json::to_string_pretty(&config)?,
    )?;
    let mut totals = RunSummary {
        cases: 0,
        format_ok: 0,
        matched: 0,
        mismatched: 0,
        timeouts: 0,
        nonzero: 0,
    };
    let mut grouped: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let mut group_order: Vec<String> = Vec::new();
    for (input, exam, expected) in &cases {
        if !grouped.contains_key(input) {
            group_order.push(input.clone());
        }
        grouped
            .entry(input.clone())
            .or_default()
            .push((exam.clone(), expected.clone()));
    }
    let mut results_file = File::create(&results_path)?;
    writeln!(
        results_file,
        "input\texamination\trc\ttimeout\tformat_ok\tmatch\telapsed_ms\texpected\tactual\tnote"
    )?;
    for (input_index, input_name) in group_order.iter().enumerate() {
        let exam_cases = &grouped[input_name];
        let archive = inputs_dir.join(format!("{input_name}.tgz"));
        if !archive.exists() {
            bail!("missing input archive: {}", archive.display());
        }
        let tmp_base = std::env::temp_dir().join(format!(
            "ty-mcc-{}-{}",
            clean_name(input_name),
            unix_micros()
        ));
        fs::create_dir_all(&tmp_base)?;
        let extract_result = (|| -> Result<()> {
            let model_dir = safe_extract_tgz(&archive, &tmp_base)?;
            for (exam, expected) in exam_cases {
                totals.cases += 1;
                let case_dir = artifacts_dir.join(format!("{}__{}", clean_name(input_name), exam));
                fs::create_dir_all(&case_dir)?;
                let outcome = run_case(&binary, &model_dir, exam, args, &case_dir)?;
                let actual = &outcome.actual;
                let matched = outcome.format_ok && actual == expected;
                if outcome.format_ok {
                    totals.format_ok += 1;
                }
                if matched {
                    totals.matched += 1;
                } else {
                    totals.mismatched += 1;
                }
                if outcome.timed_out {
                    totals.timeouts += 1;
                }
                if outcome.rc != 0 {
                    totals.nonzero += 1;
                }
                let note = outcome.format_note.replace('\t', " ");
                writeln!(
                    results_file,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    input_name,
                    exam,
                    outcome.rc,
                    i32::from(outcome.timed_out),
                    i32::from(outcome.format_ok),
                    i32::from(matched),
                    outcome.elapsed_ms,
                    expected,
                    actual,
                    note
                )?;
                results_file.flush()?;
                println!(
                    "[{}/{}] {} {}: rc={} format={} match={} elapsed_ms={}",
                    totals.cases,
                    cases.len(),
                    input_name,
                    exam,
                    outcome.rc,
                    i32::from(outcome.format_ok),
                    i32::from(matched),
                    outcome.elapsed_ms
                );
            }
            Ok(())
        })();
        let _ = fs::remove_dir_all(&tmp_base);
        extract_result?;
        eprintln!(
            "completed input {}/{}: {}",
            input_index + 1,
            group_order.len(),
            input_name
        );
    }
    fs::write(
        output_dir.join("summary.json"),
        serde_json::to_string_pretty(&totals)?,
    )?;
    let route_cause_index = load_route_cause_index(Some(&sidecar));
    let rows = read_result_rows(&results_path)?;
    write_score_report(&rows, &output_dir, Some(&route_cause_index))?;
    let summary = json!({
        "output_dir": output_dir.display().to_string(),
        "score_loss_json": output_dir.join("score-loss.json").display().to_string(),
        "score_loss_tsv": output_dir.join("score-loss.tsv").display().to_string(),
        "case_loss_tsv": output_dir.join("case-loss.tsv").display().to_string(),
        "blocker_loss_json": output_dir.join("blocker-loss.json").display().to_string(),
        "blocker_loss_tsv": output_dir.join("blocker-loss.tsv").display().to_string(),
        "case_blockers_tsv": output_dir.join("case-blockers.tsv").display().to_string(),
        "cases": totals.cases,
        "format_ok": totals.format_ok,
        "matched": totals.matched,
        "mismatched": totals.mismatched,
        "timeouts": totals.timeouts,
        "nonzero": totals.nonzero,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if args.strict && (totals.mismatched > 0 || totals.timeouts > 0 || totals.nonzero > 0) {
        return Ok(1);
    }
    Ok(0)
}

fn cmd_validate_output(args: &ValidateOutputArgs) -> Result<()> {
    // Confirm the examination name parses — the canonical Examination
    // enum rejects anything that isn't one of the 13 MCC kinds.
    let _ = parse_exam_name(&args.examination)?;
    let stdout = if args.stdin {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else if let Some(path) = &args.input {
        fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?
    } else {
        bail!("provide --stdin or --input FILE");
    };
    let expected_ids = if let Some(model_dir) = &args.model_dir {
        load_expected_formula_ids(model_dir, &args.examination)?
    } else {
        None
    };
    let (format_ok, actual, note) =
        parse_mcc_output(&args.examination, &stdout, expected_ids.as_deref());
    let report = json!({
        "examination": args.examination,
        "format_ok": format_ok,
        "actual": actual,
        "note": note,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !format_ok {
        std::process::exit(1);
    }
    Ok(())
}

fn unix_micros() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0)
}

fn format_timestamp(time: SystemTime) -> String {
    let secs = time
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // YYYYMMDD-HHMMSS in UTC. We avoid pulling chrono — the tag is
    // monotonic enough for output-dir uniqueness.
    let days = secs / 86_400;
    let mut year = 1970_i64;
    let mut remaining = days as i64;
    let is_leap = |y: i64| (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }
    let month_lengths = |y: i64| {
        [
            31,
            if is_leap(y) { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ]
    };
    let mut month = 0;
    let mut day_of_year = remaining;
    let ml = month_lengths(year);
    for (i, &len) in ml.iter().enumerate() {
        if day_of_year < len {
            month = i + 1;
            break;
        }
        day_of_year -= len;
    }
    let day = day_of_year + 1;
    let rem_secs = secs % 86_400;
    let hour = rem_secs / 3600;
    let minute = (rem_secs % 3600) / 60;
    let second = rem_secs % 60;
    format!(
        "{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}",
        year = year,
        month = month,
        day = day,
        hour = hour,
        minute = minute,
        second = second
    )
}

/// Entry point used by the standalone `ty-mcc-history` binary.
pub fn run() -> ExitCode {
    execute(Cli::parse())
}

/// Entry point used by `ty-mccctl history`.
pub fn run_from<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(err) => {
            let _ = err.print();
            return ExitCode::from(u8::from(err.use_stderr()));
        }
    };
    execute(cli)
}

fn execute(cli: Cli) -> ExitCode {
    let root = cli.root;
    let inputs_sha256_env = std::env::var("MCC_2025_INPUTS_SHA256").ok();
    let inputs_sha256 = cli
        .inputs_sha256
        .or(inputs_sha256_env)
        .filter(|s| !s.trim().is_empty());
    let result: Result<i32> = match cli.command {
        Cmd::Summary => cmd_summary(&root, inputs_sha256.as_deref()).map(|()| 0),
        Cmd::ProbeMacos(args) => cmd_probe_macos(&args).map(|()| 0),
        Cmd::Report(args) => cmd_report(&args).map(|()| 0),
        Cmd::Compare(args) => cmd_compare(&args),
        Cmd::Run(mut args) => cmd_run(&root, &mut args),
        Cmd::ValidateOutput(args) => cmd_validate_output(&args).map(|()| 0),
    };
    match result {
        Ok(code) => {
            if (0..=255).contains(&code) {
                ExitCode::from(code as u8)
            } else {
                ExitCode::from(1)
            }
        }
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(1)
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examination_canonical_kinds_are_thirteen() {
        // The Python EXAMS list duplicates this; this test fences the
        // Rust source-of-truth count so a stray addition flags the
        // matching parity guard.
        assert_eq!(Examination::ALL.len(), 13);
    }

    #[test]
    fn parse_state_space_underscored() {
        let stdout = format!(
            "{STATE_SPACE} STATES 42 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} TRANSITIONS 100 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} MAX_TOKEN_IN_PLACE 7 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} MAX_TOKEN_PER_MARKING 9 {TECHNIQUES} EXPLICIT\n"
        );
        let (ok, actual, note) = parse_mcc_output("StateSpace", &stdout, None);
        assert!(ok, "note={note}");
        assert_eq!(actual, "42 100 7 9");
    }

    /// Build the legacy spaced StateSpace fixture at runtime. Constructing
    /// the spaced tokens via `format!("STATE{sp}SPACE", sp = " ")` keeps
    /// the source free of literal spaced keywords, so the MCC keyword
    /// guard does not need a directive on this line — same regression
    /// pattern as `crates/tla-petri/src/output_tests.rs`.
    fn legacy_state_space_fixture() -> String {
        let sp = " ";
        format!(
            "STATE{sp}SPACE STATES 42 TECHNIQUES EXPLICIT\n\
             STATE{sp}SPACE TRANSITIONS 100 TECHNIQUES EXPLICIT\n\
             STATE{sp}SPACE MAX{sp}TOKEN{sp}IN{sp}PLACE 7 TECHNIQUES EXPLICIT\n\
             STATE{sp}SPACE MAX{sp}TOKEN{sp}PER{sp}MARKING 9 TECHNIQUES EXPLICIT\n"
        )
    }

    #[test]
    fn parse_state_space_legacy_spaced_form_matches_canonical() {
        let legacy = legacy_state_space_fixture();
        let (ok_legacy, actual_legacy, _) = parse_mcc_output("StateSpace", &legacy, None);
        assert!(ok_legacy);
        let canonical = format!(
            "{STATE_SPACE} STATES 42 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} TRANSITIONS 100 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} MAX_TOKEN_IN_PLACE 7 {TECHNIQUES} EXPLICIT\n\
             {STATE_SPACE} MAX_TOKEN_PER_MARKING 9 {TECHNIQUES} EXPLICIT\n"
        );
        let (ok_canon, actual_canon, _) = parse_mcc_output("StateSpace", &canonical, None);
        assert!(ok_canon);
        assert_eq!(actual_legacy, actual_canon);
    }

    #[test]
    fn parse_state_space_cannot_compute_canonical_and_spaced() {
        let canonical = format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n");
        let (ok, actual, _) = parse_mcc_output("StateSpace", &canonical, None);
        assert!(ok);
        assert_eq!(actual, "DNC");
        let (ok_bare, actual_bare, _) = parse_mcc_output("StateSpace", CANNOT_COMPUTE, None);
        assert!(ok_bare);
        assert_eq!(actual_bare, "DNC");
        // mcc-keyword-guard: allow-spaced-mention
        let spaced = "STATE SPACE CANNOT COMPUTE TECHNIQUES EXPLICIT\n";
        let (ok_spaced, actual_spaced, _) = parse_mcc_output("StateSpace", spaced, None);
        assert!(ok_spaced);
        assert_eq!(actual_spaced, "DNC");
    }

    #[test]
    fn parse_single_formula_true_false_cc() {
        let exam = Examination::ReachabilityDeadlock.as_str();
        let (ok, actual, _) = parse_mcc_output(
            exam,
            &format!("{FORMULA} {exam} TRUE {TECHNIQUES} EXPLICIT\n"),
            None,
        );
        assert!(ok);
        assert_eq!(actual, "T");
        let (ok, actual, _) = parse_mcc_output(
            exam,
            &format!("{FORMULA} {exam} FALSE {TECHNIQUES} EXPLICIT\n"),
            None,
        );
        assert!(ok);
        assert_eq!(actual, "F");
        let (ok, actual, _) = parse_mcc_output(
            exam,
            &format!("{FORMULA} {exam} {CANNOT_COMPUTE} {TECHNIQUES} EXPLICIT\n"),
            None,
        );
        assert!(ok);
        assert_eq!(actual, "DNC");
    }

    #[test]
    fn parse_multi_formula_with_expected_ids() {
        let stdout = format!(
            "{FORMULA} F1 TRUE {TECHNIQUES} EXPLICIT\n\
             {FORMULA} F0 FALSE {TECHNIQUES} EXPLICIT\n"
        );
        let expected = vec!["F0".to_string(), "F1".to_string()];
        let (ok, actual, _) = parse_mcc_output("ReachabilityFireability", &stdout, Some(&expected));
        assert!(ok);
        assert_eq!(actual, "FT");
    }

    #[test]
    fn parse_multi_formula_id_mismatch_reports_diff() {
        let stdout = format!("{FORMULA} F0 TRUE {TECHNIQUES} EXPLICIT\n");
        let expected = vec!["F0".to_string(), "F1".to_string()];
        let (ok, _, note) = parse_mcc_output("ReachabilityFireability", &stdout, Some(&expected));
        assert!(!ok);
        assert!(note.contains("F1"));
    }

    #[test]
    fn parse_upper_bounds_vector() {
        let stdout = format!(
            "{FORMULA} U1 5 {TECHNIQUES} EXPLICIT\n\
             {FORMULA} U0 3 {TECHNIQUES} EXPLICIT\n"
        );
        let expected = vec!["U0".to_string(), "U1".to_string()];
        let (ok, actual, _) = parse_mcc_output("UpperBounds", &stdout, Some(&expected));
        assert!(ok);
        assert_eq!(actual, "3 5");
    }

    #[test]
    fn unknown_examination_name_rejected() {
        // Examination::from_name returns an error for non-MCC names.
        assert!(parse_exam_name("NotAnExam").is_err());
    }

    #[test]
    fn examination_kinds_are_routed_through_the_enum() {
        // Every name we accept must be parseable by the canonical
        // Examination enum. This is the regression fence that would
        // catch a typo like "ReachibilityDeadlock" appearing in the
        // Python EXAMS list before the parity guard.
        for ex in exams_in_report_order() {
            let s = ex.as_str();
            let parsed = parse_exam_name(s).expect("name must parse");
            assert_eq!(parsed, ex);
        }
    }

    #[test]
    fn normalize_expected_preserves_state_space_numbers() {
        let normalized = normalize_expected("StateSpace", " 100 200 300 ");
        assert_eq!(normalized, "100 200 300");
    }

    #[test]
    fn normalize_expected_collapses_property_spaces() {
        let normalized = normalize_expected("ReachabilityFireability", "T F D");
        assert_eq!(normalized, "TFD");
    }

    #[test]
    fn split_result_units_state_space() {
        let units = split_result_units("StateSpace", "1 2 3 4");
        assert_eq!(units, vec!["1", "2", "3", "4"]);
    }

    #[test]
    fn split_result_units_property() {
        let units = split_result_units("ReachabilityFireability", "TFD");
        assert_eq!(units, vec!["T", "F", "D"]);
    }

    #[test]
    fn cannot_compute_token_accepts_both_forms() {
        assert!(is_cannot_compute_token(CANNOT_COMPUTE));
        // mcc-keyword-guard: allow-spaced-mention
        assert!(is_cannot_compute_token("CANNOT COMPUTE"));
        assert!(is_cannot_compute_token("DNC"));
        assert!(is_cannot_compute_token("D"));
        assert!(!is_cannot_compute_token("TRUE"));
    }

    #[test]
    fn csv_line_parser_handles_quoted_commas() {
        let parsed = parse_csv_line("a,b,\"c,d\",e");
        assert_eq!(parsed, vec!["a", "b", "c,d", "e"]);
    }

    #[test]
    fn shell_split_handles_quoted_evidence_tokens() {
        let tokens = shell_split("MCC blocker_action selected=true reason=\"a b\"");
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[3], "reason=a b");
    }

    #[test]
    fn route_cause_from_evidence_record() {
        let record = json!({
            "input": "ModelX-PT-001",
            "examination": "ReachabilityFireability",
            "evidence": [
                "MCC blocker_action selected=true priority_rank=1 blocker_piece=ay reason_code=NEEDS_MORE_TIME",
                "MCC production_selector_decision selected_lane=symbolic reason_code=PRIMARY"
            ]
        });
        let cause = route_cause_from_record(&record).expect("present");
        assert_eq!(cause.blocker_piece, "ay");
        assert_eq!(cause.selected_lane, "symbolic");
    }

    #[test]
    fn analyze_row_identifies_exact_match() {
        let row = ResultRow {
            input_name: "M".into(),
            exam: "ReachabilityFireability".into(),
            rc: 0,
            timed_out: false,
            format_ok: true,
            matched: true,
            elapsed_ms: 10,
            expected: "TF".into(),
            actual: "TF".into(),
            note: String::new(),
        };
        let a = analyze_result_row(&row);
        assert_eq!(a.exact_units, 2);
        assert_eq!(a.known_units, 2);
        assert_eq!(a.category, "exact_match");
    }

    #[test]
    fn analyze_row_identifies_wrong_known() {
        let row = ResultRow {
            input_name: "M".into(),
            exam: "ReachabilityFireability".into(),
            rc: 0,
            timed_out: false,
            format_ok: true,
            matched: false,
            elapsed_ms: 10,
            expected: "TF".into(),
            actual: "FT".into(),
            note: String::new(),
        };
        let a = analyze_result_row(&row);
        assert_eq!(a.wrong_known_units, 2);
        assert_eq!(a.category, "wrong_known");
    }
}
