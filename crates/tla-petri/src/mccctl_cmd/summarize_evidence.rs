// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (the regression-fence test below constructs legacy spaced literals at
// runtime so an auto-fixer cannot rewrite them; production sources only
// emit the canonical underscored forms via `tla_petri::mcc_keywords`.)

//! Summarize MCC backend-capability JSONL evidence sidecars.
//!
//! Replaces a former Python helper. Reads
//! one or more capability-evidence sidecars (or stdin) and aggregates
//! the embedded selected/rejected backend lanes, native/JIT gate rows,
//! trust-ir/trust_cg/AY evidence rows, and proof-replay boundary rows into the
//! same Markdown / CSV / JSON shapes the Python tool emitted.
//!
//! Routes MCC keyword references through `tla_petri::mcc_keywords` and
//! every examination name through `tla_petri::examination::Examination`
//! so the summarizer cannot drift from the canonical MCC vocabulary
//! that the validator binary enforces.
//!
//! The CLI surface mirrors the Python script exactly:
//!
//! ```text
//! ty-mcc-summarize-evidence [--csv | --json | --summary-json] JSONL...
//! ty-mcc-summarize-evidence -            (stdin)
//! ```
//!
//! Exit 0 = success, exit 1 = read/parse failure (with `path:line:`
//! diagnostics matching the Python source).

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::io::{self, Write};
use std::process::ExitCode;

use clap::Parser;
use serde_json::{Map, Value};

use crate::examination::Examination;
use crate::mcc_evidence_jsonl::read_jsonl_records;
use crate::mcc_keywords::{CANNOT_COMPUTE, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING, STATE_SPACE};

// Touch the imported MCC vocabulary so the summarizer cannot drift from
// the canonical keyword/examination authority — every spec emitted into
// these aggregates ultimately uses the underscored MCC keywords and
// the 13 canonical examinations. The arrays are private; the const
// evaluation just keeps the names live.
const _ASSERT_KEYWORD_AUTHORITY: [&str; 4] = [
    CANNOT_COMPUTE,
    STATE_SPACE,
    MAX_TOKEN_IN_PLACE,
    MAX_TOKEN_PER_MARKING,
];
const _ASSERT_EXAMINATION_AUTHORITY: [Examination; 13] = Examination::ALL;

// ---------- Permissive key lookups (mirrored from the Python script) ----------

const REPORT_KEYS: &[&str] = &[
    "capability_report",
    "backend_capability_report",
    "backendCapabilityReport",
    "report",
    "capability",
];

const IDENTITY_KEYS: &[&str] = &[
    "model",
    "model_name",
    "case",
    "spec",
    "benchmark",
    "formula",
    "property",
    "problem",
    "run_id",
    "id",
];

const VERDICT_KEYS: &[&str] = &[
    "final_verdict",
    "finalVerdict",
    "verdict",
    "result",
    "outcome",
    "answer",
];

const ROUTING_KEYS: &[&str] = &[
    "production_routing_status",
    "productionRoutingStatus",
    "routing_status",
    "routingStatus",
];

// ---------- Output column ordering (matches the Python script) ----------

const TABLE_COLUMNS: &[&str] = &[
    "row",
    "identity",
    "selected_lanes",
    "rejected_lanes",
    "production_routing_status",
    "unsupported_reason_codes",
    "final_verdict",
];

const ROUTING_COUNT_COLUMNS: &[&str] = &["production_routing_status", "count"];

const LANE_STATUS_COUNT_COLUMNS: &[&str] = &["lane", "backend", "role", "status", "count"];

const REASON_CODE_COUNT_COLUMNS: &[&str] = &["lane", "backend", "reason_code", "count"];

const NATIVE_JIT_GATE_COUNT_COLUMNS: &[&str] = &[
    "backend",
    "feature",
    "feature_enabled",
    "native_requested",
    "strict_requested",
    "parity_enabled",
    "production_selected",
    "fail_closed",
    "reason_code",
    "count",
];

const SYMBOLIC_EXECUTION_COUNT_COLUMNS: &[&str] = &[
    "scope",
    "domain",
    "problem",
    "preferred_backend_code",
    "status_code",
    "reason_code",
    "count",
];

const TRUST_IR_TRANSPORT_IDENTITY_BLOCKER_COUNT_COLUMNS: &[&str] = &[
    "scope",
    "transport",
    "identity",
    "status_code",
    "reason_code",
    "count",
];

const TRUST_CG_ADMISSION_BLOCKER_COUNT_COLUMNS: &[&str] = &[
    "scope",
    "consumer",
    "kind",
    "disposition",
    "rejection_code",
    "reason_code",
    "count",
];

const AY_SOLVER_DECISION_PROFILE_COUNT_COLUMNS: &[&str] = &[
    "scope",
    "backend",
    "decision_code",
    "problem",
    "profile_code",
    "reason_code",
    "count",
];

const TRUST_IR_TRANSPORT_IDENTITY_AVAILABILITY_COUNT_COLUMNS: &[&str] = &[
    "scope",
    "transport",
    "availability",
    "cargo_dependency",
    "schema",
    "schema_version",
    "production_selected",
    "fail_closed",
    "count",
];

const TRUST_CG_NATIVE_ADMISSION_REASON_COUNT_COLUMNS: &[&str] = &[
    "scope",
    "consumer",
    "consumer_mode",
    "kind",
    "surface",
    "disposition",
    "rejection_code",
    "reason_code",
    "requested_authority",
    "install_authority",
    "production_selected",
    "fail_closed",
    "count",
];

const AY_SOLVE_DECISION_PROFILE_AVAILABILITY_COUNT_COLUMNS: &[&str] = &[
    "scope",
    "availability",
    "status_code",
    "decision_code",
    "accepted_for_consumer",
    "consumer_rejection_code",
    "unknown_reason_code",
    "unknown_limit_code",
    "typed_consumer",
    "production_selected",
    "fail_closed",
    "count",
];

const PROOF_REPLAY_BOUNDARY_COUNT_COLUMNS: &[&str] = &[
    "scope",
    "ay_backend_code",
    "safe_proof",
    "safe_replay",
    "unsafe_witness",
    "unsafe_replay",
    "witness_attribution",
    "local_production_gate",
    "native_promotion_gate",
    "production_routing_status_code",
    "count",
];

// ---------- CLI ----------

/// Command-line arguments for the `ty-mcc-summarize-evidence` helper.
#[derive(Parser, Debug)]
#[command(
    name = "ty-mcc-summarize-evidence",
    about = "Summarize backend capability JSONL selected/rejected lanes and aggregate counts.",
    long_about = "Replaces a former Python helper.\n\n\
                  Reads one or more JSONL evidence sidecars and emits an\n\
                  aggregate summary in Markdown (default), CSV, JSON (rows\n\
                  only), or full summary JSON (counts + rows).\n\n\
                  Routes MCC keyword references through `tla_petri::mcc_keywords`\n\
                  and every examination name through\n\
                  `tla_petri::examination::Examination`.",
    arg_required_else_help = true
)]
pub struct Cli {
    /// Input JSONL path(s), or '-' for stdin.
    #[arg(value_name = "JSONL", required = true)]
    pub jsonl: Vec<String>,

    /// Emit CSV instead of Markdown.
    #[arg(long = "csv", conflicts_with_all = ["json", "summary_json"])]
    pub csv: bool,

    /// Emit JSON (rows only) instead of Markdown.
    #[arg(long = "json", conflicts_with_all = ["csv", "summary_json"])]
    pub json: bool,

    /// Emit JSON containing aggregate counts and per-row summaries.
    #[arg(long = "summary-json", conflicts_with_all = ["csv", "json"])]
    pub summary_json: bool,
}

/// Entry point used by the standalone `ty-mcc-summarize-evidence` binary.
pub fn run() -> ExitCode {
    execute(Cli::parse())
}

/// Entry point used by `ty-mccctl summarize-evidence`.
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
    match dispatch(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn dispatch(cli: &Cli) -> Result<(), String> {
    let records = read_jsonl_records(&cli.jsonl).map_err(|e| e.0)?;
    let summary = summarize_evidence(&records);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    if cli.csv {
        write_csv(&summary.rows, &mut out).map_err(|e| e.to_string())?;
    } else if cli.json {
        write_json(&summary.rows, &mut out).map_err(|e| e.to_string())?;
    } else if cli.summary_json {
        write_summary_json(&summary, &mut out).map_err(|e| e.to_string())?;
    } else {
        write_markdown_summary(&summary, &mut out).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ---------- Stringification helpers ----------

/// Mirror Python's `_stringify`: None -> "", bool -> "true"/"false",
/// numbers/strings -> Display, everything else -> stable JSON.
fn stringify(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        other => {
            // Compact JSON with sorted keys; `serde_json::Map` already
            // sorts by key when `preserve_order` is disabled (the
            // default for this workspace), so this matches Python's
            // `json.dumps(..., sort_keys=True, separators=(',', ':'))`.
            serde_json::to_string(other).unwrap_or_else(|_| String::new())
        }
    }
}

fn first_present<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let obj = value.as_object()?;
    for key in keys {
        if let Some(v) = obj.get(*key) {
            if !is_blank(v) {
                return Some(v);
            }
        }
    }
    None
}

fn is_blank(value: &Value) -> bool {
    matches!(value, Value::Null) || matches!(value, Value::String(s) if s.is_empty())
}

/// Mirror Python's `_report_from`: first dict-valued REPORT_KEYS entry,
/// else the row itself.
fn report_from<'a>(row: &'a Value) -> &'a Value {
    if let Some(obj) = row.as_object() {
        for key in REPORT_KEYS {
            if let Some(v) = obj.get(*key) {
                if v.is_object() {
                    return v;
                }
            }
        }
    }
    row
}

// ---------- Lane helpers ----------

fn lane_backend(lane: &Value) -> String {
    if let Value::String(s) = lane {
        return s.clone();
    }
    if !lane.is_object() {
        return stringify(lane);
    }
    let candidate = first_present(
        lane,
        &[
            "backend_code",
            "backendCode",
            "backend",
            "backend_kind",
            "backendKind",
            "kind",
            "lane",
            "name",
        ],
    );
    match candidate {
        Some(v) => stringify(v),
        None => "unknown".to_string(),
    }
}

fn lane_status(lane: &Value) -> String {
    if !lane.is_object() {
        return String::new();
    }
    match first_present(lane, &["status", "availability"]) {
        Some(v) => stringify(v),
        None => String::new(),
    }
}

fn lane_role(lane: &Value) -> String {
    if !lane.is_object() {
        return String::new();
    }
    match first_present(lane, &["role", "policy_role", "policyRole"]) {
        Some(v) => stringify(v),
        None => String::new(),
    }
}

/// Mirror Python's `_reason_code` for nested reason dicts.
fn reason_code(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) if s.is_empty() => String::new(),
        Value::String(s) => s.clone(),
        Value::Object(map) => {
            let direct = first_present(
                value,
                &["code", "reason_code", "reasonCode", "kind", "type"],
            );
            if let Some(v) = direct {
                return stringify(v);
            }
            if map.len() == 1 {
                // Single-key dict — Python's `next(iter(reason))` yields
                // the key itself.
                return map.keys().next().cloned().unwrap_or_default();
            }
            stringify(value)
        }
        other => stringify(other),
    }
}

fn lane_reason_code(lane: &Value) -> String {
    if !lane.is_object() {
        return String::new();
    }
    let direct = first_present(
        lane,
        &[
            "reason_code",
            "reasonCode",
            "unsupported_reason_code",
            "unsupportedReasonCode",
        ],
    );
    if let Some(v) = direct {
        return stringify(v);
    }
    let nested = first_present(lane, &["reason", "unsupported_reason", "unsupportedReason"]);
    match nested {
        Some(v) => reason_code(v),
        None => String::new(),
    }
}

fn format_lane(lane: &Value) -> String {
    let mut parts = vec![lane_backend(lane)];
    for value in [lane_role(lane), lane_status(lane), lane_reason_code(lane)] {
        if !value.is_empty() {
            parts.push(value);
        }
    }
    parts.join(":")
}

fn as_list(value: Option<&Value>) -> Vec<Value> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(arr)) => arr.clone(),
        Some(other) => vec![other.clone()],
    }
}

fn format_lanes(report: &Value, key: &str) -> String {
    let lanes = as_list(report.get(key));
    if lanes.is_empty() {
        return "-".to_string();
    }
    lanes.iter().map(format_lane).collect::<Vec<_>>().join(", ")
}

fn unsupported_reason_codes(row: &Value, report: &Value) -> String {
    let mut explicit = first_present(
        report,
        &[
            "unsupported_reason_codes",
            "unsupportedReasonCodes",
            "reason_codes",
            "reasonCodes",
        ],
    );
    if explicit.is_none() {
        explicit = first_present(
            row,
            &[
                "unsupported_reason_codes",
                "unsupportedReasonCodes",
                "reason_codes",
                "reasonCodes",
            ],
        );
    }

    let mut codes: Vec<String> = Vec::new();
    for value in as_list(explicit) {
        let code = reason_code(&value);
        if !code.is_empty() {
            codes.push(code);
        }
    }
    for lane in as_list(report.get("rejected")) {
        let code = lane_reason_code(&lane);
        if !code.is_empty() {
            codes.push(code);
        }
    }
    if codes.is_empty() {
        return "-".to_string();
    }
    // Preserve first-occurrence order while deduplicating (mirrors
    // Python's `dict.fromkeys(codes)` trick).
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut deduped = Vec::new();
    for code in codes {
        if seen.insert(code.clone()) {
            deduped.push(code);
        }
    }
    deduped.join(", ")
}

fn identity(row: &Value, report: &Value) -> String {
    let row_obj = match row.as_object() {
        Some(o) => o,
        None => return "-".to_string(),
    };
    let report_obj = report.as_object();
    let mut pairs: Vec<String> = Vec::new();
    for key in IDENTITY_KEYS {
        let value = row_obj.get(*key);
        let resolved = match value {
            Some(Value::Null) | None if *key == "problem" => {
                report_obj.and_then(|o| o.get("problem"))
            }
            Some(v) => Some(v),
            None => None,
        };
        if let Some(v) = resolved {
            if !is_blank(v) {
                pairs.push(format!("{}={}", key, stringify(v)));
            }
        }
    }
    if pairs.is_empty() {
        "-".to_string()
    } else {
        pairs.join("; ")
    }
}

// ---------- Summary row construction ----------

#[derive(Debug, Clone)]
struct SummaryRow {
    row: usize,
    identity: String,
    selected_lanes: String,
    rejected_lanes: String,
    production_routing_status: String,
    unsupported_reason_codes: String,
    final_verdict: String,
}

impl SummaryRow {
    fn get(&self, column: &str) -> Value {
        match column {
            "row" => Value::from(self.row),
            "identity" => Value::String(self.identity.clone()),
            "selected_lanes" => Value::String(self.selected_lanes.clone()),
            "rejected_lanes" => Value::String(self.rejected_lanes.clone()),
            "production_routing_status" => Value::String(self.production_routing_status.clone()),
            "unsupported_reason_codes" => Value::String(self.unsupported_reason_codes.clone()),
            "final_verdict" => Value::String(self.final_verdict.clone()),
            _ => Value::Null,
        }
    }

    fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("row".to_string(), Value::from(self.row));
        map.insert("identity".to_string(), Value::String(self.identity.clone()));
        map.insert(
            "selected_lanes".to_string(),
            Value::String(self.selected_lanes.clone()),
        );
        map.insert(
            "rejected_lanes".to_string(),
            Value::String(self.rejected_lanes.clone()),
        );
        map.insert(
            "production_routing_status".to_string(),
            Value::String(self.production_routing_status.clone()),
        );
        map.insert(
            "unsupported_reason_codes".to_string(),
            Value::String(self.unsupported_reason_codes.clone()),
        );
        map.insert(
            "final_verdict".to_string(),
            Value::String(self.final_verdict.clone()),
        );
        Value::Object(map)
    }
}

fn summarize_row(row_number: usize, row: &Value) -> SummaryRow {
    let report = report_from(row);
    let mut routing = first_present(report, ROUTING_KEYS).cloned();
    if routing.is_none() {
        routing = first_present(row, ROUTING_KEYS).cloned();
    }
    let mut verdict = first_present(row, VERDICT_KEYS).cloned();
    if verdict.is_none() {
        verdict = first_present(report, VERDICT_KEYS).cloned();
    }
    let routing_text = routing
        .as_ref()
        .map(stringify)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "-".to_string());
    let verdict_text = verdict
        .as_ref()
        .map(stringify)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "-".to_string());

    SummaryRow {
        row: row_number,
        identity: identity(row, report),
        selected_lanes: format_lanes(report, "selected"),
        rejected_lanes: format_lanes(report, "rejected"),
        production_routing_status: routing_text,
        unsupported_reason_codes: unsupported_reason_codes(row, report),
        final_verdict: verdict_text,
    }
}

// ---------- Aggregate counts ----------

/// Result of `summarize_evidence`. Each `*_counts` field is sorted by
/// descending count, then ascending key-tuple — matching Python's
/// `sorted(counter.items(), key=lambda item: (-item[1], item[0]))`.
struct EvidenceSummary {
    rows: Vec<SummaryRow>,
    production_routing_status_counts: Vec<(Vec<String>, u64)>,
    backend_lane_status_counts: Vec<(Vec<String>, u64)>,
    reason_code_counts: Vec<(Vec<String>, u64)>,
    native_jit_fail_closed_gate_counts: Vec<(Vec<String>, u64)>,
    symbolic_execution_counts: Vec<(Vec<String>, u64)>,
    trust_ir_transport_identity_blocker_counts: Vec<(Vec<String>, u64)>,
    trust_cg_admission_blocker_counts: Vec<(Vec<String>, u64)>,
    ay_solver_decision_profile_counts: Vec<(Vec<String>, u64)>,
    trust_ir_transport_identity_availability_counts: Vec<(Vec<String>, u64)>,
    trust_cg_native_admission_reason_counts: Vec<(Vec<String>, u64)>,
    ay_solve_decision_profile_availability_counts: Vec<(Vec<String>, u64)>,
    proof_replay_boundary_counts: Vec<(Vec<String>, u64)>,
}

fn count_value_with_default(value: Option<&Value>, default: &str) -> String {
    let text = match value {
        Some(v) => stringify(v),
        None => String::new(),
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    }
}

fn count_value(value: Option<&Value>) -> String {
    count_value_with_default(value, "-")
}

fn sorted_counter(counter: BTreeMap<Vec<String>, u64>) -> Vec<(Vec<String>, u64)> {
    let mut items: Vec<(Vec<String>, u64)> = counter.into_iter().collect();
    // Sort by descending count, then ascending key tuple.
    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    items
}

// ---------- Evidence-row parsing (whitespace split, key=value pairs) ----------

fn parse_kv_evidence(line: &str) -> BTreeMap<String, String> {
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    for token in line.split_ascii_whitespace() {
        if let Some((key, value)) = token.split_once('=') {
            if !key.is_empty() {
                values.insert(key.to_string(), value.to_string());
            }
        }
    }
    values
}

fn evidence_strings(report: &Value) -> Vec<String> {
    as_list(report.get("evidence"))
        .into_iter()
        .filter_map(|v| match v {
            Value::String(s) => Some(s),
            _ => None,
        })
        .collect()
}

/// For each evidence string that contains any of the listed marker
/// tokens (whitespace-delimited), yield a parsed map. Adds
/// `scope` (tokens before the marker) and `status` (token after the
/// marker, if no `=`) as defaults if not already present. Mirrors
/// Python `_evidence_rows_with_marker`.
fn evidence_rows_with_marker(report: &Value, markers: &[&str]) -> Vec<BTreeMap<String, String>> {
    let mut rows = Vec::new();
    for evidence in evidence_strings(report) {
        let tokens: Vec<&str> = evidence.split_ascii_whitespace().collect();
        for marker in markers {
            if let Some(marker_index) = tokens.iter().position(|t| t == marker) {
                let mut fields = parse_kv_evidence(&evidence);
                let scope = tokens[..marker_index].join(" ");
                fields.entry("scope".to_string()).or_insert(scope);
                if marker_index + 1 < tokens.len() {
                    let next = tokens[marker_index + 1];
                    if !next.contains('=') {
                        fields
                            .entry("status".to_string())
                            .or_insert_with(|| next.to_string());
                    }
                }
                rows.push(fields);
                break;
            }
        }
    }
    rows
}

/// Native-JIT fail-closed gate rows: must contain the literal
/// `"native_jit fail_closed_gate"` substring.
fn native_jit_fail_closed_gates(report: &Value) -> Vec<BTreeMap<String, String>> {
    let mut rows = Vec::new();
    for evidence in evidence_strings(report) {
        if !evidence.contains("native_jit fail_closed_gate") {
            continue;
        }
        let mut fields = parse_kv_evidence(&evidence);
        fields
            .entry("backend".to_string())
            .or_insert_with(|| "native_kernel".to_string());
        rows.push(fields);
    }
    rows
}

fn symbolic_execution_evidence_rows(report: &Value) -> Vec<BTreeMap<String, String>> {
    evidence_rows_with_marker(report, &["symbolic_execution"])
}

/// trust-ir transport identity blocker rows. Skip rows whose marker is
/// `trust_ir_transport_identity` (those are availability rows, handled
/// separately).
fn trust_ir_transport_identity_blocker_rows(report: &Value) -> Vec<BTreeMap<String, String>> {
    let mut rows = Vec::new();
    for evidence in evidence_strings(report) {
        let tokens: Vec<&str> = evidence.split_ascii_whitespace().collect();
        let mut marker_index: Option<usize> = None;
        let mut matched_marker: Option<&str> = None;
        for marker in &[
            "trust_ir_transport_identity_blocker",
            "trust_ir_transport_identity",
        ] {
            if let Some(idx) = tokens.iter().position(|t| t == marker) {
                marker_index = Some(idx);
                matched_marker = Some(*marker);
                break;
            }
        }
        let (idx, matched) = match (marker_index, matched_marker) {
            (Some(i), Some(m)) => (i, m),
            _ => continue,
        };
        if matched == "trust_ir_transport_identity" {
            continue;
        }

        let mut fields = parse_kv_evidence(&evidence);
        let prefix: &[&str] = &tokens[..idx];
        let mut scope_tokens: Vec<&str> = prefix.to_vec();
        if !fields.contains_key("transport") {
            if let Some(last) = prefix.last() {
                if matches!(*last, "native" | "jit" | "native_jit") {
                    fields.insert("transport".to_string(), (*last).to_string());
                    scope_tokens = prefix[..prefix.len() - 1].to_vec();
                }
            }
        }
        fields
            .entry("scope".to_string())
            .or_insert_with(|| scope_tokens.join(" "));
        if idx + 1 < tokens.len() {
            let next = tokens[idx + 1];
            if !next.contains('=') {
                fields
                    .entry("status_code".to_string())
                    .or_insert_with(|| next.to_string());
            }
        }
        let identity = first_present_map(
            &fields,
            &[
                "identity",
                "identity_code",
                "identityCode",
                "api",
                "required_api",
                "requiredApi",
            ],
        );
        if let Some(v) = identity {
            fields.insert("identity".to_string(), v);
        }
        rows.push(fields);
    }
    rows
}

fn trust_ir_transport_identity_availability_rows(report: &Value) -> Vec<BTreeMap<String, String>> {
    let mut rows = Vec::new();
    for evidence in evidence_strings(report) {
        let tokens: Vec<&str> = evidence.split_ascii_whitespace().collect();
        let idx = match tokens
            .iter()
            .position(|t| *t == "trust_ir_transport_identity")
        {
            Some(i) => i,
            None => continue,
        };
        let mut fields = parse_kv_evidence(&evidence);
        let prefix: &[&str] = &tokens[..idx];
        let mut scope_tokens: Vec<&str> = prefix.to_vec();
        if !fields.contains_key("transport") {
            if let Some(last) = prefix.last() {
                if matches!(*last, "native" | "jit" | "native_jit") {
                    fields.insert("transport".to_string(), (*last).to_string());
                    scope_tokens = prefix[..prefix.len() - 1].to_vec();
                }
            }
        }
        fields
            .entry("scope".to_string())
            .or_insert_with(|| scope_tokens.join(" "));
        if idx + 1 < tokens.len() {
            let next = tokens[idx + 1];
            if !next.contains('=') {
                fields
                    .entry("status_code".to_string())
                    .or_insert_with(|| next.to_string());
            }
        }
        rows.push(fields);
    }
    rows
}

fn trust_cg_admission_blocker_rows(report: &Value) -> Vec<BTreeMap<String, String>> {
    evidence_rows_with_marker(report, &["trust_cg_admission_blocker"])
}

/// Subset of `trust_cg_admission_blocker_rows`: only rows tagged with the
/// `NativeInstallGateAdmissionSummary` source or a `schema` field.
fn trust_cg_native_admission_reason_rows(report: &Value) -> Vec<BTreeMap<String, String>> {
    evidence_rows_with_marker(report, &["trust_cg_admission_blocker"])
        .into_iter()
        .filter(|row| {
            row.get("source").map(String::as_str) == Some("NativeInstallGateAdmissionSummary")
                || row.contains_key("schema")
        })
        .collect()
}

fn ay_solver_decision_profile_rows(report: &Value) -> Vec<BTreeMap<String, String>> {
    evidence_rows_with_marker(
        report,
        &[
            "ay_solver_decision_profile",
            "ay_solver_decision_profile_summary",
        ],
    )
}

fn ay_solve_decision_profile_availability_rows(report: &Value) -> Vec<BTreeMap<String, String>> {
    evidence_rows_with_marker(report, &["ay_solver_decision_profile_summary"])
}

fn proof_replay_boundary_rows(report: &Value) -> Vec<BTreeMap<String, String>> {
    evidence_rows_with_marker(report, &["proof_replay_boundary"])
}

fn first_present_map(map: &BTreeMap<String, String>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(v) = map.get(*key) {
            if !v.is_empty() {
                return Some(v.clone());
            }
        }
    }
    None
}

fn count_field(map: &BTreeMap<String, String>, keys: &[&str]) -> String {
    let raw = first_present_map(map, keys);
    let text = raw.unwrap_or_default();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        "-".to_string()
    } else {
        trimmed.to_string()
    }
}

fn count_field_with_default(
    map: &BTreeMap<String, String>,
    keys: &[&str],
    default: &str,
) -> String {
    let raw = first_present_map(map, keys);
    let text = raw.unwrap_or_default();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    }
}

fn count_single(map: &BTreeMap<String, String>, key: &str) -> String {
    count_field(map, &[key])
}

fn count_single_with_default(map: &BTreeMap<String, String>, key: &str, default: &str) -> String {
    count_field_with_default(map, &[key], default)
}

fn explicit_reason_codes(row: &Value, report: &Value) -> Vec<String> {
    let mut explicit = first_present(
        report,
        &[
            "unsupported_reason_codes",
            "unsupportedReasonCodes",
            "reason_codes",
            "reasonCodes",
        ],
    );
    if explicit.is_none() {
        explicit = first_present(
            row,
            &[
                "unsupported_reason_codes",
                "unsupportedReasonCodes",
                "reason_codes",
                "reasonCodes",
            ],
        );
    }
    let mut codes = Vec::new();
    for value in as_list(explicit) {
        let code = reason_code(&value);
        if !code.is_empty() {
            codes.push(code);
        }
    }
    codes
}

fn transport_identity_availability(value: Option<&str>) -> String {
    let raw = value.unwrap_or("");
    let lower = raw.to_ascii_lowercase();
    if matches!(lower.as_str(), "unavailable" | "missing") {
        return "missing".to_string();
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "-".to_string()
    } else {
        trimmed.to_string()
    }
}

fn ay_profile_availability(evidence: &BTreeMap<String, String>) -> String {
    let status =
        count_field(evidence, &["status_code", "statusCode", "status"]).to_ascii_lowercase();
    let decision =
        count_field(evidence, &["decision_code", "decisionCode", "decision"]).to_ascii_lowercase();
    let typed_consumer =
        count_field(evidence, &["typed_consumer", "typedConsumer"]).to_ascii_lowercase();

    if status == "missing_typed_summary" || status == "unavailable" {
        return "missing".to_string();
    }
    if typed_consumer == "false" {
        return "missing".to_string();
    }
    if decision == "unknown" {
        return "unknown".to_string();
    }
    if status == "typed_summary_available" || typed_consumer == "true" {
        return "typed".to_string();
    }
    // Python falls back to `_count_value(status)`; status was lowered
    // here, but `_count_value` does not lower-case its input. Re-derive
    // the raw status using the same key precedence.
    let raw =
        first_present_map(evidence, &["status_code", "statusCode", "status"]).unwrap_or_default();
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "-".to_string()
    } else {
        trimmed.to_string()
    }
}

// ---------- The big summary builder ----------

fn summarize_evidence(records: &[(usize, Value)]) -> EvidenceSummary {
    let mut summary_rows: Vec<SummaryRow> = Vec::new();
    let mut routing_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut lane_status_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut reason_code_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut native_jit_gate_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut symbolic_execution_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut trust_ir_blocker_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut trust_cg_blocker_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut ay_decision_profile_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut trust_ir_availability_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut trust_cg_native_admission_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut ay_profile_availability_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();
    let mut proof_replay_counts: BTreeMap<Vec<String>, u64> = BTreeMap::new();

    for (row_number, row) in records {
        let report = report_from(row);
        summary_rows.push(summarize_row(*row_number, row));

        let mut routing = first_present(report, ROUTING_KEYS).cloned();
        if routing.is_none() {
            routing = first_present(row, ROUTING_KEYS).cloned();
        }
        let routing_key = count_value(routing.as_ref());
        *routing_counts.entry(vec![routing_key]).or_insert(0) += 1;

        for lane_name in ["selected", "rejected"] {
            for lane in as_list(report.get(lane_name)) {
                let backend = count_value(Some(&Value::String(lane_backend(&lane))));
                let role = count_value(Some(&Value::String(lane_role(&lane))));
                let status = count_value(Some(&Value::String(lane_status(&lane))));
                *lane_status_counts
                    .entry(vec![lane_name.to_string(), backend.clone(), role, status])
                    .or_insert(0) += 1;

                let reason = lane_reason_code(&lane);
                if !reason.is_empty() || lane_name == "rejected" {
                    let reason_key = count_value(Some(&Value::String(reason)));
                    *reason_code_counts
                        .entry(vec![lane_name.to_string(), backend, reason_key])
                        .or_insert(0) += 1;
                }
            }
        }

        for reason in explicit_reason_codes(row, report) {
            let reason_key = count_value(Some(&Value::String(reason)));
            *reason_code_counts
                .entry(vec!["report".to_string(), "-".to_string(), reason_key])
                .or_insert(0) += 1;
        }

        for gate in native_jit_fail_closed_gates(report) {
            *native_jit_gate_counts
                .entry(vec![
                    count_single_with_default(&gate, "backend", "native_kernel"),
                    count_single(&gate, "feature"),
                    count_single(&gate, "feature_enabled"),
                    count_single(&gate, "native_requested"),
                    count_single(&gate, "strict_requested"),
                    count_single(&gate, "parity_enabled"),
                    count_single(&gate, "production_selected"),
                    count_single(&gate, "fail_closed"),
                    count_single(&gate, "reason_code"),
                ])
                .or_insert(0) += 1;
        }

        for evidence in symbolic_execution_evidence_rows(report) {
            *symbolic_execution_counts
                .entry(vec![
                    count_single(&evidence, "scope"),
                    count_field(&evidence, &["domain", "backend_domain", "backendDomain"]),
                    count_single(&evidence, "problem"),
                    count_field(
                        &evidence,
                        &[
                            "preferred_backend_code",
                            "preferredBackendCode",
                            "backend_code",
                            "backendCode",
                        ],
                    ),
                    count_field(&evidence, &["status_code", "statusCode", "status"]),
                    count_field(&evidence, &["reason_code", "reasonCode", "reason"]),
                ])
                .or_insert(0) += 1;
        }

        for blocker in trust_ir_transport_identity_blocker_rows(report) {
            *trust_ir_blocker_counts
                .entry(vec![
                    count_single(&blocker, "scope"),
                    count_field(&blocker, &["transport", "transport_code", "transportCode"]),
                    count_field(&blocker, &["identity", "identity_code", "identityCode"]),
                    count_field(
                        &blocker,
                        &["status_code", "statusCode", "status", "availability"],
                    ),
                    count_field(
                        &blocker,
                        &[
                            "reason_code",
                            "reasonCode",
                            "blocker_code",
                            "blockerCode",
                            "unsupported_mode_code",
                            "unsupportedModeCode",
                        ],
                    ),
                ])
                .or_insert(0) += 1;
        }

        for blocker in trust_cg_admission_blocker_rows(report) {
            if blocker.get("source").map(String::as_str)
                == Some("NativeInstallGateAdmissionSummary")
                || blocker.contains_key("schema")
            {
                continue;
            }
            *trust_cg_blocker_counts
                .entry(vec![
                    count_single(&blocker, "scope"),
                    count_field(&blocker, &["consumer", "consumer_code", "consumerCode"]),
                    count_field(
                        &blocker,
                        &[
                            "kind",
                            "artifact_kind",
                            "artifactKind",
                            "kernel_kind",
                            "kernelKind",
                            "surface",
                        ],
                    ),
                    count_field(
                        &blocker,
                        &[
                            "disposition",
                            "disposition_code",
                            "dispositionCode",
                            "status_code",
                            "statusCode",
                            "status",
                        ],
                    ),
                    count_field(
                        &blocker,
                        &[
                            "rejection_code",
                            "rejectionCode",
                            "consumer_rejection_code",
                            "consumerRejectionCode",
                            "proof_rejection_code",
                            "proofRejectionCode",
                            "reason_code",
                            "reasonCode",
                        ],
                    ),
                    count_field(
                        &blocker,
                        &[
                            "reason_code",
                            "reasonCode",
                            "proof_rejection_code",
                            "proofRejectionCode",
                            "rejection_code",
                            "rejectionCode",
                        ],
                    ),
                ])
                .or_insert(0) += 1;
        }

        for evidence in trust_ir_transport_identity_availability_rows(report) {
            let availability_raw = first_present_map(
                &evidence,
                &["availability", "status_code", "statusCode", "status"],
            );
            let availability = transport_identity_availability(availability_raw.as_deref());
            *trust_ir_availability_counts
                .entry(vec![
                    count_single(&evidence, "scope"),
                    count_field(&evidence, &["transport", "transport_code", "transportCode"]),
                    availability,
                    count_field(&evidence, &["cargo_dependency", "cargoDependency"]),
                    count_single(&evidence, "schema"),
                    count_field(&evidence, &["schema_version", "schemaVersion"]),
                    count_field(&evidence, &["production_selected", "productionSelected"]),
                    count_field(&evidence, &["fail_closed", "failClosed"]),
                ])
                .or_insert(0) += 1;
        }

        for evidence in trust_cg_native_admission_reason_rows(report) {
            *trust_cg_native_admission_counts
                .entry(vec![
                    count_single(&evidence, "scope"),
                    count_field(&evidence, &["consumer", "consumer_code", "consumerCode"]),
                    count_field(&evidence, &["consumer_mode", "consumerMode"]),
                    count_field(&evidence, &["kind", "artifact_kind", "artifactKind"]),
                    count_single(&evidence, "surface"),
                    count_field(
                        &evidence,
                        &[
                            "disposition",
                            "disposition_code",
                            "dispositionCode",
                            "status_code",
                            "statusCode",
                        ],
                    ),
                    count_field(
                        &evidence,
                        &[
                            "rejection_code",
                            "rejectionCode",
                            "reason_code",
                            "reasonCode",
                        ],
                    ),
                    count_field(
                        &evidence,
                        &[
                            "reason_code",
                            "reasonCode",
                            "rejection_code",
                            "rejectionCode",
                        ],
                    ),
                    count_field(&evidence, &["requested_authority", "requestedAuthority"]),
                    count_field(&evidence, &["install_authority", "installAuthority"]),
                    count_field(&evidence, &["production_selected", "productionSelected"]),
                    count_field(&evidence, &["fail_closed", "failClosed"]),
                ])
                .or_insert(0) += 1;
        }

        for evidence in ay_solve_decision_profile_availability_rows(report) {
            *ay_profile_availability_counts
                .entry(vec![
                    count_single(&evidence, "scope"),
                    ay_profile_availability(&evidence),
                    count_field(&evidence, &["status_code", "statusCode", "status"]),
                    count_field(&evidence, &["decision_code", "decisionCode", "decision"]),
                    count_field(&evidence, &["accepted_for_consumer", "acceptedForConsumer"]),
                    count_field(
                        &evidence,
                        &["consumer_rejection_code", "consumerRejectionCode"],
                    ),
                    count_field(&evidence, &["unknown_reason_code", "unknownReasonCode"]),
                    count_field(&evidence, &["unknown_limit_code", "unknownLimitCode"]),
                    count_field(&evidence, &["typed_consumer", "typedConsumer"]),
                    count_field(&evidence, &["production_selected", "productionSelected"]),
                    count_field(&evidence, &["fail_closed", "failClosed"]),
                ])
                .or_insert(0) += 1;
        }

        for evidence in ay_solver_decision_profile_rows(report) {
            *ay_decision_profile_counts
                .entry(vec![
                    count_single(&evidence, "scope"),
                    count_field_with_default(
                        &evidence,
                        &[
                            "backend",
                            "backend_code",
                            "backendCode",
                            "solver",
                            "solver_code",
                            "solverCode",
                        ],
                        "ay",
                    ),
                    count_field(
                        &evidence,
                        &[
                            "decision_code",
                            "decisionCode",
                            "action",
                            "decision",
                            "status_code",
                            "statusCode",
                            "status",
                        ],
                    ),
                    count_field(&evidence, &["problem", "problem_code", "problemCode"]),
                    count_field(
                        &evidence,
                        &[
                            "profile_code",
                            "profileCode",
                            "route_profile",
                            "routeProfile",
                            "profile",
                            "expected_schema",
                            "expectedSchema",
                        ],
                    ),
                    count_field(
                        &evidence,
                        &[
                            "reason_code",
                            "reasonCode",
                            "consumer_rejection_code",
                            "consumerRejectionCode",
                            "unknown_reason_code",
                            "unknownReasonCode",
                            "status_code",
                            "statusCode",
                            "reason",
                        ],
                    ),
                ])
                .or_insert(0) += 1;
        }

        for boundary in proof_replay_boundary_rows(report) {
            *proof_replay_counts
                .entry(vec![
                    count_single(&boundary, "scope"),
                    count_field(
                        &boundary,
                        &[
                            "ay_backend_code",
                            "ayBackendCode",
                            "ay_backend",
                            "ayBackend",
                            "backend_code",
                            "backendCode",
                            "backend",
                        ],
                    ),
                    count_field(
                        &boundary,
                        &[
                            "safe_proof",
                            "safeProof",
                            "proof_safety_code",
                            "proofSafetyCode",
                            "proof_safety",
                            "proofSafety",
                        ],
                    ),
                    count_field(
                        &boundary,
                        &[
                            "safe_replay",
                            "safeReplay",
                            "replay_availability_code",
                            "replayAvailabilityCode",
                            "proof_replay",
                            "proofReplay",
                        ],
                    ),
                    count_field(
                        &boundary,
                        &[
                            "unsafe_witness",
                            "unsafeWitness",
                            "witness_code",
                            "witnessCode",
                        ],
                    ),
                    count_field(
                        &boundary,
                        &[
                            "unsafe_replay",
                            "unsafeReplay",
                            "witness_replay",
                            "witnessReplay",
                        ],
                    ),
                    count_field(
                        &boundary,
                        &[
                            "witness_attribution",
                            "witnessAttribution",
                            "witness_attribution_code",
                            "witnessAttributionCode",
                        ],
                    ),
                    count_field(
                        &boundary,
                        &[
                            "local_production_gate",
                            "localProductionGate",
                            "local_gate",
                            "localGate",
                        ],
                    ),
                    count_field(
                        &boundary,
                        &[
                            "native_promotion_gate",
                            "nativePromotionGate",
                            "native_gate",
                            "nativeGate",
                        ],
                    ),
                    count_field(
                        &boundary,
                        &[
                            "production_routing_status_code",
                            "productionRoutingStatusCode",
                            "routing_status_code",
                            "routingStatusCode",
                            "production_routing_status",
                            "productionRoutingStatus",
                        ],
                    ),
                ])
                .or_insert(0) += 1;
        }
    }

    EvidenceSummary {
        rows: summary_rows,
        production_routing_status_counts: sorted_counter(routing_counts),
        backend_lane_status_counts: sorted_counter(lane_status_counts),
        reason_code_counts: sorted_counter(reason_code_counts),
        native_jit_fail_closed_gate_counts: sorted_counter(native_jit_gate_counts),
        symbolic_execution_counts: sorted_counter(symbolic_execution_counts),
        trust_ir_transport_identity_blocker_counts: sorted_counter(trust_ir_blocker_counts),
        trust_cg_admission_blocker_counts: sorted_counter(trust_cg_blocker_counts),
        ay_solver_decision_profile_counts: sorted_counter(ay_decision_profile_counts),
        trust_ir_transport_identity_availability_counts: sorted_counter(
            trust_ir_availability_counts,
        ),
        trust_cg_native_admission_reason_counts: sorted_counter(trust_cg_native_admission_counts),
        ay_solve_decision_profile_availability_counts: sorted_counter(
            ay_profile_availability_counts,
        ),
        proof_replay_boundary_counts: sorted_counter(proof_replay_counts),
    }
}

// ---------- Output writers ----------

fn markdown_escape_value(value: &Value) -> String {
    let text = stringify(value);
    text.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', " ")
}

fn write_markdown_header<W: Write>(columns: &[&str], out: &mut W) -> io::Result<()> {
    write!(out, "| ")?;
    for (i, col) in columns.iter().enumerate() {
        if i > 0 {
            write!(out, " | ")?;
        }
        write!(out, "{}", col)?;
    }
    writeln!(out, " |")?;
    write!(out, "| ")?;
    for i in 0..columns.len() {
        if i > 0 {
            write!(out, " | ")?;
        }
        write!(out, "---")?;
    }
    writeln!(out, " |")
}

fn write_markdown_summary_rows<W: Write>(rows: &[SummaryRow], out: &mut W) -> io::Result<()> {
    write_markdown_header(TABLE_COLUMNS, out)?;
    for row in rows {
        write!(out, "| ")?;
        for (i, col) in TABLE_COLUMNS.iter().enumerate() {
            if i > 0 {
                write!(out, " | ")?;
            }
            write!(out, "{}", markdown_escape_value(&row.get(col)))?;
        }
        writeln!(out, " |")?;
    }
    Ok(())
}

fn write_markdown_count_table<W: Write>(
    columns: &[&str],
    rows: &[(Vec<String>, u64)],
    out: &mut W,
) -> io::Result<()> {
    write_markdown_header(columns, out)?;
    for (key, count) in rows {
        write!(out, "| ")?;
        for (i, col) in columns.iter().enumerate() {
            if i > 0 {
                write!(out, " | ")?;
            }
            if *col == "count" {
                write!(out, "{}", count)?;
            } else {
                let idx = i;
                let v = Value::String(key.get(idx).cloned().unwrap_or_default());
                write!(out, "{}", markdown_escape_value(&v))?;
            }
        }
        writeln!(out, " |")?;
    }
    Ok(())
}

fn write_markdown_summary<W: Write>(summary: &EvidenceSummary, out: &mut W) -> io::Result<()> {
    writeln!(out, "### Production routing status counts")?;
    write_markdown_count_table(
        ROUTING_COUNT_COLUMNS,
        &summary.production_routing_status_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### Backend lane status counts")?;
    write_markdown_count_table(
        LANE_STATUS_COUNT_COLUMNS,
        &summary.backend_lane_status_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### Reason code counts")?;
    write_markdown_count_table(REASON_CODE_COUNT_COLUMNS, &summary.reason_code_counts, out)?;
    writeln!(out)?;
    writeln!(out, "### Native JIT fail-closed gate counts")?;
    write_markdown_count_table(
        NATIVE_JIT_GATE_COUNT_COLUMNS,
        &summary.native_jit_fail_closed_gate_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### Symbolic execution counts")?;
    write_markdown_count_table(
        SYMBOLIC_EXECUTION_COUNT_COLUMNS,
        &summary.symbolic_execution_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### trust-ir transport identity blocker counts")?;
    write_markdown_count_table(
        TRUST_IR_TRANSPORT_IDENTITY_BLOCKER_COUNT_COLUMNS,
        &summary.trust_ir_transport_identity_blocker_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### trust-codegen admission blocker counts")?;
    write_markdown_count_table(
        TRUST_CG_ADMISSION_BLOCKER_COUNT_COLUMNS,
        &summary.trust_cg_admission_blocker_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### AY solver decision/profile counts")?;
    write_markdown_count_table(
        AY_SOLVER_DECISION_PROFILE_COUNT_COLUMNS,
        &summary.ay_solver_decision_profile_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### trust-ir transport identity availability counts")?;
    write_markdown_count_table(
        TRUST_IR_TRANSPORT_IDENTITY_AVAILABILITY_COUNT_COLUMNS,
        &summary.trust_ir_transport_identity_availability_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### trust-codegen native admission reason counts")?;
    write_markdown_count_table(
        TRUST_CG_NATIVE_ADMISSION_REASON_COUNT_COLUMNS,
        &summary.trust_cg_native_admission_reason_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### AY solve-decision profile availability counts")?;
    write_markdown_count_table(
        AY_SOLVE_DECISION_PROFILE_AVAILABILITY_COUNT_COLUMNS,
        &summary.ay_solve_decision_profile_availability_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### Proof/replay boundary counts")?;
    write_markdown_count_table(
        PROOF_REPLAY_BOUNDARY_COUNT_COLUMNS,
        &summary.proof_replay_boundary_counts,
        out,
    )?;
    writeln!(out)?;
    writeln!(out, "### Evidence rows")?;
    write_markdown_summary_rows(&summary.rows, out)
}

/// CSV writer matching Python's `csv.DictWriter` default dialect:
/// comma-separated, double quotes around any field containing comma,
/// quote, CR, or LF, with internal quotes doubled, and `\r\n` line
/// terminators. Fields without those special characters are emitted raw.
fn csv_escape(field: &str) -> String {
    let needs_quote = field
        .as_bytes()
        .iter()
        .any(|&b| matches!(b, b',' | b'"' | b'\r' | b'\n'));
    if !needs_quote {
        return field.to_string();
    }
    let mut out = String::with_capacity(field.len() + 2);
    out.push('"');
    for ch in field.chars() {
        if ch == '"' {
            out.push('"');
            out.push('"');
        } else {
            out.push(ch);
        }
    }
    out.push('"');
    out
}

fn write_csv<W: Write>(rows: &[SummaryRow], out: &mut W) -> io::Result<()> {
    for (i, col) in TABLE_COLUMNS.iter().enumerate() {
        if i > 0 {
            out.write_all(b",")?;
        }
        out.write_all(csv_escape(col).as_bytes())?;
    }
    out.write_all(b"\r\n")?;
    for row in rows {
        for (i, col) in TABLE_COLUMNS.iter().enumerate() {
            if i > 0 {
                out.write_all(b",")?;
            }
            let value = row.get(col);
            let cell = stringify(&value);
            out.write_all(csv_escape(&cell).as_bytes())?;
        }
        out.write_all(b"\r\n")?;
    }
    Ok(())
}

fn write_json<W: Write>(rows: &[SummaryRow], out: &mut W) -> io::Result<()> {
    let values: Vec<Value> = rows.iter().map(SummaryRow::to_json).collect();
    serde_json::to_writer_pretty(&mut *out, &Value::Array(values)).map_err(io::Error::other)?;
    out.write_all(b"\n")
}

fn count_row_to_json(columns: &[&str], key: &[String], count: u64) -> Value {
    let mut map = Map::new();
    map.insert("count".to_string(), Value::from(count));
    // Columns alternate non-count keys with the trailing "count" entry.
    for (i, col) in columns.iter().enumerate() {
        if *col == "count" {
            continue;
        }
        let raw = key.get(i).cloned().unwrap_or_default();
        map.insert((*col).to_string(), Value::String(raw));
    }
    Value::Object(map)
}

fn count_rows_to_json(columns: &[&str], rows: &[(Vec<String>, u64)]) -> Value {
    let entries: Vec<Value> = rows
        .iter()
        .map(|(key, count)| count_row_to_json(columns, key, *count))
        .collect();
    Value::Array(entries)
}

fn summary_to_json(summary: &EvidenceSummary) -> Value {
    let mut counts = Map::new();
    counts.insert(
        "production_routing_status".to_string(),
        count_rows_to_json(
            ROUTING_COUNT_COLUMNS,
            &summary.production_routing_status_counts,
        ),
    );
    counts.insert(
        "backend_lane_status".to_string(),
        count_rows_to_json(
            LANE_STATUS_COUNT_COLUMNS,
            &summary.backend_lane_status_counts,
        ),
    );
    counts.insert(
        "reason_code".to_string(),
        count_rows_to_json(REASON_CODE_COUNT_COLUMNS, &summary.reason_code_counts),
    );
    counts.insert(
        "native_jit_fail_closed_gate".to_string(),
        count_rows_to_json(
            NATIVE_JIT_GATE_COUNT_COLUMNS,
            &summary.native_jit_fail_closed_gate_counts,
        ),
    );
    counts.insert(
        "symbolic_execution".to_string(),
        count_rows_to_json(
            SYMBOLIC_EXECUTION_COUNT_COLUMNS,
            &summary.symbolic_execution_counts,
        ),
    );
    counts.insert(
        "trust_ir_transport_identity_blocker".to_string(),
        count_rows_to_json(
            TRUST_IR_TRANSPORT_IDENTITY_BLOCKER_COUNT_COLUMNS,
            &summary.trust_ir_transport_identity_blocker_counts,
        ),
    );
    counts.insert(
        "trust_cg_admission_blocker".to_string(),
        count_rows_to_json(
            TRUST_CG_ADMISSION_BLOCKER_COUNT_COLUMNS,
            &summary.trust_cg_admission_blocker_counts,
        ),
    );
    counts.insert(
        "ay_solver_decision_profile".to_string(),
        count_rows_to_json(
            AY_SOLVER_DECISION_PROFILE_COUNT_COLUMNS,
            &summary.ay_solver_decision_profile_counts,
        ),
    );
    counts.insert(
        "trust_ir_transport_identity_availability".to_string(),
        count_rows_to_json(
            TRUST_IR_TRANSPORT_IDENTITY_AVAILABILITY_COUNT_COLUMNS,
            &summary.trust_ir_transport_identity_availability_counts,
        ),
    );
    counts.insert(
        "trust_cg_native_admission_reason".to_string(),
        count_rows_to_json(
            TRUST_CG_NATIVE_ADMISSION_REASON_COUNT_COLUMNS,
            &summary.trust_cg_native_admission_reason_counts,
        ),
    );
    counts.insert(
        "ay_solve_decision_profile_availability".to_string(),
        count_rows_to_json(
            AY_SOLVE_DECISION_PROFILE_AVAILABILITY_COUNT_COLUMNS,
            &summary.ay_solve_decision_profile_availability_counts,
        ),
    );
    counts.insert(
        "proof_replay_boundary".to_string(),
        count_rows_to_json(
            PROOF_REPLAY_BOUNDARY_COUNT_COLUMNS,
            &summary.proof_replay_boundary_counts,
        ),
    );

    let mut root = Map::new();
    root.insert("counts".to_string(), Value::Object(counts));
    root.insert(
        "rows".to_string(),
        Value::Array(summary.rows.iter().map(SummaryRow::to_json).collect()),
    );
    Value::Object(root)
}

fn write_summary_json<W: Write>(summary: &EvidenceSummary, out: &mut W) -> io::Result<()> {
    let value = summary_to_json(summary);
    serde_json::to_writer_pretty(&mut *out, &value).map_err(io::Error::other)?;
    out.write_all(b"\n")
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- Spaced-legacy regression fence -------------------------------
    // mcc-keyword-guard: allow-spaced-mention
    // (constructed at runtime so the keyword guard does not flag this
    // file and the assertions below cannot be silently rewritten into
    // tautologies by an auto-fixer.)
    fn spaced_state_space() -> String {
        format!("STATE{sp}SPACE", sp = " ")
    }

    #[test]
    fn spaced_legacy_literal_distinct_from_canonical() {
        let spaced = spaced_state_space();
        assert_eq!(spaced.len(), STATE_SPACE.len());
        assert!(spaced.contains(' '));
        assert!(!spaced.contains('_'));
        assert_ne!(spaced, STATE_SPACE);
    }

    #[test]
    fn examination_authority_covers_all_thirteen_kinds() {
        // Touches `Examination::ALL`/`as_str`/`from_name` so the
        // summarizer cannot drift from the canonical examination set.
        let names: Vec<&'static str> = Examination::ALL.iter().map(|e| e.as_str()).collect();
        assert_eq!(names.len(), 13);
        for name in &names {
            let parsed = Examination::from_name(name).expect("canonical name must parse");
            assert_eq!(parsed.as_str(), *name);
        }
    }

    fn one_row(value: Value) -> Vec<(usize, Value)> {
        vec![(1, value)]
    }

    #[test]
    fn summarize_row_routes_through_capability_report_keys() {
        let row = json!({
            "model": "demo",
            "capability_report": {
                "production_routing_status": "JustifiedLocalFallback",
                "selected": [
                    {"backend_code": "explicit_state", "role": "production", "status": "available"}
                ],
                "rejected": [
                    {"backend_code": "ay_chc", "role": "production", "status": "unsupported", "reason_code": "no_chc"}
                ]
            }
        });
        let s = summarize_row(1, &row);
        assert_eq!(s.production_routing_status, "JustifiedLocalFallback");
        assert!(s
            .selected_lanes
            .contains("explicit_state:production:available"));
        assert!(s
            .rejected_lanes
            .contains("ay_chc:production:unsupported:no_chc"));
        assert!(s.unsupported_reason_codes.contains("no_chc"));
        assert_eq!(s.identity, "model=demo");
    }

    #[test]
    fn routing_counts_sort_by_count_desc_then_name_asc() {
        let records = vec![
            (
                1,
                json!({"report": {"production_routing_status": "AYFirst"}}),
            ),
            (
                2,
                json!({"report": {"production_routing_status": "AYFirst"}}),
            ),
            (
                3,
                json!({"report": {"production_routing_status": "JustifiedLocalFallback"}}),
            ),
        ];
        let s = summarize_evidence(&records);
        assert_eq!(
            s.production_routing_status_counts,
            vec![
                (vec!["AYFirst".to_string()], 2),
                (vec!["JustifiedLocalFallback".to_string()], 1),
            ]
        );
    }

    #[test]
    fn lane_and_reason_counts_track_selected_and_rejected_lanes() {
        let row = json!({
            "report": {
                "selected": [
                    {"backend_code": "explicit_state", "role": "production", "status": "available"}
                ],
                "rejected": [
                    {"backend_code": "external_ay_binary", "role": "production", "status": "unavailable", "reason_code": "missing_binary"}
                ]
            }
        });
        let s = summarize_evidence(&one_row(row));
        let lane_map: BTreeMap<Vec<String>, u64> =
            s.backend_lane_status_counts.iter().cloned().collect();
        assert_eq!(
            lane_map
                .get(&vec![
                    "selected".into(),
                    "explicit_state".into(),
                    "production".into(),
                    "available".into()
                ])
                .copied(),
            Some(1)
        );
        let reason_map: BTreeMap<Vec<String>, u64> = s.reason_code_counts.iter().cloned().collect();
        assert_eq!(
            reason_map
                .get(&vec![
                    "rejected".into(),
                    "external_ay_binary".into(),
                    "missing_binary".into()
                ])
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn native_jit_gate_counts_match_python_fixture() {
        let row = json!({
            "report": {
                "evidence": [
                    "Petri native_jit fail_closed_gate feature=trust-cg-petri-native feature_enabled=false native_env=TY_MCC_TRUST_CG_PETRI_NATIVE native_requested=true strict_env=TY_MCC_TRUST_CG_PETRI_NATIVE_STRICT strict_requested=false parity_env=TY_MCC_TRUST_CG_PETRI_PARITY parity_enabled=true production_selected=false fail_closed=true reason_code=disabled_by_policy",
                ]
            }
        });
        let s = summarize_evidence(&one_row(row));
        assert_eq!(s.native_jit_fail_closed_gate_counts.len(), 1);
        let (key, count) = &s.native_jit_fail_closed_gate_counts[0];
        assert_eq!(*count, 1);
        assert_eq!(
            key,
            &vec![
                "native_kernel".to_string(),
                "trust-cg-petri-native".to_string(),
                "false".to_string(),
                "true".to_string(),
                "false".to_string(),
                "true".to_string(),
                "false".to_string(),
                "true".to_string(),
                "disabled_by_policy".to_string(),
            ]
        );
    }

    #[test]
    fn symbolic_execution_rows_count_cross_domain_evidence() {
        let row = json!({
            "report": {
                "evidence": [
                    "MCC symbolic_execution domain=petri_mcc status=AYPreferred status_code=ay_preferred problem=Sat reason=ModelEnumeration reason_code=model_enumeration preferred_backend=AYSat preferred_backend_code=ay_sat",
                    "BTOR2 symbolic_execution domain=btor2 status=AYPreferred status_code=ay_preferred problem=Chc reason=BitVectorFormula reason_code=bit_vector_formula preferred_backend=AYChc preferred_backend_code=ay_chc",
                ]
            }
        });
        let s = summarize_evidence(&one_row(row));
        let map: BTreeMap<Vec<String>, u64> = s.symbolic_execution_counts.iter().cloned().collect();
        assert_eq!(
            map.get(&vec![
                "MCC".to_string(),
                "petri_mcc".to_string(),
                "Sat".to_string(),
                "ay_sat".to_string(),
                "ay_preferred".to_string(),
                "model_enumeration".to_string(),
            ])
            .copied(),
            Some(1)
        );
        assert_eq!(
            map.get(&vec![
                "BTOR2".to_string(),
                "btor2".to_string(),
                "Chc".to_string(),
                "ay_chc".to_string(),
                "ay_preferred".to_string(),
                "bit_vector_formula".to_string(),
            ])
            .copied(),
            Some(1)
        );
    }

    #[test]
    fn trust_ir_blocker_and_availability_are_distinct() {
        let row = json!({
            "report": {
                "evidence": [
                    "trust-ir trust_ir_transport_identity_blocker transport=native identity=trust_ir_module_digest status_code=blocked reason_code=missing_canonical_module_digest",
                    "Petri native_jit trust_ir_transport_identity unavailable required_trust_ir_rev=4e38cb current_trust_ir_rev=6642874 cargo_dependency=false api=NativeVerificationBundle::transport_identity production_selected=false fail_closed=true",
                ]
            }
        });
        let s = summarize_evidence(&one_row(row));
        assert_eq!(s.trust_ir_transport_identity_blocker_counts.len(), 1);
        assert_eq!(
            s.trust_ir_transport_identity_blocker_counts[0].0,
            vec![
                "trust-ir".to_string(),
                "native".to_string(),
                "trust_ir_module_digest".to_string(),
                "blocked".to_string(),
                "missing_canonical_module_digest".to_string(),
            ]
        );
        assert_eq!(s.trust_ir_transport_identity_availability_counts.len(), 1);
        let avail_key = &s.trust_ir_transport_identity_availability_counts[0].0;
        assert_eq!(avail_key[0], "Petri");
        assert_eq!(avail_key[1], "native_jit");
        assert_eq!(avail_key[2], "missing");
    }

    #[test]
    fn trust_cg_typed_admission_and_blockers_kept_separate() {
        let row = json!({
            "report": {
                "evidence": [
                    "trust-cg trust_cg_admission_blocker source=NativeInstallGateAdmissionSummary schema=trust_cg.phase6.native_install_gate.admission_summary.v1 schema_version=1 consumer=ty consumer_mode=ty_trust_cg_bfs_runtime kind=ty_native_activation surface=ty_activation disposition=rejected status_code=rejected rejection_code=missing_manifest reason_code=missing_manifest requested_authority=active_callable install_authority=none production_selected=false fail_closed=true actions_ty_native_activate=false actions_compiled=2 actions_total=3",
                    "trust-cg trust_cg_admission_blocker consumer=mcc kind=petri_successor disposition=rejected rejection_code=missing_trust_ir_identity reason_code=missing_trust_ir_identity",
                ]
            }
        });
        let s = summarize_evidence(&one_row(row));
        assert_eq!(s.trust_cg_admission_blocker_counts.len(), 1);
        assert_eq!(s.trust_cg_admission_blocker_counts[0].0[0], "trust-cg");
        assert_eq!(s.trust_cg_admission_blocker_counts[0].0[1], "mcc");
        assert_eq!(s.trust_cg_native_admission_reason_counts.len(), 1);
        assert_eq!(s.trust_cg_native_admission_reason_counts[0].0[1], "ty");
    }

    #[test]
    fn ay_profile_availability_buckets_missing_typed_unknown() {
        let row = json!({
            "report": {
                "evidence": [
                    "TLA ay_solver_decision_profile_summary status=Unavailable status_code=missing_typed_summary typed_consumer=false expected_schema=ay.solve-decision-profile-summary.v1 expected_schema_version=1 production_selected=false fail_closed=true",
                    "TLA ay_solver_decision_profile_summary status=Available status_code=typed_summary_available schema=ay.solve-decision-profile-summary.v1 schema_version=1 decision=SAT decision_code=sat accepted_for_consumer=true consumer_rejection_code=none model_validated=true verification_level_code=model_validated unknown_reason_code=none unknown_limit_code=none typed_consumer=true production_selected=false fail_closed=false",
                    "TLA ay_solver_decision_profile_summary status=Available status_code=typed_summary_available schema=ay.solve-decision-profile-summary.v1 schema_version=1 decision=Unknown decision_code=unknown accepted_for_consumer=false consumer_rejection_code=unknown_result model_validated=false verification_level_code=not_validated unknown_reason_code=timeout unknown_limit_code=timeout typed_consumer=true production_selected=false fail_closed=true",
                ]
            }
        });
        let s = summarize_evidence(&one_row(row));
        let by_avail: BTreeMap<String, u64> = s
            .ay_solve_decision_profile_availability_counts
            .iter()
            .map(|(k, v)| (k[1].clone(), *v))
            .collect();
        assert_eq!(by_avail.get("missing").copied(), Some(1));
        assert_eq!(by_avail.get("typed").copied(), Some(1));
        assert_eq!(by_avail.get("unknown").copied(), Some(1));
    }

    #[test]
    fn proof_replay_boundary_counts_cross_backend_evidence() {
        let row = json!({
            "report": {
                "evidence": [
                    "BTOR2 proof_replay_boundary ay_backend_code=ay_chc safe_proof=ay_chc_verified_result safe_replay=not_available unsafe_witness=ay_chc_counterexample unsafe_replay=not_available witness_attribution=query_clause local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code=ay_first",
                    "AIGER proof_replay_boundary ay_backend_code=ay_sat safe_proof=aiger_safe_witness_validation safe_replay=validate_safe unsafe_witness=aiger_counterexample_trace unsafe_replay=transys_verify_witness witness_attribution=engine_trace local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code=ay_first",
                ]
            }
        });
        let s = summarize_evidence(&one_row(row));
        let scopes: BTreeMap<String, u64> = s
            .proof_replay_boundary_counts
            .iter()
            .map(|(k, v)| (k[0].clone(), *v))
            .collect();
        assert_eq!(scopes.get("AIGER").copied(), Some(1));
        assert_eq!(scopes.get("BTOR2").copied(), Some(1));
    }

    #[test]
    fn markdown_output_contains_section_headings_and_data_rows() {
        let records = one_row(json!({
            "model": "demo",
            "report": {
                "production_routing_status": "JustifiedLocalFallback",
                "selected": [
                    {"backend_code": "explicit_state", "role": "production", "status": "available"}
                ],
                "rejected": [
                    {"backend_code": "native_kernel", "role": "validation", "status": "unsupported", "reason_code": "native_kernel_unavailable"}
                ],
                "evidence": [
                    "Petri native_jit fail_closed_gate feature=trust-cg-petri-native feature_enabled=false native_env=TY_MCC_TRUST_CG_PETRI_NATIVE native_requested=false strict_env=TY_MCC_TRUST_CG_PETRI_NATIVE_STRICT strict_requested=false parity_env=TY_MCC_TRUST_CG_PETRI_PARITY parity_enabled=false production_selected=false fail_closed=true reason_code=disabled_by_policy",
                ]
            }
        }));
        let summary = summarize_evidence(&records);
        let mut buf: Vec<u8> = Vec::new();
        write_markdown_summary(&summary, &mut buf).expect("write");
        let text = String::from_utf8(buf).expect("utf-8");
        assert!(text.contains("### Production routing status counts"));
        assert!(text.contains("### Backend lane status counts"));
        assert!(text.contains("### Reason code counts"));
        assert!(text.contains("### Native JIT fail-closed gate counts"));
        assert!(text.contains("### Evidence rows"));
        assert!(text.contains("| rejected | native_kernel | validation | unsupported | 1 |"));
        assert!(text.contains(
            "| native_kernel | trust-cg-petri-native | false | false | false | false | false | true | disabled_by_policy | 1 |"
        ));
    }

    #[test]
    fn summary_json_round_trips_through_serde() {
        let records = one_row(json!({
            "model": "demo",
            "report": {
                "production_routing_status": "AYFirst",
                "selected": [
                    {"backend_code": "explicit_state", "role": "production", "status": "available"}
                ],
                "rejected": [
                    {"backend_code": "external_ay_binary", "role": "production", "status": "unavailable", "reason_code": "missing_binary"}
                ]
            }
        }));
        let summary = summarize_evidence(&records);
        let mut buf: Vec<u8> = Vec::new();
        write_summary_json(&summary, &mut buf).expect("write");
        let parsed: Value = serde_json::from_slice(&buf).expect("parse");
        assert_eq!(
            parsed["rows"][0]["identity"],
            Value::String("model=demo".into())
        );
        let routing = &parsed["counts"]["production_routing_status"];
        assert_eq!(routing[0]["production_routing_status"], "AYFirst");
        assert_eq!(routing[0]["count"], 1);
        let reasons = &parsed["counts"]["reason_code"];
        let has_missing_binary = reasons.as_array().expect("array").iter().any(|r| {
            r["lane"] == "rejected"
                && r["backend"] == "external_ay_binary"
                && r["reason_code"] == "missing_binary"
        });
        assert!(
            has_missing_binary,
            "missing_binary reason code expected: {parsed}"
        );
    }

    #[test]
    fn csv_output_quotes_fields_with_commas_and_uses_crlf() {
        let records = one_row(json!({
            "model": "demo,with,commas",
            "report": {
                "selected": [
                    {"backend_code": "explicit_state", "role": "production", "status": "available"}
                ]
            }
        }));
        let summary = summarize_evidence(&records);
        let mut buf: Vec<u8> = Vec::new();
        write_csv(&summary.rows, &mut buf).expect("write");
        let text = String::from_utf8(buf).expect("utf-8");
        assert!(text.contains("\r\n"));
        // The identity value contains commas, so it must be quoted.
        assert!(text.contains("\"model=demo,with,commas\""));
    }

    #[test]
    fn markdown_escape_handles_pipe_backslash_and_newline() {
        let v = Value::String("a|b\\c\nd".to_string());
        let escaped = markdown_escape_value(&v);
        assert_eq!(escaped, "a\\|b\\\\c d");
    }

    #[test]
    fn reason_code_returns_single_dict_key_when_only_one() {
        let v = json!({"only_key": "ignored_value"});
        assert_eq!(reason_code(&v), "only_key");
    }

    #[test]
    fn stringify_emits_python_compatible_forms() {
        assert_eq!(stringify(&Value::Null), "");
        assert_eq!(stringify(&Value::Bool(true)), "true");
        assert_eq!(stringify(&Value::Bool(false)), "false");
        assert_eq!(stringify(&Value::String("x".into())), "x");
        assert_eq!(stringify(&json!(42)), "42");
        // BTreeMap-ordered serialization for objects (matches sort_keys=True).
        let obj = json!({"b": 2, "a": 1});
        assert_eq!(stringify(&obj), r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn report_from_falls_back_to_row_when_no_capability_report() {
        let row = json!({"selected": [{"backend_code": "explicit_state"}]});
        let report = report_from(&row);
        assert!(report.get("selected").is_some());
    }

    #[test]
    fn ay_solver_decision_profile_default_backend_is_ay() {
        let row = json!({
            "report": {
                "evidence": [
                    "TLA ay_solver_decision_profile_summary status=Unavailable status_code=dependency_pin_blocked expected_schema=ay.solve-decision-profile-summary.v1 consumer_rejection_code=dependency_pin_blocked",
                ]
            }
        });
        let s = summarize_evidence(&one_row(row));
        // Default backend = "ay" applied because the row has no explicit
        // backend/solver column.
        let tuple = &s.ay_solver_decision_profile_counts[0].0;
        assert_eq!(tuple[0], "TLA");
        assert_eq!(tuple[1], "ay");
        assert_eq!(tuple[2], "dependency_pin_blocked");
    }
}
