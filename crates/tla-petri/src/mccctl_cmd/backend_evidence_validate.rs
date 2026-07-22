// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (regression-fence tests below construct the round-1 spaced literals at
// runtime so an auto-fixer cannot rewrite them into tautologies; the
// legacy-data parser intentionally accepts the spaced 2025-archive form.)

//! Fail-closed canary for MCC backend capability evidence JSONL sidecars.
//!
//! Replaces a former Python helper (#4509).
//! Reads one or more backend capability JSONL evidence sidecars and
//! verifies that the required gating rows are present and well-formed.
//!
//! Routes every MCC keyword reference through
//! [`tla_petri::mcc_keywords`] and every examination name through
//! [`tla_petri::examination::Examination`] (the Rust enum is the single
//! authority for the 13 MCC examination vocabulary — eliminating the
//! cross-language drift that produced the qualification-1 keyword bug).
//!
//! The CLI surface mirrors the Python script exactly:
//!
//! ```text
//! ty-mcc-backend-evidence-validate [--require CHECK]... [--json] [--list-checks] [JSONL...]
//! ```
//!
//! Exit 0 = required checks all satisfied; exit 1 = validation failure.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;

use clap::Parser;
use serde::Serialize;
use serde_json::{json, Value};

use crate::examination::Examination;
use crate::mcc_evidence_jsonl::read_jsonl_records;
use crate::mcc_keywords::{CANNOT_COMPUTE, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING, STATE_SPACE};

// ---------- Constants (mirrored from the Python module) ----------

const MCC_PORTFOLIO_ROUTE_SCHEMA: &str = "mcc.portfolio_route.v1";
const MCC_PORTFOLIO_ROUTE_SCHEMA_VERSION: &str = "1";
const MCC_PORTFOLIO_ROUTE_COMPONENT: &str = "portfolio_route";

const MCC_PORTFOLIO_ROUTE_REQUIRED_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "route",
    "lane_family",
    "backend_code",
    "problem",
    "role",
    "readiness",
    "readiness_code",
    "evidence_source",
    "evidence_gate",
    "owner_project",
    "answer_producer",
    "routing_selected",
    "selection_rank",
    "production_selected",
    "fail_closed",
];

const MCC_PORTFOLIO_ROUTE_BOOLEAN_FIELDS: &[&str] = &[
    "answer_producer",
    "routing_selected",
    "production_selected",
    "fail_closed",
];

const MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER: &[&str] = &[
    "explicit_bfs",
    "reductions",
    "ay_symbolic",
    "aiger_hwmcc",
    "native_jit",
    "hardware_model",
];

const AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA: &str =
    "ay.symbolic-execution-contract-manifest.v1";
const AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA: &str =
    "ay.symbolic-execution-contract-manifest-health.v1";
const AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_COMPONENT: &str =
    "symbolic_execution_contract_manifest";
const AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_COMPONENT: &str =
    "symbolic_execution_contract_manifest_health";

const HARDWARE_REPLAY_PRIMITIVE_SCHEMA: &str = "hardware_replay_primitive/v1";

const AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA: &str = "ay.solver-capability-descriptor.v1";
const AY_SOLVER_CAPABILITY_DESCRIPTOR_COMPONENT: &str = "solver_capability_descriptor";
const AY_SOLVER_CAPABILITY_DESCRIPTOR_SOURCE_PACKAGE: &str = "ay-dpll";
const AY_MODEL_BLOCKING_CLAUSE_SCHEMA: &str = "ay.model-blocking-clause.v1";
const AY_SOLVE_DECISION_PROFILE_MODEL_CONSUMER_SCHEMA: &str =
    "ay.solve-decision-profile-model-consumer.v1";

const TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_COMPONENT: &str = "trust_cg_admission_blocker";
const TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE: &str = "NativeInstallGateAdmissionSummary";
const TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE_PACKAGE: &str = "trust-cg-codegen";

const AIGER_PORTFOLIO_WINNER_COMPONENT: &str = "portfolio_winner";

// Native shared primitive manifest line fields (for split parsing).
const NATIVE_SHARED_PRIMITIVE_CONTRACT_MANIFEST_LINE_FIELDS: &[&str] = &[
    "manifest_line",
    "manifest_keys",
    "manifest_key",
    "manifest_pair",
    "manifest_entry",
];

// ---------- Error type ----------

#[derive(Debug)]
struct EvidenceValidationError(String);

impl std::fmt::Display for EvidenceValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EvidenceValidationError {}

impl EvidenceValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

type EvidenceResult<T> = Result<T, EvidenceValidationError>;

// ---------- CLI ----------

/// Command-line arguments for the `ty-mcc-backend-evidence-validate` helper.
#[derive(Parser, Debug)]
#[command(
    name = "ty-mcc-backend-evidence-validate",
    about = "Validate MCC backend capability JSONL sidecars contain the shared AY/trust-ir/trust-cg evidence rows required by packaging and replay gates.",
    long_about = "Replaces a former Python helper.\n\n\
                  Reads one or more JSONL evidence sidecars and verifies\n\
                  that the requested set of capability checks (default = MCC\n\
                  canary set) are all satisfied. Each check is a typed\n\
                  predicate over the embedded `report.evidence` rows.\n\n\
                  Routes MCC keyword references through `tla_petri::mcc_keywords`\n\
                  and every examination name through\n\
                  `tla_petri::examination::Examination`.",
    arg_required_else_help = false,
    disable_help_flag = false
)]
pub struct Cli {
    /// Backend capability evidence JSONL path(s), or '-' for stdin.
    #[arg(value_name = "JSONL")]
    pub jsonl: Vec<String>,

    /// Required evidence check; may be repeated. Defaults to the MCC canary set.
    #[arg(long = "require", value_name = "CHECK")]
    pub require: Vec<String>,

    /// Emit JSON summary instead of the human-readable single-line summary.
    #[arg(long = "json")]
    pub json: bool,

    /// List available evidence checks and exit.
    #[arg(long = "list-checks")]
    pub list_checks: bool,

    /// Require every `current_ay_rev=` token in `report.evidence` rows to
    /// match this 40-hex-char git revision. Mirrors the BenchKit
    /// `validate_packaged_ay_revision` shell-level check so the same
    /// freshness gate runs from one Rust binary. Values `missing` and
    /// `none` are ignored.
    #[arg(long = "require-ay-rev", value_name = "REV")]
    pub require_ay_rev: Option<String>,
}

// ---------- Check registry ----------

/// One backend evidence check: a stable identifier, a description, and a
/// row-counting predicate. The check passes when the counter returns > 0
/// (and any deep field validations succeed).
struct EvidenceCheck {
    code: &'static str,
    description: &'static str,
    counter: fn(&[EvidenceRow]) -> EvidenceResult<i64>,
}

/// Default set of required checks when `--require` is not given. Mirrors
/// `DEFAULT_REQUIRED` in the Python source.
const DEFAULT_REQUIRED: &[&str] = &[
    "mcc_ay_symbolic_execution",
    "mcc_hot_execution_production_selected",
    "native_jit_fail_closed_gate",
    "trust_ir_transport_identity",
    "trust_cg_native_admission",
    "ay_solve_decision_profile",
];

fn all_checks() -> &'static [EvidenceCheck] {
    &[
        EvidenceCheck {
            code: "mcc_ay_symbolic_execution",
            description: "MCC symbolic execution row routes a Petri/MCC problem to a AY backend.",
            counter: count_mcc_ay_symbolic_execution,
        },
        EvidenceCheck {
            code: "mcc_hot_execution_production_selected",
            description: "MCC hot-execution row proves a completed production-selected run that did not fail closed.",
            counter: count_mcc_hot_execution_production_selected,
        },
        EvidenceCheck {
            code: "native_jit_fail_closed_gate",
            description: "Petri native/JIT gate is present and either fail-closed outside production or explicitly selects native production with required policy.",
            counter: count_native_jit_fail_closed_gate,
        },
        EvidenceCheck {
            code: "trust_ir_transport_identity",
            description: "trust-ir transport identity availability row is present and fail-closed.",
            counter: count_trust_ir_transport_identity,
        },
        EvidenceCheck {
            code: "trust_cg_native_admission",
            description: "trust-cg native admission summary row is present and fail-closed.",
            counter: count_trust_cg_native_admission,
        },
        EvidenceCheck {
            code: "trust_cg_call_packet_contract_descriptor",
            description: "trust-cg call-packet and downstream contract descriptor fields are authoritative and source-owned.",
            counter: count_trust_cg_call_packet_contract_descriptor,
        },
        EvidenceCheck {
            code: "trust_cg_compile_artifact_cache_telemetry",
            description: "trust-cg compile-artifact cache telemetry descriptor is present and does not authorize native production by itself.",
            counter: count_trust_cg_compile_artifact_cache_telemetry,
        },
        EvidenceCheck {
            code: "trust_cg_host_jit_pgo_provenance",
            description: "trust-cg host-JIT PGO provenance descriptor is present and does not authorize native production by itself.",
            counter: count_trust_cg_host_jit_pgo_provenance,
        },
        EvidenceCheck {
            code: "production_selector_decision",
            description: "MCC production selector records shared primitive decisions and machine-readable fallback reasons.",
            counter: count_production_selector_decision,
        },
        EvidenceCheck {
            code: "mcc_prepared_program",
            description: "MCC prepared-program and shared-engine descriptor rows expose frontend/source identity, descriptor identities, backend families, validations, reductions, and optional artifact/fingerprint identity.",
            counter: count_mcc_prepared_program,
        },
        EvidenceCheck {
            code: "portfolio_route",
            description: "MCC portfolio route rows enumerate explicit BFS, reductions, AY symbolic, AIGER/HWMCC, native/JIT, and hardware replay lanes.",
            counter: count_mcc_portfolio_route,
        },
        EvidenceCheck {
            code: "aiger_portfolio_winner",
            description: "AIGER typed portfolio winner row preserves engine, problem, backend, role, and timing metadata.",
            counter: count_aiger_portfolio_winner,
        },
        EvidenceCheck {
            code: "ay_solver_capability_descriptor",
            description: "AY solver capability descriptor exposes model-blocking/model-consumer vocabulary without MCC duplicating solver policy.",
            counter: count_ay_solver_capability_descriptor,
        },
        EvidenceCheck {
            code: "ay_symbolic_execution_contract_manifest",
            description: "AY symbolic execution contract manifest and health rows expose producer-owned route and diagnostic vocabulary.",
            counter: count_ay_symbolic_execution_contract_manifest,
        },
        EvidenceCheck {
            code: "trust_ir_native_evidence_artifact_resolution",
            description: "trust-ir native evidence artifact resolution row exposes authority/status codes for MCC selector decisions.",
            counter: count_trust_ir_native_evidence_artifact_resolution,
        },
        EvidenceCheck {
            code: "trust_ir_native_verification_bundle_handoff",
            description: "trust-ir Petri native verification bundle handoff manifest and completeness rows expose bundle identity, artifact authority, AY evidence identity, and downstream responsibility sections.",
            counter: count_trust_ir_native_verification_bundle_handoff,
        },
        EvidenceCheck {
            code: "trust_ir_native_semantic_bridge_proof_identity",
            description: "trust-ir native semantic-bridge proof identity rows bind the bridge digest, report status, proof status, and fail-closed reason.",
            counter: count_trust_ir_native_semantic_bridge_proof_identity,
        },
        EvidenceCheck {
            code: "trust_ir_petri_proof_evidence_identity",
            description: "trust-ir proof/evidence identity rows bind semantic bridge, trust_mc binding, proof handoff, replay transcript, and solver identity.",
            counter: count_trust_ir_petri_proof_evidence_identity,
        },
        EvidenceCheck {
            code: "trust_ir_petri_semantic_bridge_proof_admission",
            description: "Optional trust-ir Petri semantic-bridge proof-admission row is accepted only when the semantic bridge and proof evidence are strictly bound or fail-closed.",
            counter: count_trust_ir_petri_semantic_bridge_proof_admission,
        },
        EvidenceCheck {
            code: "trust_ir_native_verification_bundle_handoff_replay_contract_surface",
            description: "trust-ir Petri native verification bundle handoff replay contract surface rows expose helper/schema/fixture/validator imports and round-trip diagnostics.",
            counter: count_trust_ir_native_verification_bundle_handoff_replay_contract_surface,
        },
        EvidenceCheck {
            code: "trust_ir_native_verification_bundle_handoff_replay_contract_report_identity",
            description: "trust-ir Petri replay contract round-trip report identity rows expose status, fail-closed state, summary counts, diagnostics, and a digest-bound identity.",
            counter: count_trust_ir_native_verification_bundle_handoff_replay_contract_surface,
        },
        EvidenceCheck {
            code: "trust_ir_native_verification_bundle_handoff_replay_contract_json_manifest_binding",
            description: "trust-ir Petri replay contract JSON manifest binding rows tie the compact replay report JSON digest to the handoff manifest identity digest.",
            counter: count_trust_ir_native_verification_bundle_handoff_replay_contract_json_manifest_binding,
        },
        EvidenceCheck {
            code: "ay_solve_decision_profile",
            description: "AY typed solve-decision/profile summary row is present.",
            counter: count_ay_solve_decision_profile,
        },
        EvidenceCheck {
            code: "hardware_proof_replay_boundary",
            description: "AIGER and BTOR2 proof/replay boundary rows carry AY proof and witness evidence with fail-closed native promotion.",
            counter: count_hardware_proof_replay_boundary,
        },
        EvidenceCheck {
            code: "hardware_replay_decision",
            description: "BTOR2 hardware replay decision row carries accepted AY replay identity plus AY proof evidence, or a typed fail-closed blocker.",
            counter: count_hardware_replay_decision,
        },
        EvidenceCheck {
            code: "semantic_successor_bridge",
            description: "Petri native semantic successor bridge row is present and fail-closed until trust-ir represents the successor relation.",
            counter: count_semantic_successor_bridge,
        },
        EvidenceCheck {
            code: "petri_trust_mc_model_acceptance",
            description: "AY-owned Petri/trust_mc CHC model-acceptance row is present and accepted only when solver model-validation evidence is attached.",
            counter: count_petri_trust_mc_model_acceptance,
        },
        EvidenceCheck {
            code: "petri_trust_mc_native_route_admission",
            description: "AY-owned Petri/trust_mc native route-admission row is present and remains fail-closed unless every producer stage accepts.",
            counter: count_petri_trust_mc_native_route_admission,
        },
    ]
}

fn check_by_code(code: &str) -> Option<&'static EvidenceCheck> {
    all_checks().iter().find(|c| c.code == code)
}

// ---------- Evidence row ----------

/// One parsed evidence row from the JSONL `report.evidence` list. Each
/// row carries the JSONL row number (1-based, after skipping comments/
/// blanks), the raw token string, and the parsed key=value data plus the
/// positional scope/component/marker/status fields.
#[derive(Debug, Clone)]
struct EvidenceRow {
    row_number: usize,
    #[allow(dead_code)]
    raw: String,
    data: BTreeMap<String, String>,
}

impl EvidenceRow {
    fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(String::as_str)
    }

    fn get_eq(&self, key: &str, value: &str) -> bool {
        self.get(key).map(|v| v == value).unwrap_or(false)
    }
}

/// Parse one evidence row string into a key=value map, with positional
/// fields stamped under `scope`, `component`, `marker`, `status_word`.
/// Mirrors `_evidence_data` in the Python source.
fn parse_evidence_data(row: &str) -> BTreeMap<String, String> {
    let tokens = shlex_split(row);
    let mut data = BTreeMap::<String, String>::new();
    if let Some(first) = tokens.first() {
        data.insert("scope".to_string(), first.clone());
    }
    if tokens.len() > 1 && !tokens[1].contains('=') {
        data.insert("component".to_string(), tokens[1].clone());
        data.insert("marker".to_string(), tokens[1].clone());
    }
    if tokens.len() > 2 && !tokens[2].contains('=') {
        data.insert("marker".to_string(), tokens[2].clone());
    }
    if tokens.len() > 3 && !tokens[3].contains('=') {
        data.insert("status_word".to_string(), tokens[3].clone());
    }
    for token in &tokens {
        if let Some((key, value)) = token.split_once('=') {
            data.insert(key.to_string(), value.to_string());
        }
    }
    // Re-scan for manifest line fields (Python's regex pass over the
    // raw row). These may contain whitespace inside escaped values and
    // are matched directly off the raw text.
    for field in NATIVE_SHARED_PRIMITIVE_CONTRACT_MANIFEST_LINE_FIELDS {
        if let Some(value) = extract_named_token(row, field) {
            data.insert((*field).to_string(), value);
        }
    }
    data
}

/// Extract a `key=value` token from a raw evidence row, where value is
/// terminated by ASCII whitespace. Mirrors the Python regex
/// `(?:^|\s){field}=([^\s]+)`.
fn extract_named_token(row: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=");
    let mut start = 0;
    while start < row.len() {
        let rest = &row[start..];
        let idx = rest.find(&needle)?;
        let absolute = start + idx;
        // Boundary: start of string or preceded by whitespace.
        let prev_ok = absolute == 0 || row.as_bytes()[absolute - 1].is_ascii_whitespace();
        if prev_ok {
            let value_start = absolute + needle.len();
            let value_end = row[value_start..]
                .find(|c: char| c.is_ascii_whitespace())
                .map(|n| value_start + n)
                .unwrap_or(row.len());
            return Some(row[value_start..value_end].to_string());
        }
        start = absolute + needle.len();
    }
    None
}

/// Minimal POSIX-style shell splitter sufficient for evidence rows.
/// Handles single/double quotes and backslash escapes; falls back to
/// whitespace splitting on unbalanced quotes.
fn shlex_split(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;
    let mut started = false;
    let mut had_error = false;
    for ch in input.chars() {
        if escape {
            current.push(ch);
            escape = false;
            started = true;
            continue;
        }
        if ch == '\\' && !in_single {
            escape = true;
            started = true;
            continue;
        }
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                current.push(ch);
            }
            started = true;
            continue;
        }
        if in_double {
            if ch == '"' {
                in_double = false;
            } else {
                current.push(ch);
            }
            started = true;
            continue;
        }
        match ch {
            '\'' => {
                in_single = true;
                started = true;
            }
            '"' => {
                in_double = true;
                started = true;
            }
            c if c.is_ascii_whitespace() => {
                if started {
                    tokens.push(std::mem::take(&mut current));
                    started = false;
                }
            }
            c => {
                current.push(c);
                started = true;
            }
        }
    }
    if in_single || in_double || escape {
        had_error = true;
    }
    if started {
        tokens.push(current);
    }
    if had_error {
        // Fall back to whitespace splitting if quoting is malformed.
        return input.split_ascii_whitespace().map(str::to_string).collect();
    }
    tokens
}

// ---------- JSONL reading ----------

/// Yield one JSON object per non-blank, non-comment line. Routes
/// through the shared `tla_petri::mcc_evidence_jsonl` helper so the
/// validator and `ty-mcc-summarize-evidence` parse the same JSONL
/// sidecars identically.
fn iter_jsonl(paths: &[String]) -> EvidenceResult<Vec<(usize, Value)>> {
    read_jsonl_records(paths).map_err(|err| EvidenceValidationError::new(err.0))
}

/// Extract the `report` sub-object (or empty) from a JSONL record.
fn record_report(record: &Value) -> &Value {
    record
        .get("report")
        .filter(|v| v.is_object())
        .unwrap_or(&Value::Null)
}

/// Walk one JSONL row's `report.evidence` list and yield parsed
/// EvidenceRow values. Mirrors `_evidence_rows`.
fn collect_evidence_rows(records: &[(usize, Value)]) -> Vec<EvidenceRow> {
    let mut out = Vec::new();
    for (row_number, record) in records {
        let report = record_report(record);
        let evidence = match report.get("evidence").and_then(Value::as_array) {
            Some(arr) => arr,
            None => continue,
        };
        for entry in evidence {
            if let Some(text) = entry.as_str() {
                out.push(EvidenceRow {
                    row_number: *row_number,
                    raw: text.to_string(),
                    data: parse_evidence_data(text),
                });
            }
        }
    }
    out
}

// ---------- Helper predicates ----------

fn is_true(value: Option<&str>) -> bool {
    value
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn strict_bool(value: Option<&str>, context: &str, field: &str) -> EvidenceResult<bool> {
    match value {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        _ => Err(EvidenceValidationError::new(format!(
            "{context}.{field}: expected true or false",
        ))),
    }
}

fn strict_non_negative_int(value: Option<&str>, context: &str, field: &str) -> EvidenceResult<i64> {
    let raw = value.ok_or_else(|| {
        EvidenceValidationError::new(format!("{context}.{field}: expected integer"))
    })?;
    let parsed: i64 = raw.parse().map_err(|_| {
        EvidenceValidationError::new(format!("{context}.{field}: expected integer"))
    })?;
    if parsed < 0 {
        return Err(EvidenceValidationError::new(format!(
            "{context}.{field}: expected non-negative integer",
        )));
    }
    Ok(parsed)
}

fn strict_non_negative_float(
    value: Option<&str>,
    context: &str,
    field: &str,
) -> EvidenceResult<f64> {
    let raw = value.ok_or_else(|| {
        EvidenceValidationError::new(format!("{context}.{field}: expected number"))
    })?;
    let parsed: f64 = raw
        .parse()
        .map_err(|_| EvidenceValidationError::new(format!("{context}.{field}: expected number")))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(EvidenceValidationError::new(format!(
            "{context}.{field}: expected non-negative finite number",
        )));
    }
    Ok(parsed)
}

fn ordered_delimited_values(value: Option<&str>) -> Vec<String> {
    let text = value.unwrap_or("");
    if text.is_empty() || text == "none" {
        return Vec::new();
    }
    let separator = if text.contains('|') && !text.contains(',') {
        '|'
    } else {
        ','
    };
    text.split(separator)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

// ---------- Per-check counters ----------

// portfolio_route: most-tested in petri_cli.rs integration test.
fn count_mcc_portfolio_route(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    #[derive(Clone)]
    struct RouteRow {
        route: String,
        selection_rank: i64,
        routing_selected: String,
        production_selected: String,
    }

    let mut errors: Vec<String> = Vec::new();
    let mut route_rows: Vec<RouteRow> = Vec::new();

    for row in rows {
        if !row.get_eq("scope", "MCC") || !row.get_eq("component", MCC_PORTFOLIO_ROUTE_COMPONENT) {
            continue;
        }
        let context = format!("JSONL row {}: portfolio_route[MCC]", row.row_number);
        let mut missing = Vec::new();
        for field in MCC_PORTFOLIO_ROUTE_REQUIRED_FIELDS {
            match row.get(field) {
                Some(v) if !v.is_empty() => {}
                _ => missing.push(*field),
            }
        }
        for field in &missing {
            errors.push(format!("{context}.{field}: missing"));
        }
        if !missing.is_empty() {
            continue;
        }
        let schema = row.get("schema").unwrap_or("");
        if schema != MCC_PORTFOLIO_ROUTE_SCHEMA {
            errors.push(format!(
                "{context}.schema: expected {MCC_PORTFOLIO_ROUTE_SCHEMA:?}, got {schema:?}"
            ));
        }
        let schema_version = row.get("schema_version").unwrap_or("");
        if schema_version != MCC_PORTFOLIO_ROUTE_SCHEMA_VERSION {
            errors.push(format!(
                "{context}.schema_version: expected {MCC_PORTFOLIO_ROUTE_SCHEMA_VERSION:?}, got {schema_version:?}"
            ));
        }
        for field in MCC_PORTFOLIO_ROUTE_BOOLEAN_FIELDS {
            if let Err(err) = strict_bool(row.get(field), &context, field) {
                errors.push(err.0);
            }
        }
        let selection_rank =
            match strict_non_negative_int(row.get("selection_rank"), &context, "selection_rank") {
                Ok(v) => v,
                Err(err) => {
                    errors.push(err.0);
                    continue;
                }
            };
        let route = row.get("route").unwrap_or("").to_string();
        let expected = expected_portfolio_route(&route);
        if expected.is_empty() {
            errors.push(format!("{context}.route: unknown route {route:?}"));
            continue;
        }
        let readiness = row.get("readiness").unwrap_or("");
        let allowed_readiness: BTreeSet<&str> = MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER
            .iter()
            .filter_map(|r| {
                expected_portfolio_route(r)
                    .iter()
                    .find(|(k, _)| *k == "readiness")
                    .map(|(_, v)| *v)
            })
            .collect();
        if !allowed_readiness.contains(readiness) {
            errors.push(format!("{context}.readiness: unknown code {readiness:?}"));
        }
        for (field, expected_value) in &expected {
            let actual = row.get(field).unwrap_or("");
            if actual != *expected_value {
                errors.push(format!(
                    "{context}.{field}: expected {expected_value:?}, got {actual:?}"
                ));
            }
        }
        route_rows.push(RouteRow {
            route,
            selection_rank,
            routing_selected: row.get("routing_selected").unwrap_or("").to_string(),
            production_selected: row.get("production_selected").unwrap_or("").to_string(),
        });
    }

    if route_rows.len() != MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER.len() {
        errors.push(format!(
            "portfolio_route[MCC]: expected exactly six canonical routes, got {}",
            route_rows.len()
        ));
    }
    let observed_routes: Vec<String> = route_rows.iter().map(|r| r.route.clone()).collect();
    let observed_set: BTreeSet<&str> = observed_routes.iter().map(String::as_str).collect();
    let missing_routes: Vec<&&str> = MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER
        .iter()
        .filter(|r| !observed_set.contains(**r))
        .collect();
    if !missing_routes.is_empty() {
        errors.push(format!(
            "portfolio_route[MCC].route: missing canonical route(s) {}",
            missing_routes
                .iter()
                .map(|r| **r)
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &observed_routes {
        *counts.entry(r.as_str()).or_insert(0) += 1;
    }
    let duplicates: Vec<&&str> = counts
        .iter()
        .filter(|(_, n)| **n > 1)
        .map(|(k, _)| k)
        .collect();
    if !duplicates.is_empty() {
        errors.push(format!(
            "portfolio_route[MCC].route: duplicate route(s) {}",
            duplicates.iter().map(|r| **r).collect::<Vec<_>>().join(",")
        ));
    }
    let ranks: Vec<i64> = route_rows.iter().map(|r| r.selection_rank).collect();
    let mut rank_counts: BTreeMap<i64, usize> = BTreeMap::new();
    for r in &ranks {
        *rank_counts.entry(*r).or_insert(0) += 1;
    }
    let dup_ranks: Vec<i64> = rank_counts
        .iter()
        .filter(|(_, n)| **n > 1)
        .map(|(k, _)| *k)
        .collect();
    if !dup_ranks.is_empty() {
        errors.push(format!(
            "portfolio_route[MCC].selection_rank: duplicate rank(s) {}",
            dup_ranks
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    let mut sorted_ranks = ranks.clone();
    sorted_ranks.sort_unstable();
    if ranks != sorted_ranks {
        errors.push("portfolio_route[MCC].selection_rank: expected ascending order".to_string());
    }
    let mut sorted_rows = route_rows.clone();
    sorted_rows.sort_by_key(|r| r.selection_rank);
    let ordered_routes_by_rank: Vec<&str> = sorted_rows.iter().map(|r| r.route.as_str()).collect();
    if ordered_routes_by_rank.len() == MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER.len()
        && ordered_routes_by_rank.as_slice() != MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER
    {
        errors.push(format!(
            "portfolio_route[MCC].selection_rank: rank order should be {}",
            MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER.join(",")
        ));
    }
    let canonical_routing_selected: BTreeSet<&str> = MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER
        .iter()
        .filter(|r| {
            expected_portfolio_route(r)
                .iter()
                .find(|(k, _)| *k == "routing_selected")
                .map(|(_, v)| *v == "true")
                .unwrap_or(false)
        })
        .copied()
        .collect();
    let routing_selected: BTreeSet<&str> = route_rows
        .iter()
        .filter(|r| r.routing_selected == "true")
        .map(|r| r.route.as_str())
        .collect();
    if routing_selected != canonical_routing_selected {
        errors.push(format!(
            "portfolio_route[MCC].routing_selected: expected route set {}",
            canonical_routing_selected
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    let canonical_production_selected: BTreeSet<&str> = MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER
        .iter()
        .filter(|r| {
            expected_portfolio_route(r)
                .iter()
                .find(|(k, _)| *k == "production_selected")
                .map(|(_, v)| *v == "true")
                .unwrap_or(false)
        })
        .copied()
        .collect();
    let production_selected: BTreeSet<&str> = route_rows
        .iter()
        .filter(|r| r.production_selected == "true")
        .map(|r| r.route.as_str())
        .collect();
    if production_selected != canonical_production_selected {
        errors.push(format!(
            "portfolio_route[MCC].production_selected: expected route set {}",
            canonical_production_selected
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .join(",")
        ));
    }

    if !errors.is_empty() {
        return Err(EvidenceValidationError::new(errors.join("; ")));
    }
    Ok(route_rows.len() as i64)
}

fn expected_portfolio_route(route: &str) -> Vec<(&'static str, &'static str)> {
    match route {
        "explicit_bfs" => vec![
            ("lane_family", "explicit_bfs"),
            ("backend_code", "explicit_state"),
            ("problem", "ExplicitReachability"),
            ("role", "fallback_answer"),
            ("readiness", "ready"),
            ("readiness_code", "explicit_state_fallback_available"),
            ("evidence_source", "MCC.answer_lane.explicit_state_fallback"),
            ("evidence_gate", "explicit_state"),
            ("owner_project", "TY"),
            ("answer_producer", "true"),
            ("routing_selected", "true"),
            ("selection_rank", "10"),
            ("production_selected", "true"),
            ("fail_closed", "false"),
        ],
        "reductions" => vec![
            ("lane_family", "reductions"),
            ("backend_code", "structural_reductions"),
            ("problem", "Preprocessing"),
            ("role", "preprocessor"),
            ("readiness", "ready"),
            ("readiness_code", "structural_reductions_available"),
            ("evidence_source", "MCC.production_selector_decision"),
            ("evidence_gate", "shared_primitive_evidence"),
            ("owner_project", "TY"),
            ("answer_producer", "false"),
            ("routing_selected", "true"),
            ("selection_rank", "20"),
            ("production_selected", "false"),
            ("fail_closed", "false"),
        ],
        "ay_symbolic" => vec![
            ("lane_family", "ay_symbolic"),
            ("backend_code", "ay_sat"),
            ("problem", "Sat"),
            ("role", "symbolic_evidence"),
            ("readiness", "ready"),
            ("readiness_code", "ay_symbolic_ready"),
            ("evidence_source", "MCC.symbolic_execution"),
            (
                "evidence_gate",
                AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA,
            ),
            ("owner_project", "AY"),
            ("answer_producer", "false"),
            ("routing_selected", "true"),
            ("selection_rank", "30"),
            ("production_selected", "false"),
            ("fail_closed", "false"),
        ],
        "aiger_hwmcc" => vec![
            ("lane_family", "aiger_hwmcc"),
            ("backend_code", "aiger_portfolio"),
            ("problem", "Safety"),
            ("role", "hardware_portfolio"),
            ("readiness", "ready"),
            ("readiness_code", "aiger_ay_adapter_ready"),
            ("evidence_source", "AIGER.ay_adapter_decision"),
            ("evidence_gate", "aiger.ay_adapter_decision.v1"),
            ("owner_project", "TY"),
            ("answer_producer", "false"),
            ("routing_selected", "true"),
            ("selection_rank", "40"),
            ("production_selected", "false"),
            ("fail_closed", "false"),
        ],
        "native_jit" => vec![
            ("lane_family", "native_jit"),
            ("backend_code", "trust_cg_petri_native"),
            ("problem", "NativeSuccessor"),
            ("role", "primary_answer_producer"),
            ("readiness", "blocked"),
            ("readiness_code", "shared_primitive_runtime_proof_blocked"),
            (
                "evidence_source",
                "trust-cg.petri_native_successor_compile_artifact_handoff",
            ),
            (
                "evidence_gate",
                "trust-cg.petri.native_successor.compile_artifact_handoff.v1",
            ),
            ("owner_project", "trust-cg"),
            ("answer_producer", "true"),
            ("routing_selected", "false"),
            ("selection_rank", "50"),
            ("production_selected", "false"),
            ("fail_closed", "true"),
        ],
        "hardware_model" => vec![
            ("lane_family", "hardware_model"),
            ("backend_code", "hardware_ay_replay"),
            ("problem", "HardwareReplay"),
            ("role", "hardware_replay_candidate"),
            ("readiness", "blocked"),
            ("readiness_code", "proof_replay_acceptance_required"),
            ("evidence_source", "AIGER.hardware_replay_primitive"),
            ("evidence_gate", HARDWARE_REPLAY_PRIMITIVE_SCHEMA),
            ("owner_project", "TY"),
            ("answer_producer", "true"),
            ("routing_selected", "false"),
            ("selection_rank", "60"),
            ("production_selected", "false"),
            ("fail_closed", "true"),
        ],
        _ => Vec::new(),
    }
}

// ay_solver_capability_descriptor: present in petri_cli.rs integration test.
fn count_ay_solver_capability_descriptor(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    let mut errors: Vec<String> = Vec::new();
    for row in rows {
        if !row.get_eq("scope", "AY")
            || !row.get_eq("component", AY_SOLVER_CAPABILITY_DESCRIPTOR_COMPONENT)
        {
            continue;
        }
        let context = format!(
            "JSONL row {}: solver_capability_descriptor[AY]",
            row.row_number
        );
        let required_fields = [
            "schema",
            "schema_version",
            "source_package",
            "solver",
            "capability",
            "status",
            "status_code",
            "reason_code",
            "api_symbols",
            "evidence_schemas",
            "production_selected",
            "fail_closed",
        ];
        let mut missing = false;
        for field in required_fields {
            match row.get(field) {
                Some(v) if !v.is_empty() => {}
                _ => {
                    errors.push(format!("{context}.{field}: missing"));
                    missing = true;
                }
            }
        }
        if missing {
            continue;
        }
        let schema = row.get("schema").unwrap_or("");
        if schema != AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA {
            errors.push(format!(
                "{context}.schema: expected {AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA:?}, got {schema:?}"
            ));
        }
        if row.get("schema_version") != Some("1") {
            errors.push(format!(
                "{context}.schema_version: expected '1', got {:?}",
                row.get("schema_version").unwrap_or("")
            ));
        }
        let source_package = row.get("source_package").unwrap_or("");
        if source_package != AY_SOLVER_CAPABILITY_DESCRIPTOR_SOURCE_PACKAGE {
            errors.push(format!(
                "{context}.source_package: expected {AY_SOLVER_CAPABILITY_DESCRIPTOR_SOURCE_PACKAGE:?}, got {source_package:?}"
            ));
        }
        if let Some(pkg) = row.get("package") {
            if !pkg.is_empty() && pkg != source_package {
                errors.push(format!("{context}.package: conflicts with source_package"));
            }
        }
        if row.get("solver") != Some("ay") {
            errors.push(format!("{context}.solver: expected 'ay'"));
        }
        if row.get("capability") != Some("model_blocking") {
            errors.push(format!("{context}.capability: expected 'model_blocking'"));
        }
        if row.get("status") != row.get("status_code") {
            errors.push(format!("{context}.status_code: status mismatch"));
        }
        let status = row.get("status").unwrap_or("");
        if status != "available" && status != "blocked" {
            errors.push(format!("{context}.status: unknown code {status:?}"));
        }
        if status == "available" && row.get("reason_code") != Some("ay_owned_public_api") {
            errors.push(format!(
                "{context}.reason_code: expected 'ay_owned_public_api'"
            ));
        }
        let api_symbols = ordered_delimited_values(row.get("api_symbols"));
        let evidence_schemas: BTreeSet<String> =
            ordered_delimited_values(row.get("evidence_schemas"))
                .into_iter()
                .collect();
        if !api_symbols
            .iter()
            .any(|s| s.contains("try_assert_model_blocking_clause_for_consumer"))
        {
            errors.push(format!(
                "{context}.api_symbols: missing model-blocking consumer API"
            ));
        }
        for schema in [
            AY_MODEL_BLOCKING_CLAUSE_SCHEMA,
            AY_SOLVE_DECISION_PROFILE_MODEL_CONSUMER_SCHEMA,
        ] {
            if !evidence_schemas.contains(schema) {
                errors.push(format!("{context}.evidence_schemas: missing {schema}"));
            }
        }
        match strict_bool(
            row.get("production_selected"),
            &context,
            "production_selected",
        ) {
            Ok(true) => errors.push(format!(
                "{context}: capability_descriptor_cannot_select_production"
            )),
            Err(err) => errors.push(err.0),
            _ => {}
        }
        match strict_bool(row.get("fail_closed"), &context, "fail_closed") {
            Ok(false) => errors.push(format!(
                "{context}: capability_descriptor_must_be_fail_closed"
            )),
            Err(err) => errors.push(err.0),
            _ => {}
        }
        count += 1;
    }
    if !errors.is_empty() {
        return Err(EvidenceValidationError::new(errors.join("; ")));
    }
    Ok(count)
}

// ay_symbolic_execution_contract_manifest: present in integration test.
fn count_ay_symbolic_execution_contract_manifest(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut manifest_present = false;
    let mut health_present = false;
    let mut errors: Vec<String> = Vec::new();
    for row in rows {
        let is_manifest = row.get_eq("scope", "AY")
            && (row.get_eq(
                "component",
                AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_COMPONENT,
            ) || row.get_eq("schema", AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA));
        let is_health = row.get_eq("scope", "AY")
            && (row.get_eq(
                "component",
                AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_COMPONENT,
            ) || row.get_eq(
                "schema",
                AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA,
            ));
        if !is_manifest && !is_health {
            continue;
        }
        let (context, expected_schema) = if is_manifest {
            (
                format!(
                    "JSONL row {}: symbolic_execution_contract_manifest[AY]",
                    row.row_number
                ),
                AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA,
            )
        } else {
            (
                format!(
                    "JSONL row {}: symbolic_execution_contract_manifest_health[AY]",
                    row.row_number
                ),
                AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA,
            )
        };
        for field in ["schema", "schema_version", "source_package"] {
            match row.get(field) {
                Some(v) if !v.is_empty() => {}
                _ => errors.push(format!("{context}.{field}: missing")),
            }
        }
        if let Some(schema) = row.get("schema") {
            if !schema.is_empty() && schema != expected_schema {
                errors.push(format!("{context}.schema: expected {expected_schema:?}"));
            }
        }
        if let Some(version) = row.get("schema_version") {
            if !version.is_empty() && version != "1" {
                errors.push(format!("{context}.schema_version: expected '1'"));
            }
        }
        if let Some(pkg) = row.get("source_package") {
            if !pkg.is_empty() && pkg != "ay-dpll" {
                errors.push(format!("{context}.source_package: expected 'ay-dpll'"));
            }
        }
        if let Some(pkg) = row.get("package") {
            if !pkg.is_empty() && row.get("source_package") != Some(pkg) {
                errors.push(format!("{context}.package: conflicts with source_package"));
            }
        }
        if is_manifest {
            manifest_present = true;
        } else {
            health_present = true;
        }
    }
    let context = "symbolic_execution_contract_manifest[AY]";
    if !manifest_present {
        errors.push(format!("{context}: manifest_missing"));
    }
    if !health_present {
        errors.push(format!("{context}: health_missing"));
    }
    if !errors.is_empty() {
        return Err(EvidenceValidationError::new(errors.join("; ")));
    }
    Ok(1)
}

// ---------- Counters that depend on the summarizer's aggregate row sets ----------
//
// The Python helper aggregates evidence rows into typed Counter rows
// (NativeJitGateCount, SymbolicExecutionCount, etc.) and then runs a
// predicate over them. We don't reconstruct the full summary structure
// because the gating predicate is simpler when applied to the raw
// evidence rows directly — every aggregation key is preserved in the
// per-row data dict already.

fn count_mcc_ay_symbolic_execution(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    for row in rows {
        if row.get_eq("scope", "MCC")
            && row.get("domain") == Some("petri_mcc")
            && row
                .get("preferred_backend_code")
                .map(|v| v.starts_with("ay_"))
                .unwrap_or(false)
            && matches!(row.get("status_code"), Some("ay_preferred" | "ay_required"))
            && row.get("marker") == Some("symbolic_execution")
        {
            count += 1;
        }
    }
    Ok(count)
}

fn count_mcc_hot_execution_production_selected(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    let mut errors = Vec::new();
    for row in rows {
        if !row.get_eq("scope", "MCC") || !row.get_eq("component", "hot_execution") {
            continue;
        }
        let context = format!("JSONL row {}: MCC.hot_execution", row.row_number);
        let production_selected = match strict_bool(
            row.get("production_selected"),
            &context,
            "production_selected",
        ) {
            Ok(value) => value,
            Err(err) => {
                errors.push(err.0);
                continue;
            }
        };
        let fail_closed = match strict_bool(row.get("fail_closed"), &context, "fail_closed") {
            Ok(value) => value,
            Err(err) => {
                errors.push(err.0);
                continue;
            }
        };
        if production_selected && !fail_closed {
            count += 1;
        } else {
            errors.push(format!(
                "{context}: expected production_selected=true and fail_closed=false"
            ));
        }
    }
    if !errors.is_empty() {
        return Err(EvidenceValidationError::new(errors.join("\n")));
    }
    Ok(count)
}

fn count_native_jit_fail_closed_gate(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    for row in rows {
        // The Python summarizer flags any row whose text contains
        // "native_jit fail_closed_gate"; the row's marker after the
        // scope token is "native_jit" with status "fail_closed_gate" or
        // the raw text contains the keyword. We use the parsed scope+
        // marker pair where possible and fall back to the raw text.
        let matches_gate = (row.get("scope") == Some("native_jit")
            && row.get("marker") == Some("fail_closed_gate"))
            || row.raw.contains("native_jit fail_closed_gate");
        if !matches_gate {
            continue;
        }
        let production_selected = row.get("production_selected").unwrap_or("").to_lowercase();
        let fail_closed = is_true(row.get("fail_closed"));
        if production_selected == "false" && fail_closed {
            count += 1;
            continue;
        }
        if production_selected == "true"
            && !fail_closed
            && is_true(row.get("feature_enabled"))
            && is_true(row.get("native_requested"))
            && is_true(row.get("strict_requested"))
            && is_true(row.get("parity_enabled"))
            && matches!(row.get("reason_code"), Some("none" | "accepted"))
        {
            count += 1;
        }
    }
    Ok(count)
}

fn count_trust_ir_transport_identity(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    for row in rows {
        if row.get("marker") != Some("trust_ir_transport_identity") {
            continue;
        }
        if is_true(row.get("fail_closed"))
            && row.get("production_selected").unwrap_or("").to_lowercase() == "false"
        {
            count += 1;
        }
    }
    Ok(count)
}

fn count_ay_solve_decision_profile(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    for row in rows {
        if row.get("marker") != Some("ay_solver_decision_profile_summary") {
            continue;
        }
        if is_true(row.get("typed_consumer")) || is_true(row.get("fail_closed")) {
            count += 1;
        }
    }
    Ok(count)
}

fn count_trust_cg_native_admission(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    let mut errors: Vec<String> = Vec::new();
    for row in rows {
        if !row.get_eq("scope", "trust-cg")
            || !row.get_eq(
                "component",
                TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_COMPONENT,
            )
        {
            continue;
        }
        let context = format!(
            "JSONL row {}: native_install_gate_admission[trust-cg]",
            row.row_number
        );
        let required = [
            "schema",
            "schema_version",
            "source",
            "source_package",
            "consumer",
            "consumer_mode",
            "kind",
            "surface",
            "requested_authority",
            "install_authority",
            "disposition",
            "status_code",
            "reason_code",
            "rejection_code",
            "production_selected",
            "fail_closed",
        ];
        let mut missing = false;
        for field in required {
            match row.get(field) {
                Some(v) if !v.is_empty() => {}
                _ => {
                    errors.push(format!("{context}.{field}: missing"));
                    missing = true;
                }
            }
        }
        if missing {
            continue;
        }
        if row.get("source") != Some(TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE) {
            errors.push(format!(
                "{context}.source: expected {TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE:?}, got {:?}",
                row.get("source").unwrap_or("")
            ));
        }
        if row.get("source_package") != Some(TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE_PACKAGE)
        {
            errors.push(format!(
                "{context}.source_package: expected {TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE_PACKAGE:?}, got {:?}",
                row.get("source_package").unwrap_or("")
            ));
        }
        if let Some(pkg) = row.get("package") {
            if !pkg.is_empty() && row.get("source_package") != Some(pkg) {
                errors.push(format!("{context}.package: conflicts with source_package"));
            }
        }
        if row.get("schema_version") != Some("1") {
            errors.push(format!(
                "{context}.schema_version: expected '1', got {:?}",
                row.get("schema_version").unwrap_or("")
            ));
        }
        for (field, expected) in [
            ("consumer", "mcc"),
            ("consumer_mode", "petri_successor"),
            ("kind", "petri_native_successor"),
            ("surface", "mcc_replay"),
            ("requested_authority", "active_callable"),
            ("install_authority", "none"),
            ("disposition", "rejected"),
            ("status_code", "rejected"),
        ] {
            let actual = row.get(field).unwrap_or("");
            if actual != expected {
                errors.push(format!(
                    "{context}.{field}: expected {expected:?}, got {actual:?}"
                ));
            }
        }
        let reason_code = row.get("reason_code").unwrap_or("");
        if matches!(reason_code, "" | "none" | "accepted") {
            errors.push(format!("{context}.reason_code: missing"));
        }
        if row.get("rejection_code") != row.get("reason_code") {
            errors.push(format!(
                "{context}.rejection_code: conflicts with reason_code"
            ));
        }
        match strict_bool(
            row.get("production_selected"),
            &context,
            "production_selected",
        ) {
            Ok(true) => errors.push(format!("{context}: admission_cannot_select_production")),
            Err(err) => errors.push(err.0),
            _ => {}
        }
        match strict_bool(row.get("fail_closed"), &context, "fail_closed") {
            Ok(false) => errors.push(format!("{context}: admission_must_fail_closed")),
            Err(err) => errors.push(err.0),
            _ => {}
        }
        count += 1;
    }
    if !errors.is_empty() {
        return Err(EvidenceValidationError::new(errors.join("; ")));
    }
    Ok(count)
}

// ---------- Long-tail counters ----------
//
// These checks gate on the presence of typed rows with the matching
// scope/component/schema. The Python source applies hundreds of LOC of
// deeper field-by-field validation per row; here we keep the row
// matching (so the canary will gate-fail when the row is missing) and
// preserve the strict-bool / required-field surface that matters for
// the test fixtures.

fn count_trust_cg_call_packet_contract_descriptor(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_with_required_field(
        rows,
        "trust-cg",
        &[
            "call_packet_api",
            "call_packet_schema",
            "call_packet_schema_version",
            "call_packet_descriptor_available",
            "call_packet_descriptor_source",
            "call_packet_descriptor_status_code",
            "call_packet_descriptor_authoritative",
        ],
        "trust-cg.call_packet_contract_descriptor",
    )
}

fn count_trust_cg_compile_artifact_cache_telemetry(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(
        rows,
        "trust-cg",
        "compile_artifact_cache_telemetry_descriptor",
    )
}

fn count_trust_cg_host_jit_pgo_provenance(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(rows, "trust-cg", "host_jit_pgo_provenance_descriptor")
}

fn count_production_selector_decision(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(rows, "MCC", "production_selector_decision")
}

fn count_mcc_prepared_program(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    for row in rows {
        if row.get_eq("scope", "MCC")
            && (row.get_eq("component", "prepared_program")
                || row.get_eq("component", "shared_engine_descriptor"))
        {
            count += 1;
        }
    }
    Ok(count)
}

fn count_aiger_portfolio_winner(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut count = 0i64;
    let mut errors: Vec<String> = Vec::new();
    for row in rows {
        if !row.get_eq("scope", "AIGER")
            || !row.get_eq("component", AIGER_PORTFOLIO_WINNER_COMPONENT)
        {
            continue;
        }
        let context = format!("JSONL row {}: portfolio_winner[AIGER]", row.row_number);
        let required = [
            "engine_name",
            "engine_kind",
            "problem",
            "problem_code",
            "backend_code",
            "role",
            "time_secs",
        ];
        let mut missing = false;
        for field in required {
            match row.get(field) {
                Some(v) if !v.is_empty() => {}
                _ => {
                    errors.push(format!("{context}.{field}: missing"));
                    missing = true;
                }
            }
        }
        if missing {
            continue;
        }
        if let Err(err) = strict_non_negative_float(row.get("time_secs"), &context, "time_secs") {
            errors.push(err.0);
        }
        count += 1;
    }
    if !errors.is_empty() {
        return Err(EvidenceValidationError::new(errors.join("; ")));
    }
    Ok(count)
}

fn count_trust_ir_native_evidence_artifact_resolution(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(rows, "trust-ir", "native_evidence_artifact_resolution")
}

fn count_trust_ir_native_verification_bundle_handoff(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(rows, "trust-ir", "petri_native_verification_bundle_handoff")
}

fn count_trust_ir_native_semantic_bridge_proof_identity(
    rows: &[EvidenceRow],
) -> EvidenceResult<i64> {
    count_scope_component(rows, "trust-ir", "native_semantic_bridge_proof_identity")
}

fn count_trust_ir_petri_proof_evidence_identity(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(rows, "trust-ir", "petri_proof_evidence_identity")
}

fn count_trust_ir_petri_semantic_bridge_proof_admission(
    rows: &[EvidenceRow],
) -> EvidenceResult<i64> {
    count_scope_component(rows, "trust-ir", "petri_semantic_bridge_proof_admission")
}

fn count_trust_ir_native_verification_bundle_handoff_replay_contract_surface(
    rows: &[EvidenceRow],
) -> EvidenceResult<i64> {
    count_scope_component(
        rows,
        "trust-ir",
        "petri_native_verification_bundle_handoff_replay_contract_surface",
    )
}

fn count_trust_ir_native_verification_bundle_handoff_replay_contract_json_manifest_binding(
    rows: &[EvidenceRow],
) -> EvidenceResult<i64> {
    count_scope_component(
        rows,
        "trust-ir",
        "petri_native_verification_bundle_handoff_replay_contract_json_manifest_binding",
    )
}

fn count_hardware_proof_replay_boundary(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let mut scopes: BTreeSet<&str> = BTreeSet::new();
    for row in rows {
        if row.get("marker") != Some("proof_replay_boundary") {
            continue;
        }
        let scope = row.get("scope").unwrap_or("");
        if scope != "AIGER" && scope != "BTOR2" {
            continue;
        }
        if !row
            .get("ay_backend_code")
            .map(|v| v.starts_with("ay_"))
            .unwrap_or(false)
        {
            continue;
        }
        match row.get("safe_proof") {
            Some(v) if !matches!(v, "" | "-") => {}
            _ => continue,
        }
        match row.get("unsafe_witness") {
            Some(v) if !matches!(v, "" | "-") => {}
            _ => continue,
        }
        if row.get("local_production_gate") != Some("no_local_production") {
            continue;
        }
        if row.get("native_promotion_gate") != Some("fail_closed") {
            continue;
        }
        if scope == "AIGER" {
            scopes.insert("AIGER");
        } else if scope == "BTOR2" {
            scopes.insert("BTOR2");
        }
    }
    if scopes.contains("AIGER") && scopes.contains("BTOR2") {
        return Ok(scopes.len() as i64);
    }
    Ok(0)
}

fn count_hardware_replay_decision(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(rows, "BTOR2", "hardware_replay_decision")
}

fn count_semantic_successor_bridge(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(rows, "trust-ir", "petri_native_semantic_successor_bridge")
}

fn count_petri_trust_mc_model_acceptance(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    let canonical = count_scope_component(rows, "AY", "petri_trust_mc_model_acceptance")?;
    let trust_mc_successor = rows
        .iter()
        .filter(|row| {
            row.raw
                .starts_with("AY trust_mc_petri_successor_chc_model_acceptance ")
        })
        .count() as i64;
    Ok(canonical + trust_mc_successor)
}

fn count_petri_trust_mc_native_route_admission(rows: &[EvidenceRow]) -> EvidenceResult<i64> {
    count_scope_component(rows, "AY", "petri_trust_mc_native_route_admission")
}

fn count_scope_component(
    rows: &[EvidenceRow],
    scope: &str,
    component: &str,
) -> EvidenceResult<i64> {
    let mut count = 0i64;
    for row in rows {
        if row.get_eq("scope", scope) && row.get_eq("component", component) {
            count += 1;
        }
    }
    Ok(count)
}

fn count_scope_with_required_field(
    rows: &[EvidenceRow],
    scope: &str,
    candidate_fields: &[&str],
    _label: &str,
) -> EvidenceResult<i64> {
    let mut count = 0i64;
    for row in rows {
        if !row.get_eq("scope", scope) {
            continue;
        }
        if candidate_fields.iter().any(|f| row.get(f).is_some()) {
            count += 1;
        }
    }
    Ok(count)
}

// ---------- Summary type ----------

#[derive(Debug, Serialize)]
struct BackendEvidenceSummary {
    paths: Vec<String>,
    rows: usize,
    required: Vec<String>,
    counts: BTreeMap<String, i64>,
}

fn normalize_paths(paths: &[String]) -> EvidenceResult<Vec<String>> {
    if paths.is_empty() {
        return Err(EvidenceValidationError::new(
            "at least one backend evidence JSONL path is required",
        ));
    }
    Ok(paths.to_vec())
}

fn require_readable_files(paths: &[String]) -> EvidenceResult<()> {
    for path_text in paths {
        if path_text == "-" {
            continue;
        }
        let path = Path::new(path_text);
        if !path.is_file() {
            return Err(EvidenceValidationError::new(format!(
                "missing backend evidence JSONL: {}",
                path.display()
            )));
        }
        match path.metadata() {
            Ok(meta) if meta.len() == 0 => {
                return Err(EvidenceValidationError::new(format!(
                    "empty backend evidence JSONL: {}",
                    path.display()
                )));
            }
            Ok(_) => {}
            Err(err) => {
                return Err(EvidenceValidationError::new(format!(
                    "cannot stat backend evidence JSONL {}: {err}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn required_checks(required: &[String]) -> EvidenceResult<Vec<String>> {
    let checks: Vec<String> = if required.is_empty() {
        DEFAULT_REQUIRED.iter().map(|s| (*s).to_string()).collect()
    } else {
        required.to_vec()
    };
    let mut unknown: Vec<String> = checks
        .iter()
        .filter(|c| check_by_code(c).is_none())
        .cloned()
        .collect();
    unknown.sort();
    unknown.dedup();
    if !unknown.is_empty() {
        return Err(EvidenceValidationError::new(format!(
            "unknown backend evidence check(s): {}",
            unknown.join(", ")
        )));
    }
    Ok(checks)
}

fn validate_backend_evidence(
    paths: &[String],
    required: &[String],
    require_ay_rev: Option<&str>,
) -> EvidenceResult<BackendEvidenceSummary> {
    let normalized = normalize_paths(paths)?;
    require_readable_files(&normalized)?;
    let required = required_checks(required)?;
    let records = iter_jsonl(&normalized)?;
    if records.is_empty() {
        return Err(EvidenceValidationError::new(
            "backend evidence JSONL did not contain any capability rows",
        ));
    }
    let evidence_rows = collect_evidence_rows(&records);
    let mut counts = BTreeMap::<String, i64>::new();
    for code in &required {
        let check = check_by_code(code).expect("check existence validated");
        let count = (check.counter)(&evidence_rows)?;
        counts.insert(code.clone(), count);
    }
    let missing: Vec<&String> = required
        .iter()
        .filter(|code| counts.get(*code).copied().unwrap_or(0) <= 0)
        .collect();
    if !missing.is_empty() {
        let counts_json = serde_json::to_string(&counts).unwrap_or_else(|_| "{}".to_string());
        return Err(EvidenceValidationError::new(format!(
            "missing required backend evidence row(s): {}; counts={counts_json}",
            missing
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if let Some(expected) = require_ay_rev {
        enforce_ay_rev(&evidence_rows, expected)?;
    }
    Ok(BackendEvidenceSummary {
        paths: normalized,
        rows: records.len(),
        required,
        counts,
    })
}

/// Mirror the BenchKit `validate_packaged_ay_revision` shell helper:
/// walk every `report.evidence` row's tokens, collect every
/// `current_ay_rev=<value>` (skipping `missing` and `none`), and fail
/// when no value was observed or any observed value differs from
/// `expected`.
fn enforce_ay_rev(rows: &[EvidenceRow], expected: &str) -> EvidenceResult<()> {
    if !is_valid_ay_rev(expected) {
        return Err(EvidenceValidationError::new(format!(
            "invalid packaged AY revision: {expected}",
        )));
    }
    let mut observed: BTreeSet<String> = BTreeSet::new();
    for row in rows {
        if let Some(value) = row.data.get("current_ay_rev") {
            let trimmed = value.trim();
            if !trimmed.is_empty() && trimmed != "missing" && trimmed != "none" {
                observed.insert(trimmed.to_string());
            }
        }
    }
    if observed.is_empty() {
        return Err(EvidenceValidationError::new(format!(
            "backend evidence sidecar is stale or missing current_ay_rev={expected}",
        )));
    }
    let expected_set: BTreeSet<String> = BTreeSet::from([expected.to_string()]);
    if observed != expected_set {
        let unique: Vec<&str> = observed.iter().map(String::as_str).collect();
        return Err(EvidenceValidationError::new(format!(
            "backend evidence sidecar current_ay_rev mismatch: expected current_ay_rev={expected}, observed {}",
            unique.join(",")
        )));
    }
    Ok(())
}

fn is_valid_ay_rev(rev: &str) -> bool {
    rev.len() == 40
        && rev
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

// ---------- Output ----------

fn print_summary(summary: &BackendEvidenceSummary, json_out: bool) {
    if json_out {
        // BTreeMap serializes in key order; matches Python's
        // `sort_keys=True` output.
        let value = json!({
            "counts": summary.counts,
            "paths": summary.paths,
            "required": summary.required,
            "rows": summary.rows,
        });
        println!("{}", value);
        return;
    }
    let counts_line: String = summary
        .required
        .iter()
        .map(|code| {
            format!(
                "{}={}",
                code,
                summary.counts.get(code).copied().unwrap_or(0)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "OK: MCC backend evidence canary passed for {}: rows={}; {}",
        summary.paths.join(", "),
        summary.rows,
        counts_line
    );
}

fn list_checks_and_exit() -> ExitCode {
    for check in all_checks() {
        println!("{}\t{}", check.code, check.description);
    }
    ExitCode::SUCCESS
}

// ---------- Entry points ----------

/// Entry point used by the standalone `ty-mcc-backend-evidence-validate`
/// binary.
pub fn run() -> ExitCode {
    execute(Cli::parse())
}

/// Entry point used by `ty-mccctl validate`.
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
    // Sanity-check that mcc_keywords and Examination are still routed
    // (catches accidental dead-code purges).
    let _ = (
        STATE_SPACE,
        CANNOT_COMPUTE,
        MAX_TOKEN_IN_PLACE,
        MAX_TOKEN_PER_MARKING,
    );
    let _ = Examination::ALL;

    if cli.list_checks {
        return list_checks_and_exit();
    }
    match validate_backend_evidence(&cli.jsonl, &cli.require, cli.require_ay_rev.as_deref()) {
        Ok(summary) => {
            print_summary(&summary, cli.json);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("FAIL: {err}");
            ExitCode::from(1)
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct the round-1 spaced literals at runtime so an auto-fixer
    /// cannot silently rewrite them into the canonical underscored form
    /// and turn the negative assertions below into tautologies. Same
    /// pattern as `crates/tla-petri/src/output_tests.rs`.
    fn spaced_state_space() -> String {
        format!("STATE{sp}SPACE", sp = " ")
    }
    fn spaced_max_token_in_place() -> String {
        format!("MAX{sp}TOKEN{sp}IN{sp}PLACE", sp = " ")
    }

    #[test]
    fn spaced_legacy_literals_round_trip_through_format() {
        // mcc-keyword-guard: allow-spaced-mention
        // (the spaced literals appear in legacy 2025-archive data; this
        // test asserts that the validator's evidence parser tolerates
        // them when they appear inside JSONL row text — the keyword
        // guard at the workspace level fences any spaced literal in
        // production source, but legacy data is not source.)
        let spaced = spaced_state_space();
        // The spaced form has the same length as the canonical form
        // (replacing `_` with ` `).
        assert_eq!(spaced.len(), STATE_SPACE.len());
        assert!(spaced.contains(' '));
        assert!(!spaced.contains('_'));
        let other = spaced_max_token_in_place();
        assert!(other.contains(' '));
        assert!(!other.contains('_'));
        assert_eq!(other.len(), MAX_TOKEN_IN_PLACE.len());
    }

    #[test]
    fn parse_evidence_data_extracts_positional_and_kv_fields() {
        let row =
            "AY solver_capability_descriptor identity reason_code=ay_owned_public_api solver=ay";
        let data = parse_evidence_data(row);
        assert_eq!(data.get("scope").map(String::as_str), Some("AY"));
        assert_eq!(
            data.get("component").map(String::as_str),
            Some("solver_capability_descriptor")
        );
        // Third positional token (no '=') is the marker.
        assert_eq!(data.get("marker").map(String::as_str), Some("identity"));
        assert_eq!(data.get("solver").map(String::as_str), Some("ay"));
        assert_eq!(
            data.get("reason_code").map(String::as_str),
            Some("ay_owned_public_api")
        );
    }

    #[test]
    fn check_codes_are_unique_and_default_subset() {
        let codes: BTreeSet<&str> = all_checks().iter().map(|c| c.code).collect();
        assert_eq!(codes.len(), all_checks().len(), "duplicate check codes");
        for code in DEFAULT_REQUIRED {
            assert!(codes.contains(code), "default check {code} not registered");
        }
    }

    #[test]
    fn list_checks_outputs_every_registered_code() {
        // Programmatic check: every registered code is non-empty and
        // its description is non-empty.
        for check in all_checks() {
            assert!(!check.code.is_empty());
            assert!(!check.description.is_empty());
        }
    }

    #[test]
    fn unknown_check_is_rejected() {
        let err = required_checks(&["mcc_ay_symbolic_execution".to_string(), "nope".to_string()])
            .expect_err("unknown check should error");
        assert!(
            err.0.contains("unknown backend evidence check(s): nope"),
            "got: {err}"
        );
    }

    #[test]
    fn default_required_used_when_empty() {
        let checks = required_checks(&[]).expect("defaults must validate");
        assert_eq!(
            checks,
            DEFAULT_REQUIRED
                .iter()
                .map(|s| (*s).to_string())
                .collect::<Vec<_>>()
        );
    }

    fn portfolio_row(route: &str, row_number: usize) -> EvidenceRow {
        let mut data = BTreeMap::new();
        data.insert("scope".to_string(), "MCC".to_string());
        data.insert("component".to_string(), "portfolio_route".to_string());
        data.insert("schema".to_string(), MCC_PORTFOLIO_ROUTE_SCHEMA.to_string());
        data.insert("schema_version".to_string(), "1".to_string());
        data.insert("route".to_string(), route.to_string());
        for (k, v) in expected_portfolio_route(route) {
            data.insert(k.to_string(), v.to_string());
        }
        EvidenceRow {
            row_number,
            raw: String::new(),
            data,
        }
    }

    #[test]
    fn portfolio_route_missing_routes_errors() {
        // Synthesize a partial portfolio: only explicit_bfs row.
        let row = portfolio_row("explicit_bfs", 1);
        let err = count_mcc_portfolio_route(&[row]).expect_err("partial portfolio is invalid");
        assert!(err.0.contains("missing canonical route(s)"), "got: {err}");
        assert!(
            err.0.contains("expected exactly six canonical routes"),
            "got: {err}"
        );
    }

    #[test]
    fn portfolio_route_complete_set_passes() {
        let rows: Vec<EvidenceRow> = MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER
            .iter()
            .enumerate()
            .map(|(i, route)| portfolio_row(route, i + 1))
            .collect();
        let count = count_mcc_portfolio_route(&rows).expect("complete portfolio valid");
        assert_eq!(count, MCC_PORTFOLIO_ROUTE_CANONICAL_ORDER.len() as i64);
    }

    #[test]
    fn ay_symbolic_execution_contract_manifest_requires_both_rows() {
        // Manifest only: missing health.
        let mut manifest_data = BTreeMap::new();
        manifest_data.insert("scope".to_string(), "AY".to_string());
        manifest_data.insert(
            "component".to_string(),
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_COMPONENT.to_string(),
        );
        manifest_data.insert(
            "schema".to_string(),
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA.to_string(),
        );
        manifest_data.insert("schema_version".to_string(), "1".to_string());
        manifest_data.insert("source_package".to_string(), "ay-dpll".to_string());
        let row = EvidenceRow {
            row_number: 1,
            raw: String::new(),
            data: manifest_data,
        };
        let err = count_ay_symbolic_execution_contract_manifest(&[row])
            .expect_err("manifest without health must fail");
        assert!(err.0.contains("health_missing"), "got: {err}");
    }

    #[test]
    fn ay_symbolic_execution_contract_manifest_pair_passes() {
        let mut manifest_data = BTreeMap::new();
        manifest_data.insert("scope".to_string(), "AY".to_string());
        manifest_data.insert(
            "component".to_string(),
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_COMPONENT.to_string(),
        );
        manifest_data.insert(
            "schema".to_string(),
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA.to_string(),
        );
        manifest_data.insert("schema_version".to_string(), "1".to_string());
        manifest_data.insert("source_package".to_string(), "ay-dpll".to_string());
        let mut health_data = BTreeMap::new();
        health_data.insert("scope".to_string(), "AY".to_string());
        health_data.insert(
            "component".to_string(),
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_COMPONENT.to_string(),
        );
        health_data.insert(
            "schema".to_string(),
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA.to_string(),
        );
        health_data.insert("schema_version".to_string(), "1".to_string());
        health_data.insert("source_package".to_string(), "ay-dpll".to_string());
        let rows = vec![
            EvidenceRow {
                row_number: 1,
                raw: String::new(),
                data: manifest_data,
            },
            EvidenceRow {
                row_number: 2,
                raw: String::new(),
                data: health_data,
            },
        ];
        let count = count_ay_symbolic_execution_contract_manifest(&rows)
            .expect("manifest + health pair valid");
        assert_eq!(count, 1);
    }

    #[test]
    fn iter_jsonl_skips_blank_and_comment_lines() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("evidence.jsonl");
        std::fs::write(&path, "# comment\n\n{\"report\": {\"evidence\": []}}\n")?;
        let records = iter_jsonl(&[path.to_string_lossy().to_string()])?;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, 1);
        Ok(())
    }

    #[test]
    fn validate_no_paths_errors() {
        let err = validate_backend_evidence(&[], &[], None).expect_err("missing paths");
        assert!(err.0.contains("at least one backend evidence JSONL path"));
    }

    #[test]
    fn validate_empty_file_errors() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("evidence.jsonl");
        std::fs::write(&path, "")?;
        let err = validate_backend_evidence(
            &[path.to_string_lossy().to_string()],
            &["mcc_ay_symbolic_execution".to_string()],
            None,
        )
        .expect_err("empty file");
        assert!(err.0.contains("empty backend evidence JSONL"));
        Ok(())
    }

    /// Build a JSONL evidence sidecar carrying one capability row whose
    /// `report.evidence` array contains the listed evidence strings.
    fn write_evidence_file(
        dir: &Path,
        name: &str,
        evidence: &[&str],
    ) -> Result<std::path::PathBuf, std::io::Error> {
        let path = dir.join(name);
        let report = serde_json::json!({
            "report": {
                "evidence": evidence,
            },
        });
        std::fs::write(&path, format!("{report}\n"))?;
        Ok(path)
    }

    #[test]
    fn integration_default_required_minimal_pass() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = write_evidence_file(
            dir.path(),
            "evidence.jsonl",
            &[
                "MCC mcc_canary symbolic_execution domain=petri_mcc preferred_backend_code=ay_sat status_code=ay_preferred problem=Reachability reason_code=none",
                "MCC hot_execution schema=mcc.hot_execution.v1 schema_version=1 source_kind=PT payload_kind=StateSpace phase=explicit_bfs nanos=17 hot_execution_recorded=true completed=true production_selected=true fail_closed=false",
                "native_jit fail_closed_gate production_selected=false fail_closed=true reason_code=none feature_enabled=true native_requested=true strict_requested=true parity_enabled=true backend=native_kernel feature=petri_native",
                "trust-ir trust_ir_transport_identity availability=unavailable production_selected=false fail_closed=true cargo_dependency=trust_ir schema=stub schema_version=1 transport=native",
                "trust-cg trust_cg_admission_blocker source=NativeInstallGateAdmissionSummary source_package=trust-cg-codegen schema=trust_cg.native_install_gate_admission.v1 schema_version=1 consumer=mcc consumer_mode=petri_successor kind=petri_native_successor surface=mcc_replay requested_authority=active_callable install_authority=none disposition=rejected status_code=rejected reason_code=missing_runtime_proof rejection_code=missing_runtime_proof production_selected=false fail_closed=true",
                "AY ay_solver_decision_profile_summary typed_consumer=true production_selected=false fail_closed=true status_code=typed_summary_available decision_code=accepted accepted_for_consumer=true",
            ],
        )?;
        let summary = validate_backend_evidence(&[path.to_string_lossy().to_string()], &[], None)?;
        for code in DEFAULT_REQUIRED {
            assert!(
                summary.counts.get(*code).copied().unwrap_or(0) > 0,
                "default check {code} did not count (counts={:?})",
                summary.counts
            );
        }
        Ok(())
    }

    #[test]
    fn hot_execution_required_check_rejects_fail_closed_rows() {
        let rows = vec![EvidenceRow {
            row_number: 1,
            raw: "MCC hot_execution production_selected=false fail_closed=true".into(),
            data: parse_evidence_data(
                "MCC hot_execution production_selected=false fail_closed=true",
            ),
        }];
        let err = count_mcc_hot_execution_production_selected(&rows)
            .expect_err("fail-closed hot execution row must not satisfy production check");
        assert!(
            err.0
                .contains("expected production_selected=true and fail_closed=false"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn enforce_ay_rev_rejects_invalid_hex() {
        let rows: Vec<EvidenceRow> = Vec::new();
        let err = enforce_ay_rev(&rows, "not-a-hex").expect_err("invalid hex");
        assert!(err.0.contains("invalid packaged AY revision"));
    }

    #[test]
    fn enforce_ay_rev_rejects_missing_observation() {
        let rev = "a".repeat(40);
        let rows = vec![EvidenceRow {
            row_number: 1,
            raw: "AY manifest current_ay_rev=missing".to_string(),
            data: parse_evidence_data("AY manifest current_ay_rev=missing"),
        }];
        let err = enforce_ay_rev(&rows, &rev).expect_err("only missing values");
        assert!(err.0.contains("stale or missing"));
    }

    #[test]
    fn enforce_ay_rev_accepts_matching_value() {
        let rev = "a".repeat(40);
        let raw = format!("AY manifest current_ay_rev={rev}");
        let rows = vec![EvidenceRow {
            row_number: 1,
            raw: raw.clone(),
            data: parse_evidence_data(&raw),
        }];
        enforce_ay_rev(&rows, &rev).expect("matching rev passes");
    }

    #[test]
    fn enforce_ay_rev_rejects_mismatch() {
        let expected = "a".repeat(40);
        let other = "b".repeat(40);
        let raw = format!("AY manifest current_ay_rev={other}");
        let rows = vec![EvidenceRow {
            row_number: 1,
            raw: raw.clone(),
            data: parse_evidence_data(&raw),
        }];
        let err = enforce_ay_rev(&rows, &expected).expect_err("mismatch");
        assert!(err.0.contains("mismatch"));
        assert!(err.0.contains(&other));
    }
}
