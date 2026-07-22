// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Output parsers for single-thread supremacy benchmark artifacts.
//!
//! These helpers own the parsing contract for Rust `ty supremacy benchmark`
//! artifacts.

use regex::Regex;
use serde::Serialize;

const FLAT_PRIMARY_REBUILD_MARKER: &str =
    "[compiled-bfs] clearing layout-sensitive compiled artifacts before rebuild: \
     reason=flat_state_primary layout promotion";
const MAX_FALLBACK_REASONS: usize = 12;
const MAX_REASON_CHARS: usize = 240;
const TRUNCATED_FALLBACK_REASON: &str =
    "[benchmark] trust-codegen fallback reasons truncated; hidden reasons exist";

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(super) struct ParsedRunCounts {
    pub(super) states_found: Option<u64>,
    pub(super) distinct_states: Option<u64>,
    pub(super) states_generated: Option<u64>,
    pub(super) states_left: Option<u64>,
    pub(super) transitions: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(super) struct TrustCgTelemetry {
    pub(super) trust_cg_actions_compiled: Option<u64>,
    pub(super) trust_cg_actions_total: Option<u64>,
    pub(super) trust_cg_invariants_compiled: Option<u64>,
    pub(super) trust_cg_invariants_total: Option<u64>,
    pub(super) trust_cg_state_constraints_compiled: Option<u64>,
    pub(super) trust_cg_state_constraints_total: Option<u64>,
    pub(super) compiled_bfs_step_active: bool,
    pub(super) compiled_bfs_level_active: bool,
    pub(super) compiled_bfs_level_loop_started: bool,
    pub(super) compiled_bfs_level_loop_initial_states: Option<u64>,
    pub(super) compiled_bfs_level_loop_fused: Option<bool>,
    pub(super) compiled_bfs_levels_completed: Option<u64>,
    pub(super) compiled_bfs_parents_processed: Option<u64>,
    pub(super) compiled_bfs_successors_generated: Option<u64>,
    pub(super) compiled_bfs_successors_new: Option<u64>,
    pub(super) compiled_bfs_total_states: Option<u64>,
    pub(super) compiled_bfs_zero_work: bool,
    pub(super) compiled_bfs_execution_nanos: Option<u64>,
    pub(super) compiled_bfs_execution_seconds: Option<f64>,
    pub(super) trust_cg_bfs_level_active: bool,
    pub(super) trust_cg_native_fused_level_built: bool,
    pub(super) trust_cg_native_fused_level_active: bool,
    pub(super) trust_cg_native_fused_regular_invariants_checked: Option<bool>,
    pub(super) trust_cg_native_fused_mode: Option<String>,
    pub(super) trust_cg_native_fused_invariant_count: Option<u64>,
    pub(super) trust_cg_native_fused_state_constraint_count: Option<u64>,
    pub(super) trust_cg_native_fused_state_len: Option<u64>,
    pub(super) trust_cg_native_fused_local_dedup: Option<bool>,
    pub(super) trust_cg_native_bfs_trace_generated: Option<u64>,
    pub(super) trust_cg_native_bfs_trace_state_count: Option<u64>,
    pub(super) trust_cg_native_bfs_trace_parents_processed: Option<u64>,
    pub(super) trust_cg_bfs_level_loop_kind: Option<String>,
    pub(super) trust_cg_native_fused_flat_frontier_admission_active: Option<bool>,
    pub(super) compiled_bfs_flat_frontier_admitted: Option<bool>,
    pub(super) flat_state_primary: Option<bool>,
    pub(super) flat_bfs_frontier_active: Option<bool>,
    pub(super) flat_bfs_frontier_fallbacks: Option<u64>,
    pub(super) native_action_callout_batch_artifact_identity_source: Option<String>,
    pub(super) native_action_callout_batch_artifact_identity: Option<String>,
    pub(super) native_action_callout_batch_artifact_cache_digest: Option<String>,
    pub(super) native_action_callout_batch_cache_key: Option<String>,
    pub(super) native_action_callout_batch_artifact_cacheable: Option<bool>,
    pub(super) native_action_callout_batch_artifact_cache_disabled_by_env: Option<bool>,
    pub(super) native_action_callout_batch_shard_count: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_enabled: Option<bool>,
    pub(super) native_action_callout_batch_warm_cache_lookup_attempted: Option<bool>,
    pub(super) native_action_callout_batch_warm_cache_hits: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_misses: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_stores: Option<u64>,
    pub(super) native_action_callout_batch_setup_ms: Option<u64>,
    pub(super) native_action_callout_batch_lowering_ms: Option<u64>,
    pub(super) native_action_callout_batch_assembly_ms: Option<u64>,
    pub(super) native_action_callout_batch_compile_ms: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_lookup_ms: Option<u64>,
    pub(super) native_action_callout_batch_artifact_materialization_ms: Option<u64>,
    pub(super) native_action_callout_batch_fallback_per_action_compile_ms: Option<u64>,
    pub(super) native_action_callout_batch_shard_warm_cache_statuses: Vec<String>,
    pub(super) fallback_reasons: Vec<String>,
}

pub(super) fn parse_tlc_final_counts(stdout: &str, stderr: &str) -> ParsedRunCounts {
    let combined = format!("{stdout}\n{stderr}");
    let initial_summary = Regex::new(
        r"(?m)^\s*Finished computing initial states:\s+([0-9][0-9,]*)\s+distinct states generated\b",
    )
    .unwrap();
    let final_summary = Regex::new(concat!(
        r"(?m)^\s*([0-9][0-9,]*)\s+states generated,\s+",
        r"([0-9][0-9,]*)\s+distinct states found,\s+",
        r"([0-9][0-9,]*)\s+states left(?:\s+on queue\.)?\s*$",
    ))
    .unwrap();
    let mut counts = ParsedRunCounts::default();
    let mut initial_states = None;
    for captures in initial_summary.captures_iter(&combined) {
        initial_states = parse_count(&captures[1]);
    }
    for captures in final_summary.captures_iter(&combined) {
        let generated = parse_count(&captures[1]);
        let distinct = parse_count(&captures[2]);
        counts.states_generated = generated;
        counts.states_found = distinct;
        counts.distinct_states = distinct;
        counts.states_left = parse_count(&captures[3]);
    }
    if let (Some(generated), Some(initial)) = (counts.states_generated, initial_states) {
        counts.transitions = generated.checked_sub(initial);
    }
    counts
}

pub(super) fn parse_ty_final_counts(stdout: &str, stderr: &str) -> ParsedRunCounts {
    let combined = format!("{stdout}\n{stderr}");
    let states_found = Regex::new(
        r"(?m)^\s*(?:States found:\s+([0-9][0-9,]*)|([0-9][0-9,]*)\s+states?\s+found\.?)\s*$",
    )
    .unwrap();
    let transitions = Regex::new(r"(?m)^\s*Transitions:\s+([0-9][0-9,]*)\s*$").unwrap();
    let mut counts = ParsedRunCounts::default();
    for captures in states_found.captures_iter(&combined) {
        let states = captures
            .get(1)
            .or_else(|| captures.get(2))
            .and_then(|matched| parse_count(matched.as_str()));
        counts.states_found = states;
        counts.distinct_states = states;
    }
    for captures in transitions.captures_iter(&combined) {
        counts.transitions = parse_count(&captures[1]);
    }
    counts
}

pub(super) fn parse_trust_cg_telemetry(stdout: &str, stderr: &str) -> TrustCgTelemetry {
    let combined = format!("{stdout}\n{stderr}");
    let combined = latest_flat_primary_backend_segment(&combined);
    let mut telemetry = TrustCgTelemetry {
        fallback_reasons: Vec::new(),
        ..TrustCgTelemetry::default()
    };
    let mut action_compiled_lines = 0;
    let mut action_failed_lines = 0;
    let mut invariant_compiled_lines = 0;
    let mut invariant_failed_lines = 0;
    let mut state_constraint_compiled_lines = 0;
    let mut state_constraint_failed_lines = 0;
    let mut saw_no_safe_actions = false;
    let mut explicit_native_fused_level_active = None;

    let numeric_kv = Regex::new(concat!(
        r"\b(trust_cg_actions_compiled|trust_cg_actions_total|",
        r"trust_cg_invariants_compiled|trust_cg_invariants_total|",
        r"trust_cg_state_constraints_compiled|trust_cg_state_constraints_total|",
        r"trust_cg_native_fused_invariant_count|",
        r"trust_cg_native_fused_state_constraint_count|",
        r"trust_cg_native_fused_state_len|",
        r"compiled_bfs_level_loop_initial_states|",
        r"compiled_bfs_levels_completed|compiled_bfs_parents_processed|",
        r"compiled_bfs_successors_generated|compiled_bfs_successors_new|",
        r"compiled_bfs_total_states|flat_bfs_frontier_fallbacks)",
        r"\s*[:=]\s*([0-9][0-9,]*)\b",
    ))
    .unwrap();
    let bool_kv = Regex::new(concat!(
        r"(?i)\b(compiled_bfs_step_active|compiled_bfs_level_active|",
        r"compiled_bfs_level_loop_started|compiled_bfs_level_loop_fused|",
        r"compiled_bfs_flat_frontier_admitted|",
        r"trust_cg_bfs_level_active|trust_cg_native_fused_level_built|",
        r"trust_cg_native_fused_level_active|",
        r"trust_cg_native_fused_regular_invariants_checked|",
        r"trust_cg_native_fused_local_dedup|",
        r"trust_cg_native_fused_flat_frontier_admission_active|flat_state_primary|",
        r"flat_bfs_frontier_active)\s*[:=]\s*(true|false)\b",
    ))
    .unwrap();
    let compiling = Regex::new(
        r"(?i)\[trust[_-]cg\]\s+compiling\s+([0-9][0-9,]*)\s+actions\s+\(([0-9][0-9,]*)\s+invariants,",
    )
    .unwrap();
    let compilation_complete_constraints = Regex::new(concat!(
        r"(?i)\[trust[_-]cg\]\s+compilation complete:\s+",
        r"([0-9][0-9,]*)/([0-9][0-9,]*)\s+actions,\s+",
        r"([0-9][0-9,]*)/([0-9][0-9,]*)\s+invariants,\s+",
        r"([0-9][0-9,]*)/([0-9][0-9,]*)\s+state constraints compiled",
    ))
    .unwrap();
    let compilation_complete_invariants = Regex::new(concat!(
        r"(?i)\[trust[_-]cg\]\s+compilation complete:\s+",
        r"([0-9][0-9,]*)/([0-9][0-9,]*)\s+actions,\s+",
        r"([0-9][0-9,]*)/([0-9][0-9,]*)\s+invariants compiled",
    ))
    .unwrap();
    let compilation_complete_actions = Regex::new(concat!(
        r"(?i)\[trust[_-]cg\]\s+compilation complete:\s+",
        r"([0-9][0-9,]*)/([0-9][0-9,]*)\s+actions compiled",
    ))
    .unwrap();
    let level_start = Regex::new(concat!(
        r"(?i)\[compiled-bfs\]\s+starting compiled BFS level loop\s+",
        r"\(([0-9][0-9,]*)\s+initial states in arena,\s+fused=(true|false)\)",
    ))
    .unwrap();
    let level_built = Regex::new(concat!(
        r"(?i)\[trust[_-]cg\]\s+CompiledBfsLevel built \((?P<label>[^)]+)\):",
        r".*?\binvariants(?:,\s+[0-9][0-9,]*\s+state constraints)?",
        r"(?:,\s+state_len=(?P<state_len>[0-9][0-9,]*))?\b",
    ))
    .unwrap();
    let compiled_bfs_level_line =
        Regex::new(r"(?i)\[compiled-bfs\]\s+(?:fused\s+)?level\s+\d+:").unwrap();

    for raw_line in combined.lines() {
        let line = normalize_line(raw_line);
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();

        record_native_action_callout_batch_setup_telemetry(&mut telemetry, &line);

        for captures in numeric_kv.captures_iter(&line) {
            let Some(value) = parse_count(&captures[2]) else {
                continue;
            };
            match captures[1].to_ascii_lowercase().as_str() {
                "trust_cg_actions_compiled" => telemetry.trust_cg_actions_compiled = Some(value),
                "trust_cg_actions_total" => telemetry.trust_cg_actions_total = Some(value),
                "trust_cg_invariants_compiled" => {
                    telemetry.trust_cg_invariants_compiled = Some(value)
                }
                "trust_cg_invariants_total" => telemetry.trust_cg_invariants_total = Some(value),
                "trust_cg_state_constraints_compiled" => {
                    telemetry.trust_cg_state_constraints_compiled = Some(value)
                }
                "trust_cg_state_constraints_total" => {
                    telemetry.trust_cg_state_constraints_total = Some(value)
                }
                "trust_cg_native_fused_invariant_count" => {
                    telemetry.trust_cg_native_fused_invariant_count = Some(value)
                }
                "trust_cg_native_fused_state_constraint_count" => {
                    telemetry.trust_cg_native_fused_state_constraint_count = Some(value)
                }
                "trust_cg_native_fused_state_len" => {
                    telemetry.trust_cg_native_fused_state_len = Some(value)
                }
                "compiled_bfs_level_loop_initial_states" => {
                    telemetry.compiled_bfs_level_loop_initial_states = Some(value)
                }
                "compiled_bfs_levels_completed" => {
                    telemetry.compiled_bfs_levels_completed = Some(value)
                }
                "compiled_bfs_parents_processed" => {
                    telemetry.compiled_bfs_parents_processed = Some(value)
                }
                "compiled_bfs_successors_generated" => {
                    telemetry.compiled_bfs_successors_generated = Some(value)
                }
                "compiled_bfs_successors_new" => {
                    telemetry.compiled_bfs_successors_new = Some(value)
                }
                "compiled_bfs_total_states" => telemetry.compiled_bfs_total_states = Some(value),
                "flat_bfs_frontier_fallbacks" => {
                    telemetry.flat_bfs_frontier_fallbacks = Some(value)
                }
                _ => {}
            }
        }

        for captures in bool_kv.captures_iter(&line) {
            let parsed_value = captures[2].eq_ignore_ascii_case("true");
            match captures[1].to_ascii_lowercase().as_str() {
                "compiled_bfs_step_active" => telemetry.compiled_bfs_step_active = parsed_value,
                "compiled_bfs_level_active" => telemetry.compiled_bfs_level_active = parsed_value,
                "compiled_bfs_level_loop_started" => {
                    telemetry.compiled_bfs_level_loop_started = parsed_value
                }
                "compiled_bfs_level_loop_fused" => {
                    telemetry.compiled_bfs_level_loop_fused = Some(parsed_value)
                }
                "compiled_bfs_flat_frontier_admitted" => sticky_false(
                    &mut telemetry.compiled_bfs_flat_frontier_admitted,
                    parsed_value,
                ),
                "trust_cg_bfs_level_active" => {
                    telemetry.trust_cg_bfs_level_active = parsed_value;
                    if parsed_value {
                        telemetry.compiled_bfs_level_active = true;
                    }
                }
                "trust_cg_native_fused_level_built" => {
                    telemetry.trust_cg_native_fused_level_built = parsed_value
                }
                "trust_cg_native_fused_level_active" => {
                    if parsed_value && explicit_native_fused_level_active == Some(false) {
                        continue;
                    }
                    explicit_native_fused_level_active = Some(parsed_value);
                    telemetry.trust_cg_native_fused_level_active = parsed_value;
                    if parsed_value {
                        telemetry.trust_cg_bfs_level_active = true;
                        telemetry.compiled_bfs_level_active = true;
                    }
                }
                "trust_cg_native_fused_regular_invariants_checked" => {
                    telemetry.trust_cg_native_fused_regular_invariants_checked = Some(parsed_value)
                }
                "trust_cg_native_fused_local_dedup" => {
                    telemetry.trust_cg_native_fused_local_dedup = Some(parsed_value)
                }
                "trust_cg_native_fused_flat_frontier_admission_active" => sticky_false(
                    &mut telemetry.trust_cg_native_fused_flat_frontier_admission_active,
                    parsed_value,
                ),
                "flat_state_primary" => {
                    sticky_false(&mut telemetry.flat_state_primary, parsed_value)
                }
                "flat_bfs_frontier_active" => {
                    sticky_false(&mut telemetry.flat_bfs_frontier_active, parsed_value)
                }
                _ => {}
            }
        }

        if let Some(loop_kind) = telemetry_text_value(&line, "trust_cg_bfs_level_loop_kind") {
            let loop_kind = normalize_loop_kind(&loop_kind);
            telemetry.trust_cg_bfs_level_active = true;
            telemetry.compiled_bfs_level_active = true;
            telemetry.trust_cg_bfs_level_loop_kind = Some(loop_kind.clone());
            if loop_kind == "native_fused_trust_cg_parent_loop" {
                // Derive level_built from the stable telemetry token, not only
                // the human-readable "CompiledBfsLevel built (...)" prose. Both
                // are emitted together post-build (run_helpers ~6123-6136), but
                // the prose label once drifted (trust-codegen vs Trust-CG) and
                // silently failed the gate's ends_with check.
                telemetry.trust_cg_native_fused_level_built = true;
            }
            if explicit_native_fused_level_active != Some(false) {
                telemetry.trust_cg_native_fused_level_active =
                    loop_kind == "native_fused_trust_cg_parent_loop";
            }
        }
        if let Some(mode) = telemetry_text_value(&line, "trust_cg_native_fused_mode") {
            telemetry.trust_cg_native_fused_mode = Some(normalize_loop_kind(&mode));
        }

        if let Some(captures) = compiling.captures(&line) {
            telemetry.trust_cg_actions_total = parse_count(&captures[1]);
            telemetry.trust_cg_invariants_total = parse_count(&captures[2]);
        }
        if let Some(captures) = compilation_complete_constraints.captures(&line) {
            telemetry.trust_cg_actions_compiled = parse_count(&captures[1]);
            telemetry.trust_cg_actions_total = parse_count(&captures[2]);
            telemetry.trust_cg_invariants_compiled = parse_count(&captures[3]);
            telemetry.trust_cg_invariants_total = parse_count(&captures[4]);
            telemetry.trust_cg_state_constraints_compiled = parse_count(&captures[5]);
            telemetry.trust_cg_state_constraints_total = parse_count(&captures[6]);
        } else if let Some(captures) = compilation_complete_invariants.captures(&line) {
            telemetry.trust_cg_actions_compiled = parse_count(&captures[1]);
            telemetry.trust_cg_actions_total = parse_count(&captures[2]);
            telemetry.trust_cg_invariants_compiled = parse_count(&captures[3]);
            telemetry.trust_cg_invariants_total = parse_count(&captures[4]);
        } else if let Some(captures) = compilation_complete_actions.captures(&line) {
            telemetry.trust_cg_actions_compiled = parse_count(&captures[1]);
            telemetry.trust_cg_actions_total = parse_count(&captures[2]);
        }

        if lower.contains("[trust-cg] no safe action bytecodes available") {
            saw_no_safe_actions = true;
        }
        if lower.contains("[trust-cg] compiled next-state for action")
            || lower.contains("[trust-cg] specialized '")
        {
            action_compiled_lines += 1;
        }
        if (lower.contains("[trust-cg] skipping action")
            || lower.contains("[trust-cg] failed to compile action")
            || lower.contains("[trust-cg] failed to compile specialization"))
            && !is_benign_trust_cg_action_diagnostic(&lower)
        {
            action_failed_lines += 1;
        }
        if lower.contains("[trust-cg] compiled invariant") {
            invariant_compiled_lines += 1;
        }
        if lower.contains("[trust-cg] failed to compile invariant") {
            invariant_failed_lines += 1;
        }
        if lower.contains("[trust-cg] compiled state constraint") {
            state_constraint_compiled_lines += 1;
        }
        if lower.contains("[trust-cg] failed to compile state constraint")
            || lower.contains("[trust-cg] missing bytecode for state constraint")
        {
            state_constraint_failed_lines += 1;
        }

        if lower.contains("[compiled-bfs]")
            && (lower.contains("activating compiled bfs loop")
                || lower.contains("starting compiled bfs level loop")
                || compiled_bfs_level_line.is_match(&line)
                || lower.contains("completed:"))
        {
            telemetry.compiled_bfs_step_active = true;
        }

        if let Some(captures) = level_start.captures(&line) {
            telemetry.compiled_bfs_step_active = true;
            telemetry.compiled_bfs_level_loop_started = true;
            telemetry.compiled_bfs_level_loop_initial_states = parse_count(&captures[1]);
            let fused = captures[2].eq_ignore_ascii_case("true");
            telemetry.compiled_bfs_level_loop_fused = Some(fused);
            if fused {
                telemetry.compiled_bfs_level_active = true;
            }
        }

        let completion = parse_compiled_bfs_completion(&line);
        if let Some(completion) = completion {
            record_compiled_bfs_completion(&mut telemetry, completion);
        }
        if completion.is_some()
            || (lower.contains("[compiled-bfs]") && lower.contains("compiled_bfs_execution_"))
        {
            record_compiled_bfs_execution_timing(&mut telemetry, &line);
        }

        if lower.contains("[compiled-bfs]")
            && (lower.contains("fused=true") || compiled_bfs_level_line.is_match(&line))
        {
            telemetry.compiled_bfs_level_active = true;
        }

        if let Some(captures) = level_built.captures(&line) {
            telemetry.trust_cg_bfs_level_active = true;
            telemetry.compiled_bfs_level_active = true;
            if let Some(state_len) = captures.name("state_len") {
                telemetry.trust_cg_native_fused_state_len = parse_count(state_len.as_str());
            }
            let loop_kind = normalize_loop_kind(&captures["label"]);
            if loop_kind.ends_with("native_fused_trust_cg_parent_loop") {
                telemetry.trust_cg_bfs_level_loop_kind =
                    Some("native_fused_trust_cg_parent_loop".to_string());
                telemetry.trust_cg_native_fused_level_built = true;
                if explicit_native_fused_level_active != Some(false) {
                    telemetry.trust_cg_native_fused_level_active = true;
                }
                if loop_kind.starts_with("state_constrained") {
                    telemetry.trust_cg_native_fused_mode =
                        Some("state_constraint_checking".to_string());
                } else if loop_kind.starts_with("action_only") {
                    telemetry.trust_cg_native_fused_mode = Some("action_only".to_string());
                } else if loop_kind.starts_with("invariant_checking") {
                    telemetry.trust_cg_native_fused_mode = Some("invariant_checking".to_string());
                }
            } else {
                telemetry.trust_cg_bfs_level_loop_kind = Some(loop_kind);
                telemetry.trust_cg_native_fused_level_active = false;
            }
        }

        if lower.contains("[trust-cg-native-bfs]") {
            record_native_bfs_trace_telemetry(&mut telemetry, &line);
        }

        if lower.contains("flat_state_primary=true") {
            sticky_false(&mut telemetry.flat_state_primary, true);
        } else if lower.contains("flat_state_primary=false") {
            sticky_false(&mut telemetry.flat_state_primary, false);
        }
        if lower.contains("[flat-frontier]") {
            if let Some(active) = parse_flat_frontier_active(&line) {
                sticky_false(&mut telemetry.flat_bfs_frontier_active, active);
            }
            if let Some(fallbacks) = parse_flat_frontier_fallbacks(&line) {
                telemetry.flat_bfs_frontier_fallbacks =
                    Some(max_known(telemetry.flat_bfs_frontier_fallbacks, fallbacks));
            }
        }

        if is_fallback_reason(&line) {
            append_reason(&mut telemetry.fallback_reasons, &line);
        }
    }

    if telemetry.trust_cg_actions_compiled.is_none() {
        if action_compiled_lines != 0 || action_failed_lines != 0 {
            telemetry.trust_cg_actions_compiled = Some(action_compiled_lines);
            telemetry.trust_cg_actions_total = Some(action_compiled_lines + action_failed_lines);
        } else if saw_no_safe_actions {
            telemetry.trust_cg_actions_compiled = Some(0);
            telemetry.trust_cg_actions_total = Some(0);
        }
    }
    if telemetry.trust_cg_invariants_compiled.is_none() {
        if invariant_compiled_lines != 0 || invariant_failed_lines != 0 {
            telemetry.trust_cg_invariants_compiled = Some(invariant_compiled_lines);
            telemetry.trust_cg_invariants_total =
                Some(invariant_compiled_lines + invariant_failed_lines);
        } else if telemetry.trust_cg_invariants_total == Some(0) {
            telemetry.trust_cg_invariants_compiled = Some(0);
        }
    }
    if telemetry.trust_cg_invariants_total.is_none()
        && telemetry.trust_cg_invariants_compiled == Some(0)
    {
        telemetry.trust_cg_invariants_total = Some(0);
    }
    if telemetry.trust_cg_state_constraints_compiled.is_none() {
        if state_constraint_compiled_lines != 0 || state_constraint_failed_lines != 0 {
            telemetry.trust_cg_state_constraints_compiled = Some(state_constraint_compiled_lines);
            telemetry.trust_cg_state_constraints_total =
                Some(state_constraint_compiled_lines + state_constraint_failed_lines);
        } else if telemetry.trust_cg_state_constraints_total == Some(0) {
            telemetry.trust_cg_state_constraints_compiled = Some(0);
        }
    }
    if telemetry.trust_cg_state_constraints_total.is_none()
        && telemetry.trust_cg_state_constraints_compiled == Some(0)
    {
        telemetry.trust_cg_state_constraints_total = Some(0);
    }
    if telemetry.compiled_bfs_execution_nanos.is_some()
        && !telemetry
            .compiled_bfs_execution_seconds
            .is_some_and(|value| value.is_finite() && value > 0.0)
    {
        telemetry.compiled_bfs_execution_seconds = telemetry
            .compiled_bfs_execution_nanos
            .map(|value| value as f64 / 1_000_000_000.0);
    }

    telemetry
}

pub(super) fn latest_flat_primary_backend_segment(combined: &str) -> String {
    let lines = combined.lines().collect::<Vec<_>>();
    let Some(marker_line_index) = lines
        .iter()
        .rposition(|line| line.contains(FLAT_PRIMARY_REBUILD_MARKER))
    else {
        return combined.to_string();
    };

    let flat_primary = Regex::new(r"(?i)\bflat_state_primary\s*[:=]\s*true\b").unwrap();
    for idx in (0..marker_line_index).rev() {
        if flat_primary.is_match(lines[idx]) {
            let mut latest = Vec::with_capacity(lines.len() - marker_line_index + 1);
            latest.push(lines[idx]);
            latest.extend_from_slice(&lines[marker_line_index..]);
            return latest.join("\n");
        }
    }
    lines[marker_line_index..].join("\n")
}

fn parse_compiled_bfs_completion(line: &str) -> Option<(u64, u64, u64, u64, u64)> {
    let completed = Regex::new(concat!(
        r"(?i)\[compiled-bfs\]\s+completed:\s+",
        r"([0-9][0-9,]*)\s+levels,\s+",
        r"([0-9][0-9,]*)\s+parents,\s+",
        r"([0-9][0-9,]*)\s+generated,\s+",
        r"([0-9][0-9,]*)\s+new,\s+",
        r"([0-9][0-9,]*)\s+total states\b",
    ))
    .unwrap();
    let captures = completed.captures(line)?;
    Some((
        parse_count(&captures[1])?,
        parse_count(&captures[2])?,
        parse_count(&captures[3])?,
        parse_count(&captures[4])?,
        parse_count(&captures[5])?,
    ))
}

fn record_compiled_bfs_completion(
    telemetry: &mut TrustCgTelemetry,
    completion: (u64, u64, u64, u64, u64),
) {
    let (levels, parents, generated, new, total_states) = completion;
    telemetry.compiled_bfs_levels_completed =
        Some(min_known(telemetry.compiled_bfs_levels_completed, levels));
    telemetry.compiled_bfs_parents_processed =
        Some(min_known(telemetry.compiled_bfs_parents_processed, parents));
    telemetry.compiled_bfs_successors_generated = Some(min_known(
        telemetry.compiled_bfs_successors_generated,
        generated,
    ));
    telemetry.compiled_bfs_successors_new =
        Some(min_known(telemetry.compiled_bfs_successors_new, new));
    telemetry.compiled_bfs_total_states =
        Some(max_known(telemetry.compiled_bfs_total_states, total_states));
    if levels == 0 && parents == 0 && generated == 0 && new == 0 {
        telemetry.compiled_bfs_zero_work = true;
    }
}

fn record_compiled_bfs_execution_timing(telemetry: &mut TrustCgTelemetry, line: &str) {
    let nanos = Regex::new(concat!(
        r"(?i)\b(?:compiled_bfs_execution_nanos|execution_time_ns|execution_time_nanos)",
        r"\s*[:=]\s*([0-9][0-9,]*)\b",
    ))
    .unwrap();
    for captures in nanos.captures_iter(line) {
        telemetry.compiled_bfs_execution_nanos = parse_count(&captures[1]);
    }
    let seconds = Regex::new(concat!(
        r"(?i)\b(?:compiled_bfs_execution_seconds|execution_time_seconds)",
        r"\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\b",
    ))
    .unwrap();
    for captures in seconds.captures_iter(line) {
        telemetry.compiled_bfs_execution_seconds = captures[1].parse::<f64>().ok();
    }
}

fn record_native_bfs_trace_telemetry(telemetry: &mut TrustCgTelemetry, line: &str) {
    if let Some(value) = bool_value(line, "local_dedup") {
        telemetry.trust_cg_native_fused_local_dedup = Some(value);
    }
    if let Some(value) = numeric_value(line, "generated") {
        telemetry.trust_cg_native_bfs_trace_generated = Some(max_known(
            telemetry.trust_cg_native_bfs_trace_generated,
            value,
        ));
    }
    if let Some(value) = numeric_value(line, "state_count") {
        telemetry.trust_cg_native_bfs_trace_state_count = Some(max_known(
            telemetry.trust_cg_native_bfs_trace_state_count,
            value,
        ));
    }
    if let Some(value) = numeric_value(line, "parents_processed") {
        telemetry.trust_cg_native_bfs_trace_parents_processed = Some(max_known(
            telemetry.trust_cg_native_bfs_trace_parents_processed,
            value,
        ));
    }
}

fn record_native_action_callout_batch_setup_telemetry(
    telemetry: &mut TrustCgTelemetry,
    line: &str,
) {
    let lower = line.to_ascii_lowercase();
    if !lower.contains("ty_native_action_callout_batch_setup")
        && !lower.contains("native_action_callout_batch_summary")
        && !lower.contains("native_action_callout_batch_")
    {
        return;
    }

    if let Some(value) =
        telemetry_non_none_text_value(line, "artifact_identity_source").or_else(|| {
            telemetry_non_none_text_value(
                line,
                "native_action_callout_batch_artifact_identity_source",
            )
        })
    {
        telemetry.native_action_callout_batch_artifact_identity_source = Some(value);
    }
    if let Some(value) = telemetry_non_none_text_value(line, "artifact_identity").or_else(|| {
        telemetry_non_none_text_value(line, "native_action_callout_batch_artifact_identity")
    }) {
        telemetry.native_action_callout_batch_artifact_identity = Some(value);
    }
    if let Some(value) =
        telemetry_non_none_text_value(line, "artifact_cache_digest").or_else(|| {
            telemetry_non_none_text_value(line, "native_action_callout_batch_artifact_cache_digest")
        })
    {
        telemetry.native_action_callout_batch_cache_key =
            Some(native_action_callout_batch_cache_key(&value));
        telemetry.native_action_callout_batch_artifact_cache_digest = Some(value);
    }
    if let Some(value) = bool_value(line, "artifact_cacheable")
        .or_else(|| bool_value(line, "native_action_callout_batch_artifact_cacheable"))
    {
        telemetry.native_action_callout_batch_artifact_cacheable = Some(value);
    }
    if let Some(value) = bool_value(line, "artifact_cache_disabled_by_env").or_else(|| {
        bool_value(
            line,
            "native_action_callout_batch_artifact_cache_disabled_by_env",
        )
    }) {
        telemetry.native_action_callout_batch_artifact_cache_disabled_by_env = Some(value);
    }
    if let Some(value) = numeric_value(line, "shard_count")
        .or_else(|| numeric_value(line, "native_action_callout_batch_shard_count"))
    {
        telemetry.native_action_callout_batch_shard_count = Some(value);
    }
    if let Some(value) = bool_value(line, "warm_cache_enabled")
        .or_else(|| bool_value(line, "native_action_callout_batch_warm_cache_enabled"))
    {
        telemetry.native_action_callout_batch_warm_cache_enabled = Some(value);
    }
    if let Some(value) = bool_value(line, "warm_cache_lookup_attempted").or_else(|| {
        bool_value(
            line,
            "native_action_callout_batch_warm_cache_lookup_attempted",
        )
    }) {
        telemetry.native_action_callout_batch_warm_cache_lookup_attempted = Some(value);
    }
    if let Some(value) = numeric_value(line, "warm_cache_hits")
        .or_else(|| numeric_value(line, "native_action_callout_batch_warm_cache_hits"))
    {
        telemetry.native_action_callout_batch_warm_cache_hits = Some(value);
    }
    if let Some(value) = numeric_value(line, "warm_cache_misses")
        .or_else(|| numeric_value(line, "native_action_callout_batch_warm_cache_misses"))
    {
        telemetry.native_action_callout_batch_warm_cache_misses = Some(value);
    }
    if let Some(value) = numeric_value(line, "warm_cache_stores")
        .or_else(|| numeric_value(line, "native_action_callout_batch_warm_cache_stores"))
    {
        telemetry.native_action_callout_batch_warm_cache_stores = Some(value);
    }
    if let Some(value) =
        max_numeric_value(line, &["setup_ms", "native_action_callout_batch_setup_ms"])
    {
        telemetry.native_action_callout_batch_setup_ms = Some(max_known(
            telemetry.native_action_callout_batch_setup_ms,
            value,
        ));
    }
    if let Some(value) = max_numeric_value(
        line,
        &["lowering_ms", "native_action_callout_batch_lowering_ms"],
    ) {
        telemetry.native_action_callout_batch_lowering_ms = Some(max_known(
            telemetry.native_action_callout_batch_lowering_ms,
            value,
        ));
    }
    if let Some(value) = max_numeric_value(
        line,
        &["assembly_ms", "native_action_callout_batch_assembly_ms"],
    ) {
        telemetry.native_action_callout_batch_assembly_ms = Some(max_known(
            telemetry.native_action_callout_batch_assembly_ms,
            value,
        ));
    }
    if let Some(value) = max_numeric_value(
        line,
        &["compile_ms", "native_action_callout_batch_compile_ms"],
    ) {
        telemetry.native_action_callout_batch_compile_ms = Some(max_known(
            telemetry.native_action_callout_batch_compile_ms,
            value,
        ));
    }
    if let Some(value) = max_numeric_value(
        line,
        &[
            "warm_cache_lookup_ms",
            "native_action_callout_batch_warm_cache_lookup_ms",
        ],
    ) {
        telemetry.native_action_callout_batch_warm_cache_lookup_ms = Some(max_known(
            telemetry.native_action_callout_batch_warm_cache_lookup_ms,
            value,
        ));
    }
    if let Some(value) = max_numeric_value(
        line,
        &[
            "artifact_materialization_ms",
            "native_action_callout_batch_artifact_materialization_ms",
        ],
    ) {
        telemetry.native_action_callout_batch_artifact_materialization_ms = Some(max_known(
            telemetry.native_action_callout_batch_artifact_materialization_ms,
            value,
        ));
    }
    if let Some(value) = max_numeric_value(
        line,
        &[
            "fallback_per_action_compile_ms",
            "native_action_callout_batch_fallback_per_action_compile_ms",
        ],
    ) {
        telemetry.native_action_callout_batch_fallback_per_action_compile_ms = Some(max_known(
            telemetry.native_action_callout_batch_fallback_per_action_compile_ms,
            value,
        ));
    }
    if let Some(value) =
        telemetry_non_none_text_value(line, "shard_warm_cache_statuses").or_else(|| {
            telemetry_non_none_text_value(
                line,
                "native_action_callout_batch_shard_warm_cache_statuses",
            )
        })
    {
        let statuses = csv_values(&value);
        if !statuses.is_empty() {
            telemetry.native_action_callout_batch_shard_warm_cache_statuses = statuses;
        }
    }
}

fn native_action_callout_batch_cache_key(cache_digest: &str) -> String {
    format!("trust_cg_batch_jit_cache:{cache_digest}")
}

fn parse_flat_frontier_active(line: &str) -> Option<bool> {
    let lower = line.to_ascii_lowercase();
    if lower.contains("flat_bfs_frontier_active=false") {
        return Some(false);
    }
    if !lower.contains("flat_bfs_frontier_active=true") {
        return None;
    }
    let fallbacks = parse_flat_frontier_fallbacks(line)?;
    Some(fallbacks == 0)
}

fn parse_flat_frontier_fallbacks(line: &str) -> Option<u64> {
    let fallback = Regex::new(r"(?i)\b([0-9][0-9,]*)\s+fallback\b").unwrap();
    fallback
        .captures(line)
        .and_then(|captures| parse_count(&captures[1]))
}

fn is_fallback_reason(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if lower.contains("attempting compile via trust_ir anyway") {
        return false;
    }
    if is_benign_trust_cg_action_diagnostic(&lower) {
        return false;
    }
    if is_benign_trust_cg_native_action_batch_miss(&lower) {
        return false;
    }
    if is_benign_trust_cg_compiled_bfs_step_skip(&lower) {
        return false;
    }
    if lower.contains("[trust-cg] compiledbfsstep not eligible")
        && lower.contains("state constraints require native fused constraint pruning")
    {
        return false;
    }
    let fallback_scan = Regex::new(r"\(\s*0\s+failed\s*\)")
        .unwrap()
        .replace_all(&lower, "")
        .into_owned();
    let trust_cg_line = lower.contains("[trust-cg]") || lower.contains("[trust-cg][trust-ir-dump]");
    let trust_cg_error = fallback_scan.contains("trust-cg")
        && [
            "failed",
            "fallback",
            "falling back",
            "skip",
            "skipping",
            "skipped",
            "unavailable",
            "not eligible",
            "missing bytecode",
            "missing native code",
            "unsupported",
            "not yet supported",
            "lowering",
            "outside the scalar",
            "requires",
        ]
        .iter()
        .any(|token| fallback_scan.contains(token));
    let trust_cg_selftest_runtime_issue = lower.contains("[trust_cg-selftest]")
        && [
            "failed",
            "runtime_error",
            "runtime error",
            "failing closed",
            "missing function pointer",
            "requested interpreter fallback",
            "fallback",
        ]
        .iter()
        .any(|token| lower.contains(token));
    let flat_fallback =
        lower.contains("[flat_state]") && (lower.contains("fail") || lower.contains("fallback"));
    let flat_frontier_fallback = lower.contains("[flat-frontier]")
        && Regex::new(r"\b[1-9][0-9]*\s+fallback\b")
            .unwrap()
            .is_match(&lower);

    (trust_cg_line && trust_cg_error)
        || trust_cg_selftest_runtime_issue
        || is_compiled_bfs_runtime_issue(line)
        || flat_fallback
        || flat_frontier_fallback
}

fn is_benign_trust_cg_action_diagnostic(lower: &str) -> bool {
    is_benign_trust_cg_wrapper_skip(lower)
        || is_benign_trust_cg_shadowed_raw_action_skip(lower)
        || is_benign_trust_cg_specialization_bytecode_miss(lower)
        || is_benign_trust_cg_alias_raw_recovery(lower)
        || is_benign_trust_cg_native_action_callout_summary(lower)
}

fn is_benign_trust_cg_wrapper_skip(lower: &str) -> bool {
    lower.contains("[trust-cg] skipping action")
        && lower.contains("arity-positive wrapper")
        && lower.contains("executable bindingspec specializations are counted separately")
}

fn is_benign_trust_cg_shadowed_raw_action_skip(lower: &str) -> bool {
    (lower.contains("[trust-cg] skipping action")
        && lower.contains("shadowed raw split action")
        && lower.contains("executable bindingspec alias")
        && lower.contains("is counted separately"))
        || (lower.contains("[trust-cg] skipping")
            && lower.contains("shadowed raw action callout")
            && lower.contains("executable bindingspec aliases will supply native dispatch"))
}

// Split-action specialization uses a two-phase native-compile protocol: first
// try the BindingSpec alias (keyed on the original base name), then on failure
// compile the raw split form directly. The alias attempt fails benignly when
// the base name is absent from the bytecode map (only the split forms `X__N`
// are keyed, not the original `X`); the raw-form recovery below keeps the
// action fully native. Neither phase message is an interpreter fallback — a
// genuinely uncovered action is caught by the compile-complete count and the
// gate's generated-state parity check, not by these planner path-selection logs.
fn is_benign_trust_cg_specialization_bytecode_miss(lower: &str) -> bool {
    lower.contains("[trust-cg] specialization")
        && lower.contains("not in bytecode map")
        && lower.contains("skipping")
}

fn is_benign_trust_cg_alias_raw_recovery(lower: &str) -> bool {
    lower.contains("[trust-cg] alias")
        && lower.contains("failed to plan")
        && lower.contains("compiling shadowed raw split action")
        && lower.contains("directly as fallback")
}

fn is_benign_trust_cg_native_action_callout_summary(lower: &str) -> bool {
    lower.contains("[trust-cg] native action callouts:")
        && lower.contains("planned=")
        && lower.contains("compiled=")
        && lower.contains("skipped_shadowed=")
}

fn is_benign_trust_cg_native_action_batch_miss(lower: &str) -> bool {
    lower.contains("[trust-cg] native action callout batch unavailable:")
        && lower.contains("using per-action compilation")
}

fn is_benign_trust_cg_compiled_bfs_step_skip(lower: &str) -> bool {
    lower.contains("[trust-cg] compiledbfsstep skipped:")
        && lower.contains("native fused level is the only admissible compiled bfs path")
}

fn is_compiled_bfs_runtime_issue(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if !lower.contains("[compiled-bfs]") {
        return false;
    }
    if lower.contains("state constraints require") {
        return true;
    }
    [
        "interpreter path used",
        "not enabled",
        "compiled bfs disabled",
        "fused level build failed",
        "fused level error",
        "became unavailable",
        "fallback",
        "falling back",
        "step error",
        "disabling",
        "disabled",
    ]
    .iter()
    .any(|token| lower.contains(token))
}

fn append_reason(reasons: &mut Vec<String>, line: &str) {
    let line = if line.len() > MAX_REASON_CHARS {
        format!("{}...", &line[..MAX_REASON_CHARS - 3])
    } else {
        line.to_string()
    };
    if reasons.contains(&line) {
        return;
    }
    if reasons.len() < MAX_FALLBACK_REASONS {
        reasons.push(line);
    } else if !reasons
        .iter()
        .any(|reason| reason == TRUNCATED_FALLBACK_REASON)
    {
        if let Some(last) = reasons.last_mut() {
            *last = TRUNCATED_FALLBACK_REASON.to_string();
        }
    }
}

fn telemetry_text_value(line: &str, key: &str) -> Option<String> {
    let pattern = Regex::new(&format!(r"(?i)\b{}\s*[:=]\s*", regex::escape(key))).unwrap();
    let matched = pattern.find(line)?;
    let mut value_parts = Vec::new();
    for token in line[matched.end()..].split_whitespace() {
        if is_assignment_token(token) {
            break;
        }
        value_parts.push(token.trim_end_matches(','));
    }
    if value_parts.is_empty() {
        None
    } else {
        Some(value_parts.join(" "))
    }
}

fn telemetry_non_none_text_value(line: &str, key: &str) -> Option<String> {
    telemetry_text_value(line, key).and_then(|value| {
        let value = value.trim();
        (!value.is_empty() && !value.eq_ignore_ascii_case("none")).then(|| value.to_string())
    })
}

fn csv_values(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty() && !item.eq_ignore_ascii_case("none"))
        .map(ToString::to_string)
        .collect()
}

fn is_assignment_token(token: &str) -> bool {
    let key = if let Some((key, _)) = token.split_once('=') {
        key
    } else if let Some(key) = token.strip_suffix(':') {
        key
    } else {
        return false;
    };
    let mut chars = key.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn numeric_value(line: &str, key: &str) -> Option<u64> {
    let pattern = Regex::new(&format!(
        r"(?i)\b{}\s*[:=]\s*([0-9][0-9,]*)\b",
        regex::escape(key)
    ))
    .unwrap();
    pattern
        .captures(line)
        .and_then(|captures| parse_count(&captures[1]))
}

fn max_numeric_value(line: &str, keys: &[&str]) -> Option<u64> {
    keys.iter().filter_map(|key| numeric_value(line, key)).max()
}

fn bool_value(line: &str, key: &str) -> Option<bool> {
    let pattern = Regex::new(&format!(
        r"(?i)\b{}\s*[:=]\s*(true|false)\b",
        regex::escape(key)
    ))
    .unwrap();
    pattern
        .captures(line)
        .map(|captures| captures[1].eq_ignore_ascii_case("true"))
}

fn normalize_loop_kind(value: &str) -> String {
    let mut output = String::new();
    let mut last_was_sep = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            output.push(ch);
            last_was_sep = false;
        } else if !last_was_sep && !output.is_empty() {
            output.push('_');
            last_was_sep = true;
        }
    }
    output.trim_matches('_').to_string()
}

fn normalize_line(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parse_count(value: &str) -> Option<u64> {
    value.replace(',', "").parse::<u64>().ok()
}

fn sticky_false(slot: &mut Option<bool>, value: bool) {
    if !value {
        *slot = Some(false);
    } else if *slot != Some(false) {
        *slot = Some(true);
    }
}

fn min_known(current: Option<u64>, value: u64) -> u64 {
    current.map_or(value, |current| current.min(value))
}

fn max_known(current: Option<u64>, value: u64) -> u64 {
    current.map_or(value, |current| current.max(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tlc_final_counts_from_last_summary_line() {
        let stdout = "\
            TLC startup\n\
            Finished computing initial states: 1,001 distinct states generated at 2026-04-27 16:28:47.\n\
            10 states generated, 8 distinct states found, 0 states left on queue.\n\
            1,499,503 states generated, 501,500 distinct states found, 0 states left on queue.\n";

        let counts = parse_tlc_final_counts(stdout, "");

        assert_eq!(counts.states_generated, Some(1_499_503));
        assert_eq!(counts.states_found, Some(501_500));
        assert_eq!(counts.distinct_states, Some(501_500));
        assert_eq!(counts.states_left, Some(0));
        assert_eq!(counts.transitions, Some(1_498_502));
    }

    #[test]
    fn parses_ty_final_states_and_transitions() {
        let stdout = "\
            Model checking complete.\n\
            States found: 501,500\n\
            Transitions: 1,498,502\n";

        let counts = parse_ty_final_counts(stdout, "");

        assert_eq!(counts.states_found, Some(501_500));
        assert_eq!(counts.distinct_states, Some(501_500));
        assert_eq!(counts.transitions, Some(1_498_502));
        assert_eq!(counts.states_generated, None);
    }

    #[test]
    fn parses_ty_human_sentence_states_found() {
        let stdout = "\
            Model checking complete. No error has been found.\n\
              1,520,618 states found.\n\
              Resolved by: BFS (explicit-state)\n\
            \n\
            Time: 4.538s\n";

        let counts = parse_ty_final_counts(stdout, "");

        assert_eq!(counts.states_found, Some(1_520_618));
        assert_eq!(counts.distinct_states, Some(1_520_618));
        assert_eq!(counts.transitions, None);
    }

    #[test]
    fn parses_full_native_fused_trust_cg_telemetry() {
        let stdout = format!(
            "\
             [trust_cg] old fallback before rebuild\n\
             flat_state_primary=true\n\
             {FLAT_PRIMARY_REBUILD_MARKER}\n\
             [trust-cg] compilation complete: 27/27 actions, 3/3 invariants, 1/1 state constraints compiled\n\
             [trust-cg] native action callouts: planned=27 compiled=27 skipped_shadowed=27\n\
             [trust-cg] skipping action 'ReceiveRequest' as arity-positive wrapper 2; executable BindingSpec specializations are counted separately\n\
             [trust-cg] skipping 27 shadowed raw action callout(s); executable BindingSpec aliases will supply native dispatch\n\
             [trust-cg] skipping action 'ReceiveRequest__1_2' as shadowed raw split action; executable BindingSpec alias 'ReceiveRequest__1_2_1_2' is counted separately\n\
             [trust-cg] CompiledBfsStep not eligible: state constraints require native fused constraint pruning (first state constraint: ClockConstraint)\n\
             [trust-cg] CompiledBfsLevel built (state-constrained native fused Trust-CG parent loop): 27 actions, 3 invariants, 1 state constraints, state_len=89\n\
             trust_cg_native_fused_regular_invariants_checked=true trust_cg_native_fused_invariant_count=3 trust_cg_native_fused_state_constraint_count=1\n\
             [trust_cg] trust_cg_native_fused_flat_frontier_admission_active=true compiled_bfs_flat_frontier_admitted=true\n\
             [compiled-bfs] starting compiled BFS level loop (4 initial states in arena, fused=true)\n\
             [trust-cg-native-bfs] generated=2,496,350 state_count=724,274 parents_processed=724,274 local_dedup=false\n\
             [compiled-bfs] completed: 10 levels, 724,274 parents, 2,496,350 generated, 724,270 new, 724,274 total states, compiled_bfs_execution_nanos=123,456,789 compiled_bfs_execution_seconds=0.123456789\n\
             [flat-frontier] flat_bfs_frontier_active=true (0 fallback)\n"
        );

        let telemetry = parse_trust_cg_telemetry(&stdout, "");

        assert_eq!(telemetry.trust_cg_actions_compiled, Some(27));
        assert_eq!(telemetry.trust_cg_actions_total, Some(27));
        assert_eq!(telemetry.trust_cg_invariants_compiled, Some(3));
        assert_eq!(telemetry.trust_cg_invariants_total, Some(3));
        assert_eq!(telemetry.trust_cg_state_constraints_compiled, Some(1));
        assert_eq!(telemetry.trust_cg_state_constraints_total, Some(1));
        assert!(telemetry.trust_cg_native_fused_level_built);
        assert!(telemetry.trust_cg_native_fused_level_active);
        assert_eq!(
            telemetry.trust_cg_bfs_level_loop_kind.as_deref(),
            Some("native_fused_trust_cg_parent_loop")
        );
        assert_eq!(
            telemetry.trust_cg_native_fused_mode.as_deref(),
            Some("state_constraint_checking")
        );
        assert_eq!(telemetry.trust_cg_native_fused_state_len, Some(89));
        assert_eq!(telemetry.trust_cg_native_fused_invariant_count, Some(3));
        assert_eq!(
            telemetry.trust_cg_native_fused_state_constraint_count,
            Some(1)
        );
        assert_eq!(
            telemetry.trust_cg_native_fused_regular_invariants_checked,
            Some(true)
        );
        assert!(telemetry.compiled_bfs_level_loop_started);
        assert_eq!(telemetry.compiled_bfs_level_loop_initial_states, Some(4));
        assert_eq!(telemetry.compiled_bfs_level_loop_fused, Some(true));
        assert_eq!(telemetry.compiled_bfs_levels_completed, Some(10));
        assert_eq!(telemetry.compiled_bfs_parents_processed, Some(724_274));
        assert_eq!(telemetry.compiled_bfs_successors_generated, Some(2_496_350));
        assert_eq!(telemetry.compiled_bfs_successors_new, Some(724_270));
        assert_eq!(telemetry.compiled_bfs_total_states, Some(724_274));
        assert_eq!(telemetry.compiled_bfs_execution_nanos, Some(123_456_789));
        assert_eq!(telemetry.compiled_bfs_execution_seconds, Some(0.123456789));
        assert_eq!(telemetry.trust_cg_native_fused_local_dedup, Some(false));
        assert_eq!(
            telemetry.trust_cg_native_fused_flat_frontier_admission_active,
            Some(true)
        );
        assert_eq!(telemetry.compiled_bfs_flat_frontier_admitted, Some(true));
        assert_eq!(telemetry.flat_state_primary, Some(true));
        assert_eq!(telemetry.flat_bfs_frontier_active, Some(true));
        assert_eq!(telemetry.flat_bfs_frontier_fallbacks, Some(0));
        assert!(telemetry.fallback_reasons.is_empty());
    }

    #[test]
    fn parses_native_action_callout_batch_setup_identity() {
        let stdout = "\
            [trust_cg-evidence] trust-codegen ty_native_action_callout_batch_setup schema=ty.trust_cg.native_action_callout_batch_setup.v1 artifact_identity_source=trust_cg_compiled_batch_stats artifact_identity=trust_cg_batch_jit:shared_high_performance_engine:abc123 artifact_cache_digest=abc123 artifact_cacheable=false artifact_cache_disabled_by_env=true shard_count=2 warm_cache_enabled=true warm_cache_lookup_attempted=true warm_cache_hits=1 warm_cache_misses=1 warm_cache_stores=1 shard_warm_cache_statuses=hit,miss fallback_reason=none\n";

        let telemetry = parse_trust_cg_telemetry(stdout, "");

        assert_eq!(
            telemetry
                .native_action_callout_batch_artifact_identity_source
                .as_deref(),
            Some("trust_cg_compiled_batch_stats")
        );
        assert_eq!(
            telemetry
                .native_action_callout_batch_artifact_identity
                .as_deref(),
            Some("trust_cg_batch_jit:shared_high_performance_engine:abc123")
        );
        assert_eq!(
            telemetry
                .native_action_callout_batch_artifact_cache_digest
                .as_deref(),
            Some("abc123")
        );
        assert_eq!(
            telemetry.native_action_callout_batch_cache_key.as_deref(),
            Some("trust_cg_batch_jit_cache:abc123")
        );
        assert_eq!(
            telemetry.native_action_callout_batch_artifact_cacheable,
            Some(false)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_artifact_cache_disabled_by_env,
            Some(true)
        );
        assert_eq!(telemetry.native_action_callout_batch_shard_count, Some(2));
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_enabled,
            Some(true)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_lookup_attempted,
            Some(true)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_hits,
            Some(1)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_misses,
            Some(1)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_stores,
            Some(1)
        );
        assert_eq!(telemetry.native_action_callout_batch_setup_ms, None);
        assert_eq!(
            telemetry.native_action_callout_batch_shard_warm_cache_statuses,
            vec!["hit".to_string(), "miss".to_string()]
        );
        assert!(telemetry.fallback_reasons.is_empty());
    }

    #[test]
    fn parses_prefixed_batch_warm_cache_fields_from_summary_line() {
        let stdout = "\
            [trust_cg] coverage native_action_callout_batch_artifact_identity_source=trust_cg_compiled_batch_stats native_action_callout_batch_artifact_identity=trust_cg_batch_jit:shared_high_performance_engine:abc123 native_action_callout_batch_artifact_cache_digest=abc123 native_action_callout_batch_artifact_cacheable=true native_action_callout_batch_artifact_cache_disabled_by_env=false native_action_callout_batch_shard_count=3 native_action_callout_batch_warm_cache_enabled=true native_action_callout_batch_warm_cache_lookup_attempted=true native_action_callout_batch_warm_cache_hits=2 native_action_callout_batch_warm_cache_misses=1 native_action_callout_batch_warm_cache_stores=1 native_action_callout_batch_shard_warm_cache_statuses=hit,hit,miss\n";

        let telemetry = parse_trust_cg_telemetry(stdout, "");

        assert_eq!(
            telemetry
                .native_action_callout_batch_artifact_identity_source
                .as_deref(),
            Some("trust_cg_compiled_batch_stats")
        );
        assert_eq!(
            telemetry
                .native_action_callout_batch_artifact_identity
                .as_deref(),
            Some("trust_cg_batch_jit:shared_high_performance_engine:abc123")
        );
        assert_eq!(
            telemetry
                .native_action_callout_batch_artifact_cache_digest
                .as_deref(),
            Some("abc123")
        );
        assert_eq!(
            telemetry.native_action_callout_batch_cache_key.as_deref(),
            Some("trust_cg_batch_jit_cache:abc123")
        );
        assert_eq!(
            telemetry.native_action_callout_batch_artifact_cacheable,
            Some(true)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_artifact_cache_disabled_by_env,
            Some(false)
        );
        assert_eq!(telemetry.native_action_callout_batch_shard_count, Some(3));
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_enabled,
            Some(true)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_lookup_attempted,
            Some(true)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_hits,
            Some(2)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_misses,
            Some(1)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_stores,
            Some(1)
        );
        assert_eq!(telemetry.native_action_callout_batch_setup_ms, None);
        assert_eq!(
            telemetry.native_action_callout_batch_shard_warm_cache_statuses,
            vec!["hit".to_string(), "hit".to_string(), "miss".to_string()]
        );
    }

    #[test]
    fn parses_native_action_callout_batch_phase_timings() {
        let stdout = "\
            [trust_cg-timing] native_action_callout_batch_summary setup_ms=42 lowering_ms=3 assembly_ms=4 compile_ms=20 warm_cache_lookup_ms=2 artifact_materialization_ms=5 fallback_per_action_compile_ms=0\n\
            [trust_cg-evidence] trust-codegen ty_native_action_callout_batch_setup schema=ty.trust_cg.native_action_callout_batch_setup.v1 native_action_callout_batch_setup_ms=40 native_action_callout_batch_lowering_ms=3 native_action_callout_batch_assembly_ms=4 native_action_callout_batch_compile_ms=19 native_action_callout_batch_warm_cache_lookup_ms=2 native_action_callout_batch_artifact_materialization_ms=5 native_action_callout_batch_fallback_per_action_compile_ms=0\n";

        let telemetry = parse_trust_cg_telemetry(stdout, "");

        assert_eq!(telemetry.native_action_callout_batch_setup_ms, Some(42));
        assert_eq!(telemetry.native_action_callout_batch_lowering_ms, Some(3));
        assert_eq!(telemetry.native_action_callout_batch_assembly_ms, Some(4));
        assert_eq!(telemetry.native_action_callout_batch_compile_ms, Some(20));
        assert_eq!(
            telemetry.native_action_callout_batch_warm_cache_lookup_ms,
            Some(2)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_artifact_materialization_ms,
            Some(5)
        );
        assert_eq!(
            telemetry.native_action_callout_batch_fallback_per_action_compile_ms,
            Some(0)
        );
    }

    #[test]
    fn ignores_shadowed_alias_callout_diagnostics_as_fallback_reasons() {
        let stdout = "\
            [trust-cg] native action callouts: planned=4 compiled=4 skipped_shadowed=0\n\
            [trust-cg] skipping 11 shadowed raw action callout(s); executable BindingSpec aliases will supply native dispatch\n\
            [trust-cg] skipping action 'RecvMsg__1' as shadowed raw split action; executable BindingSpec alias 'RecvMsg__1_1' is counted separately\n";

        let telemetry = parse_trust_cg_telemetry(stdout, "");

        assert!(
            telemetry.fallback_reasons.is_empty(),
            "{:?}",
            telemetry.fallback_reasons
        );
    }

    #[test]
    fn ignores_specialization_bytecode_miss_and_raw_recovery_as_fallback_reasons() {
        // Two-phase split-action compile: the BindingSpec alias (keyed on the
        // original base name) misses the bytecode map, then the raw split form
        // is compiled directly. Both phases are native; neither is an
        // interpreter fallback. Real MCLamportMutex telemetry.
        let stdout = "\
            [trust-cg] specialization 'Request__1_1': base action 'Request' not in bytecode map, skipping\n\
            [trust-cg] specialization 'Enter__2_2': base action 'Enter' not in bytecode map, skipping\n\
            [trust-cg] specialization 'Exit__3_3': base action 'Exit' not in bytecode map, skipping\n\
            [trust-cg] alias 'Request__1_1' failed to plan; compiling shadowed raw split action 'Request__1' directly as fallback\n\
            [trust-cg] alias 'Enter__2_2' failed to plan; compiling shadowed raw split action 'Enter__2' directly as fallback\n\
            [trust-cg] compilation complete: 27/27 actions, 3/3 invariants, 1/1 state constraints compiled in 1651ms\n";

        let telemetry = parse_trust_cg_telemetry(stdout, "");

        assert!(
            telemetry.fallback_reasons.is_empty(),
            "{:?}",
            telemetry.fallback_reasons
        );
    }

    #[test]
    fn ignores_native_fused_compiled_bfs_step_skip_as_fallback_reason() {
        let stdout = "\
            [trust_cg] CompiledBfsStep skipped: native fused level is the only admissible compiled BFS path for this run\n\
            [trust_cg] CompiledBfsLevel built (state-constrained native fused Trust-CG parent loop): 15 actions, 3 invariants, state_len=15\n";

        let telemetry = parse_trust_cg_telemetry(stdout, "");

        assert!(telemetry.trust_cg_native_fused_level_built);
        assert!(
            telemetry.fallback_reasons.is_empty(),
            "{:?}",
            telemetry.fallback_reasons
        );
    }

    #[test]
    fn ignores_native_action_batch_miss_as_fallback_reason() {
        let stdout = "\
            [trust_cg] native action callout batch unavailable: trust-ir module metadata differs; using per-action compilation\n\
            [trust_cg] native action callouts: planned=4 compiled=4 skipped_shadowed=0\n";

        let telemetry = parse_trust_cg_telemetry(stdout, "");

        assert!(
            telemetry.fallback_reasons.is_empty(),
            "{:?}",
            telemetry.fallback_reasons
        );
    }

    #[test]
    fn parses_native_fused_flat_frontier_admission_false_sticky() {
        let stdout = "\
            [trust_cg] trust_cg_native_fused_flat_frontier_admission_active=true compiled_bfs_flat_frontier_admitted=true\n\
            [trust_cg] trust_cg_native_fused_flat_frontier_admission_active=false compiled_bfs_flat_frontier_admitted=false\n";

        let telemetry = parse_trust_cg_telemetry(stdout, "");

        assert_eq!(
            telemetry.trust_cg_native_fused_flat_frontier_admission_active,
            Some(false)
        );
        assert_eq!(telemetry.compiled_bfs_flat_frontier_admitted, Some(false));
    }

    #[test]
    fn preserves_post_rebuild_fallback_reasons_only() {
        let stdout = format!(
            "\
             [trust-cg] failed to compile action Old: unsupported pre-rebuild\n\
             flat_state_primary=true\n\
             {FLAT_PRIMARY_REBUILD_MARKER}\n\
             [trust-cg] failed to compile action New: unsupported opcode for trust-ir backend\n\
             [compiled-bfs] fused level error: runtime_error\n"
        );

        let telemetry = parse_trust_cg_telemetry(&stdout, "");

        assert_eq!(telemetry.trust_cg_actions_compiled, Some(0));
        assert_eq!(telemetry.trust_cg_actions_total, Some(1));
        assert_eq!(telemetry.fallback_reasons.len(), 2);
        assert!(telemetry
            .fallback_reasons
            .iter()
            .any(|reason| reason.contains("failed to compile action New")));
        assert!(telemetry
            .fallback_reasons
            .iter()
            .all(|reason| !reason.contains("Old")));
    }

    #[test]
    fn drops_stale_fallback_reasons_between_flat_primary_and_rebuild_marker() {
        let stdout = format!(
            "\
             flat_state_primary=true\n\
             [trust_cg] failed to compile action Old: unsupported pre-rebuild\n\
             {FLAT_PRIMARY_REBUILD_MARKER}\n\
             [trust_cg] compilation complete: 1/1 actions compiled\n\
             [trust_cg] CompiledBfsLevel built (invariant-checking native fused Trust-CG parent loop): 1 actions, 0 invariants, state_len=1\n\
             [flat-frontier] flat_bfs_frontier_active=true (0 fallback)\n"
        );

        let telemetry = parse_trust_cg_telemetry(&stdout, "");

        assert_eq!(telemetry.flat_state_primary, Some(true));
        assert!(telemetry.trust_cg_native_fused_level_built);
        assert!(
            telemetry.fallback_reasons.is_empty(),
            "{:?}",
            telemetry.fallback_reasons
        );
    }

    #[test]
    fn records_missing_native_code_as_current_backend_fallback() {
        let stdout = format!(
            "\
             flat_state_primary=true\n\
             {FLAT_PRIMARY_REBUILD_MARKER}\n\
             [trust-cg] missing native code for action SendMsg__self_1_other_2\n"
        );

        let telemetry = parse_trust_cg_telemetry(&stdout, "");

        assert_eq!(telemetry.flat_state_primary, Some(true));
        assert_eq!(telemetry.fallback_reasons.len(), 1);
        assert!(telemetry.fallback_reasons[0].contains("missing native code"));
    }

    #[test]
    fn explicit_native_fused_false_is_not_overridden_by_built_line() {
        let stdout = format!(
            "\
             flat_state_primary=true\n\
             {FLAT_PRIMARY_REBUILD_MARKER}\n\
             trust_cg_native_fused_level_active=false\n\
             [trust_cg] CompiledBfsLevel built (invariant-checking native fused Trust-CG parent loop): 4 actions, 1 invariants, state_len=2\n"
        );

        let telemetry = parse_trust_cg_telemetry(&stdout, "");

        assert!(telemetry.trust_cg_native_fused_level_built);
        assert!(!telemetry.trust_cg_native_fused_level_active);
        assert_eq!(
            telemetry.trust_cg_native_fused_mode.as_deref(),
            Some("invariant_checking")
        );
        assert_eq!(telemetry.trust_cg_native_fused_state_len, Some(2));
    }
}
