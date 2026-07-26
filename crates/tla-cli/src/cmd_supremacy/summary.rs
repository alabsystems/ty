// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Typed summary model for the single-thread supremacy benchmark JSON.
//!
//! This module intentionally contains only data shaping and aggregation. The
//! subprocess runner still lives outside this module.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

pub(super) const SUMMARY_SCHEMA: &str = "ty.single_thread_supremacy.summary.v1";

#[derive(Clone, Debug, Serialize)]
pub(super) struct BenchmarkSummary {
    pub(super) schema: &'static str,
    pub(super) timestamp: String,
    pub(super) git_commit: String,
    pub(super) artifact_bundle: String,
    pub(super) invocation: String,
    pub(super) build_identity: BenchmarkBuildIdentity,
    pub(super) backend_controls: BackendControls,
    pub(super) launch_controls: LaunchControls,
    pub(super) gate_flags: BenchmarkGateFlags,
    pub(super) rows: Vec<BenchmarkRow>,
}

impl BenchmarkSummary {
    pub(super) fn new(
        timestamp: impl Into<String>,
        git_commit: impl Into<String>,
        artifact_bundle: impl Into<String>,
        invocation: impl Into<String>,
        build_identity: BenchmarkBuildIdentity,
        backend_controls: BackendControls,
        launch_controls: LaunchControls,
        gate_flags: BenchmarkGateFlags,
        rows: Vec<BenchmarkRow>,
    ) -> Self {
        Self {
            schema: SUMMARY_SCHEMA,
            timestamp: timestamp.into(),
            git_commit: git_commit.into(),
            artifact_bundle: artifact_bundle.into(),
            invocation: invocation.into(),
            build_identity,
            backend_controls,
            launch_controls,
            gate_flags,
            rows,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct BenchmarkBuildIdentity {
    pub(super) cargo_profile: String,
    pub(super) ty_binary_path: String,
    pub(super) ty_binary_sha256: String,
}

impl BenchmarkBuildIdentity {
    pub(super) fn new(
        cargo_profile: impl Into<String>,
        ty_binary_path: impl Into<String>,
        ty_binary_sha256: impl Into<String>,
    ) -> Self {
        Self {
            cargo_profile: cargo_profile.into(),
            ty_binary_path: ty_binary_path.into(),
            ty_binary_sha256: ty_binary_sha256.into(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct BackendControls {
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub(super) interp_env: BTreeMap<String, String>,
    pub(super) trust_cg_env: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct LaunchControls {
    pub(super) tlc: TlcLaunchControls,
    pub(super) ty: TyLaunchControls,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TlcLaunchControls {
    pub(super) workers: usize,
    pub(super) jvm_args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) heap_xms: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) heap_xmx: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TyLaunchControls {
    pub(super) interp: TyModeLaunchControls,
    pub(super) trust_cg: TyModeLaunchControls,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TyModeLaunchControls {
    pub(super) workers: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cache_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) artifact_cache_disabled_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) native_callout_compile_jobs: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct BenchmarkGateFlags {
    pub(super) require_trust_cg_compiled_actions: bool,
    pub(super) require_trust_cg_all_actions: bool,
    pub(super) require_trust_cg_compiled_invariants: bool,
    pub(super) require_trust_cg_compiled_bfs: bool,
    pub(super) require_trust_cg_fused_level: bool,
    pub(super) require_trust_cg_native_fused_level: bool,
    pub(super) require_trust_cg_successor_telemetry: bool,
    pub(super) require_native_fused_flat_frontier_admission: bool,
    pub(super) require_flat_state_primary: bool,
    pub(super) require_flat_bfs_frontier: bool,
    pub(super) require_no_trust_cg_fallbacks: bool,
    pub(super) allow_trust_cg_invariant_rust_fallbacks: bool,
    pub(super) require_trust_cg_faster_than_tlc: bool,
    pub(super) require_trust_cg_execution_faster_than_tlc: bool,
}

impl BenchmarkGateFlags {
    pub(super) fn from_names(enabled: &[String], disabled: &[String]) -> Self {
        let mut flags = Self::default();
        for name in disabled {
            flags.set(name, false);
        }
        for name in enabled {
            flags.set(name, true);
        }
        flags
    }

    pub(super) fn set(&mut self, name: &str, value: bool) -> bool {
        match name {
            "require_trust_cg_compiled_actions" => self.require_trust_cg_compiled_actions = value,
            "require_trust_cg_all_actions" => self.require_trust_cg_all_actions = value,
            "require_trust_cg_compiled_invariants" => {
                self.require_trust_cg_compiled_invariants = value
            }
            "require_trust_cg_compiled_bfs" => self.require_trust_cg_compiled_bfs = value,
            "require_trust_cg_fused_level" => self.require_trust_cg_fused_level = value,
            "require_trust_cg_native_fused_level" => {
                self.require_trust_cg_native_fused_level = value
            }
            "require_trust_cg_successor_telemetry" => {
                self.require_trust_cg_successor_telemetry = value;
            }
            "require_native_fused_flat_frontier_admission" => {
                self.require_native_fused_flat_frontier_admission = value;
            }
            "require_flat_state_primary" => self.require_flat_state_primary = value,
            "require_flat_bfs_frontier" => self.require_flat_bfs_frontier = value,
            "require_no_trust_cg_fallbacks" => self.require_no_trust_cg_fallbacks = value,
            "allow_trust_cg_invariant_rust_fallbacks" => {
                self.allow_trust_cg_invariant_rust_fallbacks = value;
            }
            "require_trust_cg_faster_than_tlc" => self.require_trust_cg_faster_than_tlc = value,
            "require_trust_cg_execution_faster_than_tlc" => {
                self.require_trust_cg_execution_faster_than_tlc = value;
            }
            _ => return false,
        }
        true
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct BenchmarkRow {
    pub(super) spec: String,
    pub(super) tlc: TlcModeSummary,
    pub(super) interp: TyModeSummary,
    pub(super) trust_cg: TyModeSummary,
    pub(super) parity_interp_vs_tlc: bool,
    pub(super) parity_trust_cg_vs_tlc: bool,
    pub(super) speedup_interp_vs_tlc: Option<f64>,
    pub(super) speedup_trust_cg_vs_tlc: Option<f64>,
    pub(super) speedup_trust_cg_execution_vs_tlc: Option<f64>,
    pub(super) trust_cg_outcome: String,
    pub(super) trust_cg_evidence: TrustCgEvidence,
    pub(super) trust_cg_gate_failures: Vec<String>,
}

impl BenchmarkRow {
    pub(super) fn from_runs(
        spec: impl Into<String>,
        expected_states: Option<u64>,
        tlc_runs: Vec<TlcRunResult>,
        interp_runs: Vec<TyRunResult>,
        trust_cg_runs: Vec<TyRunResult>,
    ) -> Self {
        let spec = spec.into();
        let tlc = TlcModeSummary::from_runs(tlc_runs, expected_states);
        let interp = TyModeSummary::from_runs(interp_runs, expected_states);
        let trust_cg = TyModeSummary::from_runs(trust_cg_runs, expected_states);
        let parity_interp_vs_tlc = parity_vs_tlc(&tlc, &interp);
        let parity_trust_cg_vs_tlc = parity_vs_tlc(&tlc, &trust_cg);
        let speedup_interp_vs_tlc = ratio_or_none(tlc.median_seconds, interp.median_seconds);
        let speedup_trust_cg_vs_tlc = ratio_or_none(tlc.median_seconds, trust_cg.median_seconds);
        let speedup_trust_cg_execution_vs_tlc =
            ratio_or_none(tlc.median_seconds, trust_cg.execution_median_seconds);
        let trust_cg_outcome = trust_cg_outcome_label(
            tlc.median_seconds,
            trust_cg.median_seconds,
            trust_cg.execution_median_seconds,
        )
        .to_string();
        let trust_cg_evidence =
            TrustCgEvidence::classify(trust_cg.telemetry.as_ref(), speedup_trust_cg_vs_tlc);
        Self {
            spec,
            tlc,
            interp,
            trust_cg,
            parity_interp_vs_tlc,
            parity_trust_cg_vs_tlc,
            speedup_interp_vs_tlc,
            speedup_trust_cg_vs_tlc,
            speedup_trust_cg_execution_vs_tlc,
            trust_cg_outcome,
            trust_cg_evidence,
            trust_cg_gate_failures: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TlcModeSummary {
    pub(super) runs: Vec<TlcRunResult>,
    pub(super) median_seconds: Option<f64>,
    pub(super) median_peak_rss_bytes: Option<u64>,
    pub(super) state_values: Vec<u64>,
    pub(super) consistent_states: bool,
    pub(super) generated_state_values: Vec<u64>,
    pub(super) consistent_generated_states: bool,
    pub(super) raw_initial_state_values: Vec<u64>,
    pub(super) consistent_raw_initial_states: bool,
    pub(super) raw_successor_values: Vec<u64>,
    pub(super) consistent_raw_successors: bool,
    pub(super) expected_states: Option<u64>,
    pub(super) expected_states_match: Option<bool>,
    pub(super) all_ok: bool,
}

impl TlcModeSummary {
    pub(super) fn from_runs(runs: Vec<TlcRunResult>, expected_states: Option<u64>) -> Self {
        let elapsed = runs
            .iter()
            .filter(|run| run.ok())
            .map(|run| run.elapsed_seconds)
            .collect::<Vec<_>>();
        let state_values = sorted_known_values(runs.iter().filter_map(|run| run.states_found));
        let generated_state_values =
            sorted_known_values(runs.iter().filter_map(|run| run.states_generated));
        let raw_initial_state_values = sorted_known_values(
            runs.iter()
                .filter_map(|run| run.raw_initial_states_generated),
        );
        let raw_successor_values =
            sorted_known_values(runs.iter().filter_map(|run| run.raw_successors_generated));
        let expected_states_match = expected_states
            .map(|expected| runs.iter().all(|run| run.states_found == Some(expected)));
        Self {
            median_seconds: median(&elapsed),
            median_peak_rss_bytes: median_u64(runs.iter().filter_map(|run| run.peak_rss_bytes)),
            consistent_states: state_values.len() <= 1,
            consistent_generated_states: generated_state_values.len() <= 1,
            consistent_raw_initial_states: raw_initial_state_values.len() <= 1,
            consistent_raw_successors: raw_successor_values.len() <= 1,
            state_values,
            generated_state_values,
            raw_initial_state_values,
            raw_successor_values,
            expected_states,
            expected_states_match,
            all_ok: runs.iter().all(TlcRunResult::ok),
            runs,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TyModeSummary {
    pub(super) runs: Vec<TyRunResult>,
    pub(super) median_seconds: Option<f64>,
    pub(super) median_peak_rss_bytes: Option<u64>,
    pub(super) state_values: Vec<u64>,
    pub(super) consistent_states: bool,
    pub(super) generated_state_values: Vec<u64>,
    pub(super) consistent_generated_states: bool,
    pub(super) raw_initial_state_values: Vec<u64>,
    pub(super) consistent_raw_initial_states: bool,
    pub(super) raw_successor_values: Vec<u64>,
    pub(super) consistent_raw_successors: bool,
    pub(super) all_ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) telemetry: Option<TrustCgTelemetryAggregate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) execution_median_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) phase_median_seconds: Option<TrustCgPhaseMedianSeconds>,
    pub(super) expected_states: Option<u64>,
    pub(super) expected_states_match: Option<bool>,
}

impl TyModeSummary {
    pub(super) fn from_runs(runs: Vec<TyRunResult>, expected_states: Option<u64>) -> Self {
        let elapsed = runs
            .iter()
            .filter(|run| run.ok())
            .map(|run| run.elapsed_seconds)
            .collect::<Vec<_>>();
        let execution_times = runs
            .iter()
            .filter(|run| run.ok())
            .filter_map(|run| {
                run.trust_cg_telemetry
                    .as_ref()
                    .and_then(TrustCgTelemetry::compiled_bfs_execution_seconds)
            })
            .collect::<Vec<_>>();
        let phase_median_seconds = TrustCgPhaseMedianSeconds::from_runs(&runs);
        let state_values = sorted_known_values(runs.iter().filter_map(|run| run.states_found));
        let generated_state_values =
            sorted_known_values(runs.iter().filter_map(|run| run.states_generated));
        let raw_initial_state_values = sorted_known_values(
            runs.iter()
                .filter_map(|run| run.raw_initial_states_generated),
        );
        let raw_successor_values =
            sorted_known_values(runs.iter().filter_map(|run| run.raw_successors_generated));
        let telemetry = TrustCgTelemetryAggregate::from_run_telemetry(
            runs.iter()
                .filter(|run| run.ok())
                .filter_map(|run| run.trust_cg_telemetry.as_ref()),
        );
        let expected_states_match = expected_states
            .map(|expected| runs.iter().all(|run| run.states_found == Some(expected)));
        Self {
            median_seconds: median(&elapsed),
            median_peak_rss_bytes: median_u64(runs.iter().filter_map(|run| run.peak_rss_bytes)),
            consistent_states: state_values.len() <= 1,
            consistent_generated_states: generated_state_values.len() <= 1,
            consistent_raw_initial_states: raw_initial_state_values.len() <= 1,
            consistent_raw_successors: raw_successor_values.len() <= 1,
            state_values,
            generated_state_values,
            raw_initial_state_values,
            raw_successor_values,
            all_ok: runs.iter().all(TyRunResult::ok),
            telemetry,
            execution_median_seconds: if execution_times.is_empty() {
                None
            } else {
                median(&execution_times)
            },
            phase_median_seconds,
            expected_states,
            expected_states_match,
            runs,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TrustCgPhaseMedianSeconds {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cold_setup: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) native_runtime: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch_setup: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch_lowering: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch_assembly: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch_compile: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch_warm_cache_lookup: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch_artifact_materialization: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch_fallback_per_action_compile: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch_unattributed_setup: Option<f64>,
}

impl TrustCgPhaseMedianSeconds {
    fn from_runs(runs: &[TyRunResult]) -> Option<Self> {
        let ok_runs = runs.iter().filter(|run| run.ok()).collect::<Vec<_>>();
        let native_runtime = median_known(ok_runs.iter().filter_map(|run| {
            run.trust_cg_telemetry
                .as_ref()
                .and_then(TrustCgTelemetry::compiled_bfs_execution_seconds)
        }));
        let cold_setup = median_known(ok_runs.iter().filter_map(|run| {
            let native_runtime = run
                .trust_cg_telemetry
                .as_ref()
                .and_then(TrustCgTelemetry::compiled_bfs_execution_seconds)?;
            Some((run.elapsed_seconds - native_runtime).max(0.0))
        }));
        let batch_setup =
            phase_timing_median(&ok_runs, |item| item.native_action_callout_batch_setup_ms);
        let phase = Self {
            cold_setup,
            native_runtime,
            batch_setup,
            batch_lowering: phase_timing_median(&ok_runs, |item| {
                item.native_action_callout_batch_lowering_ms
            }),
            batch_assembly: phase_timing_median(&ok_runs, |item| {
                item.native_action_callout_batch_assembly_ms
            }),
            batch_compile: phase_timing_median(&ok_runs, |item| {
                item.native_action_callout_batch_compile_ms
            }),
            batch_warm_cache_lookup: phase_timing_median(&ok_runs, |item| {
                item.native_action_callout_batch_warm_cache_lookup_ms
            }),
            batch_artifact_materialization: phase_timing_median(&ok_runs, |item| {
                item.native_action_callout_batch_artifact_materialization_ms
            }),
            batch_fallback_per_action_compile: phase_timing_median(&ok_runs, |item| {
                item.native_action_callout_batch_fallback_per_action_compile_ms
            }),
            batch_unattributed_setup: median_known(ok_runs.iter().filter_map(|run| {
                run.trust_cg_telemetry.as_ref().and_then(
                    TrustCgTelemetry::native_action_callout_batch_unattributed_setup_seconds,
                )
            })),
        };
        phase.has_any_timing().then_some(phase)
    }

    fn has_any_timing(&self) -> bool {
        self.cold_setup.is_some()
            || self.native_runtime.is_some()
            || self.batch_setup.is_some()
            || self.batch_lowering.is_some()
            || self.batch_assembly.is_some()
            || self.batch_compile.is_some()
            || self.batch_warm_cache_lookup.is_some()
            || self.batch_artifact_materialization.is_some()
            || self.batch_fallback_per_action_compile.is_some()
            || self.batch_unattributed_setup.is_some()
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TlcRunResult {
    pub(super) tool: String,
    pub(super) spec_name: String,
    pub(super) run_index: Option<usize>,
    pub(super) workers: usize,
    pub(super) elapsed_seconds: f64,
    pub(super) peak_rss_bytes: Option<u64>,
    pub(super) states_found: Option<u64>,
    pub(super) distinct_states: Option<u64>,
    pub(super) transitions: Option<u64>,
    pub(super) raw_initial_states_generated: Option<u64>,
    pub(super) raw_successors_generated: Option<u64>,
    pub(super) states_generated: Option<u64>,
    pub(super) returncode: i32,
    pub(super) error: Option<String>,
    pub(super) artifact_dir: Option<String>,
}

impl TlcRunResult {
    fn ok(&self) -> bool {
        self.returncode == 0 && self.error.is_none()
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TyRunResult {
    pub(super) tool: String,
    pub(super) mode: String,
    pub(super) spec_name: String,
    pub(super) run_index: usize,
    pub(super) elapsed_seconds: f64,
    pub(super) peak_rss_bytes: Option<u64>,
    pub(super) states_found: Option<u64>,
    pub(super) transitions: Option<u64>,
    pub(super) raw_initial_states_generated: Option<u64>,
    pub(super) raw_successors_generated: Option<u64>,
    pub(super) states_generated: Option<u64>,
    pub(super) returncode: i32,
    pub(super) error: Option<String>,
    pub(super) artifact_dir: Option<String>,
    pub(super) workers: usize,
    pub(super) env_overrides: Option<BTreeMap<String, String>>,
    pub(super) trust_cg_telemetry: Option<TrustCgTelemetry>,
}

impl TyRunResult {
    fn ok(&self) -> bool {
        self.returncode == 0 && self.error.is_none()
    }
}

#[derive(Clone, Debug, Serialize, Default)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) transitions: Option<u64>,
}

impl TrustCgTelemetry {
    pub(super) fn compiled_bfs_execution_seconds(&self) -> Option<f64> {
        if let Some(nanos) = self.compiled_bfs_execution_nanos.filter(|nanos| *nanos > 0) {
            return Some(nanos as f64 / 1_000_000_000.0);
        }
        self.compiled_bfs_execution_seconds
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
    }

    fn native_action_callout_batch_unattributed_setup_seconds(&self) -> Option<f64> {
        let setup = millis_to_seconds(self.native_action_callout_batch_setup_ms)?;
        let known = [
            self.native_action_callout_batch_lowering_ms,
            self.native_action_callout_batch_assembly_ms,
            self.native_action_callout_batch_compile_ms,
            self.native_action_callout_batch_warm_cache_lookup_ms,
            self.native_action_callout_batch_artifact_materialization_ms,
            self.native_action_callout_batch_fallback_per_action_compile_ms,
        ]
        .into_iter()
        .filter_map(millis_to_seconds)
        .sum::<f64>();
        Some((setup - known).max(0.0))
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TrustCgTelemetryAggregate {
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
    pub(super) trust_cg_native_fused_regular_invariants_checked: bool,
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
    // Compatibility aliases keep the old aggregate JSON names populated; the
    // explicit fields spell out max, total, and all-runs meanings.
    pub(super) native_action_callout_batch_shard_count: Option<u64>,
    pub(super) native_action_callout_batch_shard_count_max: Option<u64>,
    pub(super) native_action_callout_batch_shard_count_total: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_enabled: Option<bool>,
    pub(super) native_action_callout_batch_warm_cache_enabled_all_runs: Option<bool>,
    pub(super) native_action_callout_batch_warm_cache_lookup_attempted: Option<bool>,
    pub(super) native_action_callout_batch_warm_cache_lookup_attempted_all_runs: Option<bool>,
    pub(super) native_action_callout_batch_warm_cache_hits: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_hits_max: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_hits_total: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_misses: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_misses_max: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_misses_total: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_stores: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_stores_max: Option<u64>,
    pub(super) native_action_callout_batch_warm_cache_stores_total: Option<u64>,
    pub(super) native_action_callout_batch_shard_warm_cache_statuses: Vec<String>,
    pub(super) native_action_callout_batch_shard_warm_cache_statuses_unique_all_runs: Vec<String>,
    pub(super) runs_with_native_action_callout_batch_warm_cache_lookup: usize,
    pub(super) runs_with_native_action_callout_batch_warm_cache_hit: usize,
    pub(super) runs_with_native_action_callout_batch_warm_cache_miss: usize,
    pub(super) runs_with_native_action_callout_batch_warm_cache_store: usize,
    pub(super) runs_with_compiled_bfs_step_active: usize,
    pub(super) runs_with_compiled_bfs_level_active: usize,
    pub(super) runs_with_compiled_bfs_level_loop_started: usize,
    pub(super) runs_with_compiled_bfs_zero_work: usize,
    pub(super) runs_with_compiled_bfs_execution_timing: usize,
    pub(super) runs_with_trust_cg_native_fused_level_active: usize,
    pub(super) runs_with_trust_cg_native_fused_level_built: usize,
    pub(super) runs_with_trust_cg_native_fused_regular_invariants_checked: usize,
    pub(super) runs_with_trust_cg_native_fused_state_constraints: usize,
    pub(super) runs_with_flat_bfs_frontier_active: usize,
    pub(super) fallback_reasons: Vec<String>,
}

impl TrustCgTelemetryAggregate {
    pub(super) fn from_run_telemetry<'a>(
        telemetry: impl IntoIterator<Item = &'a TrustCgTelemetry>,
    ) -> Option<Self> {
        let telemetry = telemetry.into_iter().collect::<Vec<_>>();
        if telemetry.is_empty() {
            return None;
        }
        let native_action_callout_batch_shard_count_max = max_known(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_shard_count),
        );
        let native_action_callout_batch_shard_count_total = sum_known(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_shard_count),
        );
        let native_action_callout_batch_warm_cache_enabled_all_runs = strict_optional_bool(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_warm_cache_enabled),
        );
        let native_action_callout_batch_warm_cache_lookup_attempted_all_runs = strict_optional_bool(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_warm_cache_lookup_attempted),
        );
        let native_action_callout_batch_warm_cache_hits_max = max_known(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_warm_cache_hits),
        );
        let native_action_callout_batch_warm_cache_hits_total = sum_known(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_warm_cache_hits),
        );
        let native_action_callout_batch_warm_cache_misses_max = max_known(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_warm_cache_misses),
        );
        let native_action_callout_batch_warm_cache_misses_total = sum_known(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_warm_cache_misses),
        );
        let native_action_callout_batch_warm_cache_stores_max = max_known(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_warm_cache_stores),
        );
        let native_action_callout_batch_warm_cache_stores_total = sum_known(
            telemetry
                .iter()
                .map(|item| item.native_action_callout_batch_warm_cache_stores),
        );
        let native_action_callout_batch_shard_warm_cache_statuses_unique_all_runs =
            unique_warm_cache_statuses(&telemetry);
        Some(Self {
            trust_cg_actions_compiled: min_known(
                telemetry.iter().map(|item| item.trust_cg_actions_compiled),
            ),
            trust_cg_actions_total: max_known(
                telemetry.iter().map(|item| item.trust_cg_actions_total),
            ),
            trust_cg_invariants_compiled: min_known(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_invariants_compiled),
            ),
            trust_cg_invariants_total: max_known(
                telemetry.iter().map(|item| item.trust_cg_invariants_total),
            ),
            trust_cg_state_constraints_compiled: min_known(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_state_constraints_compiled),
            ),
            trust_cg_state_constraints_total: max_known(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_state_constraints_total),
            ),
            compiled_bfs_step_active: all_true(
                telemetry.iter().map(|item| item.compiled_bfs_step_active),
            ),
            compiled_bfs_level_active: all_true(
                telemetry.iter().map(|item| item.compiled_bfs_level_active),
            ),
            compiled_bfs_level_loop_started: all_true(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_level_loop_started),
            ),
            compiled_bfs_level_loop_initial_states: min_known(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_level_loop_initial_states),
            ),
            compiled_bfs_level_loop_fused: strict_optional_bool(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_level_loop_fused),
            ),
            compiled_bfs_levels_completed: min_known(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_levels_completed),
            ),
            compiled_bfs_parents_processed: min_known(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_parents_processed),
            ),
            compiled_bfs_successors_generated: min_known(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_successors_generated),
            ),
            compiled_bfs_successors_new: min_known(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_successors_new),
            ),
            compiled_bfs_total_states: max_known(
                telemetry.iter().map(|item| item.compiled_bfs_total_states),
            ),
            compiled_bfs_zero_work: telemetry.iter().any(|item| item.compiled_bfs_zero_work),
            compiled_bfs_execution_nanos: max_known(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_execution_nanos),
            ),
            compiled_bfs_execution_seconds: max_known_f64(
                telemetry
                    .iter()
                    .filter_map(|item| item.compiled_bfs_execution_seconds()),
            ),
            trust_cg_bfs_level_active: all_true(
                telemetry.iter().map(|item| item.trust_cg_bfs_level_active),
            ),
            trust_cg_native_fused_level_built: all_true(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_fused_level_built),
            ),
            trust_cg_native_fused_level_active: all_true(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_fused_level_active),
            ),
            trust_cg_native_fused_regular_invariants_checked: telemetry
                .iter()
                .all(|item| item.trust_cg_native_fused_regular_invariants_checked == Some(true)),
            trust_cg_native_fused_mode: aggregate_native_fused_mode(&telemetry),
            trust_cg_native_fused_invariant_count: aggregate_native_fused_min(&telemetry, |item| {
                item.trust_cg_native_fused_invariant_count
            }),
            trust_cg_native_fused_state_constraint_count: aggregate_native_fused_min(
                &telemetry,
                |item| item.trust_cg_native_fused_state_constraint_count,
            ),
            trust_cg_native_fused_state_len: aggregate_native_fused_state_len(&telemetry),
            trust_cg_native_fused_local_dedup: strict_optional_bool(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_fused_local_dedup),
            ),
            trust_cg_native_bfs_trace_generated: max_known(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_bfs_trace_generated),
            ),
            trust_cg_native_bfs_trace_state_count: max_known(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_bfs_trace_state_count),
            ),
            trust_cg_native_bfs_trace_parents_processed: max_known(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_bfs_trace_parents_processed),
            ),
            trust_cg_bfs_level_loop_kind: aggregate_loop_kind(&telemetry),
            trust_cg_native_fused_flat_frontier_admission_active: strict_optional_bool(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_fused_flat_frontier_admission_active),
            ),
            compiled_bfs_flat_frontier_admitted: strict_optional_bool(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_flat_frontier_admitted),
            ),
            flat_state_primary: strict_optional_bool(
                telemetry.iter().map(|item| item.flat_state_primary),
            ),
            flat_bfs_frontier_active: strict_optional_bool(
                telemetry.iter().map(|item| item.flat_bfs_frontier_active),
            ),
            flat_bfs_frontier_fallbacks: max_known(
                telemetry
                    .iter()
                    .map(|item| item.flat_bfs_frontier_fallbacks),
            ),
            native_action_callout_batch_artifact_identity_source: aggregate_optional_string(
                telemetry.iter().filter_map(|item| {
                    item.native_action_callout_batch_artifact_identity_source
                        .as_deref()
                }),
            ),
            native_action_callout_batch_artifact_identity: aggregate_optional_string(
                telemetry.iter().filter_map(|item| {
                    item.native_action_callout_batch_artifact_identity
                        .as_deref()
                }),
            ),
            native_action_callout_batch_artifact_cache_digest: aggregate_optional_string(
                telemetry.iter().filter_map(|item| {
                    item.native_action_callout_batch_artifact_cache_digest
                        .as_deref()
                }),
            ),
            native_action_callout_batch_cache_key: aggregate_optional_string(
                telemetry
                    .iter()
                    .filter_map(|item| item.native_action_callout_batch_cache_key.as_deref()),
            ),
            native_action_callout_batch_artifact_cacheable: strict_optional_bool(
                telemetry
                    .iter()
                    .map(|item| item.native_action_callout_batch_artifact_cacheable),
            ),
            native_action_callout_batch_artifact_cache_disabled_by_env: strict_optional_bool(
                telemetry
                    .iter()
                    .map(|item| item.native_action_callout_batch_artifact_cache_disabled_by_env),
            ),
            native_action_callout_batch_shard_count: native_action_callout_batch_shard_count_max,
            native_action_callout_batch_shard_count_max,
            native_action_callout_batch_shard_count_total,
            native_action_callout_batch_warm_cache_enabled:
                native_action_callout_batch_warm_cache_enabled_all_runs,
            native_action_callout_batch_warm_cache_enabled_all_runs,
            native_action_callout_batch_warm_cache_lookup_attempted:
                native_action_callout_batch_warm_cache_lookup_attempted_all_runs,
            native_action_callout_batch_warm_cache_lookup_attempted_all_runs,
            native_action_callout_batch_warm_cache_hits:
                native_action_callout_batch_warm_cache_hits_max,
            native_action_callout_batch_warm_cache_hits_max,
            native_action_callout_batch_warm_cache_hits_total,
            native_action_callout_batch_warm_cache_misses:
                native_action_callout_batch_warm_cache_misses_max,
            native_action_callout_batch_warm_cache_misses_max,
            native_action_callout_batch_warm_cache_misses_total,
            native_action_callout_batch_warm_cache_stores:
                native_action_callout_batch_warm_cache_stores_max,
            native_action_callout_batch_warm_cache_stores_max,
            native_action_callout_batch_warm_cache_stores_total,
            native_action_callout_batch_shard_warm_cache_statuses:
                native_action_callout_batch_shard_warm_cache_statuses_unique_all_runs.clone(),
            native_action_callout_batch_shard_warm_cache_statuses_unique_all_runs,
            runs_with_native_action_callout_batch_warm_cache_lookup: telemetry
                .iter()
                .filter(|item| {
                    item.native_action_callout_batch_warm_cache_lookup_attempted == Some(true)
                })
                .count(),
            runs_with_native_action_callout_batch_warm_cache_hit: telemetry
                .iter()
                .filter(|item| {
                    item.native_action_callout_batch_warm_cache_hits
                        .is_some_and(|count| count > 0)
                })
                .count(),
            runs_with_native_action_callout_batch_warm_cache_miss: telemetry
                .iter()
                .filter(|item| {
                    item.native_action_callout_batch_warm_cache_misses
                        .is_some_and(|count| count > 0)
                })
                .count(),
            runs_with_native_action_callout_batch_warm_cache_store: telemetry
                .iter()
                .filter(|item| {
                    item.native_action_callout_batch_warm_cache_stores
                        .is_some_and(|count| count > 0)
                })
                .count(),
            runs_with_compiled_bfs_step_active: count_true(
                telemetry.iter().map(|item| item.compiled_bfs_step_active),
            ),
            runs_with_compiled_bfs_level_active: count_true(
                telemetry.iter().map(|item| item.compiled_bfs_level_active),
            ),
            runs_with_compiled_bfs_level_loop_started: count_true(
                telemetry
                    .iter()
                    .map(|item| item.compiled_bfs_level_loop_started),
            ),
            runs_with_compiled_bfs_zero_work: count_true(
                telemetry.iter().map(|item| item.compiled_bfs_zero_work),
            ),
            runs_with_compiled_bfs_execution_timing: telemetry
                .iter()
                .filter(|item| item.compiled_bfs_execution_seconds().is_some())
                .count(),
            runs_with_trust_cg_native_fused_level_active: count_true(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_fused_level_active),
            ),
            runs_with_trust_cg_native_fused_level_built: count_true(
                telemetry
                    .iter()
                    .map(|item| item.trust_cg_native_fused_level_built),
            ),
            runs_with_trust_cg_native_fused_regular_invariants_checked: telemetry
                .iter()
                .filter(|item| item.trust_cg_native_fused_regular_invariants_checked == Some(true))
                .count(),
            runs_with_trust_cg_native_fused_state_constraints: telemetry
                .iter()
                .filter(|item| {
                    item.trust_cg_native_fused_state_constraint_count
                        .is_some_and(|count| count > 0)
                })
                .count(),
            runs_with_flat_bfs_frontier_active: telemetry
                .iter()
                .filter(|item| item.flat_bfs_frontier_active == Some(true))
                .count(),
            fallback_reasons: unique_fallback_reasons(&telemetry),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TrustCgEvidence {
    pub(super) native_fused: bool,
    pub(super) native_fused_regular_invariants_checked: Option<bool>,
    pub(super) action_only: Option<bool>,
    pub(super) flat_layout: Option<bool>,
    pub(super) tlc_wins: bool,
    pub(super) winner: String,
}

impl TrustCgEvidence {
    fn classify(
        telemetry: Option<&TrustCgTelemetryAggregate>,
        speedup_trust_cg_vs_tlc: Option<f64>,
    ) -> Self {
        let native_fused = telemetry.is_some_and(native_fused_execution_evidence_active);
        let native_fused_regular_invariants_checked =
            telemetry.map(|item| item.trust_cg_native_fused_regular_invariants_checked);
        let action_only = telemetry.and_then(|item| {
            if item.trust_cg_native_fused_mode.as_deref() == Some("action_only") {
                return Some(true);
            }
            if item.trust_cg_native_fused_mode.as_deref() == Some("invariant_checking") {
                return Some(false);
            }
            let actions_all = fraction_is_complete(
                item.trust_cg_actions_compiled,
                item.trust_cg_actions_total,
                true,
            )?;
            let invariants_all = fraction_is_complete(
                item.trust_cg_invariants_compiled,
                item.trust_cg_invariants_total,
                false,
            )?;
            Some(actions_all && !invariants_all)
        });
        let flat_layout = telemetry.and_then(|item| {
            let flat_primary = item.flat_state_primary == Some(true);
            let flat_frontier_clean = item.flat_bfs_frontier_active == Some(true)
                && item.flat_bfs_frontier_fallbacks == Some(0);
            let native_fused_flat_frontier_admitted =
                item.trust_cg_native_fused_flat_frontier_admission_active == Some(true)
                    && item.compiled_bfs_flat_frontier_admitted == Some(true);

            if item.flat_bfs_frontier_active == Some(false)
                || item
                    .flat_bfs_frontier_fallbacks
                    .is_some_and(|fallbacks| fallbacks != 0)
                || (!flat_primary
                    && (item.trust_cg_native_fused_flat_frontier_admission_active == Some(false)
                        || item.compiled_bfs_flat_frontier_admitted == Some(false)))
            {
                return Some(false);
            }
            if flat_frontier_clean && (flat_primary || native_fused_flat_frontier_admitted) {
                return Some(true);
            }
            None
        });
        Self {
            native_fused,
            native_fused_regular_invariants_checked,
            action_only,
            flat_layout,
            tlc_wins: speedup_trust_cg_vs_tlc.is_some_and(|speedup| speedup < 1.0),
            winner: winner_label(speedup_trust_cg_vs_tlc).to_string(),
        }
    }
}

pub(super) fn median(values: &[f64]) -> Option<f64> {
    let mut values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) / 2.0)
    }
}

fn median_u64(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some(values[mid - 1] / 2 + values[mid] / 2 + (values[mid - 1] % 2 + values[mid] % 2) / 2)
    }
}

fn median_known(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let values = values.into_iter().collect::<Vec<_>>();
    median(&values)
}

fn phase_timing_median(
    runs: &[&TyRunResult],
    field: impl Fn(&TrustCgTelemetry) -> Option<u64>,
) -> Option<f64> {
    median_known(runs.iter().filter_map(|run| {
        let millis = run.trust_cg_telemetry.as_ref().and_then(&field)?;
        millis_to_seconds(Some(millis))
    }))
}

fn millis_to_seconds(value: Option<u64>) -> Option<f64> {
    value.map(|millis| millis as f64 / 1000.0)
}

fn parity_vs_tlc(tlc: &TlcModeSummary, mode: &TyModeSummary) -> bool {
    let Some(tlc_state_value) = only_value(&tlc.state_values) else {
        return false;
    };
    let Some(tlc_generated_value) = only_value(&tlc.generated_state_values) else {
        return false;
    };
    let Some(tlc_raw_initial_value) = only_value(&tlc.raw_initial_state_values) else {
        return false;
    };
    let Some(tlc_raw_successor_value) = only_value(&tlc.raw_successor_values) else {
        return false;
    };
    mode.consistent_states
        && only_value(&mode.state_values) == Some(tlc_state_value)
        && mode.consistent_generated_states
        && only_value(&mode.generated_state_values) == Some(tlc_generated_value)
        && mode.consistent_raw_initial_states
        && only_value(&mode.raw_initial_state_values) == Some(tlc_raw_initial_value)
        && mode.consistent_raw_successors
        && only_value(&mode.raw_successor_values) == Some(tlc_raw_successor_value)
        && tlc.expected_states_match != Some(false)
        && mode.expected_states_match != Some(false)
}

fn only_value(values: &[u64]) -> Option<u64> {
    if values.len() == 1 {
        Some(values[0])
    } else {
        None
    }
}

fn ratio_or_none(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    let numerator = numerator.filter(|value| value.is_finite())?;
    let denominator = denominator.filter(|value| value.is_finite() && *value != 0.0)?;
    Some(numerator / denominator)
}

fn sorted_known_values(values: impl IntoIterator<Item = u64>) -> Vec<u64> {
    values
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn min_known(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    values.into_iter().flatten().min()
}

fn max_known(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    values.into_iter().flatten().max()
}

fn sum_known(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    let mut total = 0_u64;
    let mut has_known = false;
    for value in values.into_iter().flatten() {
        total = total.checked_add(value)?;
        has_known = true;
    }
    has_known.then_some(total)
}

fn max_known_f64(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    values
        .into_iter()
        .filter(|value| value.is_finite())
        .max_by(f64::total_cmp)
}

fn all_true(values: impl IntoIterator<Item = bool>) -> bool {
    values.into_iter().all(|value| value)
}

fn count_true(values: impl IntoIterator<Item = bool>) -> usize {
    values.into_iter().filter(|value| *value).count()
}

fn strict_optional_bool(values: impl IntoIterator<Item = Option<bool>>) -> Option<bool> {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.contains(&Some(false)) {
        return Some(false);
    }
    if values.iter().all(|value| *value == Some(true)) {
        return Some(true);
    }
    None
}

fn aggregate_optional_string<'a>(values: impl IntoIterator<Item = &'a str>) -> Option<String> {
    let values = values.into_iter().collect::<BTreeSet<_>>();
    match values.len() {
        0 => None,
        1 => values.into_iter().next().map(ToString::to_string),
        _ => Some("mixed".to_string()),
    }
}

fn aggregate_loop_kind(telemetry: &[&TrustCgTelemetry]) -> Option<String> {
    let kinds = telemetry
        .iter()
        .filter_map(|item| item.trust_cg_bfs_level_loop_kind.as_deref())
        .collect::<BTreeSet<_>>();
    match kinds.len() {
        0 => None,
        1 => kinds.into_iter().next().map(ToString::to_string),
        _ => Some("mixed".to_string()),
    }
}

fn aggregate_native_fused_mode(telemetry: &[&TrustCgTelemetry]) -> Option<String> {
    let native_runs = telemetry
        .iter()
        .copied()
        .filter(|item| item.trust_cg_native_fused_level_active)
        .collect::<Vec<_>>();
    if native_runs.is_empty()
        || native_runs
            .iter()
            .any(|item| item.trust_cg_native_fused_mode.is_none())
    {
        return None;
    }
    let modes = native_runs
        .iter()
        .filter_map(|item| item.trust_cg_native_fused_mode.as_deref())
        .collect::<BTreeSet<_>>();
    match modes.len() {
        0 => None,
        1 => modes.into_iter().next().map(ToString::to_string),
        _ => Some("mixed".to_string()),
    }
}

fn aggregate_native_fused_min(
    telemetry: &[&TrustCgTelemetry],
    field: impl Fn(&TrustCgTelemetry) -> Option<u64>,
) -> Option<u64> {
    let native_runs = telemetry
        .iter()
        .copied()
        .filter(|item| item.trust_cg_native_fused_level_active)
        .collect::<Vec<_>>();
    if native_runs.is_empty() || native_runs.iter().any(|item| field(item).is_none()) {
        return None;
    }
    native_runs.into_iter().filter_map(field).min()
}

fn aggregate_native_fused_state_len(telemetry: &[&TrustCgTelemetry]) -> Option<u64> {
    let native_runs = telemetry
        .iter()
        .copied()
        .filter(|item| item.trust_cg_native_fused_level_active)
        .collect::<Vec<_>>();
    if native_runs.is_empty()
        || native_runs
            .iter()
            .any(|item| item.trust_cg_native_fused_state_len.is_none())
    {
        return None;
    }
    let lengths = native_runs
        .iter()
        .filter_map(|item| item.trust_cg_native_fused_state_len)
        .collect::<BTreeSet<_>>();
    if lengths.len() == 1 {
        lengths.into_iter().next()
    } else {
        None
    }
}

fn unique_fallback_reasons(telemetry: &[&TrustCgTelemetry]) -> Vec<String> {
    let mut reasons = Vec::new();
    for item in telemetry {
        for reason in &item.fallback_reasons {
            if !reasons.contains(reason) {
                reasons.push(reason.clone());
            }
        }
    }
    reasons
}

fn unique_warm_cache_statuses(telemetry: &[&TrustCgTelemetry]) -> Vec<String> {
    let mut statuses = Vec::new();
    for item in telemetry {
        for status in &item.native_action_callout_batch_shard_warm_cache_statuses {
            if !statuses.contains(status) {
                statuses.push(status.clone());
            }
        }
    }
    statuses
}

fn fraction_is_complete(
    numerator: Option<u64>,
    denominator: Option<u64>,
    require_positive: bool,
) -> Option<bool> {
    let numerator = numerator?;
    let denominator = denominator?;
    if require_positive && denominator == 0 {
        return Some(false);
    }
    Some(numerator == denominator)
}

fn native_fused_execution_evidence_active(telemetry: &TrustCgTelemetryAggregate) -> bool {
    telemetry.trust_cg_native_fused_level_active
        && telemetry.trust_cg_native_fused_level_built
        && telemetry.compiled_bfs_level_active
        && telemetry.compiled_bfs_level_loop_started
        && telemetry.compiled_bfs_level_loop_fused == Some(true)
        && telemetry
            .compiled_bfs_level_loop_initial_states
            .is_some_and(|value| value > 0)
        && telemetry.trust_cg_bfs_level_loop_kind.as_deref()
            == Some("native_fused_trust_cg_parent_loop")
        && telemetry
            .compiled_bfs_levels_completed
            .is_some_and(|value| value > 0)
        && telemetry
            .compiled_bfs_parents_processed
            .is_some_and(|value| value > 0)
        && telemetry
            .compiled_bfs_successors_generated
            .is_some_and(|value| value > 0)
        && telemetry
            .compiled_bfs_total_states
            .is_some_and(|value| value > 0)
}

fn winner_label(speedup_trust_cg_vs_tlc: Option<f64>) -> &'static str {
    let Some(speedup) = speedup_trust_cg_vs_tlc.filter(|value| value.is_finite()) else {
        return "n/a";
    };
    if speedup > 1.0 {
        "trust-cg"
    } else if speedup < 1.0 {
        "TLC"
    } else {
        "tie"
    }
}

fn trust_cg_outcome_label(
    tlc_seconds: Option<f64>,
    trust_cg_wall_seconds: Option<f64>,
    trust_cg_native_seconds: Option<f64>,
) -> &'static str {
    let (Some(tlc_seconds), Some(trust_cg_wall_seconds)) = (
        tlc_seconds.filter(|value| value.is_finite()),
        trust_cg_wall_seconds.filter(|value| value.is_finite()),
    ) else {
        return "runtime evidence incomplete";
    };
    if trust_cg_wall_seconds < tlc_seconds {
        return "trust-cg cold start wins";
    }
    if trust_cg_native_seconds
        .filter(|value| value.is_finite())
        .is_some_and(|native_seconds| native_seconds < tlc_seconds)
    {
        return "native execution wins, cold start loses";
    }
    "TLC wall-clock wins"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tlc_run(index: usize, elapsed_seconds: f64) -> TlcRunResult {
        TlcRunResult {
            tool: "tlc".to_string(),
            spec_name: "Spec".to_string(),
            run_index: Some(index),
            workers: 1,
            elapsed_seconds,
            peak_rss_bytes: Some(8 * 1024 * 1024),
            states_found: Some(10),
            distinct_states: Some(10),
            transitions: None,
            raw_initial_states_generated: Some(1),
            raw_successors_generated: Some(19),
            states_generated: Some(20),
            returncode: 0,
            error: None,
            artifact_dir: Some(format!("Spec/tlc-run{index}")),
        }
    }

    fn ty_run(
        mode: &str,
        index: usize,
        elapsed_seconds: f64,
        telemetry: Option<TrustCgTelemetry>,
    ) -> TyRunResult {
        TyRunResult {
            tool: "ty".to_string(),
            mode: mode.to_string(),
            spec_name: "Spec".to_string(),
            run_index: index,
            elapsed_seconds,
            peak_rss_bytes: Some(4 * 1024 * 1024),
            states_found: Some(10),
            transitions: Some(20),
            raw_initial_states_generated: Some(1),
            raw_successors_generated: Some(19),
            states_generated: Some(20),
            returncode: 0,
            error: None,
            artifact_dir: Some(format!("Spec/{mode}-run{index}")),
            workers: 1,
            env_overrides: Some(BTreeMap::from([(
                "TY_trust_cg".to_string(),
                "1".to_string(),
            )])),
            trust_cg_telemetry: telemetry,
        }
    }

    fn native_telemetry(execution_nanos: u64) -> TrustCgTelemetry {
        TrustCgTelemetry {
            trust_cg_actions_compiled: Some(4),
            trust_cg_actions_total: Some(4),
            trust_cg_invariants_compiled: Some(1),
            trust_cg_invariants_total: Some(1),
            compiled_bfs_step_active: true,
            compiled_bfs_level_active: true,
            compiled_bfs_level_loop_started: true,
            compiled_bfs_level_loop_initial_states: Some(1),
            compiled_bfs_level_loop_fused: Some(true),
            compiled_bfs_levels_completed: Some(1),
            compiled_bfs_parents_processed: Some(2),
            compiled_bfs_successors_generated: Some(20),
            compiled_bfs_successors_new: Some(9),
            compiled_bfs_total_states: Some(10),
            compiled_bfs_execution_nanos: Some(execution_nanos),
            trust_cg_bfs_level_active: true,
            trust_cg_native_fused_level_built: true,
            trust_cg_native_fused_level_active: true,
            trust_cg_native_fused_regular_invariants_checked: Some(true),
            trust_cg_native_fused_mode: Some("invariant_checking".to_string()),
            trust_cg_native_fused_invariant_count: Some(1),
            trust_cg_native_fused_state_constraint_count: Some(0),
            trust_cg_native_fused_state_len: Some(2),
            trust_cg_native_fused_local_dedup: Some(false),
            trust_cg_bfs_level_loop_kind: Some("native_fused_trust_cg_parent_loop".to_string()),
            trust_cg_native_fused_flat_frontier_admission_active: Some(false),
            compiled_bfs_flat_frontier_admitted: Some(true),
            flat_state_primary: Some(true),
            flat_bfs_frontier_active: Some(true),
            flat_bfs_frontier_fallbacks: Some(0),
            native_action_callout_batch_artifact_identity_source: Some(
                "trust_cg_compiled_batch_stats".to_string(),
            ),
            native_action_callout_batch_artifact_identity: Some(
                "trust_cg_batch_jit:shared_high_performance_engine:abc123".to_string(),
            ),
            native_action_callout_batch_artifact_cache_digest: Some("abc123".to_string()),
            native_action_callout_batch_cache_key: Some(
                "trust_cg_batch_jit_cache:abc123".to_string(),
            ),
            native_action_callout_batch_artifact_cacheable: Some(false),
            native_action_callout_batch_artifact_cache_disabled_by_env: Some(true),
            native_action_callout_batch_shard_count: Some(1),
            native_action_callout_batch_warm_cache_enabled: Some(false),
            native_action_callout_batch_warm_cache_lookup_attempted: Some(false),
            native_action_callout_batch_warm_cache_hits: Some(0),
            native_action_callout_batch_warm_cache_misses: Some(0),
            native_action_callout_batch_warm_cache_stores: Some(0),
            native_action_callout_batch_setup_ms: Some(80),
            native_action_callout_batch_lowering_ms: Some(10),
            native_action_callout_batch_assembly_ms: Some(5),
            native_action_callout_batch_compile_ms: Some(30),
            native_action_callout_batch_warm_cache_lookup_ms: Some(4),
            native_action_callout_batch_artifact_materialization_ms: Some(6),
            native_action_callout_batch_fallback_per_action_compile_ms: Some(0),
            native_action_callout_batch_shard_warm_cache_statuses: vec!["disabled".to_string()],
            transitions: Some(20),
            ..TrustCgTelemetry::default()
        }
    }

    #[test]
    fn median_handles_odd_even_and_empty_inputs() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 2.0]), Some(3.0));
    }

    #[test]
    fn row_aggregates_policy_summary_fields_and_speedups() {
        let mut failed_telemetry = native_telemetry(9_000_000_000);
        failed_telemetry.native_action_callout_batch_shard_count = Some(99);
        failed_telemetry.native_action_callout_batch_warm_cache_hits = Some(99);
        failed_telemetry.native_action_callout_batch_setup_ms = Some(9_000);
        let mut failed_run = ty_run("trust-cg", 3, 99.0, Some(failed_telemetry));
        failed_run.returncode = 1;
        failed_run.error = Some("failed run should not contribute telemetry".to_string());

        let row = BenchmarkRow::from_runs(
            "Spec",
            Some(10),
            vec![tlc_run(1, 3.0), tlc_run(2, 5.0)],
            vec![
                ty_run("interp", 1, 2.0, None),
                ty_run("interp", 2, 4.0, None),
            ],
            vec![
                ty_run("trust-cg", 1, 1.0, Some(native_telemetry(1_000_000_000))),
                ty_run("trust-cg", 2, 3.0, Some(native_telemetry(3_000_000_000))),
                failed_run,
            ],
        );

        assert!(row.parity_interp_vs_tlc);
        assert!(row.parity_trust_cg_vs_tlc);
        assert_eq!(row.speedup_interp_vs_tlc, Some(4.0 / 3.0));
        assert_eq!(row.speedup_trust_cg_vs_tlc, Some(2.0));
        assert_eq!(row.speedup_trust_cg_execution_vs_tlc, Some(2.0));
        assert_eq!(row.trust_cg.median_seconds, Some(2.0));
        assert_eq!(row.trust_cg.execution_median_seconds, Some(2.0));
        assert_eq!(row.trust_cg_outcome, "trust-cg cold start wins");
        let phase = row.trust_cg.phase_median_seconds.as_ref().unwrap();
        assert_eq!(phase.native_runtime, Some(2.0));
        assert_eq!(phase.batch_setup, Some(0.080));
        assert_eq!(phase.batch_compile, Some(0.030));
        assert_eq!(phase.batch_warm_cache_lookup, Some(0.004));
        assert_eq!(phase.batch_artifact_materialization, Some(0.006));
        assert_eq!(phase.batch_unattributed_setup, Some(0.025));
        assert!(row.trust_cg_evidence.native_fused);
        assert_eq!(row.trust_cg_evidence.winner, "trust-cg");

        let value = serde_json::to_value(&row).unwrap();
        assert_eq!(value["tlc"]["runs"][0]["states_generated"], json!(20));
        assert_eq!(
            value["tlc"]["runs"][0]["raw_initial_states_generated"],
            json!(1)
        );
        assert_eq!(
            value["tlc"]["runs"][0]["raw_successors_generated"],
            json!(19)
        );
        assert_eq!(value["tlc"]["runs"][0]["transitions"], json!(None::<u64>));
        assert_eq!(value["interp"]["runs"][0]["transitions"], json!(20));
        assert_eq!(value["interp"]["runs"][0]["states_generated"], json!(20));
        assert_eq!(
            value["interp"]["runs"][0]["raw_successors_generated"],
            json!(19)
        );
        assert!(value["interp"].get("telemetry").is_none());
        assert_eq!(value["trust_cg"]["execution_median_seconds"], json!(2.0));
        assert_eq!(value["trust_cg_outcome"], json!("trust-cg cold start wins"));
        assert_eq!(
            value["trust_cg"]["telemetry"]["compiled_bfs_execution_seconds"],
            json!(3.0)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_shard_count_total"],
            json!(2)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_hits_max"],
            json!(0)
        );
        assert_eq!(
            value["trust_cg"]["runs"][0]["trust_cg_telemetry"]["fallback_reasons"],
            json!([])
        );
        assert_eq!(
            value["trust_cg"]["runs"][0]["trust_cg_telemetry"]
                ["native_action_callout_batch_artifact_identity"],
            json!("trust_cg_batch_jit:shared_high_performance_engine:abc123")
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_cache_key"],
            json!("trust_cg_batch_jit_cache:abc123")
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_shard_count"],
            json!(1)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_shard_count_max"],
            json!(1)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_shard_count_total"],
            json!(2)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_enabled"],
            json!(false)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]
                ["native_action_callout_batch_warm_cache_enabled_all_runs"],
            json!(false)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]
                ["native_action_callout_batch_warm_cache_lookup_attempted_all_runs"],
            json!(false)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_hits"],
            json!(0)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_hits_max"],
            json!(0)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_hits_total"],
            json!(0)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_misses_max"],
            json!(0)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_misses_total"],
            json!(0)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_stores_max"],
            json!(0)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_warm_cache_stores_total"],
            json!(0)
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]["native_action_callout_batch_shard_warm_cache_statuses"],
            json!(["disabled"])
        );
        assert_eq!(
            value["trust_cg"]["telemetry"]
                ["native_action_callout_batch_shard_warm_cache_statuses_unique_all_runs"],
            json!(["disabled"])
        );
        assert_eq!(
            value["trust_cg"]["runs"][0]["trust_cg_telemetry"]
                ["native_action_callout_batch_setup_ms"],
            json!(80)
        );
        assert_eq!(
            value["trust_cg"]["phase_median_seconds"]["batch_setup"],
            json!(0.080)
        );
        assert_eq!(
            value["trust_cg"]["phase_median_seconds"]["batch_unattributed_setup"],
            json!(0.025)
        );
    }

    #[test]
    fn row_reports_native_execution_win_when_cold_wall_loses() {
        let row = BenchmarkRow::from_runs(
            "Spec",
            Some(10),
            vec![tlc_run(1, 2.0)],
            vec![ty_run("interp", 1, 2.5, None)],
            vec![ty_run(
                "trust-cg",
                1,
                3.0,
                Some(native_telemetry(1_000_000_000)),
            )],
        );

        assert_eq!(row.speedup_trust_cg_vs_tlc, Some(2.0 / 3.0));
        assert_eq!(row.speedup_trust_cg_execution_vs_tlc, Some(2.0));
        assert_eq!(row.trust_cg_evidence.winner, "TLC");
        assert_eq!(
            row.trust_cg_outcome,
            "native execution wins, cold start loses"
        );
    }

    #[test]
    fn telemetry_aggregate_uses_min_max_and_strict_bool_policy() {
        let mut first = native_telemetry(1_000_000_000);
        first.trust_cg_actions_compiled = Some(3);
        first.trust_cg_actions_total = Some(4);
        first.fallback_reasons = vec!["reason A".to_string()];
        let mut second = native_telemetry(2_000_000_000);
        second.trust_cg_actions_compiled = Some(4);
        second.trust_cg_actions_total = Some(5);
        second.fallback_reasons = vec!["reason A".to_string(), "reason B".to_string()];
        second.native_action_callout_batch_shard_count = Some(2);
        second.native_action_callout_batch_warm_cache_enabled = Some(true);
        second.native_action_callout_batch_warm_cache_lookup_attempted = Some(true);
        second.native_action_callout_batch_warm_cache_hits = Some(1);
        second.native_action_callout_batch_warm_cache_misses = Some(2);
        second.native_action_callout_batch_warm_cache_stores = Some(3);
        second.native_action_callout_batch_shard_warm_cache_statuses =
            vec!["hit".to_string(), "miss".to_string()];

        let aggregate = TrustCgTelemetryAggregate::from_run_telemetry([&first, &second]).unwrap();

        assert_eq!(aggregate.trust_cg_actions_compiled, Some(3));
        assert_eq!(aggregate.trust_cg_actions_total, Some(5));
        assert_eq!(aggregate.compiled_bfs_level_loop_fused, Some(true));
        assert_eq!(aggregate.compiled_bfs_execution_nanos, Some(2_000_000_000));
        assert_eq!(aggregate.compiled_bfs_execution_seconds, Some(2.0));
        assert_eq!(aggregate.runs_with_compiled_bfs_execution_timing, 2);
        assert_eq!(
            aggregate
                .native_action_callout_batch_artifact_identity
                .as_deref(),
            Some("trust_cg_batch_jit:shared_high_performance_engine:abc123")
        );
        assert_eq!(
            aggregate.native_action_callout_batch_cache_key.as_deref(),
            Some("trust_cg_batch_jit_cache:abc123")
        );
        assert_eq!(
            aggregate.native_action_callout_batch_artifact_cacheable,
            Some(false)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_artifact_cache_disabled_by_env,
            Some(true)
        );
        assert_eq!(aggregate.native_action_callout_batch_shard_count, Some(2));
        assert_eq!(
            aggregate.native_action_callout_batch_shard_count_max,
            Some(2)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_shard_count_total,
            Some(3)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_enabled,
            Some(false)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_enabled_all_runs,
            Some(false)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_lookup_attempted,
            Some(false)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_lookup_attempted_all_runs,
            Some(false)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_hits,
            Some(1)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_hits_max,
            Some(1)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_hits_total,
            Some(1)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_misses,
            Some(2)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_misses_max,
            Some(2)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_misses_total,
            Some(2)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_stores,
            Some(3)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_stores_max,
            Some(3)
        );
        assert_eq!(
            aggregate.native_action_callout_batch_warm_cache_stores_total,
            Some(3)
        );
        assert_eq!(
            aggregate.runs_with_native_action_callout_batch_warm_cache_hit,
            1
        );
        assert_eq!(
            aggregate.runs_with_native_action_callout_batch_warm_cache_miss,
            1
        );
        assert_eq!(
            aggregate.runs_with_native_action_callout_batch_warm_cache_store,
            1
        );
        assert_eq!(
            aggregate.native_action_callout_batch_shard_warm_cache_statuses,
            vec![
                "disabled".to_string(),
                "hit".to_string(),
                "miss".to_string()
            ]
        );
        assert_eq!(
            aggregate.native_action_callout_batch_shard_warm_cache_statuses_unique_all_runs,
            vec![
                "disabled".to_string(),
                "hit".to_string(),
                "miss".to_string()
            ]
        );
        assert_eq!(
            aggregate.fallback_reasons,
            vec!["reason A".to_string(), "reason B".to_string()]
        );
    }

    #[test]
    fn flat_layout_accepts_native_fused_flat_frontier_admission() {
        let mut telemetry = native_telemetry(1_000_000_000);
        telemetry.flat_state_primary = Some(false);
        telemetry.trust_cg_native_fused_flat_frontier_admission_active = Some(true);

        let aggregate = TrustCgTelemetryAggregate::from_run_telemetry([&telemetry]).unwrap();
        let evidence = TrustCgEvidence::classify(Some(&aggregate), Some(2.0));

        assert_eq!(evidence.flat_layout, Some(true));

        telemetry.compiled_bfs_flat_frontier_admitted = Some(false);
        let aggregate = TrustCgTelemetryAggregate::from_run_telemetry([&telemetry]).unwrap();
        let evidence = TrustCgEvidence::classify(Some(&aggregate), Some(2.0));

        assert_eq!(evidence.flat_layout, Some(false));
    }

    #[test]
    fn flat_layout_preserves_flat_primary_proof_when_admission_candidate_is_false() {
        let mut telemetry = native_telemetry(1_000_000_000);
        telemetry.trust_cg_native_fused_flat_frontier_admission_active = Some(false);

        let aggregate = TrustCgTelemetryAggregate::from_run_telemetry([&telemetry]).unwrap();
        let evidence = TrustCgEvidence::classify(Some(&aggregate), Some(2.0));

        assert_eq!(evidence.flat_layout, Some(true));
    }

    #[test]
    fn benchmark_summary_serializes_top_level_contract() {
        let row = BenchmarkRow::from_runs(
            "Spec",
            Some(10),
            vec![tlc_run(1, 4.0)],
            vec![ty_run("interp", 1, 2.0, None)],
            vec![ty_run(
                "trust-cg",
                1,
                1.0,
                Some(native_telemetry(1_000_000_000)),
            )],
        );
        let gate_flags = BenchmarkGateFlags::from_names(
            &[
                "require_trust_cg_compiled_actions".to_string(),
                "require_trust_cg_execution_faster_than_tlc".to_string(),
            ],
            &["allow_trust_cg_invariant_rust_fallbacks".to_string()],
        );
        let summary = BenchmarkSummary::new(
            "2026-04-27T120000",
            "abcdef0",
            "reports/perf/example",
            "ty supremacy benchmark --runs 1",
            BenchmarkBuildIdentity::new("release", "target/user/release/ty", "abc123"),
            BackendControls {
                interp_env: BTreeMap::from([("TY_trust_cg".to_string(), "0".to_string())]),
                trust_cg_env: BTreeMap::from([
                    ("TY_trust_cg".to_string(), "1".to_string()),
                    (
                        "TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS".to_string(),
                        "1".to_string(),
                    ),
                    ("TY_DISABLE_ARTIFACT_CACHE".to_string(), "1".to_string()),
                    ("TY_CACHE_DIR".to_string(), "reports/cache".to_string()),
                ]),
            },
            LaunchControls {
                tlc: TlcLaunchControls {
                    workers: 1,
                    jvm_args: vec![
                        "-XX:ActiveProcessorCount=1".to_string(),
                        "-Xmx4g".to_string(),
                    ],
                    heap_xms: None,
                    heap_xmx: Some("4g".to_string()),
                },
                ty: TyLaunchControls {
                    interp: TyModeLaunchControls {
                        workers: 1,
                        cache_dir: None,
                        artifact_cache_disabled_env: None,
                        native_callout_compile_jobs: None,
                    },
                    trust_cg: TyModeLaunchControls {
                        workers: 1,
                        cache_dir: Some("reports/cache".to_string()),
                        artifact_cache_disabled_env: Some("1".to_string()),
                        native_callout_compile_jobs: Some("1".to_string()),
                    },
                },
            },
            gate_flags,
            vec![row],
        );

        let value = serde_json::to_value(summary).unwrap();
        assert_eq!(value["timestamp"], json!("2026-04-27T120000"));
        assert_eq!(value["git_commit"], json!("abcdef0"));
        assert_eq!(value["build_identity"]["cargo_profile"], json!("release"));
        assert_eq!(value["build_identity"]["ty_binary_sha256"], json!("abc123"));
        assert_eq!(
            value["backend_controls"]["trust_cg_env"]["TY_trust_cg"],
            json!("1")
        );
        assert_eq!(
            value["backend_controls"]["interp_env"]["TY_trust_cg"],
            json!("0")
        );
        assert_eq!(value["launch_controls"]["tlc"]["workers"], json!(1));
        assert_eq!(value["launch_controls"]["tlc"]["heap_xmx"], json!("4g"));
        assert_eq!(
            value["launch_controls"]["ty"]["trust_cg"]["cache_dir"],
            json!("reports/cache")
        );
        assert_eq!(
            value["launch_controls"]["ty"]["trust_cg"]["native_callout_compile_jobs"],
            json!("1")
        );
        assert_eq!(
            value["gate_flags"]["require_trust_cg_compiled_actions"],
            json!(true)
        );
        assert_eq!(
            value["gate_flags"]["allow_trust_cg_invariant_rust_fallbacks"],
            json!(false)
        );
        assert_eq!(
            value["rows"][0]["speedup_trust_cg_execution_vs_tlc"],
            json!(4.0)
        );
        assert_eq!(
            value["rows"][0]["tlc"]["median_peak_rss_bytes"],
            json!(8 * 1024 * 1024)
        );
        assert_eq!(
            value["rows"][0]["trust_cg"]["runs"][0]["peak_rss_bytes"],
            json!(4 * 1024 * 1024)
        );
        assert_eq!(
            value["rows"][0]["trust_cg"]["runs"][0]["env_overrides"]["TY_trust_cg"],
            json!("1")
        );
    }
}
