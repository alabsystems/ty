// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Typed policy model for `ty supremacy`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::cli_schema::{SupremacyGateMode, SupremacyMode};

const STRUCTURAL_SELECTION_BASIS: &str = "structural";
const EXACT_SPEC_NAME_ALLOWLIST_INPUT: &str = "exact_spec_name_allowlist";
const ANALYTICAL_FUTURE_ENGINE: &str = "analytical";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct SupremacyPolicy {
    pub(super) specs: Vec<String>,
    #[serde(default)]
    pub(super) engine_selection_contract: Option<EngineSelectionContract>,
    #[serde(default)]
    pub(super) matrix_policy: MatrixPolicy,
    #[serde(default)]
    pub(super) expected_state_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub(super) expected_generated_state_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub(super) required_trust_cg_gate_flags: Vec<String>,
    #[serde(default)]
    pub(super) default_gate_mode: Option<String>,
    #[serde(default)]
    pub(super) final_gate_mode: Option<String>,
    #[serde(default)]
    pub(super) gate_modes: BTreeMap<String, GateModePolicy>,
    #[serde(default)]
    pub(super) thresholds: BTreeMap<String, ThresholdPolicy>,
}

impl SupremacyPolicy {
    pub(super) fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read supremacy policy {}", path.display()))?;
        let policy: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse supremacy policy {}", path.display()))?;
        policy.validate_specs()?;
        policy.validate_engine_selection_contract()?;
        policy.matrix_policy.validate()?;
        Ok(policy)
    }

    pub(super) fn resolve_gate_mode(
        &self,
        run_mode: SupremacyMode,
        requested: Option<SupremacyGateMode>,
    ) -> Result<ResolvedGateMode<'_>> {
        if self.gate_modes.is_empty() {
            let name = requested
                .map(policy_gate_mode_key)
                .or(self.default_gate_mode.as_deref())
                .unwrap_or("legacy");
            return Ok(ResolvedGateMode {
                name,
                policy: None,
                benchmark_flags: self.required_trust_cg_gate_flags.clone(),
            });
        }

        let name = match run_mode {
            SupremacyMode::Enforce => self
                .final_gate_mode
                .as_deref()
                .or(self.default_gate_mode.as_deref())
                .context("policy has gate_modes but no final_gate_mode/default_gate_mode")?,
            SupremacyMode::Warn => requested.map(policy_gate_mode_key).with_context(|| {
                let available = self
                    .gate_modes
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("--gate-mode is required outside --mode enforce; available: {available}")
            })?,
        };
        let policy = self.gate_modes.get(name).with_context(|| {
            let available = self
                .gate_modes
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            format!("unknown gate mode {name:?}; available: {available}")
        })?;
        Ok(ResolvedGateMode {
            name,
            policy: Some(policy),
            benchmark_flags: policy.benchmark_flags.clone(),
        })
    }

    pub(super) fn validate_gate_ready(&self) -> Result<()> {
        self.validate_specs()?;
        self.validate_engine_selection_contract()?;
        if self.specs.is_empty() {
            bail!("supremacy policy must list at least one spec");
        }
        for spec in &self.specs {
            let Some(expected_states) = self.expected_state_counts.get(spec) else {
                bail!("supremacy gate policy missing expected_state_counts[{spec:?}]");
            };
            if *expected_states == 0 {
                bail!("supremacy gate policy expected_state_counts[{spec:?}] must be positive");
            }
            let Some(expected_generated) = self.expected_generated_state_counts.get(spec) else {
                bail!("supremacy gate policy missing expected_generated_state_counts[{spec:?}]");
            };
            if *expected_generated == 0 {
                bail!(
                    "supremacy gate policy expected_generated_state_counts[{spec:?}] must be positive"
                );
            }
            let Some(thresholds) = self.thresholds.get(spec) else {
                bail!("supremacy gate policy missing thresholds[{spec:?}]");
            };
            thresholds.validate(spec)?;
        }
        require_non_empty_strings(
            &self.required_trust_cg_gate_flags,
            "required_trust_cg_gate_flags",
        )?;
        if self.gate_modes.is_empty() {
            bail!("supremacy gate policy must define at least one gate mode");
        }
        for (name, mode) in &self.gate_modes {
            mode.validate(name)?;
        }
        for field in [&self.default_gate_mode, &self.final_gate_mode]
            .into_iter()
            .flatten()
        {
            if !self.gate_modes.contains_key(field) {
                bail!("supremacy gate policy references unknown gate mode {field:?}");
            }
        }
        Ok(())
    }

    fn validate_specs(&self) -> Result<()> {
        if self.specs.is_empty() {
            bail!("supremacy policy must list at least one spec");
        }
        for spec in &self.specs {
            if spec.is_empty() {
                bail!("supremacy policy specs must not contain empty names");
            }
        }
        Ok(())
    }

    fn validate_engine_selection_contract(&self) -> Result<()> {
        if let Some(contract) = &self.engine_selection_contract {
            contract.validate("engine_selection_contract")?;
        }
        Ok(())
    }
}

pub(super) fn load_matrix_policy(path: &Path) -> Result<MatrixPolicy> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read supremacy matrix policy {}", path.display()))?;
    let document: MatrixPolicyDocument = serde_json::from_str(&text)
        .with_context(|| format!("parse supremacy matrix policy {}", path.display()))?;
    document.matrix_policy.validate()?;
    Ok(document.matrix_policy)
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct MatrixPolicyDocument {
    #[serde(default)]
    matrix_policy: MatrixPolicy,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub(super) struct MatrixPolicy {
    #[serde(default)]
    pub(super) allow_runtime_to_error: bool,
    #[serde(default)]
    pub(super) allow_timeout_dominance: bool,
}

impl MatrixPolicy {
    pub(super) fn has_comparable_outcome_opt_in(&self) -> bool {
        self.allow_runtime_to_error || self.allow_timeout_dominance
    }

    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct EngineSelectionContract {
    pub(super) selection_basis: String,
    #[serde(default)]
    pub(super) forbidden_selector_inputs: Vec<String>,
    #[serde(default)]
    pub(super) permitted_future_engines: Vec<String>,
}

impl EngineSelectionContract {
    fn validate(&self, path: &str) -> Result<()> {
        if self.selection_basis.is_empty() {
            bail!("supremacy policy {path}.selection_basis must not be empty");
        }
        if self.selection_basis != STRUCTURAL_SELECTION_BASIS {
            bail!("supremacy policy {path}.selection_basis must be {STRUCTURAL_SELECTION_BASIS:?}");
        }
        if self.forbidden_selector_inputs.is_empty() {
            bail!("supremacy policy {path}.forbidden_selector_inputs must not be empty");
        }
        require_non_empty_strings(
            &self.forbidden_selector_inputs,
            &format!("{path}.forbidden_selector_inputs"),
        )?;
        if !self
            .forbidden_selector_inputs
            .iter()
            .any(|input| input == EXACT_SPEC_NAME_ALLOWLIST_INPUT)
        {
            bail!(
                "supremacy policy {path}.forbidden_selector_inputs must include {EXACT_SPEC_NAME_ALLOWLIST_INPUT:?}"
            );
        }
        if self.permitted_future_engines.is_empty() {
            bail!("supremacy policy {path}.permitted_future_engines must not be empty");
        }
        require_non_empty_strings(
            &self.permitted_future_engines,
            &format!("{path}.permitted_future_engines"),
        )?;
        if !self
            .permitted_future_engines
            .iter()
            .any(|engine| engine == ANALYTICAL_FUTURE_ENGINE)
        {
            bail!(
                "supremacy policy {path}.permitted_future_engines must include {ANALYTICAL_FUTURE_ENGINE:?}"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct GateModePolicy {
    #[serde(default)]
    pub(super) description: Option<String>,
    #[serde(default)]
    pub(super) benchmark_flags: Vec<String>,
    #[serde(default)]
    pub(super) forbidden_benchmark_flags: Vec<String>,
    #[serde(default)]
    pub(super) required_trust_cg_env: BTreeMap<String, String>,
    #[serde(default)]
    pub(super) required_trust_cg_compilation_total_matches: Vec<String>,
    #[serde(default)]
    pub(super) require_generated_state_parity_by_run_index: bool,
    #[serde(default)]
    pub(super) required_trust_cg_run_telemetry: BTreeMap<String, TelemetryRequirement>,
    #[serde(default)]
    pub(super) required_trust_cg_run_telemetry_by_spec:
        BTreeMap<String, BTreeMap<String, TelemetryRequirement>>,
    #[serde(default)]
    pub(super) required_trust_cg_selftest_by_spec: BTreeMap<String, SelftestRequirement>,
}

impl GateModePolicy {
    fn validate(&self, name: &str) -> Result<()> {
        if name.is_empty() {
            bail!("supremacy gate policy gate_modes must not contain an empty mode name");
        }
        if self.benchmark_flags.is_empty() {
            bail!("supremacy gate policy gate_modes.{name}.benchmark_flags must not be empty");
        }
        require_non_empty_strings(
            &self.benchmark_flags,
            &format!("gate_modes.{name}.benchmark_flags"),
        )?;
        require_non_empty_strings(
            &self.forbidden_benchmark_flags,
            &format!("gate_modes.{name}.forbidden_benchmark_flags"),
        )?;
        require_non_empty_strings(
            &self.required_trust_cg_compilation_total_matches,
            &format!("gate_modes.{name}.required_trust_cg_compilation_total_matches"),
        )?;
        require_non_empty_string_map(
            &self.required_trust_cg_env,
            &format!("gate_modes.{name}.required_trust_cg_env"),
        )?;
        for (field, requirement) in &self.required_trust_cg_run_telemetry {
            requirement.validate(&format!(
                "gate_modes.{name}.required_trust_cg_run_telemetry.{field}"
            ))?;
        }
        for (spec, requirements) in &self.required_trust_cg_run_telemetry_by_spec {
            if spec.is_empty() {
                bail!(
                    "supremacy gate policy gate_modes.{name}.required_trust_cg_run_telemetry_by_spec contains empty spec"
                );
            }
            for (field, requirement) in requirements {
                requirement.validate(&format!(
                    "gate_modes.{name}.required_trust_cg_run_telemetry_by_spec.{spec}.{field}"
                ))?;
            }
        }
        for (spec, requirement) in &self.required_trust_cg_selftest_by_spec {
            requirement.validate(&format!(
                "gate_modes.{name}.required_trust_cg_selftest_by_spec.{spec}"
            ))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub(super) enum TelemetryRequirement {
    Bool(bool),
    Integer(i64),
    Text(String),
}

impl TelemetryRequirement {
    fn validate(&self, path: &str) -> Result<()> {
        if let Self::Text(value) = self {
            if value.is_empty() {
                bail!("supremacy gate policy {path} must not be an empty string");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct SelftestRequirement {
    pub(super) actions: u64,
    pub(super) state_constraints: u64,
    pub(super) invariants: u64,
    pub(super) state_len: u64,
}

impl SelftestRequirement {
    fn validate(&self, path: &str) -> Result<()> {
        if self.actions == 0 {
            bail!("supremacy gate policy {path}.actions must be positive");
        }
        if self.state_len == 0 {
            bail!("supremacy gate policy {path}.state_len must be positive");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(super) struct ThresholdPolicy {
    #[serde(default)]
    pub(super) min_speedup_interp_vs_tlc: Option<f64>,
    #[serde(default)]
    pub(super) min_speedup_trust_cg_vs_tlc: Option<f64>,
    #[serde(default)]
    pub(super) min_trust_cg_vs_interp_ratio: Option<f64>,
}

impl ThresholdPolicy {
    fn validate(&self, spec: &str) -> Result<()> {
        validate_positive_threshold(
            self.min_speedup_interp_vs_tlc,
            spec,
            "min_speedup_interp_vs_tlc",
        )?;
        validate_positive_threshold(
            self.min_speedup_trust_cg_vs_tlc,
            spec,
            "min_speedup_trust_cg_vs_tlc",
        )?;
        validate_positive_threshold(
            self.min_trust_cg_vs_interp_ratio,
            spec,
            "min_trust_cg_vs_interp_ratio",
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct PlannedGate {
    pub(super) gate_mode: String,
    pub(super) benchmark_flags: Vec<String>,
    pub(super) forbidden_benchmark_flags: Vec<String>,
    pub(super) required_trust_cg_env: BTreeMap<String, String>,
    pub(super) required_trust_cg_compilation_total_matches: Vec<String>,
    pub(super) require_generated_state_parity_by_run_index: bool,
    pub(super) required_trust_cg_run_telemetry: BTreeMap<String, TelemetryRequirement>,
    pub(super) required_trust_cg_run_telemetry_by_spec:
        BTreeMap<String, BTreeMap<String, TelemetryRequirement>>,
    pub(super) required_trust_cg_selftest_by_spec: BTreeMap<String, SelftestRequirement>,
}

impl PlannedGate {
    pub(super) fn from_resolved(resolved: ResolvedGateMode<'_>) -> Self {
        let Some(policy) = resolved.policy else {
            return Self {
                gate_mode: resolved.name.to_string(),
                benchmark_flags: resolved.benchmark_flags,
                forbidden_benchmark_flags: Vec::new(),
                required_trust_cg_env: BTreeMap::new(),
                required_trust_cg_compilation_total_matches: Vec::new(),
                require_generated_state_parity_by_run_index: false,
                required_trust_cg_run_telemetry: BTreeMap::new(),
                required_trust_cg_run_telemetry_by_spec: BTreeMap::new(),
                required_trust_cg_selftest_by_spec: BTreeMap::new(),
            };
        };
        Self {
            gate_mode: resolved.name.to_string(),
            benchmark_flags: resolved.benchmark_flags,
            forbidden_benchmark_flags: policy.forbidden_benchmark_flags.clone(),
            required_trust_cg_env: policy.required_trust_cg_env.clone(),
            required_trust_cg_compilation_total_matches: policy
                .required_trust_cg_compilation_total_matches
                .clone(),
            require_generated_state_parity_by_run_index: policy
                .require_generated_state_parity_by_run_index,
            required_trust_cg_run_telemetry: policy.required_trust_cg_run_telemetry.clone(),
            required_trust_cg_run_telemetry_by_spec: policy
                .required_trust_cg_run_telemetry_by_spec
                .clone(),
            required_trust_cg_selftest_by_spec: policy.required_trust_cg_selftest_by_spec.clone(),
        }
    }

    pub(super) fn enforce_required_env(&self) -> BTreeMap<String, String> {
        let mut required = BTreeMap::new();
        if self.gate_mode == "full_native_fused" {
            required.extend(full_native_fused_protected_env());
        }
        required.extend(self.required_trust_cg_env.clone());
        required
    }

    pub(super) fn unexpected_enforce_env_keys<'a>(
        &self,
        keys: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let required = self.enforce_required_env();
        keys.into_iter()
            .filter(|key| !required.contains_key(*key) && env_key_requires_gate_policy_control(key))
            .map(str::to_string)
            .collect()
    }
}

fn env_key_requires_gate_policy_control(key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    // TY_CACHE_DIR only redirects *where* the artifact cache lives; it is a
    // directory path, not a behavioral switch. The benchmark runner injects it
    // on every trust-cg gate run (benchmark.rs), while caching itself is
    // force-disabled by the protected control TY_DISABLE_ARTIFACT_CACHE=1. A
    // pointer to an already-disabled, per-run-fresh cache dir cannot influence
    // engine selection or state counts, so it is not gate-control surface — yet
    // it would otherwise be flagged via the "CACHE" substring below, making the
    // harness inject a key its own verdict rejects. Exempt it explicitly.
    if upper == "TY_CACHE_DIR" {
        return false;
    }
    upper.starts_with("TY_")
        || upper.contains("INVERSE")
        || upper.contains("DISABLE")
        || upper.contains("ENABLE")
        || upper.contains("CACHE")
        || upper.contains("PROFILE")
}

pub(super) struct ResolvedGateMode<'a> {
    pub(super) name: &'a str,
    pub(super) policy: Option<&'a GateModePolicy>,
    pub(super) benchmark_flags: Vec<String>,
}

pub(super) fn policy_gate_mode_key(mode: SupremacyGateMode) -> &'static str {
    match mode {
        SupremacyGateMode::InterimActionOnlyNativeFused => "interim_action_only_native_fused",
        SupremacyGateMode::FullNativeFused => "full_native_fused",
    }
}

// Only exercised by unit tests (maps_cli_gate_modes_to_policy_keys); kept as a
// counterpart to policy_gate_mode_key for the CLI-facing name.
#[allow(dead_code)]
pub(super) fn cli_gate_mode_name(mode: SupremacyGateMode) -> &'static str {
    match mode {
        SupremacyGateMode::InterimActionOnlyNativeFused => "interim-action-only-native-fused",
        SupremacyGateMode::FullNativeFused => "full-native-fused",
    }
}

pub(super) fn full_native_fused_protected_env() -> BTreeMap<String, String> {
    // Auto-POR/auto-symmetry are NOT env pins any more: those semantic levers
    // are controlled by CLI flags only (`--no-reduction` in the child argv);
    // the child `ty check` ignores ambient TY_AUTO_POR / TY_AUTO_SYMMETRY.
    [
        ("TY_trust_cg", "1"),
        ("TY_TRUST_CG_BFS", "1"),
        ("TY_TRUST_CG_EXISTS", "1"),
        ("TY_BYTECODE_VM", "1"),
        ("TY_BYTECODE_VM_STATS", "1"),
        ("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST", "strict"),
        ("TY_TRUST_CG_NATIVE_FUSED_STRICT", "1"),
        ("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS", "27"),
        ("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP", "1"),
        ("TY_DISABLE_ARTIFACT_CACHE", "1"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect()
}

fn require_non_empty_strings(values: &[String], path: &str) -> Result<()> {
    if let Some(value) = values.iter().find(|value| value.is_empty()) {
        bail!("supremacy policy {path} contains empty string {value:?}");
    }
    Ok(())
}

fn require_non_empty_string_map(values: &BTreeMap<String, String>, path: &str) -> Result<()> {
    for (key, value) in values {
        if key.is_empty() {
            bail!("supremacy policy {path} contains empty key");
        }
        if value.is_empty() {
            bail!("supremacy policy {path}.{key} contains empty value");
        }
    }
    Ok(())
}

fn validate_positive_threshold(value: Option<f64>, spec: &str, field: &str) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || value <= 0.0 {
        bail!("supremacy gate policy thresholds[{spec:?}].{field} must be positive");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_policy_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/tlc_comparison/single_thread_supremacy_gate.json")
    }

    fn load_policy_text(text: &str) -> Result<SupremacyPolicy> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("policy.json");
        fs::write(&path, text).unwrap();
        SupremacyPolicy::load(&path)
    }

    #[test]
    fn parses_policy_gate_modes() {
        let policy = SupremacyPolicy::load(&repo_policy_path()).unwrap();
        policy.validate_gate_ready().unwrap();

        let full = policy
            .resolve_gate_mode(
                SupremacyMode::Enforce,
                Some(SupremacyGateMode::InterimActionOnlyNativeFused),
            )
            .unwrap();
        assert_eq!(full.name, "full_native_fused");
        assert!(full
            .benchmark_flags
            .contains(&"require_trust_cg_compiled_invariants".to_string()));

        let planned = PlannedGate::from_resolved(full);
        assert_eq!(
            planned
                .required_trust_cg_env
                .get("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST")
                .map(String::as_str),
            Some("strict")
        );
        let enforced = planned.enforce_required_env();
        assert_eq!(
            enforced
                .get("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS")
                .map(String::as_str),
            Some("27")
        );
        assert_eq!(
            enforced
                .get("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP")
                .map(String::as_str),
            Some("1")
        );
        assert!(!enforced.contains_key("TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP"));
        assert!(planned.require_generated_state_parity_by_run_index);
        assert!(planned
            .benchmark_flags
            .contains(&"require_native_fused_flat_frontier_admission".to_string()));
        assert!(planned
            .required_trust_cg_run_telemetry
            .contains_key("compiled_bfs_execution_nanos"));
        assert!(planned
            .required_trust_cg_run_telemetry
            .contains_key("compiled_bfs_flat_frontier_admitted"));
        assert!(planned
            .required_trust_cg_selftest_by_spec
            .contains_key("MCLamportMutex"));
    }

    #[test]
    fn parses_engine_selection_contract() {
        let policy = SupremacyPolicy::load(&repo_policy_path()).unwrap();

        let contract = policy.engine_selection_contract.as_ref().unwrap();
        assert_eq!(contract.selection_basis, STRUCTURAL_SELECTION_BASIS);
        assert!(contract
            .forbidden_selector_inputs
            .contains(&EXACT_SPEC_NAME_ALLOWLIST_INPUT.to_string()));
        assert!(contract
            .permitted_future_engines
            .contains(&ANALYTICAL_FUTURE_ENGINE.to_string()));
    }

    #[test]
    fn engine_selection_contract_requires_structural_selection() {
        let err = load_policy_text(
            r#"{
              "specs": ["TinySpec"],
              "engine_selection_contract": {
                "selection_basis": "exact_spec_name_allowlist",
                "forbidden_selector_inputs": ["exact_spec_name_allowlist"],
                "permitted_future_engines": ["analytical"]
              }
            }"#,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("engine_selection_contract.selection_basis must be \"structural\""));
    }

    #[test]
    fn engine_selection_contract_forbids_exact_spec_name_allowlist() {
        let err = load_policy_text(
            r#"{
              "specs": ["TinySpec"],
              "engine_selection_contract": {
                "selection_basis": "structural",
                "forbidden_selector_inputs": ["manual_override"],
                "permitted_future_engines": ["analytical"]
              }
            }"#,
        )
        .unwrap_err();

        assert!(err.to_string().contains(
            "engine_selection_contract.forbidden_selector_inputs must include \"exact_spec_name_allowlist\""
        ));
    }

    #[test]
    fn engine_selection_contract_permits_analytical_future_engines() {
        let err = load_policy_text(
            r#"{
              "specs": ["TinySpec"],
              "engine_selection_contract": {
                "selection_basis": "structural",
                "forbidden_selector_inputs": ["exact_spec_name_allowlist"],
                "permitted_future_engines": ["symbolic"]
              }
            }"#,
        )
        .unwrap_err();

        assert!(err.to_string().contains(
            "engine_selection_contract.permitted_future_engines must include \"analytical\""
        ));
    }

    #[test]
    fn engine_selection_contract_rejects_empty_values() {
        let err = load_policy_text(
            r#"{
              "specs": ["TinySpec"],
              "engine_selection_contract": {
                "selection_basis": "structural",
                "forbidden_selector_inputs": ["exact_spec_name_allowlist"],
                "permitted_future_engines": [""]
              }
            }"#,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("engine_selection_contract.permitted_future_engines contains empty string"));
    }

    #[test]
    fn ty_cache_dir_is_exempt_but_other_cache_keys_are_controlled() {
        // TY_CACHE_DIR is a harness-injected, inert artifact-cache path and must
        // not be flagged as gate-control surface (otherwise the gate is forever
        // red since benchmark.rs always injects it).
        assert!(!env_key_requires_gate_policy_control("TY_CACHE_DIR"));
        // The exemption must be exact: anything else that looks like a cache or
        // gate control is still policed, so the anti-gaming guard stays intact.
        assert!(env_key_requires_gate_policy_control("TY_CACHE_DIR_EXTRA"));
        assert!(env_key_requires_gate_policy_control(
            "TY_DISABLE_ARTIFACT_CACHE"
        ));
        assert!(env_key_requires_gate_policy_control("SOME_CACHE_KNOB"));
        assert!(env_key_requires_gate_policy_control("TY_TRUST_CG_BFS"));
    }

    #[test]
    fn ty_cache_dir_not_flagged_as_unexpected_enforce_env_key() {
        // End-to-end: a trust-cg env that matches the protected controls exactly
        // *plus* the harness-injected TY_CACHE_DIR must report zero unexpected
        // keys (this is exactly the env the benchmark runner produces).
        let planned = PlannedGate {
            gate_mode: "full_native_fused".to_string(),
            benchmark_flags: Vec::new(),
            forbidden_benchmark_flags: Vec::new(),
            required_trust_cg_env: BTreeMap::new(),
            required_trust_cg_compilation_total_matches: Vec::new(),
            require_generated_state_parity_by_run_index: false,
            required_trust_cg_run_telemetry: BTreeMap::new(),
            required_trust_cg_run_telemetry_by_spec: BTreeMap::new(),
            required_trust_cg_selftest_by_spec: BTreeMap::new(),
        };
        let mut keys: Vec<String> = planned.enforce_required_env().keys().cloned().collect();
        keys.push("TY_CACHE_DIR".to_string());
        let unexpected = planned.unexpected_enforce_env_keys(keys.iter().map(String::as_str));
        assert!(
            unexpected.is_empty(),
            "TY_CACHE_DIR should not be flagged as unexpected, got: {unexpected:?}"
        );
    }

    #[test]
    fn loads_minimal_policy_for_smoke_specs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("policy.json");
        fs::write(&path, r#"{"specs":["TinySpec"]}"#).unwrap();

        let policy = SupremacyPolicy::load(&path).unwrap();

        assert_eq!(policy.specs, vec!["TinySpec".to_string()]);
        assert!(policy.validate_gate_ready().is_err());
    }

    #[test]
    fn loads_minimal_matrix_policy_opt_ins() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("matrix-policy.json");
        fs::write(
            &path,
            r#"{
              "matrix_policy": {
                "allow_runtime_to_error": true,
                "allow_timeout_dominance": true
              }
            }"#,
        )
        .unwrap();

        let policy = load_matrix_policy(&path).unwrap();

        assert!(policy.allow_runtime_to_error);
        assert!(policy.allow_timeout_dominance);
        assert!(policy.has_comparable_outcome_opt_in());
    }

    #[test]
    fn missing_matrix_policy_keeps_strict_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("matrix-policy.json");
        fs::write(&path, r#"{"specs":["TinySpec"]}"#).unwrap();

        let policy = load_matrix_policy(&path).unwrap();

        assert!(!policy.allow_runtime_to_error);
        assert!(!policy.allow_timeout_dominance);
        assert!(!policy.has_comparable_outcome_opt_in());
    }

    #[test]
    fn maps_cli_gate_modes_to_policy_keys() {
        assert_eq!(
            policy_gate_mode_key(SupremacyGateMode::FullNativeFused),
            "full_native_fused"
        );
        assert_eq!(
            cli_gate_mode_name(SupremacyGateMode::FullNativeFused),
            "full-native-fused"
        );
    }

    #[test]
    fn warn_mode_requires_explicit_gate_mode_choice() {
        let policy = load_policy_text(
            r#"{
              "specs": ["TinySpec"],
              "default_gate_mode": "full_native_fused",
              "final_gate_mode": "full_native_fused",
              "gate_modes": {
                "full_native_fused": {
                  "benchmark_flags": ["require_trust_cg_native_fused_level"]
                }
              }
            }"#,
        )
        .unwrap();

        let err = match policy.resolve_gate_mode(SupremacyMode::Warn, None) {
            Ok(_) => panic!("warn mode must require an explicit --gate-mode"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(
            message.contains("--gate-mode is required outside --mode enforce"),
            "{message}"
        );
        assert!(message.contains("full_native_fused"), "{message}");
    }

    #[test]
    fn enforce_required_env_includes_wrapper_protected_controls() {
        let planned = PlannedGate {
            gate_mode: "full_native_fused".to_string(),
            benchmark_flags: Vec::new(),
            forbidden_benchmark_flags: Vec::new(),
            required_trust_cg_env: BTreeMap::new(),
            required_trust_cg_compilation_total_matches: Vec::new(),
            require_generated_state_parity_by_run_index: false,
            required_trust_cg_run_telemetry: BTreeMap::new(),
            required_trust_cg_run_telemetry_by_spec: BTreeMap::new(),
            required_trust_cg_selftest_by_spec: BTreeMap::new(),
        };

        let required = planned.enforce_required_env();
        assert_eq!(required.get("TY_trust_cg").map(String::as_str), Some("1"));
        assert_eq!(
            required
                .get("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS")
                .map(String::as_str),
            Some("27")
        );
        assert_eq!(
            required
                .get("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP")
                .map(String::as_str),
            Some("1")
        );
        assert!(!required.contains_key("TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP"));
    }
}
