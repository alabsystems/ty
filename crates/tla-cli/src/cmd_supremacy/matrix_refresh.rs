// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Source-path planning for all-runnable matrix runtime refresh.
//!
//! This is intentionally only the source/readiness layer. CLI execution can
//! call this planner before running TLC/trust_cg subprocesses.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::matrix::{self, SupremacyMatrixClass};

const MATRIX_REFRESH_PLAN_SCHEMA: &str = "ty.supremacy.matrix_refresh_plan.v1";

pub(super) const NO_CONFIG_INIT: &str = "MyInit";
pub(super) const NO_CONFIG_NEXT: &str = "MyNext";
pub(super) const NO_CONFIG_INVARIANT: &str = "TypeOK";
pub(super) const NO_CONFIG_CLI_FLAGS: &[&str] = &[
    "--no-config",
    "--init",
    NO_CONFIG_INIT,
    "--next",
    NO_CONFIG_NEXT,
    "--inv",
    NO_CONFIG_INVARIANT,
];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct MatrixRefreshPlan {
    pub(super) schema: &'static str,
    #[serde(skip_serializing_if = "MatrixRefreshScope::is_missing_runtime")]
    pub(super) scope: MatrixRefreshScope,
    pub(super) baseline_path: PathBuf,
    pub(super) examples_dir: PathBuf,
    pub(super) counts: MatrixRefreshPlanCounts,
    pub(super) can_batch_all_selected_runtime_rows: bool,
    pub(super) can_batch_all_missing_runtime_rows: bool,
    pub(super) batchable_runtime_specs: Vec<String>,
    pub(super) blocked_runtime_specs: Vec<String>,
    pub(super) batchable_runtime_spec_args: Vec<String>,
    pub(super) rows: Vec<MatrixRefreshRow>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MatrixRefreshScope {
    MissingRuntime,
    AllRunnable,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(super) struct MatrixRefreshPlanCounts {
    pub(super) total_runtime_rows: usize,
    pub(super) total_missing_runtime: usize,
    pub(super) batchable_runtime_rows: usize,
    pub(super) blocked_runtime_rows: usize,
    pub(super) runnable_with_config: usize,
    pub(super) examples_relative_paths: usize,
    pub(super) absolute_source_paths: usize,
    pub(super) missing_source_metadata: usize,
    pub(super) missing_source_files: usize,
    pub(super) no_config_cli_flags: usize,
    pub(super) tlc_runtime_missing: usize,
    pub(super) ty_runtime_missing: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct MatrixRefreshRow {
    pub(super) spec: String,
    #[serde(skip_serializing_if = "is_missing_runtime_class")]
    pub(super) class: SupremacyMatrixClass,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) category: Option<String>,
    pub(super) source: MatrixRefreshSource,
    pub(super) readiness: MatrixRefreshReadiness,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tlc_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ty_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expected_tlc_states: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expected_ty_states: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(super) struct MatrixRefreshSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tla_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cfg_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) resolved_tla_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) resolved_cfg_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) no_config_cli_flags: Option<Vec<String>>,
    pub(super) path_kind: MatrixRefreshPathKind,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MatrixRefreshPathKind {
    #[default]
    Missing,
    ExamplesRelative,
    Absolute,
    Mixed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(super) enum MatrixRefreshReadiness {
    RunnableWithConfig,
    MissingSourceMetadata { reason: String },
    MissingSourceFiles { missing: Vec<PathBuf> },
    NeedsNoConfigCliFlags { reason: String, flags: Vec<String> },
}

pub(super) fn plan_missing_runtime_refresh_from_path(
    baseline_path: &Path,
    examples_dir_override: Option<&Path>,
) -> Result<MatrixRefreshPlan> {
    plan_runtime_refresh_from_path(
        baseline_path,
        examples_dir_override,
        MatrixRefreshScope::MissingRuntime,
    )
}

pub(super) fn plan_runtime_refresh_from_path(
    baseline_path: &Path,
    examples_dir_override: Option<&Path>,
    scope: MatrixRefreshScope,
) -> Result<MatrixRefreshPlan> {
    let text = fs::read_to_string(baseline_path)
        .with_context(|| format!("read baseline {}", baseline_path.display()))?;
    plan_runtime_refresh_str(&text, baseline_path, examples_dir_override, scope).with_context(
        || {
            format!(
                "plan {scope:?} matrix runtime refresh for {}",
                baseline_path.display()
            )
        },
    )
}

pub(super) fn plan_missing_runtime_refresh_str(
    baseline_text: &str,
    baseline_path: &Path,
    examples_dir_override: Option<&Path>,
) -> Result<MatrixRefreshPlan> {
    plan_runtime_refresh_str(
        baseline_text,
        baseline_path,
        examples_dir_override,
        MatrixRefreshScope::MissingRuntime,
    )
}

pub(super) fn plan_runtime_refresh_str(
    baseline_text: &str,
    baseline_path: &Path,
    examples_dir_override: Option<&Path>,
    scope: MatrixRefreshScope,
) -> Result<MatrixRefreshPlan> {
    let baseline: RefreshBaseline = serde_json::from_str(baseline_text)
        .with_context(|| format!("parse baseline {}", baseline_path.display()))?;
    let summary = matrix::classify_baseline_str(baseline_text)
        .with_context(|| format!("classify baseline {}", baseline_path.display()))?;
    let examples_dir = examples_dir_override
        .map(Path::to_path_buf)
        .or_else(|| baseline.inputs.and_then(|inputs| inputs.examples_dir))
        .context("matrix refresh requires an examples_dir from baseline inputs or override")?;

    let mut rows = Vec::new();
    for row in summary
        .rows
        .iter()
        .filter(|row| class_in_scope(scope, row.class))
    {
        let entry = baseline
            .specs
            .get(&row.spec)
            .with_context(|| format!("missing baseline entry for {}", row.spec))?;
        rows.push(plan_row(&row.spec, row.class, entry, &examples_dir));
    }

    let counts = MatrixRefreshPlanCounts::from_rows(&rows);
    let batchable_runtime_specs = batchable_specs_from_rows(&rows);
    let blocked_runtime_specs = blocked_specs_from_rows(&rows);
    let batchable_runtime_spec_args = runtime_spec_args(&batchable_runtime_specs);
    let can_batch_all_missing_runtime_rows = can_batch_all_missing_runtime_rows(&rows);
    let can_batch_all_selected_runtime_rows = counts.blocked_runtime_rows == 0;
    Ok(MatrixRefreshPlan {
        schema: MATRIX_REFRESH_PLAN_SCHEMA,
        scope,
        baseline_path: baseline_path.to_path_buf(),
        examples_dir,
        counts,
        can_batch_all_selected_runtime_rows,
        can_batch_all_missing_runtime_rows,
        batchable_runtime_specs,
        blocked_runtime_specs,
        batchable_runtime_spec_args,
        rows,
    })
}

impl MatrixRefreshScope {
    fn is_missing_runtime(&self) -> bool {
        *self == Self::MissingRuntime
    }
}

fn class_in_scope(scope: MatrixRefreshScope, class: SupremacyMatrixClass) -> bool {
    match scope {
        MatrixRefreshScope::MissingRuntime => class == SupremacyMatrixClass::MissingRuntime,
        MatrixRefreshScope::AllRunnable => is_refreshable_runnable_class(class),
    }
}

fn is_refreshable_runnable_class(class: SupremacyMatrixClass) -> bool {
    class != SupremacyMatrixClass::Unsupported
}

fn is_missing_runtime_class(class: &SupremacyMatrixClass) -> bool {
    *class == SupremacyMatrixClass::MissingRuntime
}

impl MatrixRefreshPlanCounts {
    fn from_rows(rows: &[MatrixRefreshRow]) -> Self {
        let mut counts = Self {
            total_runtime_rows: rows.len(),
            total_missing_runtime: rows
                .iter()
                .filter(|row| row.class == SupremacyMatrixClass::MissingRuntime)
                .count(),
            ..Self::default()
        };
        for row in rows {
            if row.readiness.is_batchable() {
                counts.batchable_runtime_rows += 1;
            } else {
                counts.blocked_runtime_rows += 1;
            }
            match &row.readiness {
                MatrixRefreshReadiness::RunnableWithConfig => counts.runnable_with_config += 1,
                MatrixRefreshReadiness::MissingSourceMetadata { .. } => {
                    counts.missing_source_metadata += 1;
                }
                MatrixRefreshReadiness::MissingSourceFiles { .. } => {
                    counts.missing_source_files += 1;
                }
                MatrixRefreshReadiness::NeedsNoConfigCliFlags { .. } => {
                    counts.no_config_cli_flags += 1;
                }
            }
            match row.source.path_kind {
                MatrixRefreshPathKind::ExamplesRelative => counts.examples_relative_paths += 1,
                MatrixRefreshPathKind::Absolute => counts.absolute_source_paths += 1,
                MatrixRefreshPathKind::Missing | MatrixRefreshPathKind::Mixed => {}
            }
            if !finite_positive(row.tlc_seconds) {
                counts.tlc_runtime_missing += 1;
            }
            if !finite_positive(row.ty_seconds) {
                counts.ty_runtime_missing += 1;
            }
        }
        counts
    }
}

fn can_batch_all_missing_runtime_rows(rows: &[MatrixRefreshRow]) -> bool {
    rows.iter()
        .filter(|row| row.class == SupremacyMatrixClass::MissingRuntime)
        .all(|row| row.readiness.is_batchable())
}

impl MatrixRefreshPlan {
    pub(super) fn row(&self, spec: &str) -> Option<&MatrixRefreshRow> {
        self.rows.iter().find(|row| row.spec == spec)
    }

    pub(super) fn batchable_specs(&self) -> Vec<&str> {
        self.batchable_runtime_specs
            .iter()
            .map(String::as_str)
            .collect()
    }

    pub(super) fn batchable_specs_limited(&self, limit: Option<usize>) -> Vec<String> {
        let limit = limit.unwrap_or(self.batchable_runtime_specs.len());
        self.batchable_runtime_specs
            .iter()
            .take(limit)
            .cloned()
            .collect()
    }

    pub(super) fn skipped_batchable_specs_by_limit(&self, limit: Option<usize>) -> Vec<String> {
        let Some(limit) = limit else {
            return Vec::new();
        };
        self.batchable_runtime_specs
            .iter()
            .skip(limit)
            .cloned()
            .collect()
    }

    pub(super) fn blocked_specs(&self) -> Vec<&str> {
        self.blocked_runtime_specs
            .iter()
            .map(String::as_str)
            .collect()
    }

    pub(super) fn runtime_spec_args_for_batchable_rows(&self) -> Vec<String> {
        self.batchable_runtime_spec_args.clone()
    }

    pub(super) fn can_batch_all_missing_runtime_rows(&self) -> bool {
        self.can_batch_all_missing_runtime_rows
    }

    pub(super) fn can_batch_all_selected_runtime_rows(&self) -> bool {
        self.can_batch_all_selected_runtime_rows
    }
}

impl MatrixRefreshReadiness {
    fn is_batchable(&self) -> bool {
        matches!(
            self,
            MatrixRefreshReadiness::RunnableWithConfig
                | MatrixRefreshReadiness::NeedsNoConfigCliFlags { .. }
        )
    }
}

fn plan_row(
    spec: &str,
    class: SupremacyMatrixClass,
    entry: &RefreshBaselineSpec,
    examples_dir: &Path,
) -> MatrixRefreshRow {
    let source = MatrixRefreshSource::from_entry(entry, examples_dir);
    let readiness = refresh_readiness(&source);
    MatrixRefreshRow {
        spec: spec.to_string(),
        class,
        category: entry.category.clone(),
        source,
        readiness,
        tlc_seconds: entry.tlc.runtime_seconds,
        ty_seconds: entry.ty.runtime_seconds,
        expected_tlc_states: entry.tlc.states,
        expected_ty_states: entry.ty.states,
    }
}

fn batchable_specs_from_rows(rows: &[MatrixRefreshRow]) -> Vec<String> {
    rows.iter()
        .filter(|row| row.readiness.is_batchable())
        .map(|row| row.spec.clone())
        .collect()
}

fn blocked_specs_from_rows(rows: &[MatrixRefreshRow]) -> Vec<String> {
    rows.iter()
        .filter(|row| !row.readiness.is_batchable())
        .map(|row| row.spec.clone())
        .collect()
}

fn runtime_spec_args(specs: &[String]) -> Vec<String> {
    specs
        .iter()
        .flat_map(|spec| ["--runtime-spec".to_string(), spec.clone()])
        .collect()
}

impl MatrixRefreshSource {
    fn from_entry(entry: &RefreshBaselineSpec, examples_dir: &Path) -> Self {
        let Some(source) = entry.source.as_ref() else {
            return Self::default();
        };
        let resolved_tla_path = source
            .tla_path
            .as_ref()
            .map(|path| resolve_baseline_path(path, examples_dir));
        let resolved_cfg_path = source
            .cfg_path
            .as_ref()
            .map(|path| resolve_baseline_path(path, examples_dir));
        let path_kind = path_kind(source.tla_path.as_ref(), source.cfg_path.as_ref());
        let no_config_cli_flags = no_config_cli_flags_from_entry(entry);
        Self {
            mode: source.mode.clone(),
            tla_path: source.tla_path.clone(),
            cfg_path: source.cfg_path.clone(),
            resolved_tla_path,
            resolved_cfg_path,
            no_config_cli_flags,
            path_kind,
        }
    }
}

fn refresh_readiness(source: &MatrixRefreshSource) -> MatrixRefreshReadiness {
    let Some(tla_path) = &source.resolved_tla_path else {
        return MatrixRefreshReadiness::MissingSourceMetadata {
            reason: "baseline row lacks source.tla_path".to_string(),
        };
    };

    if let Some(readiness) = no_config_readiness(source, tla_path) {
        return readiness;
    }

    let Some(cfg_path) = &source.resolved_cfg_path else {
        return MatrixRefreshReadiness::MissingSourceMetadata {
            reason: "baseline row lacks source.cfg_path and is not a recognized config-free row"
                .to_string(),
        };
    };

    let missing = [tla_path, cfg_path]
        .into_iter()
        .filter(|path| !path.is_file())
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        MatrixRefreshReadiness::RunnableWithConfig
    } else {
        MatrixRefreshReadiness::MissingSourceFiles { missing }
    }
}

fn no_config_readiness(
    source: &MatrixRefreshSource,
    tla_path: &Path,
) -> Option<MatrixRefreshReadiness> {
    if !tla_path.is_file() {
        return None;
    }

    if let Some(flags) = source
        .no_config_cli_flags
        .as_ref()
        .filter(|flags| source_text_supports_no_config_cli_flags(tla_path, flags))
    {
        let reason =
            no_config_readiness_reason(source, "source metadata supplies config-free CLI flags");
        return Some(MatrixRefreshReadiness::NeedsNoConfigCliFlags {
            reason,
            flags: flags.clone(),
        });
    }

    let source_mode_is_config_free = source
        .mode
        .as_deref()
        .is_some_and(is_config_free_source_mode);
    if !source_mode_is_config_free
        && source
            .resolved_cfg_path
            .as_ref()
            .is_some_and(|cfg_path| cfg_path.is_file())
    {
        return None;
    }

    let flags = inferred_no_config_cli_flags(source, tla_path)?;

    let reason = no_config_readiness_reason(source, "source declares config-free entry points");
    Some(MatrixRefreshReadiness::NeedsNoConfigCliFlags { reason, flags })
}

fn no_config_readiness_reason(source: &MatrixRefreshSource, prefix: &str) -> String {
    match &source.resolved_cfg_path {
        Some(cfg_path) if cfg_path.is_file() => format!(
            "{prefix}; source.cfg_path {} exists but source is config-free",
            cfg_path.display()
        ),
        Some(cfg_path) => format!("{prefix} and {} is not present", cfg_path.display()),
        None => format!("{prefix} and baseline row has no source.cfg_path"),
    }
}

fn no_config_cli_flags_from_entry(entry: &RefreshBaselineSpec) -> Option<Vec<String>> {
    let mut candidates = Vec::new();
    if let Some(source) = entry.source.as_ref() {
        append_flag_candidates_from_extra(&mut candidates, &source.extra);
    }
    append_flag_candidates_from_extra(&mut candidates, &entry.extra);
    append_flag_candidates_from_extra(&mut candidates, &entry.ty.extra);

    candidates
        .iter()
        .find_map(|flags| normalize_no_config_cli_flags(flags))
}

fn append_flag_candidates_from_extra(
    candidates: &mut Vec<Vec<String>>,
    extra: &BTreeMap<String, Value>,
) {
    const FLAG_KEYS: &[&str] = &[
        "check_flags",
        "required_flags",
        "run_flags",
        "cli_flags",
        "ty_flags",
        "check_args",
        "run_args",
        "ty_args",
        "flags",
        "args",
        "argv",
    ];
    const OBJECT_KEYS: &[&str] = &["check", "run", "command", "ty_check", "ty_run"];

    for key in FLAG_KEYS {
        if let Some(flags) = extra.get(*key).and_then(flags_from_json_value) {
            candidates.push(flags);
        }
    }
    for key in OBJECT_KEYS {
        let Some(Value::Object(object)) = extra.get(*key) else {
            continue;
        };
        for flag_key in FLAG_KEYS {
            if let Some(flags) = object.get(*flag_key).and_then(flags_from_json_value) {
                candidates.push(flags);
            }
        }
    }
}

fn flags_from_json_value(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::Array(values) => values
            .iter()
            .map(|value| value.as_str().map(str::to_string))
            .collect(),
        Value::String(text) => Some(
            text.split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>(),
        )
        .filter(|flags| !flags.is_empty()),
        _ => None,
    }
}

fn normalize_no_config_cli_flags(flags: &[String]) -> Option<Vec<String>> {
    let mut has_no_config = false;
    let mut has_config = false;
    let mut init = None;
    let mut next = None;
    let mut invariants = Vec::new();

    let mut index = 0;
    while index < flags.len() {
        let flag = &flags[index];
        match flag.as_str() {
            "--no-config" => has_no_config = true,
            "--config" => {
                has_config = true;
                index += 1;
            }
            "--init" => {
                index += 1;
                init = flags
                    .get(index)
                    .filter(|value| !value.starts_with("--"))
                    .cloned();
            }
            "--next" => {
                index += 1;
                next = flags
                    .get(index)
                    .filter(|value| !value.starts_with("--"))
                    .cloned();
            }
            "--inv" => {
                index += 1;
                if let Some(value) = flags.get(index).filter(|value| !value.starts_with("--")) {
                    invariants.push(value.clone());
                }
            }
            _ if flag.starts_with("--config=") => has_config = true,
            _ => {
                if let Some(value) = flag.strip_prefix("--init=") {
                    init = Some(value.to_string());
                } else if let Some(value) = flag.strip_prefix("--next=") {
                    next = Some(value.to_string());
                } else if let Some(value) = flag.strip_prefix("--inv=") {
                    invariants.push(value.to_string());
                }
            }
        }
        index += 1;
    }

    if !has_no_config || has_config || init.is_none() || next.is_none() {
        return None;
    }

    let mut normalized = vec![
        "--no-config".to_string(),
        "--init".to_string(),
        init.unwrap(),
        "--next".to_string(),
        next.unwrap(),
    ];
    for invariant in invariants {
        normalized.extend(["--inv".to_string(), invariant]);
    }
    Some(normalized)
}

fn source_text_supports_no_config_cli_flags(tla_path: &Path, flags: &[String]) -> bool {
    let Some(entrypoints) = no_config_entrypoints_from_flags(flags) else {
        return false;
    };
    let Ok(text) = fs::read_to_string(tla_path) else {
        return false;
    };
    source_text_defines_operator(&text, &entrypoints.init)
        && source_text_defines_operator(&text, &entrypoints.next)
        && entrypoints
            .invariants
            .iter()
            .all(|invariant| source_text_defines_operator(&text, invariant))
}

fn no_config_entrypoints_from_flags(flags: &[String]) -> Option<NoConfigEntryPoints> {
    let normalized = normalize_no_config_cli_flags(flags)?;
    let mut init = None;
    let mut next = None;
    let mut invariants = Vec::new();

    let mut index = 0;
    while index < normalized.len() {
        match normalized[index].as_str() {
            "--init" => {
                index += 1;
                init = normalized.get(index).cloned();
            }
            "--next" => {
                index += 1;
                next = normalized.get(index).cloned();
            }
            "--inv" => {
                index += 1;
                if let Some(invariant) = normalized.get(index) {
                    invariants.push(invariant.clone());
                }
            }
            _ => {}
        }
        index += 1;
    }
    Some(NoConfigEntryPoints {
        init: init?,
        next: next?,
        invariants,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NoConfigEntryPoints {
    init: String,
    next: String,
    invariants: Vec<String>,
}

fn inferred_no_config_cli_flags(
    source: &MatrixRefreshSource,
    tla_path: &Path,
) -> Option<Vec<String>> {
    let Ok(text) = fs::read_to_string(tla_path) else {
        return None;
    };
    let source_mode_is_config_free = source
        .mode
        .as_deref()
        .is_some_and(is_config_free_source_mode);
    source_text_inferred_no_config_cli_flags(&text, source_mode_is_config_free)
}

fn is_config_free_source_mode(mode: &str) -> bool {
    matches!(
        mode,
        "no_config" | "no-config" | "config_free" | "config-free"
    )
}

fn source_text_inferred_no_config_cli_flags(
    text: &str,
    source_mode_is_config_free: bool,
) -> Option<Vec<String>> {
    let has_cli_hint = text.contains("--no-config") || text.contains("config-free");
    if has_cli_hint
        && source_text_defines_operator(text, NO_CONFIG_INIT)
        && source_text_defines_operator(text, NO_CONFIG_NEXT)
        && source_text_defines_operator(text, NO_CONFIG_INVARIANT)
    {
        return Some(no_config_cli_flags());
    }
    if !source_mode_is_config_free && !has_cli_hint {
        return None;
    }
    if !source_text_defines_operator(text, "Init") || !source_text_defines_operator(text, "Next") {
        return None;
    }

    let mut flags = vec![
        "--no-config".to_string(),
        "--init".to_string(),
        "Init".to_string(),
        "--next".to_string(),
        "Next".to_string(),
    ];
    if source_text_defines_operator(text, "TypeOK") {
        flags.extend(["--inv".to_string(), "TypeOK".to_string()]);
    }
    Some(flags)
}

fn source_text_defines_operator(text: &str, name: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim_start();
        if line.starts_with("\\*") {
            return false;
        }
        line.strip_prefix(name)
            .is_some_and(|rest| rest.trim_start().starts_with("=="))
    })
}

pub(super) fn no_config_cli_flags() -> Vec<String> {
    NO_CONFIG_CLI_FLAGS
        .iter()
        .map(|flag| (*flag).to_string())
        .collect()
}

pub(super) fn no_config_tlc_config_text() -> String {
    no_config_tlc_config_text_from_flags(&no_config_cli_flags())
        .expect("default config-free CLI flags should define init/next/invariant")
}

pub(super) fn no_config_tlc_config_text_from_flags(flags: &[String]) -> Result<String> {
    let entrypoints = no_config_entrypoints_from_flags(flags)
        .context("config-free CLI flags must include --no-config, --init, and --next")?;
    let mut text = format!("INIT {}\nNEXT {}\n", entrypoints.init, entrypoints.next);
    for invariant in entrypoints.invariants {
        let _ = writeln!(text, "INVARIANT {invariant}");
    }
    Ok(text)
}

fn resolve_baseline_path(path: &Path, examples_dir: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        examples_dir.join(path)
    }
}

fn path_kind(tla_path: Option<&PathBuf>, cfg_path: Option<&PathBuf>) -> MatrixRefreshPathKind {
    let (Some(tla_path), Some(cfg_path)) = (tla_path, cfg_path) else {
        return MatrixRefreshPathKind::Missing;
    };
    match (tla_path.is_absolute(), cfg_path.is_absolute()) {
        (false, false) => MatrixRefreshPathKind::ExamplesRelative,
        (true, true) => MatrixRefreshPathKind::Absolute,
        (true, false) | (false, true) => MatrixRefreshPathKind::Mixed,
    }
}

fn finite_positive(seconds: Option<f64>) -> bool {
    seconds.is_some_and(|seconds| seconds.is_finite() && seconds > 0.0)
}

#[derive(Clone, Debug, Deserialize)]
struct RefreshBaseline {
    #[serde(default)]
    inputs: Option<RefreshBaselineInputs>,
    specs: BTreeMap<String, RefreshBaselineSpec>,
}

#[derive(Clone, Debug, Deserialize)]
struct RefreshBaselineInputs {
    #[serde(default)]
    examples_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
struct RefreshBaselineSpec {
    #[serde(default)]
    category: Option<String>,
    tlc: RefreshBaselineMode,
    ty: RefreshBaselineMode,
    #[serde(default)]
    source: Option<RefreshBaselineSource>,
    #[serde(default, flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct RefreshBaselineMode {
    #[serde(default)]
    runtime_seconds: Option<f64>,
    #[serde(default)]
    states: Option<u64>,
    #[serde(default, flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct RefreshBaselineSource {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    tla_path: Option<PathBuf>,
    #[serde(default)]
    cfg_path: Option<PathBuf>,
    #[serde(default, flatten)]
    extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, text).unwrap();
    }

    fn baseline(specs: &str, examples_dir: &Path) -> String {
        format!(
            r#"{{
              "inputs": {{"examples_dir": "{}"}},
              "specs": {{{specs}}}
            }}"#,
            examples_dir.display()
        )
    }

    fn missing_runtime_spec(name: &str, tla_path: &Path, cfg_path: &Path) -> String {
        format!(
            r#""{name}": {{
              "category": "small",
              "source": {{
                "tla_path": "{}",
                "cfg_path": "{}"
              }},
              "tlc": {{
                "status": "pass",
                "runtime_seconds": 0.5,
                "states": 3,
                "error_type": null
              }},
              "ty": {{
                "status": "pass",
                "states": 3,
                "error_type": null
              }},
              "verified_match": true
            }}"#,
            tla_path.display(),
            cfg_path.display()
        )
    }

    fn timed_spec(name: &str, tla_path: &Path, cfg_path: &Path, tlc: f64, ty: f64) -> String {
        format!(
            r#""{name}": {{
              "category": "small",
              "source": {{
                "tla_path": "{}",
                "cfg_path": "{}"
              }},
              "tlc": {{
                "status": "pass",
                "runtime_seconds": {tlc},
                "states": 3,
                "error_type": null
              }},
              "ty": {{
                "status": "pass",
                "runtime_seconds": {ty},
                "states": 3,
                "error_type": null
              }},
              "verified_match": true
            }}"#,
            tla_path.display(),
            cfg_path.display()
        )
    }

    fn config_free_tla(module: &str) -> String {
        format!(
            r#"---- MODULE {module} ----
\* Config-free checking: --no-config --init --next --inv.
VARIABLE counter
MyInit == counter = 0
MyNext == counter' = IF counter < 2 THEN counter + 1 ELSE counter
TypeOK == counter \in {{0, 1, 2}}
====
"#
        )
    }

    #[test]
    fn plans_examples_relative_missing_runtime_rows() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/A.tla"),
            "---- MODULE A ----\n====\n",
        );
        write_file(&examples_dir.join("specs/A.cfg"), "INIT Init\n");
        let text = baseline(
            &missing_runtime_spec("A", Path::new("specs/A.tla"), Path::new("specs/A.cfg")),
            &examples_dir,
        );

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.batchable_runtime_rows, 1);
        assert_eq!(plan.counts.blocked_runtime_rows, 0);
        assert_eq!(plan.counts.runnable_with_config, 1);
        assert_eq!(plan.counts.examples_relative_paths, 1);
        assert!(plan.can_batch_all_missing_runtime_rows());
        assert_eq!(plan.schema, MATRIX_REFRESH_PLAN_SCHEMA);
        assert_eq!(plan.batchable_runtime_specs, vec!["A"]);
        assert!(plan.blocked_runtime_specs.is_empty());
        assert_eq!(
            plan.batchable_runtime_spec_args,
            vec!["--runtime-spec".to_string(), "A".to_string()]
        );
        assert_eq!(plan.batchable_specs(), vec!["A"]);
        assert_eq!(plan.batchable_specs_limited(Some(1)), vec!["A"]);
        assert!(plan.skipped_batchable_specs_by_limit(Some(1)).is_empty());
        assert_eq!(plan.row("A").map(|row| row.spec.as_str()), Some("A"));
        assert!(plan.row("missing").is_none());
        assert!(plan.blocked_specs().is_empty());
        assert_eq!(
            plan.rows[0].source.resolved_tla_path.as_deref(),
            Some(examples_dir.join("specs/A.tla").as_path())
        );
        assert_eq!(
            plan.rows[0].readiness,
            MatrixRefreshReadiness::RunnableWithConfig
        );
    }

    #[test]
    fn plans_absolute_repo_local_staged_fixtures() {
        let dir = tempfile::tempdir().unwrap();
        let tla_path = dir.path().join("repo/specs/ay/MCTest.tla");
        let cfg_path = dir.path().join("repo/specs/ay/MCTest.cfg");
        write_file(&tla_path, "---- MODULE MCTest ----\n====\n");
        write_file(&cfg_path, "INIT Init\n");
        let text = baseline(
            &missing_runtime_spec("MCTest", &tla_path, &cfg_path),
            &dir.path().join("examples"),
        );

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.runnable_with_config, 1);
        assert_eq!(plan.counts.absolute_source_paths, 1);
        assert_eq!(
            plan.rows[0].source.resolved_cfg_path.as_deref(),
            Some(cfg_path.as_path())
        );
    }

    #[test]
    fn reports_no_config_cli_flags_without_requiring_cfg() {
        let dir = tempfile::tempdir().unwrap();
        let tla_path = dir
            .path()
            .join("tests/apalache_parity/specs/ConfigFreeCounter.tla");
        let cfg_path = dir
            .path()
            .join("tests/apalache_parity/configs/ConfigFreeCounter.cfg");
        write_file(&tla_path, &config_free_tla("ConfigFreeCounter"));
        let text = baseline(
            &missing_runtime_spec("ConfigFreeCounter", &tla_path, &cfg_path),
            &dir.path().join("examples"),
        );

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.runnable_with_config, 0);
        assert_eq!(plan.counts.no_config_cli_flags, 1);
        assert_eq!(plan.counts.batchable_runtime_rows, 1);
        assert_eq!(plan.counts.blocked_runtime_rows, 0);
        match &plan.rows[0].readiness {
            MatrixRefreshReadiness::NeedsNoConfigCliFlags { flags, .. } => {
                assert_eq!(
                    flags,
                    &[
                        "--no-config",
                        "--init",
                        "MyInit",
                        "--next",
                        "MyNext",
                        "--inv",
                        "TypeOK"
                    ]
                );
            }
            other => panic!("expected no-config readiness, got {other:?}"),
        }
    }

    #[test]
    fn detects_config_free_rows_without_exact_spec_name() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/ConfigFreeCounter.tla"),
            &config_free_tla("ConfigFreeCounter"),
        );
        let text = baseline(
            &missing_runtime_spec(
                "ConfigFreeCounter",
                Path::new("specs/ConfigFreeCounter.tla"),
                Path::new("specs/ConfigFreeCounter.cfg"),
            ),
            &examples_dir,
        );

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.no_config_cli_flags, 1);
        assert_eq!(plan.counts.batchable_runtime_rows, 1);
        assert_eq!(
            plan.batchable_runtime_specs,
            vec!["ConfigFreeCounter".to_string()]
        );
        assert_eq!(plan.batchable_specs(), vec!["ConfigFreeCounter"]);
        assert_eq!(
            plan.batchable_runtime_spec_args,
            vec![
                "--runtime-spec".to_string(),
                "ConfigFreeCounter".to_string()
            ]
        );
        assert_eq!(
            plan.runtime_spec_args_for_batchable_rows(),
            vec![
                "--runtime-spec".to_string(),
                "ConfigFreeCounter".to_string()
            ]
        );
    }

    #[test]
    fn detects_config_free_rows_without_cfg_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/ConfigFreeNoCfgMetadata.tla"),
            &config_free_tla("ConfigFreeNoCfgMetadata"),
        );
        let spec = r#""ConfigFreeNoCfgMetadata": {
          "category": "small",
          "source": {"tla_path": "specs/ConfigFreeNoCfgMetadata.tla"},
          "tlc": {"status": "pass", "runtime_seconds": 0.5, "states": 3, "error_type": null},
          "ty": {"status": "pass", "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(spec, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.no_config_cli_flags, 1);
        assert_eq!(plan.counts.missing_source_metadata, 0);
        assert!(matches!(
            plan.rows[0].readiness,
            MatrixRefreshReadiness::NeedsNoConfigCliFlags { .. }
        ));
    }

    #[test]
    fn detects_no_config_source_mode_with_cli_convention_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/ConventionNoConfig.tla"),
            r#"---- MODULE ConventionNoConfig ----
VARIABLE x
Init == x = 0
Next == x' = IF x < 2 THEN x + 1 ELSE x
====
"#,
        );
        let spec = r#""ConventionNoConfig": {
          "category": "apalache",
          "source": {"mode": "no-config", "tla_path": "specs/ConventionNoConfig.tla"},
          "tlc": {"status": "pass", "runtime_seconds": 0.5, "states": 3, "error_type": null},
          "ty": {"status": "pass", "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(spec, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.no_config_cli_flags, 1);
        assert_eq!(plan.counts.missing_source_metadata, 0);
        assert_eq!(plan.counts.batchable_runtime_rows, 1);
        match &plan.rows[0].readiness {
            MatrixRefreshReadiness::NeedsNoConfigCliFlags { flags, .. } => {
                assert_eq!(flags, &["--no-config", "--init", "Init", "--next", "Next"]);
                assert_eq!(
                    no_config_tlc_config_text_from_flags(flags).unwrap(),
                    "INIT Init\nNEXT Next\n"
                );
            }
            other => panic!("expected convention no-config readiness, got {other:?}"),
        }
    }

    #[test]
    fn detects_no_config_source_mode_with_convention_typeok() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/ConventionTypeOk.tla"),
            r#"---- MODULE ConventionTypeOk ----
VARIABLE x
Init == x = 0
Next == x' = x
TypeOK == x = 0
====
"#,
        );
        let spec = r#""ConventionTypeOk": {
          "category": "apalache",
          "source": {"mode": "config_free", "tla_path": "specs/ConventionTypeOk.tla"},
          "tlc": {"status": "pass", "runtime_seconds": 0.5, "states": 1, "error_type": null},
          "ty": {"status": "pass", "states": 1, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(spec, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        match &plan.rows[0].readiness {
            MatrixRefreshReadiness::NeedsNoConfigCliFlags { flags, .. } => {
                assert_eq!(
                    flags,
                    &[
                        "--no-config",
                        "--init",
                        "Init",
                        "--next",
                        "Next",
                        "--inv",
                        "TypeOK"
                    ]
                );
            }
            other => panic!("expected convention no-config readiness, got {other:?}"),
        }
    }

    #[test]
    fn honors_no_config_source_mode_even_when_cfg_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/NoConfigWithCfg.tla"),
            r#"---- MODULE NoConfigWithCfg ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        write_file(
            &examples_dir.join("specs/NoConfigWithCfg.cfg"),
            "INIT OtherInit\nNEXT OtherNext\n",
        );
        let spec = r#""NoConfigWithCfg": {
          "category": "apalache",
          "source": {
            "mode": "no_config",
            "tla_path": "specs/NoConfigWithCfg.tla",
            "cfg_path": "specs/NoConfigWithCfg.cfg"
          },
          "tlc": {"status": "pass", "runtime_seconds": 0.5, "states": 1, "error_type": null},
          "ty": {"status": "pass", "states": 1, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(spec, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.no_config_cli_flags, 1);
        assert_eq!(plan.counts.runnable_with_config, 0);
        match &plan.rows[0].readiness {
            MatrixRefreshReadiness::NeedsNoConfigCliFlags { reason, flags } => {
                assert_eq!(flags, &["--no-config", "--init", "Init", "--next", "Next"]);
                assert!(
                    reason.contains("exists but source is config-free"),
                    "{reason}"
                );
            }
            other => panic!("expected no-config source-mode readiness, got {other:?}"),
        }
    }

    #[test]
    fn detects_config_free_rows_from_check_flags_without_cfg_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/NoConfigCliFlags.tla"),
            r#"---- MODULE NoConfigCliFlags ----
VARIABLE counter
MyInit == counter = 0
MyNext == counter' = IF counter < 2 THEN counter + 1 ELSE counter
TypeOK == counter \in {0, 1, 2}
====
"#,
        );
        let spec = r#""NoConfigCliFlags": {
          "category": "apalache",
          "source": {
            "tla_path": "specs/NoConfigCliFlags.tla",
            "check_flags": [
              "--no-config",
              "--init",
              "MyInit",
              "--next",
              "MyNext",
              "--inv",
              "TypeOK"
            ]
          },
          "tlc": {"status": "pass", "runtime_seconds": 0.5, "states": 3, "error_type": null},
          "ty": {"status": "pass", "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(spec, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.no_config_cli_flags, 1);
        assert_eq!(plan.counts.missing_source_metadata, 0);
        assert_eq!(plan.counts.batchable_runtime_rows, 1);
        assert!(plan.can_batch_all_missing_runtime_rows());
        assert_eq!(
            plan.batchable_runtime_spec_args,
            vec!["--runtime-spec".to_string(), "NoConfigCliFlags".to_string()]
        );
        assert_eq!(
            plan.rows[0].source.no_config_cli_flags,
            Some(no_config_cli_flags())
        );
        match &plan.rows[0].readiness {
            MatrixRefreshReadiness::NeedsNoConfigCliFlags { reason, flags } => {
                assert!(
                    reason.contains("source metadata supplies config-free CLI flags"),
                    "{reason}"
                );
                assert_eq!(flags, &no_config_cli_flags());
            }
            other => panic!("expected no-config readiness from check flags, got {other:?}"),
        }
    }

    #[test]
    fn honors_metadata_no_config_flags_even_when_cfg_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/ExplicitNoConfig.tla"),
            r#"---- MODULE ExplicitNoConfig ----
VARIABLE counter
MyInit == counter = 0
MyNext == counter' = counter
TypeOK == counter = 0
====
"#,
        );
        write_file(
            &examples_dir.join("specs/ExplicitNoConfig.cfg"),
            "INIT OtherInit\nNEXT OtherNext\n",
        );
        let spec = r#""ExplicitNoConfig": {
          "category": "apalache",
          "source": {
            "tla_path": "specs/ExplicitNoConfig.tla",
            "cfg_path": "specs/ExplicitNoConfig.cfg",
            "check_flags": [
              "--no-config",
              "--init",
              "MyInit",
              "--next",
              "MyNext",
              "--inv",
              "TypeOK"
            ]
          },
          "tlc": {"status": "pass", "runtime_seconds": 0.5, "states": 1, "error_type": null},
          "ty": {"status": "pass", "states": 1, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(spec, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.no_config_cli_flags, 1);
        assert_eq!(plan.counts.runnable_with_config, 0);
        match &plan.rows[0].readiness {
            MatrixRefreshReadiness::NeedsNoConfigCliFlags { reason, flags } => {
                assert_eq!(flags, &no_config_cli_flags());
                assert!(reason.contains("source metadata supplies config-free CLI flags"));
                assert!(
                    reason.contains("exists but source is config-free"),
                    "{reason}"
                );
            }
            other => panic!("expected no-config metadata readiness, got {other:?}"),
        }
    }

    #[test]
    fn detects_config_free_rows_from_nested_check_required_flags() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/NestedRequiredFlags.tla"),
            r#"---- MODULE NestedRequiredFlags ----
VARIABLE counter
MyInit == counter = 0
MyNext == counter' = counter
TypeOK == counter = 0
====
"#,
        );
        let spec = r#""NestedRequiredFlags": {
          "category": "apalache",
          "source": {
            "tla_path": "specs/NestedRequiredFlags.tla",
            "check": {
              "required_flags": [
                "--no-config",
                "--init",
                "MyInit",
                "--next",
                "MyNext",
                "--inv",
                "TypeOK"
              ]
            }
          },
          "tlc": {"status": "pass", "runtime_seconds": 0.5, "states": 3, "error_type": null},
          "ty": {"status": "pass", "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(spec, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.no_config_cli_flags, 1);
        assert_eq!(plan.counts.missing_source_metadata, 0);
        assert_eq!(plan.counts.batchable_runtime_rows, 1);
        assert_eq!(
            plan.rows[0].source.no_config_cli_flags,
            Some(no_config_cli_flags())
        );
        assert!(matches!(
            plan.rows[0].readiness,
            MatrixRefreshReadiness::NeedsNoConfigCliFlags { .. }
        ));
    }

    #[test]
    fn normalizes_no_config_cli_flags_from_run_argv_metadata() {
        let flags = [
            "ty",
            "check",
            "Spec.tla",
            "--no-config",
            "--init=MyInit",
            "--next=MyNext",
            "--inv=TypeOK",
            "--workers",
            "1",
        ]
        .iter()
        .map(|value| (*value).to_string())
        .collect::<Vec<_>>();

        assert_eq!(
            normalize_no_config_cli_flags(&flags),
            Some(no_config_cli_flags())
        );
    }

    #[test]
    fn normalizes_no_config_cli_flags_without_invariant() {
        let flags = ["--no-config", "--init", "Init", "--next", "Next"]
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            normalize_no_config_cli_flags(&flags),
            Some(vec![
                "--no-config".to_string(),
                "--init".to_string(),
                "Init".to_string(),
                "--next".to_string(),
                "Next".to_string(),
            ])
        );
        assert_eq!(
            no_config_tlc_config_text_from_flags(&flags).unwrap(),
            "INIT Init\nNEXT Next\n"
        );
    }

    #[test]
    fn keeps_regular_missing_cfg_rows_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/NeedsCfg.tla"),
            "---- MODULE NeedsCfg ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\nTypeOK == x = 0\n====\n",
        );
        let text = baseline(
            &missing_runtime_spec(
                "NeedsCfg",
                Path::new("specs/NeedsCfg.tla"),
                Path::new("specs/NeedsCfg.cfg"),
            ),
            &examples_dir,
        );

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.no_config_cli_flags, 0);
        assert_eq!(plan.counts.batchable_runtime_rows, 0);
        assert_eq!(plan.counts.blocked_runtime_rows, 1);
        assert!(plan.batchable_runtime_specs.is_empty());
        assert_eq!(plan.blocked_runtime_specs, vec!["NeedsCfg".to_string()]);
        assert_eq!(plan.blocked_specs(), vec!["NeedsCfg"]);
        assert!(!plan.can_batch_all_missing_runtime_rows());
        match &plan.rows[0].readiness {
            MatrixRefreshReadiness::MissingSourceFiles { missing } => {
                assert_eq!(missing.len(), 1);
                assert!(missing[0].ends_with("specs/NeedsCfg.cfg"));
            }
            other => panic!("expected missing cfg file, got {other:?}"),
        }
    }

    #[test]
    fn reports_missing_source_files_for_regular_rows() {
        let dir = tempfile::tempdir().unwrap();
        let text = baseline(
            &missing_runtime_spec(
                "MissingFiles",
                Path::new("specs/MissingFiles.tla"),
                Path::new("specs/MissingFiles.cfg"),
            ),
            &dir.path().join("examples"),
        );

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.missing_source_files, 1);
        assert_eq!(plan.counts.runnable_with_config, 0);
        match &plan.rows[0].readiness {
            MatrixRefreshReadiness::MissingSourceFiles { missing } => {
                assert_eq!(missing.len(), 2);
            }
            other => panic!("expected missing files, got {other:?}"),
        }
    }

    #[test]
    fn skips_non_missing_runtime_rows_by_using_matrix_classifier() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/A.tla"),
            "---- MODULE A ----\n====\n",
        );
        write_file(&examples_dir.join("specs/A.cfg"), "INIT Init\n");
        let passing = r#""AlreadyTimed": {
          "category": "small",
          "source": {"tla_path": "specs/A.tla", "cfg_path": "specs/A.cfg"},
          "tlc": {"status": "pass", "runtime_seconds": 2.0, "states": 3, "error_type": null},
          "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(passing, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 0);
        assert!(plan.rows.is_empty());
    }

    #[test]
    fn all_runnable_scope_plans_every_batchable_non_unsupported_row() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        for spec in [
            "MissingRuntime",
            "ExpectedViolation",
            "Pass",
            "PerfLoser",
            "PerfTie",
            "ParityFail",
            "TyTimeout",
            "TlcError",
            "TlcTimeout",
            "Unsupported",
        ] {
            write_file(
                &examples_dir.join(format!("specs/{spec}.tla")),
                &format!("---- MODULE {spec} ----\n====\n"),
            );
            write_file(
                &examples_dir.join(format!("specs/{spec}.cfg")),
                "INIT Init\n",
            );
        }
        let parity_fail = r#""ParityFail": {
          "category": "small",
          "source": {"tla_path": "specs/ParityFail.tla", "cfg_path": "specs/ParityFail.cfg"},
          "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
          "ty": {"status": "pass", "runtime_seconds": 0.5, "states": 4, "error_type": null},
          "verified_match": false
        }"#;
        let tlc_error = r#""TlcError": {
          "category": "small",
          "source": {"tla_path": "specs/TlcError.tla", "cfg_path": "specs/TlcError.cfg"},
          "tlc": {"status": "fail", "runtime_seconds": 1.0, "states": 3, "error_type": "runtime_error"},
          "ty": {"status": "pass", "runtime_seconds": 0.5, "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let tlc_timeout = r#""TlcTimeout": {
          "category": "small",
          "source": {"tla_path": "specs/TlcTimeout.tla", "cfg_path": "specs/TlcTimeout.cfg"},
          "tlc": {"status": "timeout", "runtime_seconds": 300.0, "states": 3, "error_type": "timeout"},
          "ty": {"status": "pass", "runtime_seconds": 0.5, "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let expected_violation = r#""ExpectedViolation": {
          "category": "small",
          "source": {"tla_path": "specs/ExpectedViolation.tla", "cfg_path": "specs/ExpectedViolation.cfg"},
          "tlc": {"status": "fail", "runtime_seconds": 3.0, "states": 12, "error_type": "invariant"},
          "ty": {"status": "pass", "runtime_seconds": 1.0, "states": 12, "error_type": "invariant_violation"},
          "verified_match": true
        }"#;
        let ty_timeout = r#""TyTimeout": {
          "category": "small",
          "source": {"tla_path": "specs/TyTimeout.tla", "cfg_path": "specs/TyTimeout.cfg"},
          "tlc": {"status": "pass", "runtime_seconds": 4.0, "states": 3, "error_type": null},
          "ty": {"status": "timeout", "runtime_seconds": 300.0, "states": null, "error_type": "timeout"},
          "verified_match": false
        }"#;
        let unsupported = r#""Unsupported": {
          "category": "small",
          "source": {
            "mode": "unsupported-mode",
            "tla_path": "specs/Unsupported.tla",
            "cfg_path": "specs/Unsupported.cfg"
          },
          "tlc": {"status": "pass", "runtime_seconds": 1.0, "states": 3, "error_type": null},
          "ty": {"status": "pass", "runtime_seconds": 0.5, "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(
            &[
                expected_violation.to_string(),
                missing_runtime_spec(
                    "MissingRuntime",
                    Path::new("specs/MissingRuntime.tla"),
                    Path::new("specs/MissingRuntime.cfg"),
                ),
                timed_spec(
                    "Pass",
                    Path::new("specs/Pass.tla"),
                    Path::new("specs/Pass.cfg"),
                    2.0,
                    1.0,
                ),
                timed_spec(
                    "PerfLoser",
                    Path::new("specs/PerfLoser.tla"),
                    Path::new("specs/PerfLoser.cfg"),
                    1.0,
                    2.0,
                ),
                timed_spec(
                    "PerfTie",
                    Path::new("specs/PerfTie.tla"),
                    Path::new("specs/PerfTie.cfg"),
                    1.0,
                    1.0,
                ),
                parity_fail.to_string(),
                ty_timeout.to_string(),
                tlc_error.to_string(),
                tlc_timeout.to_string(),
                unsupported.to_string(),
            ]
            .join(","),
            &examples_dir,
        );

        let default_plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();
        let plan = plan_runtime_refresh_str(
            &text,
            Path::new("baseline.json"),
            None,
            MatrixRefreshScope::AllRunnable,
        )
        .unwrap();

        assert_eq!(default_plan.scope, MatrixRefreshScope::MissingRuntime);
        assert_eq!(default_plan.batchable_runtime_specs, vec!["MissingRuntime"]);
        assert_eq!(plan.scope, MatrixRefreshScope::AllRunnable);
        // Order-independent: assert set-equality of the batchable specs rather
        // than a stale element order (the list is built from row iteration).
        let mut batchable = plan.batchable_runtime_specs.clone();
        batchable.sort();
        assert_eq!(
            batchable,
            vec![
                "ExpectedViolation",
                "MissingRuntime",
                "ParityFail",
                "Pass",
                "PerfLoser",
                "PerfTie",
                "TlcError",
                "TlcTimeout",
                "TyTimeout"
            ]
        );
        assert_eq!(plan.counts.total_runtime_rows, 9);
        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.batchable_runtime_rows, 9);
        assert_eq!(plan.counts.blocked_runtime_rows, 0);
        assert!(plan.can_batch_all_selected_runtime_rows());
        assert!(plan.can_batch_all_missing_runtime_rows());
        // Order-independent: compare the multiset of row classes (sorted by name)
        // rather than a stale element order.
        let mut classes = plan.rows.iter().map(|row| row.class).collect::<Vec<_>>();
        classes.sort_by_key(|c| format!("{c:?}"));
        assert_eq!(
            classes,
            vec![
                SupremacyMatrixClass::ExpectedViolationMatch,
                SupremacyMatrixClass::MissingRuntime,
                SupremacyMatrixClass::ParityFail,
                SupremacyMatrixClass::Pass,
                SupremacyMatrixClass::PerfLoser,
                SupremacyMatrixClass::PerfTie,
                SupremacyMatrixClass::TlcError,
                SupremacyMatrixClass::TlcTimeout,
                SupremacyMatrixClass::TyTimeout,
            ]
        );
        assert!(plan.row("TlcError").is_some());
        assert!(plan.row("TlcTimeout").is_some());
        assert!(plan.row("Unsupported").is_none());
        assert!(plan.row("ParityFail").is_some());
    }

    #[test]
    fn all_runnable_scope_keeps_non_batchable_runnable_rows_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/MissingRuntime.tla"),
            "---- MODULE MissingRuntime ----\n====\n",
        );
        write_file(
            &examples_dir.join("specs/MissingRuntime.cfg"),
            "INIT Init\n",
        );
        write_file(
            &examples_dir.join("specs/PassMissingCfg.tla"),
            "---- MODULE PassMissingCfg ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\nTypeOK == x = 0\n====\n",
        );
        let text = baseline(
            &[
                missing_runtime_spec(
                    "MissingRuntime",
                    Path::new("specs/MissingRuntime.tla"),
                    Path::new("specs/MissingRuntime.cfg"),
                ),
                timed_spec(
                    "PassMissingCfg",
                    Path::new("specs/PassMissingCfg.tla"),
                    Path::new("specs/PassMissingCfg.cfg"),
                    2.0,
                    1.0,
                ),
            ]
            .join(","),
            &examples_dir,
        );

        let plan = plan_runtime_refresh_str(
            &text,
            Path::new("baseline.json"),
            None,
            MatrixRefreshScope::AllRunnable,
        )
        .unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.total_runtime_rows, 2);
        assert_eq!(plan.batchable_runtime_specs, vec!["MissingRuntime"]);
        assert_eq!(plan.blocked_runtime_specs, vec!["PassMissingCfg"]);
        assert!(!plan.can_batch_all_selected_runtime_rows());
        assert!(plan.can_batch_all_missing_runtime_rows());
        assert_eq!(
            plan.row("PassMissingCfg").map(|row| row.class),
            Some(SupremacyMatrixClass::Pass)
        );
    }

    #[test]
    fn counts_tlc_and_ty_runtime_gaps() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        write_file(
            &examples_dir.join("specs/A.tla"),
            "---- MODULE A ----\n====\n",
        );
        write_file(&examples_dir.join("specs/A.cfg"), "INIT Init\n");
        let spec = r#""A": {
          "category": "small",
          "source": {"tla_path": "specs/A.tla", "cfg_path": "specs/A.cfg"},
          "tlc": {"status": "pass", "states": 3, "error_type": null},
          "ty": {"status": "pass", "states": 3, "error_type": null},
          "verified_match": true
        }"#;
        let text = baseline(spec, &examples_dir);

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.counts.total_missing_runtime, 1);
        assert_eq!(plan.counts.tlc_runtime_missing, 1);
        assert_eq!(plan.counts.ty_runtime_missing, 1);
    }

    #[test]
    fn limits_batchable_specs_without_including_blocked_rows() {
        let dir = tempfile::tempdir().unwrap();
        let examples_dir = dir.path().join("examples");
        for spec in ["A", "B", "Blocked"] {
            write_file(
                &examples_dir.join(format!("specs/{spec}.tla")),
                &format!("---- MODULE {spec} ----\n====\n"),
            );
        }
        write_file(&examples_dir.join("specs/A.cfg"), "INIT Init\n");
        write_file(&examples_dir.join("specs/B.cfg"), "INIT Init\n");
        let text = baseline(
            &[
                missing_runtime_spec("A", Path::new("specs/A.tla"), Path::new("specs/A.cfg")),
                missing_runtime_spec("B", Path::new("specs/B.tla"), Path::new("specs/B.cfg")),
                missing_runtime_spec(
                    "Blocked",
                    Path::new("specs/Blocked.tla"),
                    Path::new("specs/Blocked.cfg"),
                ),
            ]
            .join(","),
            &examples_dir,
        );

        let plan =
            plan_missing_runtime_refresh_str(&text, Path::new("baseline.json"), None).unwrap();

        assert_eq!(plan.batchable_runtime_specs, vec!["A", "B"]);
        assert_eq!(plan.blocked_runtime_specs, vec!["Blocked"]);
        assert_eq!(plan.batchable_specs_limited(Some(1)), vec!["A"]);
        assert_eq!(plan.skipped_batchable_specs_by_limit(Some(1)), vec!["B"]);
        assert_eq!(plan.batchable_specs_limited(None), vec!["A", "B"]);
        assert!(plan.skipped_batchable_specs_by_limit(None).is_empty());
    }
}
