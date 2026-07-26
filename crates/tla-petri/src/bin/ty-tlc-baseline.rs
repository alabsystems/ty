// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLC baseline collector — Rust port of `scripts/collect_tlc_baseline/`.
//!
//! Runs the Java TLC baseline tool (`~/tlaplus/tytools.jar`) against
//! every eligible row in
//! `tests/tlc_comparison/strict_corpus_manifest.json`, parses TLC
//! stdout/stderr, classifies the verdict, and emits or refreshes
//! `tests/tlc_comparison/spec_baseline.json` (schema v4).
//!
//! The checked-in manifest is the only default catalog. It enumerates every
//! pinned `tlaplus/Examples` config, records non-same-stem TLA mappings, and
//! retains excluded rows with stable reason codes. `--list`, `--dry-run`, and
//! `--write-skeleton` validate this contract without starting model checking.
//!
//! Provenance fields (TLC version, jar SHA-256, examples-repo git
//! head, etc.) are recorded deterministically so consumers like
//! `system_health_check` can detect baseline drift. The `specs` map is
//! canonicalized via JCS-style ordering and a SHA-256 digest is
//! recorded in `specs_jcs_sha256` so byte-for-byte formatting drift
//! is detectable without re-running TLC.
//!
//! This is the single compiler-enforced interface for baseline
//! collection — the Python package
//! `scripts/collect_tlc_baseline/` it replaces has been deleted in
//! the same commit.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

// ---------- Constants ----------

const SCHEMA_VERSION: u64 = 4;
const DEFAULT_TIMEOUT_SECONDS: u64 = 600;
const WORK_EQUIVALENCE_POLICY_SCHEMA_VERSION: u64 = 1;
const EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID: &str = "exhaustive_generated_work_parity_v1";
const STRICT_EXCLUSION_REASON_CODES: &[&str] = &[
    "deadlock_first_found_noncomparable",
    "expected_violation_first_found_noncomparable",
    "external_io_dependency",
    "external_io_side_effect",
    "nested_tool_driver",
    "randomized_external_operator",
    "semantic_assertion_only",
    "simulation_only",
];
const STATS_KEY_ORDER: &[&str] = &[
    "tlc_pass",
    "tlc_error",
    "tlc_timeout",
    "tlc_unsupported",
    "tlc_uncollected",
    "ty_match",
    "ty_mismatch",
    "ty_fail",
    "ty_untested",
];
const CATEGORIES_KEY_ORDER: &[&str] = &["small", "medium", "large", "xlarge", "skip", "unknown"];
const TLC_ENTRY_KEY_ORDER: &[&str] = &[
    "status",
    "states",
    "raw_initial_states_generated",
    "raw_successors_generated",
    "states_generated",
    "runtime_seconds",
    "error_type",
    "error",
];
const TY_ENTRY_KEY_ORDER: &[&str] = &[
    "status",
    "states",
    "raw_initial_states_generated",
    "raw_successors_generated",
    "states_generated",
    "error_type",
    "last_run",
    "git_commit",
];
const SPEC_ENTRY_KEY_ORDER: &[&str] = &[
    "tlc",
    "ty",
    "verified_match",
    "eligibility",
    "work_equivalence",
    "exclusion",
    "issue",
    "category",
    "source",
];
const LEGACY_V2_KEYS: &[&str] = &[
    "expected_states",
    "tlc_runtime_seconds",
    "status",
    "error",
    "error_type",
];

// ---------- CLI ----------

#[derive(Parser, Debug)]
#[command(
    name = "ty-tlc-baseline",
    about = "Collect TLC baselines from the pinned strict-corpus manifest",
    long_about = "Runs TLC against every eligible row in \
                  tests/tlc_comparison/strict_corpus_manifest.json and writes \
                  tests/tlc_comparison/spec_baseline.json (schema v4). The manifest \
                  is the auditable catalog: all pinned configs, exact TLA mappings, \
                  and stable exclusions remain visible in the output."
)]
struct Cli {
    /// Timeout per spec in seconds.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECONDS)]
    timeout: u64,

    /// Do not reuse the existing baseline file as a resume cache.
    #[arg(long)]
    no_resume: bool,

    /// Override the strict-corpus manifest. Defaults to
    /// `tests/tlc_comparison/strict_corpus_manifest.json`.
    #[arg(long, value_name = "PATH")]
    manifest: Option<PathBuf>,

    /// Print the normalized catalog as TSV and exit without reading source
    /// files, writing output, or launching TLC.
    #[arg(long, conflicts_with_all = ["dry_run", "write_skeleton"])]
    list: bool,

    /// Validate the manifest and pinned source checkout, then exit without
    /// writing output or launching TLC.
    #[arg(long, conflicts_with_all = ["list", "write_skeleton"])]
    dry_run: bool,

    /// Write a complete 181-row baseline skeleton without launching TLC.
    #[arg(long, conflicts_with_all = ["list", "dry_run"])]
    write_skeleton: bool,

    /// Override `spec_baseline.json` output path.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Override TLA+ examples specifications directory.
    /// Defaults to `~/tlaplus-examples/specifications`.
    #[arg(long, value_name = "PATH")]
    examples_dir: Option<PathBuf>,

    /// Override `tytools.jar` path. Defaults to `~/tlaplus/tytools.jar`.
    #[arg(long, value_name = "PATH")]
    tlc_jar: Option<PathBuf>,

    /// Override `CommunityModules.jar` path. Defaults to
    /// `~/tlaplus/CommunityModules.jar`.
    #[arg(long, value_name = "PATH")]
    community_modules: Option<PathBuf>,

    /// Override the `~/tlaplus` git repo path (provenance only).
    #[arg(long, value_name = "PATH")]
    tlaplus_dir: Option<PathBuf>,

    /// Override the `~/tlaplus-examples` git repo path (provenance only).
    #[arg(long, value_name = "PATH")]
    examples_base_dir: Option<PathBuf>,

    /// Override the project root. Defaults to the current working directory.
    #[arg(long, value_name = "PATH")]
    project_root: Option<PathBuf>,
}

// ---------- Catalog model ----------

#[derive(Debug, Clone)]
struct SpecInfo {
    name: String,
    tla_path: String,
    cfg_path: String,
    exclusion: Option<ManifestExclusion>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct ManifestExclusion {
    reason_code: String,
    detail: String,
}

#[derive(Debug, Deserialize)]
struct CorpusManifest {
    schema_version: u64,
    claim: String,
    source: ManifestSource,
    eligibility: ManifestEligibility,
    work_equivalence_policy: ManifestWorkEquivalencePolicy,
    #[serde(default)]
    tla_path_overrides: BTreeMap<String, String>,
    #[serde(default)]
    baseline_gaps: BTreeMap<String, ManifestGap>,
    rows: Vec<ManifestRow>,
}

#[derive(Debug, Deserialize)]
struct ManifestSource {
    repository: String,
    commit: String,
    root: String,
    enumeration: String,
    default_tla_mapping: String,
    expected_cfg_count: usize,
}

#[derive(Debug, Deserialize)]
struct ManifestEligibility {
    default: String,
    #[serde(default)]
    exclusions: BTreeMap<String, ManifestExclusion>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWorkEquivalencePolicy {
    schema_version: u64,
    default_eligible_rule_id: String,
    rules: BTreeMap<String, ManifestWorkEquivalenceRule>,
    outcome_dispositions: ManifestOutcomeDispositions,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWorkEquivalenceRule {
    kind: String,
    required_verdict: String,
    require_complete_exploration: bool,
    distinct_state_parity: String,
    raw_initial_state_generation_parity: String,
    raw_successor_generation_parity: String,
    total_state_generation_parity: String,
    count_arm: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestOutcomeDispositions {
    expected_violation: String,
    deadlock: String,
    simulation: String,
    randomized_external_operator: String,
    external_io: String,
    timeout: String,
}

#[derive(Debug, Deserialize)]
struct ManifestGap {
    row_name: String,
    tla_path: String,
}

#[derive(Debug, Deserialize)]
struct ManifestRow {
    name: String,
    cfg_path: String,
    tla_path: String,
}

fn parse_corpus_manifest(text: &str) -> Result<CorpusManifest> {
    let manifest: CorpusManifest =
        serde_json::from_str(text).context("parse strict-corpus manifest JSON")?;
    validate_corpus_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_corpus_manifest(manifest: &CorpusManifest) -> Result<()> {
    if manifest.schema_version != 1 {
        bail!(
            "unsupported strict-corpus manifest schema {}; expected 1",
            manifest.schema_version
        );
    }
    if manifest.claim != "ty_vs_tlc_strict_superiority" {
        bail!("unexpected corpus claim: {}", manifest.claim);
    }
    if manifest.source.enumeration != "all_cfg_files"
        || manifest.source.default_tla_mapping != "same_stem"
    {
        bail!("manifest must enumerate all cfg files with same-stem default mapping");
    }
    if manifest.eligibility.default != "eligible" {
        bail!("manifest eligibility.default must be eligible");
    }
    validate_work_equivalence_policy(&manifest.work_equivalence_policy)?;
    if manifest.source.commit.len() != 40
        || !manifest
            .source
            .commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("manifest source commit must be a full 40-hex Git object ID");
    }
    validate_relative_manifest_path(&manifest.source.root, None)?;
    if manifest.rows.len() != manifest.source.expected_cfg_count {
        bail!(
            "manifest has {} rows but source.expected_cfg_count is {}",
            manifest.rows.len(),
            manifest.source.expected_cfg_count
        );
    }

    let mut names = BTreeSet::new();
    let mut cfg_paths = BTreeSet::new();
    let mut previous_cfg: Option<&str> = None;
    for row in &manifest.rows {
        if row.name.trim().is_empty() {
            bail!("manifest row has an empty name");
        }
        if !names.insert(row.name.as_str()) {
            bail!("duplicate manifest row name: {}", row.name);
        }
        if !cfg_paths.insert(row.cfg_path.as_str()) {
            bail!("duplicate manifest cfg path: {}", row.cfg_path);
        }
        if previous_cfg.is_some_and(|previous| previous >= row.cfg_path.as_str()) {
            bail!("manifest rows must be strictly sorted by cfg_path");
        }
        previous_cfg = Some(&row.cfg_path);
        validate_relative_manifest_path(&row.cfg_path, Some("cfg"))?;
        validate_relative_manifest_path(&row.tla_path, Some("tla"))?;

        let same_stem = format!(
            "{}.tla",
            row.cfg_path
                .strip_suffix(".cfg")
                .expect("validated cfg extension")
        );
        let expected_tla = manifest
            .tla_path_overrides
            .get(&row.cfg_path)
            .map(String::as_str)
            .unwrap_or(&same_stem);
        if row.tla_path != expected_tla {
            bail!(
                "manifest row {} maps {} to {}, expected {} from mapping rules",
                row.name,
                row.cfg_path,
                row.tla_path,
                expected_tla
            );
        }
    }

    for cfg_path in manifest.tla_path_overrides.keys() {
        if !cfg_paths.contains(cfg_path.as_str()) {
            bail!("TLA override references unknown cfg: {cfg_path}");
        }
    }
    for cfg_path in manifest.eligibility.exclusions.keys() {
        if !cfg_paths.contains(cfg_path.as_str()) {
            bail!("exclusion references unknown cfg: {cfg_path}");
        }
    }
    for (cfg_path, exclusion) in &manifest.eligibility.exclusions {
        if !STRICT_EXCLUSION_REASON_CODES.contains(&exclusion.reason_code.as_str()) {
            bail!(
                "exclusion for {cfg_path} uses unsupported reason code {:?}",
                exclusion.reason_code
            );
        }
        if exclusion.detail.trim().is_empty() {
            bail!("exclusion for {cfg_path} must include a nonempty detail");
        }
    }
    for (cfg_path, gap) in &manifest.baseline_gaps {
        let Some(row) = manifest.rows.iter().find(|row| row.cfg_path == *cfg_path) else {
            bail!("baseline gap references unknown cfg: {cfg_path}");
        };
        if row.name != gap.row_name || row.tla_path != gap.tla_path {
            bail!("baseline gap metadata disagrees with explicit row for {cfg_path}");
        }
    }
    Ok(())
}

fn validate_work_equivalence_policy(policy: &ManifestWorkEquivalencePolicy) -> Result<()> {
    if policy.schema_version != WORK_EQUIVALENCE_POLICY_SCHEMA_VERSION {
        bail!(
            "unsupported work-equivalence policy schema {}; expected {}",
            policy.schema_version,
            WORK_EQUIVALENCE_POLICY_SCHEMA_VERSION
        );
    }
    if policy.default_eligible_rule_id != EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID {
        bail!("work-equivalence default rule must be {EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID:?}");
    }
    if policy.rules.len() != 1 {
        bail!(
            "work-equivalence policy schema {} must define exactly one rule",
            WORK_EQUIVALENCE_POLICY_SCHEMA_VERSION
        );
    }
    let rule = policy
        .rules
        .get(EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID)
        .context("work-equivalence policy is missing its default exhaustive rule")?;
    if rule.kind != "exhaustive_state_space"
        || rule.required_verdict != "holds"
        || !rule.require_complete_exploration
        || rule.distinct_state_parity != "exact"
        || rule.raw_initial_state_generation_parity != "exact"
        || rule.raw_successor_generation_parity != "exact"
        || rule.total_state_generation_parity != "exact"
        || rule.count_arm != "bfs_no_reduction_single_worker"
    {
        bail!(
            "work-equivalence rule {EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID:?} does not match the schema-v1 exhaustive contract"
        );
    }

    let dispositions = &policy.outcome_dispositions;
    if dispositions.expected_violation != "exclude_unless_predeclared_typed_rule"
        || dispositions.deadlock != "exclude_unless_predeclared_typed_rule"
        || dispositions.simulation != "exclude"
        || dispositions.randomized_external_operator != "exclude"
        || dispositions.external_io != "exclude"
        || dispositions.timeout != "missing_or_stale"
    {
        bail!("work-equivalence outcome dispositions do not match the schema-v1 contract");
    }
    Ok(())
}

fn validate_relative_manifest_path(path: &str, extension: Option<&str>) -> Result<()> {
    if path.is_empty() || Path::new(path).is_absolute() {
        bail!("manifest path must be nonempty and relative: {path:?}");
    }
    if Path::new(path)
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("manifest path may not contain traversal or special components: {path}");
    }
    if let Some(extension) = extension {
        if Path::new(path).extension().and_then(|value| value.to_str()) != Some(extension) {
            bail!("manifest path must end in .{extension}: {path}");
        }
    }
    Ok(())
}

fn catalog_from_manifest(manifest: &CorpusManifest) -> Vec<SpecInfo> {
    manifest
        .rows
        .iter()
        .map(|row| SpecInfo {
            name: row.name.clone(),
            tla_path: row.tla_path.clone(),
            cfg_path: row.cfg_path.clone(),
            exclusion: manifest.eligibility.exclusions.get(&row.cfg_path).cloned(),
        })
        .collect()
}

// ---------- TLC execution ----------

#[derive(Debug, Clone)]
struct TlcOutcome {
    status: String,
    states: Option<u64>,
    raw_initial_states_generated: Option<u64>,
    raw_successors_generated: Option<u64>,
    states_generated: Option<u64>,
    runtime_seconds: Option<f64>,
    error_type: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
struct TlcParsedCounts {
    states: Option<u64>,
    raw_initial_states_generated: Option<u64>,
    raw_successors_generated: Option<u64>,
    states_generated: Option<u64>,
    states_left: Option<u64>,
}

impl TlcParsedCounts {
    fn complete(self) -> bool {
        self.states.is_some()
            && self.raw_initial_states_generated.is_some()
            && self.raw_successors_generated.is_some()
            && self.states_generated.is_some()
            && self.states_left == Some(0)
    }
}

fn run_tlc(
    spec_path: &Path,
    cfg_path: &Path,
    timeout_seconds: u64,
    tlc_jar: &Path,
    community_modules: &Path,
    project_root: &Path,
) -> TlcOutcome {
    let mut outcome = TlcOutcome {
        status: "unknown".into(),
        states: None,
        raw_initial_states_generated: None,
        raw_successors_generated: None,
        states_generated: None,
        runtime_seconds: None,
        error_type: None,
        error: None,
    };

    let use_ephemeral_metadir = std::env::var("TY_KEEP_STATES")
        .map(|v| v.trim() != "1")
        .unwrap_or(true);
    let preserve_states_dir = std::env::var("TY_PRESERVE_STATES_DIR")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let states_dir = spec_path
        .parent()
        .map(|p| p.join("states"))
        .unwrap_or_else(|| PathBuf::from("states"));

    let metadir_root = std::env::var_os("TY_TLC_METADIR_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root.join("target").join("tlc_metadir"));
    if use_ephemeral_metadir {
        if let Err(e) = fs::create_dir_all(&metadir_root) {
            outcome.status = "error".into();
            outcome.error = Some(format!(
                "failed to create TLC metadir root {}: {e}",
                metadir_root.display()
            ));
            outcome.error_type = Some("metadir_setup".into());
            return outcome;
        }
    }

    let metadir = if use_ephemeral_metadir {
        match tempfile::Builder::new()
            .prefix("tlc-")
            .tempdir_in(&metadir_root)
        {
            Ok(td) => Some(td),
            Err(e) => {
                outcome.status = "error".into();
                outcome.error = Some(format!("failed to create TLC metadir: {e}"));
                outcome.error_type = Some("metadir_setup".into());
                return outcome;
            }
        }
    } else {
        None
    };

    let mut classpath = OsString::from(tlc_jar.as_os_str());
    if community_modules.exists() {
        #[cfg(unix)]
        classpath.push(":");
        #[cfg(not(unix))]
        classpath.push(";");
        classpath.push(community_modules.as_os_str());
    }

    let mut cmd = Command::new("java");
    cmd.arg("-Xmx4g").arg("-cp").arg(&classpath).arg("tlc2.TLC");
    if let Some(md) = metadir.as_ref() {
        cmd.arg("-metadir").arg(md.path());
    }
    cmd.arg("-config")
        .arg(cfg_path)
        .arg("-workers")
        .arg("1")
        .arg(spec_path);
    if let Some(parent) = spec_path.parent() {
        cmd.current_dir(parent);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let start = Instant::now();
    let result = run_with_timeout(cmd, Duration::from_secs(timeout_seconds));

    drop(metadir); // remove the tempdir before cleaning ./states

    if use_ephemeral_metadir && !preserve_states_dir {
        cleanup_states_dir(&states_dir);
    }

    match result {
        Ok((code, timed_out, stdout, stderr)) => {
            let elapsed = start.elapsed().as_secs_f64();
            outcome.runtime_seconds = Some(round2(elapsed));
            if timed_out {
                outcome.status = "timeout".into();
                outcome.runtime_seconds = Some(timeout_seconds as f64);
                outcome.error = Some(format!("Timeout after {timeout_seconds}s"));
                outcome.error_type = Some("timeout".into());
                return outcome;
            }
            let combined = format!("{stdout}{stderr}");
            let (counts, parse_err) = parse_tlc_output(&combined);
            outcome.states = counts.states;
            outcome.raw_initial_states_generated = counts.raw_initial_states_generated;
            outcome.raw_successors_generated = counts.raw_successors_generated;
            outcome.states_generated = counts.states_generated;
            match validate_tlc_completion(code, &combined, counts, parse_err.as_deref()) {
                Ok(()) => outcome.status = "pass".into(),
                Err((error_type, error)) => {
                    outcome.status = "error".into();
                    outcome.error_type = Some(error_type);
                    outcome.error = Some(error);
                }
            }
        }
        Err(e) => {
            outcome.status = "error".into();
            let mut msg = format!("{e}");
            if msg.len() > 200 {
                msg.truncate(200);
            }
            outcome.error = Some(msg);
        }
    }

    outcome
}

fn parse_tlc_output(output: &str) -> (TlcParsedCounts, Option<String>) {
    let mut counts = TlcParsedCounts::default();

    for line in output.lines() {
        if let Some(initial) = parse_tlc_initial_generated(line.trim()) {
            counts.raw_initial_states_generated = Some(initial);
        }
    }

    // Pattern 1: "N state(s) generated, M distinct state(s) found,
    // L state(s) left ..."
    for line in output.lines() {
        let trimmed = line.trim_start();
        for (generated_marker, distinct_marker) in [
            ("states generated,", "distinct states found,"),
            ("states generated,", "distinct state found,"),
            ("state generated,", "distinct states found,"),
            ("state generated,", "distinct state found,"),
        ] {
            if let Some((generated, distinct, states_left)) =
                split_states_generated_distinct_left(trimmed, generated_marker, distinct_marker)
            {
                counts.states_generated = Some(generated);
                counts.states = Some(distinct);
                counts.states_left = Some(states_left);
                break;
            }
        }
    }

    // Pattern 2: "N distinct state(s) found"
    if counts.states.is_none() {
        counts.states = parse_last_count_before(output, "distinct states found")
            .or_else(|| parse_last_count_before(output, "distinct state found"));
    }

    if let (Some(generated), Some(initial)) =
        (counts.states_generated, counts.raw_initial_states_generated)
    {
        counts.raw_successors_generated = generated.checked_sub(initial);
        if counts.raw_successors_generated.is_none() {
            return (
                counts,
                Some(format!(
                    "total generated-state count {generated} is smaller than raw initial-state count {initial}"
                )),
            );
        }
    }

    if counts.complete() {
        return (counts, None);
    }

    // Pattern 3: "Cannot find source file for module Foo"
    if let Some(idx) = output.find("Cannot find source file for module ") {
        let tail = &output[idx + "Cannot find source file for module ".len()..];
        let module: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !module.is_empty() {
            return (counts, Some(format!("missing_module:{module}")));
        }
    }

    let mut missing = Vec::new();
    if counts.states.is_none() {
        missing.push("distinct_states");
    }
    if counts.raw_initial_states_generated.is_none() {
        missing.push("raw_initial_states_generated");
    }
    if counts.raw_successors_generated.is_none() {
        missing.push("raw_successors_generated");
    }
    if counts.states_generated.is_none() {
        missing.push("states_generated");
    }
    match counts.states_left {
        Some(0) => {}
        Some(states_left) => {
            return (
                counts,
                Some(format!(
                    "TLC completion summary reports {states_left} state(s) left on the queue"
                )),
            );
        }
        None => missing.push("states_left"),
    }
    (
        counts,
        Some(format!(
            "missing TLC count field(s): {}",
            missing.join(", ")
        )),
    )
}

/// Match `"N states generated, M distinct states found, L states left"` and
/// return the total-generated, distinct, and queue-left counts.
fn split_states_generated_distinct_left(
    line: &str,
    sep_generated: &str,
    sep_distinct: &str,
) -> Option<(u64, u64, u64)> {
    let gen_idx = line.find(sep_generated)?;
    let generated_token = line[..gen_idx].trim();
    let generated_cleaned: String = generated_token.chars().filter(|c| *c != ',').collect();
    let after_gen = &line[gen_idx + sep_generated.len()..];
    let dist_idx = after_gen.find(sep_distinct)?;
    let distinct_token = after_gen[..dist_idx].trim();
    let distinct_cleaned: String = distinct_token.chars().filter(|c| *c != ',').collect();
    let after_distinct = after_gen[dist_idx + sep_distinct.len()..].trim_start();
    let left_token_end = after_distinct
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit() || *ch == ',')
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()?;
    let left_token: String = after_distinct[..left_token_end]
        .chars()
        .filter(|ch| *ch != ',')
        .collect();
    let left_suffix = after_distinct[left_token_end..].trim_start();
    if !left_suffix.starts_with("state left") && !left_suffix.starts_with("states left") {
        return None;
    }
    Some((
        generated_cleaned.parse::<u64>().ok()?,
        distinct_cleaned.parse::<u64>().ok()?,
        left_token.parse::<u64>().ok()?,
    ))
}

fn parse_tlc_initial_generated(line: &str) -> Option<u64> {
    let tail = line
        .strip_prefix("Finished computing initial states:")?
        .trim_start();
    let describes_initial_generation = tail.contains(" distinct state generated")
        || tail.contains(" distinct states generated")
        || tail.contains(" state generated, with ")
        || tail.contains(" states generated, with ");
    if !describes_initial_generation {
        return None;
    }
    let token: String = tail
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == ',')
        .filter(|ch| *ch != ',')
        .collect();
    token.parse::<u64>().ok()
}

fn parse_last_count_before(output: &str, marker: &str) -> Option<u64> {
    let mut found = None;
    let mut search_start = 0;
    while let Some(idx) = output[search_start..].find(marker) {
        let absolute = search_start + idx;
        let head = &output[..absolute];
        let trimmed = head.trim_end();
        let digit_start = trimmed
            .char_indices()
            .rev()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == ','))
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        let token = &trimmed[digit_start..];
        let cleaned: String = token.chars().filter(|c| *c != ',').collect();
        if let Ok(value) = cleaned.parse::<u64>() {
            found = Some(value);
        }
        search_start = absolute + marker.len();
    }
    found
}

fn classify_error(output: &str) -> Option<String> {
    if !output.contains("Error:") {
        return None;
    }
    if output.contains("Invariant") && output.contains("violated") {
        return Some("invariant".into());
    }
    if output.contains("Deadlock") {
        return Some("deadlock".into());
    }
    if output.contains("Parsing or semantic analysis failed") {
        return Some("parse".into());
    }
    if output.contains("Temporal properties were violated") {
        return Some("liveness".into());
    }
    if output.contains("Action property") && output.contains("violated") {
        return Some("action".into());
    }
    // "Property <name> is violated"
    for line in output.lines() {
        if line.contains("Property ") && line.contains(" is violated") {
            return Some("safety".into());
        }
    }
    Some("unknown".into())
}

const TLC_SUCCESS_MARKER: &str = "Model checking completed. No error has been found.";

fn validate_tlc_completion(
    exit_code: i32,
    output: &str,
    counts: TlcParsedCounts,
    parse_error: Option<&str>,
) -> std::result::Result<(), (String, String)> {
    if let Some(module) = parse_error.and_then(|error| error.strip_prefix("missing_module:")) {
        return Err(("missing_module".into(), format!("Missing module: {module}")));
    }
    if let Some(error_type) = classify_error(output) {
        return Err((
            error_type,
            first_tlc_failure_line(output).unwrap_or_else(|| "TLC reported an error".into()),
        ));
    }
    if exit_code != 0 {
        return Err((
            "process_exit".into(),
            first_tlc_failure_line(output)
                .unwrap_or_else(|| format!("TLC exited with status {exit_code}")),
        ));
    }
    if !output.contains(TLC_SUCCESS_MARKER) {
        return Err((
            "incomplete_completion".into(),
            format!("missing exact TLC success marker: {TLC_SUCCESS_MARKER}"),
        ));
    }
    if !counts.complete() {
        return Err((
            "count_parse".into(),
            parse_error
                .unwrap_or(
                    "TLC success output did not contain a complete empty-queue count summary",
                )
                .to_string(),
        ));
    }
    Ok(())
}

fn first_tlc_failure_line(output: &str) -> Option<String> {
    output
        .lines()
        .find(|line| line.contains("Error:") || line.contains("Exception"))
        .map(|line| truncate(line.trim(), 200))
}

fn cleanup_states_dir(states_dir: &Path) {
    // Safety guard: only ever remove a directory named "states" under a spec dir.
    if states_dir.file_name().and_then(|n| n.to_str()) != Some("states") {
        return;
    }
    if states_dir.is_symlink() {
        let _ = fs::remove_file(states_dir);
    } else if states_dir.is_dir() {
        let _ = fs::remove_dir_all(states_dir);
    }
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<(i32, bool, String, String)> {
    let mut child = cmd.spawn().context("spawn java tlc child")?;
    let mut child_stdout = child
        .stdout
        .take()
        .context("TLC child stdout was not piped")?;
    let mut child_stderr = child
        .stderr
        .take()
        .context("TLC child stderr was not piped")?;
    // Drain both pipes while TLC runs. Waiting for process exit before reading
    // can deadlock once either finite kernel pipe buffer fills.
    let stdout_reader = std::thread::spawn(move || {
        let mut output = String::new();
        child_stdout.read_to_string(&mut output)?;
        std::io::Result::Ok(output)
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut output = String::new();
        child_stderr.read_to_string(&mut output)?;
        std::io::Result::Ok(output)
    });

    let start = Instant::now();
    let (exit_code, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status.code().unwrap_or(-1), false),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break (-1, true);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(error).context("wait for TLC child");
            }
        }
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("TLC stdout reader thread panicked"))?
        .context("read TLC stdout")?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("TLC stderr reader thread panicked"))?
        .context("read TLC stderr")?;
    Ok((exit_code, timed_out, stdout, stderr))
}

fn categorize_runtime(seconds: Option<f64>) -> &'static str {
    match seconds {
        None => "unknown",
        Some(s) if s < 1.0 => "small",
        Some(s) if s < 30.0 => "medium",
        Some(s) if s < 300.0 => "large",
        Some(_) => "xlarge",
    }
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

// ---------- Provenance ----------

fn build_provenance(
    timeout_seconds: u64,
    ctx: &PathContext,
    manifest: &CorpusManifest,
) -> Map<String, Value> {
    let has_community = ctx.community_modules.exists();
    let mut collector = Map::new();
    collector.insert(
        "ty_git_commit".into(),
        Value::String(git_short_head(&ctx.project_root)),
    );
    collector.insert(
        "script".into(),
        Value::String("crates/tla-petri/src/bin/ty-tlc-baseline.rs".into()),
    );
    collector.insert(
        "script_sha256".into(),
        Value::String(sha256_file(
            &ctx.project_root
                .join("crates/tla-petri/src/bin/ty-tlc-baseline.rs"),
        )),
    );
    collector.insert(
        "cargo_lock_sha256".into(),
        Value::String(sha256_file(&ctx.project_root.join("Cargo.lock"))),
    );
    let collector_binary = std::env::current_exe().ok();
    collector.insert(
        "binary_path".into(),
        collector_binary
            .as_ref()
            .map(|path| Value::String(path.display().to_string()))
            .unwrap_or(Value::Null),
    );
    collector.insert(
        "binary_sha256".into(),
        collector_binary
            .as_ref()
            .map(|path| Value::String(sha256_file(path)))
            .unwrap_or(Value::Null),
    );
    collector.insert(
        "manifest".into(),
        Value::String(ctx.manifest.display().to_string()),
    );
    collector.insert(
        "manifest_sha256".into(),
        Value::String(sha256_file(&ctx.manifest)),
    );

    let mut tlc = Map::new();
    tlc.insert(
        "jar_path".into(),
        Value::String(ctx.tlc_jar.display().to_string()),
    );
    tlc.insert(
        "jar_sha256".into(),
        Value::String(sha256_file(&ctx.tlc_jar)),
    );
    tlc.insert(
        "community_modules_path".into(),
        if has_community {
            Value::String(ctx.community_modules.display().to_string())
        } else {
            Value::Null
        },
    );
    tlc.insert(
        "community_modules_sha256".into(),
        if has_community {
            Value::String(sha256_file(&ctx.community_modules))
        } else {
            Value::Null
        },
    );
    tlc.insert(
        "tlc_version".into(),
        Value::String(tlc_version(&ctx.tlc_jar)),
    );
    tlc.insert("java_version".into(), Value::String(java_version()));
    tlc.insert("jvm_args".into(), json!(["-Xmx4g"]));
    tlc.insert("workers".into(), json!(1));

    let mut inputs = Map::new();
    inputs.insert(
        "examples_dir".into(),
        Value::String(ctx.examples_dir.display().to_string()),
    );
    inputs.insert("examples_git".into(), git_info(&ctx.examples_base_dir));
    inputs.insert("tlaplus_git".into(), git_info(&ctx.tlaplus_dir));
    inputs.insert(
        "strict_corpus".into(),
        json!({
            "manifest_schema_version": manifest.schema_version,
            "repository": manifest.source.repository,
            "commit": manifest.source.commit,
            "root": manifest.source.root,
            "total_rows": manifest.rows.len(),
            "eligible_rows": manifest.rows.len() - manifest.eligibility.exclusions.len(),
            "excluded_rows": manifest.eligibility.exclusions.len(),
            "work_equivalence_policy_schema_version":
                manifest.work_equivalence_policy.schema_version,
            "default_eligible_work_equivalence_rule_id":
                manifest.work_equivalence_policy.default_eligible_rule_id,
        }),
    );

    let mut seed = Map::new();
    seed.insert("enabled".into(), Value::Bool(false));
    seed.insert("policy".into(), Value::String("no_seed".into()));
    seed.insert("source_path".into(), Value::Null);

    let mut prov = Map::new();
    prov.insert("schema_version".into(), json!(SCHEMA_VERSION));
    prov.insert("collector".into(), Value::Object(collector));
    prov.insert("tlc".into(), Value::Object(tlc));
    prov.insert("inputs".into(), Value::Object(inputs));
    prov.insert("seed".into(), Value::Object(seed));
    prov.insert("tlc_timeout_seconds".into(), json!(timeout_seconds));
    prov
}

fn git_info(repo: &Path) -> Value {
    let mut out = Map::new();
    out.insert("head".into(), Value::String("unknown".into()));
    out.insert("head_short".into(), Value::String("unknown".into()));
    out.insert("is_dirty".into(), Value::Null);
    out.insert("status_porcelain_sha256".into(), Value::Null);

    if !repo.exists() || !repo.join(".git").exists() {
        return Value::Object(out);
    }
    if let Some(head) = git_capture(repo, &["rev-parse", "HEAD"]) {
        out.insert("head".into(), Value::String(head));
    }
    if let Some(short) = git_capture(repo, &["rev-parse", "--short", "HEAD"]) {
        out.insert("head_short".into(), Value::String(short));
    }
    if let Some(status) = git_capture_raw(repo, &["status", "--porcelain=v1"]) {
        let is_dirty = !status.trim().is_empty();
        out.insert("is_dirty".into(), Value::Bool(is_dirty));
        let mut hasher = Sha256::new();
        hasher.update(status.as_bytes());
        let digest = format!("{:x}", hasher.finalize());
        out.insert(
            "status_porcelain_sha256".into(),
            Value::String(digest[..16.min(digest.len())].to_string()),
        );
    }
    Value::Object(out)
}

fn git_capture(repo: &Path, args: &[&str]) -> Option<String> {
    let out = git_capture_raw(repo, args)?;
    Some(out.trim().to_string())
}

fn git_capture_raw(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn git_short_head(repo: &Path) -> String {
    git_capture(repo, &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into())
}

fn sha256_file(path: &Path) -> String {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return "unknown".into(),
    };
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!("{:x}", hasher.finalize())
}

fn tlc_version(jar: &Path) -> String {
    if !jar.exists() {
        return "unknown".into();
    }
    let output = match Command::new("java")
        .arg("-jar")
        .arg(jar)
        .arg("-version")
        .output()
    {
        Ok(o) => o,
        Err(_) => return "unknown".into(),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if let Some(v) = extract_tlc_version(&text) {
        return v;
    }
    let trimmed = text.trim();
    if trimmed.is_empty() {
        "unknown".into()
    } else {
        truncate(trimmed, 50)
    }
}

fn extract_tlc_version(text: &str) -> Option<String> {
    // Look for "TLC Version X.Y.Z" or "TLC2 Version X.Y.Z" (case-insensitive).
    let lower = text.to_ascii_lowercase();
    for marker in ["tlc2 version", "tlc version"] {
        if let Some(idx) = lower.find(marker) {
            let tail = &text[idx + marker.len()..];
            if let Some(v) = first_dotted_triple(tail) {
                return Some(v);
            }
        }
    }
    first_dotted_triple(text)
}

fn first_dotted_triple(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let mut j = i;
            let mut dots = 0;
            let mut last_was_digit = false;
            while j < bytes.len() {
                let c = bytes[j];
                if c.is_ascii_digit() {
                    last_was_digit = true;
                    j += 1;
                } else if c == b'.' && last_was_digit {
                    dots += 1;
                    last_was_digit = false;
                    j += 1;
                } else {
                    break;
                }
            }
            if dots == 2 && last_was_digit {
                return Some(text[i..j].to_string());
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    None
}

fn java_version() -> String {
    let output = match Command::new("java").arg("-version").output() {
        Ok(o) => o,
        Err(_) => return "unknown".into(),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    if let Some(start) = text.find("version \"") {
        let tail = &text[start + "version \"".len()..];
        if let Some(end) = tail.find('"') {
            return tail[..end].to_string();
        }
    }
    let first_line = text.lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        "unknown".into()
    } else {
        truncate(first_line, 50)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    s.char_indices()
        .take_while(|(i, _)| *i < max)
        .map(|(_, c)| c)
        .collect()
}

// ---------- Baseline schema ----------

fn make_untested_ty_entry() -> Value {
    json!({
        "status": "untested",
        "states": Value::Null,
        "raw_initial_states_generated": Value::Null,
        "raw_successors_generated": Value::Null,
        "states_generated": Value::Null,
        "error_type": Value::Null,
        "last_run": Value::Null,
        "git_commit": Value::Null,
    })
}

fn exhaustive_work_equivalence_entry() -> Value {
    json!({
        "schema_version": WORK_EQUIVALENCE_POLICY_SCHEMA_VERSION,
        "rule_id": EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID,
    })
}

fn load_existing_output(path: &Path) -> Option<Value> {
    let text = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    if value.get("specs").and_then(Value::as_object).is_some() {
        Some(value)
    } else {
        None
    }
}

fn resume_output_matches_provenance(existing: &Value, provenance: &Map<String, Value>) -> bool {
    let specs_digest_matches = existing
        .get("specs")
        .and_then(|specs| sha256_jcs(specs).ok())
        .zip(
            existing
                .get("specs_jcs_sha256")
                .and_then(Value::as_str)
                .map(str::to_owned),
        )
        .is_some_and(|(actual, recorded)| actual == recorded);
    specs_digest_matches
        && existing.get("schema_version").and_then(Value::as_u64) == Some(SCHEMA_VERSION)
        && ["collector", "tlc", "inputs", "seed", "tlc_timeout_seconds"]
            .into_iter()
            .all(|key| existing.get(key) == provenance.get(key))
}

fn order_spec_entry(entry: Value) -> Value {
    let mut entry_obj = match entry {
        Value::Object(map) => map,
        _ => return Value::Null,
    };

    // Schema-v2 migration: if "tlc" is missing or not an object, lift the
    // legacy flat keys (`expected_states`, `tlc_runtime_seconds`, etc.)
    // into a nested `tlc` object.
    let needs_migration = entry_obj.get("tlc").map(|v| !v.is_object()).unwrap_or(true);
    if needs_migration {
        let mut tlc = Map::new();
        tlc.insert(
            "status".into(),
            entry_obj
                .get("status")
                .cloned()
                .unwrap_or_else(|| Value::String("unknown".into())),
        );
        tlc.insert(
            "states".into(),
            entry_obj
                .get("expected_states")
                .cloned()
                .unwrap_or(Value::Null),
        );
        tlc.insert(
            "runtime_seconds".into(),
            entry_obj
                .get("tlc_runtime_seconds")
                .cloned()
                .unwrap_or(Value::Null),
        );
        tlc.insert(
            "error_type".into(),
            entry_obj.get("error_type").cloned().unwrap_or(Value::Null),
        );
        tlc.insert(
            "error".into(),
            entry_obj.get("error").cloned().unwrap_or(Value::Null),
        );

        let category = entry_obj
            .get("category")
            .cloned()
            .unwrap_or_else(|| Value::String("unknown".into()));
        let source = entry_obj.get("source").cloned();

        let mut new_entry = Map::new();
        new_entry.insert("tlc".into(), Value::Object(tlc));
        new_entry.insert("ty".into(), make_untested_ty_entry());
        new_entry.insert("verified_match".into(), Value::Bool(false));
        new_entry.insert("category".into(), category);
        if let Some(src) = source {
            new_entry.insert("source".into(), src);
        }

        for (k, v) in entry_obj.into_iter() {
            if new_entry.contains_key(&k) {
                continue;
            }
            if LEGACY_V2_KEYS.iter().any(|legacy| *legacy == k) {
                continue;
            }
            new_entry.insert(k, v);
        }
        entry_obj = new_entry;
    }

    let mut result = Map::new();
    for key in SPEC_ENTRY_KEY_ORDER {
        if let Some(value) = entry_obj.remove(*key) {
            match (*key, &value) {
                ("tlc", Value::Object(_)) => {
                    let Value::Object(inner) = value else {
                        unreachable!()
                    };
                    result.insert(
                        (*key).into(),
                        Value::Object(order_inner(inner, TLC_ENTRY_KEY_ORDER)),
                    );
                }
                ("ty", Value::Object(_)) => {
                    let Value::Object(inner) = value else {
                        unreachable!()
                    };
                    result.insert(
                        (*key).into(),
                        Value::Object(order_inner(inner, TY_ENTRY_KEY_ORDER)),
                    );
                }
                _ => {
                    result.insert((*key).into(), value);
                }
            }
        }
    }
    let mut leftover_keys: Vec<String> = entry_obj.keys().cloned().collect();
    leftover_keys.sort();
    for key in leftover_keys {
        if let Some(value) = entry_obj.remove(&key) {
            result.insert(key, value);
        }
    }
    Value::Object(result)
}

fn order_inner(mut inner: Map<String, Value>, order: &[&str]) -> Map<String, Value> {
    let mut out = Map::new();
    for key in order {
        if let Some(value) = inner.remove(*key) {
            out.insert((*key).into(), value);
        }
    }
    let mut leftover: Vec<String> = inner.keys().cloned().collect();
    leftover.sort();
    for key in leftover {
        if let Some(value) = inner.remove(&key) {
            out.insert(key, value);
        }
    }
    out
}

fn compute_stats(specs: &Map<String, Value>) -> Map<String, Value> {
    let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
    for key in STATS_KEY_ORDER {
        counts.insert(key, 0);
    }
    for data in specs.values() {
        let tlc_status = data
            .get("tlc")
            .and_then(Value::as_object)
            .and_then(|m| m.get("status"))
            .and_then(Value::as_str)
            .or_else(|| data.get("status").and_then(Value::as_str))
            .unwrap_or("unknown");
        match tlc_status {
            "pass" => *counts.entry("tlc_pass").or_default() += 1,
            "timeout" => *counts.entry("tlc_timeout").or_default() += 1,
            "unsupported" => *counts.entry("tlc_unsupported").or_default() += 1,
            "uncollected" => *counts.entry("tlc_uncollected").or_default() += 1,
            _ => *counts.entry("tlc_error").or_default() += 1,
        }

        let Some(ty) = data.get("ty").and_then(Value::as_object) else {
            *counts.entry("ty_untested").or_default() += 1;
            continue;
        };
        let ty_status = ty
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("untested");
        let verified = data
            .get("verified_match")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        match ty_status {
            "pass" if verified => *counts.entry("ty_match").or_default() += 1,
            "mismatch" => *counts.entry("ty_mismatch").or_default() += 1,
            "fail" => *counts.entry("ty_fail").or_default() += 1,
            _ => *counts.entry("ty_untested").or_default() += 1,
        }
    }
    let mut out = Map::new();
    for key in STATS_KEY_ORDER {
        out.insert((*key).into(), json!(*counts.get(key).unwrap_or(&0)));
    }
    out
}

fn compute_categories(specs: &Map<String, Value>) -> Map<String, Value> {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for data in specs.values() {
        let cat = data
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        *counts.entry(cat).or_default() += 1;
    }
    let mut out = Map::new();
    for key in CATEGORIES_KEY_ORDER {
        let value = counts.remove(*key).unwrap_or(0);
        out.insert((*key).into(), json!(value));
    }
    let mut leftover: Vec<String> = counts.keys().cloned().collect();
    leftover.sort();
    for key in leftover {
        let value = counts.remove(&key).unwrap_or(0);
        out.insert(key, json!(value));
    }
    out
}

fn complete_raw_generated_counts(tlc: &Map<String, Value>) -> bool {
    let Some(raw_initial) = tlc
        .get("raw_initial_states_generated")
        .and_then(Value::as_u64)
    else {
        return false;
    };
    let Some(raw_successors) = tlc.get("raw_successors_generated").and_then(Value::as_u64) else {
        return false;
    };
    let Some(total) = tlc.get("states_generated").and_then(Value::as_u64) else {
        return false;
    };
    raw_initial.checked_add(raw_successors) == Some(total)
}

fn validate_baselines(specs: &Map<String, Value>) -> Vec<String> {
    let mut warnings = Vec::new();
    for (name, data) in specs {
        let Some(tlc) = data.get("tlc").and_then(Value::as_object) else {
            continue;
        };
        let status = tlc
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if status == "pass" && tlc.get("states").and_then(Value::as_u64).is_none() {
            warnings.push(format!(
                "{name}: TLC status=pass but states=null — state count not parsed"
            ));
        }
        if status == "pass" && !complete_raw_generated_counts(tlc) {
            warnings.push(format!(
                "{name}: TLC status=pass but raw initial + raw successor != total generated count"
            ));
        }
        let error_type = tlc.get("error_type").and_then(Value::as_str);
        if status == "pass" && matches!(error_type, Some(et) if et != "unknown") {
            warnings.push(format!(
                "{name}: TLC status=pass but error_type={} — status should likely be 'error'",
                error_type.unwrap()
            ));
        }
    }
    warnings
}

fn build_ordered_specs(
    baselines: BTreeMap<String, Value>,
    catalog: &[SpecInfo],
) -> Map<String, Value> {
    let mut work = baselines;
    let mut result = Map::new();
    for spec in catalog {
        if let Some(value) = work.remove(&spec.name) {
            result.insert(spec.name.clone(), order_spec_entry(value));
        }
    }
    result
}

// ---------- JCS digest (matches scripts/json_jcs.py) ----------

fn sha256_jcs(value: &Value) -> Result<String> {
    let mut canonical = String::new();
    write_canonical_json(value, &mut canonical)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_canonical_json(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&canonicalize_number(number)?),
        Value::String(text) => out.push_str(&serde_json::to_string(text)?),
        Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(a, _)| *a);
            out.push('{');
            for (idx, (key, item)) in entries.into_iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key)?);
                out.push(':');
                write_canonical_json(item, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn canonicalize_number(number: &serde_json::Number) -> Result<String> {
    if let Some(value) = number.as_i64() {
        return Ok(value.to_string());
    }
    if let Some(value) = number.as_u64() {
        return Ok(value.to_string());
    }
    if let Some(value) = number.as_f64() {
        return format_float_jcs(value);
    }
    bail!("unsupported JSON number for canonicalization: {number}")
}

fn format_float_jcs(value: f64) -> Result<String> {
    if !value.is_finite() {
        bail!("non-finite float not allowed in canonical JSON: {value:?}");
    }
    if value == 0.0 {
        return Ok("0".to_string());
    }
    let mut formatted = format!("{value:?}");
    if let Some(exp_index) = formatted.find(['e', 'E']) {
        let mantissa = formatted[..exp_index].to_string();
        let exponent = &formatted[exp_index + 1..];
        let (sign, digits) = match exponent.as_bytes().first().copied() {
            Some(b'+') => ("+", &exponent[1..]),
            Some(b'-') => ("-", &exponent[1..]),
            _ => ("", exponent),
        };
        let digits = digits.trim_start_matches('0');
        let digits = if digits.is_empty() { "0" } else { digits };
        return Ok(format!("{mantissa}e{sign}{digits}"));
    }
    if formatted.contains('.') {
        while formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.ends_with('.') {
            formatted.pop();
        }
    }
    Ok(formatted)
}

// ---------- Output writer ----------

fn write_output(
    path: &Path,
    baselines: BTreeMap<String, Value>,
    provenance: &Map<String, Value>,
    catalog: &[SpecInfo],
) -> Result<Map<String, Value>> {
    let ordered_specs = build_ordered_specs(baselines, catalog);
    for warning in validate_baselines(&ordered_specs) {
        eprintln!("WARNING: baseline anomaly: {warning}");
    }
    let stats = compute_stats(&ordered_specs);
    let categories = compute_categories(&ordered_specs);
    let specs_value = Value::Object(ordered_specs.clone());
    let specs_jcs = sha256_jcs(&specs_value)?;

    let mut output = Map::new();
    output.insert(
        "schema_version".into(),
        provenance
            .get("schema_version")
            .cloned()
            .unwrap_or_else(|| json!(SCHEMA_VERSION)),
    );
    output.insert("generated".into(), Value::String(now_iso_local()));
    output.insert(
        "collector".into(),
        provenance.get("collector").cloned().unwrap_or_default(),
    );
    output.insert(
        "tlc".into(),
        provenance.get("tlc").cloned().unwrap_or_default(),
    );
    output.insert(
        "inputs".into(),
        provenance.get("inputs").cloned().unwrap_or_default(),
    );
    output.insert(
        "seed".into(),
        provenance.get("seed").cloned().unwrap_or_default(),
    );
    output.insert(
        "tlc_timeout_seconds".into(),
        provenance
            .get("tlc_timeout_seconds")
            .cloned()
            .unwrap_or_else(|| json!(DEFAULT_TIMEOUT_SECONDS)),
    );
    output.insert("total_specs".into(), json!(catalog.len()));
    output.insert(
        "eligible_specs".into(),
        json!(catalog
            .iter()
            .filter(|spec| spec.exclusion.is_none())
            .count()),
    );
    output.insert(
        "excluded_specs".into(),
        json!(catalog
            .iter()
            .filter(|spec| spec.exclusion.is_some())
            .count()),
    );
    output.insert("specs_jcs_sha256".into(), Value::String(specs_jcs));
    output.insert("stats".into(), Value::Object(stats));
    output.insert("categories".into(), Value::Object(categories));
    output.insert("specs".into(), specs_value);

    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let text = serde_json::to_string_pretty(&Value::Object(output.clone()))?;
    fs::write(&tmp_path, text).with_context(|| format!("writing {}", tmp_path.display()))?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("renaming {} -> {}", tmp_path.display(), path.display()))?;
    Ok(output)
}

fn now_iso_local() -> String {
    // Mirrors Python's `datetime.now().isoformat()` shape with UTC time
    // (the local-naive baseline format only used the wall-clock time, so
    // emitting UTC keeps the output reproducible across machines).
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let micros = now.subsec_micros();
    let days = (secs / 86_400) as i64;
    let secs_of_day = (secs % 86_400) as u32;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}.{micros:06}")
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    // Howard Hinnant's days-to-civil algorithm.
    let z = z + 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = (y + i64::from(m <= 2)) as i32;
    (y, m, d)
}

// ---------- Path discovery ----------

struct PathContext {
    project_root: PathBuf,
    examples_dir: PathBuf,
    tlc_jar: PathBuf,
    community_modules: PathBuf,
    tlaplus_dir: PathBuf,
    examples_base_dir: PathBuf,
    output: PathBuf,
    manifest: PathBuf,
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

impl PathContext {
    fn from_cli(cli: &Cli) -> PathContext {
        let project_root = cli
            .project_root
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let tlaplus_dir = cli
            .tlaplus_dir
            .clone()
            .unwrap_or_else(|| home().join("tlaplus"));
        let examples_dir = cli.examples_dir.clone().unwrap_or_else(|| {
            cli.examples_base_dir
                .clone()
                .unwrap_or_else(|| home().join("tlaplus-examples"))
                .join("specifications")
        });
        let examples_base_dir = cli.examples_base_dir.clone().unwrap_or_else(|| {
            cli.examples_dir
                .as_ref()
                .and_then(|path| path.parent())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| home().join("tlaplus-examples"))
        });
        let tlc_jar = cli
            .tlc_jar
            .clone()
            .unwrap_or_else(|| tlaplus_dir.join("tytools.jar"));
        let community_modules = cli
            .community_modules
            .clone()
            .unwrap_or_else(|| tlaplus_dir.join("CommunityModules.jar"));
        let output = cli.output.clone().unwrap_or_else(|| {
            project_root
                .join("tests")
                .join("tlc_comparison")
                .join("spec_baseline.json")
        });
        let manifest = cli.manifest.clone().unwrap_or_else(|| {
            project_root
                .join("tests")
                .join("tlc_comparison")
                .join("strict_corpus_manifest.json")
        });
        PathContext {
            project_root,
            examples_dir,
            tlc_jar,
            community_modules,
            tlaplus_dir,
            examples_base_dir,
            output,
            manifest,
        }
    }
}

fn print_catalog(catalog: &[SpecInfo]) {
    println!("eligibility\tname\tcfg_path\ttla_path\treason_code");
    for spec in catalog {
        let (eligibility, reason_code) = match &spec.exclusion {
            Some(exclusion) => ("excluded", exclusion.reason_code.as_str()),
            None => ("eligible", ""),
        };
        println!(
            "{eligibility}\t{}\t{}\t{}\t{reason_code}",
            spec.name, spec.cfg_path, spec.tla_path
        );
    }
}

fn verify_collection_source(
    ctx: &PathContext,
    manifest: &CorpusManifest,
    catalog: &[SpecInfo],
) -> Result<()> {
    let git_dir = ctx.examples_base_dir.join(".git");
    if !git_dir.exists() {
        bail!(
            "strict collection requires a Git checkout at {}; use a detached worktree at {}",
            ctx.examples_base_dir.display(),
            manifest.source.commit
        );
    }
    let actual_head = git_capture(&ctx.examples_base_dir, &["rev-parse", "HEAD"])
        .context("read examples checkout HEAD")?;
    if actual_head != manifest.source.commit {
        bail!(
            "examples checkout is at {actual_head}, but strict corpus requires {}; \
             use a separate detached worktree rather than changing the existing checkout",
            manifest.source.commit
        );
    }
    let status = git_capture_raw(&ctx.examples_base_dir, &["status", "--porcelain=v1"])
        .context("read examples checkout status")?;
    if !status.trim().is_empty() {
        bail!(
            "examples checkout at {} is dirty; strict collection requires the clean pinned tree",
            ctx.examples_base_dir.display()
        );
    }
    if !ctx.examples_dir.is_dir() {
        bail!(
            "examples source root is missing: {}",
            ctx.examples_dir.display()
        );
    }

    let mut actual_cfg_paths = Vec::new();
    collect_relative_cfg_paths(&ctx.examples_dir, &ctx.examples_dir, &mut actual_cfg_paths)?;
    actual_cfg_paths.sort();
    let expected_cfg_paths: Vec<&str> = catalog.iter().map(|spec| spec.cfg_path.as_str()).collect();
    if actual_cfg_paths.len() != expected_cfg_paths.len() {
        bail!(
            "pinned source has {} cfg files, manifest has {}",
            actual_cfg_paths.len(),
            expected_cfg_paths.len()
        );
    }
    for (actual, expected) in actual_cfg_paths.iter().zip(expected_cfg_paths) {
        if actual != expected {
            bail!("pinned cfg set differs from manifest: found {actual}, expected {expected}");
        }
    }
    for spec in catalog {
        let tla_path = ctx.examples_dir.join(&spec.tla_path);
        if !tla_path.is_file() {
            bail!(
                "manifest row {} maps to missing TLA module {}",
                spec.name,
                tla_path.display()
            );
        }
    }
    Ok(())
}

fn collect_relative_cfg_paths(root: &Path, dir: &Path, output: &mut Vec<String>) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("reading corpus directory {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("reading entries under {}", dir.display()))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("reading file type for {}", path.display()))?;
        if file_type.is_dir() {
            collect_relative_cfg_paths(root, &path, output)?;
        } else if file_type.is_file()
            && path.extension().and_then(|value| value.to_str()) == Some("cfg")
        {
            let relative = path
                .strip_prefix(root)
                .with_context(|| format!("relativizing {}", path.display()))?;
            output.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

fn skeleton_entry(spec: &SpecInfo) -> Value {
    let (tlc_status, error_type, error, category, eligibility) = match &spec.exclusion {
        Some(exclusion) => (
            "unsupported",
            Value::String(exclusion.reason_code.clone()),
            Value::String(exclusion.detail.clone()),
            "skip",
            "excluded",
        ),
        None => (
            "uncollected",
            Value::Null,
            Value::Null,
            "unknown",
            "eligible",
        ),
    };
    let mut entry = Map::new();
    entry.insert(
        "tlc".into(),
        json!({
            "status": tlc_status,
            "states": Value::Null,
            "raw_initial_states_generated": Value::Null,
            "raw_successors_generated": Value::Null,
            "states_generated": Value::Null,
            "runtime_seconds": Value::Null,
            "error_type": error_type,
            "error": error,
        }),
    );
    entry.insert("ty".into(), make_untested_ty_entry());
    entry.insert("verified_match".into(), Value::Bool(false));
    entry.insert("eligibility".into(), Value::String(eligibility.into()));
    if let Some(exclusion) = &spec.exclusion {
        entry.insert(
            "exclusion".into(),
            json!({
                "reason_code": exclusion.reason_code,
                "detail": exclusion.detail,
            }),
        );
    } else {
        entry.insert(
            "work_equivalence".into(),
            exhaustive_work_equivalence_entry(),
        );
    }
    entry.insert("category".into(), Value::String(category.into()));
    entry.insert(
        "source".into(),
        json!({
            "tla_path": spec.tla_path,
            "cfg_path": spec.cfg_path,
        }),
    );
    order_spec_entry(Value::Object(entry))
}

fn entry_matches_manifest_source(entry: &Value, spec: &SpecInfo) -> bool {
    let source = entry.get("source").and_then(Value::as_object);
    source
        .and_then(|source| source.get("tla_path"))
        .and_then(Value::as_str)
        == Some(spec.tla_path.as_str())
        && source
            .and_then(|source| source.get("cfg_path"))
            .and_then(Value::as_str)
            == Some(spec.cfg_path.as_str())
}

fn normalize_existing_entry(entry: Value, spec: &SpecInfo) -> Value {
    let Some(mut object) = entry.as_object().cloned() else {
        return skeleton_entry(spec);
    };
    object.insert("eligibility".into(), Value::String("eligible".into()));
    object.remove("exclusion");
    object.remove("work_equivalence_rule");
    object.remove("equivalent_work_rule");
    object.remove("performance_work_equivalence_rule");
    object.insert(
        "work_equivalence".into(),
        exhaustive_work_equivalence_entry(),
    );
    object.insert(
        "source".into(),
        json!({
            "tla_path": spec.tla_path,
            "cfg_path": spec.cfg_path,
        }),
    );
    order_spec_entry(Value::Object(object))
}

fn initialize_baselines(
    catalog: &[SpecInfo],
    existing: BTreeMap<String, Value>,
) -> BTreeMap<String, Value> {
    catalog
        .iter()
        .map(|spec| {
            let entry = if spec.exclusion.is_some() {
                skeleton_entry(spec)
            } else {
                existing
                    .get(&spec.name)
                    .filter(|entry| entry_matches_manifest_source(entry, spec))
                    .cloned()
                    .map(|entry| normalize_existing_entry(entry, spec))
                    .unwrap_or_else(|| skeleton_entry(spec))
            };
            (spec.name.clone(), entry)
        })
        .collect()
}

// ---------- Main ----------

fn run(cli: Cli) -> Result<()> {
    let ctx = PathContext::from_cli(&cli);
    let timeout_seconds = cli.timeout;

    let manifest_text = fs::read_to_string(&ctx.manifest)
        .with_context(|| format!("reading {}", ctx.manifest.display()))?;
    let manifest = parse_corpus_manifest(&manifest_text)
        .with_context(|| format!("validating {}", ctx.manifest.display()))?;
    let catalog = catalog_from_manifest(&manifest);
    let eligible_count = catalog
        .iter()
        .filter(|spec| spec.exclusion.is_none())
        .count();
    let excluded_count = catalog.len() - eligible_count;

    if cli.list {
        print_catalog(&catalog);
        eprintln!(
            "normalized catalog: {} rows ({eligible_count} eligible, {excluded_count} excluded)",
            catalog.len()
        );
        return Ok(());
    }

    if cli.dry_run {
        verify_collection_source(&ctx, &manifest, &catalog)?;
        println!(
            "dry run OK: {} pinned rows ({eligible_count} eligible, {excluded_count} excluded); no TLC processes launched",
            catalog.len()
        );
        return Ok(());
    }

    let provenance = build_provenance(timeout_seconds, &ctx, &manifest);

    if cli.write_skeleton {
        let baselines = initialize_baselines(&catalog, BTreeMap::new());
        write_output(&ctx.output, baselines, &provenance, &catalog)?;
        println!(
            "Wrote normalized baseline skeleton with {} rows ({eligible_count} eligible, {excluded_count} excluded) to {}",
            catalog.len(),
            ctx.output.display()
        );
        return Ok(());
    }

    verify_collection_source(&ctx, &manifest, &catalog)?;

    println!(
        "Collecting TLC baselines for {eligible_count} eligible specs ({} excluded rows retained)...",
        excluded_count
    );
    println!("Output: {}", ctx.output.display());
    println!("Manifest: {}", ctx.manifest.display());
    println!("Timeout: {timeout_seconds}s per spec");
    let prov_collector_short = provenance
        .get("collector")
        .and_then(|v| v.get("ty_git_commit"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let tlc_version_str = provenance
        .get("tlc")
        .and_then(|v| v.get("tlc_version"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let examples_short = provenance
        .get("inputs")
        .and_then(|v| v.get("examples_git"))
        .and_then(|v| v.get("head_short"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    println!(
        "Provenance: schema_version={SCHEMA_VERSION}, tlc={tlc_version_str}, examples={examples_short}, collector={prov_collector_short}"
    );

    let existing: BTreeMap<String, Value> = if cli.no_resume {
        BTreeMap::new()
    } else if let Some(existing) = load_existing_output(&ctx.output) {
        if resume_output_matches_provenance(&existing, &provenance) {
            let specs = existing
                .get("specs")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            specs
                .into_iter()
                .filter(|(_, v)| v.is_object())
                .map(|(k, v)| (k, order_spec_entry(v)))
                .collect()
        } else {
            eprintln!(
                "Ignoring stale resume data in {}: schema or collection provenance differs",
                ctx.output.display()
            );
            BTreeMap::new()
        }
    } else {
        BTreeMap::new()
    };
    let mut baselines = initialize_baselines(&catalog, existing);
    write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;

    let total = catalog.len();
    for (idx, spec) in catalog.iter().enumerate() {
        let progress_label = if spec.name.len() > 40 {
            spec.name.chars().take(40).collect::<String>()
        } else {
            spec.name.clone()
        };
        eprint!("\r[{}/{}] {progress_label:<40}", idx + 1, total);

        if spec.exclusion.is_some() {
            continue;
        }

        let existing_entry = baselines.get(&spec.name).cloned();
        if let Some(Value::Object(existing_obj)) = &existing_entry {
            if let Some(tlc) = existing_obj.get("tlc").and_then(Value::as_object) {
                let status = tlc.get("status").and_then(Value::as_str);
                let states_present = tlc.get("states").and_then(Value::as_u64).is_some();
                let no_error = matches!(tlc.get("error_type"), None | Some(Value::Null));
                if status == Some("pass")
                    && states_present
                    && no_error
                    && complete_raw_generated_counts(tlc)
                {
                    continue;
                }
            }
        }

        let spec_path = ctx.examples_dir.join(&spec.tla_path);
        let cfg_path = ctx.examples_dir.join(&spec.cfg_path);

        if !spec_path.exists() {
            baselines.insert(
                spec.name.clone(),
                missing_entry(
                    &existing_entry,
                    spec,
                    "missing_file",
                    &format!("File not found: {}", spec.tla_path),
                ),
            );
            write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;
            continue;
        }
        if !cfg_path.exists() {
            baselines.insert(
                spec.name.clone(),
                missing_entry(
                    &existing_entry,
                    spec,
                    "missing_config",
                    &format!("Config not found: {}", spec.cfg_path),
                ),
            );
            write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;
            continue;
        }

        let outcome = run_tlc(
            &spec_path,
            &cfg_path,
            timeout_seconds,
            &ctx.tlc_jar,
            &ctx.community_modules,
            &ctx.project_root,
        );
        let category = categorize_runtime(outcome.runtime_seconds);

        let (ty_data, issue) = match existing_entry.as_ref().and_then(Value::as_object) {
            Some(existing_obj) => {
                let ty = existing_obj
                    .get("ty")
                    .filter(|v| v.is_object())
                    .cloned()
                    .unwrap_or_else(make_untested_ty_entry);
                let issue = existing_obj.get("issue").cloned();
                (ty, issue)
            }
            None => (make_untested_ty_entry(), None),
        };

        let tlc_states = outcome.states;
        let ty_states = ty_data.get("states").and_then(|v| v.as_u64());
        let ty_raw_initial_states_generated = ty_data
            .get("raw_initial_states_generated")
            .and_then(Value::as_u64);
        let ty_raw_successors_generated = ty_data
            .get("raw_successors_generated")
            .and_then(Value::as_u64);
        let ty_states_generated = ty_data.get("states_generated").and_then(Value::as_u64);
        let ty_status = ty_data
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("untested");
        let verified_match = outcome.status == "pass"
            && ty_status == "pass"
            && ty_states.is_some()
            && tlc_states.is_some()
            && ty_states == tlc_states;
        let verified_match = verified_match
            && outcome.raw_initial_states_generated.is_some()
            && outcome.raw_initial_states_generated == ty_raw_initial_states_generated
            && outcome.raw_successors_generated.is_some()
            && outcome.raw_successors_generated == ty_raw_successors_generated
            && outcome.states_generated.is_some()
            && outcome.states_generated == ty_states_generated;

        let mut tlc_entry = Map::new();
        tlc_entry.insert("status".into(), Value::String(outcome.status.clone()));
        tlc_entry.insert(
            "states".into(),
            match outcome.states {
                Some(s) => json!(s),
                None => Value::Null,
            },
        );
        tlc_entry.insert(
            "raw_initial_states_generated".into(),
            outcome
                .raw_initial_states_generated
                .map_or(Value::Null, Value::from),
        );
        tlc_entry.insert(
            "raw_successors_generated".into(),
            outcome
                .raw_successors_generated
                .map_or(Value::Null, Value::from),
        );
        tlc_entry.insert(
            "states_generated".into(),
            outcome.states_generated.map_or(Value::Null, Value::from),
        );
        tlc_entry.insert(
            "runtime_seconds".into(),
            match outcome.runtime_seconds {
                Some(rt) => json!(rt),
                None => Value::Null,
            },
        );
        tlc_entry.insert(
            "error_type".into(),
            match outcome.error_type.as_deref() {
                Some(et) => Value::String(et.into()),
                None => Value::Null,
            },
        );
        tlc_entry.insert(
            "error".into(),
            match outcome.error.as_deref() {
                Some(e) => Value::String(e.into()),
                None => Value::Null,
            },
        );

        let mut spec_entry = Map::new();
        spec_entry.insert("tlc".into(), Value::Object(tlc_entry));
        spec_entry.insert("ty".into(), ty_data);
        spec_entry.insert("verified_match".into(), Value::Bool(verified_match));
        spec_entry.insert("eligibility".into(), Value::String("eligible".into()));
        spec_entry.insert(
            "work_equivalence".into(),
            exhaustive_work_equivalence_entry(),
        );
        spec_entry.insert("category".into(), Value::String(category.into()));
        spec_entry.insert(
            "source".into(),
            json!({
                "tla_path": spec.tla_path,
                "cfg_path": spec.cfg_path,
            }),
        );
        if let Some(issue) = issue {
            spec_entry.insert("issue".into(), issue);
        }

        baselines.insert(
            spec.name.clone(),
            order_spec_entry(Value::Object(spec_entry)),
        );
        write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;
    }

    eprintln!();
    println!();
    println!("{}", "=".repeat(60));
    println!("TLC BASELINE COLLECTION SUMMARY");
    println!("{}", "=".repeat(60));
    let final_output = write_output(&ctx.output, baselines.clone(), &provenance, &catalog)?;
    let stats = final_output
        .get("stats")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let categories = final_output
        .get("categories")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let get =
        |m: &Map<String, Value>, k: &str| -> u64 { m.get(k).and_then(Value::as_u64).unwrap_or(0) };
    println!("TLC pass:    {}", get(&stats, "tlc_pass"));
    println!("TLC error:   {}", get(&stats, "tlc_error"));
    println!("TLC timeout: {}", get(&stats, "tlc_timeout"));
    println!("Excluded:    {}", get(&stats, "tlc_unsupported"));
    println!("Uncollected: {}", get(&stats, "tlc_uncollected"));
    println!();
    println!("Runtime Categories:");
    println!("  Small (<1s):     {}", get(&categories, "small"));
    println!("  Medium (<30s):   {}", get(&categories, "medium"));
    println!("  Large (<300s):   {}", get(&categories, "large"));
    println!("  XLarge (>300s):  {}", get(&categories, "xlarge"));
    println!(
        "  Skip/Unknown:    {}",
        get(&categories, "skip") + get(&categories, "unknown")
    );
    println!();
    println!("Wrote {}", ctx.output.display());

    let errors: Vec<(&String, &Value)> = final_output
        .get("specs")
        .and_then(Value::as_object)
        .map(|specs| {
            specs
                .iter()
                .filter(|(_, v)| {
                    let status = v
                        .get("tlc")
                        .and_then(Value::as_object)
                        .and_then(|m| m.get("status"))
                        .and_then(Value::as_str);
                    matches!(status, Some("error") | Some("timeout"))
                })
                .collect()
        })
        .unwrap_or_default();
    if !errors.is_empty() {
        println!();
        println!("ERRORS/TIMEOUTS:");
        for (name, data) in errors.iter().take(20) {
            let tlc = data.get("tlc").and_then(Value::as_object);
            let status = tlc
                .and_then(|m| m.get("status"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let etype = tlc
                .and_then(|m| m.get("error_type"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            println!("  {name}: {status} - {etype}");
        }
        if errors.len() > 20 {
            println!("  ... and {} more", errors.len() - 20);
        }
    }
    Ok(())
}

fn missing_entry(
    existing: &Option<Value>,
    spec: &SpecInfo,
    error_type: &str,
    message: &str,
) -> Value {
    let mut tlc = Map::new();
    tlc.insert("status".into(), Value::String("error".into()));
    tlc.insert("states".into(), Value::Null);
    tlc.insert("raw_initial_states_generated".into(), Value::Null);
    tlc.insert("raw_successors_generated".into(), Value::Null);
    tlc.insert("states_generated".into(), Value::Null);
    tlc.insert("runtime_seconds".into(), Value::Null);
    tlc.insert("error_type".into(), Value::String(error_type.into()));
    tlc.insert("error".into(), Value::String(message.into()));

    let ty = existing
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|m| m.get("ty"))
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(make_untested_ty_entry);

    let mut entry = Map::new();
    entry.insert("tlc".into(), Value::Object(tlc));
    entry.insert("ty".into(), ty);
    entry.insert("verified_match".into(), Value::Bool(false));
    entry.insert("eligibility".into(), Value::String("eligible".into()));
    entry.insert(
        "work_equivalence".into(),
        exhaustive_work_equivalence_entry(),
    );
    entry.insert("category".into(), Value::String("unknown".into()));
    entry.insert(
        "source".into(),
        json!({
            "tla_path": spec.tla_path,
            "cfg_path": spec.cfg_path,
        }),
    );
    order_spec_entry(Value::Object(entry))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:?}");
            ExitCode::FAILURE
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_tlc_counts() -> TlcParsedCounts {
        TlcParsedCounts {
            states: Some(3),
            raw_initial_states_generated: Some(1),
            raw_successors_generated: Some(2),
            states_generated: Some(3),
            states_left: Some(0),
        }
    }

    #[test]
    fn parse_manifest_resolves_override_and_exclusion() {
        let body = serde_json::to_string(&json!({
            "schema_version": 1,
            "claim": "ty_vs_tlc_strict_superiority",
            "source": {
                "repository": "https://example.invalid/Examples.git",
                "commit": "0123456789abcdef0123456789abcdef01234567",
                "root": "specifications",
                "enumeration": "all_cfg_files",
                "default_tla_mapping": "same_stem",
                "expected_cfg_count": 2
            },
            "eligibility": {
                "default": "eligible",
                "exclusions": {
                    "B/B.cfg": {
                        "reason_code": "simulation_only",
                        "detail": "simulation"
                    }
                }
            },
            "work_equivalence_policy": {
                "schema_version": 1,
                "default_eligible_rule_id": "exhaustive_generated_work_parity_v1",
                "rules": {
                    "exhaustive_generated_work_parity_v1": {
                        "kind": "exhaustive_state_space",
                        "required_verdict": "holds",
                        "require_complete_exploration": true,
                        "distinct_state_parity": "exact",
                        "raw_initial_state_generation_parity": "exact",
                        "raw_successor_generation_parity": "exact",
                        "total_state_generation_parity": "exact",
                        "count_arm": "bfs_no_reduction_single_worker"
                    }
                },
                "outcome_dispositions": {
                    "expected_violation": "exclude_unless_predeclared_typed_rule",
                    "deadlock": "exclude_unless_predeclared_typed_rule",
                    "simulation": "exclude",
                    "randomized_external_operator": "exclude",
                    "external_io": "exclude",
                    "timeout": "missing_or_stale"
                }
            },
            "tla_path_overrides": {
                "B/B.cfg": "B/Shared.tla"
            },
            "baseline_gaps": {},
            "rows": [
                {"name": "A", "cfg_path": "A/A.cfg", "tla_path": "A/A.tla"},
                {"name": "B", "cfg_path": "B/B.cfg", "tla_path": "B/Shared.tla"}
            ]
        }))
        .unwrap();
        let manifest = parse_corpus_manifest(&body).unwrap();
        let specs = catalog_from_manifest(&manifest);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "A");
        assert_eq!(specs[1].tla_path, "B/Shared.tla");
        assert_eq!(
            specs[1]
                .exclusion
                .as_ref()
                .map(|exclusion| exclusion.reason_code.as_str()),
            Some("simulation_only")
        );
    }

    #[test]
    fn checked_in_manifest_is_complete_and_includes_legacy_gaps() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/tlc_comparison/strict_corpus_manifest.json");
        if !path.exists() {
            return; // tolerate missing checkout (e.g. published crate)
        }
        let text = fs::read_to_string(&path).unwrap();
        let manifest = parse_corpus_manifest(&text).unwrap();
        let specs = catalog_from_manifest(&manifest);
        assert_eq!(specs.len(), 181);
        assert_eq!(manifest.tla_path_overrides.len(), 35);
        assert_eq!(manifest.baseline_gaps.len(), 5);
        assert_eq!(
            manifest.work_equivalence_policy.default_eligible_rule_id,
            EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID
        );
        assert_eq!(
            specs.iter().filter(|spec| spec.exclusion.is_none()).count(),
            141
        );
        for cfg_path in [
            "CheckpointCoordination/MCCheckpointCoordinationFailure.cfg",
            "DieHard/DieHard.cfg",
            "DieHard/MCDieHarder.cfg",
            "EinsteinRiddle/Einstein.cfg",
            "FiniteMonotonic/MCDistributedReplicatedLog.cfg",
            "MissionariesAndCannibals/MissionariesAndCannibals.cfg",
            "N-Queens/Queens.toolbox/FourQueens/MC.cfg",
            "N-Queens/QueensPluscal.toolbox/FourQueens/MC.cfg",
            "SDP_Verification/SDP_Attack_Spec/MC.cfg",
            "SlidingPuzzles/SlidingPuzzles.cfg",
            "SlidingPuzzles/SlidingPuzzles_anim.cfg",
            "SpecifyingSystems/RealTime/MCRealTimeHourClock.cfg",
            "spanning/MC_spanning.cfg",
            "tower_of_hanoi/Hanoi.toolbox/Model_1/MC.cfg",
            "tower_of_hanoi/Hanoi_anim.cfg",
        ] {
            let spec = specs
                .iter()
                .find(|spec| spec.cfg_path == cfg_path)
                .unwrap_or_else(|| panic!("missing intentional-violation row {cfg_path}"));
            assert_eq!(
                spec.exclusion
                    .as_ref()
                    .map(|value| value.reason_code.as_str()),
                Some("expected_violation_first_found_noncomparable"),
                "{cfg_path} must remain correctness-only until a typed replay rule exists"
            );
        }
        for (cfg_path, reason_code) in [
            ("ewd687a/EWD687a_anim.cfg", "external_io_side_effect"),
            ("ewd840/EWD840_json.cfg", "external_io_side_effect"),
            ("ewd998/EWD998ChanTrace.cfg", "external_io_dependency"),
        ] {
            let external_io = specs
                .iter()
                .find(|spec| spec.cfg_path == cfg_path)
                .unwrap_or_else(|| panic!("missing external-IO row {cfg_path}"));
            assert_eq!(
                external_io
                    .exclusion
                    .as_ref()
                    .map(|value| value.reason_code.as_str()),
                Some(reason_code)
            );
        }
        for cfg_path in [
            "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_1/MC.cfg",
            "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_2/MC.cfg",
            "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_3/MC.cfg",
            "MisraReachability/MCReachabilityTestAllGraphs.cfg",
            "SpecifyingSystems/AsynchronousInterface/PrintValues.cfg",
            "SpecifyingSystems/SimpleMath/SimpleMath.cfg",
            "Stones/Stones.cfg",
            "TransitiveClosure/TransitiveClosure.cfg",
            "sums_even/MC_sums_even.cfg",
        ] {
            let spec = specs
                .iter()
                .find(|spec| spec.cfg_path == cfg_path)
                .unwrap_or_else(|| panic!("missing semantic-assertion row {cfg_path}"));
            assert_eq!(
                spec.exclusion
                    .as_ref()
                    .map(|value| value.reason_code.as_str()),
                Some("semantic_assertion_only"),
                "{cfg_path} must remain outside the reachable-state performance claim"
            );
        }
        assert!(specs.iter().any(|s| s.name == "MCBakery"));
        for name in [
            "SlidingPuzzles_anim",
            "BlockDagTest",
            "TLCSailfish1",
            "TLCSailfish2",
            "Hanoi_anim",
        ] {
            assert!(
                specs.iter().any(|spec| spec.name == name),
                "missing normalized row {name}"
            );
        }
    }

    #[test]
    fn manifest_rejects_free_form_or_unknown_work_equivalence_rules() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/tlc_comparison/strict_corpus_manifest.json");
        if !path.exists() {
            return;
        }
        let mut value: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        value["work_equivalence_policy"]["default_eligible_rule_id"] =
            json!("whatever the collector says");
        let error = parse_corpus_manifest(&serde_json::to_string(&value).unwrap()).unwrap_err();
        assert!(
            error.to_string().contains("work-equivalence default rule"),
            "{error:#}"
        );

        let mut value: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        value["work_equivalence_policy"]["rules"][EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID]
            ["free_form_exception"] = json!("close enough");
        let error = parse_corpus_manifest(&serde_json::to_string(&value).unwrap()).unwrap_err();
        let error_chain = format!("{error:#}");
        assert!(
            error_chain.contains("unknown field"),
            "unexpected parse error: {error_chain}"
        );

        let mut value: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        value["eligibility"]["exclusions"]["FiniteMonotonic/MCCRDT.cfg"]["reason_code"] =
            json!("ty_was_too_slow");
        let error = parse_corpus_manifest(&serde_json::to_string(&value).unwrap()).unwrap_err();
        assert!(
            error.to_string().contains("unsupported reason code"),
            "unexpected exclusion error: {error:#}"
        );
    }

    #[test]
    fn parse_tlc_output_states_generated_distinct_left() {
        let text = "Progress(7) at 2024-01-01 12:00:00\n\
                    Finished computing initial states: 1,001 distinct states generated at 2024-01-01 12:00:01.\n\
                    1,234,567 states generated, 1,234,000 distinct states found, 0 states left on queue.\n";
        let (counts, err) = parse_tlc_output(text);
        assert_eq!(counts.states, Some(1_234_000));
        assert_eq!(counts.raw_initial_states_generated, Some(1_001));
        assert_eq!(counts.raw_successors_generated, Some(1_233_566));
        assert_eq!(counts.states_generated, Some(1_234_567));
        assert_eq!(counts.states_left, Some(0));
        assert!(err.is_none());
    }

    #[test]
    fn parse_tlc_output_distinct_only() {
        let text = "Model checking completed.\n42 distinct states found.\n";
        let (counts, err) = parse_tlc_output(text);
        assert_eq!(counts.states, Some(42));
        assert!(err
            .as_deref()
            .is_some_and(|error| error.contains("raw_initial_states_generated")));
    }

    #[test]
    fn parse_tlc_output_non_distinct_initial_generation() {
        let text = "Finished computing initial states: 5 states generated, with 2 of them distinct at 2026-07-23 07:34:16.\n\
                    17 states generated, 9 distinct states found, 0 states left on queue.\n";
        let (counts, err) = parse_tlc_output(text);
        assert_eq!(
            counts,
            TlcParsedCounts {
                states: Some(9),
                raw_initial_states_generated: Some(5),
                raw_successors_generated: Some(12),
                states_generated: Some(17),
                states_left: Some(0),
            }
        );
        assert!(err.is_none());
    }

    #[test]
    fn parse_tlc_output_singular_generated_summary() {
        let text = "Finished computing initial states: 1 distinct state generated at 2026-07-23 07:34:16.\n\
                    1 state generated, 1 distinct state found, 0 states left on queue.\n";
        let (counts, err) = parse_tlc_output(text);
        assert_eq!(
            counts,
            TlcParsedCounts {
                states: Some(1),
                raw_initial_states_generated: Some(1),
                raw_successors_generated: Some(0),
                states_generated: Some(1),
                states_left: Some(0),
            }
        );
        assert!(err.is_none());
    }

    #[test]
    fn parse_tlc_output_rejects_generated_total_smaller_than_initial() {
        let text = "Finished computing initial states: 5 distinct states generated at 2026-07-23 07:34:16.\n\
                    4 states generated, 4 distinct states found, 0 states left on queue.\n";
        let (counts, err) = parse_tlc_output(text);
        assert_eq!(counts.raw_successors_generated, None);
        assert!(err
            .as_deref()
            .is_some_and(|error| error.contains("smaller than raw initial-state count")));
    }

    #[test]
    fn parse_tlc_output_rejects_nonempty_final_queue() {
        let text =
            "Finished computing initial states: 1 distinct state generated at fixture time.\n\
                    3 states generated, 3 distinct states found, 2 states left on queue.\n";
        let (counts, err) = parse_tlc_output(text);
        assert_eq!(counts.states_left, Some(2));
        assert!(!counts.complete());
        assert!(err
            .as_deref()
            .is_some_and(|error| error.contains("2 state(s) left")));
    }

    #[test]
    fn parse_tlc_output_missing_module() {
        let text = "Cannot find source file for module Naturals imported";
        let (counts, err) = parse_tlc_output(text);
        assert_eq!(counts, TlcParsedCounts::default());
        assert_eq!(err.as_deref(), Some("missing_module:Naturals"));
    }

    #[test]
    fn parse_tlc_output_no_state_count() {
        let (counts, err) = parse_tlc_output("nothing useful here\n");
        assert_eq!(counts, TlcParsedCounts::default());
        assert!(err
            .as_deref()
            .is_some_and(|error| error.contains("distinct_states")));
    }

    #[test]
    fn classify_invariant_error() {
        let text = "Error: Invariant TypeOK is violated.";
        assert_eq!(classify_error(text).as_deref(), Some("invariant"));
    }

    #[test]
    fn classify_deadlock_error() {
        let text = "Error: Deadlock reached.";
        assert_eq!(classify_error(text).as_deref(), Some("deadlock"));
    }

    #[test]
    fn classify_safety_property_error() {
        let text = "Error: Property MySafety is violated.";
        assert_eq!(classify_error(text).as_deref(), Some("safety"));
    }

    #[test]
    fn classify_no_error_returns_none() {
        assert_eq!(classify_error("Model checking completed.").as_deref(), None);
    }

    #[test]
    fn completion_requires_success_marker_zero_exit_empty_queue_and_counts() {
        let output = format!(
            "{TLC_SUCCESS_MARKER}\n3 states generated, 3 distinct states found, 0 states left on queue."
        );
        assert_eq!(
            validate_tlc_completion(0, &output, complete_tlc_counts(), None),
            Ok(())
        );

        for (error_line, expected_type) in [
            ("Error: Invariant TypeOK is violated.", "invariant"),
            ("Error: Deadlock reached.", "deadlock"),
            ("Error: Temporal properties were violated.", "liveness"),
            ("Error: Action property StepOK is violated.", "action"),
            ("Error: evaluator failed.", "unknown"),
        ] {
            let output = format!("{error_line}\n{TLC_SUCCESS_MARKER}");
            let error =
                validate_tlc_completion(0, &output, complete_tlc_counts(), None).unwrap_err();
            assert_eq!(error.0, expected_type);
        }

        let error = validate_tlc_completion(2, TLC_SUCCESS_MARKER, complete_tlc_counts(), None)
            .unwrap_err();
        assert_eq!(error.0, "process_exit");

        let error =
            validate_tlc_completion(0, "no error text", complete_tlc_counts(), None).unwrap_err();
        assert_eq!(error.0, "incomplete_completion");

        let mut nonempty_queue = complete_tlc_counts();
        nonempty_queue.states_left = Some(1);
        let error = validate_tlc_completion(
            0,
            TLC_SUCCESS_MARKER,
            nonempty_queue,
            Some("TLC completion summary reports 1 state(s) left on the queue"),
        )
        .unwrap_err();
        assert_eq!(error.0, "count_parse");
    }

    #[test]
    fn raw_generated_count_completeness_requires_all_three_fields() {
        let complete = json!({
            "raw_initial_states_generated": 2,
            "raw_successors_generated": 5,
            "states_generated": 7,
        });
        assert!(complete_raw_generated_counts(complete.as_object().unwrap()));

        for missing in [
            json!({
                "raw_successors_generated": 5,
                "states_generated": 7,
            }),
            json!({
                "raw_initial_states_generated": 2,
                "states_generated": 7,
            }),
            json!({
                "raw_initial_states_generated": 2,
                "raw_successors_generated": 5,
            }),
            json!({
                "raw_initial_states_generated": 2,
                "raw_successors_generated": 5,
                "states_generated": 8,
            }),
        ] {
            assert!(!complete_raw_generated_counts(missing.as_object().unwrap()));
        }
    }

    #[test]
    fn resume_requires_v4_exact_provenance_and_intact_specs_digest() {
        let provenance = json!({
            "schema_version": SCHEMA_VERSION,
            "collector": {"script_sha256": "script"},
            "tlc": {"jar_sha256": "jar"},
            "inputs": {"strict_corpus": {"commit": "pin"}},
            "seed": {"enabled": false},
            "tlc_timeout_seconds": 60,
        })
        .as_object()
        .unwrap()
        .clone();
        let specs = json!({"A": {"tlc": {"status": "pass"}}});
        let digest = sha256_jcs(&specs).unwrap();
        let mut existing = json!({
            "schema_version": SCHEMA_VERSION,
            "collector": {"script_sha256": "script"},
            "tlc": {"jar_sha256": "jar"},
            "inputs": {"strict_corpus": {"commit": "pin"}},
            "seed": {"enabled": false},
            "tlc_timeout_seconds": 60,
            "specs_jcs_sha256": digest,
            "specs": specs,
        });
        assert!(resume_output_matches_provenance(&existing, &provenance));

        existing["schema_version"] = json!(3);
        assert!(!resume_output_matches_provenance(&existing, &provenance));
        existing["schema_version"] = json!(SCHEMA_VERSION);

        existing["specs"]["A"]["tlc"]["states"] = json!(99);
        assert!(!resume_output_matches_provenance(&existing, &provenance));
        existing["specs"]["A"]["tlc"]["states"] = Value::Null;
        existing["specs_jcs_sha256"] = json!(sha256_jcs(&existing["specs"]).unwrap());
        existing["collector"]["script_sha256"] = json!("other-script");
        assert!(!resume_output_matches_provenance(&existing, &provenance));
    }

    #[cfg(unix)]
    #[test]
    fn timeout_runner_drains_output_larger_than_pipe_capacity() {
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("yes x | head -c 262144")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let (code, timed_out, stdout, stderr) =
            run_with_timeout(command, Duration::from_secs(5)).unwrap();
        assert_eq!(code, 0);
        assert!(!timed_out);
        assert_eq!(stdout.len(), 262_144);
        assert!(stderr.is_empty());
    }

    #[test]
    fn categorize_runtime_buckets() {
        assert_eq!(categorize_runtime(None), "unknown");
        assert_eq!(categorize_runtime(Some(0.5)), "small");
        assert_eq!(categorize_runtime(Some(15.0)), "medium");
        assert_eq!(categorize_runtime(Some(100.0)), "large");
        assert_eq!(categorize_runtime(Some(500.0)), "xlarge");
    }

    #[test]
    fn sha256_jcs_is_order_independent() {
        let left = json!({"b": 1, "a": 2, "c": [3, 4]});
        let right = json!({"a": 2, "c": [3, 4], "b": 1});
        assert_eq!(sha256_jcs(&left).unwrap(), sha256_jcs(&right).unwrap());
    }

    #[test]
    fn sha256_jcs_matches_python_reference_for_specs_shape() {
        // The Python `json_jcs.sha256_jcs` returns the same digest for
        // a small fixture; encode that contract here as a regression.
        let value = json!({
            "Foo": {
                "ty": {"status": "untested"},
                "tlc": {"states": 42, "status": "pass"},
            },
        });
        let digest = sha256_jcs(&value).unwrap();
        // Recomputed via Python: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        // Both impls canonicalize identically for integer/string/null types.
        // Keys are emitted in lexicographic order (RFC 8785 / sort_keys=True):
        // "tlc" < "ty" at the Foo level, "states" < "status" within tlc.
        let expected_canonical =
            "{\"Foo\":{\"tlc\":{\"states\":42,\"status\":\"pass\"},\"ty\":{\"status\":\"untested\"}}}";
        let mut hasher = Sha256::new();
        hasher.update(expected_canonical.as_bytes());
        let expected = format!("{:x}", hasher.finalize());
        assert_eq!(digest, expected);
    }

    #[test]
    fn order_spec_entry_migrates_v2_flat_keys() {
        let legacy = json!({
            "status": "pass",
            "expected_states": 17,
            "tlc_runtime_seconds": 0.42,
            "category": "small",
            "issue": null,
        });
        let ordered = order_spec_entry(legacy);
        let tlc = ordered.get("tlc").unwrap().as_object().unwrap();
        assert_eq!(tlc.get("status").unwrap(), &json!("pass"));
        assert_eq!(tlc.get("states").unwrap(), &json!(17));
        assert_eq!(tlc.get("runtime_seconds").unwrap(), &json!(0.42));
        assert_eq!(ordered.get("category").unwrap(), &json!("small"));
        assert!(ordered.get("ty").unwrap().is_object());
    }

    #[test]
    fn order_spec_entry_preserves_unknown_keys() {
        // serde_json::Map is a BTreeMap (no preserve_order feature), so the
        // serialized form is alphabetical regardless of insertion order.
        // The contract we care about is: (1) all known keys retained,
        // (2) unknown keys retained (no data loss), (3) deterministic order.
        let entry = json!({
            "tlc": {"status": "pass", "states": 1},
            "ty": {"status": "untested"},
            "verified_match": false,
            "category": "small",
            "source": {"tla_path": "x.tla", "cfg_path": "x.cfg"},
            "expected_mismatch": true,
            "issue": "#42",
        });
        let ordered = order_spec_entry(entry).as_object().unwrap().clone();
        for key in [
            "tlc",
            "ty",
            "verified_match",
            "category",
            "source",
            "expected_mismatch",
            "issue",
        ] {
            assert!(ordered.contains_key(key), "missing key {key}");
        }
        // Determinism: keys are sorted (serde_json::Map = BTreeMap).
        let keys: Vec<&String> = ordered.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn compute_stats_counts_tlc_and_ty_buckets() {
        let mut specs: Map<String, Value> = Map::new();
        specs.insert(
            "A".into(),
            json!({
                "tlc": {"status": "pass", "states": 1},
                "ty": {"status": "pass", "states": 1},
                "verified_match": true,
            }),
        );
        specs.insert(
            "B".into(),
            json!({
                "tlc": {"status": "pass", "states": 1},
                "ty": {"status": "mismatch"},
                "verified_match": false,
            }),
        );
        specs.insert(
            "C".into(),
            json!({
                "tlc": {"status": "timeout"},
                "ty": {"status": "untested"},
            }),
        );
        specs.insert(
            "D".into(),
            json!({
                "tlc": {"status": "error"},
                "ty": {"status": "fail"},
            }),
        );
        let stats = compute_stats(&specs);
        assert_eq!(stats.get("tlc_pass").unwrap(), &json!(2));
        assert_eq!(stats.get("tlc_timeout").unwrap(), &json!(1));
        assert_eq!(stats.get("tlc_error").unwrap(), &json!(1));
        assert_eq!(stats.get("tlc_unsupported").unwrap(), &json!(0));
        assert_eq!(stats.get("tlc_uncollected").unwrap(), &json!(0));
        assert_eq!(stats.get("ty_match").unwrap(), &json!(1));
        assert_eq!(stats.get("ty_mismatch").unwrap(), &json!(1));
        assert_eq!(stats.get("ty_fail").unwrap(), &json!(1));
        assert_eq!(stats.get("ty_untested").unwrap(), &json!(1));
    }

    #[test]
    fn compute_categories_counts_all_buckets() {
        // serde_json::Map = BTreeMap; key order is alphabetical when
        // serialized. Contract: every observed category is present with
        // the right count, plus the known buckets are always populated
        // even when empty.
        let mut specs: Map<String, Value> = Map::new();
        specs.insert("A".into(), json!({"category": "small"}));
        specs.insert("B".into(), json!({"category": "medium"}));
        specs.insert("C".into(), json!({"category": "apalache"}));
        let cats = compute_categories(&specs);
        assert_eq!(cats.get("small").unwrap(), &json!(1));
        assert_eq!(cats.get("medium").unwrap(), &json!(1));
        assert_eq!(cats.get("apalache").unwrap(), &json!(1));
        for key in CATEGORIES_KEY_ORDER {
            assert!(cats.contains_key(*key), "missing canonical key {key}");
        }
    }

    #[test]
    fn extract_tlc_version_finds_dotted_triple() {
        let text = "TLC Version 2.18.0 of Day Month Year\n";
        assert_eq!(extract_tlc_version(text), Some("2.18.0".into()));
    }

    #[test]
    fn first_dotted_triple_returns_none_for_two_dots_only() {
        assert_eq!(first_dotted_triple("just 1.2"), None);
    }

    #[test]
    fn build_ordered_specs_discards_rows_outside_manifest_catalog() {
        // The strict manifest is the complete source of truth. Legacy
        // repo-local and deleted-fixture baseline rows must not leak into a
        // normalized refresh.
        let catalog = vec![
            SpecInfo {
                name: "Beta".into(),
                tla_path: "B.tla".into(),
                cfg_path: "B.cfg".into(),
                exclusion: None,
            },
            SpecInfo {
                name: "Alpha".into(),
                tla_path: "A.tla".into(),
                cfg_path: "A.cfg".into(),
                exclusion: None,
            },
        ];
        let mut baselines: BTreeMap<String, Value> = BTreeMap::new();
        baselines.insert(
            "Beta".into(),
            json!({"tlc": {"status": "pass"}, "ty": {"status": "untested"}, "category": "small"}),
        );
        baselines.insert(
            "Alpha".into(),
            json!({"tlc": {"status": "pass"}, "ty": {"status": "untested"}, "category": "small"}),
        );
        baselines.insert(
            "Zeta".into(),
            json!({"tlc": {"status": "pass"}, "ty": {"status": "untested"}, "category": "small"}),
        );
        let ordered = build_ordered_specs(baselines, &catalog);
        for name in ["Alpha", "Beta"] {
            assert!(ordered.contains_key(name), "missing {name}");
        }
        assert!(!ordered.contains_key("Zeta"));
        assert_eq!(ordered.len(), 2);
    }

    #[test]
    fn skeleton_is_complete_and_marks_exclusions_without_measurements() {
        let catalog = vec![
            SpecInfo {
                name: "Eligible".into(),
                tla_path: "A/A.tla".into(),
                cfg_path: "A/A.cfg".into(),
                exclusion: None,
            },
            SpecInfo {
                name: "Excluded".into(),
                tla_path: "B/B.tla".into(),
                cfg_path: "B/B.cfg".into(),
                exclusion: Some(ManifestExclusion {
                    reason_code: "simulation_only".into(),
                    detail: "simulation".into(),
                }),
            },
        ];
        let skeleton = initialize_baselines(&catalog, BTreeMap::new());
        assert_eq!(skeleton.len(), 2);
        assert_eq!(
            skeleton["Eligible"]["tlc"]["status"],
            Value::String("uncollected".into())
        );
        assert_eq!(
            skeleton["Eligible"]["eligibility"],
            Value::String("eligible".into())
        );
        assert_eq!(
            skeleton["Eligible"]["work_equivalence"]["rule_id"],
            Value::String(EXHAUSTIVE_WORK_EQUIVALENCE_RULE_ID.into())
        );
        assert_eq!(
            skeleton["Excluded"]["tlc"]["status"],
            Value::String("unsupported".into())
        );
        assert!(skeleton["Excluded"].get("work_equivalence").is_none());
        assert_eq!(
            skeleton["Excluded"]["exclusion"]["reason_code"],
            Value::String("simulation_only".into())
        );
        let map: Map<String, Value> = skeleton.into_iter().collect();
        let stats = compute_stats(&map);
        assert_eq!(stats["tlc_uncollected"], json!(1));
        assert_eq!(stats["tlc_unsupported"], json!(1));
    }

    #[test]
    fn missing_entry_marks_status_error_and_keeps_existing_ty() {
        let prev = json!({"ty": {"status": "pass", "states": 7}});
        let spec = SpecInfo {
            name: "X".into(),
            tla_path: "x.tla".into(),
            cfg_path: "x.cfg".into(),
            exclusion: None,
        };
        let entry = missing_entry(&Some(prev), &spec, "missing_file", "File not found: x.tla");
        let obj = entry.as_object().unwrap();
        assert_eq!(
            obj.get("tlc").unwrap().get("error_type").unwrap(),
            &json!("missing_file")
        );
        assert_eq!(obj.get("verified_match").unwrap(), &Value::Bool(false));
        assert_eq!(
            obj.get("ty").unwrap().get("status").unwrap(),
            &json!("pass")
        );
    }

    #[test]
    fn write_output_round_trip_matches_jcs_digest() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("baseline.json");
        let catalog = vec![SpecInfo {
            name: "Alpha".into(),
            tla_path: "A.tla".into(),
            cfg_path: "A.cfg".into(),
            exclusion: None,
        }];
        let mut baselines: BTreeMap<String, Value> = BTreeMap::new();
        baselines.insert(
            "Alpha".into(),
            json!({
                "tlc": {"status": "pass", "states": 5, "runtime_seconds": 0.1},
                "ty": {"status": "untested"},
                "verified_match": false,
                "category": "small",
                "source": {"tla_path": "A.tla", "cfg_path": "A.cfg"},
            }),
        );
        let provenance = {
            let mut p = Map::new();
            p.insert("schema_version".into(), json!(SCHEMA_VERSION));
            p.insert("collector".into(), json!({"ty_git_commit": "deadbeef"}));
            p.insert("tlc".into(), json!({"tlc_version": "X"}));
            p.insert("inputs".into(), json!({}));
            p.insert("seed".into(), json!({}));
            p.insert("tlc_timeout_seconds".into(), json!(60));
            p
        };
        write_output(&path, baselines, &provenance, &catalog).unwrap();
        let body = fs::read_to_string(&path).unwrap();
        let value: Value = serde_json::from_str(&body).unwrap();
        let specs = value.get("specs").cloned().unwrap();
        let expected = sha256_jcs(&specs).unwrap();
        assert_eq!(
            value
                .get("specs_jcs_sha256")
                .and_then(Value::as_str)
                .unwrap(),
            expected
        );
    }
}
