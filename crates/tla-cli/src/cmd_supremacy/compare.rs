// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLC-vs-TY comparison gate for `ty supremacy compare`.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{de::IgnoredAny, Deserialize, Deserializer, Serialize};

use super::parse;
use super::policy;
use super::runner::{
    run_command_with_envelope, CommandResult, CommandSpec, CpuConfinementMethod,
    DiskHighWaterEvidence, DiskHighWaterMethod, DiskHighWaterScope, DiskSamplingExecution,
    ExecutionEnvelope, PeakMemoryMethod, PeakMemoryMetric, PeakMemoryScope, ResourceEvidence,
    COMMAND_SCOPED_ENV_KEYS, COMMAND_SCRATCH_DIR_NAME, DISK_SCOPE_CONTRACT_SCHEMA,
    DISK_USAGE_SAMPLE_INTERVAL, DISK_USAGE_SCAN_BUDGET, DISK_USAGE_SCAN_ENTRY_LIMIT,
};
#[cfg(test)]
use super::tlc_java_single_thread_args;
use super::tlc_java_single_thread_base_argv;
use super::work_equivalence::{WorkEquivalenceEvidence, WorkEquivalenceVerdict};
use crate::cli_schema::{
    SupremacyCompareArgs, SupremacyCompareBackend, SupremacyComparePolicy,
    SupremacyCompareSpecSource, SupremacyMode, SupremacyOutputFormat,
};

const COMPARE_REPORT_SCHEMA: &str = "ty.supremacy.compare.v4";
const DEFAULT_TLC_JAR: &str = "tlaplus/tytools.jar";
const DEFAULT_COMMUNITY_MODULES_JAR: &str = "tlaplus/CommunityModules.jar";
const DEFAULT_TLA_LIBRARY: &str = "test_specs/tla_library";
const STRICT_MIN_SPEEDUP: f64 = 1.05;
const STRICT_MAX_MEMORY_RATIO: f64 = 0.95;
const STRICT_MIN_BALANCED_PAIRED_RUNS: usize = 6;
const PAIRED_STATISTIC: &str = "median_within_pair_ratio.v2";
const ENV_TLC_BIN: &str = "TLC_BIN";
const ENV_TYTOOLS_JAR: &str = "TYTOOLS_JAR";
const ENV_TLC_JAR: &str = "TLC_JAR";
const ENV_COMMUNITY_MODULES: &str = "COMMUNITY_MODULES";
const ENV_TLA_LIBRARY: &str = "TLA_LIBRARY";
const ENV_TLA_PLUS_LIBRARY: &str = "TLA_PLUS_LIBRARY";
const DEFAULT_CASE: &str = "default";
const ALLOWED_COMPARE_CASE_ENV_KEYS: &[&str] = &["TY_PARALLEL_READONLY_VALUE_CACHES"];
pub(super) fn run(args: SupremacyCompareArgs) -> Result<()> {
    validate_args(&args)?;
    let cases = resolve_cases(&args)?;
    let repo_root = env::current_dir().context("resolve current working directory")?;
    let requested_output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir("compare"));
    // TLC resolves `-metadir` relative to the spec-directory cwd. Keep the
    // entire evidence tree absolute so TLC and the parent runner address the
    // same per-run directory even when the CLI receives a relative path.
    let output_dir = absolutize(&repo_root, &requested_output_dir);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("create output dir {}", output_dir.display()))?;

    let specs = resolve_specs(&args, &repo_root)?;
    let tlc_runner = resolve_tlc_runner(&args, &repo_root)?;
    let ty_bin = args
        .ty_bin
        .clone()
        .map(Ok)
        .unwrap_or_else(env::current_exe)
        .context("resolve ty binary")?;
    validate_file(&ty_bin).with_context(|| format!("validate --ty-bin {}", ty_bin.display()))?;

    let mut rows = Vec::new();
    for spec in &specs {
        for workers in &args.workers {
            for case in &cases {
                if matches!(args.format, SupremacyOutputFormat::Human) {
                    eprintln!(
                        "[supremacy] compare {} case={} backend={} workers={}",
                        spec.name,
                        case.name,
                        backend_cli_name(args.backend),
                        workers
                    );
                }
                let row = run_compare_row(
                    spec,
                    *workers,
                    case,
                    &args,
                    &repo_root,
                    &output_dir,
                    &tlc_runner,
                    &ty_bin,
                )?;
                rows.push(row);
            }
        }
    }

    let report = CompareReport::new(&args, output_dir.clone(), cases, rows);
    let report_json =
        serde_json::to_string_pretty(&report).context("serialize supremacy compare report")?;
    fs::write(output_dir.join("compare.json"), report_json + "\n")
        .with_context(|| format!("write {}", output_dir.join("compare.json").display()))?;
    fs::write(output_dir.join("compare.md"), report.to_markdown())
        .with_context(|| format!("write {}", output_dir.join("compare.md").display()))?;

    print_report(&report, args.format)?;

    if !report.passed && args.mode == SupremacyMode::Enforce {
        bail!(
            "ty supremacy compare failed {} row(s); see {}",
            report.failed_rows,
            output_dir.join("compare.json").display()
        );
    }
    Ok(())
}

fn validate_args(args: &SupremacyCompareArgs) -> Result<()> {
    if args.timeout == 0 {
        bail!("--timeout must be >= 1");
    }
    if args.runs == 0 {
        bail!("--runs must be >= 1");
    }
    if args.workers.is_empty() {
        bail!("--workers must list at least one worker count");
    }
    if args.workers.contains(&0) {
        bail!("--workers values must be >= 1");
    }
    if !args.min_speedup.is_finite() || args.min_speedup <= 0.0 {
        bail!("--min-speedup must be finite and > 0");
    }
    if !args.max_memory_ratio.is_finite() || args.max_memory_ratio <= 0.0 {
        bail!("--max-memory-ratio must be finite and > 0");
    }
    if args.mode == SupremacyMode::Enforce
        && policy_checks_speed(args.policy)
        && (args.tlc_bin.is_some() || non_empty_env_path(ENV_TLC_BIN).is_some())
    {
        bail!(
            "enforced performance compare requires the auditable Java TLC runner; unset {ENV_TLC_BIN} and omit --tlc-bin so single-thread JVM controls are recorded in command artifacts"
        );
    }
    if args.mode == SupremacyMode::Enforce && policy_checks_speed(args.policy) {
        if args.min_speedup < STRICT_MIN_SPEEDUP {
            bail!("enforced performance compare requires --min-speedup >= {STRICT_MIN_SPEEDUP}");
        }
        if policy_checks_memory(args.policy) && args.max_memory_ratio > STRICT_MAX_MEMORY_RATIO {
            bail!(
                "enforced both-axis compare requires --max-memory-ratio <= {STRICT_MAX_MEMORY_RATIO}"
            );
        }
        if args.runs < STRICT_MIN_BALANCED_PAIRED_RUNS || args.runs % 2 != 0 {
            bail!(
                "enforced performance compare requires an even --runs >= {STRICT_MIN_BALANCED_PAIRED_RUNS} so both within-pair launch orders and both AUTO pair-block orders have equal representation"
            );
        }
        if args.workers.iter().any(|workers| *workers != 1) {
            bail!(
                "enforced single-thread performance compare requires --workers 1; use --mode warn for diagnostic multi-worker comparisons"
            );
        }
        if !args.ty_flag.is_empty() {
            bail!(
                "enforced performance compare does not allow TY-only --ty-flag values; use shared TLA+/cfg settings or --mode warn diagnostics"
            );
        }
    }
    match args.spec_source {
        SupremacyCompareSpecSource::Baseline => {
            if args.tla.is_some() || args.config.is_some() {
                bail!("--tla/--config require --spec-source explicit");
            }
        }
        SupremacyCompareSpecSource::Explicit => {
            if args.tla.is_none() || args.config.is_none() {
                bail!("--spec-source explicit requires --tla and --config");
            }
        }
    }
    Ok(())
}

fn policy_checks_speed(policy: SupremacyComparePolicy) -> bool {
    matches!(
        policy,
        SupremacyComparePolicy::ParityAndSpeed | SupremacyComparePolicy::ParityAndSpeedAndMemory
    )
}

fn policy_checks_memory(policy: SupremacyComparePolicy) -> bool {
    policy == SupremacyComparePolicy::ParityAndSpeedAndMemory
}

#[derive(Clone, Debug)]
struct CompareSpec {
    name: String,
    tla_path: PathBuf,
    cfg_path: PathBuf,
    expected_tlc_states: Option<u64>,
    expected_backend_states: Option<u64>,
    expected_tlc_error: Option<String>,
    expected_backend_error: Option<String>,
    work_equivalence: Option<WorkEquivalenceEvidence>,
}

#[derive(Clone, Debug, Serialize)]
struct EnvCase {
    name: String,
    env_overrides: BTreeMap<String, String>,
}

fn resolve_cases(args: &SupremacyCompareArgs) -> Result<Vec<EnvCase>> {
    let case_names = if args.cases.is_empty() {
        vec![DEFAULT_CASE.to_string()]
    } else {
        args.cases.clone()
    };

    let mut seen = BTreeSet::new();
    for name in &case_names {
        validate_case_name(name)?;
        if !seen.insert(name.clone()) {
            bail!("duplicate --case {name:?}");
        }
    }

    let protected_keys = protected_ty_env_keys(args.backend);
    let global_env = parse_env_assignments(&args.ty_env, "--ty-env", &protected_keys)?;
    let mut case_env = BTreeMap::<String, BTreeMap<String, String>>::new();
    for value in &args.case_env {
        let (case_name, key, env_value) =
            parse_case_env_assignment(value, "--case-env", &protected_keys)?;
        if !seen.contains(&case_name) {
            bail!("--case-env references unknown case {case_name:?}");
        }
        case_env
            .entry(case_name)
            .or_default()
            .insert(key, env_value);
    }

    Ok(case_names
        .into_iter()
        .map(|name| {
            let mut env_overrides = global_env.clone();
            if let Some(overrides) = case_env.remove(&name) {
                env_overrides.extend(overrides);
            }
            EnvCase {
                name,
                env_overrides,
            }
        })
        .collect())
}

fn validate_case_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("--case names must not be empty");
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.')
    {
        bail!(
            "--case {name:?} contains unsupported characters; use ASCII letters, digits, '.', '-', or '_'"
        );
    }
    Ok(())
}

fn parse_env_assignments(
    values: &[String],
    flag: &str,
    protected_keys: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for value in values {
        let (key, env_value) = parse_env_assignment(value, flag, protected_keys)?;
        result.insert(key, env_value);
    }
    Ok(result)
}

fn parse_case_env_assignment(
    value: &str,
    flag: &str,
    protected_keys: &BTreeSet<String>,
) -> Result<(String, String, String)> {
    let Some((case_name, assignment)) = value.split_once(':') else {
        bail!("{flag} must be NAME:KEY=VALUE");
    };
    validate_case_name(case_name)?;
    let (key, env_value) = parse_env_assignment(assignment, flag, protected_keys)?;
    Ok((case_name.to_string(), key, env_value))
}

fn parse_env_assignment(
    value: &str,
    flag: &str,
    protected_keys: &BTreeSet<String>,
) -> Result<(String, String)> {
    let Some((key, env_value)) = value.split_once('=') else {
        bail!("{flag} must be KEY=VALUE");
    };
    validate_user_ty_env_key(key, flag, protected_keys)?;
    validate_user_ty_env_value(key, env_value, flag)?;
    Ok((key.to_string(), env_value.to_string()))
}

fn validate_user_ty_env_key(
    key: &str,
    flag: &str,
    protected_keys: &BTreeSet<String>,
) -> Result<()> {
    if key.is_empty() {
        bail!("{flag} env key must not be empty");
    }
    if protected_keys.contains(key) {
        bail!("{flag} cannot override protected backend env key {key}");
    }
    if !key.starts_with("TY_") {
        bail!("{flag} env key {key} is not allowed; only TY_* keys may be varied");
    }
    if !key
        .chars()
        .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
    {
        bail!("{flag} env key {key} must use ASCII uppercase letters, digits, and '_'");
    }
    if !ALLOWED_COMPARE_CASE_ENV_KEYS.contains(&key) {
        bail!(
            "{flag} env key {key} is not allowed for compare env cases; allowed keys: {}",
            ALLOWED_COMPARE_CASE_ENV_KEYS.join(", ")
        );
    }
    Ok(())
}

fn validate_user_ty_env_value(key: &str, value: &str, flag: &str) -> Result<()> {
    match key {
        "TY_PARALLEL_READONLY_VALUE_CACHES" => {
            if matches!(value, "" | "0" | "1") {
                Ok(())
            } else {
                bail!(
                    "{flag} env key {key} accepts only \"\", \"0\", or \"1\" for compare env cases"
                );
            }
        }
        _ => Ok(()),
    }
}

fn protected_ty_env_keys(backend: SupremacyCompareBackend) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    keys.extend(backend_env(backend).into_keys());
    keys.extend(
        policy::full_native_fused_protected_env()
            .into_keys()
            .collect::<Vec<_>>(),
    );
    keys.extend([
        "TY_CACHE_DIR".to_string(),
        "TLA_LIBRARY".to_string(),
        "TLA_PLUS_LIBRARY".to_string(),
    ]);
    keys
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SpecBaseline {
    #[serde(default)]
    inputs: BaselineInputs,
    specs: BTreeMap<String, SpecBaselineEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct BaselineInputs {
    #[serde(default)]
    examples_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
struct SpecBaselineEntry {
    source: Option<SpecBaselineSource>,
    #[serde(default)]
    tlc: SpecBaselineMode,
    #[serde(default)]
    ty: SpecBaselineMode,
    #[serde(default)]
    ty_expected_states: Option<u64>,
    #[serde(default)]
    verified_match: Option<bool>,
    #[serde(default)]
    work_equivalence: Option<WorkEquivalenceEvidence>,
    #[serde(default, deserialize_with = "deserialize_field_presence")]
    work_equivalence_rule: FieldPresence,
    #[serde(default, deserialize_with = "deserialize_field_presence")]
    equivalent_work_rule: FieldPresence,
    #[serde(default, deserialize_with = "deserialize_field_presence")]
    performance_work_equivalence_rule: FieldPresence,
}

#[derive(Clone, Copy, Debug, Default)]
struct FieldPresence(bool);

fn deserialize_field_presence<'de, D>(
    deserializer: D,
) -> std::result::Result<FieldPresence, D::Error>
where
    D: Deserializer<'de>,
{
    let _ = IgnoredAny::deserialize(deserializer)?;
    Ok(FieldPresence(true))
}

#[derive(Clone, Debug, Deserialize)]
struct SpecBaselineSource {
    tla_path: PathBuf,
    cfg_path: PathBuf,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SpecBaselineMode {
    #[serde(default)]
    states: Option<u64>,
    #[serde(default)]
    error_type: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

fn validated_baseline_work_equivalence(
    spec_name: &str,
    entry: &SpecBaselineEntry,
) -> Result<Option<WorkEquivalenceEvidence>> {
    let legacy_aliases = [
        ("work_equivalence_rule", entry.work_equivalence_rule.0),
        ("equivalent_work_rule", entry.equivalent_work_rule.0),
        (
            "performance_work_equivalence_rule",
            entry.performance_work_equivalence_rule.0,
        ),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect::<Vec<_>>();
    if !legacy_aliases.is_empty() {
        bail!(
            "baseline spec {spec_name:?} uses rejected legacy work-equivalence alias(es): {}; use the exact typed work_equivalence field",
            legacy_aliases.join(", ")
        );
    }
    if entry
        .work_equivalence
        .as_ref()
        .is_some_and(|evidence| !evidence.is_exact_exhaustive_holds_rule())
    {
        bail!(
            "baseline spec {spec_name:?} work_equivalence must be exactly schema_version={} rule_id={:?}",
            super::work_equivalence::WORK_EQUIVALENCE_SCHEMA_VERSION,
            super::work_equivalence::EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID
        );
    }
    Ok(entry.work_equivalence.clone())
}

fn resolve_specs(args: &SupremacyCompareArgs, repo_root: &Path) -> Result<Vec<CompareSpec>> {
    match args.spec_source {
        SupremacyCompareSpecSource::Baseline => resolve_baseline_specs(args, repo_root),
        SupremacyCompareSpecSource::Explicit => resolve_explicit_spec(args, repo_root),
    }
}

fn resolve_baseline_specs(
    args: &SupremacyCompareArgs,
    repo_root: &Path,
) -> Result<Vec<CompareSpec>> {
    let text = fs::read_to_string(&args.baseline)
        .with_context(|| format!("read baseline {}", args.baseline.display()))?;
    let baseline: SpecBaseline = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline {}", args.baseline.display()))?;
    let examples_dir = super::resolve_examples_dir(baseline.inputs.examples_dir.as_deref());
    let explicit_names = !args.specs.is_empty();
    let names = if explicit_names {
        args.specs.clone()
    } else {
        baseline.specs.keys().cloned().collect::<Vec<_>>()
    };

    let mut specs = Vec::new();
    for name in names {
        let entry = baseline.specs.get(&name).with_context(|| {
            format!(
                "baseline spec {name:?} not found in {}",
                args.baseline.display()
            )
        })?;
        let work_equivalence = validated_baseline_work_equivalence(&name, entry)?;
        if !explicit_names && !entry.verified_match.unwrap_or(false) {
            continue;
        }
        let Some(source) = entry.source.as_ref() else {
            if explicit_names {
                bail!("baseline spec {name:?} has no source paths");
            }
            continue;
        };
        let mode = source.mode.as_deref().unwrap_or("check");
        if mode != "check" {
            if explicit_names {
                bail!(
                    "baseline spec {name:?} source mode is {mode:?}; supremacy compare supports only check-mode specs"
                );
            }
            continue;
        }
        let tla_path = resolve_source_path(repo_root, &examples_dir, &source.tla_path);
        let cfg_path = resolve_source_path(repo_root, &examples_dir, &source.cfg_path);
        if let Err(error) = validate_spec_files(&tla_path, &cfg_path) {
            if explicit_names {
                return Err(error);
            }
            // All-rows mode: a baseline row whose pinned source no longer
            // resolves (moved repo-test specs, stale paths) is SKIPPED WITH A
            // LOG, not a fatal error — one rotten row must not abort a corpus
            // sweep. Explicitly named specs still fail loudly. The skip list
            // is itself burndown evidence (P0.7 baseline refresh).
            eprintln!("[compare] skipping baseline row {name:?}: {error:#}");
            continue;
        }
        specs.push(CompareSpec {
            name,
            tla_path,
            cfg_path,
            expected_tlc_states: entry.tlc.states,
            expected_backend_states: entry
                .ty_expected_states
                .or(entry.ty.states)
                .or(entry.tlc.states),
            expected_tlc_error: expected_error_type(&entry.tlc),
            expected_backend_error: expected_error_type(&entry.ty),
            work_equivalence,
        });
    }
    if specs.is_empty() {
        bail!(
            "no check-mode specs selected from {}",
            args.baseline.display()
        );
    }
    Ok(specs)
}

fn resolve_explicit_spec(
    args: &SupremacyCompareArgs,
    repo_root: &Path,
) -> Result<Vec<CompareSpec>> {
    let tla_path = absolutize(repo_root, args.tla.as_ref().expect("validated --tla"));
    let cfg_path = absolutize(repo_root, args.config.as_ref().expect("validated --config"));
    validate_spec_files(&tla_path, &cfg_path)?;
    let name = args
        .specs
        .first()
        .cloned()
        .or_else(|| {
            tla_path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "explicit".to_string());
    Ok(vec![CompareSpec {
        name,
        tla_path,
        cfg_path,
        expected_tlc_states: None,
        expected_backend_states: None,
        expected_tlc_error: None,
        expected_backend_error: None,
        work_equivalence: None,
    }])
}

fn run_compare_row(
    spec: &CompareSpec,
    workers: usize,
    case: &EnvCase,
    args: &SupremacyCompareArgs,
    repo_root: &Path,
    output_dir: &Path,
    tlc_runner: &TlcRunner,
    ty_bin: &Path,
) -> Result<CompareRow> {
    let spec_dir = output_dir
        .join(safe_name(&spec.name))
        .join(format!("workers-{workers}"))
        .join(safe_name(&case.name));
    let execution_envelope =
        if args.mode == SupremacyMode::Enforce && policy_checks_speed(args.policy) {
            ExecutionEnvelope::strict_single_core_process_tree()
        } else {
            ExecutionEnvelope::diagnostic()
        };
    let mut paired_runs = Vec::with_capacity(args.runs);
    for run_index in 0..args.runs {
        let repetition = run_index + 1;
        let run_dir = spec_dir.join(format!("run-{repetition:03}"));
        if is_production_auto(args.backend) && policy_checks_speed(args.policy) {
            let schedule = auto_performance_schedule(repetition);
            let run_production_pair = || -> Result<(RunObservation, RunObservation)> {
                match schedule.production_pair_order {
                    CompareRunOrder::TlcThenTy => {
                        let tlc = run_tlc(
                            spec,
                            workers,
                            args.timeout,
                            repo_root,
                            &run_dir,
                            "production-tlc",
                            tlc_runner,
                            execution_envelope,
                        )?;
                        let backend = run_ty_backend(
                            spec,
                            workers,
                            case,
                            args,
                            repo_root,
                            &run_dir,
                            ty_bin,
                            args.backend,
                            backend_cli_name(args.backend),
                            &[],
                            execution_envelope,
                        )?;
                        Ok((tlc, backend))
                    }
                    CompareRunOrder::TyThenTlc => {
                        let backend = run_ty_backend(
                            spec,
                            workers,
                            case,
                            args,
                            repo_root,
                            &run_dir,
                            ty_bin,
                            args.backend,
                            backend_cli_name(args.backend),
                            &[],
                            execution_envelope,
                        )?;
                        let tlc = run_tlc(
                            spec,
                            workers,
                            args.timeout,
                            repo_root,
                            &run_dir,
                            "production-tlc",
                            tlc_runner,
                            execution_envelope,
                        )?;
                        Ok((tlc, backend))
                    }
                }
            };
            let run_count_pair = || -> Result<(RunObservation, RunObservation)> {
                match schedule.count_pair_order {
                    CompareRunOrder::TlcThenTy => {
                        let tlc = run_tlc(
                            spec,
                            workers,
                            args.timeout,
                            repo_root,
                            &run_dir,
                            "count-tlc",
                            tlc_runner,
                            execution_envelope,
                        )?;
                        let count_verify = run_ty_backend(
                            spec,
                            workers,
                            case,
                            args,
                            repo_root,
                            &run_dir,
                            ty_bin,
                            args.backend,
                            "count-verify",
                            &["--bfs-only", "--no-reduction"],
                            execution_envelope,
                        )?;
                        Ok((tlc, count_verify))
                    }
                    CompareRunOrder::TyThenTlc => {
                        let count_verify = run_ty_backend(
                            spec,
                            workers,
                            case,
                            args,
                            repo_root,
                            &run_dir,
                            ty_bin,
                            args.backend,
                            "count-verify",
                            &["--bfs-only", "--no-reduction"],
                            execution_envelope,
                        )?;
                        let tlc = run_tlc(
                            spec,
                            workers,
                            args.timeout,
                            repo_root,
                            &run_dir,
                            "count-tlc",
                            tlc_runner,
                            execution_envelope,
                        )?;
                        Ok((tlc, count_verify))
                    }
                }
            };
            let ((tlc, backend), (count_tlc, count_verify)) = match schedule.pair_block_order {
                ComparePairBlockOrder::ProductionThenCount => {
                    (run_production_pair()?, run_count_pair()?)
                }
                ComparePairBlockOrder::CountThenProduction => {
                    let count_pair = run_count_pair()?;
                    let production_pair = run_production_pair()?;
                    (production_pair, count_pair)
                }
                ComparePairBlockOrder::ProductionOnly => {
                    unreachable!("AUTO speed schedule always has two pair blocks")
                }
            };
            paired_runs.push(CompareRun::new_with_count_pair(
                repetition,
                schedule.production_pair_order,
                schedule.count_pair_order,
                schedule.pair_block_order,
                tlc,
                count_tlc,
                count_verify,
                backend,
            ));
            continue;
        }

        let order = if run_index % 2 == 0 {
            CompareRunOrder::TlcThenTy
        } else {
            CompareRunOrder::TyThenTlc
        };
        let (tlc_result, count_verify_result, backend_result) = match order {
            CompareRunOrder::TlcThenTy => {
                let tlc_result = run_tlc(
                    spec,
                    workers,
                    args.timeout,
                    repo_root,
                    &run_dir,
                    "tlc",
                    tlc_runner,
                    execution_envelope,
                )?;
                let backend_result = run_ty_backend(
                    spec,
                    workers,
                    case,
                    args,
                    repo_root,
                    &run_dir,
                    ty_bin,
                    args.backend,
                    backend_cli_name(args.backend),
                    &[],
                    execution_envelope,
                )?;
                let count_verify_result = if is_production_auto(args.backend) {
                    Some(run_ty_backend(
                        spec,
                        workers,
                        case,
                        args,
                        repo_root,
                        &run_dir,
                        ty_bin,
                        args.backend,
                        "count-verify",
                        &["--bfs-only", "--no-reduction"],
                        execution_envelope,
                    )?)
                } else {
                    None
                };
                (tlc_result, count_verify_result, backend_result)
            }
            CompareRunOrder::TyThenTlc => {
                let backend_result = run_ty_backend(
                    spec,
                    workers,
                    case,
                    args,
                    repo_root,
                    &run_dir,
                    ty_bin,
                    args.backend,
                    backend_cli_name(args.backend),
                    &[],
                    execution_envelope,
                )?;
                let tlc_result = run_tlc(
                    spec,
                    workers,
                    args.timeout,
                    repo_root,
                    &run_dir,
                    "tlc",
                    tlc_runner,
                    execution_envelope,
                )?;
                let count_verify_result = if is_production_auto(args.backend) {
                    Some(run_ty_backend(
                        spec,
                        workers,
                        case,
                        args,
                        repo_root,
                        &run_dir,
                        ty_bin,
                        args.backend,
                        "count-verify",
                        &["--bfs-only", "--no-reduction"],
                        execution_envelope,
                    )?)
                } else {
                    None
                };
                (tlc_result, count_verify_result, backend_result)
            }
        };
        paired_runs.push(CompareRun::new(
            repetition,
            order,
            tlc_result,
            count_verify_result,
            backend_result,
        ));
    }
    Ok(CompareRow::classify(
        spec,
        workers,
        &case.name,
        args.backend,
        args.policy,
        args.min_speedup,
        args.max_memory_ratio,
        paired_runs,
    ))
}

fn run_tlc(
    spec: &CompareSpec,
    workers: usize,
    timeout_seconds: u64,
    repo_root: &Path,
    spec_dir: &Path,
    artifact_name: &str,
    tlc_runner: &TlcRunner,
    execution_envelope: ExecutionEnvelope,
) -> Result<RunObservation> {
    let artifact_dir = spec_dir.join(artifact_name);
    let metadir = artifact_dir.join("tlc-metadir");
    let mut env_overrides = BTreeMap::new();
    let mut argv = match tlc_runner {
        TlcRunner::Executable {
            tlc_bin,
            tla_library,
        } => {
            if let Some(tla_library) = tla_library {
                env_overrides.insert(
                    "JAVA_TOOL_OPTIONS".to_string(),
                    format!("-DTLA-Library={}", tla_library.display()),
                );
            }
            vec![
                absolutize(repo_root, tlc_bin).display().to_string(),
                "-workers".to_string(),
                workers.to_string(),
            ]
        }
        TlcRunner::Java {
            tlc_jar,
            community_modules,
            tla_library,
        } => {
            let mut argv = tlc_java_single_thread_base_argv();
            if let Some(tla_library) = tla_library {
                argv.push(format!("-DTLA-Library={}", tla_library.display()));
            }
            argv.extend([
                "-cp".to_string(),
                tlc_classpath(tlc_jar, community_modules.as_deref())?,
                "tlc2.TLC".to_string(),
                "-workers".to_string(),
                workers.to_string(),
            ]);
            argv
        }
    };
    argv.extend([
        "-config".to_string(),
        spec.cfg_path.display().to_string(),
        "-metadir".to_string(),
        metadir.display().to_string(),
        spec.tla_path.display().to_string(),
    ]);
    let result = run_command_with_envelope(
        CommandSpec {
            argv,
            cwd: spec
                .tla_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            env_overrides,
            timeout_seconds,
            capture_limits: None,
            artifact_dir,
            payload_dir: None,
            observation_storage_contract: None,
            observation_storage_binding: None,
            tlc_metadir: None,
        },
        execution_envelope,
    )?;
    Ok(observe_tlc_run(result))
}

fn run_ty_backend(
    spec: &CompareSpec,
    workers: usize,
    case: &EnvCase,
    args: &SupremacyCompareArgs,
    repo_root: &Path,
    spec_dir: &Path,
    ty_bin: &Path,
    backend: SupremacyCompareBackend,
    artifact_name: &str,
    extra_flags: &[&str],
    execution_envelope: ExecutionEnvelope,
) -> Result<RunObservation> {
    let artifact_dir = spec_dir.join(artifact_name);
    let mut argv = vec![
        ty_bin.display().to_string(),
        "check".to_string(),
        spec.tla_path.display().to_string(),
        "--config".to_string(),
        spec.cfg_path.display().to_string(),
        "--workers".to_string(),
        workers.to_string(),
        "--force".to_string(),
    ];
    if backend == SupremacyCompareBackend::TrustCg {
        // Count-parity lever (was the TY_AUTO_POR/TY_AUTO_SYMMETRY env pins in
        // the protected trust-cg env): the child `ty check` ignores ambient
        // env for these semantic levers, so the flag is the only control.
        argv.push("--no-reduction".to_string());
    }
    argv.extend(extra_flags.iter().map(|flag| (*flag).to_string()));
    // User --ty-flag values are appended AFTER the arm's own flags; skip any
    // the argv already carries — `ty check` rejects duplicated flags, and a
    // user passing --ty-flag=--no-reduction alongside the count-verify arm's
    // built-in --no-reduction killed the child with a usage error (rc 2)
    // before it ran anything.
    for flag in &args.ty_flag {
        if !argv.contains(flag) {
            argv.push(flag.clone());
        }
    }
    append_ty_backend_args(&mut argv, backend);
    let mut env_overrides = backend_env(backend);
    // Diagnostic-only provenance: the child reports the execution tier that
    // actually completed the run. Strict evidence fails closed when this is
    // absent or changes across repetitions.
    env_overrides.insert("TY_ENGINE_TIER".to_string(), "1".to_string());
    if matches!(
        backend,
        SupremacyCompareBackend::TrustCg
            | SupremacyCompareBackend::Auto
            | SupremacyCompareBackend::AutoCpu
    ) {
        env_overrides.insert(
            "TY_CACHE_DIR".to_string(),
            artifact_dir
                .join("trust_cg-artifact-cache")
                .display()
                .to_string(),
        );
    }
    if let Some(tla_library) = resolve_tla_library(args, repo_root) {
        env_overrides.insert("TLA_LIBRARY".to_string(), tla_library.display().to_string());
    }
    env_overrides.extend(case.env_overrides.clone());
    let result = run_command_with_envelope(
        CommandSpec {
            argv,
            cwd: repo_root.to_path_buf(),
            env_overrides,
            timeout_seconds: args.timeout,
            capture_limits: None,
            artifact_dir,
            payload_dir: None,
            observation_storage_contract: None,
            observation_storage_binding: None,
            tlc_metadir: None,
        },
        execution_envelope,
    )?;
    let mut observation = observe_ty_run(result, backend);
    observation.mode = artifact_name.to_string();
    Ok(observation)
}

fn append_ty_backend_args(argv: &mut Vec<String>, backend: SupremacyCompareBackend) {
    match backend {
        SupremacyCompareBackend::Interpreter | SupremacyCompareBackend::TrustCg => {
            argv.extend([
                "--backend".to_string(),
                backend_cli_name(backend).to_string(),
            ]);
        }
        // Production AUTO: no --backend flag — the child routes exactly as a
        // user's `ty check` would (burndown P4). auto-cpu excludes the GPU so
        // rows stay single-thread-eligible on CUDA hosts.
        SupremacyCompareBackend::Auto => {}
        SupremacyCompareBackend::AutoCpu => argv.push("--no-gpu".to_string()),
    }
}

#[derive(Clone, Debug, Serialize)]
struct RunObservation {
    tool: String,
    mode: String,
    status: String,
    elapsed_seconds: f64,
    resource_evidence: ResourceEvidence,
    disk_high_water: DiskHighWaterEvidence,
    states_found: Option<u64>,
    transitions: Option<u64>,
    raw_initial_states_generated: Option<u64>,
    raw_successors_generated: Option<u64>,
    states_generated: Option<u64>,
    returncode: i32,
    timed_out: bool,
    error_type: Option<String>,
    violated_obligation: Option<String>,
    error: Option<String>,
    artifact_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    engine_tier: Option<String>,
}

fn process_tree_peak_memory_bytes(observation: &RunObservation) -> Option<u64> {
    (observation.resource_evidence.memory.scope == PeakMemoryScope::ProcessTree)
        .then_some(observation.resource_evidence.memory.peak_bytes)
        .flatten()
}

fn sampled_peak_allocated_disk_bytes(observation: &RunObservation) -> Option<u64> {
    observation.disk_high_water.peak_allocated_bytes
}

fn sampled_peak_apparent_disk_bytes(observation: &RunObservation) -> Option<u64> {
    observation.disk_high_water.peak_apparent_bytes
}

fn observe_tlc_run(result: CommandResult) -> RunObservation {
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_tlc_final_counts(&stdout, &stderr);
    let error_type = run_error_type(result.returncode, result.timed_out, &stdout, &stderr);
    let violated_obligation =
        violation_obligation(error_type.as_deref(), &format!("{stdout}\n{stderr}"));
    let error = run_error(
        result.returncode,
        result.timed_out,
        &stderr,
        counts.states_found,
    );
    let artifact_dir = result
        .disk_high_water
        .scope_root
        .clone()
        .unwrap_or_else(|| result.artifact_dir.display().to_string());
    RunObservation {
        tool: "tlc".to_string(),
        mode: "tlc".to_string(),
        status: status_for_error(error_type.as_deref()).to_string(),
        elapsed_seconds: result.elapsed_seconds,
        resource_evidence: result.resource_evidence,
        disk_high_water: result.disk_high_water,
        states_found: counts.states_found,
        transitions: counts.transitions,
        raw_initial_states_generated: counts.raw_initial_states_generated,
        raw_successors_generated: counts.raw_successors_generated,
        states_generated: counts.states_generated,
        returncode: result.returncode,
        timed_out: result.timed_out,
        error_type,
        violated_obligation,
        error,
        artifact_dir,
        engine_tier: None,
    }
}

/// Extract the final tier report: AUTO may tier up and re-report during a run.
fn parse_engine_tier(stderr: &str) -> Option<String> {
    stderr
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix("[engine] execution tier: "))
        .map(str::trim)
        .filter(|tier| !tier.is_empty())
        .map(str::to_string)
}

fn observe_ty_run(result: CommandResult, backend: SupremacyCompareBackend) -> RunObservation {
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let counts = parse::parse_ty_final_counts(&stdout, &stderr);
    let engine_tier = parse_engine_tier(&stderr);
    let error_type = run_error_type(result.returncode, result.timed_out, &stdout, &stderr);
    let violated_obligation =
        violation_obligation(error_type.as_deref(), &format!("{stdout}\n{stderr}"));
    let error = run_error(
        result.returncode,
        result.timed_out,
        &stderr,
        counts.states_found,
    );
    let artifact_dir = result
        .disk_high_water
        .scope_root
        .clone()
        .unwrap_or_else(|| result.artifact_dir.display().to_string());
    RunObservation {
        tool: "ty".to_string(),
        mode: backend_cli_name(backend).to_string(),
        status: status_for_error(error_type.as_deref()).to_string(),
        elapsed_seconds: result.elapsed_seconds,
        resource_evidence: result.resource_evidence,
        disk_high_water: result.disk_high_water,
        states_found: counts.states_found,
        transitions: counts.transitions,
        raw_initial_states_generated: counts.raw_initial_states_generated,
        raw_successors_generated: counts.raw_successors_generated,
        states_generated: counts.states_generated,
        returncode: result.returncode,
        timed_out: result.timed_out,
        error_type,
        violated_obligation,
        error,
        artifact_dir,
        engine_tier,
    }
}

fn run_error(
    returncode: i32,
    timed_out: bool,
    stderr: &str,
    required_states: Option<u64>,
) -> Option<String> {
    if timed_out {
        return Some("timeout".to_string());
    }
    if returncode != 0 {
        return Some(
            first_error_line(stderr).unwrap_or_else(|| format!("returncode {returncode}")),
        );
    }
    if required_states.is_none() {
        return Some("missing states_found".to_string());
    }
    None
}

fn first_error_line(stderr: &str) -> Option<String> {
    stderr
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.chars().take(240).collect())
}

fn run_error_type(returncode: i32, timed_out: bool, stdout: &str, stderr: &str) -> Option<String> {
    if timed_out {
        return Some("timeout".to_string());
    }
    if returncode == 0 {
        return None;
    }
    Some(classify_output_error_type(stdout, stderr))
}

fn classify_output_error_type(stdout: &str, stderr: &str) -> String {
    let output = format!("{stdout}\n{stderr}");
    let lower = output.to_ascii_lowercase();
    if has_liveness_violation_marker(&lower) {
        "liveness".to_string()
    } else if lower.lines().any(|line| {
        (line.contains("action property") || line.contains("property")) && line.contains("violated")
    }) {
        "property".to_string()
    } else if has_invariant_violation_marker(&lower) {
        "invariant".to_string()
    } else if lower.lines().any(|line| {
        (line.contains("assumption") || line.contains("assume")) && line.contains("violated")
    }) {
        "assume_violation".to_string()
    } else if lower.contains("deadlock") {
        "deadlock".to_string()
    } else if lower.contains("parse") || lower.contains("syntax") {
        "parse".to_string()
    } else if lower.contains("unsupported") || lower.contains("not supported") {
        "unsupported".to_string()
    } else if lower.contains("action") && lower.contains("failed") {
        "action".to_string()
    } else if lower.contains("safety") {
        "safety".to_string()
    } else {
        "unknown".to_string()
    }
}

fn has_liveness_violation_marker(output_lower: &str) -> bool {
    output_lower.lines().any(|line| {
        let line = line.trim();
        line.contains("temporal properties were violated")
            || line.contains("liveness violation")
            || (line.contains("liveness property") && line.contains("violated"))
    })
}

fn has_invariant_violation_marker(output_lower: &str) -> bool {
    output_lower.lines().any(|line| {
        let line = line.trim();
        line.contains("invariant") && line.contains("violated")
    })
}

fn status_for_error(error_type: Option<&str>) -> &'static str {
    match error_type {
        None => "pass",
        Some("timeout") => "timeout",
        Some(_) => "fail",
    }
}

fn expected_error_type(mode: &SpecBaselineMode) -> Option<String> {
    if let Some(error_type) = &mode.error_type {
        return Some(normalize_error_type(error_type));
    }
    match mode.status.as_deref() {
        Some("timeout") => Some("timeout".to_string()),
        Some("error" | "fail") => Some("unknown".to_string()),
        _ => None,
    }
}

fn normalize_error_type(error_type: &str) -> String {
    match error_type {
        "invariant_violation" => "invariant".to_string(),
        "liveness_violation" => "liveness".to_string(),
        value if value.starts_with("timeout") => "timeout".to_string(),
        value => value.to_string(),
    }
}

fn error_types_compatible(tlc: Option<&str>, backend: Option<&str>) -> bool {
    tlc == backend
        || matches!(
            (tlc, backend),
            (Some("invariant"), Some("safety")) | (Some("safety"), Some("invariant"))
        )
}

fn violation_obligation(error_type: Option<&str>, text: &str) -> Option<String> {
    let markers: &[&str] = match error_type? {
        "invariant" | "safety" => &["invariant ", "invariant '"],
        "liveness" => &["temporal property ", "liveness property ", "property "],
        "action" | "property" => &["action property ", "property ", "property '"],
        "assume_violation" => &["assumption ", "assume "],
        _ => return None,
    };
    markers
        .iter()
        .find_map(|marker| violation_name_after_marker(text, marker))
}

fn violation_name_after_marker(text: &str, marker: &str) -> Option<String> {
    let marker_lower = marker.to_ascii_lowercase();
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        let Some(marker_index) = lower.find(&marker_lower) else {
            continue;
        };
        let name_start = marker_index + marker_lower.len();
        let rest = &lower[name_start..];
        let Some(name_end) = [" is violated", " violated", " has been violated"]
            .into_iter()
            .filter_map(|suffix| rest.find(suffix))
            .min()
            .map(|offset| name_start + offset)
        else {
            continue;
        };
        let name = line[name_start..name_end]
            .trim()
            .trim_matches(|ch: char| {
                matches!(
                    ch,
                    '\'' | '"' | '`' | ':' | ';' | ',' | '.' | '(' | ')' | '[' | ']'
                )
            })
            .trim();
        if !name.is_empty() && !matches!(name.to_ascii_lowercase().as_str(), "is" | "was") {
            return Some(name.to_string());
        }
    }
    None
}

fn violated_obligations_compatible(
    error_type: Option<&str>,
    left: Option<&str>,
    right: Option<&str>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        (None, None) => !matches!(
            error_type,
            Some("invariant" | "safety" | "liveness" | "action" | "property" | "assume_violation")
        ),
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CompareRunOrder {
    TlcThenTy,
    TyThenTlc,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ComparePairBlockOrder {
    ProductionThenCount,
    CountThenProduction,
    ProductionOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AutoPerformanceSchedule {
    production_pair_order: CompareRunOrder,
    count_pair_order: CompareRunOrder,
    pair_block_order: ComparePairBlockOrder,
}

/// Two-period crossover: both within-pair orders and both pair-block orders
/// have equal representation whenever the repetition count is even.
fn auto_performance_schedule(run_index: usize) -> AutoPerformanceSchedule {
    if run_index % 2 == 1 {
        AutoPerformanceSchedule {
            production_pair_order: CompareRunOrder::TlcThenTy,
            count_pair_order: CompareRunOrder::TyThenTlc,
            pair_block_order: ComparePairBlockOrder::ProductionThenCount,
        }
    } else {
        AutoPerformanceSchedule {
            production_pair_order: CompareRunOrder::TyThenTlc,
            count_pair_order: CompareRunOrder::TlcThenTy,
            pair_block_order: ComparePairBlockOrder::CountThenProduction,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CompareAxis {
    NotRequired,
    Pass,
    Loss,
    MissingOrStale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ClaimClass {
    PassBoth,
    RuntimeLoss,
    MemoryLoss,
    BothLoss,
    ParityBlocker,
    MissingOrStale,
}

#[derive(Clone, Debug, Serialize)]
struct CompareRun {
    run_index: usize,
    production_pair_order: CompareRunOrder,
    #[serde(skip_serializing_if = "Option::is_none")]
    count_pair_order: Option<CompareRunOrder>,
    pair_block_order: ComparePairBlockOrder,
    tlc: RunObservation,
    #[serde(skip_serializing_if = "Option::is_none")]
    count_tlc_run: Option<RunObservation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    count_verify_run: Option<RunObservation>,
    backend_run: RunObservation,
    speedup_tlc_vs_backend: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speedup_count_tlc_vs_count_verify: Option<f64>,
    memory_ratio_backend_vs_tlc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_ratio_count_verify_vs_count_tlc: Option<f64>,
}

impl CompareRun {
    fn new(
        run_index: usize,
        order: CompareRunOrder,
        tlc: RunObservation,
        count_verify_run: Option<RunObservation>,
        backend_run: RunObservation,
    ) -> Self {
        Self {
            run_index,
            production_pair_order: order,
            count_pair_order: None,
            pair_block_order: if count_verify_run.is_some() {
                ComparePairBlockOrder::ProductionThenCount
            } else {
                ComparePairBlockOrder::ProductionOnly
            },
            count_tlc_run: None,
            count_verify_run,
            speedup_tlc_vs_backend: speedup(tlc.elapsed_seconds, backend_run.elapsed_seconds),
            speedup_count_tlc_vs_count_verify: None,
            memory_ratio_backend_vs_tlc: memory_ratio(
                process_tree_peak_memory_bytes(&backend_run),
                process_tree_peak_memory_bytes(&tlc),
            ),
            memory_ratio_count_verify_vs_count_tlc: None,
            tlc,
            backend_run,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_count_pair(
        run_index: usize,
        production_pair_order: CompareRunOrder,
        count_pair_order: CompareRunOrder,
        pair_block_order: ComparePairBlockOrder,
        tlc: RunObservation,
        count_tlc_run: RunObservation,
        count_verify_run: RunObservation,
        backend_run: RunObservation,
    ) -> Self {
        Self {
            run_index,
            production_pair_order,
            count_pair_order: Some(count_pair_order),
            pair_block_order,
            speedup_tlc_vs_backend: speedup(tlc.elapsed_seconds, backend_run.elapsed_seconds),
            speedup_count_tlc_vs_count_verify: speedup(
                count_tlc_run.elapsed_seconds,
                count_verify_run.elapsed_seconds,
            ),
            memory_ratio_backend_vs_tlc: memory_ratio(
                process_tree_peak_memory_bytes(&backend_run),
                process_tree_peak_memory_bytes(&tlc),
            ),
            memory_ratio_count_verify_vs_count_tlc: memory_ratio(
                process_tree_peak_memory_bytes(&count_verify_run),
                process_tree_peak_memory_bytes(&count_tlc_run),
            ),
            tlc,
            count_tlc_run: Some(count_tlc_run),
            count_verify_run: Some(count_verify_run),
            backend_run,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CompareClass {
    Pass,
    TlcFailed,
    BackendFailed,
    ExpectedStateMismatch,
    ExpectedErrorMismatch,
    ErrorMismatch,
    ParityFail,
    MissingRuntime,
    SpeedFail,
    MissingMemory,
    MemoryFail,
    BothFail,
    MissingEvidence,
}

#[derive(Clone, Debug, Serialize)]
struct CompareRow {
    spec: String,
    workers: usize,
    case: String,
    backend: String,
    class: CompareClass,
    claim_class: ClaimClass,
    runtime_axis: CompareAxis,
    memory_axis: CompareAxis,
    passed: bool,
    reason: String,
    tlc: RunObservation,
    #[serde(skip_serializing_if = "Option::is_none")]
    count_tlc_run: Option<RunObservation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    count_verify_run: Option<RunObservation>,
    backend_run: RunObservation,
    parity_states: bool,
    parity_generated_work: bool,
    expected_tlc_states: Option<u64>,
    expected_backend_states: Option<u64>,
    expected_tlc_error: Option<String>,
    expected_backend_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    work_equivalence: Option<WorkEquivalenceEvidence>,
    speedup_tlc_vs_backend: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speedup_count_tlc_vs_count_verify: Option<f64>,
    memory_ratio_backend_vs_tlc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_ratio_count_verify_vs_count_tlc: Option<f64>,
    policy: String,
    min_speedup: f64,
    max_memory_ratio: f64,
    run_count: usize,
    paired_statistic: &'static str,
    runs: Vec<CompareRun>,
}

impl CompareRow {
    fn classify(
        spec: &CompareSpec,
        workers: usize,
        case: &str,
        backend: SupremacyCompareBackend,
        policy: SupremacyComparePolicy,
        min_speedup: f64,
        max_memory_ratio: f64,
        runs: Vec<CompareRun>,
    ) -> Self {
        debug_assert!(!runs.is_empty());
        let tlc_observations = runs.iter().map(|run| run.tlc.clone()).collect::<Vec<_>>();
        let backend_observations = runs
            .iter()
            .map(|run| run.backend_run.clone())
            .collect::<Vec<_>>();
        let count_tlc_observations = runs
            .iter()
            .filter_map(|run| run.count_tlc_run.clone())
            .collect::<Vec<_>>();
        let count_verify_observations = runs
            .iter()
            .filter_map(|run| run.count_verify_run.clone())
            .collect::<Vec<_>>();
        let tlc = aggregate_observations(&tlc_observations);
        let backend_run = aggregate_observations(&backend_observations);
        let count_tlc_run = (!count_tlc_observations.is_empty())
            .then(|| aggregate_observations(&count_tlc_observations));
        let count_verify_run = (!count_verify_observations.is_empty())
            .then(|| aggregate_observations(&count_verify_observations));
        let speedup_values = runs
            .iter()
            .filter_map(|run| run.speedup_tlc_vs_backend)
            .collect::<Vec<_>>();
        let speedup = (speedup_values.len() == runs.len())
            .then(|| median_f64(speedup_values))
            .flatten();
        let count_speedup_values = runs
            .iter()
            .filter_map(|run| run.speedup_count_tlc_vs_count_verify)
            .collect::<Vec<_>>();
        let count_speedup = (count_speedup_values.len() == runs.len())
            .then(|| median_f64(count_speedup_values))
            .flatten();
        let memory_ratio_values = runs
            .iter()
            .filter_map(|run| run.memory_ratio_backend_vs_tlc)
            .collect::<Vec<_>>();
        let memory_ratio = (memory_ratio_values.len() == runs.len())
            .then(|| median_f64(memory_ratio_values))
            .flatten();
        let count_memory_ratio_values = runs
            .iter()
            .filter_map(|run| run.memory_ratio_count_verify_vs_count_tlc)
            .collect::<Vec<_>>();
        let count_memory_ratio = (count_memory_ratio_values.len() == runs.len())
            .then(|| median_f64(count_memory_ratio_values))
            .flatten();
        let classification = classify_repeated_observations(
            spec,
            backend,
            policy,
            min_speedup,
            max_memory_ratio,
            &runs,
            &tlc,
            &backend_run,
            speedup,
            count_speedup,
            memory_ratio,
        );
        let run_count = runs.len();
        let claim_class = claim_class(&classification, policy, min_speedup, max_memory_ratio);
        Self {
            spec: spec.name.clone(),
            workers,
            case: case.to_string(),
            backend: backend_cli_name(backend).to_string(),
            passed: classification.class == CompareClass::Pass,
            class: classification.class,
            claim_class,
            runtime_axis: classification.runtime_axis,
            memory_axis: classification.memory_axis,
            reason: classification.reason,
            tlc,
            count_tlc_run,
            count_verify_run,
            backend_run,
            parity_states: classification.parity_states,
            parity_generated_work: classification.parity_generated_work,
            expected_tlc_states: spec.expected_tlc_states,
            expected_backend_states: spec.expected_backend_states,
            expected_tlc_error: spec.expected_tlc_error.clone(),
            expected_backend_error: spec.expected_backend_error.clone(),
            work_equivalence: spec.work_equivalence.clone(),
            speedup_tlc_vs_backend: speedup,
            speedup_count_tlc_vs_count_verify: count_speedup,
            memory_ratio_backend_vs_tlc: memory_ratio,
            memory_ratio_count_verify_vs_count_tlc: count_memory_ratio,
            policy: policy_name(policy).to_string(),
            min_speedup,
            max_memory_ratio,
            run_count,
            paired_statistic: PAIRED_STATISTIC,
            runs,
        }
    }
}

#[derive(Clone, Debug)]
struct Classification {
    class: CompareClass,
    reason: String,
    parity_states: bool,
    parity_generated_work: bool,
    runtime_axis: CompareAxis,
    memory_axis: CompareAxis,
}

#[allow(clippy::too_many_arguments)]
fn classify_repeated_observations(
    spec: &CompareSpec,
    backend: SupremacyCompareBackend,
    policy: SupremacyComparePolicy,
    min_speedup: f64,
    max_memory_ratio: f64,
    runs: &[CompareRun],
    aggregate_tlc: &RunObservation,
    aggregate_backend: &RunObservation,
    paired_speedup: Option<f64>,
    paired_count_speedup: Option<f64>,
    paired_memory_ratio: Option<f64>,
) -> Classification {
    let production_arm = is_production_auto(backend);
    let count_performance_required = production_arm && policy_checks_speed(policy);
    let parity_observations = if production_arm {
        let observations = runs
            .iter()
            .filter_map(|run| run.count_verify_run.as_ref())
            .collect::<Vec<_>>();
        if observations.len() != runs.len() {
            if count_performance_required {
                return Classification {
                    class: CompareClass::MissingEvidence,
                    reason:
                        "production comparison is missing a timed count-verification observation"
                            .to_string(),
                    parity_states: false,
                    parity_generated_work: false,
                    runtime_axis: CompareAxis::MissingOrStale,
                    memory_axis: if policy_checks_memory(policy) {
                        CompareAxis::MissingOrStale
                    } else {
                        CompareAxis::NotRequired
                    },
                };
            }
            return classified(
                CompareClass::ParityFail,
                "production comparison is missing a count-verify arm".to_string(),
                false,
            );
        }
        observations
    } else {
        runs.iter().map(|run| &run.backend_run).collect()
    };
    let parity_tlc_observations = if count_performance_required {
        let observations = runs
            .iter()
            .filter_map(|run| run.count_tlc_run.as_ref())
            .collect::<Vec<_>>();
        if observations.len() != runs.len() {
            return Classification {
                class: CompareClass::MissingEvidence,
                reason: "production comparison is missing a paired count-arm TLC observation"
                    .to_string(),
                parity_states: false,
                parity_generated_work: false,
                runtime_axis: CompareAxis::MissingOrStale,
                memory_axis: if policy_checks_memory(policy) {
                    CompareAxis::MissingOrStale
                } else {
                    CompareAxis::NotRequired
                },
            };
        }
        observations
    } else {
        runs.iter().map(|run| &run.tlc).collect()
    };

    if policy_checks_speed(policy) {
        if let Some(reason) = missing_engine_provenance_reason(runs, production_arm) {
            return Classification {
                class: CompareClass::MissingEvidence,
                reason,
                parity_states: false,
                parity_generated_work: false,
                runtime_axis: CompareAxis::MissingOrStale,
                memory_axis: if policy_checks_memory(policy) {
                    CompareAxis::MissingOrStale
                } else {
                    CompareAxis::NotRequired
                },
            };
        }
    }
    if let Some(reason) =
        unstable_observation_reason("TLC", &runs.iter().map(|run| &run.tlc).collect::<Vec<_>>())
    {
        return classified(CompareClass::ParityFail, reason, false);
    }
    if count_performance_required {
        if let Some(reason) = unstable_observation_reason("count-arm TLC", &parity_tlc_observations)
        {
            return classified(CompareClass::ParityFail, reason, false);
        }
    }
    if let Some(reason) = unstable_observation_reason("count-verify backend", &parity_observations)
    {
        return classified(CompareClass::ParityFail, reason, false);
    }
    if production_arm {
        if let Some(reason) = unstable_observation_reason(
            "production backend",
            &runs.iter().map(|run| &run.backend_run).collect::<Vec<_>>(),
        ) {
            return classified(CompareClass::ParityFail, reason, false);
        }
    }
    for (index, run) in runs.iter().enumerate() {
        let parity_tlc = parity_tlc_observations[index];
        let parity_backend = parity_observations[index];
        if count_performance_required && !same_semantic_observation(&run.tlc, parity_tlc) {
            return classified(
                CompareClass::ParityFail,
                format!(
                    "run {}: production-pair TLC and count-pair TLC outcomes differ",
                    run.run_index
                ),
                false,
            );
        }
        let parity = classify_observations_with_limits(
            spec.expected_tlc_states,
            spec.expected_backend_states,
            spec.expected_tlc_error.as_deref(),
            spec.expected_backend_error.as_deref(),
            SupremacyComparePolicy::Parity,
            min_speedup,
            max_memory_ratio,
            parity_tlc,
            parity_backend,
            None,
            None,
        );
        if parity.class != CompareClass::Pass {
            return Classification {
                reason: format!("run {}: {}", run.run_index, parity.reason),
                ..parity
            };
        }
        if production_arm && !outcomes_match(parity_backend, &run.backend_run) {
            return classified(
                CompareClass::ParityFail,
                format!(
                    "run {}: production outcome differs from count-verify outcome: count_verify={:?}, production={:?}",
                    run.run_index, parity_backend.error_type, run.backend_run.error_type
                ),
                true,
            );
        }
    }

    if policy_checks_speed(policy) {
        let work_equivalence_failure = if parity_observations
            .iter()
            .any(|observation| observation.error_type.is_some())
        {
            Some(
                "matching early-violation outcomes cannot qualify under the exhaustive work_equivalence rule; first-found counterexamples are correctness-only evidence"
                    .to_string(),
            )
        } else if spec
            .work_equivalence
            .as_ref()
            .is_none_or(|evidence| !evidence.qualifies(WorkEquivalenceVerdict::Holds))
        {
            Some(
                "successful performance observations lack exact typed work_equivalence evidence for exhaustive raw initial, raw successor, total generated, and distinct-state parity"
                    .to_string(),
            )
        } else {
            None
        };
        if let Some(reason) = work_equivalence_failure {
            return Classification {
                class: CompareClass::MissingEvidence,
                reason,
                parity_states: true,
                parity_generated_work: true,
                runtime_axis: CompareAxis::MissingOrStale,
                memory_axis: if policy_checks_memory(policy) {
                    CompareAxis::MissingOrStale
                } else {
                    CompareAxis::NotRequired
                },
            };
        }
    }

    let policy_speedup = required_runtime_speedup(
        paired_speedup,
        paired_count_speedup,
        count_performance_required,
    );

    if policy_checks_speed(policy) {
        if let Some(reason) =
            strict_disk_evidence_reason(runs, production_arm, count_performance_required)
        {
            return Classification {
                class: CompareClass::MissingEvidence,
                reason,
                parity_states: true,
                parity_generated_work: true,
                runtime_axis: CompareAxis::MissingOrStale,
                memory_axis: if policy_checks_memory(policy) {
                    CompareAxis::MissingOrStale
                } else {
                    CompareAxis::NotRequired
                },
            };
        }
        if let Some(reason) =
            strict_cpu_evidence_reason(runs, production_arm, count_performance_required)
        {
            return Classification {
                class: CompareClass::MissingEvidence,
                reason,
                parity_states: true,
                parity_generated_work: true,
                runtime_axis: CompareAxis::MissingOrStale,
                memory_axis: if policy_checks_memory(policy) {
                    CompareAxis::MissingOrStale
                } else {
                    CompareAxis::NotRequired
                },
            };
        }
        if !policy_checks_memory(policy) {
            if let Some(reason) =
                strict_envelope_evidence_reason(runs, production_arm, count_performance_required)
            {
                return Classification {
                    class: CompareClass::MissingEvidence,
                    reason,
                    parity_states: true,
                    parity_generated_work: true,
                    runtime_axis: CompareAxis::MissingOrStale,
                    memory_axis: CompareAxis::NotRequired,
                };
            }
        }
        if policy_checks_memory(policy) {
            if let Some(reason) =
                strict_memory_evidence_reason(runs, production_arm, count_performance_required)
            {
                let runtime_axis = match policy_speedup {
                    Some(value) if value > min_speedup => CompareAxis::Pass,
                    Some(_) => CompareAxis::Loss,
                    None => CompareAxis::MissingOrStale,
                };
                return Classification {
                    class: if runtime_axis == CompareAxis::Pass {
                        CompareClass::MissingMemory
                    } else {
                        CompareClass::MissingEvidence
                    },
                    reason,
                    parity_states: true,
                    parity_generated_work: true,
                    runtime_axis,
                    memory_axis: CompareAxis::MissingOrStale,
                };
            }
        }
    }

    let aggregate_parity_backend = if production_arm {
        aggregate_observations(
            &parity_observations
                .iter()
                .map(|observation| (*observation).clone())
                .collect::<Vec<_>>(),
        )
    } else {
        aggregate_backend.clone()
    };
    let aggregate_parity_tlc = if count_performance_required {
        aggregate_observations(
            &parity_tlc_observations
                .iter()
                .map(|observation| (*observation).clone())
                .collect::<Vec<_>>(),
        )
    } else {
        aggregate_tlc.clone()
    };
    let parity = classify_observations_with_limits(
        spec.expected_tlc_states,
        spec.expected_backend_states,
        spec.expected_tlc_error.as_deref(),
        spec.expected_backend_error.as_deref(),
        SupremacyComparePolicy::Parity,
        min_speedup,
        max_memory_ratio,
        &aggregate_parity_tlc,
        &aggregate_parity_backend,
        None,
        None,
    );
    if parity.class != CompareClass::Pass {
        return parity;
    }
    let performance = evaluate_performance(
        policy,
        min_speedup,
        max_memory_ratio,
        policy_speedup,
        paired_memory_ratio,
    );
    if let Some(mut failure) = performance.failure() {
        if count_performance_required && failure.runtime_axis != CompareAxis::Pass {
            failure.reason = format!(
                "production paired speedup={}, count paired speedup={}; {}",
                format_optional_ratio(paired_speedup),
                format_optional_ratio(paired_count_speedup),
                failure.reason
            );
        }
        return failure;
    }
    Classification {
        class: CompareClass::Pass,
        reason: if production_arm {
            if count_performance_required {
                "count-verify parity passed on every independently paired run; production AUTO and count-verification both passed the median paired runtime policy, and production AUTO passed the process-tree-memory policy".to_string()
            } else {
                "count-verify parity passed on every run; production AUTO passed the requested policy"
                    .to_string()
            }
        } else {
            "parity passed on every run; median paired runtime/process-tree-memory policy passed"
                .to_string()
        },
        parity_states: parity.parity_states,
        parity_generated_work: parity.parity_generated_work,
        runtime_axis: performance.runtime_axis,
        memory_axis: performance.memory_axis,
    }
}

fn resource_observations(
    runs: &[CompareRun],
    production_arm: bool,
    count_performance_required: bool,
) -> Result<Vec<(usize, &'static str, &RunObservation)>, String> {
    let mut observations = Vec::new();
    for run in runs {
        observations.push((run.run_index, "tlc", &run.tlc));
        observations.push((run.run_index, "backend", &run.backend_run));
        if production_arm {
            let Some(count_verify) = run.count_verify_run.as_ref() else {
                return Err(format!(
                    "run {}: strict resource evidence is missing for count-verify",
                    run.run_index
                ));
            };
            observations.push((run.run_index, "count-verify", count_verify));
            if count_performance_required {
                let Some(count_tlc) = run.count_tlc_run.as_ref() else {
                    return Err(format!(
                        "run {}: strict resource evidence is missing for count-arm TLC",
                        run.run_index
                    ));
                };
                observations.push((run.run_index, "count-tlc", count_tlc));
            }
        }
    }
    Ok(observations)
}

fn strict_disk_evidence_reason(
    runs: &[CompareRun],
    production_arm: bool,
    count_performance_required: bool,
) -> Option<String> {
    let observations = match resource_observations(runs, production_arm, count_performance_required)
    {
        Ok(observations) => observations,
        Err(reason) => return Some(reason),
    };
    let sampling_interval_ms =
        u64::try_from(DISK_USAGE_SAMPLE_INTERVAL.as_millis()).expect("interval fits u64");
    let scan_budget_ms =
        u64::try_from(DISK_USAGE_SCAN_BUDGET.as_millis()).expect("budget fits u64");
    for (run_index, label, observation) in observations {
        let disk = &observation.disk_high_water;
        let mut failures = Vec::new();
        if disk.contract_schema != DISK_SCOPE_CONTRACT_SCHEMA {
            failures.push(format!(
                "contract_schema={:?}, expected {:?}",
                disk.contract_schema, DISK_SCOPE_CONTRACT_SCHEMA
            ));
        }
        if disk.scope != DiskHighWaterScope::CommandArtifactAndScratchTree {
            failures.push("scope was not command_artifact_and_scratch_tree".to_string());
        }
        if disk.method != DiskHighWaterMethod::RecursiveFilesystemMetadataPolling {
            failures.push("method was not recursive_filesystem_metadata_polling".to_string());
        }
        if disk.peak_exact {
            failures.push("peak_exact must be false for sampled directory polling".to_string());
        }
        if disk.sampling_execution != DiskSamplingExecution::InlineRunnerPollLoop {
            failures.push("sampling_execution was not inline_runner_poll_loop".to_string());
        }
        if !disk.sampling_can_perturb_elapsed {
            failures.push("sampling_can_perturb_elapsed was not disclosed".to_string());
        }
        if disk.sampling_interval_ms != sampling_interval_ms
            || disk.scan_budget_ms != scan_budget_ms
            || disk.scan_entry_limit != DISK_USAGE_SCAN_ENTRY_LIMIT
        {
            failures.push(format!(
                "sampling bounds were interval={}ms budget={}ms entries={}, expected {sampling_interval_ms}ms/{scan_budget_ms}ms/{DISK_USAGE_SCAN_ENTRY_LIMIT}",
                disk.sampling_interval_ms, disk.scan_budget_ms, disk.scan_entry_limit
            ));
        }
        if disk.max_scan_nanoseconds > disk.total_scan_nanoseconds {
            failures.push("scan timing evidence was inconsistent".to_string());
        }
        if disk.samples_attempted < 2
            || disk.samples_complete != disk.samples_attempted
            || disk.samples_partial != 0
        {
            failures.push(format!(
                "sample accounting was attempted={} complete={} partial={}",
                disk.samples_attempted, disk.samples_complete, disk.samples_partial
            ));
        }
        if disk.peak_allocated_bytes.is_none()
            || disk.peak_apparent_bytes.is_none()
            || disk.peak_entries_observed == 0
        {
            failures
                .push("allocated/apparent peak or observed-entry evidence was missing".to_string());
        }
        if !disk.initial_sample_complete
            || !disk.final_sample_complete
            || !disk.setup_complete
            || !disk.environment_confinement_complete
            || !disk.scope_identity_stable
            || !disk.ownership_verified
            || !disk.accounting_complete
            || !disk.polling_complete
            || !disk.process_tree_lifetime_complete
            || !disk.complete
            || !disk.strict_qualified
        {
            failures.push("one or more strict disk completeness flags were false".to_string());
        }
        if !disk.diagnostics.is_empty() || !disk.qualification_failures.is_empty() {
            failures.push(format!(
                "diagnostics={:?} qualification_failures={:?}",
                disk.diagnostics, disk.qualification_failures
            ));
        }

        let scope_root = disk.scope_root.as_deref().map(Path::new);
        let scratch_root = disk.scratch_root.as_deref().map(Path::new);
        if scope_root.is_none_or(|path| !path.is_absolute()) {
            failures.push("scope_root was missing or non-absolute".to_string());
        }
        if scratch_root.is_none_or(|path| !path.is_absolute()) {
            failures.push("scratch_root was missing or non-absolute".to_string());
        }
        if let (Some(scope_root), Some(scratch_root)) = (scope_root, scratch_root) {
            if scratch_root != scope_root.join(COMMAND_SCRATCH_DIR_NAME) {
                failures.push(
                    "scratch_root was not the canonical direct command-scratch child".to_string(),
                );
            }
            if Path::new(&observation.artifact_dir) != scope_root {
                failures.push("scope_root did not match the run artifact directory".to_string());
            }
            let expected_keys = COMMAND_SCOPED_ENV_KEYS
                .iter()
                .map(|key| (*key).to_string())
                .collect::<BTreeSet<_>>();
            let actual_keys = disk
                .environment_confinement
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            if actual_keys != expected_keys
                || COMMAND_SCOPED_ENV_KEYS.iter().any(|key| {
                    disk.environment_confinement
                        .get(*key)
                        .is_none_or(|value| Path::new(value) != scratch_root)
                })
            {
                failures.push(
                    "environment_confinement was not the exact full command-scratch map"
                        .to_string(),
                );
            }
        }

        if !failures.is_empty() {
            return Some(format!(
                "run {run_index} {label}: strict sampled disk high-water evidence is unavailable: {}",
                failures.join("; ")
            ));
        }
    }
    None
}

fn strict_envelope_evidence_reason(
    runs: &[CompareRun],
    production_arm: bool,
    count_performance_required: bool,
) -> Option<String> {
    let observations = match resource_observations(runs, production_arm, count_performance_required)
    {
        Ok(observations) => observations,
        Err(reason) => return Some(reason),
    };
    for (run_index, label, observation) in observations {
        let evidence = &observation.resource_evidence;
        if evidence.strict_qualified {
            continue;
        }
        let failures = if evidence.qualification_failures.is_empty() {
            "no qualification reason recorded".to_string()
        } else {
            evidence.qualification_failures.join("; ")
        };
        return Some(format!(
            "run {run_index} {label}: strict execution envelope is unqualified: {failures}"
        ));
    }
    None
}

fn strict_cpu_evidence_reason(
    runs: &[CompareRun],
    production_arm: bool,
    count_performance_required: bool,
) -> Option<String> {
    let observations = match resource_observations(runs, production_arm, count_performance_required)
    {
        Ok(observations) => observations,
        Err(reason) => return Some(reason),
    };
    let reference_cpu_ids = observations
        .first()
        .and_then(|(_, _, observation)| {
            observation.resource_evidence.cpu.effective_cpu_ids.as_ref()
        })
        .cloned();
    for (run_index, label, observation) in observations {
        let cpu = &observation.resource_evidence.cpu;
        if !cpu.confined
            || !cpu.process_tree_inherited
            || !cpu.isolation.isolated
            || cpu.method != CpuConfinementMethod::LinuxSchedSetaffinityInherited
            || cpu.effective_cpu_ids.as_deref().map(<[_]>::len) != Some(1)
        {
            return Some(format!(
                "run {run_index} {label}: strict isolated one-CPU inherited process-tree confinement is unavailable"
            ));
        }
        if cpu.effective_cpu_ids.as_deref() != reference_cpu_ids.as_deref() {
            return Some(format!(
                "run {run_index} {label}: effective logical CPU differs across the paired evidence"
            ));
        }
    }
    None
}

fn strict_memory_evidence_reason(
    runs: &[CompareRun],
    production_arm: bool,
    count_performance_required: bool,
) -> Option<String> {
    let observations = match resource_observations(runs, production_arm, count_performance_required)
    {
        Ok(observations) => observations,
        Err(reason) => return Some(reason),
    };
    for (run_index, label, observation) in observations {
        let evidence = &observation.resource_evidence;
        let memory = &evidence.memory;
        if !evidence.strict_qualified
            || !memory.complete
            || memory.metric != PeakMemoryMetric::CgroupAccountedMemory
            || memory.scope != PeakMemoryScope::ProcessTree
            || memory.method != PeakMemoryMethod::LinuxCgroupV2MemoryPeak
            || memory.peak_bytes.is_none()
            || memory.peak_bytes == Some(0)
        {
            let failures = if evidence.qualification_failures.is_empty() {
                "no qualification reason recorded".to_string()
            } else {
                evidence.qualification_failures.join("; ")
            };
            return Some(format!(
                "run {run_index} {label}: strict process-tree peak-memory evidence is unavailable: {failures}"
            ));
        }
    }
    None
}

fn outcomes_match(count_verify: &RunObservation, production: &RunObservation) -> bool {
    match (
        count_verify.error_type.as_deref(),
        production.error_type.as_deref(),
    ) {
        (None, None) => count_verify.status == production.status,
        (Some(left), Some(right)) => {
            error_types_compatible(Some(left), Some(right))
                && violated_obligations_compatible(
                    Some(left),
                    count_verify.violated_obligation.as_deref(),
                    production.violated_obligation.as_deref(),
                )
        }
        _ => false,
    }
}

fn missing_engine_provenance_reason(runs: &[CompareRun], production_arm: bool) -> Option<String> {
    for run in runs {
        if run.backend_run.engine_tier.is_none() {
            return Some(format!(
                "run {} production/backend TY observation is missing engine-tier provenance",
                run.run_index
            ));
        }
        if production_arm
            && run
                .count_verify_run
                .as_ref()
                .is_none_or(|observation| observation.engine_tier.is_none())
        {
            return Some(format!(
                "run {} count-verification TY observation is missing engine-tier provenance",
                run.run_index
            ));
        }
    }
    None
}

fn unstable_observation_reason(label: &str, observations: &[&RunObservation]) -> Option<String> {
    let first = observations.first()?;
    observations
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, observation)| !same_semantic_observation(first, observation))
        .map(|(index, observation)| {
            format!(
                "{label} outcome is not deterministic across repetitions: run 1 status={} error={:?} obligation={:?} states={:?} raw_generated={:?}/{:?}/{:?} engine_tier={:?}; run {} status={} error={:?} obligation={:?} states={:?} raw_generated={:?}/{:?}/{:?} engine_tier={:?}",
                first.status,
                first.error_type,
                first.violated_obligation,
                first.states_found,
                first.raw_initial_states_generated,
                first.raw_successors_generated,
                first.states_generated,
                first.engine_tier,
                index + 1,
                observation.status,
                observation.error_type,
                observation.violated_obligation,
                observation.states_found,
                observation.raw_initial_states_generated,
                observation.raw_successors_generated,
                observation.states_generated,
                observation.engine_tier
            )
        })
}

fn same_semantic_observation(left: &RunObservation, right: &RunObservation) -> bool {
    left.status == right.status
        && left.error_type == right.error_type
        && left.violated_obligation == right.violated_obligation
        && left.states_found == right.states_found
        && left.raw_initial_states_generated == right.raw_initial_states_generated
        && left.raw_successors_generated == right.raw_successors_generated
        && left.states_generated == right.states_generated
        && left.timed_out == right.timed_out
        && left.engine_tier == right.engine_tier
}

fn aggregate_observations(observations: &[RunObservation]) -> RunObservation {
    let mut aggregate = observations
        .first()
        .expect("repeated compare requires at least one observation")
        .clone();
    aggregate.elapsed_seconds = median_f64(
        observations
            .iter()
            .map(|observation| observation.elapsed_seconds)
            .collect(),
    )
    .unwrap_or(aggregate.elapsed_seconds);
    let peak_memory_values = observations
        .iter()
        .filter_map(process_tree_peak_memory_bytes)
        .collect::<Vec<_>>();
    aggregate.resource_evidence.memory.peak_bytes = (peak_memory_values.len()
        == observations.len())
    .then(|| median_u64(peak_memory_values))
    .flatten();
    let allocated_disk_values = observations
        .iter()
        .filter_map(sampled_peak_allocated_disk_bytes)
        .collect::<Vec<_>>();
    aggregate.disk_high_water.peak_allocated_bytes = (allocated_disk_values.len()
        == observations.len())
    .then(|| median_u64(allocated_disk_values))
    .flatten();
    let apparent_disk_values = observations
        .iter()
        .filter_map(sampled_peak_apparent_disk_bytes)
        .collect::<Vec<_>>();
    aggregate.disk_high_water.peak_apparent_bytes = (apparent_disk_values.len()
        == observations.len())
    .then(|| median_u64(apparent_disk_values))
    .flatten();
    aggregate
}

fn median_f64(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[midpoint])
    } else {
        Some((values[midpoint - 1] + values[midpoint]) / 2.0)
    }
}

fn median_u64(mut values: Vec<u64>) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let midpoint = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[midpoint])
    } else {
        let sum = u128::from(values[midpoint - 1]) + u128::from(values[midpoint]);
        u64::try_from(sum / 2).ok()
    }
}

fn classify_observations_with_limits(
    expected_tlc_states: Option<u64>,
    expected_backend_states: Option<u64>,
    expected_tlc_error: Option<&str>,
    expected_backend_error: Option<&str>,
    policy: SupremacyComparePolicy,
    min_speedup: f64,
    max_memory_ratio: f64,
    tlc: &RunObservation,
    backend: &RunObservation,
    speedup: Option<f64>,
    memory_ratio: Option<f64>,
) -> Classification {
    if expected_tlc_error.is_some_and(|expected| tlc.error_type.as_deref() != Some(expected)) {
        return classified(
            CompareClass::ExpectedErrorMismatch,
            format!(
                "TLC error {:?} did not match expected {:?}",
                tlc.error_type, expected_tlc_error
            ),
            false,
        );
    }
    if expected_backend_error
        .is_some_and(|expected| backend.error_type.as_deref() != Some(expected))
    {
        return classified(
            CompareClass::ExpectedErrorMismatch,
            format!(
                "backend error {:?} did not match expected {:?}",
                backend.error_type, expected_backend_error
            ),
            false,
        );
    }
    if expected_tlc_states.is_some_and(|expected| tlc.states_found != Some(expected)) {
        return classified(
            CompareClass::ExpectedStateMismatch,
            format!(
                "TLC states {:?} did not match expected {:?}",
                tlc.states_found, expected_tlc_states
            ),
            false,
        );
    }
    if expected_backend_states.is_some_and(|expected| backend.states_found != Some(expected)) {
        return classified(
            CompareClass::ExpectedStateMismatch,
            format!(
                "backend states {:?} did not match expected {:?}",
                backend.states_found, expected_backend_states
            ),
            false,
        );
    }

    let tlc_has_error = tlc.error_type.is_some();
    let backend_has_error = backend.error_type.is_some();
    if tlc_has_error || backend_has_error {
        if !tlc_has_error || !backend_has_error {
            return classified(
                CompareClass::ErrorMismatch,
                format!(
                    "error detection mismatch: TLC={:?} backend={:?}",
                    tlc.error_type, backend.error_type
                ),
                false,
            );
        }
        if !error_types_compatible(tlc.error_type.as_deref(), backend.error_type.as_deref()) {
            return classified(
                CompareClass::ErrorMismatch,
                format!(
                    "error type mismatch: TLC={:?} backend={:?}",
                    tlc.error_type, backend.error_type
                ),
                false,
            );
        }
        if !violated_obligations_compatible(
            tlc.error_type.as_deref(),
            tlc.violated_obligation.as_deref(),
            backend.violated_obligation.as_deref(),
        ) {
            return classified(
                CompareClass::ErrorMismatch,
                format!(
                    "violated-obligation mismatch or missing identity: TLC={:?} backend={:?}",
                    tlc.violated_obligation, backend.violated_obligation
                ),
                false,
            );
        }
        let performance =
            evaluate_performance(policy, min_speedup, max_memory_ratio, speedup, memory_ratio);
        if let Some(failure) = performance.failure() {
            return failure;
        }
        return Classification {
            class: CompareClass::Pass,
            reason: "compatible error outcome".to_string(),
            parity_states: true,
            parity_generated_work: true,
            runtime_axis: performance.runtime_axis,
            memory_axis: performance.memory_axis,
        };
    }

    if let Some(error) = &tlc.error {
        return classified(
            CompareClass::TlcFailed,
            format!("TLC failed: {error}"),
            false,
        );
    }
    if let Some(error) = &backend.error {
        return classified(
            CompareClass::BackendFailed,
            format!("backend failed: {error}"),
            false,
        );
    }
    let parity_states = tlc.states_found == backend.states_found;
    if !parity_states {
        return classified(
            CompareClass::ParityFail,
            format!(
                "state-count parity failed: TLC={:?} backend={:?}",
                tlc.states_found, backend.states_found
            ),
            false,
        );
    }
    let tlc_generated = consistent_raw_generated_counts(tlc);
    let backend_generated = consistent_raw_generated_counts(backend);
    if tlc_generated.is_none() || backend_generated.is_none() {
        return Classification {
            class: CompareClass::ParityFail,
            reason: format!(
                "raw generated-state evidence was missing or arithmetically inconsistent: TLC initial={:?} successors={:?} total={:?}; backend initial={:?} successors={:?} total={:?}",
                tlc.raw_initial_states_generated,
                tlc.raw_successors_generated,
                tlc.states_generated,
                backend.raw_initial_states_generated,
                backend.raw_successors_generated,
                backend.states_generated
            ),
            parity_states: true,
            parity_generated_work: false,
            runtime_axis: CompareAxis::NotRequired,
            memory_axis: CompareAxis::NotRequired,
        };
    }
    if tlc_generated != backend_generated {
        return Classification {
            class: CompareClass::ParityFail,
            reason: format!(
                "generated-state parity failed: TLC initial={:?} successors={:?} total={:?}; backend initial={:?} successors={:?} total={:?}",
                tlc.raw_initial_states_generated,
                tlc.raw_successors_generated,
                tlc.states_generated,
                backend.raw_initial_states_generated,
                backend.raw_successors_generated,
                backend.states_generated
            ),
            parity_states: true,
            parity_generated_work: false,
            runtime_axis: CompareAxis::NotRequired,
            memory_axis: CompareAxis::NotRequired,
        };
    }
    let performance =
        evaluate_performance(policy, min_speedup, max_memory_ratio, speedup, memory_ratio);
    if let Some(failure) = performance.failure() {
        return failure;
    }
    Classification {
        class: CompareClass::Pass,
        reason: "passed".to_string(),
        parity_states: true,
        parity_generated_work: true,
        runtime_axis: performance.runtime_axis,
        memory_axis: performance.memory_axis,
    }
}

fn consistent_raw_generated_counts(observation: &RunObservation) -> Option<(u64, u64, u64)> {
    let initial = observation.raw_initial_states_generated?;
    let successors = observation.raw_successors_generated?;
    let total = observation.states_generated?;
    (initial.checked_add(successors) == Some(total)).then_some((initial, successors, total))
}

#[derive(Clone, Debug)]
struct PerformanceEvaluation {
    runtime_axis: CompareAxis,
    memory_axis: CompareAxis,
    failure_class: Option<CompareClass>,
    failure_reason: Option<String>,
}

impl PerformanceEvaluation {
    fn failure(&self) -> Option<Classification> {
        Some(Classification {
            class: self.failure_class?,
            reason: self.failure_reason.clone().unwrap_or_default(),
            parity_states: true,
            parity_generated_work: true,
            runtime_axis: self.runtime_axis,
            memory_axis: self.memory_axis,
        })
    }
}

fn evaluate_performance(
    policy: SupremacyComparePolicy,
    min_speedup: f64,
    max_memory_ratio: f64,
    speedup: Option<f64>,
    memory_ratio: Option<f64>,
) -> PerformanceEvaluation {
    let runtime_axis = if !policy_checks_speed(policy) {
        CompareAxis::NotRequired
    } else {
        match speedup {
            None => CompareAxis::MissingOrStale,
            Some(value) if value <= min_speedup => CompareAxis::Loss,
            Some(_) => CompareAxis::Pass,
        }
    };
    let memory_axis = if !policy_checks_memory(policy) {
        CompareAxis::NotRequired
    } else {
        match memory_ratio {
            None => CompareAxis::MissingOrStale,
            Some(value) if value >= max_memory_ratio => CompareAxis::Loss,
            Some(_) => CompareAxis::Pass,
        }
    };

    let (failure_class, failure_reason) = match (runtime_axis, memory_axis) {
        (CompareAxis::MissingOrStale, CompareAxis::MissingOrStale) => (
            Some(CompareClass::MissingEvidence),
            Some(
                "missing finite positive runtime and positive process-tree peak memory".to_string(),
            ),
        ),
        (CompareAxis::MissingOrStale, CompareAxis::Loss) => (
            Some(CompareClass::MissingEvidence),
            Some(format!(
                "runtime evidence is missing or stale; TY/TLC process-tree peak-memory ratio {:.6}x is not strictly below allowed {:.6}x",
                memory_ratio.unwrap_or(f64::NAN),
                max_memory_ratio
            )),
        ),
        (CompareAxis::Loss, CompareAxis::MissingOrStale) => (
            Some(CompareClass::MissingEvidence),
            Some(format!(
                "speedup {:.6}x is not strictly above required {:.6}x; process-tree peak memory is missing or stale",
                speedup.unwrap_or(f64::NAN),
                min_speedup
            )),
        ),
        (CompareAxis::MissingOrStale, _) => (
            Some(CompareClass::MissingRuntime),
            Some("missing finite positive runtime for speed policy".to_string()),
        ),
        (_, CompareAxis::MissingOrStale) => (
            Some(CompareClass::MissingMemory),
            Some("missing positive process-tree peak-memory evidence for memory policy".to_string()),
        ),
        (CompareAxis::Loss, CompareAxis::Loss) => (
            Some(CompareClass::BothFail),
            Some(format!(
                "speedup {:.6}x is not strictly above required {:.6}x and TY/TLC process-tree peak-memory ratio {:.6}x is not strictly below allowed {:.6}x",
                speedup.unwrap_or(f64::NAN),
                min_speedup,
                memory_ratio.unwrap_or(f64::NAN),
                max_memory_ratio
            )),
        ),
        (CompareAxis::Loss, _) => (
            Some(CompareClass::SpeedFail),
            Some(format!(
                "speedup {:.6}x is not strictly above required {:.6}x",
                speedup.unwrap_or(f64::NAN),
                min_speedup
            )),
        ),
        (_, CompareAxis::Loss) => (
            Some(CompareClass::MemoryFail),
            Some(format!(
                "TY/TLC process-tree peak-memory ratio {:.6}x is not strictly below allowed {:.6}x",
                memory_ratio.unwrap_or(f64::NAN),
                max_memory_ratio
            )),
        ),
        _ => (None, None),
    };

    PerformanceEvaluation {
        runtime_axis,
        memory_axis,
        failure_class,
        failure_reason,
    }
}

#[cfg(test)]
fn classify_observations(
    expected_tlc_states: Option<u64>,
    expected_backend_states: Option<u64>,
    expected_tlc_error: Option<&str>,
    expected_backend_error: Option<&str>,
    policy: SupremacyComparePolicy,
    min_speedup: f64,
    tlc: &RunObservation,
    backend: &RunObservation,
    speedup: Option<f64>,
) -> Classification {
    classify_observations_with_limits(
        expected_tlc_states,
        expected_backend_states,
        expected_tlc_error,
        expected_backend_error,
        policy,
        min_speedup,
        1.0,
        tlc,
        backend,
        speedup,
        memory_ratio(
            process_tree_peak_memory_bytes(backend),
            process_tree_peak_memory_bytes(tlc),
        ),
    )
}

fn classified(class: CompareClass, reason: String, parity_states: bool) -> Classification {
    Classification {
        class,
        reason,
        parity_states,
        parity_generated_work: true,
        runtime_axis: CompareAxis::NotRequired,
        memory_axis: CompareAxis::NotRequired,
    }
}

fn claim_class(
    classification: &Classification,
    policy: SupremacyComparePolicy,
    min_speedup: f64,
    max_memory_ratio: f64,
) -> ClaimClass {
    match classification.class {
        CompareClass::ExpectedStateMismatch
        | CompareClass::ExpectedErrorMismatch
        | CompareClass::ErrorMismatch
        | CompareClass::ParityFail
        | CompareClass::BackendFailed => ClaimClass::ParityBlocker,
        CompareClass::TlcFailed
        | CompareClass::MissingRuntime
        | CompareClass::MissingMemory
        | CompareClass::MissingEvidence => ClaimClass::MissingOrStale,
        CompareClass::SpeedFail => ClaimClass::RuntimeLoss,
        CompareClass::MemoryFail => ClaimClass::MemoryLoss,
        CompareClass::BothFail => ClaimClass::BothLoss,
        CompareClass::Pass
            if policy == SupremacyComparePolicy::ParityAndSpeedAndMemory
                && min_speedup >= STRICT_MIN_SPEEDUP
                && max_memory_ratio <= STRICT_MAX_MEMORY_RATIO
                && classification.runtime_axis == CompareAxis::Pass
                && classification.memory_axis == CompareAxis::Pass =>
        {
            ClaimClass::PassBoth
        }
        CompareClass::Pass => ClaimClass::MissingOrStale,
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct ClaimCounts {
    pass_both: usize,
    runtime_loss: usize,
    memory_loss: usize,
    both_loss: usize,
    parity_blocker: usize,
    missing_or_stale: usize,
}

impl ClaimCounts {
    fn from_rows(rows: &[CompareRow]) -> Self {
        let mut counts = Self::default();
        for row in rows {
            match row.claim_class {
                ClaimClass::PassBoth => counts.pass_both += 1,
                ClaimClass::RuntimeLoss => counts.runtime_loss += 1,
                ClaimClass::MemoryLoss => counts.memory_loss += 1,
                ClaimClass::BothLoss => counts.both_loss += 1,
                ClaimClass::ParityBlocker => counts.parity_blocker += 1,
                ClaimClass::MissingOrStale => counts.missing_or_stale += 1,
            }
        }
        counts
    }
}

#[derive(Clone, Debug, Serialize)]
struct DiskHighWaterDisclosure {
    contract_schema: &'static str,
    measurement_role: &'static str,
    scope: &'static str,
    method: &'static str,
    peak_exact: bool,
    sampling_execution: &'static str,
    sampling_can_perturb_elapsed: bool,
    sampling_interval_ms: u64,
    scan_budget_ms: u64,
    scan_entry_limit: u64,
}

impl DiskHighWaterDisclosure {
    fn sampled_informational() -> Self {
        Self {
            contract_schema: DISK_SCOPE_CONTRACT_SCHEMA,
            measurement_role: "informational_only_no_superiority_threshold",
            scope: "command_artifact_and_scratch_tree",
            method: "recursive_filesystem_metadata_polling",
            peak_exact: false,
            sampling_execution: "inline_runner_poll_loop",
            sampling_can_perturb_elapsed: true,
            sampling_interval_ms: u64::try_from(DISK_USAGE_SAMPLE_INTERVAL.as_millis())
                .expect("interval fits u64"),
            scan_budget_ms: u64::try_from(DISK_USAGE_SCAN_BUDGET.as_millis())
                .expect("budget fits u64"),
            scan_entry_limit: DISK_USAGE_SCAN_ENTRY_LIMIT,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct CompareReport {
    schema: &'static str,
    timestamp: String,
    backend: String,
    policy: String,
    mode: String,
    min_speedup: f64,
    max_memory_ratio: f64,
    runs_per_row: usize,
    tool_order: &'static str,
    count_verify_schedule: &'static str,
    paired_statistic: &'static str,
    disk_high_water_disclosure: DiskHighWaterDisclosure,
    passed: bool,
    strict_superiority_passed: bool,
    total_rows: usize,
    failed_rows: usize,
    claim_counts: ClaimCounts,
    output_dir: PathBuf,
    workers: Vec<usize>,
    cases: Vec<EnvCase>,
    rows: Vec<CompareRow>,
}

impl CompareReport {
    fn new(
        args: &SupremacyCompareArgs,
        output_dir: PathBuf,
        cases: Vec<EnvCase>,
        rows: Vec<CompareRow>,
    ) -> Self {
        let failed_rows = rows.iter().filter(|row| !row.passed).count();
        let strict_superiority_passed = args.mode == SupremacyMode::Enforce
            && args.backend == SupremacyCompareBackend::AutoCpu
            && args.policy == SupremacyComparePolicy::ParityAndSpeedAndMemory
            && args.runs >= STRICT_MIN_BALANCED_PAIRED_RUNS
            && args.runs % 2 == 0
            && args.workers.iter().all(|workers| *workers == 1)
            && args.min_speedup >= STRICT_MIN_SPEEDUP
            && args.max_memory_ratio <= STRICT_MAX_MEMORY_RATIO
            && !rows.is_empty()
            && rows
                .iter()
                .all(|row| row.claim_class == ClaimClass::PassBoth);
        let claim_counts = ClaimCounts::from_rows(&rows);
        let count_performance_protocol =
            is_production_auto(args.backend) && policy_checks_speed(args.policy);
        Self {
            schema: COMPARE_REPORT_SCHEMA,
            timestamp: chrono::Utc::now().to_rfc3339(),
            backend: backend_cli_name(args.backend).to_string(),
            policy: policy_name(args.policy).to_string(),
            mode: mode_name(args.mode).to_string(),
            min_speedup: args.min_speedup,
            max_memory_ratio: args.max_memory_ratio,
            runs_per_row: args.runs,
            tool_order: if count_performance_protocol {
                "alternating_within_production_and_count_pairs"
            } else {
                "alternating_tlc_ty"
            },
            count_verify_schedule: if count_performance_protocol {
                "independent_tlc_count_pair_with_alternating_pair_blocks"
            } else if is_production_auto(args.backend) {
                "after_each_production_pair_parity_only"
            } else {
                "not_applicable"
            },
            paired_statistic: PAIRED_STATISTIC,
            disk_high_water_disclosure: DiskHighWaterDisclosure::sampled_informational(),
            passed: failed_rows == 0,
            strict_superiority_passed,
            total_rows: rows.len(),
            failed_rows,
            claim_counts,
            output_dir,
            workers: args.workers.clone(),
            cases,
            rows,
        }
    }

    fn to_human(&self) -> String {
        let mut out = String::new();
        let status = if self.passed { "PASS" } else { "FAIL" };
        let _ = writeln!(
            out,
            "Supremacy compare {status}: {} rows, {} failed",
            self.total_rows, self.failed_rows
        );
        let _ = writeln!(
            out,
            "strict_superiority_passed={} pass_both={} runtime_loss={} memory_loss={} both_loss={} parity_blocker={} missing_or_stale={}",
            self.strict_superiority_passed,
            self.claim_counts.pass_both,
            self.claim_counts.runtime_loss,
            self.claim_counts.memory_loss,
            self.claim_counts.both_loss,
            self.claim_counts.parity_blocker,
            self.claim_counts.missing_or_stale,
        );
        let _ = writeln!(
            out,
            "backend={} policy={} min_speedup={} max_memory_ratio={} runs={} order={} count_verify_schedule={} statistic={} cases={} output_dir={}",
            self.backend,
            self.policy,
            self.min_speedup,
            self.max_memory_ratio,
            self.runs_per_row,
            self.tool_order,
            self.count_verify_schedule,
            self.paired_statistic,
            self.cases
                .iter()
                .map(|case| case.name.as_str())
                .collect::<Vec<_>>()
                .join(","),
            self.output_dir.display()
        );
        let disk = &self.disk_high_water_disclosure;
        let _ = writeln!(
            out,
            "disk_high_water=informational_only peak_exact={} interval_ms={} scan_budget_ms={} scan_entry_limit={} sampling_execution={} sampling_can_perturb_elapsed={}",
            disk.peak_exact,
            disk.sampling_interval_ms,
            disk.scan_budget_ms,
            disk.scan_entry_limit,
            disk.sampling_execution,
            disk.sampling_can_perturb_elapsed,
        );
        for row in &self.rows {
            let row_status = if row.passed { "PASS" } else { "FAIL" };
            let speedup = row
                .speedup_tlc_vs_backend
                .map(|value| format!("{value:.3}x"))
                .unwrap_or_else(|| "n/a".to_string());
            let count_speedup = row
                .speedup_count_tlc_vs_count_verify
                .map(|value| format!("{value:.3}x"))
                .unwrap_or_else(|| "n/a".to_string());
            let memory_ratio = row
                .memory_ratio_backend_vs_tlc
                .map(|value| format!("{value:.3}x"))
                .unwrap_or_else(|| "n/a".to_string());
            let count_memory_ratio = row
                .memory_ratio_count_verify_vs_count_tlc
                .map(|value| format!("{value:.3}x"))
                .unwrap_or_else(|| "n/a".to_string());
            let _ = writeln!(
                out,
                "- {row_status} {} case={} workers={} class={:?} claim_class={:?} runtime_axis={:?} memory_axis={:?} production_tlc_states={:?} count_tlc_states={:?} count_verify_states={:?} backend_states={:?} production_tlc_process_tree_peak_memory={} backend_process_tree_peak_memory={} count_tlc_process_tree_peak_memory={} count_verify_process_tree_peak_memory={} production_speedup={} count_speedup={} production_memory_ratio={} count_memory_ratio={} reason={}",
                row.spec,
                row.case,
                row.workers,
                row.class,
                row.claim_class,
                row.runtime_axis,
                row.memory_axis,
                row.tlc.states_found,
                row.count_tlc_run
                    .as_ref()
                    .and_then(|run| run.states_found),
                row.count_verify_run
                    .as_ref()
                    .and_then(|run| run.states_found),
                row.backend_run.states_found,
                fmt_bytes(process_tree_peak_memory_bytes(&row.tlc)),
                fmt_bytes(process_tree_peak_memory_bytes(&row.backend_run)),
                fmt_bytes(
                    row.count_tlc_run
                        .as_ref()
                        .and_then(process_tree_peak_memory_bytes)
                ),
                fmt_bytes(
                    row.count_verify_run
                        .as_ref()
                        .and_then(process_tree_peak_memory_bytes)
                ),
                speedup,
                count_speedup,
                memory_ratio,
                count_memory_ratio,
                row.reason
            );
            let _ = writeln!(
                out,
                "  sampled_disk_high_water production_tlc_allocated={} production_tlc_apparent={} backend_allocated={} backend_apparent={} count_tlc_allocated={} count_tlc_apparent={} count_verify_allocated={} count_verify_apparent={}",
                fmt_bytes(sampled_peak_allocated_disk_bytes(&row.tlc)),
                fmt_bytes(sampled_peak_apparent_disk_bytes(&row.tlc)),
                fmt_bytes(sampled_peak_allocated_disk_bytes(&row.backend_run)),
                fmt_bytes(sampled_peak_apparent_disk_bytes(&row.backend_run)),
                fmt_bytes(
                    row.count_tlc_run
                        .as_ref()
                        .and_then(sampled_peak_allocated_disk_bytes)
                ),
                fmt_bytes(
                    row.count_tlc_run
                        .as_ref()
                        .and_then(sampled_peak_apparent_disk_bytes)
                ),
                fmt_bytes(
                    row.count_verify_run
                        .as_ref()
                        .and_then(sampled_peak_allocated_disk_bytes)
                ),
                fmt_bytes(
                    row.count_verify_run
                        .as_ref()
                        .and_then(sampled_peak_apparent_disk_bytes)
                ),
            );
        }
        out
    }

    fn to_markdown(&self) -> String {
        let mut lines = vec![
            "# Supremacy Compare".to_string(),
            String::new(),
            format!("Verdict: **{}**", if self.passed { "PASS" } else { "FAIL" }),
            format!(
                "Strict both-axis claim: **{}**",
                if self.strict_superiority_passed {
                    "PASS"
                } else {
                    "NOT ESTABLISHED"
                }
            ),
            format!("Backend: `{}`", self.backend),
            format!("Policy: `{}`", self.policy),
            format!("Minimum TLC/TY speedup: `{}`", self.min_speedup),
            format!(
                "Maximum TY/TLC process-tree peak-memory ratio: `{}`",
                self.max_memory_ratio
            ),
            format!("Paired cold runs per row: `{}`", self.runs_per_row),
            format!("Tool order: `{}`", self.tool_order),
            format!(
                "Count verification schedule: `{}`",
                self.count_verify_schedule
            ),
            format!("Policy statistic: `{}`", self.paired_statistic),
            format!(
                "Disk high-water: informational only (no superiority threshold); sampled every `{} ms` inline with a `{} ms` / `{}`-entry scan bound; peaks are not exact and sampling can perturb elapsed time.",
                self.disk_high_water_disclosure.sampling_interval_ms,
                self.disk_high_water_disclosure.scan_budget_ms,
                self.disk_high_water_disclosure.scan_entry_limit,
            ),
            format!(
                "Cases: `{}`",
                self.cases
                    .iter()
                    .map(|case| case.name.as_str())
                    .collect::<Vec<_>>()
                    .join("`, `")
            ),
            format!("Output dir: `{}`", self.output_dir.display()),
            String::new(),
            "| Spec | Case | Workers | Class | Claim class | Runtime axis | Memory axis | Production TLC states | Count TLC states | Count TY states | Backend states | Production TLC seconds | Backend seconds | Count TLC seconds | Count TY seconds | Production TLC memory | Backend memory | Count TLC memory | Count TY memory | Production speedup | Count speedup | Production TY/TLC memory | Count TY/TLC memory | Reason |".to_string(),
            "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |".to_string(),
        ];
        for row in &self.rows {
            lines.push(format!(
                "| {} | {} | {} | {:?} | {:?} | {:?} | {:?} | {} | {} | {} | {} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                row.spec,
                row.case,
                row.workers,
                row.class,
                row.claim_class,
                row.runtime_axis,
                row.memory_axis,
                fmt_opt_u64(row.tlc.states_found),
                fmt_opt_u64(
                    row.count_tlc_run
                        .as_ref()
                        .and_then(|run| run.states_found)
                ),
                fmt_opt_u64(
                    row.count_verify_run
                        .as_ref()
                        .and_then(|run| run.states_found)
                ),
                fmt_opt_u64(row.backend_run.states_found),
                row.tlc.elapsed_seconds,
                row.backend_run.elapsed_seconds,
                row.count_tlc_run
                    .as_ref()
                    .map(|run| format!("{:.3}", run.elapsed_seconds))
                    .unwrap_or_else(|| "n/a".to_string()),
                row.count_verify_run
                    .as_ref()
                    .map(|run| format!("{:.3}", run.elapsed_seconds))
                    .unwrap_or_else(|| "n/a".to_string()),
                fmt_bytes(process_tree_peak_memory_bytes(&row.tlc)),
                fmt_bytes(process_tree_peak_memory_bytes(&row.backend_run)),
                fmt_bytes(
                    row.count_tlc_run
                        .as_ref()
                        .and_then(process_tree_peak_memory_bytes)
                ),
                fmt_bytes(
                    row.count_verify_run
                        .as_ref()
                        .and_then(process_tree_peak_memory_bytes)
                ),
                row.speedup_tlc_vs_backend
                    .map(|value| format!("{value:.3}x"))
                    .unwrap_or_else(|| "n/a".to_string()),
                row.speedup_count_tlc_vs_count_verify
                    .map(|value| format!("{value:.3}x"))
                    .unwrap_or_else(|| "n/a".to_string()),
                row.memory_ratio_backend_vs_tlc
                    .map(|value| format!("{value:.3}x"))
                    .unwrap_or_else(|| "n/a".to_string()),
                row.memory_ratio_count_verify_vs_count_tlc
                    .map(|value| format!("{value:.3}x"))
                    .unwrap_or_else(|| "n/a".to_string()),
                row.reason.replace('|', "\\|")
            ));
        }
        lines.push(String::new());
        lines.extend([
            "## Sampled disk high-water (informational only)".to_string(),
            String::new(),
            "| Spec | Case | Production TLC allocated | Production TLC apparent | Backend allocated | Backend apparent | Count TLC allocated | Count TLC apparent | Count TY allocated | Count TY apparent |".to_string(),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |".to_string(),
        ]);
        for row in &self.rows {
            lines.push(format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                row.spec,
                row.case,
                fmt_bytes(sampled_peak_allocated_disk_bytes(&row.tlc)),
                fmt_bytes(sampled_peak_apparent_disk_bytes(&row.tlc)),
                fmt_bytes(sampled_peak_allocated_disk_bytes(&row.backend_run)),
                fmt_bytes(sampled_peak_apparent_disk_bytes(&row.backend_run)),
                fmt_bytes(
                    row.count_tlc_run
                        .as_ref()
                        .and_then(sampled_peak_allocated_disk_bytes)
                ),
                fmt_bytes(
                    row.count_tlc_run
                        .as_ref()
                        .and_then(sampled_peak_apparent_disk_bytes)
                ),
                fmt_bytes(
                    row.count_verify_run
                        .as_ref()
                        .and_then(sampled_peak_allocated_disk_bytes)
                ),
                fmt_bytes(
                    row.count_verify_run
                        .as_ref()
                        .and_then(sampled_peak_apparent_disk_bytes)
                ),
            ));
        }
        lines.push(String::new());
        lines.join("\n")
    }
}

fn print_report(report: &CompareReport, format: SupremacyOutputFormat) -> Result<()> {
    match format {
        SupremacyOutputFormat::Json => println!("{}", serde_json::to_string_pretty(report)?),
        SupremacyOutputFormat::Markdown => println!("{}", report.to_markdown()),
        SupremacyOutputFormat::Human => print!("{}", report.to_human()),
    }
    Ok(())
}

fn speedup(tlc_seconds: f64, backend_seconds: f64) -> Option<f64> {
    if tlc_seconds.is_finite()
        && backend_seconds.is_finite()
        && tlc_seconds > 0.0
        && backend_seconds > 0.0
    {
        let ratio = tlc_seconds / backend_seconds;
        ratio.is_finite().then_some(ratio)
    } else {
        None
    }
}

fn required_runtime_speedup(
    production_speedup: Option<f64>,
    count_speedup: Option<f64>,
    count_performance_required: bool,
) -> Option<f64> {
    if count_performance_required {
        production_speedup
            .zip(count_speedup)
            .map(|(production, count)| {
                // The policy boundary is strict, so requiring the lower of the two
                // arm medians is exactly equivalent to requiring each separately.
                production.min(count)
            })
    } else {
        production_speedup
    }
}

fn format_optional_ratio(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.6}x"))
        .unwrap_or_else(|| "missing".to_string())
}

fn memory_ratio(backend_bytes: Option<u64>, tlc_bytes: Option<u64>) -> Option<f64> {
    let backend_bytes = backend_bytes?;
    let tlc_bytes = tlc_bytes?;
    if backend_bytes == 0 || tlc_bytes == 0 {
        return None;
    }
    Some(backend_bytes as f64 / tlc_bytes as f64)
}

fn backend_env(backend: SupremacyCompareBackend) -> BTreeMap<String, String> {
    match backend {
        SupremacyCompareBackend::Interpreter => BTreeMap::from([
            ("TY_trust_cg".to_string(), "0".to_string()),
            ("TY_TRUST_CG_BFS".to_string(), "0".to_string()),
        ]),
        SupremacyCompareBackend::TrustCg => policy::full_native_fused_protected_env(),
        // Production AUTO arms run with NO env pins — the point is to measure
        // exactly what a user's `ty check` selects (burndown P4: "beat TLC
        // without environment variables"). GPU exclusion for auto-cpu is a
        // CLI flag, not env.
        SupremacyCompareBackend::Auto | SupremacyCompareBackend::AutoCpu => BTreeMap::new(),
    }
}

#[derive(Clone, Debug)]
enum TlcRunner {
    Executable {
        tlc_bin: PathBuf,
        tla_library: Option<PathBuf>,
    },
    Java {
        tlc_jar: PathBuf,
        community_modules: Option<PathBuf>,
        tla_library: Option<PathBuf>,
    },
}

fn resolve_tlc_runner(args: &SupremacyCompareArgs, repo_root: &Path) -> Result<TlcRunner> {
    let tla_library = resolve_tla_library(args, repo_root);
    if let Some(tlc_bin) = args
        .tlc_bin
        .clone()
        .or_else(|| non_empty_env_path(ENV_TLC_BIN))
    {
        let tlc_bin = absolutize(repo_root, &tlc_bin);
        validate_file(&tlc_bin)
            .with_context(|| format!("validate TLC executable {}", tlc_bin.display()))?;
        return Ok(TlcRunner::Executable {
            tlc_bin,
            tla_library,
        });
    }

    let tlc_jar = args
        .tlc_jar
        .clone()
        .map(Ok)
        .unwrap_or_else(default_tlc_jar)?;
    let tlc_jar = absolutize(repo_root, &tlc_jar);
    validate_file(&tlc_jar).with_context(|| format!("validate TLC jar {}", tlc_jar.display()))?;
    let community_modules = args
        .community_modules
        .clone()
        .or_else(|| non_empty_env_path(ENV_COMMUNITY_MODULES))
        .or_else(default_community_modules_jar)
        .map(|path| absolutize(repo_root, &path));
    if let Some(community_modules) = &community_modules {
        validate_file(community_modules).with_context(|| {
            format!(
                "validate CommunityModules jar {}",
                community_modules.display()
            )
        })?;
    }
    Ok(TlcRunner::Java {
        tlc_jar,
        community_modules,
        tla_library,
    })
}

/// Resolve the TLA library injected into BOTH tools' module paths.
///
/// The installed **upstream** proof library (`ty install-tlc proof-library`)
/// is preferred over the repo's first-party `test_specs/tla_library` stub set.
/// 25 of the 141 eligible corpus rows EXTEND TLAPS / FiniteSetTheorems /
/// NaturalsInduction and cannot be parsed by TLC without one of them; letting
/// the first-party stub win would make ~18% of the claim corpus depend on a
/// TY-authored artifact that the recorded toolchain never names. The stub
/// remains the fallback so a checkout with no install still runs.
fn resolve_tla_library(args: &SupremacyCompareArgs, repo_root: &Path) -> Option<PathBuf> {
    args.tla_library
        .clone()
        .or_else(|| non_empty_env_path(ENV_TLA_LIBRARY))
        .or_else(|| non_empty_env_path(ENV_TLA_PLUS_LIBRARY))
        .or_else(|| {
            let installed = crate::cmd_tlc::default_proof_library();
            installed.is_dir().then_some(installed)
        })
        .or_else(|| {
            let repo_library = repo_root.join(DEFAULT_TLA_LIBRARY);
            repo_library.is_dir().then_some(repo_library)
        })
        .map(|path| absolutize(repo_root, &path))
}

fn default_community_modules_jar() -> Option<PathBuf> {
    env::var_os("HOME")
        .map(|home| PathBuf::from(home).join(DEFAULT_COMMUNITY_MODULES_JAR))
        .filter(|path| path.is_file())
}

fn tlc_classpath(tlc_jar: &Path, community_modules: Option<&Path>) -> Result<String> {
    let mut paths = vec![tlc_jar.to_path_buf()];
    if let Some(community_modules) = community_modules {
        paths.push(community_modules.to_path_buf());
    }
    let classpath = env::join_paths(paths).context("build TLC classpath")?;
    Ok(classpath.to_string_lossy().to_string())
}

fn backend_cli_name(backend: SupremacyCompareBackend) -> &'static str {
    match backend {
        SupremacyCompareBackend::Interpreter => "interpreter",
        SupremacyCompareBackend::TrustCg => "trust-cg",
        SupremacyCompareBackend::Auto => "auto",
        SupremacyCompareBackend::AutoCpu => "auto-cpu",
    }
}

fn is_production_auto(backend: SupremacyCompareBackend) -> bool {
    matches!(
        backend,
        SupremacyCompareBackend::Auto | SupremacyCompareBackend::AutoCpu
    )
}

fn policy_name(policy: SupremacyComparePolicy) -> &'static str {
    match policy {
        SupremacyComparePolicy::Parity => "parity",
        SupremacyComparePolicy::ParityAndSpeed => "parity-and-speed",
        SupremacyComparePolicy::ParityAndSpeedAndMemory => "parity-and-speed-and-memory",
    }
}

fn mode_name(mode: SupremacyMode) -> &'static str {
    match mode {
        SupremacyMode::Warn => "warn",
        SupremacyMode::Enforce => "enforce",
    }
}

fn resolve_source_path(repo_root: &Path, examples_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        let examples_path = examples_dir.join(path);
        if examples_path.exists() {
            examples_path
        } else {
            repo_root.join(path)
        }
    }
}

fn default_tlc_jar() -> Result<PathBuf> {
    if let Some(path) = non_empty_env_path(ENV_TYTOOLS_JAR) {
        return Ok(path);
    }
    if let Some(path) = non_empty_env_path(ENV_TLC_JAR) {
        return Ok(path);
    }
    let home = env::var_os("HOME").context("HOME is not set; pass --tlc-jar")?;
    Ok(PathBuf::from(home).join(DEFAULT_TLC_JAR))
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    let value = env::var_os(name)?;
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value))
    }
}

fn validate_file(path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        bail!("required file not found: {}", path.display())
    }
}

fn validate_spec_files(tla_path: &Path, cfg_path: &Path) -> Result<()> {
    validate_file(tla_path).with_context(|| format!("validate TLA file {}", tla_path.display()))?;
    validate_file(cfg_path).with_context(|| format!("validate config file {}", cfg_path.display()))
}

fn absolutize(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn default_output_dir(command: &str) -> PathBuf {
    Path::new("reports").join("perf").join(format!(
        "{}-supremacy-{}",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        command
    ))
}

fn safe_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "spec".to_string()
    } else {
        out
    }
}

fn fmt_opt_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_bytes(value: Option<u64>) -> String {
    value
        .map(|bytes| {
            if bytes >= 1024 * 1024 * 1024 {
                format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            } else if bytes >= 1024 * 1024 {
                format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
            } else if bytes >= 1024 {
                format!("{:.1} KiB", bytes as f64 / 1024.0)
            } else {
                format!("{bytes} B")
            }
        })
        .unwrap_or_else(|| "n/a".to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::runner::{
        CgroupCpuStatEvidence, CgroupCpuStatSnapshot, CgroupEffectiveCpusetEvidence,
        CgroupLimitValue, CgroupMemorySwapMaxEvidence, CgroupParentSource, CgroupResourceEvidence,
        CpuIsolationEvidence, CpuIsolationMethod, CpuResourceEvidence, PeakMemoryEvidence,
    };
    use super::super::work_equivalence::{
        EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID, WORK_EQUIVALENCE_SCHEMA_VERSION,
    };
    use super::*;

    fn test_args() -> SupremacyCompareArgs {
        SupremacyCompareArgs {
            spec_source: SupremacyCompareSpecSource::Baseline,
            baseline: PathBuf::from("tests/tlc_comparison/spec_baseline.json"),
            specs: vec![],
            tla: None,
            config: None,
            backend: SupremacyCompareBackend::Interpreter,
            workers: vec![1],
            runs: 6,
            mode: SupremacyMode::Enforce,
            policy: SupremacyComparePolicy::Parity,
            min_speedup: 1.05,
            max_memory_ratio: 0.95,
            output_dir: None,
            ty_bin: None,
            tlc_jar: None,
            tlc_bin: None,
            community_modules: None,
            tla_library: None,
            timeout: 300,
            ty_flag: vec![],
            cases: vec![],
            ty_env: vec![],
            case_env: vec![],
            format: SupremacyOutputFormat::Human,
        }
    }

    fn obs(states_found: Option<u64>, elapsed_seconds: f64, error: Option<&str>) -> RunObservation {
        obs_with_transitions(states_found, states_found, elapsed_seconds, error)
    }

    fn obs_with_transitions(
        states_found: Option<u64>,
        transitions: Option<u64>,
        elapsed_seconds: f64,
        error: Option<&str>,
    ) -> RunObservation {
        let error_type = error.map(str::to_string);
        let raw_initial_states_generated = states_found.map(|states| u64::from(states > 0));
        let raw_successors_generated = transitions
            .and_then(|generated| generated.checked_sub(raw_initial_states_generated.unwrap_or(0)));
        let artifact_dir = "/tmp/ty-supremacy-compare-test-artifact";
        RunObservation {
            tool: "test".to_string(),
            mode: "test".to_string(),
            status: status_for_error(error).to_string(),
            elapsed_seconds,
            resource_evidence: qualifying_resource_evidence(Some(1024)),
            disk_high_water: qualifying_disk_evidence(artifact_dir),
            states_found,
            transitions,
            raw_initial_states_generated,
            raw_successors_generated,
            states_generated: transitions,
            returncode: i32::from(error.is_some()),
            timed_out: false,
            error_type,
            violated_obligation: error
                .filter(|kind| {
                    matches!(
                        *kind,
                        "invariant"
                            | "safety"
                            | "liveness"
                            | "action"
                            | "property"
                            | "assume_violation"
                    )
                })
                .map(|_| "TestObligation".to_string()),
            error: error.map(str::to_string),
            artifact_dir: artifact_dir.to_string(),
            engine_tier: Some("test-tier".to_string()),
        }
    }

    fn qualifying_disk_evidence(scope_root: &str) -> DiskHighWaterEvidence {
        let scratch_root = Path::new(scope_root)
            .join(COMMAND_SCRATCH_DIR_NAME)
            .display()
            .to_string();
        DiskHighWaterEvidence {
            contract_schema: DISK_SCOPE_CONTRACT_SCHEMA,
            scope_root: Some(scope_root.to_string()),
            scratch_root: Some(scratch_root.clone()),
            scope: DiskHighWaterScope::CommandArtifactAndScratchTree,
            method: DiskHighWaterMethod::RecursiveFilesystemMetadataPolling,
            peak_exact: false,
            sampling_execution: DiskSamplingExecution::InlineRunnerPollLoop,
            sampling_can_perturb_elapsed: true,
            peak_allocated_bytes: Some(4096),
            allocated_high_water_semantics: "sampled_observed_peak",
            kernel_enforced_allocated_upper_bound_bytes: None,
            kernel_enforced_inode_upper_bound: None,
            live_recursive_payload_sampling: true,
            peak_apparent_bytes: Some(2048),
            final_allocated_bytes: Some(4096),
            final_apparent_bytes: Some(2048),
            final_entries_observed: Some(2),
            filesystem_capacity_probe_root: Some(scope_root.to_string()),
            filesystem_capacity_probe_device: None,
            minimum_filesystem_available_bytes_observed: Some(1),
            minimum_filesystem_available_inodes_observed: Some(1),
            minimum_project_quota_available_bytes_observed: None,
            minimum_project_quota_available_inodes_observed: None,
            project_quota_byte_reserve: None,
            project_quota_inode_reserve: None,
            storage_contract: None,
            storage_limit_trigger: None,
            sampling_interval_ms: u64::try_from(DISK_USAGE_SAMPLE_INTERVAL.as_millis()).unwrap(),
            scan_budget_ms: u64::try_from(DISK_USAGE_SCAN_BUDGET.as_millis()).unwrap(),
            scan_entry_limit: DISK_USAGE_SCAN_ENTRY_LIMIT,
            total_scan_nanoseconds: 2000,
            max_scan_nanoseconds: 1000,
            samples_attempted: 2,
            samples_complete: 2,
            samples_partial: 0,
            peak_entries_observed: 2,
            initial_sample_complete: true,
            final_sample_complete: true,
            setup_complete: true,
            environment_confinement: COMMAND_SCOPED_ENV_KEYS
                .iter()
                .map(|key| ((*key).to_string(), scratch_root.clone()))
                .collect(),
            environment_confinement_complete: true,
            scope_identity_stable: true,
            ownership_verified: true,
            accounting_complete: true,
            polling_complete: true,
            process_tree_lifetime_complete: true,
            process_tree_naturally_unpopulated: true,
            process_tree_forced_quiescence_complete: false,
            complete: true,
            strict_qualified: true,
            diagnostics: Vec::new(),
            qualification_failures: Vec::new(),
        }
    }

    fn set_test_artifact_dir(observation: &mut RunObservation, artifact_dir: String) {
        observation.artifact_dir = artifact_dir.clone();
        observation.disk_high_water = qualifying_disk_evidence(&artifact_dir);
    }

    fn qualifying_resource_evidence(peak_bytes: Option<u64>) -> ResourceEvidence {
        ResourceEvidence {
            platform: "linux",
            cpu: CpuResourceEvidence {
                requested_logical_cpus: Some(1),
                effective_cpu_ids: Some(vec![0]),
                method: CpuConfinementMethod::LinuxSchedSetaffinityInherited,
                process_tree_inherited: true,
                confined: true,
                isolation: CpuIsolationEvidence {
                    isolated: true,
                    method: CpuIsolationMethod::KernelIsolatedCpu,
                    kernel_isolated_cpu_ids: Some(vec![0]),
                    cgroup_partition_root: None,
                    diagnostic: None,
                },
                diagnostic: None,
            },
            cgroup: CgroupResourceEvidence {
                requested_parent: Some("/sys/fs/cgroup/benchmark.scope".to_string()),
                resolved_parent: Some("/sys/fs/cgroup/benchmark.scope".to_string()),
                parent_source: CgroupParentSource::ExplicitEnvironment,
                mount_point: Some("/sys/fs/cgroup".to_string()),
                current_membership: Some("/user.slice/benchmark.scope/supervisor".to_string()),
                migration_common_ancestor: Some("/sys/fs/cgroup/benchmark.scope".to_string()),
                leaf_path: Some("/sys/fs/cgroup/benchmark.scope/ty-supremacy-test".to_string()),
                parent_direct_processes_empty: true,
                memory_controller_delegated: true,
                parent_writable: true,
                parent_verified: true,
                process_tree_naturally_unpopulated: true,
                effective_cpuset: CgroupEffectiveCpusetEvidence {
                    before_command_cpu_ids: Some(vec![0]),
                    after_command_cpu_ids: Some(vec![0]),
                    selected_cpu_id: Some(0),
                    selected_cpu_present_before: true,
                    selected_cpu_present_after: true,
                    unchanged: true,
                    verified: true,
                    diagnostic: None,
                },
                memory_swap_max: CgroupMemorySwapMaxEvidence {
                    before_command: Some(CgroupLimitValue::Bytes(0)),
                    after_command: Some(CgroupLimitValue::Bytes(0)),
                    zero_before_command: true,
                    zero_after_command: true,
                    unchanged: true,
                    verified: true,
                    diagnostic: None,
                },
                cpu_stat: CgroupCpuStatEvidence {
                    before_command: Some(CgroupCpuStatSnapshot {
                        nr_throttled: 0,
                        throttled_usec: 0,
                    }),
                    after_command: Some(CgroupCpuStatSnapshot {
                        nr_throttled: 0,
                        throttled_usec: 0,
                    }),
                    nr_throttled_delta: Some(0),
                    throttled_usec_delta: Some(0),
                    nr_throttled_unchanged: true,
                    throttled_usec_unchanged: true,
                    verified: true,
                    diagnostic: None,
                },
                leaf_removed: Some(true),
                diagnostic: None,
            },
            memory: PeakMemoryEvidence {
                peak_bytes,
                metric: PeakMemoryMetric::CgroupAccountedMemory,
                scope: PeakMemoryScope::ProcessTree,
                method: PeakMemoryMethod::LinuxCgroupV2MemoryPeak,
                complete: peak_bytes.is_some(),
                sampling_interval_ms: None,
                samples: None,
                direct_child_peak_rss_bytes: None,
                diagnostic: None,
            },
            strict_qualified: peak_bytes.is_some(),
            qualification_failures: if peak_bytes.is_some() {
                Vec::new()
            } else {
                vec!["missing peak".to_string()]
            },
        }
    }

    fn test_spec() -> CompareSpec {
        CompareSpec {
            name: "Spec".to_string(),
            tla_path: PathBuf::from("Spec.tla"),
            cfg_path: PathBuf::from("Spec.cfg"),
            expected_tlc_states: None,
            expected_backend_states: None,
            expected_tlc_error: None,
            expected_backend_error: None,
            work_equivalence: Some(WorkEquivalenceEvidence::exhaustive_holds()),
        }
    }

    fn successful_run(
        run_index: usize,
        tlc_seconds: f64,
        backend_seconds: f64,
        tlc_rss: Option<u64>,
        backend_rss: Option<u64>,
    ) -> CompareRun {
        let mut tlc = obs_with_transitions(Some(10), Some(20), tlc_seconds, None);
        let mut backend = obs_with_transitions(Some(10), Some(20), backend_seconds, None);
        tlc.resource_evidence = qualifying_resource_evidence(tlc_rss);
        backend.resource_evidence = qualifying_resource_evidence(backend_rss);
        CompareRun::new(
            run_index,
            if run_index % 2 == 1 {
                CompareRunOrder::TlcThenTy
            } else {
                CompareRunOrder::TyThenTlc
            },
            tlc,
            None,
            backend,
        )
    }

    fn successful_auto_run(
        run_index: usize,
        production_tlc_seconds: f64,
        backend_seconds: f64,
        count_tlc_seconds: f64,
        count_seconds: f64,
    ) -> CompareRun {
        let schedule = auto_performance_schedule(run_index);
        let mut production_tlc =
            obs_with_transitions(Some(10), Some(20), production_tlc_seconds, None);
        let mut backend = obs_with_transitions(Some(7), Some(12), backend_seconds, None);
        let mut count_tlc = obs_with_transitions(Some(10), Some(20), count_tlc_seconds, None);
        let mut count_verify = obs_with_transitions(Some(10), Some(20), count_seconds, None);
        production_tlc.resource_evidence = qualifying_resource_evidence(Some(1000));
        backend.resource_evidence = qualifying_resource_evidence(Some(500));
        count_tlc.resource_evidence = qualifying_resource_evidence(Some(900));
        count_verify.resource_evidence = qualifying_resource_evidence(Some(900));
        set_test_artifact_dir(
            &mut production_tlc,
            format!("/tmp/run-{run_index:03}/production-tlc"),
        );
        set_test_artifact_dir(&mut backend, format!("/tmp/run-{run_index:03}/auto-cpu"));
        set_test_artifact_dir(&mut count_tlc, format!("/tmp/run-{run_index:03}/count-tlc"));
        set_test_artifact_dir(
            &mut count_verify,
            format!("/tmp/run-{run_index:03}/count-verify"),
        );
        CompareRun::new_with_count_pair(
            run_index,
            schedule.production_pair_order,
            schedule.count_pair_order,
            schedule.pair_block_order,
            production_tlc,
            count_tlc,
            count_verify,
            backend,
        )
    }

    #[test]
    fn repeated_compare_uses_complete_median_paired_ratios_for_both_axes() {
        let runs = [4.0, 1.0, 2.0, 8.0, 3.0]
            .into_iter()
            .enumerate()
            .map(|(index, backend_seconds)| {
                successful_run(index + 1, 10.0, backend_seconds, Some(1000), Some(500))
            })
            .collect();

        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::Pass);
        assert_eq!(row.claim_class, ClaimClass::PassBoth);
        assert_eq!(row.runtime_axis, CompareAxis::Pass);
        assert_eq!(row.memory_axis, CompareAxis::Pass);
        assert_eq!(row.speedup_tlc_vs_backend, Some(10.0 / 3.0));
        assert_eq!(row.memory_ratio_backend_vs_tlc, Some(0.5));
    }

    #[test]
    fn repeated_compare_rejects_one_missing_memory_sample_instead_of_partial_median() {
        let mut runs = (1..=5)
            .map(|index| successful_run(index, 10.0, 5.0, Some(1000), Some(500)))
            .collect::<Vec<_>>();
        runs[3].backend_run.resource_evidence = qualifying_resource_evidence(None);
        runs[3].memory_ratio_backend_vs_tlc = None;

        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::MissingMemory);
        assert_eq!(row.claim_class, ClaimClass::MissingOrStale);
        assert_eq!(row.runtime_axis, CompareAxis::Pass);
        assert_eq!(row.memory_axis, CompareAxis::MissingOrStale);
        assert_eq!(row.memory_ratio_backend_vs_tlc, None);
    }

    #[test]
    fn repeated_compare_rejects_diagnostic_resource_fallback_as_strict_evidence() {
        let mut runs = (1..=5)
            .map(|index| successful_run(index, 10.0, 5.0, Some(1000), Some(500)))
            .collect::<Vec<_>>();
        runs[2].backend_run.resource_evidence.strict_qualified = false;
        runs[2].backend_run.resource_evidence.cpu.method =
            CpuConfinementMethod::InheritedUnmodified;
        runs[2].backend_run.resource_evidence.cpu.confined = false;
        runs[2]
            .backend_run
            .resource_evidence
            .qualification_failures
            .push("diagnostic fallback".to_string());

        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::MissingEvidence);
        assert_eq!(row.claim_class, ClaimClass::MissingOrStale);
        assert_eq!(row.runtime_axis, CompareAxis::MissingOrStale);
        assert_eq!(row.memory_axis, CompareAxis::MissingOrStale);
        assert!(row.reason.contains("one-CPU"));
    }

    #[test]
    fn speed_only_policy_rejects_unqualified_non_cpu_resource_evidence() {
        let mut runs = (1..=5)
            .map(|index| successful_run(index, 10.0, 5.0, Some(1000), Some(500)))
            .collect::<Vec<_>>();
        runs[2].backend_run.resource_evidence.strict_qualified = false;
        runs[2]
            .backend_run
            .resource_evidence
            .qualification_failures
            .push("memory.swap.max was not proven equal to zero".to_string());

        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeed,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::MissingEvidence);
        assert_eq!(row.claim_class, ClaimClass::MissingOrStale);
        assert_eq!(row.runtime_axis, CompareAxis::MissingOrStale);
        assert_eq!(row.memory_axis, CompareAxis::NotRequired);
        assert!(row
            .reason
            .contains("strict execution envelope is unqualified"));
        assert!(row.reason.contains("memory.swap.max"));
    }

    #[test]
    fn repeated_compare_rejects_nondeterministic_count_observations() {
        let mut runs = (1..=5)
            .map(|index| successful_run(index, 10.0, 5.0, Some(1000), Some(500)))
            .collect::<Vec<_>>();
        runs[4].backend_run.states_found = Some(11);

        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::ParityFail);
        assert_eq!(row.claim_class, ClaimClass::ParityBlocker);
        assert!(row.reason.contains("not deterministic"));
    }

    #[test]
    fn repeated_compare_requires_stable_engine_provenance() {
        let mut missing = (1..=6)
            .map(|index| successful_run(index, 10.0, 5.0, Some(1000), Some(500)))
            .collect::<Vec<_>>();
        missing[2].backend_run.engine_tier = None;
        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            missing,
        );
        assert_eq!(row.class, CompareClass::MissingEvidence);
        assert!(row.reason.contains("engine-tier provenance"));

        let mut drift = (1..=6)
            .map(|index| successful_run(index, 10.0, 5.0, Some(1000), Some(500)))
            .collect::<Vec<_>>();
        drift[4].backend_run.engine_tier = Some("different-tier".to_string());
        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            drift,
        );
        assert_eq!(row.class, CompareClass::ParityFail);
        assert!(row.reason.contains("not deterministic"));
        assert!(row.reason.contains("engine_tier"));
    }

    #[test]
    fn engine_tier_parser_uses_last_nonempty_report() {
        assert_eq!(
            parse_engine_tier(
                "noise\n[engine] execution tier: interpreter\n[engine] execution tier: trust-cg native-fused (compiled BFS)\n"
            )
            .as_deref(),
            Some("trust-cg native-fused (compiled BFS)")
        );
        assert_eq!(parse_engine_tier("[engine] execution tier:   \n"), None);
    }

    #[test]
    fn production_uses_exact_count_arm_and_reduced_auto_arm_only_for_performance() {
        let runs: Vec<_> = (1..=STRICT_MIN_BALANCED_PAIRED_RUNS)
            .map(|index| successful_auto_run(index, 10.0, 5.0, 10.0, 4.0))
            .collect();

        for backend in [
            SupremacyCompareBackend::Auto,
            SupremacyCompareBackend::AutoCpu,
        ] {
            let row = CompareRow::classify(
                &test_spec(),
                1,
                "default",
                backend,
                SupremacyComparePolicy::ParityAndSpeedAndMemory,
                1.05,
                0.95,
                runs.clone(),
            );

            assert_eq!(row.class, CompareClass::Pass, "{backend:?}");
            assert_eq!(row.claim_class, ClaimClass::PassBoth, "{backend:?}");
            assert_eq!(
                row.count_verify_run.as_ref().unwrap().states_found,
                Some(10),
                "{backend:?}"
            );
            assert_eq!(row.backend_run.states_found, Some(7), "{backend:?}");
            assert_eq!(
                row.speedup_count_tlc_vs_count_verify,
                Some(2.5),
                "{backend:?}"
            );
            assert_eq!(row.memory_ratio_backend_vs_tlc, Some(0.5), "{backend:?}");
            assert_eq!(
                row.memory_ratio_count_verify_vs_count_tlc,
                Some(1.0),
                "{backend:?}"
            );

            let mut args = test_args();
            args.backend = backend;
            args.policy = SupremacyComparePolicy::ParityAndSpeedAndMemory;
            let report = CompareReport::new(&args, PathBuf::from("out"), Vec::new(), vec![row]);
            let json = serde_json::to_value(&report).unwrap();
            assert_eq!(
                report.strict_superiority_passed,
                backend == SupremacyCompareBackend::AutoCpu,
                "{backend:?}"
            );
            assert_eq!(report.schema, "ty.supremacy.compare.v4");
            assert_eq!(
                json["disk_high_water_disclosure"]["measurement_role"],
                "informational_only_no_superiority_threshold"
            );
            assert_eq!(
                json["disk_high_water_disclosure"]["sampling_can_perturb_elapsed"],
                true
            );
            assert_eq!(
                json["rows"][0]["work_equivalence"]["rule_id"],
                EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID
            );
            assert!(json["rows"][0]["backend_run"]["disk_high_water"].is_object());
            assert_eq!(
                report.tool_order,
                "alternating_within_production_and_count_pairs"
            );
            assert_eq!(
                report.count_verify_schedule,
                "independent_tlc_count_pair_with_alternating_pair_blocks"
            );
            let first_run = &json["rows"][0]["runs"][0];
            assert_eq!(first_run["production_pair_order"], "tlc_then_ty");
            assert_eq!(first_run["count_pair_order"], "ty_then_tlc");
            assert_eq!(first_run["pair_block_order"], "production_then_count");
            assert_eq!(
                first_run["tlc"]["artifact_dir"],
                "/tmp/run-001/production-tlc"
            );
            assert_eq!(
                first_run["count_tlc_run"]["artifact_dir"],
                "/tmp/run-001/count-tlc"
            );
            assert_eq!(
                first_run["count_verify_run"]["artifact_dir"],
                "/tmp/run-001/count-verify"
            );
            assert_eq!(first_run["speedup_count_tlc_vs_count_verify"], 2.5);
            assert_eq!(first_run["memory_ratio_count_verify_vs_count_tlc"], 1.0);
        }
    }

    #[test]
    fn auto_speed_schedule_balances_both_pair_orders_and_pair_block_order() {
        let schedules = (1..=STRICT_MIN_BALANCED_PAIRED_RUNS)
            .map(auto_performance_schedule)
            .collect::<Vec<_>>();

        for order in [CompareRunOrder::TlcThenTy, CompareRunOrder::TyThenTlc] {
            assert_eq!(
                schedules
                    .iter()
                    .filter(|schedule| schedule.production_pair_order == order)
                    .count(),
                STRICT_MIN_BALANCED_PAIRED_RUNS / 2
            );
            assert_eq!(
                schedules
                    .iter()
                    .filter(|schedule| schedule.count_pair_order == order)
                    .count(),
                STRICT_MIN_BALANCED_PAIRED_RUNS / 2
            );
        }
        assert_eq!(
            schedules
                .iter()
                .filter(|schedule| {
                    schedule.pair_block_order == ComparePairBlockOrder::ProductionThenCount
                })
                .count(),
            STRICT_MIN_BALANCED_PAIRED_RUNS / 2
        );
        assert_eq!(
            schedules
                .iter()
                .filter(|schedule| {
                    schedule.pair_block_order == ComparePairBlockOrder::CountThenProduction
                })
                .count(),
            STRICT_MIN_BALANCED_PAIRED_RUNS / 2
        );
        assert!(schedules.iter().all(|schedule| {
            schedule.production_pair_order != schedule.count_pair_order
                && schedule.pair_block_order != ComparePairBlockOrder::ProductionOnly
        }));
    }

    #[test]
    fn auto_speed_policy_reports_count_arm_runtime_loss_independently() {
        let runs = (1..=STRICT_MIN_BALANCED_PAIRED_RUNS)
            .map(|index| successful_auto_run(index, 10.0, 5.0, 10.0, 10.0))
            .collect();
        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::AutoCpu,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.speedup_tlc_vs_backend, Some(2.0));
        assert_eq!(row.speedup_count_tlc_vs_count_verify, Some(1.0));
        assert_eq!(row.class, CompareClass::SpeedFail);
        assert_eq!(row.claim_class, ClaimClass::RuntimeLoss);
        assert_eq!(row.runtime_axis, CompareAxis::Loss);
        assert_eq!(row.memory_axis, CompareAxis::Pass);
        assert!(row.reason.contains("count paired speedup=1.000000x"));
    }

    #[test]
    fn auto_speed_policy_reports_missing_count_ratio_as_stale() {
        let mut runs = (1..=STRICT_MIN_BALANCED_PAIRED_RUNS)
            .map(|index| successful_auto_run(index, 10.0, 5.0, 10.0, 4.0))
            .collect::<Vec<_>>();
        runs[2].count_verify_run.as_mut().unwrap().elapsed_seconds = 0.0;
        runs[2].speedup_count_tlc_vs_count_verify = None;
        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::AutoCpu,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.speedup_tlc_vs_backend, Some(2.0));
        assert_eq!(row.speedup_count_tlc_vs_count_verify, None);
        assert_eq!(row.class, CompareClass::MissingRuntime);
        assert_eq!(row.claim_class, ClaimClass::MissingOrStale);
        assert_eq!(row.runtime_axis, CompareAxis::MissingOrStale);
        assert_eq!(row.memory_axis, CompareAxis::Pass);
        assert!(row.reason.contains("count paired speedup=missing"));
    }

    #[test]
    fn auto_speed_rejects_stable_semantic_drift_between_tlc_pairs() {
        let mut runs = (1..=STRICT_MIN_BALANCED_PAIRED_RUNS)
            .map(|index| successful_auto_run(index, 10.0, 5.0, 10.0, 4.0))
            .collect::<Vec<_>>();
        for run in &mut runs {
            let count_tlc = run.count_tlc_run.as_mut().unwrap();
            count_tlc.states_found = Some(11);
            count_tlc.transitions = Some(21);
        }
        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::AutoCpu,
            SupremacyComparePolicy::ParityAndSpeed,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::ParityFail);
        assert_eq!(row.claim_class, ClaimClass::ParityBlocker);
        assert!(row
            .reason
            .contains("production-pair TLC and count-pair TLC outcomes differ"));
    }

    #[test]
    fn auto_parity_only_retains_the_single_tlc_compatibility_protocol() {
        let tlc = obs_with_transitions(Some(10), Some(20), 10.0, None);
        let count = obs_with_transitions(Some(10), Some(20), 4.0, None);
        let production = obs_with_transitions(Some(7), Some(12), 5.0, None);
        let run = CompareRun::new(1, CompareRunOrder::TlcThenTy, tlc, Some(count), production);
        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::AutoCpu,
            SupremacyComparePolicy::Parity,
            1.05,
            0.95,
            vec![run],
        );

        assert_eq!(row.class, CompareClass::Pass);
        assert!(row.count_tlc_run.is_none());
        assert!(row.speedup_count_tlc_vs_count_verify.is_none());
        assert_eq!(
            row.runs[0].pair_block_order,
            ComparePairBlockOrder::ProductionThenCount
        );
    }

    #[test]
    fn auto_speed_strict_resources_cover_count_tlc_pair() {
        let mut runs = (1..=STRICT_MIN_BALANCED_PAIRED_RUNS)
            .map(|index| successful_auto_run(index, 10.0, 5.0, 10.0, 4.0))
            .collect::<Vec<_>>();
        let count_tlc = runs[3].count_tlc_run.as_mut().unwrap();
        count_tlc.resource_evidence.strict_qualified = false;
        count_tlc
            .resource_evidence
            .qualification_failures
            .push("count TLC envelope was not isolated".to_string());
        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::AutoCpu,
            SupremacyComparePolicy::ParityAndSpeed,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::MissingEvidence);
        assert_eq!(row.claim_class, ClaimClass::MissingOrStale);
        assert_eq!(row.runtime_axis, CompareAxis::MissingOrStale);
        assert!(row.reason.contains("count-tlc"));
        assert!(row.reason.contains("not isolated"));
    }

    #[test]
    fn auto_speed_strict_disk_evidence_covers_every_count_pair_arm() {
        let mut runs = (1..=STRICT_MIN_BALANCED_PAIRED_RUNS)
            .map(|index| successful_auto_run(index, 10.0, 5.0, 10.0, 4.0))
            .collect::<Vec<_>>();
        let count_tlc = runs[2].count_tlc_run.as_mut().unwrap();
        count_tlc.disk_high_water.samples_partial = 1;
        count_tlc.disk_high_water.samples_complete = 1;
        count_tlc.disk_high_water.polling_complete = false;
        count_tlc.disk_high_water.complete = false;
        count_tlc.disk_high_water.strict_qualified = false;
        count_tlc
            .disk_high_water
            .qualification_failures
            .push("scan budget exceeded".to_string());
        count_tlc
            .disk_high_water
            .environment_confinement
            .remove("HOME");

        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::AutoCpu,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::MissingEvidence);
        assert_eq!(row.runtime_axis, CompareAxis::MissingOrStale);
        assert_eq!(row.memory_axis, CompareAxis::MissingOrStale);
        assert!(row.reason.contains("count-tlc"));
        assert!(row.reason.contains("strict sampled disk high-water"));
        assert!(row.reason.contains("scan budget exceeded"));
        assert!(row.reason.contains("exact full command-scratch map"));
    }

    #[test]
    fn sampled_disk_peaks_are_aggregated_separately_and_remain_in_raw_runs() {
        let mut runs = (1..=3)
            .map(|index| successful_run(index, 10.0, 5.0, Some(1000), Some(500)))
            .collect::<Vec<_>>();
        for (run, (allocated, apparent)) in
            runs.iter_mut()
                .zip([(100_u64, 1000_u64), (500, 3000), (300, 2000)])
        {
            run.backend_run.disk_high_water.peak_allocated_bytes = Some(allocated);
            run.backend_run.disk_high_water.peak_apparent_bytes = Some(apparent);
        }

        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::Parity,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(
            row.backend_run.disk_high_water.peak_allocated_bytes,
            Some(300)
        );
        assert_eq!(
            row.backend_run.disk_high_water.peak_apparent_bytes,
            Some(2000)
        );
        assert_eq!(
            row.runs[0].backend_run.disk_high_water.peak_allocated_bytes,
            Some(100)
        );
    }

    #[test]
    fn successful_performance_requires_exact_typed_work_equivalence() {
        let mut spec = test_spec();
        spec.work_equivalence = None;
        let runs = (1..=STRICT_MIN_BALANCED_PAIRED_RUNS)
            .map(|index| successful_run(index, 10.0, 5.0, Some(1000), Some(500)))
            .collect();

        let row = CompareRow::classify(
            &spec,
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::MissingEvidence);
        assert!(row.reason.contains("exact typed work_equivalence"));
        assert!(row.reason.contains("raw initial"));
    }

    #[test]
    fn early_violation_cannot_borrow_the_exhaustive_work_equivalence_rule() {
        let runs = (1..=5)
            .map(|index| {
                let mut tlc = obs(None, 10.0, Some("invariant"));
                let mut backend = obs(None, 5.0, Some("invariant"));
                tlc.resource_evidence = qualifying_resource_evidence(Some(1000));
                backend.resource_evidence = qualifying_resource_evidence(Some(500));
                CompareRun::new(
                    index,
                    if index % 2 == 1 {
                        CompareRunOrder::TlcThenTy
                    } else {
                        CompareRunOrder::TyThenTlc
                    },
                    tlc,
                    None,
                    backend,
                )
            })
            .collect();

        let row = CompareRow::classify(
            &test_spec(),
            1,
            "default",
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.05,
            0.95,
            runs,
        );

        assert_eq!(row.class, CompareClass::MissingEvidence);
        assert_eq!(row.claim_class, ClaimClass::MissingOrStale);
        assert!(row.reason.contains("correctness-only evidence"));
    }

    #[test]
    fn baseline_work_equivalence_rejects_legacy_aliases_and_wrong_typed_rule() {
        for alias in [
            "work_equivalence_rule",
            "equivalent_work_rule",
            "performance_work_equivalence_rule",
        ] {
            for alias_value in [json!("same work"), json!(null)] {
                let mut value = json!({
                    "source": null,
                    "tlc": {},
                    "ty": {},
                });
                value[alias] = alias_value;
                let entry: SpecBaselineEntry = serde_json::from_value(value).unwrap();
                let error = validated_baseline_work_equivalence("Spec", &entry).unwrap_err();
                assert!(error.to_string().contains(alias), "{alias}: {error:#}");
            }
        }

        let entry: SpecBaselineEntry = serde_json::from_value(json!({
            "source": null,
            "tlc": {},
            "ty": {},
            "work_equivalence": {
                "schema_version": 2,
                "rule_id": EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID,
            }
        }))
        .unwrap();
        assert!(validated_baseline_work_equivalence("Spec", &entry).is_err());

        let entry: SpecBaselineEntry = serde_json::from_value(json!({
            "source": null,
            "tlc": {},
            "ty": {},
            "work_equivalence": {
                "schema_version": WORK_EQUIVALENCE_SCHEMA_VERSION,
                "rule_id": EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID,
            }
        }))
        .unwrap();
        assert_eq!(
            validated_baseline_work_equivalence("Spec", &entry).unwrap(),
            Some(WorkEquivalenceEvidence::exhaustive_holds())
        );
    }

    #[test]
    fn parity_policy_passes_matching_states_without_speed_requirement() {
        let tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let backend = obs_with_transitions(Some(10), Some(20), 10.0, None);

        let result = classify_observations(
            Some(10),
            Some(10),
            None,
            None,
            SupremacyComparePolicy::Parity,
            2.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::Pass);
        assert!(result.parity_states);
        assert!(result.parity_generated_work);
    }

    #[test]
    fn parity_policy_rejects_state_drift() {
        let tlc = obs(Some(10), 1.0, None);
        let backend = obs(Some(11), 0.5, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(!result.parity_states);
    }

    #[test]
    fn violation_obligation_identity_is_extracted_and_required_for_parity() {
        assert_eq!(
            violation_obligation(Some("invariant"), "Error: Invariant TypeOK is violated.")
                .as_deref(),
            Some("TypeOK")
        );
        let tlc = obs(None, 1.0, Some("invariant"));
        let mut backend = obs(None, 0.5, Some("invariant"));
        backend.violated_obligation = Some("DifferentInvariant".to_string());

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.05,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ErrorMismatch);
        assert!(result.reason.contains("violated-obligation"));
    }

    #[test]
    fn parity_policy_rejects_transition_drift_after_state_parity() {
        let tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let backend = obs_with_transitions(Some(10), Some(21), 0.5, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(result.parity_states);
        assert!(!result.parity_generated_work);
        assert!(result.reason.contains("generated-state parity failed"));
    }

    #[test]
    fn parity_policy_rejects_one_sided_missing_raw_generated_count() {
        let tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let mut backend = obs_with_transitions(Some(10), None, 0.5, None);
        backend.transitions = Some(20);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(result.parity_states);
        assert!(!result.parity_generated_work);
        assert!(result
            .reason
            .contains("backend initial=Some(1) successors=None total=None"));
    }

    #[test]
    fn parity_policy_rejects_matching_but_arithmetically_invalid_raw_counts() {
        let mut tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let mut backend = obs_with_transitions(Some(10), Some(20), 0.5, None);
        tlc.raw_successors_generated = Some(20);
        backend.raw_successors_generated = Some(20);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(!result.parity_generated_work);
        assert!(result.reason.contains("arithmetically inconsistent"));
    }

    #[test]
    fn speed_policy_rejects_transition_drift_before_speed_check() {
        let tlc = obs_with_transitions(Some(10), Some(20), 1.0, None);
        let backend = obs_with_transitions(Some(10), Some(21), 2.0, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeed,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(!result.parity_generated_work);
    }

    #[test]
    fn speed_policy_rejects_below_min_speedup() {
        let tlc = obs(Some(10), 1.0, None);
        let backend = obs(Some(10), 2.0, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeed,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::SpeedFail);
        assert_eq!(result.runtime_axis, CompareAxis::Loss);
        assert_eq!(result.memory_axis, CompareAxis::NotRequired);
        assert!(result.parity_states);
    }

    #[test]
    fn speed_policy_rejects_min_speedup_boundary() {
        let tlc = obs_with_transitions(Some(10), Some(20), 2.0, None);
        let backend = obs_with_transitions(Some(10), Some(20), 2.0, None);

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeed,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::SpeedFail);
        assert_eq!(result.runtime_axis, CompareAxis::Loss);
        assert_eq!(result.memory_axis, CompareAxis::NotRequired);
    }

    #[test]
    fn memory_policy_rejects_peak_memory_above_limit() {
        let mut tlc = obs(Some(10), 2.0, None);
        let mut backend = obs(Some(10), 1.0, None);
        tlc.resource_evidence = qualifying_resource_evidence(Some(1000));
        backend.resource_evidence = qualifying_resource_evidence(Some(1100));

        let result = classify_observations_with_limits(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.0,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
            memory_ratio(
                process_tree_peak_memory_bytes(&backend),
                process_tree_peak_memory_bytes(&tlc),
            ),
        );

        assert_eq!(result.class, CompareClass::MemoryFail);
        assert!(result.reason.contains("peak-memory ratio"));
    }

    #[test]
    fn memory_policy_rejects_configured_ratio_boundary() {
        let mut tlc = obs(Some(10), 2.0, None);
        let mut backend = obs(Some(10), 1.0, None);
        tlc.resource_evidence = qualifying_resource_evidence(Some(1000));
        backend.resource_evidence = qualifying_resource_evidence(Some(900));

        let result = classify_observations_with_limits(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.0,
            0.9,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
            memory_ratio(
                process_tree_peak_memory_bytes(&backend),
                process_tree_peak_memory_bytes(&tlc),
            ),
        );

        assert_eq!(result.class, CompareClass::MemoryFail);
    }

    #[test]
    fn memory_policy_rejects_missing_peak_memory() {
        let tlc = obs(Some(10), 2.0, None);
        let mut backend = obs(Some(10), 1.0, None);
        backend.resource_evidence = qualifying_resource_evidence(None);

        let result = classify_observations_with_limits(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.0,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
            memory_ratio(
                process_tree_peak_memory_bytes(&backend),
                process_tree_peak_memory_bytes(&tlc),
            ),
        );

        assert_eq!(result.class, CompareClass::MissingMemory);
    }

    #[test]
    fn compatible_error_outcomes_pass_before_state_parity() {
        let tlc = obs(None, 1.0, Some("invariant"));
        let backend = obs(None, 0.5, Some("safety"));

        let result = classify_observations(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::Pass);
    }

    #[test]
    fn compatible_error_outcomes_still_enforce_efficiency_policy() {
        let mut tlc = obs(None, 1.0, Some("invariant"));
        let mut backend = obs(None, 2.0, Some("safety"));
        tlc.resource_evidence = qualifying_resource_evidence(Some(1000));
        backend.resource_evidence = qualifying_resource_evidence(Some(2000));

        let result = classify_observations_with_limits(
            None,
            None,
            None,
            None,
            SupremacyComparePolicy::ParityAndSpeedAndMemory,
            1.0,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
            memory_ratio(
                process_tree_peak_memory_bytes(&backend),
                process_tree_peak_memory_bytes(&tlc),
            ),
        );

        assert_eq!(result.class, CompareClass::BothFail);
        assert_eq!(result.runtime_axis, CompareAxis::Loss);
        assert_eq!(result.memory_axis, CompareAxis::Loss);
    }

    #[test]
    fn compatible_error_outcomes_still_enforce_expected_counts() {
        let tlc = obs(Some(10), 1.0, Some("invariant"));
        let backend = obs(Some(11), 0.5, Some("safety"));

        let result = classify_observations(
            Some(10),
            Some(10),
            Some("invariant"),
            Some("safety"),
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ExpectedStateMismatch);
    }

    #[test]
    fn baseline_error_aliases_normalize_before_comparison() {
        let tlc = SpecBaselineMode {
            states: Some(10),
            error_type: Some("invariant".to_string()),
            status: Some("fail".to_string()),
        };
        let backend = SpecBaselineMode {
            states: Some(10),
            error_type: Some("invariant_violation".to_string()),
            status: Some("pass".to_string()),
        };

        assert_eq!(expected_error_type(&tlc).as_deref(), Some("invariant"));
        assert_eq!(expected_error_type(&backend).as_deref(), Some("invariant"));
    }

    #[test]
    fn tool_specific_expected_counts_do_not_override_cross_tool_parity() {
        let tlc = obs(Some(87898), 20.0, None);
        let backend = obs(Some(27242), 10.0, None);

        let result = classify_observations(
            Some(87898),
            Some(27242),
            None,
            None,
            SupremacyComparePolicy::Parity,
            1.0,
            &tlc,
            &backend,
            speedup(tlc.elapsed_seconds, backend.elapsed_seconds),
        );

        assert_eq!(result.class, CompareClass::ParityFail);
        assert!(!result.parity_states);
    }

    #[test]
    fn java_tlc_runner_uses_auditable_single_thread_jvm_profile() {
        let argv = tlc_java_single_thread_base_argv();

        assert_eq!(argv[0], "java");
        for arg in tlc_java_single_thread_args() {
            assert!(argv.contains(&(*arg).to_string()), "{arg}");
        }
        assert!(!argv.contains(&"-XX:+UseParallelGC".to_string()));
    }

    #[test]
    fn enforced_speed_compare_rejects_opaque_tlc_executable_runner() {
        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeed;
        args.tlc_bin = Some(PathBuf::from("/tmp/tlc-wrapper"));

        let error = validate_args(&args).expect_err("opaque TLC runner should be rejected");

        assert!(error.to_string().contains("auditable Java TLC runner"));
    }

    #[test]
    fn enforced_speed_compare_rejects_multiworker_or_ty_only_flag_claims() {
        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeed;
        args.workers = vec![1, 2];

        let error = validate_args(&args).expect_err("multiworker speed claim should be rejected");
        assert!(error.to_string().contains("--workers 1"));

        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeed;
        args.ty_flag = vec!["--max-depth".to_string(), "3".to_string()];

        let error = validate_args(&args).expect_err("TY-only flags should be rejected");
        assert!(error.to_string().contains("--ty-flag"));
    }

    #[test]
    fn enforced_performance_compare_requires_six_balanced_repetitions() {
        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeedAndMemory;
        args.runs = 4;

        let error = validate_args(&args).expect_err("undersampled performance claim must fail");

        assert!(error.to_string().contains("even --runs >= 6"));

        args.runs = 5;
        let error = validate_args(&args).expect_err("odd undersampled claim must fail");
        assert!(error.to_string().contains("even --runs >= 6"));

        args.runs = 7;
        let error = validate_args(&args).expect_err("unbalanced performance claim must fail");
        assert!(error.to_string().contains("equal representation"));

        args.runs = 6;
        validate_args(&args).expect("six repetitions balance both launch orders");
    }

    #[test]
    fn enforced_performance_compare_rejects_weakened_strict_margins() {
        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeedAndMemory;
        args.min_speedup = 1.0;

        let error = validate_args(&args).expect_err("weak runtime margin must fail");
        assert!(error.to_string().contains("--min-speedup >= 1.05"));

        let mut args = test_args();
        args.mode = SupremacyMode::Enforce;
        args.policy = SupremacyComparePolicy::ParityAndSpeedAndMemory;
        args.max_memory_ratio = 1.0;

        let error = validate_args(&args).expect_err("weak memory margin must fail");
        assert!(error.to_string().contains("--max-memory-ratio <= 0.95"));
    }

    #[test]
    fn compare_rejects_invalid_memory_ratio() {
        let mut args = test_args();
        args.max_memory_ratio = 0.0;

        let error = validate_args(&args).expect_err("zero memory ratio should be rejected");

        assert!(error.to_string().contains("--max-memory-ratio"));
    }

    #[test]
    fn trust_cg_backend_env_uses_strict_native_fused_launch_controls() {
        let env = backend_env(SupremacyCompareBackend::TrustCg);

        for (key, value) in [
            ("TY_trust_cg", "1"),
            ("TY_TRUST_CG_BFS", "1"),
            ("TY_TRUST_CG_EXISTS", "1"),
            ("TY_BYTECODE_VM", "1"),
            ("TY_BYTECODE_VM_STATS", "1"),
            ("TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST", "strict"),
            ("TY_TRUST_CG_NATIVE_FUSED_STRICT", "1"),
            ("TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS", "1"),
            ("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP", "1"),
            ("TY_DISABLE_ARTIFACT_CACHE", "1"),
        ] {
            assert_eq!(env.get(key).map(String::as_str), Some(value), "{key}");
        }
        // Count-parity (auto-POR/auto-symmetry off) is the `--no-reduction`
        // CLI flag in the child argv, never an env pin: the child `ty check`
        // ignores ambient TY_AUTO_POR / TY_AUTO_SYMMETRY.
        assert_eq!(env.get("TY_AUTO_POR"), None);
        assert_eq!(env.get("TY_AUTO_SYMMETRY"), None);
    }

    #[test]
    fn auto_cpu_keeps_no_gpu_on_performance_and_count_arms() {
        let mut auto_performance = Vec::new();
        append_ty_backend_args(&mut auto_performance, SupremacyCompareBackend::Auto);
        assert!(auto_performance.is_empty());

        let mut auto_cpu_performance = Vec::new();
        append_ty_backend_args(&mut auto_cpu_performance, SupremacyCompareBackend::AutoCpu);
        assert_eq!(auto_cpu_performance, ["--no-gpu"]);

        let mut auto_cpu_count = vec!["--bfs-only".to_string(), "--no-reduction".to_string()];
        append_ty_backend_args(&mut auto_cpu_count, SupremacyCompareBackend::AutoCpu);
        assert_eq!(auto_cpu_count, ["--bfs-only", "--no-reduction", "--no-gpu"]);
    }

    #[test]
    fn compare_cases_default_to_single_default_case() {
        let args = test_args();

        let cases = resolve_cases(&args).expect("default case should resolve");

        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].name, DEFAULT_CASE);
        assert!(cases[0].env_overrides.is_empty());
    }

    #[test]
    fn compare_cases_apply_global_env_to_every_case() {
        let mut args = test_args();
        args.cases = vec!["control".to_string(), "treatment".to_string()];
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()];

        let cases = resolve_cases(&args).expect("global env should resolve");

        assert_eq!(cases.len(), 2);
        for case in cases {
            assert_eq!(
                case.env_overrides
                    .get("TY_PARALLEL_READONLY_VALUE_CACHES")
                    .map(String::as_str),
                Some("1")
            );
        }
    }

    #[test]
    fn compare_case_env_applies_to_named_case_and_overrides_global() {
        let mut args = test_args();
        args.cases = vec!["control".to_string(), "treatment".to_string()];
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES=0".to_string()];
        args.case_env = vec!["treatment:TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()];

        let cases = resolve_cases(&args).expect("case env should resolve");
        let control = cases.iter().find(|case| case.name == "control").unwrap();
        let treatment = cases.iter().find(|case| case.name == "treatment").unwrap();

        assert_eq!(
            control
                .env_overrides
                .get("TY_PARALLEL_READONLY_VALUE_CACHES")
                .map(String::as_str),
            Some("0")
        );
        assert_eq!(
            treatment
                .env_overrides
                .get("TY_PARALLEL_READONLY_VALUE_CACHES")
                .map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn compare_cases_reject_duplicate_case_names() {
        let mut args = test_args();
        args.cases = vec!["same".to_string(), "same".to_string()];

        let error = resolve_cases(&args).expect_err("duplicate cases should fail");

        assert!(error.to_string().contains("duplicate --case"));
    }

    #[test]
    fn compare_case_env_rejects_unknown_case() {
        let mut args = test_args();
        args.cases = vec!["control".to_string()];
        args.case_env = vec!["treatment:TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()];

        let error = resolve_cases(&args).expect_err("unknown case should fail");

        assert!(error.to_string().contains("unknown case"));
    }

    #[test]
    fn compare_env_rejects_malformed_assignment() {
        let mut args = test_args();
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES".to_string()];

        let error = resolve_cases(&args).expect_err("malformed env should fail");

        assert!(error.to_string().contains("KEY=VALUE"));
    }

    #[test]
    fn compare_env_rejects_protected_backend_keys() {
        let mut args = test_args();
        args.backend = SupremacyCompareBackend::TrustCg;
        args.ty_env = vec!["TY_trust_cg=0".to_string()];

        let error = resolve_cases(&args).expect_err("protected env should fail");

        assert!(error.to_string().contains("protected backend env key"));
    }

    #[test]
    fn compare_env_accepts_allowed_non_semantic_case_key() {
        let mut args = test_args();
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()];

        let cases = resolve_cases(&args).expect("allowed env should resolve");

        assert_eq!(
            cases[0]
                .env_overrides
                .get("TY_PARALLEL_READONLY_VALUE_CACHES")
                .map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn compare_env_rejects_disallowed_ty_keys() {
        let mut args = test_args();
        args.ty_env = vec!["TY_EXPERIMENT=1".to_string()];

        let error = resolve_cases(&args).expect_err("disallowed TY env should fail");

        assert!(error
            .to_string()
            .contains("not allowed for compare env cases"));
    }

    #[test]
    fn compare_env_rejects_inverse_backend_flags() {
        for key in [
            "TY_NO_FLAT_BFS",
            "TY_NO_COMPILED_BFS",
            "TY_TRUST_CG_DISABLE_COMPILED_BFS_LEVEL",
            "TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP",
            "TY_TRUST_CG_ENTRY_COUNTER_GATE",
        ] {
            let mut args = test_args();
            args.backend = SupremacyCompareBackend::TrustCg;
            args.ty_env = vec![format!("{key}=1")];

            let error = resolve_cases(&args).expect_err("inverse backend env should fail");

            assert!(
                error
                    .to_string()
                    .contains("not allowed for compare env cases")
                    || error.to_string().contains("protected backend env key"),
                "{key}: {error}"
            );
        }
    }

    #[test]
    fn compare_case_env_rejects_inverse_backend_flags() {
        for key in [
            "TY_NO_FLAT_BFS",
            "TY_NO_COMPILED_BFS",
            "TY_TRUST_CG_DISABLE_COMPILED_BFS_LEVEL",
            "TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP",
            "TY_TRUST_CG_ENTRY_COUNTER_GATE",
        ] {
            let mut args = test_args();
            args.backend = SupremacyCompareBackend::TrustCg;
            args.cases = vec!["control".to_string()];
            args.case_env = vec![format!("control:{key}=1")];

            let error = resolve_cases(&args).expect_err("inverse backend case env should fail");

            assert!(
                error
                    .to_string()
                    .contains("not allowed for compare env cases")
                    || error.to_string().contains("protected backend env key"),
                "{key}: {error}"
            );
        }
    }

    #[test]
    fn compare_env_rejects_invalid_allowed_case_key_value() {
        let mut args = test_args();
        args.ty_env = vec!["TY_PARALLEL_READONLY_VALUE_CACHES=true".to_string()];

        let error = resolve_cases(&args).expect_err("invalid allowed env value should fail");

        assert!(error.to_string().contains("accepts only"));
    }

    #[test]
    fn compare_env_rejects_non_ty_keys() {
        let mut args = test_args();
        args.ty_env = vec!["JAVA_TOOL_OPTIONS=-Xmx1g".to_string()];

        let error = resolve_cases(&args).expect_err("non-TY env should fail");

        assert!(error.to_string().contains("only TY_* keys"));
    }

    #[test]
    fn compare_report_serializes_case_inventory_and_rows() {
        let args = test_args();
        let case = EnvCase {
            name: "control".to_string(),
            env_overrides: BTreeMap::from([(
                "TY_PARALLEL_READONLY_VALUE_CACHES".to_string(),
                "1".to_string(),
            )]),
        };
        let spec = CompareSpec {
            name: "Spec".to_string(),
            tla_path: PathBuf::from("Spec.tla"),
            cfg_path: PathBuf::from("Spec.cfg"),
            expected_tlc_states: None,
            expected_backend_states: None,
            expected_tlc_error: None,
            expected_backend_error: None,
            work_equivalence: None,
        };
        let run = CompareRun::new(
            1,
            CompareRunOrder::TlcThenTy,
            obs(Some(1), 1.0, None),
            None,
            obs(Some(1), 0.5, None),
        );
        let row = CompareRow::classify(
            &spec,
            1,
            &case.name,
            SupremacyCompareBackend::Interpreter,
            SupremacyComparePolicy::Parity,
            1.0,
            1.0,
            vec![run],
        );
        let report = CompareReport::new(&args, PathBuf::from("out"), vec![case], vec![row]);
        let json = serde_json::to_value(&report).expect("report serializes");

        assert_eq!(json["cases"][0]["name"], "control");
        assert_eq!(
            json["cases"][0]["env_overrides"]["TY_PARALLEL_READONLY_VALUE_CACHES"],
            "1"
        );
        assert_eq!(json["rows"][0]["case"], "control");
        assert!(report.to_markdown().contains("| Spec | control | 1 |"));
    }
}
