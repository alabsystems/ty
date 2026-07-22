// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared Petri/MCC CLI runtime for `ty` and `pnml-tools`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Args, ValueEnum};

use crate::error::PnmlError;
use crate::examination::Examination;
use crate::explorer::{CheckpointConfig, ExplorationConfig, FpsetBackend, StorageMode};
use crate::output::{cannot_compute_line, formula_line, print_mcc_line, Verdict};

/// Shared runtime knobs for Petri/MCC execution.
#[derive(Debug, Clone, Args)]
pub struct PetriRunArgs {
    /// Timeout in seconds (overrides BK_TIME_CONFINEMENT env var).
    #[arg(long)]
    pub timeout: Option<u64>,

    /// Maximum number of states to explore (0 = auto-size from available memory).
    #[arg(long)]
    pub max_states: Option<usize>,

    /// Fraction of available memory to use for state storage (>0.0 and <=1.0).
    #[arg(long, default_value = "0.25", value_parser = parse_memory_fraction)]
    pub memory_fraction: f64,

    /// Number of BFS worker threads (must be >= 1; 1 = sequential).
    #[arg(long, default_value = "1", value_parser = parse_positive_usize)]
    pub threads: usize,

    /// State-storage backend.
    #[arg(long, value_enum, default_value = "auto")]
    pub storage: RequestedStorageMode,

    /// Directory where mmap/disk storage keeps backing files.
    #[arg(long, value_name = "DIR")]
    pub storage_dir: Option<PathBuf>,

    /// Directory where explorer checkpoints are written.
    #[arg(long)]
    pub checkpoint_dir: Option<PathBuf>,

    /// Save a checkpoint every N explored states (must be >= 1).
    #[arg(long, default_value = "100000", value_parser = parse_positive_usize)]
    pub checkpoint_interval_states: usize,

    /// Resume observer exploration from `--checkpoint-dir`.
    #[arg(long, requires = "checkpoint_dir")]
    pub resume_checkpoint: bool,

    /// Fingerprint set backend for parallel BFS deduplication.
    ///
    /// Defaults to `TY_MCC_FPSET_BACKEND` when set, otherwise `sharded`
    /// for all worker counts (exactness-first); `cas` must be requested
    /// explicitly.
    #[arg(long, value_enum)]
    pub fpset_backend: Option<RequestedFpsetBackend>,
}

const MCC_FPSET_BACKEND_ENV: &str = "TY_MCC_FPSET_BACKEND";

impl Default for PetriRunArgs {
    fn default() -> Self {
        Self {
            timeout: None,
            max_states: None,
            memory_fraction: 0.25,
            threads: 1,
            storage: RequestedStorageMode::Auto,
            storage_dir: None,
            checkpoint_dir: None,
            checkpoint_interval_states: 100_000,
            resume_checkpoint: false,
            fpset_backend: None,
        }
    }
}

/// User-facing storage selection before it is mapped onto explorer backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RequestedStorageMode {
    /// Choose the backend automatically from the state limit and memory budget.
    Auto,
    /// Keep the full visited-state set in RAM.
    Memory,
    /// Back the visited-state set with a memory-mapped file.
    Mmap,
    /// Back the visited-state set with on-disk storage.
    Disk,
    /// Fingerprint-only BFS: 8 bytes/state via lock-free CAS table.
    FingerprintOnly,
}

/// User-facing fingerprint set backend selection before it is mapped onto
/// the internal [`FpsetBackend`] configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RequestedFpsetBackend {
    /// RwLock-based sharded fingerprint sets (default).
    Sharded,
    /// Lock-free CAS-based open addressing (16 partitions).
    ///
    /// Eliminates RwLock contention that caps sharded throughput at ~4 workers.
    Cas,
}

/// Which top-level command is invoking the shared Petri runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PetriCommandMode {
    /// Direct `ty petri` invocation with explicit inputs.
    Petri,
    /// MCC-style invocation (`ty mcc` or legacy `pnml-tools`).
    Mcc,
}

impl PetriCommandMode {
    fn requires_model_input(self) -> bool {
        matches!(self, Self::Petri)
    }

    fn requires_explicit_examination(self) -> bool {
        matches!(self, Self::Petri)
    }
}

/// Parse the shared memory-fraction flag.
pub fn parse_memory_fraction(raw: &str) -> Result<f64, String> {
    let fraction = raw
        .parse::<f64>()
        .map_err(|err| format!("invalid memory fraction '{raw}': {err}"))?;
    if !(fraction > 0.0 && fraction <= 1.0) {
        return Err(String::from("memory-fraction must be > 0 and <= 1.0"));
    }
    Ok(fraction)
}

/// Parse shared positive-integer flags that must be at least 1.
pub fn parse_positive_usize(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid positive integer '{raw}': {err}"))?;
    if value == 0 {
        return Err(String::from("value must be >= 1"));
    }
    Ok(value)
}

/// Normalize the state limit option: `0` means auto-size.
#[must_use]
pub fn resolve_max_states(max_states: Option<usize>) -> Option<usize> {
    max_states.filter(|&limit| limit != 0)
}

fn auto_size_message(
    num_places: usize,
    packed_places: usize,
    packed_bytes: usize,
    bytes_per_place: usize,
    primary_capacity: usize,
    storage_mode: StorageMode,
    unbounded: bool,
) -> String {
    let bytes_per_state = packed_bytes + 48;
    let limit_suffix = if unbounded {
        String::from("state_limit=unbounded")
    } else {
        format!("max_states={primary_capacity}")
    };
    if packed_places == num_places {
        format!(
            "auto-sized ({storage_mode:?}): {} places × {}B/place = {}B/state → primary_capacity={} {}",
            num_places, bytes_per_place, bytes_per_state, primary_capacity, limit_suffix,
        )
    } else {
        format!(
            "auto-sized ({storage_mode:?}): {} places ({} packed, {}B exact) × {}B/place max = {}B/state → primary_capacity={} {}",
            num_places, packed_places, packed_bytes, bytes_per_place, bytes_per_state, primary_capacity, limit_suffix,
        )
    }
}

/// Maps a user-facing [`RequestedFpsetBackend`] onto the internal explorer
/// [`FpsetBackend`].
#[must_use]
pub fn resolve_fpset_backend(requested: RequestedFpsetBackend) -> FpsetBackend {
    match requested {
        RequestedFpsetBackend::Sharded => FpsetBackend::Sharded,
        RequestedFpsetBackend::Cas => FpsetBackend::Cas,
    }
}

fn parse_requested_fpset_backend(raw: &str) -> Result<RequestedFpsetBackend, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "sharded" => Ok(RequestedFpsetBackend::Sharded),
        "cas" => Ok(RequestedFpsetBackend::Cas),
        other => Err(format!(
            "invalid {MCC_FPSET_BACKEND_ENV} value '{other}'; expected 'sharded' or 'cas'"
        )),
    }
}

/// Resolve the fingerprint set backend from CLI/env.
///
/// Precedence (highest first):
/// 1. Explicit `--fpset-backend cas|sharded` on the CLI.
/// 2. `TY_MCC_FPSET_BACKEND=cas|sharded` env var.
/// 3. Exactness-first default: `Sharded` for all worker counts.
///
/// `Cas` remains available for explicit and guarded storage paths, but the
/// exact observer dispatcher refuses unguarded folded-fingerprint CAS until it
/// has a payload collision guard.
fn resolve_requested_fpset_backend(
    requested: Option<RequestedFpsetBackend>,
    _workers: usize,
) -> Result<RequestedFpsetBackend> {
    if let Some(requested) = requested {
        return Ok(requested);
    }
    match read_trimmed_env_var(MCC_FPSET_BACKEND_ENV) {
        Some(value) => parse_requested_fpset_backend(&value).map_err(anyhow::Error::msg),
        None => Ok(default_fpset_backend_for_workers(_workers)),
    }
}

/// Pick the auto-default fingerprint set backend for a given worker count.
///
/// Returns the guarded `Sharded` backend by default. `Cas` is still accepted
/// as an explicit request, but exact observer dispatch refuses it until the
/// folded-fingerprint path has a payload collision guard.
#[must_use]
pub(crate) fn default_fpset_backend_for_workers(_workers: usize) -> RequestedFpsetBackend {
    RequestedFpsetBackend::Sharded
}

/// Resolves the concrete explorer [`StorageMode`] from the requested mode.
///
/// Non-`Auto` requests map directly. For `Auto`, the choice depends on
/// `explicit_limit` relative to `auto_budget` (the memory-derived state
/// budget): within budget keeps state in memory, up to ~4× spills to mmap,
/// beyond that goes to disk, and an absent limit selects fingerprint-only BFS.
#[must_use]
pub fn resolve_storage_mode(
    requested: RequestedStorageMode,
    explicit_limit: Option<usize>,
    auto_budget: usize,
) -> StorageMode {
    match requested {
        RequestedStorageMode::Memory => StorageMode::Memory,
        RequestedStorageMode::Mmap => StorageMode::Mmap,
        RequestedStorageMode::Disk => StorageMode::Disk,
        RequestedStorageMode::FingerprintOnly => StorageMode::FingerprintOnly,
        RequestedStorageMode::Auto => match explicit_limit {
            Some(limit) if limit <= auto_budget => StorageMode::Memory,
            Some(limit) if limit <= auto_budget.saturating_mul(4).max(auto_budget) => {
                StorageMode::Mmap
            }
            Some(_) => StorageMode::Disk,
            None => StorageMode::FingerprintOnly,
        },
    }
}

/// Whether the explorer can checkpoint/resume the given examination.
///
/// Only the BFS-over-full-reachable-set examinations support checkpointing;
/// property and bounds examinations are not resumable.
#[must_use]
pub fn checkpoint_supported(examination: Examination) -> bool {
    matches!(
        examination,
        Examination::ReachabilityDeadlock
            | Examination::OneSafe
            | Examination::QuasiLiveness
            | Examination::StableMarking
            | Examination::StateSpace
    )
}

fn read_trimmed_env_var(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn env_flag_setting(key: &str) -> Option<bool> {
    read_trimmed_env_var(key).map(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn model_dir_has_default_ctl_route(model_dir: &Path) -> bool {
    match crate::parser::parse_pnml_dir(model_dir) {
        Ok(_) => true,
        Err(crate::error::PnmlError::UnsupportedNetType { net_type }) => {
            let supported_colored = net_type.contains("symmetricnet");
            if supported_colored {
                eprintln!(
                    "CTL: MCC routing colored PNML net type `{net_type}` through uncollapsed source at {}",
                    model_dir.display()
                );
            } else {
                eprintln!(
                    "CTL: MCC fail-closed for unsupported PNML net type `{net_type}` at {}",
                    model_dir.display()
                );
            }
            supported_colored
        }
        Err(error) => {
            eprintln!(
                "CTL: MCC fail-closed because P/T PNML parsing failed at {}: {error}",
                model_dir.display()
            );
            false
        }
    }
}

fn should_fail_closed_ctl(
    mode: PetriCommandMode,
    examination: Examination,
    model_dir: &Path,
) -> bool {
    if !matches!(mode, PetriCommandMode::Mcc)
        || !matches!(
            examination,
            Examination::CTLCardinality | Examination::CTLFireability
        )
    {
        return false;
    }

    match env_flag_setting("TY_MCC_ENABLE_CTL") {
        Some(enabled) => !enabled,
        None => !model_dir_has_default_ctl_route(model_dir),
    }
}

fn ctl_cannot_compute_lines(model_dir: &Path, examination: Examination) -> Vec<String> {
    match crate::property_xml::parse_properties(model_dir, examination.as_str()) {
        Ok(properties) => properties
            .into_iter()
            .map(|property| formula_line("", &property.id, Verdict::CannotCompute))
            .collect(),
        Err(error) => {
            eprintln!(
                "Warning: failed to parse {}.xml while failing closed: {error}",
                examination.as_str()
            );
            match crate::property_xml::parse_property_ids(model_dir, examination.as_str()) {
                Ok(ids) => ids
                    .into_iter()
                    .map(|id| formula_line("", &id, Verdict::CannotCompute))
                    .collect(),
                Err(id_error) => {
                    eprintln!(
                        "Warning: failed to recover property ids from {}.xml while failing closed: {id_error}",
                        examination.as_str()
                    );
                    vec![cannot_compute_line("", examination.as_str())]
                }
            }
        }
    }
}

/// Build the fail-closed `CANNOT_COMPUTE` line(s) the watchdog emits for an
/// examination that ran away past its hard deadline.
///
/// The format must be IDENTICAL to what each examination emits for a normal
/// CANNOT_COMPUTE, so a forced watchdog CC is indistinguishable from a real one
/// to the MCC grader. The match is exhaustive over [`Examination`]; any
/// unforeseen variant defaults to a single `cannot_compute_line` (fail-closed).
fn cannot_compute_fallback_lines(model_dir: &Path, examination: Examination) -> Vec<String> {
    match examination {
        // Multi-formula examinations: one per-formula CANNOT_COMPUTE line per
        // property id parsed from `<E>.xml`. Mirrors how the normal property
        // pipelines emit a `FORMULA <id> CANNOT_COMPUTE TECHNIQUES ...` row per
        // property; see `ctl_cannot_compute_lines` (the existing generalization
        // target) and `ExaminationRecord::to_mcc_line`.
        Examination::CTLCardinality
        | Examination::CTLFireability
        | Examination::ReachabilityCardinality
        | Examination::ReachabilityFireability
        | Examination::LTLCardinality
        | Examination::LTLFireability => ctl_cannot_compute_lines(model_dir, examination),

        // StateSpace: the single `STATE_SPACE CANNOT_COMPUTE TECHNIQUES ...`
        // line, exactly as `ExaminationRecord::to_mcc_line` emits for
        // `ExaminationValue::StateSpace(None)`.
        Examination::StateSpace => {
            vec![crate::output::state_space_cannot_compute_line(
                &crate::output::Techniques::default(),
            )]
        }

        // Single-verdict examinations: one `FORMULA <E> CANNOT_COMPUTE ...`
        // line keyed on the examination name. `cannot_compute_line` routes
        // StateSpace to the STATE_SPACE form, but StateSpace is handled above,
        // so every variant here emits the FORMULA form. `UpperBounds` is a
        // property examination but its CC fallback is the single per-formula
        // line when no ids are available; emit the examination-keyed CC line.
        Examination::ReachabilityDeadlock
        | Examination::OneSafe
        | Examination::QuasiLiveness
        | Examination::StableMarking
        | Examination::UpperBounds
        | Examination::Liveness => {
            vec![cannot_compute_line("", examination.as_str())]
        }
    }
}

/// EMIT-ONCE GATE decision for the hard watchdog (pure; testable without
/// `process::exit`). Given whether the worker has already emitted any real line
/// (`output_started`) and the fallback lines it WOULD emit, return the lines the
/// watchdog should actually print:
///
/// - `output_started == false` (atomic StateSpace-style runaway: worker stuck
///   before its single emission) ⇒ emit the full fallback (runaway guarantee).
/// - `output_started == true` (multi-formula examination already flushed
///   ≥1 verdict via `print_mcc_line`) ⇒ emit NOTHING; the flushed verdicts stand
///   and not-yet-decided formulas are absent (= MCC no-answer), never duplicated.
fn watchdog_fallback_to_emit(output_started: bool, fallback_lines: Vec<String>) -> Vec<String> {
    if output_started {
        Vec::new()
    } else {
        fallback_lines
    }
}

fn print_ctl_cannot_compute(model_dir: &Path, examination: Examination) -> Result<()> {
    eprintln!(
        "{}: disabled for MCC soundness on this input; set TY_MCC_ENABLE_CTL=1 to force CTL",
        examination.as_str()
    );
    for line in ctl_cannot_compute_lines(model_dir, examination) {
        print_mcc_line(line);
    }
    Ok(())
}

fn normalize_model_dir(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let treat_as_file = path.is_file()
        || path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pnml"));
    if !treat_as_file {
        return Ok(path.to_path_buf());
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if parent.as_os_str().is_empty() {
        Ok(PathBuf::from("."))
    } else {
        Ok(parent.to_path_buf())
    }
}

fn resolve_model_dir(mode: PetriCommandMode, model_input: Option<PathBuf>) -> Result<PathBuf> {
    match model_input {
        Some(path) => normalize_model_dir(&path),
        None if mode.requires_model_input() => {
            bail!("model path is required for `ty petri`");
        }
        None => read_trimmed_env_var("BK_INPUT")
            .map(|value| normalize_model_dir(Path::new(&value)))
            .transpose()
            .map(|path| path.unwrap_or_else(|| PathBuf::from("."))),
    }
}

fn resolve_examination(
    mode: PetriCommandMode,
    examination_name: Option<String>,
) -> Result<Examination> {
    let exam_name = examination_name
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .or_else(|| read_trimmed_env_var("BK_EXAMINATION"));
    let Some(exam_name) = exam_name else {
        if mode.requires_explicit_examination() {
            bail!("`ty petri` requires --examination <NAME>");
        }
        bail!("examination not specified (use --examination or set BK_EXAMINATION)");
    };
    Ok(Examination::from_name(&exam_name)?)
}

fn validate_run_args(args: &PetriRunArgs) -> Result<()> {
    if args.threads == 0 {
        bail!("--threads must be >= 1");
    }
    if args.checkpoint_interval_states == 0 {
        bail!("--checkpoint-interval-states must be >= 1");
    }
    Ok(())
}

/// Build the shared Petri/MCC exploration configuration for a prepared model.
///
/// This is the same resolver used by `ty mcc`, `ty petri`, and the legacy
/// `pnml-tools` wrapper. Non-CLI MCC callers should use this instead of
/// constructing [`ExplorationConfig`] directly when they want the competition
/// defaults for storage, fingerprint admission, worker-aware CAS selection,
/// checkpointing, and deadlines.
pub fn build_exploration_config(
    model: &crate::model::PreparedModel,
    args: &PetriRunArgs,
) -> Result<ExplorationConfig> {
    validate_run_args(args)?;
    let deadline = crate::timeout::compute_deadline(args.timeout);
    let workers = args.threads;
    let explicit_limit = resolve_max_states(args.max_states);
    let info = ExplorationConfig::describe_auto(model.net(), Some(args.memory_fraction));
    let storage_mode = resolve_storage_mode(args.storage, explicit_limit, info.max_states);

    let mut config = if let Some(max_states) = explicit_limit {
        let base = ExplorationConfig::new(max_states)
            .with_deadline(deadline)
            .with_workers(workers)
            .with_storage_mode(storage_mode)
            .with_storage_dir(args.storage_dir.clone());
        if storage_mode == StorageMode::Memory {
            base
        } else {
            base.with_storage_primary_capacity(info.max_states.min(max_states).max(1))
        }
    } else if storage_mode == StorageMode::Memory {
        eprintln!(
            "{}",
            auto_size_message(
                info.num_places,
                info.packed_places,
                info.packed_bytes,
                info.bytes_per_place,
                info.max_states,
                storage_mode,
                false,
            )
        );
        ExplorationConfig::auto_sized(model.net(), deadline, Some(args.memory_fraction))
            .with_workers(workers)
            .with_storage_mode(storage_mode)
            .with_storage_dir(args.storage_dir.clone())
    } else {
        eprintln!(
            "{}",
            auto_size_message(
                info.num_places,
                info.packed_places,
                info.packed_bytes,
                info.bytes_per_place,
                info.max_states,
                storage_mode,
                true,
            )
        );
        ExplorationConfig::new(usize::MAX)
            .with_deadline(deadline)
            .with_workers(workers)
            .with_storage_mode(storage_mode)
            .with_storage_primary_capacity(info.max_states)
            .with_full_graph_auto_sizing(model.net(), Some(args.memory_fraction))
            .with_storage_dir(args.storage_dir.clone())
    };

    if let Some(dir) = args.checkpoint_dir.clone() {
        config = config.with_checkpoint(
            CheckpointConfig::new(dir, args.checkpoint_interval_states)
                .with_resume(args.resume_checkpoint),
        );
    }

    config = config.with_fpset_backend(resolve_fpset_backend(resolve_requested_fpset_backend(
        args.fpset_backend,
        workers,
    )?));

    Ok(config)
}

/// Run Petri/MCC execution through the shared `tla-petri` runtime.
pub fn run_cli(
    mode: PetriCommandMode,
    model_input: Option<PathBuf>,
    examination_name: Option<String>,
    args: PetriRunArgs,
) -> Result<()> {
    let run_started_at = Instant::now();
    if args.resume_checkpoint && args.checkpoint_dir.is_none() {
        bail!("--resume-checkpoint requires --checkpoint-dir");
    }

    let model_dir = resolve_model_dir(mode, model_input)?;
    let examination = resolve_examination(mode, examination_name)?;
    if should_fail_closed_ctl(mode, examination, &model_dir) {
        return print_ctl_cannot_compute(&model_dir, examination);
    }
    if args.checkpoint_dir.is_some() && !checkpoint_supported(examination) {
        bail!(
            "checkpoint/resume currently supports ReachabilityDeadlock, OneSafe, QuasiLiveness, StableMarking, and StateSpace"
        );
    }
    if args.resume_checkpoint {
        let dir = args
            .checkpoint_dir
            .as_ref()
            .expect("checkpoint dir validated above");
        if !dir.join("checkpoint.json").exists() {
            bail!(
                "resume requested but no checkpoint found at {}",
                dir.display()
            );
        }
    }

    let source_load_started = Instant::now();
    let model = match crate::model::load_model_dir(&model_dir) {
        Ok(model) => model,
        Err(PnmlError::UnsupportedNetType { .. }) => {
            let model_name = model_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            crate::output::print_mcc_line(crate::output::cannot_compute_line(
                model_name,
                examination.as_str(),
            ));
            return Ok(());
        }
        Err(error)
            if matches!(mode, PetriCommandMode::Mcc)
                && matches!(
                    examination,
                    Examination::CTLCardinality | Examination::CTLFireability
                ) =>
        {
            eprintln!(
                "{}: model load failed while CTL was default-enabled ({error}); failing closed",
                examination.as_str()
            );
            return print_ctl_cannot_compute(&model_dir, examination);
        }
        Err(error) => return Err(error.into()),
    };
    let source_load_duration = source_load_started.elapsed();

    let config_started = Instant::now();
    let config = build_exploration_config(&model, &args)?.with_examination(Some(examination));
    let config_resolution_duration = config_started.elapsed();
    let setup_evidence = crate::mcc_backend_evidence::build_mcc_setup_evidence(
        &model,
        examination,
        &config,
        run_started_at,
        source_load_duration,
        config_resolution_duration,
    );
    // Hard wall-clock watchdog around the in-process examination.
    //
    // The cooperative deadline (budget - 5s) is polled only by code paths that
    // bother to check it; any tight loop, single long symbolic op, or detached
    // backend thread that ignores it can run forever. This watchdog is the hard
    // backstop: at budget - 3s (the hard fire point) plus a 2s grace recv (so
    // the absolute kill lands by budget - 1s, <= budget) it forces a fail-closed
    // CANNOT_COMPUTE and exits the process so it can NEVER run away.
    //
    // No-double-emit invariant (two complementary guards):
    //   1. EMIT-ONCE GATE: the on_timeout fallback fires ONLY when the worker
    //      has emitted NOTHING (`output_started() == false`). A multi-formula
    //      examination flushes each verdict incrementally via print_mcc_line
    //      (which sets OUTPUT_STARTED), so once it has produced >=1 line the
    //      watchdog stays silent — no duplicate FORMULA lines.
    //   2. GRACE RECV: a worker finishing right at the deadline gets a short
    //      grace window to send its Completed signal (see run_with_hard_deadline)
    //      so it is never needlessly killed mid-flush.
    // The genuine StateSpace-style atomic runaway (stuck before its single
    // emission, OUTPUT_STARTED still false) still triggers the fallback — the
    // runaway guarantee is preserved.
    //
    // A forced CANNOT_COMPUTE is ALWAYS sound: it is never a numeric/boolean
    // verdict, so it can never be a wrong or partial answer.
    match crate::timeout::compute_hard_deadline(args.timeout) {
        Some(hard) => {
            // Build the fallback lines FIRST, while `model_dir` is still owned by
            // this frame (it is NOT moved into the worker; only `setup_evidence`
            // is moved, and `&model`/`&config` are borrowed by the scoped worker).
            // `fallback_lines` is moved into the `on_timeout` closure.
            let fallback_lines = cannot_compute_fallback_lines(&model_dir, examination);
            let outcome = crate::timeout::run_with_hard_deadline(
                hard,
                move || {
                    maybe_force_hang_for_tests();
                    crate::model::run_examination_for_model_with_setup_evidence(
                        &model,
                        examination,
                        &config,
                        setup_evidence,
                    );
                },
                move || {
                    // Runaway. EMIT-ONCE GATE: emit the fail-closed CC fallback
                    // ONLY if the worker has emitted NOTHING yet
                    // (`!output_started()`). This is the genuine StateSpace-style
                    // atomic runaway — the single-verdict worker is stuck before
                    // its one emission. In that case the forced CC is the only
                    // output and is ALWAYS sound (never a wrong/partial verdict).
                    //
                    // If the worker already incrementally flushed >=1 line (every
                    // multi-formula examination flushes each verdict via
                    // `print_mcc_line`, which sets OUTPUT_STARTED), the watchdog
                    // emits NOTHING: those real verdicts stand, and any
                    // not-yet-decided formulas are simply ABSENT — which MCC
                    // treats as no-answer (= CANNOT_COMPUTE), NOT a malformed
                    // duplicate FORMULA line. This is what fixes the
                    // Philosophers-PT-000050 double-emit.
                    //
                    // Either way we `process::exit(0)` to kill the stuck worker
                    // thread (still burning CPU). The exit fires BEFORE the
                    // scope's join of that worker, so we never block on the join.
                    for line in
                        watchdog_fallback_to_emit(crate::output::output_started(), fallback_lines)
                    {
                        print_mcc_line(line);
                    }
                    let _ = std::io::stdout().flush();
                    std::process::exit(0);
                },
            );
            match outcome {
                // The TimedOut arm is effectively unreachable in production
                // because `on_timeout` above `process::exit`s, but we handle it
                // exhaustively (and it is the path the unit tests exercise).
                crate::timeout::RunOutcome::Completed | crate::timeout::RunOutcome::TimedOut => {
                    Ok(())
                }
            }
        }
        None => {
            // No timeout / BK_TIME_CONFINEMENT: run inline exactly as before
            // (zero behavior change when no budget is set).
            crate::model::run_examination_for_model_with_setup_evidence(
                &model,
                examination,
                &config,
                setup_evidence,
            );
            Ok(())
        }
    }
}

/// Test-only hang hook: when `TY_TEST_FORCE_HANG` is set, sleep forever inside
/// the worker so an end-to-end run can be driven into the hard watchdog. Has
/// ZERO effect unless the env var is present, so production runs are unaffected.
fn maybe_force_hang_for_tests() {
    if std::env::var_os("TY_TEST_FORCE_HANG").is_some() {
        eprintln!("TY_TEST_FORCE_HANG set — worker hanging to exercise the hard watchdog");
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvVarGuard<'a> {
        key: &'a str,
        prev: Option<String>,
    }

    impl<'a> EnvVarGuard<'a> {
        fn set(key: &'a str, value: Option<&str>) -> Self {
            let prev = std::env::var(key).ok();
            match value {
                Some(value) => crate::env_guard::set_var(key, value),
                None => crate::env_guard::remove_var(key),
            }
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard<'_> {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                crate::env_guard::set_var(self.key, prev);
            } else {
                crate::env_guard::remove_var(self.key);
            }
        }
    }

    fn with_env_var<T>(key: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        // Single crate-wide env lock: BK_INPUT/BK_EXAMINATION mutations here must
        // not race any other module's env-touching test.
        let _lock = crate::env_test_lock();
        let _guard = EnvVarGuard::set(key, value);
        f()
    }

    fn write_minimal_pt_model(dir: &Path) {
        std::fs::write(
            dir.join("model.pnml"),
            r#"<?xml version="1.0" encoding="UTF-8"?>
<pnml>
  <net id="n0" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="page0">
      <place id="p0"><initialMarking><text>1</text></initialMarking></place>
      <transition id="t0"/>
    </page>
  </net>
</pnml>
"#,
        )
        .expect("minimal P/T PNML should be written");
    }

    fn write_minimal_colored_marker_model(dir: &Path) {
        std::fs::write(
            dir.join("model.pnml"),
            r#"<?xml version="1.0" encoding="UTF-8"?>
<pnml>
  <net id="n0" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0"/>
  </net>
</pnml>
"#,
        )
        .expect("minimal colored PNML marker should be written");
    }

    #[test]
    fn mcc_ctl_examinations_default_enable_supported_pt_and_colored_inputs() {
        with_env_var("TY_MCC_ENABLE_CTL", None, || {
            let mounted_pt = tempfile::tempdir().expect("tempdir should be created");
            write_minimal_pt_model(mounted_pt.path());
            let misnamed_pt = tempfile::Builder::new()
                .prefix("Sudoku-COL-AN01")
                .tempdir()
                .expect("COL-looking tempdir should be created");
            write_minimal_pt_model(misnamed_pt.path());
            let actual_col = tempfile::Builder::new()
                .prefix("Sudoku-COL-AN01")
                .tempdir()
                .expect("COL-looking tempdir should be created");
            write_minimal_colored_marker_model(actual_col.path());

            assert!(should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::CTLCardinality,
                Path::new("Sudoku-PT-AN01")
            ));
            assert!(!should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::CTLCardinality,
                mounted_pt.path()
            ));
            assert!(!should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::CTLCardinality,
                misnamed_pt.path()
            ));
            assert!(!should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::CTLCardinality,
                actual_col.path()
            ));
            assert!(!should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::CTLFireability,
                mounted_pt.path()
            ));
            assert!(should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::CTLCardinality,
                Path::new("Sudoku-COL-AN01")
            ));
            assert!(should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::CTLFireability,
                Path::new("unknown")
            ));
            assert!(!should_fail_closed_ctl(
                PetriCommandMode::Petri,
                Examination::CTLCardinality,
                Path::new("Sudoku-COL-AN01")
            ));
            assert!(!should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::ReachabilityDeadlock,
                Path::new("Sudoku-COL-AN01")
            ));
        });

        with_env_var("TY_MCC_ENABLE_CTL", Some("0"), || {
            assert!(should_fail_closed_ctl(
                PetriCommandMode::Mcc,
                Examination::CTLCardinality,
                Path::new("Sudoku-PT-AN01")
            ));
        });
    }

    #[test]
    fn mcc_ctl_examinations_run_when_explicitly_enabled() {
        for value in ["1", "true"] {
            with_env_var("TY_MCC_ENABLE_CTL", Some(value), || {
                assert!(!should_fail_closed_ctl(
                    PetriCommandMode::Mcc,
                    Examination::CTLCardinality,
                    Path::new("Sudoku-COL-AN01")
                ));
                assert!(!should_fail_closed_ctl(
                    PetriCommandMode::Mcc,
                    Examination::CTLFireability,
                    Path::new("unknown")
                ));
            });
        }
    }

    #[test]
    fn ctl_cannot_compute_lines_fall_back_when_xml_is_missing() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let lines = ctl_cannot_compute_lines(tempdir.path(), Examination::CTLFireability);
        assert_eq!(
            lines,
            vec!["FORMULA CTLFireability CANNOT_COMPUTE TECHNIQUES EXPLICIT"]
        );
    }

    #[test]
    fn ctl_cannot_compute_lines_recover_ids_when_formula_parse_fails() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        std::fs::write(
            tempdir.path().join("CTLFireability.xml"),
            r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>Model-CTLFireability-00</id>
    <formula><unsupported-ctl /></formula>
  </property>
  <property>
    <id>Model-CTLFireability-01</id>
    <formula><unsupported-ctl /></formula>
  </property>
</property-set>"#,
        )
        .expect("CTL xml should write");

        let lines = ctl_cannot_compute_lines(tempdir.path(), Examination::CTLFireability);
        assert_eq!(
            lines,
            vec![
                "FORMULA Model-CTLFireability-00 CANNOT_COMPUTE TECHNIQUES EXPLICIT",
                "FORMULA Model-CTLFireability-01 CANNOT_COMPUTE TECHNIQUES EXPLICIT",
            ]
        );
    }

    #[test]
    fn fallback_lines_state_space_emits_state_space_cc_line() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let lines = cannot_compute_fallback_lines(tempdir.path(), Examination::StateSpace);
        assert_eq!(
            lines,
            vec![crate::output::state_space_cannot_compute_line(
                &crate::output::Techniques::default()
            )]
        );
        // Sanity: this is the STATE_SPACE CC shape the grader's ty_statespace reads.
        assert_eq!(
            lines,
            vec!["STATE_SPACE CANNOT_COMPUTE TECHNIQUES EXPLICIT"]
        );
    }

    #[test]
    fn fallback_lines_multi_formula_emits_one_cc_per_property_id() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        std::fs::write(
            tempdir.path().join("ReachabilityCardinality.xml"),
            r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>Model-ReachabilityCardinality-00</id>
    <formula><unsupported-reach /></formula>
  </property>
  <property>
    <id>Model-ReachabilityCardinality-01</id>
    <formula><unsupported-reach /></formula>
  </property>
</property-set>"#,
        )
        .expect("reachability xml should write");

        let lines =
            cannot_compute_fallback_lines(tempdir.path(), Examination::ReachabilityCardinality);
        assert_eq!(
            lines,
            vec![
                "FORMULA Model-ReachabilityCardinality-00 CANNOT_COMPUTE TECHNIQUES EXPLICIT",
                "FORMULA Model-ReachabilityCardinality-01 CANNOT_COMPUTE TECHNIQUES EXPLICIT",
            ]
        );
    }

    #[test]
    fn fallback_lines_multi_formula_falls_back_when_xml_missing() {
        // No <E>.xml present: emit a single examination-keyed CC line.
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let lines = cannot_compute_fallback_lines(tempdir.path(), Examination::LTLFireability);
        assert_eq!(
            lines,
            vec!["FORMULA LTLFireability CANNOT_COMPUTE TECHNIQUES EXPLICIT"]
        );
    }

    #[test]
    fn fallback_lines_single_verdict_emits_one_cc_line() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        // ReachabilityDeadlock is a single-verdict GlobalProperty.
        let lines =
            cannot_compute_fallback_lines(tempdir.path(), Examination::ReachabilityDeadlock);
        assert_eq!(
            lines,
            vec!["FORMULA ReachabilityDeadlock CANNOT_COMPUTE TECHNIQUES EXPLICIT"]
        );
        // QuasiLiveness is another single-verdict GlobalProperty.
        let lines = cannot_compute_fallback_lines(tempdir.path(), Examination::QuasiLiveness);
        assert_eq!(
            lines,
            vec!["FORMULA QuasiLiveness CANNOT_COMPUTE TECHNIQUES EXPLICIT"]
        );
    }

    #[test]
    fn max_states_zero_means_auto_sizing() {
        assert_eq!(resolve_max_states(Some(0)), None);
    }

    #[test]
    fn max_states_nonzero_stays_explicit() {
        assert_eq!(resolve_max_states(Some(123)), Some(123));
    }

    #[test]
    fn parse_memory_fraction_accepts_valid_value() {
        assert_eq!(parse_memory_fraction("0.25").unwrap(), 0.25);
    }

    #[test]
    fn parse_memory_fraction_rejects_zero() {
        let err = parse_memory_fraction("0").expect_err("zero should be rejected");
        assert!(err.contains("memory-fraction must be > 0 and <= 1.0"));
    }

    #[test]
    fn parse_memory_fraction_rejects_above_one() {
        let err = parse_memory_fraction("1.5").expect_err("values above one should fail");
        assert!(err.contains("memory-fraction must be > 0 and <= 1.0"));
    }

    #[test]
    fn parse_positive_usize_accepts_nonzero() {
        assert_eq!(parse_positive_usize("4").unwrap(), 4);
    }

    #[test]
    fn parse_positive_usize_rejects_zero() {
        let err = parse_positive_usize("0").expect_err("zero should be rejected");
        assert!(err.contains(">= 1"));
    }

    #[test]
    fn resolve_storage_mode_prefers_fingerprint_only_for_unbounded_auto_runs() {
        assert_eq!(
            resolve_storage_mode(RequestedStorageMode::Auto, None, 100),
            StorageMode::FingerprintOnly
        );
    }

    #[test]
    fn normalize_model_dir_keeps_directories() {
        assert_eq!(
            normalize_model_dir(Path::new("benchmarks/mcc/2024/INPUTS")).unwrap(),
            PathBuf::from("benchmarks/mcc/2024/INPUTS")
        );
    }

    #[test]
    fn normalize_model_dir_maps_model_file_to_parent_directory() {
        assert_eq!(
            normalize_model_dir(Path::new("benchmarks/mcc/2024/INPUTS/model.pnml")).unwrap(),
            PathBuf::from("benchmarks/mcc/2024/INPUTS")
        );
    }

    #[test]
    fn normalize_model_dir_maps_bare_model_file_to_current_dir() {
        assert_eq!(
            normalize_model_dir(Path::new("model.pnml")).unwrap(),
            PathBuf::from(".")
        );
    }

    #[test]
    fn resolve_model_dir_requires_input_for_petri_mode() {
        let err = resolve_model_dir(PetriCommandMode::Petri, None)
            .expect_err("petri mode should require an explicit model path");
        assert!(err.to_string().contains("model path is required"));
    }

    #[test]
    fn resolve_examination_requires_explicit_name_for_petri_mode() {
        let err = with_env_var("BK_EXAMINATION", None, || {
            resolve_examination(PetriCommandMode::Petri, None)
                .expect_err("petri mode should require an explicit examination")
        });
        assert!(err.to_string().contains("requires --examination"));
    }

    #[test]
    fn resolve_model_dir_uses_bk_input_for_mcc_mode() {
        let model_dir = with_env_var(
            "BK_INPUT",
            Some("benchmarks/mcc/2024/INPUTS/TokenRing-PT-010"),
            || {
                resolve_model_dir(PetriCommandMode::Mcc, None)
                    .expect("mcc mode should use BK_INPUT")
            },
        );
        assert_eq!(
            model_dir,
            PathBuf::from("benchmarks/mcc/2024/INPUTS/TokenRing-PT-010")
        );
    }

    #[test]
    fn resolve_model_dir_normalizes_bk_input_model_file_for_mcc_mode() {
        let model_dir = with_env_var(
            "BK_INPUT",
            Some("  benchmarks/mcc/2024/INPUTS/TokenRing-PT-010/model.pnml  "),
            || {
                resolve_model_dir(PetriCommandMode::Mcc, None)
                    .expect("mcc mode should normalize file-style BK_INPUT")
            },
        );
        assert_eq!(
            model_dir,
            PathBuf::from("benchmarks/mcc/2024/INPUTS/TokenRing-PT-010")
        );
    }

    #[test]
    fn resolve_model_dir_defaults_to_current_dir_for_mcc_mode() {
        let model_dir = with_env_var("BK_INPUT", None, || {
            resolve_model_dir(PetriCommandMode::Mcc, None)
                .expect("mcc mode should default to current dir")
        });
        assert_eq!(model_dir, PathBuf::from("."));
    }

    #[test]
    fn resolve_model_dir_treats_empty_bk_input_as_current_dir() {
        let model_dir = with_env_var("BK_INPUT", Some("   "), || {
            resolve_model_dir(PetriCommandMode::Mcc, None)
                .expect("blank BK_INPUT should fall back to current dir")
        });
        assert_eq!(model_dir, PathBuf::from("."));
    }

    #[test]
    fn resolve_examination_uses_bk_examination_for_mcc_mode() {
        let examination = with_env_var("BK_EXAMINATION", Some("  ReachabilityDeadlock  "), || {
            resolve_examination(PetriCommandMode::Mcc, None)
                .expect("mcc mode should use BK_EXAMINATION")
        });
        assert_eq!(examination, Examination::ReachabilityDeadlock);
    }

    #[test]
    fn resolve_examination_ignores_blank_bk_examination_in_mcc_mode() {
        let err = with_env_var("BK_EXAMINATION", Some("   "), || {
            resolve_examination(PetriCommandMode::Mcc, None)
                .expect_err("blank BK_EXAMINATION should be treated as missing")
        });
        assert!(err.to_string().contains("examination not specified"));
    }

    #[test]
    fn resolve_examination_rejects_blank_cli_name_for_petri_mode() {
        let err = resolve_examination(PetriCommandMode::Petri, Some("   ".to_string()))
            .expect_err("blank CLI examination should be treated as missing");
        assert!(err.to_string().contains("requires --examination"));
    }

    #[test]
    fn validate_run_args_rejects_zero_threads() {
        let err = validate_run_args(&PetriRunArgs {
            threads: 0,
            fpset_backend: Some(RequestedFpsetBackend::Sharded),
            ..PetriRunArgs::default()
        })
        .expect_err("zero threads should be rejected");
        assert!(err.to_string().contains("--threads must be >= 1"));
    }

    #[test]
    fn validate_run_args_rejects_zero_checkpoint_interval() {
        let err = validate_run_args(&PetriRunArgs {
            checkpoint_dir: Some(PathBuf::from("checkpoint")),
            checkpoint_interval_states: 0,
            fpset_backend: Some(RequestedFpsetBackend::Sharded),
            ..PetriRunArgs::default()
        })
        .expect_err("zero checkpoint interval should be rejected");
        assert!(err
            .to_string()
            .contains("--checkpoint-interval-states must be >= 1"));
    }

    #[test]
    fn resolve_fpset_backend_sharded() {
        assert_eq!(
            resolve_fpset_backend(RequestedFpsetBackend::Sharded),
            FpsetBackend::Sharded,
        );
    }

    #[test]
    fn resolve_fpset_backend_cas() {
        assert_eq!(
            resolve_fpset_backend(RequestedFpsetBackend::Cas),
            FpsetBackend::Cas,
        );
    }

    #[test]
    fn resolve_requested_fpset_backend_defaults_to_sharded_without_env_sequential() {
        let resolved = with_env_var(MCC_FPSET_BACKEND_ENV, None, || {
            resolve_requested_fpset_backend(None, 1)
                .expect("default backend should resolve for sequential runs")
        });
        assert_eq!(resolved, RequestedFpsetBackend::Sharded);
    }

    #[test]
    fn resolve_requested_fpset_backend_defaults_to_sharded_without_env_parallel() {
        // Keep the exact-result default on the guarded sharded backend. CAS may
        // still be requested explicitly, but exact observer dispatch refuses it
        // until folded-fingerprint admission has a payload collision guard.
        for workers in [2, 4, 8, 16] {
            let resolved = with_env_var(MCC_FPSET_BACKEND_ENV, None, || {
                resolve_requested_fpset_backend(None, workers)
                    .expect("default backend should resolve for parallel runs")
            });
            assert_eq!(
                resolved,
                RequestedFpsetBackend::Sharded,
                "workers={workers} should stay on the guarded sharded backend"
            );
        }
    }

    #[test]
    fn default_fpset_backend_for_workers_always_sharded() {
        assert_eq!(
            default_fpset_backend_for_workers(1),
            RequestedFpsetBackend::Sharded,
        );
        assert_eq!(
            default_fpset_backend_for_workers(2),
            RequestedFpsetBackend::Sharded,
        );
        assert_eq!(
            default_fpset_backend_for_workers(usize::MAX),
            RequestedFpsetBackend::Sharded,
        );
    }

    #[test]
    fn resolve_requested_fpset_backend_uses_mcc_env() {
        // Env override wins over the worker-aware auto default, in both
        // sequential and parallel worker regimes.
        for workers in [1usize, 4] {
            let resolved = with_env_var(MCC_FPSET_BACKEND_ENV, Some("  CAS  "), || {
                resolve_requested_fpset_backend(None, workers).expect("env backend should resolve")
            });
            assert_eq!(resolved, RequestedFpsetBackend::Cas);
        }
    }

    #[test]
    fn resolve_requested_fpset_backend_env_can_force_sharded_in_parallel() {
        // R-3 changes the no-env default for workers>=2 but must not
        // override an explicit `TY_MCC_FPSET_BACKEND=sharded` opt-out.
        let resolved = with_env_var(MCC_FPSET_BACKEND_ENV, Some("sharded"), || {
            resolve_requested_fpset_backend(None, 8)
                .expect("explicit sharded env should beat parallel auto-default")
        });
        assert_eq!(resolved, RequestedFpsetBackend::Sharded);
    }

    #[test]
    fn resolve_requested_fpset_backend_cli_overrides_mcc_env() {
        let resolved = with_env_var(MCC_FPSET_BACKEND_ENV, Some("cas"), || {
            resolve_requested_fpset_backend(Some(RequestedFpsetBackend::Sharded), 4)
                .expect("explicit CLI backend should win")
        });
        assert_eq!(resolved, RequestedFpsetBackend::Sharded);
    }

    #[test]
    fn resolve_requested_fpset_backend_cli_overrides_parallel_auto_default() {
        // CLI flag must beat the new parallel auto-default too.
        let resolved = with_env_var(MCC_FPSET_BACKEND_ENV, None, || {
            resolve_requested_fpset_backend(Some(RequestedFpsetBackend::Sharded), 8)
                .expect("explicit CLI sharded should beat parallel auto-default")
        });
        assert_eq!(resolved, RequestedFpsetBackend::Sharded);
    }

    #[test]
    fn resolve_requested_fpset_backend_rejects_invalid_mcc_env() {
        let err = with_env_var(MCC_FPSET_BACKEND_ENV, Some("locks"), || {
            resolve_requested_fpset_backend(None, 1).expect_err("invalid env value should fail")
        });
        assert!(err.to_string().contains(MCC_FPSET_BACKEND_ENV));
    }

    fn write_minimal_model(dir: &Path) {
        std::fs::write(
            dir.join("model.pnml"),
            r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p1">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <place id="P1"/>
      <transition id="T0"/>
      <arc id="a1" source="P0" target="T0"/>
      <arc id="a2" source="T0" target="P1"/>
    </page>
  </net>
</pnml>"#,
        )
        .expect("minimal PNML should write");
    }

    #[test]
    fn build_exploration_config_uses_shared_parallel_mcc_defaults() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        write_minimal_model(dir.path());
        let model = crate::model::load_model_dir(dir.path()).expect("minimal model should load");

        let config = with_env_var(MCC_FPSET_BACKEND_ENV, None, || {
            build_exploration_config(
                &model,
                &PetriRunArgs {
                    threads: 4,
                    ..PetriRunArgs::default()
                },
            )
            .expect("shared MCC config should build")
        });

        assert_eq!(config.workers(), 4);
        assert_eq!(config.storage_mode(), StorageMode::FingerprintOnly);
        assert_eq!(config.fpset_backend(), FpsetBackend::Sharded);
        assert_eq!(config.max_states(), usize::MAX);
        assert!(
            config.refitted_for_full_graph(model.net()).max_states() < usize::MAX,
            "auto fingerprint storage must still preserve a bounded full-graph budget for CTL"
        );
    }

    #[test]
    fn build_exploration_config_honors_shared_env_override() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        write_minimal_model(dir.path());
        let model = crate::model::load_model_dir(dir.path()).expect("minimal model should load");

        let config = with_env_var(MCC_FPSET_BACKEND_ENV, Some("sharded"), || {
            build_exploration_config(
                &model,
                &PetriRunArgs {
                    threads: 8,
                    ..PetriRunArgs::default()
                },
            )
            .expect("shared MCC config should build")
        });

        assert_eq!(config.workers(), 8);
        assert_eq!(config.storage_mode(), StorageMode::FingerprintOnly);
        assert_eq!(config.fpset_backend(), FpsetBackend::Sharded);
    }

    /// EMIT-ONCE GATE: when the worker has emitted nothing (atomic
    /// StateSpace-style runaway), the watchdog emits its full fallback — the
    /// runaway guarantee. When the worker already flushed >=1 line (a
    /// multi-formula examination that incrementally flushed verdicts), the
    /// watchdog emits NOTHING, so it can never double-emit duplicate FORMULA
    /// lines. Tested on the pure gate decision (no process::exit).
    #[test]
    fn watchdog_fallback_suppressed_iff_output_started() {
        let fallback = vec![
            "FORMULA Id-00 CANNOT_COMPUTE TECHNIQUES EXPLICIT".to_string(),
            "FORMULA Id-01 CANNOT_COMPUTE TECHNIQUES EXPLICIT".to_string(),
        ];

        // Nothing emitted yet ⇒ the fallback fires (runaway guarantee).
        assert_eq!(
            watchdog_fallback_to_emit(false, fallback.clone()),
            fallback,
            "with no prior output, the watchdog MUST emit the full CC fallback"
        );

        // Worker already flushed ≥1 line ⇒ the fallback is fully suppressed.
        assert!(
            watchdog_fallback_to_emit(true, fallback).is_empty(),
            "once output has started, the watchdog must emit nothing (no double-emit)"
        );
    }
}
