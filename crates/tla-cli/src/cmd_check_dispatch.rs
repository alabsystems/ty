// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Dispatch handler for the `ty check` subcommand.
//!
//! This module holds the (large, feature-gated) body that backs
//! `Command::Check`. It was lifted verbatim out of `async_main` in `main.rs`
//! as pure code motion: the resolved behaviour for `ty check` is byte-for-byte
//! identical. The destructuring of the `Command::Check` variant stays at the
//! `async_main` call site; this function receives each field as a parameter
//! (with the same `#[cfg(feature = "ay")]` gating it carried in the pattern).

use std::time::Instant;

use anyhow::{bail, Result};

use crate::cli_schema::{
    CheckBackend, LivenessModeArg, OutputFormat, SoundnessGate, StrategyArg, TraceFormat,
    TypeSpecializeArg,
};
use crate::cmd_check::{cmd_check, CheckConfig};
use crate::{cmd_simulate, emit_check_cli_error, tlc_tool, JsonErrorCtx};
use tla_check::{error_codes, ErrorSuggestion, SearchCompleteness};

// This helper mirrors the full set of CLI flags 1:1 to detect mutually
// incompatible combinations; the bool count tracks the CLI surface by design.
#[allow(clippy::fn_params_excessive_bools)]
fn incompatible_check_simulate_flags(
    workers: usize,
    no_deadlock: bool,
    max_states: usize,
    max_depth: usize,
    memory_limit: usize,
    disk_limit: usize,
    soundness: SoundnessGate,
    require_exhaustive: bool,
    bmc: usize,
    #[cfg(feature = "ay")] pdr: bool,
    #[cfg(feature = "ay")] kinduction: bool,
    pipeline: bool,
    strategy: &Option<StrategyArg>,
    por: bool,
    coverage: bool,
    allow_vacuous: &[String],
    strict_vacuity: bool,
    no_trace: bool,
    store_states: bool,
    initial_capacity: Option<usize>,
    mmap_fingerprints: Option<usize>,
    disk_fingerprints: Option<usize>,
    mmap_dir: &Option<std::path::PathBuf>,
    trace_file: &Option<std::path::PathBuf>,
    mmap_trace_locations: Option<usize>,
    checkpoint: &Option<std::path::PathBuf>,
    checkpoint_interval: u64,
    resume: &Option<std::path::PathBuf>,
    output: OutputFormat,
    tool: bool,
    trace_format: TraceFormat,
    difftrace: bool,
    continue_on_error: bool,
    allow_incomplete: bool,
    force: bool,
    profile_enum: bool,
    profile_enum_detail: bool,
    profile_eval: bool,
    liveness_mode: LivenessModeArg,
) -> Vec<&'static str> {
    let mut incompatible = Vec::new();
    if workers != 0 {
        incompatible.push("--workers");
    }
    if no_deadlock {
        incompatible.push("--no-deadlock");
    }
    if max_states != 0 {
        incompatible.push("--max-states");
    }
    if max_depth != 0 {
        incompatible.push("--max-depth");
    }
    if memory_limit != 0 {
        incompatible.push("--memory-limit");
    }
    if disk_limit != 0 {
        incompatible.push("--disk-limit");
    }
    if !matches!(soundness, SoundnessGate::Sound) {
        incompatible.push("--soundness");
    }
    if require_exhaustive {
        incompatible.push("--require-exhaustive");
    }
    if bmc != 0 {
        incompatible.push("--bmc");
    }
    #[cfg(feature = "ay")]
    if pdr {
        incompatible.push("--pdr");
    }
    #[cfg(feature = "ay")]
    if kinduction {
        incompatible.push("--kinduction");
    }
    if pipeline {
        incompatible.push("--pipeline");
    }
    if strategy.is_some() {
        incompatible.push("--strategy");
    }
    if por {
        incompatible.push("--por");
    }
    if coverage {
        incompatible.push("--coverage");
    }
    if !allow_vacuous.is_empty() {
        incompatible.push("--allow-vacuous");
    }
    if strict_vacuity {
        incompatible.push("--strict-vacuity");
    }
    if no_trace {
        incompatible.push("--no-trace");
    }
    if store_states {
        incompatible.push("--store-states");
    }
    if initial_capacity.is_some() {
        incompatible.push("--initial-capacity");
    }
    if mmap_fingerprints.is_some() {
        incompatible.push("--mmap-fingerprints");
    }
    if disk_fingerprints.is_some() {
        incompatible.push("--disk-fingerprints");
    }
    if mmap_dir.is_some() {
        incompatible.push("--mmap-dir");
    }
    if trace_file.is_some() {
        incompatible.push("--trace-file");
    }
    if mmap_trace_locations.is_some() {
        incompatible.push("--mmap-trace-locations");
    }
    if checkpoint.is_some() {
        incompatible.push("--checkpoint");
    }
    if checkpoint_interval != 300 {
        incompatible.push("--checkpoint-interval");
    }
    if resume.is_some() {
        incompatible.push("--resume");
    }
    if !matches!(output, OutputFormat::Human) {
        incompatible.push("--output");
    }
    if tool {
        incompatible.push("--tool");
    }
    if !matches!(trace_format, TraceFormat::Text) {
        incompatible.push("--trace-format");
    }
    if difftrace {
        incompatible.push("--difftrace");
    }
    if continue_on_error {
        incompatible.push("--continue-on-error");
    }
    if allow_incomplete {
        incompatible.push("--allow-incomplete");
    }
    if force {
        incompatible.push("--force");
    }
    if profile_enum {
        incompatible.push("--profile-enum");
    }
    if profile_enum_detail {
        incompatible.push("--profile-enum-detail");
    }
    if profile_eval {
        incompatible.push("--profile-eval");
    }
    if !matches!(liveness_mode, LivenessModeArg::Full) {
        incompatible.push("--liveness-mode");
    }
    incompatible
}

/// Backend selection + flag wiring + dispatch for `ty check`.
///
/// Lifted verbatim from the `Command::Check` arm of `async_main`. The parameter
/// list mirrors the `Command::Check` variant fields one-to-one (including the
/// `#[cfg(feature = "ay")]` gates), so the resolved behaviour is unchanged.
#[allow(clippy::fn_params_excessive_bools)]
pub(crate) fn cmd_check_dispatch(
    file: std::path::PathBuf,
    config: Option<std::path::PathBuf>,
    compiled: bool,
    gpu: bool,
    no_gpu: bool,
    quint: bool,
    random_walks: usize,
    walk_depth: usize,
    simulate: bool,
    workers: usize,
    no_deadlock: bool,
    max_states: usize,
    max_depth: usize,
    memory_limit: usize,
    disk_limit: usize,
    soundness: SoundnessGate,
    require_exhaustive: bool,
    bmc: usize,
    bmc_incremental: bool,
    #[cfg(feature = "ay")] pdr: bool,
    #[cfg(feature = "ay")] kinduction: bool,
    #[cfg(feature = "ay")] kinduction_max_k: usize,
    #[cfg(feature = "ay")] kinduction_incremental: bool,
    bfs_only: bool,
    pipeline: bool,
    strategy: Option<StrategyArg>,
    #[cfg(feature = "ay")] fused: bool,
    portfolio: bool,
    portfolio_strategies: Vec<String>,
    por: bool,
    auto_por: bool,
    no_auto_por: bool,
    auto_symmetry: bool,
    no_auto_symmetry: bool,
    no_reduction: bool,
    record_set_native: bool,
    no_record_set_native: bool,
    estimate: bool,
    estimate_only: Option<usize>,
    coverage: bool,
    allow_vacuous: Vec<String>,
    strict_vacuity: bool,
    profile_enum: bool,
    profile_enum_detail: bool,
    profile_eval: bool,
    liveness_mode: LivenessModeArg,
    strict_liveness: bool,
    jit: bool,
    jit_verify: bool,
    show_tiers: bool,
    type_specialize: TypeSpecializeArg,
    no_trace: bool,
    store_states: bool,
    initial_capacity: Option<usize>,
    mmap_fingerprints: Option<usize>,
    huge_pages: bool,
    disk_fingerprints: Option<usize>,
    mmap_dir: Option<std::path::PathBuf>,
    trace_file: Option<std::path::PathBuf>,
    mmap_trace_locations: Option<usize>,
    collision_check: String,
    checkpoint: Option<std::path::PathBuf>,
    checkpoint_interval: u64,
    resume: Option<std::path::PathBuf>,
    output: OutputFormat,
    tool: bool,
    trace_format: TraceFormat,
    difftrace: bool,
    explain_trace: bool,
    continue_on_error: bool,
    allow_incomplete: bool,
    force: bool,
    init: Option<String>,
    next: Option<String>,
    invariants: Vec<String>,
    properties: Vec<String>,
    constants: Vec<String>,
    no_config: bool,
    no_preprocess: bool,
    partial_eval: bool,
    allow_io: bool,
    trace_invariants: Vec<String>,
    #[cfg(feature = "ay")] inductive_check: Option<String>,
    #[cfg(feature = "ay")] symbolic_sim: bool,
    #[cfg(feature = "ay")] sim_runs: usize,
    #[cfg(feature = "ay")] sim_length: usize,
    backend: Option<CheckBackend>,
) -> Result<()> {
    if strict_vacuity && auto_por {
        bail!("--strict-vacuity cannot be combined with --auto-por");
    }
    if strict_vacuity && auto_symmetry {
        bail!("--strict-vacuity cannot be combined with --auto-symmetry");
    }

    // Semantic levers are controlled ONLY by CLI flags. The TY_* env vars
    // below are internal plumbing (read once via OnceLock/LazyLock — or via
    // the set-once global overlay snapshot installed further down — deep in
    // the libraries before worker threads spawn); we resolve them explicitly
    // here, BEFORE `set_global_overlay` captures the environment, so AMBIENT
    // environment variables can never silently change checking semantics.
    // The `--auto-por` / `--auto-symmetry` force-on flags stay meaningful:
    // they install the explicit "1" (which the engine must honor over e.g.
    // the native-fused POR release), while the flagless default REMOVES the
    // var — on-by-default without the "explicit user request" semantics —
    // keeping default behavior byte-identical to before.
    for lever in resolve_semantic_lever_env(SemanticLeverFlags {
        auto_por,
        no_auto_por,
        auto_symmetry,
        no_auto_symmetry,
        no_reduction,
        record_set_native,
        no_record_set_native,
    }) {
        if let Ok(ambient) = std::env::var(lever.var) {
            if lever.value != Some(ambient.as_str()) {
                eprintln!(
                    "warning: ambient {} ignored; use {} (env vars no longer control semantics)",
                    lever.var,
                    lever.suggested_flag_for(&ambient),
                );
            }
        }
        match lever.value {
            Some(value) => crate::env_guard::set_var(lever.var, value),
            None => crate::env_guard::remove_var(lever.var),
        }
    }
    // Backend / engine selection.
    //
    // `--backend trust-cg` routes to the trust-codegen native-compiled
    // BFS path. The trust-codegen path is always linked in; activation
    // is environment-variable based: setting `TY_TRUST_CG_BFS=1` enables
    // per-action trust-codegen compilation inside the BFS loop, with
    // interpreter fallback for ineligible actions (arity > 0,
    // unsupported opcodes). See
    // `crates/tla-check/src/check/model_checker/trust_cg_dispatch/` (the dispatch module).
    //
    // PRODUCTION DEFAULT (no `--backend` flag): AUTO engine selection.
    // The native path is the default engine, but a cheap structural
    // pre-check (in tla-check) routes a run to the interpreter when the
    // native path would not help — keeping the default <= interpreter on
    // every spec. We signal AUTO mode to the checker via
    // `TY_TRUST_CG_AUTO_SELECT=1` so the structural veto and the
    // post-compile coverage teardown are active; we do NOT enable them
    // for an EXPLICIT `--backend trust-cg`, so the supremacy harnesses
    // and oracle cross-checks (which pass `--backend trust-cg`
    // explicitly) keep their unchanged forced-native behavior.
    //
    // `--backend interpreter` forces the oracle — no dispatch needed.
    // Unified-backend env-handoff (docs/env-handoff-set-once-global-2026-06-06.md):
    // the typed `EngineRequest` replaces the inline boolean/env logic, and
    // `set_global_overlay(build_engine_overlay(..))` installs a set-once process-global
    // IMMUTABLE env snapshot. It captures the real env and synthesizes the SAME two
    // `TY_TRUST_CG_*=1` values under the SAME `!contains_key` guards (so an explicit
    // `TY_TRUST_CG_BFS=0` still disables), keeping `ty check` byte-identical — with no
    // `unsafe set_var`. The deep readers (R1 `is_enabled`, R2 `trust_cg_auto_select_enabled`)
    // consult the global with a legacy env fallback. MUST run at single-threaded startup,
    // before any checker worker thread is spawned.
    let engine_request = tla_backend::EngineRequest::for_check(
        crate::cli_schema::CheckBackend::to_selection_mode(backend),
    );
    tla_backend::set_global_overlay(tla_backend::build_engine_overlay(&engine_request));
    if simulate {
        let incompatible = incompatible_check_simulate_flags(
            workers,
            no_deadlock,
            max_states,
            max_depth,
            memory_limit,
            disk_limit,
            soundness,
            require_exhaustive,
            bmc,
            #[cfg(feature = "ay")]
            pdr,
            #[cfg(feature = "ay")]
            kinduction,
            pipeline,
            &strategy,
            por,
            coverage,
            &allow_vacuous,
            strict_vacuity,
            no_trace,
            store_states,
            initial_capacity,
            mmap_fingerprints,
            disk_fingerprints,
            &mmap_dir,
            &trace_file,
            mmap_trace_locations,
            &checkpoint,
            checkpoint_interval,
            &resume,
            output,
            tool,
            trace_format,
            difftrace,
            continue_on_error,
            allow_incomplete,
            force,
            profile_enum,
            profile_enum_detail,
            profile_eval,
            liveness_mode,
        );
        if !incompatible.is_empty() {
            bail!(
                "`ty check --simulate` is a compatibility alias for `ty simulate`. \
                 Unsupported check-only flags: {}. Use `ty simulate` for simulation \
                 controls such as `--num-traces`, `--max-trace-length`, `--seed`, and \
                 `--no-invariants`.",
                incompatible.join(", ")
            );
        }
        return cmd_simulate::cmd_simulate(&file, config.as_deref(), 1000, 100, 0, false, false);
    }
    #[cfg(feature = "ay")]
    let pdr_enabled = pdr;
    #[cfg(not(feature = "ay"))]
    let pdr_enabled = false;
    #[cfg(feature = "ay")]
    let kinduction_enabled = kinduction;
    #[cfg(not(feature = "ay"))]
    let kinduction_enabled = false;
    #[cfg(feature = "ay")]
    let kinduction_max_k_val = kinduction_max_k;
    #[cfg(not(feature = "ay"))]
    let kinduction_max_k_val: usize = 20;
    #[cfg(feature = "ay")]
    let kinduction_incremental_val = kinduction_incremental;
    #[cfg(not(feature = "ay"))]
    let kinduction_incremental_val = false;
    #[cfg(feature = "ay")]
    let inductive_check_invariant = inductive_check;
    #[cfg(not(feature = "ay"))]
    let inductive_check_invariant: Option<String> = None;
    // Part of #3953: CDEMC/fused is the default when ay is enabled.
    // Auto-enable unless the user explicitly requested --bfs-only,
    // selected another explicit mode, or uses features that require
    // the full BFS CLI path (checkpoint, trace-file, tlc-tool output).
    #[cfg(feature = "ay")]
    let fused_enabled = if bfs_only {
        false
    } else if fused {
        // Explicit --fused (deprecated but still accepted).
        true
    } else if pdr || kinduction || bmc > 0 || pipeline || strategy.is_some() || portfolio {
        // User requested a specific mode — don't override.
        false
    } else {
        // Fall back to BFS for features the fused path doesn't wire.
        let needs_full_bfs =
            tool || checkpoint.is_some() || resume.is_some() || trace_file.is_some();
        !needs_full_bfs
    };
    #[cfg(not(feature = "ay"))]
    let fused_enabled = false;
    #[cfg(not(feature = "ay"))]
    let _ = bfs_only;
    #[cfg(feature = "ay")]
    let symbolic_sim_enabled = symbolic_sim;
    #[cfg(not(feature = "ay"))]
    let symbolic_sim_enabled = false;
    #[cfg(feature = "ay")]
    let sim_runs_val = sim_runs;
    #[cfg(not(feature = "ay"))]
    let sim_runs_val: usize = 100;
    #[cfg(feature = "ay")]
    let sim_length_val = sim_length;
    #[cfg(not(feature = "ay"))]
    let sim_length_val: usize = 10;
    let output_format = if tool { OutputFormat::TlcTool } else { output };
    let effective_workers = if matches!(output_format, OutputFormat::TlcTool) && workers == 0 {
        // Tool mode prioritizes Toolbox compatibility (especially error traces).
        // Today, traces are only reconstructed in sequential mode.
        1
    } else {
        workers
    };
    // Part of #3746: Wire --strict-liveness to env var before OnceLock init.
    if strict_liveness {
        crate::env_guard::set_var("TY_STRICT_LIVENESS", "1");
    }
    // Part of #4035: Wire --jit to env var before OnceLock init.
    // JIT is off by default; --jit enables it at runtime.
    if jit {
        {
            crate::env_guard::set_var("TY_JIT", "1");
        }
    }
    // Auto-POR / auto-symmetry / record-set-native env resolution happens at
    // the very top of this function (before the engine overlay snapshot is
    // installed); see `resolve_semantic_lever_env`.
    if no_preprocess {
        crate::env_guard::set_var("TY_NO_PREPROCESS", "1");
    }
    // Part of #4251 Stream 5: partial-evaluate CONSTANTS into TIR
    // before the preprocessing pipeline. Gated by env var that is
    // read at most once per process via LazyLock, so it must be set
    // before any tla-eval module that calls `partial_eval_enabled()`.
    if partial_eval {
        crate::env_guard::set_var("TY_PARTIAL_EVAL", "1");
    }
    // Part of #3965: Wire --allow-io to enable IOExec command execution.
    if allow_io {
        tla_check::eval::set_io_exec_allowed(true);
        eprintln!(
            "Warning: --allow-io is enabled. IOExec and related operators can execute \
             arbitrary shell commands. Only use this with trusted specs."
        );
    }
    let tool_cli_started = Instant::now();
    // Auto-detect Quint JSON IR from file extension.
    let quint_mode = quint || tla_core::quint::is_quint_json_path(&file);
    let result = cmd_check(CheckConfig {
        file: file.clone(),
        config_path: config.clone(),
        compiled,
        gpu,
        no_gpu,
        quint: quint_mode,
        random_walks,
        walk_depth,
        workers: effective_workers,
        no_deadlock,
        max_states,
        max_depth,
        memory_limit,
        disk_limit,
        soundness_gate: soundness,
        require_exhaustive,
        bmc_depth: bmc,
        bmc_incremental,
        pdr_enabled,
        kinduction_enabled,
        kinduction_max_k: kinduction_max_k_val,
        kinduction_incremental: kinduction_incremental_val,
        por_enabled: por,
        // show_progress removed: always-on for Human output (#3247)
        show_coverage: coverage,
        allow_vacuous,
        strict_vacuity,
        estimate: estimate || estimate_only.is_some(),
        estimate_only,
        no_trace,
        store_states,
        initial_capacity,
        mmap_fingerprints,
        huge_pages: huge_pages || std::env::var("TY_HUGE_PAGES").is_ok(),
        disk_fingerprints,
        mmap_dir,
        trace_file_path: trace_file,
        mmap_trace_locations,
        checkpoint_dir: checkpoint,
        checkpoint_interval,
        resume_from: resume,
        output_format,
        trace_format,
        difftrace,
        explain_trace,
        continue_on_error,
        allow_incomplete,
        force,
        profile_enum,
        profile_enum_detail,
        profile_eval,
        liveness_mode,
        strict_liveness,
        jit,
        jit_verify,
        show_tiers,
        type_specialize,
        pipeline,
        strategy,
        fused: fused_enabled,
        portfolio,
        portfolio_strategies,
        cli_init: init,
        cli_next: next,
        cli_invariants: invariants,
        cli_properties: properties,
        cli_constants: constants,
        no_config,
        no_preprocess,
        partial_eval,
        trace_invariants,
        inductive_check_invariant,
        symbolic_sim: symbolic_sim_enabled,
        symbolic_sim_runs: sim_runs_val,
        symbolic_sim_length: sim_length_val,
        collision_check,
    });

    if matches!(
        output_format,
        OutputFormat::Json | OutputFormat::Jsonl | OutputFormat::Itf
    ) {
        if let Err(e) = result {
            let completeness = SearchCompleteness::from_bounds(max_states, max_depth);
            emit_check_cli_error(
                &JsonErrorCtx {
                    output_format,
                    spec_file: &file,
                    config_file: config.as_deref(),
                    module_name: None,
                    workers: effective_workers,
                    completeness,
                },
                error_codes::SYS_SETUP_ERROR,
                format!("{e:#}"),
                Some(ErrorSuggestion::new(
                    "Fix the spec/config error, then re-run the command",
                )),
                std::iter::empty::<String>(),
            );
        }
        Ok(())
    } else if matches!(output_format, OutputFormat::TlcTool) {
        if let Err(e) = result {
            let completeness = SearchCompleteness::from_bounds(max_states, max_depth);
            tlc_tool::emit_check_tool_cli_error(
                &file,
                config.as_deref(),
                effective_workers,
                completeness,
                tool_cli_started.elapsed(),
                &format!("{e}"),
            );
        }
        Ok(())
    } else {
        result
    }
}

/// The `ty check` flags that resolve the flag-controlled semantic levers.
///
/// Grouped in a struct (rather than positional bools) so the resolution helper
/// below is unit-testable and call sites stay readable.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SemanticLeverFlags {
    pub(crate) auto_por: bool,
    pub(crate) no_auto_por: bool,
    pub(crate) auto_symmetry: bool,
    pub(crate) no_auto_symmetry: bool,
    pub(crate) no_reduction: bool,
    pub(crate) record_set_native: bool,
    pub(crate) no_record_set_native: bool,
}

/// One resolved semantic lever: the internal env var the CLI installs and the
/// value it must carry. `value: None` means "remove the var" — the library
/// default (on for POR/symmetry, off for record-set-native) without the
/// "explicit user request" semantics that an installed `"1"` carries for POR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedSemanticLever {
    pub(crate) var: &'static str,
    pub(crate) value: Option<&'static str>,
    enable_flag: &'static str,
    disable_flag: &'static str,
}

impl ResolvedSemanticLever {
    /// The flag to suggest in the ambient-env-ignored warning: the one that
    /// reproduces what the ambient value was asking for.
    pub(crate) fn suggested_flag_for(&self, ambient: &str) -> &'static str {
        let v = ambient.trim();
        let ambient_off = v == "0"
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("off")
            || v.eq_ignore_ascii_case("no");
        if ambient_off {
            self.disable_flag
        } else {
            self.enable_flag
        }
    }
}

/// Resolve the three flag-controlled semantic levers (auto-POR, auto-symmetry,
/// record-set-native) to the `TY_*` env state the CLI installs. Ambient env is
/// deliberately NOT consulted: flags are the only user-facing control surface;
/// the env vars are internal plumbing to the LazyLock/overlay readers.
///
/// - `--no-auto-por` / `--no-auto-symmetry` / `--no-reduction` → `"0"`.
/// - `--auto-por` / `--auto-symmetry` → explicit `"1"` (for POR this is a
///   deliberate request the engine must honor over the native-fused POR
///   release; see `tla_check::por::auto_por_explicitly_enabled`).
/// - Default (no flags) → remove the var: on-by-default, not explicit.
/// - `--record-set-native` → `"1"`; default / `--no-record-set-native` → `"0"`
///   (default-OFF pending trust-toolchain soundness validation).
pub(crate) fn resolve_semantic_lever_env(flags: SemanticLeverFlags) -> [ResolvedSemanticLever; 3] {
    let on_off = |force_on: bool, off: bool| {
        if off {
            Some("0")
        } else if force_on {
            Some("1")
        } else {
            None
        }
    };
    [
        ResolvedSemanticLever {
            var: "TY_AUTO_POR",
            value: on_off(flags.auto_por, flags.no_auto_por || flags.no_reduction),
            enable_flag: "--auto-por",
            disable_flag: "--no-auto-por",
        },
        ResolvedSemanticLever {
            var: "TY_AUTO_SYMMETRY",
            value: on_off(
                flags.auto_symmetry,
                flags.no_auto_symmetry || flags.no_reduction,
            ),
            enable_flag: "--auto-symmetry",
            disable_flag: "--no-auto-symmetry",
        },
        ResolvedSemanticLever {
            var: "TY_RECORD_SET_NATIVE",
            // Authority quarantine: keep the native RecordSetBitmask kernel
            // default OFF. Corpus-wide state equality is strong regression
            // evidence, but it is not a universal proof that every admitted
            // RecordSetBitmask action preserves interpreter semantics. The
            // explicit --record-set-native flag remains available for controlled
            // validation; normal and --no-reduction runs stay on the interpreter.
            value: if flags.record_set_native {
                Some("1")
            } else {
                let _ = flags.no_record_set_native;
                Some("0")
            },
            enable_flag: "--record-set-native",
            disable_flag: "--no-record-set-native",
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::{resolve_semantic_lever_env, SemanticLeverFlags};

    fn value_of(levers: &[super::ResolvedSemanticLever; 3], var: &str) -> Option<&'static str> {
        levers.iter().find(|lever| lever.var == var).unwrap().value
    }

    #[test]
    fn default_resolution_leaves_reductions_on_and_record_set_native_off() {
        let levers = resolve_semantic_lever_env(SemanticLeverFlags::default());
        // Default-on reducer levers are installed as REMOVED vars (on-by-default,
        // not the explicit "1", which for POR suppresses the native-fused POR
        // release); the experimental native record-set kernel is pinned off.
        assert_eq!(value_of(&levers, "TY_AUTO_POR"), None);
        assert_eq!(value_of(&levers, "TY_AUTO_SYMMETRY"), None);
        assert_eq!(value_of(&levers, "TY_RECORD_SET_NATIVE"), Some("0"));
    }

    #[test]
    fn no_reduction_pins_reducers_and_record_set_native_off() {
        let levers = resolve_semantic_lever_env(SemanticLeverFlags {
            no_reduction: true,
            ..SemanticLeverFlags::default()
        });
        assert_eq!(value_of(&levers, "TY_AUTO_POR"), Some("0"));
        assert_eq!(value_of(&levers, "TY_AUTO_SYMMETRY"), Some("0"));
        assert_eq!(value_of(&levers, "TY_RECORD_SET_NATIVE"), Some("0"));
    }

    #[test]
    fn individual_no_flags_pin_only_their_lever() {
        let levers = resolve_semantic_lever_env(SemanticLeverFlags {
            no_auto_symmetry: true,
            ..SemanticLeverFlags::default()
        });
        assert_eq!(value_of(&levers, "TY_AUTO_POR"), None);
        assert_eq!(value_of(&levers, "TY_AUTO_SYMMETRY"), Some("0"));
    }

    #[test]
    fn force_on_flags_install_the_explicit_one() {
        let levers = resolve_semantic_lever_env(SemanticLeverFlags {
            auto_por: true,
            auto_symmetry: true,
            record_set_native: true,
            ..SemanticLeverFlags::default()
        });
        assert_eq!(value_of(&levers, "TY_AUTO_POR"), Some("1"));
        assert_eq!(value_of(&levers, "TY_AUTO_SYMMETRY"), Some("1"));
        assert_eq!(value_of(&levers, "TY_RECORD_SET_NATIVE"), Some("1"));
    }

    #[test]
    fn ambient_warning_suggests_the_flag_matching_the_ambient_intent() {
        let levers = resolve_semantic_lever_env(SemanticLeverFlags::default());
        let symmetry = levers
            .iter()
            .find(|lever| lever.var == "TY_AUTO_SYMMETRY")
            .unwrap();
        assert_eq!(symmetry.suggested_flag_for("0"), "--no-auto-symmetry");
        assert_eq!(symmetry.suggested_flag_for("false"), "--no-auto-symmetry");
        assert_eq!(symmetry.suggested_flag_for("1"), "--auto-symmetry");
    }
}
