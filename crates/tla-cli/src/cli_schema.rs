// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CLI argument schema: command definitions, output format types, and gating enums.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;

/// Template for the `init` command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum InitTemplate {
    /// Simple spec with Init, Next, TypeOK (default).
    #[default]
    Basic,
    /// Multi-process spec with VARIABLE pc, messages, EXTENDS Sequences.
    Distributed,
    /// Mutual exclusion protocol template.
    Mutex,
    /// Cache coherence protocol template.
    Cache,
}

/// Output format for diagnose command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DiagnoseOutputFormat {
    /// Human-readable summary table (default)
    #[default]
    Human,
    /// Structured JSON output compatible with metrics/spec_coverage.json
    Json,
}

/// Differential oracle mode for `ty diagnose`.
///
/// The tree-walking interpreter is the permanent correctness oracle for every
/// compiled backend (trust_cg today, trust-ir/others later). `compare` runs both and
/// records divergences to `metrics/oracle_parity.json`. `fail-closed` also
/// exits non-zero on any divergence, making it the CI gate mode.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum DiagnoseOracleMode {
    /// No oracle run — interpreter only (default).
    #[default]
    Off,
    /// Run both interpreter and trust_cg, record divergences, do NOT fail the build.
    Compare,
    /// Run both and exit non-zero on any divergence (CI gate).
    #[value(name = "fail-closed")]
    FailClosed,
}

/// Which evaluation backend `ty check` should use.
///
/// `interpreter` is the default and the permanent correctness oracle.
/// `trust_cg` is the native-compiled AOT path gated by current eligibility checks.
/// It may compile action/invariant artifacts, admit only some fast paths, or
/// fall back to the interpreter for soundness. If the backend is unavailable,
/// it emits a JSON `backend_unavailable` sentinel and exits with code 3 so the
/// oracle harness can classify the run as infra-gap rather than divergence.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum CheckBackend {
    /// Tree-walking interpreter (the oracle). Default.
    #[default]
    Interpreter,
    /// trust_cg-compiled path. Prototype/gated; falls back or reports backend_unavailable when needed.
    TrustCg,
}

impl CheckBackend {
    /// Map the `--backend` flag to the unified engine-selection mode.
    ///
    /// `None` (flag omitted) = AUTO (native default + structural auto-selection);
    /// `Some(TrustCg)` = forced native with NO structural veto (the explicit form the
    /// supremacy/oracle harnesses pass); `Some(Interpreter)` = forced oracle. This is
    /// the typed equivalent of the legacy `main.rs` resolution; the env effect is pinned
    /// by `tla_backend::legacy_env_plan`.
    pub(crate) fn to_selection_mode(backend: Option<CheckBackend>) -> tla_backend::SelectionMode {
        match backend {
            None => tla_backend::SelectionMode::Auto,
            Some(CheckBackend::TrustCg) => {
                tla_backend::SelectionMode::Forced(tla_backend::EngineId::TrustCgNative)
            }
            Some(CheckBackend::Interpreter) => tla_backend::SelectionMode::Oracle,
        }
    }
}

/// Enable the AUTO native engine for a BFS-shaped subcommand (Stage 5 of the
/// unified-backend migration) — the same default `ty check` uses. Native compilation is
/// gated and falls back to the interpreter per action (with the AUTO structural veto), so
/// verification results are unchanged; `TY_TRUST_CG_BFS=0` still disables it. Safe for
/// trace-printing subcommands now that the native path reconstructs full deadlock traces.
/// MUST be called at single-threaded dispatch, before the checker spawns worker threads.
pub(crate) fn enable_auto_native_engine(problem: tla_backend::ProblemKind) {
    let req = tla_backend::EngineRequest::for_problem(problem, tla_backend::SelectionMode::Auto);
    tla_backend::set_global_overlay(tla_backend::build_engine_overlay(&req));
}

#[cfg(test)]
mod backend_selection_tests {
    use super::CheckBackend;
    use tla_backend::{EngineId, SelectionMode};

    /// Golden: pin the `--backend` flag → selection-mode mapping. Combined with
    /// `tla_backend`'s `legacy_env_plan` truth-table tests, this fixes the exact
    /// `TY_TRUST_CG_*` env effect per `--backend` value so the migration stays
    /// byte-identical.
    #[test]
    fn check_backend_flag_maps_to_selection_mode() {
        assert_eq!(CheckBackend::to_selection_mode(None), SelectionMode::Auto);
        assert_eq!(
            CheckBackend::to_selection_mode(Some(CheckBackend::TrustCg)),
            SelectionMode::Forced(EngineId::TrustCgNative)
        );
        assert_eq!(
            CheckBackend::to_selection_mode(Some(CheckBackend::Interpreter)),
            SelectionMode::Oracle
        );
    }
}

/// Output format for bench command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum BenchOutputFormat {
    /// Human-readable table output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
    /// GitHub-Flavored Markdown table (for pasting into issues)
    Markdown,
}

/// Output format for supremacy evidence commands.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SupremacyOutputFormat {
    /// Human-readable status output (default).
    #[default]
    Human,
    /// Structured JSON output.
    Json,
    /// GitHub-Flavored Markdown output.
    Markdown,
}

/// Row set selected by `ty supremacy matrix --refresh-runtime` when no named
/// `--runtime-spec` values are provided.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum SupremacyMatrixRuntimeScope {
    /// Refresh only rows that currently lack TLC or TY runtime evidence.
    #[default]
    #[value(name = "missing-runtime")]
    MissingRuntime,
    /// Refresh every batchable row with a runnable source mode.
    #[value(name = "all-runnable")]
    AllRunnable,
}

/// Whether a supremacy gate warns or fails the process.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum SupremacyMode {
    /// Print warnings for failed policy checks.
    #[default]
    Warn,
    /// Fail closed for failed policy checks.
    Enforce,
}

/// Backend compared against TLC by `ty supremacy compare`.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum SupremacyCompareBackend {
    /// Tree-walking interpreter (the pure oracle: forces `--backend
    /// interpreter`, which disengages the AUTO-only certified value-action VM
    /// and trust-cg native — verified via engine provenance).
    #[default]
    Interpreter,
    /// trust-codegen compiled backend.
    TrustCg,
    /// Production-default AUTO routing (burndown P4): no `--backend` flag, no
    /// env pins — the child selects native / value-VM / interpreter / GPU
    /// exactly as a user's `ty check` would. Rows attribute via engine_tier.
    #[value(alias = "production")]
    Auto,
    /// Production AUTO with the GPU excluded (`--no-gpu`): the
    /// single-thread-eligible sound production arm — the only configuration
    /// that measures the AUTO-certified value-action VM on a CUDA host
    /// without hardware-track rows.
    AutoCpu,
}

/// Spec source mode for `ty supremacy compare`.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum SupremacyCompareSpecSource {
    /// Resolve named specs from the baseline JSON.
    #[default]
    Baseline,
    /// Use the explicit --tla/--config pair.
    Explicit,
}

/// Comparison policy for `ty supremacy compare`.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum SupremacyComparePolicy {
    /// Require state-count parity only.
    #[default]
    Parity,
    /// Require state-count parity and minimum TLC/TY speedup.
    #[value(name = "parity-and-speed")]
    ParityAndSpeed,
    /// Require state-count parity, minimum TLC/TY speedup, and a maximum
    /// TY/TLC peak-memory ratio.
    #[value(name = "parity-and-speed-and-memory")]
    ParityAndSpeedAndMemory,
}

/// Reference tool(s) `ty supremacy reproduce` compares TY against.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum SupremacyReproduceVs {
    /// Explicit-state model checker TLC (state-count + verdict parity, exhaustive).
    #[default]
    Tlc,
    /// Symbolic checker Apalache (verdict parity only, bounded by --len).
    Apalache,
    /// Both TLC and Apalache.
    Both,
}

/// Which canary set `ty canary-gate` should evaluate.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum CanaryGateKind {
    /// Eval/check regression canaries.
    Eval,
    /// Enumeration state-count canaries.
    Enumerate,
    /// API consumer compatibility canaries.
    Api,
    /// Silent eval-error coercion guard.
    #[value(name = "silent-error", alias = "silent-error-coercion")]
    SilentError,
    /// Eval/check, enumeration, API, and silent-error canaries.
    #[default]
    All,
}

/// Whether `ty canary-gate` warns or fails the process.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum CanaryGateMode {
    /// Print warnings for failed gates.
    #[default]
    Warn,
    /// Fail closed for failed gates.
    Enforce,
}

/// Arguments for `ty canary-gate`.
#[derive(Clone, Debug, Args)]
pub(crate) struct CanaryGateArgs {
    /// Canary set to run.
    #[arg(long, value_enum, default_value = "all")]
    pub kind: CanaryGateKind,
    /// Gate behavior: warn for development, enforce for blocking hooks.
    #[arg(long, value_enum, default_value = "warn")]
    pub mode: CanaryGateMode,
    /// Print captured canary build output when a canary fails.
    #[arg(long)]
    pub verbose: bool,
    /// Read staged files from `git diff --cached` and apply staged-only skip rules.
    #[arg(long, conflicts_with = "changed_files")]
    pub staged: bool,
    /// Changed files used to decide whether each canary set is relevant.
    ///
    /// If omitted, the gate reads `git diff --name-only HEAD`.
    #[arg(long, num_args = 1.., conflicts_with = "staged")]
    pub changed_files: Vec<PathBuf>,
}

/// Arguments for `ty rust-function-span-scan`.
#[derive(Clone, Debug, Args)]
pub(crate) struct RustFunctionSpanScanArgs {
    /// Function length limit. Functions strictly over this line count are reported.
    #[arg(long, required = true)]
    pub limit: usize,
    /// Explicit Rust source files to scan; directories are not expanded.
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

/// Whether `ty system-health-gate` fails closed or reports diagnostics only.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum SystemHealthGateMode {
    /// Preserve the legacy system-health failure contract.
    #[default]
    Enforce,
    /// Print diagnostics but exit successfully for transitional callers.
    Warn,
}

/// Arguments for `ty system-health-gate`.
#[derive(Clone, Debug, Args)]
pub(crate) struct SystemHealthGateArgs {
    /// Gate behavior: enforce legacy failures or report diagnostics only.
    #[arg(long, value_enum, default_value = "enforce")]
    pub mode: SystemHealthGateMode,
    /// Write a JSON manifest (schema v1.0).
    #[arg(long)]
    pub json_output: Option<PathBuf>,
    /// Repo root to check. Hidden test hook; defaults to the current directory.
    #[arg(long, hide = true)]
    pub project_root: Option<PathBuf>,
}

/// Supremacy policy mode.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum SupremacyGateMode {
    /// Interim native-fused action-only policy.
    #[value(
        name = "interim-action-only-native-fused",
        alias = "interim_action_only_native_fused"
    )]
    InterimActionOnlyNativeFused,
    /// Final strict native-fused launch policy.
    #[default]
    #[value(name = "full-native-fused", alias = "full_native_fused")]
    FullNativeFused,
}

/// Shared options for `ty supremacy` subcommands.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyCommonArgs {
    /// Supremacy policy JSON file. Defaults to tests/tlc_comparison/single_thread_supremacy_gate.json.
    #[arg(long)]
    pub policy: Option<PathBuf>,
    /// Output artifact directory. Defaults under reports/perf/.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    /// Pre-built ty binary to exercise for diagnostics or warn-mode runs; final enforce collection rejects this flag so it can build a fresh binary.
    #[arg(long)]
    pub ty_bin: Option<PathBuf>,
    /// Cargo target directory to build/use when --ty-bin is not supplied.
    #[arg(long)]
    pub target_dir: Option<PathBuf>,
    /// Cargo profile to build/use when --ty-bin is not supplied.
    #[arg(long, default_value = "release")]
    pub cargo_profile: String,
    /// Extra ty check flag passed to interpreter and trust-codegen runs before --backend.
    #[arg(long = "ty-flag")]
    pub ty_flag: Vec<String>,
    /// Timeout per subprocess run in seconds.
    #[arg(long, default_value = "300")]
    pub timeout: u64,
    /// Pinned spec names to run. Defaults to the policy's final corpus; non-corpus benchmark specs fall back to check-mode rows in spec_baseline.json for diagnostics only.
    #[arg(long, num_args = 1..)]
    pub specs: Vec<String>,
    /// Interpreter env override, as KEY=VALUE. May be repeated.
    #[arg(long = "interp-env")]
    pub interp_env: Vec<String>,
    /// trust-codegen env override, as KEY=VALUE. May be repeated.
    #[arg(long = "trust_cg-env")]
    pub trust_cg_env: Vec<String>,
    /// Output format.
    #[arg(long, value_enum, default_value = "human")]
    pub format: SupremacyOutputFormat,
}

/// Arguments for bounded native-fused smoke readiness checks.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacySmokeArgs {
    #[command(flatten)]
    pub common: SupremacyCommonArgs,
}

/// Arguments for raw single-thread supremacy benchmarking.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyBenchmarkArgs {
    #[command(flatten)]
    pub common: SupremacyCommonArgs,
    /// Number of repeated runs per backend/spec.
    #[arg(long, default_value = "3")]
    pub runs: usize,
}

/// Arguments for policy-enforced launch evidence.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyGateArgs {
    #[command(flatten)]
    pub common: SupremacyCommonArgs,
    /// Gate behavior: warn for development, enforce for launch evidence. Defaults to enforce.
    #[arg(long, value_enum)]
    pub mode: Option<SupremacyMode>,
    /// Policy mode to evaluate. Required in warn mode when the policy defines gate modes.
    #[arg(long, value_enum)]
    pub gate_mode: Option<SupremacyGateMode>,
    /// Number of repeated runs per backend/spec. Defaults to 3.
    #[arg(long)]
    pub runs: Option<usize>,
    /// Existing benchmark summary.json to evaluate with the Rust policy gate.
    #[arg(long)]
    pub summary_json: Option<PathBuf>,
}

/// Arguments for TLC-vs-TY comparison gate replacement.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyCompareArgs {
    /// Spec source: resolve --spec from --baseline, or use explicit --tla/--config.
    #[arg(long = "spec-source", value_enum, default_value = "baseline")]
    pub spec_source: SupremacyCompareSpecSource,
    /// Baseline JSON used by --spec-source baseline.
    #[arg(long, default_value = "tests/tlc_comparison/spec_baseline.json")]
    pub baseline: PathBuf,
    /// Baseline spec names to compare. In explicit mode, the first value is used as the report name.
    #[arg(long = "spec", num_args = 1..)]
    pub specs: Vec<String>,
    /// Explicit TLA+ spec path used by --spec-source explicit.
    #[arg(long)]
    pub tla: Option<PathBuf>,
    /// Explicit TLC config path used by --spec-source explicit.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// TY backend compared against TLC.
    #[arg(long, value_enum, default_value = "interpreter")]
    pub backend: SupremacyCompareBackend,
    /// Worker counts to evaluate for both TLC and TY.
    #[arg(long = "workers", num_args = 1.., default_values_t = vec![1usize])]
    pub workers: Vec<usize>,
    /// Number of paired TLC/TY repetitions per spec, worker count, and case.
    /// Enforced speed policies require an even count of at least 6 pairs.
    #[arg(long, default_value = "1")]
    pub runs: usize,
    /// Gate behavior: warn for reporting, enforce to fail on policy failures.
    #[arg(long, value_enum, default_value = "enforce")]
    pub mode: SupremacyMode,
    /// Policy to enforce over completed runs.
    #[arg(long = "policy", value_enum, default_value = "parity")]
    pub policy: SupremacyComparePolicy,
    /// TLC/TY speedup threshold that either performance policy must strictly exceed.
    #[arg(long, default_value = "1.05")]
    pub min_speedup: f64,
    /// TY/TLC process-tree peak-memory threshold for
    /// --policy parity-and-speed-and-memory.
    ///
    /// Qualifying Linux evidence uses cgroup-v2 `memory.peak`, which includes
    /// all cgroup-accounted memory and is broader than literal RSS. Values
    /// below 1 require TY to use proportionally less peak memory than TLC; the
    /// observed ratio must be strictly below this value.
    #[arg(long, default_value = "0.95")]
    pub max_memory_ratio: f64,
    /// Output artifact directory. Defaults under reports/perf/.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    /// Pre-built ty binary to run. Defaults to the current executable.
    #[arg(long)]
    pub ty_bin: Option<PathBuf>,
    /// TLC jar. Defaults to TYTOOLS_JAR, TLC_JAR, or ~/tlaplus/tytools.jar.
    #[arg(long)]
    pub tlc_jar: Option<PathBuf>,
    /// TLC executable wrapper. Defaults to TLC_BIN when set; otherwise Java+TLC jar is used.
    #[arg(long)]
    pub tlc_bin: Option<PathBuf>,
    /// CommunityModules jar for Java TLC classpath. Defaults to COMMUNITY_MODULES or ~/tlaplus/CommunityModules.jar when present.
    #[arg(long)]
    pub community_modules: Option<PathBuf>,
    /// TLA library directory injected into BOTH tools' module paths. Resolution
    /// order: this flag, `TLA_LIBRARY`, `TLA_PLUS_LIBRARY`, the installed upstream
    /// proof library (`~/tlaplus/tla-library`, from `ty install-tlc proof-library`),
    /// a system TLAPS install (`~/tlapm/library`), then the repo's first-party
    /// `test_specs/tla_library` stub set. Upstream outranks the stub because 25
    /// eligible corpus rows cannot be parsed by TLC without a proof library, and
    /// the claim should not depend on a TY-authored one. `ty corpus doctor` reports
    /// which resolved and whether it is strict-qualified.
    #[arg(long)]
    pub tla_library: Option<PathBuf>,
    /// Timeout per subprocess run in seconds.
    #[arg(long, default_value = "300")]
    pub timeout: u64,
    /// Extra ty check flag passed before --backend. May be repeated.
    #[arg(long = "ty-flag")]
    pub ty_flag: Vec<String>,
    /// TY env case name. May be repeated. Defaults to one `default` case.
    #[arg(long = "case")]
    pub cases: Vec<String>,
    /// Allowed non-semantic TY env case variable applied to every case, as KEY=VALUE. Currently only TY_PARALLEL_READONLY_VALUE_CACHES=0|1 is accepted. May be repeated.
    #[arg(long = "ty-env")]
    pub ty_env: Vec<String>,
    /// Allowed non-semantic TY env case variable for one case, as NAME:KEY=VALUE. Currently only TY_PARALLEL_READONLY_VALUE_CACHES=0|1 is accepted. May be repeated.
    #[arg(long = "case-env")]
    pub case_env: Vec<String>,
    /// Output format.
    #[arg(long, value_enum, default_value = "human")]
    pub format: SupremacyOutputFormat,
}

/// Arguments for the one-command single-thread reproduce umbrella.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyReproduceArgs {
    /// Corpus spec names to run. Defaults to a small fast built-in demo set
    /// (DiningPhilosophers, EWD840, DieHard) resolved from the baseline JSON.
    #[arg(long = "spec", num_args = 1..)]
    pub specs: Vec<String>,
    /// Reference tool to compare TY against: tlc (exhaustive), apalache (bounded/symbolic), or both.
    #[arg(long, value_enum, default_value = "tlc")]
    pub vs: SupremacyReproduceVs,
    /// Worker count for the single-thread comparison (default 1).
    #[arg(long, default_value = "1")]
    pub workers: usize,
    /// Bounded length passed to Apalache (--length); ignored for tlc.
    #[arg(long, default_value = "8")]
    pub len: usize,
    /// Per-subprocess timeout in seconds.
    #[arg(long, default_value = "300")]
    pub timeout: u64,
    /// Skip the auto-install of missing prerequisites (corpus / TLC / Apalache).
    #[arg(long)]
    pub no_install: bool,
    /// Output artifact directory. Defaults under reports/perf/.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
}

/// Arguments for the anti-overfit static scanner.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyAntiOverfitArgs {
    /// Supremacy policy JSON file. Defaults to tests/tlc_comparison/single_thread_supremacy_gate.json.
    #[arg(long)]
    pub policy: Option<PathBuf>,
    /// TLC baseline/matrix JSON file. Defaults to tests/tlc_comparison/spec_baseline.json.
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// Gate behavior: warn for reporting, enforce to fail on forbidden references.
    #[arg(long, value_enum, default_value = "enforce")]
    pub mode: SupremacyMode,
    /// Output format.
    #[arg(long, value_enum, default_value = "human")]
    pub format: SupremacyOutputFormat,
    /// Also treat matches in Rust comments as findings. Comments are ignored by default.
    #[arg(long)]
    pub include_comments: bool,
    /// Production Rust root to scan. Defaults to the runtime/backend source roots.
    #[arg(long = "scan-root", num_args = 1..)]
    pub scan_roots: Vec<PathBuf>,
}

/// Arguments for the baseline-backed all-runnable supremacy matrix.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyMatrixArgs {
    /// Baseline JSON to classify. Defaults to tests/tlc_comparison/spec_baseline.json.
    #[arg(long, default_value = "tests/tlc_comparison/spec_baseline.json")]
    pub baseline: PathBuf,
    /// Optional supremacy policy file with matrix_policy comparable-outcome opt-ins.
    #[arg(long)]
    pub policy: Option<PathBuf>,
    /// Gate behavior: warn for reporting, enforce to fail when any row is not a strict pass.
    #[arg(long, value_enum, default_value = "warn")]
    pub mode: SupremacyMode,
    /// Output format.
    #[arg(long, value_enum, default_value = "human")]
    pub format: SupremacyOutputFormat,
    /// Collect TLC and TY trust-codegen runtimes and write a refreshed baseline copy.
    ///
    /// With the default --runtime-scope missing-runtime, no --runtime-spec and
    /// no --runtime-limit runs every batchable missing-runtime row from the
    /// input baseline. Use --runtime-scope all-runnable or the
    /// matrix-full-suite subcommand for a complete batchable row refresh.
    #[arg(long)]
    pub refresh_runtime: bool,
    /// Runtime row set selected when --refresh-runtime has no --runtime-spec.
    #[arg(long = "runtime-scope", value_enum, default_value = "missing-runtime")]
    pub runtime_scope: SupremacyMatrixRuntimeScope,
    /// Directory for --refresh-runtime command artifacts and refreshed baseline output.
    /// Defaults to reports/perf/<UTC timestamp>-supremacy-matrix-runtime.
    #[arg(long)]
    pub runtime_output_dir: Option<PathBuf>,
    /// Maximum total sampled specs to collect in this invocation.
    ///
    /// Omit this flag for the one-command full selected suite. Set it only when
    /// chunking long refreshes; simulation and generate rows consume slots in
    /// the same hard cap as check-mode rows.
    #[arg(long)]
    pub runtime_limit: Option<usize>,
    /// Specific runnable spec to collect. Repeat to refresh named stale rows.
    #[arg(long = "runtime-spec")]
    pub runtime_specs: Vec<String>,
    /// Timeout per --refresh-runtime subprocess in seconds.
    #[arg(long, default_value = "300")]
    pub runtime_timeout: u64,
    /// Number of paired TLC/production-TY repetitions per check-mode row.
    /// Tool order alternates; strict PASS_BOTH evidence requires an even count
    /// of at least six so both launch orders have equal representation.
    #[arg(long, default_value = "6")]
    pub runtime_runs: usize,
    /// Also measure each refreshed check-mode row under the TY production-default
    /// configuration (AUTO routing and reductions, with no backend/reduction override)
    /// and record it separately from the exact count-verification arm.
    /// Speed classification then uses the production number while verified_match keeps
    /// using the pinned count-verify run. Roughly doubles per-row TY refresh cost; each
    /// production run respects --runtime-timeout. Disabling it leaves check rows
    /// without promotable strict performance evidence.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub production_runtime: bool,
    /// Pre-built ty binary for --refresh-runtime TY trust-codegen runs; preflighted with `ty check --backend trust_cg` before row collection.
    #[arg(long)]
    pub runtime_ty_bin: Option<PathBuf>,
    /// Allow a debug-profile TY binary for --refresh-runtime development smoke runs.
    #[arg(long)]
    pub allow_debug_runtime: bool,
    /// TLC jar for --refresh-runtime TLC runs. Defaults to TYTOOLS_JAR, TLC_JAR, or ~/tlaplus/tytools.jar.
    #[arg(long)]
    pub runtime_tlc_jar: Option<PathBuf>,
    /// CommunityModules jar for --refresh-runtime TLC classpath. Defaults to COMMUNITY_MODULES or ~/tlaplus/CommunityModules.jar when present.
    #[arg(long)]
    pub runtime_community_modules: Option<PathBuf>,
    /// TLA library directory injected into BOTH tools' module paths. Resolution
    /// order: this flag, `TLA_LIBRARY`, `TLA_PLUS_LIBRARY`, the installed upstream
    /// proof library (`~/tlaplus/tla-library`, from `ty install-tlc proof-library`),
    /// a system TLAPS install (`~/tlapm/library`), then the repo's first-party
    /// `test_specs/tla_library` stub set. Upstream outranks the stub because 25
    /// eligible corpus rows cannot be parsed by TLC without a proof library, and
    /// the claim should not depend on a TY-authored one. `ty corpus doctor` reports
    /// which resolved and whether it is strict-qualified.
    #[arg(long)]
    pub runtime_tla_library: Option<PathBuf>,
}

/// Arguments for explicit full-suite matrix runtime refresh.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyMatrixFullSuiteArgs {
    /// Baseline JSON to classify and refresh. Defaults to tests/tlc_comparison/spec_baseline.json.
    #[arg(long, default_value = "tests/tlc_comparison/spec_baseline.json")]
    pub baseline: PathBuf,
    /// Optional supremacy policy file with matrix_policy comparable-outcome opt-ins.
    #[arg(long)]
    pub policy: Option<PathBuf>,
    /// Gate behavior: warn for reporting, enforce to fail when any row remains non-passing.
    #[arg(long, value_enum, default_value = "warn")]
    pub mode: SupremacyMode,
    /// Output format.
    #[arg(long, value_enum, default_value = "human")]
    pub format: SupremacyOutputFormat,
    /// Directory for runtime command artifacts and refreshed baseline output.
    /// Defaults to reports/perf/<UTC timestamp>-supremacy-matrix-runtime.
    #[arg(long)]
    pub runtime_output_dir: Option<PathBuf>,
    /// Timeout per runtime-refresh subprocess in seconds.
    #[arg(long, default_value = "300")]
    pub runtime_timeout: u64,
    /// Number of paired TLC/production-TY repetitions per check-mode row.
    /// Tool order alternates; strict PASS_BOTH evidence requires an even count
    /// of at least six so both launch orders have equal representation.
    #[arg(long, default_value = "6")]
    pub runtime_runs: usize,
    /// Also measure each refreshed check-mode row under the TY production-default
    /// configuration (auto-POR/auto-symmetry free to engage). Default ON for the full
    /// suite so the speed axis reflects what users get; the pinned count-verify run
    /// still owns verified_match. Disable with --production-runtime false.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub production_runtime: bool,
    /// Pre-built ty binary for TY trust-codegen runs; preflighted with `ty check --backend trust_cg` before row collection.
    #[arg(long)]
    pub runtime_ty_bin: Option<PathBuf>,
    /// Allow a debug-profile TY binary for development smoke runs.
    #[arg(long)]
    pub allow_debug_runtime: bool,
    /// TLC jar for TLC runs. Defaults to TYTOOLS_JAR, TLC_JAR, or ~/tlaplus/tytools.jar.
    #[arg(long)]
    pub runtime_tlc_jar: Option<PathBuf>,
    /// CommunityModules jar for TLC classpath. Defaults to COMMUNITY_MODULES or ~/tlaplus/CommunityModules.jar when present.
    #[arg(long)]
    pub runtime_community_modules: Option<PathBuf>,
    /// TLA library directory injected into BOTH tools' module paths. Resolution
    /// order: this flag, `TLA_LIBRARY`, `TLA_PLUS_LIBRARY`, the installed upstream
    /// proof library (`~/tlaplus/tla-library`, from `ty install-tlc proof-library`),
    /// a system TLAPS install (`~/tlapm/library`), then the repo's first-party
    /// `test_specs/tla_library` stub set. Upstream outranks the stub because 25
    /// eligible corpus rows cannot be parsed by TLC without a proof library, and
    /// the claim should not depend on a TY-authored one. `ty corpus doctor` reports
    /// which resolved and whether it is strict-qualified.
    #[arg(long)]
    pub runtime_tla_library: Option<PathBuf>,
}

/// Arguments for a canonical, digest-bound segmented matrix campaign plan.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyMatrixCampaignPlanArgs {
    /// Baseline JSON whose complete batchable all-runnable row set is planned.
    #[arg(long, default_value = "tests/tlc_comparison/spec_baseline.json")]
    pub baseline: PathBuf,
    /// Optional supremacy policy file bound into the digest-bound campaign.
    #[arg(long)]
    pub policy: Option<PathBuf>,
    /// New absolute campaign-plan JSON path. Existing files are never overwritten.
    #[arg(long)]
    pub output: PathBuf,
    /// Fresh absolute root that owns every Rust evidence output in this campaign.
    #[arg(long)]
    pub artifact_root: PathBuf,
    /// Maximum number of complete rows assigned to each deterministic segment.
    #[arg(long, default_value = "1")]
    pub segment_size: usize,
    /// Required timeout per runtime subprocess, immutably bound by the campaign plan.
    #[arg(long)]
    pub runtime_timeout: u64,
    /// Number of paired repetitions per check-mode row, bound by the plan.
    #[arg(long, default_value = "6")]
    pub runtime_runs: usize,
    /// Require the production-default TY measurement arm.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub production_runtime: bool,
    /// Exact TY binary independently attested by path and SHA-256.
    #[arg(long)]
    pub runtime_ty_bin: Option<PathBuf>,
    /// TLC jar independently attested by path and SHA-256.
    #[arg(long)]
    pub runtime_tlc_jar: Option<PathBuf>,
    /// CommunityModules jar independently attested by path and SHA-256.
    #[arg(long)]
    pub runtime_community_modules: Option<PathBuf>,
    /// TLA library tree independently attested by path and tree SHA-256.
    #[arg(long)]
    pub runtime_tla_library: Option<PathBuf>,
    /// Sampled per-observation allocation limit. Kernel quota remains the hard guard.
    #[arg(long, default_value_t = 135_291_469_824)]
    pub max_observation_allocated_bytes: u64,
    /// Kernel-enforced per-observation allocated-byte ceiling.
    #[arg(long, default_value_t = 137_438_953_472)]
    pub hard_observation_allocated_bytes: u64,
    /// Sampled per-observation filesystem-entry limit.
    #[arg(long, default_value_t = 80_000)]
    pub max_observation_entries: u64,
    /// Kernel-enforced per-observation inode ceiling.
    #[arg(long, default_value_t = 90_000)]
    pub hard_observation_inodes: u64,
    /// Normal evidence-write ceiling for the segment evidence project.
    #[arg(long, default_value_t = 5_368_709_120)]
    pub evidence_soft_allocated_bytes: u64,
    /// Kernel-enforced evidence-project ceiling, including finalization reserve.
    #[arg(long, default_value_t = 6_442_450_944)]
    pub evidence_hard_allocated_bytes: u64,
    /// Normal evidence-project inode ceiling.
    #[arg(long, default_value_t = 10_000)]
    pub evidence_soft_inodes: u64,
    /// Kernel-enforced evidence-project inode ceiling.
    #[arg(long, default_value_t = 12_000)]
    pub evidence_hard_inodes: u64,
    /// Filesystem free-space floor maintained throughout every observation.
    #[arg(long, default_value_t = 80_530_636_800)]
    pub minimum_filesystem_available_bytes: u64,
    /// Filesystem free-space required before each observation is admitted.
    #[arg(long, default_value_t = 226_559_524_864)]
    pub minimum_prelaunch_available_bytes: u64,
    /// Filesystem free-inode floor maintained throughout every observation.
    #[arg(long, default_value_t = 1_000_000)]
    pub minimum_filesystem_available_inodes: u64,
    /// Filesystem free inodes required before assigning both project quotas.
    #[arg(long, default_value_t = 1_104_000)]
    pub minimum_prelaunch_available_inodes: u64,
    /// Disk-budget polling interval in milliseconds.
    #[arg(long, default_value_t = 50)]
    pub monitor_interval_ms: u64,
    /// Maximum retained stdout bytes for one observation.
    #[arg(long, default_value_t = 67_108_864)]
    pub stdout_max_bytes: u64,
    /// Maximum retained stderr bytes for one observation.
    #[arg(long, default_value_t = 67_108_864)]
    pub stderr_max_bytes: u64,
    /// First ext4 project ID in this campaign's unique even evidence/payload pair range.
    #[arg(long)]
    pub segment_project_id_start: u32,
}

/// Arguments for one plan-bound matrix campaign segment.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyMatrixSegmentArgs {
    /// Absolute canonical campaign plan created by matrix-campaign-plan.
    #[arg(long)]
    pub campaign_plan: PathBuf,
    /// Exact segment identifier from the campaign plan.
    #[arg(long)]
    pub segment_id: String,
    /// Fresh directory for this segment's four strict receipt artifacts.
    #[arg(long)]
    pub runtime_output_dir: Option<PathBuf>,
    /// Gate behavior. Strict evidence requires explicit enforce mode.
    #[arg(long, value_enum, default_value = "enforce")]
    pub mode: SupremacyMode,
    /// Output format.
    #[arg(long, value_enum, default_value = "human")]
    pub format: SupremacyOutputFormat,
}

/// Arguments for fail-closed aggregation of finalized campaign segments.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacyMatrixMergeArgs {
    /// Absolute canonical campaign plan shared by every segment.
    #[arg(long)]
    pub campaign_plan: PathBuf,
    /// Finalized segment runtime_evidence.json. Supply exactly one per planned segment.
    #[arg(long = "segment-report", num_args = 1..)]
    pub segment_reports: Vec<PathBuf>,
    /// Fresh directory for the aggregate's four strict receipt artifacts.
    #[arg(long)]
    pub runtime_output_dir: Option<PathBuf>,
    /// Gate behavior. Strict aggregate evidence requires explicit enforce mode.
    #[arg(long, value_enum, default_value = "enforce")]
    pub mode: SupremacyMode,
    /// Output format.
    #[arg(long, value_enum, default_value = "human")]
    pub format: SupremacyOutputFormat,
}

/// Arguments for the native production-default soundness sweep + triage.
#[derive(Clone, Debug, Args)]
pub(crate) struct SupremacySoundnessSweepArgs {
    /// Baseline JSON whose `ty.states` counts are the soundness reference. Defaults to tests/tlc_comparison/spec_baseline.json.
    #[arg(long, default_value = "tests/tlc_comparison/spec_baseline.json")]
    pub baseline: PathBuf,
    /// Pre-built ty binary run under pure production-default env.
    #[arg(long = "ty-bin", default_value = "./target/release/ty")]
    pub ty_bin: PathBuf,
    /// Base directory that relative baseline tla/cfg paths resolve against. `~` expands to $HOME.
    #[arg(long = "base-dir", default_value = "~/tlaplus-examples/specifications")]
    pub base_dir: PathBuf,
    /// Per-spec sweep timeout in seconds.
    #[arg(long = "timeout-secs", default_value = "130")]
    pub timeout_secs: u64,
    /// Per-spec triage timeout in seconds.
    #[arg(long = "triage-timeout-secs", default_value = "180")]
    pub triage_timeout_secs: u64,
    /// TSV output path. Defaults to reports/perf/<UTC timestamp>-soundness-sweep/results.tsv.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Restrict the sweep to these baseline spec names. Repeatable; empty runs every baseline spec sorted by name.
    #[arg(long = "specs", num_args = 1..)]
    pub specs: Vec<String>,
}

/// `ty supremacy` command family.
///
/// This is the authoritative Rust evidence/gate surface for TLC-vs-TY
/// comparison, all-runnable matrix classification, and single-thread trust-codegen
/// launch evidence. Shell or Python paths may delegate to this binary for
/// compatibility, but they must not implement independent corpus selection,
/// telemetry parsing, or verdict policy.
#[derive(Debug, Args)]
pub(crate) struct SupremacyArgs {
    #[command(subcommand)]
    pub command: SupremacyCommand,
}

#[derive(Debug, Subcommand)]
pub(crate) enum SupremacyCommand {
    /// Run bounded native-fused smoke readiness checks.
    #[command(
        long_about = "Run bounded native-fused smoke readiness checks.\n\nThis command produces diagnostic proof that selected specs can reach the Rust-owned native-fused path and emit bounded run artifacts without running the full launch benchmark. It is readiness evidence only; it is not launch acceptance evidence.\n\nThe Rust CLI is the authority for corpus selection, command construction, artifact layout, and exit status. Python helpers, shell wrappers, and JQ filters may call this command for compatibility, but they must preserve its artifacts and exit code and must not implement independent supremacy policy."
    )]
    Smoke(SupremacySmokeArgs),
    /// Run raw TLC/interpreter/trust_cg single-thread comparisons.
    #[command(
        long_about = "Run raw TLC/interpreter/trust-cg single-thread comparisons.\n\nThis command collects timing proof and summary artifacts for selected specs, including the JSON and Markdown consumed by `ty supremacy gate`. Benchmark-only output is analysis evidence; the Rust gate remains the authority for launch verdicts.\n\nThe Rust CLI owns command construction, backend selection, artifact schema, and exit status. Python helpers, shell wrappers, and JQ filters are compatibility surfaces only when they delegate to this command and preserve its artifacts and exit code."
    )]
    Benchmark(SupremacyBenchmarkArgs),
    /// Apply the Rust supremacy policy gate to launch evidence.
    ///
    /// Without --summary-json, this command runs the Rust benchmark collector
    /// and then evaluates the produced summary. With --summary-json, it runs
    /// only the Rust policy verdict engine over an existing benchmark or
    /// matrix-summary artifact.
    #[command(
        long_about = "Apply the Rust supremacy policy gate to launch evidence.\n\nWithout --summary-json, this command runs the Rust benchmark collector and then evaluates the produced summary. With --summary-json, it runs only the Rust policy verdict engine over an existing benchmark summary. Matrix summaries (`ty.supremacy.matrix_summary.v1`) are validated as all-runnable diagnostic evidence, but they cannot satisfy the current launch-corpus gate.\n\nThis is the authoritative three-spec single-thread launch-corpus gate. For launch acceptance, use --mode enforce --gate-mode full-native-fused --runs 3. Wrapper scripts, Python helpers, JQ filters, smoke runs, benchmark-only runs, matrix summaries, and interim action-only mode are diagnostic or compatibility surfaces only."
    )]
    Gate(SupremacyGateArgs),
    /// Run a Rust TLC-vs-TY comparison gate over baseline or explicit specs.
    #[command(
        long_about = "Run the Rust TLC-vs-TY comparison gate for targeted parity, runtime, and process-tree peak-memory diagnostics.\n\nThis command replaces Python perf-gate scripts for baseline-backed or explicit TLA+/cfg pairs. It runs paired repetitions with alternating TLC/TY order, writes every raw run to compare.json, and gates on the median within-pair ratio. Enforced performance policies require an even count of at least six pairs so both launch orders have equal representation. `--backend auto` measures genuine AUTO/default routing without a child `--backend` flag or environment pins; it is GPU-eligible and therefore a separate hardware track. `--backend auto-cpu` adds `--no-gpu`. Both AUTO modes use a separate `--bfs-only --no-reduction --workers 1` count-verification arm. Enforced performance diagnostics also require the auditable Java TLC runner, exact count-arm parity, runtime speedup strictly greater than 1.05 by default, and TY/TLC process-tree peak memory strictly less than 0.95 by default.\n\nThis remains targeted diagnostic evidence and is not admitted by the tightened strict Linux launcher. Broad strict-superiority acceptance requires the complete pinned all-runnable matrix artifact; the three-spec native-fused gate remains a separate hot-path release check. Any compatibility wrapper around this command must preserve its exit code and must not implement independent corpus selection, parsing, or verdict policy."
    )]
    Compare(SupremacyCompareArgs),
    /// One-command reproduce of the single-thread TY-vs-TLC(+Apalache) runtime+memory comparison.
    #[command(
        long_about = "Reproduce the single-thread TY-vs-TLC (and optionally TY-vs-Apalache) runtime+memory comparison with one command, and print an HONEST scorecard.\n\nThis is a CLI-orchestration umbrella: it ensures prerequisites (downloads the corpus with `ty corpus fetch`, installs TLC with `ty install-tlc install`, installs Apalache with `ty install-apalache install` — each skipped when already present, or all skipped with --no-install), then runs the existing `ty supremacy compare` (single-thread TLC-vs-TY, wall-time + peak RSS, gated on verdict/state parity) and/or the cross-platform `scripts/ty_vs_apalache_memtime.sh` Apalache differential, over a small fast demo spec set (override with --spec).\n\nThe scorecard reports the REAL per-spec verdict-parity status and TY-vs-tool time and memory ratios. TY does NOT win every spec: wins and structural losses are both shown truthfully, with the comparability caveats surfaced (Apalache is bounded by --len and symbolic, so it is verdict-parity only and corroborates a TY \"ok\" only up to the bound; some specs are non-comparable because TLC/Apalache cannot parse or type them). This is reproduction evidence, not the launch-acceptance gate."
    )]
    Reproduce(SupremacyReproduceArgs),
    /// Scan production runtime/backend code for exact launch-corpus overfit references.
    #[command(
        name = "anti-overfit",
        long_about = "Scan production runtime/backend code for exact launch-corpus overfit references.\n\nThis command produces static proof artifacts showing whether Rust runtime and backend code contains forbidden exact corpus references from the supremacy policy or baseline. It is an anti-overfit guard for launch evidence, not a benchmark runner.\n\nThe Rust CLI owns policy parsing, baseline parsing, scan-root defaults, comment handling, artifact schema, and failure policy. Python helpers, shell wrappers, and JQ filters are compatibility surfaces only when they delegate to this command and preserve its artifacts and exit code."
    )]
    AntiOverfit(SupremacyAntiOverfitArgs),
    /// Classify every spec baseline row against strict TLC speed-supremacy requirements.
    #[command(
        long_about = "Classify every spec baseline row against strict TLC speed-supremacy requirements.\n\nThis command produces all-runnable matrix proof and artifacts from the TLC baseline, optionally refreshing TLC and TY trust-codegen runtimes before writing a refreshed baseline copy. Matrix output is broad audit evidence; it is not the three-spec launch-corpus acceptance gate.\n\nThe Rust CLI owns baseline parsing, comparable-outcome policy, runtime artifact layout, refreshed baseline output, and verdict exit status. Python helpers, shell wrappers, and JQ filters are compatibility surfaces only when they delegate to this command and preserve its artifacts and exit code."
    )]
    Matrix(SupremacyMatrixArgs),
    /// Explicitly refresh the full batchable all-runnable matrix runtime suite.
    #[command(
        name = "matrix-full-suite",
        long_about = "Explicitly refresh the full batchable all-runnable matrix runtime suite.\n\nThis is a Rust convenience alias for `ty supremacy matrix --refresh-runtime --runtime-scope all-runnable` with no --runtime-limit and no --runtime-spec values: every batchable row with a runnable source mode is selected, and the refreshed baseline is written under --runtime-output-dir, or the timestamped reports/perf default when --runtime-output-dir is omitted. It is broad all-runnable matrix refresh/audit evidence, not launch acceptance evidence. Use `ty supremacy matrix --refresh-runtime --runtime-scope all-runnable --runtime-limit <N>` for intentionally chunked refreshes; in chunked runs, simulation and generate rows consume slots in the same hard cap as check-mode rows."
    )]
    MatrixFullSuite(SupremacyMatrixFullSuiteArgs),
    /// Write an exclusively created, digest-bound row-segment campaign plan.
    #[command(
        name = "matrix-campaign-plan",
        long_about = "Write a canonical, digest-bound plan for a segmented full all-runnable matrix campaign. The exclusively created plan fixes the input baseline and policy, runtime contract, independently attested source revision and tool/input hashes, complete batchable row order, blocked rows, and deterministic segment membership. The plan is preparation metadata, not runtime evidence or a superiority claim."
    )]
    MatrixCampaignPlan(SupremacyMatrixCampaignPlanArgs),
    /// Collect one complete, plan-bound row segment.
    #[command(
        name = "matrix-segment",
        long_about = "Collect exactly one row segment named by a digest-bound matrix campaign plan. Rows and runtime/tool inputs cannot be overridden on this command. A successful segment proves only its selected-row collection and remains explicitly ineligible to claim full-corpus completion or superiority until every finalized segment is validated by matrix-merge."
    )]
    MatrixSegment(SupremacyMatrixSegmentArgs),
    /// Seal the complete campaign inventory without claiming superiority.
    #[command(
        name = "matrix-merge-inventory",
        long_about = "Fail-closed integrity merge of exactly one finalized report for every planned segment. It receipt-seals a complete current loser inventory and exits successfully after coverage/integrity completion even when blockers remain. It always records corpus_claim_pass=false and is categorically inadmissible as a public superiority baseline."
    )]
    MatrixMergeInventory(SupremacyMatrixMergeArgs),
    /// Merge an exact set of finalized campaign segments.
    #[command(
        name = "matrix-merge",
        long_about = "Fail-closed merge of exactly one finalized, receipt-bound report for every segment in a matrix campaign plan. The merge rejects missing, duplicate, overlapping, foreign, stale, or contract-mismatched segments, reconstructs the refreshed baseline from the original plan baseline and segment rows, and is the only segmented-campaign command eligible to claim full-corpus completion or pass."
    )]
    MatrixMerge(SupremacyMatrixMergeArgs),
    /// Validate that the candidate preserves soundness vs baselines on two axes: exact reachability (POR off) and safety verdict (production default).
    #[command(
        name = "soundness-sweep",
        long_about = "Validate that a candidate `ty` binary preserves model-checking soundness vs the recorded baselines, WITHOUT conflating sound auto-POR state-count reductions with real regressions. Each spec is run twice with `--workers 1 --backend trust-cg` and every TY_* knob stripped:\n\n1. EXACT-COUNT CHECK (primary): every sound count-reducing default FORCED OFF (the --no-reduction flag: auto-POR + auto-symmetry off). The final `States found:` total must equal the recorded raw `ty.states` baseline. Equal -> COUNT_OK; a violation-halting spec (order-dependent count-to-first-violation) -> VIOLATION_DELTA (benign); otherwise -> COUNT_REGRESSION (a genuine reachability regression).\n\n2. VERDICT CHECK (secondary): production default (auto-POR and auto-symmetry free to engage). Only the safety VERDICT is compared (invariant/property holds vs violated). Production default may explore FEWER states under POR or symmetry orbit reduction, so counts are NOT compared here. Match -> VERDICT_OK; differ -> VERDICT_REGRESSION (a POR-soundness regression). The per-spec POR count reduction is recorded as informational only.\n\nOverall verdict: PASS iff zero COUNT_REGRESSION and zero VERDICT_REGRESSION.\n\nThis is soundness/audit evidence only; it is not launch acceptance evidence. The Rust CLI owns corpus selection, env control, state-count parsing (final total, never the mid-run Progress checkpoint), verdict policy, and classification."
    )]
    SoundnessSweep(SupremacySoundnessSweepArgs),
}

/// Output format for profile command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ProfileOutputFormat {
    /// Human-readable profiling report with bars (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for diff command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DiffOutputFormat {
    /// Human-readable colored diff output (default)
    #[default]
    Human,
    /// Structured JSON output for tooling
    Json,
}

/// Input format for convert command
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ConvertFrom {
    /// TLA+ source file
    Tla,
    /// JSON AST
    Json,
    /// JSON trace output from `ty check --output json`
    Trace,
}

/// Output format for convert command
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ConvertTo {
    /// Structured JSON AST
    Json,
    /// TLA+ source
    Tla,
    /// GitHub-Flavored Markdown documentation
    Markdown,
    /// Aligned table of trace states
    Table,
}

/// Output format for repair command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum RepairOutputFormat {
    /// Human-readable repair suggestions (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Solver engine for the AIGER subcommand.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AigerEngine {
    /// SAT-based portfolio: IC3/PDR + BMC + k-induction (default, faster for most benchmarks).
    #[default]
    Sat,
    /// CHC-based: translates to CHC and uses ay-chc adaptive portfolio solver.
    Chc,
    /// BMC only: bounded model checking (finds bugs, cannot prove safety).
    Bmc,
    /// k-induction only: can prove safety for k-inductive properties.
    Kind,
    /// Strengthened k-induction: k-induction with auxiliary invariant discovery.
    /// Extends standard k-induction with single-literal invariant discovery,
    /// BMC-based confirmation, and counterexample-guided strengthening.
    KindStrengthened,
    /// IC3/PDR only: full invariant-based safety checking.
    Ic3,
}

/// Portfolio mode for the SAT engine.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AigerPortfolio {
    /// Default: conservative IC3 + BMC + k-induction (3 threads).
    #[default]
    Default,
    /// Full IC3 portfolio: 8 IC3 configs + BMC + k-induction (10 threads).
    Full,
    /// Competition: 13 IC3 configs + 3 BMC + k-induction (17 threads).
    Competition,
    /// Adaptive preset rotation: when a worker returns Unknown, restart it
    /// with the next preset and a rotated random seed (rIC3 PolyNexus port).
    /// Target: recover from stuck IC3 runs on hard benchmarks.
    Adaptive,
}

/// Output format for deps command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DepsOutputFormat {
    /// Indented text tree showing the dependency hierarchy (default)
    #[default]
    Tree,
    /// Graphviz DOT format for graph visualization
    Dot,
    /// Structured JSON output for tooling integration
    Json,
}

/// Output format for doc command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DocOutputFormat {
    /// GitHub-Flavored Markdown (default)
    #[default]
    Markdown,
    /// Self-contained HTML with CSS styling and navigation sidebar
    Html,
    /// Structured JSON output for tooling
    Json,
}

/// Output format for graph command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum GraphOutputFormat {
    /// Graphviz DOT format (default)
    #[default]
    Dot,
    /// Mermaid.js flowchart syntax for GitHub markdown rendering
    Mermaid,
    /// Structured JSON adjacency list for programmatic consumption
    Json,
}

/// Output format for snapshot command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SnapshotOutputFormat {
    /// Human-readable colored summary table (default)
    #[default]
    Human,
    /// Structured JSON regression report
    Json,
}

/// Output format for bisect command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum BisectOutputFormat {
    /// Human-readable progress and result output (default)
    #[default]
    Human,
    /// Structured JSON bisect report
    Json,
}

/// Output format for merge command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum MergeOutputFormat {
    /// Human-readable merged TLA+ source with optional conflict markers (default)
    #[default]
    Human,
    /// Structured JSON merge report
    Json,
}

/// Output format for thread-check command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ThreadCheckOutputFormat {
    /// Human-readable terminal output (default)
    #[default]
    Human,
    /// Structured JSON output (ConcurrentCheckResult)
    Json,
}

/// Output format for validate command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ValidateOutputFormat {
    /// Human-readable colored terminal output (default)
    #[default]
    Human,
    /// Structured JSON validation report
    Json,
}

/// Output format for coverage command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CoverageOutputFormat {
    /// Human-readable table with ASCII bar chart (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for `ty trace view` command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TraceViewOutputFormat {
    /// Human-readable colored output with change markers (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
    /// Aligned columns table
    Table,
}

/// Output format for stats command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum StatsOutputFormat {
    /// Human-readable table output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// What kind of TLA+ entity to search for.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SearchKind {
    /// Find operator definitions matching a name pattern.
    #[default]
    Operator,
    /// Find variable declarations matching a pattern.
    Variable,
    /// Find constant declarations matching a pattern.
    Constant,
    /// Find expressions matching a text pattern in operator bodies.
    Expr,
    /// Find Next disjuncts (actions) matching a pattern.
    Action,
}

/// Output format for search command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SearchOutputFormat {
    /// Grep-like colored terminal output (default).
    #[default]
    Human,
    /// Structured JSON output.
    Json,
}

/// Output format for lint command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum LintOutputFormat {
    /// Human-readable colored terminal output (default)
    #[default]
    Human,
    /// Structured JSON output for IDE integration
    Json,
}

/// Minimum severity level for lint output
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum LintSeverityArg {
    /// Show warnings and above (default)
    #[default]
    Warning,
    /// Show all diagnostics including informational messages
    Info,
}

/// Output format for typecheck command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TypecheckOutputFormat {
    /// Human-readable text output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for explain command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ExplainOutputFormat {
    /// Human-readable step-by-step explanation (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for summary command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SummaryOutputFormat {
    /// Human-readable colored table (default)
    #[default]
    Human,
    /// Structured JSON array of results
    Json,
    /// CSV format for spreadsheets
    Csv,
}

/// Output format for check-summary command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CheckSummaryOutputFormat {
    /// Tab-separated fields for shell harnesses (default)
    #[default]
    Tsv,
    /// Structured JSON object
    Json,
}

/// Sort order for summary command
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SummarySortBy {
    /// Sort by spec name alphabetically (default)
    #[default]
    Name,
    /// Sort by wall-clock time (slowest first)
    Time,
    /// Sort by state count (most states first)
    States,
    /// Sort by status (errors first, then timeouts, then passes)
    Status,
}

/// Protocol template kind for `ty template`.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum TemplateKind {
    /// Peterson's mutual exclusion
    Mutex,
    /// Simple consensus with proposals and votes
    Consensus,
    /// MSI-like cache coherence protocol
    Cache,
    /// Bounded FIFO queue with producer/consumer
    Queue,
    /// Leader election in a ring
    Leader,
    /// Token passing ring protocol
    TokenRing,
}

/// Deadlock analysis mode for `ty deadlock`.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DeadlockMode {
    /// Static analysis only (fast, no model checking)
    #[default]
    Quick,
    /// Full model checking for deadlock states
    Full,
}

/// Output format for deadlock command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DeadlockOutputFormat {
    /// Human-readable output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for abstract command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AbstractOutputFormat {
    /// Human-readable structured text (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
    /// Mermaid state diagram
    Mermaid,
}

/// Detail level for abstract command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AbstractDetail {
    /// Just action names
    Brief,
    /// Actions + affected variables (default)
    #[default]
    Normal,
    /// Everything including expressions
    Full,
}

/// Output format for witness command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum WitnessOutputFormat {
    /// Human-readable step-by-step trace (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for compare command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CompareOutputFormat {
    /// Diff-like colored output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for scope command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ScopeOutputFormat {
    /// Human-readable tree output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
    /// Graphviz DOT graph
    Dot,
}

/// Strategy for the constrain command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ConstrainStrategy {
    /// Suggest smallest constants that exercise all actions (default)
    #[default]
    Minimize,
    /// Generate incrementally larger configs (N=1, N=2, ...)
    Incremental,
    /// Detect and add SYMMETRY declarations
    Symmetric,
}

/// Output format for audit command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AuditOutputFormat {
    /// Human-readable scorecard (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Import source format for `ty import`.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ImportFormat {
    /// JSON state machine description
    JsonStateMachine,
    /// Promela (SPIN) model
    Promela,
    /// Alloy specification
    Alloy,
}

/// Output format for symmetry command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SymmetryOutputFormat {
    /// Human-readable output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for partition command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum PartitionOutputFormat {
    /// Human-readable output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for sim-report command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SimReportOutputFormat {
    /// Human-readable table output (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for trace-gen command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TraceGenOutputFormat {
    /// Human-readable trace listing (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
    /// Informal Trace Format (Apalache-compatible)
    Itf,
}

/// Trace generation mode.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TraceGenMode {
    /// Find a trace reaching a state where a target predicate holds
    #[default]
    Target,
    /// Find a minimal set of traces covering every Next disjunct
    Coverage,
    /// Generate diverse random traces for testing
    Random,
}

/// Output format for inv-gen command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum InvGenOutputFormat {
    /// Human-readable summary with explanations (default)
    #[default]
    Human,
    /// Machine-readable JSON array of candidate invariants
    Json,
    /// TLA+ operator definitions ready to paste into a spec
    Tla,
}

/// Output format for action-graph command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ActionGraphOutputFormat {
    /// Human-readable summary (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
    /// GraphViz DOT format for visualization
    Dot,
}

/// Output format for refine command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum RefineOutputFormat {
    /// Human-readable summary (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for model-diff command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ModelDiffOutputFormat {
    /// Human-readable colored diff (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for state-filter command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum StateFilterOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for unfold command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum UnfoldOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for project command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ProjectOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for bound command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum BoundOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for slice command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SliceOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for reach command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ReachOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for compose command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ComposeOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for census command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CensusOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for equiv command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum EquivOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for induct command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum InductOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for lasso command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum LassoOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for assume-guarantee command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AssumeGuaranteeOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for predicate-abs command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum PredicateAbsOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for sandbox command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SandboxOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for timeline command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TimelineOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for metric command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum MetricOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for scaffold command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ScaffoldOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for stutter command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum StutterOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for quorum command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum QuorumOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for fingerprint command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum FingerprintOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for normalize command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum NormalizeOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for countex command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CountexOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for heatmap command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum HeatmapOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for protocol command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ProtocolOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for hierarchy command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum HierarchyOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for crossref command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CrossrefOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for invariantgen command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum InvariantgenOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for drift command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DriftOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for safety command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SafetyOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for liveness-check command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum LivenesscheckOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for translate command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TranslateOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for tableau command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TableauOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for alphabet command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AlphabetOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for weight command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum WeightOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for absorb command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AbsorbOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for cluster command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ClusterOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for rename command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum RenameOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for reachset command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ReachsetOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for guard command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum GuardOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for symmetry-detect command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SymmetrydetectOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for deadlock-free command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DeadlockfreeOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for action-count command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ActioncountOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for trust_cg-coverage command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TrustCgCoverageOutputFormat {
    /// Human-readable summary (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for const-check command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ConstcheckOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for spec-info command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SpecinfoOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for var-track command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum VartrackOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for cfg-gen command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CfggenOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for dep-graph command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum DepgraphOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
    /// DOT graph format
    Dot,
}

/// Output format for init-count command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum InitcountOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for branch-factor command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum BranchfactorOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for state-graph command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum StategraphOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for predicate command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum PredicateOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for module-info command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ModuleinfoOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for op-arity command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum OparityOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for unused-var command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum UnusedvarOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for expr-count command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ExprcountOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for spec-size command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SpecsizeOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for const-list command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ConstlistOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for var-list command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum VarlistOutputFormat {
    /// Human-readable table (default)
    #[default]
    Human,
    /// Structured JSON output
    Json,
}

/// Output format for unused-const command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum UnusedconstOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for ast-depth command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum AstdepthOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for op-list command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum OplistOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for extends command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ExtendsOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for set-ops command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum SetopsOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for quant-count command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum QuantcountOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for prime-count command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum PrimecountOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for if-count command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum IfcountOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for let-count command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum LetcountOutputFormat {
    #[default]
    Human,
    Json,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ChoosecountOutputFormat {
    #[default]
    Human,
    Json,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum CasecountOutputFormat {
    #[default]
    Human,
    Json,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum RecordopsOutputFormat {
    #[default]
    Human,
    Json,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TemporalopsOutputFormat {
    #[default]
    Human,
    Json,
}
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum UnchangedOutputFormat {
    #[default]
    Human,
    Json,
}
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum EnabledOutputFormat {
    #[default]
    Human,
    Json,
}

/// Output format for model checking results
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum OutputFormat {
    /// Human-readable text output (default)
    #[default]
    Human,
    /// Structured JSON output for AI agents and automation
    Json,
    /// Streaming JSONL format (one JSON object per line)
    #[value(name = "jsonl", alias = "json-lines")]
    Jsonl,
    /// TLC "-tool" tagged output for Eclipse Toolbox compatibility
    #[value(name = "tlc-tool", alias = "tool")]
    TlcTool,
    /// Informal Trace Format (ITF) JSON for Apalache/TLA+ tooling interoperability
    ///
    /// Emits the full check result as an ITF JSON document to stdout.
    /// On error (invariant/property/liveness violation, deadlock), the counterexample
    /// trace is encoded in ITF format. On success, a minimal ITF document with an
    /// empty states array is emitted.
    Itf,
}

/// Format for counterexample traces
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TraceFormat {
    /// Human-readable text format (default)
    #[default]
    Text,
    /// GraphViz DOT format for visualization
    Dot,
    /// Informal Trace Format (ITF) JSON for Apalache/TLA+ tooling interoperability
    Itf,
}

/// Speculative type specialization mode for JIT Tier 2 recompilation.
///
/// Controls whether the JIT type profiler observes runtime types of state
/// variables during BFS warmup and builds a specialization plan for Tier 2
/// recompilation.
// JIT V2 speculative type specialization.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum TypeSpecializeArg {
    /// Enable type specialization when JIT is active and profiling detects
    /// monomorphic state variables (default).
    #[default]
    Auto,
    /// Always enable type specialization profiling (even without --jit).
    On,
    /// Disable type specialization entirely.
    Off,
}

/// Post-BFS liveness execution strategy.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum LivenessModeArg {
    /// Reuse the BFS-cached successor graph during post-BFS liveness.
    #[default]
    Full,
    /// Regenerate system successors lazily during product exploration.
    #[value(name = "on-the-fly", alias = "on_the_fly")]
    OnTheFly,
}

/// Maximum allowed soundness mode for this run (automation gate).
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum SoundnessGate {
    /// Require a path that satisfies the checker soundness gate.
    #[default]
    Sound,
    /// Allow experimental engine paths.
    Experimental,
    /// Allow heuristic / incomplete engine paths.
    Heuristic,
}

/// Named verification strategy for multi-phase pipeline mode.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum StrategyArg {
    /// Fast feedback: RandomWalk + BMC (no exhaustive BFS).
    Quick,
    /// RandomWalk + exhaustive BFS + liveness within configured budgets.
    Full,
    /// Adaptive: walk -> BMC -> k-induction -> PDR -> BFS with early exit.
    Auto,
}

/// Exploration mode for the `explore` subcommand.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ExploreModeArg {
    /// Interactive terminal REPL (default).
    #[default]
    Repl,
    /// HTTP JSON API server.
    Http,
}

/// Exploration engine for the `explore` subcommand.
///
/// Controls whether the server uses concrete-state enumeration or
/// symbolic SMT-based exploration.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(crate) enum ExploreEngineArg {
    /// Concrete-state exploration (default).
    #[default]
    Concrete,
    /// Symbolic SMT-based exploration (requires ay feature).
    Symbolic,
}

#[derive(Debug, Parser)]
#[command(
    name = "ty",
    version,
    about = "Model check, prove, and certify TLA+ specs — a fast, TLC-compatible verifier",
    long_about = "ty is one binary for the whole verification workflow. Author and analyze TLA+ \
        specs. Model-check them explicitly or symbolically (BMC, k-induction, IC3/PDR) — a \
        drop-in TLC replacement. Explain, shrink, and repair counterexamples. Emit \
        independently re-checkable proof certificates. The same engines check AIGER/BTOR2 \
        hardware and PNML Petri nets (MCC).\n\n\
        Start with `ty init` to scaffold a project, or `ty check Spec.tla` if you already \
        have one.\n\n\
        Run `ty <command> --help` for details on any command.",
    after_help = crate::catalog::epilogue(),
    after_long_help = crate::catalog::after_long_help()
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

// This is the top-level clap subcommand enum; clap owns its construction and a
// large variant is unavoidable for the CLI surface. Boxing would complicate every
// match arm without a meaningful size win for a short-lived parse-time value.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Create a TLA+ specification project from a template
    ///
    /// Generates a `.tla` module file and a `.cfg` configuration file
    /// with boilerplate for the chosen template (basic, distributed,
    /// mutex, or cache).
    #[command(display_order = 10)]
    Init {
        /// Specification name (becomes the TLA+ module name and file prefix).
        name: String,
        /// Template to use for scaffolding.
        #[arg(short, long, value_enum, default_value = "basic")]
        template: InitTemplate,
        /// Output directory (default: current directory).
        #[arg(short, long, default_value = ".")]
        dir: PathBuf,
        /// Overwrite existing files.
        #[arg(long)]
        force: bool,
    },
    /// Parse a TLA+ source file and report syntax errors
    ///
    /// Syntax-checks a single `.tla` file and prints any parse errors
    /// found.
    #[command(display_order = 11)]
    Parse { file: PathBuf },
    /// Parse and lower a TLA+ file, then dump the lowered AST
    ///
    /// Developer tool: prints the lowered AST in Rust `Debug` formatting.
    #[command(hide = true)]
    Ast {
        file: PathBuf,
        /// Dump the lowered TIR instead of the lowered AST.
        #[arg(long, hide = true)]
        tir: bool,
    },
    /// Format a TLA+ source file
    ///
    /// By default, prints the formatted output to stdout.
    /// Use `--write` to modify the file in place, `--check` to verify
    /// formatting, or `--diff` to preview changes as a unified diff.
    #[command(display_order = 12)]
    Fmt {
        /// TLA+ source file(s) to format. Use `-` for stdin.
        files: Vec<PathBuf>,
        /// Write formatted output back to the source file(s) in place.
        #[arg(short, long, conflicts_with = "diff")]
        write: bool,
        /// Number of spaces per indentation level (default: 2).
        #[arg(long, default_value = "2")]
        indent: usize,
        /// Target maximum line width before expressions break to multiple lines.
        #[arg(long, default_value = "80")]
        width: usize,
        /// Check formatting without modifying files. Exit with code 1 if unformatted.
        #[arg(long, conflicts_with_all = ["write", "diff"])]
        check: bool,
        /// Show unified diff of what would change (implies no writes).
        #[arg(long, conflicts_with_all = ["write", "check"])]
        diff: bool,
    },
    /// Model check a TLA+ specification
    ///
    /// By default runs exhaustive verification with a fused explicit-state
    /// BFS + symbolic engine (CDEMC) and automatic backend selection.
    /// Checks invariants, deadlock (disable with `--no-deadlock`), and
    /// temporal properties under the spec's declared fairness, all from
    /// the `.cfg`. Pass `--bfs-only` for pure explicit-state BFS matching
    /// TLC behavior. (Builds without the `ay` feature default to pure BFS.)
    ///
    /// Automatic symmetry and partial-order reduction are ON by default,
    /// so distinct-state counts may differ from TLC; `--no-reduction`
    /// gives TLC-identical counts. A SYMMETRY declared in the `.cfg` is
    /// honored for safety but ignored (soundly, with a warning) when a
    /// property requires genuine liveness — TLC applies it anyway and can
    /// report wrong verdicts; set TY_MATCH_DECLARED_SYMMETRY=1 to match
    /// TLC. `--auto-symmetry`/`--no-auto-symmetry` control only the
    /// automatic reduction, not the declared SYMMETRY.
    ///
    /// Config files: the standard TLC `.cfg` statements are honored —
    /// SPECIFICATION, INIT/NEXT, INVARIANT(S), PROPERTY(IES),
    /// CONSTANT(S) (assignments, model values, and `<-` replacements),
    /// CONSTRAINT(S), ACTION_CONSTRAINT(S), SYMMETRY, VIEW, ALIAS,
    /// POSTCONDITION, and CHECK_DEADLOCK. The Config-overrides flags
    /// (`--init/--next/--inv/--prop/--const`) replace the corresponding
    /// statements, and `--no-config` runs without any `.cfg`.
    ///
    /// Alternative engines: bounded model checking (BMC), PDR,
    /// k-induction, compiled, GPU, portfolio, and pipeline. Fingerprint
    /// storage tiers and checkpoints are set by the flag groups below.
    #[command(display_order = 20)]
    Check {
        /// TLA+ source file to check.
        ///
        /// Supports `.tla` (TLA+ source) and `.qnt.json` (Quint IR) files.
        /// For `.qnt.json` files, the `--quint` flag is automatically inferred.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        ///
        /// When no `.cfg` exists, supply the model on the command line
        /// instead: `--init`/`--next` (with `--inv`/`--prop`/`--const` as
        /// needed) run without a config file, and `--no-config` skips the
        /// `.cfg` lookup entirely, defaulting to the convention names
        /// `Init` and `Next`. See the "Config overrides" section below.
        #[arg(display_order = 0, short, long)]
        config: Option<PathBuf>,
        /// Override the Init operator (replaces INIT from .cfg).
        ///
        /// With --next, allows running without a .cfg file.
        #[arg(help_heading = "Config overrides", long, value_name = "OPERATOR")]
        init: Option<String>,
        /// Override the Next operator (replaces NEXT from .cfg).
        ///
        /// With --init, allows running without a .cfg file.
        #[arg(help_heading = "Config overrides", long, value_name = "OPERATOR")]
        next: Option<String>,
        /// Override invariants to check (replaces INVARIANT from .cfg).
        ///
        /// Repeatable: --inv TypeOK --inv Safety.
        #[arg(
            help_heading = "Config overrides",
            long = "inv",
            value_name = "INVARIANT"
        )]
        invariants: Vec<String>,
        /// Override temporal properties to check (replaces PROPERTY from .cfg).
        ///
        /// Repeatable: --prop Liveness --prop Fairness.
        #[arg(
            help_heading = "Config overrides",
            long = "prop",
            value_name = "PROPERTY"
        )]
        properties: Vec<String>,
        /// Override constant assignments (replaces CONSTANT from .cfg).
        ///
        /// VALUE is a TLA+ expression. Repeatable:
        /// --const N=3 --const "Procs={p1,p2}".
        #[arg(
            help_heading = "Config overrides",
            long = "const",
            value_name = "NAME=VALUE"
        )]
        constants: Vec<String>,
        /// Skip .cfg file entirely; use only CLI flags and convention defaults.
        ///
        /// Combine with --init/--next/--inv for config-free checking.
        /// Without --init/--next, convention names "Init" and "Next" are used.
        #[arg(help_heading = "Config overrides", long, conflicts_with = "config")]
        no_config: bool,
        /// Compile the TLA+ spec to native Rust code, build, and run
        /// instead of interpreting.
        ///
        /// Generates Rust source via TIR codegen, builds a temporary Cargo
        /// project with `cargo build --release`, and runs the binary, which
        /// performs BFS checking and reports in the interpreter's format.
        ///
        /// Can be much faster than interpretation for large state spaces.
        ///
        /// Requires: Rust toolchain (cargo/rustc).
        #[arg(help_heading = "Engine selection", long)]
        compiled: bool,
        /// Force the CUDA GPU engine (testing lever; skips the CPU probe).
        ///
        /// The GPU engine is part of the default AUTO engine family: when a
        /// CUDA device is present and the spec admits (safety-only, flat i64
        /// slot layout, every action + invariant lowers to self-contained
        /// native code), a bounded CPU probe sizes the state space and large
        /// spaces run on the GPU automatically. Admission is fail-closed:
        /// anything that does not qualify falls back to the CPU engines with
        /// a printed reason, verdict-neutral.
        ///
        /// Requires: an NVIDIA GPU with a CUDA driver (libcuda) and NVRTC
        /// (libnvrtc) available at runtime. No CUDA toolchain is needed at
        /// build time.
        #[arg(help_heading = "Engine selection", long, hide_short_help = true)]
        gpu: bool,
        /// Disable the GPU engine (testing lever; the default AUTO selection
        /// then never considers the GPU).
        #[arg(
            help_heading = "Engine selection",
            long,
            conflicts_with = "gpu",
            hide_short_help = true
        )]
        no_gpu: bool,
        /// Parse the input as Quint JSON IR instead of TLA+ source.
        ///
        /// Automatically enabled when the file extension is `.qnt.json`.
        /// Use this flag to force Quint parsing for files with other extensions.
        #[arg(help_heading = "Engine selection", long)]
        quint: bool,
        /// Run N random walks before exhaustive BFS for fast shallow bug detection.
        ///
        /// Each walk starts from a random initial state and fires random transitions
        /// up to 10,000 steps. Zero memory overhead (no state set). If a violation is
        /// found, the trace is reported immediately and BFS is skipped.
        #[arg(help_heading = "Engine selection", long, default_value = "0")]
        random_walks: usize,
        /// Maximum depth (steps) per random walk.
        ///
        /// Higher values explore deeper state spaces but take longer.
        #[arg(help_heading = "Engine selection", long, default_value = "10000")]
        walk_depth: usize,
        /// Compatibility alias that routes `ty check` through simulation mode.
        ///
        /// Use `ty simulate` for simulation-specific controls such as
        /// `--num-traces`, `--max-trace-length`, `--seed`, and `--no-invariants`.
        #[arg(help_heading = "Engine selection", long)]
        simulate: bool,
        /// Number of worker threads.
        /// 0 = auto (adaptive selection based on spec characteristics).
        /// 1 = sequential (no parallelism overhead).
        /// N = parallel with N workers.
        #[arg(display_order = 1, short, long, default_value = "0")]
        workers: usize,
        /// Disable deadlock checking.
        #[arg(display_order = 2, long)]
        no_deadlock: bool,
        /// Maximum number of states to explore (0 = unlimited).
        #[arg(display_order = 3, long, default_value = "0")]
        max_states: usize,
        /// Maximum BFS depth to explore (0 = unlimited).
        #[arg(display_order = 4, long, default_value = "0")]
        max_depth: usize,
        /// Memory limit in megabytes (0 = unlimited).
        ///
        /// When RSS reaches 80% of this limit, a checkpoint is saved (if configured).
        /// When RSS reaches 95%, exploration stops gracefully with a LimitReached result.
        #[arg(help_heading = "Bounds & limits", long, default_value = "0")]
        memory_limit: usize,
        /// Disk usage limit in megabytes for state storage (0 = unlimited).
        ///
        /// When disk-backed storage approaches this limit, exploration stops
        /// gracefully with a LimitReached result. This prevents filling disk
        /// volumes with .fp fingerprint files and state dumps.
        #[arg(help_heading = "Bounds & limits", long, default_value = "0")]
        disk_limit: usize,
        /// Gate: maximum allowed soundness mode (default: `sound`).
        ///
        /// This gate is about engine parity/experimental status. It does not encode
        /// boundedness; use `--require-exhaustive` to gate on configured bounds.
        #[arg(
            help_heading = "Bounds & limits",
            long,
            value_enum,
            default_value = "sound"
        )]
        soundness: SoundnessGate,
        /// Gate: require exhaustive exploration (rejects configured bounds).
        ///
        /// Fails fast if `--max-states` or `--max-depth` are non-zero.
        #[arg(help_heading = "Bounds & limits", long)]
        require_exhaustive: bool,
        /// Enable Bounded Model Checking (BMC) mode with given depth bound.
        ///
        /// BMC encodes k-step transition sequences as SAT formulas for quick bug
        /// finding. If a violation exists within k steps, BMC finds it directly.
        /// Use 0 to disable (default). Typical values: 10-100.
        ///
        /// Note: BMC is experimental and only supports a subset of TLA+ specs.
        /// Works best with specs using Bool/Int variables and simple arithmetic.
        #[arg(help_heading = "Engine selection", long, default_value = "0", conflicts_with_all = [
            "workers",
            "por", "coverage", "no_deadlock", "max_states", "max_depth",
            "memory_limit", "disk_limit", "require_exhaustive",
            "no_trace", "store_states", "initial_capacity",
            "mmap_fingerprints", "disk_fingerprints", "mmap_dir",
            "trace_file", "mmap_trace_locations",
            "checkpoint", "checkpoint_interval", "resume",
            "continue_on_error", "difftrace",
            "profile_enum", "profile_enum_detail", "profile_eval", "liveness_mode",
        ])]
        bmc: usize,
        /// Use incremental solving for BMC: reuse solver state across depths.
        ///
        /// When enabled, BMC keeps one solver instance across all unrolling depths,
        /// using push/pop scoping to retract per-depth safety queries while retaining
        /// learned clauses from Init + accumulated transition assertions. This avoids
        /// re-encoding all transitions from scratch at each depth.
        ///
        /// Requires `--bmc` > 0
        #[arg(help_heading = "Engine selection", long, requires = "bmc")]
        bmc_incremental: bool,
        /// Enable PDR (Property-Directed Reachability) symbolic safety checking.
        ///
        /// PDR uses IC3/PDR algorithm with CHC (Constrained Horn Clauses) to prove
        /// safety properties symbolically. Unlike explicit-state model checking,
        /// PDR can prove safety for infinite-state systems.
        ///
        /// Note: PDR requires the ay feature and only supports a subset of TLA+ specs
        /// (Bool/Int variables, arithmetic, comparisons)
        #[cfg(feature = "ay")]
        #[arg(help_heading = "Engine selection", long, conflicts_with_all = [
            "workers", "bmc",
            // Explicit-state features unsupported by the symbolic PDR engine.
            // Part of #3576: fail closed instead of silently dropping flags.
            "por", "coverage", "no_deadlock", "max_states", "max_depth",
            "memory_limit", "disk_limit", "require_exhaustive",
            "no_trace", "store_states", "initial_capacity",
            "mmap_fingerprints", "disk_fingerprints", "mmap_dir",
            "trace_file", "mmap_trace_locations",
            "checkpoint", "checkpoint_interval", "resume",
            "continue_on_error", "difftrace",
            "profile_enum", "profile_enum_detail", "profile_eval", "liveness_mode",
        ])]
        pdr: bool,
        /// Enable k-induction symbolic safety proving.
        ///
        /// K-induction extends BMC by attempting to prove safety properties hold
        /// for ALL reachable states, not just bounded depth. First runs BMC as
        /// a base case, then checks an inductive step. If the inductive step
        /// succeeds (UNSAT), the property is proved for all reachable states.
        ///
        /// Note: requires the ay feature and only supports a subset of TLA+ specs
        /// (Bool/Int variables, arithmetic, comparisons)
        #[cfg(feature = "ay")]
        #[arg(help_heading = "Engine selection", long, conflicts_with_all = [
            "workers", "bmc", "pdr",
            "por", "coverage", "no_deadlock", "max_states", "max_depth",
            "memory_limit", "disk_limit", "require_exhaustive",
            "no_trace", "store_states", "initial_capacity",
            "mmap_fingerprints", "disk_fingerprints", "mmap_dir",
            "trace_file", "mmap_trace_locations",
            "checkpoint", "checkpoint_interval", "resume",
            "continue_on_error", "difftrace",
            "profile_enum", "profile_enum_detail", "profile_eval", "liveness_mode",
        ])]
        kinduction: bool,
        /// Maximum induction depth for k-induction (default: 20).
        ///
        /// The algorithm tries increasing depths k=1..N until the inductive step
        /// succeeds (property proved) or the maximum is reached (inconclusive).
        /// Higher values increase the chance of proving the property but take longer.
        #[cfg(feature = "ay")]
        #[arg(
            help_heading = "Engine selection",
            long,
            default_value = "20",
            requires = "kinduction"
        )]
        kinduction_max_k: usize,
        /// Use incremental solving for k-induction's inductive step.
        ///
        /// Keeps a single solver instance across all depth iterations, retaining
        /// learned clauses via push/pop scoping. Can significantly speed up the
        /// inductive step for specs where each depth builds on the previous one.
        #[cfg(feature = "ay")]
        #[arg(help_heading = "Engine selection", long, requires = "kinduction")]
        kinduction_incremental: bool,
        /// Enable multi-phase verification pipeline.
        ///
        /// Runs cheap verification phases first (random walk, BMC) to resolve
        /// easy properties, then expensive phases (PDR, BFS) only on remaining
        /// hard properties. The default pipeline is:
        /// walk(5s) -> BMC(30s) -> PDR(60s) -> BFS(300s).
        ///
        /// BMC and PDR phases require the ay feature; they are silently skipped
        /// if ay is not available
        #[arg(help_heading = "Engine selection", long, conflicts_with = "strategy")]
        pipeline: bool,
        /// Named verification strategy for multi-phase pipeline mode.
        ///
        /// Selects a pre-configured pipeline of verification phases:
        ///   quick  — RandomWalk(2s) + BMC(10s). Fast feedback, no BFS.
        ///   full   — RandomWalk(5s) + exhaustive BFS(600s). Exhaustive fallback within the timeout.
        ///   auto   — walk -> BMC -> k-induction -> PDR -> BFS (adaptive).
        ///
        /// Implies `--pipeline`. Conflicts with `--bmc` and with plain
        /// `--pipeline` (which uses the default auto strategy).
        #[arg(help_heading = "Engine selection", long, value_enum, conflicts_with_all = [
            "pipeline", "bmc",
        ])]
        strategy: Option<StrategyArg>,
        /// Force pure BFS model checking (disable CDEMC/fused default).
        ///
        /// With the ay feature, the checker defaults to fused BFS+symbolic
        /// verification (CDEMC); this flag disables the symbolic lanes and
        /// runs pure explicit-state BFS, matching TLC.
        #[arg(help_heading = "Engine selection", long, conflicts_with_all = ["pipeline", "bmc", "portfolio", "strategy"])]
        bfs_only: bool,
        /// Enable cooperative fused BFS+symbolic verification (now the default).
        ///
        /// This is now the default when the ay feature is enabled. The flag is
        /// retained for backward compatibility but is no longer required.
        /// Use `--bfs-only` to opt out.
        #[cfg(feature = "ay")]
        #[arg(help_heading = "Engine selection", long, conflicts_with_all = ["pipeline", "bmc", "pdr", "kinduction", "portfolio"], hide = true)]
        fused: bool,
        /// Enable portfolio racing: run BFS + symbolic strategies in parallel.
        ///
        /// Spawns multiple verification lanes simultaneously and terminates when
        /// the first one reaches a definitive result. BFS always runs; symbolic
        /// lanes (PDR, BMC, k-induction) require the ay feature.
        ///
        /// Use `--portfolio-strategies` to select which strategies to race.
        /// Default: bfs + all available symbolic strategies.
        #[arg(help_heading = "Engine selection", long, conflicts_with_all = [
            "pipeline", "bmc", "strategy",
            "workers",
        ])]
        portfolio: bool,
        /// Comma-separated list of portfolio strategies to race.
        ///
        /// Available strategies: bfs, random (always available);
        /// bmc, pdr, kinduction (require ay feature).
        /// Default when omitted: all available strategies.
        ///
        /// Examples:
        ///   --portfolio --portfolio-strategies bfs,random
        ///   --portfolio --portfolio-strategies bfs,bmc,pdr
        #[arg(
            help_heading = "Engine selection",
            long,
            requires = "portfolio",
            value_delimiter = ','
        )]
        portfolio_strategies: Vec<String>,
        /// Enable Partial Order Reduction (POR) for state-space reduction.
        ///
        /// POR computes ample sets of independent actions to reduce the number
        /// of states explored while preserving safety properties. Works with
        /// both sequential and parallel BFS engines.
        #[arg(help_heading = "Reduction", long)]
        por: bool,
        /// Force-enable automatic POR (independence-based state-space reduction).
        ///
        /// Auto-POR is ON by default; this flag makes it explicit, e.g. for
        /// scripting alongside `--no-auto-por`.
        #[arg(help_heading = "Reduction", long, conflicts_with = "no_auto_por")]
        auto_por: bool,
        /// Disable automatic POR (independence-based state-space reduction).
        ///
        /// Auto-POR is ON by default; pass this to turn it off (e.g. to compare
        /// reduced vs. full state spaces, or to isolate a POR-related issue).
        #[arg(help_heading = "Reduction", long, conflicts_with = "auto_por")]
        no_auto_por: bool,
        /// Force-enable automatic symmetry reduction.
        ///
        /// Auto-symmetry is ON by default; this flag makes it explicit, e.g.
        /// for scripting alongside `--no-auto-symmetry`.
        #[arg(help_heading = "Reduction", long, conflicts_with = "no_auto_symmetry")]
        auto_symmetry: bool,
        /// Disable automatic symmetry reduction.
        ///
        /// Auto-symmetry is ON by default; pass this to turn it off (e.g. to
        /// compare reduced vs. full state spaces, or to isolate a symmetry-related
        /// issue).
        #[arg(help_heading = "Reduction", long, conflicts_with = "auto_symmetry")]
        no_auto_symmetry: bool,
        /// Disable ALL automatic state-space reductions (auto-symmetry +
        /// auto-POR) in one flag — the apples-to-apples lever for
        /// engine-vs-engine comparison with TLC (identical state counts).
        /// Equivalent to `--no-auto-symmetry --no-auto-por`.
        #[arg(help_heading = "Reduction", long, conflicts_with_all = ["auto_por", "auto_symmetry"])]
        no_reduction: bool,
        /// Enable native record-set (RecordSetBitmask) state layout + action
        /// compilation.
        ///
        /// Currently default-OFF pending trust-toolchain soundness validation;
        /// `--record-set-native` enables it.
        #[arg(
            help_heading = "Engine selection",
            long,
            conflicts_with = "no_record_set_native"
        )]
        record_set_native: bool,
        /// Disable native record-set (RecordSetBitmask) state layout + action
        /// compilation. This is the default; the flag makes it explicit (e.g.
        /// for scripting alongside `--record-set-native`).
        #[arg(
            help_heading = "Engine selection",
            long,
            conflicts_with = "record_set_native"
        )]
        no_record_set_native: bool,
        /// Show state space size estimate during model checking.
        ///
        /// Tracks states discovered per BFS level and fits growth models
        /// (exponential, logistic, linear) to predict total reachable states.
        #[arg(help_heading = "Diagnostics & profiling", long)]
        estimate: bool,
        /// Run estimation-only mode: explore first N BFS levels, then stop.
        ///
        /// Implies `--estimate`.
        #[arg(help_heading = "Diagnostics & profiling", long, value_name = "LEVELS")]
        estimate_only: Option<usize>,
        /// Show per-action coverage statistics.
        ///
        /// Note: Coverage collection is only supported in sequential mode today.
        /// Use `--workers 1` or `--workers 0` (auto, which will force sequential).
        #[arg(help_heading = "Diagnostics & profiling", long)]
        coverage: bool,
        /// Downgrade named vacuity class(es) from the default VACUOUS verdict to a
        /// recorded WARNING. Comma-separated. Classes: `empty-init`, `dead-action`,
        /// `vacuous-invariant`. The relaxation is audited (printed), not silent.
        ///
        /// Vacuity gate (TRUST_VACUITY_GATE §1.A): by default a model that admits
        /// no states / has a never-fired anchored action / a vacuously-true
        /// invariant is reported as VACUOUS (exit 3). This is the escape hatch.
        #[arg(
            help_heading = "Checking behavior",
            long,
            value_name = "CLASS[,CLASS...]",
            value_delimiter = ','
        )]
        allow_vacuous: Vec<String>,
        /// Promote default-on vacuity WARNINGs (V2 dead actions, V3 vacuously-true
        /// invariants) to the hard VACUOUS verdict (exit code 3).
        #[arg(help_heading = "Checking behavior", long)]
        strict_vacuity: bool,
        /// Enable enumeration profiling (coarse timing breakdown).
        ///
        /// Prints a high-level time breakdown showing: Successor gen, Fingerprinting,
        /// Dedup check, Invariant check, and Other. Equivalent to TY_PROFILE_ENUM=1.
        #[arg(help_heading = "Diagnostics & profiling", long)]
        profile_enum: bool,
        /// Enable detailed enumeration profiling.
        ///
        /// Prints detailed breakdowns inside the enumerator: domain/guard/assignment
        /// time, EXISTS loop details, etc. Equivalent to TY_PROFILE_ENUM_DETAIL=1.
        #[arg(help_heading = "Diagnostics & profiling", long)]
        profile_enum_detail: bool,
        /// Enable eval() call count profiling.
        ///
        /// Prints eval call count summary at the end of checking. Useful for
        /// identifying expensive sub-expression evaluation. Equivalent to TY_PROFILE_EVAL=1.
        #[arg(help_heading = "Diagnostics & profiling", long)]
        profile_eval: bool,
        /// Liveness execution strategy.
        ///
        /// `full` reuses the BFS-cached successor graph. `on-the-fly` skips
        /// that graph and regenerates successors lazily during liveness
        /// product exploration to reduce memory usage.
        #[arg(
            help_heading = "Liveness",
            long = "liveness-mode",
            value_enum,
            default_value = "full"
        )]
        liveness_mode: LivenessModeArg,
        /// Strict liveness: panic on missing states instead of skipping.
        ///
        /// By default, the parallel liveness checker gracefully skips states
        /// that are missing from the post-BFS state cache (with a warning).
        /// Use --strict-liveness to panic on any missing state, which is
        /// useful for debugging nondeterministic liveness crashes.
        /// Equivalent to setting TY_STRICT_LIVENESS=1.
        #[arg(help_heading = "Liveness", long)]
        strict_liveness: bool,
        /// Enable JIT compilation of invariant and action operators.
        ///
        /// Hidden legacy compatibility path. Prefer `--backend trust_cg` for native
        /// execution experiments; set `TY_JIT=1` only when investigating the
        /// retained legacy runtime surface.
        #[arg(help_heading = "Engine selection", long, hide = true)]
        jit: bool,
        /// Cross-check JIT invariant results against the interpreter.
        ///
        /// Hidden legacy verification flag retained with `--jit`.
        #[arg(help_heading = "Engine selection", long, hide = true)]
        jit_verify: bool,
        /// Show per-action tier compilation summary at end of run.
        ///
        /// Prints a table showing each action's compilation tier, evaluation
        /// count, branching factor, and JIT dispatch counters (hits, fallbacks,
        /// not_compiled, errors). Useful for diagnosing JIT coverage.
        #[arg(help_heading = "Diagnostics & profiling", long)]
        show_tiers: bool,
        /// Control speculative type specialization for JIT Tier 2.
        ///
        /// When `auto` (default), type profiling is enabled when --jit is
        /// active. The profiler samples runtime types of state variables
        /// during BFS warmup (~1000 states). If variables are monomorphic
        /// (always Int or Bool), a specialization plan guides Tier 2
        /// recompilation to skip type checks in compiled code.
        ///
        /// `on` forces profiling even without --jit (useful for diagnostics).
        /// `off` disables profiling entirely.
        #[arg(
            help_heading = "Diagnostics & profiling",
            long,
            value_enum,
            default_value = "auto"
        )]
        type_specialize: TypeSpecializeArg,
        /// Maximum memory efficiency: disable all trace reconstruction.
        ///
        /// By default, TY stores only fingerprints with a temp trace file for
        /// counterexample reconstruction (42x less memory than full states).
        /// Use --no-trace to also disable the trace file, which maximizes memory
        /// savings but may leave safety counterexample traces unavailable.
        ///
        /// Temporal checking still runs when the checker can replay from cached
        /// graph data, and the checker may still enable full-state storage when
        /// required for soundness-critical cases.
        #[arg(help_heading = "State storage & checkpoints", long)]
        no_trace: bool,
        /// Store full states in memory (legacy mode, 42x more memory).
        ///
        /// By default, TY stores only fingerprints with disk-based trace
        /// reconstruction. Use --store-states to keep full states in memory,
        /// which provides faster trace reconstruction but uses ~42x more memory.
        /// This was the default behavior before v0.6.
        ///
        /// Conflicts with --no-trace.
        #[arg(
            help_heading = "State storage & checkpoints",
            long,
            conflicts_with = "no_trace"
        )]
        store_states: bool,
        /// Pre-allocate in-memory fingerprint storage for the expected
        /// number of states (e.g., "6000000" for 6M states).
        ///
        /// If not set, the hash set grows dynamically with O(n) resize events
        /// during model checking, which can degrade performance at scale.
        /// For large specs, this can add 20+ resize events that each rehash all
        /// fingerprints. Pre-allocation avoids this overhead entirely.
        #[arg(help_heading = "State storage & checkpoints", long, value_name = "CAPACITY", conflicts_with_all = ["mmap_fingerprints", "disk_fingerprints"])]
        initial_capacity: Option<usize>,
        /// Use memory-mapped fingerprint storage with given capacity.
        ///
        /// This enables exploring state spaces larger than available RAM by using
        /// memory-mapped storage that can page to disk. The capacity specifies the
        /// maximum number of fingerprints to store (e.g., "1000000" for 1M states).
        ///
        /// Incompatible with --store-states. If not set, uses in-memory hash set.
        #[arg(
            help_heading = "State storage & checkpoints",
            long,
            value_name = "CAPACITY",
            conflicts_with = "store_states"
        )]
        mmap_fingerprints: Option<usize>,
        /// Enable huge page hints for mmap fingerprint storage.
        ///
        /// When used with --mmap-fingerprints, requests the OS to back the
        /// memory-mapped storage with huge pages (2MB) for reduced TLB pressure.
        #[arg(help_heading = "State storage & checkpoints", long)]
        huge_pages: bool,
        /// Use disk-backed fingerprint storage with automatic eviction.
        ///
        /// This enables exploring billion-state specs by automatically evicting
        /// fingerprints from memory to disk when the primary storage fills up.
        /// The capacity specifies the in-memory primary storage size before eviction.
        ///
        /// Requires --mmap-dir. Incompatible with --store-states and --mmap-fingerprints.
        #[arg(help_heading = "State storage & checkpoints", long, value_name = "CAPACITY", conflicts_with_all = ["mmap_fingerprints", "store_states"])]
        disk_fingerprints: Option<usize>,
        /// Directory for memory-mapped or disk-backed fingerprint storage.
        ///
        /// If specified with --mmap-fingerprints, creates a file-backed mapping
        /// in this directory, allowing the OS to page fingerprints to disk.
        /// If not specified, uses anonymous memory mapping (in-memory, but with
        /// mmap semantics for potentially better OS memory management).
        ///
        /// Required for --disk-fingerprints. The evicted fingerprints are stored
        /// as sorted files in this directory.
        #[arg(help_heading = "State storage & checkpoints", long, value_name = "DIR")]
        mmap_dir: Option<PathBuf>,
        /// Path to explicit disk-based trace file for counterexample reconstruction.
        ///
        /// By default, TY creates a temporary trace file automatically. Use this
        /// to specify a persistent location. The file stores (predecessor, fingerprint)
        /// pairs for trace reconstruction. Useful for debugging or keeping traces.
        ///
        /// Incompatible with --store-states. If file already exists, it will be overwritten.
        #[arg(
            help_heading = "State storage & checkpoints",
            long,
            value_name = "FILE",
            conflicts_with = "store_states"
        )]
        trace_file: Option<PathBuf>,
        /// Use memory-mapped storage for trace file location mapping.
        ///
        /// When using --trace-file, this option enables memory-mapped storage
        /// for the fingerprint-to-offset mapping. Specify the capacity (maximum
        /// number of states). This reduces memory usage for large state spaces.
        ///
        /// Requires --trace-file. Uses the same directory as --mmap-dir if specified.
        #[arg(
            help_heading = "State storage & checkpoints",
            long,
            value_name = "CAPACITY"
        )]
        mmap_trace_locations: Option<usize>,
        /// Fingerprint collision detection mode (none/sampling/sampling:N/full).
        #[arg(
            help_heading = "State storage & checkpoints",
            long,
            default_value = "none",
            value_name = "MODE"
        )]
        collision_check: String,
        /// Directory for saving checkpoints during model checking.
        ///
        /// Checkpoints let interrupted runs resume (--resume). Set the
        /// interval with --checkpoint-interval.
        #[arg(help_heading = "State storage & checkpoints", long, value_name = "DIR")]
        checkpoint: Option<PathBuf>,
        /// Checkpoint interval in seconds.
        ///
        /// Only used when --checkpoint is specified.
        #[arg(
            help_heading = "State storage & checkpoints",
            long,
            default_value = "300"
        )]
        checkpoint_interval: u64,
        /// Resume model checking from a checkpoint directory.
        #[arg(help_heading = "State storage & checkpoints", long, value_name = "DIR")]
        resume: Option<PathBuf>,
        /// Output format: human (default), json, jsonl, tlc-tool, or itf.
        ///
        /// `json` suits AI agents and automation. Aliases: `json-lines`
        /// for jsonl, `tool` for tlc-tool.
        #[arg(display_order = 5, long, value_enum, default_value = "human")]
        output: OutputFormat,
        /// Emit TLC "-tool" tagged output (Eclipse Toolbox-compatible).
        ///
        /// Equivalent to `--output tlc-tool`.
        #[arg(help_heading = "Output & traces", long, conflicts_with = "output")]
        tool: bool,
        /// Format for counterexample traces: text (default), dot, or itf.
        ///
        /// Use `dot` to output traces in GraphViz DOT format for visualization.
        /// The DOT output can be rendered using: dot -Tpng trace.dot -o trace.png
        ///
        /// Use `itf` for Informal Trace Format (ITF) JSON output compatible with
        /// Apalache and other TLA+ ecosystem tooling.
        #[arg(
            help_heading = "Output & traces",
            long,
            value_enum,
            default_value = "text"
        )]
        trace_format: TraceFormat,
        /// Show only changed variables in counterexample traces (TLC -difftrace).
        ///
        /// When enabled, each state after the first shows only variables whose
        /// values differ from the previous state. The initial state always shows
        /// all variables. Matches TLC's `-difftrace` behavior.
        #[arg(help_heading = "Output & traces", long)]
        difftrace: bool,
        /// Annotate counterexample traces with human-readable explanations.
        #[arg(help_heading = "Output & traces", long)]
        explain_trace: bool,
        /// Continue exploring after finding an invariant or property violation.
        ///
        /// By default, model checking stops at the first violation. With this
        /// flag, exploration runs until the state space is exhausted (or limits
        /// are reached), then reports the first violation with final stats —
        /// stable state counts comparable with TLC -continue. Violating states
        /// are counted as "seen" but not expanded further.
        #[arg(help_heading = "Checking behavior", long)]
        continue_on_error: bool,
        /// Allow incomplete results when fingerprint storage overflows.
        ///
        /// By default, mmap fingerprint overflow is a fatal error because dropped
        /// states make verification unsound. Use this only when you intentionally
        /// want partial exploration and accept incomplete results.
        #[arg(help_heading = "State storage & checkpoints", long)]
        allow_incomplete: bool,
        /// Bypass the local check cache (always re-run).
        #[arg(display_order = 6, long)]
        force: bool,
        /// Disable TIR preprocessing (NNF normalization, keramelization,
        /// constant folding).
        ///
        /// By default, the model checker applies a preprocessing pipeline to
        /// TIR expressions after lowering: negation normal form (NNF),
        /// conjunction/disjunction flattening (keramelization), and boolean
        /// constant folding. This improves evaluation performance by
        /// normalizing expression trees.
        ///
        /// Use --no-preprocess to skip the pipeline (for debugging or
        /// comparison). Equivalent to setting TY_NO_PREPROCESS=1.
        #[arg(help_heading = "Diagnostics & profiling", long)]
        no_preprocess: bool,
        /// Enable partial evaluation of CONSTANT bindings into TIR operator
        /// bodies before the preprocessing pipeline runs.
        ///
        /// Substitutes each module-level `CONSTANT` reference with its
        /// concrete `.cfg` value in the lowered TIR; const_prop then cascades
        /// through the baked literals (arithmetic folds, IF simplification,
        /// dead-branch elimination). Equivalent to `TY_PARTIAL_EVAL=1`.
        #[arg(help_heading = "Diagnostics & profiling", long)]
        partial_eval: bool,
        /// Allow IOExec and related operators to execute shell commands.
        ///
        /// IOExec, IOEnvExec, IOExecTemplate, and IOEnvExecTemplate are
        /// disabled by default: a malicious spec could run arbitrary shell
        /// commands on the host. Opt in only when you trust the spec.
        #[arg(help_heading = "Checking behavior", long)]
        allow_io: bool,
        /// Check trace invariants over the execution history (Apalache-style).
        ///
        /// A trace invariant is an operator that takes a `Seq(Record)` argument
        /// representing the execution trace up to the current state. Each record
        /// in the sequence has fields matching the spec's state variables.
        ///
        /// The operator is called at each BFS state with the trace leading to
        /// that state. If it returns FALSE, a violation is reported with the
        /// full trace as the counterexample.
        ///
        /// Can be specified multiple times: --trace-inv TraceInv1 --trace-inv TraceInv2.
        #[arg(
            help_heading = "Checking behavior",
            long = "trace-inv",
            alias = "trace-invariant",
            value_name = "OPERATOR"
        )]
        trace_invariants: Vec<String>,
        /// Check that an operator is an inductive invariant (Apalache-style).
        ///
        /// Runs a two-phase check:
        /// 1. Initiation: `Init => IndInv` (initial states satisfy the invariant)
        /// 2. Consecution: `IndInv /\ Next => IndInv'` (transitions preserve the invariant)
        ///
        /// Equivalent to `--kinduction --kinduction-max-k 1 --inv <INVARIANT>`
        /// (1-induction), but reports which phase fails for clearer diagnostics.
        ///
        /// Requires the ay feature
        #[cfg(feature = "ay")]
        #[arg(help_heading = "Engine selection", long, value_name = "INVARIANT")]
        inductive_check: Option<String>,
        /// Enable symbolic simulation mode (Apalache-style).
        ///
        /// Uses ay SMT solving to explore random execution paths symbolically.
        /// Unlike BMC (which checks all paths up to depth k), symbolic simulation
        /// follows one path per run, extracting concrete witnesses at each step.
        /// Multiple runs with different solver choices explore different paths.
        ///
        /// Requires the ay feature
        #[cfg(feature = "ay")]
        #[arg(help_heading = "Engine selection", long, conflicts_with_all = [
            "workers", "bmc", "pdr", "kinduction", "fused", "pipeline",
            "por", "coverage", "no_deadlock", "max_states", "max_depth",
            "memory_limit", "disk_limit", "require_exhaustive",
            "no_trace", "store_states", "initial_capacity",
            "mmap_fingerprints", "disk_fingerprints", "mmap_dir",
            "trace_file", "mmap_trace_locations",
            "checkpoint", "checkpoint_interval", "resume",
            "continue_on_error", "difftrace",
            "profile_enum", "profile_enum_detail", "profile_eval", "liveness_mode",
            "inductive_check",
        ])]
        symbolic_sim: bool,
        /// Number of simulation runs for `--symbolic-sim` (default: 100).
        ///
        /// Each run explores an independent random execution path.
        /// More runs increase the probability of finding violations.
        #[cfg(feature = "ay")]
        #[arg(
            help_heading = "Engine selection",
            long,
            default_value = "100",
            requires = "symbolic_sim"
        )]
        sim_runs: usize,
        /// Maximum depth (steps) per simulation run for `--symbolic-sim` (default: 10).
        ///
        /// Controls how many Next transitions each run explores.
        /// Higher values explore deeper state spaces but take longer per run.
        #[cfg(feature = "ay")]
        #[arg(
            help_heading = "Engine selection",
            long,
            default_value = "10",
            requires = "symbolic_sim"
        )]
        sim_length: usize,
        /// Evaluation backend.
        ///
        /// When omitted (the production default), TY uses AUTO engine selection:
        /// it runs the trust-cg native-compiled path when a cheap structural
        /// pre-check predicts it will help, and transparently falls back to the
        /// interpreter (the permanent correctness oracle) otherwise. The
        /// selection is purely structural (action compilability / native
        /// coverage / fused-loop eligibility) — never spec-name based.
        ///
        /// Pass `--backend interpreter` to force the oracle, or
        /// `--backend trust-cg` to force the native path with NO structural
        /// veto (the explicit form used by the supremacy harnesses and oracle
        /// cross-checks; falls back per-action for soundness but is never routed
        /// away by the auto-selector). If the backend is unavailable it emits a
        /// `backend_unavailable` JSON result and exits with code 3.
        #[arg(help_heading = "Engine selection", long, value_enum)]
        backend: Option<CheckBackend>,
    },
    /// Parse, validate, and visualize counterexample traces
    ///
    /// Umbrella for the trace subcommands, which parse, validate, and
    /// visualize counterexample traces produced by model checking.
    #[command(display_order = 31)]
    Trace {
        #[command(subcommand)]
        command: crate::trace_cmd::TraceCommand,
    },
    /// Run one MCC examination on a PNML Petri net
    ///
    /// Runs a single Model Checking Contest (MCC) examination against a
    /// Petri net given as a PNML (Petri Net Markup Language) model, with
    /// the model path and examination name passed explicitly on the
    /// command line (see `mcc` for the BenchKit environment-variable
    /// driven entrypoint).
    #[command(display_order = 52)]
    Petri {
        /// Model directory or a direct path to `model.pnml`.
        model: PathBuf,
        /// MCC examination name.
        #[arg(long)]
        examination: String,
        #[command(flatten)]
        args: tla_petri::cli::PetriRunArgs,
    },
    /// Print a JSON formula-simplification report for one examination
    ///
    /// Loads the model, parses the property XML for the given examination,
    /// simplifies each formula against the net's structure, and prints a
    /// JSON report showing which formulas were simplified, resolved, or
    /// left unchanged.
    #[command(name = "petri-simplify", hide = true)]
    PetriSimplify {
        /// Model directory containing `model.pnml` and examination XML files.
        model_dir: PathBuf,
        /// Property examination name (e.g. ReachabilityFireability, CTLFireability).
        #[arg(long)]
        examination: String,
    },
    /// Run MCC Petri-net examinations BenchKit-style
    ///
    /// Model Checking Contest (MCC) entrypoint for Petri nets given as
    /// PNML (Petri Net Markup Language) models — a drop-in replacement
    /// for the legacy `pnml-tools` harness, honoring the same `BK_INPUT`,
    /// `BK_EXAMINATION`, and `BK_TIME_CONFINEMENT` environment variables.
    #[command(display_order = 53)]
    Mcc {
        /// Model directory or a direct path to `model.pnml`.
        ///
        /// If omitted, uses `BK_INPUT` or the current directory.
        model_dir: Option<PathBuf>,
        /// MCC examination name.
        ///
        /// If omitted, uses `BK_EXAMINATION`.
        #[arg(long)]
        examination: Option<String>,
        #[command(flatten)]
        args: tla_petri::cli::PetriRunArgs,
    },
    /// Run property-based random-walk tests on a TLA+ spec
    ///
    /// Runs N random walks through the state space, checking invariants
    /// along each trace. Reports results in a test-framework style with
    /// per-invariant pass/fail and trace statistics.
    ///
    /// Unlike exhaustive model checking (`check`), `test` uses random
    /// exploration for fast feedback during development. Any bug found
    /// is real, but absence of bugs is not a proof of correctness.
    ///
    /// Exit code 0 on all-pass, 1 on any failure. Use `--seed` for
    /// deterministic replay of failures.
    #[command(display_order = 22)]
    Test {
        /// TLA+ source file to test.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of random walks (test runs) to execute.
        #[arg(short, long, default_value = "100")]
        runs: usize,
        /// Maximum depth (steps) per random walk.
        #[arg(short, long, default_value = "10000")]
        depth: usize,
        /// Random seed for reproducibility (0 = random seed).
        #[arg(short, long, default_value = "0")]
        seed: u64,
        /// Number of worker threads for parallel trace generation.
        /// 0 = auto (use available cores), 1 = sequential (default).
        #[arg(short, long, default_value = "1")]
        workers: usize,
        /// Disable deadlock checking.
        #[arg(long)]
        no_deadlock: bool,
    },
    /// Simulate a TLA+ spec by generating random traces
    ///
    /// Unlike exhaustive model checking, simulation generates random traces
    /// through the state space. This is useful for:
    /// - Quick exploration of large state spaces
    /// - Finding bugs that require deep traces
    /// - Probabilistic coverage when exhaustive checking is infeasible
    #[command(display_order = 23)]
    Simulate {
        /// TLA+ source file to simulate.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of random traces to generate.
        #[arg(short, long, default_value = "1000")]
        num_traces: usize,
        /// Maximum length of each trace (steps from initial state).
        #[arg(short = 'l', long, default_value = "100")]
        max_trace_length: usize,
        /// Random seed for reproducibility (0 = random seed).
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Disable invariant checking during simulation.
        #[arg(long)]
        no_invariants: bool,
        /// Allow IOExec and related operators to execute shell commands.
        ///
        /// By default, the IOExec, IOEnvExec, IOExecTemplate, and
        /// IOEnvExecTemplate operators are disabled for security. Pass
        /// --allow-io only when simulating trusted specs.
        #[arg(long)]
        allow_io: bool,
    },
    /// Start the LSP server
    ///
    /// Launches the LSP server for TLA+ editor and IDE integration.
    #[command(display_order = 16)]
    Lsp,
    /// Start a JSON-RPC server for interactive state exploration
    ///
    /// Loads a TLA+ spec and exposes it via a TCP JSON-RPC 2.0 interface for
    /// step-by-step model exploration. Supports: `init()`, `step(state_id)`,
    /// `eval(state_id, expr)`, `check_invariant(state_id, inv)`.
    #[command(hide = true)]
    Server {
        /// TLA+ source file to load.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// TCP port to listen on (default: 8765).
        #[arg(short, long, default_value = "8765")]
        port: u16,
    },
    /// Explore a TLA+ spec's state space interactively
    ///
    /// Default mode is an interactive terminal REPL with commands:
    ///   init, step, pick, eval, inv, back, forward, trace, actions, help, quit
    ///
    /// With `--mode http`, starts an HTTP JSON API server instead:
    ///   POST /explore/init          — compute initial states
    ///   POST /explore/eval          — evaluate expression in a state
    ///   POST /explore/successors    — compute successor states
    ///   POST /explore/random-trace  — generate a random execution trace
    #[command(display_order = 24)]
    Explore {
        /// TLA+ source file to load.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// HTTP port for `--mode http` (default: 8080).
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Exploration mode: 'repl' (default) or 'http'.
        #[arg(short, long, default_value = "repl")]
        mode: ExploreModeArg,
        /// Exploration engine: 'concrete' (default) or 'symbolic'.
        ///
        /// 'concrete' uses explicit-state enumeration (always available).
        /// 'symbolic' uses SMT-based exploration via ay (requires ay feature).
        /// You can also switch engines at runtime via the REPL 'symbolic'/'concrete'
        /// commands or the POST /explore/mode HTTP endpoint.
        #[arg(short, long, default_value = "concrete")]
        engine: ExploreEngineArg,
        /// Maximum symbolic exploration depth (default: 20).
        ///
        /// Limits how many symbolic transition steps can be taken before
        /// the solver rejects further exploration. Only meaningful when
        /// `--engine symbolic` is used or when switching to symbolic mode
        /// at runtime.
        #[arg(long, default_value = "20")]
        max_symbolic_depth: usize,
        /// Disable automatic invariant checking at each step (REPL mode only).
        #[arg(long)]
        no_invariants: bool,
    },
    /// Run diagnostic coverage analysis against the TLC baseline
    ///
    /// Tests the current binary against known TLC results from spec_baseline.json.
    /// The binary tests itself via std::env::current_exe(), so the binary being
    /// measured is always the binary doing the measuring.
    #[command(hide = true)]
    Diagnose(DiagnoseArgs),

    /// Learn to model-check specifications, interactively
    Tutorial {
        /// Topic: basics, soundness, certificates, frontends, features, or demo
        topic: Option<String>,
    },
    /// Classify Next action shapes for trust-codegen coverage
    ///
    /// Statically reports compiled-path coverage: action counts by
    /// binding-free, single-EXISTS-bound, multi-EXISTS-bound, and unsupported
    /// shape without running BFS.
    #[command(name = "trust_cg-coverage", hide = true)]
    TrustCgCoverage(TrustCgCoverageArgs),
    /// Run eval/check, enumerate, API, and silent-error canary gates.
    ///
    /// The Rust CLI owns changed-file selection, skip/warn env handling,
    /// diagnose policy, API canary execution, and silent-error scan policy.
    /// Shell script paths are compatibility wrappers only.
    #[command(
        name = "canary-gate",
        hide = true,
        long_about = "Run eval/check, enumerate, API, and silent-error canary gates.\n\nThe Rust CLI owns changed-file selection, staged-file handling, skip/warn env handling, diagnose policy, API canary execution, and silent-error scan policy. Shell script paths are compatibility wrappers only; do not add independent product gate logic outside this command."
    )]
    CanaryGate(CanaryGateArgs),
    /// Scan explicit Rust source files and report functions over a line limit
    ///
    /// This is the authoritative implementation for the code-quality function
    /// size gate. It ignores comments and string literals, prints
    /// Python-compatible offender lines, and exits 0 when oversized functions
    /// are found so shell gates can aggregate failures.
    #[command(
        name = "rust-function-span-scan",
        hide = true,
        long_about = "Scan explicit Rust source files and report functions over a line limit.\n\nDirectories are not expanded. Comments and string literals are ignored while counting braces. This is the authoritative implementation for the code-quality function size gate. It preserves the legacy Python scanner contract: each offender is printed as `path:line: fn name (N lines)`, and the command exits 0 even when offenders are reported so wrapper gates can aggregate all check failures."
    )]
    RustFunctionSpanScan(RustFunctionSpanScanArgs),
    /// Check repo and dev-environment health
    ///
    /// Single, compiler-enforced interface for the TY system-health
    /// contract: prints `OK`/`WARN`/`ERR` lines, optionally writes a JSON
    /// manifest, and fails only on error-level checks when run in enforce
    /// mode. The former `scripts/system_health_check.py` facade and its
    /// `system_health_check_support/` Python helpers have been deleted.
    #[command(
        name = "system-health-gate",
        hide = true,
        long_about = "Run the system-health gate through the Rust CLI.\n\nThis is the single, compiler-enforced TY health-gate entry point. Each check prints an `OK`, `WARN`, or `ERR` line, `--json-output` writes the health manifest (schema v1.0), and enforce mode exits non-zero when error-level checks are present. The former `scripts/system_health_check.py` facade and its `system_health_check_support/` Python helpers have been deleted; new checks belong inside `crates/tla-cli/src/cmd_system_health_gate.rs`."
    )]
    SystemHealthGate(SystemHealthGateArgs),
    /// Run Rust supremacy evidence commands.
    ///
    /// This Rust entrypoint owns targeted TLC-vs-TY comparison, bounded
    /// native-fused smoke, raw single-thread benchmark collection, policy
    /// verdict evaluation, and all-runnable baseline matrix classification.
    /// Only `ty supremacy gate --mode enforce --gate-mode full-native-fused
    /// --runs 3` is the current three-spec single-thread launch-corpus gate.
    #[command(
        hide = true,
        long_about = "Run Rust supremacy evidence commands.\n\nThis Rust entrypoint owns targeted TLC-vs-TY comparison, bounded native-fused smoke, raw single-thread benchmark collection, policy verdict evaluation, and all-runnable baseline matrix classification. Only `ty supremacy gate --mode enforce --gate-mode full-native-fused --runs 3` is the current three-spec single-thread launch-corpus gate; `matrix-full-suite` is broad all-runnable matrix refresh/audit evidence. Python, shell, JQ, and wrapper paths are diagnostic or compatibility surfaces only unless they delegate to this Rust command family and preserve its exit code."
    )]
    Supremacy(SupremacyArgs),
    /// Benchmark TLA+ specs with baseline comparison
    ///
    /// Runs each spec through `ty check` with precise timing, reports throughput
    /// (states/sec), wall time, and optionally compares against TLC baselines or
    /// previously-saved benchmark data.
    #[command(display_order = 70)]
    Bench {
        /// Spec file(s) or directory to benchmark.
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Config file (.cfg) for single-spec benchmarks.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of iterations per spec for statistical accuracy.
        #[arg(long, default_value = "1")]
        iterations: usize,
        /// Number of worker threads (0 = auto).
        #[arg(short, long, default_value = "0")]
        workers: usize,
        /// Path to spec_baseline.json for TLC comparison.
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Save results as new benchmark baseline JSON file.
        #[arg(long)]
        save_baseline: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: BenchOutputFormat,
    },
    /// Summarize check results for many specs in a table
    ///
    /// Summarizes model checking results for one or more specs in a compact
    /// table. Prefers existing `<spec>.ty-check.json` sidecars, and runs
    /// `ty check --output json` for specs whose sidecar is missing. Produces
    /// a one-line-per-spec summary showing status, state count, time, invariant
    /// count, and deadlock status. Useful for batch checking directories.
    ///
    /// Output formats: human (colored table), json (array), csv (spreadsheets).
    #[command(hide = true)]
    Summary {
        /// Spec file(s) or directory to check.
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Config file (.cfg) for single-spec runs.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of worker threads per check subprocess (0 = auto).
        #[arg(short, long, default_value = "0")]
        workers: usize,
        /// Output format: human (default), json, or csv.
        #[arg(long, value_enum, default_value = "human")]
        format: SummaryOutputFormat,
        /// Sort results by: name (default), time, states, or status.
        #[arg(long, value_enum, default_value = "name")]
        sort: SummarySortBy,
        /// Filter results by status: pass, fail, or error.
        #[arg(long)]
        status: Option<String>,
    },
    /// Summarize a saved `ty check --output json` result
    ///
    /// This is the Rust-owned compatibility entrypoint for shell harnesses
    /// that need compact fields from a check JSON file.
    #[command(hide = true)]
    CheckSummary {
        /// JSON file from `ty check --output json`, or `-` for stdin.
        input: String,
        /// Output format: tsv (default) or json.
        #[arg(long, value_enum, default_value = "tsv")]
        format: CheckSummaryOutputFormat,
    },
    /// Reduce a failing TLA+ spec to a minimal reproducer
    ///
    /// Uses delta debugging to systematically remove operators, variables,
    /// constants, and expression sub-terms until a minimal spec that still
    /// exhibits the same property violation is found.
    #[command(display_order = 34)]
    Minimize {
        /// TLA+ source file to minimize.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Maximum number of model-checker oracle calls before stopping.
        #[arg(long, default_value = "1000")]
        max_oracle_calls: usize,
        /// Disable fine-grained expression simplification.
        #[arg(long)]
        no_fine: bool,
        /// Write the minimized spec to a file instead of stdout.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    /// Lint a TLA+ specification
    ///
    /// Parses the spec and runs static analysis checks including:
    /// unused operators, unused variables, shadowed names, missing Init/Next,
    /// stuttering detection, and naming convention warnings.
    #[command(display_order = 13)]
    Lint {
        /// TLA+ source file to lint.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: LintOutputFormat,
        /// Minimum severity to report: warning (default) or info.
        #[arg(long, value_enum, default_value = "warning")]
        severity: LintSeverityArg,
    },
    /// Search across TLA+ specifications for patterns
    ///
    /// Parses TLA+ specs and searches for operator definitions, variable
    /// declarations, constant declarations, expression text, and Next-action
    /// disjuncts matching a pattern. Recursively walks directories for .tla files.
    ///
    /// The pattern is a glob: `*` matches zero or more characters, `?` matches
    /// exactly one character. A bare substring (no wildcards) matches anywhere
    /// (case-insensitive).
    #[command(hide = true)]
    Search {
        /// Pattern to search for (glob: * and ? supported; plain text = substring).
        pattern: String,
        /// Files or directories to search. Directories are walked recursively for .tla files.
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// What to search for: operator (default), variable, constant, expr, action.
        #[arg(short, long, value_enum, default_value = "operator")]
        kind: SearchKind,
        /// Output format: human (default, grep-like) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: SearchOutputFormat,
    },
    /// Generate documentation from a TLA+ specification
    ///
    /// Parses the spec and extracts structural documentation including:
    /// module-level and operator-level comments, operator signatures,
    /// variable/constant declarations, EXTENDS/INSTANCE dependencies,
    /// and cross-reference graphs (calls/called-by).
    #[command(hide = true)]
    Doc {
        /// TLA+ source file to document.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: markdown (default), html, or json.
        #[arg(long, value_enum, default_value = "markdown")]
        format: DocOutputFormat,
        /// Output file. If not specified, writes to stdout.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    /// Type check a TLA+ specification
    ///
    /// Parses the spec, runs TIR lowering (which includes type inference),
    /// and reports inferred types for each operator. Also parses Apalache-style
    /// `@type:` annotations from TLA+ comments and reports any mismatches.
    #[command(display_order = 14)]
    Typecheck {
        /// TLA+ source file to type check.
        file: PathBuf,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        output: TypecheckOutputFormat,
        /// Show inferred types for all operators, variables, and constants.
        ///
        /// Without this flag, the command only reports type errors and
        /// annotation mismatches (exit 0 = no errors). With this flag,
        /// it additionally prints the full type map for every definition
        /// in the module.
        #[arg(long)]
        infer_types: bool,
    },
    /// Analyze dependency graph of a TLA+ specification
    ///
    /// Parses the spec and produces dependency information including:
    /// module dependencies (EXTENDS/INSTANCE), operator call graph,
    /// variable usage, root reachability, dead code detection, and
    /// circular dependency detection.
    ///
    /// Output formats: tree (default indented text), dot (Graphviz),
    /// or json (structured data for tooling).
    #[command(hide = true)]
    Deps {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: tree (default), dot, or json.
        #[arg(long, value_enum, default_value = "tree")]
        format: DepsOutputFormat,
        /// Highlight unused/unreachable operators in the output.
        #[arg(long)]
        unused: bool,
        /// Show only module-level dependencies (skip operator-level analysis).
        #[arg(long)]
        modules_only: bool,
    },
    /// Re-check a TLA+ specification on file changes
    ///
    /// Monitors the spec file and any EXTENDS'd modules for changes.
    /// On each change, re-parses and re-checks the specification,
    /// displaying results in the terminal. Debounces rapid changes
    /// (100ms) to avoid redundant re-checks during editing.
    #[command(display_order = 21)]
    Watch {
        /// TLA+ source file to watch and check.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of worker threads (0 = auto, 1 = sequential).
        #[arg(short, long, default_value = "0")]
        workers: usize,
        /// Disable deadlock checking.
        #[arg(long)]
        no_deadlock: bool,
        /// Debounce interval in milliseconds.
        #[arg(long, default_value = "100")]
        debounce_ms: u64,
        /// Clear terminal before each re-check.
        #[arg(long)]
        clear: bool,
    },
    /// Generate Rust code from a TLA+ specification
    ///
    /// Emits Rust code from the spec: optionally TIR-based, with runtime
    /// checker adapters, Kani harnesses, proptest suites, a full Cargo
    /// scaffold, and source maps.
    #[command(display_order = 60)]
    Codegen {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg) providing CONSTANTS and INVARIANTS.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output file (default: stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Use TIR-based code generation (typed intermediate representation).
        ///
        /// The TIR path benefits from type information, resolved INSTANCE
        /// references, and operator inlining. Requires --config for constant
        /// values
        #[arg(long)]
        tir: bool,
        /// Generate a production-type adapter layer (runtime checkers).
        #[arg(long)]
        checker: bool,
        /// Mapping config (TOML) for generating `impl checker::To<Spec>State for <RustType>` blocks.
        #[arg(long, value_name = "FILE")]
        checker_map: Option<PathBuf>,
        /// Generate Kani verification harnesses.
        #[arg(long)]
        kani: bool,
        /// Generate proptest property-based tests.
        #[arg(long)]
        proptest: bool,
        /// Generate a complete runnable Cargo project (directory with Cargo.toml,
        /// src/lib.rs, src/main.rs) instead of just the generated module.
        ///
        /// When combined with --kani, also generates a trust_mc/Kani verification
        /// harness file. The output directory is created at --output (required
        /// with --scaffold).
        #[arg(long)]
        scaffold: bool,
        /// Emit a source map JSON file alongside the generated Rust code.
        ///
        /// The source map records which TLA+ operators correspond to which
        /// line ranges in the generated Rust file, enabling counterexample
        /// traces to be annotated with generated source locations.
        /// Written to `<output>.source_map.json` when --output is provided.
        #[arg(long)]
        source_map: bool,
    },
    /// Export a TLA+ spec as a VMT transition system
    ///
    /// VMT (Verification Modulo Theories) is an SMT-LIB2 extension used by
    /// external model checkers such as nuXmv, ic3ia, and AVR.
    #[command(display_order = 61)]
    Vmt {
        /// TLA+ source file to export.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// Emit a re-checkable ty.cert/v1 inductive-safety certificate
    ///
    /// Proves an inductive invariant via the AY symbolic engine and serializes it
    /// into a self-contained JSON certificate. Re-check it with `ty cert-check`.
    /// Certifying verification: a proof you can check yourself, not just a verdict.
    #[command(display_order = 41)]
    Certify {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg). Defaults to `<spec>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output path for the certificate JSON.
        #[arg(short, long)]
        out: PathBuf,
        /// Fail closed on incomplete domain coverage: DECLINE to certify when
        /// any completeness obligation's domain rests on a trusted (not
        /// kernel-verified) bound rule instead of covering its full value
        /// universe by construction. Use this to restrict certification to
        /// the smallest trust base. Default off: such obligations are kept,
        /// and the reliance is disclosed in the certificate output.
        #[arg(long)]
        require_domain_complete: bool,
        /// Disable deadlock checking. By DEFAULT `ty certify` verifies deadlock-freedom (every
        /// reachable state has a successor under Next) — matching `ty check`, which checks for
        /// deadlock by default — and DECLINES a spec with a reachable deadlock. With this flag,
        /// certify proves inductive SAFETY only and discloses that deadlock-freedom is unverified.
        #[arg(long)]
        no_deadlock: bool,
    },
    /// Print the trust census of the real kernel environment
    ///
    /// Prints the honest trust census of the REAL kernel environment
    /// `Certified` verdicts run over.
    ///
    /// Clean's green "3-axiom" soundness certificate describes a CURATED env, not
    /// the `with_prelude` env TY type-checks proof terms against. This census counts
    /// that real env: its trust markers (each a proof of ANY proposition — blocked
    /// per-term by the Phase-0 gate) and its admitted domain axioms (surfaced, not
    /// hidden). Requires a `clean-cic` build.
    #[command(hide = true)]
    TcbCensus {
        /// Also list every domain-axiom name (not just the count).
        #[arg(long)]
        full: bool,
    },
    /// Certify a kernel-checked refinement mapping
    ///
    /// Certifies a KERNEL-CHECKED refinement mapping: the implementation spec
    /// refines the abstract spec (safety part, enumerated implementation
    /// graph). Every implementation transition must map to an abstract step
    /// or stutter, re-evaluated by the Clean CIC kernel — unlike `ty refine`,
    /// which checks the simulation in Rust with no re-checkable certificate.
    /// Requires a `clean-cic` build. Re-check with `ty refine-check`.
    #[command(hide = true)]
    RefineCertify {
        /// Implementation TLA+ source file.
        impl_file: PathBuf,
        /// Implementation configuration (.cfg) with INIT/NEXT.
        #[arg(short, long)]
        config: PathBuf,
        /// Abstract TLA+ source file.
        #[arg(long = "abstract")]
        abstract_file: PathBuf,
        /// Abstract configuration (.cfg) with INIT/NEXT.
        #[arg(long)]
        abstract_config: PathBuf,
        /// Refinement mapping `abs1=<expr1>,abs2=<expr2>`, each RHS an affine expression over
        /// the implementation variables: a bare variable (`k=c`) is a projection; a compound
        /// affine combination (`sum=a+b`) is a DERIVED data-refinement mapping. Unmapped
        /// abstract variables default to the same-named implementation variable.
        #[arg(long)]
        map: Option<String>,
        /// Output path for the refinement certificate JSON.
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Re-check a ty.refine-cert/v1 refinement certificate
    ///
    /// Independently re-checks the certificate: re-enumerates the
    /// implementation graph, re-recognizes the abstract predicates, and
    /// re-runs the kernel.
    #[command(hide = true)]
    RefineCheck {
        /// Path to the certificate JSON produced by `ty refine`.
        cert: PathBuf,
    },
    /// Re-check a ty.cert/v1 inductive-safety certificate
    ///
    /// Independently re-checks the certificate with the built-in AY engine
    /// only — no external solver, and no re-run of model checking. The
    /// certificate is self-contained, so only the certificate file is needed.
    /// Exits 0 on VERIFIED, 1 on REJECTED.
    #[command(hide = true)]
    CertCheck {
        /// Path to the certificate JSON produced by `ty certify`.
        cert: PathBuf,
        /// Additionally re-check each embedded proof with carcara, a SEPARATE,
        /// independently-implemented Alethe checker (N-version redundancy).
        #[arg(long)]
        carcara: bool,
    },
    /// Re-check a safety cert via the kernel's reflected evaluator
    ///
    /// Re-discharges an explicit-state cert's `R ⊆ Safety` obligation through
    /// the KERNEL-DEFINED reflected evaluator instead of the shallow Rust
    /// embedder (embedder-free safety leg).
    ///
    /// Quotes the recognized invariant with the line-auditable 1:1 quoter and lets the clean
    /// kernel reduce the deep evaluator over every reachable state, then binds the discharged
    /// obligation to the cert's own spec by re-derivation. Exits 0 on REFLECTED-SAFE, 1 on
    /// NOT-SAFE / REJECTED, 2 on INCONCLUSIVE (out-of-fragment / non-clean-cic build).
    #[command(hide = true)]
    ReflectCheck {
        /// Path to the certificate JSON produced by `ty certify`.
        cert: PathBuf,
        /// ALL-LEGS reflected discharge (R2 milestone): also re-discharge the two COMPLETENESS
        /// legs — Init-completeness (`Init⊆R`) and Next-completeness / closure (`R` closed under
        /// `Next`) — through the reflected evaluator (`TyReflectEvalP` composed with `TyReflectMem`),
        /// so a spec that passes all THREE reflected legs has its safety verdict backed with the
        /// shallow embedder (`embed_pred_ir`) OUT of every obligation. Prints REFLECTED-CERTIFIED
        /// (embedder-free) on success; exits 1 on NOT-SAFE / NOT-CLOSED / NOT-INIT-COMPLETE /
        /// REJECTED, 2 on INCONCLUSIVE. Without `--full`, only the `R⊆Safety` leg is reflected.
        #[arg(long)]
        full: bool,
        /// (with `--full`) FAIL-CLOSED domain coverage: DECLINE (INCONCLUSIVE) when the completeness
        /// domain `D` rests on a trusted-Rust bound rule (an axis that is not its column's full
        /// universe). The reflected legs still reduce, but closure is only RELATIVE to a Rust-bounded
        /// `D`; this flag refuses to certify closure on such a domain.
        #[arg(long)]
        require_domain_complete: bool,
        /// RECOGNIZER-FREE all-legs discharge (design pivot increment 1): quote Init/Next/Safety
        /// DIRECTLY from the spec's own (re-parsed, operator-inlined) TLA+ AST into the kernel's
        /// deep embedding — the RECOGNIZER (cleancic recognition arms) is OUT of every obligation
        /// and out of the spec-bind. Trust base = kernel + AST-quoter + parser/inliner + the AST
        /// domain-bound rule (+ enumerator-provided R, kernel-verified). Runs the recognized-IR
        /// reflected lane as a CROSS-CHECK and HARD-FAILS on any conclusive divergence. Exits 0 on
        /// REFLECTED-CERTIFIED (AST-DIRECT), 1 on NOT-SAFE / NOT-CLOSED / NOT-INIT-COMPLETE /
        /// CROSS-CHECK-DIVERGENCE, 2 on INCONCLUSIVE (out of the v1 fragment — the `--full`
        /// recognized-IR lane is the fallback tier).
        #[arg(long, conflicts_with_all = ["full", "require_domain_complete"])]
        ast_direct: bool,
    },
    /// Export certificate proofs as carcara-checkable files
    ///
    /// Exports each obligation's embedded proof as a carcara-checkable
    /// `<name>.problem.smt2` + `<name>.proof.alethe` pair (third-party re-check).
    #[command(hide = true)]
    CertExport {
        /// Path to the certificate JSON (safety / liveness / all-N).
        cert: PathBuf,
        /// Directory to write the per-obligation proof + problem files into.
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Emit a re-checkable ty.verdict/v1 violation envelope
    ///
    /// Runs the spec and, on an invariant/property violation, packages the
    /// counterexample trace + the embedded spec into a content-addressed envelope that
    /// `ty verdict-check` re-validates independently (eval-only). Exits 0 when an
    /// envelope is written, 2 when the run produced no violation to certify.
    #[command(hide = true)]
    VerdictEmit {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg). Defaults to `<spec>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output path for the verdict envelope JSON.
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Re-check a ty.verdict/v1 violation envelope
    ///
    /// Independently re-checks the envelope with the evaluator only — no
    /// solver and no BFS. The envelope is self-contained (embeds the spec +
    /// counterexample), so only the envelope file is needed. Replays the
    /// trace through the evaluator and confirms the violation. Exits 0 on
    /// VERIFIED, 1 on REJECTED, 2 on INCONCLUSIVE.
    #[command(hide = true)]
    VerdictCheck {
        /// Path to the verdict envelope JSON produced by `ty verdict-emit`.
        envelope: PathBuf,
    },
    /// Certify a liveness property under weak fairness
    ///
    /// Certifies a LIVENESS property `<>P` under WF(Next) by well-founded
    /// descent on an integer measure — a re-checkable `ty.live-cert/v1`
    /// certificate.
    #[command(hide = true)]
    CertifyLiveness {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg). Defaults to `<spec>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Operator whose body is `<>P` (the eventual-target property).
        #[arg(long)]
        property: String,
        /// Integer measure operator (the descent rank, e.g. `Measure`).
        #[arg(long)]
        measure: String,
        /// Output path for the liveness certificate JSON.
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Re-check a ty.live-cert/v1 liveness certificate
    ///
    /// Independently re-checks the certificate. Exits 0 on VERIFIED, 1 on
    /// REJECTED, 2 on INCONCLUSIVE.
    #[command(hide = true)]
    LiveCheck {
        /// Path to the liveness certificate JSON produced by `ty certify-liveness`.
        cert: PathBuf,
    },
    /// Certify an invariant for all values of a scalar CONSTANT
    ///
    /// Certifies a supplied invariant for ALL values of a scalar `CONSTANT`
    /// (all-N): the constant is kept SYMBOLIC and free, so the proof is
    /// parametric — a re-checkable `ty.alln-cert/v1` certificate.
    #[command(hide = true)]
    CertifyAllN {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg). Defaults to `<spec>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// The scalar CONSTANT to keep symbolic (prove for all values), e.g. `N`.
        #[arg(long)]
        constant: String,
        /// The inductive invariant `J` as TLA text, e.g. `"x >= N"`. When omitted
        /// (AUTO-J), `J` defaults to the spec's configured INVARIANT(s): the whole
        /// conjunction first, then per-conjunct coverage (one certificate per
        /// top-level conjunct) if the whole declines.
        #[arg(long = "invariant-j")]
        invariant_j: Option<String>,
        /// Output path for the all-N certificate JSON.
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Re-check a ty.alln-cert/v1 all-N certificate
    ///
    /// Independently re-checks the certificate. Exits 0 on VERIFIED, 1 on
    /// REJECTED, 2 on INCONCLUSIVE.
    #[command(hide = true)]
    AllNCheck {
        /// Path to the all-N certificate JSON produced by `ty certify-all-n`.
        cert: PathBuf,
    },
    /// Explain a counterexample trace step by step
    ///
    /// Reads a saved JSON output file produced by `ty check --output json` and
    /// generates a human-readable step-by-step explanation of the
    /// counterexample trace.
    #[command(display_order = 30)]
    Explain {
        /// Path to the JSON output file from `ty check --output json`.
        trace_file: PathBuf,
        /// Optional TLA+ spec file for invariant structure analysis.
        #[arg(long)]
        spec: Option<PathBuf>,
        /// Optional config file (.cfg) to locate the invariant name.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Specific invariant to analyze (overrides config file).
        #[arg(long)]
        invariant: Option<String>,
        /// Show only variable changes between consecutive steps.
        #[arg(long)]
        diff: bool,
        /// Show full state dump at each step.
        #[arg(long)]
        verbose: bool,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ExplainOutputFormat,
    },
    /// Analyze action coverage from a check's JSON output
    ///
    /// Reads a JSON output file produced by `ty check --output json` and reports
    /// which actions (Next disjuncts) were exercised during model checking, how many
    /// states each generated, and what percentage of actions were explored.
    #[command(hide = true)]
    Coverage {
        /// Path to the JSON output file from `ty check --output json`.
        trace_file: PathBuf,
        /// Optional TLA+ spec file for cross-referencing Next disjuncts.
        #[arg(long)]
        spec: Option<PathBuf>,
        /// Optional config file (.cfg) to locate the Next operator name.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: CoverageOutputFormat,
    },
    /// Visualize the state graph of a counterexample trace
    ///
    /// Reads a JSON output file produced by `ty check --output json` and
    /// generates a state transition graph where each state is a node and
    /// each transition is an edge labeled with the action name.
    #[command(display_order = 32)]
    Graph {
        /// Path to the JSON output file from `ty check --output json`.
        trace_file: PathBuf,
        /// Output format: dot (default), mermaid, or json.
        #[arg(long, value_enum, default_value = "dot")]
        format: GraphOutputFormat,
        /// Maximum number of states to render (0 = unlimited, default: 50).
        #[arg(long, default_value = "50")]
        max_states: usize,
        /// Highlight error/violating states in red (default: true).
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        highlight_error: bool,
        /// Group states by the action that created them.
        #[arg(long)]
        cluster_by_action: bool,
    },
    /// Check a BTOR2 hardware model checking benchmark
    ///
    /// Parses a BTOR2 file, translates each `bad` property to a CHC safety
    /// query, and dispatches to the ay-chc portfolio solver (PDR/BMC/k-induction).
    /// Output follows HWMCC convention: `sat` / `unsat` / `unknown` on stdout.
    #[cfg(feature = "ay")]
    #[command(display_order = 51)]
    Btor2 {
        /// BTOR2 file to check.
        file: PathBuf,
        /// Verbose output: print parse stats, per-property results, and timing.
        #[arg(short, long)]
        verbose: bool,
        /// Write HWMCC-style witness to a file (only written if a counterexample exists).
        #[arg(long, value_name = "FILE")]
        witness: Option<PathBuf>,
        /// Solver time budget in seconds (default: 27s from AdaptiveConfig).
        #[arg(long, value_name = "SECONDS")]
        timeout: Option<u64>,
        /// Bit-blast narrow bitvectors to AIGER and use the IC3/PDR engine.
        /// Automatically enabled for eligible benchmarks with bitvectors <= max-bv-width.
        #[arg(long)]
        bitblast: bool,
        /// Maximum bitvector width for bit-blasting (default: 32).
        #[arg(long, value_name = "BITS", default_value = "32")]
        max_bv_width: u32,
        /// EXPERIMENTAL: enable the lazy-array trace-unrolled BMC lane for
        /// bit-blast-ineligible array nets (wide-index memories), as a bounded
        /// leading slice before the CHC portfolio. SAT verdicts are
        /// replay-validated; bounded no-cex results fall through unchanged.
        /// Also enabled by TY_BTOR2_LAZY_ARRAY_BMC=1. Default: off.
        #[arg(long = "array-bmc")]
        array_bmc: bool,
    },
    /// Check an AIGER hardware model checking benchmark
    ///
    /// Parses an AIGER file (.aag or .aig), translates each bad-state literal
    /// to a safety query, and dispatches to the selected solver engine.
    /// Output follows HWMCC convention: `sat` / `unsat` / `unknown` on stdout.
    #[cfg(feature = "ay")]
    #[command(display_order = 50)]
    Aiger {
        /// AIGER file to check (.aag or .aig).
        file: PathBuf,
        /// Verbose output: print parse stats, per-property results, and timing.
        #[arg(short, long)]
        verbose: bool,
        /// Write HWMCC-style witness to a file (only written if a counterexample exists).
        #[arg(long, value_name = "FILE")]
        witness: Option<PathBuf>,
        /// Solver time budget in seconds.
        #[arg(long, value_name = "SECONDS")]
        timeout: Option<u64>,
        /// Solver engine: 'chc' (ay-chc portfolio), 'sat' (BMC + k-induction portfolio).
        /// Default: 'sat'.
        #[arg(long, value_name = "ENGINE", default_value = "sat")]
        engine: AigerEngine,
        /// Portfolio mode for the SAT engine (ignored when engine=chc).
        /// 'default': IC3 + BMC + k-induction (3 threads).
        /// 'full': 6 IC3 variants + BMC + k-induction (8 threads).
        /// 'competition': 8 IC3 variants + BMC + k-induction (10 threads).
        #[arg(long, value_name = "MODE", default_value = "default")]
        portfolio: AigerPortfolio,
    },
    /// Suggest repairs from a counterexample trace
    ///
    /// Reads a JSON output file produced by `ty check --output json`, analyzes
    /// the counterexample trace, and suggests potential fixes for invariant
    /// violations found in the trace:
    /// - Which variables changed in the violating step
    /// - What values would satisfy the invariant
    /// - Which action (Next disjunct) was taken
    /// - Minimal state change to restore the invariant
    #[command(display_order = 33)]
    Repair {
        /// Path to the JSON output file from `ty check --output json`.
        trace_file: PathBuf,
        /// TLA+ spec file for deeper analysis.
        #[arg(long)]
        spec: Option<PathBuf>,
        /// Config file (.cfg) for invariant identification.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Specific invariant to repair.
        #[arg(long)]
        invariant: Option<String>,
        /// Maximum number of repair suggestions to generate.
        #[arg(long, default_value = "5")]
        max_suggestions: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: RepairOutputFormat,
    },
    /// Profile a model-checking run's timing and hotspots
    ///
    /// Runs `ty check` with timing and resource instrumentation (profiling
    /// flags) and reports overall statistics, per-action breakdown, operator
    /// hotspots, and BFS level statistics.
    #[command(display_order = 71)]
    Profile {
        /// TLA+ source file to profile.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of worker threads (0 = auto).
        #[arg(short, long, default_value = "0")]
        workers: usize,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: ProfileOutputFormat,
        /// Show top N hottest operators/metrics (default: 20).
        #[arg(long, default_value = "20")]
        top: usize,
        /// Include memory timeline (RSS snapshots over time).
        #[arg(long)]
        memory: bool,
    },
    /// Compare two TLA+ specifications at the semantic (AST) level
    ///
    /// Parses both specs and reports added, removed, and modified operators,
    /// variable/constant declaration changes, EXTENDS changes, and invariant
    /// changes (when .cfg files are provided).
    #[command(hide = true)]
    Diff {
        /// Old (baseline) TLA+ source file.
        old: PathBuf,
        /// New (updated) TLA+ source file.
        new: PathBuf,
        /// Old configuration file (.cfg).
        #[arg(long)]
        old_config: Option<PathBuf>,
        /// New configuration file (.cfg).
        #[arg(long)]
        new_config: Option<PathBuf>,
        /// Output format: human (default, colored) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: DiffOutputFormat,
        /// Skip variable, constant, extends, and invariant diff; show only operator changes.
        #[arg(long)]
        operators_only: bool,
    },
    /// Convert between TLA+ and related formats
    ///
    /// Supported conversions: tla-to-json (AST), json-to-tla (pretty-print),
    /// tla-to-markdown (documentation), trace-to-table (counterexample table).
    #[command(display_order = 62)]
    Convert {
        /// Input file.
        input: PathBuf,
        /// Input format (auto-detected from extension if omitted).
        #[arg(long, value_enum)]
        from: Option<ConvertFrom>,
        /// Output format (required).
        #[arg(long, value_enum)]
        to: ConvertTo,
        /// Output file (stdout if omitted).
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    /// Show specification statistics and complexity metrics
    ///
    /// Parses a TLA+ spec and reports structural metrics: line counts,
    /// declaration counts, operator complexity, nesting depths, primed
    /// variable usage, UNCHANGED usage, and state-space size hints.
    #[command(hide = true)]
    Stats {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: StatsOutputFormat,
    },
    /// Generate shell completion scripts
    ///
    /// Prints a completion script for the specified shell to stdout.
    /// Redirect the output to the appropriate location for your shell.
    #[command(display_order = 81)]
    Completions {
        /// Shell to generate completions for.
        shell: Shell,
    },
    /// Inspect or clear the native-compilation cache
    ///
    /// Manages the trust-codegen on-disk compilation cache. The cache lives
    /// at `~/.cache/ty/compiled/` (override with
    /// `TY_CACHE_DIR`). Each cached artifact is a native dynamic library
    /// plus a JSON sidecar keyed by `sha256(trust-ir || trust_cg-version ||
    /// opt-level || target-triple)`. See design doc §7.
    #[command(display_order = 85)]
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
    /// Fetch the tlaplus/Examples benchmark corpus
    ///
    /// Downloads and caches the TLA+ benchmark corpus used by the TY-vs-TLC
    /// comparison harnesses and the spec_regression test.
    ///
    /// The corpus is NOT in this repo; tools expect it at `~/tlaplus-examples`
    /// (override with `TLAPLUS_EXAMPLES`). `ty corpus fetch` downloads the
    /// reproducible release asset (pinned to the commit in
    /// `tests/tlc_comparison/spec_baseline.json`) and verifies its sha256.
    #[command(display_order = 82)]
    Corpus {
        #[command(subcommand)]
        action: CorpusAction,
    },
    /// Install TLC, the reference TLA+ model checker
    ///
    /// Downloads and installs TLC so the TLC-vs-TY comparison runs with no
    /// manual setup.
    ///
    /// `ty install-tlc install` fetches the working nightly `tla2tools.jar`
    /// plus the pinned `CommunityModules.jar` into `~/tlaplus/` (override with
    /// `--dest`), landing them at the exact paths `ty supremacy compare`
    /// auto-discovers (`tytools.jar` / `CommunityModules.jar`). Verifies TLC
    /// actually runs. (`ty tlc` still works as a hidden alias.)
    #[command(name = "install-tlc", alias = "tlc", display_order = 83)]
    Tlc {
        #[command(subcommand)]
        action: TlcAction,
    },
    /// Install the Apalache symbolic TLA+ model checker
    ///
    /// Downloads and installs Apalache (the symbolic TLA+ model checker) so
    /// the Apalache-vs-TY comparison runs with no manual setup.
    ///
    /// `ty install-apalache install` fetches the pinned, sha256-verified
    /// release tarball and unpacks it so `~/apalache/bin/apalache-mc` resolves
    /// (override with `--dest`). Verifies the launcher runs. (`ty apalache`
    /// still works as a hidden alias.)
    #[command(name = "install-apalache", alias = "apalache", display_order = 84)]
    Apalache {
        #[command(subcommand)]
        action: ApalacheAction,
    },
    /// Refactor a TLA+ spec with semantic-preserving transforms
    ///
    /// Performs AST-guided source transformations: extracting expressions into
    /// named operators, renaming identifiers, inlining simple operators, and
    /// removing unused definitions.
    #[command(display_order = 15)]
    Refactor {
        #[command(subcommand)]
        action: RefactorAction,
    },
    /// Run snapshot tests for regression detection
    ///
    /// Records model-checking results (status, state counts, max depth) as
    /// snapshot files, then compares future runs against them to catch regressions.
    #[command(hide = true)]
    Snapshot {
        /// TLA+ source file(s) to snapshot.
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Config file (.cfg) for single-spec snapshots.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Directory to store/read snapshot files (default: .snapshots).
        #[arg(long, default_value = ".snapshots")]
        snapshot_dir: PathBuf,
        /// Update (record) snapshots instead of checking against them.
        #[arg(long)]
        update: bool,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: SnapshotOutputFormat,
    },
    /// Bisect an integer CONSTANT to the minimal failing value
    ///
    /// Binary search over a CONSTANT integer value to find the minimal
    /// configuration that triggers an invariant violation or exceeds a
    /// state-count threshold.
    #[command(hide = true)]
    Bisect {
        /// TLA+ source file to check.
        file: PathBuf,
        /// Configuration file (.cfg) to use as a template.
        #[arg(short, long)]
        config: PathBuf,
        /// Name of the CONSTANT integer to bisect over.
        #[arg(long)]
        constant: String,
        /// Lower bound of the search range (inclusive).
        #[arg(long)]
        low: i64,
        /// Upper bound of the search range (inclusive).
        #[arg(long)]
        high: i64,
        /// Bisect for the minimal value where state count exceeds this threshold.
        /// If omitted, bisects for the minimal invariant violation.
        #[arg(long, value_name = "STATES")]
        state_count: Option<u64>,
        /// Per-check timeout in seconds (0 = no timeout).
        #[arg(long, default_value = "60")]
        timeout: u64,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: BisectOutputFormat,
    },
    /// Merge two TLA+ specifications at the AST level
    ///
    /// Applies a "patch" spec onto a "base" spec by unioning declarations
    /// and detecting operator conflicts.
    #[command(hide = true)]
    Merge {
        /// Base TLA+ source file.
        base: PathBuf,
        /// Patch TLA+ source file to merge onto the base.
        patch: PathBuf,
        /// Output file. If not specified, prints to stdout.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Force merge: use patch version for all conflicting operators.
        #[arg(long)]
        force: bool,
        /// Output format: human (default) or json (structured report).
        #[arg(long, value_enum, default_value = "human")]
        format: MergeOutputFormat,
    },
    /// Run deep pre-flight validation of a TLA+ spec
    ///
    /// Runs 12 semantic validation checks (V001-V012) that catch errors
    /// before model checking: undefined operators, undeclared variables,
    /// config/spec mismatches, missing EXTENDS, arity errors, etc.
    #[command(hide = true)]
    Validate {
        /// TLA+ source file to validate.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: ValidateOutputFormat,
        /// Treat warnings as errors (exit code 1 on any warning).
        #[arg(long)]
        strict: bool,
    },
    /// Generate a protocol template TLA+ specification
    ///
    /// Creates a complete, runnable TLA+ spec and matching .cfg file
    /// for common distributed systems patterns.
    #[command(hide = true)]
    Template {
        /// Protocol template to generate.
        kind: TemplateKind,
        /// Module name for the generated spec (default: template kind name).
        #[arg(long, default_value = "Spec")]
        name: String,
        /// Number of processes (default: 3).
        #[arg(long, default_value = "3")]
        processes: u32,
        /// Output directory for generated files (default: current directory).
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,
        /// Print to stdout instead of writing files.
        #[arg(long)]
        stdout: bool,
    },
    /// Analyze a TLA+ spec for deadlock conditions
    ///
    /// Quick mode performs static analysis; full mode runs model checking.
    #[command(hide = true)]
    Deadlock {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Analysis mode: quick (static) or full (model checking).
        #[arg(long, value_enum, default_value = "quick")]
        mode: DeadlockMode,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: DeadlockOutputFormat,
    },
    /// Generate an abstract view of a TLA+ specification
    ///
    /// Extracts state variables, actions, invariants, and transitions
    /// into a high-level summary suitable for documentation and review.
    #[command(hide = true)]
    Abstract {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: human (default), json, or mermaid.
        #[arg(long, value_enum, default_value = "human")]
        format: AbstractOutputFormat,
        /// Detail level: brief, normal (default), or full.
        #[arg(long, value_enum, default_value = "normal")]
        detail: AbstractDetail,
    },
    /// Import a specification from another format into TLA+
    ///
    /// Supports JSON state machines, basic Promela, and basic Alloy.
    #[command(hide = true)]
    Import {
        /// Input file to import.
        file: PathBuf,
        /// Source format.
        #[arg(long, value_enum)]
        from: ImportFormat,
        /// Output file. If not specified, prints to stdout.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    /// Analyze which actions could reach a target predicate
    ///
    /// Unlike `check` (which finds violations), `witness` reasons about positive
    /// examples: it identifies the target operator, reports which actions
    /// reference it and how deep a reaching search would need to be. Concrete
    /// trace generation (bounded BFS) is future work.
    #[command(hide = true)]
    Witness {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Operator name to find a witness for (e.g., "TypeOK").
        #[arg(long)]
        target: String,
        /// Maximum search depth (default: 20).
        #[arg(long, default_value = "20")]
        max_depth: usize,
        /// Number of witness traces to find (default: 1).
        #[arg(long, default_value = "1")]
        count: usize,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: WitnessOutputFormat,
    },
    /// Compare the declarations of two TLA+ specs
    ///
    /// Shows semantic differences: added/removed/modified operators,
    /// variables, constants, EXTENDS, and INSTANCE declarations. For a
    /// diff with impact analysis on verification obligations, see
    /// `ty model-diff`.
    #[command(hide = true)]
    Compare {
        /// Left (base) TLA+ file.
        left: PathBuf,
        /// Right (new) TLA+ file.
        right: PathBuf,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: CompareOutputFormat,
    },
    /// Inline INSTANCE modules into a self-contained TLA+ file
    ///
    /// Resolves INSTANCE declarations by finding and inlining the
    /// referenced module's operators, variables, and constants.
    #[command(hide = true)]
    Inline {
        /// TLA+ source file.
        file: PathBuf,
        /// Output file. If not specified, prints to stdout.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Keep comments from inlined modules.
        #[arg(long)]
        keep_comments: bool,
    },
    /// Analyze scope and dependency graphs of a TLA+ spec
    ///
    /// Parses a spec and produces a detailed scope analysis: operator
    /// call graph, variable reads/writes, constant usage, dead operators,
    /// and reachability from Init/Next/invariants.
    #[command(hide = true)]
    Scope {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Output format: human (default), json, or dot.
        #[arg(long, value_enum, default_value = "human")]
        format: ScopeOutputFormat,
    },
    /// Generate constrained TLC config files from a TLA+ spec
    ///
    /// Generates the configs by analyzing the spec. Three strategies: minimize (smallest useful constants), incremental
    /// (progressively larger configs), symmetric (add SYMMETRY declarations).
    #[command(hide = true)]
    Constrain {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Configuration file (.cfg) to read/modify.
        #[arg(short, long)]
        config: PathBuf,
        /// Constraining strategy: minimize (default), incremental, or symmetric.
        #[arg(long, value_enum, default_value = "minimize")]
        strategy: ConstrainStrategy,
        /// Output directory for generated configs (incremental mode).
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,
    },
    /// Audit a TLA+ specification directory project-wide
    ///
    /// Scans a directory for .tla and .cfg files and runs structural checks:
    /// spec/config pairing, parse validation, naming conventions, complexity
    /// metrics, anti-pattern detection, and config completeness.
    #[command(hide = true)]
    Audit {
        /// Directory to audit (default: current directory).
        #[arg(default_value = ".")]
        dir: PathBuf,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: AuditOutputFormat,
    },
    /// Run a spec under every engine and assert verdict parity
    ///
    /// Cross-mode verdict-parity self-check: runs a spec under the trusted
    /// interpreter oracle, the default fused engine, and the native trust-cg
    /// backend, and asserts they reach the SAME verdict.
    ///
    /// This is self-service differential soundness — the on-demand check that catches a
    /// single engine silently diverging (a symbolic-safe lane masking a deadlock, a
    /// native-codegen successor divergence). Exits 0 on agreement, 1 on a disagreement
    /// between conclusive engines.
    #[command(hide = true)]
    Parity {
        /// TLA+ source file (omit when using --corpus).
        file: Option<PathBuf>,
        /// Configuration file (.cfg). Defaults to `<spec>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Sweep every `<name>.tla` + `<name>.cfg` pair in this directory instead of a
        /// single file, reporting a parity scorecard (exit 1 if any spec disagrees).
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// Per-engine timeout in seconds.
        #[arg(long, default_value = "60")]
        timeout: u64,
        /// Bound each engine's exploration to N states (0 = unbounded). Bounding makes
        /// large specs tractable, but a bounded engine reports `inconclusive` rather
        /// than a comparable verdict.
        #[arg(long, default_value = "0")]
        max_states: usize,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: AuditOutputFormat,
    },
    /// Generate random specs and hunt for engine divergence
    ///
    /// Differential fuzzer: generates random specs and runs cross-mode parity on
    /// each, hunting for an engine that diverges (a soundness bug). Deterministic
    /// from --seed.
    ///
    /// Divergent specs are kept as reproducible fixtures. Exits 0 when none diverge,
    /// 1 when at least one does.
    #[command(hide = true)]
    Fuzz {
        /// Seed for deterministic, reproducible generation.
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Number of specs to generate and differential-check.
        #[arg(long, default_value = "50")]
        count: usize,
        /// Directory to save any divergent specs into (kept across runs).
        #[arg(long)]
        keep: Option<PathBuf>,
    },
    /// Prove safety for all reachable states and re-check the proof
    ///
    /// Runs the inductive-safety prover; on success emits a `ty.cert/v1` certificate and
    /// re-validates it on the spot (proved + re-checked, not "trust me"). `--out` saves
    /// it for later `ty recheck`. Exits 0 PROVED, 1 if the proof fails its own re-check,
    /// 2 if the spec is not inductively provable.
    #[command(display_order = 40)]
    Prove {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg). Defaults to `<spec>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Optional path to save the `ty.cert/v1` certificate.
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Re-validate a ty.verdict/v1 or ty.cert/v1 artifact
    ///
    /// The unified minimal re-checker: re-validates any TY assurance artifact by its
    /// schema (`ty.verdict/v1` or `ty.cert/v1`), or prints the trusted computing base.
    ///
    /// Dispatches to the artifact's independent verifier. `--tcb` prints the small named
    /// kernel the re-check trusts (parser + evaluator + proof checker — not the model
    /// checker / JIT / SMT search). Exits 0 VERIFIED / 1 REJECTED / 2 INCONCLUSIVE.
    #[command(display_order = 43)]
    Recheck {
        /// Path to a `ty.verdict/v1` or `ty.cert/v1` artifact JSON.
        artifact: Option<PathBuf>,
        /// Print the declared trusted computing base and exit.
        #[arg(long)]
        tcb: bool,
    },
    /// Compose every independent re-check for one TLA+ spec
    ///
    /// A self-service trust report for a single spec.
    ///
    /// Runs cross-mode verdict parity AND re-checks the verdict from a second angle (a
    /// VIOLATED verdict's counterexample is replayed eval-only; a SAFE verdict is given
    /// a re-checkable inductive proof when provable), printing a scorecard. The
    /// self-service equivalent of a required gate. Exits 0 when every applicable check
    /// is consistent, 1 when any disagrees.
    #[command(display_order = 44)]
    Selfcheck {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg). Defaults to `<spec>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: AuditOutputFormat,
    },
    /// Analyze symmetry properties of CONSTANT sets
    ///
    /// Detects which model-value sets are used symmetrically (no ordering,
    /// no distinguished elements) and suggests SYMMETRY declarations to
    /// reduce the state space.
    #[command(hide = true)]
    Symmetry {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: SymmetryOutputFormat,
    },
    /// Analyze state-space partitioning for parallel checking
    ///
    /// Identifies natural partitioning boundaries for parallel or
    /// distributed checking by analyzing variable domains, action
    /// independence, and symmetry classes.
    #[command(hide = true)]
    Partition {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of partitions to suggest (default: 4).
        #[arg(long, default_value = "4")]
        partitions: usize,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: PartitionOutputFormat,
    },
    /// Run simulation traces and report statistical coverage
    ///
    /// Generates N random walks, collects action frequency distributions,
    /// variable value ranges, coverage estimates, and identifies "cold"
    /// actions that are rarely or never fired.
    #[command(name = "sim-report", hide = true)]
    SimReport {
        /// TLA+ source file to simulate.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of simulation traces (default: 1000).
        #[arg(short, long, default_value = "1000")]
        num_traces: usize,
        /// Maximum depth per trace (default: 100).
        #[arg(short, long, default_value = "100")]
        max_depth: usize,
        /// Output format: human (default) or json.
        #[arg(long, value_enum, default_value = "human")]
        format: SimReportOutputFormat,
    },
    /// Generate targeted traces matching specific patterns
    ///
    /// Unlike `check` (which finds violations) or `simulate` (which explores
    /// randomly), `trace-gen` generates traces matching specific patterns:
    /// target (reach a predicate), coverage (cover all actions), or random.
    #[command(name = "trace-gen", hide = true)]
    TraceGen {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Trace generation mode.
        #[arg(long, value_enum, default_value = "target")]
        mode: TraceGenMode,
        /// Target predicate expression (required for target mode).
        #[arg(long)]
        target: Option<String>,
        /// Number of traces to generate (default: 10).
        #[arg(long, default_value = "10")]
        count: usize,
        /// Maximum trace depth (default: 100).
        #[arg(long, default_value = "100")]
        max_depth: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: TraceGenOutputFormat,
    },
    /// Generate invariant candidates from Init/Next analysis
    ///
    /// Analyzes Init/Next to suggest type invariants, range preservation,
    /// monotonicity, conservation laws, and mutual exclusion patterns.
    /// Pass `--verify` to quick-check each candidate with a bounded model
    /// check. See also `ty invariantgen`, which suggests formulas from
    /// Init constraints, variable types, and naming patterns without
    /// verification.
    #[command(name = "inv-gen", hide = true)]
    InvGen {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Quick-check each candidate with a bounded model check.
        #[arg(long)]
        verify: bool,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: InvGenOutputFormat,
    },
    /// Analyze action dependencies and conflicts
    ///
    /// Decomposes the Next relation into individual actions and builds a
    /// dependency graph showing enables/disables, conflicts, and independence
    /// relationships. Independent actions are POR ample-set candidates.
    #[command(name = "action-graph", hide = true)]
    ActionGraph {
        /// TLA+ source file to analyze.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format: human (default), json, or dot.
        #[arg(long, value_enum, default_value = "human")]
        format: ActionGraphOutputFormat,
    },
    /// Verify one TLA+ spec refines another
    ///
    /// Refinement checking: checks the simulation relation from an implementation
    /// spec to an abstract spec — every implementation behavior, when mapped
    /// through the refinement mapping, must be a valid abstract behavior.
    #[command(display_order = 42)]
    Refine {
        /// Implementation TLA+ file.
        #[arg(name = "impl-file")]
        impl_file: PathBuf,
        /// Abstract TLA+ file.
        #[arg(name = "abstract-file")]
        abstract_file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Refinement mapping file.
        #[arg(short, long)]
        mapping: Option<PathBuf>,
        /// Maximum states to explore (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: RefineOutputFormat,
    },
    /// Diff two spec versions with verification impact analysis
    ///
    /// Shows what operators, variables, and invariants changed between spec
    /// versions, with impact analysis on verification obligations. For a
    /// plain declaration-level diff, see `ty compare`.
    #[command(name = "model-diff", hide = true)]
    ModelDiff {
        /// Old TLA+ file.
        #[arg(name = "old-file")]
        old_file: PathBuf,
        /// New TLA+ file.
        #[arg(name = "new-file")]
        new_file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ModelDiffOutputFormat,
    },
    /// Filter and query states from a model checking run
    ///
    /// Explores state space via bounded BFS and finds states matching
    /// user-specified filter predicates. Reports matching states with traces.
    #[command(name = "state-filter", hide = true)]
    StateFilter {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Filter expression (TLA+ predicate evaluated against each state).
        #[arg(short, long)]
        filter: String,
        /// Maximum states to explore (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Maximum matching states to report (default: 100).
        #[arg(long, default_value = "100")]
        max_results: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: StateFilterOutputFormat,
    },
    /// Analyze lasso-shaped counterexamples for liveness violations
    ///
    /// Decomposes a liveness violation trace into a stem (finite prefix)
    /// and loop (repeating cycle), reporting the lasso structure.
    #[command(hide = true)]
    Lasso {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Temporal property to check (overrides .cfg).
        #[arg(short, long)]
        property: Option<String>,
        /// Maximum states to explore (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: LassoOutputFormat,
    },
    /// Verify compositionally with assume-guarantee reasoning
    ///
    /// Decomposes the specification into action groups and verifies each
    /// group independently, checking that assumptions between groups hold.
    #[command(name = "assume-guarantee", hide = true)]
    AssumeGuarantee {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Maximum states to explore per group (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: AssumeGuaranteeOutputFormat,
    },
    /// Analyze a spec via predicate abstraction
    ///
    /// Constructs an abstract model by tracking boolean predicates over
    /// the state, reporting abstraction metrics and compression ratio.
    #[command(name = "predicate-abs", hide = true)]
    PredicateAbs {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Additional predicate expressions.
        #[arg(short, long)]
        predicate: Vec<String>,
        /// Maximum states to explore (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: PredicateAbsOutputFormat,
    },
    /// Explore states and report state-space statistics
    ///
    /// Reports variable count, total states, initial states, max depth,
    /// and exploration status.
    #[command(hide = true)]
    Census {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Maximum states to explore (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: CensusOutputFormat,
    },
    /// Check equivalence of two TLA+ specifications
    ///
    /// Runs model checking on both specs and compares state counts,
    /// depth, and initial states to determine if they produce identical
    /// state spaces.
    #[command(hide = true)]
    Equiv {
        /// First TLA+ source file.
        #[arg(name = "file-a")]
        file_a: PathBuf,
        /// Second TLA+ source file.
        #[arg(name = "file-b")]
        file_b: PathBuf,
        /// Configuration file for spec A (.cfg).
        #[arg(long)]
        config_a: Option<PathBuf>,
        /// Configuration file for spec B (.cfg).
        #[arg(long)]
        config_b: Option<PathBuf>,
        /// Maximum states to explore per spec (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: EquivOutputFormat,
    },
    /// Check whether a candidate invariant is inductive
    ///
    /// Verifies whether a candidate invariant holds over all reachable
    /// states. Reports whether the invariant is maintained by Init and
    /// all transitions.
    #[command(hide = true)]
    Induct {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Name of the candidate invariant operator.
        #[arg(short, long)]
        invariant: String,
        /// Maximum states to explore (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: InductOutputFormat,
    },
    /// Slice a spec to the portion relevant to a target
    ///
    /// Computes the transitive dependency closure of a target operator,
    /// identifying which variables, operators, and constants are needed.
    #[command(hide = true)]
    Slice {
        /// TLA+ source file.
        file: PathBuf,
        /// Target operator name to slice for.
        #[arg(short, long)]
        target: String,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: SliceOutputFormat,
    },
    /// Analyze reachability of a target predicate
    ///
    /// Checks whether the negation of a target predicate is reachable
    /// by adding it as an invariant and looking for violations.
    #[command(hide = true)]
    Reach {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Target predicate operator name.
        #[arg(short, long)]
        target: String,
        /// Maximum states to explore (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ReachOutputFormat,
    },
    /// Analyze the parallel composition of two TLA+ specs
    ///
    /// Analyzes two specifications for composition compatibility,
    /// reports shared variables and interface analysis.
    #[command(hide = true)]
    Compose {
        /// First TLA+ source file.
        #[arg(name = "file-a")]
        file_a: PathBuf,
        /// Second TLA+ source file.
        #[arg(name = "file-b")]
        file_b: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ComposeOutputFormat,
    },
    /// Unfold operator definitions, showing expanded bodies
    ///
    /// Displays the body of a target operator and its transitive
    /// dependencies up to a configurable depth.
    #[command(hide = true)]
    Unfold {
        /// TLA+ source file.
        file: PathBuf,
        /// Target operator name to unfold.
        #[arg(short, long)]
        target: String,
        /// Maximum unfolding depth (default: 5).
        #[arg(long, default_value = "5")]
        max_depth: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: UnfoldOutputFormat,
    },
    /// Project state space onto a variable subset
    ///
    /// Explores the state space and reports statistics for a projection
    /// onto a user-specified subset of variables.
    #[command(hide = true)]
    Project {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Variables to project onto.
        #[arg(short, long, required = true)]
        variable: Vec<String>,
        /// Maximum states to explore (default: 100000).
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ProjectOutputFormat,
    },
    /// Estimate state space bounds from type information
    ///
    /// Analyzes Init predicate structure and config to compute
    /// upper bounds on the state space size.
    #[command(hide = true)]
    Bound {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: BoundOutputFormat,
    },
    /// Run a spec in a sandboxed environment with limits
    ///
    /// Checks a TLA+ spec in a sandboxed environment with configurable
    /// state/depth/time resource limits and reports whether exploration
    /// stayed within bounds.
    #[command(hide = true)]
    Sandbox {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Maximum number of states to explore.
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Maximum BFS depth.
        #[arg(long, default_value = "100")]
        max_depth: usize,
        /// Timeout in seconds.
        #[arg(long, default_value = "30")]
        timeout: u64,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: SandboxOutputFormat,
    },
    /// Analyze temporal behavior timeline of a specification
    ///
    /// Extracts actions from the Next relation, identifies temporal
    /// properties, invariants, and fairness constraints.
    #[command(hide = true)]
    Timeline {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: TimelineOutputFormat,
    },
    /// Compute structural complexity metrics for a specification
    ///
    /// Reports operator count, nesting depth, expression size,
    /// variable count, quantifiers, and prime expressions.
    #[command(hide = true)]
    Metric {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: MetricOutputFormat,
    },
    /// Generate a configuration file scaffold from a specification
    ///
    /// Analyzes the spec to detect Init/Next, constants, invariant
    /// candidates, and produces a .cfg file.
    #[command(hide = true)]
    Scaffold {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ScaffoldOutputFormat,
    },
    /// Analyze stuttering and UNCHANGED patterns
    ///
    /// Reports which variables each action primes, detects stuttering
    /// steps, and flags variables that are never modified.
    #[command(hide = true)]
    Stutter {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: StutterOutputFormat,
    },
    /// Detect quorum and majority patterns in distributed specs
    ///
    /// Scans for Cardinality thresholds, SUBSET selections, voting
    /// variables, and quorum-related operators.
    #[command(hide = true)]
    Quorum {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: QuorumOutputFormat,
    },
    /// Compute state fingerprints for debugging and comparison
    ///
    /// Runs model checking to a limit and reports an aggregate
    /// fingerprint of the state space for cross-implementation comparison.
    #[command(hide = true)]
    Fingerprint {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Maximum number of states to explore.
        #[arg(long, default_value = "10000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: FingerprintOutputFormat,
    },
    /// Normalize a specification to canonical form
    ///
    /// Outputs the spec with sorted constants, variables, and operators
    /// in a canonical ordering for diffing and comparison.
    #[command(hide = true)]
    Normalize {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: NormalizeOutputFormat,
    },
    /// Analyze counterexamples from model checking
    ///
    /// Runs model checking and classifies any violation found by type,
    /// trace length, and involved variables.
    #[command(hide = true)]
    Countex {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Maximum number of states to explore.
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: CountexOutputFormat,
    },
    /// Show a state-space exploration heatmap
    ///
    /// Shows per-action complexity contribution to the state space.
    #[command(hide = true)]
    Heatmap {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Maximum number of states to explore.
        #[arg(long, default_value = "10000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: HeatmapOutputFormat,
    },
    /// Detect distributed protocol patterns
    ///
    /// Scans for message passing, leader election, consensus,
    /// mutual exclusion, and state machine patterns.
    #[command(hide = true)]
    Protocol {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ProtocolOutputFormat,
    },
    /// Display operator call hierarchy
    ///
    /// Builds and visualizes the call graph between operators,
    /// showing roots, leaves, and dependency chains.
    #[command(hide = true)]
    Hierarchy {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: HierarchyOutputFormat,
    },
    /// Build a cross-reference index of definitions and usages
    ///
    /// Shows where each operator, variable, and constant is defined
    /// and which operators reference it. Flags unused definitions.
    #[command(hide = true)]
    Crossref {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: CrossrefOutputFormat,
    },
    /// Suggest invariants from types and naming patterns
    ///
    /// Analyzes Init constraints, variable types, and naming patterns
    /// to suggest invariant formulas for model checking. See also
    /// `ty inv-gen`, which derives candidates from Init/Next analysis
    /// and can bounded-check them with `--verify`.
    #[command(hide = true)]
    Invariantgen {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: InvariantgenOutputFormat,
    },
    /// Detect specification drift between two versions
    ///
    /// Compares two TLA+ files and reports added, removed, and
    /// modified operators, variables, and constants.
    #[command(hide = true)]
    Drift {
        /// First (baseline) TLA+ source file.
        file_a: PathBuf,
        /// Second (current) TLA+ source file.
        file_b: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: DriftOutputFormat,
    },
    /// Analyze safety properties and invariants
    ///
    /// Reports configured invariants, classifies them by type,
    /// and suggests additional invariant candidates.
    #[command(hide = true)]
    Safety {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: SafetyOutputFormat,
    },
    /// Analyze liveness properties and fairness constraints
    ///
    /// Reports temporal properties, classifies them (leads-to,
    /// recurrence, stability), and detects fairness.
    #[command(name = "liveness-check", hide = true)]
    LivenessCheck {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: LivenesscheckOutputFormat,
    },
    /// Translate specification to pseudocode
    ///
    /// Converts TLA+ operators into readable pseudocode with
    /// assignments, conditions, and loops.
    #[command(hide = true)]
    Translate {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: TranslateOutputFormat,
    },
    /// Display liveness tableau construction
    ///
    /// Analyzes temporal properties and shows the tableau structure
    /// used for liveness checking, including SCC requirements.
    #[command(hide = true)]
    Tableau {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: TableauOutputFormat,
    },
    /// Extract the action alphabet from the Next relation
    ///
    /// Lists all distinct actions with their parameter counts.
    #[command(hide = true)]
    Alphabet {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: AlphabetOutputFormat,
    },
    /// Compute action weights for guided search
    ///
    /// Assigns weights based on structural complexity and variable
    /// coverage for use in heuristic-guided model checking.
    #[command(hide = true)]
    Weight {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: WeightOutputFormat,
    },
    /// Absorb constant values from config into the spec
    ///
    /// Reads constant assignments and displays operators with
    /// constants replaced by their configured values.
    #[command(hide = true)]
    Absorb {
        /// TLA+ source file.
        file: PathBuf,
        /// Configuration file (.cfg).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: AbsorbOutputFormat,
    },
    /// Cluster operators by variable affinity
    ///
    /// Groups operators that reference the same variables,
    /// identifying natural functional clusters.
    #[command(hide = true)]
    Cluster {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ClusterOutputFormat,
    },
    /// Rename an identifier across the specification
    ///
    /// Performs word-boundary-aware renaming of operators,
    /// variables, or constants.
    #[command(hide = true)]
    Rename {
        /// TLA+ source file.
        file: PathBuf,
        /// Current name of the identifier.
        #[arg(long)]
        from: String,
        /// New name for the identifier.
        #[arg(long)]
        to: String,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: RenameOutputFormat,
    },
    /// Summarize the reachable state set
    ///
    /// Runs bounded model checking and reports summary statistics
    /// about the reachable state set.
    #[command(hide = true)]
    Reachset {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Maximum number of states to explore.
        #[arg(long, default_value = "1000000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ReachsetOutputFormat,
    },
    /// Extract enabling conditions (guards) from actions
    ///
    /// Analyzes each action in the Next-state relation to identify
    /// the enabling condition (guard).
    #[command(hide = true)]
    Guard {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: GuardOutputFormat,
    },
    /// Detect potential symmetry in a specification
    ///
    /// Analyzes constants, variables, and operators to identify
    /// potential symmetry sets for the SYMMETRY keyword.
    #[command(name = "symmetry-detect", hide = true)]
    SymmetryDetect {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: SymmetrydetectOutputFormat,
    },
    /// Verify deadlock freedom with diagnostics
    ///
    /// Runs model checking focused on deadlock detection and provides
    /// detailed analysis of deadlock states when found.
    #[command(name = "deadlock-free", hide = true)]
    DeadlockFree {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Maximum number of states to explore.
        #[arg(long, default_value = "1000000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: DeadlockfreeOutputFormat,
    },
    /// Count transitions per action in the Next-state relation
    ///
    /// Analyzes the structure of Next to identify and count
    /// individual action disjuncts.
    #[command(name = "action-count", hide = true)]
    ActionCount {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ActioncountOutputFormat,
    },
    /// Validate constant assignments between spec and config
    ///
    /// Checks that all CONSTANT declarations have assignments
    /// and no config assignments reference non-existent constants.
    #[command(name = "const-check", hide = true)]
    ConstCheck {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ConstcheckOutputFormat,
    },
    /// Summarize a specification's structure and contents
    ///
    /// Displays module name, EXTENDS, constants, variables, operators,
    /// and structural statistics.
    #[command(name = "spec-info", hide = true)]
    SpecInfo {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: SpecinfoOutputFormat,
    },
    /// Track variable read/write usage across operators
    ///
    /// For each variable, identifies which operators read (unprimed)
    /// and write (primed) it.
    #[command(name = "var-track", hide = true)]
    VarTrack {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: VartrackOutputFormat,
    },
    /// Generate a .cfg configuration file from spec analysis
    ///
    /// Produces Init/Next, constant declarations, and invariant
    /// candidates by analyzing the spec.
    #[command(name = "cfg-gen", hide = true)]
    CfgGen {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: CfggenOutputFormat,
    },
    /// Emit the operator dependency graph
    ///
    /// Produces a dependency graph showing which operators
    /// call which other operators.
    #[command(name = "dep-graph", hide = true)]
    DepGraph {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: DepgraphOutputFormat,
    },
    /// Count initial states
    ///
    /// Enumerates initial states and reports the count without
    /// running full model checking.
    #[command(name = "init-count", hide = true)]
    InitCount {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: InitcountOutputFormat,
    },
    /// Compute average branching factor of the state graph
    ///
    /// Runs bounded model checking and reports the average number
    /// of successor states per state.
    #[command(name = "branch-factor", hide = true)]
    BranchFactor {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Maximum number of states to explore.
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: BranchfactorOutputFormat,
    },
    /// Export state graph summary
    ///
    /// Runs bounded model checking and exports state graph statistics.
    #[command(name = "state-graph", hide = true)]
    StateGraph {
        /// TLA+ source file.
        file: PathBuf,
        /// Path to a .cfg configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Maximum number of states to explore.
        #[arg(long, default_value = "10000")]
        max_states: usize,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: StategraphOutputFormat,
    },
    /// Extract and classify predicates
    ///
    /// Identifies boolean-valued operators and classifies them
    /// as state predicates, action predicates, or temporal formulas.
    #[command(hide = true)]
    Predicate {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: PredicateOutputFormat,
    },
    /// Display module structure and metadata
    ///
    /// Shows module name, EXTENDS list, and unit counts by type.
    #[command(name = "module-info", hide = true)]
    ModuleInfo {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ModuleinfoOutputFormat,
    },
    /// List operators with their arities
    ///
    /// Extracts all operator definitions and displays their names
    /// and parameter counts.
    #[command(name = "op-arity", hide = true)]
    OpArity {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: OparityOutputFormat,
    },
    /// Detect unused variables
    ///
    /// Identifies variables declared in VARIABLE that are never
    /// referenced in any operator body.
    #[command(name = "unused-var", hide = true)]
    UnusedVar {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: UnusedvarOutputFormat,
    },
    /// Count expression nodes by type
    ///
    /// Walks the AST and counts how many nodes of each expression
    /// type appear, providing a structural complexity profile.
    #[command(name = "expr-count", hide = true)]
    ExprCount {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ExprcountOutputFormat,
    },
    /// Measure specification size metrics
    ///
    /// Reports line counts, character counts, and structural
    /// size metrics.
    #[command(name = "spec-size", hide = true)]
    SpecSize {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: SpecsizeOutputFormat,
    },
    /// List all CONSTANT declarations
    ///
    /// Extracts and displays all CONSTANT declarations
    /// including their arity.
    #[command(name = "const-list", hide = true)]
    ConstList {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ConstlistOutputFormat,
    },
    /// List all VARIABLE declarations
    ///
    /// Extracts and displays all VARIABLE declarations.
    #[command(name = "var-list", hide = true)]
    VarList {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: VarlistOutputFormat,
    },
    /// Detect unused constants
    #[command(name = "unused-const", hide = true)]
    UnusedConst {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: UnusedconstOutputFormat,
    },
    /// Compute maximum AST nesting depth per operator
    #[command(name = "ast-depth", hide = true)]
    AstDepth {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: AstdepthOutputFormat,
    },
    /// List all operator definitions
    #[command(name = "op-list", hide = true)]
    OpList {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: OplistOutputFormat,
    },
    /// List EXTENDS dependencies
    #[command(hide = true)]
    Extends {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: ExtendsOutputFormat,
    },
    /// Count set operation usage
    #[command(name = "set-ops", hide = true)]
    SetOps {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: SetopsOutputFormat,
    },
    /// Count quantifier usage per operator
    #[command(name = "quant-count", hide = true)]
    QuantCount {
        /// TLA+ source file.
        file: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        format: QuantcountOutputFormat,
    },
    /// Count primed variable references per operator
    #[command(name = "prime-count", hide = true)]
    PrimeCount {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: PrimecountOutputFormat,
    },
    /// Count IF-THEN-ELSE expressions per operator
    #[command(name = "if-count", hide = true)]
    IfCount {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: IfcountOutputFormat,
    },
    /// Count LET-IN definitions per operator
    #[command(name = "let-count", hide = true)]
    LetCount {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: LetcountOutputFormat,
    },
    /// Count CHOOSE expressions
    #[command(name = "choose-count", hide = true)]
    ChooseCount {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: ChoosecountOutputFormat,
    },
    /// Count CASE expressions
    #[command(name = "case-count", hide = true)]
    CaseCount {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: CasecountOutputFormat,
    },
    /// Count record/function operations
    #[command(name = "record-ops", hide = true)]
    RecordOps {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: RecordopsOutputFormat,
    },
    /// Count temporal operator usage
    #[command(name = "temporal-ops", hide = true)]
    TemporalOps {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: TemporalopsOutputFormat,
    },
    /// Find UNCHANGED clauses and their variables
    #[command(hide = true)]
    Unchanged {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: UnchangedOutputFormat,
    },
    /// Find ENABLED expressions
    #[command(hide = true)]
    Enabled {
        file: PathBuf,
        #[arg(long, value_enum, default_value = "human")]
        format: EnabledOutputFormat,
    },
    /// Check a concurrent model (ConcurrentModel JSON from tRust)
    ///
    /// Reads a ConcurrentModel JSON file, generates a TLA+ module, runs the
    /// model checker, and reports verification results with source-mapped
    /// counterexamples.
    #[command(name = "thread-check", hide = true)]
    ThreadCheck {
        /// Path to ConcurrentModel JSON file.
        file: PathBuf,
        /// Number of BFS worker threads (0 = auto).
        #[arg(long, default_value = "0")]
        workers: usize,
        /// Maximum states to explore (0 = unlimited).
        #[arg(long, default_value = "0")]
        max_states: usize,
        /// Maximum BFS depth (0 = unlimited).
        #[arg(long, default_value = "0")]
        max_depth: usize,
        /// Print generated TLA+ module to stdout.
        #[arg(long)]
        emit_tla: bool,
        /// Output format.
        #[arg(long, value_enum, default_value = "human")]
        output: ThreadCheckOutputFormat,
    },
    /// List every ty command, grouped by purpose
    ///
    /// Prints the complete catalog of top-level commands: the curated
    /// workflow groups shown in `--help` first, then the specialized
    /// families hidden from it, then internal repo/CI tooling — one line
    /// per command, with descriptions pulled live from the CLI definition.
    #[command(display_order = 80)]
    Commands {
        /// Emit the catalog as JSON `{group, name, about}` records.
        #[arg(long)]
        json: bool,
    },
}

/// Subcommands for `ty cache` — trust-codegen on-disk compilation cache.
///
/// The cache stores compiled trust-ir modules at
/// `~/.cache/ty/compiled/<digest>.{dylib,so,dll}` plus a JSON sidecar
/// describing the compilation context. See design doc §7.
#[derive(Debug, Subcommand)]
pub(crate) enum CacheAction {
    /// Remove all cached artifacts under `~/.cache/ty/compiled/`.
    ///
    /// Prints a per-extension count of removed files. Safe to run while
    /// other `ty` processes are not actively loading artifacts (each
    /// artifact is self-contained once loaded).
    Clear,
    /// List cached artifacts with their digests, opt levels, and sizes.
    ///
    /// Reads the JSON sidecars without touching the dynamic libraries,
    /// so it is cheap even for large caches.
    List,
    /// Print the cache root directory and exit. Respects `TY_CACHE_DIR`.
    Path,
}

/// Corpus management subcommands for `ty corpus`.
#[derive(Debug, Subcommand)]
pub(crate) enum CorpusAction {
    /// Download + extract the benchmark corpus to the install dir.
    ///
    /// Default mode downloads the published release asset and verifies its
    /// sha256; `--from-upstream` instead clones tlaplus/Examples and checks out
    /// the recorded pin (matches the TLC baselines in spec_baseline.json).
    Fetch {
        /// Install directory (default: `$TLAPLUS_EXAMPLES` or `~/tlaplus-examples`).
        /// The corpus extracts so `<dest>/specifications/...` resolves.
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Clone tlaplus/Examples at the pinned commit instead of downloading
        /// the release asset.
        #[arg(long)]
        from_upstream: bool,
        /// Re-fetch even if a corpus is already present at the install dir.
        #[arg(long)]
        force: bool,
    },
    /// Print the resolved corpus install directory and exit.
    Path {
        /// Install directory override (see `fetch --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
    },
    /// Verify a corpus is present and non-empty at the install dir.
    Verify {
        /// Install directory override (see `fetch --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
    },
    /// Run the CERTIFY census over every `.cfg` spec in the corpus: the
    /// corpus-evaluation table as a reproducible self-service command
    /// instead of a session artifact.
    ///
    /// Each `.cfg` is paired with its `.tla` (same-stem first, then a recorded
    /// best-effort fallback — the pairing rule used is printed per row, never
    /// guessed silently), run through the in-process `ty certify` pipeline with
    /// a per-spec wall-clock timeout, and reported as one row: outcome tier
    /// (derived from the actual certify verdicts — kernel-certified variants /
    /// smt-certified / declined / timeout / unpaired), blocking-feature
    /// categories, and elapsed ms. Honest by construction: the header names the
    /// corpus pin and the build features (a build without `ay`/`clean-cic`
    /// reaches fewer tiers and SAYS so), and the summary totals always add up
    /// to the number of `.cfg` files found.
    Sweep {
        /// Install directory override (see `fetch --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Per-spec wall-clock timeout in seconds. A spec that exceeds it is
        /// marked `timeout` and the sweep keeps going (the certify worker
        /// thread is leaked — acceptable for a census run).
        #[arg(long, default_value_t = 60)]
        timeout: u64,
        /// Worker threads (default 1 = the single-threaded CI shape).
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        /// Only sweep specs whose corpus-relative path contains this substring.
        #[arg(long)]
        filter: Option<String>,
        /// Output format.
        #[arg(long, value_enum, default_value_t = CorpusSweepFormat::Table)]
        format: CorpusSweepFormat,
        /// Write the report to this path instead of stdout (the headline
        /// counts still print to stdout).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Preflight the corpus for TY-vs-TLC comparability, one row at a time.
    ///
    /// For every eligible row of `strict_corpus_manifest.json`, answers whether
    /// it can be compared against TLC at all, and whether we already know where
    /// it stands. Three things make a row non-comparable, and the definition of
    /// done requires each be classified explicitly rather than silently omitted:
    ///
    /// * `tlc-unparseable` — TLC's parser cannot resolve a module. 25 eligible
    ///   rows EXTEND TLAPS / FiniteSetTheorems / NaturalsInduction; the missing
    ///   module is named. Fix with `ty install-tlc proof-library`.
    /// * `parity-impossible` — TY refuses TLC's declared SYMMETRY because a
    ///   property needs the genuine liveness checker (that orbit quotient is
    ///   unsound for liveness). TY explores the full space, TLC explores the
    ///   quotient, so exact distinct-state parity can never hold. This is a
    ///   soundness premium, NOT a performance loss.
    /// * `unmeasured` — comparable, but no TY runtime exists yet.
    ///
    /// Both engine-side checks run the real tools (SANY for TLC; `ty check
    /// --max-states 1` for the symmetry decision, which stops after model
    /// preparation), so the answers cannot drift from what a collection run
    /// would actually do.
    ///
    /// Also reconciles the manifest against the baseline that `ty supremacy
    /// matrix` classifies, and reports which TLA library resolved and whether
    /// it is strict-qualified (upstream + pinned) or the repo's first-party
    /// stub set.
    Doctor {
        /// Corpus install directory override (see `fetch --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Strict corpus manifest (default:
        /// `tests/tlc_comparison/strict_corpus_manifest.json`).
        #[arg(long)]
        manifest: Option<PathBuf>,
        /// Baseline whose TY-runtime coverage is reported (default:
        /// `tests/tlc_comparison/spec_baseline.json`).
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// TLC jar (default: `~/tlaplus/tytools.jar`).
        #[arg(long)]
        tlc_jar: Option<PathBuf>,
        /// CommunityModules jar (default: `~/tlaplus/CommunityModules.jar`).
        #[arg(long)]
        community_modules: Option<PathBuf>,
        /// TLA library directory given to both tools. Default resolution order:
        /// `TLA_LIBRARY`, `TLA_PLUS_LIBRARY`, the installed upstream proof
        /// library (`~/tlaplus/tla-library`), then the repo's first-party
        /// `test_specs/tla_library`.
        #[arg(long)]
        tla_library: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value_t = CorpusDoctorFormat::Table)]
        format: CorpusDoctorFormat,
        /// `warn` reports and exits 0; `enforce` exits non-zero if any eligible
        /// row is not ready for strict collection.
        #[arg(long, value_enum, default_value_t = SupremacyMode::Warn)]
        mode: SupremacyMode,
        /// Parallel probe workers.
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        /// Only check rows whose name or cfg path contains this substring.
        #[arg(long)]
        filter: Option<String>,
        /// Skip the TLC parse probe (no JDK available, or a fast re-check of
        /// only the symmetry and measurement axes).
        #[arg(long)]
        skip_parse: bool,
        /// Write the report to this path instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

/// Output format for `ty corpus doctor`.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum CorpusDoctorFormat {
    /// Human-readable toolchain block, reconciliation, and per-row table.
    #[default]
    Table,
    /// Structured JSON (`ty.corpus-doctor/v1`).
    Json,
    /// Markdown, ready to paste into a burndown or session report.
    Markdown,
}

/// Output format for `ty corpus sweep`.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub(crate) enum CorpusSweepFormat {
    /// Human-readable per-spec table + summary (default).
    #[default]
    Table,
    /// Structured JSON (`ty.corpus-sweep/v1`).
    Json,
    /// Markdown: headline + blocking-feature census tables, ready to paste
    /// into a corpus-evaluation report.
    Markdown,
}

/// TLC management subcommands for `ty install-tlc` (alias: `ty tlc`).
#[derive(Debug, Subcommand)]
pub(crate) enum TlcAction {
    /// Download + install `tytools.jar` (TLC) and `CommunityModules.jar`.
    ///
    /// `tla2tools.jar` comes from the nightly channel (the upstream stable
    /// release is broken); it is verify-functional-checked by running `tlc2.TLC
    /// -h`. `CommunityModules.jar` is pinned by sha256. Both land at the paths
    /// `ty supremacy compare` auto-discovers under the install dir.
    Install {
        /// Install directory (default: `~/tlaplus`). Jars land at
        /// `<dest>/tytools.jar` and `<dest>/CommunityModules.jar`.
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Re-download even if the jars are already present.
        #[arg(long)]
        force: bool,
        /// Also install the upstream TLA+ proof library (see `proof-library`).
        ///
        /// Recommended for corpus work: without it TLC cannot parse the 25
        /// eligible rows that EXTEND TLAPS / FiniteSetTheorems /
        /// NaturalsInduction.
        #[arg(long)]
        with_proof_library: bool,
    },
    /// Print the resolved install directory and exit.
    Path {
        /// Install directory override (see `install --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
    },
    /// Verify the TLC jars are present (and TLC runs) at the install dir.
    Verify {
        /// Install directory override (see `install --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
    },
    /// Install the upstream TLA+ proof-system module library into
    /// `<dest>/tla-library`.
    ///
    /// This is `tlaplus/tlapm`'s `library/` at a pinned commit — the genuine
    /// `TLAPS.tla`, `FiniteSetTheorems.tla`, `NaturalsInduction.tla` and
    /// companions — with every module verified against a per-file sha256 pin
    /// (a tarball digest would not survive GitHub recompression).
    ///
    /// Why you want it: 25 of the 141 eligible rows in the strict corpus
    /// manifest do not parse under TLC without these modules. The repo also
    /// ships a first-party stub set at `test_specs/tla_library` that makes
    /// those rows run, but a TY-authored artifact mediating 18% of the claim
    /// corpus is not something strict evidence should depend on. With this
    /// library installed, all 141 eligible rows parse from upstream sources.
    ProofLibrary {
        /// Install directory override (see `install --dest`). The library lands
        /// at `<dest>/tla-library`.
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Re-download even if a verified library is already present.
        #[arg(long)]
        force: bool,
    },
    /// Verify the installed proof library matches its per-file sha256 pin.
    VerifyProofLibrary {
        /// Install directory override (see `install --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
    },
}

/// Apalache management subcommands for `ty install-apalache` (alias: `ty apalache`).
#[derive(Debug, Subcommand)]
pub(crate) enum ApalacheAction {
    /// Download + install the pinned Apalache release.
    ///
    /// Fetches the sha256-verified release tarball and unpacks it so
    /// `<dest>/bin/apalache-mc` resolves, then verify-functional-checks it by
    /// running `apalache-mc version`.
    Install {
        /// Install directory (default: `~/apalache`). The launcher lands at
        /// `<dest>/bin/apalache-mc`.
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Re-download even if Apalache is already present.
        #[arg(long)]
        force: bool,
    },
    /// Print the resolved install directory and exit.
    Path {
        /// Install directory override (see `install --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
    },
    /// Verify Apalache is present (and runs) at the install dir.
    Verify {
        /// Install directory override (see `install --dest`).
        #[arg(long)]
        dest: Option<PathBuf>,
    },
}

/// Refactoring action subcommands for `ty refactor`.
#[derive(Debug, Subcommand)]
pub(crate) enum RefactorAction {
    /// Extract an expression into a new named operator.
    #[command(name = "extract-operator")]
    ExtractOperator {
        /// TLA+ source file to refactor.
        #[arg(long)]
        file: PathBuf,
        /// Expression text to extract (must appear verbatim in the source).
        #[arg(long)]
        expr: String,
        /// Name for the new operator.
        #[arg(long)]
        name: String,
        /// Write output to a file instead of showing a diff.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Modify the file in place.
        #[arg(long, conflicts_with = "output")]
        in_place: bool,
        /// Skip the preview diff output.
        #[arg(long)]
        no_preview: bool,
    },
    /// Rename an operator, variable, or constant throughout the spec.
    Rename {
        /// TLA+ source file to refactor.
        #[arg(long)]
        file: PathBuf,
        /// Current name of the operator/variable/constant.
        #[arg(long)]
        from: String,
        /// New name to replace it with.
        #[arg(long)]
        to: String,
        /// Write output to a file instead of showing a diff.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Modify the file in place.
        #[arg(long, conflicts_with = "output")]
        in_place: bool,
        /// Skip the preview diff output.
        #[arg(long)]
        no_preview: bool,
    },
    /// Inline a simple (zero-parameter) operator.
    Inline {
        /// TLA+ source file to refactor.
        #[arg(long)]
        file: PathBuf,
        /// Name of the operator to inline.
        #[arg(long)]
        name: String,
        /// Write output to a file instead of showing a diff.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Modify the file in place.
        #[arg(long, conflicts_with = "output")]
        in_place: bool,
        /// Skip the preview diff output.
        #[arg(long)]
        no_preview: bool,
    },
    /// Remove all unused operators from the spec.
    Cleanup {
        /// TLA+ source file to refactor.
        #[arg(long)]
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Write output to a file instead of showing a diff.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Modify the file in place.
        #[arg(long, conflicts_with = "output")]
        in_place: bool,
        /// Skip the preview diff output.
        #[arg(long)]
        no_preview: bool,
    },
}

#[derive(Debug, clap::Args)]
pub(crate) struct DiagnoseArgs {
    /// Path to spec_baseline.json.
    #[arg(long, default_value = "tests/tlc_comparison/spec_baseline.json")]
    pub baseline: PathBuf,
    /// Output format: human (default) or json.
    #[arg(long, value_enum, default_value = "human")]
    pub output: DiagnoseOutputFormat,
    /// Default timeout in seconds for specs without a baseline override.
    /// Per-spec `diagnose_timeout_seconds` in the baseline can extend beyond
    /// this value for slow specs. Default is 1800s (30 min). A full
    /// `--category small` run that leaves this at the default is bounded to
    /// 120s per spec so it can produce closure evidence for non-pass rows.
    #[arg(long, default_value = "1800")]
    pub timeout: u64,
    /// Filter by category (small, medium, large, xlarge).
    #[arg(long)]
    pub category: Option<String>,
    /// Run only specific spec(s) by name. Can be repeated.
    #[arg(long, conflicts_with = "spec_list")]
    pub spec: Vec<String>,
    /// Read spec names from a file (one per line, # comments, blank lines skipped).
    #[arg(long, conflicts_with = "spec")]
    pub spec_list: Option<PathBuf>,
    /// Retry timed-out specs up to N times. Specs that pass on retry are
    /// classified as `flaky_timeout` instead of `timeout`.
    #[arg(long, default_value = "0")]
    pub retries: u32,
    /// Number of concurrent spec runs (minimum 1). Full `--category small`
    /// runs that leave this at the default use 4-way concurrency.
    #[arg(long, default_value = "1")]
    pub parallel: usize,
    /// Inner checker worker count for each `ty check` subprocess.
    /// Omit to preserve baseline parity mode (`--workers 1 --continue-on-error`).
    /// Use 0 for adaptive/auto, 1 for explicit sequential, or N > 1 for explicit parallel.
    #[arg(long, value_name = "N")]
    pub checker_workers: Option<usize>,
    /// Update ty fields in baseline after run.
    #[arg(long)]
    pub update_baseline: bool,
    /// Root directory for tlaplus-examples/specifications.
    #[arg(long)]
    pub examples_dir: Option<PathBuf>,
    /// Exit with code 1 if any spec has a state count mismatch.
    /// FlakyTimeout and Skip do NOT trigger failure.
    #[arg(long)]
    pub fail_on_mismatch: bool,
    /// Exit with code 1 if any spec does not pass.
    /// FlakyTimeout and Skip do NOT trigger failure.
    #[arg(long)]
    pub fail_on_non_pass: bool,
    /// Write metrics JSON to a file. If no path given, writes to metrics/spec_coverage.json.
    /// Can combine with --output human.
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "metrics/spec_coverage.json")]
    pub output_metrics: Option<PathBuf>,
    /// Differential oracle harness mode.
    ///
    /// `off` (default): interpreter only. `compare`: run interpreter AND
    /// trust-codegen for each spec, record divergences to `metrics/oracle_parity.json`.
    /// `fail-closed`: like compare but also exit non-zero on any divergence.
    /// Can also be set via `TY_ORACLE={off|compare|fail-closed}`. The CLI
    /// flag takes precedence over the env var.
    #[arg(long, value_enum, value_name = "MODE")]
    pub oracle_mode: Option<DiagnoseOracleMode>,
    /// Output path for oracle parity report JSON. Defaults to
    /// `metrics/oracle_parity.json`. Only used when `--oracle-mode` is
    /// `compare` or `fail-closed`.
    #[arg(long, value_name = "PATH")]
    pub oracle_output: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TrustCgCoverageArgs {
    /// Optional single TLA+ source file. Omit to scan the diagnose baseline.
    pub file: Option<PathBuf>,
    /// Config file for single-spec mode. If omitted, looks for NEXT Next.
    #[arg(short, long, requires = "file")]
    pub config: Option<PathBuf>,
    /// Path to spec_baseline.json for baseline mode.
    #[arg(long, default_value = "tests/tlc_comparison/spec_baseline.json")]
    pub baseline: PathBuf,
    /// Root directory for tlaplus-examples/specifications in baseline mode.
    #[arg(long)]
    pub examples_dir: Option<PathBuf>,
    /// Restrict baseline mode to one or more spec names.
    #[arg(long)]
    pub spec: Vec<String>,
    /// Output format.
    #[arg(long, value_enum, default_value = "human")]
    pub output: TrustCgCoverageOutputFormat,
    /// Write JSON inventory to a file.
    #[arg(long, value_name = "PATH")]
    pub output_file: Option<PathBuf>,
    /// Write a Markdown summary table.
    #[arg(long, value_name = "PATH")]
    pub report: Option<PathBuf>,
}
