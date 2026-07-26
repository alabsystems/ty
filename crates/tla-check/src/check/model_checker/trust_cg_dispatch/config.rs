// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Environment-variable gates, diagnostic flags, and tuning thresholds for the
//! trust-codegen dispatch path.
//!
//! These are the runtime configuration readers (env-flag truth tables, lazy /
//! deferred compile thresholds, replay-artifact and dump filters) that the
//! dispatch, admission, and cache layers consult. They are pure config: no
//! codegen, dispatch-decision, or verdict effect beyond the documented gates.

pub(super) const TRUST_CG_NATIVE_CALLOUT_SELFTEST_ENV: &str = "TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST";
pub(super) const TRUST_CG_NATIVE_CALLOUT_SELFTEST_FAIL_CLOSED_ENV: &str =
    "TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST_FAIL_CLOSED";
pub(super) const TRUST_CG_DUMP_ACTION_BYTECODE_ENV: &str = "TY_TRUST_CG_DUMP_ACTION_BYTECODE";
pub(super) const TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV: &str =
    "TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS";
pub(super) const TRUST_CG_NATIVE_CALLOUT_BATCH_ENV: &str = "TY_TRUST_CG_NATIVE_CALLOUT_BATCH";
pub(super) const TRUST_CG_SETUP_TIMING_ENV: &str = "TY_TRUST_CG_SETUP_TIMING";
pub(super) const TRUST_CG_REPLAY_ARTIFACT_DIR_ENV: &str = "TY_TRUST_CG_REPLAY_ARTIFACT_DIR";
pub(super) const TRUST_CG_REPLAY_ARTIFACT_FILTER_ENV: &str = "TY_TRUST_CG_REPLAY_ARTIFACT_FILTER";
pub(super) const TRUST_CG_REPLAY_TY_GIT_COMMIT_ENV: &str = "TY_TRUST_CG_REPLAY_TY_GIT_COMMIT";
/// Observability-only (un-darkening STEP 1): per-action runtime native-vs-interp
/// wall-clock telemetry in the compiled BFS loop. Default OFF = byte-identical
/// behavior; the timing reads are gated behind this flag so a default run never
/// calls the clock per state.
pub(super) const TRUST_CG_RUNTIME_TELEMETRY_ENV: &str = "TY_TRUST_CG_TELEMETRY";
/// Observability-only (un-darkening STEP 1): dump, per action that fails to
/// admit native, the precise rejection reason (UnsupportedOpcode/which opcode,
/// Unknown-universe set op, NextStateLoop NotYetSupported, admission-declined).
/// Surfaces the per-action reasons already recorded during cache build plus the
/// batch artifact-admission rejection reasons. Pure diagnostic; no codegen,
/// dispatch-decision, or verdict change.
pub(super) const TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES_ENV: &str =
    "TY_TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES";
pub(super) const TRUST_CG_DISABLE_ARTIFACT_CACHE_ENV: &str = "TY_DISABLE_ARTIFACT_CACHE";
pub(super) const TRUST_CG_DISABLE_PROCESS_LOCAL_WARM_ARTIFACT_CACHE_ENV: &str =
    "TY_TRUST_CG_DISABLE_PROCESS_LOCAL_WARM_ARTIFACT_CACHE";

pub(super) fn trust_cg_env_flag_enabled(name: &str) -> bool {
    // Canonical truth-table lives in tla-backend (one source of truth shared with the
    // CLI's engine-selection layer). Behavior is identical to the prior inline `== "1"`.
    std::env::var(name).map_or(false, |value| tla_backend::env_flag_enabled(&value))
}

/// Whether an env var explicitly requests the interpreter instead of trust-cg.
///
/// Recognizes `0`, `false`, `off`, `no` (case-insensitive). Used to let a run
/// opt out of the default JIT path — the JIT is always compiled in, so this is
/// a runtime engine-selection override, not a way to remove the backend.
pub(super) fn trust_cg_env_flag_disabled(name: &str) -> bool {
    std::env::var(name).map_or(false, |value| tla_backend::env_flag_disabled(&value))
}

pub(in crate::check) fn trust_cg_setup_timing_enabled() -> bool {
    trust_cg_env_flag_enabled(TRUST_CG_SETUP_TIMING_ENV)
}

/// Env override for the native action/invariant/state-constraint callout compile
/// optimization level.
pub(super) const TRUST_CG_ACTION_OPT_LEVEL_ENV: &str = "TY_TRUST_CG_ACTION_OPT_LEVEL";

/// Optimization level used to compile native action / invariant / state-constraint
/// callouts.
///
/// Defaults to `O1`. The `TY_TRUST_CG_ACTION_OPT_LEVEL` env var overrides it with
/// one of `O0`/`O1`/`O2`/`O3` (case-insensitive) so the compile-time vs BFS-runtime
/// trade-off can be measured and tuned without a rebuild (e.g. `=O3` restores the
/// historic default). Every level is value-preserving — trust-cg's opt passes
/// (CSE/GVN already omitted, plus the O2+-only post-RA opt and pressure-aware
/// scheduling) only change generated-code quality and compile throughput, never the
/// computed successors — so the BFS result and verdict are byte-identical across
/// levels. An unrecognized value is logged and falls back to the `O1` default.
///
/// # Why O1 (was O3)
///
/// The former O3 default paid trust-cg's O2+-only post-RA optimization and
/// pressure-aware scheduling on every per-action codegen pass. For the
/// action-callout kernels those passes are a large, serial compile cost with **no
/// runtime benefit** — measured on `MCLamportMutex` (27 native actions, single-core
/// `taskset -c 9`, 724274/2496350 exact state graph):
///
/// | level | action compile | BFS wall | total | verdict/states |
/// |-------|---------------:|---------:|------:|----------------|
/// | O3    | ~18.4s         | ~10.1s   | ~28.5s| 724274, no error |
/// | O1    | ~3.6s          | ~5.7s    | ~9.4s | 724274, no error |
///
/// O1 is ~5x cheaper to compile AND (surprisingly) ~1.75x faster at runtime — the
/// O3 scheduler pessimizes this spec's hot next-state loop. Native coverage is
/// identical (27 compiled / 9 fallback / 3 invariants at both levels). On specs
/// whose native compile is already cheap (`btree` ~0ms, `MCBakery` ~1s) O1 ties O3
/// on wall and is byte-identical on states (64685 / 655200), so the default flip is
/// a strict win-or-tie on the measured corpus while flipping MCLamportMutex from a
/// 28.5s loss to a 9.4s win vs the 15.3s TLC single-thread reference.
pub(in crate::check) fn trust_cg_action_compile_opt_level() -> tla_trust_cg::OptLevel {
    match std::env::var(TRUST_CG_ACTION_OPT_LEVEL_ENV) {
        Ok(value) => match value.trim().to_ascii_uppercase().as_str() {
            "O0" => tla_trust_cg::OptLevel::O0,
            "" | "O1" => tla_trust_cg::OptLevel::O1,
            "O2" => tla_trust_cg::OptLevel::O2,
            "O3" => tla_trust_cg::OptLevel::O3,
            other => {
                eprintln!(
                    "[trust-cg] ignoring {TRUST_CG_ACTION_OPT_LEVEL_ENV}={other:?}; expected O0/O1/O2/O3"
                );
                tla_trust_cg::OptLevel::O1
            }
        },
        Err(_) => tla_trust_cg::OptLevel::O1,
    }
}

/// Whether per-action runtime native-vs-interp wall-clock telemetry is enabled
/// (un-darkening STEP 1). Observability-only: the timing reads in the compiled
/// BFS loop are gated behind this, so default runs pay zero clock overhead.
pub(in crate::check) fn trust_cg_runtime_telemetry_enabled() -> bool {
    trust_cg_env_flag_enabled(TRUST_CG_RUNTIME_TELEMETRY_ENV)
}

/// Whether the per-action native-admission failure dump is enabled (un-darkening
/// STEP 1). Observability-only: surfaces the rejection reasons already recorded
/// during cache build. No codegen, dispatch, or verdict effect.
pub(in crate::check) fn trust_cg_dump_native_admission_failures_enabled() -> bool {
    trust_cg_env_flag_enabled(TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES_ENV)
}

pub(super) const TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD_ENV: &str =
    "TY_TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD";

/// Default state-count trigger for the deferred native fused-level compile.
///
/// Setup skips the expensive fused parent-loop module compile (hundreds of
/// milliseconds of trust-codegen regalloc on one large generated function)
/// when the per-parent `CompiledBfsStep` can drive the compiled BFS loop, and
/// `run_compiled_bfs_loop` promotes to the fused level at the first level
/// boundary where the cumulative distinct-state count reaches this threshold.
/// Runs that finish below it never pay the compile. The value is a structural
/// work bound, not a benchmark constant: it only has to be (a) large enough
/// that sub-second runs stay under it and (b) small enough that the step-path
/// warmup before promotion is a negligible slice of any run big enough to
/// need the fused loop.
pub(super) const TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD_DEFAULT: usize = 32_768;

/// State-count threshold at which a deferred fused-level compile is promoted.
///
/// `TY_TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD` overrides the default; `0`
/// disables deferral entirely (setup compiles the fused level eagerly, the
/// pre-deferral behavior).
pub(in crate::check) fn trust_cg_fused_level_defer_threshold() -> usize {
    static THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var(TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD_DEFAULT)
    })
}

pub(super) const FUSED_INVARIANT_MIN_STATES_ENV: &str = "TY_FUSED_INVARIANT_MIN_STATES";

/// Default state-count floor below which the native-fused level is built in
/// *action-only* mode (invariants checked by the interpreter) instead of
/// fusing the invariant predicates into the generated parent loop.
///
/// Fusing invariants into the native fused parent loop makes a much larger
/// generated function: on SimpleRegular (`TY_SUBSET_POWERSET_SHAPE=1`, 277726
/// states) the *invariant-checking* level build costs ~6.5s of trust-codegen
/// regalloc — vs ~60ms for the action-only level and ~3ms for the invariant
/// bytecode compile itself — while the fused BFS execution only speeds up by
/// ~2.1s. Below some state count that fixed fusion cost dominates and the run
/// regresses (SimpleRegular: 4.0s -> 7.7s). Above it, the per-state native
/// invariant-check saving amortizes the one-time fusion cost.
///
/// The action-only level with interpreter invariant checks is the *default*
/// path for non-native and invariant-uncompilable specs today (e.g. baseline
/// SimpleRegular without the subset shape runs `native_fused_mode=action_only`,
/// `regular_invariants_checked=false`, and the compiled BFS loop's per-successor
/// `check_successor_invariant` Rust check), so the size-gate only chooses WHICH
/// path checks invariants — never the verdict or the state count.
///
/// This is a structural work bound, not a benchmark constant: it only has to be
/// (a) above the state count where the fixed fusion cost stops amortizing
/// (SimpleRegular regressed at 277726) and (b) below the state count of the
/// runs that currently win via native invariant fusion. The named native-fused
/// invariant/flat-safety wins (EWD998Small 1.5M, MCLamportMutex 724K) are
/// *state-constrained* and therefore exempt from this gate entirely (constrained
/// runs must fuse eagerly for native constraint pruning), so the default only
/// needs to sit above SimpleRegular's 277726 and below the smaller of those
/// (724K) to honor the "wins stay native" intent for any unconstrained
/// equivalent. 500_000 satisfies both.
pub(super) const FUSED_INVARIANT_MIN_STATES_DEFAULT: usize = 500_000;

/// State-count floor for fusing invariants into the native fused parent loop.
///
/// When the best available state estimate at the level-build decision point is
/// below this, the native fused level is built action-only and invariants are
/// checked by the interpreter; a runtime level-boundary promotion re-fuses the
/// invariants once the cumulative distinct-state count crosses this floor.
///
/// `TY_FUSED_INVARIANT_MIN_STATES` overrides the default; `0` disables the gate
/// (invariants are always fused when native, the pre-gate behavior).
pub(in crate::check) fn trust_cg_fused_invariant_min_states() -> usize {
    static THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var(FUSED_INVARIANT_MIN_STATES_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(FUSED_INVARIANT_MIN_STATES_DEFAULT)
    })
}

pub(super) const TRUST_CG_LAZY_COMPILE_THRESHOLD_ENV: &str = "TY_TRUST_CG_LAZY_COMPILE_THRESHOLD";

/// Default distinct-state trigger for the AUTO-mode lazy trust-cg compile.
///
/// In AUTO engine-selection mode (`ty check` with no `--backend` flag) the
/// trust-codegen native action-callout cache is no longer built eagerly during
/// setup. Building it costs ~0.5-0.6s of JIT compile time that only pays off on
/// large state spaces; on small/medium specs the interpreter finishes the whole
/// run in a fraction of that. So setup DEFERS the compile, the interpreter Rust
/// BFS loop runs, and the per-parent BFS step promotes to the native per-action
/// callout cache once the distinct-state count (the seen-set length) reaches
/// this threshold. Runs that finish below it never pay the compile (the win).
///
/// This is a structural work bound, not a benchmark constant: it only has to be
/// (a) large enough that interpreter runs which beat single-thread TLC stay under
/// it and (b) small enough that the interpreter warmup before promotion is a
/// negligible slice of any run big enough to amortize the native compile.
///
/// Empirically (2026-06-14, single-thread cold): the interpreter sustains
/// ~65K distinct states/s and the native action-callout path is frequently no
/// faster per state (e.g. ACP_SB_TLC at 54,944 states: interpreter 0.84s beats
/// the compile-then-native 0.96s and TLC's 1.04s). The ~0.5s compile therefore
/// only amortizes for genuinely large state spaces, so the trigger sits well
/// above the medium-spec band — at 131,072 distinct states — keeping the whole
/// small/medium corpus on the interpreter (which beats TLC outright on those)
/// and compiling only the large runs where native execution dominates the
/// warmup+compile cost. Override with `TY_TRUST_CG_LAZY_COMPILE_THRESHOLD`
/// (`0` disables lazy and restores eager AUTO-mode compilation).
pub(super) const TRUST_CG_LAZY_COMPILE_THRESHOLD_DEFAULT: u64 = 131_072;

/// Distinct-state threshold at which a deferred AUTO-mode trust-cg compile fires,
/// or `None` when lazy deferral is disabled (always eager).
///
/// `TY_TRUST_CG_LAZY_COMPILE_THRESHOLD` overrides the default. A value of `0`
/// disables lazy deferral entirely (setup compiles the cache eagerly, the
/// pre-deferral behavior). An absent or unparseable value uses the default.
///
/// Only consulted in AUTO mode; forced `--backend trust-cg` never defers.
pub(in crate::check) fn trust_cg_lazy_compile_threshold() -> Option<u64> {
    static THRESHOLD: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        trust_cg_lazy_compile_threshold_from_env(
            std::env::var(TRUST_CG_LAZY_COMPILE_THRESHOLD_ENV)
                .ok()
                .as_deref(),
        )
    })
}

/// Pure parsing core of [`trust_cg_lazy_compile_threshold`], split out so the
/// env-to-threshold mapping is unit-testable without the process-global
/// `OnceLock` memoization in the public accessor.
///
/// - `None` raw (env absent) or unparseable => the default.
/// - `0` => `None` (lazy disabled / always eager).
/// - any other `n` => `Some(n)`.
pub(super) fn trust_cg_lazy_compile_threshold_from_env(raw: Option<&str>) -> Option<u64> {
    let value = raw
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(TRUST_CG_LAZY_COMPILE_THRESHOLD_DEFAULT);
    if value == 0 {
        None
    } else {
        Some(value)
    }
}

pub(super) const TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_ENV: &str =
    "TY_TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD";

/// Default *work* (transition-count) trigger for the AUTO-mode lazy trust-cg
/// compile, used as an OR-condition beside the distinct-state trigger.
///
/// The distinct-state gate ([`TRUST_CG_LAZY_COMPILE_THRESHOLD_DEFAULT`]) is a
/// stand-in for "invocation count" that only holds when cost-per-transition is
/// uniform. Specs with few distinct states but expensive transitions (e.g. the
/// Disruptor liveness specs) never cross it, so the JIT never engages even when
/// the cumulative work would amortize the compile many times over. This work
/// bound fires the same lazy compile once `stats.transitions` (a live work
/// counter, incremented per generated successor) crosses it, regardless of how
/// small the reachable graph is — the JVM-style "hotness" instrument the
/// state-count gate lacks.
///
/// Defaults to `u64::MAX` so the OR-condition is a no-op: this ships *dark*
/// (current behavior is exactly preserved) until the bound is tuned and the
/// default flipped. Override with `TY_TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD`.
pub(super) const TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_DEFAULT: u64 = u64::MAX;

/// Work (transition-count) threshold at which a deferred AUTO-mode trust-cg
/// compile fires, as an OR-condition beside [`trust_cg_lazy_compile_threshold`].
///
/// `TY_TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD` overrides the default. An absent or
/// unparseable value uses the default ([`u64::MAX`] — the OR-condition never
/// fires, so this ships dark / behavior-preserving). A value of `0` is treated
/// as "fire immediately on any accumulated work" (it is parsed as-is and any
/// nonzero transition count clears the `>=` test).
///
/// Only consulted in AUTO mode; forced `--backend trust-cg` never defers.
pub(in crate::check) fn trust_cg_lazy_compile_work_threshold() -> u64 {
    static WORK_THRESHOLD: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *WORK_THRESHOLD.get_or_init(|| {
        trust_cg_lazy_compile_work_threshold_from_env(
            std::env::var(TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_ENV)
                .ok()
                .as_deref(),
        )
    })
}

/// Pure parsing core of [`trust_cg_lazy_compile_work_threshold`], split out for
/// unit-testing without the process-global `OnceLock` memoization.
///
/// - `None` raw (env absent) or unparseable => the default (`u64::MAX`).
/// - any parseable `n` (including `0`) => `n`.
pub(super) fn trust_cg_lazy_compile_work_threshold_from_env(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_DEFAULT)
}

/// Pure OR-gate decision for the AUTO-mode lazy trust-cg compile trigger.
///
/// Fires (returns `true`) once *either* the distinct-state count crosses the
/// state threshold *or* the accumulated transition (work) count crosses the
/// work threshold. The work arm is what lets the JIT engage on small-state /
/// expensive-transition specs the state-count arm never reaches; with the
/// default `work_threshold == u64::MAX` it never contributes, so the gate
/// reduces to the original `states >= state_threshold` behavior (ships dark).
///
/// Split out as a pure function so the OR-gate is unit-testable without
/// constructing a `ModelChecker`.
pub(in crate::check) fn trust_cg_lazy_compile_gate_fires(
    states: u64,
    transitions: u64,
    state_threshold: u64,
    work_threshold: u64,
) -> bool {
    states >= state_threshold || transitions >= work_threshold
}

// Stage 3 of the unified-backend migration: the AUTO-selector structural veto moved
// from a process-global static to a per-instance `ModelChecker::trust_cg_structural_veto`
// field (see `ModelChecker::{trust_cg_structurally_vetoed, set_trust_cg_structural_veto}`
// in setup_build.rs). `is_enabled` / `should_use_trust_cg` now take the veto as a `bool`
// parameter, so daemon/library/server reuse no longer couples runs through global state.

pub(super) fn trust_cg_elapsed_ms(start: std::time::Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

pub(super) fn trust_cg_artifact_cache_disabled_by_env() -> bool {
    std::env::var_os(TRUST_CG_DISABLE_ARTIFACT_CACHE_ENV).is_some()
}

/// Process-local warm reuse is intentionally separate from the persistent
/// artifact-cache gate. The launch gate disables reusable artifacts across
/// runs, but a single cold checker process may still avoid compiling the same
/// semantically guarded batch twice.
pub(super) fn trust_cg_process_local_warm_artifact_cache_enabled() -> bool {
    !trust_cg_env_flag_enabled(TRUST_CG_DISABLE_PROCESS_LOCAL_WARM_ARTIFACT_CACHE_ENV)
}

pub(super) fn trust_cg_native_callout_selftest_enabled() -> bool {
    // BATTERIES-ON by default: validate the native callouts against the interpreter on
    // the first native parent (bounded) for every native run, unless explicitly disabled
    // with TY_TRUST_CG_NATIVE_CALLOUT_SELFTEST=0/off/false. The DEFAULT mode is warn
    // (surface a divergence, do not change the verdict — see
    // `trust_cg_native_callout_selftest_fail_closed`); strict/fail-closed remains opt-in.
    // "Compiles in Trust" should be checked, not assumed.
    if trust_cg_env_flag_enabled(TRUST_CG_NATIVE_CALLOUT_SELFTEST_FAIL_CLOSED_ENV) {
        return true;
    }
    match std::env::var(TRUST_CG_NATIVE_CALLOUT_SELFTEST_ENV) {
        Ok(value) => {
            let value = value.trim();
            !(value == "0"
                || value.is_empty()
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("false"))
        }
        Err(_) => true,
    }
}

pub(super) fn trust_cg_native_callout_selftest_fail_closed() -> bool {
    trust_cg_env_flag_enabled(TRUST_CG_NATIVE_CALLOUT_SELFTEST_FAIL_CLOSED_ENV)
        || std::env::var(TRUST_CG_NATIVE_CALLOUT_SELFTEST_ENV).map_or(false, |value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("fail_closed")
                || value.eq_ignore_ascii_case("fail-closed")
                || value.eq_ignore_ascii_case("strict")
        })
}

pub(super) fn trust_cg_native_callout_batch_enabled() -> bool {
    std::env::var(TRUST_CG_NATIVE_CALLOUT_BATCH_ENV)
        .map(|value| value.trim() != "0")
        .unwrap_or(true)
}

pub(super) fn trust_cg_native_callout_compile_jobs(task_count: usize) -> usize {
    if task_count <= 1 {
        return task_count;
    }

    match std::env::var(TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV) {
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(0 | 1) => 1,
            Ok(jobs) => jobs.min(task_count),
            Err(_) => {
                eprintln!(
                    "[trust-cg] ignoring {TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV}={value:?}; expected positive integer",
                );
                1
            }
        },
        // Native compilation is part of the measured TY process. Keep its
        // production default single-job so `--workers 1` cannot silently turn
        // into a many-core compile advantage over single-worker TLC.
        Err(_) => 1,
    }
}

pub(super) fn trust_cg_dump_filter_matches(env_name: &str, name: &str) -> bool {
    std::env::var(env_name).is_ok_and(|value| {
        let value = value.trim();
        value == "1"
            || value.eq_ignore_ascii_case("all")
            || name.contains(value)
            || value.split(',').any(|part| {
                let part = part.trim();
                !part.is_empty() && name.contains(part)
            })
    })
}

pub(super) fn trust_cg_replay_artifact_dir() -> Option<std::path::PathBuf> {
    let value = std::env::var_os(TRUST_CG_REPLAY_ARTIFACT_DIR_ENV)?;
    if value.to_string_lossy().trim().is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(value))
}

pub(super) fn trust_cg_replay_filter_allows(symbol_name: &str, name: &str) -> bool {
    match std::env::var(TRUST_CG_REPLAY_ARTIFACT_FILTER_ENV) {
        Ok(filter) => {
            let filter = filter.trim();
            !filter.is_empty()
                && (filter == "1"
                    || filter.eq_ignore_ascii_case("all")
                    || symbol_name.contains(filter)
                    || name.contains(filter)
                    || filter.split(',').any(|part| {
                        let part = part.trim();
                        !part.is_empty() && (symbol_name.contains(part) || name.contains(part))
                    }))
        }
        Err(_) => true,
    }
}

pub(super) fn trust_cg_replay_ty_git_commit() -> String {
    std::env::var(TRUST_CG_REPLAY_TY_GIT_COMMIT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| option_env!("TY_GIT_COMMIT").map(ToOwned::to_owned))
        .unwrap_or_else(|| "unknown".to_string())
}

pub(super) fn trust_cg_replay_artifact_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len().min(96));
    for ch in value.chars().take(96) {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unnamed".to_string()
    } else {
        out
    }
}
