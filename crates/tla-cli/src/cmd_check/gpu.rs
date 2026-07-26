//! GPU engine dispatch: part of TY's default AUTO engine family.
//!
//! Default (no flags): if a CUDA device is present and the spec admits, a
//! bounded CPU probe runs first (cap [`GPU_AUTO_PROBE_STATE_CAP`] distinct
//! states, mirroring the trust-cg lazy-compile threshold). Small state spaces
//! complete inside the probe and never pay GPU setup; when the probe hits the
//! cap the space is large and the GPU engine takes over. `--gpu` forces the
//! GPU engine (skips the probe); `--no-gpu` disables it — both are testing
//! levers, per the all-powers-on-by-default contract.
//!
//! Fail-closed at every step: GPU unavailable, spec not admissible, CUDA
//! source not emittable, or any engine runtime failure → print the reason and
//! return `Ok(false)` so `cmd_check` falls through to the normal CPU engines
//! (verdict-neutral). Only a completed exhaustive search reports a verdict.
//!
//! Design of record: docs/perf/gpu-cuda-plan-2026-07-02.md.

use std::path::Path;

use anyhow::Result;
use tla_check::{
    CheckResult, CheckStats, Config, LimitType, ModelChecker, SearchCompleteness,
    SoundnessProvenance,
};
use tla_core::ast::Module;

use crate::check_report::{report_check_json, JsonReportCtx};
use crate::cli_schema::OutputFormat;

/// Distinct-state cap for the auto-mode CPU probe. Mirrors
/// `TY_TRUST_CG_LAZY_COMPILE_THRESHOLD`: under this, the CPU finishes fast and
/// GPU setup (~1s nvrtc + allocations) would be pure overhead.
const GPU_AUTO_PROBE_STATE_CAP: usize = 131_072;

/// Attempt the GPU engine. `Ok(true)` = verdict produced and printed;
/// `Ok(false)` = declined, caller falls through to the CPU engines.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_run_gpu_check(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    no_deadlock: bool,
    output_format: OutputFormat,
    force: bool,
    user_max_states: usize,
    // JSON reporting context, threaded from the caller so a GPU verdict
    // serializes through the SAME `report_check_json` path as the CPU
    // engine — byte-identical schema whichever engine produced the numbers.
    spec_file: &Path,
    config_path: &Path,
    workers: usize,
    soundness: &SoundnessProvenance,
    completeness: SearchCompleteness,
    max_depth: usize,
) -> Result<bool> {
    let info = match tla_gpu::probe() {
        Ok(info) => info,
        Err(e) => {
            // No CUDA on this host: stay silent in auto mode (this runs on
            // every default check), explain when the user forced --gpu.
            if force {
                eprintln!("[gpu] unavailable: {e}; falling back to CPU engine");
            }
            return Ok(false);
        }
    };
    // Human/Json/Jsonl all serialize a bare exhaustive verdict; only the
    // trace/tool streams (Itf needs a counterexample trace, TlcTool the
    // tool-protocol stream) genuinely can't be served by the GPU verdict.
    if matches!(output_format, OutputFormat::TlcTool | OutputFormat::Itf) {
        if force {
            eprintln!(
                "[gpu] declined: {output_format:?} output needs the CPU trace/tool stream; \
                 falling back to CPU engine"
            );
        }
        return Ok(false);
    }
    // Emit a completed-verdict result through the shared JSON serializer
    // (non-Human formats). Reused by the probe-success and GPU-success arms.
    let emit_json = |result: &CheckResult, elapsed: std::time::Duration| -> Result<()> {
        report_check_json(&JsonReportCtx {
            output_format,
            checker_modules,
            module,
            file: spec_file,
            config_path,
            workers,
            soundness,
            completeness,
            config,
            result,
            elapsed,
            strategy_info: Some("gpu-cuda-bfs"),
            max_states: user_max_states,
            max_depth,
        })
    };

    let gpu_debug = force || std::env::var("TY_GPU_DEBUG").is_ok_and(|v| v != "0");
    let mut checker = ModelChecker::new_with_extends(module, checker_modules, config);
    let program = match checker.try_prepare_gpu_program() {
        Ok(p) => p,
        Err(reason) => {
            // Silent in auto mode: this runs on every default check and most
            // specs legitimately decline. --gpu / TY_GPU_DEBUG=1 explain why.
            if gpu_debug {
                eprintln!("[gpu] declined: {reason}; falling back to CPU engine");
            }
            return Ok(false);
        }
    };
    if gpu_debug {
        eprintln!(
            "[gpu] admission: program prepared ({} actions, {} invariants, {} constraints); emitting CUDA...",
            program.actions.len(),
            program.invariants.len(),
            program.constraints.len(),
        );
    }
    if std::env::var("TY_GPU_DUMP_IR").is_ok() {
        for f in program.invariants.iter().chain(program.constraints.iter()) {
            eprintln!("=== IR (predicate) {} sym={} ===", f.name, f.symbol);
            for func in &f.module.functions {
                eprintln!("  fn {} blocks={}", func.name, func.blocks.len());
                for b in &func.blocks {
                    eprintln!("    bb{} params={:?}", b.id.index(), b.params);
                    for n in &b.body {
                        eprintln!("      results={:?} inst={:?}", n.results, n.inst);
                    }
                }
            }
        }
    }

    // Auto mode: bounded CPU probe. Completes → small space, report its
    // result directly (no GPU setup cost). Hits the auto cap → large space,
    // the GPU engine takes over. Any other outcome (violation, deadlock,
    // user-configured limit, error) → decline so the standard CPU path
    // re-runs and reports with full trace/output fidelity (bounded cost: the
    // probe stopped early on exactly those outcomes).
    if !force {
        let cap = if user_max_states > 0 {
            user_max_states.min(GPU_AUTO_PROBE_STATE_CAP)
        } else {
            GPU_AUTO_PROBE_STATE_CAP
        };
        let mut probe_config = config.clone();
        probe_config.check_deadlock = probe_config.check_deadlock && !no_deadlock;
        let probe_started = std::time::Instant::now();
        let mut probe = ModelChecker::new_with_extends(module, checker_modules, &probe_config);
        probe.set_max_states(cap);
        match probe.check() {
            CheckResult::Success(stats) => {
                let elapsed = probe_started.elapsed();
                if matches!(output_format, OutputFormat::Human) {
                    print_cpu_probe_summary(&stats, elapsed);
                } else {
                    emit_json(&CheckResult::Success(stats), elapsed)?;
                }
                return Ok(true);
            }
            CheckResult::LimitReached {
                limit_type: LimitType::States,
                stats,
            } if cap == GPU_AUTO_PROBE_STATE_CAP
                && (user_max_states == 0 || user_max_states > cap) =>
            {
                eprintln!(
                    "[gpu] CPU probe hit {} distinct states in {:.2}s (large state space); \
                     switching to the GPU engine",
                    stats.states_found,
                    probe_started.elapsed().as_secs_f64(),
                );
            }
            _ => {
                // Violation / deadlock / user limit / error: the standard CPU
                // path owns reporting for these.
                return Ok(false);
            }
        }
    }

    let actions: Vec<_> = program
        .actions
        .iter()
        .map(|f| (f.name.clone(), f.symbol.clone(), &f.module))
        .collect();
    let invariants: Vec<_> = program
        .invariants
        .iter()
        .map(|f| (f.name.clone(), f.symbol.clone(), &f.module))
        .collect();
    let constraints: Vec<_> = program
        .constraints
        .iter()
        .map(|f| (f.name.clone(), f.symbol.clone(), &f.module))
        .collect();
    let source = match tla_gpu::emit_program_with_constraints(&actions, &invariants, &constraints) {
        Ok(s) => s,
        Err(e) => {
            if gpu_debug {
                eprintln!("[gpu] declined: {e}; falling back to CPU engine");
            }
            return Ok(false);
        }
    };
    if let Ok(dump_path) = std::env::var("TY_GPU_DUMP_CUDA") {
        if let Err(e) = std::fs::write(&dump_path, &source.source) {
            eprintln!("[gpu] warning: TY_GPU_DUMP_CUDA write failed: {e}");
        } else {
            eprintln!("[gpu] emitted CUDA source dumped to {dump_path}");
        }
        // Companion init-rows dump for standalone differential harnesses.
        let mut rows = format!("{} {}\n", program.slots, source.action_count);
        for row in program.init_rows.chunks(program.slots) {
            for (i, v) in row.iter().enumerate() {
                if i > 0 {
                    rows.push(' ');
                }
                rows.push_str(&v.to_string());
            }
            rows.push('\n');
        }
        let _ = std::fs::write(format!("{dump_path}.init"), rows);
    }

    eprintln!(
        "[gpu] engine admitted: {} action kernels, {} invariants, {} slots/state ({} bytes) on {} (cc {}.{}, {} SMs)",
        source.action_count,
        invariants.len(),
        program.slots,
        program.slots * 8,
        info.device_name,
        info.cc_major,
        info.cc_minor,
        info.multiprocessors,
    );

    // Capture the initial-state count before `init_rows` is moved into the spec.
    let initial_state_count = program.init_rows.len() / program.slots.max(1);
    let spec = tla_gpu::GpuBfsSpec {
        slots: program.slots,
        action_count: source.action_count,
        actions_src: source.source,
        init_rows: program.init_rows,
        track_slot_stats: false,
    };

    // Grow-and-retry on fail-closed capacity bounds (each retry restarts the
    // search from scratch; the fingerprint table and arenas are reallocated).
    let mut engine_config = tla_gpu::GpuBfsConfig::default();
    if user_max_states > 0 {
        // Honor the user's exploration bound on-device: decline past it (the
        // CPU engines report the same bound) instead of exploring beyond.
        engine_config.max_distinct = user_max_states as u64;
    }
    // Reconstruct the init->violation counterexample path on-device (a violation
    // stops the search early, so the retained arena stays shallow) instead of
    // declining to the CPU engine to re-derive it.
    engine_config.trace_on_violation = true;

    // Never ask the device for a configuration that cannot fit: project the
    // footprint of every configuration (initial AND each retry rung) against
    // the CUDA budget and decline to the CPU engine instead of requesting the
    // impossible. Without this, the grow-and-retry ladder below escalated to
    // a single 96 GiB arena request on unified-memory hardware (2026-07-21),
    // where "GPU memory" is host RAM and the request consumed the machine.
    // Advisory only (`None` when the projection or budget is unavailable):
    // the allocation-time budget transaction in tla-gpu stays authoritative.
    let footprint_over_budget =
        |config: &tla_gpu::GpuBfsConfig, spec: &tla_gpu::GpuBfsSpec| -> Option<(u64, u64)> {
            let projected = tla_gpu::projected_device_bytes(spec, config).ok()?;
            let budget = tla_gpu::allocation_headroom_bytes().ok()?;
            (projected > budget).then_some((projected, budget))
        };
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    if let Some((projected, budget)) = footprint_over_budget(&engine_config, &spec) {
        eprintln!(
            "[gpu] projected device footprint {:.1} GiB exceeds the {:.1} GiB CUDA budget; \
             falling back to the CPU engine",
            gib(projected),
            gib(budget)
        );
        return Ok(false);
    }

    let mut attempts = 0;
    let outcome = loop {
        match tla_gpu::run_bfs(&spec, &engine_config) {
            Ok(outcome) => break outcome,
            Err(tla_gpu::GpuError::CapacityExceeded {
                what,
                needed,
                capacity,
            }) if what == "distinct-state cap" => {
                eprintln!(
                    "[gpu] distinct states exceed --max-states ({needed} > {capacity}); \
                     falling back to the CPU engine"
                );
                return Ok(false);
            }
            Err(tla_gpu::GpuError::CapacityExceeded {
                what,
                needed,
                capacity,
            }) if attempts < 3 => {
                attempts += 1;
                if what == "fingerprint table" {
                    engine_config.table_bits += 2;
                } else {
                    engine_config.frontier_cap_rows *= 4;
                }
                if let Some((projected, budget)) = footprint_over_budget(&engine_config, &spec) {
                    eprintln!(
                        "[gpu] {what} capacity exceeded ({needed} > {capacity}); a larger \
                         allocation would need {:.1} GiB against the {:.1} GiB CUDA budget; \
                         falling back to the CPU engine",
                        gib(projected),
                        gib(budget)
                    );
                    return Ok(false);
                }
                eprintln!(
                    "[gpu] {what} capacity exceeded ({needed} > {capacity}); retrying with larger allocation"
                );
            }
            Err(e) => {
                eprintln!("[gpu] engine failed: {e}; falling back to CPU engine");
                return Ok(false);
            }
        }
    };
    let outcome_distinct_states = usize::try_from(outcome.distinct_states)
        .map_err(|_| anyhow::anyhow!("GPU distinct-state count does not fit usize"))?;
    let outcome_raw_successors = usize::try_from(outcome.transitions)
        .map_err(|_| anyhow::anyhow!("GPU raw successor count does not fit usize"))?;

    if outcome.violation.is_some() {
        // The GPU engine reconstructed the init->bad counterexample path on
        // device (parent-pointer walk). Decode it with the same flat layout and
        // report it through the SAME reporter the CPU engine uses — identical
        // human/JSON output and exit behavior — instead of re-deriving it on the
        // CPU. Decline only if the trace could not be built (no flat layout).
        if let Some(trace_rows) = &outcome.violation_trace {
            if let Some((invariant, states)) = checker.gpu_violation_report(trace_rows) {
                let mut vstats = CheckStats::default();
                vstats.states_found = outcome_distinct_states;
                vstats.initial_states = initial_state_count;
                vstats.raw_initial_states_generated = program.raw_initial_states_generated;
                vstats.transitions = outcome_raw_successors;
                vstats.raw_successors_generated = outcome_raw_successors;
                vstats.max_depth = states.len().saturating_sub(1);
                vstats.engine_provenance = Some(serde_json::json!({
                    "tier": "gpu",
                    "device": info.device_name,
                    "search_wall_s": outcome.wall.as_secs_f64(),
                    "nvrtc_compile_ms": outcome.compile_wall.as_secs_f64() * 1e3,
                }));
                let result = CheckResult::InvariantViolation {
                    invariant,
                    trace: tla_check::Trace::from_states(states),
                    stats: vstats,
                };
                let elapsed = outcome.wall + outcome.compile_wall;
                eprintln!(
                    "[gpu] invariant violation — counterexample reconstructed on-device \
                     ({} states)",
                    trace_rows.len()
                );
                if matches!(output_format, OutputFormat::Human) {
                    crate::check_report::report_check_human(
                        result,
                        elapsed,
                        user_max_states,
                        max_depth,
                        crate::cli_schema::TraceFormat::default(),
                        false,
                        false,
                    )?;
                } else {
                    emit_json(&result, elapsed)?;
                }
                return Ok(true);
            }
        }
        eprintln!(
            "[gpu] invariant violation detected but the trace could not be \
             reconstructed; falling back to the CPU engine for the counterexample"
        );
        return Ok(false);
    }
    let effective_check_deadlock = program.check_deadlock && !no_deadlock;
    if effective_check_deadlock && outcome.deadlock_states > 0 {
        eprintln!(
            "[gpu] {} deadlocked state(s) detected; falling back to the CPU engine \
             for the standard deadlock report",
            outcome.deadlock_states
        );
        return Ok(false);
    }

    // Progress line to STDERR — never corrupts the STDOUT JSON.
    eprintln!(
        "[gpu] search wall {:.3}s (nvrtc compile {:.0} ms), {:.1}M distinct states/s, {:.1}M transitions/s, {} BFS levels",
        outcome.wall.as_secs_f64(),
        outcome.compile_wall.as_secs_f64() * 1e3,
        outcome.distinct_states as f64 / outcome.wall.as_secs_f64() / 1e6,
        outcome.transitions as f64 / outcome.wall.as_secs_f64() / 1e6,
        outcome.levels,
    );

    // Map the device outcome onto the SAME CheckStats/CheckResult the CPU
    // engine reports, so the serializer (human or JSON) is engine-agnostic.
    // CheckStats is #[non_exhaustive] — construct via Default + field sets.
    let mut gpu_stats = CheckStats::default();
    gpu_stats.states_found = outcome_distinct_states;
    gpu_stats.initial_states = initial_state_count;
    gpu_stats.raw_initial_states_generated = program.raw_initial_states_generated;
    gpu_stats.transitions = outcome_raw_successors;
    gpu_stats.raw_successors_generated = outcome_raw_successors;
    // `levels` = diameter + 1 (the seed level counts as level 0).
    gpu_stats.max_depth = (outcome.levels.saturating_sub(1)) as usize;
    gpu_stats.engine_provenance = Some(serde_json::json!({
        "tier": "gpu",
        "device": info.device_name,
        "search_wall_s": outcome.wall.as_secs_f64(),
        "nvrtc_compile_ms": outcome.compile_wall.as_secs_f64() * 1e3,
    }));
    let gpu_states_generated = gpu_stats.states_generated();
    let result = CheckResult::Success(gpu_stats);
    let elapsed = outcome.wall + outcome.compile_wall;

    if matches!(output_format, OutputFormat::Human) {
        println!("Model checking complete: No errors found (exhaustive).");
        println!();
        println!("Statistics:");
        println!("  States found: {}", outcome.distinct_states);
        println!("  Initial states: {}", initial_state_count);
        println!(
            "  Initial states generated: {}",
            program.raw_initial_states_generated
        );
        println!("  States generated: {}", gpu_states_generated);
        println!("  Transitions: {}", outcome.transitions);
        println!("  Time: {:.3}s", elapsed.as_secs_f64());
    } else {
        emit_json(&result, elapsed)?;
    }
    Ok(true)
}

/// Print the standard success summary for a completed auto-probe CPU run
/// (same lines the runner emits, so scripts keep parsing).
fn print_cpu_probe_summary(stats: &CheckStats, elapsed: std::time::Duration) {
    println!("Model checking complete: No errors found (exhaustive).");
    println!();
    println!("Statistics:");
    println!("  States found: {}", stats.states_found);
    println!("  Initial states: {}", stats.initial_states);
    println!(
        "  Initial states generated: {}",
        stats.raw_initial_states_generated
    );
    println!("  States generated: {}", stats.states_generated());
    println!("  Transitions: {}", stats.transitions);
    println!("  Max queue depth: {}", stats.max_queue_depth);
    println!("  Time: {:.3}s", elapsed.as_secs_f64());
}
