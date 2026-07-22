// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::soundness::print_provenance_human;
use super::*;

#[cfg(test)]
mod tests;

/// Context for human-readable model checking header output.
pub(super) struct CheckHeaderCtx<'a> {
    pub(super) file: &'a Path,
    pub(super) config_path: &'a Path,
    pub(super) config: &'a tla_check::Config,
    pub(super) workers: usize,
    pub(super) continue_on_error: bool,
    pub(super) strict_vacuity: bool,
    pub(super) max_states: usize,
    pub(super) max_depth: usize,
    pub(super) memory_limit: usize,
    pub(super) disk_limit: usize,
    pub(super) store_states: bool,
    pub(super) no_trace: bool,
    pub(super) skip_liveness_for_benchmark: bool,
    pub(super) soundness: &'a SoundnessProvenance,
    pub(super) completeness: SearchCompleteness,
}

fn no_trace_status_line() -> &'static str {
    "Requested --no-trace: trace-file reconstruction is disabled; safety counterexample traces may be unavailable."
}

fn auto_sequential_reason(
    config: &tla_check::Config,
    workers: usize,
    _continue_on_error: bool,
    strict_vacuity: bool,
) -> Option<&'static str> {
    if workers != 0 {
        return None;
    }
    if strict_vacuity {
        return Some("strict-vacuity exhaustive action evidence");
    }
    // Fixes #4021: Force sequential mode in auto (workers=0) whenever invariants
    // are present, regardless of --continue-on-error. The parallel BFS enumerator
    // can produce ghost states for specs with state-dependent CHOOSE (e.g., btree's
    // ChooseFreeNode), causing false invariant violations and deadlock detection.
    // Previously, this only triggered when !continue_on_error, so `ty diagnose`
    // (which passes --continue-on-error) would run btree in parallel and fail.
    // TLC always checks invariants sequentially; this matches that behavior.
    if !config.invariants.is_empty() {
        return Some("invariant-stop TLC parity");
    }
    None
}

/// Print the human-readable header for model checking.
pub(super) fn print_check_header(ctx: &CheckHeaderCtx<'_>) {
    println!("Model checking: {}", ctx.file.display());
    println!("Config: {}", ctx.config_path.display());
    if let Some(ref spec) = ctx.config.specification {
        println!(
            "SPECIFICATION: {} (resolved to INIT: {}, NEXT: {})",
            spec,
            ctx.config.init.as_deref().unwrap_or("?"),
            ctx.config.next.as_deref().unwrap_or("?")
        );
    } else {
        if let Some(ref init) = ctx.config.init {
            println!("INIT: {}", init);
        }
        if let Some(ref next) = ctx.config.next {
            println!("NEXT: {}", next);
        }
    }
    if !ctx.config.invariants.is_empty() {
        println!("INVARIANTS: {}", ctx.config.invariants.join(", "));
    }
    if !ctx.config.trace_invariants.is_empty() {
        println!(
            "TRACE INVARIANTS: {}",
            ctx.config.trace_invariants.join(", ")
        );
    }
    if !ctx.config.properties.is_empty() {
        println!("PROPERTIES: {}", ctx.config.properties.join(", "));
    }
    if ctx.config.liveness_execution.uses_on_the_fly() && !ctx.config.properties.is_empty() {
        println!("Liveness: on-the-fly");
    }
    if let Some(reason) = auto_sequential_reason(
        ctx.config,
        ctx.workers,
        ctx.continue_on_error,
        ctx.strict_vacuity,
    ) {
        println!("Mode: sequential (auto: {reason})");
    } else if ctx.workers == 0 {
        println!("Mode: auto (adaptive strategy selection)");
    } else if ctx.workers == 1 {
        println!("Mode: sequential (1 worker)");
    } else {
        println!("Mode: parallel ({} workers)", ctx.workers);
    }
    println!();

    if cfg!(debug_assertions) {
        println!(
            "Note: running an unoptimized debug build; for performance runs use a release build (e.g., `cargo run --release --bin ty -- check ...`)."
        );
        println!();
    }
    if ctx.skip_liveness_for_benchmark && !ctx.config.properties.is_empty() {
        println!("Note: `TY_SKIP_LIVENESS=1` set; PROPERTY/liveness checking will be skipped.");
        println!();
    }
    if ctx.max_states > 0 {
        println!("Max states: {}", ctx.max_states);
    }
    if ctx.max_depth > 0 {
        println!("Max depth: {}", ctx.max_depth);
    }
    if ctx.memory_limit > 0 {
        println!("Memory limit: {} MB", ctx.memory_limit);
    }
    if ctx.disk_limit > 0 {
        println!("Disk limit: {} MB", ctx.disk_limit);
    }
    // The former SYMMETRY+liveness "auto-enabled for PROPERTY/liveness
    // checking" store-states line is gone: the checker now disables declared
    // SYMMETRY for genuine liveness runs instead of upgrading storage.
    if ctx.store_states {
        println!("Store-states mode: full states in memory (42x more memory)");
    } else if ctx.no_trace {
        println!("{}", no_trace_status_line());
    }
    println!();
    print_provenance_human(ctx.soundness, ctx.completeness);
    println!();
}

#[cfg(feature = "ay")]
fn print_structured_symbolic_result(
    output_format: OutputFormat,
    value: &serde_json::Value,
) -> Result<()> {
    let rendered = render_structured_json_value(output_format, value)?;
    println!("{rendered}");
    Ok(())
}

fn render_structured_json_value(
    output_format: OutputFormat,
    value: &serde_json::Value,
) -> Result<String> {
    let rendered = if matches!(output_format, OutputFormat::Json) {
        serde_json::to_string_pretty(value)?
    } else {
        serde_json::to_string(value)?
    };
    Ok(rendered)
}

#[cfg(feature = "ay")]
fn print_bmc_trace_human(trace: &[tla_check::BmcState]) {
    eprintln!("Counterexample trace ({} states):", trace.len());
    for state in trace {
        eprintln!("  State {}:", state.step);
        let mut assignments: Vec<_> = state.assignments.iter().collect();
        assignments.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
        for (var, value) in assignments {
            match value {
                tla_check::BmcValue::Bool(v) => eprintln!("    {} = {}", var, v),
                tla_check::BmcValue::Int(v) => eprintln!("    {} = {}", var, v),
                tla_check::BmcValue::BigInt(v) => eprintln!("    {} = {}", var, v),
                tla_check::BmcValue::String(v) => eprintln!("    {} = \"{}\"", var, v),
                tla_check::BmcValue::Set(members) => {
                    eprintln!(
                        "    {} = {{{}}} ({} elements)",
                        var,
                        members.len(),
                        members.len()
                    );
                }
                tla_check::BmcValue::Sequence(elems) => {
                    eprintln!("    {} = <<{}>>", var, render_bmc_elems(elems));
                }
                tla_check::BmcValue::Function(entries) => {
                    eprintln!("    {} = [func] ({} entries)", var, entries.len());
                }
                tla_check::BmcValue::Record(fields) => {
                    let field_names: Vec<&str> =
                        fields.iter().map(|(name, _)| name.as_str()).collect();
                    eprintln!("    {} = [{}]", var, field_names.join(", "));
                }
                tla_check::BmcValue::Tuple(elems) => {
                    eprintln!("    {} = <<{}>>", var, render_bmc_elems(elems));
                }
            }
        }
    }
}

/// Render the scalar elements of a sequence/tuple `BmcValue` for a trace line.
#[cfg(feature = "ay")]
fn render_bmc_elems(elems: &[tla_check::BmcValue]) -> String {
    elems
        .iter()
        .map(|e| match e {
            tla_check::BmcValue::Bool(v) => v.to_string(),
            tla_check::BmcValue::Int(v) => v.to_string(),
            tla_check::BmcValue::BigInt(v) => v.to_string(),
            tla_check::BmcValue::String(v) => format!("\"{v}\""),
            _ => "...".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Run BMC symbolic bug finding (extracted from cmd_check).
#[cfg(feature = "ay")]
pub(super) fn run_bmc_mode(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    max_depth: usize,
    incremental: bool,
    output_format: OutputFormat,
) -> Result<()> {
    use tla_check::{check_bmc, BmcConfig, BmcResult};

    if matches!(output_format, OutputFormat::TlcTool) {
        bail!("BMC output is not supported in --output tlc-tool mode");
    }

    if matches!(output_format, OutputFormat::Human) {
        println!("BMC mode: symbolic bounded model checking via ay");
        println!("Depth bound: {}", max_depth);
        if incremental {
            println!("Incremental solving: enabled (reusing solver across depths)");
        }
        println!();
    }

    // Validate constant binding up front (fast, no solver) so a config error still
    // bails with its original message before the timed solve.
    {
        let mut probe_ctx = tla_check::EvalCtx::new();
        probe_ctx.load_module(module);
        for m in checker_modules {
            probe_ctx.load_module(m);
        }
        if let Err(e) = tla_check::bind_constants_from_config(&mut probe_ctx, config) {
            bail!("Failed to bind constants: {}", e);
        }
    }

    let bmc_config = BmcConfig {
        max_depth,
        incremental,
        ..BmcConfig::default()
    };

    // Run the solve under a hard wall-clock watchdog. `BmcConfig.solve_timeout` is
    // handed to the ay solver, but ay's LIA `solve_lia_eager_split_loop` does not
    // check that deadline mid-search, so a non-converging eager case-split can run
    // unbounded (observed on `flag' = (x' >= 3)`-style boolean-from-comparison
    // encodings) — a silent HANG, the worst failure for a verification tool. A
    // detached worker raced against the budget converts that into a bounded,
    // reported `Unknown` (exit non-zero), honoring the "never hang" contract. The
    // default fused engine does not take this path. Raise TY_BMC_TIMEOUT_SECS to
    // extend the budget; the abandoned worker is reaped at process exit.
    // Total budget = per-solve `solve_timeout` × (depths + 1). Generous by
    // construction: ay honors `solve_timeout` *between* operations for normal
    // solves, so a legitimate incremental-deepening run over `max_depth` depths
    // always finishes inside this bound and is never cut short — only a single
    // non-terminating solve (which ignores the deadline) hits it.
    let per_solve = bmc_config
        .solve_timeout
        .unwrap_or(std::time::Duration::from_secs(300));
    let watchdog = per_solve.saturating_mul((max_depth as u32).saturating_add(1));
    let module_owned = module.clone();
    let config_owned = config.clone();
    let checker_owned: Vec<tla_core::ast::Module> =
        checker_modules.iter().map(|m| (*m).clone()).collect();
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut ctx = tla_check::EvalCtx::new();
        ctx.load_module(&module_owned);
        let checker_refs: Vec<&tla_core::ast::Module> = checker_owned.iter().collect();
        for m in &checker_refs {
            ctx.load_module(m);
        }
        // EXTENDS-inherited VARIABLES (MC-wrapper specs) must be registered so
        // BMC sees the real state vars instead of bailing on "No state variables".
        tla_check::register_state_vars_for_symbolic(&mut ctx, &module_owned, &checker_refs);
        // Known-ok: the probe above already validated binding.
        let _ = tla_check::bind_constants_from_config(&mut ctx, &config_owned);
        let _ = tx.send(check_bmc(&module_owned, &config_owned, &ctx, bmc_config));
    });

    let start = Instant::now();
    let bmc_result = match rx.recv_timeout(watchdog) {
        Ok(result) => result,
        Err(_) => Ok(BmcResult::Unknown {
            depth: max_depth,
            reason: format!(
                "standalone BMC exceeded its overall {}s solve budget without a verdict \
                 (a non-converging SMT solve in ay's LIA theory; raise TY_BMC_TIMEOUT_SECS \
                 to extend). The default fused engine is unaffected.",
                watchdog.as_secs()
            ),
        }),
    };
    let elapsed = start.elapsed();

    match bmc_result {
        Ok(BmcResult::BoundReached { max_depth }) => {
            if matches!(output_format, OutputFormat::Human) {
                println!("BMC: NO BUG FOUND up to depth {}.", max_depth);
                println!();
                println!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let value = serde_json::json!({
                    "result": "no_bug_found",
                    "max_depth": max_depth,
                    "time_secs": elapsed.as_secs_f64(),
                });
                print_structured_symbolic_result(output_format, &value)?;
            }
            Ok(())
        }
        Ok(BmcResult::Violation { depth, trace }) => {
            // SOUNDNESS NET: replay the BMC counterexample through the trusted
            // BFS interpreter. The symbolic translation (FuncSet enumeration,
            // sequence-builder reduction, string interning, ...) is heuristic;
            // the interpreter is the oracle. We only report a violation the
            // interpreter confirms — a disagreement signals a translation bug
            // and is surfaced as an error, never a (possibly wrong) verdict.
            let xval = tla_check::cross_validate_bmc_trace(module, config, &trace);
            if !xval.engine_agrees {
                if matches!(output_format, OutputFormat::Human) {
                    eprintln!(
                        "BMC: reported a counterexample at depth {depth}, but the trusted \
                         interpreter did NOT confirm it — discarding (cross-validation: {}).",
                        xval.detail
                    );
                    eprintln!();
                    eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                }
                bail!(
                    "BMC counterexample failed interpreter cross-validation: {}",
                    xval.detail
                );
            }

            if matches!(output_format, OutputFormat::Human) {
                eprintln!("BMC: VIOLATION - Counterexample found at depth {}.", depth);
                eprintln!("  (cross-validated: {})", xval.detail);
                eprintln!();
                print_bmc_trace_human(&trace);
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!("BMC found counterexample at depth {}", depth);
            }

            let value = serde_json::json!({
                "result": "violation",
                "depth": depth,
                "trace_length": trace.len(),
                "cross_validated": true,
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Ok(BmcResult::Deadlock { depth, trace }) => {
            // A reachable deadlock state (no Next successor) — a property failure,
            // like explicit-BFS Deadlock. Report and exit nonzero, mirroring the
            // Violation arm.
            if matches!(output_format, OutputFormat::Human) {
                eprintln!(
                    "BMC: DEADLOCK - Reachable deadlock state found at depth {}.",
                    depth
                );
                eprintln!();
                print_bmc_trace_human(&trace);
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!("BMC found reachable deadlock at depth {}", depth);
            }

            let value = serde_json::json!({
                "result": "deadlock",
                "depth": depth,
                "trace_length": trace.len(),
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Ok(BmcResult::Unknown { depth, reason }) => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("BMC: UNKNOWN - Could not determine safety.");
                eprintln!("Depth: {}", depth);
                eprintln!("Reason: {}", reason);
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!("BMC result inconclusive at depth {}: {}", depth, reason);
            }

            let value = serde_json::json!({
                "result": "unknown",
                "depth": depth,
                "reason": reason,
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("BMC error: {}", e);
            bail!("BMC failed: {}", e);
        }
    }
}

/// Run PDR/IC3 symbolic safety checking (extracted from cmd_check).
#[cfg(feature = "ay")]
pub(super) fn run_pdr_mode(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    output_format: OutputFormat,
) -> Result<()> {
    use tla_check::{check_pdr, PdrResult};

    if matches!(output_format, OutputFormat::TlcTool) {
        bail!("PDR output is not supported in --output tlc-tool mode");
    }

    if matches!(output_format, OutputFormat::Human) {
        println!("PDR mode: symbolic safety checking via CHC/IC3");
        println!();
    }

    // Set up evaluation context
    let mut ctx = tla_check::EvalCtx::new();
    ctx.load_module(module);
    for m in checker_modules {
        ctx.load_module(m);
    }
    // EXTENDS-inherited VARIABLES (MC-wrapper specs) must be registered so PDR
    // sees the real state vars instead of bailing on "No state variables".
    tla_check::register_state_vars_for_symbolic(&mut ctx, module, checker_modules);

    // Bind constants from config
    if let Err(e) = tla_check::bind_constants_from_config(&mut ctx, config) {
        bail!("Failed to bind constants: {}", e);
    }

    // Run PDR
    let start = Instant::now();
    let pdr_result = check_pdr(module, config, &ctx);
    let elapsed = start.elapsed();

    match pdr_result {
        Ok(PdrResult::Safe { invariant }) => {
            if matches!(output_format, OutputFormat::Human) {
                println!("PDR: SAFE - All invariants hold.");
                println!();
                println!("Synthesized invariant:");
                println!("  {}", invariant);
                println!();
                println!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let value = serde_json::json!({
                    "result": "safe",
                    "invariant": invariant,
                    "time_secs": elapsed.as_secs_f64(),
                });
                print_structured_symbolic_result(output_format, &value)?;
            }
            Ok(())
        }
        Ok(PdrResult::Unsafe { trace }) => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("PDR: UNSAFE - Counterexample found!");
                eprintln!();
                eprintln!("Counterexample trace ({} states):", trace.len());
                for (i, state) in trace.iter().enumerate() {
                    eprintln!("  State {}:", i);
                    for (var, val) in &state.assignments {
                        eprintln!("    {} = {}", var, val);
                    }
                }
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!("PDR found counterexample");
            }

            let value = serde_json::json!({
                "result": "unsafe",
                "trace_length": trace.len(),
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Ok(PdrResult::Unknown { reason }) => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("PDR: UNKNOWN - Could not determine safety.");
                eprintln!("Reason: {}", reason);
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!("PDR result inconclusive: {}", reason);
            }

            let value = serde_json::json!({
                "result": "unknown",
                "reason": reason,
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("PDR error: {}", e);
            bail!("PDR failed: {}", e);
        }
    }
}

/// Run k-induction symbolic safety proving (Part of #3722).
#[cfg(feature = "ay")]
pub(super) fn run_kinduction_mode(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    max_k: usize,
    incremental: bool,
    output_format: OutputFormat,
) -> Result<()> {
    use tla_check::{check_kinduction, KInductionConfig, KInductionResult};

    if matches!(output_format, OutputFormat::TlcTool) {
        bail!("K-induction output is not supported in --output tlc-tool mode");
    }

    if matches!(output_format, OutputFormat::Human) {
        println!("K-induction mode: symbolic safety proving via ay");
        println!("Maximum induction depth: {}", max_k);
        if incremental {
            println!("Incremental solving: enabled (reusing solver across depths)");
        }
        println!();
    }

    let mut ctx = tla_check::EvalCtx::new();
    ctx.load_module(module);
    for m in checker_modules {
        ctx.load_module(m);
    }
    // EXTENDS-inherited VARIABLES (MC-wrapper specs) must be registered so
    // k-induction sees the real state vars instead of bailing on "No state variables".
    tla_check::register_state_vars_for_symbolic(&mut ctx, module, checker_modules);

    if let Err(e) = tla_check::bind_constants_from_config(&mut ctx, config) {
        bail!("Failed to bind constants: {}", e);
    }

    let start = Instant::now();
    let kind_config = KInductionConfig {
        max_k,
        incremental,
        ..KInductionConfig::default()
    };
    let kind_result = check_kinduction(module, config, &ctx, kind_config);
    let elapsed = start.elapsed();

    match kind_result {
        Ok(KInductionResult::Proved { k }) => {
            if matches!(output_format, OutputFormat::Human) {
                println!("K-INDUCTION: PROVED - All invariants hold (k={}).", k);
                println!();
                println!(
                    "The property is {}-inductive: it holds for ALL reachable states.",
                    k
                );
                println!();
                println!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let value = serde_json::json!({
                    "result": "proved",
                    "k": k,
                    "time_secs": elapsed.as_secs_f64(),
                });
                print_structured_symbolic_result(output_format, &value)?;
            }
            Ok(())
        }
        Ok(KInductionResult::Counterexample { depth, trace }) => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!(
                    "K-INDUCTION: VIOLATION - Counterexample found at depth {}.",
                    depth
                );
                eprintln!();
                print_bmc_trace_human(&trace);
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!("K-induction found counterexample at depth {}", depth);
            }

            let value = serde_json::json!({
                "result": "violation",
                "depth": depth,
                "trace_length": trace.len(),
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Ok(KInductionResult::Unknown { max_k, reason }) => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("K-INDUCTION: UNKNOWN - Could not prove safety.");
                eprintln!("Max depth: {}", max_k);
                eprintln!("Reason: {}", reason);
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!(
                    "K-induction result inconclusive at depth {}: {}",
                    max_k,
                    reason
                );
            }

            let value = serde_json::json!({
                "result": "unknown",
                "max_k": max_k,
                "reason": reason,
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("K-induction error: {}", e);
            bail!("K-induction failed: {}", e);
        }
    }
}

/// Run symbolic simulation mode (Part of #3757, Apalache Gap 9).
///
/// Uses ay SMT solving to explore random execution paths symbolically.
/// Each run follows one path by solving Init, then iteratively solving
/// Next to find concrete successor states, checking invariants at each step.
#[cfg(feature = "ay")]
pub(super) fn run_symbolic_sim_mode(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    num_runs: usize,
    max_depth: usize,
    output_format: OutputFormat,
) -> Result<()> {
    use tla_check::{symbolic_simulate, SymbolicSimConfig, SymbolicSimResult};

    if matches!(output_format, OutputFormat::TlcTool) {
        bail!("Symbolic simulation output is not supported in --output tlc-tool mode");
    }

    if matches!(output_format, OutputFormat::Human) {
        println!(
            "Symbolic simulation mode (Apalache-style): ay SMT-based random trace exploration"
        );
        println!("Runs: {}", num_runs);
        println!("Max depth per run: {}", max_depth);
        println!();
    }

    let mut ctx = tla_check::EvalCtx::new();
    ctx.load_module(module);
    for m in checker_modules {
        ctx.load_module(m);
    }
    // EXTENDS-inherited VARIABLES (MC-wrapper specs) must be registered for the
    // symbolic engine instead of bailing on "No state variables".
    tla_check::register_state_vars_for_symbolic(&mut ctx, module, checker_modules);

    if let Err(e) = tla_check::bind_constants_from_config(&mut ctx, config) {
        bail!("Failed to bind constants: {}", e);
    }

    let start = Instant::now();
    let sim_config = SymbolicSimConfig {
        num_runs,
        max_depth,
        ..SymbolicSimConfig::default()
    };
    let result = symbolic_simulate(module, config, &ctx, sim_config);
    let elapsed = start.elapsed();

    match result {
        Ok(SymbolicSimResult::NoViolation {
            runs_completed,
            max_depth_reached,
            total_states,
        }) => {
            if matches!(output_format, OutputFormat::Human) {
                println!("Symbolic simulation complete: No invariant violation found.");
                println!();
                println!("Statistics:");
                println!("  Runs completed: {}", runs_completed);
                println!("  Max depth reached: {}", max_depth_reached);
                println!("  Total states explored: {}", total_states);
                println!("  Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let value = serde_json::json!({
                    "mode": "symbolic_simulation",
                    "result": "no_violation",
                    "runs_completed": runs_completed,
                    "max_depth_reached": max_depth_reached,
                    "total_states": total_states,
                    "time_secs": elapsed.as_secs_f64(),
                });
                print_structured_symbolic_result(output_format, &value)?;
            }
            Ok(())
        }
        Ok(SymbolicSimResult::Violation {
            run_index,
            depth,
            trace,
        }) => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!(
                    "Symbolic simulation: VIOLATION found in run {} at depth {}.",
                    run_index, depth
                );
                eprintln!();
                print_bmc_trace_human(&trace);
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!(
                    "Symbolic simulation found counterexample in run {} at depth {}",
                    run_index,
                    depth
                );
            }

            let value = serde_json::json!({
                "mode": "symbolic_simulation",
                "result": "violation",
                "run_index": run_index,
                "depth": depth,
                "trace_length": trace.len(),
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Ok(SymbolicSimResult::Timeout {
            runs_completed,
            total_states,
            reason,
        }) => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("Symbolic simulation: TIMEOUT");
                eprintln!("Reason: {}", reason);
                eprintln!("Runs completed before timeout: {}", runs_completed);
                eprintln!("Total states explored: {}", total_states);
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!("Symbolic simulation timed out: {}", reason);
            }

            let value = serde_json::json!({
                "mode": "symbolic_simulation",
                "result": "timeout",
                "runs_completed": runs_completed,
                "total_states": total_states,
                "reason": reason,
                "time_secs": elapsed.as_secs_f64(),
            });
            print_structured_symbolic_result(output_format, &value)?;
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Symbolic simulation error: {}", e);
            bail!("Symbolic simulation failed: {}", e);
        }
    }
}

/// Run inductive invariant check (Part of #3756, Apalache Gap 8).
#[cfg(feature = "ay")]
pub(super) fn run_inductive_check_mode(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    invariant: &str,
    output_format: OutputFormat,
) -> Result<()> {
    use tla_check::{check_inductive, InductiveCheckConfig, InductiveCheckResult};

    if matches!(output_format, OutputFormat::TlcTool) {
        bail!("Inductive check output is not supported in --output tlc-tool mode");
    }

    if matches!(output_format, OutputFormat::Human) {
        println!("Inductive invariant check mode (Apalache-style)");
        println!("Invariant: {}", invariant);
        println!("Phase 1: Init => {}", invariant);
        println!("Phase 2: {} /\\ Next => {}\'", invariant, invariant);
        println!();
    }

    let mut ctx = tla_check::EvalCtx::new();
    ctx.load_module(module);
    for m in checker_modules {
        ctx.load_module(m);
    }
    // EXTENDS-inherited VARIABLES (MC-wrapper specs) must be registered for the
    // symbolic engine instead of bailing on "No state variables".
    tla_check::register_state_vars_for_symbolic(&mut ctx, module, checker_modules);

    if let Err(e) = tla_check::bind_constants_from_config(&mut ctx, config) {
        bail!("Failed to bind constants: {}", e);
    }

    let start = Instant::now();
    let ind_config = InductiveCheckConfig::new(invariant.to_string());
    let result = check_inductive(module, config, &ctx, &ind_config);
    let elapsed = start.elapsed();

    match result {
        Ok(InductiveCheckResult::Proved) => {
            if matches!(output_format, OutputFormat::Human) {
                println!("INDUCTIVE CHECK: PROVED");
                println!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let value = serde_json::json!({
                    "result": "proved",
                    "invariant": invariant,
                    "time_secs": elapsed.as_secs_f64(),
                });
                print_structured_symbolic_result(output_format, &value)?;
            }
            Ok(())
        }
        Ok(InductiveCheckResult::InitiationFailed { reason }) => {
            eprintln!("INDUCTIVE CHECK: FAILED (Phase 1 - Initiation)");
            eprintln!("Reason: {}", reason);
            bail!("Inductive check failed: initiation");
        }
        Ok(InductiveCheckResult::ConsecutionFailed { reason }) => {
            eprintln!("INDUCTIVE CHECK: FAILED (Phase 2 - Consecution)");
            eprintln!("Reason: {}", reason);
            bail!("Inductive check failed: consecution");
        }
        Ok(InductiveCheckResult::Unknown { phase, reason }) => {
            eprintln!("INDUCTIVE CHECK: UNKNOWN (Phase: {})", phase);
            eprintln!("Reason: {}", reason);
            bail!("Inductive check inconclusive");
        }
        Err(e) => bail!("Inductive check error: {}", e),
    }
}

/// Apply checker configuration common to all three modes (adaptive, sequential, parallel).
macro_rules! apply_common_checker_config {
    ($checker:expr, $cfg:expr) => {
        $checker.set_deadlock_check($cfg.check_deadlock);
        $checker.set_continue_on_error($cfg.continue_on_error);
        // `false` is semantically meaningful here: the sequential checker uses
        // set_store_states(false) to enable fp-only liveness caching (#3175).
        $checker.set_store_states($cfg.store_states);
        if let Some(ref storage) = $cfg.fingerprint_storage {
            $checker.set_fingerprint_storage(storage.clone());
        }
        $checker.set_collision_check_mode($cfg.collision_check_mode);
        if !$cfg.resolved_fairness.is_empty() {
            $checker.set_fairness($cfg.resolved_fairness.to_vec());
        }
        if $cfg.max_states > 0 {
            $checker.set_max_states($cfg.max_states);
        }
        if $cfg.max_depth > 0 {
            $checker.set_max_depth($cfg.max_depth);
        }
        if $cfg.memory_limit > 0 {
            // Convert megabytes to bytes; saturate to prevent silent wrapping
            // on 32-bit targets (Part of #2751).
            let limit_bytes = $cfg.memory_limit.saturating_mul(1024 * 1024);
            $checker.set_memory_limit(limit_bytes);
        } else {
            // Part of #2751: auto-detect system RAM so memory monitoring is
            // active by default. Users get a warning at ~72% of RAM and
            // graceful stop at ~85% instead of an OOM kill with no warning.
            // Multi-instance aware: divides budget by concurrent ty processes.
            if let Some((limit_bytes, total_bytes, instances)) =
                tla_check::memory_policy_system_default_info()
            {
                $checker.set_memory_limit(limit_bytes);
                tla_check::log_memory_budget(limit_bytes, total_bytes, instances);
            }
        }
        if $cfg.disk_limit > 0 {
            // Part of #3282: Convert megabytes to bytes for disk limit.
            let limit_bytes = $cfg.disk_limit.saturating_mul(1024 * 1024);
            $checker.set_disk_limit(limit_bytes);
        } else {
            // Part of #3282: auto-detect available disk so disk monitoring is
            // active by default. Users get a graceful stop instead of filling
            // the disk and crashing (TLC disk-exhaustion post-mortem).
            if let Some(limit_bytes) = tla_check::disk_limit_system_default() {
                $checker.set_disk_limit(limit_bytes);
            }
        }
    };
}

/// Register file paths and resolved spec — shared by adaptive and sequential modes.
macro_rules! register_files_and_spec {
    ($checker:expr, $cfg:expr) => {
        $checker.register_file_path(FileId(0), $cfg.file.to_path_buf());
        for (fid, path) in &$cfg.file_paths {
            $checker.register_file_path(*fid, path.clone());
        }
        if let Some(ref resolved) = $cfg.resolved_spec {
            $checker.register_inline_next(resolved)?;
            $checker.set_stuttering_allowed(resolved.stuttering_allowed);
        }
    };
}

/// Set up TLC tool output format callbacks for the sequential model checker.
///
/// Emits the TLC_COMPUTING_INIT message immediately and installs init-progress
/// and BFS-progress callbacks that produce TLC-compatible machine-readable output.
fn setup_tlc_tool_callbacks(
    checker: &mut ModelChecker<'_>,
    tool_out: &mut Option<tlc_tool::TlcToolOutput>,
) {
    if let Some(out) = tool_out.as_mut() {
        out.emit(
            tlc_codes::ec::TLC_COMPUTING_INIT,
            tlc_codes::mp::NONE,
            tlc_tool::format_tlc_computing_init_message(),
        );
    }
    checker.set_init_progress_callback(Box::new(|init| {
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let mut out = tlc_tool::TlcToolOutput::new();
        out.emit(
            tlc_codes::ec::TLC_INIT_GENERATED1,
            tlc_codes::mp::NONE,
            &tlc_tool::format_tlc_init_generated1_message(init.distinct_states as u64, &now),
        );
    }));

    let last_emit = std::sync::Arc::new(std::sync::Mutex::new(0.0f64));
    let last_emit2 = std::sync::Arc::clone(&last_emit);
    checker.set_progress_callback(Box::new(move |progress| {
        let should_emit = match last_emit2.lock() {
            Ok(mut last) => {
                if progress.elapsed_secs - *last >= 5.0 {
                    *last = progress.elapsed_secs;
                    true
                } else {
                    false
                }
            }
            Err(_) => true,
        };
        if !should_emit {
            return;
        }
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let mut out = tlc_tool::TlcToolOutput::new();
        out.emit(
            tlc_codes::ec::TLC_PROGRESS_STATS,
            tlc_codes::mp::NONE,
            &tlc_tool::format_tlc_progress_stats_message(
                progress.current_depth as u64,
                progress.states_found as u64,
                progress.states_found as u64,
                progress.queue_size as u64,
                &now,
                progress.elapsed_secs,
            ),
        );
        // Emit memory usage as a separate info line (not part of TLC protocol)
        if let Some(rss) = progress.memory_usage_bytes {
            let mb = rss as f64 / (1024.0 * 1024.0);
            eprintln!("  Memory: {mb:.1} MB RSS");
        }
    }));
}

/// Configuration for a model checker run (auto, sequential, or parallel).
pub(super) struct ModelCheckerRunCfg<'a> {
    pub(super) module: &'a tla_core::ast::Module,
    pub(super) checker_modules: &'a [&'a tla_core::ast::Module],
    pub(super) config: &'a Config,
    pub(super) workers: usize,
    pub(super) file: &'a Path,
    pub(super) file_paths: Vec<(FileId, PathBuf)>,
    pub(super) resolved_spec: &'a Option<tla_check::ResolvedSpec>,
    pub(super) check_deadlock: bool,
    pub(super) show_coverage: bool,
    pub(super) strict_vacuity: bool,
    pub(super) continue_on_error: bool,
    pub(super) store_states: bool,
    pub(super) no_trace: bool,
    pub(super) fingerprint_storage: &'a Option<std::sync::Arc<dyn tla_check::FingerprintSet>>,
    pub(super) trace_file: Option<TraceFile>,
    pub(super) trace_locs_storage: Option<TraceLocationsStorage>,
    pub(super) resolved_fairness: &'a [tla_check::FairnessConstraint],
    pub(super) max_states: usize,
    pub(super) max_depth: usize,
    /// Part of #2751: memory limit in megabytes (0 = unlimited).
    pub(super) memory_limit: usize,
    /// Part of #3282: disk limit in megabytes (0 = unlimited).
    pub(super) disk_limit: usize,
    pub(super) output_format: OutputFormat,
    pub(super) progress_callback: Box<dyn Fn(&Progress) + Send + Sync>,
    pub(super) checkpoint_dir: &'a Option<PathBuf>,
    pub(super) checkpoint_interval: u64,
    pub(super) resume_from: &'a Option<PathBuf>,
    pub(super) config_path: &'a Path,
    pub(super) tool_out: &'a mut Option<tlc_tool::TlcToolOutput>,
    /// Fingerprint collision detection mode.
    pub(super) collision_check_mode: tla_check::CollisionCheckMode,
}

/// Run the model checker in auto, sequential, or parallel mode (extracted from cmd_check).
// Thin wrapper over `run_model_checker_with_frontend_source`; currently only
// exercised by unit tests, but kept as the simple default-frontend entry point.
#[allow(dead_code)]
pub(super) fn run_model_checker(
    cfg: ModelCheckerRunCfg<'_>,
) -> Result<(CheckResult, Option<String>)> {
    run_model_checker_with_frontend_source(cfg, false)
}

/// Run the model checker while preserving frontend source-family provenance.
pub(super) fn run_model_checker_with_frontend_source(
    cfg: ModelCheckerRunCfg<'_>,
    frontend_source_is_quint: bool,
) -> Result<(CheckResult, Option<String>)> {
    if cfg.strict_vacuity {
        if cfg.config.view.is_some() {
            bail!("--strict-vacuity cannot be combined with a VIEW state quotient");
        }
        if cfg.config.symmetry.is_some() {
            bail!("--strict-vacuity cannot be combined with a SYMMETRY state quotient");
        }
        if cfg.config.por_enabled {
            bail!("--strict-vacuity cannot be combined with explicit POR");
        }
        if cfg.config.auto_por == Some(true) {
            bail!("--strict-vacuity cannot be combined with explicit auto-POR");
        }
    }
    if cfg.strict_vacuity && cfg.workers > 1 {
        bail!(
            "--strict-vacuity is only supported with --workers 0 or --workers 1 \
             because dead-action evidence requires exhaustive sequential BFS"
        );
    }
    let force_sequential_auto = auto_sequential_reason(
        cfg.config,
        cfg.workers,
        cfg.continue_on_error,
        cfg.strict_vacuity,
    )
    .is_some();
    if cfg.workers == 0 && !force_sequential_auto {
        // Auto mode: use adaptive checker
        if cfg.trace_file.is_some() {
            bail!("--trace-file is only supported with --workers 1 (sequential mode)");
        }
        if cfg.trace_locs_storage.is_some() {
            bail!("--mmap-trace-locations is only supported with --workers 1 (sequential mode)");
        }
        let runtime_config = cfg.config.runtime_model_config();
        let mut checker =
            AdaptiveChecker::new_with_extends(cfg.module, cfg.checker_modules, &runtime_config);
        register_files_and_spec!(checker, cfg);
        apply_common_checker_config!(checker, cfg);
        checker.set_collect_coverage(cfg.show_coverage);
        if cfg.no_trace {
            checker.set_auto_create_trace_file(false);
        }
        // Part of #3247: progress always on for Human output.
        if matches!(cfg.output_format, OutputFormat::Human) {
            checker.set_progress_callback(cfg.progress_callback);
        }
        let (mut result, analysis) = checker.check();
        enrich_structured_check_result_shared_engine_report(
            &mut result,
            cfg.output_format,
            frontend_source_is_quint,
        );
        let strategy_info = analysis.map(|a| {
            format!(
                "Strategy: {} (estimated {} states, branching factor {:.2})",
                a.strategy, a.estimated_states, a.avg_branching_factor
            )
        });
        Ok((result, strategy_info))
    } else if cfg.workers == 1 || force_sequential_auto {
        let forced_strategy_info = auto_sequential_reason(
            cfg.config,
            cfg.workers,
            cfg.continue_on_error,
            cfg.strict_vacuity,
        )
        .map(|reason| format!("Strategy: sequential (auto: {reason})"));

        let mut runtime_config = cfg.config.runtime_model_config();
        if cfg.strict_vacuity {
            runtime_config.auto_por = Some(false);
            runtime_config.use_compiled_bfs = Some(false);
        }
        let mut checker =
            ModelChecker::new_with_extends(cfg.module, cfg.checker_modules, &runtime_config);
        checker.set_frontend_source_is_quint(frontend_source_is_quint);
        register_files_and_spec!(checker, cfg);
        apply_common_checker_config!(checker, cfg);
        checker.set_collect_coverage(cfg.show_coverage);
        // V2 vacuity gate (TRUST_VACUITY_GATE §1.A): enable per-action coverage
        // TRACKING by default on the sequential path so the dead-action WARNING
        // is default-on (the verbose report stays gated behind --coverage). This
        // is the forced-sequential / `--workers 1` path where invariants are
        // present, so it does not regress the parallel auto path.
        checker.set_default_dead_action_coverage();
        // Strict vacuity changes the command verdict, so its dead-action
        // evidence is explicit and must never yield to an optional fast path.
        if cfg.strict_vacuity {
            checker.set_force_explicit_bfs(true);
            checker.set_auto_symmetry(false);
            checker.set_track_coverage(true);
        }
        if cfg.no_trace {
            checker.set_auto_create_trace_file(false);
        }
        if let Some(tf) = cfg.trace_file {
            checker.set_trace_file(tf);
        }
        if let Some(storage) = cfg.trace_locs_storage {
            checker.set_trace_locations_storage(storage);
        }
        if matches!(cfg.output_format, OutputFormat::TlcTool) {
            setup_tlc_tool_callbacks(&mut checker, cfg.tool_out);
        } else if matches!(cfg.output_format, OutputFormat::Human) {
            // Part of #3247: progress always on for Human output.
            checker.set_progress_callback(cfg.progress_callback);
        }
        if cfg.checkpoint_dir.is_some() || cfg.resume_from.is_some() {
            checker.set_checkpoint_paths(
                Some(cfg.file.to_string_lossy().to_string()),
                Some(cfg.config_path.to_string_lossy().to_string()),
            );
        }
        if let Some(ref dir) = cfg.checkpoint_dir {
            checker.set_checkpoint(
                dir.clone(),
                std::time::Duration::from_secs(cfg.checkpoint_interval),
            );
        }
        let mut result = if let Some(ref resume_dir) = cfg.resume_from {
            checker.check_with_resume(resume_dir).with_context(|| {
                format!("Failed to resume from checkpoint: {}", resume_dir.display())
            })?
        } else {
            checker.check()
        };
        enrich_structured_check_result_shared_engine_report(
            &mut result,
            cfg.output_format,
            frontend_source_is_quint,
        );
        Ok((result, forced_strategy_info))
    } else {
        // Parallel mode
        if cfg.trace_file.is_some() {
            bail!("--trace-file is only supported with --workers 1 (sequential mode)");
        }
        if cfg.trace_locs_storage.is_some() {
            bail!("--mmap-trace-locations is only supported with --workers 1 (sequential mode)");
        }
        if cfg.show_coverage {
            bail!("--coverage is only supported with --workers 0 or --workers 1");
        }
        // Trace invariants are implemented in ModelChecker (sequential) only;
        // ParallelChecker would silently skip them and report a false PASS.
        // The --workers 0 auto route already forces sequential when trace
        // invariants are present (AdaptiveChecker), so only the explicit
        // --workers N>1 request must be rejected here.
        if !cfg.config.trace_invariants.is_empty() {
            bail!(
                "--trace-inv is only supported with --workers 0 or --workers 1 \
                 (trace invariants are evaluated by the sequential checker)"
            );
        }
        let runtime_config = cfg.config.runtime_model_config();
        let mut checker = ParallelChecker::new_with_extends(
            cfg.module,
            cfg.checker_modules,
            &runtime_config,
            cfg.workers,
        );
        register_files_and_spec!(checker, cfg);
        apply_common_checker_config!(checker, cfg);
        // Part of #3247: progress always on for Human output.
        if matches!(cfg.output_format, OutputFormat::Human) {
            checker.set_progress_callback(cfg.progress_callback);
        }
        // Part of #2749: Wire checkpoint/resume for parallel mode.
        if cfg.checkpoint_dir.is_some() || cfg.resume_from.is_some() {
            checker.set_checkpoint_paths(
                Some(cfg.file.to_string_lossy().to_string()),
                Some(cfg.config_path.to_string_lossy().to_string()),
            );
        }
        if let Some(ref dir) = cfg.checkpoint_dir {
            checker.set_checkpoint(
                dir.clone(),
                std::time::Duration::from_secs(cfg.checkpoint_interval),
            );
        }
        let mut result = if let Some(ref resume_dir) = cfg.resume_from {
            checker.check_with_resume(resume_dir).with_context(|| {
                format!("Failed to resume from checkpoint: {}", resume_dir.display())
            })?
        } else {
            checker.check()
        };
        enrich_structured_check_result_shared_engine_report(
            &mut result,
            cfg.output_format,
            frontend_source_is_quint,
        );
        Ok((result, None))
    }
}

/// Run portfolio racing mode: parallel BFS + symbolic strategies (Part of #3717).
///
/// Spawns multiple verification lanes in parallel via [`run_portfolio`] and
/// terminates when the first one reaches a definitive result.
pub(super) fn run_portfolio_mode(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    strategy_names: &[String],
    output_format: OutputFormat,
) -> Result<()> {
    run_portfolio_mode_with_frontend_source(
        module,
        checker_modules,
        config,
        strategy_names,
        output_format,
        portfolio_frontend_source_is_quint_from_cli_invocation(),
    )
}

fn portfolio_frontend_source_code(frontend_source_is_quint: bool) -> &'static str {
    if frontend_source_is_quint {
        "quint"
    } else {
        "tla"
    }
}

fn shared_engine_frontend_family_code(frontend_source_is_quint: bool) -> &'static str {
    if frontend_source_is_quint {
        "quint"
    } else {
        "tla_plus"
    }
}

fn alternate_shared_engine_beneficiary(frontend_family: &str) -> &'static str {
    if frontend_family == "quint" {
        "tla_plus"
    } else {
        "quint"
    }
}

const SHARED_ENGINE_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer";
const SHARED_ENGINE_FRONTEND_BLOCKERS: &str = concat!(
    "tla_plus:blocked_cli_report_not_default_production_support:F3;",
    "quint:blocked_cli_report_not_default_production_support:F3;",
    "mcc_petri:blocked_cli_report_not_default_production_support:F3;",
    "aiger:blocked_cli_report_not_default_production_support:F3;",
    "btor2:blocked_cli_report_not_default_production_support:F3;",
    "vmt_transition_system:blocked_cli_report_not_default_production_support:F3;",
    "ay_analytical:blocked_cli_report_not_default_production_support:F3;",
    "witness_replay:blocked_cli_report_not_default_production_support:F3;",
    "future_importer:blocked_reserved_importer_contract:F3"
);

fn json_scalar_to_report_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Bool(v) => Some(v.to_string()),
        serde_json::Value::Number(v) => Some(v.to_string()),
        _ => None,
    }
}

fn collect_report_fields(
    report: Option<&serde_json::Value>,
) -> std::collections::BTreeMap<String, String> {
    let mut fields = std::collections::BTreeMap::new();
    let Some(report) = report else {
        return fields;
    };

    if let Some(field_obj) = report.get("fields").and_then(serde_json::Value::as_object) {
        for (key, value) in field_obj {
            if let Some(value) = json_scalar_to_report_string(value) {
                fields.insert(key.clone(), value);
            }
        }
    }

    if let Some(evidence_rows) = report.get("evidence").and_then(serde_json::Value::as_array) {
        for row in evidence_rows.iter().filter_map(serde_json::Value::as_str) {
            if let Some(row_fields) = evidence_row_fields(row) {
                for (key, value) in row_fields {
                    fields.entry(key).or_insert(value);
                }
            }
        }
    }

    fields
}

fn evidence_row_fields(row: &str) -> Option<std::collections::BTreeMap<String, String>> {
    let mut fields = std::collections::BTreeMap::new();
    for token in row.split_whitespace() {
        let Some((key, value)) = token.split_once('=') else {
            continue;
        };
        let value = value.trim_matches('"').to_string();
        if let Some(previous) = fields.insert(key.to_string(), value.clone()) {
            if previous != value {
                return None;
            }
        }
    }
    Some(fields)
}

fn strict_shared_engine_validation_receipt_fields(
    row: &str,
) -> Option<std::collections::BTreeMap<String, String>> {
    let mut tokens = row.split_whitespace();
    let producer = tokens.next()?;
    if producer.is_empty() || producer.contains('=') {
        return None;
    }
    if tokens.next()? != "shared_engine_validation_receipt" {
        return None;
    }

    let mut fields = std::collections::BTreeMap::new();
    for token in tokens {
        let (key, value) = token.split_once('=')?;
        if key.is_empty() || value.is_empty() || key.contains('"') || value.contains('"') {
            return None;
        }
        if fields.insert(key.to_string(), value.to_string()).is_some() {
            return None;
        }
    }
    Some(fields)
}

fn report_field_value(
    fields: &std::collections::BTreeMap<String, String>,
    aliases: &[&str],
) -> Option<String> {
    aliases
        .iter()
        .find_map(|key| fields.get(*key))
        .filter(|value| !value.trim().is_empty())
        .cloned()
}

fn analytical_receipt_ready(receipts: &[String]) -> bool {
    receipts.iter().any(|row| {
        let Some(fields) = strict_shared_engine_validation_receipt_fields(row) else {
            return false;
        };
        field_equals(&fields, "receipt_role", "analytical_solve")
            && field_equals(&fields, "validator_kind", "ay_proof")
            && field_equals(&fields, "solver_family", "ay")
            && field_starts_with(&fields, "backend_code", "ay_")
            && field_is_concrete(&fields, "receipt_identity")
            && field_equals(&fields, "digest_algorithm", "ay_fingerprint_identity")
            && field_is_concrete(&fields, "digest")
            && field_equals(&fields, "receipt_status", "accepted")
            && field_equals(&fields, "receipt_validation", "valid")
            && field_equals(&fields, "failure_reason", "none")
            && field_equals(&fields, "publication_blocker", "none")
            && field_equals(&fields, "publication_readiness", "ready")
    })
}

fn field_equals(
    fields: &std::collections::BTreeMap<String, String>,
    key: &str,
    expected: &str,
) -> bool {
    fields
        .get(key)
        .is_some_and(|value| value.eq_ignore_ascii_case(expected))
}

fn field_starts_with(
    fields: &std::collections::BTreeMap<String, String>,
    key: &str,
    expected_prefix: &str,
) -> bool {
    fields
        .get(key)
        .is_some_and(|value| value.to_ascii_lowercase().starts_with(expected_prefix))
}

fn field_is_concrete(fields: &std::collections::BTreeMap<String, String>, key: &str) -> bool {
    fields.get(key).is_some_and(|value| {
        let value = value.trim();
        !value.is_empty()
            && !matches!(
                value.to_ascii_lowercase().as_str(),
                "none" | "missing" | "not_observed" | "not_checked" | "not_evaluated"
            )
    })
}

fn normalized_report_token(value: &str) -> String {
    let mut token = String::new();
    let mut previous_was_separator = true;
    for ch in value.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            token.push(ch.to_ascii_lowercase());
            previous_was_separator = false;
        } else if !previous_was_separator {
            token.push('_');
            previous_was_separator = true;
        }
    }
    if previous_was_separator {
        token.pop();
    }
    token
}

fn current_head_freshness_allows_faster_than_tlc_claim(status: &str) -> bool {
    matches!(
        normalized_report_token(status).as_str(),
        "current"
            | "current_head"
            | "current_head_evidence"
            | "current_repo_head"
            | "current_repo_head_evidence"
            | "current_commit"
            | "current_commit_evidence"
    )
}

fn benchmark_gate_allows_faster_than_tlc_claim(status: &str) -> bool {
    matches!(
        normalized_report_token(status).as_str(),
        "pass" | "passed" | "enforced"
    )
}

fn report_field_nonnegative_f64(
    fields: &std::collections::BTreeMap<String, String>,
    aliases: &[&str],
) -> Option<f64> {
    report_field_value(fields, aliases).and_then(|value| {
        let parsed = value.parse::<f64>().ok()?;
        (parsed.is_finite() && parsed >= 0.0).then_some(parsed)
    })
}

fn cold_wall_evidence_allows_faster_than_tlc_claim(
    fields: &std::collections::BTreeMap<String, String>,
) -> bool {
    let cold_wall_speedup = report_field_nonnegative_f64(
        fields,
        &[
            "cold_wall_speedup_vs_tlc",
            "trust_cg_cold_wall_speedup_vs_tlc",
            "total_speedup_vs_tlc",
        ],
    );
    if cold_wall_speedup.is_some_and(|speedup| speedup > 1.0) {
        return true;
    }
    let tlc_wall = report_field_nonnegative_f64(
        fields,
        &[
            "tlc_wall_seconds",
            "tlc_median_seconds",
            "tlc_wall_median_seconds",
        ],
    );
    let trust_cg_cold_wall = report_field_nonnegative_f64(
        fields,
        &[
            "trust_cg_cold_wall_seconds",
            "trust_cg_wall_seconds",
            "trust_cg_wall_median_seconds",
            "trust_cg_cold_wall_median_seconds",
        ],
    );
    matches!((tlc_wall, trust_cg_cold_wall), (Some(tlc), Some(trust_cg)) if trust_cg < tlc)
}

fn faster_than_tlc_claim_blocker(
    gate_allowed: bool,
    current_head_allowed: bool,
    cold_wall_allowed: bool,
) -> &'static str {
    if !gate_allowed {
        "blocked_benchmark_gate_not_passed"
    } else if !current_head_allowed {
        "blocked_current_head_evidence_missing"
    } else if !cold_wall_allowed {
        "blocked_cold_wall_win_missing"
    } else {
        "none"
    }
}

fn put_strict_field(
    strict_fields: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: impl Into<String>,
) {
    strict_fields.insert(key.to_string(), serde_json::json!(value.into()));
}

fn put_strict_field_from_report(
    strict_fields: &mut serde_json::Map<String, serde_json::Value>,
    report_fields: &std::collections::BTreeMap<String, String>,
    key: &str,
    aliases: &[&str],
    default: &str,
) {
    let value = report_field_value(report_fields, aliases).unwrap_or_else(|| default.to_string());
    put_strict_field(strict_fields, key, value);
}

fn strict_shared_engine_report_fields(
    existing_report: Option<&serde_json::Value>,
    frontend_source_is_quint: bool,
    validation_receipts: &[String],
) -> serde_json::Map<String, serde_json::Value> {
    let report_fields = collect_report_fields(existing_report);
    let mut fields = serde_json::Map::new();
    let origin_frontend = shared_engine_frontend_family_code(frontend_source_is_quint);

    put_strict_field(&mut fields, "row_kind", "cli_shared_engine_report");
    put_strict_field(&mut fields, "origin_frontend", origin_frontend);
    put_strict_field(&mut fields, "shared_engine_component", "cli_report_json");
    put_strict_field(
        &mut fields,
        "generic_prerequisites",
        "cli_json_shared_engine_strict_fields",
    );
    put_strict_field(&mut fields, "first_beneficiary", origin_frontend);
    put_strict_field(
        &mut fields,
        "second_beneficiary",
        alternate_shared_engine_beneficiary(origin_frontend),
    );
    put_strict_field(
        &mut fields,
        "compatible_frontend_families",
        SHARED_ENGINE_FRONTEND_FAMILIES,
    );
    put_strict_field(&mut fields, "active_frontend_families", origin_frontend);
    put_strict_field(&mut fields, "default_compatible_frontend_families", "");
    put_strict_field(
        &mut fields,
        "frontend_family_blockers",
        SHARED_ENGINE_FRONTEND_BLOCKERS,
    );
    put_strict_field(&mut fields, "extraction_status", "cli_json_report_ready");
    put_strict_field(&mut fields, "blocker_status", "tracked_blockers");
    put_strict_field(&mut fields, "owner", "F3");
    put_strict_field(
        &mut fields,
        "acceptance_test",
        "cargo_test_-p_tla-cli_cmd_check_runner_shared_engine_report_contract",
    );
    put_strict_field(
        &mut fields,
        "downstream_consumption_status",
        "evidence_only_contract",
    );
    put_strict_field(&mut fields, "evidence_basis", "cli_json_report_contract");

    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "prepared_admission_receipt",
        &[
            "prepared_admission_receipt",
            "prepared_program_admission_receipt",
            "prepared_frontier_admission_receipt",
            "prepared_program_receipt",
            "shared_engine_prepared_admission_receipt",
        ],
        "blocked_missing_prepared_admission_receipt",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "prepared_admission_status",
        &[
            "prepared_admission_status",
            "prepared_program_admission_status",
            "prepared_frontier_admission_status",
            "shared_engine_prepared_admission_status",
        ],
        "blocked_missing_prepared_admission_receipt",
    );

    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "fingerprint_evidence_label",
        &[
            "fingerprint_evidence_label",
            "fingerprint_label",
            "fingerprint_chain_label",
            "shared_engine_fingerprint_evidence_label",
        ],
        "blocked_missing_current_fingerprint_chain",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "prepared_program_fingerprint",
        &[
            "prepared_program_fingerprint",
            "prepared_program_fingerprint_label",
            "shared_engine_prepared_program_fingerprint",
        ],
        "blocked_missing_prepared_program_fingerprint",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "storage_layout_fingerprint",
        &[
            "storage_layout_fingerprint",
            "storage_layout_fingerprint_label",
            "shared_engine_storage_layout_fingerprint",
        ],
        "blocked_missing_storage_layout_fingerprint",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "artifact_fingerprint",
        &[
            "artifact_fingerprint",
            "artifact_fingerprint_label",
            "shared_engine_artifact_fingerprint",
        ],
        "blocked_missing_artifact_fingerprint",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "proof_or_witness_fingerprint",
        &[
            "proof_or_witness_fingerprint",
            "proof_fingerprint",
            "witness_fingerprint",
            "certificate_fingerprint",
            "proof_or_witness_fingerprint_label",
        ],
        "blocked_missing_proof_or_witness_fingerprint",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "prepared_program_identity",
        &["prepared_program_identity", "source_identity"],
        "blocked_missing_prepared_program_identity",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "frontend_payload_identity",
        &["frontend_payload_identity", "payload_identity"],
        "blocked_missing_frontend_payload_identity",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "artifact_identity",
        &["artifact_identity", "artifact_cache_identity"],
        "blocked_missing_artifact_identity",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "storage_policy_identity",
        &["storage_policy_identity", "storage_layout_identity"],
        "blocked_missing_storage_policy_identity",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "fingerprint_policy_identity",
        &["fingerprint_policy_identity"],
        "blocked_missing_fingerprint_policy_identity",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "fingerprint_identity",
        &["fingerprint_identity", "current_fingerprint_identity"],
        "blocked_missing_fingerprint_identity",
    );
    put_strict_field(
        &mut fields,
        "fingerprint_chain_current_status",
        if report_field_value(
            &report_fields,
            &[
                "prepared_program_fingerprint",
                "storage_layout_fingerprint",
                "artifact_fingerprint",
                "proof_or_witness_fingerprint",
                "fingerprint_identity",
            ],
        )
        .is_some()
        {
            "observed"
        } else {
            "blocked_missing_current_identity"
        },
    );

    let analytical_ready = analytical_receipt_ready(validation_receipts);
    let analytical_receipt = report_field_value(
        &report_fields,
        &[
            "analytical_solve_receipt",
            "analytical_ay_solve_receipt",
            "ay_solve_receipt",
            "solver_obligation_receipt",
            "shared_engine_analytical_solve_receipt",
        ],
    )
    .unwrap_or_else(|| {
        if analytical_ready {
            "analytical_ay_solve_receipt_observed"
        } else {
            "blocked_missing_analytical_solve_receipt"
        }
        .to_string()
    });
    put_strict_field(&mut fields, "analytical_solve_receipt", analytical_receipt);
    put_strict_field(
        &mut fields,
        "analytical_solve_receipt_readiness",
        if analytical_ready {
            "ready"
        } else {
            "blocked_missing_analytical_solve_receipt"
        },
    );
    put_strict_field(
        &mut fields,
        "ay_solve_receipt_readiness",
        if analytical_ready {
            "ready"
        } else {
            "blocked_missing_ay_solve_receipt"
        },
    );

    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "native_callable_receipt",
        &[
            "native_callable_receipt",
            "native_action_callout_receipt",
            "native_action_callout_batch_receipt",
            "trust_cg_native_callable_receipt",
            "shared_engine_native_callable_receipt",
        ],
        "blocked_missing_native_callable_receipt",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "native_callable_receipt_readiness",
        &[
            "native_callable_receipt_readiness",
            "native_callable_readiness",
            "native_action_callout_readiness",
        ],
        "blocked_missing_native_callable_receipt",
    );

    let current_head_freshness = report_field_value(
        &report_fields,
        &[
            "current_head_freshness",
            "benchmark_freshness",
            "evidence_freshness",
            "current_head_evidence",
        ],
    )
    .unwrap_or_else(|| "not_checked".to_string());
    put_strict_field(
        &mut fields,
        "current_head_freshness",
        current_head_freshness.clone(),
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "evidence_json_freshness",
        &[
            "evidence_json_freshness",
            "evidence_json_current_head",
            "evidence_json_commit_freshness",
            "evidence_json_git_commit",
            "shared_engine_evidence_json_freshness",
        ],
        "not_checked",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "cold_warm_cache_label",
        &[
            "cold_warm_cache_label",
            "cold_warm_label",
            "cache_reuse_label",
            "artifact_cache_label",
            "launch_cache_label",
            "run_cache_label",
            "cache_label",
            "runtime_setup_cache_label",
        ],
        "not_timed",
    );
    put_strict_field_from_report(
        &mut fields,
        &report_fields,
        "cache_temperature",
        &[
            "cache_temperature",
            "runtime_setup_temperature_label",
            "runtime_setup_temperature",
        ],
        "not_timed",
    );
    let benchmark_gate_status = report_field_value(
        &report_fields,
        &[
            "benchmark_gate_status",
            "benchmark_gate",
            "cold_wall_benchmark_gate",
            "faster_than_tlc_benchmark_gate",
            "shared_engine_benchmark_gate_status",
        ],
    )
    .unwrap_or_else(|| "not_claimed".to_string());
    put_strict_field(
        &mut fields,
        "benchmark_gate_status",
        benchmark_gate_status.clone(),
    );
    let requested_faster_than_tlc = report_field_value(
        &report_fields,
        &[
            "faster_than_tlc_claim_supported",
            "faster_than_tlc_supported",
            "cold_wall_faster_than_tlc_supported",
        ],
    )
    .is_some_and(|value| value.eq_ignore_ascii_case("true"));
    let benchmark_gate_allows = benchmark_gate_allows_faster_than_tlc_claim(&benchmark_gate_status);
    let current_head_allows =
        current_head_freshness_allows_faster_than_tlc_claim(&current_head_freshness);
    let cold_wall_allows = cold_wall_evidence_allows_faster_than_tlc_claim(&report_fields);
    let faster_than_tlc_supported = requested_faster_than_tlc
        && benchmark_gate_allows
        && current_head_allows
        && cold_wall_allows;
    put_strict_field(
        &mut fields,
        "faster_than_tlc_claim_supported",
        if faster_than_tlc_supported {
            "true"
        } else {
            "false"
        },
    );
    if requested_faster_than_tlc && !faster_than_tlc_supported {
        put_strict_field(
            &mut fields,
            "faster_than_tlc_claim_blocker",
            faster_than_tlc_claim_blocker(
                benchmark_gate_allows,
                current_head_allows,
                cold_wall_allows,
            ),
        );
    }

    fields
}

fn strict_shared_engine_report_row(fields: &serde_json::Map<String, serde_json::Value>) -> String {
    let mut row = String::from("TY cli_shared_engine_current_status");
    for (key, value) in fields {
        let Some(value) = json_scalar_to_report_string(value) else {
            continue;
        };
        row.push(' ');
        row.push_str(key);
        row.push('=');
        row.push_str(&value.split_whitespace().collect::<Vec<_>>().join("_"));
    }
    row
}

fn strict_shared_engine_report_json(
    existing_report: Option<serde_json::Value>,
    frontend_source_is_quint: bool,
    validation_receipts: &[String],
) -> serde_json::Value {
    let strict_fields = strict_shared_engine_report_fields(
        existing_report.as_ref(),
        frontend_source_is_quint,
        validation_receipts,
    );
    let strict_row = strict_shared_engine_report_row(&strict_fields);

    let mut report = match existing_report {
        Some(serde_json::Value::Object(map)) => map,
        Some(other) => {
            let mut map = serde_json::Map::new();
            map.insert("source_report".to_string(), other);
            map
        }
        None => serde_json::Map::new(),
    };

    report
        .entry("schema".to_string())
        .or_insert_with(|| serde_json::json!("ty.cli.shared_engine_current_report.v1"));
    report
        .entry("schema_version".to_string())
        .or_insert_with(|| serde_json::json!(1));
    report
        .entry("backend".to_string())
        .or_insert_with(|| serde_json::json!("CLI"));
    report
        .entry("kind".to_string())
        .or_insert_with(|| serde_json::json!("shared_engine_current_status"));

    let mut merged_fields = report
        .remove("fields")
        .and_then(|value| match value {
            serde_json::Value::Object(map) => Some(map),
            _ => None,
        })
        .unwrap_or_default();
    for (key, value) in strict_fields {
        merged_fields.insert(key, value);
    }
    report.insert(
        "fields".to_string(),
        serde_json::Value::Object(merged_fields),
    );

    let mut evidence_rows = report
        .remove("evidence")
        .and_then(|value| match value {
            serde_json::Value::Array(rows) => Some(rows),
            serde_json::Value::String(row) => Some(vec![serde_json::Value::String(row)]),
            _ => None,
        })
        .unwrap_or_default();
    evidence_rows.push(serde_json::Value::String(strict_row));
    report.insert(
        "evidence".to_string(),
        serde_json::Value::Array(evidence_rows),
    );

    serde_json::Value::Object(report)
}

fn stats_mut_for_cli_report(result: &mut CheckResult) -> Option<&mut tla_check::CheckStats> {
    Some(match result {
        CheckResult::Success(stats)
        | CheckResult::InvariantViolation { stats, .. }
        | CheckResult::PropertyViolation { stats, .. }
        | CheckResult::LivenessViolation { stats, .. }
        | CheckResult::Deadlock { stats, .. }
        | CheckResult::Vacuous { stats, .. }
        | CheckResult::Error { stats, .. }
        | CheckResult::LimitReached { stats, .. } => stats,
        _ => return None,
    })
}

fn enrich_structured_check_result_shared_engine_report(
    result: &mut CheckResult,
    output_format: OutputFormat,
    frontend_source_is_quint: bool,
) {
    if !matches!(output_format, OutputFormat::Json | OutputFormat::Jsonl) {
        return;
    }
    let Some(stats) = stats_mut_for_cli_report(result) else {
        return;
    };
    stats.backend_capability_report = Some(strict_shared_engine_report_json(
        stats.backend_capability_report.take(),
        frontend_source_is_quint,
        &[],
    ));
}

fn portfolio_common_json_fields(
    result: &tla_check::PortfolioResult,
    elapsed: std::time::Duration,
    frontend_source_is_quint: bool,
) -> serde_json::Map<String, serde_json::Value> {
    let mut fields = serde_json::Map::new();
    fields.insert("mode".to_string(), serde_json::json!("portfolio"));
    fields.insert(
        "winner".to_string(),
        serde_json::json!(format!("{:?}", result.winner)),
    );
    fields.insert(
        "frontend_source".to_string(),
        serde_json::json!(portfolio_frontend_source_code(frontend_source_is_quint)),
    );
    fields.insert(
        "analytical_eligibility".to_string(),
        serde_json::json!(result.analytical_eligibility.code()),
    );
    fields.insert(
        "analytical_solve_evidence".to_string(),
        serde_json::json!(result.analytical_solve_evidence),
    );
    fields.insert(
        "shared_engine_validation_receipts".to_string(),
        serde_json::json!(result.shared_engine_validation_receipts),
    );
    fields.insert(
        "time_secs".to_string(),
        serde_json::json!(elapsed.as_secs_f64()),
    );
    fields.insert(
        "backend_capability_report".to_string(),
        strict_shared_engine_report_json(
            result.bfs_result.stats().backend_capability_report.clone(),
            frontend_source_is_quint,
            &result.shared_engine_validation_receipts,
        ),
    );
    #[cfg(feature = "ay")]
    {
        fields.insert(
            "pdr_proof_replay_evidence".to_string(),
            serde_json::json!(result.pdr_proof_replay_evidence),
        );
        fields.insert(
            "ay_shared_engine_evidence".to_string(),
            serde_json::json!(result.ay_shared_engine_evidence),
        );
    }
    fields
}

fn portfolio_success_json_value(
    result: &tla_check::PortfolioResult,
    stats: &tla_check::CheckStats,
    elapsed: std::time::Duration,
    frontend_source_is_quint: bool,
) -> serde_json::Value {
    let mut fields = portfolio_common_json_fields(result, elapsed, frontend_source_is_quint);
    fields.insert("result".to_string(), serde_json::json!("success"));
    fields.insert(
        "states_found".to_string(),
        serde_json::json!(stats.states_found),
    );
    serde_json::Value::Object(fields)
}

fn portfolio_invariant_violation_json_value(
    result: &tla_check::PortfolioResult,
    invariant: &str,
    elapsed: std::time::Duration,
    frontend_source_is_quint: bool,
) -> serde_json::Value {
    let mut fields = portfolio_common_json_fields(result, elapsed, frontend_source_is_quint);
    fields.insert(
        "result".to_string(),
        serde_json::json!("invariant_violation"),
    );
    fields.insert("invariant".to_string(), serde_json::json!(invariant));
    serde_json::Value::Object(fields)
}

fn portfolio_unexpected_json_value(
    result: &tla_check::PortfolioResult,
    other: &tla_check::CheckResult,
    elapsed: std::time::Duration,
    frontend_source_is_quint: bool,
) -> serde_json::Value {
    let mut fields = portfolio_common_json_fields(result, elapsed, frontend_source_is_quint);
    fields.insert(
        "result".to_string(),
        serde_json::json!(format!("{other:?}")),
    );
    serde_json::Value::Object(fields)
}

/// Run portfolio racing mode while preserving source-family provenance in
/// analytical/AY evidence rows. The solver lanes still consume the lowered TLA
/// AST; this only affects descriptor identity.
pub(super) fn run_portfolio_mode_with_frontend_source(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    strategy_names: &[String],
    output_format: OutputFormat,
    frontend_source_is_quint: bool,
) -> Result<()> {
    use tla_check::{PortfolioResult, PortfolioWinner};

    if matches!(output_format, OutputFormat::Human) {
        if strategy_names.is_empty() {
            println!("Portfolio mode: racing all available strategies");
        } else {
            println!(
                "Portfolio mode: racing strategies: {}",
                strategy_names.join(", ")
            );
        }
        if !strategy_names.is_empty() {
            let unsupported: Vec<_> = strategy_names
                .iter()
                .filter(|s| {
                    !matches!(
                        s.as_str(),
                        "bfs" | "random" | "bmc" | "pdr" | "kinduction" | "analytical"
                    )
                })
                .collect();
            if !unsupported.is_empty() {
                eprintln!(
                    "Warning: unknown strategies ignored: {}",
                    unsupported
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
        println!();
    }

    let start = Instant::now();
    let result = PortfolioResult::run_with_frontend_source(
        module,
        checker_modules,
        config,
        strategy_names,
        frontend_source_is_quint,
    );
    // Graceful flat-storage value-overflow handling, mirroring the sequential
    // path in cmd_check: when the BFS lane aborted because the flat i64 state
    // layout cannot represent a value the spec produced, transparently re-run
    // the whole portfolio ONCE with flat state storage disabled
    // (`Config::use_flat_state = Some(false)`). Single-shot by construction:
    // if the retried result somehow still carries the flat-overflow error, it
    // falls through to the result handler below and is reported as a real
    // error (no second retry).
    let result = match super::flat_layout_unsupported_detail(&result.bfs_result) {
        None => result,
        Some(detail) => {
            eprintln!(
                "note: flat state layout cannot represent a value produced by this spec \
                 ({detail}); re-running portfolio with flat state storage disabled"
            );
            let mut retry_config = config.clone();
            retry_config.use_flat_state = Some(false);
            PortfolioResult::run_with_frontend_source(
                module,
                checker_modules,
                &retry_config,
                strategy_names,
                frontend_source_is_quint,
            )
        }
    };
    let elapsed = start.elapsed();

    let winner_str = match result.winner {
        PortfolioWinner::Analytical => "Analytical proof",
        PortfolioWinner::Bfs => "BFS (explicit-state)",
        PortfolioWinner::Pdr => "PDR (symbolic safety proving)",
        PortfolioWinner::Bmc => "BMC (symbolic bug finding)",
        PortfolioWinner::KInduction => "k-Induction (symbolic proving)",
        PortfolioWinner::Random => "Random walk",
    };

    // Verdict-masking reconciliation (mirrors fused mode): a racing lane that
    // resolved the Violated verdict truncates BFS into a clean-looking result;
    // reporting that bfs_result would print "No error has been found" while a
    // lane holds a real counterexample. Promote a validated counterexample to
    // the violation it is; fail closed to an inconclusive verdict when the
    // race-winning counterexample could not be re-validated.
    match result.reconcile_masked_violation(module, config) {
        tla_check::ReconciledVerdict::SymbolicViolation {
            lane,
            detail,
            invariant,
            trace,
        } => {
            if matches!(output_format, OutputFormat::Human) {
                if let Some(ref inv) = invariant {
                    eprintln!("Error: Invariant {inv} is violated.");
                    eprintln!(
                        "  Counterexample found by the {lane} lane (the BFS lane did not \
                         reach it)."
                    );
                } else {
                    eprintln!(
                        "Error: violation found by the {lane} lane (the BFS lane did not \
                         reach it)."
                    );
                }
                eprintln!("  {detail}");
                if !trace.is_empty() {
                    crate::check_report::emit_counterexample_trace(
                        &trace,
                        crate::cli_schema::TraceFormat::Text,
                        false,
                        "Counterexample trace",
                    );
                }
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let mut fields =
                    portfolio_common_json_fields(&result, elapsed, frontend_source_is_quint);
                fields.insert(
                    "result".to_string(),
                    serde_json::json!("invariant_violation"),
                );
                if let Some(ref inv) = invariant {
                    fields.insert("invariant".to_string(), serde_json::json!(inv));
                }
                fields.insert("masked_violation_lane".to_string(), serde_json::json!(lane));
                fields.insert("detail".to_string(), serde_json::json!(detail));
                let value = serde_json::Value::Object(fields);
                println!("{}", render_structured_json_value(output_format, &value)?);
            }
            std::process::exit(1);
        }
        tla_check::ReconciledVerdict::UnvalidatedSymbolicViolation { lane, detail } => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!(
                    "Error: inconclusive verdict — the {lane} lane won the race with a \
                     counterexample that could not be re-validated by the explicit-state \
                     evaluator, and the BFS lane was cut short by that race win."
                );
                eprintln!("  {detail}");
                eprintln!(
                    "  Fail-closed: refusing to report \"no error\". \
                     Re-run with --bfs-only for an authoritative explicit-state result."
                );
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let mut fields =
                    portfolio_common_json_fields(&result, elapsed, frontend_source_is_quint);
                fields.insert(
                    "result".to_string(),
                    serde_json::json!("inconclusive_symbolic_race"),
                );
                fields.insert("masked_violation_lane".to_string(), serde_json::json!(lane));
                fields.insert("detail".to_string(), serde_json::json!(detail));
                let value = serde_json::Value::Object(fields);
                println!("{}", render_structured_json_value(output_format, &value)?);
            }
            std::process::exit(2);
        }
        tla_check::ReconciledVerdict::FromBfs => {}
    }

    match &result.bfs_result {
        tla_check::CheckResult::Success(stats) => {
            if matches!(output_format, OutputFormat::Human) {
                println!("Model checking complete. No error has been found.");
                println!("  {} states found.", stats.states_found);
                println!("  Resolved by: {winner_str}");
                println!();
                println!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let value =
                    portfolio_success_json_value(&result, stats, elapsed, frontend_source_is_quint);
                println!("{}", render_structured_json_value(output_format, &value)?);
            }
            Ok(())
        }
        tla_check::CheckResult::InvariantViolation { invariant, .. } => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("Error: Invariant {invariant} is violated.");
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let value = portfolio_invariant_violation_json_value(
                    &result,
                    invariant,
                    elapsed,
                    frontend_source_is_quint,
                );
                println!("{}", render_structured_json_value(output_format, &value)?);
            }
            std::process::exit(1);
        }
        other => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("Portfolio result: {other:?}");
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                let value = portfolio_unexpected_json_value(
                    &result,
                    other,
                    elapsed,
                    frontend_source_is_quint,
                );
                println!("{}", render_structured_json_value(output_format, &value)?);
            }
            bail!("Portfolio mode produced unexpected result: {other:?}");
        }
    }
}

fn portfolio_frontend_source_is_quint_from_cli_invocation() -> bool {
    std::env::args_os()
        .any(|arg| arg == "--quint" || tla_core::quint::is_quint_json_path(Path::new(&arg)))
}

/// Run fused cooperative BFS+symbolic verification (Part of #3770, Epic #3762).
///
/// Spawns BFS, BMC, and PDR lanes cooperatively via `FusedOrchestrator`.
/// BFS frontier states seed BMC; PDR proofs skip BFS invariant checks.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_fused_mode(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    output_format: OutputFormat,
    spec_file: &Path,
    config_file: Option<&Path>,
    workers: usize,
    completeness: SearchCompleteness,
    soundness: &SoundnessProvenance,
    trace_format: crate::cli_schema::TraceFormat,
    extended_modules: &[&tla_core::ast::Module],
    allow_vacuous: &[String],
    strict_vacuity: bool,
    checker_config: tla_check::FusedCheckerConfig,
    mut tool_out: Option<tlc_tool::TlcToolOutput>,
) -> Result<()> {
    use tla_check::{run_fused_check_with_config, FusedWinner};

    if matches!(output_format, OutputFormat::Human) {
        println!("Fused mode: cooperative BFS + symbolic verification (CDEMC)");
        println!();
    }

    let start = Instant::now();
    // Retry material, captured before `checker_config` is moved into the run:
    // the graceful flat-overflow retry below rebuilds a fresh
    // FusedCheckerConfig from these (with FRESH fingerprint storage — never
    // the storage the aborted flat run may have partially populated).
    let retry_file_paths = checker_config.file_paths.clone();
    // FileId -> path map for action-label attribution when a symbolic lane's
    // confirmed counterexample is promoted below (kept out of the retry
    // material, which is moved into the retry FusedCheckerConfig).
    let label_file_paths = checker_config.file_paths.clone();
    let retry_fp_template = checker_config.fingerprint_storage.clone();
    let retry_max_states = checker_config.max_states;
    let retry_max_depth = checker_config.max_depth;
    let retry_memory_limit_bytes = checker_config.memory_limit_bytes;
    let retry_disk_limit_bytes = checker_config.disk_limit_bytes;
    let retry_continue_on_error = checker_config.continue_on_error;
    let retry_store_states = checker_config.store_states;
    let mut result =
        run_fused_check_with_config(module, checker_modules, config, checker_config);

    // Graceful flat-storage value-overflow handling, mirroring the sequential
    // path in cmd_check and the portfolio route above: when the fused BFS lane
    // aborted because the flat i64 state layout cannot represent a value the
    // spec produced, transparently re-run the fused check ONCE with flat state
    // storage disabled (`Config::use_flat_state = Some(false)`). Single-shot
    // by construction: if the retried result somehow still carries the
    // flat-overflow error, it falls through to the `CheckResult::Error` arm
    // below and is reported as a real error (no second retry).
    if let Some(detail) = super::flat_layout_unsupported_detail(&result.bfs_result) {
        eprintln!(
            "note: flat state layout cannot represent a value produced by this spec \
             ({detail}); re-running with flat state storage disabled"
        );
        // Also mark the restart in the TLC tool-protocol stream (the retry
        // re-emits the init-phase lifecycle; the eprintln! note above is
        // stderr-only).
        if let Some(out) = tool_out.as_mut() {
            out.emit(
                tlc_codes::ec::GENERAL,
                tlc_codes::mp::NONE,
                &format!(
                    "note: flat state layout cannot represent a value produced by this \
                     spec ({detail}); re-running with flat state storage disabled\n"
                ),
            );
        }
        // Fresh storage for the fresh run — never reuse fingerprint state the
        // aborted flat run may have partially populated.
        let retry_fingerprint_storage = match retry_fp_template {
            None => None,
            Some(template) => Some(template.fresh_empty_clone().map_err(|e| {
                anyhow::anyhow!(
                    "failed to create fresh fingerprint storage for the flat-overflow \
                     retry: {e}"
                )
            })?),
        };
        let retry_checker_config = tla_check::FusedCheckerConfig {
            file_paths: retry_file_paths,
            fingerprint_storage: retry_fingerprint_storage,
            max_states: retry_max_states,
            max_depth: retry_max_depth,
            memory_limit_bytes: retry_memory_limit_bytes,
            disk_limit_bytes: retry_disk_limit_bytes,
            continue_on_error: retry_continue_on_error,
            store_states: retry_store_states,
        };
        let mut retry_config = config.clone();
        retry_config.use_flat_state = Some(false);
        result = run_fused_check_with_config(
            module,
            checker_modules,
            &retry_config,
            retry_checker_config,
        );
    }

    let winner_str = match result.winner {
        FusedWinner::Bfs => "BFS (explicit-state)",
        FusedWinner::Bmc => "BMC (symbolic bug finding)",
        FusedWinner::Pdr => "PDR (symbolic safety proving)",
        FusedWinner::KInduction => "k-Induction (inductive safety)",
    };

    // Part of #4 (verdict-masking, fail-closed): before trusting `bfs_result`,
    // reconcile it against the symbolic lanes. A symbolic lane that resolves
    // the `Violated` race truncates BFS into a clean-looking result; reporting
    // that `bfs_result` silently drops a found bug ("No error has been found.
    // Resolved by: k-Induction" on a really-violated spec — the k-Induction
    // verdict-masking bug). A violation is REPORTED only when the explicit
    // evaluator confirmed the counterexample by FULL Init/Next trace replay
    // (so it can never produce a false alarm); a symbolic Violated race win
    // whose counterexample could NOT be confirmed fails closed to an
    // inconclusive verdict — never "No error", never an unvalidated violation.
    //
    // Reconciliation runs BEFORE the ALIAS/vacuity transforms: a CONFIRMED
    // symbolic counterexample is substituted into `bfs_result` as a standard
    // `CheckResult::InvariantViolation`, so the whole normal violation pipeline
    // applies — TLC-parity "Error: Invariant X is violated." output, the ALIAS
    // transform, `--trace-format` (e.g. ITF) rendering, storage-stats
    // reporting, and the JSON `counterexample` field `ty verdict-emit` needs.
    match result.reconcile_masked_violation() {
        tla_check::ReconciledVerdict::SymbolicViolation {
            lane,
            detail,
            invariant: Some(invariant),
            mut trace,
        } => {
            // Provenance note (stderr): which lane found it and that the
            // explicit-state evaluator re-derived it.
            eprintln!(
                "note: counterexample found by the {lane} lane and confirmed by the \
                 explicit-state evaluator via full Init/Next trace replay \
                 (the BFS lane was cut short by the race win): {detail}"
            );
            // TLC-parity action labels (#2470/#2920): the replay-validated
            // trace carries bare states; attribute each transition to its
            // action so the report shows `<Name line N, col N to ... of
            // module M>` exactly like a BFS-lane counterexample (best-effort —
            // failures keep the `<Action>` placeholders).
            tla_check::label_trace_actions(
                module,
                extended_modules,
                config,
                &label_file_paths,
                &mut trace,
            );
            let stats = result.bfs_result.stats().clone();
            result.bfs_result = tla_check::CheckResult::InvariantViolation {
                invariant,
                trace,
                stats,
            };
        }
        // Fused lanes only promote invariant counterexamples; a confirmed
        // violation WITHOUT an invariant name cannot be routed through the
        // standard pipeline — treat like an unvalidated race win (defensive;
        // reconcile_masked_violation never constructs this shape for fused).
        tla_check::ReconciledVerdict::SymbolicViolation {
            lane,
            detail,
            invariant: None,
            ..
        }
        | tla_check::ReconciledVerdict::UnvalidatedSymbolicViolation { lane, detail } => {
            // FAIL CLOSED: the {lane} lane won the Violated race (truncating the
            // BFS lane), but its counterexample could not be re-validated by the
            // explicit-state evaluator. The truncated BFS result proves nothing,
            // so neither "No error" nor a violation may be reported.
            let elapsed = start.elapsed();
            if matches!(output_format, OutputFormat::Human) {
                eprintln!(
                    "Error: inconclusive verdict — the {lane} lane reported a counterexample \
                     that the explicit-state evaluator could not confirm, and the BFS lane \
                     was cut short by that race win (its clean result is not authoritative)."
                );
                eprintln!("  {detail}");
                eprintln!(
                    "  Fail-closed: refusing to report \"no error\". \
                     Re-run with --bfs-only for an authoritative explicit-state result."
                );
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                eprintln!(
                    "[fused] winner={winner_str}, unvalidated symbolic violation from {lane}: \
                     {detail}"
                );
                let module_name = &module.name.node;
                let json_output = JsonOutput::new(spec_file, config_file, module_name, workers)
                    .with_completeness(completeness)
                    .with_soundness(soundness.clone())
                    .with_unvalidated_symbolic_race(lane, &detail, &result.bfs_result, elapsed);
                let json_str = if matches!(output_format, OutputFormat::Jsonl) {
                    json_output.to_json_compact()
                } else {
                    json_output.to_json()
                };
                match json_str {
                    Ok(s) => println!("{s}"),
                    Err(e) => eprintln!("error: failed to serialize JSON output: {e}"),
                }
            }
            std::process::exit(2);
        }
        tla_check::ReconciledVerdict::FromBfs => {}
    }

    // Apply the ALIAS transform to the BFS result, mirroring the explicit-state
    // path — otherwise a fused-mode counterexample renders the raw state variables
    // instead of the configured `ALIAS` view. No-op when no ALIAS is configured.
    result.bfs_result = super::helpers::apply_alias_transform(
        std::mem::replace(
            &mut result.bfs_result,
            tla_check::CheckResult::Success(tla_check::CheckStats::default()),
        ),
        config,
        module,
        checker_modules,
        extended_modules,
    );
    // Apply the vacuity gate policy (--allow-vacuous / --strict-vacuity) to the BFS
    // result, mirroring the explicit-state path. This downgrades a named VACUOUS
    // class to a WARNING or promotes default-on V2/V3 dead-action / vacuously-true
    // WARNINGs to the hard VACUOUS verdict (exit 3). Without this, fused mode
    // ignored the vacuity flags entirely.
    let vacuity_policy =
        super::vacuity_policy::VacuityPolicy::parse(allow_vacuous, strict_vacuity)?;
    let vacuity_outcome = vacuity_policy.apply(std::mem::replace(
        &mut result.bfs_result,
        tla_check::CheckResult::Success(tla_check::CheckStats::default()),
    ));
    result.bfs_result = vacuity_outcome.result;
    if matches!(output_format, OutputFormat::Human) {
        for line in &vacuity_outcome.lines {
            eprintln!("{line}");
        }
    }
    let elapsed = start.elapsed();

    // TLC-tool output: emit the in-run progress messages (the preamble — version /
    // mode / starting — was already emitted by setup_tlc_tool_output) and the final
    // TLC-format verdict via the shared reporter, mirroring the explicit path. The
    // fused arms below only know Human vs JSON, so handle tlc-tool here and return.
    if matches!(output_format, OutputFormat::TlcTool) {
        let mut out = tool_out;
        if let Some(ref mut o) = out {
            o.emit(
                tlc_codes::ec::TLC_COMPUTING_INIT,
                tlc_codes::mp::NONE,
                tlc_tool::format_tlc_computing_init_message(),
            );
            let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
            o.emit(
                tlc_codes::ec::TLC_INIT_GENERATED1,
                tlc_codes::mp::NONE,
                &tlc_tool::format_tlc_init_generated1_message(
                    result.bfs_result.stats().initial_states as u64,
                    &now,
                ),
            );
        }
        return crate::check_report::report_check_tlc_tool(out, &result.bfs_result, elapsed);
    }

    // Report graceful degradation when symbolic lanes fail (Part of #3837).
    let degradation_summary = result.degradation.summary();
    let symbolic_coverage = result.symbolic_coverage;
    let lane_coverage = result.degradation.symbolic_coverage();

    // Part of #3837: Build a single-line user-friendly degradation info message.
    // Shows per-action symbolic coverage and lists unsupported action names.
    let degradation_info: Option<String> = if result.degradation.any_degraded()
        || !result.degradation.unsupported_action_names.is_empty()
    {
        let pct = (symbolic_coverage * 100.0) as u32;
        let compat = result.degradation.actions_smt_compatible;
        let total = result.degradation.actions_total;
        let unsupported = &result.degradation.unsupported_action_names;
        if unsupported.is_empty() {
            Some(format!(
                "Symbolic coverage: {pct}% ({compat}/{total} actions translatable)"
            ))
        } else {
            Some(format!(
                "Symbolic coverage: {pct}% ({compat}/{total} actions translatable, unsupported: {})",
                unsupported.join(", ")
            ))
        }
    } else {
        None
    };

    // Part of #3805: For JSON output, use the standard JsonOutput format so that
    // `ty diagnose` can parse the output. Previously the fused-mode raw JSON blob
    // was printed to stdout, breaking the diagnostic parser which expects a single
    // JSON object in SubprocessOutput format (result.status, statistics.states_found).
    // For Human output, the inline debug info is preserved unchanged.
    match &result.bfs_result {
        tla_check::CheckResult::Success(stats) => {
            if matches!(output_format, OutputFormat::Human) {
                println!("Model checking complete. No error has been found.");
                println!();
                println!("Statistics:");
                println!("  States found: {}", stats.states_found);
                println!("  Initial states: {}", stats.initial_states);
                println!("  Transitions: {}", stats.transitions);
                println!("  Max queue depth: {}", stats.max_queue_depth);
                if let Some(ref summary) = result.symbolic_summary {
                    println!("  {summary}");
                }
                if let Some(ref info) = degradation_info {
                    println!("  {info}");
                }
                println!("  Resolved by: {winner_str}");
                crate::check_report::print_storage_stats(&stats.storage_stats, false);
                println!();
                println!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                // Emit fused metadata to stderr for debugging.
                eprintln!(
                    "[fused] winner={winner_str}, coverage={symbolic_coverage:.0}%, lane={lane_coverage:.2}"
                );
                if let Some(ref deg) = degradation_summary {
                    eprintln!("[fused] degradation: {deg}");
                }
                // Emit standard JsonOutput to stdout.
                let module_name = &module.name.node;
                let json_output = JsonOutput::new(spec_file, config_file, module_name, workers)
                    .with_completeness(completeness)
                    .with_soundness(soundness.clone())
                    .with_check_result(&result.bfs_result, elapsed);
                let json_str = if matches!(output_format, OutputFormat::Jsonl) {
                    json_output.to_json_compact()
                } else {
                    json_output.to_json()
                };
                match json_str {
                    Ok(s) => println!("{s}"),
                    Err(e) => eprintln!("error: failed to serialize JSON output: {e}"),
                }
            }
            Ok(())
        }
        tla_check::CheckResult::InvariantViolation {
            invariant,
            trace,
            stats,
        } => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("Error: Invariant {invariant} is violated.");
                // Print the counterexample trace, like the explicit-state path —
                // fused mode previously reported the violation with NO trace, so a
                // default `ty check` gave the user no counterexample to inspect.
                crate::check_report::emit_counterexample_trace(
                    trace,
                    trace_format,
                    false,
                    "Counterexample trace",
                );
                if let Some(ref info) = degradation_info {
                    eprintln!("  {info}");
                }
                eprintln!("  Resolved by: {winner_str}");
                // Storage backend stats (mmap/disk), to stderr like the verdict.
                crate::check_report::print_storage_stats(&stats.storage_stats, true);
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                eprintln!("[fused] winner={winner_str}, invariant={invariant}");
                // Emit standard JsonOutput to stdout.
                let module_name = &module.name.node;
                let json_output = JsonOutput::new(spec_file, config_file, module_name, workers)
                    .with_completeness(completeness)
                    .with_soundness(soundness.clone())
                    .with_check_result(&result.bfs_result, elapsed);
                let json_str = if matches!(output_format, OutputFormat::Jsonl) {
                    json_output.to_json_compact()
                } else {
                    json_output.to_json()
                };
                match json_str {
                    Ok(s) => println!("{s}"),
                    Err(e) => eprintln!("error: failed to serialize JSON output: {e}"),
                }
            }
            std::process::exit(1);
        }
        tla_check::CheckResult::Deadlock { trace, .. } => {
            // A reachable deadlock state (a state with no Next successor) is a
            // property failure, exactly like the explicit-BFS/BMC deadlock paths
            // (and the InvariantViolation arm above). Render it cleanly and exit
            // nonzero rather than letting it fall into the catch-all, which
            // dumped the raw Debug and bailed with "unexpected result". This is
            // what "works correctly by default" requires: deadlock-checking is on
            // by default, and a reached deadlock must be a clean verdict.
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("Error: Deadlock reached (a reachable state has no successors).");
                crate::check_report::emit_counterexample_trace(
                    trace,
                    trace_format,
                    false,
                    "Deadlock trace",
                );
                if let Some(ref info) = degradation_info {
                    eprintln!("  {info}");
                }
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                eprintln!("[fused] winner={winner_str}, deadlock");
                let module_name = &module.name.node;
                let json_output = JsonOutput::new(spec_file, config_file, module_name, workers)
                    .with_completeness(completeness)
                    .with_soundness(soundness.clone())
                    .with_check_result(&result.bfs_result, elapsed);
                let json_str = if matches!(output_format, OutputFormat::Jsonl) {
                    json_output.to_json_compact()
                } else {
                    json_output.to_json()
                };
                match json_str {
                    Ok(s) => println!("{s}"),
                    Err(e) => eprintln!("error: failed to serialize JSON output: {e}"),
                }
            }
            std::process::exit(1);
        }
        tla_check::CheckResult::LimitReached { limit_type, stats } => {
            // An exploration-bound stop (--max-states / --max-depth / memory /
            // disk) is a clean exit-0 outcome with statistics, like the
            // explicit-state path — not the catch-all's raw Debug + exit 2.
            if matches!(output_format, OutputFormat::Human) {
                let (limit_name, hint) = match limit_type {
                    tla_check::LimitType::States => {
                        ("state", "Use --max-states or --max-depth to adjust limits")
                    }
                    tla_check::LimitType::Depth => {
                        ("depth", "Use --max-states or --max-depth to adjust limits")
                    }
                    tla_check::LimitType::Memory => {
                        ("memory", "Use --memory-limit to adjust the threshold")
                    }
                    tla_check::LimitType::Disk => {
                        ("disk", "Use --disk-limit to adjust the threshold")
                    }
                    tla_check::LimitType::Exit => (
                        "exit (TLCSet)",
                        "Spec requested early termination via TLCSet(\"exit\", TRUE)",
                    ),
                    _ => ("unknown", "Exploration limit reached"),
                };
                println!("Model checking stopped: {limit_name} limit reached.");
                println!();
                println!("Statistics:");
                println!("  States found: {}", stats.states_found);
                println!("  Initial states: {}", stats.initial_states);
                println!("  Transitions: {}", stats.transitions);
                println!("  Max depth: {}", stats.max_depth);
                println!("  Resolved by: {winner_str}");
                crate::check_report::print_storage_stats(&stats.storage_stats, false);
                println!();
                println!("Time: {:.3}s", elapsed.as_secs_f64());
                println!();
                println!("Hint: {hint}");
            } else {
                let module_name = &module.name.node;
                let json_output = JsonOutput::new(spec_file, config_file, module_name, workers)
                    .with_completeness(completeness)
                    .with_soundness(soundness.clone())
                    .with_check_result(&result.bfs_result, elapsed);
                let json_str = if matches!(output_format, OutputFormat::Jsonl) {
                    json_output.to_json_compact()
                } else {
                    json_output.to_json()
                };
                match json_str {
                    Ok(s) => println!("{s}"),
                    Err(e) => eprintln!("error: failed to serialize JSON output: {e}"),
                }
            }
            Ok(())
        }
        tla_check::CheckResult::Vacuous { reason, stats } => {
            // Vacuity gate (TRUST_VACUITY_GATE §1.A): a vacuous result (e.g. an
            // empty initial set) must surface as a distinct VACUOUS verdict with
            // exit code 3 — exactly like the explicit-state path — instead of
            // falling into the catch-all, which dumped the raw Debug and exited 1
            // (a generic failure). Default `ty check` runs fused, so vacuity must
            // be reported here too, or a vacuous spec silently looks like a plain
            // error rather than "the model proved nothing".
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("VACUOUS: the model proved nothing.");
                eprintln!("  Reason: {}", reason.message());
                eprintln!(
                    "  Class: {} (relax with --allow-vacuous={})",
                    reason.class().as_str(),
                    reason.class().as_str()
                );
                if let Some(ref info) = degradation_info {
                    eprintln!("  {info}");
                }
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Statistics:");
                eprintln!("  States found: {}", stats.states_found);
                eprintln!("  Time: {:.3}s", elapsed.as_secs_f64());
                std::process::exit(3);
            }
            let module_name = &module.name.node;
            let json_output = JsonOutput::new(spec_file, config_file, module_name, workers)
                .with_completeness(completeness)
                .with_soundness(soundness.clone())
                .with_check_result(&result.bfs_result, elapsed);
            let json_str = if matches!(output_format, OutputFormat::Jsonl) {
                json_output.to_json_compact()
            } else {
                json_output.to_json()
            };
            match json_str {
                Ok(s) => println!("{s}"),
                Err(e) => eprintln!("error: failed to serialize JSON output: {e}"),
            }
            std::process::exit(3);
        }
        tla_check::CheckResult::Error { error, .. } => {
            // Render the error's friendly Display (e.g. "TLC cannot handle the
            // temporal formula bytes .. of module ..") like the explicit path,
            // instead of dumping the raw Debug of the whole result.
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("Error: {error}");
                if let Some(ref info) = degradation_info {
                    eprintln!("  {info}");
                }
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                bail!("Model checking failed: {error}");
            }
            let module_name = &module.name.node;
            let json_output = JsonOutput::new(spec_file, config_file, module_name, workers)
                .with_completeness(completeness)
                .with_soundness(soundness.clone())
                .with_check_result(&result.bfs_result, elapsed);
            let json_str = if matches!(output_format, OutputFormat::Jsonl) {
                json_output.to_json_compact()
            } else {
                json_output.to_json()
            };
            match json_str {
                Ok(s) => println!("{s}"),
                Err(e) => eprintln!("error: failed to serialize JSON output: {e}"),
            }
            std::process::exit(2);
        }
        other => {
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("Fused result: {other:?}");
                if let Some(ref info) = degradation_info {
                    eprintln!("  {info}");
                }
                eprintln!("  Resolved by: {winner_str}");
                eprintln!();
                eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
            } else {
                // Emit standard JsonOutput to stdout. The caller's error handler
                // (emit_check_cli_error) would produce a generic error, but we have
                // the actual CheckResult with state counts here, so emit it properly.
                let module_name = &module.name.node;
                let json_output = JsonOutput::new(spec_file, config_file, module_name, workers)
                    .with_completeness(completeness)
                    .with_soundness(soundness.clone())
                    .with_check_result(other, elapsed);
                let json_str = if matches!(output_format, OutputFormat::Jsonl) {
                    json_output.to_json_compact()
                } else {
                    json_output.to_json()
                };
                match json_str {
                    Ok(s) => println!("{s}"),
                    Err(e) => eprintln!("error: failed to serialize JSON output: {e}"),
                }
                // Exit directly — do NOT bail, which would cause the caller's error
                // handler to emit a second JSON object to stdout.
                std::process::exit(2);
            }
            bail!("Fused mode produced unexpected result: {other:?}");
        }
    }
}

/// Run multi-phase verification pipeline (Part of #3723).
///
/// Executes phases in order: RandomWalk(5s) -> BMC(30s) -> PDR(60s) -> BFS(300s).
/// Early-exits when all properties are resolved. BMC and PDR phases are only
/// available when the `ay` feature is enabled; otherwise they are silently skipped.
/// Run the multi-phase verification pipeline with a named strategy.
///
/// The strategy selects the phase configuration:
/// - Quick: RandomWalk(2s) + BMC(10s). Fast feedback for development.
/// - Full: RandomWalk(5s) + BFS(600s). Exhaustive fallback within the timeout.
/// - Auto: walk -> BMC -> k-induction -> PDR -> BFS (adaptive early exit).
///
/// Part of #3723.
pub(super) fn run_pipeline_mode_with_strategy(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    config: &tla_check::Config,
    strategy: tla_check::VerificationStrategy,
    output_format: OutputFormat,
) -> Result<()> {
    use rustc_hash::FxHashMap as HashMap;
    use tla_check::{
        BfsRunner, PhaseRunner, PropertyVerdict, RandomWalkConfig, RandomWalkRunner,
        VerificationPhase,
    };

    let pipeline = strategy.into_pipeline();

    if matches!(output_format, OutputFormat::Human) {
        eprintln!("Pipeline mode: strategy={strategy}");
        let phase_names: Vec<String> = pipeline
            .phases()
            .iter()
            .filter(|p| p.enabled)
            .map(|p| format!("{}({}s)", p.phase, p.time_budget.as_secs()))
            .collect();
        eprintln!("  Phases: {}", phase_names.join(" -> "));
        eprintln!();
    }

    let runtime_config = config.runtime_model_config();

    // Collect invariant names as property IDs.
    let properties: Vec<String> = config.invariants.clone();
    if properties.is_empty() {
        // No invariants — but a reachable deadlock is still a verifiable property
        // when deadlock-checking is on (the TLC default). The explicit and fused
        // modes report such a deadlock; the pipeline must not silently declare
        // "nothing to verify" and exit 0 (a missed-deadlock / silent-wrong-result
        // — the `--strategy full` masking surfaced by the cross-mode parity scan).
        // Run an exhaustive deadlock-aware BFS instead.
        if config.check_deadlock {
            let start = Instant::now();
            let mut checker =
                tla_check::ModelChecker::new_with_extends(module, checker_modules, &runtime_config);
            let result = checker.check();
            let elapsed = start.elapsed();
            match &result {
                tla_check::CheckResult::Success(stats) => {
                    if matches!(output_format, OutputFormat::Human) {
                        println!("Model checking complete. No error has been found.");
                        println!("  {} states found.", stats.states_found);
                        println!("  (deadlock-freedom verified; no invariants configured)");
                        println!();
                        println!("Time: {:.3}s", elapsed.as_secs_f64());
                    }
                    return Ok(());
                }
                tla_check::CheckResult::Deadlock { .. } => {
                    if matches!(output_format, OutputFormat::Human) {
                        eprintln!("Error: Deadlock reached (a reachable state has no successors).");
                        eprintln!();
                        eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                    } else {
                        eprintln!("[pipeline] deadlock reached");
                    }
                    std::process::exit(1);
                }
                other => {
                    // Any other terminal result (runtime error, limit reached):
                    // surface it and exit non-zero rather than masquerading as OK.
                    if matches!(output_format, OutputFormat::Human) {
                        eprintln!("Pipeline deadlock-check result: {other:?}");
                        eprintln!();
                        eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
                    } else {
                        eprintln!("[pipeline] non-success result: {other:?}");
                    }
                    std::process::exit(2);
                }
            }
        }
        if matches!(output_format, OutputFormat::Human) {
            eprintln!(
                "Pipeline: no invariants configured and deadlock-checking is off, nothing to verify."
            );
        }
        return Ok(());
    }

    // Build runners for each phase.
    let mut runners: HashMap<VerificationPhase, Box<dyn PhaseRunner>> = HashMap::default();

    // RandomWalk runner
    let walk_checker =
        tla_check::ModelChecker::new_with_extends(module, checker_modules, &runtime_config);
    let walk_config = RandomWalkConfig {
        num_walks: 100,
        max_depth: 10_000,
        seed: None,
    };
    runners.insert(
        VerificationPhase::RandomWalk,
        Box::new(RandomWalkRunner::new(walk_checker, walk_config)),
    );

    // BMC/PDR/KInduction runners (ay feature gate)
    #[cfg(feature = "ay")]
    {
        let mut ctx = tla_check::EvalCtx::new();
        ctx.load_module(module);
        for m in checker_modules {
            ctx.load_module(m);
        }
        // EXTENDS-inherited VARIABLES (MC-wrapper specs) must be registered so the
        // BMC/PDR/KInduction pipeline phases see the real state vars.
        tla_check::register_state_vars_for_symbolic(&mut ctx, module, checker_modules);
        if let Err(e) = tla_check::bind_constants_from_config(&mut ctx, config) {
            eprintln!("Pipeline: failed to bind constants for BMC/PDR: {e}");
        } else {
            // Leak the EvalCtx to get a 'static reference that can be stored
            // in Box<dyn PhaseRunner>. This is acceptable because pipeline mode
            // runs once per invocation and the process exits afterward.
            let ctx: &'static tla_check::EvalCtx = Box::leak(Box::new(ctx));

            runners.insert(
                VerificationPhase::Bmc,
                Box::new(tla_check::BmcRunner::new(module, config, ctx, 20)),
            );
            runners.insert(
                VerificationPhase::Pdr,
                Box::new(tla_check::PdrRunner::new(module, config, ctx)),
            );
            runners.insert(
                VerificationPhase::KInduction,
                Box::new(tla_check::KInductionRunner::new(module, config, ctx, 20)),
            );
        }
    }

    // BFS runner (always available)
    runners.insert(
        VerificationPhase::Bfs,
        Box::new(BfsRunner::new(module, checker_modules, config)),
    );

    let start = Instant::now();
    let result = pipeline.run(&properties, &mut runners);
    let elapsed = start.elapsed();

    // A reachable deadlock is a global property failure that the phase runners
    // cannot express as a per-invariant verdict (they leave properties Unknown).
    // Without this the WITH-invariants pipeline silently dropped a reached
    // deadlock and exited 0. Report it (exit 1) when deadlock-checking is on,
    // matching the explicit/fused modes and the empty-invariants branch above.
    if result.deadlock_reached && config.check_deadlock {
        if matches!(output_format, OutputFormat::Human) {
            eprintln!();
            eprintln!("Error: Deadlock reached (a reachable state has no successors).");
            eprintln!();
            eprintln!(
                "Pipeline ({strategy}) complete in {:.3}s",
                elapsed.as_secs_f64()
            );
        } else {
            eprintln!("[pipeline] deadlock reached");
        }
        std::process::exit(1);
    }

    // Report results.
    if matches!(output_format, OutputFormat::Human) {
        eprintln!();
        eprintln!(
            "Pipeline ({strategy}) complete in {:.3}s",
            elapsed.as_secs_f64()
        );
        for record in &result.phases_run {
            eprintln!(
                "  {}: {:.3}s, {} properties resolved",
                record.phase,
                record.elapsed.as_secs_f64(),
                record.properties_resolved,
            );
        }
        eprintln!();
        let mut any_violated = false;
        for (prop, verdict) in &result.verdicts {
            let label = match verdict {
                PropertyVerdict::Satisfied => "SATISFIED",
                PropertyVerdict::Violated => {
                    any_violated = true;
                    "VIOLATED"
                }
                PropertyVerdict::Unknown => "UNKNOWN",
            };
            eprintln!("  {prop}: {label}");
        }
        if any_violated {
            bail!("Pipeline found invariant violation(s)");
        }
    } else {
        // JSON output
        let verdicts_json: serde_json::Value = result
            .verdicts
            .iter()
            .map(|(k, v)| {
                let label = match v {
                    PropertyVerdict::Satisfied => "satisfied",
                    PropertyVerdict::Violated => "violated",
                    PropertyVerdict::Unknown => "unknown",
                };
                (k.clone(), serde_json::Value::String(label.to_string()))
            })
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();

        let json = serde_json::json!({
            "mode": "pipeline",
            "strategy": strategy.to_string(),
            "time_secs": elapsed.as_secs_f64(),
            "phases_run": result.phases_run.len(),
            "verdicts": verdicts_json,
        });
        println!("{json}");

        if result
            .verdicts
            .values()
            .any(|v| *v == PropertyVerdict::Violated)
        {
            std::process::exit(1);
        }
    }

    Ok(())
}
