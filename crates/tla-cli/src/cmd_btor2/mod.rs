// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty btor2` subcommand: check BTOR2 hardware model checking benchmarks.
//!
//! Parses a BTOR2 file and runs the full HWMCC portfolio pipeline:
//! 1. Cone-of-influence (COI) reduction — eliminate irrelevant states/inputs
//! 2. Expression simplification — constant folding, identity elimination
//! 3. BMC preprocessing — try shallow bounded model checking first
//! 4. Full CHC solving — PDR/k-induction via ay-chc adaptive portfolio
//!
//! Output follows the HWMCC convention:
//!   - `sat` on stdout if the property is violated (bad state reachable)
//!   - `unsat` on stdout if the property holds (bad state unreachable)
//!   - `unknown` if the solver cannot determine the result

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Run the `ty btor2` subcommand.
///
/// Reads and parses a BTOR2 file, then runs the full HWMCC portfolio
/// strategy: COI reduction -> BMC preprocessing -> full CHC solving.
/// If `bitblast` is true (or auto-detected as eligible), narrow bitvector
/// benchmarks are bit-blasted to AIGER and solved via the IC3/PDR engine.
pub(crate) fn cmd_btor2(
    file: &Path,
    verbose: bool,
    witness_file: Option<&Path>,
    timeout_secs: Option<u64>,
    bitblast: bool,
    max_bv_width: u32,
    array_bmc: bool,
) -> Result<()> {
    let start = Instant::now();

    // Read the BTOR2 source.
    let source = std::fs::read_to_string(file)
        .with_context(|| format!("failed to read BTOR2 file: {}", file.display()))?;

    // Parse.
    let program =
        tla_btor2::parse_btor2(&source).map_err(|e| anyhow::anyhow!("BTOR2 parse error: {e}"))?;

    if verbose {
        eprintln!(
            "Parsed BTOR2: {} lines, {} state(s), {} input(s), {} bad property(ies), {} constraint(s)",
            program.lines.len(),
            program.num_states,
            program.num_inputs,
            program.bad_properties.len(),
            program.constraints.len(),
        );
    }

    if program.bad_properties.is_empty() {
        if verbose {
            eprintln!("No bad properties to check.");
        }
        println!("unsat");
        return Ok(());
    }

    // Opt-in, additive independent SAFE certifier (HWMCC Track 2, disjoint-trust
    // gate). Gated behind `TY_BTOR2_INDEPENDENT_CERT`; emits only a `c ...`
    // comment to STDERR and NEVER touches the stdout verdict below. For a
    // bounded-array net, it re-proves SAFE through a second, disjoint path
    // (bit-blast the arrays + LRAT-checked ay-sat over the invariant's one-step
    // VCs) that never enters ay-dpll's array theory.
    run_independent_cert_if_requested(&program, verbose, timeout_secs);

    // Engine selection. TY's mature bit-level IC3/PDR engine solves narrow-
    // bitvector nets that the word-level CHC path stalls on: bit-precise
    // hardware semantics blow up the LIA/CEGAR encoding (modular wraparound +
    // per-variable range constraints), whereas the SAT-based IC3 engine handles
    // them directly. So for any bit-blast-eligible net we try that engine first,
    // then fall back to the word-level CHC portfolio on `unknown` — we never
    // lose a result the CHC path would have found. An explicit `--bitblast`
    // forces the bit-level engine only (full budget, no fallback); the default
    // auto-routes, which is what the help text advertises.
    let orig_eligibility = tla_btor2::bitblast_eligible(&program, max_bv_width);
    // Large arrays are declined by 2^index expansion. Under an EXPLICIT
    // `--bitblast`, try Ackermann array elimination (equisatisfiable, validated)
    // and bit-blast the array-free net when IT is eligible — a large array with
    // few distinct accesses then reaches the bit-level engine instead of a silent
    // CHC fallback. Gated to `--bitblast` (opt-in) so AUTO mode's default routing,
    // and its ay-sat exposure, is unchanged; if elimination declines or the result
    // is still ineligible, we fall through to CHC exactly as before (fail-closed).
    let eliminated = if bitblast && orig_eligibility.is_err() {
        tla_btor2::array_elim::ackermann_eliminate(&program)
            .filter(|e| tla_btor2::bitblast_eligible(e, max_bv_width).is_ok())
    } else {
        None
    };
    let (bb_program, eligibility) = match &eliminated {
        Some(e) => {
            if verbose {
                eprintln!("Bit-blast: eliminated arrays via Ackermann reduction");
            }
            (e, tla_btor2::bitblast_eligible(e, max_bv_width))
        }
        None => (&program, orig_eligibility),
    };
    match &eligibility {
        Ok(max_w) if verbose => {
            eprintln!("Bit-blast: eligible (max bitvector width = {max_w} bits)");
        }
        Err(reason) if verbose && bitblast => {
            eprintln!("Bit-blast: not eligible ({reason}), falling back to CHC");
        }
        _ => {}
    }

    if eligibility.is_ok() {
        // Auto mode reserves half the budget for the CHC fallback; an explicit
        // `--bitblast` gets the full budget and returns whatever it finds.
        let bb_budget = if bitblast {
            timeout_secs
        } else {
            timeout_secs.map(|t| (t / 2).max(1))
        };
        let (bb_circuit, bb_results) = run_btor2_bitblast(bb_program, verbose, bb_budget)?;
        let all_definitive = !bb_results.is_empty()
            && bb_results
                .iter()
                .all(|r| !matches!(r, tla_aiger::AigerCheckResult::Unknown { .. }));
        if bitblast || all_definitive {
            // Additive: the `sat` verdict below is unchanged; when a witness is
            // requested and some property is violated, project the bit-level
            // counterexample back to a BTOR2 witness. Fail-closed — a net that
            // is SAFE (unsat) or whose CEX cannot be re-derived writes no file.
            if let Some(witness_path) = witness_file {
                write_bitblast_witness(
                    bb_program,
                    &bb_circuit,
                    &bb_results,
                    witness_path,
                    verbose,
                )?;
            }
            return print_aiger_verdicts(&bb_results, verbose, start);
        }
        if verbose {
            eprintln!("Bit-blast inconclusive — falling back to word-level CHC portfolio");
        }
    }

    // FLAG-GATED (default OFF — `--array-bmc` or TY_BTOR2_LAZY_ARRAY_BMC):
    // the lazy-array trace-unrolled BMC lane, only for bit-blast-INELIGIBLE
    // nets with an array state (wide-index memories — exactly the class that
    // otherwise goes straight to the CHC portfolio). Runs as a bounded leading
    // time slice (the GPU-lane pattern) so it can never starve the CHC lane.
    // Its SAT verdicts are replay-validated inside the lane; anything else
    // (bounded no-cex, decline) falls through to the portfolio UNCHANGED, so
    // with the flag unset — or on any non-SAT outcome — the decision tree
    // below is byte-identical to today.
    if eligibility.is_err() && array_bmc_lane_requested(array_bmc) && has_array_state(&program) {
        if run_array_bmc_lane(&program, verbose, timeout_secs, witness_file, start)? {
            return Ok(());
        }
    }

    // FLAG-GATED (default OFF — TY_BTOR2_ARRAY_KIND): k-induction over the
    // lazy-array core, same shape gate and bounded-leading-slice budget as
    // the BMC lane. Returns into the pipeline ONLY on (a) a replay-validated
    // counterexample covering every bad property, or (b) an unbounded-safe
    // proof whose base and step queries were independently re-discharged
    // through the LRAT-checked second trust path inside the lane. Everything
    // else falls through to the portfolio byte-identical to today.
    if eligibility.is_err() && array_kind_lane_requested() && has_array_state(&program) {
        if run_array_kind_lane(&program, verbose, timeout_secs, witness_file, start)? {
            return Ok(());
        }
    }

    // FLAG-GATED (default OFF — TY_BTOR2_ARRAY_IC3): IC3/PDR frames over the
    // lazy-array core, same shape gate and bounded-leading-slice budget.
    // Returns into the pipeline ONLY on (a) a replay-validated counterexample
    // covering every bad property, or (b) an unbounded-safe proof whose
    // serialized frame invariant passed the INDEPENDENT LRAT-checked triple
    // validation (Tier A flattened / Tier B eager one-step) inside the lane.
    // Everything else falls through to the portfolio byte-identical to today.
    if eligibility.is_err() && array_ic3_lane_requested() && has_array_state(&program) {
        if run_array_ic3_lane(&program, verbose, timeout_secs, witness_file, start)? {
            return Ok(());
        }
    }

    // Run the full portfolio strategy (COI + BMC + CHC) with the budget that
    // remains after any bit-blast attempt above.
    let chc_budget = timeout_secs.map(|t| t.saturating_sub(start.elapsed().as_secs()).max(1));
    let portfolio_config = tla_btor2::PortfolioConfig {
        time_budget: chc_budget.map(Duration::from_secs),
        enable_coi: true,
        enable_simplify: true,
        enable_bmc: true,
        bmc_budget_fraction: 0.2,
        bmc_max_depth: 20,
        verbose,
    };

    let (results, stats) = tla_btor2::check_btor2_portfolio(&program, &portfolio_config)
        .map_err(|e| anyhow::anyhow!("BTOR2 portfolio error: {e}"))?;

    if verbose {
        eprintln!(
            "Portfolio stats: COI {}/{} states ({}/{} inputs), phase={:?}",
            stats.states_after_coi,
            stats.states_before_coi,
            stats.inputs_after_coi,
            stats.inputs_before_coi,
            stats.result_phase,
        );
        eprintln!(
            "  COI: {:.3}s, BMC: {:.3}s, CHC: {:.3}s, Total: {:.3}s",
            stats.coi_time.as_secs_f64(),
            stats.bmc_time.as_secs_f64(),
            stats.chc_time.as_secs_f64(),
            stats.total_time.as_secs_f64(),
        );
    }

    let elapsed = start.elapsed();
    let mut any_sat = false;
    let mut any_unknown = false;

    for (idx, result) in results.iter().enumerate() {
        match result {
            tla_btor2::Btor2CheckResult::Sat { trace, .. } => {
                any_sat = true;
                if verbose {
                    eprintln!(
                        "Property {idx}: VIOLATED (counterexample with {} step(s))",
                        trace.len()
                    );
                    for (step_idx, step) in trace.iter().enumerate() {
                        let mut assignments: Vec<_> = step.iter().collect();
                        assignments.sort_by(|a, b| a.0.cmp(b.0));
                        eprintln!("  Step {step_idx}:");
                        for (name, val) in &assignments {
                            eprintln!("    {name} = {val}");
                        }
                    }
                }
            }
            tla_btor2::Btor2CheckResult::Unsat => {
                if verbose {
                    eprintln!("Property {idx}: HOLDS");
                }
            }
            tla_btor2::Btor2CheckResult::Unknown { reason } => {
                any_unknown = true;
                if verbose {
                    eprintln!("Property {idx}: UNKNOWN ({reason})");
                }
            }
        }
    }

    // Print HWMCC result to stdout.
    // For multi-property benchmarks, emit one verdict line per property so
    // that each `bad` statement's result is correctly attributed.
    if results.len() > 1 {
        for result in &results {
            match result {
                tla_btor2::Btor2CheckResult::Sat { .. } => println!("sat"),
                tla_btor2::Btor2CheckResult::Unsat => println!("unsat"),
                tla_btor2::Btor2CheckResult::Unknown { .. } => println!("unknown"),
            }
        }
    } else if any_sat {
        println!("sat");
    } else if any_unknown {
        println!("unknown");
    } else {
        println!("unsat");
    }

    // Write witness file if requested and there is a counterexample. Additive:
    // the stdout verdict above is unchanged. The word-level portfolio's own
    // counterexample is not directly replayable (opaque CHC-internal names;
    // empty for array states — the concrete array model is trapped in ay-chc's
    // discarded derivation witness), so we emit ONLY a sound, re-simulated,
    // standard witness through the shared projector — the SAME serializer as the
    // bit-blast lane — and write nothing (with an honest note) for a net that is
    // not bit-blastable. Never the old non-replayable `key=value` output.
    if any_sat {
        if let Some(witness_path) = witness_file {
            write_word_level_witness(&program, &results, witness_path, max_bv_width, verbose)?;
        }
    }

    if verbose {
        eprintln!("Elapsed: {:.3}s", elapsed.as_secs_f64());
    }

    Ok(())
}

/// Whether the lazy-array BMC lane was requested: the `--array-bmc` flag or
/// the `TY_BTOR2_LAZY_ARRAY_BMC` env var (mirroring the
/// `TY_BTOR2_INDEPENDENT_CERT` precedent). Default OFF.
fn array_bmc_lane_requested(flag: bool) -> bool {
    flag || std::env::var("TY_BTOR2_LAZY_ARRAY_BMC")
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false)
}

/// Whether the k-induction lane was requested: the `TY_BTOR2_ARRAY_KIND`
/// env var (no CLI flag yet — same opt-in pattern as
/// `TY_BTOR2_INDEPENDENT_CERT`). Default OFF.
fn array_kind_lane_requested() -> bool {
    std::env::var("TY_BTOR2_ARRAY_KIND")
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false)
}

/// Whether the array IC3 lane was requested: the `TY_BTOR2_ARRAY_IC3` env
/// var (same opt-in pattern). Default OFF.
fn array_ic3_lane_requested() -> bool {
    std::env::var("TY_BTOR2_ARRAY_IC3")
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false)
}

/// Run the flag-gated array IC3/PDR lane as a bounded leading slice. Returns
/// `Ok(true)` ONLY when the lane produced (a) a replay-validated
/// counterexample covering every bad property, or (b) a `ProvedSafe` whose
/// frame invariant was independently validated through the LRAT-checked
/// triple (the only unsat verdict this lane can mint). Everything else
/// returns `Ok(false)` and the existing pipeline proceeds unchanged.
fn run_array_ic3_lane(
    program: &tla_btor2::Btor2Program,
    verbose: bool,
    timeout_secs: Option<u64>,
    witness_file: Option<&Path>,
    start: Instant,
) -> Result<bool> {
    let slice_secs = timeout_secs
        .map(|t| (t as f64 * 0.25).clamp(1.0, 30.0))
        .unwrap_or(10.0);
    let config = tla_btor2::ArrayIc3Config {
        time_budget: Some(Duration::from_secs_f64(slice_secs)),
        verbose,
        ..tla_btor2::ArrayIc3Config::default()
    };

    match tla_btor2::check_array_ic3(program, &config) {
        tla_btor2::ArrayIc3Outcome::ProvedSafe {
            converged_at,
            invariant,
            tier,
        } => {
            if verbose {
                eprintln!(
                    "array-ic3: frames converged at level {converged_at} ({} clauses, {} probes); \
                     triple validated via {tier:?} (LRAT-checked)",
                    invariant.clauses.len(),
                    invariant.probes.len()
                );
                eprintln!("Elapsed (array-ic3): {:.3}s", start.elapsed().as_secs_f64());
            }
            for _ in 0..program.bad_properties.len().max(1) {
                println!("unsat");
            }
            Ok(true)
        }
        tla_btor2::ArrayIc3Outcome::Unsafe {
            depth,
            fired,
            witness,
            ..
        } => {
            if fired.len() != program.bad_properties.len() {
                if verbose {
                    eprintln!(
                        "array-ic3: counterexample at depth {depth} covers {}/{} properties — \
                         falling through to the portfolio for full attribution",
                        fired.len(),
                        program.bad_properties.len()
                    );
                }
                return Ok(false);
            }
            if verbose {
                eprintln!(
                    "array-ic3: replay-validated counterexample at depth {depth} (BMC-confirmed)"
                );
            }
            if let Some(witness_path) = witness_file {
                match witness {
                    Some(w) => {
                        std::fs::write(witness_path, w.to_btor2_string()).with_context(|| {
                            format!("failed to write witness file: {}", witness_path.display())
                        })?;
                    }
                    None => {
                        eprintln!(
                            "c witness: array-ic3 counterexample is replay-proven but could not \
                             be serialized in the btorsim per-cell format; no witness emitted \
                             (fail-closed)"
                        );
                    }
                }
            }
            for _ in 0..program.bad_properties.len().max(1) {
                println!("sat");
            }
            if verbose {
                eprintln!("Elapsed (array-ic3): {:.3}s", start.elapsed().as_secs_f64());
            }
            Ok(true)
        }
        tla_btor2::ArrayIc3Outcome::BoundedNoCex { frames_completed } => {
            if verbose {
                eprintln!(
                    "array-ic3: no convergence within {frames_completed} frame(s) (or the \
                     triple validation did not discharge) — falling through to the portfolio"
                );
            }
            Ok(false)
        }
        tla_btor2::ArrayIc3Outcome::Declined { reason } => {
            if verbose {
                eprintln!("array-ic3: declined ({reason}) — falling through to the portfolio");
            }
            Ok(false)
        }
    }
}

/// Run the flag-gated k-induction lane as a bounded leading slice. Returns
/// `Ok(true)` ONLY when the lane produced (a) a replay-validated
/// counterexample covering every bad property, or (b) an independently
/// LRAT-discharged unbounded-safe proof (the lane's `ProvedSafe` — the only
/// unsat verdict this lane can mint). Everything else returns `Ok(false)`
/// and the existing pipeline proceeds unchanged.
fn run_array_kind_lane(
    program: &tla_btor2::Btor2Program,
    verbose: bool,
    timeout_secs: Option<u64>,
    witness_file: Option<&Path>,
    start: Instant,
) -> Result<bool> {
    let slice_secs = timeout_secs
        .map(|t| (t as f64 * 0.25).clamp(1.0, 30.0))
        .unwrap_or(10.0);
    let config = tla_btor2::ArrayKindConfig {
        time_budget: Some(Duration::from_secs_f64(slice_secs)),
        verbose,
        ..tla_btor2::ArrayKindConfig::default()
    };

    match tla_btor2::check_array_kinduction(program, &config) {
        tla_btor2::ArrayKindOutcome::ProvedSafe { k } => {
            if verbose {
                eprintln!(
                    "array-kind: every property is {k}-inductive; base and step queries \
                     independently re-discharged (LRAT-checked)"
                );
                eprintln!(
                    "Elapsed (array-kind): {:.3}s",
                    start.elapsed().as_secs_f64()
                );
            }
            for _ in 0..program.bad_properties.len().max(1) {
                println!("unsat");
            }
            Ok(true)
        }
        tla_btor2::ArrayKindOutcome::Unsafe {
            depth,
            fired,
            witness,
            ..
        } => {
            // Same attribution rule as the BMC lane: accept only full
            // property coverage so each printed verdict is replay-proven.
            if fired.len() != program.bad_properties.len() {
                if verbose {
                    eprintln!(
                        "array-kind: counterexample at depth {depth} covers {}/{} properties — \
                         falling through to the portfolio for full attribution",
                        fired.len(),
                        program.bad_properties.len()
                    );
                }
                return Ok(false);
            }
            if verbose {
                eprintln!(
                    "array-kind: replay-validated counterexample at depth {depth} (base BMC)"
                );
            }
            if let Some(witness_path) = witness_file {
                match witness {
                    Some(w) => {
                        std::fs::write(witness_path, w.to_btor2_string()).with_context(|| {
                            format!("failed to write witness file: {}", witness_path.display())
                        })?;
                    }
                    None => {
                        eprintln!(
                            "c witness: array-kind counterexample is replay-proven but could not \
                             be serialized in the btorsim per-cell format; no witness emitted \
                             (fail-closed)"
                        );
                    }
                }
            }
            for _ in 0..program.bad_properties.len().max(1) {
                println!("sat");
            }
            if verbose {
                eprintln!(
                    "Elapsed (array-kind): {:.3}s",
                    start.elapsed().as_secs_f64()
                );
            }
            Ok(true)
        }
        tla_btor2::ArrayKindOutcome::BoundedNoCex { depth_reached } => {
            if verbose {
                eprintln!(
                    "array-kind: no counterexample within {depth_reached} step(s) and no k \
                     became inductive (or the independent discharge did not verify) — \
                     falling through to the portfolio"
                );
            }
            Ok(false)
        }
        tla_btor2::ArrayKindOutcome::Declined { reason } => {
            if verbose {
                eprintln!("array-kind: declined ({reason}) — falling through to the portfolio");
            }
            Ok(false)
        }
    }
}

/// Whether `program` declares at least one array-sorted state (ANY index
/// width) — the lazy-array BMC lane's shape gate.
fn has_array_state(program: &tla_btor2::Btor2Program) -> bool {
    use tla_btor2::{Btor2Node, Btor2Sort};
    program.lines.iter().any(|line| {
        matches!(&line.node, Btor2Node::State(sort_id, _)
            if matches!(program.sorts.get(sort_id), Some(Btor2Sort::Array { .. })))
    })
}

/// Run the flag-gated lazy-array BMC lane as a bounded leading slice. Returns
/// `Ok(true)` ONLY when the lane produced a replay-validated counterexample
/// covering every bad property (the stdout verdict is then printed here);
/// every other outcome — bounded no-cex, decline, partial property coverage —
/// returns `Ok(false)` and the caller's existing pipeline proceeds unchanged.
fn run_array_bmc_lane(
    program: &tla_btor2::Btor2Program,
    verbose: bool,
    timeout_secs: Option<u64>,
    witness_file: Option<&Path>,
    start: Instant,
) -> Result<bool> {
    // Bounded leading slice (GPU-lane pattern): a fraction of the total
    // budget, clamped, so the CHC portfolio can never be starved.
    let slice_secs = timeout_secs
        .map(|t| (t as f64 * 0.25).clamp(1.0, 30.0))
        .unwrap_or(10.0);
    let config = tla_btor2::ArrayBmcConfig {
        time_budget: Some(Duration::from_secs_f64(slice_secs)),
        verbose,
        ..tla_btor2::ArrayBmcConfig::default()
    };

    match tla_btor2::check_array_bmc(program, &config) {
        tla_btor2::ArrayBmcOutcome::Unsafe {
            depth,
            fired,
            witness,
            ..
        } => {
            // Phase-1 attribution rule: accept only when the replay-confirmed
            // firing covers EVERY bad property, so the per-property verdict
            // lines below are each individually replay-proven. A partial
            // firing falls through to the portfolio for a complete answer.
            if fired.len() != program.bad_properties.len() {
                if verbose {
                    eprintln!(
                        "array-bmc: counterexample at depth {depth} covers {}/{} properties — \
                         falling through to the portfolio for full attribution",
                        fired.len(),
                        program.bad_properties.len()
                    );
                }
                return Ok(false);
            }
            if verbose {
                eprintln!(
                    "array-bmc: replay-validated counterexample at depth {depth} \
                     (lazy-array trace-unrolled BMC)"
                );
            }
            if let Some(witness_path) = witness_file {
                match witness {
                    Some(w) => {
                        std::fs::write(witness_path, w.to_btor2_string()).with_context(|| {
                            format!("failed to write witness file: {}", witness_path.display())
                        })?;
                        if verbose {
                            eprintln!(
                                "Witness written to {} ({} frame(s)) — word-level replay-verified",
                                witness_path.display(),
                                w.frame_count(),
                            );
                        }
                    }
                    None => {
                        eprintln!(
                            "c witness: array-bmc counterexample is replay-proven but could not \
                             be serialized in the btorsim per-cell format (e.g. nonzero-default \
                             initial array); no witness emitted (fail-closed)"
                        );
                    }
                }
            }
            for _ in 0..program.bad_properties.len().max(1) {
                println!("sat");
            }
            if verbose {
                eprintln!("Elapsed (array-bmc): {:.3}s", start.elapsed().as_secs_f64());
            }
            Ok(true)
        }
        tla_btor2::ArrayBmcOutcome::BoundedNoCex { depth_reached } => {
            // Bounded evidence only — NEVER an unsat verdict (absolute rule 3).
            if verbose {
                eprintln!(
                    "array-bmc: no counterexample within {depth_reached} step(s) \
                     (bounded k-safety evidence only) — falling through to the portfolio"
                );
            }
            Ok(false)
        }
        tla_btor2::ArrayBmcOutcome::Declined { reason } => {
            if verbose {
                eprintln!("array-bmc: declined ({reason}) — falling through to the portfolio");
            }
            Ok(false)
        }
    }
}

/// Whether `program` declares at least one bounded state array (index width
/// ≤ 12), i.e. is in the independent certifier's target class.
fn has_bounded_array_state(program: &tla_btor2::Btor2Program) -> bool {
    use tla_btor2::{Btor2Node, Btor2Sort};
    for line in &program.lines {
        if let Btor2Node::State(sort_id, _) = &line.node {
            if let Some(Btor2Sort::Array { index, .. }) = program.sorts.get(sort_id) {
                if let Btor2Sort::BitVec(iw) = index.as_ref() {
                    if *iw <= 12 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Opt-in, additive independent SAFE certifier (design §4.2). Runs only when
/// `TY_BTOR2_INDEPENDENT_CERT` is set and the net has a bounded array state.
/// Prints a single `c ...` comment to STDERR; the caller's stdout verdict path
/// is untouched. Fail-closed: any error/decline is reported as a withheld
/// confirmation, never a promotion of the verdict.
fn run_independent_cert_if_requested(
    program: &tla_btor2::Btor2Program,
    verbose: bool,
    timeout_secs: Option<u64>,
) {
    let requested = std::env::var("TY_BTOR2_INDEPENDENT_CERT")
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false);
    if !requested {
        return;
    }
    if !has_bounded_array_state(program) {
        if verbose {
            eprintln!("c independent-cert: skipped (no bounded array state — out of scope)");
        }
        return;
    }

    let budget = timeout_secs.map(std::time::Duration::from_secs);
    match tla_btor2::certify_btor2_safe_independent(program, budget) {
        Ok(tla_btor2::IndependentCertResult::Certified {
            index_width_bits,
            cells,
            vcs_discharged,
        }) => {
            eprintln!(
                "c independently-certified SAFE (disjoint LRAT gate): \
                 {vcs_discharged} one-step VCs UNSAT, array bit-blasted (index_width={index_width_bits} bits, {cells} cells)"
            );
        }
        Ok(tla_btor2::IndependentCertResult::NotConfirmed { reason }) => {
            eprintln!("c not-independently-confirmed: {reason}");
        }
        Err(e) => {
            eprintln!("c independent-cert: declined (translation error: {e})");
        }
    }
}

/// Convert a bit-blasted BTOR2 circuit to an AIGER circuit. The latch/input
/// ordering is preserved one-to-one, so a trace keyed by AIGER latch/input
/// index (`l{idx}`/`i{idx}`) maps straight back to `bb.latches`/`bb.inputs`.
fn bitblast_to_aiger(bb: &tla_btor2::BitblastedCircuit) -> tla_aiger::AigerCircuit {
    use tla_aiger::{AigerAnd, AigerCircuit, AigerLatch, AigerSymbol};
    AigerCircuit {
        maxvar: bb.max_var,
        inputs: bb
            .inputs
            .iter()
            .map(|&lit| AigerSymbol { lit, name: None })
            .collect(),
        latches: bb
            .latches
            .iter()
            .map(|&(curr, next, reset)| AigerLatch {
                lit: curr,
                next,
                reset,
                name: None,
            })
            .collect(),
        outputs: Vec::new(),
        ands: bb
            .ands
            .iter()
            .map(|&(lhs, rhs0, rhs1)| AigerAnd { lhs, rhs0, rhs1 })
            .collect(),
        bad: bb
            .bad
            .iter()
            .map(|&lit| AigerSymbol { lit, name: None })
            .collect(),
        constraints: bb
            .constraints
            .iter()
            .map(|&lit| AigerSymbol { lit, name: None })
            .collect(),
        justice: Vec::new(),
        fairness: Vec::new(),
        comments: vec!["bit-blasted from BTOR2".into()],
    }
}

/// Run the bit-blast path: BTOR2 -> AIGER -> IC3/PDR portfolio.
///
/// Returns the bit-blasted circuit (metadata for witness projection) alongside
/// the per-property results, without printing verdicts, so the caller can
/// decide whether to accept them or fall back to the word-level CHC path.
fn run_btor2_bitblast(
    program: &tla_btor2::Btor2Program,
    verbose: bool,
    timeout_secs: Option<u64>,
) -> Result<(
    tla_btor2::BitblastedCircuit,
    Vec<tla_aiger::AigerCheckResult>,
)> {
    // Bit-blast to AIGER-compatible circuit.
    let bb =
        tla_btor2::bitblast(program, 32).map_err(|e| anyhow::anyhow!("bit-blast error: {e}"))?;

    if verbose {
        eprintln!(
            "Bit-blasted: {} vars, {} inputs, {} latches, {} AND gates, {} bad, {} constraints",
            bb.max_var,
            bb.inputs.len(),
            bb.latches.len(),
            bb.ands.len(),
            bb.bad.len(),
            bb.constraints.len(),
        );
    }

    let circuit = bitblast_to_aiger(&bb);

    // Run the AIGER IC3/PDR portfolio.
    let timeout = timeout_secs.map(std::time::Duration::from_secs);
    let results = tla_aiger::check_aiger_sat(&circuit, timeout);
    Ok((bb, results))
}

/// Project the bit-blast counterexample into a BTOR2 witness file.
///
/// The portfolio's own trace is keyed to a *preprocessed* circuit (COI, SCORR,
/// constant elimination reshape the latch/input set) with no reconstruction map
/// back to the source, so it cannot be projected directly. Instead we re-derive
/// a concrete counterexample on the UN-preprocessed circuit via BMC — its
/// `l{idx}`/`i{idx}` ordinals then align with `bb.latches`/`bb.inputs` and each
/// bit maps back to its word/array state. The BMC depth bound comes from the
/// depth the portfolio already reported (property-preserving preprocessing keeps
/// the shortest-counterexample depth, so `trace.len()` bounds it).
///
/// Fail-closed and additive: no `sat` property, an un-reproducible (deeper than
/// the bound) counterexample, or a replay that does not reach a bad state all
/// leave the witness file unwritten and the stdout verdict untouched.
fn write_bitblast_witness(
    program: &tla_btor2::Btor2Program,
    bb: &tla_btor2::BitblastedCircuit,
    results: &[tla_aiger::AigerCheckResult],
    witness_path: &Path,
    verbose: bool,
) -> Result<()> {
    let depth_hint = results.iter().find_map(|r| match r {
        tla_aiger::AigerCheckResult::Sat { trace } => Some(trace.len()),
        _ => None,
    });
    let Some(depth_hint) = depth_hint else {
        // No violated property -> no counterexample -> no witness (correct for
        // an unsat/unknown net).
        return Ok(());
    };

    write_projected_witness(program, bb, depth_hint, witness_path, verbose)
}

/// Shared, fail-closed emitter for the standard btorsim-compatible BTOR2 witness.
///
/// Re-derives a concrete bit-level counterexample on the UN-preprocessed circuit
/// `bb` via BMC (the portfolio's own trace is keyed to a preprocessed circuit
/// with no reconstruction map, so it cannot be projected directly), then projects
/// it back to the word/array level and re-simulates it forward
/// ([`tla_btor2::project_bitblast_witness`], btorsim replay semantics), emitting
/// a witness only if a `bad` literal genuinely fires. `depth_hint` bounds the
/// BMC re-derivation: it is the length of the counterexample the solving lane
/// already reported (property-preserving preprocessing keeps the shortest-
/// counterexample depth), `+1` slack, `.max(2)` for a degenerate short hint.
///
/// Both the bit-blast lane and the word-level lane route through here so they
/// emit byte-identical, replayable witnesses (DRY). Fail-closed and additive: an
/// un-reproducible (deeper than the bound) counterexample or a replay that does
/// not reach a bad state leaves the file unwritten and prints an honest note.
fn write_projected_witness(
    program: &tla_btor2::Btor2Program,
    bb: &tla_btor2::BitblastedCircuit,
    depth_hint: usize,
    witness_path: &Path,
    verbose: bool,
) -> Result<()> {
    let aiger = bitblast_to_aiger(bb);
    let max_depth = (depth_hint + 1).max(2);
    let Some(bool_trace) = tla_aiger::extract_original_cex_trace(&aiger, max_depth) else {
        eprintln!(
            "c witness: verdict is sat but a concrete counterexample could not be \
             re-derived within depth {max_depth}; no witness emitted (next step: \
             deeper / word-level witness extraction)"
        );
        return Ok(());
    };

    let Some(witness) = tla_btor2::project_bitblast_witness(program, bb, &bool_trace) else {
        eprintln!(
            "c witness: reconstructed trace did not replay into a bad state; \
             withholding witness (fail-closed)"
        );
        return Ok(());
    };

    let content = witness.to_btor2_string();
    std::fs::write(witness_path, &content)
        .with_context(|| format!("failed to write witness file: {}", witness_path.display()))?;
    if verbose {
        eprintln!(
            "Witness written to {} ({} frame(s), {} violated bad property/ies)",
            witness_path.display(),
            witness.frame_count(),
            witness.bad_property_count(),
        );
    }
    Ok(())
}

/// Emit a standard btorsim-compatible BTOR2 witness for a counterexample found
/// by the WORD-LEVEL portfolio (COI + BMC + CHC), fail-closed and additive.
///
/// For a bit-blast-*eligible* net we re-derive and re-simulate a concrete
/// counterexample for the SAME net and serialize it through the shared
/// [`write_projected_witness`] (well-tested bit-level lane). For a bit-blast-
/// *ineligible* net (e.g. a wide-index array — precisely the class that reaches
/// the word-level lane), the concrete per-frame array model recovered from
/// ay-chc's derivation witness is threaded through
/// [`tla_btor2::Btor2CheckResult::Sat`]'s `model`: we replay it forward over the
/// BTOR2 program with [`tla_btor2::build_word_level_witness`] (bit-blast-free,
/// arrays-as-maps) and emit the standard witness ONLY if the replay genuinely
/// reaches a bad state. If the model is absent/incomplete or does not replay to
/// bad, no file is written and an honest note is printed — never a fabricated
/// witness, and never the old non-replayable `key=value` output.
fn write_word_level_witness(
    program: &tla_btor2::Btor2Program,
    results: &[tla_btor2::Btor2CheckResult],
    witness_path: &Path,
    max_bv_width: u32,
    verbose: bool,
) -> Result<()> {
    // Depth hint = the word-level counterexample's step count (present even when
    // its per-step assignments are empty, as they are for array states).
    let depth_hint = results.iter().find_map(|r| match r {
        tla_btor2::Btor2CheckResult::Sat { trace, .. } => Some(trace.len()),
        _ => None,
    });
    let Some(depth_hint) = depth_hint else {
        // No violated property -> no counterexample -> no witness.
        return Ok(());
    };

    // Bit-blast-eligible: reuse the battle-tested bit-level projection lane.
    if tla_btor2::bitblast_eligible(program, max_bv_width).is_ok() {
        let bb = match tla_btor2::bitblast(program, max_bv_width) {
            Ok(bb) => bb,
            Err(e) => {
                eprintln!(
                    "c witness: bit-blast for word-level witness projection failed ({e}); \
                     no witness emitted (fail-closed)"
                );
                return Ok(());
            }
        };
        return write_projected_witness(program, &bb, depth_hint, witness_path, verbose);
    }

    // Bit-blast-INELIGIBLE (wide-index array, ...): project the concrete
    // word-level model from ay-chc's derivation witness, replay-verified.
    let model = results.iter().find_map(|r| match r {
        tla_btor2::Btor2CheckResult::Sat { model: Some(m), .. } => Some(m),
        _ => None,
    });
    if let Some(model) = model {
        if let Some(witness) = tla_btor2::build_word_level_witness(program, model) {
            let content = witness.to_btor2_string();
            std::fs::write(witness_path, &content).with_context(|| {
                format!("failed to write witness file: {}", witness_path.display())
            })?;
            if verbose {
                eprintln!(
                    "Witness written to {} ({} frame(s), {} violated bad property/ies) \
                     — word-level replay-verified (bit-blast-free array model)",
                    witness_path.display(),
                    witness.frame_count(),
                    witness.bad_property_count(),
                );
            }
            return Ok(());
        }
    }

    eprintln!(
        "c witness: sat from the word-level CHC portfolio on a net that is not \
         bit-blastable (e.g. a wide-index array). The concrete array model from \
         ay-chc's derivation witness was {}; no witness emitted (fail-closed).",
        if model.is_some() {
            "recovered but did not replay forward into a bad state"
        } else {
            "not available on this result"
        }
    );
    Ok(())
}

/// Print HWMCC verdicts (one `sat`/`unsat`/`unknown` line per property, or a
/// single line for single-property nets) from bit-blast IC3/PDR results.
fn print_aiger_verdicts(
    results: &[tla_aiger::AigerCheckResult],
    verbose: bool,
    start: Instant,
) -> Result<()> {
    let elapsed = start.elapsed();

    if results.is_empty() {
        println!("unsat");
        if verbose {
            eprintln!("No properties to check after bit-blasting.");
        }
        return Ok(());
    }

    // Print results following HWMCC convention.
    let mut any_sat = false;
    let mut any_unknown = false;

    for (idx, result) in results.iter().enumerate() {
        match result {
            tla_aiger::AigerCheckResult::Sat { .. } => {
                any_sat = true;
                if verbose {
                    eprintln!("Property {idx}: VIOLATED (bit-blast IC3/PDR)");
                }
            }
            tla_aiger::AigerCheckResult::Unsat => {
                if verbose {
                    eprintln!("Property {idx}: HOLDS (bit-blast IC3/PDR)");
                }
            }
            tla_aiger::AigerCheckResult::Unknown { reason } => {
                any_unknown = true;
                if verbose {
                    eprintln!("Property {idx}: UNKNOWN ({reason})");
                }
            }
        }
    }

    if results.len() > 1 {
        for result in results {
            match result {
                tla_aiger::AigerCheckResult::Sat { .. } => println!("sat"),
                tla_aiger::AigerCheckResult::Unsat => println!("unsat"),
                tla_aiger::AigerCheckResult::Unknown { .. } => println!("unknown"),
            }
        }
    } else if any_sat {
        println!("sat");
    } else if any_unknown {
        println!("unknown");
    } else {
        println!("unsat");
    }

    if verbose {
        eprintln!("Elapsed (bit-blast): {:.3}s", elapsed.as_secs_f64());
    }

    Ok(())
}
