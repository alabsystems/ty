// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bounded model checking (BMC) for reachability properties via ay.
//!
//! Encodes Petri net reachability as SMT-LIB queries and runs ay to find
//! witnesses:
//! - `EF(phi)` -> `TRUE` via SAT (witness found)
//! - `AG(phi)` -> `FALSE` via SAT (counterexample for not phi)
//! - UNSAT, timeout, unknown -> inconclusive (falls through to later phases)

use std::time::{Duration, Instant};

use tla_mc_core::{CapabilityReport, ProblemKind, SolverLimits};

use crate::petri_net::{PetriNet, TransitionIdx};
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::eval_predicate;

use super::bmc_runner::{
    emit_bmc_incremental_preamble, run_depth_ladder_incremental_with_report,
    run_depth_ladder_with_report, DepthAction, DepthQuery, IncrementalPropertyQuery,
};
use super::reachability::{resolve_tracker, PropertyTracker, ReachabilityResolutionSource};
use super::reachability_witness::WitnessValidationTarget;
#[cfg(test)]
use super::smt_encoding::encode_int_expr;
#[cfg(test)]
use super::smt_encoding::find_ay;
use super::smt_encoding::{
    encode_predicate, find_ay_with_report, run_ay_bool_model, run_ay_with_report,
    AYSolveProfileEvidence, SolverBoolModel, SolverOutcome, SolverRunReport, DEPTH_LADDER,
    PER_DEPTH_TIMEOUT,
};

const BMC_SPLIT_RETRY_FALLBACK_RESERVE: Duration = Duration::from_secs(8);
const BMC_SPLIT_RETRY_MIN_BUDGET: Duration = Duration::from_millis(250);
const BMC_SHORT_DEADLINE_MULTI_PROPERTY_LIMIT: Duration = Duration::from_secs(30);
const BMC_LARGE_PENDING_THRESHOLD: usize = 12;
const BMC_DEPTH_ONE_LADDER: &[usize] = &[1];
const ENABLE_BMC_DEPTH1_CHUNKING_ENV: &str = "TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING";
const DISABLE_BMC_DEPTH1_CHUNKING_ENV: &str = "TY_MCC_DISABLE_BMC_DEPTH1_CHUNKING";
const BMC_DEPTH1_CHUNK_SIZE_ENV: &str = "TY_MCC_BMC_DEPTH1_CHUNK_SIZE";
const BMC_DEPTH1_CHUNK_SIZE_DEFAULT: usize = 4;
const AY_BMC_DEADLINE_INCREMENTAL_ENV: &str = "TY_MCC_AY_BMC_DEADLINE_INCREMENTAL";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::examinations) enum BmcTraceStep {
    Stay,
    Fire(TransitionIdx),
}

/// Run BMC seeding on pre-resolved trackers.
///
/// For each unresolved property, attempts to find a witness via ay-backed
/// BMC on the original net. Seeds `verdict` on trackers where witnesses
/// are found. Does nothing if ay is not available.
///
/// Returns the maximum depth at which BMC completed without UNKNOWN results
/// (i.e., all pending properties returned SAT or UNSAT). This is the base-case
/// depth for subsequent k-induction - k-induction at depth k requires the BMC
/// base case to cover at least k states (i.e., `max_bmc_depth + 1 >= k`).
pub(crate) fn run_bmc_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    deadline: Option<Instant>,
) -> Option<usize> {
    run_bmc_seeding_with_report(net, trackers, validation_targets, deadline).0
}

/// Run BMC seeding and return the shared AY capability evidence used by this
/// runtime solver selection.
pub(crate) fn run_bmc_seeding_with_report(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    deadline: Option<Instant>,
) -> (Option<usize>, CapabilityReport) {
    let (ay_path, mut capability_report) = find_ay_with_report(
        ProblemKind::Bmc,
        SolverLimits {
            time_budget: deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now())),
            max_depth: DEPTH_LADDER.last().copied().map(|depth| depth as u32),
            max_states: None,
            max_memory_bytes: None,
        },
    );
    let ay_path = match ay_path {
        Some(path) => path,
        None => {
            capability_report.add_evidence("reachability BMC skipped because ay was unavailable");
            eprintln!("BMC: ay not found, skipping bounded model checking");
            crate::mcc_backend_evidence::record_runtime_reachability_bmc_report(&capability_report);
            return (None, capability_report);
        }
    };
    capability_report.add_evidence(format!(
        "reachability BMC runtime selected ay at {}",
        ay_path.display()
    ));
    eprintln!("BMC: using ay at {}", ay_path.display());
    let ReachabilityBmcRunReport {
        depth,
        solve_profile,
    } = run_bmc_seeding_with_solver_path_report(
        net,
        trackers,
        validation_targets,
        deadline,
        &ay_path,
    );
    if let Some(profile) = solve_profile {
        capability_report.add_evidence(profile.into_row());
    }
    crate::mcc_backend_evidence::record_runtime_reachability_bmc_report(&capability_report);
    (depth, capability_report)
}

struct ReachabilityBmcRunReport {
    depth: Option<usize>,
    solve_profile: Option<AYSolveProfileEvidence>,
}

/// Run BMC seeding with an explicit solver path.
///
/// Split from [`run_bmc_seeding`] so tests can inject a fake solver without
/// mutating global environment variables.
///
/// Returns the maximum depth at which all pending properties completed
/// without UNKNOWN (the base case for k-induction soundness).
fn run_bmc_seeding_with_solver_path(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    deadline: Option<Instant>,
    ay_path: &std::path::Path,
) -> Option<usize> {
    run_bmc_seeding_with_solver_path_report(net, trackers, validation_targets, deadline, ay_path)
        .depth
}

fn run_bmc_seeding_with_solver_path_report(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    deadline: Option<Instant>,
    ay_path: &std::path::Path,
) -> ReachabilityBmcRunReport {
    let use_deadline_incremental = deadline.is_some() && bmc_deadline_incremental_enabled();
    run_bmc_seeding_with_solver_path_mode_report(
        net,
        trackers,
        validation_targets,
        deadline,
        ay_path,
        use_deadline_incremental,
    )
}

fn run_bmc_seeding_with_solver_path_mode(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    deadline: Option<Instant>,
    ay_path: &std::path::Path,
    use_deadline_incremental: bool,
) -> Option<usize> {
    run_bmc_seeding_with_solver_path_mode_report(
        net,
        trackers,
        validation_targets,
        deadline,
        ay_path,
        use_deadline_incremental,
    )
    .depth
}

fn run_bmc_seeding_with_solver_path_mode_report(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    deadline: Option<Instant>,
    ay_path: &std::path::Path,
    use_deadline_incremental: bool,
) -> ReachabilityBmcRunReport {
    if validation_targets.len() != trackers.len() {
        eprintln!(
            "BMC: validation target count mismatch ({} targets for {} trackers), skipping",
            validation_targets.len(),
            trackers.len()
        );
        return ReachabilityBmcRunReport {
            depth: None,
            solve_profile: None,
        };
    }

    let unresolved: Vec<usize> = trackers
        .iter()
        .enumerate()
        .filter(|(_, tracker)| tracker.verdict.is_none())
        .map(|(index, _)| index)
        .collect();

    if unresolved.is_empty() {
        return ReachabilityBmcRunReport {
            depth: None,
            solve_profile: None,
        };
    }

    struct ReachabilityBmcState<'a> {
        net: &'a PetriNet,
        trackers: &'a mut [PropertyTracker],
        validation_targets: &'a [WitnessValidationTarget],
        unresolved: Vec<usize>,
        pending: Vec<usize>,
        last_solve_profile: Option<AYSolveProfileEvidence>,
    }

    let mut state = ReachabilityBmcState {
        net,
        trackers,
        validation_targets,
        unresolved,
        pending: Vec::new(),
        last_solve_profile: None,
    };
    let depths = bmc_depth_ladder(deadline, state.unresolved.len());
    if depths == BMC_DEPTH_ONE_LADDER && DEPTH_LADDER.len() > 1 {
        eprintln!(
            "BMC: short deadline with {} pending properties, capping BMC at depth 1",
            state.unresolved.len()
        );
    }

    let apply_results = |state: &mut ReachabilityBmcState<'_>,
                         depth: usize,
                         pending: &[usize],
                         results: &[SolverOutcome]|
     -> bool {
        let mut had_unknown = results.len() != pending.len();
        for (property_idx, outcome) in pending.iter().zip(results.iter()) {
            match outcome {
                SolverOutcome::Sat => {
                    match validate_bmc_sat_witness(
                        ay_path,
                        state.net,
                        &state.trackers[*property_idx],
                        &state.validation_targets[*property_idx],
                        depth,
                        deadline,
                    ) {
                        Ok(()) => {
                            let verdict = bmc_sat_verdict(state.trackers[*property_idx].quantifier);
                            let reason = match state.trackers[*property_idx].quantifier {
                                PathQuantifier::EF => "EF witness",
                                PathQuantifier::AG => "AG counterexample",
                            };
                            let label = if verdict { "TRUE" } else { "FALSE" };
                            let property_id = state.trackers[*property_idx].id.clone();
                            resolve_tracker(
                                &mut state.trackers[*property_idx],
                                verdict,
                                ReachabilityResolutionSource::Bmc,
                                Some(depth),
                            );
                            eprintln!("BMC depth {depth}: {property_id} = {label} ({reason})");
                        }
                        Err(reason) => {
                            had_unknown = true;
                            eprintln!(
                                "BMC depth {depth}: {} SAT model rejected ({reason}); leaving unresolved",
                                state.trackers[*property_idx].id
                            );
                        }
                    }
                }
                SolverOutcome::Unsat => {}
                SolverOutcome::Unknown => had_unknown = true,
            }
        }
        had_unknown
    };

    let complete_depth = |state: &mut ReachabilityBmcState<'_>,
                          depth: usize,
                          pending: &[usize],
                          results: &[SolverOutcome]|
     -> DepthAction {
        if apply_results(state, depth, pending, results) {
            eprintln!("BMC depth {depth}: unknown result, stopping BMC");
            DepthAction::StopDeepening
        } else {
            DepthAction::Explored
        }
    };

    let retry_depth_individually =
        |state: &mut ReachabilityBmcState<'_>, depth: usize| -> Vec<SolverOutcome> {
            let pending = state.pending.clone();
            let mut outcomes = Vec::with_capacity(pending.len());
            for (offset, property_idx) in pending.iter().copied().enumerate() {
                let retries_left = pending.len() - offset;
                let timeout = bmc_split_retry_timeout(deadline, retries_left, Instant::now());
                if timeout < BMC_SPLIT_RETRY_MIN_BUDGET {
                    outcomes.push(SolverOutcome::Unknown);
                    continue;
                }

                let script = encode_bmc_script(state.net, state.trackers, &[property_idx], depth);
                match run_ay_with_report(ay_path, &script, 1, timeout) {
                    Some(report) => {
                        if let Some(profile) = report.solve_profile() {
                            state.last_solve_profile = Some(profile.clone());
                        }
                        // Fail-closed reports carry profile evidence but no real
                        // solver outcomes — record `Unknown` for this property.
                        if report.is_fail_closed() {
                            outcomes.push(SolverOutcome::Unknown);
                        } else {
                            match report.outcomes().first().copied() {
                                Some(outcome) => outcomes.push(outcome),
                                None => outcomes.push(SolverOutcome::Unknown),
                            }
                        }
                    }
                    None => outcomes.push(SolverOutcome::Unknown),
                }
            }
            outcomes
        };

    let capture_solver_profile =
        |state: &mut ReachabilityBmcState<'_>, report: Option<&SolverRunReport>| {
            if let Some(profile) = report.and_then(SolverRunReport::solve_profile) {
                state.last_solve_profile = Some(profile.clone());
            }
        };

    // Shared result handler - used by both incremental and fallback paths.
    //
    // A fail-closed report (timeout / non-zero exit / short stdout) still
    // carries raw-SMT profile evidence but has no real solver outcomes. Treat
    // it the same as `None`: capture the profile, then retry per-property or
    // stop deepening.
    let handle_result = |state: &mut ReachabilityBmcState<'_>,
                         depth: usize,
                         report: Option<&SolverRunReport>|
     -> DepthAction {
        capture_solver_profile(state, report);
        let pending = state.pending.clone();
        let results = report
            .filter(|r| !r.is_fail_closed())
            .map(SolverRunReport::outcomes);
        match results {
            Some(results) => complete_depth(state, depth, &pending, results),
            None if should_retry_depth_individually(deadline, pending.len()) => {
                eprintln!(
                    "BMC depth {depth}: solver failed for {} properties; retrying individually",
                    pending.len()
                );
                let individual_results = retry_depth_individually(state, depth);
                complete_depth(state, depth, &pending, &individual_results)
            }
            None => {
                eprintln!("BMC depth {depth}: solver failed, stopping BMC");
                DepthAction::StopDeepening
            }
        }
    };
    let fallback_handle_result = |state: &mut ReachabilityBmcState<'_>,
                                  depth: usize,
                                  report: Option<&SolverRunReport>|
     -> DepthAction {
        capture_solver_profile(state, report);
        let pending = state.pending.clone();
        let results = report
            .filter(|r| !r.is_fail_closed())
            .map(SolverRunReport::outcomes);
        match results {
            Some(results) => complete_depth(state, depth, &pending, results),
            None if should_retry_depth_individually(deadline, pending.len()) => {
                eprintln!(
                    "BMC depth {depth}: solver failed for {} properties; retrying individually",
                    pending.len()
                );
                let individual_results = retry_depth_individually(state, depth);
                complete_depth(state, depth, &pending, &individual_results)
            }
            None => {
                eprintln!("BMC depth {depth}: solver failed, stopping BMC");
                DepthAction::StopDeepening
            }
        }
    };

    // Shared pending-refresh - used by both incremental and fallback query builders.
    let refresh_pending = |state: &mut ReachabilityBmcState<'_>| -> bool {
        state.pending = state
            .unresolved
            .iter()
            .copied()
            .filter(|&index| state.trackers[index].verdict.is_none())
            .collect();
        !state.pending.is_empty()
    };

    let build_incremental_query = |state: &mut ReachabilityBmcState<'_>, depth| {
        if !refresh_pending(state) {
            return None;
        }
        Some(IncrementalPropertyQuery {
            assertions: encode_property_assertions(
                state.net,
                state.trackers,
                &state.pending,
                depth,
            ),
        })
    };

    let build_batch_query = |state: &mut ReachabilityBmcState<'_>, depth| {
        if !refresh_pending(state) {
            return None;
        }
        Some(DepthQuery::new(
            encode_bmc_script(state.net, state.trackers, &state.pending, depth),
            state.pending.len(),
        ))
    };

    if let Some(global_deadline) = deadline {
        if !use_deadline_incremental
            && should_chunk_depth1_all_property_bmc(deadline, depths, state.unresolved.len())
        {
            let chunk_size = bmc_depth1_chunk_size();
            eprintln!(
                "BMC depth 1: chunking {} pending properties into groups of {}",
                state.unresolved.len(),
                chunk_size
            );
            if !refresh_pending(&mut state) {
                return ReachabilityBmcRunReport {
                    depth: None,
                    solve_profile: state.last_solve_profile,
                };
            }
            let starting_pending = state.pending.clone();
            let total_chunks = starting_pending.len().div_ceil(chunk_size);
            let mut depth_complete = true;
            for (chunk_index, chunk) in starting_pending.chunks(chunk_size).enumerate() {
                let chunks_left = total_chunks - chunk_index;
                let remaining = global_deadline.saturating_duration_since(Instant::now());
                let available = remaining.saturating_sub(BMC_SPLIT_RETRY_FALLBACK_RESERVE);
                let timeout = PER_DEPTH_TIMEOUT.min(available / chunks_left as u32);
                if timeout < BMC_SPLIT_RETRY_MIN_BUDGET {
                    depth_complete = false;
                    break;
                }

                let script = encode_bmc_script(state.net, state.trackers, chunk, 1);
                match run_ay_with_report(ay_path, &script, chunk.len(), timeout) {
                    Some(report)
                        if !report.is_fail_closed() && report.outcomes().len() == chunk.len() =>
                    {
                        if let Some(profile) = report.solve_profile() {
                            state.last_solve_profile = Some(profile.clone());
                        }
                        if apply_results(&mut state, 1, chunk, report.outcomes()) {
                            depth_complete = false;
                        }
                    }
                    Some(report) => {
                        // Fail-closed: still capture profile evidence so downstream
                        // gates can render the unavailable-status row.
                        if let Some(profile) = report.solve_profile() {
                            state.last_solve_profile = Some(profile.clone());
                        }
                        depth_complete = false;
                    }
                    None => depth_complete = false,
                }
            }
            return ReachabilityBmcRunReport {
                depth: depth_complete.then_some(1),
                solve_profile: state.last_solve_profile,
            };
        }
    }

    let depth = if deadline.is_some() && !use_deadline_incremental {
        eprintln!("BMC: deadline set, using batch ay mode");
        run_depth_ladder_with_report(
            ay_path,
            depths,
            deadline,
            PER_DEPTH_TIMEOUT,
            &mut state,
            build_batch_query,
            handle_result,
        )
    } else {
        if use_deadline_incremental {
            eprintln!(
                "BMC: deadline set, using incremental ay mode via {AY_BMC_DEADLINE_INCREMENTAL_ENV}"
            );
        }
        run_depth_ladder_incremental_with_report(
            ay_path,
            depths,
            deadline,
            PER_DEPTH_TIMEOUT,
            net,
            &mut state,
            // One-time preamble (no transition relation - added incrementally).
            emit_bmc_incremental_preamble,
            // Incremental query builder: per-property assertions only.
            build_incremental_query,
            // Incremental result handler.
            handle_result,
            // Fallback: batch query builder (full script per depth).
            build_batch_query,
            // Fallback: batch result handler.
            fallback_handle_result,
        )
    };

    ReachabilityBmcRunReport {
        depth,
        solve_profile: state.last_solve_profile,
    }
}

fn env_flag_enabled(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => default,
    }
}

fn bmc_depth1_chunking_enabled() -> bool {
    // Default-ON: under a tight deadline a large all-property depth-1 batch is
    // exactly the call that times out wholesale, abandoning every pending
    // property. Chunking solves the same depth-1 queries in small groups so a
    // single slow chunk no longer sinks the rest. Verdict-preserving: each chunk
    // still routes SAT results through replay validation and leaves Unknowns
    // unresolved. The explicit disable env still wins for benchmarking.
    env_flag_enabled(ENABLE_BMC_DEPTH1_CHUNKING_ENV, true)
        && !env_flag_enabled(DISABLE_BMC_DEPTH1_CHUNKING_ENV, false)
}

fn bmc_depth1_chunk_size() -> usize {
    std::env::var(BMC_DEPTH1_CHUNK_SIZE_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(BMC_DEPTH1_CHUNK_SIZE_DEFAULT)
}

fn bmc_deadline_incremental_enabled() -> bool {
    // Default-ON: under a deadline, the incremental BMC path reuses a single ay
    // process across the depth ladder (push/pop per property, transition relation
    // added incrementally so learned clauses carry forward) instead of re-encoding
    // and respawning ay at every depth. That reaches the same — or deeper — BMC
    // depths within the same time budget, so more reachability properties get
    // seeded before the contest timeout.
    //
    // Verdict-preserving: the incremental path routes every depth through the SAME
    // `handle_result`/`apply_results` closures as batch, so the only verdict BMC
    // seeds (SAT = EF witness / AG counterexample) is still replay-validated
    // against the real net before it resolves a tracker. UNSAT seeds nothing, so
    // any clause carryover can at worst leave a property unresolved (handed to
    // k-induction / BFS), never flip a verdict. The parity test
    // `test_incremental_bmc_matches_batch_verdicts_and_depth` pins identical
    // verdicts + depth against the real solver. The explicit env override
    // (`TY_MCC_AY_BMC_DEADLINE_INCREMENTAL=0`) still forces the batch fallback.
    env_flag_enabled(AY_BMC_DEADLINE_INCREMENTAL_ENV, true)
}

fn should_chunk_depth1_all_property_bmc(
    deadline: Option<Instant>,
    depths: &[usize],
    pending_len: usize,
) -> bool {
    deadline.is_some()
        && depths == BMC_DEPTH_ONE_LADDER
        && pending_len >= BMC_LARGE_PENDING_THRESHOLD
        && bmc_depth1_chunking_enabled()
}

fn should_retry_depth_individually(deadline: Option<Instant>, pending_len: usize) -> bool {
    if pending_len <= 1 {
        return false;
    }
    let Some(deadline) = deadline else {
        return true;
    };
    let now = Instant::now();
    // Previously a large pending queue under a short (<=30s) deadline gave up
    // wholesale here, abandoning EVERY pending property on a single failed
    // multi-property batch call. That starves the per-property witness lane
    // that can still seed cheap EF=TRUE / AG=FALSE verdicts at depth 1. Instead,
    // always fall back to per-property retries whenever the fair per-property
    // time-slice clears the minimum solve budget; `bmc_split_retry_timeout`
    // already divides the non-reserved budget across the pending properties, so
    // a queue too large to slice meaningfully is still gated below. This is
    // verdict-preserving: per-property SAT results are replay-validated before
    // they seed a verdict, and Unknowns simply remain unresolved.
    bmc_split_retry_timeout(Some(deadline), pending_len, now) >= BMC_SPLIT_RETRY_MIN_BUDGET
}

fn bmc_split_retry_timeout(
    deadline: Option<Instant>,
    retries_left: usize,
    now: Instant,
) -> Duration {
    let Some(deadline) = deadline else {
        return PER_DEPTH_TIMEOUT;
    };
    let retries_left = retries_left.max(1);
    let remaining = deadline.saturating_duration_since(now);
    let available = remaining.saturating_sub(BMC_SPLIT_RETRY_FALLBACK_RESERVE);
    PER_DEPTH_TIMEOUT.min(available / retries_left as u32)
}

fn bmc_depth_ladder(deadline: Option<Instant>, pending_len: usize) -> &'static [usize] {
    let Some(deadline) = deadline else {
        return DEPTH_LADDER;
    };
    if pending_len >= BMC_LARGE_PENDING_THRESHOLD
        && deadline.saturating_duration_since(Instant::now())
            <= BMC_SHORT_DEADLINE_MULTI_PROPERTY_LIMIT
    {
        BMC_DEPTH_ONE_LADDER
    } else {
        DEPTH_LADDER
    }
}

fn bmc_sat_verdict(quantifier: PathQuantifier) -> bool {
    match quantifier {
        PathQuantifier::EF => true,
        PathQuantifier::AG => false,
    }
}

fn validate_bmc_sat_witness(
    ay_path: &std::path::Path,
    net: &PetriNet,
    tracker: &PropertyTracker,
    target: &WitnessValidationTarget,
    depth: usize,
    deadline: Option<Instant>,
) -> Result<(), String> {
    let timeout = deadline
        .map(|global_deadline| {
            PER_DEPTH_TIMEOUT.min(global_deadline.saturating_duration_since(Instant::now()))
        })
        .unwrap_or(PER_DEPTH_TIMEOUT);
    if timeout < BMC_SPLIT_RETRY_MIN_BUDGET {
        return Err("insufficient model validation budget".to_string());
    }

    let script = encode_bmc_model_script(net, tracker, depth);
    let model = run_ay_bool_model(ay_path, &script, timeout)
        .ok_or_else(|| "solver did not return a parseable SAT model".to_string())?;

    validate_bmc_model_replay(net, tracker, target, depth, &model)
}

fn validate_bmc_model_replay(
    net: &PetriNet,
    tracker: &PropertyTracker,
    target: &WitnessValidationTarget,
    depth: usize,
    model: &SolverBoolModel,
) -> Result<(), String> {
    let trace = decode_bmc_model_trace(net, depth, model)?;
    let markings = replay_bmc_trace(net, &trace)?;

    for (step, marking) in markings.iter().enumerate() {
        let predicate_holds = eval_predicate(&target.original_predicate, marking, net);
        let validates_property = match tracker.quantifier {
            PathQuantifier::EF => predicate_holds,
            PathQuantifier::AG => !predicate_holds,
        };
        if validates_property {
            return Ok(());
        }
        if step >= depth {
            break;
        }
    }

    Err("replayed trace does not satisfy the SAT target predicate".to_string())
}

/// Decode a SAT model's `stay_*`/`fire_*_*` assignments into a concrete BMC
/// firing sequence.
///
/// Shared with sibling SAT->verdict lanes (e.g. QuasiLiveness) so every
/// symbolic SAT can be replayed on the original net before it commits a
/// definite verdict, rejecting spurious solver SATs.
pub(in crate::examinations) fn decode_bmc_model_trace(
    net: &PetriNet,
    depth: usize,
    model: &SolverBoolModel,
) -> Result<Vec<BmcTraceStep>, String> {
    let mut trace = Vec::with_capacity(depth);
    for step in 0..depth {
        let stay_name = format!("stay_{step}");
        let stay = model
            .bool_value(&stay_name)
            .ok_or_else(|| format!("missing {stay_name}"))?;
        let mut fired = Vec::new();
        for transition in 0..net.num_transitions() {
            let fire_name = format!("fire_{}_{}", step, transition);
            let selected = model
                .bool_value(&fire_name)
                .ok_or_else(|| format!("missing {fire_name}"))?;
            if selected {
                fired.push(TransitionIdx(transition as u32));
            }
        }

        match (stay, fired.as_slice()) {
            (true, []) => trace.push(BmcTraceStep::Stay),
            (false, [transition]) => trace.push(BmcTraceStep::Fire(*transition)),
            (false, []) => return Err(format!("step {step} selects no transition or stutter")),
            _ => return Err(format!("step {step} selects multiple transition choices")),
        }
    }
    Ok(trace)
}

/// Replay a decoded BMC firing sequence on the original net, returning the
/// sequence of visited markings (initial marking first).
///
/// Rejects any model that fires a transition disabled at the marking it claims
/// to fire from — the most common spurious-SAT shape. Shared with sibling
/// SAT->verdict lanes (e.g. QuasiLiveness).
pub(in crate::examinations) fn replay_bmc_trace(
    net: &PetriNet,
    trace: &[BmcTraceStep],
) -> Result<Vec<Vec<u64>>, String> {
    let mut marking = net.initial_marking.clone();
    let mut markings = Vec::with_capacity(trace.len() + 1);
    markings.push(marking.clone());

    for (step, action) in trace.iter().enumerate() {
        match *action {
            BmcTraceStep::Stay => {}
            BmcTraceStep::Fire(transition) => {
                if !net.is_enabled(&marking, transition) {
                    return Err(format!(
                        "step {step} fires disabled transition {}",
                        transition.0
                    ));
                }
                // Fail-closed (#22): token-count overflow means the trace
                // marking is not representable — reject the trace.
                net.apply_delta(&mut marking, transition)
                    .map_err(|e| format!("step {step} overflows place token count: {e}"))?;
            }
        }
        markings.push(marking.clone());
    }

    Ok(markings)
}

/// Generate the SMT-LIB script for BMC at a given depth.
///
/// Produces one shared declaration/transition block, then per-property
/// `push/assert/check-sat/pop` blocks. Each step can either fire one
/// transition or stutter (no-op).
fn encode_bmc_script(
    net: &PetriNet,
    trackers: &[PropertyTracker],
    pending: &[usize],
    depth: usize,
) -> String {
    let mut script = String::with_capacity(4096);
    super::bmc_runner::emit_bmc_preamble(&mut script, net, depth);

    for &property_idx in pending {
        let tracker = &trackers[property_idx];
        script.push_str("(push 1)\n");
        script.push_str(&encode_property_assertion(net, tracker, depth));
        script.push_str("(check-sat)\n");
        script.push_str("(pop 1)\n");
    }

    script.push_str("(exit)\n");
    script
}

fn encode_bmc_model_script(net: &PetriNet, tracker: &PropertyTracker, depth: usize) -> String {
    let mut script = String::with_capacity(4096);
    script.push_str("(set-option :produce-models true)\n");
    super::bmc_runner::emit_bmc_preamble(&mut script, net, depth);
    script.push_str(&encode_property_assertion(net, tracker, depth));
    script.push_str("(check-sat)\n");
    append_bmc_step_value_query(&mut script, net, depth);
    script.push_str("(exit)\n");
    script
}

/// Append a `(get-value (...))` query requesting every `stay_*`/`fire_*_*`
/// decision variable for steps `0..depth`, so the SAT model can be decoded into
/// a concrete firing sequence. Shared with sibling SAT->verdict lanes.
pub(in crate::examinations) fn append_bmc_step_value_query(
    script: &mut String,
    net: &PetriNet,
    depth: usize,
) {
    if depth == 0 {
        return;
    }

    script.push_str("(get-value (");
    let mut first = true;
    for step in 0..depth {
        append_get_value_symbol(script, &mut first, &format!("stay_{step}"));
        for transition in 0..net.num_transitions() {
            append_get_value_symbol(script, &mut first, &format!("fire_{}_{}", step, transition));
        }
    }
    script.push_str("))\n");
}

fn append_get_value_symbol(script: &mut String, first: &mut bool, symbol: &str) {
    if *first {
        *first = false;
    } else {
        script.push(' ');
    }
    script.push_str(symbol);
}

fn encode_property_assertion(net: &PetriNet, tracker: &PropertyTracker, depth: usize) -> String {
    let check_negated = tracker.quantifier == PathQuantifier::AG;
    let mut step_assertions = Vec::new();
    for step in 0..=depth {
        let predicate = encode_predicate(&tracker.predicate, step, net);
        if check_negated {
            step_assertions.push(format!("(not {})", predicate));
        } else {
            step_assertions.push(predicate);
        }
    }

    let mut assertion = String::new();
    if step_assertions.len() == 1 {
        assertion.push_str(&format!("(assert {})\n", step_assertions[0]));
    } else {
        assertion.push_str("(assert (or");
        for step_assertion in &step_assertions {
            assertion.push_str(&format!(" {}", step_assertion));
        }
        assertion.push_str("))\n");
    }
    assertion
}

/// Build per-property assertion strings for incremental BMC at a given depth.
///
/// Each returned string is a self-contained assertion block (without push/pop or
/// check-sat) that the incremental solver wraps in a push/pop scope.
fn encode_property_assertions(
    net: &PetriNet,
    trackers: &[PropertyTracker],
    pending: &[usize],
    depth: usize,
) -> Vec<String> {
    pending
        .iter()
        .map(|&property_idx| {
            let tracker = &trackers[property_idx];
            encode_property_assertion(net, tracker, depth)
        })
        .collect()
}

#[cfg(test)]
#[path = "reachability_bmc_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "reachability_bmc_incremental_tests.rs"]
mod incremental_tests;
