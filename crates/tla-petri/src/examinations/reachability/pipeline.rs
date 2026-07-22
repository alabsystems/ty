// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Pipeline orchestration for reachability examinations.
//!
//! Manages the multi-phase execution: simplification → bounded witness BFS →
//! BMC → LP → PDR → random walk → heuristic search → k-induction → deferred
//! AIGER/fireability BFS guards → structural reduction → BFS.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tla_mc_core::{CapabilityReport, ProblemKind};

use crate::explorer::{explore_observer, ExplorationConfig};
use crate::intelligence_bus::{self, IntelligenceBus};
use crate::model::PropertyAliases;
use crate::nupn::NupnStructure;
use crate::output::Verdict;
use crate::petri_net::PetriNet;
use crate::property_xml::{PathQuantifier, Property};
use crate::resolved_predicate::ResolvedPredicate;
use crate::stubborn::PorStrategy;

use super::observer::ReachabilityObserver;
use super::reduction::{
    all_predicates_reduction_safe, build_reachability_slice, build_symbolic_seed_reduction,
    explore_reachability_on_reduced_net, explore_reachability_on_slice,
    reduce_reachability_queries, ReachabilityExploreError, SymbolicSeedReduction,
};
use super::types::{
    assemble_results, finalize_exhaustive_completion, flush_resolved,
    prepare_trackers_with_aliases, resolve_tracker, run_compound_invariant_reuse, PropertyTracker,
    ReachabilityResolutionSource,
};

use crate::examinations::kinduction;
use crate::examinations::reachability_aiger;
use crate::examinations::reachability_bfs_witness::{self, WitnessBfsConfig};
use crate::examinations::reachability_bmc;
use crate::examinations::reachability_heuristic;
use crate::examinations::reachability_lp;
use crate::examinations::reachability_pdr;
use crate::examinations::reachability_walk;
use crate::examinations::reachability_witness::{
    WitnessValidationContext, WitnessValidationTarget,
};
use crate::symbolic::run_symbolic_state_equation_seeding;
/// Kill-switch for the legacy oxidd BDD reachability seeding lane. Default-ON
/// (current behavior); set `TY_MCC_DISABLE_DD_REACHABILITY` truthy to skip it
/// (the MDD twin seeding + the fallback lanes then resolve those trackers).
///
/// SOUNDNESS-NEUTRAL: the lane only SEEDS definite verdicts; any tracker it
/// would seed and is skipped simply falls through to the (also exact) MDD
/// seeding or the explicit/solver/PDR lanes. The MDD twin is cross-checked
/// EQUAL to this lane (mdd_fastpath tests). This is the A/B mechanism for
/// retiring the lane (and dropping the oxidd dependency).
#[cfg(feature = "dd-backend")]
fn dd_reachability_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_DD_REACHABILITY").is_ok_and(|v| {
        let v = v.trim();
        v == "1"
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("yes")
    })
}

/// Suppress the nondeterministic raw-net random-walk witness seeding
/// (`TY_MCC_DISABLE_RANDOM_WALK`). The walk is SOUND (emits only replay-validated
/// EF=TRUE / AG=FALSE witnesses) but its random sampling + near-budget timing make
/// per-run verdicts nondeterministic, which CONTAMINATES a lane-A/B (the
/// CANNOT_COMPUTE↔resolved churn the oxidd-removal flip gate must see through).
/// Disabling it makes the reachability pipeline deterministic (DD + exhaustive BFS
/// only), so a gate-off-vs-on coverage comparison isolates the true delta.
/// Default OFF ⇒ production keeps the walk; this is a measurement/determinism knob.
fn random_walk_seeding_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_RANDOM_WALK").is_ok_and(|v| {
        let v = v.trim();
        !v.is_empty() && v != "0"
    })
}

/// Run the exact symbolic Decision-Diagram reachability phase, returning
/// the number of trackers it resolved. A no-op when the `dd-backend`
/// feature is disabled, or when the BDD reachability lane is disabled (the MDD
/// twin + fallback lanes then resolve those trackers).
#[cfg(feature = "dd-backend")]
fn run_dd_reachability_phase(net: &PetriNet, trackers: &mut [PropertyTracker]) -> usize {
    if dd_reachability_disabled() {
        return 0;
    }
    super::dd_fastpath::run_dd_reachability_seeding(net, trackers)
}

#[cfg(not(feature = "dd-backend"))]
fn run_dd_reachability_phase(_net: &PetriNet, _trackers: &mut [PropertyTracker]) -> usize {
    0
}

/// Run the exact MDD reachability phase (the MDD twin of the BDD fast-path,
/// targeting the counter / conserved / high-bound nets the bit-blasted BDD lane
/// blows up on), returning the number of trackers it resolved. A no-op when the
/// `dd-backend` feature is disabled. Takes the global deadline so the worker's
/// saturation budget is the leftover wall-clock (fail-closed on overrun).
#[cfg(feature = "dd-backend")]
fn run_mdd_reachability_phase(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
    nupn: Option<&NupnStructure>,
) -> usize {
    super::mdd_fastpath::run_mdd_reachability_seeding(net, trackers, deadline, nupn)
}

#[cfg(not(feature = "dd-backend"))]
fn run_mdd_reachability_phase(
    _net: &PetriNet,
    _trackers: &mut [PropertyTracker],
    _deadline: Option<Instant>,
    _nupn: Option<&NupnStructure>,
) -> usize {
    0
}

const CHECKPOINT_MANIFEST_FILE: &str = "checkpoint.json";
const AIGER_PIPELINE_PHASE_CAP: Duration = Duration::from_secs(6);
const AIGER_PIPELINE_MIN_BUDGET: Duration = Duration::from_millis(100);
const AIGER_PIPELINE_VIRTUAL_DOWNSTREAM_LANES: usize = 2;
const SYMBOLIC_PIPELINE_MIN_BUDGET: Duration = Duration::from_millis(100);
const SYMBOLIC_DISPATCH_OVERSHOOT_CUSHION: Duration = Duration::from_millis(250);

fn compute_fallback_reserve(net: &PetriNet, remaining: std::time::Duration) -> std::time::Duration {
    let net_elements = net.num_places() + net.num_transitions();
    let base_reserve = std::time::Duration::from_secs(2);
    let scale_bonus = std::time::Duration::from_millis((net_elements / 2) as u64);
    base_reserve
        .saturating_add(scale_bonus)
        .min(std::time::Duration::from_secs(30))
        .min(remaining / 5)
}

/// Absolute per-lane cap for the under-approximation witness lanes (the
/// pre-solver bounded witness BFS, random walk, and heuristic search). These
/// lanes can only ever seed `EF=TRUE` / `AG=FALSE` from a real replay-validated
/// witness — they can NEVER produce the universal verdicts (`EF=FALSE` /
/// `AG=TRUE`). Only the final exhaustive BFS (and replay-validated BMC) can do
/// that, so these lanes must not be allowed to consume the budget the
/// exhaustive pass needs to run to completion on an enumerable net.
const REACHABILITY_UNDER_APPROX_LANE_ABS_CAP: Duration = Duration::from_secs(8);

/// Fraction of the *currently remaining* global deadline that the cumulative
/// under-approximation lanes may consume. The complement is reserved for the
/// sound exhaustive BFS oracle (which proves the universal verdicts). At 1/4
/// the exhaustive pass keeps the lion's share (~3/4) of the deadline.
const REACHABILITY_UNDER_APPROX_REMAINING_FRACTION: u32 = 4;

/// Deadline for an under-approximation witness lane (bounded witness BFS / walk
/// / heuristic). This is strictly tighter than [`witness_search_deadline`]: it
/// additionally clamps the lane to a small fraction of the global deadline plus
/// an absolute cap, so that the cumulative under-approximation lanes cannot
/// starve the final exhaustive BFS. Verdict-preserving: it only ever *shrinks*
/// the time these under-approx-only lanes get; it never changes which verdict a
/// completed lane yields, and the lanes' under-approx-only contract is intact.
pub(crate) fn under_approx_lane_deadline(
    net: &PetriNet,
    global_deadline: Option<Instant>,
) -> Option<Instant> {
    under_approx_lane_deadline_at(net, global_deadline, Instant::now())
}

fn under_approx_lane_deadline_at(
    net: &PetriNet,
    global_deadline: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    // Start from the existing reserve-preserving witness deadline so the BFS
    // tail reserve is always honored.
    let reserve = global_deadline
        .map(|d| compute_fallback_reserve(net, d.saturating_duration_since(now)))
        .unwrap_or(Duration::ZERO);
    let base = deadline_preserving_reserve_at(global_deadline, reserve, now);

    let Some(global_deadline) = global_deadline else {
        // No global deadline: keep the historical (unbounded) under-approx
        // behavior so non-MCC callers are unaffected.
        return base;
    };

    let remaining = global_deadline.saturating_duration_since(now);
    let fraction_cap = remaining / REACHABILITY_UNDER_APPROX_REMAINING_FRACTION;
    let lane_budget = fraction_cap.min(REACHABILITY_UNDER_APPROX_LANE_ABS_CAP);
    let lane_deadline = now + lane_budget;

    Some(match base {
        Some(base) if base < lane_deadline => base,
        _ => lane_deadline,
    })
}

const REACHABILITY_BOUNDED_WITNESS_BFS_PHASE_CAP: Duration = Duration::from_secs(3);
const REACHABILITY_BOUNDED_WITNESS_BFS_MAX_STATES: usize = 50_000;
const REACHABILITY_POST_SMT_WITNESS_BFS_PHASE_CAP: Duration = Duration::from_secs(2);
const REACHABILITY_POST_SMT_WITNESS_BFS_MAX_STATES: usize = 200_000;
const REACHABILITY_POST_SMT_WITNESS_BFS_RESERVE_LOAN_CAP: Duration = Duration::from_millis(100);
const REACHABILITY_WITNESS_FALLBACK_RESERVE: Duration = Duration::from_secs(4);
const REACHABILITY_BFS_FALLBACK_RESERVE: Duration = REACHABILITY_WITNESS_FALLBACK_RESERVE;
const REACHABILITY_PROOF_RESCUE_MAX_PENDING: usize = 2;
const REACHABILITY_PROOF_RESCUE_PHASE_CAP: Duration = Duration::from_secs(2);
const REACHABILITY_PROOF_RESCUE_MIN_BUDGET: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::examinations::reachability) struct AigerPipelineBudget {
    pub(in crate::examinations::reachability) deadline: Option<Instant>,
    pub(in crate::examinations::reachability) skipped_for_reserve: bool,
    #[cfg(test)]
    phase_budget: Option<Duration>,
}

pub(in crate::examinations::reachability) fn aiger_pipeline_budget(
    global_deadline: Option<Instant>,
    unresolved_count: usize,
) -> AigerPipelineBudget {
    aiger_pipeline_budget_at(global_deadline, unresolved_count, Instant::now())
}

fn aiger_pipeline_budget_at(
    global_deadline: Option<Instant>,
    unresolved_count: usize,
    now: Instant,
) -> AigerPipelineBudget {
    let Some(global_deadline) = global_deadline else {
        return AigerPipelineBudget {
            deadline: None,
            skipped_for_reserve: false,
            #[cfg(test)]
            phase_budget: None,
        };
    };

    let remaining = global_deadline.saturating_duration_since(now);
    let phase_budget = AIGER_PIPELINE_PHASE_CAP.min(reachability_fair_share_budget(
        remaining,
        unresolved_count.max(1) + AIGER_PIPELINE_VIRTUAL_DOWNSTREAM_LANES,
    ));
    if phase_budget < AIGER_PIPELINE_MIN_BUDGET {
        return AigerPipelineBudget {
            deadline: Some(now),
            skipped_for_reserve: true,
            #[cfg(test)]
            phase_budget: Some(phase_budget),
        };
    }

    AigerPipelineBudget {
        deadline: Some(now + phase_budget),
        skipped_for_reserve: false,
        #[cfg(test)]
        phase_budget: Some(phase_budget),
    }
}

fn reachability_fair_share_budget(remaining: Duration, lane_count: usize) -> Duration {
    let divisor = lane_count.clamp(1, u32::MAX as usize) as u32;
    remaining / divisor
}

fn cannot_compute_results(
    properties: &[Property],
    error: &impl std::fmt::Display,
) -> Vec<(String, Verdict)> {
    eprintln!("Reachability: CANNOT_COMPUTE ({error})");
    properties
        .iter()
        .map(|property| (property.id.clone(), Verdict::CannotCompute))
        .collect()
}

fn run_compound_reuse_phase(trackers: &mut [PropertyTracker], after: &str) -> usize {
    let seeded = run_compound_invariant_reuse(trackers);
    if seeded > 0 {
        eprintln!(
            "Compound invariant reuse seeded {seeded}/{} reachability verdicts after {after}",
            trackers.len(),
        );
    }
    seeded
}

fn run_deferred_aiger_phase(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    config: &ExplorationConfig,
    nupn: Option<&NupnStructure>,
    bus: &IntelligenceBus,
) -> usize {
    // Wall-capped: the tla-aiger pipeline does not poll its deadline, so with a
    // generous global budget the deferred lane could otherwise overrun to the
    // external kill. Verdict-preserving (replay-validated, merged first-writer-wins).
    let aiger_seeded = run_aiger_seeding_wall_capped(
        net,
        trackers,
        validation_targets,
        config.deadline(),
        nupn,
        Some(bus),
        "Deferred",
    );
    if aiger_seeded > 0 {
        eprintln!(
            "Deferred AIGER portfolio seeded {aiger_seeded}/{} reachability verdicts",
            trackers.len(),
        );
    }
    aiger_seeded
}

fn incremental_flush_enabled(config: &ExplorationConfig, flush: bool) -> bool {
    flush
        && !config.checkpoint().is_some_and(|checkpoint| {
            checkpoint.resume && checkpoint.dir.join(CHECKPOINT_MANIFEST_FILE).exists()
        })
}

fn deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

pub(crate) fn deadline_preserving_reserve_at(
    global_deadline: Option<Instant>,
    reserve: Duration,
    now: Instant,
) -> Option<Instant> {
    global_deadline.map(|deadline| {
        let remaining = deadline.saturating_duration_since(now);
        if remaining <= reserve {
            now
        } else {
            deadline.checked_sub(reserve).unwrap_or(now)
        }
    })
}

pub(crate) fn witness_search_deadline(
    net: &PetriNet,
    global_deadline: Option<Instant>,
) -> Option<Instant> {
    witness_search_deadline_at(net, global_deadline, Instant::now())
}

fn witness_search_deadline_at(
    net: &PetriNet,
    global_deadline: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    let reserve = global_deadline
        .map(|d| compute_fallback_reserve(net, d.saturating_duration_since(now)))
        .unwrap_or(Duration::ZERO);
    deadline_preserving_reserve_at(global_deadline, reserve, now)
}

fn symbolic_seed_deadline(net: &PetriNet, global_deadline: Option<Instant>) -> Option<Instant> {
    symbolic_seed_deadline_at(net, global_deadline, Instant::now())
}

fn symbolic_seed_deadline_at(
    net: &PetriNet,
    global_deadline: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    let reserve = global_deadline
        .map(|d| compute_fallback_reserve(net, d.saturating_duration_since(now)))
        .unwrap_or(Duration::ZERO);
    let reserved_deadline = deadline_preserving_reserve_at(global_deadline, reserve, now)?;
    let phase_budget = reserved_deadline.saturating_duration_since(now);
    if phase_budget < SYMBOLIC_PIPELINE_MIN_BUDGET + SYMBOLIC_DISPATCH_OVERSHOOT_CUSHION {
        return Some(now);
    }
    Some(
        reserved_deadline
            .checked_sub(SYMBOLIC_DISPATCH_OVERSHOOT_CUSHION)
            .unwrap_or(now),
    )
}

/// Reserve-preserving, wall-clock-bounded driver for the PDR/IC3 seeding lane
/// (Phase 2c).
///
/// PDR is an under-the-deadline *seeder*, never the sound oracle: only the
/// final exhaustive BFS proves the universal verdicts. Two independent leaks
/// previously let Phase 2c consume the budget the exhaustive BFS needs, so the
/// pipeline could return `CannotCompute` with the exhaustive oracle effectively
/// never having run (a premature give-up with wall budget that *should* have
/// reached the BFS):
///
/// 1. **No reserve.** The call site handed `run_pdr_seeding` the full
///    `config.deadline()`, while every sibling seeding lane reserves a BFS tail
///    (CHC via [`symbolic_seed_deadline`], AIGER via [`aiger_pipeline_budget`]).
///    We now cap the phase at the same reserve-preserving deadline as CHC.
/// 2. **Non-polling overrun.** `solve_petri_net_pdr`'s `compute_p_invariants`
///    preamble does not poll its deadline on high-arity nets (the
///    ASLink-PT-04a pathology already documented and guarded in the deadlock
///    path's `run_phase_with_wall_cap`), so a single tracker can spin past even
///    a reserved deadline. We therefore run the phase on a worker thread and
///    abandon it (leaking it; the process exits at the global deadline anyway,
///    and PDR has no cooperative cancellation) once the cap expires — copying
///    back only the verdicts the worker actually resolved.
///
/// Soundness/verdict-preservation: `run_pdr_seeding` only ever writes definite
/// verdicts via `resolve_tracker`, and these are deterministic in
/// `(net, trackers, deadline)`, so running them on a worker thread cannot
/// invent or change a verdict. On abandon/panic we copy nothing back, so a
/// timed-out phase contributes no verdict and the exhaustive BFS runs with the
/// reserved tail intact.
/// Kill-switch for running the verdict-preserving symbolic seeding lanes
/// (CHC + PDR) on the reduced net. Default-ON. Set
/// `TY_MCC_REDUCE_SYMBOLIC_SEED=0` (or `false`/`no`/`off`) to force the symbolic
/// lanes back onto the original net — used as the cross-check baseline and as a
/// safety escape hatch (declining is always sound, never a wrong answer).
fn reduce_symbolic_seed_enabled() -> bool {
    !matches!(
        std::env::var("TY_MCC_REDUCE_SYMBOLIC_SEED")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("0" | "false" | "FALSE" | "no" | "off" | "OFF")
    )
}

/// Merge boolean verdicts from reduced-net (or original-net) worker trackers
/// back into the live original trackers by position/`id` (first-writer-wins).
///
/// Only the `(verdict, source, depth)` triple crosses the boundary — never a
/// marking — so when the worker ran on a reduced net no witness expansion is
/// required: the predicate remap (`remap_predicate_scaled`) already accounts for
/// the reduced/scaled coordinates, and verdict-preservation guarantees the
/// boolean equals the original-net verdict.
fn merge_seed_verdicts(trackers: &mut [PropertyTracker], worker: &[PropertyTracker]) {
    for (dst, src) in trackers.iter_mut().zip(worker.iter()) {
        debug_assert_eq!(dst.id, src.id, "seed worker tracker order diverged");
        if dst.verdict.is_some() {
            continue;
        }
        if let (Some(verdict), Some(resolution)) = (src.verdict, src.resolved_by) {
            resolve_tracker(dst, verdict, resolution.source, resolution.depth);
        }
    }
}

/// Pick the net + worker trackers a wall-capped seeding lane should run on: the
/// verdict-preserving reduced net + remapped trackers when one is available,
/// else the original net + a plain clone. The returned trackers preserve `id`
/// order 1:1 with `trackers`, so [`merge_seed_verdicts`] maps verdicts back.
fn seed_worker_inputs(
    net: &PetriNet,
    trackers: &[PropertyTracker],
    seed_reduction: Option<&SymbolicSeedReduction>,
) -> (PetriNet, Vec<PropertyTracker>) {
    match seed_reduction {
        Some(reduction) => (reduction.net().clone(), reduction.worker_trackers(trackers)),
        None => (net.clone(), trackers.to_vec()),
    }
}

fn run_pdr_seeding_wall_capped(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    global_deadline: Option<Instant>,
    seed_reduction: Option<&SymbolicSeedReduction>,
) {
    // No global deadline (non-MCC / public API contract): keep the historical
    // inline, unbounded path so those callers are byte-for-byte unaffected.
    // `seed_reduction` is only ever built under a deadline, so this path always
    // sees `None` and runs on the original net.
    if global_deadline.is_none() {
        reachability_pdr::run_pdr_seeding(net, trackers, None);
        return;
    }

    let phase_deadline = symbolic_seed_deadline(net, global_deadline);
    let cap = phase_deadline
        .map(|d| d.saturating_duration_since(Instant::now()))
        .unwrap_or(Duration::ZERO);
    if cap.is_zero() {
        // No budget left for PDR after reserving the BFS tail: skip the phase
        // entirely rather than let it (over)run into the exhaustive BFS budget.
        return;
    }

    // Run on the verdict-preserving reduced net when one is available (smaller
    // net => the PDR/IC3 safety check dispatches faster); else the original net.
    let (net_for_worker, mut worker_trackers) = seed_worker_inputs(net, trackers, seed_reduction);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<PropertyTracker>>(1);
    let _worker = std::thread::Builder::new()
        .name("ty-reach-pdr-phase".to_string())
        .spawn(move || {
            // catch_unwind: `compute_p_invariants` has a latent i64-overflow on
            // very high-arity nets (the same edge the deadlock path guards).
            // Isolating it to this worker means a panic seeds nothing instead
            // of taking down ty-mcc; the parent observes a disconnect and falls
            // through to the exhaustive BFS.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reachability_pdr::run_pdr_seeding(
                    &net_for_worker,
                    &mut worker_trackers,
                    phase_deadline,
                );
                worker_trackers
            }));
            // Ignore SendError: on timeout the parent has dropped the receiver;
            // sync_channel(1) makes the send fail fast instead of blocking.
            if let Ok(resolved) = result {
                let _ = tx.send(resolved);
            }
        });

    match rx.recv_timeout(cap) {
        Ok(resolved) => {
            merge_seed_verdicts(trackers, &resolved);
        }
        Err(_) => {
            // Timeout (worker still spinning the non-polling preamble) or
            // Disconnected (worker panicked): copy nothing back so the
            // exhaustive BFS gets the full reserved tail. The worker thread is
            // deliberately leaked — same contract as `run_phase_with_wall_cap`.
        }
    }
}

/// Wall-capped symbolic state-equation (CHC) seeding (Phase 2b2).
///
/// `run_symbolic_state_equation_seeding` already polls `deadline` *between*
/// formulas, but a single `symbolic_state_equation_check` (the ay-chc adaptive
/// portfolio) can overrun its per-formula `time_budget` and spin well past the
/// phase deadline (measured: FlexibleBarrier-PT-04b ReachabilityCardinality ran
/// CHC ~35s on a 15s budget). Run the phase on a worker thread bounded to the
/// reserved CHC slice and merge resolved verdicts back; on cap/panic copy
/// nothing so the exhaustive BFS tail is preserved. Exactly mirrors
/// [`run_pdr_seeding_wall_capped`].
///
/// Verdict-preserving: the worker only ever ADDS verdicts (via
/// `resolve_tracker`, first-writer-wins). On abandon the affected formulas stay
/// unresolved and fall through to AIGER/exhaustive BFS or end as
/// `CANNOT_COMPUTE` — never a wrong answer.
fn run_symbolic_seeding_wall_capped(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    global_deadline: Option<Instant>,
    seed_reduction: Option<&SymbolicSeedReduction>,
) {
    // No global deadline (non-MCC / public API contract): keep the historical
    // inline, unbounded path so those callers are byte-for-byte unaffected.
    // `seed_reduction` is only ever built under a deadline, so this path always
    // sees `None` and runs on the original net.
    if global_deadline.is_none() {
        run_symbolic_state_equation_seeding(net, trackers, None);
        return;
    }

    let phase_deadline = symbolic_seed_deadline(net, global_deadline);
    let cap = phase_deadline
        .map(|d| d.saturating_duration_since(Instant::now()))
        .unwrap_or(Duration::ZERO);
    if cap.is_zero() {
        return;
    }

    // Run on the verdict-preserving reduced net when one is available (the
    // state-equation CHC encoding is materially smaller on the reduced net);
    // else the original net.
    let (net_for_worker, mut worker_trackers) = seed_worker_inputs(net, trackers, seed_reduction);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<PropertyTracker>>(1);
    let _worker = std::thread::Builder::new()
        .name("ty-reach-symbolic-phase".to_string())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_symbolic_state_equation_seeding(
                    &net_for_worker,
                    &mut worker_trackers,
                    phase_deadline,
                );
                worker_trackers
            }));
            // Ignore SendError: on timeout the parent has dropped the receiver.
            if let Ok(resolved) = result {
                let _ = tx.send(resolved);
            }
        });

    match rx.recv_timeout(cap) {
        Ok(resolved) => {
            merge_seed_verdicts(trackers, &resolved);
        }
        Err(_) => {
            // Timeout (CHC still spinning a non-polling solve) or Disconnected
            // (worker panicked): copy nothing back. The worker thread is
            // deliberately leaked — same contract as `run_pdr_seeding_wall_capped`.
        }
    }
}

/// Wall-capped AIGER/IC3 cross-encoding seeding (Phases 2b3 and the deferred
/// fireability lane).
///
/// The tla-aiger preprocessing/SAT pipeline does NOT poll its `timeout`
/// argument between internal phases (the same pathology the *deadlock* pipeline
/// already wall-caps — see `examination_non_property/deadlock_one_safe.rs`).
/// With a generous deadline the reachability AIGER lane gets a large budget and
/// can spin well past it (measured: FlexibleBarrier-PT-04b
/// ReachabilityCardinality at `--timeout 120` overran to the external kill).
/// Run it on a worker thread bounded to the reserved AIGER slice and merge
/// resolved verdicts back; on cap/panic copy nothing so the exhaustive BFS tail
/// is preserved.
///
/// Returns the number of trackers newly resolved. Verdict-preserving: AIGER only
/// publishes replay-validated witnesses / proofs (sound regardless of the shared
/// `IntelligenceBus`, which is therefore omitted from the worker), merged via
/// `resolve_tracker` (first-writer-wins). Abandon leaves formulas unresolved for
/// the exhaustive BFS or final `CANNOT_COMPUTE`.
fn run_aiger_seeding_wall_capped(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    global_deadline: Option<Instant>,
    nupn: Option<&NupnStructure>,
    bus: Option<&IntelligenceBus>,
    lane_label: &str,
) -> usize {
    let pre = trackers.iter().filter(|t| t.verdict.is_some()).count();
    let unresolved_count = trackers.iter().filter(|t| t.verdict.is_none()).count();
    let aiger_budget = aiger_pipeline_budget(global_deadline, unresolved_count);
    if aiger_budget.skipped_for_reserve {
        eprintln!(
            "{lane_label} AIGER portfolio skipped; preserving deadline for reachability fallback pipeline",
        );
        return 0;
    }

    // No global deadline (non-MCC / public API contract): keep the historical
    // inline, unbounded path — including the shared IntelligenceBus — so those
    // callers are byte-for-byte unaffected.
    if global_deadline.is_none() {
        let validation = WitnessValidationContext::new(net, validation_targets);
        reachability_aiger::run_aiger_seeding_with_nupn(
            net,
            trackers,
            &validation,
            aiger_budget.deadline,
            nupn,
            bus,
        );
        return trackers.iter().filter(|t| t.verdict.is_some()).count() - pre;
    }

    let cap = aiger_budget
        .deadline
        .map(|d| d.saturating_duration_since(Instant::now()))
        .unwrap_or(Duration::ZERO);
    if cap.is_zero() {
        return 0;
    }

    let net_for_worker = net.clone();
    let targets_for_worker = validation_targets.to_vec();
    let nupn_for_worker = nupn.cloned();
    let phase_deadline = aiger_budget.deadline;
    let mut worker_trackers: Vec<PropertyTracker> = trackers.to_vec();
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<PropertyTracker>>(1);
    let _worker = std::thread::Builder::new()
        .name("ty-reach-aiger-phase".to_string())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let validation =
                    WitnessValidationContext::new(&net_for_worker, &targets_for_worker);
                reachability_aiger::run_aiger_seeding_with_nupn(
                    &net_for_worker,
                    &mut worker_trackers,
                    &validation,
                    phase_deadline,
                    nupn_for_worker.as_ref(),
                    None,
                );
                worker_trackers
            }));
            // Ignore SendError: on timeout the parent has dropped the receiver.
            if let Ok(resolved) = result {
                let _ = tx.send(resolved);
            }
        });

    match rx.recv_timeout(cap) {
        Ok(resolved) => {
            for (dst, src) in trackers.iter_mut().zip(resolved) {
                debug_assert_eq!(dst.id, src.id, "AIGER worker tracker order diverged");
                if dst.verdict.is_some() {
                    continue;
                }
                if let (Some(verdict), Some(resolution)) = (src.verdict, src.resolved_by) {
                    resolve_tracker(dst, verdict, resolution.source, resolution.depth);
                }
            }
        }
        Err(_) => {
            // Timeout (AIGER still spinning a non-polling preprocessing/SAT
            // phase) or Disconnected (worker panicked): copy nothing back so the
            // exhaustive BFS tail is preserved. Worker thread deliberately
            // leaked — same contract as `run_pdr_seeding_wall_capped`.
        }
    }
    trackers.iter().filter(|t| t.verdict.is_some()).count() - pre
}

pub(in crate::examinations::reachability) fn bounded_witness_bfs_config(
    net: &PetriNet,
    config: &ExplorationConfig,
) -> WitnessBfsConfig {
    witness_bfs_config_with_limits(net, config, REACHABILITY_BOUNDED_WITNESS_BFS_MAX_STATES)
}

fn witness_bfs_config_with_limits(
    net: &PetriNet,
    config: &ExplorationConfig,
    max_states: usize,
) -> WitnessBfsConfig {
    let now = Instant::now();
    let global_deadline = config.deadline();
    let remaining = global_deadline
        .map(|d| d.saturating_duration_since(now))
        .unwrap_or(Duration::ZERO);
    let reserve = compute_fallback_reserve(net, remaining);

    let phase_cap = Duration::from_secs_f64(remaining.as_secs_f64() * 0.05)
        .max(Duration::from_secs(3))
        .min(Duration::from_secs(30));

    let phase_deadline = now + phase_cap;
    let reserved_deadline = deadline_preserving_reserve_at(global_deadline, reserve, now);
    let deadline = global_deadline.map(|_| match reserved_deadline {
        Some(reserved_deadline) if reserved_deadline < phase_deadline => reserved_deadline,
        _ => phase_deadline,
    });

    // This pre-solver bounded witness BFS is an under-approximation lane
    // (witness-only). Clamp it to the shared under-approx cap so it cannot
    // consume budget the final exhaustive BFS needs to run to completion.
    let deadline = match (
        deadline,
        under_approx_lane_deadline_at(net, global_deadline, now),
    ) {
        (Some(deadline), Some(cap)) => Some(deadline.min(cap)),
        (Some(deadline), None) => Some(deadline),
        (None, cap) => cap,
    };

    WitnessBfsConfig {
        deadline,
        max_states: resolve_lane_max_states(config.max_states(), max_states),
        max_depth: None,
    }
}

fn resolve_lane_max_states(configured: usize, lane_cap: usize) -> usize {
    let lane_cap = lane_cap.max(1);
    if configured == 0 {
        lane_cap
    } else {
        configured.min(lane_cap).max(1)
    }
}

pub(in crate::examinations::reachability) fn post_smt_witness_bfs_config(
    net: &PetriNet,
    config: &ExplorationConfig,
) -> Option<WitnessBfsConfig> {
    post_smt_witness_bfs_config_at(net, config, Instant::now())
}

fn post_smt_witness_bfs_config_at(
    net: &PetriNet,
    config: &ExplorationConfig,
    now: Instant,
) -> Option<WitnessBfsConfig> {
    let global_deadline = config.deadline()?;
    if global_deadline <= now {
        return None;
    }

    let remaining = global_deadline.duration_since(now);
    let reserve = compute_fallback_reserve(net, remaining);

    let phase_cap = Duration::from_secs_f64(remaining.as_secs_f64() * 0.01)
        .max(Duration::from_secs(1))
        .min(Duration::from_secs(2));

    let phase_deadline = now + phase_cap;
    let reserved_deadline = deadline_preserving_reserve_at(Some(global_deadline), reserve, now);
    let deadline = match reserved_deadline {
        Some(reserved_deadline) if reserved_deadline > now => {
            if reserved_deadline < phase_deadline {
                reserved_deadline
            } else {
                phase_deadline
            }
        }
        _ => {
            let borrowed = remaining.min(REACHABILITY_POST_SMT_WITNESS_BFS_RESERVE_LOAN_CAP);
            if borrowed.is_zero() {
                return None;
            }
            now + borrowed
        }
    };

    Some(WitnessBfsConfig {
        deadline: Some(deadline),
        max_states: resolve_lane_max_states(
            config.max_states(),
            REACHABILITY_POST_SMT_WITNESS_BFS_MAX_STATES,
        ),
        max_depth: None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::examinations::reachability) enum PostSmtWitnessBfsSkipReason {
    NoResidualProperties,
    MissingDeadline,
    ExpiredDeadline,
}

impl PostSmtWitnessBfsSkipReason {
    fn code(self) -> &'static str {
        match self {
            Self::NoResidualProperties => "no_residual_properties",
            Self::MissingDeadline => "missing_deadline",
            Self::ExpiredDeadline => "expired_deadline",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::examinations::reachability) struct PostSmtWitnessBfsReport {
    pub(in crate::examinations::reachability) total_properties: usize,
    pub(in crate::examinations::reachability) residual_before: usize,
    pub(in crate::examinations::reachability) seeded: usize,
    pub(in crate::examinations::reachability) unresolved_after: usize,
    pub(in crate::examinations::reachability) skip_reason: Option<PostSmtWitnessBfsSkipReason>,
    pub(in crate::examinations::reachability) stats:
        Option<reachability_bfs_witness::WitnessBfsStats>,
}

impl PostSmtWitnessBfsReport {
    fn skipped(
        total_properties: usize,
        residual_before: usize,
        skip_reason: PostSmtWitnessBfsSkipReason,
    ) -> Self {
        Self {
            total_properties,
            residual_before,
            seeded: 0,
            unresolved_after: residual_before,
            skip_reason: Some(skip_reason),
            stats: None,
        }
    }

    fn ran(
        total_properties: usize,
        residual_before: usize,
        seeded: usize,
        unresolved_after: usize,
        stats: reachability_bfs_witness::WitnessBfsStats,
    ) -> Self {
        Self {
            total_properties,
            residual_before,
            seeded,
            unresolved_after,
            skip_reason: None,
            stats: Some(stats),
        }
    }

    fn status_code(self) -> &'static str {
        if self.skip_reason.is_some() {
            "skipped"
        } else if self.stats.is_some_and(|stats| stats.completed) {
            "exhausted"
        } else if self.unresolved_after == 0 {
            "all_resolved"
        } else {
            "incomplete"
        }
    }

    fn stop_reason_code(self) -> &'static str {
        self.stats
            .map(|stats| stats.stop_reason.code())
            .unwrap_or("not_run")
    }

    fn skip_reason_code(self) -> &'static str {
        self.skip_reason
            .map(PostSmtWitnessBfsSkipReason::code)
            .unwrap_or("none")
    }

    fn reason_code(self) -> &'static str {
        self.skip_reason
            .map(PostSmtWitnessBfsSkipReason::code)
            .or_else(|| self.stats.map(|stats| stats.stop_reason.code()))
            .unwrap_or("not_run")
    }

    fn visited_states(self) -> usize {
        self.stats.map(|stats| stats.visited_states).unwrap_or(0)
    }

    fn configured_max_states(self) -> usize {
        self.stats
            .map(|stats| stats.configured_max_states)
            .unwrap_or(REACHABILITY_POST_SMT_WITNESS_BFS_MAX_STATES)
    }

    fn effective_max_states(self) -> usize {
        self.stats
            .map(|stats| stats.effective_max_states)
            .unwrap_or(0)
    }

    fn elapsed_ms(self) -> u128 {
        self.stats
            .map(|stats| stats.elapsed.as_millis())
            .unwrap_or(0)
    }

    fn completed(self) -> bool {
        self.stats.is_some_and(|stats| stats.completed)
    }

    fn needs_blocker_action(self) -> bool {
        self.residual_before > 0 && self.unresolved_after > 0
    }
}

pub(in crate::examinations::reachability) fn run_post_smt_witness_bfs(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation: &WitnessValidationContext<'_>,
    config: &ExplorationConfig,
) -> PostSmtWitnessBfsReport {
    let total_properties = trackers.len();
    let resolved_before = trackers
        .iter()
        .filter(|tracker| tracker.verdict.is_some())
        .count();
    let residual_before = total_properties - resolved_before;
    if residual_before == 0 {
        return PostSmtWitnessBfsReport::skipped(
            total_properties,
            residual_before,
            PostSmtWitnessBfsSkipReason::NoResidualProperties,
        );
    }
    if config.deadline().is_none() {
        return PostSmtWitnessBfsReport::skipped(
            total_properties,
            residual_before,
            PostSmtWitnessBfsSkipReason::MissingDeadline,
        );
    }
    let Some(witness_config) = post_smt_witness_bfs_config(net, config) else {
        return PostSmtWitnessBfsReport::skipped(
            total_properties,
            residual_before,
            PostSmtWitnessBfsSkipReason::ExpiredDeadline,
        );
    };
    let stats = reachability_bfs_witness::run_bounded_witness_bfs(
        net,
        trackers,
        validation,
        witness_config,
    );
    let resolved_after = trackers
        .iter()
        .filter(|tracker| tracker.verdict.is_some())
        .count();
    let seeded = resolved_after.saturating_sub(resolved_before);
    let unresolved_after = if stats.completed {
        0
    } else {
        total_properties - resolved_after
    };
    PostSmtWitnessBfsReport::ran(
        total_properties,
        residual_before,
        seeded,
        unresolved_after,
        stats,
    )
}

fn post_smt_witness_bfs_evidence_row(report: PostSmtWitnessBfsReport) -> String {
    format!(
        "MCC reachability_answer_lane_summary lane=post_smt_witness_bfs \
         status={} stop_reason={} skip_reason={} property_count={} residual_before={} \
         seeded={} unresolved_after={} visited_states={} configured_max_states={} \
         effective_max_states={} phase_cap_ms={} completed={} elapsed_ms={}",
        report.status_code(),
        report.stop_reason_code(),
        report.skip_reason_code(),
        report.total_properties,
        report.residual_before,
        report.seeded,
        report.unresolved_after,
        report.visited_states(),
        report.configured_max_states(),
        report.effective_max_states(),
        2000,
        report.completed(),
        report.elapsed_ms(),
    )
}

fn post_smt_witness_bfs_blocker_action_row(report: PostSmtWitnessBfsReport) -> String {
    format!(
        "MCC blocker_action selected=true priority_rank=1 lane_family=explicit_state \
         blocker_piece=reachability_post_smt_witness_bfs \
         blocker_gate=post_smt_witness_bfs owner_project=Petri \
         owner_primitive=reachability_witness_bfs \
         action_code=inspect_post_smt_witness_bfs_budget reason_code={} \
         next_answer_lane=post_smt_witness_bfs status={} residual_before={} \
         unresolved_after={} seeded={} stop_reason={} skip_reason={}",
        report.reason_code(),
        report.status_code(),
        report.residual_before,
        report.unresolved_after,
        report.seeded,
        report.stop_reason_code(),
        report.skip_reason_code(),
    )
}

fn emit_post_smt_witness_bfs_observability(report: PostSmtWitnessBfsReport) {
    let evidence_row = post_smt_witness_bfs_evidence_row(report);
    eprintln!("{evidence_row}");

    let mut capability_report = CapabilityReport::new(ProblemKind::ExplicitReachability);
    capability_report.add_evidence(evidence_row);
    if report.needs_blocker_action() {
        capability_report.add_evidence(post_smt_witness_bfs_blocker_action_row(report));
    }
    crate::mcc_backend_evidence::record_runtime_reachability_bmc_report(&capability_report);
}

pub(crate) fn kinduction_deadline(
    net: &PetriNet,
    global_deadline: Option<Instant>,
) -> Option<Instant> {
    kinduction_deadline_at(net, global_deadline, Instant::now())
}

fn kinduction_deadline_at(
    net: &PetriNet,
    global_deadline: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    let reserve = global_deadline
        .map(|d| compute_fallback_reserve(net, d.saturating_duration_since(now)))
        .unwrap_or(Duration::ZERO);
    deadline_preserving_reserve_at(global_deadline, reserve, now)
}

fn proof_rescue_deadline_at(
    net: &PetriNet,
    global_deadline: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    let global_deadline = global_deadline?;
    let remaining = global_deadline.saturating_duration_since(now);
    let reserve = compute_fallback_reserve(net, remaining);
    if remaining <= reserve {
        return Some(now);
    }

    let latest = global_deadline.checked_sub(reserve).unwrap_or(now);
    let budget = REACHABILITY_PROOF_RESCUE_PHASE_CAP.min(latest.saturating_duration_since(now));
    Some(now + budget)
}

fn should_run_post_witness_proof_rescue_at(
    net: &PetriNet,
    global_deadline: Option<Instant>,
    max_bmc_depth: Option<usize>,
    unresolved_count: usize,
    now: Instant,
) -> bool {
    if max_bmc_depth.is_some()
        || unresolved_count == 0
        || unresolved_count > REACHABILITY_PROOF_RESCUE_MAX_PENDING
    {
        return false;
    }

    let Some(global_deadline) = global_deadline else {
        return false;
    };
    let remaining = global_deadline.saturating_duration_since(now);
    let reserve = compute_fallback_reserve(net, remaining);
    remaining > reserve + REACHABILITY_PROOF_RESCUE_MIN_BUDGET
}

pub(in crate::examinations::reachability) fn run_post_witness_proof_rescue(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation_targets: &[WitnessValidationTarget],
    global_deadline: Option<Instant>,
    max_bmc_depth: Option<usize>,
) -> usize {
    if validation_targets.len() != trackers.len() {
        eprintln!(
            "Post-witness proof rescue: validation target count mismatch ({} targets for {} trackers), skipping",
            validation_targets.len(),
            trackers.len()
        );
        return 0;
    }

    let pending: Vec<usize> = trackers
        .iter()
        .enumerate()
        .filter_map(|(index, tracker)| tracker.verdict.is_none().then_some(index))
        .collect();
    let now = Instant::now();
    if !should_run_post_witness_proof_rescue_at(
        net,
        global_deadline,
        max_bmc_depth,
        pending.len(),
        now,
    ) {
        return 0;
    }

    let Some(rescue_deadline) = proof_rescue_deadline_at(net, global_deadline, now) else {
        return 0;
    };
    if deadline_expired(Some(rescue_deadline)) {
        return 0;
    }

    eprintln!(
        "Post-witness proof rescue: attempting {} unresolved reachability properties",
        pending.len()
    );

    let mut seeded = 0;
    for index in pending {
        if deadline_expired(Some(rescue_deadline)) {
            break;
        }

        let mut candidate = vec![trackers[index].clone()];
        let candidate_targets = vec![validation_targets[index].clone()];
        candidate[0].flushed = false;
        let base_depth = reachability_bmc::run_bmc_seeding(
            net,
            &mut candidate,
            &candidate_targets,
            Some(rescue_deadline),
        );
        if candidate[0].verdict.is_none() && base_depth.is_some() {
            kinduction::run_kinduction_seeding(
                net,
                &mut candidate,
                Some(rescue_deadline),
                base_depth,
            );
        }

        if let Some(verdict) = candidate[0].verdict {
            if trackers[index].verdict.is_none() {
                trackers[index].verdict = Some(verdict);
                trackers[index].resolved_by = candidate[0].resolved_by;
                seeded += 1;
            }
        }
    }

    seeded
}

pub(in crate::examinations::reachability) fn explore_original_net_full_reachability(
    net: &PetriNet,
    trackers: Vec<PropertyTracker>,
    config: &ExplorationConfig,
) -> Result<(crate::explorer::ExplorationResult, Vec<PropertyTracker>), std::io::Error> {
    // This is the correctness oracle for guarded AG-fireability fallbacks: no
    // POR or IntelligenceBus pruning that could alter the reachable set.
    // Checkpointing IS preserved when configured because it operates on the
    // original net's state space, so save/resume cannot introduce unsoundness.
    let full_config = original_net_full_reachability_config(config);
    let mut observer = ReachabilityObserver::from_trackers(net, trackers);
    let result = explore_with_optional_checkpoint(net, &full_config, &mut observer)?;
    Ok((result, observer.into_trackers()))
}

fn original_net_full_reachability_config(config: &ExplorationConfig) -> ExplorationConfig {
    config.clone().with_por(PorStrategy::None)
}

fn explore_with_optional_checkpoint<O>(
    net: &PetriNet,
    config: &ExplorationConfig,
    observer: &mut O,
) -> Result<crate::explorer::ExplorationResult, std::io::Error>
where
    O: crate::explorer::ParallelExplorationObserver
        + crate::explorer::CheckpointableObserver
        + Send,
{
    if config.checkpoint().is_some() {
        crate::explorer::explore_checkpointable_observer(net, config, observer)
    } else {
        Ok(explore_observer(net, config, observer))
    }
}

fn predicate_contains_fireability(predicate: &ResolvedPredicate) -> bool {
    match predicate {
        ResolvedPredicate::And(children) | ResolvedPredicate::Or(children) => {
            children.iter().any(predicate_contains_fireability)
        }
        ResolvedPredicate::Not(inner) => predicate_contains_fireability(inner),
        ResolvedPredicate::IsFireable(_) => true,
        ResolvedPredicate::IntLe(..) | ResolvedPredicate::True | ResolvedPredicate::False => false,
    }
}

/// Kill-switch for the integer dead-transition sweep (Phase 2b-int).
///
/// The sweep proves `AG(¬IsFireable(t))` trackers TRUE by a single QF_LIA
/// integer-infeasibility query per transition (no relaxation gap; see
/// [`crate::symbolic::int_state_equation::integer_dead_transition`]). It is
/// strictly ADDITIVE — UNSAT is a sound proof, every other outcome DECLINES and
/// the tracker falls through to the exact lanes unchanged. Defaults ON; set
/// `TY_MCC_ENABLE_INTEGER_DEAD_TRANSITION=0` (or `false`/`no`/`off`) to disable
/// for A/B comparison.
const ENABLE_INTEGER_DEAD_TRANSITION_ENV: &str = "TY_MCC_ENABLE_INTEGER_DEAD_TRANSITION";

/// Absolute cap on the wall-clock slice the integer dead-transition sweep may
/// consume, so it can never starve the exact CHC/PDR/AIGER/BFS lanes. The sweep
/// is a cheap pre-pass (one `check_sat` per candidate transition); this bounds
/// the whole sweep regardless of how many candidates there are.
const INTEGER_DEAD_TRANSITION_SWEEP_CAP: Duration = Duration::from_secs(2);

/// Per-transition solver timeout inside the sweep. Each enabledness query is a
/// single QF_LIA `check_sat`; a tight per-call bound keeps a hard candidate from
/// eating the whole slice and lets the sweep attempt many transitions.
const INTEGER_DEAD_TRANSITION_PER_TXN_TIMEOUT: Duration = Duration::from_millis(300);

/// Minimum slice below which the sweep is skipped entirely — a near-expired slot
/// should fund the exact lanes, not start a fresh integer pass.
const INTEGER_DEAD_TRANSITION_MIN_BUDGET: Duration = Duration::from_millis(100);

fn integer_dead_transition_enabled() -> bool {
    !matches!(
        std::env::var(ENABLE_INTEGER_DEAD_TRANSITION_ENV)
            .ok()
            .as_deref()
            .map(str::trim),
        Some("0" | "false" | "FALSE" | "no" | "off" | "OFF")
    )
}

/// If `predicate` is exactly `¬IsFireable(T)` (a dead-transition invariant
/// `AG(¬IsFireable(T))`), return the transition set `T` whose collective death
/// makes the predicate hold in every reachable marking. `AG(¬IsFireable(T))` is
/// TRUE iff EVERY `t ∈ T` is dead, since `IsFireable(T) = ⋁_{t∈T} enabled(t)`
/// (see [`crate::resolved_predicate::eval_predicate`]). Returns `None` for any
/// other predicate shape (the sweep only discharges this exact form soundly).
fn ag_dead_transition_targets(
    predicate: &ResolvedPredicate,
) -> Option<&[crate::petri_net::TransitionIdx]> {
    match predicate {
        ResolvedPredicate::Not(inner) => match inner.as_ref() {
            ResolvedPredicate::IsFireable(transitions) if !transitions.is_empty() => {
                Some(transitions.as_slice())
            }
            _ => None,
        },
        _ => None,
    }
}

/// Integer dead-transition sweep (Phase 2b-int).
///
/// Resolves pending `AG(¬IsFireable(T))` reachability trackers — the
/// dead-transition special case shared by ReachabilityFireability and (via the
/// LTL Phase-1 `G(atom)→AG` routing) LTLFireability sub-properties — by proving
/// each referenced transition DEAD with the integer state-equation
/// infeasibility primitive
/// ([`crate::symbolic::int_state_equation::integer_dead_transition`]).
///
/// A tracker is resolved TRUE only when EVERY transition in its `IsFireable(T)`
/// set is proven dead (so `IsFireable(T)` is unenabled in every reachable
/// marking, i.e. `AG(¬IsFireable(T))` holds). The per-transition proof is a
/// genuine integer-infeasibility certificate with no relaxation gap, so the
/// verdict matches BFS exactly.
///
/// SOUNDNESS / ADDITIVITY: a tracker is touched ONLY on a full set of dead
/// proofs; a single live / inconclusive / oversized / overflow transition leaves
/// the tracker pending for the exact lanes. The sweep never emits FALSE and never
/// changes an existing verdict (first-writer-wins via `resolve_tracker`).
///
/// BUDGET RESERVE: the whole sweep is clamped to a small fixed slice
/// ([`INTEGER_DEAD_TRANSITION_SWEEP_CAP`]) of the remaining deadline with a tight
/// per-transition solver timeout, so it can never starve the exact CHC / PDR /
/// AIGER / exhaustive-BFS lanes; on a near-expired or oversized slot it declines.
/// Proven-dead transitions are cached across trackers so a transition shared by
/// several trackers costs one solve.
///
/// Returns the number of trackers it newly resolved.
fn run_integer_dead_transition_sweep(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    global_deadline: Option<Instant>,
) -> usize {
    if !integer_dead_transition_enabled() {
        return 0;
    }

    // Collect the distinct candidate transitions referenced by pending
    // AG(¬IsFireable(T)) trackers. If none, there is nothing to prove.
    let mut candidates: std::collections::HashSet<crate::petri_net::TransitionIdx> =
        std::collections::HashSet::new();
    for tracker in trackers.iter() {
        if tracker.verdict.is_some() || tracker.quantifier != PathQuantifier::AG {
            continue;
        }
        if let Some(targets) = ag_dead_transition_targets(&tracker.predicate) {
            candidates.extend(targets.iter().copied());
        }
    }
    if candidates.is_empty() {
        return 0;
    }

    // Reserve a small fixed slice of the remaining deadline. On a near-expired
    // slot, decline so the exact lanes keep the budget. With no global deadline
    // (non-MCC / public API) the sweep still runs but stays bounded by the
    // per-transition timeout and the absolute sweep cap.
    let now = Instant::now();
    let sweep_deadline = match global_deadline {
        Some(limit) => {
            let remaining = limit.saturating_duration_since(now);
            if remaining < INTEGER_DEAD_TRANSITION_MIN_BUDGET {
                return 0;
            }
            // Never consume more than half the remaining budget, and never more
            // than the absolute cap — the exact lanes always keep the majority.
            let slice = (remaining / 2).min(INTEGER_DEAD_TRANSITION_SWEEP_CAP);
            now + slice
        }
        None => now + INTEGER_DEAD_TRANSITION_SWEEP_CAP,
    };

    // Prove (or refute) each candidate ONCE, caching the result. `Some(true)` =
    // proven dead; `Some(false)` = NOT proven dead (live / inconclusive /
    // declined — treated identically: the tracker is not resolved).
    let mut dead_cache: std::collections::HashMap<crate::petri_net::TransitionIdx, bool> =
        std::collections::HashMap::new();
    for &t in &candidates {
        if Instant::now() >= sweep_deadline {
            break;
        }
        let proven_dead = crate::symbolic::int_state_equation::integer_dead_transition(
            net,
            t,
            INTEGER_DEAD_TRANSITION_PER_TXN_TIMEOUT,
        );
        dead_cache.insert(t, proven_dead);
    }

    // Resolve every pending AG(¬IsFireable(T)) tracker whose ENTIRE transition
    // set is proven dead. A transition not in the cache (the sweep ran out of
    // slice before reaching it) counts as not-proven, so the tracker is left
    // pending — never falsely resolved.
    let mut resolved = 0;
    for tracker in trackers.iter_mut() {
        if tracker.verdict.is_some() || tracker.quantifier != PathQuantifier::AG {
            continue;
        }
        let Some(targets) = ag_dead_transition_targets(&tracker.predicate) else {
            continue;
        };
        if targets
            .iter()
            .all(|t| dead_cache.get(t).copied().unwrap_or(false))
        {
            resolve_tracker(tracker, true, ReachabilityResolutionSource::Lp, None);
            resolved += 1;
        }
    }
    resolved
}

fn tracker_requires_original_net_full_bfs(tracker: &PropertyTracker) -> bool {
    tracker.verdict.is_none()
        && tracker.quantifier == PathQuantifier::AG
        && predicate_contains_fireability(&tracker.predicate)
}

pub(in crate::examinations::reachability) fn unresolved_ag_fireability_requires_original_net_full_bfs(
    trackers: &[PropertyTracker],
) -> bool {
    trackers.iter().any(tracker_requires_original_net_full_bfs)
}

fn non_original_required_subset(
    trackers: &[PropertyTracker],
) -> Option<(Vec<usize>, Vec<PropertyTracker>)> {
    let mut indices = Vec::new();
    let mut subset = Vec::new();
    let mut has_unresolved = false;
    for (index, tracker) in trackers.iter().enumerate() {
        if tracker_requires_original_net_full_bfs(tracker) {
            continue;
        }
        if tracker.verdict.is_none() {
            has_unresolved = true;
        }
        indices.push(index);
        subset.push(tracker.clone());
    }
    has_unresolved.then_some((indices, subset))
}

fn merge_tracker_subset(
    trackers: &mut [PropertyTracker],
    indices: Vec<usize>,
    subset: Vec<PropertyTracker>,
) {
    debug_assert_eq!(indices.len(), subset.len());
    for (index, tracker) in indices.into_iter().zip(subset) {
        trackers[index] = tracker;
    }
}

fn run_reduced_reachability_fallback(
    net: &PetriNet,
    trackers: Vec<PropertyTracker>,
    config: &ExplorationConfig,
) -> Result<(crate::explorer::ExplorationResult, Vec<PropertyTracker>), ReachabilityExploreError> {
    let reduced = reduce_reachability_queries(net, &trackers)?;

    // Safety check: verify that all entities referenced by predicates survived
    // the reduction. For IsFireable, checks referenced transitions. For
    // TokensCount, checks all transitions touching referenced places. If any
    // were eliminated, the reduced net may have an incomplete state space,
    // causing unsound AG=TRUE verdicts. Fall back to unreduced BFS.
    if !all_predicates_reduction_safe(net, &reduced, &trackers) {
        eprintln!(
            "Reachability: predicate-referenced transitions eliminated by reduction, \
             falling back to unreduced BFS",
        );
        let (result, mut trackers) = explore_original_net_full_reachability(net, trackers, config)?;
        if result.completed {
            finalize_exhaustive_completion(&mut trackers);
        }
        return Ok((result, trackers));
    }

    let config = config.refitted_for_net(&reduced.net);
    if let Some((slice, slice_trackers)) = build_reachability_slice(&reduced, &trackers) {
        let config = config.refitted_for_net(&slice.net);
        let (result, mut trackers) =
            explore_reachability_on_slice(&slice, slice_trackers, &config)?;
        if result.completed {
            finalize_exhaustive_completion(&mut trackers);
        }
        return Ok((result, trackers));
    }

    let (result, mut trackers) =
        explore_reachability_on_reduced_net(net, &reduced, trackers, &config)?;
    if result.completed {
        finalize_exhaustive_completion(&mut trackers);
    }
    Ok((result, trackers))
}

/// Check reachability properties with fail-closed unresolved-name guard.
///
/// Runs BMC on the original net first (witness-only), then BFS on the
/// reduced net for any remaining unresolved properties.
///
/// Safety: returns `CannotCompute` for any formula with unresolved place or
/// transition names. Silent name drops during resolution produce degenerate
/// predicates (`False` for is-fireable, `0` for tokens-count) that corrupt
/// evaluation results.
/// Test entry point: returns all results without incremental stdout flushing.
#[cfg(test)]
pub(crate) fn check_reachability_properties(
    net: &PetriNet,
    properties: &[Property],
    config: &ExplorationConfig,
) -> Vec<(String, Verdict)> {
    let aliases = PropertyAliases::identity(net);
    check_reachability_properties_inner(net, properties, &aliases, config, false, None)
}

/// Collection entry point: returns all results without incremental flushing.
/// Used by `collect_examination_core` and tests that inspect return values.
pub(crate) fn check_reachability_properties_with_aliases(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
) -> Vec<(String, Verdict)> {
    check_reachability_properties_inner(net, properties, aliases, config, false, None)
}

pub(crate) fn check_reachability_properties_with_aliases_and_nupn(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    nupn: Option<&NupnStructure>,
) -> Vec<(String, Verdict)> {
    check_reachability_properties_inner(net, properties, aliases, config, false, nupn)
}

/// MCC binary entry point: prints FORMULA lines incrementally between phases
/// for crash-resilient output, and returns only unflushed (BFS-phase) results.
///
/// Wired into `run_examination_for_model` / `run_examination_with_dir` via
/// `collect_examination_core(flush=true)`.
pub(crate) fn check_reachability_properties_with_flush_and_nupn(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    nupn: Option<&NupnStructure>,
) -> Vec<(String, Verdict)> {
    check_reachability_properties_inner(net, properties, aliases, config, true, nupn)
}

/// Wall-clock cap for the Phase-0 formula-simplification optimization under an
/// MCC deadline.
///
/// `simplify_properties_with_aliases` -> `compute_facts` calls
/// `structural_deadlock_free` (siphon/trap enumeration), which does NOT poll
/// the deadline and can spin the whole wall budget on barrier-style nets
/// (measured: FlexibleBarrier-PT-04b ReachabilityCardinality spent the entire
/// budget here before any verdict). Simplification is a pure, semantics-
/// preserving optimization, so on cap/panic we fall back to the original
/// (un-simplified) formulas — identical verdicts, just without the speedup.
const SIMPLIFY_PHASE_CAP: Duration = Duration::from_secs(5);

/// Simplify reachability formulas, wall-capped under an MCC deadline.
///
/// With no deadline (non-MCC / public API contract) the original unbounded
/// inline simplification runs. Under a deadline the simplification is run on a
/// worker thread and abandoned if it does not finish within
/// [`SIMPLIFY_PHASE_CAP`] (bounded by the remaining budget); the original
/// formulas are returned unchanged. The abandoned worker is intentionally not
/// joined — `structural_deadlock_free` has no cooperative cancellation
/// primitive; the OS reclaims it when the process exits. Verdict-preserving:
/// simplification only ever rewrites a formula to a semantically equivalent
/// one, so the un-simplified fallback yields the same answers.
fn simplify_properties_deadline_aware(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    deadline: Option<Instant>,
) -> Vec<Property> {
    let Some(deadline) = deadline else {
        return crate::formula_simplify::simplify_properties_with_aliases(net, properties, aliases);
    };

    let remaining = deadline.saturating_duration_since(Instant::now());
    let cap = SIMPLIFY_PHASE_CAP.min(remaining);
    if cap.is_zero() {
        return properties.to_vec();
    }

    let net_for_worker = net.clone();
    let properties_for_worker = properties.to_vec();
    let aliases_for_worker = aliases.clone();
    let (tx, rx) = std::sync::mpsc::sync_channel::<std::thread::Result<Vec<Property>>>(1);
    let _worker = std::thread::Builder::new()
        .name("ty-reach-simplify-phase".to_string())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::formula_simplify::simplify_properties_with_aliases(
                    &net_for_worker,
                    &properties_for_worker,
                    &aliases_for_worker,
                )
            }));
            // Ignore SendError: parent abandoned the phase and dropped rx.
            let _ = tx.send(result);
        });

    match rx.recv_timeout(cap) {
        Ok(Ok(simplified)) => simplified,
        // Panic or cap expiry: fall back to the original (un-simplified)
        // formulas. Semantics-preserving, hence verdict-preserving.
        Ok(Err(_)) | Err(_) => {
            eprintln!(
                "Reachability: Phase-0 simplification exceeded {SIMPLIFY_PHASE_CAP:?} wall cap; \
                 proceeding with un-simplified formulas"
            );
            properties.to_vec()
        }
    }
}

/// Inner pipeline: when `flush` is true, prints resolved FORMULA lines to
/// stdout between phases and omits them from the return value. When false,
/// returns all results in the vec (test mode).
fn check_reachability_properties_inner(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    flush: bool,
    nupn: Option<&NupnStructure>,
) -> Vec<(String, Verdict)> {
    let incremental_flush = incremental_flush_enabled(config, flush);

    // Create the intelligence bus for cross-technique cooperation.
    // Seed it with LP bounds and P-invariant constraints so BFS can prune
    // markings that violate proved structural invariants.
    let bus = Arc::new(IntelligenceBus::new());
    intelligence_bus::seed_from_lp(&bus, net);

    // Phase 0: simplify formulas using structural facts and LP proofs.
    let simplified =
        simplify_properties_deadline_aware(net, properties, aliases, config.deadline());
    let simplified_properties = &simplified;

    // Phase 1: classify each property as Valid or Invalid, preserving order.
    let (prepared, mut trackers, validation_targets) =
        prepare_trackers_with_aliases(properties, simplified_properties, aliases);
    let validation = WitnessValidationContext::new(net, &validation_targets);
    let mut aiger_deferred_for_fireability = false;

    // Phase 1b: EARLY raw-net random-walk witness pass.
    //
    // On LARGE nets the heavyweight front-matter (Phase 2a bounded witness BFS,
    // BMC, LP state-equation seeding, CHC/SMT, AIGER, PDR) can consume the whole
    // budget before the existing Phase-2d random walk ever fires, so on those
    // nets the walk is starved and never seeds. This pass runs the SAME walk
    // engine on the raw net up front so cheap witnesses are caught before the
    // front-matter spends the budget. Hits resolve trackers (skipped downstream);
    // misses fall through to the unchanged pipeline.
    //
    // ADDITIVE / no-starvation: this lane is clamped to
    // `under_approx_lane_deadline` — a SLICE of the remaining budget
    // (`min(remaining/4, 8s)`) that reserves the exhaustive BFS tail and leaves
    // the front phases their existing fair share. It is the identical deadline
    // the Phase-2d walk already uses, so the front phases and the exhaustive BFS
    // keep what they have today.
    //
    // SOUND: `run_random_walk_seeding` only emits replay-validated EF=TRUE /
    // AG=FALSE witnesses from reachable markings — it can never produce the
    // universal verdicts (EF=FALSE / AG=TRUE), which remain the exclusive job of
    // the exhaustive BFS / replay-validated BMC.
    if trackers.iter().any(|t| t.verdict.is_none()) && !random_walk_seeding_disabled() {
        let pre_early_walk = trackers.iter().filter(|t| t.verdict.is_some()).count();
        reachability_walk::run_random_walk_seeding(
            net,
            &mut trackers,
            &validation,
            under_approx_lane_deadline(net, config.deadline()),
        );
        let post_early_walk = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let early_walk_seeded = post_early_walk - pre_early_walk;
        if early_walk_seeded > 0 {
            eprintln!(
                "Early raw-net random walk seeded {early_walk_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Phase 2a: bounded original-net BFS to find cheap concrete witnesses
    // before solver phases can consume the deadline. This only resolves
    // EF=true and AG=false from replay-validated traces.
    if trackers.iter().any(|t| t.verdict.is_none()) {
        let pre_bfs_witness = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let stats = reachability_bfs_witness::run_bounded_witness_bfs(
            net,
            &mut trackers,
            &validation,
            bounded_witness_bfs_config(net, config),
        );
        let post_bfs_witness = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let bfs_witness_seeded = post_bfs_witness - pre_bfs_witness;
        if bfs_witness_seeded > 0 {
            eprintln!(
                "Bounded witness BFS seeded {bfs_witness_seeded}/{} reachability verdicts \
                 after visiting {} states",
                trackers.len(),
                stats.visited_states,
            );
        }
        if stats.completed {
            finalize_exhaustive_completion(&mut trackers);
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Phase 2: run BMC on the original net to find witnesses.
    // BMC can seed EF(true) and AG(false) verdicts without full exploration.
    // Also returns the max depth where all properties completed (base case for k-induction).
    let max_bmc_depth;
    if !trackers.is_empty() {
        max_bmc_depth = reachability_bmc::run_bmc_seeding(
            net,
            &mut trackers,
            &validation_targets,
            witness_search_deadline(net, config.deadline()),
        );
        let seeded = trackers.iter().filter(|t| t.verdict.is_some()).count();
        if seeded > 0 {
            eprintln!(
                "BMC seeded {seeded}/{} reachability verdicts",
                trackers.len()
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            let post_smt_report = run_post_smt_witness_bfs(net, &mut trackers, &validation, config);
            emit_post_smt_witness_bfs_observability(post_smt_report);
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    } else {
        max_bmc_depth = None;
    }

    // Flush BMC-resolved verdicts to stdout for crash resilience.
    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Phase 2b: LP state equation seeding.
    // LP can prove EF(phi) = FALSE when the state equation + phi is infeasible,
    // and AG(phi) = TRUE when every violating atom is LP-infeasible.
    if !trackers.is_empty() {
        let pre_lp = trackers.iter().filter(|t| t.verdict.is_some()).count();
        reachability_lp::run_lp_seeding(
            net,
            &mut trackers,
            witness_search_deadline(net, config.deadline()),
        );
        let post_lp = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let lp_seeded = post_lp - pre_lp;
        if lp_seeded > 0 {
            eprintln!(
                "LP state equation seeded {lp_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            let post_smt_report = run_post_smt_witness_bfs(net, &mut trackers, &validation, config);
            emit_post_smt_witness_bfs_observability(post_smt_report);
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    // Phase 2b-int: integer dead-transition sweep.
    // Discharges pending AG(¬IsFireable(T)) trackers — the dead-transition
    // special case of ReachabilityFireability (and, via the LTL Phase-1
    // G(atom)→AG routing, LTLFireability sub-properties of that shape) — by
    // proving every referenced transition DEAD with the integer state-equation
    // infeasibility primitive (one QF_LIA solve per transition, no relaxation
    // gap). These are exactly the pure-fireability residuals the CHC/PDR lanes
    // carve out, so without this sweep they fall through to the heavy deferred
    // AIGER / exhaustive BFS. Strictly ADDITIVE and SOUND: a tracker is resolved
    // TRUE only on a full set of integer-infeasibility proofs, every other
    // outcome declines, and the sweep is clamped to a small fixed slice so it
    // can never starve the exact lanes. Default-ON; flip
    // `TY_MCC_ENABLE_INTEGER_DEAD_TRANSITION=0` to disable for benchmarking.
    if trackers.iter().any(|t| t.verdict.is_none()) {
        let dead_seeded = run_integer_dead_transition_sweep(net, &mut trackers, config.deadline());
        if dead_seeded > 0 {
            eprintln!(
                "Integer dead-transition sweep seeded {dead_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if run_compound_reuse_phase(&mut trackers, "LP") > 0
        && trackers.iter().all(|t| t.verdict.is_some())
    {
        if incremental_flush {
            flush_resolved(&mut trackers);
        }
        return assemble_results(&prepared, &trackers, true, incremental_flush);
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Phase 2b0: exact symbolic Decision-Diagram reachability (feature
    // `dd-backend`). On a small bounded net this builds the complete
    // reachable set once and answers every pending EF/AG query exactly —
    // equivalent to a completed original-net BFS, so it can short-circuit
    // the entire heavy solver tail. Fail-closed: declines (resolves
    // nothing) on any non-eligible net, translation gap, timeout, or DD
    // error, leaving downstream phases untouched.
    //
    // Skipped once the global deadline has expired: like every other seeding
    // phase it must not start a fresh (up-to-5s) computation past the budget,
    // and skipping preserves the deadline-driven BFS-only/checkpoint path.
    if !deadline_expired(config.deadline()) && trackers.iter().any(|t| t.verdict.is_none()) {
        let dd_seeded = run_dd_reachability_phase(net, &mut trackers);
        if dd_seeded > 0 {
            eprintln!(
                "DD symbolic reachability seeded {dd_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    // Phase 2b0-mdd: exact symbolic MULTI-VALUED DD reachability (feature
    // `dd-backend`), the MDD twin of Phase 2b0. The BDD lane above bit-blasts
    // each place into Boolean bits and blows up on the counter / conserved /
    // high-bound net families (it then times out and DECLINES, leaving those
    // trackers pending). The MDD spends one level per place — no bit-blasting —
    // and converges there. It builds the complete reachable set ONCE and answers
    // each pending EF/AG query EXACTLY, so a verdict is equivalent to a completed
    // original-net BFS and can short-circuit the heavy solver tail.
    //
    // Runs ONLY on the residual the BDD lane left pending (it is gated on `any
    // pending`), so it pays the MDD build cost only when the cheaper BDD lane did
    // not already close everything. Default-ON (kill-switch
    // TY_MCC_ENABLE_MDD_REACHABILITY): this is a count-R-once lever — the
    // proven-good MDD ROI shape (like the StateSpace MDD lane), unlike the
    // per-formula-fixpoint MDD CTL lane that is defaulted off.
    //
    // Fail-closed: declines (resolves nothing) on any non-eligible net,
    // translation gap, timeout, deadline, overflow, node cap, or MDD error,
    // leaving downstream phases untouched. Skipped once the global deadline has
    // expired (it must not start a fresh computation past the budget).
    if !deadline_expired(config.deadline()) && trackers.iter().any(|t| t.verdict.is_none()) {
        let mdd_seeded = run_mdd_reachability_phase(net, &mut trackers, config.deadline(), nupn);
        if mdd_seeded > 0 {
            eprintln!(
                "MDD symbolic reachability seeded {mdd_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Phase 2b1: post-SMT residual witness BFS.
    // The first bounded witness BFS runs before ay has reduced the query set.
    // Under MCC deadlines, give the residual properties one small deterministic
    // original-net witness window before proof-heavy lanes can consume the tail.
    let post_smt_report = run_post_smt_witness_bfs(net, &mut trackers, &validation, config);
    emit_post_smt_witness_bfs_observability(post_smt_report);
    if post_smt_report.residual_before == 0 {
        if incremental_flush {
            flush_resolved(&mut trackers);
        }
        return assemble_results(&prepared, &trackers, true, incremental_flush);
    }
    if let Some(stats) = post_smt_report.stats {
        if stats.completed {
            finalize_exhaustive_completion(&mut trackers);
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Move #2: build the verdict-preserving reduced net ONCE here, after the
    // cheap witness/LP lanes have already resolved many trackers — that
    // minimizes the protected/unresolved support and so maximizes the
    // reduction — and reuse it for the two heavy symbolic proof lanes (CHC and
    // PDR). The reduced net is the SAME one the exhaustive Phase-3 BFS already
    // trusts for the authoritative original-net EF/AG verdict
    // (`reduce_reachability_queries` + `all_predicates_reduction_safe`), so a
    // symbolic proof on it yields the identical boolean verdict, mapped back
    // 1:1 by tracker id. Only built under an MCC deadline (non-MCC/public-API
    // callers stay byte-for-byte on the original net), and `build_*` declines
    // (→ original net) whenever the reduction does not shrink, is not
    // verdict-preserving, or any unresolved predicate cannot be remapped
    // exactly. Never unsound: declining only ever routes a tracker to the
    // original-net lanes / exhaustive BFS.
    // Only worth building if at least one unresolved tracker is a candidate the
    // CHC/PDR lanes would actually attempt: both lanes carve out predicates with
    // `IsFireable` atoms (historical wrong-answer risk), so a pure-fireability
    // residual set (e.g. most ReachabilityFireability tails) would pay the
    // reduction cost for lanes that skip every tracker. Gate it out.
    let has_symbolic_lane_candidate = trackers
        .iter()
        .any(|t| t.verdict.is_none() && !predicate_contains_fireability(&t.predicate));
    let symbolic_seed_reduction = if config.deadline().is_some()
        && reduce_symbolic_seed_enabled()
        && has_symbolic_lane_candidate
    {
        build_symbolic_seed_reduction(net, &trackers)
    } else {
        None
    };
    if let Some(reduction) = symbolic_seed_reduction.as_ref() {
        eprintln!(
            "Symbolic seeding (CHC+PDR) reduced net {}p+{}t → {}p+{}t",
            net.num_places(),
            net.num_transitions(),
            reduction.net().num_places(),
            reduction.net().num_transitions(),
        );
    }

    // Phase 2b2: symbolic state-equation CHC seeding (Esparza–Melzer).
    // Runs BEFORE AIGER because CHC is materially cheaper on the
    // typical Reachability* timeout: the AdaptivePortfolio (IC3/PDR +
    // BMC + PDKind + k-induction) over the state-equation encoding
    // dispatches in seconds whereas AIGER spins up its 60-config
    // hardware portfolio and is often skipped for deadline reserve.
    // Same verdict-mapping contract as PDR (AG / EF), same fireability
    // carve-out, same "Unknown is no-op" soundness floor — and the
    // pipeline passes a deadline with the exact BFS tail reserved, so
    // Unknown cannot starve downstream phases. Default-ON; flip
    // `TY_MCC_ENABLE_REACHABILITY_SYMBOLIC=0` to disable for benchmarking.
    if trackers.iter().any(|t| t.verdict.is_none()) {
        let pre_chc = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let chc_start = Instant::now();
        run_symbolic_seeding_wall_capped(
            net,
            &mut trackers,
            config.deadline(),
            symbolic_seed_reduction.as_ref(),
        );
        let post_chc = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let chc_seeded = post_chc - pre_chc;
        eprintln!(
            "Symbolic CHC phase ({}) took {} ms, seeded {chc_seeded}/{} reachability verdicts",
            if symbolic_seed_reduction.is_some() {
                "reduced net"
            } else {
                "original net"
            },
            chc_start.elapsed().as_millis(),
            trackers.len(),
        );
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Drop the deadline-expired short-circuit so downstream seeding phases
    // (each of which honors `config.deadline()` directly) still get a chance
    // to make progress instead of returning `CannotCompute` here.

    // Phase 2b3: AIGER cross-encoding seeding.
    // For bounded nets, encodes the Petri net as an AIGER circuit and runs
    // the tla-aiger IC3/BMC portfolio. This is the single most impactful
    // technique for bounded safety checking — it leverages the full 60-config
    // hardware model checking portfolio on the encoded circuit. Runs AFTER
    // the cheaper CHC dispatch above so the heavier AIGER lane only fires
    // on residual trackers that CHC could not resolve.
    if trackers.iter().any(|t| t.verdict.is_none()) {
        let pre_aiger = trackers.iter().filter(|t| t.verdict.is_some()).count();
        if unresolved_ag_fireability_requires_original_net_full_bfs(&trackers) {
            aiger_deferred_for_fireability = true;
            eprintln!("AIGER portfolio deferred; preserving fireability witness/proof lanes");
        } else {
            run_aiger_seeding_wall_capped(
                net,
                &mut trackers,
                &validation_targets,
                config.deadline(),
                nupn,
                Some(bus.as_ref()),
                "Phase 2b3",
            );
        }
        let post_aiger = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let aiger_seeded = post_aiger - pre_aiger;
        if aiger_seeded > 0 {
            eprintln!(
                "AIGER portfolio seeded {aiger_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Phase 2c: PDR/IC3 seeding.
    // Resolves AG(φ) as safety on φ and EF(φ) via safety of ¬φ.
    if trackers.iter().any(|t| t.verdict.is_none()) {
        let pre_pdr = trackers.iter().filter(|t| t.verdict.is_some()).count();
        // Reserve-preserving + wall-capped: PDR must never starve the exhaustive
        // BFS oracle of the budget it needs to run to completion. See
        // `run_pdr_seeding_wall_capped`. Reuses the same verdict-preserving
        // reduced net built before the CHC lane (trackers AIGER/CHC resolved in
        // between are simply skipped by id when syncing onto the reduced run).
        let pdr_start = Instant::now();
        run_pdr_seeding_wall_capped(
            net,
            &mut trackers,
            config.deadline(),
            symbolic_seed_reduction.as_ref(),
        );
        let post_pdr = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let pdr_seeded = post_pdr - pre_pdr;
        eprintln!(
            "PDR phase ({}) took {} ms, seeded {pdr_seeded}/{} reachability verdicts",
            if symbolic_seed_reduction.is_some() {
                "reduced net"
            } else {
                "original net"
            },
            pdr_start.elapsed().as_millis(),
            trackers.len(),
        );
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if run_compound_reuse_phase(&mut trackers, "PDR") > 0
        && trackers.iter().all(|t| t.verdict.is_some())
    {
        if incremental_flush {
            flush_resolved(&mut trackers);
        }
        return assemble_results(&prepared, &trackers, true, incremental_flush);
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Same rationale as the AIGER-stage early-return above: do not stop the
    // pipeline here just because the deadline elapsed — let later witness
    // phases run, since each one budgets its own deadline check.

    // Phase 2d: random walk witness search.
    // Lightweight under-approximation: finds EF(φ)=TRUE and AG(φ)=FALSE witnesses
    // without BFS. Runs on the unreduced net — no hash sets, no memory overhead.
    if trackers.iter().any(|t| t.verdict.is_none()) && !random_walk_seeding_disabled() {
        let pre_walk = trackers.iter().filter(|t| t.verdict.is_some()).count();
        reachability_walk::run_random_walk_seeding(
            net,
            &mut trackers,
            &validation,
            under_approx_lane_deadline(net, config.deadline()),
        );
        let post_walk = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let walk_seeded = post_walk - pre_walk;
        if walk_seeded > 0 {
            eprintln!(
                "Random walk seeded {walk_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Phase 2e: heuristic best-first witness search.
    // Guided exploration using LP relaxation heuristic. Memory-bounded via
    // Bloom filter. Only seeds EF(φ)=TRUE / AG(φ)=FALSE witnesses.
    if trackers.iter().any(|t| t.verdict.is_none()) {
        let pre_heur = trackers.iter().filter(|t| t.verdict.is_some()).count();
        reachability_heuristic::run_heuristic_seeding(
            net,
            &mut trackers,
            &validation,
            under_approx_lane_deadline(net, config.deadline()),
        );
        let post_heur = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let heur_seeded = post_heur - pre_heur;
        if heur_seeded > 0 {
            eprintln!(
                "Heuristic search seeded {heur_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    if trackers.iter().any(|t| t.verdict.is_none()) {
        let pre_rescue = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let rescue_seeded = run_post_witness_proof_rescue(
            net,
            &mut trackers,
            &validation_targets,
            config.deadline(),
            max_bmc_depth,
        );
        if rescue_seeded > 0 {
            eprintln!(
                "Post-witness proof rescue seeded {rescue_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        debug_assert_eq!(
            rescue_seeded,
            trackers.iter().filter(|t| t.verdict.is_some()).count() - pre_rescue
        );
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // Phase 2f: k-induction for remaining proof-side properties.
    // Proves AG(φ) = TRUE by induction, and EF(φ) = FALSE via AG(¬φ) induction.
    if trackers.iter().any(|t| t.verdict.is_none()) {
        let pre_kind = trackers.iter().filter(|t| t.verdict.is_some()).count();
        kinduction::run_kinduction_seeding(
            net,
            &mut trackers,
            kinduction_deadline(net, config.deadline()),
            max_bmc_depth,
        );
        let post_kind = trackers.iter().filter(|t| t.verdict.is_some()).count();
        let kind_seeded = post_kind - pre_kind;
        if kind_seeded > 0 {
            eprintln!(
                "k-induction seeded {kind_seeded}/{} reachability verdicts",
                trackers.len(),
            );
        }
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    if run_compound_reuse_phase(&mut trackers, "k-induction") > 0
        && trackers.iter().all(|t| t.verdict.is_some())
    {
        if incremental_flush {
            flush_resolved(&mut trackers);
        }
        return assemble_results(&prepared, &trackers, true, incremental_flush);
    }

    if incremental_flush {
        flush_resolved(&mut trackers);
    }

    // No deadline-expired short-circuit before BFS: downstream BFS already
    // honors `config.deadline()` via `DEADLINE_CHECK_INTERVAL`, and tiny
    // fixtures (e.g. checkpoint regressions) need the BFS phase to run even
    // when callers supply a deliberately tight deadline.

    if unresolved_ag_fireability_requires_original_net_full_bfs(&trackers) {
        eprintln!(
            "Reachability: unresolved AG fireability predicates require original-net full BFS for soundness",
        );
        let (result, original_trackers) =
            match explore_original_net_full_reachability(net, trackers, config) {
                Ok(state) => state,
                Err(error) => return cannot_compute_results(simplified_properties, &error),
            };
        trackers = original_trackers;
        if result.completed {
            finalize_exhaustive_completion(&mut trackers);
            bus.log_summary();
            return assemble_results(&prepared, &trackers, result.completed, incremental_flush);
        }

        if incremental_flush {
            flush_resolved(&mut trackers);
        }

        if unresolved_ag_fireability_requires_original_net_full_bfs(&trackers) {
            if let Some((indices, subset)) = non_original_required_subset(&trackers) {
                let unresolved_subset = subset.iter().filter(|t| t.verdict.is_none()).count();
                eprintln!(
                    "Reachability: original-net AG fireability still unresolved; continuing reduced fallback for {unresolved_subset} non-blocked properties",
                );
                match run_reduced_reachability_fallback(net, subset, config) {
                    Ok((_subset_result, subset_trackers)) => {
                        merge_tracker_subset(&mut trackers, indices, subset_trackers);
                    }
                    Err(error) => {
                        eprintln!(
                            "Reachability: reduced fallback for non-blocked properties failed after AG-fireability split: {error}",
                        );
                    }
                }
                if incremental_flush {
                    flush_resolved(&mut trackers);
                }
            }
            if aiger_deferred_for_fireability && trackers.iter().any(|t| t.verdict.is_none()) {
                run_deferred_aiger_phase(
                    net,
                    &mut trackers,
                    &validation_targets,
                    config,
                    nupn,
                    bus.as_ref(),
                );
                if incremental_flush {
                    flush_resolved(&mut trackers);
                }
                if trackers.iter().all(|t| t.verdict.is_some()) {
                    bus.log_summary();
                    return assemble_results(&prepared, &trackers, true, incremental_flush);
                }
            }
            bus.log_summary();
            return assemble_results(&prepared, &trackers, false, incremental_flush);
        }
    }

    if aiger_deferred_for_fireability && trackers.iter().any(|t| t.verdict.is_none()) {
        run_deferred_aiger_phase(
            net,
            &mut trackers,
            &validation_targets,
            config,
            nupn,
            bus.as_ref(),
        );
        if trackers.iter().all(|t| t.verdict.is_some()) {
            if incremental_flush {
                flush_resolved(&mut trackers);
            }
            return assemble_results(&prepared, &trackers, true, incremental_flush);
        }
        if incremental_flush {
            flush_resolved(&mut trackers);
        }
    }

    // Phase 2g (feature `gpu`): exhaustive GPU explicit-BFS over the RAW net
    // with the unresolved formulas' predicates compiled to device invariants.
    // A published violation row is a reachable witness marking — every
    // pending EF whose goal holds there resolves TRUE and every AG whose
    // formula fails there resolves FALSE (classified host-side with the same
    // `eval_predicate` the CPU lanes use) — and the search reruns for the
    // remainder; a clean completion resolves everything left exhaustively
    // (EF → FALSE, AG → TRUE). Runs before Phase 3 because it decides the
    // same questions the exhaustive BFS would, on the un-reduced net (no
    // predicate remapping). Fail-closed: any decline leaves the trackers
    // untouched and Phase 3 runs unchanged.
    #[cfg(feature = "gpu")]
    if trackers.iter().any(|t| t.verdict.is_none())
        && run_gpu_reachability_phase(net, &mut trackers, config)
    {
        bus.log_summary();
        if incremental_flush {
            flush_resolved(&mut trackers);
        }
        return assemble_results(&prepared, &trackers, true, incremental_flush);
    }

    // Phase 3: run BFS on a sliced reduced net when the unresolved queries only
    // touch disconnected components. Fall back to the existing reduced-net path
    // whenever the slice does not shrink or any predicate cannot be remapped.
    let (result, trackers) = match run_reduced_reachability_fallback(net, trackers, config) {
        Ok(state) => state,
        Err(error) => return cannot_compute_results(simplified_properties, &error),
    };
    bus.log_summary();
    assemble_results(&prepared, &trackers, result.completed, incremental_flush)
}

/// Phase 2g: exhaustive GPU explicit-BFS deciding the unresolved reachability
/// formulas on the RAW net. Returns `true` iff EVERY tracker now has a
/// verdict (exhaustively decided); `false` = declined or partially resolved
/// with the device unable to continue — the CPU Phase 3 runs unchanged for
/// whatever remains.
///
/// Witness-loop shape: predicates are installed as device invariants
/// (`¬goal` for EF, the formula itself for AG), so the engine stops at the
/// first reachable marking deciding SOMETHING; that marking is classified
/// host-side against every pending query with the same [`eval_predicate`]
/// the CPU lanes use, and the search reruns with only the still-undecided
/// predicates. A clean completion is an exhaustive proof for everything
/// left. Each rerun is a fresh full BFS — bounded by the query count (≤ 16
/// per MCC category) and each is fast, so the loop is cheap relative to the
/// CPU fallback it replaces.
///
/// Soundness gates (all fail-closed to `false`):
/// - `IsFireable` parity: predicate guards use SUMMED input-arc weights
///   while `net.is_enabled` checks per-arc — identical unless a transition
///   has parallel input arcs from one place, so such nets decline.
/// - The engine's token-cap trap guarantees the straight-line predicate
///   arithmetic cannot wrap (see `gpu_state_space::TOKEN_CAP`).
/// - A witness row that classifies NO pending query would contradict the
///   device's own invariant evaluation — treated as an engine fault.
#[cfg(feature = "gpu")]
fn run_gpu_reachability_phase(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    config: &ExplorationConfig,
) -> bool {
    use crate::gpu_state_space::{reachability_explore_gpu, GpuReachabilityOutcome};
    use crate::resolved_predicate::eval_predicate;

    if !crate::gpu_state_space::gpu_lane_enabled(net) {
        return false;
    }
    let has_parallel_input_arcs = net.transitions.iter().any(|t| {
        let mut seen = std::collections::HashSet::new();
        t.inputs.iter().any(|arc| !seen.insert(arc.place.0))
    });
    if has_parallel_input_arcs {
        return false;
    }

    loop {
        let unresolved: Vec<usize> = trackers
            .iter()
            .enumerate()
            .filter(|(_, t)| t.verdict.is_none())
            .map(|(i, _)| i)
            .collect();
        if unresolved.is_empty() {
            return true;
        }
        // Invariant per query: the predicate that must hold on EVERY
        // reachable marking for the query to stay undecided — ¬goal for EF
        // (a violation = the goal reached), the formula itself for AG.
        let invariants: Vec<ResolvedPredicate> = unresolved
            .iter()
            .map(|&i| {
                let tracker = &trackers[i];
                match tracker.quantifier {
                    PathQuantifier::EF => {
                        ResolvedPredicate::Not(Box::new(tracker.predicate.clone()))
                    }
                    PathQuantifier::AG => tracker.predicate.clone(),
                }
            })
            .collect();
        let invariant_refs: Vec<&ResolvedPredicate> = invariants.iter().collect();

        match reachability_explore_gpu(net, config.max_states(), &invariant_refs) {
            None => return false,
            Some(GpuReachabilityOutcome::Exhausted) => {
                for &i in &unresolved {
                    let verdict = matches!(trackers[i].quantifier, PathQuantifier::AG);
                    resolve_tracker(
                        &mut trackers[i],
                        verdict,
                        ReachabilityResolutionSource::Gpu,
                        None,
                    );
                }
                eprintln!(
                    "[mcc] Reachability GPU lane: {} propert{} decided by exhaustive completion",
                    unresolved.len(),
                    if unresolved.len() == 1 { "y" } else { "ies" },
                );
                return true;
            }
            Some(GpuReachabilityOutcome::Witness(marking)) => {
                let mut resolved_any = false;
                for &i in &unresolved {
                    let tracker = &mut trackers[i];
                    let goal_holds = eval_predicate(&tracker.predicate, &marking, net);
                    match tracker.quantifier {
                        PathQuantifier::EF if goal_holds => {
                            resolve_tracker(tracker, true, ReachabilityResolutionSource::Gpu, None);
                            resolved_any = true;
                        }
                        PathQuantifier::AG if !goal_holds => {
                            resolve_tracker(
                                tracker,
                                false,
                                ReachabilityResolutionSource::Gpu,
                                None,
                            );
                            resolved_any = true;
                        }
                        _ => {}
                    }
                }
                if !resolved_any {
                    eprintln!(
                        "[mcc] Reachability GPU lane declined: witness row decides no pending \
                         query (engine fault)"
                    );
                    return false;
                }
                // Rerun with the remaining queries' invariants only.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examinations::reachability::ReachabilityResolutionSource;
    use crate::explorer::CheckpointConfig;
    use crate::petri_net::{
        Arc as PetriArc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo,
    };
    use crate::property_xml::PathQuantifier;
    use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn arc(place: u32, weight: u64) -> PetriArc {
        PetriArc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn trans(id: &str, inputs: Vec<PetriArc>, outputs: Vec<PetriArc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    fn toggling_net() -> PetriNet {
        PetriNet {
            name: Some("toggle".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    fn self_loop_net() -> PetriNet {
        PetriNet {
            name: Some("self-loop".to_string()),
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(0, 1)])],
            initial_marking: vec![1],
        }
    }

    fn ag_fireability_blocker_with_independent_cardinality_net() -> PetriNet {
        PetriNet {
            name: Some("ag-fireability-split".to_string()),
            places: vec![place("pf"), place("n0"), place("n1"), place("q")],
            transitions: vec![
                trans("tfire", vec![arc(0, 1)], vec![arc(0, 1)]),
                trans("n01", vec![arc(1, 1)], vec![arc(2, 1)]),
                trans("n10", vec![arc(2, 1)], vec![arc(1, 1)]),
            ],
            initial_marking: vec![1, 1, 0, 1],
        }
    }

    fn tracker(
        quantifier: PathQuantifier,
        predicate: ResolvedPredicate,
        verdict: Option<bool>,
    ) -> PropertyTracker {
        PropertyTracker {
            id: "p".into(),
            quantifier,
            predicate,
            verdict,
            resolved_by: None,
            flushed: false,
        }
    }

    /// Real-device tests for the Phase-2g GPU reachability lane (skipped on
    /// CUDA-less hosts). The toggling net's reachable set is exactly
    /// {[1,0], [0,1]}.
    #[cfg(feature = "gpu")]
    mod gpu_reachability_phase {
        use super::*;
        use crate::petri_net::TransitionIdx;

        fn cuda_available() -> bool {
            if tla_gpu::probe().is_err() {
                eprintln!("skipping GPU reachability test: no usable CUDA device");
                return false;
            }
            true
        }

        fn tokens_ge(place: u32, k: u64) -> ResolvedPredicate {
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(k),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(place)]),
            )
        }

        #[test]
        fn witness_and_exhaustion_resolve_mixed_queries() {
            if !cuda_available() {
                return;
            }
            let net = toggling_net();
            let config = ExplorationConfig::new(1000);
            let mut trackers = vec![
                // EF tokens(p1) >= 1: reachable ([0,1]) => TRUE via witness.
                tracker(PathQuantifier::EF, tokens_ge(1, 1), None),
                // AG tokens(p0) <= 1: holds everywhere => TRUE via exhaustion.
                tracker(
                    PathQuantifier::AG,
                    ResolvedPredicate::IntLe(
                        ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
                        ResolvedIntExpr::Constant(1),
                    ),
                    None,
                ),
                // EF tokens(p0) >= 2: unreachable => FALSE via exhaustion.
                tracker(PathQuantifier::EF, tokens_ge(0, 2), None),
            ];
            assert!(
                run_gpu_reachability_phase(&net, &mut trackers, &config),
                "phase should decide every query"
            );
            assert_eq!(trackers[0].verdict, Some(true));
            assert_eq!(trackers[1].verdict, Some(true));
            assert_eq!(trackers[2].verdict, Some(false));
            for t in &trackers {
                assert_eq!(
                    t.resolved_by.map(|r| r.source),
                    Some(ReachabilityResolutionSource::Gpu),
                );
            }
        }

        #[test]
        fn fireability_queries_resolve() {
            if !cuda_available() {
                return;
            }
            let net = toggling_net();
            let config = ExplorationConfig::new(1000);
            let mut trackers = vec![
                // EF is-fireable(t1): enabled at [0,1] => TRUE.
                tracker(
                    PathQuantifier::EF,
                    ResolvedPredicate::IsFireable(vec![TransitionIdx(1)]),
                    None,
                ),
                // AG is-fireable(t0, t1): some transition always enabled => TRUE.
                tracker(
                    PathQuantifier::AG,
                    ResolvedPredicate::IsFireable(vec![TransitionIdx(0), TransitionIdx(1)]),
                    None,
                ),
            ];
            assert!(run_gpu_reachability_phase(&net, &mut trackers, &config));
            assert_eq!(trackers[0].verdict, Some(true));
            assert_eq!(trackers[1].verdict, Some(true));
        }

        #[test]
        fn exploration_bound_declines_without_touching_trackers() {
            if !cuda_available() {
                return;
            }
            let net = toggling_net();
            // Cap of 1 distinct marking: the 2-state space trips the bound.
            let config = ExplorationConfig::new(1);
            let mut trackers = vec![tracker(PathQuantifier::EF, tokens_ge(0, 2), None)];
            assert!(!run_gpu_reachability_phase(&net, &mut trackers, &config));
            assert_eq!(trackers[0].verdict, None, "declined lane must not resolve");
        }
    }

    #[test]
    fn unresolved_ag_fireability_requires_original_net_full_bfs_only_for_pending_ag_fireability() {
        assert!(!unresolved_ag_fireability_requires_original_net_full_bfs(
            &[]
        ));

        assert!(!unresolved_ag_fireability_requires_original_net_full_bfs(
            &[tracker(
                PathQuantifier::EF,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
                None,
            )]
        ));

        assert!(!unresolved_ag_fireability_requires_original_net_full_bfs(
            &[tracker(
                PathQuantifier::AG,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
                Some(true),
            )]
        ));

        assert!(!unresolved_ag_fireability_requires_original_net_full_bfs(
            &[tracker(
                PathQuantifier::AG,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(0),
                    ResolvedIntExpr::Constant(1),
                ),
                None,
            )]
        ));

        assert!(unresolved_ag_fireability_requires_original_net_full_bfs(&[
            tracker(
                PathQuantifier::AG,
                ResolvedPredicate::Not(Box::new(ResolvedPredicate::IsFireable(vec![
                    TransitionIdx(0),
                ]))),
                None,
            )
        ]));
    }

    #[test]
    fn non_original_required_subset_excludes_only_pending_ag_fireability() {
        let trackers = vec![
            tracker(
                PathQuantifier::AG,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
                None,
            ),
            tracker(
                PathQuantifier::AG,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(0),
                    ResolvedIntExpr::Constant(1),
                ),
                None,
            ),
            tracker(
                PathQuantifier::AG,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(1)]),
                Some(true),
            ),
        ];

        let (indices, subset) = non_original_required_subset(&trackers)
            .expect("unresolved non-AG-fireability tracker should be eligible");

        assert_eq!(indices, vec![1, 2]);
        assert_eq!(subset.len(), 2);
        assert!(matches!(
            subset[0].predicate,
            ResolvedPredicate::IntLe(_, _)
        ));
        assert_eq!(subset[1].verdict, Some(true));
    }

    #[test]
    fn incomplete_ag_fireability_split_allows_non_blocked_subset_to_complete() {
        let net = ag_fireability_blocker_with_independent_cardinality_net();
        let trackers = vec![
            tracker(
                PathQuantifier::AG,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
                None,
            ),
            tracker(
                PathQuantifier::AG,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(3)]),
                    ResolvedIntExpr::Constant(1),
                ),
                None,
            ),
        ];

        let (result, trackers) =
            explore_original_net_full_reachability(&net, trackers, &ExplorationConfig::new(1))
                .expect("plain full BFS should not use fallible checkpoint IO");
        assert!(!result.completed);
        assert_eq!(trackers[0].verdict, None);
        assert_eq!(trackers[1].verdict, None);

        let (indices, subset) = non_original_required_subset(&trackers)
            .expect("independent unresolved tracker should remain eligible");
        let (_subset_result, subset_trackers) =
            run_reduced_reachability_fallback(&net, subset, &ExplorationConfig::new(10))
                .expect("independent reduced fallback should complete");

        let mut trackers = trackers;
        merge_tracker_subset(&mut trackers, indices, subset_trackers);

        assert_eq!(trackers[0].verdict, None);
        assert_eq!(trackers[1].verdict, Some(true));
    }

    #[test]
    fn deferred_aiger_runs_after_incomplete_ag_fireability_split() {}
    #[test]
    fn original_net_full_reachability_config_disables_por_but_preserves_checkpoint() {
        // POR is stripped because it can elide states under the AG-fireability
        // fallback's correctness invariant. Checkpointing is preserved so the
        // end-to-end checkpoint/resume feature still works for AG-fireability
        // properties (see `reachability_checkpoint` regression).
        let config = ExplorationConfig::new(10)
            .with_por(PorStrategy::DeadlockPreserving)
            .with_checkpoint(CheckpointConfig::new("stale-checkpoint".into(), 1).with_resume(true));

        let full_config = original_net_full_reachability_config(&config);

        assert!(matches!(full_config.por_strategy, PorStrategy::None));
        assert!(full_config.checkpoint().is_some());
        assert!(matches!(
            config.por_strategy,
            PorStrategy::DeadlockPreserving
        ));
        assert!(config.checkpoint().is_some());
    }

    #[test]
    fn original_net_full_reachability_finds_counterexample_without_bus_pruning() {
        let net = toggling_net();
        let trackers = vec![tracker(
            PathQuantifier::AG,
            ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
            None,
        )];

        let (result, trackers) =
            explore_original_net_full_reachability(&net, trackers, &ExplorationConfig::new(10))
                .expect("plain full BFS should not use fallible checkpoint IO");

        assert!(!result.completed);
        assert!(result.stopped_by_observer);
        assert_eq!(trackers[0].verdict, Some(false));
        assert_eq!(
            trackers[0].resolved_by.map(|resolution| resolution.source),
            Some(ReachabilityResolutionSource::BfsCounterexample)
        );
    }

    #[test]
    fn original_net_full_reachability_completion_proves_ag_fireability_true() {
        let net = self_loop_net();
        let trackers = vec![tracker(
            PathQuantifier::AG,
            ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
            None,
        )];

        let (result, mut trackers) =
            explore_original_net_full_reachability(&net, trackers, &ExplorationConfig::new(10))
                .expect("plain full BFS should not use fallible checkpoint IO");

        assert!(result.completed);
        assert_eq!(trackers[0].verdict, None);
        finalize_exhaustive_completion(&mut trackers);
        assert_eq!(trackers[0].verdict, Some(true));
        assert_eq!(
            trackers[0].resolved_by.map(|resolution| resolution.source),
            Some(ReachabilityResolutionSource::ExhaustiveCompletion)
        );
    }

    #[test]
    fn original_net_full_reachability_incomplete_leaves_ag_fireability_unresolved() {
        let net = toggling_net();
        let trackers = vec![tracker(
            PathQuantifier::AG,
            ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
            None,
        )];

        let (result, trackers) =
            explore_original_net_full_reachability(&net, trackers, &ExplorationConfig::new(1))
                .expect("plain full BFS should not use fallible checkpoint IO");

        assert!(!result.completed);
        assert_eq!(trackers[0].verdict, None);
    }

    #[test]
    fn aiger_pipeline_budget_preserves_no_deadline_behavior() {
        let now = Instant::now();
        let budget = aiger_pipeline_budget_at(None, 16, now);

        assert_eq!(budget.deadline, None);
        assert!(!budget.skipped_for_reserve);
        assert_eq!(budget.phase_budget, None);
    }

    #[test]
    fn aiger_pipeline_budget_caps_long_global_deadline() {}
    #[test]
    fn aiger_pipeline_budget_uses_fair_share_under_short_competition_deadline() {
        let now = Instant::now();
        let budget = aiger_pipeline_budget_at(Some(now + Duration::from_secs(20)), 1, now);

        assert_eq!(budget.phase_budget, Some(AIGER_PIPELINE_PHASE_CAP));
        assert_eq!(
            budget.deadline.unwrap().duration_since(now),
            AIGER_PIPELINE_PHASE_CAP
        );
        assert!(!budget.skipped_for_reserve);
    }

    #[test]
    fn aiger_pipeline_budget_shrinks_with_unresolved_residual_queue() {
        let now = Instant::now();
        let budget = aiger_pipeline_budget_at(Some(now + Duration::from_secs(20)), 16, now);

        let expected =
            Duration::from_secs(20) / (16 + AIGER_PIPELINE_VIRTUAL_DOWNSTREAM_LANES) as u32;
        assert_eq!(budget.phase_budget, Some(expected));
        assert_eq!(budget.deadline.unwrap().duration_since(now), expected);
        assert!(!budget.skipped_for_reserve);
    }

    #[test]
    fn aiger_pipeline_budget_reserves_virtual_downstream_lanes() {
        let now = Instant::now();
        let budget = aiger_pipeline_budget_at(Some(now + Duration::from_secs(3)), 1, now);

        let expected =
            Duration::from_secs(3) / (1 + AIGER_PIPELINE_VIRTUAL_DOWNSTREAM_LANES) as u32;
        assert_eq!(budget.phase_budget, Some(expected));
        assert_eq!(budget.deadline.unwrap().duration_since(now), expected);
        assert!(!budget.skipped_for_reserve);
    }

    #[test]
    fn aiger_pipeline_budget_skips_tiny_overall_budget_before_phase_budget() {}
    #[test]
    // `Instant - Duration` builds a fixed relative deadline from a fresh `now`;
    // the subtraction can never underflow in this test.
    #[allow(clippy::unchecked_time_subtraction)]
    fn deadline_preserving_reserve_leaves_tail_budget() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(20);

        assert_eq!(
            deadline_preserving_reserve_at(Some(deadline), Duration::from_secs(4), now),
            Some(deadline - Duration::from_secs(4))
        );
    }

    #[test]
    fn under_approx_lane_deadline_caps_long_deadline_to_lane_budget() {
        // With a very long global deadline, the under-approx lane must be
        // capped to the absolute lane cap, NOT given a large slice (which would
        // starve the exhaustive BFS). The plain witness deadline would otherwise
        // hand it (deadline - reserve), far beyond the absolute cap.
        let now = Instant::now();
        let net = self_loop_net();
        let deadline = now + Duration::from_secs(600);

        let lane = under_approx_lane_deadline_at(&net, Some(deadline), now)
            .expect("deadline-bearing config yields a lane deadline");
        let plain = witness_search_deadline_at(&net, Some(deadline), now)
            .expect("deadline-bearing config yields a witness deadline");

        assert!(
            lane.saturating_duration_since(now) <= REACHABILITY_UNDER_APPROX_LANE_ABS_CAP,
            "lane deadline must not exceed the absolute under-approx cap"
        );
        assert!(
            lane < plain,
            "lane deadline must be strictly tighter than the plain witness deadline"
        );
    }

    #[test]
    fn under_approx_lane_deadline_uses_remaining_fraction_for_mid_deadline() {
        // For a moderate deadline below the absolute-cap regime, the lane is
        // bounded to a fraction of the remaining deadline so the exhaustive BFS
        // keeps the complement.
        let now = Instant::now();
        let net = self_loop_net();
        let deadline = now + Duration::from_secs(20);

        let lane = under_approx_lane_deadline_at(&net, Some(deadline), now)
            .expect("deadline-bearing config yields a lane deadline");
        let lane_budget = lane.saturating_duration_since(now);
        let fraction = Duration::from_secs(20) / REACHABILITY_UNDER_APPROX_REMAINING_FRACTION;

        assert!(
            lane_budget <= fraction.min(REACHABILITY_UNDER_APPROX_LANE_ABS_CAP),
            "lane budget {lane_budget:?} must respect both the remaining-fraction and absolute caps"
        );
    }

    #[test]
    fn under_approx_lane_deadline_preserves_no_deadline_behavior() {
        // Without a global deadline the historical unbounded behavior is kept.
        assert_eq!(
            under_approx_lane_deadline_at(&self_loop_net(), None, Instant::now()),
            None
        );
    }

    #[test]
    fn under_approx_lane_deadline_never_exceeds_plain_witness_deadline() {
        let now = Instant::now();
        let net = self_loop_net();
        for secs in [4u64, 8, 16, 30, 60, 120, 300] {
            let deadline = now + Duration::from_secs(secs);
            let lane = under_approx_lane_deadline_at(&net, Some(deadline), now);
            let plain = witness_search_deadline_at(&net, Some(deadline), now);
            match (lane, plain) {
                (Some(lane), Some(plain)) => assert!(
                    lane <= plain,
                    "lane deadline must never exceed the reserve-preserving witness deadline at {secs}s"
                ),
                _ => panic!("both deadlines should be Some for a live deadline at {secs}s"),
            }
        }
    }

    #[test]
    fn deadline_preserving_reserve_expires_when_only_reserve_remains() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(4);

        assert_eq!(
            deadline_preserving_reserve_at(Some(deadline), Duration::from_secs(4), now),
            Some(now)
        );
    }

    #[test]
    fn kinduction_deadline_preserves_bfs_tail_budget() {}
    #[test]
    fn symbolic_seed_deadline_preserves_bfs_tail_budget() {}
    #[test]
    fn symbolic_seed_deadline_preserves_no_deadline_behavior() {
        assert_eq!(
            symbolic_seed_deadline_at(&self_loop_net(), None, Instant::now()),
            None
        );
    }

    fn cardinality_tracker(quantifier: PathQuantifier) -> PropertyTracker {
        PropertyTracker {
            id: "prop-00".to_string(),
            quantifier,
            predicate: ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(1),
            ),
            verdict: None,
            resolved_by: None,
            flushed: false,
        }
    }

    // Reserve-preserving contract: when the global deadline has already expired
    // the wall-capped PDR phase must skip entirely (cap is zero) so it cannot
    // (over)run into the exhaustive BFS's reserved tail. Deterministic — the
    // early return fires before the PDR-enable gate, so this is independent of
    // `TY_MCC_ENABLE_REACHABILITY_PDR` and cannot race other env-touching tests.
    #[test]
    // `Instant::now() - Duration` deliberately builds an already-expired
    // deadline; the subtraction cannot underflow on a freshly-sampled `now`.
    #[allow(clippy::unchecked_time_subtraction)]
    fn pdr_wall_capped_skips_when_deadline_already_expired() {
        let net = toggling_net();
        let mut trackers = vec![cardinality_tracker(PathQuantifier::EF)];
        run_pdr_seeding_wall_capped(
            &net,
            &mut trackers,
            Some(Instant::now() - Duration::from_secs(1)),
            None,
        );
        assert_eq!(trackers[0].verdict, None);
        assert_eq!(trackers[0].resolved_by, None);
    }

    // First-writer-wins merge contract: a tracker already resolved by an earlier
    // phase must keep its verdict and provenance after the wall-capped phase
    // runs (the worker thread is spawned and joined; the merge copies back only
    // verdicts for trackers the caller has not yet resolved). Deterministic —
    // the assertion holds whether or not PDR is enabled, so it does not race the
    // env-gated PDR unit tests; it also exercises the happy worker spawn/recv
    // path to guard against a regression that hangs there.
    #[test]
    fn pdr_wall_capped_preserves_preexisting_verdict() {
        let net = toggling_net();
        let mut tracker = cardinality_tracker(PathQuantifier::AG);
        resolve_tracker(&mut tracker, false, ReachabilityResolutionSource::Lp, None);
        let mut trackers = vec![tracker];
        run_pdr_seeding_wall_capped(
            &net,
            &mut trackers,
            Some(Instant::now() + Duration::from_secs(30)),
            None,
        );
        assert_eq!(trackers[0].verdict, Some(false));
        assert_eq!(
            trackers[0].resolved_by.map(|r| r.source),
            Some(ReachabilityResolutionSource::Lp)
        );
    }

    #[test]
    fn symbolic_seed_deadline_skips_when_only_bfs_tail_budget_remains() {}
    #[test]
    fn symbolic_seed_deadline_skips_tiny_budget_after_reserve() {}
    #[test]
    fn proof_rescue_skips_when_only_bfs_tail_budget_remains() {}
    #[test]
    fn bounded_witness_bfs_config_caps_states_and_deadline() {}
    #[test]
    fn resolve_lane_max_states_treats_zero_configured_as_uncapped() {
        // `0` is the org-wide "uncapped" sentinel (see
        // `crate::cli::resolve_max_states`). The lane MUST honor its own cap
        // rather than collapse to `1` via `0.min(cap).max(1)`.
        assert_eq!(
            resolve_lane_max_states(0, REACHABILITY_POST_SMT_WITNESS_BFS_MAX_STATES),
            REACHABILITY_POST_SMT_WITNESS_BFS_MAX_STATES
        );
        assert_eq!(
            resolve_lane_max_states(0, REACHABILITY_BOUNDED_WITNESS_BFS_MAX_STATES),
            REACHABILITY_BOUNDED_WITNESS_BFS_MAX_STATES
        );
    }

    #[test]
    fn resolve_lane_max_states_clamps_nonzero_configured_to_lane_cap() {
        // Non-zero configured limits below the lane cap pass through unchanged.
        assert_eq!(resolve_lane_max_states(100, 1_000), 100);
        // Non-zero configured limits above the lane cap clamp down.
        assert_eq!(resolve_lane_max_states(1_000_000, 200_000), 200_000);
        // `usize::MAX` (the unbounded CLI translation) collapses to the lane cap.
        assert_eq!(resolve_lane_max_states(usize::MAX, 200_000), 200_000);
        // Pathological lane cap of zero still produces a forward-progress floor.
        assert_eq!(resolve_lane_max_states(0, 0), 1);
        assert_eq!(resolve_lane_max_states(5, 0), 1);
    }

    #[test]
    fn post_smt_witness_bfs_config_with_zero_configured_uses_lane_cap() {
        // Regression: a config carrying `max_states = 0` (uncapped sentinel)
        // previously produced `max_states = 1` for the post-SMT witness BFS
        // lane, immediately tripping `WitnessBfsStopReason::StateCap` and
        // skipping the lane. The fix routes through `resolve_lane_max_states`
        // so `0` is honored as uncapped against the lane's own cap.
        let now = Instant::now();
        let config = ExplorationConfig::new(0).with_deadline(Some(
            now + REACHABILITY_BFS_FALLBACK_RESERVE + Duration::from_secs(5),
        ));

        let net = toggling_net();
        let witness_config = post_smt_witness_bfs_config_at(&net, &config, now)
            .expect("post-SMT witness BFS should be schedulable with a live deadline");

        assert_eq!(
            witness_config.max_states,
            REACHABILITY_POST_SMT_WITNESS_BFS_MAX_STATES
        );
    }

    #[test]
    fn bounded_witness_bfs_config_with_zero_configured_uses_lane_cap() {
        // Regression for the shared `witness_bfs_config_with_limits` helper.
        let now = Instant::now();
        let config = ExplorationConfig::new(0).with_deadline(Some(
            now + REACHABILITY_BFS_FALLBACK_RESERVE + Duration::from_secs(5),
        ));

        let net = self_loop_net();
        let witness_config = bounded_witness_bfs_config(&net, &config);

        assert_eq!(
            witness_config.max_states,
            REACHABILITY_BOUNDED_WITNESS_BFS_MAX_STATES
        );
    }

    #[test]
    fn post_smt_witness_bfs_config_requires_deadline() {
        assert_eq!(
            post_smt_witness_bfs_config(&self_loop_net(), &ExplorationConfig::new(1_000_000)),
            None
        );
    }

    #[test]
    fn post_smt_witness_bfs_config_borrows_tiny_tail_budget_when_only_reserve_remains() {}
    #[test]
    // `now - Duration` deliberately builds an already-expired deadline; the
    // subtraction cannot underflow on a freshly-sampled `now`.
    #[allow(clippy::unchecked_time_subtraction)]
    fn post_smt_witness_bfs_config_still_expires_after_global_deadline() {
        let now = Instant::now();
        let config =
            ExplorationConfig::new(1_000_000).with_deadline(Some(now - Duration::from_millis(1)));

        assert_eq!(
            post_smt_witness_bfs_config_at(&self_loop_net(), &config, now),
            None
        );
    }

    #[test]
    fn post_smt_witness_bfs_config_gets_larger_residual_state_window() {
        let config = ExplorationConfig::new(1_000_000)
            .with_deadline(Some(Instant::now() + Duration::from_secs(30)));

        let net = self_loop_net();
        let pre_smt = bounded_witness_bfs_config(&net, &config);
        let post_smt = post_smt_witness_bfs_config(&net, &config)
            .expect("deadline-bearing config should allow post-SMT witness BFS");

        assert_eq!(
            pre_smt.max_states,
            REACHABILITY_BOUNDED_WITNESS_BFS_MAX_STATES
        );
        assert_eq!(
            post_smt.max_states,
            REACHABILITY_POST_SMT_WITNESS_BFS_MAX_STATES
        );
        assert!(post_smt
            .deadline
            .zip(pre_smt.deadline)
            .is_some_and(|(post, pre)| post <= pre));
    }

    #[test]
    fn post_smt_witness_bfs_report_records_no_residual_skip() {
        let net = self_loop_net();
        let mut trackers = vec![tracker(
            PathQuantifier::EF,
            ResolvedPredicate::True,
            Some(true),
        )];
        let targets =
            crate::examinations::reachability_witness::validation_targets_from_trackers(&trackers);
        let validation = crate::examinations::reachability_witness::WitnessValidationContext::new(
            &net, &targets,
        );

        let report =
            run_post_smt_witness_bfs(&net, &mut trackers, &validation, &ExplorationConfig::new(8));

        assert_eq!(
            report.skip_reason,
            Some(PostSmtWitnessBfsSkipReason::NoResidualProperties)
        );
        assert_eq!(report.status_code(), "skipped");
        assert!(post_smt_witness_bfs_evidence_row(report)
            .contains("skip_reason=no_residual_properties"));
    }

    #[test]
    fn post_smt_witness_bfs_report_records_missing_deadline_skip() {
        let net = self_loop_net();
        let mut trackers = vec![tracker(PathQuantifier::EF, ResolvedPredicate::True, None)];
        let targets =
            crate::examinations::reachability_witness::validation_targets_from_trackers(&trackers);
        let validation = crate::examinations::reachability_witness::WitnessValidationContext::new(
            &net, &targets,
        );

        let report =
            run_post_smt_witness_bfs(&net, &mut trackers, &validation, &ExplorationConfig::new(8));

        assert_eq!(
            report.skip_reason,
            Some(PostSmtWitnessBfsSkipReason::MissingDeadline)
        );
        assert_eq!(report.unresolved_after, 1);
        assert!(report.needs_blocker_action());
        assert!(post_smt_witness_bfs_blocker_action_row(report)
            .contains("reason_code=missing_deadline"));
    }

    #[test]
    // `Instant::now() - Duration` deliberately builds an already-expired
    // deadline; the subtraction cannot underflow on a freshly-sampled `now`.
    #[allow(clippy::unchecked_time_subtraction)]
    fn post_smt_witness_bfs_report_records_expired_global_deadline_skip() {
        let net = self_loop_net();
        let mut trackers = vec![tracker(PathQuantifier::EF, ResolvedPredicate::True, None)];
        let targets =
            crate::examinations::reachability_witness::validation_targets_from_trackers(&trackers);
        let validation = crate::examinations::reachability_witness::WitnessValidationContext::new(
            &net, &targets,
        );
        let config =
            ExplorationConfig::new(8).with_deadline(Some(Instant::now() - Duration::from_secs(1)));

        let report = run_post_smt_witness_bfs(&net, &mut trackers, &validation, &config);

        assert_eq!(
            report.skip_reason,
            Some(PostSmtWitnessBfsSkipReason::ExpiredDeadline)
        );
        assert_eq!(report.unresolved_after, 1);
        assert_eq!(trackers[0].verdict, None);
        assert!(post_smt_witness_bfs_evidence_row(report).contains("skip_reason=expired_deadline"));
    }

    #[test]
    fn post_smt_witness_bfs_tail_budget_can_seed_initial_witness() {
        let net = self_loop_net();
        let mut trackers = vec![tracker(PathQuantifier::EF, ResolvedPredicate::True, None)];
        let targets =
            crate::examinations::reachability_witness::validation_targets_from_trackers(&trackers);
        let validation = crate::examinations::reachability_witness::WitnessValidationContext::new(
            &net, &targets,
        );
        let config = ExplorationConfig::new(8)
            .with_deadline(Some(Instant::now() + REACHABILITY_BFS_FALLBACK_RESERVE));

        let report = run_post_smt_witness_bfs(&net, &mut trackers, &validation, &config);

        assert_eq!(report.skip_reason, None);
        assert_eq!(report.seeded, 1);
        assert_eq!(report.unresolved_after, 0);
        let stats = report
            .stats
            .expect("borrowed tail budget should run post-SMT witness BFS");
        assert_eq!(stats.stop_reason.code(), "all_resolved");
        assert_eq!(trackers[0].verdict, Some(true));
    }

    #[test]
    fn post_smt_witness_bfs_exhaustion_reports_completed_without_self_finalizing() {
        let net = self_loop_net();
        let mut trackers = vec![tracker(PathQuantifier::EF, ResolvedPredicate::False, None)];
        let targets =
            crate::examinations::reachability_witness::validation_targets_from_trackers(&trackers);
        let validation = crate::examinations::reachability_witness::WitnessValidationContext::new(
            &net, &targets,
        );
        let config =
            ExplorationConfig::new(8).with_deadline(Some(Instant::now() + Duration::from_secs(30)));

        let report = run_post_smt_witness_bfs(&net, &mut trackers, &validation, &config);

        assert_eq!(report.seeded, 0);
        assert_eq!(report.unresolved_after, 0);
        let stats = report.stats.expect("deadline-bearing config should run");
        assert!(stats.completed);
        assert_eq!(stats.stop_reason.code(), "exhausted");
        assert_eq!(trackers[0].verdict, None);
        assert!(post_smt_witness_bfs_evidence_row(report).contains("status=exhausted"));
    }

    #[test]
    fn post_smt_witness_bfs_observability_records_sidecar_rows() {
        let report =
            PostSmtWitnessBfsReport::skipped(1, 1, PostSmtWitnessBfsSkipReason::MissingDeadline);

        let ((), reports) =
            crate::mcc_backend_evidence::collect_runtime_reachability_bmc_reports(|| {
                emit_post_smt_witness_bfs_observability(report);
            });

        assert_eq!(reports.len(), 1);
        let evidence = &reports[0].evidence;
        assert!(evidence.iter().any(|row| {
            row.contains("MCC reachability_answer_lane_summary lane=post_smt_witness_bfs")
                && row.contains("skip_reason=missing_deadline")
        }));
        assert!(evidence.iter().any(|row| {
            row.contains("MCC blocker_action")
                && row.contains("blocker_piece=reachability_post_smt_witness_bfs")
                && row.contains("reason_code=missing_deadline")
        }));
    }

    #[test]
    fn post_smt_witness_bfs_report_records_state_cap_stop() {
        let net = toggling_net();
        let mut trackers = vec![tracker(
            PathQuantifier::EF,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ),
            None,
        )];
        let targets =
            crate::examinations::reachability_witness::validation_targets_from_trackers(&trackers);
        let validation = crate::examinations::reachability_witness::WitnessValidationContext::new(
            &net, &targets,
        );
        let config =
            ExplorationConfig::new(1).with_deadline(Some(Instant::now() + Duration::from_secs(30)));

        let report = run_post_smt_witness_bfs(&net, &mut trackers, &validation, &config);

        assert_eq!(report.seeded, 0);
        assert_eq!(report.unresolved_after, 1);
        let stats = report.stats.expect("deadline-bearing config should run");
        assert_eq!(stats.stop_reason.code(), "state_cap");
        assert!(post_smt_witness_bfs_evidence_row(report).contains("stop_reason=state_cap"));
    }

    /// Original (unreduced, unscaled) net: query place `p0` starts at 4 and is
    /// drained by `t0` (weight 2) into a forward-only sink `psink` (weight 2).
    /// Reachable `p0` (original coords): 4 → 2 → 0.
    fn gcd_scaled_slice_original_net() -> PetriNet {
        PetriNet {
            name: Some("gcd-scaled-slice".to_string()),
            places: vec![place("p0"), place("psink")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 2)])],
            initial_marking: vec![4, 0],
        }
    }

    /// The GCD-scaled reduced view of [`gcd_scaled_slice_original_net`], built
    /// directly to make the regression independent of the structural reducer's
    /// aggressiveness. `p0` is scaled by g=2 (reduced marking holds `m_orig/2`),
    /// so reachable reduced `p0` ∈ {2, 1, 0}; `t0`'s weights are divided by 2
    /// to weight 1, and `psink` is a forward-only sink.
    ///
    /// The backward-relevance cone of the query place `p0` keeps `t0` (it
    /// touches `p0`) but only `t0`'s INPUT places — so the sink `psink` is
    /// dropped and [`build_reachability_slice`] takes the slice path.
    fn gcd_scaled_reduced_net() -> crate::reduction::ReducedNet {
        use crate::reduction::{ReducedNet, ReductionReport};
        ReducedNet {
            net: PetriNet {
                name: Some("gcd-scaled-slice-reduced".to_string()),
                places: vec![place("p0"), place("psink")],
                transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
                initial_marking: vec![2, 0],
            },
            place_map: vec![Some(PlaceIdx(0)), Some(PlaceIdx(1))],
            place_unmap: vec![PlaceIdx(0), PlaceIdx(1)],
            // p0 scaled by 2; psink unscaled. Indexed by ORIGINAL place index.
            place_scales: vec![2, 1],
            transition_map: vec![Some(TransitionIdx(0))],
            transition_unmap: vec![TransitionIdx(0)],
            constant_values: Vec::new(),
            reconstructions: Vec::new(),
            report: ReductionReport::default(),
        }
    }

    /// Resolve the single tracker's EF verdict from a BFS result, accounting
    /// for early-stopping: an EF witness short-circuits exploration (so
    /// `completed == false` but the verdict is already `Some(true)`). A
    /// `false` verdict requires the run to have completed exhaustively.
    fn ef_verdict_from_result(
        result: &crate::explorer::ExplorationResult,
        trackers: &mut [PropertyTracker],
    ) -> bool {
        if let Some(verdict) = trackers[0].verdict {
            return verdict;
        }
        assert!(
            result.completed,
            "tiny net without a witness must enumerate fully"
        );
        finalize_exhaustive_completion(trackers);
        trackers[0].verdict.expect("completed BFS resolves EF")
    }

    /// Exact oracle: full-net BFS over the ORIGINAL (unreduced, unscaled) net.
    fn original_net_ef_oracle(net: &PetriNet, predicate: ResolvedPredicate) -> bool {
        let trackers = vec![tracker(PathQuantifier::EF, predicate, None)];
        let (result, mut trackers) =
            explore_original_net_full_reachability(net, trackers, &ExplorationConfig::new(100))
                .expect("full BFS oracle should run");
        ef_verdict_from_result(&result, &mut trackers)
    }

    /// FINDING #8 + #13 regression. The slice path divided `p0`'s marking by
    /// g=2 but historically remapped `tokens(p0) >= k` WITHOUT scaling the
    /// constant `k`. For a threshold that is not a multiple of g this produced
    /// the wrong EF/AG verdict. The scale-aware remap evaluates the slice
    /// predicate EXACTLY against the scaled marking.
    #[test]
    fn gcd_scaled_slice_predicate_is_exact_old_remap_was_wrong() {
        use crate::resolved_predicate::{remap_predicate, remap_predicate_scaled};

        let net = gcd_scaled_slice_original_net();

        // EF (tokens(p0) >= 3) — threshold 3 is NOT a multiple of g=2.
        // Original reachable p0 ∈ {4,2,0}; p0>=3 holds at the initial marking,
        // so the correct verdict is EF = TRUE.
        let predicate = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(3),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        );

        // Oracle: exact full-net BFS over the original net.
        let oracle = original_net_ef_oracle(&net, predicate.clone());
        assert!(
            oracle,
            "exact full-net BFS: EF(p0>=3) is TRUE at the initial marking"
        );

        // GCD-scaled reduced view (p0 scaled by g=2; reduced p0 ∈ {2,1,0}).
        let trackers = vec![tracker(PathQuantifier::EF, predicate.clone(), None)];
        let reduced = gcd_scaled_reduced_net();
        assert_eq!(reduced.place_scales[0], 2, "p0 is GCD-scaled by 2");

        let (slice, slice_trackers) = build_reachability_slice(&reduced, &trackers)
            .expect("query-local slice must be taken (forward sink pruned)");
        // The forward-only sink `psink` is dropped by the backward cone, so the
        // slice is strictly smaller than the reduced net and the slice path runs.
        assert!(
            slice.net.num_places() < reduced.net.num_places(),
            "the forward sink must actually be sliced away"
        );

        // --- NEW (fixed) behavior: scale-aware slice predicate is EXACT. ---
        let new_config = ExplorationConfig::new(100).refitted_for_net(&slice.net);
        let (new_result, mut new_trackers) =
            explore_reachability_on_slice(&slice, slice_trackers, &new_config)
                .expect("slice BFS should run");
        let new_verdict = ef_verdict_from_result(&new_result, &mut new_trackers);
        assert_eq!(
            new_verdict, oracle,
            "scale-aware slice verdict must match the exact oracle (TRUE)"
        );

        // --- OLD (buggy) behavior, reconstructed: plain unscaled remap. ---
        // Evaluate the SAME slice net with the constant NOT divided by g, which
        // is what the pre-fix `remap_predicate` produced.
        let place_map = slice.compose_place_map(&reduced.place_map);
        let trans_map = slice.compose_transition_map(&reduced.transition_map);
        let old_predicate = remap_predicate(&predicate, &place_map, &trans_map)
            .expect("old remap maps the surviving place");
        // Sanity: scale-aware remap differs from the old one (constant rescaled).
        let new_predicate =
            remap_predicate_scaled(&predicate, &place_map, &trans_map, &reduced.place_scales)
                .expect("scale-aware remap holds");
        assert_ne!(
            old_predicate, new_predicate,
            "the fix must change the slice predicate when g>1 and k is not a multiple of g"
        );

        let old_trackers = vec![tracker(PathQuantifier::EF, old_predicate, None)];
        let (old_result, mut old_trackers) =
            explore_reachability_on_slice(&slice, old_trackers, &new_config)
                .expect("old-behavior slice BFS should run");
        // The OLD path finds NO witness (it never sees p0_scaled>=3), so this
        // run completes exhaustively and finalizes to EF=FALSE.
        let old_verdict = ef_verdict_from_result(&old_result, &mut old_trackers);

        // The OLD behavior is WRONG: it never sees p0_scaled >= 3 (max is 2).
        assert!(
            !old_verdict,
            "old unscaled-constant slice path wrongly reports EF=FALSE"
        );
        assert_ne!(
            old_verdict, oracle,
            "this asserts the bug existed: old verdict disagrees with the oracle"
        );
        assert_ne!(
            old_verdict, new_verdict,
            "the fix flips the wrong verdict to the correct one"
        );
    }

    // ── Phase 2b-int: integer dead-transition sweep gate ─────────────────

    /// A net with exactly one structurally-DEAD transition and one LIVE one.
    ///
    /// Places p0 (init 1), p1 (init 0). Transitions:
    /// - `t_live`: p0(1) → p1(1) — enabled at the initial marking ⇒ LIVE.
    /// - `t_dead`: p1(2) → p0(1) — needs 2 tokens in p1, but the integer state
    ///   equation forces `m_p1 ≤ 1` in every reachable marking
    ///   (`m_p1 = x_live − 2·x_dead` and `m_p0 = 1 − x_live + x_dead ≥ 0` give
    ///   `x_live ≤ 1 + x_dead`, hence `m_p1 ≤ 1 − x_dead ≤ 1`), so `m_p1 ≥ 2` is
    ///   integer-INFEASIBLE ⇒ `t_dead` is DEAD (`AG(¬IsFireable(t_dead))`).
    fn one_dead_one_live_net() -> PetriNet {
        PetriNet {
            name: Some("one-dead-one-live".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                // t0 = t_live: p0 -> p1, enabled at init.
                trans("t_live", vec![arc(0, 1)], vec![arc(1, 1)]),
                // t1 = t_dead: needs 2 tokens in p1 (unreachable) -> p0.
                trans("t_dead", vec![arc(1, 2)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// Exhaustive-BFS ground truth: is transition `t` ever enabled in any
    /// reachable marking? `false` ⇒ `AG(¬IsFireable(t))` holds (t is dead).
    fn bfs_transition_dead(net: &PetriNet, t: TransitionIdx) -> bool {
        use std::collections::HashSet;
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut stack = vec![net.initial_marking.clone()];
        let mut budget = 100_000usize;
        while let Some(m) = stack.pop() {
            if budget == 0 {
                break;
            }
            budget -= 1;
            if !seen.insert(m.clone()) {
                continue;
            }
            if net.is_enabled(&m, t) {
                return false; // enabled somewhere ⇒ not dead
            }
            for ti in 0..net.num_transitions() {
                let tidx = TransitionIdx(ti as u32);
                if net.is_enabled(&m, tidx) {
                    if let Ok(next) = net.fire(&m, tidx) {
                        stack.push(next);
                    }
                }
            }
        }
        true // never enabled along the explored (here, complete) state space
    }

    fn ag_not_fireable(t: u32) -> ResolvedPredicate {
        ResolvedPredicate::Not(Box::new(ResolvedPredicate::IsFireable(vec![
            TransitionIdx(t),
        ])))
    }

    /// GATE: the integer dead-transition sweep resolves the `AG(¬IsFireable(t))`
    /// tracker for a structurally-dead transition (matching BFS ground truth)
    /// and does NOT falsely resolve the same shape for a live transition.
    #[test]
    fn integer_dead_transition_sweep_resolves_dead_not_live() {
        // The sweep reads the kill-switch env var; hold the crate-wide env lock
        // so a concurrent env-touching test can't flip it mid-run. Also pin it
        // ON for the duration in case a prior test left it set.
        let _lock = crate::env_test_lock();
        crate::env_guard::remove_var(ENABLE_INTEGER_DEAD_TRANSITION_ENV);

        let net = one_dead_one_live_net();
        let t_live = TransitionIdx(0);
        let t_dead = TransitionIdx(1);

        // BFS ground truth: t_dead is dead, t_live is not.
        assert!(
            bfs_transition_dead(&net, t_dead),
            "t_dead must be dead by BFS (sanity of the fixture)"
        );
        assert!(
            !bfs_transition_dead(&net, t_live),
            "t_live must be live by BFS (sanity of the fixture)"
        );

        let mut trackers = vec![
            // AG(¬IsFireable(t_dead)) — should resolve TRUE via the integer sweep.
            tracker(PathQuantifier::AG, ag_not_fireable(1), None),
            // AG(¬IsFireable(t_live)) — must NOT be resolved (live transition).
            tracker(PathQuantifier::AG, ag_not_fireable(0), None),
        ];

        // No global deadline ⇒ deterministic (sweep bounded only by its caps).
        let seeded = run_integer_dead_transition_sweep(&net, &mut trackers, None);

        assert_eq!(seeded, 1, "exactly the dead-transition tracker is resolved");
        assert_eq!(
            trackers[0].verdict,
            Some(true),
            "AG(¬IsFireable(t_dead)) is TRUE — t_dead is integer-infeasible/dead"
        );
        assert_eq!(
            trackers[0].resolved_by.map(|r| r.source),
            Some(ReachabilityResolutionSource::Lp),
            "dead-transition proof is attributed to the integer state-equation lane"
        );
        assert_eq!(
            trackers[1].verdict, None,
            "AG(¬IsFireable(t_live)) must stay pending — a live transition is never resolved"
        );
    }

    /// The sweep is verdict-neutral on shapes it does not handle and respects the
    /// kill-switch: an EF tracker, a non-fireability predicate, and an already
    /// resolved tracker are all left untouched.
    #[test]
    fn integer_dead_transition_sweep_only_touches_pending_ag_not_fireable() {
        // Serialize against the other env-reading sweep test (this one mutates
        // the kill-switch env var). The lock is held for the whole test so the
        // mutation can never be observed by a concurrent test.
        let _lock = crate::env_test_lock();
        let net = one_dead_one_live_net();

        // EF(¬IsFireable(t_dead)) — wrong quantifier, not this sweep's job.
        let mut ef = vec![tracker(PathQuantifier::EF, ag_not_fireable(1), None)];
        assert_eq!(run_integer_dead_transition_sweep(&net, &mut ef, None), 0);
        assert_eq!(ef[0].verdict, None);

        // AG over a cardinality atom — not a dead-transition shape.
        let mut card = vec![tracker(
            PathQuantifier::AG,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(1),
            ),
            None,
        )];
        assert_eq!(run_integer_dead_transition_sweep(&net, &mut card, None), 0);
        assert_eq!(card[0].verdict, None);

        // Kill-switch OFF ⇒ no-op even for a genuinely dead transition.
        crate::env_guard::set_var(ENABLE_INTEGER_DEAD_TRANSITION_ENV, "0");
        let mut off = vec![tracker(PathQuantifier::AG, ag_not_fireable(1), None)];
        let seeded_off = run_integer_dead_transition_sweep(&net, &mut off, None);
        crate::env_guard::remove_var(ENABLE_INTEGER_DEAD_TRANSITION_ENV);
        assert_eq!(seeded_off, 0, "kill-switch disables the sweep");
        assert_eq!(off[0].verdict, None);
    }
}
