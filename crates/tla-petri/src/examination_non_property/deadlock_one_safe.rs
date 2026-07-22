// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::super::super::examination_plan::ExecutionPlan;
use super::common::{checkpoint_cannot_compute, reduction_cannot_compute};
use crate::examinations::deadlock::{DeadlockObserver, PortfolioDeadlockObserver};
use crate::examinations::global_properties_bmc;
use crate::examinations::global_properties_pdr;
use crate::examinations::one_safe::{one_safe_por_config, OneSafeObserver};
use crate::explorer::{explore_observer, ExplorationConfig};
use crate::output::Verdict;
use crate::petri_net::{PetriNet, PlaceIdx};
use crate::portfolio::SharedVerdict;
use crate::reduction::{
    reduce_iterative_structural_one_safe, reduce_iterative_structural_with_mode, ReducedNet,
    ReductionMode,
};
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};
use crate::stubborn::PorStrategy;
use crate::symbolic::{symbolic_state_equation_check, SymbolicConfig, SymbolicVerdict};

const ONE_SAFE_SYMBOLIC_PHASE_CAP: Duration = Duration::from_secs(3);
const ONE_SAFE_BFS_FALLBACK_RESERVE: Duration = Duration::from_secs(10);

/// Wall-clock cap for the OneSafe structural short-circuit (P-invariants + LP)
/// under an MCC deadline.
///
/// `compute_p_invariants` (Gaussian elimination over the incidence matrix) and
/// `lp_upper_bound` (one LP solve per safety unit) do NOT poll the deadline, so
/// on high-arity nets they can spin the whole wall budget before any
/// deadline-aware engine or the explicit BFS runs — the BridgeAndVehicles
/// OneSafe overrun. Bounding the phase is strictly verdict-preserving: it only
/// ever PROVES 1-safety (`Some(Verdict::True)`); every other outcome (`None`,
/// cap-abandon, panic) is inconclusive and falls through to BMC/PDR/reduction/
/// BFS exactly as the `None` case already does. Mirrors
/// [`DEADLOCK_STRUCTURAL_PHASE_CAP`].
const ONE_SAFE_STRUCTURAL_PHASE_CAP: Duration = Duration::from_secs(8);

/// Wall-clock budget for the post-BFS-exhaustion symbolic state-equation
/// fallback (Tier 3 Item 2). Kept small because by this point the BFS
/// path has already burned most of the global budget; the symbolic call
/// is a last-shot supplement, not a replacement.
const ONE_SAFE_SYMBOLIC_FALLBACK_BUDGET: Duration = Duration::from_secs(5);

/// Triage/benchmark kill-switch for the OneSafe random-walk FALSE-witness lane.
///
/// `true` iff `TY_MCC_DISABLE_ONE_SAFE_WALK` is set to `1`/`on`/`true`. The walk
/// is a strict under-approximation that only ever emits `Verdict::False` from a
/// directly-observed reachable safety-unit overflow, so disabling it is always
/// verdict-preserving — it only removes a FALSE-witness shortcut, falling through
/// to the existing reduction/BFS pipeline. Mirrors `ltl_symbolic_disabled()` and
/// the other `TY_MCC_DISABLE_*` switches.
fn one_safe_walk_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_ONE_SAFE_WALK")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
}

/// Build the OneSafe safety predicate `AND(m_p <= 1)` over all places.
///
/// Shared by the early-phase PDR/BMC routes (encoded inline) and the
/// post-BFS symbolic state-equation fallback. Centralising avoids drift
/// between encodings that all answer the same MCC OneSafe question.
fn one_safe_safety_predicate(net: &PetriNet) -> ResolvedPredicate {
    let conjuncts: Vec<ResolvedPredicate> = (0..net.num_places())
        .map(|p| {
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(p as u32)]),
                ResolvedIntExpr::Constant(1),
            )
        })
        .collect();
    ResolvedPredicate::And(conjuncts)
}

/// Structural short-circuit for OneSafe: prove every safety unit bounded ≤ 1.
///
/// Returns `Some(Verdict::True)` iff P-invariants (or an LP fallback for units
/// not covered by an invariant) bound every safety unit's token sum ≤ 1, i.e.
/// the net is 1-safe without state-space exploration. Returns `None` when the
/// structural argument is inconclusive — the caller MUST fall through to the
/// BMC/PDR/reduction/BFS pipeline and never invent a verdict from `None`.
///
/// Extracted as a free function so it can run inside [`run_phase_with_wall_cap`]
/// (it is `Send + 'static`-compatible: it borrows only `net` and `safety_units`).
fn one_safe_structural_short_circuit(
    net: &PetriNet,
    safety_units: &[Vec<usize>],
) -> Option<Verdict> {
    let invariants = crate::invariant::compute_p_invariants(net);
    let unit_bound_by_invariant = |unit: &Vec<usize>| -> bool {
        crate::invariant::structural_set_bound(&invariants, unit).is_some_and(|bound| bound <= 1)
    };
    if safety_units.iter().all(unit_bound_by_invariant) {
        eprintln!(
            "OneSafe: structurally 1-safe (all {} units bounded ≤ 1 by P-invariants)",
            safety_units.len()
        );
        return Some(Verdict::True);
    }

    // LP fallback for units not bounded by P-invariants.
    let unbounded_units: Vec<&Vec<usize>> = safety_units
        .iter()
        .filter(|unit| !unit_bound_by_invariant(unit))
        .collect();
    if !unbounded_units.is_empty() {
        use crate::lp_state_equation::lp_upper_bound;
        let all_lp_bounded = unbounded_units.iter().all(|unit| {
            let place_indices: Vec<PlaceIdx> = unit.iter().map(|&p| PlaceIdx(p as u32)).collect();
            lp_upper_bound(net, &place_indices).is_some_and(|b| b <= 1)
        });
        if all_lp_bounded {
            eprintln!(
                "OneSafe: structurally 1-safe (LP bounds ≤ 1 for {} units not covered by P-invariants)",
                unbounded_units.len()
            );
            return Some(Verdict::True);
        }
    }

    None
}

/// Post-BFS symbolic state-equation fallback for OneSafe (Tier 3 Item 2).
///
/// Called when explicit BFS exhausts (incomplete by deadline / state cap)
/// on a PT-shaped OneSafe query. Encodes `AG(forall p. m_p <= 1)` as a
/// CHC system and dispatches to ay-chc's adaptive portfolio.
///
/// Soundness contract — STRICT:
/// - `Some(Verdict::True)`  iff symbolic proved 1-safety
/// - `Some(Verdict::False)` iff symbolic produced a validated counterexample
/// - `None`                 on every solver `Unknown` / encoder overflow:
///   the caller MUST surface `CannotCompute` rather than guess.
///
/// Never overrides a verdict the BFS path actually produced — callers must
/// only invoke this on the exhaustion branch.
fn try_one_safe_symbolic_state_equation(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> Option<Verdict> {
    if net.num_places() == 0 {
        return Some(Verdict::True);
    }

    let remaining = deadline
        .map(|limit| limit.saturating_duration_since(Instant::now()))
        .unwrap_or(ONE_SAFE_SYMBOLIC_FALLBACK_BUDGET);
    if remaining.is_zero() {
        return None;
    }
    let budget = ONE_SAFE_SYMBOLIC_FALLBACK_BUDGET.min(remaining);

    let property = one_safe_safety_predicate(net);
    let config = SymbolicConfig {
        time_budget: budget,
        ..SymbolicConfig::default()
    };

    match symbolic_state_equation_check(net, &property, &config) {
        SymbolicVerdict::Safe => {
            eprintln!("OneSafe: symbolic state-equation fallback proved 1-safety");
            Some(Verdict::True)
        }
        SymbolicVerdict::Unsafe { .. } => {
            eprintln!("OneSafe: symbolic state-equation fallback found violation");
            Some(Verdict::False)
        }
        SymbolicVerdict::Unknown { .. } => None,
    }
}

/// Wall-clock cap for the deadlock PDR/IC3 phase under finite deadlines.
///
/// Re-enable PDR under deadlines but cap it so a slow PDR call cannot starve
/// the BMC+BFS fallback.
const DEADLOCK_PDR_PHASE_CAP: Duration = Duration::from_secs(5);

/// Minimum remaining wall budget before the deadlock-region LP-relaxation
/// second phase is spent. Its wall cap ([`DEADLOCK_STRUCTURAL_PHASE_CAP`], 8s)
/// must be a small fraction of the remaining budget so it cannot starve the
/// downstream BMC/BFS engines. At the contest 3600s budget the LP phase is pure
/// gain on the large structurally-deadlock-free nets that integer branch-and-
/// bound times out on; below this it is skipped (no regression vs integer-only).
const DEADLOCK_REGION_LP_MIN_REMAINING: Duration = Duration::from_secs(120);

/// Wall-clock cap for the Phase-0 structural deadlock-free pre-checks
/// (`structural_deadlock_free` siphon/trap and `lp_deadlock_free`) under an MCC
/// deadline. These are "cheap" on typical nets but `lp_deadlock_free` issues an
/// LP solve per input arc with no internal deadline polling, so on very
/// high-arity nets (e.g. ASLink-PT/GPPP-PT with 1000+ transitions) it can spin
/// the entire wall budget before any deadline-aware engine or the explicit BFS
/// runs — turning a tractable model into a DNF/CANNOT_COMPUTE. Bounding the
/// phase is strictly verdict-preserving: an abandoned pre-check yields no
/// verdict and we fall through to PDR/AIGER/BMC+BFS exactly as the inconclusive
/// (`None`) case already does.
const DEADLOCK_STRUCTURAL_PHASE_CAP: Duration = Duration::from_secs(8);

/// Share of the remaining budget granted to the deadlock-preserving structural
/// pre-reduction at the top of [`deadlock_verdict`] (`remaining /
/// DEADLOCK_REDUCTION_DEADLINE_FRACTION`), mirroring the
/// [`DEADLOCK_PDR_DEADLINE_FRACTION`] pattern.
///
/// The reduction's first fixpoint round calls `compute_p_invariants` (Farkas
/// elimination) several times, and on giant nets a single round runs ~140 s
/// without ever polling its deadline (DLCflexbar-PT-2b, 4456 places, measured) —
/// consuming the entire 150 s budget and starving every deciding lane, while the
/// cell's σ-witness region solve alone would answer in ~8.5 s. A proportional
/// share is the adaptive fix: at 150 s the reduction gets ~18 s (the deciding
/// engines keep ≥7/8 of the budget); at the contest's 3600 s it gets 450 s, which
/// covers the measured ~140 s convergence — so full-budget behaviour is
/// unchanged. Abandoning the reduction is verdict-preserving: the caller falls
/// back to the identity (unreduced) net, exactly the pre-existing panic/`Err`
/// fallback path.
const DEADLOCK_REDUCTION_DEADLINE_FRACTION: u32 = 8;

/// Share of the remaining budget granted to the σ-guided reachable-deadlock
/// WITNESS phase (Phase 0a-TRUE): `remaining / DEADLOCK_WITNESS_DEADLINE_FRACTION`,
/// kept SEPARATE from — and larger than — the shared 8 s
/// [`DEADLOCK_STRUCTURAL_PHASE_CAP`].
///
/// The witness first solves the integer deadlock region (one SMT check) and only
/// then realizes σ. On a giant TRUE net that region solve alone is ~8.5 s on an
/// idle host (DLCflexbar-PT-2b) and stretches further under memory pressure —
/// measured to slide past a flat 15 s cap exactly when the host is loaded. A
/// PROPORTIONAL share is the adaptive bound: ~29 s at a 150 s budget, ~15 min at
/// the contest's 3600 s, always leaving ≥3/4 of the remainder for DD/PDR/BMC/BFS.
/// The phase exits far earlier than its share in the common cases — UNSAT
/// (deadlock-free) returns at the solve, and a non-realizable σ trips the
/// realize loop's no-progress patience — so the share only gates the genuinely
/// slow SAT solves it exists to accommodate. It does NOT enlarge the FALSE
/// lanes' caps (on a TRUE net those cannot conclude and would only burn more
/// budget). Soundness-neutral: the witness only ever emits `True`, and only at a
/// marking reached by firing exclusively `is_enabled`-checked transitions.
const DEADLOCK_WITNESS_DEADLINE_FRACTION: u32 = 4;

/// Cooperative soft cap for the complete-minimal-siphon LP deadlock-freedom
/// check ([`crate::structural::lp_siphon_deadlock_free`]).
///
/// The minimal-siphon enumeration is worst-case exponential (Anderson-style
/// mutex arrays explode), so the check is given a small internal deadline it
/// polls between recursion nodes and LP solves. On the deadlock-free families it
/// targets (TokenRing / SharedMemory) it completes well under this cap; on a
/// siphon explosion it declines (`None`) here instead of consuming the full
/// [`DEADLOCK_STRUCTURAL_PHASE_CAP`] wall budget, keeping the regression on
/// nets it cannot decide to a couple of seconds. Declining is verdict-
/// preserving (the exact engine still decides), so the bound is sound.
const DEADLOCK_SIPHON_SOFT_CAP: Duration = Duration::from_secs(2);

/// Fraction of the remaining global deadline allotted to PDR under MCC.
///
/// `1/3` leaves at least `2/3` of the deadline for BMC+BFS even when PDR is
/// inconclusive. Combined with [`DEADLOCK_PDR_PHASE_CAP`] this caps PDR at
/// `min(5s, remaining/3)`.
const DEADLOCK_PDR_DEADLINE_FRACTION: u32 = 3;

/// Wall-clock cap for the deadlock AIGER+IC3 seeding phase.
///
/// Surfaced by the 2026-05-23 MCC measurement: on AutoFlight-PT-02a
/// (StateSpace = 700 ms with PorStrategy::None), the deadlock pipeline ran
/// for >14 minutes of CPU on a `--timeout 15` invocation. The hang was inside
/// the AIGER preprocessing (`Transys::preprocess_configured`) which has
/// individual phases that do not poll the deadline between SAT calls, so a
/// deadline-only budget did not protect the BFS fallback. Cap the seeding
/// phase at a small wall-clock slice so the BMC+BFS portfolio is always
/// reached with the bulk of the budget intact. Mirrors the OneSafe phase cap
/// (see [`ONE_SAFE_SYMBOLIC_PHASE_CAP`]).
const DEADLOCK_AIGER_PHASE_CAP: Duration = Duration::from_secs(3);

/// Minimum wall-clock budget reserved for the BMC+BFS portfolio.
///
/// If the remaining deadline at the start of `deadlock_verdict` is less than
/// or equal to this reserve, the AIGER seeding phase is skipped entirely:
/// every spare millisecond goes to the explicit-state portfolio. Mirrors
/// [`ONE_SAFE_BFS_FALLBACK_RESERVE`].
const DEADLOCK_BFS_FALLBACK_RESERVE: Duration = Duration::from_secs(10);

/// Wall-clock cap for the random-walk deadlock-witness lane.
///
/// The lane is a TRUE-only under-approximation that runs AFTER PDR/AIGER and
/// BEFORE the explicit BMC+BFS portfolio. On flat, no-symmetry nets where BFS
/// cannot reach the stuck marking within budget (PolyORBLF-PT-S04J06T06,
/// ResAllocation-PT-R003C050), a short random walk frequently stumbles onto the
/// reachable deadlock that consensus tools find. Budget is ADDITIVE /
/// leftover-only: it takes `min(remaining/4, this cap)` of the budget remaining
/// after AIGER and still reserves the full [`DEADLOCK_BFS_FALLBACK_RESERVE`]
/// tail for the exhaustive portfolio, so it can NEVER starve BFS.
const DEADLOCK_WALK_PHASE_CAP: Duration = Duration::from_secs(8);

/// Fraction of the post-AIGER remaining budget the random-walk lane may use.
const DEADLOCK_WALK_DEADLINE_FRACTION: u32 = 4;

/// Compute BFS worker count for the portfolio, if applicable.
///
/// Returns `None` when `workers == 1` (stay sequential — the public
/// `ExplorationConfig::workers()` contract says 1 = sequential).
/// Returns `Some(bfs_workers)` when `workers >= 2`, reserving 1 for BMC.
fn deadlock_portfolio_bfs_workers(total_workers: usize) -> Option<usize> {
    (total_workers >= 2).then_some(total_workers.saturating_sub(1))
}

fn one_safe_symbolic_deadline_at(
    global_deadline: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    let global_deadline = global_deadline?;
    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= ONE_SAFE_BFS_FALLBACK_RESERVE {
        return Some(now);
    }

    let phase_budget =
        ONE_SAFE_SYMBOLIC_PHASE_CAP.min(remaining.saturating_sub(ONE_SAFE_BFS_FALLBACK_RESERVE));
    Some(now + phase_budget)
}

fn one_safe_symbolic_deadline(global_deadline: Option<Instant>) -> Option<Instant> {
    one_safe_symbolic_deadline_at(global_deadline, Instant::now())
}

/// Compute the AIGER seeding phase deadline for the deadlock pipeline.
///
/// Returns:
/// - `None` when there is no global deadline — AIGER may run to its own
///   internal timeout ([`crate::examinations::reachability_aiger`] caps each
///   property at 10s) without external interference.
/// - `Some(now)` (already expired) when the remaining global budget is at or
///   below [`DEADLOCK_BFS_FALLBACK_RESERVE`]. The caller MUST skip the AIGER
///   seeding phase and give the entire remaining budget to the explicit
///   BMC+BFS portfolio.
/// - `Some(phase_deadline)` capped at [`DEADLOCK_AIGER_PHASE_CAP`] otherwise.
fn deadlock_aiger_deadline_at(global_deadline: Option<Instant>, now: Instant) -> Option<Instant> {
    let global_deadline = global_deadline?;
    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= DEADLOCK_BFS_FALLBACK_RESERVE {
        return Some(now);
    }

    let phase_budget =
        DEADLOCK_AIGER_PHASE_CAP.min(remaining.saturating_sub(DEADLOCK_BFS_FALLBACK_RESERVE));
    Some(now + phase_budget)
}

fn deadlock_aiger_deadline(global_deadline: Option<Instant>) -> Option<Instant> {
    deadlock_aiger_deadline_at(global_deadline, Instant::now())
}

/// `true` iff the AIGER phase deadline says "skip" (already expired). The
/// `Some(now)` sentinel from [`deadlock_aiger_deadline_at`] is the contract
/// the caller must honour by NOT entering the AIGER pipeline.
fn deadlock_aiger_skip(phase_deadline: Option<Instant>) -> bool {
    phase_deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

/// Compute the random-walk deadlock-witness phase deadline.
///
/// Additive / leftover-only, mirroring [`deadlock_aiger_deadline_at`]:
/// - `None` when there is no global deadline (the walk runs to its own
///   internal walk/step budget without external interference).
/// - `Some(now)` (already expired, i.e. "skip") when the remaining budget is at
///   or below [`DEADLOCK_BFS_FALLBACK_RESERVE`] — the full remaining budget is
///   reserved for the explicit BMC+BFS portfolio.
/// - `Some(phase_deadline)` capped at `min(DEADLOCK_WALK_PHASE_CAP,
///   (remaining - DEADLOCK_BFS_FALLBACK_RESERVE) / DEADLOCK_WALK_DEADLINE_FRACTION)`
///   otherwise. The subtraction of the BFS reserve BEFORE taking the fraction
///   guarantees the walk can never eat into the BFS tail.
fn deadlock_walk_deadline_at(global_deadline: Option<Instant>, now: Instant) -> Option<Instant> {
    let global_deadline = global_deadline?;
    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= DEADLOCK_BFS_FALLBACK_RESERVE {
        return Some(now);
    }

    let leftover = remaining
        .checked_sub(DEADLOCK_BFS_FALLBACK_RESERVE)
        .unwrap();
    let phase_budget = DEADLOCK_WALK_PHASE_CAP.min(leftover / DEADLOCK_WALK_DEADLINE_FRACTION);
    Some(now + phase_budget)
}

fn deadlock_walk_deadline(global_deadline: Option<Instant>) -> Option<Instant> {
    deadlock_walk_deadline_at(global_deadline, Instant::now())
}

/// Structural deadlock pre-check phase deadline (B4 scheduling hygiene). Mirrors
/// [`deadlock_aiger_deadline_at`]: cap at [`DEADLOCK_STRUCTURAL_PHASE_CAP`] but
/// never eat into the [`DEADLOCK_BFS_FALLBACK_RESERVE`] reserved for the explicit
/// BMC+BFS portfolio. At the contest budget this is a no-op (`remaining` ≫ cap +
/// reserve, so the cap stands); near the deadline it stops a structural lane from
/// spinning its full flat cap and starving — or overrunning past — the BFS tail.
/// Soundness-neutral: the structural checks only ever PROVE deadlock-freedom and
/// otherwise fall through, so clamping changes WHEN they yield, never the verdict.
fn deadlock_structural_deadline_at(
    global_deadline: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    let global_deadline = global_deadline?;
    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= DEADLOCK_BFS_FALLBACK_RESERVE {
        return Some(now);
    }
    let phase_budget =
        DEADLOCK_STRUCTURAL_PHASE_CAP.min(remaining.saturating_sub(DEADLOCK_BFS_FALLBACK_RESERVE));
    Some(now + phase_budget)
}

fn deadlock_structural_deadline(global_deadline: Option<Instant>) -> Option<Instant> {
    deadlock_structural_deadline_at(global_deadline, Instant::now())
}

/// Deadline for the σ-witness phase: its proportional share
/// (`remaining / DEADLOCK_WITNESS_DEADLINE_FRACTION`), still never eating into
/// the BFS reserve. Same clamp shape as [`deadlock_structural_deadline_at`],
/// with the witness's adaptive share so its region solve (~8.5 s+ on giant
/// nets, load-dependent) completes instead of timing out under the flat 8 s
/// structural cap.
fn deadlock_witness_deadline(global_deadline: Option<Instant>) -> Option<Instant> {
    let global_deadline = global_deadline?;
    let now = Instant::now();
    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= DEADLOCK_BFS_FALLBACK_RESERVE {
        return Some(now);
    }
    let phase_budget = (remaining / DEADLOCK_WITNESS_DEADLINE_FRACTION)
        .min(remaining.saturating_sub(DEADLOCK_BFS_FALLBACK_RESERVE));
    Some(now + phase_budget)
}

/// Wall-clock-bounded structural pre-reduction (the [`run_phase_with_wall_cap`]
/// pattern, specialized to return the reduced net instead of a verdict).
///
/// The reduction fixpoint polls its deadline only BETWEEN rounds, and one round's
/// `compute_p_invariants` calls (Farkas elimination) can run ~140 s on giant nets
/// (DLCflexbar-PT-2b, measured) — the exact non-polling pathology
/// [`run_phase_with_wall_cap`] documents for AIGER/PDR. Running the reduction on
/// a worker thread and abandoning it at the cap is the only soundness-preserving
/// bound available without threading a deadline through the whole reduction
/// crate; the abandoned thread is not joined (no cooperative cancellation
/// primitive), matching the documented house policy. `None` = the reduction did
/// not finish in time (or panicked): the caller falls back to the identity net,
/// which is the pre-existing panic/`Err` fallback and therefore
/// verdict-preserving by construction.
fn run_reduction_with_wall_cap(net: &PetriNet, cap: Duration) -> Option<ReducedNet> {
    if cap.is_zero() {
        return None;
    }
    let net_for_worker = net.clone();
    // The worker also receives the cap as its cooperative deadline: nets whose
    // rounds DO reach the between-rounds poll return their accumulated partial
    // reduction in time; the thread abandon only bites when a single round
    // overruns without polling (the giant-net Farkas case).
    let worker_deadline = Some(Instant::now() + cap);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Option<ReducedNet>>(1);
    let _worker = std::thread::Builder::new()
        .name("ty-deadlock-prereduce".to_string())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reduce_iterative_structural_with_mode(
                    &net_for_worker,
                    &[],
                    ReductionMode::ReachabilityDeadlock,
                    worker_deadline,
                )
            }))
            .ok()
            .and_then(|r| r.ok());
            let _ = tx.send(result);
        });
    match rx.recv_timeout(cap) {
        Ok(reduced) => reduced,
        Err(_) => None, // timeout (abandon) or worker died — identity fallback
    }
}

/// `true` iff the walk phase deadline says "skip" (already expired).
fn deadlock_walk_skip(phase_deadline: Option<Instant>) -> bool {
    phase_deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

/// Compute the PDR phase deadline for the deadlock pipeline under MCC.
///
/// Returns:
/// - `None` when there is no global deadline — PDR may run to its own
///   internal timeout without external interference.
/// - `Some(phase_deadline)` capped at `min(DEADLOCK_PDR_PHASE_CAP,
///   remaining / DEADLOCK_PDR_DEADLINE_FRACTION)` otherwise.
fn deadlock_pdr_deadline_at(global_deadline: Option<Instant>, now: Instant) -> Option<Instant> {
    let global_deadline = global_deadline?;
    let remaining = global_deadline.saturating_duration_since(now);
    let share = remaining / DEADLOCK_PDR_DEADLINE_FRACTION;
    let phase_budget = DEADLOCK_PDR_PHASE_CAP.min(share);
    Some(now + phase_budget)
}

fn deadlock_pdr_deadline(global_deadline: Option<Instant>) -> Option<Instant> {
    deadlock_pdr_deadline_at(global_deadline, Instant::now())
}

/// Returns `Some(Verdict::True)` ONLY when AIGER produces a SAT trace that
/// replays to a real Petri marking in which NO transition is enabled.
///
/// On UNSAT, Unknown, encoding rejection, or replay failure: returns `None`
/// so the BMC+BFS portfolio gets the unspent budget.
///
/// Soundness: we never return `Verdict::False` from UNSAT. The safety
/// property for deadlock is `IsFireable(all_transitions)` and the AIGER
/// encoder over-approximates fireability semantics, so UNSAT on the circuit
/// does not soundly imply UNSAT on the Petri net. See the analogous refusal
/// in `reachability_aiger::resolve_aiger_unsat` and its
/// `predicate_contains_fireability` check.
///
/// Technique attribution: on AIGER success we set a thread-local flag via
/// [`crate::examination::note_aiger_resolved_deadlock`] so the
/// MCC technique line for this examination renders `SAT_SMT` (IC3/PDR via
/// AIGER), not the default `EXPLICIT`.
fn try_aiger_deadlock(net: &PetriNet, deadline: Option<Instant>) -> Option<Verdict> {
    match crate::examinations::reachability_aiger::run_aiger_deadlock_check(net, deadline) {
        Some(_witness) => {
            crate::examination::note_aiger_resolved_deadlock();
            Some(Verdict::True)
        }
        None => None,
    }
}

/// Outcome of the wall-clock-bounded AIGER deadlock phase.
enum PhaseOutcome {
    /// AIGER produced a validated `Verdict::True` deadlock witness.
    Verdict(Verdict),
    /// AIGER ran to completion within the phase cap but produced no verdict.
    NoVerdict,
    /// The phase cap expired before AIGER returned; the worker thread is
    /// abandoned (leaked) so the BMC+BFS portfolio gets the remaining
    /// global budget. Soundness: no verdict is published from this branch.
    Abandoned,
    /// The AIGER worker thread panicked (heap-heavy preprocessing has known
    /// overflow edge cases). Soundness: no verdict is published.
    Panicked,
}

/// Wall-clock-bounded wrapper for a phase that might not poll its deadline.
///
/// AIGER preprocessing and PDR's `compute_p_invariants` both exhibit the
/// same pathology: they accept a `deadline: Option<Instant>` argument but
/// do not poll it between expensive inner phases (verified on
/// Philosophers-COL-000020 for AIGER and ASLink-PT-04a for PDR — both
/// spin the full wall budget despite soft caps). The only soundness-
/// preserving way to bound their wall-clock impact is to run the phase on
/// a separate thread and abandon the thread once the cap expires.
///
/// The worker thread is deliberately not joined when we abandon it: the
/// underlying pipelines have no cooperative cancellation primitive.
/// Joining would re-introduce the original wall-clock hang. The OS
/// reclaims the thread when the process exits at the global deadline,
/// which is the next step on the MCC pipeline anyway.
///
/// Returns one of the four [`PhaseOutcome`] cases. The caller MUST
/// treat `Abandoned` and `Panicked` as "no verdict" and continue to the
/// BMC+BFS portfolio — never invent a verdict from these branches.
fn run_phase_with_wall_cap<F>(net: &PetriNet, deadline: Option<Instant>, phase: F) -> PhaseOutcome
where
    F: FnOnce(&PetriNet, Option<Instant>) -> Option<Verdict> + Send + 'static,
{
    let cap = deadline
        .map(|d| d.saturating_duration_since(Instant::now()))
        .unwrap_or(DEADLOCK_AIGER_PHASE_CAP);
    if cap.is_zero() {
        return PhaseOutcome::NoVerdict;
    }

    let net_for_worker = net.clone();
    let (tx, rx) = std::sync::mpsc::sync_channel::<std::thread::Result<Option<Verdict>>>(1);
    let _worker = std::thread::Builder::new()
        .name("ty-deadlock-phase".to_string())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                phase(&net_for_worker, deadline)
            }));
            // Ignore SendError: parent already gave up and the receiver is
            // dropped; nothing useful to do with the result.
            let _ = tx.send(result);
        });

    match rx.recv_timeout(cap) {
        Ok(Ok(Some(verdict))) => PhaseOutcome::Verdict(verdict),
        Ok(Ok(None)) => PhaseOutcome::NoVerdict,
        Ok(Err(_)) => PhaseOutcome::Panicked,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => PhaseOutcome::Abandoned,
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => PhaseOutcome::NoVerdict,
    }
}

pub(crate) fn deadlock_verdict(net: &PetriNet, config: &ExplorationConfig) -> Verdict {
    // --- P1: deadlock-preserving structural pre-reduction (GlobalProperties) ---
    //
    // Deadlock-freedom is a BOOLEAN over the net: "∃ a reachable marking with
    // no enabled transition?". Under the `ReachabilityDeadlock` rule subset
    // (dead/constant/isolated + redundant-place + duplicate/dominated-transition
    // + parallel-place-merge + never-disabling-arc), every admitted rule leaves
    // the set of transitions enabled at every reachable marking unchanged, so
    // the reduced net's reachable-deadlock-existence is IDENTICAL to the
    // original's. The verdict therefore lifts with NO marking expansion: a
    // yes/no answer, never a witness coordinate, so re-indexing places /
    // transitions in the reduced net is irrelevant to the reported Verdict.
    //
    // Every downstream lane (structural trap/LP pre-checks, DD, PDR, AIGER,
    // BMC, BFS) then runs on the materially smaller `reduced.net`. The rules
    // that can create or hide a deadlock — agglomeration / Rule R / Rule S /
    // token-cycle (fuse producer-then-consumer and delete the transient
    // intermediate marking), self-loop-transition and sink-transition removal
    // (may delete the only enabled transition in some marking), self-loop-arc
    // removal (strips an input requirement, erasing a deadlock) — are all
    // gated FALSE for this mode (see `ReductionMode::ReachabilityDeadlock`).
    //
    // `catch_unwind` + `unwrap_or_else(identity)` keep this strictly
    // verdict-preserving: redundant-place removal calls `compute_p_invariants`,
    // which has a latent i64 overflow on very high-arity nets (GPPP-PT-C1000N*);
    // a panic or an `Err` falls back to the identity (unreduced) net, i.e. the
    // exact prior behaviour, so the reduction can only ever shrink the net,
    // never change the answer.
    // Under an MCC deadline the reduction runs on a wall-capped worker with a
    // PROPORTIONAL share of the remaining budget (see
    // [`DEADLOCK_REDUCTION_DEADLINE_FRACTION`]) — its fixpoint does not reliably
    // poll a deadline inside a round, and on giant nets one round's Farkas
    // eliminations consume the entire budget, starving every deciding lane.
    // Abandon ⇒ identity net, the pre-existing fallback, so this is strictly
    // verdict-preserving. Without a deadline the original inline run-to-fixpoint
    // behaviour is unchanged.
    let reduced = match config.deadline() {
        Some(dl) => {
            let remaining = dl.saturating_duration_since(Instant::now());
            let share = (remaining / DEADLOCK_REDUCTION_DEADLINE_FRACTION)
                .min(remaining.saturating_sub(DEADLOCK_BFS_FALLBACK_RESERVE));
            run_reduction_with_wall_cap(net, share)
        }
        None => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            reduce_iterative_structural_with_mode(
                net,
                &[],
                ReductionMode::ReachabilityDeadlock,
                None,
            )
        }))
        .ok()
        .and_then(|result| result.ok()),
    }
    .unwrap_or_else(|| ReducedNet::identity(net));
    let net = &reduced.net;
    let refitted_config = config.refitted_for_net(net);
    let config = &refitted_config;

    // --- Phase 0: cheap structural pre-checks (sequential, <1s) ---
    //
    // catch_unwind guards: the structural pre-checks call into
    // `compute_p_invariants` (LP) which has a latent i64 multiplication
    // overflow on very high-arity nets (GPPP-PT-C1000N* with 5000+ arcs
    // hits this). Without the guard, the entire ty-mcc process panics
    // before any other phase or examination on the same model gets to
    // run. Soundness contract: a panicking pre-check produces no verdict
    // and we fall through to PDR/AIGER/BMC+BFS.
    //
    // Under an MCC deadline, bound the structural phase with a wall cap:
    // `lp_deadlock_free` issues an LP solve per input arc without polling the
    // deadline, so on very high-arity nets it can spin the whole wall budget
    // before any deadline-aware engine or the explicit BFS runs. Both checks
    // only ever prove deadlock-FREEDOM (`Some(true)` -> `Verdict::False`); any
    // other outcome (`Some(false)`, `None`, panic, cap-abandon) is treated as
    // inconclusive and falls through, so capping is strictly verdict-preserving.
    // With no deadline (non-MCC / public API contract) we keep the original
    // unbounded inline path.
    if config.deadline().is_some() {
        match run_phase_with_wall_cap(
            net,
            deadlock_structural_deadline(config.deadline()),
            |net, _dl| match crate::structural::structural_deadlock_free(net) {
                Some(true) => Some(Verdict::False),
                _ => None,
            },
        ) {
            PhaseOutcome::Verdict(verdict) => {
                eprintln!("ReachabilityDeadlock: structurally deadlock-free (siphon/trap)");
                return verdict;
            }
            PhaseOutcome::Abandoned => eprintln!(
                "ReachabilityDeadlock: structural pre-check exceeded {:?} wall cap; abandoning",
                DEADLOCK_STRUCTURAL_PHASE_CAP,
            ),
            PhaseOutcome::Panicked => eprintln!(
                "ReachabilityDeadlock: structural pre-check panicked; falling through to PDR/AIGER/BMC+BFS"
            ),
            PhaseOutcome::NoVerdict => {}
        }

        match run_phase_with_wall_cap(
            net,
            deadlock_structural_deadline(config.deadline()),
            |net, _dl| match crate::structural::lp_deadlock_free(net) {
                Some(true) => Some(Verdict::False),
                _ => None,
            },
        ) {
            PhaseOutcome::Verdict(verdict) => {
                eprintln!(
                    "ReachabilityDeadlock: LP-proved deadlock-free (always-enabled transition)"
                );
                return verdict;
            }
            PhaseOutcome::Abandoned => eprintln!(
                "ReachabilityDeadlock: LP pre-check exceeded {:?} wall cap; abandoning",
                DEADLOCK_STRUCTURAL_PHASE_CAP,
            ),
            PhaseOutcome::Panicked => eprintln!(
                "ReachabilityDeadlock: LP pre-check panicked; falling through to PDR/AIGER/BMC+BFS"
            ),
            PhaseOutcome::NoVerdict => {}
        }

        // Complete-minimal-siphon LP deadlock-freedom: sound on ALL ordinary
        // nets (not just free-choice). Proves deadlock-FREEDOM when every minimal
        // siphon is LP-provably non-emptiable, which decides deadlock-free
        // families (TokenRing / SharedMemory / HexagonalGrid) structurally in a
        // few seconds instead of burning the PDR+AIGER wall caps and a full BFS.
        // Only ever emits `Some(true)` -> `Verdict::False`; an emptiable siphon,
        // an incomplete/exploded enumeration, or the soft-cap timeout returns
        // `None` and falls through (e.g. Philosophers / CSRepetitions, which
        // genuinely deadlock, are NOT certified free; Anderson, whose minimal
        // siphons explode combinatorially, declines at the soft cap). Wall-capped
        // like the checks above.
        match run_phase_with_wall_cap(
            net,
            deadlock_structural_deadline(config.deadline()),
            |net, _dl| {
                let soft_deadline = Instant::now() + DEADLOCK_SIPHON_SOFT_CAP;
                match crate::structural::lp_siphon_deadlock_free(net, Some(soft_deadline)) {
                    Some(true) => Some(Verdict::False),
                    _ => None,
                }
            },
        ) {
            PhaseOutcome::Verdict(verdict) => {
                eprintln!(
                    "ReachabilityDeadlock: LP-proved deadlock-free (all minimal siphons non-emptiable)"
                );
                return verdict;
            }
            PhaseOutcome::Abandoned => eprintln!(
                "ReachabilityDeadlock: siphon-LP pre-check exceeded {:?} wall cap; abandoning",
                DEADLOCK_STRUCTURAL_PHASE_CAP,
            ),
            PhaseOutcome::Panicked => eprintln!(
                "ReachabilityDeadlock: siphon-LP pre-check panicked; falling through to PDR/AIGER/BMC+BFS"
            ),
            PhaseOutcome::NoVerdict => {}
        }
    } else {
        let structural_safe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::structural::structural_deadlock_free(net)
        }));
        if let Ok(Some(true)) = structural_safe {
            eprintln!("ReachabilityDeadlock: structurally deadlock-free (siphon/trap)");
            return Verdict::False;
        }

        // LP state equation: proves deadlock-freedom on ALL net types by checking
        // if some transition is always enabled in the LP relaxation.
        let lp_safe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::structural::lp_deadlock_free(net)
        }));
        if let Ok(Some(true)) = lp_safe {
            eprintln!("ReachabilityDeadlock: LP-proved deadlock-free (always-enabled transition)");
            return Verdict::False;
        }

        // Complete-minimal-siphon LP deadlock-freedom (sound on all ordinary
        // nets; see the deadline branch above). `Some(true)` -> deadlock-free.
        // Even without a global deadline, bound the worst-case-exponential
        // enumeration with the soft cap so the public-API path cannot hang on a
        // siphon explosion.
        let siphon_safe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::structural::lp_siphon_deadlock_free(
                net,
                Some(Instant::now() + DEADLOCK_SIPHON_SOFT_CAP),
            )
        }));
        if let Ok(Some(true)) = siphon_safe {
            eprintln!(
                "ReachabilityDeadlock: LP-proved deadlock-free (all minimal siphons non-emptiable)"
            );
            return Verdict::False;
        }
        if structural_safe.is_err() || lp_safe.is_err() || siphon_safe.is_err() {
            eprintln!(
                "ReachabilityDeadlock: structural pre-check panicked; \
                 falling through to PDR/AIGER/BMC+BFS"
            );
        }
    }

    // --- Phase 0a: marking-equation deadlock-region infeasibility ---
    //
    // The COMPLETE structural deadlock-freedom check that the LP/siphon
    // pre-checks above only approximate (`lp_deadlock_free` needs a *single*
    // always-enabled transition; global-structure freedom — Raft / consensus /
    // token-conservation nets — has none). We ask ay whether the DEADLOCK REGION
    // is inhabited over the marking (state) equation:
    //
    //   ∃ M,σ: M = M₀ + C·σ ∧ M,σ ≥ 0 ∧ ⋀_t ⋁_{p∈•t} M[p] ≤ W(p,t)−1
    //
    // The marking-equation solution set is a SUPERSET of the reachable set, so
    // UNSAT ⇒ no reachable deadlock ⇒ `Verdict::False` (a rigorous proof). SAT is
    // possibly spurious ⇒ inconclusive; solver Unknown/timeout ⇒ inconclusive.
    // The lane emits ONLY `Verdict::False` and ONLY on proved-UNSAT, so it can
    // never contribute a wrong verdict. Wall-capped and panic-guarded like the
    // sibling structural phases. See `crate::deadlock_region`.
    // Marking-equation deadlock-region infeasibility. The tight INTEGER query
    // (QF_LIA) first — decides the small nets exactly. Only if it *times out*
    // (a large net, where the downstream BMC/BFS would time out too, so no
    // downstream win is at stake) do we spend a second phase on the faster LP
    // RELAXATION (QF_LRA), which proves the large structurally-deadlock-free
    // nets that integer branch-and-bound cannot finish. This targeting keeps the
    // LP phase from stealing budget from the downstream engines on the nets they
    // can still win. Both only ever emit `Verdict::False` on a proved-UNSAT
    // region (sound; see `crate::deadlock_region`).
    if config.deadline().is_some() {
        let integer = run_phase_with_wall_cap(
            net,
            deadlock_structural_deadline(config.deadline()),
            |net, dl| match crate::deadlock_region::deadlock_region_infeasible(net, dl, false) {
                Some(true) => Some(Verdict::False),
                _ => None,
            },
        );
        match integer {
            PhaseOutcome::Verdict(verdict) => {
                eprintln!(
                    "ReachabilityDeadlock: marking-equation deadlock region infeasible (deadlock-free, integer)"
                );
                return verdict;
            }
            PhaseOutcome::Abandoned
                if config.deadline().is_some_and(|dl| {
                    // Only spend a second region phase when the budget is large
                    // enough that its wall cap cannot starve the downstream
                    // BMC/BFS engines (which can still win medium nets). At the
                    // contest's 3600s budget this is pure gain on the large
                    // structurally-free nets; at tight local budgets it is
                    // skipped, so there is no regression vs the integer-only lane.
                    dl.saturating_duration_since(Instant::now()) >= DEADLOCK_REGION_LP_MIN_REMAINING
                }) =>
            {
                // Large net: integer timed out and budget is ample. Try the LP relaxation.
                match run_phase_with_wall_cap(
                    net,
                    deadlock_structural_deadline(config.deadline()),
                    |net, dl| match crate::deadlock_region::deadlock_region_infeasible(net, dl, true) {
                        Some(true) => Some(Verdict::False),
                        _ => None,
                    },
                ) {
                    PhaseOutcome::Verdict(verdict) => {
                        eprintln!(
                            "ReachabilityDeadlock: marking-equation deadlock region infeasible (deadlock-free, LP relaxation)"
                        );
                        return verdict;
                    }
                    _ => eprintln!(
                        "ReachabilityDeadlock: deadlock-region LP relaxation inconclusive; falling through to PDR/AIGER/BMC+BFS"
                    ),
                }
            }
            PhaseOutcome::Abandoned | PhaseOutcome::Panicked | PhaseOutcome::NoVerdict => {}
        }
    } else {
        for real in [false, true] {
            if let Some(true) = crate::deadlock_region::deadlock_region_infeasible(net, None, real)
            {
                eprintln!(
                    "ReachabilityDeadlock: marking-equation deadlock region infeasible (deadlock-free)"
                );
                return Verdict::False;
            }
        }
    }

    // --- Phase 0a-TRUE: σ-guided reachable-deadlock witness ---
    //
    // The TRUE counterpart of the deadlock-region FALSE lanes above: solve the
    // integer region for a candidate deadlock firing-count vector σ, then greedily
    // realize it from M₀ (firing only ENABLED transitions). Reaching a marking
    // with no enabled transition is a concretely-reachable deadlock ⇒
    // `Verdict::True`. Sound — a deadlock-free net can never reach a dead marking,
    // so it never false-TRUEs; wall-capped and fail-closed (only ever emits True).
    // Finds σ-*directed* deadlocks that an unguided walk / bounded BMC miss.
    if config.deadline().is_some() {
        if let PhaseOutcome::Verdict(verdict) = run_phase_with_wall_cap(
            net,
            deadlock_witness_deadline(config.deadline()),
            |net, dl| match crate::deadlock_region::deadlock_witness(net, dl) {
                Some(true) => Some(Verdict::True),
                _ => None,
            },
        ) {
            eprintln!(
                "ReachabilityDeadlock: σ-guided reachable deadlock witnessed (deadlock exists)"
            );
            return verdict;
        }
    } else if let Some(true) = crate::deadlock_region::deadlock_witness(net, None) {
        eprintln!("ReachabilityDeadlock: σ-guided reachable deadlock witnessed (deadlock exists)");
        return Verdict::True;
    }

    // --- Phase 0a': Decision-Diagram exact fast-path ---
    //
    // Off by default (gated by `dd-backend`). Placed AFTER the cheap
    // structural deadlock-freedom pre-checks and BEFORE PDR/AIGER/BMC/BFS.
    // On a small bounded net the DD backend builds the EXACT reachable-
    // marking set and decides deadlock-existence directly:
    //
    //   deadlock exists  ⟺  EF(AND_t NOT IsFireable([t]))
    //
    // i.e. some reachable marking enables no transition ⇒ Verdict::True;
    // every reachable marking enables at least one ⇒ Verdict::False. The
    // predicate is expressed exactly with the DD And/Not/IsFireable
    // combinators over the exact reachable set, so the verdict equals a
    // completed exact check.
    //
    // Soundness: build_sound_dd_spec guarantees the BDD reachable set is a
    // superset of every place's reachable projection (exact). On ANY DD
    // failure (decline/timeout/panic) no verdict is emitted and we fall
    // through to PDR/AIGER/BMC/BFS unchanged.
    #[cfg(feature = "dd-backend")]
    if let Some(verdict) = super::dd_fastpath::try_dd_deadlock(net, config.deadline()) {
        eprintln!("ReachabilityDeadlock: resolved exactly by DD reachable-set fast-path");
        return verdict;
    }

    // --- Phase 0b: PDR/IC3 (symbolic, no state space) ---
    //
    // Under finite MCC deadlines, give PDR a small phase slice
    // (`min(DEADLOCK_PDR_PHASE_CAP, remaining/DEADLOCK_PDR_DEADLINE_FRACTION)`)
    // so a slow PDR call cannot starve the BMC+BFS fallback below. PDR is
    // one of the fastest deadlock checkers when it converges; gating it to
    // no-deadline runs only (the prior policy) starved every MCC row of a
    // cheap proof opportunity and was the dominant cause of the v3
    // ReachabilityDeadlock 196-row timeout cliff.
    let pdr_deadline = match config.deadline() {
        Some(_) => deadlock_pdr_deadline(config.deadline()),
        None => None,
    };
    // PDR is dispatched on a separate thread with a hard wall-clock cap
    // mirroring the AIGER strategy below. `compute_p_invariants` inside
    // `solve_bounded_exact` does NOT poll the deadline on high-arity nets
    // (verified on ASLink-PT-04a, 1016 places, which spins the full wall
    // budget despite the 5s `DEADLOCK_PDR_PHASE_CAP`). Without abandon-on-
    // timeout, enabling PDR under MCC deadlines regresses the
    // ASLink-PT-04a..10a corpus from "CANNOT_COMPUTE within budget" to
    // "outer-harness kill at 60s+". The thread-leak is acceptable: the
    // ty-mcc process exits shortly anyway.
    //
    // catch_unwind covers the secondary `compute_p_invariants` i64
    // overflow surfaced on GPPP-PT-C1000N* (Gaussian elimination on very
    // high-arity nets).
    match run_phase_with_wall_cap(net, pdr_deadline, |net, dl| {
        global_properties_pdr::run_deadlock_pdr(net, dl).map(|verdict| {
            if verdict {
                Verdict::True
            } else {
                Verdict::False
            }
        })
    }) {
        PhaseOutcome::Verdict(verdict) => return verdict,
        PhaseOutcome::NoVerdict => {}
        PhaseOutcome::Abandoned => {
            eprintln!(
                "ReachabilityDeadlock: PDR phase exceeded {:?} wall cap; abandoning",
                DEADLOCK_PDR_PHASE_CAP,
            );
        }
        PhaseOutcome::Panicked => {
            eprintln!("ReachabilityDeadlock: PDR phase panicked; falling through to BMC+BFS");
        }
    }

    // --- Phase 0c: AIGER+IC3 (TRUE-only seeding) ---
    //
    // Spawned on a separate thread with abandon-on-timeout enforcement of
    // `DEADLOCK_AIGER_PHASE_CAP` (3s). This is required because the
    // underlying tla-aiger preprocessing pipeline does NOT poll its
    // `timeout` argument between SAT/preprocessing phases — passing a soft
    // deadline through to `check_aiger_sat` is insufficient (verified on
    // Philosophers-COL-000020 which spins the full wall budget). The only
    // way to bound wall-clock impact is to abandon the work after the cap
    // expires; the thread (and its memory) leak is acceptable because the
    // ty-mcc process exits at the global deadline anyway.
    //
    // Soundness: `try_aiger_deadlock` only returns `Verdict::True` and only
    // when a SAT witness replays to a real deadlocked Petri marking
    // (`run_aiger_deadlock_check` validates the terminal marking has zero
    // enabled transitions). Abandon, UNSAT, Unknown, and replay failure
    // all fall through to BMC+BFS — the fireability carve-out is preserved.
    //
    // `catch_unwind` covers latent panics in the AIGER stack (heap-heavy
    // with known overflow edge cases on extreme nets).
    let aiger_phase_deadline = deadlock_aiger_deadline(config.deadline());
    if !deadlock_aiger_skip(aiger_phase_deadline) {
        match run_phase_with_wall_cap(net, aiger_phase_deadline, try_aiger_deadlock) {
            PhaseOutcome::Verdict(verdict) => return verdict,
            PhaseOutcome::NoVerdict => {}
            PhaseOutcome::Abandoned => {
                eprintln!(
                    "ReachabilityDeadlock: AIGER phase exceeded {:?} wall cap; abandoning",
                    DEADLOCK_AIGER_PHASE_CAP,
                );
            }
            PhaseOutcome::Panicked => {
                eprintln!("ReachabilityDeadlock: AIGER phase panicked; falling through to BMC+BFS");
            }
        }
    }

    // --- Phase 0d: random-walk deadlock witness (TRUE-only) ---
    //
    // Runs AFTER PDR/AIGER and BEFORE the explicit BMC+BFS portfolio. On flat,
    // no-symmetry nets, BFS often cannot reach the stuck marking within budget
    // (PolyORBLF-PT-S04J06T06, ResAllocation-PT-R003C050 CANNOT_COMPUTE) while
    // a short random walk stumbles onto the reachable deadlock that consensus
    // tools report. The lane fires ONLY enabled transitions from the initial
    // marking and returns TRUE only from a directly-verified, reachable,
    // non-initial dead marking (zero transitions enabled, checked via
    // `net.is_enabled`).
    //
    // Soundness: this is a strict under-approximation. It NEVER returns False
    // or any universal verdict; a miss or timeout returns `false` and we fall
    // through to BMC+BFS unchanged. A reachable dead marking is a valid proof
    // that `ReachabilityDeadlock = TRUE`.
    //
    // Budget: ADDITIVE / leftover-only. `deadlock_walk_deadline` takes at most
    // `(remaining - DEADLOCK_BFS_FALLBACK_RESERVE) / 4` (capped at
    // DEADLOCK_WALK_PHASE_CAP) of the budget remaining after AIGER, and reserves
    // the full BFS-fallback tail, so it can NEVER starve the exhaustive BFS.
    let walk_phase_deadline = deadlock_walk_deadline(config.deadline());
    if !deadlock_walk_skip(walk_phase_deadline)
        && crate::examinations::reachability_walk::run_random_walk_deadlock(
            net,
            walk_phase_deadline,
        )
    {
        return Verdict::True;
    }

    // --- Phase 1: expensive verification ---
    //
    // Under finite MCC deadlines, do not use the scoped BMC/BFS portfolio:
    // even if BMC proves a verdict, the scope cannot return until the BFS
    // lane exits. Running BMC first and then a single BFS fallback keeps the
    // process deadline-aligned.
    if config.checkpoint().is_some() || config.deadline().is_some() {
        return deadlock_sequential(net, &config.clone().with_workers(1));
    }

    // Portfolio (BMC vs BFS in parallel) when workers >= 2;
    // sequential fallback when workers == 1 to honour the public API contract.
    match deadlock_portfolio_bfs_workers(config.workers()) {
        Some(bfs_workers) => deadlock_portfolio(net, config, bfs_workers),
        None => deadlock_sequential(net, config),
    }
}

/// Sequential deadlock verification: BMC then BFS. Used when workers == 1.
fn deadlock_sequential(net: &PetriNet, config: &ExplorationConfig) -> Verdict {
    // catch_unwind: `run_deadlock_bmc` calls `emit_p_invariant_strengthening`
    // which calls `compute_p_invariants` — same latent i64 overflow that
    // hits GPPP-PT-C1000N* in the structural pre-check above. Without the
    // guard, an overflow inside BMC's k-induction preamble takes down the
    // whole ty-mcc process. Soundness contract: a panicking BMC produces no
    // verdict and we fall through to BFS+POR.
    let bmc_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        global_properties_bmc::run_deadlock_bmc(net, config.deadline())
    }));
    match bmc_result {
        Ok(Some(true)) => return Verdict::True,
        Ok(Some(false)) => return Verdict::False,
        Ok(None) => {}
        Err(_) => {
            eprintln!("ReachabilityDeadlock: BMC phase panicked; falling through to BFS");
        }
    }

    // GPU explicit-BFS tier (probe-then-GPU, mirroring the StateSpace lane).
    // A bounded CPU probe decides small nets without touching the device:
    // a deadlock found under the cap is a valid witness (True), and a
    // POR-reduced exploration COMPLETING under the cap proves absence
    // (False — DeadlockPreserving POR preserves deadlock existence). Only a
    // tripped cap escalates to the exhaustive device BFS, whose
    // `deadlock_states` counter checks every distinct reachable marking
    // exactly once (each passes through a frontier once; the count kernel
    // runs on every frontier before the loop can exit). Fail-closed: any
    // GPU decline falls through to the full CPU BFS+POR unchanged.
    // Kill-switch `TY_MCC_DISABLE_GPU_STATESPACE`; probe skip lever
    // `TY_MCC_GPU_STATESPACE_FORCE` (testing).
    #[cfg(feature = "gpu")]
    if crate::gpu_state_space::gpu_lane_enabled(net) {
        if let Some(cap) = crate::gpu_state_space::cpu_probe_cap(config.max_states()) {
            let probe_config = ExplorationConfig::new(cap)
                .with_deadline(config.deadline())
                .with_examination(config.examination());
            let plan = ExecutionPlan::observer(PorStrategy::DeadlockPreserving);
            let mut observer = DeadlockObserver::new();
            let result = plan.run_observer(net, &probe_config, &mut observer);
            if observer.found_deadlock() {
                return Verdict::True;
            }
            if result.completed {
                return Verdict::False;
            }
            eprintln!(
                "[mcc] ReachabilityDeadlock: bounded CPU probe tripped (cap {cap}); \
                 escalating to the GPU lane"
            );
        }
        if let Some(deadlock) =
            crate::gpu_state_space::deadlock_exists_gpu(net, config.max_states())
        {
            return if deadlock {
                Verdict::True
            } else {
                Verdict::False
            };
        }
    }

    let plan = ExecutionPlan::observer(PorStrategy::DeadlockPreserving);
    let mut observer = DeadlockObserver::new();
    let result = match plan.run_checkpointable_observer(net, config, &mut observer) {
        Ok(result) => result,
        Err(error) => return checkpoint_cannot_compute("ReachabilityDeadlock", &error),
    };

    if observer.found_deadlock() {
        Verdict::True
    } else if result.completed {
        Verdict::False
    } else {
        Verdict::CannotCompute
    }
}

/// Portfolio deadlock verification: race BMC vs BFS. Used when workers >= 2.
fn deadlock_portfolio(net: &PetriNet, config: &ExplorationConfig, bfs_workers: usize) -> Verdict {
    let shared = Arc::new(SharedVerdict::new());
    let bfs_config = config.clone().with_workers(bfs_workers);

    std::thread::scope(|s| {
        let shared_bmc = Arc::clone(&shared);
        let bmc_handle = s.spawn(move || {
            if shared_bmc.is_resolved() {
                return;
            }
            match global_properties_bmc::run_deadlock_bmc(net, config.deadline()) {
                Some(true) => {
                    shared_bmc.publish(Verdict::True);
                }
                Some(false) => {
                    shared_bmc.publish(Verdict::False);
                }
                None => {}
            }
        });

        let shared_bfs = Arc::clone(&shared);
        let bfs_handle = s.spawn(move || {
            if shared_bfs.is_resolved() {
                return;
            }
            let plan = ExecutionPlan::observer(PorStrategy::DeadlockPreserving);
            let mut observer = PortfolioDeadlockObserver::new(Arc::clone(&shared_bfs));
            let result = plan.run_observer(net, &bfs_config, &mut observer);

            if shared_bfs.is_resolved() {
                return; // BMC won while we explored
            }
            if observer.found_deadlock() {
                shared_bfs.publish(Verdict::True);
            } else if result.completed {
                shared_bfs.publish(Verdict::False);
            }
        });

        let _ = bmc_handle.join();
        let _ = bfs_handle.join();
    });

    shared.verdict().unwrap_or(Verdict::CannotCompute)
}

/// Colored place groups for OneSafe checking.
///
/// For colored models, MCC defines 1-safe as: no colored place holds more
/// than 1 token total (sum across all color instances). Each group is a set
/// of unfolded PT place indices from the same colored parent. When empty,
/// the standard per-place check is used.
pub(crate) fn one_safe_verdict(
    net: &PetriNet,
    config: &ExplorationConfig,
    colored_groups: &[Vec<usize>],
) -> Verdict {
    one_safe_verdict_with_nupn(net, config, colored_groups, None)
}

pub(crate) fn one_safe_verdict_with_nupn(
    net: &PetriNet,
    config: &ExplorationConfig,
    colored_groups: &[Vec<usize>],
    nupn: Option<&crate::nupn::NupnStructure>,
) -> Verdict {
    // Build the unified safety-unit list. For colored inputs, the
    // multi-member colored groups have their token *sum* constrained ≤ 1;
    // every place NOT covered by any group is a singleton unit constrained
    // individually ≤ 1. Singletons must be checked too — a colored place
    // whose sort has cardinality 1 unfolds to a single PT place and is
    // filtered out of `colored_place_groups_as_usize` (members.len() > 1),
    // and a plain P/T place sharing the net with colored places never
    // appears in any group. Missing these checks turned BridgeAndVehicles
    // OneSafe FALSE→TRUE in 13-exam (initial marking `5'(dot)` ignored).
    let colored_mode = !colored_groups.is_empty();
    let safety_units: Vec<Vec<usize>> = if colored_mode {
        let mut covered = vec![false; net.num_places()];
        for group in colored_groups {
            for &p in group {
                covered[p] = true;
            }
        }
        let mut units: Vec<Vec<usize>> = colored_groups.to_vec();
        for (p, is_covered) in covered.iter().enumerate() {
            if !is_covered {
                units.push(vec![p]);
            }
        }
        units
    } else {
        (0..net.num_places()).map(|p| vec![p]).collect()
    };

    // Fast FALSE: initial marking is reachable.
    for unit in &safety_units {
        let sum: u64 = unit.iter().map(|&p| net.initial_marking[p]).sum();
        if sum > 1 {
            return Verdict::False;
        }
    }

    // NUPN safe=true metadata is useful context, but it is not a standalone
    // OneSafe proof: stale annotations can cover all places while the net has
    // reachable overflows. Keep proving through independent engines below.
    if !colored_mode && nupn.is_some_and(|nupn| nupn.proves_individual_one_safe(net)) {
        eprintln!(
            "OneSafe: NUPN unit-safe metadata covers all P/T places; requiring independent proof"
        );
    }

    // Structural short-circuit: if P-invariants (or an LP fallback) prove every
    // safety unit bounded ≤ 1, the net is 1-safe without state-space
    // exploration. Under an MCC deadline this is wall-capped on a worker thread
    // because `compute_p_invariants`/`lp_upper_bound` do NOT poll the deadline
    // and can otherwise spin the whole wall budget on high-arity nets (the
    // BridgeAndVehicles OneSafe overrun). Abandon/panic/`None` all fall through
    // to BMC/PDR/reduction/BFS, so capping is strictly verdict-preserving (the
    // phase only ever PROVES 1-safety). With no deadline (public API contract)
    // the original unbounded inline path runs.
    if config.deadline().is_some() {
        let units = safety_units.clone();
        let now = Instant::now();
        let phase_cap = ONE_SAFE_STRUCTURAL_PHASE_CAP.min(
            config
                .deadline()
                .map_or(ONE_SAFE_STRUCTURAL_PHASE_CAP, |d| {
                    d.saturating_duration_since(now)
                }),
        );
        match run_phase_with_wall_cap(net, Some(now + phase_cap), move |net, _dl| {
            one_safe_structural_short_circuit(net, &units)
        }) {
            PhaseOutcome::Verdict(verdict) => return verdict,
            PhaseOutcome::Abandoned => eprintln!(
                "OneSafe: structural P-invariant/LP phase exceeded {ONE_SAFE_STRUCTURAL_PHASE_CAP:?} wall cap; abandoning",
            ),
            PhaseOutcome::Panicked => eprintln!(
                "OneSafe: structural P-invariant/LP phase panicked; falling through to BMC/PDR/BFS",
            ),
            PhaseOutcome::NoVerdict => {}
        }
    } else if let Some(verdict) = one_safe_structural_short_circuit(net, &safety_units) {
        return verdict;
    }

    // Marking-equation 1-safe prover — P/T ONLY. One SMT query over the LP
    // relaxation (`∃p. m[p]≥2` UNSAT ⇒ every place ≤1 ⇒ 1-safe), trap-refined,
    // instead of the per-place LP bounds above — decides the large P/T nets whose
    // 505-place structural phase times out (Anderson-PT-*). Emits ONLY True, so
    // it cannot produce a wrong verdict. GATED to `!colored_mode`: for colored
    // nets OneSafe is a per-unit GROUP SUM, not the per-place predicate this
    // proves (the BridgeAndVehicles-COL wrong-TRUE trap). See
    // `crate::deadlock_region::onesafe_bounded`.
    if !colored_mode {
        if config.deadline().is_some() {
            if let PhaseOutcome::Verdict(verdict) = run_phase_with_wall_cap(
                net,
                one_safe_symbolic_deadline(config.deadline()),
                |net, dl| {
                    (crate::deadlock_region::onesafe_bounded(net, dl) == Some(true))
                        .then_some(Verdict::True)
                },
            ) {
                eprintln!("OneSafe: marking-equation region proves 1-safety (LP relaxation)");
                return verdict;
            }
        } else if crate::deadlock_region::onesafe_bounded(net, None) == Some(true) {
            eprintln!("OneSafe: marking-equation region proves 1-safety (LP relaxation)");
            return Verdict::True;
        }
    }

    // Decision-Diagram exact fast-path (off by default — gated by
    // `dd-backend`). Placed AFTER the cheap structural/LP shortcuts and
    // BEFORE BMC/PDR/reduction/BFS. On a small bounded net the DD backend
    // builds the EXACT reachable-marking set and answers OneSafe directly:
    //
    //   * P/T (non-colored): `forall p. m[p] <= 1` ⟺ `max_token_in_place
    //     <= 1`, read off the StateSpace metrics. This IS the PT OneSafe
    //     predicate.
    //   * Colored: the predicate is a per-safety-unit GROUP SUM, NOT the
    //     per-individual-place `max_token_in_place` (the
    //     BridgeAndVehicles-COL wrong-TRUE trap). We instead compute the
    //     exact reachable max of each unit's coefficient-1 weighted sum via
    //     `dispatch_upper_bounds_for_queries`, TRUE iff every unit max <= 1.
    //
    // Soundness: the reachable set is exact (build_sound_dd_spec gates the
    // unary encoding to a superset of every place's reachable projection),
    // so the verdict equals a completed exact check. On ANY DD failure
    // (decline/timeout/panic) we emit no verdict and fall through to the
    // existing pipeline unchanged.
    #[cfg(feature = "dd-backend")]
    {
        let dd_verdict = if colored_mode {
            super::dd_fastpath::try_dd_one_safe_colored(net, &safety_units, config.deadline())
        } else {
            super::dd_fastpath::try_dd_one_safe_pt(net, config.deadline())
        };
        if let Some(verdict) = dd_verdict {
            eprintln!("OneSafe: resolved exactly by DD reachable-set fast-path");
            return verdict;
        }
    }

    // SMT-based BMC + k-induction on the original net before reduction/BFS.
    // Note: BMC checks individual places only; skip for colored models to avoid
    // false TRUE on group-level violations.
    let symbolic_deadline = one_safe_symbolic_deadline(config.deadline());
    if !colored_mode {
        match global_properties_bmc::run_one_safe_bmc(net, symbolic_deadline) {
            Some(true) => return Verdict::True,
            Some(false) => return Verdict::False,
            None => {}
        }
    }

    // PDR/IC3 for OneSafe (PT nets only, same rationale as BMC above).
    if !colored_mode {
        match global_properties_pdr::run_one_safe_pdr(net, symbolic_deadline) {
            Some(true) => return Verdict::True,
            Some(false) => return Verdict::False,
            None => {}
        }
    }

    // --- Random-walk OneSafe FALSE-witness lane (FALSE-only) ---
    //
    // Runs AFTER PDR and BEFORE reduction/BFS, on the RAW `net` (never the
    // reduced net — a reduction can remap/collapse token counts and a
    // safety-unit place index may no longer mean the same thing). On flat,
    // no-symmetry nets the explicit BFS often cannot reach the overflowing
    // marking within budget while a short random walk stumbles onto it. The
    // lane fires ONLY enabled transitions from the initial marking and returns
    // FALSE only from a directly-verified reachable marking in which some
    // safety unit's token SUM is ≥ 2 (the group SUM, never the per-place max —
    // per-place max is the documented BridgeAndVehicles-COL wrong-TRUE trap).
    //
    // PT-only for now (gated on `!colored_mode`, matching BMC/PDR above);
    // colored is a follow-up. The trivial initial-marking overflow is already
    // settled by the fast-FALSE pre-check; this lane targets reachable-but-
    // non-initial overflows.
    //
    // Soundness: a strict under-approximation. It NEVER returns True or touches
    // the universal 1-safe side (that stays with structural/BMC/PDR/exact-BFS);
    // a miss, overflow, or timeout returns `false` and we fall through to the
    // reduction/BFS pipeline unchanged.
    //
    // Budget: ADDITIVE / leftover-only. `under_approx_lane_deadline` takes at
    // most `min(remaining / 4, 8s)` and reserves the BFS-fallback tail, so the
    // lane can NEVER starve the exhaustive BFS. A skip/already-expired sentinel
    // means there is no leftover slice — fall through without walking.
    if !colored_mode && !one_safe_walk_disabled() {
        let walk_deadline =
            crate::examinations::reachability::under_approx_lane_deadline(net, config.deadline());
        let walk_skip = walk_deadline.is_some_and(|d| Instant::now() >= d);
        if !walk_skip
            && crate::examinations::reachability_walk::run_random_walk_one_safe(
                net,
                &safety_units,
                walk_deadline,
            )
        {
            return Verdict::False;
        }
    }

    // For colored models, skip reduction (group-level token accounting is
    // complex with reduction place remapping) and BFS on original net with
    // the colored group observer. The observer must see *all* safety units
    // (multi-member colored groups AND any place not covered by a group) so
    // a violation at an uncovered singleton place is not silently missed —
    // this was the BridgeAndVehicles-COL OneSafe wrong-TRUE regression.
    if colored_mode {
        let mut observer = OneSafeObserver::new_colored(safety_units.clone());
        let result = if config.checkpoint().is_some() {
            match crate::explorer::explore_checkpointable_observer(net, config, &mut observer) {
                Ok(result) => result,
                Err(error) => return checkpoint_cannot_compute("OneSafe", &error),
            }
        } else {
            explore_observer(net, config, &mut observer)
        };
        return if !observer.is_safe() {
            Verdict::False
        } else if result.completed {
            Verdict::True
        } else {
            // BFS on the colored model did not complete (state cap or
            // deadline). Tier 3 Item 2: try the symbolic state-equation
            // fallback before giving up. Skipped for colored models
            // because the symbolic encoder operates on the unfolded PT
            // semantics — the colored group accounting would not match
            // and could surface a spurious verdict. Soundness > coverage.
            Verdict::CannotCompute
        };
    }

    // GPU explicit-BFS tier (probe-then-GPU, mirroring the StateSpace lane),
    // PT-only and on the RAW net (colored group-sum accounting stays with the
    // colored observer above; raw-net semantics avoid the reduction
    // reconstruction-bound caveats below entirely). A bounded CPU probe
    // decides small nets without touching the device (any place > 1 is a
    // reachable FALSE witness; completing under the cap proves TRUE); a
    // tripped cap escalates to the exhaustive device BFS, whose per-place
    // token maxima over the full reachable set decide 1-safety exactly:
    // all ≤ 1 ⇒ True, any ≥ 2 ⇒ False. Fail-closed: any GPU decline falls
    // through to the reduction/BFS pipeline unchanged.
    #[cfg(feature = "gpu")]
    if crate::gpu_state_space::gpu_lane_enabled(net) {
        if let Some(cap) = crate::gpu_state_space::cpu_probe_cap(config.max_states()) {
            let probe_config = ExplorationConfig::new(cap)
                .with_deadline(config.deadline())
                .with_examination(config.examination());
            let mut observer = OneSafeObserver::new();
            let result = explore_observer(net, &probe_config, &mut observer);
            if !observer.is_safe() {
                return Verdict::False;
            }
            if result.completed {
                return Verdict::True;
            }
            eprintln!(
                "[mcc] OneSafe: bounded CPU probe tripped (cap {cap}); \
                 escalating to the GPU lane"
            );
        }
        if let Some(maxima) =
            crate::gpu_state_space::place_maxima_gpu(net, config.max_states(), "OneSafe")
        {
            return if maxima.iter().all(|&m| m <= 1) {
                Verdict::True
            } else {
                Verdict::False
            };
        }
    }

    // OneSafe reasons directly on reduced token magnitudes, so it must stay on
    // the structural-only path until it becomes scale-aware. It also needs a
    // stricter reduction contract than deadlock: source-place elimination,
    // agglomeration, and non-decreasing place removal can hide token counts > 1.
    let reduced = match reduce_iterative_structural_one_safe(net) {
        Ok(reduced) => reduced,
        Err(error) => return reduction_cannot_compute("OneSafe", &error),
    };
    let config = config.refitted_for_net(&reduced.net).with_workers(1);
    let removed_unsafe = reduced
        .report
        .constant_places
        .iter()
        .chain(reduced.report.isolated_places.iter())
        .any(|&place| reduced.constant_value(place).is_some_and(|value| value > 1));
    if removed_unsafe {
        return Verdict::False;
    }

    // POR: places already proven 1-safe by reduced-net structural bounds do not
    // need to stay visible. That leaves a smaller safety-relevant subset on many
    // MCC-style models with independent resource-conserving subnets.
    let por_config = one_safe_por_config(&reduced, &config);
    let mut observer = OneSafeObserver::new();
    let result = if por_config.checkpoint().is_some() {
        match crate::explorer::explore_checkpointable_observer(
            &reduced.net,
            &por_config,
            &mut observer,
        ) {
            Ok(result) => result,
            Err(error) => return checkpoint_cannot_compute("OneSafe", &error),
        }
    } else {
        explore_observer(&reduced.net, &por_config, &mut observer)
    };

    if !observer.is_safe() {
        Verdict::False
    } else if result.completed {
        // LP-redundant places were removed from the reduced net but their
        // values may exceed 1 in the original net. The P-invariant upper
        // bound C/d is the maximum possible value; if it exceeds 1, we
        // cannot guarantee 1-safety from the reduced net alone.
        let redundant_bounded = reduced
            .reconstructions
            .iter()
            .all(|r| r.constant / r.divisor <= 1);
        if redundant_bounded {
            Verdict::True
        } else {
            eprintln!("OneSafe: reduced BFS completed but reconstruction bounds are not 1-safe");
            // Symbolic state-equation fallback (Tier 3 Item 2): the BFS
            // saw no violation but reduced-net reconstruction bounds
            // are too loose to declare 1-safety. The original net's
            // state equation can still rule out unsafe markings via
            // IC3-synthesised invariants. Dispatched on the original
            // net (not the reduced one) so the PT semantics line up.
            try_one_safe_symbolic_state_equation(net, config.deadline())
                .unwrap_or(Verdict::CannotCompute)
        }
    } else {
        eprintln!("OneSafe: reduced BFS incomplete before deadline/state limit");
        // Symbolic state-equation fallback (Tier 3 Item 2): explicit
        // BFS hopeless — Murphy-class blowups (4.15E+10 states) fall
        // through here. Last-shot symbolic dispatch on the original
        // net; UNKNOWN/overflow surfaces as CannotCompute (no guess).
        try_one_safe_symbolic_state_equation(net, config.deadline())
            .unwrap_or(Verdict::CannotCompute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc as PetriArc, PlaceInfo, TransitionInfo};

    fn stale_safe_true_nupn_overflow_net() -> PetriNet {
        PetriNet {
            name: Some("stale-safe-true-nupn-overflow".into()),
            places: vec![
                PlaceInfo {
                    id: "P0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "P1".into(),
                    name: None,
                },
            ],
            transitions: vec![TransitionInfo {
                id: "T0".into(),
                name: None,
                inputs: vec![PetriArc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![
                    PetriArc {
                        place: PlaceIdx(0),
                        weight: 1,
                    },
                    PetriArc {
                        place: PlaceIdx(1),
                        weight: 1,
                    },
                ],
            }],
            initial_marking: vec![1, 0],
        }
    }

    fn stale_safe_true_nupn(net: &PetriNet) -> crate::nupn::NupnStructure {
        let pnml = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="stale-safe-true-nupn-overflow" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="page0">
      <place id="P0"/>
      <place id="P1"/>
      <transition id="T0"/>
      <toolspecific tool="nupn" version="1.1">
        <size places="2" transitions="1" arcs="3"/>
        <structure units="3" root="u_root" safe="true">
          <unit id="u_root">
            <places/>
            <subunits>u0 u1</subunits>
          </unit>
          <unit id="u0">
            <places>P0</places>
            <subunits/>
          </unit>
          <unit id="u1">
            <places>P1</places>
            <subunits/>
          </unit>
        </structure>
      </toolspecific>
    </page>
  </net>
</pnml>"#;

        crate::nupn::parse_nupn(pnml, net)
            .expect("stale NUPN metadata should parse")
            .expect("NUPN metadata should be present")
    }

    /// P1 cross-check: the deadlock-preserving pre-reduction inside
    /// `deadlock_verdict` must lift the SAME boolean verdict as
    /// `deadlock_sequential` on the unreduced net. Covers both a deadlock-free
    /// net (no reachable dead marking) and a net that genuinely deadlocks (a
    /// sink transition that drains the only token), the latter exercising the
    /// gates kept OFF (`sink_transition_removal`) so the deadlock is preserved.
    #[test]
    fn test_p1_deadlock_reduction_matches_unreduced_sequential() {
        // Deadlock-free: 2-state alternator, always exactly one enabled.
        let alternator = PetriNet {
            name: Some("alternator".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t0".into(),
                    name: None,
                    inputs: vec![PetriArc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![PetriArc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t1".into(),
                    name: None,
                    inputs: vec![PetriArc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![PetriArc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1, 0],
        };

        // Deadlocking: a sink transition consumes the only token, after which
        // no transition is enabled.
        let sink_deadlock = PetriNet {
            name: Some("sink-deadlock".into()),
            places: vec![PlaceInfo {
                id: "p0".into(),
                name: None,
            }],
            transitions: vec![TransitionInfo {
                id: "drain".into(),
                name: None,
                inputs: vec![PetriArc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![],
            }],
            initial_marking: vec![1],
        };

        for net in [&alternator, &sink_deadlock] {
            let config = ExplorationConfig::new(10_000).with_workers(1);
            let reduced_verdict = deadlock_verdict(net, &config);
            let raw_verdict = deadlock_sequential(net, &config);
            assert_eq!(
                reduced_verdict, raw_verdict,
                "P1 reduced-net deadlock verdict must equal the unreduced \
                 sequential verdict for {:?}",
                net.name
            );
        }

        // And the expected ground-truth verdicts.
        let config = ExplorationConfig::new(10_000).with_workers(1);
        assert_eq!(deadlock_verdict(&alternator, &config), Verdict::False);
        assert_eq!(deadlock_verdict(&sink_deadlock, &config), Verdict::True);
    }

    /// P1 + Rule B regression: parallel-place pairs in the
    /// `ReachabilityDeadlock` pre-reduction must not distort the deadlock
    /// verdict. Both nets are deadlock-free alternators whose places A,B form
    /// a Rule B parallel pair (identical signatures, equal initial marking):
    ///
    ///   t1: A+B -> C;  t2: C -> A+B
    ///
    /// With m0(A,B) = 0 the unsound materialization DELETED the live consumer
    /// t1 (`blocked_by_constant` treated the replenished merged place as
    /// frozen at 0 tokens); with m0(A,B) = 1 it SUMMED the duplicate's arcs
    /// (t1 needed A >= 2) against an un-summed initial marking. Both produced
    /// a definite wrong TRUE for a deadlock-free net.
    #[test]
    fn test_p1_deadlock_rule_b_parallel_pair_no_false_deadlock() {
        let parallel_pair_net = |name: &str, m0_pair: u64, m0_c: u64| PetriNet {
            name: Some(name.into()),
            places: vec![
                PlaceInfo {
                    id: "A".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "B".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "C".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t1".into(),
                    name: None,
                    inputs: vec![
                        PetriArc {
                            place: PlaceIdx(0),
                            weight: 1,
                        },
                        PetriArc {
                            place: PlaceIdx(1),
                            weight: 1,
                        },
                    ],
                    outputs: vec![PetriArc {
                        place: PlaceIdx(2),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t2".into(),
                    name: None,
                    inputs: vec![PetriArc {
                        place: PlaceIdx(2),
                        weight: 1,
                    }],
                    outputs: vec![
                        PetriArc {
                            place: PlaceIdx(0),
                            weight: 1,
                        },
                        PetriArc {
                            place: PlaceIdx(1),
                            weight: 1,
                        },
                    ],
                },
            ],
            initial_marking: vec![m0_pair, m0_pair, m0_c],
        };

        // Mechanism 1 repro (m0 = 0: blocked-by-constant consumer deletion).
        let zero_marked = parallel_pair_net("rule-b-zero-marked", 0, 1);
        // Mechanism 2 repro (m0 = 1: summed arcs vs un-summed marking).
        let one_marked = parallel_pair_net("rule-b-one-marked", 1, 0);

        for net in [&zero_marked, &one_marked] {
            let config = ExplorationConfig::new(10_000).with_workers(1);
            let reduced_verdict = deadlock_verdict(net, &config);
            let raw_verdict = deadlock_sequential(net, &config);
            assert_eq!(
                reduced_verdict, raw_verdict,
                "P1 reduced-net deadlock verdict must equal the unreduced \
                 sequential verdict for {:?}",
                net.name
            );
            assert_eq!(
                reduced_verdict,
                Verdict::False,
                "{:?} alternates forever — reporting a deadlock is a wrong \
                 definite TRUE",
                net.name
            );
        }
    }

    #[test]
    fn test_deadlock_portfolio_bfs_workers_contract() {
        // workers == 1: sequential, no portfolio
        assert_eq!(deadlock_portfolio_bfs_workers(1), None);
        // workers == 2: portfolio with 1 BFS worker
        assert_eq!(deadlock_portfolio_bfs_workers(2), Some(1));
        // workers == 4 (MCC default): portfolio with 3 BFS workers
        assert_eq!(deadlock_portfolio_bfs_workers(4), Some(3));
        // workers == 0 edge case: sequential
        assert_eq!(deadlock_portfolio_bfs_workers(0), None);
    }

    #[test]
    fn test_one_safe_symbolic_deadline_preserves_bfs_tail_budget() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(20);

        assert_eq!(
            one_safe_symbolic_deadline_at(Some(deadline), now),
            Some(now + ONE_SAFE_SYMBOLIC_PHASE_CAP)
        );
    }

    #[test]
    fn test_one_safe_symbolic_deadline_none_stays_unbounded() {
        let now = Instant::now();

        assert_eq!(one_safe_symbolic_deadline_at(None, now), None);
    }

    #[test]
    fn test_one_safe_symbolic_deadline_expired_deadline_is_now() {
        let now = Instant::now();

        assert_eq!(one_safe_symbolic_deadline_at(Some(now), now), Some(now));
    }

    #[test]
    fn test_one_safe_symbolic_deadline_skips_when_only_bfs_reserve_remains() {
        let now = Instant::now();
        let deadline = now + ONE_SAFE_BFS_FALLBACK_RESERVE;

        assert_eq!(
            one_safe_symbolic_deadline_at(Some(deadline), now),
            Some(now)
        );
    }

    #[test]
    fn test_one_safe_symbolic_deadline_uses_remaining_time_above_reserve() {
        let now = Instant::now();
        let deadline = now + ONE_SAFE_BFS_FALLBACK_RESERVE + Duration::from_millis(1);

        assert_eq!(
            one_safe_symbolic_deadline_at(Some(deadline), now),
            Some(now + Duration::from_millis(1))
        );
    }

    #[test]
    fn test_one_safe_verdict_rejects_stale_safe_true_nupn_metadata() {
        let net = stale_safe_true_nupn_overflow_net();
        let nupn = stale_safe_true_nupn(&net);
        let config = ExplorationConfig::new(64);

        assert!(
            nupn.proves_individual_one_safe(&net),
            "regression fixture must model stale safe=true metadata that previously proved TRUE"
        );
        assert_eq!(
            one_safe_verdict_with_nupn(&net, &config, &[], Some(&nupn)),
            Verdict::False,
            "P0 -> P0+P1 reaches P1=2, so stale NUPN metadata must not publish TRUE"
        );
    }

    #[test]
    fn test_deadlock_aiger_deadline_caps_phase_when_budget_is_large() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(60);

        assert_eq!(
            deadlock_aiger_deadline_at(Some(deadline), now),
            Some(now + DEADLOCK_AIGER_PHASE_CAP)
        );
    }

    #[test]
    fn test_deadlock_aiger_deadline_none_stays_unbounded() {
        let now = Instant::now();

        assert_eq!(deadlock_aiger_deadline_at(None, now), None);
    }

    #[test]
    fn test_deadlock_aiger_deadline_skips_when_only_bfs_reserve_remains() {
        let now = Instant::now();
        let deadline = now + DEADLOCK_BFS_FALLBACK_RESERVE;

        // Returning Some(now) signals "skip" via deadlock_aiger_skip.
        assert_eq!(deadlock_aiger_deadline_at(Some(deadline), now), Some(now));
        assert!(deadlock_aiger_skip(Some(now)));
    }

    #[test]
    fn test_deadlock_aiger_deadline_skips_when_below_reserve() {
        // The exact `--timeout 15` scenario from the 2026-05-23 measurement:
        // after the 5s safety margin the deadline is +10s, which is ≤ the
        // 10s BFS reserve. AIGER must be skipped so BMC+BFS gets the budget.
        let now = Instant::now();
        let deadline = now + Duration::from_secs(10);

        assert_eq!(deadlock_aiger_deadline_at(Some(deadline), now), Some(now));
        assert!(deadlock_aiger_skip(deadlock_aiger_deadline_at(
            Some(deadline),
            now
        )));
    }

    #[test]
    fn test_deadlock_aiger_deadline_uses_remaining_time_above_reserve() {
        let now = Instant::now();
        let deadline = now + DEADLOCK_BFS_FALLBACK_RESERVE + Duration::from_millis(1);

        assert_eq!(
            deadlock_aiger_deadline_at(Some(deadline), now),
            Some(now + Duration::from_millis(1))
        );
    }

    #[test]
    fn test_deadlock_aiger_skip_none_is_false() {
        // None deadline means run-to-completion; never skip.
        assert!(!deadlock_aiger_skip(None));
    }
    #[test]
    fn test_deadlock_pdr_deadline_none_stays_unbounded() {
        let now = Instant::now();
        assert_eq!(deadlock_pdr_deadline_at(None, now), None);
    }

    #[test]
    fn test_deadlock_pdr_deadline_caps_at_phase_cap_for_large_budget() {
        // With a generous 60s deadline, the 1/3 share (20s) exceeds the 5s
        // phase cap, so the cap wins.
        let now = Instant::now();
        let deadline = now + Duration::from_secs(60);
        assert_eq!(
            deadlock_pdr_deadline_at(Some(deadline), now),
            Some(now + DEADLOCK_PDR_PHASE_CAP)
        );
    }

    #[test]
    fn test_deadlock_pdr_deadline_uses_fraction_for_small_budget() {
        // With a tight 9s deadline, the 1/3 share (3s) is below the 5s cap,
        // so the fraction wins.
        let now = Instant::now();
        let deadline = now + Duration::from_secs(9);
        assert_eq!(
            deadlock_pdr_deadline_at(Some(deadline), now),
            Some(now + Duration::from_secs(3))
        );
    }

    #[test]
    fn test_deadlock_aiger_phase_bounded_to_3s_on_deadlock_path() {
        // Bug 2 regression: confirm the AIGER phase the deadlock path
        // dispatches with is capped at `DEADLOCK_AIGER_PHASE_CAP` (3s) and
        // never receives the full global deadline. This is the structural
        // contract that prevents an AIGER preprocessing pass (which does
        // not poll the deadline between SAT calls — see the
        // `Transys::preprocess_configured` hang note on
        // `DEADLOCK_AIGER_PHASE_CAP`) from consuming the entire wall
        // budget on MCC rows.
        let now = Instant::now();
        let global = now + Duration::from_secs(60);
        let phase = deadlock_aiger_deadline_at(Some(global), now)
            .expect("AIGER phase deadline must be Some(_) under a finite global deadline");
        let phase_budget = phase.saturating_duration_since(now);
        assert!(
            phase_budget <= DEADLOCK_AIGER_PHASE_CAP,
            "AIGER phase budget {:?} exceeds cap {:?}",
            phase_budget,
            DEADLOCK_AIGER_PHASE_CAP,
        );
        // And it must leave at least DEADLOCK_BFS_FALLBACK_RESERVE for BMC+BFS.
        let remaining_after_aiger = global.saturating_duration_since(phase);
        assert!(
            remaining_after_aiger >= DEADLOCK_BFS_FALLBACK_RESERVE,
            "AIGER phase consumed too much: {:?} left for BMC+BFS (< {:?} reserve)",
            remaining_after_aiger,
            DEADLOCK_BFS_FALLBACK_RESERVE,
        );
        // The phase deadline must NOT be the global deadline itself.
        assert_ne!(
            phase, global,
            "AIGER phase deadline must be strictly less than the global deadline"
        );
    }

    #[test]
    fn test_deadlock_pdr_runs_with_finite_deadline() {
        // Soundness regression for Bug 1: under a finite deadline the PDR
        // phase deadline must be `Some(_)` (i.e. PDR is invoked under a
        // bounded budget), never `None` (skip). Prior to the fix, the gate
        // at deadlock_verdict skipped PDR entirely on every MCC row.
        let now = Instant::now();
        let global = now + Duration::from_secs(30);
        let phase = deadlock_pdr_deadline_at(Some(global), now);
        assert!(phase.is_some(), "PDR must run under finite deadlines");
        let phase = phase.unwrap();
        // The phase deadline must not exceed the global deadline.
        assert!(phase <= global);
        // And it must leave most of the budget for BMC+BFS.
        let pdr_budget = phase.saturating_duration_since(now);
        let remaining_after_pdr = global.saturating_duration_since(phase);
        assert!(
            remaining_after_pdr >= pdr_budget,
            "PDR budget ({:?}) must not exceed remaining ({:?})",
            pdr_budget,
            remaining_after_pdr
        );
    }
}
