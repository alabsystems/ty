// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Decision-Diagram exact fast-paths for the non-property examinations
//! `OneSafe`, `QuasiLiveness`, and `ReachabilityDeadlock`.
//!
//! Each of these examinations asks a question that is decided **exactly** by
//! the BDD reachable-marking set the DD backend already builds for
//! `StateSpace` / `ReachabilityCardinality` / `UpperBounds`:
//!
//! - **OneSafe** (P/T): `forall p. m[p] <= 1` over all reachable markings.
//!   Equivalently `max_token_in_place <= 1`, read directly off the
//!   `StateSpace` metrics.
//! - **QuasiLiveness**: every transition (P/T) — or every colored
//!   transition group — has some binding fireable from *some* reachable
//!   marking, i.e. `EF(IsFireable(group_members))` for each group.
//! - **ReachabilityDeadlock**: some reachable marking enables no
//!   transition, i.e. `EF(AND_t NOT IsFireable([t]))`.
//!
//! # Soundness
//!
//! Identical fail-closed contract to the StateSpace / Reachability /
//! UpperBounds DD paths:
//!
//! 1. [`crate::examinations::dd_spec::build_sound_dd_spec`] bounds every
//!    place by a sound LP upper bound, so the encoded value range is a
//!    *superset* of every place's reachable projection. The BDD reachable
//!    set is therefore **exact** (a converged `Ok` is never a truncated
//!    set). On `None` the DD path declines and the existing pipeline runs
//!    unchanged.
//! 2. `dispatch_reachable_state_space_metrics` /
//!    `dispatch_reachability_queries` error (never truncate) unless the
//!    fixed point converges, so any `Ok` here is over the exact reachable
//!    set. A metric / EF / AG read off an exact reachable set is ground
//!    truth — equivalent to a completed original-net BFS.
//! 3. The DD computation runs on a worker thread with a wall-clock budget
//!    clipped to preserve a BFS reserve under finite deadlines. On
//!    timeout, spawn failure, worker panic, or any DD error we emit **no**
//!    verdict and fall through (sound).
//!
//! Because the result is exact, a DD verdict here is committed as
//! authoritative TRUE/FALSE exactly like a completed-exploration verdict.
//! Every entry point is `#[cfg(feature = "dd-backend")]`, so default-off
//! builds are inert.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crate::output::Verdict;
use crate::petri_net::PetriNet;

/// Absolute ceiling on a non-property DD fast-path phase budget under a
/// finite deadline.
///
/// The DD reachable-set fixpoint runs *sequentially, blocking,* BEFORE the
/// explicit BFS+symmetry lane that is the actual decider for these three
/// examinations. The previous ceiling (`remaining − reserve`, up to 3600s)
/// handed DD essentially the entire wall clock, so on a 3600s competition run
/// a non-converging DD starved the BFS workhorse down to the 10s reserve and
/// turned BFS-decidable rows into CANNOT_COMPUTE (measured: Philosophers-PT
/// Deadlock spends ~37–44s wholly inside DD; SharedMemory-class rows time out
/// behind it). A DD fixpoint that has not converged in this ceiling is
/// overwhelmingly a BDD blow-up that would not converge in 3600s either, so
/// bounding it to a generous-but-finite slice captures essentially every DD
/// win while returning the bulk of the budget to BFS+symmetry.
///
/// Soundness-neutral: the DD result is EXACT, so a shorter budget can only make
/// DD *decline* (timeout ⇒ `None` ⇒ fall through to the equally-exact explicit
/// pipeline). It can never change a value — it is strictly verdict-preserving.
const DD_MAX_BUDGET: Duration = Duration::from_secs(45);

/// DD fast-path budget when **no** wall-clock deadline is supplied. Kept
/// small (mirrors the prior 5s cap) so a deadline-less run still declines
/// promptly to the existing pipeline; production MCC always supplies a
/// deadline and takes the scaled branch.
const DD_NO_DEADLINE_BUDGET: Duration = Duration::from_secs(5);

/// Budget reserved for the existing explicit/symbolic pipeline when the DD
/// path declines or times out under a finite deadline. Mirrors the
/// per-examination BFS fallback reserves (10s).
const DD_FALLBACK_RESERVE: Duration = Duration::from_secs(10);

/// Clip the DD phase budget under a finite global deadline so a BFS / BMC /
/// PDR reserve always survives.
///
/// Returns `None` when only the fallback reserve remains (DD is skipped),
/// `Some(min(remaining − reserve, DD_MAX_BUDGET))` under a finite deadline —
/// DD gets a fair slice on short runs but is clamped to [`DD_MAX_BUDGET`] on
/// long (competition-scale) deadlines so the explicit BFS+symmetry decider is
/// never starved behind a non-converging fixpoint — or [`DD_NO_DEADLINE_BUDGET`]
/// when no deadline is supplied.
fn dd_budget(global_deadline: Option<Instant>, now: Instant) -> Option<Duration> {
    let Some(global_deadline) = global_deadline else {
        return Some(DD_NO_DEADLINE_BUDGET);
    };
    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= DD_FALLBACK_RESERVE {
        return None;
    }
    // DD gets everything above the BFS reserve, clamped to the absolute ceiling
    // so a long deadline cannot let a non-converging fixpoint monopolize the run.
    let budget = DD_MAX_BUDGET.min(remaining.checked_sub(DD_FALLBACK_RESERVE).unwrap());
    (!budget.is_zero()).then_some(budget)
}

/// Run `f` (which performs the heavy BDD work over the sound spec) on a
/// worker thread with a hard wall-clock `budget`. Returns `Some(result)`
/// only when the worker finishes within `budget` and `f` returns `Some`.
///
/// On spawn failure, timeout, worker panic, or `f` returning `None`, this
/// returns `None` and the caller falls through to the existing pipeline.
/// The worker thread is detached on timeout: it drops its BDD manager on
/// the way out, so the budget is a soft cap with no resource leak — exactly
/// the StateSpace / Reachability DD detach-on-timeout pattern.
#[cfg(feature = "dd-backend")]
fn run_dd_phase<T, F>(
    spec: tla_dd::DdNetSpec,
    budget: Duration,
    name: &'static str,
    f: F,
) -> Option<T>
where
    T: Send + 'static,
    F: FnOnce(&tla_dd::DdNetSpec) -> Option<T> + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new()
        .name(name.to_string())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            // Install the wall-clock deadline INSIDE the worker so the
            // symbolic fixpoint (now possibly on the binary band, since
            // `MAX_PER_PLACE_BOUND` was raised to the binary cap) stops at its
            // budget instead of running this detached thread to the
            // billion-iteration backstop after the caller's `recv_timeout`
            // gives up — the leaked-unbounded-worker hazard a high-bound,
            // non-converging net (e.g. Kanban-PT-00200) would otherwise hit.
            // `run_isolated` inside each `dispatch_*` re-installs it on its own
            // big-stack worker, so the budget binds end to end. The big stack
            // also keeps deep OxiDD recursion off the small caller stack so a
            // high-bound net DECLINES, never crashes.
            let _deadline = tla_dd::set_thread_deadline(Instant::now() + budget);
            let _ = tx.send(f(&spec));
        });
    if handle.is_err() {
        eprintln!("{name}: DD fast-path thread spawn failed — using existing pipeline");
        return None;
    }
    match rx.recv_timeout(budget) {
        Ok(Some(result)) => Some(result),
        Ok(None) => {
            // `f` returned None: DD error / non-convergence. Fall through.
            eprintln!(
                "{name}: DD fast-path fell through (no convergence) — using existing pipeline"
            );
            None
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "{name}: DD fast-path exceeded {}s budget — using existing pipeline",
                budget.as_secs(),
            );
            None
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            eprintln!("{name}: DD fast-path worker panicked — using existing pipeline");
            None
        }
    }
}

/// Evaluate tla-dd reachability queries on the NATIVE ROBDD engine (the oxidd
/// `dispatch_reachability_queries` replacement). Converts each `DdReachQuery` to the
/// `(MddReachQuantifier, DdPredicate)` shape and runs `evaluate_reachability_via_bdd`;
/// the budget binds via tla-bdd's `reachable_within`. Intended to run INSIDE a
/// `run_dd_phase` worker (big stack + panic isolation). `None` on lowering decline
/// or budget exhaustion ⇒ caller falls through to the existing pipeline.
#[cfg(feature = "dd-backend")]
fn bdd_reachability(
    spec: &tla_dd::DdNetSpec,
    queries: &[tla_dd::DdReachQuery],
    budget: Duration,
) -> Option<Vec<bool>> {
    let qs: Vec<(tla_mdd::MddReachQuantifier, tla_dd::DdPredicate)> = queries
        .iter()
        .map(|q| {
            let quant = match q.quantifier {
                tla_dd::DdQuantifier::Ef => tla_mdd::MddReachQuantifier::Ef,
                tla_dd::DdQuantifier::Ag => tla_mdd::MddReachQuantifier::Ag,
            };
            (quant, q.predicate.clone())
        })
        .collect();
    crate::examinations::mdd_common::evaluate_reachability_via_bdd(
        spec,
        &qs,
        Some(Instant::now() + budget),
    )
}

/// Native-ROBDD UpperBounds for `u64` coefficient queries (the oxidd
/// `dispatch_upper_bounds_for_queries` replacement). Returns per-query reachable
/// maxima as `u64` (saturating); `None` on decline / overflow / budget exhaustion.
#[cfg(feature = "dd-backend")]
fn bdd_upper_bounds(
    spec: &tla_dd::DdNetSpec,
    coeff_vecs: &[Vec<u64>],
    budget: Duration,
) -> Option<Vec<u64>> {
    let queries: Vec<Vec<i128>> = coeff_vecs
        .iter()
        .map(|q| q.iter().map(|&c| c as i128).collect())
        .collect();
    let raw = crate::examinations::mdd_common::upper_bounds_via_bdd(
        spec,
        &queries,
        Some(Instant::now() + budget),
    )?;
    raw.into_iter().map(|x| u64::try_from(x).ok()).collect()
}

/// Exact DD fast-path for **P/T OneSafe**.
///
/// Returns `Some(Verdict::True)` iff every reachable marking has at most
/// one token in every place (`max_token_in_place <= 1`), `Some(False)`
/// otherwise, or `None` when the DD path declines / fails (caller continues
/// with the existing pipeline).
///
/// # Why P/T only
///
/// `max_token_in_place` is the per-**individual**-place reachable maximum.
/// On a colored net the OneSafe predicate is a per-safety-unit **group
/// sum**, which `max_token_in_place` does NOT measure: a net that is
/// 1-safe per individual unfolded place can still hold 2 tokens in a
/// colored place's group sum (the BridgeAndVehicles-COL wrong-TRUE trap).
/// Colored OneSafe is therefore handled by the dedicated group-sum DD path
/// [`try_dd_one_safe_colored`].
#[cfg(feature = "dd-backend")]
pub(super) fn try_dd_one_safe_pt(
    net: &PetriNet,
    global_deadline: Option<Instant>,
) -> Option<Verdict> {
    let budget = dd_budget(global_deadline, Instant::now())?;
    let (spec, _bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    let metrics = run_dd_phase(spec, budget, "OneSafe", |spec| {
        Some(crate::examinations::mdd_common::state_space_metrics_via_bdd(spec))
    })?;
    Some(if metrics.max_token_in_place <= 1 {
        Verdict::True
    } else {
        Verdict::False
    })
}

/// Exact DD fast-path for **colored OneSafe** via per-safety-unit group
/// sums.
///
/// The colored OneSafe predicate is `forall unit. (sum_{p in unit} m[p])
/// <= 1` over all reachable markings. We compute, via
/// `dispatch_upper_bounds_for_queries`, the exact reachable maximum of each
/// unit's coefficient-1 weighted sum and return TRUE iff every unit's max
/// is `<= 1`. This is exact (the reachable set is exact and the per-query
/// weighted-sum max is differentially tested vs BFS), so it does not fall
/// into the `max_token_in_place` individual-place trap.
///
/// `safety_units` is the unified safety-unit list assembled by the OneSafe
/// verdict function (multi-member colored groups plus every uncovered
/// singleton place). Returns `None` when the DD path declines / fails.
#[cfg(feature = "dd-backend")]
pub(super) fn try_dd_one_safe_colored(
    net: &PetriNet,
    safety_units: &[Vec<usize>],
    global_deadline: Option<Instant>,
) -> Option<Verdict> {
    let budget = dd_budget(global_deadline, Instant::now())?;
    let (spec, _bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    let num_places = net.num_places();

    // Build one coefficient-1 vector per safety unit. A place listed twice
    // in a unit would count twice — units never repeat a place, so each
    // coefficient is 0/1. Defensive: reject any out-of-range index.
    let mut coeff_vecs: Vec<Vec<u64>> = Vec::with_capacity(safety_units.len());
    for unit in safety_units {
        let mut coeffs = vec![0u64; num_places];
        for &p in unit {
            if p >= num_places {
                return None;
            }
            coeffs[p] = 1;
        }
        coeff_vecs.push(coeffs);
    }
    if coeff_vecs.is_empty() {
        // No units to check ⇒ vacuously 1-safe. (The caller's fast-FALSE
        // initial-marking scan already ran; an empty unit list means there
        // are no places, which the verdict fn treats as trivially safe.)
        return Some(Verdict::True);
    }

    let maxima = run_dd_phase(spec, budget, "OneSafe", move |spec| {
        bdd_upper_bounds(spec, &coeff_vecs, budget)
    })?;
    Some(if maxima.iter().all(|&m| m <= 1) {
        Verdict::True
    } else {
        Verdict::False
    })
}

/// Exact DD fast-path for **QuasiLiveness**.
///
/// QuasiLive iff every transition group has some member transition fireable
/// from some reachable marking. For P/T input each transition is its own
/// singleton group; for colored input each colored transition's unfolded
/// bindings form one group. We issue one `EF(IsFireable(group_members))`
/// query per group (`IsFireable` has OR semantics over the listed
/// transitions, so this is exactly "some binding of this group is reachable
/// and fireable"). Verdict TRUE iff every query is true, FALSE otherwise.
///
/// `transition_groups` mirrors the verdict function's `colored_transition_
/// groups`: empty ⇒ P/T (each transition is its own group). Returns `None`
/// when the DD path declines / fails.
#[cfg(feature = "dd-backend")]
pub(super) fn try_dd_quasi_liveness(
    net: &PetriNet,
    transition_groups: &[Vec<usize>],
    global_deadline: Option<Instant>,
) -> Option<Verdict> {
    let budget = dd_budget(global_deadline, Instant::now())?;
    let (spec, _bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    let num_transitions = net.num_transitions();

    // One EF(IsFireable(group)) query per group. The groups must cover
    // every transition: colored groups list their members, and any
    // transition not in a group is checked as a singleton.
    let mut queries: Vec<tla_dd::DdReachQuery> = Vec::new();
    if transition_groups.is_empty() {
        for t in 0..num_transitions {
            queries.push(tla_dd::DdReachQuery {
                quantifier: tla_dd::DdQuantifier::Ef,
                predicate: tla_dd::DdPredicate::IsFireable(vec![t]),
            });
        }
    } else {
        let mut covered = vec![false; num_transitions];
        for group in transition_groups {
            let mut members = Vec::with_capacity(group.len());
            for &t in group {
                if t >= num_transitions {
                    return None;
                }
                covered[t] = true;
                members.push(t);
            }
            queries.push(tla_dd::DdReachQuery {
                quantifier: tla_dd::DdQuantifier::Ef,
                predicate: tla_dd::DdPredicate::IsFireable(members),
            });
        }
        for (t, &is_covered) in covered.iter().enumerate() {
            if !is_covered {
                queries.push(tla_dd::DdReachQuery {
                    quantifier: tla_dd::DdQuantifier::Ef,
                    predicate: tla_dd::DdPredicate::IsFireable(vec![t]),
                });
            }
        }
    }

    if queries.is_empty() {
        // No transitions ⇒ vacuously quasi-live.
        return Some(Verdict::True);
    }

    let expected = queries.len();
    let verdicts = run_dd_phase(spec, budget, "QuasiLiveness", move |spec| {
        bdd_reachability(spec, &queries, budget)
    })?;
    if verdicts.len() != expected {
        eprintln!(
            "QuasiLiveness: DD fast-path returned {} verdicts for {expected} queries — \
             treating as failure and using existing pipeline",
            verdicts.len(),
        );
        return None;
    }
    Some(if verdicts.iter().all(|&v| v) {
        Verdict::True
    } else {
        Verdict::False
    })
}

/// Exact DD fast-path for **ReachabilityDeadlock**.
///
/// A deadlock marking is one that enables no transition. The deadlock-
/// existence question is therefore `EF(AND_t NOT IsFireable([t]))` over the
/// exact reachable set: TRUE iff some reachable marking enables no
/// transition, FALSE iff every reachable marking enables at least one.
///
/// This is expressible exactly with the available DD combinators
/// (`And` + `Not` + `IsFireable`), so the verdict is ground truth. Returns
/// `None` when the DD path declines / fails.
///
/// A net with no transitions has the initial marking trivially deadlocked
/// ⇒ TRUE; the conjunction over an empty transition set is `And([]) == true`
/// and the initial marking is always reachable, so the EF query yields
/// TRUE without special-casing.
#[cfg(feature = "dd-backend")]
pub(super) fn try_dd_deadlock(net: &PetriNet, global_deadline: Option<Instant>) -> Option<Verdict> {
    let budget = dd_budget(global_deadline, Instant::now())?;
    let (spec, _bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    let num_transitions = net.num_transitions();

    // AND over all transitions of NOT IsFireable([t]) — "no transition is
    // enabled". `And([])` is true, so a net with zero transitions makes
    // the initial marking trivially deadlocked (matches Petri semantics).
    let conjuncts: Vec<tla_dd::DdPredicate> = (0..num_transitions)
        .map(|t| tla_dd::DdPredicate::Not(Box::new(tla_dd::DdPredicate::IsFireable(vec![t]))))
        .collect();
    let query = tla_dd::DdReachQuery {
        quantifier: tla_dd::DdQuantifier::Ef,
        predicate: tla_dd::DdPredicate::And(conjuncts),
    };

    let verdicts = run_dd_phase(spec, budget, "ReachabilityDeadlock", move |spec| {
        bdd_reachability(spec, std::slice::from_ref(&query), budget)
    })?;
    let &deadlock_reachable = verdicts.first()?;
    Some(if deadlock_reachable {
        Verdict::True
    } else {
        Verdict::False
    })
}

/// Exact DD fast-path for **P/T StableMarking**.
///
/// StableMarking asks, for each place `p`: is `m[p]` CONSTANT across **all**
/// reachable markings (a *stable place*)? The examination verdict is TRUE iff
/// **some** place is stable, FALSE iff every place is unstable.
///
/// A place is constant iff it never deviates from its initial marking on any
/// reachable marking — `build_sound_dd_spec` guarantees the initial marking is
/// reachable and is the unique reachable value of a constant place, so
/// "constant" ⟺ "equal to `init[p]` on every reachable marking". Equivalently
/// place `p` is **unstable** iff
///
/// ```text
///   EF( m[p] != init[p] )   ⟺   EF( m[p] < init[p]  OR  m[p] > init[p] )
/// ```
///
/// is reachable. We issue exactly that EF query per place over the exact
/// reachable set and read each verdict back:
///
/// - EF false ⇒ place `p` is constant ⇒ stable ⇒ StableMarking **TRUE**.
/// - EF true for every place ⇒ no stable place ⇒ StableMarking **FALSE**.
///
/// The inequality `!=` is expressed with the available combinators as
/// `Or( IntLe(m[p], init-1), Not(IntLe(m[p], init)) )`: the first disjunct is
/// `m[p] <= init-1` (omitted when `init == 0`, since `m[p] < 0` is
/// unsatisfiable), the second is `m[p] > init`. Each query's predicate is exact
/// on one-hot markings, so a converged reachable set makes every verdict ground
/// truth — equivalent to a completed StableMarking BFS.
///
/// # Why P/T only
///
/// On a colored net StableMarking is a per-colored-group **sum** constancy
/// question (`sum_{p in group} m[p]` constant), not a per-individual-place
/// constancy question — the same group-sum vs individual-place distinction that
/// forces the dedicated [`try_dd_one_safe_colored`] path. This lane is the
/// per-individual-place formulation and is therefore restricted to P/T; the
/// caller only invokes it when `colored_groups` is empty.
///
/// Returns `None` when the DD path declines / fails (caller continues with the
/// existing pipeline), `Some(Verdict::True)` if some place is provably constant,
/// `Some(Verdict::False)` if every place is provably non-constant.
#[cfg(feature = "dd-backend")]
pub(super) fn try_dd_stable_marking(
    net: &PetriNet,
    global_deadline: Option<Instant>,
) -> Option<Verdict> {
    let budget = dd_budget(global_deadline, Instant::now())?;
    let (spec, _bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    let num_places = net.num_places();
    if num_places == 0 {
        // No places ⇒ vacuously every (zero) place is constant. Match the
        // explicit observer, which reports TRUE on a completed empty net.
        return Some(Verdict::True);
    }
    if net.initial_marking.len() != num_places {
        // Defensive: build_sound_dd_spec already enforces this.
        return None;
    }

    // One `EF( m[p] != init[p] )` query per place, in place order.
    let queries: Vec<tla_dd::DdReachQuery> = (0..num_places)
        .map(|p| {
            let init = net.initial_marking[p];
            // m[p] > init : NOT( m[p] <= init ).
            let greater = tla_dd::DdPredicate::Not(Box::new(tla_dd::DdPredicate::IntLe(
                tla_dd::DdIntExpr::TokensCount(vec![p]),
                tla_dd::DdIntExpr::Constant(init),
            )));
            let predicate = if init == 0 {
                // m[p] < 0 is unsatisfiable, so `!= 0` reduces to `> 0`.
                greater
            } else {
                // m[p] < init : m[p] <= init - 1.
                let less = tla_dd::DdPredicate::IntLe(
                    tla_dd::DdIntExpr::TokensCount(vec![p]),
                    tla_dd::DdIntExpr::Constant(init - 1),
                );
                tla_dd::DdPredicate::Or(vec![less, greater])
            };
            tla_dd::DdReachQuery {
                quantifier: tla_dd::DdQuantifier::Ef,
                predicate,
            }
        })
        .collect();

    let expected = queries.len();
    let unstable = run_dd_phase(spec, budget, "StableMarking", move |spec| {
        bdd_reachability(spec, &queries, budget)
    })?;
    if unstable.len() != expected {
        eprintln!(
            "StableMarking: DD fast-path returned {} verdicts for {expected} queries — \
             treating as failure and using existing pipeline",
            unstable.len(),
        );
        return None;
    }
    // `unstable[p]` is true iff `m[p] != init[p]` is reachable. Some place is
    // stable ⇒ TRUE; every place reaches a deviation ⇒ FALSE.
    Some(if unstable.iter().any(|&deviates| !deviates) {
        Verdict::True
    } else {
        Verdict::False
    })
}

/// Exact DD fast-path for **L4-Liveness** via the nested-CTL symbolic
/// evaluator.
///
/// A transition is L4-live iff from **every** reachable marking it can
/// **eventually** fire — the nested CTL property `AG(EF(IsFireable(t)))`.
/// The net is L4-live iff every transition (P/T) — or every colored
/// transition group (some binding fireable) — is L4-live, i.e.
///
/// ```text
///   AND_groups  AG( EF( IsFireable(group_members) ) )
/// ```
///
/// holds at the initial marking over the exact reachable set.
///
/// # Why the nested-CTL evaluator (not `dispatch_reachability_queries`)
///
/// `dispatch_reachability_queries`'s `DdQuantifier` only expresses a *flat*,
/// single-level `EF`/`AG` over the reachable set — it cannot express the
/// `AG(EF(...))` nesting. The 18k-check oracle-verified
/// [`tla_dd::symbolic_ctl`] evaluator computes the nested CTL fixpoints
/// (`EF`/`AG`) directly over the exact reachable set and matches
/// `tla_mc_core::CtlEngine` set-for-set, so it is the sound primitive for
/// this property. This mirrors the proven production CTL lane
/// (`examinations::ctl::pipeline::try_symbolic_ctl_verdict`).
///
/// We build ONE combined conjunction `And([AG(EF(IsFireable(g))) for g])`
/// and evaluate it once, so the (expensive) reachable-set build is shared
/// across all groups. The empty-transition case falls out: `And([])` is
/// `true`, so a net with no transitions is vacuously live — matching the
/// mu-calculus / SCC observers.
///
/// `IsFireable(group_members)` has OR-over-members semantics, so for a
/// colored group the inner `EF` asks "can we reach a marking where SOME
/// binding of this colored transition is enabled?" — exactly the colored-
/// L4-liveness question, and for P/T singletons it is the classic
/// `AG(EF(IsFireable([t])))`.
///
/// # Soundness
///
/// - `build_sound_dd_spec` bounds every place by a sound LP upper bound, so
///   the BDD reachable set is **exact** (a converged result is never a
///   truncated set). The spec preserves transition order 1:1 with
///   `net.transitions`, so each `IsFireable(vec![t])` references the right
///   transition — the same invariant [`try_dd_quasi_liveness`] relies on.
/// - The CTL evaluator returns the verdict at the initial marking
///   (intersected with the reachable set), which is ground truth over the
///   exact reachable set — equivalent to a completed original-net liveness
///   analysis.
/// - Fail-closed: on `build_sound_dd_spec` decline, any out-of-range index,
///   timeout, worker panic, or `symbolic_ctl_holds_seeded` returning `None`
///   (above-cap band / over budget / OOM), we emit **no** verdict and the
///   caller falls through to the existing mu-calculus / SCC pipeline
///   unchanged.
///
/// # Why P/T only (caller-enforced)
///
/// The colored grouping is carried by `transition_groups` exactly as in
/// [`try_dd_quasi_liveness`]; the per-individual-place reachable set the DD
/// builds is the unfolded P/T reachable set, and `IsFireable` over a group's
/// binding indices asks the colored question via OR semantics. This is sound
/// for both P/T (singleton groups) and colored (passed groups). The caller
/// runs it on the **original** `net` (the Liveness verdict fn uses an
/// identity reduction for the colored path, and for P/T the DD reachable set
/// is exact over the original net regardless of any explicit-pipeline
/// reduction), so the group indices remain valid.
///
/// Returns `None` when the DD path declines / fails (caller continues with
/// the existing pipeline), `Some(Verdict::True)` if every group is provably
/// L4-live, `Some(Verdict::False)` if some group is provably not L4-live.
#[cfg(feature = "dd-backend")]
pub(super) fn try_dd_liveness(
    net: &PetriNet,
    transition_groups: &[Vec<usize>],
    global_deadline: Option<Instant>,
) -> Option<Verdict> {
    use tla_mdd::CtlFormulaTemplate as T;

    let budget = dd_budget(global_deadline, Instant::now())?;
    let (spec, _bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    let num_transitions = net.num_transitions();

    // One `AG(EF(IsFireable(group)))` conjunct per group. The groups must
    // cover every transition: colored groups list their members, and any
    // transition not in a group is its own singleton.
    let mut conjuncts: Vec<T<tla_dd::DdPredicate>> = Vec::new();
    let ag_ef_fireable = |members: Vec<usize>| {
        T::AG(Box::new(T::EF(Box::new(T::Atom(
            tla_dd::DdPredicate::IsFireable(members),
        )))))
    };
    if transition_groups.is_empty() {
        for t in 0..num_transitions {
            conjuncts.push(ag_ef_fireable(vec![t]));
        }
    } else {
        let mut covered = vec![false; num_transitions];
        for group in transition_groups {
            let mut members = Vec::with_capacity(group.len());
            for &t in group {
                if t >= num_transitions {
                    return None;
                }
                covered[t] = true;
                members.push(t);
            }
            conjuncts.push(ag_ef_fireable(members));
        }
        for (t, &is_covered) in covered.iter().enumerate() {
            if !is_covered {
                conjuncts.push(ag_ef_fireable(vec![t]));
            }
        }
    }

    // `And([])` is true, so a net with zero transitions is vacuously live —
    // matching the mu-calculus / SCC observers. No special-case needed.
    let formula = T::And(conjuncts);

    let holds = run_dd_phase(spec, budget, "Liveness", move |spec| {
        // Native tla-bdd CTL (≡ the MDD CTL lane; oxidd removed). run_dd_phase
        // keeps the worker/panic isolation + budget; the caller's deadline binds
        // the fixpoint via `reachable_within` ⇒ a clean DECLINE (`None`) on
        // timeout / above-cap / OOM. Mirrors the migrated `try_symbolic_ctl_verdict`.
        crate::examinations::mdd_common::evaluate_ctl_via_bdd(spec, &formula, global_deadline)
    })?;
    Some(if holds { Verdict::True } else { Verdict::False })
}

#[cfg(all(test, feature = "dd-backend"))]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
    use std::collections::HashSet;

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.into(),
            name: None,
        }
    }

    fn trans(id: &str, inputs: Vec<(u32, u64)>, outputs: Vec<(u32, u64)>) -> TransitionInfo {
        TransitionInfo {
            id: id.into(),
            name: None,
            inputs: inputs
                .into_iter()
                .map(|(p, w)| Arc {
                    place: PlaceIdx(p),
                    weight: w,
                })
                .collect(),
            outputs: outputs
                .into_iter()
                .map(|(p, w)| Arc {
                    place: PlaceIdx(p),
                    weight: w,
                })
                .collect(),
        }
    }

    /// 2-place swap net: p0+p1 conserved at 1. Reachable {(1,0),(0,1)}.
    /// 1-safe; both transitions fire; no deadlock.
    fn swap_net() -> PetriNet {
        PetriNet {
            name: Some("swap".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t01", vec![(0, 1)], vec![(1, 1)]),
                trans("t10", vec![(1, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// Explicit BFS over the reachable set, returning the set of markings.
    fn reachable_markings(net: &PetriNet) -> HashSet<Vec<u64>> {
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        seen.insert(net.initial_marking.clone());
        let mut frontier = vec![net.initial_marking.clone()];
        while let Some(m) = frontier.pop() {
            for tid in 0..net.transitions.len() {
                let t = TransitionIdx(tid as u32);
                if !net.is_enabled(&m, t) {
                    continue;
                }
                let mut next = m.clone();
                for arc in &net.transitions[tid].inputs {
                    next[arc.place.0 as usize] -= arc.weight;
                }
                for arc in &net.transitions[tid].outputs {
                    next[arc.place.0 as usize] += arc.weight;
                }
                if seen.insert(next.clone()) {
                    frontier.push(next);
                }
            }
        }
        seen
    }

    fn brute_one_safe(net: &PetriNet) -> bool {
        reachable_markings(net)
            .iter()
            .all(|m| m.iter().all(|&v| v <= 1))
    }

    fn brute_deadlock_exists(net: &PetriNet) -> bool {
        reachable_markings(net).iter().any(|m| {
            (0..net.transitions.len()).all(|tid| !net.is_enabled(m, TransitionIdx(tid as u32)))
        })
    }

    fn brute_quasi_live(net: &PetriNet) -> bool {
        let reach = reachable_markings(net);
        (0..net.transitions.len()).all(|tid| {
            reach
                .iter()
                .any(|m| net.is_enabled(m, TransitionIdx(tid as u32)))
        })
    }

    /// Exhaustive L4-Liveness reference over the full reachable set.
    ///
    /// `t` is L4-live iff from EVERY reachable marking there is a firing
    /// sequence that eventually enables `t`, i.e. `AG(EF(IsFireable(t)))`.
    /// The net is live iff every transition is L4-live (empty transition set
    /// ⇒ vacuously live).
    ///
    /// Computed directly over the explicit reachable-marking graph: for each
    /// marking `m`, the set of transitions that are "eventually-fireable from
    /// `m`" is the union of transitions enabled in any marking reachable from
    /// `m` (forward closure). `t` is L4-live iff it is eventually-fireable
    /// from every reachable marking. This is the textbook `AG(EF(enabled t))`
    /// semantics, computed independently of the DD evaluator.
    fn brute_live(net: &PetriNet) -> bool {
        let nt = net.transitions.len();
        if nt == 0 {
            return true;
        }
        let reach: Vec<Vec<u64>> = reachable_markings(net).into_iter().collect();
        let index: std::collections::HashMap<Vec<u64>, usize> = reach
            .iter()
            .enumerate()
            .map(|(i, m)| (m.clone(), i))
            .collect();
        let n = reach.len();
        // successors[i] = markings reachable in one step from reach[i].
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        // enabled_here[i][t] = transition t enabled at reach[i].
        let mut enabled_here: Vec<Vec<bool>> = vec![vec![false; nt]; n];
        for (i, m) in reach.iter().enumerate() {
            for tid in 0..nt {
                let t = TransitionIdx(tid as u32);
                if !net.is_enabled(m, t) {
                    continue;
                }
                enabled_here[i][tid] = true;
                let mut next = m.clone();
                for arc in &net.transitions[tid].inputs {
                    next[arc.place.0 as usize] -= arc.weight;
                }
                for arc in &net.transitions[tid].outputs {
                    next[arc.place.0 as usize] += arc.weight;
                }
                successors[i].push(index[&next]);
            }
        }
        // ef_fireable[i] = bitset of transitions enabled in SOME marking
        // reachable from reach[i] (forward closure). Iterate to fixpoint.
        let mut ef: Vec<Vec<bool>> = enabled_here.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                for &j in &successors[i].clone() {
                    for tid in 0..nt {
                        if ef[j][tid] && !ef[i][tid] {
                            ef[i][tid] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        // Net is live iff EVERY transition is EF-fireable from EVERY marking.
        (0..nt).all(|tid| (0..n).all(|i| ef[i][tid]))
    }

    /// Exhaustive per-place StableMarking reference over the full reachable
    /// set. Returns `(verdict, stable_mask)` where `verdict` is TRUE iff some
    /// place is constant (= its initial marking on every reachable marking)
    /// and `stable_mask[p]` is that per-place constancy bit.
    fn brute_stable_marking(net: &PetriNet) -> (bool, Vec<bool>) {
        let reach = reachable_markings(net);
        let init = &net.initial_marking;
        let stable: Vec<bool> = (0..net.places.len())
            .map(|p| reach.iter().all(|m| m[p] == init[p]))
            .collect();
        (stable.iter().any(|&s| s), stable)
    }

    /// DD verdict ⟺ exhaustive BFS verdict on a battery of nets, including
    /// genuinely-stable, all-unstable, and deadlocking nets. 0 disagreements.
    #[test]
    fn stable_marking_dd_matches_brute_force_battery() {
        // 1. swap net: p0+p1 conserved at 1, but each individual place
        //    flips between 0 and 1 ⇒ NO stable place ⇒ FALSE.
        let swap = swap_net();
        // 2. one genuinely stable place: p2 is a constant sink-free isolated
        //    counter that no transition touches; p0/p1 swap.
        let one_stable = PetriNet {
            name: Some("one-stable".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t01", vec![(0, 1)], vec![(1, 1)]),
                trans("t10", vec![(1, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![1, 0, 5], // p2 untouched ⇒ constant at 5.
        };
        // 3. deadlock net: p0 -> p1, then nothing. p0 flips 1->0, p1 flips
        //    0->1 ⇒ both unstable ⇒ FALSE (and (0,1) is a dead marking).
        let sink = PetriNet {
            name: Some("sink".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t", vec![(0, 1)], vec![(1, 1)])],
            initial_marking: vec![1, 0],
        };
        // 4. fully stable net: no transitions ⇒ every place constant ⇒ TRUE.
        let frozen = PetriNet {
            name: Some("frozen".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![],
            initial_marking: vec![3, 7],
        };
        // 5. higher-token / many-marking net: a 3-place ring moving 4 tokens
        //    around. Each place ranges over several values ⇒ all unstable.
        //    Exercises the binary-band encoding path and a non-trivial
        //    reachable set (the DD-eligible profile).
        let ring = PetriNet {
            name: Some("ring4".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t0", vec![(0, 1)], vec![(1, 1)]),
                trans("t1", vec![(1, 1)], vec![(2, 1)]),
                trans("t2", vec![(2, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![4, 0, 0],
        };
        // 6. stable place pinned by a P-invariant but coupled: p0+p1 = 1 and
        //    a separate constant place p2 = 2 with a self-loop transition
        //    (zero net effect) ⇒ p2 stable ⇒ TRUE.
        let selfloop = PetriNet {
            name: Some("selfloop".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t01", vec![(0, 1)], vec![(1, 1)]),
                trans("t10", vec![(1, 1)], vec![(0, 1)]),
                // self-loop on p2: consumes and produces one ⇒ p2 constant.
                trans("loop", vec![(2, 1)], vec![(2, 1)]),
            ],
            initial_marking: vec![1, 0, 2],
        };

        for net in [&swap, &one_stable, &sink, &frozen, &ring, &selfloop] {
            let (expected_verdict, expected_mask) = brute_stable_marking(net);
            let expected = if expected_verdict {
                Verdict::True
            } else {
                Verdict::False
            };
            let got = try_dd_stable_marking(net, None);
            assert_eq!(
                got,
                Some(expected),
                "net {:?}: DD verdict {:?} != brute-force {:?} (per-place stable mask {:?})",
                net.name,
                got,
                expected,
                expected_mask,
            );
        }
    }

    #[test]
    fn stable_marking_unbounded_declines() {
        // Source transition ⇒ unbounded ⇒ build_sound_dd_spec None ⇒ decline.
        let net = PetriNet {
            name: Some("source".into()),
            places: vec![place("p0")],
            transitions: vec![trans("gen", vec![], vec![(0, 1)])],
            initial_marking: vec![0],
        };
        assert_eq!(try_dd_stable_marking(&net, None), None);
    }

    #[test]
    fn stable_marking_expired_deadline_declines() {
        let net = swap_net();
        let deadline = Some(Instant::now() + DD_FALLBACK_RESERVE);
        assert_eq!(try_dd_stable_marking(&net, deadline), None);
    }

    #[test]
    fn dd_budget_preserves_reserve_under_deadline() {
        let now = Instant::now();
        assert_eq!(dd_budget(None, now), Some(DD_NO_DEADLINE_BUDGET));
        // Below the ceiling, DD gets `remaining − reserve` (a fair slice that
        // still leaves the BFS reserve), not a hard 5s slice.
        assert_eq!(
            dd_budget(
                Some(now + DD_FALLBACK_RESERVE + Duration::from_secs(30)),
                now
            ),
            Some(Duration::from_secs(30)),
        );
        // A long (e.g. 3600s competition) deadline is clamped to the absolute
        // ceiling so the BFS+symmetry workhorse keeps the bulk of the budget.
        assert_eq!(
            dd_budget(
                Some(now + DD_FALLBACK_RESERVE + Duration::from_secs(600)),
                now
            ),
            Some(DD_MAX_BUDGET),
        );
        assert_eq!(
            dd_budget(
                Some(now + DD_FALLBACK_RESERVE + DD_MAX_BUDGET + Duration::from_secs(100)),
                now,
            ),
            Some(DD_MAX_BUDGET),
        );
        assert_eq!(
            dd_budget(
                Some(now + DD_FALLBACK_RESERVE + Duration::from_millis(250)),
                now
            ),
            Some(Duration::from_millis(250)),
        );
        assert_eq!(dd_budget(Some(now + DD_FALLBACK_RESERVE), now), None);
        // `now - Duration` is an intentional past Instant; checked_sub would change
        // the value type (Option<Instant>) and the expected None result.
        #[allow(clippy::unchecked_time_subtraction)]
        {
            assert_eq!(dd_budget(Some(now - Duration::from_millis(1)), now), None);
        }
    }

    #[test]
    fn one_safe_pt_swap_net_is_true() {
        let net = swap_net();
        assert!(brute_one_safe(&net));
        assert_eq!(try_dd_one_safe_pt(&net, None), Some(Verdict::True));
    }

    #[test]
    fn one_safe_pt_two_token_place_is_false() {
        // p0 --t--> p1 (x2). Reachable {(1,0),(0,2)}: p1 hits 2 ⇒ not 1-safe.
        let net = PetriNet {
            name: Some("doubler".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t", vec![(0, 1)], vec![(1, 2)])],
            initial_marking: vec![1, 0],
        };
        assert!(!brute_one_safe(&net), "cross-check: net is not 1-safe");
        assert_eq!(try_dd_one_safe_pt(&net, None), Some(Verdict::False));
    }

    #[test]
    fn one_safe_colored_group_sum_unsafe_not_wrong_true() {
        // Two places p0,p1 each individually 1-safe, but they form one
        // colored safety unit whose SUM reaches 2 (a token can sit in each
        // simultaneously). Individual-place max is 1 (would wrongly say
        // TRUE), but the group-sum max is 2 ⇒ colored OneSafe is FALSE.
        //
        // gen: p2(1) -> p0 + p1 puts a token in each. p2 starts at 1.
        // Reachable: {(0,0,1),(1,1,0)}. Individual max = 1; group {0,1}
        // sum max = 2.
        let net = PetriNet {
            name: Some("group-unsafe".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("gen", vec![(2, 1)], vec![(0, 1), (1, 1)])],
            initial_marking: vec![0, 0, 1],
        };
        // Individual-place metric would say TRUE (max_token_in_place == 1).
        let pt = try_dd_one_safe_pt(&net, None);
        assert_eq!(
            pt,
            Some(Verdict::True),
            "individual-place fast-path sees max 1 (this is why it must NOT be used for colored)"
        );
        // Group-sum path must catch the violation.
        let safety_units = vec![vec![0usize, 1usize], vec![2usize]];
        assert_eq!(
            try_dd_one_safe_colored(&net, &safety_units, None),
            Some(Verdict::False),
            "colored group-sum OneSafe must be FALSE (unit {{p0,p1}} sum reaches 2)"
        );
    }

    #[test]
    fn one_safe_colored_group_sum_safe_is_true() {
        // Swap net as one colored unit {p0,p1}: sum is conserved at 1 ⇒
        // group-sum 1-safe.
        let net = swap_net();
        let safety_units = vec![vec![0usize, 1usize]];
        assert_eq!(
            try_dd_one_safe_colored(&net, &safety_units, None),
            Some(Verdict::True),
        );
    }

    #[test]
    fn quasi_liveness_all_fire_is_true() {
        let net = swap_net();
        assert!(brute_quasi_live(&net));
        assert_eq!(try_dd_quasi_liveness(&net, &[], None), Some(Verdict::True));
    }

    #[test]
    fn quasi_liveness_never_fireable_transition_is_false() {
        // t_dead requires a token in p2 which never has one ⇒ never fires.
        let net = PetriNet {
            name: Some("dead-trans".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t01", vec![(0, 1)], vec![(1, 1)]),
                trans("t10", vec![(1, 1)], vec![(0, 1)]),
                trans("t_dead", vec![(2, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![1, 0, 0],
        };
        assert!(!brute_quasi_live(&net), "cross-check: t_dead never fires");
        assert_eq!(try_dd_quasi_liveness(&net, &[], None), Some(Verdict::False));
    }

    #[test]
    fn quasi_liveness_colored_group_quasi_live_via_one_binding() {
        // Colored transition with two bindings: t_a (fires) and t_dead
        // (never fires). As singletons the net is NOT quasi-live, but as a
        // single colored group {t_a, t_dead} it IS quasi-live (t_a fires).
        let net = PetriNet {
            name: Some("colored-group".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t_a", vec![(0, 1)], vec![(1, 1)]),
                trans("t_dead", vec![(2, 1)], vec![(0, 1)]),
                trans("t_b", vec![(1, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![1, 0, 0],
        };
        // Without grouping: t_dead never fires ⇒ FALSE.
        assert_eq!(try_dd_quasi_liveness(&net, &[], None), Some(Verdict::False));
        // Group {t_a (0), t_dead (1)} together, t_b (2) singleton: every
        // group has a fireable member ⇒ TRUE.
        let groups = vec![vec![0usize, 1usize]];
        assert_eq!(
            try_dd_quasi_liveness(&net, &groups, None),
            Some(Verdict::True),
        );
    }

    /// DD L4-Liveness verdict ⟺ exhaustive BFS L4-liveness reference on a
    /// battery of nets: live cycles, non-live dead transitions, deadlocks,
    /// lazy self-loops, and a higher-token DD-eligible net. 0 disagreements.
    #[test]
    fn liveness_dd_matches_brute_force_battery() {
        // 1. swap net: p0<->p1 cycle, both transitions always eventually
        //    fireable ⇒ LIVE.
        let swap = swap_net();
        // 2. dead transition: t_dead needs p2 which never has a token ⇒ NOT
        //    live (t_dead never fireable from any marking).
        let dead = PetriNet {
            name: Some("dead-trans".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t01", vec![(0, 1)], vec![(1, 1)]),
                trans("t10", vec![(1, 1)], vec![(0, 1)]),
                trans("t_dead", vec![(2, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![1, 0, 0],
        };
        // 3. deadlocking sink: p0 -> p1 then nothing. (0,1) is dead ⇒ t is NOT
        //    eventually-fireable from (0,1) ⇒ NOT live.
        let sink = PetriNet {
            name: Some("sink".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t", vec![(0, 1)], vec![(1, 1)])],
            initial_marking: vec![1, 0],
        };
        // 4. no transitions ⇒ vacuously LIVE.
        let frozen = PetriNet {
            name: Some("frozen".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![],
            initial_marking: vec![3, 7],
        };
        // 5. self-loop pair: both transitions perpetually enabled ⇒ LIVE.
        let lazy = PetriNet {
            name: Some("lazy".into()),
            places: vec![place("p0")],
            transitions: vec![
                trans("t_cycle", vec![(0, 1)], vec![(0, 1)]),
                trans("t_lazy", vec![(0, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![1],
        };
        // 6. one-shot then cycle: t_init fires once moving the token out of
        //    p0; after that p0 is never re-marked, so t_init is NOT
        //    eventually-fireable from the post-init markings ⇒ NOT live.
        let one_shot = PetriNet {
            name: Some("one-shot".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t_init", vec![(0, 1)], vec![(1, 1)]),
                trans("t12", vec![(1, 1)], vec![(2, 1)]),
                trans("t21", vec![(2, 1)], vec![(1, 1)]),
            ],
            initial_marking: vec![1, 0, 0],
        };
        // 7. higher-token ring: 4 tokens circulating a 3-place ring. Every
        //    transition is eventually-fireable from every reachable marking
        //    ⇒ LIVE. Exercises the binary-band encoding + a non-trivial
        //    reachable set (the DD-eligible profile).
        let ring = PetriNet {
            name: Some("ring4".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t0", vec![(0, 1)], vec![(1, 1)]),
                trans("t1", vec![(1, 1)], vec![(2, 1)]),
                trans("t2", vec![(2, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![4, 0, 0],
        };

        for net in [&swap, &dead, &sink, &frozen, &lazy, &one_shot, &ring] {
            let expected = if brute_live(net) {
                Verdict::True
            } else {
                Verdict::False
            };
            let got = try_dd_liveness(net, &[], None);
            assert_eq!(
                got,
                Some(expected),
                "net {:?}: DD liveness verdict {:?} != brute-force {:?}",
                net.name,
                got,
                expected,
            );
        }
    }

    #[test]
    fn liveness_colored_group_one_live_binding_is_live() {
        // Colored transition unfolded into two bindings: t0 (live) and t1
        // (dead — its input place p_dead is never marked). Both move a token
        // INTO the live cycle so no place ever exceeds its bound (keeping the
        // net DD-eligible; a self-loop on a 0-bound place would trip the
        // sound-spec output-bound gate and make the lane DECLINE instead).
        //
        // p_live = 1, p_dead = 0, p_sink bound 1.
        //   t0: p_live --> p_sink         (live: p_live is marked)
        //   t1: p_dead --> p_sink         (dead: p_dead never marked)
        //   t2: p_sink --> p_live         (live cycle back)
        // As P/T singletons t1 is dead ⇒ NOT live. As colored group {t0,t1}
        // the colored transition fires via t0 ⇒ LIVE (and t2 singleton live).
        let net = PetriNet {
            name: Some("colored-mixed".into()),
            places: vec![place("p_live"), place("p_dead"), place("p_sink")],
            transitions: vec![
                trans("t0", vec![(0, 1)], vec![(2, 1)]),
                trans("t1", vec![(1, 1)], vec![(2, 1)]),
                trans("t2", vec![(2, 1)], vec![(0, 1)]),
            ],
            initial_marking: vec![1, 0, 0],
        };
        // Cross-check the brute-force reference agrees t1 makes P/T non-live.
        assert!(!brute_live(&net), "cross-check: t1 (p_dead) never fires");
        // P/T (no grouping): t1 is dead ⇒ NOT live.
        assert_eq!(try_dd_liveness(&net, &[], None), Some(Verdict::False));
        // Colored group {t0, t1}, t2 singleton: every group has a live member
        // ⇒ LIVE.
        let groups = vec![vec![0usize, 1usize]];
        assert_eq!(try_dd_liveness(&net, &groups, None), Some(Verdict::True));
    }

    #[test]
    fn liveness_empty_net_is_vacuously_live() {
        let net = PetriNet {
            name: Some("empty".into()),
            places: vec![place("p0")],
            transitions: vec![],
            initial_marking: vec![0],
        };
        assert_eq!(try_dd_liveness(&net, &[], None), Some(Verdict::True));
    }

    #[test]
    fn liveness_unbounded_declines() {
        // Source transition ⇒ unbounded ⇒ build_sound_dd_spec None ⇒ decline.
        let net = PetriNet {
            name: Some("source".into()),
            places: vec![place("p0")],
            transitions: vec![trans("gen", vec![], vec![(0, 1)])],
            initial_marking: vec![0],
        };
        assert_eq!(try_dd_liveness(&net, &[], None), None);
    }

    #[test]
    fn liveness_expired_deadline_declines() {
        let net = swap_net();
        let deadline = Some(Instant::now() + DD_FALLBACK_RESERVE);
        assert_eq!(try_dd_liveness(&net, &[], deadline), None);
    }

    #[test]
    fn deadlock_present_is_true() {
        // p0 --t--> p1, then nothing. Reachable {(1,0),(0,1)}; (0,1) is a
        // deadlock (t needs a token in p0).
        let net = PetriNet {
            name: Some("sink".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t", vec![(0, 1)], vec![(1, 1)])],
            initial_marking: vec![1, 0],
        };
        assert!(brute_deadlock_exists(&net), "cross-check: (0,1) deadlocks");
        assert_eq!(try_dd_deadlock(&net, None), Some(Verdict::True));
    }

    #[test]
    fn deadlock_absent_is_false() {
        // Swap net cycles forever — no reachable marking is dead.
        let net = swap_net();
        assert!(!brute_deadlock_exists(&net), "cross-check: swap never dead");
        assert_eq!(try_dd_deadlock(&net, None), Some(Verdict::False));
    }

    #[test]
    fn unbounded_net_declines_all_fast_paths() {
        // Source transition with no input ⇒ unbounded ⇒ build_sound_dd_spec
        // returns None ⇒ every DD fast-path declines (None).
        let net = PetriNet {
            name: Some("source".into()),
            places: vec![place("p0")],
            transitions: vec![trans("gen", vec![], vec![(0, 1)])],
            initial_marking: vec![0],
        };
        assert_eq!(try_dd_one_safe_pt(&net, None), None);
        assert_eq!(try_dd_one_safe_colored(&net, &[vec![0]], None), None);
        assert_eq!(try_dd_quasi_liveness(&net, &[], None), None);
        assert_eq!(try_dd_deadlock(&net, None), None);
    }

    #[test]
    fn expired_deadline_declines() {
        // Only the BFS reserve remains ⇒ DD is skipped (None) even on an
        // otherwise-eligible net.
        let net = swap_net();
        let deadline = Some(Instant::now() + DD_FALLBACK_RESERVE);
        assert_eq!(try_dd_one_safe_pt(&net, deadline), None);
        assert_eq!(try_dd_quasi_liveness(&net, &[], deadline), None);
        assert_eq!(try_dd_deadlock(&net, deadline), None);
    }
}
