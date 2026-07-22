// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Reachability seeding via symbolic state-equation dispatch.
//!
//! Mirrors the shape of `examinations/reachability_pdr.rs::run_pdr_seeding`
//! but routes through the Esparza–Melzer CHC encoder in
//! [`super::chc_dispatch`]. Both seeders are env-gated for the same
//! soundness reason: any historical wrong-answer risk on fireability
//! predicates would compound the existing PDR risk profile, and
//! enabling them simultaneously without a measurement run could shift
//! the MCC verdict distribution silently.
//!
//! # Soundness contract
//!
//! - Fireability predicates ARE handled: the encoder lowers `IsFireable(T)`
//!   to `⋁_{t∈T} ⋀_{p∈•t} (w(p,t) ≤ M[p])` (see
//!   `state_equation::encode_predicate_expr`). The historical skip was a
//!   conservative carve-out; it is now safe to drop because every
//!   witness-derived TRUE/FALSE is independently replay-validated on the
//!   concrete net (below), so a spurious or mis-decoded counterexample can
//!   only ever leave the tracker pending, never emit a wrong verdict.
//! - `Safe`   on `AG(φ)` → `φ`-violation infeasible (state equation is an
//!   over-approximation, so infeasible ⇒ genuinely unreachable) → `TRUE`.
//!   No replay needed — Safe is sound by over-approximation.
//! - `Unsafe` on `AG(φ)` → candidate witness violating `φ`; emit `FALSE`
//!   ONLY if [`chc_witness_reaches_violation`] replays it to a genuinely
//!   reachable `¬φ` marking, else leave pending.
//! - `Safe`   on `EF(φ)` is `¬φ` safety → `φ` unreachable → `FALSE` (sound by
//!   over-approximation, no replay).
//! - `Unsafe` on `EF(φ)` → candidate witness reaching `φ`; emit `TRUE` ONLY
//!   if the replay validates a genuinely reachable `φ` marking.
//! - `Unknown`  is propagated as no-op — the tracker remains pending.

use std::time::{Duration, Instant};

use super::chc_dispatch::{
    symbolic_state_equation_check, SymbolicConfig, SymbolicVerdict, SymbolicWitnessStep,
};
use super::int_state_equation::{integer_state_equation_safe, IntStateEquationVerdict};
use crate::examinations::reachability::{
    resolve_tracker, PropertyTracker, ReachabilityResolutionSource,
};
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::{eval_predicate, ResolvedPredicate};

/// Hard cap for one symbolic dispatch. Raised from the
/// original 5 s after the dispatch wiring landed: empirically the
/// AdaptivePortfolio (IC3/PDR + BMC + PDKind + k-induction) needs more
/// than a few seconds to discover the inductive invariant on most
/// Reachability* timeouts. CHC is sound regardless of budget — extra
/// time only converts "Unknown" results into proofs/counterexamples, it
/// can never invent a wrong verdict. The seeder still honours
/// `config.deadline()`, so this caps individual trackers without
/// starving the rest of the pipeline.
const SYMBOLIC_SEED_TIMEOUT: Duration = Duration::from_secs(20);

/// Environment variable for the symbolic state-equation reachability
/// seeder. Now defaults to ON after differential testing (8/8 fixtures,
/// 23/23 mcc_benchmarks) confirmed the AG/EF mapping agrees with BFS
/// ground truth on every reachable case. Set
/// `TY_MCC_ENABLE_REACHABILITY_SYMBOLIC=0` (or `false`/`no`/`off`) to
/// disable for clean-baseline benchmarking; any other value — or
/// leaving the variable unset — enables the seeder.
pub(crate) const ENABLE_REACHABILITY_SYMBOLIC_ENV: &str = "TY_MCC_ENABLE_REACHABILITY_SYMBOLIC";

/// Kill-switch for the integer firing-count state-equation pre-check.
///
/// The integer lane ([`integer_state_equation_safe`]) runs a single QF_LIA
/// feasibility query of `m = m0 + C·x ∧ ¬φ` over the *integers* before the CHC
/// portfolio. It can prove `AG φ` (Safe) on nets where the LP relaxation is
/// spuriously feasible and the CHC portfolio times out — and it NEVER emits a
/// wrong verdict (UNSAT is a sound proof; SAT/Unknown fall through to CHC).
/// Defaults ON; set to `0`/`false`/`no`/`off` to disable for A/B comparison.
pub(crate) const ENABLE_INTEGER_STATE_EQUATION_ENV: &str = "TY_MCC_ENABLE_INTEGER_STATE_EQUATION";

/// Fraction of a tracker's symbolic budget reserved for the integer pre-check.
///
/// The integer state equation is a *single* `check_sat`, so it is cheap relative
/// to the multi-engine CHC portfolio. We hand it one quarter of the slot's
/// budget; if it declines (SAT/Unknown), the remaining three quarters fund the
/// existing CHC portfolio dispatch, so the integer lane can only ADD proofs —
/// never starve the fallback. The divisor is applied with a small floor so a
/// near-expired slot still attempts at least a brief integer query.
const INTEGER_PRECHECK_BUDGET_DIVISOR: u32 = 4;

fn env_flag_enabled(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(default)
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

fn fair_share_duration(remaining: Duration, pending_count: usize) -> Duration {
    let divisor = pending_count.clamp(1, u32::MAX as usize) as u32;
    remaining / divisor
}

fn symbolic_tracker_timeout_at(
    global_deadline: Option<Instant>,
    pending_count: usize,
    now: Instant,
) -> Duration {
    global_deadline
        .map(|limit| {
            SYMBOLIC_SEED_TIMEOUT.min(fair_share_duration(
                limit.saturating_duration_since(now),
                pending_count,
            ))
        })
        .unwrap_or(SYMBOLIC_SEED_TIMEOUT)
}

/// Symbolic state-equation seeder for reachability formulas.
///
/// Walks unresolved trackers, builds the CHC system via the
/// state-equation encoder, and dispatches to AdaptivePortfolio.
/// Resolved trackers are stamped with
/// [`ReachabilityResolutionSource::Pdr`] (no dedicated variant — the
/// MCC technique line reports CHC-flavour resolutions uniformly).
pub(crate) fn run_symbolic_state_equation_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
) {
    if !env_flag_enabled(ENABLE_REACHABILITY_SYMBOLIC_ENV, true) {
        return;
    }

    // Fireability predicates are now admitted: the encoder lowers them and
    // every witness-derived verdict is replay-validated below, so the old
    // `!predicate_contains_fireability` carve-out is no longer needed.
    let eligible_slots: Vec<usize> = trackers
        .iter()
        .enumerate()
        .filter_map(|(slot, tracker)| tracker.verdict.is_none().then_some(slot))
        .collect();

    let eligible_count = eligible_slots.len();
    for (offset, slot) in eligible_slots.into_iter().enumerate() {
        if trackers[slot].verdict.is_some() {
            continue;
        }

        let pending_count = eligible_count.saturating_sub(offset).max(1);
        let timeout = symbolic_tracker_timeout_at(deadline, pending_count, Instant::now());
        if timeout.is_zero() {
            break;
        }

        let safety_property = match trackers[slot].quantifier {
            PathQuantifier::AG => trackers[slot].predicate.clone(),
            PathQuantifier::EF => {
                ResolvedPredicate::Not(Box::new(trackers[slot].predicate.clone()))
            }
        };

        // ── Integer firing-count state-equation pre-check ────────────────
        //
        // Before the CHC portfolio, attempt the one-shot INTEGER feasibility
        // query `m = m0 + C·x ∧ ¬φ ∧ x≥0 ∧ m≥0 ∧ traps` over ℤ. INFEASIBLE ⇒
        // no reachable marking violates the safety property ⇒ `Safe`, SOUND by
        // the over-approximation theorem (every reachable marking solves the
        // integer state equation) with NO LP relaxation gap. This proves `Safe`
        // on nets the rational LP and the CHC portfolio leave Unknown. SAT
        // (a candidate in the over-approximation) and Unknown both DECLINE —
        // we fall through to the unchanged CHC portfolio below, so the lane is
        // purely additive and can never emit a wrong verdict.
        if env_flag_enabled(ENABLE_INTEGER_STATE_EQUATION_ENV, true) {
            let int_timeout = (timeout / INTEGER_PRECHECK_BUDGET_DIVISOR)
                .max(Duration::from_millis(50))
                .min(timeout);
            if matches!(
                integer_state_equation_safe(net, &safety_property, int_timeout),
                IntStateEquationVerdict::Safe
            ) {
                let tracker = &mut trackers[slot];
                match tracker.quantifier {
                    PathQuantifier::AG => {
                        resolve_tracker(tracker, true, ReachabilityResolutionSource::Pdr, None);
                    }
                    PathQuantifier::EF => {
                        resolve_tracker(tracker, false, ReachabilityResolutionSource::Pdr, None);
                    }
                }
                continue;
            }
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                break;
            }
        }

        let config = SymbolicConfig {
            time_budget: timeout,
            ..SymbolicConfig::default()
        };
        let verdict = symbolic_state_equation_check(net, &safety_property, &config);
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            break;
        }

        let tracker = &mut trackers[slot];
        match (tracker.quantifier, verdict) {
            // Safe is sound by over-approximation (the state equation contains
            // the exact reachable set, so an infeasible bad state is genuinely
            // unreachable) — no replay needed.
            (PathQuantifier::AG, SymbolicVerdict::Safe) => {
                resolve_tracker(tracker, true, ReachabilityResolutionSource::Pdr, None);
            }
            (PathQuantifier::EF, SymbolicVerdict::Safe) => {
                resolve_tracker(tracker, false, ReachabilityResolutionSource::Pdr, None);
            }
            // Unsafe carries a CANDIDATE counterexample. The CHC encoding is the
            // exact transition relation, so a genuine derivation IS a real path
            // — but we never trust the decoded witness blindly: replay it on the
            // concrete net and emit the definite verdict ONLY if it reaches a
            // genuinely reachable safety violation. A spurious / mis-decoded
            // witness leaves the tracker pending (CC), never a wrong verdict.
            (PathQuantifier::AG, SymbolicVerdict::Unsafe { witness }) => {
                if chc_witness_reaches_violation(net, &witness, &safety_property) {
                    resolve_tracker(tracker, false, ReachabilityResolutionSource::Pdr, None);
                }
            }
            (PathQuantifier::EF, SymbolicVerdict::Unsafe { witness }) => {
                if chc_witness_reaches_violation(net, &witness, &safety_property) {
                    resolve_tracker(tracker, true, ReachabilityResolutionSource::Pdr, None);
                }
            }
            (_, SymbolicVerdict::Unknown { .. }) => {}
        }
    }
}

/// Replay a decoded CHC counterexample on the concrete net and confirm it
/// witnesses a genuine safety violation.
///
/// `witness` is the marking sequence decoded from the ay-chc counterexample
/// (`m_0, m_1, …, m_k`); `safety_property` is the invariant the CHC query
/// asserted (`φ` for `AG φ`, `¬φ` for `EF φ`). Returns `true` iff:
///
/// 1. every step's marking is representable (length matches, no negative
///    token counts — a negative is a decode artifact, never a real marking);
/// 2. the path starts at the net's initial marking;
/// 3. every consecutive pair is either a stutter (`mᵢ == mᵢ₊₁`) or a real
///    transition firing (`∃ t: enabled(mᵢ, t) ∧ fire(mᵢ, t) == mᵢ₊₁`); and
/// 4. some marking along this *proven-reachable* path violates
///    `safety_property` (i.e. `eval_predicate(safety, m) == false`).
///
/// Because every marking it accepts is reached by a validated firing chain
/// from the initial marking, a `true` result is an independent reachability
/// certificate: the verdict cannot be wrong even if the encoder, the solver,
/// or the witness decoder had a bug. A `false` result is fail-closed (the
/// caller leaves the tracker pending), so this can only ever *withhold* a
/// verdict, never invent one.
fn chc_witness_reaches_violation(
    net: &PetriNet,
    witness: &[SymbolicWitnessStep],
    safety_property: &ResolvedPredicate,
) -> bool {
    let np = net.num_places();
    // Decode + validate each step's marking (reject negatives / wrong arity).
    let mut path: Vec<Vec<u64>> = Vec::with_capacity(witness.len());
    for step in witness {
        if step.marking.len() != np {
            return false;
        }
        let mut m = Vec::with_capacity(np);
        for &v in &step.marking {
            if v < 0 {
                return false;
            }
            m.push(v as u64);
        }
        path.push(m);
    }
    if path.is_empty() || path[0] != net.initial_marking {
        return false;
    }
    // Each consecutive pair must be a real firing (or an explicit stutter).
    for pair in path.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if a == b {
            continue;
        }
        let realizable = (0..net.num_transitions()).any(|t| {
            let ti = TransitionIdx(t as u32);
            net.is_enabled(a, ti) && net.fire(a, ti).is_ok_and(|succ| &succ == b)
        });
        if !realizable {
            return false;
        }
    }
    // Some validated (hence genuinely reachable) marking must violate safety.
    path.iter()
        .any(|m| !eval_predicate(safety_property, m, net))
}

#[cfg(test)]
#[path = "reachability_seed_tests.rs"]
mod tests;
