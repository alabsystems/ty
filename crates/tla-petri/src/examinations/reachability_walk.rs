// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Random walk witness search for reachability properties.
//!
//! Lightweight under-approximation: fires random enabled transitions from the
//! initial marking to find witnesses for EF(φ) properties. Cannot prove
//! EF(φ)=FALSE or AG(φ)=TRUE — unresolved trackers remain `verdict: None`
//! and fall through to BFS.
//!
//! Sound because: any marking reached by firing enabled transitions from the
//! initial marking is by definition reachable, so a witness found here is valid.

use std::time::Instant;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tla_mc_core::{
    random_walk_witness, RandomWalkBudget, RandomWalkPoll, RandomWalkStep, RandomWalkStepper,
};

use crate::examinations::reachability_witness::{
    apply_validated_witnesses, candidates_for_marking, WitnessSeedSource, WitnessValidationContext,
};
use crate::petri_net::{PetriNet, TransitionIdx};

use super::reachability::PropertyTracker;

/// Walk-index cadence at which the bespoke Petri walks polled the deadline /
/// early-exit gate. The shared engine drives the poll at this exact cadence so
/// the deadline-poll behavior is byte-identical to the pre-extraction loops.
const WALK_POLL_INTERVAL: u32 = 100;

/// Build the shared-engine budget for a Petri walk, preserving the historical
/// poll cadence so trajectories match the bespoke loops exactly.
fn walk_budget(walks: u32, max_steps: u32) -> RandomWalkBudget {
    RandomWalkBudget::new(walks, max_steps).with_poll_interval(WALK_POLL_INTERVAL)
}

/// Default number of independent random walks.
const DEFAULT_WALKS: u32 = 1000;

/// Default maximum steps per walk before restarting.
const DEFAULT_MAX_STEPS: u32 = 10_000;

/// Run random walks to find witnesses for unresolved EF(φ) properties.
///
/// For each unresolved `PropertyTracker` where `quantifier == EF` and
/// `verdict.is_none()`, attempt to find a marking satisfying the predicate
/// via random simulation on the original (unreduced) net.
///
/// Also seeds AG(φ)=FALSE by finding counterexamples (markings where ¬φ).
pub(crate) fn run_random_walk_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation: &WitnessValidationContext<'_>,
    deadline: Option<Instant>,
) {
    run_random_walk_seeding_params(
        net,
        trackers,
        validation,
        deadline,
        DEFAULT_WALKS,
        DEFAULT_MAX_STEPS,
    );
}

/// Parameterized version for testing.
pub(crate) fn run_random_walk_seeding_params(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation: &WitnessValidationContext<'_>,
    deadline: Option<Instant>,
    walks: u32,
    max_steps: u32,
) {
    // The initial marking is reachable even if the net has no enabled moves.
    validate_marking(trackers, &net.initial_marking, net, &[], validation);
    if net.num_transitions() == 0 || walks == 0 || all_walkable_resolved(trackers) {
        return;
    }

    let mut rng = SmallRng::from_entropy();
    let mut marking = vec![0u64; net.num_places()];
    let mut enabled = Vec::with_capacity(net.num_transitions());
    let mut path = Vec::with_capacity(max_steps as usize);

    for walk_id in 0..walks {
        // Check deadline and early-exit conditions every 100 walks.
        if walk_id % 100 == 0 {
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }
            if all_walkable_resolved(trackers) {
                break;
            }
        }

        // Start from initial marking.
        marking.copy_from_slice(&net.initial_marking);
        path.clear();

        for _step in 0..max_steps {
            // Collect enabled transitions.
            enabled.clear();
            for t in 0..net.num_transitions() {
                let tidx = TransitionIdx(t as u32);
                if net.is_enabled(&marking, tidx) {
                    enabled.push(tidx);
                }
            }

            // Deadlock: no enabled transitions, restart walk.
            if enabled.is_empty() {
                break;
            }

            // Fire a random enabled transition in place.
            let chosen = enabled[rng.gen_range(0..enabled.len())];
            // #22: token-count overflow means this walk reached a
            // non-representable marking — abandon it (the next walk reinitialises
            // `marking`). This seeder only emits TRUE witnesses, so dropping a
            // walk is sound.
            if net.apply_delta(&mut marking, chosen).is_err() {
                break;
            }
            path.push(chosen);
            validate_marking(trackers, &marking, net, &path, validation);
            if all_walkable_resolved(trackers) {
                break;
            }
        }
    }
}

/// Random-walk witness search for `ReachabilityDeadlock`.
///
/// Returns `true` as soon as a walk reaches a NON-initial reachable marking in
/// which NO transition is enabled — a genuine reachable deadlock, which proves
/// `ReachabilityDeadlock = TRUE`. Returns `false` if no such marking is found
/// within the walk/step budget or before `deadline`.
///
/// Soundness: this is a strict under-approximation that emits ONLY positive
/// witnesses. Every marking it inspects is reached by firing ONLY enabled
/// transitions from `net.initial_marking`, so it is reachable by construction,
/// and "deadlock" is verified directly via [`PetriNet::is_enabled`] over EVERY
/// transition. It NEVER returns a universal/False verdict: a miss or a timeout
/// returns `false`, which the caller treats as "fall through, unresolved".
///
/// The initial marking itself is deliberately NOT reported as a deadlock here:
/// a net that is dead in its initial marking is the trivial case already
/// settled by the cheaper structural/BMC lanes, and excluding it keeps this
/// lane focused on the reachable-but-non-trivial deadlocks (PolyORBLF /
/// ResAllocation) that motivated it. A net with zero transitions therefore
/// reports `false` here.
pub(crate) fn run_random_walk_deadlock(net: &PetriNet, deadline: Option<Instant>) -> bool {
    run_random_walk_deadlock_params(net, deadline, DEFAULT_WALKS, DEFAULT_MAX_STEPS)
}

/// Parameterized version of [`run_random_walk_deadlock`] for testing.
pub(crate) fn run_random_walk_deadlock_params(
    net: &PetriNet,
    deadline: Option<Instant>,
    walks: u32,
    max_steps: u32,
) -> bool {
    if net.num_transitions() == 0 || walks == 0 || max_steps == 0 {
        return false;
    }

    // The shared-engine control loop drives the walk/step budget, deadline
    // polling cadence, and restart-from-initial; the Petri-specific work
    // (enabled-set enumeration, RNG-driven single-transition selection,
    // `apply_delta`, deadlock detection) stays in `DeadlockWalk` so the
    // trajectory is byte-identical to the pre-extraction loop.
    let mut stepper = DeadlockWalk {
        net,
        deadline,
        rng: SmallRng::from_entropy(),
        marking: vec![0u64; net.num_places()],
        enabled: Vec::with_capacity(net.num_transitions()),
        step_in_walk: 0,
        found: false,
    };
    random_walk_witness(&mut stepper, walk_budget(walks, max_steps));
    stepper.found
}

/// Petri adapter for [`run_random_walk_deadlock_params`]: advances the walk by
/// one random enabled transition and reports a reachable, non-initial dead
/// marking as a deadlock witness.
struct DeadlockWalk<'a> {
    net: &'a PetriNet,
    deadline: Option<Instant>,
    rng: SmallRng,
    marking: Vec<u64>,
    enabled: Vec<TransitionIdx>,
    /// Steps fired in the current walk (mirrors the `step` loop counter): a
    /// dead marking only witnesses a deadlock when at least one step fired.
    step_in_walk: u32,
    /// Set once a reachable, non-initial dead marking is observed.
    found: bool,
}

impl RandomWalkStepper for DeadlockWalk<'_> {
    fn should_stop(&mut self) -> RandomWalkPoll {
        if let Some(dl) = self.deadline {
            if Instant::now() >= dl {
                return RandomWalkPoll::Stop;
            }
        }
        RandomWalkPoll::Continue
    }

    fn reset_to_initial(&mut self) {
        self.marking.copy_from_slice(&self.net.initial_marking);
        self.step_in_walk = 0;
    }

    fn step(&mut self) -> RandomWalkStep {
        // Collect enabled transitions at the current marking.
        self.enabled.clear();
        for t in 0..self.net.num_transitions() {
            let tidx = TransitionIdx(t as u32);
            if self.net.is_enabled(&self.marking, tidx) {
                self.enabled.push(tidx);
            }
        }

        if self.enabled.is_empty() {
            // Reached a marking with zero enabled transitions. `step > 0`
            // means at least one transition fired from the initial marking,
            // so this is a reachable, NON-initial dead marking: a genuine
            // deadlock witness. (`step == 0` is the trivial initial-dead
            // case, which we leave to the structural/BMC lanes.)
            if self.step_in_walk > 0 {
                self.found = true;
                return RandomWalkStep::Done;
            }
            // Initial marking is already dead — nothing to walk; abandon.
            return RandomWalkStep::Dead;
        }

        // Fire a uniformly-random enabled transition in place.
        let chosen = self.enabled[self.rng.gen_range(0..self.enabled.len())];
        // Token-count overflow means this walk reached a non-representable
        // marking — abandon it (the next walk reinitialises `marking`).
        // This lane only emits TRUE witnesses, so dropping a walk is sound.
        if self.net.apply_delta(&mut self.marking, chosen).is_err() {
            return RandomWalkStep::Abandon;
        }
        self.step_in_walk += 1;
        RandomWalkStep::Advanced
    }
}

/// Random-walk FALSE-witness search for `StableMarking`.
///
/// MCC `StableMarking` is TRUE iff SOME place is constant (equal to its initial
/// value) across ALL reachable markings, and FALSE iff NO place is constant.
/// Since the initial marking is itself reachable, a place `p` is constant iff
/// `marking[p] == net.initial_marking[p]` in every reachable marking. Therefore
/// observing ANY reachable marking with `marking[p] != net.initial_marking[p]`
/// PROVES that `p` is non-constant. If EVERY place is proven non-constant, then
/// no place is stable and `StableMarking = FALSE`.
///
/// Returns `Some(unstable_places)` — the per-place "proven non-constant" vector
/// (all `true`) — as soon as EVERY place has been observed to differ from its
/// initial value in a directly-reached reachable marking. Returns `None` if at
/// least one candidate-stable place survives the entire walk/step budget or the
/// `deadline` expires first.
///
/// `initial_unstable[p] == true` seeds place `p` as already-proven non-constant
/// (e.g. from the BMC/PDR phase): such places are excluded from the candidate
/// set and need no walk witness. `stable[p]` (the candidate-constant flag) is
/// initialised to `!initial_unstable[p]`.
///
/// Soundness: this is a strict under-approximation that emits ONLY the FALSE
/// witness. Every marking it inspects is reached by firing ONLY enabled
/// transitions from `net.initial_marking`, so it is reachable by construction,
/// and each place flagged non-constant was directly observed to differ from its
/// initial value in such a reachable marking. It NEVER claims a place is stable
/// / NEVER returns TRUE: a miss or a timeout returns `None`, which the caller
/// treats as "fall through, unresolved". The seed places in `initial_unstable`
/// are the caller's responsibility — this routine only ADDS witnesses, it never
/// trusts a seed to conclude FALSE on its own unless the walk also corroborates
/// every remaining candidate.
pub(crate) fn run_random_walk_stable_marking(
    net: &PetriNet,
    initial_unstable: &[bool],
    deadline: Option<Instant>,
) -> Option<Vec<bool>> {
    run_random_walk_stable_marking_params(
        net,
        initial_unstable,
        deadline,
        DEFAULT_WALKS,
        DEFAULT_MAX_STEPS,
    )
}

/// Parameterized version of [`run_random_walk_stable_marking`] for testing.
pub(crate) fn run_random_walk_stable_marking_params(
    net: &PetriNet,
    initial_unstable: &[bool],
    deadline: Option<Instant>,
    walks: u32,
    max_steps: u32,
) -> Option<Vec<bool>> {
    let num_places = net.num_places();
    if num_places == 0 {
        // No places to be (un)stable. Vacuously there is no place at all, so we
        // cannot witness "every place non-constant" — leave it to other lanes.
        return None;
    }

    // `stable[p] == true` means p is STILL a candidate constant place (not yet
    // proven non-constant). Seed from the caller's already-proven set.
    let mut stable: Vec<bool> = (0..num_places)
        .map(|p| !initial_unstable.get(p).copied().unwrap_or(false))
        .collect();
    let mut stable_count = stable.iter().filter(|&&s| s).count();

    // Every place is already proven non-constant by the seed alone ⇒ FALSE.
    // (The seed comes from a sound BMC/PDR phase; corroboration is unchanged.)
    if stable_count == 0 {
        return Some(vec![true; num_places]);
    }

    // The initial marking equals itself on every place, so it can never perturb
    // a candidate. Without transitions there is nothing to walk: no candidate
    // can be falsified, so we cannot prove FALSE.
    if net.num_transitions() == 0 || walks == 0 || max_steps == 0 {
        return None;
    }

    let mut rng = SmallRng::from_entropy();
    let mut marking = vec![0u64; num_places];
    let mut enabled = Vec::with_capacity(net.num_transitions());

    for walk_id in 0..walks {
        // Poll the deadline every 100 walks (matching the other walk loops).
        if walk_id % 100 == 0 {
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    return None;
                }
            }
        }

        // Restart each walk from the initial marking.
        marking.copy_from_slice(&net.initial_marking);

        for _step in 0..max_steps {
            // Collect enabled transitions at the current marking.
            enabled.clear();
            for t in 0..net.num_transitions() {
                let tidx = TransitionIdx(t as u32);
                if net.is_enabled(&marking, tidx) {
                    enabled.push(tidx);
                }
            }

            // Deadlock: no enabled transitions, restart the walk.
            if enabled.is_empty() {
                break;
            }

            // Fire a uniformly-random enabled transition in place.
            let chosen = enabled[rng.gen_range(0..enabled.len())];
            // Token-count overflow means this walk reached a non-representable
            // marking — abandon it (the next walk reinitialises `marking`).
            // This lane only emits FALSE witnesses corroborated per place, so
            // dropping a walk is sound.
            if net.apply_delta(&mut marking, chosen).is_err() {
                break;
            }

            // This is a reachable marking. Any place differing from its initial
            // value is PROVEN non-constant.
            for (p, slot) in stable.iter_mut().enumerate() {
                if *slot && marking[p] != net.initial_marking[p] {
                    *slot = false;
                    stable_count -= 1;
                    if stable_count == 0 {
                        // Every place has now been directly observed to differ
                        // from its initial value in a reachable marking ⇒ no
                        // place is constant ⇒ StableMarking = FALSE.
                        return Some(vec![true; num_places]);
                    }
                }
            }
        }
    }

    // A candidate-constant place survived the entire budget: inconclusive.
    None
}

/// Random-walk FALSE-witness search for `OneSafe`.
///
/// MCC `OneSafe` is TRUE iff every safety unit (a P/T place is a singleton unit;
/// a colored NUPN group is the set of its member places) holds a token SUM ≤ 1
/// in ALL reachable markings, and FALSE iff SOME reachable marking has a
/// safety-unit token SUM ≥ 2. Each `unit` in `safety_units` is a slice of place
/// indices whose token counts are summed; the per-place maximum is NOT used —
/// for a colored group it would over-count (the documented BridgeAndVehicles-COL
/// wrong-TRUE trap), so the GROUP SUM is the correct, parity-preserving quantity.
///
/// Returns `true` as soon as a walk reaches a reachable marking in which SOME
/// `unit`'s token sum is ≥ 2 — a directly-verified reachable safety-unit
/// overflow, which proves `OneSafe = FALSE`. Returns `false` if no such marking
/// is found within the walk/step budget or before `deadline`.
///
/// Soundness: this is a strict under-approximation that emits ONLY the FALSE
/// witness. Every marking it inspects is reached by firing ONLY enabled
/// transitions from `net.initial_marking`, so it is reachable by construction,
/// and the overflow is verified directly by summing the unit's observed token
/// counts. It NEVER returns TRUE / never claims 1-safety: a miss, an overflow,
/// or a timeout returns `false`, which the caller treats as "fall through,
/// unresolved". The trivial initial-marking overflow is handled by the caller's
/// fast-FALSE pre-check; this lane targets reachable-but-non-initial overflows
/// that the explicit BFS cannot reach within budget.
pub(crate) fn run_random_walk_one_safe(
    net: &PetriNet,
    safety_units: &[Vec<usize>],
    deadline: Option<Instant>,
) -> bool {
    run_random_walk_one_safe_params(
        net,
        safety_units,
        deadline,
        DEFAULT_WALKS,
        DEFAULT_MAX_STEPS,
    )
}

/// Parameterized version of [`run_random_walk_one_safe`] for testing.
pub(crate) fn run_random_walk_one_safe_params(
    net: &PetriNet,
    safety_units: &[Vec<usize>],
    deadline: Option<Instant>,
    walks: u32,
    max_steps: u32,
) -> bool {
    if net.num_transitions() == 0 || walks == 0 || max_steps == 0 || safety_units.is_empty() {
        return false;
    }

    let mut rng = SmallRng::from_entropy();
    let mut marking = vec![0u64; net.num_places()];
    let mut enabled = Vec::with_capacity(net.num_transitions());

    for walk_id in 0..walks {
        // Poll the deadline every 100 walks (matching the other walk loops).
        if walk_id % 100 == 0 {
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    return false;
                }
            }
        }

        // Restart each walk from the initial marking.
        marking.copy_from_slice(&net.initial_marking);

        for _step in 0..max_steps {
            // Collect enabled transitions at the current marking.
            enabled.clear();
            for t in 0..net.num_transitions() {
                let tidx = TransitionIdx(t as u32);
                if net.is_enabled(&marking, tidx) {
                    enabled.push(tidx);
                }
            }

            // Deadlock: no enabled transitions, restart the walk.
            if enabled.is_empty() {
                break;
            }

            // Fire a uniformly-random enabled transition in place.
            let chosen = enabled[rng.gen_range(0..enabled.len())];
            // Token-count overflow means this walk reached a non-representable
            // marking — abandon it (the next walk reinitialises `marking`).
            // This lane only emits the FALSE witness, so dropping a walk is
            // sound (we simply forgo a potential witness on that path).
            if net.apply_delta(&mut marking, chosen).is_err() {
                break;
            }

            // This is a reachable marking. If ANY safety unit's token SUM is
            // ≥ 2, this marking directly witnesses OneSafe = FALSE.
            for unit in safety_units {
                let sum: u64 = unit.iter().map(|&p| marking[p]).sum();
                if sum >= 2 {
                    return true;
                }
            }
        }
    }

    false
}

/// Random-walk ACHIEVABLE-MAXIMUM search for `UpperBounds`.
///
/// MCC `UpperBounds` asks, per query, for the MAXIMUM over all reachable markings
/// of the query's token SUM (a query is a place multiset — a place that appears
/// `k` times contributes `k · marking[place]`; here it is passed flattened so a
/// repeated place index simply appears `k` times in the slice and is summed with
/// that multiplicity, exactly mirroring `UpperBoundsObserver`). This lane returns,
/// per query, the maximum token sum it OBSERVES across the reachable markings it
/// visits.
///
/// Soundness: this is a strict under-approximation that produces ONLY a LOWER
/// bound. Every marking it inspects is reached by firing ONLY enabled transitions
/// from `net.initial_marking`, so it is reachable by construction; the observed
/// query sum is therefore a value the query genuinely attains in some reachable
/// marking, i.e. `observed[q] <= true_max[q]`. The caller raises each query's
/// achievable lower bound (`BoundTracker::max_bound`) to `max(max_bound, observed)`
/// and lets the existing `is_structurally_resolved()` pin the answer ONLY when the
/// lower bound meets the sound structural/LP UPPER bound (`max_bound >= effective_bound`
/// ⇒ lower == upper ⇒ the exact maximum). It NEVER lowers an upper bound and NEVER
/// reports the observed value as the answer on its own — a query whose observed max
/// stays below its upper bound stays unresolved and falls through unchanged. A
/// timeout or empty net simply returns the initial-marking sums (a reachable value),
/// so the result is always a valid lower bound.
///
/// `targets[q]`, when `Some(t)`, is the effective UPPER bound for query `q`: once
/// every query has observed a sum `>= target`, no further walking can improve any
/// pin, so the walk stops early. A `None` target means "no known ceiling" — that
/// query can never trigger the early stop (it just keeps tracking its observed max
/// for the budget). This is a pure performance gate; it never changes the values
/// returned, only when the search terminates.
pub(crate) fn run_random_walk_upper_bounds(
    net: &PetriNet,
    queries: &[Vec<usize>],
    targets: &[Option<u64>],
    deadline: Option<Instant>,
) -> Vec<u64> {
    run_random_walk_upper_bounds_params(
        net,
        queries,
        targets,
        deadline,
        DEFAULT_WALKS,
        DEFAULT_MAX_STEPS,
    )
}

/// Parameterized version of [`run_random_walk_upper_bounds`] for testing.
pub(crate) fn run_random_walk_upper_bounds_params(
    net: &PetriNet,
    queries: &[Vec<usize>],
    targets: &[Option<u64>],
    deadline: Option<Instant>,
    walks: u32,
    max_steps: u32,
) -> Vec<u64> {
    // Per-query observed maximum, seeded from the initial marking (always
    // reachable, hence always a valid lower bound even if no walk runs).
    let mut observed: Vec<u64> = queries
        .iter()
        .map(|query| query.iter().map(|&p| net.initial_marking[p]).sum())
        .collect();

    // `pinned[q]` once query `q` has observed its known upper-bound target:
    // further observation cannot improve the pin, so it stops contributing to
    // the early-exit gate. Queries with no target are never "pinned" and so
    // never let the walk stop early.
    let all_pinned = |observed: &[u64]| {
        targets
            .iter()
            .zip(observed.iter())
            .all(|(target, &obs)| target.is_some_and(|t| obs >= t))
    };

    if net.num_transitions() == 0 || walks == 0 || max_steps == 0 || queries.is_empty() {
        return observed;
    }
    if all_pinned(&observed) {
        return observed;
    }

    let mut rng = SmallRng::from_entropy();
    let mut marking = vec![0u64; net.num_places()];
    let mut enabled = Vec::with_capacity(net.num_transitions());

    for walk_id in 0..walks {
        // Poll the deadline and early-exit gate every 100 walks (matching the
        // other walk loops).
        if walk_id % 100 == 0 {
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    return observed;
                }
            }
            if all_pinned(&observed) {
                return observed;
            }
        }

        // Restart each walk from the initial marking.
        marking.copy_from_slice(&net.initial_marking);

        for _step in 0..max_steps {
            // Collect enabled transitions at the current marking.
            enabled.clear();
            for t in 0..net.num_transitions() {
                let tidx = TransitionIdx(t as u32);
                if net.is_enabled(&marking, tidx) {
                    enabled.push(tidx);
                }
            }

            // Deadlock: no enabled transitions, restart the walk.
            if enabled.is_empty() {
                break;
            }

            // Fire a uniformly-random enabled transition in place.
            let chosen = enabled[rng.gen_range(0..enabled.len())];
            // Token-count overflow means this walk reached a non-representable
            // marking — abandon it (the next walk reinitialises `marking`).
            // This lane only emits a LOWER bound, so dropping a walk is sound
            // (we simply forgo a potentially-larger observation on that path).
            if net.apply_delta(&mut marking, chosen).is_err() {
                break;
            }

            // This is a reachable marking. Record each query's observed maximum
            // — exactly the sum `UpperBoundsObserver` would compute (the query
            // multiset summed with multiplicity).
            let mut all_pinned_now = true;
            for (q, query) in queries.iter().enumerate() {
                let sum: u64 = query.iter().map(|&p| marking[p]).sum();
                if sum > observed[q] {
                    observed[q] = sum;
                }
                if !targets[q].is_some_and(|t| observed[q] >= t) {
                    all_pinned_now = false;
                }
            }
            if all_pinned_now {
                return observed;
            }
        }
    }

    observed
}

/// Random-walk QUASI-LIVENESS witness search.
///
/// MCC `QuasiLiveness` is TRUE iff EVERY transition is quasi-live, i.e. SOME
/// reachable marking enables it. A transition `t` is quasi-live iff there is a
/// reachable marking `M` with `net.is_enabled(M, t)`.
///
/// This lane fires ONLY enabled transitions from `net.initial_marking`, so every
/// marking it visits is reachable BY CONSTRUCTION. At each visited marking it
/// records (in `observed[t] = true`) every transition `t` that is enabled there.
/// Because the marking is reachable and enablement is checked directly via
/// [`PetriNet::is_enabled`], each `observed[t] == true` is a PROVEN quasi-liveness
/// witness for `t` — a sound TRUE-direction fact.
///
/// Returns the per-transition "observed enabled at some reachable marking"
/// vector (length `net.num_transitions()`). The caller EXPANDS the exhaustive-BFS
/// observer seed with these flags; the identical exhaustive BFS still keeps the
/// final word, so the walk can only ADD covered transitions, never remove or fake
/// one.
///
/// Soundness: this is a strict under-approximation that emits ONLY positive
/// (quasi-live) witnesses. It NEVER infers quasi-liveness for an unobserved
/// transition and NEVER returns a universal / FALSE verdict: a transition the
/// walk never sees enabled stays `false`, leaving it unresolved for the
/// exhaustive lanes. A miss, an overflow, or a timeout simply yields fewer `true`
/// flags — never a wrong one.
///
/// Note: the initial marking IS inspected (unlike the deadlock lane, which skips
/// the trivial initial-dead case). Any transition enabled at the initial marking
/// is genuinely quasi-live, so recording it is sound and useful.
pub(crate) fn run_random_walk_quasi_liveness_witness(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> Vec<bool> {
    run_random_walk_quasi_liveness_witness_params(net, deadline, DEFAULT_WALKS, DEFAULT_MAX_STEPS)
}

/// Parameterized version of [`run_random_walk_quasi_liveness_witness`] for testing.
pub(crate) fn run_random_walk_quasi_liveness_witness_params(
    net: &PetriNet,
    deadline: Option<Instant>,
    walks: u32,
    max_steps: u32,
) -> Vec<bool> {
    let num_transitions = net.num_transitions();
    let mut observed = vec![false; num_transitions];
    if num_transitions == 0 {
        return observed;
    }

    // Number of transitions not yet observed enabled; once zero, every
    // transition is witnessed quasi-live and no further walking can improve the
    // result, so we stop early (mirrors the deadlock/upper-bounds early-exit).
    let mut remaining = num_transitions;

    let mut enabled: Vec<TransitionIdx> = Vec::with_capacity(num_transitions);

    // The initial marking is reachable; record its enabled transitions even if
    // the walk/step budget is empty.
    record_enabled(
        net,
        &net.initial_marking,
        &mut enabled,
        &mut observed,
        &mut remaining,
    );
    if remaining == 0 || walks == 0 || max_steps == 0 {
        return observed;
    }

    // The shared-engine control loop drives the walk/step budget, deadline +
    // early-exit polling cadence, and restart-from-initial; the Petri-specific
    // work (per-marking enabled-transition recording via the same
    // `record_enabled`, RNG-driven single-transition selection, `apply_delta`)
    // stays in `QuasiLivenessWalk` so the trajectory is byte-identical to the
    // pre-extraction loop.
    let mut stepper = QuasiLivenessWalk {
        net,
        deadline,
        rng: SmallRng::from_entropy(),
        marking: vec![0u64; net.num_places()],
        enabled,
        observed,
        remaining,
    };
    random_walk_witness(&mut stepper, walk_budget(walks, max_steps));
    stepper.observed
}

/// Record every transition enabled at `marking` (a reachable marking by
/// construction) into `observed`, leaving `enabled` populated for the caller
/// to pick a transition to fire. Decrements `remaining` for each newly-seen
/// transition. A free fn (not a closure) so it does not hold a long-lived
/// mutable borrow of `observed` / `remaining`.
fn record_enabled(
    net: &PetriNet,
    marking: &[u64],
    enabled: &mut Vec<TransitionIdx>,
    observed: &mut [bool],
    remaining: &mut usize,
) {
    enabled.clear();
    for t in 0..observed.len() {
        let tidx = TransitionIdx(t as u32);
        if net.is_enabled(marking, tidx) {
            enabled.push(tidx);
            if !observed[t] {
                observed[t] = true;
                *remaining -= 1;
            }
        }
    }
}

/// Petri adapter for [`run_random_walk_quasi_liveness_witness_params`]:
/// records every transition enabled at each reachable marking it visits and
/// fires one random enabled transition per step.
struct QuasiLivenessWalk<'a> {
    net: &'a PetriNet,
    deadline: Option<Instant>,
    rng: SmallRng,
    marking: Vec<u64>,
    enabled: Vec<TransitionIdx>,
    observed: Vec<bool>,
    /// Transitions not yet observed enabled; once zero, every transition is
    /// witnessed quasi-live and no further walking can improve the result.
    remaining: usize,
}

impl RandomWalkStepper for QuasiLivenessWalk<'_> {
    fn should_stop(&mut self) -> RandomWalkPoll {
        if let Some(dl) = self.deadline {
            if Instant::now() >= dl {
                return RandomWalkPoll::Stop;
            }
        }
        if self.remaining == 0 {
            return RandomWalkPoll::Stop;
        }
        RandomWalkPoll::Continue
    }

    fn reset_to_initial(&mut self) {
        self.marking.copy_from_slice(&self.net.initial_marking);
    }

    fn step(&mut self) -> RandomWalkStep {
        // Record the transitions enabled at the current (reachable) marking,
        // leaving `enabled` populated for the random selection below.
        record_enabled(
            self.net,
            &self.marking,
            &mut self.enabled,
            &mut self.observed,
            &mut self.remaining,
        );
        if self.remaining == 0 {
            return RandomWalkStep::Done;
        }

        // Deadlock: no enabled transitions, restart the walk.
        if self.enabled.is_empty() {
            return RandomWalkStep::Dead;
        }

        // Fire a uniformly-random enabled transition in place.
        let chosen = self.enabled[self.rng.gen_range(0..self.enabled.len())];
        // Token-count overflow means this walk reached a non-representable
        // marking — abandon it (the next walk reinitialises `marking`).
        // This lane only emits TRUE (quasi-live) witnesses, so dropping a
        // walk is sound (we simply forgo potential observations on that path).
        if self.net.apply_delta(&mut self.marking, chosen).is_err() {
            return RandomWalkStep::Abandon;
        }
        RandomWalkStep::Advanced
    }
}

/// Check if all walkable (EF or AG) trackers have been resolved.
fn all_walkable_resolved(trackers: &[PropertyTracker]) -> bool {
    trackers.iter().all(|t| t.verdict.is_some())
}

fn validate_marking(
    trackers: &mut [PropertyTracker],
    marking: &[u64],
    net: &PetriNet,
    path: &[TransitionIdx],
    validation: &WitnessValidationContext<'_>,
) {
    let candidates =
        candidates_for_marking(trackers, marking, net, WitnessSeedSource::RandomWalk, path);
    apply_validated_witnesses(validation, trackers, candidates);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examinations::reachability_witness::{
        validation_targets_from_trackers, WitnessValidationContext,
    };
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
    use crate::property_xml::PathQuantifier;
    use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn transition(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    /// Build a simple mutex net:
    ///   p_free(2) -> t_enter -> p_critical
    ///   p_critical -> t_exit -> p_free
    ///
    /// Initial: p_free=2, p_critical=0
    /// Reachable: p_critical can reach 1 (but not 2 in a well-designed mutex,
    /// though this net allows it since there's no mutual exclusion guard).
    fn mutex_net() -> PetriNet {
        PetriNet {
            name: Some("mutex".to_string()),
            places: vec![place("p_free"), place("p_critical")],
            transitions: vec![
                transition("t_enter", vec![arc(0, 1)], vec![arc(1, 1)]),
                transition("t_exit", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![2, 0],
        }
    }

    /// Build a 1-safe net where tokens(p1) can never reach 100.
    fn one_safe_net() -> PetriNet {
        PetriNet {
            name: Some("one_safe".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                transition("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                transition("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// Build a net that deadlocks after 1 step.
    fn deadlock_net() -> PetriNet {
        PetriNet {
            name: Some("deadlock".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                transition("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                // No transition to consume from p1 — deadlock after firing t0.
            ],
            initial_marking: vec![1, 0],
        }
    }

    fn ef_tracker(id: &str, pred: ResolvedPredicate) -> PropertyTracker {
        PropertyTracker {
            id: id.to_string(),
            quantifier: PathQuantifier::EF,
            predicate: pred,
            verdict: None,
            resolved_by: None,
            flushed: false,
        }
    }

    fn ag_tracker(id: &str, pred: ResolvedPredicate) -> PropertyTracker {
        PropertyTracker {
            id: id.to_string(),
            quantifier: PathQuantifier::AG,
            predicate: pred,
            verdict: None,
            resolved_by: None,
            flushed: false,
        }
    }

    /// tokens(place) >= threshold as IntLe(Constant(threshold), TokensCount([place])).
    fn tokens_ge(place: u32, threshold: u64) -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(threshold),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(place)]),
        )
    }

    #[test]
    fn test_random_walk_finds_witness() {
        let net = mutex_net();
        // EF(tokens(p_critical) >= 1) — should be TRUE (reachable by firing t_enter).
        let pred = tokens_ge(1, 1);
        let mut trackers = vec![ef_tracker("prop1", pred)];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        run_random_walk_seeding_params(&net, &mut trackers, &validation, None, 100, 100);

        assert_eq!(
            trackers[0].verdict,
            Some(true),
            "random walk should find EF witness"
        );
    }

    #[test]
    fn test_random_walk_finds_ag_counterexample() {
        let net = mutex_net();
        // AG(NOT(tokens(p_critical) >= 1)) — FALSE because t_enter makes p_critical >= 1.
        let pred = ResolvedPredicate::Not(Box::new(tokens_ge(1, 1)));
        let mut trackers = vec![ag_tracker("prop_ag", pred)];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        run_random_walk_seeding_params(&net, &mut trackers, &validation, None, 100, 100);

        assert_eq!(
            trackers[0].verdict,
            Some(false),
            "random walk should find AG counterexample"
        );
    }

    #[test]
    fn test_random_walk_unreachable_leaves_none() {
        let net = one_safe_net();
        // EF(tokens(p1) >= 100) — unreachable on a 1-safe net.
        let pred = tokens_ge(1, 100);
        let mut trackers = vec![ef_tracker("prop_unreach", pred)];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        run_random_walk_seeding_params(&net, &mut trackers, &validation, None, 100, 100);

        assert_eq!(
            trackers[0].verdict, None,
            "random walk should not claim FALSE for unreachable EF"
        );
    }

    #[test]
    fn test_random_walk_deadlock_restarts() {
        let net = deadlock_net();
        // EF(tokens(p1) >= 1) — reachable after 1 step (t0 fires once then deadlock).
        let pred = tokens_ge(1, 1);
        let mut trackers = vec![ef_tracker("prop_dead", pred)];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        run_random_walk_seeding_params(&net, &mut trackers, &validation, None, 100, 100);

        assert_eq!(
            trackers[0].verdict,
            Some(true),
            "random walk should find witness before deadlock"
        );
    }

    #[test]
    fn test_random_walk_respects_deadline() {
        let net = one_safe_net();
        let pred = tokens_ge(1, 100);
        let mut trackers = vec![ef_tracker("prop_timeout", pred)];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        // Deadline already passed — should return immediately.
        let deadline = Some(Instant::now());
        run_random_walk_seeding_params(
            &net,
            &mut trackers,
            &validation,
            deadline,
            100_000,
            100_000,
        );

        assert_eq!(
            trackers[0].verdict, None,
            "random walk should respect deadline and not resolve"
        );
    }

    #[test]
    fn test_random_walk_empty_net() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![],
            initial_marking: vec![5],
        };
        let pred = tokens_ge(0, 10);
        let mut trackers = vec![ef_tracker("prop_empty", pred)];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        run_random_walk_seeding_params(&net, &mut trackers, &validation, None, 100, 100);

        assert_eq!(
            trackers[0].verdict, None,
            "empty net should leave verdict as None"
        );
    }

    #[test]
    fn test_random_walk_empty_net_checks_initial_marking() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![],
            initial_marking: vec![5],
        };
        let mut ef_trackers = vec![ef_tracker("prop_init_true", tokens_ge(0, 5))];
        let ef_targets = validation_targets_from_trackers(&ef_trackers);
        let ef_validation = WitnessValidationContext::new(&net, &ef_targets);

        run_random_walk_seeding_params(&net, &mut ef_trackers, &ef_validation, None, 100, 100);

        assert_eq!(
            ef_trackers[0].verdict,
            Some(true),
            "initial marking is reachable and should satisfy EF witness checks"
        );

        let pred = ResolvedPredicate::Not(Box::new(tokens_ge(0, 5)));
        let mut ag_trackers = vec![ag_tracker("prop_init_false", pred)];
        let ag_targets = validation_targets_from_trackers(&ag_trackers);
        let ag_validation = WitnessValidationContext::new(&net, &ag_targets);

        run_random_walk_seeding_params(&net, &mut ag_trackers, &ag_validation, None, 100, 100);

        assert_eq!(
            ag_trackers[0].verdict,
            Some(false),
            "initial marking counterexample should seed AG(false) even without transitions"
        );
    }

    /// A live mutex cycle: `t_enter` / `t_exit` can always fire (there is always
    /// either a free or a critical token), so there is NO reachable deadlock.
    fn live_cycle_net() -> PetriNet {
        // p_free=1, p_critical=0; t_enter consumes p_free->p_critical,
        // t_exit consumes p_critical->p_free. Always exactly one enabled.
        PetriNet {
            name: Some("live_cycle".to_string()),
            places: vec![place("p_free"), place("p_critical")],
            transitions: vec![
                transition("t_enter", vec![arc(0, 1)], vec![arc(1, 1)]),
                transition("t_exit", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    #[test]
    fn test_random_walk_deadlock_finds_reachable_deadlock() {
        // deadlock_net(): t0 fires once (p0->p1), then p1 has no consumer ->
        // a reachable, non-initial dead marking.
        let net = deadlock_net();
        assert!(
            run_random_walk_deadlock_params(&net, None, 100, 100),
            "random walk should find the reachable deadlock"
        );
    }

    #[test]
    fn test_random_walk_deadlock_live_cycle_returns_false() {
        // A live cycle has no reachable deadlock; the walk must NOT claim TRUE,
        // even with a generous budget. (No false positive.)
        let net = live_cycle_net();
        assert!(
            !run_random_walk_deadlock_params(&net, None, 200, 200),
            "live cycle has no deadlock; walk must return false"
        );
    }

    #[test]
    fn test_random_walk_deadlock_respects_deadline() {
        // Even on a net WITH a reachable deadlock, an already-expired deadline
        // returns false immediately (verdict-preserving miss).
        let net = deadlock_net();
        let deadline = Some(Instant::now());
        assert!(
            !run_random_walk_deadlock_params(&net, deadline, 100_000, 100_000),
            "expired deadline should return false without searching"
        );
    }

    #[test]
    fn test_random_walk_deadlock_no_transitions_returns_false() {
        // Net with no transitions: initial-dead trivial case is left to other
        // lanes; this lane reports false.
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![],
            initial_marking: vec![5],
        };
        assert!(
            !run_random_walk_deadlock_params(&net, None, 100, 100),
            "net with no transitions should return false"
        );
    }

    /// A net where EVERY place is perturbed by some firing. The live mutex cycle
    /// flips both p_free (2->1) and p_critical (0->1) away from their initial
    /// values, so both places are observably non-constant ⇒ FALSE witness.
    #[test]
    fn test_random_walk_stable_marking_all_unstable_returns_false() {
        let net = mutex_net(); // init [2, 0]; t_enter -> [1,1], t_exit back.
        let initial_unstable = vec![false; net.num_places()];
        let result = run_random_walk_stable_marking_params(&net, &initial_unstable, None, 100, 100);
        assert_eq!(
            result,
            Some(vec![true; net.num_places()]),
            "every place is perturbable; walk should prove StableMarking FALSE"
        );
    }

    /// A net with a provably-constant place: `p_const` is isolated (no transition
    /// touches it), so it can never deviate from its initial value. The walk must
    /// NEVER conclude FALSE (no false positive) and must return None.
    #[test]
    fn test_random_walk_stable_marking_constant_place_returns_none() {
        // p_free / p_critical cycle (both perturbable) PLUS an isolated p_const
        // that no transition reads or writes. p_const stays at its initial 7.
        let net = PetriNet {
            name: Some("with_const".to_string()),
            places: vec![place("p_free"), place("p_critical"), place("p_const")],
            transitions: vec![
                transition("t_enter", vec![arc(0, 1)], vec![arc(1, 1)]),
                transition("t_exit", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0, 7],
        };
        let initial_unstable = vec![false; net.num_places()];
        // Generous budget: even so, p_const can never be falsified.
        let result = run_random_walk_stable_marking_params(&net, &initial_unstable, None, 200, 200);
        assert_eq!(
            result, None,
            "an isolated constant place must keep the walk inconclusive (no false FALSE)"
        );
    }

    /// A P-invariant-pinned constant place. In a 1-safe mutex `p_free + p_critical
    /// == 1`, neither place is individually constant, but consider a net where a
    /// place is pinned constant by a self-loop-only structure: here we reuse the
    /// 1-safe cycle and seed it so the only remaining candidate is genuinely
    /// constant, confirming the walk does not over-claim.
    #[test]
    fn test_random_walk_stable_marking_pinned_candidate_returns_none() {
        // p_lock is consumed and immediately returned by every transition, so it
        // is constant (a P-invariant pins it). p_work toggles.
        // t: consume p_lock + p_work_a -> produce p_lock + p_work_b.
        let net = PetriNet {
            name: Some("pinned".to_string()),
            places: vec![place("p_lock"), place("p_work_a"), place("p_work_b")],
            transitions: vec![
                transition(
                    "t_fwd",
                    vec![arc(0, 1), arc(1, 1)],
                    vec![arc(0, 1), arc(2, 1)],
                ),
                transition(
                    "t_bwd",
                    vec![arc(0, 1), arc(2, 1)],
                    vec![arc(0, 1), arc(1, 1)],
                ),
            ],
            initial_marking: vec![1, 1, 0],
        };
        let initial_unstable = vec![false; net.num_places()];
        let result = run_random_walk_stable_marking_params(&net, &initial_unstable, None, 200, 200);
        assert_eq!(
            result, None,
            "a place pinned constant by a P-invariant must never yield a FALSE witness"
        );
    }

    /// Expired deadline returns None without searching (verdict-preserving miss),
    /// even on a net where every place is perturbable.
    #[test]
    fn test_random_walk_stable_marking_respects_deadline() {
        let net = mutex_net();
        let initial_unstable = vec![false; net.num_places()];
        let deadline = Some(Instant::now());
        let result = run_random_walk_stable_marking_params(
            &net,
            &initial_unstable,
            deadline,
            100_000,
            100_000,
        );
        assert_eq!(
            result, None,
            "expired deadline should return None without searching"
        );
    }

    /// No transitions: nothing to walk, no candidate can be falsified ⇒ None.
    #[test]
    fn test_random_walk_stable_marking_no_transitions_returns_none() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![],
            initial_marking: vec![5, 3],
        };
        let initial_unstable = vec![false; net.num_places()];
        let result = run_random_walk_stable_marking_params(&net, &initial_unstable, None, 100, 100);
        assert_eq!(
            result, None,
            "net with no transitions cannot prove any place non-constant"
        );
    }

    /// When the seed already marks every place non-constant (BMC proved them
    /// all), the lane reports FALSE without needing a walk — the seed is sound.
    #[test]
    fn test_random_walk_stable_marking_seed_all_unstable_returns_false() {
        let net = mutex_net();
        let initial_unstable = vec![true; net.num_places()];
        let result = run_random_walk_stable_marking_params(&net, &initial_unstable, None, 1, 1);
        assert_eq!(
            result,
            Some(vec![true; net.num_places()]),
            "all-unstable seed should immediately yield the FALSE witness"
        );
    }

    /// A net with a REACHABLE 2-token overflow on a single place. The mutex net
    /// starts with p_free=2; firing `t_exit` is impossible initially, but
    /// `t_enter` moves p_free->p_critical. The place `p_free` already starts at
    /// 2, so that is the trivial initial overflow (handled by the caller). To
    /// exercise the reachable-but-non-initial path we build a producer net where
    /// a place accumulates tokens via firing.
    fn reachable_overflow_net() -> PetriNet {
        // p_src=1 -> t_split produces 2 tokens into p_dst (initial p_dst=0).
        // After one firing p_dst=2: a reachable, non-initial unit overflow.
        PetriNet {
            name: Some("reachable_overflow".to_string()),
            places: vec![place("p_src"), place("p_dst")],
            transitions: vec![transition("t_split", vec![arc(0, 1)], vec![arc(1, 2)])],
            initial_marking: vec![1, 0],
        }
    }

    #[test]
    fn test_random_walk_one_safe_finds_reachable_overflow() {
        // p_dst reaches 2 after firing t_split from the initial marking — a
        // reachable, non-initial safety-unit overflow ⇒ OneSafe FALSE witness.
        let net = reachable_overflow_net();
        let safety_units: Vec<Vec<usize>> = (0..net.num_places()).map(|p| vec![p]).collect();
        assert!(
            run_random_walk_one_safe_params(&net, &safety_units, None, 100, 100),
            "random walk should find the reachable 2-token overflow"
        );
    }

    #[test]
    fn test_random_walk_one_safe_group_sum_witness() {
        // Two places that each stay ≤ 1 individually but whose GROUP SUM reaches
        // 2: t_split puts one token in p_a and one in p_b from p_src. Neither
        // place exceeds 1 (per-place max would say SAFE), but the group {p_a,p_b}
        // sums to 2 ⇒ FALSE. This is the parity-preserving group-sum check.
        let net = PetriNet {
            name: Some("group_overflow".to_string()),
            places: vec![place("p_src"), place("p_a"), place("p_b")],
            transitions: vec![transition(
                "t_split",
                vec![arc(0, 1)],
                vec![arc(1, 1), arc(2, 1)],
            )],
            initial_marking: vec![1, 0, 0],
        };
        // Single colored group {p_a, p_b}; p_src is its own singleton unit.
        let safety_units: Vec<Vec<usize>> = vec![vec![1, 2], vec![0]];
        assert!(
            run_random_walk_one_safe_params(&net, &safety_units, None, 100, 100),
            "group sum {{p_a,p_b}} reaches 2 even though each place stays ≤ 1"
        );
        // Per-place singletons alone would NOT witness FALSE here (each ≤ 1).
        let per_place: Vec<Vec<usize>> = (0..net.num_places()).map(|p| vec![p]).collect();
        assert!(
            !run_random_walk_one_safe_params(&net, &per_place, None, 100, 100),
            "no single place exceeds 1; only the group sum overflows"
        );
    }

    #[test]
    fn test_random_walk_one_safe_genuinely_one_safe_returns_false() {
        // one_safe_net(): p0=1,p1=0; t0 moves the single token p0->p1 and t1
        // moves it back. Every reachable marking has each place ∈ {0,1} and the
        // total token count is the invariant 1, so no safety unit ever reaches
        // a sum ≥ 2. The walk must NOT claim FALSE, even with a generous budget.
        let net = one_safe_net();
        let safety_units: Vec<Vec<usize>> = (0..net.num_places()).map(|p| vec![p]).collect();
        assert!(
            !run_random_walk_one_safe_params(&net, &safety_units, None, 200, 200),
            "a genuinely 1-safe net must never yield a FALSE witness"
        );
    }

    #[test]
    fn test_random_walk_one_safe_respects_deadline() {
        // Even on a net WITH a reachable overflow, an already-expired deadline
        // returns false immediately (verdict-preserving miss).
        let net = reachable_overflow_net();
        let safety_units: Vec<Vec<usize>> = (0..net.num_places()).map(|p| vec![p]).collect();
        let deadline = Some(Instant::now());
        assert!(
            !run_random_walk_one_safe_params(&net, &safety_units, deadline, 100_000, 100_000),
            "expired deadline should return false without searching"
        );
    }

    /// The walk observes a query's reachable maximum that equals its structural
    /// ceiling. `reachable_overflow_net` fires `t_split` once: p_dst goes 0 -> 2,
    /// so query {p_dst} attains 2. With a target of 2 (the structural cap), the
    /// walk observes 2 ⇒ the caller can pin the exact answer.
    #[test]
    fn test_random_walk_upper_bounds_finds_reachable_max() {
        let net = reachable_overflow_net(); // init [1, 0]; t_split: p_src->2*p_dst.
        let queries: Vec<Vec<usize>> = vec![vec![1]]; // {p_dst}
        let targets = vec![Some(2u64)];
        let observed =
            run_random_walk_upper_bounds_params(&net, &queries, &targets, None, 100, 100);
        assert_eq!(
            observed,
            vec![2],
            "walk should observe the reachable maximum (2) for {{p_dst}}"
        );
    }

    /// The walk's observed max stays strictly below an inflated structural ceiling
    /// ⇒ no pin (the returned value is a sound lower bound, never the ceiling). On
    /// the 1-safe net every place stays in {0,1}; a query {p1} can never reach 100,
    /// so the walk returns at most 1 and the caller must NOT pin to 100.
    #[test]
    fn test_random_walk_upper_bounds_does_not_overclaim() {
        let net = one_safe_net(); // init [1, 0]; single token cycles p0<->p1.
        let queries: Vec<Vec<usize>> = vec![vec![1]]; // {p1}
                                                      // Pretend the structural/LP upper bound is a (correct but unachieved) 100.
        let targets = vec![Some(100u64)];
        let observed =
            run_random_walk_upper_bounds_params(&net, &queries, &targets, None, 200, 200);
        assert!(
            observed[0] <= 1,
            "1-safe place can never exceed 1; walk observed {} (must not reach the 100 ceiling)",
            observed[0]
        );
        assert!(
            observed[0] < 100,
            "walk must never observe the unachievable ceiling (no over-claim)"
        );
    }

    /// An already-expired deadline returns the initial-marking sums unchanged
    /// (no searching). The initial-marking sum is itself a reachable lower bound,
    /// so the result is sound; crucially it does not climb toward any target.
    #[test]
    fn test_random_walk_upper_bounds_respects_deadline() {
        let net = reachable_overflow_net(); // init [1, 0].
        let queries: Vec<Vec<usize>> = vec![vec![1]]; // {p_dst}, initial sum 0.
        let targets = vec![Some(2u64)];
        let deadline = Some(Instant::now());
        let observed = run_random_walk_upper_bounds_params(
            &net, &queries, &targets, deadline, 100_000, 100_000,
        );
        assert_eq!(
            observed,
            vec![0],
            "expired deadline should return the initial-marking sum without searching"
        );
    }

    /// A query with no known ceiling (`target = None`) never triggers the early
    /// stop but still tracks its observed maximum across the budget.
    #[test]
    fn test_random_walk_upper_bounds_unbounded_target_tracks_max() {
        let net = reachable_overflow_net(); // p_dst reaches 2.
        let queries: Vec<Vec<usize>> = vec![vec![1]];
        let targets = vec![None];
        let observed =
            run_random_walk_upper_bounds_params(&net, &queries, &targets, None, 100, 100);
        assert_eq!(
            observed,
            vec![2],
            "without a target the walk still records the reachable maximum"
        );
    }

    /// A repeated place index in a query is summed with multiplicity, matching
    /// `UpperBoundsObserver`. Query {p_dst, p_dst} on the overflow net reaches
    /// 2 * 2 = 4.
    #[test]
    fn test_random_walk_upper_bounds_multiplicity() {
        let net = reachable_overflow_net(); // p_dst reaches 2.
        let queries: Vec<Vec<usize>> = vec![vec![1, 1]]; // p_dst counted twice.
        let targets = vec![Some(4u64)];
        let observed =
            run_random_walk_upper_bounds_params(&net, &queries, &targets, None, 100, 100);
        assert_eq!(
            observed,
            vec![4],
            "repeated place must be counted with multiplicity (2 * 2 = 4)"
        );
    }

    #[test]
    fn test_random_walk_one_safe_no_transitions_returns_false() {
        // Net with no transitions: the trivial initial case is left to other
        // lanes; this lane reports false.
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![],
            initial_marking: vec![5],
        };
        let safety_units: Vec<Vec<usize>> = vec![vec![0]];
        assert!(
            !run_random_walk_one_safe_params(&net, &safety_units, None, 100, 100),
            "net with no transitions should return false"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // QuasiLiveness witness-walk lane tests.
    // ─────────────────────────────────────────────────────────────────────────

    /// A net where transition `t1` is enabled ONLY after firing `t0` from the
    /// initial marking. p0=1, p1=0; t0 moves p0->p1, t1 consumes p1. t1 is NOT
    /// enabled initially but becomes enabled in the marking reached by firing t0,
    /// so the walk must record BOTH transitions as quasi-live.
    fn enable_after_sequence_net() -> PetriNet {
        PetriNet {
            name: Some("enable_after_sequence".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                transition("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                transition("t1", vec![arc(1, 1)], vec![]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// A net with a structurally-dead transition. p0=1, p1=0, p2=0; t0 moves
    /// p0->p1, t_dead needs p2>=1 but NOTHING ever produces p2, so t_dead is
    /// never enabled in any reachable marking. The walk must NEVER record it.
    fn dead_transition_witness_net() -> PetriNet {
        PetriNet {
            name: Some("dead_transition_witness".to_string()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                transition("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                transition("t_dead", vec![arc(2, 1)], vec![]),
            ],
            initial_marking: vec![1, 0, 0],
        }
    }

    /// (a) A transition enabled only after a firing sequence is recorded
    /// quasi-live by the walk — matching the exhaustive answer (both live).
    #[test]
    fn test_quasi_liveness_witness_records_sequence_enabled_transition() {
        let net = enable_after_sequence_net();
        let observed = run_random_walk_quasi_liveness_witness_params(&net, None, 200, 200);
        assert_eq!(
            observed,
            vec![true, true],
            "walk should witness both t0 (initial) and t1 (after firing t0) as quasi-live"
        );
    }

    /// (b) A transition NEVER enabled in any reachable marking must NOT be
    /// recorded — no false positive (the sound TRUE-witness-only contract).
    #[test]
    fn test_quasi_liveness_witness_never_records_dead_transition() {
        let net = dead_transition_witness_net();
        let observed = run_random_walk_quasi_liveness_witness_params(&net, None, 500, 500);
        assert!(
            observed[0],
            "t0 is enabled at the initial marking and must be witnessed"
        );
        assert!(
            !observed[1],
            "t_dead is never enabled in any reachable marking — must never be recorded"
        );
    }

    /// (c) An already-expired deadline returns no spurious witnesses without
    /// panicking. Only the initial-marking enablement (computed before the first
    /// deadline poll, and itself sound) may be present — never a deeper one.
    #[test]
    fn test_quasi_liveness_witness_respects_deadline() {
        let net = enable_after_sequence_net();
        let deadline = Some(Instant::now());
        let observed =
            run_random_walk_quasi_liveness_witness_params(&net, deadline, 100_000, 100_000);
        // The initial marking enables only t0; with an expired deadline the walk
        // never fires anything, so t1 (enabled only after a step) is NOT recorded.
        assert_eq!(
            observed,
            vec![true, false],
            "expired deadline records only the (sound) initial-marking enablement"
        );
    }

    /// Empty-transition net: returns an empty vector, no panic.
    #[test]
    fn test_quasi_liveness_witness_empty_net() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![],
            initial_marking: vec![5],
        };
        let observed = run_random_walk_quasi_liveness_witness_params(&net, None, 100, 100);
        assert!(
            observed.is_empty(),
            "net with no transitions yields an empty witness vector"
        );
    }

    /// A transition enabled at the initial marking is recorded even with a
    /// zero walk/step budget (the initial marking is reachable by definition).
    #[test]
    fn test_quasi_liveness_witness_initial_marking_zero_budget() {
        let net = enable_after_sequence_net();
        let observed = run_random_walk_quasi_liveness_witness_params(&net, None, 0, 0);
        assert_eq!(
            observed,
            vec![true, false],
            "the initial marking enables t0; zero budget still records it soundly"
        );
    }

    /// (d) Group-awareness sanity: the per-transition flags this lane returns are
    /// exactly what the colored-group re-aggregation (`all_groups_covered`)
    /// consumes. On a 2-binding colored transition unfolded into t0 (enabled
    /// after firing) and t1 (dead binding), the walk witnesses t0 — enough for
    /// the colored group {t0, t1} to be covered, while never faking t1.
    #[test]
    fn test_quasi_liveness_witness_grouped_binding_coverage() {
        // p_live=1, p_dead=0: t_live is enabled (self-loop on p_live), t_dead
        // needs p_dead which is never produced. A colored transition with
        // bindings {t_live, t_dead} is quasi-live via t_live alone.
        let net = PetriNet {
            name: Some("grouped_quasi".to_string()),
            places: vec![place("p_live"), place("p_dead")],
            transitions: vec![
                transition("t_live", vec![arc(0, 1)], vec![arc(0, 1)]),
                transition("t_dead", vec![arc(1, 1)], vec![]),
            ],
            initial_marking: vec![1, 0],
        };
        let observed = run_random_walk_quasi_liveness_witness_params(&net, None, 200, 200);
        assert_eq!(
            observed,
            vec![true, false],
            "walk witnesses the live binding t_live but never the dead binding t_dead"
        );
        // Group re-aggregation: the colored group {t_live, t_dead} has at least
        // one witnessed binding ⇒ covered (the exact check the dispatcher runs).
        let group = [0usize, 1usize];
        assert!(
            group.iter().any(|&idx| observed[idx]),
            "colored group {{t_live, t_dead}} is covered by the live binding"
        );
    }
}
