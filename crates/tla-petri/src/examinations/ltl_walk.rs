// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Random-walk FALSE-witness search for MCC LTL (`LTLCardinality` /
//! `LTLFireability`).
//!
//! MCC LTL is plain (UNFAIR) LTL over maximal paths with deadlock-stutter, and
//! the property is the universal `A(φ)`. The shared
//! [`build_ltl_counterexample_gba`] builds the GBA accepting `¬φ`. A reachable
//! path of the net whose marking trace contains a fair accepting lasso of that
//! GBA is therefore a concrete counterexample, proving `A(φ)` is **FALSE**.
//!
//! This lane fires random enabled transitions from the initial marking (a strict
//! under-approximation: every visited marking is reachable by construction) and
//! delegates the entire acceptance / deadlock-stutter / X-faithfulness decision
//! to the trusted, differentially-tested oracle
//! [`accepting_lasso_exists`](super::ltl_lasso_bmc::accepting_lasso_exists) — the
//! SAME oracle the explicit lasso-BMC lane uses to validate its SMT models. No
//! marking-loop / acceptance / deadlock-stutter logic is reimplemented here.
//!
//! Soundness: emits ONLY [`Verdict::False`], ONLY when `accepting_lasso_exists`
//! confirms a fair accepting GBA lasso over a reachable marking trace. It NEVER
//! returns TRUE; a miss, an overflow, or a deadline expiry returns `None`, which
//! the caller treats as "fall through, unresolved". A spurious self-loop can
//! never be reported because the oracle requires a genuine marking loop, and the
//! only stutter step this lane appends is at a verified deadlock (no enabled
//! transition), matching the on-the-fly Büchi self-loop convention.

use std::collections::HashSet;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tla_mc_core::{
    random_walk_witness, RandomWalkBudget, RandomWalkPoll, RandomWalkStep, RandomWalkStepper,
};

use crate::buchi::{build_ltl_counterexample_gba, Gba, LtlNnf};
use crate::output::Verdict;
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::resolved_predicate::ResolvedPredicate;

use super::ltl_lasso_bmc::accepting_lasso_exists;

/// Walk-index cadence at which the bespoke LTL walk polled the deadline. The
/// shared engine drives the poll at this exact cadence so the deadline-poll
/// behavior is byte-identical to the pre-extraction loop.
const WALK_POLL_INTERVAL: u32 = 100;

/// Default number of independent random walks (mirrors `reachability_walk`).
const DEFAULT_WALKS: u32 = 1000;

/// Default maximum steps per walk before restarting (mirrors `reachability_walk`).
const DEFAULT_MAX_STEPS: u32 = 10_000;

/// Per-walk accumulated-trace ceiling. The acceptance oracle is `O(depth^2)`
/// over the trace (it scans every loop-start alignment ending at the current
/// marking), so the trace length used for oracle checks is capped well below
/// `DEFAULT_MAX_STEPS` to keep the per-walk cost bounded on tight cycles. A walk
/// that has not closed an accepting lasso within this many steps restarts to
/// explore a different random path rather than grinding one long trace.
const WALK_TRACE_CAP: usize = 256;

/// Random-walk FALSE-witness search for the universal LTL property `A(φ)`.
///
/// Builds `gba = build_ltl_counterexample_gba(nnf)` (the automaton accepting
/// `¬φ`) and random-walks the RAW `net`, accumulating the marking trace. When
/// the current marking repeats an earlier one in the same walk (a marking loop
/// is NECESSARY for a product lasso — a cheap gate before the oracle), OR at a
/// genuine deadlock (after appending one deadlock-stutter step), it asks
/// [`accepting_lasso_exists`] whether the accumulated trace contains a fair
/// accepting GBA lasso. On `true` it returns `Some(Verdict::False)`; otherwise it
/// keeps walking and returns `None` on a miss or when `deadline` expires.
///
/// `resolved_atoms` are the LTL atom predicates already resolved against the net
/// aliases (exactly as the Phase 2.5 symbolic prefilter prepares them); their
/// indices line up with the GBA's `pos_atoms` / `neg_atoms`.
#[must_use]
pub(crate) fn try_ltl_witness_walk(
    net: &PetriNet,
    nnf: &LtlNnf,
    resolved_atoms: &[ResolvedPredicate],
    deadline: Option<Instant>,
) -> Option<Verdict> {
    try_ltl_witness_walk_params(
        net,
        nnf,
        resolved_atoms,
        deadline,
        DEFAULT_WALKS,
        DEFAULT_MAX_STEPS,
    )
}

/// Parameterized version of [`try_ltl_witness_walk`] for testing.
#[must_use]
pub(crate) fn try_ltl_witness_walk_params(
    net: &PetriNet,
    nnf: &LtlNnf,
    resolved_atoms: &[ResolvedPredicate],
    deadline: Option<Instant>,
    walks: u32,
    max_steps: u32,
) -> Option<Verdict> {
    // Respect an already-expired deadline before doing any work (verdict-
    // preserving miss).
    if deadline.is_some_and(|dl| Instant::now() >= dl) {
        return None;
    }

    let gba: Gba = build_ltl_counterexample_gba(nnf);
    // A GBA with no states / no initial transitions accepts nothing, so no
    // counterexample can ever exist — decline cheaply.
    if gba.num_states == 0 || gba.initial_transitions.is_empty() {
        return None;
    }

    // Even with zero transitions the initial marking forms a maximal (deadlocked)
    // path: a single deadlock-stutter step may already witness `¬φ` (e.g. an
    // initially-dead net falsifying `A(F p)`). Probe that one trace, then bail if
    // there is nothing to walk.
    let initial = net.initial_marking.clone();
    if net.num_transitions() == 0 {
        if marking_is_dead(net, &initial) {
            let markings = vec![initial.clone(), initial];
            if accepting_lasso_exists(&gba, resolved_atoms, net, &markings) {
                return Some(Verdict::False);
            }
        }
        return None;
    }
    if walks == 0 || max_steps == 0 {
        return None;
    }

    // Effective per-walk step ceiling: bound the accumulated trace so the
    // O(depth^2) oracle cost per walk stays small even on tight cycles.
    let step_cap = (max_steps as usize).min(WALK_TRACE_CAP);

    // The shared-engine control loop drives the walk/step budget, deadline
    // polling cadence, and restart-from-initial; the Petri/LTL-specific work
    // (RNG-driven single-transition selection, enabled-set enumeration,
    // `apply_delta`, marking-trace accumulation, the `accepting_lasso_exists`
    // oracle calls, and the `Verdict::False` emission) stays in `LtlWalk` so the
    // walk trajectory, the accumulated marking trace, and the oracle calls are
    // byte-identical to the pre-extraction loop.
    let mut stepper = LtlWalk {
        net,
        gba: &gba,
        resolved_atoms,
        deadline,
        rng: SmallRng::from_entropy(),
        initial: &initial,
        marking: initial.clone(),
        enabled: Vec::with_capacity(net.num_transitions()),
        markings: Vec::with_capacity(WALK_TRACE_CAP + 1),
        seen: HashSet::new(),
        found: None,
    };
    // The engine drives at most `step_cap` steps per walk (the historical inner
    // bound), so the budget's `max_steps` is `step_cap`, NOT `max_steps`.
    let budget =
        RandomWalkBudget::new(walks, step_cap as u32).with_poll_interval(WALK_POLL_INTERVAL);
    random_walk_witness(&mut stepper, budget);
    stepper.found
}

/// Petri/LTL adapter for [`try_ltl_witness_walk_params`]: random-walks the raw
/// net accumulating the marking trace, and at every marking loop / verified
/// deadlock-stutter asks the trusted [`accepting_lasso_exists`] oracle whether
/// the trace contains a fair accepting GBA lasso, emitting [`Verdict::False`]
/// when it does.
///
/// All Petri/LTL-specific state and decisions live here; the shared engine owns
/// only the loop/budget/restart cadence. The RNG is consumed at exactly the same
/// point (one `gen_range` per advancing step) and the oracle is called at exactly
/// the same trace endpoints as the pre-extraction loop, so the trajectory, the
/// accumulated trace, and the oracle calls are identical.
struct LtlWalk<'a> {
    net: &'a PetriNet,
    gba: &'a Gba,
    resolved_atoms: &'a [ResolvedPredicate],
    deadline: Option<Instant>,
    rng: SmallRng,
    /// The walk's initial marking (restored at the start of every walk).
    initial: &'a Vec<u64>,
    /// Current marking of the in-progress walk.
    marking: Vec<u64>,
    /// Scratch buffer for the enabled-transition set at the current marking.
    enabled: Vec<TransitionIdx>,
    /// The full marking trace of the current walk, starting at the initial
    /// marking.
    markings: Vec<Vec<u64>>,
    /// Distinct markings seen so far in the current walk, the cheap marking-loop
    /// gate: a product lasso REQUIRES a marking loop, so the oracle only runs
    /// once the current marking has been seen before in this walk.
    seen: HashSet<Vec<u64>>,
    /// Set to `Some(Verdict::False)` once the oracle confirms a fair accepting
    /// GBA lasso over the reachable marking trace.
    found: Option<Verdict>,
}

impl RandomWalkStepper for LtlWalk<'_> {
    fn should_stop(&mut self) -> RandomWalkPoll {
        if self.deadline.is_some_and(|dl| Instant::now() >= dl) {
            return RandomWalkPoll::Stop;
        }
        RandomWalkPoll::Continue
    }

    fn reset_to_initial(&mut self) {
        // Restart each walk from the initial marking.
        self.markings.clear();
        self.seen.clear();
        self.marking.clear();
        self.marking.extend_from_slice(self.initial);
        self.markings.push(self.marking.clone());
        self.seen.insert(self.marking.clone());
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
            // Genuine deadlock: the maximal path stutters here forever. Append
            // exactly one stutter step (the same marking) so the trace ends in
            // a self-loop, then ask the oracle — this matches the on-the-fly
            // Büchi self-loop-only-at-deadlock convention. (The oracle treats
            // the repeated final marking as `loop_start == depth - 1`.)
            self.markings.push(self.marking.clone());
            if accepting_lasso_exists(self.gba, self.resolved_atoms, self.net, &self.markings) {
                self.found = Some(Verdict::False);
                return RandomWalkStep::Done;
            }
            return RandomWalkStep::Dead;
        }

        // Fire a uniformly-random enabled transition in place.
        let chosen = self.enabled[self.rng.gen_range(0..self.enabled.len())];
        // #22: token-count overflow means this walk reached a non-representable
        // marking — abandon it (the next walk reinitialises). This lane only
        // emits the FALSE witness, so dropping a walk is sound.
        if self.net.apply_delta(&mut self.marking, chosen).is_err() {
            return RandomWalkStep::Abandon;
        }

        // This marking is reachable. Once it repeats an earlier marking in
        // this walk the trace contains a marking loop — a NECESSARY condition
        // for a product lasso, so the oracle is worth running. We do NOT break
        // on a non-accepting check: the oracle only considers loops that end
        // EXACTLY at the current (final) marking, so one repeat exposes a
        // single loop alignment that may not be the accepting one. Continuing
        // a few more steps lands the trace at a different cycle phase, which
        // the oracle then re-checks at the new endpoint (the F/response
        // discharge often needs the loop to start at a specific marking). The
        // walk keeps extending until it accepts, deadlocks, or hits the trace
        // cap — bounding the cumulative oracle cost per walk.
        self.markings.push(self.marking.clone());
        if self.seen.contains(&self.marking)
            && accepting_lasso_exists(self.gba, self.resolved_atoms, self.net, &self.markings)
        {
            self.found = Some(Verdict::False);
            return RandomWalkStep::Done;
        }
        self.seen.insert(self.marking.clone());
        RandomWalkStep::Advanced
    }
}

/// Whether `marking` has NO enabled transition (a genuine deadlock).
fn marking_is_dead(net: &PetriNet, marking: &[u64]) -> bool {
    !(0..net.num_transitions()).any(|t| net.is_enabled(marking, TransitionIdx(t as u32)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::petri_net::{Arc, PlaceIdx, PlaceInfo, TransitionInfo};
    use crate::resolved_predicate::ResolvedIntExpr;

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

    /// `tokens(place) >= 1` as `IntLe(Constant(1), TokensCount([place]))`.
    fn atom_ge_one(place: u32) -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(place)]),
        )
    }

    /// A live cycle: a single token toggles between `p0` and `p1`.
    ///   p0 --t0--> p1 --t1--> p0
    /// Initial `[1, 0]`. Reachable markings: `[1,0]` and `[0,1]`, forever.
    fn toggle_net() -> PetriNet {
        PetriNet {
            name: Some("ltl-walk-toggle".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                transition("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                transition("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// `G(p0 >= 1)` as an NNF: `Release(False, Atom(0))`.
    fn globally_atom0() -> LtlNnf {
        LtlNnf::Release(Box::new(LtlNnf::False), Box::new(LtlNnf::Atom(0)))
    }

    /// `G(F(p0 >= 1))` as an NNF: `Release(False, F atom0)` where
    /// `F x = Until(True, x)`.
    fn globally_finally_atom0() -> LtlNnf {
        LtlNnf::Release(
            Box::new(LtlNnf::False),
            Box::new(LtlNnf::Until(
                Box::new(LtlNnf::True),
                Box::new(LtlNnf::Atom(0)),
            )),
        )
    }

    /// On the toggle net, `A(G (p0 >= 1))` is FALSE: the reachable loop
    /// `[1,0] -> [0,1] -> [1,0]` visits `[0,1]` where `p0 >= 1` is false, so the
    /// run `(... [0,1] ...)^ω` falsifies `G (p0 >= 1)`. The walk must find it.
    #[test]
    fn walk_finds_reachable_g_counterexample() {
        let net = toggle_net();
        let atoms = vec![atom_ge_one(0)];
        let verdict = try_ltl_witness_walk_params(&net, &globally_atom0(), &atoms, None, 200, 200);
        assert_eq!(
            verdict,
            Some(Verdict::False),
            "the reachable loop through [0,1] falsifies A(G (p0 >= 1))"
        );
    }

    /// On the toggle net, `A(G F (p0 >= 1))` is TRUE: every reachable maximal path
    /// is the infinite toggle, which visits `[1,0]` (where `p0 >= 1` holds)
    /// infinitely often. NO accepting lasso of the negation exists, so the walk
    /// must NEVER return FALSE — it falls through with `None`.
    #[test]
    fn walk_satisfied_property_returns_none() {
        let net = toggle_net();
        let atoms = vec![atom_ge_one(0)];
        // The toggle net has one enabled transition at every marking, hence one
        // unique infinite path. Four 16-step walks still cross dozens of
        // repeated markings and exercise the lasso oracle non-vacuously without
        // multiplying its O(depth^2) work into billions of redundant checks.
        let verdict =
            try_ltl_witness_walk_params(&net, &globally_finally_atom0(), &atoms, None, 4, 16);
        assert_eq!(
            verdict, None,
            "A(G F (p0 >= 1)) holds; the walk must not emit a false FALSE"
        );
    }

    /// An already-expired deadline returns `None` without searching, even on the
    /// net where a counterexample exists (verdict-preserving miss).
    #[test]
    fn walk_respects_expired_deadline() {
        let net = toggle_net();
        let atoms = vec![atom_ge_one(0)];
        let deadline = Some(Instant::now());
        let verdict = try_ltl_witness_walk_params(
            &net,
            &globally_atom0(),
            &atoms,
            deadline,
            100_000,
            100_000,
        );
        assert_eq!(
            verdict, None,
            "expired deadline must return None without searching"
        );
    }

    /// A net that fires `t0` once (`p0 -> p1`) and then deadlocks at `[0,1]`. The
    /// property `A(G (p0 >= 1))` is FALSE: the maximal path `[1,0] -> [0,1]` then
    /// stutters at the dead `[0,1]` where `p0 >= 1` is false forever. The walk's
    /// deadlock-stutter step must surface this counterexample.
    #[test]
    fn walk_finds_deadlock_stutter_counterexample() {
        let net = PetriNet {
            name: Some("ltl-walk-deadlock".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![transition("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        };
        let atoms = vec![atom_ge_one(0)];
        let verdict = try_ltl_witness_walk_params(&net, &globally_atom0(), &atoms, None, 50, 50);
        assert_eq!(
            verdict,
            Some(Verdict::False),
            "the deadlock at [0,1] stutters forever where p0 >= 1 is false"
        );
    }

    /// A net with no transitions whose initial marking already deadlocks. For
    /// `A(F (p1 >= 1))` with the initial `[1, 0]` (p1 = 0 forever) the property is
    /// FALSE — the single dead state never reaches `p1 >= 1`. The zero-transition
    /// initial-deadlock probe must catch it.
    #[test]
    fn walk_initial_deadlock_no_transitions_finds_counterexample() {
        let net = PetriNet {
            name: Some("ltl-walk-init-dead".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![],
            initial_marking: vec![1, 0],
        };
        // F(p1 >= 1) = Until(True, atom0) with atom0 = (p1 >= 1).
        let finally_p1 = LtlNnf::Until(Box::new(LtlNnf::True), Box::new(LtlNnf::Atom(0)));
        let atoms = vec![atom_ge_one(1)];
        let verdict = try_ltl_witness_walk_params(&net, &finally_p1, &atoms, None, 10, 10);
        assert_eq!(
            verdict,
            Some(Verdict::False),
            "an initially-dead state never reaches p1 >= 1, falsifying A(F (p1 >= 1))"
        );
    }

    /// The same satisfied-property net under a tiny budget still returns `None`
    /// (no false FALSE under a small attempt).
    #[test]
    fn walk_tiny_budget_satisfied_returns_none() {
        let net = toggle_net();
        let atoms = vec![atom_ge_one(0)];
        let verdict = try_ltl_witness_walk_params(
            &net,
            &globally_finally_atom0(),
            &atoms,
            Some(Instant::now() + Duration::from_millis(200)),
            5,
            5,
        );
        assert_eq!(verdict, None, "tiny budget must not manufacture a FALSE");
    }
}
