// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Structural analysis for Petri nets.
//!
//! Provides verdicts without state-space exploration using siphon/trap
//! theory. A *trap* is a set of places T where every transition consuming
//! from T also produces into T: once marked, stays marked. A *siphon* is
//! the dual: once empty, stays empty.
//!
//! **Deadlock-freedom theorem:** If every minimal siphon contains a marked
//! trap, the net is deadlock-free (Murata 1989, §V-A).
//!
//! **Liveness (Commoner–Hack, exact both directions):** an ordinary
//! free-choice system is live ⟺ every (minimal) siphon contains an
//! initially marked trap [Commoner 1972; Hack 1972; Desel & Esparza,
//! *Free Choice Petri Nets*, CUP 1995, Thm 4.21; Murata 1989, Thm 20].
//! [`structural_live`] decides liveness exactly on classified state
//! machines, marked graphs (see [`crate::net_class`]) and — via the
//! complete minimal-siphon enumeration here — free-choice nets.

use std::time::Instant;

use crate::petri_net::PetriNet;

/// Maximum number of minimal siphons before aborting enumeration.
/// Pathological nets can have exponentially many siphons.
const MAX_SIPHONS: usize = 10_000;

/// Check whether the net is ordinary (all arc weights are 1).
fn is_ordinary(net: &PetriNet) -> bool {
    net.transitions.iter().all(|transition| {
        transition.inputs.iter().all(|arc| arc.weight == 1)
            && transition.outputs.iter().all(|arc| arc.weight == 1)
    })
}

/// Check whether the net satisfies the ordinary free-choice condition.
///
/// For every place with multiple outgoing choices, each target transition
/// must have that place as its sole input.
fn is_ordinary_free_choice(net: &PetriNet) -> bool {
    if !is_ordinary(net) {
        return false;
    }

    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); net.num_places()];
    for (transition_idx, transition) in net.transitions.iter().enumerate() {
        for arc in &transition.inputs {
            consumers[arc.place.0 as usize].push(transition_idx);
        }
    }

    for (place_idx, place_consumers) in consumers.iter().enumerate() {
        if place_consumers.len() <= 1 {
            continue;
        }

        for &transition_idx in place_consumers {
            let transition = &net.transitions[transition_idx];
            if transition.inputs.len() != 1 || transition.inputs[0].place.0 as usize != place_idx {
                return false;
            }
        }
    }

    true
}

/// Check if a place set forms a trap in the net.
///
/// A trap T satisfies: for every transition t that consumes from T
/// (•t ∩ T ≠ ∅), t also produces into T (t• ∩ T ≠ ∅).
#[allow(dead_code)] // Used in tests; will be used by Commoner's theorem (Phase 3)
fn is_trap(net: &PetriNet, places: &[bool]) -> bool {
    for t in &net.transitions {
        let consumes_from_set = t.inputs.iter().any(|a| places[a.place.0 as usize]);
        if consumes_from_set {
            let produces_into_set = t.outputs.iter().any(|a| places[a.place.0 as usize]);
            if !produces_into_set {
                return false;
            }
        }
    }
    true
}

/// Check if a place set forms a siphon in the net.
///
/// A siphon S satisfies: for every transition t that produces into S
/// (t• ∩ S ≠ ∅), t also consumes from S (•t ∩ S ≠ ∅).
fn is_siphon(net: &PetriNet, places: &[bool]) -> bool {
    for t in &net.transitions {
        let produces_into_set = t.outputs.iter().any(|a| places[a.place.0 as usize]);
        if produces_into_set {
            let consumes_from_set = t.inputs.iter().any(|a| places[a.place.0 as usize]);
            if !consumes_from_set {
                return false;
            }
        }
    }
    true
}

/// Compute the siphon closure of an initial place set.
///
/// Start with the given places and expand: for every transition t that
/// produces into the current set but does NOT consume from it, add all
/// of t's input places to the set. Iterate until fixpoint.
fn siphon_closure(net: &PetriNet, initial: &[bool]) -> Vec<bool> {
    let num_places = net.num_places();
    let mut set = initial.to_vec();
    loop {
        let mut changed = false;
        for t in &net.transitions {
            let produces_into = t.outputs.iter().any(|a| set[a.place.0 as usize]);
            if !produces_into {
                continue;
            }
            let consumes_from = t.inputs.iter().any(|a| set[a.place.0 as usize]);
            if consumes_from {
                continue;
            }
            // t produces into set but doesn't consume from it — add t's inputs
            for a in &t.inputs {
                let p = a.place.0 as usize;
                if !set[p] {
                    set[p] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    debug_assert!(is_siphon(net, &set), "siphon_closure must produce a siphon");
    // Check if it's actually a siphon (could be empty if initial was empty)
    let non_empty = set.iter().any(|&b| b);
    if !non_empty {
        return vec![false; num_places];
    }
    set
}

/// Try to minimize a siphon by removing one place at a time.
///
/// Returns a minimal siphon (no proper subset is also a siphon).
fn minimize_siphon(net: &PetriNet, siphon: &[bool]) -> Vec<bool> {
    let mut result = siphon.to_vec();
    for p in 0..result.len() {
        if !result[p] {
            continue;
        }
        result[p] = false;
        if !is_siphon(net, &result) || !result.iter().any(|&b| b) {
            result[p] = true; // must keep this place
        }
    }
    result
}

/// Whether single-place-seed closure+minimize is a PROVABLY COMPLETE
/// minimal-siphon enumerator for this net.
///
/// `minimal_siphons` enumerates at most one minimal siphon per seed place
/// (`minimize(siphon_closure({p}))`), deduplicated. In general this is NOT a
/// complete minimal-siphon enumerator — *even for ordinary free-choice nets*
/// it can miss minimal siphons (a counterexample: `t0: {p1,p2}->{p0,p3}`,
/// `t1: {p0,p3}->{p1,p2}` is ordinary free-choice yet the heuristic misses the
/// minimal siphon `{p0,p1}`). A missed siphon lacking a marked trap silently
/// breaks the Commoner/Hack coverage certificate, so the *positive* shortcut
/// (`Some(true)` => live / deadlock-free) is unsound unless enumeration is
/// complete.
///
/// Completeness condition (empirically validated against exhaustive
/// subset-enumeration over millions of random nets, zero counterexamples):
/// **every transition has exactly one input place.** With a single input per
/// transition the siphon-membership constraint is functional — the smallest
/// siphon containing a place `p` is unique and equals `siphon_closure({p})` —
/// so seeding from each place reaches every minimal siphon. (This class is
/// strictly broader than state machines: outputs are unconstrained. It is
/// neither implied by nor implies free-choice, and is checked independently.)
///
/// Source transitions (no inputs) are treated as NOT one-input: a transition
/// that produces into a place without consuming makes the producing place a
/// non-siphon and the closure reasoning above no longer applies, so we
/// conservatively decline.
fn single_seed_enumeration_complete(net: &PetriNet) -> bool {
    !net.transitions.is_empty()
        && net
            .transitions
            .iter()
            .all(|transition| transition.inputs.len() == 1)
}

/// Whether every place is incident to at least one transition (input or
/// output). An *isolated* place forms a vacuous singleton siphon: if it is
/// initially unmarked it has no marked trap, which would make the *negative*
/// shortcut (`Some(false)` => NOT live / has-deadlock) wrongly fire even
/// though the isolated place gates no transition and the net may well be
/// live/deadlock-free. The negative direction is therefore only trusted when
/// no such degenerate place exists. (The positive direction is unaffected:
/// an unmarked isolated place yields `Some(false)`, never a spurious
/// `Some(true)`.)
fn all_places_incident(net: &PetriNet) -> bool {
    let mut incident = vec![false; net.num_places()];
    for transition in &net.transitions {
        for arc in transition.inputs.iter().chain(transition.outputs.iter()) {
            incident[arc.place.0 as usize] = true;
        }
    }
    incident.iter().all(|&seen| seen)
}

/// Enumerate minimal siphons using closure-then-minimize.
///
/// For each place, compute the siphon closure of {p}, then minimize.
/// Deduplicate results. Aborts if more than `MAX_SIPHONS` are found.
///
/// **Incomplete in general** (see `single_seed_enumeration_complete`): callers
/// must not treat a fully-covered result as a definite positive verdict unless
/// `single_seed_enumeration_complete(net)` holds.
///
/// Returns `None` if enumeration was aborted (too many siphons).
fn minimal_siphons(net: &PetriNet) -> Option<Vec<Vec<bool>>> {
    let num_places = net.num_places();
    let mut siphons: Vec<Vec<bool>> = Vec::new();

    for seed_place in 0..num_places {
        let mut initial = vec![false; num_places];
        initial[seed_place] = true;
        let closure = siphon_closure(net, &initial);
        if !closure.iter().any(|&b| b) {
            continue;
        }
        let minimal = minimize_siphon(net, &closure);
        if !minimal.iter().any(|&b| b) {
            continue;
        }
        // Deduplicate
        if !siphons.iter().any(|s| s == &minimal) {
            siphons.push(minimal);
            if siphons.len() > MAX_SIPHONS {
                return None; // too many siphons, abort
            }
        }
    }
    Some(siphons)
}

/// Check if a siphon contains a marked trap (a subset that is both a trap
/// and contains at least one initially marked place).
fn contains_marked_trap(net: &PetriNet, siphon: &[bool]) -> bool {
    let num_places = net.num_places();
    // Strategy: find a marked subset of the siphon that forms a trap.
    // Start with all initially marked places within the siphon (the "marked
    // core"), then use trap closure to extend it. If the result is still
    // within the siphon and is a trap, we're done.
    //
    // Trap closure: for each transition t that produces into the current set
    // but does NOT consume from it, we must add places to make t consume
    // from the set. But that's the wrong direction for traps.
    //
    // Instead, use a simpler approach: the "maximal trap within the siphon"
    // is computable by iterative removal. Start with all siphon places,
    // remove any place p where some transition consumes from {p} but
    // doesn't produce into the remaining set. Iterate until fixpoint.
    let mut trap_candidate: Vec<bool> = siphon.to_vec();
    loop {
        let mut changed = false;
        for p in 0..num_places {
            if !trap_candidate[p] {
                continue;
            }
            // Check if removing p would still leave a trap
            // Actually: check if every transition consuming from p also
            // produces into the candidate set (trap condition for p)
            let p_ok = net.transitions.iter().all(|t| {
                let consumes_p = t.inputs.iter().any(|a| a.place.0 as usize == p);
                if !consumes_p {
                    return true; // transition doesn't touch p
                }
                // t consumes from p — must produce into trap_candidate
                t.outputs.iter().any(|a| trap_candidate[a.place.0 as usize])
            });
            if !p_ok {
                trap_candidate[p] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // The remaining trap_candidate is the maximal trap within the siphon.
    // Check if it's non-empty and marked.
    trap_candidate
        .iter()
        .enumerate()
        .any(|(p, &in_trap)| in_trap && net.initial_marking[p] > 0)
}

/// LP-based deadlock-freedom check for arbitrary Petri nets.
///
/// For each transition t, uses the LP state equation relaxation to check if
/// ALL input places always have sufficient tokens (m(p) >= w(p,t) in all
/// reachable markings). If any transition is provably always enabled, the
/// net is deadlock-free.
///
/// **Sound for ALL Petri net classes** — not limited to free-choice or
/// ordinary nets. The LP overapproximates the reachable set, so "always
/// enabled" in the LP implies always enabled in reality.
///
/// Returns:
/// - `Some(true)` if some transition is LP-proved always enabled
/// - `None` if inconclusive (no transition proved always enabled)
///
/// Does NOT return `Some(false)` — LP feasibility of disabling does not
/// prove a deadlock is actually reachable.
pub(crate) fn lp_deadlock_free(net: &PetriNet) -> Option<bool> {
    use crate::lp_state_equation::lp_strictly_greater_unreachable;
    use crate::petri_net::PlaceIdx;
    use crate::resolved_predicate::ResolvedIntExpr;

    if net.num_transitions() == 0 {
        return None;
    }

    for transition in &net.transitions {
        if transition.inputs.is_empty() {
            // Source transition (no inputs) is always enabled → deadlock-free.
            return Some(true);
        }

        let all_inputs_always_sufficient = transition.inputs.iter().all(|arc| {
            // Check: is w > m(p) LP-infeasible?
            // If yes → m(p) >= w always holds → this input is always satisfied.
            lp_strictly_greater_unreachable(
                net,
                &ResolvedIntExpr::Constant(arc.weight),
                &ResolvedIntExpr::TokensCount(vec![PlaceIdx(arc.place.0)]),
            )
        });

        if all_inputs_always_sufficient {
            return Some(true);
        }
    }

    None
}

/// LP-based dead-transition detection for arbitrary Petri nets.
///
/// For each transition t, checks if SOME input place can never have enough
/// tokens for t to fire (the LP upper bound for m(p) is strictly less than
/// the required arc weight). If any transition is LP-proved dead, the net
/// is NOT live.
///
/// **Sound for ALL Petri net classes.** The LP overapproximates reachable
/// markings: if even the LP says the place can't reach the required count,
/// it truly can't.
///
/// Returns `Some(false)` if a dead transition is found, `None` otherwise.
///
/// `deadline` is a cooperative soft cap polled between per-arc LP solves: the
/// check issues ONE `lp_upper_bound` LP per input arc, so on high-arity nets
/// the un-bounded loop ran the entire examination budget (measured 100+ s on
/// AutoFlight-PT-96a's 2 225 transitions; BlocksWorld-PT-10 never finished
/// inside 150 s) and starved every deciding lane downstream. Expiry returns
/// `None` — the prover merely declines its shortcut, which is exactly the
/// fall-through the caller already takes on a completed-but-inconclusive scan,
/// so the bound is verdict-preserving by construction.
pub(crate) fn lp_dead_transition(
    net: &PetriNet,
    deadline: Option<std::time::Instant>,
) -> Option<bool> {
    use crate::lp_state_equation::lp_upper_bound;
    use crate::petri_net::PlaceIdx;

    for transition in &net.transitions {
        for arc in &transition.inputs {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return None; // budget-out: decline the shortcut (sound)
            }
            let upper = lp_upper_bound(net, &[PlaceIdx(arc.place.0)]);
            if let Some(bound) = upper {
                if bound < arc.weight {
                    // This place can never have enough tokens → transition dead.
                    return Some(false);
                }
            }
        }
    }

    None
}

/// Fail-closed Commoner/Hack coverage test: "does every minimal siphon
/// contain a marked trap?"
///
/// The underlying minimal-siphon enumeration (`minimal_siphons`) is incomplete
/// in general, so a naive `all(covered)` over the enumerated set can emit a
/// WRONG positive verdict (live / deadlock-free) when an unenumerated minimal
/// siphon lacks a marked trap. This wrapper restricts each direction to the
/// regime where it is provably sound and returns `None` (decline → caller's
/// exact BFS/BMC path) otherwise:
///
/// - `Some(true)`  — only when `single_seed_enumeration_complete(net)` holds,
///   i.e. the enumeration is provably complete, so "all enumerated covered"
///   truly means "all minimal siphons covered" ⇒ deadlock-free / live.
/// - `Some(false)` — a genuine enumerated siphon without a marked trap is a
///   sound non-liveness/deadlock witness, but ONLY when no degenerate isolated
///   place can manufacture a spurious uncovered singleton siphon
///   (`all_places_incident`) AND the enumeration is complete (so the witness
///   is a real minimal siphon of the live/deadlock theorem's hypothesis).
/// - `None` — otherwise: the shortcut declines; the exact path decides.
///
/// Returning `None` can never introduce a wrong answer — it only removes an
/// unsound shortcut, routing to the exact decision procedure.
fn minimal_siphons_have_marked_traps(net: &PetriNet) -> Option<bool> {
    // The positive certificate requires a PROVABLY COMPLETE enumeration.
    // Without it, a missed minimal siphon could lack a marked trap, so we
    // cannot trust either aggregate direction — decline to the exact path.
    if !single_seed_enumeration_complete(net) {
        return None;
    }

    let siphons = minimal_siphons(net)?;
    let all_covered = siphons
        .iter()
        .all(|siphon| contains_marked_trap(net, siphon));

    if all_covered {
        // Complete enumeration + all covered ⇒ every minimal siphon contains a
        // marked trap ⇒ deadlock-free / live. Sound.
        Some(true)
    } else if all_places_incident(net) {
        // A real minimal siphon lacks a marked trap and no isolated place can
        // have faked it ⇒ NOT deadlock-free / NOT live. Sound.
        Some(false)
    } else {
        // Some uncovered siphon exists, but a degenerate isolated place could
        // be the (irrelevant) cause. Decline rather than risk a wrong negative.
        None
    }
}

/// Structural deadlock-freedom analysis for ordinary free-choice nets.
///
/// Returns `Some(true)` if the net is ordinary free-choice and provably
/// deadlock-free (every minimal siphon contains a marked trap).
/// Returns `Some(false)` if the net is ordinary free-choice but a siphon
/// vulnerability exists. Returns `None` if the net is not ordinary
/// free-choice or siphon enumeration was aborted.
pub(crate) fn structural_deadlock_free(net: &PetriNet) -> Option<bool> {
    // A net with no transitions is always deadlocked — the theorem about
    // "some transition is always enabled" is vacuously false.
    if net.num_transitions() == 0 {
        return Some(false);
    }
    if !is_ordinary_free_choice(net) {
        // The siphon/trap theorem holds for ALL ordinary nets (Murata 1989),
        // but our siphon enumerator uses single-place seeds which is only
        // complete for free-choice nets. Non-free-choice nets can have
        // minimal siphons requiring multi-place seeds — missing one causes
        // false Some(true). See: wrong_answer_investigation_tests::
        // test_squaregrid_deadlock_structural_not_falsely_free
        return None;
    }
    // `minimal_siphons_have_marked_traps` is itself fail-closed: it only
    // returns `Some(true)` when the single-place-seed enumeration is provably
    // complete, and only `Some(false)` when a genuine uncovered minimal siphon
    // is a sound witness (no degenerate isolated place). Otherwise `None`,
    // which we propagate so the caller falls through to the exact path.
    minimal_siphons_have_marked_traps(net)
}

/// Maximum minimal siphons to enumerate before declining (incomplete → None).
///
/// The complete enumeration ([`enumerate_all_minimal_siphons`]) is exponential
/// in the worst case; exceeding this cap means we cannot certify completeness,
/// so the deadlock-freedom shortcut must decline rather than risk a missed
/// emptiable siphon. The MCC deadlock-free families in scope have 16–31 minimal
/// siphons, well under the cap.
const MAX_MINIMAL_SIPHONS_COMPLETE: usize = 4_096;

/// Recursion-node budget for [`enumerate_all_minimal_siphons`]. Bounds the
/// (redundant) branch-and-bound search so a pathological net cannot spin;
/// exceeding it yields `None` (decline), never an incomplete "complete" set.
const MINIMAL_SIPHON_NODE_BUDGET: usize = 200_000;

/// Net-static adjacency used by the linear-time maximal-siphon worklist.
///
/// Built once per net and reused across the many `maximal_siphon_within_adj`
/// calls that the complete enumeration issues, so each maximal-siphon
/// computation costs `O(arcs)` instead of `O(places · arcs)` per fixpoint
/// iteration.
struct SiphonAdj {
    nt: usize,
    /// Distinct input places of each transition.
    t_inputs: Vec<Vec<usize>>,
    /// Distinct output places of each transition.
    t_outputs: Vec<Vec<usize>>,
    /// For each place, the transitions that consume from it (input arc).
    consumers: Vec<Vec<usize>>,
}

impl SiphonAdj {
    fn build(net: &PetriNet) -> Self {
        let np = net.num_places();
        let nt = net.num_transitions();
        let mut t_inputs = vec![Vec::new(); nt];
        let mut t_outputs = vec![Vec::new(); nt];
        let mut consumers = vec![Vec::new(); np];
        for (t, transition) in net.transitions.iter().enumerate() {
            for arc in &transition.inputs {
                let p = arc.place.0 as usize;
                if !t_inputs[t].contains(&p) {
                    t_inputs[t].push(p);
                    consumers[p].push(t);
                }
            }
            for arc in &transition.outputs {
                let p = arc.place.0 as usize;
                if !t_outputs[t].contains(&p) {
                    t_outputs[t].push(p);
                }
            }
        }
        SiphonAdj {
            nt,
            t_inputs,
            t_outputs,
            consumers,
        }
    }
}

/// Compute the maximal siphon contained in `allowed` via a linear-time worklist.
///
/// A place set `S` is a siphon iff every transition producing into `S` also
/// consumes from `S`. Siphons are closed under union, so there is a unique
/// maximal siphon contained in any place set `allowed`; it equals the result of
/// iteratively removing every place `p` that some transition produces into while
/// that transition has no input place left in the set. Maintaining each
/// transition's count of in-set input places lets us propagate removals in
/// `O(arcs)` total: when a transition's in-set input count hits zero, every
/// output place it still touches must leave the siphon.
///
/// Soundness (used by the deadlock proof): the returned set contains *every*
/// siphon `S ⊆ allowed` — the loop only ever removes a place that provably
/// cannot belong to any such siphon — and is itself a siphon, hence maximal.
fn maximal_siphon_within_adj(adj: &SiphonAdj, allowed: &[bool]) -> Vec<bool> {
    let np = allowed.len();
    let mut set = allowed.to_vec();
    let mut input_count = vec![0usize; adj.nt];
    for t in 0..adj.nt {
        input_count[t] = adj.t_inputs[t].iter().filter(|&&p| set[p]).count();
    }

    let mut queue: Vec<usize> = Vec::new();
    let mut queued = vec![false; np];
    let enqueue = |p: usize, set: &[bool], queued: &mut [bool], queue: &mut Vec<usize>| {
        if set[p] && !queued[p] {
            queued[p] = true;
            queue.push(p);
        }
    };
    // A transition with no in-set input (count 0, including source transitions)
    // threatens every output place it touches.
    for t in 0..adj.nt {
        if input_count[t] == 0 {
            for &q in &adj.t_outputs[t] {
                enqueue(q, &set, &mut queued, &mut queue);
            }
        }
    }

    while let Some(p) = queue.pop() {
        if !set[p] {
            continue;
        }
        set[p] = false;
        for &t in &adj.consumers[p] {
            if input_count[t] > 0 {
                input_count[t] -= 1;
                if input_count[t] == 0 {
                    for &q in &adj.t_outputs[t] {
                        enqueue(q, &set, &mut queued, &mut queue);
                    }
                }
            }
        }
    }

    set
}

/// Minimize a nonempty siphon to a *minimal* siphon (no proper nonempty subset
/// is a siphon).
///
/// Repeatedly tries to drop one place and re-close: if the maximal siphon within
/// the remaining places is still nonempty, replace the current siphon with that
/// strictly smaller one. At the fixed point no single-place removal admits a
/// nonempty sub-siphon, which (since any non-minimal siphon `X` with sub-siphon
/// `R ⊊ X` has a removable place `p ∈ X\R` whose re-closure recovers `R`) means
/// the result is minimal.
fn minimal_siphon_within(adj: &SiphonAdj, siphon: &[bool]) -> Vec<bool> {
    let mut current = siphon.to_vec();
    loop {
        let mut reduced = false;
        for p in 0..current.len() {
            if !current[p] {
                continue;
            }
            let mut without = current.clone();
            without[p] = false;
            let sub = maximal_siphon_within_adj(adj, &without);
            if sub.iter().any(|&in_set| in_set) {
                current = sub;
                reduced = true;
                break;
            }
        }
        if !reduced {
            break;
        }
    }
    current
}

/// Pack a boolean place mask into a compact `u64` bitset key (for memoization).
fn pack_mask(mask: &[bool]) -> Vec<u64> {
    let mut packed = vec![0u64; mask.len().div_ceil(64)];
    for (p, &set) in mask.iter().enumerate() {
        if set {
            packed[p / 64] |= 1u64 << (p % 64);
        }
    }
    packed
}

/// Recursive branch-and-bound worker for [`enumerate_all_minimal_siphons`].
///
/// Invariant: finds every minimal siphon disjoint from `excluded`. At each node
/// it takes the maximal siphon `Smax` avoiding `excluded`, minimizes it to one
/// minimal siphon `Smin`, records it, then recurses excluding each place of
/// `Smin` in turn. Any other minimal siphon `T` disjoint from `excluded` is not
/// a subset of `Smin` (both minimal, distinct), so some `p ∈ Smin\T` exists and
/// `T` is found in the branch excluding `p`.
///
/// Redundancy control: the set of minimal siphons reachable below a node depends
/// on `excluded` ONLY through `Smax` (they are exactly the minimal siphons
/// `⊆ Smax`). Memoizing on `Smax` therefore collapses every distinct `excluded`
/// set that yields the same maximal siphon into a single full exploration —
/// without affecting completeness, since the first exploration of an `Smax`
/// already finds all minimal siphons contained in it.
///
/// Returns `false` if the node budget or siphon cap was exceeded (enumeration
/// incomplete → caller must decline).
fn collect_minimal_siphons(
    adj: &SiphonAdj,
    excluded: &mut [bool],
    results: &mut Vec<Vec<bool>>,
    visited_maximal: &mut std::collections::HashSet<Vec<u64>>,
    budget: &mut usize,
    deadline: Option<Instant>,
) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    // Cooperative wall-clock bail: nets with a combinatorial siphon explosion
    // (e.g. Anderson-style mutex arrays) would otherwise grind the whole phase
    // cap before declining. Checking per node is cheap relative to the
    // `maximal_siphon_within_adj` work each node performs.
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return false;
    }

    let allowed: Vec<bool> = excluded.iter().map(|&e| !e).collect();
    let maximal = maximal_siphon_within_adj(adj, &allowed);
    if !maximal.iter().any(|&in_set| in_set) {
        return true; // no siphon disjoint from `excluded`
    }
    if !visited_maximal.insert(pack_mask(&maximal)) {
        return true; // this maximal siphon's subtree was already fully explored
    }

    let smin = minimal_siphon_within(adj, &maximal);
    if !results.iter().any(|existing| existing == &smin) {
        results.push(smin.clone());
        if results.len() > MAX_MINIMAL_SIPHONS_COMPLETE {
            return false;
        }
    }

    let members: Vec<usize> = (0..smin.len()).filter(|&p| smin[p]).collect();
    for p in members {
        excluded[p] = true;
        let ok = collect_minimal_siphons(adj, excluded, results, visited_maximal, budget, deadline);
        excluded[p] = false;
        if !ok {
            return false;
        }
    }
    true
}

/// Complete enumeration of every minimal siphon of `net`.
///
/// Returns `Some(siphons)` (possibly empty — a net with no nonempty siphon) when
/// the enumeration finished within the node/siphon budget, in which case the set
/// is provably complete. Returns `None` when the budget was exceeded, signalling
/// that completeness could not be certified and any positive deadlock-freedom
/// verdict must be declined.
///
/// Unlike [`minimal_siphons`] (single-place-seed, incomplete on non-free-choice
/// nets), this is sound to use as a *completeness certificate* for the
/// siphon-coverage deadlock theorem on arbitrary ordinary nets.
fn enumerate_all_minimal_siphons(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> Option<Vec<Vec<bool>>> {
    let (results, complete) = enumerate_minimal_siphons_partial(net, deadline);
    complete.then_some(results)
}

/// Like [`enumerate_all_minimal_siphons`], but also exposes the (genuine)
/// minimal siphons found by an INCOMPLETE run.
///
/// Returns `(siphons, complete)`. Every returned siphon is a real minimal
/// siphon of `net` regardless of `complete` — each is fully minimized by
/// [`minimal_siphon_within`] before being recorded, and the budget/deadline
/// can only TRUNCATE the search, never corrupt a recorded entry. Only the
/// "this is ALL of them" claim requires `complete == true`.
///
/// This split matters for the Commoner–Hack liveness certificate: the
/// positive direction (live) needs completeness, but the negative direction
/// (a minimal siphon with no marked trap ⇒ not live) is per-siphon, so a
/// witness found before the budget breach is still sound.
fn enumerate_minimal_siphons_partial(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> (Vec<Vec<bool>>, bool) {
    let np = net.num_places();
    let adj = SiphonAdj::build(net);

    let all = vec![true; np];
    let root = maximal_siphon_within_adj(&adj, &all);
    let mut results: Vec<Vec<bool>> = Vec::new();
    if !root.iter().any(|&in_set| in_set) {
        return (results, true); // no siphons at all
    }

    let mut visited_maximal: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
    let mut budget = MINIMAL_SIPHON_NODE_BUDGET;
    let mut excluded = vec![false; np];
    let complete = collect_minimal_siphons(
        &adj,
        &mut excluded,
        &mut results,
        &mut visited_maximal,
        &mut budget,
        deadline,
    );
    (results, complete)
}

/// Maximum LP problem size (places + transitions) for the siphon-LP deadlock
/// check. Mirrors the `lp_state_equation` LP cap so the per-siphon solves stay
/// bounded; larger nets fall through to the exact engine.
const MAX_SIPHON_LP_VARIABLES: usize = 50_000;

/// Sound structural deadlock-FREEDOM proof for **arbitrary ordinary** nets via
/// complete minimal-siphon enumeration + per-siphon LP non-emptiability.
///
/// **Theorem (Murata 1989, ordinary nets).** If a marking `M` is dead (no
/// transition enabled), then for every transition `t` some input place of `t`
/// is empty at `M` (ordinariness: arc weight 1 ⇒ disabled ⇒ an input place holds
/// 0 tokens). Hence the set `U = {p : M[p] = 0}` of empty places is a *siphon*:
/// any transition producing into `U` is itself disabled, so it consumes from
/// `U`. `U` is nonempty (a net with ≥1 transition and all places marked has an
/// enabled transition), so it contains a minimal siphon that is also empty at
/// `M`. **Contrapositive:** if every minimal siphon can never be simultaneously
/// token-free at any reachable marking, then no dead marking is reachable — the
/// net is deadlock-free.
///
/// "Minimal siphon `S` can never be emptied" is the predicate
/// `Σ_{p∈S} M[p] ≤ 0` being unreachable, which we prove soundly with the
/// state-equation + initially-marked-trap LP (a reachability over-approximation:
/// if the LP says the empty-siphon marking is infeasible, it is genuinely
/// unreachable). Because `lp_unreachable_with_traps` is strictly stronger than
/// the `contains_marked_trap` test, this also certifies siphons kept marked by a
/// P-invariant / state-equation argument with no contained marked trap.
///
/// Returns `Some(true)` ONLY when:
/// - the net is ordinary (`is_ordinary`) and has ≥1 transition, **and**
/// - the minimal-siphon enumeration is provably complete
///   (`enumerate_all_minimal_siphons` returned `Some`), **and**
/// - every enumerated minimal siphon is LP-proved non-emptiable.
///
/// Every other outcome returns `None` (decline). It never returns `Some(false)`:
/// an emptiable siphon witnesses only a *candidate* dead marking, not a reachable
/// one, so a deadlock-existence verdict must come from the exact engine with a
/// real witness. This is the sound generalization of [`structural_deadlock_free`]
/// off the free-choice restriction.
///
/// `deadline` bounds the (worst-case exponential) minimal-siphon enumeration and
/// the per-siphon LP sweep cooperatively: on a combinatorial siphon explosion
/// the check declines (`None`) at the deadline instead of grinding, so it never
/// starves the downstream PDR/AIGER/BMC/BFS engines. Bailing is verdict-
/// preserving — an incomplete enumeration can only decline, never certify.
pub(crate) fn lp_siphon_deadlock_free(net: &PetriNet, deadline: Option<Instant>) -> Option<bool> {
    use crate::lp_state_equation::{find_initially_marked_traps, lp_unreachable_with_traps_using};
    use crate::petri_net::PlaceIdx;
    use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

    if net.num_transitions() == 0 {
        // A net with no transitions is dead at M0 — not a deadlock-FREE proof.
        return None;
    }
    if !is_ordinary(net) {
        // The "disabled ⇒ empty input place" step requires unit arc weights.
        return None;
    }
    let np = net.num_places();
    if np + net.num_transitions() > MAX_SIPHON_LP_VARIABLES {
        return None;
    }

    let siphons = enumerate_all_minimal_siphons(net, deadline)?;

    // Traps depend only on the net; compute once and reuse for every siphon.
    let traps = find_initially_marked_traps(net);
    for siphon in &siphons {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        let places: Vec<PlaceIdx> = (0..np)
            .filter(|&p| siphon[p])
            .map(|p| PlaceIdx(p as u32))
            .collect();
        if places.is_empty() {
            continue;
        }
        // `Σ_{p∈S} M[p] ≤ 0` ⇔ siphon S is token-free. If LP+traps proves this
        // unreachable, S can never be emptied.
        let empty_siphon = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(places),
            ResolvedIntExpr::Constant(0),
        );
        if !lp_unreachable_with_traps_using(net, &empty_siphon, &traps) {
            // This minimal siphon might be emptiable ⇒ a dead marking could be
            // reachable. Cannot prove deadlock-freedom — decline to the exact
            // engine (which produces a witnessed verdict either way).
            return None;
        }
    }

    // Every minimal siphon is provably non-emptiable (vacuously so when there
    // are no siphons) ⇒ no reachable dead marking ⇒ deadlock-free.
    Some(true)
}

/// Exact structural L4-liveness certificate chain, gated by precise net-class
/// membership ([`crate::net_class::classify`]). Ordered cheap → expensive:
///
/// 1. **State machine** (shape + every place incident): per weakly-connected
///    component, live ⟺ strongly connected ∧ ≥ 1 token [Murata 1989, §VI].
///    Exact both directions, O(arcs).
/// 2. **Marked graph** (shape): live ⟺ every directed circuit initially
///    marked, checked as acyclicity of the empty-place transition digraph
///    [Commoner–Holt–Even–Pnueli 1971; Murata 1989, Thm 19]. Exact both
///    directions, O(arcs).
/// 3. **Free choice — Commoner–Hack** (simple FC + every place incident + no
///    source transition): live ⟺ every minimal siphon contains an initially
///    marked trap [Commoner 1972; Hack 1972; Desel & Esparza 1995, Thm 4.21;
///    Murata 1989, Thm 20]. Uses the provably-COMPLETE minimal-siphon
///    enumeration: `Some(true)` requires a complete enumeration with every
///    siphon covered; `Some(false)` requires only ONE genuine uncovered
///    minimal siphon (the only-if direction is per-siphon, so a witness from
///    a budget-truncated run is still sound); cap/deadline breach without a
///    witness → `None`.
///
/// Returns `None` (decline — caller's exact mu-calculus/SCC engines decide)
/// whenever no class gate holds, including: non-ordinary nets (weighted or
/// parallel arcs), transition-free or place-free nets, nets with an isolated
/// place (a vacuous unmarked singleton siphon would falsify the FC negative
/// direction — see [`all_places_incident`]), and FC nets with source
/// transitions (outside the textbook free-choice "system" setting).
///
/// ## Verdict semantics (MCC L4, colored)
///
/// `Some(true)` means EVERY P/T transition satisfies `AG EF enabled(t)`; this
/// implies every colored group's disjunctive `AG EF (∨ enabled(b))`, so the
/// TRUE direction is sound for colored consumers. `Some(false)` is
/// per-transition and therefore NOT colored-sound; callers must gate it on
/// the absence of colored groups (they do — see
/// `examination_non_property/liveness.rs`).
///
/// `deadline` bounds only the (worst-case exponential) Commoner enumeration;
/// the SM/MG certificates are linear and always complete. Expiry can only
/// turn a verdict into `None`, never flip one.
pub(crate) fn structural_live(net: &PetriNet, deadline: Option<Instant>) -> Option<bool> {
    use crate::net_class::{classify, marked_graph_live, state_machine_live};

    if net.num_transitions() == 0 || net.num_places() == 0 {
        return None;
    }
    let class = classify(net);
    if !class.ordinary {
        return None;
    }

    let sm_applies = class.state_machine && class.all_places_incident;
    let mg_applies = class.marked_graph;
    let fc_applies = class.free_choice && class.all_places_incident && !class.has_source_transition;

    let sm_verdict = sm_applies.then(|| state_machine_live(net));
    let mg_verdict = mg_applies.then(|| marked_graph_live(net));

    // Cross-certificate soundness check: whenever ≥ 2 exact certificates
    // apply to the same net (e.g. CircularTrains is both a marked graph and
    // free-choice), they MUST agree — disagreement means one of the exact
    // theorems was mis-implemented. Debug builds (tests, differential runs)
    // verify this; release builds skip the redundant work.
    #[cfg(debug_assertions)]
    {
        if let (Some(sm), Some(mg)) = (sm_verdict, mg_verdict) {
            assert_eq!(
                sm, mg,
                "state-machine and marked-graph liveness certificates disagree"
            );
        }
        if fc_applies {
            if let (Some(shape), Some(commoner)) = (
                sm_verdict.or(mg_verdict),
                free_choice_commoner_live(net, deadline),
            ) {
                assert_eq!(
                    shape, commoner,
                    "shape and Commoner-Hack liveness certificates disagree"
                );
            }
        }
    }

    if let Some(verdict) = sm_verdict.or(mg_verdict) {
        return Some(verdict);
    }
    if fc_applies {
        return free_choice_commoner_live(net, deadline);
    }
    None
}

/// Commoner–Hack liveness decision for an ordinary simple free-choice net.
///
/// Caller must have verified the gates: ordinary, simple free choice, every
/// place incident, no source transitions (see [`structural_live`]).
///
/// - `Some(true)` — the minimal-siphon enumeration completed within budget
///   AND every minimal siphon contains an initially marked trap. Complete +
///   covered ⇒ EVERY siphon contains a marked trap (any siphon contains a
///   minimal one; a marked trap inside the subset is one inside the
///   superset) ⇒ live (Commoner's theorem, if-direction).
/// - `Some(false)` — some enumerated minimal siphon contains NO initially
///   marked trap (its maximal trap is unmarked, the exact test). The only-if
///   direction is per-siphon, so this is sound EVEN IF the enumeration was
///   truncated: every recorded siphon is a genuine minimal siphon.
/// - `None` — enumeration incomplete with all found siphons covered, or the
///   deadline expired mid-coverage-check. Decline; exact engines decide.
fn free_choice_commoner_live(net: &PetriNet, deadline: Option<Instant>) -> Option<bool> {
    let (siphons, complete) = enumerate_minimal_siphons_partial(net, deadline);

    for siphon in &siphons {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            // TRUE needs every siphon verified and the remaining ones are
            // unchecked; no FALSE witness found so far → decline.
            return None;
        }
        if !contains_marked_trap(net, siphon) {
            return Some(false);
        }
    }

    if complete {
        Some(true)
    } else {
        None
    }
}

/// Structural non-liveness via T-semiflow coverage.
///
/// Returns `Some(false)` if any transition is NOT covered by any T-semiflow
/// AND the net is bounded (proved via P-invariants covering all places).
/// Returns `None` if all transitions are covered or boundedness cannot be
/// proved structurally.
///
/// **Sound for ALL Petri net classes** (not just free-choice). This
/// complements `structural_live` which only handles free-choice nets.
///
/// The theorem: in a bounded net, a transition not in any T-semiflow
/// can fire at most finitely many times — contradicting L4-liveness.
pub(crate) fn structural_not_live_t_semiflows(net: &PetriNet) -> Option<bool> {
    use crate::invariant::{
        all_transitions_covered, compute_p_invariants, compute_t_semiflows, structural_place_bound,
    };

    if net.num_transitions() == 0 {
        return None;
    }

    let result = compute_t_semiflows(net);
    if all_transitions_covered(&result.semiflows, net.num_transitions()) {
        return None; // all covered — can't conclude non-liveness
    }
    // If Farkas was truncated, we may be missing semiflows that would cover
    // the transition — can't soundly conclude non-coverage.
    if !result.complete {
        return None;
    }

    // Uncovered transition found. Need boundedness for the theorem to apply.
    // Check if P-invariants structurally bound every place.
    let p_inv = compute_p_invariants(net);
    let all_bounded = (0..net.num_places()).all(|p| structural_place_bound(&p_inv, p).is_some());

    if all_bounded {
        Some(false) // bounded + uncovered transition → NOT live
    } else {
        None // can't prove boundedness
    }
}

#[cfg(test)]
#[path = "structural_tests.rs"]
mod tests;
