// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! LP relaxation of the Petri net state equation for structural analysis.
//!
//! The state equation M = M0 + C*x (where C is the incidence matrix,
//! x >= 0 is a firing count vector, and M >= 0 is the target marking)
//! is a *necessary* condition for reachability. If the LP is infeasible
//! under added constraints, the target marking is provably unreachable.
//!
//! Two capabilities:
//!
//! 1. **Reachability pruning**: encode a predicate as LP constraints.
//!    If infeasible, the predicate is unreachable (EF = FALSE).
//!
//! 2. **Upper bound tightening**: maximize token sum for a place set
//!    subject to the state equation. At least as tight as P-invariant
//!    bounds (P-invariants are the LP dual).
//!
//! Reference: Murata (1989) "Petri nets: Properties, analysis and
//! applications." Proc. IEEE 77(4).

use std::time::Instant;

use minilp::{ComparisonOp, OptimizationDirection, Problem, Solution, Variable};

use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

/// Maximum LP variables before skipping (prevents pathological nets).
pub(crate) const MAX_LP_VARIABLES: usize = 50_000;

/// Maximum disabled-transition combinations to try for `IsFireable`.
///
/// Proving `IsFireable(t1, ..., tn)` always true requires proving that every
/// way of disabling all listed transitions is LP-infeasible. The number of
/// cases is the product of input-arc counts, so cap it and return unknown.
pub(crate) const MAX_FIREABILITY_CASE_SPLITS: usize = 256;

/// Maximum trap-refinement (CEGAR) iterations per state-equation LP query.
///
/// Each iteration adds at least one SOUND trap cut that strictly excludes the
/// current LP vertex (the vertex violates the cut, the re-solved vertex must
/// satisfy it), so the loop makes monotone progress and never re-derives an
/// already-added cut. The cap only bounds worst-case work on pathological nets;
/// hitting it returns `false` (inconclusive), which falls through to BFS and is
/// verdict-preserving. It also subsumes the old fixed `0..100` precomputed-trap
/// pass (the fast path below adds every violated precomputed trap per iteration,
/// so it converges well within the cap).
const MAX_CEGAR_ITERS: usize = 128;

/// A marking variable below this value is treated as "empty" when deriving a
/// separating trap. The separating trap is the maximal trap contained in
/// `Q = {p : M*[p] < TRAP_EMPTY_EPS}`; every place of that trap therefore has
/// `M*[p] < TRAP_EMPTY_EPS`, so even for a many-place trap the cut
/// `sum_{p in T} M >= 1` strictly excludes the current vertex (well below the
/// `< 0.5` violation threshold for the LP sizes this lane admits).
const TRAP_EMPTY_EPS: f64 = 1e-6;

/// Maximum LP variables (places + transitions) for the StableMarking pinning
/// sweep. The sweep issues up to `2 * num_places` LP solves, so this is kept
/// well below [`MAX_LP_VARIABLES`] to keep the whole pre-pass cheap; larger nets
/// fall through to the BMC/PDR/BFS engines unchanged.
const MAX_PINNING_LP_VARIABLES: usize = 8_192;

/// Add state equation constraints to an LP problem.
///
/// Given firing count variables `x_vars` and marking variables `m_vars`,
/// adds: m[p] - sum_t C[p][t]*x[t] = m0[p] for each place p.
fn add_state_equation(
    problem: &mut Problem,
    net: &PetriNet,
    x_vars: &[Variable],
    m_vars: &[Variable],
) {
    let np = net.num_places();
    for p in 0..np {
        let mut row = Vec::new();
        row.push((m_vars[p], 1.0));
        for (t, transition) in net.transitions.iter().enumerate() {
            // -C[p][t] = input_weight - output_weight
            let mut coeff = 0.0_f64;
            for arc in &transition.inputs {
                if arc.place.0 as usize == p {
                    coeff += arc.weight as f64;
                }
            }
            for arc in &transition.outputs {
                if arc.place.0 as usize == p {
                    coeff -= arc.weight as f64;
                }
            }
            if coeff.abs() > f64::EPSILON {
                row.push((x_vars[t], coeff));
            }
        }
        problem.add_constraint(&row, ComparisonOp::Eq, net.initial_marking[p] as f64);
    }
}

/// Accumulate an integer expression into a linear constraint row.
///
/// `variable_coeff` scales marking-variable terms, while `constant_coeff`
/// contributes to the scalar right-hand side.
fn accumulate_int_expr(
    expr: &ResolvedIntExpr,
    coefficients: &mut [f64],
    rhs: &mut f64,
    variable_coeff: f64,
    constant_coeff: f64,
) {
    match expr {
        ResolvedIntExpr::Constant(c) => *rhs += constant_coeff * (*c as f64),
        ResolvedIntExpr::TokensCount(places) => {
            for place in places {
                coefficients[place.0 as usize] += variable_coeff;
            }
        }
    }
}

/// Build the linear constraint row for `lhs <= rhs`.
fn build_int_le_constraint(
    lhs: &ResolvedIntExpr,
    rhs: &ResolvedIntExpr,
    m_vars: &[Variable],
) -> (Vec<(Variable, f64)>, f64) {
    let mut coefficients = vec![0.0_f64; m_vars.len()];
    let mut rhs_bound = 0.0_f64;
    accumulate_int_expr(lhs, &mut coefficients, &mut rhs_bound, 1.0, -1.0);
    accumulate_int_expr(rhs, &mut coefficients, &mut rhs_bound, -1.0, 1.0);
    let row = coefficients
        .into_iter()
        .enumerate()
        .filter_map(|(index, coefficient)| {
            (coefficient.abs() > f64::EPSILON).then_some((m_vars[index], coefficient))
        })
        .collect();
    (row, rhs_bound)
}

/// Check if a predicate is LP-amenable (pure conjunctions of IntLe).
///
/// The LP encodes conjunctions of linear inequalities over marking
/// variables. Or, Not, and IsFireable require case analysis.
fn is_lp_amenable(pred: &ResolvedPredicate) -> bool {
    match pred {
        ResolvedPredicate::And(children) => children.iter().all(is_lp_amenable),
        ResolvedPredicate::IntLe(_, _) | ResolvedPredicate::True | ResolvedPredicate::False => true,
        ResolvedPredicate::Or(_) | ResolvedPredicate::Not(_) | ResolvedPredicate::IsFireable(_) => {
            false
        }
    }
}

/// Add LP constraints encoding a resolved predicate.
///
/// Returns `false` if the predicate is `False` (trivially infeasible).
fn add_predicate_constraints(
    problem: &mut Problem,
    pred: &ResolvedPredicate,
    m_vars: &[Variable],
) -> bool {
    match pred {
        ResolvedPredicate::And(children) => {
            for child in children {
                if !add_predicate_constraints(problem, child, m_vars) {
                    return false;
                }
            }
            true
        }
        ResolvedPredicate::IntLe(left, right) => {
            let (row, rhs_bound) = build_int_le_constraint(left, right, m_vars);
            problem.add_constraint(&row, ComparisonOp::Le, rhs_bound);
            true
        }
        ResolvedPredicate::True => true,
        ResolvedPredicate::False => false,
        _ => true,
    }
}

fn normalized_transition_indices(
    net: &PetriNet,
    transitions: &[TransitionIdx],
) -> Option<Vec<TransitionIdx>> {
    let mut seen = vec![false; net.num_transitions()];
    let mut normalized = Vec::with_capacity(transitions.len());

    for transition in transitions {
        let index = transition.0 as usize;
        if index >= net.num_transitions() {
            return None;
        }
        if !seen[index] {
            seen[index] = true;
            normalized.push(*transition);
        }
    }

    Some(normalized)
}

fn transition_enabled_predicate(
    net: &PetriNet,
    transition: TransitionIdx,
) -> Option<ResolvedPredicate> {
    let transition = net.transitions.get(transition.0 as usize)?;
    let inputs: Vec<_> = transition
        .inputs
        .iter()
        .filter(|arc| arc.weight > 0)
        .map(|arc| {
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(arc.weight),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(arc.place.0)]),
            )
        })
        .collect();

    match inputs.len() {
        0 => Some(ResolvedPredicate::True),
        1 => inputs.into_iter().next(),
        _ => Some(ResolvedPredicate::And(inputs)),
    }
}

fn input_deficit_predicate(place: PlaceIdx, weight: u64) -> Option<ResolvedPredicate> {
    (weight > 0).then(|| {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(place.0)]),
            ResolvedIntExpr::Constant(weight - 1),
        )
    })
}

/// Sound necessary marking condition for an `IsFireable` disjunction.
///
/// `IsFireable([t1..tn])` holds iff some `t_i` is enabled, i.e.
/// `Or_i AND_p (w_{i,p} <= M[p])` where `w_{i,p}` is `t_i`'s total input demand
/// at place `p`. Two *necessary* (implied) linear conditions are emitted, both
/// implied by whichever disjunct happens to hold:
///
/// 1. **Per-place minimum demand**: for every place `p`,
///    `min_i w_{i,p} <= M[p]`. The enabled `t_{i*}` forces `M[p] >= w_{i*,p}`,
///    which is at least the per-place minimum. Exact for a single transition.
/// 2. **Union-sum demand**: `min_i (sum_p w_{i,p}) <= sum_{p in U} M[p]`, where
///    `U` is the union of all input places. The enabled `t_{i*}` forces
///    `sum_{p in inputs(t_{i*})} M[p] >= total_demand(t_{i*}) >= min_i total`,
///    and `inputs(t_{i*}) ⊆ U` with `M >= 0`. This captures shared-resource
///    infeasibility (e.g. a fixed token budget shared across atoms) that the
///    per-place bound misses when the listed transitions consume disjoint
///    places — exactly the conjunction-of-fireability shape on ring/mutex nets.
///
/// The returned predicate `ψ` therefore satisfies `IsFireable(..) ⟹ ψ` and is a
/// pure conjunction of `IntLe` (LP-amenable). Conjoining several conditions that
/// are each implied by the atom only strengthens the relaxation soundly.
fn fireability_necessary_marking_lower_bounds(
    net: &PetriNet,
    transitions: &[TransitionIdx],
) -> ResolvedPredicate {
    let Some(transitions) = normalized_transition_indices(net, transitions) else {
        // Invalid transition index: cannot soundly constrain, so relax to True
        // (dropping a conjunct only weakens the relaxation, preserving soundness).
        return ResolvedPredicate::True;
    };
    if transitions.is_empty() {
        // `IsFireable([])` enables no transition, so it is never satisfiable.
        return ResolvedPredicate::False;
    }

    let np = net.num_places();
    let mut min_demand = vec![0u64; np];
    let mut in_union = vec![false; np];
    let mut min_total_demand: Option<u64> = None;
    for (idx, transition) in transitions.iter().enumerate() {
        // `transition` is in range (normalized above).
        let t = &net.transitions[transition.0 as usize];
        let mut demand = vec![0u64; np];
        for arc in &t.inputs {
            demand[arc.place.0 as usize] += arc.weight;
        }
        let mut total = 0u64;
        for (place, &d) in demand.iter().enumerate() {
            if d > 0 {
                in_union[place] = true;
            }
            total += d;
        }
        if idx == 0 {
            min_demand = demand;
        } else {
            for (slot, value) in min_demand.iter_mut().zip(demand) {
                *slot = (*slot).min(value);
            }
        }
        min_total_demand = Some(min_total_demand.map_or(total, |m| m.min(total)));
    }

    let mut constraints: Vec<ResolvedPredicate> = Vec::new();
    // (1) Per-place minimum demand.
    for (place, &weight) in min_demand.iter().enumerate() {
        if weight > 0 {
            constraints.push(ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(weight),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(place as u32)]),
            ));
        }
    }
    // (2) Union-sum demand over the union of input places.
    if let Some(total) = min_total_demand {
        let union: Vec<PlaceIdx> = in_union
            .iter()
            .enumerate()
            .filter_map(|(place, &present)| present.then_some(PlaceIdx(place as u32)))
            .collect();
        if total > 0 && !union.is_empty() {
            let sum_constraint = ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(total),
                ResolvedIntExpr::TokensCount(union),
            );
            // For single-place atoms this duplicates the per-place bound.
            if !constraints.contains(&sum_constraint) {
                constraints.push(sum_constraint);
            }
        }
    }

    match constraints.len() {
        0 => ResolvedPredicate::True,
        1 => constraints.into_iter().next().unwrap(),
        _ => ResolvedPredicate::And(constraints),
    }
}

/// Lower an arbitrary predicate to an LP-amenable relaxation `ψ` with `φ ⟹ ψ`.
///
/// Every marking satisfying `φ` also satisfies `ψ`, so proving `ψ` unreachable
/// under the state equation proves `φ` unreachable too — a *sound* `Some(false)`
/// route. The relaxation:
/// - keeps `IntLe`/`True`/`False` and `And` exactly,
/// - replaces each `IsFireable` by its necessary marking lower bounds (exact for
///   single-transition atoms, a sound weakening for disjunctions),
/// - drops `Or`/`Not` to `True` (a disjunction/negation cannot be conjoined as a
///   single LP; dropping it only weakens the relaxation).
///
/// The result is a conjunction of `IntLe`/`True`/`False`, i.e. `is_lp_amenable`
/// holds, so it can be fed straight to `lp_unreachable_with_traps`.
fn lower_for_unreachability(net: &PetriNet, pred: &ResolvedPredicate) -> ResolvedPredicate {
    match pred {
        ResolvedPredicate::And(children) => ResolvedPredicate::And(
            children
                .iter()
                .map(|child| lower_for_unreachability(net, child))
                .collect(),
        ),
        ResolvedPredicate::IntLe(..) | ResolvedPredicate::True | ResolvedPredicate::False => {
            pred.clone()
        }
        ResolvedPredicate::IsFireable(transitions) => {
            fireability_necessary_marking_lower_bounds(net, transitions)
        }
        ResolvedPredicate::Or(_) | ResolvedPredicate::Not(_) => ResolvedPredicate::True,
    }
}

/// True once `deadline` has passed (always false when there is no deadline).
fn deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|limit| Instant::now() >= limit)
}

pub(crate) fn transition_enabled_unreachable(
    net: &PetriNet,
    transition: TransitionIdx,
    traps: &[Vec<bool>],
) -> Option<bool> {
    let predicate = transition_enabled_predicate(net, transition)?;
    Some(lp_unreachable_with_traps_using(net, &predicate, traps))
}

/// SOUND structural quasi-liveness FALSE prover: find the first transition
/// that is LP-provably never enabled in any reachable marking.
///
/// For each transition `t`, the JOINT enabling conjunction
/// `AND_{p in inputs(t)} weight(p, t) <= M[p]` ([`transition_enabled_predicate`])
/// is tested for infeasibility over the state-equation + initially-marked-trap
/// polytope ([`lp_unreachable_with_traps_using`]). Because that polytope is a
/// *superset* of the reachable markings, infeasibility of "t enabled" proves
/// that no reachable marking enables `t` — i.e. `t` is structurally dead and
/// can never fire. A net is quasi-live iff EVERY transition is quasi-live, so a
/// single never-enabled transition proves the net is NOT quasi-live.
///
/// This is strictly stronger than [`crate::structural::lp_dead_transition`],
/// which maximises each input place in ISOLATION over the bare state equation
/// (no traps): it cannot see the *joint* infeasibility "all input places
/// simultaneously >= their weights." Whenever the per-place test proves a
/// transition dead, this joint+trap test does too (the per-place bound is one
/// conjunct of the joint predicate over a tighter polytope), so wiring this in
/// only ever decides MORE transitions, never fewer.
///
/// Returns:
/// - `Some(t)` for the first transition proven never-enabled (a sound FALSE
///   witness for quasi-liveness),
/// - `None` if no transition is proven dead, the net exceeds the LP size
///   guard, or `deadline` elapses mid-sweep. `None` is inconclusive — the
///   caller MUST fall through to its exact engine and must NEVER read it as
///   "quasi-live."
///
/// The initially-marked traps depend only on the net, so they are computed
/// once and reused across every per-transition LP — the dominant cost on
/// transition-heavy nets. `deadline` is polled between transitions so the
/// sweep stays inside its reserved budget.
pub(crate) fn lp_first_dead_transition(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> Option<TransitionIdx> {
    let np = net.num_places();
    let nt = net.num_transitions();
    if np + nt > MAX_LP_VARIABLES {
        return None;
    }

    let traps = find_initially_marked_traps(net);
    for index in 0..nt {
        if deadline_expired(deadline) {
            return None;
        }
        let transition = TransitionIdx(index as u32);
        if transition_enabled_unreachable(net, transition, &traps) == Some(true) {
            return Some(transition);
        }
    }
    None
}

/// Sweep EVERY transition and return the complete set proven never enabled in
/// any reachable marking (LP-infeasible joint enabling conjunction over the
/// state-equation + initially-marked-trap polytope).
///
/// This is the all-transitions companion to [`lp_first_dead_transition`]: that
/// function short-circuits on the first dead transition (it only needs to
/// *witness* non-quasi-liveness), whereas a structural *reduction* needs the
/// full set so it can delete every dead transition at once.
///
/// ## Soundness (one-directional, never a false positive)
///
/// `transition_enabled_unreachable` returns `Some(true)` only when "t enabled"
/// is LP-infeasible over a polytope that is a *superset* of the reachable
/// markings, so every returned transition is genuinely dead (never enabled in
/// any reachable marking). On the size guard (`np + nt > MAX_LP_VARIABLES`) or
/// once `deadline` elapses mid-sweep, the partial set collected so far is
/// returned — still a subset of the truly-dead transitions, never a
/// transition that can actually fire. Dead-transition removal is
/// verdict-preserving for every examination, so an under-approximation only
/// forgoes shrinkage; it can never change a verdict.
#[must_use]
pub(crate) fn lp_all_dead_transitions(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> Vec<TransitionIdx> {
    let np = net.num_places();
    let nt = net.num_transitions();
    if np + nt > MAX_LP_VARIABLES {
        return Vec::new();
    }

    let traps = find_initially_marked_traps(net);
    let mut dead = Vec::new();
    for index in 0..nt {
        if deadline_expired(deadline) {
            break;
        }
        let transition = TransitionIdx(index as u32);
        if transition_enabled_unreachable(net, transition, &traps) == Some(true) {
            dead.push(transition);
        }
    }
    dead
}

fn transition_always_enabled(
    net: &PetriNet,
    transition: TransitionIdx,
    traps: &[Vec<bool>],
) -> Option<bool> {
    let transition = net.transitions.get(transition.0 as usize)?;
    for arc in &transition.inputs {
        let Some(deficit) = input_deficit_predicate(arc.place, arc.weight) else {
            continue;
        };
        if !lp_unreachable_with_traps_using(net, &deficit, traps) {
            return Some(false);
        }
    }
    Some(true)
}

fn transition_disable_blockers(
    net: &PetriNet,
    transition: TransitionIdx,
) -> Option<Vec<ResolvedPredicate>> {
    let transition = net.transitions.get(transition.0 as usize)?;
    let mut blockers = Vec::new();
    for arc in &transition.inputs {
        let Some(blocker) = input_deficit_predicate(arc.place, arc.weight) else {
            continue;
        };
        if !blockers.contains(&blocker) {
            blockers.push(blocker);
        }
    }
    Some(blockers)
}

fn fireability_case_count_within_cap(blockers_by_transition: &[Vec<ResolvedPredicate>]) -> bool {
    let mut cases = 1usize;
    for blockers in blockers_by_transition {
        let Some(next_cases) = cases.checked_mul(blockers.len()) else {
            return false;
        };
        if next_cases > MAX_FIREABILITY_CASE_SPLITS {
            return false;
        }
        cases = next_cases;
    }
    true
}

fn disable_case_unreachable(
    net: &PetriNet,
    blockers_by_transition: &[Vec<ResolvedPredicate>],
    transition_index: usize,
    selected_blockers: &mut Vec<ResolvedPredicate>,
    traps: &[Vec<bool>],
) -> bool {
    if transition_index == blockers_by_transition.len() {
        let predicate = ResolvedPredicate::And(selected_blockers.clone());
        return lp_unreachable_with_traps_using(net, &predicate, traps);
    }

    for blocker in &blockers_by_transition[transition_index] {
        selected_blockers.push(blocker.clone());
        let unreachable = disable_case_unreachable(
            net,
            blockers_by_transition,
            transition_index + 1,
            selected_blockers,
            traps,
        );
        selected_blockers.pop();
        if !unreachable {
            return false;
        }
    }

    true
}

/// Determine whether an `IsFireable` atom is LP-provably true or false.
///
/// Returns:
/// - `Some(false)` when every listed transition is LP-provably never enabled.
/// - `Some(true)` when LP proves no reachable marking can disable all listed
///   transitions.
/// - `None` when the proof would require too many disabled-case splits, an
///   index is invalid, the LP relaxation is inconclusive, or `deadline` elapsed.
///
/// `deadline` bounds the per-transition scan: MCC fireability atoms list
/// hundreds of (unfolded) transitions, and each check is an LP solve, so the
/// scan can run long on big nets. Checking the deadline lets the seeding phase
/// stay inside its reserved budget — bailing returns `None` (unknown), which is
/// verdict-preserving: the formula simply falls through to the exhaustive BFS.
/// The initially-marked traps depend only on the net, so they are computed once
/// here and reused across every per-transition LP check (previously they were
/// recomputed on each call — the dominant cost on transition-heavy nets).
pub(crate) fn lp_fireability_truth(
    net: &PetriNet,
    transitions: &[TransitionIdx],
    deadline: Option<Instant>,
) -> Option<bool> {
    let transitions = normalized_transition_indices(net, transitions)?;
    if transitions.is_empty() {
        return Some(false);
    }

    let traps = if net.num_places() + net.num_transitions() <= MAX_LP_VARIABLES {
        find_initially_marked_traps(net)
    } else {
        Vec::new()
    };

    for transition in &transitions {
        if deadline_expired(deadline) {
            return None;
        }
        if transition_always_enabled(net, *transition, &traps)? {
            return Some(true);
        }
    }

    let mut all_transitions_unreachable = true;
    for transition in &transitions {
        if deadline_expired(deadline) {
            return None;
        }
        if !transition_enabled_unreachable(net, *transition, &traps)? {
            all_transitions_unreachable = false;
            break;
        }
    }
    if all_transitions_unreachable {
        return Some(false);
    }

    let mut blockers_by_transition = Vec::with_capacity(transitions.len());
    for transition in &transitions {
        let blockers = transition_disable_blockers(net, *transition)?;
        if blockers.is_empty() {
            return Some(true);
        }
        blockers_by_transition.push(blockers);
    }

    if !fireability_case_count_within_cap(&blockers_by_transition) {
        return None;
    }

    if deadline_expired(deadline) {
        return None;
    }

    let mut selected_blockers = Vec::with_capacity(blockers_by_transition.len());
    if disable_case_unreachable(
        net,
        &blockers_by_transition,
        0,
        &mut selected_blockers,
        &traps,
    ) {
        Some(true)
    } else {
        None
    }
}

/// Check if a place set forms a trap.
///
/// A trap T satisfies: for every transition t that consumes from T
/// (has an input arc from some place in T), t also produces into T
/// (has an output arc to some place in T). Key property: once marked,
/// a trap stays marked forever.
fn is_trap(net: &PetriNet, places: &[bool]) -> bool {
    for transition in &net.transitions {
        let consumes_from_set = transition
            .inputs
            .iter()
            .any(|arc| places[arc.place.0 as usize]);
        if consumes_from_set
            && !transition
                .outputs
                .iter()
                .any(|arc| places[arc.place.0 as usize])
        {
            return false;
        }
    }
    true
}

/// Compute the siphon closure of an initial place set.
///
/// A siphon S satisfies: every transition producing into S also consumes
/// from S. This function grows the initial set until the siphon property
/// holds by adding input places of transitions that produce into the set
/// but do not yet consume from it.
fn siphon_closure(net: &PetriNet, initial: &[bool]) -> Vec<bool> {
    let mut set = initial.to_vec();
    loop {
        let mut changed = false;
        for transition in &net.transitions {
            let produces_into_set = transition
                .outputs
                .iter()
                .any(|arc| set[arc.place.0 as usize]);
            if !produces_into_set {
                continue;
            }

            let consumes_from_set = transition
                .inputs
                .iter()
                .any(|arc| set[arc.place.0 as usize]);
            if consumes_from_set {
                continue;
            }

            for arc in &transition.inputs {
                let place = arc.place.0 as usize;
                if !set[place] {
                    set[place] = true;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
    set
}

/// Extract the maximal trap contained within a siphon.
///
/// Every siphon contains a (possibly empty) maximal trap. This function
/// iteratively removes places that violate the trap property until a
/// fixed point is reached. The result is guaranteed to be a trap (or empty).
fn maximal_trap_within_siphon(net: &PetriNet, siphon: &[bool]) -> Vec<bool> {
    let mut trap_candidate = siphon.to_vec();
    loop {
        let mut changed = false;
        for place in 0..trap_candidate.len() {
            if !trap_candidate[place] {
                continue;
            }

            let place_is_valid = net.transitions.iter().all(|transition| {
                let consumes_place = transition
                    .inputs
                    .iter()
                    .any(|arc| arc.place.0 as usize == place);
                !consumes_place
                    || transition
                        .outputs
                        .iter()
                        .any(|arc| trap_candidate[arc.place.0 as usize])
            });

            if !place_is_valid {
                trap_candidate[place] = false;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
    debug_assert!(!trap_candidate.iter().any(|&in_trap| in_trap) || is_trap(net, &trap_candidate));
    trap_candidate
}

/// Minimize a trap by greedily removing places.
///
/// Tries to remove each place one at a time; if the remaining set is
/// still a non-empty trap, the removal is kept. Produces a minimal
/// (not necessarily minimum) trap.
fn minimize_trap(net: &PetriNet, trap: &[bool]) -> Vec<bool> {
    let mut result = trap.to_vec();
    loop {
        let mut changed = false;
        for place in 0..result.len() {
            if !result[place] {
                continue;
            }

            result[place] = false;
            if result.iter().any(|&in_trap| in_trap) && is_trap(net, &result) {
                changed = true;
            } else {
                result[place] = true;
            }
        }

        if !changed {
            break;
        }
    }
    result
}

/// Enumerate all distinct minimal traps that are initially marked.
///
/// For each place, computes the siphon closure, extracts the maximal
/// trap within, minimizes it, and keeps it if at least one place in
/// the trap has tokens in the initial marking. Deduplicates results.
/// These traps generate valid LP constraints: sum of marking >= 1.
pub(crate) fn find_initially_marked_traps(net: &PetriNet) -> Vec<Vec<bool>> {
    let num_places = net.num_places();
    let mut traps = Vec::new();

    for seed_place in 0..num_places {
        let mut initial = vec![false; num_places];
        initial[seed_place] = true;

        let closure = siphon_closure(net, &initial);
        if !closure.iter().any(|&in_set| in_set) {
            continue;
        }

        let maximal_trap = maximal_trap_within_siphon(net, &closure);
        if !maximal_trap.iter().any(|&in_set| in_set) {
            continue;
        }

        let minimal_trap = minimize_trap(net, &maximal_trap);
        if !minimal_trap.iter().any(|&in_set| in_set) {
            continue;
        }

        let initially_marked = minimal_trap
            .iter()
            .enumerate()
            .any(|(place, &in_trap)| in_trap && net.initial_marking[place] > 0);
        if !initially_marked {
            continue;
        }

        if !traps.iter().any(|trap| trap == &minimal_trap) {
            traps.push(minimal_trap);
        }
    }

    traps
}

/// Extract the maximal siphon contained within a given place set `allowed`.
///
/// A siphon S satisfies the DUAL condition to a trap: every transition that
/// *produces* into S (`t• ∩ S ≠ ∅`) also *consumes* from S (`•t ∩ S ≠ ∅`).
/// Starting from `S = allowed`, this iteratively removes the output places (that
/// lie in `S`) of any transition that produces into `S` without consuming from
/// it, until a fixed point. The result is the unique maximal siphon `⊆ allowed`
/// (siphons are union-closed, so a maximal siphon within any set is well-defined
/// and contains every siphon `⊆ allowed`). It is the exact structural dual of
/// [`maximal_trap_within_siphon`].
fn maximal_siphon_within(net: &PetriNet, allowed: &[bool]) -> Vec<bool> {
    let mut siphon_candidate = allowed.to_vec();
    loop {
        let mut changed = false;
        for place in 0..siphon_candidate.len() {
            if !siphon_candidate[place] {
                continue;
            }
            // `place` may stay only if EVERY transition producing into it also
            // consumes from the current set — i.e. there is no transition `t`
            // with `place ∈ t•`, `t• ∩ S ≠ ∅` (trivially true via `place`), and
            // `•t ∩ S = ∅`. Equivalently: no producer of `place` consumes
            // nothing from `S`.
            let place_is_valid = net.transitions.iter().all(|transition| {
                let produces_place = transition
                    .outputs
                    .iter()
                    .any(|arc| arc.place.0 as usize == place);
                !produces_place
                    || transition
                        .inputs
                        .iter()
                        .any(|arc| siphon_candidate[arc.place.0 as usize])
            });
            if !place_is_valid {
                siphon_candidate[place] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    siphon_candidate
}

/// Enumerate distinct initially-UNMARKED siphons of the net.
///
/// # The dual-of-trap invariant
///
/// A *siphon* `S` is a place set such that every transition producing into `S`
/// also consumes from `S`. Its defining structural property is the DUAL of a
/// trap: **once a siphon is empty it stays empty.** Concretely, if `M0(S) = 0`
/// (no member of `S` is initially marked), then `Σ_{p∈S} M[p] = 0` — equivalently
/// `M[p] = 0` for every `p ∈ S` — in *every* reachable marking. (Proof: a
/// transition can only add tokens to `S` by producing into it, but every such
/// transition also consumes from `S`; if `S` is empty that transition is dead, so
/// no firing can ever re-mark `S`.)
///
/// This is exactly the cut needed to tighten the integer/LP state equation: an
/// initially-unmarked siphon pins all its places to zero, ruling out the spurious
/// state-equation solutions in which the siphon spuriously gains tokens.
///
/// # What is returned
///
/// For each initially-unmarked seed place we close it under [`siphon_closure`]
/// (the smallest siphon containing the seed) and keep that siphon *only* when it
/// is entirely initially-unmarked. We also include the single maximal siphon
/// contained in the set of all initially-unmarked places (the strongest such
/// cut). Every returned siphon is guaranteed initially unmarked (gated by
/// [`siphon_is_initially_unmarked`]), so the derived `M[p] = 0` constraints are
/// SOUND invariants. Results are de-duplicated. An empty result (no nonempty
/// initially-unmarked siphon) is normal and simply adds no cuts.
pub(crate) fn find_initially_unmarked_siphons(net: &PetriNet) -> Vec<Vec<bool>> {
    let num_places = net.num_places();
    let mut siphons: Vec<Vec<bool>> = Vec::new();

    // SOUNDNESS GATE: only accept a candidate that is (a) non-empty, (b) a GENUINE
    // siphon, and (c) entirely initially unmarked. (b) is essential — `siphon_*`
    // helpers can return a non-siphon (see [`is_siphon`]'s source-transition case);
    // pinning a non-siphon's places to zero would be UNSOUND. Verifying every
    // returned set here makes the function's zero-cut contract hold regardless of
    // any helper imprecision.
    let mut push_if_valid = |siphon: Vec<bool>| {
        if siphon.iter().any(|&in_set| in_set)
            && is_siphon(net, &siphon)
            && siphon_is_initially_unmarked(net, &siphon)
            && !siphons.iter().any(|s| s == &siphon)
        {
            siphons.push(siphon);
        }
    };

    // (1) The maximal siphon contained in the initially-unmarked places. This is
    //     the single strongest unmarked-siphon cut: it pins the largest set of
    //     places to zero. Any member is initially unmarked by construction.
    let unmarked: Vec<bool> = (0..num_places)
        .map(|p| net.initial_marking[p] == 0)
        .collect();
    let maximal = maximal_siphon_within(net, &unmarked);
    push_if_valid(maximal);

    // (2) Per-seed closure siphons that happen to be initially unmarked. The
    //     closure of a single place is the smallest siphon containing it; when it
    //     is fully unmarked it gives a (generally smaller, sometimes distinct)
    //     sound cut the maximal one may not subsume member-for-member. The closure
    //     is NOT guaranteed to be a real siphon (a source-fed seed yields a
    //     non-siphon), so `push_if_valid`'s `is_siphon` check is the gate that
    //     drops such spurious candidates.
    for seed_place in 0..num_places {
        if net.initial_marking[seed_place] != 0 {
            continue;
        }
        let mut initial = vec![false; num_places];
        initial[seed_place] = true;
        let closure = siphon_closure(net, &initial);
        push_if_valid(closure);
    }

    siphons
}

/// True iff NO place of `siphon` holds a token in the initial marking.
///
/// SOUNDNESS GATE for siphon zero-cuts: the invariant `M[p] = 0 for all p ∈ S`
/// (equivalently `Σ_{p∈S} M[p] = 0`) is only a valid reachability-preserving
/// constraint when `S` is *initially unmarked* — a siphon, once empty, stays
/// empty forever, so an initially-unmarked siphon can never gain a token. A
/// siphon that IS initially marked may legitimately hold tokens in reachable
/// markings, so its zero-cut would be unsound. Every siphon cut is gated here.
fn siphon_is_initially_unmarked(net: &PetriNet, siphon: &[bool]) -> bool {
    siphon
        .iter()
        .enumerate()
        .all(|(place, &in_siphon)| !in_siphon || net.initial_marking[place] == 0)
}

/// True iff `S` is a genuine SIPHON: every transition that produces into `S`
/// also consumes from `S`.
///
/// This is the DEFINITIONAL check, and it is the load-bearing SOUNDNESS GATE for
/// every initially-unmarked-siphon zero-cut: the "once empty, stays empty"
/// theorem (hence the invariant `Σ_{p∈S} M[p] = 0`) holds ONLY for real siphons.
///
/// It is needed because [`siphon_closure`] only ever *adds* places and can return
/// a NON-siphon when a producer of a member consumes from NOTHING (e.g. a *source*
/// transition `t: ∅ → p`): such a transition produces into `S` without consuming
/// from `S`, so `S` is not a siphon, yet the closure has no input place to pull in
/// and leaves the candidate unchanged. Without this gate a source-fed,
/// initially-unmarked place `p` would be mis-reported as a siphon and pinned to
/// `M[p] = 0` — an UNSOUND cut, since `p` is in fact unbounded. (Empty `S` is
/// vacuously a siphon; callers separately require a non-empty member set.)
fn is_siphon(net: &PetriNet, siphon: &[bool]) -> bool {
    net.transitions.iter().all(|transition| {
        let produces_into_set = transition
            .outputs
            .iter()
            .any(|arc| siphon[arc.place.0 as usize]);
        let consumes_from_set = transition
            .inputs
            .iter()
            .any(|arc| siphon[arc.place.0 as usize]);
        !produces_into_set || consumes_from_set
    })
}

/// True iff some place of `trap` holds a token in the initial marking.
///
/// SOUNDNESS GATE for trap cuts: the invariant `sum_{p in T} M >= 1` is only a
/// valid (reachability-preserving) constraint when `T` is *initially marked* — a
/// trap, once marked, stays marked forever, so an initially-marked trap can never
/// be emptied by any firing sequence. A trap that is NOT initially marked may be
/// legitimately empty in reachable markings, so its cut would be unsound. Every
/// cut added by the CEGAR loop is gated through this predicate.
fn trap_is_initially_marked(net: &PetriNet, trap: &[bool]) -> bool {
    trap.iter()
        .enumerate()
        .any(|(place, &in_trap)| in_trap && net.initial_marking[place] > 0)
}

/// Sum of the LP marking variables over the places of `trap` at the current `M*`.
fn trap_marking_sum(trap: &[bool], solution: &Solution, m_vars: &[Variable]) -> f64 {
    trap.iter()
        .enumerate()
        .filter_map(|(place, &in_trap)| in_trap.then_some(*solution.var_value(m_vars[place])))
        .sum()
}

/// Build the LP row `sum_{p in trap} M[p]` for the trap invariant `... >= 1`.
fn trap_cut_row(trap: &[bool], m_vars: &[Variable]) -> Vec<(Variable, f64)> {
    trap.iter()
        .enumerate()
        .filter_map(|(place, &in_trap)| in_trap.then_some((m_vars[place], 1.0)))
        .collect()
}

/// Derive a SOUND separating trap that the current LP vertex `M*` violates, if
/// one exists.
///
/// The separating object is the *maximal trap contained in*
/// `Q = {p : M*[p] < TRAP_EMPTY_EPS}` (the empty support of `M*`). Justification:
/// traps are union-closed, so the maximal trap inside `Q` contains every trap
/// `⊆ Q`; an initially-marked trap that is empty at `M*` exists *iff* this maximal
/// trap is non-empty and initially marked. [`maximal_trap_within_siphon`] already
/// computes the maximal trap inside an arbitrary place set (it never uses the
/// siphon property), so feeding it `Q` reuses that routine directly.
///
/// Returns `Some(trap)` only when the result is BOTH a real trap and initially
/// marked — i.e. a sound cut `sum_{p in trap} M >= 1`. Because every place of the
/// returned trap is in `Q` (the routine only removes places from `Q`), the cut
/// strictly excludes `M*` (`sum_{p in trap} M* < |trap| * TRAP_EMPTY_EPS < 1`),
/// guaranteeing progress. Returns `None` when no initially-marked separating trap
/// exists at this vertex — the inconclusive signal.
///
/// An optional minimization (smaller cut, fewer non-zeros) is applied *only* when
/// it preserves the initially-marked property; otherwise the maximal trap is
/// kept. This mirrors the existing precomputed pass, which checks
/// initially-marked AFTER minimizing, and never lets [`minimize_trap`] drop the
/// initially-marked place out from under the soundness gate.
fn separating_trap_for_vertex(
    net: &PetriNet,
    solution: &Solution,
    m_vars: &[Variable],
) -> Option<Vec<bool>> {
    let np = net.num_places();
    let empty_support: Vec<bool> = (0..np)
        .map(|place| *solution.var_value(m_vars[place]) < TRAP_EMPTY_EPS)
        .collect();

    let maximal = maximal_trap_within_siphon(net, &empty_support);
    if !maximal.iter().any(|&in_trap| in_trap) {
        return None;
    }
    if !trap_is_initially_marked(net, &maximal) {
        return None;
    }

    // The maximal trap is already a sound, initially-marked, separating cut.
    // Try to shrink it for a tighter constraint, but only adopt the smaller
    // trap if it remains initially marked (so the cut stays sound).
    let minimized = minimize_trap(net, &maximal);
    if minimized.iter().any(|&in_trap| in_trap) && trap_is_initially_marked(net, &minimized) {
        Some(minimized)
    } else {
        Some(maximal)
    }
}

/// Check unreachability using the LP state equation with iterative trap tightening.
///
/// Solves M = M0 + C*x with the predicate constraints, then iteratively
/// adds trap invariants (sum of marking in trap >= 1) for initially-marked
/// traps whose LP solution violates the invariant. If adding these
/// constraints makes the LP infeasible, the predicate marking is provably
/// unreachable. This is a key SMPT technique that strengthens the basic
/// state equation overapproximation.
pub(crate) fn lp_unreachable_with_traps(net: &PetriNet, predicate: &ResolvedPredicate) -> bool {
    lp_unreachable_with_traps_deadline(net, predicate, None)
}

/// Same as [`lp_unreachable_with_traps`] but bounded by a wall-clock `deadline`.
///
/// Computes the net's initially-marked traps once, then runs the CEGAR loop
/// (see [`lp_unreachable_with_traps_using`]) with the deadline threaded in so the
/// per-vertex trap refinement stays inside the seeding budget. On expiry the loop
/// returns `false` (inconclusive) and the query falls through to BFS.
pub(crate) fn lp_unreachable_with_traps_deadline(
    net: &PetriNet,
    predicate: &ResolvedPredicate,
    deadline: Option<Instant>,
) -> bool {
    if !is_lp_amenable(predicate) {
        return false;
    }

    let np = net.num_places();
    let nt = net.num_transitions();

    if np + nt > MAX_LP_VARIABLES {
        return false;
    }

    let traps = find_initially_marked_traps(net);
    lp_unreachable_with_traps_using_deadline(net, predicate, &traps, deadline)
}

/// Same as [`lp_unreachable_with_traps`] but reuses a precomputed trap set.
///
/// Callers that probe many predicates against the *same* net (the traps depend
/// only on the net) compute [`find_initially_marked_traps`] once and pass it in.
/// Assumes the amenability and size guards already passed.
pub(crate) fn lp_unreachable_with_traps_using(
    net: &PetriNet,
    predicate: &ResolvedPredicate,
    traps: &[Vec<bool>],
) -> bool {
    lp_unreachable_with_traps_using_deadline(net, predicate, traps, None)
}

/// Iterative trap-CEGAR core of the state-equation lane (reuses a precomputed
/// trap set, bounded by a wall-clock `deadline`).
///
/// Solves the state-equation polytope `M = M0 + C*x, x >= 0, M >= 0` under the
/// predicate constraints, then *refines* it with trap invariants until it is
/// either proven infeasible (unreachable) or no further sound refinement exists:
///
/// 1. **First solve.** `Infeasible` ⇒ `true` (genuinely unreachable). `Unbounded`
///    ⇒ `false` (inconclusive). A feasible `M*` is NEVER a reachability proof — it
///    is only a candidate the LP cannot yet rule out.
/// 2. **Fast path (a).** Add every *precomputed* trap (`traps`) whose marking sum
///    at `M*` is `< 0.5`, incrementally (dual-simplex warm start). These are the
///    cheap, net-only minimal traps; adding them all per iteration matches the old
///    behaviour.
/// 3. **Separation oracle (b).** When no precomputed trap is violated, derive a
///    FRESH separating trap — the maximal initially-marked trap empty at this `M*`
///    ([`separating_trap_for_vertex`]). This is the genuine CEGAR step: it targets
///    the *current* vertex rather than a fixed seed set, so it cuts vertices the
///    single-shot pass leaves standing. No separating trap ⇒ `false` (inconclusive,
///    fall through to BFS).
/// 4. Repeat. Each added cut `sum_{p in T} M >= 1` strictly excludes the current
///    `M*` (every place of `T` has `M*[p] ≈ 0`), so the loop makes monotone
///    progress and cannot re-derive an already-added cut (the re-solved vertex
///    satisfies all prior cuts). `MAX_CEGAR_ITERS` and `deadline` bound the work.
///
/// SOUNDNESS: the *only* `true` return is `Infeasible` after adding cuts that are
/// every one a SOUND invariant — an initially-marked trap stays marked, so
/// `sum_{p in T} M >= 1` holds in every reachable marking (gated through
/// [`trap_is_initially_marked`]). A feasible/SAT `M*` is never a verdict; the
/// inconclusive paths (`Unbounded`, no separating trap, iter cap, deadline) all
/// return `false`, leaving the verdict to the exhaustive engine.
pub(crate) fn lp_unreachable_with_traps_using_deadline(
    net: &PetriNet,
    predicate: &ResolvedPredicate,
    traps: &[Vec<bool>],
    deadline: Option<Instant>,
) -> bool {
    if !is_lp_amenable(predicate) {
        return false;
    }

    let np = net.num_places();
    let nt = net.num_transitions();

    if np + nt > MAX_LP_VARIABLES {
        return false;
    }

    let mut problem = Problem::new(OptimizationDirection::Minimize);

    let x_vars: Vec<_> = (0..nt)
        .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
        .collect();
    let m_vars: Vec<_> = (0..np)
        .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
        .collect();

    add_state_equation(&mut problem, net, &x_vars, &m_vars);

    if !add_predicate_constraints(&mut problem, predicate, &m_vars) {
        return true;
    }

    let solution = match problem.solve() {
        Ok(solution) => solution,
        Err(minilp::Error::Infeasible) => return true,
        Err(minilp::Error::Unbounded) => return false,
    };

    cegar_refute(net, solution, &m_vars, traps, deadline)
}

/// Iterative trap-CEGAR refinement core, shared by every state-equation prover.
///
/// Given a *feasible* first `solution` to `M = M0 + C*x` plus arbitrary problem
/// constraints (the predicate, or the negated inequality for the always-true
/// route), refines it with SOUND trap invariants until the polytope is driven
/// infeasible (⇒ `true`, the target marking is genuinely unreachable) or no
/// further sound refinement applies (⇒ `false`, inconclusive — fall through):
///
/// - **Fast path (a).** Add every *precomputed* trap (`traps`) whose marking sum
///   at the current `M*` is `< 0.5`, incrementally (dual-simplex warm start).
/// - **Separation oracle (b).** When no precomputed trap is violated, derive a
///   FRESH separating trap — the maximal initially-marked trap empty at `M*`
///   ([`separating_trap_for_vertex`]). This is the genuine CEGAR step: it targets
///   the *current* vertex rather than a fixed seed set. No separating trap ⇒
///   inconclusive.
///
/// Each added cut `sum_{p in T} M >= 1` strictly excludes the current `M*` (every
/// place of `T` has `M*[p] ≈ 0`), so the loop makes monotone progress and cannot
/// re-derive an already-added cut. `MAX_CEGAR_ITERS` and `deadline` bound the
/// work; both expiry paths return `false`.
///
/// SOUNDNESS: the only `true` return is `Infeasible` after adding cuts that are
/// each a SOUND invariant — an initially-marked trap stays marked, so
/// `sum_{p in T} M >= 1` holds in every reachable marking (gated through
/// [`trap_is_initially_marked`]). A feasible/SAT `M*` is NEVER a verdict.
fn cegar_refute(
    net: &PetriNet,
    mut solution: Solution,
    m_vars: &[Variable],
    traps: &[Vec<bool>],
    deadline: Option<Instant>,
) -> bool {
    let mut added_constraints = vec![false; traps.len()];
    for _ in 0..MAX_CEGAR_ITERS {
        if deadline_expired(deadline) {
            // Out of budget: inconclusive. A feasible LP is not a proof, so we
            // must NOT return a verdict here — fall through to BFS.
            return false;
        }

        // (a) Fast path: precomputed-trap violations, warm-started incrementally.
        let violated_traps: Vec<_> = traps
            .iter()
            .enumerate()
            .filter_map(|(trap_idx, trap)| {
                if added_constraints[trap_idx] {
                    return None;
                }
                (trap_marking_sum(trap, &solution, m_vars) < 0.5).then_some(trap_idx)
            })
            .collect();

        if !violated_traps.is_empty() {
            for trap_idx in violated_traps {
                added_constraints[trap_idx] = true;
                solution = match solution.add_constraint(
                    trap_cut_row(&traps[trap_idx], m_vars),
                    ComparisonOp::Ge,
                    1.0,
                ) {
                    Ok(solution) => solution,
                    Err(minilp::Error::Infeasible) => return true,
                    Err(minilp::Error::Unbounded) => return false,
                };
            }
            continue;
        }

        // (b) CEGAR separation oracle: derive a FRESH initially-marked trap that
        // is empty at the current `M*`. Absent one, the polytope admits this
        // vertex with no sound refinement left — inconclusive.
        let Some(separating) = separating_trap_for_vertex(net, &solution, m_vars) else {
            return false;
        };

        solution =
            match solution.add_constraint(trap_cut_row(&separating, m_vars), ComparisonOp::Ge, 1.0)
            {
                Ok(solution) => solution,
                Err(minilp::Error::Infeasible) => return true,
                Err(minilp::Error::Unbounded) => return false,
            };
    }

    false
}

/// Find a place LP-provably pinned to its initial token count.
///
/// Returns `Some(p)` when the state-equation polytope — a superset of the
/// reachable markings, tightened with initially-marked trap invariants — admits
/// no marking with `M[p] > M0[p]` and none with `M[p] < M0[p]`. Because the
/// polytope over-approximates reachability, a place pinned there is constant in
/// *every* reachable marking: a stable place that proves `StableMarking` TRUE.
///
/// This is the sound state-equation/P-invariant dual of the structural
/// zero-incidence-row constant-place test: it additionally catches places kept
/// constant by a P-invariant or by a transition the state equation forces never
/// to fire, even though some transition has a nonzero incidence row on `p`.
///
/// Returns `None` if no place is provably pinned, the net is too large, or the
/// `deadline` elapses (checked between places). It never reports a place that is
/// not provably constant, so it can only ever justify a sound TRUE verdict.
pub(crate) fn lp_pinned_place(net: &PetriNet, deadline: Option<Instant>) -> Option<PlaceIdx> {
    let np = net.num_places();
    let nt = net.num_transitions();
    // Tighter cap than `MAX_LP_VARIABLES`: this sweeps up to `2 * np` LP solves,
    // so keep each solve small enough that the whole sweep stays a cheap
    // pre-pass. Larger nets are left to the existing BMC/PDR/BFS engines.
    if np == 0 || np + nt > MAX_PINNING_LP_VARIABLES {
        return None;
    }

    // Traps depend only on the net; compute once and reuse for every place.
    let traps = find_initially_marked_traps(net);

    for place in 0..np {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            break;
        }
        let m0 = net.initial_marking[place];
        let tokens = ResolvedIntExpr::TokensCount(vec![PlaceIdx(place as u32)]);

        // (1) `M[p] >= M0[p] + 1` must be unreachable: `p` can never exceed `M0`.
        let above = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(m0.saturating_add(1)),
            tokens.clone(),
        );
        if !lp_unreachable_with_traps_using(net, &above, &traps) {
            continue;
        }

        // (2) `M[p] <= M0[p] - 1` must be unreachable: `p` can never fall below
        // `M0`. For `M0[p] == 0` this is `M[p] <= -1`, vacuously unreachable since
        // markings are non-negative, so no LP solve is needed.
        if m0 > 0 {
            let below = ResolvedPredicate::IntLe(tokens, ResolvedIntExpr::Constant(m0 - 1));
            if !lp_unreachable_with_traps_using(net, &below, &traps) {
                continue;
            }
        }

        return Some(PlaceIdx(place as u32));
    }

    None
}

/// Check if a target predicate is LP-provably false under the state equation.
///
/// Returns `true` if the LP relaxation proves no reachable marking can satisfy
/// the predicate. Supports pure marking predicates directly, plus conservative
/// fireability proofs for `IsFireable` atoms and simple boolean structure.
///
/// Returns `false` if the LP is feasible (inconclusive) or if the
/// predicate is not decisively provable.
pub(crate) fn lp_unreachable(net: &PetriNet, predicate: &ResolvedPredicate) -> bool {
    lp_unreachable_deadline(net, predicate, None)
}

/// Deadline-bounded variant of [`lp_unreachable`].
///
/// On expiry the LP returns `None`, so this returns `false` (inconclusive) —
/// the predicate is left un-folded, which is verdict-preserving.
pub(crate) fn lp_unreachable_deadline(
    net: &PetriNet,
    predicate: &ResolvedPredicate,
    deadline: Option<Instant>,
) -> bool {
    lp_predicate_truth(net, predicate, deadline) == Some(false)
}

/// Check whether the strict inequality `lhs > rhs` is LP-infeasible.
///
/// Over the integer Petri-net domain, `lhs > rhs` is equivalent to
/// `rhs + 1 <= lhs`. If the LP relaxation is infeasible under that
/// constraint, no reachable marking can violate `lhs <= rhs`.
///
/// NOTE: this always-true route is deliberately kept on the *bare* state
/// equation (no trap refinement). Threading the trap-CEGAR machinery here
/// regressed the cited large nets (`SharedMemory-PT-000020`, `TokenRing-PT-010`):
/// it doubled the per-`IntLe` `find_initially_marked_traps` cost in the hot
/// reachability seeding path, starving the shared deadline so borderline
/// formulas that previously finished in BFS timed out — with no offsetting gain.
/// The trap CEGAR therefore lives only in the false/unreachable lane
/// ([`lp_unreachable_with_traps_using_deadline`]).
pub(crate) fn lp_strictly_greater_unreachable(
    net: &PetriNet,
    lhs: &ResolvedIntExpr,
    rhs: &ResolvedIntExpr,
) -> bool {
    let np = net.num_places();
    let nt = net.num_transitions();

    if np + nt > MAX_LP_VARIABLES {
        return false;
    }

    let mut problem = Problem::new(OptimizationDirection::Minimize);

    let x_vars: Vec<_> = (0..nt)
        .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
        .collect();
    let m_vars: Vec<_> = (0..np)
        .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
        .collect();

    add_state_equation(&mut problem, net, &x_vars, &m_vars);

    let (row, rhs_bound) = build_int_le_constraint(rhs, lhs, &m_vars);
    problem.add_constraint(&row, ComparisonOp::Le, rhs_bound - 1.0);

    matches!(problem.solve(), Err(minilp::Error::Infeasible))
}

/// Compute an LP upper bound on the sum of tokens at specified places.
///
/// Solves: maximize sum(M[p] for p in places)
/// subject to: M = M0 + C*x, x >= 0, M >= 0.
///
/// Returns `Some(bound)` if the LP has a finite optimum, `None` if
/// unbounded or the net is too large.
pub(crate) fn lp_upper_bound(net: &PetriNet, places: &[PlaceIdx]) -> Option<u64> {
    if places.is_empty() {
        return Some(0);
    }

    let np = net.num_places();
    let nt = net.num_transitions();

    if np + nt > MAX_LP_VARIABLES {
        return None;
    }

    let mut problem = Problem::new(OptimizationDirection::Maximize);

    let x_vars: Vec<_> = (0..nt)
        .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
        .collect();

    // Objective coefficients: multiplicity-aware for repeated places.
    let mut obj_coeffs = vec![0.0_f64; np];
    for place in places {
        obj_coeffs[place.0 as usize] += 1.0;
    }
    let m_vars: Vec<_> = (0..np)
        .map(|p| problem.add_var(obj_coeffs[p], (0.0, f64::INFINITY)))
        .collect();

    add_state_equation(&mut problem, net, &x_vars, &m_vars);

    match problem.solve() {
        Ok(solution) => lp_objective_to_bound(solution.objective()),
        // Infeasible should never occur (x=0, m=m0 is always feasible for
        // valid Petri nets). If minilp reports it, treat as numerical
        // instability → unknown. Returning Some(0) here was unsound: it
        // overrode correct P-invariant bounds (NQueens-PT-30 #1501).
        Err(minilp::Error::Infeasible) => None,
        Err(minilp::Error::Unbounded) => None,
    }
}

fn lp_objective_to_bound(obj: f64) -> Option<u64> {
    if !obj.is_finite() {
        return None;
    }

    // minilp may return a very large finite value instead of Unbounded for
    // effectively unconstrained problems. Treat values above 1e15 as
    // unbounded (no Petri net in MCC has token counts anywhere near this).
    if obj > 1e15 {
        return None;
    }

    // ceil() is sound: ceil(LP_max) >= LP_max >= true_max. floor() is
    // theoretically tighter (floor(1.5)=1 is correct since LP
    // over-approximates), but floating-point error in the simplex solver can
    // push a true-integer optimum below the integer boundary (e.g., 0.9999
    // instead of 1.0), and floor(0.9999) = 0 is unsound. This caused wrong
    // answers on NQueens-PT-30 and IBM5964-PT-none. (#1501)
    Some(obj.ceil() as u64)
}

/// Determine the truth value of a resolved predicate via LP.
///
/// Returns:
/// - `Some(false)` if LP proves the predicate is unreachable
/// - `Some(true)` if LP proves the predicate always holds
/// - `None` if LP is inconclusive or the required case split is too large
///
/// This function never treats LP feasibility as reachability. Feasible cases
/// remain unresolved unless another proof path is decisive.
///
/// `deadline` bounds the (potentially many) LP solves a single fireability-heavy
/// formula triggers, so the seeding phase stays within its reserved budget;
/// bailing returns `None` (unknown), which is verdict-preserving — the formula
/// falls through to the exhaustive BFS.
pub(crate) fn lp_predicate_truth(
    net: &PetriNet,
    predicate: &ResolvedPredicate,
    deadline: Option<Instant>,
) -> Option<bool> {
    match predicate {
        ResolvedPredicate::True => Some(true),
        ResolvedPredicate::False => Some(false),
        ResolvedPredicate::IntLe(lhs, rhs) => {
            let singleton = ResolvedPredicate::IntLe(lhs.clone(), rhs.clone());
            if lp_unreachable_with_traps_deadline(net, &singleton, deadline) {
                return Some(false);
            }
            if lp_strictly_greater_unreachable(net, lhs, rhs) {
                return Some(true);
            }
            None
        }
        ResolvedPredicate::IsFireable(transitions) => {
            lp_fireability_truth(net, transitions, deadline)
        }
        ResolvedPredicate::And(children) => {
            // Joint state-equation + trap LP on the LP-amenable relaxation `ψ` of
            // `φ` (with `φ ⟹ ψ`): if `ψ` is unreachable, so is `φ`. For a pure
            // `IntLe` conjunction `ψ == φ` (the Cardinality path, unchanged);
            // fireability conjuncts are relaxed to their necessary marking lower
            // bounds, so the same proven joint LP can now decide `EF/AG` over
            // fireability conjunctions instead of bailing on the disjunctive atom.
            // This front-check is one bounded LP, run before the costlier
            // per-child fireability recursion.
            let lowered = lower_for_unreachability(net, predicate);
            if lp_unreachable_with_traps_deadline(net, &lowered, deadline) {
                return Some(false);
            }

            let mut all_true = true;
            for child in children {
                if deadline_expired(deadline) {
                    return None;
                }
                match lp_predicate_truth(net, child, deadline) {
                    Some(false) => return Some(false),
                    Some(true) => {}
                    None => all_true = false,
                }
            }

            all_true.then_some(true)
        }
        ResolvedPredicate::Or(children) => {
            let mut all_false = true;
            for child in children {
                if deadline_expired(deadline) {
                    return None;
                }
                match lp_predicate_truth(net, child, deadline) {
                    Some(true) => return Some(true),
                    Some(false) => {}
                    None => all_false = false,
                }
            }

            all_false.then_some(false)
        }
        ResolvedPredicate::Not(inner) => {
            lp_predicate_truth(net, inner, deadline).map(|truth| !truth)
        }
    }
}

/// Determine the truth value of an individual resolved predicate atom via LP.
///
/// Keeps the old atom-level entry point for the formula simplifier cache.
pub(crate) fn lp_atom_truth(net: &PetriNet, atom: &ResolvedPredicate) -> Option<bool> {
    lp_atom_truth_deadline(net, atom, None)
}

/// Deadline-bounded variant of [`lp_atom_truth`].
///
/// Threads `deadline` into the underlying state-equation/trap CEGAR LP so the
/// formula-simplifier pre-pass cannot overrun the global wall budget. On expiry
/// the LP returns `None` (inconclusive), which is verdict-preserving: the atom
/// is left unresolved and falls through to the exhaustive temporal engine.
pub(crate) fn lp_atom_truth_deadline(
    net: &PetriNet,
    atom: &ResolvedPredicate,
    deadline: Option<Instant>,
) -> Option<bool> {
    match atom {
        ResolvedPredicate::IntLe(..)
        | ResolvedPredicate::IsFireable(..)
        | ResolvedPredicate::True
        | ResolvedPredicate::False => lp_predicate_truth(net, atom, deadline),
        _ => None,
    }
}

/// Check if a predicate holds in ALL reachable markings using LP.
///
/// Returns `true` when every LP-checkable violating branch is infeasible,
/// meaning the predicate can never be violated. Fireability atoms are handled
/// only when LP/P-invariant reasoning is decisive; capped case splits that do
/// not finish return `false` (unknown), not a guessed verdict.
///
/// This is the shared dual of [`lp_unreachable`]: where `lp_unreachable`
/// proves `φ` NEVER holds, this proves `φ` ALWAYS holds.
pub(crate) fn lp_always_true(net: &PetriNet, predicate: &ResolvedPredicate) -> bool {
    lp_always_true_deadline(net, predicate, None)
}

/// Deadline-bounded variant of [`lp_always_true`].
///
/// On expiry the LP returns `None`, so this returns `false` (inconclusive) —
/// the predicate is left un-folded, which is verdict-preserving.
pub(crate) fn lp_always_true_deadline(
    net: &PetriNet,
    predicate: &ResolvedPredicate,
    deadline: Option<Instant>,
) -> bool {
    lp_predicate_truth(net, predicate, deadline) == Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceInfo, TransitionInfo};

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    fn conserving_net() -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![3, 0],
        }
    }

    fn dual_kill_trap_net() -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1), arc(1, 1)], vec![arc(0, 1)]),
                trans("t1", vec![arc(0, 1), arc(1, 1)], vec![arc(1, 1)]),
                trans("t2", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 1],
        }
    }

    fn basic_lp_unreachable_without_traps(net: &PetriNet, predicate: &ResolvedPredicate) -> bool {
        if !is_lp_amenable(predicate) {
            return false;
        }

        let np = net.num_places();
        let nt = net.num_transitions();

        if np + nt > MAX_LP_VARIABLES {
            return false;
        }

        let mut problem = Problem::new(OptimizationDirection::Minimize);

        let x_vars: Vec<_> = (0..nt)
            .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
            .collect();
        let m_vars: Vec<_> = (0..np)
            .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
            .collect();

        add_state_equation(&mut problem, net, &x_vars, &m_vars);

        if !add_predicate_constraints(&mut problem, predicate, &m_vars) {
            return true;
        }

        matches!(problem.solve(), Err(minilp::Error::Infeasible))
    }

    #[test]
    fn test_lp_upper_bound_conserving_net() {
        let net = conserving_net();
        assert_eq!(lp_upper_bound(&net, &[PlaceIdx(0)]), Some(3));
        assert_eq!(lp_upper_bound(&net, &[PlaceIdx(1)]), Some(3));
        assert_eq!(lp_upper_bound(&net, &[PlaceIdx(0), PlaceIdx(1)]), Some(3));
    }

    #[test]
    fn test_lp_upper_bound_unbounded_net() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };
        assert_eq!(lp_upper_bound(&net, &[PlaceIdx(0)]), None);
    }

    #[test]
    fn test_lp_upper_bound_empty_places() {
        let net = conserving_net();
        assert_eq!(lp_upper_bound(&net, &[]), Some(0));
    }

    #[test]
    fn test_lp_objective_to_bound_rejects_non_finite_values() {
        assert_eq!(lp_objective_to_bound(f64::NAN), None);
        assert_eq!(lp_objective_to_bound(f64::INFINITY), None);
        assert_eq!(lp_objective_to_bound(f64::NEG_INFINITY), None);
        assert_eq!(lp_objective_to_bound(1.1), Some(2));
        assert_eq!(lp_objective_to_bound(1e15 + 1.0), None);
    }

    /// Regression: minilp can terminate with `objective() == NaN` on
    /// effectively-unbounded problems (degenerate basis with a primal
    /// variable at +∞). Reproduced on CryptoMiner-COL-D03N000 OneSafe:
    /// the LP for `sum(resource_c0..c3)` is genuinely unbounded
    /// (`x_ComputeFirst` is free), and minilp returned `NaN`. Without
    /// the explicit `is_finite()` guard in `lp_upper_bound`, the
    /// `obj > 1e15` check failed (NaN comparisons are false) and
    /// `NaN as u64 = 0` produced `Some(0)` — silently the strongest
    /// possible safety bound, causing OneSafe to return wrong TRUE.
    ///
    /// This minimal fixture mirrors the CryptoMiner shape: a producer
    /// transition with a single self-loop input/output on a separate
    /// "control" place keeps state coupled while leaving the resource
    /// growth unbounded — exactly the configuration that triggered
    /// the NaN return.
    #[test]
    fn test_lp_upper_bound_unbounded_sum_with_self_loop() {
        // Two places, one transition with self-loop on p0 and pure
        // production into p1. State equation: m[p1] = x_t. Unbounded.
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(0, 1), arc(1, 1)])],
            initial_marking: vec![1, 0],
        };
        // Single-place: unbounded.
        assert_eq!(lp_upper_bound(&net, &[PlaceIdx(1)]), None);
        // Sum: still unbounded; must NOT silently become Some(0) on NaN.
        assert_eq!(
            lp_upper_bound(&net, &[PlaceIdx(0), PlaceIdx(1)]),
            None,
            "unbounded LP must return None, never Some(0) — NaN soundness"
        );
    }

    #[test]
    fn test_lp_unreachable_impossible_marking() {
        let net = conserving_net();
        // p0 >= 5 is unreachable (p0 + p1 = 3, p1 >= 0 => p0 <= 3).
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(5),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        );
        assert!(lp_unreachable(&net, &pred));
    }

    #[test]
    fn test_lp_unreachable_possible_marking() {
        let net = conserving_net();
        // p1 >= 2 is reachable (fire t0 twice).
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(2),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        );
        assert!(!lp_unreachable(&net, &pred));
    }

    #[test]
    fn test_lp_unreachable_conjunction() {
        let net = conserving_net();
        // p0 >= 2 AND p1 >= 2 is impossible (sum = 3).
        let pred = ResolvedPredicate::And(vec![
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(2),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(2),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ),
        ]);
        assert!(lp_unreachable(&net, &pred));
    }

    #[test]
    fn test_lp_unreachable_trivially_false() {
        let net = conserving_net();
        assert!(lp_unreachable(&net, &ResolvedPredicate::False));
    }

    #[test]
    fn test_lp_unreachable_trivially_true() {
        let net = conserving_net();
        assert!(!lp_unreachable(&net, &ResolvedPredicate::True));
    }

    #[test]
    fn test_lp_unreachable_non_amenable_skipped() {
        let net = conserving_net();
        let pred = ResolvedPredicate::Or(vec![
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(5),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ),
            ResolvedPredicate::True,
        ]);
        assert!(!lp_unreachable(&net, &pred));
    }

    #[test]
    fn test_find_initially_marked_traps_dual_kill_net() {
        let net = dual_kill_trap_net();
        assert_eq!(find_initially_marked_traps(&net), vec![vec![true, true]]);
    }

    #[test]
    fn test_find_initially_unmarked_siphons_self_feeding_dead_place() {
        // p0 init 0; t0: input {p0} weight 1, output {p0} weight 2. The only
        // producer into p0 (t0) also consumes p0 ⇒ {p0} is a siphon; it is
        // initially unmarked, so it must be returned as a zero-cut.
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(0, 2)])],
            initial_marking: vec![0],
        };
        let siphons = find_initially_unmarked_siphons(&net);
        assert!(
            siphons.iter().any(|s| s == &vec![true]),
            "{{p0}} (initially-unmarked self-feeding siphon) must be found: {siphons:?}"
        );
    }

    #[test]
    fn test_find_initially_unmarked_siphons_excludes_marked_siphon() {
        // Same structure but p0 starts MARKED. The siphon is no longer initially
        // unmarked, so the "stays empty" theorem does not apply and NO zero-cut
        // may be emitted — soundness gate.
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(0, 2)])],
            initial_marking: vec![1],
        };
        assert!(
            find_initially_unmarked_siphons(&net).is_empty(),
            "an initially-MARKED siphon must NOT be returned as an unmarked-siphon zero-cut"
        );
    }

    #[test]
    fn test_find_initially_unmarked_siphons_returns_only_unmarked_members() {
        // Every returned siphon must be entirely initially unmarked (SOUNDNESS).
        // Two-place feeder: p0 init 1 (marked), p1 init 0; t0: p0->p1, t1: p1->p0.
        // {p0,p1} is a siphon but it IS initially marked (p0), so it must be
        // excluded; no fully-unmarked nonempty siphon exists here.
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        };
        for siphon in find_initially_unmarked_siphons(&net) {
            assert!(
                siphon_is_initially_unmarked(&net, &siphon),
                "every returned siphon must be initially unmarked: {siphon:?}"
            );
        }
    }

    #[test]
    fn test_lp_unreachable_with_traps_closes_feasible_basic_lp() {
        let net = dual_kill_trap_net();
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
            ResolvedIntExpr::Constant(0),
        );

        assert!(
            !basic_lp_unreachable_without_traps(&net, &pred),
            "basic state-equation LP admits the empty-trap marking"
        );
        assert!(lp_unreachable_with_traps(&net, &pred));
        assert!(lp_unreachable(&net, &pred));
    }

    #[test]
    fn test_lp_unreachable_with_traps_stays_feasible_when_traps_do_not_help() {
        let net = dual_kill_trap_net();
        let pred = ResolvedPredicate::And(vec![
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(0),
            ),
        ]);

        assert!(!basic_lp_unreachable_without_traps(&net, &pred));
        assert!(!lp_unreachable_with_traps(&net, &pred));
    }

    #[test]
    fn test_lp_unreachable_with_traps_preserves_basic_infeasibility() {
        let net = conserving_net();
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(5),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        );

        assert!(basic_lp_unreachable_without_traps(&net, &pred));
        assert!(lp_unreachable_with_traps(&net, &pred));
    }

    #[test]
    fn test_lp_strictly_greater_unreachable_token_rhs() {
        let net = conserving_net();
        assert!(lp_strictly_greater_unreachable(
            &net,
            &ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            &ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
        ));
    }

    #[test]
    fn test_lp_strictly_greater_unreachable_feasible_violation() {
        let net = conserving_net();
        assert!(!lp_strictly_greater_unreachable(
            &net,
            &ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            &ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ));
    }

    #[test]
    fn test_lp_upper_bound_weighted_net() {
        // p0(4) -2-> t0 -1-> p1. m0 = 4-2x, m1 = x, x <= 2.
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 1)])],
            initial_marking: vec![4, 0],
        };
        assert_eq!(lp_upper_bound(&net, &[PlaceIdx(1)]), Some(2));
        assert_eq!(lp_upper_bound(&net, &[PlaceIdx(0)]), Some(4));
    }

    #[test]
    fn test_lp_upper_bound_with_multiplicity() {
        let net = conserving_net();
        // max(2*p1) = 2*3 = 6
        assert_eq!(lp_upper_bound(&net, &[PlaceIdx(1), PlaceIdx(1)]), Some(6));
    }

    /// 1-safe shuttle p0(1) <-> p1(0) (so p0 + p1 = 1), plus self-loop probes
    /// ta on p0 and tb on p1. ta and tb can never be enabled simultaneously.
    fn mutex_fireability_net() -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t01", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t10", vec![arc(1, 1)], vec![arc(0, 1)]),
                trans("ta", vec![arc(0, 1)], vec![arc(0, 1)]),
                trans("tb", vec![arc(1, 1)], vec![arc(1, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// Place p0(1) is pinned: t_live (self-loop on p0) has zero net effect, and
    /// t_dead consumes p0 and p2, but p2(0) has no producer so the state
    /// equation forces t_dead's firing count to 0. The zero-incidence-row test
    /// misses p0 because t_dead's incidence row on p0 is -1 (nonzero), and
    /// cascade-isolation misses it because t_live is a live transition on p0.
    fn pinned_by_dead_transition_net() -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("p0"), place("p2")],
            transitions: vec![
                trans("t_live", vec![arc(0, 1)], vec![arc(0, 1)]),
                trans("t_dead", vec![arc(0, 1), arc(1, 1)], vec![]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    #[test]
    fn test_fireability_lower_bounds_single_transition_is_exact() {
        // ta needs p0 >= 1: necessary condition is exactly 1 <= M[p0].
        let net = mutex_fireability_net();
        let lowered = fireability_necessary_marking_lower_bounds(&net, &[TransitionIdx(2)]);
        assert_eq!(
            lowered,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            )
        );
    }

    #[test]
    fn test_fireability_lower_bounds_empty_is_false() {
        let net = mutex_fireability_net();
        assert_eq!(
            fireability_necessary_marking_lower_bounds(&net, &[]),
            ResolvedPredicate::False
        );
    }

    #[test]
    fn test_fireability_lower_bounds_common_place_min_demand() {
        // Two transitions both consuming p0 (weights 2 and 3): the necessary
        // shared condition is min(2,3) = 2 <= M[p0].
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![
                trans("t0", vec![arc(0, 2)], vec![]),
                trans("t1", vec![arc(0, 3)], vec![]),
            ],
            initial_marking: vec![5],
        };
        let lowered =
            fireability_necessary_marking_lower_bounds(&net, &[TransitionIdx(0), TransitionIdx(1)]);
        assert_eq!(
            lowered,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(2),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            )
        );
    }

    #[test]
    fn test_conjunction_of_mutex_fireability_is_unreachable() {
        // EF/AG over AND(IsFireable(ta), IsFireable(tb)): ta needs p0>=1, tb
        // needs p1>=1, but p0 + p1 = 1, so they are never jointly enabled. The
        // joint relaxation LP must prove the conjunction unreachable (Some(false)).
        let net = mutex_fireability_net();
        let conjunction = ResolvedPredicate::And(vec![
            ResolvedPredicate::IsFireable(vec![TransitionIdx(2)]),
            ResolvedPredicate::IsFireable(vec![TransitionIdx(3)]),
        ]);
        assert_eq!(
            lp_predicate_truth(&net, &conjunction, None),
            Some(false),
            "mutually exclusive fireability conjunction must be LP-unreachable"
        );

        // Each atom alone is satisfiable, so neither is independently decisive.
        assert_ne!(
            lp_predicate_truth(
                &net,
                &ResolvedPredicate::IsFireable(vec![TransitionIdx(2)]),
                None
            ),
            Some(false),
            "IsFireable(ta) alone is reachable"
        );
        assert_ne!(
            lp_predicate_truth(
                &net,
                &ResolvedPredicate::IsFireable(vec![TransitionIdx(3)]),
                None
            ),
            Some(false),
            "IsFireable(tb) alone is reachable"
        );
    }

    #[test]
    fn test_conjunction_with_satisfiable_fireability_stays_unresolved() {
        // AND(IsFireable(ta), p1 <= 1): ta needs p0>=1 (reachable) and p1<=1
        // always holds, so the conjunction IS reachable. LP must not claim false.
        let net = mutex_fireability_net();
        let conjunction = ResolvedPredicate::And(vec![
            ResolvedPredicate::IsFireable(vec![TransitionIdx(2)]),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(1),
            ),
        ]);
        assert_ne!(
            lp_predicate_truth(&net, &conjunction, None),
            Some(false),
            "a satisfiable fireability conjunction must not be proven unreachable"
        );
    }

    /// One token shuttles around a 3-place ring (p0+p1+p2 = 1). Self-loop probes
    /// ta0/ta1/tb2 require a token in p0/p1/p2 respectively.
    fn shared_token_ring_net() -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t01", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t12", vec![arc(1, 1)], vec![arc(2, 1)]),
                trans("t20", vec![arc(2, 1)], vec![arc(0, 1)]),
                trans("ta0", vec![arc(0, 1)], vec![arc(0, 1)]),
                trans("ta1", vec![arc(1, 1)], vec![arc(1, 1)]),
                trans("tb2", vec![arc(2, 1)], vec![arc(2, 1)]),
            ],
            initial_marking: vec![1, 0, 0],
        }
    }

    #[test]
    fn test_fireability_lower_bounds_disjoint_places_uses_union_sum() {
        // IsFireable(ta0, ta1): ta0 needs p0>=1, ta1 needs p1>=1. The per-place
        // minimum is 0 for both places, but the union-sum bound `1 <= p0+p1` is a
        // sound necessary condition.
        let net = shared_token_ring_net();
        let lowered =
            fireability_necessary_marking_lower_bounds(&net, &[TransitionIdx(3), TransitionIdx(4)]);
        assert_eq!(
            lowered,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
            ),
            "disjoint-place fireability must lower to the union-sum bound"
        );
    }

    #[test]
    fn test_conjunction_shared_token_unreachable_via_union_sum() {
        // AND(IsFireable(ta0,ta1), IsFireable(tb2)) requires (p0+p1>=1) AND
        // (p2>=1), but p0+p1+p2 = 1, so the conjunction is unreachable. Only the
        // union-sum bound (not the per-place bound) makes the joint LP infeasible.
        let net = shared_token_ring_net();
        let conjunction = ResolvedPredicate::And(vec![
            ResolvedPredicate::IsFireable(vec![TransitionIdx(3), TransitionIdx(4)]),
            ResolvedPredicate::IsFireable(vec![TransitionIdx(5)]),
        ]);
        assert_eq!(
            lp_predicate_truth(&net, &conjunction, None),
            Some(false),
            "shared-token fireability conjunction must be proven unreachable"
        );
    }

    #[test]
    fn test_lp_pinned_place_finds_invariant_pinned_place() {
        let net = pinned_by_dead_transition_net();
        // p0 is pinned at its initial value 1 even though t_dead's incidence row
        // on p0 is nonzero (-1): the state equation forces t_dead never to fire.
        assert_eq!(lp_pinned_place(&net, None), Some(PlaceIdx(0)));
    }

    #[test]
    fn test_lp_pinned_place_none_when_place_varies() {
        // The shuttle lets p0 reach both 0 and 1, so no place is pinned.
        let net = mutex_fireability_net();
        assert_eq!(lp_pinned_place(&net, None), None);
    }

    // ---- CEGAR exploration scaffolding (brute-force oracle + old fast path) ----

    /// Exhaustive forward reachability over markings, capped to keep BFS finite.
    /// Returns the full reachable marking set, or `None` if the cap is exceeded
    /// (so the test only trusts a complete enumeration).
    fn brute_force_reachable_markings(net: &PetriNet, cap: usize) -> Option<Vec<Vec<u64>>> {
        use std::collections::HashSet;
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut frontier = vec![net.initial_marking.clone()];
        seen.insert(net.initial_marking.clone());
        while let Some(marking) = frontier.pop() {
            for transition in &net.transitions {
                let enabled = transition
                    .inputs
                    .iter()
                    .all(|arc| marking[arc.place.0 as usize] >= arc.weight);
                if !enabled {
                    continue;
                }
                let mut next = marking.clone();
                for arc in &transition.inputs {
                    next[arc.place.0 as usize] -= arc.weight;
                }
                for arc in &transition.outputs {
                    next[arc.place.0 as usize] += arc.weight;
                }
                if seen.insert(next.clone()) {
                    if seen.len() > cap {
                        return None;
                    }
                    frontier.push(next);
                }
            }
        }
        Some(seen.into_iter().collect())
    }

    /// True iff some reachable marking satisfies `sum_{p in places} M[p] <= bound`.
    fn brute_force_sum_le_reachable(net: &PetriNet, places: &[usize], bound: u64) -> Option<bool> {
        let markings = brute_force_reachable_markings(net, 100_000)?;
        Some(markings.iter().any(|marking| {
            let sum: u64 = places.iter().map(|&p| marking[p]).sum();
            sum <= bound
        }))
    }

    /// Replicates the PRE-CEGAR single-shot trap loop: bare state equation +
    /// the precomputed traps, with NO separation oracle. Used to prove that the
    /// old behaviour is inconclusive (`false`) on a net the new CEGAR decides.
    fn precomputed_only_unreachable(
        net: &PetriNet,
        predicate: &ResolvedPredicate,
        traps: &[Vec<bool>],
    ) -> bool {
        if !is_lp_amenable(predicate) {
            return false;
        }
        let np = net.num_places();
        let nt = net.num_transitions();
        if np + nt > MAX_LP_VARIABLES {
            return false;
        }
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x_vars: Vec<_> = (0..nt)
            .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
            .collect();
        let m_vars: Vec<_> = (0..np)
            .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
            .collect();
        add_state_equation(&mut problem, net, &x_vars, &m_vars);
        if !add_predicate_constraints(&mut problem, predicate, &m_vars) {
            return true;
        }
        let mut solution = match problem.solve() {
            Ok(solution) => solution,
            Err(minilp::Error::Infeasible) => return true,
            Err(minilp::Error::Unbounded) => return false,
        };
        let mut added = vec![false; traps.len()];
        for _ in 0..100 {
            let violated: Vec<_> = traps
                .iter()
                .enumerate()
                .filter_map(|(i, trap)| {
                    (!added[i] && trap_marking_sum(trap, &solution, &m_vars) < 0.5).then_some(i)
                })
                .collect();
            if violated.is_empty() {
                return false;
            }
            for i in violated {
                added[i] = true;
                solution = match solution.add_constraint(
                    trap_cut_row(&traps[i], &m_vars),
                    ComparisonOp::Ge,
                    1.0,
                ) {
                    Ok(solution) => solution,
                    Err(minilp::Error::Infeasible) => return true,
                    Err(minilp::Error::Unbounded) => return false,
                };
            }
        }
        false
    }

    /// Two interlocked dual-kill trap pairs that share a common "hub" place.
    ///
    /// Places: hub h(2) marked, a0(0), a1(0), b0(0), b1(0). The marked trap that
    /// refutes "all five places empty" is the whole-net trap {h,a0,a1,b0,b1}
    /// (h holds 2 tokens initially and is a trap source). The state equation
    /// alone admits the all-zero marking (the dual-kill transitions let the LP
    /// drain every place with fractional firing counts), so it is inconclusive.
    fn hub_dual_kill_net() -> PetriNet {
        // h=0, a0=1, a1=2, b0=3, b1=4
        PetriNet {
            name: None,
            places: vec![
                place("h"),
                place("a0"),
                place("a1"),
                place("b0"),
                place("b1"),
            ],
            transitions: vec![
                // Seed the satellites from the hub (h -> a0, h -> b0): keeps the
                // whole set a trap (consumes h, produces into the set).
                trans("seed_a", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("seed_b", vec![arc(0, 1)], vec![arc(3, 1)]),
                // Dual-kill interlock on {a0,a1}: each consumes both, produces one.
                trans("ka0", vec![arc(1, 1), arc(2, 1)], vec![arc(1, 1)]),
                trans("ka1", vec![arc(1, 1), arc(2, 1)], vec![arc(2, 1)]),
                trans("ra", vec![arc(2, 1)], vec![arc(1, 1)]),
                // Dual-kill interlock on {b0,b1}.
                trans("kb0", vec![arc(3, 1), arc(4, 1)], vec![arc(3, 1)]),
                trans("kb1", vec![arc(3, 1), arc(4, 1)], vec![arc(4, 1)]),
                trans("rb", vec![arc(4, 1)], vec![arc(3, 1)]),
            ],
            initial_marking: vec![2, 0, 0, 0, 0],
        }
    }

    /// Core CEGAR regression: a net the single-shot trap pass leaves inconclusive
    /// but iterative trap CEGAR proves UNSAT, with the verdict cross-checked
    /// against exhaustive reachability.
    ///
    /// On `hub_dual_kill_net` the only refuting invariant for "every place empty"
    /// is the whole-net marked trap {h,a0,a1,b0,b1} (h holds tokens and every
    /// transition produces back into the place set, so the set is a trap that
    /// stays marked). The seed-based enumeration MISSES it: `minimize_trap`
    /// greedily strips the trap down to unmarked dual-kill sub-traps ({a0,a1},
    /// {b0,b1}), which `find_initially_marked_traps` then discards as not
    /// initially marked — so the precomputed set is empty and the single-shot
    /// pass is inconclusive. The CEGAR separation oracle instead takes the
    /// MAXIMAL trap inside the empty support of the LP vertex, recovers the marked
    /// whole-net trap, and closes the LP.
    #[test]
    fn test_cegar_decides_when_single_shot_trap_pass_is_inconclusive() {
        let net = hub_dual_kill_net();

        // (1) The single-shot seed enumeration genuinely finds NO usable trap.
        assert!(
            find_initially_marked_traps(&net).is_empty(),
            "seed/minimize/dedup enumeration discards the whole-net marked trap"
        );

        // Predicate: every place empty (sum of all places <= 0). With M >= 0 this
        // forces the all-zero marking.
        let all_places: Vec<PlaceIdx> = (0..net.num_places() as u32).map(PlaceIdx).collect();
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(all_places),
            ResolvedIntExpr::Constant(0),
        );

        // (2) The bare state equation admits the all-zero marking (inconclusive):
        // the dual-kill interlocks let the LP drain every place with fractional
        // firing counts that no real sequence realises.
        assert!(
            !basic_lp_unreachable_without_traps(&net, &pred),
            "basic state-equation LP cannot refute the all-zero marking"
        );

        // (3) The PRE-CEGAR single-shot trap loop is also inconclusive (the
        // precomputed trap set is empty, so it has nothing to add).
        let traps = find_initially_marked_traps(&net);
        assert!(
            !precomputed_only_unreachable(&net, &pred, &traps),
            "single-shot precomputed-trap pass leaves the marking inconclusive"
        );

        // (4) Iterative trap CEGAR PROVES the marking unreachable.
        assert!(
            lp_unreachable_with_traps(&net, &pred),
            "CEGAR must derive the whole-net marked trap and prove UNSAT"
        );

        // (5) The full predicate-truth pipeline reports the sound FALSE verdict.
        assert_eq!(lp_predicate_truth(&net, &pred, None), Some(false));

        // (6) Cross-check the verdict against EXHAUSTIVE reachability: the marking
        // is genuinely unreachable, so the LP's FALSE is correct (not a spurious
        // claim from a feasible solution).
        assert_eq!(
            brute_force_sum_le_reachable(&net, &[0, 1, 2, 3, 4], 0),
            Some(false),
            "exhaustive reachability confirms the all-zero marking is unreachable"
        );
    }

    /// CEGAR must NEVER emit a verdict from a feasible (SAT) LP solution: a
    /// genuinely reachable target must stay inconclusive (`false`), even though
    /// the LP is feasible.
    #[test]
    fn test_cegar_does_not_decide_reachable_target() {
        let net = hub_dual_kill_net();
        // "a0 >= 1" is reachable (fire seed_a from the hub), so the unreachability
        // prover must return false and the truth pipeline must not claim FALSE.
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        );
        let markings = brute_force_reachable_markings(&net, 100_000).unwrap();
        assert!(
            markings.iter().any(|marking| marking[1] >= 1),
            "a0 >= 1 must be genuinely reachable"
        );
        assert!(
            !lp_unreachable_with_traps(&net, &pred),
            "a reachable target must never be proven unreachable by CEGAR"
        );
        assert_ne!(lp_predicate_truth(&net, &pred, None), Some(false));
    }

    /// An already-expired deadline makes the CEGAR loop bail out as inconclusive
    /// (`false`) rather than fabricating a verdict — but the predicate is still
    /// genuinely unreachable, so falling through to BFS stays correct.
    #[test]
    fn test_cegar_expired_deadline_is_inconclusive() {
        let net = hub_dual_kill_net();
        let all_places: Vec<PlaceIdx> = (0..net.num_places() as u32).map(PlaceIdx).collect();
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(all_places),
            ResolvedIntExpr::Constant(0),
        );
        let past = Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap();
        // With budget, CEGAR decides; with an expired deadline it declines.
        assert!(lp_unreachable_with_traps_deadline(&net, &pred, None));
        assert!(!lp_unreachable_with_traps_deadline(&net, &pred, Some(past)));
    }
}
