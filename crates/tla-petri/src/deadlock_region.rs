// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Structural deadlock-freedom via **marking-equation infeasibility of the
//! deadlock region**, refined by marked-trap constraints (a CEGAR loop) — the
//! complete counterpart to the (weaker) [`crate::structural::lp_deadlock_free`]
//! "some transition is always enabled" check, and to
//! [`crate::structural::structural_deadlock_free`] which is complete only for
//! free-choice nets.
//!
//! # The query
//!
//! A P/T net has a *deadlock* iff some reachable marking `M` disables every
//! transition. Over-approximate reachability by the **state (marking) equation**
//! and ask whether that region is inhabited:
//!
//! ```text
//! ∃ M, σ :  M = M₀ + C·σ  ∧  M ≥ 0  ∧  σ ≥ 0  ∧  ⋀_t ( ⋁_{p∈•t} M[p] ≤ W(p,t)−1 )
//! ```
//!
//! The bare marking equation admits *spurious* deadlocks (solutions that are not
//! reachable), so on each `SAT` we extract a **marked trap** that the spurious
//! solution empties and add the sound invariant `Σ_{p∈Q} M[p] ≥ 1` (a marked
//! trap stays marked in every reachable marking), then re-solve. This is the
//! standard trap-refinement (CEGAR) that decides deadlock-freedom for the broad
//! non-free-choice class the siphon/trap pre-checks miss (Raft / consensus /
//! token-conservation nets).
//!
//! # Soundness (only ever emits `Some(true)` = deadlock-free)
//!
//! Every reachable marking satisfies `M = M₀ + C·σ` (σ its firing-count vector)
//! *and* every added marked-trap invariant, so the refined region is always a
//! **superset** of the reachable set. Therefore **UNSAT ⇒ no reachable deadlock
//! ⇒ `Some(true)`** (a rigorous proof). `SAT` with no refuting trap is a
//! candidate that may be real ⇒ `None`; solver `Unknown`/timeout/error ⇒ `None`.
//! The function never returns `Some(false)`, so it can never contribute a wrong
//! `ReachabilityDeadlock` verdict. Every trap constraint is self-checked with
//! [`is_trap`] before being asserted, so a construction bug degrades to
//! inconclusive, never to an unsound proof.
//!
//! The [`PetriNet`] model is pure P/T (regular arcs only), so "disabled" is
//! exactly "some input place under-marked"; parallel input arcs are summed.

use std::collections::HashMap;
use std::time::Instant;

use ay_dpll::api::{Logic, ModelValue, SolveResult, Solver, Sort, Term};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::petri_net::{PetriNet, TransitionIdx};

/// Prove deadlock-freedom via (trap-refined) marking-equation infeasibility of
/// the deadlock region. `Some(true)` = deadlock-free (UNSAT-proved); `None` =
/// inconclusive (spurious/real candidate, or budget exhausted).
///
/// Two-stage: the fast **LP relaxation** (real σ) first — it decides the large
/// structurally-deadlock-free nets that integer branch-and-bound times out on —
/// then the tighter **integer** query as a fallback for the (usually small) nets
/// whose LP relaxation has a spurious fractional deadlock but no integer one.
/// Both are sound (UNSAT ⇒ deadlock-free); the union answers strictly more than
/// either alone.
///
/// `real` selects the query flavor: `false` = tight integer (QF_LIA), `true` =
/// LP relaxation (QF_LRA, real σ — proves the large structurally-free nets that
/// integer branch-and-bound times out on). Both are sound. The caller runs them
/// as two independent wall-capped phases so neither starves the other.
pub(crate) fn deadlock_region_infeasible(
    net: &PetriNet,
    deadline: Option<Instant>,
    real: bool,
) -> Option<bool> {
    let num_places = net.places.len();
    let num_trans = net.transitions.len();
    if num_trans == 0 || num_places == 0 {
        return None;
    }
    // A transition with no input arcs is unconditionally enabled ⇒ the net can
    // never deadlock (same shortcut as `lp_deadlock_free`).
    if net.transitions.iter().any(|t| t.inputs.is_empty()) {
        return Some(true);
    }
    let logic = if real { Logic::QfLra } else { Logic::QfLia };
    let mut solver = Solver::try_new(logic).ok()?;
    // Constant builder: exact rational for QF_LRA, plain int for QF_LIA.
    macro_rules! konst {
        ($s:expr, $v:expr) => {
            if real {
                $s.rational_const($v as i64, 1)
            } else {
                $s.int_const($v as i64)
            }
        };
    }
    let set_budget = |solver: &mut Solver| -> Option<()> {
        if let Some(dl) = deadline {
            let budget = dl.saturating_duration_since(Instant::now());
            if budget.is_zero() {
                return None;
            }
            solver.set_timeout(Some(budget));
        }
        Some(())
    };
    set_budget(&mut solver)?;

    let zero = konst!(solver, 0);

    // Variables M[p] (place tokens) and σ[t] (firing counts). Under `real` these
    // are Reals (the state-equation LP relaxation — real σ solutions are a
    // superset of the integer/reachable ones, so UNSAT is still a sound
    // deadlock-freedom proof, and each theory check is a fast LP instead of
    // integer branch-and-bound: Anderson-PT-09 6.6s LRA vs 18.1s LIA).
    let m: Vec<Term> = (0..num_places)
        .map(|p| solver.declare_const(&format!("m_{p}"), if real { Sort::Real } else { Sort::Int }))
        .collect();
    let sigma: Vec<Term> = (0..num_trans)
        .map(|t| solver.declare_const(&format!("s_{t}"), if real { Sort::Real } else { Sort::Int }))
        .collect();

    for &v in m.iter().chain(sigma.iter()) {
        let ge = solver.try_ge(v, zero).ok()?;
        solver.try_assert_term(ge).ok()?;
    }

    // Marking equation: M[p] = M₀[p] + Σ_t C[p,t]·σ[t]  (parallel arcs summed).
    let mut rhs_terms: Vec<Vec<Term>> = (0..num_places)
        .map(|p| vec![konst!(solver, net.initial_marking[p].min(i64::MAX as u64))])
        .collect();
    for (t, tr) in net.transitions.iter().enumerate() {
        let mut coeff: HashMap<usize, i64> = HashMap::new();
        for arc in &tr.inputs {
            *coeff.entry(arc.place.0 as usize).or_default() -= arc.weight as i64;
        }
        for arc in &tr.outputs {
            *coeff.entry(arc.place.0 as usize).or_default() += arc.weight as i64;
        }
        for (p, c) in coeff {
            if c == 0 {
                continue;
            }
            let cc = konst!(solver, c);
            let term = solver.try_mul(cc, sigma[t]).ok()?;
            rhs_terms[p].push(term);
        }
    }
    for (p, terms) in rhs_terms.into_iter().enumerate() {
        let rhs = solver.try_add_many(&terms).ok()?;
        let eq = solver.try_eq(m[p], rhs).ok()?;
        solver.try_assert_term(eq).ok()?;
    }

    // Deadlock region: every transition disabled — ⋁_{p∈•t} M[p] ≤ W(p,t)−1.
    for tr in &net.transitions {
        let mut in_weight: HashMap<usize, u64> = HashMap::new();
        for arc in &tr.inputs {
            *in_weight.entry(arc.place.0 as usize).or_default() += arc.weight;
        }
        let mut disjuncts: Vec<Term> = Vec::with_capacity(in_weight.len());
        for (p, w) in in_weight {
            let bound = konst!(solver, w.max(1) - 1);
            disjuncts.push(solver.try_le(m[p], bound).ok()?);
        }
        let disabled = solver.try_or_many(&disjuncts).ok()?;
        solver.try_assert_term(disabled).ok()?;
    }

    // CEGAR: refine with marked-trap invariants until UNSAT (deadlock-free), a
    // non-refutable SAT (inconclusive), or the iteration/budget bound.
    let max_refinements = num_places.saturating_add(num_trans).saturating_add(8);
    for _ in 0..max_refinements {
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            return None;
        }
        set_budget(&mut solver)?;
        match solver.try_check_sat().ok()?.into_inner() {
            SolveResult::Unsat(_) => return Some(true),
            SolveResult::Sat => {}
            // Unknown, or any future non-`Sat`/`Unsat` verdict ⇒ inconclusive.
            _ => return None,
        }
        // Places empty in the spurious deadlock model M*.
        let mut empty = vec![false; num_places];
        for (p, &mp) in m.iter().enumerate() {
            // Dependency-free zero test: BigInt / BigRational render zero as "0".
            empty[p] = match solver.value(mp) {
                Some(ModelValue::Int(ref v)) => v.to_string() == "0",
                Some(ModelValue::Real(ref r)) => r.to_string() == "0",
                _ => false,
            };
        }
        // Maximal trap contained in the empty set; if it is marked at M₀ it must
        // stay marked, so M* is spurious — add Σ_{p∈Q} M[p] ≥ 1 and re-solve.
        let trap = maximal_trap_within(net, &empty);
        let marked_at_init = trap
            .iter()
            .enumerate()
            .any(|(p, &in_q)| in_q && net.initial_marking[p] > 0);
        if !marked_at_init || !is_trap(net, &trap) {
            // No refuting marked trap: M* is a genuine candidate in the
            // trap-refined over-approximation ⇒ inconclusive (fall through).
            return None;
        }
        let one = konst!(solver, 1);
        let trap_terms: Vec<Term> = trap
            .iter()
            .enumerate()
            .filter_map(|(p, &in_q)| in_q.then_some(m[p]))
            .collect();
        let sum = solver.try_add_many(&trap_terms).ok()?;
        let ge1 = solver.try_ge(sum, one).ok()?;
        solver.try_assert_term(ge1).ok()?;
    }
    None
}

/// The trap condition: every transition consuming from `places` also produces
/// into it. (Local copy — `crate::structural::is_trap` is private.)
fn is_trap(net: &PetriNet, places: &[bool]) -> bool {
    for t in &net.transitions {
        let consumes = t.inputs.iter().any(|a| places[a.place.0 as usize]);
        if consumes && !t.outputs.iter().any(|a| places[a.place.0 as usize]) {
            return false;
        }
    }
    true
}

/// Maximal trap contained in `allowed` (dual of `siphon_closure`): start with
/// `allowed` and remove, to a fixpoint, the input places of every transition
/// that consumes from the set but produces nothing back into it. The result
/// satisfies [`is_trap`] and is ⊆ `allowed`.
fn maximal_trap_within(net: &PetriNet, allowed: &[bool]) -> Vec<bool> {
    let mut set = allowed.to_vec();
    loop {
        let mut changed = false;
        for t in &net.transitions {
            let consumes = t.inputs.iter().any(|a| set[a.place.0 as usize]);
            if !consumes {
                continue;
            }
            let produces = t.outputs.iter().any(|a| set[a.place.0 as usize]);
            if produces {
                continue;
            }
            // t leaks the set: none of its Q-inputs can be in a trap ⊆ allowed.
            for a in &t.inputs {
                let p = a.place.0 as usize;
                if set[p] {
                    set[p] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    debug_assert!(
        is_trap(net, &set),
        "maximal_trap_within must produce a trap"
    );
    set
}

/// σ-guided reachable-deadlock **witness** — the TRUE counterpart to
/// [`deadlock_region_infeasible`].
///
/// Solves the integer deadlock region for a candidate firing-count vector σ,
/// then greedily *realizes* it from `M₀`: at each step it fires an **enabled**
/// transition that still has σ-budget, so the fired sequence is a genuine run of
/// the net. If the simulation ever reaches a marking with **no enabled
/// transition**, that is a concretely-reachable deadlock ⇒ `Some(true)`
/// (ReachabilityDeadlock TRUE).
///
/// # Soundness (only ever emits `Some(true)` = deadlock exists)
///
/// Every fired transition is checked `is_enabled` at the current marking, so the
/// visited markings are all reachable. A dead marking is only *reached* if the
/// net genuinely has a reachable deadlock — a deadlock-free net can never reach
/// one, so this cannot emit a false TRUE. `None` = no candidate (region UNSAT /
/// unknown), or σ is not greedily realizable to a deadlock (spurious or needs a
/// different order) ⇒ fall through to BMC/BFS. It never emits `Some(false)`.
///
/// σ guides the walk toward the marking-equation's candidate deadlock, so it
/// finds *directed* deadlocks that an unguided walk / bounded BMC miss.
pub(crate) fn deadlock_witness(net: &PetriNet, deadline: Option<Instant>) -> Option<bool> {
    let num_places = net.places.len();
    let num_trans = net.transitions.len();
    if num_trans == 0 || num_places == 0 {
        return None;
    }
    // A transition with no inputs is always enabled ⇒ no deadlock is possible.
    if net.transitions.iter().any(|t| t.inputs.is_empty()) {
        return None;
    }

    // Candidate σ from the integer deadlock region (one shot — the greedy
    // realization below IS the reachability check, so no trap refinement needed).
    let mut solver = Solver::try_new(Logic::QfLia).ok()?;
    if let Some(dl) = deadline {
        let budget = dl.saturating_duration_since(Instant::now());
        if budget.is_zero() {
            return None;
        }
        solver.set_timeout(Some(budget));
    }
    let zero = solver.int_const(0);
    let m: Vec<Term> = (0..num_places)
        .map(|p| solver.declare_const(&format!("m_{p}"), Sort::Int))
        .collect();
    let sigma: Vec<Term> = (0..num_trans)
        .map(|t| solver.declare_const(&format!("s_{t}"), Sort::Int))
        .collect();
    for &v in m.iter().chain(sigma.iter()) {
        let ge = solver.try_ge(v, zero).ok()?;
        solver.try_assert_term(ge).ok()?;
    }
    let mut rhs_terms: Vec<Vec<Term>> = (0..num_places)
        .map(|p| vec![solver.int_const(net.initial_marking[p].min(i64::MAX as u64) as i64)])
        .collect();
    for (t, tr) in net.transitions.iter().enumerate() {
        let mut coeff: HashMap<usize, i64> = HashMap::new();
        for arc in &tr.inputs {
            *coeff.entry(arc.place.0 as usize).or_default() -= arc.weight as i64;
        }
        for arc in &tr.outputs {
            *coeff.entry(arc.place.0 as usize).or_default() += arc.weight as i64;
        }
        for (p, c) in coeff {
            if c == 0 {
                continue;
            }
            let cc = solver.int_const(c);
            let term = solver.try_mul(cc, sigma[t]).ok()?;
            rhs_terms[p].push(term);
        }
    }
    for (p, terms) in rhs_terms.into_iter().enumerate() {
        let rhs = solver.try_add_many(&terms).ok()?;
        let eq = solver.try_eq(m[p], rhs).ok()?;
        solver.try_assert_term(eq).ok()?;
    }
    for tr in &net.transitions {
        let mut in_weight: HashMap<usize, u64> = HashMap::new();
        for arc in &tr.inputs {
            *in_weight.entry(arc.place.0 as usize).or_default() += arc.weight;
        }
        let mut disjuncts: Vec<Term> = Vec::with_capacity(in_weight.len());
        for (p, w) in in_weight {
            let bound = solver.int_const((w.max(1) - 1) as i64);
            disjuncts.push(solver.try_le(m[p], bound).ok()?);
        }
        let disabled = solver.try_or_many(&disjuncts).ok()?;
        solver.try_assert_term(disabled).ok()?;
    }
    match solver.try_check_sat().ok()?.into_inner() {
        SolveResult::Sat => {}
        _ => return None, // UNSAT (deadlock-free) / Unknown ⇒ no witness here
    }

    // Extract the candidate σ (non-negative integer firing counts) AND the target
    // dead marking M* the solver already computed — both are model values in hand.
    let mut sigma_budget: Vec<u64> = Vec::with_capacity(num_trans);
    for &st in &sigma {
        let v = match solver.value(st) {
            Some(ModelValue::Int(ref x)) => x.to_string().parse::<u64>().unwrap_or(0),
            _ => 0,
        };
        sigma_budget.push(v);
    }
    let target: Vec<u64> = m
        .iter()
        .map(|&mp| match solver.value(mp) {
            Some(ModelValue::Int(ref x)) => x.to_string().parse::<u64>().unwrap_or(0),
            _ => 0,
        })
        .collect();

    // Net token change per place for each transition (precomputed once), so the
    // M*-distance of a candidate firing is evaluated in O(|arcs|), not O(|places|).
    let deltas: Vec<Vec<(usize, i64)>> = net
        .transitions
        .iter()
        .map(|tr| {
            let mut d: HashMap<usize, i64> = HashMap::new();
            for arc in &tr.inputs {
                *d.entry(arc.place.0 as usize).or_default() -= arc.weight as i64;
            }
            for arc in &tr.outputs {
                *d.entry(arc.place.0 as usize).or_default() += arc.weight as i64;
            }
            d.into_iter().filter(|&(_, c)| c != 0).collect()
        })
        .collect();

    let total: u64 = sigma_budget
        .iter()
        .copied()
        .fold(0u64, |a, b| a.saturating_add(b));
    let step_cap = total.saturating_add(num_trans as u64);

    // Realize σ as an M*-DIRECTED walk with randomized restarts. The single greedy
    // fixed-order pass (which discarded M* and gave up on the first stuck state)
    // fell through to the exhaustive BFS; a distance-to-M* choice with restart
    // diversity reaches the deadlock in O(Σσ) steps when it is greedily reachable.
    //
    // Seeded DETERMINISTICALLY from the net's shape so the verdict is reproducible
    // (the TwoPhase determinism gate): same net ⇒ same restart sequence. Each
    // restart is an independent fresh attempt from M₀ with the full σ budget.
    //
    // Bounded by BOTH the wall deadline AND a no-progress patience: restarts only
    // diversify via distance ties, so once `step_cap` consecutive restarts fail to
    // reach a strictly-closer marking to M*, further restarts are structurally
    // stuck (not merely unlucky) — give up and fall through to the exhaustive lanes
    // instead of spinning to the deadline. `step_cap` (= Σσ + |T|, the realization
    // depth) is the natural, net-derived diversity budget — no magic constant.
    let base_seed = (num_places as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((num_trans as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(total);
    let mut best_dist = u64::MAX;
    let mut stale: u64 = 0;
    let mut restart: u64 = 0;
    loop {
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            return None;
        }
        let mut rng =
            SmallRng::seed_from_u64(base_seed ^ restart.wrapping_mul(0x2545_F491_4F6C_DD1D));
        let (dead, dist) = realize_sigma(
            net,
            &target,
            &deltas,
            &sigma_budget,
            step_cap,
            deadline,
            &mut rng,
        );
        if dead {
            return Some(true); // reachable deadlock witnessed
        }
        if dist < best_dist {
            best_dist = dist;
            stale = 0;
        } else {
            stale = stale.saturating_add(1);
            if stale >= step_cap {
                return None; // structurally stuck — let the exhaustive lanes decide
            }
        }
        restart = restart.wrapping_add(1);
    }
}

/// One independent M*-directed realization attempt of the firing-count vector
/// `sigma` from M₀.
///
/// Fires ONLY `is_enabled`-checked, σ-budgeted transitions and returns
/// `(true, 0)` only at a marking where *no* transition is enabled — so every
/// witness is a genuinely reachable deadlock (self-certifying; the strongest
/// soundness posture — a bug can only fail to find a witness, never fabricate
/// one). Among the candidate firings it prefers the one whose post-fire marking
/// is closest (L1) to the solver's target dead marking `target`, breaking ties
/// randomly so restarts explore different paths. When the attempt cannot reach a
/// dead marking (stuck at a non-dead marking, out of budget, or over `step_cap`)
/// it returns `(false, best)` where `best` is the smallest L1 distance to `target`
/// this attempt achieved — the caller uses it to detect when restarts have stopped
/// making progress and are structurally (not just unluckily) stuck.
fn realize_sigma(
    net: &PetriNet,
    target: &[u64],
    deltas: &[Vec<(usize, i64)>],
    sigma: &[u64],
    step_cap: u64,
    deadline: Option<Instant>,
    rng: &mut SmallRng,
) -> (bool, u64) {
    let num_trans = net.transitions.len();
    let mut remaining: Vec<u64> = sigma.to_vec();
    let mut marking = net.initial_marking.clone();
    // Running L1 distance to M*, maintained incrementally (initialized O(|P|) once
    // per attempt, then updated by each firing's precomputed `ddist`).
    let mut dist: i64 = marking
        .iter()
        .zip(target.iter())
        .map(|(&m, &t)| (m as i64 - t as i64).abs())
        .sum();
    let mut best_dist = dist.max(0) as u64;
    let mut steps: u64 = 0;
    loop {
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            return (false, best_dist);
        }
        let any_enabled = (0..num_trans).any(|t| net.is_enabled(&marking, TransitionIdx(t as u32)));
        if !any_enabled {
            return (true, 0); // reachable deadlock witnessed
        }
        // Among enabled, σ-budgeted transitions, choose the firing that most
        // reduces L1 distance to the target dead marking M* (ties broken at
        // random). `ddist` is the change in distance from firing t, computed
        // incrementally over t's affected places only.
        let mut best_ddist = i64::MAX;
        let mut best: Option<usize> = None;
        let mut ties: u32 = 0;
        for t in 0..num_trans {
            if remaining[t] == 0 || !net.is_enabled(&marking, TransitionIdx(t as u32)) {
                continue;
            }
            let mut ddist: i64 = 0;
            for &(p, d) in &deltas[t] {
                let old = (marking[p] as i64 - target[p] as i64).abs();
                let new = (marking[p] as i64 + d - target[p] as i64).abs();
                ddist += new - old;
            }
            if ddist < best_ddist {
                best_ddist = ddist;
                best = Some(t);
                ties = 1;
            } else if ddist == best_ddist {
                // Reservoir tie-break: pick uniformly among equal-distance moves.
                ties += 1;
                if rng.gen_range(0..ties) == 0 {
                    best = Some(t);
                }
            }
        }
        match best {
            Some(t) => {
                if net
                    .apply_delta(&mut marking, TransitionIdx(t as u32))
                    .is_err()
                {
                    return (false, best_dist);
                }
                remaining[t] -= 1;
                dist = (dist + best_ddist).max(0);
                best_dist = best_dist.min(dist as u64);
            }
            // No σ-budgeted transition enabled, yet the marking is not dead:
            // this attempt is stuck ⇒ let the caller restart from M₀.
            None => return (false, best_dist),
        }
        steps += 1;
        if steps > step_cap {
            return (false, best_dist);
        }
    }
}

/// Structural **OneSafe** (1-safe) prover via marking-equation infeasibility of
/// the bound-violation region.
///
/// A P/T net is 1-safe iff no reachable marking puts ≥2 tokens in any place.
/// Over the state (marking) equation:
/// `∃ M,σ: M=M₀+C·σ ∧ M,σ≥0 ∧ ⋁_p M[p]≥2`. The marking-equation region ⊇ the
/// reachable set (and is further tightened by marked-trap invariants, CEGAR), so
/// **UNSAT ⇒ no reachable marking exceeds 1 ⇒ 1-safe ⇒ `Some(true)`** (a rigorous
/// proof). `SAT`/`Unknown`/timeout ⇒ `None`. Emits only `Some(true)`, so it can
/// never contribute a wrong OneSafe verdict. Uses the LP relaxation (real σ) —
/// fast on the large nets the DD max-token path times out on.
pub(crate) fn onesafe_bounded(net: &PetriNet, deadline: Option<Instant>) -> Option<bool> {
    let num_places = net.places.len();
    let num_trans = net.transitions.len();
    if num_places == 0 {
        return None;
    }

    let mut solver = Solver::try_new(Logic::QfLra).ok()?;
    let set_budget = |solver: &mut Solver| -> Option<()> {
        if let Some(dl) = deadline {
            let b = dl.saturating_duration_since(Instant::now());
            if b.is_zero() {
                return None;
            }
            solver.set_timeout(Some(b));
        }
        Some(())
    };
    set_budget(&mut solver)?;
    let zero = solver.rational_const(0, 1);
    let m: Vec<Term> = (0..num_places)
        .map(|p| solver.declare_const(&format!("m_{p}"), Sort::Real))
        .collect();
    let sigma: Vec<Term> = (0..num_trans)
        .map(|t| solver.declare_const(&format!("s_{t}"), Sort::Real))
        .collect();
    for &v in m.iter().chain(sigma.iter()) {
        let ge = solver.try_ge(v, zero).ok()?;
        solver.try_assert_term(ge).ok()?;
    }
    // Marking equation M[p] = M₀[p] + Σ_t C[p,t]·σ[t].
    let mut rhs_terms: Vec<Vec<Term>> = (0..num_places)
        .map(|p| vec![solver.rational_const(net.initial_marking[p].min(i64::MAX as u64) as i64, 1)])
        .collect();
    for (t, tr) in net.transitions.iter().enumerate() {
        let mut coeff: HashMap<usize, i64> = HashMap::new();
        for arc in &tr.inputs {
            *coeff.entry(arc.place.0 as usize).or_default() -= arc.weight as i64;
        }
        for arc in &tr.outputs {
            *coeff.entry(arc.place.0 as usize).or_default() += arc.weight as i64;
        }
        for (p, c) in coeff {
            if c == 0 {
                continue;
            }
            let cc = solver.rational_const(c, 1);
            let term = solver.try_mul(cc, sigma[t]).ok()?;
            rhs_terms[p].push(term);
        }
    }
    for (p, terms) in rhs_terms.into_iter().enumerate() {
        let rhs = solver.try_add_many(&terms).ok()?;
        let eq = solver.try_eq(m[p], rhs).ok()?;
        solver.try_assert_term(eq).ok()?;
    }
    // Bound-violation region: some place holds ≥ 2 tokens.
    let two = solver.rational_const(2, 1);
    let mut over: Vec<Term> = Vec::with_capacity(num_places);
    for &mp in &m {
        over.push(solver.try_ge(mp, two).ok()?);
    }
    let some_over = solver.try_or_many(&over).ok()?;
    solver.try_assert_term(some_over).ok()?;

    // Trap-refinement CEGAR (marked traps stay marked — tightens the region).
    let max_refinements = num_places.saturating_add(num_trans).saturating_add(8);
    for _ in 0..max_refinements {
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            return None;
        }
        set_budget(&mut solver)?;
        match solver.try_check_sat().ok()?.into_inner() {
            SolveResult::Unsat(_) => return Some(true), // 1-safe
            SolveResult::Sat => {}
            _ => return None,
        }
        let mut empty = vec![false; num_places];
        for (p, &mp) in m.iter().enumerate() {
            empty[p] = match solver.value(mp) {
                Some(ModelValue::Int(ref v)) => v.to_string() == "0",
                Some(ModelValue::Real(ref r)) => r.to_string() == "0",
                _ => false,
            };
        }
        let trap = maximal_trap_within(net, &empty);
        let marked = trap
            .iter()
            .enumerate()
            .any(|(p, &q)| q && net.initial_marking[p] > 0);
        if !marked || !is_trap(net, &trap) {
            return None;
        }
        let one = solver.rational_const(1, 1);
        let terms: Vec<Term> = trap
            .iter()
            .enumerate()
            .filter_map(|(p, &q)| q.then_some(m[p]))
            .collect();
        let sum = solver.try_add_many(&terms).ok()?;
        let ge1 = solver.try_ge(sum, one).ok()?;
        solver.try_assert_term(ge1).ok()?;
    }
    None
}
