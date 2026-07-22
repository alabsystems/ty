// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! PDR/IC3 encoding for Petri net safety verification via ay-chc.
//!
//! Encodes a Petri net and a safety property as a Constrained Horn Clause (CHC)
//! problem and solves it with ay-chc's [`AdaptivePortfolio`]. The encoding uses
//! one uninterpreted predicate `Inv(m_0, ..., m_{P-1})` over integer marking
//! variables, with:
//!
//! - **Init clause**: `m_0 = init[0] /\ ... => Inv(m_0, ...)`
//! - **Consecution**: For each transition `t`: `Inv(m) /\ guard_t(m) /\ effect_t(m, m') => Inv(m')`
//! - **Stuttering**: `Inv(m) /\ m' = m => Inv(m')` (identity transition)
//! - **Query**: `Inv(m) /\ NOT safety(m) => false`
//! - **Strengthening** (optional): P-invariant equalities as additional constraints
//!
//! This is a sound reduction for Petri net safety. When the CHC solver finds
//! an inductive invariant, the property is proved for all reachable markings;
//! otherwise the result may remain `Unknown`. Because the state is encoded as
//! integer markings, the technique also applies to unbounded nets without
//! requiring explicit-state exploration.

use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use ay_chc::{
    AdaptiveConfig, AdaptivePortfolio, ChcExpr, ChcProblem, ChcSort, ChcVar, ClauseBody,
    ClauseHead, HornClause, VerifiedChcResult,
};

use crate::invariant::{compute_p_invariants, structural_place_bound, PInvariant};
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::resolved_predicate::{eval_predicate, ResolvedIntExpr, ResolvedPredicate};

/// Maximum `num_places × num_transitions` "cells" the PDR CHC encoder will
/// materialize. The consecution encoding frames every place in every
/// transition clause (sound transition relation) plus a per-place
/// non-negativity term, so the CHC problem size is Θ(places × transitions).
/// Beyond this budget the dense problem would exhaust memory (and is far past
/// what the solver could discharge), so [`solve_petri_net_pdr`] declines
/// up front. 1e6 cells ≈ a 1000×1000 net ≈ a few million CHC nodes (~1 GB
/// worst case); real PDR-amenable MCC nets are orders of magnitude smaller.
const PDR_ENCODING_MAX_CELLS: usize = 1_000_000;

/// Configuration for PDR-based Petri net verification.
#[derive(Debug, Clone)]
pub(crate) struct PdrConfig {
    /// Time budget for the adaptive portfolio solver (default: 30s).
    pub time_budget: Duration,
    /// Whether to add P-invariant strengthening clauses.
    pub use_p_invariants: bool,
    /// Whether to add symmetry clauses for orbit generalization.
    pub use_symmetry: bool,
    /// Whether to add a stuttering (identity) transition clause.
    pub add_stuttering: bool,
    /// Enable verbose output from the CHC solver.
    pub verbose: bool,
    /// Shared phase budget for the exact bounded fallback after the CHC solver
    /// is inconclusive. `None` keeps the historical unbounded fallback behavior.
    pub exact_fallback_budget: Option<Duration>,
    /// Absolute deadline for the exact bounded fallback. When both this and a
    /// phase budget are present, the earlier deadline wins.
    pub exact_fallback_deadline: Option<Instant>,
}

impl Default for PdrConfig {
    fn default() -> Self {
        Self {
            time_budget: Duration::from_secs(30),
            use_p_invariants: true,
            use_symmetry: false, // Default false, can be enabled by callers
            add_stuttering: true,
            verbose: false,
            exact_fallback_budget: None,
            exact_fallback_deadline: None,
        }
    }
}

/// Result of PDR-based safety verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PdrCheckResult {
    /// Property holds: an inductive invariant was found.
    Safe,
    /// Property violated: a counterexample trace exists.
    Unsafe,
    /// Solver could not determine the result within the budget.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoundedExactOutcome {
    Safe,
    Unsafe,
    Inconclusive(BoundedExactInconclusive),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoundedExactInconclusive {
    DeadlineExpired,
    MissingStructuralBounds,
    InitialExceedsBound,
    SuccessorExceedsBound,
    StateLimitExceeded,
}

const EXACT_BOUNDED_FALLBACK_STATE_LIMIT: usize = 100_000;
const EXACT_BOUNDED_FALLBACK_DEADLINE_CHECK_INTERVAL: u32 = 1024;
const EXACT_BOUNDED_FALLBACK_TRANSITION_DEADLINE_CHECK_INTERVAL: u32 = 64;

// ── Variable naming helpers ─────────────────────────────────────────

/// Create a marking variable for the current state: `m_<place_idx>`.
fn make_var(place_idx: usize) -> ChcVar {
    ChcVar::new(format!("m_{place_idx}"), ChcSort::Int)
}

/// Create a marking variable for the next state: `m_<place_idx>'`.
fn make_primed_var(place_idx: usize) -> ChcVar {
    ChcVar::new(format!("m_{place_idx}'"), ChcSort::Int)
}

/// Create a `ChcExpr::Var` for the current-state marking of a place.
fn var_expr(place_idx: usize) -> ChcExpr {
    ChcExpr::var(make_var(place_idx))
}

/// Create a `ChcExpr::Var` for the next-state marking of a place.
fn primed_var_expr(place_idx: usize) -> ChcExpr {
    ChcExpr::var(make_primed_var(place_idx))
}

// ── CHC encoding ────────────────────────────────────────────────────

/// Encode a Petri net and safety property as a CHC problem.
///
/// The returned `ChcProblem` has a single predicate `Inv` with `P` integer
/// arguments (one per place), plus clauses for initialization, transition
/// consecution, and the safety query.
pub(crate) fn encode_petri_net_pdr(
    net: &PetriNet,
    property: &ResolvedPredicate,
    config: &PdrConfig,
    canonicalizer: Option<&crate::explorer::symmetry::PetriCanonicalizer>,
) -> ChcProblem {
    let np = net.num_places();

    let mut problem = ChcProblem::new();

    // Declare Inv(m_0: Int, m_1: Int, ..., m_{P-1}: Int)
    let arg_sorts: Vec<ChcSort> = (0..np).map(|_| ChcSort::Int).collect();
    let inv = problem.declare_predicate("Inv", arg_sorts);

    // Current-state and next-state argument vectors for Inv
    let current_args: Vec<ChcExpr> = (0..np).map(var_expr).collect();
    let primed_args: Vec<ChcExpr> = (0..np).map(primed_var_expr).collect();

    // ── Init clause ──────────────────────────────────────────────
    // m_0 = init[0] /\ m_1 = init[1] /\ ... => Inv(m_0, m_1, ...)
    let init_conjuncts: Vec<ChcExpr> = (0..np)
        .map(|p| ChcExpr::eq(var_expr(p), ChcExpr::int(net.initial_marking[p] as i64)))
        .collect();
    let init_constraint = ChcExpr::and_all(init_conjuncts);
    problem.add_clause(HornClause::new(
        ClauseBody::constraint(init_constraint),
        ClauseHead::Predicate(inv, current_args.clone()),
    ));

    // ── Non-negativity constraints ──────────────────────────────
    // PDR synthesises over-approximating invariants.  Without explicit
    // non-negativity bounds the solver may include states with negative
    // token counts, slowing convergence.  We add m_p' >= 0 to every
    // consecution clause to keep the abstraction tight.
    let nonneg_primed: Vec<ChcExpr> = (0..np)
        .map(|p| ChcExpr::ge(primed_var_expr(p), ChcExpr::int(0)))
        .collect();

    // ── Consecution clauses (one per transition) ─────────────────
    // For each transition t:
    //   Inv(m) /\ guard_t(m) /\ effect_t(m, m') /\ m' >= 0 => Inv(m')
    for transition in &net.transitions {
        // Guard: all input places have enough tokens
        let guard_conjuncts: Vec<ChcExpr> = transition
            .inputs
            .iter()
            .map(|arc| {
                ChcExpr::ge(
                    var_expr(arc.place.0 as usize),
                    ChcExpr::int(arc.weight as i64),
                )
            })
            .collect();

        // Effect: compute deltas for each place
        let mut deltas = vec![0_i64; np];
        let mut affected = vec![false; np];
        for arc in &transition.inputs {
            let p = arc.place.0 as usize;
            deltas[p] -= arc.weight as i64;
            affected[p] = true;
        }
        for arc in &transition.outputs {
            let p = arc.place.0 as usize;
            deltas[p] += arc.weight as i64;
            affected[p] = true;
        }

        // Build effect constraints: m_p' = m_p + delta_p for affected places,
        // m_p' = m_p for frame (unaffected) places.
        let mut effect_conjuncts: Vec<ChcExpr> = Vec::with_capacity(np);
        for p in 0..np {
            if !affected[p] || deltas[p] == 0 {
                effect_conjuncts.push(ChcExpr::eq(primed_var_expr(p), var_expr(p)));
            } else {
                effect_conjuncts.push(ChcExpr::eq(
                    primed_var_expr(p),
                    ChcExpr::add(var_expr(p), ChcExpr::int(deltas[p])),
                ));
            }
        }

        // Combine guard + effect + non-negativity into the transition constraint
        let mut all_conjuncts = guard_conjuncts;
        all_conjuncts.extend(effect_conjuncts);
        all_conjuncts.extend(nonneg_primed.clone());
        let transition_constraint = ChcExpr::and_all(all_conjuncts);

        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, current_args.clone())],
                Some(transition_constraint),
            ),
            ClauseHead::Predicate(inv, primed_args.clone()),
        ));
    }

    // ── Stuttering clause (optional) ─────────────────────────────
    // Inv(m) /\ m' = m => Inv(m')
    if config.add_stuttering {
        let stutter_conjuncts: Vec<ChcExpr> = (0..np)
            .map(|p| ChcExpr::eq(primed_var_expr(p), var_expr(p)))
            .collect();
        let stutter_constraint = ChcExpr::and_all(stutter_conjuncts);
        problem.add_clause(HornClause::new(
            ClauseBody::new(vec![(inv, current_args.clone())], Some(stutter_constraint)),
            ClauseHead::Predicate(inv, primed_args.clone()),
        ));
    }

    // ── P-invariant strengthening (optional) ─────────────────────
    // For each P-invariant y: sum(y[p] * m[p]) = y^T * m0.
    // Added as: Inv(m) /\ sum(y[p] * m'[p]) = constant /\ frame => Inv(m')
    // Actually, we add them as additional constraints on the init and query
    // to strengthen the invariant discovery.
    let query_strengthening = if config.use_p_invariants {
        let invariants = compute_p_invariants(net);
        p_invariant_constraints(&invariants)
    } else {
        Vec::new()
    };

    // ── Symmetric Clause Learning (optional) ─────────────────────
    // Inv(m) => Inv(pi(m))
    if config.use_symmetry {
        if let Some(canon) = canonicalizer {
            for gen in canon.generators() {
                let mut permuted_args = Vec::with_capacity(np);
                for _i in 0..np {
                    // If place `i` moves to `gen[i]`, then the token count at `gen[i]`
                    // in the symmetric state `m'` is the token count at `i` in `m`.
                    // So m'_{gen[i]} = m_i.
                    // Meaning the i-th argument of the permuted head is current_args[gen[i]]?
                    // Let's verify: we want ClauseHead::Predicate(inv, args') such that
                    // args'[gen[i]] = current_args[i].
                    // Equivalently, args'[i] = current_args[inv_gen[i]].
                    // But in PetriCanonicalizer, canonicalize() does: temp[i] = marking[perm[i]];
                    // where perm maps indices to new indices.
                    // Wait, if temp[i] = marking[perm[i]], this means the new i-th element is the old perm[i]-th element.
                    // So we can just use current_args[gen[i]].
                }
                for i in 0..np {
                    permuted_args.push(current_args[gen[i]].clone());
                }
                problem.add_clause(HornClause::new(
                    ClauseBody::new(vec![(inv, current_args.clone())], None),
                    ClauseHead::Predicate(inv, permuted_args),
                ));
            }
        }
    }

    // ── Query clause ─────────────────────────────────────────────
    // Inv(m) /\ NOT safety(m) => false
    let mut query_conjuncts = Vec::with_capacity(1 + query_strengthening.len());
    query_conjuncts.push(encode_negated_predicate(property, net));
    query_conjuncts.extend(query_strengthening);
    problem.add_clause(HornClause::new(
        ClauseBody::new(
            vec![(inv, current_args)],
            Some(ChcExpr::and_all(query_conjuncts)),
        ),
        ClauseHead::False,
    ));

    problem
}

/// Solve a Petri net safety property using PDR/IC3 via ay-chc.
///
/// Returns `Safe` if an inductive invariant is found proving the property,
/// `Unsafe` if a counterexample is found, or `Unknown` if the solver
/// cannot determine the result within the time budget.
pub(crate) fn solve_petri_net_pdr(
    net: &PetriNet,
    property: &ResolvedPredicate,
    config: &PdrConfig,
    canonicalizer: Option<&crate::explorer::symmetry::PetriCanonicalizer>,
) -> PdrCheckResult {
    // Size gate (wide-net OOM guard). The CHC consecution encoding is
    // inherently O(num_places × num_transitions): every transition clause must
    // frame ALL places (`m_p' = m_p` for unaffected p) to keep the transition
    // relation sound, plus a per-place non-negativity term. On AirplaneLD-PT-
    // 4000 (28 019 places × 32 008 transitions) that is ~9×10⁸ retained CHC
    // nodes — well over 100 GB — built up front before any solving or deadline
    // poll. Such nets are far past what the CHC portfolio could discharge
    // anyway, so decline (Unknown) without materializing the dense problem:
    // PDR here is a best-effort seeder, and every caller treats Unknown as
    // "no seed / fall through" — declining can never change a verdict.
    let cells = net.num_places().saturating_mul(net.num_transitions());
    if cells > PDR_ENCODING_MAX_CELLS {
        eprintln!(
            "PDR: declining encode — {} places × {} transitions = {cells} cells \
             exceeds budget {PDR_ENCODING_MAX_CELLS} (dense CHC frame would OOM); \
             falling through",
            net.num_places(),
            net.num_transitions(),
        );
        return PdrCheckResult::Unknown;
    }

    let problem = encode_petri_net_pdr(net, property, config, canonicalizer);

    let adaptive_config = AdaptiveConfig::with_budget(config.time_budget, config.verbose);
    let solver = AdaptivePortfolio::new(problem, adaptive_config);
    let result = solver.solve();

    match result {
        VerifiedChcResult::Safe(_) => PdrCheckResult::Safe,
        VerifiedChcResult::Unsafe(_) => PdrCheckResult::Unsafe,
        VerifiedChcResult::Unknown(_) => bounded_exact_to_pdr_result(solve_bounded_exact(
            net,
            property,
            exact_fallback_deadline(config),
        )),
        _ => bounded_exact_to_pdr_result(solve_bounded_exact(
            net,
            property,
            exact_fallback_deadline(config),
        )),
    }
}

fn exact_fallback_deadline(config: &PdrConfig) -> Option<Instant> {
    let budget_deadline = config
        .exact_fallback_budget
        .map(|budget| Instant::now() + budget);
    match (config.exact_fallback_deadline, budget_deadline) {
        (Some(absolute), Some(budgeted)) => Some(absolute.min(budgeted)),
        (Some(absolute), None) => Some(absolute),
        (None, Some(budgeted)) => Some(budgeted),
        (None, None) => None,
    }
}

fn bounded_exact_to_pdr_result(outcome: BoundedExactOutcome) -> PdrCheckResult {
    match outcome {
        BoundedExactOutcome::Safe => PdrCheckResult::Safe,
        BoundedExactOutcome::Unsafe => PdrCheckResult::Unsafe,
        BoundedExactOutcome::Inconclusive(_) => PdrCheckResult::Unknown,
    }
}

/// Resolve CHC `Unknown` for structurally bounded nets by exhaustive reachability.
///
/// PDR is allowed to be inconclusive. When P-invariants give every place a
/// finite structural bound, exact BFS over that finite state space is a sound
/// fallback: a visited violation is `Unsafe`; exhausting all reachable markings
/// proves `Safe`. Large or only partially bounded nets remain `Unknown`.
pub(super) fn solve_bounded_exact(
    net: &PetriNet,
    property: &ResolvedPredicate,
    deadline: Option<Instant>,
) -> BoundedExactOutcome {
    if exact_fallback_deadline_expired(deadline) {
        return BoundedExactOutcome::Inconclusive(BoundedExactInconclusive::DeadlineExpired);
    }

    let invariants = compute_p_invariants(net);
    if exact_fallback_deadline_expired(deadline) {
        return BoundedExactOutcome::Inconclusive(BoundedExactInconclusive::DeadlineExpired);
    }

    let Some(bounds) = (0..net.num_places())
        .map(|p| structural_place_bound(&invariants, p))
        .collect::<Option<Vec<_>>>()
    else {
        return BoundedExactOutcome::Inconclusive(
            BoundedExactInconclusive::MissingStructuralBounds,
        );
    };

    if net
        .initial_marking
        .iter()
        .zip(&bounds)
        .any(|(&tokens, &bound)| tokens > bound)
    {
        return BoundedExactOutcome::Inconclusive(BoundedExactInconclusive::InitialExceedsBound);
    }

    if exact_fallback_deadline_expired(deadline) {
        return BoundedExactOutcome::Inconclusive(BoundedExactInconclusive::DeadlineExpired);
    }
    if !eval_predicate(property, &net.initial_marking, net) {
        return BoundedExactOutcome::Unsafe;
    }

    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();
    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());

    let mut deadline_counter: u32 = 0;
    let mut transition_deadline_counter: u32 = 0;
    while let Some(marking) = queue.pop_front() {
        deadline_counter = deadline_counter.wrapping_add(1);
        if deadline_counter % EXACT_BOUNDED_FALLBACK_DEADLINE_CHECK_INTERVAL == 0
            && exact_fallback_deadline_expired(deadline)
        {
            return BoundedExactOutcome::Inconclusive(BoundedExactInconclusive::DeadlineExpired);
        }

        for tidx in 0..net.num_transitions() {
            transition_deadline_counter = transition_deadline_counter.wrapping_add(1);
            if transition_deadline_counter
                % EXACT_BOUNDED_FALLBACK_TRANSITION_DEADLINE_CHECK_INTERVAL
                == 0
                && exact_fallback_deadline_expired(deadline)
            {
                return BoundedExactOutcome::Inconclusive(
                    BoundedExactInconclusive::DeadlineExpired,
                );
            }

            let tidx = TransitionIdx(tidx as u32);
            if !net.is_enabled(&marking, tidx) {
                continue;
            }

            // Fail-closed (#22): a token-count overflow means the successor is
            // not representable in u64 and therefore exceeds every structural
            // bound — decline rather than fabricate a wrapped marking.
            let Ok(successor) = net.fire(&marking, tidx) else {
                return BoundedExactOutcome::Inconclusive(
                    BoundedExactInconclusive::SuccessorExceedsBound,
                );
            };
            if successor
                .iter()
                .zip(&bounds)
                .any(|(&tokens, &bound)| tokens > bound)
            {
                return BoundedExactOutcome::Inconclusive(
                    BoundedExactInconclusive::SuccessorExceedsBound,
                );
            }

            if exact_fallback_deadline_expired(deadline) {
                return BoundedExactOutcome::Inconclusive(
                    BoundedExactInconclusive::DeadlineExpired,
                );
            }
            if !eval_predicate(property, &successor, net) {
                return BoundedExactOutcome::Unsafe;
            }

            if seen.insert(successor.clone()) {
                if seen.len() > EXACT_BOUNDED_FALLBACK_STATE_LIMIT {
                    return BoundedExactOutcome::Inconclusive(
                        BoundedExactInconclusive::StateLimitExceeded,
                    );
                }
                queue.push_back(successor);
            }
        }
    }

    if exact_fallback_deadline_expired(deadline) {
        return BoundedExactOutcome::Inconclusive(BoundedExactInconclusive::DeadlineExpired);
    }
    BoundedExactOutcome::Safe
}

fn exact_fallback_deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

// ── Predicate encoding ──────────────────────────────────────────────

/// Encode a resolved predicate as a `ChcExpr` over current-state marking
/// variables (`m_0`, `m_1`, ...).
fn encode_predicate_expr(pred: &ResolvedPredicate, net: &PetriNet) -> ChcExpr {
    match pred {
        ResolvedPredicate::True => ChcExpr::Bool(true),
        ResolvedPredicate::False => ChcExpr::Bool(false),
        ResolvedPredicate::And(children) => {
            let exprs: Vec<ChcExpr> = children
                .iter()
                .map(|c| encode_predicate_expr(c, net))
                .collect();
            ChcExpr::and_all(exprs)
        }
        ResolvedPredicate::Or(children) => {
            let exprs: Vec<ChcExpr> = children
                .iter()
                .map(|c| encode_predicate_expr(c, net))
                .collect();
            ChcExpr::or_all(exprs)
        }
        ResolvedPredicate::Not(inner) => ChcExpr::not(encode_predicate_expr(inner, net)),
        ResolvedPredicate::IntLe(left, right) => {
            ChcExpr::le(encode_int_expr(left), encode_int_expr(right))
        }
        ResolvedPredicate::IsFireable(transitions) => {
            let disjuncts: Vec<ChcExpr> = transitions
                .iter()
                .map(|t_idx| {
                    let transition = &net.transitions[t_idx.0 as usize];
                    if transition.inputs.is_empty() {
                        return ChcExpr::Bool(true);
                    }
                    let guards: Vec<ChcExpr> = transition
                        .inputs
                        .iter()
                        .map(|arc| {
                            ChcExpr::ge(
                                var_expr(arc.place.0 as usize),
                                ChcExpr::int(arc.weight as i64),
                            )
                        })
                        .collect();
                    ChcExpr::and_all(guards)
                })
                .collect();
            ChcExpr::or_all(disjuncts)
        }
    }
}

/// Encode a resolved integer expression as a `ChcExpr`.
fn encode_int_expr(expr: &ResolvedIntExpr) -> ChcExpr {
    match expr {
        ResolvedIntExpr::Constant(value) => ChcExpr::int(*value as i64),
        ResolvedIntExpr::TokensCount(places) => {
            if places.is_empty() {
                ChcExpr::int(0)
            } else if places.len() == 1 {
                var_expr(places[0].0 as usize)
            } else {
                let terms: Vec<ChcExpr> = places.iter().map(|p| var_expr(p.0 as usize)).collect();
                // Build a left-associative sum via repeated add
                terms
                    .into_iter()
                    .reduce(ChcExpr::add)
                    .unwrap_or_else(|| ChcExpr::int(0))
            }
        }
    }
}

/// Encode the negation of a safety predicate as a `ChcExpr`.
///
/// The query clause asserts `Inv(m) /\ NOT safety(m) => false`, so we need
/// the negation of the property.
fn encode_negated_predicate(pred: &ResolvedPredicate, net: &PetriNet) -> ChcExpr {
    ChcExpr::not(encode_predicate_expr(pred, net))
}

// ── P-invariant strengthening ───────────────────────────────────────

/// Encode current-state P-invariant equalities as query-strengthening facts.
///
/// Every reachable marking satisfies these equalities, so adding them to the
/// safety query only prunes unreachable over-approximation states from `Inv`.
fn p_invariant_constraints(invariants: &[PInvariant]) -> Vec<ChcExpr> {
    invariants
        .iter()
        .filter_map(|inv| {
            // Build the P-invariant expression: sum(y[p] * m[p]) over the
            // sparse support.
            let terms: Vec<ChcExpr> = inv
                .support()
                .map(|(p, w)| {
                    if w == 1 {
                        var_expr(p)
                    } else {
                        ChcExpr::mul(ChcExpr::int(w as i64), var_expr(p))
                    }
                })
                .collect();

            if terms.is_empty() {
                return None;
            }

            let sum = terms
                .into_iter()
                .reduce(ChcExpr::add)
                .expect("non-empty terms");

            Some(ChcExpr::eq(sum, ChcExpr::int(inv.token_count as i64)))
        })
        .collect()
}

#[cfg(test)]
#[path = "pdr_encoding_tests.rs"]
mod tests;
