// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! LTL-to-Buchi automaton conversion and product emptiness checking.
//!
//! Implements a simplified GPVW (Gerth-Peled-Vardi-Wolper) on-the-fly
//! construction for converting LTL formulas to Generalized Buchi Automata,
//! then checks language emptiness via product construction with the system
//! state graph and SCC-based accepting cycle detection.

mod atoms;
mod gba;
mod nnf;
mod on_the_fly;
mod product;

#[cfg(test)]
pub(crate) use atoms::resolve_atom;
pub(crate) use atoms::resolve_atom_with_aliases;
pub(crate) use atoms::LtlContext;
pub(crate) use gba::{Gba, GbaStateId, GbaTransition};
pub(crate) use nnf::{to_nnf, LtlNnf};

pub(crate) use on_the_fly::PorContext;

use gba::build_gba;
use nnf::negate;
use on_the_fly::on_the_fly_product_emptiness;
use product::product_has_accepting_cycle;

/// Check if A(φ) holds — i.e., all paths from state 0 satisfy φ.
///
/// Returns `Some(true)` if the formula holds, `Some(false)` if a
/// counterexample exists, or `None` if the product graph exceeded the
/// size limit (result is inconclusive).
///
/// **Legacy path**: requires a pre-built [`crate::explorer::FullReachabilityGraph`].
/// New code should prefer [`check_ltl_on_the_fly`] which computes system
/// successors lazily.
pub(crate) fn check_ltl_formula(formula: &LtlNnf, ctx: &LtlContext<'_>) -> Option<bool> {
    // We want to check A(φ). Negate to get ¬φ and check E(¬φ) = ∅.
    let neg = negate(formula);

    // Build the Generalized Buchi Automaton for ¬φ.
    let gba = build_gba(&neg);

    // Check if the product of the system × GBA has an accepting cycle
    // reachable from the initial state.
    // None = product overflow (inconclusive), Some(true) = accepting cycle found,
    // Some(false) = no accepting cycle.
    let has_cycle = product_has_accepting_cycle(&gba, ctx, None)?;
    Some(!has_cycle)
}

/// Check `A(formula)` on an already completed full reachability graph.
///
/// This is an exact fallback for cases where the lazy on-the-fly product times
/// out, but the system graph itself is small enough to materialize once and
/// reuse across several residual LTL formulas.
pub(crate) fn check_ltl_on_full_graph(
    formula: &LtlNnf,
    full: &crate::explorer::FullReachabilityGraph,
    net: &crate::petri_net::PetriNet,
    atoms: &[crate::resolved_predicate::ResolvedPredicate],
    deadline: Option<std::time::Instant>,
) -> Option<bool> {
    if !full.graph.completed {
        return None;
    }

    let ctx = LtlContext::new(atoms.to_vec(), full, net);
    check_ltl_formula_with_deadline(formula, &ctx, deadline)
}

fn check_ltl_formula_with_deadline(
    formula: &LtlNnf,
    ctx: &LtlContext<'_>,
    deadline: Option<std::time::Instant>,
) -> Option<bool> {
    let neg = negate(formula);
    let gba = build_gba(&neg);
    let has_cycle = product_has_accepting_cycle(&gba, ctx, deadline)?;
    Some(!has_cycle)
}

/// Build the GBA used to search for counterexamples to `formula`.
pub(crate) fn build_ltl_counterexample_gba(formula: &LtlNnf) -> Gba {
    let neg = negate(formula);
    build_gba(&neg)
}

/// Convert the counterexample [`Gba`] for `formula` into the engine-agnostic
/// `tla_dd::symbolic_ltl::SymbolicGba` for the symbolic LTL product lane.
///
/// `translate_atom(i)` lowers atom index `i` (into the LTL atom table that the
/// GBA's `pos_atoms`/`neg_atoms` reference) to a `tla_dd::DdPredicate`, or
/// `None` if the atom is unsupported by the DD encoding — in which case the
/// whole conversion declines (`None`), keeping the lane FAIL-CLOSED.
///
/// The conversion is structure-preserving (same states, same atom-guarded
/// transitions, same generalized acceptance with mixed state/edge sets), so
/// the symbolic product is the SAME automaton the explicit checker uses; the
/// `examinations::ltl_symbolic` differential test confirms the two agree.
#[cfg(feature = "dd-backend")]
pub(crate) fn gba_to_symbolic(
    formula: &LtlNnf,
    translate_atom: impl Fn(usize) -> Option<tla_dd::DdPredicate>,
) -> Option<tla_dd::symbolic_ltl::SymbolicGba> {
    use tla_dd::symbolic_ltl::{SymbolicGba, SymbolicGbaTransition};

    let gba = build_ltl_counterexample_gba(formula);

    // Collect every atom index the GBA references and lower it once.
    let mut max_atom: i64 = -1;
    let scan = |t: &GbaTransition, max_atom: &mut i64| {
        for &a in t.pos_atoms.iter().chain(t.neg_atoms.iter()) {
            *max_atom = (*max_atom).max(a as i64);
        }
    };
    for t in &gba.initial_transitions {
        scan(t, &mut max_atom);
    }
    for ts in &gba.transitions {
        for t in ts {
            scan(t, &mut max_atom);
        }
    }
    let num_atoms = (max_atom + 1) as usize;
    let mut atoms = Vec::with_capacity(num_atoms);
    for i in 0..num_atoms {
        atoms.push(translate_atom(i)?);
    }

    let num_accept = gba.acceptance.len();
    let conv = |t: &GbaTransition| SymbolicGbaTransition {
        pos_atoms: t.pos_atoms.clone(),
        neg_atoms: t.neg_atoms.clone(),
        successor: t.successor,
        // The GBA always sets `edge_accept` to length `num_accept`; mirror it
        // (default `false` defensively for any short vector).
        edge_accept: (0..num_accept)
            .map(|i| t.edge_accept.get(i).copied().unwrap_or(false))
            .collect(),
    };

    let initial_transitions = gba.initial_transitions.iter().map(conv).collect();
    let transitions = gba
        .transitions
        .iter()
        .map(|ts| ts.iter().map(conv).collect())
        .collect();
    let acceptance = gba
        .acceptance
        .iter()
        .map(|set| {
            let mut v: Vec<u32> = set.iter().copied().collect();
            v.sort_unstable();
            v
        })
        .collect();

    Some(SymbolicGba {
        num_states: gba.num_states,
        atoms,
        initial_transitions,
        transitions,
        acceptance,
    })
}

/// Whether the counterexample automaton for `formula` contains a Release
/// obligation.
pub(crate) fn ltl_counterexample_contains_release(formula: &LtlNnf) -> bool {
    let neg = negate(formula);
    ltl_nnf_contains_release(&neg)
}

fn ltl_nnf_contains_release(formula: &LtlNnf) -> bool {
    match formula {
        LtlNnf::True | LtlNnf::False | LtlNnf::Atom(_) | LtlNnf::NegAtom(_) => false,
        LtlNnf::And(items) | LtlNnf::Or(items) => items.iter().any(ltl_nnf_contains_release),
        LtlNnf::Next(inner) => ltl_nnf_contains_release(inner),
        LtlNnf::Until(left, right) => {
            ltl_nnf_contains_release(left) || ltl_nnf_contains_release(right)
        }
        LtlNnf::Release(_, _) => true,
    }
}

/// On-the-fly variant of [`check_ltl_formula`]: checks A(φ) without a
/// pre-built reachability graph.
///
/// System successors are computed lazily by firing transitions on `net`
/// (the reduced Petri net). Atom predicates are evaluated by expanding
/// reduced markings to `original_net` space via `reduced`.
///
/// Returns `Some(true)` if φ holds, `Some(false)` if ¬φ has an accepting
/// run, or `None` if the system-marking or product-state budget was exceeded.
pub(crate) fn check_ltl_on_the_fly(
    formula: &LtlNnf,
    net: &crate::petri_net::PetriNet,
    reduced: &crate::reduction::ReducedNet,
    original_net: &crate::petri_net::PetriNet,
    atoms: &[crate::resolved_predicate::ResolvedPredicate],
    por: Option<&PorContext>,
    max_system_states: usize,
    deadline: Option<std::time::Instant>,
) -> Result<Option<bool>, crate::error::PnmlError> {
    let gba = build_ltl_counterexample_gba(formula);
    let has_cycle = on_the_fly_product_emptiness(
        &gba,
        net,
        reduced,
        original_net,
        atoms,
        por,
        max_system_states,
        deadline,
    )?;
    Ok(has_cycle.map(|value| !value))
}

#[cfg(test)]
#[path = "integration_tests.rs"]
mod integration_tests;
