// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Symbolic-LTL product data types: an engine-agnostic Generalized Büchi
//! Automaton lowering for the DD lane.
//!
//! This module once hosted a BDD LTL emptiness checker
//! (`symbolic_ltl_has_accepting_cycle` — an Emerson–Lei fair-cycle fixpoint
//! over the OxiDD-based `DdReachability` `system × GBA` product). That evaluator
//! has been removed with the `oxidd` dependency; the native Büchi-product LTL
//! engine now lives in `tla-bdd`.
//!
//! What remains here are the two **data types** that engine consumes:
//! [`SymbolicGba`] and [`SymbolicGbaTransition`] — a faithful, engine-agnostic
//! copy of the petri-side `buchi::gba::Gba`: the same state set, the same
//! atom-guarded transitions (guards are [`DdPredicate`]s over the system
//! marking, GPVW convention: read against the successor marking), and the same
//! generalized acceptance with mixed state-based ([`SymbolicGba::acceptance`])
//! and edge-based ([`SymbolicGbaTransition::edge_accept`]) sets.
//!
//! `tla_petri::buchi` lowers its `Gba` for the **negated** property into these
//! types; the tla-bdd engine builds the `system × GBA` product (with the
//! deadlock stutter self-loop for maximal-path semantics) and runs the
//! fair-cycle emptiness check, declining fail-closed on any unsupported atom /
//! over-budget / OOM condition rather than guessing a verdict.

use crate::DdPredicate;

/// A single GBA transition, lowered to DD land.
///
/// `pos_atoms` / `neg_atoms` index into [`SymbolicGba::atoms`]; the guard is
/// `⋀ pos hold ∧ ⋀ neg ¬hold`, evaluated against the **successor** system
/// marking (GPVW convention).
#[derive(Debug, Clone)]
pub struct SymbolicGbaTransition {
    /// Atoms (indices into [`SymbolicGba::atoms`]) that must hold.
    pub pos_atoms: Vec<usize>,
    /// Atoms that must NOT hold.
    pub neg_atoms: Vec<usize>,
    /// Successor GBA state id.
    pub successor: u32,
    /// Edge-based acceptance: `edge_accept[i]` is true iff this transition
    /// discharges acceptance set `i`. Parallel to [`SymbolicGba::acceptance`].
    pub edge_accept: Vec<bool>,
}

/// A Generalized Büchi Automaton, lowered to DD land.
///
/// A faithful, engine-agnostic copy of the petri-side `buchi::gba::Gba`: same
/// state set, same atom-guarded transitions, same generalized acceptance with
/// mixed state-based ([`Self::acceptance`]) and edge-based
/// ([`SymbolicGbaTransition::edge_accept`]) sets.
#[derive(Debug, Clone)]
pub struct SymbolicGba {
    /// Number of GBA states.
    pub num_states: u32,
    /// Atom predicates (`DdPredicate` over the system marking), referenced by
    /// index from every transition's `pos_atoms` / `neg_atoms`.
    pub atoms: Vec<DdPredicate>,
    /// Transitions from the virtual initial node (guards read the INITIAL
    /// system marking).
    pub initial_transitions: Vec<SymbolicGbaTransition>,
    /// `transitions[q]` = outgoing transitions of GBA state `q` (guards read
    /// the successor marking).
    pub transitions: Vec<Vec<SymbolicGbaTransition>>,
    /// State-based acceptance: `acceptance[i]` is the set of GBA states that
    /// are accepting for set `i`.
    pub acceptance: Vec<Vec<u32>>,
}
