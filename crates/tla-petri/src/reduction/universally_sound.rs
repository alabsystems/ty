// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Universally-sound structural pre-reductions.
//!
//! A *universally-sound* reduction is one whose verdict is preserved for
//! **every** MCC examination (reachability, CTL, LTL, upper-bounds, deadlock,
//! one-safe, state-space, quasi-liveness, liveness, stable-marking). Unlike the
//! mode-gated [`reduce_iterative_structural_with_mode`](super::reduce_iterative_structural_with_mode)
//! pipeline — which mixes in rules (agglomeration, source-place, token-cycle,
//! sink-transition) that change the marking or the firing relation and are only
//! sound for a *subset* of the property classes — this module exposes the
//! narrow, verdict-preserving-for-all reductions that can be applied **before**
//! any heavy engine (explicit BFS / symbolic / SMT) runs.
//!
//! Currently this is **dead-transition removal**, reusing the proven
//! never-enabled LP oracle ([`crate::lp_state_equation::lp_all_dead_transitions`])
//! in union with the cheap structural detector ([`super::analyze`]).
//!
//! # Why dead-transition removal is verdict-preserving for every examination
//!
//! A transition `t` is *dead* iff it is enabled in no reachable marking. The LP
//! oracle proves this soundly: "t enabled" is infeasible over a polytope that
//! is a superset of the reachable markings, so no reachable marking enables
//! `t`. A dead transition therefore fires on **no** run, so deleting it (and
//! its incident arcs) removes **no** edge from the reachable transition system.
//! The reduced net's reachable marking set, step relation, edge set, and every
//! place's reachable token range are **identical** to the original's — only the
//! never-taken transition disappears. With the reachable state graph identical
//! and **places preserved exactly** (this module removes transitions only, never
//! places), every state / path / branching property atom evaluates identically,
//! so every examination verdict is preserved:
//!
//! - reachability / cardinality / CTL / LTL / upper-bounds: same reachable
//!   markings ⇒ same atom truth on the same graph;
//! - deadlock: a marking is a deadlock iff *no* transition is enabled there; a
//!   dead `t` is never enabled, so it never affects whether a marking is a
//!   deadlock ⇒ same deadlock set;
//! - state-space: same reachable markings (counts), same edges (a dead `t`
//!   contributes zero outgoing edges from every reachable marking), same
//!   `max_token_in_place` / `max_token_sum` (identical markings, identical place
//!   set);
//! - quasi-liveness / liveness: a dead `t` is itself a *witness* that the net
//!   is **not** quasi-live / not live — callers that consume this reduction must
//!   read `report.dead_transitions` and decide FALSE rather than silently
//!   exploring the survivors (see the per-caller guards). For a fireability
//!   atom over a removed `t`, the atom is `False` (t can never fire).
//!
//! A fireability *property atom* `is-fireable(t)` over a removed `t` would lose
//! its referent, so [`reduce_dead_transitions_only`] records the removed set in
//! `report.dead_transitions`; callers with transition-referencing atoms either
//! resolve the atom to `False` or exclude such transitions from removal. The
//! current consumer (StateSpace) has **no** transition atoms, so this concern
//! does not arise there.

use std::time::{Duration, Instant};

use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};

use super::{ReducedNet, ReductionReport};

/// Wall-clock slice reserved for the universally-sound pre-reduction analysis.
///
/// The LP sweep does not poll a deadline inside a single per-transition solve,
/// so on a transition-heavy net the sweep can run for a while before the next
/// between-transition poll. Capping it at this slice (or the global deadline,
/// whichever is sooner) reserves the remaining budget for the heavy engine.
/// Abandoning the sweep early yields a (sound) under-approximation of the dead
/// set, so the bound is strictly verdict-preserving.
pub(crate) const UNIVERSALLY_SOUND_REDUCTION_CAP: Duration = Duration::from_secs(5);

/// Soft deadline for the pre-reduction sweep: the sooner of [`UNIVERSALLY_SOUND_REDUCTION_CAP`]
/// from now and the caller's global deadline.
fn soft_deadline(global: Option<Instant>) -> Option<Instant> {
    let cap = Instant::now() + UNIVERSALLY_SOUND_REDUCTION_CAP;
    Some(match global {
        Some(global) => cap.min(global),
        None => cap,
    })
}

/// The complete set of transitions proven never enabled in any reachable
/// marking: the union of the cheap structural detector ([`super::analyze`],
/// uncapped) and the strong joint-enabling + trap LP oracle
/// ([`crate::lp_state_equation::lp_all_dead_transitions`], size-guarded and
/// wall-capped at `deadline`).
///
/// Both sources are one-directional sound (they only ever report a transition
/// as dead when it provably can never fire), so their union is exactly a set of
/// genuinely-dead transitions — removing any of them is verdict-preserving.
#[must_use]
pub(crate) fn proven_dead_transitions(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> Vec<TransitionIdx> {
    let nt = net.num_transitions();
    let mut is_dead = vec![false; nt];

    // Structural dead set (cheap, uncapped): cascading no-producer/under-marked
    // input places plus initially-marked-trap dead transitions.
    for &TransitionIdx(t) in &super::analyze(net).dead_transitions {
        if (t as usize) < nt {
            is_dead[t as usize] = true;
        }
    }

    // Strong LP dead set (joint enabling conjunction + traps). Size-guarded and
    // wall-capped inside; returns a sound under-approximation on cap/deadline.
    for TransitionIdx(t) in crate::lp_state_equation::lp_all_dead_transitions(net, deadline) {
        if (t as usize) < nt {
            is_dead[t as usize] = true;
        }
    }

    is_dead
        .iter()
        .enumerate()
        .filter_map(|(i, &dead)| dead.then_some(TransitionIdx(i as u32)))
        .collect()
}

/// Build a [`ReducedNet`] that removes ONLY the proven-dead transitions,
/// preserving **every** place (identity place mapping). The returned net's
/// reachable state graph is identical to the original's modulo the never-taken
/// dead transitions, so it is verdict-preserving for every examination whose
/// property does not quantify over the removed transitions (StateSpace,
/// reachability/CTL/LTL/bounds over cardinality atoms, deadlock).
///
/// `deadline` is the caller's global deadline; the internal LP sweep is capped
/// at the sooner of [`UNIVERSALLY_SOUND_REDUCTION_CAP`] and that deadline. When
/// no dead transition is found (or the size guard / cap fires before any is
/// proven), the identity reduction is returned — a verdict-preserving
/// fall-through with no structural change.
#[must_use]
pub(crate) fn reduce_dead_transitions_only(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> ReducedNet {
    let dead = proven_dead_transitions(net, soft_deadline(deadline));
    build_transition_filtered(net, &dead)
}

/// Construct a transition-filtered [`ReducedNet`]: drop the transitions in
/// `remove`, keep every place. Place maps are identity; transition maps skip
/// the removed indices and renumber the survivors densely.
#[must_use]
fn build_transition_filtered(net: &PetriNet, remove: &[TransitionIdx]) -> ReducedNet {
    let nt = net.num_transitions();
    let mut removed = vec![false; nt];
    for &TransitionIdx(t) in remove {
        if (t as usize) < nt {
            removed[t as usize] = true;
        }
    }
    if !removed.iter().any(|&r| r) {
        return ReducedNet::identity(net);
    }

    let np = net.num_places();
    let place_map = (0..np).map(|i| Some(PlaceIdx(i as u32))).collect();
    let place_unmap = (0..np).map(|i| PlaceIdx(i as u32)).collect();
    let place_scales = vec![1u64; np];

    let mut transition_map = vec![None; nt];
    let mut transition_unmap = Vec::new();
    let mut transitions = Vec::new();
    for (i, transition) in net.transitions.iter().enumerate() {
        if removed[i] {
            continue;
        }
        let new_idx = TransitionIdx(transitions.len() as u32);
        transition_map[i] = Some(new_idx);
        transition_unmap.push(TransitionIdx(i as u32));
        transitions.push(transition.clone());
    }

    let reduced_net = PetriNet {
        name: net.name.clone(),
        places: net.places.clone(),
        transitions,
        initial_marking: net.initial_marking.clone(),
    };

    let mut dead_transitions: Vec<TransitionIdx> = (0..nt)
        .filter(|&i| removed[i])
        .map(|i| TransitionIdx(i as u32))
        .collect();
    dead_transitions.sort_by_key(|t| t.0);

    ReducedNet {
        net: reduced_net,
        place_map,
        place_unmap,
        place_scales,
        transition_map,
        transition_unmap,
        constant_values: Vec::new(),
        reconstructions: Vec::new(),
        report: ReductionReport {
            dead_transitions,
            ..ReductionReport::default()
        },
    }
}
