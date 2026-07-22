// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Symbolic-COLORED StateSpace engine (FIRST INCREMENT, gate-only).
//!
//! Computes the four MCC `StateSpace` metrics of a colored (HLPN / symmetric)
//! net by building the reachable-marking set as a single compact MDD over the
//! unfolded `(place, color)` level encoding and saturating it — WITHOUT first
//! materializing the unfolded P/T [`crate::petri_net::PetriNet`] (its place /
//! transition / alias tables). The colored marking is encoded EXACTLY as the
//! unfolded `(place, color)` marking, identical to [`crate::unfold`]'s
//! `place_map`, so the result is the StateSpace of the unfolded net by
//! construction.
//!
//! # The only new symbolic primitive
//!
//! Firing a colored transition over ALL of its variable bindings is one
//! symbolic image: [`tla_mdd::colored_transition_image`] = the union over the
//! transition's (guard-filtered) bindings of each binding's per-place P/T image
//! ([`tla_mdd::transition_image_pub`]). The per-binding `(pre, post)` over the
//! `(place, color)` level space are resolved by the SAME unfold resolvers
//! (`enumerate_bindings` + `resolve_arcs_for_binding` via [`ColoredMddBuild`]),
//! so a single binding is byte-identical to the corresponding unfolded P/T
//! transition.
//!
//! # V1 SUB-CLASS (fail-closed-DECLINE anything outside)
//!
//! - Sorts: `CyclicEnum` / `FiniteIntRange` / `Dot` / `Product` of those (the
//!   unfolder already declines other sorts).
//! - Arc inscriptions: `NumberOf` / `All` / `Add` (sums of `<n>'<term>`). A
//!   `Subtract` arc DECLINES (excluded in v1).
//! - Guards: handled VERBATIM by the unfold `enumerate_bindings` (conjunction /
//!   disjunction / equality / inequality / successor / predecessor over
//!   variables / constants); a guard the unfolder cannot resolve already
//!   declines there.
//! - Per-slot bound: the engine admits only **token-non-increasing** nets
//!   (every per-binding transition has `Σ post <= Σ pre`), for which the total
//!   token count never grows, so `bound[slot] = Σ initial_marking` is a SOUND
//!   over-approximation of every slot's reachable maximum (no reachable marking
//!   is ever truncated). A net with any token-PRODUCING binding DECLINES (v1
//!   sub-class boundary). This covers the conserving COL families
//!   (Philosophers / TokenRing / DatabaseWithMutex).
//!
//! # SOUNDNESS (proof-by-oracle, ABSOLUTE)
//!
//! The metric bundle is published ONLY through the gate-only / test path, where
//! it is cross-checked EQUAL to the trusted P/T MDD StateSpace on the EXPLICITLY
//! unfolded net (`unfold_to_pt` + `tla_dd::build_sound_dd_spec` +
//! `MddNet::state_space_metrics`) — 0 disagreements required — on the colored
//! test battery and a proptest over random small colored nets in the sub-class.
//! Any per-binding resolution error, out-of-sub-class construct, overflow, node
//! budget or deadline is fail-closed (`Err`), never a wrong count.

#![cfg(feature = "dd-backend")]

use std::time::Instant;

use tla_bignum::{BigUint, ToPrimitive, Zero};
use tla_mdd::{
    colored_transition_image_quantified, fireable_set, max_token_in_place_of, max_token_sum_of,
    BindingDriverError, CountError, MddNet, MddRef, MddStateSpaceMetrics, MddStore,
};

use crate::error::PnmlError;
use crate::hlpnml::{ColorExpr, ColoredNet};
use crate::unfold::symbolic_build::{BindingQuantDriver, ColoredMddBuild, MAX_COLORED_MDD_LEVELS};

/// Why the symbolic-colored engine declined (fail-closed; never a wrong count).
#[derive(Debug)]
pub(crate) enum SymbolicColoredError {
    /// The colored net (or one of its constructs) is outside the v1 sub-class
    /// the engine supports — fall through to the existing placeholder
    /// CANNOT_COMPUTE. Carries the unfold error or a sub-class reason.
    OutOfSubclass(String),
    /// The MDD reachability / metric computation failed closed (overflow, node
    /// budget, or deadline).
    Mdd(CountError),
}

impl From<PnmlError> for SymbolicColoredError {
    fn from(e: PnmlError) -> Self {
        SymbolicColoredError::OutOfSubclass(format!("{e:?}"))
    }
}

impl From<CountError> for SymbolicColoredError {
    fn from(e: CountError) -> Self {
        SymbolicColoredError::Mdd(e)
    }
}

impl From<BindingDriverError> for SymbolicColoredError {
    fn from(e: BindingDriverError) -> Self {
        match e {
            // An out-of-sub-class construct surfaced by the binding-quantified
            // driver's resolvers ⇒ recoverable DECLINE (fall back / CANNOT_COMPUTE).
            BindingDriverError::OutOfSubclass(r) => SymbolicColoredError::OutOfSubclass(r),
            // A resource cap inside the quantified image ⇒ map to the MDD
            // fail-closed variant (DECLINE), never a partial count.
            BindingDriverError::ResourceCap(r) => {
                SymbolicColoredError::Mdd(CountError::ResourceCap(r))
            }
        }
    }
}

/// Build the symbolic-colored `MddNet` for `colored` (one MDD level per
/// `(place, color)` slot, one `MddTransition` per guard-satisfying binding), or
/// DECLINE (fail-closed) when the net is outside the v1 sub-class.
///
/// This bypasses the unfolder's P/T `MAX_UNFOLDED_PLACES` /
/// `MAX_UNFOLDED_TRANSITIONS` materialization caps (no `TransitionInfo` / alias
/// tables are built) — the actual win for nets whose unfolded P/T form is too
/// large to materialize but whose reachable set is a compact MDD.
pub(crate) fn build_colored_mdd_net(colored: &ColoredNet) -> Result<MddNet, SymbolicColoredError> {
    // Reject Subtract arc inscriptions up front (excluded in v1): the per-color
    // monus is not part of the supported `NumberOf`/`All`/`Add` arc grammar.
    for arc in &colored.arcs {
        if contains_subtract(&arc.inscription) {
            return Err(SymbolicColoredError::OutOfSubclass(
                "Subtract arc inscription (excluded in v1)".to_string(),
            ));
        }
    }

    let build = ColoredMddBuild::new(colored)?;
    // Cap the level count so a pathological colored net cannot allocate an
    // astronomically wide MDD store. (Mirrors the unfolder's place cap; the
    // builder also fail-closes on per-binding resolution errors.)
    if build.num_levels() > MAX_COLORED_MDD_LEVELS {
        return Err(SymbolicColoredError::OutOfSubclass(format!(
            "{} (place,color) levels exceeds the symbolic cap {MAX_COLORED_MDD_LEVELS}",
            build.num_levels()
        )));
    }

    let initial_marking = build.initial_marking().to_vec();
    let transitions = build.build_transitions()?;

    // V1 sub-class: admit only PER-SORT token-non-increasing nets. Arcs are
    // sort-typed, so a token of sort `S` can only occupy a slot of an `S`-place;
    // if every transition is per-sort non-increasing, each sort's TOTAL token
    // count is invariant, and the per-sort total is a SOUND per-slot bound (no
    // slot can exceed it ⇒ no reachable marking is truncated ⇒ all four metrics
    // are EXACT). A token-PRODUCING or cross-sort-PUMPING transition makes a
    // sort's total grow, which could truncate a reachable marking — DECLINE.
    if let Some((ti, sort)) = build.first_sort_increasing_transition(&transitions) {
        return Err(SymbolicColoredError::OutOfSubclass(format!(
            "binding-transition {ti} increases sort '{sort}' token total (token-producing or \
             cross-sort-pumping); v1 admits only per-sort token-non-increasing nets"
        )));
    }
    // Tight, sound per-slot bound = the slot's sort token total (conserved).
    let bounds = build.sound_per_slot_bounds();

    Ok(MddNet {
        bounds,
        initial_marking,
        transitions,
    })
}

/// Compute the four `StateSpace` metrics of `colored` symbolically (gate-only).
///
/// `deadline` is an optional wall-clock cap the underlying saturation engine
/// declines (fail-closed) rather than overrun. Returns the metric bundle, or a
/// [`SymbolicColoredError`] on any out-of-sub-class construct / resource cap.
pub(crate) fn colored_state_space_metrics(
    colored: &ColoredNet,
    deadline: Option<Instant>,
) -> Result<MddStateSpaceMetrics, SymbolicColoredError> {
    let net = build_colored_mdd_net(colored)?;
    Ok(net.state_space_metrics(deadline)?)
}

// ===========================================================================
// THE BINDING-QUANTIFIED StateSpace PATH
// ===========================================================================
//
// `colored_state_space_metrics` (v1) materializes one `MddTransition` per
// guard-satisfying binding via `build_transitions` → `enumerate_bindings`, so it
// is bounded by `MAX_BINDING_ITERATIONS` (50M) / `MAX_UNFOLDED_TRANSITIONS`
// (500k) and DECLINES on BART-scale colored nets (≈ 1.4 BILLION bindings).
//
// The QUANTIFIED path below NEVER materializes the binding list. It builds the
// reachable set by a breadth-first relational-product fixpoint whose per-colored-
// transition image is `tla_mdd::colored_transition_image_quantified` — a DD
// recursion that branches the binding variables, prunes guard-killed sub-trees,
// and shares the per-binding image across identical marking-effects. So a net
// whose binding count blows the cap but whose reachable SET is a compact MDD is
// decided where v1 cannot.
//
// SOUNDNESS: a quantified leaf binding's `(pre,post)` is byte-identical to the
// enumerate path's `MddTransition` for that binding (same `resolve_arcs_for_
// binding`), so the quantified image == the v1 enumerated image as a SET on every
// net v1 can enumerate (the differential gate). The reachability fixpoint is the
// SAME monotone breadth-first relprod the trusted `MddNet::reachable_count_relprod`
// runs; the four metrics are read off the resulting set with the SAME functions
// the `MddNet` metric path uses (`max_token_*`, `fireable_set`), and `edge_count`
// is summed binding-quantified (`Σ_b |R ∩ Fireable(b)|`). Fail-closed on the
// node budget / deadline / out-of-sub-class, never a partial count.

/// Hard ceiling on live interior MDD nodes for the quantified path (matches the
/// symbolic engine's posture). DECLINE rather than OOM.
const MAX_INTERIOR_NODES: usize = 8_000_000;
/// Iteration backstop for the breadth-first quantified fixpoint (monotone, so it
/// always converges; this guards a logic bug, not a semantic limit).
const MAX_ROUNDS: u32 = 100_000_000;

/// Compute the four `StateSpace` metrics of `colored` via the BINDING-QUANTIFIED
/// path — branching binding variables symbolically instead of enumerating them,
/// so it decides nets whose binding count exceeds the enumerate caps.
///
/// Returns the metric bundle, or a [`SymbolicColoredError`] on any
/// out-of-sub-class construct / resource cap / deadline (fail-closed; never a
/// wrong count).
pub(crate) fn colored_state_space_metrics_quantified(
    colored: &ColoredNet,
    deadline: Option<Instant>,
) -> Result<MddStateSpaceMetrics, SymbolicColoredError> {
    // Reject Subtract arcs up front (excluded in v1), same as the enumerate path.
    for arc in &colored.arcs {
        if contains_subtract(&arc.inscription) {
            return Err(SymbolicColoredError::OutOfSubclass(
                "Subtract arc inscription (excluded in v1)".to_string(),
            ));
        }
    }

    let build = ColoredMddBuild::new(colored)?;
    if build.num_levels() > MAX_COLORED_MDD_LEVELS {
        return Err(SymbolicColoredError::OutOfSubclass(format!(
            "{} (place,color) levels exceeds the symbolic cap {MAX_COLORED_MDD_LEVELS}",
            build.num_levels()
        )));
    }

    let initial_marking = build.initial_marking().to_vec();
    let bounds = build.sound_per_slot_bounds();
    let drivers: Vec<_> = build
        .binding_drivers()?
        .into_iter()
        .map(|d| d.with_deadline(deadline))
        .collect();
    let n = build.level_count();
    debug_assert_eq!(bounds.len(), n);
    debug_assert_eq!(initial_marking.len(), n);

    // --- Build the reachable set by a breadth-first quantified relprod
    // fixpoint. ---
    let mut store = MddStore::new(bounds.clone());
    // Range-check the initial marking (singleton returns ZERO if out of range).
    for (l, (&m0, &b)) in initial_marking.iter().zip(&bounds).enumerate() {
        if m0 > b {
            return Err(SymbolicColoredError::Mdd(CountError::Malformed(format!(
                "initial marking[{l}] = {m0} exceeds sound bound {b}"
            ))));
        }
    }
    let mut reach = store.singleton(&initial_marking);

    // EXACT arbitrary-precision convergence count: the binding-quantified
    // colored reachable set can exceed `u128` (the colored families are large),
    // and the bignum carrier lets the fixpoint converge and report it instead of
    // declining on the cap. `|R|` is strictly monotone and exact, so equality is
    // a sound fixpoint witness at any magnitude.
    let mut prev_count = store.count_markings_big(reach);
    let mut rounds: u32 = 0;

    loop {
        rounds += 1;
        if rounds > MAX_ROUNDS {
            return Err(SymbolicColoredError::Mdd(CountError::ResourceCap(
                "round backstop exceeded (quantified)".to_string(),
            )));
        }
        check_deadline(deadline)?;

        let mut next = reach;
        for d in &drivers {
            let img = colored_transition_image_quantified(&mut store, reach, d)?;
            next = store.union(next, img);
            if store.interior_node_count() > MAX_INTERIOR_NODES {
                return Err(SymbolicColoredError::Mdd(CountError::ResourceCap(format!(
                    "interior node cap {MAX_INTERIOR_NODES} exceeded (quantified relprod)"
                ))));
            }
        }
        reach = next;
        let new_count = store.count_markings_big(reach);
        if new_count == prev_count {
            break; // fixpoint
        }
        prev_count = new_count;
    }

    // --- Extract the four StateSpace metrics off the reachable set. ---
    // The bignum count is authoritative; narrow fail-closed to the back-compat
    // `u64`/`u128` fields (saturated marker when it does not fit).
    let state_count_big = prev_count;
    let state_count = state_count_big.to_u64();
    let state_count_u128 = state_count_big.to_u128().unwrap_or(u128::MAX);
    let edge_count_big = quantified_edge_count(&mut store, reach, &bounds, &drivers, deadline)?;
    let edge_count = edge_count_big.to_u128().unwrap_or(u128::MAX);
    let max_token_in_place = max_token_in_place_of(&store, reach);
    let max_token_sum = max_token_sum_of(&store, reach);

    Ok(MddStateSpaceMetrics {
        state_count,
        state_count_u128,
        state_count_big,
        edge_count,
        edge_count_big,
        max_token_in_place,
        max_token_sum,
        iterations: rounds,
    })
}

/// `edge_count` = Σ over reachable markings of the number of enabled (in-bounds)
/// firings, computed BINDING-QUANTIFIED: for each colored transition, Σ over its
/// guard-satisfying bindings `b` of `|R ∩ Fireable(b)|`. Each binding is one
/// distinct event, and `R ∩ Fireable(b)` is exactly the reachable markings that
/// fire THAT binding — so the per-binding counts ADD (no double-count), matching
/// the BFS observer / `tla_dd::edge_count` semantics on the unfolded net.
///
/// Computed by re-running the binding recursion with a SUMMING accumulator
/// instead of a unioning one: a tiny edge-driver wrapper reuses the SAME
/// driver's domains / prune / leaf, so it visits exactly the same guard-feasible
/// bindings as the image and stays cap-free.
fn quantified_edge_count(
    store: &mut MddStore,
    reach: MddRef,
    bounds: &[u64],
    drivers: &[BindingQuantDriver<'_>],
    deadline: Option<Instant>,
) -> Result<BigUint, SymbolicColoredError> {
    // EXACT bignum accumulation: the edge sum no longer declines on magnitude.
    let mut total = BigUint::zero();
    for d in drivers {
        check_deadline(deadline)?;
        let mut prefix = Vec::new();
        total += edge_sum_recur(store, reach, bounds, d, 0, &mut prefix)?;
    }
    Ok(total)
}

/// Recurse the binding variables (SAME branching + guard prune as the image),
/// summing `|R ∩ Fireable(b)|` over fired leaf bindings. Counts add (distinct
/// events), so this is a SUM, not a union.
fn edge_sum_recur(
    store: &mut MddStore,
    reach: MddRef,
    bounds: &[u64],
    driver: &BindingQuantDriver<'_>,
    var_idx: usize,
    prefix: &mut Vec<usize>,
) -> Result<BigUint, SymbolicColoredError> {
    use tla_mdd::BindingDriver;
    // Same guard characteristic prune as the image: a prefix no completion can
    // satisfy contributes ZERO edges.
    if !driver.prefix_feasible(prefix)? {
        return Ok(BigUint::zero());
    }
    if var_idx == driver.num_vars() {
        // Leaf: resolve the (exact-guard) effect; count R ∩ Fireable(effect),
        // EXACT as bignum (no decline on magnitude).
        let Some(t) = driver.resolve_binding(prefix)? else {
            return Ok(BigUint::zero()); // guard rejects this binding
        };
        let f = fireable_set(store, bounds, &t);
        let inter = store.intersect(reach, f);
        return Ok(store.count_markings_big(inter));
    }
    let dom = driver.var_domain(var_idx);
    let mut acc = BigUint::zero();
    for v in 0..dom {
        prefix.push(v);
        let sub = edge_sum_recur(store, reach, bounds, driver, var_idx + 1, prefix);
        prefix.pop();
        acc += sub?;
    }
    Ok(acc)
}

/// Wall-clock deadline check (fail-closed). `None` never trips.
#[inline]
fn check_deadline(deadline: Option<Instant>) -> Result<(), SymbolicColoredError> {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(SymbolicColoredError::Mdd(CountError::ResourceCap(
                "deadline exceeded (quantified)".to_string(),
            )));
        }
    }
    Ok(())
}

fn contains_subtract(expr: &ColorExpr) -> bool {
    match expr {
        ColorExpr::Subtract { .. } => true,
        ColorExpr::Add(children) => children.iter().any(contains_subtract),
        ColorExpr::All { .. } | ColorExpr::NumberOf { .. } => false,
    }
}
