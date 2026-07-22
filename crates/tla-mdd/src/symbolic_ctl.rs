// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Symbolic (MDD) CTL evaluator over a bounded Petri net.
//!
//! This is the MDD counterpart of [`crate::symbolic`]'s reachability engine and
//! the set-for-set mirror of `tla_dd::symbolic_ctl` (the BDD evaluator) — which
//! is in turn the symbolic counterpart of the explicit-state
//! `tla_mc_core::CtlEngine`. It evaluates a CTL formula whose atoms are
//! **pre-built characteristic [`MddRef`]s** against the reachable marking set of
//! a [`MddNet`], entirely in MDD land — never enumerating markings — and answers
//! whether the formula holds *at the initial marking*.
//!
//! # Why atoms are pre-built MddRefs
//!
//! `tla-mdd` has NO non-dev dependencies (no predicate AST, no `tla-petri`
//! types). So the evaluator does not lower predicates itself: the caller
//! (`tla-petri`) lowers each atomic predicate to its characteristic MDD (the
//! set of markings satisfying it, confined to the net's bounds) and hands it in
//! as an [`MddRef`]. The local [`MddCtlFormula`] is therefore parameterized by
//! `atom = MddRef`.
//!
//! # The convention it matches (non-totalized; A-family = duals of E-family)
//!
//! Reproduced set-for-set from `tla_mc_core::CtlEngine::eval` (the MCC-2025
//! consensus engine), exactly as the BDD evaluator does. The transition
//! relation is **non-totalized**: a deadlock marking (no enabled, in-bounds
//! transition) genuinely has no successor. With that relation:
//!
//! - `EX φ  = pre_e(φ)`              — exists a successor in `φ`; false at a deadlock.
//! - `AX φ  = ¬ EX ¬φ`              — all successors in `φ`; vacuously true at a deadlock.
//! - `EF φ  = μZ. φ ∨ EX Z`.
//! - `EG φ  = νZ. φ ∧ (deadlock ∨ EX Z)`  — the **maximal-path** deadlock term, below.
//! - `AF φ  = ¬ EG ¬φ`.
//! - `AG φ  = ¬ EF ¬φ`.
//! - `E[φ U ψ] = μZ. ψ ∨ (φ ∧ EX Z)`.
//! - `A[φ U ψ] = ¬( E[¬ψ U (¬φ ∧ ¬ψ)] ∨ EG ¬ψ )`.
//!
//! The A-family is computed **strictly** as the De Morgan duals — NEVER as an
//! independent maximal-path fixpoint. (A previous attempt that gave AF/AU their
//! own finite-maximal-path fixpoint while EX/EG used `pre_e` mixed two
//! incompatible deadlock semantics, broke `AF == ¬EG¬`, and disagreed with
//! `CtlEngine`. This module avoids that by computing the A-family only via the
//! duals.)
//!
//! # The load-bearing `deadlock ∪` term in EG
//!
//! `CtlEngine::gfp_eg` assigns a deadlock `succ_in_set = u32::MAX` and **never
//! removes it**, i.e. it uses maximal-path semantics: a deadlock at which `φ`
//! holds *stays* in `EG φ` (its `deadlock_semantics_match_mcc_rules` test
//! asserts `EG(atom)=true` at a deadlock where the atom holds). To reproduce the
//! engine exactly, the gfp keeps a state if it is a **deadlock** OR has a
//! successor still in the set: `Z = Z ∩ (deadlock ∪ pre_e(Z))`. AF/AU, as duals
//! of this gfp, inherit the same semantics automatically.
//!
//! # The answer at the initial marking, confined to the reachable set
//!
//! `pre_e` ranges over the reachable set only (every `pre_e` result is
//! intersected with `reach`), so the fixpoints run on exactly `CtlEngine`'s
//! graph (reachable markings, reachable edges, deadlocks maximal-path). The
//! satisfying set therefore equals `CtlEngine`'s on *all* reachable markings,
//! and `holds_at_initial` tests membership of the initial marking. The
//! differential battery checks the verdict against `CtlEngine` on every
//! reachable marking of every net (not just the initial one), so any divergence
//! — including one through an unreachable intermediate — is caught.
//!
//! # Fail-closed contract
//!
//! Every fixpoint loop polls a wall-clock deadline + a live-node budget; on
//! breach it returns a fail-closed [`CtlError`] (the production gate maps that
//! to a DECLINE, never a partial verdict). Convergence is `MddRef` equality
//! (canonical, so equal sets share the same ref).

use crate::image::transition_preimage;
use crate::metrics::build_fireable_set;
use crate::node::{MddRef, MddStore};
use crate::reach::{CountError, MddNet};
use std::time::Instant;

/// Hard ceiling on live interior MDD nodes for the CTL fixpoints. Mirrors the
/// reachability engine's posture: a fixpoint that would blow past this DECLINES
/// (fail-closed) rather than risk an OOM. Generous for the structured nets the
/// lane targets; a decline never changes a verdict, it only forgoes the lane.
/// Interior-node cap, DERIVED from effective memory (was a fixed 8_000_000).
/// Adaptive to the machine/confinement via the shared node-store budget.
#[inline]
fn max_interior_nodes() -> usize {
    crate::node::max_interior_nodes()
}

/// Iteration backstop for the CTL fixpoints. Each is monotone over the finite
/// reachable MDD and converges; this guards against a logic bug, not a semantic
/// limit.
const MAX_FIXPOINT_ITERS: u32 = 100_000_000;

/// CTL formula whose atoms are pre-built characteristic [`MddRef`]s.
///
/// Mirrors `tla_mc_core::CtlFormula` / `tla_dd::symbolic_ctl::CtlFormula`
/// one-for-one (same operator set, same non-totalized A-family-as-duals
/// semantics). The atom payload is a marking-set MDD the *caller* builds (e.g.
/// from a Petri-net state predicate), confined to the net's bounds.
#[derive(Debug, Clone)]
pub enum MddCtlFormula {
    /// Atomic state predicate, as its characteristic marking-set MDD.
    Atom(MddRef),
    /// Boolean negation.
    Not(Box<MddCtlFormula>),
    /// Boolean conjunction (empty ⇒ true).
    And(Vec<MddCtlFormula>),
    /// Boolean disjunction (empty ⇒ false).
    Or(Vec<MddCtlFormula>),
    /// `EX φ`: some successor satisfies `φ` (false at a deadlock).
    EX(Box<MddCtlFormula>),
    /// `AX φ`: all successors satisfy `φ` (vacuously true at a deadlock).
    AX(Box<MddCtlFormula>),
    /// `EF φ`: some path eventually reaches `φ`.
    EF(Box<MddCtlFormula>),
    /// `AF φ`: all paths eventually reach `φ` (`= ¬ EG ¬φ`).
    AF(Box<MddCtlFormula>),
    /// `EG φ`: some maximal path stays in `φ` (a deadlock at which `φ` holds
    /// stays — maximal-path semantics, matching `CtlEngine`).
    EG(Box<MddCtlFormula>),
    /// `AG φ`: all maximal paths stay in `φ` (`= ¬ EF ¬φ`).
    AG(Box<MddCtlFormula>),
    /// `E[φ U ψ]`: some path has `φ` until `ψ`.
    EU(Box<MddCtlFormula>, Box<MddCtlFormula>),
    /// `A[φ U ψ]`: all paths have `φ` until `ψ` (the dual identity).
    AU(Box<MddCtlFormula>, Box<MddCtlFormula>),
    /// `E(GF φ)`: some path visits `φ` infinitely often — the Emerson–Lei
    /// fair-cycle `νZ. EF(φ ∧ EXˢZ)` with the DEADLOCK-STUTTER successor
    /// (`EXˢZ = EX Z ∨ (deadlock ∧ Z)`): a deadlocked `φ`-state self-stutters
    /// and is a fair witness — verdict-identical to `CtlEngine::gfp_egf` (the
    /// pinned oracle) and the GPU `CtlOp::EGF` engine.
    EGF(Box<MddCtlFormula>),
}

/// Why the MDD CTL lane declined (fail-closed; never a guessed verdict).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CtlError {
    /// The underlying reachability build declined (overflow / node budget /
    /// deadline / malformed net).
    Reach(CountError),
    /// A CTL fixpoint hit the live-node ceiling.
    NodeBudget,
    /// A CTL fixpoint passed its wall-clock deadline.
    Deadline,
    /// A fixpoint exceeded the iteration backstop (logic-bug guard).
    IterationBackstop,
}

impl From<CountError> for CtlError {
    fn from(e: CountError) -> Self {
        CtlError::Reach(e)
    }
}

/// Build the reachable set + deadlock set for `net`, lower the formula's atoms
/// via `lower_atom` (which receives the live store and the net so it can build
/// each predicate's characteristic MDD in-store), then evaluate `formula` at the
/// initial marking.
///
/// `lower_atom` is the seam that keeps `tla-mdd` predicate-free: the caller
/// supplies a closure turning its own atom representation into an [`MddRef`] in
/// the evaluator's store. The closure is invoked once per `Atom` node, in the
/// order the formula is walked.
///
/// Fail-closed: any reachability decline, atom-lowering failure (`None`), or
/// fixpoint over-budget yields `Err`.
pub fn evaluate_at_initial<A, F>(
    net: &MddNet,
    formula: &CtlFormulaTemplate<A>,
    deadline: Option<Instant>,
    mut lower_atom: F,
) -> Result<bool, CtlError>
where
    F: FnMut(&mut MddStore, &MddNet, &A) -> Option<MddRef>,
{
    net.validate()?;
    // Build the reachable set via the saturation engine (the scalable pillar;
    // its set is pinned EQUAL to relprod + BFS by the crate's differential
    // battery). This is the SAME reachable set the StateSpace-MDD metric path
    // consumes, so the CTL fixpoints run on exactly the cross-checked graph.
    let (mut store, reach, _rounds) = net.build_reachable_saturation(deadline)?;

    // Lower the formula's atoms into the live store. A `None` from the caller
    // (an unsupported predicate shape) is a fail-closed decline.
    let lowered = lower_formula(formula, &mut store, net, &mut lower_atom).ok_or(
        CtlError::Reach(CountError::Malformed("atom lowering declined".to_string())),
    )?;

    let initial = store.singleton(&net.initial_marking);
    let mut ev = CtlEvaluator::new(&mut store, net, reach, initial, deadline);
    ev.holds_at_initial(&lowered)
}

/// Decide LTL/Büchi emptiness for `net`: does the reachable graph contain a
/// cycle through a marking satisfying `accepting`? Builds the (saturation)
/// reachable set, lowers the accepting atom-set via `lower_atom`, and runs the
/// Emerson-Lei fair-cycle gfp. `Ok(true)` ⟺ a reachable accepting cycle exists
/// (⟺ the Büchi product language is non-empty ⟺ the LTL property is violated).
///
/// This is the native-MDD LTL core: feeding `accepting` from an LTL
/// formula→Büchi translation closes `tla-mdd`'s LTL gap. Fail-closed on any
/// reachability/atom decline (NEVER a guessed verdict).
pub fn evaluate_buchi_emptiness_at_initial<A, F>(
    net: &MddNet,
    accepting: &A,
    deadline: Option<Instant>,
    mut lower_atom: F,
) -> Result<bool, CtlError>
where
    F: FnMut(&mut MddStore, &MddNet, &A) -> Option<MddRef>,
{
    net.validate()?;
    let (mut store, reach, _rounds) = net.build_reachable_saturation(deadline)?;
    let acc = lower_atom(&mut store, net, accepting).ok_or(CtlError::Reach(
        CountError::Malformed("accepting atom lowering declined".to_string()),
    ))?;
    let initial = store.singleton(&net.initial_marking);
    let mut ev = CtlEvaluator::new(&mut store, net, reach, initial, deadline);
    ev.fair_cycle_exists(acc, None)
}

/// Is there a reachable cycle ENTIRELY WITHIN the set `within` (i.e. an infinite
/// run that, after a finite prefix, stays in `within` forever)? This is the
/// `GF φ` LTL pattern: `GF φ` is violated ⟺ some run is eventually-always `¬φ` ⟺
/// a reachable cycle lies entirely in `¬φ` — call this with `within = ¬φ`. Built
/// on the same Emerson-Lei fair-cycle gfp restricted to the `within` subgraph.
/// Fail-closed on any decline.
pub fn evaluate_recurrent_cycle_within<A, F>(
    net: &MddNet,
    within: &A,
    deadline: Option<Instant>,
    mut lower_atom: F,
) -> Result<bool, CtlError>
where
    F: FnMut(&mut MddStore, &MddNet, &A) -> Option<MddRef>,
{
    net.validate()?;
    let (mut store, reach, _rounds) = net.build_reachable_saturation(deadline)?;
    let dom = lower_atom(&mut store, net, within).ok_or(CtlError::Reach(CountError::Malformed(
        "within atom lowering declined".to_string(),
    )))?;
    let initial = store.singleton(&net.initial_marking);
    let mut ev = CtlEvaluator::new(&mut store, net, reach, initial, deadline);
    // Accepting = the whole domain ⇒ any cycle inside `within` qualifies.
    ev.fair_cycle_exists(dom, Some(dom))
}

/// Path quantifier for a flat reachability query (`ReachabilityCardinality` /
/// `ReachabilityFireability`): the EF/AG fragment that asks only about the
/// reachable-marking SET, never a nested temporal fixpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MddReachQuantifier {
    /// `EF φ`: some reachable marking satisfies `φ`.
    Ef,
    /// `AG φ`: every reachable marking satisfies `φ`.
    Ag,
}

/// Build the reachable set of `net` ONCE (via the saturation engine, the same
/// cross-checked set [`evaluate_at_initial`] uses) and answer a batch of flat
/// EF/AG reachability queries against it EXACTLY.
///
/// Each query is a `(quantifier, atom)` pair; `lower_atom` lowers the caller's
/// atom representation into the live store as its characteristic marking-set
/// MDD (the same closure seam [`evaluate_at_initial`] uses, keeping `tla-mdd`
/// predicate-free). With the EXACT reachable set `R` and the EXACT
/// characteristic set `charset(φ)`:
///
/// - `EF φ` ⟺ `R ∩ charset(φ) ≠ ∅`  (some reachable marking satisfies `φ`).
/// - `AG φ` ⟺ `R ∩ charset(¬φ) = ∅`  (no reachable marking violates `φ`),
///   where `charset(¬φ) = R \ charset(φ)` (the reachable-confined complement,
///   matching `CtlEngine`'s reachable-only satisfying sets).
///
/// Both directions are exact because both `R` and `charset(φ)` are exact — this
/// decides BOTH the TRUE and the FALSE verdict, strictly stronger than an
/// over-approximate one-sided closer. (Equivalently: `EF φ = R ∩ charset(φ)`
/// reached as the lfp `μZ. φ ∨ EX Z` collapses on the *whole* reachable set
/// since every reachable marking is already in `R`; the membership form is the
/// same set, computed without a fixpoint.)
///
/// Returns one verdict per query in input order, or a fail-closed [`CtlError`]
/// on any reachability decline (overflow / node budget / deadline / malformed
/// net) or atom-lowering failure. NEVER a partial or guessed verdict.
pub fn evaluate_reachability_at_initial<A, F>(
    net: &MddNet,
    queries: &[(MddReachQuantifier, A)],
    deadline: Option<Instant>,
    mut lower_atom: F,
) -> Result<Vec<bool>, CtlError>
where
    F: FnMut(&mut MddStore, &MddNet, &A) -> Option<MddRef>,
{
    net.validate()?;
    // Build the EXACT reachable set ONCE. Fail-closed: any Err here (overflow /
    // node cap / deadline / malformed) propagates and the caller declines — it
    // is never a partial/truncated set.
    let (mut store, reach, _rounds) = net.build_reachable_saturation(deadline)?;

    let mut verdicts = Vec::with_capacity(queries.len());
    for (quantifier, atom) in queries {
        // GC safepoint (non-moving mark-sweep). Between queries `reach` is the
        // ONLY live root — each query's `charset` and its intermediate sets are
        // dead once its verdict is recorded — so collecting keeps a large batch's
        // transient characteristic-set nodes from accumulating across the batch.
        // (The recursive CTL fixpoints in `holds_at_initial` are intentionally
        // NOT GC'd: they hold many nested intermediate result sets live across
        // the recursion, a high root-surface where under-supply would be
        // unsound, and they run on the already-built — already-GC'd — reachable
        // set, so their store growth is bounded.)
        if store.should_collect() {
            store.gc(&[reach]);
        }
        // Lower the atom to its EXACT characteristic marking-set MDD. A `None`
        // (unsupported atom shape / out-of-range index) is a fail-closed
        // decline of the WHOLE batch (we never publish a verdict for some
        // queries and silently drop others).
        let charset = lower_atom(&mut store, net, atom).ok_or(CtlError::Reach(
            CountError::Malformed("atom lowering declined".to_string()),
        ))?;
        let verdict = match quantifier {
            // EF φ: R ∩ charset(φ) ≠ ∅.
            MddReachQuantifier::Ef => {
                let hit = store.intersect(reach, charset);
                !hit.is_zero()
            }
            // AG φ: no reachable marking violates φ ⇔ R ∩ (R \ charset(φ)) = ∅
            // ⇔ R \ charset(φ) = ∅ (since R \ charset ⊆ R already). Compute the
            // reachable-confined complement and test emptiness.
            MddReachQuantifier::Ag => {
                let violators = store.difference(reach, charset);
                violators.is_zero()
            }
        };
        verdicts.push(verdict);
        // Poll the deadline between queries so a many-query batch stays
        // responsive (fail-closed → decline, never overrun-and-return-stale).
        if let Some(d) = deadline {
            if Instant::now() >= d {
                return Err(CtlError::Deadline);
            }
        }
    }
    Ok(verdicts)
}

/// A CTL formula parameterized by an arbitrary caller atom type `A`, lowered to
/// [`MddCtlFormula`] (atom = [`MddRef`]) by [`evaluate_at_initial`].
///
/// Mirrors [`MddCtlFormula`] exactly; the only difference is the atom payload is
/// the caller's representation `A` rather than a pre-built [`MddRef`]. This is
/// what lets `tla-petri` pass a formula whose atoms are its own predicate type
/// without `tla-mdd` depending on it.
#[derive(Debug, Clone)]
pub enum CtlFormulaTemplate<A> {
    /// Atomic state predicate in the caller's representation.
    Atom(A),
    /// Boolean negation.
    Not(Box<CtlFormulaTemplate<A>>),
    /// Boolean conjunction (empty ⇒ true).
    And(Vec<CtlFormulaTemplate<A>>),
    /// Boolean disjunction (empty ⇒ false).
    Or(Vec<CtlFormulaTemplate<A>>),
    /// `EX φ`.
    EX(Box<CtlFormulaTemplate<A>>),
    /// `AX φ`.
    AX(Box<CtlFormulaTemplate<A>>),
    /// `EF φ`.
    EF(Box<CtlFormulaTemplate<A>>),
    /// `AF φ`.
    AF(Box<CtlFormulaTemplate<A>>),
    /// `EG φ`.
    EG(Box<CtlFormulaTemplate<A>>),
    /// `AG φ`.
    AG(Box<CtlFormulaTemplate<A>>),
    /// `E[φ U ψ]`.
    EU(Box<CtlFormulaTemplate<A>>, Box<CtlFormulaTemplate<A>>),
    /// `A[φ U ψ]`.
    AU(Box<CtlFormulaTemplate<A>>, Box<CtlFormulaTemplate<A>>),
    /// `E(GF φ)` — the fair-cycle recurrence (deadlock-stutter convention).
    EGF(Box<CtlFormulaTemplate<A>>),
}

/// Recursively lower a [`CtlFormulaTemplate<A>`] to an [`MddCtlFormula`] by
/// invoking `lower_atom` on each atom. `None` (unsupported atom) short-circuits
/// the whole lowering to `None` (fail-closed at the caller).
fn lower_formula<A, F>(
    formula: &CtlFormulaTemplate<A>,
    store: &mut MddStore,
    net: &MddNet,
    lower_atom: &mut F,
) -> Option<MddCtlFormula>
where
    F: FnMut(&mut MddStore, &MddNet, &A) -> Option<MddRef>,
{
    use CtlFormulaTemplate as T;
    Some(match formula {
        T::Atom(a) => MddCtlFormula::Atom(lower_atom(store, net, a)?),
        T::Not(f) => MddCtlFormula::Not(Box::new(lower_formula(f, store, net, lower_atom)?)),
        T::And(cs) => {
            let mut out = Vec::with_capacity(cs.len());
            for c in cs {
                out.push(lower_formula(c, store, net, lower_atom)?);
            }
            MddCtlFormula::And(out)
        }
        T::Or(cs) => {
            let mut out = Vec::with_capacity(cs.len());
            for c in cs {
                out.push(lower_formula(c, store, net, lower_atom)?);
            }
            MddCtlFormula::Or(out)
        }
        T::EX(f) => MddCtlFormula::EX(Box::new(lower_formula(f, store, net, lower_atom)?)),
        T::AX(f) => MddCtlFormula::AX(Box::new(lower_formula(f, store, net, lower_atom)?)),
        T::EF(f) => MddCtlFormula::EF(Box::new(lower_formula(f, store, net, lower_atom)?)),
        T::AF(f) => MddCtlFormula::AF(Box::new(lower_formula(f, store, net, lower_atom)?)),
        T::EG(f) => MddCtlFormula::EG(Box::new(lower_formula(f, store, net, lower_atom)?)),
        T::AG(f) => MddCtlFormula::AG(Box::new(lower_formula(f, store, net, lower_atom)?)),
        T::EU(p, q) => MddCtlFormula::EU(
            Box::new(lower_formula(p, store, net, lower_atom)?),
            Box::new(lower_formula(q, store, net, lower_atom)?),
        ),
        T::AU(p, q) => MddCtlFormula::AU(
            Box::new(lower_formula(p, store, net, lower_atom)?),
            Box::new(lower_formula(q, store, net, lower_atom)?),
        ),
        T::EGF(f) => MddCtlFormula::EGF(Box::new(lower_formula(f, store, net, lower_atom)?)),
    })
}

/// CTL evaluator bound to one store + net + reachable set.
///
/// Holds the reachable-set root, the deadlock set (reachable markings with no
/// in-bounds successor), the initial-marking singleton, the net's transitions
/// (for `pre_e`), and the fixpoint deadline.
struct CtlEvaluator<'a> {
    store: &'a mut MddStore,
    net: &'a MddNet,
    /// Reachable marking set.
    reach: MddRef,
    /// Reachable deadlocks: reachable markings with no in-bounds successor under
    /// the (non-totalized) relation. Used by the gfp to give deadlocks
    /// maximal-path semantics, matching `CtlEngine::gfp_eg`.
    deadlock: MddRef,
    /// Initial-marking singleton (always reachable).
    initial: MddRef,
    deadline: Option<Instant>,
}

impl<'a> CtlEvaluator<'a> {
    fn new(
        store: &'a mut MddStore,
        net: &'a MddNet,
        reach: MddRef,
        initial: MddRef,
        deadline: Option<Instant>,
    ) -> Self {
        // Reachable deadlock set = reach \ (⋃_t Fireable(t)), using the SAME
        // bound-truncated fireability that defines the transition relation (a
        // marking is a deadlock iff no transition's guard holds with an
        // in-bounds successor). `build_fireable_set` is exactly the per-place
        // `pre[l] <= v ∧ v - pre[l] + post[l] <= bound[l]` product set the
        // metric edge_count uses, so the deadlock set is consistent with
        // `transition_preimage`'s guard/bound.
        let mut has_succ = MddRef::ZERO;
        for t in &net.transitions {
            let fireable = build_fireable_set(store, &net.bounds, t);
            has_succ = store.union(has_succ, fireable);
        }
        let deadlock = store.difference(reach, has_succ);
        Self {
            store,
            net,
            reach,
            deadlock,
            initial,
            deadline,
        }
    }

    /// Fail-closed guard polled inside every fixpoint loop. Returns `Err` once
    /// the wall-clock deadline elapses, the live interior-node count crosses
    /// [`max_interior_nodes`], or the store's approximate byte size crosses
    /// [`crate::node::max_store_bytes`] (audit 2026-07-02: a huge-bound place
    /// means tens of MB PER NODE, so the item cap alone admitted a multi-GB
    /// store — the byte arm the five sibling cap sites already have).
    /// Declining never changes a verdict — it forgoes the lane (the
    /// production gate falls through to the explicit checker).
    fn budget_guard(&self) -> Result<(), CtlError> {
        if let Some(d) = self.deadline {
            if Instant::now() >= d {
                return Err(CtlError::Deadline);
            }
        }
        if self.store.interior_node_count() > max_interior_nodes()
            || self.store.approx_store_bytes() > crate::node::max_store_bytes()
        {
            return Err(CtlError::NodeBudget);
        }
        Ok(())
    }

    /// Backward existential pre-image **confined to the reachable set**:
    ///
    /// `pre_e(S) = reach ∩ ⋃_t transition_preimage(S, t)`.
    ///
    /// `transition_preimage` only contains arcs whose successor is in-bounds and
    /// in `S`, and a deadlock has no such arc, so `pre_e` is empty at a deadlock
    /// — matching `CtlEngine::pre_e` (EX false at a deadlock) with no
    /// `has_successor` guard and no totalization. Intersecting with `reach`
    /// keeps the whole computation on `CtlEngine`'s graph.
    fn pre_e(&mut self, sat: MddRef) -> Result<MddRef, CtlError> {
        let mut acc = MddRef::ZERO;
        for t in &self.net.transitions {
            let pre = transition_preimage(self.store, sat, t);
            acc = self.store.union(acc, pre);
            self.budget_guard()?;
        }
        Ok(self.store.intersect(acc, self.reach))
    }

    /// Reachable-confined complement: `reach \ S`. All satisfying sets here are
    /// subsets of the reachable set, so the meaningful complement is relative to
    /// that universe — exactly what `CtlEngine` computes (its bitsets flip
    /// reachable states only).
    fn complement(&mut self, sat: MddRef) -> MddRef {
        self.store.difference(self.reach, sat)
    }

    /// Evaluate `formula` to its (reachable-confined) satisfying-set MDD.
    fn eval(&mut self, formula: &MddCtlFormula) -> Result<MddRef, CtlError> {
        match formula {
            MddCtlFormula::Atom(set) => {
                // Confine the atom to the reachable set (matches the BDD lane:
                // CtlEngine's satisfying sets are reachable-only).
                Ok(self.store.intersect(*set, self.reach))
            }
            MddCtlFormula::Not(inner) => {
                let s = self.eval(inner)?;
                Ok(self.complement(s))
            }
            MddCtlFormula::And(children) => {
                let mut acc = self.reach;
                for child in children {
                    let s = self.eval(child)?;
                    acc = self.store.intersect(acc, s);
                }
                Ok(acc)
            }
            MddCtlFormula::Or(children) => {
                let mut acc = MddRef::ZERO;
                for child in children {
                    let s = self.eval(child)?;
                    acc = self.store.union(acc, s);
                }
                // Each child ⊆ reach already; intersect defensively to keep the
                // invariant explicit (and to confine the empty-Or = ∅ case).
                Ok(self.store.intersect(acc, self.reach))
            }
            MddCtlFormula::EX(inner) => {
                let s = self.eval(inner)?;
                self.pre_e(s)
            }
            MddCtlFormula::AX(inner) => {
                // AX φ = ¬ EX ¬φ.
                let s = self.eval(inner)?;
                let not_s = self.complement(s);
                let ex_not = self.pre_e(not_s)?;
                Ok(self.complement(ex_not))
            }
            MddCtlFormula::EF(inner) => {
                let s = self.eval(inner)?;
                self.lfp_ef(s)
            }
            MddCtlFormula::AF(inner) => {
                // AF φ = ¬ EG ¬φ.
                let s = self.eval(inner)?;
                let not_s = self.complement(s);
                let eg_not = self.gfp_eg(not_s)?;
                Ok(self.complement(eg_not))
            }
            MddCtlFormula::EG(inner) => {
                let s = self.eval(inner)?;
                self.gfp_eg(s)
            }
            MddCtlFormula::EGF(inner) => {
                let s = self.eval(inner)?;
                self.gfp_egf(s)
            }
            MddCtlFormula::AG(inner) => {
                // AG φ = ¬ EF ¬φ.
                let s = self.eval(inner)?;
                let not_s = self.complement(s);
                let ef_not = self.lfp_ef(not_s)?;
                Ok(self.complement(ef_not))
            }
            MddCtlFormula::EU(phi, psi) => {
                let p = self.eval(phi)?;
                let q = self.eval(psi)?;
                self.lfp_eu(p, q)
            }
            MddCtlFormula::AU(phi, psi) => {
                // A[φ U ψ] = ¬( E[¬ψ U (¬φ ∧ ¬ψ)] ∨ EG ¬ψ ).
                let p = self.eval(phi)?;
                let q = self.eval(psi)?;
                let not_p = self.complement(p);
                let not_q = self.complement(q);
                let not_p_and_not_q = self.store.intersect(not_p, not_q);
                let eu = self.lfp_eu(not_q, not_p_and_not_q)?;
                let eg = self.gfp_eg(not_q)?;
                let union = self.store.union(eu, eg);
                Ok(self.complement(union))
            }
        }
    }

    /// `EF φ = μZ. φ ∨ EX Z`, confined to the reachable set. Monotone-increasing
    /// over the finite reachable MDD ⇒ converges.
    fn lfp_ef(&mut self, sat: MddRef) -> Result<MddRef, CtlError> {
        let mut z = self.store.intersect(sat, self.reach);
        let mut iters: u32 = 0;
        loop {
            self.budget_guard()?;
            iters += 1;
            if iters > MAX_FIXPOINT_ITERS {
                return Err(CtlError::IterationBackstop);
            }
            let pre = self.pre_e(z)?;
            let next = self.store.union(z, pre);
            if next == z {
                return Ok(z);
            }
            z = next;
        }
    }

    /// `EG φ = νZ. (φ ∩ reach) ∩ (deadlock ∪ EX Z)`, the maximal-path gfp that
    /// matches `CtlEngine::gfp_eg` exactly (a deadlock at which `φ` holds stays).
    /// Monotone-decreasing over the finite reachable MDD ⇒ converges.
    fn gfp_eg(&mut self, sat: MddRef) -> Result<MddRef, CtlError> {
        let mut z = self.store.intersect(sat, self.reach);
        let mut iters: u32 = 0;
        loop {
            self.budget_guard()?;
            iters += 1;
            if iters > MAX_FIXPOINT_ITERS {
                return Err(CtlError::IterationBackstop);
            }
            // Keep states in z that are deadlocks OR have a successor in z.
            let ex_z = self.pre_e(z)?;
            let dead_or_ex = self.store.union(self.deadlock, ex_z);
            let next = self.store.intersect(z, dead_or_ex);
            if next == z {
                return Ok(z);
            }
            z = next;
        }
    }

    /// Büchi/fair-cycle emptiness: is there a reachable cycle through a marking
    /// in `accepting`? The one-acceptance-set Emerson-Lei gfp
    /// `νZ. Z ∩ pre⁺_Z(Z ∩ F)` over the reachable set, where `pre⁺_Z` is the
    /// transitive backward image ([`pre_e`], which excludes deadlocks) restricted
    /// to `Z`. On convergence every state of `Z` can re-reach an accepting state
    /// of `Z` in ≥1 step within `Z` ⇒ an infinite accepting path (a fair cycle);
    /// so `Z ≠ ∅ ⟺ a reachable accepting cycle exists`. This is the symbolic core
    /// of LTL model checking (a reachable accepting cycle ⟺ the Büchi product
    /// language is non-empty ⟺ the property is violated) — the MDD-frontier twin
    /// of `tla_bdd::petri::fair_cycle_exists`.
    /// `domain` (default = the reachable set) bounds the search to a subgraph:
    /// with `domain = reach` it finds a cycle THROUGH `accepting` (FG φ violation
    /// with `accepting = ¬φ`); with `domain = accepting = S` it finds a cycle
    /// ENTIRELY WITHIN `S` (GF φ violation with `S = ¬φ`).
    /// `Sat(E GF φ)` — the Emerson–Lei fair-cycle gfp with the DEADLOCK-STUTTER
    /// successor, mirroring `CtlEngine::gfp_egf` LITERALLY (the pinned oracle):
    /// `νZ. EF(φ ∧ EXˢZ)` where `EXˢZ = EX Z ∪ (deadlock ∩ Z)` — a deadlocked
    /// `φ`-state self-stutters into a fair witness; `¬φ` never-recurring tails
    /// are excluded. The inner `EFˢ` coincides with plain `lfp_ef` (a deadlock
    /// contributes only its own base bit under either successor). Decreasing +
    /// finite ⇒ converges. (The BÜCHI lane's `fair_cycle_exists` keeps its own
    /// ≥1-real-step convention — deliberately NOT unified.)
    fn gfp_egf(&mut self, sat: MddRef) -> Result<MddRef, CtlError> {
        // Deadlocks within reach: reach ∖ pre_e(reach) (no successor at all).
        let has_succ = self.pre_e(self.reach)?;
        let deadlocks = {
            let not_succ = self.complement(has_succ);
            self.store.intersect(not_succ, self.reach)
        };
        let mut z = self.reach;
        let mut iters: u32 = 0;
        loop {
            self.budget_guard()?;
            iters += 1;
            if iters > MAX_FIXPOINT_ITERS {
                return Err(CtlError::IterationBackstop);
            }
            let ex_z = {
                let pre = self.pre_e(z)?;
                let stutter = self.store.intersect(deadlocks, z);
                self.store.union(pre, stutter)
            };
            let base = self.store.intersect(sat, ex_z);
            let next = self.lfp_ef(base)?;
            if next == z {
                return Ok(z);
            }
            z = next;
        }
    }

    pub(crate) fn fair_cycle_exists(
        &mut self,
        accepting: MddRef,
        domain: Option<MddRef>,
    ) -> Result<bool, CtlError> {
        let dom = self
            .store
            .intersect(domain.unwrap_or(self.reach), self.reach);
        let acc = self.store.intersect(accepting, dom);
        let mut z = dom;
        let mut iters: u32 = 0;
        loop {
            self.budget_guard()?;
            iters += 1;
            if iters > MAX_FIXPOINT_ITERS {
                return Err(CtlError::IterationBackstop);
            }
            let f_in_z = self.store.intersect(z, acc);
            let reach_back = self.pre_plus_in(z, f_in_z)?;
            let next = self.store.intersect(z, reach_back);
            if next == z {
                return Ok(z != MddRef::ZERO);
            }
            z = next;
        }
    }

    /// `pre⁺` within `z` reaching `seed`: `μW. (EX(seed ∪ W)) ∩ z` — states of
    /// `z` that reach `seed` in ≥1 real transition staying inside `z`.
    fn pre_plus_in(&mut self, z: MddRef, seed: MddRef) -> Result<MddRef, CtlError> {
        let mut w = MddRef::ZERO;
        let mut iters: u32 = 0;
        loop {
            self.budget_guard()?;
            iters += 1;
            if iters > MAX_FIXPOINT_ITERS {
                return Err(CtlError::IterationBackstop);
            }
            let target = self.store.union(seed, w);
            let pre = self.pre_e(target)?; // already ∩ reach
            let pre_z = self.store.intersect(pre, z);
            if pre_z == w {
                return Ok(w);
            }
            w = pre_z;
        }
    }

    /// `E[φ U ψ] = μZ. ψ ∨ (φ ∧ EX Z)`, confined to the reachable set.
    fn lfp_eu(&mut self, phi: MddRef, psi: MddRef) -> Result<MddRef, CtlError> {
        let phi_r = self.store.intersect(phi, self.reach);
        let mut z = self.store.intersect(psi, self.reach);
        let mut iters: u32 = 0;
        loop {
            self.budget_guard()?;
            iters += 1;
            if iters > MAX_FIXPOINT_ITERS {
                return Err(CtlError::IterationBackstop);
            }
            let ex_z = self.pre_e(z)?;
            let phi_ex = self.store.intersect(phi_r, ex_z);
            let next = self.store.union(z, phi_ex);
            if next == z {
                return Ok(z);
            }
            z = next;
        }
    }

    /// Does the **initial marking** satisfy `formula`?  Evaluates to the
    /// (reachable-confined) satisfying set and tests membership of the initial
    /// marking: `initial ∩ sat ≠ ∅`. Since the initial marking is a single full
    /// marking, this is exactly `CtlEngine.eval(formula)[initial]`.
    fn holds_at_initial(&mut self, formula: &MddCtlFormula) -> Result<bool, CtlError> {
        let sat = self.eval(formula)?;
        let conj = self.store.intersect(self.initial, sat);
        Ok(!conj.is_zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reach::MddTransition;

    fn t(pre: Vec<u64>, post: Vec<u64>) -> MddTransition {
        MddTransition { pre, post }
    }

    /// A tiny atom representation for the in-crate tests: a per-place lower
    /// bound `place >= c` (or its negation handled by the formula `Not`).
    /// Lowered to a characteristic MDD via `build_threshold_set`.
    #[derive(Debug, Clone)]
    enum TestAtom {
        /// `tokens(place) >= c`.
        Ge(usize, u64),
        /// constant true / false.
        Const(bool),
    }

    fn build_threshold_set(store: &mut MddStore, net: &MddNet, a: &TestAtom) -> Option<MddRef> {
        match a {
            TestAtom::Const(true) => Some(MddRef::ONE),
            TestAtom::Const(false) => Some(MddRef::ZERO),
            TestAtom::Ge(place, c) => {
                // Characteristic set of `m[place] >= c`: a chain that is ONE
                // everywhere except level `place`, where edges `< c` go to ZERO
                // and edges `>= c` go to ONE-below. Built bottom-up.
                let n = net.bounds.len();
                if *place >= n {
                    return None;
                }
                let mut acc = MddRef::ONE;
                for level in (0..n).rev() {
                    let dom = store.domain_size(level as u32);
                    let mut children = vec![MddRef::ZERO; dom];
                    for v in 0..dom as u64 {
                        let ok = if level == *place { v >= *c } else { true };
                        if ok {
                            children[v as usize] = acc;
                        }
                    }
                    acc = store.get_node(level as u32, children);
                }
                Some(acc)
            }
        }
    }

    fn eval(net: &MddNet, f: &CtlFormulaTemplate<TestAtom>) -> bool {
        evaluate_at_initial(net, f, None, build_threshold_set).expect("must not decline")
    }

    use CtlFormulaTemplate as F;
    fn ge(p: usize, c: u64) -> F<TestAtom> {
        F::Atom(TestAtom::Ge(p, c))
    }
    fn tt() -> F<TestAtom> {
        F::Atom(TestAtom::Const(true))
    }
    fn ff() -> F<TestAtom> {
        F::Atom(TestAtom::Const(false))
    }

    /// drain-to-deadlock: [1,0] -> [0,1], [0,1] is a deadlock.
    fn drain() -> MddNet {
        MddNet {
            bounds: vec![1, 1],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 1])],
        }
    }

    /// ping-pong: [1,0] <-> [0,1], live (no deadlock).
    fn ping_pong() -> MddNet {
        MddNet {
            bounds: vec![1, 1],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])],
        }
    }

    /// no transitions: the initial marking [1,1] is itself a deadlock.
    fn no_trans() -> MddNet {
        MddNet {
            bounds: vec![2, 2],
            initial_marking: vec![1, 1],
            transitions: vec![],
        }
    }

    #[test]
    fn ef_reaches_deadlock_target() {
        // EF(p1>=1): [1,0]->[0,1] reaches p1=1 ⇒ true.
        assert!(eval(&drain(), &F::EF(Box::new(ge(1, 1)))));
        // EF(p0>=2): never (bound 1) ⇒ false.
        assert!(!eval(&drain(), &F::EF(Box::new(ge(0, 2)))));
    }

    fn fair_cycle(net: &MddNet, acc: &TestAtom) -> bool {
        evaluate_buchi_emptiness_at_initial(net, acc, None, build_threshold_set)
            .expect("must not decline")
    }

    #[test]
    fn buchi_fair_cycle_emptiness() {
        // ping-pong is a live 2-cycle [1,0]<->[0,1]. An accepting cycle through
        // {p1>=1} exists ([0,1] is on the cycle) ⇒ true. Through {p0>=1} too.
        assert!(fair_cycle(&ping_pong(), &TestAtom::Ge(1, 1)));
        assert!(fair_cycle(&ping_pong(), &TestAtom::Ge(0, 1)));
        // Const(true) ⇒ every state accepting; the cycle exists ⇒ true.
        assert!(fair_cycle(&ping_pong(), &TestAtom::Const(true)));

        // drain-to-deadlock [1,0]->[0,1] (deadlock) is ACYCLIC ⇒ NO fair cycle
        // through any accepting set (a deadlock is not on a cycle).
        assert!(!fair_cycle(&drain(), &TestAtom::Const(true)));
        assert!(!fair_cycle(&drain(), &TestAtom::Ge(1, 1)));

        // no_trans: the initial marking is a lone deadlock ⇒ no cycle ⇒ false.
        assert!(!fair_cycle(&no_trans(), &TestAtom::Const(true)));

        // Accepting set disjoint from the cycle ⇒ false: in ping-pong, {p0>=2}
        // is unreachable (bound 1) so no accepting state on the cycle.
        assert!(!fair_cycle(&ping_pong(), &TestAtom::Ge(0, 2)));
    }

    #[test]
    fn eg_at_deadlock_equals_phi() {
        // no_trans initial is a deadlock. EG(True) is true (φ holds, stays).
        assert!(eval(&no_trans(), &F::EG(Box::new(tt()))));
        // EG(False) is false.
        assert!(!eval(&no_trans(), &F::EG(Box::new(ff()))));
        // EG(p0>=1): holds at [1,1] ⇒ true. EG(p0>=2): does not hold ⇒ false.
        assert!(eval(&no_trans(), &F::EG(Box::new(ge(0, 1)))));
        assert!(!eval(&no_trans(), &F::EG(Box::new(ge(0, 2)))));
    }

    #[test]
    fn eg_true_on_live_cycle() {
        assert!(eval(&ping_pong(), &F::EG(Box::new(tt()))));
    }

    #[test]
    fn eg_drops_when_successor_leaves_set() {
        // drain: EG(p0>=1) — [1,0] has p0=1 but its only successor [0,1] (a
        // deadlock) has p0=0, so EG fails at init ⇒ false.
        assert!(!eval(&drain(), &F::EG(Box::new(ge(0, 1)))));
    }

    #[test]
    fn ax_vacuous_ex_false_at_deadlock() {
        // At the no_trans deadlock: AX(anything) true, EX(anything) false.
        assert!(eval(&no_trans(), &F::AX(Box::new(ge(0, 1)))));
        assert!(eval(&no_trans(), &F::AX(Box::new(ff()))));
        assert!(!eval(&no_trans(), &F::EX(Box::new(ge(0, 1)))));
        assert!(!eval(&no_trans(), &F::EX(Box::new(tt()))));
    }

    #[test]
    fn af_reaches_target_on_only_path() {
        // drain: AF(p1>=1) — the only path [1,0]->[0,1] reaches p1=1 (and the
        // deadlock [0,1] satisfies it) ⇒ true.
        assert!(eval(&drain(), &F::AF(Box::new(ge(1, 1)))));
    }

    #[test]
    fn ag_ef_back_to_p0_on_cycle() {
        // ping-pong: AG(EF(p0>=1)) — from anywhere you can get back to p0 ⇒ true.
        assert!(eval(
            &ping_pong(),
            &F::AG(Box::new(F::EF(Box::new(ge(0, 1)))))
        ));
    }

    #[test]
    fn deadline_in_past_declines() {
        let net = ping_pong();
        let past = Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap();
        let r = evaluate_at_initial(
            &net,
            &F::EG(Box::new(tt())),
            Some(past),
            build_threshold_set,
        );
        assert!(r.is_err(), "a past deadline must decline (fail-closed)");
    }

    /// The flat EF/AG reachability batch must agree, query-for-query, with the
    /// proven CTL `evaluate_at_initial` EF/AG fixpoint on the same atoms — the
    /// two exact lanes must never disagree (membership form == fixpoint form).
    #[test]
    fn reachability_batch_matches_ctl_ef_ag() {
        for net in [drain(), ping_pong(), no_trans()] {
            let atoms = [
                TestAtom::Ge(0, 1),
                TestAtom::Ge(1, 1),
                TestAtom::Ge(0, 2),
                TestAtom::Const(true),
                TestAtom::Const(false),
            ];
            let mut queries: Vec<(MddReachQuantifier, TestAtom)> = Vec::new();
            for a in &atoms {
                queries.push((MddReachQuantifier::Ef, a.clone()));
                queries.push((MddReachQuantifier::Ag, a.clone()));
            }
            let batch = evaluate_reachability_at_initial(&net, &queries, None, build_threshold_set)
                .expect("batch must not decline");
            for (i, (q, a)) in queries.iter().enumerate() {
                let ctl = match q {
                    MddReachQuantifier::Ef => F::EF(Box::new(F::Atom(a.clone()))),
                    MddReachQuantifier::Ag => F::AG(Box::new(F::Atom(a.clone()))),
                };
                let expected = eval(&net, &ctl);
                assert_eq!(
                    batch[i], expected,
                    "batch {q:?}({a:?}) = {} but CTL fixpoint = {expected}",
                    batch[i],
                );
            }
        }
    }

    #[test]
    fn reachability_batch_stable_under_forced_gc() {
        // GC root-supply for the reachability-batch safepoint (step 5): forcing
        // gc(&[reach]) before every query — and inside the reachable-set build —
        // must not change any verdict, since only `reach` is live across
        // queries and each query's charset is dead once its verdict is recorded.
        let net = ping_pong();
        let atoms = [
            TestAtom::Ge(0, 1),
            TestAtom::Ge(1, 1),
            TestAtom::Const(true),
            TestAtom::Const(false),
        ];
        let mut queries: Vec<(MddReachQuantifier, TestAtom)> = Vec::new();
        for a in &atoms {
            queries.push((MddReachQuantifier::Ef, a.clone()));
            queries.push((MddReachQuantifier::Ag, a.clone()));
        }
        let baseline = evaluate_reachability_at_initial(&net, &queries, None, build_threshold_set)
            .expect("baseline batch ok");
        crate::node::set_gc_stress(true);
        let forced = evaluate_reachability_at_initial(&net, &queries, None, build_threshold_set)
            .expect("forced-gc batch ok");
        crate::node::set_gc_stress(false);
        assert_eq!(
            baseline, forced,
            "forced GC must not change any batch verdict"
        );
    }

    #[test]
    fn reachability_batch_deadline_in_past_declines() {
        let net = ping_pong();
        let past = Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap();
        let queries = [(MddReachQuantifier::Ef, TestAtom::Const(true))];
        let r = evaluate_reachability_at_initial(&net, &queries, Some(past), build_threshold_set);
        assert!(r.is_err(), "a past deadline must decline (fail-closed)");
    }
}
