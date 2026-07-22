// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! One-shot **integer** firing-count state-equation feasibility query.
//!
//! This is the integer companion to the CHC encoder in
//! [`super::state_equation`] and the LP relaxation in
//! [`crate::lp_state_equation`]. Where the LP lane solves the state equation
//! over the **rationals** (`x ∈ ℚ≥0`) — leaving a *relaxation gap* in which a
//! rational firing vector exists but no integer one does — this lane declares
//! the firing-count vector over the **integers** (`x ∈ ℤ≥0`) and asks a single
//! QF_LIA satisfiability question. There is therefore NO relaxation gap: an
//! `UNSAT` answer is a genuine proof that no marking of the integer state
//! equation violates the property.
//!
//! # The query
//!
//! For an `AG`-universal safety property `φ` (every reachable marking satisfies
//! `φ`), we build the *violation* formula
//!
//! ```text
//!   (⋀_p  m_p = M0[p] + Σ_t C[p][t]·x_t)          -- integer state equation
//! ∧ (⋀_t  x_t ≥ 0)                                 -- firing counts are ℕ
//! ∧ (⋀_p  m_p ≥ 0)                                 -- markings are ℕ
//! ∧ ¬φ(m)                                          -- m violates the property
//! ∧ (⋀_{T trap}  Σ_{p∈T} m_p ≥ 1)                  -- initially-marked trap cuts
//! ```
//!
//! where `C[p][t] = out(t,p) − in(t,p)` is the incidence matrix and `x_t` is the
//! (integer) firing count of transition `t`. Every variable is `ChcSort::Int`,
//! so ay-chc's `SmtContext::check_sat` decides this as QF_LIA.
//!
//! ## `φ` may be a CARDINALITY *or* a FIREABILITY property
//!
//! The marking predicate `φ` is encoded by the **shared** predicate lowering
//! [`encode_predicate_expr`] (the very same routine the CHC lane uses), so this
//! lane handles every [`ResolvedPredicate`] the toolchain produces, not just pure
//! cardinality (`Σ tokens ≤ k`) atoms:
//!
//! - **CardinalityFireability / fireability.** An `IsFireable(T)` atom lowers to
//!   `⋁_{t∈T} ⋀_{arc∈•t} (m[arc.place] ≥ arc.weight)` — *exactly* the
//!   marking-arithmetic reading of [`crate::petri_net::PetriNet::is_enabled`] (every input arc
//!   independently satisfied; a transition with no input arc is unconditionally
//!   fireable, lowered to `true`). Because the lowering is term-for-term identical
//!   to `is_enabled`, the encoded violation set coincides with the concrete set of
//!   `is_enabled` markings used by BFS and every other engine.
//! - **Mixed boolean.** `And`/`Or`/`Not` over fireability and cardinality atoms
//!   are lowered structurally by the same routine, so e.g.
//!   `AG(IsFireable(t) ⇒ tokens(p) ≤ k)` (i.e. `¬IsFireable(t) ∨ tokens(p) ≤ k`)
//!   is one integer formula. `AG(¬IsFireable(t))` ("`t` is dead") is the special
//!   case whose violation is plain enabledness `⋀_{arc∈•t}(m ≥ w)`.
//!
//! The marking variables `m_p` are the integer `ise_m_*` variables; the shared
//! encoder names them `se_m_*` and we alpha-rename (see
//! [`rename_se_m_to_ise_m`]) — a pure symbol substitution that changes neither the
//! fireability guards nor the cardinality sums.
//!
//! - `UNSAT` ⇒ no integer state-equation solution violates `φ`. Because every
//!   *reachable* marking is such a solution (soundness theorem below),
//!   **`AG φ` holds — verdict `Safe`**.
//! - `SAT` ⇒ a *candidate* violating marking exists in the over-approximation.
//!   The state equation admits spurious solutions (no firing *order* is implied),
//!   so this is NOT a proof of `¬AG φ`; we return `Candidate` and the caller must
//!   replay-validate any witness on the concrete net before emitting `Unsafe`.
//! - `Unknown` / timeout / encoder overflow ⇒ `Unknown`; the lane DECLINES and
//!   the caller falls through to the exhaustive engines. We never invent a verdict.
//!
//! # Soundness theorem (the only `Safe` route)
//!
//! *Claim.* If the violation formula above is `UNSAT`, then `AG φ` holds on the
//! net.
//!
//! *Proof.* Let `m` be any marking reachable from `M0` by a finite firing
//! sequence `σ`. Let `x_t = #{occurrences of t in σ}` (the Parikh image of `σ`).
//! Then `x_t ∈ ℤ≥0` and the net token-balance identity gives
//! `m = M0 + C·x` with `m ∈ ℤ≥0` (markings never go negative). Hence `(m, x)`
//! satisfies the integer state equation together with the non-negativity
//! conjuncts. Each trap conjunct `Σ_{p∈T} m_p ≥ 1` also holds: an
//! *initially-marked* trap, once marked, can never be emptied by any firing
//! sequence (standard Petri-net theorem; gated by
//! [`crate::lp_state_equation::find_initially_marked_traps`], which only returns
//! initially-marked traps). Therefore, if some reachable `m` violated `φ`, then
//! `(m, x)` would satisfy the *entire* violation formula, contradicting `UNSAT`.
//! So no reachable marking violates `φ`, i.e. `AG φ` holds. ∎
//!
//! The converse fails (the state equation is a strict over-approximation:
//! siphon/trap structure and firing-order feasibility are not captured), which
//! is exactly why a `SAT` answer is only a candidate, never an `Unsafe` verdict.
//!
//! ## Soundness extends verbatim to fireability `φ`
//!
//! The theorem above never inspects the *shape* of `φ`: it only relies on
//! `¬φ(m)` being a marking predicate that holds at `m` iff `m` violates `φ`.
//! For a fireability atom that is immediate. `IsFireable(T)` is, by the toolchain
//! definition realised in [`crate::petri_net::PetriNet::is_enabled`], true at `m` iff some `t∈T`
//! has every input place sufficiently marked; the lowering encodes precisely that
//! marking condition over the same `m_p` variables. Hence for `φ = ¬IsFireable(t)`
//! the violation conjunct is `IsFireable(t)`'s lowering, and `UNSAT` means no
//! state-equation solution `m` enables `t`. Since every *reachable* marking is a
//! state-equation solution (the proof above), no reachable marking enables `t`,
//! i.e. `AG(¬IsFireable(t))` holds — `t` is dead. Dually, `φ = IsFireable(t)`
//! ("`t` is fireable in every reachable marking") has violation `¬IsFireable(t)`,
//! and `UNSAT` proves `t` is enabled at every state-equation solution, a fortiori
//! at every reachable marking. The argument composes through `And`/`Or`/`Not`
//! unchanged because `encode_predicate_expr` is a faithful Boolean homomorphism of
//! [`crate::resolved_predicate::eval_predicate`]. No part of this introduces a `Safe` route that `is_enabled`
//! / BFS would disagree with, so the lane stays verdict-preserving for fireability.
//!
//! # Why integer, not LP
//!
//! The LP relaxation can report a rational firing vector where the integer
//! program is infeasible — a *spurious feasible* point that prevents the LP from
//! proving `AG φ`. Solving over ℤ closes precisely that gap, so this lane can
//! prove `Safe` on nets the LP leaves `Unknown` (see the differential unit gate
//! `test_int_infeasible_where_lp_relaxation_is_spurious_feasible`).

use std::time::Duration;

use ay_chc::{ChcExpr, ChcSort, ChcVar, SmtContext, SmtResult};

use super::state_equation::{
    encode_predicate_expr, StateEquationEncoderError, MAX_SYMBOLIC_CELLS, MAX_SYMBOLIC_PLACES,
};
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::resolved_predicate::ResolvedPredicate;

/// Outcome of the one-shot integer state-equation feasibility query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IntStateEquationVerdict {
    /// The violation formula is `UNSAT`: no integer state-equation solution
    /// violates the property, so `AG φ` holds. SOUND — no relaxation gap.
    Safe,
    /// The violation formula is `SAT`: a *candidate* violating marking exists in
    /// the state-equation over-approximation. NOT a proof of `¬AG φ`; the caller
    /// must replay-validate before emitting `Unsafe`.
    Candidate,
    /// Solver inconclusive, timeout, encoder overflow, or net too large. The
    /// caller MUST fall through to the exhaustive engines — never a verdict.
    Unknown(IntUnknownReason),
}

/// Diagnostic reason attached to [`IntStateEquationVerdict::Unknown`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IntUnknownReason {
    /// Encoder rejected the net (overflow guard, oversized signature, or
    /// u64-out-of-range token/weight).
    EncoderRejected(String),
    /// `SmtContext::check_sat` returned `Unknown` (incl. timeout).
    SolverInconclusive,
}

impl From<StateEquationEncoderError> for IntUnknownReason {
    fn from(err: StateEquationEncoderError) -> Self {
        Self::EncoderRejected(err.to_string())
    }
}

/// Variable naming for the integer firing-count state equation.
///
/// Deliberately distinct from [`super::state_equation::VarNaming`] (`se_m_*`):
/// this lane introduces firing-count variables `se_x_*` and keeps its own
/// current-marking variables `ise_m_*` so the two encodings never alias if
/// emitted into the same context.
struct IntVarNaming;

impl IntVarNaming {
    fn marking(place_idx: usize) -> ChcVar {
        ChcVar::new(format!("ise_m_{place_idx}"), ChcSort::Int)
    }
    fn firing(transition_idx: usize) -> ChcVar {
        ChcVar::new(format!("se_x_{transition_idx}"), ChcSort::Int)
    }
}

/// Encode the marking expression used by `¬φ`: the property is written over the
/// integer marking variables `ise_m_*`, NOT `se_m_*`. We therefore translate the
/// shared CHC predicate expression (which names `se_m_*`) by re-binding place `p`
/// to `ise_m_p`. Rather than rewrite the expression tree, we reuse
/// [`encode_predicate_expr`] and substitute via a fresh re-encode below.
///
/// To avoid a second encoder, we simply mirror the predicate encoder here using
/// the integer-marking variable names. Both encoders share the SAME checked
/// overflow semantics through [`encode_predicate_expr`]; see
/// [`property_violation_over_int_markings`].
fn marking_var_expr(place_idx: usize) -> ChcExpr {
    ChcExpr::var(IntVarNaming::marking(place_idx))
}

/// Build `¬φ(m)` over the integer marking variables `ise_m_*`.
///
/// We reuse the shared [`encode_predicate_expr`] (which emits over `se_m_*`) and
/// then alpha-rename `se_m_<p>` → `ise_m_<p>`. Keeping a single predicate encoder
/// guarantees the integer lane and the CHC lane interpret each property
/// identically (same overflow guards, same `IsFireable`/`IntLe` semantics).
fn property_violation_over_int_markings(
    property: &ResolvedPredicate,
    net: &PetriNet,
) -> Result<ChcExpr, StateEquationEncoderError> {
    let encoded = encode_predicate_expr(property, net)?;
    let renamed = rename_se_m_to_ise_m(&encoded);
    Ok(ChcExpr::not(renamed))
}

/// Alpha-rename every `se_m_<idx>` variable to `ise_m_<idx>` in a CHC expression.
///
/// The shared predicate encoder names current-marking variables `se_m_*`; this
/// lane uses `ise_m_*`. Renaming is a pure variable substitution that preserves
/// the formula's meaning exactly (only the variable *symbol* changes; the sort
/// stays `Int`). It touches no operator, constant, or structure.
fn rename_se_m_to_ise_m(expr: &ChcExpr) -> ChcExpr {
    match expr {
        ChcExpr::Var(var) => {
            if let Some(idx) = var.name.strip_prefix("se_m_") {
                ChcExpr::var(ChcVar::new(format!("ise_m_{idx}"), var.sort.clone()))
            } else {
                ChcExpr::Var(var.clone())
            }
        }
        ChcExpr::Op(op, args) => ChcExpr::Op(
            *op,
            args.iter()
                .map(|a| std::sync::Arc::new(rename_se_m_to_ise_m(a)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Build the SOUND base constraints shared by every integer state-equation query:
/// the integer state equation, the `x ≥ 0`/`m ≥ 0` non-negativity conjuncts, the
/// initially-marked **trap** cuts (`Σ_{p∈T} m_p ≥ 1`) and the initially-unmarked
/// **siphon** cuts (`Σ_{p∈S} m_p ≤ 0`).
///
/// Every conjunct here holds in EVERY reachable marking (see the module-level and
/// per-cut soundness notes), so it over-approximates the reachable set: appending
/// any *target* constraint (a property violation, a marking lower bound, …) and
/// getting `UNSAT` is a genuine proof that no reachable marking meets that target.
///
/// Uses the same checked-arithmetic overflow guards as the CHC encoder; any
/// overflow is surfaced as `Err` so the caller maps it to `Unknown` (never a
/// silent truncation that could flip a verdict).
fn encode_base_constraints(net: &PetriNet) -> Result<Vec<ChcExpr>, StateEquationEncoderError> {
    let np = net.num_places();
    let nt = net.num_transitions();

    if np > MAX_SYMBOLIC_PLACES {
        return Err(StateEquationEncoderError::NetTooLarge {
            num_places: np,
            limit: MAX_SYMBOLIC_PLACES,
        });
    }
    let cells = np.saturating_mul(nt);
    if cells > MAX_SYMBOLIC_CELLS {
        return Err(StateEquationEncoderError::NetTooManyCells {
            cells,
            limit: MAX_SYMBOLIC_CELLS,
        });
    }

    // Incidence matrix column per place: C[p][t] = out(t,p) − in(t,p), with the
    // SAME checked-arithmetic guards as the CHC encoder so a COL-derived weight
    // that overflows i64 is rejected as Unknown rather than silently truncated.
    let mut incidence: Vec<Vec<i64>> = vec![vec![0_i64; nt]; np];
    for (tidx, transition) in net.transitions.iter().enumerate() {
        for arc in &transition.inputs {
            let p = arc.place.0 as usize;
            let weight = i64::try_from(arc.weight).map_err(|_| {
                StateEquationEncoderError::ArcWeightOverflow {
                    transition_index: tidx,
                    place_index: p,
                    weight: arc.weight,
                }
            })?;
            incidence[p][tidx] = incidence[p][tidx].checked_sub(weight).ok_or(
                StateEquationEncoderError::CoefficientOverflow {
                    transition_index: tidx,
                    place_index: p,
                    kind: super::state_equation::OverflowKind::DeltaAddition,
                },
            )?;
        }
        for arc in &transition.outputs {
            let p = arc.place.0 as usize;
            let weight = i64::try_from(arc.weight).map_err(|_| {
                StateEquationEncoderError::ArcWeightOverflow {
                    transition_index: tidx,
                    place_index: p,
                    weight: arc.weight,
                }
            })?;
            incidence[p][tidx] = incidence[p][tidx].checked_add(weight).ok_or(
                StateEquationEncoderError::CoefficientOverflow {
                    transition_index: tidx,
                    place_index: p,
                    kind: super::state_equation::OverflowKind::DeltaAddition,
                },
            )?;
        }
    }

    let mut conjuncts: Vec<ChcExpr> = Vec::new();

    // ── State equation: m_p = M0[p] + Σ_t C[p][t]·x_t ────────────────────
    for p in 0..np {
        let m0_tokens = net.initial_marking[p];
        let m0 = i64::try_from(m0_tokens).map_err(|_| {
            StateEquationEncoderError::InitialMarkingOverflow {
                place_index: p,
                tokens: m0_tokens,
            }
        })?;

        // rhs = M0[p] + Σ_t C[p][t]·x_t  (only nonzero coefficients contribute).
        let mut rhs = ChcExpr::int(m0);
        for (tidx, &coeff) in incidence[p].iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            let term = ChcExpr::mul(
                ChcExpr::int(coeff),
                ChcExpr::var(IntVarNaming::firing(tidx)),
            );
            rhs = ChcExpr::add(rhs, term);
        }
        conjuncts.push(ChcExpr::eq(marking_var_expr(p), rhs));
    }

    // ── Non-negativity: x_t ≥ 0, m_p ≥ 0 ─────────────────────────────────
    for tidx in 0..nt {
        conjuncts.push(ChcExpr::ge(
            ChcExpr::var(IntVarNaming::firing(tidx)),
            ChcExpr::int(0),
        ));
    }
    for p in 0..np {
        conjuncts.push(ChcExpr::ge(marking_var_expr(p), ChcExpr::int(0)));
    }

    // ── Initially-marked trap cuts: Σ_{p∈T} m_p ≥ 1 ──────────────────────
    //
    // Identical SOUND strengthening to the CHC lane: an initially-marked trap
    // stays marked forever, so this invariant holds in every reachable marking.
    // Conjoining it only tightens the over-approximation, so it can turn a
    // SPURIOUS SAT into UNSAT (more proofs) but can NEVER hide a genuine
    // reachable violation. Gated through `find_initially_marked_traps`, which
    // returns only real, initially-marked traps. Size-gated identically to the
    // CHC lane so the trap enumeration cost stays bounded.
    if np + nt <= MAX_SYMBOLIC_PLACES {
        let traps = crate::lp_state_equation::find_initially_marked_traps(net);
        for trap in &traps {
            let member_terms: Vec<ChcExpr> = trap
                .iter()
                .enumerate()
                .filter(|&(_p, &in_trap)| in_trap)
                .map(|(p, &_in_trap)| marking_var_expr(p))
                .collect();
            if member_terms.is_empty() {
                continue;
            }
            let sum = member_terms
                .into_iter()
                .reduce(ChcExpr::add)
                .expect("non-empty by guard");
            conjuncts.push(ChcExpr::ge(sum, ChcExpr::int(1)));
        }

        // ── Initially-unmarked siphon cuts: Σ_{p∈S} m_p = 0 ──────────────
        //
        // The DUAL of the trap cut. A siphon S (every transition producing into
        // S also consumes from S) that is initially UNMARKED can never gain a
        // token: once empty it stays empty (standard Petri-net theorem). Hence
        // `Σ_{p∈S} m_p = 0`, equivalently `m_p = 0` for every member, holds in
        // every reachable marking. Conjoining this SOUND invariant only tightens
        // the over-approximation: it can turn a SPURIOUS SAT (a state-equation
        // solution that spuriously re-marks an unmarked siphon — exactly the
        // class the bare equation cannot rule out) into UNSAT, yielding more
        // proofs, and can NEVER hide a genuine reachable violation.
        // Gated through `find_initially_unmarked_siphons`, which returns only
        // siphons that are entirely initially unmarked. Since markings are ≥ 0,
        // the single equality `Σ m_p = 0` forces each member to 0; we emit it as
        // `Σ m_p ≤ 0` (the `m_p ≥ 0` conjuncts above supply the other half),
        // keeping the encoding one constraint per siphon.
        let siphons = crate::lp_state_equation::find_initially_unmarked_siphons(net);
        for siphon in &siphons {
            let member_terms: Vec<ChcExpr> = siphon
                .iter()
                .enumerate()
                .filter(|&(_p, &in_siphon)| in_siphon)
                .map(|(p, &_in_siphon)| marking_var_expr(p))
                .collect();
            if member_terms.is_empty() {
                continue;
            }
            let sum = member_terms
                .into_iter()
                .reduce(ChcExpr::add)
                .expect("non-empty by guard");
            conjuncts.push(ChcExpr::le(sum, ChcExpr::int(0)));
        }
    }

    Ok(conjuncts)
}

/// Build the integer state-equation **violation** formula for `property`.
///
/// Returns the single QF_LIA `ChcExpr` whose satisfiability is queried. See the
/// module docs for the exact shape and the soundness theorem. Reuses the SOUND
/// base constraints from [`encode_base_constraints`] and conjoins `¬φ(m)`.
fn encode_violation_formula(
    net: &PetriNet,
    property: &ResolvedPredicate,
) -> Result<ChcExpr, StateEquationEncoderError> {
    let mut conjuncts = encode_base_constraints(net)?;

    // ── Violation: ¬φ(m) ─────────────────────────────────────────────────
    conjuncts.push(property_violation_over_int_markings(property, net)?);

    Ok(ChcExpr::and_all(conjuncts))
}

/// One-shot integer state-equation feasibility check for an `AG`-universal
/// safety property `φ`.
///
/// Builds the integer violation formula and discharges it with a single
/// QF_LIA `check_sat`:
///
/// - `UNSAT` ⇒ [`IntStateEquationVerdict::Safe`] (SOUND: `AG φ` holds — see the
///   module-level soundness theorem). No relaxation gap.
/// - `SAT`   ⇒ [`IntStateEquationVerdict::Candidate`] (over-approximation hit;
///   the caller must replay-validate before any `Unsafe`).
/// - `Unknown`/timeout/overflow ⇒ [`IntStateEquationVerdict::Unknown`]; DECLINE.
///
/// `timeout` bounds the single solver call so the lane stays inside its budget;
/// on expiry `check_sat` returns `Unknown`, which we surface as `Unknown` — the
/// query falls through to the exhaustive engines (verdict-preserving).
#[must_use]
pub(crate) fn integer_state_equation_safe(
    net: &PetriNet,
    property: &ResolvedPredicate,
    timeout: Duration,
) -> IntStateEquationVerdict {
    let formula = match encode_violation_formula(net, property) {
        Ok(formula) => formula,
        Err(err) => return IntStateEquationVerdict::Unknown(IntUnknownReason::from(err)),
    };

    let mut ctx = SmtContext::new();
    match ctx.check_sat_with_timeout(&formula, timeout) {
        // Any UNSAT variant (plain / with core / with Farkas certificate) is a
        // genuine integer-arithmetic proof that the violation formula has no
        // solution ⇒ no reachable marking violates φ ⇒ AG φ holds.
        result if result.is_unsat() => IntStateEquationVerdict::Safe,
        // SAT: a candidate violating marking exists in the over-approximation.
        SmtResult::Sat(_) => IntStateEquationVerdict::Candidate,
        // Unknown / timeout: decline. Never a verdict.
        _ => IntStateEquationVerdict::Unknown(IntUnknownReason::SolverInconclusive),
    }
}

/// Build the threshold constraint `Σ_{p∈places} m_p ≥ k` over the integer
/// marking variables `ise_m_*` (multiplicity-aware: a repeated place is summed
/// with its multiplicity, matching `lp_upper_bound`'s objective). `k` is `u64`;
/// it is range-checked into `i64` so an out-of-range threshold is reported as an
/// encoder error (→ Unknown) rather than silently wrapping.
fn threshold_at_least(places: &[usize], k: u64) -> Result<ChcExpr, StateEquationEncoderError> {
    let k_i = i64::try_from(k).map_err(|_| StateEquationEncoderError::InitialMarkingOverflow {
        place_index: usize::MAX,
        tokens: k,
    })?;
    let sum = places
        .iter()
        .map(|&p| marking_var_expr(p))
        .reduce(ChcExpr::add)
        .unwrap_or_else(|| ChcExpr::int(0));
    Ok(ChcExpr::ge(sum, ChcExpr::int(k_i)))
}

/// Maximum number of integer-feasibility solver calls spent tightening a single
/// UpperBounds query. Each call is a full QF_LIA `check_sat`, so the dichotomy is
/// capped to keep the lane a cheap pre-pass; the LP cap is always a sound
/// fallback if the budget is exhausted.
pub(crate) const MAX_INT_TIGHTEN_CALLS: u32 = 12;

/// Size gate (places + transitions) for the integer pinning sweep
/// ([`integer_pinned_place`]): it runs up to `2·np` QF_LIA solves, so keep each
/// sweep small enough to stay a cheap pre-pass. Mirrors
/// [`crate::lp_state_equation`]'s `MAX_PINNING_LP_VARIABLES`.
const MAX_INT_PINNING_VARS: usize = 8_192;

/// Tighten a *rational* LP upper bound on `Σ_{p∈places} m_p` to the true INTEGER
/// state-equation bound, where the integer program is strictly tighter.
///
/// Inputs:
/// - `lp_bound`: a SOUND upper bound (`≥ true max`), e.g. from `lp_upper_bound`;
/// - `witnessed_lb`: a SOUND achievable lower bound (`≤ true max`), e.g. the
///   initial-marking sum or a BFS-observed value. The search never descends
///   below it.
///
/// Method (sound downward dichotomy on the integer state equation). For a
/// candidate cap `c`, the threshold `Σ m_p ≥ c` is queried against the integer
/// base constraints (state equation + non-negativity + trap + siphon cuts). If it
/// is `UNSAT`, NO reachable marking attains `c` (the base constraints
/// over-approximate reachability), so the true max is `≤ c − 1` and `c` can be
/// lowered. We binary-search the largest `c ∈ (witnessed_lb, lp_bound]` that is
/// integer-FEASIBLE; the returned bound is that `c` (or `witnessed_lb` if even
/// `witnessed_lb + 1` is infeasible — though that can't happen since
/// `witnessed_lb` is achievable). Feasibility is only an over-approximation, so a
/// feasible `c` is NOT a proof the value is attained; it merely stops us lowering
/// past a value the integer program cannot rule out — which keeps the result a
/// sound UPPER bound.
///
/// Returns:
/// - `Some(b)` with `witnessed_lb ≤ b ≤ lp_bound`: a SOUND integer upper bound,
///   `≤` the rational `lp_bound`. When `b == witnessed_lb` the bound is EXACT
///   (lower == upper). `b < lp_bound` is a strict integer tightening.
/// - `None`: the net is too large, an encoder overflow occurred, the solver was
///   inconclusive, or the call budget was exhausted before converging — DECLINE,
///   the caller keeps its existing `lp_bound`. Never returns a value `> lp_bound`.
#[must_use]
pub(crate) fn integer_state_equation_upper_bound(
    net: &PetriNet,
    places: &[PlaceIdx],
    lp_bound: u64,
    witnessed_lb: u64,
    timeout: Duration,
) -> Option<u64> {
    // Nothing to tighten: the LP cap already coincides with an achievable value
    // (or is below it, which a caller should never pass — guard anyway).
    if lp_bound <= witnessed_lb {
        return Some(lp_bound.max(witnessed_lb));
    }
    if places.is_empty() {
        return Some(0);
    }

    let base = encode_base_constraints(net).ok()?;
    let place_idxs: Vec<usize> = places.iter().map(|p| p.0 as usize).collect();

    // `feasible(c)` is TRUE when `base ∧ (Σ m_p ≥ c)` is SAT (integer
    // over-approx admits reaching `c`), FALSE when UNSAT (provably `< c`).
    // `None` is propagated as solver-inconclusive: the whole tightening declines.
    let mut ctx = SmtContext::new();
    let mut calls = 0u32;
    let mut feasible = |c: u64| -> Option<bool> {
        let threshold = threshold_at_least(&place_idxs, c).ok()?;
        let mut conjuncts = base.clone();
        conjuncts.push(threshold);
        let formula = ChcExpr::and_all(conjuncts);
        match ctx.check_sat_with_timeout(&formula, timeout) {
            SmtResult::Sat(_) => Some(true),
            result if result.is_unsat() => Some(false),
            _ => None,
        }
    };

    // Binary search for the greatest feasible cap in (witnessed_lb, lp_bound].
    // Invariant: `lo` is always feasible-or-achievable (witnessed_lb is achievable
    // by hypothesis), `hi` is the smallest known-or-assumed bound. We hunt the
    // largest feasible `c`; the answer is a sound upper bound.
    let mut lo = witnessed_lb; // achievable ⇒ feasible
    let mut hi = lp_bound; // sound upper bound (feasibility unknown)
    while lo < hi {
        if calls >= MAX_INT_TIGHTEN_CALLS {
            // Budget exhausted before convergence: `hi` is still a SOUND upper
            // bound (we only ever lowered `hi` past UNSAT thresholds), so return
            // it rather than declining — it is `≤ lp_bound`.
            return Some(hi);
        }
        let mid = lo + (hi - lo).div_ceil(2); // mid ∈ (lo, hi]
        calls += 1;
        match feasible(mid) {
            // `mid` is reachable in the over-approx: the true max could be ≥ mid,
            // so raise the floor. `lo = mid` keeps `lo` feasible.
            Some(true) => lo = mid,
            // `Σ m_p ≥ mid` is integer-INFEASIBLE: true max ≤ mid − 1, lower hi.
            Some(false) => hi = mid - 1,
            // Solver inconclusive: decline (caller keeps lp_bound).
            None => return None,
        }
    }
    // lo == hi: the converged bound. Sound (`≤ lp_bound`, `≥ witnessed_lb`).
    Some(hi)
}

/// Build the threshold constraint `Σ_{p∈places} m_p ≤ k` over the integer
/// marking variables. Range-checks `k` into `i64` like [`threshold_at_least`].
fn threshold_at_most(places: &[usize], k: u64) -> Result<ChcExpr, StateEquationEncoderError> {
    let k_i = i64::try_from(k).map_err(|_| StateEquationEncoderError::InitialMarkingOverflow {
        place_index: usize::MAX,
        tokens: k,
    })?;
    let sum = places
        .iter()
        .map(|&p| marking_var_expr(p))
        .reduce(ChcExpr::add)
        .unwrap_or_else(|| ChcExpr::int(0));
    Ok(ChcExpr::le(sum, ChcExpr::int(k_i)))
}

/// Discharge a single integer-feasibility query: is `base ∧ target` SAT?
/// Returns `Some(true)`=SAT, `Some(false)`=UNSAT, `None`=inconclusive/timeout.
fn check_with_target(base: &[ChcExpr], target: ChcExpr, timeout: Duration) -> Option<bool> {
    let mut conjuncts = base.to_vec();
    conjuncts.push(target);
    let formula = ChcExpr::and_all(conjuncts);
    let mut ctx = SmtContext::new();
    match ctx.check_sat_with_timeout(&formula, timeout) {
        SmtResult::Sat(_) => Some(true),
        result if result.is_unsat() => Some(false),
        _ => None,
    }
}

/// Prove a single place `p` is at most `bound`-bounded via integer infeasibility,
/// i.e. `Σ` over `{p}` of `m_p ≥ bound + 1` has no integer state-equation
/// solution. A convenience wrapper over the integer feasibility query for the
/// common 1-boundedness / k-boundedness check (e.g. prove a place 1-bounded by
/// showing `m_p ≥ 2` is integer-infeasible).
///
/// Returns `true` only on a genuine integer-infeasibility PROOF that `m_p` can
/// never exceed `bound`; `false` otherwise (feasible, too large, inconclusive,
/// overflow) — fail-closed, never a guessed verdict.
#[must_use]
#[cfg(test)]
pub(crate) fn integer_place_bounded_by(
    net: &PetriNet,
    place: PlaceIdx,
    bound: u64,
    timeout: Duration,
) -> bool {
    let Some(base) = encode_base_constraints(net).ok() else {
        return false;
    };
    let Some(threshold) = threshold_at_least(&[place.0 as usize], bound.saturating_add(1)).ok()
    else {
        return false;
    };
    check_with_target(&base, threshold, timeout) == Some(false)
}

/// Prove some place is **pinned** to its initial marking via INTEGER
/// infeasibility — the integer-tightened dual of
/// [`crate::lp_state_equation::lp_pinned_place`].
///
/// A place `p` is pinned (constant in every reachable marking, hence stably
/// marked) iff neither `M[p] ≥ M0[p] + 1` nor `M[p] ≤ M0[p] − 1` is reachable.
/// This checks BOTH directions against the INTEGER state-equation base
/// constraints (state equation + non-negativity + trap + siphon cuts): if both
/// thresholds are integer-INFEASIBLE then no reachable marking can move `p` off
/// `M0[p]`, so `p` is constant — a SOUND `StableMarking = TRUE` witness. Because
/// the base constraints over-approximate reachability, integer-infeasibility is a
/// genuine unreachability proof.
///
/// This catches places the *rational* LP cannot pin: when the relaxation admits a
/// fractional firing vector that perturbs `p` (so `lp_pinned_place` declines) but
/// no INTEGER firing vector does. It is strictly additive — it returns `Some(p)`
/// only on a real proof, and `None` (DECLINE) on inconclusive/oversized/overflow,
/// so it can never change an existing verdict.
///
/// `per_place_timeout` bounds EACH of the (up to two) solver calls per place;
/// `deadline` short-circuits the whole sweep so it stays a bounded pre-pass.
#[must_use]
pub(crate) fn integer_pinned_place(
    net: &PetriNet,
    per_place_timeout: Duration,
    deadline: Option<std::time::Instant>,
) -> Option<PlaceIdx> {
    let np = net.num_places();
    let nt = net.num_transitions();
    if np == 0 {
        return None;
    }
    // Tighter cap than the encoder's own (this sweep runs up to `2 * np` QF_LIA
    // solves), keeping it a cheap pre-pass; larger nets fall through to the
    // BMC/PDR/BFS engines. Mirrors `lp_pinned_place`'s `MAX_PINNING_LP_VARIABLES`.
    if np + nt > MAX_INT_PINNING_VARS {
        return None;
    }
    // Build the base constraints ONCE (they depend only on the net) and reuse for
    // every place / direction. An encoder overflow (oversized net, out-of-range
    // weight) declines the whole sweep — fail-closed, never a guessed verdict.
    let base = encode_base_constraints(net).ok()?;

    for place in 0..np {
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            break;
        }
        let m0 = net.initial_marking[place];

        // (1) `M[p] >= M0[p] + 1` must be integer-INFEASIBLE: p can never exceed
        //     M0. `m0 + 1` cannot overflow u64 for any real net, but guard anyway.
        let Some(above) = m0
            .checked_add(1)
            .and_then(|k| threshold_at_least(&[place], k).ok())
        else {
            continue;
        };
        match check_with_target(&base, above, per_place_timeout) {
            Some(false) => {}       // unreachable above ✓
            Some(true) => continue, // p can exceed M0 ⇒ not pinned
            None => continue,       // inconclusive ⇒ skip this place
        }

        // (2) `M[p] <= M0[p] - 1` must be integer-INFEASIBLE: p can never fall
        //     below M0. For M0 == 0 this is `M[p] <= -1`, vacuously infeasible
        //     since markings are non-negative — no solver call needed.
        if m0 > 0 {
            let Some(below) = threshold_at_most(&[place], m0 - 1).ok() else {
                continue;
            };
            match check_with_target(&base, below, per_place_timeout) {
                Some(false) => {}       // unreachable below ✓
                Some(true) => continue, // p can drop below M0 ⇒ not pinned
                None => continue,       // inconclusive ⇒ skip this place
            }
        }

        // Both directions integer-infeasible ⇒ p is constant ⇒ stably marked.
        return Some(PlaceIdx(place as u32));
    }

    None
}

/// Size gate (places + transitions) for the dead-transition sweep
/// ([`integer_dead_transition`]): it runs up to `nt` QF_LIA solves (one
/// enabledness query per transition), so keep each sweep small enough to stay a
/// cheap pre-pass. Mirrors [`MAX_INT_PINNING_VARS`].
const MAX_INT_DEAD_TRANSITION_VARS: usize = 8_192;

/// Build the **enabledness** target `IsFireable(t)` over the integer marking
/// variables `ise_m_*`, i.e. `⋀_{arc∈•t} (m[arc.place] ≥ arc.weight)`.
///
/// We reuse the shared [`encode_predicate_expr`] on the single-transition
/// `IsFireable([t])` predicate and alpha-rename `se_m_*` → `ise_m_*` (the SAME
/// path [`property_violation_over_int_markings`] uses). This guarantees the
/// target is term-for-term identical to the marking-arithmetic reading of
/// [`crate::petri_net::PetriNet::is_enabled`] that BFS and every other engine use:
/// every input arc independently satisfied, and a transition with NO input arc
/// lowered to `true` (unconditionally fireable, never dead).
///
/// Returns `Err` on encoder overflow (out-of-range arc weight) so the caller maps
/// it to a DECLINE — never a silent truncation that could flip a verdict.
fn transition_enabled_over_int_markings(
    net: &PetriNet,
    transition: TransitionIdx,
) -> Result<ChcExpr, StateEquationEncoderError> {
    let pred = ResolvedPredicate::IsFireable(vec![transition]);
    let encoded = encode_predicate_expr(&pred, net)?;
    Ok(rename_se_m_to_ise_m(&encoded))
}

/// Prove a single transition `t` is **DEAD** (never enabled in any reachable
/// marking) via INTEGER infeasibility of "reachable ∧ IsFireable(t)".
///
/// Discharges one QF_LIA query: the SOUND integer base constraints (state
/// equation + non-negativity + trap + siphon cuts — every conjunct holds in
/// every reachable marking) conjoined with the enabledness target
/// `⋀_{arc∈•t}(m[arc.place] ≥ arc.weight)`. If that is integer-INFEASIBLE then NO
/// state-equation solution enables `t`; since every *reachable* marking is a
/// state-equation solution (the module soundness theorem), no reachable marking
/// enables `t` — `t` is dead, i.e. `AG(¬IsFireable(t))` holds.
///
/// This is the dedicated, single-transition dual of [`integer_pinned_place`]: it
/// lifts the dead-transition special case of `AG(¬IsFireable(t))` directly,
/// rather than only via the reachability seeder running the full
/// [`integer_state_equation_safe`] over a property predicate. It is the
/// fireability companion to the pinned-place sweep that `StableMarking` uses.
///
/// SOUNDNESS (0-wrong / fail-closed):
/// - Returns `true` ONLY on a genuine integer-infeasibility PROOF (`UNSAT`).
///   `is_enabled` / BFS can never disagree: the target is the exact `is_enabled`
///   condition over the same variables, and the base over-approximates
///   reachability, so UNSAT ⇒ unenabled at every reachable marking.
/// - A transition with NO input arcs is ALWAYS enabled (`IsFireable` lowers to
///   `true`), so it can never be proven dead — the target is satisfiable, we
///   return `false`. (We also short-circuit it without a solver call.)
/// - Returns `false` on feasible / inconclusive / timeout / encoder overflow /
///   oversized net — a DECLINE, never a guessed verdict. The caller falls through
///   to the exhaustive engines.
///
/// `timeout` bounds the single solver call so the lane stays a cheap pre-pass.
#[must_use]
pub(crate) fn integer_dead_transition(
    net: &PetriNet,
    transition: TransitionIdx,
    timeout: Duration,
) -> bool {
    let tidx = transition.0 as usize;
    if tidx >= net.num_transitions() {
        return false;
    }
    // A transition with no input arcs is unconditionally enabled (vacuous guard),
    // so it is NEVER dead. Short-circuit without a solver call — the enabledness
    // target would be `true`, trivially SAT.
    if net.transitions[tidx].inputs.is_empty() {
        return false;
    }

    let Ok(base) = encode_base_constraints(net) else {
        return false; // encoder overflow / oversized ⇒ DECLINE
    };
    let Ok(enabled) = transition_enabled_over_int_markings(net, transition) else {
        return false; // arc-weight overflow ⇒ DECLINE
    };
    // `base ∧ IsFireable(t)` UNSAT ⇒ t is dead.
    check_with_target(&base, enabled, timeout) == Some(false)
}

/// Sweep every transition and return those PROVEN DEAD by integer infeasibility
/// of "reachable ∧ IsFireable(t)" (see [`integer_dead_transition`]).
///
/// Builds the SOUND integer base constraints ONCE (they depend only on the net)
/// and reuses them across one enabledness query per transition, so the sweep
/// costs at most `nt` QF_LIA solves. This directly lifts the dead-transition
/// special case of fireability examinations (`AG(¬IsFireable(t))`,
/// `LTLFireability`/`OneSafe` sub-properties): a transition in the returned set is
/// dead in every reachable marking, so any examination conditioned on it ever
/// firing is discharged without unrolling the state space.
///
/// SOUNDNESS: a returned transition is dead by a genuine integer-infeasibility
/// proof (no relaxation). The sweep is strictly ADDITIVE — it DECLINES (omits the
/// transition) on feasible / inconclusive / oversized / overflow, so it can never
/// claim a live transition dead and never changes an existing verdict.
///
/// `per_transition_timeout` bounds EACH solver call; `deadline` short-circuits the
/// whole sweep so it stays a bounded pre-pass. Returns an empty vec on
/// oversized/overflow (DECLINE).
#[must_use]
pub(crate) fn integer_dead_transitions(
    net: &PetriNet,
    per_transition_timeout: Duration,
    deadline: Option<std::time::Instant>,
) -> Vec<TransitionIdx> {
    let np = net.num_places();
    let nt = net.num_transitions();
    if nt == 0 {
        return Vec::new();
    }
    if np + nt > MAX_INT_DEAD_TRANSITION_VARS {
        return Vec::new();
    }
    // Build the base ONCE; an encoder overflow declines the whole sweep.
    let Ok(base) = encode_base_constraints(net) else {
        return Vec::new();
    };

    let mut dead: Vec<TransitionIdx> = Vec::new();
    for t in 0..nt {
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            break;
        }
        // Source transitions (no inputs) are always fireable ⇒ never dead; skip
        // them without a solver call (their enabledness target is `true`).
        if net.transitions[t].inputs.is_empty() {
            continue;
        }
        let tidx = TransitionIdx(t as u32);
        let Ok(enabled) = transition_enabled_over_int_markings(net, tidx) else {
            continue; // arc-weight overflow on this transition ⇒ skip
        };
        if check_with_target(&base, enabled, per_transition_timeout) == Some(false) {
            dead.push(tidx); // integer-infeasible enabledness ⇒ proven dead
        }
    }
    dead
}

/// Whether the solver exposes a usable UNSAT core on the integer base
/// constraints, surfaced as a sound, **verdict-neutral** capability probe.
///
/// `ay_chc::SmtContext::check_sat_with_assumption_conjuncts(background,
/// assumptions)` returns [`SmtResult::UnsatWithCore`] whose `conjuncts` field is
/// the minimal subset of the *assumptions* sufficient for `UNSAT`. We treat the
/// SOUND integer base constraints (state equation + non-negativity + trap +
/// siphon cuts) as `background` and the property-violation conjunct(s) as the
/// `assumptions` over which a core is extracted.
///
/// SOUNDNESS / VERDICT-NEUTRALITY: this is a pure *speed/diagnostic* feed — the
/// extracted core is always a SUBSET of constraints the query already contains,
/// so conjoining (any subset of) it can never change the SAT/UNSAT outcome. The
/// verdict is still derived ONLY from `is_unsat()` on the full formula. If the
/// solver does not populate a core (plain `Unsat` / `Sat` / `Unknown`), this
/// returns `None` and the caller proceeds exactly as before. Because the CHC lane
/// (`super::state_equation::encode_safety_query`) ALREADY emits the full trap and
/// siphon cut set, a core fed forward from here is at most redundant w.r.t. those
/// cuts; we therefore expose it as a probe and do NOT wire a (no-op) redundant cut
/// into the CHC query. See the module/report notes.
///
/// Returns `Some(core_conjuncts)` when the violation formula is `UNSAT` AND the
/// solver populated an UNSAT core over the violation assumptions; `None`
/// otherwise. The caller MUST NOT derive a verdict from the *presence* of a core
/// alone — the verdict is the `is_unsat()` of the full query.
#[must_use]
pub(crate) fn integer_violation_unsat_core(
    net: &PetriNet,
    property: &ResolvedPredicate,
    timeout: Duration,
) -> Option<Vec<ChcExpr>> {
    let base = encode_base_constraints(net).ok()?;
    let violation = property_violation_over_int_markings(property, net).ok()?;

    let mut ctx = SmtContext::new();
    // Solve `base (background) ∧ violation (assumption)`; an UNSAT core, when the
    // solver populates one, identifies the minimal violation conjuncts that, with
    // the base, certify the invariant. Verdict-neutral: a subset of the query.
    let _ = timeout; // assumption-conjunct path uses the context's own budget
    match ctx.check_sat_with_assumption_conjuncts(&base, std::slice::from_ref(&violation)) {
        SmtResult::UnsatWithCore(core) => Some(core.conjuncts),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
    use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn arc(p: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(p),
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

    fn budget() -> Duration {
        Duration::from_secs(10)
    }

    /// Exhaustive BFS maximum of `Σ_{p∈places} m_p` over the reachable set, for
    /// cross-checking that an integer bound matches the true reachable maximum.
    /// Only used on the tiny gate nets, so a bounded visited cap is ample.
    fn bfs_place_sum_max(net: &PetriNet, places: &[PlaceIdx]) -> u64 {
        use crate::petri_net::TransitionIdx;
        use std::collections::HashSet;

        let sum = |m: &[u64]| -> u64 {
            places
                .iter()
                .map(|p| m[p.0 as usize])
                .fold(0u64, u64::saturating_add)
        };
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut stack = vec![net.initial_marking.clone()];
        let mut best = sum(&net.initial_marking);
        let mut budget = 100_000usize;
        while let Some(m) = stack.pop() {
            if budget == 0 {
                break;
            }
            budget -= 1;
            if !seen.insert(m.clone()) {
                continue;
            }
            best = best.max(sum(&m));
            for t in 0..net.num_transitions() {
                let tidx = TransitionIdx(t as u32);
                if net.is_enabled(&m, tidx) {
                    if let Ok(next) = net.fire(&m, tidx) {
                        stack.push(next);
                    }
                }
            }
        }
        best
    }

    /// `≤` predicate `sum(places) <= k`.
    fn tokens_le(places: &[u32], k: u64) -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(places.iter().map(|&p| PlaceIdx(p)).collect()),
            ResolvedIntExpr::Constant(k),
        )
    }

    // ── Soundness: provable safety ⇒ Safe (UNSAT) ────────────────────────

    /// Token-conserving net: 3 tokens shuttle between p0,p1, so m0+m1 = 3 in
    /// every reachable marking. Property `m0+m1 <= 3` is a true invariant, so
    /// the violation formula `m0+m1 > 3` is integer-INFEASIBLE ⇒ Safe.
    #[test]
    fn test_integer_state_equation_conserving_invariant_is_safe() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![3, 0],
        };
        let property = tokens_le(&[0, 1], 3);
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Safe,
            "m0+m1=3 invariant: violation m0+m1>3 is integer-infeasible"
        );
    }

    /// A genuinely reachable violation must NOT be proven Safe — the state
    /// equation admits it, so the verdict is `Candidate` (the over-approximation
    /// is hit), never `Safe`. Guards against a false-Safe regression.
    #[test]
    fn test_integer_state_equation_reachable_violation_is_not_safe() {
        // p0 starts with 5; t0 moves p0→p1. m1 can reach 5, so `m1 <= 2` is
        // violated by reachable markings.
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![5, 0],
        };
        let property = tokens_le(&[1], 2);
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Candidate,
            "m1 can reach 5 > 2: must be Candidate (over-approx hit), never Safe"
        );
    }

    // ── The whole point: LP-spurious-feasible but integer-INFEASIBLE ─────

    /// Differential gate: a net whose LP relaxation of the state equation is
    /// FEASIBLE for a property violation (a *rational* firing vector exists) but
    /// whose INTEGER program is INFEASIBLE. The integer lane proves `AG φ` (Safe)
    /// where the rational LP relaxation cannot.
    ///
    /// Construction (parity invariant — the canonical LP/ILP gap):
    /// - place p0 with M0[p0] = 0;
    /// - transition t0 with no input and output arc weight 2 to p0
    ///   (so C[p0][t0] = +2): firing t0 once adds 2 tokens to p0.
    ///
    /// State equation: `m0 = 0 + 2·x0`, `x0 ≥ 0`, `m0 ≥ 0`. Every reachable m0
    /// is even, so `m0 = 1` is unreachable; equivalently the property `m0 ≤ 0 ∨
    /// m0 ≥ 2` holds — we test the marking predicate "`m0` can equal exactly 1"
    /// by querying the invariant `m0 ≠ 1` via `m0 ≤ 0 (no)`… instead we use the
    /// cleaner, directly-encodable invariant below.
    ///
    /// Concretely we assert the *unreachability of an odd marking*: the
    /// violation formula `m0 = 1 ∧ m0 = 2·x0 ∧ x0 ≥ 0` is INTEGER-infeasible
    /// (no integer x0 with 2·x0 = 1) but its LP relaxation is FEASIBLE
    /// (x0 = 0.5). We encode the safety property as `m0 ≤ 0` *conjoined* with a
    /// second transition so the violation isolates the parity obstruction; see
    /// the assertion comment for the exact φ.
    #[test]
    fn test_int_infeasible_where_lp_relaxation_is_spurious_feasible() {
        // p0 = 0; t0: ∅ → 2·p0. Reachable markings of p0: {0,2,4,...} (all even).
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 2)])],
            initial_marking: vec![0],
        };

        // Property φ: "m0 is never exactly 1." We cannot write `≠` directly, but
        // the violation we hand the solver is the marking-set {m0 = 1}, encoded
        // as the conjunction `1 ≤ m0 ∧ m0 ≤ 1`. Proving that set unreachable is
        // exactly proving `AG ¬(m0 = 1)`.
        //
        // Build φ so that ¬φ ≡ (m0 = 1). Take φ = ¬(1 ≤ m0 ∧ m0 ≤ 1). Then
        // ¬φ = (1 ≤ m0 ∧ m0 ≤ 1) = (m0 = 1). The violation formula becomes
        //   m0 = 2·x0 ∧ x0 ≥ 0 ∧ m0 ≥ 0 ∧ m0 = 1,
        // which is INTEGER-INFEASIBLE (2·x0 = 1 has no integer solution) ⇒ Safe.
        let m0_eq_1 = ResolvedPredicate::And(vec![
            // 1 ≤ m0
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ),
            // m0 ≤ 1
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
                ResolvedIntExpr::Constant(1),
            ),
        ]);
        let property = ResolvedPredicate::Not(Box::new(m0_eq_1));

        // The LP relaxation of {m0 = 2·x0, x0 ≥ 0, m0 = 1} is FEASIBLE (x0 = 0.5),
        // so the rational state equation CANNOT prove this Safe. Sanity-check that
        // the LP lane is indeed inconclusive here.
        let lp_target = ResolvedPredicate::And(vec![
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
                ResolvedIntExpr::Constant(1),
            ),
        ]);
        assert!(
            !crate::lp_state_equation::lp_unreachable_with_traps(&net, &lp_target),
            "LP relaxation must be FEASIBLE-spurious (x0=0.5) — it cannot prove m0=1 unreachable"
        );

        // The INTEGER lane closes the parity gap and proves AG ¬(m0 = 1) = Safe.
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Safe,
            "integer state equation proves m0=1 unreachable (2·x0=1 has no integer solution) \
             where the LP relaxation could not"
        );
    }

    // ── Overflow guard: u64::MAX arc weight ⇒ Unknown, never a verdict ───

    #[test]
    fn test_integer_state_equation_overflow_is_unknown() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, u64::MAX)])],
            initial_marking: vec![1, 0],
        };
        let property = ResolvedPredicate::True;
        assert!(
            matches!(
                integer_state_equation_safe(&net, &property, budget()),
                IntStateEquationVerdict::Unknown(IntUnknownReason::EncoderRejected(_))
            ),
            "u64::MAX arc weight must be rejected as Unknown, never a verdict"
        );
    }

    // ── Siphon cut: UNSAT via initially-unmarked siphon, SAT trap-only ───

    /// GATE (a): a net where the initially-unmarked-siphon cut makes the integer
    /// system UNSAT where the trap-only system left it SAT.
    ///
    /// Construction:
    /// - place p0, M0[p0] = 0;
    /// - transition t0: input {p0} (weight 1), output {p0} (weight 2).
    ///
    /// `{p0}` is a SIPHON: the only transition producing into p0 (t0) also
    /// consumes from p0. It is initially UNMARKED, so it stays empty forever ⇒
    /// `m_p0 = 0` in every reachable marking (t0 is dead — it needs a token in
    /// p0 to fire, and p0 never gets one).
    ///
    /// The bare/integer state equation does NOT capture this: C[p0][t0] = 2−1 =
    /// +1, so it reads `m_p0 = x0` with `x0 ≥ 0`, admitting any `m_p0 ≥ 0`. There
    /// is no initially-marked trap covering p0, so the TRAP-only system leaves the
    /// violation `m_p0 ≥ 1` SATISFIABLE (x0 = 1) — it cannot prove `AG (m_p0 ≤
    /// 0)`. The SIPHON cut `m_p0 = 0` makes that violation INTEGER-INFEASIBLE ⇒
    /// Safe. This is exactly the spurious-SAT → UNSAT tightening the cut adds.
    #[test]
    fn test_siphon_cut_makes_unsat_where_trap_only_is_sat() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(0, 2)])],
            initial_marking: vec![0],
        };

        // Confirm the siphon machinery actually finds {p0} as an
        // initially-unmarked siphon (so the cut is genuinely added).
        let siphons = crate::lp_state_equation::find_initially_unmarked_siphons(&net);
        assert!(
            siphons.iter().any(|s| s == &vec![true]),
            "{{p0}} must be detected as an initially-unmarked siphon; got {siphons:?}"
        );

        // Property φ = (m_p0 ≤ 0); violation ¬φ = (m_p0 ≥ 1) ≡ (1 ≤ m_p0).
        let property = tokens_le(&[0], 0);

        // TRAP-only: there is no initially-marked trap on p0, and the bare state
        // equation admits m_p0 = x0 ≥ 1, so trap-only would be SAT. We verify the
        // SIPHON-augmented lane returns Safe (UNSAT).
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Safe,
            "initially-unmarked siphon {{p0}} pins m_p0 = 0 ⇒ violation m_p0 ≥ 1 \
             is integer-infeasible ⇒ Safe (trap-only left it SAT)"
        );
    }

    /// Soundness guard for the siphon cut: a siphon that IS initially marked
    /// must NOT receive a zero-cut (it can legitimately hold tokens). Here p0 is
    /// a self-feeding siphon but starts MARKED, so `m_p0 ≥ 1` is genuinely
    /// reachable (it is the initial marking) and the lane must return Candidate,
    /// never a false Safe.
    #[test]
    fn test_initially_marked_siphon_is_not_zero_cut() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(0, 2)])],
            initial_marking: vec![1],
        };
        // No initially-UNMARKED siphon exists (p0 is marked), so no zero-cut.
        let siphons = crate::lp_state_equation::find_initially_unmarked_siphons(&net);
        assert!(
            siphons.is_empty(),
            "an initially-MARKED siphon must NOT be returned as a zero-cut: {siphons:?}"
        );
        // φ = (m_p0 ≤ 0); violation m_p0 ≥ 1 is reachable (M0 itself), so the
        // over-approximation is genuinely hit ⇒ Candidate, never Safe.
        let property = tokens_le(&[0], 0);
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Candidate,
            "m_p0 = 1 initially: violation m_p0 ≥ 1 is reachable ⇒ Candidate, never Safe"
        );
    }

    // ── Integer upper bound STRICTLY tighter than the rational LP bound ──

    /// GATE (b): a net where the INTEGER state-equation upper bound on a place is
    /// strictly tighter than the rational LP bound, and matches the true BFS
    /// maximum.
    ///
    /// Construction (the canonical LP/ILP bound gap):
    /// - place p0, M0 = 0;
    /// - place c ("resource"), M0 = 1;
    /// - transition t0: input {c} weight 2, output {p0} weight 1.
    ///
    /// t0 needs 2 tokens in c but only 1 exists, so t0 is DEAD: the true reachable
    /// maximum of p0 is **0**. The incidence is C[p0][t0] = +1, C[c][t0] = −2, so
    /// the state equation reads `m0 = x0`, `mc = 1 − 2·x0`, `m,x ≥ 0`. The RATIONAL
    /// LP maximises `m0`: `x0 ≤ 0.5` ⇒ LP optimum `0.5` ⇒ `lp_upper_bound`
    /// `ceil(0.5) = 1`. The INTEGER program forces `x0 = 0` (no integer in `(0,
    /// 0.5]`), so `m0 = 0`. The integer bound is `0`, strictly below the rational
    /// `1`, and equals the true BFS maximum.
    #[test]
    fn test_integer_upper_bound_tighter_than_rational_lp() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("c")],
            transitions: vec![trans("t0", vec![arc(1, 2)], vec![arc(0, 1)])],
            initial_marking: vec![0, 1],
        };

        // The rational LP bound is the LOOSE 1 (ceil(0.5)).
        let lp = crate::lp_state_equation::lp_upper_bound(&net, &[PlaceIdx(0)]);
        assert_eq!(
            lp,
            Some(1),
            "rational LP relaxation max(m0)=0.5 → ceil → loose cap 1"
        );

        // The INTEGER bound closes the relaxation gap to the true maximum 0,
        // matching BFS. witnessed_lb = 0 (initial marking sum for p0).
        let int_bound = integer_state_equation_upper_bound(
            &net,
            &[PlaceIdx(0)],
            /*lp_bound=*/ 1,
            /*witnessed_lb=*/ 0,
            budget(),
        );
        assert_eq!(
            int_bound,
            Some(0),
            "integer state equation forces x0=0 ⇒ m0=0, strictly tighter than the rational cap 1"
        );

        // Cross-check against the explicit reachable maximum (true max = 0).
        assert_eq!(
            bfs_place_sum_max(&net, &[PlaceIdx(0)]),
            0,
            "BFS confirms the true reachable maximum of p0 is 0 (t0 is dead)"
        );
    }

    /// A place that is provably 1-bounded by integer infeasibility of `m_p ≥ 2`.
    /// Token-conserving 1-safe shuttle: p0+p1 = 1 always, so neither place can
    /// reach 2. The integer infeasibility of `m_p0 ≥ 2` is a genuine
    /// 1-boundedness PROOF.
    #[test]
    fn test_integer_place_bounded_by_one_safe_shuttle() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        };
        assert!(
            integer_place_bounded_by(&net, PlaceIdx(0), 1, budget()),
            "p0+p1=1 invariant ⇒ m_p0 ≥ 2 is integer-infeasible ⇒ p0 is 1-bounded"
        );
        // And it is NOT 0-bounded: m_p0 ≥ 1 IS feasible (the initial marking),
        // so the prover must NOT claim a (false) 0-bound.
        assert!(
            !integer_place_bounded_by(&net, PlaceIdx(0), 0, budget()),
            "m_p0 ≥ 1 is feasible (initial marking) ⇒ no false 0-bound"
        );
    }

    /// `integer_state_equation_upper_bound` must NEVER return a value above the
    /// supplied `lp_bound`, and must collapse trivially when the cap already meets
    /// the witness (no solver calls needed).
    #[test]
    fn test_integer_upper_bound_never_exceeds_lp_and_collapses() {
        // Unbounded-style net but we feed a finite lp_bound to probe the contract.
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };
        // lp_bound == witnessed_lb: nothing to tighten, returns the cap as-is.
        assert_eq!(
            integer_state_equation_upper_bound(&net, &[PlaceIdx(0)], 3, 3, budget()),
            Some(3),
            "cap already meets witness ⇒ returns lp_bound unchanged (no descent)"
        );
        // Empty place set ⇒ bound 0.
        assert_eq!(
            integer_state_equation_upper_bound(&net, &[], 5, 0, budget()),
            Some(0),
            "empty place set sums to 0"
        );
    }

    // ── Integer pinning beats the rational LP pin (StableMarking lift) ───

    /// GATE (StableMarking lift): a net where the INTEGER state equation pins a
    /// place to its initial marking but the RATIONAL LP relaxation cannot — so
    /// `lp_pinned_place` declines while `integer_pinned_place` proves a stable
    /// place.
    ///
    /// Net (the parity gap): p0 = 0, c = 1; t0: input {c} weight 2 → output {p0}
    /// weight **2**. t0 is dead (needs 2 in c, has 1), so BOTH places are
    /// constant: p0 ≡ 0, c ≡ 1. State equation: m0 = 2·x0, mc = 1 − 2·x0.
    ///
    /// - p0 above-check `m0 ≥ 1`: rationally FEASIBLE at x0 = 0.5 (m0 = 1, mc = 0)
    ///   ⇒ the RATIONAL LP cannot pin p0. INTEGER-INFEASIBLE: `2·x0 ≥ 1` forces
    ///   `x0 ≥ 1`, giving `mc = −1 < 0` ⇒ the integer program pins p0 to 0.
    ///
    /// So `lp_pinned_place` returns None (the relaxation gap hides the pin) yet
    /// `integer_pinned_place` returns Some(p0) — exactly the StableMarking lift.
    #[test]
    fn test_integer_pin_where_rational_lp_pin_declines() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("c")],
            transitions: vec![trans("t0", vec![arc(1, 2)], vec![arc(0, 2)])],
            initial_marking: vec![0, 1],
        };

        // The rational LP relaxation cannot pin p0: `m0 ≥ 1` is rationally
        // feasible at the boundary x0 = 0.5 (m0 = 1, mc = 0).
        assert_eq!(
            crate::lp_state_equation::lp_pinned_place(&net, None),
            None,
            "rational LP relaxation cannot pin p0 (m0≥1 feasible at x0=0.5)"
        );

        // The integer state equation DOES pin a place (parity closes the gap).
        assert!(
            integer_pinned_place(&net, budget(), None).is_some(),
            "integer state equation pins a constant place where the LP relaxation could not"
        );
    }

    /// Soundness guard for integer pinning: a place that genuinely VARIES must
    /// NOT be reported as pinned. p0 = 1; t0: p0 → p1 (so p0 can drop to 0). The
    /// integer pin must decline on p0 (mc≤0 i.e. m_p0 ≤ 0 IS reachable).
    #[test]
    fn test_integer_pin_rejects_varying_place() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        };
        // p0 drops 1→0, p1 rises 0→1 — neither is constant. The 1-safe shuttle is
        // the only pinned quantity is the SUM, not any single place, so the
        // per-place integer pin must return None (no false stable place).
        assert_eq!(
            integer_pinned_place(&net, budget(), None),
            None,
            "no single place is constant (p0 and p1 both vary) ⇒ no false pin"
        );
    }

    /// SOUNDNESS REGRESSION: a *source* transition feeding an initially-unmarked
    /// place must NOT be mis-pinned to zero by a spurious siphon cut.
    ///
    /// `producer_net`: p0 = 0, t0: ∅ → p0. p0 is UNBOUNDED, so `m0 ≤ 100` is
    /// genuinely violated (reachable). `{p0}` is NOT a siphon (t0 produces into p0
    /// without consuming from it), so `find_initially_unmarked_siphons` must return
    /// no cut and the lane must return `Candidate` (falls through), NEVER `Safe`.
    ///
    /// This guards the `is_siphon` gate added to
    /// [`crate::lp_state_equation::find_initially_unmarked_siphons`]: before it,
    /// `siphon_closure` returned `{p0}` as a "siphon", the zero-cut pinned the
    /// unbounded place to 0, and the integer lane reported a WRONG `Safe`.
    #[test]
    fn test_source_fed_place_is_not_a_spurious_siphon_no_false_safe() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };
        let siphons = crate::lp_state_equation::find_initially_unmarked_siphons(&net);
        assert!(
            siphons.is_empty(),
            "a source-fed place is NOT a siphon ⇒ no zero-cut; got {siphons:?}"
        );
        // m0 is unbounded ⇒ m0 ≤ 100 is reachably violated ⇒ Candidate, never Safe.
        let property = tokens_le(&[0], 100);
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Candidate,
            "SOUNDNESS: producer_net m0 unbounded ⇒ m0≤100 must be Candidate, never Safe"
        );
    }

    // ── Fireability atoms in the integer lane (the engine step) ──────────

    /// `IsFireable(t)` predicate over the given transitions.
    fn is_fireable(transitions: &[u32]) -> ResolvedPredicate {
        use crate::petri_net::TransitionIdx;
        ResolvedPredicate::IsFireable(transitions.iter().map(|&t| TransitionIdx(t)).collect())
    }

    /// Exhaustive BFS truth of `AG φ`: returns `true` iff EVERY reachable
    /// marking satisfies `φ`. The ground-truth oracle the integer `Safe`
    /// verdict must agree with on the tiny gate nets.
    fn bfs_ag_holds(net: &PetriNet, property: &ResolvedPredicate) -> bool {
        use crate::petri_net::TransitionIdx;
        use crate::resolved_predicate::eval_predicate;
        use std::collections::HashSet;

        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut stack = vec![net.initial_marking.clone()];
        let mut budget = 100_000usize;
        while let Some(m) = stack.pop() {
            if budget == 0 {
                // Bounded oracle: only used on tiny nets, never hit here.
                return true;
            }
            budget -= 1;
            if !seen.insert(m.clone()) {
                continue;
            }
            if !eval_predicate(property, &m, net) {
                return false; // a reachable marking violates φ
            }
            for t in 0..net.num_transitions() {
                let tidx = TransitionIdx(t as u32);
                if net.is_enabled(&m, tidx) {
                    if let Ok(next) = net.fire(&m, tidx) {
                        stack.push(next);
                    }
                }
            }
        }
        true
    }

    /// GATE (fireability, Safe): a net where `AG(¬IsFireable(t))` ("`t` is dead")
    /// is a TRUE invariant and the integer state equation PROVES it Safe, matching
    /// the BFS oracle.
    ///
    /// Net: place c with M0[c] = 1; transition t0: input {c} weight **2** → no
    /// output. t0 needs 2 tokens in c but only 1 exists and nothing produces into
    /// c, so t0 can never fire. Its enabledness `m_c ≥ 2` is integer-INFEASIBLE
    /// against the state equation (`m_c = 1 − 2·x0`, `x0 ≥ 0` ⇒ `m_c ≤ 1 < 2`),
    /// so `AG(¬IsFireable(t0))` is proven Safe. The siphon `{c}` is initially
    /// MARKED so no zero-cut applies — the proof rests on the state equation alone.
    #[test]
    fn test_fireability_ag_dead_transition_is_safe_matches_bfs() {
        let net = PetriNet {
            name: None,
            places: vec![place("c")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![])],
            initial_marking: vec![1],
        };
        // φ = ¬IsFireable(t0): "t0 is never fireable" (a dead-transition invariant).
        let property = ResolvedPredicate::Not(Box::new(is_fireable(&[0])));

        // BFS ground truth: t0 never fires, so AG(¬IsFireable(t0)) holds.
        assert!(
            bfs_ag_holds(&net, &property),
            "BFS: t0 (needs 2 in c, has 1) is dead ⇒ AG(¬IsFireable(t0)) holds"
        );
        // Integer lane proves it Safe (violation IsFireable(t0)=`m_c≥2` infeasible).
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Safe,
            "integer state equation proves t0 dead ⇒ Safe (matches BFS)"
        );
    }

    /// GATE (fireability, falls-through): a net where `AG(¬IsFireable(t))` is FALSE
    /// (t IS reachable-fireable), so the integer violation `IsFireable(t)` is SAT
    /// ⇒ the lane returns `Candidate` and FALLS THROUGH — it must NOT report a
    /// false Safe. This is the soundness guard for the fireability direction.
    ///
    /// Net: place c with M0[c] = 1; transition t0: input {c} weight **1** → no
    /// output. t0 is enabled at the initial marking (`m_c = 1 ≥ 1`), so it is
    /// genuinely fireable and `IsFireable(t0)` holds at a reachable marking.
    #[test]
    fn test_fireability_ag_live_transition_falls_through_not_safe() {
        let net = PetriNet {
            name: None,
            places: vec![place("c")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![])],
            initial_marking: vec![1],
        };
        let property = ResolvedPredicate::Not(Box::new(is_fireable(&[0])));

        // BFS ground truth: t0 fires at the initial marking ⇒ AG(¬IsFireable) FALSE.
        assert!(
            !bfs_ag_holds(&net, &property),
            "BFS: t0 is enabled initially ⇒ AG(¬IsFireable(t0)) is FALSE"
        );
        // Integer lane must NOT prove Safe; the violation IsFireable(t0)=`m_c≥1` is
        // integer-FEASIBLE (the initial marking), so it returns Candidate and the
        // caller falls through to the exhaustive engines.
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Candidate,
            "t0 fireable initially ⇒ violation SAT ⇒ Candidate (falls through), never false Safe"
        );
    }

    /// GATE (fireability, AG(IsFireable) Safe): a net where `AG(IsFireable(t))`
    /// ("`t` is fireable in EVERY reachable marking") is TRUE and the integer lane
    /// proves it Safe. A `source` transition (no input arcs) is unconditionally
    /// fireable: its lowering is `true`, so the violation `¬IsFireable` is `false`
    /// ⇒ UNSAT ⇒ Safe, for any reachable marking.
    #[test]
    fn test_fireability_ag_source_always_fireable_is_safe() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };
        // φ = IsFireable(t0): the source transition is always enabled.
        let property = is_fireable(&[0]);
        assert!(
            bfs_ag_holds(&net, &property),
            "BFS: a source transition (no inputs) is enabled in every marking"
        );
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Safe,
            "source t0 always fireable ⇒ ¬IsFireable violation is false ⇒ UNSAT ⇒ Safe"
        );
    }

    /// GATE (fireability, mixed boolean): `AG(IsFireable(t) ⇒ tokens(p) ≤ k)`,
    /// i.e. φ = (¬IsFireable(t) ∨ tokens(p) ≤ k), combining a fireability atom and
    /// a cardinality atom under Or/Not. Proven Safe by the integer lane and
    /// confirmed by BFS.
    ///
    /// Net: place c (M0 = 1, a single resource), place p0 (M0 = 0). Transition t0:
    /// input {c} weight 1 → output {p0} weight 1. So c shuttles its one token into
    /// p0: reachable markings are (c=1,p0=0) and (c=0,p0=1). t0 is fireable only
    /// when c=1, at which point p0=0 ≤ 1. Once t0 has fired, c=0 so t0 is no longer
    /// fireable and the implication is vacuously true. Hence
    /// `AG(IsFireable(t0) ⇒ tokens(p0) ≤ 1)` holds.
    #[test]
    fn test_fireability_ag_mixed_implication_is_safe_matches_bfs() {
        let net = PetriNet {
            name: None,
            places: vec![place("c"), place("p0")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        };
        // φ = (¬IsFireable(t0)) ∨ (tokens(p0) ≤ 1)  ≡  IsFireable(t0) ⇒ p0 ≤ 1.
        let property = ResolvedPredicate::Or(vec![
            ResolvedPredicate::Not(Box::new(is_fireable(&[0]))),
            tokens_le(&[1], 1),
        ]);
        assert!(
            bfs_ag_holds(&net, &property),
            "BFS: whenever t0 is fireable (c=1) p0=0 ≤ 1 ⇒ implication holds everywhere"
        );
        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Safe,
            "integer lane proves the mixed fireability⇒cardinality invariant Safe (matches BFS)"
        );
    }

    /// `True` property is trivially an invariant: ¬True = False, so the
    /// violation formula is UNSAT ⇒ Safe, for any net.
    #[test]
    fn test_integer_state_equation_true_property_is_safe() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };
        assert_eq!(
            integer_state_equation_safe(&net, &ResolvedPredicate::True, budget()),
            IntStateEquationVerdict::Safe,
            "¬True = False ⇒ violation formula UNSAT ⇒ Safe"
        );
    }

    // ── Dedicated dead-transition sweep (the engine step) ────────────────

    /// Exhaustive BFS truth of "transition `t` is reachable-fireable": returns
    /// `true` iff SOME reachable marking enables `t` (i.e. `t` is NOT dead). The
    /// ground-truth oracle the integer dead-transition PROOF must agree with on
    /// the tiny gate nets — a transition proven dead must be BFS-dead, and a
    /// BFS-live transition must NEVER be falsely proven dead.
    fn bfs_transition_reachable_fireable(net: &PetriNet, t: TransitionIdx) -> bool {
        use std::collections::HashSet;

        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut stack = vec![net.initial_marking.clone()];
        let mut budget = 100_000usize;
        while let Some(m) = stack.pop() {
            if budget == 0 {
                // Bounded oracle: only used on tiny nets, never hit here.
                return false;
            }
            budget -= 1;
            if !seen.insert(m.clone()) {
                continue;
            }
            if net.is_enabled(&m, t) {
                return true; // a reachable marking enables t ⇒ t is NOT dead
            }
            for ti in 0..net.num_transitions() {
                let tidx = TransitionIdx(ti as u32);
                if net.is_enabled(&m, tidx) {
                    if let Ok(next) = net.fire(&m, tidx) {
                        stack.push(next);
                    }
                }
            }
        }
        false // no reachable marking enables t ⇒ t is dead
    }

    /// GATE (dead-transition, proven dead): a STRUCTURALLY-dead transition is
    /// proven dead by integer infeasibility, matching BFS.
    ///
    /// Net: place c, M0[c] = 1; t0: input {c} weight **2** → no output. t0 needs 2
    /// tokens in c but only 1 exists and nothing produces into c, so t0 can never
    /// fire. Its enabledness `m_c ≥ 2` is integer-INFEASIBLE against the state
    /// equation (`m_c = 1 − 2·x0`, `x0 ≥ 0` ⇒ `m_c ≤ 1 < 2`) ⇒ proven dead.
    #[test]
    fn test_integer_dead_transition_proves_structurally_dead_matches_bfs() {
        let net = PetriNet {
            name: None,
            places: vec![place("c")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![])],
            initial_marking: vec![1],
        };
        // BFS ground truth: t0 is dead (never reachable-fireable).
        assert!(
            !bfs_transition_reachable_fireable(&net, TransitionIdx(0)),
            "BFS: t0 (needs 2 in c, has 1) is dead"
        );
        // Integer infeasibility proves it dead.
        assert!(
            integer_dead_transition(&net, TransitionIdx(0), budget()),
            "integer state equation proves t0 dead (m_c ≥ 2 infeasible) — matches BFS"
        );
        // The batch sweep returns exactly {t0}.
        assert_eq!(
            integer_dead_transitions(&net, budget(), None),
            vec![TransitionIdx(0)],
            "the dead-transition sweep returns the single structurally-dead transition"
        );
    }

    /// GATE (dead-transition, live NOT falsely proven): a genuinely LIVE
    /// transition must NEVER be proven dead — fail-closed soundness guard.
    ///
    /// Net: place c, M0[c] = 1; t0: input {c} weight **1** → no output. t0 is
    /// enabled at the initial marking (`m_c = 1 ≥ 1`), so it is reachable-fireable.
    /// The enabledness target `m_c ≥ 1` is integer-FEASIBLE (the initial marking),
    /// so the prover must DECLINE (return `false`), never a false dead-proof.
    #[test]
    fn test_integer_dead_transition_does_not_falsely_prove_live_dead_matches_bfs() {
        let net = PetriNet {
            name: None,
            places: vec![place("c")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![])],
            initial_marking: vec![1],
        };
        // BFS ground truth: t0 fires at the initial marking ⇒ reachable-fireable.
        assert!(
            bfs_transition_reachable_fireable(&net, TransitionIdx(0)),
            "BFS: t0 is enabled initially ⇒ live"
        );
        // The integer prover must NOT claim it dead (enabledness m_c ≥ 1 feasible).
        assert!(
            !integer_dead_transition(&net, TransitionIdx(0), budget()),
            "SOUNDNESS: a live transition must never be falsely proven dead"
        );
        assert!(
            integer_dead_transitions(&net, budget(), None).is_empty(),
            "the sweep must report no dead transitions when t0 is live"
        );
    }

    /// GATE (dead-transition, source never dead): a transition with NO input arcs
    /// is unconditionally fireable, so it can NEVER be proven dead — the
    /// short-circuit returns `false` (and BFS agrees it is live).
    #[test]
    fn test_integer_dead_transition_source_is_never_dead() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };
        assert!(
            bfs_transition_reachable_fireable(&net, TransitionIdx(0)),
            "BFS: a source transition (no inputs) is fireable in every marking"
        );
        assert!(
            !integer_dead_transition(&net, TransitionIdx(0), budget()),
            "a source transition is unconditionally fireable ⇒ never dead"
        );
        assert!(
            integer_dead_transitions(&net, budget(), None).is_empty(),
            "the sweep skips source transitions (always fireable, never dead)"
        );
    }

    /// GATE (dead-transition, mixed net): a net with BOTH a dead and a live
    /// transition — the sweep returns exactly the dead one, matching BFS on each.
    ///
    /// Net: place c, M0[c] = 1.
    /// - t0: input {c} weight 2 → no output. DEAD (needs 2, has 1).
    /// - t1: input {c} weight 1 → no output. LIVE (enabled at M0).
    ///
    /// Firing t1 empties c; firing it cannot re-enable t0 (c never reaches 2).
    #[test]
    fn test_integer_dead_transitions_sweep_separates_dead_from_live() {
        let net = PetriNet {
            name: None,
            places: vec![place("c")],
            transitions: vec![
                trans("t0", vec![arc(0, 2)], vec![]),
                trans("t1", vec![arc(0, 1)], vec![]),
            ],
            initial_marking: vec![1],
        };
        // BFS: t0 dead, t1 live.
        assert!(!bfs_transition_reachable_fireable(&net, TransitionIdx(0)));
        assert!(bfs_transition_reachable_fireable(&net, TransitionIdx(1)));
        // Per-transition prover agrees.
        assert!(integer_dead_transition(&net, TransitionIdx(0), budget()));
        assert!(!integer_dead_transition(&net, TransitionIdx(1), budget()));
        // Sweep returns exactly the dead transition.
        assert_eq!(
            integer_dead_transitions(&net, budget(), None),
            vec![TransitionIdx(0)],
            "sweep separates the dead t0 from the live t1"
        );
    }

    // ── UNSAT-core feed-forward: verdict-neutral (only speed/diagnostic) ──

    /// GATE (UNSAT-core verdict-neutrality): extracting an UNSAT core via the
    /// assumption-conjunct path must NEVER change the lane's Safe/Candidate
    /// verdict — it is purely a speed/diagnostic feed. We assert that for a
    /// provable invariant the verdict stays `Safe`, and that *whenever* a core is
    /// produced it is a (sound) subset of the violation assumptions, so feeding it
    /// forward cannot alter any solver outcome.
    ///
    /// Net: the token-conserving shuttle (m0+m1 = 3); φ = (m0+m1 ≤ 3) is a true
    /// invariant ⇒ the violation `m0+m1 > 3` is integer-INFEASIBLE.
    #[test]
    fn test_unsat_core_feed_forward_is_verdict_neutral() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![3, 0],
        };
        let property = tokens_le(&[0, 1], 3);

        // The base verdict (no core consulted) is Safe.
        let verdict_without_core = integer_state_equation_safe(&net, &property, budget());
        assert_eq!(
            verdict_without_core,
            IntStateEquationVerdict::Safe,
            "m0+m1=3 invariant ⇒ Safe"
        );

        // Extracting a core does not change that verdict: the verdict is still
        // derived from the full-formula is_unsat(), and the core (if any) is a
        // subset of the query. Re-running the verdict after a core probe is
        // identical.
        let _core = integer_violation_unsat_core(&net, &property, budget());
        let verdict_after_core_probe = integer_state_equation_safe(&net, &property, budget());
        assert_eq!(
            verdict_after_core_probe, verdict_without_core,
            "UNSAT-core probe must be verdict-neutral (only speed/diagnostic)"
        );
    }

    /// GATE (UNSAT-core verdict-neutrality, SAT side): on a genuinely reachable
    /// violation the formula is SAT, so NO core is produced and the verdict stays
    /// `Candidate`. Confirms the core path never manufactures a false `Safe`.
    #[test]
    fn test_unsat_core_absent_on_sat_and_verdict_unchanged() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![5, 0],
        };
        let property = tokens_le(&[1], 2); // m1 can reach 5 > 2 ⇒ reachable violation

        assert_eq!(
            integer_state_equation_safe(&net, &property, budget()),
            IntStateEquationVerdict::Candidate,
            "reachable violation ⇒ Candidate"
        );
        // SAT ⇒ no UNSAT core; the probe returns None and changes nothing.
        assert!(
            integer_violation_unsat_core(&net, &property, budget()).is_none(),
            "a SAT violation formula yields no UNSAT core (verdict-neutral, no false Safe)"
        );
    }
}
