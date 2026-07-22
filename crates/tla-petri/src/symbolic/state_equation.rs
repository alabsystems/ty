// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Esparza–Melzer state-equation CHC encoder for Petri-net reachability.
//!
//! Encodes the reachability question
//!
//! ```text
//!   ∃M reachable from M0 . Safety(M) is violated
//! ```
//!
//! as a Constrained Horn Clause system over an uninterpreted predicate
//! `Inv(m_0,…,m_{P-1})` whose intended interpretation is "tuple `m` is
//! reachable from `M0`":
//!
//! ```text
//!   m = M0                                     ⇒ Inv(m)        -- Init
//!   Inv(m) ∧ m[•t] ≥ pre(t) ∧ m' = m − Δt    ⇒ Inv(m')       -- Trans
//!   Inv(m) ∧ ¬Safety(m)                       ⇒ ⊥             -- Query
//! ```
//!
//! Together with `m_p ≥ 0` non-negativity, this is the **state equation**:
//! every reachable marking is a solution to `M = M0 + C·x`, x ≥ 0. The
//! converse is *not* true (siphons/traps can rule out spurious solutions),
//! so this encoding is sound for `UNSAFE` witnesses on its own, and is
//! complete for safety when combined with ay-chc's IC3 invariant
//! synthesis.
//!
//! # Overflow safety
//!
//! All transition deltas and coefficient products use `checked_add` /
//! `checked_mul`. Wraparound returns
//! [`StateEquationEncoderError::CoefficientOverflow`] rather than
//! truncating — the dispatcher converts this to
//! [`super::SymbolicVerdict::Unknown`].

use std::fmt;

use ay_chc::{ChcExpr, ChcProblem, ChcSort, ChcVar, ClauseBody, ClauseHead, HornClause};

use crate::petri_net::PetriNet;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

/// Maximum number of places the encoder will accept before bailing out.
///
/// Above this threshold the resulting CHC predicate signature becomes too
/// wide for AdaptivePortfolio's classifier to make useful decisions.
/// Aligns with `MAX_LP_VARIABLES` in `lp_state_equation.rs` so the two
/// state-equation paths bail at the same scale.
pub(crate) const MAX_SYMBOLIC_PLACES: usize = 50_000;

/// Maximum `num_places × num_transitions` the state-equation encoder will
/// materialize. Each transition clause frames every place (`m_p' = m_p` for
/// unaffected p) plus a per-place non-negativity term, so the CHC problem is
/// Θ(places × transitions) — exactly the dense shape that OOMs the PDR encoder
/// (see `pdr_encoding::PDR_ENCODING_MAX_CELLS`). The `MAX_SYMBOLIC_PLACES` gate
/// alone is insufficient: a net can sit under 50 000 places yet have a huge
/// place×transition product (AirplaneLD-PT-4000 is 28 019 × 32 008 ≈ 9×10⁸).
/// Declining above this budget is verdict-preserving — symbolic seeding is a
/// best-effort lane that falls through to BFS.
pub(crate) const MAX_SYMBOLIC_CELLS: usize = 1_000_000;

/// Kill-switch for the trap-invariant query-strengthening cuts.
///
/// When set to a truthy value (`1`, `true`, `yes`, `on`; case-insensitive),
/// `encode_safety_query` emits the bare state-equation CHC system with no
/// initially-marked-trap *and* no initially-unmarked-siphon conjuncts —
/// byte-identical to the pre-cut encoding. (Both structural cuts share this one
/// switch: they are the trap/siphon dual pair, enabled and disabled together.)
/// Provided for differential A/B comparison and as a safety escape hatch.
pub(crate) const DISABLE_CHC_TRAP_CUTS_ENV: &str = "TY_MCC_DISABLE_CHC_TRAP_CUTS";

/// True iff the trap-cut kill-switch env var is set to a truthy value.
fn trap_cuts_disabled() -> bool {
    std::env::var(DISABLE_CHC_TRAP_CUTS_ENV)
        .ok()
        .is_some_and(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
}

/// Errors raised while building a state-equation CHC system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StateEquationEncoderError {
    /// Coefficient computation wrapped `u64` / `i64`. Returned for COL-net
    /// transitions whose arc weight, scaled by an implicit cardinality
    /// factor, exceeds `i64::MAX`. Must be propagated as `UNKNOWN`.
    CoefficientOverflow {
        transition_index: usize,
        place_index: usize,
        kind: OverflowKind,
    },
    /// Net is too large for the symbolic encoding to be productive.
    NetTooLarge { num_places: usize, limit: usize },
    /// The place×transition product exceeds the dense-encoding budget. The CHC
    /// system is Θ(places × transitions); past this it would OOM. Propagated as
    /// `UNKNOWN` (symbolic lane declines, BFS runs).
    NetTooManyCells { cells: usize, limit: usize },
    /// Initial marking refers to a token count above `i64::MAX`. The CHC
    /// solver operates on signed integers, so any unsigned marking that
    /// does not fit is rejected here rather than silently truncated.
    InitialMarkingOverflow { place_index: usize, tokens: u64 },
    /// Arc weight does not fit in `i64`. Same rationale as
    /// `InitialMarkingOverflow`.
    ArcWeightOverflow {
        transition_index: usize,
        place_index: usize,
        weight: u64,
    },
}

/// Which arithmetic operation overflowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OverflowKind {
    /// Addition of input + output contributions on the same place.
    DeltaAddition,
    /// Conversion of `u64` to signed coefficient.
    SignedConversion,
}

impl fmt::Display for StateEquationEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoefficientOverflow {
                transition_index,
                place_index,
                kind,
            } => write!(
                f,
                "state-equation coefficient overflow at transition {transition_index}, place {place_index} ({kind:?})"
            ),
            Self::NetTooLarge { num_places, limit } => write!(
                f,
                "net too large for symbolic state-equation encoding ({num_places} places > {limit})"
            ),
            Self::NetTooManyCells { cells, limit } => write!(
                f,
                "net too large for symbolic state-equation encoding \
                 ({cells} place×transition cells > {limit})"
            ),
            Self::InitialMarkingOverflow {
                place_index,
                tokens,
            } => write!(
                f,
                "initial marking overflow at place {place_index}: {tokens} > i64::MAX"
            ),
            Self::ArcWeightOverflow {
                transition_index,
                place_index,
                weight,
            } => write!(
                f,
                "arc weight overflow at transition {transition_index}, place {place_index}: {weight} > i64::MAX"
            ),
        }
    }
}

impl std::error::Error for StateEquationEncoderError {}

/// Variable-name shape used by the encoder.
///
/// Centralised so test code and the witness decoder agree on naming.
pub(crate) struct VarNaming;

impl VarNaming {
    pub(crate) fn current(place_idx: usize) -> ChcVar {
        ChcVar::new(format!("se_m_{place_idx}"), ChcSort::Int)
    }
    pub(crate) fn primed(place_idx: usize) -> ChcVar {
        ChcVar::new(format!("se_m_{place_idx}'"), ChcSort::Int)
    }
    pub(crate) fn current_name(place_idx: usize) -> String {
        format!("se_m_{place_idx}")
    }
}

/// Encoder for the Esparza–Melzer state-equation CHC system.
///
/// The encoder borrows its input net for its lifetime so successive
/// `encode_*` calls share the same predicate / variable layout.
pub(crate) struct StateEquationEncoder<'net> {
    net: &'net PetriNet,
}

impl<'net> StateEquationEncoder<'net> {
    pub(crate) fn new(net: &'net PetriNet) -> Self {
        Self { net }
    }

    /// Build the CHC problem for `Init ∧ Trans ⇒ Inv`, plus a query
    /// `Inv(m) ∧ ¬property(m) ⇒ ⊥`.
    pub(crate) fn encode_safety_query(
        &self,
        property: &ResolvedPredicate,
    ) -> Result<ChcProblem, StateEquationEncoderError> {
        let np = self.net.num_places();
        if np > MAX_SYMBOLIC_PLACES {
            return Err(StateEquationEncoderError::NetTooLarge {
                num_places: np,
                limit: MAX_SYMBOLIC_PLACES,
            });
        }
        // Product gate: the encoding is Θ(places × transitions) (every
        // transition clause frames all places), so a net under the place limit
        // can still OOM if it has many transitions. Decline above the cell
        // budget — verdict-preserving (BFS fallback runs).
        let cells = np.saturating_mul(self.net.num_transitions());
        if cells > MAX_SYMBOLIC_CELLS {
            return Err(StateEquationEncoderError::NetTooManyCells {
                cells,
                limit: MAX_SYMBOLIC_CELLS,
            });
        }

        let mut problem = ChcProblem::new();

        let arg_sorts: Vec<ChcSort> = (0..np).map(|_| ChcSort::Int).collect();
        let inv = problem.declare_predicate("SEInv", arg_sorts);

        let current_args: Vec<ChcExpr> = (0..np)
            .map(|p| ChcExpr::var(VarNaming::current(p)))
            .collect();
        let primed_args: Vec<ChcExpr> = (0..np)
            .map(|p| ChcExpr::var(VarNaming::primed(p)))
            .collect();

        // ── Init clause: m == M0 ⇒ Inv(m) ────────────────────────────
        let mut init_conjuncts: Vec<ChcExpr> = Vec::with_capacity(np);
        for p in 0..np {
            let tokens = self.net.initial_marking[p];
            let signed: i64 = i64::try_from(tokens).map_err(|_| {
                StateEquationEncoderError::InitialMarkingOverflow {
                    place_index: p,
                    tokens,
                }
            })?;
            init_conjuncts.push(ChcExpr::eq(
                ChcExpr::var(VarNaming::current(p)),
                ChcExpr::int(signed),
            ));
        }
        problem.add_clause(HornClause::new(
            ClauseBody::constraint(ChcExpr::and_all(init_conjuncts)),
            ClauseHead::Predicate(inv, current_args.clone()),
        ));

        // Non-negativity on every primed variable. Esparza–Melzer's state
        // equation is over ℕ — without this the solver may admit
        // negative-token over-approximations and weaken the inductive
        // invariant search.
        let nonneg_primed: Vec<ChcExpr> = (0..np)
            .map(|p| ChcExpr::ge(ChcExpr::var(VarNaming::primed(p)), ChcExpr::int(0)))
            .collect();

        // ── Transition clauses: one Horn clause per transition ───────
        for (tidx, transition) in self.net.transitions.iter().enumerate() {
            let mut deltas: Vec<i64> = vec![0_i64; np];
            let mut affected: Vec<bool> = vec![false; np];

            for arc in &transition.inputs {
                let p = arc.place.0 as usize;
                let weight = i64::try_from(arc.weight).map_err(|_| {
                    StateEquationEncoderError::ArcWeightOverflow {
                        transition_index: tidx,
                        place_index: p,
                        weight: arc.weight,
                    }
                })?;
                deltas[p] = deltas[p].checked_sub(weight).ok_or(
                    StateEquationEncoderError::CoefficientOverflow {
                        transition_index: tidx,
                        place_index: p,
                        kind: OverflowKind::DeltaAddition,
                    },
                )?;
                affected[p] = true;
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
                deltas[p] = deltas[p].checked_add(weight).ok_or(
                    StateEquationEncoderError::CoefficientOverflow {
                        transition_index: tidx,
                        place_index: p,
                        kind: OverflowKind::DeltaAddition,
                    },
                )?;
                affected[p] = true;
            }

            // Guard: each input place has at least its arc weight.
            let mut guard_conjuncts: Vec<ChcExpr> = Vec::with_capacity(transition.inputs.len());
            for arc in &transition.inputs {
                let p = arc.place.0 as usize;
                let weight = i64::try_from(arc.weight).map_err(|_| {
                    StateEquationEncoderError::ArcWeightOverflow {
                        transition_index: tidx,
                        place_index: p,
                        weight: arc.weight,
                    }
                })?;
                guard_conjuncts.push(ChcExpr::ge(
                    ChcExpr::var(VarNaming::current(p)),
                    ChcExpr::int(weight),
                ));
            }

            // Effect: m'[p] = m[p] + delta[p] (or m'[p] = m[p] when frame).
            let mut effect_conjuncts: Vec<ChcExpr> = Vec::with_capacity(np);
            for p in 0..np {
                let lhs = ChcExpr::var(VarNaming::primed(p));
                let rhs = if !affected[p] || deltas[p] == 0 {
                    ChcExpr::var(VarNaming::current(p))
                } else {
                    ChcExpr::add(ChcExpr::var(VarNaming::current(p)), ChcExpr::int(deltas[p]))
                };
                effect_conjuncts.push(ChcExpr::eq(lhs, rhs));
            }

            let mut all_conjuncts = guard_conjuncts;
            all_conjuncts.extend(effect_conjuncts);
            all_conjuncts.extend(nonneg_primed.iter().cloned());
            let trans_constraint = ChcExpr::and_all(all_conjuncts);

            problem.add_clause(HornClause::new(
                ClauseBody::new(vec![(inv, current_args.clone())], Some(trans_constraint)),
                ClauseHead::Predicate(inv, primed_args.clone()),
            ));
        }

        // Stuttering: Inv(m) ∧ m' = m ⇒ Inv(m'). Helps the solver close
        // out trivially-safe predicates without enumerating every
        // transition.
        let stutter_conjuncts: Vec<ChcExpr> = (0..np)
            .map(|p| {
                ChcExpr::eq(
                    ChcExpr::var(VarNaming::primed(p)),
                    ChcExpr::var(VarNaming::current(p)),
                )
            })
            .collect();
        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, current_args.clone())],
                Some(ChcExpr::and_all(stutter_conjuncts)),
            ),
            ClauseHead::Predicate(inv, primed_args.clone()),
        ));

        // ── Trap-invariant strengthening ─────────────────────────────────
        //
        // An *initially-marked trap* `T` (a set of places that, once marked,
        // can never be emptied by any firing sequence — a standard Petri-net
        // theorem) yields the sound invariant `sum_{p in T} m[p] >= 1` over
        // every reachable marking. `find_initially_marked_traps` is PROVEN
        // SOUND: every returned trap is a real trap, minimized, and gated
        // through `trap_is_initially_marked` (lp_state_equation.rs:831).
        //
        // We inject each trap invariant as a *query-strengthening* conjunct —
        // exactly the mechanism the PDR encoder uses for P-invariants
        // (`pdr_encoding.rs:251-304`, "we add them as additional constraints
        // on the query to strengthen the invariant discovery"). The query
        // becomes `Inv(m) ∧ ¬property(m) ∧ (⋀_T sum_{p∈T} m[p] >= 1) ⇒ ⊥`.
        //
        // SOUNDNESS: a query-side conjunct only *narrows* the set of bad
        // states IC3 must refute — it lets the solver assume the trap
        // invariant when discharging the bad state. Because every genuinely
        // reachable marking satisfies the trap invariant, restricting to
        // trap-satisfying bad states cannot hide a real counterexample: it
        // can only turn a SPURIOUS over-approximation SAT (false UNSAFE) into
        // UNSAT (Safe = AG holds), NEVER a genuine UNSAFE into SAFE. (UNSAFE
        // witnesses are additionally replay-validated on the concrete net by
        // the seeder — reachability_seed.rs:177.) A tautological
        // `Inv(m) ⇒ Inv(m)` consecution clause carries no information to IC3;
        // the query-strengthening form is what actually accelerates it.
        //
        // Without these cuts, ay-chc's IC3/PDR must *rediscover* trap
        // invariants from scratch on large AG-universal nets and frequently
        // times out; TY already computes them, so we hand them over.
        //
        // Size-gated identically to `lp_fireability_truth`
        // (lp_state_equation.rs:588): enumerate traps only when
        // `num_places + num_transitions <= MAX_SYMBOLIC_PLACES`. Above the
        // gate we emit no trap conjuncts — byte-identical to the prior
        // encoding (verdict-preserving; BFS still runs). A kill-switch env var
        // disables the feature entirely for differential A/B comparison.
        let mut trap_constraints: Vec<ChcExpr> = Vec::new();
        if !trap_cuts_disabled()
            && self.net.num_places() + self.net.num_transitions() <= MAX_SYMBOLIC_PLACES
        {
            let traps = crate::lp_state_equation::find_initially_marked_traps(self.net);
            for trap in &traps {
                // Sum over the trap's member places in the *current* state.
                let member_terms: Vec<ChcExpr> = trap
                    .iter()
                    .enumerate()
                    .filter(|&(_p, &in_trap)| in_trap)
                    .map(|(p, &_in_trap)| ChcExpr::var(VarNaming::current(p)))
                    .collect();
                if member_terms.is_empty() {
                    continue;
                }
                let sum = member_terms
                    .into_iter()
                    .reduce(ChcExpr::add)
                    .expect("non-empty by guard");
                trap_constraints.push(ChcExpr::ge(sum, ChcExpr::int(1)));
            }

            // ── Initially-unmarked siphon cuts (dual of the trap cut) ───────
            //
            // Fed forward from the integer state-equation lane
            // (`super::int_state_equation::encode_base_constraints`), which already
            // strengthens its QF_LIA query with these. An initially-UNMARKED siphon
            // `S` (every transition producing into `S` also consumes from `S`) can
            // never gain a token — once empty it stays empty (standard Petri-net
            // theorem) — so `Σ_{p∈S} m[p] = 0` holds in every reachable marking.
            // Emitting the half `Σ_{p∈S} m[p] <= 0` as a query-strengthening
            // conjunct is SOUND for the same reason as the trap cut: it only
            // *narrows* the bad states IC3/PDR must refute (the solver may assume
            // the siphon invariant), so it can turn a SPURIOUS over-approximation
            // SAT (false UNSAFE) into UNSAT (Safe), NEVER a genuine UNSAFE into
            // SAFE. The other half (`m[p] >= 0`) is supplied by the per-place
            // non-negativity already framed on every transition clause, and any
            // genuine UNSAFE witness remains replay-validated on the concrete net.
            // Gated through `find_initially_unmarked_siphons`, which returns ONLY
            // entirely-initially-unmarked siphons (the soundness gate). The
            // existing trap kill-switch (`trap_cuts_disabled`) disables this too, so
            // the A/B baseline stays byte-identical to the bare encoding.
            let siphons = crate::lp_state_equation::find_initially_unmarked_siphons(self.net);
            for siphon in &siphons {
                let member_terms: Vec<ChcExpr> = siphon
                    .iter()
                    .enumerate()
                    .filter(|&(_p, &in_siphon)| in_siphon)
                    .map(|(p, &_in_siphon)| ChcExpr::var(VarNaming::current(p)))
                    .collect();
                if member_terms.is_empty() {
                    continue;
                }
                let sum = member_terms
                    .into_iter()
                    .reduce(ChcExpr::add)
                    .expect("non-empty by guard");
                trap_constraints.push(ChcExpr::le(sum, ChcExpr::int(0)));
            }
        }

        // ── Query clause: Inv(m) ∧ ¬property(m) ∧ traps ∧ siphons ⇒ ⊥ ───
        let negated = ChcExpr::not(encode_predicate_expr(property, self.net)?);
        let mut query_conjuncts = Vec::with_capacity(1 + trap_constraints.len());
        query_conjuncts.push(negated);
        query_conjuncts.extend(trap_constraints);
        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, current_args)],
                Some(ChcExpr::and_all(query_conjuncts)),
            ),
            ClauseHead::False,
        ));

        Ok(problem)
    }
}

/// Encode a [`ResolvedPredicate`] as a CHC expression over current-state
/// variables (`se_m_<idx>`).
pub(crate) fn encode_predicate_expr(
    pred: &ResolvedPredicate,
    net: &PetriNet,
) -> Result<ChcExpr, StateEquationEncoderError> {
    match pred {
        ResolvedPredicate::True => Ok(ChcExpr::Bool(true)),
        ResolvedPredicate::False => Ok(ChcExpr::Bool(false)),
        ResolvedPredicate::And(children) => {
            let mut exprs: Vec<ChcExpr> = Vec::with_capacity(children.len());
            for child in children {
                exprs.push(encode_predicate_expr(child, net)?);
            }
            Ok(ChcExpr::and_all(exprs))
        }
        ResolvedPredicate::Or(children) => {
            let mut exprs: Vec<ChcExpr> = Vec::with_capacity(children.len());
            for child in children {
                exprs.push(encode_predicate_expr(child, net)?);
            }
            Ok(ChcExpr::or_all(exprs))
        }
        ResolvedPredicate::Not(inner) => Ok(ChcExpr::not(encode_predicate_expr(inner, net)?)),
        ResolvedPredicate::IntLe(left, right) => {
            Ok(ChcExpr::le(encode_int_expr(left)?, encode_int_expr(right)?))
        }
        ResolvedPredicate::IsFireable(transitions) => {
            let mut disjuncts: Vec<ChcExpr> = Vec::with_capacity(transitions.len());
            for tidx in transitions {
                let transition_index = tidx.0 as usize;
                let transition = &net.transitions[transition_index];
                if transition.inputs.is_empty() {
                    disjuncts.push(ChcExpr::Bool(true));
                    continue;
                }
                let mut guards: Vec<ChcExpr> = Vec::with_capacity(transition.inputs.len());
                for arc in &transition.inputs {
                    let p = arc.place.0 as usize;
                    let weight = i64::try_from(arc.weight).map_err(|_| {
                        StateEquationEncoderError::ArcWeightOverflow {
                            transition_index,
                            place_index: p,
                            weight: arc.weight,
                        }
                    })?;
                    guards.push(ChcExpr::ge(
                        ChcExpr::var(VarNaming::current(p)),
                        ChcExpr::int(weight),
                    ));
                }
                disjuncts.push(ChcExpr::and_all(guards));
            }
            Ok(ChcExpr::or_all(disjuncts))
        }
    }
}

/// Encode an integer expression over current-state marking variables.
pub(crate) fn encode_int_expr(
    expr: &ResolvedIntExpr,
) -> Result<ChcExpr, StateEquationEncoderError> {
    match expr {
        ResolvedIntExpr::Constant(value) => {
            // ResolvedIntExpr::Constant is `u64`-valued. Convert with
            // checked semantics so a future-widened or COL-derived
            // constant above i64::MAX is rejected here as UNKNOWN
            // rather than silently truncated to a negative integer.
            let signed = i64::try_from(*value).map_err(|_| {
                StateEquationEncoderError::InitialMarkingOverflow {
                    place_index: usize::MAX,
                    tokens: *value,
                }
            })?;
            Ok(ChcExpr::int(signed))
        }
        ResolvedIntExpr::TokensCount(places) => {
            if places.is_empty() {
                return Ok(ChcExpr::int(0));
            }
            let mut terms: Vec<ChcExpr> = places
                .iter()
                .map(|p| ChcExpr::var(VarNaming::current(p.0 as usize)))
                .collect();
            if terms.len() == 1 {
                return Ok(terms.pop().expect("len==1"));
            }
            Ok(terms
                .into_iter()
                .reduce(ChcExpr::add)
                .expect("non-empty by branch guard"))
        }
    }
}

#[cfg(test)]
mod tests {
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

    /// Total number of `ChcExpr::Op` nodes in an expression (structural size).
    fn op_node_count(expr: &ChcExpr) -> usize {
        match expr {
            ChcExpr::Op(_, args) => 1 + args.iter().map(|a| op_node_count(a)).sum::<usize>(),
            _ => 0,
        }
    }

    /// Extract the query clause's body constraint from an encoded problem.
    fn query_constraint(problem: &ChcProblem) -> ChcExpr {
        problem
            .clauses()
            .iter()
            .find(|c| c.head.is_query())
            .and_then(|c| c.body.constraint.clone())
            .expect("safety query clause must carry a constraint")
    }

    /// φ = (tokens(q) ≤ 1) over place index 1 (`q`). Its negation `q > 1` does NOT
    /// interact with a `p0 ≤ 0` siphon cut, so the query constraint SURVIVES the
    /// `simplify_constants` pass (it is not contradictory) and the siphon conjunct
    /// is structurally observable. (A property ON the siphon place would make the
    /// query `false` and the clause would be pruned — the cut working, but
    /// invisible to a node count.)
    fn le_q_one() -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ResolvedIntExpr::Constant(1),
        )
    }

    /// Net with a trap-FREE initially-unmarked siphon `{p0}` plus a place `q`.
    ///
    /// - `t0: p0 → q` (consume p0, produce q). No transition outputs to p0, so
    ///   `{p0}` is a SIPHON (vacuously: no producer to violate the condition); and
    ///   the only transition consuming p0 (t0) does NOT produce into p0, so `{p0}`
    ///   is NOT a trap. Thus when p0 is initially unmarked it yields a siphon cut
    ///   but NO trap cut — isolating the siphon contribution for a clean node
    ///   count. (A self-feeding siphon `p0→2·p0` would also be a trap, confounding
    ///   the count.)
    ///
    /// `p0_tokens` toggles whether `{p0}` is an initially-unmarked siphon.
    fn trap_free_siphon_net(p0_tokens: u64) -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("p0"), place("q")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![p0_tokens, 0],
        }
    }

    /// GATE (CHC siphon cut fed forward) + its soundness guard, in ONE test to
    /// avoid racing on the process-global `DISABLE_CHC_TRAP_CUTS_ENV` kill-switch.
    ///
    /// Part A — the cut is wired: with `p0` an initially-UNMARKED, trap-FREE siphon
    /// the CHC safety query gains a `Σ_{p∈S} m ≤ 0` conjunct — exactly the cut the
    /// integer lane already uses. Proven deterministically: the siphon finder
    /// detects `{p0}`, and the query constraint (over the unrelated place `q`, so
    /// it is not simplified to `false`) has strictly MORE operator nodes with the
    /// cuts enabled than with the kill-switch set. Because the net is trap-free,
    /// the ONLY added structure is the siphon conjunct.
    ///
    /// Part B — the soundness gate: with `p0` initially MARKED there is no
    /// initially-unmarked siphon, and the net is trap-free, so the query is
    /// structurally identical to the bare encoding (no unsound zero-cut on a
    /// marked siphon).
    #[test]
    fn test_chc_query_siphon_cut_wired_and_soundness_gated() {
        // ── Part A: p0 unmarked ⇒ {p0} is a trap-free unmarked siphon ⇒ cut ──
        let unmarked = trap_free_siphon_net(/*p0=*/ 0);
        // Sanity: trap-free (no initially-marked trap to confound the count).
        assert!(
            crate::lp_state_equation::find_initially_marked_traps(&unmarked).is_empty(),
            "gate net must be trap-free so the only added query structure is the siphon"
        );
        let siphons = crate::lp_state_equation::find_initially_unmarked_siphons(&unmarked);
        assert!(
            siphons.iter().any(|s| s.first() == Some(&true)),
            "{{p0}} must be detected as an initially-unmarked siphon; got {siphons:?}"
        );

        let encoder = StateEquationEncoder::new(&unmarked);
        crate::env_guard::remove_var(DISABLE_CHC_TRAP_CUTS_ENV);
        let with_nodes = op_node_count(&query_constraint(
            &encoder.encode_safety_query(&le_q_one()).expect("encode"),
        ));
        crate::env_guard::set_var(DISABLE_CHC_TRAP_CUTS_ENV, "1");
        let bare_nodes = op_node_count(&query_constraint(
            &encoder.encode_safety_query(&le_q_one()).expect("encode"),
        ));
        crate::env_guard::remove_var(DISABLE_CHC_TRAP_CUTS_ENV);
        assert!(
            with_nodes > bare_nodes,
            "siphon cut must add structure to the CHC query (with={with_nodes}, bare={bare_nodes})"
        );

        // ── Part B: p0 marked ⇒ NO unmarked siphon ⇒ NO cut (soundness gate) ──
        let marked = trap_free_siphon_net(/*p0=*/ 1);
        let marked_siphons = crate::lp_state_equation::find_initially_unmarked_siphons(&marked);
        assert!(
            marked_siphons.is_empty(),
            "an initially-MARKED siphon must NOT be returned as a zero-cut: {marked_siphons:?}"
        );
        assert!(
            crate::lp_state_equation::find_initially_marked_traps(&marked).is_empty(),
            "marked gate net must remain trap-free for an exact structural comparison"
        );
        let marked_encoder = StateEquationEncoder::new(&marked);
        crate::env_guard::remove_var(DISABLE_CHC_TRAP_CUTS_ENV);
        let marked_with = op_node_count(&query_constraint(
            &marked_encoder
                .encode_safety_query(&le_q_one())
                .expect("encode"),
        ));
        crate::env_guard::set_var(DISABLE_CHC_TRAP_CUTS_ENV, "1");
        let marked_bare = op_node_count(&query_constraint(
            &marked_encoder
                .encode_safety_query(&le_q_one())
                .expect("encode"),
        ));
        crate::env_guard::remove_var(DISABLE_CHC_TRAP_CUTS_ENV);
        assert_eq!(
            marked_with, marked_bare,
            "no unmarked siphon (and no trap) ⇒ no cut ⇒ query identical to bare encoding"
        );
    }
}
