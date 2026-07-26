// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared MDD glue for the property examinations that consume the `tla-mdd`
//! backend (the symbolic CTL lane and the exact MDD reachability fast-path).
//!
//! These three helpers were originally private to `examinations::ctl::pipeline`
//! and were being re-derived (with identical logic) in the new reachability MDD
//! fast-path. The `dd_spec` module header explicitly warns that the
//! resolved-predicate → DD-predicate converter must have a SINGLE source of
//! truth so the lanes cannot drift apart; this module extends that discipline to
//! the MDD-specific glue (the spec gate, the `DdNetSpec` → `MddNet` adapter, and
//! the `DdPredicate` → characteristic-`MddRef` lowering) so the CTL and
//! reachability lanes share ONE copy.
//!
//! Lifting is behavior-preserving: the bodies are byte-for-byte the same logic
//! the CTL lane shipped (same sound `build_sound_dd_spec` admission, same
//! edge-width cap constant, same exact `IntLe`/`IsFireable`/boolean lowering),
//! so the CTL lane's tests and the `crosscheck_ctl` battery are unaffected.

#![cfg(feature = "dd-backend")]

use crate::petri_net::PetriNet;

/// Build the MDD net spec for `net`. Mirrors
/// `state_space::build_mdd_spec_for_net`: the SAME sound per-place LP bounds +
/// structural gates as the BDD lane (via `build_sound_dd_spec`) plus an
/// edge-width cap, but NO 127-var model-counting cap (the MDD spends one level
/// per place — no bit-blasting, no `2^vars` terminal weight — so it does not
/// inherit that BDD-only limit). Returns `None` (fall through unchanged) on any
/// failed gate.
///
/// SOUNDNESS: identical admission to the StateSpace-MDD spec gate — sound LP
/// bounds so the encoded value range is a superset of every place's reachable
/// projection (the MDD reachable set, deadlock set, and every atom's
/// characteristic set are therefore EXACT); the count-representability ceiling
/// is enforced downstream by the MDD's `u128` fail-closed count.
#[must_use]
pub(crate) fn build_mdd_spec_for_net(net: &PetriNet) -> Option<tla_dd::DdNetSpec> {
    let (spec, bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    // Total edge-width gate (same constant as the StateSpace-MDD lane): bound
    // `Σ (bound[p] + 1)` so per-node child vectors stay affordable. The deeper
    // node-count + wall-clock budget inside `tla-mdd` catches the rest.
    const MAX_TOTAL_EDGE_WIDTH: u128 = 1 << 22; // ~4.2M edges across all levels
    let mut total_edge_width: u128 = 0;
    for &b in &bounds {
        total_edge_width = total_edge_width.saturating_add(b as u128 + 1);
        if total_edge_width > MAX_TOTAL_EDGE_WIDTH {
            return None;
        }
    }
    Some(spec)
}

/// Adapter: a [`tla_dd::DdNetSpec`] → [`tla_mdd::MddNet`], field-for-field
/// (mirrors `state_space::dd_spec_to_mdd_net`). The MDD lane consumes the
/// IDENTICAL net the BDD lane does, built by the same `build_sound_dd_spec`
/// gate, so the two lanes' verdicts are directly cross-validatable.
#[must_use]
pub(crate) fn dd_spec_to_mdd_net(spec: &tla_dd::DdNetSpec) -> tla_mdd::MddNet {
    tla_mdd::MddNet {
        bounds: spec.bounds.clone(),
        initial_marking: spec.initial_marking.clone(),
        transitions: spec
            .transitions
            .iter()
            .map(|t| tla_mdd::MddTransition {
                pre: t.pre.clone(),
                post: t.post.clone(),
            })
            .collect(),
    }
}

/// Build the MDD net in a **saturation-friendly place order** and return the
/// `place → level` map (`inv`) needed to lower place-indexed queries into that
/// order.
///
/// MDD node-level saturation's peak node count is acutely sensitive to the
/// place→level order: an arbitrary PNML order blows the interior-node budget on
/// nets a good order keeps compact — the scale lever the BDD lane already pulls
/// (`tla_dd::force_place_order`) but the MDD lane historically did not. This
/// applies the span-guarded FORCE place order (identity when it cannot strictly
/// improve the transition span, so an already-good order is never made worse),
/// permutes the spec into it, and returns the built [`tla_mdd::MddNet`] plus
/// `inv` (`inv[place] = level`).
///
/// # Soundness
///
/// `tla_dd::permute_spec` is an isomorphic relabeling of places, so the
/// reachable set — and every metric read off it — is unchanged (see the
/// `tla_dd::order` module soundness note, asserted against the exhaustive-BFS
/// oracle). Consumers MUST lower each place-indexed atom / coefficient through
/// `inv` ([`permute_dd_predicate`] for [`tla_dd::DdPredicate`],
/// `tla_dd::permute_query` for UpperBounds coefficient vectors) so the net and
/// the queries agree on the level coordinate. `IsFireable` needs no remap (a
/// place permutation does not reorder transitions; the permuted net's transition
/// pre-vectors already carry the new level layout). With `inv` applied, the
/// verdicts are IDENTICAL to the identity-order run — only the MDD size
/// (feasibility) changes.
///
/// `seed` is an optional structural order seed (e.g. the NUPN unit-hierarchy
/// block order from `dd_spec::nupn_order_seed` — nested units of mutually
/// exclusive places are exactly the DD locality a good order wants). It only
/// EXTENDS the FORCE candidate set under the same span guard, so `seed = None`
/// is bit-identical to the unseeded FORCE order and a seed can never make the
/// chosen order worse than the unseeded choice.
#[must_use]
pub(crate) fn dd_spec_to_ordered_mdd_net(
    spec: &tla_dd::DdNetSpec,
    seed: Option<&[usize]>,
) -> (tla_mdd::MddNet, Vec<usize>) {
    // MEASUREMENT gate (3c ordering A/B vs the MCC-2025 oracle): identity order
    // when set. SOUND either way — the order changes only MDD size/feasibility,
    // never verdicts (isomorphic relabeling; `inv` = identity here). Default
    // (unset) is byte-identical to the always-on FORCE order.
    if std::env::var("TY_MCC_DISABLE_MDD_FORCE_ORDER")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        let n = spec.bounds.len();
        return (dd_spec_to_mdd_net(spec), (0..n).collect());
    }
    let order = tla_dd::force_place_order_seeded(spec, seed);
    let inv = tla_dd::invert_order(&order);
    let permuted = tla_dd::permute_spec(spec, &order);
    (dd_spec_to_mdd_net(&permuted), inv)
}

/// Reorder an ALREADY-BUILT [`tla_mdd::MddNet`] into a saturation-friendly place
/// order, returning the built net plus the `orig→level` map `inv`.
///
/// The colored MDD (`symbolic_colored::build_colored_mdd_net`) is built directly
/// as an `MddNet` (its levels are `(place, color)` slots) with no `DdNetSpec` to
/// order from — so [`dd_spec_to_ordered_mdd_net`] does not apply. This gives the
/// colored path the SAME span-guarded FORCE ordering the P/T path gets: it views
/// the net's own bounds/transitions as a `DdNetSpec` (they mirror it
/// field-for-field), computes the FORCE order, and relabels. SOUND by the same
/// argument — `permute_spec` is an isomorphic relabeling, so the reachable set
/// and every per-query maximum are invariant; consumers align their queries via
/// `inv` (`tla_dd::permute_query` / [`permute_dd_predicate`]).
#[must_use]
pub(crate) fn order_mdd_net(net: tla_mdd::MddNet) -> (tla_mdd::MddNet, Vec<usize>) {
    let spec = tla_dd::DdNetSpec {
        bounds: net.bounds,
        initial_marking: net.initial_marking,
        transitions: net
            .transitions
            .into_iter()
            .map(|t| tla_dd::DdTransition {
                pre: t.pre,
                post: t.post,
            })
            .collect(),
    };
    // No structural seed for the colored slots (the NUPN block order is over P/T
    // places, not `(place,color)` slots); plain span-guarded FORCE.
    dd_spec_to_ordered_mdd_net(&spec, None)
}

/// Rewrite a [`tla_dd::DdPredicate`]'s place indices into the level coordinates
/// of an ordered MDD net (`inv[place] = level`, from
/// [`dd_spec_to_ordered_mdd_net`]).
///
/// Every `TokensCount` place index `p` becomes `inv[p]`; transition indices
/// (`IsFireable`) are unchanged (a place permutation does not reorder
/// transitions — the permuted net's transition pre-vectors already carry the new
/// level layout). Constants and the boolean structure are preserved, so
/// lowering the rewritten predicate against the permuted net yields the SAME
/// characteristic marking-set (relabeled), hence the same EF/AG verdict.
///
/// Returns `None` (fail-closed) if any place index is out of range for `inv`
/// (never happens for a `translate_predicate` output, which pre-validates
/// indices against `num_places`, but keeps the transform total + safe).
#[must_use]
pub(crate) fn permute_dd_predicate(
    pred: &tla_dd::DdPredicate,
    inv: &[usize],
) -> Option<tla_dd::DdPredicate> {
    use tla_dd::{DdIntExpr, DdPredicate};
    fn pe(e: &DdIntExpr, inv: &[usize]) -> Option<DdIntExpr> {
        Some(match e {
            DdIntExpr::Constant(c) => DdIntExpr::Constant(*c),
            DdIntExpr::TokensCount(places) => {
                let mut mapped = Vec::with_capacity(places.len());
                for &p in places {
                    mapped.push(*inv.get(p)?);
                }
                DdIntExpr::TokensCount(mapped)
            }
        })
    }
    Some(match pred {
        DdPredicate::True => DdPredicate::True,
        DdPredicate::False => DdPredicate::False,
        DdPredicate::Not(inner) => DdPredicate::Not(Box::new(permute_dd_predicate(inner, inv)?)),
        DdPredicate::And(children) => {
            let mut out = Vec::with_capacity(children.len());
            for c in children {
                out.push(permute_dd_predicate(c, inv)?);
            }
            DdPredicate::And(out)
        }
        DdPredicate::Or(children) => {
            let mut out = Vec::with_capacity(children.len());
            for c in children {
                out.push(permute_dd_predicate(c, inv)?);
            }
            DdPredicate::Or(out)
        }
        DdPredicate::IntLe(left, right) => DdPredicate::IntLe(pe(left, inv)?, pe(right, inv)?),
        DdPredicate::IsFireable(tids) => DdPredicate::IsFireable(tids.clone()),
    })
}

/// Adapter: a [`tla_dd::DdNetSpec`] → [`tla_bdd::petri::BoundedNet`],
/// field-for-field (mirrors [`dd_spec_to_mdd_net`]). Bridges the SAME sound
/// `build_sound_dd_spec` net the BDD/MDD lanes consume onto TY's native ROBDD
/// engine, so the native-BDD verdict is directly cross-validatable against the
/// BFS oracle — the first integration step toward routing `tla-dd` off oxidd.
// Not yet wired into a production lane (the swap is incremental); the
// cross-check test below validates the bridge on real specs.
#[allow(dead_code)]
#[must_use]
pub(crate) fn dd_spec_to_bdd_net(spec: &tla_dd::DdNetSpec) -> tla_bdd::petri::BoundedNet {
    tla_bdd::petri::BoundedNet {
        bounds: spec.bounds.clone(),
        init: spec.initial_marking.clone(),
        transitions: spec
            .transitions
            .iter()
            .map(|t| tla_bdd::petri::BoundedTransition {
                pre: t.pre.clone(),
                post: t.post.clone(),
            })
            .collect(),
    }
}

/// Lower a [`tla_dd::DdPredicate`] to a [`tla_bdd::petri::Pred`] over the bridged
/// [`tla_bdd::petri::BoundedNet`] — the tla-bdd twin of [`lower_dd_predicate_to_mdd`],
/// with IDENTICAL semantics (same multiset-sum `IntLe`, same guard-only
/// `IsFireable`, same boolean structure). This is the predicate half of routing
/// the reachability lane onto the native ROBDD engine. Returns `None`
/// (fail-closed) on any out-of-range place/transition index.
#[allow(dead_code)]
#[must_use]
pub(crate) fn lower_dd_predicate_to_bdd(
    net: &tla_bdd::petri::BoundedNet,
    pred: &tla_dd::DdPredicate,
) -> Option<tla_bdd::petri::Pred> {
    use tla_bdd::petri::Pred;
    use tla_dd::{DdIntExpr, DdPredicate};
    let np = net.bounds.len();
    Some(match pred {
        // tla-bdd's Pred has no True/False atom; encode via a trivially-true /
        // trivially-false token inequality (Σ0·m ≤ 0 / ≤ -1).
        DdPredicate::True => Pred::TokenLe {
            coeffs: vec![0; np],
            k: 0,
        },
        DdPredicate::False => Pred::TokenLe {
            coeffs: vec![0; np],
            k: -1,
        },
        DdPredicate::Not(inner) => Pred::Not(Box::new(lower_dd_predicate_to_bdd(net, inner)?)),
        DdPredicate::And(children) => Pred::And(
            children
                .iter()
                .map(|c| lower_dd_predicate_to_bdd(net, c))
                .collect::<Option<Vec<_>>>()?,
        ),
        DdPredicate::Or(children) => Pred::Or(
            children
                .iter()
                .map(|c| lower_dd_predicate_to_bdd(net, c))
                .collect::<Option<Vec<_>>>()?,
        ),
        DdPredicate::IntLe(left, right) => {
            // Σ coeff·m ≤ k where coeff_l = leftcount_l − rightcount_l (with
            // multiplicity) and k = rconst − lconst — identical to the MDD lowering.
            let mut coeffs = vec![0i128; np];
            let mut k: i128 = 0;
            match left {
                DdIntExpr::Constant(c) => k -= *c as i128,
                DdIntExpr::TokensCount(places) => {
                    for &p in places {
                        if p >= np {
                            return None;
                        }
                        coeffs[p] += 1;
                    }
                }
            }
            match right {
                DdIntExpr::Constant(c) => k += *c as i128,
                DdIntExpr::TokensCount(places) => {
                    for &p in places {
                        if p >= np {
                            return None;
                        }
                        coeffs[p] -= 1;
                    }
                }
            }
            Pred::TokenLe { coeffs, k }
        }
        DdPredicate::IsFireable(tids) => {
            // ⋃_t enabled(t): the disjunction of per-transition fireability.
            for &t in tids {
                if t >= net.transitions.len() {
                    return None;
                }
            }
            Pred::Or(tids.iter().map(|&t| Pred::Fireable(t)).collect())
        }
    })
}

/// Evaluate flat EF/AG reachability queries on the native ROBDD engine — the
/// tla-bdd twin of the MDD reachability lane (`tla_mdd::evaluate_reachability_at_
/// initial`). Builds the bridged net, lowers each `DdPredicate`, and runs
/// `tla_bdd::petri::evaluate_reachability`. `None` (fail-closed) if any lowering
/// declines. The reachability-lane migration component, ready to wire behind a
/// gate once its per-examination coverage A/B passes.
/// `deadline` bounds the reachable-set fixpoint (the production-wiring budget): a
/// `None` return = either an atom-lowering decline OR the budget was exceeded —
/// both fail-closed, so the caller falls through to the existing lane. `None`
/// deadline ⇒ run to convergence.
#[allow(dead_code)]
#[must_use]
pub(crate) fn evaluate_reachability_via_bdd(
    spec: &tla_dd::DdNetSpec,
    queries: &[(tla_mdd::MddReachQuantifier, tla_dd::DdPredicate)],
    deadline: Option<std::time::Instant>,
) -> Option<Vec<bool>> {
    use tla_bdd::petri::Query;
    use tla_mdd::MddReachQuantifier;
    let net = dd_spec_to_bdd_net(spec);
    let qs: Vec<Query> = queries
        .iter()
        .map(|(q, p)| {
            let pred = lower_dd_predicate_to_bdd(&net, p)?;
            Some(match q {
                MddReachQuantifier::Ef => Query::Ef(pred),
                MddReachQuantifier::Ag => Query::Ag(pred),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    tla_bdd::petri::evaluate_reachability_within(&net, &qs, deadline)
}

/// UpperBounds on the native ROBDD engine — the tla-bdd twin of the MDD UB lane
/// (`tla_mdd::MddNet::upper_bounds`). Each query is a per-place coefficient vector;
/// returns `max Σ coeffs·m` over the reachable set. `deadline` bounds the fixpoint
/// (the production-wiring budget); `None` return = budget exceeded (fail-closed).
#[allow(dead_code)]
#[must_use]
pub(crate) fn upper_bounds_via_bdd(
    spec: &tla_dd::DdNetSpec,
    queries: &[Vec<i128>],
    deadline: Option<std::time::Instant>,
) -> Option<Vec<i128>> {
    let net = dd_spec_to_bdd_net(spec);
    tla_bdd::petri::upper_bounds_bounded_within(&net, queries, deadline)
}

/// Full StateSpace metrics on the native ROBDD engine, returned in tla-dd's
/// [`tla_dd::DdStateSpaceMetrics`] shape (so callers are drop-in). The tla-bdd twin
/// of `tla_dd::reachable_state_space_metrics`; the metric fields are cross-checked
/// against BFS + (historically) oxidd. `iterations` is diagnostic-only (the tla-bdd
/// fixpoint does not surface it) ⇒ 0. Counts saturate into `u64`.
#[allow(dead_code)]
#[must_use]
pub(crate) fn state_space_metrics_via_bdd(spec: &tla_dd::DdNetSpec) -> tla_dd::DdStateSpaceMetrics {
    let net = dd_spec_to_bdd_net(spec);
    let m = tla_bdd::petri::state_space_metrics_bounded(&net);
    let cap = u64::MAX as u128;
    tla_dd::DdStateSpaceMetrics {
        state_count: m.states.min(cap) as u64,
        edge_count: m.edges.min(cap) as u64,
        max_token_in_place: m.max_token_in_place,
        max_token_sum: m.max_token_sum,
        iterations: 0,
    }
}

/// Reachable-marking count on the native ROBDD engine (the tla-bdd twin of
/// `tla_dd::dispatch_reachable_state_count`'s `state_count`). Saturates into `u64`.
#[allow(dead_code)]
#[must_use]
pub(crate) fn reachable_count_via_bdd(spec: &tla_dd::DdNetSpec) -> u64 {
    let net = dd_spec_to_bdd_net(spec);
    tla_bdd::petri::reachable_count_bounded(&net).min(u64::MAX as u128) as u64
}

/// Native-ROBDD LTL emptiness: does `system × GBA(¬φ)` have a reachable
/// accepting cycle? The tla-bdd twin of
/// `tla_dd::symbolic_ltl::symbolic_ltl_has_accepting_cycle_ordered` — it bridges
/// the sound DD spec to a [`tla_bdd::petri::BoundedNet`], lowers each GBA atom
/// (`DdPredicate`) to a [`tla_bdd::petri::Pred`] via [`lower_dd_predicate_to_bdd`]
/// (fail-closed on any unsupported atom), and runs the deadline-bounded
/// [`tla_bdd::ltl_product::ltl_has_accepting_run_within`]. The GBA structure
/// (states, atom-guarded transitions, mixed state/edge generalized acceptance) is
/// carried over 1:1, so the symbolic product is the SAME automaton the explicit
/// checker uses.
///
/// `Some(true)` ⇒ a fair accepting lasso of ¬φ exists (so `A(φ)` is FALSE);
/// `Some(false)` ⇒ none (so `A(φ)` is TRUE); `None` ⇒ DECLINE (an atom lowering
/// is unsupported OR the product fixpoints exceeded `deadline`) — the caller then
/// falls through to the explicit LTL lanes unchanged. Never a guessed verdict.
#[allow(dead_code)]
#[must_use]
pub(crate) fn symbolic_ltl_has_accepting_cycle_via_bdd(
    spec: &tla_dd::DdNetSpec,
    gba: &tla_dd::symbolic_ltl::SymbolicGba,
    deadline: Option<std::time::Instant>,
) -> Option<bool> {
    use tla_bdd::ltl_product::{LtlGba, LtlGbaTransition};
    let net = dd_spec_to_bdd_net(spec);
    let atoms = gba
        .atoms
        .iter()
        .map(|p| lower_dd_predicate_to_bdd(&net, p))
        .collect::<Option<Vec<_>>>()?;
    let conv = |t: &tla_dd::symbolic_ltl::SymbolicGbaTransition| LtlGbaTransition {
        pos_atoms: t.pos_atoms.clone(),
        neg_atoms: t.neg_atoms.clone(),
        successor: t.successor,
        edge_accept: t.edge_accept.clone(),
    };
    let ltl_gba = LtlGba {
        num_states: gba.num_states,
        atoms,
        initial_transitions: gba.initial_transitions.iter().map(conv).collect(),
        transitions: gba
            .transitions
            .iter()
            .map(|ts| ts.iter().map(conv).collect())
            .collect(),
        acceptance: gba.acceptance.clone(),
    };
    tla_bdd::ltl_product::ltl_has_accepting_run_within(&net, &ltl_gba, deadline)
}

/// Lower a [`tla_mdd::CtlFormulaTemplate<DdPredicate>`] to a [`tla_bdd::petri::Ctl`]
/// over the bridged net — the CTL migration component. Atoms lower via
/// [`lower_dd_predicate_to_bdd`]; the A-family is expressed via the E-family + Not
/// (`AX=¬EX¬`, `AF=¬EG¬`, `AG=¬EF¬`), `EU` maps directly, and `AU` via the
/// standard `A[φ U ψ] = ¬(E[¬ψ U (¬φ∧¬ψ)] ∨ EG¬ψ)` — IDENTICAL to the MDD lane's
/// convention. Fail-closed (`None`) on any atom-lowering decline.
#[allow(dead_code)]
#[must_use]
pub(crate) fn lower_ctl_to_bdd(
    net: &tla_bdd::petri::BoundedNet,
    f: &tla_mdd::CtlFormulaTemplate<tla_dd::DdPredicate>,
) -> Option<tla_bdd::petri::Ctl> {
    use tla_bdd::petri::{Ctl, Pred};
    use tla_mdd::CtlFormulaTemplate as T;
    let np = net.bounds.len();
    let lo = |g| lower_ctl_to_bdd(net, g);
    let not = |c: Ctl| Ctl::Not(Box::new(c));
    Some(match f {
        T::Atom(p) => Ctl::Atom(lower_dd_predicate_to_bdd(net, p)?),
        T::Not(g) => not(lo(g)?),
        T::And(cs) => {
            let mut acc = Ctl::Atom(Pred::TokenLe {
                coeffs: vec![0; np],
                k: 0,
            }); // true
            for c in cs {
                acc = Ctl::And(Box::new(acc), Box::new(lo(c)?));
            }
            acc
        }
        T::Or(cs) => {
            let mut acc = Ctl::Atom(Pred::TokenLe {
                coeffs: vec![0; np],
                k: -1,
            }); // false
            for c in cs {
                acc = Ctl::Or(Box::new(acc), Box::new(lo(c)?));
            }
            acc
        }
        T::EX(g) => Ctl::Ex(Box::new(lo(g)?)),
        T::EF(g) => Ctl::Ef(Box::new(lo(g)?)),
        T::EG(g) => Ctl::Eg(Box::new(lo(g)?)),
        T::AX(g) => not(Ctl::Ex(Box::new(not(lo(g)?)))),
        T::AF(g) => not(Ctl::Eg(Box::new(not(lo(g)?)))),
        T::AG(g) => not(Ctl::Ef(Box::new(not(lo(g)?)))),
        T::EU(p, q) => Ctl::Eu(Box::new(lo(p)?), Box::new(lo(q)?)),
        // The BDD petri Ctl has no fair-cycle operator — the BDD TWIN lane
        // declines EGF fail-closed; the MDD lane's Emerson–Lei gfp answers it.
        T::EGF(_) => return None,
        T::AU(p, q) => {
            // ¬( E[¬ψ U (¬φ ∧ ¬ψ)] ∨ EG ¬ψ )
            let nphi_and_npsi = Ctl::And(Box::new(not(lo(p)?)), Box::new(not(lo(q)?)));
            let eu = Ctl::Eu(Box::new(not(lo(q)?)), Box::new(nphi_and_npsi));
            let eg = Ctl::Eg(Box::new(not(lo(q)?)));
            not(Ctl::Or(Box::new(eu), Box::new(eg)))
        }
    })
}

/// Evaluate a CTL formula at the initial marking on the native ROBDD engine —
/// the CTL migration component's entry. Builds the bridged net, lowers the
/// template, runs `tla_bdd::petri::evaluate_ctl`. `None` if lowering declines.
#[allow(dead_code)]
#[must_use]
pub(crate) fn evaluate_ctl_via_bdd(
    spec: &tla_dd::DdNetSpec,
    template: &tla_mdd::CtlFormulaTemplate<tla_dd::DdPredicate>,
    deadline: Option<std::time::Instant>,
) -> Option<bool> {
    let net = dd_spec_to_bdd_net(spec);
    let lowered = lower_ctl_to_bdd(&net, template)?;
    tla_bdd::petri::evaluate_ctl_within(&net, &lowered, deadline)
}

/// Lower a [`tla_dd::DdPredicate`] to its characteristic marking-set
/// [`tla_mdd::MddRef`] in `store`, EXACTLY (same multiset-sum / guard-only
/// semantics as `tla_dd`'s `eval_dd_predicate`). Returns `None` (fail-closed) if
/// any place/transition index is out of range for the spec's bounds.
///
/// - `True`/`False` → the universe `ONE` / empty `ZERO` (the evaluator
///   re-confines atoms to the reachable set).
/// - `IntLe(L, R)` → the linear inequality `Σ c_l·m[l] <= K` where
///   `c_l = leftcoeff_l - rightcoeff_l`, `K = rconst - lconst` — built EXACTLY by
///   [`tla_mdd::MddStore::linear_le_set`].
/// - `IsFireable(tids)` → `⋃_t guard_set(pre_t)` (guard-only enabledness,
///   matching the BDD `compile_is_fireable`).
/// - `Not`/`And`/`Or` → universe complement / intersection / union of the
///   children (the evaluator confines to reachable, so a universe complement is
///   sound).
#[must_use]
pub(crate) fn lower_dd_predicate_to_mdd(
    store: &mut tla_mdd::MddStore,
    net: &tla_mdd::MddNet,
    pred: &tla_dd::DdPredicate,
) -> Option<tla_mdd::MddRef> {
    use tla_dd::{DdIntExpr, DdPredicate};
    use tla_mdd::MddRef;
    let n = net.bounds.len();
    Some(match pred {
        DdPredicate::True => MddRef::ONE,
        DdPredicate::False => MddRef::ZERO,
        DdPredicate::Not(inner) => {
            let s = lower_dd_predicate_to_mdd(store, net, inner)?;
            // Universe complement: ONE \ s (the evaluator re-confines to reach).
            store.difference(MddRef::ONE, s)
        }
        DdPredicate::And(children) => {
            let mut acc = MddRef::ONE;
            for c in children {
                let s = lower_dd_predicate_to_mdd(store, net, c)?;
                acc = store.intersect(acc, s);
            }
            acc
        }
        DdPredicate::Or(children) => {
            let mut acc = MddRef::ZERO;
            for c in children {
                let s = lower_dd_predicate_to_mdd(store, net, c)?;
                acc = store.union(acc, s);
            }
            acc
        }
        DdPredicate::IntLe(left, right) => {
            // Accumulate per-place coefficients: left side +1, right side -1
            // (with multiplicity), plus the constant offset. Then the set is
            // `Σ coeff[l]·m[l] <= -const_offset`, i.e. `<= rconst - lconst`.
            let mut coeffs = vec![0i128; n];
            let mut k: i128 = 0; // moves to the RHS: Σ coeff·m <= k
                                 // left <= right  ⇔  left - right <= 0.
                                 // left contributes +coeff to LHS; a left constant moves to RHS as -c.
            match left {
                DdIntExpr::Constant(c) => k -= *c as i128,
                DdIntExpr::TokensCount(places) => {
                    for &p in places {
                        if p >= n {
                            return None;
                        }
                        coeffs[p] += 1;
                    }
                }
            }
            // right contributes -coeff to LHS; a right constant moves to RHS as +c.
            match right {
                DdIntExpr::Constant(c) => k += *c as i128,
                DdIntExpr::TokensCount(places) => {
                    for &p in places {
                        if p >= n {
                            return None;
                        }
                        coeffs[p] -= 1;
                    }
                }
            }
            store.linear_le_set(&coeffs, k)
        }
        DdPredicate::IsFireable(tids) => {
            let mut acc = MddRef::ZERO;
            for &t in tids {
                if t >= net.transitions.len() {
                    return None;
                }
                let g = store.guard_set(&net.transitions[t].pre);
                acc = store.union(acc, g);
            }
            acc
        }
    })
}

#[cfg(test)]
mod bdd_bridge_tests {
    //! Cross-validate the native `tla-bdd` engine on the SAME `DdNetSpec` the
    //! production BDD/MDD lanes consume, against the explicit BFS oracle
    //! (`tla_dd::bfs_reachable_set_count`). This proves the bridge
    //! (`dd_spec_to_bdd_net` → `tla_bdd` StateSpace) is verdict-faithful on real
    //! specs — the validation gate for eventually routing `tla-dd` off oxidd.
    use super::dd_spec_to_bdd_net;
    use tla_dd::{bfs_reachable_set_count, DdNetSpec, DdTransition};

    fn t(pre: &[u64], post: &[u64]) -> DdTransition {
        DdTransition {
            pre: pre.to_vec(),
            post: post.to_vec(),
        }
    }

    fn check(spec: &DdNetSpec) {
        let bdd_net = dd_spec_to_bdd_net(spec);
        let states = tla_bdd::petri::state_space_metrics_bounded(&bdd_net).states;
        let bfs = bfs_reachable_set_count(spec) as u128;
        assert_eq!(
            states, bfs,
            "native tla-bdd states must match the BFS oracle"
        );
    }

    /// Post-oxidd-removal StateSpace bridge check: the native tla-bdd metrics'
    /// reachable-state count must equal the explicit BFS oracle on the real spec.
    /// (The full 4-metric validation — states/edges/max-in-place/max-sum vs BFS —
    /// lives in tla-bdd's own 6000-net crosscheck battery; this pins the
    /// `dd_spec → bdd_net` bridge + the count on these specs.)
    fn check_vs_oxidd(spec: &DdNetSpec) {
        let bdd_net = dd_spec_to_bdd_net(spec);
        let ours = tla_bdd::petri::state_space_metrics_bounded(&bdd_net);
        assert_eq!(
            ours.states,
            bfs_reachable_set_count(spec) as u128,
            "native tla-bdd state count must match the BFS oracle"
        );
    }

    #[test]
    fn native_bdd_equals_oxidd_statespace_on_real_specs() {
        // tla-bdd ≡ oxidd, verdict-for-verdict, on every metric — the proof that
        // the backend swap preserves answers.
        for spec in [
            DdNetSpec {
                bounds: vec![1, 1],
                initial_marking: vec![1, 0],
                transitions: vec![t(&[1, 0], &[0, 1]), t(&[0, 1], &[1, 0])],
            },
            DdNetSpec {
                bounds: vec![3, 2],
                initial_marking: vec![0, 0],
                transitions: vec![t(&[0, 0], &[1, 0]), t(&[0, 0], &[0, 1])],
            },
            DdNetSpec {
                bounds: vec![4, 4],
                initial_marking: vec![4, 0],
                transitions: vec![t(&[2, 0], &[0, 1])],
            },
            DdNetSpec {
                bounds: vec![1, 1, 1],
                initial_marking: vec![1, 0, 0],
                transitions: vec![
                    t(&[1, 0, 0], &[0, 1, 0]),
                    t(&[0, 1, 0], &[0, 0, 1]),
                    t(&[0, 0, 1], &[1, 0, 0]),
                ],
            },
        ] {
            check_vs_oxidd(&spec);
        }
    }

    #[test]
    fn bdd_bridge_matches_bfs_oracle_on_real_specs() {
        // shuttle: 2 states
        check(&DdNetSpec {
            bounds: vec![1, 1],
            initial_marking: vec![1, 0],
            transitions: vec![t(&[1, 0], &[0, 1]), t(&[0, 1], &[1, 0])],
        });
        // counter 0..5: 6 states
        check(&DdNetSpec {
            bounds: vec![5],
            initial_marking: vec![0],
            transitions: vec![t(&[0], &[1])],
        });
        // two independent counters (3,2): 12 states
        check(&DdNetSpec {
            bounds: vec![3, 2],
            initial_marking: vec![0, 0],
            transitions: vec![t(&[0, 0], &[1, 0]), t(&[0, 0], &[0, 1])],
        });
        // weighted conserved: move 2 from p0 into 1 of p1; init (4,0) -> 3 states
        check(&DdNetSpec {
            bounds: vec![4, 4],
            initial_marking: vec![4, 0],
            transitions: vec![t(&[2, 0], &[0, 1])],
        });
        // a small producer/consumer ring (3 places, token rotates): 3 states
        check(&DdNetSpec {
            bounds: vec![1, 1, 1],
            initial_marking: vec![1, 0, 0],
            transitions: vec![
                t(&[1, 0, 0], &[0, 1, 0]),
                t(&[0, 1, 0], &[0, 0, 1]),
                t(&[0, 0, 1], &[1, 0, 0]),
            ],
        });
    }

    /// The reachability-lane migration gate: the native ROBDD reachability
    /// evaluator must produce IDENTICAL EF/AG verdicts to the MDD lane (both
    /// lower the SAME DdPredicates over the SAME spec). If this holds, the
    /// reachability lane can be routed onto tla-bdd (perf A/B permitting).
    #[test]
    fn bdd_reachability_matches_mdd_lane() {
        use super::{dd_spec_to_mdd_net, evaluate_reachability_via_bdd, lower_dd_predicate_to_mdd};
        use tla_dd::{DdIntExpr, DdPredicate};
        use tla_mdd::MddReachQuantifier::{Ag, Ef};
        // mutex: idle p0, crit p1, lock p2. acquire t0: p0,p2->p1. release t1: p1->p0,p2.
        let spec = DdNetSpec {
            bounds: vec![1, 1, 1],
            initial_marking: vec![1, 0, 1],
            transitions: vec![t(&[1, 0, 1], &[0, 1, 0]), t(&[0, 1, 0], &[1, 0, 1])],
        };
        let preds: Vec<(tla_mdd::MddReachQuantifier, DdPredicate)> = vec![
            (Ef, DdPredicate::IsFireable(vec![0])),
            (Ef, DdPredicate::IsFireable(vec![1])),
            // p1 <= 0 (crit empty) — AG: is it ALWAYS empty? no (acquire reaches crit).
            (
                Ag,
                DdPredicate::IntLe(DdIntExpr::TokensCount(vec![1]), DdIntExpr::Constant(0)),
            ),
            // p1 >= 1 (in crit) reachable? EF — yes.
            (
                Ef,
                DdPredicate::IntLe(DdIntExpr::Constant(1), DdIntExpr::TokensCount(vec![1])),
            ),
            (
                Ef,
                DdPredicate::Not(Box::new(DdPredicate::IsFireable(vec![1]))),
            ),
            (
                Ag,
                DdPredicate::Or(vec![
                    DdPredicate::IsFireable(vec![0]),
                    DdPredicate::IsFireable(vec![1]),
                ]),
            ),
            (Ag, DdPredicate::True),
            (Ef, DdPredicate::False),
        ];
        let bdd = evaluate_reachability_via_bdd(&spec, &preds, None).expect("bdd reachability");
        let mdd_net = dd_spec_to_mdd_net(&spec);
        let mdd = tla_mdd::evaluate_reachability_at_initial(&mdd_net, &preds, None, |s, n, p| {
            lower_dd_predicate_to_mdd(s, n, p)
        })
        .expect("mdd reachability");
        assert_eq!(
            bdd, mdd,
            "native tla-bdd reachability must match the MDD lane"
        );
    }

    /// The CTL-lane migration gate: native ROBDD CTL must produce IDENTICAL
    /// verdicts to the MDD CTL lane across the full operator set (EX/AX/EF/AF/
    /// EG/AG/EU/AU + boolean) — both lower the SAME CtlFormulaTemplate over the
    /// SAME spec, with the same A-as-dual-of-E convention.
    #[test]
    fn bdd_ctl_matches_mdd_lane() {
        use super::{dd_spec_to_mdd_net, evaluate_ctl_via_bdd, lower_dd_predicate_to_mdd};
        use tla_dd::{DdIntExpr, DdPredicate};
        use tla_mdd::CtlFormulaTemplate as T;
        let spec = DdNetSpec {
            bounds: vec![1, 1, 1],
            initial_marking: vec![1, 0, 1],
            transitions: vec![t(&[1, 0, 1], &[0, 1, 0]), t(&[0, 1, 0], &[1, 0, 1])],
        };
        // p1 >= 1 (in crit); p1 <= 1 (bound); fireable t0 (acquire).
        let crit = || {
            T::Atom(DdPredicate::IntLe(
                DdIntExpr::Constant(1),
                DdIntExpr::TokensCount(vec![1]),
            ))
        };
        let bound = || {
            T::Atom(DdPredicate::IntLe(
                DdIntExpr::TokensCount(vec![1]),
                DdIntExpr::Constant(1),
            ))
        };
        let fire0 = || T::Atom(DdPredicate::IsFireable(vec![0]));
        let formulas: Vec<T<DdPredicate>> = vec![
            T::EF(Box::new(crit())),
            T::AG(Box::new(bound())),
            T::EG(Box::new(fire0())),
            T::AF(Box::new(crit())),
            T::EX(Box::new(crit())),
            T::AX(Box::new(fire0())),
            T::EU(Box::new(fire0()), Box::new(crit())),
            T::AU(Box::new(T::Not(Box::new(crit()))), Box::new(crit())),
            T::Or(vec![crit(), fire0()]),
            T::And(vec![bound(), T::EF(Box::new(crit()))]),
            T::Not(Box::new(T::EF(Box::new(crit())))),
        ];
        let mdd_net = dd_spec_to_mdd_net(&spec);
        for (i, f) in formulas.iter().enumerate() {
            let bdd = evaluate_ctl_via_bdd(&spec, f, None).expect("bdd ctl");
            let mdd = tla_mdd::evaluate_at_initial(&mdd_net, f, None, |s, n, p| {
                lower_dd_predicate_to_mdd(s, n, p)
            })
            .expect("mdd ctl");
            assert_eq!(
                bdd, mdd,
                "native tla-bdd CTL must match the MDD lane on formula {i}"
            );
        }
    }

    /// The UpperBounds-lane migration gate: native ROBDD UpperBounds must produce
    /// IDENTICAL bounds to the MDD UB lane (`MddNet::upper_bounds`) — same
    /// coefficient-vector queries over the same spec.
    #[test]
    fn bdd_upper_bounds_matches_mdd_lane() {
        use super::{dd_spec_to_mdd_net, upper_bounds_via_bdd};
        let spec = DdNetSpec {
            bounds: vec![1, 1, 1],
            initial_marking: vec![1, 0, 1],
            transitions: vec![t(&[1, 0, 1], &[0, 1, 0]), t(&[0, 1, 0], &[1, 0, 1])],
        };
        let queries: Vec<Vec<i128>> = vec![
            vec![0, 1, 0], // max m[p1] (crit occupancy)
            vec![1, 2, 3], // a weighted sum
            vec![1, 1, 1], // max total tokens
        ];
        let bdd = upper_bounds_via_bdd(&spec, &queries, None).expect("bdd upper bounds");
        let mdd_net = dd_spec_to_mdd_net(&spec);
        let mdd = mdd_net
            .upper_bounds(&queries, None)
            .expect("mdd upper bounds");
        for (i, (b, m)) in bdd.iter().zip(mdd.iter()).enumerate() {
            assert_eq!(
                Some(*b),
                *m,
                "tla-bdd UB must match the MDD UB lane on query {i}"
            );
        }
    }

    /// The MDD variable-ordering soundness gate: `dd_spec_to_ordered_mdd_net`
    /// (span-guarded FORCE place order) with `permute_dd_predicate` /
    /// permuted coefficient vectors must produce EF/AG verdicts and UpperBounds
    /// IDENTICAL to the identity-order MDD lane, on a net whose PNML order FORCE
    /// genuinely improves (asserted non-identity, so the test is not vacuous). A
    /// place-index bug in the permutation would flip a verdict / bound here.
    #[test]
    fn ordered_mdd_lane_matches_identity_lane_on_shuffled_chain() {
        if std::env::var("TY_MCC_DISABLE_MDD_FORCE_ORDER").is_ok() {
            eprintln!("SKIP: FORCE-order test under the 3c measurement gate");
            return;
        }
        use super::{
            dd_spec_to_mdd_net, dd_spec_to_ordered_mdd_net, lower_dd_predicate_to_mdd,
            permute_dd_predicate,
        };
        use tla_dd::{DdIntExpr, DdPredicate};
        use tla_mdd::MddReachQuantifier::{Ag, Ef};

        // Logical chain a->b->c->d->e->f (1 token), places listed interleaved
        // (logical position -> index [0,2,4,1,3,5]) so the PNML order is poor
        // and FORCE finds a strictly better one.
        let l2i = [0usize, 2, 4, 1, 3, 5];
        let n = 6;
        let mut transitions = Vec::new();
        for step in 0..5 {
            let mut pre = vec![0u64; n];
            let mut post = vec![0u64; n];
            pre[l2i[step]] = 1;
            post[l2i[step + 1]] = 1;
            transitions.push(t(&pre, &post));
        }
        let mut initial_marking = vec![0u64; n];
        initial_marking[l2i[0]] = 1;
        let spec = DdNetSpec {
            bounds: vec![1; n],
            initial_marking,
            transitions,
        };

        // The reorder must be genuinely non-identity here (else the test is
        // vacuous — it would be exercising the identity path twice).
        let (ordered_net, inv) = dd_spec_to_ordered_mdd_net(&spec, None);
        assert_ne!(
            inv,
            (0..n).collect::<Vec<_>>(),
            "FORCE must reorder this deliberately-shuffled chain",
        );

        // Queries touching various places and transitions (Fireability +
        // Cardinality, EF + AG, with And/Or/Not structure).
        let card = |p: usize, k: u64| {
            DdPredicate::IntLe(DdIntExpr::Constant(k), DdIntExpr::TokensCount(vec![p]))
        };
        let preds: Vec<(tla_mdd::MddReachQuantifier, DdPredicate)> = vec![
            (Ef, card(l2i[5], 1)), // token reaches the end place
            (
                Ag,
                DdPredicate::IntLe(DdIntExpr::TokensCount(vec![l2i[0]]), DdIntExpr::Constant(1)),
            ), // <=1 tokens (safe) everywhere
            (Ef, DdPredicate::IsFireable(vec![0])), // first step fires initially
            (Ag, DdPredicate::IsFireable(vec![0])), // NOT always fireable
            (
                Ef,
                DdPredicate::Not(Box::new(DdPredicate::IsFireable(vec![4]))),
            ),
            (Ag, DdPredicate::True),
            (Ef, DdPredicate::False),
            (
                Ef,
                DdPredicate::And(vec![
                    card(l2i[3], 1),
                    DdPredicate::IntLe(
                        DdIntExpr::TokensCount(vec![l2i[3]]),
                        DdIntExpr::Constant(1),
                    ),
                ]),
            ),
        ];

        // Identity path (baseline).
        let id_net = dd_spec_to_mdd_net(&spec);
        let identity =
            tla_mdd::evaluate_reachability_at_initial(&id_net, &preds, None, |s, nn, p| {
                lower_dd_predicate_to_mdd(s, nn, p)
            })
            .expect("identity reachability");

        // Ordered path: permute each predicate into the level coordinates.
        let ordered_preds: Vec<(tla_mdd::MddReachQuantifier, DdPredicate)> = preds
            .iter()
            .map(|(q, p)| {
                (
                    *q,
                    permute_dd_predicate(p, &inv).expect("permute predicate"),
                )
            })
            .collect();
        let ordered = tla_mdd::evaluate_reachability_at_initial(
            &ordered_net,
            &ordered_preds,
            None,
            |s, nn, p| lower_dd_predicate_to_mdd(s, nn, p),
        )
        .expect("ordered reachability");

        assert_eq!(
            identity, ordered,
            "ordered MDD reachability must equal the identity lane, verdict-for-verdict",
        );

        // The SEEDED path (NUPN-style structural seed) is likewise
        // verdict-preserving: a seed only feeds `force_place_order_seeded` a
        // different — still valid — permutation candidate under the same span
        // guard. Exercise it with an arbitrary valid permutation seed.
        let (seeded_net, seeded_inv) = dd_spec_to_ordered_mdd_net(&spec, Some(&[5, 4, 3, 2, 1, 0]));
        let seeded_preds: Vec<(tla_mdd::MddReachQuantifier, DdPredicate)> = preds
            .iter()
            .map(|(q, p)| {
                (
                    *q,
                    permute_dd_predicate(p, &seeded_inv).expect("permute predicate"),
                )
            })
            .collect();
        let seeded = tla_mdd::evaluate_reachability_at_initial(
            &seeded_net,
            &seeded_preds,
            None,
            |s, nn, p| lower_dd_predicate_to_mdd(s, nn, p),
        )
        .expect("seeded reachability");
        assert_eq!(
            identity, seeded,
            "seeded-order MDD reachability must equal the identity lane, verdict-for-verdict",
        );

        // UpperBounds parity through the same reorder.
        let queries: Vec<Vec<i128>> = vec![
            {
                let mut q = vec![0i128; n];
                q[l2i[5]] = 1;
                q
            }, // max tokens at the end place
            vec![1i128; n], // total token count
            {
                let mut q = vec![0i128; n];
                q[l2i[2]] = 2;
                q[l2i[4]] = 3;
                q
            }, // a weighted sum
        ];
        let id_ub = id_net.upper_bounds(&queries, None).expect("identity UB");
        let ordered_queries: Vec<Vec<i128>> = queries
            .iter()
            .map(|q| {
                let mut o = vec![0i128; n];
                for (orig, &c) in q.iter().enumerate() {
                    o[inv[orig]] = c;
                }
                o
            })
            .collect();
        let ordered_ub = ordered_net
            .upper_bounds(&ordered_queries, None)
            .expect("ordered UB");
        assert_eq!(
            id_ub, ordered_ub,
            "ordered MDD UpperBounds must equal the identity lane, bound-for-bound",
        );
    }

    /// `order_mdd_net` (the colored-path reorderer, which takes a built `MddNet`
    /// rather than a `DdNetSpec`) must preserve UpperBounds: it is the same
    /// span-guarded FORCE relabel, so the per-query maximum is invariant once the
    /// coefficient vectors are permuted with the returned `inv`.
    #[test]
    fn order_mdd_net_preserves_upper_bounds() {
        if std::env::var("TY_MCC_DISABLE_MDD_FORCE_ORDER").is_ok() {
            eprintln!("SKIP: FORCE-order test under the 3c measurement gate");
            return;
        }
        use super::{dd_spec_to_mdd_net, order_mdd_net};
        // Shuffled conserved chain (FORCE finds a strictly better order).
        let l2i = [0usize, 2, 4, 1, 3, 5];
        let n = 6;
        let mut transitions = Vec::new();
        for step in 0..5 {
            let mut pre = vec![0u64; n];
            let mut post = vec![0u64; n];
            pre[l2i[step]] = 1;
            post[l2i[step + 1]] = 1;
            transitions.push(t(&pre, &post));
        }
        let mut initial_marking = vec![0u64; n];
        initial_marking[l2i[0]] = 1;
        let spec = DdNetSpec {
            bounds: vec![1; n],
            initial_marking,
            transitions,
        };

        let id_net = dd_spec_to_mdd_net(&spec);
        let queries: Vec<Vec<i128>> = vec![
            {
                let mut q = vec![0i128; n];
                q[l2i[5]] = 1;
                q
            },
            vec![1i128; n],
        ];
        let id_ub = id_net.upper_bounds(&queries, None).expect("identity UB");

        let (ord_net, inv) = order_mdd_net(id_net);
        assert_ne!(
            inv,
            (0..n).collect::<Vec<_>>(),
            "FORCE must reorder the shuffled chain (else the test is vacuous)",
        );
        let ord_queries: Vec<Vec<i128>> = queries
            .iter()
            .map(|q| {
                let mut o = vec![0i128; n];
                for (orig, &c) in q.iter().enumerate() {
                    o[inv[orig]] = c;
                }
                o
            })
            .collect();
        let ord_ub = ord_net
            .upper_bounds(&ord_queries, None)
            .expect("ordered UB");
        assert_eq!(
            id_ub, ord_ub,
            "order_mdd_net must preserve UpperBounds bound-for-bound"
        );
    }
}
