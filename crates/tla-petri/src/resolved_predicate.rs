// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared resolved predicate types and evaluation for MCC property checking.
//!
//! This module provides name-to-index resolved representations of MCC
//! state predicates and integer expressions.  The same resolve/eval logic
//! is used by the reachability, CTL, and LTL/Buchi engines — deduplicating
//! what was previously three parallel implementations.

#[cfg(test)]
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::model::PropertyAliases;
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::property_xml::{IntExpr, StatePredicate};

/// Resolved integer expression with place indices instead of names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum ResolvedIntExpr {
    Constant(u64),
    TokensCount(Vec<PlaceIdx>),
}

/// Resolved state predicate with place/transition indices instead of names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum ResolvedPredicate {
    And(Vec<ResolvedPredicate>),
    Or(Vec<ResolvedPredicate>),
    Not(Box<ResolvedPredicate>),
    IntLe(ResolvedIntExpr, ResolvedIntExpr),
    IsFireable(Vec<TransitionIdx>),
    True,
    False,
}

impl ResolvedIntExpr {
    pub(crate) fn collect_places(&self, out: &mut std::collections::HashSet<PlaceIdx>) {
        match self {
            ResolvedIntExpr::Constant(_) => {}
            ResolvedIntExpr::TokensCount(places) => {
                for p in places {
                    out.insert(*p);
                }
            }
        }
    }
}

impl ResolvedPredicate {
    pub(crate) fn collect_places(&self, out: &mut std::collections::HashSet<PlaceIdx>) {
        match self {
            ResolvedPredicate::And(children) | ResolvedPredicate::Or(children) => {
                for child in children {
                    child.collect_places(out);
                }
            }
            ResolvedPredicate::Not(inner) => inner.collect_places(out),
            ResolvedPredicate::IntLe(left, right) => {
                left.collect_places(out);
                right.collect_places(out);
            }
            ResolvedPredicate::IsFireable(_)
            | ResolvedPredicate::True
            | ResolvedPredicate::False => {}
        }
    }
}

/// Resolve a [`StatePredicate`] from string names to integer indices.
pub(crate) fn resolve_predicate_with_aliases(
    pred: &StatePredicate,
    aliases: &PropertyAliases,
) -> ResolvedPredicate {
    match pred {
        StatePredicate::And(children) => ResolvedPredicate::And(
            children
                .iter()
                .map(|c| resolve_predicate_with_aliases(c, aliases))
                .collect(),
        ),
        StatePredicate::Or(children) => ResolvedPredicate::Or(
            children
                .iter()
                .map(|c| resolve_predicate_with_aliases(c, aliases))
                .collect(),
        ),
        StatePredicate::Not(inner) => {
            ResolvedPredicate::Not(Box::new(resolve_predicate_with_aliases(inner, aliases)))
        }
        StatePredicate::IntLe(left, right) => ResolvedPredicate::IntLe(
            resolve_int_expr_with_aliases(left, aliases),
            resolve_int_expr_with_aliases(right, aliases),
        ),
        StatePredicate::IsFireable(transitions) => {
            let indices: Vec<TransitionIdx> = transitions
                .iter()
                .flat_map(|name| {
                    aliases
                        .resolve_transitions(name)
                        .into_iter()
                        .flat_map(|resolved| resolved.iter().copied())
                })
                .collect();
            if indices.is_empty() {
                ResolvedPredicate::False
            } else {
                ResolvedPredicate::IsFireable(indices)
            }
        }
        StatePredicate::True => ResolvedPredicate::True,
        StatePredicate::False => ResolvedPredicate::False,
    }
}

/// Resolve an [`IntExpr`] from string place names to integer indices.
pub(crate) fn resolve_int_expr_with_aliases(
    expr: &IntExpr,
    aliases: &PropertyAliases,
) -> ResolvedIntExpr {
    match expr {
        IntExpr::Constant(v) => ResolvedIntExpr::Constant(*v),
        IntExpr::TokensCount(places) => {
            ResolvedIntExpr::TokensCount(resolve_place_bound(places, aliases))
        }
    }
}

pub(crate) fn resolve_place_bound(
    place_names: &[String],
    aliases: &PropertyAliases,
) -> Vec<PlaceIdx> {
    place_names
        .iter()
        .flat_map(|name| {
            // Defense-in-depth: dedup the indices that a SINGLE name
            // resolves to. A colored place name must contribute each
            // unfolded instance exactly once; an alias-accumulation path
            // that duplicated indices would otherwise be read as
            // coefficient 2, doubling the UpperBounds / token-sum.
            //
            // Dedup is PER-NAME only — cross-name multiplicity is the
            // legitimate `place-bound(P, P) = 2*tokens(P)` multiset
            // semantics that `structural_query_bound` relies on, so we must
            // NOT dedup across the whole flattened list.
            let mut resolved: Vec<PlaceIdx> = aliases
                .resolve_places(name)
                .map(<[_]>::to_vec)
                .unwrap_or_default();
            resolved.sort_unstable();
            resolved.dedup();
            resolved
        })
        .collect()
}

pub(crate) fn remap_int_expr(
    expr: &ResolvedIntExpr,
    place_map: &[Option<PlaceIdx>],
) -> Option<ResolvedIntExpr> {
    match expr {
        ResolvedIntExpr::Constant(value) => Some(ResolvedIntExpr::Constant(*value)),
        ResolvedIntExpr::TokensCount(places) => {
            let remapped = places
                .iter()
                .map(|place| place_map[place.0 as usize])
                .collect::<Option<Vec<_>>>()?;
            Some(ResolvedIntExpr::TokensCount(remapped))
        }
    }
}

/// Check whether all transitions in `IsFireable` predicates survived reduction.
///
/// Returns `true` if every transition index referenced by `IsFireable` nodes
/// in the predicate has a mapping in `trans_map`. If any transition was
/// eliminated (`trans_map[idx] == None`), returns `false`.
#[cfg(test)]
pub(crate) fn predicate_transitions_survived(
    pred: &ResolvedPredicate,
    trans_map: &[Option<TransitionIdx>],
) -> bool {
    match pred {
        ResolvedPredicate::And(children) | ResolvedPredicate::Or(children) => children
            .iter()
            .all(|child| predicate_transitions_survived(child, trans_map)),
        ResolvedPredicate::Not(inner) => predicate_transitions_survived(inner, trans_map),
        ResolvedPredicate::IsFireable(transitions) => transitions
            .iter()
            .all(|t| trans_map[t.0 as usize].is_some()),
        ResolvedPredicate::IntLe(..) | ResolvedPredicate::True | ResolvedPredicate::False => true,
    }
}

/// Check whether the reduction preserves all transitions relevant to a predicate.
///
/// Extends [`predicate_transitions_survived`] to also cover `TokensCount`:
/// for each place referenced by `TokensCount`, verifies that ALL transitions
/// in the original `net` with arcs to/from that place survived the reduction.
/// If any such transition was eliminated, the reduced net's state space may be
/// incomplete for the monitored places, causing unsound AG=TRUE verdicts.
pub(crate) fn predicate_reduction_safe(
    pred: &ResolvedPredicate,
    net: &PetriNet,
    trans_map: &[Option<TransitionIdx>],
) -> bool {
    match pred {
        ResolvedPredicate::And(children) | ResolvedPredicate::Or(children) => children
            .iter()
            .all(|child| predicate_reduction_safe(child, net, trans_map)),
        ResolvedPredicate::Not(inner) => predicate_reduction_safe(inner, net, trans_map),
        ResolvedPredicate::IsFireable(transitions) => transitions
            .iter()
            .all(|t| trans_map[t.0 as usize].is_some()),
        ResolvedPredicate::IntLe(left, right) => {
            int_expr_place_transitions_survived(left, net, trans_map)
                && int_expr_place_transitions_survived(right, net, trans_map)
        }
        ResolvedPredicate::True | ResolvedPredicate::False => true,
    }
}

fn int_expr_place_transitions_survived(
    expr: &ResolvedIntExpr,
    net: &PetriNet,
    trans_map: &[Option<TransitionIdx>],
) -> bool {
    match expr {
        ResolvedIntExpr::Constant(_) => true,
        ResolvedIntExpr::TokensCount(places) => {
            for place in places {
                for (tidx, trans) in net.transitions.iter().enumerate() {
                    let touches = trans.inputs.iter().any(|arc| arc.place == *place)
                        || trans.outputs.iter().any(|arc| arc.place == *place);
                    if touches && trans_map[tidx].is_none() {
                        return false;
                    }
                }
            }
            true
        }
    }
}

pub(crate) fn remap_predicate(
    pred: &ResolvedPredicate,
    place_map: &[Option<PlaceIdx>],
    trans_map: &[Option<TransitionIdx>],
) -> Option<ResolvedPredicate> {
    match pred {
        ResolvedPredicate::And(children) => Some(ResolvedPredicate::And(
            children
                .iter()
                .map(|child| remap_predicate(child, place_map, trans_map))
                .collect::<Option<Vec<_>>>()?,
        )),
        ResolvedPredicate::Or(children) => Some(ResolvedPredicate::Or(
            children
                .iter()
                .map(|child| remap_predicate(child, place_map, trans_map))
                .collect::<Option<Vec<_>>>()?,
        )),
        ResolvedPredicate::Not(inner) => Some(ResolvedPredicate::Not(Box::new(remap_predicate(
            inner, place_map, trans_map,
        )?))),
        ResolvedPredicate::IntLe(left, right) => Some(ResolvedPredicate::IntLe(
            remap_int_expr(left, place_map)?,
            remap_int_expr(right, place_map)?,
        )),
        ResolvedPredicate::IsFireable(transitions) => Some(ResolvedPredicate::IsFireable(
            transitions
                .iter()
                .map(|transition| trans_map[transition.0 as usize])
                .collect::<Option<Vec<_>>>()?,
        )),
        ResolvedPredicate::True => Some(ResolvedPredicate::True),
        ResolvedPredicate::False => Some(ResolvedPredicate::False),
    }
}

/// A side of an `IntLe` comparison after slice remapping, expressed in slice
/// coordinates together with the GCD scaling factor that the slice marking was
/// divided by.
///
/// In a GCD-scaled slice, a surviving place `p` holds `m_slice[p]` where the
/// original-coordinate marking is `m_orig = scale * m_slice[p]`. So an integer
/// expression's original value is `scale * <slice value>`. `Const` always has
/// scale 1 (constants are never scaled); `Tokens` carries the *common* scale
/// shared by all its referenced places.
enum ScaledSide {
    /// Constant value `c`; original value == slice value == `c`.
    Const(u64),
    /// Sum of slice token counts with a single common scale `g`: original value
    /// == `g * (sum of slice tokens)`.
    Tokens { scale: u64, places: Vec<PlaceIdx> },
}

/// Resolve one side of an `IntLe` into slice coordinates plus its common scale.
///
/// Returns `None` (decline the slice) when a `TokensCount` references places
/// with *differing* scales — the unweighted [`ResolvedIntExpr::TokensCount`]
/// sum cannot represent `g1*tokens(p1) + g2*tokens(p2)` exactly when
/// `g1 != g2`.
fn remap_int_expr_scaled(
    expr: &ResolvedIntExpr,
    place_map: &[Option<PlaceIdx>],
    place_scales: &[u64],
) -> Option<ScaledSide> {
    match expr {
        ResolvedIntExpr::Constant(value) => Some(ScaledSide::Const(*value)),
        ResolvedIntExpr::TokensCount(places) => {
            let mut remapped = Vec::with_capacity(places.len());
            let mut common_scale: Option<u64> = None;
            for place in places {
                let mapped = place_map[place.0 as usize]?;
                let scale = place_scales[place.0 as usize].max(1);
                match common_scale {
                    None => common_scale = Some(scale),
                    Some(existing) if existing == scale => {}
                    // Mixed scales within one TokensCount sum: not exactly
                    // representable as an unweighted slice token sum. Decline.
                    Some(_) => return None,
                }
                remapped.push(mapped);
            }
            // An empty TokensCount sums to 0; treat as the constant 0.
            let scale = common_scale.unwrap_or(1);
            if remapped.is_empty() {
                Some(ScaledSide::Const(0))
            } else {
                Some(ScaledSide::Tokens {
                    scale,
                    places: remapped,
                })
            }
        }
    }
}

/// Build a slice-coordinate `IntLe(left, right)` that is *exactly* equivalent to
/// the original-coordinate comparison `left_orig <= right_orig`, accounting for
/// GCD scaling of the slice's places.
///
/// Original semantics: `lhs_scale * L_slice <= rhs_scale * R_slice` where each
/// side's slice value is either a constant or a token sum. Because slice token
/// sums are non-negative integers we can clear the scale factors exactly:
///
/// - tokens `g*S` vs const `c`:  `S <= floor(c / g)`
/// - const `c`  vs tokens `g*S`:  `ceil(c / g) <= S`
/// - tokens `g*S` vs tokens `g*T` (equal scale):  `S <= T`
/// - const vs const: unchanged.
///
/// Returns `None` (decline the slice) for tokens-vs-tokens with *differing*
/// scales, which cannot be represented exactly with unweighted token sums.
fn remap_int_le_scaled(
    left: &ResolvedIntExpr,
    right: &ResolvedIntExpr,
    place_map: &[Option<PlaceIdx>],
    place_scales: &[u64],
) -> Option<ResolvedPredicate> {
    let lhs = remap_int_expr_scaled(left, place_map, place_scales)?;
    let rhs = remap_int_expr_scaled(right, place_map, place_scales)?;

    let (l, r) = match (lhs, rhs) {
        (ScaledSide::Const(a), ScaledSide::Const(b)) => {
            (ResolvedIntExpr::Constant(a), ResolvedIntExpr::Constant(b))
        }
        (ScaledSide::Tokens { scale, places }, ScaledSide::Const(c)) => {
            // g*S <= c  <=>  S <= floor(c / g)
            (
                ResolvedIntExpr::TokensCount(places),
                ResolvedIntExpr::Constant(c / scale),
            )
        }
        (ScaledSide::Const(c), ScaledSide::Tokens { scale, places }) => {
            // c <= g*S  <=>  ceil(c / g) <= S
            let ceil = c.div_ceil(scale);
            (
                ResolvedIntExpr::Constant(ceil),
                ResolvedIntExpr::TokensCount(places),
            )
        }
        (
            ScaledSide::Tokens {
                scale: ls,
                places: lp,
            },
            ScaledSide::Tokens {
                scale: rs,
                places: rp,
            },
        ) => {
            if ls != rs {
                // ls*SL <= rs*SR with differing scales is not exactly
                // expressible via unweighted token sums. Decline the slice.
                return None;
            }
            // Equal scale g > 0: g*SL <= g*SR  <=>  SL <= SR.
            (
                ResolvedIntExpr::TokensCount(lp),
                ResolvedIntExpr::TokensCount(rp),
            )
        }
    };
    Some(ResolvedPredicate::IntLe(l, r))
}

/// Scale-aware variant of [`remap_predicate`] for the GCD-scaled slice path.
///
/// Identical to [`remap_predicate`] except that `IntLe` comparisons are rewritten
/// to be evaluated *exactly* against the slice's GCD-scaled marking, where a
/// surviving place `p` holds `m_orig / place_scales[p]`. `place_scales` is indexed
/// by *original* place index (the same index space as the predicate's places and
/// the `place_map` source index).
///
/// Returns `None` to decline the slice — propagated by the caller to route every
/// tracker to the sound non-slice reduced-net path — whenever the transform cannot
/// be represented exactly (mixed scales in one token sum, or scaled
/// tokens-vs-tokens comparisons). When every referenced place has scale 1 this is
/// behaviorally identical to [`remap_predicate`].
pub(crate) fn remap_predicate_scaled(
    pred: &ResolvedPredicate,
    place_map: &[Option<PlaceIdx>],
    trans_map: &[Option<TransitionIdx>],
    place_scales: &[u64],
) -> Option<ResolvedPredicate> {
    match pred {
        ResolvedPredicate::And(children) => Some(ResolvedPredicate::And(
            children
                .iter()
                .map(|child| remap_predicate_scaled(child, place_map, trans_map, place_scales))
                .collect::<Option<Vec<_>>>()?,
        )),
        ResolvedPredicate::Or(children) => Some(ResolvedPredicate::Or(
            children
                .iter()
                .map(|child| remap_predicate_scaled(child, place_map, trans_map, place_scales))
                .collect::<Option<Vec<_>>>()?,
        )),
        ResolvedPredicate::Not(inner) => Some(ResolvedPredicate::Not(Box::new(
            remap_predicate_scaled(inner, place_map, trans_map, place_scales)?,
        ))),
        ResolvedPredicate::IntLe(left, right) => {
            remap_int_le_scaled(left, right, place_map, place_scales)
        }
        ResolvedPredicate::IsFireable(transitions) => Some(ResolvedPredicate::IsFireable(
            transitions
                .iter()
                .map(|transition| trans_map[transition.0 as usize])
                .collect::<Option<Vec<_>>>()?,
        )),
        ResolvedPredicate::True => Some(ResolvedPredicate::True),
        ResolvedPredicate::False => Some(ResolvedPredicate::False),
    }
}

/// Count unresolved names in a predicate resolution.
///
/// Compares the original formula's name counts against the resolved index counts.
/// Returns (total_names, unresolved_count) — if unresolved_count > 0, the formula
/// has names that didn't match the model's place/transition IDs.
pub(crate) fn count_unresolved_with_aliases(
    pred: &StatePredicate,
    aliases: &PropertyAliases,
) -> (usize, usize) {
    match pred {
        StatePredicate::And(children) | StatePredicate::Or(children) => {
            children.iter().fold((0, 0), |(t, u), c| {
                let (ct, cu) = count_unresolved_with_aliases(c, aliases);
                (t + ct, u + cu)
            })
        }
        StatePredicate::Not(inner) => count_unresolved_with_aliases(inner, aliases),
        StatePredicate::IntLe(left, right) => {
            let (lt, lu) = count_unresolved_int_with_aliases(left, aliases);
            let (rt, ru) = count_unresolved_int_with_aliases(right, aliases);
            (lt + rt, lu + ru)
        }
        StatePredicate::IsFireable(transitions) => {
            let unresolved = transitions
                .iter()
                .filter(|name| aliases.resolve_transitions(name).is_none())
                .count();
            (transitions.len(), unresolved)
        }
        StatePredicate::True | StatePredicate::False => (0, 0),
    }
}

fn count_unresolved_int_with_aliases(expr: &IntExpr, aliases: &PropertyAliases) -> (usize, usize) {
    match expr {
        IntExpr::Constant(_) => (0, 0),
        IntExpr::TokensCount(places) => count_unresolved_place_bound(places, aliases),
    }
}

pub(crate) fn count_unresolved_place_bound(
    place_names: &[String],
    aliases: &PropertyAliases,
) -> (usize, usize) {
    let unresolved = place_names
        .iter()
        .filter(|name| aliases.resolve_places(name).is_none())
        .count();
    (place_names.len(), unresolved)
}

#[cfg(test)]
fn aliases_from_maps(
    place_map: &HashMap<&str, PlaceIdx>,
    trans_map: &HashMap<&str, TransitionIdx>,
) -> PropertyAliases {
    PropertyAliases {
        place_aliases: place_map
            .iter()
            .map(|(name, idx)| ((*name).to_string(), vec![*idx]))
            .collect(),
        transition_aliases: trans_map
            .iter()
            .map(|(name, idx)| ((*name).to_string(), vec![*idx]))
            .collect(),
        colored_place_group_aliases: HashSet::new(),
    }
}

#[cfg(test)]
pub(crate) fn resolve_predicate(
    pred: &StatePredicate,
    place_map: &HashMap<&str, PlaceIdx>,
    trans_map: &HashMap<&str, TransitionIdx>,
) -> ResolvedPredicate {
    let aliases = aliases_from_maps(place_map, trans_map);
    resolve_predicate_with_aliases(pred, &aliases)
}

#[cfg(test)]
pub(crate) fn resolve_int_expr(
    expr: &IntExpr,
    place_map: &HashMap<&str, PlaceIdx>,
) -> ResolvedIntExpr {
    let aliases = PropertyAliases {
        place_aliases: place_map
            .iter()
            .map(|(name, idx)| ((*name).to_string(), vec![*idx]))
            .collect(),
        transition_aliases: HashMap::new(),
        colored_place_group_aliases: HashSet::new(),
    };
    resolve_int_expr_with_aliases(expr, &aliases)
}

#[cfg(test)]
pub(crate) fn count_unresolved(
    pred: &StatePredicate,
    place_map: &HashMap<&str, PlaceIdx>,
    trans_map: &HashMap<&str, TransitionIdx>,
) -> (usize, usize) {
    let aliases = aliases_from_maps(place_map, trans_map);
    count_unresolved_with_aliases(pred, &aliases)
}

/// Evaluate a resolved predicate against a marking.
pub(crate) fn eval_predicate(pred: &ResolvedPredicate, marking: &[u64], net: &PetriNet) -> bool {
    match pred {
        ResolvedPredicate::And(children) => {
            children.iter().all(|c| eval_predicate(c, marking, net))
        }
        ResolvedPredicate::Or(children) => children.iter().any(|c| eval_predicate(c, marking, net)),
        ResolvedPredicate::Not(inner) => !eval_predicate(inner, marking, net),
        ResolvedPredicate::IntLe(left, right) => {
            eval_int_expr(left, marking) <= eval_int_expr(right, marking)
        }
        ResolvedPredicate::IsFireable(transitions) => {
            transitions.iter().any(|&t| net.is_enabled(marking, t))
        }
        ResolvedPredicate::True => true,
        ResolvedPredicate::False => false,
    }
}

/// Evaluate a resolved integer expression against a marking.
pub(crate) fn eval_int_expr(expr: &ResolvedIntExpr, marking: &[u64]) -> u64 {
    match expr {
        ResolvedIntExpr::Constant(v) => *v,
        ResolvedIntExpr::TokensCount(indices) => {
            indices.iter().map(|idx| marking[idx.0 as usize]).sum()
        }
    }
}

#[cfg(test)]
#[path = "resolved_predicate_tests.rs"]
mod resolved_predicate_tests;
