// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Sound colored-net shortcuts that resolve MCC examinations without
//! unfolding to a P/T net.
//!
//! Unfolding a HLPN net to its P/T equivalent is the default path, but for
//! the largest colored MCC families the unfolded net exceeds the place cap
//! and the answer is structurally forced before unfolding ever runs.
//! Each shortcut here returns `Some(verdict)` ONLY when a syntactic check on
//! the colored IR is sufficient to soundly answer the examination; otherwise
//! it returns `None` and the caller falls through to the unfolding-based
//! pipeline unchanged.
//!
//! # Soundness contract
//!
//! Every shortcut MUST be a conservative one-sided proof: false positives
//! are forbidden (returning `Verdict::True` when the real answer is `False`
//! is a competition-losing soundness bug). False negatives (returning `None`
//! when a tighter proof would have given `True`) are acceptable — the
//! unfolding path catches those.

use crate::hlpnml::{ColorExpr, ColorSort, ColorTerm, ColoredNet, GuardExpr};
use crate::output::Verdict;

/// Try to prove the colored net is 1-safe by structural inspection alone.
///
/// Returns `Some(Verdict::True)` only when the syntactic preconditions
/// guarantee no place can ever hold more than 1 token of any color (which
/// is the MCC 1-safe property after unfolding, since each unfolded place
/// corresponds to one (place, color) instance).
///
/// Returns `None` whenever the preconditions are not met — the caller must
/// then fall through to the standard unfolding-based OneSafe pipeline.
/// We never return `Verdict::False` here: refuting 1-safety requires
/// reachability evidence that this structural pass does not gather.
///
/// # Conservative sound conditions
///
/// 1. **Initial marking ≤ 1 per color per place.** Each place's initial
///    marking is `None`, `All { count: 1, .. }` (1 of every color), or
///    `NumberOf { count: 1, .. }` (1 of a single color). `Add`, and any
///    `count > 1`, are rejected — these can put more than one token on a
///    single (place, color) instance.
///
/// 2. **No transition has a guard.** A guard introduces binding-dependent
///    firing semantics that are not visible from arc syntax alone; we
///    refuse to reason about them here.
///
/// 3. **Per-transition arc balance.** For every transition `T` and every
///    output arc `(T → P)` with inscription `O`, there must exist a
///    distinct input arc `(P → T)` with inscription syntactically equal
///    to `O`. This guarantees firing `T` consumes from each (place, color)
///    instance at least as many tokens as it produces, so the per-instance
///    token count is monotone non-increasing on every step. Combined with
///    (1), no instance ever exceeds 1 token.
///
/// Why "syntactically equal" is sound: `All { sort_id, count: 1 }` produces
/// 1 token per color of `sort_id`; a matching input arc consumes 1 per color
/// of the same `sort_id`. `NumberOf { count: 1, color }` produces 1 token of
/// a single color; the matching input consumes the same. In both cases the
/// input multiset dominates the output multiset pointwise.
///
/// Why these conditions are conservative: real 1-safe colored nets can
/// also have transitions whose outputs are dominated by *aggregated*
/// inputs across multiple arcs, or whose firing is bounded by a guard,
/// or whose initial markings sum to ≤ 1 via additive expressions. Those
/// cases fall through and exercise the unfolding path.
pub(crate) fn try_one_safe_colored_shortcut(colored: &ColoredNet) -> Option<Verdict> {
    // Condition 1: every colored place's initial marking deposits ≤ 1 token
    // *in total* across all of its color instances. MCC defines 1-safe for
    // colored nets as a GROUP-level property: a colored place P holds the sum
    // of its unfolded (P, color) instances, and that sum must stay ≤ 1. A
    // per-instance bound is NOT sufficient — e.g. `All` with `count: 1` over a
    // 3-color sort deposits 1 token in each color, for a group total of 3 > 1,
    // which is NOT 1-safe even though every instance holds exactly 1.
    for place in &colored.places {
        if let Some(marking) = &place.initial_marking {
            match marking_group_total_upper_bound(marking, &place.sort_id, colored) {
                Some(total) if total <= 1 => {}
                // total > 1, or could-not-bound (None): refuse and fall through
                // to the unfolding-based group-sum OneSafe path, which can
                // soundly issue `Verdict::False`.
                _ => return None,
            }
        }
    }

    // Condition 2: no transition has a guard. (`GuardExpr::True` is
    // semantically vacuous but the parser only emits `Some(_)` when a
    // `<condition>` element was present, so we treat any non-`True` `Some(_)`
    // as "non-trivial enough to refuse".)
    for transition in &colored.transitions {
        if let Some(guard) = &transition.guard {
            if !matches!(guard, GuardExpr::True) {
                return None;
            }
        }
    }

    // Build a transition-id set so we can distinguish arcs that touch
    // transitions from arcs between places (which the HLPNML schema
    // forbids but we defensively skip).
    let transition_ids: std::collections::HashSet<&str> =
        colored.transitions.iter().map(|t| t.id.as_str()).collect();

    // Condition 3: arc balance per transition.
    //
    // For every (T → P, inscription O), require a distinct (P → T) input
    // arc with the same inscription. We track which input arcs have
    // already been "claimed" by a matching output arc so two outputs to
    // the same place cannot both rely on the same input.
    for transition in &colored.transitions {
        let outputs: Vec<&_> = colored
            .arcs
            .iter()
            .filter(|a| a.source == transition.id && !transition_ids.contains(a.target.as_str()))
            .collect();
        let inputs: Vec<&_> = colored
            .arcs
            .iter()
            .filter(|a| a.target == transition.id && !transition_ids.contains(a.source.as_str()))
            .collect();

        let mut claimed = vec![false; inputs.len()];
        for out_arc in &outputs {
            let matched = inputs.iter().enumerate().find(|(idx, in_arc)| {
                !claimed[*idx]
                    && in_arc.source == out_arc.target
                    && color_expr_eq(&in_arc.inscription, &out_arc.inscription)
            });
            match matched {
                Some((idx, _)) => claimed[idx] = true,
                None => return None,
            }
        }
    }

    Some(Verdict::True)
}

/// Conservative UPPER BOUND on the number of tokens an initial marking
/// deposits on a colored place *in total* across every color instance of its
/// sort. Returns `None` when no exact, cheap bound can be computed (the caller
/// then refuses the shortcut and falls through to the unfolding path).
///
/// This is the group-level quantity that MCC's colored 1-safe property
/// constrains: the unfolded place sum, not the per-instance count.
///
/// - `All { count, .. }` fills *every* one of the place sort's `cardinality`
///   color slots with `count`, so the group total is `count * cardinality`.
///   The HLPNML loader (`eval_initial_marking`) uses the place sort's
///   cardinality for `<all>` markings, so using it here matches the unfolded
///   semantics exactly.
/// - `NumberOf { count, color }` deposits `count` tokens into each color value
///   the term `color` denotes. A single concrete color term (a `UserConstant`,
///   `IntegerConstant`, or `DotConstant`, possibly under `Predecessor`/
///   `Successor`) denotes exactly one value, so the group total is `count`. A
///   color term that can denote *multiple* values (`All`, a free `Variable`,
///   or a `Tuple` containing either) would spread `count` across several slots;
///   we return `None` to refuse rather than under-count.
/// - `Add` aggregates its children; a sound upper bound is the sum of the
///   children's bounds (refusing if any child is unbounded). Note an `Add` can
///   place more than one token on a single instance, so this correctly forces
///   refusal unless the children jointly sum to ≤ 1.
fn marking_group_total_upper_bound(
    expr: &ColorExpr,
    place_sort_id: &str,
    colored: &ColoredNet,
) -> Option<u64> {
    match expr {
        ColorExpr::All { count, .. } => {
            let cardinality = sort_cardinality(place_sort_id, colored)?;
            count.checked_mul(cardinality)
        }
        ColorExpr::NumberOf { count, color } => {
            if color_term_is_single_value(color) {
                Some(*count)
            } else {
                // Multi-valued color term (All / Variable / Tuple with either):
                // `count` is spread across multiple slots — refuse.
                None
            }
        }
        ColorExpr::Add(children) => {
            let mut total: u64 = 0;
            for child in children {
                let child_total = marking_group_total_upper_bound(child, place_sort_id, colored)?;
                total = total.checked_add(child_total)?;
            }
            Some(total)
        }
        // A truncated difference in an initial marking: refuse rather than risk
        // an unsound group bound. The left operand's bound is a sound *upper*
        // bound on `lhs - rhs`, but the shortcuts here also need to reason
        // about exact per-instance distribution; refusing keeps every consumer
        // sound (it simply declines the shortcut, never returns a wrong bound).
        ColorExpr::Subtract { .. } => None,
    }
}

/// Resolve the number of color values of the sort named `sort_id` within
/// `colored`. Mirrors the unfolding `sort_cardinality`: `Dot` is 1, a cyclic
/// enumeration is its constant count, a finite int range is its span, and a
/// product is the product of its components. Returns `None` (refuse) on any
/// missing sort, empty range, or arithmetic overflow.
fn sort_cardinality(sort_id: &str, colored: &ColoredNet) -> Option<u64> {
    let sort = colored.sorts.iter().find(|s| s.id() == sort_id)?;
    match sort {
        ColorSort::Dot { .. } => Some(1),
        ColorSort::CyclicEnum { constants, .. } => u64::try_from(constants.len()).ok(),
        ColorSort::FiniteIntRange { start, end, .. } => {
            if end < start {
                return None;
            }
            let span = (*end as i128) - (*start as i128) + 1;
            u64::try_from(span).ok()
        }
        ColorSort::Product { components, .. } => {
            let mut cardinality: u64 = 1;
            for component_id in components {
                let component = sort_cardinality(component_id, colored)?;
                cardinality = cardinality.checked_mul(component)?;
            }
            Some(cardinality)
        }
    }
}

/// True when a color term denotes exactly one color value. Conservative:
/// anything that may range over multiple values (`All`, a free `Variable`, or
/// a `Tuple` containing either) returns `false` so the caller refuses.
fn color_term_is_single_value(term: &ColorTerm) -> bool {
    match term {
        ColorTerm::UserConstant(_) | ColorTerm::IntegerConstant(_) | ColorTerm::DotConstant => true,
        ColorTerm::Predecessor(inner) | ColorTerm::Successor(inner) => {
            color_term_is_single_value(inner)
        }
        ColorTerm::Tuple(components) => components.iter().all(color_term_is_single_value),
        ColorTerm::All(_) | ColorTerm::Variable(_) => false,
    }
}

/// Syntactic equality on color expressions used by the per-transition
/// arc-balance check. Strict structural comparison is sound: when input
/// and output inscriptions are identical, the transition consumes from
/// the same (place, color) instance(s) that it produces to.
fn color_expr_eq(lhs: &ColorExpr, rhs: &ColorExpr) -> bool {
    match (lhs, rhs) {
        (
            ColorExpr::All {
                sort_id: l,
                count: lc,
            },
            ColorExpr::All {
                sort_id: r,
                count: rc,
            },
        ) => l == r && lc == rc,
        (
            ColorExpr::NumberOf {
                count: lc,
                color: lt,
            },
            ColorExpr::NumberOf {
                count: rc,
                color: rt,
            },
        ) => lc == rc && color_term_eq(lt, rt),
        (ColorExpr::Add(lc), ColorExpr::Add(rc)) => {
            lc.len() == rc.len() && lc.iter().zip(rc.iter()).all(|(a, b)| color_expr_eq(a, b))
        }
        (ColorExpr::Subtract { lhs: ll, rhs: lr }, ColorExpr::Subtract { lhs: rl, rhs: rr }) => {
            color_expr_eq(ll, rl) && color_expr_eq(lr, rr)
        }
        _ => false,
    }
}

/// Syntactic equality on color terms.
fn color_term_eq(lhs: &ColorTerm, rhs: &ColorTerm) -> bool {
    match (lhs, rhs) {
        (ColorTerm::Variable(l), ColorTerm::Variable(r)) => l == r,
        (ColorTerm::Tuple(lc), ColorTerm::Tuple(rc)) => {
            lc.len() == rc.len() && lc.iter().zip(rc.iter()).all(|(a, b)| color_term_eq(a, b))
        }
        (ColorTerm::Predecessor(l), ColorTerm::Predecessor(r))
        | (ColorTerm::Successor(l), ColorTerm::Successor(r)) => color_term_eq(l, r),
        (ColorTerm::UserConstant(l), ColorTerm::UserConstant(r)) => l == r,
        (ColorTerm::IntegerConstant(l), ColorTerm::IntegerConstant(r)) => l == r,
        (ColorTerm::All(l), ColorTerm::All(r)) => l == r,
        (ColorTerm::DotConstant, ColorTerm::DotConstant) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hlpnml::{ColorConstant, ColorSort, ColoredArc, ColoredPlace, ColoredTransition};

    fn cyclic_sort(id: &str, n: usize) -> ColorSort {
        ColorSort::CyclicEnum {
            id: id.to_string(),
            name: id.to_string(),
            constants: (0..n)
                .map(|i| ColorConstant {
                    id: format!("{id}_c{i}"),
                    name: format!("c{i}"),
                })
                .collect(),
        }
    }

    fn place(id: &str, sort_id: &str, marking: Option<ColorExpr>) -> ColoredPlace {
        ColoredPlace {
            id: id.to_string(),
            name: Some(id.to_string()),
            sort_id: sort_id.to_string(),
            initial_marking: marking,
        }
    }

    fn transition(id: &str) -> ColoredTransition {
        ColoredTransition {
            id: id.to_string(),
            name: Some(id.to_string()),
            guard: None,
        }
    }

    fn arc(source: &str, target: &str, inscription: ColorExpr) -> ColoredArc {
        ColoredArc {
            id: format!("{source}_to_{target}"),
            source: source.to_string(),
            target: target.to_string(),
            inscription,
        }
    }

    fn all(sort_id: &str) -> ColorExpr {
        ColorExpr::All {
            sort_id: sort_id.to_string(),
            count: 1,
        }
    }

    fn empty_net() -> ColoredNet {
        ColoredNet {
            name: Some("test".to_string()),
            sorts: vec![],
            variables: vec![],
            places: vec![],
            transitions: vec![],
            arcs: vec![],
        }
    }

    #[test]
    fn shortcut_refuses_on_all_marking_over_multicolor_sort() {
        // Two places, one transition that consumes-and-produces 1-per-color
        // from each place. Initial markings are `All` over a 3-color sort, so
        // each place holds 3 tokens TOTAL (one per color). MCC's colored
        // 1-safe property is group-level (sum across colors), so the group sum
        // is 3 > 1 and the net is NOT 1-safe. The shortcut MUST refuse (return
        // None) so the unfolding group-sum path can issue the correct
        // `Verdict::False`. (Previously this asserted `Some(True)`, encoding
        // the per-instance bug that flipped CollapseAll OneSafe TRUE→FALSE.)
        let mut net = empty_net();
        net.sorts = vec![cyclic_sort("S", 3)];
        net.places = vec![
            place("p1", "S", Some(all("S"))),
            place("p2", "S", Some(all("S"))),
        ];
        net.transitions = vec![transition("t1")];
        net.arcs = vec![
            arc("p1", "t1", all("S")),
            arc("t1", "p1", all("S")),
            arc("p2", "t1", all("S")),
            arc("t1", "p2", all("S")),
        ];

        assert_eq!(
            try_one_safe_colored_shortcut(&net),
            None,
            "`All` over a 3-color sort deposits 3 tokens total per place; the \
             group sum exceeds 1 so the shortcut must fall through, NOT return TRUE"
        );
    }

    #[test]
    fn shortcut_fires_on_balanced_dot_net() {
        // Two Dot places (cardinality 1) each starting with a single token and
        // a transition that consumes-and-produces that token. The group total
        // per place is 1, so the net is genuinely 1-safe and the shortcut may
        // fire. This exercises the "real TRUE is preserved" path after the
        // group-total fix.
        let one_dot = || ColorExpr::NumberOf {
            count: 1,
            color: Box::new(ColorTerm::DotConstant),
        };
        let mut net = empty_net();
        net.sorts = vec![ColorSort::Dot {
            id: "dot".to_string(),
            name: "dot".to_string(),
        }];
        net.places = vec![
            place("p1", "dot", Some(one_dot())),
            place("p2", "dot", Some(one_dot())),
        ];
        net.transitions = vec![transition("t1")];
        net.arcs = vec![
            arc("p1", "t1", one_dot()),
            arc("t1", "p1", one_dot()),
            arc("p2", "t1", one_dot()),
            arc("t1", "p2", one_dot()),
        ];

        assert_eq!(
            try_one_safe_colored_shortcut(&net),
            Some(Verdict::True),
            "balanced Dot net with one token per place is provably 1-safe"
        );
    }

    #[test]
    fn shortcut_refuses_on_marking_above_one_per_color() {
        // A place with initial marking `NumberOf { count: 2, .. }` already
        // violates 1-safety. The shortcut MUST refuse (return None) so the
        // unfolding-based check can issue the correct `Verdict::False`.
        let mut net = empty_net();
        net.sorts = vec![cyclic_sort("S", 3)];
        net.places = vec![place(
            "p1",
            "S",
            Some(ColorExpr::NumberOf {
                count: 2,
                color: Box::new(ColorTerm::DotConstant),
            }),
        )];
        net.transitions = vec![transition("t1")];
        net.arcs = vec![arc("p1", "t1", all("S")), arc("t1", "p1", all("S"))];

        assert_eq!(
            try_one_safe_colored_shortcut(&net),
            None,
            "initial marking > 1 per color must trigger fall-through, NOT a TRUE verdict"
        );
    }

    #[test]
    fn shortcut_refuses_on_all_marking_above_one() {
        // `All { count: 2 }` deposits 2 tokens on every (place, color)
        // instance and must trigger fall-through.
        let mut net = empty_net();
        net.sorts = vec![cyclic_sort("S", 3)];
        net.places = vec![place(
            "p1",
            "S",
            Some(ColorExpr::All {
                sort_id: "S".to_string(),
                count: 2,
            }),
        )];
        net.transitions = vec![transition("t1")];
        net.arcs = vec![arc("p1", "t1", all("S")), arc("t1", "p1", all("S"))];

        assert_eq!(try_one_safe_colored_shortcut(&net), None);
    }

    #[test]
    fn shortcut_refuses_on_pure_producer_transition() {
        // Transition `t1` produces a token to `p1` with no matching input
        // arc. After firing it once, p1 holds 1; firing again would push
        // p1 to 2. The shortcut MUST refuse — even though initial markings
        // are safe — because the arc-balance precondition fails.
        let mut net = empty_net();
        net.sorts = vec![cyclic_sort("S", 3)];
        net.places = vec![place("p1", "S", None)];
        net.transitions = vec![transition("t1")];
        net.arcs = vec![arc("t1", "p1", all("S"))];

        assert_eq!(
            try_one_safe_colored_shortcut(&net),
            None,
            "pure producer breaks the arc-balance precondition"
        );
    }

    #[test]
    fn shortcut_refuses_on_guarded_transition() {
        // A guard introduces binding-dependent firing semantics. Even when
        // arcs are balanced and markings are safe, we refuse to reason
        // about guards in this fast path.
        let mut net = empty_net();
        net.sorts = vec![cyclic_sort("S", 3)];
        net.places = vec![place("p1", "S", Some(all("S")))];
        net.transitions = vec![ColoredTransition {
            id: "t1".to_string(),
            name: Some("t1".to_string()),
            guard: Some(GuardExpr::Equality(
                Box::new(ColorTerm::Variable("x".to_string())),
                Box::new(ColorTerm::Variable("y".to_string())),
            )),
        }];
        net.arcs = vec![arc("p1", "t1", all("S")), arc("t1", "p1", all("S"))];

        assert_eq!(
            try_one_safe_colored_shortcut(&net),
            None,
            "guarded transitions must trigger fall-through"
        );
    }

    #[test]
    fn shortcut_refuses_on_additive_marking() {
        // `Add` markings can deposit > 1 token on a single (place, color)
        // instance (e.g. `1'c + 1'c == 2'c`). The shortcut does not
        // analyse the color terms for non-overlap and must refuse.
        let mut net = empty_net();
        net.sorts = vec![cyclic_sort("S", 3)];
        net.places = vec![place(
            "p1",
            "S",
            Some(ColorExpr::Add(vec![all("S"), all("S")])),
        )];
        net.transitions = vec![transition("t1")];
        net.arcs = vec![arc("p1", "t1", all("S")), arc("t1", "p1", all("S"))];

        assert_eq!(try_one_safe_colored_shortcut(&net), None);
    }

    #[test]
    fn shortcut_consumes_each_input_arc_at_most_once() {
        // Two output arcs `t1 → p1` (both `All`) but only one matching
        // input arc `p1 → t1` (`All`). The second output would have to
        // "share" the single input; we forbid that to preserve the
        // pointwise-dominance invariant. Shortcut must refuse.
        let mut net = empty_net();
        net.sorts = vec![cyclic_sort("S", 3)];
        net.places = vec![place("p1", "S", Some(all("S")))];
        net.transitions = vec![transition("t1")];
        net.arcs = vec![
            arc("p1", "t1", all("S")),
            arc("t1", "p1", all("S")),
            arc("t1", "p1", all("S")), // second producer, no second consumer
        ];

        assert_eq!(try_one_safe_colored_shortcut(&net), None);
    }

    #[test]
    fn shortcut_fires_on_empty_net() {
        // No transitions and no places — vacuously 1-safe.
        let net = empty_net();
        assert_eq!(try_one_safe_colored_shortcut(&net), Some(Verdict::True));
    }
}
