// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Dead-transition removal for colored Petri nets.
//!
//! Removes transitions that can never fire because they have an input
//! place that can never hold tokens. A place is *permanently empty* if
//! it has no initial tokens AND no incoming arc from any live transition.
//!
//! This is computed as a fixpoint: removing dead transitions can make
//! additional places permanently empty, which can make more transitions
//! dead.
//!
//! This is query-independent and runs at load time before unfolding.
//!
//! Part of #1438 (colored reduction Tier 1).

use std::collections::HashSet;

use crate::hlpnml::{ColorExpr, ColoredNet};

/// Remove transitions whose input places can never hold tokens.
///
/// A place is permanently empty if it has no initial tokens and no
/// incoming arcs from live (non-dead) transitions. A transition is
/// dead if any input place is permanently empty. Computed to fixpoint.
///
/// Returns the number of transitions removed.
pub(crate) fn reduce(net: &mut ColoredNet) -> DeadTransitionReport {
    let trans_id_set: HashSet<&str> = net.transitions.iter().map(|t| t.id.as_str()).collect();

    let mut dead_transitions: HashSet<String> = HashSet::new();

    loop {
        // Compute permanently empty places: no initial tokens AND
        // no incoming arc from a live transition.
        let empty_places: HashSet<&str> = net
            .places
            .iter()
            .filter(|p| initial_marking_is_empty(p.initial_marking.as_ref()))
            .filter(|p| {
                // Check if any live transition produces tokens into this place.
                !net.arcs.iter().any(|arc| {
                    arc.target == p.id
                        && trans_id_set.contains(arc.source.as_str())
                        && !dead_transitions.contains(&arc.source)
                })
            })
            .map(|p| p.id.as_str())
            .collect();

        if empty_places.is_empty() {
            break;
        }

        // Find transitions with at least one input arc from an empty place.
        let prev_dead = dead_transitions.len();
        for trans in &net.transitions {
            if dead_transitions.contains(&trans.id) {
                continue;
            }
            let has_empty_input = net
                .arcs
                .iter()
                .any(|arc| arc.target == trans.id && empty_places.contains(arc.source.as_str()));
            if has_empty_input {
                dead_transitions.insert(trans.id.clone());
            }
        }

        // No new dead transitions found — fixpoint reached.
        if dead_transitions.len() == prev_dead {
            break;
        }
    }

    if dead_transitions.is_empty() {
        return DeadTransitionReport::default();
    }

    let removed = dead_transitions.len();

    // Remove dead transitions and their arcs.
    net.transitions
        .retain(|t| !dead_transitions.contains(&t.id));
    net.arcs
        .retain(|a| !dead_transitions.contains(&a.source) && !dead_transitions.contains(&a.target));

    DeadTransitionReport {
        transitions_removed: removed,
    }
}

fn initial_marking_is_empty(expr: Option<&ColorExpr>) -> bool {
    match expr {
        None => true,
        Some(expr) => color_expr_is_zero(expr),
    }
}

fn color_expr_is_zero(expr: &ColorExpr) -> bool {
    match expr {
        ColorExpr::All { count, .. } | ColorExpr::NumberOf { count, .. } => *count == 0,
        ColorExpr::Add(children) => children.iter().all(color_expr_is_zero),
        // A truncated difference is zero whenever the left operand is zero
        // (the right operand can only remove tokens). Conservative: a
        // non-zero `lhs` reports non-zero even if `rhs` happens to cancel it
        // for some binding — never spuriously marks a transition dead.
        ColorExpr::Subtract { lhs, .. } => color_expr_is_zero(lhs),
    }
}

/// Report from dead-transition removal.
#[derive(Debug, Default)]
pub(crate) struct DeadTransitionReport {
    pub transitions_removed: usize,
}
