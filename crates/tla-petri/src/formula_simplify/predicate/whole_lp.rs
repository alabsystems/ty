// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tier 6: whole-predicate LP proofs (conjunction-level).

use crate::formula_simplify::SimplificationContext;
use crate::lp_state_equation::{lp_always_true_deadline, lp_unreachable_deadline};
use crate::property_xml::StatePredicate;
use crate::resolved_predicate::{count_unresolved_with_aliases, resolve_predicate_with_aliases};

fn contains_fireability(pred: &StatePredicate) -> bool {
    match pred {
        StatePredicate::IsFireable(_) => true,
        StatePredicate::And(children) | StatePredicate::Or(children) => {
            children.iter().any(contains_fireability)
        }
        StatePredicate::Not(inner) => contains_fireability(inner),
        StatePredicate::IntLe(_, _) | StatePredicate::True | StatePredicate::False => false,
    }
}

/// If the predicate is fully resolved (no unresolved names), try:
/// - `lp_unreachable(φ)` → the predicate is never satisfiable → False
/// - `lp_always_true(φ)` → the predicate always holds → True
///
/// Predicates containing `IsFireable` are intentionally skipped here. The
/// explicit reachability LP phase may still use capped fireability proofs, but
/// formula simplification runs before BMC and must stay cheap for MCC rows.
pub(super) fn lp_prove(pred: &StatePredicate, ctx: &SimplificationContext<'_>) -> StatePredicate {
    if matches!(pred, StatePredicate::True | StatePredicate::False) {
        return pred.clone();
    }

    // Respect the pre-pass wall-cap: an expired budget leaves the predicate
    // unchanged (verdict-preserving) instead of launching another LP solve.
    if ctx.deadline_expired() {
        return pred.clone();
    }

    let (_total_names, unresolved_count) = count_unresolved_with_aliases(pred, ctx.aliases);
    if unresolved_count > 0 {
        return pred.clone();
    }

    if contains_fireability(pred) {
        return pred.clone();
    }

    let resolved = resolve_predicate_with_aliases(pred, ctx.aliases);

    if lp_unreachable_deadline(ctx.net, &resolved, ctx.deadline) {
        return StatePredicate::False;
    }

    if lp_always_true_deadline(ctx.net, &resolved, ctx.deadline) {
        return StatePredicate::True;
    }

    pred.clone()
}
