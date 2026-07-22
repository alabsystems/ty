// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tier 1: per-atom LP truth for IntLe leaves.

use crate::formula_simplify::SimplificationContext;
use crate::property_xml::StatePredicate;

/// Resolves individual `IntLe` atoms to `True` or `False` via the cached LP
/// atom truth layer. Fireability stays out of formula simplification so the
/// MCC reachability pipeline preserves its BMC/AIGER/fallback ordering.
pub(super) fn simplify_lp_atom(
    pred: &StatePredicate,
    ctx: &mut SimplificationContext<'_>,
) -> StatePredicate {
    if matches!(pred, StatePredicate::IntLe(..)) {
        if let Some(truth) = ctx.resolve_and_query_atom(pred) {
            return if truth {
                StatePredicate::True
            } else {
                StatePredicate::False
            };
        }
    }
    pred.clone()
}
