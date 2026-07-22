// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Place expansion and initial-marking evaluation for colored-net unfolding.

use std::collections::{HashMap, HashSet};

use crate::error::PnmlError;
use crate::hlpnml::{ColorExpr, ColorSort, ColoredNet};
use crate::petri_net::{PlaceIdx, PlaceInfo};

use super::context::UnfoldContext;
use super::{ColorValue, UnfoldBudget, MAX_UNFOLDED_PLACES};

/// Result of unfolding all colored places.
pub(super) struct PlaceUnfolding {
    pub(super) places: Vec<PlaceInfo>,
    pub(super) initial_marking: Vec<u64>,
    pub(super) place_aliases: HashMap<String, Vec<PlaceIdx>>,
    pub(super) colored_place_group_aliases: HashSet<String>,
    /// Map: (colored_place_id, color_value) → PlaceIdx in unfolded net.
    pub(super) place_map: HashMap<(String, ColorValue), PlaceIdx>,
    pub(super) place_sort_ids: HashMap<String, String>,
}

/// Unfold all colored places into P/T places.
pub(super) fn unfold_places(
    ctx: &UnfoldContext,
    colored: &ColoredNet,
    budget: &UnfoldBudget,
) -> Result<PlaceUnfolding, PnmlError> {
    let mut places = Vec::new();
    let mut initial_marking = Vec::new();
    let mut place_aliases: HashMap<String, Vec<PlaceIdx>> = HashMap::new();
    let mut colored_place_group_aliases = HashSet::new();
    let mut place_map: HashMap<(String, ColorValue), PlaceIdx> = HashMap::new();
    let place_sort_ids: HashMap<String, String> = colored
        .places
        .iter()
        .map(|place| (place.id.clone(), place.sort_id.clone()))
        .collect();

    for cp in &colored.places {
        let sort = ctx.sort_for_place(cp)?;
        let cardinality = ctx.sort_cardinality(sort)?;
        // Cardinality gate BEFORE any materialization (audit 2026-07-02):
        // `sort_value_names` / `eval_initial_marking` / the alias vector
        // below each allocate O(cardinality); a huge (e.g. product-sort)
        // cardinality allocated tens of GB before the per-place cap in the
        // loop could fire. Every color becomes one place, so a sort that
        // cannot fit under MAX_UNFOLDED_PLACES can be declined up front —
        // the same fail-closed error the in-loop cap raises.
        if cardinality > MAX_UNFOLDED_PLACES.saturating_sub(places.len()) {
            return Err(PnmlError::ColoredUnfoldUnavailable {
                reason: format!(
                    "unfolded net exceeds {} places (place '{}' alone has \
                     cardinality {cardinality}; model too large)",
                    MAX_UNFOLDED_PLACES, cp.id
                ),
            });
        }
        let sort_name = sort.name().to_string();
        let constants = ctx.sort_value_names(sort)?;

        let mut alias_indices = Vec::with_capacity(cardinality);

        // Evaluate initial marking per color.
        let marking_per_color = ctx.eval_initial_marking(cp, sort)?;

        for (color_idx, constant_name) in constants.iter().enumerate() {
            let pidx = PlaceIdx(places.len() as u32);

            if places.len() >= MAX_UNFOLDED_PLACES {
                return Err(PnmlError::ColoredUnfoldUnavailable {
                    reason: format!(
                        "unfolded net exceeds {} places (model too large)",
                        MAX_UNFOLDED_PLACES
                    ),
                });
            }
            // Cooperative deadline check, throttled to keep steady-state cost
            // negligible. `places.len()` is the monotone progress counter.
            if places.len() & 0xFFF == 0 {
                budget.check("place expansion")?;
            }

            let unfolded_id = format!("{}_{}", cp.id, constant_name);
            places.push(PlaceInfo {
                id: unfolded_id,
                name: None,
            });

            initial_marking.push(marking_per_color[color_idx]);

            place_map.insert((cp.id.clone(), color_idx), pidx);
            alias_indices.push(pidx);
        }

        // Register aliases: both the place id and name map to all unfolded copies.
        place_aliases.insert(cp.id.clone(), alias_indices.clone());
        if cardinality > 1 {
            colored_place_group_aliases.insert(cp.id.clone());
        }
        if let Some(ref name) = cp.name {
            if name != &cp.id {
                place_aliases.insert(name.clone(), alias_indices.clone());
                if cardinality > 1 {
                    colored_place_group_aliases.insert(name.clone());
                }
            }
        }
        // Also register by sort name for `tokens-count` on sort names —
        // but ONLY when the sort name does not collide with a place
        // id/name bucket. If `sort_name` equals `cp.id`, `cp.name`, or any
        // colored place id, `entry(sort_name)` returns an existing bucket
        // (the place's own alias vector) and `extend_from_slice` would
        // append the same unfolded indices a second time, double-counting
        // every color. That poisoned vector then flows un-deduped into
        // `colored_place_groups` (spurious OneSafe FALSE) and
        // `resolve_place_bound` (doubled UpperBounds). Skip the collision;
        // the place-id/name alias already covers the tokens-count query for
        // that name. The legitimate sort-name aggregate over DISTINCT places
        // (sort name differs from every place id/name) is preserved.
        let collides_with_place = sort_name == cp.id
            || cp.name.as_deref() == Some(sort_name.as_str())
            || place_sort_ids.contains_key(&sort_name);
        if !collides_with_place {
            place_aliases
                .entry(sort_name)
                .or_default()
                .extend_from_slice(&alias_indices);
        }
    }

    Ok(PlaceUnfolding {
        places,
        initial_marking,
        place_aliases,
        colored_place_group_aliases,
        place_map,
        place_sort_ids,
    })
}

// ---------------------------------------------------------------------------
// Marking evaluation (impl UnfoldContext)
// ---------------------------------------------------------------------------

impl UnfoldContext {
    /// Evaluate initial marking per color value.
    pub(super) fn eval_initial_marking(
        &self,
        place: &crate::hlpnml::ColoredPlace,
        sort: &ColorSort,
    ) -> Result<Vec<u64>, PnmlError> {
        let cardinality = self.sort_cardinality(sort)?;
        let mut marking = vec![0u64; cardinality];

        match &place.initial_marking {
            None => {} // No initial marking = all zeros.
            Some(ColorExpr::All { sort_id, count }) => {
                self.validate_all_sort_for_target(sort_id, sort)?;
                marking.fill(*count);
            }
            Some(expr) => {
                self.eval_marking_expr(expr, sort, &mut marking)?;
            }
        }

        Ok(marking)
    }

    /// Evaluate a marking expression, adding tokens to the per-color vector.
    fn eval_marking_expr(
        &self,
        expr: &ColorExpr,
        sort: &ColorSort,
        marking: &mut [u64],
    ) -> Result<(), PnmlError> {
        match expr {
            ColorExpr::All { sort_id, count } => {
                self.validate_all_sort_for_target(sort_id, sort)?;
                for m in marking.iter_mut() {
                    *m += count;
                }
            }
            ColorExpr::NumberOf { count, color } => {
                let empty_binding = HashMap::new();
                let values = self.eval_color_values_for_sort(color, &empty_binding, sort)?;
                if values.is_empty() {
                    // Variable in initial marking — treat as "all" with multiplicity.
                    for m in marking.iter_mut() {
                        *m += count;
                    }
                } else {
                    for idx in values {
                        let slot = marking.get_mut(idx).ok_or_else(|| {
                            PnmlError::InvalidMarking(format!(
                                "initial marking color value {idx} outside place sort '{}'",
                                sort.id()
                            ))
                        })?;
                        *slot += count;
                    }
                }
            }
            ColorExpr::Add(children) => {
                for child in children {
                    self.eval_marking_expr(child, sort, marking)?;
                }
            }
            ColorExpr::Subtract { lhs, rhs } => {
                // Truncated (monus) per-color multiset difference, matching the
                // arc-inscription `Subtract` semantics. Markings are
                // non-negative token counts; subtracting more than is present
                // saturates at zero. Evaluate each side into its own buffer so
                // the subtraction is exact per color, then fold into `marking`.
                let mut lhs_marking = vec![0u64; marking.len()];
                self.eval_marking_expr(lhs, sort, &mut lhs_marking)?;
                let mut rhs_marking = vec![0u64; marking.len()];
                self.eval_marking_expr(rhs, sort, &mut rhs_marking)?;
                for (slot, (l, r)) in marking
                    .iter_mut()
                    .zip(lhs_marking.into_iter().zip(rhs_marking))
                {
                    *slot += l.saturating_sub(r);
                }
            }
        }
        Ok(())
    }
}
