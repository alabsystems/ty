// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Formula simplification using structural analysis and LP proofs.
//!
//! Rewrites Reachability, CTL, and LTL properties before the expensive
//! evaluators run. Reuses facts the crate already knows how to prove:
//! P-invariant bounds, LP state-equation infeasibility, siphon/trap
//! deadlock-freedom, and per-atom LP truth caching.
//!
//! # Usage
//!
//! ```text
//! let simplified = simplify_properties_with_aliases(&net, &properties, &aliases);
//! ```

mod facts;
mod predicate;
mod temporal;

#[cfg(test)]
mod tests;

pub(crate) use facts::compute_facts;
#[cfg(test)]
pub(crate) use predicate::simplify_predicate;

use std::time::Instant;

use rustc_hash::FxHashMap;

use temporal::simplify_formula_ctx;

use crate::model::PropertyAliases;
use crate::petri_net::PetriNet;
use crate::property_xml::Property;
use crate::resolved_predicate::{
    count_unresolved_with_aliases, resolve_predicate_with_aliases, ResolvedPredicate,
};
use crate::simplification_report::{PropertySimplificationReport, SimplificationReport};

use crate::lp_state_equation::lp_atom_truth_deadline;

/// Mutable simplification context reused across all properties on one net.
pub(crate) struct SimplificationContext<'a> {
    pub(crate) net: &'a PetriNet,
    pub(crate) aliases: &'a PropertyAliases,
    pub(crate) facts: facts::SimplificationFacts,
    /// Wall-clock cap for the simplifier's LP deciders. `None` keeps the
    /// historical un-bounded behaviour (test callers); the production CTL/LTL
    /// pre-passes pass a fraction of the global deadline so the simplifier can
    /// never overrun the budget reserved for the temporal engine.
    pub(crate) deadline: Option<Instant>,
    lp_atom_cache: FxHashMap<ResolvedPredicate, Option<bool>>,
}

impl<'a> SimplificationContext<'a> {
    pub(crate) fn new(
        net: &'a PetriNet,
        aliases: &'a PropertyAliases,
        facts: facts::SimplificationFacts,
    ) -> Self {
        Self::with_deadline(net, aliases, facts, None)
    }

    pub(crate) fn with_deadline(
        net: &'a PetriNet,
        aliases: &'a PropertyAliases,
        facts: facts::SimplificationFacts,
        deadline: Option<Instant>,
    ) -> Self {
        Self {
            net,
            aliases,
            facts,
            deadline,
            lp_atom_cache: FxHashMap::default(),
        }
    }

    /// True once the simplifier pre-pass budget is exhausted. LP-backed tiers
    /// short-circuit on this (leaving the sub-formula unchanged), which is
    /// verdict-preserving — the survivor falls through to the temporal engine.
    pub(crate) fn deadline_expired(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    pub(crate) fn cached_lp_atom_truth(&mut self, atom: &ResolvedPredicate) -> Option<bool> {
        if let Some(&cached) = self.lp_atom_cache.get(atom) {
            return cached;
        }
        if self.deadline_expired() {
            // Do not cache the expiry-driven `None`: a later call with budget
            // restored (different property batch) should still attempt the LP.
            return None;
        }
        let result = lp_atom_truth_deadline(self.net, atom, self.deadline);
        self.lp_atom_cache.insert(atom.clone(), result);
        result
    }

    pub(crate) fn resolve_and_query_atom(
        &mut self,
        pred: &crate::property_xml::StatePredicate,
    ) -> Option<bool> {
        let (_total, unresolved) = count_unresolved_with_aliases(pred, self.aliases);
        if unresolved > 0 {
            return None;
        }
        let resolved = resolve_predicate_with_aliases(pred, self.aliases);
        self.cached_lp_atom_truth(&resolved)
    }

    #[cfg(test)]
    pub(crate) fn lp_atom_cache_len(&self) -> usize {
        self.lp_atom_cache.len()
    }
}

/// Simplify all properties using structural facts from the net.
///
/// Computes per-net facts once, builds a shared mutable context with an LP
/// atom cache, then simplifies each property's formula through that context.
/// Repeated `IntLe` atoms across properties hit the cache instead of
/// re-solving the same LP.
///
/// Returns a new vector of properties with simplified formulas. Property
/// IDs are preserved.
pub(crate) fn simplify_properties_with_aliases(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
) -> Vec<Property> {
    simplify_properties_with_report(net, properties, aliases).simplified_properties
}

/// Deadline-bounded variant of [`simplify_properties_with_aliases`].
///
/// `deadline` caps the whole pre-pass: the per-property loop polls it (any
/// property reached after expiry is copied through unchanged) and each LP
/// decider is run with the same bound. Passing `None` reproduces the
/// historical un-bounded behaviour. Verdict-preserving: a copied-through or
/// LP-timed-out formula is identical-or-weaker to its input, so the temporal
/// engine still decides it exactly.
pub(crate) fn simplify_properties_with_deadline(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    deadline: Option<Instant>,
) -> Vec<Property> {
    simplify_properties_with_report_deadline(net, properties, aliases, deadline)
        .simplified_properties
}

/// Result of simplification with a typed impact report attached.
pub(crate) struct SimplificationRun {
    pub(crate) simplified_properties: Vec<Property>,
    pub(crate) report: SimplificationReport,
}

/// Simplify all properties and produce a typed impact report.
///
/// Same as [`simplify_properties_with_aliases`] but also computes per-property
/// outcome classification (unchanged / simplified / resolved true / false).
pub(crate) fn simplify_properties_with_report(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
) -> SimplificationRun {
    simplify_properties_with_report_deadline(net, properties, aliases, None)
}

/// Deadline-bounded core shared by [`simplify_properties_with_report`] and
/// [`simplify_properties_with_deadline`].
pub(crate) fn simplify_properties_with_report_deadline(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    deadline: Option<Instant>,
) -> SimplificationRun {
    let facts = compute_facts(net);
    let mut ctx = SimplificationContext::with_deadline(net, aliases, facts, deadline);

    let mut simplified_properties = Vec::with_capacity(properties.len());
    let mut reports = Vec::with_capacity(properties.len());

    for prop in properties {
        // Wall-cap: once the pre-pass budget is spent, copy the remaining
        // properties through unchanged rather than starving the temporal
        // engine. An unsimplified formula is still decided exactly downstream.
        let simplified_formula = if ctx.deadline_expired() {
            prop.formula.clone()
        } else {
            simplify_formula_ctx(&prop.formula, &mut ctx)
        };
        reports.push(PropertySimplificationReport::from_formulas(
            &prop.id,
            &prop.formula,
            &simplified_formula,
        ));
        simplified_properties.push(Property {
            id: prop.id.clone(),
            formula: simplified_formula,
        });
    }

    SimplificationRun {
        simplified_properties,
        report: SimplificationReport::from_properties(reports),
    }
}
