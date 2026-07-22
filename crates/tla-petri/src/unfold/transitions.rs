// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Transition expansion, variable discovery, binding enumeration, and arc
//! resolution for colored-net unfolding.

use std::collections::HashMap;

use crate::error::PnmlError;
use crate::hlpnml::{ColorExpr, ColorSort, ColorTerm, ColoredNet, GuardExpr};
use crate::petri_net::{Arc, PlaceIdx, TransitionIdx, TransitionInfo};

use super::context::UnfoldContext;
use super::places::PlaceUnfolding;
use super::{ColorValue, UnfoldBudget, MAX_BINDING_ITERATIONS};

/// Three-valued truth for the binding-quantified guard prune
/// ([`UnfoldContext::guard_prefix_feasible`]). `Unknown` means "a future
/// variable assignment could make this either true or false" — it keeps the
/// prefix feasible (no false negative).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tri {
    True,
    False,
    Unknown,
}

impl Tri {
    fn from_bool(b: bool) -> Self {
        if b {
            Tri::True
        } else {
            Tri::False
        }
    }
    fn negate(self) -> Self {
        match self {
            Tri::True => Tri::False,
            Tri::False => Tri::True,
            Tri::Unknown => Tri::Unknown,
        }
    }
    /// OR-accumulator helper: once a disjunct is Unknown the OR is at least
    /// Unknown (unless a later disjunct is definitely True).
    fn join_unknown(self) -> Self {
        match self {
            Tri::True => Tri::True,
            _ => Tri::Unknown,
        }
    }
}

/// Result of unfolding all colored transitions.
pub(super) struct TransitionUnfolding {
    pub(super) transitions: Vec<TransitionInfo>,
    pub(super) transition_aliases: HashMap<String, Vec<TransitionIdx>>,
}

/// Unfold all colored transitions into P/T transitions.
pub(super) fn unfold_transitions(
    ctx: &UnfoldContext,
    colored: &ColoredNet,
    pu: &PlaceUnfolding,
    budget: &UnfoldBudget,
) -> Result<TransitionUnfolding, PnmlError> {
    let mut transitions = Vec::new();
    let mut transition_aliases: HashMap<String, Vec<TransitionIdx>> = HashMap::new();

    let trans_variables = collect_transition_variables(colored, ctx)?;
    // Pre-compute place id set for arc direction detection.
    let place_ids: std::collections::HashSet<&str> =
        colored.places.iter().map(|p| p.id.as_str()).collect();

    for ct in &colored.transitions {
        let vars = trans_variables.get(&ct.id).cloned().unwrap_or_default();

        let bindings = ctx.enumerate_bindings(&vars, &ct.guard, budget)?;

        let mut alias_indices = Vec::with_capacity(bindings.len());

        for binding in &bindings {
            let tidx = TransitionIdx(transitions.len() as u32);

            if transitions.len() >= super::max_unfolded_transitions() {
                return Err(PnmlError::ColoredUnfoldUnavailable {
                    reason: format!(
                        "unfolded net exceeds {} transitions (model too large)",
                        super::max_unfolded_transitions()
                    ),
                });
            }
            // Cooperative deadline check, throttled. Transition expansion is
            // the dominant cost (per-transition x per-binding x per-arc), so
            // this is where a near-cap model would otherwise stall the whole
            // wall-clock budget at load time.
            if transitions.len() & 0xFFF == 0 {
                budget.check("transition expansion")?;
            }

            let binding_suffix = ctx.binding_suffix(&vars, binding);
            let unfolded_id = format!("{}_{}", ct.id, binding_suffix);

            let inputs = resolve_arcs_for_binding(
                colored,
                &ct.id,
                true,
                binding,
                &vars,
                ctx,
                &pu.place_map,
                &place_ids,
                &pu.place_sort_ids,
            )?;
            let outputs = resolve_arcs_for_binding(
                colored,
                &ct.id,
                false,
                binding,
                &vars,
                ctx,
                &pu.place_map,
                &place_ids,
                &pu.place_sort_ids,
            )?;

            transitions.push(TransitionInfo {
                id: unfolded_id,
                name: None,
                inputs,
                outputs,
            });

            alias_indices.push(tidx);
        }

        transition_aliases.insert(ct.id.clone(), alias_indices.clone());
        if let Some(ref name) = ct.name {
            if name != &ct.id {
                transition_aliases.insert(name.clone(), alias_indices);
            }
        }
    }

    Ok(TransitionUnfolding {
        transitions,
        transition_aliases,
    })
}

// ---------------------------------------------------------------------------
// Binding enumeration and guard evaluation (impl UnfoldContext)
// ---------------------------------------------------------------------------

impl UnfoldContext {
    /// Enumerate all valid variable bindings for a transition.
    pub(super) fn enumerate_bindings(
        &self,
        vars: &[(String, String)], // (var_id, sort_id)
        guard: &Option<GuardExpr>,
        budget: &UnfoldBudget,
    ) -> Result<Vec<Vec<ColorValue>>, PnmlError> {
        if vars.is_empty() {
            return Ok(vec![vec![]]);
        }

        let mut cardinalities = Vec::with_capacity(vars.len());
        for (_, sort_id) in vars {
            let sort = self
                .sorts
                .get(sort_id)
                .ok_or_else(|| PnmlError::MissingElement(format!("sort '{sort_id}' not found")))?;
            cardinalities.push(self.sort_cardinality(sort)?);
        }

        // Wide-net OOM/stall guard. The odometer below visits the FULL
        // Cartesian product of the per-variable sort cardinalities (one guard
        // evaluation per combination), so a transition over a few large sorts
        // can iterate astronomically many combinations and accumulate a
        // gigantic `bindings` Vec before the downstream `MAX_UNFOLDED_
        // TRANSITIONS` cap is ever consulted. Bound the iteration count up
        // front: if the product overflows `usize` or exceeds
        // `MAX_BINDING_ITERATIONS`, decline (recoverable CANNOT_COMPUTE)
        // without allocating or looping.
        let product = cardinalities
            .iter()
            .try_fold(1usize, |acc, &c| acc.checked_mul(c));
        match product {
            Some(p) if p <= MAX_BINDING_ITERATIONS => {}
            _ => {
                return Err(PnmlError::ColoredUnfoldUnavailable {
                    reason: format!(
                        "transition binding space ({} variables, cardinalities {:?}) exceeds \
                         the {MAX_BINDING_ITERATIONS}-combination unfolding budget",
                        vars.len(),
                        cardinalities,
                    ),
                });
            }
        }

        let mut bindings = Vec::new();
        let mut current = vec![0usize; vars.len()];
        let mut iterations: usize = 0;

        loop {
            // Cooperative deadline check, throttled — a restrictive guard can
            // still force a long sweep over the product even when few
            // combinations pass.
            iterations += 1;
            if iterations & 0xFFFF == 0 {
                budget.check("transition binding enumeration")?;
            }

            let var_map: HashMap<&str, ColorValue> = vars
                .iter()
                .zip(current.iter())
                .map(|((var_id, _), &val)| (var_id.as_str(), val))
                .collect();

            if self.eval_guard(guard, &var_map)? {
                bindings.push(current.clone());
                // A single transition cannot usefully yield more than the
                // global unfolded-transition cap; stop accumulating before the
                // Vec itself becomes the OOM (each entry is a Vec<usize>).
                let cap = super::max_unfolded_transitions();
                if bindings.len() > cap {
                    return Err(PnmlError::ColoredUnfoldUnavailable {
                        reason: format!(
                            "transition yields more than {cap} bindings \
                             (model too large to unfold)"
                        ),
                    });
                }
            }

            // Increment counter (odometer style).
            let mut carry = true;
            for i in (0..vars.len()).rev() {
                if carry {
                    current[i] += 1;
                    if current[i] >= cardinalities[i] {
                        current[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }

        Ok(bindings)
    }

    /// Evaluate a guard under a variable binding.
    ///
    /// Soundness: this returns `Result<bool, PnmlError>` rather than a bare
    /// `bool` so that a guard whose operands cannot be resolved to concrete
    /// color values becomes a propagated error (→ `CANNOT_COMPUTE`) instead of
    /// a silent `false`. A silent `false` would drop a semantically-firable
    /// transition from the unfolded P/T net — a *definite* but corrupted net —
    /// and a `Not(<unresolved>)` would silently over-generate bindings. Both
    /// produce catastrophic wrong verdicts (the sibling of the arc-inscription
    /// bug). When an operand genuinely cannot be resolved, fail closed with
    /// `ColoredUnfoldUnavailable` (the recoverable variant the loader maps to
    /// per-examination `CANNOT_COMPUTE`).
    pub(super) fn eval_guard(
        &self,
        guard: &Option<GuardExpr>,
        binding: &HashMap<&str, ColorValue>,
    ) -> Result<bool, PnmlError> {
        match guard {
            None => Ok(true),
            Some(expr) => self.eval_guard_expr(expr, binding),
        }
    }

    /// THREE-VALUED guard feasibility over a PARTIAL binding (the binding-
    /// quantified driver's characteristic prune). `partial` assigns only the
    /// variables decided so far; `assigned` is the set of those variable ids.
    ///
    /// Returns `Ok(false)` ONLY when the guard is `False` under EVERY completion
    /// of `partial` — i.e. it already evaluates to a definite `false` using only
    /// the assigned variables (an atom all of whose variables are assigned and
    /// which evaluates false, propagated through the boolean connectives). If any
    /// deciding variable is still unassigned, the atom is `Unknown` and the guard
    /// stays feasible (`Ok(true)`) — NO false negatives, so no guard-satisfying
    /// binding is ever pruned. Exactness is the leaf's job ([`Self::eval_guard`]).
    ///
    /// A resolution error fails closed (`Err`), exactly like `eval_guard` — never
    /// a silent verdict.
    pub(super) fn guard_prefix_feasible(
        &self,
        guard: &Option<GuardExpr>,
        partial: &HashMap<&str, ColorValue>,
        assigned: &std::collections::HashSet<&str>,
    ) -> Result<bool, PnmlError> {
        match guard {
            None => Ok(true),
            Some(expr) => Ok(self.eval_guard_tri(expr, partial, assigned)? != Tri::False),
        }
    }

    /// Three-valued evaluation of a guard expression under a partial binding.
    fn eval_guard_tri(
        &self,
        expr: &GuardExpr,
        partial: &HashMap<&str, ColorValue>,
        assigned: &std::collections::HashSet<&str>,
    ) -> Result<Tri, PnmlError> {
        match expr {
            GuardExpr::True => Ok(Tri::True),
            GuardExpr::False => Ok(Tri::False),
            GuardExpr::And(children) => {
                let mut acc = Tri::True;
                for child in children {
                    match self.eval_guard_tri(child, partial, assigned)? {
                        // A definite-false conjunct makes the whole AND false
                        // under EVERY completion — the prune.
                        Tri::False => return Ok(Tri::False),
                        Tri::Unknown => acc = Tri::Unknown,
                        Tri::True => {}
                    }
                }
                Ok(acc)
            }
            GuardExpr::Or(children) => {
                let mut acc = Tri::False;
                for child in children {
                    match self.eval_guard_tri(child, partial, assigned)? {
                        Tri::True => return Ok(Tri::True),
                        Tri::Unknown => acc = acc.join_unknown(),
                        Tri::False => {}
                    }
                }
                Ok(acc)
            }
            GuardExpr::Not(inner) => Ok(self.eval_guard_tri(inner, partial, assigned)?.negate()),
            GuardExpr::Equality(_lhs, _rhs)
            | GuardExpr::Inequality(_lhs, _rhs)
            | GuardExpr::LessThan(_lhs, _rhs)
            | GuardExpr::LessThanOrEqual(_lhs, _rhs)
            | GuardExpr::GreaterThan(_lhs, _rhs)
            | GuardExpr::GreaterThanOrEqual(_lhs, _rhs) => {
                // An atom is decidable iff EVERY variable it mentions is already
                // assigned in `partial`; otherwise it is Unknown (a future
                // assignment could make it either true or false). When decidable,
                // evaluate EXACTLY via the same `eval_guard_expr` path so the
                // three-valued atom never disagrees with the leaf guard.
                if self.atom_fully_assigned(expr, assigned) {
                    Ok(Tri::from_bool(self.eval_guard_expr(expr, partial)?))
                } else {
                    Ok(Tri::Unknown)
                }
            }
        }
    }

    /// True iff every variable mentioned by this comparison atom is in
    /// `assigned`. Conservative on shapes whose variables it cannot enumerate
    /// (then NOT fully-assigned ⇒ Unknown ⇒ feasible ⇒ no false negative).
    fn atom_fully_assigned(
        &self,
        expr: &GuardExpr,
        assigned: &std::collections::HashSet<&str>,
    ) -> bool {
        let mut vars: Vec<(String, String)> = Vec::new();
        if collect_vars_from_guard(expr, &mut vars, self).is_err() {
            return false;
        }
        vars.iter().all(|(v, _)| assigned.contains(v.as_str()))
    }

    fn eval_guard_expr(
        &self,
        expr: &GuardExpr,
        binding: &HashMap<&str, ColorValue>,
    ) -> Result<bool, PnmlError> {
        match expr {
            GuardExpr::True => Ok(true),
            GuardExpr::False => Ok(false),
            GuardExpr::And(children) => {
                for child in children {
                    if !self.eval_guard_expr(child, binding)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            GuardExpr::Or(children) => {
                for child in children {
                    if self.eval_guard_expr(child, binding)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            GuardExpr::Not(inner) => Ok(!self.eval_guard_expr(inner, binding)?),
            GuardExpr::Equality(lhs, rhs) => {
                let (l, r) = self.eval_guard_terms(lhs, rhs, binding)?;
                Ok(l == r)
            }
            GuardExpr::Inequality(lhs, rhs) => {
                let (l, r) = self.eval_guard_terms(lhs, rhs, binding)?;
                Ok(l != r)
            }
            GuardExpr::LessThan(lhs, rhs) => {
                let (l, r) = self.eval_guard_terms(lhs, rhs, binding)?;
                Ok(l < r)
            }
            GuardExpr::LessThanOrEqual(lhs, rhs) => {
                let (l, r) = self.eval_guard_terms(lhs, rhs, binding)?;
                Ok(l <= r)
            }
            GuardExpr::GreaterThan(lhs, rhs) => {
                let (l, r) = self.eval_guard_terms(lhs, rhs, binding)?;
                Ok(l > r)
            }
            GuardExpr::GreaterThanOrEqual(lhs, rhs) => {
                let (l, r) = self.eval_guard_terms(lhs, rhs, binding)?;
                Ok(l >= r)
            }
        }
    }

    /// Resolve both operands of a comparison guard to concrete, comparable
    /// color values.
    ///
    /// Resolution must be EXACT: a 1-tuple over a scalar equals the bare
    /// scalar; a tuple-vs-tuple compares component-wise (via the flattened
    /// product index); two integer constants compare numerically. Anything
    /// that cannot be soundly resolved fails closed with an error so the guard
    /// never silently evaluates as `false`/`true`.
    fn eval_guard_terms(
        &self,
        lhs: &ColorTerm,
        rhs: &ColorTerm,
        binding: &HashMap<&str, ColorValue>,
    ) -> Result<(ColorValue, ColorValue), PnmlError> {
        // Two closed integer constants: compare numerically. They carry no
        // sort context (neither `sort_for_term` nor `eval_color_value`
        // resolves a bare `IntegerConstant`), but the boolean is well-defined.
        // Map the (possibly-negative) i64 values to a monotonic `ColorValue`
        // ordering so <, <=, >, >= and == all agree with numeric comparison.
        if let (ColorTerm::IntegerConstant(a), ColorTerm::IntegerConstant(b)) = (lhs, rhs) {
            let (a, b) = Self::normalize_int_constants(*a, *b);
            return Ok((a, b));
        }

        // Derive the comparison sort from whichever operand carries one
        // (a variable's declared sort, a user constant's sort, a dot sort,
        // an `<all>` sort, or a product/scalar context for a tuple). The other
        // operand — a tuple, integer constant, predecessor/successor, … — is
        // then resolved IN that sort via `eval_color_value_for_sort`, which
        // covers the (Product,Tuple) and 1-tuple-over-scalar paths added for
        // the arc fix.
        if let Some(sort) = self.sort_for_term(lhs).or_else(|| self.sort_for_term(rhs)) {
            let left = self.resolve_guard_operand(lhs, binding, sort)?;
            let right = self.resolve_guard_operand(rhs, binding, sort)?;
            return Ok((left, right));
        }

        // Tuple-vs-tuple with no product variable in scope: a guard like
        // `<x, y> = <a, b>` carries no place/variable product sort directly,
        // but each component term has a derivable sort (a variable's declared
        // sort, etc.). Synthesize the product sort from one tuple's component
        // sorts and resolve BOTH tuples in it via the existing
        // `(Product, Tuple)` flatten path, so the comparison is on the exact
        // flattened product index (component-wise equality; product-order for
        // relational ops). Both tuples must share that arity for the index to
        // be comparable.
        if let (ColorTerm::Tuple(lhs_components), ColorTerm::Tuple(rhs_components)) = (lhs, rhs) {
            if lhs_components.len() == rhs_components.len() {
                if let Some(product_sort) = self.derive_product_sort_for_tuple(lhs_components) {
                    let left = self.resolve_guard_operand(lhs, binding, &product_sort)?;
                    let right = self.resolve_guard_operand(rhs, binding, &product_sort)?;
                    return Ok((left, right));
                }
            }
        }

        // No sort context from either side, not two int constants, not a
        // resolvable tuple pair: the only remaining shapes are unknown
        // user/dot constants or arity-mismatched / unsorted tuples. Fall back
        // to the sortless evaluator; if it cannot resolve, fail closed rather
        // than fabricate a verdict.
        let left = self.resolve_guard_operand_sortless(lhs, binding)?;
        let right = self.resolve_guard_operand_sortless(rhs, binding)?;
        Ok((left, right))
    }

    /// Build a synthetic product sort from the sorts of a tuple's component
    /// terms, for resolving a tuple-vs-tuple guard that has no place/variable
    /// product context. Returns `None` (→ caller fails closed) if any
    /// component's sort cannot be determined, so an unresolvable tuple never
    /// silently compares.
    fn derive_product_sort_for_tuple(&self, components: &[ColorTerm]) -> Option<ColorSort> {
        let mut component_ids = Vec::with_capacity(components.len());
        for component in components {
            // A nested tuple would itself need a product sort id, which does
            // not exist as a named component — bail to fail-closed.
            if matches!(component, ColorTerm::Tuple(_)) {
                return None;
            }
            component_ids.push(self.sort_for_term(component)?.id().to_string());
        }
        Some(ColorSort::Product {
            id: String::new(),
            name: String::new(),
            components: component_ids,
        })
    }

    /// Map two i64 integer constants onto a monotonic `ColorValue` (usize)
    /// ordering. Subtracting the smaller value pins the minimum at 0 and keeps
    /// the difference (which fits in u64 for any two i64) representable, so the
    /// usize comparison reproduces the original numeric comparison exactly.
    fn normalize_int_constants(a: i64, b: i64) -> (ColorValue, ColorValue) {
        let lo = a.min(b);
        let da = (a as i128 - lo as i128) as ColorValue;
        let db = (b as i128 - lo as i128) as ColorValue;
        (da, db)
    }

    /// Resolve one comparison operand within a known sort, failing closed.
    fn resolve_guard_operand(
        &self,
        term: &ColorTerm,
        binding: &HashMap<&str, ColorValue>,
        sort: &ColorSort,
    ) -> Result<ColorValue, PnmlError> {
        self.eval_color_value_for_sort(term, binding, sort)?
            .ok_or_else(|| PnmlError::ColoredUnfoldUnavailable {
                reason: format!(
                    "guard operand {term:?} could not be resolved in sort '{}'",
                    sort.id()
                ),
            })
    }

    /// Resolve one comparison operand without sort context, failing closed.
    fn resolve_guard_operand_sortless(
        &self,
        term: &ColorTerm,
        binding: &HashMap<&str, ColorValue>,
    ) -> Result<ColorValue, PnmlError> {
        self.eval_color_value(term, binding)
            .ok_or_else(|| PnmlError::ColoredUnfoldUnavailable {
                reason: format!("guard operand {term:?} could not be resolved to a color value"),
            })
    }

    /// Create a human-readable binding suffix for unfolded transition IDs.
    pub(super) fn binding_suffix(
        &self,
        vars: &[(String, String)],
        binding: &[ColorValue],
    ) -> String {
        if vars.is_empty() {
            return "0".to_string();
        }

        vars.iter()
            .zip(binding.iter())
            .map(|((_, sort_id), &val)| {
                if let Some(sort) = self.sorts.get(sort_id) {
                    self.sort_value_name(sort, val)
                        .unwrap_or_else(|_| val.to_string())
                } else {
                    val.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("_")
    }
}

// ---------------------------------------------------------------------------
// Variable collection
// ---------------------------------------------------------------------------

/// Collect variables used by each transition (from arc inscriptions and guards).
pub(super) fn collect_transition_variables(
    colored: &ColoredNet,
    ctx: &UnfoldContext,
) -> Result<HashMap<String, Vec<(String, String)>>, PnmlError> {
    let mut result: HashMap<String, Vec<(String, String)>> = HashMap::new();

    let trans_ids: std::collections::HashSet<&str> =
        colored.transitions.iter().map(|t| t.id.as_str()).collect();

    for arc in &colored.arcs {
        let trans_id = if trans_ids.contains(arc.source.as_str()) {
            &arc.source
        } else if trans_ids.contains(arc.target.as_str()) {
            &arc.target
        } else {
            continue;
        };

        let entry = result.entry(trans_id.clone()).or_default();
        collect_vars_from_expr(&arc.inscription, entry, ctx)?;
    }

    // Also collect from guards.
    for ct in &colored.transitions {
        if let Some(ref guard) = ct.guard {
            let entry = result.entry(ct.id.clone()).or_default();
            collect_vars_from_guard(guard, entry, ctx)?;
        }
    }

    // Deduplicate while preserving order.
    for vars in result.values_mut() {
        let mut seen = std::collections::HashSet::new();
        vars.retain(|(var_id, _)| seen.insert(var_id.clone()));
    }

    Ok(result)
}

fn collect_vars_from_expr(
    expr: &ColorExpr,
    vars: &mut Vec<(String, String)>,
    ctx: &UnfoldContext,
) -> Result<(), PnmlError> {
    match expr {
        ColorExpr::All { .. } => {}
        ColorExpr::NumberOf { color, .. } => {
            collect_vars_from_term(color, vars, ctx)?;
        }
        ColorExpr::Add(children) => {
            for child in children {
                collect_vars_from_expr(child, vars, ctx)?;
            }
        }
        ColorExpr::Subtract { lhs, rhs } => {
            collect_vars_from_expr(lhs, vars, ctx)?;
            collect_vars_from_expr(rhs, vars, ctx)?;
        }
    }
    Ok(())
}

fn collect_vars_from_term(
    term: &ColorTerm,
    vars: &mut Vec<(String, String)>,
    ctx: &UnfoldContext,
) -> Result<(), PnmlError> {
    match term {
        ColorTerm::Variable(var_id) => {
            let sort_id = ctx.var_sorts.get(var_id).ok_or_else(|| {
                PnmlError::MissingElement(format!("variable '{var_id}' not declared"))
            })?;
            vars.push((var_id.clone(), sort_id.clone()));
        }
        ColorTerm::Tuple(terms) => {
            for term in terms {
                collect_vars_from_term(term, vars, ctx)?;
            }
        }
        ColorTerm::Predecessor(inner) | ColorTerm::Successor(inner) => {
            collect_vars_from_term(inner, vars, ctx)?;
        }
        ColorTerm::UserConstant(_)
        | ColorTerm::IntegerConstant(_)
        | ColorTerm::All(_)
        | ColorTerm::DotConstant => {}
    }
    Ok(())
}

fn collect_vars_from_guard(
    guard: &GuardExpr,
    vars: &mut Vec<(String, String)>,
    ctx: &UnfoldContext,
) -> Result<(), PnmlError> {
    match guard {
        GuardExpr::True | GuardExpr::False => {}
        GuardExpr::And(children) | GuardExpr::Or(children) => {
            for child in children {
                collect_vars_from_guard(child, vars, ctx)?;
            }
        }
        GuardExpr::Not(inner) => collect_vars_from_guard(inner, vars, ctx)?,
        GuardExpr::Equality(lhs, rhs)
        | GuardExpr::Inequality(lhs, rhs)
        | GuardExpr::LessThan(lhs, rhs)
        | GuardExpr::LessThanOrEqual(lhs, rhs)
        | GuardExpr::GreaterThan(lhs, rhs)
        | GuardExpr::GreaterThanOrEqual(lhs, rhs) => {
            collect_vars_from_term(lhs, vars, ctx)?;
            collect_vars_from_term(rhs, vars, ctx)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Arc resolution
// ---------------------------------------------------------------------------

/// Resolve arcs for a specific transition binding.
///
/// `is_input`: true for place→transition arcs, false for transition→place.
pub(super) fn resolve_arcs_for_binding(
    colored: &ColoredNet,
    trans_id: &str,
    is_input: bool,
    binding: &[ColorValue],
    vars: &[(String, String)],
    ctx: &UnfoldContext,
    place_map: &HashMap<(String, ColorValue), PlaceIdx>,
    place_ids: &std::collections::HashSet<&str>,
    place_sort_ids: &HashMap<String, String>,
) -> Result<Vec<Arc>, PnmlError> {
    let var_map: HashMap<&str, ColorValue> = vars
        .iter()
        .zip(binding.iter())
        .map(|((var_id, _), &val)| (var_id.as_str(), val))
        .collect();

    let mut arcs = Vec::new();

    for colored_arc in &colored.arcs {
        let (place_id, matches) = if is_input {
            if colored_arc.target != trans_id {
                continue;
            }
            if !place_ids.contains(colored_arc.source.as_str()) {
                continue;
            }
            (&colored_arc.source, true)
        } else {
            if colored_arc.source != trans_id {
                continue;
            }
            if !place_ids.contains(colored_arc.target.as_str()) {
                continue;
            }
            (&colored_arc.target, true)
        };

        if !matches {
            continue;
        }

        let place_sort_id = place_sort_ids
            .get(place_id)
            .ok_or_else(|| PnmlError::MissingElement(format!("sort for place '{place_id}'")))?;
        let place_sort = ctx
            .sorts
            .get(place_sort_id)
            .ok_or_else(|| PnmlError::MissingElement(format!("sort '{place_sort_id}'")))?;
        let contributions = eval_inscription(&colored_arc.inscription, &var_map, ctx, place_sort)?;

        for (color_val, weight) in contributions {
            if weight == 0 {
                continue;
            }
            let pidx = place_map
                .get(&(place_id.clone(), color_val))
                .copied()
                .ok_or_else(|| {
                    PnmlError::InvalidMarking(format!(
                        "arc '{}' contributes color value {color_val} outside place '{place_id}' sort '{place_sort_id}'",
                        colored_arc.id
                    ))
                })?;
            if let Some(existing) = arcs.iter_mut().find(|a: &&mut Arc| a.place == pidx) {
                existing.weight += weight;
            } else {
                arcs.push(Arc {
                    place: pidx,
                    weight,
                });
            }
        }
    }

    Ok(arcs)
}

/// Evaluate an inscription expression under a binding.
///
/// Returns a list of (color_value, weight) pairs.
fn eval_inscription(
    expr: &ColorExpr,
    binding: &HashMap<&str, ColorValue>,
    ctx: &UnfoldContext,
    target_sort: &ColorSort,
) -> Result<Vec<(ColorValue, u64)>, PnmlError> {
    match expr {
        ColorExpr::All { sort_id, count } => {
            ctx.validate_all_sort_for_target(sort_id, target_sort)?;
            Ok((0..ctx.sort_cardinality(target_sort)?)
                .map(|i| (i, *count))
                .collect())
        }
        ColorExpr::NumberOf { count, color } => Ok(ctx
            .eval_color_values_for_sort(color, binding, target_sort)?
            .into_iter()
            .map(|value| (value, *count))
            .collect()),
        ColorExpr::Add(children) => {
            let mut result = Vec::new();
            for child in children {
                let child_result = eval_inscription(child, binding, ctx, target_sort)?;
                for (color_val, weight) in child_result {
                    if let Some(existing) = result
                        .iter_mut()
                        .find(|(c, _): &&mut (ColorValue, u64)| *c == color_val)
                    {
                        existing.1 += weight;
                    } else {
                        result.push((color_val, weight));
                    }
                }
            }
            Ok(result)
        }
        ColorExpr::Subtract { lhs, rhs } => {
            // Multiset difference with truncated (monus) per-color semantics:
            // `result(c) = max(0, lhs(c) - rhs(c))`. Arc weights are
            // non-negative, so removing more of a color than the left operand
            // supplies yields zero, not a negative weight. The "broadcast to
            // all-but-self" pattern (`1'all - 1'(x)`) cancels exactly one
            // color per binding.
            let mut result = eval_inscription(lhs, binding, ctx, target_sort)?;
            let rhs_result = eval_inscription(rhs, binding, ctx, target_sort)?;
            for (color_val, sub_weight) in rhs_result {
                if let Some(existing) = result
                    .iter_mut()
                    .find(|(c, _): &&mut (ColorValue, u64)| *c == color_val)
                {
                    existing.1 = existing.1.saturating_sub(sub_weight);
                }
                // A color present only in `rhs` removes nothing from `lhs`
                // (truncated subtraction); it contributes no negative weight.
            }
            Ok(result)
        }
    }
}
