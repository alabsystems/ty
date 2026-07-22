// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Safety classification and tautology detection for PROPERTY entries.
//!
//! Part of #2740: provides the three-way classification (state invariant,
//! action implied, temporal) and pre-BFS tautology detection shared by
//! both the sequential and parallel checker paths.

use crate::check::CheckError;
use crate::eval::EvalCtx;
use crate::liveness::{AstToLive, LiveExpr};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use tla_core::ast::{Expr, OperatorDef};
use tla_core::{Spanned, VarIndex};

use super::property_plan::contains_enabled_standalone;
use super::{
    contains_temporal_standalone, flatten_and_terms_standalone, plan_property_terms,
    PlannedPropertyTerm,
};
use crate::LivenessCheckError;

// ---------------------------------------------------------------------------
// Tautology detection — canonical function shared by sequential and parallel.
//
// Part of #2740: eliminates duplicated `is_trivially_unsatisfiable()` definitions
// in check/model_checker/liveness.rs and parallel/checker/liveness.rs.
// ---------------------------------------------------------------------------

/// Check if a LiveExpr formula is trivially unsatisfiable (tautology detection).
///
/// Used for TLC_LIVE_FORMULA_TAUTOLOGY detection (EC 2253): if the negation
/// of a liveness property is trivially unsatisfiable, the property is a tautology.
/// Detects patterns like `Bool(false)`, `Always(Bool(false))`, `Eventually(Bool(false))`,
/// and conjunctions containing any trivially unsatisfiable term.
///
/// Both the sequential and parallel checker paths MUST use this function for all
/// tautology detection to prevent parity drift.
pub(crate) fn is_trivially_unsatisfiable(expr: &LiveExpr) -> bool {
    match expr {
        LiveExpr::Bool(false) => true,
        LiveExpr::Always(inner) | LiveExpr::Eventually(inner) => is_trivially_unsatisfiable(inner),
        LiveExpr::And(conjuncts) => conjuncts.iter().any(is_trivially_unsatisfiable),
        LiveExpr::Or(disjuncts) => disjuncts.iter().all(is_trivially_unsatisfiable),
        LiveExpr::Not(inner) => matches!(**inner, LiveExpr::Bool(true)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// PROPERTY safety classification — canonical function shared by both checkers.
//
// Part of #2740: the parallel checker was missing state-level PROPERTY promotion.
// This function provides the three-way classification that was previously only
// available on ModelChecker::classify_property_safety_parts.
// ---------------------------------------------------------------------------

pub(crate) struct PropertySafetyClassification {
    /// Canonical action-level PROPERTY checks evaluated via `eval_entry()`.
    pub eval_implied_actions: Vec<EvalImpliedActionTerm>,
    /// Action-level PROPERTY checks whose bodies are eligible for native
    /// next-state predicate lowering. These are also present in
    /// `eval_implied_actions` so non-native BFS keeps the same semantics.
    pub native_implied_actions: Vec<NativeImpliedActionTerm>,
    /// Canonical state-level PROPERTY invariants evaluated via `eval_entry()`.
    pub eval_state_invariants: Vec<(String, Spanned<Expr>)>,
    /// Property names that contribute state-level `PROPERTY` checks during BFS.
    ///
    /// This is broader than `promoted_invariant_properties`: mixed
    /// `[]P /\ <>Q` properties still need PROPERTY attribution if `[]P` fails
    /// during BFS, even though the property is not fully promoted.
    pub state_violation_properties: Vec<String>,
    /// Init predicates: non-Always state/constant-level terms (e.g., `M!Init` in
    /// `M!Init /\ [][M!Next]_M!vars`). Checked against initial states only during BFS.
    /// Part of #2834: without this, properties with init terms are sent to the post-BFS
    /// safety_temporal pass, which iterates over ALL transitions with per-transition
    /// cache clearing — causing hangs for specs like EWD998PCal (2.4M transitions).
    pub init_predicates: Vec<(String, Spanned<Expr>)>,
    /// Properties whose ALL terms were state-level (fully promoted to invariants).
    pub promoted_invariant_properties: Vec<String>,
    /// Properties whose ALL terms were action-level (fully promoted to implied actions).
    pub promoted_action_properties: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct NativeImpliedActionTerm {
    pub name: String,
    pub expr: Spanned<Expr>,
}

impl NativeImpliedActionTerm {
    fn new(name: String, expr: Spanned<Expr>) -> Self {
        Self { name, expr }
    }
}

/// Bytecode-VM fast path for one eval implied-action term.
///
/// The term's expression is compiled to bytecode at prepare time; root-module
/// state-dependent zero-arg operators (refinement mappings like EWD998PCal's
/// `token`/`pending`) are pinned to `CallExternal` interpreter callbacks, so
/// their values stay interpreter-produced (transition-memo served, CHOOSE
/// order preserved by construction) while the boolean skeleton of the
/// disjunction executes in the VM. Only a VM verdict of `Bool(true)` is
/// consumed directly (the same trust boundary as the production invariant
/// bytecode path in `check_invariants_via_bytecode`); `false`, non-boolean,
/// and errors all fall back to a full interpreter evaluation, so every
/// user-visible violation/error is produced by the tree-walker,
/// byte-identically to a run with `TY_NO_IMPLIED_ACTION_BYTECODE=1`.
#[derive(Clone)]
pub(crate) struct EvalImpliedActionVm {
    pub(crate) compiled: std::sync::Arc<tla_eval::bytecode_vm::CompiledBytecode>,
    pub(crate) func_idx: u16,
    /// Set on the first `VmError::Unsupported` so subsequent transitions skip
    /// the VM attempt instead of paying a doomed execution per transition.
    pub(crate) disabled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// TRUE-verdict cache plan for this term (see `implied_verdict_cache`).
    /// `None` when the fail-closed footprint analysis rejected the bytecode
    /// or `TY_NO_IMPLIED_VERDICT_CACHE=1` is set.
    pub(crate) verdict_cache:
        Option<std::sync::Arc<super::implied_verdict_cache::ImpliedVerdictCacheSpec>>,
}

impl std::fmt::Debug for EvalImpliedActionVm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvalImpliedActionVm")
            .field("func_idx", &self.func_idx)
            .field(
                "disabled",
                &self.disabled.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EvalImpliedActionTerm {
    pub(crate) name: String,
    pub(crate) expr: Spanned<Expr>,
    /// Any one trigger set proves the term true; every variable in that set
    /// must be unchanged across the transition.
    pub(crate) truth_if_unchanged: SmallVec<[SmallVec<[VarIndex; 4]>; 2]>,
    /// Optional bytecode-VM fast path, attached at prepare time by
    /// `attach_eval_implied_action_bytecode`. `None` = interpreter only.
    pub(crate) vm: Option<EvalImpliedActionVm>,
}

/// Classify PROPERTY entries into BFS-phase checking buckets (#2332, #2670, #2740).
///
/// Standalone version callable from both sequential and parallel checker setup.
/// Performs the three-way classification:
///
/// 1. **State-level invariants** (`[]P` where P is state/constant level):
///    Checked during BFS for unseen states only.
///
/// 2. **Implied actions** (`[]A` where A is action-level, no nested temporal):
///    Checked during BFS for ALL transitions.
///
/// 3. **Temporal** (WF, SF, <>P, etc.): Left for post-BFS liveness checking.
///
/// Also tracks which properties were *fully* promoted (all terms extracted) so
/// the post-BFS liveness phase can skip them.
pub(crate) fn classify_property_safety_parts(
    ctx: &EvalCtx,
    properties: &[String],
    op_defs: &FxHashMap<String, OperatorDef>,
) -> PropertySafetyClassification {
    let mut eval_action_terms: Vec<EvalImpliedActionTerm> = Vec::new();
    let mut native_action_terms: Vec<NativeImpliedActionTerm> = Vec::new();
    let mut eval_state_terms: Vec<(String, Spanned<Expr>)> = Vec::new();
    let mut state_violation_properties: Vec<String> = Vec::new();
    let mut init_preds: Vec<(String, Spanned<Expr>)> = Vec::new();
    let mut promoted_invariant_names: Vec<String> = Vec::new();
    let mut promoted_action_names: Vec<String> = Vec::new();

    for prop_name in properties {
        if let Some(plan) = plan_property_terms(ctx, op_defs, prop_name) {
            let property_name = plan.property;
            let mut state_term_bodies: Vec<Spanned<Expr>> = Vec::new();
            let mut action_term_bodies: Vec<Spanned<Expr>> = Vec::new();
            let mut eval_action_term_bodies: Vec<Spanned<Expr>> = Vec::new();
            let mut eval_state_term_bodies: Vec<Spanned<Expr>> = Vec::new();
            let mut init_term_bodies: Vec<Spanned<Expr>> = Vec::new();
            let mut has_non_safety_terms = false;

            for term in plan.terms {
                match term {
                    PlannedPropertyTerm::Init(body) => init_term_bodies.push(body),
                    PlannedPropertyTerm::StateCompiled(body) => state_term_bodies.push(body),
                    PlannedPropertyTerm::StateEval(body) => eval_state_term_bodies.push(body),
                    PlannedPropertyTerm::ActionCompiled(body) => action_term_bodies.push(body),
                    PlannedPropertyTerm::ActionEval(body) => eval_action_term_bodies.push(body),
                    PlannedPropertyTerm::Liveness(_) => has_non_safety_terms = true,
                }
            }

            // Part of #3354 Slice 2: route promoted state-level PROPERTY terms
            // through the canonical eval path instead of compiling a second guard IR.
            if !state_term_bodies.is_empty() {
                for body in &state_term_bodies {
                    eval_state_terms.push((prop_name.clone(), body.clone()));
                }
                if !state_violation_properties.contains(prop_name) {
                    state_violation_properties.push(prop_name.clone());
                }
                // Only mark as fully promoted if no other terms remain
                if action_term_bodies.is_empty()
                    && init_term_bodies.is_empty()
                    && !has_non_safety_terms
                {
                    promoted_invariant_names.push(prop_name.clone());
                }
            }

            // Part of #3354 Slice 2: route promoted action-level PROPERTY terms
            // through the canonical eval path instead of compiling a fast-path guard.
            if !action_term_bodies.is_empty() {
                for body in &action_term_bodies {
                    native_action_terms.push(NativeImpliedActionTerm::new(
                        prop_name.clone(),
                        body.clone(),
                    ));
                    eval_action_terms.push(plan_eval_implied_action_term(ctx, prop_name, body));
                }
            }

            // Collect eval-based action terms from ModuleRef properties (#2983).
            if !eval_action_term_bodies.is_empty() {
                for body in &eval_action_term_bodies {
                    eval_action_terms.push(plan_eval_implied_action_term(ctx, prop_name, body));
                }
            }

            // Collect eval-based state invariants for ENABLED-containing terms (#3113).
            if !eval_state_term_bodies.is_empty() {
                for body in &eval_state_term_bodies {
                    eval_state_terms.push((prop_name.clone(), body.clone()));
                }
                if !state_violation_properties.contains(prop_name) {
                    state_violation_properties.push(prop_name.clone());
                }
            }

            // Collect init predicates (#2834)
            if !init_term_bodies.is_empty() {
                for body in &init_term_bodies {
                    init_preds.push((prop_name.clone(), body.clone()));
                }
            }

            // If a property has ALL terms handled (any combination of promoted
            // state/action eval terms and init predicates) and no liveness terms,
            // it's fully promoted.
            // Both promoted lists are used for the post-BFS liveness skip.
            let all_terms_handled = !has_non_safety_terms
                && (!state_term_bodies.is_empty()
                    || !action_term_bodies.is_empty()
                    || !eval_action_term_bodies.is_empty()
                    || !eval_state_term_bodies.is_empty()
                    || !init_term_bodies.is_empty());
            if all_terms_handled {
                if !promoted_invariant_names.contains(&property_name) {
                    promoted_invariant_names.push(property_name.clone());
                }
                if !promoted_action_names.contains(&property_name) {
                    promoted_action_names.push(property_name.clone());
                }
            }
        }
    }

    PropertySafetyClassification {
        eval_implied_actions: eval_action_terms,
        native_implied_actions: native_action_terms,
        eval_state_invariants: eval_state_terms,
        state_violation_properties,
        init_predicates: init_preds,
        promoted_invariant_properties: promoted_invariant_names,
        promoted_action_properties: promoted_action_names,
    }
}

fn plan_eval_implied_action_term(
    ctx: &EvalCtx,
    prop_name: &str,
    body: &Spanned<Expr>,
) -> EvalImpliedActionTerm {
    EvalImpliedActionTerm {
        name: prop_name.to_string(),
        expr: body.clone(),
        truth_if_unchanged: implied_action_truth_triggers(ctx, &body.node),
        vm: None,
    }
}

fn implied_action_truth_triggers(
    ctx: &EvalCtx,
    expr: &Expr,
) -> SmallVec<[SmallVec<[VarIndex; 4]>; 2]> {
    match expr {
        Expr::Implies(antecedent, _) => implied_action_false_triggers(ctx, &antecedent.node),
        Expr::Or(left, right) => {
            let mut triggers = implied_action_truth_triggers(ctx, &left.node);
            triggers.extend(implied_action_truth_triggers(ctx, &right.node));
            triggers
        }
        Expr::Unchanged(inner) => unchanged_trigger_set(ctx, &inner.node)
            .into_iter()
            .collect(),
        _ => same_var_prime_current_cmp(ctx, expr, true)
            .map(single_var_trigger)
            .into_iter()
            .collect(),
    }
}

fn implied_action_false_triggers(
    ctx: &EvalCtx,
    expr: &Expr,
) -> SmallVec<[SmallVec<[VarIndex; 4]>; 2]> {
    match expr {
        Expr::And(left, right) => {
            let mut triggers = implied_action_false_triggers(ctx, &left.node);
            triggers.extend(implied_action_false_triggers(ctx, &right.node));
            triggers
        }
        _ => same_var_prime_current_cmp(ctx, expr, false)
            .map(single_var_trigger)
            .into_iter()
            .collect(),
    }
}

fn single_var_trigger(idx: VarIndex) -> SmallVec<[VarIndex; 4]> {
    let mut trigger = SmallVec::new();
    trigger.push(idx);
    trigger
}

fn unchanged_trigger_set(ctx: &EvalCtx, expr: &Expr) -> Option<SmallVec<[VarIndex; 4]>> {
    match expr {
        Expr::Tuple(elems) => {
            let mut trigger = SmallVec::new();
            for elem in elems {
                let (_, idx) = property_state_var_index(ctx, &elem.node)?;
                if !trigger.contains(&idx) {
                    trigger.push(idx);
                }
            }
            (!trigger.is_empty()).then_some(trigger)
        }
        _ => property_state_var_index(ctx, expr).map(|(_, idx)| single_var_trigger(idx)),
    }
}

fn same_var_prime_current_cmp(ctx: &EvalCtx, expr: &Expr, equality: bool) -> Option<VarIndex> {
    match (equality, expr) {
        (true, Expr::Eq(left, right)) => same_var_prime_current_sides(ctx, &left.node, &right.node)
            .or_else(|| same_var_prime_current_sides(ctx, &right.node, &left.node)),
        (false, Expr::Neq(left, right)) => {
            same_var_prime_current_sides(ctx, &left.node, &right.node)
                .or_else(|| same_var_prime_current_sides(ctx, &right.node, &left.node))
        }
        _ => None,
    }
}

fn same_var_prime_current_sides(
    ctx: &EvalCtx,
    prime_side: &Expr,
    current_side: &Expr,
) -> Option<VarIndex> {
    let Expr::Prime(inner) = prime_side else {
        return None;
    };
    let (prime_name, prime_idx) = property_state_var_index(ctx, &inner.node)?;
    let (current_name, current_idx) = property_state_var_index(ctx, current_side)?;
    (prime_idx == current_idx && prime_name == current_name).then_some(prime_idx)
}

fn property_state_var_index<'a>(ctx: &'a EvalCtx, expr: &'a Expr) -> Option<(&'a str, VarIndex)> {
    match expr {
        Expr::StateVar(name, raw_idx, _) => {
            let idx = VarIndex(*raw_idx);
            if idx.as_usize() < ctx.var_registry().len() && ctx.var_registry().name(idx) == name {
                Some((name.as_str(), idx))
            } else {
                ctx.var_registry()
                    .get(name.as_str())
                    .map(|idx| (name.as_str(), idx))
            }
        }
        Expr::Ident(name, _) => ctx
            .var_registry()
            .get(name.as_str())
            .map(|idx| (name.as_str(), idx)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Pre-BFS tautology detection — shared by both checker paths.
//
// Part of #2740: the parallel checker previously detected tautologies only
// after BFS completion, wasting the entire BFS run. This function enables
// pre-BFS detection matching the sequential checker's behavior.
// ---------------------------------------------------------------------------

/// Check if a conjunct term is a "liveness" term (needs post-BFS checking).
///
/// Approximates the sequential checker's `classify_term` logic in `safety_split.rs`:
/// - `Always(inner)` where inner has no temporal operators and no ENABLED -> safety
/// - `Eventually`, `LeadsTo`, `WeakFair`, `StrongFair` -> liveness
/// - Non-temporal, non-ENABLED terms -> safety (init predicates)
/// - Everything else -> liveness
///
/// **Known gap (P3):** Unlike `classify_term`, this function does NOT resolve
/// `Ident` references through operator definitions (no `classify_ident_term`
/// equivalent). Properties defined as named operator refs (e.g., `Liveness`
/// where `Liveness == <>TRUE`) will be classified by syntactic form only.
/// See test `tautology_via_operator_ref_known_gap`.
///
/// **Known gap (P3):** `ModuleRef` nodes are also not resolved — they fall
/// through to the catch-all and are classified by syntactic form. The sequential
/// checker handles these via `classify_module_ref_term` in `safety_split.rs:291`.
/// See test `is_liveness_conjunct_module_ref_known_gap`.
///
/// `Apply(Ident("WF_xxx"), _)` / `Apply(Ident("SF_xxx"), _)` are treated as
/// liveness because the Apply form can arise from INSTANCE expansion even when
/// the parser normally lowers source fairness syntax to native WF/SF nodes.
pub(crate) fn is_liveness_conjunct(expr: &Expr) -> bool {
    match expr {
        Expr::Always(inner) => {
            // []P is safety only if P has no nested temporal operators and no ENABLED.
            // Otherwise it's liveness (e.g., []<>P, [][ENABLED A => A']_v).
            contains_temporal_standalone(&inner.node) || contains_enabled_standalone(&inner.node)
        }
        Expr::Eventually(_)
        | Expr::LeadsTo(_, _)
        | Expr::WeakFair(_, _)
        | Expr::StrongFair(_, _) => true,
        _ => {
            // Non-temporal terms are init predicates (safety). Terms with
            // temporal operators or ENABLED that aren't one of the above
            // forms are liveness.
            contains_temporal_standalone(expr) || contains_enabled_standalone(expr)
        }
    }
}

/// Check all PROPERTY entries for tautological liveness formulas before BFS.
///
/// TLC rejects tautological liveness properties before state exploration begins
/// (EC 2253). A property like `<>TRUE` negates to `[]FALSE`, which is trivially
/// unsatisfiable, meaning the property can never be violated — it's a tautology.
///
/// For mixed safety+liveness properties (e.g., `[]TypeOK /\ <>TRUE`), the
/// liveness terms are extracted first and only those are checked for tautology.
/// This matches the sequential checker's `check_properties_for_tautologies`
/// which calls `separate_safety_liveness_parts` before tautology analysis.
///
/// Returns `Some(CheckError)` on the first tautological property found, or `None`
/// if all properties are valid.
pub(crate) fn check_property_tautologies(
    ctx: &EvalCtx,
    properties: &[String],
    op_defs: &FxHashMap<String, OperatorDef>,
    root_module_name: &str,
) -> Option<CheckError> {
    if properties.is_empty() {
        return None;
    }

    let converter = AstToLive::new().with_location_module_name(root_module_name);

    for prop_name in properties {
        let def = match op_defs.get(prop_name) {
            Some(d) => d,
            None => continue, // Missing property errors are reported later
        };

        // Flatten the property into conjuncts and extract only liveness terms.
        // This matches the sequential checker's pre-separation via
        // separate_safety_liveness_parts before tautology checking.
        let mut terms = Vec::new();
        flatten_and_terms_standalone(&def.body, &mut terms);

        let liveness_terms: Vec<&Spanned<Expr>> = terms
            .iter()
            .filter(|t| is_liveness_conjunct(&t.node))
            .collect();

        if liveness_terms.is_empty() {
            continue; // Purely safety property, no liveness to check
        }

        // Reconjoin liveness terms for tautology analysis.
        let liveness_expr = if liveness_terms.len() == 1 {
            liveness_terms[0].clone()
        } else {
            let mut iter = liveness_terms.into_iter().cloned();
            let mut result = iter
                .next()
                .expect("invariant: liveness_terms has >= 2 elements");
            for term in iter {
                result = Spanned::new(Expr::And(Box::new(result), Box::new(term)), def.body.span);
            }
            result
        };

        // Convert the liveness portion to LiveExpr for tautology analysis.
        let prop_live = match converter.convert(ctx, &liveness_expr) {
            Ok(live) => live,
            Err(_e) => {
                // Part of #2793: Log conversion errors instead of silently skipping.
                // The actual error is re-reported during liveness checking, but logging
                // here makes the tautology-skip visible for debugging.
                debug_eprintln!(
                    crate::check::debug::ty_debug(),
                    "[property-classify] skipping tautology check for '{}': conversion error: {}",
                    prop_name,
                    _e
                );
                continue;
            }
        };

        let negated = LiveExpr::not(prop_live).push_negation();
        if is_trivially_unsatisfiable(&negated) {
            return Some(
                LivenessCheckError::FormulaTautology {
                    property: prop_name.clone(),
                }
                .into(),
            );
        }
    }

    None
}
