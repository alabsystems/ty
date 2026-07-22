// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Main recursive dispatch for unified successor enumeration.
//!
//! `enumerate_unified_inner` is the core recursive function that handles all
//! TLA+ expression types (Or, And, Exists, If, Let, Case, Apply, Ident,
//! ModuleRef, and catch-all guard/assignment). It dispatches to specialized
//! handlers for each expression type.
//!
//! Extracted from unified.rs as part of #2360.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use smallvec::SmallVec;
use tla_core::ast::{Expr, Substitution};
use tla_core::{get_primed_var_refs_spanned_v, Spanned, VarIndex};

use crate::error::EvalError;
use crate::eval::{apply_substitutions, EvalCtx};
use crate::state::{ArrayState, DiffChanges, DiffSuccessor};
use crate::Value;

use super::action_validation::action_holds_in_next_state_array;
use super::build_diff::build_successor_diffs_from_array_into;
use super::expr_analysis::{
    expr_is_action_level, flatten_and_spanned, is_operator_reference_guard_unsafe,
    might_need_prime_binding,
};
use super::guard_check::check_and_guards;
use super::symbolic_assignments::{
    evaluate_symbolic_assignments, extract_symbolic_assignments_with_registry,
};
use super::tir_leaf::eval_leaf;
use super::unified::enumerate_conjuncts;
use super::unified_exists::enumerate_exists;
use super::unified_module_ref::enumerate_module_ref;
use super::unified_scope::try_let_guard_first_shortcircuit;
use super::unified_types::{Cont, EnumMut, EnumParams, RecState};
use super::{
    and_guard_precheck, case_guard_error, debug_enum, enabled_early_exit, is_let_lazy_safe_error,
};

thread_local! {
    static STATE_INDEPENDENT_BRANCH_ELIGIBLE: RefCell<HashMap<usize, bool>> =
        RefCell::new(HashMap::new());
    static STATE_INDEPENDENT_BRANCH_RESULTS: RefCell<HashMap<usize, Arc<[ArrayState]>>> =
        RefCell::new(HashMap::new());
}

/// Clear raw-AST-pointer branch caches between independently parsed modules.
/// Pointer identity is stable only for one live syntax tree; allocator reuse
/// across runs must never replay another action's eligibility or successors.
pub(crate) fn clear_state_independent_branch_caches() {
    STATE_INDEPENDENT_BRANCH_ELIGIBLE.with(|cache| cache.borrow_mut().clear());
    STATE_INDEPENDENT_BRANCH_RESULTS.with(|cache| cache.borrow_mut().clear());
    super::first_guard_sched::clear_first_guard_sched_cache();
    super::complete_action_filter::clear_complete_action_filter_cache();
    super::subset_constrained::clear_quorum_subset_syntax_cache();
}

/// Number of leading conjuncts that are pure, action-free guards which the
/// whole-action [`check_and_guards`] precheck has already validated as true.
///
/// This is a *purely syntactic* count (no evaluation, no side effects): a
/// conjunct counts only if it is not action-level and does not reference an
/// operator whose body could hide action content. The successor enumerator can
/// therefore start at this index and avoid re-evaluating these guards, which is
/// a strict performance win and does not change which successors are produced.
///
/// Stops at the first conjunct that is action-level or operator-reference-unsafe,
/// because everything from there on must flow through the ordered enumerator.
fn leading_guard_prefix_len(ctx: &EvalCtx, conjuncts: &[&Spanned<Expr>]) -> usize {
    let mut idx = 0;
    while idx < conjuncts.len() {
        let conjunct = conjuncts[idx];
        if expr_is_action_level(ctx, conjunct) || is_operator_reference_guard_unsafe(ctx, conjunct)
        {
            break;
        }
        idx += 1;
    }
    idx
}

fn flatten_action_or_spanned<'a>(
    ctx: &EvalCtx,
    expr: &'a Spanned<Expr>,
    out: &mut SmallVec<[&'a Spanned<Expr>; 32]>,
) {
    if let Expr::Or(left, right) = &expr.node {
        if expr_is_action_level(ctx, expr) {
            flatten_action_or_spanned(ctx, left, out);
            flatten_action_or_spanned(ctx, right, out);
            return;
        }
    }
    out.push(expr);
}

fn replay_or_cache_state_independent_branch(
    ctx: &mut EvalCtx,
    branch: &Spanned<Expr>,
    p: &EnumParams<'_>,
    s: &mut RecState<'_>,
) -> Result<bool, EvalError> {
    if p.full_mask == 0 || ctx.local_stack_len() != 0 {
        return Ok(false);
    }

    let key = branch as *const Spanned<Expr> as usize;
    let eligible = STATE_INDEPENDENT_BRANCH_ELIGIBLE.with(|cache| {
        let mut cache = cache.borrow_mut();
        *cache.entry(key).or_insert_with(|| {
            let mut bound = SmallVec::<[Arc<str>; 8]>::new();
            state_independent_full_assignment_mask(ctx, branch, p, &mut bound, 0)
                == Some(p.full_mask)
        })
    });
    if !eligible {
        return Ok(false);
    }

    if let Some(cached) =
        STATE_INDEPENDENT_BRANCH_RESULTS.with(|cache| cache.borrow().get(&key).cloned())
    {
        // #frame-fp-pop: these replayed emissions do NOT flow through
        // note_emission (they were recorded on an EARLIER state's enumeration
        // and are re-pushed verbatim). If any provenance frame is active,
        // its fingerprint records can no longer be proven complete — poison
        // them fail-closed. (Witness recording was already skipped on this
        // path before #frame-fp-pop; the TRUE side merely loses witnesses.)
        crate::liveness::enabled_provenance::note_unattributed_emission();
        for successor in cached.iter() {
            let diff = diff_absolute_successor(p.base_with_fp, successor, p);
            if s.results.push_with_ctx(ctx, diff).is_break() {
                break;
            }
        }
        return Ok(true);
    }

    let mut local_results = Vec::new();
    {
        let mut local_state = RecState {
            working: s.working,
            undo: s.undo,
            results: &mut local_results,
        };
        enumerate_unified_inner(ctx, branch, p, &mut local_state)?;
    }

    let absolute_successors: Arc<[ArrayState]> = local_results
        .iter()
        .map(|diff| diff.materialize(p.base_with_fp, p.registry))
        .collect::<Vec<_>>()
        .into();
    STATE_INDEPENDENT_BRANCH_RESULTS.with(|cache| {
        cache
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| Arc::clone(&absolute_successors));
    });

    for diff in local_results {
        if s.results.push_with_ctx(ctx, diff).is_break() {
            break;
        }
    }
    Ok(true)
}

fn diff_absolute_successor(
    base: &ArrayState,
    successor: &ArrayState,
    p: &EnumParams<'_>,
) -> DiffSuccessor {
    let mut changes: DiffChanges = SmallVec::new();
    for idx_usize in 0..p.vars.len() {
        let idx = VarIndex::new(idx_usize);
        if base.get_compact(idx) != successor.get_compact(idx) {
            changes.push((idx, successor.get(idx)));
        }
    }
    if changes.is_empty() {
        DiffSuccessor::from_smallvec(
            base.cached_fingerprint()
                .expect("base state missing fingerprint cache in branch replay"),
            changes,
        )
    } else {
        DiffSuccessor::from_changes(changes)
    }
}

fn state_independent_full_assignment_mask(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    p: &EnumParams<'_>,
    bound: &mut SmallVec<[Arc<str>; 8]>,
    depth: usize,
) -> Option<u64> {
    if depth > 32 {
        return None;
    }

    match &expr.node {
        Expr::And(left, right) => {
            let left_mask = state_independent_full_assignment_mask(ctx, left, p, bound, depth)?;
            let right_mask = state_independent_full_assignment_mask(ctx, right, p, bound, depth)?;
            Some(left_mask | right_mask)
        }
        Expr::Exists(bounds, body) => {
            let mark = bound.len();
            for bound_var in bounds {
                let domain = bound_var.domain.as_ref()?;
                if bound_var.pattern.is_some()
                    || !state_independent_value_expr(ctx, domain, p, bound, depth + 1)
                {
                    bound.truncate(mark);
                    return None;
                }
                bound.push(Arc::from(bound_var.name.node.as_str()));
            }
            let result = state_independent_full_assignment_mask(ctx, body, p, bound, depth + 1);
            bound.truncate(mark);
            result
        }
        Expr::Apply(op_expr, args) => {
            let Expr::Ident(op_name, _) = &op_expr.node else {
                return state_independent_value_expr(ctx, expr, p, bound, depth).then_some(0);
            };
            let resolved = ctx.resolve_op_name(op_name.as_str());
            let Some(def) = ctx.get_op(resolved) else {
                return args
                    .iter()
                    .all(|arg| state_independent_value_expr(ctx, arg, p, bound, depth + 1))
                    .then_some(0);
            };
            if def.params.len() != args.len()
                || !args
                    .iter()
                    .all(|arg| state_independent_value_expr(ctx, arg, p, bound, depth + 1))
            {
                return None;
            }
            let mark = bound.len();
            for param in &def.params {
                bound.push(Arc::from(param.name.node.as_str()));
            }
            let result =
                state_independent_full_assignment_mask(ctx, &def.body, p, bound, depth + 1);
            bound.truncate(mark);
            result
        }
        Expr::Ident(name, _) => {
            let resolved = ctx.resolve_op_name(name.as_str());
            let Some(def) = ctx.get_op(resolved) else {
                return state_independent_value_expr(ctx, expr, p, bound, depth).then_some(0);
            };
            if !def.params.is_empty() {
                return None;
            }
            state_independent_full_assignment_mask(ctx, &def.body, p, bound, depth + 1)
        }
        Expr::Eq(left, right) => {
            if let Some(idx) = direct_primed_assignment_target(&left.node, p) {
                return state_independent_assignment_rhs(ctx, right, p, bound, depth + 1)
                    .then_some(1u64 << idx.as_usize());
            }
            if let Some(idx) = direct_primed_assignment_target(&right.node, p) {
                return state_independent_assignment_rhs(ctx, left, p, bound, depth + 1)
                    .then_some(1u64 << idx.as_usize());
            }
            state_independent_value_expr(ctx, expr, p, bound, depth).then_some(0)
        }
        Expr::Unchanged(_) | Expr::Prime(_) | Expr::Or(_, _) | Expr::If(_, _, _) => None,
        _ => state_independent_value_expr(ctx, expr, p, bound, depth).then_some(0),
    }
}

fn direct_primed_assignment_target(expr: &Expr, p: &EnumParams<'_>) -> Option<VarIndex> {
    let Expr::Prime(inner) = expr else {
        return None;
    };
    match &inner.node {
        Expr::Ident(name, name_id) => {
            if *name_id != tla_core::NameId::INVALID {
                if let Some(idx) = p.registry.get_by_name_id(*name_id) {
                    return Some(idx);
                }
            }
            p.registry.get(name.as_str())
        }
        Expr::StateVar(_, raw_idx, _) => Some(VarIndex(*raw_idx)),
        _ => None,
    }
}

fn state_independent_assignment_rhs(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    p: &EnumParams<'_>,
    bound: &mut SmallVec<[Arc<str>; 8]>,
    depth: usize,
) -> bool {
    !matches!(expr.node, Expr::Prime(_)) && state_independent_value_expr(ctx, expr, p, bound, depth)
}

// `ctx` is threaded through this recursion (and its sibling callers) only to be
// passed down; it is kept in the signature for symmetry with the surrounding
// state-independence helpers and to avoid a wide churn across every recursive
// call site, so the lint is allowed rather than dropping the parameter.
#[allow(clippy::only_used_in_recursion)]
fn state_independent_value_expr(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    p: &EnumParams<'_>,
    bound: &mut SmallVec<[Arc<str>; 8]>,
    depth: usize,
) -> bool {
    if depth > 32 {
        return false;
    }

    match &expr.node {
        Expr::Ident(name, _) => {
            bound
                .iter()
                .any(|bound_name| bound_name.as_ref() == name.as_str())
                || p.registry.get(name.as_str()).is_none()
        }
        Expr::StateVar(..) | Expr::Prime(_) | Expr::Unchanged(_) => false,
        Expr::Apply(op, args) => {
            state_independent_value_expr(ctx, op, p, bound, depth + 1)
                && args
                    .iter()
                    .all(|arg| state_independent_value_expr(ctx, arg, p, bound, depth + 1))
        }
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            let mark = bound.len();
            for bound_var in bounds {
                let Some(domain) = bound_var.domain.as_ref() else {
                    bound.truncate(mark);
                    return false;
                };
                if bound_var.pattern.is_some()
                    || !state_independent_value_expr(ctx, domain, p, bound, depth + 1)
                {
                    bound.truncate(mark);
                    return false;
                }
                bound.push(Arc::from(bound_var.name.node.as_str()));
            }
            let result = state_independent_value_expr(ctx, body, p, bound, depth + 1);
            bound.truncate(mark);
            result
        }
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => true,
        Expr::Not(a) | Expr::Powerset(a) | Expr::BigUnion(a) | Expr::Domain(a) | Expr::Neg(a) => {
            state_independent_value_expr(ctx, a, p, bound, depth + 1)
        }
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::FuncApply(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b) => {
            state_independent_value_expr(ctx, a, p, bound, depth + 1)
                && state_independent_value_expr(ctx, b, p, bound, depth + 1)
        }
        Expr::SetEnum(items) | Expr::Tuple(items) | Expr::Times(items) => items
            .iter()
            .all(|item| state_independent_value_expr(ctx, item, p, bound, depth + 1)),
        Expr::FuncSet(domain, range) => {
            state_independent_value_expr(ctx, domain, p, bound, depth + 1)
                && state_independent_value_expr(ctx, range, p, bound, depth + 1)
        }
        Expr::RecordSet(fields) => fields
            .iter()
            .all(|(_, value)| state_independent_value_expr(ctx, value, p, bound, depth + 1)),
        Expr::SetBuilder(element, bounds) | Expr::FuncDef(bounds, element) => {
            let mark = bound.len();
            for bound_var in bounds {
                let Some(domain) = bound_var.domain.as_ref() else {
                    bound.truncate(mark);
                    return false;
                };
                if bound_var.pattern.is_some()
                    || !state_independent_value_expr(ctx, domain, p, bound, depth + 1)
                {
                    bound.truncate(mark);
                    return false;
                }
                bound.push(Arc::from(bound_var.name.node.as_str()));
            }
            let result = state_independent_value_expr(ctx, element, p, bound, depth + 1);
            bound.truncate(mark);
            result
        }
        Expr::SetFilter(bound_var, predicate) => {
            let mark = bound.len();
            let Some(domain) = bound_var.domain.as_ref() else {
                return false;
            };
            if bound_var.pattern.is_some()
                || !state_independent_value_expr(ctx, domain, p, bound, depth + 1)
            {
                return false;
            }
            bound.push(Arc::from(bound_var.name.node.as_str()));
            let result = state_independent_value_expr(ctx, predicate, p, bound, depth + 1);
            bound.truncate(mark);
            result
        }
        Expr::Record(fields) => fields
            .iter()
            .all(|(_, value)| state_independent_value_expr(ctx, value, p, bound, depth + 1)),
        Expr::RecordAccess(base, _) => state_independent_value_expr(ctx, base, p, bound, depth + 1),
        Expr::If(cond, then_expr, else_expr) => {
            state_independent_value_expr(ctx, cond, p, bound, depth + 1)
                && state_independent_value_expr(ctx, then_expr, p, bound, depth + 1)
                && state_independent_value_expr(ctx, else_expr, p, bound, depth + 1)
        }
        Expr::Case(arms, other) => {
            arms.iter().all(|arm| {
                state_independent_value_expr(ctx, &arm.guard, p, bound, depth + 1)
                    && state_independent_value_expr(ctx, &arm.body, p, bound, depth + 1)
            }) && other.as_ref().map_or(true, |default| {
                state_independent_value_expr(ctx, default, p, bound, depth + 1)
            })
        }
        Expr::Except(base, specs) => {
            state_independent_value_expr(ctx, base, p, bound, depth + 1)
                && specs.iter().all(|spec| {
                    spec.path.iter().all(|element| match element {
                        tla_core::ast::ExceptPathElement::Index(index) => {
                            state_independent_value_expr(ctx, index, p, bound, depth + 1)
                        }
                        tla_core::ast::ExceptPathElement::Field(_) => true,
                    }) && state_independent_value_expr(ctx, &spec.value, p, bound, depth + 1)
                })
        }
        Expr::Let(_, _)
        | Expr::Choose(_, _)
        | Expr::Lambda(_, _)
        | Expr::Label(_)
        | Expr::ModuleRef(_, _, _)
        | Expr::InstanceExpr(_, _)
        | Expr::Always(_)
        | Expr::Eventually(_)
        | Expr::LeadsTo(_, _)
        | Expr::WeakFair(_, _)
        | Expr::StrongFair(_, _)
        | Expr::Enabled(_)
        | Expr::SubstIn(_, _) => false,
    }
}

/// Inner recursive dispatch for `enumerate_unified`.
pub(super) fn enumerate_unified_inner(
    ctx: &mut EvalCtx,
    expr: &Spanned<Expr>,
    p: &EnumParams<'_>,
    s: &mut RecState<'_>,
) -> Result<(), EvalError> {
    enumerate_unified_inner_with_certificate(ctx, expr, p, s, None)
}

fn enumerate_unified_inner_with_certificate(
    ctx: &mut EvalCtx,
    expr: &Spanned<Expr>,
    p: &EnumParams<'_>,
    s: &mut RecState<'_>,
    complete_action: Option<super::complete_action_filter::CompleteActionCertificate>,
) -> Result<(), EvalError> {
    crate::eval::stack_safe(|| {
        let debug = debug_enum();

        if debug {
            eprintln!(
                "enumerate_unified: expr type={:?}",
                std::mem::discriminant(&expr.node)
            );
        }

        match &expr.node {
            // Label: transparent wrapper — unwrap and recurse into body.
            Expr::Label(label) => {
                enumerate_unified_inner_with_certificate(ctx, &label.body, p, s, complete_action)
            }

            // Disjunction: try each branch, accumulate results.
            // Clone-at-branch pattern (#2834): each branch gets a cloned EvalCtx,
            // guaranteeing ctx is never mutated by either branch.
            Expr::Or(a, b) => {
                let save_point = s.undo.len();
                // Flatten nested action-level Or trees into a single branch list so the
                // state-independent-branch replay cache can key on the leaf disjuncts.
                let mut branches: SmallVec<[&Spanned<Expr>; 32]> = SmallVec::new();
                flatten_action_or_spanned(ctx, a, &mut branches);
                flatten_action_or_spanned(ctx, b, &mut branches);

                // Part of #3893: mark/restore replaces ctx.clone() per branch.
                let enum_mark = ctx.mark_enum();

                for branch in branches {
                    // Part of #3923: PlusCal pc-guard hoisting.
                    // When pc_guard_hoist is active, skip branches whose `pc = "label"`
                    // guard does not match the current state — they yield zero successors.
                    let skip_branch = p.pc_guard_hoist.as_ref().is_some_and(|h| {
                        crate::checker_ops::pc_dispatch::or_branch_pc_guard_mismatches_cached(
                            &branch.node,
                            &h.current_pc,
                            ctx,
                            &h.pc_guard_label_cache,
                        )
                    });
                    if skip_branch {
                        continue;
                    }

                    if replay_or_cache_state_independent_branch(ctx, branch, p, s)? {
                        ctx.pop_to_enum_mark(&enum_mark);
                        s.working.unbind_to_no_invalidate(s.undo, save_point);

                        // Part of #1285: ENABLED early-exit — stop once we found a successor.
                        if enabled_early_exit() && s.results.has_results() {
                            return Ok(());
                        }
                        // Part of #3027: Early termination — stop if the sink halted.
                        if s.results.is_stopped() {
                            return Ok(());
                        }
                        continue;
                    }

                    let branch_result = enumerate_unified_inner(ctx, branch, p, s);
                    ctx.pop_to_enum_mark(&enum_mark);
                    s.working.unbind_to_no_invalidate(s.undo, save_point);
                    branch_result?;

                    // Part of #1285: ENABLED early-exit — skip remaining branches.
                    if enabled_early_exit() && s.results.has_results() {
                        return Ok(());
                    }
                    // Part of #3027: Early termination — skip remaining branches.
                    if s.results.is_stopped() {
                        return Ok(());
                    }
                }

                Ok(())
            }

            // Conjunction: flatten and process via continuation-based enumeration.
            Expr::And(_, _) => {
                // Check guards before enumeration (TY-specific optimization, no TLC analog).
                // Correctness depends on check_and_guards correctly identifying disabled
                // actions. This runs on the whole action so that all error semantics —
                // CASE-guard wrapping (#1425), action-level error propagation (#1467) —
                // are preserved exactly as on the legacy path.
                if and_guard_precheck() && !check_and_guards(ctx, expr, debug, p.tir_leaf)? {
                    if debug {
                        eprintln!("enumerate_unified: AND guard check failed");
                    }
                    return Ok(());
                }

                // Flatten AND tree into conjuncts
                // Part of #3897: SmallVec avoids heap allocation for <=8 conjuncts.
                let mut conjuncts = SmallVec::new();
                flatten_and_spanned(expr, &mut conjuncts);

                // The leading run of pure (action-free, operator-safe) guard conjuncts was
                // already validated by `check_and_guards` above, so the enumerator can start
                // past them rather than re-evaluating them as trivially-true guards. This is
                // a side-effect-free performance skip — it never changes which successors are
                // produced. Only enabled when the guard precheck actually ran.
                let start_idx = if and_guard_precheck() {
                    leading_guard_prefix_len(ctx, &conjuncts)
                } else {
                    0
                };

                // Check might_need_prime_binding BEFORE allocating buffers.
                // This is cached per AST node pointer so it's O(1) after first call.
                // The common case (~95%+ of AND conjuncts) does NOT need prime binding
                // validation, so we can skip the intermediate local_results buffer
                // entirely and enumerate directly into the outer sink.
                let certified_complete_action =
                    complete_action.is_some_and(|certificate| certificate.matches(expr));
                let needs_prime_filter = if certified_complete_action {
                    super::complete_action_filter::note_complete_action_filter_bypass();
                    false
                } else {
                    might_need_prime_binding(ctx, expr)
                };

                if needs_prime_filter {
                    // Slow path: need intermediate buffer for post-filtering.
                    // Part of #3027: Enumerate into a local Vec buffer so that
                    // might_need_prime_binding post-filtering (which needs random access:
                    // swap/truncate/indexing) works regardless of whether the outer sink
                    // is a Vec or a streaming callback.
                    let mut accumulated = Vec::with_capacity(p.vars.len());
                    let mut local_results: Vec<crate::state::DiffSuccessor> = Vec::new();
                    {
                        // TRUE-only ENABLED provenance (#3208 redo of #3100):
                        // emissions in this block are PROVISIONAL — the
                        // `action_holds_in_next_state_array` validation below
                        // may drop them, so recording them as ENABLED
                        // witnesses would be unsound. Suppress recording for
                        // the inner enumeration; the validated SURVIVORS are
                        // re-noted after the filter (any frame enclosing this
                        // whole And-block is still live there).
                        let _prov_suppress = crate::liveness::enabled_provenance::suppress_scope();
                        let mut m = EnumMut {
                            rec: RecState {
                                working: s.working,
                                undo: s.undo,
                                results: &mut local_results,
                            },
                            accumulated: &mut accumulated,
                            assigned_mask: 0,
                            has_complex: false,
                            certified_complete_action: false,
                        };
                        let cont = Cont {
                            conjuncts: &conjuncts,
                            next_idx: start_idx,
                            scope_restore: None,
                        };
                        enumerate_conjuncts(ctx, &cont, None, p, &mut m)?;
                    }

                    // Validate successors: expression contains operators with hidden primes.
                    // Keep the surviving suffix in stable order while compacting in O(n).
                    // This avoids repeated Vec::remove shifts and does not reorder accepted
                    // successors, which keeps downstream traversal deterministic.
                    let end = local_results.len();
                    let mut write = 0;
                    for read in 0..end {
                        let succ_arr = local_results[read].materialize(p.base_with_fp, p.registry);
                        if action_holds_in_next_state_array(
                            ctx, expr, &succ_arr, p.registry, p.tir_leaf,
                        )? {
                            if write != read {
                                local_results.swap(write, read);
                            }
                            write += 1;
                        }
                    }
                    local_results.truncate(write);

                    // Push survivors to the real sink.
                    // Part of #3027: Propagate ControlFlow — stop pushing if
                    // the sink signals early termination (Break).
                    for diff in local_results {
                        // TRUE-only ENABLED provenance: the survivor is now a
                        // VALIDATED genuine successor — note it for any still
                        // live enclosing frame. `changes` entries may still
                        // equal the base value (InSet domains are not
                        // pre-filtered), so state-change is decided by exact
                        // per-value comparison against the base state.
                        // #frame-fp-pop: third closure = lazy successor fp
                        // (same incremental XOR routine the BFS worker uses).
                        crate::liveness::enabled_provenance::note_emission(
                            p.base_with_fp.cached_fingerprint(),
                            || {
                                diff.changes
                                    .iter()
                                    .any(|(idx, v)| p.base_with_fp.get(*idx) != *v)
                            },
                            || {
                                Some(crate::state::compute_changes_fingerprint(
                                    p.base_with_fp,
                                    diff.changes.iter().map(|(idx, v)| (*idx, v)),
                                    p.registry,
                                ))
                            },
                        );
                        if s.results.push_with_ctx(ctx, diff).is_break() {
                            break;
                        }
                    }
                } else {
                    // Fast path: no prime binding validation needed — enumerate
                    // directly into the outer sink, avoiding the intermediate
                    // local_results Vec allocation entirely. This eliminates one
                    // heap allocation + final copy loop per AND conjunct in the
                    // common case (called millions of times in BFS).
                    let mut accumulated = Vec::with_capacity(p.vars.len());
                    {
                        let mut m = EnumMut {
                            rec: RecState {
                                working: s.working,
                                undo: s.undo,
                                results: s.results,
                            },
                            accumulated: &mut accumulated,
                            assigned_mask: 0,
                            has_complex: false,
                            certified_complete_action,
                        };
                        let cont = Cont {
                            conjuncts: &conjuncts,
                            next_idx: start_idx,
                            scope_restore: None,
                        };
                        enumerate_conjuncts(ctx, &cont, None, p, &mut m)?;
                    }
                }

                Ok(())
            }

            // Apply: inline operator and recurse
            Expr::Apply(op_expr, args) => {
                if let Expr::Ident(op_name, _) = &op_expr.node {
                    let resolved_name = ctx.resolve_op_name(op_name.as_str());
                    if let Some(def) = ctx.get_op(resolved_name) {
                        let resolved_def_ptr = Arc::as_ptr(def) as usize;
                        let def = Arc::clone(def);
                        // Part of #3073: use precomputed field instead of per-call AST walk.
                        let needs_substitution = def.has_primed_param;
                        let args_are_action = args.iter().any(|arg| expr_is_action_level(ctx, arg));

                        if needs_substitution || args_are_action {
                            // Call-by-name: substitute argument expressions into body.
                            // Part of #3063: cache substituted body per call site —
                            // apply_substitutions deep-clones the entire AST tree but
                            // always produces the same result for a given call site.
                            let substituted_body = super::subst_cache::cached_substitute(
                                ctx,
                                expr,
                                resolved_def_ptr,
                                || {
                                    let subs: Vec<Substitution> = def
                                        .params
                                        .iter()
                                        .zip(args.iter())
                                        .map(|(param, arg)| Substitution {
                                            from: param.name.clone(),
                                            to: arg.clone(),
                                        })
                                        .collect();
                                    apply_substitutions(&def.body, &subs)
                                },
                            );
                            let _guard = ctx.skip_prime_guard(
                                !def.guards_depend_on_prime && !def.contains_prime,
                            );
                            return enumerate_unified_inner(ctx, &substituted_body, p, s);
                        }

                        // Call-by-value: bind parameters to evaluated argument values
                        // Part of #3194: use eval_leaf to try TIR for arguments.
                        //
                        // TRUE-only ENABLED provenance (#3208 redo of #3100):
                        // collect the evaluated argument values when this
                        // operator definition is provenance-registered; see
                        // conjunct_apply for the witness argument.
                        let complete_action =
                            super::complete_action_filter::certify_complete_action_call(
                                ctx,
                                expr,
                                op_name,
                                resolved_name,
                                &def,
                                args,
                                p.registry,
                            );
                        let mut prov_args: Option<smallvec::SmallVec<[Value; 4]>> =
                            if crate::liveness::enabled_provenance::wants_frame(resolved_def_ptr) {
                                Some(smallvec::SmallVec::new())
                            } else {
                                None
                            };
                        let mark = ctx.mark_stack();
                        for (param, arg) in def.params.iter().zip(args.iter()) {
                            match eval_leaf(ctx, arg, p.tir_leaf) {
                                Ok(arg_val) => {
                                    if let Some(vals) = prov_args.as_mut() {
                                        vals.push(arg_val.clone());
                                    }
                                    ctx.push_binding(Arc::from(param.name.node.as_str()), arg_val);
                                }
                                Err(e) => {
                                    ctx.pop_to_mark(&mark);
                                    return Err(e);
                                }
                            }
                        }
                        let _prov_frame = prov_args.map(|vals| {
                            crate::liveness::enabled_provenance::push_frame(resolved_def_ptr, &vals)
                        });
                        let _guard = ctx
                            .skip_prime_guard(!def.guards_depend_on_prime && !def.contains_prime);
                        let result = enumerate_unified_inner_with_certificate(
                            ctx,
                            &def.body,
                            p,
                            s,
                            complete_action,
                        );
                        drop(_guard);
                        ctx.pop_to_mark(&mark);
                        return result;
                    }
                }
                // Unknown operator — try to evaluate as boolean guard
                // Part of #3194: use eval_leaf to try TIR first.
                match eval_leaf(ctx, expr, p.tir_leaf) {
                    Ok(Value::Bool(true)) => Ok(()),
                    Ok(Value::Bool(false)) => Ok(()),
                    // Part of #1433: Preserve original eval error instead of replacing
                    // with generic Internal error. Non-boolean Ok is still an internal error.
                    Ok(_) => Err(EvalError::Internal {
                        message: format!(
                            "enumerate_unified: cannot resolve Apply operator at {:?}",
                            expr.span
                        ),
                        span: Some(expr.span),
                    }),
                    Err(e) => Err(e),
                }
            }

            // Ident: lookup zero-arg operator and inline
            Expr::Ident(name, _) => {
                let resolved = ctx.resolve_op_name(name.as_str());
                if let Some(def) = ctx.get_op(resolved) {
                    let resolved_def_ptr = Arc::as_ptr(def) as usize;
                    let def = Arc::clone(def);
                    if def.params.is_empty() {
                        // TRUE-only ENABLED provenance (#3208 redo of #3100):
                        // zero-arg operator frame; see conjunct_apply.
                        let _prov_frame =
                            crate::liveness::enabled_provenance::push_frame(resolved_def_ptr, &[]);
                        let _guard = ctx
                            .skip_prime_guard(!def.guards_depend_on_prime && !def.contains_prime);
                        return enumerate_unified_inner(ctx, &def.body, p, s);
                    }
                }
                // Try evaluating as boolean (could be TRUE/FALSE constant)
                // Part of #3194: use eval_leaf to try TIR first.
                match eval_leaf(ctx, expr, p.tir_leaf) {
                    Ok(Value::Bool(true)) => Ok(()),
                    Ok(Value::Bool(false)) => Ok(()),
                    // Part of #1433: Preserve original eval error instead of replacing
                    // with generic Internal error. Non-boolean Ok is still an internal error.
                    Ok(_) => Err(EvalError::Internal {
                        message: format!(
                            "enumerate_unified: cannot resolve Ident operator '{}' at {:?}",
                            name, expr.span
                        ),
                        span: Some(expr.span),
                    }),
                    Err(e) => Err(e),
                }
            }

            // Existential quantification: iterate domain, recurse into body
            Expr::Exists(bounds, body) => {
                let mut accumulated = Vec::new();
                let mut m = EnumMut {
                    rec: RecState {
                        working: s.working,
                        undo: s.undo,
                        results: s.results,
                    },
                    accumulated: &mut accumulated,
                    assigned_mask: 0,
                    has_complex: false,
                    certified_complete_action: false,
                };
                enumerate_exists(ctx, bounds, 0, body, p, &mut m)
            }

            // IF: evaluate condition, recurse into chosen branch
            Expr::If(cond, then_branch, else_branch) => {
                // Bind working state so condition can access primed variables
                // Part of #3194: use eval_leaf to try TIR first at this leaf site.
                let guard_result = {
                    let _env = ctx.bind_next_state_env_guard(s.working.env_ref());
                    eval_leaf(ctx, cond, p.tir_leaf)
                };

                let guard = guard_result?;
                match guard.as_bool() {
                    Some(true) => enumerate_unified_inner(ctx, then_branch, p, s),
                    Some(false) => enumerate_unified_inner(ctx, else_branch, p, s),
                    None => Err(EvalError::TypeError {
                        expected: "BOOLEAN",
                        got: guard.type_name(),
                        span: Some(cond.span),
                    }),
                }
            }

            // LET: bind definitions, recurse into body
            Expr::Let(defs, body) => {
                if let Some(false) =
                    try_let_guard_first_shortcircuit(ctx, defs, body, s.working.env_ref(), p)
                {
                    return Ok(());
                }

                let mark = ctx.mark_stack();

                let all_guards_safe = defs
                    .iter()
                    .all(|def| !def.guards_depend_on_prime && !def.contains_prime);
                let _skip_guard = ctx.skip_prime_guard(all_guards_safe);

                // Register all definitions (including parameterized) in local_ops.
                // Merged env is memoized on Arc identity of the ambient scope +
                // the interned defs — no per-state HAMT clone/insert/id
                // derivation (see tla-eval cache/openv_memo.rs).
                let (merged, merged_id, merged_recursive) = tla_eval::merged_let_env_memoized(
                    ctx.local_ops().as_ref(),
                    defs,
                    tla_eval::MergedLetSite::EnumDispatch,
                    |_| true,
                );
                let saved_local_ops = ctx.local_ops().clone();
                // Enter the LET scope with the memoized scope id instead of
                // INVALIDATED, so cache-key builds inside the body do not re-walk
                // the merged local_ops HAMT on every lookup.
                let saved_outer_id =
                    ctx.enter_let_scope_premerged(merged, merged_id, merged_recursive);

                // A zero-arg LET referenced under prime must remain in local_ops so
                // `a'` evaluates the body of `a` in the next-state context.
                let primed_refs = get_primed_var_refs_spanned_v(body);

                // Bind zero-arg definitions to values eagerly when possible.
                // Part of #1262: TLC evaluates LET bindings lazily — unused bindings that
                // would fail (e.g., `LET parent == CHOOSE p \in {} : TRUE IN ...` where
                // `parent` is never used) don't cause errors. We attempt eager evaluation
                // as an optimization, but on "disabled action" errors (ChooseFailed,
                // NotInDomain, TypeError, etc.) we skip the binding and fall through to
                // local_ops lazy lookup.
                // Part of #3194: use eval_leaf for LET binding evaluation.
                for def in defs {
                    if def.params.is_empty() {
                        if primed_refs.contains(def.name.node.as_str()) {
                            continue;
                        }
                        match eval_leaf(ctx, &def.body, p.tir_leaf) {
                            Ok(val) => {
                                ctx.push_binding(Arc::from(def.name.node.as_str()), val);
                            }
                            Err(e) if is_let_lazy_safe_error(&e) => {
                                // Skip binding — if the def is actually used, the reference
                                // will re-evaluate via local_ops and propagate the error then.
                            }
                            Err(e) => {
                                ctx.pop_to_mark(&mark);
                                ctx.restore_local_ops_with_id(saved_local_ops, saved_outer_id);
                                return Err(e);
                            }
                        }
                    }
                }

                let result =
                    enumerate_unified_inner_with_certificate(ctx, body, p, s, complete_action);

                ctx.pop_to_mark(&mark);
                ctx.restore_local_ops_with_id(saved_local_ops, saved_outer_id);
                result
            }

            // CASE: evaluate arm guards in order, take first match
            // Part of #3194: use eval_leaf for CASE guard evaluation.
            Expr::Case(arms, other) => {
                for arm in arms {
                    match eval_leaf(ctx, &arm.guard, p.tir_leaf) {
                        Ok(Value::Bool(true)) => {
                            return enumerate_unified_inner(ctx, &arm.body, p, s);
                        }
                        Ok(Value::Bool(false)) => {}
                        // Part of #1425: Non-boolean CASE guard is a type error.
                        // TLC: Assert.fail("A non-boolean expression was used as guard condition of CASE")
                        Ok(other) => {
                            return Err(case_guard_error(
                                EvalError::TypeError {
                                    expected: "BOOLEAN",
                                    got: other.type_name(),
                                    span: Some(arm.guard.span),
                                },
                                arm.guard.span,
                            ));
                        }
                        Err(e) => return Err(case_guard_error(e, arm.guard.span)),
                    }
                }
                // No arm matched — use OTHER if present
                if let Some(other_expr) = other {
                    enumerate_unified_inner(ctx, other_expr, p, s)
                } else {
                    // Part of #1425: TLC raises fatal error when no CASE arm matches
                    // and no OTHER clause is present. Previously returned Ok(()) silently.
                    Err(EvalError::CaseNoMatch {
                        span: Some(expr.span),
                    })
                }
            }

            // ModuleRef: INSTANCE operator inlining with substitutions
            Expr::ModuleRef(instance_name, op_name, args) => {
                enumerate_module_ref(ctx, expr, instance_name, op_name, args, p, s)
            }

            // Default: try symbolic assignment extraction first, then boolean guard.
            // Part of #1275: Assignment extraction must come before boolean eval because
            // expressions like `x' = x + 1` evaluate to false (comparing current working
            // state) but are actually primed assignments that should produce successors.
            _ => {
                // First, try extracting symbolic assignments (handles primed assignments)
                let mut symbolic = Vec::new();
                extract_symbolic_assignments_with_registry(
                    ctx,
                    expr,
                    p.vars,
                    &mut symbolic,
                    p.registry,
                    p.tir_leaf,
                )?;
                if debug {
                    let expr_name = match &expr.node {
                        Expr::Unchanged(_) => "Unchanged",
                        Expr::Not(_) => "Not",
                        Expr::Prime(_) => "Prime",
                        Expr::Eq(_, _) => "Eq",
                        _ => "Other",
                    };
                    eprintln!(
                        "enumerate_unified: catch-all expr={}, symbolic len={}, results_before={}",
                        expr_name,
                        symbolic.len(),
                        s.results.count()
                    );
                }
                if !symbolic.is_empty() {
                    let assignments = evaluate_symbolic_assignments(ctx, &symbolic, p.tir_leaf)?;
                    if debug {
                        eprintln!(
                            "enumerate_unified: evaluated {} assignments, results_before={}",
                            assignments.len(),
                            s.results.count()
                        );
                    }
                    build_successor_diffs_from_array_into(
                        ctx,
                        p.base_with_fp,
                        p.vars,
                        &assignments,
                        p.registry,
                        s.results,
                    );
                    if debug {
                        eprintln!(
                            "enumerate_unified: after build, results_after={}",
                            s.results.count()
                        );
                    }
                    return Ok(());
                }

                // No assignments found — evaluate as boolean guard.
                // Part of #1432: Previously discarded eval result entirely. TLC propagates
                // all eval errors fatally in getNextStates; we now match that behavior.
                // Part of #3194: use eval_leaf to try TIR first.
                let eval_result = {
                    let _env = ctx.bind_next_state_env_guard(s.working.env_ref());
                    eval_leaf(ctx, expr, p.tir_leaf)
                };

                match eval_result {
                    Ok(Value::Bool(_)) => Ok(()),
                    Ok(other) => Err(EvalError::TypeError {
                        expected: "BOOLEAN",
                        got: other.type_name(),
                        span: Some(expr.span),
                    }),
                    Err(e) => Err(e),
                }
            }
        }
    })
}
