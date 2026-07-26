// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Forking conjunct dispatch helpers for unified successor enumeration.
//!
//! These handlers process AND conjuncts that branch or iterate: nested AND,
//! OR disjunction, EXISTS quantification, IF conditionals, IN membership
//! enumeration, and CASE expressions. Each handler follows the continuation
//! pattern, calling back into `enumerate_conjuncts` to process remaining
//! conjuncts.
//!
//! Extracted from unified.rs as part of #2360.

use std::rc::Rc;
use std::sync::Arc;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use tla_core::ast::{BoundPattern, BoundVar, CaseArm, Expr, OperatorDef};
use tla_core::Spanned;

use crate::error::EvalError;
use crate::eval::{eval_iter_set_tlc_normalized, push_bound_var_mut, EvalCtx};
use crate::Value;

use super::const_domain_cache::eval_domain_cached;
use super::expr_analysis::{expr_contains_prime_ctx, flatten_and_spanned};
use super::tir_leaf::eval_leaf;
use super::unified::{enumerate_conjuncts, trace_expr_tag};
use super::unified_emit::process_conjunct_guard_or_assignment;
use super::unified_exists::{
    enumerate_exists_in_conjuncts, iterate_exists_values_in_conjuncts,
    try_collect_constrained_subset_values, BoundName, PreparedSubsetExists,
};
use super::unified_types::{Cont, EnumMut, EnumParams};
use super::{case_guard_error, debug_enum_trace, enabled_early_exit, SymbolicAssignment};

// ─── Conjunct dispatch helpers ───────────────────────────────────────────────

/// Nested AND within conjuncts: flatten and continue.
/// Phase H (#3073): uses references instead of cloned AST nodes.
pub(super) fn conjunct_and<'a>(
    ctx: &mut EvalCtx,
    conjunct: &'a Spanned<Expr>,
    c: &Cont<'a>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let mut flat: SmallVec<[&Spanned<Expr>; 8]> = SmallVec::new();
    flatten_and_spanned(conjunct, &mut flat);

    if flat.is_empty() {
        return enumerate_conjuncts(ctx, c, None, p, m);
    }

    let mut new_conjuncts: SmallVec<[&Spanned<Expr>; 8]> = flat;
    new_conjuncts.extend_from_slice(&c.conjuncts[c.next_idx..]);
    enumerate_conjuncts(
        ctx,
        &Cont {
            conjuncts: &new_conjuncts,
            next_idx: 0,
            scope_restore: c.scope_restore.clone(),
        },
        None,
        p,
        m,
    )
}

/// OR within conjuncts: fork into branches, each continuing with remaining.
///
/// Clone-at-branch pattern (#2834): each branch gets a cloned EvalCtx,
/// guaranteeing the parent ctx is never mutated. This is the Rust-idiomatic
/// equivalent of TLC's immutable Context + cons() — structural isolation
/// via ownership. Eliminates the scope_restore corruption class where
/// `conjunct_let` pops LET bindings from local_stack during one branch,
/// causing "Undefined variable" errors in subsequent branches.
pub(super) fn conjunct_or(
    ctx: &mut EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let acc_len = m.accumulated.len();
    let save_point = m.rec.undo.len();
    let saved_mask = m.assigned_mask;
    let saved_complex = m.has_complex;

    // Part of #3893: mark/restore replaces ctx.clone() per branch.
    // EnumMark captures all mutable EvalCtx fields so LET scope
    // mutations during branch evaluation are correctly discarded.
    let enum_mark = ctx.mark_enum();

    // Left branch — mark/restore isolates all ctx state
    enumerate_conjuncts(ctx, c, Some(a), p, m)?;
    ctx.pop_to_enum_mark(&enum_mark);
    m.accumulated.truncate(acc_len);
    m.rec
        .working
        .unbind_to_no_invalidate(m.rec.undo, save_point);
    m.assigned_mask = saved_mask;
    m.has_complex = saved_complex;

    // Part of #3027: Early termination — skip right branch if sink stopped.
    if m.rec.results.is_stopped() {
        return Ok(());
    }

    // Right branch — mark already captured, ctx restored to pristine
    enumerate_conjuncts(ctx, c, Some(b), p, m)?;
    ctx.pop_to_enum_mark(&enum_mark);
    m.accumulated.truncate(acc_len);
    m.rec
        .working
        .unbind_to_no_invalidate(m.rec.undo, save_point);
    m.assigned_mask = saved_mask;
    m.has_complex = saved_complex;

    Ok(())
}

/// EXISTS within conjuncts: iterate domain, each binding continues with body + remaining.
///
/// Clone-at-branch pattern (#2834): each iteration gets a cloned EvalCtx
/// with the binding pushed. Same structural isolation as conjunct_or —
/// scope_restore cannot corrupt subsequent iterations because each
/// operates on its own clone.
pub(super) fn conjunct_exists(
    ctx: &mut EvalCtx,
    bounds: &[tla_core::ast::BoundVar],
    body: &Spanned<Expr>,
    conjunct: &Spanned<Expr>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    if bounds.len() != 1 {
        return enumerate_exists_in_conjuncts(ctx, bounds, 0, body, c, p, m);
    }

    let bound = &bounds[0];
    if let Some(prepared) =
        try_collect_constrained_subset_values(ctx, bound, body, &m.rec.working, p)?
    {
        match prepared {
            PreparedSubsetExists::Optimized(constrained) => {
                return iterate_exists_values_in_conjuncts(
                    ctx,
                    constrained.var_name,
                    constrained.values,
                    constrained.remaining_body.as_deref(),
                    c,
                    p,
                    m,
                );
            }
            PreparedSubsetExists::GenericDomain(domain) => {
                let domain_iter = eval_iter_set_tlc_normalized(
                    ctx,
                    &domain,
                    bound.domain.as_ref().map(|domain| domain.span),
                )?;
                return iterate_exists_values_in_conjuncts(
                    ctx,
                    BoundName::new(bound.name.node.as_str()),
                    domain_iter,
                    Some(body),
                    c,
                    p,
                    m,
                );
            }
        }
    }

    let var_name = BoundName::new(bound.name.node.as_str());
    let domain = match &bound.domain {
        // Part of #3194: use eval_leaf to try TIR for EXISTS domain expressions.
        // Constant-domain cache (#set-construction-redundancy): reuse the value
        // for state-independent, capture-free domains across states.
        Some(domain_expr) => eval_domain_cached(ctx, domain_expr, p.tir_leaf, p.vars)?,
        None => {
            return Err(EvalError::Internal {
                message: format!(
                    "enumerate_conjuncts: unbounded EXISTS at {:?}",
                    conjunct.span
                ),
                span: Some(conjunct.span),
            });
        }
    };

    // TLC parity (#2328): iterate EXISTS domains in TLC-normalized order.
    // Keep lazy domains lazy: powerset candidates are consumed once, so an
    // eager Vec only extends every subset's lifetime to the end of the loop.
    let domain_iter =
        eval_iter_set_tlc_normalized(ctx, &domain, bound.domain.as_ref().map(|d| d.span))?;
    iterate_exists_values_in_conjuncts(ctx, var_name, domain_iter, Some(body), c, p, m)
}

/// One predicate plus the lexical context TLC stores in its ActionItemList.
///
/// This deliberately models only the certified, action-free predicate subset
/// below. The real action continuation remains in TY's ordinary `Cont`.
#[derive(Clone)]
struct ProofItem<'a> {
    expr: &'a Spanned<Expr>,
    ctx: EvalCtx,
    next: Option<Rc<ProofItem<'a>>>,
}

#[derive(Clone, Copy)]
struct ProofStateMark {
    accumulated_len: usize,
    undo_len: usize,
    assigned_mask: u64,
    has_complex: bool,
}

impl ProofStateMark {
    fn capture(m: &EnumMut<'_>) -> Self {
        Self {
            accumulated_len: m.accumulated.len(),
            undo_len: m.rec.undo.len(),
            assigned_mask: m.assigned_mask,
            has_complex: m.has_complex,
        }
    }

    fn restore(self, m: &mut EnumMut<'_>) {
        m.accumulated.truncate(self.accumulated_len);
        m.rec
            .working
            .unbind_to_no_invalidate(m.rec.undo, self.undo_len);
        m.assigned_mask = self.assigned_mask;
        m.has_complex = self.has_complex;
    }
}

/// FORALL within a certified action-free predicate: preserve TLC's exact
/// ActionItemList proof-path enumeration.
///
/// TLC queues one body/context pair per universal instance. A nested OR or
/// EXISTS then reaches the remaining bodies and the real action continuation
/// once per successful proof path. We mirror that DFS directly instead of
/// computing a multiplicity up front. Besides raw-generation parity, direct
/// DFS preserves TLC's proof-continuation order and sink early-stop behavior.
/// Certified user-operator leaves use a positive, side-effect-free value
/// grammar, so evaluator result caching cannot suppress observable calls.
///
/// Certification is deliberately conservative. Unsupported scope, higher-order,
/// multi-bound, or action-level forms retain the legacy one-Boolean guard path;
/// importantly, preflight is evaluation-free, so fallback never sees a partly
/// evaluated predicate.
pub(super) fn conjunct_forall<'a>(
    ctx: &mut EvalCtx,
    _bounds: &'a [BoundVar],
    _body: &'a Spanned<Expr>,
    conjunct: &'a Spanned<Expr>,
    c: &Cont<'a>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    if !tlc_proof_dfs_supported(ctx, conjunct, m.allow_tlc_proof_dfs, p.full_mask) {
        return process_conjunct_guard_or_assignment(ctx, conjunct, c, p, m);
    }

    let resume_ctx = ctx.clone();
    let mark = ProofStateMark::capture(m);
    let result = enumerate_tlc_proof(ctx.clone(), conjunct, None, &resume_ctx, c, p, m);
    mark.restore(m);
    result
}

fn tlc_proof_dfs_supported(
    ctx: &EvalCtx,
    conjunct: &Spanned<Expr>,
    allow_tlc_proof_dfs: bool,
    full_mask: u64,
) -> bool {
    // Hidden-prime action validation currently buffers provisional successors
    // in an uncapped Vec, so its caller passes false here. That route
    // deliberately retains the legacy one-Boolean FORALL behavior: exact TLC
    // raw proof-path parity there remains a full-corpus blocker until the
    // validation buffer can stream or enforce the outer sink's early stop.
    // A zero full_mask means either no state variables or more than 64. In
    // both cases TY cannot prove via assigned_mask that the real action tail
    // is already complete. Disable proof DFS so a trailing FORALL retains the
    // legacy single-Boolean behavior instead of expanding after TLC would
    // already have emitted the completed successor.
    allow_tlc_proof_dfs
        && full_mask != 0
        && !expr_contains_prime_ctx(ctx, &conjunct.node)
        && tlc_proof_formula_supported(
            ctx,
            conjunct,
            &FxHashSet::default(),
            &mut FxHashSet::default(),
        )
}

/// Evaluation-free certification for the TLC proof DFS subset.
///
/// Quantifiers are limited to one bound group because TY's lowered `BoundVar`
/// list no longer records TLC's comma-grouping. Supporting multiple entries
/// without that information could evaluate a shared domain too many times.
fn tlc_proof_formula_supported(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    lexical_names: &FxHashSet<String>,
    visiting_domain_ops: &mut FxHashSet<usize>,
) -> bool {
    match &expr.node {
        Expr::And(a, b) | Expr::Or(a, b) => {
            tlc_proof_formula_supported(ctx, a, lexical_names, visiting_domain_ops)
                && tlc_proof_formula_supported(ctx, b, lexical_names, visiting_domain_ops)
        }
        Expr::Exists(bounds, body) | Expr::Forall(bounds, body) => {
            if bounds.len() != 1 {
                return false;
            }
            let Some(domain) = bounds[0].domain.as_deref() else {
                return false;
            };
            if !tlc_pure_domain_supported(ctx, domain, lexical_names, visiting_domain_ops) {
                return false;
            }
            let mut body_names = lexical_names.clone();
            insert_bound_names(&bounds[0], &mut body_names);
            tlc_proof_formula_supported(ctx, body, &body_names, visiting_domain_ops)
        }
        Expr::Label(label) => {
            tlc_proof_formula_supported(ctx, &label.body, lexical_names, visiting_domain_ops)
        }
        Expr::Apply(op_expr, args) => {
            let Expr::Ident(name, _) = &op_expr.node else {
                return false;
            };
            if lexical_names.contains(name)
                || ctx.name_in_local_scope(name)
                || ctx.is_config_constant(name)
            {
                return false;
            }
            let resolved = ctx.resolve_op_name(name);
            if resolved != name {
                return false;
            }
            let Some(def) = ctx.get_op(name) else {
                return false;
            };
            tlc_leaf_operator_supported(def, args, lexical_names)
        }
        Expr::Bool(_) => true,
        Expr::Eq(left, right) | Expr::Neq(left, right) | Expr::In(left, right) => {
            tlc_pure_leaf_value_supported(left, &|name| lexical_names.contains(name))
                && tlc_pure_leaf_value_supported(right, &|name| lexical_names.contains(name))
        }
        // Target-only certification: every other form retains the legacy
        // one-Boolean FORALL path. In particular, wholesale evaluation of
        // IF/CASE/Implies conditions or arbitrary leaves could interact with
        // user-operator result caching and change observable call counts.
        _ => false,
    }
}

fn insert_bound_names(bound: &BoundVar, names: &mut FxHashSet<String>) {
    match &bound.pattern {
        Some(BoundPattern::Var(name)) => {
            names.insert(name.node.clone());
        }
        Some(BoundPattern::Tuple(tuple_names)) => {
            names.extend(tuple_names.iter().map(|name| name.node.clone()));
        }
        None => {
            names.insert(bound.name.node.clone());
        }
    }
}

/// A user predicate call is safe to leave as one evaluator leaf only when its
/// body itself is one TLC leaf and call-by-name cannot expose a proof generator.
fn tlc_leaf_operator_supported(
    def: &Arc<OperatorDef>,
    args: &[Spanned<Expr>],
    lexical_names: &FxHashSet<String>,
) -> bool {
    if def.is_recursive
        || def.contains_prime
        || def.has_primed_param
        || def.params.len() != args.len()
        || def.params.iter().any(|param| param.arity != 0)
        || !args
            .iter()
            .all(|arg| tlc_proof_safe_value_atom(arg, lexical_names))
    {
        return false;
    }

    let formal_is_scalar = |name: &str| {
        def.params
            .iter()
            .any(|param| param.arity == 0 && param.name.node == name)
    };
    match &def.body.node {
        Expr::Eq(left, right) | Expr::Neq(left, right) | Expr::In(left, right) => {
            tlc_pure_leaf_value_supported(left, &formal_is_scalar)
                && tlc_pure_leaf_value_supported(right, &formal_is_scalar)
        }
        _ => false,
    }
}

fn tlc_proof_safe_value_atom(expr: &Spanned<Expr>, lexical_names: &FxHashSet<String>) -> bool {
    match &expr.node {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::StateVar(_, _, _) => true,
        Expr::Ident(name, _) => lexical_names.contains(name),
        _ => false,
    }
}

/// Positive purity grammar for values nested inside a certified evaluator leaf.
///
/// This intentionally admits only the 2PC predicate shape (for example,
/// `rmState[rm] = "prepared"`). Any builtin/user call, non-formal Ident,
/// scope/control node, or lazy function source fails closed so proof DFS cannot
/// change observable call counts through evaluator caching.
fn tlc_pure_leaf_value_supported(
    expr: &Spanned<Expr>,
    ident_is_scalar: &impl Fn(&str) -> bool,
) -> bool {
    match &expr.node {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::StateVar(_, _, _) => true,
        Expr::Ident(name, _) => ident_is_scalar(name),
        Expr::FuncApply(function, index) => {
            matches!(&function.node, Expr::StateVar(_, _, _))
                && tlc_pure_leaf_value_supported(index, ident_is_scalar)
        }
        Expr::SetEnum(elements) => elements
            .iter()
            .all(|element| tlc_pure_leaf_value_supported(element, ident_is_scalar)),
        _ => false,
    }
}

/// Evaluation-free domain grammar for the target TLC proof-path subset.
///
/// It admits materialized configured domains (such as `RM`), literal
/// SetEnum/Range domains, and zero-argument constant aliases like
/// `S == {"a", "b", "c"}`. Calls, binders, lazy values, and every other
/// expression fail closed.
fn tlc_pure_domain_supported(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    lexical_names: &FxHashSet<String>,
    visiting_domain_ops: &mut FxHashSet<usize>,
) -> bool {
    match &expr.node {
        Expr::Ident(name, name_id) => {
            if lexical_names.contains(name) {
                return true;
            }
            if let Some(value) = ctx.lookup_binding(name) {
                return tlc_materialized_domain_value(&value);
            }
            if ctx.name_in_local_scope(name) {
                return false;
            }
            let resolved = ctx.resolve_op_name(name);
            if resolved != name {
                return false;
            }
            if let Some(value) = ctx.precomputed_constants().get(name_id) {
                return tlc_materialized_domain_value(value);
            }

            let Some(def) = ctx.get_op(name) else {
                return false;
            };
            if def.is_recursive
                || def.contains_prime
                || def.has_primed_param
                || !def.params.is_empty()
            {
                return false;
            }
            let ptr = Arc::as_ptr(def) as usize;
            if !visiting_domain_ops.insert(ptr) {
                return false;
            }
            let supported = tlc_pure_domain_supported(
                ctx,
                &def.body,
                &FxHashSet::default(),
                visiting_domain_ops,
            );
            visiting_domain_ops.remove(&ptr);
            supported
        }
        Expr::SetEnum(elements) => elements
            .iter()
            .all(|element| tlc_pure_domain_element_supported(element, lexical_names)),
        Expr::Range(start, end) => {
            tlc_pure_domain_element_supported(start, lexical_names)
                && tlc_pure_domain_element_supported(end, lexical_names)
        }
        _ => false,
    }
}

fn tlc_pure_domain_element_supported(
    expr: &Spanned<Expr>,
    lexical_names: &FxHashSet<String>,
) -> bool {
    match &expr.node {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,
        Expr::Ident(name, _) => lexical_names.contains(name),
        _ => false,
    }
}

fn tlc_materialized_domain_value(value: &Value) -> bool {
    matches!(value, Value::Set(_) | Value::Interval(_))
}

#[allow(clippy::too_many_arguments)]
fn enumerate_tlc_proof<'a>(
    ctx: EvalCtx,
    expr: &'a Spanned<Expr>,
    tail: Option<Rc<ProofItem<'a>>>,
    resume_ctx: &EvalCtx,
    continuation: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    crate::eval::stack_safe(|| {
        enumerate_tlc_proof_inner(ctx, expr, tail, resume_ctx, continuation, p, m)
    })
}

#[allow(clippy::too_many_arguments)]
fn enumerate_tlc_proof_inner<'a>(
    mut ctx: EvalCtx,
    expr: &'a Spanned<Expr>,
    tail: Option<Rc<ProofItem<'a>>>,
    resume_ctx: &EvalCtx,
    continuation: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    if m.rec.results.is_stopped() {
        return Ok(());
    }

    match &expr.node {
        Expr::And(left, right) => {
            let next = Some(Rc::new(ProofItem {
                expr: right,
                ctx: ctx.clone(),
                next: tail,
            }));
            enumerate_tlc_proof(ctx, left, next, resume_ctx, continuation, p, m)
        }
        Expr::Or(left, right) => {
            let mark = ProofStateMark::capture(m);
            let left_result = enumerate_tlc_proof(
                ctx.clone(),
                left,
                tail.clone(),
                resume_ctx,
                continuation,
                p,
                m,
            );
            mark.restore(m);
            left_result?;
            if proof_should_stop(m) {
                return Ok(());
            }
            let right_result =
                enumerate_tlc_proof(ctx, right, tail, resume_ctx, continuation, p, m);
            mark.restore(m);
            right_result
        }
        Expr::Implies(antecedent, consequent) => {
            if eval_tlc_proof_bool(&mut ctx, antecedent, p, m)? {
                enumerate_tlc_proof(ctx, consequent, tail, resume_ctx, continuation, p, m)
            } else {
                enumerate_tlc_proof_tail(tail, resume_ctx, continuation, p, m)
            }
        }
        Expr::Exists(bounds, body) => enumerate_tlc_exists_proof(
            ctx,
            &bounds[0],
            body,
            tail,
            resume_ctx,
            continuation,
            p,
            m,
            expr.span,
        ),
        Expr::Forall(bounds, body) => enumerate_tlc_forall_proof(
            ctx,
            &bounds[0],
            body,
            tail,
            resume_ctx,
            continuation,
            p,
            m,
            expr.span,
        ),
        Expr::If(condition, then_branch, else_branch) => {
            let selected = if eval_tlc_proof_bool(&mut ctx, condition, p, m)? {
                then_branch
            } else {
                else_branch
            };
            enumerate_tlc_proof(ctx, selected, tail, resume_ctx, continuation, p, m)
        }
        Expr::Case(arms, other) => {
            for arm in arms {
                match eval_tlc_proof_bool(&mut ctx, &arm.guard, p, m) {
                    Ok(true) => {
                        return enumerate_tlc_proof(
                            ctx,
                            &arm.body,
                            tail,
                            resume_ctx,
                            continuation,
                            p,
                            m,
                        );
                    }
                    Ok(false) => {}
                    Err(error) => {
                        return Err(case_guard_error(error, arm.guard.span));
                    }
                }
            }
            match other.as_deref() {
                Some(other) => {
                    enumerate_tlc_proof(ctx, other, tail, resume_ctx, continuation, p, m)
                }
                None => Err(EvalError::CaseNoMatch {
                    span: Some(expr.span),
                }),
            }
        }
        Expr::Label(label) => {
            enumerate_tlc_proof(ctx, &label.body, tail, resume_ctx, continuation, p, m)
        }
        _ => {
            if eval_tlc_proof_bool(&mut ctx, expr, p, m)? {
                enumerate_tlc_proof_tail(tail, resume_ctx, continuation, p, m)
            } else {
                Ok(())
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn enumerate_tlc_proof_tail<'a>(
    tail: Option<Rc<ProofItem<'a>>>,
    resume_ctx: &EvalCtx,
    continuation: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    if m.rec.results.is_stopped() {
        return Ok(());
    }
    match tail {
        Some(item) => enumerate_tlc_proof(
            item.ctx.clone(),
            item.expr,
            item.next.clone(),
            resume_ctx,
            continuation,
            p,
            m,
        ),
        None => {
            let mut ctx = resume_ctx.clone();
            enumerate_conjuncts(&mut ctx, continuation, None, p, m)
        }
    }
}

fn eval_tlc_proof_bool(
    ctx: &mut EvalCtx,
    expr: &Spanned<Expr>,
    p: &EnumParams<'_>,
    m: &EnumMut<'_>,
) -> Result<bool, EvalError> {
    let value = {
        let _env = ctx.bind_next_state_env_guard(m.rec.working.env_ref());
        eval_leaf(ctx, expr, p.tir_leaf)?
    };
    value.as_bool().ok_or(EvalError::TypeError {
        expected: "BOOLEAN",
        got: value.type_name(),
        span: Some(expr.span),
    })
}

fn eval_tlc_proof_domain(
    ctx: &mut EvalCtx,
    domain: &Spanned<Expr>,
    p: &EnumParams<'_>,
    m: &EnumMut<'_>,
) -> Result<Value, EvalError> {
    let _env = ctx.bind_next_state_env_guard(m.rec.working.env_ref());
    eval_leaf(ctx, domain, p.tir_leaf)
}

#[allow(clippy::too_many_arguments)]
fn enumerate_tlc_exists_proof<'a>(
    mut ctx: EvalCtx,
    bound: &BoundVar,
    body: &'a Spanned<Expr>,
    tail: Option<Rc<ProofItem<'a>>>,
    resume_ctx: &EvalCtx,
    continuation: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
    span: tla_core::Span,
) -> Result<(), EvalError> {
    let domain_expr = bound.domain.as_deref().ok_or_else(|| EvalError::Internal {
        message: "EXISTS requires bounded quantification".into(),
        span: Some(span),
    })?;
    // TLC's contexts(...) evaluates each quantifier domain once in the
    // quantifier's original context before visiting any candidate.
    let domain = eval_tlc_proof_domain(&mut ctx, domain_expr, p, m)?;
    let values = eval_iter_set_tlc_normalized(&ctx, &domain, Some(domain_expr.span))?;
    let mark = ProofStateMark::capture(m);

    for value in values {
        let mut body_ctx = ctx.clone();
        push_bound_var_mut(&mut body_ctx, bound, &value, Some(span))?;
        let result =
            enumerate_tlc_proof(body_ctx, body, tail.clone(), resume_ctx, continuation, p, m);
        mark.restore(m);
        result?;
        if proof_should_stop(m) {
            break;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn enumerate_tlc_forall_proof<'a>(
    mut ctx: EvalCtx,
    bound: &BoundVar,
    body: &'a Spanned<Expr>,
    tail: Option<Rc<ProofItem<'a>>>,
    resume_ctx: &EvalCtx,
    continuation: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
    span: tla_core::Span,
) -> Result<(), EvalError> {
    let domain_expr = bound.domain.as_deref().ok_or_else(|| EvalError::Internal {
        message: "FORALL requires bounded quantification".into(),
        span: Some(span),
    })?;
    let domain = eval_tlc_proof_domain(&mut ctx, domain_expr, p, m)?;
    let values = eval_iter_set_tlc_normalized(&ctx, &domain, Some(domain_expr.span))?;

    // TLC consumes all contexts before evaluating the first body: c1 is
    // processed first; c2..cn are consed, so the remaining order is cn..c2.
    let mut contexts = Vec::new();
    for value in values {
        let mut body_ctx = ctx.clone();
        push_bound_var_mut(&mut body_ctx, bound, &value, Some(span))?;
        contexts.push(body_ctx);
    }
    let Some(first_ctx) = contexts.first().cloned() else {
        return enumerate_tlc_proof_tail(tail, resume_ctx, continuation, p, m);
    };

    let mut body_tail = tail;
    for body_ctx in contexts.into_iter().skip(1) {
        body_tail = Some(Rc::new(ProofItem {
            expr: body,
            ctx: body_ctx,
            next: body_tail,
        }));
    }
    enumerate_tlc_proof(first_ctx, body, body_tail, resume_ctx, continuation, p, m)
}

#[inline]
fn proof_should_stop(m: &EnumMut<'_>) -> bool {
    m.rec.results.is_stopped() || (enabled_early_exit() && m.rec.results.has_results())
}

#[cfg(test)]
mod tlc_proof_dfs_tests {
    use super::*;
    use tla_core::ast::Unit;
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    #[test]
    fn proof_dfs_policy_declines_hidden_buffer_and_unrepresentable_var_mask() {
        let tree = parse_to_syntax_tree(
            r#"
---- MODULE TlcForallHiddenPrimeBufferPolicy ----
VARIABLE x
Inner == x' = 1
Outer == Inner
Init == x = 0
Next ==
    /\ x = 0
    /\ \A i \in {1, 2} : TRUE \/ TRUE
    /\ Outer
====
"#,
        );
        let lowered = lower(FileId(0), &tree);
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        let module = lowered.module.expect("test module should lower");
        let next = module
            .units
            .iter()
            .find_map(|unit| match &unit.node {
                Unit::Operator(def) if def.name.node == "Next" => Some(def),
                _ => None,
            })
            .expect("Next definition");

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        assert!(
            expr_contains_prime_ctx(&ctx, &next.body.node),
            "the full action must take hidden-prime validation"
        );

        let mut conjuncts = SmallVec::<[&Spanned<Expr>; 8]>::new();
        flatten_and_spanned(&next.body, &mut conjuncts);
        let forall = conjuncts
            .into_iter()
            .find(|expr| matches!(expr.node, Expr::Forall(_, _)))
            .expect("test action should contain a FORALL conjunct");

        assert!(
            tlc_proof_dfs_supported(&ctx, forall, true, 1),
            "the predicate itself is in the certified proof-DFS subset"
        );
        assert!(
            !tlc_proof_dfs_supported(&ctx, forall, false, 1),
            "the uncapped hidden-prime validation buffer must disable proof DFS"
        );
        assert!(
            !tlc_proof_dfs_supported(&ctx, forall, true, 0),
            "an unrepresentable assigned-variable mask must disable proof DFS"
        );
    }

    #[test]
    fn proof_dfs_certification_accepts_only_pure_state_function_predicates() {
        let tree = parse_to_syntax_tree(
            r#"
---- MODULE TlcForallLocalBindingPolicy ----
EXTENDS TLC
CONSTANT RM
VARIABLE f
Leaf(v) == v = TRUE
Direct(local) == \A i \in {1} : local
Actual(local) == \A i \in {1} : Leaf(local)
Function(local) == \A i \in {1} : local[i]
Map == [i \in {1} |-> TRUE]
MapAlias == Map
AliasFunction == \A i \in {1} : MapAlias[i]
NestedPrint(v) == PrintT("must not be cached") = TRUE
NestedUser(v) == Leaf(v) = TRUE
NestedPrintProof == \A i \in {1} : NestedPrint(i)
NestedUserProof == \A i \in {1} : NestedUser(i)
Effect == PrintT("must not run during proof certification")
EffectDomain == \A i \in Effect : TRUE
ScalarDomain == \A i \in 1 : TRUE
EffectImplies == \A i \in {1} : Effect => TRUE
EffectIf == \A i \in {1} : IF Effect THEN TRUE ELSE TRUE
EffectCase == \A i \in {1} : CASE Effect -> TRUE [] OTHER -> TRUE
EffectAtomic == \A i \in {1} : Effect = TRUE
Prepared(i) == f[i] \in {TRUE}
SafeStateFunctionProof == \A i \in {1} : Prepared(i)
ConfiguredDomainProof == \A i \in RM : TRUE
====
"#,
        );
        let lowered = lower(FileId(0), &tree);
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        let module = lowered.module.expect("test module should lower");
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        ctx.register_var(Arc::from("f"));
        ctx.resolve_state_vars_in_loaded_ops();
        Arc::make_mut(ctx.shared_arc_mut())
            .precomputed_constants_mut()
            .insert(
                tla_core::intern_name("RM"),
                Value::set([Value::string("rm1")]),
            );
        ctx.push_binding(Arc::from("local"), Value::Bool(true));

        for name in [
            "Direct",
            "Actual",
            "Function",
            "AliasFunction",
            "NestedPrintProof",
            "NestedUserProof",
            "EffectDomain",
            "ScalarDomain",
            "EffectImplies",
            "EffectIf",
            "EffectCase",
            "EffectAtomic",
        ] {
            let body = &ctx.get_op(name).expect("test operator should exist").body;
            assert!(
                !tlc_proof_dfs_supported(&ctx, body, true, 1),
                "{name} must retain legacy evaluation for local bindings/thunks"
            );
        }

        let safe = ctx
            .get_op("SafeStateFunctionProof")
            .expect("safe proof operator should exist")
            .body
            .clone();
        assert!(
            tlc_proof_dfs_supported(&ctx, &safe, true, 1),
            "direct state-function lookup with a scalar formal must remain certified"
        );

        let configured = ctx
            .get_op("ConfiguredDomainProof")
            .expect("configured-domain proof operator should exist")
            .body
            .clone();
        assert!(
            tlc_proof_dfs_supported(&ctx, &configured, true, 1),
            "a materialized configured domain must remain certified"
        );

        ctx.push_binding(Arc::from("Prepared"), Value::Bool(true));
        assert!(
            !tlc_proof_dfs_supported(&ctx, &safe, true, 1),
            "a locally shadowed predicate name must disable proof DFS"
        );
    }
}

/// IF within conjuncts: evaluate condition, process chosen branch as pending.
///
/// Accepts the full IF expression and destructures internally to avoid
/// passing cond/then/else as separate parameters.
pub(super) fn conjunct_if(
    ctx: &mut EvalCtx,
    conjunct: &Spanned<Expr>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let (cond, then_branch, else_branch) = match &conjunct.node {
        Expr::If(cond, then_branch, else_branch) => {
            (cond.as_ref(), then_branch.as_ref(), else_branch.as_ref())
        }
        _ => {
            return Err(EvalError::Internal {
                message: "conjunct_if called with non-If expression".to_string(),
                span: Some(conjunct.span),
            });
        }
    };

    // Part of #3194: use eval_leaf to try TIR for IF condition evaluation.
    let guard_result = {
        let _env = ctx.bind_next_state_env_guard(m.rec.working.env_ref());
        eval_leaf(ctx, cond, p.tir_leaf)
    };

    let guard = guard_result?;

    let branch = match guard.as_bool() {
        Some(true) => then_branch,
        Some(false) => else_branch,
        None => {
            return process_conjunct_guard_or_assignment(ctx, conjunct, c, p, m);
        }
    };

    enumerate_conjuncts(ctx, c, Some(branch), p, m)
}

/// IN within conjuncts: handle `x' \in S` primed membership enumeration.
pub(super) fn conjunct_in<'a>(
    ctx: &mut EvalCtx,
    lhs: &Spanned<Expr>,
    rhs: &'a Spanned<Expr>,
    conjunct: &'a Spanned<Expr>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let trace = debug_enum_trace();
    if let Expr::Prime(inner_lhs) = &lhs.node {
        if let Expr::Ident(name, _) | Expr::StateVar(name, _, _) = &inner_lhs.node {
            if let Some(idx) = p.registry.get(name.as_str()) {
                let var = &p.vars[idx.as_usize()];
                // O(1) bitmask check instead of O(n) linear scan over accumulated
                let already_bound = if idx.as_usize() < 64 {
                    m.assigned_mask & (1u64 << idx.as_usize()) != 0
                } else {
                    m.accumulated.iter().any(|a| match a {
                        SymbolicAssignment::Value(v, _)
                        | SymbolicAssignment::Expr(v, _, _)
                        | SymbolicAssignment::InSet(v, _, _)
                        | SymbolicAssignment::Unchanged(v) => v == var,
                    })
                };

                if already_bound {
                    if trace {
                        eprintln!(
                            "[enum-trace] level={} kind=binder tag=In var={} already-bound=true action=guard-fallback",
                            ctx.get_tlc_level(),
                            var
                        );
                    }
                    return process_conjunct_guard_or_assignment(ctx, conjunct, c, p, m);
                }

                if trace {
                    eprintln!(
                        "[enum-trace] level={} kind=binder tag=In var={} already-bound=false action=enumerate-domain",
                        ctx.get_tlc_level(),
                        var
                    );
                }
                return enumerate_in_domain(ctx, var, rhs, c, p, m);
            }
        }
    }
    // Not a primed membership — fall through to guard/assignment
    process_conjunct_guard_or_assignment(ctx, conjunct, c, p, m)
}

/// Enumerate all values in a domain set for a primed variable membership.
fn enumerate_in_domain(
    ctx: &mut EvalCtx,
    var: &Arc<str>,
    rhs: &Spanned<Expr>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let trace = debug_enum_trace();
    // FIX #1482: Bind partially-constructed next state so that primed variables
    // already assigned by earlier conjuncts (e.g. opQ') are visible when
    // evaluating the domain expression (e.g. SUBSET(opId' \X opId')).
    // Without this, `eval(ctx, rhs)` fails with "Primed variable cannot be
    // evaluated" when the domain references another primed variable.
    // Part of #3194: use eval_leaf to try TIR for IN domain expression.
    let domain = {
        let _env = ctx.bind_next_state_env_guard(m.rec.working.env_ref());
        eval_domain_cached(ctx, rhs, p.tir_leaf, p.vars)
    };
    let domain = domain?;

    if trace {
        eprintln!(
            "[enum-trace] level={} binder var={} domain-tag={} domain-type={}",
            ctx.get_tlc_level(),
            var,
            trace_expr_tag(&rhs.node),
            domain.type_name()
        );
    }

    let var_idx = p.registry.get(var);
    let saved_mask = m.assigned_mask;
    if let Some(idx) = var_idx {
        if idx.as_usize() < 64 {
            m.assigned_mask |= 1u64 << idx.as_usize();
        }
    }

    // #1482 follow-up: SetPred materialization may evaluate predicates that
    // reference primed vars (e.g., {v \in S : v = y'}), so preserve the
    // partially-built next-state binding while iterating the domain.
    // TLC parity (#2328): iterate x' \in S domains in TLC-normalized order.
    // Same rationale as conjunct_exists — BFS exploration order must match TLC.
    let domain_iter = {
        let _env = ctx.bind_next_state_env_guard(m.rec.working.env_ref());
        eval_iter_set_tlc_normalized(ctx, &domain, Some(rhs.span))
    };

    let mut domain_iter = match domain_iter {
        Ok(iter) => iter.peekable(),
        // Non-enumerable domain (value is not a Set type): produce no
        // successors. Matches TLC behavior where membership in a non-set
        // is structurally rejected (no states generated, no error).
        Err(EvalError::TypeError {
            expected: "Set", ..
        }) => {
            m.assigned_mask = saved_mask;
            return Ok(());
        }
        // All other errors propagate as model checking failures.
        Err(e) => {
            m.assigned_mask = saved_mask;
            return Err(e);
        }
    };

    if domain_iter.peek().is_none() {
        if trace {
            eprintln!(
                "[enum-trace] level={} binder var={} domain-empty=true",
                ctx.get_tlc_level(),
                var
            );
        }
        m.assigned_mask = saved_mask;
        return Ok(());
    }

    let acc_len = m.accumulated.len();
    let save_point = m.rec.undo.len();
    let results_before = m.rec.results.count();
    let mut iterated_total = 0usize;

    // Part of #3893: mark/restore replaces ctx.clone() per iteration.
    let enum_mark = ctx.mark_enum();

    for val in domain_iter {
        iterated_total += 1;
        m.accumulated
            .push(SymbolicAssignment::Value(Arc::clone(var), val.clone()));

        if let Some(idx) = var_idx {
            m.rec.working.bind_no_invalidate(idx, val, m.rec.undo);
        }

        match enumerate_conjuncts(ctx, c, None, p, m) {
            Ok(()) => {}
            Err(e) => {
                ctx.pop_to_enum_mark(&enum_mark);
                m.rec
                    .working
                    .unbind_to_no_invalidate(m.rec.undo, save_point);
                m.assigned_mask = saved_mask;
                return Err(e);
            }
        }
        ctx.pop_to_enum_mark(&enum_mark);

        m.accumulated.truncate(acc_len);
        m.rec
            .working
            .unbind_to_no_invalidate(m.rec.undo, save_point);
        // Reset assigned_mask for next iteration (same fix as above, #1316)
        m.assigned_mask = saved_mask;
        if let Some(idx) = var_idx {
            if idx.as_usize() < 64 {
                m.assigned_mask |= 1u64 << idx.as_usize();
            }
        }

        // Part of #3027: Early termination — stop domain iteration if sink stopped.
        if m.rec.results.is_stopped() {
            break;
        }
    }

    if trace {
        eprintln!(
            "[enum-trace] level={} binder var={} iterated={} successors-added={}",
            ctx.get_tlc_level(),
            var,
            iterated_total,
            m.rec.results.count().saturating_sub(results_before)
        );
    }
    m.assigned_mask = saved_mask;
    Ok(())
}

/// CASE within conjuncts: evaluate arm guards and continue with matching body.
///
/// Mirrors the top-level CASE handler in `enumerate_unified_inner` but uses
/// `enumerate_conjuncts` continuation so the matching arm body is processed as
/// part of the AND chain. Without this, CASE arms containing primed assignments
/// would be evaluated as boolean guards (via the catch-all), potentially losing
/// state assignments.
///
/// FIX #1427: self-audit — same dispatch gap class as ModuleRef.
pub(super) fn conjunct_case(
    ctx: &mut EvalCtx,
    arms: &[CaseArm],
    other: Option<&Spanned<Expr>>,
    conjunct: &Spanned<Expr>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    for arm in arms {
        // Part of #3194: use eval_leaf to try TIR for CASE guard evaluation.
        match eval_leaf(ctx, &arm.guard, p.tir_leaf) {
            Ok(Value::Bool(true)) => {
                return enumerate_conjuncts(ctx, c, Some(&arm.body), p, m);
            }
            Ok(Value::Bool(false)) => {}
            Ok(other_val) => {
                return Err(case_guard_error(
                    EvalError::TypeError {
                        expected: "BOOLEAN",
                        got: other_val.type_name(),
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
        enumerate_conjuncts(ctx, c, Some(other_expr), p, m)
    } else {
        Err(EvalError::CaseNoMatch {
            span: Some(conjunct.span),
        })
    }
}
