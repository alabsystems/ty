// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! EXISTS quantifier enumeration helpers for unified successor enumeration.
//!
//! Handles multi-bound EXISTS expressions by recursively processing bound
//! variables and iterating over their domains. Two variants:
//! - `enumerate_exists_in_conjuncts`: EXISTS within AND conjuncts (uses continuation)
//! - `enumerate_exists`: EXISTS at top level (recurses into `enumerate_unified_inner`)
//!
//! Extracted from unified.rs as part of #2360.

#[cfg(test)]
use std::cell::Cell;
use std::sync::{Arc, OnceLock};

use smallvec::SmallVec;
use tla_core::ast::Expr;
use tla_core::{intern_name, NameId, Spanned};

use crate::error::EvalError;
use crate::eval::{eval_iter_set_tlc_normalized, EvalCtx};
use crate::state::ArrayState;
use crate::Value;

use super::const_domain_cache::eval_domain_cached;
use super::enabled_early_exit;
use super::expr_analysis::{expr_is_action_level, is_operator_reference_guard_unsafe};
use super::first_guard_sched::prepare_first_guard_runtime;
use super::subset_constrained::{
    generate_constrained_subsets, generate_quorum_subsets, match_constrained_subset_exists,
    match_quorum_subset_exists, ConstrainedSubsetPattern,
};
use super::subset_profile;
use super::tir_leaf::eval_leaf;
use super::unified::{enumerate_conjuncts, enumerate_unified_inner, Cont, EnumMut, EnumParams};

#[cfg(test)]
thread_local! {
    static EXISTS_NAME_ID_BINDINGS_TEST_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

fn parse_exists_name_id_bindings_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

/// Same-binary A/B gate for NameId-only EXISTS bindings.  Production resolves
/// the environment once per process; tests use a thread-local override so ON
/// and OFF semantics can be compared without mutating process-global state.
fn exists_name_id_bindings_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = EXISTS_NAME_ID_BINDINGS_TEST_OVERRIDE.with(Cell::get) {
        return enabled;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_exists_name_id_bindings_enabled(
            std::env::var("TY_EXISTS_NAME_ID_BINDINGS").ok().as_deref(),
        )
    })
}

#[cfg(test)]
fn with_exists_name_id_bindings_test_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Reset(Option<bool>);

    impl Drop for Reset {
        fn drop(&mut self) {
            EXISTS_NAME_ID_BINDINGS_TEST_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }

    let previous = EXISTS_NAME_ID_BINDINGS_TEST_OVERRIDE.with(|slot| slot.replace(Some(enabled)));
    let _reset = Reset(previous);
    f()
}

/// One prepared EXISTS binder.  The legacy arm deliberately retains the Arc
/// clone and `push_binding` behavior for A/B parity; the opt-in arm stores only
/// the already-interned identifier and uses the BindingChain's ID-only API.
#[derive(Clone)]
pub(super) enum BoundName {
    LegacyArc(Arc<str>),
    NameId(NameId),
}

impl BoundName {
    pub(super) fn new(name: &str) -> Self {
        Self::for_mode(name, exists_name_id_bindings_enabled())
    }

    fn for_mode(name: &str, by_id: bool) -> Self {
        if by_id {
            Self::NameId(intern_name(name))
        } else {
            Self::LegacyArc(Arc::from(name))
        }
    }

    #[inline]
    fn push(&self, ctx: &mut EvalCtx, value: Value) {
        match self {
            Self::LegacyArc(name) => ctx.push_binding(Arc::clone(name), value),
            Self::NameId(name_id) => ctx.push_binding_by_id(*name_id, value),
        }
    }
}

/// Prepared names for a multi-bound EXISTS.  OFF intentionally preserves the
/// former `Vec<Arc<str>>`; ON uses inline SmallVec storage for the common small
/// quantifier and allocates no name Arcs.
enum BoundNames {
    LegacyArcs(Vec<Arc<str>>),
    NameIds(SmallVec<[NameId; 4]>),
}

impl BoundNames {
    fn new(bounds: &[tla_core::ast::BoundVar]) -> Self {
        Self::for_mode(bounds, exists_name_id_bindings_enabled())
    }

    fn for_mode(bounds: &[tla_core::ast::BoundVar], by_id: bool) -> Self {
        if by_id {
            Self::NameIds(
                bounds
                    .iter()
                    .map(|bound| intern_name(bound.name.node.as_str()))
                    .collect(),
            )
        } else {
            Self::LegacyArcs(
                bounds
                    .iter()
                    .map(|bound| Arc::from(bound.name.node.as_str()))
                    .collect(),
            )
        }
    }

    fn get(&self, index: usize) -> BoundName {
        match self {
            Self::LegacyArcs(names) => BoundName::LegacyArc(Arc::clone(&names[index])),
            Self::NameIds(ids) => BoundName::NameId(ids[index]),
        }
    }
}

pub(super) struct ConstrainedSubsetRuntime {
    pub(super) var_name: BoundName,
    pub(super) values: Vec<Value>,
    pub(super) remaining_body: Option<Arc<Spanned<Expr>>>,
}

pub(super) enum PreparedSubsetExists {
    Optimized(ConstrainedSubsetRuntime),
    /// The quorum certificate matched and domain evaluation already happened,
    /// but a runtime representation check declined pruning.  Reuse this value
    /// on the ordinary path instead of evaluating a potentially expensive (or
    /// error-sensitive) domain expression twice.
    GenericDomain(Value),
}

#[inline]
fn constrained_bounds_require_generic_fallback(
    ctx: &EvalCtx,
    pattern: &ConstrainedSubsetPattern<'_>,
) -> bool {
    expr_is_action_level(ctx, pattern.superset_expr)
        || is_operator_reference_guard_unsafe(ctx, pattern.superset_expr)
        || expr_is_action_level(ctx, pattern.subset_expr)
        || is_operator_reference_guard_unsafe(ctx, pattern.subset_expr)
}

pub(super) fn try_collect_constrained_subset_values(
    ctx: &mut EvalCtx,
    bound: &tla_core::ast::BoundVar,
    body: &Spanned<Expr>,
    working: &ArrayState,
    p: &EnumParams<'_>,
) -> Result<Option<PreparedSubsetExists>, EvalError> {
    let Some(domain_expr) = bound.domain.as_ref() else {
        return Ok(None);
    };

    // The quorum lane is deliberately tried before the older two-bound
    // constrained-SUBSET matcher because both top-level and conjunct EXISTS
    // routes share this entry point.  A syntactic/certificate miss performs no
    // evaluation and falls straight through.
    if let Some(runtime) = try_collect_quorum_subset_values(ctx, bound, domain_expr, body, p)? {
        return Ok(Some(runtime));
    }

    // Keep the legacy speculative Arc preparation before the matcher in the
    // OFF arm so the same-binary allocation baseline remains exact. The ON
    // arm defers interning until a constrained pattern actually matches; on
    // the common ordinary-domain path, BoundNames will intern the name once.
    let name = bound.name.node.as_str();
    let by_id = exists_name_id_bindings_enabled();
    let legacy_var_name = (!by_id).then(|| BoundName::LegacyArc(Arc::from(name)));
    let Some(pattern) = match_constrained_subset_exists(name, domain_expr, body) else {
        return Ok(None);
    };
    // Bounds are precomputed before the quantified value is installed. Any
    // direct or hidden action-level read would instead have to observe the
    // partially built next state at its original source position, so leave it
    // to ordinary enumeration.
    if constrained_bounds_require_generic_fallback(ctx, &pattern) {
        return Ok(None);
    }
    let var_name = legacy_var_name.unwrap_or_else(|| BoundName::NameId(intern_name(name)));

    subset_profile::record_entry();

    let base_set = eval_leaf(ctx, pattern.base_set_expr, p.tir_leaf)?;
    let base_elements: Vec<Value> =
        eval_iter_set_tlc_normalized(ctx, &base_set, Some(pattern.base_set_expr.span))?.collect();

    let superset_bound = {
        let _env = ctx.bind_next_state_env_guard(working.env_ref());
        eval_leaf(ctx, pattern.superset_expr, p.tir_leaf)?
    };
    // A non-set upper bound has candidate-sensitive generic behavior: in
    // particular, an empty LHS can finish without consulting RHS membership.
    // Fall back instead of manufacturing either a result or an eager error.
    if !superset_bound.is_set() {
        subset_profile::record_fallback();
        return Ok(None);
    }
    let subset_bound = {
        let _env = ctx.bind_next_state_env_guard(working.env_ref());
        eval_leaf(ctx, pattern.subset_expr, p.tir_leaf)?
    };
    if !subset_bound.is_set() {
        subset_profile::record_fallback();
        return Ok(None);
    }

    let Some(values) = generate_constrained_subsets(&base_elements, &superset_bound, &subset_bound)
    else {
        return Ok(None);
    };

    Ok(Some(PreparedSubsetExists::Optimized(
        ConstrainedSubsetRuntime {
            var_name,
            values,
            remaining_body: pattern.remaining_body.map(Arc::new),
        },
    )))
}

fn try_collect_quorum_subset_values(
    ctx: &mut EvalCtx,
    bound: &tla_core::ast::BoundVar,
    domain_expr: &Spanned<Expr>,
    body: &Spanned<Expr>,
    p: &EnumParams<'_>,
) -> Result<Option<PreparedSubsetExists>, EvalError> {
    if bound.pattern.is_some() {
        return Ok(None);
    }
    let name = bound.name.node.as_str();
    let Some(pattern) = match_quorum_subset_exists(ctx, name, domain_expr, body) else {
        return Ok(None);
    };
    // Reuse the existing opt-in SUBSET telemetry so a production run can prove
    // this certificate actually activated. The ordinary two-bound lane cannot
    // match this first-guard shape, making ON/OFF profile deltas unambiguous.
    subset_profile::record_entry();

    // Preserve the generic path's ordering: evaluate the complete domain and
    // TLC-normalize its powerset base before inspecting the first guard. This
    // is exactly the preparation performed by SubsetIterator::from_elements,
    // but it does not construct any of the 2^n candidate subsets.
    let domain = eval_domain_cached(ctx, domain_expr, p.tir_leaf, p.vars)?;
    let base = match &domain {
        Value::Subset(subset) => subset.base(),
        _ => {
            subset_profile::record_fallback();
            return Ok(Some(PreparedSubsetExists::GenericDomain(domain)));
        }
    };
    let base_elements: SmallVec<[Value; 8]> = base
        .iter_set_tlc_normalized()
        .map_err(|err| map_quorum_base_iteration_error_span(err, domain_expr.span))?
        .collect();

    // In the certified operator body the literal quorum-family RHS is the
    // first expression that can do work for the empty-set candidate. Preserve
    // that ordering by initializing the run cache only after domain/base
    // evaluation. Its exact literals and pinned precomputed constants are
    // state-independent, so no next-state binding (and its cache invalidation)
    // is needed. Evaluator errors are deliberately not cached.
    if pattern.quorum_family_value.get().is_none() {
        let value = eval_leaf(ctx, pattern.quorum_family_expr.as_ref(), p.tir_leaf)?;
        let _ = pattern.quorum_family_value.set(value);
    }
    let quorum_family = pattern
        .quorum_family_value
        .get()
        .expect("quorum family OnceLock was initialized above");
    let Some(values) = generate_quorum_subsets(&base_elements, quorum_family) else {
        subset_profile::record_fallback();
        return Ok(Some(PreparedSubsetExists::GenericDomain(domain)));
    };
    subset_profile::record_success(base_elements.len(), values.len());

    Ok(Some(PreparedSubsetExists::Optimized(
        ConstrainedSubsetRuntime {
            var_name: BoundName::NameId(pattern.bound_name_id),
            values,
            remaining_body: pattern.remaining_body,
        },
    )))
}

/// Match the span attachment performed by `eval_iter_set_tlc_normalized` on a
/// `Value::Subset`. Calling the base value's native normalized iterator is
/// intentional: unlike the EvalCtx-aware helper, this retains the generic
/// powerset path's error for a non-native (for example SetPred) base.
fn map_quorum_base_iteration_error_span(err: EvalError, span: tla_core::Span) -> EvalError {
    match err {
        EvalError::TypeError {
            expected,
            got,
            span: None,
        } => EvalError::TypeError {
            expected,
            got,
            span: Some(span),
        },
        EvalError::SetTooLarge { span: None } => EvalError::SetTooLarge { span: Some(span) },
        EvalError::Internal {
            message,
            span: None,
        } => EvalError::Internal {
            message,
            span: Some(span),
        },
        other => other,
    }
}

pub(super) fn iterate_exists_values_in_conjuncts(
    ctx: &mut EvalCtx,
    var_name: BoundName,
    values: impl IntoIterator<Item = Value>,
    body: Option<&Spanned<Expr>>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let acc_len = m.accumulated.len();
    let save_point = m.rec.undo.len();
    let saved_mask = m.assigned_mask;
    let saved_complex = m.has_complex;

    // Part of #3893: mark/restore replaces ctx.clone() per iteration.
    // EnumMark captures all mutable EvalCtx fields (bindings, let_def_overlay,
    // stable Rc, call_by_name_subs) so that LET scope mutations during body
    // evaluation are correctly discarded between iterations.
    let enum_mark = ctx.mark_enum();

    for value in values {
        var_name.push(ctx, value);

        let result = match body {
            Some(body) => enumerate_conjuncts(ctx, c, Some(body), p, m),
            None => enumerate_conjuncts(ctx, c, None, p, m),
        };
        match result {
            Ok(()) => {}
            Err(err) => {
                ctx.pop_to_enum_mark(&enum_mark);
                m.rec
                    .working
                    .unbind_to_no_invalidate(m.rec.undo, save_point);
                return Err(err);
            }
        }

        ctx.pop_to_enum_mark(&enum_mark);
        m.accumulated.truncate(acc_len);
        m.rec
            .working
            .unbind_to_no_invalidate(m.rec.undo, save_point);
        m.assigned_mask = saved_mask;
        m.has_complex = saved_complex;

        if enabled_early_exit() && m.rec.results.has_results() {
            break;
        }
        if m.rec.results.is_stopped() {
            break;
        }
    }

    Ok(())
}

fn iterate_exists_values_top_level(
    ctx: &mut EvalCtx,
    var_name: BoundName,
    values: impl IntoIterator<Item = Value>,
    body: Option<&Spanned<Expr>>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let save_point = m.rec.undo.len();

    // Part of #3893: mark/restore replaces ctx.clone() per iteration.
    let enum_mark = ctx.mark_enum();

    for value in values {
        var_name.push(ctx, value);

        let result = match body {
            Some(body) => enumerate_unified_inner(ctx, body, p, &mut m.rec),
            None => {
                let empty: [&Spanned<Expr>; 0] = [];
                let cont = Cont {
                    conjuncts: &empty,
                    next_idx: 0,
                    scope_restore: None,
                };
                enumerate_conjuncts(ctx, &cont, None, p, m)
            }
        };
        match result {
            Ok(()) => {}
            Err(err) => {
                ctx.pop_to_enum_mark(&enum_mark);
                m.rec
                    .working
                    .unbind_to_no_invalidate(m.rec.undo, save_point);
                return Err(err);
            }
        }

        ctx.pop_to_enum_mark(&enum_mark);
        m.rec
            .working
            .unbind_to_no_invalidate(m.rec.undo, save_point);

        if enabled_early_exit() && m.rec.results.has_results() {
            break;
        }
        if m.rec.results.is_stopped() {
            break;
        }
    }

    Ok(())
}

/// Handle multi-bound EXISTS within conjuncts by recursively processing bounds.
///
/// Part of #3893: Uses EnumMark (mark/restore) instead of clone-at-branch.
/// EnumMark captures all mutable EvalCtx fields including stable Rc (which
/// holds local_ops), let_def_overlay, and call_by_name_subs — making
/// mark/restore safe even when the body contains LET with scope_restore.
///
/// Part of #3900: pre-computes bound names once at entry.  The experiment keeps
/// the legacy Arc vector when disabled and uses inline NameIds when enabled.
pub(super) fn enumerate_exists_in_conjuncts(
    ctx: &mut EvalCtx,
    bounds: &[tla_core::ast::BoundVar],
    bound_idx: usize,
    body: &Spanned<Expr>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let bound_names = BoundNames::new(bounds);
    enumerate_exists_in_conjuncts_inner(ctx, bounds, &bound_names, bound_idx, body, c, p, m)
}

fn enumerate_exists_in_conjuncts_inner(
    ctx: &mut EvalCtx,
    bounds: &[tla_core::ast::BoundVar],
    bound_names: &BoundNames,
    bound_idx: usize,
    body: &Spanned<Expr>,
    c: &Cont<'_>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    crate::eval::stack_safe(|| {
        if bound_idx >= bounds.len() {
            // All bounds processed — continue with body + remaining conjuncts
            return enumerate_conjuncts(ctx, c, Some(body), p, m);
        }

        let bound = &bounds[bound_idx];
        // Part of #3900: reuse the pre-computed Arc or NameId for this level.
        let var_name = bound_names.get(bound_idx);

        let domain = match &bound.domain {
            // Part of #3194: use eval_leaf to try TIR for EXISTS domain expressions.
            Some(domain_expr) => eval_domain_cached(ctx, domain_expr, p.tir_leaf, p.vars)?,
            None => {
                return Err(EvalError::Internal {
                    message: "enumerate_exists_in_conjuncts: unbounded EXISTS".to_string(),
                    span: None,
                })
            }
        };

        let acc_len = m.accumulated.len();
        let save_point = m.rec.undo.len();
        let saved_mask = m.assigned_mask;
        let saved_complex = m.has_complex;

        // Part of #3893: mark/restore replaces ctx.clone() per iteration.
        let enum_mark = ctx.mark_enum();

        // TLC parity (#2328): visit the domain in TLC-normalized order. Keep
        // lazy domains lazy here: collecting a powerset retains every subset
        // until the quantifier finishes even though successor enumeration
        // consumes each candidate exactly once.
        for val in
            eval_iter_set_tlc_normalized(ctx, &domain, bound.domain.as_ref().map(|d| d.span))?
        {
            var_name.push(ctx, val);

            match enumerate_exists_in_conjuncts_inner(
                ctx,
                bounds,
                bound_names,
                bound_idx + 1,
                body,
                c,
                p,
                m,
            ) {
                Ok(()) => {}
                Err(e) => {
                    ctx.pop_to_enum_mark(&enum_mark);
                    m.rec
                        .working
                        .unbind_to_no_invalidate(m.rec.undo, save_point);
                    return Err(e);
                }
            }

            ctx.pop_to_enum_mark(&enum_mark);
            m.accumulated.truncate(acc_len);
            m.rec
                .working
                .unbind_to_no_invalidate(m.rec.undo, save_point);
            m.assigned_mask = saved_mask;
            m.has_complex = saved_complex;

            // Part of #1285: ENABLED early-exit — stop iterating domain values.
            if enabled_early_exit() && m.rec.results.has_results() {
                break;
            }
            // Part of #3027: Early termination — stop domain iteration if sink stopped.
            if m.rec.results.is_stopped() {
                break;
            }
        }

        Ok(())
    })
}

/// Handle multi-bound EXISTS at top level (not within AND conjuncts).
///
/// Part of #3900: pre-computes bound names once at entry.  The experiment keeps
/// the legacy Arc vector when disabled and uses inline NameIds when enabled.
pub(super) fn enumerate_exists(
    ctx: &mut EvalCtx,
    bounds: &[tla_core::ast::BoundVar],
    bound_idx: usize,
    body: &Spanned<Expr>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    let bound_names = BoundNames::new(bounds);
    enumerate_exists_inner(ctx, bounds, &bound_names, bound_idx, body, p, m)
}

fn enumerate_exists_inner(
    ctx: &mut EvalCtx,
    bounds: &[tla_core::ast::BoundVar],
    bound_names: &BoundNames,
    bound_idx: usize,
    body: &Spanned<Expr>,
    p: &EnumParams<'_>,
    m: &mut EnumMut<'_>,
) -> Result<(), EvalError> {
    crate::eval::stack_safe(|| {
        if bound_idx >= bounds.len() {
            // All bounds processed — recurse into body
            return enumerate_unified_inner(ctx, body, p, &mut m.rec);
        }

        let bound = &bounds[bound_idx];
        if bound_idx == 0 && bounds.len() == 1 {
            if let Some(prepared) =
                try_collect_constrained_subset_values(ctx, bound, body, &m.rec.working, p)?
            {
                match prepared {
                    PreparedSubsetExists::Optimized(constrained) => {
                        return iterate_exists_values_top_level(
                            ctx,
                            constrained.var_name,
                            constrained.values,
                            constrained.remaining_body.as_deref(),
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
                        return iterate_exists_values_top_level(
                            ctx,
                            BoundName::new(bound.name.node.as_str()),
                            domain_iter,
                            Some(body),
                            p,
                            m,
                        );
                    }
                }
            }
        }

        // Part of #3900: reuse the pre-computed Arc or NameId for this level.
        let var_name = bound_names.get(bound_idx);
        let domain = match &bound.domain {
            // Part of #3194: use eval_leaf to try TIR for EXISTS domain expressions.
            Some(domain_expr) => eval_domain_cached(ctx, domain_expr, p.tir_leaf, p.vars)?,
            None => {
                return Err(EvalError::Internal {
                    message: "enumerate_exists: unbounded EXISTS".to_string(),
                    span: None,
                })
            }
        };

        // TLC parity (#2328): visit the domain in TLC-normalized order. Peeking
        // preserves the first-guard scheduler's empty-domain behavior without
        // eagerly retaining every element of lazy powerset/function domains.
        let mut domain_iter =
            eval_iter_set_tlc_normalized(ctx, &domain, bound.domain.as_ref().map(|d| d.span))?
                .peekable();
        let first_guard = if bound_idx == 0 && bounds.len() == 1 && domain_iter.peek().is_some() {
            prepare_first_guard_runtime(ctx, bounds, body, p)
        } else {
            None
        };

        let save_point = m.rec.undo.len();

        // Part of #3893: mark/restore replaces ctx.clone() per iteration.
        let enum_mark = ctx.mark_enum();

        for val in domain_iter {
            if first_guard
                .as_ref()
                .is_some_and(|runtime| runtime.candidate_mismatches(&val))
            {
                continue;
            }
            var_name.push(ctx, val);

            match enumerate_exists_inner(ctx, bounds, bound_names, bound_idx + 1, body, p, m) {
                Ok(()) => {}
                Err(e) => {
                    ctx.pop_to_enum_mark(&enum_mark);
                    m.rec
                        .working
                        .unbind_to_no_invalidate(m.rec.undo, save_point);
                    return Err(e);
                }
            }

            ctx.pop_to_enum_mark(&enum_mark);
            m.rec
                .working
                .unbind_to_no_invalidate(m.rec.undo, save_point);

            // Part of #1285: ENABLED early-exit — stop iterating domain values.
            if enabled_early_exit() && m.rec.results.has_results() {
                break;
            }
            // Part of #3027: Early termination — stop domain iteration if sink stopped.
            if m.rec.results.is_stopped() {
                break;
            }
        }

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerate::enumerate_successors;
    use crate::state::State;
    use std::sync::Arc;
    use tla_core::ast::{BoundVar, Module, Unit};
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    use super::super::first_guard_sched::{
        first_guard_sched_test_prepares, reset_first_guard_sched_test_prepares,
    };
    use super::super::subset_constrained::{
        quorum_subset_prune_test_activations, reset_quorum_subset_prune_test_activations,
        with_quorum_subset_prune_test_override,
    };

    fn simple_bound(name: &str) -> BoundVar {
        BoundVar {
            name: Spanned::dummy(name.to_string()),
            domain: None,
            pattern: None,
        }
    }

    fn setup(
        src: &str,
    ) -> (
        Module,
        EvalCtx,
        Vec<Arc<str>>,
        Arc<tla_core::ast::OperatorDef>,
    ) {
        let tree = parse_to_syntax_tree(src);
        let lowered = lower(FileId(0), &tree);
        let module = lowered.module.expect("test module should lower");

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        let vars: Vec<Arc<str>> = module
            .units
            .iter()
            .filter_map(|unit| match &unit.node {
                Unit::Variable(names) => Some(names.as_slice()),
                _ => None,
            })
            .flatten()
            .map(|name| Arc::from(name.node.as_str()))
            .collect();
        ctx.register_vars(vars.iter().cloned());
        ctx.resolve_state_vars_in_loaded_ops();
        let next = Arc::clone(ctx.get_op("Next").expect("Next should exist"));
        (module, ctx, vars, next)
    }

    fn enumerate_with_mode(
        enabled: bool,
        ctx: &mut EvalCtx,
        next: &tla_core::ast::OperatorDef,
        current: &State,
        vars: &[Arc<str>],
    ) -> Result<Vec<State>, EvalError> {
        // Every caller in this test module builds an independently parsed AST.
        // Clear all run-scoped TLS caches, not just branch replay: other
        // enumeration/evaluator caches also use raw AST identity and a test
        // worker can recycle an address after a preceding fixture is dropped.
        crate::clear_thread_local_eval_caches();
        with_exists_name_id_bindings_test_override(enabled, || {
            enumerate_successors(ctx, next, current, vars)
        })
    }

    fn assert_exact_constrained_bounds_require_generic_fallback(src: &str) {
        let (_module, ctx, _vars, next) = setup(src);
        let Expr::Exists(bounds, body) = &next.body.node else {
            panic!("Next must be a top-level EXISTS")
        };
        assert_eq!(bounds.len(), 1);
        let bound = &bounds[0];
        let domain = bound.domain.as_ref().expect("EXISTS must be bounded");
        let pattern = match_constrained_subset_exists(bound.name.node.as_str(), domain, body)
            .expect("constraints in exact positions 0/1 must match");
        assert!(constrained_bounds_require_generic_fallback(&ctx, &pattern));
    }

    #[test]
    fn exact_flag_parser_accepts_only_one() {
        assert!(parse_exists_name_id_bindings_enabled(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!parse_exists_name_id_bindings_enabled(value));
        }
    }

    #[test]
    fn first_guard_prepare_skips_multi_bound_and_empty_domains_at_call_site() {
        fn run(src: &str) -> (usize, u64) {
            let (_module, mut ctx, vars, next) = setup(src);
            let current = State::from_pairs([("y", Value::int(0))]);
            reset_first_guard_sched_test_prepares();
            let successors = enumerate_with_mode(false, &mut ctx, &next, &current, &vars)
                .expect("EXISTS enumeration should succeed");
            (successors.len(), first_guard_sched_test_prepares())
        }

        let nonempty_single = r#"
---- MODULE FirstGuardPrepareSingle ----
VARIABLE y
Next == \E a \in {1} : y' = a
====
"#;
        assert_eq!(run(nonempty_single), (1, 1));

        let empty_single = r#"
---- MODULE FirstGuardPrepareEmpty ----
VARIABLE y
Next == \E a \in {} : y' = a
====
"#;
        assert_eq!(run(empty_single), (0, 0));

        let nonempty_multi = r#"
---- MODULE FirstGuardPrepareMulti ----
VARIABLE y
Next == \E a \in {1}, b \in {2} : y' = a
====
"#;
        assert_eq!(run(nonempty_multi), (1, 0));
    }

    #[test]
    fn quorum_subset_prune_preserves_top_level_and_conjunct_successor_order() {
        let top_level = r#"
---- MODULE QuorumSubsetTopLevel ----
EXTENDS FiniteSets
VARIABLE y
Node(v) == v[1]
IsQuorum(Q) == Q \in {{3}, {1, 2}}
Next == \E delivered \in SUBSET {<<1, 10>>, <<2, 20>>, <<3, 30>>}:
          /\ IsQuorum({Node(v) : v \in delivered})
          /\ y' = Cardinality(delivered)
====
"#;
        let conjunct = r#"
---- MODULE QuorumSubsetConjunct ----
EXTENDS FiniteSets
VARIABLE y
Node(v) == v[1]
IsQuorum(Q) == Q \in {{3}, {1, 2}}
Next ==
  /\ \E delivered \in SUBSET {<<1, 10>>, <<2, 20>>, <<3, 30>>}:
       /\ IsQuorum({Node(v) : v \in delivered})
       /\ y' = Cardinality(delivered)
  /\ y = 0
====
"#;

        for src in [top_level, conjunct] {
            let (_module, mut ctx, vars, next) = setup(src);
            let current = State::from_pairs([("y", Value::int(0))]);

            reset_quorum_subset_prune_test_activations();
            let baseline = with_quorum_subset_prune_test_override(false, || {
                enumerate_with_mode(false, &mut ctx, &next, &current, &vars)
            })
            .expect("ordinary SUBSET enumeration should succeed");
            assert_eq!(quorum_subset_prune_test_activations(), 0);

            reset_quorum_subset_prune_test_activations();
            let pruned = with_quorum_subset_prune_test_override(true, || {
                enumerate_with_mode(false, &mut ctx, &next, &current, &vars)
            })
            .expect("certified quorum SUBSET enumeration should succeed");
            assert_eq!(quorum_subset_prune_test_activations(), 1);

            let ys = |states: &[State]| {
                states
                    .iter()
                    .map(|state| state.get("y").and_then(Value::as_i64).unwrap())
                    .collect::<Vec<_>>()
            };
            assert_eq!(ys(&baseline), vec![1, 2]);
            assert_eq!(ys(&pruned), ys(&baseline));
        }
    }

    #[test]
    fn quorum_subset_prune_reaches_nested_apply_let_conjunct_path() {
        let src = r#"
---- MODULE QuorumConjunctLet ----
VARIABLES out, tag
Node(v) == v[1]
IsQuorum(Q) == Q \in {{3}, {1, 2}}
Base == {<<1, 10>>, <<2, 20>>, <<3, 30>>}
Action(self) ==
  LET newV == <<self, 77>> IN
    /\ self = 9
    /\ \E delivered \in SUBSET Base:
         /\ IsQuorum({Node(v) : v \in delivered})
         /\ out' = delivered
    /\ tag' = newV[2]
Next == \E self \in {9}: Action(self)
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        let current = State::from_pairs([("out", Value::empty_set()), ("tag", Value::int(0))]);
        let initial_depth = ctx.local_stack_len();

        reset_quorum_subset_prune_test_activations();
        let baseline = with_quorum_subset_prune_test_override(false, || {
            enumerate_with_mode(false, &mut ctx, &next, &current, &vars)
        })
        .expect("ordinary nested SUBSET enumeration should succeed");
        assert_eq!(quorum_subset_prune_test_activations(), 0);

        reset_quorum_subset_prune_test_activations();
        let pruned = with_quorum_subset_prune_test_override(true, || {
            enumerate_with_mode(false, &mut ctx, &next, &current, &vars)
        })
        .expect("certified nested quorum SUBSET enumeration should succeed");
        assert_eq!(quorum_subset_prune_test_activations(), 1);
        assert_eq!(pruned, baseline);
        assert_eq!(ctx.local_stack_len(), initial_depth);

        let v1 = Value::tuple([Value::int(1), Value::int(10)]);
        let v2 = Value::tuple([Value::int(2), Value::int(20)]);
        let v3 = Value::tuple([Value::int(3), Value::int(30)]);
        assert_eq!(
            pruned
                .iter()
                .map(|state| state.get("out").cloned().unwrap())
                .collect::<Vec<_>>(),
            vec![Value::set([v3]), Value::set([v1, v2])],
        );
        assert!(pruned
            .iter()
            .all(|state| state.get("tag") == Some(&Value::int(77))));
    }

    #[test]
    fn quorum_subset_runtime_decline_reuses_domain_and_preserves_error() {
        let src = r#"
---- MODULE QuorumRuntimeFallback ----
VARIABLE out
Node(v) == v[1]
IsQuorum(Q) == Q \in {{3}}
Base == {<<>>, <<3, 30>>}
Next == \E delivered \in SUBSET Base:
          /\ IsQuorum({Node(v) : v \in delivered})
          /\ out' = delivered
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        let current = State::from_pairs([("out", Value::empty_set())]);

        reset_quorum_subset_prune_test_activations();
        let baseline = with_quorum_subset_prune_test_override(false, || {
            enumerate_with_mode(false, &mut ctx, &next, &current, &vars)
        })
        .expect_err("the empty tuple must retain Node's indexing error");
        assert_eq!(quorum_subset_prune_test_activations(), 0);

        reset_quorum_subset_prune_test_activations();
        let declined = with_quorum_subset_prune_test_override(true, || {
            enumerate_with_mode(false, &mut ctx, &next, &current, &vars)
        })
        .expect_err("runtime projection decline must use the generic body");
        assert_eq!(quorum_subset_prune_test_activations(), 0);
        assert!(matches!(baseline, EvalError::IndexOutOfBounds { .. }));
        assert!(matches!(declined, EvalError::IndexOutOfBounds { .. }));
        assert_eq!(baseline.to_string(), declined.to_string());
    }

    #[test]
    fn single_name_id_uses_inline_storage_and_exact_interned_id() {
        let bounds = [simple_bound("actor")];
        let names = BoundNames::for_mode(&bounds, true);
        let BoundNames::NameIds(ids) = &names else {
            panic!("enabled experiment must prepare NameIds")
        };
        assert_eq!(ids.as_slice(), &[intern_name("actor")]);
        assert!(!ids.spilled(), "one bound name must remain inline");
        assert!(matches!(names.get(0), BoundName::NameId(id) if id == intern_name("actor")));

        assert!(matches!(
            BoundNames::for_mode(&bounds, false),
            BoundNames::LegacyArcs(names) if names.len() == 1
        ));
    }

    #[test]
    fn name_id_binding_preserves_domain_then_body_shadowing_and_cleanup() {
        let src = r#"
---- MODULE ExistsNameIdShadow ----
EXTENDS Integers
CONSTANT x
VARIABLE y
Next == \E x \in {x + 1} : y' = x
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        ctx.push_binding(Arc::from("x"), Value::int(7));
        let initial_depth = ctx.local_stack_len();
        let current = State::from_pairs([("y", Value::int(0))]);

        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("EXISTS enumeration should succeed");
            assert_eq!(successors.len(), 1);
            assert_eq!(successors[0].get("y"), Some(&Value::int(8)));
            assert_eq!(ctx.local_stack_len(), initial_depth);
            assert_eq!(ctx.lookup_binding("x"), Some(Value::int(7)));
        }
    }

    #[test]
    fn name_id_binding_preserves_dependent_multi_bound_domains() {
        let src = r#"
---- MODULE ExistsNameIdMultiBound ----
EXTENDS Integers
CONSTANT seed
VARIABLE y
Next == \E a \in {seed, seed + 1}, b \in {a + 10} : y' = b
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        ctx.push_binding(Arc::from("seed"), Value::int(1));
        let initial_depth = ctx.local_stack_len();
        let current = State::from_pairs([("y", Value::int(0))]);

        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("dependent multi-bound EXISTS should succeed");
            let mut ys: Vec<i64> = successors
                .iter()
                .map(|state| state.get("y").and_then(Value::as_i64).unwrap())
                .collect();
            ys.sort_unstable();
            assert_eq!(ys, vec![11, 12]);
            assert_eq!(ctx.local_stack_len(), initial_depth);
            assert_eq!(ctx.lookup_binding("seed"), Some(Value::int(1)));
        }
    }

    #[test]
    fn name_id_binding_covers_generic_singleton_conjunct_route() {
        let src = r#"
---- MODULE ExistsNameIdGenericConjunct ----
VARIABLES y, z
Next ==
  /\ \E x \in {1, 2}: y' = x
  /\ z' = 9
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        let initial_depth = ctx.local_stack_len();
        let current = State::from_pairs([("y", Value::int(0)), ("z", Value::int(0))]);

        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("generic conjunct EXISTS should succeed");
            let mut ys: Vec<i64> = successors
                .iter()
                .map(|state| state.get("y").and_then(Value::as_i64).unwrap())
                .collect();
            ys.sort_unstable();
            assert_eq!(ys, vec![1, 2]);
            assert!(successors
                .iter()
                .all(|state| state.get("z") == Some(&Value::int(9))));
            assert_eq!(ctx.local_stack_len(), initial_depth);
        }
    }

    #[test]
    fn name_id_binding_covers_constrained_subset_conjunct_route() {
        let src = r#"
---- MODULE ExistsNameIdConstrained ----
EXTENDS Integers
CONSTANT seed
VARIABLE y
Next ==
  /\ \E r \in SUBSET {seed, seed + 1} :
       /\ r \subseteq {seed, seed + 1}
       /\ {seed} \subseteq r
       /\ y' = IF r = {seed} THEN 1 ELSE 2
  /\ y = 0
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        ctx.push_binding(Arc::from("seed"), Value::int(4));
        let initial_depth = ctx.local_stack_len();
        let current = State::from_pairs([("y", Value::int(0))]);

        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("constrained-subset conjunct EXISTS should succeed");
            let mut ys: Vec<i64> = successors
                .iter()
                .map(|state| state.get("y").and_then(Value::as_i64).unwrap())
                .collect();
            ys.sort_unstable();
            assert_eq!(ys, vec![1, 2]);
            assert_eq!(ctx.local_stack_len(), initial_depth);
            assert_eq!(ctx.lookup_binding("seed"), Some(Value::int(4)));
        }
    }

    #[test]
    fn name_id_binding_covers_constrained_subset_top_level_route() {
        let src = r#"
---- MODULE ExistsNameIdConstrainedTopLevel ----
EXTENDS Integers
CONSTANT seed
VARIABLE y
Next ==
  \E r \in SUBSET {seed, seed + 1}:
    /\ r \subseteq {seed, seed + 1}
    /\ {seed} \subseteq r
    /\ y' = IF r = {seed} THEN 1 ELSE 2
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        ctx.push_binding(Arc::from("seed"), Value::int(4));
        let initial_depth = ctx.local_stack_len();
        let current = State::from_pairs([("y", Value::int(0))]);

        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("top-level constrained-subset EXISTS should succeed");
            let mut ys: Vec<i64> = successors
                .iter()
                .map(|state| state.get("y").and_then(Value::as_i64).unwrap())
                .collect();
            ys.sort_unstable();
            assert_eq!(ys, vec![1, 2]);
            assert_eq!(ctx.local_stack_len(), initial_depth);
            assert_eq!(ctx.lookup_binding("seed"), Some(Value::int(4)));
        }
    }

    #[test]
    fn constrained_subset_falls_back_when_a_bound_mentions_the_inner_name() {
        let src = r#"
---- MODULE ExistsConstrainedInnerName ----
CONSTANT r
VARIABLE y
Next ==
  \E r \in SUBSET {1}:
    /\ r \subseteq r
    /\ {} \subseteq r
    /\ y' = IF r = {} THEN 0 ELSE 1
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        ctx.push_binding(Arc::from("r"), Value::empty_set());
        let initial_depth = ctx.local_stack_len();
        let current = State::from_pairs([("y", Value::int(-1))]);

        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("binder-dependent constraints must use generic SUBSET enumeration");
            let mut ys: Vec<i64> = successors
                .iter()
                .map(|state| state.get("y").and_then(Value::as_i64).unwrap())
                .collect();
            ys.sort_unstable();
            assert_eq!(ys, vec![0, 1]);
            assert_eq!(ctx.local_stack_len(), initial_depth);
            assert_eq!(ctx.lookup_binding("r"), Some(Value::empty_set()));
        }
    }

    #[test]
    fn constrained_subset_preserves_prefix_short_circuit_and_prime_order() {
        let false_prefix = r#"
---- MODULE ExistsConstrainedFalsePrefix ----
EXTENDS Integers
VARIABLE y
Next ==
  \E r \in SUBSET {1}:
    /\ FALSE
    /\ r \subseteq (1 \div 0)
    /\ {} \subseteq r
    /\ y' = 1
====
"#;
        let (_module, mut ctx, vars, next) = setup(false_prefix);
        let current = State::from_pairs([("y", Value::int(0))]);
        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("FALSE prefix must short-circuit before the later error");
            assert!(successors.is_empty());
        }

        let primed_prefix = r#"
---- MODULE ExistsConstrainedPrimePrefix ----
VARIABLES x, y
Next ==
  \E r \in SUBSET {1}:
    /\ x' = {1}
    /\ r \subseteq x'
    /\ {} \subseteq r
    /\ y' = IF r = {} THEN 0 ELSE 1
====
"#;
        let (_module, mut ctx, vars, next) = setup(primed_prefix);
        let current = State::from_pairs([("x", Value::empty_set()), ("y", Value::int(-1))]);
        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("primed prefix must be observed before subset constraints");
            let mut ys: Vec<i64> = successors
                .iter()
                .map(|state| state.get("y").and_then(Value::as_i64).unwrap())
                .collect();
            ys.sort_unstable();
            assert_eq!(ys, vec![0, 1]);
            assert!(successors
                .iter()
                .all(|state| state.get("x") == Some(&Value::set([Value::int(1)]))));
        }
    }

    #[test]
    fn constrained_subset_exact_action_level_bounds_require_generic_fallback() {
        let direct_prime_upper = r#"
---- MODULE ExistsConstrainedDirectPrimeUpper ----
VARIABLES x, y
Next ==
  \E r \in SUBSET {1}:
    /\ r \subseteq x'
    /\ {} \subseteq r
    /\ y' = IF r = {} THEN 0 ELSE 1
====
"#;
        assert_exact_constrained_bounds_require_generic_fallback(direct_prime_upper);

        let hidden_prime_lower = r#"
---- MODULE ExistsConstrainedHiddenPrimeLower ----
VARIABLES x, y
HiddenLower == x'
Next ==
  \E r \in SUBSET {1}:
    /\ r \subseteq {1}
    /\ HiddenLower \subseteq r
    /\ y' = IF r = {} THEN 0 ELSE 1
====
"#;
        assert_exact_constrained_bounds_require_generic_fallback(hidden_prime_lower);
    }

    #[test]
    fn constrained_subset_nonset_and_lazy_bounds_use_generic_semantics() {
        let scalar_upper = r#"
---- MODULE ExistsConstrainedScalarUpper ----
VARIABLE y
Next ==
  \E r \in SUBSET {}:
    /\ r \subseteq 1
    /\ {} \subseteq r
    /\ y' = 1
====
"#;
        let (_module, mut ctx, vars, next) = setup(scalar_upper);
        let current = State::from_pairs([("y", Value::int(0))]);
        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("empty generic LHS must retain candidate-sensitive subset semantics");
            assert_eq!(successors.len(), 1);
            assert_eq!(successors[0].get("y"), Some(&Value::int(1)));
        }

        let scalar_lower = r#"
---- MODULE ExistsConstrainedScalarLower ----
VARIABLE y
Next ==
  \E r \in SUBSET {}:
    /\ r \subseteq {}
    /\ 1 \subseteq r
    /\ y' = 1
====
"#;
        let (_module, mut ctx, vars, next) = setup(scalar_lower);
        let current = State::from_pairs([("y", Value::int(0))]);
        for enabled in [false, true] {
            let error = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect_err("scalar lower bound must retain the generic TypeError");
            assert!(matches!(error, EvalError::TypeError { .. }), "{error:?}");
        }

        let lazy_lower = r#"
---- MODULE ExistsConstrainedLazyLower ----
EXTENDS Integers
VARIABLE y
Next ==
  \E r \in SUBSET {1}:
    /\ r \subseteq {1}
    /\ (1..1) \subseteq r
    /\ y' = 1
====
"#;
        let (_module, mut ctx, vars, next) = setup(lazy_lower);
        let current = State::from_pairs([("y", Value::int(0))]);
        for enabled in [false, true] {
            let successors = enumerate_with_mode(enabled, &mut ctx, &next, &current, &vars)
                .expect("lazy set lower bound must fall back to generic enumeration");
            assert_eq!(successors.len(), 1);
            assert_eq!(successors[0].get("y"), Some(&Value::int(1)));
        }
    }

    #[test]
    fn name_id_binding_error_path_restores_outer_binding() {
        let src = r#"
---- MODULE ExistsNameIdErrorCleanup ----
EXTENDS Integers
VARIABLE y
Next == \E x \in {1} : y' = 1 \div 0
====
"#;
        let (_module, mut ctx, vars, next) = setup(src);
        ctx.push_binding(Arc::from("x"), Value::int(99));
        let initial_depth = ctx.local_stack_len();
        let current = State::from_pairs([("y", Value::int(0))]);

        let result = enumerate_with_mode(true, &mut ctx, &next, &current, &vars);
        assert!(result.is_err(), "body evaluation error must propagate");
        assert_eq!(ctx.local_stack_len(), initial_depth);
        assert_eq!(ctx.lookup_binding("x"), Some(Value::int(99)));
    }
}
