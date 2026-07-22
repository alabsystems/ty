// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Non-chained INSTANCE evaluation and substitution helpers.
//!
//! Contains `eval_module_ref` (M!Op), `get_instance_info`, `compose_substitutions`,
//! lazy substitution binding builders, and AST utility wrappers.
//!
//! Part of #1643 (module_ref.rs decomposition).

mod binding_builders;
mod instance_resolution;

// Re-export public API — preserves all existing import paths.
pub(crate) use binding_builders::{
    build_lazy_subst_bindings, build_lazy_subst_bindings_with_local_ops,
};
pub use binding_builders::{expr_has_any_prime, expr_has_primed_param};
pub(super) use instance_resolution::get_instance_info;
pub use instance_resolution::{apply_substitutions, compose_substitutions};
pub(crate) use instance_resolution::{inherit_substitutions, qualify_instance_expr_substitutions};

#[cfg(debug_assertions)]
use super::super::debug_module_ref;
use super::super::{eval, EvalCtx, EvalError, EvalResult, InstanceInfo, OpEnv};
use super::module_ref_cache::{ModuleRefScopeEntry, ModuleRefScopeKey, MODULE_REF_CACHES};
use super::module_ref_label::select_named_label;
use crate::binding_chain::BindingValue;
use crate::eval_dispatch::EMPTY_EAGER_SUBST;
use crate::value::Value;
use smallvec::SmallVec;
use std::sync::Arc;
use tla_core::ast::Expr;
use tla_core::name_intern::intern_name;
use tla_core::{Span, Spanned};

// Part of #4114: Cache debug env vars with OnceLock instead of calling
// std::env::var on every INSTANCE evaluation.
debug_flag!(debug_3447, "TY_DEBUG_3447");

struct PreparedNamedModuleRefCall {
    ctx: EvalCtx,
    op_def: Arc<tla_core::ast::OperatorDef>,
    saved_param_bindings: SmallVec<[(tla_core::name_intern::NameId, Value); 4]>,
}

/// Whether evaluator dispatch reaches a registered named INSTANCE directly.
///
/// `Def!n` conjunct selection and `Def!Label` selection take precedence over
/// INSTANCE resolution. Callers that intend to enter a registered instance
/// body directly must first prove that neither selector applies.
#[doc(hidden)]
pub fn registered_named_module_ref_dispatch_is_direct(
    ctx: &EvalCtx,
    instance_name: &str,
    op_name: &str,
) -> bool {
    let target_def = ctx.get_op(instance_name);
    let numeric_selector_preempts = op_name.parse::<usize>().is_ok_and(|index| index > 0)
        && target_def.is_some_and(|def| matches!(&def.body.node, Expr::And(..)));
    !numeric_selector_preempts
        && target_def.is_none_or(|def| select_named_label(def, op_name).is_none())
}

/// Prepare the exact evaluation scope for a registered named INSTANCE call.
///
/// This is the call-by-value half of normal `M!Op(args)` evaluation: actuals
/// are evaluated in the caller context, INSTANCE substitutions retain their
/// definition-site binding chain, instance-local operators take precedence,
/// and formal values are then installed in the rebuilt module scope. It is
/// exposed for execution engines that must enumerate the operator body rather
/// than immediately evaluate it as a value.
///
/// Config replacements and non-INSTANCE `Def!label` selection are dispatch
/// concerns and must be handled before calling this helper.
#[doc(hidden)]
pub fn prepare_registered_named_module_ref_call(
    ctx: &EvalCtx,
    instance_name: &str,
    op_name: &str,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<(EvalCtx, Arc<tla_core::ast::OperatorDef>)> {
    let instance_info = ctx
        .get_instance(instance_name)
        .ok_or_else(|| EvalError::UndefinedOp {
            name: format!("{instance_name}!{op_name}"),
            span,
        })?;
    let op_def = ctx
        .get_instance_op_arc(&instance_info.module_name, op_name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedOp {
            name: format!("{instance_name}!{op_name}"),
            span,
        })?;
    if op_def.params.len() != args.len() {
        return Err(EvalError::ArityMismatch {
            op: format!("{instance_name}!{op_name}"),
            expected: op_def.params.len(),
            got: args.len(),
            span,
        });
    }

    let prepared = prepare_resolved_named_module_ref_call(
        ctx,
        instance_name,
        op_name,
        instance_info,
        op_def,
        args,
    )?;
    Ok((prepared.ctx, prepared.op_def))
}

fn prepare_resolved_named_module_ref_call(
    ctx: &EvalCtx,
    instance_name: &str,
    op_name: &str,
    instance_info: &InstanceInfo,
    op_def: Arc<tla_core::ast::OperatorDef>,
    args: &[Spanned<Expr>],
) -> EvalResult<PreparedNamedModuleRefCall> {
    // Fix #2364: Cache the composed substitutions and merged local_ops in
    // pre-wrapped Arcs so SUBST_CACHE keys remain stable across eval entries.
    let scope_key = ModuleRefScopeKey {
        shared_id: ctx.shared.id,
        instance_name_id: intern_name(instance_name),
        outer_subs_id: ctx
            .instance_substitutions
            .as_ref()
            .map_or(0, |subs| Arc::as_ptr(subs) as usize),
        outer_local_ops_id: ctx
            .local_ops
            .as_ref()
            .map_or(0, |ops| Arc::as_ptr(ops) as usize),
    };

    let scope_entry = MODULE_REF_CACHES
        .with(|c| c.borrow().module_ref_scope.get(&scope_key).cloned())
        .unwrap_or_else(|| {
            let effective_substitutions = ctx.compute_effective_instance_substitutions(
                &instance_info.module_name,
                &instance_info.substitutions,
            );

            let mut instance_local_ops: OpEnv = ctx
                .shared
                .instance_ops
                .get(&instance_info.module_name)
                .cloned()
                .unwrap_or_default();
            if let Some(parent_local_ops) = ctx.local_ops.as_ref() {
                for (name, def) in parent_local_ops.iter() {
                    if !instance_local_ops.contains_key(name.as_str()) {
                        instance_local_ops.insert(name.clone(), def.clone());
                    }
                }
            }

            let entry = ModuleRefScopeEntry {
                effective_subs_arc: Arc::new(effective_substitutions),
                local_ops_arc: Arc::new(instance_local_ops),
            };
            MODULE_REF_CACHES.with(|c| {
                c.borrow_mut()
                    .module_ref_scope
                    .insert(scope_key, entry.clone())
            });
            entry
        });

    let has_effective_substitutions = !scope_entry.effective_subs_arc.is_empty();
    debug_eprintln!(
        debug_module_ref(),
        "[MODULE_REF] {}!{}: module={}, subs=[{}]",
        instance_name,
        op_name,
        instance_info.module_name,
        scope_entry
            .effective_subs_arc
            .iter()
            .map(|s| format!("{} <- {:?}", s.from.node, s.to.node))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Evaluate every actual in the caller scope before installing any formal.
    let mut bindings: SmallVec<[(Arc<str>, Value); 4]> = SmallVec::new();
    for (param, arg) in op_def.params.iter().zip(args.iter()) {
        let value = eval(ctx, arg)?;
        bindings.push((Arc::from(param.name.node.as_str()), value));
    }
    let saved_param_bindings: SmallVec<[(tla_core::name_intern::NameId, Value); 4]> = bindings
        .iter()
        .map(|(name, value)| (intern_name(name), value.clone()))
        .collect();

    let is_chained =
        ctx.chained_ref_eval || (ctx.local_ops.is_some() && has_effective_substitutions);
    let mut new_ctx = ctx.with_module_scope_arced_subs(
        Arc::clone(&scope_entry.local_ops_arc),
        bindings,
        Arc::clone(&scope_entry.effective_subs_arc),
    );
    if is_chained {
        new_ctx.stable_mut().chained_ref_eval = true;
    }

    // Uniform lazy substitution bindings preserve the definition-site chain.
    if has_effective_substitutions {
        new_ctx.stable_mut().eager_subst_bindings = Some(Arc::clone(&EMPTY_EAGER_SUBST));
        // SAFETY: every expression pointer belongs to effective_subs_arc,
        // retained by new_ctx.instance_substitutions for the body lifetime.
        new_ctx.bindings = binding_builders::build_lazy_subst_bindings_with_local_ops(
            &ctx.bindings,
            ctx.local_ops.clone(),
            &scope_entry.effective_subs_arc,
        );
        for (name_id, value) in saved_param_bindings.iter().rev() {
            new_ctx.bindings = new_ctx.bindings.cons_local(
                *name_id,
                BindingValue::eager(value.clone()),
                new_ctx.binding_depth,
            );
        }
    }

    crate::cache::lifecycle::clear_for_eval_scope_boundary_if_needed(
        ctx.state_env.map_or(0, |s| s.identity()),
    );
    Ok(PreparedNamedModuleRefCall {
        ctx: new_ctx,
        op_def,
        saved_param_bindings,
    })
}

pub(crate) fn eval_module_ref(
    ctx: &EvalCtx,
    instance_name: &str,
    op_name: &str,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    // Check for conjunct selection: Def!n where Def is an operator with a conjunction body
    // and n is a numeric index (1-based)
    if let Ok(conjunct_idx) = op_name.parse::<usize>() {
        if conjunct_idx > 0 {
            // This might be conjunct selection, not module reference
            if let Some(def) = ctx.get_op(instance_name) {
                // Check if the operator body is a conjunction
                if let Expr::And(_, _) = &def.body.node {
                    // Collect all conjuncts from the definition
                    let conjuncts = tla_core::collect_conjuncts_v(&def.body);
                    let idx = conjunct_idx - 1; // Convert to 0-based
                    if idx < conjuncts.len() {
                        // Evaluate the selected conjunct
                        return eval(ctx, &conjuncts[idx]);
                    }
                    return Err(EvalError::UndefinedOp {
                        name: format!(
                            "{}!{} (conjunct index {} out of range, definition has {} conjuncts)",
                            instance_name,
                            op_name,
                            conjunct_idx,
                            conjuncts.len()
                        ),
                        span,
                    });
                }
            }
        }
    }

    if let Some(def) = ctx.get_op(instance_name) {
        if let Some(selected) = select_named_label(def, op_name) {
            if !args.is_empty() {
                return Err(EvalError::ArityMismatch {
                    op: format!("{instance_name}!{op_name}"),
                    expected: 0,
                    got: args.len(),
                    span,
                });
            }
            return eval(ctx, &selected);
        }
    }

    // Resolve the instance.
    //
    // TLC allows nested named instances: when evaluating `Outer!Op` we must be able to resolve
    // instance references inside that module (e.g., `cleanInstance!Spec`) even if those
    // instances are not declared in the main module.
    //
    // We support this by:
    // 1) looking for a globally-registered instance (from loading the main/extended modules), and
    // 2) falling back to a locally-visible operator definition whose body is an `InstanceExpr`
    //    (available via `instance_local_ops` when evaluating an instanced module).
    let instance_info: InstanceInfo = if let Some(info) = ctx.get_instance(instance_name) {
        info.clone()
    } else if let Some(def) = ctx.get_op(instance_name) {
        match &def.body.node {
            Expr::InstanceExpr(module_name, substitutions) => InstanceInfo {
                module_name: module_name.clone(),
                substitutions: substitutions.clone(),
            },
            _ => {
                return Err(EvalError::UndefinedOp {
                    name: format!("{instance_name}!{op_name}"),
                    span,
                });
            }
        }
    } else {
        return Err(EvalError::UndefinedOp {
            name: format!("{instance_name}!{op_name}"),
            span,
        });
    };

    // Get the operator from the instanced module.
    //
    // Part of #liveness-leaf-cache: clone the operator definition by Arc, not by
    // deep value. `OperatorDef` carries the full `Spanned<Expr>` body; the prior
    // `def.clone()` deep-cloned the entire instanced-operator AST on EVERY
    // `M!Op` call. On INSTANCE-routed liveness leaves (e.g. `Sched!Allocator`)
    // this fires per-transition, so the Arc clone removes a per-call AST copy on
    // the hot path. `op_def` is used only via `Deref` (`.params`, `.body`).
    //
    // Also: `compute_effective_instance_substitutions` is only needed for the
    // op-not-found substitution fallback and for building the scope-cache entry
    // on a MISS. Defer it so the common (op found + scope cached) path skips it.
    let op_def: Arc<tla_core::ast::OperatorDef> = match ctx
        .get_instance_op_arc(&instance_info.module_name, op_name)
    {
        Some(def) => Arc::clone(def),
        None => {
            let effective_substitutions = ctx.compute_effective_instance_substitutions(
                &instance_info.module_name,
                &instance_info.substitutions,
            );
            if let Some(sub) = effective_substitutions
                .iter()
                .find(|sub| sub.from.node == op_name)
            {
                return eval(ctx, &sub.to);
            }

            // Builtin stdlib module fallback: stdlib/community modules
            // (Graphs, SequencesExt, ...) are implemented natively and never
            // loaded from `.tla` sources, so their operators are absent from
            // `instance_ops`. A named instance of such a module (e.g.,
            // `G == INSTANCE Graphs` in a LET, then `G!Transpose(...)`)
            // must dispatch to the builtin evaluator — mirroring TLC's Java
            // module overrides. Fail-closed: only ops declared in the
            // module's stdlib table are eligible, and an unimplemented
            // builtin still returns UndefinedOp below.
            if tla_core::stdlib_module_has_op(&instance_info.module_name, op_name) {
                if let Some(v) = super::builtin_dispatch::eval_builtin(ctx, op_name, args, span)? {
                    return Ok(v);
                }
            }

            return Err(EvalError::UndefinedOp {
                name: format!("{instance_name}!{op_name}"),
                span,
            });
        }
    };

    // Check arity
    if op_def.params.len() != args.len() {
        return Err(EvalError::ArityMismatch {
            op: format!("{instance_name}!{op_name}"),
            expected: op_def.params.len(),
            got: args.len(),
            span,
        });
    }

    let prepared = prepare_resolved_named_module_ref_call(
        ctx,
        instance_name,
        op_name,
        &instance_info,
        op_def,
        args,
    )?;
    let PreparedNamedModuleRefCall {
        ctx: new_ctx,
        op_def,
        saved_param_bindings,
    } = prepared;
    let result = eval(&new_ctx, &op_def.body);

    // #3447 debug: trace ShowsSafeAt calls with args and votes binding
    if debug_3447() && op_name == "ShowsSafeAt" {
        if let Ok(ref v) = result {
            if v == &Value::Bool(false) {
                // Print args for the failing call
                let args_str: Vec<String> = saved_param_bindings
                    .iter()
                    .zip(op_def.params.iter())
                    .map(|((_, val), param)| format!("{}={:?}", param.name.node, val))
                    .collect();
                // Look up votes in the binding chain
                let votes_val = {
                    let votes_id = intern_name("votes");
                    new_ctx
                        .bindings
                        .lookup(votes_id)
                        .map(|(bv, source)| {
                            bv.get_if_ready(crate::cache::StateLookupMode::Current, source)
                                .map(|v| format!("{:?}", v))
                                .unwrap_or_else(|| "UNFORCED".to_string())
                        })
                        .unwrap_or_else(|| "NOT_IN_CHAIN".to_string())
                };
                let maxbal_val = {
                    let maxbal_id = intern_name("maxBal");
                    new_ctx
                        .bindings
                        .lookup(maxbal_id)
                        .map(|(bv, source)| {
                            bv.get_if_ready(crate::cache::StateLookupMode::Current, source)
                                .map(|v| format!("{:?}", v))
                                .unwrap_or_else(|| "UNFORCED".to_string())
                        })
                        .unwrap_or_else(|| "NOT_IN_CHAIN".to_string())
                };
                eprintln!(
                    "[DEBUG_3447] ShowsSafeAt=FALSE args=[{}] votes={} maxBal={} state_env_id={}",
                    args_str.join(", "),
                    votes_val,
                    maxbal_val,
                    new_ctx.state_env.map_or(0, |s| s.identity()),
                );
            }
        }
    }

    result
}
