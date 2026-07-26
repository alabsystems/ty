// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Recursive action-instance splitting.
//!
//! Implements the TLC-style expansion of the Next relation into a flat list of
//! action instances by expanding disjunctions and bounded existentials.
//!
//! Extracted from action_instance/mod.rs to reduce file size.

use super::{ActionInstance, SplitCtx, SplitWrapper};
use crate::enumerate::{classify_iter_error_for_speculative_path, IterDomainAction};
use crate::error::EvalResult;
use crate::eval::{
    apply_substitutions, compose_substitutions, eval_iter_set_tlc_normalized, try_eval_const_level,
    EvalCtx,
};
use crate::value::Value;
use crate::OpEnv;
use std::sync::Arc;
use tla_core::ast::{BoundVar, Expr, ModuleTarget, Substitution};
use tla_core::{Span, Spanned};

pub(super) fn split_action_instances_rec(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    split: &SplitCtx,
    out: &mut Vec<ActionInstance>,
) -> EvalResult<()> {
    match &expr.node {
        Expr::Or(a, b) => {
            split_action_instances_rec(ctx, a, split, out)?;
            split_action_instances_rec(ctx, b, split, out)?;
        }

        // LET wrapper - transparent for splitting but must be preserved for evaluation parity.
        Expr::Let(defs, body) => {
            let mut next = split.clone();
            next.wrappers.push(SplitWrapper::Let(defs.clone()));
            split_action_instances_rec(ctx, body, &next, out)?;
        }

        // IF expression: split named branch actions, but keep direct IF-shaped
        // actions monolithic. Preserve branch selection with IF wrappers: an
        // action-level `cond /\ action` would enumerate every proof of an
        // existential condition and could emit duplicate successors, whereas
        // IF consumes the condition as one Boolean value. A direct
        // `Next == IF ... THEN x' = ... ELSE ...` has no branch operator name
        // for safe bytecode compilation; keeping it as one leaf lets the caller
        // name it `Next` and compile the rewritten branch bytecode.
        Expr::If(cond, then_branch, else_branch) => {
            let cond_expr = (**cond).clone();
            let mut branch_actions = Vec::new();

            let mut then_split = split.clone();
            then_split
                .wrappers
                .push(SplitWrapper::IfThen(cond_expr.clone()));
            split_action_instances_rec(ctx, then_branch, &then_split, &mut branch_actions)?;

            let mut else_split = split.clone();
            else_split.wrappers.push(SplitWrapper::IfElse(cond_expr));
            split_action_instances_rec(ctx, else_branch, &else_split, &mut branch_actions)?;

            if if_branch_split_has_stable_action_names(split, &branch_actions) {
                out.extend(branch_actions);
            } else {
                push_leaf_action(ctx, expr, split, out);
            }
        }

        // Bounded EXISTS: split only when all bounds have an enumerable const-level domain.
        Expr::Exists(bounds, body) => {
            let mut tmp = Vec::new();
            if try_split_exists_all_bounds(ctx, bounds, 0, body, split, &mut tmp)? {
                out.extend(tmp);
            } else {
                // If multiple bounds were flattened into a single Exists, try the reverse order as a
                // fallback. This is important for dependent domains, where later bounds may mention
                // earlier ones. If the bounds are already in spec order, the forward split succeeds
                // and we never take this path.
                if bounds.len() > 1 {
                    let mut rev = bounds.clone();
                    rev.reverse();
                    let mut tmp2 = Vec::new();
                    if try_split_exists_all_bounds(ctx, &rev, 0, body, split, &mut tmp2)? {
                        out.extend(tmp2);
                        return Ok(());
                    }
                }
                push_leaf_action(ctx, expr, split, out);
            }
        }

        // Zero-argument operator reference.
        Expr::Ident(name, _) => {
            let resolved = ctx.resolve_op_name(name.as_str());
            if let Some(def) = ctx.get_op(resolved) {
                let action_name = scoped_action_name(split, resolved);
                if def.params.is_empty() && !split.op_stack.iter().any(|n| n == &action_name) {
                    let mut next = split.clone();
                    next.action_name = Some(action_name.clone());
                    next.op_stack.push(action_name);
                    split_action_instances_rec(ctx, &def.body, &next, out)?;
                    return Ok(());
                }
            }
            push_leaf_action(ctx, expr, split, out);
        }

        // Module/Instance reference: M!Op(...) or IS(x,y)!Op(...).
        //
        // Treat module refs as leaf actions, pre-binding const-level actual args
        // to operator formal params so the instance carries TLC-style bindings.
        //
        // Part of #3100: Also bind parameterized target actuals (e.g., the `7` in
        // `Z7(7)!Next`) so that `split_action_meta` can distinguish `I(1)!Next`
        // from `I(2)!Next` via binding-based provenance matching.
        Expr::ModuleRef(target, op_name, args) => {
            let Some(target_name) = module_target_simple_name(target) else {
                push_leaf_action(ctx, expr, split, out);
                return Ok(());
            };

            // Part of #3100: Bind parameterized target actuals first.
            // For I(7)!Next, bind the target operator's formals (e.g., "n") to
            // the evaluated target actuals (e.g., 7). These bindings flow into
            // split_action_meta and allow the fairness matcher to distinguish
            // I(1)!Next from I(2)!Next.
            let ctx_with_target = if let ModuleTarget::Parameterized(tname, target_actuals) = target
            {
                if let Some(target_def) = ctx.get_op(tname) {
                    if target_def.params.len() == target_actuals.len() {
                        let mut target_values = Vec::with_capacity(target_actuals.len());
                        let mut all_const = true;
                        for arg in target_actuals {
                            if let Some(v) = try_eval_const_level(ctx, arg) {
                                target_values.push(v);
                            } else {
                                all_const = false;
                                break;
                            }
                        }
                        if all_const {
                            let target_bindings: Vec<(Arc<str>, Value)> = target_def
                                .params
                                .iter()
                                .zip(target_values)
                                .map(|(p, v)| (Arc::from(p.name.node.as_str()), v))
                                .collect();
                            Some(ctx.bind_all(target_bindings))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let base_ctx = ctx_with_target.as_ref().unwrap_or(ctx);

            let Some(instance_info) = resolve_instance_info(base_ctx, target_name) else {
                push_leaf_action(ctx, expr, split, out);
                return Ok(());
            };

            let resolved_op_name = base_ctx.resolve_op_name(op_name.as_str());
            let Some(def) = base_ctx.get_instance_op(&instance_info.module_name, resolved_op_name)
            else {
                push_leaf_action(ctx, expr, split, out);
                return Ok(());
            };
            if def.params.len() != args.len() {
                push_leaf_action(ctx, expr, split, out);
                return Ok(());
            }

            // Bind operator-call actuals on top of target bindings.
            let mut values: Vec<Value> = Vec::with_capacity(args.len());
            for arg in args {
                let Some(v) = try_eval_const_level(base_ctx, arg) else {
                    push_leaf_action(ctx, expr, split, out);
                    return Ok(());
                };
                values.push(v);
            }

            let bindings: Vec<(Arc<str>, Value)> = def
                .params
                .iter()
                .zip(values)
                .map(|(param, v)| (Arc::from(param.name.node.as_str()), v))
                .collect();
            let formal_binding_exclusions =
                formal_alias_exclusions_for_args(base_ctx, args, &bindings);
            let ctx2 = base_ctx.bind_all(bindings.clone());
            let ctx2 = ctx2.with_instance_substitutions(instance_info.substitutions.clone());
            let ctx2 = if let Some(local_ops) =
                merged_instance_local_ops(base_ctx, &instance_info.module_name)
            {
                ctx2.with_local_ops(local_ops)
            } else {
                ctx2
            };

            let mut next = split.clone();
            let action_name = format!("{target_name}!{resolved_op_name}");
            if next.op_stack.iter().any(|n| n == &action_name) {
                next.action_name = Some(action_name);
                next.formal_bindings = bindings;
                push_leaf_action(&ctx2, expr, &next, out);
                return Ok(());
            }
            next.action_name = Some(action_name.clone());
            next.instance_prefix = Some(target_name.to_string());
            next.formal_bindings = bindings;
            next.formal_binding_exclusions
                .extend(formal_binding_exclusions);
            next.op_stack.push(action_name);

            // Qualify the instanced operator body's OWN module-local references
            // BEFORE splicing in the INSTANCE WITH substitutions, excluding the
            // operator's formal-parameter names (bound as values in `ctx2`). A
            // WITH right-hand side is an outer-module expression, so qualifying
            // after substitution could rewrite an outer identifier that collides
            // by name with an instanced-module operator — silently binding the
            // wrong operator in a native-compiled next-state action (the
            // action-path analogue of the LockHS property-path arity bug).
            let param_names: Vec<String> = def.params.iter().map(|p| p.name.node.clone()).collect();
            let qualified = crate::checker_ops::qualify_instance_ops_with_bound(
                base_ctx,
                target,
                &instance_info.module_name,
                def.body.clone(),
                &param_names,
            );
            let substituted = apply_substitutions(&qualified, &instance_info.substitutions);
            split_action_instances_rec(&ctx2, &substituted, &next, out)?;
        }

        // Operator application specialization guarded by const-level arg evaluation.
        Expr::Apply(op_expr, args) => {
            let Expr::Ident(op_name, _) = &op_expr.node else {
                push_leaf_action(ctx, expr, split, out);
                return Ok(());
            };

            let resolved = ctx.resolve_op_name(op_name.as_str());
            let Some(def) = ctx.get_op(resolved) else {
                push_leaf_action(ctx, expr, split, out);
                return Ok(());
            };

            let action_name = scoped_action_name(split, resolved);
            if def.params.len() != args.len() || split.op_stack.iter().any(|n| n == &action_name) {
                push_leaf_action(ctx, expr, split, out);
                return Ok(());
            }

            let mut values: Vec<Value> = Vec::with_capacity(args.len());
            for arg in args {
                let Some(v) = try_eval_const_level(ctx, arg) else {
                    push_leaf_action(ctx, expr, split, out);
                    return Ok(());
                };
                values.push(v);
            }

            // Specialize: bind formals to values, then recurse into operator body.
            let bindings: Vec<(Arc<str>, Value)> = def
                .params
                .iter()
                .zip(values)
                .map(|(param, v)| (Arc::from(param.name.node.as_str()), v))
                .collect();
            let formal_binding_exclusions = formal_alias_exclusions_for_args(ctx, args, &bindings);
            let ctx2 = ctx.bind_all(bindings.clone());

            let mut next = split.clone();
            next.action_name = Some(action_name.clone());
            next.formal_bindings = bindings;
            next.formal_binding_exclusions
                .extend(formal_binding_exclusions);
            next.op_stack.push(action_name);
            split_action_instances_rec(&ctx2, &def.body, &next, out)?;
        }

        _ => push_leaf_action(ctx, expr, split, out),
    }
    Ok(())
}

fn push_leaf_action(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    split: &SplitCtx,
    out: &mut Vec<ActionInstance>,
) {
    let mut wrapped = expr.clone();
    for wrapper in split.wrappers.iter().rev() {
        wrapped = match wrapper {
            SplitWrapper::Let(defs) => {
                let span = wrapped.span;
                Spanned::new(Expr::Let(defs.clone(), Box::new(wrapped)), span)
            }
            SplitWrapper::IfThen(cond) => {
                let span = merged_span(cond.span, wrapped.span);
                let false_expr = Spanned::new(Expr::Bool(false), span);
                Spanned::new(
                    Expr::If(
                        Box::new(cond.clone()),
                        Box::new(wrapped),
                        Box::new(false_expr),
                    ),
                    span,
                )
            }
            SplitWrapper::IfElse(cond) => {
                let span = merged_span(cond.span, wrapped.span);
                let false_expr = Spanned::new(Expr::Bool(false), span);
                Spanned::new(
                    Expr::If(
                        Box::new(cond.clone()),
                        Box::new(false_expr),
                        Box::new(wrapped),
                    ),
                    span,
                )
            }
        };
    }

    let name = split.action_name.clone().or_else(|| match &expr.node {
        Expr::Ident(n, _) => Some(ctx.resolve_op_name(n.as_str()).to_string()),
        Expr::Apply(op, _) => match &op.node {
            Expr::Ident(n, _) => Some(ctx.resolve_op_name(n.as_str()).to_string()),
            _ => None,
        },
        Expr::ModuleRef(target, op, _) => match target {
            ModuleTarget::Named(t) | ModuleTarget::Parameterized(t, _) => Some(format!("{t}!{op}")),
            ModuleTarget::Chained(_) => None,
        },
        _ => None,
    });

    let complete_bindings = ctx.get_local_bindings();
    let mut bindings = complete_bindings.clone();
    remove_formal_alias_bindings(&mut bindings, &split.formal_binding_exclusions);

    out.push(ActionInstance {
        name,
        expr: wrapped,
        bindings,
        formal_bindings: split.formal_bindings.clone(),
        complete_bindings,
    });
}

fn formal_alias_exclusions_for_args(
    ctx: &EvalCtx,
    args: &[Spanned<Expr>],
    formal_bindings: &[(Arc<str>, Value)],
) -> Vec<(Arc<str>, Value)> {
    let local_bindings = ctx.get_local_bindings();
    let actual_names = args
        .iter()
        .filter_map(|arg| match &arg.node {
            Expr::Ident(name, _) => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut exclusions = Vec::new();
    for (arg, formal) in args.iter().zip(formal_bindings) {
        let Expr::Ident(actual_name, _) = &arg.node else {
            continue;
        };
        for (local_name, local_value) in &local_bindings {
            if local_value == &formal.1
                && !actual_names
                    .iter()
                    .any(|actual| local_name.as_ref() == *actual)
            {
                exclusions.push((Arc::clone(local_name), local_value.clone()));
            }
        }
        if local_bindings.iter().any(|(local_name, local_value)| {
            local_name.as_ref() == actual_name.as_str() && local_value == &formal.1
        }) {
            exclusions.push(formal.clone());
        }
    }
    exclusions
}

fn remove_formal_alias_bindings(
    bindings: &mut Vec<(Arc<str>, Value)>,
    exclusions: &[(Arc<str>, Value)],
) {
    for (exclude_name, exclude_value) in exclusions {
        if let Some(pos) = bindings
            .iter()
            .rposition(|(name, value)| name == exclude_name && value == exclude_value)
        {
            bindings.remove(pos);
        }
    }
}

fn merged_span(a: Span, b: Span) -> Span {
    if a.file == b.file {
        a.merge(b)
    } else {
        Span::dummy()
    }
}

fn if_branch_split_has_stable_action_names(split: &SplitCtx, actions: &[ActionInstance]) -> bool {
    let inherited_name = split.action_name.as_deref();
    !actions.is_empty()
        && actions.iter().all(|action| match action.name.as_deref() {
            Some(name) => inherited_name != Some(name),
            None => false,
        })
}

fn try_split_exists_all_bounds(
    ctx: &EvalCtx,
    bounds: &[BoundVar],
    idx: usize,
    body: &Spanned<Expr>,
    split: &SplitCtx,
    out: &mut Vec<ActionInstance>,
) -> EvalResult<bool> {
    if idx >= bounds.len() {
        split_action_instances_rec(ctx, body, split, out)?;
        return Ok(true);
    }

    if bounds[idx].pattern.is_some() {
        return Ok(false);
    }

    let domain_expr = match bounds[idx].domain.as_ref() {
        Some(e) => e,
        None => return Ok(false),
    };
    let domain_val = match try_eval_const_level(ctx, domain_expr) {
        Some(v) => v,
        None => return Ok(false),
    };
    // Part of #1828: use eval_iter_set for SetPred-aware iteration.
    // Part of #1886: discriminate "not enumerable" (defer) from fatal eval errors.
    // Part of #2987: use TLC-normalized ordering for BFS parity.
    let domain_elems: Vec<Value> =
        match eval_iter_set_tlc_normalized(ctx, &domain_val, Some(domain_expr.span)) {
            Ok(iter) => iter.collect(),
            Err(ref e)
                if classify_iter_error_for_speculative_path(e) == IterDomainAction::Defer =>
            {
                return Ok(false);
            }
            Err(e) => return Err(e),
        };

    let name: Arc<str> = Arc::from(bounds[idx].name.node.as_str());
    for v in domain_elems {
        let ctx2 = ctx.bind_local(Arc::clone(&name), v);
        if !try_split_exists_all_bounds(&ctx2, bounds, idx + 1, body, split, out)? {
            return Ok(false);
        }
    }

    Ok(true)
}

fn module_target_simple_name(target: &ModuleTarget) -> Option<&str> {
    match target {
        ModuleTarget::Named(name) => Some(name.as_str()),
        ModuleTarget::Parameterized(name, _) => Some(name.as_str()),
        ModuleTarget::Chained(_) => None,
    }
}

#[derive(Debug, Clone)]
struct ResolvedInstanceInfo {
    module_name: String,
    substitutions: Vec<Substitution>,
}

fn resolve_instance_info(ctx: &EvalCtx, instance_name: &str) -> Option<ResolvedInstanceInfo> {
    if let Some(info) = ctx.get_instance(instance_name) {
        let effective =
            ctx.compute_effective_instance_substitutions(&info.module_name, &info.substitutions);
        return Some(ResolvedInstanceInfo {
            module_name: info.module_name.clone(),
            substitutions: compose_substitutions(&effective, ctx.instance_substitutions()),
        });
    }
    let def = ctx.get_op(instance_name)?;
    let Expr::InstanceExpr(module_name, substitutions) = &def.body.node else {
        return None;
    };
    let effective = ctx.compute_effective_instance_substitutions(module_name, substitutions);
    Some(ResolvedInstanceInfo {
        module_name: module_name.clone(),
        substitutions: compose_substitutions(&effective, ctx.instance_substitutions()),
    })
}

fn merged_instance_local_ops(ctx: &EvalCtx, module_name: &str) -> Option<OpEnv> {
    let mut merged = ctx.instance_ops().get(module_name)?.clone();
    if let Some(parent) = ctx.local_ops().as_deref() {
        for (name, def) in parent.iter() {
            merged
                .entry(name.clone())
                .or_insert_with(|| Arc::clone(def));
        }
    }
    Some(merged)
}

fn scoped_action_name(split: &SplitCtx, name: &str) -> String {
    split
        .instance_prefix
        .as_ref()
        .map(|prefix| format!("{prefix}!{name}"))
        .unwrap_or_else(|| name.to_string())
}
