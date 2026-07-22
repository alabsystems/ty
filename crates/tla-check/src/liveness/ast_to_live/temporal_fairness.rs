// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fairness and INSTANCE-target helpers split from `temporal_convert.rs`.
//!
//! Part of #2853.

use super::super::live_expr::LiveExpr;
use super::core::{ActionPredHint, AstToLive};
use crate::eval::{
    apply_substitutions, compose_substitutions, eval_entry, try_eval_const_level, EvalCtx,
};
use crate::Value;
use std::sync::Arc;
use tla_core::ast::{Expr, ModuleTarget, Substitution};
use tla_core::span::Span;
use tla_core::Spanned;

/// Part of #3100: Extract a structured action hint from a fairness action expression.
///
/// For `Ident("Next")` -> hint with name="Next", no bindings.
/// For `Apply(Ident("Op"), [1])` -> hint with name="Op", bindings=[("a", 1)]
///   if the operator has formal params and all actuals are const-level.
/// Falls back to name-only hint when actuals can't be evaluated at conversion time.
fn name_only_hint(tag: u32, name: Arc<str>, split_action_fast_path_safe: bool) -> ActionPredHint {
    ActionPredHint {
        tag,
        name,
        split_action_fast_path_safe,
        actual_arg_bindings: Vec::new(),
        wildcard_arg_domains: Vec::new(),
    }
}

fn extract_apply_action_hint(
    ctx: &EvalCtx,
    name: &str,
    args: &[Spanned<Expr>],
    tag: u32,
) -> ActionPredHint {
    let mut bindings = Vec::new();
    if let Some(def) = ctx.get_op(name) {
        if def.params.len() == args.len() {
            for (param, arg) in def.params.iter().zip(args.iter()) {
                // CONST-level evaluation only (#3208 ENABLED-provenance
                // hardening): a STATE-dependent argument expression must never
                // produce a binding value — the value would be a snapshot of
                // whatever state happened to be bound at conversion time,
                // and downstream consumers (split-action matching, the
                // TRUE-only ENABLED provenance registration) treat these
                // values as the run-constant identity of the fairness action.
                // `try_eval_const_level` resolves constants and quantifier-
                // expansion bindings but refuses state variables; failure
                // falls back to a name-only hint (which every binding-aware
                // consumer rejects — fail closed).
                match try_eval_const_level(ctx, arg) {
                    Some(val) => {
                        bindings.push((Arc::from(param.name.node.as_str()), val));
                    }
                    None => return name_only_hint(tag, Arc::from(name), true),
                }
            }
        }
    }
    ActionPredHint {
        tag,
        name: Arc::from(name),
        split_action_fast_path_safe: true,
        actual_arg_bindings: bindings,
        wildcard_arg_domains: Vec::new(),
    }
}

/// #3208 wildcard frames: extract a hint from an existentially-quantified
/// fairness action `\E x1 \in D1, .., xk \in Dk : Op(args)`.
///
/// Eligibility (all fail-closed):
/// * every bound variable has a pattern-free, CONST-level, enumerable domain
///   whose expression mentions NO bound variable (rules out dependent domains
///   AND outer-scope shadowing of a bound name);
/// * `Op` resolves with matching arity;
/// * every argument is either EXACTLY one of the bound variables (by `Ident`)
///   or a const-level expression mentioning NO bound variable (again ruling
///   out shadowed evaluation against an outer binding);
/// * every bound variable occurs in exactly one argument position.
///
/// The hint then identifies the leaf action as
/// `∃ (wildcards ∈ domains): Op(consts, wildcards)`: a BFS emission through
/// `Op` matching the const positions whose wildcard-position values lie in the
/// registered domains witnesses the existential — the domain-membership check
/// keeps the witness sound even when the `Next` relation quantifies the same
/// operator over a WIDER domain than the fairness leaf.
fn extract_exists_apply_hint(
    ctx: &EvalCtx,
    bounds: &[tla_core::ast::BoundVar],
    body: &Spanned<Expr>,
    tag: u32,
) -> Option<ActionPredHint> {
    let Expr::Apply(op_expr, args) = &body.node else {
        return None;
    };
    let Expr::Ident(name, _) = &op_expr.node else {
        return None;
    };
    let def = ctx.get_op(name)?;
    if def.params.len() != args.len() {
        return None;
    }
    let bound_names: Vec<&str> = bounds.iter().map(|b| b.name.node.as_str()).collect();

    // Enumerate every quantifier domain (const-level, bound-var-free).
    const MAX_WILDCARD_DOMAIN: usize = 256;
    let mut domains: Vec<(&str, Vec<Value>)> = Vec::with_capacity(bounds.len());
    for b in bounds {
        if b.pattern.is_some() {
            return None;
        }
        let domain_expr = b.domain.as_ref()?;
        if bound_names
            .iter()
            .any(|n| tla_core::expr_mentions_name_v(&domain_expr.node, n))
        {
            return None;
        }
        let domain_val = try_eval_const_level(ctx, domain_expr)?;
        let elems: Vec<Value> =
            match crate::eval::eval_iter_set_tlc_normalized(ctx, &domain_val, None) {
                Ok(iter) => iter.collect(),
                Err(_) => return None,
            };
        if elems.is_empty() || elems.len() > MAX_WILDCARD_DOMAIN {
            return None;
        }
        domains.push((b.name.node.as_str(), elems));
    }

    let mut const_bindings: Vec<(Arc<str>, Value)> = Vec::new();
    let mut wildcards: Vec<(Arc<str>, Vec<Value>)> = Vec::new();
    let mut used = vec![false; bounds.len()];
    for (param, arg) in def.params.iter().zip(args.iter()) {
        if let Expr::Ident(arg_name, _) = &arg.node {
            if let Some(pos) = bound_names.iter().position(|n| *n == arg_name.as_str()) {
                if used[pos] {
                    return None; // bound var used twice: positions not independent
                }
                used[pos] = true;
                wildcards.push((Arc::from(param.name.node.as_str()), domains[pos].1.clone()));
                continue;
            }
        }
        // Const argument: must not mention any bound variable (a non-Ident
        // expression over a bound var, or a shadowed outer name, would
        // otherwise evaluate against the WRONG scope).
        if bound_names
            .iter()
            .any(|n| tla_core::expr_mentions_name_v(&arg.node, n))
        {
            return None;
        }
        let val = try_eval_const_level(ctx, arg)?;
        const_bindings.push((Arc::from(param.name.node.as_str()), val));
    }
    if !used.iter().all(|u| *u) {
        // A bound variable not consumed by any argument would leave its
        // existential undecided by an emission witness: fail closed.
        return None;
    }

    Some(ActionPredHint {
        tag,
        name: Arc::from(name.as_str()),
        split_action_fast_path_safe: true,
        actual_arg_bindings: const_bindings,
        wildcard_arg_domains: wildcards,
    })
}

fn collect_parameterized_target_bindings(
    ctx: &EvalCtx,
    name: &str,
    target_actuals: &[Spanned<Expr>],
) -> Option<Vec<(Arc<str>, crate::Value)>> {
    let Some(def) = ctx.get_op(name) else {
        return Some(Vec::new());
    };
    if def.params.len() != target_actuals.len() {
        return Some(Vec::new());
    }
    let mut bindings = Vec::new();
    for (param, arg) in def.params.iter().zip(target_actuals.iter()) {
        match eval_entry(ctx, arg) {
            Ok(val) => bindings.push((Arc::from(param.name.node.as_str()), val)),
            Err(_) => return None,
        }
    }
    Some(bindings)
}

fn append_instance_op_bindings(
    ctx: &EvalCtx,
    module_name: &str,
    op_name: &str,
    args: &[Spanned<Expr>],
    bindings: &mut Vec<(Arc<str>, crate::Value)>,
) -> bool {
    let Some(def) = ctx.get_instance_op(module_name, op_name) else {
        return true;
    };
    if def.params.len() != args.len() {
        return true;
    }
    for (param, arg) in def.params.iter().zip(args.iter()) {
        match eval_entry(ctx, arg) {
            Ok(val) => bindings.push((Arc::from(param.name.node.as_str()), val)),
            Err(_) => return false,
        }
    }
    true
}

fn extract_module_ref_action_hint(
    ctx: &EvalCtx,
    target: &ModuleTarget,
    op_name: &str,
    args: &[Spanned<Expr>],
    tag: u32,
) -> Option<ActionPredHint> {
    let target_name = match target {
        ModuleTarget::Named(name) | ModuleTarget::Parameterized(name, _) => name.as_str(),
        ModuleTarget::Chained(_) => return None,
    };
    let qualified_name: Arc<str> = Arc::from(format!("{target_name}!{op_name}"));
    let mut bindings = match target {
        ModuleTarget::Parameterized(name, target_actuals) => {
            match collect_parameterized_target_bindings(ctx, name, target_actuals) {
                Some(bindings) => bindings,
                None => return Some(name_only_hint(tag, Arc::clone(&qualified_name), false)),
            }
        }
        ModuleTarget::Named(_) => Vec::new(),
        ModuleTarget::Chained(_) => return None,
    };

    if !args.is_empty() {
        if let Some(module_name) = resolve_instance_module_name_from_target(ctx, target) {
            if !append_instance_op_bindings(ctx, &module_name, op_name, args, &mut bindings) {
                return Some(ActionPredHint {
                    tag,
                    name: qualified_name,
                    split_action_fast_path_safe: false,
                    actual_arg_bindings: bindings,
                    wildcard_arg_domains: Vec::new(),
                });
            }
        }
    }

    Some(ActionPredHint {
        tag,
        name: qualified_name,
        split_action_fast_path_safe: false,
        actual_arg_bindings: bindings,
        wildcard_arg_domains: Vec::new(),
    })
}

fn extract_action_hint(ctx: &EvalCtx, expr: &Expr, tag: u32) -> Option<ActionPredHint> {
    match expr {
        Expr::Ident(name, _) => Some(name_only_hint(tag, Arc::from(name.as_str()), true)),
        Expr::Apply(op, args) => match &op.node {
            Expr::Ident(name, _) => Some(extract_apply_action_hint(ctx, name, args, tag)),
            _ => None,
        },
        Expr::ModuleRef(target, op_name, args) => {
            extract_module_ref_action_hint(ctx, target, op_name, args, tag)
        }
        // #3208 wildcard frames: `\E x \in D, .. : Op(.., x, ..)`.
        Expr::Exists(bounds, body) => extract_exists_apply_hint(ctx, bounds, body, tag),
        _ => None,
    }
}

/// Mirrors `resolve_instance_module_name` in `action_instance.rs`, but works with
/// the `ModuleTarget` enum directly.
fn resolve_instance_module_name_from_target(
    ctx: &EvalCtx,
    target: &ModuleTarget,
) -> Option<String> {
    match target {
        ModuleTarget::Named(name) => {
            if let Some(info) = ctx.get_instance(name) {
                return Some(info.module_name.clone());
            }
            let def = ctx.get_op(name)?;
            let Expr::InstanceExpr(module_name, _) = &def.body.node else {
                return None;
            };
            Some(module_name.clone())
        }
        ModuleTarget::Parameterized(name, _) => {
            let def = ctx.get_op(name)?;
            let Expr::InstanceExpr(module_name, _) = &def.body.node else {
                return None;
            };
            Some(module_name.clone())
        }
        ModuleTarget::Chained(_) => None,
    }
}

/// Parse the subscript portion of a fused fairness operator like `WF_vars` or
/// `SF_<<x, y>>` into a real AST expression.
pub(super) fn parse_subscript_from_fused_name(subscript_name: &str, span: Span) -> Spanned<Expr> {
    if subscript_name.starts_with("<<") {
        let inner = subscript_name
            .trim_start_matches("<<")
            .trim_end_matches(">>");
        let tuple_elems: Vec<_> = inner
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| Spanned::new(Expr::Ident(name.into(), tla_core::NameId::INVALID), span))
            .collect();
        Spanned::new(Expr::Tuple(tuple_elems), span)
    } else {
        Spanned::new(
            Expr::Ident(subscript_name.into(), tla_core::NameId::INVALID),
            span,
        )
    }
}

fn resolve_instance_info(
    ctx: &EvalCtx,
    instance_name: &str,
) -> Option<(String, Vec<Substitution>)> {
    if let Some(info) = ctx.get_instance(instance_name) {
        let effective_subs =
            ctx.compute_effective_instance_substitutions(&info.module_name, &info.substitutions);
        return Some((
            info.module_name.clone(),
            compose_substitutions(&effective_subs, ctx.instance_substitutions()),
        ));
    }
    if let Some(def) = ctx.get_op(instance_name) {
        if !def.params.is_empty() {
            return None;
        }
        if let Expr::InstanceExpr(module_name, substitutions) = &def.body.node {
            let effective_subs =
                ctx.compute_effective_instance_substitutions(module_name, substitutions);
            return Some((
                module_name.clone(),
                compose_substitutions(&effective_subs, ctx.instance_substitutions()),
            ));
        }
    }
    None
}

pub(super) fn resolve_module_target_ctx(ctx: &EvalCtx, target: &ModuleTarget) -> Option<EvalCtx> {
    match target {
        ModuleTarget::Named(name) => {
            let (module_name, instance_subs) = resolve_instance_info(ctx, name)?;
            // Part of #1433: empty default is correct - modules without registered
            // instance operators simply have no local ops in scope. Not error-swallowing.
            let instance_ops = ctx
                .instance_ops()
                .get(&module_name)
                .cloned()
                .unwrap_or_default();
            Some(ctx.with_instance_scope(instance_ops, instance_subs))
        }
        ModuleTarget::Parameterized(name, params) => {
            let def = ctx.get_op(name)?;
            let Expr::InstanceExpr(module_name, instance_subs) = &def.body.node else {
                return None;
            };
            if def.params.len() != params.len() {
                return None;
            }

            // Bind instance operator parameters into the WITH substitutions.
            let param_subs: Vec<Substitution> = def
                .params
                .iter()
                .zip(params.iter())
                .map(|(p, arg)| Substitution {
                    from: p.name.clone(),
                    to: arg.clone(),
                })
                .collect();

            let instantiated_subs: Vec<Substitution> = instance_subs
                .iter()
                .map(|s| Substitution {
                    from: s.from.clone(),
                    to: apply_substitutions(&s.to, &param_subs),
                })
                .collect();

            let effective_subs = compose_substitutions(
                &ctx.compute_effective_instance_substitutions(module_name, &instantiated_subs),
                ctx.instance_substitutions(),
            );
            // Part of #1433: empty default is correct - modules without registered
            // instance operators simply have no local ops in scope. Not error-swallowing.
            let instance_ops = ctx
                .instance_ops()
                .get(module_name)
                .cloned()
                .unwrap_or_default();
            Some(ctx.with_instance_scope(instance_ops, effective_subs))
        }
        ModuleTarget::Chained(base_expr) => resolve_chained_target_ctx(ctx, base_expr),
    }
}

fn resolve_chained_target_ctx(ctx: &EvalCtx, base_expr: &Spanned<Expr>) -> Option<EvalCtx> {
    let Expr::ModuleRef(base_target, inst_name, inst_args) = &base_expr.node else {
        return None;
    };
    if !inst_args.is_empty() {
        return None;
    }

    let base_ctx = resolve_module_target_ctx(ctx, base_target)?;
    let inst_def = base_ctx.get_op(inst_name)?;
    let Expr::InstanceExpr(module_name, inst_subs) = &inst_def.body.node else {
        return None;
    };

    let effective_subs = compose_substitutions(
        &base_ctx.compute_effective_instance_substitutions(module_name, inst_subs),
        base_ctx.instance_substitutions(),
    );
    // Chained liveness references must preserve the intermediate module scope.
    // Example: EWD998Chan!EWD998!TD!Live enters TD, but TD predicates refer to
    // operators declared in the intermediate EWD998 module. Keeping those
    // parent operators in local_ops prevents mixed substitutions such as an
    // outer model-value Node being used with an int-keyed substituted inbox.
    let mut instance_ops = ctx
        .instance_ops()
        .get(module_name)
        .cloned()
        .unwrap_or_default();
    if let Some(parent_ops) = base_ctx.local_ops().as_ref() {
        for (name, def) in parent_ops.iter() {
            if !instance_ops.contains_key(name.as_str()) {
                instance_ops.insert(name.clone(), def.clone());
            }
        }
    }

    Some(base_ctx.with_instance_scope(instance_ops, effective_subs))
}

impl AstToLive {
    /// Resolve a fairness subscript expression for WF_/SF_ formulas.
    ///
    /// For simple identifier subscripts (e.g., `vars` in `WF_vars(A)`), preserves
    /// the operator boundary by using `qualify_predicate_expr` instead of
    /// `resolve_action_expr`. This prevents the operator body from being inlined
    /// into the quantifier scope, which would expose its internal references to
    /// quantifier bindings and cause accidental variable capture (#2116).
    ///
    /// For compound subscripts (tuples, etc.), delegates to `resolve_action_expr`
    /// which correctly inlines and resolves operator references.
    pub(super) fn resolve_fairness_subscript(
        &self,
        ctx: &EvalCtx,
        subscript: &Spanned<Expr>,
    ) -> Spanned<Expr> {
        let resolved = match &subscript.node {
            Expr::Ident(_, _) => {
                if self.current_target().is_some() {
                    self.reify_current_target_leaf_expr(ctx, subscript)
                } else {
                    subscript.clone()
                }
            }
            _ => self.resolve_action_expr(ctx, subscript),
        };
        self.wrap_in_let_defs(resolved)
    }

    pub(super) fn build_fairness_expansion(
        &self,
        ctx: &EvalCtx,
        action: &Spanned<Expr>,
        subscript: &Spanned<Expr>,
        is_weak: bool,
    ) -> LiveExpr {
        // Resolve any operator references in the action expression so it can be
        // evaluated later without the original context's operator bindings.
        let resolved_action = self.wrap_in_let_defs(self.resolve_action_expr(ctx, action));

        // Resolve the subscript expression - this is the tuple we check for changes.
        // For WF_vars(A), vars is typically <<v1, v2, ...>>.
        // Part of #2116: Use resolve_fairness_subscript to preserve operator
        // boundaries for simple identifier subscripts, preventing accidental
        // capture of quantifier bindings in the inlined body.
        let resolved_subscript = self.resolve_fairness_subscript(ctx, subscript);
        let subscript_arc = Some(Arc::new(resolved_subscript));

        let not_enabled = LiveExpr::not(LiveExpr::enabled_with_bindings(
            Arc::new(resolved_action.clone()),
            true,
            subscript_arc.clone(),
            self.alloc_tag(),
            self.current_quantifier_bindings(),
        ));

        // Part of #3100, fix #3183: Capture the tag and record structured action
        // hint (with const-level actual-arg bindings) for provenance matching.
        //
        // We must extract the hint from the ORIGINAL action expression when not
        // inside an instance scope, because resolve_action_expr inlines operator
        // bodies (e.g., Op(1) -> x'=1), destroying the name+binding structure
        // that extract_action_hint needs (Apply(Ident("Op"), [1]) -> name="Op",
        // bindings=[("a",1)]). When inside an instance scope, the resolved form
        // is needed for qualification (e.g., "Schedule" -> "Sched!Schedule").
        let ap_tag = self.alloc_tag();
        let hint = if self.current_target().is_some() {
            // Instance scope: use resolved/qualified expression for correct naming
            extract_action_hint(ctx, &resolved_action.node, ap_tag)
        } else {
            // No instance scope: prefer original to preserve name+bindings,
            // fall back to resolved if original doesn't yield a hint
            extract_action_hint(ctx, &action.node, ap_tag)
                .or_else(|| extract_action_hint(ctx, &resolved_action.node, ap_tag))
        };
        if let Some(hint) = hint {
            self.record_action_hint(hint);
        }
        // Part of #3100: StateChanged before ActionPred - matches TLC's LNAction
        // which evaluates the subscript first (cheap) and short-circuits the body
        // evaluation when v' == v (<<A>>_v = false if subscript unchanged).
        // For property-level plans where And is evaluated as a combined leaf,
        // this puts the cheap StateChanged check first in the And short-circuit.
        let action_occurs = LiveExpr::and(vec![
            LiveExpr::state_changed_with_bindings(
                subscript_arc,
                self.alloc_tag(),
                self.current_quantifier_bindings(),
            ),
            LiveExpr::action_pred_with_bindings(
                Arc::new(resolved_action),
                ap_tag,
                self.current_quantifier_bindings(),
            ),
        ]);

        if is_weak {
            // Weak Fairness: WF_e(A) expands to []<>(~ENABLED<A>_e \/ <A>_e)
            // i.e., "infinitely often, either A is not enabled or A happens"
            let disj = LiveExpr::or(vec![not_enabled, action_occurs]);
            LiveExpr::always(LiveExpr::eventually(disj))
        } else {
            // Strong Fairness: SF_e(A) expands to <>[]~ENABLED<A>_e \/ []<><A>_e
            // i.e., "eventually always not enabled, or infinitely often happens"
            let eventually_always_disabled = LiveExpr::eventually(LiveExpr::always(not_enabled));
            let infinitely_often = LiveExpr::always(LiveExpr::eventually(action_occurs));
            LiveExpr::or(vec![eventually_always_disabled, infinitely_often])
        }
    }
}
