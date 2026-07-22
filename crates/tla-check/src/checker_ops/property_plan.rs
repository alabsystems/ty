// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared PROPERTY planning for BFS promotion and safety-temporal fast paths.

use crate::check::{expr_contains, ScanDecision};
use crate::eval::{apply_substitutions, compose_substitutions, EvalCtx};
use crate::liveness::{AstToLive, ExprLevel};
use rustc_hash::FxHashMap;
use tla_core::ast::{Expr, ModuleTarget, OperatorDef, Substitution};
use tla_core::{ExprFold, Spanned};

use super::instance_qualify::{qualify_instance_ops, qualify_instance_ops_with_bound};
use super::{contains_module_ref, contains_temporal_standalone, flatten_and_terms_standalone};

/// Maximum INSTANCE-resolution recursion depth when flattening property-term
/// `ModuleRef` nodes. A small bound guards against pathological / cyclic
/// instance graphs; in practice refinement chains are shallow (`B!Spec` →
/// `B!Next` → `B!b0`/`B!b1`). Exceeding the bound leaves the residual
/// `ModuleRef` in place (fail-closed → the term stays interpreter-only).
const MODULE_REF_FLATTEN_MAX_DEPTH: usize = 64;

/// Shared semantic buckets for PROPERTY terms.
///
/// The bucket shapes intentionally mirror the existing execution lanes so
/// adapters can preserve current behavior while sharing one semantic plan.
pub(crate) enum PlannedPropertyTerm {
    Init(Spanned<Expr>),
    StateCompiled(Spanned<Expr>),
    StateEval(Spanned<Expr>),
    ActionCompiled(Spanned<Expr>),
    ActionEval(Spanned<Expr>),
    Liveness(Spanned<Expr>),
}

/// Shared PROPERTY plan consumed by BFS preparation and safety-temporal logic.
pub(crate) struct PlannedProperty {
    pub(crate) property: String,
    pub(crate) terms: Vec<PlannedPropertyTerm>,
}

pub(crate) fn wrap_with_let_defs(
    defs: &Option<Vec<OperatorDef>>,
    expr: Spanned<Expr>,
) -> Spanned<Expr> {
    match defs {
        Some(defs) => Spanned::new(Expr::Let(defs.clone(), Box::new(expr.clone())), expr.span),
        None => expr,
    }
}

pub(crate) fn contains_enabled_standalone(expr: &Expr) -> bool {
    expr_contains(expr, &|e| match e {
        Expr::Enabled(_) => ScanDecision::Found,
        _ => ScanDecision::Continue,
    })
}

fn is_real_action_subscript(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    ctx.is_action_subscript_span(expr.span)
}

pub(crate) fn plan_property_terms(
    ctx: &EvalCtx,
    op_defs: &FxHashMap<String, OperatorDef>,
    prop_name: &str,
) -> Option<PlannedProperty> {
    let def = op_defs.get(prop_name)?;

    let (let_defs, prop_body) = match &def.body.node {
        Expr::Let(defs, inner) => (Some(defs.clone()), (**inner).clone()),
        Expr::ModuleRef(target, op_name, args) => {
            (None, resolve_module_ref_body(ctx, target, op_name, args)?)
        }
        _ => (None, def.body.clone()),
    };

    // INSTANCE name-resolution for PROPERTY action terms (analog of the
    // next-state-relation fix in run_prepare.rs::override_instance_action_callees_from_split).
    //
    // Refinement properties reach their action predicate through an INSTANCE
    // layer, e.g. `BSpec == B!Spec` expanding to `B!Init /\ [][B!Next]_B!vars`.
    // The `B!Next` / `B!vars` references lower to `ModuleRef` nodes, which the
    // downstream classification tags as `ActionEval` (interpreter-only) — so the
    // native fused implied-action path can never engage. Structurally flatten
    // those `ModuleRef` nodes down to their substitution-applied, qualified
    // underlying action expressions (the SAME resolution the action splitter
    // performs for next-state actions via `apply_substitutions` +
    // `qualify_instance_ops`). When the result contains no residual `ModuleRef`,
    // the term classifies as `ActionCompiled` like any `A \/ UNCHANGED v`
    // predicate. Resolution is fail-closed: any node we cannot resolve is left
    // intact, so the term simply remains interpreter-only (never unsound).
    let prop_body = flatten_property_module_refs(ctx, prop_body);

    let mut split_terms = Vec::new();
    flatten_and_terms_standalone(&prop_body, &mut split_terms);

    let converter = AstToLive::new();
    let mut planned_terms = Vec::new();

    for term in split_terms {
        match &term.node {
            Expr::Always(inner) => {
                let body = wrap_with_let_defs(&let_defs, (**inner).clone());
                let inner_level = converter.get_level_with_ctx(ctx, &inner.node);
                let real_action_subscript = is_real_action_subscript(ctx, inner);

                // Real `[A]_v` / `<<A>>_v` syntax remains an action property even when
                // the lowered `A \/ UNCHANGED v` body is semantically state-level
                // (for example `[][decision = none]_<<decision>>` in FastPaxos).
                if real_action_subscript && !matches!(inner_level, ExprLevel::Temporal) {
                    if contains_module_ref(&inner.node) || contains_enabled_standalone(&inner.node)
                    {
                        planned_terms.push(PlannedPropertyTerm::ActionEval(body));
                    } else {
                        planned_terms.push(PlannedPropertyTerm::ActionCompiled(body));
                    }
                    continue;
                }

                if contains_temporal_standalone(&inner.node) {
                    if matches!(inner_level, ExprLevel::Constant | ExprLevel::State)
                        && contains_enabled_standalone(&inner.node)
                    {
                        planned_terms.push(PlannedPropertyTerm::StateEval(body));
                    } else {
                        planned_terms.push(PlannedPropertyTerm::Liveness(wrap_with_let_defs(
                            &let_defs,
                            term.clone(),
                        )));
                    }
                    continue;
                }

                match inner_level {
                    ExprLevel::Constant | ExprLevel::State => {
                        planned_terms.push(PlannedPropertyTerm::StateCompiled(body));
                    }
                    ExprLevel::Action => {
                        if !real_action_subscript {
                            planned_terms.push(PlannedPropertyTerm::Liveness(wrap_with_let_defs(
                                &let_defs,
                                term.clone(),
                            )));
                        } else if contains_module_ref(&inner.node) {
                            planned_terms.push(PlannedPropertyTerm::ActionEval(body));
                        } else {
                            planned_terms.push(PlannedPropertyTerm::ActionCompiled(body));
                        }
                    }
                    ExprLevel::Temporal => {
                        planned_terms.push(PlannedPropertyTerm::Liveness(wrap_with_let_defs(
                            &let_defs,
                            term.clone(),
                        )));
                    }
                }
            }
            _ => {
                if contains_temporal_standalone(&term.node) {
                    planned_terms.push(PlannedPropertyTerm::Liveness(wrap_with_let_defs(
                        &let_defs,
                        term.clone(),
                    )));
                    continue;
                }

                match converter.get_level_with_ctx(ctx, &term.node) {
                    ExprLevel::Constant | ExprLevel::State => {
                        planned_terms.push(PlannedPropertyTerm::Init(wrap_with_let_defs(
                            &let_defs,
                            term.clone(),
                        )));
                    }
                    ExprLevel::Action | ExprLevel::Temporal => {
                        planned_terms.push(PlannedPropertyTerm::Liveness(wrap_with_let_defs(
                            &let_defs,
                            term.clone(),
                        )));
                    }
                }
            }
        }
    }

    Some(PlannedProperty {
        property: prop_name.to_string(),
        terms: planned_terms,
    })
}

/// Resolve a ModuleRef property body for classification.
///
/// Returns the operator body with module-local references qualified as
/// ModuleRef nodes. Substitutions are not applied here; the eval path handles
/// them when it resolves the ModuleRef nodes during evaluation.
fn resolve_module_ref_body(
    ctx: &EvalCtx,
    target: &ModuleTarget,
    op_name: &str,
    args: &[Spanned<Expr>],
) -> Option<Spanned<Expr>> {
    if !args.is_empty() {
        return None;
    }
    let ModuleTarget::Named(instance_name) = target else {
        return None;
    };
    let info = ctx.get_instance(instance_name)?;
    let op_def = ctx.get_instance_op(&info.module_name, op_name)?;
    if !op_def.params.is_empty() {
        return None;
    }
    let qualified = qualify_instance_ops(ctx, target, &info.module_name, op_def.body.clone());
    Some(qualified)
}

/// Structurally flatten INSTANCE-namespaced `ModuleRef` nodes in a PROPERTY
/// expression into their substitution-applied, qualified underlying bodies.
///
/// This mirrors the action splitter's `Expr::ModuleRef` expansion
/// (`action_instance::split`): for `M!Op(args)` it looks up the instance's
/// substitutions and the operator body in the instanced module, applies the
/// INSTANCE `WITH` substitutions (`apply_substitutions`), substitutes operator
/// formals with the actual argument expressions, and re-qualifies the remaining
/// module-local references (`qualify_instance_ops`) so nested module-local ops
/// become `ModuleRef` nodes that are then recursively resolved.
///
/// Unlike the evaluator's lazy `ModuleRef` resolution, this produces a fully
/// inlined AST with no `ModuleRef` indirection, which is what the native
/// action-bytecode compile path requires (it rejects unresolved identifiers).
/// The argument expressions are kept symbolic (substituted as expressions, not
/// evaluated to values) so quantifier-bound variables such as the `p` in
/// `\E p \in S : b0(p)` survive intact.
///
/// Fail-closed: any `ModuleRef` whose instance / operator / arity cannot be
/// resolved is returned unchanged, so the enclosing term keeps its residual
/// `ModuleRef` and stays interpreter-only. Resolution is bounded by
/// `MODULE_REF_FLATTEN_MAX_DEPTH` to guard against cyclic instance graphs.
pub(crate) fn flatten_property_module_refs(ctx: &EvalCtx, expr: Spanned<Expr>) -> Spanned<Expr> {
    let mut folder = FlattenModuleRefsFolder { ctx, depth: 0 };
    folder.fold_expr(expr)
}

struct FlattenModuleRefsFolder<'a> {
    ctx: &'a EvalCtx,
    depth: usize,
}

impl FlattenModuleRefsFolder<'_> {
    /// Resolve a single `ModuleRef(target, op_name, args)` to its
    /// substitution-applied, qualified body. Returns `None` (fail-closed) when
    /// the reference cannot be structurally resolved.
    fn resolve_module_ref(
        &self,
        target: &ModuleTarget,
        op_name: &str,
        args: &[Spanned<Expr>],
    ) -> Option<Spanned<Expr>> {
        // Only named instances are handled here (e.g. `B!Next`). Parameterized
        // and chained targets are left to the lazy eval path (fail-closed).
        let ModuleTarget::Named(instance_name) = target else {
            return None;
        };

        // Resolve instance metadata. Compose the instance substitutions through
        // any outer instance scope so nested INSTANCE chains remain correct,
        // matching the evaluator's `resolve_named_or_parameterized_module_ref_body`.
        let info = self.ctx.get_instance(instance_name)?;
        let module_name = info.module_name.clone();
        let effective_subs: Vec<Substitution> =
            compose_substitutions(&info.substitutions, self.ctx.instance_substitutions());

        let resolved_op_name = self.ctx.resolve_op_name(op_name);
        let op_def = self.ctx.get_instance_op(&module_name, resolved_op_name)?;
        if op_def.params.len() != args.len() {
            return None;
        }

        // Qualify the instanced operator's OWN module-local references FIRST —
        // BEFORE splicing in the INSTANCE WITH values (`effective_subs`) and the
        // actual arguments (`param_subs`). Both of those are expressions of the
        // INSTANTIATING (outer) module: their identifiers must resolve THERE,
        // never in the instanced module. Qualifying AFTER substitution (the old
        // order) wrongly rewrites an outer-module identifier that shares a name
        // with an instanced-module operator into a `ModuleRef` — observed as
        // LockHS's `pc_translation(p, pc[p], s)` in a WITH-RHS colliding with the
        // instanced module's 1-arg `pc_translation`, yielding "Arity mismatch:
        // P!pc_translation expects 1 arguments, got 3"; with matching arities it
        // would SILENTLY bind the wrong operator (a latent soundness defect).
        //
        // Soundness: substitution only replaces the instanced module's own
        // WITH-parameters and the operator's formal params — it never introduces
        // a NEW reference to an instanced-module operator — so the set of
        // instanced-module operator refs that qualification must rewrite is
        // exactly the set already present in the pristine `op_def.body`.
        // Qualifying it there is therefore identical to the old order for every
        // spec EXCEPT those where a substituted-in identifier collides by name
        // with an instanced-module operator, which the old order mis-qualified.
        // The formal-parameter names are excluded from qualification (they are
        // outer-scope placeholders the param substitution replaces), restoring
        // the immunity the old substitute-then-qualify order got for free.
        let param_names: Vec<String> = op_def.params.iter().map(|p| p.name.node.clone()).collect();
        let qualified = qualify_instance_ops_with_bound(
            self.ctx,
            target,
            &module_name,
            op_def.body.clone(),
            &param_names,
        );
        let mut body = if effective_subs.is_empty() {
            qualified
        } else {
            apply_substitutions(&qualified, &effective_subs)
        };
        if !op_def.params.is_empty() {
            let param_subs: Vec<Substitution> = op_def
                .params
                .iter()
                .zip(args.iter())
                .map(|(param, arg)| Substitution {
                    from: param.name.clone(),
                    to: arg.clone(),
                })
                .collect();
            body = apply_substitutions(&body, &param_subs);
        }
        Some(body)
    }
}

impl ExprFold for FlattenModuleRefsFolder<'_> {
    fn fold_expr(&mut self, expr: Spanned<Expr>) -> Spanned<Expr> {
        if self.depth >= MODULE_REF_FLATTEN_MAX_DEPTH {
            // Bound exceeded: stop resolving and leave the residual subtree as
            // is (fail-closed — any remaining ModuleRef keeps the term eval-only).
            return expr;
        }
        let span = expr.span;
        match expr.node {
            Expr::ModuleRef(target, op_name, args) => {
                // Resolve nested arguments first so symbolic argument
                // expressions are themselves fully flattened.
                let folded_args = self.fold_vec(args);
                match self.resolve_module_ref(&target, &op_name, &folded_args) {
                    Some(resolved) => {
                        // Recurse into the resolved body to flatten nested
                        // module-local `ModuleRef` nodes introduced by
                        // `qualify_instance_ops`.
                        self.depth += 1;
                        let out = self.fold_expr(resolved);
                        self.depth -= 1;
                        out
                    }
                    None => {
                        // Fail-closed: keep the (arg-folded) ModuleRef intact.
                        Spanned {
                            node: Expr::ModuleRef(target, op_name, folded_args),
                            span,
                        }
                    }
                }
            }
            node => {
                let node = self.fold_expr_inner(node);
                Spanned { node, span }
            }
        }
    }
}
