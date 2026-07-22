// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CHC-specific operator expansion for PDR translation.
//!
//! Unlike the standard `expand_operators`, this version DOES inline operators
//! containing primed variables when `allow_primed` is true. This is required
//! for Next relation translation where `x' = x + 1` patterns are common.
//!
//! ## SOUNDNESS: capture-avoidance (2026-07-05 AY-lane false-safe)
//!
//! A zero-arity operator (or config CONSTANT) reference is inlined ONLY when its name is NOT
//! currently bound by an enclosing binder (`FuncDef`/`∀`/`∃`/`CHOOSE`/`SetFilter`/`SetBuilder`/
//! `Lambda`/`LET`). Without this, an operator whose name COLLIDES with a binder's bound variable
//! (e.g. `x == 99` and `[x \in 1..2 |-> IF x = 1 THEN "a" ELSE "b"]`) would be substituted INTO
//! the binder's body — capturing the bound occurrence, collapsing the function to a constant
//! (`IF 99 = 1 …` ⇒ `"b"`), and letting the symbolic prover "prove" an otherwise-FALSE invariant
//! (a FALSE SAFE: `ty check` says VIOLATED, `ty certify` said CERTIFIED). We track the bound names
//! in `bound_vars` and decline to inline any name in that set — the bound occurrence refers to the
//! binder, never to the module operator. This mirrors the discipline `enumerate::expand_operators`
//! already uses (fix for #1558). FAIL-CLOSED: a shadowed reference is left as a bare `Ident`, and
//! the downstream CHC translation/recognizer declines exactly as before.

use std::collections::{HashMap, HashSet};

use tla_core::ast::{BoundVar, Expr, OperatorDef, Substitution};
use tla_core::name_intern::NameId;
use tla_core::ExprFold;
use tla_core::{free_vars, inlining_is_capture_safe, single_bound_var_names};
use tla_core::{Span, Spanned};

use crate::enumerate::try_value_to_expr;
use crate::error_policy::{eval_speculative, FallbackClass};
use crate::eval::{apply_substitutions, EvalCtx};
use crate::expr_visitor::expr_contains_prime_v as expr_contains_prime;

/// Expand operators in an expression for CHC translation
///
/// # Arguments
/// * `ctx` - Evaluation context with operator definitions
/// * `expr` - Expression to expand
/// * `allow_primed` - If true, inline operators even if they contain primes
pub fn expand_operators_for_chc(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    allow_primed: bool,
) -> Spanned<Expr> {
    let mut expand = ExpandOperatorsChc::new(ctx, allow_primed);
    expand.fold_expr(expr.clone())
}

/// A LET-local definition eligible for inlining (T2 widening 4). The body is
/// already fully folded (globals + earlier same-LET defs inlined) when the
/// entry is created, so an inline site substitutes/clones without re-folding.
#[derive(Clone)]
struct LetInlineDef {
    def: OperatorDef,
    /// FREE names of the folded body, minus the def's own parameters. An inline
    /// site is declined when any of these is bound AT THE SITE (the inlined
    /// copy would be captured by the site's binder) — fail-closed: the declined
    /// reference stays a bare `Ident`, and the LET wrapper is then kept.
    free: Vec<String>,
}

struct ExpandOperatorsChc<'a> {
    ctx: &'a EvalCtx,
    allow_primed: bool,
    expanding: HashSet<String>,
    /// Names bound by an enclosing binder (`FuncDef`/`∀`/`∃`/`CHOOSE`/`SetFilter`/`SetBuilder`/
    /// `Lambda`/`LET`). A reference to such a name refers to the BINDER, so it must NOT be inlined
    /// as a module operator / config constant — doing so would be variable capture (see the module
    /// doc). Threaded like `enumerate::ExpandOperators::bound_vars` (fix for #1558).
    bound_vars: HashSet<String>,
    /// One frame per enclosing LET holding its INLINABLE (non-recursive,
    /// non-shadowing) defs. A name in a frame is NOT in `bound_vars` (it was
    /// unshadowed when it became inlinable) unless an inner binder re-bound it
    /// — `bound_vars` is checked first, so the innermost binding always wins.
    /// Lookup is innermost-frame-first.
    let_env: Vec<HashMap<String, LetInlineDef>>,
}

impl<'a> ExpandOperatorsChc<'a> {
    fn new(ctx: &'a EvalCtx, allow_primed: bool) -> Self {
        Self {
            ctx,
            allow_primed,
            expanding: HashSet::new(),
            bound_vars: HashSet::new(),
            let_env: Vec::new(),
        }
    }

    /// Innermost-first lookup of a LET-local inlinable def.
    fn lookup_let_def(&self, name: &str) -> Option<&LetInlineDef> {
        self.let_env.iter().rev().find_map(|frame| frame.get(name))
    }

    /// `true` iff every free name of the candidate inlined body is UNBOUND at
    /// the current site (no capture by the site's enclosing binders).
    fn let_body_free_at_site(&self, entry: &LetInlineDef) -> bool {
        entry.free.iter().all(|f| !self.bound_vars.contains(f))
    }

    fn can_inline(&self, body: &Spanned<Expr>, contains_prime: bool) -> bool {
        let has_primes = contains_prime || expr_contains_prime(&body.node);
        self.allow_primed || !has_primes
    }

    /// Fold `body` with `names` added to the shadow scope, restoring the scope afterwards. Only
    /// NEWLY-introduced names are removed on exit, so nested binders that re-bind the same name do
    /// not prematurely unshadow it (mirrors `enumerate::ExpandOperators::fold_with_bound_scope`).
    fn fold_with_bound_scope(&mut self, names: Vec<String>, body: Spanned<Expr>) -> Spanned<Expr> {
        let added: Vec<String> = names
            .into_iter()
            .filter(|n| self.bound_vars.insert(n.clone()))
            .collect();
        let result = self.fold_expr(body);
        for n in added {
            self.bound_vars.remove(&n);
        }
        result
    }

    /// Fold a MULTI-bound binder (`∀`/`∃`/`FuncDef`/`SetBuilder`) with TELESCOPING scope. TLA+
    /// scopes each EARLIER bound variable into every LATER domain — in `\A x \in S, y \in T(x)`, the
    /// bound `x` is IN SCOPE inside `T(x)`. So fold `bounds[i].domain` only AFTER `bounds[0..i]`'s
    /// names are shadowed (a var is NOT in scope in its OWN domain, so it is registered only once its
    /// domain has been folded), then fold `body` with EVERY bound name shadowed. Without this an
    /// earlier bound var referenced in a later domain was left UNSHADOWED, so a colliding config
    /// CONSTANT / zero-arity operator captured it (`\E n \in 2..2, j \in 1..n` with `CONSTANT n = 10`
    /// widened `1..n` to `1..10`) — the TELESCOPING-DOMAIN false-safe. Restores the scope afterwards;
    /// only NEWLY-introduced names are removed, so a nested re-bind of the same name is preserved.
    fn fold_telescoping_binder(
        &mut self,
        bounds: Vec<BoundVar>,
        body: Spanned<Expr>,
    ) -> (Vec<BoundVar>, Spanned<Expr>) {
        let mut added: Vec<String> = Vec::new();
        let mut new_bounds: Vec<BoundVar> = Vec::with_capacity(bounds.len());
        for bv in bounds {
            // Fold THIS var's domain in the scope that already shadows every EARLIER bound name.
            let new_bv = self.fold_bound_var(bv);
            // Now register THIS var's name(s) so all SUBSEQUENT domains — and the body — see it
            // shadowed and never inline a colliding operator/constant over the bound occurrence.
            for n in single_bound_var_names(&new_bv) {
                if self.bound_vars.insert(n.clone()) {
                    added.push(n);
                }
            }
            new_bounds.push(new_bv);
        }
        let new_body = self.fold_expr(body);
        for n in added {
            self.bound_vars.remove(&n);
        }
        (new_bounds, new_body)
    }

    fn fold_config_constant_ident(&self, name: String, span: Span) -> Spanned<Expr> {
        let ident = Spanned::new(Expr::Ident(name.clone(), NameId::INVALID), span);
        match eval_speculative(self.ctx, &ident, &[FallbackClass::ConstantResolution]) {
            Ok(Some(value)) => {
                if let Some(expr) = try_value_to_expr(&value) {
                    return Spanned::new(expr, span);
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!(
                    "Warning: unexpected eval error during config constant '{name}' resolution (kept as Ident): {e}"
                );
            }
        }
        ident
    }

    fn fold_ident_expr(&mut self, name: String, span: Span) -> Spanned<Expr> {
        // SOUNDNESS: a name bound by an enclosing binder shadows any module operator / config
        // constant of the same name. Leave it as a bare `Ident` (the binder's variable) — inlining
        // would capture the bound occurrence (the AY-lane false-safe). Checked FIRST.
        if self.bound_vars.contains(&name) {
            return Spanned::new(Expr::Ident(name, NameId::INVALID), span);
        }

        // T2 widening 4 — a LET-local inlinable def SHADOWS globals: inline its
        // (pre-folded) body when zero-arity and capture-safe at this site;
        // otherwise leave a bare `Ident` (NEVER fall through to the global
        // lookups — the LET name shadows them; the kept reference then forces
        // the LET wrapper to be preserved, fail-closed).
        if let Some(entry) = self.lookup_let_def(&name) {
            if entry.def.params.is_empty() && self.let_body_free_at_site(entry) {
                return entry.def.body.clone();
            }
            return Spanned::new(Expr::Ident(name, NameId::INVALID), span);
        }

        if self.ctx.is_config_constant(name.as_str()) {
            return self.fold_config_constant_ident(name, span);
        }

        let resolved_name = self.ctx.resolve_op_name(&name);
        if let Some(def) = self.ctx.get_op(resolved_name) {
            if def.params.is_empty()
                && !self.expanding.contains(resolved_name)
                && self.can_inline(&def.body, def.contains_prime)
            {
                self.expanding.insert(resolved_name.to_string());
                let expanded = self.fold_expr(def.body.clone());
                self.expanding.remove(resolved_name);
                return expanded;
            }
        }

        Spanned::new(Expr::Ident(name, NameId::INVALID), span)
    }

    fn fold_apply_expr(
        &mut self,
        op_expr: Box<Spanned<Expr>>,
        args: Vec<Spanned<Expr>>,
        span: Span,
    ) -> Spanned<Expr> {
        if let Expr::Ident(op_name, _) = &op_expr.node {
            // T2 widening 4 — a LET-local parameterized def SHADOWS globals at
            // application sites too. Beta-reduce (capture-avoidingly, via the
            // same `inlining_is_capture_safe` + `apply_substitutions` machinery
            // as module operators) when arity matches and both the args and the
            // body's residual free names are capture-safe here; otherwise keep
            // the application with folded args (bare head — the kept reference
            // forces the LET wrapper to be preserved, fail-closed). The entry's
            // body is pre-folded, so no re-fold is needed after substitution.
            if !self.bound_vars.contains(op_name.as_str()) {
                if let Some(entry) = self.lookup_let_def(op_name).cloned() {
                    let folded_args: Vec<_> =
                        args.into_iter().map(|arg| self.fold_expr(arg)).collect();
                    if !entry.def.params.is_empty()
                        && entry.def.params.len() == folded_args.len()
                        && inlining_is_capture_safe(&entry.def, &folded_args)
                        && self.let_body_free_at_site(&entry)
                    {
                        let subs: Vec<Substitution> = entry
                            .def
                            .params
                            .iter()
                            .zip(folded_args.iter())
                            .map(|(param, arg)| Substitution {
                                from: param.name.clone(),
                                to: arg.clone(),
                            })
                            .collect();
                        return apply_substitutions(&entry.def.body, &subs);
                    }
                    return Spanned::new(Expr::Apply(op_expr, folded_args), span);
                }
            }

            let resolved_op_name = self.ctx.resolve_op_name(op_name);

            // SOUNDNESS: a bound name shadows the parameterized operator — `Op(a)` is then an
            // APPLICATION of the bound variable `Op`, not a call of the module operator. Do not
            // inline; fold the (bound-var) head and the arguments instead. Mirrors the zero-arity
            // guard above.
            let shadowed = self.bound_vars.contains(op_name.as_str())
                || self.bound_vars.contains(resolved_op_name);

            // Keep operator token untouched while still expanding args under recursion.
            if !shadowed && self.expanding.contains(resolved_op_name) {
                let new_args = args.into_iter().map(|arg| self.fold_expr(arg)).collect();
                return Spanned::new(Expr::Apply(op_expr, new_args), span);
            }

            if !shadowed {
                if let Some(def) = self.ctx.get_op(resolved_op_name) {
                    if self.can_inline(&def.body, def.contains_prime) {
                        let expanded_args: Vec<_> =
                            args.into_iter().map(|arg| self.fold_expr(arg)).collect();
                        if def.params.len() != expanded_args.len()
                            || !inlining_is_capture_safe(def, &expanded_args)
                        {
                            return Spanned::new(Expr::Apply(op_expr, expanded_args), span);
                        }
                        let subs: Vec<Substitution> = def
                            .params
                            .iter()
                            .zip(expanded_args.iter())
                            .map(|(param, arg)| Substitution {
                                from: param.name.clone(),
                                to: arg.clone(),
                            })
                            .collect();
                        let substituted = apply_substitutions(&def.body, &subs);
                        self.expanding.insert(resolved_op_name.to_string());
                        let expanded = self.fold_expr(substituted);
                        self.expanding.remove(resolved_op_name);
                        return expanded;
                    }
                }
            }
        }

        Spanned::new(
            Expr::Apply(
                Box::new(self.fold_expr(*op_expr)),
                args.into_iter().map(|arg| self.fold_expr(arg)).collect(),
            ),
            span,
        )
    }

    /// Fold a `LET defs IN body`, INLINING non-recursive defs (T2 widening 4)
    /// so action bodies built from LET-local helpers become classifiable by the
    /// deadlock-freedom / Enabled analysis. Semantics-preserving: a LET is a
    /// local definition, so substituting a NON-RECURSIVE def's (folded) body
    /// for its references — parameterless directly, parameterized beta-reduced
    /// at applications — is a pure unfolding, made capture-avoiding by the same
    /// machinery module operators use plus the site-freeness check.
    ///
    /// Processing is sequential (TLA scopes each def over the LATER defs and
    /// the body): every def's body is folded with the earlier inlinable defs
    /// active, then the def either joins the inline frame or stays RESIDUAL:
    ///   - RECURSIVE defs (folded body still mentions the def's own name —
    ///     covers `LET RECURSIVE f(_)` and any self-reference the recursion
    ///     guard left verbatim) stay verbatim, and the LET wrapper is kept —
    ///     the downstream classifier then declines honestly (fail-closed);
    ///   - a def whose name was ALREADY bound in an enclosing scope stays
    ///     residual too (unshadowing it here could unshadow the outer binder).
    /// The wrapper is DROPPED only when every def inlined AND no def name
    /// remains mentioned in the folded body (a capture-declined site keeps its
    /// bare reference and therefore keeps the wrapper).
    fn fold_let_expr(
        &mut self,
        defs: Vec<OperatorDef>,
        body: Spanned<Expr>,
        span: Span,
    ) -> Spanned<Expr> {
        let names: Vec<String> = defs.iter().map(|d| d.name.node.clone()).collect();
        // Shadow ALL def names first: forward references, self references and
        // residual-def references stay bare `Ident`s (never captured by a
        // colliding global).
        let added: HashSet<String> = names
            .iter()
            .filter(|n| self.bound_vars.insert((*n).clone()))
            .cloned()
            .collect();
        self.let_env.push(HashMap::new());

        let mut folded_defs: Vec<OperatorDef> = Vec::with_capacity(defs.len());
        let mut all_inlinable = true;
        for d in defs {
            let folded_body = self.fold_expr(d.body);
            let own = d.name.node.clone();
            let folded = OperatorDef {
                body: folded_body,
                ..d
            };
            let recursive = free_vars(&folded.body.node).contains(&own);
            let outer_shadowed = !added.contains(&own);
            if recursive || outer_shadowed {
                all_inlinable = false;
                folded_defs.push(folded);
                continue;
            }
            // Inlinable: unshadow the name (references from here on resolve to
            // the frame entry; an inner binder re-binding it wins via the
            // `bound_vars`-first check) and record the folded def + its free
            // names (minus params) for the per-site capture check.
            self.bound_vars.remove(&own);
            let param_names: HashSet<&str> =
                folded.params.iter().map(|p| p.name.node.as_str()).collect();
            let free: Vec<String> = free_vars(&folded.body.node)
                .into_iter()
                .filter(|n| !param_names.contains(n.as_str()))
                .collect();
            let entry = LetInlineDef {
                def: folded.clone(),
                free,
            };
            self.let_env
                .last_mut()
                .expect("frame pushed above")
                .insert(own, entry);
            folded_defs.push(folded);
        }

        let new_body = self.fold_expr(body);

        self.let_env.pop();
        for n in &added {
            self.bound_vars.remove(n);
        }

        if all_inlinable && {
            let body_free = free_vars(&new_body.node);
            names.iter().all(|n| !body_free.contains(n))
        } {
            return new_body;
        }
        Spanned::new(Expr::Let(folded_defs, Box::new(new_body)), span)
    }
}

impl ExprFold for ExpandOperatorsChc<'_> {
    fn fold_expr(&mut self, expr: Spanned<Expr>) -> Spanned<Expr> {
        let span = expr.span;
        match expr.node {
            Expr::Ident(name, _) => self.fold_ident_expr(name, span),
            Expr::Apply(op_expr, args) => self.fold_apply_expr(op_expr, args, span),
            Expr::Let(defs, body) => self.fold_let_expr(defs, *body, span),

            // Binding constructs: domains are TELESCOPING — an earlier bound var is in scope inside a
            // LATER domain — so `fold_telescoping_binder` folds each domain in the scope that already
            // shadows all earlier bound names (a var is not in scope in its own domain), then folds
            // the BODY with every bound name shadowed. So a module operator / constant that collides
            // with a bound name is never inlined into a later domain OR the body (capture-avoidance;
            // the BODY-scope + TELESCOPING-DOMAIN false-safe fixes).
            Expr::FuncDef(bounds, body) => {
                let (new_bounds, new_body) = self.fold_telescoping_binder(bounds, *body);
                Spanned::new(Expr::FuncDef(new_bounds, Box::new(new_body)), span)
            }
            Expr::Forall(bounds, body) => {
                let (new_bounds, new_body) = self.fold_telescoping_binder(bounds, *body);
                Spanned::new(Expr::Forall(new_bounds, Box::new(new_body)), span)
            }
            Expr::Exists(bounds, body) => {
                let (new_bounds, new_body) = self.fold_telescoping_binder(bounds, *body);
                Spanned::new(Expr::Exists(new_bounds, Box::new(new_body)), span)
            }
            Expr::Choose(bound, body) => {
                // Single binder: the domain folds in the outer scope (a var is not in scope in its
                // own domain), the body with the name shadowed. No telescoping (one bound var).
                let names = single_bound_var_names(&bound);
                let new_bound = self.fold_bound_var(bound);
                let new_body = self.fold_with_bound_scope(names, *body);
                Spanned::new(Expr::Choose(new_bound, Box::new(new_body)), span)
            }
            Expr::SetFilter(bound, body) => {
                // Single binder — no telescoping (one bound var).
                let names = single_bound_var_names(&bound);
                let new_bound = self.fold_bound_var(bound);
                let new_body = self.fold_with_bound_scope(names, *body);
                Spanned::new(Expr::SetFilter(new_bound, Box::new(new_body)), span)
            }
            Expr::SetBuilder(body, bounds) => {
                let (new_bounds, new_body) = self.fold_telescoping_binder(bounds, *body);
                Spanned::new(Expr::SetBuilder(Box::new(new_body), new_bounds), span)
            }
            Expr::Lambda(params, body) => {
                let names: Vec<String> = params.iter().map(|p| p.node.clone()).collect();
                let new_body = self.fold_with_bound_scope(names, *body);
                Spanned::new(Expr::Lambda(params, Box::new(new_body)), span)
            }

            node => Spanned::new(self.fold_expr_inner(node), span),
        }
    }
}

#[cfg(test)]
#[path = "expand_tests.rs"]
mod tests;
