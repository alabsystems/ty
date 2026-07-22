// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Guard-first LET evaluation (WP-21).
//!
//! TLA+ `LET x == e IN body` is a *definition*, not an evaluation-order
//! promise: the meaning is `body` with `x` textually bound to `e`, and TLC
//! evaluates a def lazily at its first use. The bytecode compiler, however,
//! evaluates every zero-arg def eagerly BEFORE the body. For action
//! predicates of the shape
//!
//! ```tla
//! UpdateLeaf ==
//!     LET key == args[1]
//!         val == args[2]
//!     IN /\ state = UPDATE_LEAF
//!        /\ ...
//! ```
//!
//! that eager order evaluates `args[1]` on states where `state /=
//! UPDATE_LEAF` and `args = NIL` — an evaluation TLC would never perform
//! (the first conjunct is false and none of the defs are used). The
//! interpreter tolerates it; compiled native code hits the fail-closed
//! union-read runtime guard and errors out, wasting the native dispatch
//! (btree: 195,234 such errors per full run).
//!
//! This module reorders compilation of `LET defs IN /\ c1 .. /\ cn` to
//! evaluate a *provably safe* prefix of conjuncts BEFORE the defs, short-
//! circuiting to FALSE without touching any def. The reorder is semantics-
//! preserving under the substitution semantics of LET:
//!
//! * a hoisted conjunct references **no name defined by this LET** (so its
//!   substituted form is itself, and shadowing cannot change what it means);
//! * a hoisted conjunct is a **pure state predicate** — reads of state
//!   variables, resolved constants and already-bound outer names only; no
//!   primes, no `UNCHANGED`, no `ENABLED`, no `CHOOSE`, no operator calls —
//!   so evaluating it earlier (before the defs, which bind registers and
//!   write nothing) yields the identical value, and it cannot raise an
//!   error the original order would not also raise (TLC's own conjunct
//!   order evaluates it first anyway);
//! * every zero-arg def is a **pure expression** by the same predicate (no
//!   CHOOSE / effects), so *skipping* its evaluation when a hoisted guard
//!   is false is unobservable — exactly TLC's lazy-LET behavior. Skipped
//!   evaluation can only *remove* operational errors TLC-conformant
//!   checking does not have (the args-NIL class); it can never change a
//!   produced value. Parameterized defs evaluate nothing at binding time.
//!
//! Conjunct order is preserved (only the def evaluation moves later, from
//! before-all-conjuncts to after-the-hoisted-prefix), and when every guard
//! passes the very same set of evaluations happens in an order TLC itself
//! could use. Any impurity, unresolvable name, or def reference fails the
//! analysis closed: the LET compiles exactly as before.
//!
//! Escape hatch: `TY_GUARD_FIRST_LET=0` disables the reorder entirely.

use tla_core::Spanned;

use super::FnCompileState;
use crate::nodes::{
    TirBoundPattern, TirBoundVar, TirExpr, TirLetDef, TirNameKind, TirNameRef,
};

/// Escape hatch: `TY_GUARD_FIRST_LET=0` restores the eager defs-first order.
/// Default ON (campaign convention for semantics-tightening fixes).
pub(super) fn guard_first_let_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| !matches!(std::env::var("TY_GUARD_FIRST_LET").as_deref(), Ok("0")))
}

/// Flatten a same-operator `And` tree in evaluation (in-order) order.
fn flatten_and_chain<'e>(expr: &'e Spanned<TirExpr>, out: &mut Vec<&'e Spanned<TirExpr>>) {
    if let TirExpr::BoolBinOp {
        left,
        op: crate::nodes::TirBoolOp::And,
        right,
    } = &expr.node
    {
        flatten_and_chain(left, out);
        flatten_and_chain(right, out);
    } else {
        out.push(expr);
    }
}

impl<'a> FnCompileState<'a> {
    /// Try to split `LET defs IN body` for guard-first compilation.
    ///
    /// Returns `Some((guards, rest))` — the hoistable conjunct prefix and the
    /// remaining conjuncts (both non-empty, original order) — when:
    /// * the body is a conjunction of at least two conjuncts,
    /// * every zero-arg def is provably pure ([`Self::expr_is_pure_guard`]),
    /// * at least one leading conjunct is a pure guard referencing no name
    ///   defined by this LET.
    ///
    /// Any other shape returns `None` and the caller compiles the LET
    /// unchanged (fail closed).
    pub(super) fn guard_first_split<'e>(
        &self,
        defs: &[TirLetDef],
        body: &'e Spanned<TirExpr>,
    ) -> Option<(Vec<&'e Spanned<TirExpr>>, Vec<&'e Spanned<TirExpr>>)> {
        if !guard_first_let_enabled() || defs.is_empty() {
            return None;
        }
        let mut conjuncts = Vec::new();
        flatten_and_chain(body, &mut conjuncts);
        if conjuncts.len() < 2 {
            return None;
        }

        // Every zero-arg def must be provably pure: a FALSE hoisted guard
        // skips their evaluation entirely. Defs may reference earlier defs
        // of the same LET (evaluation stays in binding order).
        let no_forbidden: [String; 0] = [];
        let mut def_locals: Vec<String> = Vec::new();
        for def in defs {
            if def.params.is_empty()
                && !self.expr_is_pure_guard(&def.body, &mut def_locals.clone(), &no_forbidden)
            {
                return None;
            }
            def_locals.push(def.name.clone());
        }

        // Names defined by this LET may not appear in a hoisted conjunct
        // (references would leave their binding scope; shadowed outer names
        // would silently change meaning).
        let forbidden: Vec<String> = defs.iter().map(|d| d.name.clone()).collect();

        // Maximal hoistable prefix, capped so at least one conjunct stays
        // inside the LET (the defs still evaluate on the all-guards-true
        // path, exactly as many times as before).
        let mut k = 0usize;
        while k < conjuncts.len() - 1 {
            let mut locals: Vec<String> = Vec::new();
            if !self.expr_is_pure_guard(conjuncts[k], &mut locals, &forbidden) {
                break;
            }
            k += 1;
        }
        if k == 0 {
            return None;
        }
        Some((conjuncts[..k].to_vec(), conjuncts[k..].to_vec()))
    }

    /// Whether a name reference is a provably pure read in the current scope.
    ///
    /// Mirrors `compile_name_expr`'s resolution precedence exactly: any
    /// resolution that would emit a `Call`/`CallExternal`/closure — or that
    /// this method cannot attribute — is treated as impure (fail closed).
    fn name_is_pure_guard_ref(
        &self,
        name_ref: &TirNameRef,
        locals: &[String],
        forbidden: &[String],
    ) -> bool {
        let name = name_ref.name.as_str();
        // Binder introduced inside the analyzed expression (quantifier /
        // set-comprehension / nested LET): a plain register read.
        if locals.iter().any(|n| n == name) {
            return true;
        }
        // A name defined by the LET being split: never hoistable.
        if forbidden.iter().any(|n| n == name) {
            return false;
        }
        // Outer LET/quantifier binding already held in a register.
        if self.lookup_binding(name).is_some() {
            return true;
        }
        match name_ref.kind {
            TirNameKind::StateVar { .. } => true,
            TirNameKind::Ident => {
                // 1. Pre-resolved constant -> LoadConst (pure). Same lookup
                //    order as compile_name_expr.
                if let Some(resolved_constants) = self.resolved_constants {
                    let lookup_id = if name_ref.name_id != tla_core::NameId::INVALID {
                        Some(name_ref.name_id)
                    } else {
                        tla_core::name_intern::lookup_name_id(name)
                    };
                    if let Some(id) = lookup_id {
                        if resolved_constants.contains_key(&id) {
                            return true;
                        }
                    }
                }
                let resolved = self.resolve_op_name(name).to_string();
                // 2..6. Anything that resolves to an operator (external
                // callback, callee body, inlineable body, LET-local or
                // global op index) compiles to a call/closure: impure here.
                if self.is_force_external(name, &resolved) {
                    return false;
                }
                if self
                    .callee_bodies
                    .is_some_and(|m| m.contains_key(resolved.as_str()))
                {
                    return false;
                }
                if self.local_op_indices.contains_key(name) {
                    return false;
                }
                if self
                    .op_indices
                    .is_some_and(|m| m.contains_key(resolved.as_str()))
                {
                    return false;
                }
                // 7. Ident that is really a state variable -> LoadVar (pure).
                if self.state_vars.is_some_and(|m| m.contains_key(name)) {
                    return true;
                }
                false
            }
        }
    }

    /// Conservative purity analysis for guard-first hoisting.
    ///
    /// `true` only when the expression provably compiles to pure reads and
    /// pure computation: no primes, no `UNCHANGED`/`ENABLED`/temporal
    /// operators, no `CHOOSE`, no operator calls or closures, and every name
    /// resolvable as a binding, state variable, or resolved constant.
    /// `locals` accumulates binder names introduced while walking (treated
    /// as pure register reads); `forbidden` names reject the expression.
    pub(super) fn expr_is_pure_guard(
        &self,
        expr: &Spanned<TirExpr>,
        locals: &mut Vec<String>,
        forbidden: &[String],
    ) -> bool {
        // Walk one bound-var list: domains are checked in the enclosing
        // scope, then the binder names extend `locals` for the body.
        fn push_bound_var_names(var: &TirBoundVar, locals: &mut Vec<String>) {
            match &var.pattern {
                Some(TirBoundPattern::Tuple(parts)) => {
                    for (n, _) in parts {
                        locals.push(n.clone());
                    }
                }
                Some(TirBoundPattern::Var(n, _)) => locals.push(n.clone()),
                None => locals.push(var.name.clone()),
            }
        }

        let check_bound =
            |this: &Self,
             vars: &[TirBoundVar],
             body: &Spanned<TirExpr>,
             locals: &mut Vec<String>,
             forbidden: &[String]| {
                for var in vars {
                    if let Some(domain) = &var.domain {
                        if !this.expr_is_pure_guard(domain, locals, forbidden) {
                            return false;
                        }
                    }
                }
                let saved = locals.len();
                for var in vars {
                    push_bound_var_names(var, locals);
                }
                let ok = this.expr_is_pure_guard(body, locals, forbidden);
                locals.truncate(saved);
                ok
            };

        match &expr.node {
            TirExpr::Const { .. } | TirExpr::ExceptAt => true,
            TirExpr::Name(name_ref) => self.name_is_pure_guard_ref(name_ref, locals, forbidden),

            TirExpr::ArithNeg(e)
            | TirExpr::BoolNot(e)
            | TirExpr::Powerset(e)
            | TirExpr::BigUnion(e)
            | TirExpr::Domain(e)
            | TirExpr::Label { body: e, .. } => self.expr_is_pure_guard(e, locals, forbidden),

            TirExpr::ArithBinOp { left, right, .. }
            | TirExpr::BoolBinOp { left, right, .. }
            | TirExpr::Cmp { left, right, .. }
            | TirExpr::Subseteq { left, right }
            | TirExpr::SetBinOp { left, right, .. } => {
                self.expr_is_pure_guard(left, locals, forbidden)
                    && self.expr_is_pure_guard(right, locals, forbidden)
            }
            TirExpr::In { elem, set } => {
                self.expr_is_pure_guard(elem, locals, forbidden)
                    && self.expr_is_pure_guard(set, locals, forbidden)
            }
            TirExpr::Range { lo, hi } => {
                self.expr_is_pure_guard(lo, locals, forbidden)
                    && self.expr_is_pure_guard(hi, locals, forbidden)
            }
            TirExpr::KSubset { base, k } => {
                self.expr_is_pure_guard(base, locals, forbidden)
                    && self.expr_is_pure_guard(k, locals, forbidden)
            }
            TirExpr::FuncApply { func, arg } => {
                self.expr_is_pure_guard(func, locals, forbidden)
                    && self.expr_is_pure_guard(arg, locals, forbidden)
            }
            TirExpr::FuncSet { domain, range } => {
                self.expr_is_pure_guard(domain, locals, forbidden)
                    && self.expr_is_pure_guard(range, locals, forbidden)
            }
            TirExpr::If { cond, then_, else_ } => {
                self.expr_is_pure_guard(cond, locals, forbidden)
                    && self.expr_is_pure_guard(then_, locals, forbidden)
                    && self.expr_is_pure_guard(else_, locals, forbidden)
            }
            TirExpr::SetEnum(elems) | TirExpr::Tuple(elems) | TirExpr::Times(elems) => elems
                .iter()
                .all(|e| self.expr_is_pure_guard(e, locals, forbidden)),
            TirExpr::Record(fields) | TirExpr::RecordSet(fields) => fields
                .iter()
                .all(|(_, e)| self.expr_is_pure_guard(e, locals, forbidden)),
            TirExpr::RecordAccess { record, .. } => {
                self.expr_is_pure_guard(record, locals, forbidden)
            }
            TirExpr::Except { base, specs } => {
                self.expr_is_pure_guard(base, locals, forbidden)
                    && specs.iter().all(|spec| {
                        spec.path.iter().all(|elem| match elem {
                            crate::nodes::TirExceptPathElement::Index(e) => {
                                self.expr_is_pure_guard(e, locals, forbidden)
                            }
                            crate::nodes::TirExceptPathElement::Field(_) => true,
                        }) && self.expr_is_pure_guard(&spec.value, locals, forbidden)
                    })
            }
            TirExpr::Case { arms, other } => {
                arms.iter().all(|arm| {
                    self.expr_is_pure_guard(&arm.guard, locals, forbidden)
                        && self.expr_is_pure_guard(&arm.body, locals, forbidden)
                }) && other
                    .as_ref()
                    .is_none_or(|e| self.expr_is_pure_guard(e, locals, forbidden))
            }

            TirExpr::SetFilter { var, body } => {
                check_bound(self, std::slice::from_ref(var), body, locals, forbidden)
            }
            TirExpr::SetBuilder { body, vars }
            | TirExpr::FuncDef { vars, body }
            | TirExpr::Forall { vars, body }
            | TirExpr::Exists { vars, body } => check_bound(self, vars, body, locals, forbidden),

            TirExpr::Let { defs, body } => {
                let saved = locals.len();
                for def in defs {
                    if def.params.is_empty() {
                        if !self.expr_is_pure_guard(&def.body, locals, forbidden) {
                            locals.truncate(saved);
                            return false;
                        }
                    } else {
                        // Parameterized nested def: nothing evaluates at
                        // binding time, but its body becomes callable from
                        // this subtree — require it pure too (params bound).
                        let inner_saved = locals.len();
                        locals.push(def.name.clone());
                        for p in &def.params {
                            locals.push(p.clone());
                        }
                        let ok = self.expr_is_pure_guard(&def.body, locals, forbidden);
                        locals.truncate(inner_saved);
                        if !ok {
                            locals.truncate(saved);
                            return false;
                        }
                        locals.push(def.name.clone());
                        continue;
                    }
                    locals.push(def.name.clone());
                }
                let ok = self.expr_is_pure_guard(body, locals, forbidden);
                locals.truncate(saved);
                ok
            }

            // Effectful / action-level / unresolvable-by-construction forms:
            // never hoist across or defer these (fail closed).
            TirExpr::Prime(_)
            | TirExpr::Unchanged(_)
            | TirExpr::ActionSubscript { .. }
            | TirExpr::Always(_)
            | TirExpr::Eventually(_)
            | TirExpr::LeadsTo { .. }
            | TirExpr::WeakFair { .. }
            | TirExpr::StrongFair { .. }
            | TirExpr::Enabled(_)
            | TirExpr::Choose { .. }
            | TirExpr::Apply { .. }
            | TirExpr::OperatorRef(_)
            | TirExpr::OpRef(_)
            | TirExpr::Lambda { .. } => false,
        }
    }
}
