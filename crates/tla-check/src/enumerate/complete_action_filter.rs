// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fail-closed proof for bypassing whole-action prime post-validation.
//!
//! The normal unified enumerator must replay an action over every candidate
//! successor when an operator call hides primed expressions.  For a narrow
//! class of complete, deterministic-shape actions, that replay is redundant:
//! every state variable is assigned exactly once by a prime-free RHS (or named
//! directly by `UNCHANGED`), and every other expression is a replay-stable,
//! prime-free guard.  This module proves that property once per resolved call
//! site.  Unknown syntax, dynamic operator scope, recursion, replacements, and
//! unrecognised builtins all reject the proof and retain the normal filter.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use rustc_hash::{FxHashMap, FxHashSet};
use tla_core::ast::{BoundVar, ExceptPathElement, Expr, ModuleTarget, OperatorDef};
use tla_core::{single_bound_var_names, NameId, Span, Spanned, VarIndex, VarRegistry};

use crate::eval::EvalCtx;
use crate::Value;

const MAX_PROOF_DEPTH: usize = 48;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CompleteActionCacheKey {
    shared_id: u64,
    call_site_ptr: usize,
    call_site_span: Span,
    resolved_def_ptr: usize,
}

thread_local! {
    /// `Some(target_and_ptr)` is a positive certificate; `None` is a cached
    /// rejection.  Both outcomes matter because most calls are intentionally
    /// outside this optimization's small proof language.
    static COMPLETE_ACTION_CACHE: RefCell<FxHashMap<CompleteActionCacheKey, Option<CachedCompleteAction>>> =
        RefCell::new(FxHashMap::default());
}

#[derive(Clone, Debug)]
struct CachedCompleteAction {
    target_and_ptr: usize,
    /// Holding this Arc forces every later SharedCtx mutation through COW.
    /// Pointer equality therefore pins operators, instances, config metadata,
    /// precomputed constants, and the variable registry in one exact check.
    shared: Arc<tla_eval::SharedCtx>,
    dynamic_values: Arc<[DynamicValueRef]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DynamicValueRef {
    Local(String),
    Config(String),
    /// A shared/precomputed/builtin resolution whose structural inputs are
    /// pinned by `CachedCompleteAction::shared`; only dynamic lexical
    /// shadowing can change its meaning on a cache hit.
    Unshadowed(String),
}

impl DynamicValueRef {
    fn is_still_valid(&self, ctx: &EvalCtx) -> bool {
        match self {
            Self::Local(name) => ctx
                .lookup_binding(name)
                .is_some_and(|value| value.is_concrete_data()),
            Self::Config(name) => {
                !ctx.name_in_local_scope(name) && resolved_dynamic_value_is_concrete(ctx, name)
            }
            Self::Unshadowed(name) => !ctx.name_in_local_scope(name),
        }
    }
}

#[cfg(test)]
thread_local! {
    static TEST_ENABLE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static TEST_PROOF_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_BYPASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Capability passed only from a certified resolved call to its exact outer
/// `And` node.  Raw pointer equality prevents a proof for one operator body
/// from leaking into an unrelated nested conjunction.
#[derive(Clone, Copy, Debug)]
pub(super) struct CompleteActionCertificate {
    target_and_ptr: usize,
}

impl CompleteActionCertificate {
    #[inline]
    pub(super) fn matches(self, expr: &Spanned<Expr>) -> bool {
        self.target_and_ptr == expr as *const Spanned<Expr> as usize
    }
}

fn env_value_enables(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|value| value == "1")
}

#[inline]
fn complete_action_filter_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_ENABLE_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_value_enables(std::env::var_os("TY_COMPLETE_ACTION_NO_PRIME_FILTER").as_deref())
    })
}

/// Clear positive and negative certificates at the normal checker reset
/// boundary.  The key already carries `SharedCtx` identity, but clearing also
/// bounds memory and prevents stale raw AST pointers surviving a run.
pub(crate) fn clear_complete_action_filter_cache() {
    COMPLETE_ACTION_CACHE.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
pub(super) fn note_complete_action_filter_bypass() {
    TEST_BYPASSES.with(|count| count.set(count.get() + 1));
}

#[cfg(not(test))]
#[inline]
pub(super) fn note_complete_action_filter_bypass() {}

/// Try to certify one resolved call-by-value operator application.
///
/// This function performs no evaluation.  In particular, it is safe to call
/// before argument values are pushed into `ctx`: bound actuals are checked as
/// syntax, while formal names are tracked lexically by the proof.
pub(super) fn certify_complete_action_call(
    ctx: &EvalCtx,
    call_site: &Spanned<Expr>,
    source_name: &str,
    resolved_name: &str,
    def: &Arc<OperatorDef>,
    args: &[Spanned<Expr>],
    registry: &VarRegistry,
) -> Option<CompleteActionCertificate> {
    if !complete_action_filter_enabled()
        || !call_site_matches_source(call_site, source_name, args)
        || source_name != resolved_name
        || registry.is_empty()
        || !registry.ptr_eq(ctx.var_registry())
        || ctx.name_in_local_scope(source_name)
        || ctx.is_config_constant(source_name)
        || ctx.local_ops().is_some()
        || ctx.instance_substitutions().is_some()
        || ctx.call_by_name_subs().is_some()
        || registry
            .names()
            .iter()
            .any(|name| ctx.has_local_binding(name))
        || !definition_header_is_safe(def, args.len(), registry)
        // Unified successor enumeration enters this definition body directly,
        // but the replay evaluator uses the Rust builtin when this predicate
        // is true. A certificate for the body could therefore authorize a
        // successor that canonical replay would reject.
        || crate::eval::should_prefer_builtin_override(source_name, def, args.len(), ctx)
    {
        return None;
    }

    let key = CompleteActionCacheKey {
        shared_id: ctx.shared().id(),
        call_site_ptr: call_site as *const Spanned<Expr> as usize,
        call_site_span: call_site.span,
        resolved_def_ptr: Arc::as_ptr(def) as usize,
    };
    if let Some(cached) = COMPLETE_ACTION_CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        return cached.and_then(|cached| {
            (Arc::ptr_eq(&cached.shared, ctx.shared())
                && cached
                    .dynamic_values
                    .iter()
                    .all(|value| value.is_still_valid(ctx)))
            .then_some(CompleteActionCertificate {
                target_and_ptr: cached.target_and_ptr,
            })
        });
    }

    #[cfg(test)]
    TEST_PROOF_RUNS.with(|count| count.set(count.get() + 1));

    let target_and_ptr = prove_complete_action(ctx, def, args, registry);
    COMPLETE_ACTION_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, target_and_ptr.clone());
    });
    target_and_ptr.map(|cached| CompleteActionCertificate {
        target_and_ptr: cached.target_and_ptr,
    })
}

fn call_site_matches_source(
    call_site: &Spanned<Expr>,
    source_name: &str,
    args: &[Spanned<Expr>],
) -> bool {
    let Expr::Apply(op, site_args) = &call_site.node else {
        return false;
    };
    let Expr::Ident(name, name_id) = &op.node else {
        return false;
    };
    name == source_name
        && ident_name_id_matches(name, *name_id)
        && std::ptr::eq(site_args.as_slice(), args)
}

fn definition_header_is_safe(def: &OperatorDef, arg_count: usize, registry: &VarRegistry) -> bool {
    def.params.len() == arg_count
        && !def.is_recursive
        && !def.has_primed_param
        && def
            .params
            .iter()
            .all(|param| param.arity == 0 && registry.get(param.name.node.as_str()).is_none())
        && {
            let mut names = FxHashSet::default();
            def.params
                .iter()
                .all(|param| names.insert(param.name.node.as_str()))
        }
}

fn prove_complete_action(
    ctx: &EvalCtx,
    def: &Arc<OperatorDef>,
    args: &[Spanned<Expr>],
    registry: &VarRegistry,
) -> Option<CachedCompleteAction> {
    let target = exact_outer_and_target(&def.body)?;
    let target_and_ptr = target as *const Spanned<Expr> as usize;

    let mut proof = CompleteActionProof {
        registry,
        writes: vec![false; registry.len()],
        active_defs: FxHashSet::default(),
        bound_names: Vec::new(),
        dynamic_values: Vec::new(),
    };
    if !args.iter().all(|arg| proof.stable_prime_free(ctx, arg, 0)) {
        return None;
    }

    let def_ptr = Arc::as_ptr(def) as usize;
    proof.active_defs.insert(def_ptr);
    let mark = proof.bound_names.len();
    proof
        .bound_names
        .extend(def.params.iter().map(|param| param.name.node.clone()));
    let valid = proof.root_action(ctx, &def.body, 0);
    proof.bound_names.truncate(mark);
    proof.active_defs.remove(&def_ptr);

    (valid && proof.writes.iter().all(|written| *written)).then(|| CachedCompleteAction {
        target_and_ptr,
        shared: Arc::clone(ctx.shared()),
        dynamic_values: proof.dynamic_values.into(),
    })
}

/// Labels and one action-level LET are transparent, but the capability is
/// anchored to the exact conjunction which owns the prime filter.
fn exact_outer_and_target(mut expr: &Spanned<Expr>) -> Option<&Spanned<Expr>> {
    loop {
        match &expr.node {
            Expr::Label(label) => expr = &label.body,
            Expr::Let(_, body) => {
                expr = body;
                while let Expr::Label(label) = &expr.node {
                    expr = &label.body;
                }
                return matches!(&expr.node, Expr::And(..)).then_some(expr);
            }
            Expr::And(..) => return Some(expr),
            _ => return None,
        }
    }
}

struct CompleteActionProof<'a> {
    registry: &'a VarRegistry,
    writes: Vec<bool>,
    active_defs: FxHashSet<usize>,
    bound_names: Vec<String>,
    dynamic_values: Vec<DynamicValueRef>,
}

impl CompleteActionProof<'_> {
    /// Follow only the wrappers through which unified *top-level* dispatch
    /// carries the capability.  Labels reached through a nested conjunct
    /// operator take a different fallback path and are therefore rejected by
    /// `action` below.
    fn root_action(&mut self, ctx: &EvalCtx, expr: &Spanned<Expr>, depth: usize) -> bool {
        if depth > MAX_PROOF_DEPTH {
            return false;
        }
        match &expr.node {
            Expr::Label(label) => self.root_action(ctx, &label.body, depth + 1),
            Expr::Let(defs, body) => {
                let Some(let_ctx) = self.safe_let_context(ctx, defs, depth + 1) else {
                    return false;
                };
                let mark = self.bound_names.len();
                self.bound_names
                    .extend(defs.iter().map(|def| def.name.node.clone()));
                let result = self.root_action(&let_ctx, body, depth + 1);
                self.bound_names.truncate(mark);
                result
            }
            _ => self.action(ctx, expr, depth + 1),
        }
    }

    fn action(&mut self, ctx: &EvalCtx, expr: &Spanned<Expr>, depth: usize) -> bool {
        if depth > MAX_PROOF_DEPTH {
            return false;
        }

        match &expr.node {
            Expr::Label(_) => false,
            Expr::And(left, right) => {
                self.action(ctx, left, depth + 1) && self.action(ctx, right, depth + 1)
            }
            // These nodes fork the conjunct continuation even when their
            // children are prime-free.  A complete-action certificate is a
            // single-emission proof, so they are admissible only in value/RHS
            // context (`stable_prime_free`), never on the action spine.
            Expr::Or(..) | Expr::Exists(..) | Expr::If(..) | Expr::Case(..) => false,
            Expr::Let(defs, body) => {
                let Some(let_ctx) = self.safe_let_context(ctx, defs, depth + 1) else {
                    return false;
                };
                let mark = self.bound_names.len();
                self.bound_names
                    .extend(defs.iter().map(|def| def.name.node.clone()));
                let result = self.action(&let_ctx, body, depth + 1);
                self.bound_names.truncate(mark);
                result
            }
            Expr::Eq(left, right) => {
                if let Some(idx) = self.exact_prime_target(left) {
                    self.mark_write(idx) && self.stable_prime_free(ctx, right, depth + 1)
                } else if self.exact_prime_target(right).is_some() {
                    false
                } else {
                    self.stable_prime_free(ctx, expr, depth + 1)
                }
            }
            Expr::Unchanged(inner) => self.unchanged_targets(inner),
            Expr::Apply(op, args) => self.action_apply(ctx, op, args, expr, depth + 1),
            Expr::Ident(name, name_id) => self.action_ident(ctx, name, *name_id, expr, depth + 1),
            Expr::ModuleRef(target, name, args) => {
                self.action_module_ref(ctx, target, name, args, expr, depth + 1)
            }
            // A prime-free expression is an ordinary guard.  This admits pure
            // membership, IF, CASE, and quantifiers while rejecting any such
            // construct that tries to branch or enumerate primed assignments.
            _ => self.stable_prime_free(ctx, expr, depth + 1),
        }
    }

    fn action_apply(
        &mut self,
        ctx: &EvalCtx,
        op: &Spanned<Expr>,
        args: &[Spanned<Expr>],
        whole: &Spanned<Expr>,
        depth: usize,
    ) -> bool {
        let Expr::Ident(name, name_id) = &op.node else {
            return false;
        };
        if !ident_name_id_matches(name, *name_id)
            || self.is_bound(name)
            || ctx.has_local_binding(name)
            || ctx.name_in_local_scope(name)
            || ctx.is_config_constant(name)
            || ctx.resolve_op_name(name) != name
        {
            return false;
        }
        let Some(def) = ctx.get_op(name) else {
            return self.stable_prime_free(ctx, whole, depth);
        };
        if ctx.instance_substitutions().is_some()
            || !self.definition_call_is_safe(def, args.len())
            || !args
                .iter()
                .all(|arg| self.stable_prime_free(ctx, arg, depth + 1))
        {
            return false;
        }
        self.record_unshadowed(name);
        if crate::eval::should_prefer_builtin_override(name, def, args.len(), ctx) {
            return false;
        }
        self.with_operator_body(ctx, def, |proof, ctx, body| {
            proof.action(ctx, body, depth + 1)
        })
    }

    fn action_ident(
        &mut self,
        ctx: &EvalCtx,
        name: &str,
        name_id: NameId,
        whole: &Spanned<Expr>,
        depth: usize,
    ) -> bool {
        if !ident_name_id_matches(name, name_id) || ctx.resolve_op_name(name) != name {
            return false;
        }
        let Some(def) = ctx.get_op(name) else {
            return self.stable_prime_free(ctx, whole, depth);
        };
        if self.is_bound(name)
            || ctx.has_local_binding(name)
            || ctx.name_in_local_scope(name)
            || ctx.is_config_constant(name)
            || ctx.instance_substitutions().is_some()
            || !self.definition_call_is_safe(def, 0)
        {
            return false;
        }
        self.record_unshadowed(name);
        if crate::eval::should_prefer_builtin_override(name, def, 0, ctx) {
            return false;
        }
        self.with_operator_body(ctx, def, |proof, ctx, body| {
            proof.action(ctx, body, depth + 1)
        })
    }

    fn action_module_ref(
        &mut self,
        ctx: &EvalCtx,
        target: &ModuleTarget,
        op_name: &str,
        args: &[Spanned<Expr>],
        _whole: &Spanned<Expr>,
        depth: usize,
    ) -> bool {
        let Some((instance_ctx, body, params, def)) =
            self.resolve_named_module_ref(ctx, target, op_name, args, depth)
        else {
            return false;
        };
        self.with_resolved_body(&instance_ctx, &body, &params, &def, |proof, ctx, body| {
            proof.action(ctx, body, depth + 1)
        })
    }

    fn stable_prime_free(&mut self, ctx: &EvalCtx, expr: &Spanned<Expr>, depth: usize) -> bool {
        if depth > MAX_PROOF_DEPTH {
            return false;
        }

        match &expr.node {
            Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,
            Expr::StateVar(name, raw_idx, name_id) => {
                self.exact_state_var_read(ctx, name, *raw_idx, *name_id)
            }
            Expr::Ident(name, name_id) => self.stable_ident(ctx, name, *name_id, depth + 1),
            Expr::Label(label) => self.stable_prime_free(ctx, &label.body, depth + 1),
            Expr::Apply(op, args) => self.stable_apply(ctx, op, args, depth + 1),
            Expr::ModuleRef(target, name, args) => {
                self.stable_module_ref(ctx, target, name, args, depth + 1)
            }
            Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Implies(a, b)
            | Expr::Equiv(a, b)
            | Expr::Eq(a, b)
            | Expr::Neq(a, b)
            | Expr::Lt(a, b)
            | Expr::Leq(a, b)
            | Expr::Gt(a, b)
            | Expr::Geq(a, b)
            | Expr::In(a, b)
            | Expr::NotIn(a, b)
            | Expr::Subseteq(a, b)
            | Expr::Union(a, b)
            | Expr::Intersect(a, b)
            | Expr::SetMinus(a, b)
            | Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::IntDiv(a, b)
            | Expr::Mod(a, b)
            | Expr::Pow(a, b)
            | Expr::Range(a, b)
            | Expr::FuncApply(a, b)
            | Expr::FuncSet(a, b) => {
                self.stable_prime_free(ctx, a, depth + 1)
                    && self.stable_prime_free(ctx, b, depth + 1)
            }
            Expr::Not(inner)
            | Expr::Neg(inner)
            | Expr::Domain(inner)
            | Expr::Powerset(inner)
            | Expr::BigUnion(inner) => self.stable_prime_free(ctx, inner, depth + 1),
            Expr::SetEnum(items) | Expr::Tuple(items) | Expr::Times(items) => items
                .iter()
                .all(|item| self.stable_prime_free(ctx, item, depth + 1)),
            Expr::Record(fields) | Expr::RecordSet(fields) => fields
                .iter()
                .all(|(_, value)| self.stable_prime_free(ctx, value, depth + 1)),
            Expr::RecordAccess(base, _) => self.stable_prime_free(ctx, base, depth + 1),
            Expr::Except(base, specs) => {
                self.stable_prime_free(ctx, base, depth + 1)
                    && specs.iter().all(|spec| {
                        spec.path.iter().all(|part| match part {
                            ExceptPathElement::Index(index) => {
                                self.stable_prime_free(ctx, index, depth + 1)
                            }
                            ExceptPathElement::Field(_) => true,
                        }) && self.stable_prime_free(ctx, &spec.value, depth + 1)
                    })
            }
            Expr::If(cond, then_expr, else_expr) => {
                self.stable_prime_free(ctx, cond, depth + 1)
                    && self.stable_prime_free(ctx, then_expr, depth + 1)
                    && self.stable_prime_free(ctx, else_expr, depth + 1)
            }
            Expr::Case(arms, other) => {
                arms.iter().all(|arm| {
                    self.stable_prime_free(ctx, &arm.guard, depth + 1)
                        && self.stable_prime_free(ctx, &arm.body, depth + 1)
                }) && other
                    .as_ref()
                    .is_none_or(|other| self.stable_prime_free(ctx, other, depth + 1))
            }
            Expr::Let(defs, body) => {
                let Some(let_ctx) = self.safe_let_context(ctx, defs, depth + 1) else {
                    return false;
                };
                let mark = self.bound_names.len();
                self.bound_names
                    .extend(defs.iter().map(|def| def.name.node.clone()));
                let result = self.stable_prime_free(&let_ctx, body, depth + 1);
                self.bound_names.truncate(mark);
                result
            }
            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                self.stable_binder(ctx, bounds, body, depth + 1)
            }
            Expr::SetBuilder(body, bounds) => self.stable_binder(ctx, bounds, body, depth + 1),
            Expr::Choose(bound, body) => {
                self.stable_binder(ctx, std::slice::from_ref(bound), body, depth + 1)
            }
            // Deliberately excluded: action/temporal constructs, deferred
            // substitutions, higher-order values, and instance constructors.
            Expr::Prime(_)
            | Expr::Unchanged(_)
            | Expr::Enabled(_)
            | Expr::Always(_)
            | Expr::Eventually(_)
            | Expr::LeadsTo(_, _)
            | Expr::WeakFair(_, _)
            | Expr::StrongFair(_, _)
            | Expr::SubstIn(_, _)
            | Expr::InstanceExpr(_, _)
            | Expr::Lambda(_, _)
            | Expr::FuncDef(_, _)
            | Expr::SetFilter(_, _)
            | Expr::OpRef(_) => false,
        }
    }

    fn stable_ident(&mut self, ctx: &EvalCtx, name: &str, name_id: NameId, depth: usize) -> bool {
        if !ident_name_id_matches(name, name_id) {
            return false;
        }
        if self.is_bound(name) || name == "@" {
            return true;
        }
        if ctx.has_local_binding(name) {
            let concrete = ctx
                .lookup_binding(name)
                .is_some_and(|value| value.is_concrete_data());
            if concrete {
                self.record_dynamic_value(DynamicValueRef::Local(name.to_string()));
            }
            return concrete;
        }
        // An instance/LET-local operator wins runtime lookup even when an
        // outer config/precomputed value or builtin has the same spelling.
        // Resolution signatures are rechecked in the root context, so reject
        // such local operator residue instead of certifying the wrong scope.
        if ctx.name_in_local_scope(name) {
            return false;
        }
        if ctx.is_config_constant(name) {
            let concrete =
                ctx.resolve_op_name(name) == name && resolved_dynamic_value_is_concrete(ctx, name);
            if concrete {
                self.record_dynamic_value(DynamicValueRef::Config(name.to_string()));
            }
            return concrete;
        }
        if ctx.resolve_op_name(name) != name {
            return false;
        }
        // The outer instance operator body has had WITH substitutions applied,
        // but raw helper definitions in its local operator environment have
        // not.  Inlining such a helper here can therefore disagree with both
        // conjunct enumeration and replay evaluation.
        if ctx.instance_substitutions().is_some() && ctx.get_op(name).is_some() {
            return false;
        }
        if let Some(concrete) = precomputed_value_is_concrete(ctx, name) {
            if concrete {
                self.record_unshadowed(name);
            }
            return concrete;
        }
        if let Some(def) = ctx.get_op(name) {
            if !self.definition_call_is_safe(def, 0) || self.definition_is_action(def) {
                return false;
            }
            self.record_unshadowed(name);
            if crate::eval::should_prefer_builtin_override(name, def, 0, ctx) {
                return is_replay_stable_named_builtin(name);
            }
            return self.with_operator_body(ctx, def, |proof, ctx, body| {
                proof.stable_prime_free(ctx, body, depth + 1)
            });
        }
        if is_pure_builtin_constant(name) {
            self.record_unshadowed(name);
            return true;
        }
        false
    }

    fn stable_apply(
        &mut self,
        ctx: &EvalCtx,
        op: &Spanned<Expr>,
        args: &[Spanned<Expr>],
        depth: usize,
    ) -> bool {
        let Expr::Ident(name, name_id) = &op.node else {
            return false;
        };
        if !ident_name_id_matches(name, *name_id)
            || self.is_bound(name)
            || ctx.has_local_binding(name)
            || ctx.name_in_local_scope(name)
            || ctx.is_config_constant(name)
            || ctx.resolve_op_name(name) != name
            || !args
                .iter()
                .all(|arg| self.stable_prime_free(ctx, arg, depth + 1))
        {
            return false;
        }
        self.record_unshadowed(name);
        let Some(def) = ctx.get_op(name) else {
            return is_replay_stable_named_builtin(name);
        };
        if ctx.instance_substitutions().is_some()
            || !self.definition_call_is_safe(def, args.len())
            || self.definition_is_action(def)
        {
            return false;
        }
        if crate::eval::should_prefer_builtin_override(name, def, args.len(), ctx) {
            return is_replay_stable_named_builtin(name);
        }
        self.with_operator_body(ctx, def, |proof, ctx, body| {
            proof.stable_prime_free(ctx, body, depth + 1)
        })
    }

    fn stable_module_ref(
        &mut self,
        ctx: &EvalCtx,
        target: &ModuleTarget,
        op_name: &str,
        args: &[Spanned<Expr>],
        depth: usize,
    ) -> bool {
        let Some((instance_ctx, body, params, def)) =
            self.resolve_named_module_ref(ctx, target, op_name, args, depth)
        else {
            return false;
        };
        if self.definition_is_action(&def) {
            return false;
        }
        self.with_resolved_body(&instance_ctx, &body, &params, &def, |proof, ctx, body| {
            proof.stable_prime_free(ctx, body, depth + 1)
        })
    }

    fn stable_binder(
        &mut self,
        ctx: &EvalCtx,
        bounds: &[BoundVar],
        body: &Spanned<Expr>,
        depth: usize,
    ) -> bool {
        let mark = self.bound_names.len();
        for bound in bounds {
            let Some(domain) = &bound.domain else {
                self.bound_names.truncate(mark);
                return false;
            };
            if !self.stable_prime_free(ctx, domain, depth + 1) {
                self.bound_names.truncate(mark);
                return false;
            }
            for name in single_bound_var_names(bound) {
                if self.registry.get(name.as_str()).is_some() {
                    self.bound_names.truncate(mark);
                    return false;
                }
                self.bound_names.push(name);
            }
        }
        let result = self.stable_prime_free(ctx, body, depth + 1);
        self.bound_names.truncate(mark);
        result
    }

    fn safe_let_context(
        &mut self,
        ctx: &EvalCtx,
        defs: &[OperatorDef],
        depth: usize,
    ) -> Option<EvalCtx> {
        if defs.is_empty()
            || !let_dependencies_are_prior(defs)
            || defs.iter().any(|def| {
                !def.params.is_empty()
                    || def.is_recursive
                    || def.has_primed_param
                    || def.contains_prime
                    || crate::enumerate::expr_contains_any_prime(&def.body.node)
                    || self.registry.get(def.name.node.as_str()).is_some()
            })
        {
            return None;
        }
        let mut names = FxHashSet::default();
        if !defs.iter().all(|def| names.insert(def.name.node.as_str())) {
            return None;
        }

        let mut local_ops = ctx
            .local_ops()
            .as_ref()
            .map(|ops| (**ops).clone())
            .unwrap_or_default();
        for def in defs {
            local_ops.insert(def.name.node.clone(), tla_eval::intern_let_def_arc(def));
        }
        let mut let_ctx = ctx.clone();
        let_ctx.set_local_ops_eager(Arc::new(local_ops));
        let mark = self.bound_names.len();
        self.bound_names
            .extend(defs.iter().map(|def| def.name.node.clone()));
        let definitions_are_stable = defs
            .iter()
            .all(|def| self.stable_prime_free(&let_ctx, &def.body, depth + 1));
        self.bound_names.truncate(mark);
        if !definitions_are_stable {
            return None;
        }
        Some(let_ctx)
    }

    fn resolve_named_module_ref(
        &mut self,
        ctx: &EvalCtx,
        target: &ModuleTarget,
        op_name: &str,
        args: &[Spanned<Expr>],
        depth: usize,
    ) -> Option<(EvalCtx, Spanned<Expr>, Vec<Arc<str>>, Arc<OperatorDef>)> {
        let ModuleTarget::Named(instance_name) = target else {
            return None;
        };
        let compound_name = format!("{instance_name}!{op_name}");
        if self.is_bound(instance_name)
            || ctx.name_in_local_scope(instance_name)
            || !tla_eval::registered_named_module_ref_dispatch_is_direct(
                ctx,
                instance_name,
                op_name,
            )
            || ctx.instance_substitutions().is_some()
            || ctx.op_replacements().contains_key(&compound_name)
            || ctx.op_replacements().contains_key(op_name)
            || ctx.resolve_op_name(op_name) != op_name
            || !args
                .iter()
                .all(|arg| self.stable_prime_free(ctx, arg, depth + 1))
        {
            return None;
        }
        // The direct-dispatch proof also depends on this target remaining free
        // of per-EvalCtx LET/operator shadowing on positive cache hits.
        self.record_unshadowed(instance_name);
        let info = ctx.get_instance(instance_name)?;
        let def = Arc::clone(ctx.get_instance_op_arc(&info.module_name, op_name)?);
        if !self.definition_call_is_safe(&def, args.len()) {
            return None;
        }
        let (instance_ctx, body, params) =
            crate::enabled::resolve_named_module_ref_body_ast_with_params(
                ctx,
                instance_name,
                op_name,
                args,
            )?;
        // A nonempty INSTANCE substitution scope rebuilds the canonical binding
        // chain from substitution thunks plus these operator formals. Unrelated
        // outer action/LET bindings are not present there. Keeping them in the
        // proof would let `self.is_bound` authorize a free body name that
        // runtime resolves as an instance helper, shared name, builtin, or
        // undefined value. Conservatively reject that shape for every instance;
        // true instance formals are rebound canonically and are excluded.
        let formal_names = params
            .iter()
            .map(|name| name.as_ref())
            .collect::<FxHashSet<_>>();
        let forbidden_outer_names = self
            .bound_names
            .iter()
            .map(String::as_str)
            .filter(|name| !formal_names.contains(name))
            .collect::<std::collections::HashSet<_>>();
        if !forbidden_outer_names.is_empty()
            && tla_core::expr_references_any_free_name_v(&body.node, &forbidden_outer_names)
        {
            return None;
        }
        // Certified execution enumerates the raw operator body under the
        // evaluator's lazy INSTANCE scope. Unified assignment extraction,
        // however, resolves a primed LHS structurally by its raw spelling.
        // A non-identity variable substitution (`inner <- outer`) would make
        // those targets disagree, so require the ordered target sequence to be
        // unchanged by WITH substitution. Constant-only substitutions (the
        // RingBuffer `Values <- Int` case) remain eligible.
        if next_state_target_sequence(&def.body.node) != next_state_target_sequence(&body.node) {
            return None;
        }
        Some((instance_ctx, body, params, def))
    }

    fn with_operator_body(
        &mut self,
        ctx: &EvalCtx,
        def: &Arc<OperatorDef>,
        f: impl FnOnce(&mut Self, &EvalCtx, &Spanned<Expr>) -> bool,
    ) -> bool {
        let params: Vec<Arc<str>> = def
            .params
            .iter()
            .map(|param| Arc::from(param.name.node.as_str()))
            .collect();
        self.with_resolved_body(ctx, &def.body, &params, def, f)
    }

    fn with_resolved_body(
        &mut self,
        ctx: &EvalCtx,
        body: &Spanned<Expr>,
        params: &[Arc<str>],
        def: &Arc<OperatorDef>,
        f: impl FnOnce(&mut Self, &EvalCtx, &Spanned<Expr>) -> bool,
    ) -> bool {
        let def_ptr = Arc::as_ptr(def) as usize;
        if !self.active_defs.insert(def_ptr) {
            return false;
        }
        let mark = self.bound_names.len();
        self.bound_names
            .extend(params.iter().map(|name| name.to_string()));
        let result = f(self, ctx, body);
        self.bound_names.truncate(mark);
        self.active_defs.remove(&def_ptr);
        result
    }

    fn definition_call_is_safe(&self, def: &OperatorDef, arg_count: usize) -> bool {
        definition_header_is_safe(def, arg_count, self.registry)
    }

    fn definition_is_action(&self, def: &OperatorDef) -> bool {
        def.contains_prime || crate::enumerate::expr_contains_any_prime(&def.body.node)
    }

    fn exact_prime_target(&self, expr: &Spanned<Expr>) -> Option<VarIndex> {
        let Expr::Prime(inner) = &expr.node else {
            return None;
        };
        let Expr::StateVar(name, raw_idx, name_id) = &inner.node else {
            return None;
        };
        if self.is_bound(name) {
            return None;
        }
        let idx = self.registry.get(name)?;
        (idx.as_usize() == usize::from(*raw_idx) && self.registry.name_id_at(idx) == *name_id)
            .then_some(idx)
    }

    fn exact_state_var_read(
        &self,
        ctx: &EvalCtx,
        name: &str,
        raw_idx: u16,
        name_id: NameId,
    ) -> bool {
        if self.is_bound(name) || ctx.has_local_binding(name) {
            return false;
        }
        let Some(idx) = self.registry.get(name) else {
            return false;
        };
        idx.as_usize() == usize::from(raw_idx) && self.registry.name_id_at(idx) == name_id
    }

    fn unchanged_targets(&mut self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::Tuple(items) => items.iter().all(|item| self.unchanged_targets(item)),
            Expr::StateVar(name, raw_idx, name_id) if !self.is_bound(name) => {
                let Some(idx) = self.registry.get(name) else {
                    return false;
                };
                idx.as_usize() == usize::from(*raw_idx)
                    && self.registry.name_id_at(idx) == *name_id
                    && self.mark_write(idx)
            }
            _ => false,
        }
    }

    fn mark_write(&mut self, idx: VarIndex) -> bool {
        let slot = &mut self.writes[idx.as_usize()];
        if *slot {
            return false;
        }
        *slot = true;
        true
    }

    fn is_bound(&self, name: &str) -> bool {
        self.bound_names.iter().rev().any(|bound| bound == name)
    }

    fn record_dynamic_value(&mut self, value: DynamicValueRef) {
        if !self.dynamic_values.contains(&value) {
            self.dynamic_values.push(value);
        }
    }

    fn record_unshadowed(&mut self, name: &str) {
        self.record_dynamic_value(DynamicValueRef::Unshadowed(name.to_string()));
    }
}

fn let_dependencies_are_prior(defs: &[OperatorDef]) -> bool {
    defs.iter().enumerate().all(|(idx, def)| {
        let forbidden = defs[idx..]
            .iter()
            .map(|candidate| candidate.name.node.as_str())
            .collect::<std::collections::HashSet<_>>();
        !tla_core::expr_references_any_free_name_v(&def.body.node, &forbidden)
    })
}

fn is_pure_builtin_constant(name: &str) -> bool {
    matches!(
        name,
        "Nat" | "Int" | "Real" | "Infinity" | "BOOLEAN" | "STRING"
    )
}

fn ident_name_id_matches(name: &str, name_id: NameId) -> bool {
    name_id == NameId::INVALID || tla_core::name_intern::intern_name(name) == name_id
}

#[derive(Debug, Eq, PartialEq)]
enum NextStateTarget {
    Prime(Option<String>),
    Unchanged(Option<String>),
}

/// Ordered structural targets seen by unified assignment extraction.
///
/// Ordering (rather than a set) also rejects swaps such as `x <- y, y <- x`.
fn next_state_target_sequence(expr: &Expr) -> Vec<NextStateTarget> {
    struct Collector {
        targets: Vec<NextStateTarget>,
    }

    fn collect_unchanged(expr: &Expr, targets: &mut Vec<NextStateTarget>) {
        match expr {
            Expr::Ident(name, _) | Expr::StateVar(name, _, _) => {
                targets.push(NextStateTarget::Unchanged(Some(name.clone())));
            }
            Expr::Tuple(items) => {
                for item in items {
                    collect_unchanged(&item.node, targets);
                }
            }
            _ => targets.push(NextStateTarget::Unchanged(None)),
        }
    }

    impl tla_core::ExprVisitor for Collector {
        type Output = ();

        fn visit_node(&mut self, expr: &Expr) -> Option<Self::Output> {
            match expr {
                Expr::Prime(inner) => {
                    let name = match &inner.node {
                        Expr::Ident(name, _) | Expr::StateVar(name, _, _) => Some(name.clone()),
                        _ => None,
                    };
                    self.targets.push(NextStateTarget::Prime(name));
                    Some(())
                }
                Expr::Unchanged(inner) => {
                    collect_unchanged(&inner.node, &mut self.targets);
                    Some(())
                }
                _ => None,
            }
        }
    }

    let mut collector = Collector {
        targets: Vec::new(),
    };
    tla_core::walk_expr(&mut collector, expr);
    collector.targets
}

fn precomputed_value_is_concrete(ctx: &EvalCtx, name: &str) -> Option<bool> {
    tla_core::name_intern::lookup_name_id(name)
        .and_then(|name_id| ctx.precomputed_constants().get(&name_id))
        .map(Value::is_concrete_data)
}

/// Match runtime value lookup for dynamic/config identifiers, then consult the
/// authoritative promoted-constant table used by the BFS ident fast path.
fn resolved_dynamic_value_is_concrete(ctx: &EvalCtx, name: &str) -> bool {
    ctx.lookup(name)
        .map(|value| value.is_concrete_data())
        .or_else(|| precomputed_value_is_concrete(ctx, name))
        .unwrap_or(false)
}

/// Positive list of name-dispatched builtins whose value is a deterministic
/// function of their already-certified arguments.  Future/unknown builtins
/// fail closed.
fn is_replay_stable_named_builtin(name: &str) -> bool {
    matches!(
        name,
        "Append"
            | "Cardinality"
            | "Front"
            | "Head"
            | "IsFiniteSet"
            | "Last"
            | "Len"
            | "Max"
            | "Mean"
            | "Min"
            | "Permutations"
            | "Product"
            | "Reverse"
            | "SetToSeq"
            | "SubSeq"
            | "Sum"
            | "Tail"
            | "TLCModelValue"
    )
}

#[cfg(test)]
pub(super) fn with_complete_action_filter_test_override<R>(
    enabled: bool,
    f: impl FnOnce() -> R,
) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_ENABLE_OVERRIDE.with(|value| value.set(self.0));
        }
    }

    let previous = TEST_ENABLE_OVERRIDE.with(|value| value.replace(Some(enabled)));
    let _restore = Restore(previous);
    f()
}

#[cfg(test)]
fn reset_test_counts() {
    TEST_PROOF_RUNS.with(|count| count.set(0));
    TEST_BYPASSES.with(|count| count.set(0));
}

#[cfg(test)]
fn test_proof_runs() -> usize {
    TEST_PROOF_RUNS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn test_bypasses() -> usize {
    TEST_BYPASSES.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerate::enumerate_successors;
    use crate::state::State;
    use crate::test_support::{parse_module, parse_module_with_id};
    use crate::Value;
    use tla_core::ast::{Module, Unit};
    use tla_core::{FileId, NameId};

    const REAL_DISRUPTOR_MPMC: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/test/disruptor/Disruptor_MPMC.tla"
    ));
    const REAL_RINGBUFFER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/test/disruptor/RingBuffer.tla"
    ));

    fn register_module_vars(ctx: &mut EvalCtx, module: &Module) -> Vec<Arc<str>> {
        let mut vars = Vec::new();
        for unit in &module.units {
            if let Unit::Variable(names) = &unit.node {
                for name in names {
                    let name: Arc<str> = Arc::from(name.node.as_str());
                    ctx.register_var(Arc::clone(&name));
                    vars.push(name);
                }
            }
        }
        vars
    }

    fn basic_ctx(source: &str) -> (Module, EvalCtx, Vec<Arc<str>>) {
        let module = parse_module(source);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        let vars = register_module_vars(&mut ctx, &module);
        ctx.resolve_state_vars_in_loaded_ops();
        (module, ctx, vars)
    }

    fn operator(module: &Module, name: &str) -> OperatorDef {
        module
            .units
            .iter()
            .find_map(|unit| match &unit.node {
                Unit::Operator(def) if def.name.node == name => Some(def.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing operator {name}"))
    }

    fn call(name: &str, args: Vec<Spanned<Expr>>) -> Spanned<Expr> {
        Spanned::dummy(Expr::Apply(
            Box::new(Spanned::dummy(Expr::Ident(
                name.to_string(),
                NameId::INVALID,
            ))),
            args,
        ))
    }

    fn zero_call(name: &str) -> Spanned<Expr> {
        call(name, Vec::new())
    }

    fn int_call(name: &str, value: i64) -> Spanned<Expr> {
        call(name, vec![Spanned::dummy(Expr::Int(value.into()))])
    }

    fn ident_call(name: &str, arg: &str, name_id: NameId) -> Spanned<Expr> {
        call(
            name,
            vec![Spanned::dummy(Expr::Ident(arg.to_string(), name_id))],
        )
    }

    fn certify_named(
        ctx: &EvalCtx,
        call: &Spanned<Expr>,
        name: &str,
    ) -> Option<CompleteActionCertificate> {
        let def = Arc::clone(ctx.get_op(name).unwrap());
        let Expr::Apply(_, args) = &call.node else {
            unreachable!();
        };
        certify_complete_action_call(ctx, call, name, name, &def, args, ctx.var_registry())
    }

    #[test]
    fn env_gate_is_exact_and_defaults_off() {
        use std::ffi::OsStr;

        assert!(!env_value_enables(None));
        assert!(env_value_enables(Some(OsStr::new("1"))));
        for value in ["", "0", "01", "true", "TRUE", " 1", "1 "] {
            assert!(!env_value_enables(Some(OsStr::new(value))), "{value:?}");
        }
    }

    #[test]
    fn real_disruptor_mpmc_certifies_all_four_actions() {
        let main = parse_module_with_id(REAL_DISRUPTOR_MPMC, FileId(0));
        let ring = parse_module_with_id(REAL_RINGBUFFER, FileId(1));
        let mut ctx = EvalCtx::new();
        ctx.load_module(&main);
        let modules = [(ring.name.node.as_str(), &ring)]
            .into_iter()
            .collect::<rustc_hash::FxHashMap<_, _>>();
        ctx.load_instance_module_with_extends(ring.name.node.clone(), &ring, &modules);
        register_module_vars(&mut ctx, &main);
        ctx.resolve_state_vars_in_loaded_ops();

        // Certification reads no values, but unresolved declared constants are
        // intentionally rejected.  Mark the same bindings a real configured
        // checker installs before enumeration.
        for name in ["MaxPublished", "Writers", "Readers", "Size", "NULL"] {
            ctx.add_config_constant(name.to_string());
            ctx.env_mut().insert(Arc::from(name), Value::bool(false));
        }

        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            for name in ["BeginWrite", "EndWrite", "BeginRead", "EndRead"] {
                let call = int_call(name, 0);
                assert!(
                    certify_named(&ctx, &call, name).is_some(),
                    "real Disruptor_MPMC action {name} must certify"
                );
            }

            // Runtime dispatch resolves the qualified spelling, not merely
            // the operator name inside the instanced module. Revalidate this
            // exact dependency on a same-id clone so a cached positive cannot
            // survive a late compound config replacement.
            let begin_write = int_call("BeginWrite", 0);
            clear_complete_action_filter_cache();
            reset_test_counts();
            assert!(certify_named(&ctx, &begin_write, "BeginWrite").is_some());
            assert_eq!(test_proof_runs(), 1);
            let mut replaced_ctx = ctx.clone();
            replaced_ctx
                .add_op_replacement("Buffer!Write".to_string(), "ReplacementWrite".to_string());
            assert_eq!(replaced_ctx.shared().id(), ctx.shared().id());
            assert!(
                certify_named(&replaced_ctx, &begin_write, "BeginWrite").is_none(),
                "cached module resolution must reject a compound replacement"
            );
            assert_eq!(
                test_proof_runs(),
                1,
                "positive-cache revalidation must fail closed without replaying the proof"
            );
            clear_complete_action_filter_cache();
            assert!(
                certify_named(&replaced_ctx, &begin_write, "BeginWrite").is_none(),
                "a fresh proof must also reject the active compound replacement"
            );

            let mut proof = CompleteActionProof {
                registry: ctx.var_registry(),
                writes: vec![false; ctx.var_registry().len()],
                active_defs: FxHashSet::default(),
                bound_names: Vec::new(),
                dynamic_values: Vec::new(),
            };
            let target = ModuleTarget::Named("Buffer".to_string());
            let args = vec![
                Spanned::dummy(Expr::Int(0.into())),
                Spanned::dummy(Expr::Int(0.into())),
                Spanned::dummy(Expr::Int(0.into())),
            ];
            let (mut instance_ctx, _, _, _) = proof
                .resolve_named_module_ref(&ctx, &target, "Write", &args, 0)
                .expect("first-level Buffer reference must resolve");
            assert!(instance_ctx.instance_substitutions().is_some());
            assert!(
                proof
                    .resolve_named_module_ref(&instance_ctx, &target, "Write", &args, 0)
                    .is_none(),
                "nested module references under an active substitution scope must reject"
            );

            let helper = Spanned::dummy(Expr::Ident(
                "LastIndex".to_string(),
                tla_core::name_intern::intern_name("LastIndex"),
            ));
            assert!(
                !proof.stable_prime_free(&instance_ctx, &helper, 0),
                "raw instance helper definitions must not be traversed under WITH substitutions"
            );

            let shadow_module = parse_module(
                r#"
---- MODULE BuiltinShadow ----
Nat == FALSE
====
"#,
            );
            let mut local_ops = instance_ctx
                .local_ops()
                .as_ref()
                .map(|ops| (**ops).clone())
                .unwrap_or_default();
            local_ops.insert("Nat".to_string(), Arc::new(operator(&shadow_module, "Nat")));
            instance_ctx.set_local_ops_eager(Arc::new(local_ops));
            let builtin_shadow = Spanned::dummy(Expr::Ident(
                "Nat".to_string(),
                tla_core::name_intern::intern_name("Nat"),
            ));
            assert!(
                !proof.stable_prime_free(&instance_ctx, &builtin_shadow, 0),
                "a slow-scope operator must take precedence over a same-named builtin"
            );
        });
    }

    #[test]
    fn rejects_missing_duplicate_prime_rhs_membership_and_action_branching() {
        let source = r#"
---- MODULE CompleteReject ----
VARIABLES x, y
Hidden(v) == x' = v
PureOr == TRUE \/ TRUE
PureExists == \E z \in {1, 2} : TRUE
PureIf == IF TRUE THEN TRUE ELSE FALSE
PureCase == CASE TRUE -> TRUE [] OTHER -> FALSE
Good(v) == /\ Hidden(v) /\ y' = v
Missing(v) == /\ x' = v /\ TRUE
Duplicate(v) == /\ Hidden(v) /\ x' = v /\ y' = v
PrimeRhs(v) == /\ x' = y' /\ y' = v
Membership(v) == /\ x' \in {v} /\ y' = v
Branching(v) == /\ TRUE /\ ((x' = v /\ y' = v) \/ (x' = v + 1 /\ y' = v))
PureOrForkThenError(v) == /\ PureOr /\ x' = v /\ y' = v /\ 1 \div 0 = 0
PureExistsForkThenError(v) == /\ PureExists /\ x' = v /\ y' = v /\ 1 \div 0 = 0
PureIfDispatch(v) == /\ PureIf /\ x' = v /\ y' = v
PureCaseDispatch(v) == /\ PureCase /\ x' = v /\ y' = v
FuncRhs(v) == /\ x' = [i \in Nat |-> v] /\ y' = v
SetFilterRhs(v) == /\ x' = {i \in SUBSET (1..9) : TRUE} /\ y' = v
PrimedFormal(v) == /\ v' = 1 /\ x' = v /\ y' = v
====
"#;
        let (_module, ctx, _vars) = basic_ctx(source);
        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            let good = int_call("Good", 1);
            assert!(certify_named(&ctx, &good, "Good").is_some());
            for name in [
                "Missing",
                "Duplicate",
                "PrimeRhs",
                "Membership",
                "Branching",
                "PureOrForkThenError",
                "PureExistsForkThenError",
                "PureIfDispatch",
                "PureCaseDispatch",
                "FuncRhs",
                "SetFilterRhs",
                "PrimedFormal",
            ] {
                let call = int_call(name, 1);
                assert!(
                    certify_named(&ctx, &call, name).is_none(),
                    "unsafe action {name} must retain the prime filter"
                );
            }
        });
    }

    #[test]
    fn caches_positive_and_negative_results_and_reset_clears_both() {
        let source = r#"
---- MODULE CompleteCache ----
VARIABLES x, y
Hidden(v) == x' = v
Good(v) == /\ Hidden(v) /\ y' = v
Bad(v) == /\ Hidden(v) /\ x' = v /\ y' = v
====
"#;
        let (_module, ctx, _vars) = basic_ctx(source);
        let good = int_call("Good", 1);
        let bad = int_call("Bad", 1);
        clear_complete_action_filter_cache();
        reset_test_counts();

        with_complete_action_filter_test_override(true, || {
            assert!(certify_named(&ctx, &good, "Good").is_some());
            assert!(certify_named(&ctx, &good, "Good").is_some());
            assert_eq!(test_proof_runs(), 1, "positive result must be cached");

            assert!(certify_named(&ctx, &bad, "Bad").is_none());
            assert!(certify_named(&ctx, &bad, "Bad").is_none());
            assert_eq!(test_proof_runs(), 2, "negative result must be cached");

            clear_complete_action_filter_cache();
            assert!(certify_named(&ctx, &good, "Good").is_some());
            assert_eq!(test_proof_runs(), 3, "reset must force a fresh proof");
        });
    }

    #[test]
    fn independent_registry_layout_cannot_reuse_complete_action_proof() {
        let source = r#"
---- MODULE CompleteRegistryIdentity ----
VARIABLE x
Step == /\ x' = 1 /\ TRUE
====
"#;
        let (_module, ctx, _vars) = basic_ctx(source);
        let call = zero_call("Step");
        let def = Arc::clone(ctx.get_op("Step").unwrap());
        let Expr::Apply(_, args) = &call.node else {
            unreachable!();
        };
        let independent = VarRegistry::from_names(ctx.var_registry().names().iter().cloned());
        assert!(!independent.ptr_eq(ctx.var_registry()));

        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            assert!(
                certify_complete_action_call(
                    &ctx,
                    &call,
                    "Step",
                    "Step",
                    &def,
                    args,
                    &independent,
                )
                .is_none(),
                "a structurally equal but independent registry is not the enumerator layout"
            );
        });
    }

    #[test]
    fn cached_resolution_rejects_same_id_helper_override() {
        let source = r#"
---- MODULE CompleteResolutionCache ----
VARIABLE x
Guard == TRUE
Use == /\ Guard /\ x' = 1
====
"#;
        let (_module, ctx, _vars) = basic_ctx(source);
        let call = zero_call("Use");
        clear_complete_action_filter_cache();
        reset_test_counts();

        with_complete_action_filter_test_override(true, || {
            assert!(certify_named(&ctx, &call, "Use").is_some());
            assert!(certify_named(&ctx, &call, "Use").is_some());
            assert_eq!(test_proof_runs(), 1);

            let override_module = parse_module(
                r#"
---- MODULE CompleteResolutionCacheOverride ----
Guard == FALSE
====
"#,
            );
            let mut changed_ctx = ctx.clone();
            changed_ctx.load_module(&override_module);
            assert_eq!(changed_ctx.shared().id(), ctx.shared().id());
            assert!(
                certify_named(&changed_ctx, &call, "Use").is_none(),
                "a cached positive must retain the exact helper definition identity"
            );
            assert_eq!(
                test_proof_runs(),
                1,
                "cache revalidation must reject the stale signature without rerunning the proof"
            );

            let mut widened_registry_ctx = ctx.clone();
            widened_registry_ctx.register_var(Arc::from("y"));
            assert_eq!(widened_registry_ctx.shared().id(), ctx.shared().id());
            assert!(
                certify_named(&widened_registry_ctx, &call, "Use").is_none(),
                "a cached action complete for x cannot cover a newly registered y"
            );
            assert_eq!(test_proof_runs(), 1);
        });
    }

    #[test]
    fn operator_replacement_and_dynamic_local_scope_fail_closed() {
        let source = r#"
---- MODULE CompleteScope ----
VARIABLES x, y
Hidden(v) == x' = v
Good(v) == /\ Hidden(v) /\ y' = v
Other(v) == /\ x' = v + 1 /\ y' = v
====
"#;
        let (_module, mut ctx, _vars) = basic_ctx(source);
        let call = int_call("Good", 1);
        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            let def = Arc::clone(ctx.get_op("Good").unwrap());
            let Expr::Apply(_, args) = &call.node else {
                unreachable!();
            };
            assert!(certify_complete_action_call(
                &ctx,
                &call,
                "Good",
                "Other",
                &def,
                args,
                ctx.var_registry(),
            )
            .is_none());

            let mut local_ops = tla_core::OpEnv::default();
            local_ops.insert(
                "Local".to_string(),
                Arc::clone(ctx.get_op("Other").unwrap()),
            );
            ctx.set_local_ops_eager(Arc::new(local_ops));
            assert!(certify_named(&ctx, &call, "Good").is_none());
        });
    }

    #[test]
    fn dynamic_values_are_concrete_and_positive_cache_rechecks_them() {
        let source = r#"
---- MODULE CompleteDynamicValues ----
VARIABLES x, y
Use(v) == /\ x' = 1 /\ y' = 1 /\ v = v
MakeClosure == LAMBDA z : z
MakeLazy == [z \in Nat |-> z]
MakeSetPred == {z \in SUBSET (1..9) : TRUE}
====
"#;
        let (_module, ctx, _vars) = basic_ctx(source);
        let eval_body = |name: &str| {
            crate::eval::eval(&ctx, &ctx.get_op(name).unwrap().body)
                .unwrap_or_else(|err| panic!("evaluating {name}: {err}"))
        };
        let opaque = [
            eval_body("MakeClosure"),
            eval_body("MakeLazy"),
            eval_body("MakeSetPred"),
        ];
        assert!(matches!(&opaque[0], Value::Closure(_)));
        assert!(matches!(&opaque[1], Value::LazyFunc(_)));
        assert!(matches!(&opaque[2], Value::SetPred(_)));

        let call = ident_call("Use", "Opaque", NameId::INVALID);
        with_complete_action_filter_test_override(true, || {
            for value in &opaque {
                let mut local_ctx = ctx.clone();
                local_ctx.push_binding(Arc::from("Opaque"), value.clone());
                clear_complete_action_filter_cache();
                assert!(
                    certify_named(&local_ctx, &call, "Use").is_none(),
                    "expression-bearing local value must reject"
                );

                let mut config_ctx = ctx.clone();
                config_ctx.add_config_constant("Opaque".to_string());
                config_ctx
                    .env_mut()
                    .insert(Arc::from("Opaque"), value.clone());
                clear_complete_action_filter_cache();
                assert!(
                    certify_named(&config_ctx, &call, "Use").is_none(),
                    "expression-bearing config value must reject"
                );

                let mut precomputed_ctx = ctx.clone();
                Arc::make_mut(precomputed_ctx.shared_arc_mut())
                    .precomputed_constants_mut()
                    .insert(tla_core::name_intern::intern_name("Opaque"), value.clone());
                clear_complete_action_filter_cache();
                assert!(
                    certify_named(&precomputed_ctx, &call, "Use").is_none(),
                    "expression-bearing precomputed value must reject"
                );
            }

            let mut cached_ctx = ctx.clone();
            let mark = cached_ctx.mark_stack();
            cached_ctx.push_binding(Arc::from("Opaque"), Value::int(7));
            clear_complete_action_filter_cache();
            reset_test_counts();
            assert!(certify_named(&cached_ctx, &call, "Use").is_some());
            assert!(certify_named(&cached_ctx, &call, "Use").is_some());
            assert_eq!(test_proof_runs(), 1);

            cached_ctx.pop_to_mark(&mark);
            cached_ctx.push_binding(Arc::from("Opaque"), opaque[0].clone());
            assert!(
                certify_named(&cached_ctx, &call, "Use").is_none(),
                "a cached certificate must recheck the current local value"
            );
            assert_eq!(
                test_proof_runs(),
                1,
                "cache revalidation must not rerun the structural proof"
            );

            cached_ctx.pop_to_mark(&mark);
            cached_ctx.push_binding(Arc::from("Opaque"), Value::int(8));
            assert!(certify_named(&cached_ctx, &call, "Use").is_some());

            let mut precomputed_ctx = ctx.clone();
            Arc::make_mut(precomputed_ctx.shared_arc_mut())
                .precomputed_constants_mut()
                .insert(tla_core::name_intern::intern_name("Opaque"), Value::int(9));
            clear_complete_action_filter_cache();
            reset_test_counts();
            assert!(certify_named(&precomputed_ctx, &call, "Use").is_some());
            Arc::make_mut(precomputed_ctx.shared_arc_mut())
                .precomputed_constants_mut()
                .insert(
                    tla_core::name_intern::intern_name("Opaque"),
                    opaque[1].clone(),
                );
            assert!(
                certify_named(&precomputed_ctx, &call, "Use").is_none(),
                "a cached certificate must recheck the current precomputed value"
            );
            assert_eq!(test_proof_runs(), 1);
        });
    }

    #[test]
    fn forged_ident_and_state_var_metadata_fail_closed() {
        let source = r#"
---- MODULE CompleteMetadata ----
VARIABLES x, y
Use(v) == /\ x' = 1 /\ y' = 1 /\ v = v
Read(v) == /\ x' = y /\ y' = v
====
"#;
        let (_module, mut ctx, _vars) = basic_ctx(source);
        ctx.push_binding(Arc::from("Actor"), Value::int(1));

        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            let forged_ident = ident_call(
                "Use",
                "Actor",
                tla_core::name_intern::intern_name("DifferentName"),
            );
            assert!(certify_named(&ctx, &forged_ident, "Use").is_none());

            let mut forged_head = int_call("Use", 1);
            let Expr::Apply(op, _) = &mut forged_head.node else {
                unreachable!();
            };
            let Expr::Ident(_, name_id) = &mut op.node else {
                unreachable!();
            };
            *name_id = tla_core::name_intern::intern_name("DifferentOperator");
            assert!(certify_named(&ctx, &forged_head, "Use").is_none());

            let call = int_call("Read", 1);
            let Expr::Apply(_, args) = &call.node else {
                unreachable!();
            };
            let original = Arc::clone(ctx.get_op("Read").unwrap());
            assert!(certify_complete_action_call(
                &ctx,
                &call,
                "Read",
                "Read",
                &original,
                args,
                ctx.var_registry(),
            )
            .is_some());

            let mut shadow_ctx = ctx.clone();
            shadow_ctx.push_binding(Arc::from("y"), Value::int(99));
            assert!(
                certify_complete_action_call(
                    &shadow_ctx,
                    &call,
                    "Read",
                    "Read",
                    &original,
                    args,
                    shadow_ctx.var_registry(),
                )
                .is_none(),
                "a local binding must not shadow a certified state-variable read"
            );

            let mut wrong_index = original.as_ref().clone();
            let Expr::And(first, _) = &mut wrong_index.body.node else {
                panic!("Read body must be a conjunction");
            };
            let Expr::Eq(_, rhs) = &mut first.node else {
                panic!("Read first conjunct must be an equality");
            };
            let Expr::StateVar(_, raw_idx, _) = &mut rhs.node else {
                panic!("Read RHS must be a resolved state variable");
            };
            *raw_idx = 0;
            let wrong_index = Arc::new(wrong_index);
            assert!(
                certify_complete_action_call(
                    &ctx,
                    &call,
                    "Read",
                    "Read",
                    &wrong_index,
                    args,
                    ctx.var_registry(),
                )
                .is_none(),
                "forged state-variable index must reject"
            );

            let mut wrong_name_id = original.as_ref().clone();
            let Expr::And(first, _) = &mut wrong_name_id.body.node else {
                unreachable!();
            };
            let Expr::Eq(_, rhs) = &mut first.node else {
                unreachable!();
            };
            let Expr::StateVar(_, _, name_id) = &mut rhs.node else {
                unreachable!();
            };
            *name_id = tla_core::name_intern::intern_name("x");
            let wrong_name_id = Arc::new(wrong_name_id);
            assert!(
                certify_complete_action_call(
                    &ctx,
                    &call,
                    "Read",
                    "Read",
                    &wrong_name_id,
                    args,
                    ctx.var_registry(),
                )
                .is_none(),
                "forged state-variable NameId must reject"
            );
        });
    }

    #[test]
    fn action_spine_name_collisions_and_builtin_override_fail_closed() {
        let source = r#"
---- MODULE CompleteCollisions ----
VARIABLES x, y
Guard == TRUE
GuardApply(v) == TRUE
SetToSeq(s) == TRUE
Nat == y' = 2
Plain == /\ Guard /\ x' = 1 /\ y' = 1
Applied == /\ GuardApply(1) /\ x' = 1 /\ y' = 1
FormalCollision(Guard) == /\ Guard /\ x' = 1 /\ y' = 1
BuiltinOverride == /\ SetToSeq({1}) /\ x' = 1 /\ y' = 1
BuiltinConstantShadow == /\ x' = Nat /\ y' = 1
====
"#;
        let (_module, ctx, _vars) = basic_ctx(source);
        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            let plain = zero_call("Plain");
            let plain_def = Arc::clone(ctx.get_op("Plain").unwrap());
            let Expr::Apply(_, plain_args) = &plain.node else {
                unreachable!();
            };
            assert!(certify_complete_action_call(
                &ctx,
                &plain,
                "Plain",
                "Plain",
                &plain_def,
                plain_args,
                ctx.var_registry(),
            )
            .is_some());

            let formal = int_call("FormalCollision", 1);
            assert!(certify_named(&ctx, &formal, "FormalCollision").is_none());

            let builtin_shadow = zero_call("BuiltinConstantShadow");
            assert!(
                certify_named(&ctx, &builtin_shadow, "BuiltinConstantShadow").is_none(),
                "a shared operator must take precedence over a same-named builtin constant"
            );

            let mut local_ctx = ctx.clone();
            local_ctx.push_binding(Arc::from("Guard"), Value::bool(false));
            assert!(certify_complete_action_call(
                &local_ctx,
                &plain,
                "Plain",
                "Plain",
                &plain_def,
                plain_args,
                local_ctx.var_registry(),
            )
            .is_none());

            let applied = zero_call("Applied");
            let applied_def = Arc::clone(ctx.get_op("Applied").unwrap());
            let Expr::Apply(_, applied_args) = &applied.node else {
                unreachable!();
            };
            assert!(certify_complete_action_call(
                &ctx,
                &applied,
                "Applied",
                "Applied",
                &applied_def,
                applied_args,
                ctx.var_registry(),
            )
            .is_some());

            let mut apply_ctx = ctx.clone();
            apply_ctx.push_binding(Arc::from("GuardApply"), Value::bool(false));
            assert!(certify_complete_action_call(
                &apply_ctx,
                &applied,
                "Applied",
                "Applied",
                &applied_def,
                applied_args,
                apply_ctx.var_registry(),
            )
            .is_none());

            let mut config_ctx = ctx.clone();
            config_ctx.add_config_constant("Guard".to_string());
            config_ctx
                .env_mut()
                .insert(Arc::from("Guard"), Value::bool(false));
            assert!(certify_complete_action_call(
                &config_ctx,
                &plain,
                "Plain",
                "Plain",
                &plain_def,
                plain_args,
                config_ctx.var_registry(),
            )
            .is_none());

            let builtin = zero_call("BuiltinOverride");
            let builtin_def = Arc::clone(ctx.get_op("BuiltinOverride").unwrap());
            let Expr::Apply(_, builtin_args) = &builtin.node else {
                unreachable!();
            };
            clear_complete_action_filter_cache();
            assert!(
                crate::eval::should_prefer_builtin_override(
                    "SetToSeq",
                    ctx.get_op("SetToSeq").unwrap(),
                    1,
                    &ctx,
                ),
                "test requires the unconditional SetToSeq builtin override"
            );
            assert!(certify_complete_action_call(
                &ctx,
                &builtin,
                "BuiltinOverride",
                "BuiltinOverride",
                &builtin_def,
                builtin_args,
                ctx.var_registry(),
            )
            .is_none());
        });
    }

    #[test]
    fn root_builtin_override_cannot_certify_user_action_body() {
        let source = r#"
---- MODULE CompleteRootBuiltinOverride ----
VARIABLE x
SetToSeq(s) == /\ x' = 1 /\ TRUE
Next == SetToSeq({1})
====
"#;
        let (_module, ctx, _vars) = basic_ctx(source);
        let call = Spanned::dummy(Expr::Apply(
            Box::new(Spanned::dummy(Expr::Ident(
                "SetToSeq".to_string(),
                tla_core::name_intern::intern_name("SetToSeq"),
            ))),
            vec![Spanned::dummy(Expr::SetEnum(vec![Spanned::dummy(
                Expr::Int(1.into()),
            )]))],
        ));
        let def = Arc::clone(ctx.get_op("SetToSeq").expect("SetToSeq should be loaded"));
        let Expr::Apply(_, args) = &call.node else {
            unreachable!();
        };

        assert!(crate::eval::should_prefer_builtin_override(
            "SetToSeq",
            &def,
            args.len(),
            &ctx,
        ));
        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            assert!(
                certify_complete_action_call(
                    &ctx,
                    &call,
                    "SetToSeq",
                    "SetToSeq",
                    &def,
                    args,
                    ctx.var_registry(),
                )
                .is_none(),
                "root replay dispatches the builtin rather than this user action body",
            );
        });
    }

    fn module_ref_call_by_value_ctx() -> (Module, EvalCtx, Vec<Arc<str>>) {
        let instance = parse_module_with_id(
            r#"
---- MODULE CompleteModuleRefInstance ----
VARIABLE x
F == 2
C == 2
Use(a, b) == /\ x' = a /\ a = b
ReadsLocal == /\ x' = 1 /\ C = 2
Ignore(unused) == x' = 1
====
"#,
            FileId(1),
        );
        let main = parse_module_with_id(
            r#"
---- MODULE CompleteModuleRefMain ----
EXTENDS Integers
CONSTANT C
VARIABLE x
F == 1
I == INSTANCE CompleteModuleRefInstance WITH x <- x
Scoped(dummy) == /\ I!Use(F, C) /\ F = 1 /\ C = 1
ReadsInstanceLocal(dummy) == /\ I!ReadsLocal /\ TRUE
Erroring(dummy) == /\ I!Ignore(1 \div 0) /\ TRUE
NextScoped == Scoped(0)
NextErroring == Erroring(0)
====
"#,
            FileId(0),
        );
        let mut ctx = EvalCtx::new();
        ctx.load_module(&main);
        let modules = [(instance.name.node.as_str(), &instance)]
            .into_iter()
            .collect::<rustc_hash::FxHashMap<_, _>>();
        ctx.load_instance_module_with_extends(instance.name.node.clone(), &instance, &modules);
        let vars = register_module_vars(&mut ctx, &main);
        ctx.resolve_state_vars_in_loaded_ops();
        ctx.add_config_constant("C".to_string());
        ctx.env_mut().insert(Arc::from("C"), Value::int(1));
        (main, ctx, vars)
    }

    #[test]
    fn renamed_instance_variable_cannot_certify_complete_action() {
        let instance = parse_module_with_id(
            r#"
---- MODULE CompleteRenamedVariableInstance ----
VARIABLE inner
Write(v) == inner' = v
====
"#,
            FileId(1),
        );
        let main = parse_module_with_id(
            r#"
---- MODULE CompleteRenamedVariableMain ----
VARIABLE outer
I == INSTANCE CompleteRenamedVariableInstance WITH inner <- outer
Step(v) == /\ I!Write(v) /\ TRUE
Next == Step(1)
====
"#,
            FileId(0),
        );
        let mut ctx = EvalCtx::new();
        ctx.load_module(&main);
        let modules = [(instance.name.node.as_str(), &instance)]
            .into_iter()
            .collect::<rustc_hash::FxHashMap<_, _>>();
        ctx.load_instance_module_with_extends(instance.name.node.clone(), &instance, &modules);
        register_module_vars(&mut ctx, &main);
        ctx.resolve_state_vars_in_loaded_ops();

        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            assert!(
                certify_named(&ctx, &int_call("Step", 1), "Step").is_none(),
                "raw unified extraction sees `inner`, while replay writes `outer`"
            );
        });
    }

    #[test]
    fn outer_bound_name_cannot_mask_instance_scope_resolution() {
        let instance = parse_module_with_id(
            r#"
---- MODULE CompleteOuterBoundCollisionInstance ----
VARIABLE x
shadow == TRUE
Write == x' = IF shadow THEN 2 ELSE 1
====
"#,
            FileId(1),
        );
        let main = parse_module_with_id(
            r#"
---- MODULE CompleteOuterBoundCollisionMain ----
VARIABLE x
I == INSTANCE CompleteOuterBoundCollisionInstance WITH x <- x
Step(shadow) == /\ I!Write /\ TRUE
Next == Step(FALSE)
====
"#,
            FileId(0),
        );
        let mut ctx = EvalCtx::new();
        ctx.load_module(&main);
        let modules = [(instance.name.node.as_str(), &instance)]
            .into_iter()
            .collect::<rustc_hash::FxHashMap<_, _>>();
        ctx.load_instance_module_with_extends(instance.name.node.clone(), &instance, &modules);
        register_module_vars(&mut ctx, &main);
        ctx.resolve_state_vars_in_loaded_ops();

        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            assert!(
                certify_named(&ctx, &int_call("Step", 0), "Step").is_none(),
                "an outer formal dropped by INSTANCE scope must not mask an inner helper"
            );
        });
    }

    #[test]
    fn local_operator_cannot_mask_registered_instance_dispatch() {
        let instance = parse_module_with_id(
            r#"
---- MODULE CompleteTargetCollisionInstance ----
VARIABLE x
Write == x' = 1
====
"#,
            FileId(1),
        );
        let main = parse_module_with_id(
            r#"
---- MODULE CompleteTargetCollisionMain ----
VARIABLE x
I == INSTANCE CompleteTargetCollisionInstance WITH x <- x
Step == LET I == Write:: FALSE IN /\ I!Write /\ TRUE
Next == Step
====
"#,
            FileId(0),
        );
        let mut ctx = EvalCtx::new();
        ctx.load_module(&main);
        let modules = [(instance.name.node.as_str(), &instance)]
            .into_iter()
            .collect::<rustc_hash::FxHashMap<_, _>>();
        ctx.load_instance_module_with_extends(instance.name.node.clone(), &instance, &modules);
        register_module_vars(&mut ctx, &main);
        ctx.resolve_state_vars_in_loaded_ops();

        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            assert!(
                certify_named(&ctx, &zero_call("Step"), "Step").is_none(),
                "evaluator label dispatch must win over a registered instance of the same name"
            );
        });
    }

    #[test]
    fn shared_label_selector_cannot_preempt_registered_instance_dispatch() {
        let instance = parse_module_with_id(
            r#"
---- MODULE CompleteSubstitutionLabelInstance ----
CONSTANT C
VARIABLE x
Write == x' = 1
====
"#,
            FileId(1),
        );
        let main = parse_module_with_id(
            r#"
---- MODULE CompleteSubstitutionLabelMain ----
VARIABLE x
I == INSTANCE CompleteSubstitutionLabelInstance
       WITH C <- FALSE,
            x <- x
Step == /\ I!Write /\ TRUE
Next == Step
====
"#,
            FileId(0),
        );
        let selector = parse_module_with_id(
            r#"
---- MODULE CompleteSubstitutionLabelSelector ----
I == Write:: FALSE
====
"#,
            FileId(2),
        );
        let mut ctx = EvalCtx::new();
        ctx.load_module(&main);
        let modules = [(instance.name.node.as_str(), &instance)]
            .into_iter()
            .collect::<rustc_hash::FxHashMap<_, _>>();
        ctx.load_instance_module_with_extends(instance.name.node.clone(), &instance, &modules);
        // Keep the registered instance metadata while installing a shared
        // operator of the same name. Evaluator `Def!Label` dispatch checks this
        // operator before consulting the registered instance table.
        ctx.load_module(&selector);
        register_module_vars(&mut ctx, &main);
        ctx.resolve_state_vars_in_loaded_ops();

        assert!(
            !tla_eval::registered_named_module_ref_dispatch_is_direct(&ctx, "I", "Write"),
            "loaded target definition did not retain the selector: {:?}",
            ctx.get_op("I")
        );
        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            assert!(
                certify_named(&ctx, &zero_call("Step"), "Step").is_none(),
                "Def!Label selection preempts registered INSTANCE dispatch"
            );
        });
    }

    #[test]
    fn certified_module_ref_uses_caller_values_and_instance_scope() {
        let (main, ctx, vars) = module_ref_call_by_value_ctx();
        let current = State::from_pairs([("x", Value::int(0))]);

        clear_complete_action_filter_cache();
        with_complete_action_filter_test_override(true, || {
            assert!(certify_named(&ctx, &int_call("Scoped", 0), "Scoped").is_some());
            assert!(
                certify_named(
                    &ctx,
                    &int_call("ReadsInstanceLocal", 0),
                    "ReadsInstanceLocal",
                )
                .is_none(),
                "instance-local operators must not be mistaken for outer config values"
            );
        });

        clear_complete_action_filter_cache();
        reset_test_counts();
        let mut on_ctx = ctx;
        let successors = with_complete_action_filter_test_override(true, || {
            enumerate_successors(&mut on_ctx, &operator(&main, "NextScoped"), &current, &vars)
                .unwrap()
        });
        assert!(test_bypasses() > 0);
        assert_eq!(successors.len(), 1);
        assert_eq!(successors[0].get("x"), Some(&Value::int(1)));
        assert_ne!(
            successors[0].get("x"),
            Some(&Value::int(2)),
            "instance-local F/C must not capture caller-scope actual arguments"
        );
        assert_eq!(on_ctx.local_stack_len(), 0);
        assert!(
            on_ctx.local_ops().is_none(),
            "canonical instance local_ops must be restored after the outer continuation"
        );
        assert!(!on_ctx.skip_prime_validation());
    }

    #[test]
    fn certified_module_ref_evaluates_unused_actuals_call_by_value() {
        let (main, ctx, vars) = module_ref_call_by_value_ctx();
        let next = operator(&main, "NextErroring");
        let current = State::from_pairs([("x", Value::int(0))]);

        // Canonical expression evaluation is call-by-value: the unused actual
        // fails before the instance body reaches its primed assignment.  The
        // legacy unified ModuleRef route substitutes actual syntax lazily and
        // can skip this error, so compare the certified route to the evaluator
        // itself rather than preserving that pre-existing enumeration bug.
        let canonical_error = crate::eval::eval(&ctx, &next.body).unwrap_err();

        clear_complete_action_filter_cache();
        reset_test_counts();
        let mut on_ctx = ctx;
        let on_error = with_complete_action_filter_test_override(true, || {
            enumerate_successors(&mut on_ctx, &next, &current, &vars).unwrap_err()
        });
        assert!(test_bypasses() > 0);
        assert_eq!(on_error.to_string(), canonical_error.to_string());
        assert_eq!(on_error.to_string(), "The second argument of \\div is 0.");
    }

    #[test]
    fn opt_in_direct_route_matches_default_filtered_route() {
        let source = r#"
---- MODULE CompleteParity ----
VARIABLES x, y
Hidden(v) == x' = v
Step(v) ==
  /\ Hidden(v)
  /\ y' = x + v
Next == Step(1)
====
"#;
        let (module, ctx, vars) = basic_ctx(source);
        let next = operator(&module, "Next");
        let current = State::from_pairs([("x", Value::int(4)), ("y", Value::int(9))]);

        clear_complete_action_filter_cache();
        reset_test_counts();
        let mut off_ctx = ctx.clone();
        let off = with_complete_action_filter_test_override(false, || {
            enumerate_successors(&mut off_ctx, &next, &current, &vars).unwrap()
        });
        assert_eq!(test_bypasses(), 0);

        clear_complete_action_filter_cache();
        let mut on_ctx = ctx;
        let on = with_complete_action_filter_test_override(true, || {
            enumerate_successors(&mut on_ctx, &next, &current, &vars).unwrap()
        });
        assert!(
            test_bypasses() > 0,
            "the certified And must take the direct route"
        );

        let rows = |states: &[State]| {
            states
                .iter()
                .map(|state| {
                    state
                        .vars()
                        .map(|(name, value)| (name.to_string(), value.clone()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(rows(&off), rows(&on));
        assert_eq!(
            rows(&on),
            vec![vec![
                ("x".into(), Value::int(1)),
                ("y".into(), Value::int(5))
            ]]
        );
    }
}
