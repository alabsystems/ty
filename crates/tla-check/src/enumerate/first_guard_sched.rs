// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Conservative strict-first-guard scheduling for quantified actions.
//!
//! The optimization recognized here is intentionally much narrower than a
//! general action scheduler.  It certifies only the shape used by the safe
//! `EndWrite` / `EndRead` actions in `Disruptor_MPMC`:
//!
//! ```text
//! \E actor \in Actors : Action(actor)
//! Action(t) == ... /\ Transition(t, "expected", "next") /\ ...
//! Transition(t, from, to) == /\ pc[t] = from /\ ...
//! ```
//!
//! Once the quantifier domain has been evaluated and one candidate exists, a
//! certified action may inspect the immutable base state's `pc[candidate]`.
//! A successfully applied string value that is definitely different from the
//! certified literal cannot satisfy the first guard, so recursion into the
//! action body is skipped.  Every unknown case falls through to the ordinary
//! evaluator, preserving its errors and successor order.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::sync::Arc;
#[cfg(test)]
use tla_value::Rp;

use rustc_hash::FxHashMap;
use tla_core::ast::{BoundVar, Expr, OperatorDef};
use tla_core::{expr_mentions_name_v, NameId, Spanned, VarIndex, VarRegistry};

use crate::eval::EvalCtx;
use crate::Value;

use super::unified_types::EnumParams;

#[derive(Clone, Debug, PartialEq, Eq)]
struct FirstGuardPlan {
    pc_var_idx: VarIndex,
    expected: Arc<str>,
}

/// One per-enumeration runtime instance.  `pc` is cloned once from the
/// immutable base state, outside the normalized-domain loop.
pub(super) struct FirstGuardRuntime {
    plan: FirstGuardPlan,
    pc: Value,
}

impl FirstGuardRuntime {
    /// Return true only for the one case proven to make the first guard FALSE.
    /// Missing keys, non-functions, non-string values, and representations not
    /// handled here deliberately fall through to normal evaluation.
    #[inline]
    pub(super) fn candidate_mismatches(&self, candidate: &Value) -> bool {
        let applied = match &self.pc {
            Value::Func(func) => func.apply(candidate),
            Value::IntFunc(func) => func.apply(candidate),
            _ => None,
        };
        let mismatches = matches!(
            applied,
            Some(Value::String(actual)) if actual.as_ref() != self.plan.expected.as_ref()
        );
        #[cfg(test)]
        if mismatches {
            FIRST_GUARD_SCHED_TEST_SKIPS.with(|count| count.set(count.get().saturating_add(1)));
        }
        mismatches
    }
}

thread_local! {
    /// Raw AST addresses are stable for one model-checking run.  The normal
    /// enumerate reset path clears this cache before another parsed module can
    /// reuse an address.
    static FIRST_GUARD_PLAN_CACHE: RefCell<FxHashMap<usize, Option<FirstGuardPlan>>> =
        RefCell::new(FxHashMap::default());

    #[cfg(test)]
    static FIRST_GUARD_SCHED_TEST_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };

    #[cfg(test)]
    static FIRST_GUARD_SCHED_TEST_SKIPS: Cell<u64> = const { Cell::new(0) };

    #[cfg(test)]
    static FIRST_GUARD_SCHED_TEST_PREPARES: Cell<u64> = const { Cell::new(0) };

    #[cfg(test)]
    static FIRST_GUARD_SCHED_TEST_PREFILTERS: Cell<u64> = const { Cell::new(0) };

    #[cfg(test)]
    static FIRST_GUARD_SCHED_TEST_CERTIFICATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Clear AST-identity certificates before independently parsed modules can
/// reuse their addresses.
pub(crate) fn clear_first_guard_sched_cache() {
    FIRST_GUARD_PLAN_CACHE.with(|cache| cache.borrow_mut().clear());
}

fn first_guard_sched_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = FIRST_GUARD_SCHED_TEST_OVERRIDE.with(Cell::get) {
        return enabled;
    }

    true
}

/// Prepare the strict filter after the EXISTS domain has already been
/// evaluated.  This ordering is part of the semantic contract: even a wholly
/// disabled action must retain domain evaluation and its errors.
pub(super) fn prepare_first_guard_runtime(
    ctx: &EvalCtx,
    bounds: &[BoundVar],
    body: &Spanned<Expr>,
    p: &EnumParams<'_>,
) -> Option<FirstGuardRuntime> {
    #[cfg(test)]
    FIRST_GUARD_SCHED_TEST_PREPARES.with(|count| count.set(count.get().saturating_add(1)));

    if !first_guard_sched_enabled() || p.tir_leaf.is_none() {
        return None;
    }

    // An AST-address-only cache is scope-independent only at the root action
    // site.  Refuse dynamic LET/INSTANCE/call-by-name environments rather than
    // trying to include them in the key.  The binding-depth test comes before
    // the bytecode-VM TLS lookup because nested quantified actions are a common
    // ineligible case.
    if !ctx.local_stack_is_empty() {
        return None;
    }

    // Reject the overwhelmingly common non-target syntax before touching the
    // VM TLS or plan cache.  NameId validation remains in the one-time
    // certificate so cache hits do not take global interner lookups.
    if direct_call_syntax_names(bounds, body).is_none() {
        return None;
    }

    if ctx.local_ops().is_some()
        || ctx.instance_substitutions().is_some()
        || ctx.call_by_name_subs().is_some()
        || ctx.name_in_local_scope("pc")
    {
        return None;
    }

    if !tla_eval::tir::bytecode_vm_enabled() {
        return None;
    }

    let key = body as *const Spanned<Expr> as usize;
    let cached = FIRST_GUARD_PLAN_CACHE.with(|cache| cache.borrow().get(&key).cloned());
    let plan = match cached {
        Some(plan) => plan,
        None => {
            #[cfg(test)]
            FIRST_GUARD_SCHED_TEST_CERTIFICATIONS
                .with(|count| count.set(count.get().saturating_add(1)));
            let plan = certify_first_guard_plan(ctx, bounds, body, p.registry);
            FIRST_GUARD_PLAN_CACHE.with(|cache| {
                cache.borrow_mut().insert(key, plan.clone());
            });
            plan
        }
    }?;

    // Registry/body mismatches are already rejected by the certificate.  Keep
    // the array access fail-closed as well in case a non-standard caller built
    // inconsistent `vars` and registry collections.
    if plan.pc_var_idx.as_usize() >= p.vars.len()
        || plan.pc_var_idx.as_usize() >= p.base_with_fp.len()
    {
        return None;
    }
    let pc = p.base_with_fp.get(plan.pc_var_idx);
    Some(FirstGuardRuntime { plan, pc })
}

fn certify_first_guard_plan(
    ctx: &EvalCtx,
    bounds: &[BoundVar],
    body: &Spanned<Expr>,
    registry: &VarRegistry,
) -> Option<FirstGuardPlan> {
    // The quantifier body must be a direct call Action(bound).  Literal actual
    // arguments and aliases are refused even when they happen to have the same
    // value, because the runtime candidate is the only key we inspect.
    let (bound_name, action_name) = exact_direct_call_names(bounds, body)?;
    let action_def = resolve_original_shared_op(ctx, action_name, &[bound_name])?;
    let [action_param] = action_def.params.as_slice() else {
        return None;
    };
    if action_param.arity != 0 || action_def.is_recursive || action_def.has_primed_param {
        return None;
    }
    let action_formal = action_param.name.node.as_str();

    // An outer LET is allowed only when the first effective conjunct is
    // independent of every local definition.  This is the same
    // local-definition-independence condition used by the unified LET
    // guard-first short circuit; unused LET bodies therefore remain lazy on a
    // mismatching candidate.
    let (first_action_conjunct, let_names) = first_action_conjunct(action_def)?;
    if let_names
        .iter()
        .any(|name| expr_mentions_name_v(&first_action_conjunct.node, name))
        // `eval_state_var` deliberately honors ordinary evaluator locals even
        // for an already-resolved `StateVar` node.  These bindings/LET values
        // remain installed while the shared Transition body is evaluated, so
        // any prospective `pc` shadow makes the static state read ambiguous.
        || bound_name == "pc"
        || action_formal == "pc"
        || let_names.contains(&"pc")
    {
        return None;
    }

    let first_action_conjunct = unwrap_labels(first_action_conjunct);
    let Expr::Apply(transition_expr, transition_args) = &first_action_conjunct.node else {
        return None;
    };
    let Expr::Ident(transition_name, transition_name_id) = &transition_expr.node else {
        return None;
    };
    if !name_id_matches_spelling(transition_name, *transition_name_id) {
        return None;
    }
    let [key_arg, from_arg, to_arg] = transition_args.as_slice() else {
        return None;
    };
    if !is_exact_ident(key_arg, action_formal) {
        return None;
    }

    let mut action_shadowed = Vec::with_capacity(2 + let_names.len());
    // The EXISTS binding remains installed while the shared Action body is
    // evaluated.  Reject any involved operator/literal name that could be
    // selected differently by a dynamic binding with the same spelling.
    action_shadowed.push(bound_name);
    action_shadowed.push(action_formal);
    action_shadowed.extend(let_names.iter().copied());

    let transition_def = resolve_original_shared_op(ctx, transition_name, &action_shadowed)?;
    let [key_param, from_param, to_param] = transition_def.params.as_slice() else {
        return None;
    };
    if transition_def.is_recursive
        || transition_def.has_primed_param
        || key_param.arity != 0
        || from_param.arity != 0
        || to_param.arity != 0
        || [key_param, from_param, to_param]
            .iter()
            .any(|param| param.name.node.as_str() == "pc")
        || !all_distinct([
            key_param.name.node.as_str(),
            from_param.name.node.as_str(),
            to_param.name.node.as_str(),
        ])
    {
        return None;
    }

    // TY evaluates and installs Transition actuals from left to right.  Thus
    // `from` is evaluated with the key formal already bound, and `to` with the
    // key and from formals already bound.  Carry those prospective shadows
    // through literal resolution instead of resolving every actual in the
    // Action's pre-call scope.
    let mut from_shadowed = action_shadowed.clone();
    from_shadowed.push(key_param.name.node.as_str());
    let expected = certified_literal_string(ctx, from_arg, &from_shadowed)?;

    let mut to_shadowed = from_shadowed;
    to_shadowed.push(from_param.name.node.as_str());
    if !certified_total_literal_or_bound(
        ctx,
        to_arg,
        action_formal,
        &to_shadowed,
        &[key_param.name.node.as_str(), from_param.name.node.as_str()],
    ) {
        return None;
    }

    let first_transition_conjunct = first_conjunct(&transition_def.body)?;
    let Expr::Eq(lhs, rhs) = &unwrap_labels(first_transition_conjunct).node else {
        return None;
    };
    let Expr::FuncApply(pc_expr, guard_key) = &lhs.node else {
        return None;
    };
    let pc_var_idx = exact_pc_state_var(&pc_expr.node, ctx, registry)?;
    if !is_exact_ident(guard_key, key_param.name.node.as_str())
        || !is_exact_ident(rhs, from_param.name.node.as_str())
    {
        return None;
    }

    Some(FirstGuardPlan {
        pc_var_idx,
        expected,
    })
}

/// Return the first flattened conjunct, requiring an actual conjunction so a
/// callee that is merely a value expression cannot be mistaken for an action.
fn first_conjunct(mut expr: &Spanned<Expr>) -> Option<&Spanned<Expr>> {
    expr = unwrap_labels(expr);
    let Expr::And(left, _) = &expr.node else {
        return None;
    };
    expr = left;
    loop {
        expr = unwrap_labels(expr);
        match &expr.node {
            Expr::And(left, _) => expr = left,
            _ => return Some(expr),
        }
    }
}

fn first_action_conjunct(def: &OperatorDef) -> Option<(&Spanned<Expr>, Vec<&str>)> {
    let body = unwrap_labels(&def.body);
    match &body.node {
        Expr::Let(defs, let_body) => {
            // Nested LET and substitution wrappers are deliberately refused.
            let first = first_conjunct(let_body)?;
            let names = defs.iter().map(|def| def.name.node.as_str()).collect();
            Some((first, names))
        }
        _ => Some((first_conjunct(body)?, Vec::new())),
    }
}

fn unwrap_labels(mut expr: &Spanned<Expr>) -> &Spanned<Expr> {
    while let Expr::Label(label) = &expr.node {
        expr = &label.body;
    }
    expr
}

/// Extract the only call-site syntax eligible for certification.  This hot
/// prefilter deliberately checks spelling but not NameIds: the one-time
/// certificate below validates those identities, while cache hits avoid global
/// interner lookups.
fn direct_call_syntax_names<'a>(
    bounds: &'a [BoundVar],
    body: &'a Spanned<Expr>,
) -> Option<(&'a str, &'a str)> {
    #[cfg(test)]
    FIRST_GUARD_SCHED_TEST_PREFILTERS.with(|count| count.set(count.get().saturating_add(1)));

    let [bound] = bounds else {
        return None;
    };
    if bound.pattern.is_some() || bound.domain.is_none() {
        return None;
    }
    let bound_name = bound.name.node.as_str();

    let Expr::Apply(action_expr, action_args) = &unwrap_labels(body).node else {
        return None;
    };
    let Expr::Ident(action_name, _) = &action_expr.node else {
        return None;
    };
    let [action_arg] = action_args.as_slice() else {
        return None;
    };
    is_ident_spelled(action_arg, bound_name).then_some((bound_name, action_name.as_str()))
}

fn exact_direct_call_names<'a>(
    bounds: &'a [BoundVar],
    body: &'a Spanned<Expr>,
) -> Option<(&'a str, &'a str)> {
    let names = direct_call_syntax_names(bounds, body)?;
    let Expr::Apply(action_expr, action_args) = &unwrap_labels(body).node else {
        return None;
    };
    let Expr::Ident(action_name, action_name_id) = &action_expr.node else {
        return None;
    };
    let [action_arg] = action_args.as_slice() else {
        return None;
    };
    (name_id_matches_spelling(action_name, *action_name_id) && is_exact_ident(action_arg, names.0))
        .then_some(names)
}

#[inline]
fn name_id_matches_spelling(name: &str, name_id: NameId) -> bool {
    name_id == NameId::INVALID || tla_core::lookup_name_id(name) == Some(name_id)
}

fn is_ident_spelled(expr: &Spanned<Expr>, expected: &str) -> bool {
    matches!(&unwrap_labels(expr).node, Expr::Ident(name, _) if name == expected)
}

fn is_exact_ident(expr: &Spanned<Expr>, expected: &str) -> bool {
    matches!(
        &unwrap_labels(expr).node,
        Expr::Ident(name, name_id)
            if name == expected && name_id_matches_spelling(name, *name_id)
    )
}

fn all_distinct<const N: usize>(names: [&str; N]) -> bool {
    names
        .iter()
        .enumerate()
        .all(|(idx, name)| names[..idx].iter().all(|prior| prior != name))
}

/// Resolve only the exact module-shared definition selected by the source
/// spelling.  Config replacements and lexical/runtime shadows all fail closed.
fn resolve_original_shared_op<'a>(
    ctx: &'a EvalCtx,
    name: &str,
    shadowed: &[&str],
) -> Option<&'a OperatorDef> {
    if shadowed.iter().rev().any(|bound| *bound == name)
        || ctx.name_in_local_scope(name)
        || ctx.resolve_op_name(name) != name
    {
        return None;
    }
    let shared = ctx.ops().get(name)?;
    let selected = ctx.get_op(name)?;
    Arc::ptr_eq(shared, selected).then_some(shared.as_ref())
}

fn certified_literal_string(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    shadowed: &[&str],
) -> Option<Arc<str>> {
    match &unwrap_labels(expr).node {
        Expr::String(value) => Some(Arc::from(value.as_str())),
        Expr::Ident(name, name_id) => {
            if !name_id_matches_spelling(name, *name_id) {
                return None;
            }
            let def = resolve_original_shared_op(ctx, name, shadowed)?;
            if !def.params.is_empty() || def.is_recursive {
                return None;
            }
            let Expr::String(body_value) = &unwrap_labels(&def.body).node else {
                return None;
            };

            // `eval_ident` selects a precomputed constant before evaluating a
            // shared zero-arg operator body.  Mirror that winning value: using
            // the source literal when a configured/precomputed non-string
            // would actually reach the guard would make the filter unsound.
            match ctx
                .precomputed_constants()
                .get(&effective_name_id(name, *name_id))
            {
                Some(Value::String(value)) => Some(value.into()),
                Some(_) => None,
                None => Some(Arc::from(body_value.as_str())),
            }
        }
        _ => None,
    }
}

fn certified_total_literal_or_bound(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    bound: &str,
    shadowed: &[&str],
    prior_formals: &[&str],
) -> bool {
    match &unwrap_labels(expr).node {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,
        Expr::Ident(name, name_id) if name == bound && name_id_matches_spelling(name, *name_id) => {
            !prior_formals.iter().any(|formal| *formal == name.as_str())
        }
        Expr::Ident(name, name_id) => {
            if !name_id_matches_spelling(name, *name_id) {
                return false;
            }
            resolve_original_shared_op(ctx, name, shadowed).is_some_and(|def| {
                if !def.params.is_empty()
                    || def.is_recursive
                    || !matches!(
                        &unwrap_labels(&def.body).node,
                        Expr::Bool(_) | Expr::Int(_) | Expr::String(_)
                    )
                {
                    return false;
                }
                ctx.precomputed_constants()
                    .get(&effective_name_id(name, *name_id))
                    .is_none_or(|value| {
                        matches!(
                            value,
                            Value::Bool(_) | Value::SmallInt(_) | Value::Int(_) | Value::String(_)
                        )
                    })
            })
        }
        _ => false,
    }
}

#[inline]
fn effective_name_id(name: &str, name_id: NameId) -> NameId {
    if name_id == NameId::INVALID {
        tla_core::lookup_name_id(name).unwrap_or_else(|| tla_core::intern_name(name))
    } else {
        name_id
    }
}

fn exact_pc_state_var(expr: &Expr, ctx: &EvalCtx, registry: &VarRegistry) -> Option<VarIndex> {
    let expected_idx = registry.get("pc")?;
    let Expr::StateVar(name, raw_idx, name_id) = expr else {
        return None;
    };
    (name == "pc"
        && *raw_idx == expected_idx.0
        && (*name_id == NameId::INVALID || *name_id == registry.name_id_at(expected_idx))
        && ctx.var_registry().get("pc") == Some(expected_idx)
        && !ctx.name_in_local_scope("pc"))
    .then_some(expected_idx)
}

#[cfg(test)]
fn with_first_guard_sched_test_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Reset(Option<bool>);

    impl Drop for Reset {
        fn drop(&mut self) {
            FIRST_GUARD_SCHED_TEST_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }

    let previous = FIRST_GUARD_SCHED_TEST_OVERRIDE.with(|slot| slot.replace(Some(enabled)));
    let _reset = Reset(previous);
    f()
}

#[cfg(test)]
fn reset_first_guard_sched_test_skips() {
    FIRST_GUARD_SCHED_TEST_SKIPS.with(|count| count.set(0));
}

#[cfg(test)]
fn first_guard_sched_test_skips() -> u64 {
    FIRST_GUARD_SCHED_TEST_SKIPS.with(Cell::get)
}

#[cfg(test)]
pub(super) fn reset_first_guard_sched_test_prepares() {
    FIRST_GUARD_SCHED_TEST_PREPARES.with(|count| count.set(0));
}

#[cfg(test)]
pub(super) fn first_guard_sched_test_prepares() -> u64 {
    FIRST_GUARD_SCHED_TEST_PREPARES.with(Cell::get)
}

#[cfg(test)]
fn reset_first_guard_sched_test_gate_counts() {
    FIRST_GUARD_SCHED_TEST_PREFILTERS.with(|count| count.set(0));
    FIRST_GUARD_SCHED_TEST_CERTIFICATIONS.with(|count| count.set(0));
}

#[cfg(test)]
fn first_guard_sched_test_prefilters() -> u64 {
    FIRST_GUARD_SCHED_TEST_PREFILTERS.with(Cell::get)
}

#[cfg(test)]
fn first_guard_sched_test_certifications() -> u64 {
    FIRST_GUARD_SCHED_TEST_CERTIFICATIONS.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerate::successor_engine::run_unified_with_tir;
    use crate::error::EvalError;
    use crate::state::ArrayState;
    use crate::value::{FuncBuilder, IntIntervalFunc};
    use std::sync::Arc;
    use tla_core::ast::{Module, Unit};
    use tla_core::{lower, parse_to_syntax_tree, FileId};
    use tla_eval::tir::TirProgram;

    fn setup(src: &str) -> (Module, EvalCtx, Vec<Arc<str>>) {
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
        (module, ctx, vars)
    }

    fn exists_parts<'a>(ctx: &'a EvalCtx, name: &str) -> (&'a [BoundVar], &'a Spanned<Expr>) {
        let def = ctx.get_op(name).expect("operator should exist");
        let Expr::Exists(bounds, body) = &unwrap_labels(&def.body).node else {
            panic!("{name} should be an EXISTS action")
        };
        (bounds, body)
    }

    fn plan_for(ctx: &EvalCtx, name: &str) -> Option<FirstGuardPlan> {
        let (bounds, body) = exists_parts(ctx, name);
        certify_first_guard_plan(ctx, bounds, body, ctx.var_registry())
    }

    fn collect_or_branches<'a>(expr: &'a Spanned<Expr>, out: &mut Vec<&'a Spanned<Expr>>) {
        match &unwrap_labels(expr).node {
            Expr::Or(left, right) => {
                collect_or_branches(left, out);
                collect_or_branches(right, out);
            }
            _ => out.push(expr),
        }
    }

    fn matcher_fixture() -> &'static str {
        r#"
---- MODULE FirstGuardMatcher ----
EXTENDS Integers
VARIABLES pc, x

Access  == "Access"
Advance == "Advance"
AccessAlias == Access

Transition(t, from, to) ==
  /\ pc[t] = from
  /\ pc' = [pc EXCEPT ![t] = to]

TransitionKeyShadowsFrom(Access, from, to) ==
  /\ pc[Access] = from
  /\ pc' = [pc EXCEPT ![Access] = to]

TransitionFromShadowsTo(t, Advance, to) ==
  /\ pc[t] = Advance
  /\ pc' = [pc EXCEPT ![t] = to]

TransitionFromShadowsAction(k, t, to) ==
  /\ pc[k] = t
  /\ pc' = [pc EXCEPT ![k] = to]

Good(t) ==
  LET unused == 1 \div 0
  IN  /\ Transition(t, Access, Advance)
      /\ x' = x + 1

PcFormal(pc) ==
  /\ Transition(pc, Access, Advance)
  /\ x' = x + 1

PcLet(t) ==
  LET pc == "not-the-state-variable"
  IN  /\ Transition(t, Access, Advance)
      /\ x' = x + 1

KeyShadowsFrom(t) ==
  /\ TransitionKeyShadowsFrom(t, Access, Advance)
  /\ x' = x + 1

FromShadowsTo(t) ==
  /\ TransitionFromShadowsTo(t, Access, Advance)
  /\ x' = x + 1

FromShadowsAction(t) ==
  /\ TransitionFromShadowsAction(t, Access, t)
  /\ x' = x + 1

LateGuard(t) ==
  /\ x = 0
  /\ Transition(t, Access, Advance)
  /\ x' = x + 1

DynamicTo(t) ==
  /\ Transition(t, Access, x + 1)
  /\ x' = x + 1

AliasedFrom(t) ==
  /\ Transition(t, AccessAlias, Advance)
  /\ x' = x + 1

PoisonBefore(t) ==
  /\ 1 \div 0 = 0
  /\ Transition(t, Access, Advance)
  /\ x' = x + 1

PoisonLate(t) ==
  /\ Transition(t, Access, Advance)
  /\ 1 \div 0 = 0
  /\ x' = x + 1

NextGood == \E a \in {"a", "b"} : Good(a)
NextLate == \E a \in {"a", "b"} : LateGuard(a)
NextDynamicTo == \E a \in {"a", "b"} : DynamicTo(a)
NextAliasedFrom == \E a \in {"a", "b"} : AliasedFrom(a)
NextPoisonBefore == \E a \in {"a", "b"} : PoisonBefore(a)
NextPoisonLate == \E a \in {"a", "b"} : PoisonLate(a)
NextWrongActual == \E a \in {"a", "b"} : Good("a")
NextBoundShadow == \E Access \in {"a", "b"} : Good(Access)
NextPcBound == \E pc \in {"a", "b"} : Good(pc)
NextPcFormal == \E a \in {"a", "b"} : PcFormal(a)
NextPcLet == \E a \in {"a", "b"} : PcLet(a)
NextKeyShadowsFrom == \E a \in {"a", "b"} : KeyShadowsFrom(a)
NextFromShadowsTo == \E a \in {"a", "b"} : FromShadowsTo(a)
NextFromShadowsAction == \E a \in {"a", "b"} : FromShadowsAction(a)
NextMultiple == \E a \in {"a"}, b \in {"b"} : Good(a)
====
"#
    }

    #[test]
    fn scheduler_is_enabled_by_default_and_test_override_is_scoped() {
        assert!(first_guard_sched_enabled());
        with_first_guard_sched_test_override(false, || {
            assert!(!first_guard_sched_enabled());
        });
        assert!(first_guard_sched_enabled());
        with_first_guard_sched_test_override(true, || {
            assert!(first_guard_sched_enabled());
        });
        assert!(first_guard_sched_enabled());
    }

    #[test]
    fn matcher_accepts_only_the_strict_first_guard_shape() {
        let (_module, ctx, _vars) = setup(matcher_fixture());
        let plan = plan_for(&ctx, "NextGood").expect("strict End-style shape should certify");
        assert_eq!(plan.pc_var_idx, ctx.var_registry().get("pc").unwrap());
        assert_eq!(plan.expected.as_ref(), "Access");
        assert!(
            plan_for(&ctx, "NextPoisonLate").is_some(),
            "a late expression is ordered after the certified false guard"
        );

        for rejected in [
            "NextLate",
            "NextDynamicTo",
            "NextAliasedFrom",
            "NextPoisonBefore",
            "NextWrongActual",
            "NextBoundShadow",
            "NextPcBound",
            "NextPcFormal",
            "NextPcLet",
            "NextKeyShadowsFrom",
            "NextFromShadowsTo",
            "NextFromShadowsAction",
            "NextMultiple",
        ] {
            assert!(
                plan_for(&ctx, rejected).is_none(),
                "{rejected} must fail closed"
            );
        }
    }

    #[test]
    fn direct_call_prefilter_is_exact_but_leaves_deeper_checks_to_certification() {
        let (_module, ctx, _vars) = setup(matcher_fixture());

        let (bounds, body) = exists_parts(&ctx, "NextGood");
        assert_eq!(direct_call_syntax_names(bounds, body), Some(("a", "Good")));
        assert_eq!(exact_direct_call_names(bounds, body), Some(("a", "Good")));

        let (bounds, body) = exists_parts(&ctx, "NextLate");
        assert!(
            direct_call_syntax_names(bounds, body).is_some(),
            "a direct Action(bound) call must reach the deeper first-conjunct checks"
        );
        assert!(certify_first_guard_plan(&ctx, bounds, body, ctx.var_registry()).is_none());

        for rejected in ["NextWrongActual", "NextMultiple"] {
            let (bounds, body) = exists_parts(&ctx, rejected);
            assert!(
                direct_call_syntax_names(bounds, body).is_none(),
                "{rejected} must be rejected before TLS/cache access"
            );
        }
    }

    #[test]
    fn matcher_rejects_noninvalid_name_ids_with_the_wrong_spelling() {
        let (_module, ctx, _vars) = setup(matcher_fixture());
        let actor_id = tla_core::intern_name("actor");
        let wrong_id = tla_core::intern_name("__first_guard_wrong_name_id");

        let resolved = Spanned::dummy(Expr::Ident("actor".to_string(), actor_id));
        let unresolved = Spanned::dummy(Expr::Ident("actor".to_string(), NameId::INVALID));
        let inconsistent = Spanned::dummy(Expr::Ident("actor".to_string(), wrong_id));
        assert!(is_exact_ident(&resolved, "actor"));
        assert!(is_exact_ident(&unresolved, "actor"));
        assert!(!is_exact_ident(&inconsistent, "actor"));

        let (bounds, body) = exists_parts(&ctx, "NextGood");
        let mut forged_body = body.clone();
        let Expr::Apply(_, action_args) = &mut forged_body.node else {
            panic!("NextGood body should be a direct call")
        };
        let Expr::Ident(_, action_arg_id) = &mut action_args[0].node else {
            panic!("NextGood actual should be an identifier")
        };
        *action_arg_id = wrong_id;
        assert!(direct_call_syntax_names(bounds, &forged_body).is_some());
        assert!(exact_direct_call_names(bounds, &forged_body).is_none());
        assert!(certify_first_guard_plan(&ctx, bounds, &forged_body, ctx.var_registry()).is_none());

        let inconsistent_literal = Spanned::dummy(Expr::Ident("Access".to_string(), wrong_id));
        assert!(certified_literal_string(&ctx, &inconsistent_literal, &[]).is_none());
        assert!(!certified_total_literal_or_bound(
            &ctx,
            &inconsistent_literal,
            "not-the-bound",
            &[],
            &[],
        ));
    }

    #[test]
    fn matcher_rejects_config_replacement_ambiguity() {
        let (_module, mut ctx, _vars) = setup(matcher_fixture());
        assert!(plan_for(&ctx, "NextGood").is_some());
        ctx.add_op_replacement("Good".to_string(), "LateGuard".to_string());
        assert!(plan_for(&ctx, "NextGood").is_none());
    }

    #[test]
    fn matcher_uses_the_winning_precomputed_expected_value() {
        let (_module, mut ctx, _vars) = setup(matcher_fixture());
        let access_id = tla_core::intern_name("Access");

        Arc::make_mut(ctx.shared_arc_mut())
            .precomputed_constants_mut()
            .insert(access_id, Value::string("ConfiguredAccess"));
        assert_eq!(
            plan_for(&ctx, "NextGood")
                .expect("a winning precomputed string remains statically comparable")
                .expected
                .as_ref(),
            "ConfiguredAccess"
        );

        Arc::make_mut(ctx.shared_arc_mut())
            .precomputed_constants_mut()
            .insert(access_id, Value::int(7));
        assert!(
            plan_for(&ctx, "NextGood").is_none(),
            "a winning non-string value must not inherit the source body's string"
        );
    }

    #[test]
    fn real_mpmc_certifies_only_endwrite_and_endread() {
        let src = include_str!("../../../../examples/test/disruptor/Disruptor_MPMC.tla");
        let (_module, ctx, _vars) = setup(src);
        let next = ctx.get_op("Next").expect("real MPMC Next should exist");
        let mut branches = Vec::new();
        collect_or_branches(&next.body, &mut branches);
        assert_eq!(branches.len(), 4);

        let plans: Vec<Option<FirstGuardPlan>> = branches
            .into_iter()
            .map(|branch| {
                let Expr::Exists(bounds, body) = &unwrap_labels(branch).node else {
                    panic!("every MPMC Next branch should be a quantified action")
                };
                certify_first_guard_plan(&ctx, bounds, body, ctx.var_registry())
            })
            .collect();

        assert!(plans[0].is_none(), "BeginWrite has a preceding guard");
        assert_eq!(
            plans[1].as_ref().map(|plan| plan.expected.as_ref()),
            Some("Access")
        );
        assert!(plans[2].is_none(), "BeginRead has a preceding guard");
        assert_eq!(
            plans[3].as_ref().map(|plan| plan.expected.as_ref()),
            Some("Access")
        );
    }

    #[test]
    fn runtime_skips_only_successful_unequal_string_applications() {
        let mut builder = FuncBuilder::new();
        builder.insert(Value::string("a"), Value::string("Access"));
        builder.insert(Value::string("b"), Value::string("Advance"));
        builder.insert(Value::string("not-string"), Value::Bool(false));
        let func_pc = Value::Func(Rp::new(builder.build()));
        let plan = FirstGuardPlan {
            pc_var_idx: VarIndex::new(0),
            expected: Arc::from("Access"),
        };
        let runtime = FirstGuardRuntime {
            plan: plan.clone(),
            pc: func_pc,
        };

        assert!(!runtime.candidate_mismatches(&Value::string("a")));
        assert!(runtime.candidate_mismatches(&Value::string("b")));
        assert!(!runtime.candidate_mismatches(&Value::string("missing")));
        assert!(!runtime.candidate_mismatches(&Value::string("not-string")));

        let int_runtime = FirstGuardRuntime {
            plan: plan.clone(),
            pc: Value::IntFunc(Rp::new(IntIntervalFunc::new(
                1,
                2,
                vec![Value::string("Access"), Value::string("Advance")],
            ))),
        };
        assert!(!int_runtime.candidate_mismatches(&Value::int(1)));
        assert!(int_runtime.candidate_mismatches(&Value::int(2)));
        assert!(!int_runtime.candidate_mismatches(&Value::int(3)));

        let wrong_type = FirstGuardRuntime {
            plan,
            pc: Value::string("Advance"),
        };
        assert!(!wrong_type.candidate_mismatches(&Value::string("a")));
    }

    #[test]
    fn cache_hit_reuses_only_the_plan_and_refreshes_pc_for_each_enumeration() {
        let src = r#"
---- MODULE FirstGuardCacheHit ----
EXTENDS Integers
VARIABLES pc, x
Access == "Access"
Advance == "Advance"
Transition(t, from, to) ==
  /\ pc[t] = from
  /\ pc' = [pc EXCEPT ![t] = to]
End(t) ==
  /\ Transition(t, Access, Advance)
  /\ x' = x + 1
Next == \E a \in {"a", "b"} : End(a)
====
"#;
        let (module, mut ctx, vars) = setup(src);
        let next = Arc::clone(ctx.get_op("Next").expect("Next should exist"));
        let tir = TirProgram::from_modules(&module, &[]);
        let registry = ctx.var_registry().clone();

        let state = |a_pc: &str, b_pc: &str| {
            let mut pc_builder = FuncBuilder::new();
            pc_builder.insert(Value::string("a"), Value::string(a_pc));
            pc_builder.insert(Value::string("b"), Value::string(b_pc));
            ArrayState::from_values(vec![
                Value::Func(Rp::new(pc_builder.build())),
                Value::int(0),
            ])
        };
        let first = state("Access", "Advance");
        let second = state("Advance", "Access");

        clear_first_guard_sched_cache();
        reset_first_guard_sched_test_skips();
        reset_first_guard_sched_test_gate_counts();

        let mut run = |current: &ArrayState| {
            let _state_guard = ctx.bind_state_env_guard(current.env_ref());
            with_first_guard_sched_test_override(true, || {
                run_unified_with_tir(&mut ctx, &next.body, current, &vars, &registry, Some(&tir))
            })
            .expect("scheduled enumeration should succeed")
        };

        assert_eq!(run(&first).len(), 1);
        assert_eq!(first_guard_sched_test_certifications(), 1);
        assert_eq!(first_guard_sched_test_skips(), 1);

        // The same AST address must hit the cached certificate, but the reversed
        // pc values must still select the other candidate from the new base state.
        assert_eq!(run(&second).len(), 1);
        assert_eq!(
            first_guard_sched_test_certifications(),
            1,
            "the second enumeration must reuse the cached plan"
        );
        assert_eq!(first_guard_sched_test_skips(), 2);
        clear_first_guard_sched_cache();
    }

    #[test]
    fn nonempty_singleton_with_outer_binding_rejects_before_spelling_prefilter() {
        let src = r#"
---- MODULE FirstGuardOuterBinding ----
EXTENDS Integers
VARIABLES pc, x
Access == "Access"
Advance == "Advance"
Transition(t, from, to) ==
  /\ pc[t] = from
  /\ pc' = [pc EXCEPT ![t] = to]
End(t) ==
  /\ Transition(t, Access, Advance)
  /\ x' = x + 1
Next == \E a \in {"a"} : End(a)
====
"#;
        let (module, mut ctx, vars) = setup(src);
        let next = Arc::clone(ctx.get_op("Next").expect("Next should exist"));
        let tir = TirProgram::from_modules(&module, &[]);
        let registry = ctx.var_registry().clone();

        let mut pc_builder = FuncBuilder::new();
        pc_builder.insert(Value::string("a"), Value::string("Access"));
        let current = ArrayState::from_values(vec![
            Value::Func(Rp::new(pc_builder.build())),
            Value::int(0),
        ]);

        let mark = ctx.mark_stack();
        ctx.push_binding(Arc::from("unrelated_outer"), Value::int(9));
        clear_first_guard_sched_cache();
        reset_first_guard_sched_test_prepares();
        reset_first_guard_sched_test_skips();
        reset_first_guard_sched_test_gate_counts();

        let successors = {
            let _state_guard = ctx.bind_state_env_guard(current.env_ref());
            with_first_guard_sched_test_override(true, || {
                run_unified_with_tir(&mut ctx, &next.body, &current, &vars, &registry, Some(&tir))
            })
            .expect("ordinary enumeration under an outer binding should succeed")
        };

        assert_eq!(successors.len(), 1);
        assert_eq!(
            first_guard_sched_test_prepares(),
            1,
            "the singleton/nonempty call-site gate must reach prepare"
        );
        assert_eq!(
            first_guard_sched_test_prefilters(),
            0,
            "binding depth must reject before the spelling prefilter"
        );
        assert_eq!(first_guard_sched_test_certifications(), 0);
        assert_eq!(first_guard_sched_test_skips(), 0);
        assert_eq!(ctx.lookup_binding("unrelated_outer"), Some(Value::int(9)));
        ctx.pop_to_mark(&mark);
    }

    #[test]
    fn enabled_and_disabled_enumeration_produce_identical_successors() {
        let src = r#"
---- MODULE FirstGuardDifferential ----
EXTENDS Integers
VARIABLES pc, x
Access == "Access"
Advance == "Advance"
Transition(t, from, to) ==
  /\ pc[t] = from
  /\ pc' = [pc EXCEPT ![t] = to]
End(t) ==
  LET unused == 1 \div 0
  IN  /\ Transition(t, Access, Advance)
      /\ x' = x + 1
Next == \E a \in {"a", "b", "c"} : End(a)
====
"#;
        let (module, mut ctx, vars) = setup(src);
        let next = Arc::clone(ctx.get_op("Next").expect("Next should exist"));
        let tir = TirProgram::from_modules(&module, &[]);
        let registry = ctx.var_registry().clone();

        let mut pc_builder = FuncBuilder::new();
        pc_builder.insert(Value::string("a"), Value::string("Access"));
        pc_builder.insert(Value::string("b"), Value::string("Advance"));
        pc_builder.insert(Value::string("c"), Value::string("Access"));
        let current = ArrayState::from_values(vec![
            Value::Func(Rp::new(pc_builder.build())),
            Value::int(0),
        ]);
        let _state_guard = ctx.bind_state_env_guard(current.env_ref());

        clear_first_guard_sched_cache();
        reset_first_guard_sched_test_skips();
        let off = with_first_guard_sched_test_override(false, || {
            run_unified_with_tir(&mut ctx, &next.body, &current, &vars, &registry, Some(&tir))
        })
        .expect("baseline enumeration should succeed");
        assert_eq!(first_guard_sched_test_skips(), 0);

        clear_first_guard_sched_cache();
        reset_first_guard_sched_test_skips();
        let on = with_first_guard_sched_test_override(true, || {
            run_unified_with_tir(&mut ctx, &next.body, &current, &vars, &registry, Some(&tir))
        })
        .expect("scheduled enumeration should succeed");
        assert_eq!(
            first_guard_sched_test_skips(),
            1,
            "the interleaved disabled candidate must exercise the scheduler"
        );

        let materialize = |diffs: Vec<crate::state::DiffSuccessor>| {
            let values: Vec<Vec<Value>> = diffs
                .into_iter()
                .map(|diff| diff.materialize(&current, &registry).materialize_values())
                .collect();
            values
        };
        assert_eq!(materialize(off), materialize(on));
    }

    #[test]
    fn fallthroughs_and_late_errors_match_when_only_proven_false_strings_skip() {
        let src = r#"
---- MODULE FirstGuardFallthrough ----
EXTENDS Integers
VARIABLES pc, x
Access == "Access"
Advance == "Advance"
Transition(t, from, to) ==
  /\ pc[t] = from
  /\ pc' = [pc EXCEPT ![t] = to]
End(t) ==
  /\ Transition(t, Access, Advance)
  /\ 1 \div 0 = 0
  /\ x' = x + 1
Next == \E a \in {"a"} : End(a)
====
"#;
        let (module, ctx, vars) = setup(src);
        let next = Arc::clone(ctx.get_op("Next").expect("Next should exist"));
        let tir = TirProgram::from_modules(&module, &[]);
        let registry = ctx.var_registry().clone();

        let function = |value: Option<Value>| {
            let mut builder = FuncBuilder::new();
            if let Some(value) = value {
                builder.insert(Value::string("a"), value);
            }
            Value::Func(Rp::new(builder.build()))
        };

        enum Expected {
            NotInDomain,
            TypeError,
            NoSuccessors,
            DivisionByZero,
        }

        let cases = [
            ("missing key", function(None), Expected::NotInDomain, 0),
            ("non-function pc", Value::int(7), Expected::TypeError, 0),
            (
                "non-string result",
                function(Some(Value::Bool(false))),
                Expected::NoSuccessors,
                0,
            ),
            (
                "unequal string",
                function(Some(Value::string("Advance"))),
                Expected::NoSuccessors,
                1,
            ),
            (
                "equal string",
                function(Some(Value::string("Access"))),
                Expected::DivisionByZero,
                0,
            ),
        ];

        for (label, pc, expected, expected_skips) in cases {
            let current = ArrayState::from_values(vec![pc, Value::int(0)]);
            let run = |enabled| {
                let mut run_ctx = ctx.clone();
                let _state_guard = run_ctx.bind_state_env_guard(current.env_ref());
                clear_first_guard_sched_cache();
                reset_first_guard_sched_test_skips();
                let result = with_first_guard_sched_test_override(enabled, || {
                    run_unified_with_tir(
                        &mut run_ctx,
                        &next.body,
                        &current,
                        &vars,
                        &registry,
                        Some(&tir),
                    )
                });
                let skips = first_guard_sched_test_skips();
                let result = result.map(|diffs| {
                    diffs
                        .into_iter()
                        .map(|diff| diff.materialize(&current, &registry).materialize_values())
                        .collect::<Vec<_>>()
                });
                (result, skips)
            };

            let (off, off_skips) = run(false);
            let (on, on_skips) = run(true);
            assert_eq!(off_skips, 0, "{label}: disabled scheduler skipped");
            assert_eq!(on_skips, expected_skips, "{label}: unexpected skip count");

            match expected {
                Expected::NoSuccessors => {
                    assert_eq!(off.expect(label), Vec::<Vec<Value>>::new(), "{label}");
                    assert_eq!(on.expect(label), Vec::<Vec<Value>>::new(), "{label}");
                }
                Expected::NotInDomain => {
                    let off = off.expect_err(label);
                    let on = on.expect_err(label);
                    assert!(
                        matches!(off, EvalError::NotInDomain { .. }),
                        "{label}: {off:?}"
                    );
                    assert!(
                        matches!(on, EvalError::NotInDomain { .. }),
                        "{label}: {on:?}"
                    );
                    assert_eq!(on.to_string(), off.to_string(), "{label}");
                }
                Expected::TypeError => {
                    let off = off.expect_err(label);
                    let on = on.expect_err(label);
                    assert!(
                        matches!(off, EvalError::TypeError { .. }),
                        "{label}: {off:?}"
                    );
                    assert!(matches!(on, EvalError::TypeError { .. }), "{label}: {on:?}");
                    assert_eq!(on.to_string(), off.to_string(), "{label}");
                }
                Expected::DivisionByZero => {
                    let off = off.expect_err(label);
                    let on = on.expect_err(label);
                    assert!(
                        matches!(off, EvalError::DivisionByZero { .. }),
                        "{label}: {off:?}"
                    );
                    assert!(
                        matches!(on, EvalError::DivisionByZero { .. }),
                        "{label}: {on:?}"
                    );
                    assert_eq!(on.to_string(), off.to_string(), "{label}");
                }
            }
        }
    }
}
