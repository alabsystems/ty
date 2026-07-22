// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLC-style action instance splitting.
//!
//! This module is primarily an internal implementation detail, but some helpers are exported for
//! trace validation / diagnostics (attributing a transition to an action instance).
//!
//! TLC pre-splits the Next relation into a list of "action instances" by expanding a maximal
//! prefix of disjunctions and bounded existentials. When it encounters an application of a
//! user-defined operator `Op(args...)`, it only specializes (recurses into) `Op` when every
//! actual argument is constant-level (`argLevel == 0`), evaluating args under `TLCState.Empty`
//! and binding formals in the split-time context.
//!
//! TY does not expose SANY level numbers in the Rust AST, so we approximate TLC's
//! `argLevel == 0` check using dependency tracking (`eval::try_eval_const_level`).

mod split;

use crate::enumerate::enumerate_successors;
use crate::error::EvalResult;
use crate::eval::EvalCtx;
use crate::state::State;
use crate::value::Value;
use std::sync::Arc;
use tla_core::ast::{Expr, OperatorDef};
use tla_core::Spanned;

use split::split_action_instances_rec;

/// A single action instance: a leaf of the split `Next` relation with its
/// name and any witness/formal bindings introduced while specializing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ActionInstance {
    /// Action name (with instance suffix), if one could be attributed.
    pub name: Option<String>,
    /// The specialized action sub-expression for this instance.
    pub expr: Spanned<tla_core::ast::Expr>,
    /// Bound-variable witness bindings from split bounded `\E` enumeration.
    pub bindings: Vec<(Arc<str>, Value)>,
    /// Operator formal-parameter bindings introduced when specializing `Op(args...)`.
    pub formal_bindings: Vec<(Arc<str>, Value)>,
    /// Complete oldest-to-newest lexical binding chain at the split leaf,
    /// before dispatch-key alias pruning. Internal optimization certificates
    /// use this to preserve exact shadowing semantics.
    #[doc(hidden)]
    pub(crate) complete_bindings: Vec<(Arc<str>, Value)>,
}

#[derive(Debug, Clone, Default)]
struct SplitCtx {
    pub(self) action_name: Option<String>,
    pub(self) instance_prefix: Option<String>,
    pub(self) formal_bindings: Vec<(Arc<str>, Value)>,
    pub(self) formal_binding_exclusions: Vec<(Arc<str>, Value)>,
    pub(self) wrappers: Vec<SplitWrapper>,
    pub(self) op_stack: Vec<String>,
}

#[derive(Debug, Clone)]
enum SplitWrapper {
    Let(Vec<OperatorDef>),
    Guard(Spanned<Expr>),
}

/// Splits an action expression into its TLC-style action instances.
///
/// Expands a maximal prefix of disjunctions and bounded existentials and
/// specializes operator applications whose arguments are all constant-level.
///
/// # Errors
/// Returns an [`EvalError`](crate::error::EvalError) if evaluating a const-level
/// domain or argument during splitting fails.
pub fn split_action_instances(
    ctx: &EvalCtx,
    expr: &Spanned<tla_core::ast::Expr>,
) -> EvalResult<Vec<ActionInstance>> {
    let mut out = Vec::new();
    // Action splitting is a preprocessing step and must not depend on the current/next state.
    let split_ctx = ctx.without_state_and_next();
    split_action_instances_rec(&split_ctx, expr, &SplitCtx::default(), &mut out)?;
    Ok(out)
}

/// An action instance paired with the successor states it generated.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ActionInstanceSuccessors {
    /// The action instance the successors are attributed to.
    pub instance: ActionInstance,
    /// Successor states produced by this instance.
    pub successors: Vec<State>,
}

/// Enumerate successors separately for each split action instance.
///
/// This is intended for trace validation / diagnostics, where we want to attribute a transition to
/// an action instance (name + bindings). It is not used on the hot path for model checking.
///
/// Limitations (intentionally mirrors the current splitter):
/// - Only bounded `\E` with const-level enumerable domains are split into per-witness instances.
/// - Only simple bound variables (no destructuring patterns) are split.
/// - Operator applications are specialized only when every actual argument is const-level under
///   the split-time binding context (TLC `argLevel == 0` analogue).
pub fn enumerate_successors_by_action_instance(
    ctx: &mut EvalCtx,
    next_def: &OperatorDef,
    state: &State,
    vars: &[Arc<str>],
) -> EvalResult<Vec<ActionInstanceSuccessors>> {
    let instances = split_action_instances(ctx, &next_def.body)?;
    let mut out = Vec::with_capacity(instances.len());

    for instance in instances {
        // RAII guard restores env + next_state on drop (Part of #2738)
        let _scope_guard = ctx.scope_guard_with_next_state();
        let mark = ctx.mark_stack();
        for (k, v) in &instance.bindings {
            ctx.push_binding(Arc::clone(k), v.clone());
        }
        for (k, v) in &instance.formal_bindings {
            ctx.push_binding(Arc::clone(k), v.clone());
        }

        let op_def = OperatorDef {
            name: next_def.name.clone(),
            params: next_def.params.clone(),
            body: instance.expr.clone(),
            local: next_def.local,
            contains_prime: next_def.contains_prime,
            guards_depend_on_prime: next_def.guards_depend_on_prime,
            has_primed_param: next_def.has_primed_param,
            is_recursive: next_def.is_recursive,
            self_call_count: next_def.self_call_count,
        };

        let successors_res = enumerate_successors(ctx, &op_def, state, vars);
        ctx.pop_to_mark(&mark);
        drop(_scope_guard);
        let successors = successors_res?;

        out.push(ActionInstanceSuccessors {
            instance,
            successors,
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests;
