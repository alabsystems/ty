// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! State enumeration for model checking
//!
//! This module implements constraint extraction and state enumeration for:
//! 1. Init predicates - extracting equality constraints to generate initial states
//! 2. Next relations - handling primed variables to generate successor states
//!
//! # Approach
//!
//! For Init predicates:
//! - Parse conjunctions to extract individual constraints
//! - Handle equality constraints: `x = value`
//! - Handle membership constraints: `x \in S`
//! - Enumerate all satisfying states
//!
//! For Next relations:
//! - Bind current state variables
//! - Find primed variable assignments: `x' = expr`
//! - Handle UNCHANGED: equivalent to `x' = x`
//! - Handle disjunctions (multiple actions)
//! - Enumerate all successor states

use crate::error::EvalError;
use crate::eval::{apply_substitutions, compose_substitutions, EvalCtx, OpEnv};
use crate::state::State;
use crate::Value;
use std::cell::Cell;
use std::sync::{Arc, OnceLock};
use tla_core::ast::Expr;
use tla_core::{Span, Spanned};

#[cfg(test)]
use crate::state::ArrayState;

thread_local! {
    static ENABLED_EARLY_EXIT: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn enabled_early_exit() -> bool {
    ENABLED_EARLY_EXIT.with(std::cell::Cell::get)
}

// ─── Per-state successor materialization cap (fail-closed, audit finding #12) ──
//
// The batch successor path collects the ENTIRE successor set of a single state
// into one `Vec<DiffSuccessor>` before any of it is processed. A pathological
// or misconfigured action (e.g. `x' \in 1..HUGE /\ y' \in 1..HUGE`, an
// unbounded CONSTANT, or an accidental Cartesian blow-up) can make that Vec grow
// without bound and OOM-kill the whole checker — a hard crash, not a verdict.
//
// To stay fail-closed we cap how many successors a *single* state may
// materialize on the batch path. When the cap is exceeded, enumeration stops
// (via the existing `DiffSink` `Break` protocol) and the engine returns
// `EvalError::SetTooLarge`, which already propagates up as a graceful
// `CheckError` / `BfsIterOutcome::Terminate` (no panic, no OOM, no wrong
// verdict). The cap is deliberately large so it never trips on legitimate specs;
// it is purely a runaway-memory guard.

/// The default cap is deliberately large so it never trips on legitimate specs;
/// it is purely a runaway-memory guard. Override per-checker via
/// [`Config::per_state_successor_cap`], or process-wide via the env var
/// `TY_PER_STATE_SUCCESSOR_CAP` (`0` disables the cap entirely).
///
/// The effective cap is resolved once, when the checker builds its evaluation
/// context, and stored on the per-context [`tla_eval::SharedCtx`]. The batch
/// successor engine reads it from there ([`tla_eval::EvalCtx::shared`]) — there
/// is no process-global cap state, so a test that injects a tiny cap on its own
/// `Config` can never leak into a concurrently-running checker.
///
/// Resolves the effective per-state successor cap for a model `Config`.
///
/// Precedence: an explicit `Config` override (`Some(_)`) wins; otherwise the
/// process env var `TY_PER_STATE_SUCCESSOR_CAP` is consulted (cached once), and
/// failing that the built-in [`DEFAULT_PER_STATE_SUCCESSOR_CAP`]. `Some(0)` from
/// the env var — like `Config` override `Some(None)` — disables the cap.
/// Returns `None` when the cap is disabled.
pub(crate) fn resolve_per_state_successor_cap(config: &crate::Config) -> Option<usize> {
    if let Some(override_value) = config.per_state_successor_cap {
        return override_value;
    }

    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    *CAP.get_or_init(|| {
        let from_env = std::env::var("TY_PER_STATE_SUCCESSOR_CAP")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok());
        match from_env {
            Some(0) => None, // Explicit 0 disables the cap.
            Some(n) => Some(n),
            None => Some(tla_eval::DEFAULT_PER_STATE_SUCCESSOR_CAP),
        }
    })
}

/// Resolve the per-state successor cap for `config` and store it on `ctx`'s
/// shared context, so the batch successor engine reads a per-checker value
/// instead of any process-global state. Call this once per evaluation context,
/// at context-build time (after the operator/ident precompute pass), on every
/// BFS exploration path (sequential and parallel-worker).
///
/// Cheap: at build time the shared context is uniquely owned, so `make_mut`
/// does not clone.
pub(crate) fn apply_per_state_successor_cap(ctx: &mut tla_eval::EvalCtx, config: &crate::Config) {
    let cap = resolve_per_state_successor_cap(config);
    Arc::make_mut(ctx.shared_arc_mut()).per_state_successor_cap = cap;
}

#[cfg(test)]
mod build;
#[cfg(test)]
pub use build::build_successor_array_states_with_ctx;

#[cfg(test)]
mod emitter;

mod build_diff;
#[cfg(test)]
pub use build_diff::{
    build_successor_diffs_from_array, build_successor_diffs_from_array_filtered,
    build_successor_diffs_from_array_into, build_successor_diffs_with_deferred_filtered,
};

mod action_successors;
pub(crate) use action_successors::enumerate_action_successors;
pub(crate) use action_successors::{
    enumerate_action_successors_witness_capped, EnabledEnumOutcome, SubscriptWatch,
};

mod constraint;
pub(crate) use constraint::{find_unconstrained_vars, find_values_for_var, Constraint, InitDomain};

mod init_constraints;
#[cfg(test)]
pub(crate) use init_constraints::count_expr_nodes;
pub(crate) use init_constraints::{extract_conjunction_remainder, extract_init_constraints};

mod expr_analysis;
pub(crate) use expr_analysis::collect_state_var_refs;
pub(crate) use expr_analysis::{clear_expr_analysis_caches, expr_contains_any_prime};
#[cfg(test)]
use expr_analysis::{expr_contains_exists, flatten_and};
use expr_analysis::{
    expr_is_action_level, expr_references_primed_vars, expr_references_state_vars,
    get_primed_var_refs, get_primed_var_refs_with_ctx, is_guard_expression,
    is_operator_reference_guard_unsafe,
};

mod first_guard_sched;

mod expand_ops;
pub(crate) use expand_ops::expand_operators;
pub(crate) use expand_ops::expand_operators_with_primes;

mod symbolic_assignments;
#[cfg(test)]
use symbolic_assignments::toposort::topological_sort_assignments;
#[cfg(test)]
use symbolic_assignments::{evaluate_symbolic_assignments, extract_symbolic_assignments};

mod const_domain_cache;
pub(crate) use const_domain_cache::clear_const_domain_cache;
mod complete_action_filter;
mod subst_cache;
pub(crate) use subst_cache::clear_enum_subst_cache;
pub(crate) mod subset_constrained;
pub(crate) mod subset_profile;
mod unified;
mod unified_classify;
mod unified_conjuncts;
mod unified_dispatch;
pub(crate) use unified_dispatch::clear_state_independent_branch_caches;
mod unified_emit;
mod unified_exists;
mod unified_fast_path;
mod unified_module_ref;
mod unified_scope;
mod unified_types;
pub(crate) use unified_types::{ClosureSink, DiffSink};
pub(crate) mod tir_leaf;

mod init_enumerate;
#[cfg(test)]
#[allow(unused_imports)]
use init_enumerate::compute_values_fingerprint;
pub(crate) use init_enumerate::{
    enumerate_constraints_to_bulk, enumerate_constraints_to_bulk_with_stats,
    enumerate_constraints_to_bulk_with_stats_filter_error,
    enumerate_states_from_constraint_branches, enumerate_states_from_constraint_branches_probed,
    eval_filter_expr, BulkConstraintEnumerationError,
    BulkConstraintEnumerationStats,
};

// Part of #3461: local_scope is only used by build_tests, gate to suppress dead_code warning.
#[cfg(test)]
mod local_scope;

mod successor_api;
mod successor_engine;
#[cfg(test)]
pub(crate) use successor_api::enumerate_successors_array;
pub(crate) use successor_api::{
    enumerate_successors, enumerate_successors_array_as_diffs,
    enumerate_successors_array_as_diffs_body, enumerate_successors_array_as_diffs_body_with_cap,
    enumerate_successors_array_as_diffs_into,
    enumerate_successors_array_as_diffs_into_with_current_values,
    enumerate_successors_array_as_diffs_into_with_pc_hoist,
    enumerate_successors_array_as_diffs_with_current_values,
    enumerate_successors_array_body_with_tir, enumerate_successors_array_with_tir,
    enumerate_successors_body,
};
#[cfg(test)]
pub(crate) use successor_engine::successor_engine_test_helpers;

mod value_to_expr;
pub(crate) use value_to_expr::try_value_to_expr;
pub(crate) use value_to_expr::value_to_expr;

mod action_validation;
mod guard_check;
mod unchanged_extraction;
use guard_check::check_and_guards;

debug_flag!(pub(crate) debug_enum, "TY_DEBUG_ENUM");

mod error_classify;
pub(super) use error_classify::is_action_level_error;
pub(crate) use error_classify::{
    classify_iter_error_for_speculative_path, is_disabled_action_error,
    is_speculative_eval_fallback, IterDomainAction,
};

pub(super) fn case_guard_error(err: EvalError, span: Span) -> EvalError {
    if matches!(err, EvalError::ExitRequested { .. }) {
        err
    } else {
        EvalError::CaseGuardError {
            source: Box::new(err),
            span: Some(span),
        }
    }
}

pub(super) fn is_let_lazy_safe_error(err: &EvalError) -> bool {
    !matches!(err, EvalError::ExitRequested { .. })
}

#[cfg(debug_assertions)]
pub(super) fn emit_debug_line(enabled: bool, args: std::fmt::Arguments<'_>) {
    if enabled {
        eprintln!("{args}");
    }
}

#[cfg(not(debug_assertions))]
pub(super) fn emit_debug_line(_: bool, _: std::fmt::Arguments<'_>) {}

feature_flag!(profile_enum_detail, "TY_PROFILE_ENUM_DETAIL");

pub(super) fn and_guard_precheck() -> bool {
    // The set-once process-global env snapshot (installed only by the CLI) wins; it mirrors
    // `feature_flag!` (any value present). Library/test callers never install it and fall
    // through to the `OnceLock`-cached env path below.
    tla_backend::global_overlay()
        .map(tla_backend::EngineEnvOverlay::and_guard_is_set)
        .unwrap_or_else(|| {
            static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            crate::debug_env::env_flag_is_set(&FLAG, "TY_AND_GUARD_PRECHECK")
        })
}

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

static PROF_ASSIGN_US: AtomicU64 = AtomicU64::new(0);

pub(crate) fn print_enum_profile_stats() {
    if !profile_enum_detail() {
        return;
    }
    let assign = PROF_ASSIGN_US.swap(0, AtomicOrdering::Relaxed);
    if assign == 0 {
        return;
    }
    eprintln!("=== Enumeration Detail Profile ===");
    eprintln!("  Assignment eval: {:>8.3}s", assign as f64 / 1_000_000.0);
}

debug_flag!(debug_extract, "TY_DEBUG_EXTRACT");
debug_flag!(debug_toposort, "TY_DEBUG_TOPOSORT");
debug_flag!(pub(super) debug_stage, "TY_DEBUG_STAGE");
debug_flag!(pub(super) debug_guards, "TY_DEBUG_GUARDS");
debug_flag!(pub(super) debug_enum_trace, "TY_DEBUG_ENUM_TRACE");

#[derive(Debug, Clone)]
pub enum PrimedAssignment {
    Assign(Arc<str>, Value),
    #[allow(dead_code)] // Constructed in test-only builders; production retains the match shape.
    Unchanged(Arc<str>),
    InSet(Arc<str>, Vec<Value>),
    #[allow(dead_code)] // Fields read only in test-gated build.rs; production matches with (_, _)
    DeferredExpr(Arc<str>, Spanned<Expr>),
}

type CapturedBindings = Vec<(Arc<str>, Value)>;

#[derive(Debug, Clone)]
pub(super) enum SymbolicAssignment {
    Expr(Arc<str>, Spanned<Expr>, CapturedBindings),
    Value(Arc<str>, Value),
    Unchanged(Arc<str>),
    InSet(Arc<str>, Spanned<Expr>, CapturedBindings),
}

impl SymbolicAssignment {
    fn var_name(&self) -> &Arc<str> {
        match self {
            SymbolicAssignment::Value(n, _)
            | SymbolicAssignment::Expr(n, _, _)
            | SymbolicAssignment::Unchanged(n)
            | SymbolicAssignment::InSet(n, _, _) => n,
        }
    }
}

// Part of #188: Removed all_vars_assigned() - replaced with O(1) bitmap check
// using assigned_mask == full_mask in enumerate_and_conjuncts_as_diffs_opt

// eval_enabled removed — Part of #3004: All ENABLED evaluation now uses
// enabled::eval_enabled_cp (constraint propagation) instead of enumerate_unified.
// The CP approach avoids ArrayState construction, fingerprint computation, and
// undo stack allocation. See crates/tla-check/src/enabled/ for the replacement.
