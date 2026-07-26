// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Static analysis of ACTION_CONSTRAINT expressions.
//!
//! Pre-analyzes each ACTION_CONSTRAINT at setup time to determine which
//! state variables it references (both primed and unprimed). During BFS,
//! the checker uses this analysis to skip constraint evaluation when none
//! of the referenced variables changed between the current and successor
//! states.
//!
//! This optimization is sound because:
//! - If an ACTION_CONSTRAINT references variables x, y (unprimed) and x', y'
//!   (primed), and none of {x, y} changed between states, then the unprimed
//!   references see the same values and the primed references also see the
//!   same values (since the successor variable values are identical to the
//!   current state for those slots). The constraint will evaluate to the
//!   same result as for a stuttering step on those variables.
//!
//! - More precisely: if the set of changed variables is disjoint from the
//!   set of variables referenced by the constraint (in either primed or
//!   unprimed form), the constraint evaluates identically to the previous
//!   evaluation with the same variable values. Since ACTION_CONSTRAINTs are
//!   pure boolean functions of state variables, the result is deterministic.
//!
//! However, we must be conservative: if the constraint references operators
//! that we cannot fully expand (e.g., recursive operators, or operators with
//! side effects like TLCGet), we mark the constraint as non-skippable.

use rustc_hash::FxHashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::por::{extract_dependencies_ast_expr, ActionDependencies};
use crate::state::ArrayState;
use crate::var_index::VarIndex;
use tla_core::ast::Expr;
use tla_core::Spanned;

use crate::enumerate::expand_operators_with_primes;
use tla_eval::EvalCtx;

/// Pre-computed analysis for a single ACTION_CONSTRAINT.
// Fields `name`, `reads`, `writes` are retained for diagnostics and tests.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ConstraintVarDeps {
    /// The constraint operator name.
    pub(crate) name: String,
    /// VarIndex values that appear unprimed (current state reads).
    pub(crate) reads: FxHashSet<VarIndex>,
    /// VarIndex values that appear primed (next state reads, recorded as writes
    /// by the POR dependency extractor).
    pub(crate) writes: FxHashSet<VarIndex>,
    /// Union of reads and writes as a sorted Vec for fast iteration.
    /// This is the set of variable indices that must be checked for changes.
    pub(crate) all_vars: Vec<VarIndex>,
    /// Whether the predicate references any primed/next-state variable,
    /// including identity writes normalized to `UNCHANGED` by POR analysis.
    pub(crate) references_next_state: bool,
    /// Whether every lowered state-variable reference names exactly the same
    /// registry slot by string, numeric index, and interned `NameId`.
    ///
    /// The evaluator repairs stale/mismatched `StateVar` metadata at runtime,
    /// while POR dependency extraction consumes the embedded numeric index.
    /// Exact-verdict reuse therefore requires this stronger certificate before
    /// trusting the extracted projection.
    pub(crate) state_var_slots_exact: bool,
    /// If true, the constraint references constructs we cannot statically
    /// analyze (e.g., TLCGet, recursive ops), so it must always be evaluated.
    pub(crate) must_always_eval: bool,
}

/// Pre-computed analysis for all ACTION_CONSTRAINTs in a spec.
#[derive(Debug)]
pub(crate) struct ActionConstraintAnalysis {
    /// Per-constraint dependency information.
    pub(crate) constraints: Vec<ConstraintVarDeps>,
    /// Metrics: how many constraint evaluations were skipped.
    pub(crate) skipped: AtomicU64,
    /// Metrics: how many constraint evaluations were performed.
    pub(crate) evaluated: AtomicU64,
}

impl ActionConstraintAnalysis {
    /// Build analysis for all configured ACTION_CONSTRAINTs.
    ///
    /// Looks up each canonical raw operator name, expands the body through
    /// operator references, and extracts variable dependencies.
    pub(crate) fn build(ctx: &EvalCtx, constraint_names: &[String]) -> Self {
        let mut constraints = Vec::with_capacity(constraint_names.len());
        for name in constraint_names {
            constraints.push(analyze_one_constraint(ctx, name));
        }
        ActionConstraintAnalysis {
            constraints,
            skipped: AtomicU64::new(0),
            evaluated: AtomicU64::new(0),
        }
    }

    /// Whether these predicates are safe to reuse after an exact seen-state
    /// payload match.
    ///
    /// This is intentionally narrower than ordinary dependency extraction:
    /// every predicate must be analyzable, context-free, and read only the
    /// unprimed state. The caller still has to prove that the candidate payload
    /// exactly matches a previously admitted state and separately exclude
    /// edge-dependent `ACTION_CONSTRAINT`s.
    #[must_use]
    pub(crate) fn supports_exact_seen_state_reuse(&self) -> bool {
        !self.constraints.is_empty()
            && self.constraints.iter().all(|constraint| {
                !constraint.must_always_eval
                    && !constraint.references_next_state
                    && constraint.state_var_slots_exact
            })
    }

    /// Exact state-variable projection for one pure, unprimed predicate.
    ///
    /// Callers may reuse a successful verdict only after comparing the full
    /// values in every returned slot. `None` is fail-closed for unresolved,
    /// context-dependent, side-effecting, or next-state predicates.
    #[must_use]
    pub(crate) fn exact_reuse_projection(&self, index: usize) -> Option<&[VarIndex]> {
        let constraint = self.constraints.get(index)?;
        if constraint.must_always_eval
            || constraint.references_next_state
            || !constraint.state_var_slots_exact
        {
            return None;
        }
        Some(&constraint.all_vars)
    }

    /// Check if a constraint can be skipped for a given transition.
    ///
    /// Returns `true` if none of the constraint's referenced variables changed
    /// between `current` and `succ`, meaning the constraint result is the same
    /// as for a stuttering step and can be skipped.
    ///
    /// DISABLED: The skip optimization is unsound for constraints that fail on
    /// stuttering steps. For example, `x' = x + 1` requires the successor to
    /// differ from the current state — when x' == x (no change), the constraint
    /// should FAIL, but the skip optimization assumed it passes. Similarly for
    /// `x' > x`, `x' /= x`, etc. The optimization would need to cache the
    /// result of evaluating the constraint on a stuttering step and only skip
    /// when that cached result is `true`. Until that is implemented, always
    /// evaluate the constraint.
    #[inline]
    pub(crate) fn can_skip_constraint(
        &self,
        _idx: usize,
        _current: &ArrayState,
        _succ: &ArrayState,
    ) -> bool {
        false
    }

    /// Record a skip event.
    #[inline]
    pub(crate) fn record_skip(&self) {
        self.skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an evaluation event.
    #[inline]
    pub(crate) fn record_eval(&self) {
        self.evaluated.fetch_add(1, Ordering::Relaxed);
    }

    /// Get skip count.
    #[allow(dead_code)] // Used in tests; retained for diagnostics
    pub(crate) fn skip_count(&self) -> u64 {
        self.skipped.load(Ordering::Relaxed)
    }

    /// Get evaluation count.
    #[allow(dead_code)] // Retained for diagnostics
    pub(crate) fn eval_count(&self) -> u64 {
        self.evaluated.load(Ordering::Relaxed)
    }

    /// Returns true if there are no analyzable constraints (nothing to optimize).
    #[allow(dead_code)] // Retained for diagnostics
    pub(crate) fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

/// Analyze a single constraint operator to extract variable dependencies.
fn analyze_one_constraint(ctx: &EvalCtx, name: &str) -> ConstraintVarDeps {
    // Backends disagree today on configured root replacements: canonical AST
    // invariant/state/action-constraint leaves enter through raw eval_op(name),
    // while implicit-default TIR resolves the name before dispatch. No single
    // root projection can certify both paths, so shared exact reuse must reject
    // the replacement entirely.
    if ctx.resolve_op_name(name) != name {
        return ConstraintVarDeps {
            name: name.to_string(),
            reads: FxHashSet::default(),
            writes: FxHashSet::default(),
            all_vars: Vec::new(),
            references_next_state: false,
            state_var_slots_exact: false,
            must_always_eval: true,
        };
    }

    // Canonical invariant/state/action-constraint evaluation enters through
    // `EvalCtx::eval_op(name)`, whose root lookup is deliberately raw. Config
    // replacements still apply to calls inside this body during expansion.
    let def = ctx.get_op(name);

    let Some(def) = def else {
        // Cannot resolve operator — must always evaluate.
        return ConstraintVarDeps {
            name: name.to_string(),
            reads: FxHashSet::default(),
            writes: FxHashSet::default(),
            all_vars: Vec::new(),
            references_next_state: false,
            state_var_slots_exact: false,
            must_always_eval: true,
        };
    };

    analyze_constraint_expr(ctx, name, &def.body)
}

/// Analyze an arbitrary expression with the same fail-closed rules used for a
/// configured constraint operator.
///
/// This is expression-scoped: callers that decompose a larger predicate must
/// separately prove that isolated evaluation preserves source control flow and
/// scope semantics.
fn analyze_constraint_expr(ctx: &EvalCtx, name: &str, expr: &Spanned<Expr>) -> ConstraintVarDeps {
    // Expand operator references so we can see through wrapper ops
    // to the underlying state variable references.
    let expanded = expand_operators_with_primes(ctx, expr);

    // The evaluator may repair stale StateVar metadata by name or NameId,
    // whereas dependency extraction trusts the embedded slot. Exact reuse
    // requires both source and expanded trees to agree on all three.
    let state_var_slots_exact =
        state_var_slots_are_exact(ctx, expr) && state_var_slots_are_exact(ctx, &expanded);

    // Check for constructs that prevent static analysis.
    let mut must_always_eval = contains_non_analyzable(&expanded.node)
        || contains_non_builtin_seq_reference(ctx, expr)
        || contains_non_builtin_seq_reference(ctx, &expanded)
        || contains_shadowed_builtin_constant_reference(ctx, expr)
        || contains_shadowed_builtin_constant_reference(ctx, &expanded)
        || contains_unsafe_config_value_reference(ctx, &expanded);

    // Extract variable dependencies using the POR infrastructure.
    let mut deps = ActionDependencies::new();
    extract_dependencies_ast_expr(ctx, &expanded.node, &mut deps);

    // FAIL CLOSED (POR hole #3): if extraction hit residue whose reads are
    // unknowable (un-inlined operators, module refs, ...), the read set may
    // under-approximate — skipping re-evaluation on "unchanged" vars would
    // be unsound. Always evaluate such constraints.
    if deps.opaque {
        must_always_eval = true;
    }

    let mut all_vars: FxHashSet<VarIndex> = FxHashSet::default();
    all_vars.extend(&deps.reads);
    all_vars.extend(&deps.writes);
    let mut all_vars_sorted: Vec<VarIndex> = all_vars.into_iter().collect();
    all_vars_sorted.sort();
    let references_next_state = !deps.writes.is_empty() || !deps.unchanged.is_empty();

    ConstraintVarDeps {
        name: name.to_string(),
        reads: deps.reads,
        writes: deps.writes,
        all_vars: all_vars_sorted,
        references_next_state,
        state_var_slots_exact,
        must_always_eval,
    }
}

/// Exact state-variable projection for one arbitrary pure, unprimed
/// expression. The returned slots are only a static certificate; cache hits
/// must still compare every projected `Value` exactly.
pub(crate) fn exact_reuse_projection_for_expr(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
) -> Option<Vec<VarIndex>> {
    let analysis = analyze_constraint_expr(ctx, "<expression>", expr);
    if analysis.must_always_eval
        || analysis.references_next_state
        || !analysis.state_var_slots_exact
    {
        return None;
    }
    Some(analysis.all_vars)
}

/// Whether evaluating an action body is safe to replay after splitting a
/// quantified/disjunctive `Next` into per-action batches.
///
/// Splitting can reorder evaluations even when it preserves the relational
/// successor multiset. AUTO routing therefore admits only expressions whose
/// expanded body is context-free, deterministic, free of side effects, and
/// whose state-variable slots are exact. Primed references are expected in an
/// action and are intentionally allowed. Pure recursive definitions are walked
/// once with call-graph cycle cutting; unresolved/higher-order calls,
/// `TLCGet`/`TLCSet`, randomization, unsafe config values, and future unknown
/// builtins all fail closed through the shared dependency analyzer.
#[cfg(test)]
pub(crate) fn action_expr_is_replay_safe(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    let (raw_slots, expanded_slots, raw_context, expanded_context) =
        action_expr_replay_safety_components(ctx, expr);
    raw_slots && expanded_slots && raw_context && expanded_context
}

/// Component verdicts for the router's opt-in admission diagnostic.
pub(crate) fn action_expr_replay_safety_components(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
) -> (bool, bool, bool, bool) {
    let expanded = expand_operators_with_primes(ctx, expr);
    (
        state_var_slots_are_exact(ctx, expr),
        state_var_slots_are_exact(ctx, &expanded),
        crate::por::replay_expr_is_context_free(ctx, &expr.node),
        crate::por::replay_expr_is_context_free(ctx, &expanded.node),
    )
}

/// First raw and expanded context-safety rejection reasons, for opt-in router
/// diagnostics after admission has already failed.
pub(crate) fn action_expr_replay_context_rejections(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
) -> (Option<String>, Option<String>) {
    let expanded = expand_operators_with_primes(ctx, expr);
    (
        crate::por::replay_expr_context_rejection(ctx, &expr.node),
        crate::por::replay_expr_context_rejection(ctx, &expanded.node),
    )
}

/// Require every lowered StateVar(name, idx, name_id) to identify one slot by
/// all three components. This mirrors the evaluator fast-path condition and
/// prevents a repaired runtime slot from disagreeing with the cached plan.
fn state_var_slots_are_exact(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    use crate::check::{expr_contains, ScanDecision};
    use tla_core::NameId;

    let registry = ctx.var_registry();
    !expr_contains(&expr.node, &|node| match node {
        Expr::StateVar(name, idx, name_id) => {
            let raw_idx = VarIndex(*idx);
            let exact = raw_idx.as_usize() < registry.len()
                && *name_id != NameId::INVALID
                && registry.name_id_at(raw_idx) == *name_id
                && registry.get(name) == Some(raw_idx);
            if exact {
                ScanDecision::Continue
            } else {
                ScanDecision::Found
            }
        }
        _ => ScanDecision::Continue,
    })
}

/// Positive list of residual name-dispatched builtins whose result is a pure
/// function of their fully walked arguments. Unknown future builtins fail
/// closed because exact verdict reuse needs an allowlist, not an impurity list.
fn is_exact_reuse_pure_named_builtin(name: &str, arity: usize) -> bool {
    crate::enumerate::is_replay_stable_named_builtin(name, arity)
}

/// `Seq(S)` is pure when it reaches the genuine builtin, but the spelling can
/// be redirected by config or shadowed lexically. Reject every such shadow
/// rather than using the spelling itself as authority.
fn contains_non_builtin_seq_reference(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    use crate::check::{expr_contains, ScanDecision};

    let global_seq_is_shadowed = ctx.name_in_local_scope("Seq")
        || ctx.resolve_op_name("Seq") != "Seq"
        || ctx.is_config_constant("Seq")
        || ctx.get_op("Seq").is_some();
    let bound_var_is_seq = |bound: &tla_core::ast::BoundVar| {
        tla_core::single_bound_var_names(bound)
            .iter()
            .any(|name| name == "Seq")
    };

    expr_contains(&expr.node, &|node| match node {
        Expr::Let(defs, _)
            if defs.iter().any(|def| {
                def.name.node.as_str() == "Seq"
                    || def
                        .params
                        .iter()
                        .any(|param| param.name.node.as_str() == "Seq")
            }) =>
        {
            ScanDecision::Found
        }
        Expr::Lambda(params, _) if params.iter().any(|param| param.node.as_str() == "Seq") => {
            ScanDecision::Found
        }
        Expr::Forall(bounds, _) | Expr::Exists(bounds, _) | Expr::FuncDef(bounds, _)
            if bounds.iter().any(|bound| bound_var_is_seq(bound)) =>
        {
            ScanDecision::Found
        }
        Expr::SetBuilder(_, bounds) if bounds.iter().any(|bound| bound_var_is_seq(bound)) => {
            ScanDecision::Found
        }
        Expr::Choose(bound, _) | Expr::SetFilter(bound, _) if bound_var_is_seq(bound) => {
            ScanDecision::Found
        }
        Expr::SubstIn(substitutions, _) | Expr::InstanceExpr(_, substitutions)
            if substitutions
                .iter()
                .any(|substitution| substitution.from.node.as_str() == "Seq") =>
        {
            ScanDecision::Found
        }
        Expr::Apply(op, _)
            if global_seq_is_shadowed
                && matches!(&op.node, Expr::Ident(name, _) if name == "Seq") =>
        {
            ScanDecision::Found
        }
        _ => ScanDecision::Continue,
    })
}

/// Reject builtin-constant spellings when runtime lookup can reach a user,
/// config, or local value with the same name. The dependency walker otherwise
/// treats these names as state-independent before consulting global operators.
fn contains_shadowed_builtin_constant_reference(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    use crate::check::{expr_contains, ScanDecision};

    let is_builtin_constant = |name: &str| {
        matches!(
            name,
            "Nat" | "Int" | "Real" | "Infinity" | "BOOLEAN" | "STRING"
        )
    };
    expr_contains(&expr.node, &|node| match node {
        Expr::Ident(name, _)
            if is_builtin_constant(name)
                && (ctx.name_in_local_scope(name)
                    || ctx.is_config_constant(name)
                    || ctx.resolve_op_name(name) != name
                    || ctx.get_op(name).is_some()) =>
        {
            ScanDecision::Found
        }
        _ => ScanDecision::Continue,
    })
}

/// The legacy dependency walker treats every config/precomputed value as
/// state-independent. Exact caching strengthens that rule: only concrete data
/// is benign; closures, lazy functions, and set predicates may execute against
/// the live state when passed to a higher-order builtin.
fn contains_unsafe_config_value_reference(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    use crate::check::{expr_contains, ScanDecision};
    use tla_core::name_intern::lookup_name_id;

    let value_is_concrete = |name: &str| {
        ctx.lookup(name)
            .map(|value| value.is_concrete_data())
            .or_else(|| {
                lookup_name_id(name)
                    .and_then(|id| ctx.precomputed_constants().get(&id))
                    .map(|value| value.is_concrete_data())
            })
            .unwrap_or(false)
    };
    let precomputed_is_non_concrete = |name: &str| {
        lookup_name_id(name)
            .and_then(|id| ctx.precomputed_constants().get(&id))
            .is_some_and(|value| !value.is_concrete_data())
    };

    expr_contains(&expr.node, &|node| match node {
        Expr::Ident(name, _) | Expr::OpRef(name)
            if ctx.op_replacements().contains_key(name)
                || (ctx.is_config_constant(name) && !value_is_concrete(name))
                || precomputed_is_non_concrete(name) =>
        {
            ScanDecision::Found
        }
        _ => ScanDecision::Continue,
    })
}

/// Check whether an expanded expression still contains executable application
/// residue that is not positively certified for exact verdict reuse.
///
/// Expansion preserves calls that runtime overrides with Rust builtins. A
/// residual named call is admitted only by the pure allowlist above; unknown,
/// context-dependent, higher-order, and future builtins fail closed.
fn contains_non_analyzable(expr: &Expr) -> bool {
    use crate::check::{expr_contains, ScanDecision};

    expr_contains(expr, &|e| match e {
        Expr::Apply(op, args) => match &op.node {
            Expr::OpRef(_) => ScanDecision::Continue,
            Expr::Ident(name, _) if is_exact_reuse_pure_named_builtin(name, args.len()) => {
                ScanDecision::Continue
            }
            _ => ScanDecision::Found,
        },
        _ => ScanDecision::Continue,
    })
}

/// Optimized ACTION_CONSTRAINT evaluation that skips constraints when
/// referenced variables haven't changed.
///
/// This is the drop-in replacement for `check_action_constraints_array`
/// that adds the skip optimization. Falls back to full evaluation when
/// the analysis indicates a constraint cannot be skipped.
pub(crate) fn check_action_constraints_with_analysis(
    ctx: &mut EvalCtx,
    action_constraints: &[String],
    current: &ArrayState,
    succ: &ArrayState,
    analysis: &ActionConstraintAnalysis,
) -> Result<bool, crate::check::CheckError> {
    if action_constraints.is_empty() {
        return Ok(true);
    }

    debug_assert_eq!(action_constraints.len(), analysis.constraints.len());

    // Fast path: check if ALL constraints can be skipped (no referenced vars changed).
    // This avoids binding state/next-state guards entirely.
    let all_skippable =
        (0..action_constraints.len()).all(|i| analysis.can_skip_constraint(i, current, succ));

    if all_skippable {
        analysis
            .skipped
            .fetch_add(action_constraints.len() as u64, Ordering::Relaxed);
        return Ok(true);
    }

    // At least one constraint needs evaluation. Bind both states and
    // evaluate only the constraints that reference changed variables.
    let _state_guard = ctx.bind_state_env_guard(current.env_ref());
    let _next_guard = ctx.bind_next_state_env_guard(succ.env_ref());
    crate::eval::clear_for_bound_state_eval_scope(ctx);

    for (i, constraint_name) in action_constraints.iter().enumerate() {
        if analysis.can_skip_constraint(i, current, succ) {
            analysis.record_skip();
            continue;
        }
        analysis.record_eval();
        if !super::eval_constraint_bool(ctx, constraint_name)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker_setup::{setup_checker_modules, SetupOptions};
    use crate::config::Config;
    use crate::test_support::parse_module;
    use crate::Value;

    fn make_test_ctx_and_analysis(
        src: &str,
        action_constraints: &[&str],
    ) -> (EvalCtx, Vec<String>, ActionConstraintAnalysis) {
        let module = parse_module(src);
        let constraint_names: Vec<String> =
            action_constraints.iter().map(|s| s.to_string()).collect();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            action_constraints: constraint_names.clone(),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        // Resolve Ident nodes to StateVar nodes in operator bodies loaded
        // into the EvalCtx. Without this, the POR dependency extractor
        // cannot see state variable references (it only recognizes
        // Expr::StateVar and Expr::Prime(StateVar), not bare Ident nodes).
        // The production code path (ModelChecker) calls this during setup.
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let analysis = ActionConstraintAnalysis::build(&setup.ctx, &constraint_names);
        (setup.ctx, constraint_names, analysis)
    }

    #[test]
    fn analysis_extracts_reads_and_writes() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE AnalysisTest ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ y' = y
OnlyIncrease == x' > x
====
"#;
        let (ctx, _, analysis) = make_test_ctx_and_analysis(src, &["OnlyIncrease"]);
        let registry = ctx.var_registry().clone();

        assert_eq!(analysis.constraints.len(), 1);
        let deps = &analysis.constraints[0];
        assert!(!deps.must_always_eval);

        // OnlyIncrease references x' (write) and x (read)
        let x_idx = registry.get("x").unwrap();
        assert!(deps.reads.contains(&x_idx) || deps.writes.contains(&x_idx));
        assert!(!deps.all_vars.is_empty());
    }

    #[test]
    fn skip_optimization_disabled_for_soundness() {
        let _serial = crate::test_utils::acquire_interner_lock();
        // The skip optimization was unsound: it assumed constraints pass when
        // no referenced variables changed, but constraints like x' > x FAIL
        // on stuttering steps (where x' == x). See the DISABLED comment on
        // can_skip_constraint. This test verifies the optimization is off.
        let src = r#"
---- MODULE SkipTest ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ y' = y
OnlyIncreaseX == x' > x
====
"#;
        let (ctx, _, analysis) = make_test_ctx_and_analysis(src, &["OnlyIncreaseX"]);
        let registry = ctx.var_registry().clone();

        // Transition where only y changes (x stays the same)
        let current = ArrayState::from_state(
            &crate::State::from_pairs([("x", Value::int(1)), ("y", Value::int(0))]),
            &registry,
        );
        let succ_y_only = ArrayState::from_state(
            &crate::State::from_pairs([("x", Value::int(1)), ("y", Value::int(5))]),
            &registry,
        );

        // Skip optimization is disabled for soundness — always returns false
        assert!(!analysis.can_skip_constraint(0, &current, &succ_y_only));

        // Transition where x changes — also cannot skip (optimization disabled)
        let succ_x_changes = ArrayState::from_state(
            &crate::State::from_pairs([("x", Value::int(2)), ("y", Value::int(0))]),
            &registry,
        );
        assert!(!analysis.can_skip_constraint(0, &current, &succ_x_changes));
    }

    #[test]
    fn no_skip_for_must_always_eval() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE TlcGetTest ----
EXTENDS Integers, TLC
VARIABLE x
Init == x = 0
Next == x' = x + 1
LevelBound == TLCGet("level") < 10
====
"#;
        let (_ctx, _, analysis) = make_test_ctx_and_analysis(src, &["LevelBound"]);

        assert_eq!(analysis.constraints.len(), 1);
        assert!(analysis.constraints[0].must_always_eval);
    }

    #[test]
    fn exact_seen_state_reuse_certifies_pure_state_constraint() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE PureStateConstraintTest ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x + 1
Bound == x \in 0..10
====
"#;
        let (_ctx, _, analysis) = make_test_ctx_and_analysis(src, &["Bound"]);

        assert!(analysis.supports_exact_seen_state_reuse());

        // Sailfish-shaped constraint: quantified function lookup through
        // zero-arity helper operators and pure FiniteSets builtins.
        let sailfish_src = r#"
---- MODULE PureSailfishStateConstraintTest ----
EXTENDS Integers, FiniteSets
VARIABLE round
N == {1, 2, 3}
F == {1}
R == 1..5
Init == round = [n \in N |-> 0]
Next == round' = round
Bound == \A n \in N \ F : round[n] \in 0..Max(R)
====
"#;
        let (_ctx, _, analysis) = make_test_ctx_and_analysis(sailfish_src, &["Bound"]);

        assert!(analysis.supports_exact_seen_state_reuse());
    }

    #[test]
    fn exact_reuse_rejects_configured_root_replacement_across_backends() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE RawNamedConstraintRootTest ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' = x /\ y' = y
Safety == x = 0
MCSafety == y = 0
====
"#;
        let (mut ctx, names, _) = make_test_ctx_and_analysis(src, &["Safety"]);
        ctx.add_op_replacement("Safety".to_string(), "MCSafety".to_string());

        assert_eq!(ctx.resolve_op_name("Safety"), "MCSafety");
        assert!(ctx.get_op("Safety").is_some());
        assert!(ctx.get_op("MCSafety").is_some());

        // Canonical AST dispatch reads raw Safety(x), while implicit-default
        // TIR currently resolves the root to MCSafety(y). Shared exact reuse
        // cannot select either projection without becoming backend-dependent.
        let analysis = ActionConstraintAnalysis::build(&ctx, &names);
        assert!(analysis.exact_reuse_projection(0).is_none());
        assert!(analysis.constraints[0].must_always_eval);
        assert!(
            !analysis.supports_exact_seen_state_reuse(),
            "run_prepare exact-duplicate reuse must reject a configured root replacement",
        );
    }

    #[test]
    fn exact_reuse_rejects_forged_or_stale_state_var_metadata() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE ExactStateVarSlotTest ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' = x /\ y' = y
Constraint == x = 0
====
"#;
        let (ctx, _, _) = make_test_ctx_and_analysis(src, &["Constraint"]);
        let registry = ctx.var_registry();
        let x = registry.get("x").unwrap();
        let y = registry.get("y").unwrap();
        let x_id = registry.name_id_at(x);
        let y_id = registry.name_id_at(y);

        let cases = [
            (
                "stale in-range index",
                Spanned::dummy(Expr::StateVar("y".to_string(), x.0, y_id)),
            ),
            (
                "forged NameId",
                Spanned::dummy(Expr::StateVar("x".to_string(), x.0, y_id)),
            ),
            (
                "missing NameId",
                Spanned::dummy(Expr::StateVar(
                    "x".to_string(),
                    x.0,
                    tla_core::NameId::INVALID,
                )),
            ),
            (
                "forged name",
                Spanned::dummy(Expr::StateVar("y".to_string(), x.0, x_id)),
            ),
            (
                "out-of-range index",
                Spanned::dummy(Expr::StateVar(
                    "x".to_string(),
                    u16::try_from(registry.len()).unwrap(),
                    x_id,
                )),
            ),
        ];

        for (reason, expr) in cases {
            let analyzed = analyze_constraint_expr(&ctx, reason, &expr);
            assert!(
                !analyzed.state_var_slots_exact,
                "{reason} must not certify a raw POR slot",
            );
            assert!(
                exact_reuse_projection_for_expr(&ctx, &expr).is_none(),
                "{reason} must fail closed for per-expression exact reuse",
            );

            let whole = ActionConstraintAnalysis {
                constraints: vec![analyzed],
                skipped: AtomicU64::new(0),
                evaluated: AtomicU64::new(0),
            };
            assert!(
                whole.exact_reuse_projection(0).is_none(),
                "{reason} must fail closed for the whole-predicate cache",
            );
            assert!(
                !whole.supports_exact_seen_state_reuse(),
                "{reason} must disable run_prepare exact-duplicate reuse",
            );
        }

        let genuine = Spanned::dummy(Expr::StateVar("x".to_string(), x.0, x_id));
        assert_eq!(
            exact_reuse_projection_for_expr(&ctx, &genuine),
            Some(vec![x]),
        );

        let stale_primed = Spanned::dummy(Expr::Prime(Box::new(Spanned::dummy(Expr::StateVar(
            "y".to_string(),
            x.0,
            y_id,
        )))));
        let primed_analysis = analyze_constraint_expr(&ctx, "primed stale slot", &stale_primed);
        assert!(primed_analysis.references_next_state);
        assert!(
            !primed_analysis.state_var_slots_exact,
            "primed residue must be slot-validated before the next-state rejection",
        );
    }

    #[test]
    fn exact_projection_certifies_genuine_seq_builtin() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE PureSeqProjectionTest ----
EXTENDS Integers, Sequences
VARIABLE consumed
Init == consumed = <<>>
Next == consumed' = consumed
Constraint == consumed \in Seq({TRUE, FALSE})
====
"#;
        let (ctx, _, analysis) = make_test_ctx_and_analysis(src, &["Constraint"]);
        let body = ctx
            .get_op("Constraint")
            .expect("Constraint should be loaded")
            .body
            .clone();
        let consumed = ctx
            .var_registry()
            .get("consumed")
            .expect("consumed should be registered");
        assert!(analysis.supports_exact_seen_state_reuse());
        assert_eq!(
            exact_reuse_projection_for_expr(&ctx, &body),
            Some(vec![consumed]),
            "the genuine pure Seq builtin should depend only on its state argument",
        );
    }

    #[test]
    fn exact_projection_rejects_replaced_or_shadowed_seq() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let replaced_src = r#"
---- MODULE ReplacedSeqProjectionTest ----
EXTENDS Integers, Sequences
VARIABLE consumed
Init == consumed = <<>>
Next == consumed' = consumed
MySeq(S) == {<<>>}
Constraint == consumed \in Seq(Nat)
====
"#;
        let (mut replaced_ctx, _, _) = make_test_ctx_and_analysis(replaced_src, &["Constraint"]);
        replaced_ctx.add_op_replacement("Seq".to_string(), "MySeq".to_string());
        let replaced_body = replaced_ctx
            .get_op("Constraint")
            .expect("Constraint should be loaded")
            .body
            .clone();
        assert!(
            exact_reuse_projection_for_expr(&replaced_ctx, &replaced_body).is_none(),
            "a config replacement must not inherit the Seq builtin certificate",
        );

        let shadowed_src = r#"
---- MODULE ShadowedSeqProjectionTest ----
VARIABLE consumed
Init == consumed = <<>>
Next == consumed' = consumed
Seq(S) == {<<>>}
Constraint == consumed \in Seq({0})
====
"#;
        let (shadowed_ctx, _, _) = make_test_ctx_and_analysis(shadowed_src, &["Constraint"]);
        let shadowed_body = shadowed_ctx
            .get_op("Constraint")
            .expect("Constraint should be loaded")
            .body
            .clone();
        assert!(
            exact_reuse_projection_for_expr(&shadowed_ctx, &shadowed_body).is_none(),
            "a main-module Seq operator must not inherit the builtin certificate",
        );

        // The surface parser reserves `Seq` as a stdlib token in formal
        // position, but programmatic/module-rewrite ASTs can still contain
        // that lexical shape. Pin the certificate directly at the AST layer.
        let lexical_ctx_src = r#"
---- MODULE LexicalSeqProjectionContext ----
EXTENDS Integers, Sequences
VARIABLE consumed
Init == consumed = <<>>
Next == consumed' = consumed
SeqClosure == LAMBDA S : {<<>>}
Constraint == consumed \in Seq({0})
====
"#;
        let (lexical_ctx, _, _) = make_test_ctx_and_analysis(lexical_ctx_src, &["Constraint"]);
        let lexical_body = Spanned::dummy(Expr::Let(
            vec![tla_core::ast::OperatorDef {
                name: Spanned::dummy("Apply".to_string()),
                params: vec![tla_core::ast::OpParam {
                    name: Spanned::dummy("Seq".to_string()),
                    arity: 1,
                }],
                body: Spanned::dummy(Expr::Apply(
                    Box::new(Spanned::dummy(Expr::Ident(
                        "Seq".to_string(),
                        tla_core::NameId::INVALID,
                    ))),
                    vec![Spanned::dummy(Expr::Ident(
                        "Nat".to_string(),
                        tla_core::NameId::INVALID,
                    ))],
                )),
                local: true,
                contains_prime: false,
                guards_depend_on_prime: false,
                has_primed_param: false,
                is_recursive: false,
                self_call_count: 0,
            }],
            Box::new(Spanned::dummy(Expr::Bool(true))),
        ));
        assert!(
            exact_reuse_projection_for_expr(&lexical_ctx, &lexical_body).is_none(),
            "an applied operator formal named Seq must not inherit the builtin certificate",
        );

        let constraint_body = lexical_ctx
            .get_op("Constraint")
            .expect("Constraint should be loaded")
            .body
            .clone();
        assert!(
            exact_reuse_projection_for_expr(&lexical_ctx, &constraint_body).is_some(),
            "unshadowed Seq should retain the genuine builtin certificate",
        );
        let seq_closure = lexical_ctx
            .eval_op("SeqClosure")
            .expect("SeqClosure should evaluate");
        assert!(matches!(seq_closure, Value::Closure(_)));
        let closure_bound_ctx = lexical_ctx.bind_local("Seq", seq_closure);
        assert!(
            exact_reuse_projection_for_expr(&closure_bound_ctx, &constraint_body).is_none(),
            "a binding-chain closure named Seq shadows membership builtin dispatch",
        );
    }

    #[test]
    fn exact_reuse_rejects_state_dependent_operator_named_like_builtin_constant() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE ShadowedNatProjectionTest ----
VARIABLE x
Init == x = 0
Next == x' = x
Nat == [i \in {1} |-> x]
Constraint == Nat[1] = 0
====
"#;
        let (ctx, _, analysis) = make_test_ctx_and_analysis(src, &["Constraint"]);
        let body = ctx
            .get_op("Constraint")
            .expect("Constraint should be loaded")
            .body
            .clone();

        assert!(ctx.get_op("Nat").is_some());
        assert!(
            !analysis.supports_exact_seen_state_reuse(),
            "a user FuncDef named Nat must not inherit the builtin-set empty projection",
        );
        assert!(analysis.constraints[0].must_always_eval);
        assert!(exact_reuse_projection_for_expr(&ctx, &body).is_none());
    }

    #[test]
    fn exact_reuse_follows_shadowed_builtin_inside_residual_funcdef() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE NestedShadowedNatProjectionTest ----
VARIABLE x
Init == x = 0
Next == x' = x
Nat == [i \in {1} |-> x]
Wrapper == [j \in {1} |-> Nat[1]]
Constraint == Wrapper[1] = 0
====
"#;
        let (ctx, _, analysis) = make_test_ctx_and_analysis(src, &["Constraint"]);
        let body = ctx
            .get_op("Constraint")
            .expect("Constraint should be loaded")
            .body
            .clone();
        let x = ctx.var_registry().get("x").expect("x should be registered");

        assert!(ctx.get_op("Nat").is_some());
        assert!(ctx.get_op("Wrapper").is_some());
        assert_eq!(
            analysis.exact_reuse_projection(0),
            Some([x].as_slice()),
            "a residual FuncDef body must resolve user Nat before the builtin set",
        );
        assert_eq!(exact_reuse_projection_for_expr(&ctx, &body), Some(vec![x]));
    }

    #[test]
    fn exact_seen_state_reuse_rejects_context_io_and_opaque_residue() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let cases = [
            (
                r#"
---- MODULE StateConstraintTlcGetTest ----
EXTENDS Integers, TLC
VARIABLE x
Init == x = 0
Next == x' = x + 1
Constraint == TLCGet("level") < 10
====
"#,
                "Constraint",
                "TLCGet",
            ),
            (
                r#"
---- MODULE StateConstraintPrintTest ----
EXTENDS Integers, TLC
VARIABLE x
Init == x = 0
Next == x' = x + 1
Constraint == Print("state", x) = x
====
"#,
                "Constraint",
                "Print",
            ),
            (
                r#"
---- MODULE StateConstraintIoTest ----
EXTENDS IOUtils
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == IOEnvGet("PATH") /= ""
====
"#,
                "Constraint",
                "IO",
            ),
            (
                r#"
---- MODULE StateConstraintLocalPlaceholderTest ----
VARIABLE x
Init == x = 0
Next == x' = x
IOEnvGet(key) == TRUE
Constraint == IOEnvGet("PATH") /= ""
====
"#,
                "Constraint",
                "runtime placeholder override",
            ),
            (
                r#"
---- MODULE StateConstraintDeferredToStringTest ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == ToString(LAMBDA value : value) /= ""
====
"#,
                "Constraint",
                "ToString over deferred executable data",
            ),
            (
                r#"
---- MODULE StateConstraintOpaqueTest ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x + 1
Constraint == MissingValue = x
====
"#,
                "Constraint",
                "unresolved residue",
            ),
        ];

        for (src, name, reason) in cases {
            let (_ctx, _, analysis) = make_test_ctx_and_analysis(src, &[name]);
            assert!(
                !analysis.supports_exact_seen_state_reuse(),
                "{reason} must fail closed"
            );
            assert!(
                analysis.constraints[0].must_always_eval,
                "{reason} must use the existing must_always_eval gate"
            );
        }
    }

    #[test]
    fn exact_seen_state_reuse_rejects_primed_state_constraint() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE PrimedStateConstraintTest ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x + 1
Constraint == x' = x
====
"#;
        let (_ctx, _, analysis) = make_test_ctx_and_analysis(src, &["Constraint"]);

        assert!(!analysis.supports_exact_seen_state_reuse());
        assert!(analysis.constraints[0].references_next_state);
    }

    #[test]
    fn optimized_eval_matches_baseline() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE EvalMatchTest ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' \in {x-1, x, x+1} /\ y' \in {y-1, y, y+1}
OnlyIncrease == x' >= x
====
"#;
        let (mut ctx, names, analysis) = make_test_ctx_and_analysis(src, &["OnlyIncrease"]);
        let registry = ctx.var_registry().clone();

        let current = ArrayState::from_state(
            &crate::State::from_pairs([("x", Value::int(1)), ("y", Value::int(0))]),
            &registry,
        );

        // Successor where x increases (constraint should pass)
        let succ_pass = ArrayState::from_state(
            &crate::State::from_pairs([("x", Value::int(2)), ("y", Value::int(0))]),
            &registry,
        );
        let result = check_action_constraints_with_analysis(
            &mut ctx, &names, &current, &succ_pass, &analysis,
        )
        .unwrap();
        assert!(result, "x increased: constraint should pass");

        // Successor where x decreases (constraint should fail)
        let succ_fail = ArrayState::from_state(
            &crate::State::from_pairs([("x", Value::int(0)), ("y", Value::int(0))]),
            &registry,
        );
        let result = check_action_constraints_with_analysis(
            &mut ctx, &names, &current, &succ_fail, &analysis,
        )
        .unwrap();
        assert!(!result, "x decreased: constraint should fail");

        // Successor where only y changes — constraint x' >= x is satisfied
        // because x' == x (1 >= 1). With skip optimization disabled, this is
        // evaluated through the full constraint path and still passes.
        let succ_y_only = ArrayState::from_state(
            &crate::State::from_pairs([("x", Value::int(1)), ("y", Value::int(5))]),
            &registry,
        );
        let result = check_action_constraints_with_analysis(
            &mut ctx,
            &names,
            &current,
            &succ_y_only,
            &analysis,
        )
        .unwrap();
        assert!(
            result,
            "only y changed, x' >= x holds (1 >= 1): constraint should pass"
        );
        // Skip optimization is disabled for soundness, so skip count stays 0
        assert_eq!(
            analysis.skip_count(),
            0,
            "skip optimization is disabled — no skips should be recorded"
        );
    }

    #[test]
    fn empty_constraints_always_pass() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE EmptyTest ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let (mut ctx, _, _) = make_test_ctx_and_analysis(src, &[]);
        let analysis = ActionConstraintAnalysis::build(&ctx, &[]);
        let registry = ctx.var_registry().clone();
        let current =
            ArrayState::from_state(&crate::State::from_pairs([("x", Value::int(0))]), &registry);
        let succ =
            ArrayState::from_state(&crate::State::from_pairs([("x", Value::int(1))]), &registry);

        let result =
            check_action_constraints_with_analysis(&mut ctx, &[], &current, &succ, &analysis)
                .unwrap();
        assert!(result);
    }

    #[test]
    fn exact_reuse_projection_certifies_only_pure_unprimed_state_reads() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE ExactProjectionTest ----
EXTENDS Integers, TLC
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == UNCHANGED <<x, y>>
Pure == x >= 0 /\ y \in 0..10
Primed == x' = x
Contextual == TLCGet("level") >= 0
====
"#;
        let (ctx, _, pure) = make_test_ctx_and_analysis(src, &["Pure"]);
        let registry = ctx.var_registry();
        assert_eq!(
            pure.exact_reuse_projection(0).unwrap(),
            &[registry.get("x").unwrap(), registry.get("y").unwrap()]
        );

        let primed = ActionConstraintAnalysis::build(&ctx, &["Primed".to_string()]);
        assert!(primed.exact_reuse_projection(0).is_none());
        let contextual = ActionConstraintAnalysis::build(&ctx, &["Contextual".to_string()]);
        assert!(contextual.exact_reuse_projection(0).is_none());
    }

    #[test]
    fn exact_reuse_projection_rejects_configured_root_replacement() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let src = r#"
---- MODULE ExactProjectionRootReplacement ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == UNCHANGED <<x, y>>
Safety == x = 0
MCSafety == y = 0
====
"#;
        let (mut ctx, names, _) = make_test_ctx_and_analysis(src, &["Safety"]);
        ctx.add_op_replacement("Safety".to_string(), "MCSafety".to_string());
        let analysis = ActionConstraintAnalysis::build(&ctx, &names);
        assert!(analysis.exact_reuse_projection(0).is_none());
    }

    #[test]
    fn action_routing_replay_gate_accepts_pure_recursion_and_rejects_effects() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let identity_src = r#"
---- MODULE ActionReplayIdentitySafety ----
VARIABLE router_replay_unique_identity_state_4053
IdentityAction ==
    router_replay_unique_identity_state_4053' =
        router_replay_unique_identity_state_4053
====
"#;
        let (identity_ctx, _, _) = make_test_ctx_and_analysis(identity_src, &[]);
        let identity = &identity_ctx
            .get_op("IdentityAction")
            .expect("operator exists")
            .body;
        // Even the identity-write shortcut must inspect the prime expression:
        // partial-state fallback can expose the state-variable spelling as a
        // TLC string, so replay cannot begin before its token is fixed.
        assert!(!action_expr_is_replay_safe(&identity_ctx, identity));
        let _identity_state_var_name = Value::string("router_replay_unique_identity_state_4053");
        assert!(action_expr_is_replay_safe(&identity_ctx, identity));

        let src = r#"
---- MODULE ActionReplaySafety ----
EXTENDS Integers, TLC, FiniteSets, Sequences
VARIABLE x, f
Init == /\ x = 0 /\ f = [i \in {0} |-> 0]
PureAction == /\ x = 0 /\ x' = x + 1
PureBuiltinAction == /\ Cardinality({x}) = 1
                     /\ Head(<<x>>) = x
                     /\ x' = x + 1
PureConcatAction == /\ x = 0
                    /\ x' = Head(<<x>> \o <<x + 1>>)
StringConcatAction == /\ x = 0
                      /\ x' = "router_replay_" \o "fresh_concat"
StringSubSeqAction == /\ x = 0
                      /\ x' = SubSeq("router_replay_fresh_subseq", 2, 8)
ModelValueAction == /\ x = 0
                    /\ x' = TLCModelValue("router_replay_fresh_model_value")
SetToSeqAction == /\ x = 0
                  /\ x' = SetToSeq({"router_set_b", "router_set_a"})
DomainRecordAction == /\ x = 0
                      /\ Cardinality(DOMAIN [fresh_field |-> x]) = 1
                      /\ x' = x + 1
RecordSetAction == /\ x = 0
                   /\ [fresh_set_field : {x}] = [fresh_set_field : {x}]
                   /\ x' = x + 1
FreshStringWriteAction == x' = "router_replay_fresh_constant_write"
FreshRecordWriteAction == x' = [router_replay_fresh_record_field |-> 0]
FreshRecordSetWriteAction == x' = [router_replay_fresh_record_set_field : {0}]
FreshExceptIdentityAction ==
    f' = [f EXCEPT !["router_replay_fresh_except_key"] =
                      f["router_replay_fresh_except_key"]]
FreshLetIdentityAction ==
    x' = LET router_replay_fresh_identity_let == 0 IN x
ConcatClosure == LAMBDA a, b : TLCGet("level")
RECURSIVE PureCountdown(_)
PureCountdown(router_replay_countdown_n) ==
    IF router_replay_countdown_n = 0
    THEN 0
    ELSE PureCountdown(router_replay_countdown_n - 1)
PureRecursiveAction == /\ x = 0 /\ x' = PureCountdown(x + 1)
PureChooseAction == /\ x = 0
                    /\ x' = CHOOSE router_replay_choice_n \in 0..1 :
                                  router_replay_choice_n = 1
ContextualAction == /\ x = 0 /\ TLCGet("level") >= 0 /\ x' = x + 1
SideEffectAction == /\ x = 0 /\ TLCSet(0, x) /\ x' = x + 1
PrintAction == /\ x = 0 /\ PrintT(x) /\ x' = x + 1
TimeAction == /\ x = 0 /\ x' = JavaTime
RandomAction == /\ x = 0 /\ x' = RandomElement({0, 1})
WrongArityAction == /\ x = 0 /\ Head(<<x>>, x) = x /\ x' = x + 1
ShadowedRandomAction == LET RandomElement(S) == CHOOSE e \in S : TRUE
                        IN /\ x = 0
                           /\ x' = RandomElement({0, 1})
RandomSubset(k, S) == TRUE
PlaceholderRandomAction == /\ x = 0
                           /\ x' = RandomSubset(1, {0, 1})
UnknownAction == /\ x = 0 /\ MysteryBuiltin(x) /\ x' = x + 1
Next == PureAction \/ ContextualAction \/ SideEffectAction
====
"#;
        let (ctx, _, _) = make_test_ctx_and_analysis(src, &[]);
        let pure = &ctx.get_op("PureAction").expect("operator exists").body;
        let pure_builtin = &ctx
            .get_op("PureBuiltinAction")
            .expect("operator exists")
            .body;
        let pure_recursive = &ctx
            .get_op("PureRecursiveAction")
            .expect("operator exists")
            .body;
        let pure_concat = &ctx
            .get_op("PureConcatAction")
            .expect("operator exists")
            .body;
        let string_concat = &ctx
            .get_op("StringConcatAction")
            .expect("operator exists")
            .body;
        let string_subseq = &ctx
            .get_op("StringSubSeqAction")
            .expect("operator exists")
            .body;
        let model_value = &ctx
            .get_op("ModelValueAction")
            .expect("operator exists")
            .body;
        let set_to_seq = &ctx.get_op("SetToSeqAction").expect("operator exists").body;
        let domain_record = &ctx
            .get_op("DomainRecordAction")
            .expect("operator exists")
            .body;
        let record_set = &ctx.get_op("RecordSetAction").expect("operator exists").body;
        let fresh_string_write = &ctx
            .get_op("FreshStringWriteAction")
            .expect("operator exists")
            .body;
        let fresh_record_write = &ctx
            .get_op("FreshRecordWriteAction")
            .expect("operator exists")
            .body;
        let fresh_record_set_write = &ctx
            .get_op("FreshRecordSetWriteAction")
            .expect("operator exists")
            .body;
        let fresh_except_identity = &ctx
            .get_op("FreshExceptIdentityAction")
            .expect("operator exists")
            .body;
        let fresh_let_identity = &ctx
            .get_op("FreshLetIdentityAction")
            .expect("operator exists")
            .body;
        let pure_choose = &ctx
            .get_op("PureChooseAction")
            .expect("operator exists")
            .body;
        let contextual = &ctx
            .get_op("ContextualAction")
            .expect("operator exists")
            .body;
        let side_effect = &ctx
            .get_op("SideEffectAction")
            .expect("operator exists")
            .body;
        let print = &ctx.get_op("PrintAction").expect("operator exists").body;
        let time = &ctx.get_op("TimeAction").expect("operator exists").body;
        let random = &ctx.get_op("RandomAction").expect("operator exists").body;
        let wrong_arity = &ctx
            .get_op("WrongArityAction")
            .expect("operator exists")
            .body;
        let shadowed_random = &ctx
            .get_op("ShadowedRandomAction")
            .expect("operator exists")
            .body;
        let placeholder_random = &ctx
            .get_op("PlaceholderRandomAction")
            .expect("operator exists")
            .body;
        let unknown = &ctx.get_op("UnknownAction").expect("operator exists").body;

        let random_subset = ctx.get_op("RandomSubset").expect("operator exists");
        assert!(crate::eval::should_prefer_builtin_override(
            "RandomSubset",
            random_subset.as_ref(),
            2,
            &ctx,
        ));

        let concat_closure = ctx
            .eval_op("ConcatClosure")
            .expect("ConcatClosure should evaluate");
        assert!(matches!(concat_closure, Value::Closure(_)));
        let shadowed_concat_ctx = ctx.bind_local("\\o", concat_closure);

        // Primed partial-state fallback can expose the state-variable spelling
        // as a TLC string. Model the canonical prefix that fixed it before
        // router admission.
        let _state_var_name = Value::string("x");
        let _function_state_var_name = Value::string("f");
        assert!(action_expr_is_replay_safe(&ctx, pure));
        assert!(action_expr_is_replay_safe(&ctx, pure_builtin));
        assert!(action_expr_is_replay_safe(&ctx, pure_concat));
        assert!(!action_expr_is_replay_safe(
            &shadowed_concat_ctx,
            pure_concat
        ));
        assert!(!action_expr_is_replay_safe(&ctx, string_concat));
        assert!(!action_expr_is_replay_safe(&ctx, string_subseq));
        assert!(!action_expr_is_replay_safe(&ctx, model_value));
        assert!(!action_expr_is_replay_safe(&ctx, set_to_seq));
        assert!(!action_expr_is_replay_safe(&ctx, domain_record));
        assert!(!action_expr_is_replay_safe(&ctx, record_set));
        assert!(!action_expr_is_replay_safe(&ctx, fresh_string_write));
        assert!(!action_expr_is_replay_safe(&ctx, fresh_record_write));
        assert!(!action_expr_is_replay_safe(&ctx, fresh_record_set_write));
        assert!(!action_expr_is_replay_safe(&ctx, fresh_except_identity));
        assert!(!action_expr_is_replay_safe(&ctx, fresh_let_identity));
        // The evaluator interns formals and bound names into the same
        // first-seen table as semantic strings. Admission is read-only and
        // must wait until the canonical prefix has fixed those tokens.
        assert!(!action_expr_is_replay_safe(&ctx, pure_recursive));
        assert!(!action_expr_is_replay_safe(&ctx, pure_choose));
        let _countdown_name = Value::string("router_replay_countdown_n");
        let _choice_name = Value::string("router_replay_choice_n");
        assert!(action_expr_is_replay_safe(&ctx, pure_recursive));
        assert!(action_expr_is_replay_safe(&ctx, pure_choose));
        assert!(!action_expr_is_replay_safe(&ctx, contextual));
        assert!(!action_expr_is_replay_safe(&ctx, side_effect));
        assert!(!action_expr_is_replay_safe(&ctx, print));
        assert!(!action_expr_is_replay_safe(&ctx, time));
        assert!(!action_expr_is_replay_safe(&ctx, random));
        assert!(!action_expr_is_replay_safe(&ctx, wrong_arity));
        assert!(!action_expr_is_replay_safe(&ctx, shadowed_random));
        assert!(!action_expr_is_replay_safe(&ctx, placeholder_random));
        assert!(!action_expr_is_replay_safe(&ctx, unknown));
    }
}
