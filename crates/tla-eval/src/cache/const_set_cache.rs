// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Constant set-producing subexpression cache.
//!
//! A large fraction of set-construction work in the interpreter is fully
//! REDUNDANT: the same set CONTENT is rebuilt on every state evaluation. The
//! redundant builds come from CONSTANT (state-independent) set subexpressions —
//! CONSTANTS-derived sets, fixed ranges `a..b`, set literals `{a, b, c}`,
//! constant `SUBSET`/`UNION`/set ops — that are re-evaluated for every state
//! even though their value never changes for the whole run.
//!
//! This cache stores a set-producing subexpression's evaluated (interned) result
//! by AST-node identity (discriminated by scope), built ONCE per run and reused
//! thereafter. It is the ACROSS-state analogue of `quantifier_hoist` (which
//! caches WITHIN a quantifier loop / state generation only).
//!
//! ## Soundness (paramount)
//!
//! A cached set must be value-identical to rebuilding it on every consult.
//! Caching a set expr that secretly depends on state or a bound variable would
//! be a correctness bug. Two independent mechanisms keep this sound:
//!
//! 1. DYNAMIC independence test. On the FIRST evaluation of a candidate set
//!    node, the node is evaluated under dependency tracking. The result is only
//!    cached when `deps_are_persistent()` holds — i.e. the evaluation read NO
//!    current-state variable, NO next-state variable, NO enclosing
//!    quantifier/LET-bound local, NO `TLCGet("level")`, touched NO INSTANCE lazy
//!    binding, and was not flagged inconsistent/nondeterministic. This reuses
//!    the exact predicate that already gates the persistent partition of the
//!    zero-arg / nary operator caches (`deps_are_persistent`, #3100/#3447). If
//!    independence is NOT proven, or the returned deferred value captures a
//!    state environment without reading it yet, the node is recorded in a
//!    "non-constant" set so future evaluations skip the (re-)probe and simply
//!    rebuild — fail-safe.
//!
//! 2. SCOPE-DISCRIMINATED key. The same AST node can evaluate to different
//!    constant sets under different INSTANCE substitutions or LET-operator
//!    scopes (e.g. module `M` instantiated twice with different CONSTANT
//!    bindings: `M(P <- {1,2})` vs `M(P <- {3,4})`). A read of a *constant*
//!    INSTANCE binding is deliberately NOT tainted as `instance_lazy_read`
//!    (eval_state_var_lookup.rs:171), so the dynamic test alone would classify
//!    such a node as persistent. The cache key therefore includes the same
//!    `local_ops_id` + `instance_subs_id` scope fingerprints that discriminate
//!    `LetScopeKey` / `NaryOpCacheKey` / `ZeroArgCacheKey`. With no INSTANCE and
//!    no LET-operator scope (the common BFS case) both ids are 0 and the key is
//!    effectively the bare AST pointer.
//!
//! Caching is additionally SUPPRESSED while a call-by-name parameter
//! substitution is active (`call_by_name_subs`, used only by primed-parameter
//! operators) and inside ENABLED scope (where dep tracking can misclassify
//! state-dependent results as constant). Both are conservative fail-safes.
//!
//! The cache lives for the run and is cleared at every full reset boundary
//! (run reset / phase boundary / test reset), exactly like the other eval
//! caches. AST pointers are stable for the run because the parsed AST is
//! immutable; the run-reset clear forecloses any cross-run pointer-reuse hazard.

use super::dep_tracking::OpDepGuard;
use super::small_caches::{let_scope_key, LetScopeKey};
use super::zero_arg_cache::deps_are_persistent;
use crate::error::EvalResult;
use crate::value::Value;
use crate::EvalCtx;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::sync::OnceLock;
use tla_core::ast::Expr;

/// Runtime gate. Enabled by default; set `TY_CONST_SET_CACHE=0` to disable
/// (used for A/B measurement — one binary serves both baseline and after).
#[inline]
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("TY_CONST_SET_CACHE")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Per-run cache state for constant set-producing subexpressions.
struct ConstSetState {
    /// Proven-constant set nodes: scope-discriminated key -> (source span, value).
    ///
    /// The source `Span` is stored alongside the value and validated on every
    /// lookup (bug #1558). The key's `body_ptr` is a raw AST-node pointer; for
    /// the immutable parsed AST it is stable, but TRANSIENT expression clones
    /// (closure / lazy-thunk bodies built via `Expr::clone()`) are freed and
    /// their addresses reused by unrelated nodes within the same run. Without
    /// validation a reused address would serve another expression's cached set
    /// — a correctness bug. Source spans are stable per source location and
    /// differ between distinct source expressions, so a span mismatch reliably
    /// detects pointer reuse and is treated as a miss (rebuild + recache).
    values: FxHashMap<LetScopeKey, (tla_core::Span, Value)>,
    /// Nodes proven NOT constant (or that errored during probe). Skips re-probing.
    non_const: FxHashSet<LetScopeKey>,
}

thread_local! {
    static CONST_SET_STATE: RefCell<ConstSetState> = RefCell::new(ConstSetState {
        values: FxHashMap::default(),
        non_const: FxHashSet::default(),
    });
}

/// Structural allowlist: set-producing expression kinds whose value is a pure
/// function of their evaluated children (no hidden runtime dependency the
/// dep-tracking probe could not observe). These are the nodes whose redundant
/// rebuilds the profiler attributed ~100% redundancy to.
///
/// `SetBuilder`/`SetFilter` introduce their OWN bound variables, but those are
/// internal iteration variables; the dep-tracking probe correctly classifies a
/// comprehension over a constant domain as persistent (internal iteration
/// variables are not recorded as deps — see `record_local_read`'s base-stack
/// filter), and a comprehension whose domain/predicate reads an outer bound var
/// or a state var as non-persistent. They are therefore sound to include.
#[inline]
pub(crate) fn is_const_cacheable_set(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::SetEnum(_)
            | Expr::Range(_, _)
            | Expr::SetBuilder(_, _)
            | Expr::SetFilter(_, _)
            | Expr::Union(_, _)
            | Expr::Intersect(_, _)
            | Expr::SetMinus(_, _)
            | Expr::Powerset(_)
            | Expr::BigUnion(_)
            | Expr::Times(_)
            | Expr::FuncSet(_, _)
            | Expr::RecordSet(_)
    )
}

/// Whether `span` is the synthetic dummy span (`Span::dummy()` == `Span::default()`,
/// i.e. file 0, range `0..0`).
///
/// Constraint extraction substitutes bound-variable VALUES back into the body as
/// freshly-built literal `Expr` nodes (`value_to_expr` → `Spanned::new(.., Span::dummy())`),
/// e.g. `\E y \in D : b = y` becomes `b = {10}` where `{10}` is a `SetEnum` carrying
/// a dummy span. These transient clones are freed and their heap addresses reused
/// across iterations, AND they all share the SAME dummy span. The span-validation
/// that normally detects AST-pointer reuse (bug #1558) is therefore defeated: a
/// reused address + identical dummy span looks like a valid hit and serves the
/// PREVIOUS iteration's value (e.g. cached `{10}` returned for a node that now
/// evaluates to `{1}`). A real source expression never has a `0..0` span (module
/// text begins with `---- MODULE`), so rejecting dummy spans only forecloses the
/// hazard and never suppresses a legitimately cacheable constant.
#[inline]
fn is_dummy_span(span: tla_core::Span) -> bool {
    span == tla_core::Span::dummy()
}

/// Build the scope-discriminated cache key for a set node.
///
/// `body_ptr` is the AST node pointer; `local_ops_id` / `instance_subs_id`
/// disambiguate distinct LET-operator / INSTANCE-substitution scopes (both 0 in
/// the common no-INSTANCE BFS path, making the key effectively the pointer).
#[inline]
pub(crate) fn key_of(ctx: &EvalCtx, expr: &Expr) -> LetScopeKey {
    let_scope_key(ctx, expr as *const Expr as usize)
}

/// Look up a proven-constant set value. Returns `Some(value)` on a cache hit
/// (no eval, no rebuild). Returns `None` on miss.
///
/// SAFETY of a hit: the value was previously proven to have empty deps, so
/// returning it WITHOUT re-recording any dependency into the enclosing
/// operator's dep frame is correct — constants contribute no deps.
#[inline]
pub(crate) fn lookup(key: &LetScopeKey, span: Option<tla_core::Span>) -> Option<Value> {
    let Some(span) = span.filter(|s| !is_dummy_span(*s)) else {
        // No span (or only a dummy/synthetic span) available to validate against —
        // cannot prove the entry belongs to this expression; treat as a miss
        // (fail safe against pointer reuse). See `is_dummy_span` for why dummy
        // spans must be rejected.
        return None;
    };
    CONST_SET_STATE.with(|s| {
        s.borrow().values.get(key).and_then(|(stored_span, value)| {
            if *stored_span == span {
                Some(value.clone())
            } else {
                // Address reused by a different source expression (bug #1558).
                None
            }
        })
    })
}

/// Whether `key` is already known to be non-constant (skip the probe).
#[inline]
pub(crate) fn is_known_non_const(key: &LetScopeKey) -> bool {
    CONST_SET_STATE.with(|s| s.borrow().non_const.contains(key))
}

/// Evaluate a candidate set node under dependency tracking and learn whether it
/// is a true constant. `dispatch` performs the RAW dispatch for this node
/// (bypassing the const-set hook to avoid re-entry); children it evaluates flow
/// through the normal hook and may themselves be cached.
///
/// On success the result is returned; if proven persistent it is also stored in
/// the per-run const cache, otherwise the node is recorded as non-constant. On
/// error the node is recorded non-constant and the error is propagated (the
/// normal dispatch path will reproduce it deterministically on later evals).
#[inline]
pub(crate) fn probe_and_record(
    ctx: &EvalCtx,
    key: LetScopeKey,
    span: Option<tla_core::Span>,
    dispatch: impl FnOnce() -> EvalResult<Value>,
) -> EvalResult<Value> {
    // Push a fresh dep frame so reads performed while evaluating THIS node are
    // attributed here. base_stack_len = current binding depth: enclosing
    // quantifier/LET variables count as outer-scope deps (→ non-constant),
    // while iteration variables introduced inside this node do not.
    let base_stack_len = ctx.binding_depth;
    // If the chain holds a binding the dep rule cannot observe — `binding_depth`
    // desynced below the chain depth, e.g. the O(1) Init-enumeration path installs
    // an `\E` variable without a matching `binding_depth++` (see
    // `EvalCtx::has_unobservable_chain_bindings`) — then `deps_are_persistent`
    // UNDERCOUNTS: a value depending on that invisible binding records no local
    // dep and looks constant. Captured HERE, at the same point `base_stack_len`
    // is fixed, so the fail-safe below matches the dep rule's own base. The old
    // raw-pointer cache key never reused such a misclassified entry (a fresh body
    // clone per iteration), but the stable span key does, so it must be guarded.
    let unobservable_bindings = ctx.has_unobservable_chain_bindings();
    let guard = OpDepGuard::from_ctx(ctx, base_stack_len);

    let result = dispatch();

    let deps = match guard.try_take_deps() {
        Some(d) => d,
        None => {
            // Dep stack unexpectedly empty: do not cache; return result as-is.
            return result;
        }
    };

    // Re-merge observed deps into the enclosing dep frame (if any) so a
    // non-constant node still contributes its real state/local deps upward.
    super::dep_tracking::propagate_cached_deps(ctx, &deps);

    // A dummy/synthetic span (`Span::dummy()`, used for transient substituted-in
    // literals) is NOT a valid discriminator against AST-pointer reuse: many
    // distinct transient clones share the SAME dummy span, so span validation on
    // lookup cannot tell them apart. Treat such a span as "no span" so the node
    // is never cached (fail safe — it is rebuilt every time instead).
    let span = span.filter(|s| !is_dummy_span(*s));

    match result {
        Ok(value) => {
            // Fail safe on the desync above: if a chain binding was unobservable
            // to the dep rule, the persistence verdict is unreliable, so make NO
            // cache decision at all — neither cache it as constant (would be
            // unsound if it actually varies) nor mark it non-constant (would be a
            // permanent over-conservative decision keyed on a transient desync).
            // Just return the value and re-probe next time, exactly like the
            // no/dummy-span case.
            if unobservable_bindings {
                return Ok(value);
            }
            // Constructing a deferred SetPred snapshots state without evaluating
            // its predicate, so the dep frame can be empty even though the value
            // is state-dependent when later consumed. Treat every state-capturing
            // deferred result as non-constant, matching the zero-arg/LET cache
            // guards. This is essential now that direct RecordSet filters remain
            // lazy for Init bulk streaming.
            let value_captures_state = value.captures_state_environment();

            // Only cache when we can record a validating span (bug #1558). Without
            // a span we cannot defend against AST-pointer reuse, so fail safe.
            if let Some(span) = span.filter(|_| deps_are_persistent(&deps) && !value_captures_state)
            {
                CONST_SET_STATE.with(|s| {
                    s.borrow_mut().values.insert(key, (span, value.clone()));
                });
            } else if !deps_are_persistent(&deps) || value_captures_state {
                CONST_SET_STATE.with(|s| {
                    s.borrow_mut().non_const.insert(key);
                });
            }
            Ok(value)
        }
        Err(e) => {
            CONST_SET_STATE.with(|s| {
                s.borrow_mut().non_const.insert(key);
            });
            Err(e)
        }
    }
}

/// Returns true if probing/insertion should be skipped in the current scope.
///
/// - ENABLED scope: dep tracking can misclassify state-dependent results as
///   constant there (state vars are rebound into `env`, not `state_env`).
/// - call-by-name parameter substitution active: primed-parameter operators
///   substitute parameters per call; a node reading such a parameter could vary
///   per call without being caught by the pointer/scope key. Rare — suppress.
#[inline]
pub(crate) fn caching_suppressed(ctx: &EvalCtx) -> bool {
    super::lifecycle::in_enabled_scope_ctx(ctx) || ctx.call_by_name_subs.is_some()
}

/// Clear the constant set cache (run reset / phase boundary / test reset).
pub(crate) fn clear_const_set_cache() {
    CONST_SET_STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.values.clear();
        s.non_const.clear();
    });
}

/// Number of proven-constant entries (tests / diagnostics).
#[cfg(test)]
pub(crate) fn const_set_cache_len() -> usize {
    CONST_SET_STATE.with(|s| s.borrow().values.len())
}

/// Number of known-non-constant entries (tests / diagnostics).
#[cfg(test)]
pub(crate) fn const_set_non_const_len() -> usize {
    CONST_SET_STATE.with(|s| s.borrow().non_const.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::lifecycle::clear_for_test_reset;
    use crate::value::Value;
    use crate::{eval, EvalCtx};
    use tla_core::ast::Unit;
    use tla_core::{lower, parse_to_syntax_tree, FileId, Spanned};

    /// Parse `Op == <body>` and return the lowered body expression.
    fn lower_body(src: &str) -> Spanned<Expr> {
        let module_src = format!("---- MODULE Test ----\n\nOp == {}\n\n====", src);
        let tree = parse_to_syntax_tree(&module_src);
        let result = lower(FileId(0), &tree);
        assert!(
            result.errors.is_empty(),
            "lower errors: {:?}",
            result.errors
        );
        let module = result.module.expect("module should lower");
        for unit in &module.units {
            if let Unit::Operator(def) = &unit.node {
                if def.name.node == "Op" {
                    return def.body.clone();
                }
            }
        }
        panic!("Op not found");
    }

    #[test]
    fn allowlist_covers_set_producing_kinds_only() {
        assert!(is_const_cacheable_set(&Expr::SetEnum(vec![])));
        assert!(is_const_cacheable_set(&Expr::Powerset(Box::new(
            Spanned::new(Expr::SetEnum(vec![]), tla_core::Span::new(FileId(0), 0, 0))
        ))));
        // Non-set kinds are rejected.
        assert!(!is_const_cacheable_set(&Expr::Bool(true)));
        assert!(!is_const_cacheable_set(&Expr::Int(1.into())));
    }

    #[test]
    fn constant_set_literal_is_cached_and_value_correct() {
        clear_for_test_reset();
        // A pure constant set literal: state- and bound-var-independent.
        let body = lower_body("{1, 2, 3}");
        let ctx = EvalCtx::new();

        assert_eq!(const_set_cache_len(), 0);
        let v1 = eval(&ctx, &body).expect("eval ok");
        // First eval probes + caches the constant.
        assert_eq!(
            const_set_cache_len(),
            1,
            "constant set literal should be cached"
        );
        assert_eq!(const_set_non_const_len(), 0);

        // Second eval on the SAME AST node returns the cached value, byte-identical.
        let v2 = eval(&ctx, &body).expect("eval ok");
        assert_eq!(v1, v2, "cached value must equal a fresh build");

        // Sanity: the value is exactly {1,2,3}.
        let expected = {
            let c2 = EvalCtx::new();
            let b2 = lower_body("{3, 2, 1}");
            // clear so the {3,2,1} literal isn't served from {1,2,3}'s cache entry
            // (different AST pointer anyway, but be explicit).
            eval(&c2, &b2).expect("eval ok")
        };
        assert_eq!(
            v1, expected,
            "constant set value must be order-independent {{1,2,3}}"
        );

        clear_for_test_reset();
        assert_eq!(const_set_cache_len(), 0, "run reset clears the cache");
    }

    #[test]
    fn bound_variable_dependent_set_is_not_cached() {
        clear_for_test_reset();
        // The inner singleton {x} depends on the enclosing quantifier variable x.
        // It must be classified non-constant (never cached), while the constant
        // domain {1, 2} IS cacheable.
        let body = lower_body("{ {x} : x \\in {1, 2} }");
        let ctx = EvalCtx::new();

        let _ = eval(&ctx, &body).expect("eval ok");
        // The bound-var-dependent inner set {x} must NOT be cached.
        // At least one node was proven non-constant.
        assert!(
            const_set_non_const_len() >= 1,
            "a bound-variable-dependent set must be recorded non-constant"
        );

        // Re-eval must still produce the correct value: {{1}, {2}}.
        let v = eval(&ctx, &body).expect("eval ok");
        let expected_inner1 = Value::set(vec![Value::int(1)]);
        let expected_inner2 = Value::set(vec![Value::int(2)]);
        let expected = Value::set(vec![expected_inner1, expected_inner2]);
        assert_eq!(
            v, expected,
            "{{ {{x}} : x in {{1,2}} }} must equal {{{{1}},{{2}}}}"
        );

        clear_for_test_reset();
    }

    #[test]
    fn unobservable_chain_binding_is_not_cached_as_constant() {
        // Regression for the pointer-keyed-cache-aliasing soundness gate: in the
        // O(1) Init-enumeration shape a binding is consed onto the chain WITHOUT
        // a matching `binding_depth++`, so the dep rule (which records deps only
        // for chain index < binding_depth) cannot observe it. Here `{x}` depends
        // on such an invisible `x`; without the fail-safe it would be
        // misclassified as a constant set and cached, then the span-keyed cache
        // would serve that stale {7} for a later iteration where x differs.
        clear_for_test_reset();
        let body = lower_body("{x}");
        let mut ctx = EvalCtx::new();
        ctx.push_binding("x".into(), Value::int(7)); // chain: [x=7], depth -> 1
        ctx.binding_depth = 0; // simulate the Init-enumeration desync (depth < chain)
        assert!(
            ctx.has_unobservable_chain_bindings(),
            "test precondition: x must be invisible to the dep rule"
        );

        // Value is still correct — the chain lookup is depth-independent.
        let v = eval(&ctx, &body).expect("eval ok");
        assert_eq!(
            v,
            Value::set(vec![Value::int(7)]),
            "{{x}} with x=7 is {{7}}"
        );

        // The fail-safe: `{x}` must NOT be cached as a constant despite its dep
        // on x being invisible (and must not be permanently marked non-const).
        assert_eq!(
            const_set_cache_len(),
            0,
            "a node with an unobservable chain binding must not be cached as constant"
        );
        assert_eq!(
            const_set_non_const_len(),
            0,
            "the desync is transient; the node must not be permanently marked non-const"
        );
        clear_for_test_reset();
    }

    #[test]
    fn disabled_gate_bypasses_cache_but_value_unchanged() {
        // The gate is read once via OnceLock; we cannot toggle it mid-process.
        // Instead, assert that with caching ON (default) repeated evaluation of a
        // constant range produces the same value as a single evaluation — the
        // observable behavior is invariant to caching.
        clear_for_test_reset();
        let body = lower_body("1 .. 5");
        let ctx = EvalCtx::new();
        let a = eval(&ctx, &body).expect("eval ok");
        let b = eval(&ctx, &body).expect("eval ok");
        let c = eval(&ctx, &body).expect("eval ok");
        assert_eq!(a, b);
        assert_eq!(b, c);
        clear_for_test_reset();
    }
}
