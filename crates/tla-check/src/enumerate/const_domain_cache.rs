// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Constant quantifier/set-builder domain cache.
//!
//! TLA+ next-state / invariant evaluation re-enumerates quantifier and
//! set-builder domains (`\A x \in S`, `\E x \in S`, `x \in S`) once per state.
//! When the domain expression `S` is *state-independent and capture-free* — its
//! value depends only on CONSTANTS and constant operators, never on a state
//! variable (primed or unprimed) nor on an enclosing quantifier/LET-bound
//! variable — it produces the **same set value** on every evaluation. Rebuilding
//! it every state is pure waste: profiling (TY_SETPROF) showed btree rebuilding
//! 3 constant domains 328k times (100% redundant) and Disruptor_MPMC 4 domains
//! 310k times (100% redundant); set construction is the largest interpreter CPU
//! bucket (~22% on Disruptor).
//!
//! This cache memoizes the evaluated set value for such provably-constant
//! domains, built once per run and reused across all states.
//!
//! ## Soundness
//!
//! A set value *is* its content. A domain expression whose free variables are
//! all CONSTANTS (no state-var refs — checked transitively through operator
//! bodies — and no reference into the current local scope: no quantifier/param
//! binding, no LET-defined operator, no local operator) yields the identical
//! value on every evaluation, so caching it is observationally equivalent to
//! re-evaluating it. The independence test is FAIL-SAFE: any domain that cannot
//! be *proven* constant is marked `Dynamic` and always rebuilt. We never change
//! set equality / ordering / fingerprinting / interning semantics — only avoid
//! recomputing an identical value.
//!
//! ### Why the cacheability decision is stable across evals of a span
//!
//! The cache key is the domain AST node identity (plus the owning `SharedCtx`
//! to discriminate runs). A given AST node occupies a fixed lexical position, so
//! its enclosing quantifier/LET binders are the same on every evaluation. Hence
//! `lookup_binding` reflects the same set of captured names every time, and the
//! state-var dependency is a static property of the (immutable) AST. The
//! decision computed on the first hit therefore holds for all subsequent hits.
//!
//! ## Lifetime / thread-safety
//!
//! The cache is thread-local (like `subst_cache` and `expr_analysis` caches) and
//! keyed by `(SharedCtx::id, node ptr, node span)` — the span is fail-closed
//! hardening against allocator address reuse (see `CacheKey`). Correct for
//! `--workers 1`; safe for
//! `workers > 1` because each worker independently proves+caches the same
//! constant value (the cached `Value` is run-interned and content-identical).
//! Cleared on run/test reset via `clear_const_domain_cache`.

use std::cell::RefCell;

use rustc_hash::FxHashMap;
use tla_core::ast::Expr;
use tla_core::Spanned;

use crate::eval::EvalCtx;
use crate::Value;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct CacheKey {
    shared_id: u64,
    node_ptr: usize,
    /// Source span of the domain node. Fail-closed hardening against
    /// allocator address reuse (the P0 stale-replay family): if an ephemeral
    /// AST clone ever reaches this cache and its freed address is recycled by
    /// a DIFFERENT node, the span mismatch turns the stale hit into a miss.
    /// (Observed on PaxosCommit: per-state LET-def body clones let `Ballot`'s
    /// cached domain value replay for `Majority`'s node — now also fixed at
    /// the source by run-pinned LET-def interning in tla-eval.)
    span: tla_core::Span,
}

#[derive(Clone)]
enum Entry {
    /// Domain proven state-independent and capture-free; value memoized.
    Const(Value),
    /// Domain proven (or assumed) to vary; always rebuild. The independence
    /// proof failed, so we never attempt it again for this node.
    Dynamic,
}

/// Result of a single cache probe.
enum Probe {
    /// Proven constant; cached value (cheap Arc clone for set-like values).
    Const(Value),
    /// Proven dynamic; rebuild without re-proving.
    Dynamic,
    /// Not yet classified.
    Unknown,
}

thread_local! {
    static CACHE: RefCell<FxHashMap<CacheKey, Entry>> = RefCell::new(FxHashMap::default());
}

#[inline]
fn key_of(ctx: &EvalCtx, domain: &Spanned<Expr>) -> CacheKey {
    CacheKey {
        shared_id: ctx.shared().id,
        node_ptr: domain as *const Spanned<Expr> as usize,
        span: domain.span,
    }
}

/// Single cache probe (one thread-local borrow + one HashMap lookup).
#[inline]
fn probe(ctx: &EvalCtx, domain: &Spanned<Expr>) -> Probe {
    let key = key_of(ctx, domain);
    CACHE.with(|cache| match cache.borrow().get(&key) {
        Some(Entry::Const(v)) => Probe::Const(v.clone()),
        Some(Entry::Dynamic) => Probe::Dynamic,
        None => Probe::Unknown,
    })
}

/// Record a classification for `domain`.
#[inline]
fn store(ctx: &EvalCtx, domain: &Spanned<Expr>, entry: Entry) {
    let key = key_of(ctx, domain);
    CACHE.with(|cache| {
        cache.borrow_mut().insert(key, entry);
    });
}

/// Prove whether `domain` is a state-independent, capture-free expression.
///
/// Returns `true` only when BOTH hold:
/// 1. No state variable is referenced (transitively through operator bodies),
///    checked via [`super::collect_state_var_refs`]. This also rejects primed
///    state vars, since the prime wraps a state-var reference.
/// 2. No free identifier of the domain resolves into the current LOCAL scope —
///    no quantifier/param binding, no LET-defined operator, no local operator
///    (checked via [`EvalCtx::name_in_local_scope`]). Such names may capture an
///    enclosing bound variable and so are not state-independent. Everything else
///    a domain can reference — CONSTANTS and module-level (shared) operators —
///    is lexically outside every quantifier and fixed for the run, hence safe.
///
/// FAIL-SAFE: if either property cannot be confirmed, returns `false`.
pub(super) fn is_constant_domain(
    ctx: &EvalCtx,
    domain: &Spanned<Expr>,
    state_vars: &[std::sync::Arc<str>],
) -> bool {
    // (1) No state-variable dependency (transitive through operators).
    let mut refs: rustc_hash::FxHashSet<std::sync::Arc<str>> = rustc_hash::FxHashSet::default();
    super::collect_state_var_refs(ctx, domain, state_vars, &mut refs);
    if !refs.is_empty() {
        return false;
    }

    // (2) No locally-scoped reference. Any free identifier resolving to the
    // current local scope — a quantifier/param binding, a LET-defined operator,
    // or a local operator — may capture an enclosing bound variable and so is
    // not state-independent. Module-level (shared) operators and CONSTANTS are
    // lexically outside every quantifier, hence cannot capture; they are safe.
    // FAIL-SAFE: anything in local scope rejects the domain (rebuild every time).
    for v in tla_core::free_vars(&domain.node) {
        if ctx.name_in_local_scope(&v) {
            return false;
        }
    }

    true
}

/// Evaluate a quantifier/set-builder DOMAIN, reusing the cached value when the
/// domain is provably state-independent and capture-free.
///
/// Fast path (cache hit on a proven-constant domain): O(1) `Value` clone (an Arc
/// refcount bump for set-like values), skipping domain re-construction entirely.
///
/// Slow path (first sight, or proven-dynamic domain): evaluates via `eval_leaf`
/// exactly as before. On first sight, runs the independence proof and records the
/// classification so future evaluations take the fast path or skip re-proving.
///
/// Soundness: identical observable result to `eval_leaf`, because a proven-
/// constant domain yields the same value on every evaluation. See module docs.
#[inline]
pub(super) fn eval_domain_cached(
    ctx: &EvalCtx,
    domain: &Spanned<Expr>,
    tir: Option<&tla_eval::tir::TirProgram<'_>>,
    state_vars: &[std::sync::Arc<str>],
) -> tla_value::error::EvalResult<Value> {
    // Single cache probe classifies this domain node.
    let classified = match probe(ctx, domain) {
        // Fast path: proven constant — return the cached value (Arc clone).
        Probe::Const(v) => return Ok(v),
        // Proven dynamic: rebuild, no re-proving, no re-store.
        Probe::Dynamic => true,
        // Unknown: rebuild and classify after.
        Probe::Unknown => false,
    };

    let result = super::tir_leaf::eval_leaf(ctx, domain, tir);

    if let Ok(value) = &result {
        // Classify only on first sight (Unknown). An error is not a stable value
        // to cache, so we only classify when eval succeeded.
        if !classified {
            if is_constant_domain(ctx, domain, state_vars) {
                store(ctx, domain, Entry::Const(value.clone()));
            } else {
                store(ctx, domain, Entry::Dynamic);
            }
        }
    }

    result
}

/// Clear the thread-local constant-domain cache.
///
/// Must be called between model-checking runs. Although keys include `shared_id`
/// for run discrimination, clearing prevents unbounded growth across specs.
pub(crate) fn clear_const_domain_cache() {
    CACHE.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::span::Spanned;

    fn clear() {
        clear_const_domain_cache();
    }

    fn is_const(p: Probe) -> Option<Value> {
        match p {
            Probe::Const(v) => Some(v),
            _ => None,
        }
    }

    #[test]
    fn probe_miss_then_store_const_hit() {
        clear();
        let ctx = EvalCtx::new();
        let node = Spanned::dummy(Expr::Bool(true));
        assert!(matches!(probe(&ctx, &node), Probe::Unknown));
        store(&ctx, &node, Entry::Const(Value::int(7)));
        assert_eq!(is_const(probe(&ctx, &node)), Some(Value::int(7)));
    }

    #[test]
    fn store_dynamic_marks_node() {
        clear();
        let ctx = EvalCtx::new();
        let node = Spanned::dummy(Expr::Bool(false));
        store(&ctx, &node, Entry::Dynamic);
        assert!(matches!(probe(&ctx, &node), Probe::Dynamic));
    }

    #[test]
    fn discriminates_shared_context_runs() {
        clear();
        let ctx1 = EvalCtx::new();
        let ctx2 = EvalCtx::new();
        let node = Spanned::dummy(Expr::Bool(true));
        assert_ne!(ctx1.shared().id, ctx2.shared().id);
        store(&ctx1, &node, Entry::Const(Value::int(1)));
        // Same node ptr, different run: must not leak.
        assert!(matches!(probe(&ctx2, &node), Probe::Unknown));
        assert_eq!(is_const(probe(&ctx1, &node)), Some(Value::int(1)));
    }
}
