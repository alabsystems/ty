// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Small caches: LITERAL_CACHE and THUNK_DEP_CACHE.
//!
//! These two caches are too small for individual files but don't belong in
//! lifecycle.rs (they define storage, not orchestration).
//!
//! Part of #2744 decomposition from eval_cache.rs.

use super::dep_tracking::{propagate_cached_deps, OpEvalDeps};
use super::op_result_cache::NaryOpCacheKey;
use crate::tlc_types::TlcActionContext;
use crate::value::Value;
use crate::EvalCtx;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;
use tla_core::ast::Substitution;
use tla_core::name_intern::NameId;
use tla_core::Span;

/// Scope-discriminated key for LET caches.
///
/// Fix #3095, Fix #3114: LET caches were previously keyed by AST body pointer alone.
/// This caused cross-scope aliasing when the same LET body was evaluated under different
/// INSTANCE/local-op scopes (e.g., promoted PROPERTY evaluation vs action enumeration).
/// Adding `local_ops_id` and `instance_subs_id` mirrors the scope discrimination already
/// accepted for ZERO_ARG_OP_CACHE (#3097) and NaryOpCacheKey (#3024).
///
/// Fields:
/// - `body_ptr`: AST pointer identity (stable because AST is immutable after parsing)
/// - `local_ops_id`: content-based u64 fingerprint from `scope_ids`
/// - `instance_subs_id`: content-based u64 fingerprint from `scope_ids`
///
/// Part of #3099/#3164: `local_ops_id` and `instance_subs_id` migrated from
/// `Arc::as_ptr() as usize` to stable content-based u64 fingerprints, matching
/// SubstCacheKey, NaryOpCacheKey, and ZeroArgCacheKey.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LetScopeKey {
    pub(crate) body_ptr: usize,
    pub(crate) local_ops_id: u64,
    pub(crate) instance_subs_id: u64,
}

/// Build a LET-scope key from the evaluation context and AST body pointer.
///
/// Part of #3099/#3164: Uses content-based u64 fingerprints from `scope_ids`
/// for `local_ops_id` and `instance_subs_id`, matching `zero_arg_cache_key()`
/// and `NaryOpCacheKey`. Replaces the previous `Arc::as_ptr` pointer-identity
/// convention which was vulnerable to allocator address reuse.
#[inline]
pub(crate) fn let_scope_key(ctx: &EvalCtx, body_ptr: usize) -> LetScopeKey {
    use super::scope_ids::{resolve_instance_subs_id, resolve_local_ops_id_with_recursive};
    let local_ops_id = resolve_local_ops_id_with_recursive(
        ctx.scope_ids.local_ops,
        &ctx.local_ops,
        ctx.scope_ids.local_ops_recursive,
    );
    let instance_subs_id = resolve_instance_subs_id(
        ctx.scope_ids.instance_substitutions,
        &ctx.instance_substitutions,
    );
    LetScopeKey {
        body_ptr,
        local_ops_id,
        instance_subs_id,
    }
}

/// Entry: (dep_hash, dep_values, cached_result). Part of #3034.
pub(crate) type ParamLetCacheEntry = (u64, Vec<Value>, Value);

/// Key for the raw SubstIn scope cache.
///
/// Identifies a raw `Expr::SubstIn` evaluation site by its AST location and
/// enclosing scope identity. `subs_ptr` and `body_ptr` identify the AST node;
/// `outer_local_ops_id` and `outer_instance_subs_id` disambiguate distinct
/// enclosing resolution contexts.
///
/// Part of #3099/#3164: Scope ids migrated from `Arc::as_ptr() as usize` to
/// content-based u64 fingerprints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RawSubstScopeKey {
    pub(crate) subs_ptr: usize,
    pub(crate) body_ptr: usize,
    pub(crate) outer_local_ops_id: u64,
    pub(crate) outer_instance_subs_id: u64,
}

// Part of #3962: Consolidated small-cache thread-locals into a single struct.
// Previously 11 separate thread_local! declarations, each requiring a separate
// `_tlv_get_addr` call on macOS. Now a single TLS access covers all small caches.
// Wave 23: Added fold_result_cache (from helpers/recursive_fold.rs) and
// action_ctx_cache (from eval_ctx_ops/mutation.rs) — 2 more thread_locals eliminated.
//
// Lifecycle notes (preserved from individual declarations):
// - literal_cache: PerRun, soft cap 50K
// - thunk_dep_cache: PerState (cleared in clear_state_boundary_core_impl)
// - thunk_taint_set: PerRun (survives scope/state boundary clears)
// - const_let_cache: PerRun, soft cap 50K
// - non_const_let_set: PerRun
// - param_let_deps: PerRun
// - param_let_cache: PerRun
// - raw_subst_scope_cache: PerRun, soft cap 10K
// - instance_lazy_cache: PerState (cleared in clear_state_boundary_core_impl)
// - choose_cache: PerState (cleared in clear_state_boundary_core_impl)
// - choose_deep_cache: PerState, soft cap 100K
// - fold_result_cache: PerRun, soft cap 100K (from helpers/recursive_fold.rs)
// - action_ctx_cache: PerRun (from eval_ctx_ops/mutation.rs, Part of #3364)
pub(crate) struct SmallCaches {
    pub(crate) literal_cache: FxHashMap<Span, Value>,
    pub(crate) thunk_dep_cache: FxHashMap<u64, OpEvalDeps>,
    pub(crate) thunk_taint_set: FxHashSet<u64>,
    pub(crate) const_let_cache: FxHashMap<LetScopeKey, Value>,
    pub(crate) non_const_let_set: FxHashSet<LetScopeKey>,
    pub(crate) param_let_deps: FxHashMap<LetScopeKey, Vec<NameId>>,
    pub(crate) param_let_cache: FxHashMap<LetScopeKey, Vec<ParamLetCacheEntry>>,
    pub(crate) raw_subst_scope_cache: FxHashMap<RawSubstScopeKey, (Arc<Vec<Substitution>>, u64)>,
    /// Forced INSTANCE lazy-binding results, keyed by `LazyBinding` address.
    /// Entry: `(mode, value, deps captured during that forcing)`. The deps
    /// ride the SAME epoch lifecycle as the value (cleared at every state and
    /// scope boundary) because `OpEvalDeps` embeds read *values* — they must
    /// never outlive the epoch that produced them. This replaced the
    /// per-`LazyBinding` `forced_deps` OnceLocks so that the subst-chain memo
    /// (`cache::subst_chain_memo`) can share `LazyBinding` allocations across
    /// states without leaking first-force deps into later states.
    pub(crate) instance_lazy_cache: FxHashMap<usize, (u8, Value, Option<Box<OpEvalDeps>>)>,
    pub(crate) choose_cache: FxHashMap<(usize, u64, usize), Value>,
    pub(crate) choose_deep_cache: FxHashMap<ChooseDeepKey, Value>,
    /// Part of #3962: Consolidated from helpers/recursive_fold.rs FOLD_RESULT_CACHE.
    /// Memoization cache for pure recursive fold results.
    /// Key: (operator def pointer, evaluated argument values).
    /// Cap at 100K entries to bound memory; cleared entirely when exceeded.
    pub(crate) fold_result_cache: FxHashMap<(usize, Vec<Value>), Value>,
    /// Memoization cache for linear/chain-recursive operators (e.g. NanoBlockchain's
    /// `PublicKeyOf` / `BalanceAt` / `ValueOfSendBlock`, which walk a ledger chain via
    /// `block.previous`). Unlike `fold_result_cache` these are NOT set-folds, so the
    /// fold path returns `None` and they otherwise re-walk the whole chain on every
    /// call (O(D) per call × O(D) blocks = O(D²) per state).
    ///
    /// Key: the established n-ary operator context key plus exact evaluated argument
    /// values. The context key discriminates shared evaluator identity, local-operator
    /// and INSTANCE-substitution scopes, resolved operator identity/location,
    /// current-vs-next lookup mode, and parametrized INSTANCE arguments. The exact
    /// values resolve argument-hash collisions.
    ///
    /// An entry is inserted ONLY after runtime dependency tracking proves the
    /// operator read NO state variable and the returned value does not capture a
    /// state environment. Such results are valid for the whole run, so this is a
    /// PerRun (process-long) cache. Cap at 131072 entries; cleared entirely when
    /// exceeded and on run reset.
    pub(crate) recursive_result_cache: FxHashMap<(NaryOpCacheKey, Vec<Value>), Value>,
    /// Part of #3962: Consolidated from eval_ctx_ops/mutation.rs ACTION_CTX_CACHE.
    /// Thread-local cache for `TlcActionContext` per `OperatorDef` pointer.
    /// Part of #3364: avoids rebuilding context on every action evaluation.
    pub(crate) action_ctx_cache: FxHashMap<usize, Arc<TlcActionContext>>,
    /// `Arc`-shared TIR body for each `LAMBDA` node, keyed by the node's stable
    /// TIR body pointer. A `LAMBDA` evaluated to a `ClosureValue` needs an
    /// `Arc<Spanned<TirExpr>>` to attach as the closure's TIR body; without this
    /// cache every construction deep-clones the entire body (SlidingPuzzles
    /// builds its `ChooseOne` lambda ~once per `move` — ~180K deep clones of the
    /// same immutable body). The cache clones once per distinct lambda body and
    /// hands out `Arc` bumps thereafter.
    ///
    /// SOUNDNESS: the key is the raw body pointer, stable while the immutable TIR
    /// program is alive; the value is a content-identical `Arc` of that body, so
    /// evaluation is byte-identical. Address reuse across independent runs is
    /// prevented by clearing this cache on run reset (see `clear_run_reset_impl`)
    /// — the exact contract already relied on by `action_ctx_cache` and
    /// `instance_lazy_cache`, which are also raw-pointer-keyed and PerRun.
    pub(crate) lambda_tir_body_cache:
        FxHashMap<usize, Arc<tla_core::Spanned<tla_tir::nodes::TirExpr>>>,
}

impl SmallCaches {
    fn new() -> Self {
        SmallCaches {
            literal_cache: FxHashMap::default(),
            thunk_dep_cache: FxHashMap::default(),
            thunk_taint_set: FxHashSet::default(),
            const_let_cache: FxHashMap::default(),
            non_const_let_set: FxHashSet::default(),
            param_let_deps: FxHashMap::default(),
            param_let_cache: FxHashMap::default(),
            raw_subst_scope_cache: FxHashMap::default(),
            instance_lazy_cache: FxHashMap::default(),
            choose_cache: FxHashMap::default(),
            choose_deep_cache: FxHashMap::default(),
            fold_result_cache: FxHashMap::default(),
            recursive_result_cache: FxHashMap::default(),
            action_ctx_cache: FxHashMap::default(),
            lambda_tir_body_cache: FxHashMap::default(),
        }
    }
}

/// Return the `Arc`-shared TIR body for a `LAMBDA` node, cloning-and-caching it
/// on first sight and handing out `Arc` bumps thereafter.
///
/// `body_ptr` MUST be the address of the same immutable `Spanned<TirExpr>` that
/// `clone_body()` would produce, so a hit returns a content-identical body. See
/// [`SmallCaches::lambda_tir_body_cache`] for the soundness contract.
#[inline]
pub(crate) fn lambda_tir_body_arc(
    body_ptr: usize,
    clone_body: impl FnOnce() -> Arc<tla_core::Spanned<tla_tir::nodes::TirExpr>>,
) -> Arc<tla_core::Spanned<tla_tir::nodes::TirExpr>> {
    SMALL_CACHES.with(|sc| {
        if let Some(existing) = sc.borrow().lambda_tir_body_cache.get(&body_ptr) {
            return Arc::clone(existing);
        }
        let arc = clone_body();
        sc.borrow_mut()
            .lambda_tir_body_cache
            .insert(body_ptr, Arc::clone(&arc));
        arc
    })
}

std::thread_local! {
    pub(crate) static SMALL_CACHES: std::cell::RefCell<SmallCaches> =
        std::cell::RefCell::new(SmallCaches::new());
}

/// Clear the per-state-scoped LET / fold memoization caches.
///
/// These caches (`param_let_cache`, `param_let_deps`, `non_const_let_set`,
/// `const_let_cache`, `fold_result_cache`) are keyed by AST/scope identity plus
/// dependency values and validated on lookup, so they are *correct* as
/// process-long PerRun caches. But on specs that evaluate a LET / recursive
/// fold whose body depends (directly or via a nested operator) on the current
/// state, each distinct state produces a fresh key/entry that is never reused
/// — the cache only grows. Clearing them at a per-state boundary bounds the
/// growth to one state's working set; it can only drop memoized entries and so
/// never changes any value, verdict, or state count.
pub fn clear_state_scoped_let_and_fold_caches() {
    SMALL_CACHES.with(|sc| {
        let mut sc = sc.borrow_mut();
        sc.param_let_cache.clear();
        sc.param_let_deps.clear();
        sc.non_const_let_set.clear();
        sc.const_let_cache.clear();
        sc.fold_result_cache.clear();
    });
}

/// Build a raw-SubstIn scope key from the evaluation context and AST pointers.
///
/// Part of #3099/#3164: Uses content-based u64 fingerprints from `scope_ids`.
#[inline]
pub(crate) fn raw_subst_scope_key(
    ctx: &EvalCtx,
    subs_ptr: usize,
    body_ptr: usize,
) -> RawSubstScopeKey {
    use super::scope_ids::{resolve_instance_subs_id, resolve_local_ops_id_with_recursive};
    let outer_local_ops_id = resolve_local_ops_id_with_recursive(
        ctx.scope_ids.local_ops,
        &ctx.local_ops,
        ctx.scope_ids.local_ops_recursive,
    );
    let outer_instance_subs_id = resolve_instance_subs_id(
        ctx.scope_ids.instance_substitutions,
        &ctx.instance_substitutions,
    );
    RawSubstScopeKey {
        subs_ptr,
        body_ptr,
        outer_local_ops_id,
        outer_instance_subs_id,
    }
}

/// Key for the deep CHOOSE cache (binding_depth > 0).
///
/// Defense-in-depth: includes state_identity (state_env pointer) so stale
/// entries from a different state will not match even if cache clearing
/// is missed (#3939).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ChooseDeepKey {
    pub(crate) expr_ptr: usize,
    pub(crate) instance_subs_id: u64,
    pub(crate) domain_hash: u64,
    pub(crate) bindings_hash: u64,
    pub(crate) state_identity: usize,
}

/// Store forced-thunk deps for later propagation on cache hit.
/// Fix #1498: Called on first force of a lazy thunk during dep tracking.
pub(crate) fn store_thunk_deps(closure_id: u64, deps: &OpEvalDeps) {
    // Part of #3962: Single TLS access for both thunk_dep_cache and thunk_taint_set.
    SMALL_CACHES.with(|sc| {
        let mut sc = sc.borrow_mut();
        // Part of #3465: Record INSTANCE taint in persistent set that survives
        // scope boundary clears. Once tainted, always tainted for this run.
        if deps.instance_lazy_read {
            sc.thunk_taint_set.insert(closure_id);
        }
        sc.thunk_dep_cache.insert(closure_id, deps.clone());
    });
}

/// Propagate stored thunk deps into the current dep tracking frame.
/// Fix #1498: Called on cache-hit path of force_lazy_thunk_if_needed.
/// Returns silently if no deps are stored or if not inside dep tracking.
pub(crate) fn propagate_thunk_deps(ctx: &EvalCtx, closure_id: u64) {
    SMALL_CACHES.with(|sc| {
        let sc = sc.borrow();
        if let Some(deps) = sc.thunk_dep_cache.get(&closure_id) {
            propagate_cached_deps(ctx, deps);
        }
    });
}

/// Store a forced INSTANCE lazy binding value for within-state reuse.
/// Deps captured during that forcing are attached separately via
/// [`instance_lazy_cache_set_deps`] (the force path stores the value before
/// its dep guard resolves).
#[inline]
pub(crate) fn instance_lazy_cache_store(lazy_ptr: usize, mode_discriminant: u8, value: Value) {
    SMALL_CACHES.with(|sc| {
        sc.borrow_mut()
            .instance_lazy_cache
            .insert(lazy_ptr, (mode_discriminant, value, None));
    });
}

/// Attach the deps captured while forcing an INSTANCE lazy binding to its
/// cache entry (no-op if the entry is gone or the mode changed — the deps
/// then simply aren't reused, and cache hits fall back to conservative
/// tainting). Epoch-scoped by construction: lives and dies with the entry.
#[inline]
pub(crate) fn instance_lazy_cache_set_deps(
    lazy_ptr: usize,
    mode_discriminant: u8,
    deps: OpEvalDeps,
) {
    SMALL_CACHES.with(|sc| {
        if let Some(entry) = sc.borrow_mut().instance_lazy_cache.get_mut(&lazy_ptr) {
            if entry.0 == mode_discriminant {
                entry.2 = Some(Box::new(deps));
            }
        }
    });
}

/// Fused lookup: cached value AND (optionally) its stored deps in ONE TLS
/// access. When `want_deps` is true and an entry matches, `use_deps` runs
/// under the borrow with the entry's deps (`None` = forcing captured none —
/// callers taint conservatively). `use_deps` must not touch `SMALL_CACHES`;
/// the dep-propagation callers only touch `ctx.runtime_state`.
///
/// This exists so the INSTANCE lazy hit path pays exactly ONE map lookup —
/// the same count as before deps moved from the per-LazyBinding OnceLock
/// into the cache entry (a second lookup per hit measurably regressed
/// MCNanoMedium).
#[inline]
pub(crate) fn instance_lazy_cache_get_with_deps(
    lazy_ptr: usize,
    mode_discriminant: u8,
    want_deps: bool,
    use_deps: impl FnOnce(Option<&OpEvalDeps>),
) -> Option<Value> {
    SMALL_CACHES.with(|sc| {
        let sc = sc.borrow();
        let (m, v, d) = sc.instance_lazy_cache.get(&lazy_ptr)?;
        if *m != mode_discriminant {
            return None;
        }
        if want_deps {
            use_deps(d.as_deref());
        }
        Some(v.clone())
    })
}

/// Look up a cached CHOOSE result for the given expression pointer and INSTANCE scope.
#[inline]
pub(crate) fn choose_cache_lookup(
    expr_ptr: usize,
    instance_subs_id: u64,
    state_identity: usize,
) -> Option<Value> {
    SMALL_CACHES
        .with(|sc| {
            sc.borrow()
                .choose_cache
                .get(&(expr_ptr, instance_subs_id, state_identity))
                .cloned()
        })
        .inspect(|_| {
            tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::ChooseCacheHit);
        })
}

/// Store a CHOOSE result for within-state reuse.
///
/// Only called when binding_depth == 0 to ensure soundness.
#[inline]
pub(crate) fn choose_cache_store(
    expr_ptr: usize,
    instance_subs_id: u64,
    state_identity: usize,
    value: Value,
) {
    SMALL_CACHES.with(|sc| {
        sc.borrow_mut()
            .choose_cache
            .insert((expr_ptr, instance_subs_id, state_identity), value);
    });
}

/// Look up a cached CHOOSE result for a deep (binding_depth > 0) invocation.
///
/// Part of #3905: Extends CHOOSE caching to inside quantifiers.
#[inline]
pub(crate) fn choose_deep_cache_lookup(key: &ChooseDeepKey) -> Option<Value> {
    SMALL_CACHES
        .with(|sc| sc.borrow().choose_deep_cache.get(key).cloned())
        .inspect(|_| {
            tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::ChooseCacheHit);
        })
}

/// Store a CHOOSE result for within-state reuse at binding_depth > 0.
///
/// Part of #3905: Cap at 100K entries to bound memory.
#[inline]
pub(crate) fn choose_deep_cache_store(key: ChooseDeepKey, value: Value) {
    SMALL_CACHES.with(|sc| {
        let mut c = sc.borrow_mut();
        if c.choose_deep_cache.len() > 100_000 {
            c.choose_deep_cache.clear();
        }
        c.choose_deep_cache.insert(key, value);
    });
}

/// Compute a hash of local binding values for CHOOSE deep cache keying.
///
/// Part of #3905: Hashes all Local/Liveness eager binding (name, value) pairs
/// in the binding chain. This captures the quantifier variable context that
/// a CHOOSE predicate body might reference.
#[inline]
pub(crate) fn hash_bindings_for_choose(ctx: &crate::EvalCtx) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    let mut count: u32 = 0;
    for (name_id, value, source) in ctx.bindings.iter() {
        match source {
            crate::binding_chain::BindingSourceRef::Local(_)
            | crate::binding_chain::BindingSourceRef::Liveness(_) => {}
            _ => continue,
        }
        if let crate::binding_chain::BindingValueRef::Eager(cv) = value {
            name_id.hash(&mut hasher);
            cv.hash(&mut hasher);
            count += 1;
        }
    }
    count.hash(&mut hasher);
    hasher.finish()
}

/// Compute a hash of a Value for use as a CHOOSE cache key component.
///
/// Part of #3905: Used to hash the evaluated domain value.
#[inline]
pub(crate) fn hash_value_for_choose(value: &Value) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Fix #3447: Check whether stored thunk deps carry the INSTANCE lazy-read taint.
/// Returns true if the closure's forced deps had `instance_lazy_read`, meaning the
/// OnceLock cached value is state-dependent and must not be reused across states.
pub(crate) fn thunk_has_instance_lazy_taint(closure_id: u64) -> bool {
    // Part of #3465/#3962: Use persistent thunk_taint_set from SMALL_CACHES.
    // thunk_dep_cache is cleared at scope/state boundaries, losing taint info.
    // thunk_taint_set survives those clears — once a closure is marked tainted,
    // it stays tainted for the entire run.
    SMALL_CACHES.with(|sc| sc.borrow().thunk_taint_set.contains(&closure_id))
}
