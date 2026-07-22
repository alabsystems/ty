// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Subscript expression evaluation and caching for liveness checking.
//!
//! Provides both the subscript value cache and the `eval_subscript_changed`
//! function that determines whether a subscript expression changed between two
//! states. Co-locating cache and evaluation eliminates redundant evaluations
//! across transitions that share the same state. For EWD998ChanID with WF_vars(A),
//! all fairness conditions share the same subscript expression `vars`, so each
//! unique state is evaluated once instead of once per transition × check. This
//! keeps SUBST_CACHE warm for ActionPred evaluations by avoiding state_env=None
//! context switches from with_explicit_env (#2364).

use super::cache_stats::{record_subscript_eviction, record_subscript_hit, record_subscript_miss};
use crate::error::EvalResult;
use crate::eval::{BindingChain, EvalCtx};
use crate::state::{value_fingerprint, ArrayState, Fingerprint, State};
use rustc_hash::FxHashMap;
use std::cell::{Cell, RefCell};
use std::sync::Arc;

/// Soft cap for SUBSCRIPT_VALUE_CACHE entries. When exceeded, retain-half
/// eviction removes roughly half the entries using HashMap's pseudo-random
/// iteration order. Prevents unbounded linear growth with state count (#4083).
///
/// Set to 5M to avoid cache thrashing on specs with many fairness tags.
/// AllocatorImplementation has 115 WF conditions × 17,701 states = ~2M entries;
/// at the old 50K cap, 24M evictions caused 59s overhead (>95% of liveness time).
/// Entries store an 8-byte content fingerprint (see below), so 5M entries ≈ 40MB.
/// The precompute phase fills the cache for all (state, tag) pairs, and
/// subsequent lookups during the eval loop must find them without eviction.
const SUBSCRIPT_VALUE_CACHE_SOFT_CAP: usize = 5_000_000;

thread_local! {
    /// Per-(state fp, subscript tag) cache of the subscript value's CONTENT
    /// FINGERPRINT (`value_fingerprint`, the canonical FP64 used for state
    /// dedup), NOT the full `Value`.
    ///
    /// Every consumer of this cache uses the subscript value ONLY for equality
    /// comparison — `StateChanged`/`Enabled` change-detection asks "did the
    /// subscript value differ between two states?" and never inspects the value
    /// structurally. Equal values always fingerprint equal (the dedup contract),
    /// and distinct values collide with probability ~2^-64 — exactly the trust
    /// the BFS dedup already places in these fps. So a fingerprint is a sound
    /// stand-in for the value and avoids Arc-retaining the (often whole-`vars`,
    /// sequence-valued) subscript value for every reachable state — which
    /// dominated liveness peak RSS on sequence-heavy specs (e.g. nbacg_guer01).
    static SUBSCRIPT_VALUE_CACHE: RefCell<FxHashMap<(Fingerprint, u32), u64>> =
        RefCell::new(FxHashMap::default());
    /// Track whether we have already emitted the first-eviction warning.
    static SUBSCRIPT_EVICTION_WARNED: Cell<bool> = const { Cell::new(false) };
    /// Part of #liveness-leaf-memo: leaf tag → canonical subscript class.
    /// Tags whose subscript expressions compute the SAME pure function of the
    /// state (see [`register_subscript_tag_classes`]) are mapped to one shared
    /// cache key (the smallest member tag), so the per-state subscript value
    /// is evaluated once per CLASS instead of once per leaf tag. Empty map =
    /// identity (every tag keys its own entries — the historical behavior).
    static SUBSCRIPT_TAG_CLASS: RefCell<FxHashMap<u32, u32>> =
        RefCell::new(FxHashMap::default());
}

/// Census probe (TY_MEM_CENSUS): current entry count of the subscript cache.
pub(crate) fn census_subscript_cache_len() -> usize {
    SUBSCRIPT_VALUE_CACHE.with(|c| c.borrow().len())
}

/// Map a leaf tag to its canonical subscript-class cache key.
#[inline]
fn subscript_class_for(tag: u32) -> u32 {
    SUBSCRIPT_TAG_CLASS.with(|m| m.borrow().get(&tag).copied().unwrap_or(tag))
}

/// Clear the subscript value cache. Called at the start of populate_node_check_masks
/// and by `reset_global_state()` for between-run isolation.
///
/// Also clears the tag→class map: the map and the value cache must be
/// set/cleared together. A class map from a previous run applied to a new
/// run's (re-assigned) tag space would alias unrelated leaves, so any point
/// that invalidates cached values also drops the mapping (falling back to the
/// sound identity mapping until the next plan registration).
pub(crate) fn clear_subscript_value_cache() {
    SUBSCRIPT_VALUE_CACHE.with(|cache| cache.borrow_mut().clear());
    SUBSCRIPT_EVICTION_WARNED.with(|warned| warned.set(false));
    SUBSCRIPT_TAG_CLASS.with(|m| m.borrow_mut().clear());
}

/// Drop all backing allocations owned by the thread-local subscript caches.
///
/// This is stronger than [`clear_subscript_value_cache`]: ordinary property
/// boundaries keep capacity warm, while a regeneration trip disables the
/// inline caching path and must return that capacity to the allocator.
pub(crate) fn release_subscript_cache_storage() {
    SUBSCRIPT_VALUE_CACHE.with(|cache| *cache.borrow_mut() = FxHashMap::default());
    SUBSCRIPT_EVICTION_WARNED.with(|warned| warned.set(false));
    SUBSCRIPT_TAG_CLASS.with(|map| *map.borrow_mut() = FxHashMap::default());
}

/// Trim SUBSCRIPT_VALUE_CACHE if it exceeds the soft cap (#4083).
/// Uses retain-half eviction: keeps ~half the entries using FxHashMap's
/// pseudo-random iteration order, same pattern as eval-layer caches.
fn trim_subscript_value_cache_if_needed() {
    SUBSCRIPT_VALUE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let len = cache.len();
        if len > SUBSCRIPT_VALUE_CACHE_SOFT_CAP {
            // Log a warning on first eviction for monitoring (#4083).
            SUBSCRIPT_EVICTION_WARNED.with(|warned| {
                if !warned.get() {
                    eprintln!(
                        "[liveness] SUBSCRIPT_VALUE_CACHE exceeded soft cap ({} > {}), evicting",
                        len, SUBSCRIPT_VALUE_CACHE_SOFT_CAP
                    );
                    warned.set(true);
                }
            });
            let target = SUBSCRIPT_VALUE_CACHE_SOFT_CAP / 2;
            let mut kept = 0;
            cache.retain(|_, _| {
                if kept < target {
                    kept += 1;
                    true
                } else {
                    false
                }
            });
            record_subscript_eviction(len.saturating_sub(cache.len()) as u64);
        }
    });
}

/// Insert a pre-computed subscript value CONTENT FINGERPRINT into the cache.
/// Trims the cache via retain-half eviction if it exceeds the soft cap (#4083).
/// Keys are canonicalized through the subscript class map (#liveness-leaf-memo).
/// `value_fp` must be `value_fingerprint(&subscript_value)` — the same canonical
/// FP64 used for state dedup, so equal subscript values map to equal fps.
pub(super) fn set_subscript_fp_cache(fp: Fingerprint, tag: u32, value_fp: u64) {
    let tag = subscript_class_for(tag);
    trim_subscript_value_cache_if_needed();
    SUBSCRIPT_VALUE_CACHE.with(|cache| {
        cache.borrow_mut().insert((fp, tag), value_fp);
    });
}

/// Check if subscript values differ between two states using the pre-computed cache.
///
/// Returns `Some(true)` if values differ, `Some(false)` if equal, `None` if either
/// value is not in the cache (caller should fall back to expression evaluation).
pub(super) fn check_subscript_changed_cached(
    fp1: Fingerprint,
    fp2: Fingerprint,
    tag: u32,
) -> Option<bool> {
    let tag = subscript_class_for(tag);
    SUBSCRIPT_VALUE_CACHE.with(|cache| {
        let c = cache.borrow();
        let v1 = c.get(&(fp1, tag));
        let v2 = c.get(&(fp2, tag));
        match (v1, v2) {
            (Some(v1), Some(v2)) => {
                record_subscript_hit();
                record_subscript_hit();
                Some(v1 != v2)
            }
            _ => {
                if v1.is_some() {
                    record_subscript_hit();
                } else {
                    record_subscript_miss();
                }
                if v2.is_some() {
                    record_subscript_hit();
                } else {
                    record_subscript_miss();
                }
                None
            }
        }
    })
}

/// Retrieve a cached subscript value CONTENT FINGERPRINT for a given state
/// fingerprint and tag.
///
/// Returns `None` if the value is not in the cache.
pub(super) fn get_subscript_fp_cached(fp: Fingerprint, tag: u32) -> Option<u64> {
    let tag = subscript_class_for(tag);
    SUBSCRIPT_VALUE_CACHE.with(|cache| {
        let result = cache.borrow().get(&(fp, tag)).copied();
        if result.is_some() {
            record_subscript_hit();
        } else {
            record_subscript_miss();
        }
        result
    })
}

/// Evaluate whether the subscript expression changed between two states.
/// Returns true iff subscript(s1) ≠ subscript(s2)
///
/// When `cache_tag` is Some, caches individual subscript values by
/// (fingerprint, tag), eliminating redundant evaluations across transitions
/// that share the same state (#2364). This also preserves SUBST_CACHE
/// warmth for subsequent ActionPred evaluations by avoiding state_env=None
/// context switches when both values are cache hits.
pub(super) fn eval_subscript_changed(
    ctx: &EvalCtx,
    s1: &State,
    s2: &State,
    subscript: &tla_core::Spanned<tla_core::ast::Expr>,
    cache_tag: Option<u32>,
) -> EvalResult<bool> {
    let debug = super::debug_subscript();
    if debug {
        eprintln!(
            "[DEBUG SUBSCRIPT] Evaluating subscript: {:?}",
            subscript.node
        );
        eprintln!(
            "[DEBUG SUBSCRIPT] s1 vars: {:?}",
            s1.vars()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "[DEBUG SUBSCRIPT] s2 vars: {:?}",
            s2.vars()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<Vec<_>>()
        );
    }

    let fp1 = s1.fingerprint();
    let fp2 = s2.fingerprint();

    // Try cache for val1 (content fingerprint, not the value)
    let val1 = if let Some(h) = cache_tag.and_then(|tag| get_subscript_fp_cached(fp1, tag)) {
        h
    } else {
        // Fix #2780: Clear SUBST_CACHE before evaluating val1 via with_explicit_env
        // (state_env=None, pointer 0). A prior call's val2 evaluation may have left
        // entries keyed on the same pointer 0, causing stale hits here.
        crate::eval::clear_subst_cache();
        // Build environment from s1 (current state).
        //
        // Preserve base env bindings (constants/config overrides like `Node`) and
        // overlay state vars for this concrete state.
        let mut env1 = ctx.env().clone();
        for (name, value) in s1.vars() {
            // Part of #2144: skip state vars that shadow local bindings.
            if !ctx.has_local_binding(name.as_ref()) {
                env1.insert(Arc::clone(name), value.clone());
            }
        }
        let ctx1 = ctx.with_explicit_env(env1);
        let h = value_fingerprint(&crate::eval::eval_entry(&ctx1, subscript)?);
        if let Some(tag) = cache_tag {
            set_subscript_fp_cache(fp1, tag, h);
        }
        h
    };

    // Try cache for val2 (content fingerprint, not the value)
    let val2 = if let Some(h) = cache_tag.and_then(|tag| get_subscript_fp_cached(fp2, tag)) {
        h
    } else {
        // Clear SUBST_CACHE before evaluating val2 via with_explicit_env (state_env=None).
        // Required when val1 was also evaluated via with_explicit_env (both have pointer
        // identity 0, so eval_entry's pointer-based invalidation sees "same state").
        // Also clear defensively when val1 was cached but val2 is not: the SUBST_CACHE
        // may contain entries from a prior with_explicit_env call (same pointer 0),
        // and the pre-population invariant (all fps cached) is not enforced here.
        crate::eval::clear_subst_cache();
        // Build environment from s2 (next state) with the same base-env preservation.
        let mut env2 = ctx.env().clone();
        for (name, value) in s2.vars() {
            // Part of #2144: skip state vars that shadow local bindings.
            if !ctx.has_local_binding(name.as_ref()) {
                env2.insert(Arc::clone(name), value.clone());
            }
        }
        let ctx2 = ctx.with_explicit_env(env2);
        let h = value_fingerprint(&crate::eval::eval_entry(&ctx2, subscript)?);
        if let Some(tag) = cache_tag {
            set_subscript_fp_cache(fp2, tag, h);
        }
        h
    };

    if debug {
        eprintln!(
            "[DEBUG SUBSCRIPT] val1_fp={}, val2_fp={}, changed={}",
            val1,
            val2,
            val1 != val2
        );
    }

    // Compare content fingerprints (equal values fingerprint equal; distinct
    // values collide with ~2^-64 probability — the dedup trust level).
    Ok(val1 != val2)
}

// ── Subscript class registration (#liveness-leaf-memo) ──────────────────

/// Collect `(tag, subscript, bindings)` for every subscripted leaf
/// (`Enabled` / `StateChanged`) in a `LiveExpr` tree.
fn collect_subscripted_leaves<'a>(
    expr: &'a crate::liveness::LiveExpr,
    out: &mut Vec<(
        u32,
        &'a Arc<tla_core::Spanned<tla_core::ast::Expr>>,
        Option<&'a BindingChain>,
    )>,
) {
    use crate::liveness::LiveExpr;
    match expr {
        LiveExpr::Enabled {
            subscript: Some(s),
            bindings,
            tag,
            ..
        } => out.push((*tag, s, bindings.as_ref())),
        LiveExpr::StateChanged {
            subscript: Some(s),
            bindings,
            tag,
        } => out.push((*tag, s, bindings.as_ref())),
        LiveExpr::And(parts) | LiveExpr::Or(parts) => {
            for part in parts {
                collect_subscripted_leaves(part, out);
            }
        }
        LiveExpr::Not(inner)
        | LiveExpr::Always(inner)
        | LiveExpr::Eventually(inner)
        | LiveExpr::Next(inner) => collect_subscripted_leaves(inner, out),
        LiveExpr::Bool(_)
        | LiveExpr::StatePred { .. }
        | LiveExpr::ActionPred { .. }
        | LiveExpr::Enabled { .. }
        | LiveExpr::StateChanged { .. } => {}
    }
}

/// Build the equivalence-class key for one subscripted leaf, or `None` when
/// the leaf must remain in its own class (fail closed).
///
/// Two leaves may share one cached subscript value per state ONLY when their
/// subscript expressions are guaranteed to compute the same pure function of
/// the state:
///
/// 1. **Structural identity.** The resolved subscript ASTs are identical,
///    compared via their `Debug` rendering (which includes nested spans —
///    over-splitting is sound, over-merging is not). Quantifier-expanded
///    fairness (`\A c \in Clients : WF_vars(A(c))`) produces many leaves from
///    the same source subscript, all structurally identical.
/// 2. **Binding independence.** A leaf's quantifier `BindingChain` can change
///    the subscript's value only through identifiers the subscript actually
///    references. If `free_vars(subscript)` does not intersect the chain's
///    bound names, the bindings are irrelevant and are dropped from the key.
///    Referenced eager bindings are folded into the key by value, so leaves
///    that DO reference bound names share only when those values are equal.
///    If any chain binding is lazy (unobservable without forcing), we cannot
///    prove independence → `None` (leaf keeps its own per-tag key).
fn subscript_class_key(
    subscript: &tla_core::Spanned<tla_core::ast::Expr>,
    bindings: Option<&BindingChain>,
) -> Option<String> {
    use std::fmt::Write as _;
    let mut key = format!("{:?}", subscript.node);
    if let Some(chain) = bindings {
        if !chain.is_empty() {
            // Fail closed when the chain cannot be fully observed.
            let all = chain.all_bindings_eager()?;
            let free = tla_core::free_vars(&subscript.node);
            let mut referenced: Vec<(String, String)> = all
                .into_iter()
                .filter(|(name, _)| free.contains(name.as_ref()))
                .map(|(name, value)| (name.to_string(), format!("{value:?}")))
                .collect();
            referenced.sort();
            for (name, value) in referenced {
                let _ = write!(key, "|{name}={value}");
            }
        }
    }
    Some(key)
}

/// Register canonical subscript classes for the given liveness expressions.
///
/// Called once per run after the inline fairness plan is built (and the
/// previous mapping has been cleared by `reset_global_state`). Classes are
/// keyed by the smallest member tag, so class ids live in the leaf-tag space
/// and cannot collide with any other leaf's raw tag (tags are unique per
/// leaf). Singleton classes are omitted (identity fallback).
///
/// Why: on fairness-heavy specs every WF/SF conjunct carries the same
/// whole-`vars` subscript. AllocatorImplementation has 345 leaf tags but only
/// ~7 distinct subscript occurrences; without classes the per-state subscript
/// value is evaluated (and stored) once per TAG — ~2M evaluations — instead
/// of once per class (~124K).
pub(crate) fn register_subscript_tag_classes(exprs: &[crate::liveness::LiveExpr]) {
    let mut leaves = Vec::new();
    for expr in exprs {
        collect_subscripted_leaves(expr, &mut leaves);
    }
    let mut class_by_key: FxHashMap<String, Vec<u32>> = FxHashMap::default();
    for (tag, subscript, bindings) in leaves {
        if let Some(key) = subscript_class_key(subscript, bindings) {
            class_by_key.entry(key).or_default().push(tag);
        }
    }
    let mut map = FxHashMap::default();
    for tags in class_by_key.into_values() {
        if tags.len() < 2 {
            continue;
        }
        let class = *tags.iter().min().expect("non-empty class");
        for tag in tags {
            map.insert(tag, class);
        }
    }
    SUBSCRIPT_TAG_CLASS.with(|m| *m.borrow_mut() = map);
}

/// State-based cached subscript-change comparison for the inline-BFS ENABLED
/// scan path (#liveness-leaf-memo).
///
/// The inline ENABLED evaluator (`eval_enabled_with_array_successors`) probes
/// `has_state_change` for every (ENABLED leaf × candidate successor) pair. The
/// previous uncached closure re-evaluated the subscript expression (typically
/// the whole-`vars` tuple) TWICE per probe — ~15M full AST evaluations on
/// AllocatorImplementation (115 fairness leaves × 64K transitions), measured at
/// ~44% of total BFS CPU. This wrapper routes the probe through the same
/// `(Fingerprint, tag)`-keyed `SUBSCRIPT_VALUE_CACHE` the SCC phase
/// (`checker/eval.rs` `LiveExprEvaluator::eval_subscript_changed`) and the
/// inline action-leaf short-circuit (`eval_subscript_changed_array_cached`)
/// already use, so each state's subscript value is evaluated at most once per
/// tag.
///
/// Soundness: identical key shape and lifecycle as the established users —
/// `tag` uniquely identifies the (subscript expr, bindings) pair fixed at
/// conversion time; `Fingerprint` identifies the state with exactly the fp64
/// trust the BFS dedup already places in it (keying memo entries on the same
/// fps adds no collision risk beyond what dedup already accepts); the cache is
/// cleared at run reset and at `populate_node_check_masks`, and is already
/// written during inline BFS by `apply_subscript_short_circuit_bitmask` in the
/// same tag space.
pub(crate) fn eval_subscript_changed_state_cached(
    ctx: &EvalCtx,
    s1: &State,
    s2: &State,
    subscript: &tla_core::Spanned<tla_core::ast::Expr>,
    tag: u32,
) -> EvalResult<bool> {
    eval_subscript_changed(ctx, s1, s2, subscript, Some(tag))
}

/// Get-or-eval a subscript value's CONTENT FINGERPRINT for the array-native
/// inline BFS path.
///
/// Checks the `(Fingerprint, tag)` subscript cache first. On cache miss,
/// evaluates the subscript expression with the same binding contract as
/// `eval_subscript_changed_array_uncached` in `inline_leaf_eval.rs`:
/// 1. Apply liveness bindings if present (#2116 contract)
/// 2. Bind the array state via `bind_state_array_guard`
/// 3. Clear next_state_env (subscript evaluates current state only)
/// 4. Clear SUBST_CACHE (prevent stale entries from prior state)
/// 5. Evaluate with `eval_entry`, then fingerprint the value
///
/// Returns the `value_fingerprint` of the subscript value (callers only ever
/// compare these for change detection).
///
/// Part of #3100 Phase A0: inline subscript value caching.
fn get_or_eval_subscript_fp_array(
    ctx: &EvalCtx,
    fp: Fingerprint,
    array: &ArrayState,
    subscript: &Arc<tla_core::Spanned<tla_core::ast::Expr>>,
    bindings: Option<&BindingChain>,
    tag: u32,
) -> EvalResult<u64> {
    // Fast path: cache hit
    if let Some(h) = get_subscript_fp_cached(fp, tag) {
        return Ok(h);
    }

    // Slow path: evaluate and cache.
    let mut eval_ctx = match bindings {
        Some(chain) => ctx.with_liveness_bindings(chain),
        None => ctx.clone(),
    };
    let _state_guard = eval_ctx.bind_state_env_guard(array.env_ref());
    let _ = eval_ctx.next_state_mut().take();
    let _next_guard = eval_ctx.take_next_state_env_guard();
    crate::eval::clear_subst_cache();
    let h = value_fingerprint(&crate::eval::eval_entry(&eval_ctx, subscript)?);

    set_subscript_fp_cache(fp, tag, h);
    Ok(h)
}

/// Cached subscript change comparison for the array-native inline BFS path.
///
/// Replaces `eval_subscript_changed_array_uncached` by caching individual
/// subscript values by `(Fingerprint, tag)`. Reduces subscript evaluations
/// from two per comparison to one per previously unseen `(Fingerprint, tag)`.
///
/// The cache is shared with the post-BFS liveness path and cleared at the
/// start of `populate_node_check_masks` via `clear_subscript_value_cache`.
///
/// Part of #3100 Phase A0: inline subscript value caching.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_subscript_changed_array_cached(
    ctx: &EvalCtx,
    current_fp: Fingerprint,
    current_array: &ArrayState,
    next_fp: Fingerprint,
    next_array: &ArrayState,
    subscript: &Arc<tla_core::Spanned<tla_core::ast::Expr>>,
    bindings: Option<&BindingChain>,
    tag: u32,
) -> EvalResult<bool> {
    let val1 =
        get_or_eval_subscript_fp_array(ctx, current_fp, current_array, subscript, bindings, tag)?;
    let val2 = get_or_eval_subscript_fp_array(ctx, next_fp, next_array, subscript, bindings, tag)?;
    Ok(val1 != val2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retained_capacity() -> usize {
        SUBSCRIPT_VALUE_CACHE.with(|cache| cache.borrow().capacity())
            + SUBSCRIPT_TAG_CLASS.with(|map| map.borrow().capacity())
    }

    #[test]
    fn release_subscript_cache_storage_drops_capacity() {
        release_subscript_cache_storage();
        SUBSCRIPT_VALUE_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.reserve(128);
            cache.insert((Fingerprint(1), 2), 3);
        });
        SUBSCRIPT_TAG_CLASS.with(|map| {
            let mut map = map.borrow_mut();
            map.reserve(128);
            map.insert(2, 2);
        });
        assert_eq!(census_subscript_cache_len(), 1);
        assert!(retained_capacity() > 0);

        release_subscript_cache_storage();

        assert_eq!(census_subscript_cache_len(), 0);
        assert_eq!(retained_capacity(), 0);
    }
}
