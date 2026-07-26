// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! User-defined operator result cache with TLC-style subset validation.
//!
//! Contains CachedOpResult/op_cache_entry_valid (shared by NARY and ZERO_ARG caches)
//! and the N-ary operator result cache (NARY_OP_CACHE, NARY_PERSISTENT_CACHE).
//! Part of #3025: OP_RESULT_CACHE removed (zero insertions after Phase 1).
//!
//! Part of #2744 decomposition from eval_cache.rs.

use super::dep_tracking::{current_state_lookup_mode, OpEvalDeps, StateLookupMode};
use super::zero_arg_cache::deps_are_persistent;
use crate::value::Value;
use crate::var_index::VarIndex;
use crate::EvalCtx;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;
use tla_core::name_intern::{intern_name, NameId};
use tla_value::Rp;

// Specs like bosco have guards that call the same operator multiple times with the same arguments:
//   rcvd01(self) >= N - T /\ rcvd0(self) >= moreNplus3Tdiv2 /\ rcvd0(self) < moreNplus3Tdiv2
//
// Each call to `rcvd0(self)` evaluates `Cardinality({m \in rcvd'[self] : m[2] = "ECHO0"})` which
// iterates over all elements in rcvd'[self]. Caching avoids redundant evaluations.
//
// Baseline alignment:
// TLC caches evaluation results via LazyValue and validates cache hits with
// `TLCState.isSubset(s0, cachedS0)` / `TLCState.isSubset(s1, cachedS1)`, rather than requiring
// an exact match on the full state.
//
// In Rust we implement a *sound* subset-style cache by tracking the concrete dependencies
// (state vars, next-state vars, and captured locals) actually read during evaluation, and
// reusing a cached value only when all recorded dependencies still match the current context.

// Part of #3025: OpResultCacheKey removed — was only used by the dead OP_RESULT_CACHE.
// NaryOpCacheKey (below) is the active cache key type for n-ary operators.

/// Cached operator result with TLC-style subset validation.
#[derive(Clone)]
pub(crate) struct CachedOpResult {
    pub(crate) value: Value,
    pub(crate) deps: OpEvalDeps,
}

/// Part of #3579: Compact variant of `next_mode_dep_matches` that accepts
/// `&CompactValue` from the new VarDepMap storage, avoiding Value reconstruction.
pub(crate) fn next_mode_dep_matches_compact(
    ctx: &EvalCtx,
    idx: VarIndex,
    expected: &tla_value::CompactValue,
) -> bool {
    let name = ctx.var_registry().name(idx);
    let name_id = intern_name(name);

    if let Some((bv, source)) = ctx.bindings.lookup(name_id) {
        if bv.matches_compact(StateLookupMode::Current, source, expected) {
            return true;
        }
    }

    if let Some(sparse_env) = ctx.sparse_next_state_env {
        // SAFETY: idx bounded by VarRegistry.
        let slot = unsafe { sparse_env.get_unchecked(idx.as_usize()) };
        if let Some(actual) = slot {
            return expected.matches_value(actual);
        }
    }

    if let Some(state_env) = ctx.state_env {
        debug_assert!(idx.as_usize() < state_env.env_len());
        // SAFETY: dependency indices are recorded from validated VarRegistry lookups.
        return unsafe { state_env.slot_matches_compact(idx.as_usize(), expected) };
    }

    // Cold path: convert CompactValue to Value for env HashMap lookup.
    let expected_val = Value::from(expected);
    ctx.env
        .get(name)
        .is_some_and(|actual| actual == &expected_val)
}

// Kill switch (#GameOfLife): when `TY_NO_STATE_DEP_BINDING_FALLBACK=1` is set,
// state-dep validation reverts to the pre-fix behavior of rejecting any entry
// with state deps whenever `ctx.state_env` is None. Used to prove the fallback
// is verdict/count-neutral.
feature_flag!(
    pub(crate) state_dep_binding_fallback_disabled,
    "TY_NO_STATE_DEP_BINDING_FALLBACK"
);

/// Validate a *current-mode* (unprimed) state dependency when `ctx.state_env`
/// is absent (e.g. `Next` action evaluation and property/ENABLED checking bind
/// state variables via the binding chain / `env` HashMap instead of the state
/// array). Mirrors the read-resolution cascade of `eval_state_var` for a plain
/// state variable so that "validation returns true" iff a fresh read of the var
/// in `ctx` would return `expected`:
///   1. binding chain (Current mode) — a ready eager binding shadows env; a
///      liveness-source binding is skipped (the read path skips it too); a lazy
///      binding cannot be compared cheaply → conservative miss (false).
///   2. `state_env` slot (None in this branch, kept for completeness/symmetry).
///   3. `env` HashMap (the read path's final fallback — where GameOfLife's
///      unprimed `grid` lives during `Next`).
///
/// Soundness: `state` deps are only ever recorded from the `state_env` or `env`
/// read branches (the binding-chain branches record *local*/instance deps), so
/// with `state_env` absent the observed value came from `env`. Checking the
/// binding chain first (first-source-wins) keeps validation correct even if the
/// var later moves into the chain: a fresh read would then resolve to the chain
/// value, and so does this check.
pub(crate) fn current_mode_dep_matches_compact(
    ctx: &EvalCtx,
    idx: VarIndex,
    expected: &tla_value::CompactValue,
) -> bool {
    let name = ctx.var_registry().name(idx);
    let name_id = intern_name(name);

    if let Some((bv, source)) = ctx.bindings.lookup(name_id) {
        if !matches!(source, crate::binding_chain::BindingSourceRef::Liveness(_)) {
            return bv.matches_compact(StateLookupMode::Current, source, expected);
        }
    }

    if let Some(state_env) = ctx.state_env {
        // SAFETY: dependency indices originate from this model's VarRegistry.
        return unsafe { state_env.slot_matches_compact(idx.as_usize(), expected) };
    }

    // Cold path: convert CompactValue to Value for env HashMap comparison.
    let expected_val = Value::from(expected);
    ctx.env
        .get(name)
        .is_some_and(|actual| actual == &expected_val)
}

pub(crate) fn op_cache_entry_valid(ctx: &EvalCtx, entry: &CachedOpResult) -> bool {
    if entry.deps.inconsistent {
        return false;
    }
    // Validate captured locals by NameId against the *current* BindingChain.
    // Part of #2955: Use BindingChain lookup instead of local_stack scan.
    for (name_id, expected) in &entry.deps.local {
        let matches = ctx.bindings.lookup(*name_id).is_some_and(|(bv, source)| {
            // Part of #3579: eager locals and recorded local deps are both
            // CompactValue, so cache validation can compare directly.
            bv.matches_compact(StateLookupMode::Current, source, &expected.value)
        });
        if !matches {
            return false;
        }
    }
    // Validate state deps against the current state array.
    // Issue #73: Only require state_env if there are state dependencies — allows caching
    // pure operators (e.g., IsStronglyConnected) during Init where state_env is None.
    if !entry.deps.state.is_empty() {
        if let Some(state_env) = ctx.state_env {
            for (idx, expected) in entry.deps.state.iter() {
                debug_assert!(idx.as_usize() < state_env.env_len());
                // SAFETY: cached dependency indices originate from this model's VarRegistry.
                // Part of #3579: VarDepMap now stores CompactValue; slot_matches_compact
                // compares directly without Value materialization on either side.
                if !unsafe { state_env.slot_matches_compact(idx.as_usize(), expected) } {
                    return false;
                }
            }
        } else if state_dep_binding_fallback_disabled() {
            // Kill switch: pre-fix behavior — reject when no state array is present.
            return false;
        } else {
            // #GameOfLife: `state_env` absent (Next action / property / ENABLED
            // evaluation binds unprimed state via the binding chain or `env`
            // HashMap). Validate each state dep against the same resolution
            // cascade a read would use, mirroring `next_mode_dep_matches_compact`
            // for the primed case. Without this, zero-arg operators that read
            // state (e.g. GameOfLife's `sc` grid function) can never be cache-hit
            // during Next, forcing full re-materialization on every use.
            for (idx, expected) in entry.deps.state.iter() {
                if !current_mode_dep_matches_compact(ctx, idx, expected) {
                    return false;
                }
            }
        }
    }
    // Fix #3062: Validate TLCGet("level") dependency.
    // If the cached evaluation read TLCGet("level"), the cache entry is only valid
    // when the current BFS level matches the recorded level.
    if let Some(cached_level) = entry.deps.tlc_level {
        if ctx.tlc_level != cached_level {
            return false;
        }
    }
    // Validate next-state deps (if any) against whatever next-state context is available.
    if !entry.deps.next.is_empty() {
        if let Some(next_env) = ctx.next_state_env {
            for (idx, expected) in entry.deps.next.iter() {
                debug_assert!(idx.as_usize() < next_env.env_len());
                // SAFETY: dependency indices are recorded from validated VarRegistry lookups.
                // Part of #3579: slot_matches_compact for CompactValue dep storage.
                if !unsafe { next_env.slot_matches_compact(idx.as_usize(), expected) } {
                    return false;
                }
            }
        } else if let Some(next_state) = &ctx.next_state {
            for (idx, expected_cv) in entry.deps.next.iter() {
                let name = ctx.var_registry().name(idx);
                let Some(actual) = next_state.get(name) else {
                    return false;
                };
                // Cold path: convert CompactValue back to Value for HashMap comparison.
                if !expected_cv.matches_value(actual) {
                    return false;
                }
            }
        } else if current_state_lookup_mode(ctx) == StateLookupMode::Next {
            // eval_prime Next mode without next_state/next_state_env: rebinds through
            // BindingChain/sparse_next_state_env/state_env/env. Covers swapped-array path,
            // sparse overlays (ENABLED), partial next overlays, full env overlays,
            // quantifier fast-bindings. Sound: each dep index must match the exact value
            // observed during cached evaluation.
            // NOTE: do not shortcut to state_env only; that misses binding/overlay paths.
            for (idx, expected_cv) in entry.deps.next.iter() {
                if !next_mode_dep_matches_compact(ctx, idx, expected_cv) {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
    true
}

// Part of #3025: OP_RESULT_CACHE thread-local removed. The cache had zero
// insertion points after Phase 1 (unified lazy args, #3000) but was still
// cleared on every state transition and ENABLED scope boundary. Removing it
// eliminates 3 unnecessary thread-local accesses per state boundary.
// OpResultCacheKey also removed (no remaining consumers).
// CachedOpResult and op_cache_entry_valid are still used by NARY_OP_CACHE
// and ZERO_ARG_OP_CACHE.

// ============================================================================
// N-ary operator result cache — Part of #2991
// ============================================================================
//
// Re-enables operator result caching after #3000 removed OP_RESULT_CACHE for the
// universal lazy arg path. Unlike the original OP_RESULT_CACHE, this cache:
//
// - Eagerly evaluates all arity-0 args to compute the cache key (cheap for
//   constant args like `Nodes` in `LimitedSeq(Nodes)`)
// - Falls back to the lazy path for higher-order params or arg eval errors
// - Uses dep validation (same as ZERO_ARG_OP_CACHE) for cache hits
// - Stores multiple entries per key with different dep contexts
//
// Target pattern: `LimitedSeq(Nodes)` called ~663K times in MCReachabilityTestAllGraphs.
// Without cache: ~100μs per eval → ~66s total. With cache: 1 eval + 663K hits → <1s.

/// Cache key for N-ary operator result caching.
///
/// Part of #3020: `args` moved from key to `NaryOpCacheEntry`
/// values, replaced by `args_hash: u64`. This avoids `Arc::from(args)` heap
/// allocation on every cache lookup — the hash is computed from `&[Value]`
/// without allocation. Actual arg values are validated after HashMap hit to
/// handle hash collisions.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct NaryOpCacheKey {
    pub(crate) shared_id: u64,
    // BUG FIX #3024: Include local_ops_id and instance_subs_id to distinguish different
    // INSTANCE contexts and LET-scoped operator environments. Without these, operators
    // with the same name and args but different local_ops or instance_substitutions
    // (e.g., different INSTANCE instantiations) can share cached results incorrectly.
    // Part of #3099: Changed from `usize` (Arc::as_ptr) to `u64` (content-based
    // fingerprint) for stable cross-reconstruction cache hits.
    pub(crate) local_ops_id: u64,
    pub(crate) instance_subs_id: u64,
    pub(crate) op_name: NameId,
    pub(crate) def_loc: u32,
    pub(crate) is_next_state: bool,
    /// Part of #3020: Hash of the args slice, replacing `Arc<[Value]>` to avoid
    /// per-lookup heap allocation. Collisions resolved by comparing actual args
    /// stored in cache entries.
    pub(crate) args_hash: u64,
    // Part of #2991: Include param_args_hash to distinguish different parametrized
    // INSTANCE argument values (defense in depth — mirrors OpResultCacheKey's BUG FIX #2986).
    // Without this, `P(Succ1)!Op(args)` and `P(Succ2)!Op(args)` can share cached results
    // when dep validation fails to detect the INSTANCE parameter change.
    pub(crate) param_args_hash: u64,
}

/// State-scoped cached operator result with args and dependency validation.
/// Part of #3020: args stored here (not in key) to avoid Arc allocation on lookup.
#[derive(Clone)]
pub(crate) struct NaryOpCacheEntry {
    pub(crate) args: Arc<[Value]>,
    pub(crate) result: CachedOpResult,
}

/// Persistent cached operator result.
///
/// Admission to this partition requires [`deps_are_persistent`], so retaining
/// an empty [`OpEvalDeps`] on every long-lived entry wastes memory and makes
/// the hit path repeat a validator and dependency-propagation no-op. Encoding
/// only the exact args and value makes that invariant structural: persistent
/// entries cannot carry dependency state at all.
#[derive(Clone)]
pub(crate) struct NaryPersistentCacheEntry {
    pub(crate) args: Arc<[Value]>,
    pub(crate) value: Value,
}

/// A successful n-ary cache probe.
///
/// Persistent hits have no dependencies by construction. State hits retain
/// the existing validated [`CachedOpResult`] so callers can propagate its
/// dependencies exactly as before.
pub(crate) enum NaryCacheHit {
    Persistent(Value),
    State(CachedOpResult),
}

// Kill switch for the fingerprint-based n-ary cache-key arg hashing.
// `TY_NO_FP_ARGS_HASH=1` reverts to deep structural `Value::hash` of every arg.
feature_flag!(pub(crate) no_fp_args_hash, "TY_NO_FP_ARGS_HASH");

/// Compute a deterministic hash of a `&[Value]` slice for use as `args_hash`.
/// Part of #4112: Uses FxHasher instead of SipHash for faster hashing.
///
/// Compound function-like args (Func/Bag/IntFunc/Record/Set/Seq) hash via
/// their cached additive dedup fingerprint (`cache_key_value_fingerprint`)
/// instead of a deep content walk. This is sound for a cache KEY because every
/// n-ary cache hit verifies full arg equality (`entry.args.as_ref() == args`)
/// before being accepted — the hash only routes lookups to a bucket, so any
/// fingerprint collision (or hash-scheme difference) degrades to a miss, never
/// a wrong value. The fingerprint is a pure deterministic function of value
/// content (never pointers), so equal args always produce equal keys within
/// and across states. Big win when a large state value (e.g. EWD998PCal's
/// `network`) is passed to helper operators millions of times: the Arc-cached
/// fingerprint makes the per-call key cost O(1) instead of O(|value|).
///
/// Kill switch: `TY_NO_FP_ARGS_HASH=1` restores the deep structural hash.
pub(crate) fn hash_args(args: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    if no_fp_args_hash() {
        args.hash(&mut hasher);
        return hasher.finish();
    }
    // Mirror the slice Hash layout (len prefix + per-element) with a
    // fingerprint fast path for compound values.
    args.len().hash(&mut hasher);
    for arg in args {
        hash_value_cache_key(arg, &mut hasher);
    }
    hasher.finish()
}

/// Hash a single `Value` into a cache-KEY hasher using the additive dedup
/// fingerprint fast path for compound values, with a deterministic deep-hash
/// fallback.
///
/// SOUNDNESS: only valid for cache keys whose hits verify FULL value equality
/// before being accepted (the n-ary op cache compares `entry.args == args`,
/// the zero-arg param LET cache compares `stored_deps == dep_vals`). The hash
/// only routes lookups to a bucket: a fingerprint collision degrades to a
/// miss, never a wrong value. The fingerprint is a pure deterministic
/// function of value content (never pointers), so equal values always hash
/// equally within and across states.
///
/// Tuple is in the fp fast-path list because compound state values are
/// routinely passed to helper operators as tuples (e.g. a `dag == <<vs, es>>`
/// pair on dag-consensus specs, where deep hashing the full edge set per
/// `Children(dag, v)` call dominated the whole run at 47% of cycles).
/// `compute_tuple_additive_fp` is O(arity) over the elements' CACHED additive
/// fingerprints, so the per-call key cost stays O(1) for large shared
/// components.
///
/// Kill switch: respects `TY_NO_FP_ARGS_HASH=1` (deep structural hash).
pub(crate) fn hash_value_cache_key<H: std::hash::Hasher>(value: &Value, hasher: &mut H) {
    use std::hash::Hash;
    if no_fp_args_hash() {
        value.hash(hasher);
        return;
    }
    match value {
        Value::Func(_)
        | Value::Bag(_)
        | Value::IntFunc(_)
        | Value::Record(_)
        | Value::Set(_)
        | Value::Seq(_)
        | Value::Tuple(_) => {
            match tla_value::dedup_fingerprint::cache_key_value_fingerprint(value) {
                // Distinct scheme tags keep fp-hashed and deep-hashed streams
                // from aliasing each other structurally.
                Ok(fp) => {
                    0xF1u8.hash(hasher);
                    fp.hash(hasher);
                }
                // Content-determined failure (e.g. nested lazy closure):
                // deterministic deep-hash fallback for this value.
                Err(_) => {
                    0xF2u8.hash(hasher);
                    value.hash(hasher);
                }
            }
        }
        _ => {
            0xF3u8.hash(hasher);
            value.hash(hasher);
        }
    }
}

/// Maximum entries per key in NARY_OP_CACHE.
pub(crate) const NARY_OP_CACHE_MAX_ENTRIES_PER_KEY: usize = 16;

/// Most persistent keys have exactly one exact-argument entry. Keep that
/// common case inline while retaining the existing multi-entry collision and
/// dependency-context behavior when a bucket grows.
pub(crate) type NaryPersistentCacheBucket = SmallVec<[NaryPersistentCacheEntry; 1]>;

// Part of #3805: Consolidated NARY_OP_CACHE + NARY_PERSISTENT_CACHE into a single
// TLS struct. Previously 2 separate `thread_local!` declarations — nary_lookup
// required 2 `_tlv_get_addr` calls on macOS (~5ns each). Now a single TLS access
// covers both partitions, saving ~5ns per nary_lookup on the BFS hot path.
// Same consolidation pattern as ZERO_ARG_CACHES (#4053/#3962).
pub(crate) struct NaryCaches {
    /// State partition: dependency-heavy entries cleared on state boundaries.
    /// Keep `Vec` here: the persistent singleton telemetry does not justify
    /// inflating every short-lived state-map value with an inline entry.
    pub(crate) state: FxHashMap<NaryOpCacheKey, Vec<NaryOpCacheEntry>>,
    /// Persistent partition: exact args + values, survives state boundaries.
    pub(crate) persistent: FxHashMap<NaryOpCacheKey, NaryPersistentCacheBucket>,
    /// Part of #4391: Release-mode counters for focused n-ary cache diagnosis.
    pub(crate) stats: NaryCacheStats,
}

impl NaryCaches {
    fn new() -> Self {
        NaryCaches {
            state: FxHashMap::default(),
            persistent: FxHashMap::default(),
            stats: NaryCacheStats::new(),
        }
    }
}

std::thread_local! {
    pub(crate) static NARY_CACHES: std::cell::RefCell<NaryCaches> =
        std::cell::RefCell::new(NaryCaches::new());
}

// Part of #4391: Focused n-ary cache counters for MCReachability hot-path scouting.
// Enabled via TY_NARY_CACHE_STATS=1 and printed by print_eval_profile_stats().
pub(crate) fn debug_nary_stats() -> bool {
    #[cfg(test)]
    if NARY_STATS_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return true;
    }

    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    tla_core::env_flag_is_set(&FLAG, "TY_NARY_CACHE_STATS")
}

#[cfg(test)]
std::thread_local! {
    static NARY_STATS_TEST_OVERRIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NaryCacheStats {
    pub(crate) lookups: u64,
    pub(crate) state_key_hits: u64,
    pub(crate) persistent_key_hits: u64,
    pub(crate) state_hits: u64,
    pub(crate) persistent_hits: u64,
    pub(crate) misses: u64,
    pub(crate) state_entries_scanned: u64,
    pub(crate) persistent_entries_scanned: u64,
}

impl NaryCacheStats {
    fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
fn with_nary_stats_for_test<R>(f: impl FnOnce() -> R) -> R {
    NARY_STATS_TEST_OVERRIDE.with(|flag| {
        let previous = flag.replace(true);
        let result = f();
        flag.set(previous);
        result
    })
}

#[cfg(test)]
fn reset_nary_cache_for_test() {
    NARY_CACHES.with(|caches| *caches.borrow_mut() = NaryCaches::new());
}

#[cfg(test)]
fn nary_cache_stats_for_test() -> NaryCacheStats {
    NARY_CACHES.with(|caches| caches.borrow().stats)
}

/// Part of #3100: Insert into the appropriate n-ary partition based on deps.
/// Part of #3805: Single TLS access for consolidated NARY_CACHES.
#[inline]
pub(crate) fn nary_insert(key: NaryOpCacheKey, entry: NaryOpCacheEntry) {
    let is_persistent = deps_are_persistent(&entry.result.deps);
    NARY_CACHES.with(|caches| {
        let mut caches = caches.borrow_mut();
        if is_persistent {
            let entries = caches.persistent.entry(key).or_default();
            while entries.len() >= NARY_OP_CACHE_MAX_ENTRIES_PER_KEY {
                entries.remove(0);
            }
            entries.push(NaryPersistentCacheEntry {
                args: entry.args,
                value: entry.result.value,
            });
        } else {
            let entries = caches.state.entry(key).or_default();
            while entries.len() >= NARY_OP_CACHE_MAX_ENTRIES_PER_KEY {
                entries.remove(0);
            }
            entries.push(entry);
        }
    });
}

/// Part of #3100: N-ary lookup probing persistent before the state partition.
/// Persistent entries need only exact arg equality: their admission proved
/// they carry no context dependencies. State entries preserve the existing
/// exact-arg check followed by caller-provided dependency validation.
/// Part of #3805: Single TLS access for consolidated NARY_CACHES (was 2 accesses).
#[inline]
pub(crate) fn nary_lookup(
    key: &NaryOpCacheKey,
    args: &[Value],
    validator: impl Fn(&EvalCtx, &CachedOpResult) -> bool,
    ctx: &EvalCtx,
) -> Option<NaryCacheHit> {
    if debug_nary_stats() {
        return NARY_CACHES.with(|caches| {
            let mut caches = caches.borrow_mut();
            let NaryCaches {
                state,
                persistent,
                stats,
            } = &mut *caches;
            stats.lookups += 1;

            // Persistent hits are context-free by construction, so take the
            // validator-free fast path before considering state entries.
            if let Some(entries) = persistent.get(key) {
                stats.persistent_key_hits += 1;
                for entry in entries {
                    stats.persistent_entries_scanned += 1;
                    if entry.args.as_ref() == args {
                        stats.persistent_hits += 1;
                        return Some(NaryCacheHit::Persistent(entry.value.clone()));
                    }
                }
            }
            // Dependency-bearing entries keep their established validator.
            if let Some(entries) = state.get(key) {
                stats.state_key_hits += 1;
                for entry in entries {
                    stats.state_entries_scanned += 1;
                    if entry.args.as_ref() == args && validator(ctx, &entry.result) {
                        stats.state_hits += 1;
                        return Some(NaryCacheHit::State(entry.result.clone()));
                    }
                }
            }

            stats.misses += 1;
            None
        });
    }

    NARY_CACHES.with(|caches| {
        let caches = caches.borrow();
        // Persistent entries carry no dependency payload and need no validator.
        if let Some(entries) = caches.persistent.get(key) {
            for entry in entries {
                if entry.args.as_ref() == args {
                    return Some(NaryCacheHit::Persistent(entry.value.clone()));
                }
            }
        }
        // State entries retain exact arg equality plus dependency validation.
        if let Some(entries) = caches.state.get(key) {
            for entry in entries {
                if entry.args.as_ref() == args && validator(ctx, &entry.result) {
                    return Some(NaryCacheHit::State(entry.result.clone()));
                }
            }
        }
        None
    })
}

/// Hit path that avoids cloning a state hit's `CachedOpResult` out of the cache.
///
/// Churn elimination: on a state-partition hit, cloning the `CachedOpResult`
/// deep-clones `OpEvalDeps`, and every heap-backed `CompactValue` dep entry
/// costs a fresh `Box` allocation plus an inner `Value` clone — freed moments
/// later (12.5M hits on btree, each with 1-3 heap state deps). The deps of a
/// hit entry are only ever READ (`propagate_cached_deps`), so propagate them
/// by reference from inside the cache borrow and clone only the result
/// `Value` out (one Arc bump). Persistent hits carry no dependency payload by
/// construction ([`NaryPersistentCacheEntry`]), so they are already
/// value-only clones.
///
/// Re-entrancy: `propagate_cached_deps` touches only
/// `ctx.runtime_state.op_dep_stack` (a different RefCell), never NARY_CACHES,
/// so calling it under the borrow cannot double-borrow.
///
/// Semantics are identical to `nary_lookup` + caller-side finishing of the
/// [`NaryCacheHit`] (propagate state deps, unwrap the value): same probe
/// order (persistent first, validator-free), same validators for state
/// entries, same propagation, same returned value.
#[inline]
pub(crate) fn nary_lookup_value(
    key: &NaryOpCacheKey,
    args: &[Value],
    validator: impl Fn(&EvalCtx, &CachedOpResult) -> bool,
    ctx: &EvalCtx,
) -> Option<Value> {
    if debug_nary_stats() {
        // Stats path: reuse the full-clone lookup (cold, opt-in via env var)
        // and finish like the historical caller did.
        return match nary_lookup(key, args, validator, ctx)? {
            NaryCacheHit::Persistent(value) => Some(value),
            NaryCacheHit::State(result) => {
                crate::cache::propagate_cached_deps(ctx, &result.deps);
                Some(result.value)
            }
        };
    }

    NARY_CACHES.with(|caches| {
        let caches = caches.borrow();
        // Persistent entries carry no dependency payload and need no validator.
        if let Some(entries) = caches.persistent.get(key) {
            for entry in entries {
                if entry.args.as_ref() == args {
                    return Some(entry.value.clone());
                }
            }
        }
        // State entries retain exact arg equality plus dependency validation;
        // propagate their deps by reference and clone only the value out.
        if let Some(entries) = caches.state.get(key) {
            for entry in entries {
                if entry.args.as_ref() == args && validator(ctx, &entry.result) {
                    crate::cache::propagate_cached_deps(ctx, &entry.result.deps);
                    return Some(entry.result.value.clone());
                }
            }
        }
        None
    })
}

/// Print n-ary cache stats summary. Called at end of model checking run.
pub(crate) fn print_nary_cache_stats() {
    if !debug_nary_stats() {
        return;
    }
    NARY_CACHES.with(|caches| {
        let caches = caches.borrow();
        let s = &caches.stats;
        let hits = s.state_hits + s.persistent_hits;
        let pct = |value: u64| {
            if s.lookups == 0 {
                "n/a".to_string()
            } else {
                format!("{:.1}%", value as f64 / s.lookups as f64 * 100.0)
            }
        };
        eprintln!("\n=== N-ary Cache Stats ===");
        eprintln!("Total lookups:             {}", s.lookups);
        eprintln!("Hits:                      {} ({})", hits, pct(hits));
        eprintln!(
            "  State hits:              {} ({})",
            s.state_hits,
            pct(s.state_hits)
        );
        eprintln!(
            "  Persistent hits:         {} ({})",
            s.persistent_hits,
            pct(s.persistent_hits)
        );
        eprintln!(
            "Misses:                    {} ({})",
            s.misses,
            pct(s.misses)
        );
        eprintln!("State key probes:          {}", s.state_key_hits);
        eprintln!("Persistent key probes:     {}", s.persistent_key_hits);
        eprintln!("State entries scanned:     {}", s.state_entries_scanned);
        eprintln!(
            "Persistent entries scanned: {}",
            s.persistent_entries_scanned
        );
        eprintln!("===========================\n");
    });
}

/// Part of #3109: Constant-entry fallback lookup with flipped `is_next_state`.
/// Part of #3100: Probes persistent partition only (constant entries live there).
/// Part of #3805: Single TLS access via consolidated NARY_CACHES.
#[inline]
pub(crate) fn nary_constant_fallback(key: &NaryOpCacheKey, args: &[Value]) -> Option<Value> {
    let flipped_key = NaryOpCacheKey {
        is_next_state: !key.is_next_state,
        ..key.clone()
    };
    NARY_CACHES.with(|caches| {
        let caches = caches.borrow();
        let entries = caches.persistent.get(&flipped_key)?;
        for entry in entries {
            if entry.args.as_ref() == args {
                return Some(entry.value.clone());
            }
        }
        None
    })
}

/// Part of #3100: Clear state partition only. Persistent entries survive.
/// Part of #3805: Single TLS access via consolidated NARY_CACHES.
#[inline]
pub(crate) fn clear_nary_state_partition() {
    NARY_CACHES.with(|c| c.borrow_mut().state.clear());
}

/// Part of #3100: Clear both partitions (run reset / test reset).
/// Part of #3805: Single TLS access via consolidated NARY_CACHES.
#[inline]
pub(crate) fn clear_nary_all_partitions() {
    NARY_CACHES.with(|c| {
        let mut c = c.borrow_mut();
        c.state.clear();
        c.persistent.clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::dep_tracking::OpEvalDeps;
    use std::sync::Arc;
    use tla_value::Rp;

    fn test_key() -> NaryOpCacheKey {
        NaryOpCacheKey {
            shared_id: 1,
            local_ops_id: 0,
            instance_subs_id: 0,
            op_name: intern_name("StatsOp"),
            def_loc: 7,
            is_next_state: false,
            args_hash: hash_args(&[Value::int(1)]),
            param_args_hash: 0,
        }
    }

    fn persistent_entry(arg: Value, value: Value) -> NaryPersistentCacheEntry {
        NaryPersistentCacheEntry {
            args: Arc::from([arg]),
            value,
        }
    }

    fn persistent_insert_entry(arg: Value, value: Value) -> NaryOpCacheEntry {
        NaryOpCacheEntry {
            args: Arc::from([arg]),
            result: CachedOpResult {
                value,
                deps: OpEvalDeps::default(),
            },
        }
    }

    fn state_entry(arg: Value, value: Value, recorded: &Value) -> NaryOpCacheEntry {
        let mut deps = OpEvalDeps::default();
        deps.record_state(VarIndex(0), recorded);
        NaryOpCacheEntry {
            args: Arc::from([arg]),
            result: CachedOpResult { value, deps },
        }
    }

    #[test]
    fn persistent_entry_layout_is_materially_smaller_than_state_entry() {
        assert!(
            std::mem::size_of::<NaryPersistentCacheEntry>()
                < std::mem::size_of::<NaryOpCacheEntry>(),
            "persistent entries must not retain the state entry's dependency payload"
        );
    }

    /// Lever 3 (#EWD998PCal): fp-based arg hashing must be a deterministic
    /// pure function of content — equal-content args (even across
    /// representations that compare equal, like Func vs Bag) produce equal
    /// `args_hash`, so lookups keep hitting entries stored earlier.
    #[test]
    fn hash_args_fp_scheme_is_content_deterministic() {
        let rec = Value::record([("type", Value::string("pl"))]);

        // Same bag content constructed twice (different Arcs) → same hash.
        let mk_bag = || {
            tla_value::BagValue::try_from_entries(vec![(rec.clone(), Value::int(2))])
                .map(|b| Value::Bag(Rp::new(b)))
                .expect("compact bags enabled by default")
        };
        let bag1 = mk_bag();
        let bag2 = mk_bag();
        assert_eq!(bag1, bag2);
        assert_eq!(hash_args(&[bag1.clone()]), hash_args(&[bag2.clone()]));

        // The equivalent Func representation compares equal to the Bag and
        // must hash equal too (cache_key_value_fingerprint converges
        // representations: Bag ≡ Func with SmallInt counts).
        let mut fb = tla_value::FuncBuilder::new();
        fb.insert(rec, Value::int(2));
        let func = Value::Func(Rp::new(fb.build()));
        assert_eq!(func, bag1);
        assert_eq!(hash_args(&[func]), hash_args(&[bag1]));

        // Mixed arg lists stay deterministic.
        let args1 = [Value::int(3), Value::string("x"), bag2.clone()];
        let args2 = [Value::int(3), Value::string("x"), bag2];
        assert_eq!(hash_args(&args1), hash_args(&args2));
    }

    /// Part of #4462: Tuple args route through the additive-fingerprint fast
    /// path. Equal tuples (different Arcs) must hash equal, and the fp scheme
    /// must converge with the equal Seq representation (Tuple ≡ Seq ≡
    /// IntFunc(min=1) share the additive fingerprint).
    #[test]
    fn hash_args_tuple_fp_fast_path_is_content_deterministic() {
        let elem_set = || {
            Value::set(vec![
                Value::tuple(vec![Value::string("n1"), Value::int(1)]),
                Value::tuple(vec![Value::string("n2"), Value::int(2)]),
            ])
        };
        // dag == <<vs, es>> shape: a pair of sets.
        let mk_dag = || Value::tuple(vec![elem_set(), elem_set()]);
        let dag1 = mk_dag();
        let dag2 = mk_dag();
        assert_eq!(dag1, dag2);
        assert_eq!(hash_args(&[dag1.clone()]), hash_args(&[dag2.clone()]));

        // Representation convergence: an equal Seq hashes identically.
        let seq = Value::seq(vec![elem_set(), elem_set()]);
        assert_eq!(seq, dag1);
        assert_eq!(hash_args(&[seq]), hash_args(&[dag1.clone()]));

        // Different content → (overwhelmingly) different hash; at minimum the
        // values must not compare equal, so a collision degrades to a miss.
        let other = Value::tuple(vec![elem_set(), Value::set(vec![Value::int(9)])]);
        assert_ne!(other, dag1);
        assert_ne!(hash_args(&[other]), hash_args(&[dag1]));
    }

    /// Part of #4462: `hash_value_cache_key` (used by the zero-arg param LET
    /// dep hash) agrees with itself across structurally-equal values and
    /// distinguishes the fp fast path from the deep path by scheme tag.
    #[test]
    fn hash_value_cache_key_is_content_deterministic() {
        use std::hash::Hasher;
        let hash_one = |v: &Value| {
            let mut h = rustc_hash::FxHasher::default();
            hash_value_cache_key(v, &mut h);
            h.finish()
        };
        let mk_set = || {
            Value::set(vec![
                Value::tuple(vec![Value::string("a"), Value::int(1)]),
                Value::int(7),
            ])
        };
        assert_eq!(hash_one(&mk_set()), hash_one(&mk_set()));
        let scalar = Value::int(42);
        assert_eq!(hash_one(&scalar), hash_one(&Value::int(42)));
        assert_ne!(hash_one(&scalar), hash_one(&Value::int(43)));
    }

    #[test]
    fn nary_cache_stats_record_persistent_hit_and_miss() {
        reset_nary_cache_for_test();
        let key = test_key();
        nary_insert(
            key.clone(),
            persistent_insert_entry(Value::int(1), Value::int(42)),
        );
        NARY_CACHES.with(|caches| {
            let caches = caches.borrow();
            let bucket = caches.persistent.get(&key).unwrap();
            assert_eq!(bucket.len(), 1);
            assert!(
                !bucket.spilled(),
                "the common persistent singleton must remain inline"
            );
        });

        let ctx = EvalCtx::new();
        with_nary_stats_for_test(|| {
            let hit = nary_lookup(&key, &[Value::int(1)], |_ctx, _entry| true, &ctx)
                .expect("matching persistent entry should hit");
            assert!(matches!(
                hit,
                NaryCacheHit::Persistent(value) if value == Value::int(42)
            ));

            assert!(
                nary_lookup(&key, &[Value::int(2)], |_ctx, _entry| true, &ctx).is_none(),
                "different args should miss after scanning the persistent entry"
            );
        });

        let stats = nary_cache_stats_for_test();
        assert_eq!(stats.lookups, 2);
        assert_eq!(stats.persistent_key_hits, 2);
        assert_eq!(stats.persistent_entries_scanned, 2);
        assert_eq!(stats.persistent_hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.state_key_hits, 0);
        assert_eq!(stats.state_entries_scanned, 0);
        assert_eq!(stats.state_hits, 0);
    }

    #[test]
    fn nary_persistent_collision_bucket_spills_and_scans_exact_args() {
        reset_nary_cache_for_test();
        let key = test_key();

        // Reuse one routing key with unequal args to force a same-key bucket,
        // exactly as an args-hash collision would. Full argument equality must
        // still select the second entry.
        nary_insert(
            key.clone(),
            persistent_insert_entry(Value::int(1), Value::int(41)),
        );
        nary_insert(
            key.clone(),
            persistent_insert_entry(Value::int(2), Value::int(42)),
        );
        NARY_CACHES.with(|caches| {
            let caches = caches.borrow();
            let bucket = caches.persistent.get(&key).unwrap();
            assert_eq!(bucket.len(), 2);
            assert!(
                bucket.spilled(),
                "a two-entry bucket must use overflow storage"
            );
        });

        let ctx = EvalCtx::new();
        with_nary_stats_for_test(|| {
            let hit = nary_lookup(&key, &[Value::int(2)], |_ctx, _entry| true, &ctx)
                .expect("the exact second entry should hit after one collision");
            assert!(matches!(
                hit,
                NaryCacheHit::Persistent(value) if value == Value::int(42)
            ));
        });

        let stats = nary_cache_stats_for_test();
        assert_eq!(stats.lookups, 1);
        assert_eq!(stats.persistent_key_hits, 1);
        assert_eq!(stats.persistent_entries_scanned, 2);
        assert_eq!(stats.persistent_hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn nary_partition_clears_retain_persistent_then_clear_all_without_resetting_stats() {
        reset_nary_cache_for_test();
        let persistent_key = test_key();
        let mut state_key = test_key();
        state_key.def_loc += 1;

        nary_insert(
            persistent_key.clone(),
            persistent_insert_entry(Value::int(1), Value::int(42)),
        );
        nary_insert(
            state_key.clone(),
            state_entry(Value::int(1), Value::int(99), &Value::int(7)),
        );

        let ctx = EvalCtx::new();
        with_nary_stats_for_test(|| {
            assert!(
                nary_lookup(&persistent_key, &[Value::int(1)], |_ctx, _entry| true, &ctx,)
                    .is_some()
            );
        });
        let stats_before_clear = nary_cache_stats_for_test();

        clear_nary_state_partition();
        NARY_CACHES.with(|caches| {
            let caches = caches.borrow();
            assert!(caches.state.is_empty());
            let bucket = caches.persistent.get(&persistent_key).unwrap();
            assert_eq!(bucket.len(), 1);
            assert!(!bucket.spilled());
        });
        assert_eq!(nary_cache_stats_for_test(), stats_before_clear);

        clear_nary_all_partitions();
        NARY_CACHES.with(|caches| {
            let caches = caches.borrow();
            assert!(caches.state.is_empty());
            assert!(caches.persistent.is_empty());
        });
        assert_eq!(nary_cache_stats_for_test(), stats_before_clear);
    }

    #[test]
    fn nary_lookup_prefers_persistent_and_never_validates_it() {
        reset_nary_cache_for_test();
        let key = test_key();
        NARY_CACHES.with(|caches| {
            let mut caches = caches.borrow_mut();
            caches.persistent.insert(
                key.clone(),
                smallvec::smallvec![persistent_entry(Value::int(1), Value::int(41))],
            );
            caches.state.insert(
                key.clone(),
                vec![state_entry(Value::int(1), Value::int(99), &Value::int(7))],
            );
        });

        let ctx = EvalCtx::new();
        let validator_calls = std::cell::Cell::new(0usize);
        with_nary_stats_for_test(|| {
            let hit = nary_lookup(
                &key,
                &[Value::int(1)],
                |_ctx, _entry| {
                    validator_calls.set(validator_calls.get() + 1);
                    true
                },
                &ctx,
            )
            .expect("the exact persistent entry should win");
            assert!(matches!(
                hit,
                NaryCacheHit::Persistent(value) if value == Value::int(41)
            ));
        });

        assert_eq!(validator_calls.get(), 0);
        let stats = nary_cache_stats_for_test();
        assert_eq!(stats.persistent_key_hits, 1);
        assert_eq!(stats.persistent_entries_scanned, 1);
        assert_eq!(stats.persistent_hits, 1);
        assert_eq!(stats.state_key_hits, 0);
        assert_eq!(stats.state_entries_scanned, 0);
        assert_eq!(stats.state_hits, 0);
    }

    #[test]
    fn nary_state_hit_still_requires_validator_and_retains_deps() {
        reset_nary_cache_for_test();
        let key = test_key();
        NARY_CACHES.with(|caches| {
            caches.borrow_mut().state.insert(
                key.clone(),
                vec![state_entry(Value::int(1), Value::int(42), &Value::int(7))],
            );
        });

        let ctx = EvalCtx::new();
        let validator_calls = std::cell::Cell::new(0usize);
        let hit = nary_lookup(
            &key,
            &[Value::int(1)],
            |_ctx, entry| {
                validator_calls.set(validator_calls.get() + 1);
                !entry.deps.state.is_empty()
            },
            &ctx,
        )
        .expect("validator-approved state entry should hit");

        assert_eq!(validator_calls.get(), 1);
        match hit {
            NaryCacheHit::State(result) => {
                assert_eq!(result.value, Value::int(42));
                assert!(!result.deps.state.is_empty());
            }
            NaryCacheHit::Persistent(_) => panic!("state entry returned as persistent"),
        }
    }

    /// `nary_lookup_value` must mirror `nary_lookup`'s probe contract exactly:
    /// persistent entries win first and are never validated.
    #[test]
    fn nary_lookup_value_persistent_hit_skips_validator() {
        reset_nary_cache_for_test();
        let key = test_key();
        NARY_CACHES.with(|caches| {
            let mut caches = caches.borrow_mut();
            caches.persistent.insert(
                key.clone(),
                smallvec::smallvec![persistent_entry(Value::int(1), Value::int(41))],
            );
            caches.state.insert(
                key.clone(),
                vec![state_entry(Value::int(1), Value::int(99), &Value::int(7))],
            );
        });

        let ctx = EvalCtx::new();
        let validator_calls = std::cell::Cell::new(0usize);
        let value = nary_lookup_value(
            &key,
            &[Value::int(1)],
            |_ctx, _entry| {
                validator_calls.set(validator_calls.get() + 1);
                true
            },
            &ctx,
        )
        .expect("the exact persistent entry should win");
        assert_eq!(value, Value::int(41));
        assert_eq!(validator_calls.get(), 0);
        assert!(
            nary_lookup_value(&key, &[Value::int(2)], |_ctx, _entry| false, &ctx).is_none(),
            "different args (and a rejecting validator) must miss"
        );
    }

    /// The fused hit path must still validate state entries and propagate
    /// their deps (by reference, from under the cache borrow) into the
    /// enclosing operator dep frame — identical to `nary_lookup` + finishing
    /// the `NaryCacheHit::State` at the callsite.
    #[test]
    fn nary_lookup_value_state_hit_validates_and_propagates_deps() {
        use crate::cache::OpDepGuard;

        reset_nary_cache_for_test();
        let key = test_key();
        NARY_CACHES.with(|caches| {
            caches.borrow_mut().state.insert(
                key.clone(),
                vec![state_entry(Value::int(1), Value::int(42), &Value::int(7))],
            );
        });

        let ctx = EvalCtx::new();
        let validator_calls = std::cell::Cell::new(0usize);
        let outer = OpDepGuard::from_ctx(&ctx, 0);
        let value = nary_lookup_value(
            &key,
            &[Value::int(1)],
            |_ctx, entry| {
                validator_calls.set(validator_calls.get() + 1);
                !entry.deps.state.is_empty()
            },
            &ctx,
        )
        .expect("validator-approved state entry should hit");
        assert_eq!(value, Value::int(42));
        assert_eq!(validator_calls.get(), 1);
        let propagated = outer.try_take_deps().expect("outer dep frame");
        assert!(
            propagated.state.contains_key(&VarIndex(0)),
            "state-hit deps must be propagated to the enclosing op frame"
        );
    }

    #[test]
    fn nary_insert_preserves_sixteen_entry_fifo_per_partition() {
        reset_nary_cache_for_test();
        let persistent_key = test_key();
        let mut state_key = test_key();
        state_key.def_loc += 1;

        for i in 0..=NARY_OP_CACHE_MAX_ENTRIES_PER_KEY {
            let arg = Value::int(i as i64);
            nary_insert(
                persistent_key.clone(),
                NaryOpCacheEntry {
                    args: Arc::from([arg.clone()]),
                    result: CachedOpResult {
                        value: arg.clone(),
                        deps: OpEvalDeps::default(),
                    },
                },
            );
            nary_insert(
                state_key.clone(),
                state_entry(arg.clone(), arg, &Value::int(7)),
            );
        }

        NARY_CACHES.with(|caches| {
            let caches = caches.borrow();
            let persistent = caches.persistent.get(&persistent_key).unwrap();
            let state = caches.state.get(&state_key).unwrap();
            assert_eq!(persistent.len(), NARY_OP_CACHE_MAX_ENTRIES_PER_KEY);
            assert!(persistent.spilled());
            assert_eq!(state.len(), NARY_OP_CACHE_MAX_ENTRIES_PER_KEY);
            assert_eq!(persistent.first().unwrap().args[0], Value::int(1));
            assert_eq!(state.first().unwrap().args[0], Value::int(1));
            assert_eq!(persistent.last().unwrap().args[0], Value::int(16));
            assert_eq!(state.last().unwrap().args[0], Value::int(16));
        });
    }

    #[test]
    fn nary_constant_fallback_uses_only_exact_persistent_args() {
        reset_nary_cache_for_test();
        let stored_key = test_key();
        nary_insert(
            stored_key.clone(),
            NaryOpCacheEntry {
                args: Arc::from([Value::int(1)]),
                result: CachedOpResult {
                    value: Value::int(42),
                    deps: OpEvalDeps::default(),
                },
            },
        );
        let lookup_key = NaryOpCacheKey {
            is_next_state: !stored_key.is_next_state,
            ..stored_key
        };

        assert_eq!(
            nary_constant_fallback(&lookup_key, &[Value::int(1)]),
            Some(Value::int(42))
        );
        assert_eq!(
            nary_constant_fallback(&lookup_key, &[Value::int(2)]),
            None,
            "hash-bucket collisions must still compare full arguments"
        );
    }

    // ------------------------------------------------------------------
    // #GameOfLife: state-dep validation via the binding-chain / env fallback
    // when `state_env` is absent (Next action / property evaluation binds the
    // unprimed state in `env` rather than the state array).
    // ------------------------------------------------------------------

    /// Build an `EvalCtx` whose sole state var lives in the `env` HashMap
    /// (no state array), mirroring GameOfLife's `Next` evaluation context.
    fn ctx_with_env_state_var(name: &str, value: Value) -> EvalCtx {
        let reg = crate::var_index::VarRegistry::from_names([name]);
        let mut ctx = EvalCtx::new();
        let s = ctx.stable_mut();
        s.shared = Arc::new(crate::SharedCtx::with_var_registry(reg));
        let mut env = crate::Env::default();
        env.insert(Arc::from(name), value);
        s.env = Arc::new(env);
        // state_env deliberately left None.
        ctx
    }

    /// A cached zero-arg result carrying a single unprimed state dep on var 0.
    fn entry_with_state_dep(recorded: &Value, cached: Value) -> CachedOpResult {
        let mut deps = OpEvalDeps::default();
        deps.record_state(VarIndex(0), recorded);
        assert!(!deps.inconsistent, "single record must not be inconsistent");
        CachedOpResult {
            value: cached,
            deps,
        }
    }

    /// Within a single state, `sc`-style entries with a `state` dep must be a
    /// cache HIT when `state_env` is None but the var is bound in `env` and
    /// still holds the recorded value. Before the fix this returned false
    /// unconditionally, forcing re-materialization on every use.
    #[test]
    fn state_dep_caches_within_state_via_env_fallback() {
        let grid = Value::int(7);
        let ctx = ctx_with_env_state_var("grid", grid.clone());
        let entry = entry_with_state_dep(&grid, Value::int(42));
        assert!(
            op_cache_entry_valid(&ctx, &entry),
            "state dep must validate against env when state_env is absent"
        );
    }

    /// Across states the recorded value diverges from the new `env` value, so
    /// the entry MUST fail validation (fail-closed → re-evaluate). This is the
    /// soundness guarantee: a stale cache never produces a wrong successor.
    #[test]
    fn state_dep_invalidates_across_states_via_env_fallback() {
        let old_grid = Value::int(7);
        let new_grid = Value::int(8);
        let entry = entry_with_state_dep(&old_grid, Value::int(42));

        // Same var, different value in env (a later state) → must be invalid.
        let ctx_next = ctx_with_env_state_var("grid", new_grid);
        assert!(
            !op_cache_entry_valid(&ctx_next, &entry),
            "state dep must invalidate when env value changed (fail-closed)"
        );
    }

    /// Fail-closed when the state var is absent from every resolution source.
    #[test]
    fn state_dep_fails_closed_when_var_unbound() {
        let grid = Value::int(7);
        let entry = entry_with_state_dep(&grid, Value::int(42));

        // Registry knows "grid" but env has no binding for it and state_env is None.
        let reg = crate::var_index::VarRegistry::from_names(["grid"]);
        let mut ctx = EvalCtx::new();
        let s = ctx.stable_mut();
        s.shared = Arc::new(crate::SharedCtx::with_var_registry(reg));
        s.env = Arc::new(crate::Env::default());
        assert!(
            !op_cache_entry_valid(&ctx, &entry),
            "unbound state var must fail validation"
        );
    }
}
