// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Zero-argument operator caches with lifecycle partitioning.
//!
//! Part of #3100: Caches are split into two physical partitions by lifecycle:
//!   - state partition: entries with non-empty deps, cleared on state boundary
//!     via simple `.clear()`.
//!   - persistent partition: entries with empty deps, survive state boundaries.
//!     Cleared only on run/test reset.
//!
//! Part of #4053/#3962: Both partitions consolidated into a single TLS struct
//! (`ZERO_ARG_CACHES`). Previously two separate `thread_local!` declarations,
//! each requiring a separate `_tlv_get_addr` call on macOS (~5ns each).
//! `zero_arg_lookup` probed both partitions sequentially = 2 TLS accesses per
//! lookup. Now a single TLS access covers both partitions, saving ~5ns per
//! `zero_arg_lookup` on the BFS hot path.
//!
//! This eliminates the O(cache_size) retain scan that previously ran once per
//! BFS parent state during inline liveness (81K calls for Huang).
//!
//! Fix #2462: Both partitions use dep validation via op_cache_entry_valid().
//! No unvalidated constant cache — the persistent partition stores validated
//! entries that happen to have empty deps.
//!
//! Part of #2744 decomposition from eval_cache.rs.

use super::dep_tracking::OpEvalDeps;
use super::op_result_cache::CachedOpResult;
use crate::EvalCtx;
use rustc_hash::FxHashMap;
use tla_core::name_intern::NameId;

/// Cache key type for zero-arg caches.
///
/// Fields: (shared_id, local_ops_id, instance_subs_id, op_name, def_loc, is_next_state)
///
/// BUG FIX #3097: Includes `local_ops_id` and `instance_subs_id` to distinguish different
/// INSTANCE contexts and LET-scoped operator environments. Without these, operators with
/// the same name and definition location but different local_ops or instance_substitutions
/// (e.g., different INSTANCE instantiations) could share cached results incorrectly.
/// This matches the discrimination already accepted for OpResultCacheKey and NaryOpCacheKey.
///
/// BUG FIX #277: `def_loc` (span.start) distinguishes different LET blocks.
/// BUG FIX #295: `is_next_state` distinguishes primed vs unprimed lookups.
/// Part of #3099/#3157: `local_ops_id` and `instance_subs_id` are now stable u64
/// content-based fingerprints (matching SubstCacheKey and NaryOpCacheKey), replacing
/// the previous `Arc::as_ptr() as usize` pattern that produced different keys for
/// logically identical scopes.
pub(crate) type ZeroArgCacheKey = (u64, u64, u64, NameId, u32, bool);

/// Cache key for the fingerprint-keyed transition partition.
///
/// Fields: (shared_id, local_ops_id, instance_subs_id, op_name, def_loc, state_fp)
///
/// Unlike [`ZeroArgCacheKey`] there is no `is_next_state` discriminator: a
/// state-level operator evaluated in `Next` mode against successor `s` has
/// exactly the value it has when evaluated in `Current` mode once `s` becomes
/// the parent, so entries are stored mode-normalized (deps on the `state`
/// side) and keyed by the fingerprint of whichever bound array the evaluation
/// actually read. This doubles the hit rate: a successor's refinement-mapped
/// values are computed once and reused both for later transitions into the
/// same state and for every transition out of it.
pub(crate) type ZeroArgTransitionCacheKey = (u64, u64, u64, NameId, u32, u64);

// Part of #4053/#3962: Consolidated zero-arg cache struct holding both partitions
// and debug stats in a single TLS entry. Previously three separate thread_local!
// declarations (ZERO_ARG_OP_CACHE + ZERO_ARG_PERSISTENT_CACHE + ZERO_ARG_STATS)
// requiring 3 TLS accesses. Now 1 TLS access covers all zero-arg state.
//
// Key: (shared_id, local_ops_id, instance_subs_id, op_name, def_loc, is_next_state)
// Value: Vec of cached results per key with dep-based validation.
pub(crate) struct ZeroArgCaches {
    /// State partition: entries with non-empty deps, cleared on state boundary.
    pub(crate) state: FxHashMap<ZeroArgCacheKey, Vec<CachedOpResult>>,
    /// Persistent partition: entries with empty deps, survive state boundaries.
    /// Cleared only on run/test reset. Part of #3100.
    pub(crate) persistent: FxHashMap<ZeroArgCacheKey, Vec<CachedOpResult>>,
    /// Transition partition: fingerprint-keyed zero-arg values for
    /// implied-action (`[][A]_v`) checking. Survives state boundaries (that
    /// is the whole point: the per-transition `clear_for_bound_state_eval_scope`
    /// clears the state partition, forcing refinement-mapping operators like
    /// EWD998PCal's `token` / `pending` to be re-evaluated for every
    /// transition). Cleared on run reset and BOUNDED (see
    /// [`TransitionPartition`]) — 2026-07 memory audit: the previous
    /// single-map 4M-entry wholesale-clear cap let this partition retain a
    /// `Value` tree per (operator × reachable state) for the whole run,
    /// which alone was ~570 MB (69%) of EWD998PCal's peak RSS.
    ///
    /// Entries are only stored for evaluations whose tracked deps are
    /// exclusively single-sided state-variable reads (see
    /// `transition_deps_eligible` in eval_ident_zero_arg.rs), with deps
    /// normalized to the `state` side. Every hit re-validates all recorded
    /// dep values against the currently bound arrays, so a fingerprint
    /// collision degrades to a miss and can never yield a wrong value.
    pub(crate) transition: TransitionPartition,
    /// Part of #3962: Debug stats consolidated from separate ZERO_ARG_STATS thread_local.
    /// Only written when `TY_ZERO_ARG_CACHE_STATS=1` environment variable is set.
    pub(crate) stats: ZeroArgCacheStats,
}

impl ZeroArgCaches {
    fn new() -> Self {
        ZeroArgCaches {
            state: FxHashMap::default(),
            persistent: FxHashMap::default(),
            transition: TransitionPartition::from_env(),
            stats: ZeroArgCacheStats::new(),
        }
    }
}

std::thread_local! {
    pub(crate) static ZERO_ARG_CACHES: std::cell::RefCell<ZeroArgCaches> =
        std::cell::RefCell::new(ZeroArgCaches::new());
}

/// Build a zero-arg cache key from the evaluation context.
///
/// Part of #3099/#3157: Uses content-based u64 fingerprints from `scope_ids`
/// for `local_ops_id` and `instance_subs_id`, matching `subst_cache_key()`
/// and `NaryOpCacheKey`. Replaces the previous `Arc::as_ptr` pointer-identity
/// convention which was vulnerable to allocator address reuse.
///
/// Part of #3097: Single helper ensures lookup and insert use the same key shape,
/// preventing drift between the three call sites in eval_ident_zero_arg.rs.
#[inline]
pub(crate) fn zero_arg_cache_key(
    ctx: &EvalCtx,
    name_id: NameId,
    def_loc: u32,
    is_next_state: bool,
) -> ZeroArgCacheKey {
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
    (
        ctx.shared.id,
        local_ops_id,
        instance_subs_id,
        name_id,
        def_loc,
        is_next_state,
    )
}

/// Maximum entries per key in zero-arg caches to prevent unbounded memory growth.
/// When exceeded, oldest entries are evicted.
pub(crate) const ZERO_ARG_CACHE_MAX_ENTRIES_PER_KEY: usize = 16;

// Part of #3805: Debug counters for zero-arg cache performance analysis.
// Enabled via TY_ZERO_ARG_CACHE_STATS=1 environment variable.
// Uses feature_flag! (not debug_flag!) to work in release mode.
feature_flag!(pub(crate) debug_zero_arg_stats, "TY_ZERO_ARG_CACHE_STATS");

pub(crate) struct ZeroArgCacheStats {
    pub(crate) primary_hits: u64,
    pub(crate) canonical_hits: u64,
    pub(crate) constant_fallback_hits: u64,
    pub(crate) misses: u64,
    /// Misses where deps were persistent (constant) — indicates canonical key
    /// should have caught this but didn't (first eval only).
    pub(crate) persistent_misses: u64,
    /// Misses where instance_lazy_read taint prevented persistent classification.
    pub(crate) instance_taint_misses: u64,
    /// Top miss operators by name.
    pub(crate) miss_names: std::collections::HashMap<String, u64>,
}

impl ZeroArgCacheStats {
    fn new() -> Self {
        Self {
            primary_hits: 0,
            canonical_hits: 0,
            constant_fallback_hits: 0,
            misses: 0,
            persistent_misses: 0,
            instance_taint_misses: 0,
            miss_names: std::collections::HashMap::new(),
        }
    }
}

// Part of #3962: ZERO_ARG_STATS consolidated into ZeroArgCaches struct above.
// Previously a separate thread_local!, now accessed as ZERO_ARG_CACHES.stats.

#[inline]
pub(crate) fn record_zero_arg_primary_hit() {
    if debug_zero_arg_stats() {
        ZERO_ARG_CACHES.with(|c| c.borrow_mut().stats.primary_hits += 1);
    }
}

#[inline]
pub(crate) fn record_zero_arg_canonical_hit() {
    if debug_zero_arg_stats() {
        ZERO_ARG_CACHES.with(|c| c.borrow_mut().stats.canonical_hits += 1);
    }
}

#[inline]
pub(crate) fn record_zero_arg_constant_fallback_hit() {
    if debug_zero_arg_stats() {
        ZERO_ARG_CACHES.with(|c| c.borrow_mut().stats.constant_fallback_hits += 1);
    }
}

#[inline]
pub(crate) fn record_zero_arg_miss(name: &str, deps: &OpEvalDeps) {
    if debug_zero_arg_stats() {
        ZERO_ARG_CACHES.with(|c| {
            let mut c = c.borrow_mut();
            c.stats.misses += 1;
            if deps_are_persistent(deps) {
                c.stats.persistent_misses += 1;
            }
            if deps.instance_lazy_read {
                c.stats.instance_taint_misses += 1;
            }
            *c.stats.miss_names.entry(name.to_string()).or_default() += 1;
        });
    }
}

/// Print cache stats summary. Called at end of model checking run.
pub(crate) fn print_zero_arg_cache_stats() {
    if !debug_zero_arg_stats() {
        return;
    }
    ZERO_ARG_CACHES.with(|c| {
        let c = c.borrow();
        let s = &c.stats;
        let total = s.primary_hits + s.canonical_hits + s.constant_fallback_hits + s.misses;
        if total == 0 {
            return;
        }
        eprintln!("\n=== Zero-Arg Cache Stats ===");
        eprintln!("Total lookups:          {}", total);
        eprintln!(
            "Primary hits:           {} ({:.1}%)",
            s.primary_hits,
            s.primary_hits as f64 / total as f64 * 100.0
        );
        eprintln!(
            "Canonical hits:         {} ({:.1}%)",
            s.canonical_hits,
            s.canonical_hits as f64 / total as f64 * 100.0
        );
        eprintln!(
            "Constant fallback hits: {} ({:.1}%)",
            s.constant_fallback_hits,
            s.constant_fallback_hits as f64 / total as f64 * 100.0
        );
        eprintln!(
            "Misses:                 {} ({:.1}%)",
            s.misses,
            s.misses as f64 / total as f64 * 100.0
        );
        eprintln!(
            "  Persistent misses:    {} (first eval of constant ops)",
            s.persistent_misses
        );
        eprintln!(
            "  Instance taint:       {} (instance_lazy_read prevented persistent)",
            s.instance_taint_misses
        );
        eprintln!("\nTop miss operators:");
        let mut miss_vec: Vec<_> = s.miss_names.iter().collect();
        miss_vec.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in miss_vec.iter().take(20) {
            eprintln!("  {:>8}  {}", count, name);
        }
        eprintln!("============================\n");
    });
}

/// Part of #3100: Check if deps qualify for the persistent partition (empty deps).
/// Fix #3447: Also rejects deps tainted by INSTANCE lazy binding reads.
#[inline]
pub(crate) fn deps_are_persistent(deps: &OpEvalDeps) -> bool {
    !deps.inconsistent
        && !deps.instance_lazy_read
        && deps.state.is_empty()
        && deps.next.is_empty()
        && deps.local.is_empty()
        && deps.tlc_level.is_none()
}

/// Part of #3100/#4053: Insert into the appropriate partition based on deps.
/// Empty deps -> persistent partition, non-empty deps -> state partition.
/// Part of #4053: Single TLS access for both partitions.
#[inline]
pub(crate) fn zero_arg_insert(key: ZeroArgCacheKey, entry: CachedOpResult) {
    ZERO_ARG_CACHES.with(|caches| {
        let mut caches = caches.borrow_mut();
        let target = if deps_are_persistent(&entry.deps) {
            &mut caches.persistent
        } else {
            &mut caches.state
        };
        let entries = target.entry(key).or_default();
        while entries.len() >= ZERO_ARG_CACHE_MAX_ENTRIES_PER_KEY {
            entries.remove(0);
        }
        entries.push(entry);
    });
}

/// Part of #3100/#4053: Lookup probing state partition first, then persistent.
/// Returns the first entry for which `validator` returns true.
/// Part of #4053: Single TLS access for both partitions (was 2 before).
#[inline]
pub(crate) fn zero_arg_lookup(
    key: &ZeroArgCacheKey,
    validator: impl Fn(&CachedOpResult) -> bool,
) -> Option<CachedOpResult> {
    zero_arg_lookup_inner(key, validator).inspect(|_| {
        tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::ZeroArgCacheHit);
    })
}

#[inline]
fn zero_arg_lookup_inner(
    key: &ZeroArgCacheKey,
    validator: impl Fn(&CachedOpResult) -> bool,
) -> Option<CachedOpResult> {
    ZERO_ARG_CACHES.with(|caches| {
        let caches = caches.borrow();
        // Probe state partition first (hot entries for current state)
        if let Some(entries) = caches.state.get(key) {
            for entry in entries {
                if validator(entry) {
                    return Some(entry.clone());
                }
            }
        }
        // Probe persistent partition
        if let Some(entries) = caches.persistent.get(key) {
            for entry in entries {
                if validator(entry) {
                    return Some(entry.clone());
                }
            }
        }
        None
    })
}

// Lever 2 (#EWD998PCal) kill switch: `TY_NO_ZERO_ARG_PROBE_MERGE=1` restores
// the previous one-TLS-access-per-probe/store sequence in
// `eval_general_zero_arg` for differential validation.
feature_flag!(pub(crate) no_zero_arg_probe_merge, "TY_NO_ZERO_ARG_PROBE_MERGE");

/// Which probe produced a combined-lookup hit. The caller replicates the
/// exact per-partition hit bookkeeping (stats counter + dep-propagation
/// semantics differ between partitions).
pub(crate) enum ZeroArgProbeHit {
    /// Scope-keyed state/persistent partition hit (validator-approved).
    Primary(CachedOpResult),
    /// Fingerprint-keyed transition partition hit (validator-approved;
    /// deps are state-side normalized — translate per evaluation mode).
    Transition(CachedOpResult),
    /// Scope-normalized canonical persistent hit (persistent deps only).
    Canonical(CachedOpResult),
}

/// Combined single-TLS-access probe for the general zero-arg lookup sequence
/// (Lever 2, #EWD998PCal). Performs — in EXACTLY the same order, with the
/// same validators and the same first-hit-wins short-circuiting as the
/// previous three separate probes in `eval_general_zero_arg`:
///   1. scope-keyed state partition, then persistent partition
///      (`primary_validator`, mirrors [`zero_arg_lookup`]);
///   2. when `tkey` is `Some`: fingerprint-keyed transition partition
///      (`transition_validator`, mirrors [`zero_arg_transition_lookup`]);
///   3. scope-normalized canonical persistent lookup
///      (mirrors [`zero_arg_canonical_lookup`]).
/// The only behavioral difference vs. the separate probes is the number of
/// `thread_local!` accesses (3 → 1); every soundness-relevant validation is
/// unchanged and still runs per hit.
pub(crate) fn zero_arg_probe_all(
    key: &ZeroArgCacheKey,
    tkey: Option<&ZeroArgTransitionCacheKey>,
    canonical_key: &ZeroArgCacheKey,
    primary_validator: impl Fn(&CachedOpResult) -> bool,
    transition_validator: impl Fn(&CachedOpResult) -> bool,
) -> Option<ZeroArgProbeHit> {
    zero_arg_probe_all_inner(
        key,
        tkey,
        canonical_key,
        primary_validator,
        transition_validator,
    )
    .inspect(|_| {
        tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::ZeroArgCacheHit);
    })
}

fn zero_arg_probe_all_inner(
    key: &ZeroArgCacheKey,
    tkey: Option<&ZeroArgTransitionCacheKey>,
    canonical_key: &ZeroArgCacheKey,
    primary_validator: impl Fn(&CachedOpResult) -> bool,
    transition_validator: impl Fn(&CachedOpResult) -> bool,
) -> Option<ZeroArgProbeHit> {
    ZERO_ARG_CACHES.with(|caches| {
        let caches = caches.borrow();
        // 1. Primary: state partition first (hot entries for current state),
        //    then persistent — identical to zero_arg_lookup.
        if let Some(entries) = caches.state.get(key) {
            for entry in entries {
                if primary_validator(entry) {
                    return Some(ZeroArgProbeHit::Primary(entry.clone()));
                }
            }
        }
        if let Some(entries) = caches.persistent.get(key) {
            for entry in entries {
                if primary_validator(entry) {
                    return Some(ZeroArgProbeHit::Primary(entry.clone()));
                }
            }
        }
        // 2. Transition partition — identical to zero_arg_transition_lookup.
        if let Some(tkey) = tkey {
            if let Some(entry) = caches.transition.get(tkey) {
                if transition_validator(entry) {
                    return Some(ZeroArgProbeHit::Transition(entry.clone()));
                }
            }
        }
        // 3. Canonical persistent — identical to zero_arg_canonical_lookup.
        if let Some(entries) = caches.persistent.get(canonical_key) {
            for entry in entries {
                if deps_are_persistent(&entry.deps) {
                    return Some(ZeroArgProbeHit::Canonical(entry.clone()));
                }
            }
        }
        None
    })
}

/// Combined single-TLS-access store for the general zero-arg miss path
/// (Lever 2, #EWD998PCal). Applies — in EXACTLY the same order and with the
/// same routing/eviction rules as the previous separate stores:
///   1. `canonical`: canonical-key persistent insert
///      (mirrors [`zero_arg_canonical_insert`]; caller must pass
///      persistent-deps entries only);
///   2. `primary`: deps-routed state/persistent insert with per-key eviction
///      (mirrors [`zero_arg_insert`]);
///   3. `transition`: transition-partition insert with wholesale-clear cap
///      (mirrors [`zero_arg_transition_insert`]).
pub(crate) fn zero_arg_store_all(
    primary: Option<(ZeroArgCacheKey, CachedOpResult)>,
    canonical: Option<(ZeroArgCacheKey, CachedOpResult)>,
    transition: Option<(ZeroArgTransitionCacheKey, CachedOpResult)>,
) {
    if primary.is_none() && canonical.is_none() && transition.is_none() {
        return;
    }
    ZERO_ARG_CACHES.with(|caches| {
        let mut caches = caches.borrow_mut();
        if let Some((ckey, centry)) = canonical {
            debug_assert!(
                deps_are_persistent(&centry.deps),
                "canonical insert must only be used for persistent entries"
            );
            let entries = caches.persistent.entry(ckey).or_default();
            // Canonical keys have at most 1 entry per key (constant value is unique).
            if entries.is_empty() {
                entries.push(centry);
            }
        }
        if let Some((key, entry)) = primary {
            let target = if deps_are_persistent(&entry.deps) {
                &mut caches.persistent
            } else {
                &mut caches.state
            };
            let entries = target.entry(key).or_default();
            while entries.len() >= ZERO_ARG_CACHE_MAX_ENTRIES_PER_KEY {
                entries.remove(0);
            }
            entries.push(entry);
        }
        if let Some((tkey, tentry)) = transition {
            caches.transition.insert(tkey, tentry);
        }
    });
}

/// Get total number of stored entries across both partitions.
#[cfg(test)]
pub(crate) fn zero_arg_op_cache_entry_count() -> usize {
    ZERO_ARG_CACHES.with(|caches| {
        let caches = caches.borrow();
        let state_count: usize = caches.state.values().map(std::vec::Vec::len).sum();
        let persistent_count: usize = caches.persistent.values().map(std::vec::Vec::len).sum();
        state_count + persistent_count
    })
}

/// Clear all partitions for test isolation.
#[cfg(test)]
pub(crate) fn zero_arg_op_cache_clear() {
    ZERO_ARG_CACHES.with(|caches| {
        let mut caches = caches.borrow_mut();
        caches.state.clear();
        caches.persistent.clear();
        caches.transition.clear();
    });
}

/// Part of #3109: Constant-entry fallback lookup with flipped `is_next_state`.
///
/// During ENABLED scope, `current_state_lookup_mode` may differ from when the
/// entry was cached during BFS. For constant operators (empty deps), the
/// `is_next_state` flag doesn't affect correctness — constant results are
/// state-independent. This function retries the lookup with the opposite
/// `is_next_state` value, accepting only constant entries.
///
/// Part of #3100: Probes persistent partition only (constant entries live there).
/// Part of #4053: Single TLS access via consolidated struct.
#[inline]
pub(crate) fn zero_arg_constant_fallback(key: &ZeroArgCacheKey) -> Option<super::CachedOpResult> {
    let flipped_key = (key.0, key.1, key.2, key.3, key.4, !key.5);
    ZERO_ARG_CACHES.with(|caches| {
        let caches = caches.borrow();
        let entries = caches.persistent.get(&flipped_key)?;
        for entry in entries {
            if deps_are_persistent(&entry.deps) {
                return Some(entry.clone());
            }
        }
        None
    })
}

/// Part of #3100/#4053: Clear state partition only. Persistent entries survive.
/// Replaces the old retain_only_zero_arg_constant_entries() retain scan.
#[inline]
pub(crate) fn clear_zero_arg_state_partition() {
    ZERO_ARG_CACHES.with(|c| c.borrow_mut().state.clear());
}

/// Part of #3100/#4053: Clear all partitions (run reset / test reset).
#[inline]
pub(crate) fn clear_zero_arg_all_partitions() {
    ZERO_ARG_CACHES.with(|c| {
        let mut c = c.borrow_mut();
        c.state.clear();
        c.persistent.clear();
        c.transition.clear();
    });
}

// ---------------------------------------------------------------------------
// Transition partition (fingerprint-keyed) — implied-action derived-value cache
// ---------------------------------------------------------------------------

/// Legacy hard cap on transition-partition entries (kill-switch mode,
/// `TY_IMPLIED_FP_CACHE_CAP=0`). When an insert would exceed the cap, the
/// partition is cleared wholesale (the values are pure memoization —
/// dropping them costs at most one re-evaluation per live state, never
/// correctness).
pub(crate) const ZERO_ARG_TRANSITION_CACHE_MAX_ENTRIES: usize = 4_000_000;

/// Default TOTAL entry budget for the bounded transition partition
/// (2026-07 memory audit). Split across two generations (current + previous)
/// that rotate when the current generation fills, so the most recently
/// inserted half is always retained — a streaming-BFS-friendly working
/// window with a hard memory bound.
///
/// Sizing: entries are one per (derived operator × recently touched state);
/// each retains the cached `Value` tree plus dep snapshots (~0.9 KB measured
/// on record/function-heavy EWD998PCal). The BFS reuse window is one to two
/// levels of states (transitions into a state cluster around its discovery
/// level; transitions out happen at dequeue one level later), so ONE
/// GENERATION must hold a full level's (states × operators) entries or
/// rotations start dropping entries mid-level. Measured on EWD998PCal
/// (level width ~31k states × 2 ops ≈ 62k entries/level, clean big core,
/// min-of-3): a 64k total budget (32k/generation) rotates twice per level
/// and cost +12% wall; 128k total (64k/generation ≥ one level) restores the
/// legacy hit pattern within noise while bounding the partition to ~120 MB
/// worst-case (vs ~570 MB unbounded). Override with
/// `TY_IMPLIED_FP_CACHE_CAP=<total entries>`; `0` restores the legacy
/// unbounded-until-4M-wholesale-clear behavior.
pub(crate) const DEFAULT_IMPLIED_FP_CACHE_CAP: usize = 131_072;

/// Total transition-partition entry budget from `TY_IMPLIED_FP_CACHE_CAP`.
/// `Some(n)` = bounded two-generation mode with per-generation cap `n/2`;
/// `None` = legacy mode (`0` explicitly requested).
fn implied_fp_cache_cap_from_env() -> Option<usize> {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("TY_IMPLIED_FP_CACHE_CAP")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            Some(0) => None, // Kill switch: legacy single-map behavior.
            Some(n) => Some(n.max(2)),
            None => Some(DEFAULT_IMPLIED_FP_CACHE_CAP),
        }
    })
}

// AB kill switch for the bounded transition-partition rotation policy. The
// default recycles both hash-table allocations across rotations; setting this
// restores the previous `take`/drop behavior without adding a branch to the
// lookup hot path.
feature_flag!(
    pub(crate) no_implied_fp_cache_recycle,
    "TY_NO_IMPLIED_FP_CACHE_RECYCLE"
);

/// Bounded two-generation storage for the fingerprint-keyed transition
/// partition (2026-07 memory audit).
///
/// Eviction can never affect correctness: entries are pure memoization and
/// every hit still runs the caller's dep revalidation. The bound only trades
/// (at most) one extra evaluation per dropped entry for a hard memory cap.
///
/// Kill switch: `TY_IMPLIED_FP_CACHE_CAP=0` restores the exact legacy
/// behavior (single map, wholesale clear at
/// [`ZERO_ARG_TRANSITION_CACHE_MAX_ENTRIES`]).
pub(crate) struct TransitionPartition {
    /// Current generation: all inserts land here.
    cur: FxHashMap<ZeroArgTransitionCacheKey, CachedOpResult>,
    /// Previous generation: read-only survivors of the last rotation.
    /// Always empty in legacy mode.
    prev: FxHashMap<ZeroArgTransitionCacheKey, CachedOpResult>,
    /// `Some(per_generation_cap)` = bounded mode; `None` = legacy mode.
    cap_per_gen: Option<usize>,
    /// Reuse the two map allocations when bounded generations rotate. Cached
    /// once per partition so the AB switch is consulted only on rotation.
    recycle_generations: bool,
}

impl TransitionPartition {
    pub(crate) fn from_env() -> Self {
        Self::new(
            implied_fp_cache_cap_from_env().map(|total| (total / 2).max(1)),
            !no_implied_fp_cache_recycle(),
        )
    }

    fn new(cap_per_gen: Option<usize>, recycle_generations: bool) -> Self {
        Self {
            cur: FxHashMap::default(),
            prev: FxHashMap::default(),
            cap_per_gen,
            recycle_generations,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_generation_cap_for_test(
        cap_per_gen: usize,
        recycle_generations: bool,
    ) -> Self {
        Self::new(Some(cap_per_gen.max(1)), recycle_generations)
    }

    #[inline]
    pub(crate) fn get(&self, key: &ZeroArgTransitionCacheKey) -> Option<&CachedOpResult> {
        self.cur.get(key).or_else(|| self.prev.get(key))
    }

    #[inline]
    pub(crate) fn get_mut(
        &mut self,
        key: &ZeroArgTransitionCacheKey,
    ) -> Option<&mut CachedOpResult> {
        match self.cur.get_mut(key) {
            Some(entry) => Some(entry),
            None => self.prev.get_mut(key),
        }
    }

    #[inline]
    pub(crate) fn insert(&mut self, key: ZeroArgTransitionCacheKey, entry: CachedOpResult) {
        match self.cap_per_gen {
            Some(cap) => {
                // Overwrites of a key already in `cur` never grow the map, so
                // only rotate when inserting would push `cur` past the cap.
                if self.cur.len() >= cap && !self.cur.contains_key(&key) {
                    if self.recycle_generations {
                        // Drop the old previous generation while retaining its
                        // allocation, then make that empty allocation the new
                        // current map. The old current generation becomes the
                        // new previous map. Entry/eviction semantics are
                        // identical to `prev = take(cur)`, but subsequent
                        // generations avoid rebuilding a 65k-entry hash table.
                        self.prev.clear();
                        std::mem::swap(&mut self.prev, &mut self.cur);
                    } else {
                        self.prev = std::mem::take(&mut self.cur);
                    }
                }
                self.cur.insert(key, entry);
            }
            None => {
                // Legacy: single map, wholesale clear at the 4M hard cap.
                if self.cur.len() >= ZERO_ARG_TRANSITION_CACHE_MAX_ENTRIES {
                    self.cur.clear();
                }
                self.cur.insert(key, entry);
            }
        }
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.cur.clear();
        self.prev.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.cur.len() + self.prev.len()
    }

    #[cfg(test)]
    pub(crate) fn generation_lens_for_test(&self) -> (usize, usize) {
        (self.cur.len(), self.prev.len())
    }

    #[cfg(test)]
    pub(crate) fn generation_capacities_for_test(&self) -> (usize, usize) {
        (self.cur.capacity(), self.prev.capacity())
    }
}

// Kill switch for the fingerprint-keyed implied-action transition cache.
// `TY_NO_IMPLIED_FP_CACHE=1` (any value) forces the pre-cache behavior:
// derived operators are re-evaluated for every transition.
feature_flag!(pub(crate) no_implied_fp_cache, "TY_NO_IMPLIED_FP_CACHE");

/// Build a transition-partition key from the evaluation context plus the
/// fingerprint of the state array the evaluation reads (parent fp in
/// `Current` mode, successor fp in `Next` mode).
///
/// Scope discrimination (`local_ops_id`, `instance_subs_id`) matches
/// [`zero_arg_cache_key`] so operators resolved through different INSTANCE
/// contexts can never share entries.
#[inline]
pub(crate) fn zero_arg_transition_key(
    ctx: &EvalCtx,
    name_id: NameId,
    def_loc: u32,
    state_fp: u64,
) -> ZeroArgTransitionCacheKey {
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
    (
        ctx.shared.id,
        local_ops_id,
        instance_subs_id,
        name_id,
        def_loc,
        state_fp,
    )
}

/// Look up a transition-partition entry. Returns a clone of the entry when
/// present AND `validator` accepts it; the validator MUST re-check every
/// recorded dep value against the currently bound arrays (fingerprint
/// collisions degrade to misses, never wrong values).
#[inline]
pub(crate) fn zero_arg_transition_lookup(
    key: &ZeroArgTransitionCacheKey,
    validator: impl Fn(&CachedOpResult) -> bool,
) -> Option<CachedOpResult> {
    ZERO_ARG_CACHES.with(|caches| {
        let caches = caches.borrow();
        let entry = caches.transition.get(key)?;
        if validator(entry) {
            Some(entry.clone())
        } else {
            None
        }
    })
}

/// Like [`zero_arg_transition_lookup`], but hands the validator a MUTABLE
/// entry so it can refresh validated-equal dep snapshots in place (see
/// `transition_entry_valid_refresh`). Semantics are otherwise identical:
/// the entry is returned (cloned) only when the validator accepts it.
#[inline]
pub(crate) fn zero_arg_transition_lookup_refresh(
    key: &ZeroArgTransitionCacheKey,
    validator: impl FnOnce(&mut CachedOpResult) -> bool,
) -> Option<CachedOpResult> {
    ZERO_ARG_CACHES.with(|caches| {
        let mut caches = caches.borrow_mut();
        let entry = caches.transition.get_mut(key)?;
        if validator(entry) {
            Some(entry.clone())
        } else {
            None
        }
    })
}

/// Insert a transition-partition entry (replacing any previous entry for the
/// key). The partition enforces its own bound (two-generation rotation, or
/// the legacy wholesale-clear cap under `TY_IMPLIED_FP_CACHE_CAP=0`).
#[inline]
pub(crate) fn zero_arg_transition_insert(key: ZeroArgTransitionCacheKey, entry: CachedOpResult) {
    ZERO_ARG_CACHES.with(|caches| {
        let mut caches = caches.borrow_mut();
        caches.transition.insert(key, entry);
    });
}

/// Number of stored transition-partition entries (observability: used by unit
/// tests and by tla-check integration tests via
/// [`crate::implied_transition_cache_len`]).
pub(crate) fn zero_arg_transition_cache_entry_count() -> usize {
    ZERO_ARG_CACHES.with(|caches| caches.borrow().transition.len())
}

/// Build a scope-normalized "canonical" cache key for constant operators.
///
/// When `local_ops` contains recursive operators, `compute_local_ops_scope_id`
/// falls back to `Arc` pointer identity, producing a different scope id every
/// time `with_outer_resolution_scope()` is called. This prevents persistent
/// (constant) cache entries from being found on subsequent lookups.
///
/// The canonical key sets `local_ops_id = 0` and `instance_subs_id = 0`,
/// making it scope-independent. This is safe ONLY for persistent entries
/// (empty deps) because constant operators produce the same value regardless
/// of the local_ops or instance_substitutions scope.
#[inline]
pub(crate) fn zero_arg_canonical_key(
    shared_id: u64,
    name_id: NameId,
    def_loc: u32,
    is_next_state: bool,
) -> ZeroArgCacheKey {
    (shared_id, 0, 0, name_id, def_loc, is_next_state)
}

/// Lookup a constant operator using the scope-normalized canonical key.
///
/// Only probes the persistent partition and only accepts entries with
/// persistent deps. This is the fallback path when the scope-discriminated
/// lookup fails due to unstable `local_ops_id` from recursive operator
/// environments (e.g., `RECURSIVE PublicKeyOf(_)` in INSTANCE Nano).
#[inline]
pub(crate) fn zero_arg_canonical_lookup(canonical_key: &ZeroArgCacheKey) -> Option<CachedOpResult> {
    ZERO_ARG_CACHES.with(|caches| {
        let caches = caches.borrow();
        let entries = caches.persistent.get(canonical_key)?;
        for entry in entries {
            if deps_are_persistent(&entry.deps) {
                return Some(entry.clone());
            }
        }
        None
    })
}

/// Insert a constant operator result under the canonical key.
///
/// Only call this for entries with persistent deps (caller must verify).
/// Routes directly to the persistent partition.
#[inline]
pub(crate) fn zero_arg_canonical_insert(key: ZeroArgCacheKey, entry: CachedOpResult) {
    debug_assert!(
        deps_are_persistent(&entry.deps),
        "canonical insert must only be used for persistent entries"
    );
    ZERO_ARG_CACHES.with(|caches| {
        let mut caches = caches.borrow_mut();
        let entries = caches.persistent.entry(key).or_default();
        // Canonical keys have at most 1 entry per key (constant value is unique).
        if entries.is_empty() {
            entries.push(entry);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deps_are_persistent_rejects_instance_lazy_read() {
        let mut deps = OpEvalDeps::default();
        assert!(
            deps_are_persistent(&deps),
            "empty deps should be persistent"
        );

        deps.instance_lazy_read = true;
        assert!(
            !deps_are_persistent(&deps),
            "deps with instance_lazy_read taint must NOT be persistent"
        );
    }

    #[test]
    fn test_deps_are_persistent_rejects_inconsistent() {
        let deps = OpEvalDeps {
            inconsistent: true,
            ..Default::default()
        };
        assert!(!deps_are_persistent(&deps));
    }

    #[test]
    fn test_deps_are_persistent_accepts_clean_empty() {
        let deps = OpEvalDeps::default();
        assert!(deps_are_persistent(&deps));
    }
}
