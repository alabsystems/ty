// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Thread-local ENABLED evaluation cache for liveness checking.
//!
//! Extracted from `checker/mod.rs`. Provides caching of `ENABLED(action)(state)`
//! booleans to avoid redundant evaluation during consistency and SCC checking.
//! No dependency on `LivenessChecker` — clean extraction boundary.

use super::cache_stats::{record_enabled_eviction, record_enabled_hit, record_enabled_miss};
use crate::error::EvalResult;
use crate::eval::EvalCtx;
use crate::state::Fingerprint;
use rustc_hash::FxHashMap;
use std::cell::{Cell, RefCell};

// Thread-local cache for ENABLED evaluation results.
//
// TLC pre-computes ENABLED(action)(state) booleans once per state during BFS
// graph construction and stores them in GraphNode BitVectors. TY previously
// re-evaluated ENABLED for every (state, tableau_node) pair during consistency
// checking, causing O(states × tableau_nodes × enumeration_cost) work instead
// of O(states × unique_enabled_tags × enumeration_cost).
//
// This cache stores `(state_fingerprint, enabled_tag) -> bool` and is shared
// across both consistency checking (explore_bfs) and SCC constraint checking.
// The `tag` field on `LiveExpr::Enabled` uniquely identifies each expanded
// ENABLED expression (including bindings from quantified fairness).
//
// Clear this cache at the start of each `check_liveness_property` call.
//
// Soft cap: 5M entries with retain-half eviction (#4083). Without a cap,
// this cache grows linearly with the number of states visited during liveness
// checking, which can reach millions for large specs.
//
// The previous 200K cap caused severe thrashing on specs with many fairness
// tags. AllocatorImplementation has 122 ENABLED checks × 17,701 states =
// 2.16M entries; at the old cap, 2M evictions occurred. The cache must hold
// at least (states × enabled_tags) entries to avoid re-evaluation.
const ENABLED_CACHE_SOFT_CAP: usize = 5_000_000;

/// Soft cap for the bitmap representation, counted in STATES (map entries),
/// not (state, tag) pairs. One entry holds every tag for its state, so
/// 2M state entries strictly dominates the legacy 5M pair cap for any spec
/// with ≥3 enabled tags while bounding the map to ~2M × ~64B ≈ 128 MB
/// worst-case (inline masks; tag ≥ 128 spills per entry).
const ENABLED_CACHE_STATE_SOFT_CAP: usize = 2_000_000;

/// Per-state known/value tag bitmasks (2026-07 memory audit).
///
/// Replaces one hashbrown entry per `(state fp, tag)` pair (~24 B each; 2.16M
/// entries ≈ 100 MB on AllocatorImplementation) with one entry per STATE
/// holding two tag-indexed bitmask words (inline up to 128 tags). Purely a
/// representation change: `get` returns exactly what the pair map returned
/// (`Some(value)` iff the pair was inserted), `set` records exactly the same
/// (fp, tag, bool) triples. Kill switch: `TY_ENABLED_CACHE_LEGACY=1` restores
/// the pair-keyed map.
#[derive(Default)]
pub(crate) struct EnabledBits {
    /// Bit `tag` set = this (state, tag) pair has a recorded result.
    known: smallvec::SmallVec<[u64; 2]>,
    /// Bit `tag` = the recorded ENABLED result (meaningful only when known).
    value: smallvec::SmallVec<[u64; 2]>,
}

impl EnabledBits {
    #[inline]
    fn get(&self, tag: u32) -> Option<bool> {
        let word = (tag / 64) as usize;
        let bit = 1u64 << (tag % 64);
        if self.known.get(word).copied().unwrap_or(0) & bit != 0 {
            Some(self.value.get(word).copied().unwrap_or(0) & bit != 0)
        } else {
            None
        }
    }

    #[inline]
    fn set(&mut self, tag: u32, result: bool) {
        let word = (tag / 64) as usize;
        let bit = 1u64 << (tag % 64);
        if word >= self.known.len() {
            self.known.resize(word + 1, 0);
            self.value.resize(word + 1, 0);
        }
        self.known[word] |= bit;
        if result {
            self.value[word] |= bit;
        } else {
            self.value[word] &= !bit;
        }
    }

    /// Number of recorded (state, tag) pairs in this entry.
    fn known_count(&self) -> usize {
        self.known.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// Storage backend: bitmap (default) or the legacy pair-keyed map.
///
/// The bitmap stores ONE hashmap entry per STATE (holding every tag's bit
/// inline), the legacy map stores one entry per (state, tag) PAIR. The bitmap
/// wins decisively on the many-enabled-tag specs that dominate the memory
/// axis — AllocatorImplementation (122 WF leaves) drops from 234 MB to 109 MB,
/// flipping it below TLC's 173 MB — at the cost of a small regression on specs
/// with a wide tag space but few tags set per state (EWD998PCal: ~2% / 67 MB
/// on a spec that is a ~5× architectural memory loss to TLC regardless, so no
/// verdict changes). Bitmap is therefore the default; `TY_ENABLED_CACHE_LEGACY=1`
/// pins the pair map for A/B testing. Both are verdict-identical by
/// construction (`get` returns exactly what the pair map returned; `set`
/// records exactly the same (fp, tag, bool) triples).
pub(crate) enum EnabledCacheImpl {
    Bitmap(FxHashMap<Fingerprint, EnabledBits>),
    Legacy(FxHashMap<(Fingerprint, u32), bool>),
}

impl EnabledCacheImpl {
    fn from_env() -> Self {
        let legacy = std::env::var("TY_ENABLED_CACHE_LEGACY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if legacy {
            Self::Legacy(FxHashMap::default())
        } else {
            Self::Bitmap(FxHashMap::default())
        }
    }

    #[inline]
    fn get(&self, state_fp: Fingerprint, tag: u32) -> Option<bool> {
        match self {
            Self::Bitmap(map) => map.get(&state_fp).and_then(|bits| bits.get(tag)),
            Self::Legacy(map) => map.get(&(state_fp, tag)).copied(),
        }
    }

    #[inline]
    fn insert(&mut self, state_fp: Fingerprint, tag: u32, result: bool) {
        match self {
            Self::Bitmap(map) => map.entry(state_fp).or_default().set(tag, result),
            Self::Legacy(map) => {
                map.insert((state_fp, tag), result);
            }
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Bitmap(map) => map.clear(),
            Self::Legacy(map) => map.clear(),
        }
    }

    /// Drop all backing allocations while preserving the selected backend.
    fn release_storage(&mut self) {
        match self {
            Self::Bitmap(map) => *map = FxHashMap::default(),
            Self::Legacy(map) => *map = FxHashMap::default(),
        }
    }

    /// Recorded (state, tag) pair count (census / diagnostics).
    fn pair_count(&self) -> usize {
        match self {
            Self::Bitmap(map) => map.values().map(EnabledBits::known_count).sum(),
            Self::Legacy(map) => map.len(),
        }
    }
}

thread_local! {
    pub(crate) static ENABLED_CACHE: RefCell<EnabledCacheImpl> =
        RefCell::new(EnabledCacheImpl::from_env());
    /// Track whether we have already emitted the first-eviction warning.
    static ENABLED_EVICTION_WARNED: Cell<bool> = const { Cell::new(false) };
}

/// Census probe (TY_MEM_CENSUS): current recorded pair count.
pub(crate) fn census_enabled_cache_len() -> usize {
    ENABLED_CACHE.with(|c| c.borrow().pair_count())
}

/// Clear the thread-local ENABLED cache.
///
/// Must be called at the start of each liveness property check to avoid
/// stale results from a previous property's formula (different tag space).
pub(crate) fn clear_enabled_cache() {
    ENABLED_CACHE.with(|c| c.borrow_mut().clear());
    ENABLED_EVICTION_WARNED.with(|warned| warned.set(false));
}

/// Drop the thread-local ENABLED cache's backing allocation.
///
/// Ordinary lifecycle clears intentionally retain capacity for reuse. The
/// mid-BFS hybrid trip calls this stronger operation because the BFS inline
/// producer is disabled for the remaining exploration. The post-BFS checker
/// may populate a fresh cache, but retaining millions of stale entries here
/// would defeat the trip's release.
pub(crate) fn release_enabled_cache_storage() {
    ENABLED_CACHE.with(|c| c.borrow_mut().release_storage());
    ENABLED_EVICTION_WARNED.with(|warned| warned.set(false));
}

/// Whether an exact-raw streaming prefill can retain all projected facts until
/// its later completeness pass without triggering retain-half eviction.
///
/// Exact-raw callers operate on the current property's raw state domain, so an
/// existing bitmap entry is one of `max_state_count` states. The legacy backend
/// stores one entry per pair and therefore uses the conservative upper bound of
/// every projected tag being new for every state.
pub(crate) fn can_retain_exact_raw_enabled_prefill(
    max_state_count: usize,
    projected_tag_count: usize,
) -> bool {
    ENABLED_CACHE.with(|cache| match &*cache.borrow() {
        EnabledCacheImpl::Bitmap(map) => {
            map.len().max(max_state_count) <= ENABLED_CACHE_STATE_SOFT_CAP
        }
        EnabledCacheImpl::Legacy(map) => {
            map.len()
                .saturating_add(max_state_count.saturating_mul(projected_tag_count))
                <= ENABLED_CACHE_SOFT_CAP
        }
    })
}

/// Trim ENABLED_CACHE if it exceeds the soft cap (#4083).
/// Uses retain-half eviction: keeps ~half the entries using FxHashMap's
/// pseudo-random iteration order, same pattern as eval-layer caches
/// (see `crates/tla-eval/src/cache/lifecycle_trim.rs`).
fn trim_enabled_cache_if_needed() {
    ENABLED_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let (len, cap) = match &*cache {
            EnabledCacheImpl::Bitmap(map) => (map.len(), ENABLED_CACHE_STATE_SOFT_CAP),
            EnabledCacheImpl::Legacy(map) => (map.len(), ENABLED_CACHE_SOFT_CAP),
        };
        if len > cap {
            // Log a warning on first eviction for monitoring (#4083).
            ENABLED_EVICTION_WARNED.with(|warned| {
                if !warned.get() {
                    eprintln!(
                        "[liveness] ENABLED_CACHE exceeded soft cap ({len} > {cap}), evicting"
                    );
                    warned.set(true);
                }
            });
            let target = cap / 2;
            let mut kept = 0;
            let evicted = match &mut *cache {
                EnabledCacheImpl::Bitmap(map) => {
                    map.retain(|_, _| {
                        if kept < target {
                            kept += 1;
                            true
                        } else {
                            false
                        }
                    });
                    len.saturating_sub(map.len())
                }
                EnabledCacheImpl::Legacy(map) => {
                    map.retain(|_, _| {
                        if kept < target {
                            kept += 1;
                            true
                        } else {
                            false
                        }
                    });
                    len.saturating_sub(map.len())
                }
            };
            record_enabled_eviction(evicted as u64);
        }
    });
}

fn get_enabled_cached_internal(state_fp: Fingerprint, tag: u32) -> Option<bool> {
    ENABLED_CACHE.with(|c| {
        let result = c.borrow().get(state_fp, tag);
        if result.is_some() {
            record_enabled_hit();
        } else {
            record_enabled_miss();
        }
        result
    })
}

/// Read a previously computed ENABLED value without evaluating a fallback.
/// Used by exact-raw mask reconstruction only after the caller has completed
/// the authoritative per-state ENABLED phase.
pub(crate) fn get_enabled_cached(state_fp: Fingerprint, tag: u32) -> Option<bool> {
    get_enabled_cached_internal(state_fp, tag)
}

/// Evaluate ENABLED with shared thread-local caching.
///
/// Caching is enabled only when `VarRegistry` is populated (production path).
/// In empty-registry fallback mode, ENABLED can depend on checker-local successor
/// maps, so `(state_fingerprint, tag)` is not a stable cache key across checker
/// instances.
///
/// Part of #2998: Enters ENABLED evaluation scope via `enter_enabled_scope()`,
/// which clears PerState evaluator caches at scope boundaries. This matches TLC's
/// `EvalControl.Enabled` bitflag that disables LazyValue caching within ENABLED
/// (EvalControl.java:22-25, Tool.java:1949-1953). The scope guard is a no-op
/// when already inside an ENABLED scope (nested calls from array path).
pub(crate) fn eval_enabled_cached<F>(
    ctx: &EvalCtx,
    state_fp: Fingerprint,
    tag: u32,
    eval_uncached: F,
) -> EvalResult<bool>
where
    F: FnOnce() -> EvalResult<bool>,
{
    let use_enabled_cache = !ctx.var_registry().is_empty();
    if use_enabled_cache {
        if let Some(result) = get_enabled_cached_internal(state_fp, tag) {
            return Ok(result);
        }
    }

    // Part of #2998: Enter ENABLED scope — clears PerState caches on first entry
    // to prevent non-ENABLED cache entries from contaminating ENABLED evaluation.
    // Returns None (no-op) if already in scope (e.g., called from within
    // ea_precompute's array path which enters the scope for the entire Phase A).
    // Part of #3962: Use ctx-aware variant to sync in_enabled_scope shadow.
    let _enabled_guard = crate::eval::enter_enabled_scope_with_ctx(ctx);

    let result = eval_uncached()?;

    if use_enabled_cache {
        trim_enabled_cache_if_needed();
        ENABLED_CACHE.with(|c| c.borrow_mut().insert(state_fp, tag, result));
    }

    Ok(result)
}

/// Mutable variant of [`eval_enabled_cached`] for array-native callers that
/// need to update `next_state` / `next_state_env` while computing ENABLED.
pub(crate) fn eval_enabled_cached_mut<F>(
    ctx: &mut EvalCtx,
    state_fp: Fingerprint,
    tag: u32,
    eval_uncached: F,
) -> EvalResult<bool>
where
    F: FnOnce(&mut EvalCtx) -> EvalResult<bool>,
{
    let use_enabled_cache = !ctx.var_registry().is_empty();
    if use_enabled_cache {
        if let Some(result) = get_enabled_cached_internal(state_fp, tag) {
            return Ok(result);
        }
    }

    // Part of #3962: Use ctx-aware variant to sync in_enabled_scope shadow.
    let _enabled_guard = crate::eval::enter_enabled_scope_with_ctx(ctx);
    let result = eval_uncached(ctx)?;

    if use_enabled_cache {
        trim_enabled_cache_if_needed();
        ENABLED_CACHE.with(|c| c.borrow_mut().insert(state_fp, tag, result));
    }

    Ok(result)
}

/// Check if an ENABLED result is already in the thread-local cache.
pub(crate) fn is_enabled_cached(state_fp: Fingerprint, tag: u32) -> bool {
    get_enabled_cached_internal(state_fp, tag).is_some()
}

/// Insert an ENABLED result into the thread-local cache.
/// Trims the cache via retain-half eviction if it exceeds the soft cap (#4083).
pub(crate) fn set_enabled_cache(state_fp: Fingerprint, tag: u32, result: bool) {
    trim_enabled_cache_if_needed();
    ENABLED_CACHE.with(|c| c.borrow_mut().insert(state_fp, tag, result));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retained_capacity() -> usize {
        ENABLED_CACHE.with(|cache| match &*cache.borrow() {
            EnabledCacheImpl::Bitmap(map) => map.capacity(),
            EnabledCacheImpl::Legacy(map) => map.capacity(),
        })
    }

    #[test]
    fn release_enabled_cache_storage_drops_capacity() {
        release_enabled_cache_storage();
        for n in 0..128 {
            set_enabled_cache(Fingerprint(n), 1, true);
        }
        assert_eq!(census_enabled_cache_len(), 128);
        assert!(retained_capacity() > 0);

        release_enabled_cache_storage();

        assert_eq!(census_enabled_cache_len(), 0);
        assert_eq!(retained_capacity(), 0);
    }
}
