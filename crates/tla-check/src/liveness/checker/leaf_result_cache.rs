// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Thread-local result cache for liveness `ActionPred` leaves.
//!
//! ## Why
//!
//! The inline-BFS liveness recorder and the EA-precompute leaf batch
//! re-evaluate every `ActionPred` leaf of the temporal property through the
//! AST/TIR interpreter for the transitions of each completed state. On specs
//! whose fairness/temporal action leaf is re-encountered for the same
//! `(current, next)` transition across exploration phases (e.g. the
//! `<<Next>>_vars` action in `ewd426/TokenRing`'s `WF_vars(Next)` /
//! `<>[]UniqueToken`), this is redundant work: the result is identical each
//! time. Profiling shows leaf re-eval dominates CPU on liveness specs and is the
//! single biggest TY-vs-TLC loss driver.
//!
//! ## What
//!
//! An `ActionPred` leaf is, by construction (its `ExprLevel` is `Action`), a
//! deterministic function of the (current, next) state PAIR alone. Its
//! quantifier `bindings` (a `BindingChain`) are baked into the `LiveExpr` at
//! conversion time and are fixed per `tag`; the spec's constants are fixed for
//! the run. Therefore the result is fully determined by
//! `(current_fp, next_fp, tag)`, and this cache stores exactly that mapping.
//!
//! `StatePred` leaves are intentionally NOT cached: the inline state-leaf
//! bitmask already dedups state-predicate evaluation per source fingerprint, so
//! a `(current_fp, tag)` result cache measured zero hits across the corpus and
//! was pure overhead.
//!
//! ## Adaptivity (avoid bloat on large unique-transition specs)
//!
//! The action bitmask already dedups exact transitions in the steady inline
//! path, so on specs with a large, mostly-unique transition space (e.g.
//! AllocatorImplementation: ~10M distinct evaluated transitions) this cache
//! would grow without hitting — pure memory + insert overhead. To stay net
//! positive everywhere, the cache self-disables: after a warmup window of
//! lookups, if the observed hit rate is below a threshold it is cleared and
//! turned off for the remainder of the property check. Specs with genuine
//! cross-phase / re-exploration redundancy (TokenRing: ~45% hit rate) keep the
//! cache; specs without it pay only the bounded warmup cost.
//!
//! ## Soundness
//!
//! - **Key completeness.** `current_fp` / `next_fp` are the collision-resistant
//!   64-bit fingerprints of the concrete states whose envs are bound for the
//!   evaluation; `tag` uniquely identifies the `(expr, bindings)` pair (each leaf
//!   gets a fresh tag at conversion time carrying its fixed quantifier bindings).
//!   An `ActionPred` reads only current+next state, so no further key components
//!   are required. Keys are fully deterministic — no pointer identity, no
//!   HashMap-iteration order — so results are reproducible across runs. This is
//!   the same `(Fingerprint, …, tag)` key shape, in the same code paths, with the
//!   same lifecycle, as the established `ENABLED_CACHE` / `SUBSCRIPT_VALUE_CACHE`.
//!
//! - **Caching only in production (non-empty registry).** Results are cached only
//!   when the `VarRegistry` is populated; the empty-registry fallback never
//!   reaches these helpers (the caller takes the `to_state` fallback first).
//!
//! - **Lifecycle.** Cleared at exactly the same points as the ENABLED / subscript
//!   caches (run reset, property-check start), so a `(fp, tag)` key from a
//!   previous spec / property — with a different tag space — can never be read.
//!
//! - **No interaction with the fairness soundness gate.** This cache is wired
//!   ONLY into the inline-BFS recording / EA-precompute leaf batch path. The
//!   authoritative witness re-verification (`witness_cycle_satisfies_pem`) and the
//!   consistency / SCC constraint evaluation go through `eval_live_expr_core`,
//!   which is left uncached, so the gate's deliberate per-edge cache resets are
//!   unaffected.

use super::cache_stats::{record_leaf_eviction, record_leaf_hit, record_leaf_miss, LeafCacheKind};
use crate::error::EvalResult;
use crate::state::Fingerprint;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::{Cell, RefCell};

// Soft cap mirrors ENABLED_CACHE (#4083): retain-half eviction at the cap. With
// the adaptive disable below, a poorly-hitting cache is turned off long before it
// reaches this cap, so the cap is only a safety bound for genuinely-hitting,
// huge-but-redundant transition spaces.
const ACTION_PRED_CACHE_SOFT_CAP: usize = 2_000_000;

// Adaptive self-disable: after this many lookups (hits + misses) in a property
// check, if the hit rate is below `MIN_HIT_RATE_PCT` percent, disable and clear
// the cache for the rest of the check. The warmup is large enough to amortize on
// genuinely-redundant specs but small relative to multi-million-transition specs,
// so the wasted warmup inserts are negligible.
const ADAPTIVE_WARMUP_LOOKUPS: u64 = 200_000;
const MIN_HIT_RATE_PCT: u64 = 10;

thread_local! {
    static ACTION_PRED_CACHE: RefCell<FxHashMap<(Fingerprint, Fingerprint, u32), bool>> =
        RefCell::new(FxHashMap::default());
    static ACTION_PRED_EVICTION_WARNED: Cell<bool> = const { Cell::new(false) };
    /// Adaptive state: once `true`, caching is off for the rest of this property
    /// check (reset by `clear_leaf_result_cache`).
    static ACTION_PRED_DISABLED: Cell<bool> = const { Cell::new(false) };
    static ACTION_PRED_LOOKUPS: Cell<u64> = const { Cell::new(0) };
    static ACTION_PRED_LOOKUP_HITS: Cell<u64> = const { Cell::new(0) };
    /// Part of #liveness-leaf-memo: ENABLED leaf tag → paired `ActionPred` tag
    /// from the same WF/SF expansion (see `build_fairness_expansion`: both
    /// leaves are clones of one `resolved_action` with the same quantifier
    /// bindings). The inline ENABLED successor scan evaluates exactly that
    /// predicate per (current, successor) pair, so its results are shared with
    /// `record_missing_action_results`, which then skips the duplicate AST
    /// evaluation.
    static ENABLED_ACTION_PRED_PAIR: RefCell<FxHashMap<u32, u32>> =
        RefCell::new(FxHashMap::default());
    /// Part of #liveness-enabled-enum-first: per-paired-tag streak of
    /// consecutive ENABLED=true outcomes. A leaf whose action is enabled in
    /// every state it is asked about (e.g. `WF_vars(Next)` on a deadlock-free
    /// spec) is decided fastest by the legacy explored-successor scan (first
    /// probe witnesses); a leaf that is frequently DISABLED is decided
    /// fastest by enumeration-first (the guard refutes immediately, and the
    /// complete empty set feeds the predicate-sharing scratchpad). The streak
    /// picks between the two EXACT decision procedures per tag — it can only
    /// affect performance, never a result. Cleared with the pair map at the
    /// same lifecycle points.
    static ENABLED_TRUE_STREAK: RefCell<FxHashMap<u32, u32>> =
        RefCell::new(FxHashMap::default());
    /// Part of #liveness-enabled-enum-first: paired `ActionPred` tags whose
    /// resolved action STATICALLY PINS every state variable's primed value on
    /// every satisfying branch (see `action_pins_all_vars`). Only these tags
    /// may receive enumeration-derived FALSE predicate entries — for an
    /// under-specified action (free primes), the enumeration's
    /// UNCHANGED-completion can miss predicate-true transitions (and the
    /// degenerate no-primed-assignment case enumerates to the EMPTY set
    /// despite a total relation), so non-membership must not be recorded as
    /// `false`. Cleared with the pair map.
    static FULL_POPULATION_TAGS: RefCell<FxHashSet<u32>> =
        RefCell::new(FxHashSet::default());
    /// Part of #liveness-enum-exact: paired tags whose ENABLED verdict is
    /// decided exactly by the action's own enumeration (subscript-support
    /// pinning — weaker than the all-vars proof above, NOT sufficient for
    /// FALSE predicate population). Cleared with the pair map.
    static ENUM_EXACT_TAGS: RefCell<FxHashSet<u32>> =
        RefCell::new(FxHashSet::default());
    /// Tags of `ENABLED(<<Next>>_vars)` leaves over the whole next-state
    /// relation. Such an ENABLED is decided by scanning the (complete) BFS
    /// successor set for a subscript change instead of a from-scratch Next
    /// re-enumeration. Registered at fairness-plan build; cleared with the rest.
    static WHOLE_NEXT_ENABLED_TAGS: RefCell<FxHashSet<u32>> =
        RefCell::new(FxHashSet::default());
    /// Tags of the `ActionPred(Next)` leaf of a `<<Next>>_vars` fairness action
    /// over the whole next-state relation (the ActionPred PAIRED with a
    /// whole-Next ENABLED leaf; see `whole_next_enabled_tag`). Every BFS
    /// successor edge `(s, t)` is produced BY Next enumeration, so
    /// `Next(s, t)` is TRUE for every real successor — the leaf value the
    /// inline recorder would otherwise re-derive by re-running the nested Next
    /// existential per successor. Such tags are set directly to TRUE per real
    /// successor and skipped from the per-transition leaf batch. Registered at
    /// fairness-plan build (gated on `action_pins_all_vars`); cleared with the
    /// rest.
    static WHOLE_NEXT_ACTION_TAGS: RefCell<FxHashSet<u32>> =
        RefCell::new(FxHashSet::default());
    /// Part of #liveness-leaf-memo: per-state scratchpad of ENABLED-scan
    /// predicate results, keyed `(cur_fp, succ_fp, action_pred_tag)`.
    ///
    /// Kept SEPARATE from `ACTION_PRED_CACHE`: the scan probes every
    /// (leaf × successor) pair — mostly for actions whose tags the
    /// ENABLED-skip bitmask later excludes from recording — so routing them
    /// through the soft-capped global cache floods it (measured: 9M evictions
    /// on AllocatorImplementation). The reuse window is exactly one state
    /// completion (scan now, record moments later on the same thread), so a
    /// small map cleared at each state boundary captures every hit with
    /// bounded memory (≤ leaves × successors entries). Keys carry full
    /// fingerprints, so even a missed clear cannot alias entries across
    /// states (same fp64-trust argument as the global cache).
    static SCAN_PRED_RESULTS: RefCell<FxHashMap<(Fingerprint, Fingerprint, u32), bool>> =
        RefCell::new(FxHashMap::default());
    /// Part of #liveness-enabled-witness-exit: per-paired-tag memo of the
    /// STATIC subscript watch analysis (`subscript_watch_vars`): `Some(vars)`
    /// when the fairness subscript is a (nested) tuple of exactly those state
    /// variables, `None` when the analysis failed (opaque subscript — the
    /// witness early-exit stays off for the leaf). The analysis is a pure
    /// function of the leaf's fixed `(subscript, bindings)` pair, which is
    /// uniquely identified by the paired tag for the whole property check —
    /// same key argument and lifecycle as `ENABLED_ACTION_PRED_PAIR`.
    static SUBSCRIPT_WATCH_VARS: RefCell<FxHashMap<u32, Option<WatchVarSet>>> =
        RefCell::new(FxHashMap::default());
}

/// Watched state-variable index set produced by the static fairness-subscript
/// analysis (#liveness-enabled-witness-exit).
pub(crate) type WatchVarSet = smallvec::SmallVec<[crate::var_index::VarIndex; 8]>;

/// Memoized subscript watch analysis for a paired leaf tag
/// (#liveness-enabled-witness-exit). Computes on first use, then serves the
/// per-tag result for the rest of the property check. Cleared by
/// [`clear_leaf_result_cache`] at the same lifecycle points as the pair map.
pub(crate) fn subscript_watch_cached(
    tag: u32,
    compute: impl FnOnce() -> Option<WatchVarSet>,
) -> Option<WatchVarSet> {
    if let Some(cached) = SUBSCRIPT_WATCH_VARS.with(|m| m.borrow().get(&tag).cloned()) {
        return cached;
    }
    let computed = compute();
    SUBSCRIPT_WATCH_VARS.with(|m| {
        m.borrow_mut().insert(tag, computed.clone());
    });
    computed
}

/// Census probe (TY_MEM_CENSUS): current entry count of the action-pred cache.
pub(crate) fn census_action_pred_cache_len() -> usize {
    ACTION_PRED_CACHE.with(|c| c.borrow().len())
}

/// Census probe (TY_MEM_CENSUS): current entry count of the scan scratchpad.
pub(crate) fn census_scan_pred_len() -> usize {
    SCAN_PRED_RESULTS.with(|c| c.borrow().len())
}

/// Clear the per-state ENABLED-scan result scratchpad (#liveness-leaf-memo).
/// Called at each inline-recording state boundary.
pub(crate) fn clear_scan_pred_results() {
    SCAN_PRED_RESULTS.with(|m| m.borrow_mut().clear());
}

/// Record one ENABLED-scan predicate result for the paired ActionPred tag.
pub(crate) fn insert_scan_pred_result(
    cur_fp: Fingerprint,
    succ_fp: Fingerprint,
    tag: u32,
    value: bool,
) {
    SCAN_PRED_RESULTS.with(|m| {
        m.borrow_mut().insert((cur_fp, succ_fp, tag), value);
    });
}

/// Look up an ENABLED-scan predicate result recorded for this state boundary.
pub(crate) fn get_scan_pred_result(
    cur_fp: Fingerprint,
    succ_fp: Fingerprint,
    tag: u32,
) -> Option<bool> {
    SCAN_PRED_RESULTS.with(|m| m.borrow().get(&(cur_fp, succ_fp, tag)).copied())
}

/// Clear the thread-local leaf-result cache and reset the adaptive state.
///
/// Must be called at the same lifecycle points as [`super::clear_enabled_cache`]
/// (run reset, start of each property check) so stale `(fp, fp, tag)` entries
/// from a previous spec / property — with a different tag space — are never read.
/// The ENABLED→ActionPred pair map is dropped at the same points: a stale map
/// applied to a re-assigned tag space would alias unrelated leaves.
pub(crate) fn clear_leaf_result_cache() {
    ACTION_PRED_CACHE.with(|c| c.borrow_mut().clear());
    ACTION_PRED_EVICTION_WARNED.with(|w| w.set(false));
    ACTION_PRED_DISABLED.with(|d| d.set(false));
    ACTION_PRED_LOOKUPS.with(|l| l.set(0));
    ACTION_PRED_LOOKUP_HITS.with(|h| h.set(0));
    ENABLED_ACTION_PRED_PAIR.with(|m| m.borrow_mut().clear());
    ENABLED_TRUE_STREAK.with(|m| m.borrow_mut().clear());
    FULL_POPULATION_TAGS.with(|s| s.borrow_mut().clear());
    ENUM_EXACT_TAGS.with(|s| s.borrow_mut().clear());
    WHOLE_NEXT_ENABLED_TAGS.with(|s| s.borrow_mut().clear());
    WHOLE_NEXT_ACTION_TAGS.with(|s| s.borrow_mut().clear());
    SCAN_PRED_RESULTS.with(|m| m.borrow_mut().clear());
    SUBSCRIPT_WATCH_VARS.with(|m| m.borrow_mut().clear());
}

/// Drop every backing allocation owned by the thread-local leaf caches.
///
/// Normal property-boundary clearing retains capacity for the next property.
/// The mid-BFS regeneration trip permanently disables inline recording for the
/// current run, so it uses this stronger release operation instead.
pub(crate) fn release_leaf_result_cache_storage() {
    ACTION_PRED_CACHE.with(|cache| *cache.borrow_mut() = FxHashMap::default());
    ACTION_PRED_EVICTION_WARNED.with(|warned| warned.set(false));
    ACTION_PRED_DISABLED.with(|disabled| disabled.set(false));
    ACTION_PRED_LOOKUPS.with(|lookups| lookups.set(0));
    ACTION_PRED_LOOKUP_HITS.with(|hits| hits.set(0));
    ENABLED_ACTION_PRED_PAIR.with(|map| *map.borrow_mut() = FxHashMap::default());
    ENABLED_TRUE_STREAK.with(|map| *map.borrow_mut() = FxHashMap::default());
    FULL_POPULATION_TAGS.with(|set| *set.borrow_mut() = FxHashSet::default());
    ENUM_EXACT_TAGS.with(|set| *set.borrow_mut() = FxHashSet::default());
    WHOLE_NEXT_ENABLED_TAGS.with(|set| *set.borrow_mut() = FxHashSet::default());
    WHOLE_NEXT_ACTION_TAGS.with(|set| *set.borrow_mut() = FxHashSet::default());
    SCAN_PRED_RESULTS.with(|map| *map.borrow_mut() = FxHashMap::default());
    SUBSCRIPT_WATCH_VARS.with(|map| *map.borrow_mut() = FxHashMap::default());
}

/// Register the ENABLED-tag → paired-ActionPred-tag map for this run
/// (#liveness-leaf-memo). Called once per run when the inline fairness plan is
/// built; cleared by [`clear_leaf_result_cache`].
pub(crate) fn set_enabled_action_pred_pairs(map: FxHashMap<u32, u32>) {
    ENABLED_ACTION_PRED_PAIR.with(|m| *m.borrow_mut() = map);
    // The pinning proofs travel with the pair map: a replaced map invalidates
    // the previous tag space, so the proof set is reset alongside it (the
    // fairness registration runs before any plan extension). The subscript
    // watch memo is keyed by the same tag space and travels with it too.
    FULL_POPULATION_TAGS.with(|s| s.borrow_mut().clear());
    ENUM_EXACT_TAGS.with(|s| s.borrow_mut().clear());
    WHOLE_NEXT_ENABLED_TAGS.with(|s| s.borrow_mut().clear());
    WHOLE_NEXT_ACTION_TAGS.with(|s| s.borrow_mut().clear());
    SUBSCRIPT_WATCH_VARS.with(|m| m.borrow_mut().clear());
}

/// Extend the ENABLED-tag → paired-ActionPred-tag map with additional pairs
/// (#liveness-enabled-enum-first: property-plan leaves live in the same
/// converter tag space as the fairness leaves, strictly above
/// `max_fairness_tag`, so extension can never alias an existing pair).
/// Cleared together with the base map by [`clear_leaf_result_cache`].
pub(crate) fn extend_enabled_action_pred_pairs(map: FxHashMap<u32, u32>) {
    ENABLED_ACTION_PRED_PAIR.with(|m| m.borrow_mut().extend(map));
}

/// Look up the paired `ActionPred` tag for an ENABLED leaf tag, if registered.
pub(crate) fn enabled_action_pred_pair(enabled_tag: u32) -> Option<u32> {
    ENABLED_ACTION_PRED_PAIR.with(|m| m.borrow().get(&enabled_tag).copied())
}

/// Current consecutive ENABLED=true streak for a paired tag
/// (#liveness-enabled-enum-first).
pub(crate) fn enabled_true_streak(tag: u32) -> u32 {
    ENABLED_TRUE_STREAK.with(|m| m.borrow().get(&tag).copied().unwrap_or(0))
}

/// Record an ENABLED outcome for a paired tag: `true` extends the streak,
/// `false` resets it (#liveness-enabled-enum-first).
pub(crate) fn note_enabled_outcome(tag: u32, enabled: bool) {
    ENABLED_TRUE_STREAK.with(|m| {
        let mut map = m.borrow_mut();
        if enabled {
            *map.entry(tag).or_insert(0) += 1;
        } else {
            map.remove(&tag);
        }
    });
}

/// Reset a paired tag's streak (#liveness-enabled-enum-first): called when the
/// scan-first probe failed to witness, so the tag flips back to
/// enumeration-first instead of paying the probe cost every state.
pub(crate) fn reset_enabled_streak(tag: u32) {
    ENABLED_TRUE_STREAK.with(|m| {
        m.borrow_mut().remove(&tag);
    });
}

/// Add paired tags proven safe for full (true AND false) enumeration-derived
/// predicate population (#liveness-enabled-enum-first).
pub(crate) fn extend_full_population_tags(tags: impl IntoIterator<Item = u32>) {
    FULL_POPULATION_TAGS.with(|s| s.borrow_mut().extend(tags));
}

/// Whether a paired tag may receive enumeration-derived FALSE predicate
/// entries (#liveness-enabled-enum-first).
pub(crate) fn full_population_tag(tag: u32) -> bool {
    FULL_POPULATION_TAGS.with(|s| s.borrow().contains(&tag))
}

/// Register paired tags whose ENABLED verdict is decided EXACTLY by the
/// action's own enumeration — subscript-support pinning, a strictly weaker
/// proof than [`extend_full_population_tags`]'s all-vars pinning
/// (#liveness-enum-exact; see `enabled_enum_decides_exactly`). These tags skip
/// the under-specification rescue scan on a complete-enumeration FALSE, but
/// are NOT authorized for enumeration-derived FALSE predicate entries (a BFS
/// transition may differ from every enumerated successor only OUTSIDE the
/// pinned set and still satisfy the action).
pub(crate) fn extend_enum_exact_tags(tags: impl IntoIterator<Item = u32>) {
    ENUM_EXACT_TAGS.with(|s| s.borrow_mut().extend(tags));
}

/// Whether a paired tag's ENABLED verdict is enumeration-exact
/// (#liveness-enum-exact).
pub(crate) fn enum_exact_tag(tag: u32) -> bool {
    ENUM_EXACT_TAGS.with(|s| s.borrow().contains(&tag))
}

/// Register ENABLED tags whose action is the whole next-state relation
/// (`WF_vars(Next)` / `SF_vars(Next)`), decidable by a successor-set scan.
pub(crate) fn extend_whole_next_enabled_tags(tags: impl IntoIterator<Item = u32>) {
    WHOLE_NEXT_ENABLED_TAGS.with(|s| s.borrow_mut().extend(tags));
}

/// Whether `tag` is an `ENABLED(<<Next>>_vars)` leaf over the whole next-state
/// relation (see [`extend_whole_next_enabled_tags`]).
pub(crate) fn whole_next_enabled_tag(tag: u32) -> bool {
    WHOLE_NEXT_ENABLED_TAGS.with(|s| s.borrow().contains(&tag))
}

/// Kill switch (`TY_DISABLE_WHOLE_NEXT_ACTION_TAGS=1`): force the whole-Next
/// `ActionPred(Next)` leaf back through the per-successor evaluator instead of
/// the direct-TRUE fast path. Used by the differential harness to prove the
/// fast path is verdict- and mask-identical to full evaluation.
fn whole_next_action_tags_disabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TY_DISABLE_WHOLE_NEXT_ACTION_TAGS").is_ok_and(|v| v == "1"))
}

/// Register `ActionPred(Next)` tags of `<<Next>>_vars` fairness actions over the
/// whole next-state relation (paired with a whole-Next ENABLED leaf and proven
/// to pin every state variable). Their value is TRUE on every real successor
/// edge, so the inline recorder sets them directly instead of re-enumerating
/// Next per transition. No-op under the kill switch.
pub(crate) fn extend_whole_next_action_tags(tags: impl IntoIterator<Item = u32>) {
    if whole_next_action_tags_disabled() {
        return;
    }
    WHOLE_NEXT_ACTION_TAGS.with(|s| s.borrow_mut().extend(tags));
}

/// Whether `tag` is the `ActionPred(Next)` leaf of a whole-Next `<<Next>>_vars`
/// fairness action (see [`extend_whole_next_action_tags`]).
pub(crate) fn whole_next_action_tag(tag: u32) -> bool {
    WHOLE_NEXT_ACTION_TAGS.with(|s| s.borrow().contains(&tag))
}

fn trim_action_pred_cache_if_needed() {
    ACTION_PRED_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let len = cache.len();
        if len > ACTION_PRED_CACHE_SOFT_CAP {
            ACTION_PRED_EVICTION_WARNED.with(|warned| {
                if !warned.get() {
                    eprintln!(
                        "[liveness] ACTION_PRED_CACHE exceeded soft cap ({} > {}), evicting",
                        len, ACTION_PRED_CACHE_SOFT_CAP
                    );
                    warned.set(true);
                }
            });
            let target = ACTION_PRED_CACHE_SOFT_CAP / 2;
            let mut kept = 0;
            cache.retain(|_, _| {
                if kept < target {
                    kept += 1;
                    true
                } else {
                    false
                }
            });
            record_leaf_eviction(
                LeafCacheKind::Action,
                len.saturating_sub(cache.len()) as u64,
            );
        }
    });
}

/// Record one lookup outcome and, at the end of the warmup window, decide whether
/// to disable the cache for the remainder of the property check.
#[inline]
fn note_lookup(hit: bool) {
    let lookups = ACTION_PRED_LOOKUPS.with(|l| {
        let n = l.get() + 1;
        l.set(n);
        n
    });
    if hit {
        ACTION_PRED_LOOKUP_HITS.with(|h| h.set(h.get() + 1));
    }
    if lookups == ADAPTIVE_WARMUP_LOOKUPS {
        let hits = ACTION_PRED_LOOKUP_HITS.with(Cell::get);
        // hits * 100 / lookups < MIN_HIT_RATE_PCT  ⇒  disable.
        if hits.saturating_mul(100) < lookups.saturating_mul(MIN_HIT_RATE_PCT) {
            ACTION_PRED_DISABLED.with(|d| d.set(true));
            ACTION_PRED_CACHE.with(|c| c.borrow_mut().clear());
        }
    }
}

/// Evaluate an `ActionPred` leaf with adaptive thread-local result caching keyed
/// on `(current_fp, next_fp, tag)`.
///
/// Caching is only performed when `use_cache` is true (production array-native
/// path with a populated registry) AND the adaptive disable has not engaged. The
/// closure `eval_uncached` performs the authoritative AST/TIR evaluation on a
/// cache miss.
#[inline]
pub(crate) fn eval_action_pred_cached<F>(
    use_cache: bool,
    current_fp: Fingerprint,
    next_fp: Fingerprint,
    tag: u32,
    eval_uncached: F,
) -> EvalResult<bool>
where
    F: FnOnce() -> EvalResult<bool>,
{
    let active = use_cache && !ACTION_PRED_DISABLED.with(Cell::get);
    if active {
        let cached =
            ACTION_PRED_CACHE.with(|c| c.borrow().get(&(current_fp, next_fp, tag)).copied());
        if let Some(result) = cached {
            record_leaf_hit(LeafCacheKind::Action);
            note_lookup(true);
            return Ok(result);
        }
        record_leaf_miss(LeafCacheKind::Action);
        note_lookup(false);
    }

    let result = eval_uncached()?;

    if active {
        trim_action_pred_cache_if_needed();
        ACTION_PRED_CACHE.with(|c| c.borrow_mut().insert((current_fp, next_fp, tag), result));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retained_capacity() -> usize {
        ACTION_PRED_CACHE.with(|map| map.borrow().capacity())
            + ENABLED_ACTION_PRED_PAIR.with(|map| map.borrow().capacity())
            + ENABLED_TRUE_STREAK.with(|map| map.borrow().capacity())
            + FULL_POPULATION_TAGS.with(|set| set.borrow().capacity())
            + ENUM_EXACT_TAGS.with(|set| set.borrow().capacity())
            + WHOLE_NEXT_ENABLED_TAGS.with(|set| set.borrow().capacity())
            + WHOLE_NEXT_ACTION_TAGS.with(|set| set.borrow().capacity())
            + SCAN_PRED_RESULTS.with(|map| map.borrow().capacity())
            + SUBSCRIPT_WATCH_VARS.with(|map| map.borrow().capacity())
    }

    #[test]
    fn release_leaf_result_cache_storage_drops_all_capacity() {
        release_leaf_result_cache_storage();
        ACTION_PRED_CACHE.with(|map| map.borrow_mut().reserve(64));
        ENABLED_ACTION_PRED_PAIR.with(|map| map.borrow_mut().reserve(64));
        ENABLED_TRUE_STREAK.with(|map| map.borrow_mut().reserve(64));
        FULL_POPULATION_TAGS.with(|set| set.borrow_mut().reserve(64));
        ENUM_EXACT_TAGS.with(|set| set.borrow_mut().reserve(64));
        WHOLE_NEXT_ENABLED_TAGS.with(|set| set.borrow_mut().reserve(64));
        WHOLE_NEXT_ACTION_TAGS.with(|set| set.borrow_mut().reserve(64));
        SCAN_PRED_RESULTS.with(|map| map.borrow_mut().reserve(64));
        SUBSCRIPT_WATCH_VARS.with(|map| map.borrow_mut().reserve(64));
        ACTION_PRED_EVICTION_WARNED.with(|warned| warned.set(true));
        ACTION_PRED_DISABLED.with(|disabled| disabled.set(true));
        ACTION_PRED_LOOKUPS.with(|lookups| lookups.set(17));
        ACTION_PRED_LOOKUP_HITS.with(|hits| hits.set(11));
        assert!(retained_capacity() > 0);

        release_leaf_result_cache_storage();

        assert_eq!(retained_capacity(), 0);
        assert_eq!(census_action_pred_cache_len(), 0);
        assert_eq!(census_scan_pred_len(), 0);
        assert!(!ACTION_PRED_EVICTION_WARNED.with(Cell::get));
        assert!(!ACTION_PRED_DISABLED.with(Cell::get));
        assert_eq!(ACTION_PRED_LOOKUPS.with(Cell::get), 0);
        assert_eq!(ACTION_PRED_LOOKUP_HITS.with(Cell::get), 0);
    }
}
