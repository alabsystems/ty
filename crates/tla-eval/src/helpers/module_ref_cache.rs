// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cache infrastructure for module reference resolution.
//!
//! Contains cache types, thread-local statics, and lifecycle functions
//! for INSTANCE WITH substitution caching.
//!
//! Part of #1643 (module_ref.rs decomposition).
//!
//! ## Cache Lifecycle Contract (Part of #2376, #2413)
//!
//! All caches in this module follow the unified lifecycle contract from
//! `crates/tla-eval/src/cache/lifecycle.rs`. Events are dispatched via
//! `on_cache_event(CacheEvent)`.
//!
//! | Cache | Lifecycle | Clear trigger | Bound |
//! |---|---|---|---|
//! | `CHAINED_REF_CACHE` | PerRun | RunReset, TestReset, PhaseBoundary | 10K soft cap (cliff) |
//! | `MODULE_REF_SCOPE_CACHE` | PerRun | RunReset, TestReset, PhaseBoundary | 10K soft cap (cliff) |
//! | `EAGER_BINDINGS_CACHE` | PerState | State change (full), next-state change (selective) | 10K soft cap (cliff) |
//!
//! **Clear path:** `clear_run_reset_impl()` → `clear_module_ref_caches()` clears all three.
//! **Trim path:** `trim_eval_entry_caches()` → `trim_module_ref_caches()` on every EvalExit.
//! **Selective eviction:** `evict_next_state_eager_bindings()` retains `is_next_mode=false` entries.

use super::super::{EvalCtx, InstanceInfo, OpEnv, OpEvalDeps};
use crate::value::Value;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::Arc;
use tla_core::ast::Substitution;
use tla_core::name_intern::NameId;

// === Cache Types ===

#[derive(Clone)]
pub(super) struct ChainedRefCacheEntry {
    pub(super) instance_info: InstanceInfo,
    pub(super) merged_local_ops: Arc<OpEnv>,
    /// Pre-wrapped Arc for instance_substitutions, reused across eval_entry calls.
    /// Stabilizes the Arc pointer so SUBST_CACHE keys match across calls.
    pub(super) instance_subs_arc: Arc<Vec<Substitution>>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub(super) struct ChainedRefCacheKey {
    shared_id: u64,
    local_ops_id: usize,
    instance_subs_id: usize,
    chain_key: String,
}

// Part of #3962: Consolidated CHAINED_REF_CACHE + MODULE_REF_SCOPE_CACHE +
// EAGER_BINDINGS_CACHE + PARAM_SUBS_CACHE into a single TLS struct.
// Previously 4 separate `thread_local!` declarations — each access was a
// separate `_tlv_get_addr` call on macOS (~5ns each). All four caches are
// cleared together in `clear_module_ref_caches()` and accessed during
// INSTANCE WITH resolution. Now 1 TLS access covers all four.
//
// Part of #3962: PARAM_SUBS_CACHE (previously in module_ref.rs) consolidated
// here. It stores pre-allocated Arc<Vec<Substitution>> for parametrized
// INSTANCE callsites, keyed by AST pointer identity. PerRun lifecycle.
pub(super) struct ModuleRefCaches {
    pub(super) chained_ref: FxHashMap<ChainedRefCacheKey, ChainedRefCacheEntry>,
    pub(super) module_ref_scope: FxHashMap<ModuleRefScopeKey, ModuleRefScopeEntry>,
    pub(super) eager_bindings: EagerBindingsMap,
    /// Part of #3962: Consolidated from standalone PARAM_SUBS_CACHE thread_local.
    /// Cache for parametrized INSTANCE substitution Arcs (Part of #2980).
    /// Key: param_exprs.as_ptr() as usize (AST Vec data pointer, stable per callsite).
    /// Value: pre-allocated Arc<Vec<Substitution>> reused across evaluations.
    /// Lifecycle: PerRun. Bounded: one entry per parametrized INSTANCE callsite.
    pub(super) param_subs: FxHashMap<usize, Arc<Vec<Substitution>>>,
}

std::thread_local! {
    #[allow(clippy::missing_const_for_thread_local)] // FxHashMap::default() is not const
    pub(super) static MODULE_REF_CACHES: RefCell<ModuleRefCaches> =
        RefCell::new(ModuleRefCaches {
            chained_ref: FxHashMap::default(),
            module_ref_scope: FxHashMap::default(),
            eager_bindings: FxHashMap::default(),
            param_subs: FxHashMap::default(),
        });
}

/// Cache for non-chained `eval_module_ref` scope construction.
///
/// Fix #2364: `eval_module_ref` calls `compose_substitutions` and creates a new
/// `Arc<Vec<Substitution>>` on every call, even when the result is identical.
/// This makes SUBST_CACHE keys unstable across eval_entry calls.
///
/// Cache key: (shared_id, instance_name hash, op_name hash, outer instance_subs_id)
/// Cache value: pre-wrapped Arc for the composed substitutions + merged local_ops
#[derive(Clone)]
pub(super) struct ModuleRefScopeEntry {
    pub(super) effective_subs_arc: Arc<Vec<Substitution>>,
    pub(super) local_ops_arc: Arc<OpEnv>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
pub(super) struct ModuleRefScopeKey {
    pub(super) shared_id: u64,
    pub(super) instance_name_id: NameId,
    pub(super) outer_subs_id: usize,
    pub(super) outer_local_ops_id: usize,
}

/// Cache for eagerly-evaluated substitution bindings within the same state.
///
/// Fix #2364: `build_eager_subst_bindings` re-evaluates ALL substitution RHS expressions
/// on every `eval_module_ref` / `eval_chained_module_ref` call, even when the same
/// INSTANCE scope is accessed multiple times for the same state (e.g., checking
/// invariants, constraints, and properties all reference EWD998Chan!...).
///
/// Key: (subs_ptr, local_ops_ptr, state_ptr, next_state_ptr, is_next_mode)
/// Value: pre-evaluated eager bindings Arc
///
/// Uses Arc pointer identity for scope identification, which works for both
/// non-chained (MODULE_REF_SCOPE_CACHE-backed Arcs) and chained
/// (CHAINED_REF_CACHE-backed Arcs) paths. Both caches provide stable Arc
/// pointers across eval_entry calls.
///
/// Fix #2364: When `is_next_mode=false`, `next_state_ptr` is set to 0 because
/// substitution RHS expressions in Current mode only read current state variables.
/// This allows cache hits across transitions from the same source state, even when
/// the next_state changes. Combined with selective eviction (retaining
/// `is_next_mode=false` entries when only next_state_changed), this reduces
/// EAGER_BINDINGS_CACHE rebuilds from N_transitions to N_from_states.
#[derive(Clone, Hash, PartialEq, Eq)]
pub(super) struct EagerBindingsKey {
    /// Arc pointer identity for the effective substitutions.
    subs_ptr: usize,
    /// Arc pointer identity for the local_ops scope.
    local_ops_ptr: usize,
    /// Fix #3447: State generation counter — monotonically increasing, no ABA.
    /// Replaces `state_ptr: usize` which was vulnerable to allocator recycling.
    state_gen: u64,
    /// Fix #3447: Next-state generation counter. Set to 0 when `is_next_mode=false`
    /// because Current-mode substitutions don't depend on next_state.
    next_state_gen: u64,
    /// Current state lookup mode (Current vs Next). Substitution RHS expressions
    /// read state variables, which resolve differently depending on the mode.
    is_next_mode: bool,
    /// Part of #2980: Content hash of parametrized INSTANCE argument values.
    ///
    /// When a parametrized INSTANCE like `P(Succ) == INSTANCE M` is called via
    /// `\A Succ \in SuccSet : P(Succ)!Op`, the `PARAM_SUBS_CACHE` reuses the same
    /// Arc pointer across all iterations (for scope cache efficiency). This hash
    /// disambiguates cache entries by the actual argument values, preventing stale
    /// cache hits when the same callsite is invoked with different arguments.
    ///
    /// Zero when no parametrized INSTANCE is active (non-parametrized callers).
    param_args_hash: u64,
}

pub(super) type EagerBindingsMap =
    FxHashMap<EagerBindingsKey, Arc<Vec<(NameId, Value, OpEvalDeps)>>>;

pub(super) fn chained_ref_cache_key(ctx: &EvalCtx, chain_key: String) -> ChainedRefCacheKey {
    let local_ops_id = ctx
        .local_ops
        .as_ref()
        .map_or(0, |local_ops| Arc::as_ptr(local_ops) as usize);
    let instance_subs_id = ctx
        .instance_substitutions
        .as_ref()
        .map_or(0, |subs| Arc::as_ptr(subs) as usize);
    ChainedRefCacheKey {
        shared_id: ctx.shared.id,
        local_ops_id,
        instance_subs_id,
        chain_key,
    }
}

// === Run-lifetime producer-identity scope-entry memo (#3447/#4170 epoch policy) ===
//
// The per-state caches above (`module_ref_scope`, `chained_ref`) are keyed by
// ambient Arc POINTERS and are therefore cleared by `clear_module_ref_caches()`
// at every eval-scope boundary (the #3447 defense against allocator ABA on
// those pointers). On INSTANCE-heavy specs with a VIEW or per-successor
// invariant boundaries (MCNano: ~13 boundary clears per state), every `M!Op`
// prepare after a clear re-runs `compute_effective_instance_substitutions`
// (CES) plus the full instance-ops merge — 6.8M CES calls over ONE distinct
// result on MCNanoMedium.
//
// This memo removes the rebuild cost WITHOUT relaxing any #3447 clear: the
// per-state pointer-keyed caches keep their exact lifecycle, and on a miss
// the builder first consults this RUN-lifetime memo. On a memo hit the entry
// is reused and re-inserted into the per-state cache; on a miss the entry is
// built exactly as before and recorded in both.
//
// # Keying: pinned-pointer identity ONLY (no content-hash trust)
//
// An early revision keyed ambient scopes by the #3099 content fingerprints.
// The debug determinism check caught that trust being violated on the
// Disruptor liveness specs: SYNTHESIZED operator defs (dummy spans) alias in
// the (NameId, span, arity) local_ops fingerprint while differing in content
// — same key, different merge result. So this memo trusts NO content hash.
// The key is exact object identity end to end:
//
// - site identity: the interned instance name (named refs) or the run-stable
//   compound chain key string (chained refs);
// - `shared_ptr`: the `Arc<SharedCtx>` address, PINNED by the memo. All
//   shared tables the builders read (instance_ops, instance_implicit_targets,
//   var_registry, config_constants, ops, instances) live in `SharedCtx`,
//   which is copy-on-write: any mutation goes through `Arc::make_mut`, and
//   the memo's pin forces refcount >= 2, so mutation ALWAYS produces a new
//   allocation at a new address -> clean memo miss. Same pointer therefore
//   proves bit-identical shared tables.
// - ambient `local_ops` / `instance_substitutions`: 0 when absent, else the
//   Arc address — accepted ONLY when that address is pinned by a live memo
//   entry (see [`is_pinned_local_ops`] / [`is_pinned_subs`]). A pinned
//   address cannot be freed or recycled while pinned (no ABA) and its
//   pointee is immutable (pin forces COW), so pointer equality proves object
//   identity. Unpinned ambient scopes BAIL to the unmemoized build — the
//   memo self-bootstraps: top-level `M!Op` calls (no ambient scope) seed
//   entries whose pinned Arcs then become the ambient scopes of nested
//   `M!Op` calls, transitively covering the INSTANCE nest while every
//   context NOT produced by this memo (per-state rebuilt envs, synthesized
//   liveness scopes, parameterized-INSTANCE frames) keeps the status-quo
//   build path.
// - `let_def_overlay` non-empty BAILS: CES's visibility probe (`get_op`)
//   consults the overlay, which no key component covers.
//
// Given identical shared tables and identical ambient objects, the builders
// (CES + instance-ops merge + chain resolution) are pure structural functions
// — they never read state values — so the memoized entry is exactly what a
// rebuild would produce. Debug builds re-run the full build on every hit and
// assert structural equality, so any violation fails loudly across the test
// suite (this is the check that caught the fingerprint aliasing above).
//
// Lifecycle: PerRun — cleared in `clear_run_reset_impl()` via
// `clear_module_ref_run_memos()`, bounded with clear-on-overflow. Maps and
// pin sets clear TOGETHER, atomically — a pointer key must never outlive the
// pin that legitimizes it. Clearing at any point is sound (rebuild fresh).
//
// Kill switch: `TY_LEGACY_EPOCH_CLEAR=1` bypasses the memo entirely
// (byte-for-byte the previous rebuild-on-every-boundary behavior).

/// `TY_LEGACY_EPOCH_CLEAR=1` restores the pre-epoch-policy behavior
/// (no run-lifetime scope memo).
#[inline]
pub(crate) fn legacy_epoch_clear() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("TY_LEGACY_EPOCH_CLEAR").is_some())
}

/// Producer-identity key for named `M!Op` scope entries. All components are
/// exact identities (interned name, pinned Arc addresses, 0 for absent) — no
/// content hashing anywhere.
#[derive(Clone, Hash, PartialEq, Eq)]
pub(super) struct RunNamedScopeKey {
    /// `Arc<SharedCtx>` address, pinned by the memo (COW ⇒ same ptr = same tables).
    pub(super) shared_ptr: usize,
    pub(super) instance_name_id: NameId,
    /// Ambient `instance_substitutions`: 0 or a pinned Arc address.
    pub(super) inst_subs_ptr: usize,
    /// Ambient `local_ops`: 0 or a pinned Arc address.
    pub(super) local_ops_ptr: usize,
}

/// Producer-identity key for chained `A!B!...!Op` scope entries. The chain key
/// string is the run-stable compound chain shape (`module_ref_compound_key`).
#[derive(Clone, Hash, PartialEq, Eq)]
pub(super) struct RunChainedScopeKey {
    pub(super) shared_ptr: usize,
    pub(super) chain_key: String,
    pub(super) inst_subs_ptr: usize,
    pub(super) local_ops_ptr: usize,
}

pub(super) struct RunScopeMemos {
    pub(super) named: FxHashMap<RunNamedScopeKey, ModuleRefScopeEntry>,
    pub(super) chained: FxHashMap<RunChainedScopeKey, ChainedRefCacheEntry>,
    /// Addresses of every `local_ops_arc` / `merged_local_ops` Arc held
    /// (pinned) by a live entry of the two maps above. A pointer in this set
    /// cannot be freed or recycled while the set contains it (the owning
    /// entry holds the Arc) and its pointee cannot be mutated in place (pin
    /// forces copy-on-write), so it is a legitimate ABA-free ambient
    /// identity. Cleared together with the maps — never partially — so no
    /// key derived from this set can outlive the pin backing it.
    pub(super) pinned_local_ops: rustc_hash::FxHashSet<usize>,
    /// Addresses of every `effective_subs_arc` / `instance_subs_arc` pinned
    /// by a live entry. Same contract as `pinned_local_ops`.
    pub(super) pinned_subs: rustc_hash::FxHashSet<usize>,
    /// Pins for every `Arc<SharedCtx>` appearing in a key. Guarantees (a) the
    /// address cannot be recycled and (b) any `SharedCtx` mutation
    /// (`Arc::make_mut`) is forced to copy-on-write to a NEW address, so a
    /// stale key can never match a mutated shared context.
    pub(super) pinned_shared: FxHashMap<usize, Arc<crate::core::SharedCtx>>,
}

impl RunScopeMemos {
    /// Full clear: all maps AND all pin sets, atomically from the caller's
    /// perspective (single &mut). Partial clears are forbidden — a
    /// pointer-keyed entry must never outlive the pin that legitimizes its
    /// key.
    fn clear_all(&mut self) {
        self.named.clear();
        self.chained.clear();
        self.pinned_local_ops.clear();
        self.pinned_subs.clear();
        self.pinned_shared.clear();
    }
}

std::thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    pub(super) static RUN_SCOPE_MEMOS: RefCell<RunScopeMemos> = RefCell::new(RunScopeMemos {
        named: FxHashMap::default(),
        chained: FxHashMap::default(),
        pinned_local_ops: rustc_hash::FxHashSet::default(),
        pinned_subs: rustc_hash::FxHashSet::default(),
        pinned_shared: FxHashMap::default(),
    });
}

/// Whether `ptr` is the address of a `local_ops` `OpEnv` Arc currently pinned
/// by a live run-memo entry (guaranteed alive, immutable, and un-recyclable
/// while pinned). Used by the run memo AND by `cache::subst_chain_memo` to
/// upgrade the "recursive ambient scope ⇒ pointer-keyed ⇒ bail" path into a
/// sound pointer key: pinned addresses have per-run identity, so keying by
/// them cannot alias across states (no allocator ABA) and cannot alias
/// recursion frames (#3156 gives each frame its own Arc — same address means
/// same frame content).
#[inline]
pub(crate) fn is_pinned_local_ops(ptr: usize) -> bool {
    RUN_SCOPE_MEMOS.with(|m| m.borrow().pinned_local_ops.contains(&ptr))
}

/// Whether `ptr` is the address of a substitution-vector Arc currently pinned
/// by a live run-memo entry. Same contract as [`is_pinned_local_ops`].
#[inline]
pub(crate) fn is_pinned_subs(ptr: usize) -> bool {
    RUN_SCOPE_MEMOS.with(|m| m.borrow().pinned_subs.contains(&ptr))
}

/// Cap on run-memo entries per family. Keys are O(INSTANCE sites × ambient
/// scopes) — effectively never hit; bounds memory for pathological workloads.
/// Overflow clears the map (sound: later calls rebuild fresh).
const RUN_SCOPE_MEMO_CAP: usize = 8_192;

/// Resolve the ambient (instance_subs, local_ops) identities for the run
/// memo, or `None` (bail to the unmemoized build) when no exact identity
/// exists.
///
/// Policy (see the module-level soundness comment): each ambient scope must
/// be ABSENT (identity 0) or a memo-PINNED Arc (identity = address). No
/// content fingerprints — the Disruptor synthesized-def aliasing showed the
/// #3099 local_ops fingerprint is not injective for dummy-span defs. A
/// non-empty `let_def_overlay` also bails: CES's `get_op` visibility probe
/// consults it and no key component covers it.
#[inline]
pub(super) fn run_scope_memo_ambient_ids(ctx: &EvalCtx) -> Option<(usize, usize)> {
    if legacy_epoch_clear() {
        return None;
    }
    if !ctx.let_def_overlay.is_empty() {
        return None;
    }
    let local_ops_ptr = match ctx.local_ops.as_ref() {
        None => 0usize,
        Some(ops) => {
            let ptr = Arc::as_ptr(ops) as usize;
            if !is_pinned_local_ops(ptr) {
                return None;
            }
            ptr
        }
    };
    let inst_subs_ptr = match ctx.instance_substitutions.as_ref() {
        None => 0usize,
        Some(subs) => {
            let ptr = Arc::as_ptr(subs) as usize;
            if !is_pinned_subs(ptr) {
                return None;
            }
            ptr
        }
    };
    Some((inst_subs_ptr, local_ops_ptr))
}

pub(super) fn run_named_scope_memo_get(key: &RunNamedScopeKey) -> Option<ModuleRefScopeEntry> {
    RUN_SCOPE_MEMOS.with(|m| m.borrow().named.get(key).cloned())
}

pub(super) fn run_named_scope_memo_insert(
    key: RunNamedScopeKey,
    entry: ModuleRefScopeEntry,
    shared: &Arc<crate::core::SharedCtx>,
) {
    debug_assert_eq!(key.shared_ptr, Arc::as_ptr(shared) as usize);
    RUN_SCOPE_MEMOS.with(|m| {
        let mut m = m.borrow_mut();
        if m.named.len() + m.chained.len() >= RUN_SCOPE_MEMO_CAP {
            // Full clear only — see RunScopeMemos::clear_all.
            m.clear_all();
        }
        m.pinned_shared
            .entry(Arc::as_ptr(shared) as usize)
            .or_insert_with(|| Arc::clone(shared));
        m.pinned_local_ops
            .insert(Arc::as_ptr(&entry.local_ops_arc) as usize);
        m.pinned_subs
            .insert(Arc::as_ptr(&entry.effective_subs_arc) as usize);
        m.named.insert(key, entry);
    });
}

pub(super) fn run_chained_scope_memo_get(key: &RunChainedScopeKey) -> Option<ChainedRefCacheEntry> {
    RUN_SCOPE_MEMOS.with(|m| m.borrow().chained.get(key).cloned())
}

pub(super) fn run_chained_scope_memo_insert(
    key: RunChainedScopeKey,
    entry: ChainedRefCacheEntry,
    shared: &Arc<crate::core::SharedCtx>,
) {
    debug_assert_eq!(key.shared_ptr, Arc::as_ptr(shared) as usize);
    RUN_SCOPE_MEMOS.with(|m| {
        let mut m = m.borrow_mut();
        if m.named.len() + m.chained.len() >= RUN_SCOPE_MEMO_CAP {
            // Full clear only — see RunScopeMemos::clear_all.
            m.clear_all();
        }
        m.pinned_shared
            .entry(Arc::as_ptr(shared) as usize)
            .or_insert_with(|| Arc::clone(shared));
        m.pinned_local_ops
            .insert(Arc::as_ptr(&entry.merged_local_ops) as usize);
        m.pinned_subs
            .insert(Arc::as_ptr(&entry.instance_subs_arc) as usize);
        m.chained.insert(key, entry);
    });
}

/// Clear the run-lifetime scope memos (run/phase/test reset). Dropping entries
/// drops their pinned Arcs in the same breath. NOT called at eval-scope
/// boundaries — that is the point: the memoized values are state-independent
/// structural data whose producer identity is content-keyed (no ABA).
pub fn clear_module_ref_run_memos() {
    RUN_SCOPE_MEMOS.with(|m| m.borrow_mut().clear_all());
}

// === Cache Lifecycle ===

/// Selectively evict EAGER_BINDINGS_CACHE entries that depend on next_state.
///
/// Fix #2364: When only next_state_changed, retain entries with `is_next_mode=false`.
/// These entries only depend on current state (unchanged) and have `next_state_ptr=0`
/// in their key, so they're safe to keep.
pub fn evict_next_state_eager_bindings() {
    MODULE_REF_CACHES.with(|c| c.borrow_mut().eager_bindings.retain(|k, _| !k.is_next_mode));
}

/// Clear module reference scope caches for test isolation.
///
/// These caches store structural data (composed substitutions, merged local_ops)
/// that is independent of state values but may become stale between test runs
/// when shared_id is reused.
/// Part of #3962: Single TLS access clears all 4 caches (was 2 before —
/// MODULE_REF_CACHES + separate PARAM_SUBS_CACHE).
pub fn clear_module_ref_caches() {
    MODULE_REF_CACHES.with(|c| {
        let mut caches = c.borrow_mut();
        caches.chained_ref.clear();
        caches.module_ref_scope.clear();
        caches.eager_bindings.clear();
        // Part of #2980/#3962: Parametrized INSTANCE substitution Arc cache,
        // now consolidated into MODULE_REF_CACHES.
        caches.param_subs.clear();
    });
}

/// Clear only the eager bindings cache (state-dependent substitution results).
///
/// Called by `eval_entry` on state change to prevent unbounded growth.
/// Scope caches (chained_ref, module_ref_scope) are NOT cleared
/// because they are state-independent and provide stable Arc pointers.
pub fn clear_eager_bindings_cache() {
    MODULE_REF_CACHES.with(|c| c.borrow_mut().eager_bindings.clear());
}

// Part of #2413 U6: Soft caps for PerRun module-ref caches.
const CHAINED_REF_CACHE_SOFT_CAP: usize = 10_000;
const MODULE_REF_SCOPE_CACHE_SOFT_CAP: usize = 10_000;
const EAGER_BINDINGS_CACHE_SOFT_CAP: usize = 10_000;

/// Trim module-ref caches when they exceed soft caps.
///
/// Part of #2413: Called from `trim_eval_entry_caches` on every `EvalExit`.
/// Uses cliff eviction (full clear on exceeding cap). Note: main lifecycle
/// caches migrated to retain-half (#3025); module-ref caches not yet migrated.
pub fn trim_module_ref_caches() {
    MODULE_REF_CACHES.with(|c| {
        let caches = c.borrow();
        let chained_over = caches.chained_ref.len() > CHAINED_REF_CACHE_SOFT_CAP;
        let scope_over = caches.module_ref_scope.len() > MODULE_REF_SCOPE_CACHE_SOFT_CAP;
        let eager_over = caches.eager_bindings.len() > EAGER_BINDINGS_CACHE_SOFT_CAP;
        drop(caches);
        if chained_over || scope_over || eager_over {
            let mut caches = c.borrow_mut();
            if chained_over {
                caches.chained_ref.clear();
            }
            if scope_over {
                caches.module_ref_scope.clear();
            }
            if eager_over {
                caches.eager_bindings.clear();
            }
        }
    });
}

#[cfg(test)]
#[path = "module_ref_cache_tests.rs"]
mod tests;
