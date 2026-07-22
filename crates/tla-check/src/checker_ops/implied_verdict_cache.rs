// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Value-keyed TRUE-verdict cache for implied-action terms.
//!
//! Refinement obligations (`[][EWD998!Next]_EWD998!vars`) are checked once
//! per state-graph transition. The VM fast path (`implied_action_bytecode`)
//! already collapses the tree-walk to a bytecode execution, but the boolean
//! skeleton still re-executes for every transition — even though the verdict
//! is a pure function of the handful of values the term actually observes:
//! the state slots it reads directly, plus the interpreter-produced results
//! of its pinned zero-arg refinement operators (`token` / `pending`), which
//! project the concrete state pair onto a far smaller abstract state pair.
//! Distinct concrete transitions collapse onto the same observed-input
//! vector, so a verdict cache keyed by those values eliminates the repeated
//! VM executions wholesale.
//!
//! # Soundness
//!
//! The cache key is the exact observed-input vector of the term's compiled
//! bytecode, derived fail-closed by the static state-footprint analysis
//! (`tla_tir::bytecode::analyze_predicate_state_footprint`):
//!
//! * the parent- and successor-side values of every state slot the bytecode
//!   can read (`LoadVar` / `LoadPrime` / `Unchanged`), and
//! * the interpreter-evaluated values of every zero-arg `CallExternal`
//!   operator in BOTH prime modes — evaluated through the same
//!   `eval_zero_arg_external` entry point the VM itself uses, under the same
//!   state bindings and the same fingerprint-keyed transition memo, so key
//!   construction observes byte-identical values to a VM execution.
//!
//! The footprint analysis guarantees the VM execution is a deterministic
//! function of these inputs: every admitted opcode computes strictly from
//! registers and (pure-data-verified) pool constants; the only evaluation-
//! context channels are the reported zero-arg externals. Every key component
//! is additionally guarded by [`Value::is_concrete_data`], so structural
//! equality of components implies semantic interchangeability. Equal keys
//! therefore imply equal VM verdicts.
//!
//! Trust boundary — identical to the VM path: ONLY `true` verdicts are
//! cached and consumed (the same boundary as consuming `Ok(Bool(true))` from
//! the VM). `false` / non-boolean / error outcomes always fall through to a
//! full interpreter evaluation, so every user-visible violation or error is
//! tree-walker-produced, byte-identically to a run with the cache disabled.
//! Cached entries are validated on every hit by FULL `Value` equality of all
//! components (a 64-bit key-hash collision degrades to a miss, never a wrong
//! verdict). Any failure while building the key — an external evaluating to
//! an error, a non-concrete component, a fingerprint failure — skips the
//! cache for that transition (fail open to the unchanged VM + interpreter
//! path).
//!
//! Kill switch: `TY_NO_IMPLIED_VERDICT_CACHE=1` (nothing is analyzed,
//! attached, or consulted). Cross-check: `TY_IMPLIED_BC_XCHECK=1` makes every
//! cache hit fall through to the interpreter and reports any divergence
//! (shared harness with the VM path). Debug counters:
//! `TY_IMPLIED_VC_DEBUG=1`.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::eval::EvalCtx;
use crate::state::ArrayState;
use tla_core::VarIndex;
use tla_value::Value;

feature_flag!(
    pub(crate) no_implied_verdict_cache,
    "TY_NO_IMPLIED_VERDICT_CACHE"
);

feature_flag!(pub(crate) debug_implied_verdict_cache, "TY_IMPLIED_VC_DEBUG");

/// Legacy hard cap on cached entries (kill-switch mode,
/// `TY_IMPLIED_VERDICT_CACHE_CAP=0`). When an insert would exceed the cap the
/// cache is cleared wholesale — entries are pure memoization, so dropping
/// them costs re-evaluation, never correctness.
const IMPLIED_VERDICT_CACHE_MAX_ENTRIES: usize = 2_000_000;

/// Default TOTAL entry budget for the bounded verdict cache (2026-07 memory
/// audit). Entries retain their full observed-input `Value` vectors (parent +
/// successor slot values and refinement-operator results), which pins the
/// underlying value trees — on BufferedRandomAccessFile the unbounded cache
/// retained ~290 MB (82% of peak RSS). Split across two generations that
/// rotate when the current one fills; validated hits in the previous
/// generation are promoted, so the hot working set survives rotations.
/// 64k entries measured ZERO wall-clock cost on BufferedRandomAccessFile
/// (352→101 MB peak RSS at unchanged runtime; promotion keeps the hot
/// abstract-transition vectors resident). Override with
/// `TY_IMPLIED_VERDICT_CACHE_CAP=<total entries>`; `0` restores the legacy
/// unbounded-until-2M-wholesale-clear behavior.
const DEFAULT_IMPLIED_VERDICT_CACHE_CAP: usize = 65_536;

/// Total verdict-cache entry budget from `TY_IMPLIED_VERDICT_CACHE_CAP`.
/// `Some(n)` = bounded two-generation mode; `None` = legacy (`0` requested).
fn implied_verdict_cache_cap_from_env() -> Option<usize> {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("TY_IMPLIED_VERDICT_CACHE_CAP")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            Some(0) => None,
            Some(n) => Some(n.max(2)),
            None => Some(DEFAULT_IMPLIED_VERDICT_CACHE_CAP),
        }
    })
}

/// Adaptive trial: after this many consultations for one term, the hit rate
/// decides whether the cache stays armed.
const ADAPTIVE_TRIAL_LOOKUPS: u64 = 32_768;
/// Minimum hit percentage at the trial boundary; below this the term's cache
/// is disarmed for the rest of the run (key construction has a real
/// per-transition cost, and a near-injective refinement mapping — e.g.
/// EWD998PCal, whose abstract projection is essentially bijective — can never
/// repay it).
const ADAPTIVE_MIN_HIT_PCT: u64 = 25;

/// Per-term verdict-cache plan, attached at bytecode-attach time.
///
/// `direct_slots` / `zero_arg_externals` come from the fail-closed static
/// footprint analysis of the term's compiled bytecode (see module docs).
#[derive(Debug)]
pub(crate) struct ImpliedVerdictCacheSpec {
    /// Process-unique id namespacing this term's entries in the thread-local
    /// cache map (terms and their clones share the id).
    pub(crate) term_id: u64,
    /// State slots the bytecode reads directly (sorted, deduplicated).
    pub(crate) direct_slots: Arc<[u16]>,
    /// Zero-arg external operator names the bytecode can call (sorted).
    pub(crate) zero_arg_externals: Arc<[String]>,
    /// Consultations of this term's cache (hits + misses; skips excluded).
    lookups: AtomicU64,
    /// Validated TRUE hits for this term.
    hits: AtomicU64,
    /// Set at the adaptive-trial boundary when the hit rate is too low;
    /// callers then skip key construction entirely (pure perf policy — the
    /// cache is memoization only, so disarming can never change a verdict).
    disarmed: std::sync::atomic::AtomicBool,
}

impl ImpliedVerdictCacheSpec {
    pub(crate) fn new(
        term_id: u64,
        direct_slots: Arc<[u16]>,
        zero_arg_externals: Arc<[String]>,
    ) -> Self {
        Self {
            term_id,
            direct_slots,
            zero_arg_externals,
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            disarmed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Whether callers should skip this term's cache (adaptive disarm).
    pub(crate) fn is_disarmed(&self) -> bool {
        self.disarmed.load(Ordering::Relaxed)
    }
}

/// Allocate a process-unique term id.
pub(crate) fn next_term_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The observed-input vector for one (parent, successor) evaluation of one
/// term, plus its 64-bit key hash.
pub(crate) struct ImpliedVerdictKey {
    hash: u64,
    components: SmallVec<[Value; 12]>,
}

thread_local! {
    static VERDICT_CACHE: RefCell<VerdictCacheMap> = RefCell::new(VerdictCacheMap::default());
}

type VerdictChain = SmallVec<[Arc<[Value]>; 1]>;

/// Bounded two-generation storage for known-TRUE verdict vectors
/// (2026-07 memory audit; see [`DEFAULT_IMPLIED_VERDICT_CACHE_CAP`]).
///
/// `(term_id, key_hash) -> known-TRUE component vectors` (hash collisions
/// chain; validated by full component equality on every hit). All inserts go
/// to `cur`; when `cur` reaches half the total budget, it rotates into
/// `prev` (whose contents are dropped). Hits found in `prev` are promoted
/// back into `cur`, so recently reused vectors survive rotations. Eviction
/// can never affect correctness — entries are pure memoization and every hit
/// still validates full component equality.
struct VerdictCacheMap {
    cur: FxHashMap<(u64, u64), VerdictChain>,
    prev: FxHashMap<(u64, u64), VerdictChain>,
    /// Component vectors stored in `cur` (chains can hold several).
    cur_entries: usize,
    /// `Some(per_generation_cap)` = bounded mode; `None` = legacy mode.
    cap_per_gen: Option<usize>,
}

impl Default for VerdictCacheMap {
    fn default() -> Self {
        Self {
            cur: FxHashMap::default(),
            prev: FxHashMap::default(),
            cur_entries: 0,
            cap_per_gen: implied_verdict_cache_cap_from_env().map(|total| (total / 2).max(1)),
        }
    }
}

impl VerdictCacheMap {
    /// Rotate generations if inserting `added` more vectors would exceed the
    /// per-generation budget (bounded mode only).
    fn maybe_rotate(&mut self, added: usize) {
        if let Some(cap) = self.cap_per_gen {
            if self.cur_entries + added > cap && self.cur_entries > 0 {
                self.prev = std::mem::take(&mut self.cur);
                self.cur_entries = 0;
            }
        }
    }
}

/// Census probe (TY_MEM_CENSUS): total component vectors currently cached.
pub(crate) fn census_implied_verdict_len() -> usize {
    VERDICT_CACHE.with(|cache| {
        let cache = cache.borrow();
        cache.cur.values().map(SmallVec::len).sum::<usize>()
            + cache.prev.values().map(SmallVec::len).sum::<usize>()
    })
}

// Debug counters (TY_IMPLIED_VC_DEBUG=1). Cheap enough to keep unconditional:
// one relaxed atomic increment per cache consultation.
static VC_HITS: AtomicU64 = AtomicU64::new(0);
static VC_MISSES: AtomicU64 = AtomicU64::new(0);
static VC_SKIPS: AtomicU64 = AtomicU64::new(0);
static VC_INSERTS: AtomicU64 = AtomicU64::new(0);
static VC_CLEARS: AtomicU64 = AtomicU64::new(0);

fn debug_tick() {
    if !debug_implied_verdict_cache() {
        return;
    }
    let total = VC_HITS.load(Ordering::Relaxed)
        + VC_MISSES.load(Ordering::Relaxed)
        + VC_SKIPS.load(Ordering::Relaxed);
    if total % 500_000 == 0 {
        eprintln!(
            "[implied-vc] lookups={total} hits={} misses={} skips={} inserts={} clears={}",
            VC_HITS.load(Ordering::Relaxed),
            VC_MISSES.load(Ordering::Relaxed),
            VC_SKIPS.load(Ordering::Relaxed),
            VC_INSERTS.load(Ordering::Relaxed),
            VC_CLEARS.load(Ordering::Relaxed),
        );
    }
}

/// Record a skipped consultation (key construction failed).
fn record_skip() {
    VC_SKIPS.fetch_add(1, Ordering::Relaxed);
    debug_tick();
}

/// Build the observed-input key for `spec` over the bound transition.
///
/// Component order is FIXED (slots in `direct_slots` order, parent side then
/// successor side; then externals in `zero_arg_externals` order, unprimed
/// then primed), so lookup- and insert-time vectors are directly comparable.
///
/// Returns `None` (and records a skip) when any component is unavailable,
/// non-concrete, non-fingerprintable, or an external evaluation fails — the
/// caller then proceeds exactly as if the cache did not exist.
pub(crate) fn build_implied_verdict_key(
    ctx: &EvalCtx,
    spec: &ImpliedVerdictCacheSpec,
    parent: &ArrayState,
    succ: &ArrayState,
) -> Option<ImpliedVerdictKey> {
    let mut components: SmallVec<[Value; 12]> =
        SmallVec::with_capacity(spec.direct_slots.len() * 2 + spec.zero_arg_externals.len() * 2);
    // Seed mixes the term id so equal component vectors of different terms
    // hash apart even before the (term_id, hash) map key disambiguates.
    let mut hash: u64 = 0x9e37_79b9_7f4a_7c15 ^ spec.term_id.wrapping_mul(0xff51_afd7_ed55_8ccd);

    let push = |components: &mut SmallVec<[Value; 12]>, hash: &mut u64, v: Value| -> bool {
        if !v.is_concrete_data() {
            return false;
        }
        match v.fingerprint_extend(*hash) {
            Ok(next) => {
                *hash = next;
                components.push(v);
                true
            }
            Err(_) => false,
        }
    };

    let parent_len = parent.values().len();
    let succ_len = succ.values().len();
    for &slot in spec.direct_slots.iter() {
        let idx = slot as usize;
        if idx >= parent_len || idx >= succ_len {
            record_skip();
            return None;
        }
        let var = VarIndex(slot);
        if !push(&mut components, &mut hash, parent.get(var))
            || !push(&mut components, &mut hash, succ.get(var))
        {
            record_skip();
            return None;
        }
    }

    for name in spec.zero_arg_externals.iter() {
        // Both prime modes are observable inputs (the dynamic prime flag
        // selects which one an execution reads; keying on both is a sound
        // superset). Evaluation goes through the exact entry point the VM's
        // `CallExternal` handler uses, under the same state bindings and
        // transition memo, so the values are byte-identical to what a VM
        // execution would observe.
        for prime in [false, true] {
            match tla_eval::bytecode_vm::eval_zero_arg_external(ctx, name, prime) {
                Ok(v) => {
                    if !push(&mut components, &mut hash, v) {
                        record_skip();
                        return None;
                    }
                }
                Err(_) => {
                    record_skip();
                    return None;
                }
            }
        }
    }

    Some(ImpliedVerdictKey { hash, components })
}

/// Probe the cache for a known-TRUE verdict of `spec.term_id` under `key`.
///
/// A hit requires full component-vector equality (`Value::eq`); a key-hash
/// collision therefore degrades to a miss.
pub(crate) fn lookup_implied_verdict_true(
    spec: &ImpliedVerdictCacheSpec,
    key: &ImpliedVerdictKey,
) -> bool {
    let hit = VERDICT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let cache = &mut *cache;
        let map_key = (spec.term_id, key.hash);
        if cache.cur.get(&map_key).is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry.as_ref() == key.components.as_slice())
        }) {
            return true;
        }
        // Previous-generation probe with promotion: a validated hit moves the
        // whole chain into the current generation so it survives the next
        // rotation (bounded mode; `prev` is always empty in legacy mode).
        let prev_hit = cache.prev.get(&map_key).is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry.as_ref() == key.components.as_slice())
        });
        if prev_hit {
            if let Some(chain) = cache.prev.remove(&map_key) {
                let moved = chain.len();
                cache.maybe_rotate(moved);
                let entries = cache.cur.entry(map_key).or_default();
                for vector in chain {
                    if !entries.iter().any(|entry| *entry == vector) {
                        entries.push(vector);
                        cache.cur_entries += 1;
                    }
                }
            }
        }
        prev_hit
    });
    if hit {
        VC_HITS.fetch_add(1, Ordering::Relaxed);
        spec.hits.fetch_add(1, Ordering::Relaxed);
    } else {
        VC_MISSES.fetch_add(1, Ordering::Relaxed);
    }
    // Adaptive disarm: at the trial boundary, a hit rate below the floor
    // means the observed-input vectors barely repeat (near-injective
    // refinement mapping) and key construction is a pure per-transition tax.
    let lookups = spec.lookups.fetch_add(1, Ordering::Relaxed) + 1;
    if lookups == ADAPTIVE_TRIAL_LOOKUPS {
        let hits = spec.hits.load(Ordering::Relaxed);
        if hits * 100 < lookups * ADAPTIVE_MIN_HIT_PCT {
            spec.disarmed.store(true, Ordering::Relaxed);
            if debug_implied_verdict_cache() {
                eprintln!(
                    "[implied-vc] term {} DISARMED at trial boundary: hits={hits}/{lookups}",
                    spec.term_id
                );
            }
        }
    }
    debug_tick();
    hit
}

/// Record a VM-produced TRUE verdict for `spec.term_id` under `key`.
pub(crate) fn insert_implied_verdict_true(spec: &ImpliedVerdictCacheSpec, key: ImpliedVerdictKey) {
    VERDICT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let cache = &mut *cache;
        match cache.cap_per_gen {
            Some(_) => cache.maybe_rotate(1),
            None => {
                // Legacy mode: single generation, wholesale clear at the cap.
                if cache.cur_entries >= IMPLIED_VERDICT_CACHE_MAX_ENTRIES {
                    cache.cur.clear();
                    cache.cur_entries = 0;
                    VC_CLEARS.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let components: Arc<[Value]> = Arc::from(key.components.into_vec());
        let entries = cache.cur.entry((spec.term_id, key.hash)).or_default();
        // Idempotent re-insert (possible under TY_IMPLIED_BC_XCHECK, where
        // hits fall through to the interpreter and reach the insert site).
        if entries.iter().any(|entry| *entry == components) {
            return;
        }
        entries.push(components);
        cache.cur_entries += 1;
        VC_INSERTS.fetch_add(1, Ordering::Relaxed);
    });
}

/// Final debug summary (TY_IMPLIED_VC_DEBUG=1), printed by the checker's
/// telemetry epilogue.
pub(crate) fn debug_summary() {
    if !debug_implied_verdict_cache() {
        return;
    }
    eprintln!(
        "[implied-vc] FINAL hits={} misses={} skips={} inserts={} clears={}",
        VC_HITS.load(Ordering::Relaxed),
        VC_MISSES.load(Ordering::Relaxed),
        VC_SKIPS.load(Ordering::Relaxed),
        VC_INSERTS.load(Ordering::Relaxed),
        VC_CLEARS.load(Ordering::Relaxed),
    );
}
