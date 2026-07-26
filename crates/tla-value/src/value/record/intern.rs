// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Per-thread record intern table: hash-consing for `RecordValue` entry
//! storage (Program A / RC1c, records-first phase).
//!
//! Record-heavy specs (MultiPaxos messages, NanoBlockchain blocks) rebuild
//! records that are STRUCTURALLY EQUAL to records already alive — the same
//! Paxos message record is re-materialized by operator application for
//! millions of successor states. Post-value-canon profiling attributes ~59%
//! of MultiPaxos' residual wall to exactly this clone/drop/malloc churn.
//! Interning at construction converts that churn into pointer traffic: equal
//! records SHARE one `entries` allocation, which also compounds with the
//! landed value-canon layers (eq/cmp ptr_eq short-circuits, cached additive
//! fingerprints computed once per DISTINCT value, pointer-keyed
//! permute/subst memos, RSS dedup).
//!
//! # What is interned (and what is deliberately NOT)
//!
//! Interned: records built by the three static constructors
//! (`from_sorted_entries` / `from_entries` / `from_sorted_str_entries`),
//! which funnel every record literal (AST + TIR), `RecordBuilder`, and
//! record-set enumeration in the tree. Cross-state reuse is real there:
//! message constructors like `PrepareMsg(r, b)` yield the same few hundred
//! distinct records for the whole run.
//!
//! NOT interned (evidence-gated, mirroring the permute-memo Set lesson —
//! its Set draft was a 2.3x regression because set storage is
//! fresh-per-successor):
//! - Mutation outputs (`insert` / `insert_owned` / `update_existing_field_id`
//!   / `take_at_field_id` + `restore_at`): the nested-EXCEPT machinery
//!   deliberately keeps refcount 1 so it can mutate in place; re-interning
//!   its outputs would both add a probe per EXCEPT and guarantee the NEXT
//!   EXCEPT on the value is a forced copy-on-write. The
//!   `TY_INTERN_STATS=1` counters below measure the in-place vs forced-COW
//!   split so this choice stays evidence-based.
//! - Records built under [`super::super::InterningSkipGuard`] (symmetry
//!   canonicalization): throwaway permuted values, already memoized by
//!   allocation identity in `permute_memo`.
//! - Records with more than [`MAX_INTERN_RECORD_FIELDS`] fields (pin-size
//!   bound), empty records (`Vec::new()` does not allocate), and records
//!   whose additive fingerprint cannot be computed (lazy/closure children).
//!
//! # Soundness contract (read before changing)
//!
//! 1. **fp is a FILTER, never the decider.** The table maps the cached
//!    additive record fingerprint to one canonical `entries` allocation.
//!    On an fp hit, the canonical is returned ONLY after full structural
//!    equality of the entry vectors (`Vec<(NameId, Value)> ==`, exactly the
//!    relation `RecordValue::eq` uses). A 64-bit fp collision between
//!    DISTINCT records keeps both values alive and separate (first entry
//!    stays canonical; the new record is returned un-interned) — distinct
//!    values are NEVER merged.
//! 2. **Interned nodes are frozen by refcount gating.** The table pins each
//!    canonical allocation (holds one strong count), so every in-place
//!    mutation path (`Rp::get_mut` / `Rp::make_mut`, refcount-1-gated)
//!    copies-on-write instead of mutating a shared canonical. Content
//!    behind a table entry is therefore immutable for the entry's lifetime.
//! 3. **Equality is representation-converging by design.** Child values
//!    compare with `Value::eq`, which treats convergent representations
//!    (Seq ≡ Tuple ≡ IntFunc, Record ≡ string-keyed Func, Bag ≡ Func) as
//!    equal — so an intern hit may substitute an eq-equal alternative child
//!    representation, exactly like the existing set intern table. All
//!    order/hash/fingerprint consumers are converging across those
//!    representations (policed in-tree), so this is semantically
//!    transparent for checking.
//! 4. **NameId keys pin the name interner epoch.** Entries key fields by
//!    `NameId`, which is only stable within a run. Every reset point that
//!    clears the other value intern tables also calls
//!    [`clear_record_intern_table`], which bumps a process-global
//!    generation; threads whose thread-local table was not directly cleared
//!    (e.g. pool workers) self-clear on their next probe by comparing
//!    generations. A stale NameId can therefore never be observed.
//!
//! # Lifecycle
//!
//! Thread-local (the hot BFS path is the sequential checker thread; other
//! threads get independent tables — no cross-thread contention), size-capped
//! with clear-on-cap (pure cache: clearing is always sound and releases the
//! pins), cleared at run-reset boundaries alongside the set/int-func intern
//! tables.
//!
//! Kill switch: `TY_NO_RECORD_INTERN=1` disables interning entirely.
//! Stats: `TY_INTERN_STATS=1` prints probe/hit/miss/collision counters and
//! the record-mutation in-place vs forced-COW split at end of run.
//! Cap override: `TY_RECORD_INTERN_CAP=<entries>` (default 131072).

use super::RecordValue;
use crate::rp::Rp;
use crate::value::Value;
use rustc_hash::FxHashMap;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use tla_core::NameId;

type EntriesRp = Rp<Vec<(NameId, Value)>>;

/// Field-count cap: bounds the per-entry pin size and the worst-case
/// structural-equality compare on an fp hit. Spec records are small
/// (Paxos messages: 3-7 fields; PlusCal node records: <= ~10).
const MAX_INTERN_RECORD_FIELDS: usize = 16;

/// Per-thread intern table with a generation stamp (see soundness point 4).
struct InternTable {
    generation: u64,
    map: FxHashMap<u64, EntriesRp>,
}

thread_local! {
    static RECORD_INTERN: RefCell<InternTable> = RefCell::new(InternTable {
        generation: 0,
        map: FxHashMap::default(),
    });
    /// Thread-local skip flag, engaged by `InterningSkipGuard` around
    /// symmetry canonicalization (throwaway permuted values).
    static SKIP_RECORD_INTERNING: Cell<bool> = const { Cell::new(false) };
}

/// Process-global generation. Bumped by [`clear_record_intern_table`];
/// thread-local tables self-clear when their stamp lags.
static RECORD_INTERN_GEN: AtomicU64 = AtomicU64::new(0);

// --- Stats (TY_INTERN_STATS=1; counters untouched otherwise) ---
static STAT_PROBES: AtomicU64 = AtomicU64::new(0);
static STAT_HITS: AtomicU64 = AtomicU64::new(0);
static STAT_HITS_PTR: AtomicU64 = AtomicU64::new(0);
static STAT_CANON_SHARED_SKIPS: AtomicU64 = AtomicU64::new(0);
static STAT_MISSES: AtomicU64 = AtomicU64::new(0);
static STAT_COLLISIONS: AtomicU64 = AtomicU64::new(0);
static STAT_FP_ERRORS: AtomicU64 = AtomicU64::new(0);
static STAT_SKIP_LARGE: AtomicU64 = AtomicU64::new(0);
static STAT_CAP_CLEARS: AtomicU64 = AtomicU64::new(0);
/// Record mutation refcount-gate outcomes (record/mutation.rs), counted
/// regardless of the intern kill switch so on/off runs are comparable.
static STAT_MUT_INPLACE: AtomicU64 = AtomicU64::new(0);
static STAT_MUT_COW: AtomicU64 = AtomicU64::new(0);

/// Kill switch: `TY_NO_RECORD_INTERN=1` (cached).
#[inline(always)]
fn record_intern_disabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_NO_RECORD_INTERN").map_or(false, |v| !v.is_empty() && v != "0")
    })
}

/// Same-binary A/B switch for probing shared records during the post-EXCEPT
/// canonicalization walk. Production rejects them locally: mutation outputs
/// are unique by the COW contract, while shared path nodes cannot release
/// their backing allocation when only this handle is replaced.
#[inline(always)]
fn legacy_shared_record_canon_probes() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_LEGACY_SHARED_RECORD_CANON_PROBES")
            .map_or(false, |v| !v.is_empty() && v != "0")
    })
}

/// Whether intern-stat collection is enabled (`TY_INTERN_STATS=1`, cached).
#[inline(always)]
fn intern_stats_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_INTERN_STATS").map_or(false, |v| !v.is_empty() && v != "0")
    })
}

/// Entry cap (clear-on-cap). Each entry pins one small entries Vec plus its
/// child handles; the default bounds worst-case table memory to tens of MB.
fn intern_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("TY_RECORD_INTERN_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1 << 17)
    })
}

#[inline(always)]
fn stat(counter: &AtomicU64) {
    if intern_stats_enabled() {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Swap the thread-local skip flag, returning the previous value.
/// Used by the scoped [`crate::value::InterningSkipGuard`].
pub(in crate::value) fn replace_skip_record_interning(skip: bool) -> bool {
    SKIP_RECORD_INTERNING.with(|cell| cell.replace(skip))
}

/// Count a record-mutation refcount-gate outcome (`true` = mutated in
/// place, `false` = forced copy-on-write). Called from `record/mutation.rs`;
/// no-op unless `TY_INTERN_STATS=1`.
#[inline(always)]
pub(super) fn count_record_mut(in_place: bool) {
    if intern_stats_enabled() {
        let counter = if in_place {
            &STAT_MUT_INPLACE
        } else {
            &STAT_MUT_COW
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Canonicalize a freshly constructed record through the intern table.
///
/// Returns either `rec` itself (miss / skip / collision) or a `RecordValue`
/// sharing the canonical `entries` allocation of a structurally equal record
/// interned earlier on this thread (hit). In both cases the returned record
/// is `==` to `rec` and carries the additive fingerprint pre-cached when it
/// was computable.
///
/// `rec` must be freshly built (exclusively owned by the caller); the
/// constructors in `record/mod.rs` are the only callers.
pub(super) fn intern_record(rec: RecordValue) -> RecordValue {
    if record_intern_disabled() || rec.entries.is_empty() || SKIP_RECORD_INTERNING.with(Cell::get) {
        return rec;
    }
    if rec.entries.len() > MAX_INTERN_RECORD_FIELDS {
        stat(&STAT_SKIP_LARGE);
        return rec;
    }
    stat(&STAT_PROBES);

    // Fingerprint filter key: the cached additive record fp (mutation paths
    // maintain it incrementally, so EXCEPT-derived inputs are O(1) here).
    // The fresh instance is exclusively owned, so seeding its cache slot is
    // always safe regardless of the parallel read-only cache mode.
    let fp = match rec.get_additive_fp() {
        Some(fp) => fp,
        None => match crate::dedup_fingerprint::compute_record_additive_fp(&rec) {
            Ok(fp) => {
                rec.cache_additive_fp(fp);
                fp
            }
            Err(_) => {
                // Non-fingerprintable children (lazy/closure values): skip.
                stat(&STAT_FP_ERRORS);
                return rec;
            }
        },
    };

    let generation = RECORD_INTERN_GEN.load(Ordering::Relaxed);

    // Probe under a SHORT borrow, then compare WITHOUT holding it: deep
    // child equality can materialize values (lazy record-set children) and
    // re-enter this function, which takes its own borrow.
    let canonical = RECORD_INTERN.with(|t| {
        let mut t = t.borrow_mut();
        if t.generation != generation {
            // Stale epoch (run reset happened on another thread): NameId
            // keys and canonical values may be from a previous run.
            t.map.clear();
            t.generation = generation;
        }
        t.map.get(&fp).cloned()
    });

    match canonical {
        Some(canon) => {
            if Rp::ptr_eq(&canon, &rec.entries) {
                // Input already IS the canonical allocation.
                stat(&STAT_HITS_PTR);
                rec
            } else if *canon == *rec.entries {
                // DEDUP: drop the fresh allocation, share the canonical one.
                stat(&STAT_HITS);
                RecordValue {
                    entries: canon,
                    additive_fp: AtomicU64::new(fp),
                }
            } else {
                // 64-bit fp collision between DISTINCT records: NEVER merge.
                // First entry stays canonical; this value stays un-interned.
                stat(&STAT_COLLISIONS);
                rec
            }
        }
        None => {
            stat(&STAT_MISSES);
            RECORD_INTERN.with(|t| {
                let mut t = t.borrow_mut();
                if t.generation != generation {
                    t.map.clear();
                    t.generation = generation;
                }
                if t.map.len() >= intern_cap() {
                    // Pure cache: clearing is sound and releases all pins.
                    stat(&STAT_CAP_CLEARS);
                    t.map.clear();
                }
                t.map.insert(fp, rec.entries.clone());
            });
            rec
        }
    }
}

/// Canonicalize `rec` in place through the intern table: on a hit its
/// `entries` handle is swapped to the canonical allocation (content-equal by
/// the structural-equality decider, so every content-derived cache — the
/// instance additive fp, enclosing additive fps — remains valid); on a miss
/// `rec`'s allocation becomes the canonical one. No-op under the kill
/// switch / skip guard / size bounds.
///
/// `rec`'s ENTRIES allocation may be shared (`&mut` only proves the
/// instance is exclusive); swapping the handle never mutates the old
/// allocation.
pub(in crate::value) fn canonicalize_record_in_place(rec: &mut RecordValue) {
    // Cheap identity guard: nothing to do if a swap could not change anything.
    if rec.entries.is_empty() {
        return;
    }
    if !legacy_shared_record_canon_probes() && rec.is_storage_shared() {
        // EXCEPT mutation makes every changed record allocation unique before
        // writing. A shared node is therefore unchanged or already pinned by
        // the intern table; swapping this one handle cannot release its shared
        // allocation and is pure probe overhead.
        stat(&STAT_CANON_SHARED_SKIPS);
        return;
    }
    let canonical = intern_record(rec.clone());
    if let Some(fp) = canonical.get_additive_fp() {
        rec.cache_additive_fp(fp);
    }
    if !Rp::ptr_eq(&canonical.entries, &rec.entries) {
        rec.entries = canonical.entries;
    }
}

/// One element of a resolved EXCEPT path, borrowed for the post-EXCEPT
/// record canonicalization walk ([`canonicalize_records_along_path`]).
pub enum RecordCanonPathElem<'a> {
    /// Function/sequence/tuple subscript (`![idx]`), or a record subscript
    /// with a string index value.
    Index(&'a Value),
    /// Record field access (`!.field`).
    Field(NameId),
}

/// Post-EXCEPT record canonicalization (the mutation-path phase of record
/// hash-consing).
///
/// EXCEPT chains deliberately keep intermediate records uniquely owned so
/// field updates mutate in place; interning PER MUTATION would pin every
/// intermediate (forcing a copy-on-write per spec — the COW cascade). This
/// walker runs ONCE, after ALL specs of an EXCEPT expression are applied,
/// re-navigating a spec's resolved path and canonicalizing every RECORD on
/// it bottom-up (deepest first, so parent probes hit `ptr_eq` fast paths on
/// their already-canonical children).
///
/// Purely an optimization pass: every bail (shared storage, missing key,
/// non-record level, kill switch) simply skips canonicalization. Descent
/// uses UNIQUE-only mutable access (`Rp::get_mut` / `Arc::get_mut`), so the
/// walk never copies storage; replacement values are structurally EQUAL, so
/// all content-derived caches (additive fps of every enclosing value) stay
/// valid.
pub fn canonicalize_records_along_path(value: &mut Value, path: &[RecordCanonPathElem<'_>]) {
    canonicalize_records_along_paths(value, &[path]);
}

/// Post-EXCEPT record canonicalization for every modified path in one
/// bottom-up traversal.
///
/// Processing paths independently is not equivalent: canonicalizing a shared
/// parent after the first path pins its storage in the intern table, so the
/// unique-only descent required by a later sibling path can no longer enter
/// that parent. Grouping equal prefixes visits every modified child first and
/// canonicalizes each common parent exactly once.
pub fn canonicalize_records_along_paths(value: &mut Value, paths: &[&[RecordCanonPathElem<'_>]]) {
    if paths.is_empty() || record_intern_disabled() || SKIP_RECORD_INTERNING.with(Cell::get) {
        return;
    }
    canon_walk_many(value, paths);
}

fn canon_walk_many(value: &mut Value, paths: &[&[RecordCanonPathElem<'_>]]) {
    // EXCEPT spec counts are normally tiny. Keep prefix-group bookkeeping
    // inline while preserving the general case.
    let mut handled = smallvec::SmallVec::<[bool; 8]>::from_elem(false, paths.len());
    for i in 0..paths.len() {
        if handled[i] || paths[i].is_empty() {
            continue;
        }
        handled[i] = true;
        let first = &paths[i][0];
        let mut suffixes = smallvec::SmallVec::<[&[RecordCanonPathElem<'_>]; 8]>::new();
        suffixes.push(&paths[i][1..]);
        for j in (i + 1)..paths.len() {
            if !handled[j]
                && !paths[j].is_empty()
                && path_elems_match_at(value, first, &paths[j][0])
            {
                handled[j] = true;
                suffixes.push(&paths[j][1..]);
            }
        }

        match (&mut *value, first) {
            (Value::Record(r), RecordCanonPathElem::Field(field_id)) => {
                if let Some(child) = record_field_mut_unique(r, *field_id) {
                    canon_walk_many(child, &suffixes);
                }
            }
            // Record subscript with a string index: `![name]` on a record.
            (Value::Record(r), RecordCanonPathElem::Index(idx)) => {
                if let Some(field) = idx.as_string() {
                    let field_id = tla_core::intern_name(field);
                    if let Some(child) = record_field_mut_unique(r, field_id) {
                        canon_walk_many(child, &suffixes);
                    }
                }
            }
            (Value::Func(f), RecordCanonPathElem::Index(idx)) => {
                if let Some(fv) = Rp::get_mut(f) {
                    if let Some(child) = fv.mapping_get_mut_unique(idx) {
                        canon_walk_many(child, &suffixes);
                    }
                }
            }
            (Value::IntFunc(f), RecordCanonPathElem::Index(idx)) => {
                if let (Some(fv), Some(i)) = (Rp::get_mut(f), idx.as_i64()) {
                    if let Some(child) = fv.value_mut_unique(i) {
                        canon_walk_many(child, &suffixes);
                    }
                }
            }
            (Value::Tuple(t), RecordCanonPathElem::Index(idx)) => {
                if let (Some(slice), Some(i)) = (Rp::get_mut(t), idx.as_i64()) {
                    if i >= 1 {
                        if let Some(child) = slice.get_mut((i - 1) as usize) {
                            canon_walk_many(child, &suffixes);
                        }
                    }
                }
            }
            // Seq (im-Vector storage: get_mut may itself copy shared nodes)
            // and all other kinds: skip descent.
            _ => {}
        }
    }
    // Canonicalize the record at THIS level after its (path) children.
    if let Value::Record(r) = value {
        canonicalize_record_in_place(r);
    }
}

/// Whether two path elements select the same child at the current value.
/// Record field syntax (`!.x`) and string-index syntax (`!["x"]`) are the
/// same selector; other function-like values only group equal indices.
fn path_elems_match_at(
    value: &Value,
    a: &RecordCanonPathElem<'_>,
    b: &RecordCanonPathElem<'_>,
) -> bool {
    if matches!(value, Value::Record(_)) {
        let record_field = |elem: &RecordCanonPathElem<'_>| match elem {
            RecordCanonPathElem::Field(field) => Some(*field),
            RecordCanonPathElem::Index(index) => index.as_string().map(tla_core::intern_name),
        };
        return record_field(a)
            .zip(record_field(b))
            .is_some_and(|(a, b)| a == b);
    }
    matches!((a, b), (RecordCanonPathElem::Index(a), RecordCanonPathElem::Index(b)) if a == b)
        || matches!((a, b), (RecordCanonPathElem::Field(a), RecordCanonPathElem::Field(b)) if a == b)
}

/// Mutable access to a record field's value, ONLY when the entries storage
/// is uniquely owned (never copies). Callers must preserve content equality
/// (canonicalization-only), keeping the cached additive fp valid.
fn record_field_mut_unique(rec: &mut RecordValue, field: NameId) -> Option<&mut Value> {
    let entries = Rp::get_mut(&mut rec.entries)?;
    entries
        .iter_mut()
        .find(|(k, _)| *k == field)
        .map(|(_, v)| v)
}

/// Clear the record intern table (releases all pinned entry allocations).
///
/// Bumps the process-global generation so tables on OTHER threads (pool
/// workers) self-clear on their next probe, then clears the calling
/// thread's table immediately. Call between model checking runs alongside
/// `clear_set_intern_table` — record entries key fields by `NameId`, which
/// is only stable within a run.
pub fn clear_record_intern_table() {
    let generation = RECORD_INTERN_GEN.fetch_add(1, Ordering::Relaxed) + 1;
    RECORD_INTERN.with(|t| {
        let mut t = t.borrow_mut();
        t.map.clear();
        t.generation = generation;
    });
}

/// Print record-intern counters to stderr (`TY_INTERN_STATS=1`; no-op
/// otherwise). Counters are drained so repeated runs don't double-count.
pub fn print_record_intern_stats() {
    if !intern_stats_enabled() {
        return;
    }
    let probes = STAT_PROBES.swap(0, Ordering::Relaxed);
    let hits = STAT_HITS.swap(0, Ordering::Relaxed);
    let hits_ptr = STAT_HITS_PTR.swap(0, Ordering::Relaxed);
    let canon_shared_skips = STAT_CANON_SHARED_SKIPS.swap(0, Ordering::Relaxed);
    let misses = STAT_MISSES.swap(0, Ordering::Relaxed);
    let collisions = STAT_COLLISIONS.swap(0, Ordering::Relaxed);
    let fp_errors = STAT_FP_ERRORS.swap(0, Ordering::Relaxed);
    let skip_large = STAT_SKIP_LARGE.swap(0, Ordering::Relaxed);
    let cap_clears = STAT_CAP_CLEARS.swap(0, Ordering::Relaxed);
    let mut_inplace = STAT_MUT_INPLACE.swap(0, Ordering::Relaxed);
    let mut_cow = STAT_MUT_COW.swap(0, Ordering::Relaxed);
    if probes == 0 && mut_inplace == 0 && mut_cow == 0 {
        return;
    }
    let pct = |part: u64, whole: u64| {
        if whole == 0 {
            0.0
        } else {
            100.0 * part as f64 / whole as f64
        }
    };
    eprintln!("\n=== Record intern stats (TY_INTERN_STATS) ===");
    eprintln!("  probes                {probes:>12}");
    eprintln!(
        "  hits (dedup)          {hits:>12}  ({:.1}% of probes)",
        pct(hits, probes)
    );
    eprintln!("  hits (already canon)  {hits_ptr:>12}");
    eprintln!("  canon shared skips    {canon_shared_skips:>12}");
    eprintln!("  misses (inserted)     {misses:>12}");
    eprintln!("  fp collisions         {collisions:>12}  (distinct values kept separate)");
    eprintln!("  fp errors (skipped)   {fp_errors:>12}");
    eprintln!("  skipped >{MAX_INTERN_RECORD_FIELDS} fields    {skip_large:>12}");
    eprintln!("  cap clears            {cap_clears:>12}");
    eprintln!(
        "  record mut in-place   {mut_inplace:>12}  ({:.1}% of gated mutations)",
        pct(mut_inplace, mut_inplace + mut_cow)
    );
    eprintln!("  record mut forced-COW {mut_cow:>12}");
    eprintln!("=== end record intern stats ===");
}

#[cfg(test)]
mod tests {
    use super::super::RecordValue;
    use super::*;
    use crate::value::Value;

    fn rec(fields: &[(&str, i64)]) -> RecordValue {
        RecordValue::from_entries(
            fields
                .iter()
                .map(|(k, v)| (tla_core::intern_name(k), Value::SmallInt(*v)))
                .collect(),
        )
    }

    #[test]
    fn equal_records_share_canonical_allocation() {
        // Serialize against other intern-state-mutating tests.
        let _guard = crate::value::lock_intern_state();
        clear_record_intern_table();
        let a = rec(&[("ri_x", 1), ("ri_y", 2)]);
        let b = rec(&[("ri_x", 1), ("ri_y", 2)]);
        assert_eq!(a, b);
        if record_intern_disabled() {
            return; // kill switch set in this environment; nothing to assert
        }
        assert!(
            a.ptr_eq(&b),
            "structurally equal records must share the canonical allocation"
        );
        // The additive fp is pre-cached by the intern probe.
        assert!(a.get_additive_fp().is_some());
        assert_eq!(a.get_additive_fp(), b.get_additive_fp());
        clear_record_intern_table();
    }

    #[test]
    fn distinct_records_never_merge() {
        let _guard = crate::value::lock_intern_state();
        clear_record_intern_table();
        let a = rec(&[("ri_x", 1)]);
        let b = rec(&[("ri_x", 2)]);
        assert_ne!(a, b);
        assert!(!a.ptr_eq(&b));
        clear_record_intern_table();
    }

    #[test]
    fn mutation_on_interned_record_copies_on_write() {
        let _guard = crate::value::lock_intern_state();
        clear_record_intern_table();
        let a = rec(&[("ri_m", 1), ("ri_n", 2)]);
        let b = rec(&[("ri_m", 1), ("ri_n", 2)]);
        // Mutate b; a (and the canonical) must be unaffected.
        let b2 = b.update_existing_field_id(tla_core::intern_name("ri_n"), Value::SmallInt(9));
        assert_eq!(
            a.get_by_id(tla_core::intern_name("ri_n")),
            Some(&Value::SmallInt(2)),
            "canonical/interned sibling must be unchanged after COW mutation"
        );
        assert_eq!(
            b2.get_by_id(tla_core::intern_name("ri_n")),
            Some(&Value::SmallInt(9))
        );
        // A fresh construction of the ORIGINAL content still equals a.
        let c = rec(&[("ri_m", 1), ("ri_n", 2)]);
        assert_eq!(a, c);
        clear_record_intern_table();
    }

    #[test]
    fn skip_guard_disables_record_interning() {
        let _guard = crate::value::lock_intern_state();
        clear_record_intern_table();
        if record_intern_disabled() {
            return;
        }
        let a = rec(&[("ri_sg", 7)]);
        let b = {
            let _skip = crate::value::InterningSkipGuard::new();
            rec(&[("ri_sg", 7)])
        };
        assert_eq!(a, b);
        assert!(
            !a.ptr_eq(&b),
            "records built under InterningSkipGuard must not be interned"
        );
        // Guard dropped: interning resumes.
        let c = rec(&[("ri_sg", 7)]);
        assert!(a.ptr_eq(&c));
        clear_record_intern_table();
    }

    #[test]
    fn clear_bumps_generation_and_releases_pins() {
        let _guard = crate::value::lock_intern_state();
        clear_record_intern_table();
        if record_intern_disabled() {
            return;
        }
        let a = rec(&[("ri_g", 3)]);
        clear_record_intern_table();
        // After a clear, a fresh equal record becomes the NEW canonical
        // (no stale sharing with pre-clear allocations).
        let b = rec(&[("ri_g", 3)]);
        assert_eq!(a, b);
        assert!(
            !a.ptr_eq(&b),
            "post-clear construction must not alias the pre-clear table"
        );
        let c = rec(&[("ri_g", 3)]);
        assert!(b.ptr_eq(&c));
        clear_record_intern_table();
    }

    #[test]
    fn post_except_canon_skips_shared_nonmutation_storage() {
        let _guard = crate::value::lock_intern_state();
        clear_record_intern_table();
        if record_intern_disabled() || legacy_shared_record_canon_probes() {
            return;
        }

        let canonical = rec(&[("ri_shared", 4)]);
        let mut shared = {
            let _skip = crate::value::InterningSkipGuard::new();
            rec(&[("ri_shared", 4)])
        };
        let sibling = shared.clone();
        assert_eq!(canonical, shared);
        assert!(!canonical.ptr_eq(&shared));

        canonicalize_record_in_place(&mut shared);
        assert!(
            shared.ptr_eq(&sibling),
            "a shared, unmodified path node must avoid a table probe and handle swap"
        );
        assert!(!canonical.ptr_eq(&shared));
        clear_record_intern_table();
    }

    #[test]
    fn multi_path_walk_canonicalizes_all_siblings_before_parent() {
        let _guard = crate::value::lock_intern_state();
        clear_record_intern_table();
        if record_intern_disabled() {
            return;
        }

        let left = tla_core::intern_name("ri_left");
        let right = tla_core::intern_name("ri_right");
        let leaf = tla_core::intern_name("ri_leaf");
        let mut value = Value::Record(RecordValue::from_entries(vec![
            (
                left,
                Value::Record(RecordValue::from_entries(vec![(leaf, Value::SmallInt(1))])),
            ),
            (
                right,
                Value::Record(RecordValue::from_entries(vec![(leaf, Value::SmallInt(2))])),
            ),
        ]));
        // Release constructor-time pins. The outer record now uniquely owns
        // both child allocations, matching a completed multi-spec EXCEPT.
        clear_record_intern_table();

        let left_path = [
            RecordCanonPathElem::Field(left),
            RecordCanonPathElem::Field(leaf),
        ];
        let right_path = [
            RecordCanonPathElem::Field(right),
            RecordCanonPathElem::Field(leaf),
        ];
        canonicalize_records_along_paths(&mut value, &[&left_path, &right_path]);

        let Value::Record(outer) = &value else {
            panic!("expected outer record");
        };
        let left_record = outer.get_by_id(left).and_then(Value::as_record).unwrap();
        let right_record = outer.get_by_id(right).and_then(Value::as_record).unwrap();
        let equal_left = rec(&[("ri_leaf", 1)]);
        let equal_right = rec(&[("ri_leaf", 2)]);
        assert!(
            left_record.ptr_eq(&equal_left),
            "left sibling must be canonicalized"
        );
        assert!(
            right_record.ptr_eq(&equal_right),
            "right sibling must be canonicalized before the parent is pinned"
        );
        clear_record_intern_table();
    }

    #[test]
    fn multi_path_walk_groups_field_and_string_index_aliases() {
        let _guard = crate::value::lock_intern_state();
        clear_record_intern_table();
        if record_intern_disabled() {
            return;
        }

        let node = tla_core::intern_name("ri_node");
        let left = tla_core::intern_name("ri_alias_left");
        let right = tla_core::intern_name("ri_alias_right");
        let leaf = tla_core::intern_name("ri_alias_leaf");
        let child = RecordValue::from_entries(vec![
            (
                left,
                Value::Record(RecordValue::from_entries(vec![(leaf, Value::SmallInt(1))])),
            ),
            (
                right,
                Value::Record(RecordValue::from_entries(vec![(leaf, Value::SmallInt(2))])),
            ),
        ]);
        let mut value = Value::Record(RecordValue::from_entries(vec![(
            node,
            Value::Record(child),
        )]));
        clear_record_intern_table();

        let node_index = Value::String(crate::Rp::from("ri_node"));
        let left_path = [
            RecordCanonPathElem::Field(node),
            RecordCanonPathElem::Field(left),
            RecordCanonPathElem::Field(leaf),
        ];
        let right_path = [
            RecordCanonPathElem::Index(&node_index),
            RecordCanonPathElem::Field(right),
            RecordCanonPathElem::Field(leaf),
        ];
        canonicalize_records_along_paths(&mut value, &[&left_path, &right_path]);

        let Value::Record(outer) = &value else {
            panic!("expected outer record");
        };
        let child = outer.get_by_id(node).and_then(Value::as_record).unwrap();
        let left_record = child.get_by_id(left).and_then(Value::as_record).unwrap();
        let right_record = child.get_by_id(right).and_then(Value::as_record).unwrap();
        assert!(left_record.ptr_eq(&rec(&[("ri_alias_leaf", 1)])));
        assert!(right_record.ptr_eq(&rec(&[("ri_alias_leaf", 2)])));
        clear_record_intern_table();
    }
}
