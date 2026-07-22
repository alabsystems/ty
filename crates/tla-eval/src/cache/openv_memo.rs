// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Memoization of per-state `OpEnv` (LET operator environment) churn.
//!
//! Profiling btree attributed the dominant tla-im HAMT heat to ONE structure:
//! the `OpEnv = HashMap<String, Arc<OperatorDef>>` rebuilt by the LET
//! machinery. Two distinct per-state wastes:
//!
//! 1. **Merged-env rebuild** ([`merged_let_env_memoized`]): every LET entry
//!    (successor enumeration `conjunct_let`, `eval_let`'s general path, guard
//!    checking) clones the ambient `OpEnv` HAMT, re-inserts the *static* LET
//!    defs, wraps a fresh `Arc`, and re-derives the scope id (a full HAMT walk
//!    with per-def `intern_name` calls) — all of it identical on every state.
//! 2. **Scope-id re-derivation for an existing Arc**
//!    ([`scope_id_and_recursive_memoized`]): closure/lazy-func forcing and
//!    LET scope save/restore recompute `(scope_id, requires_arc_identity)` for
//!    an *unchanged* `Arc<OpEnv>` — a full `ops.values().any(..)` HAMT walk
//!    (plus a content-fingerprint walk) per call.
//!
//! # Soundness: pointer keys + pinning
//!
//! Both memos key on `Arc` pointer identity and **pin** every keyed `Arc`
//! inside the memo entry (a stored clone). Pinning gives two guarantees:
//!
//! - **No ABA**: while an entry lives, its pinned allocations cannot be freed,
//!   so the allocator can never recycle a keyed address for a different
//!   `OpEnv`/`OperatorDef`. Pointer equality on lookup therefore implies *the
//!   same allocation*.
//! - **No in-place mutation**: production code never mutates an `Arc<OpEnv>`'s
//!   contents in place (audited: `local_ops_mut()` only reassigns the `Option`,
//!   and no `Arc::make_mut`/`get_mut` targets an `OpEnv`). Even if a future
//!   call site used `Arc::make_mut`, the memo's pin keeps the refcount >= 2,
//!   forcing copy-on-write to a NEW allocation — the stored entry keeps
//!   describing the old (still correct) allocation.
//!
//! Together: same pointers => same immutable content => the memoized result is
//! exactly what recomputation would produce. No content hashing is involved in
//! the keys, so there is no collision risk.
//!
//! **Recursive scopes are never memoized by the merged-env memo.** A merged
//! env containing a recursive operator def must present a fresh `Arc` identity
//! per LET entry (each recursion frame captures different bindings; sharing a
//! scope id across frames would alias `SUBST_CACHE`/`NARY_OP_CACHE` entries —
//! the #3156 bug class). Such calls fall through to the unmemoized build path,
//! byte-for-byte identical to the previous behavior.
//!
//! The scope-id memo (#2) IS sound for recursive envs: for those the scope id
//! *is* the Arc pointer, i.e. a pure function of the (pinned) key.
//!
//! # Lifecycle
//!
//! Thread-local (each worker memoizes independently). Cleared on run/test
//! reset via [`clear_openv_memos`] (wired into `clear_run_reset_impl`).
//! Clearing at ANY point is sound: a later rebuild yields a fresh Arc whose
//! content-based scope id (non-recursive case) is value-identical, and
//! downstream pointer-keyed consumers simply miss and recompute.
//!
//! Set `TY_NO_OPENV_MEMO=1` to disable both memos (bypass to the original
//! build/walk paths, e.g. for A/B soundness or perf attribution).
//! Set `TY_OPENV_STATS=1` to print per-site call/hit counters at end of run.

use crate::core::OpEnv;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tla_core::ast::OperatorDef;

use super::scope_ids::compute_local_ops_scope_id_and_recursive;

// ===========================================================================
// Env-gated switches
// ===========================================================================

/// `TY_NO_OPENV_MEMO=1` bypasses both memos (original build/walk behavior).
#[inline]
pub(crate) fn openv_memo_disabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("TY_NO_OPENV_MEMO").is_some())
}

/// `TY_OPENV_STATS=1` enables counters + end-of-run stats print.
#[inline]
fn stats_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("TY_OPENV_STATS").is_some())
}

// ===========================================================================
// Counters (process-global; summed across worker threads)
// ===========================================================================

/// Call-site tags for the merged-env memo (counter attribution only).
#[derive(Clone, Copy)]
pub enum MergedLetSite {
    /// `eval_let` general (arity>0 / primed / instance) path.
    EvalLet = 0,
    /// Successor enumeration `conjunct_let` (unified_scope.rs).
    EnumConjunct = 1,
    /// Successor enumeration LET dispatch (unified_dispatch.rs).
    EnumDispatch = 2,
    /// Action guard pre-check (guard_check.rs).
    GuardCheck = 3,
}
const MERGED_SITES: usize = 4;
static MERGED_SITE_NAMES: [&str; MERGED_SITES] =
    ["eval_let", "enum_conjunct", "enum_dispatch", "guard_check"];

/// Call-site tags for the scope-id memo (counter attribution only).
#[derive(Clone, Copy)]
pub(crate) enum ScopeIdSite {
    /// `build_closure_ctx` (thunk/closure force).
    Closure = 0,
    /// `build_lazy_func_ctx` (recursive function force).
    LazyFunc = 1,
    /// `restore_local_ops_with_id` (LET/INSTANCE scope exit).
    Restore = 2,
    /// `set_local_ops_with_id` (scope install with precomputed id).
    SetWithId = 3,
    /// `enter_let_scope` (legacy premerge entry).
    EnterLet = 4,
    /// Zero-arg cache revalidation (eval_cache_lifecycle.rs).
    Lifecycle = 5,
}
const SCOPE_SITES: usize = 6;
static SCOPE_SITE_NAMES: [&str; SCOPE_SITES] = [
    "closure_ctx",
    "lazy_func_ctx",
    "restore",
    "set_with_id",
    "enter_let",
    "zero_arg_revalidate",
];

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);

static MERGED_CALLS: [AtomicU64; MERGED_SITES] = [ZERO; MERGED_SITES];
static MERGED_HITS: [AtomicU64; MERGED_SITES] = [ZERO; MERGED_SITES];
static MERGED_RECURSIVE_BUILDS: [AtomicU64; MERGED_SITES] = [ZERO; MERGED_SITES];
/// Total entries in ambient envs cloned on (miss) builds — HAMT clone cost proxy.
static MERGED_AMBIENT_ENTRIES_CLONED: AtomicU64 = AtomicU64::new(0);

static SCOPE_CALLS: [AtomicU64; SCOPE_SITES] = [ZERO; SCOPE_SITES];
static SCOPE_HITS: [AtomicU64; SCOPE_SITES] = [ZERO; SCOPE_SITES];

/// Print `TY_OPENV_STATS=1` counters (no-op otherwise). Called from
/// `print_eval_profile_stats` at end of model checking.
pub fn print_openv_memo_stats() {
    if !stats_enabled() {
        return;
    }
    eprintln!("\n=== OpEnv memo stats (TY_OPENV_STATS) ===");
    if openv_memo_disabled() {
        eprintln!("  [memo DISABLED via TY_NO_OPENV_MEMO — calls = unmemoized builds/walks]");
    }
    eprintln!("  merged LET env memo (build = HAMT clone + inserts + Arc + id walk):");
    for i in 0..MERGED_SITES {
        let calls = MERGED_CALLS[i].load(Ordering::Relaxed);
        if calls == 0 {
            continue;
        }
        let hits = MERGED_HITS[i].load(Ordering::Relaxed);
        let rec = MERGED_RECURSIVE_BUILDS[i].load(Ordering::Relaxed);
        eprintln!(
            "    {:<16} calls {:>12}  hits {:>12}  builds {:>12}  (recursive-unmemoizable {})",
            MERGED_SITE_NAMES[i],
            calls,
            hits,
            calls - hits,
            rec
        );
    }
    eprintln!(
        "    ambient HAMT entries cloned on builds: {}",
        MERGED_AMBIENT_ENTRIES_CLONED.load(Ordering::Relaxed)
    );
    eprintln!("  scope-id memo (walk = ops.values().any + content fingerprint):");
    for i in 0..SCOPE_SITES {
        let calls = SCOPE_CALLS[i].load(Ordering::Relaxed);
        if calls == 0 {
            continue;
        }
        let hits = SCOPE_HITS[i].load(Ordering::Relaxed);
        eprintln!(
            "    {:<20} calls {:>12}  hits {:>12}  walks {:>12}",
            SCOPE_SITE_NAMES[i],
            calls,
            hits,
            calls - hits
        );
    }
}

// ===========================================================================
// Memo 1: merged LET operator environment
// ===========================================================================

/// Exact identity key for a merged LET env: the ambient env allocation plus
/// the exact interned def allocations inserted, in insertion order. All keyed
/// allocations are pinned by the entry, so pointer equality implies content
/// identity (see module docs).
#[derive(PartialEq, Eq, Hash)]
struct MergedKey {
    /// `Arc::as_ptr` of the ambient `local_ops` env; 0 when `None`.
    ambient_ptr: usize,
    /// `Arc::as_ptr` of each inserted (interned) `OperatorDef`, in order.
    def_ptrs: SmallVec<[usize; 8]>,
}

struct MergedEntry {
    env: Arc<OpEnv>,
    scope_id: u64,
    /// Pin: keeps the keyed ambient allocation alive (no ABA, no in-place COW).
    _ambient_pin: Option<Arc<OpEnv>>,
    /// Pin: keeps the keyed def allocations alive.
    _def_pins: SmallVec<[Arc<OperatorDef>; 8]>,
}

thread_local! {
    static MERGED_LET_ENV_MEMO: RefCell<FxHashMap<MergedKey, MergedEntry>> =
        RefCell::new(FxHashMap::default());
}

/// Cap on merged-env memo entries. Entries are O(LET sites x ambient scopes)
/// in practice (ambient Arcs are themselves memo-stable), so this is
/// effectively never hit; it bounds memory for pathological workloads.
const MERGED_LET_ENV_MEMO_CAP: usize = 16_384;

/// Build (or reuse) the merged `local_ops` env for a LET entry.
///
/// `defs` are the static LET definitions; `insert_def` selects which defs the
/// call site actually inserts (all of them for enumeration; a static subset
/// for `eval_let`). Returns `(merged_env, scope_id, requires_arc_identity)`
/// where `scope_id` is value-identical to
/// `compute_local_ops_scope_id(&merged_env)`.
///
/// On a memo hit the SAME `Arc` is returned every call — downstream
/// pointer-keyed consumers (closure scope-id memo) become hits too.
/// Recursive merged envs are rebuilt fresh per call (never memoized) to
/// preserve per-entry Arc identity for `SUBST_CACHE` keying.
pub fn merged_let_env_memoized(
    ambient: Option<&Arc<OpEnv>>,
    defs: &[OperatorDef],
    site: MergedLetSite,
    mut insert_def: impl FnMut(&OperatorDef) -> bool,
) -> (Arc<OpEnv>, u64, bool) {
    let stats = stats_enabled();
    if stats {
        MERGED_CALLS[site as usize].fetch_add(1, Ordering::Relaxed);
    }

    // Intern the inserted defs (run-stable Arcs; also what the unmemoized
    // build path inserts — see cache/let_def_interning.rs).
    let interned: SmallVec<[Arc<OperatorDef>; 8]> = defs
        .iter()
        .filter(|def| insert_def(def))
        .map(super::intern_let_def_arc)
        .collect();

    let build = |interned: &SmallVec<[Arc<OperatorDef>; 8]>| -> (Arc<OpEnv>, u64, bool) {
        let mut merged: OpEnv = ambient.map(|o| (**o).clone()).unwrap_or_default();
        if stats {
            if let Some(o) = ambient {
                MERGED_AMBIENT_ENTRIES_CLONED.fetch_add(o.len() as u64, Ordering::Relaxed);
            }
        }
        for def in interned {
            merged.insert(def.name.node.clone(), Arc::clone(def));
        }
        let merged = Arc::new(merged);
        let (id, recursive) = compute_local_ops_scope_id_and_recursive(&merged);
        (merged, id, recursive)
    };

    if openv_memo_disabled() {
        return build(&interned);
    }

    let key = MergedKey {
        ambient_ptr: ambient.map_or(0, |o| Arc::as_ptr(o) as usize),
        def_ptrs: interned.iter().map(|d| Arc::as_ptr(d) as usize).collect(),
    };

    MERGED_LET_ENV_MEMO.with(|memo| {
        if let Some(entry) = memo.borrow().get(&key) {
            if stats {
                MERGED_HITS[site as usize].fetch_add(1, Ordering::Relaxed);
            }
            return (Arc::clone(&entry.env), entry.scope_id, false);
        }
        let (env, id, recursive) = build(&interned);
        if recursive {
            // Never memoize: each LET entry must present a fresh Arc identity
            // (per-recursion-frame scope ids; see module docs / #3156).
            if stats {
                MERGED_RECURSIVE_BUILDS[site as usize].fetch_add(1, Ordering::Relaxed);
            }
            return (env, id, true);
        }
        let mut m = memo.borrow_mut();
        if m.len() >= MERGED_LET_ENV_MEMO_CAP {
            m.clear();
        }
        m.insert(
            key,
            MergedEntry {
                env: Arc::clone(&env),
                scope_id: id,
                _ambient_pin: ambient.map(Arc::clone),
                _def_pins: interned,
            },
        );
        (env, id, false)
    })
}

// ===========================================================================
// Memo 2: (scope_id, requires_arc_identity) for an existing Arc<OpEnv>
// ===========================================================================

struct ScopeIdEntry {
    scope_id: u64,
    recursive: bool,
    /// Pin: keeps the keyed allocation alive (no ABA, no in-place COW).
    _pin: Arc<OpEnv>,
}

thread_local! {
    static OPENV_SCOPE_ID_MEMO: RefCell<FxHashMap<usize, ScopeIdEntry>> =
        RefCell::new(FxHashMap::default());
}

/// Cap on scope-id memo entries. Recursive-LET frames insert one entry per
/// fresh Arc; the cap bounds both map size and pinned-env memory (structural
/// sharing keeps each pinned env's incremental footprint small).
const OPENV_SCOPE_ID_MEMO_CAP: usize = 8_192;

/// Memoized `compute_local_ops_scope_id_and_recursive` for an existing Arc.
///
/// The result is a pure function of the allocation behind `ops` (content for
/// non-recursive envs; the pointer itself for recursive envs), and the entry
/// pins the allocation, so a pointer hit returns exactly what recomputation
/// would.
pub(crate) fn scope_id_and_recursive_memoized(ops: &Arc<OpEnv>, site: ScopeIdSite) -> (u64, bool) {
    let stats = stats_enabled();
    if stats {
        SCOPE_CALLS[site as usize].fetch_add(1, Ordering::Relaxed);
    }
    if openv_memo_disabled() {
        return compute_local_ops_scope_id_and_recursive(ops);
    }
    let key = Arc::as_ptr(ops) as usize;
    OPENV_SCOPE_ID_MEMO.with(|memo| {
        if let Some(entry) = memo.borrow().get(&key) {
            if stats {
                SCOPE_HITS[site as usize].fetch_add(1, Ordering::Relaxed);
            }
            return (entry.scope_id, entry.recursive);
        }
        let (id, recursive) = compute_local_ops_scope_id_and_recursive(ops);
        let mut m = memo.borrow_mut();
        if m.len() >= OPENV_SCOPE_ID_MEMO_CAP {
            m.clear();
        }
        m.insert(
            key,
            ScopeIdEntry {
                scope_id: id,
                recursive,
                _pin: Arc::clone(ops),
            },
        );
        (id, recursive)
    })
}

/// Memoized `local_ops_requires_arc_identity` for an optional scope.
/// Shares the scope-id memo (the flag rides along with the id).
#[inline]
pub(crate) fn recursive_flag_memoized(ops: &Option<Arc<OpEnv>>, site: ScopeIdSite) -> bool {
    match ops {
        None => false,
        Some(ops) => scope_id_and_recursive_memoized(ops, site).1,
    }
}

/// Clear both memos (run/test reset). Dropping entries drops their pins in the
/// same breath, so no entry can ever outlive the allocations it keys on.
/// Clearing is sound at any point: later calls rebuild/recompute fresh.
pub fn clear_openv_memos() {
    MERGED_LET_ENV_MEMO.with(|memo| memo.borrow_mut().clear());
    OPENV_SCOPE_ID_MEMO.with(|memo| memo.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::scope_ids::compute_local_ops_scope_id;
    use tla_core::ast::Expr;
    use tla_core::Spanned;

    fn mk_def(name: &str, start: u32, end: u32, recursive: bool) -> OperatorDef {
        let fid = tla_core::span::FileId(0);
        OperatorDef {
            name: Spanned::new(name.into(), tla_core::Span::new(fid, start, end)),
            params: vec![],
            body: Spanned::new(Expr::Bool(true), tla_core::Span::new(fid, start, end)),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            is_recursive: recursive,
            has_primed_param: false,
            self_call_count: 0,
        }
    }

    #[test]
    fn merged_memo_returns_same_arc_and_oracle_id() {
        clear_openv_memos();
        super::super::clear_let_def_interning();
        let defs = vec![mk_def("a", 10, 14, false), mk_def("b", 20, 24, false)];
        let (e1, id1, r1) =
            merged_let_env_memoized(None, &defs, MergedLetSite::EvalLet, |_| true);
        let (e2, id2, r2) =
            merged_let_env_memoized(None, &defs, MergedLetSite::EvalLet, |_| true);
        assert!(Arc::ptr_eq(&e1, &e2), "hit must return the same Arc");
        assert_eq!(id1, id2);
        assert!(!r1 && !r2);
        // Oracle: id must equal direct computation on the merged env.
        assert_eq!(id1, compute_local_ops_scope_id(&e1));
        assert_eq!(e1.len(), 2);
        assert!(e1.get("a").is_some() && e1.get("b").is_some());
        clear_openv_memos();
        super::super::clear_let_def_interning();
    }

    #[test]
    fn merged_memo_respects_filter_and_ambient() {
        clear_openv_memos();
        super::super::clear_let_def_interning();
        let mut ambient = OpEnv::default();
        ambient.insert("outer".into(), Arc::new(mk_def("outer", 1, 5, false)));
        let ambient = Arc::new(ambient);
        let defs = vec![mk_def("a", 10, 14, false), mk_def("b", 20, 24, false)];
        // Filter inserts only "a".
        let (env, id, _) = merged_let_env_memoized(
            Some(&ambient),
            &defs,
            MergedLetSite::EvalLet,
            |d| d.name.node == "a",
        );
        assert_eq!(env.len(), 2, "ambient + a");
        assert!(env.get("outer").is_some() && env.get("a").is_some());
        assert!(env.get("b").is_none());
        // Different filter (all defs) at same ambient => different entry/content.
        let (env2, id2, _) =
            merged_let_env_memoized(Some(&ambient), &defs, MergedLetSite::EvalLet, |_| true);
        assert_eq!(env2.len(), 3);
        assert_ne!(id, id2);
        clear_openv_memos();
        super::super::clear_let_def_interning();
    }

    #[test]
    fn merged_memo_never_memoizes_recursive_envs() {
        clear_openv_memos();
        super::super::clear_let_def_interning();
        let defs = vec![mk_def("rec", 30, 34, true)];
        let (e1, id1, r1) =
            merged_let_env_memoized(None, &defs, MergedLetSite::EnumConjunct, |_| true);
        let (e2, id2, r2) =
            merged_let_env_memoized(None, &defs, MergedLetSite::EnumConjunct, |_| true);
        assert!(r1 && r2, "recursive merged env must report arc-identity");
        assert!(
            !Arc::ptr_eq(&e1, &e2),
            "recursive merged env must be fresh per call"
        );
        // Arc-identity scope ids: distinct per allocation.
        assert_ne!(id1, id2);
        assert_eq!(id1, Arc::as_ptr(&e1) as usize as u64);
        assert_eq!(id2, Arc::as_ptr(&e2) as usize as u64);
        clear_openv_memos();
        super::super::clear_let_def_interning();
    }

    #[test]
    fn scope_id_memo_matches_oracle_both_kinds() {
        clear_openv_memos();
        let mut nonrec = OpEnv::default();
        nonrec.insert("x".into(), Arc::new(mk_def("x", 40, 44, false)));
        let nonrec = Arc::new(nonrec);
        let oracle = compute_local_ops_scope_id_and_recursive(&nonrec);
        assert_eq!(
            scope_id_and_recursive_memoized(&nonrec, ScopeIdSite::Closure),
            oracle
        );
        // Hit path returns the same.
        assert_eq!(
            scope_id_and_recursive_memoized(&nonrec, ScopeIdSite::Closure),
            oracle
        );

        let mut rec = OpEnv::default();
        rec.insert("r".into(), Arc::new(mk_def("r", 50, 54, true)));
        let rec = Arc::new(rec);
        let oracle_rec = compute_local_ops_scope_id_and_recursive(&rec);
        assert_eq!(
            scope_id_and_recursive_memoized(&rec, ScopeIdSite::Restore),
            oracle_rec
        );
        assert_eq!(
            scope_id_and_recursive_memoized(&rec, ScopeIdSite::Restore),
            oracle_rec
        );
        assert!(oracle_rec.1);
        clear_openv_memos();
    }

    #[test]
    fn recursive_flag_memoized_matches_walking_flag() {
        clear_openv_memos();
        assert!(!recursive_flag_memoized(&None, ScopeIdSite::Restore));
        let mut rec = OpEnv::default();
        rec.insert("r".into(), Arc::new(mk_def("r", 60, 64, true)));
        let rec = Some(Arc::new(rec));
        assert!(recursive_flag_memoized(&rec, ScopeIdSite::Restore));
        assert_eq!(
            recursive_flag_memoized(&rec, ScopeIdSite::Restore),
            crate::cache::scope_ids::local_ops_recursive_flag(&rec)
        );
        clear_openv_memos();
    }
}
