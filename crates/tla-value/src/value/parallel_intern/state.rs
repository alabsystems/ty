// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared state, globals, and TLS worker overlay definitions for parallel interning.
//!
//! Part of #3412: extracted from `parallel_intern.rs` (lines 44-125).

use crate::rp::Rp as Arc;
use rustc_hash::FxHashMap;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::Mutex;

use super::super::Value;

/// Global flag: true when parallel interning is active.
/// Checked on the hot path with `Relaxed` ordering (~1ns atomic read).
pub(crate) static PARALLEL_INTERN_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Global flag: true when shared values may read cache fields but must not
/// write them back. Used to disable `AtomicU64` cache-line bouncing during
/// parallel BFS without changing fingerprint results.
pub(crate) static PARALLEL_READONLY_VALUE_CACHES_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Frozen snapshots of the optimization-only intern tables.
/// Immutable after creation; shared across workers via Arc.
pub(super) struct FrozenValueInterners {
    pub(super) sets: FxHashMap<u64, Arc<[Value]>>,
    pub(super) int_funcs: FxHashMap<u64, Arc<Vec<Value>>>,
    /// Part of #3285: Frozen snapshot of STRING_INTERN_TABLE.
    pub(super) strings: FxHashMap<String, Arc<str>>,
    /// Part of #3285: Frozen snapshot of TLC_STRING_TOKENS.
    pub(super) string_tokens: FxHashMap<Arc<str>, u32>,
}

/// Global storage for the frozen snapshot, accessible to workers during install.
pub(super) static FROZEN_SNAPSHOT: Mutex<Option<Arc<FrozenValueInterners>>> = Mutex::new(None);

/// Process-global lifecycle lock for a parallel value-intern run.
///
/// `FROZEN_SNAPSHOT` and the parallel intern mode flags describe a single
/// active run. Overlapping runs would otherwise let one guard's teardown clear
/// another run's snapshot before its workers install their thread-local scope.
pub(super) static PARALLEL_VALUE_INTERN_RUN_LOCK: Mutex<()> = Mutex::new(());

/// Worker-local token counter for TLC string tokens.
/// Each worker assigns tokens from a high range that won't collide with the frozen
/// snapshot tokens. Workers start at the frozen counter value and increment locally.
/// Token ordering consistency across workers is not required because TLC_STRING_TOKENS
/// is an append-only registry and worker-local tokens are only for new strings
/// first seen during parallel BFS.
pub(super) static WORKER_TOKEN_COUNTER: AtomicU32 = AtomicU32::new(1);

/// Memory cap (in entries) on the per-worker `set_overlay` and `int_func_overlay`
/// maps' growth beyond their preloaded frozen baseline.
///
/// # Why this is the SOUND family to cap (audit #7 OOM fix)
///
/// The `set_overlay` / `int_func_overlay` "token" is the `Arc` *pointer*. It is
/// used by `Value::eq` / `Value::cmp` only as a fast-path shortcut (`Arc::ptr_eq`)
/// that always falls through to full **content** comparison when pointers differ
/// (`cmp_helpers/equality.rs`, `cmp_helpers/same_type.rs`). State fingerprints
/// hash element **content**, never the `Arc` pointer (`fingerprint.rs`). The
/// lookup helpers already mint a *fresh* `Arc` on every cache miss
/// (`lookups.rs` set/int-func miss arms). Therefore evicting an overlay entry and
/// re-interning the same value later yields a *different* `Arc` that still
/// compares equal and fingerprints identically — eviction here is SOUND and
/// cannot change a verdict.
///
/// The string overlays are deliberately NOT capped here: the paired
/// `string_token_overlay` mints a monotonic `u32` ordinal via `fetch_add`, and
/// re-minting after eviction would assign a *higher* ordinal, which can reorder
/// `\prec`/sorted-set results within a worker (a string-comparison-order bug, not
/// a dedup/equality break). String overlays grow only with *distinct* string
/// content, which is bounded for realistic specs, so they are left uncapped.
///
/// Defaults to 1_000_000 new entries (≈ tens of MB of small `Arc`s per worker);
/// override via `TY_PARALLEL_OVERLAY_CAP`.
pub(super) const DEFAULT_WORKER_OVERLAY_CAP: usize = 1_000_000;

/// Resolve the per-worker set/int-func overlay growth cap.
///
/// Reads `TY_PARALLEL_OVERLAY_CAP` once (cached). A value of `0` disables the
/// cap (treated as `usize::MAX`), preserving the pre-fix unbounded behavior for
/// A/B debugging.
pub(super) fn worker_overlay_cap() -> usize {
    use std::sync::OnceLock;
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| match std::env::var("TY_PARALLEL_OVERLAY_CAP") {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(0) => usize::MAX,
            Ok(n) => n,
            Err(_) => DEFAULT_WORKER_OVERLAY_CAP,
        },
        Err(_) => DEFAULT_WORKER_OVERLAY_CAP,
    })
}

/// Per-worker intern attribution counters.
///
/// Part of #3285: tracks where intern lookups resolve (frozen vs overlay vs new insert)
/// to determine whether remaining eval overhead is in the interner or elsewhere.
/// Counters are `u64` to avoid overflow on large specs (MCKVSSafetySmall: 56M transitions).
#[derive(Debug, Clone, Default)]
pub struct InternAttributionCounters {
    /// String lookups resolved in the frozen (shared, read-only) snapshot.
    pub frozen_string_hits: u64,
    /// TLC string-token lookups resolved in the frozen snapshot.
    pub frozen_token_hits: u64,
    /// Set lookups resolved in the frozen snapshot.
    pub frozen_set_hits: u64,
    /// Int-function lookups resolved in the frozen snapshot.
    pub frozen_int_func_hits: u64,
    /// String lookups resolved in the worker-local overlay.
    pub overlay_string_hits: u64,
    /// TLC string-token lookups resolved in the worker-local overlay.
    pub overlay_token_hits: u64,
    /// Set lookups resolved in the worker-local overlay.
    pub overlay_set_hits: u64,
    /// Int-function lookups resolved in the worker-local overlay.
    pub overlay_int_func_hits: u64,
    /// New string entries inserted (missed both frozen snapshot and overlay).
    pub new_string_inserts: u64,
    /// New set entries inserted (missed both frozen snapshot and overlay).
    pub new_set_inserts: u64,
    /// New int-function entries inserted (missed both frozen snapshot and overlay).
    pub new_int_func_inserts: u64,
}

/// Per-worker interning state: frozen snapshot reference + local overlay maps.
///
/// Attribution counters use `Cell<u64>` so they can be incremented during shared
/// borrows on the hot path (overlay/frozen hit returns) without restructuring
/// the borrow pattern. Zero runtime overhead since `Cell` is transparent for
/// `Copy` types.
pub(super) struct WorkerInternState {
    pub(super) frozen: Arc<FrozenValueInterners>,
    pub(super) set_overlay: FxHashMap<u64, Arc<[Value]>>,
    pub(super) int_func_overlay: FxHashMap<u64, Arc<Vec<Value>>>,
    /// Preloaded baseline sizes of the set/int-func overlays (the frozen-snapshot
    /// entries copied in at install). The growth cap (`worker_overlay_cap()`) is
    /// applied to entries added *beyond* this baseline so cliff-clears bound the
    /// worker-local growth without churning the always-present preloaded frozen
    /// entries. On a cliff-clear the overlay is reset to just the frozen entries
    /// it can still serve from `frozen` (we drop the whole overlay and let misses
    /// re-resolve against `frozen`), so these baselines also reset to 0.
    pub(super) set_overlay_base: usize,
    pub(super) int_func_overlay_base: usize,
    /// Part of #3285: Worker-local overlay for string interning.
    pub(super) string_overlay: FxHashMap<String, Arc<str>>,
    /// Part of #3285: Worker-local overlay for TLC string tokens.
    pub(super) string_token_overlay: FxHashMap<Arc<str>, u32>,
    // Part of #3285: Attribution counters (Cell for increment during shared borrow)
    pub(super) frozen_string_hits: Cell<u64>,
    pub(super) frozen_token_hits: Cell<u64>,
    pub(super) frozen_set_hits: Cell<u64>,
    pub(super) frozen_int_func_hits: Cell<u64>,
    pub(super) overlay_string_hits: Cell<u64>,
    pub(super) overlay_token_hits: Cell<u64>,
    pub(super) overlay_set_hits: Cell<u64>,
    pub(super) overlay_int_func_hits: Cell<u64>,
    pub(super) new_string_inserts: Cell<u64>,
    pub(super) new_set_inserts: Cell<u64>,
    pub(super) new_int_func_inserts: Cell<u64>,
}

thread_local! {
    pub(super) static WORKER_INTERN: RefCell<Option<WorkerInternState>> = const { RefCell::new(None) };
}
