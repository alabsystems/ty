// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Diagnostic counters for the linear/chain-recursive operator memo
//! (`helpers::apply::try_eval_recursive_memoized`).
//!
//! Enabled with `TY_REC_STATS=1`. Measures the redundancy of recursive-operator
//! evaluation (e.g. NanoBlockchain's `PublicKeyOf` re-walking the ledger chain):
//!
//! * `apply`    — every application of a RECURSIVE operator that reached the memo
//!                (counted BEFORE the kill switch, so it reflects total recursion
//!                work even when the memo is disabled).
//! * `hit`      — served from the memo (recomputation avoided).
//! * `compute`  — memo miss: the body was actually evaluated (distinct work).
//! * `cached`   — computed results proven state-independent and stored.
//! * `impure`   — computed results that read state (not stored — sound fallback).
//!
//! Redundancy factor ≈ `apply(memo off) / compute(memo on)`: the O(D²)→O(D)
//! collapse. Printed by `print_eval_profile_stats()` at end of model checking.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub(crate) enum RecSite {
    Apply = 0,
    Hit = 1,
    Compute = 2,
    Cached = 3,
    Impure = 4,
}

const NUM_SITES: usize = 5;
static COUNTERS: [AtomicU64; NUM_SITES] = [const { AtomicU64::new(0) }; NUM_SITES];
const SITE_NAMES: [&str; NUM_SITES] = [
    "recursive-op applies (total recursion work)",
    "memo hits (recomputation avoided)",
    "memo misses / body computed (distinct work)",
    "computed + cached (state-independent)",
    "computed + NOT cached (reads state)",
];

/// Whether recursion-memo stat collection is enabled (`TY_REC_STATS`, cached).
#[inline(always)]
pub(crate) fn rec_stats_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TY_REC_STATS").map_or(false, |v| !v.is_empty() && v != "0"))
}

#[inline(always)]
pub(crate) fn rec_count(site: RecSite) {
    if rec_stats_enabled() {
        COUNTERS[site as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Print the recursion-memo counters to stderr (no-op when disabled or all-zero).
/// Counters are drained so repeated in-process runs do not double-count.
pub fn print_recursion_memo_stats() {
    if !rec_stats_enabled() {
        return;
    }
    let vals: Vec<u64> = COUNTERS
        .iter()
        .map(|c| c.swap(0, Ordering::Relaxed))
        .collect();
    if vals.iter().all(|v| *v == 0) {
        return;
    }
    eprintln!("\n=== Recursive-operator memo stats (TY_REC_STATS) ===");
    for (val, name) in vals.iter().zip(SITE_NAMES.iter()) {
        eprintln!("  {val:>14}  {name}");
    }
    let applies = vals[RecSite::Apply as usize];
    let compute = vals[RecSite::Compute as usize];
    if compute > 0 {
        eprintln!(
            "  redundancy factor (applies / distinct computes) = {:.2}x",
            applies as f64 / compute as f64
        );
    }
    eprintln!("=== end recursion memo stats ===");
}
