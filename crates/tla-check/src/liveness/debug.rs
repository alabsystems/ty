// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Debug and profiling flags for liveness checking.

use std::sync::OnceLock;

// feature_flag! macro defined in crate::debug_env (via #[macro_use])
feature_flag!(pub(crate) liveness_profile, "TY_LIVENESS_PROFILE");
// Part of #3746: When set, liveness checking panics on missing states instead
// of skipping them with a warning.  Use for debugging nondeterministic crashes.
// Set via `TY_STRICT_LIVENESS=1` or `--strict-liveness` CLI flag.
feature_flag!(pub(crate) strict_liveness, "TY_STRICT_LIVENESS");
feature_limit!(
    pub(crate) liveness_disk_graph_ptr_capacity,
    "TY_LIVENESS_DISK_PTR_CAPACITY",
    1 << 24
);

/// Right-size the disk-backed liveness graph's node pointer table (default ON).
///
/// The auto-disk path historically sized the ptr table to the
/// `states * tableau` node ESTIMATE, which over-provisions ~340x on specs like
/// cf1s_folklore (16.7M slots for ~30k actual nodes ≈ hundreds of MB of mostly
/// resident-but-empty mmap). When ON, the auto-disk path instead starts the
/// table at a modest capacity and lets it grow (rehash-exact) to the actual
/// node count. `TY_LIVENESS_PTR_RIGHTSIZE=0` restores the estimate-sized table
/// (grow is never triggered, reproducing the historical footprint for A/B).
pub(crate) fn liveness_ptr_rightsize() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    crate::debug_env::env_flag_default_on(&CACHED, "TY_LIVENESS_PTR_RIGHTSIZE")
}

/// Threshold (in estimated behavior-graph nodes) above which the liveness
/// checker automatically switches to disk-backed graph storage.
///
/// Default: 2M nodes.  Override with `TY_LIVENESS_AUTO_DISK_THRESHOLD`.
///
/// Tier-1 #4: when the byte budget (`TY_LIVENESS_DISK_BUDGET_MB`) is set, the
/// threshold is lowered to whichever is smaller of the node count and the
/// byte-budget-equivalent node count, so disk-backed storage engages near a
/// memory ceiling rather than purely a node count. With the budget disabled
/// (default) the threshold is exactly the historical node count.
pub(crate) fn liveness_auto_disk_threshold() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    let count_threshold =
        crate::debug_env::env_usize_or(&LIMIT, "TY_LIVENESS_AUTO_DISK_THRESHOLD", 2_000_000);
    // `Some(count)` so the byte budget can only lower (never raise) the
    // threshold; collapse back to the count when the budget is disabled.
    liveness_effective_unit_limit(Some(count_threshold), LIVENESS_BYTES_PER_GRAPH_NODE)
        .unwrap_or(count_threshold)
}

// ---------------------------------------------------------------------------
// Byte-budget gate (Tier-1 #4): estimated-memory metric for the disk stores.
//
// The historical liveness disk gates are keyed on NODE / ENTRY COUNT
// (`liveness_auto_disk_threshold`, `liveness_inmemory_node_limit`,
// `liveness_inmemory_successor_limit`). Count-based gates do not track the very
// thing that motivates spilling to disk — resident memory. This adds an
// estimated-BYTES budget so the disk stores can engage near a memory ceiling
// rather than a fixed node count.
//
// SHIP MOSTLY-DARK: the default budget is `0` (= disabled). With the budget
// disabled the count-based gates behave EXACTLY as before, so no verdict, state
// count, or storage decision changes unless the operator opts in by setting
// `TY_LIVENESS_DISK_BUDGET_MB`. When set, each count gate is tightened to the
// MORE CONSERVATIVE of {its count limit, budget_bytes / per-unit estimate};
// the byte budget can only make disk engage EARLIER, never later, so it can
// never relax an existing memory guard.

/// Estimated bytes for a single in-memory behavior-graph node (`NodeInfo`).
///
/// `NodeInfo` carries a successors `Vec`, a compact optional disk-trace parent,
/// and precomputed check bitmasks. The matching count default
/// (`liveness_inmemory_node_limit`, 5M nodes) is documented as ~1 GB, i.e.
/// ~200 bytes/node — used here so a byte budget maps back to an equivalent
/// node count. This is an estimate hook; refine alongside the deeper
/// per-`NodeInfo` accounting follow-up (see notes).
pub(crate) const LIVENESS_BYTES_PER_GRAPH_NODE: usize = 200;

/// Estimated bytes for a single in-memory successor-cache entry.
///
/// Each entry is `Fingerprint -> Vec<Fingerprint>` (16-byte key + 24-byte Vec
/// header + N*8 successor bytes). The matching count default
/// (`liveness_inmemory_successor_limit`, 5M entries) is documented as ~280 MB
/// at avg 3 successors, i.e. ~56 bytes/entry.
pub(crate) const LIVENESS_BYTES_PER_SUCCESSOR_ENTRY: usize = 56;

/// Estimated-memory budget (in MiB) for liveness disk-spill gating.
///
/// Default `0` disables the byte budget entirely, preserving the historical
/// count-based behavior. When non-zero, the byte budget is converted into an
/// equivalent unit count for each gate (via the per-unit estimate constants)
/// and combined with the count limit by taking the smaller of the two — the
/// disk store therefore engages as soon as EITHER bound is reached.
///
/// Override with `TY_LIVENESS_DISK_BUDGET_MB`.
pub(crate) fn liveness_disk_budget_mb() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    crate::debug_env::env_usize_or(&LIMIT, "TY_LIVENESS_DISK_BUDGET_MB", 0)
}

/// Structural-heap budget (in MiB) for the mid-BFS hybrid liveness trip.
///
/// The sequential checker normally CACHES the full system successor graph plus
/// the per-state / per-transition inline liveness structures (`bfs_seeded_states`,
/// the state/action bitmasks) during BFS so the post-BFS SCC/tableau pass can
/// reuse them (the fast path for small liveness specs). On huge liveness specs
/// (e.g. `cf1s_folklore`, ~2M states with high message-passing fan-out) those
/// caches dominate peak RSS (7.2 GB — nine times TLC).
///
/// When the estimated size of those caches crosses this budget MID-BFS, the
/// checker moves ordered successor payloads to an append-only temporary file,
/// drops the heavyweight inline maps, and switches the post-BFS liveness pass
/// to the on-the-fly checker. Exactly resolvable sources replay the disk-backed
/// adjacency; all other sources regenerate through Next. This removes O(edges)
/// graph-owned heap payloads, but is not constant-memory: the disk graph keeps
/// an O(states) parent-offset index and a fixed-slot direct cache whose payload
/// bytes depend on fanout; mapped-page residency is OS-dependent. Below the
/// budget the fast cached path is unchanged, so small/medium liveness specs
/// (and the native-fused parallel path) are untouched.
///
/// This budget is measured against a load-INDEPENDENT structural estimate of
/// the caches (entry counts × per-entry size), which under-counts true resident
/// bytes (thread-local eval caches, allocator slack, ArrayState value trees) by
/// roughly 2–2.5×. The default is chosen so the trip fires early enough to
/// avoid most of that transient cache growth on `cf1s_folklore`, while staying
/// above the largest sequential-liveness canary (`Huang`, ~90 MB estimate).
///
/// Default `128` MiB. Set to `0` to DISABLE the auto-gate entirely (kill
/// switch — restores the historical always-cached behavior). Override with
/// `TY_LIVENESS_REGEN_BUDGET_MB`.
pub(crate) fn liveness_regen_budget_mb() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    crate::debug_env::env_usize_or(&LIMIT, "TY_LIVENESS_REGEN_BUDGET_MB", 128)
}

/// Budget for the liveness-cache regeneration trip in raw bytes, or `None` when
/// disabled (`0`).
pub(crate) fn liveness_regen_budget_bytes() -> Option<usize> {
    let mb = liveness_regen_budget_mb();
    if mb == 0 {
        None
    } else {
        Some(mb.saturating_mul(1024 * 1024))
    }
}

/// Pure decision rule for the mid-BFS regeneration trip.
///
/// `None` is the absolute budget kill switch and therefore wins even over the
/// force hook. Keeping this rule pure makes that precedence testable without
/// mutating process environment or racing the cached debug flags.
pub(crate) fn liveness_regen_should_trip(
    budget_bytes: Option<usize>,
    force: bool,
    estimated_bytes: usize,
) -> bool {
    budget_bytes.is_some_and(|budget| force || estimated_bytes >= budget)
}

// Force the mid-BFS hybrid liveness trip at the next eligible monitoring poll,
// regardless of the structural estimate (including zero). Test/probe hook used
// to exercise retained-or-regenerated checking and trace reconstruction on
// small specs where the budget would never be reached.
// Enable with `TY_LIVENESS_REGEN_FORCE=1`.
feature_flag!(pub(crate) liveness_regen_force, "TY_LIVENESS_REGEN_FORCE");

fn liveness_otf_compact_cache_disabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Whether on-the-fly (retained-or-regenerated) liveness exploration should
/// retain checker-owned compact state payloads. Only the exact value `1`
/// disables the default-on cache; other values are treated as unset so a stray
/// `0` cannot silently turn the optimization off.
pub(crate) fn liveness_otf_compact_cache_enabled() -> bool {
    !liveness_otf_compact_cache_disabled_value(
        std::env::var("TY_NO_LIVENESS_OTF_COMPACT_CACHE")
            .ok()
            .as_deref(),
    )
}

/// Byte budget converted to raw bytes, or `None` when disabled (`0`).
pub(crate) fn liveness_disk_budget_bytes() -> Option<usize> {
    let mb = liveness_disk_budget_mb();
    if mb == 0 {
        None
    } else {
        Some(mb.saturating_mul(1024 * 1024))
    }
}

/// Convert the active byte budget into an equivalent count of `bytes_per_unit`
/// units, returning `None` when the byte budget is disabled.
///
/// A budget that is set but smaller than one unit clamps to `1` so the gate
/// still has a meaningful (minimum) threshold rather than `0`.
fn liveness_disk_budget_units(bytes_per_unit: usize) -> Option<usize> {
    let budget = liveness_disk_budget_bytes()?;
    let per = bytes_per_unit.max(1);
    Some((budget / per).max(1))
}

/// Combine a count limit with the byte-budget-derived unit count, taking the
/// more conservative (smaller) of the two. Used by the count-based gates so the
/// byte budget can only tighten — never loosen — an existing threshold.
///
/// `count_limit == None` means "no count limit"; in that case the byte budget
/// (if any) becomes the sole limit.
pub(crate) fn liveness_effective_unit_limit(
    count_limit: Option<usize>,
    bytes_per_unit: usize,
) -> Option<usize> {
    match (count_limit, liveness_disk_budget_units(bytes_per_unit)) {
        (Some(count), Some(budget)) => Some(count.min(budget)),
        (Some(count), None) => Some(count),
        (None, budget) => budget,
    }
}

const DISK_GRAPH_OVERRIDE_UNSET: u8 = 0;
const DISK_GRAPH_OVERRIDE_FALSE: u8 = 1;
const DISK_GRAPH_OVERRIDE_TRUE: u8 = 2;
const INMEMORY_NODE_LIMIT_OVERRIDE_UNSET: usize = usize::MAX;
const BITMASK_FLUSH_THRESHOLD_OVERRIDE_UNSET: usize = usize::MAX;

thread_local! {
    // Test/debug overrides must be local to the test thread. Process-global
    // atomics made an ordinary checker observe a concurrently-running test's
    // forced disk backend, changing liveness verdicts under the parallel test
    // harness. Production env-derived defaults remain process-wide OnceLocks.
    // The guards are deliberately !Send so they restore the installing thread.
    static FORCE_DISK_GRAPH: std::cell::Cell<u8> =
        const { std::cell::Cell::new(DISK_GRAPH_OVERRIDE_UNSET) };
    static FORCE_DISK_SUCCESSORS: std::cell::Cell<u8> =
        const { std::cell::Cell::new(DISK_GRAPH_OVERRIDE_UNSET) };
    static FORCE_DISK_BITMASKS: std::cell::Cell<u8> =
        const { std::cell::Cell::new(DISK_GRAPH_OVERRIDE_UNSET) };
    static FORCE_INMEMORY_NODE_LIMIT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(INMEMORY_NODE_LIMIT_OVERRIDE_UNSET) };
    static FORCE_INMEMORY_SUCCESSOR_LIMIT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(INMEMORY_NODE_LIMIT_OVERRIDE_UNSET) };
    static FORCE_DISK_BITMASK_FLUSH_THRESHOLD: std::cell::Cell<usize> =
        const { std::cell::Cell::new(BITMASK_FLUSH_THRESHOLD_OVERRIDE_UNSET) };
}

/// Whether the liveness checker should use the disk-backed behavior graph.
///
/// Production behavior is still controlled by `TY_LIVENESS_DISK_GRAPH`, but
/// tests can override it on the checker thread without fighting the cached env
/// lookup in a shared test binary.
pub(crate) fn use_disk_graph() -> bool {
    match FORCE_DISK_GRAPH.with(std::cell::Cell::get) {
        DISK_GRAPH_OVERRIDE_TRUE => true,
        DISK_GRAPH_OVERRIDE_FALSE => false,
        _ => {
            static FLAG: OnceLock<bool> = OnceLock::new();
            crate::debug_env::env_flag_is_set(&FLAG, "TY_LIVENESS_DISK_GRAPH")
        }
    }
}

/// Set a thread-local override for [`use_disk_graph`] until the returned guard
/// is dropped.
///
/// Hidden from normal docs via the crate root re-export; integration tests use
/// this instead of mutating `TY_LIVENESS_DISK_GRAPH` in-process.
#[cfg(any(test, feature = "testing"))]
pub fn set_use_disk_graph_override(value: bool) -> UseDiskGraphGuard {
    let value = if value {
        DISK_GRAPH_OVERRIDE_TRUE
    } else {
        DISK_GRAPH_OVERRIDE_FALSE
    };
    let previous = FORCE_DISK_GRAPH.with(|slot| slot.replace(value));
    UseDiskGraphGuard {
        previous,
        _thread_local: std::marker::PhantomData,
    }
}

/// RAII guard that restores the previous disk-graph override on drop.
#[cfg(any(test, feature = "testing"))]
pub struct UseDiskGraphGuard {
    previous: u8,
    _thread_local: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(any(test, feature = "testing"))]
impl Drop for UseDiskGraphGuard {
    fn drop(&mut self) {
        FORCE_DISK_GRAPH.with(|slot| slot.set(self.previous));
    }
}

/// Whether the BFS successor cache should use a disk-backed store.
///
/// Part of #3176: separate from `TY_LIVENESS_DISK_GRAPH` (which controls the
/// post-BFS behavior graph). This controls the BFS-time `SuccessorGraph` used
/// to record parent→child transitions for liveness SCC analysis.
pub(crate) fn use_disk_successors() -> bool {
    match FORCE_DISK_SUCCESSORS.with(std::cell::Cell::get) {
        DISK_GRAPH_OVERRIDE_TRUE => true,
        DISK_GRAPH_OVERRIDE_FALSE => false,
        _ => {
            static FLAG: OnceLock<bool> = OnceLock::new();
            crate::debug_env::env_flag_is_set(&FLAG, "TY_LIVENESS_DISK_SUCCESSORS")
        }
    }
}

/// Return the thread-local disk-successor override when present.
pub(crate) fn disk_successors_override() -> Option<bool> {
    match FORCE_DISK_SUCCESSORS.with(std::cell::Cell::get) {
        DISK_GRAPH_OVERRIDE_TRUE => Some(true),
        DISK_GRAPH_OVERRIDE_FALSE => Some(false),
        _ => None,
    }
}

/// Set a thread-local override for [`use_disk_successors`] until the returned
/// guard is dropped.
#[cfg(any(test, feature = "testing"))]
pub fn set_use_disk_successors_override(value: bool) -> UseDiskSuccessorsGuard {
    let value = if value {
        DISK_GRAPH_OVERRIDE_TRUE
    } else {
        DISK_GRAPH_OVERRIDE_FALSE
    };
    let previous = FORCE_DISK_SUCCESSORS.with(|slot| slot.replace(value));
    UseDiskSuccessorsGuard {
        previous,
        _thread_local: std::marker::PhantomData,
    }
}

/// RAII guard that restores the previous disk-successors override on drop.
#[cfg(any(test, feature = "testing"))]
pub struct UseDiskSuccessorsGuard {
    previous: u8,
    _thread_local: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(any(test, feature = "testing"))]
impl Drop for UseDiskSuccessorsGuard {
    fn drop(&mut self) {
        FORCE_DISK_SUCCESSORS.with(|slot| slot.set(self.previous));
    }
}

/// Whether the inline liveness bitmask maps should use disk-backed storage.
///
/// Part of #3177: separate from disk successors (#3176). Controls the
/// `StateBitmaskMap` and `ActionBitmaskMap` used during BFS inline liveness
/// recording. When enabled, bitmask entries spill to a sorted mmap file
/// to keep BFS memory bounded for billion-state specs.
pub(crate) fn use_disk_bitmasks() -> bool {
    match FORCE_DISK_BITMASKS.with(std::cell::Cell::get) {
        DISK_GRAPH_OVERRIDE_TRUE => true,
        DISK_GRAPH_OVERRIDE_FALSE => false,
        _ => {
            static FLAG: OnceLock<bool> = OnceLock::new();
            crate::debug_env::env_flag_is_set(&FLAG, "TY_LIVENESS_DISK_BITMASKS")
        }
    }
}

/// Set a thread-local override for [`use_disk_bitmasks`] until the returned
/// guard is dropped.
#[cfg(any(test, feature = "testing"))]
pub fn set_use_disk_bitmasks_override(value: bool) -> UseDiskBitmasksGuard {
    let value = if value {
        DISK_GRAPH_OVERRIDE_TRUE
    } else {
        DISK_GRAPH_OVERRIDE_FALSE
    };
    let previous = FORCE_DISK_BITMASKS.with(|slot| slot.replace(value));
    UseDiskBitmasksGuard {
        previous,
        _thread_local: std::marker::PhantomData,
    }
}

/// RAII guard that restores the previous disk-bitmasks override on drop.
#[cfg(any(test, feature = "testing"))]
pub struct UseDiskBitmasksGuard {
    previous: u8,
    _thread_local: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(any(test, feature = "testing"))]
impl Drop for UseDiskBitmasksGuard {
    fn drop(&mut self) {
        FORCE_DISK_BITMASKS.with(|slot| slot.set(self.previous));
    }
}

/// Optional test/debug threshold for flushing liveness disk bitmask hot tiers.
///
/// Production uses the default threshold from `storage/disk_bitmask.rs`. Tests
/// can override it to force repeated hot->cold flushes on small specs so the
/// BFS boundary flush hooks are exercised deterministically.
pub(crate) fn liveness_disk_bitmask_flush_threshold() -> Option<usize> {
    let override_threshold = FORCE_DISK_BITMASK_FLUSH_THRESHOLD.with(std::cell::Cell::get);
    if override_threshold != BITMASK_FLUSH_THRESHOLD_OVERRIDE_UNSET {
        return Some(override_threshold);
    }

    static LIMIT: OnceLock<Option<usize>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("TY_LIVENESS_DISK_BITMASK_FLUSH_THRESHOLD")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
    })
}

/// Set a thread-local override for [`liveness_disk_bitmask_flush_threshold`]
/// until the returned guard is dropped.
#[cfg(any(test, feature = "testing"))]
pub fn set_liveness_disk_bitmask_flush_threshold_override(
    value: Option<usize>,
) -> LivenessDiskBitmaskFlushThresholdGuard {
    let previous = FORCE_DISK_BITMASK_FLUSH_THRESHOLD
        .with(|slot| slot.replace(value.unwrap_or(BITMASK_FLUSH_THRESHOLD_OVERRIDE_UNSET)));
    LivenessDiskBitmaskFlushThresholdGuard {
        previous,
        _thread_local: std::marker::PhantomData,
    }
}

/// RAII guard that restores the previous disk-bitmask flush-threshold override
/// on drop.
#[cfg(any(test, feature = "testing"))]
pub struct LivenessDiskBitmaskFlushThresholdGuard {
    previous: usize,
    _thread_local: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(any(test, feature = "testing"))]
impl Drop for LivenessDiskBitmaskFlushThresholdGuard {
    fn drop(&mut self) {
        FORCE_DISK_BITMASK_FLUSH_THRESHOLD.with(|slot| slot.set(self.previous));
    }
}

/// Limit for the in-memory liveness graph node count.
///
/// Defaults to 5 million nodes (~1 GB). Override with env var
/// `TY_LIVENESS_INMEMORY_NODE_LIMIT`. Set to `0` to disable.
///
/// Only the in-memory topology store enforces the budget.
/// The disk-backed backend ignores it, which lets tests deterministically prove
/// that disk-backed graph storage bypasses the in-memory graph wall.
pub(crate) fn liveness_inmemory_node_limit() -> Option<usize> {
    let override_limit = FORCE_INMEMORY_NODE_LIMIT.with(std::cell::Cell::get);
    if override_limit != INMEMORY_NODE_LIMIT_OVERRIDE_UNSET {
        return Some(override_limit);
    }

    /// Default in-memory node limit: 5 million nodes.
    ///
    /// At ~200 bytes per `NodeInfo` (successors Vec, trace slot, check masks),
    /// 5M nodes consumes ~1 GB. Beyond this, disk-backed mode should be used.
    /// Override with `TY_LIVENESS_INMEMORY_NODE_LIMIT` env var.
    /// Set to `0` to disable the limit entirely (not recommended).
    ///
    /// Part of #4080: previously defaulted to `None` (unlimited), which
    /// contributed to OOM kills when multiple agents ran in parallel.
    const DEFAULT_INMEMORY_NODE_LIMIT: usize = 5_000_000;

    static LIMIT: OnceLock<Option<usize>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let from_env = std::env::var("TY_LIVENESS_INMEMORY_NODE_LIMIT")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok());
        let count_limit = match from_env {
            Some(0) => None, // Explicit 0 disables the limit
            Some(n) => Some(n),
            None => Some(DEFAULT_INMEMORY_NODE_LIMIT),
        };
        // Tier-1 #4: tighten the node-count gate to whichever is smaller of the
        // count limit and the byte-budget-equivalent node count. When the byte
        // budget is disabled (default) this returns the count limit unchanged.
        liveness_effective_unit_limit(count_limit, LIVENESS_BYTES_PER_GRAPH_NODE)
    })
}

/// Set a thread-local override for [`liveness_inmemory_node_limit`] until the
/// returned guard is dropped.
#[cfg(any(test, feature = "testing"))]
pub fn set_liveness_inmemory_node_limit_override(
    value: Option<usize>,
) -> LivenessInMemoryNodeLimitGuard {
    let previous = FORCE_INMEMORY_NODE_LIMIT
        .with(|slot| slot.replace(value.unwrap_or(INMEMORY_NODE_LIMIT_OVERRIDE_UNSET)));
    LivenessInMemoryNodeLimitGuard {
        previous,
        _thread_local: std::marker::PhantomData,
    }
}

/// RAII guard that restores the previous in-memory node-limit override on drop.
#[cfg(any(test, feature = "testing"))]
pub struct LivenessInMemoryNodeLimitGuard {
    previous: usize,
    _thread_local: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(any(test, feature = "testing"))]
impl Drop for LivenessInMemoryNodeLimitGuard {
    fn drop(&mut self) {
        FORCE_INMEMORY_NODE_LIMIT.with(|slot| slot.set(self.previous));
    }
}

/// Limit for the in-memory BFS successor cache entry count.
///
/// Defaults to 5 million entries (~280 MB at avg 3 successors). Override with
/// env var `TY_LIVENESS_INMEMORY_SUCCESSOR_LIMIT`. Set to `0` to disable.
///
/// Only the in-memory successor cache enforces the budget.
/// The disk-backed successor cache ignores it, which lets tests
/// deterministically prove that disk-backed successor storage bypasses the
/// in-memory successor wall that motivated #3176. Auto-migration happens only
/// when inserting a new parent would exceed this limit, so configurations that
/// fit under the advertised budget remain on the faster in-memory backend.
pub(crate) fn liveness_inmemory_successor_limit() -> Option<usize> {
    let override_limit = FORCE_INMEMORY_SUCCESSOR_LIMIT.with(std::cell::Cell::get);
    if override_limit != INMEMORY_NODE_LIMIT_OVERRIDE_UNSET {
        return Some(override_limit);
    }

    /// Default in-memory successor entry limit: 5 million entries.
    ///
    /// Each entry is `Fingerprint -> Vec<Fingerprint>` (16 bytes key + 24 bytes
    /// Vec header + N*8 bytes successors). At avg 3 successors per state, 5M
    /// entries is ~280 MB. Override with `TY_LIVENESS_INMEMORY_SUCCESSOR_LIMIT`.
    /// Set to `0` to disable the limit entirely (not recommended).
    ///
    /// Part of #4080: previously defaulted to `None` (unlimited).
    const DEFAULT_INMEMORY_SUCCESSOR_LIMIT: usize = 5_000_000;

    static LIMIT: OnceLock<Option<usize>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let from_env = std::env::var("TY_LIVENESS_INMEMORY_SUCCESSOR_LIMIT")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok());
        match from_env {
            Some(0) => None, // Explicit 0 disables the limit
            Some(n) => Some(n),
            None => Some(DEFAULT_INMEMORY_SUCCESSOR_LIMIT),
        }
    })
}

/// Set a thread-local override for [`liveness_inmemory_successor_limit`]
/// until the returned guard is dropped.
#[cfg(any(test, feature = "testing"))]
pub fn set_liveness_inmemory_successor_limit_override(
    value: Option<usize>,
) -> LivenessInMemorySuccessorLimitGuard {
    let previous = FORCE_INMEMORY_SUCCESSOR_LIMIT
        .with(|slot| slot.replace(value.unwrap_or(INMEMORY_NODE_LIMIT_OVERRIDE_UNSET)));
    LivenessInMemorySuccessorLimitGuard {
        previous,
        _thread_local: std::marker::PhantomData,
    }
}

/// RAII guard that restores the previous in-memory successor-limit override on
/// drop.
#[cfg(any(test, feature = "testing"))]
pub struct LivenessInMemorySuccessorLimitGuard {
    previous: usize,
    _thread_local: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(any(test, feature = "testing"))]
impl Drop for LivenessInMemorySuccessorLimitGuard {
    fn drop(&mut self) {
        FORCE_INMEMORY_SUCCESSOR_LIMIT.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn otf_compact_cache_kill_switch_requires_exact_one() {
        assert!(!liveness_otf_compact_cache_disabled_value(None));
        assert!(!liveness_otf_compact_cache_disabled_value(Some("")));
        assert!(!liveness_otf_compact_cache_disabled_value(Some("0")));
        assert!(!liveness_otf_compact_cache_disabled_value(Some("true")));
        assert!(liveness_otf_compact_cache_disabled_value(Some("1")));
    }

    #[test]
    fn regen_budget_kill_switch_wins_over_force() {
        assert!(!liveness_regen_should_trip(None, true, usize::MAX));
        assert!(!liveness_regen_should_trip(Some(128), false, 127));
        assert!(liveness_regen_should_trip(Some(128), false, 128));
        assert!(liveness_regen_should_trip(Some(128), true, 0));
    }

    #[test]
    fn liveness_test_overrides_are_thread_local_and_nested() {
        let _disk_graph = set_use_disk_graph_override(true);
        let _disk_successors = set_use_disk_successors_override(false);
        let _disk_bitmasks = set_use_disk_bitmasks_override(true);
        let _flush_threshold = set_liveness_disk_bitmask_flush_threshold_override(Some(1));
        let _node_limit = set_liveness_inmemory_node_limit_override(Some(11));
        let _successor_limit = set_liveness_inmemory_successor_limit_override(Some(13));

        let barrier = Arc::new(Barrier::new(2));
        let child_barrier = Arc::clone(&barrier);
        let child = std::thread::spawn(move || {
            let _disk_graph = set_use_disk_graph_override(false);
            let _disk_successors = set_use_disk_successors_override(true);
            let _disk_bitmasks = set_use_disk_bitmasks_override(false);
            let _flush_threshold = set_liveness_disk_bitmask_flush_threshold_override(Some(2));
            let _node_limit = set_liveness_inmemory_node_limit_override(Some(17));
            let _successor_limit = set_liveness_inmemory_successor_limit_override(Some(19));

            child_barrier.wait();
            assert!(!use_disk_graph());
            assert_eq!(disk_successors_override(), Some(true));
            assert!(!use_disk_bitmasks());
            assert_eq!(liveness_disk_bitmask_flush_threshold(), Some(2));
            assert_eq!(liveness_inmemory_node_limit(), Some(17));
            assert_eq!(liveness_inmemory_successor_limit(), Some(19));
            child_barrier.wait();
        });

        barrier.wait();
        assert!(use_disk_graph());
        assert_eq!(disk_successors_override(), Some(false));
        assert!(use_disk_bitmasks());
        assert_eq!(liveness_disk_bitmask_flush_threshold(), Some(1));
        assert_eq!(liveness_inmemory_node_limit(), Some(11));
        assert_eq!(liveness_inmemory_successor_limit(), Some(13));

        {
            let _nested = set_use_disk_bitmasks_override(false);
            assert!(!use_disk_bitmasks());
        }
        assert!(use_disk_bitmasks());

        barrier.wait();
        child
            .join()
            .expect("override-isolation worker should finish");
    }
}
