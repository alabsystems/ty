// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Lock-free fingerprint sets for compiled BFS deduplication.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

const EMPTY_SLOT: u64 = 0;
const MIN_CAPACITY: usize = 1024;
const NEW_STATE: u32 = 0;
const SEEN_STATE: u32 = 1;
const RESIZE_LOAD_FACTOR: f64 = 0.75;

/// Advisory probe-chain statistics for observability.
///
/// All counters use `Relaxed` ordering — they are informational only and do
/// not participate in the correctness protocol.
#[derive(Debug)]
pub struct ProbeStats {
    /// The longest probe chain observed across all operations.
    pub max_probe_len: usize,
    /// Average probe length across all insert attempts (including duplicates).
    pub avg_probe_len: f64,
    /// Total number of insert attempts (including duplicates).
    pub total_inserts: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncodedFingerprint {
    Table(u64),
    WrappedMax,
}

/// Result of inserting a fingerprint into an [`AtomicFpSet`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertResult {
    /// The fingerprint was newly inserted.
    Inserted,
    /// The fingerprint was already present.
    AlreadyPresent,
    /// The fixed-size table was full and the fingerprint could not be inserted.
    TableFull,
}

/// Lock-free fingerprint set used by compiled BFS deduplication.
///
/// The table uses open addressing with linear probing over a fixed-size
/// power-of-two array of `AtomicU64` slots. Slot value `0` means empty.
///
/// Public fingerprints are encoded as `fp + 1` before insertion so real
/// fingerprint `0` maps to stored value `1`. The single wrapped case
/// (`u64::MAX + 1 == 0`) is tracked out-of-band so the API still supports
/// the full `u64` domain.
pub struct AtomicFpSet {
    slots: Box<[AtomicU64]>,
    len: AtomicUsize,
    occupied_slots: AtomicUsize,
    wrapped_max_present: AtomicBool,
    mask: usize,
    // Advisory probe-chain observability counters (Relaxed ordering).
    max_probe_len: AtomicUsize,
    total_probes: AtomicU64,
    total_inserts: AtomicU64,
}

impl AtomicFpSet {
    /// Create a new fingerprint set.
    ///
    /// Capacity is rounded up to the next power of two with a minimum of 1024.
    #[must_use]
    pub fn new(capacity_hint: usize) -> Self {
        let min_capacity = capacity_hint.max(MIN_CAPACITY);
        let capacity = min_capacity
            .checked_next_power_of_two()
            .unwrap_or(1usize << (usize::BITS - 1));
        // 2026-07 memory audit: allocate the table as a ZEROED buffer
        // (`vec![0u64; n]` specializes to `alloc_zeroed`) instead of writing
        // `EMPTY_SLOT` into every slot. `EMPTY_SLOT == 0`, so the all-zero
        // buffer is exactly the all-empty table — but freshly zero-mapped
        // pages are only COMMITTED when first touched, so resident memory
        // scales with slots actually probed rather than with the capacity
        // hint (a 2M-slot table cost 16.8 MB of RSS on a 1-state spec).
        let slots: Box<[AtomicU64]> = {
            const _: () = assert!(EMPTY_SLOT == 0, "zeroed alloc requires EMPTY_SLOT == 0");
            let zeroed: Box<[u64]> = vec![0u64; capacity].into_boxed_slice();
            // SAFETY: `AtomicU64` has the same size, alignment, and bit
            // validity as `u64` (std guarantee), and slice metadata is
            // preserved by the pointer cast, so re-boxing the same
            // allocation as `[AtomicU64]` is sound.
            unsafe { Box::from_raw(Box::into_raw(zeroed) as *mut [AtomicU64]) }
        };

        Self {
            slots,
            len: AtomicUsize::new(0),
            occupied_slots: AtomicUsize::new(0),
            wrapped_max_present: AtomicBool::new(false),
            mask: capacity - 1,
            max_probe_len: AtomicUsize::new(0),
            total_probes: AtomicU64::new(0),
            total_inserts: AtomicU64::new(0),
        }
    }

    /// Insert a fingerprint if it is not already present.
    #[must_use]
    pub fn insert_if_absent(&self, fp: u64) -> InsertResult {
        match Self::encode(fp) {
            EncodedFingerprint::Table(encoded) => self.insert_encoded(encoded),
            EncodedFingerprint::WrappedMax => self.insert_wrapped_max(),
        }
    }

    /// Pre-seed the set with a batch of already-known fingerprints.
    ///
    /// Returns the number of newly inserted fingerprints.
    pub fn preseed<I>(&self, fingerprints: I) -> usize
    where
        I: IntoIterator<Item = u64>,
    {
        let mut inserted = 0usize;
        for fingerprint in fingerprints {
            match self.insert_if_absent(fingerprint) {
                InsertResult::Inserted => inserted += 1,
                InsertResult::AlreadyPresent => {}
                InsertResult::TableFull => {
                    panic!(
                        "AtomicFpSet::preseed exhausted capacity {} while inserting fingerprint {fingerprint:#018x}",
                        self.capacity()
                    );
                }
            }
        }
        inserted
    }

    /// Insert a batch of fingerprints, returning the number of newly inserted entries.
    ///
    /// More efficient than calling [`insert_if_absent`] in a loop when callers
    /// have a pre-collected batch (e.g., successors from a BFS level).
    ///
    /// Returns the count of newly inserted fingerprints. Fingerprints that
    /// were already present are NOT counted. Panics if the table is full
    /// (same as [`preseed`]).
    pub fn insert_batch(&self, fps: &[u64]) -> usize {
        let mut inserted = 0usize;
        for &fp in fps {
            match self.insert_if_absent(fp) {
                InsertResult::Inserted => inserted += 1,
                InsertResult::AlreadyPresent => {}
                InsertResult::TableFull => {
                    panic!(
                        "AtomicFpSet::insert_batch exhausted capacity {} while \
                         inserting fingerprint {fp:#018x}",
                        self.capacity()
                    );
                }
            }
        }
        inserted
    }

    /// Check whether a fingerprint is already present.
    #[must_use]
    pub fn contains(&self, fp: u64) -> bool {
        match Self::encode(fp) {
            EncodedFingerprint::Table(encoded) => self.contains_encoded(encoded),
            EncodedFingerprint::WrappedMax => self.wrapped_max_present.load(Ordering::Acquire),
        }
    }

    /// Return the number of distinct fingerprints inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Return the total number of table slots.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Return the table occupancy ratio.
    ///
    /// This is based on occupied table slots, so the out-of-band `u64::MAX`
    /// marker does not contribute to the ratio.
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        self.occupied_slots.load(Ordering::Acquire) as f64 / self.capacity() as f64
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return advisory probe-chain statistics.
    ///
    /// Counters are maintained with `Relaxed` ordering and may be slightly
    /// stale under concurrent inserts, but they are always non-decreasing and
    /// provide useful observability for tuning table sizing.
    #[must_use]
    pub fn probe_stats(&self) -> ProbeStats {
        let total_inserts = self.total_inserts.load(Ordering::Relaxed);
        let total_probes = self.total_probes.load(Ordering::Relaxed);
        ProbeStats {
            max_probe_len: self.max_probe_len.load(Ordering::Relaxed),
            avg_probe_len: if total_inserts == 0 {
                0.0
            } else {
                total_probes as f64 / total_inserts as f64
            },
            total_inserts,
        }
    }

    /// Record a probe of the given length into the advisory statistics.
    #[inline]
    fn record_probe(&self, probe_len: usize) {
        self.total_probes
            .fetch_add(probe_len as u64, Ordering::Relaxed);
        if probe_len > self.max_probe_len.load(Ordering::Relaxed) {
            let _ = self.max_probe_len.fetch_max(probe_len, Ordering::Relaxed);
        }
    }

    #[inline]
    fn encode(fp: u64) -> EncodedFingerprint {
        match fp.checked_add(1) {
            Some(encoded) => EncodedFingerprint::Table(encoded),
            None => EncodedFingerprint::WrappedMax,
        }
    }

    #[inline]
    fn contains_encoded(&self, encoded: u64) -> bool {
        debug_assert_ne!(encoded, EMPTY_SLOT);

        let start = self.start_index(encoded);
        for probe in 0..self.slots.len() {
            let idx = (start + probe) & self.mask;
            let current = self.slots[idx].load(Ordering::Acquire);

            if current == encoded {
                return true;
            }
            if current == EMPTY_SLOT {
                return false;
            }
        }

        false
    }

    #[inline]
    fn insert_wrapped_max(&self) -> InsertResult {
        if self
            .wrapped_max_present
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.len.fetch_add(1, Ordering::AcqRel);
            InsertResult::Inserted
        } else {
            InsertResult::AlreadyPresent
        }
    }

    #[inline]
    fn insert_encoded(&self, encoded: u64) -> InsertResult {
        debug_assert_ne!(encoded, EMPTY_SLOT);

        self.total_inserts.fetch_add(1, Ordering::Relaxed);
        let start = self.start_index(encoded);
        for probe in 0..self.slots.len() {
            let probe_len = probe + 1;
            let idx = (start + probe) & self.mask;
            let slot = &self.slots[idx];
            let current = slot.load(Ordering::Acquire);

            if current == encoded {
                self.record_probe(probe_len);
                return InsertResult::AlreadyPresent;
            }

            if current == EMPTY_SLOT {
                match slot.compare_exchange(
                    EMPTY_SLOT,
                    encoded,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.len.fetch_add(1, Ordering::AcqRel);
                        self.occupied_slots.fetch_add(1, Ordering::AcqRel);
                        self.record_probe(probe_len);
                        return InsertResult::Inserted;
                    }
                    Err(actual) if actual == encoded => {
                        self.record_probe(probe_len);
                        return InsertResult::AlreadyPresent;
                    }
                    Err(_) => {}
                }
            }
        }

        self.record_probe(self.slots.len());
        InsertResult::TableFull
    }

    /// Compute the starting probe index with splitmix64 mixing.
    ///
    /// The encoded fingerprint value may have poor low-bit entropy (e.g.,
    /// sequential fingerprints like 1, 2, 3). Mixing through splitmix64's
    /// finalizer ensures uniform distribution across the table.
    #[inline]
    fn start_index(&self, encoded: u64) -> usize {
        let mut h = encoded;
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        (h as usize) & self.mask
    }

    /// Collect all fingerprints stored in this set.
    ///
    /// Iterates all table slots and decodes committed fingerprints. Includes
    /// the out-of-band `u64::MAX` entry if present. The returned vector is
    /// unsorted and in arbitrary order.
    ///
    /// This is intended for diagnostics, testing, and resize rehashing — not
    /// for the hot path. The iteration acquires each slot with `Acquire`
    /// ordering, which is correct but not free.
    ///
    /// Part of #3988: Phase 5 compiled BFS step diagnostics.
    #[must_use]
    pub fn collect_all(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(self.len());
        for slot in self.slots.iter() {
            let encoded = slot.load(Ordering::Acquire);
            if encoded != EMPTY_SLOT {
                // Decode: stored value is `fp + 1`, so original fp is `encoded - 1`.
                result.push(encoded.wrapping_sub(1));
            }
        }
        if self.wrapped_max_present.load(Ordering::Acquire) {
            result.push(u64::MAX);
        }
        result
    }
}

/// Cumulative probe-chain statistics across all resize generations.
///
/// Unlike [`ProbeStats`], which reflects only the current inner table,
/// this struct accumulates statistics across the entire lifetime of a
/// [`ResizableAtomicFpSet`] so callers see full observability data.
#[derive(Debug, Clone, Copy)]
pub struct CumulativeProbeStats {
    /// The longest probe chain observed across ALL tables (all resize generations).
    pub max_probe_len: usize,
    /// Average probe length across ALL insert attempts (all resize generations).
    pub avg_probe_len: f64,
    /// Total number of insert attempts across ALL tables (includes rehash inserts).
    pub total_inserts: u64,
    /// Number of resize operations performed.
    pub resize_count: usize,
}

/// A resizable wrapper around [`AtomicFpSet`].
///
/// Normal inserts take a shared `RwLock` read lock so the steady-state fast
/// path remains uncontended. When the fixed-size table reaches 75% load or a
/// read-side insert sees [`InsertResult::TableFull`], the inserting thread
/// retries under the write lock after swapping in a table with 2x capacity.
pub struct ResizableAtomicFpSet {
    inner: RwLock<AtomicFpSet>,
    /// Cumulative probe statistics across all resize generations.
    cumulative_max_probe_len: AtomicUsize,
    cumulative_total_probes: AtomicU64,
    cumulative_total_inserts: AtomicU64,
    /// Number of resize operations performed.
    resize_count: AtomicUsize,
}

impl ResizableAtomicFpSet {
    /// Create a new resizable fingerprint set.
    #[must_use]
    pub fn new(capacity_hint: usize) -> Self {
        Self {
            inner: RwLock::new(AtomicFpSet::new(capacity_hint)),
            cumulative_max_probe_len: AtomicUsize::new(0),
            cumulative_total_probes: AtomicU64::new(0),
            cumulative_total_inserts: AtomicU64::new(0),
            resize_count: AtomicUsize::new(0),
        }
    }

    /// Insert a fingerprint if it is not already present.
    ///
    /// This wrapper never returns [`InsertResult::TableFull`]; it resizes and
    /// retries instead.
    #[must_use]
    pub fn insert_if_absent(&self, fp: u64) -> InsertResult {
        let (result, observed_capacity) = {
            let inner = self
                .inner
                .read()
                .expect("ResizableAtomicFpSet: read lock poisoned");
            let result = inner.insert_if_absent(fp);
            let should_resize = result == InsertResult::TableFull
                || Self::load_factor_of(&inner) >= RESIZE_LOAD_FACTOR;

            if !should_resize {
                return result;
            }

            (result, inner.capacity())
        };

        let mut inner = self
            .inner
            .write()
            .expect("ResizableAtomicFpSet: write lock poisoned");

        match result {
            InsertResult::Inserted | InsertResult::AlreadyPresent => {
                if Self::load_factor_of(&inner) >= RESIZE_LOAD_FACTOR {
                    self.resize_locked(&mut inner);
                }
                result
            }
            InsertResult::TableFull => {
                if inner.capacity() == observed_capacity
                    || Self::load_factor_of(&inner) >= RESIZE_LOAD_FACTOR
                {
                    self.resize_locked(&mut inner);
                }

                let retry = inner.insert_if_absent(fp);
                assert!(retry != InsertResult::TableFull,
                    "ResizableAtomicFpSet: insert_if_absent({fp:#018x}) returned TableFull after resize (capacity={})",
                    inner.capacity()
                );
                retry
            }
        }
    }

    /// Insert a batch of fingerprints, returning the number of newly inserted entries.
    ///
    /// Delegates to [`insert_if_absent`] per element, handling resizes
    /// automatically. Never returns [`InsertResult::TableFull`].
    pub fn insert_batch(&self, fps: &[u64]) -> usize {
        let mut inserted = 0usize;
        for &fp in fps {
            if self.insert_if_absent(fp) == InsertResult::Inserted {
                inserted += 1;
            }
        }
        inserted
    }

    /// Check whether a fingerprint is already present.
    #[must_use]
    pub fn contains(&self, fp: u64) -> bool {
        self.inner
            .read()
            .expect("ResizableAtomicFpSet: read lock poisoned")
            .contains(fp)
    }

    /// Return the number of distinct fingerprints inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .expect("ResizableAtomicFpSet: read lock poisoned")
            .len()
    }

    /// Return the total number of slots in the current inner table.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner
            .read()
            .expect("ResizableAtomicFpSet: read lock poisoned")
            .capacity()
    }

    /// Return the occupancy ratio of the current inner table.
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        self.inner
            .read()
            .expect("ResizableAtomicFpSet: read lock poisoned")
            .load_factor()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return advisory probe-chain statistics from the current inner table.
    #[must_use]
    pub fn probe_stats(&self) -> ProbeStats {
        self.inner
            .read()
            .expect("ResizableAtomicFpSet: read lock poisoned")
            .probe_stats()
    }

    /// Cumulative probe-chain statistics across all resize generations.
    ///
    /// Includes stats from the current inner table plus all prior tables
    /// that were replaced during resize. This gives a lifetime view of
    /// probe chain behavior.
    #[must_use]
    pub fn cumulative_probe_stats(&self) -> CumulativeProbeStats {
        let inner = self
            .inner
            .read()
            .expect("ResizableAtomicFpSet: read lock poisoned");
        let cur_max = inner.max_probe_len.load(Ordering::Relaxed);
        let cur_probes = inner.total_probes.load(Ordering::Relaxed);
        let cur_inserts = inner.total_inserts.load(Ordering::Relaxed);

        let cum_max = self.cumulative_max_probe_len.load(Ordering::Relaxed);
        let cum_probes = self.cumulative_total_probes.load(Ordering::Relaxed);
        let cum_inserts = self.cumulative_total_inserts.load(Ordering::Relaxed);

        let total_max = cur_max.max(cum_max);
        let total_probes = cur_probes + cum_probes;
        let total_inserts = cur_inserts + cum_inserts;
        let avg = if total_inserts == 0 {
            0.0
        } else {
            total_probes as f64 / total_inserts as f64
        };

        CumulativeProbeStats {
            max_probe_len: total_max,
            avg_probe_len: avg,
            total_inserts,
            resize_count: self.resize_count.load(Ordering::Relaxed),
        }
    }

    /// Number of resize operations performed.
    #[must_use]
    pub fn resize_count(&self) -> usize {
        self.resize_count.load(Ordering::Relaxed)
    }

    /// Collect all fingerprints stored in this set.
    ///
    /// Delegates to the inner [`AtomicFpSet::collect_all`].
    ///
    /// Part of #3988.
    #[must_use]
    pub fn collect_all(&self) -> Vec<u64> {
        self.inner
            .read()
            .expect("ResizableAtomicFpSet: read lock poisoned")
            .collect_all()
    }

    #[inline]
    fn load_factor_of(inner: &AtomicFpSet) -> f64 {
        inner.load_factor()
    }

    /// Resize the inner table to 2x capacity and rehash every committed entry.
    ///
    /// Accumulates probe stats from the old table into the cumulative counters
    /// before replacing it.
    ///
    /// The caller must hold the write lock for `inner`.
    fn resize_locked(&self, inner: &mut AtomicFpSet) {
        let old_capacity = inner.capacity();
        let new_capacity = old_capacity
            .checked_mul(2)
            .expect("ResizableAtomicFpSet: capacity overflow during resize");
        let old = std::mem::replace(inner, AtomicFpSet::new(new_capacity));

        // Accumulate probe stats from the old table before it is dropped.
        let old_max = old.max_probe_len.load(Ordering::Relaxed);
        let old_probes = old.total_probes.load(Ordering::Relaxed);
        let old_inserts = old.total_inserts.load(Ordering::Relaxed);
        let _ = self
            .cumulative_max_probe_len
            .fetch_max(old_max, Ordering::Relaxed);
        self.cumulative_total_probes
            .fetch_add(old_probes, Ordering::Relaxed);
        self.cumulative_total_inserts
            .fetch_add(old_inserts, Ordering::Relaxed);
        self.resize_count.fetch_add(1, Ordering::Relaxed);

        if old.wrapped_max_present.load(Ordering::Acquire) {
            match inner.insert_if_absent(u64::MAX) {
                InsertResult::Inserted => {}
                InsertResult::AlreadyPresent => {
                    panic!("ResizableAtomicFpSet: duplicate u64::MAX during resize");
                }
                InsertResult::TableFull => {
                    panic!(
                        "ResizableAtomicFpSet: TableFull rehashing u64::MAX (capacity {old_capacity} -> {new_capacity})"
                    );
                }
            }
        }

        for slot in old.slots.iter() {
            let encoded = slot.load(Ordering::Acquire);
            if encoded == EMPTY_SLOT {
                continue;
            }

            let fp = encoded.wrapping_sub(1);
            match inner.insert_if_absent(fp) {
                InsertResult::Inserted => {}
                InsertResult::AlreadyPresent => {
                    panic!("ResizableAtomicFpSet: duplicate fingerprint {fp:#018x} during resize");
                }
                InsertResult::TableFull => {
                    panic!(
                        "ResizableAtomicFpSet: TableFull during resize rehash (capacity {old_capacity} -> {new_capacity})"
                    );
                }
            }
        }
    }
}

/// Probe adapter for JIT-generated BFS code using a fixed-size [`AtomicFpSet`].
///
/// Returns `0` for a newly inserted fingerprint (new state) and `1` when the
/// fingerprint was already present (seen state).
#[must_use]
pub extern "C" fn atomic_fp_set_probe(fp_set: *const u8, fingerprint: u64) -> u32 {
    if fp_set.is_null() {
        return NEW_STATE;
    }

    // SAFETY: The JIT callback contract passes `fp_set` as a pointer to an
    // `AtomicFpSet`. Null is handled above.
    let fp_set = unsafe { &*fp_set.cast::<AtomicFpSet>() };
    match fp_set.insert_if_absent(fingerprint) {
        InsertResult::Inserted => NEW_STATE,
        InsertResult::AlreadyPresent => SEEN_STATE,
        InsertResult::TableFull => {
            panic!(
                "atomic_fp_set_probe: AtomicFpSet is full at capacity {} while inserting fingerprint {fingerprint:#018x}",
                fp_set.capacity()
            );
        }
    }
}

/// Probe adapter for JIT-generated BFS code using a [`ResizableAtomicFpSet`].
///
/// Returns `0` for a newly inserted fingerprint (new state) and `1` when the
/// fingerprint was already present (seen state).
#[must_use]
pub extern "C" fn resizable_fp_set_probe(fp_set: *const u8, fingerprint: u64) -> u32 {
    if fp_set.is_null() {
        return NEW_STATE;
    }

    // SAFETY: The JIT callback contract passes `fp_set` as a pointer to a
    // `ResizableAtomicFpSet`. Null is handled above.
    let fp_set = unsafe { &*fp_set.cast::<ResizableAtomicFpSet>() };
    match fp_set.insert_if_absent(fingerprint) {
        InsertResult::Inserted => NEW_STATE,
        InsertResult::AlreadyPresent => SEEN_STATE,
        InsertResult::TableFull => {
            panic!(
                "resizable_fp_set_probe: ResizableAtomicFpSet returned TableFull for fingerprint {fingerprint:#018x}"
            );
        }
    }
}

/// Non-atomic, single-threaded fingerprint set for the compiled native fused
/// BFS path.
///
/// This is the single-thread counterpart to [`ResizableAtomicFpSet`]. It uses
/// the **exact same** open-addressing layout, fingerprint encoding (`fp + 1`),
/// splitmix64 start-index mixing, out-of-band `u64::MAX` handling, and 0.75
/// resize-and-rehash policy — so for any sequence of fingerprints it produces a
/// bit-identical membership decision (`Inserted` vs `AlreadyPresent`) to
/// [`ResizableAtomicFpSet`]. The only difference is that it drops the
/// `AtomicU64`/`compare_exchange`, the `RwLock`, and the advisory probe-stat
/// counters, since the native fused level drives this set from a single thread.
///
/// # Single-thread invariant
///
/// The native fused BFS level ([`crate::bfs_level::TrustCgBfsLevelNative`]) is
/// only ever reached from the sequential `ModelChecker` (`--workers 0`/`1`); the
/// `ParallelChecker` (`--workers >= 2`) never constructs or runs it, and every
/// call into the level is serialized behind a `Mutex`. There is therefore no
/// concurrent access to this set, and the atomic machinery in
/// [`ResizableAtomicFpSet`] is pure overhead for that path. This type is *not*
/// `Sync`; sharing it across threads is a compile-time error, which guards the
/// single-thread contract.
pub struct SingleThreadFpSet {
    slots: Vec<u64>,
    len: usize,
    occupied_slots: usize,
    wrapped_max_present: bool,
    mask: usize,
}

impl SingleThreadFpSet {
    /// Create a new single-threaded fingerprint set.
    ///
    /// Capacity is rounded up to the next power of two with a minimum of 1024,
    /// matching [`AtomicFpSet::new`].
    #[must_use]
    pub fn new(capacity_hint: usize) -> Self {
        let min_capacity = capacity_hint.max(MIN_CAPACITY);
        let capacity = min_capacity
            .checked_next_power_of_two()
            .unwrap_or(1usize << (usize::BITS - 1));
        Self {
            slots: vec![EMPTY_SLOT; capacity],
            len: 0,
            occupied_slots: 0,
            wrapped_max_present: false,
            mask: capacity - 1,
        }
    }

    /// Insert a fingerprint if it is not already present.
    ///
    /// Never returns [`InsertResult::TableFull`]; it resizes and retries
    /// instead, mirroring [`ResizableAtomicFpSet::insert_if_absent`].
    #[must_use]
    #[inline]
    pub fn insert_if_absent(&mut self, fp: u64) -> InsertResult {
        let result = match Self::encode(fp) {
            EncodedFingerprint::Table(encoded) => self.insert_encoded(encoded),
            EncodedFingerprint::WrappedMax => self.insert_wrapped_max(),
        };
        // Grow eagerly at the same load factor as ResizableAtomicFpSet so the
        // table never actually reaches TableFull on the table path.
        if self.occupied_slots as f64 / self.slots.len() as f64 >= RESIZE_LOAD_FACTOR {
            self.resize();
        }
        result
    }

    /// Pre-seed the set with a batch of already-known fingerprints.
    ///
    /// Returns the number of newly inserted fingerprints.
    pub fn preseed<I>(&mut self, fingerprints: I) -> usize
    where
        I: IntoIterator<Item = u64>,
    {
        let mut inserted = 0usize;
        for fingerprint in fingerprints {
            if self.insert_if_absent(fingerprint) == InsertResult::Inserted {
                inserted += 1;
            }
        }
        inserted
    }

    /// Check whether a fingerprint is already present.
    #[must_use]
    pub fn contains(&self, fp: u64) -> bool {
        match Self::encode(fp) {
            EncodedFingerprint::Table(encoded) => self.contains_encoded(encoded),
            EncodedFingerprint::WrappedMax => self.wrapped_max_present,
        }
    }

    /// Return the number of distinct fingerprints inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the total number of table slots.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Return the table occupancy ratio.
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        self.occupied_slots as f64 / self.capacity() as f64
    }

    #[inline]
    fn encode(fp: u64) -> EncodedFingerprint {
        match fp.checked_add(1) {
            Some(encoded) => EncodedFingerprint::Table(encoded),
            None => EncodedFingerprint::WrappedMax,
        }
    }

    #[inline]
    fn contains_encoded(&self, encoded: u64) -> bool {
        debug_assert_ne!(encoded, EMPTY_SLOT);
        let start = self.start_index(encoded);
        for probe in 0..self.slots.len() {
            let idx = (start + probe) & self.mask;
            let current = self.slots[idx];
            if current == encoded {
                return true;
            }
            if current == EMPTY_SLOT {
                return false;
            }
        }
        false
    }

    #[inline]
    fn insert_wrapped_max(&mut self) -> InsertResult {
        if self.wrapped_max_present {
            InsertResult::AlreadyPresent
        } else {
            self.wrapped_max_present = true;
            self.len += 1;
            InsertResult::Inserted
        }
    }

    #[inline]
    fn insert_encoded(&mut self, encoded: u64) -> InsertResult {
        debug_assert_ne!(encoded, EMPTY_SLOT);
        let start = self.start_index(encoded);
        for probe in 0..self.slots.len() {
            let idx = (start + probe) & self.mask;
            let current = self.slots[idx];
            if current == encoded {
                return InsertResult::AlreadyPresent;
            }
            if current == EMPTY_SLOT {
                self.slots[idx] = encoded;
                self.len += 1;
                self.occupied_slots += 1;
                return InsertResult::Inserted;
            }
        }
        InsertResult::TableFull
    }

    /// Compute the starting probe index with splitmix64 mixing.
    ///
    /// Identical to [`AtomicFpSet::start_index`] so probe order matches.
    #[inline]
    fn start_index(&self, encoded: u64) -> usize {
        let mut h = encoded;
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        (h as usize) & self.mask
    }

    /// Double the table capacity and rehash every committed entry.
    fn resize(&mut self) {
        let old_capacity = self.slots.len();
        let new_capacity = old_capacity
            .checked_mul(2)
            .expect("SingleThreadFpSet: capacity overflow during resize");
        let old_slots = std::mem::replace(&mut self.slots, vec![EMPTY_SLOT; new_capacity]);
        self.mask = new_capacity - 1;
        self.occupied_slots = 0;
        // `wrapped_max_present` and table membership are preserved; only the
        // table slots are rehashed. `len` already accounts for every entry.
        let preserved_len = self.len;
        for encoded in old_slots {
            if encoded == EMPTY_SLOT {
                continue;
            }
            // Re-insert into the fresh table without touching `len`.
            let start = self.start_index(encoded);
            for probe in 0..self.slots.len() {
                let idx = (start + probe) & self.mask;
                if self.slots[idx] == EMPTY_SLOT {
                    self.slots[idx] = encoded;
                    self.occupied_slots += 1;
                    break;
                }
            }
        }
        debug_assert_eq!(self.len, preserved_len);
    }
}

/// Probe adapter for JIT-generated BFS code using a [`SingleThreadFpSet`].
///
/// Returns `0` for a newly inserted fingerprint (new state) and `1` when the
/// fingerprint was already present (seen state). This mirrors
/// [`resizable_fp_set_probe`] but operates on the non-atomic single-thread set.
///
/// # Safety
///
/// The JIT callback contract passes `fp_set` as a pointer to a
/// `SingleThreadFpSet`. The native fused BFS level drives this from a single
/// thread, so taking a `&mut` reference for the duration of the call is sound:
/// there is no concurrent access. Null is handled defensively.
pub extern "C" fn single_thread_fp_set_probe(fp_set: *mut u8, fingerprint: u64) -> u32 {
    if fp_set.is_null() {
        return NEW_STATE;
    }
    // SAFETY: see the function-level safety note. The pointer originates from a
    // live `SingleThreadFpSet` owned by the calling `TrustCgBfsLevelNative`, and
    // the native fused loop is single-threaded, so no aliasing `&mut` exists.
    let fp_set = unsafe { &mut *fp_set.cast::<SingleThreadFpSet>() };
    match fp_set.insert_if_absent(fingerprint) {
        InsertResult::Inserted => NEW_STATE,
        InsertResult::AlreadyPresent => SEEN_STATE,
        InsertResult::TableFull => {
            panic!(
                "single_thread_fp_set_probe: SingleThreadFpSet returned TableFull for fingerprint {fingerprint:#018x}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn thread_fp(thread_idx: usize, i: u64) -> u64 {
        ((thread_idx as u64) * 10_000 + i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_basic_insert_contains_and_duplicates() {
        let min_set = AtomicFpSet::new(8);
        assert_eq!(min_set.capacity(), 1024);
        assert_eq!(min_set.len(), 0);
        assert_eq!(min_set.load_factor(), 0.0);

        let set = AtomicFpSet::new(1500);
        assert_eq!(set.capacity(), 2048);

        let fp1 = 0xDEAD_BEEF_DEAD_BEEFu64;
        let fp2 = 0xCAFE_BABE_CAFE_BABEu64;

        assert!(!set.contains(fp1));
        assert_eq!(set.insert_if_absent(fp1), InsertResult::Inserted);
        assert!(set.contains(fp1));
        assert_eq!(set.insert_if_absent(fp1), InsertResult::AlreadyPresent);

        assert_eq!(set.insert_if_absent(fp2), InsertResult::Inserted);
        assert!(set.contains(fp2));
        assert_eq!(set.preseed([fp1, fp2, fp2.wrapping_add(1)]), 1);

        assert_eq!(set.len(), 3);
        assert_eq!(set.load_factor(), 3.0 / 2048.0);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_zero_and_wrapped_max_handling() {
        let set = AtomicFpSet::new(32);

        assert!(!set.contains(0));
        assert!(!set.contains(u64::MAX));

        assert_eq!(set.insert_if_absent(0), InsertResult::Inserted);
        assert_eq!(set.insert_if_absent(0), InsertResult::AlreadyPresent);
        assert!(set.contains(0));

        assert_eq!(set.insert_if_absent(u64::MAX), InsertResult::Inserted);
        assert_eq!(set.insert_if_absent(u64::MAX), InsertResult::AlreadyPresent);
        assert!(set.contains(u64::MAX));

        assert_eq!(set.len(), 2);
        assert_eq!(set.load_factor(), 1.0 / 1024.0);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_full_table_returns_table_full() {
        let set = AtomicFpSet::new(1);
        let capacity = set.capacity();

        for fingerprint in 0..capacity as u64 {
            assert_eq!(
                set.insert_if_absent(fingerprint),
                InsertResult::Inserted,
                "failed to insert fingerprint {fingerprint:#018x}"
            );
        }

        assert_eq!(set.len(), capacity);
        assert_eq!(
            set.insert_if_absent(capacity as u64 + 123_456),
            InsertResult::TableFull
        );
        assert!(!set.contains(capacity as u64 + 123_456));

        assert_eq!(set.insert_if_absent(u64::MAX), InsertResult::Inserted);
        assert_eq!(set.insert_if_absent(u64::MAX), InsertResult::AlreadyPresent);
        assert!(set.contains(u64::MAX));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_concurrent_inserts() {
        let num_threads = 4usize;
        let per_thread = 1_500u64;
        let set = Arc::new(AtomicFpSet::new(8_192));
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_idx| {
                let set = Arc::clone(&set);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();

                    let mut inserted = 0usize;
                    for i in 0..per_thread {
                        let fp = thread_fp(thread_idx, i);
                        if set.insert_if_absent(fp) == InsertResult::Inserted {
                            inserted += 1;
                        }
                    }
                    inserted
                })
            })
            .collect();

        let inserted_total: usize = handles
            .into_iter()
            .map(|handle| handle.join().expect("worker thread should not panic"))
            .sum();

        assert_eq!(inserted_total, num_threads * per_thread as usize);
        assert_eq!(set.len(), num_threads * per_thread as usize);
        assert!(set.contains(thread_fp(0, 0)));
        assert!(set.contains(thread_fp(num_threads - 1, per_thread - 1)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_probe_extern_c() {
        let set = AtomicFpSet::new(1024);
        let fp_set_ptr = std::ptr::from_ref(&set).cast::<u8>();
        let fingerprint = 0xA5A5_A5A5_5A5A_5A5Au64;

        assert_eq!(
            atomic_fp_set_probe(std::ptr::null(), fingerprint),
            NEW_STATE
        );
        assert_eq!(atomic_fp_set_probe(fp_set_ptr, fingerprint), NEW_STATE);
        assert_eq!(atomic_fp_set_probe(fp_set_ptr, fingerprint), SEEN_STATE);
        assert_eq!(set.len(), 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_atomic_fp_set_basic_insert_contains_and_duplicates() {
        let set = ResizableAtomicFpSet::new(32);
        let fp = 0x1234_5678_9ABC_DEF0u64;

        assert!(!set.contains(fp));
        assert_eq!(set.insert_if_absent(fp), InsertResult::Inserted);
        assert!(set.contains(fp));
        assert_eq!(set.insert_if_absent(fp), InsertResult::AlreadyPresent);
        assert_eq!(set.len(), 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_atomic_fp_set_zero_and_wrapped_max_survive_resize() {
        let set = ResizableAtomicFpSet::new(1);
        let initial_capacity = set.capacity();
        let threshold_entries = initial_capacity * 3 / 4;

        assert_eq!(set.insert_if_absent(0), InsertResult::Inserted);
        assert_eq!(set.insert_if_absent(u64::MAX), InsertResult::Inserted);

        for fingerprint in 1..=threshold_entries as u64 {
            let expected = if fingerprint == u64::MAX {
                InsertResult::AlreadyPresent
            } else {
                InsertResult::Inserted
            };
            assert_eq!(set.insert_if_absent(fingerprint), expected);
        }

        assert!(set.capacity() > initial_capacity);
        assert!(set.contains(0));
        assert!(set.contains(u64::MAX));
        assert_eq!(set.insert_if_absent(0), InsertResult::AlreadyPresent);
        assert_eq!(set.insert_if_absent(u64::MAX), InsertResult::AlreadyPresent);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_atomic_fp_set_resize_triggering() {
        let set = ResizableAtomicFpSet::new(1);
        let initial_capacity = set.capacity();
        let threshold_entries = initial_capacity * 3 / 4;

        for fingerprint in 0..(threshold_entries as u64 - 1) {
            assert_eq!(
                set.insert_if_absent(fingerprint),
                InsertResult::Inserted,
                "failed before resize threshold at {fingerprint}"
            );
        }
        assert_eq!(set.capacity(), initial_capacity);

        let trigger_fp = threshold_entries as u64 - 1;
        assert_eq!(set.insert_if_absent(trigger_fp), InsertResult::Inserted);
        assert_eq!(set.capacity(), initial_capacity * 2);
        assert_eq!(set.len(), threshold_entries);
        assert!(set.load_factor() < RESIZE_LOAD_FACTOR);

        for fingerprint in 0..threshold_entries as u64 {
            assert!(
                set.contains(fingerprint),
                "missing fingerprint {fingerprint}"
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_atomic_fp_set_concurrent_inserts_with_resize() {
        let num_threads = 16usize;
        let per_thread = 400u64;
        let set = Arc::new(ResizableAtomicFpSet::new(1));
        let initial_capacity = set.capacity();
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_idx| {
                let set = Arc::clone(&set);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();

                    let mut inserted = 0usize;
                    for i in 0..per_thread {
                        let fp = thread_fp(thread_idx, i);
                        match set.insert_if_absent(fp) {
                            InsertResult::Inserted => inserted += 1,
                            InsertResult::AlreadyPresent => {}
                            InsertResult::TableFull => {
                                panic!("ResizableAtomicFpSet returned TableFull");
                            }
                        }
                    }
                    inserted
                })
            })
            .collect();

        let inserted_total: usize = handles
            .into_iter()
            .map(|handle| handle.join().expect("worker thread should not panic"))
            .sum();

        assert_eq!(inserted_total, num_threads * per_thread as usize);
        assert_eq!(set.len(), num_threads * per_thread as usize);
        assert!(set.capacity() > initial_capacity);
        assert!(set.contains(thread_fp(0, 0)));
        assert!(set.contains(thread_fp(num_threads - 1, per_thread - 1)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_fp_set_probe_extern_c() {
        let set = ResizableAtomicFpSet::new(32);
        let fp_set_ptr = std::ptr::from_ref(&set).cast::<u8>();
        let fingerprint = 0xB6B6_B6B6_4D4D_4D4Du64;

        assert_eq!(
            resizable_fp_set_probe(std::ptr::null(), fingerprint),
            NEW_STATE
        );
        assert_eq!(resizable_fp_set_probe(fp_set_ptr, fingerprint), NEW_STATE);
        assert_eq!(resizable_fp_set_probe(fp_set_ptr, fingerprint), SEEN_STATE);
        assert_eq!(set.len(), 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_is_empty() {
        let set = AtomicFpSet::new(32);
        assert!(set.is_empty());
        assert_eq!(set.insert_if_absent(42), InsertResult::Inserted);
        assert!(!set.is_empty());

        let rset = ResizableAtomicFpSet::new(32);
        assert!(rset.is_empty());
        assert_eq!(rset.insert_if_absent(42), InsertResult::Inserted);
        assert!(!rset.is_empty());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_probe_stats_basic() {
        let set = AtomicFpSet::new(1024);

        // Empty set — no probes yet.
        let stats = set.probe_stats();
        assert_eq!(stats.max_probe_len, 0);
        assert_eq!(stats.avg_probe_len, 0.0);
        assert_eq!(stats.total_inserts, 0);

        // Insert 100 distinct fingerprints.
        for i in 1u64..=100 {
            let fp = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let _ = set.insert_if_absent(fp);
        }

        let stats = set.probe_stats();
        assert!(
            stats.max_probe_len >= 1,
            "max_probe_len should be >= 1, got {}",
            stats.max_probe_len
        );
        assert!(
            stats.avg_probe_len >= 1.0,
            "avg_probe_len should be >= 1.0, got {}",
            stats.avg_probe_len
        );
        assert_eq!(stats.total_inserts, 100);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_probe_stats_duplicates_counted() {
        let set = AtomicFpSet::new(1024);
        let fp = 0xDEAD_BEEF_CAFE_BABEu64;

        let _ = set.insert_if_absent(fp);
        let _ = set.insert_if_absent(fp); // duplicate
        let _ = set.insert_if_absent(fp); // duplicate

        let stats = set.probe_stats();
        assert_eq!(stats.total_inserts, 3);
        assert_eq!(set.len(), 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_atomic_fp_set_probe_stats() {
        let set = ResizableAtomicFpSet::new(32);

        for i in 1u64..=50 {
            let fp = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let _ = set.insert_if_absent(fp);
        }

        let stats = set.probe_stats();
        assert!(stats.max_probe_len >= 1);
        assert!(stats.avg_probe_len >= 1.0);
        // Note: total_inserts may be higher than 50 because resize re-inserts
        // all elements into the new table.
        assert!(stats.total_inserts >= 50);
    }

    /// Generate a unique fingerprint for throughput tests that guarantees
    /// non-overlapping ranges across threads (1M keys per thread).
    fn bench_fp(thread_idx: usize, i: u64) -> u64 {
        ((thread_idx as u64) * 1_000_000 + i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Throughput comparison: AtomicFpSet vs Mutex<HashSet<u64>>.
    ///
    /// This is not a proper criterion benchmark but a functional test that
    /// measures wall-clock time under concurrent load and asserts correctness.
    /// It serves as evidence that the lock-free path eliminates contention.
    #[cfg_attr(test, ntest::timeout(30000))]
    #[test]
    fn test_throughput_atomic_vs_mutex() {
        use std::collections::HashSet;
        use std::sync::Mutex;
        use std::time::Instant;

        let num_threads = 8usize;
        let per_thread = 50_000u64;
        let total_fps = num_threads * per_thread as usize;

        // --- AtomicFpSet ---
        let atomic_set = Arc::new(ResizableAtomicFpSet::new(total_fps));
        let barrier = Arc::new(Barrier::new(num_threads));
        let start = Instant::now();

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let set = Arc::clone(&atomic_set);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut inserted = 0usize;
                    for i in 0..per_thread {
                        let fp = bench_fp(t, i);
                        if set.insert_if_absent(fp) == InsertResult::Inserted {
                            inserted += 1;
                        }
                    }
                    inserted
                })
            })
            .collect();

        let atomic_inserted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let atomic_elapsed = start.elapsed();

        assert_eq!(atomic_inserted, total_fps);
        assert_eq!(atomic_set.len(), total_fps);

        // --- Mutex<HashSet<u64>> ---
        let mutex_set = Arc::new(Mutex::new(HashSet::with_capacity(total_fps)));
        let barrier = Arc::new(Barrier::new(num_threads));
        let start = Instant::now();

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let set = Arc::clone(&mutex_set);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut inserted = 0usize;
                    for i in 0..per_thread {
                        let fp = bench_fp(t, i);
                        let mut guard = set.lock().unwrap();
                        if guard.insert(fp) {
                            inserted += 1;
                        }
                    }
                    inserted
                })
            })
            .collect();

        let mutex_inserted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let mutex_elapsed = start.elapsed();

        assert_eq!(mutex_inserted, total_fps);
        assert_eq!(mutex_set.lock().unwrap().len(), total_fps);

        // Print comparison for observability (visible in `cargo test -- --nocapture`).
        let atomic_mops = total_fps as f64 / atomic_elapsed.as_secs_f64() / 1_000_000.0;
        let mutex_mops = total_fps as f64 / mutex_elapsed.as_secs_f64() / 1_000_000.0;
        let speedup = mutex_elapsed.as_secs_f64() / atomic_elapsed.as_secs_f64();

        eprintln!(
            "AtomicFpSet: {:.1}ms ({:.1} Mops/s), Mutex<HashSet>: {:.1}ms ({:.1} Mops/s), speedup: {:.2}x",
            atomic_elapsed.as_secs_f64() * 1000.0,
            atomic_mops,
            mutex_elapsed.as_secs_f64() * 1000.0,
            mutex_mops,
            speedup,
        );

        // Probe stats after the benchmark.
        let stats = atomic_set.probe_stats();
        eprintln!(
            "Probe stats: max={}, avg={:.2}, total_inserts={}",
            stats.max_probe_len, stats.avg_probe_len, stats.total_inserts,
        );

        // Correctness: both must agree on the total count.
        assert_eq!(atomic_set.len(), mutex_set.lock().unwrap().len());
    }

    /// High-contention stress test: many threads inserting overlapping keys.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_high_contention_correctness() {
        let num_threads = 32usize;
        let per_thread = 2_000usize;
        let overlap = 1_000usize;
        let set = Arc::new(ResizableAtomicFpSet::new(1));
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let set = Arc::clone(&set);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut inserted = 0usize;
                    for i in 0..per_thread {
                        let raw = if i < overlap {
                            (i as u64) + 1 // shared across all threads
                        } else {
                            (t as u64) * 1_000_000 + (i as u64) + 1 // unique per thread
                        };
                        let fp = raw.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        if set.insert_if_absent(fp) == InsertResult::Inserted {
                            inserted += 1;
                        }
                    }
                    inserted
                })
            })
            .collect();

        let total_inserted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let expected = overlap + num_threads * (per_thread - overlap);
        assert_eq!(total_inserted, expected);
        assert_eq!(set.len(), expected);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_collect_all_basic() {
        let set = AtomicFpSet::new(32);

        // Empty set should return empty.
        assert!(set.collect_all().is_empty());

        // Insert several fingerprints.
        let fps = [42u64, 0, 1000, 999_999_999];
        for &fp in &fps {
            assert_eq!(set.insert_if_absent(fp), InsertResult::Inserted);
        }
        assert_eq!(set.len(), fps.len());

        let mut collected = set.collect_all();
        collected.sort();
        let mut expected = fps.to_vec();
        expected.sort();
        assert_eq!(collected, expected);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_collect_all_with_wrapped_max() {
        let set = AtomicFpSet::new(32);

        let _ = set.insert_if_absent(7);
        let _ = set.insert_if_absent(u64::MAX);
        let _ = set.insert_if_absent(0);

        let mut collected = set.collect_all();
        collected.sort();
        assert_eq!(collected, vec![0, 7, u64::MAX]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_atomic_fp_set_collect_all_survives_resize() {
        let set = ResizableAtomicFpSet::new(1);
        let initial_capacity = set.capacity();
        let threshold = initial_capacity * 3 / 4;

        // Insert enough to trigger a resize.
        let mut expected = Vec::with_capacity(threshold + 2);
        for i in 0..=(threshold as u64) {
            let _ = set.insert_if_absent(i);
            expected.push(i);
        }
        assert!(
            set.capacity() > initial_capacity,
            "resize should have occurred"
        );

        let mut collected = set.collect_all();
        collected.sort();
        expected.sort();
        assert_eq!(collected, expected);
    }

    // ====================================================================
    // Batch insert tests (Part of #3991 Phase 8)
    // ====================================================================

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_batch_insert_basic() {
        let set = AtomicFpSet::new(4096);
        let fps: Vec<u64> = (1u64..=50)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();

        let inserted = set.insert_batch(&fps);
        assert_eq!(inserted, 50);
        assert_eq!(set.len(), 50);

        for &fp in &fps {
            assert!(set.contains(fp), "missing fp={fp:#018x}");
        }

        // Re-insert same batch — all should be duplicates.
        let reinserted = set.insert_batch(&fps);
        assert_eq!(reinserted, 0);
        assert_eq!(set.len(), 50);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_batch_insert_with_duplicates() {
        let set = AtomicFpSet::new(4096);
        let fps: Vec<u64> = vec![1, 2, 3, 1, 2, 3, 4, 5];
        let inserted = set.insert_batch(&fps);
        assert_eq!(inserted, 5);
        assert_eq!(set.len(), 5);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_batch_insert_empty() {
        let set = AtomicFpSet::new(1024);
        let inserted = set.insert_batch(&[]);
        assert_eq!(inserted, 0);
        assert_eq!(set.len(), 0);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_batch_insert_triggers_resize() {
        let set = ResizableAtomicFpSet::new(1);
        let initial_capacity = set.capacity();
        let threshold = initial_capacity * 3 / 4;

        let fps: Vec<u64> = (1u64..=(threshold as u64 + 10))
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();

        let inserted = set.insert_batch(&fps);
        assert_eq!(inserted, fps.len());
        assert_eq!(set.len(), fps.len());
        assert!(
            set.capacity() > initial_capacity,
            "capacity should have grown from {initial_capacity}, got {}",
            set.capacity()
        );

        for &fp in &fps {
            assert!(set.contains(fp), "missing fp after resize");
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_atomic_fp_set_batch_insert_concurrent() {
        let set = Arc::new(AtomicFpSet::new(1 << 18));
        let num_threads = 8usize;
        let fps_per_thread = 500usize;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let set = Arc::clone(&set);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let fps: Vec<u64> = (0..fps_per_thread)
                        .map(|i| thread_fp(t, i as u64))
                        .collect();
                    set.insert_batch(&fps)
                })
            })
            .collect();

        let total_inserted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total_inserted, num_threads * fps_per_thread);
        assert_eq!(set.len(), num_threads * fps_per_thread);
    }

    // ====================================================================
    // Cumulative probe stats tests (Part of #3991 Phase 8)
    // ====================================================================

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_cumulative_stats_across_resizes() {
        let set = ResizableAtomicFpSet::new(1);

        // Insert enough to trigger multiple resizes.
        for i in 1u64..=2000 {
            let fp = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let _ = set.insert_if_absent(fp);
        }

        assert!(
            set.resize_count() > 0,
            "should have resized at least once, got {}",
            set.resize_count()
        );

        // Cumulative stats should reflect all inserts including rehash inserts.
        let stats = set.cumulative_probe_stats();
        assert!(
            stats.total_inserts >= 2000,
            "cumulative inserts should be >= 2000, got {}",
            stats.total_inserts
        );
        assert!(
            stats.max_probe_len >= 1,
            "cumulative max_probe_len should be >= 1"
        );
        assert!(stats.resize_count > 0, "resize_count should be > 0");
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resizable_cumulative_stats_no_resize() {
        let set = ResizableAtomicFpSet::new(8192);

        // Insert few entries — no resize should occur.
        for i in 1u64..=10 {
            let _ = set.insert_if_absent(i);
        }

        assert_eq!(set.resize_count(), 0);
        let stats = set.cumulative_probe_stats();
        assert_eq!(stats.resize_count, 0);
        assert_eq!(stats.total_inserts, 10);
    }

    // ====================================================================
    // SingleThreadFpSet tests — single-thread compiled native fused path.
    // ====================================================================

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_single_thread_fp_set_basic_insert_contains_and_duplicates() {
        let mut set = SingleThreadFpSet::new(8);
        assert_eq!(set.capacity(), 1024);
        assert!(set.is_empty());

        let fp1 = 0xDEAD_BEEF_DEAD_BEEFu64;
        let fp2 = 0xCAFE_BABE_CAFE_BABEu64;

        assert!(!set.contains(fp1));
        assert_eq!(set.insert_if_absent(fp1), InsertResult::Inserted);
        assert!(set.contains(fp1));
        assert_eq!(set.insert_if_absent(fp1), InsertResult::AlreadyPresent);

        assert_eq!(set.insert_if_absent(fp2), InsertResult::Inserted);
        assert!(set.contains(fp2));
        assert_eq!(set.preseed([fp1, fp2, fp2.wrapping_add(1)]), 1);
        assert_eq!(set.len(), 3);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_single_thread_fp_set_zero_and_wrapped_max_handling() {
        let mut set = SingleThreadFpSet::new(32);

        assert!(!set.contains(0));
        assert!(!set.contains(u64::MAX));

        assert_eq!(set.insert_if_absent(0), InsertResult::Inserted);
        assert_eq!(set.insert_if_absent(0), InsertResult::AlreadyPresent);
        assert!(set.contains(0));

        assert_eq!(set.insert_if_absent(u64::MAX), InsertResult::Inserted);
        assert_eq!(set.insert_if_absent(u64::MAX), InsertResult::AlreadyPresent);
        assert!(set.contains(u64::MAX));

        assert_eq!(set.len(), 2);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_single_thread_fp_set_resize_preserves_all() {
        let mut set = SingleThreadFpSet::new(1);
        let initial_capacity = set.capacity();
        // Insert enough to trigger at least one resize.
        let count = initial_capacity * 4;
        for i in 0..count as u64 {
            assert_eq!(set.insert_if_absent(i), InsertResult::Inserted);
        }
        assert!(set.capacity() > initial_capacity, "should have resized");
        assert_eq!(set.len(), count);
        for i in 0..count as u64 {
            assert!(set.contains(i), "missing {i} after resize");
        }
        // Re-inserts are all duplicates.
        for i in 0..count as u64 {
            assert_eq!(set.insert_if_absent(i), InsertResult::AlreadyPresent);
        }
        assert_eq!(set.len(), count);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_single_thread_fp_set_probe_extern_c() {
        let mut set = SingleThreadFpSet::new(32);
        let fp_set_ptr = std::ptr::from_mut(&mut set).cast::<u8>();
        let fingerprint = 0xB6B6_B6B6_4D4D_4D4Du64;

        assert_eq!(
            single_thread_fp_set_probe(std::ptr::null_mut(), fingerprint),
            NEW_STATE
        );
        assert_eq!(
            single_thread_fp_set_probe(fp_set_ptr, fingerprint),
            NEW_STATE
        );
        assert_eq!(
            single_thread_fp_set_probe(fp_set_ptr, fingerprint),
            SEEN_STATE
        );
        assert_eq!(set.len(), 1);
    }

    /// Parity: SingleThreadFpSet must make the exact same membership decisions
    /// (Inserted vs AlreadyPresent) as ResizableAtomicFpSet for an arbitrary,
    /// duplicate-heavy fingerprint stream. This is the correctness guarantee
    /// that keeps `states_distinct` identical across the two dedup backends.
    #[cfg_attr(test, ntest::timeout(30000))]
    #[test]
    fn test_single_thread_fp_set_parity_with_resizable_atomic() {
        let atomic = ResizableAtomicFpSet::new(1);
        let mut single = SingleThreadFpSet::new(1);

        // Mix of sequential, overlapping, zero, and wrapped-max fingerprints.
        let mut raw: Vec<u64> = Vec::new();
        for i in 0..20_000u64 {
            raw.push(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            // Inject duplicates and edge values.
            if i % 7 == 0 {
                raw.push((i / 2).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            }
            if i % 1001 == 0 {
                raw.push(0);
                raw.push(u64::MAX);
            }
        }

        for fp in &raw {
            let a = atomic.insert_if_absent(*fp);
            let s = single.insert_if_absent(*fp);
            assert_eq!(a, s, "membership decision diverged for fp={fp:#018x}");
        }
        assert_eq!(
            atomic.len(),
            single.len(),
            "distinct count diverged between atomic and single-thread sets"
        );
        // Containment must agree on every key.
        for fp in &raw {
            assert_eq!(atomic.contains(*fp), single.contains(*fp));
        }
    }
}
