// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Disk-backed successor graph for BFS liveness caching.
//!
//! Stores `parent_fp -> [child_fps]` relationships in an append-only disk
//! file with an in-memory offset index. BFS appends successor records during
//! exploration; post-BFS liveness checking reads them back through an mmap view
//! plus a fixed-slot direct-mapped cache so SCC analysis does not rebuild every
//! decoded payload. The offset index remains O(states), cached Vec bytes depend
//! on fanout, and mapped-page residency is OS-dependent.
//!
//! Part of #3176.
//!
//! ## File Format
//!
//! Each record:
//! `[parent_fp: u64, num_successors: u32, succ_fp_0: u64, ..., succ_fp_N: u64]`
//! (`12 + 8×N` bytes per record).
//!
//! ## Index
//!
//! `FxHashMap<Fingerprint, u64>` maps parent fingerprint to file byte offset.
//! The logical key + offset payload is 16 bytes per state before HashMap control
//! bytes, spare capacity, and allocator overhead.
//!
//! ## Cache
//!
//! Fixed-slot direct-mapped cache (default 64K entries) avoids repeated mmap
//! parsing during SCC traversal. Slot count is independent of state count, but
//! cached successor-Vec bytes depend on fanout. Collisions evict one slot; the
//! mmap/index remain authoritative.

use crate::state::Fingerprint;
use memmap2::Mmap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};

/// Default cache size: 64K entries.
const DEFAULT_CACHE_SLOTS: usize = 1 << 16;

/// Record header size: `[parent_fp: u64, num_successors: u32]`.
const RECORD_HEADER_BYTES: usize = 12;

/// A single cache entry mapping a fingerprint to its successor list.
#[derive(Clone)]
struct CacheEntry {
    parent_fp: Fingerprint,
    offset: u64,
    successors: Vec<Fingerprint>,
}

/// Mutable read-side state wrapped in `RefCell` for interior mutability.
///
/// The liveness checker passes `&SuccessorGraph` through immutable contexts,
/// but the disk backend still needs mutable access to its mmap view and cache
/// state. `RefCell` keeps the backend single-threaded and cheap.
struct ReadState {
    mmap: Option<Mmap>,
    mapped_len: u64,
    cache: Vec<Option<CacheEntry>>,
    /// Whether the writer has been flushed since the last write.
    writer_flushed: bool,
}

/// Disk-backed successor graph.
///
/// Append-only file stores successor lists; an in-memory index provides O(1)
/// lookup by parent fingerprint; mmap-backed reads plus a fixed-slot direct
/// cache retain decoded successors. The index remains O(states), and cache
/// payload bytes depend on fanout.
///
/// Uses `RefCell` for interior mutability on the read path so that `get()`
/// can take `&self`, matching the immutable borrow pattern of the liveness
/// checker's context structs.
pub(crate) struct DiskSuccessorGraph {
    /// Buffered writer for appending records.
    writer: RefCell<BufWriter<File>>,
    /// Read-side state: mmap view + direct-mapped cache (interior mutability for `get(&self)`).
    read_state: RefCell<ReadState>,
    /// Current byte position in the file (next write offset).
    write_pos: u64,
    /// Index: parent fingerprint → file byte offset.
    index: FxHashMap<Fingerprint, u64>,
    /// Number of direct-mapped cache slots.
    cache_capacity: usize,
    /// Number of distinct parent entries.
    entry_count: usize,
    /// Running total of live successor fingerprints (dead overwrite records excluded).
    total_successors: usize,
    /// Backing temporary file handle (keeps the file alive).
    backing_file: tempfile::NamedTempFile,
}

impl DiskSuccessorGraph {
    /// Create a new disk-backed successor graph.
    ///
    /// The backing file is created in the system temp directory and
    /// automatically cleaned up on drop.
    pub(crate) fn new() -> io::Result<Self> {
        Self::with_cache_slots(DEFAULT_CACHE_SLOTS)
    }

    /// Create with a custom cache size.
    pub(crate) fn with_cache_slots(cache_slots: usize) -> io::Result<Self> {
        assert!(cache_slots > 0, "cache_slots must be non-zero");
        let backing = tempfile::NamedTempFile::new()?;
        let writer = BufWriter::with_capacity(64 * 1024, backing.as_file().try_clone()?);

        Ok(Self {
            writer: RefCell::new(writer),
            read_state: RefCell::new(ReadState {
                mmap: None,
                mapped_len: 0,
                cache: (0..cache_slots).map(|_| None).collect(),
                writer_flushed: true, // No writes yet, nothing to flush.
            }),
            write_pos: 0,
            index: FxHashMap::default(),
            cache_capacity: cache_slots,
            entry_count: 0,
            total_successors: 0,
            backing_file: backing,
        })
    }

    /// Shrink the direct-mapped read cache to at most `max_slots` empty slots.
    ///
    /// Changing the modulus invalidates every existing slot, so this drops the
    /// old cache rather than attempting to rehash it. The file and O(states)
    /// parent-offset index remain authoritative and unchanged.
    pub(crate) fn shrink_cache_to(&mut self, max_slots: usize) {
        assert!(max_slots > 0, "cache_slots must be non-zero");
        if self.cache_capacity <= max_slots {
            return;
        }
        let replacement = (0..max_slots).map(|_| None).collect();
        self.read_state.get_mut().cache = replacement;
        self.cache_capacity = max_slots;
    }

    /// Insert a parent fingerprint and its successor list.
    ///
    /// If the parent already exists, the old record becomes dead space in the
    /// file and the index is updated to point to the new record.
    pub(crate) fn insert(&mut self, parent_fp: Fingerprint, successors: Vec<Fingerprint>) {
        assert!(
            u32::try_from(successors.len()).is_ok(),
            "disk successor graph: too many successors for {parent_fp}: {}",
            successors.len()
        );
        let offset = self.write_pos;
        let successor_count = successors.len();
        let old_successor_count = self
            .index
            .get(&parent_fp)
            .copied()
            .map(|old_offset| self.cached_or_record_successor_count(parent_fp, old_offset));

        // Write record: [parent_fp: u64, num_successors: u32, succ_fp_0: u64, ...]
        let num_succs = successors.len() as u32;
        let mut writer = self.writer.borrow_mut();
        writer
            .write_all(&parent_fp.0.to_le_bytes())
            .expect("disk successor graph: write parent fp");
        // Unwrap is acceptable: I/O failure on a temp file is a fatal
        // system-level error (disk full, etc.).
        writer
            .write_all(&num_succs.to_le_bytes())
            .expect("disk successor graph: write num_succs");
        for &sfp in &successors {
            writer
                .write_all(&sfp.0.to_le_bytes())
                .expect("disk successor graph: write succ fp");
        }
        drop(writer);

        let record_size = RECORD_HEADER_BYTES as u64 + 8 * successors.len() as u64;
        self.write_pos += record_size;

        // Update index
        if self.index.insert(parent_fp, offset).is_none() {
            self.entry_count += 1;
        }
        self.total_successors += successor_count;
        if let Some(old_count) = old_successor_count {
            self.total_successors -= old_count;
        }

        let mut rs = self.read_state.borrow_mut();
        self.cache_put(&mut rs, parent_fp, offset, successors);
        rs.writer_flushed = false;
    }

    /// Look up successor fingerprints for a parent.
    ///
    /// Returns `None` if the parent was never inserted.
    /// Takes `&self` via interior mutability for compatibility with the
    /// liveness checker's immutable borrow pattern.
    pub(crate) fn get(&self, fp: &Fingerprint) -> Option<Vec<Fingerprint>> {
        let mut rs = self.read_state.borrow_mut();

        // Look up offset first so cache hits are validated against the current
        // authoritative record. Overwritten parents leave dead records behind.
        let &offset = self.index.get(fp)?;

        if let Some(successors) = self.cache_get(&rs, *fp, offset) {
            return Some(successors);
        }

        self.ensure_mmap_current(&mut rs);
        let mmap = rs
            .mmap
            .as_ref()
            .expect("disk successor graph: mmap unavailable after flush");
        let successors = Self::read_successors_from_mmap(mmap, offset, *fp);
        self.cache_put(&mut rs, *fp, offset, successors.clone());
        Some(successors)
    }

    #[inline]
    fn cache_index(&self, fp: Fingerprint) -> usize {
        (fp.0 as usize) % self.cache_capacity
    }

    fn cache_get(&self, rs: &ReadState, fp: Fingerprint, offset: u64) -> Option<Vec<Fingerprint>> {
        let idx = self.cache_index(fp);
        let entry = rs.cache.get(idx)?.as_ref()?;
        (entry.parent_fp == fp && entry.offset == offset).then(|| entry.successors.clone())
    }

    fn cache_successor_count(&self, rs: &ReadState, fp: Fingerprint, offset: u64) -> Option<usize> {
        let idx = self.cache_index(fp);
        let entry = rs.cache.get(idx)?.as_ref()?;
        (entry.parent_fp == fp && entry.offset == offset).then_some(entry.successors.len())
    }

    fn cache_put(
        &self,
        rs: &mut ReadState,
        fp: Fingerprint,
        offset: u64,
        successors: Vec<Fingerprint>,
    ) {
        let idx = self.cache_index(fp);
        rs.cache[idx] = Some(CacheEntry {
            parent_fp: fp,
            offset,
            successors,
        });
    }

    fn ensure_mmap_current(&self, rs: &mut ReadState) {
        if rs.writer_flushed && rs.mapped_len == self.write_pos && rs.mmap.is_some() {
            return;
        }

        if !rs.writer_flushed {
            self.writer
                .borrow_mut()
                .flush()
                .expect("disk successor graph: flush");
            rs.writer_flushed = true;
        }

        if self.write_pos == 0 {
            rs.mmap = None;
            rs.mapped_len = 0;
            return;
        }

        let mmap = unsafe { Mmap::map(self.backing_file.as_file()) }
            .expect("disk successor graph: map file");
        rs.mmap = Some(mmap);
        rs.mapped_len = self.write_pos;
    }

    fn cached_or_record_successor_count(&self, parent_fp: Fingerprint, offset: u64) -> usize {
        {
            let rs = self.read_state.borrow();
            if let Some(count) = self.cache_successor_count(&rs, parent_fp, offset) {
                return count;
            }
        }

        let mut rs = self.read_state.borrow_mut();
        self.ensure_mmap_current(&mut rs);
        let mmap = rs
            .mmap
            .as_ref()
            .expect("disk successor graph: mmap unavailable for overwrite accounting");
        Self::read_successor_count_from_mmap(mmap, offset, parent_fp)
    }

    fn read_successor_count_from_mmap(mmap: &Mmap, offset: u64, parent_fp: Fingerprint) -> usize {
        let (succ_count, _) = Self::record_layout(mmap.as_ref(), offset, parent_fp);
        succ_count
    }

    fn read_successors_from_mmap(
        mmap: &Mmap,
        offset: u64,
        parent_fp: Fingerprint,
    ) -> Vec<Fingerprint> {
        let bytes = mmap.as_ref();
        let (succ_count, succ_start) = Self::record_layout(bytes, offset, parent_fp);
        let mut successors = Vec::with_capacity(succ_count);
        for idx in 0..succ_count {
            let word_start = succ_start + idx * 8;
            let word_end = word_start + 8;
            let raw = bytes
                .get(word_start..word_end)
                .expect("disk successor graph: successor record truncated");
            successors.push(Fingerprint(u64::from_le_bytes(
                raw.try_into()
                    .expect("disk successor graph: successor width"),
            )));
        }
        successors
    }

    fn record_layout(bytes: &[u8], offset: u64, parent_fp: Fingerprint) -> (usize, usize) {
        let start = usize::try_from(offset).expect("disk successor graph: offset exceeds usize");
        let parent_end = start
            .checked_add(8)
            .expect("disk successor graph: parent offset overflow");
        let count_end = parent_end
            .checked_add(4)
            .expect("disk successor graph: count offset overflow");
        let parent_raw = bytes
            .get(start..parent_end)
            .expect("disk successor graph: parent record truncated");
        let stored_parent = Fingerprint(u64::from_le_bytes(
            parent_raw
                .try_into()
                .expect("disk successor graph: parent width"),
        ));
        assert_eq!(
            stored_parent, parent_fp,
            "disk successor graph: index mismatch at offset {offset}"
        );
        let count_raw = bytes
            .get(parent_end..count_end)
            .expect("disk successor graph: count record truncated");
        let succ_count = u32::from_le_bytes(
            count_raw
                .try_into()
                .expect("disk successor graph: count width"),
        ) as usize;
        let succ_bytes = succ_count
            .checked_mul(8)
            .expect("disk successor graph: successor byte overflow");
        let end = count_end
            .checked_add(succ_bytes)
            .expect("disk successor graph: record end overflow");
        assert!(
            end <= bytes.len(),
            "disk successor graph: record at offset {offset} exceeds mapped file"
        );
        (succ_count, count_end)
    }

    #[cfg(test)]
    fn cache_contains(&self, fp: Fingerprint) -> bool {
        let rs = self.read_state.borrow();
        let Some(&offset) = self.index.get(&fp) else {
            return false;
        };
        self.cache_successor_count(&rs, fp, offset).is_some()
    }

    #[cfg(test)]
    fn mapped_len(&self) -> u64 {
        self.read_state.borrow().mapped_len
    }

    #[cfg(test)]
    fn cache_len(&self) -> usize {
        self.read_state
            .borrow()
            .cache
            .iter()
            .filter(|entry| entry.is_some())
            .count()
    }

    #[cfg(test)]
    pub(crate) fn cache_slots(&self) -> usize {
        self.cache_capacity
    }

    #[cfg(test)]
    fn clear_cache_for_test(&self) {
        self.clear_cache_slots(&mut self.read_state.borrow_mut());
    }

    fn clear_cache_slots(&self, rs: &mut ReadState) {
        for slot in &mut rs.cache {
            *slot = None;
        }
    }

    #[cfg(test)]
    fn file_len(&self) -> u64 {
        self.write_pos
    }

    /// Number of distinct parent entries.
    pub(crate) fn len(&self) -> usize {
        self.entry_count
    }

    /// Whether the resident offset index contains a parent entry.
    ///
    /// This does not flush the writer, map the backing file, or populate the
    /// read cache.
    pub(super) fn contains_parent(&self, fp: &Fingerprint) -> bool {
        self.index.contains_key(fp)
    }

    /// Total successor fingerprints stored (for diagnostics).
    pub(crate) fn total_successors(&self) -> usize {
        self.total_successors
    }

    /// Collect all fingerprints referenced by the graph.
    ///
    /// This is a cold-path helper for fp-only liveness replay fallback. It
    /// walks the parent index and reads each successor list through the normal
    /// mmap/cache path so callers can recover the full expected fingerprint set
    /// without depending on the in-memory HashMap backend.
    pub(crate) fn collect_all_fingerprints(&self) -> FxHashSet<Fingerprint> {
        let mut fingerprints = FxHashSet::with_capacity_and_hasher(
            self.entry_count.saturating_add(self.total_successors),
            Default::default(),
        );
        for &parent_fp in self.index.keys() {
            fingerprints.insert(parent_fp);
            if let Some(successors) = self.get(&parent_fp) {
                fingerprints.extend(successors);
            }
        }
        fingerprints
    }

    /// Discard all entries (resets file and index).
    pub(crate) fn clear(&mut self) {
        {
            let mut rs = self.read_state.borrow_mut();
            rs.mmap = None;
            rs.mapped_len = 0;
            self.clear_cache_slots(&mut rs);
            rs.writer_flushed = true;
        }

        let mut writer = self.writer.borrow_mut();
        writer.flush().ok();
        // Truncate the file to zero.
        writer
            .get_ref()
            .set_len(0)
            .expect("disk successor graph: truncate");
        writer
            .seek(SeekFrom::Start(0))
            .expect("disk successor graph: seek to start");
        drop(writer);

        self.write_pos = 0;
        self.index.clear();
        self.entry_count = 0;
        self.total_successors = 0;
    }
}

#[cfg(test)]
#[path = "disk_successor_graph_tests.rs"]
mod tests;
