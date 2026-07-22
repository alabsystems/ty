// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Mmap-backed node pointer table for disk-backed liveness graphs.
//!
//! Maps `(Fingerprint, tableau_idx)` composite keys to `u64` node record
//! offsets. This is the TY equivalent of TLC's `TableauNodePtrTable`.
//!
//! # Layout
//!
//! Each entry is 32 bytes (4 × u64):
//! - Word 0: encoded fingerprint (EMPTY=0 sentinel, MSB reserved)
//! - Word 1: tableau index as u64 (meaningful only when fp ≠ EMPTY)
//! - Word 2: node record offset
//! - Word 3: dense contiguous node id (`u32` stored in a `u64` word)
//!
//! The dense id (Word 3) is the node's insertion position in
//! [`DiskGraphStore::all_nodes`](super::disk_graph::DiskGraphStore), assigned
//! once at first insert and STABLE across later record rewrites. It lets the
//! Tarjan SCC pass index its per-node arrays directly by a contiguous `u32`
//! WITHOUT a second parallel `node_to_id` hash map on top of the disk-backed
//! graph (`cf1s-tarjan` memory win). It is preserved verbatim on update so the
//! id never changes for a live node.
//!
//! Open-addressing with linear probing. The composite key hash combines
//! state fingerprint and tableau index for probe start position.
//!
//! # Design notes
//!
//! - Not concurrent: liveness graph construction is sequential (single-threaded
//!   BFS + SCC post-pass). If parallelism is added later, atomics are needed.
//! - FP(0) is handled via a separate small Vec, matching the existing pattern
//!   in `storage/trace.rs` and the `storage/mmap/` directory module.
//! - Reuses `encode_fingerprint` / `EMPTY` / `FP_MASK` from `storage/open_addressing.rs`.
//!
//! Part of #2732 Slice C.

use crate::state::Fingerprint;
use crate::storage::open_addressing::{encode_fingerprint, EMPTY, FP_MASK, MAX_PROBE};
use memmap2::MmapMut;
use std::io;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Entry size: 32 bytes (4 × u64).
const ENTRY_SIZE: usize = 32;

/// Number of u64 words per entry.
const WORDS_PER_ENTRY: usize = 4;

/// Mmap-backed open-addressed table mapping `(Fingerprint, tableau_idx)` → `u64`
/// node offset.
///
/// Designed for the liveness checker's graph node pointer index. Each unique
/// `(state_fp, tableau_idx)` pair gets one slot storing the node record's byte
/// offset in the append-only node record file (Slice D).
pub(crate) struct NodePtrTable {
    /// Memory-mapped array of `(encoded_fp, tidx, offset, dense_id)` 4-tuples.
    mmap: MmapMut,
    /// Number of slots (not bytes).
    capacity: usize,
    /// Number of occupied mmap slots.
    slot_count: usize,
    /// Backing file (kept alive for the mapping lifetime).
    _backing_file: Option<NamedTempFile>,
    /// Directory for the file-backed mapping (`None` = anonymous). Retained so
    /// [`Self::grow`] can allocate a larger replacement mapping.
    backing_dir: Option<PathBuf>,
    /// Load factor at which an insert grows (rehashes) the table.
    max_load_factor: f64,
    /// Side-channel for Fingerprint(0) entries. Vec of `(tableau_idx, offset,
    /// dense_id)`. Typically tiny (≤ tableau node count, usually < 20).
    zero_entries: Vec<(usize, u64, u32)>,
}

impl NodePtrTable {
    /// Allocate a zeroed mmap of `capacity` slots (file-backed if `backing_dir`
    /// is `Some`, else anonymous). A freshly `set_len` file / anon map reads as
    /// zeros, so every slot starts `EMPTY`.
    fn alloc_mmap(
        capacity: usize,
        backing_dir: Option<&Path>,
    ) -> io::Result<(MmapMut, Option<NamedTempFile>)> {
        let byte_size = capacity.checked_mul(ENTRY_SIZE).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "node pointer table capacity overflow: {capacity} * {ENTRY_SIZE} exceeds usize"
                ),
            )
        })?;
        if let Some(dir) = backing_dir {
            let file = NamedTempFile::new_in(dir)?;
            file.as_file().set_len(byte_size as u64)?;
            // SAFETY: the file is resized to `byte_size` and the returned
            // `NamedTempFile` keeps it alive for the mapping lifetime.
            let mmap = unsafe { MmapMut::map_mut(file.as_file())? };
            Ok((mmap, Some(file)))
        } else {
            let mmap = MmapMut::map_anon(byte_size)?;
            Ok((mmap, None))
        }
    }

    /// Create a new node pointer table.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial number of slots. The table now GROWS (rehashes)
    ///   on demand when the load factor is reached, so this is a starting size,
    ///   not a hard cap: a modest initial capacity right-sizes memory for small
    ///   behavior graphs while remaining correct for large ones (`cf1s-tarjan`
    ///   ptr-table right-sizing).
    /// * `backing_dir` - If `Some(path)`, create a file-backed mapping.
    ///   If `None`, use anonymous mapping (RAM-only).
    pub(crate) fn new(capacity: usize, backing_dir: Option<PathBuf>) -> io::Result<Self> {
        assert!(capacity > 0, "capacity must be non-zero");

        let (mmap, backing_file) = Self::alloc_mmap(capacity, backing_dir.as_deref())?;

        Ok(Self {
            mmap,
            capacity,
            slot_count: 0,
            _backing_file: backing_file,
            backing_dir,
            max_load_factor: 0.75,
            zero_entries: Vec::new(),
        })
    }

    /// Read the fingerprint word at `slot_index`.
    #[inline]
    fn fp_word(&self, slot_index: usize) -> u64 {
        debug_assert!(slot_index < self.capacity);
        let offset = slot_index * WORDS_PER_ENTRY;
        let ptr = self.mmap.as_ptr().cast::<u64>();
        // SAFETY: `slot_index < capacity` and the mmap is sized to hold
        // `capacity * WORDS_PER_ENTRY` u64 words. Reading `offset` which is
        // `slot_index * 3` is within bounds.
        unsafe { ptr.add(offset).read() }
    }

    /// Read the tableau-index word at `slot_index`.
    #[inline]
    fn tidx_word(&self, slot_index: usize) -> u64 {
        debug_assert!(slot_index < self.capacity);
        let offset = slot_index * WORDS_PER_ENTRY + 1;
        let ptr = self.mmap.as_ptr().cast::<u64>();
        // SAFETY: same bounds argument as `fp_word`, offset+1 is the second
        // word of the same entry.
        unsafe { ptr.add(offset).read() }
    }

    /// Read the node-offset word at `slot_index`.
    #[inline]
    fn offset_word(&self, slot_index: usize) -> u64 {
        debug_assert!(slot_index < self.capacity);
        let offset = slot_index * WORDS_PER_ENTRY + 2;
        let ptr = self.mmap.as_ptr().cast::<u64>();
        // SAFETY: same bounds argument, offset+2 is the third word.
        unsafe { ptr.add(offset).read() }
    }

    /// Read the dense-id word at `slot_index`.
    #[inline]
    fn dense_id_word(&self, slot_index: usize) -> u64 {
        debug_assert!(slot_index < self.capacity);
        let offset = slot_index * WORDS_PER_ENTRY + 3;
        let ptr = self.mmap.as_ptr().cast::<u64>();
        // SAFETY: same bounds argument, offset+3 is the fourth word.
        unsafe { ptr.add(offset).read() }
    }

    /// Write all four words for a slot.
    #[inline]
    fn write_slot(
        &mut self,
        slot_index: usize,
        encoded_fp: u64,
        tidx: u64,
        node_offset: u64,
        dense_id: u32,
    ) {
        debug_assert!(slot_index < self.capacity);
        let base = slot_index * WORDS_PER_ENTRY;
        let ptr = self.mmap.as_mut_ptr().cast::<u64>();
        // SAFETY: slot is within bounds (same argument as read methods).
        // Single-threaded access guaranteed by &mut self.
        unsafe {
            ptr.add(base).write(encoded_fp);
            ptr.add(base + 1).write(tidx);
            ptr.add(base + 2).write(node_offset);
            ptr.add(base + 3).write(dense_id as u64);
        }
    }

    /// Compute the primary hash index for a `(fp, tidx)` composite key.
    #[inline]
    fn hash_index(&self, fp: Fingerprint, tidx: usize) -> usize {
        Self::hash_index_raw(fp.0 & FP_MASK, tidx as u64, self.capacity)
    }

    /// Hash index from the already-encoded key words against an explicit
    /// capacity. The stored fp word equals `encode_fingerprint(fp) = fp.0 &
    /// FP_MASK` (no flushed bit is used in this table), so rehashing a slot from
    /// its stored `(encoded_fp, tidx)` words reproduces the SAME hash as the
    /// original `(fp, tidx)` — this is what makes [`Self::grow`] exact.
    #[inline]
    fn hash_index_raw(encoded_fp: u64, tidx_u64: u64, capacity: usize) -> usize {
        let h = encoded_fp
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(tidx_u64.wrapping_mul(0x517CC1B727220A95));
        (h as usize) % capacity
    }

    /// Double the table capacity and rehash every occupied slot into the new
    /// mapping, EXACTLY preserving each entry's `(encoded_fp, tidx, offset,
    /// dense_id)`. Only the slot position changes; keys, offsets, and dense ids
    /// are carried verbatim, so lookups, node offsets, and Tarjan arena ids are
    /// unaffected. FP(0) entries live in `zero_entries` and are untouched.
    ///
    /// Called by [`Self::insert`] when the load factor is reached, letting the
    /// table start modest and right-size to the actual node count instead of
    /// over-allocating from the `states * tableau` estimate.
    fn grow(&mut self) -> Result<(), NodePtrError> {
        let new_capacity = self.capacity.checked_mul(2).ok_or_else(|| {
            NodePtrError::GrowFailed(format!(
                "node pointer table capacity {} cannot double without overflow",
                self.capacity
            ))
        })?;

        let (mut new_mmap, new_backing) =
            Self::alloc_mmap(new_capacity, self.backing_dir.as_deref())
                .map_err(|e| NodePtrError::GrowFailed(format!("allocate grown mapping: {e}")))?;

        let new_ptr = new_mmap.as_mut_ptr().cast::<u64>();
        let old_ptr = self.mmap.as_ptr().cast::<u64>();

        // Rehash every occupied old slot into the (all-empty) new mapping.
        for old_slot in 0..self.capacity {
            // SAFETY: `old_slot < self.capacity`; the old mmap holds
            // `capacity * WORDS_PER_ENTRY` words.
            let base = old_slot * WORDS_PER_ENTRY;
            let encoded_fp = unsafe { old_ptr.add(base).read() };
            if encoded_fp == EMPTY {
                continue;
            }
            let tidx_u64 = unsafe { old_ptr.add(base + 1).read() };
            let node_offset = unsafe { old_ptr.add(base + 2).read() };
            let dense_id = unsafe { old_ptr.add(base + 3).read() };

            // Find the first empty slot in the new table. The new table holds
            // the same `slot_count` entries at ≤ half the load factor, so an
            // empty slot is guaranteed within `new_capacity` probes.
            let start = Self::hash_index_raw(encoded_fp, tidx_u64, new_capacity);
            let mut placed = false;
            for probe in 0..new_capacity {
                let idx = (start + probe) % new_capacity;
                let nbase = idx * WORDS_PER_ENTRY;
                // SAFETY: `idx < new_capacity`; the new mmap holds
                // `new_capacity * WORDS_PER_ENTRY` words.
                if unsafe { new_ptr.add(nbase).read() } == EMPTY {
                    unsafe {
                        new_ptr.add(nbase).write(encoded_fp);
                        new_ptr.add(nbase + 1).write(tidx_u64);
                        new_ptr.add(nbase + 2).write(node_offset);
                        new_ptr.add(nbase + 3).write(dense_id);
                    }
                    placed = true;
                    break;
                }
            }
            debug_assert!(placed, "grown table must have room for every entry");
            if !placed {
                return Err(NodePtrError::GrowFailed(format!(
                    "no empty slot for entry during rehash into capacity {new_capacity}"
                )));
            }
        }

        self.mmap = new_mmap;
        self._backing_file = new_backing;
        self.capacity = new_capacity;
        Ok(())
    }

    /// Insert or update a `(fp, tidx) → (node_offset, dense_id)` mapping.
    ///
    /// `dense_id` is the node's stable contiguous id (its
    /// [`DiskGraphStore::all_nodes`](super::disk_graph::DiskGraphStore) insert
    /// position). On an update the caller MUST pass the node's existing dense
    /// id so it is preserved unchanged (the record offset may move, the id must
    /// not). On a fresh insert it is the newly assigned id.
    ///
    /// Returns `Ok(true)` if newly inserted, `Ok(false)` if updated, or `Err`
    /// only if the table cannot grow (capacity overflow / mmap allocation
    /// failure). A full table or probe-limit cluster GROWS and retries rather
    /// than failing, so a modest initial capacity is safe for any node count.
    pub(crate) fn insert(
        &mut self,
        fp: Fingerprint,
        tidx: usize,
        node_offset: u64,
        dense_id: u32,
    ) -> Result<bool, NodePtrError> {
        // Handle FP(0) via side-channel.
        if fp.0 & FP_MASK == 0 {
            for entry in &mut self.zero_entries {
                if entry.0 == tidx {
                    entry.1 = node_offset;
                    // Preserve the stable dense id on update (caller passes it
                    // back unchanged; assigning it here is a no-op rewrite).
                    entry.2 = dense_id;
                    return Ok(false);
                }
            }
            self.zero_entries.push((tidx, node_offset, dense_id));
            return Ok(true);
        }

        let encoded = encode_fingerprint(fp);
        let tidx_u64 = tidx as u64;

        // Grow-and-retry: a new insert that would exceed the load factor, or a
        // probe run that clusters past MAX_PROBE without finding the key or an
        // empty slot, grows (rehashes) the table and retries from the new hash.
        // Each grow doubles capacity (halving the load), so this terminates.
        loop {
            let start = self.hash_index(fp, tidx);

            for probe in 0..MAX_PROBE {
                let idx = (start + probe) % self.capacity;
                let slot_fp = self.fp_word(idx);

                if slot_fp == encoded && self.tidx_word(idx) == tidx_u64 {
                    // Exact key match — update in place (offset moves, dense id
                    // preserved by the caller). Updates never need to grow.
                    self.write_slot(idx, encoded, tidx_u64, node_offset, dense_id);
                    return Ok(false);
                }

                if slot_fp == EMPTY {
                    if self.slot_count as f64 / self.capacity as f64 >= self.max_load_factor {
                        // At the load-factor limit: grow before claiming so the
                        // table stays sparse (bounded probe lengths).
                        break;
                    }
                    // Claim empty slot.
                    self.write_slot(idx, encoded, tidx_u64, node_offset, dense_id);
                    self.slot_count += 1;
                    return Ok(true);
                }
                // Different key — keep probing; if the window is exhausted the
                // loop falls through to grow-and-retry below.
            }

            self.grow()?;
        }
    }

    /// Look up the node offset for a `(fp, tidx)` key.
    pub(crate) fn get(&self, fp: Fingerprint, tidx: usize) -> Option<u64> {
        // FP(0) side-channel.
        if fp.0 & FP_MASK == 0 {
            return self.zero_entries.iter().find(|e| e.0 == tidx).map(|e| e.1);
        }

        let encoded = encode_fingerprint(fp);
        let tidx_u64 = tidx as u64;
        let start = self.hash_index(fp, tidx);

        for probe in 0..MAX_PROBE {
            let idx = (start + probe) % self.capacity;
            let slot_fp = self.fp_word(idx);

            if slot_fp == EMPTY {
                return None;
            }

            if slot_fp == encoded && self.tidx_word(idx) == tidx_u64 {
                return Some(self.offset_word(idx));
            }
        }

        None
    }

    /// Look up the stable dense node id for a `(fp, tidx)` key.
    ///
    /// Pure in-RAM (mmap) lookup — never touches the node record file — so the
    /// Tarjan SCC pass can resolve a successor `(fp, tidx)` to a contiguous
    /// `u32` arena index without a disk read or a parallel `node_to_id` map.
    pub(crate) fn get_dense_id(&self, fp: Fingerprint, tidx: usize) -> Option<u32> {
        // FP(0) side-channel.
        if fp.0 & FP_MASK == 0 {
            return self.zero_entries.iter().find(|e| e.0 == tidx).map(|e| e.2);
        }

        let encoded = encode_fingerprint(fp);
        let tidx_u64 = tidx as u64;
        let start = self.hash_index(fp, tidx);

        for probe in 0..MAX_PROBE {
            let idx = (start + probe) % self.capacity;
            let slot_fp = self.fp_word(idx);

            if slot_fp == EMPTY {
                return None;
            }

            if slot_fp == encoded && self.tidx_word(idx) == tidx_u64 {
                return Some(self.dense_id_word(idx) as u32);
            }
        }

        None
    }

    /// Check if a `(fp, tidx)` key is present.
    pub(crate) fn contains(&self, fp: Fingerprint, tidx: usize) -> bool {
        self.get(fp, tidx).is_some()
    }

    /// Number of entries in the table.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.slot_count + self.zero_entries.len()
    }

    /// Check if the table is empty.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.slot_count == 0 && self.zero_entries.is_empty()
    }

    /// Current load factor.
    #[cfg(test)]
    pub(crate) fn load_factor(&self) -> f64 {
        self.slot_count as f64 / self.capacity as f64
    }

    /// Flush mmap writes to disk.
    #[cfg(test)]
    pub(crate) fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

/// Errors from node pointer table operations.
///
/// The table grows (rehashes) on demand instead of failing when full, so the
/// only failure mode is an inability to grow: capacity would overflow `usize`,
/// or the OS refused the larger mmap allocation. Either is fatal and surfaces
/// as a liveness RuntimeFailure (inconclusive), never a wrong verdict.
#[derive(Debug, Clone)]
pub(crate) enum NodePtrError {
    /// The table could not be grown/rehashed (capacity overflow or mmap
    /// allocation failure).
    GrowFailed(String),
}

impl std::fmt::Display for NodePtrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodePtrError::GrowFailed(reason) => {
                write!(f, "node pointer table grow failed: {reason}")
            }
        }
    }
}

impl std::error::Error for NodePtrError {}

#[cfg(test)]
#[path = "node_ptr_table_tests.rs"]
mod tests;
