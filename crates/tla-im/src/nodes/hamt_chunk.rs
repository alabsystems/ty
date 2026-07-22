// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Fixed 32-slot sparse chunk for HAMT nodes.
//!
//! TY divergence from upstream `im` (which uses
//! `sized_chunks::SparseChunk<A, HashWidth>` here): upstream occupancy scans go
//! through `bitmaps::Iter::next`, which probes every one of the 32 slot
//! positions bit-by-bit and recurses once per empty slot. HAMT nodes are
//! sparse (typically 2-8 occupied slots out of 32), and the scan runs on every
//! node **iteration, clone and drop** — profiled at ~3% of single-threaded
//! model-checker wall time on operator-heavy specs (btree). This replacement
//! keeps the exact `SparseChunk` semantics — slot-indexed storage, ascending
//! slot-index iteration order, identical insert/remove/get behaviour and
//! memory layout — but drives all occupancy scans with `trailing_zeros` skips
//! over a plain `u32` bitmap, so the cost is proportional to the number of
//! occupied slots instead of the chunk capacity.
//!
//! The capacity is fixed at 32 = 2^HashLevelSize; `nodes::hamt` asserts this
//! against `HASH_WIDTH` at compile time.

use std::mem::{self, MaybeUninit};
use std::ops::Index;
use std::ptr;

/// Chunk capacity. Must equal `nodes::hamt::HASH_WIDTH` (asserted there).
pub(crate) const CAPACITY: usize = 32;

/// A fixed 32-slot sparse array with a `u32` occupancy bitmap.
///
/// Drop-in replacement for `sized_chunks::SparseChunk<A, U32>`: same
/// semantics, `trailing_zeros`-driven occupancy scans.
pub(crate) struct HamtChunk<A> {
    /// Occupancy bitmap: bit `i` set ⇔ `data[i]` is initialized.
    map: u32,
    data: [MaybeUninit<A>; CAPACITY],
}

impl<A> HamtChunk<A> {
    /// Construct a new empty chunk.
    #[inline]
    #[allow(unsafe_code)]
    pub(crate) fn new() -> Self {
        Self {
            map: 0,
            // SAFETY: an array of `MaybeUninit` does not require initialization.
            data: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    /// Construct a new chunk with one item.
    #[inline]
    pub(crate) fn unit(index: usize, value: A) -> Self {
        let mut chunk = Self::new();
        chunk.insert(index, value);
        chunk
    }

    /// Construct a new chunk with two items.
    #[inline]
    pub(crate) fn pair(index1: usize, value1: A, index2: usize, value2: A) -> Self {
        let mut chunk = Self::new();
        chunk.insert(index1, value1);
        chunk.insert(index2, value2);
        chunk
    }

    /// Get the number of occupied slots in the chunk.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.map.count_ones() as usize
    }

    #[inline]
    fn contains(&self, index: usize) -> bool {
        debug_assert!(index < CAPACITY);
        self.map & (1u32 << index) != 0
    }

    /// Insert a new value at a given index.
    ///
    /// Returns the previous value at that index, if any.
    #[allow(unsafe_code)]
    pub(crate) fn insert(&mut self, index: usize, value: A) -> Option<A> {
        assert!(index < CAPACITY, "HamtChunk::insert: index out of bounds");
        let bit = 1u32 << index;
        if self.map & bit != 0 {
            // SAFETY: the occupancy bit is set, so the slot is initialized.
            Some(mem::replace(
                unsafe { &mut *self.data[index].as_mut_ptr() },
                value,
            ))
        } else {
            // Previous contents are uninitialized — write without dropping.
            self.data[index] = MaybeUninit::new(value);
            self.map |= bit;
            None
        }
    }

    /// Remove the value at a given index.
    ///
    /// Returns the value, or `None` if the index had no value.
    #[allow(unsafe_code)]
    pub(crate) fn remove(&mut self, index: usize) -> Option<A> {
        assert!(index < CAPACITY, "HamtChunk::remove: index out of bounds");
        let bit = 1u32 << index;
        if self.map & bit != 0 {
            self.map &= !bit;
            // SAFETY: the slot was occupied, hence initialized; the bit is now
            // cleared so ownership moves out exactly once.
            Some(unsafe { self.data[index].as_ptr().read() })
        } else {
            None
        }
    }

    /// Remove the first value present in the chunk.
    ///
    /// Returns the value that was removed, or `None` if the chunk was empty.
    #[inline]
    pub(crate) fn pop(&mut self) -> Option<A> {
        self.first_index().and_then(|index| self.remove(index))
    }

    /// Get the value at a given index.
    #[inline]
    #[allow(unsafe_code)]
    pub(crate) fn get(&self, index: usize) -> Option<&A> {
        if index < CAPACITY && self.contains(index) {
            // SAFETY: the occupancy bit is set, so the slot is initialized.
            Some(unsafe { &*self.data[index].as_ptr() })
        } else {
            None
        }
    }

    /// Get a mutable reference to the value at a given index.
    #[inline]
    #[allow(unsafe_code)]
    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut A> {
        if index < CAPACITY && self.contains(index) {
            // SAFETY: the occupancy bit is set, so the slot is initialized.
            Some(unsafe { &mut *self.data[index].as_mut_ptr() })
        } else {
            None
        }
    }

    /// Find the first index which contains a value.
    #[inline]
    pub(crate) fn first_index(&self) -> Option<usize> {
        if self.map == 0 {
            None
        } else {
            Some(self.map.trailing_zeros() as usize)
        }
    }

    /// Make an iterator over the indices which contain values, in ascending
    /// order (matching `SparseChunk::indices`).
    #[inline]
    pub(crate) fn indices(&self) -> IndexIter {
        IndexIter { map: self.map }
    }

    /// Make an iterator of references to the values contained in the chunk,
    /// in ascending slot-index order (matching `SparseChunk::iter`).
    #[inline]
    pub(crate) fn iter(&self) -> Iter<'_, A> {
        Iter {
            map: self.map,
            chunk: self,
        }
    }

    /// Make an iterator of mutable references to the values contained in the
    /// chunk, in ascending slot-index order (matching `SparseChunk::iter_mut`).
    #[inline]
    pub(crate) fn iter_mut(&mut self) -> IterMut<'_, A> {
        IterMut {
            map: self.map,
            chunk: self,
        }
    }
}

impl<A: Clone> Clone for HamtChunk<A> {
    #[allow(unsafe_code)]
    fn clone(&self) -> Self {
        let mut out = Self::new();
        let mut map = self.map;
        while map != 0 {
            let index = map.trailing_zeros() as usize;
            map &= map - 1;
            // SAFETY: bit set in the source map ⇒ source slot initialized;
            // the destination slot is uninitialized, so plain write (no drop).
            unsafe {
                out.data[index]
                    .as_mut_ptr()
                    .write((*self.data[index].as_ptr()).clone());
            }
            // Claim the bit only after the slot is written, so a panicking
            // `A::clone` leaves `out` droppable (initialized slots only).
            out.map |= 1u32 << index;
        }
        out
    }
}

impl<A> Drop for HamtChunk<A> {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if mem::needs_drop::<A>() {
            let mut map = self.map;
            while map != 0 {
                let index = map.trailing_zeros() as usize;
                map &= map - 1;
                // SAFETY: bit set ⇒ slot initialized; dropped exactly once.
                unsafe { ptr::drop_in_place(self.data[index].as_mut_ptr()) };
            }
        }
    }
}

impl<A> Index<usize> for HamtChunk<A> {
    type Output = A;

    #[inline]
    fn index(&self, index: usize) -> &A {
        self.get(index)
            .expect("HamtChunk::index: index out of bounds")
    }
}

/// Iterator over occupied slot indices, ascending.
pub(crate) struct IndexIter {
    map: u32,
}

impl Iterator for IndexIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.map == 0 {
            return None;
        }
        let index = self.map.trailing_zeros() as usize;
        self.map &= self.map - 1; // clear lowest set bit
        Some(index)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.map.count_ones() as usize;
        (n, Some(n))
    }
}

/// Iterator of `&A` over occupied slots, ascending slot-index order.
pub(crate) struct Iter<'a, A> {
    map: u32,
    chunk: &'a HamtChunk<A>,
}

impl<'a, A> Iterator for Iter<'a, A> {
    type Item = &'a A;

    #[inline]
    #[allow(unsafe_code)]
    fn next(&mut self) -> Option<&'a A> {
        if self.map == 0 {
            return None;
        }
        let index = self.map.trailing_zeros() as usize;
        self.map &= self.map - 1;
        // SAFETY: bit was set in the chunk's occupancy map at iterator
        // construction and the chunk is borrowed shared for 'a, so the slot
        // is initialized and stays valid.
        Some(unsafe { &*self.chunk.data[index].as_ptr() })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.map.count_ones() as usize;
        (n, Some(n))
    }
}

/// Iterator of `&mut A` over occupied slots, ascending slot-index order.
pub(crate) struct IterMut<'a, A> {
    map: u32,
    chunk: &'a mut HamtChunk<A>,
}

impl<'a, A> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;

    #[inline]
    #[allow(unsafe_code)]
    fn next(&mut self) -> Option<&'a mut A> {
        if self.map == 0 {
            return None;
        }
        let index = self.map.trailing_zeros() as usize;
        self.map &= self.map - 1;
        // SAFETY: each occupied index is yielded at most once, so no two
        // returned `&mut A` alias; the chunk is borrowed exclusively for 'a.
        // Lifetime extension mirrors `sized_chunks::sparse_chunk::IterMut`.
        unsafe {
            let p: *mut A = self.chunk.data[index].as_mut_ptr();
            Some(&mut *p)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.map.count_ones() as usize;
        (n, Some(n))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn insert_get_remove_roundtrip() {
        let mut chunk: HamtChunk<String> = HamtChunk::new();
        assert_eq!(chunk.len(), 0);
        assert_eq!(chunk.first_index(), None);
        assert_eq!(chunk.insert(5, "five".to_string()), None);
        assert_eq!(chunk.insert(0, "zero".to_string()), None);
        assert_eq!(chunk.insert(31, "thirty-one".to_string()), None);
        assert_eq!(chunk.len(), 3);
        assert_eq!(chunk.first_index(), Some(0));
        assert_eq!(chunk.get(5).map(String::as_str), Some("five"));
        assert_eq!(chunk.get(6), None);
        assert_eq!(
            chunk.insert(5, "FIVE".to_string()),
            Some("five".to_string())
        );
        assert_eq!(chunk.remove(5), Some("FIVE".to_string()));
        assert_eq!(chunk.remove(5), None);
        assert_eq!(chunk.len(), 2);
        // pop removes the first occupied slot.
        assert_eq!(chunk.pop(), Some("zero".to_string()));
        assert_eq!(chunk.pop(), Some("thirty-one".to_string()));
        assert_eq!(chunk.pop(), None);
    }

    #[test]
    fn iteration_is_ascending_index_order() {
        let mut chunk: HamtChunk<usize> = HamtChunk::new();
        for &i in &[17usize, 3, 31, 0, 8, 24] {
            chunk.insert(i, i * 10);
        }
        let indices: Vec<usize> = chunk.indices().collect();
        assert_eq!(indices, vec![0, 3, 8, 17, 24, 31]);
        let values: Vec<usize> = chunk.iter().copied().collect();
        assert_eq!(values, vec![0, 30, 80, 170, 240, 310]);
        for v in chunk.iter_mut() {
            *v += 1;
        }
        let values: Vec<usize> = chunk.iter().copied().collect();
        assert_eq!(values, vec![1, 31, 81, 171, 241, 311]);
    }

    #[test]
    fn clone_is_deep_and_order_preserving() {
        let mut chunk: HamtChunk<String> = HamtChunk::new();
        chunk.insert(2, "a".to_string());
        chunk.insert(9, "b".to_string());
        let cloned = chunk.clone();
        assert_eq!(
            cloned.iter().cloned().collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()]
        );
        drop(chunk);
        assert_eq!(cloned.get(9).map(String::as_str), Some("b"));
    }

    #[test]
    fn drop_runs_for_occupied_slots_only() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Counter;
        impl Drop for Counter {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        {
            let mut chunk: HamtChunk<Counter> = HamtChunk::new();
            chunk.insert(1, Counter);
            chunk.insert(30, Counter);
            let removed = chunk.remove(1);
            drop(removed); // 1 drop
        } // chunk drop: 1 more
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn index_op_matches_get() {
        let mut chunk: HamtChunk<u8> = HamtChunk::new();
        chunk.insert(7, 42);
        assert_eq!(chunk[7], 42);
    }

    #[test]
    #[should_panic]
    fn index_op_panics_on_empty_slot() {
        let chunk: HamtChunk<u8> = HamtChunk::new();
        let _ = chunk[7];
    }
}
