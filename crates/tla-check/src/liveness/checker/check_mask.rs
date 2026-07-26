// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Multi-word bitmask for liveness check indices (#2890).
//!
//! Replaces the previous `u64` bitmask that capped specs at 64 action/state
//! checks. TLC uses `BitVector` (backed by `long[]`) for the same purpose,
//! supporting arbitrary numbers of check expressions. This type provides
//! the same capability with O(1) per-bit access.
//!
//! For the common case (≤64 checks), this is a single inline `u64` word
//! (`SmallVec<[u64; 1]>`), matching the previous `u64` performance with no
//! heap allocation.

use smallvec::SmallVec;
use std::fmt;

/// Inline storage for the common ≤64-check case (single `u64` word).
type Words = SmallVec<[u64; 1]>;

/// Multi-word bitmask supporting arbitrary numbers of check indices.
///
/// Mirrors TLC's `tlc2.util.BitVector` which uses `long[]` internally.
/// Auto-grows when bits beyond the current capacity are set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CheckMask {
    words: Words,
}

impl Default for CheckMask {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckMask {
    /// Create an empty (all-zero) bitmask.
    pub(crate) fn new() -> Self {
        Self {
            words: SmallVec::new(),
        }
    }

    /// Set bit at index `idx`. Grows the internal storage as needed.
    #[inline]
    pub(crate) fn set(&mut self, idx: usize) {
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        if word_idx >= self.words.len() {
            self.words.resize(word_idx + 1, 0);
        }
        self.words[word_idx] |= 1u64 << bit_idx;
    }

    /// Check if bit at index `idx` is set.
    #[inline]
    pub(crate) fn get(&self, idx: usize) -> bool {
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        word_idx < self.words.len() && (self.words[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Check if all bits set in `required` are also set in `self`.
    /// Returns true if `self` is a superset of `required`.
    #[inline]
    pub(crate) fn contains_all(&self, required: &CheckMask) -> bool {
        for (i, &req_word) in required.words.iter().enumerate() {
            if req_word == 0 {
                continue;
            }
            let self_word = self.words.get(i).copied().unwrap_or(0);
            if (self_word & req_word) != req_word {
                return false;
            }
        }
        true
    }

    /// Bitwise OR-assign: set all bits that are set in `other`.
    ///
    /// Used to accumulate aggregate masks across SCC nodes.
    #[inline]
    pub(crate) fn or_assign(&mut self, other: &CheckMask) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (i, &w) in other.words.iter().enumerate() {
            self.words[i] |= w;
        }
    }

    /// Check if the mask has no bits set.
    #[cfg(test)]
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Build a mask from a list of indices.
    pub(crate) fn from_indices(indices: &[usize]) -> Self {
        let mut mask = Self::new();
        for &idx in indices {
            mask.set(idx);
        }
        mask
    }

    /// Create a CheckMask from a raw `u64` value (for test compatibility).
    /// Bit `i` set in `raw` means index `i` is set in the mask.
    #[cfg(test)]
    pub(crate) fn from_u64(raw: u64) -> Self {
        if raw == 0 {
            Self::new()
        } else {
            Self {
                words: smallvec::smallvec![raw],
            }
        }
    }

    /// Access the raw word storage for serialization.
    pub(crate) fn as_words(&self) -> &[u64] {
        &self.words
    }

    /// Construct from raw words (deserialization).
    pub(crate) fn from_words(words: Vec<u64>) -> Self {
        Self {
            words: SmallVec::from_vec(words),
        }
    }
}

/// Inline storage for the packed action-check matrix.
///
/// Two words cover the common case of a node with a handful of outgoing
/// edges and action checks without allocating a separate mask per edge.
type ActionWords = SmallVec<[u64; 2]>;

/// Shape-validation failure for [`ActionCheckMatrix::from_raw_parts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActionCheckMatrixShapeError {
    EdgeCountTooLarge {
        edge_count: usize,
    },
    CheckCountTooLarge {
        check_count: usize,
    },
    BitCountOverflow {
        edge_count: usize,
        check_count: usize,
    },
    WordCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonZeroTailPadding {
        tail_bits: usize,
        value: u64,
    },
    NonCanonicalCheckCount {
        check_count: usize,
    },
}

impl fmt::Display for ActionCheckMatrixShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EdgeCountTooLarge { edge_count } => {
                write!(f, "action-check edge count {edge_count} exceeds u32::MAX")
            }
            Self::CheckCountTooLarge { check_count } => {
                write!(f, "action-check count {check_count} exceeds u32::MAX")
            }
            Self::BitCountOverflow {
                edge_count,
                check_count,
            } => write!(
                f,
                "action-check matrix dimensions {edge_count}x{check_count} overflow usize"
            ),
            Self::WordCountMismatch { expected, actual } => write!(
                f,
                "action-check matrix needs {expected} packed words, got {actual}"
            ),
            Self::NonZeroTailPadding { tail_bits, value } => write!(
                f,
                "action-check matrix has nonzero padding above its {tail_bits} tail bits: {value:#x}"
            ),
            Self::NonCanonicalCheckCount { check_count } => write!(
                f,
                "action-check matrix width {check_count} has no set bit in its final column"
            ),
        }
    }
}

impl std::error::Error for ActionCheckMatrixShapeError {}

/// Edge-aligned action-check bits packed into one row-major bit matrix.
///
/// Bit `(edge, check)` lives at `edge * check_count + check`. The stored
/// `check_count` is the highest check column that is set in any row, not the
/// declared number of action checks. Trimming globally-false trailing columns
/// preserves every logical lookup while avoiding unused storage. `edge_count`
/// remains explicit, so a matrix containing only all-zero rows still has the
/// same length as its successor list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ActionCheckMatrix {
    words: ActionWords,
    edge_count: u32,
    check_count: u32,
}

impl Default for ActionCheckMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionCheckMatrix {
    /// Create a matrix with no edges and no checks.
    pub(crate) fn new() -> Self {
        Self {
            words: SmallVec::new(),
            edge_count: 0,
            check_count: 0,
        }
    }

    /// Pack edge masks using `check_count` as a validated upper bound.
    ///
    /// The retained width is trimmed to the highest bit actually set in any
    /// mask. A bit at or above `check_count` is an internal caller error and
    /// panics instead of being silently discarded.
    pub(crate) fn from_masks<I>(check_count: usize, masks: I) -> Self
    where
        I: IntoIterator<Item = CheckMask>,
    {
        // Most behavior-graph nodes have only a handful of successors. Keep
        // those transient source masks inline too, so packing does not replace
        // one permanent allocation per node with one allocate/free cycle.
        let masks: SmallVec<[CheckMask; 4]> = masks.into_iter().collect();
        let edge_count_u32 =
            u32::try_from(masks.len()).expect("action-check edge count must fit in u32");
        let _declared_check_count_u32 =
            u32::try_from(check_count).expect("action-check count must fit in u32");

        let effective_check_count = masks.iter().map(Self::mask_bit_len).max().unwrap_or(0);
        assert!(
            effective_check_count <= check_count,
            "action-check mask bit {} exceeds declared check-count upper bound {}",
            effective_check_count.saturating_sub(1),
            check_count
        );
        let effective_check_count_u32 = u32::try_from(effective_check_count)
            .expect("effective action-check count must fit in u32");
        let total_bits = masks
            .len()
            .checked_mul(effective_check_count)
            .expect("action-check matrix dimensions must fit in usize");
        let mut words = ActionWords::from_elem(0, Self::words_for_bits(total_bits));

        if effective_check_count != 0 {
            for (edge_idx, mask) in masks.iter().enumerate() {
                Self::pack_row(&mut words, edge_idx, effective_check_count, mask.as_words());
            }
        }

        Self {
            words,
            edge_count: edge_count_u32,
            check_count: effective_check_count_u32,
        }
    }

    /// Restore an already-packed matrix, validating its exact shape.
    ///
    /// Raw storage must contain exactly `ceil(edge_count * check_count / 64)`
    /// words. Unused high bits in the final word must be zero so equal logical
    /// matrices have one canonical packed representation.
    pub(crate) fn from_raw_parts(
        edge_count: usize,
        check_count: usize,
        words: Vec<u64>,
    ) -> Result<Self, ActionCheckMatrixShapeError> {
        let edge_count_u32 = u32::try_from(edge_count)
            .map_err(|_| ActionCheckMatrixShapeError::EdgeCountTooLarge { edge_count })?;
        let check_count_u32 = u32::try_from(check_count)
            .map_err(|_| ActionCheckMatrixShapeError::CheckCountTooLarge { check_count })?;
        let total_bits = edge_count.checked_mul(check_count).ok_or(
            ActionCheckMatrixShapeError::BitCountOverflow {
                edge_count,
                check_count,
            },
        )?;
        let expected_words = Self::words_for_bits(total_bits);
        if words.len() != expected_words {
            return Err(ActionCheckMatrixShapeError::WordCountMismatch {
                expected: expected_words,
                actual: words.len(),
            });
        }

        let tail_bits = total_bits % 64;
        if tail_bits != 0 {
            let tail_mask = (1u64 << tail_bits) - 1;
            let tail_value = words.last().copied().unwrap_or(0);
            if tail_value & !tail_mask != 0 {
                return Err(ActionCheckMatrixShapeError::NonZeroTailPadding {
                    tail_bits,
                    value: tail_value,
                });
            }
        }

        // `from_masks` trims globally empty trailing columns. Enforce the
        // same canonical form at the raw boundary so structural equality is
        // also logical equality.
        if check_count != 0 {
            let last_check = check_count - 1;
            let final_column_is_set = (0..edge_count).any(|edge_idx| {
                let bit_idx = edge_idx * check_count + last_check;
                words[bit_idx / 64] & (1u64 << (bit_idx % 64)) != 0
            });
            if !final_column_is_set {
                return Err(ActionCheckMatrixShapeError::NonCanonicalCheckCount { check_count });
            }
        }

        Ok(Self {
            words: SmallVec::from_vec(words),
            edge_count: edge_count_u32,
            check_count: check_count_u32,
        })
    }

    /// Number of edge rows, including rows whose every bit is zero.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.edge_count as usize
    }

    /// Whether there are no edge rows.
    #[inline]
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.edge_count == 0
    }

    /// Number of retained check columns.
    #[inline]
    pub(crate) fn check_count(&self) -> usize {
        self.check_count as usize
    }

    /// Return an edge row by index.
    #[inline]
    pub(crate) fn get(&self, edge_idx: usize) -> Option<ActionCheckRow<'_>> {
        if edge_idx >= self.len() {
            return None;
        }
        Some(ActionCheckRow {
            words: &self.words,
            bit_start: edge_idx * self.check_count(),
            check_count: self.check_count(),
        })
    }

    /// Return the first edge row.
    #[inline]
    #[cfg(test)]
    pub(crate) fn first(&self) -> Option<ActionCheckRow<'_>> {
        self.get(0)
    }

    /// Iterate over every edge row in successor order.
    #[inline]
    pub(crate) fn iter(&self) -> ActionCheckRows<'_> {
        ActionCheckRows {
            matrix: self,
            front: 0,
            back: self.len(),
        }
    }

    /// Access the contiguous row-major packed words for serialization.
    #[inline]
    pub(crate) fn as_words(&self) -> &[u64] {
        &self.words
    }

    fn mask_bit_len(mask: &CheckMask) -> usize {
        mask.as_words()
            .iter()
            .rposition(|&word| word != 0)
            .map(|word_idx| {
                word_idx * 64 + (64 - mask.as_words()[word_idx].leading_zeros() as usize)
            })
            .unwrap_or(0)
    }

    #[inline]
    fn words_for_bits(bit_count: usize) -> usize {
        bit_count / 64 + usize::from(bit_count % 64 != 0)
    }

    fn pack_row(packed: &mut [u64], edge_idx: usize, check_count: usize, source_words: &[u64]) {
        let row_start = edge_idx * check_count;
        for (source_word_idx, &source_word) in source_words.iter().enumerate() {
            let source_bit_start = source_word_idx * 64;
            if source_bit_start >= check_count {
                break;
            }
            let bits_in_word = (check_count - source_bit_start).min(64);
            let source_word = if bits_in_word == 64 {
                source_word
            } else {
                source_word & ((1u64 << bits_in_word) - 1)
            };
            if source_word == 0 {
                continue;
            }

            let destination_bit = row_start + source_bit_start;
            let destination_word = destination_bit / 64;
            let destination_shift = destination_bit % 64;
            packed[destination_word] |= source_word << destination_shift;
            if destination_shift != 0 && destination_word + 1 < packed.len() {
                packed[destination_word + 1] |= source_word >> (64 - destination_shift);
            }
        }
    }
}

impl From<Vec<CheckMask>> for ActionCheckMatrix {
    fn from(masks: Vec<CheckMask>) -> Self {
        let check_count_upper_bound = masks
            .iter()
            .map(|mask| {
                mask.as_words()
                    .len()
                    .checked_mul(64)
                    .expect("action-check source mask width must fit in usize")
            })
            .max()
            .unwrap_or(0);
        Self::from_masks(check_count_upper_bound, masks)
    }
}

impl FromIterator<CheckMask> for ActionCheckMatrix {
    fn from_iter<T: IntoIterator<Item = CheckMask>>(iter: T) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl<'a> IntoIterator for &'a ActionCheckMatrix {
    type Item = ActionCheckRow<'a>;
    type IntoIter = ActionCheckRows<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Borrowed logical view of one edge row in an [`ActionCheckMatrix`].
#[derive(Clone, Copy)]
pub(crate) struct ActionCheckRow<'a> {
    words: &'a [u64],
    bit_start: usize,
    check_count: usize,
}

impl ActionCheckRow<'_> {
    /// Check whether a check bit is set for this edge.
    #[inline]
    pub(crate) fn get(&self, check_idx: usize) -> bool {
        if check_idx >= self.check_count {
            return false;
        }
        let bit_idx = self.bit_start + check_idx;
        self.words
            .get(bit_idx / 64)
            .is_some_and(|word| word & (1u64 << (bit_idx % 64)) != 0)
    }

    /// Whether this edge row has no set check bits.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        let word_count = ActionCheckMatrix::words_for_bits(self.check_count);
        (0..word_count).all(|word_idx| self.logical_word(word_idx) == 0)
    }

    /// Check whether this row contains every bit in `required`.
    #[inline]
    pub(crate) fn contains_all(&self, required: &CheckMask) -> bool {
        required
            .as_words()
            .iter()
            .enumerate()
            .all(|(word_idx, &required_word)| {
                required_word == 0 || self.logical_word(word_idx) & required_word == required_word
            })
    }

    /// OR every set bit in this row into an owned aggregate mask.
    #[inline]
    pub(crate) fn or_into(&self, target: &mut CheckMask) {
        let row_word_count = ActionCheckMatrix::words_for_bits(self.check_count);
        if target.words.len() < row_word_count {
            target.words.resize(row_word_count, 0);
        }
        for word_idx in 0..row_word_count {
            target.words[word_idx] |= self.logical_word(word_idx);
        }
    }

    #[inline]
    fn logical_word(&self, word_idx: usize) -> u64 {
        let check_start = match word_idx.checked_mul(64) {
            Some(check_start) if check_start < self.check_count => check_start,
            _ => return 0,
        };
        let bit_idx = self.bit_start + check_start;
        let packed_word_idx = bit_idx / 64;
        let shift = bit_idx % 64;
        let mut word = self.words.get(packed_word_idx).copied().unwrap_or(0) >> shift;
        if shift != 0 {
            word |= self.words.get(packed_word_idx + 1).copied().unwrap_or(0) << (64 - shift);
        }

        let remaining = self.check_count - check_start;
        if remaining < 64 {
            word &= (1u64 << remaining) - 1;
        }
        word
    }
}

impl fmt::Debug for ActionCheckRow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.check_count).filter(|&check_idx| self.get(check_idx)))
            .finish()
    }
}

impl PartialEq for ActionCheckRow<'_> {
    fn eq(&self, other: &Self) -> bool {
        let word_count = ActionCheckMatrix::words_for_bits(self.check_count.max(other.check_count));
        (0..word_count).all(|word_idx| self.logical_word(word_idx) == other.logical_word(word_idx))
    }
}

impl Eq for ActionCheckRow<'_> {}

/// Exact-size iterator over packed action-check rows.
#[derive(Clone)]
pub(crate) struct ActionCheckRows<'a> {
    matrix: &'a ActionCheckMatrix,
    front: usize,
    back: usize,
}

impl<'a> Iterator for ActionCheckRows<'a> {
    type Item = ActionCheckRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        let edge_idx = self.front;
        self.front += 1;
        self.matrix.get(edge_idx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for ActionCheckRows<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        self.matrix.get(self.back)
    }
}

impl ExactSizeIterator for ActionCheckRows<'_> {}
impl std::iter::FusedIterator for ActionCheckRows<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_set_get() {
        let mut mask = CheckMask::new();
        assert!(!mask.get(0));
        mask.set(0);
        assert!(mask.get(0));
        assert!(!mask.get(1));
    }

    #[test]
    fn test_beyond_64() {
        let mut mask = CheckMask::new();
        mask.set(64);
        assert!(!mask.get(63));
        assert!(mask.get(64));
        assert!(!mask.get(65));

        mask.set(127);
        assert!(mask.get(127));
        assert!(!mask.get(128));
    }

    #[test]
    fn test_contains_all() {
        let mut superset = CheckMask::new();
        superset.set(0);
        superset.set(5);
        superset.set(64);
        superset.set(100);

        let mut subset = CheckMask::new();
        subset.set(5);
        subset.set(64);
        assert!(superset.contains_all(&subset));

        subset.set(99); // not in superset
        assert!(!superset.contains_all(&subset));
    }

    #[test]
    fn test_empty() {
        let mask = CheckMask::new();
        assert!(mask.is_empty());

        let mut mask2 = CheckMask::new();
        mask2.set(10);
        assert!(!mask2.is_empty());
    }

    #[test]
    fn test_from_indices() {
        let mask = CheckMask::from_indices(&[3, 7, 65, 130]);
        assert!(mask.get(3));
        assert!(mask.get(7));
        assert!(mask.get(65));
        assert!(mask.get(130));
        assert!(!mask.get(4));
        assert!(!mask.get(66));
    }

    #[test]
    fn test_contains_all_empty() {
        let mask = CheckMask::new();
        let empty = CheckMask::new();
        assert!(mask.contains_all(&empty));
    }

    #[test]
    fn test_or_assign_basic() {
        let mut a = CheckMask::new();
        a.set(0);
        a.set(5);
        let mut b = CheckMask::new();
        b.set(3);
        b.set(5);
        b.set(7);
        a.or_assign(&b);
        assert!(a.get(0));
        assert!(a.get(3));
        assert!(a.get(5));
        assert!(a.get(7));
        assert!(!a.get(1));
    }

    #[test]
    fn test_or_assign_grows() {
        let mut a = CheckMask::new();
        a.set(0);
        let mut b = CheckMask::new();
        b.set(128);
        a.or_assign(&b);
        assert!(a.get(0));
        assert!(a.get(128));
        assert!(!a.get(64));
    }

    #[test]
    fn test_or_assign_empty() {
        let mut a = CheckMask::new();
        a.set(10);
        let empty = CheckMask::new();
        a.or_assign(&empty);
        assert!(a.get(10));
        assert!(!a.get(0));
    }

    #[test]
    fn action_matrix_empty_and_all_zero_rows_retain_shape() {
        let empty = ActionCheckMatrix::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.check_count(), 0);
        assert!(empty.first().is_none());
        assert!(empty.as_words().is_empty());

        let zero_rows = ActionCheckMatrix::from_masks(
            200,
            [CheckMask::new(), CheckMask::new(), CheckMask::new()],
        );
        assert!(!zero_rows.is_empty());
        assert_eq!(zero_rows.len(), 3);
        assert_eq!(zero_rows.check_count(), 0);
        assert!(zero_rows.as_words().is_empty());
        assert_eq!(zero_rows.iter().len(), 3);
        assert!(zero_rows.iter().all(|row| !row.get(0) && !row.get(199)));
    }

    #[test]
    fn action_matrix_packs_rows_across_word_boundaries() {
        let row0 = CheckMask::from_indices(&[0, 63, 64, 129]);
        let row1 = CheckMask::from_indices(&[1, 62, 65, 128]);
        let row2 = CheckMask::from_indices(&[2, 66, 127]);
        let matrix = ActionCheckMatrix::from_masks(200, [row0, row1, row2]);

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix.check_count(), 130);
        assert_eq!(matrix.as_words().len(), 7);

        let first = matrix.get(0).unwrap();
        assert!(first.get(0));
        assert!(first.get(63));
        assert!(first.get(64));
        assert!(first.get(129));
        assert!(!first.get(1));
        assert!(!first.get(130));

        let second = matrix.get(1).unwrap();
        assert!(second.get(1));
        assert!(second.get(62));
        assert!(second.get(65));
        assert!(second.get(128));
        assert!(!second.get(0));
        assert!(!second.get(129));

        let third = matrix.get(2).unwrap();
        assert!(third.get(2));
        assert!(third.get(66));
        assert!(third.get(127));
        assert!(!third.get(1));
        assert!(!third.get(128));
        assert!(matrix.get(3).is_none());
    }

    #[test]
    fn action_matrix_trims_declared_but_unused_columns() {
        let matrix = ActionCheckMatrix::from_masks(
            512,
            [
                CheckMask::new(),
                CheckMask::from_indices(&[5]),
                CheckMask::new(),
            ],
        );
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix.check_count(), 6);
        assert_eq!(matrix.as_words(), &[1u64 << 11]);
        assert!(!matrix.get(0).unwrap().get(5));
        assert!(matrix.get(1).unwrap().get(5));
        assert!(!matrix.get(2).unwrap().get(5));
        assert!(!matrix.get(1).unwrap().get(511));
    }

    #[test]
    #[should_panic(expected = "exceeds declared check-count upper bound")]
    fn action_matrix_rejects_set_bit_above_declared_upper_bound() {
        let _ = ActionCheckMatrix::from_masks(64, [CheckMask::from_indices(&[64])]);
    }

    #[test]
    fn action_matrix_row_contains_and_or_work_when_unaligned() {
        let matrix = ActionCheckMatrix::from_masks(
            131,
            [
                CheckMask::from_indices(&[0, 64, 130]),
                CheckMask::from_indices(&[1, 63, 65, 129]),
            ],
        );
        let second = matrix.get(1).unwrap();
        assert!(second.contains_all(&CheckMask::from_indices(&[1, 65, 129])));
        assert!(!second.contains_all(&CheckMask::from_indices(&[1, 64])));
        assert!(second.contains_all(&CheckMask::new()));

        let mut aggregate = CheckMask::from_indices(&[0, 128]);
        second.or_into(&mut aggregate);
        for bit in [0, 1, 63, 65, 128, 129] {
            assert!(aggregate.get(bit), "aggregate is missing bit {bit}");
        }
        assert!(!aggregate.get(64));
        assert!(!aggregate.get(130));
    }

    #[test]
    fn action_matrix_raw_parts_validate_shape_and_padding() {
        let matrix = ActionCheckMatrix::from_raw_parts(2, 3, vec![0b1_0101]).unwrap();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix.check_count(), 3);
        assert!(matrix.get(0).unwrap().get(0));
        assert!(matrix.get(0).unwrap().get(2));
        assert!(matrix.get(1).unwrap().get(1));
        assert_eq!(matrix.as_words(), &[0b1_0101]);

        assert_eq!(
            ActionCheckMatrix::from_raw_parts(2, 3, vec![]).unwrap_err(),
            ActionCheckMatrixShapeError::WordCountMismatch {
                expected: 1,
                actual: 0,
            }
        );
        assert_eq!(
            ActionCheckMatrix::from_raw_parts(2, 3, vec![0b10_00000]).unwrap_err(),
            ActionCheckMatrixShapeError::NonZeroTailPadding {
                tail_bits: 6,
                value: 0b10_00000,
            }
        );
        assert_eq!(
            ActionCheckMatrix::from_raw_parts(2, 3, vec![0b00_1001]).unwrap_err(),
            ActionCheckMatrixShapeError::NonCanonicalCheckCount { check_count: 3 }
        );
    }

    #[test]
    fn action_matrix_vec_conversion_and_iteration_are_logical() {
        let matrix: ActionCheckMatrix = vec![
            CheckMask::from_words(vec![0, 0]),
            CheckMask::from_indices(&[7]),
            CheckMask::from_indices(&[7]),
        ]
        .into();
        assert_eq!(matrix.check_count(), 8);
        assert_eq!(matrix.iter().len(), 3);
        assert_eq!(matrix.iter().next_back(), matrix.get(2));
        assert_eq!(matrix.get(1), matrix.get(2));
        assert_ne!(matrix.get(0), matrix.get(1));

        let collected: ActionCheckMatrix = [CheckMask::new(), CheckMask::from_indices(&[70])]
            .into_iter()
            .collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected.check_count(), 71);
        assert!(collected.get(1).unwrap().get(70));
    }
}
