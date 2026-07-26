// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! On-disk node record format for the disk-backed liveness graph.
//!
//! Each node record stores the complete topology for one behavior graph node:
//! the node's key `(state_fp, tableau_idx)`, disk trace parent, successor list,
//! and precomputed check masks.
//!
//! Records are variable-length and designed for append-only writing to a
//! sequential file. The [`super::node_ptr_table::NodePtrTable`] (Slice C)
//! provides the `(fp, tidx) -> byte_offset` index for random reads.
//!
//! ## Record layout
//!
//! ```text
//! Header (64 bytes, all u64 for alignment):
//!   state_fp              u64
//!   tableau_idx           u64
//!   parent_fp             u64  (NO_PARENT sentinel if init node)
//!   parent_tidx           u64  (NO_PARENT sentinel if init node)
//!   reserved              u64
//!   succ_count            u64
//!   state_mask_words      u64
//!   action_check_count    u64  (packed action-matrix row width in bits)
//!
//! Successor payload (succ_count * 16 bytes):
//!   [succ_fp: u64, succ_tidx: u64] * succ_count
//!
//! State check mask (state_mask_words * 8 bytes):
//!   [word: u64] * state_mask_words
//!
//! Action check matrix (ceil(succ_count * action_check_count / 64) * 8 bytes):
//!   tightly packed row-major bits, then zero tail padding
//! ```
//!
//! Part of #2732 Slice D.

use crate::liveness::behavior_graph::{BehaviorGraphNode, NodeInfo};
use crate::liveness::checker::{ActionCheckMatrix, CheckMask};
use crate::state::Fingerprint;
use std::io::{self, Read, Write};

/// Sentinel value for "no parent" in the parent_fp / parent_tidx fields.
const NO_PARENT: u64 = u64::MAX;

/// Fixed header size in bytes (8 fields × 8 bytes).
const HEADER_SIZE: usize = 64;

/// Bytes per successor entry (fp + tidx).
const SUCCESSOR_ENTRY_SIZE: usize = 16;

/// Number of backing words in a tightly packed edge-by-check bit matrix.
fn action_matrix_word_count(edge_count: usize, check_count: usize) -> io::Result<usize> {
    let bit_count = edge_count.checked_mul(check_count).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "action-check matrix dimensions overflow usize",
        )
    })?;
    bit_count
        .checked_add(63)
        .map(|bits| bits / 64)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "action-check matrix word count overflows usize",
            )
        })
}

/// Write a node record to the given writer. Returns the number of bytes written.
pub(crate) fn write_node_record<W: Write>(
    w: &mut W,
    node: BehaviorGraphNode,
    info: &NodeInfo,
) -> io::Result<usize> {
    let succ_count = info.successors.len();
    let state_mask_words = info.state_check_mask.as_words().len();

    let action_check_count = info.action_check_masks.check_count();
    u32::try_from(succ_count).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "successor count exceeds packed action-matrix limit",
        )
    })?;
    u32::try_from(action_check_count).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "action-check count exceeds packed action-matrix limit",
        )
    })?;
    let action_edge_count = info.action_check_masks.len();
    let matrix_is_unpopulated = action_edge_count == 0 && action_check_count == 0;
    if !matrix_is_unpopulated && action_edge_count != succ_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "action-check matrix is not aligned with successors",
        ));
    }
    let action_matrix_words = action_matrix_word_count(succ_count, action_check_count)?;
    if info.action_check_masks.as_words().len() != action_matrix_words {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "action-check matrix has an invalid packed shape",
        ));
    }

    // Header (64 bytes)
    let mut buf = [0u8; HEADER_SIZE];
    buf[0..8].copy_from_slice(&node.state_fp.0.to_le_bytes());
    buf[8..16].copy_from_slice(&(node.tableau_idx as u64).to_le_bytes());

    let (parent_fp, parent_tidx) = match info.trace_parent.as_deref() {
        Some(parent) => (parent.state_fp.0, parent.tableau_idx as u64),
        None => (NO_PARENT, NO_PARENT),
    };
    buf[16..24].copy_from_slice(&parent_fp.to_le_bytes());
    buf[24..32].copy_from_slice(&parent_tidx.to_le_bytes());
    // Header word 4 is reserved. In-memory graphs reconstruct prefixes lazily,
    // while disk graphs keep only the parent needed for O(path) reconstruction.
    buf[40..48].copy_from_slice(&(succ_count as u64).to_le_bytes());
    buf[48..56].copy_from_slice(&(state_mask_words as u64).to_le_bytes());
    buf[56..64].copy_from_slice(&(action_check_count as u64).to_le_bytes());
    w.write_all(&buf)?;

    let mut total = HEADER_SIZE;

    // Successor payload
    for succ in &info.successors {
        w.write_all(&succ.state_fp.0.to_le_bytes())?;
        w.write_all(&(succ.tableau_idx as u64).to_le_bytes())?;
        total += SUCCESSOR_ENTRY_SIZE;
    }

    // State check mask
    for &word in info.state_check_mask.as_words() {
        w.write_all(&word.to_le_bytes())?;
        total += 8;
    }

    // Tightly packed action-check matrix.
    for &word in info.action_check_masks.as_words() {
        w.write_all(&word.to_le_bytes())?;
        total += 8;
    }

    Ok(total)
}

/// Read a u64 from a fixed-size subslice of a larger buffer.
///
/// The caller guarantees `offset + 8 <= buf.len()` (the buffer is always
/// `HEADER_SIZE` or `SUCCESSOR_ENTRY_SIZE` bytes, both multiples of 8).
#[inline]
fn read_u64(buf: &[u8], offset: usize) -> u64 {
    let bytes: [u8; 8] = buf[offset..offset + 8]
        .try_into()
        .expect("invariant: fixed-size record field is 8 bytes");
    u64::from_le_bytes(bytes)
}

fn read_usize(buf: &[u8], offset: usize, field: &'static str) -> io::Result<usize> {
    usize::try_from(read_u64(buf, offset)).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("node-record {field} does not fit in usize"),
        )
    })
}

/// Read a node record from the given reader.
/// Returns `(node_key, node_info)`.
pub(crate) fn read_node_record<R: Read>(r: &mut R) -> io::Result<(BehaviorGraphNode, NodeInfo)> {
    // Header
    let mut buf = [0u8; HEADER_SIZE];
    r.read_exact(&mut buf)?;

    let state_fp = Fingerprint(read_u64(&buf, 0));
    let tableau_idx = read_usize(&buf, 8, "tableau index")?;
    let parent_fp_raw = read_u64(&buf, 16);
    let parent_tidx_raw = read_u64(&buf, 24);
    let succ_count = read_usize(&buf, 40, "successor count")?;
    let state_mask_words = read_usize(&buf, 48, "state-mask word count")?;
    let action_check_count = read_usize(&buf, 56, "action-check count")?;
    u32::try_from(succ_count).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "node-record successor count exceeds packed-matrix limit",
        )
    })?;
    u32::try_from(action_check_count).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "node-record action-check count exceeds packed-matrix limit",
        )
    })?;

    let trace_parent = if parent_fp_raw == NO_PARENT {
        None
    } else {
        Some(Box::new(BehaviorGraphNode::new(
            Fingerprint(parent_fp_raw),
            usize::try_from(parent_tidx_raw).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "node-record parent tableau index does not fit in usize",
                )
            })?,
        )))
    };

    // Successors
    let mut successors = Vec::new();
    successors.try_reserve_exact(succ_count).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cannot allocate node-record successor payload: {error}"),
        )
    })?;
    let mut entry_buf = [0u8; SUCCESSOR_ENTRY_SIZE];
    for _ in 0..succ_count {
        r.read_exact(&mut entry_buf)?;
        let succ_fp = Fingerprint(read_u64(&entry_buf, 0));
        let succ_tidx = read_usize(&entry_buf, 8, "successor tableau index")?;
        successors.push(BehaviorGraphNode::new(succ_fp, succ_tidx));
    }

    // State check mask
    let state_check_mask = read_check_mask(r, state_mask_words)?;

    // Tightly packed action-check matrix.
    let action_word_count = action_matrix_word_count(succ_count, action_check_count)?;
    let mut action_words = Vec::new();
    action_words
        .try_reserve_exact(action_word_count)
        .map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cannot allocate node-record action-check payload: {error}"),
            )
        })?;
    let mut action_word_buf = [0u8; 8];
    for _ in 0..action_word_count {
        r.read_exact(&mut action_word_buf)?;
        action_words.push(u64::from_le_bytes(action_word_buf));
    }
    let action_check_masks =
        ActionCheckMatrix::from_raw_parts(succ_count, action_check_count, action_words)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;

    let node = BehaviorGraphNode::new(state_fp, tableau_idx);
    let info = NodeInfo {
        successors,
        trace_parent,
        state_check_mask,
        action_check_masks,
    };
    Ok((node, info))
}

/// Read a CheckMask of `word_count` u64 words.
fn read_check_mask<R: Read>(r: &mut R, word_count: usize) -> io::Result<CheckMask> {
    if word_count == 0 {
        return Ok(CheckMask::new());
    }
    let mut words = Vec::new();
    words.try_reserve_exact(word_count).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cannot allocate node-record check-mask payload: {error}"),
        )
    })?;
    let mut buf = [0u8; 8];
    for _ in 0..word_count {
        r.read_exact(&mut buf)?;
        words.push(u64::from_le_bytes(buf));
    }
    Ok(CheckMask::from_words(words))
}

/// Compute the total byte size of a node record without writing it.
#[cfg(test)]
pub(crate) fn record_byte_size(info: &NodeInfo) -> usize {
    let succ_count = info.successors.len();
    let state_mask_words = info.state_check_mask.as_words().len();
    let action_matrix_words =
        action_matrix_word_count(succ_count, info.action_check_masks.check_count())
            .expect("test node-record matrix dimensions must fit");

    HEADER_SIZE + succ_count * SUCCESSOR_ENTRY_SIZE + state_mask_words * 8 + action_matrix_words * 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_node(fp: u64, tidx: usize) -> BehaviorGraphNode {
        BehaviorGraphNode::new(Fingerprint(fp), tidx)
    }

    fn make_info(
        successors: Vec<BehaviorGraphNode>,
        parent: Option<BehaviorGraphNode>,
    ) -> NodeInfo {
        NodeInfo {
            successors,
            trace_parent: parent.map(Box::new),
            state_check_mask: CheckMask::new(),
            action_check_masks: ActionCheckMatrix::new(),
        }
    }

    #[test]
    fn test_roundtrip_init_node_no_successors() {
        let node = make_node(42, 0);
        let info = make_info(vec![], None);

        let mut buf = Vec::new();
        let written = write_node_record(&mut buf, node, &info).unwrap();
        assert_eq!(written, HEADER_SIZE);
        assert_eq!(buf.len(), HEADER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let (read_node, read_info) = read_node_record(&mut cursor).unwrap();
        assert_eq!(read_node, node);
        assert!(read_info.trace_parent.is_none());
        assert!(read_info.successors.is_empty());
    }

    #[test]
    fn test_roundtrip_with_successors() {
        let node = make_node(100, 1);
        let parent = make_node(50, 0);
        let succ_a = make_node(200, 2);
        let succ_b = make_node(300, 0);
        let info = make_info(vec![succ_a, succ_b], Some(parent));

        let mut buf = Vec::new();
        let written = write_node_record(&mut buf, node, &info).unwrap();
        let expected_size = HEADER_SIZE + 2 * SUCCESSOR_ENTRY_SIZE;
        assert_eq!(written, expected_size);

        let mut cursor = Cursor::new(&buf);
        let (read_node, read_info) = read_node_record(&mut cursor).unwrap();
        assert_eq!(read_node, node);
        assert_eq!(read_info.trace_parent.as_deref(), Some(&parent));
        assert_eq!(read_info.successors.len(), 2);
        assert_eq!(read_info.successors[0], succ_a);
        assert_eq!(read_info.successors[1], succ_b);
        assert_eq!(read_info.action_check_masks.len(), 2);
        assert!(read_info.action_check_masks.iter().all(|row| !row.get(0)));
    }

    #[test]
    fn test_roundtrip_with_check_masks() {
        let node = make_node(42, 0);
        let succ = make_node(99, 1);

        let mut state_mask = CheckMask::new();
        state_mask.set(0);
        state_mask.set(5);
        state_mask.set(63);

        let mut action_mask = CheckMask::new();
        action_mask.set(1);
        action_mask.set(7);

        let info = NodeInfo {
            successors: vec![succ],
            trace_parent: None,
            state_check_mask: state_mask.clone(),
            action_check_masks: vec![action_mask.clone()].into(),
        };

        let mut buf = Vec::new();
        write_node_record(&mut buf, node, &info).unwrap();

        let mut cursor = Cursor::new(&buf);
        let (_, read_info) = read_node_record(&mut cursor).unwrap();

        assert!(read_info.state_check_mask.get(0));
        assert!(read_info.state_check_mask.get(5));
        assert!(read_info.state_check_mask.get(63));
        assert!(!read_info.state_check_mask.get(1));

        assert_eq!(read_info.action_check_masks.len(), 1);
        assert!(read_info.action_check_masks.get(0).unwrap().get(1));
        assert!(read_info.action_check_masks.get(0).unwrap().get(7));
        assert!(!read_info.action_check_masks.get(0).unwrap().get(0));
    }

    #[test]
    fn test_roundtrip_multi_word_masks() {
        let node = make_node(42, 0);
        let succ = make_node(99, 1);

        let mut state_mask = CheckMask::new();
        state_mask.set(0);
        state_mask.set(65); // forces second word
        state_mask.set(130); // forces third word

        let mut action_mask = CheckMask::new();
        action_mask.set(64);
        action_mask.set(128);

        let info = NodeInfo {
            successors: vec![succ],
            trace_parent: None,
            state_check_mask: state_mask,
            action_check_masks: vec![action_mask].into(),
        };

        let mut buf = Vec::new();
        write_node_record(&mut buf, node, &info).unwrap();

        let mut cursor = Cursor::new(&buf);
        let (_, read_info) = read_node_record(&mut cursor).unwrap();

        assert!(read_info.state_check_mask.get(0));
        assert!(read_info.state_check_mask.get(65));
        assert!(read_info.state_check_mask.get(130));
        assert!(!read_info.state_check_mask.get(1));
        assert!(!read_info.state_check_mask.get(64));

        assert!(read_info.action_check_masks.get(0).unwrap().get(64));
        assert!(read_info.action_check_masks.get(0).unwrap().get(128));
    }

    #[test]
    fn test_roundtrip_multiple_successors_with_masks() {
        let node = make_node(10, 0);
        let succs = vec![make_node(20, 1), make_node(30, 2), make_node(40, 0)];

        let mut state_mask = CheckMask::new();
        state_mask.set(3);

        let action_masks: Vec<CheckMask> = (0..3)
            .map(|i| {
                let mut m = CheckMask::new();
                m.set(i * 2);
                m
            })
            .collect();

        let info = NodeInfo {
            successors: succs.clone(),
            trace_parent: Some(Box::new(make_node(1, 0))),
            state_check_mask: state_mask,
            action_check_masks: action_masks.into(),
        };

        let mut buf = Vec::new();
        write_node_record(&mut buf, node, &info).unwrap();

        let mut cursor = Cursor::new(&buf);
        let (read_node, read_info) = read_node_record(&mut cursor).unwrap();

        assert_eq!(read_node, node);
        assert_eq!(read_info.successors, succs);
        assert!(read_info.state_check_mask.get(3));

        for i in 0..3 {
            assert!(read_info.action_check_masks.get(i).unwrap().get(i * 2));
        }
    }

    #[test]
    fn test_roundtrip_packed_rows_cross_word_boundaries() {
        let node = make_node(77, 0);
        let successors = vec![make_node(1, 0), make_node(2, 0), make_node(3, 0)];

        let row0 = CheckMask::from_indices(&[0, 64]);
        let row1 = CheckMask::from_indices(&[1, 64]);
        let row2 = CheckMask::from_indices(&[2, 63]);
        let info = NodeInfo {
            successors,
            trace_parent: None,
            state_check_mask: CheckMask::new(),
            action_check_masks: vec![row0, row1, row2].into(),
        };

        let mut buf = Vec::new();
        let written = write_node_record(&mut buf, node, &info).unwrap();
        assert_eq!(
            written,
            HEADER_SIZE + 3 * SUCCESSOR_ENTRY_SIZE + 4 * 8,
            "3 x 65 bits must occupy four tightly packed words"
        );

        let (_, read_info) = read_node_record(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(read_info.action_check_masks.check_count(), 65);
        assert!(read_info.action_check_masks.get(0).unwrap().get(0));
        assert!(read_info.action_check_masks.get(0).unwrap().get(64));
        assert!(read_info.action_check_masks.get(1).unwrap().get(1));
        assert!(read_info.action_check_masks.get(1).unwrap().get(64));
        assert!(read_info.action_check_masks.get(2).unwrap().get(2));
        assert!(read_info.action_check_masks.get(2).unwrap().get(63));
        assert!(!read_info.action_check_masks.get(2).unwrap().get(64));
    }

    #[test]
    fn test_write_rejects_misaligned_zero_width_matrix() {
        let info = NodeInfo {
            successors: vec![make_node(1, 0), make_node(2, 0)],
            trace_parent: None,
            state_check_mask: CheckMask::new(),
            action_check_masks: vec![CheckMask::new()].into(),
        };

        let error = write_node_record(&mut Vec::new(), make_node(9, 0), &info)
            .expect_err("one zero row cannot describe two successors");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("not aligned"));
    }

    #[test]
    fn test_read_rejects_oversized_matrix_header_before_payload() {
        let mut header = [0u8; HEADER_SIZE];
        header[40..48].copy_from_slice(&1u64.to_le_bytes());
        header[56..64].copy_from_slice(&((u32::MAX as u64) + 1).to_le_bytes());

        let error = read_node_record(&mut Cursor::new(header))
            .expect_err("oversized packed-matrix dimensions must fail at the header");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("packed-matrix limit"));
    }

    #[test]
    fn test_multiple_records_sequential() {
        let mut buf = Vec::new();
        let mut offsets = Vec::new();

        // Write three records sequentially.
        for i in 0..3 {
            let offset = buf.len();
            offsets.push(offset);
            let node = make_node(100 + i as u64, i);
            let parent = (i > 0).then(|| make_node(100 + (i - 1) as u64, i - 1));
            let info = make_info(vec![make_node(200 + i as u64, 0)], parent);
            write_node_record(&mut buf, node, &info).unwrap();
        }

        // Read each record back by seeking to its offset.
        for i in 0..3 {
            let mut cursor = Cursor::new(&buf[offsets[i]..]);
            let (read_node, read_info) = read_node_record(&mut cursor).unwrap();
            assert_eq!(read_node, make_node(100 + i as u64, i));
            assert_eq!(read_info.successors.len(), 1);
        }
    }

    #[test]
    fn test_record_byte_size() {
        let info = make_info(vec![], None);
        assert_eq!(record_byte_size(&info), HEADER_SIZE);

        let info2 = make_info(vec![make_node(1, 0), make_node(2, 1)], None);
        assert_eq!(
            record_byte_size(&info2),
            HEADER_SIZE + 2 * SUCCESSOR_ENTRY_SIZE
        );

        // With masks.
        let mut state_mask = CheckMask::new();
        state_mask.set(0);
        let mut action_mask = CheckMask::new();
        action_mask.set(1);
        let info3 = NodeInfo {
            successors: vec![make_node(1, 0)],
            trace_parent: None,
            state_check_mask: state_mask,
            action_check_masks: vec![action_mask].into(),
        };
        // header + 1 succ * 16 + 1 state word * 8 + 1 succ * 1 action word * 8
        assert_eq!(record_byte_size(&info3), HEADER_SIZE + 16 + 8 + 8);
    }

    #[test]
    fn test_written_size_matches_computed() {
        let node = make_node(42, 3);
        let mut state_mask = CheckMask::new();
        state_mask.set(10);
        state_mask.set(70);
        let mut am0 = CheckMask::new();
        am0.set(5);
        let mut am1 = CheckMask::new();
        am1.set(100);

        let info = NodeInfo {
            successors: vec![make_node(1, 0), make_node(2, 1)],
            trace_parent: Some(Box::new(make_node(10, 0))),
            state_check_mask: state_mask,
            action_check_masks: vec![am0, am1].into(),
        };

        let mut buf = Vec::new();
        let written = write_node_record(&mut buf, node, &info).unwrap();
        assert_eq!(written, buf.len());
        assert_eq!(record_byte_size(&info), written);
    }
}
