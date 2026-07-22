// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compact BFS frontier for payload-witness-backed states.
//!
//! A [`NoTraceQueueEntry::Witness`] needs only its fingerprint, incremental
//! fingerprint seed, depth, and trace location while it waits in the BFS
//! frontier. Keeping those fields in a compact metadata deque avoids sizing
//! every frontier slot for the much larger [`NoTraceQueueEntry::Owned`]
//! variant. Rare non-witness entries remain in a fallback deque.
//!
//! A one-byte order tag is appended for every entry. The tag deque preserves
//! exact FIFO order when witness and fallback entries are interleaved, while
//! the two payload deques preserve order within each representation.

use std::collections::VecDeque;

use super::super::frontier::BfsFrontier;
use super::storage_modes::NoTraceQueueEntry;
use crate::state::Fingerprint;

/// Selects the payload deque for one entry in the logical FIFO.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum OrderTag {
    Witness,
    Fallback,
}

/// Compact metadata retained for a witness-backed frontier entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WitnessFrontierMeta {
    fp: Fingerprint,
    combined_xor: u64,
    depth: usize,
    trace_loc: u64,
}

/// FIFO frontier specialized for payload-witness-backed `ArrayState` entries.
///
/// Witness entries are split into compact metadata plus an order tag. All
/// other representations use `fallback`, so this remains a drop-in
/// [`BfsFrontier`] for the fingerprint-only BFS path.
pub(in crate::check::model_checker) struct WitnessBfsFrontier {
    order: VecDeque<OrderTag>,
    witnesses: VecDeque<WitnessFrontierMeta>,
    fallback: VecDeque<(NoTraceQueueEntry, usize, u64)>,
}

impl WitnessBfsFrontier {
    /// Create an empty frontier without allocating backing storage.
    #[must_use]
    pub(in crate::check::model_checker) fn new() -> Self {
        Self {
            order: VecDeque::new(),
            witnesses: VecDeque::new(),
            fallback: VecDeque::new(),
        }
    }

    /// Create an empty frontier sized for a witness-dominated workload.
    ///
    /// Only the compact order and witness deques are preallocated. The rare
    /// fallback path grows independently if a non-witness entry is pushed.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(in crate::check::model_checker) fn with_capacity(capacity: usize) -> Self {
        Self {
            order: VecDeque::with_capacity(capacity),
            witnesses: VecDeque::with_capacity(capacity),
            fallback: VecDeque::new(),
        }
    }

    #[inline]
    fn witness_entry(meta: WitnessFrontierMeta) -> (NoTraceQueueEntry, usize, u64) {
        (
            NoTraceQueueEntry::Witness {
                fp: meta.fp,
                combined_xor: meta.combined_xor,
            },
            meta.depth,
            meta.trace_loc,
        )
    }

    #[cfg(debug_assertions)]
    fn assert_internal_lengths(&self) {
        debug_assert_eq!(
            self.order.len(),
            self.witnesses.len() + self.fallback.len(),
            "WitnessBfsFrontier order/payload counts diverged",
        );
    }
}

impl Default for WitnessBfsFrontier {
    fn default() -> Self {
        Self::new()
    }
}

impl BfsFrontier for WitnessBfsFrontier {
    type Entry = (NoTraceQueueEntry, usize, u64);

    #[inline]
    fn push(&mut self, (entry, depth, trace_loc): Self::Entry) {
        match entry {
            NoTraceQueueEntry::Witness { fp, combined_xor } => {
                self.witnesses.push_back(WitnessFrontierMeta {
                    fp,
                    combined_xor,
                    depth,
                    trace_loc,
                });
                self.order.push_back(OrderTag::Witness);
            }
            entry => {
                self.fallback.push_back((entry, depth, trace_loc));
                self.order.push_back(OrderTag::Fallback);
            }
        }
    }

    #[inline]
    fn pop(&mut self) -> Option<Self::Entry> {
        let tag = self.order.pop_front()?;
        let entry = match tag {
            OrderTag::Witness => {
                let meta = self
                    .witnesses
                    .pop_front()
                    .expect("WitnessBfsFrontier invariant: witness order tag has metadata");
                Self::witness_entry(meta)
            }
            OrderTag::Fallback => self
                .fallback
                .pop_front()
                .expect("WitnessBfsFrontier invariant: fallback order tag has payload"),
        };
        #[cfg(debug_assertions)]
        self.assert_internal_lengths();
        Some(entry)
    }

    #[inline]
    fn len(&self) -> usize {
        self.order.len()
    }

    fn release_after_complete_bfs(&mut self) {
        // Replacing rather than clearing releases the largest BFS level's
        // retained capacities before a subsequent liveness phase.
        self.order = VecDeque::new();
        self.witnesses = VecDeque::new();
        self.fallback = VecDeque::new();
    }

    fn checkpoint_entries(&self) -> Vec<Self::Entry>
    where
        Self::Entry: Clone,
    {
        let mut witness_iter = self.witnesses.iter().copied();
        let mut fallback_iter = self.fallback.iter().cloned();
        let mut entries = Vec::with_capacity(self.order.len());

        for tag in &self.order {
            let entry = match tag {
                OrderTag::Witness => Self::witness_entry(
                    witness_iter
                        .next()
                        .expect("WitnessBfsFrontier invariant: witness order tag has metadata"),
                ),
                OrderTag::Fallback => fallback_iter
                    .next()
                    .expect("WitnessBfsFrontier invariant: fallback order tag has payload"),
            };
            entries.push(entry);
        }

        debug_assert!(witness_iter.next().is_none());
        debug_assert!(fallback_iter.next().is_none());
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::BulkStateHandle;
    use crate::state::ArrayState;
    use crate::Value;

    fn witness(
        fp: u64,
        combined_xor: u64,
        depth: usize,
        trace_loc: u64,
    ) -> (NoTraceQueueEntry, usize, u64) {
        (
            NoTraceQueueEntry::Witness {
                fp: Fingerprint(fp),
                combined_xor,
            },
            depth,
            trace_loc,
        )
    }

    fn owned(fp: u64, value: i64, depth: usize) -> (NoTraceQueueEntry, usize, u64) {
        (
            NoTraceQueueEntry::Owned {
                state: ArrayState::from_values(vec![Value::int(value)]),
                fp: Fingerprint(fp),
            },
            depth,
            fp + 1_000,
        )
    }

    fn bulk(fp: u64, index: u32, depth: usize) -> (NoTraceQueueEntry, usize, u64) {
        (
            NoTraceQueueEntry::Bulk(BulkStateHandle::with_fingerprint(index, Fingerprint(fp))),
            depth,
            fp + 2_000,
        )
    }

    fn assert_witness(entry: (NoTraceQueueEntry, usize, u64), expected: (u64, u64, usize, u64)) {
        let (queue_entry, depth, trace_loc) = entry;
        match queue_entry {
            NoTraceQueueEntry::Witness { fp, combined_xor } => {
                assert_eq!((fp.0, combined_xor, depth, trace_loc), expected,);
            }
            _ => panic!("expected witness entry"),
        }
    }

    #[test]
    fn mixed_bulk_owned_and_witness_entries_pop_in_fifo_order() {
        let mut frontier = WitnessBfsFrontier::new();
        frontier.push(bulk(10, 3, 0));
        frontier.push(witness(20, 200, 1, 2_020));
        frontier.push(owned(30, 7, 2));
        frontier.push(witness(40, 400, 3, 2_040));
        assert_eq!(frontier.len(), 4);

        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Bulk(handle) => {
                assert_eq!(
                    handle,
                    BulkStateHandle::with_fingerprint(3, Fingerprint(10))
                );
            }
            _ => panic!("expected bulk entry"),
        }
        assert_eq!((depth, trace_loc), (0, 2_010));
        assert_eq!(frontier.len(), 3);

        assert_witness(frontier.pop().unwrap(), (20, 200, 1, 2_020));

        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Owned { fp, .. } => assert_eq!(fp, Fingerprint(30)),
            _ => panic!("expected owned entry"),
        }
        assert_eq!((depth, trace_loc), (2, 1_030));

        assert_witness(frontier.pop().unwrap(), (40, 400, 3, 2_040));
        assert_eq!(frontier.len(), 0);
        assert!(frontier.pop().is_none());
    }

    #[test]
    fn checkpoint_preserves_exact_mixed_order_without_consuming() {
        let mut frontier = WitnessBfsFrontier::new();
        frontier.push(witness(10, 100, 0, 1_010));
        frontier.push(owned(20, 8, 1));
        frontier.push(witness(30, 300, 2, 1_030));
        frontier.push(bulk(40, 9, 3));

        // Exercise checkpointing after the per-representation read heads have
        // both advanced, then append another witness before the snapshot.
        assert_witness(frontier.pop().unwrap(), (10, 100, 0, 1_010));
        let (entry, _, _) = frontier.pop().unwrap();
        assert!(matches!(
            entry,
            NoTraceQueueEntry::Owned {
                fp: Fingerprint(20),
                ..
            }
        ));
        frontier.push(witness(50, 500, 4, 1_050));

        let checkpoint = frontier.checkpoint_entries();
        assert_eq!(frontier.len(), 3);
        assert_eq!(checkpoint.len(), 3);
        assert_witness(checkpoint[0].clone(), (30, 300, 2, 1_030));
        match &checkpoint[1].0 {
            NoTraceQueueEntry::Bulk(handle) => {
                assert_eq!(
                    *handle,
                    BulkStateHandle::with_fingerprint(9, Fingerprint(40)),
                );
            }
            _ => panic!("expected bulk entry"),
        }
        assert_eq!((checkpoint[1].1, checkpoint[1].2), (3, 2_040));
        assert_witness(checkpoint[2].clone(), (50, 500, 4, 1_050));

        // A snapshot is non-consuming and has the same order as subsequent pops.
        assert_witness(frontier.pop().unwrap(), (30, 300, 2, 1_030));
        let (entry, _, _) = frontier.pop().unwrap();
        assert!(matches!(entry, NoTraceQueueEntry::Bulk(_)));
        assert_witness(frontier.pop().unwrap(), (50, 500, 4, 1_050));
        assert!(frontier.pop().is_none());
    }

    #[test]
    fn release_empties_frontier_and_drops_all_retained_capacity() {
        let mut frontier = WitnessBfsFrontier::with_capacity(128);
        for n in 0..64 {
            frontier.push(witness(n, n ^ 0x55, n as usize, n + 10));
        }
        frontier.push(owned(1_000, 9, 64));

        assert_eq!(frontier.len(), 65);
        assert!(frontier.order.capacity() >= 128);
        assert!(frontier.witnesses.capacity() >= 128);
        assert!(frontier.fallback.capacity() > 0);

        frontier.release_after_complete_bfs();

        assert_eq!(frontier.len(), 0);
        assert!(frontier.pop().is_none());
        assert_eq!(frontier.order.capacity(), 0);
        assert_eq!(frontier.witnesses.capacity(), 0);
        assert_eq!(frontier.fallback.capacity(), 0);
    }
}
