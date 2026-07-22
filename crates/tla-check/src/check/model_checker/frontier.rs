// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BFS frontier queue abstraction.
//!
//! Part of #2335: Extracts the BFS frontier queue behind a trait so the BFS loop
//! is not hardcoded to `VecDeque`. Phase 1 provides the trait + VecDeque impl only.
//! Future phases add checkpoint persistence (Phase 2) and disk-backed spillover (Phase 3).

use std::collections::VecDeque;

/// Abstraction over the BFS frontier queue.
///
/// The BFS loop pushes successor states and pops the next state to explore.
/// This trait decouples the loop from a specific queue implementation, enabling
/// future disk-backed or checkpoint-aware frontiers without changing loop logic.
///
/// Phase 1: Only `VecDeque` impl. Monomorphized — zero vtable overhead.
pub(super) trait BfsFrontier {
    type Entry;

    fn push(&mut self, entry: Self::Entry);
    fn pop(&mut self) -> Option<Self::Entry>;
    fn len(&self) -> usize;

    /// Release queue storage that is no longer needed after a complete BFS.
    ///
    /// Post-BFS liveness checking can run for a substantial amount of time.
    /// Dropping the drained frontier's retained capacity before that phase keeps
    /// BFS-only memory out of the liveness peak. This hook must only be called
    /// after [`BfsLoopOutcome::Complete`](super::bfs::BfsLoopOutcome::Complete)
    /// reports `frontier_exhausted: true`, never merely because the outcome is
    /// `Complete`: portfolio early exit may hold an unprocessed dequeued item,
    /// and depth-limited completion is a truncated, potentially resumable run.
    fn release_after_complete_bfs(&mut self);

    /// Snapshot the current queue contents in dequeue order.
    ///
    /// Checkpointing uses this rather than `iter()` because some frontier
    /// implementations store entries in compressed/arena form and must
    /// materialize them on demand.
    fn checkpoint_entries(&self) -> Vec<Self::Entry>
    where
        Self::Entry: Clone;
}

impl<T> BfsFrontier for VecDeque<T> {
    type Entry = T;

    fn push(&mut self, entry: T) {
        self.push_back(entry);
    }

    fn pop(&mut self) -> Option<T> {
        self.pop_front()
    }

    fn len(&self) -> usize {
        VecDeque::len(self)
    }

    fn release_after_complete_bfs(&mut self) {
        *self = VecDeque::new();
    }

    fn checkpoint_entries(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::BfsFrontier;
    use std::collections::VecDeque;

    #[test]
    fn vec_deque_releases_retained_capacity_after_complete_bfs() {
        let mut frontier = VecDeque::with_capacity(1024);
        frontier.extend(0..16);
        assert!(frontier.capacity() >= 1024);

        frontier.release_after_complete_bfs();

        assert!(frontier.is_empty());
        assert_eq!(frontier.capacity(), 0);
    }
}
