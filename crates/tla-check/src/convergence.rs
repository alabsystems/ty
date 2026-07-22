// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BFS convergence tracking for oracle routing decisions.
//!
//! Records how many new (unique) states BFS discovers at each depth level.
//! When the rate of new state discovery drops, BFS is approaching the
//! fixed point and the oracle can route more work to symbolic engines.

/// Tracks BFS convergence rate for oracle routing decisions.
///
/// Records how many new (unique) states BFS discovers at each depth level.
/// When the rate of new state discovery drops, BFS is approaching the
/// fixed point and the oracle can route more work to symbolic engines.
///
/// Part of #3785.
pub(crate) struct ConvergenceTracker {
    /// (depth, new_states_at_depth, total_states_at_depth) tuples.
    /// Kept in order of increasing depth.
    history: Vec<(usize, u64, u64)>,
}

impl ConvergenceTracker {
    pub(crate) fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    /// Record new and total state counts at a BFS depth boundary.
    pub(crate) fn record(&mut self, depth: usize, new_states: u64, total_states: u64) {
        self.history.push((depth, new_states, total_states));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_tracker_records() {
        let mut tracker = ConvergenceTracker::new();
        tracker.record(1, 100, 100);
        tracker.record(2, 50, 150);
        tracker.record(3, 5, 10000);
    }
}
