// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Heuristic best-first search for reachability witness finding.
//!
//! Phase 2e in the reachability pipeline, between random walk (2d) and
//! full BFS (Phase 3). Uses LP state-equation relaxation as an admissible
//! distance heuristic to guide exploration toward EF(φ) witnesses.
//!
//! **Memory-bounded:** Uses a Bloom filter for visited-state tracking (O(1)
//! per state, ~10 bits/entry) and truncates the priority queue when it
//! exceeds a configurable bound. This allows exploring nets with 10^8+
//! reachable states without OOM.
//!
//! **Witness-only:** Like random walk, this phase only resolves
//! EF(φ)=TRUE and AG(φ)=FALSE. It cannot prove universal properties.
//! Unresolved trackers fall through to BFS.
//!
//! **Sound because:** Any marking reached by firing enabled transitions from
//! the initial marking is reachable. Bloom filter false positives only skip
//! states (safe for witness search). Queue truncation only discards states.

mod frontier;
mod heuristic;
mod search;

#[cfg(test)]
mod tests;

use std::time::Instant;

use crate::examinations::reachability_witness::WitnessValidationContext;
use crate::petri_net::PetriNet;

use super::reachability::PropertyTracker;

/// Default maximum states in the Bloom filter before stopping.
const DEFAULT_BLOOM_CAPACITY: usize = 10_000_000;

/// Default maximum entries in the priority queue.
const DEFAULT_MAX_QUEUE_SIZE: usize = 1_000_000;

/// Fraction of available memory the best-first open set may consume.
/// Conservative — leaves room for the bloom filter, successor scratch, one
/// expansion's worth of pushes, and the rest of the process.
const HEURISTIC_QUEUE_MEMORY_FRACTION: f64 = 0.4;

/// Memory-aware priority-queue capacity. Each `ScoredNode` owns a dense
/// `Vec<u64>` marking (`num_places × 8` bytes) plus node/trace overhead, so a
/// flat 1 000 000-entry cap is `num_places × 8 MB` — tens to hundreds of GB on
/// wide nets (AirplaneLD-PT-4000: 28 019 places ⇒ ~224 KB/node ⇒ 224 GB at the
/// flat cap). Bound the entry count so `entries × per_node_bytes` stays within
/// a memory budget, reserving a one-expansion (`num_transitions`) margin for
/// the successors pushed before each per-expansion truncation. Falls back to
/// the flat cap when memory detection fails.
fn memory_aware_max_queue_size(net: &PetriNet) -> usize {
    let per_node = net
        .num_places()
        .saturating_mul(std::mem::size_of::<u64>())
        .saturating_add(128)
        .max(1);
    let Some(available) = crate::memory::available_memory_bytes() else {
        return DEFAULT_MAX_QUEUE_SIZE;
    };
    let budget = (available as f64 * HEURISTIC_QUEUE_MEMORY_FRACTION) as usize;
    (budget / per_node)
        .saturating_sub(net.num_transitions())
        .clamp(1024, DEFAULT_MAX_QUEUE_SIZE)
}

/// Default budget: maximum state expansions before giving up.
const DEFAULT_MAX_EXPANSIONS: u64 = 5_000_000;

/// Run heuristic best-first witness search on unresolved trackers.
///
/// For each unresolved tracker, computes LP-derived heuristic weights and
/// uses them to guide a memory-bounded best-first search toward witness
/// states.
pub(crate) fn run_heuristic_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation: &WitnessValidationContext<'_>,
    deadline: Option<Instant>,
) {
    search::run_heuristic_seeding_params(
        net,
        trackers,
        validation,
        deadline,
        DEFAULT_MAX_EXPANSIONS,
        DEFAULT_BLOOM_CAPACITY,
        memory_aware_max_queue_size(net),
    );
}
