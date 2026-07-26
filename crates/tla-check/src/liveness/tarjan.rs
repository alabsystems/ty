// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Arena-based iterative Tarjan's algorithm for SCC detection
//!
//! This module implements an arena-indexed iterative Tarjan's algorithm to find
//! strongly connected components (SCCs) in the behavior graph. SCCs are cycles
//! in the product graph that may indicate liveness violations.
//!
//! # Algorithm Overview
//!
//! Tarjan's algorithm uses depth-first search to find SCCs in O(V + E) time.
//! This implementation is iterative (rather than recursive) to handle large
//! graphs without stack overflow.
//!
//! The key insight is that an SCC is identified when we find a node that is
//! its own "low link" - meaning it can reach itself through a cycle.
//!
//! # Performance — Arena-Based Indexing
//!
//! For large graphs (e.g., CoffeeCan3000Beans with 9M+ nodes), hash-map-based
//! node state lookups dominate runtime. This implementation uses a two-phase
//! approach:
//!
//! **Phase 1 — Indexing:** All `BehaviorGraphNode` keys already have contiguous
//! `u32` indices in the graph store's persistent reverse index. Completed
//! in-memory graphs lend their packed CSR directly; raw and disk graphs build
//! an owned CSR fallback. The historical one-time O(V) `FxHashMap` remains
//! available for fallback-path A/B validation.
//!
//! **Phase 2 — Arena DFS:** All node states, successor lists, and stacks use
//! `u32` indices into flat `Vec<T>` arrays. This replaces hash-map lookups with
//! O(1) array indexing during the DFS, which is cache-friendly and avoids the
//! ~7ns/lookup overhead of `FxHashMap` at 9M entries.
//!
//! Successor lists use CSR offsets and dense `u32` targets. The production
//! in-memory path borrows both slices from the behavior graph, so Tarjan does
//! not duplicate topology at peak RSS. A filtered pass adds only one
//! eligibility bit per physical edge.
//!
//! Trivial SCC filtering (single-node SCCs without self-loops) is done inline
//! during extraction via `has_self_loop` on `ArenaNodeState`, avoiding a
//! post-hoc graph lookup pass.
//!
//! # TLC Reference
//!
//! This follows TLC's implementation in:
//! - `tlc2/tool/liveness/LiveWorker.java` - checkSccs method
//!
//! # References
//!
//! - Tarjan, R. E. (1972). "Depth-first search and linear graph algorithms"
//! - <https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm>

use super::behavior_graph::{BehaviorGraph, BehaviorGraphNode, NodeInfo};
#[cfg(test)]
use super::scc::is_trivial_scc_in_graph;
#[cfg(test)]
use super::scc::TarjanStats;
use rustc_hash::FxHashMap;
use std::borrow::Cow;

// Re-export SCC types for backward compatibility with existing references
// to `tarjan::Scc`, `tarjan::TarjanResult`, etc. in checker modules.
pub(super) use super::scc::{Scc, TarjanResult};

/// Optional edge filter predicate for Tarjan's algorithm (#2704).
///
/// Receives `(from_info, succ_idx, to, to_info)` so the filter can read
/// precomputed bitmasks without performing graph lookups. `succ_idx` is the
/// source-local logical edge index, including when the physical adjacency is
/// one global packed CSR array.
type EdgeFilter<'a> = Option<&'a dyn Fn(&NodeInfo, usize, &BehaviorGraphNode, &NodeInfo) -> bool>;

/// Sentinel value indicating an arena node has not been visited yet.
const NOT_VISITED: u32 = u32::MAX;

/// Default node-count threshold above which Tarjan arena allocation emits an
/// OOM-advisory note to stderr.
///
/// The borrowed path primarily allocates node state and traversal stacks; the
/// raw/disk fallback additionally owns node keys and CSR topology. Operators
/// on small-memory hosts may want a lower threshold; large hosts may want
/// higher.
///
/// Override via `TY_TARJAN_ARENA_WARN_NODES`. `0` disables the warning.
///
/// Part of #4080: OOM safety — tarjan SCC arena visibility.
const DEFAULT_TARJAN_ARENA_WARN_THRESHOLD: usize = 1_000_000;

/// Read the Tarjan arena warn threshold from the environment, falling back to
/// [`DEFAULT_TARJAN_ARENA_WARN_THRESHOLD`].
///
/// Cached on first call so repeated SCC runs pay no env-parsing cost.
fn tarjan_arena_warn_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TY_TARJAN_ARENA_WARN_NODES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_TARJAN_ARENA_WARN_THRESHOLD)
    })
}

/// Whether the dense-id arena indexing path is enabled (default ON).
///
/// On the raw/disk fallback, enabling this resolves successor arena ids through
/// the store and drops the parallel `node_to_id` hash map. Both the compact
/// in-memory reverse index and the disk pointer table preserve the
/// `node_keys()` position as the id. Packed in-memory CSR already consists of
/// stable dense ids and therefore always uses that representation.
///
/// Kill-switch `TY_TARJAN_DENSE_IDS=0` forces the historical `node_to_id` path
/// for A/B comparison. Cached on first call.
fn tarjan_dense_ids_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::debug_env::env_flag_default_on(&CACHED, "TY_TARJAN_DENSE_IDS")
}

/// Per-node state in the arena, indexed by `u32` node id.
///
/// Packed into 12 bytes for cache efficiency (vs 24+ bytes with a hash-map
/// entry including key and bucket metadata).
#[derive(Debug, Clone, Copy)]
struct ArenaNodeState {
    /// Discovery index (NOT_VISITED if unvisited)
    index: u32,
    /// Lowest reachable index
    low_link: u32,
    /// Whether node is currently on the Tarjan stack
    on_stack: bool,
    /// Whether this node has a self-loop (edge to itself).
    has_self_loop: bool,
}

impl ArenaNodeState {
    /// Unvisited sentinel state.
    const UNVISITED: Self = Self {
        index: NOT_VISITED,
        low_link: NOT_VISITED,
        on_stack: false,
        has_self_loop: false,
    };
}

/// One bit per physical CSR edge indicating whether an edge filter admitted it.
///
/// Packed graphs keep their shared targets array borrowed. Materializing only
/// this bitmap makes a filtered Tarjan pass cost `E / 8` bytes instead of a
/// second `4 * E`-byte targets arena. Unfiltered passes use no bitmap.
#[derive(Debug)]
struct AllowedEdgeBitmap {
    words: Vec<u64>,
    edge_count: usize,
}

impl AllowedEdgeBitmap {
    fn try_new(edge_count: usize) -> Result<Self, String> {
        let word_count = edge_count
            .checked_add(63)
            .ok_or_else(|| "Tarjan edge-filter bitmap size overflow".to_string())?
            / 64;
        let mut words = Vec::new();
        words.try_reserve_exact(word_count).map_err(|error| {
            format!("Tarjan edge-filter bitmap allocation failed ({word_count} words): {error}")
        })?;
        words.resize(word_count, 0);
        Ok(Self { words, edge_count })
    }

    #[inline]
    fn set(&mut self, edge_idx: usize) {
        debug_assert!(edge_idx < self.edge_count);
        self.words[edge_idx / 64] |= 1_u64 << (edge_idx % 64);
    }

    /// Return the first admitted edge in `[start, end)`, skipping rejected
    /// words in O(number of bitmap words) rather than O(number of edges).
    #[inline]
    fn next_set(&self, start: u32, end: u32) -> Option<u32> {
        let start = start as usize;
        let end = end as usize;
        debug_assert!(start <= end);
        debug_assert!(end <= self.edge_count);
        if start == end {
            return None;
        }

        let last_word = (end - 1) / 64;
        let mut word_idx = start / 64;
        let mut word = self.words[word_idx] & (u64::MAX << (start % 64));

        loop {
            if word_idx == last_word {
                let tail_bits = end % 64;
                if tail_bits != 0 {
                    word &= (1_u64 << tail_bits) - 1;
                }
            }
            if word != 0 {
                let edge_idx = word_idx * 64 + word.trailing_zeros() as usize;
                return Some(edge_idx as u32);
            }
            if word_idx == last_word {
                return None;
            }
            word_idx += 1;
            word = self.words[word_idx];
        }
    }
}

/// CSR adjacency used by the DFS. A packed in-memory graph lends all three
/// slices to Tarjan; raw in-memory and disk graphs use owned offsets/targets.
#[derive(Debug)]
struct ArenaAdjacency<'g> {
    /// `offsets[node_id..=node_id + 1]` bounds the node's logical edge row.
    offsets: Cow<'g, [u32]>,
    /// Stable dense destination ids in logical edge order.
    targets: Cow<'g, [u32]>,
    /// Present only for a filtered pass over borrowed packed targets.
    allowed: Option<AllowedEdgeBitmap>,
}

impl ArenaAdjacency<'_> {
    #[inline]
    fn row_bounds(&self, node_id: u32) -> (u32, u32) {
        let node_id = node_id as usize;
        (self.offsets[node_id], self.offsets[node_id + 1])
    }

    #[inline]
    fn next_allowed(&self, start: u32, end: u32) -> Option<u32> {
        match &self.allowed {
            Some(allowed) => allowed.next_set(start, end),
            None => (start < end).then_some(start),
        }
    }

    #[inline]
    fn target(&self, edge_idx: u32) -> u32 {
        self.targets[edge_idx as usize]
    }
}

/// Push without allowing `Vec` to invoke the infallible allocation path.
#[inline]
fn try_push<T>(values: &mut Vec<T>, value: T, context: &str) -> Result<(), String> {
    if values.len() == values.capacity() {
        values
            .try_reserve(1)
            .map_err(|error| format!("Tarjan {context} allocation failed: {error}"))?;
    }
    values.push(value);
    Ok(())
}

/// Frame for iterative DFS traversal using arena indices.
///
/// Uses `u32` indices instead of `BehaviorGraphNode` (16 bytes) for compactness.
#[derive(Debug, Clone, Copy)]
enum DfsFrame {
    /// First visit to a node - discover and push to stack.
    Visit { node_id: u32 },
    /// After processing all successors - check for SCC root.
    PostProcess { node_id: u32 },
    /// Process next successor.
    ProcessSuccessor {
        node_id: u32,
        /// Global CSR edge cursor. Source-local indices are recovered by
        /// subtracting `offsets[node_id]` when edge filters are built.
        edge_idx: u32,
    },
}

/// Find all strongly connected components using arena-based iterative Tarjan's algorithm.
///
/// This is the main entry point for SCC detection. It returns all non-trivial
/// SCCs in the graph in reverse topological order (i.e., if SCC A can reach
/// SCC B, then B comes before A in the output).
///
/// Trivial SCCs (single-node without self-loop) are filtered inline during
/// extraction and are NOT included in the result.
///
/// # Arguments
///
/// * `graph` - The behavior graph to analyze
///
/// # Returns
///
/// A `TarjanResult` containing all non-trivial SCCs and statistics
pub(super) fn find_sccs(graph: &BehaviorGraph) -> TarjanResult {
    ArenaTarjan::run(graph, None)
}

/// Find SCCs using Tarjan's algorithm, restricting edges with a predicate (#2704).
///
/// The edge filter receives `(from_info, succ_idx, to, to_info)` where
/// `from_info` and `to_info` are the source and destination records and
/// `succ_idx` is the index in the source's logical successor row. This allows
/// O(1) bitmask checks without redundant graph lookups, matching TLC's inline
/// `getCheckAction()` pattern.
///
/// Trivial SCCs (single-node without self-loop) are filtered inline during
/// extraction and are NOT included in the result.
pub(super) fn find_sccs_with_edge_filter(
    graph: &BehaviorGraph,
    edge_filter: &dyn Fn(&NodeInfo, usize, &BehaviorGraphNode, &NodeInfo) -> bool,
) -> TarjanResult {
    ArenaTarjan::run(graph, Some(edge_filter))
}

/// Find non-trivial SCCs (actual cycles) in the behavior graph.
///
/// This is a convenience function that filters out trivial SCCs
/// (single nodes without self-loops). Returns a full `TarjanResult`
/// so callers can check for algorithm errors. Part of #1817.
///
/// Note: `find_sccs` already filters trivial SCCs inline. This function
/// provides a secondary validation via `is_trivial_scc_in_graph` for tests.
///
/// If `is_trivial_scc_in_graph` detects a missing graph node (invariant violation),
/// the error is recorded in `result.errors` and the SCC is retained
/// (not silently dropped).
#[cfg(test)]
pub(super) fn find_cycles(graph: &BehaviorGraph) -> TarjanResult {
    let mut result = find_sccs(graph);
    let mut trivial_errors = Vec::new();
    result
        .sccs
        .retain(|scc| match is_trivial_scc_in_graph(scc, graph) {
            Ok(true) => false, // trivial -> filter out
            Ok(false) => true, // non-trivial -> keep
            Err(e) => {
                // Missing node invariant violation -> keep SCC and record error
                trivial_errors.push(format!(
                    "is_trivial failed for SCC ({} nodes): {}",
                    scc.len(),
                    e
                ));
                true
            }
        });
    result.errors.extend(trivial_errors);
    result
}

/// Arena-based Tarjan's algorithm.
///
/// Two-phase approach for cache-friendly SCC detection on large graphs:
///
/// 1. **Index phase:** Borrow completed in-memory CSR, or map raw/disk
///    `BehaviorGraphNode`s to contiguous `u32` ids and build owned CSR.
///
/// 2. **DFS phase:** Run iterative Tarjan using only flat `Vec<T>` arrays
///    indexed by `u32`. No hash-map lookups during the DFS itself.
struct ArenaTarjan<'g> {
    /// Number of nodes in the arena.
    node_count: u32,
    /// Map from arena id back to `BehaviorGraphNode` (for SCC output).
    id_to_node: Cow<'g, [BehaviorGraphNode]>,
    /// CSR offsets and targets, borrowed from a packed graph when possible.
    adjacency: ArenaAdjacency<'g>,
    /// Per-node Tarjan state.
    states: Vec<ArenaNodeState>,
    /// Discovery index counter.
    index: u32,
    /// Tarjan stack (arena ids).
    stack: Vec<u32>,
    /// Found SCCs.
    sccs: Vec<Scc>,
    /// Algorithm invariant violations.
    errors: Vec<String>,
    /// Statistics (test-only: not read in production).
    #[cfg(test)]
    stats: TarjanStats,
}

impl<'g> ArenaTarjan<'g> {
    /// Run the full two-phase arena Tarjan algorithm.
    fn run(graph: &'g BehaviorGraph, edge_filter: EdgeFilter<'_>) -> TarjanResult {
        let mut arena = Self::build_arena(graph, edge_filter);
        arena.find_all_sccs();

        TarjanResult {
            sccs: arena.sccs,
            #[cfg(test)]
            stats: arena.stats,
            errors: arena.errors,
        }
    }

    /// Phase 1: borrow an already-packed in-memory CSR, or build an owned CSR
    /// for raw in-memory/disk graphs. The borrowed path retains only Tarjan
    /// state plus (for filtered passes) one eligibility bit per edge.
    fn build_arena(graph: &'g BehaviorGraph, edge_filter: EdgeFilter<'_>) -> Self {
        if let Some(view) = graph.packed_tarjan_view() {
            return Self::build_borrowed_arena(view, edge_filter);
        }
        Self::build_owned_arena(graph, edge_filter)
    }

    fn build_borrowed_arena(
        view: super::graph_store::PackedTarjanView<'g>,
        edge_filter: EdgeFilter<'_>,
    ) -> Self {
        let n = view.node_keys.len();
        let edge_count = view.targets.len();

        if view.node_infos.len() != n {
            return Self::failed(format!(
                "packed Tarjan view has {n} node keys but {} node infos",
                view.node_infos.len()
            ));
        }
        if view.offsets.len() != n.saturating_add(1) {
            return Self::failed(format!(
                "packed Tarjan CSR has {} offsets for {n} nodes (expected {})",
                view.offsets.len(),
                n.saturating_add(1)
            ));
        }
        if n > u32::MAX as usize {
            return Self::failed(format!("packed Tarjan view exceeds u32 node limit: {n}"));
        }
        if edge_count > u32::MAX as usize {
            return Self::failed(format!(
                "packed Tarjan view exceeds u32 edge limit: {edge_count}"
            ));
        }
        if view.offsets.first().copied() != Some(0) {
            return Self::failed("packed Tarjan CSR must start at offset zero".to_string());
        }
        let expected_terminal = edge_count as u32;
        if view.offsets.last().copied() != Some(expected_terminal) {
            return Self::failed(format!(
                "packed Tarjan CSR terminal offset {:?} does not match {edge_count} targets",
                view.offsets.last()
            ));
        }
        for (row, bounds) in view.offsets.windows(2).enumerate() {
            if bounds[0] > bounds[1] || bounds[1] > expected_terminal {
                return Self::failed(format!(
                    "packed Tarjan CSR row {row} has invalid bounds {}..{} for {edge_count} targets",
                    bounds[0], bounds[1]
                ));
            }
        }
        for (edge_idx, &target) in view.targets.iter().enumerate() {
            if target as usize >= n {
                return Self::failed(format!(
                    "packed Tarjan edge {edge_idx} targets out-of-range dense id {target} (nodes={n})"
                ));
            }
        }

        let allowed = match edge_filter {
            None => None,
            Some(filter) => {
                let mut bitmap = match AllowedEdgeBitmap::try_new(edge_count) {
                    Ok(bitmap) => bitmap,
                    Err(error) => return Self::failed(error),
                };
                for source_id in 0..n {
                    let start = view.offsets[source_id] as usize;
                    let end = view.offsets[source_id + 1] as usize;
                    let from_info = &view.node_infos[source_id];
                    for (local_idx, edge_idx) in (start..end).enumerate() {
                        let target_id = view.targets[edge_idx] as usize;
                        if filter(
                            from_info,
                            local_idx,
                            &view.node_keys[target_id],
                            &view.node_infos[target_id],
                        ) {
                            bitmap.set(edge_idx);
                        }
                    }
                }
                Some(bitmap)
            }
        };

        Self::finish(
            Cow::Borrowed(view.node_keys),
            ArenaAdjacency {
                offsets: Cow::Borrowed(view.offsets),
                targets: Cow::Borrowed(view.targets),
                allowed,
            },
            Vec::new(),
        )
    }

    fn build_owned_arena(graph: &BehaviorGraph, edge_filter: EdgeFilter<'_>) -> Self {
        let all_nodes = graph.node_keys();
        let n = all_nodes.len();
        if n > u32::MAX as usize {
            return Self::failed(format!("Tarjan arena exceeds u32 node limit: {n}"));
        }

        let use_dense = tarjan_dense_ids_enabled() && graph.supports_dense_ids();
        let node_to_id: Option<FxHashMap<BehaviorGraphNode, u32>> = if use_dense {
            None
        } else {
            let mut map = FxHashMap::default();
            if let Err(error) = map.try_reserve(n) {
                return Self::failed(format!(
                    "Tarjan node reverse-index allocation failed ({n} nodes): {error}"
                ));
            }
            for (i, node) in all_nodes.iter().enumerate() {
                map.insert(*node, i as u32);
            }
            Some(map)
        };

        let mut offsets = Vec::new();
        if let Err(error) = offsets.try_reserve_exact(n.saturating_add(1)) {
            return Self::failed(format!(
                "Tarjan CSR offset allocation failed ({} offsets): {error}",
                n.saturating_add(1)
            ));
        }
        offsets.push(0_u32);

        let target_estimate = n.saturating_mul(4).min(u32::MAX as usize);
        let mut targets = Vec::new();
        if let Err(error) = targets.try_reserve_exact(target_estimate) {
            return Self::failed(format!(
                "Tarjan CSR target allocation failed ({target_estimate} estimated edges): {error}"
            ));
        }
        let mut errors = Vec::new();

        for (arena_id, node) in all_nodes.iter().enumerate() {
            debug_assert!(
                !use_dense || graph.node_dense_id(node) == Some(arena_id as u32),
                "dense-id / node_keys ordering mismatch at arena_id {arena_id}"
            );

            match graph.try_get_node_info(node) {
                Ok(Some(info)) => {
                    for (local_idx, successor) in info.successors().iter().enumerate() {
                        let Some(successor_id) =
                            Self::resolve_arena_id(graph, node_to_id.as_ref(), successor)
                        else {
                            errors.push(format!(
                                "Tarjan edge from {:?} (arena_id={arena_id}, edge={local_idx}) targets missing node {:?}",
                                node, successor
                            ));
                            continue;
                        };
                        if successor_id as usize >= n {
                            errors.push(format!(
                                "Tarjan edge from {:?} resolves to out-of-range dense id {successor_id} (nodes={n})",
                                node
                            ));
                            continue;
                        }

                        let admitted = match edge_filter {
                            None => true,
                            Some(filter) => match graph.try_get_node_info(successor) {
                                Ok(Some(to_info)) => {
                                    filter(&info, local_idx, successor, &to_info)
                                }
                                Ok(None) => {
                                    errors.push(format!(
                                        "Tarjan edge filter could not read missing destination {:?}",
                                        successor
                                    ));
                                    false
                                }
                                Err(error) => {
                                    errors.push(format!(
                                        "Tarjan edge filter failed to read destination {:?}: {error}",
                                        successor
                                    ));
                                    false
                                }
                            },
                        };
                        if admitted {
                            if targets.len() >= u32::MAX as usize {
                                return Self::failed_with_errors(
                                    errors,
                                    "Tarjan CSR exceeds u32 edge limit".to_string(),
                                );
                            }
                            if let Err(error) =
                                try_push(&mut targets, successor_id, "CSR target")
                            {
                                return Self::failed_with_errors(errors, error);
                            }
                        }
                    }
                }
                Ok(None) => errors.push(format!(
                    "Tarjan DFS visited node {:?} (arena_id={arena_id}) but it is missing from behavior graph — graph construction invariant violated",
                    node
                )),
                Err(error) => errors.push(format!(
                    "Tarjan DFS failed to read node {:?} (arena_id={arena_id}): {error}",
                    node
                )),
            }
            offsets.push(targets.len() as u32);
        }

        Self::finish(
            Cow::Owned(all_nodes),
            ArenaAdjacency {
                offsets: Cow::Owned(offsets),
                targets: Cow::Owned(targets),
                allowed: None,
            },
            errors,
        )
    }

    fn finish(
        id_to_node: Cow<'g, [BehaviorGraphNode]>,
        adjacency: ArenaAdjacency<'g>,
        errors: Vec<String>,
    ) -> Self {
        let n = id_to_node.len();
        if n > tarjan_arena_warn_threshold() {
            let estimated_mb = n.saturating_mul(36) / (1024 * 1024);
            eprintln!(
                "Note: Tarjan SCC arena allocating for {n} nodes (~{estimated_mb} MB). \
                 Consider disk-backed liveness graph for large state spaces."
            );
        }

        let mut states = Vec::new();
        if let Err(error) = states.try_reserve_exact(n) {
            return Self::failed_with_errors(
                errors,
                format!("Tarjan node-state allocation failed ({n} nodes): {error}"),
            );
        }
        states.resize(n, ArenaNodeState::UNVISITED);

        let stack_capacity = n.min(1 << 20);
        let mut stack = Vec::new();
        if let Err(error) = stack.try_reserve_exact(stack_capacity) {
            return Self::failed_with_errors(
                errors,
                format!("Tarjan node-stack allocation failed ({stack_capacity} entries): {error}"),
            );
        }

        Self {
            node_count: n as u32,
            id_to_node,
            adjacency,
            states,
            index: 0,
            stack,
            sccs: Vec::new(),
            errors,
            #[cfg(test)]
            stats: TarjanStats::default(),
        }
    }

    fn failed(error: String) -> Self {
        Self::failed_with_errors(Vec::new(), error)
    }

    fn failed_with_errors(mut errors: Vec<String>, error: String) -> Self {
        errors.push(error);
        Self {
            node_count: 0,
            id_to_node: Cow::Borrowed(&[]),
            adjacency: ArenaAdjacency {
                offsets: Cow::Borrowed(&[0]),
                targets: Cow::Borrowed(&[]),
                allowed: None,
            },
            states: Vec::new(),
            index: 0,
            stack: Vec::new(),
            sccs: Vec::new(),
            errors,
            #[cfg(test)]
            stats: TarjanStats::default(),
        }
    }

    /// Resolve a behavior-graph node to its arena id.
    ///
    /// Both paths return the node's position in `graph.node_keys()`:
    /// - `node_to_id = Some(map)`: historical temporary reverse index, selected
    ///   by the dense-id kill-switch.
    /// - `node_to_id = None`: dense-id path — the store's persistent reverse
    ///   index returns the stable dense id directly (pure in-RAM, even for the
    ///   disk-backed graph).
    #[inline]
    fn resolve_arena_id(
        graph: &BehaviorGraph,
        node_to_id: Option<&FxHashMap<BehaviorGraphNode, u32>>,
        succ: &BehaviorGraphNode,
    ) -> Option<u32> {
        match node_to_id {
            Some(map) => map.get(succ).copied(),
            None => graph.node_dense_id(succ),
        }
    }

    /// Phase 2: Run iterative Tarjan on all unvisited nodes.
    fn find_all_sccs(&mut self) {
        for node_id in 0..self.node_count {
            if self.states[node_id as usize].index == NOT_VISITED {
                if let Err(error) = self.tarjan_iterative(node_id) {
                    self.errors.push(error);
                    break;
                }
            }
        }

        #[cfg(test)]
        {
            self.stats.scc_count = self.sccs.len();
        }
    }

    /// Iterative Tarjan's algorithm starting from a given arena node.
    ///
    /// All lookups are O(1) array indexing -- no hash maps touched.
    fn tarjan_iterative(&mut self, start: u32) -> Result<(), String> {
        let mut dfs_stack = Vec::new();
        try_push(
            &mut dfs_stack,
            DfsFrame::Visit { node_id: start },
            "DFS frame stack",
        )?;

        while let Some(frame) = dfs_stack.pop() {
            match frame {
                DfsFrame::Visit { node_id } => {
                    self.handle_visit(node_id, &mut dfs_stack)?;
                }
                DfsFrame::ProcessSuccessor { node_id, edge_idx } => {
                    self.handle_process_successor(node_id, edge_idx, &mut dfs_stack)?;
                }
                DfsFrame::PostProcess { node_id } => {
                    self.handle_post_process(node_id)?;
                }
            }
        }
        Ok(())
    }

    /// DFS Visit: discover a node, assign index/low_link, push to Tarjan stack.
    #[inline]
    fn handle_visit(&mut self, node_id: u32, dfs_stack: &mut Vec<DfsFrame>) -> Result<(), String> {
        let idx = node_id as usize;

        // Already visited? (can happen when multiple DFS roots queue the same node)
        if self.states[idx].index != NOT_VISITED {
            return Ok(());
        }

        let discovery = self.index;
        self.index += 1;

        self.states[idx].index = discovery;
        self.states[idx].low_link = discovery;
        self.states[idx].on_stack = true;
        try_push(&mut self.stack, node_id, "node stack")?;

        #[cfg(test)]
        {
            self.stats.nodes_processed += 1;
        }

        // Push PostProcess first (it will execute after all successors are done).
        try_push(
            dfs_stack,
            DfsFrame::PostProcess { node_id },
            "DFS frame stack",
        )?;

        // Push first successor processing frame if this node has successors.
        let (start, end) = self.adjacency.row_bounds(node_id);
        if let Some(edge_idx) = self.adjacency.next_allowed(start, end) {
            try_push(
                dfs_stack,
                DfsFrame::ProcessSuccessor { node_id, edge_idx },
                "DFS frame stack",
            )?;
        }
        Ok(())
    }

    /// DFS ProcessSuccessor: handle one successor, push Visit or update low_link.
    #[inline]
    fn handle_process_successor(
        &mut self,
        node_id: u32,
        edge_idx: u32,
        dfs_stack: &mut Vec<DfsFrame>,
    ) -> Result<(), String> {
        let (_start, end) = self.adjacency.row_bounds(node_id);
        let Some(edge_idx) = self.adjacency.next_allowed(edge_idx, end) else {
            return Ok(());
        };
        let succ_id = self.adjacency.target(edge_idx);

        // Enqueue next successor before recursing into this one.
        if let Some(next_edge_idx) = self.adjacency.next_allowed(edge_idx + 1, end) {
            try_push(
                dfs_stack,
                DfsFrame::ProcessSuccessor {
                    node_id,
                    edge_idx: next_edge_idx,
                },
                "DFS frame stack",
            )?;
        }

        if succ_id == node_id {
            self.states[node_id as usize].has_self_loop = true;
        }

        let succ_state = self.states[succ_id as usize];
        if succ_state.index != NOT_VISITED {
            // Successor already visited.
            if succ_state.on_stack {
                let node_state = &mut self.states[node_id as usize];
                node_state.low_link = node_state.low_link.min(succ_state.index);
            }
        } else {
            // Successor not visited -- Visit it; low_link update handled in PostProcess.
            try_push(
                dfs_stack,
                DfsFrame::Visit { node_id: succ_id },
                "DFS frame stack",
            )?;
        }
        Ok(())
    }

    /// DFS PostProcess: update low_link from successors, extract SCC if root.
    #[inline]
    fn handle_post_process(&mut self, node_id: u32) -> Result<(), String> {
        // Collect min low_link from on-stack successors (Tarjan 1972 S3).
        let mut min_low_link = self.states[node_id as usize].low_link;

        let (mut edge_idx, end) = self.adjacency.row_bounds(node_id);
        while let Some(allowed_edge_idx) = self.adjacency.next_allowed(edge_idx, end) {
            let succ_id = self.adjacency.target(allowed_edge_idx);
            let succ_state = self.states[succ_id as usize];
            if succ_state.on_stack {
                min_low_link = min_low_link.min(succ_state.low_link);
            }
            edge_idx = allowed_edge_idx + 1;
        }

        self.states[node_id as usize].low_link = min_low_link;

        // Check if this node is the root of an SCC.
        let state = self.states[node_id as usize];
        if state.low_link == state.index {
            self.extract_scc(node_id)?;
        }
        Ok(())
    }

    /// Pop SCC members from the Tarjan stack and record the SCC.
    ///
    /// Inline trivial SCC filtering: single-node SCCs without self-loops are
    /// skipped (not added to `self.sccs`), avoiding a post-hoc graph lookup.
    fn extract_scc(&mut self, root_id: u32) -> Result<(), String> {
        let mut scc_ids: Vec<u32> = Vec::new();

        loop {
            let top = match self.stack.pop() {
                Some(top) => top,
                None => {
                    self.errors.push(format!(
                        "Tarjan invariant violated: SCC root {:?} \
                         must be on stack (index={}, stack empty, \
                         partial SCC has {} nodes)",
                        self.id_to_node[root_id as usize],
                        self.states[root_id as usize].index,
                        scc_ids.len()
                    ));
                    break;
                }
            };

            self.states[top as usize].on_stack = false;
            try_push(&mut scc_ids, top, "SCC member buffer")?;

            if top == root_id {
                break;
            }
        }

        let scc_size = scc_ids.len();

        // Inline trivial SCC filtering: a single-node SCC without a self-loop
        // is trivial (no actual cycle). Skip it to avoid post-hoc graph lookups.
        if scc_size == 1 && !self.states[scc_ids[0] as usize].has_self_loop {
            return Ok(());
        }

        #[cfg(test)]
        {
            if scc_size > self.stats.max_scc_size {
                self.stats.max_scc_size = scc_size;
            }
            if scc_size > 1 {
                self.stats.nontrivial_sccs += 1;
            }
        }

        // Convert arena ids back to BehaviorGraphNode for the output SCC.
        let mut scc_nodes = Vec::new();
        scc_nodes.try_reserve_exact(scc_size).map_err(|error| {
            format!("Tarjan SCC output allocation failed ({scc_size} nodes): {error}")
        })?;
        for &id in &scc_ids {
            scc_nodes.push(self.id_to_node[id as usize]);
        }
        try_push(&mut self.sccs, Scc::new(scc_nodes), "SCC result")?;
        Ok(())
    }
}

/// Find SCCs that contain accepting nodes (potential liveness violations)
///
/// This is the main function for liveness checking. It finds all non-trivial
/// SCCs that contain at least one accepting tableau node.
///
/// # Arguments
///
/// * `graph` - The behavior graph
/// * `is_accepting` - Function to check if a node is accepting
///
/// # Returns
///
/// `TarjanResult` with SCCs filtered to non-trivial accepting cycles.
/// Callers should check `result.errors` for algorithm invariant violations.
#[cfg(test)]
pub(crate) fn find_accepting_sccs<F>(graph: &BehaviorGraph, is_accepting: F) -> TarjanResult
where
    F: Fn(&BehaviorGraphNode) -> bool,
{
    let mut result = find_sccs(graph);
    // Trivial SCCs (single-node without self-loop) are already filtered inline
    // by extract_scc. Only retain SCCs that contain at least one accepting node.
    result
        .sccs
        .retain(|scc| scc.nodes().iter().any(&is_accepting));
    result
}

#[cfg(test)]
#[path = "tarjan_tests/mod.rs"]
mod tests;
