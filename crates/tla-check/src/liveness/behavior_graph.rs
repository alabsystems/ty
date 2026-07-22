// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Behavior graph for liveness checking
//!
//! The behavior graph is the product of the state graph and the tableau automaton.
//! Each node is a `(state, tableau_node)` pair, and transitions follow both:
//! - The state graph (via the Next relation)
//! - The tableau automaton (via tableau node successors)
//!
//! A liveness violation exists iff there is a reachable accepting cycle in this
//! product graph.
//!
//! # TLC Reference
//!
//! This follows TLC's implementation in:
//! - `tlc2/tool/liveness/GraphNode.java` - Node representation
//! - `tlc2/tool/liveness/TableauNodePtrTable.java` - (fp, tidx) tracking
//! - `tlc2/tool/liveness/LiveCheck.java` - Product graph construction

use crate::error::EvalResult;
use crate::liveness::checker::CheckMask;
use crate::liveness::debug::{liveness_disk_graph_ptr_capacity, use_disk_graph};
use crate::liveness::graph_store::{invariant_error, NodeInfoView, RuntimeGraphStore};
use crate::state::{ArrayState, Fingerprint, State};
use crate::var_index::VarRegistry;
use rustc_hash::FxHashMap;
use std::fmt;
use std::sync::Arc;

/// Initial node-pointer-table capacity for an auto-disk behavior graph when
/// right-sizing is enabled (see
/// [`liveness_ptr_rightsize`](super::debug::liveness_ptr_rightsize)).
///
/// 1M slots (32 MB of ptr-table mmap at 32 B/slot) holds ~786k nodes before the
/// first grow, covering the vast majority of liveness behavior graphs with zero
/// rehashes, while collapsing the historical estimate-sized allocation (e.g.
/// 16.7M slots for cf1s_folklore's ~30k real nodes). Larger graphs grow
/// (rehash-exact) from here.
const AUTO_DISK_INITIAL_PTR_CAPACITY: usize = 1 << 20;

/// A node in the behavior graph: (state fingerprint, tableau node index) pair
///
/// This is the fundamental unit for liveness checking. Two behavior graph nodes
/// are equal iff they have the same state fingerprint AND the same tableau index.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct BehaviorGraphNode {
    /// Fingerprint of the TLA+ state
    pub(crate) state_fp: Fingerprint,
    /// Index of the tableau node
    pub(crate) tableau_idx: usize,
}

impl BehaviorGraphNode {
    /// Create a new behavior graph node
    pub(crate) fn new(state_fp: Fingerprint, tableau_idx: usize) -> Self {
        Self {
            state_fp,
            tableau_idx,
        }
    }

    /// Create from a state and tableau index
    pub(crate) fn from_state(state: &State, tableau_idx: usize) -> Self {
        Self::new(state.fingerprint(), tableau_idx)
    }
}

impl fmt::Display for BehaviorGraphNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, t{})", self.state_fp, self.tableau_idx)
    }
}

/// The behavior graph: product of state graph × tableau
///
/// This structure tracks:
/// - All visited (state, tableau) pairs
/// - Transitions between pairs (for SCC detection)
/// - Parent pointers (for counterexample trace reconstruction)
///
/// States are stored separately from graph topology, deduplicated by fingerprint.
/// Multiple behavior graph nodes with the same state fingerprint (different tableau
/// indices) share a single State entry. This matches TLC's `GraphNode.java` which
/// stores only fingerprints + BitVectors in the graph, not full states.
/// (Approach I from the liveness-architecture design.)
#[derive(Debug)]
pub(crate) struct BehaviorGraph {
    /// Graph topology storage: nodes, edges, parent pointers, check masks.
    /// Part of #2732: runtime-selectable between the historical in-memory map
    /// and the new disk-backed node-record store.
    pub(crate) store: RuntimeGraphStore,
    /// Deduplicated state storage, keyed by fingerprint.
    /// One entry per unique state (shared across tableau indices).
    state_cache: FxHashMap<Fingerprint, State>,
    /// Part of #3065 / flat-state-liveness: Optional shared state cache from the
    /// model checker, storing COMPACT `ArrayState`s (not `im::OrdMap` `State`).
    /// On the fp-only liveness path this holds one entry per distinct reachable
    /// state; the heavy `State`/`OrdMap` form dominated peak RSS (nbacg 74MB).
    /// `get_array_state_by_fp` returns the ArrayState directly (O(1) Arc clone);
    /// `get_state*` reconstruct `State` on demand (lossless) ONLY for the rare
    /// State-needing callers (counterexample trace) — hot paths use ArrayState.
    shared_state_cache: Option<Arc<FxHashMap<Fingerprint, ArrayState>>>,
    /// Compact payloads owned by an on-the-fly liveness checker.
    ///
    /// Unlike `shared_state_cache`, this cache starts empty and is populated
    /// exactly as transient `State` successors are generated. It lets the
    /// regenerated path retain one compact `ArrayState` per fingerprint while
    /// dropping the much heavier local `State` and `Vec<State>` caches.
    /// Fingerprint-only APIs still require `shared_state_cache`: the owned mode
    /// is only safe when exploration supplies each concrete state payload.
    owned_state_cache: Option<FxHashMap<Fingerprint, ArrayState>>,
    /// Registry for reconstructing `State` from compact ArrayStates on demand.
    /// Set together with either compact cache mode.
    compact_state_registry: Option<Arc<VarRegistry>>,
}

/// Information stored for each behavior graph node
///
/// Does NOT store the full State — states are deduplicated in
/// `BehaviorGraph::state_cache` by fingerprint. Use `BehaviorGraph::get_state()`
/// to retrieve the state for a node.
#[derive(Debug, Clone)]
pub(crate) struct NodeInfo {
    /// Successor nodes (Vec for cache-friendly iteration; typical out-degree 2-20)
    pub(crate) successors: Vec<BehaviorGraphNode>,
    /// Parent node (for trace reconstruction)
    pub(crate) parent: Option<BehaviorGraphNode>,
    /// BFS depth
    pub(crate) depth: usize,
    /// Precomputed state check bitmask (#2572, #2890).
    /// Bit i set means `check_state[i]` is true for this node's state.
    /// Populated by `populate_node_check_masks()` after BFS construction.
    /// Uses multi-word `CheckMask` to support >64 check indices.
    pub(crate) state_check_mask: CheckMask,
    /// Precomputed action check bitmasks (#2572, #2890), aligned with `successors`.
    /// `action_check_masks[j]` has bit i set if `check_action[i]` is true for
    /// the transition `(this_node -> successors[j])`.
    /// Populated by `populate_node_check_masks()` after BFS construction.
    /// Uses multi-word `CheckMask` to support >64 check indices.
    pub(crate) action_check_masks: Vec<CheckMask>,
}

impl BehaviorGraph {
    fn with_store(store: RuntimeGraphStore) -> Self {
        Self {
            store,
            state_cache: FxHashMap::default(),
            shared_state_cache: None,
            owned_state_cache: None,
            compact_state_registry: None,
        }
    }

    /// Create a new empty in-memory behavior graph.
    pub(crate) fn new() -> Self {
        Self::with_store(RuntimeGraphStore::new_in_memory())
    }

    /// Create a disk-backed behavior graph with a specific pointer-table capacity.
    pub(crate) fn new_disk_backed(ptr_table_capacity: usize) -> EvalResult<Self> {
        Ok(Self::with_store(RuntimeGraphStore::new_disk_backed(
            ptr_table_capacity,
        )?))
    }

    /// Create a behavior graph using the current runtime liveness storage gate.
    pub(crate) fn new_from_env() -> EvalResult<Self> {
        if use_disk_graph() {
            Self::new_disk_backed(liveness_disk_graph_ptr_capacity())
        } else {
            Ok(Self::new())
        }
    }

    /// Create a behavior graph using the runtime liveness storage gate, with an
    /// optional size hint for auto-detecting when disk-backed storage is needed.
    ///
    /// When `estimated_nodes` exceeds `TY_LIVENESS_AUTO_DISK_THRESHOLD`
    /// (default 2M), the graph automatically uses disk-backed storage to prevent
    /// OOM on multi-property liveness specs (e.g., CoffeeCan3000Beans with 4.5M
    /// states and 5 grouped plans).
    pub(crate) fn new_from_env_with_hint(estimated_nodes: Option<usize>) -> EvalResult<Self> {
        // If explicitly requested via env var, use that unconditionally.
        if use_disk_graph() {
            return Self::new_disk_backed(liveness_disk_graph_ptr_capacity());
        }

        // Auto-detect: if estimated nodes exceed the threshold, use disk-backed.
        use super::debug::{liveness_auto_disk_threshold, liveness_profile, liveness_ptr_rightsize};
        if let Some(est) = estimated_nodes {
            let threshold = liveness_auto_disk_threshold();
            if est > threshold {
                // The auto-disk TRIGGER stays keyed on the (conservative)
                // estimate, but the ALLOCATED ptr-table capacity is decoupled
                // from it. The `states * tableau` estimate wildly over-counts
                // the real product-graph size (most (state, tableau) pairs are
                // inconsistent and never created), so sizing the table to the
                // estimate wastes hundreds of MB. With right-sizing on, start
                // modest and let the table grow (rehash-exact) to the actual
                // node count; off restores the estimate-sized table.
                let capacity = if liveness_ptr_rightsize() {
                    AUTO_DISK_INITIAL_PTR_CAPACITY
                } else {
                    est.next_power_of_two().max(1 << 20)
                };
                if liveness_profile() {
                    eprintln!(
                        "[liveness] auto-disk: estimated {est} nodes > threshold {threshold}, \
                         using disk-backed graph (initial ptr capacity {capacity}, \
                         rightsize={})",
                        liveness_ptr_rightsize()
                    );
                }
                return Self::new_disk_backed(capacity);
            }
        }

        Ok(Self::new())
    }

    #[cfg(test)]
    pub(crate) fn is_disk_backed(&self) -> bool {
        self.store.is_disk_backed()
    }

    /// Part of #3065: Set a shared state cache from the model checker.
    /// When set, `get_state()` checks this cache first, and fingerprint-based
    /// methods can add nodes without cloning State objects.
    pub(crate) fn set_shared_state_cache(
        &mut self,
        cache: Arc<FxHashMap<Fingerprint, ArrayState>>,
        registry: Arc<VarRegistry>,
    ) {
        assert!(
            self.owned_state_cache.is_none(),
            "owned and shared behavior-graph state caches are mutually exclusive"
        );
        self.shared_state_cache = Some(cache);
        self.compact_state_registry = Some(registry);
    }

    /// Enable an initially empty compact payload cache owned by this graph.
    ///
    /// This is for state-taking exploration only. The caller must present every
    /// concrete initial/successor state through the normal `*_with_fp` APIs so
    /// the cache remains exact. In particular, this does not make the
    /// fingerprint-only add APIs safe: those continue to require the
    /// pre-populated shared cache.
    pub(crate) fn enable_owned_state_cache(&mut self, registry: Arc<VarRegistry>) {
        assert!(
            self.shared_state_cache.is_none(),
            "owned and shared behavior-graph state caches are mutually exclusive"
        );
        self.owned_state_cache = Some(FxHashMap::default());
        self.compact_state_registry = Some(registry);
    }

    /// Get the compact `ArrayState` for a fingerprint from either compact cache
    /// mode (O(1) Arc-backed clone). Hot liveness paths use this and bind the
    /// ArrayState for eval, avoiding `im::OrdMap` `State` materialization.
    pub(crate) fn get_array_state_by_fp(&self, fp: Fingerprint) -> Option<ArrayState> {
        self.shared_state_cache
            .as_ref()
            .and_then(|cache| cache.get(&fp).cloned())
            .or_else(|| {
                self.owned_state_cache
                    .as_ref()
                    .and_then(|cache| cache.get(&fp).cloned())
            })
    }

    /// True iff a shared compact `ArrayState` cache has been installed.
    ///
    /// When set, `get_state*` reconstruct `State` lazily from the compact cache,
    /// so the explore path stores successor *fingerprints* (not retained
    /// `Vec<State>`) and `add_successor_with_fp` skips redundant local `State`
    /// retention — the flat-state memory win for the tableau liveness path.
    #[cfg(test)]
    pub(crate) fn has_shared_state_cache(&self) -> bool {
        self.shared_state_cache.is_some()
    }

    /// True iff either compact cache mode can resolve concrete state payloads.
    ///
    /// State-taking exploration uses this to retain successor fingerprints
    /// instead of `Vec<State>`. Keep this distinct from
    /// `has_shared_state_cache`: by-fingerprint exploration has no payloads to
    /// insert and therefore still requires the pre-populated shared mode.
    pub(crate) fn has_compact_state_cache(&self) -> bool {
        self.shared_state_cache.is_some() || self.owned_state_cache.is_some()
    }

    /// True iff this graph owns the compact state payload cache.
    ///
    /// Unlike the shared-cache path, owned compact exploration records the
    /// complete pre-tableau-pruning successor fingerprint list while concrete
    /// successors are available. Callers that must distinguish that guarantee
    /// from a shared fingerprint cache use this narrow predicate.
    pub(crate) fn has_owned_state_cache(&self) -> bool {
        self.owned_state_cache.is_some()
    }

    /// Capture a concrete state in the owned compact cache, if enabled.
    /// Conversion happens only for a previously unseen fingerprint.
    pub(crate) fn cache_owned_state(&mut self, fp: Fingerprint, state: &State) {
        let Some(cache) = self.owned_state_cache.as_mut() else {
            return;
        };
        let registry = self
            .compact_state_registry
            .as_deref()
            .expect("compact_state_registry set together with owned_state_cache");
        cache
            .entry(fp)
            .or_insert_with(|| ArrayState::from_state(state, registry));
    }

    #[cfg(test)]
    pub(crate) fn remove_owned_state_for_test(&mut self, fp: Fingerprint) {
        self.owned_state_cache
            .as_mut()
            .expect("owned state cache enabled for missing-payload test")
            .remove(&fp);
    }

    /// True iff either compact cache already holds `fp` (so the local
    /// full-`State` cache need not retain a second copy).
    fn compact_cache_contains(&self, fp: Fingerprint) -> bool {
        self.shared_state_cache
            .as_ref()
            .is_some_and(|cache| cache.contains_key(&fp))
            || self
                .owned_state_cache
                .as_ref()
                .is_some_and(|cache| cache.contains_key(&fp))
    }

    /// Part of #3065: Add an initial node by fingerprint only (no State clone).
    /// Requires shared_state_cache to be set. Returns true if newly added.
    pub(crate) fn try_add_init_node_by_fp(
        &mut self,
        fp: Fingerprint,
        tableau_idx: usize,
    ) -> EvalResult<bool> {
        let node = BehaviorGraphNode::new(fp, tableau_idx);
        self.store.add_init_node(node)
    }

    /// Part of #3065: Add a successor node by fingerprint only (no State clone).
    /// Requires shared_state_cache to be set. Returns true if successor is new.
    pub(crate) fn add_successor_by_fp(
        &mut self,
        from: BehaviorGraphNode,
        to_fp: Fingerprint,
        to_tableau_idx: usize,
    ) -> EvalResult<bool> {
        let to_node = BehaviorGraphNode::new(to_fp, to_tableau_idx);
        self.store.add_successor(from, to_node)
    }

    /// Add an initial node to the behavior graph
    ///
    /// Returns true if the node was newly added, false if it already existed.
    #[cfg(test)]
    pub fn add_init_node(&mut self, state: &State, tableau_idx: usize) -> bool {
        self.try_add_init_node(state, tableau_idx)
            .expect("in-memory behavior graph add_init_node should not fail")
    }

    pub(crate) fn try_add_init_node(
        &mut self,
        state: &State,
        tableau_idx: usize,
    ) -> EvalResult<bool> {
        self.try_add_init_node_with_fp(state, state.fingerprint(), tableau_idx)
    }

    /// Add an initial node using an explicit behavior-graph fingerprint.
    pub(crate) fn try_add_init_node_with_fp(
        &mut self,
        state: &State,
        state_fp: Fingerprint,
        tableau_idx: usize,
    ) -> EvalResult<bool> {
        self.cache_owned_state(state_fp, state);
        let node = BehaviorGraphNode::new(state_fp, tableau_idx);
        if !self.store.add_init_node(node)? {
            return Ok(false);
        }
        // Skip local full-`State` retention when a compact ArrayState cache
        // already holds this state — `get_state_by_fp` reconstructs it
        // losslessly on demand (flat-state memory win). Only states absent from
        // both compact modes fall back to the local cache, so completeness is
        // preserved exactly.
        if !self.compact_cache_contains(node.state_fp) {
            self.state_cache
                .entry(node.state_fp)
                .or_insert_with(|| state.clone());
        }
        Ok(true)
    }

    /// Add a successor node to the behavior graph
    ///
    /// Returns true if the successor was newly added, false if it already existed.
    pub(crate) fn add_successor(
        &mut self,
        from: BehaviorGraphNode,
        to_state: &State,
        to_tableau_idx: usize,
    ) -> EvalResult<bool> {
        self.add_successor_with_fp(from, to_state, to_state.fingerprint(), to_tableau_idx)
    }

    /// Add a successor node using an explicit behavior-graph fingerprint.
    pub(crate) fn add_successor_with_fp(
        &mut self,
        from: BehaviorGraphNode,
        to_state: &State,
        to_fp: Fingerprint,
        to_tableau_idx: usize,
    ) -> EvalResult<bool> {
        self.cache_owned_state(to_fp, to_state);
        let to_node = BehaviorGraphNode::new(to_fp, to_tableau_idx);
        let is_new = self.store.add_successor(from, to_node)?;
        if is_new && !self.compact_cache_contains(to_node.state_fp) {
            // See `try_add_init_node_with_fp`: only retain a local `State` when
            // the compact caches lack it. This is the dominant flat-state
            // win for the tableau path — one fresh `im::OrdMap` per distinct
            // state previously doubled peak RSS (nbacg).
            self.state_cache
                .entry(to_node.state_fp)
                .or_insert_with(|| to_state.clone());
        }
        Ok(is_new)
    }

    /// Check if a behavior graph node has been visited.
    #[cfg(test)]
    pub(crate) fn contains(&self, node: &BehaviorGraphNode) -> bool {
        self.store.contains(*node)
    }

    /// Get information about a node for in-memory callers and tests.
    // Test-only convenience; production liveness uses try_get_node_info for error propagation.
    #[cfg(test)]
    pub fn get_node_info(&self, node: &BehaviorGraphNode) -> Option<NodeInfoView<'_>> {
        self.try_get_node_info(node)
            .expect("in-memory behavior graph get_node_info should not fail")
    }

    /// Get information about a node, propagating disk-backed storage errors.
    pub(crate) fn try_get_node_info(
        &self,
        node: &BehaviorGraphNode,
    ) -> EvalResult<Option<NodeInfoView<'_>>> {
        self.store.get_node_info(node)
    }

    /// Update a node's topology record in place.
    pub(crate) fn update_node_info<R>(
        &mut self,
        node: &BehaviorGraphNode,
        update: impl FnOnce(&mut NodeInfo) -> R,
    ) -> EvalResult<Option<R>> {
        self.store.update_node_info(node, update)
    }

    #[cfg(test)]
    pub fn get_node_info_mut(&mut self, node: &BehaviorGraphNode) -> Option<&mut NodeInfo> {
        self.store.get_node_info_mut(node)
    }

    /// Get the state for a behavior graph node (from deduplicated state cache).
    ///
    /// Looks up the state by fingerprint only — does NOT verify that the specific
    /// `(state_fp, tableau_idx)` pair exists in the graph topology. All callers
    /// should ensure the node is in the graph before calling this (e.g., from BFS
    /// queue, SCC iteration, or after a `get_node_info()` check).
    /// Returns an OWNED `State`. The shared cache stores compact `ArrayState`s
    /// reconstructed on demand (lossless `to_state`); the local `state_cache`
    /// (full-state path) holds `State`s and is cloned (im::OrdMap O(1) share).
    /// Hot liveness paths must prefer `get_array_state_by_fp` + bind the
    /// ArrayState — reconstructing many `State`s here defeats the purpose.
    pub(crate) fn get_state(&self, node: &BehaviorGraphNode) -> Option<State> {
        self.get_state_by_fp(node.state_fp)
    }

    /// Get a state directly by fingerprint (OWNED, reconstructed from the shared
    /// ArrayState cache on demand). Used only by State-needing callers
    /// (counterexample trace); hot paths use `get_array_state_by_fp`.
    pub(crate) fn get_state_by_fp(&self, fp: Fingerprint) -> Option<State> {
        if let Some(ref shared) = self.shared_state_cache {
            if let Some(arr) = shared.get(&fp) {
                let registry = self
                    .compact_state_registry
                    .as_ref()
                    .expect("compact_state_registry set together with shared_state_cache");
                return Some(arr.to_state(registry));
            }
        }
        if let Some(ref owned) = self.owned_state_cache {
            if let Some(arr) = owned.get(&fp) {
                let registry = self
                    .compact_state_registry
                    .as_ref()
                    .expect("compact_state_registry set together with owned_state_cache");
                return Some(arr.to_state(registry));
            }
        }
        self.state_cache.get(&fp).cloned()
    }

    /// Get initial nodes in insertion order.
    #[cfg(test)]
    pub(crate) fn init_nodes(&self) -> Vec<BehaviorGraphNode> {
        self.store.init_nodes()
    }

    /// Get the number of nodes in the behavior graph.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.store.node_count()
    }

    /// Get all node keys.
    ///
    /// For the disk-backed store the returned order is the dense-id order:
    /// `node_keys()[i]` is the node whose [`Self::node_dense_id`] is `i`.
    pub(crate) fn node_keys(&self) -> Vec<BehaviorGraphNode> {
        self.store.node_keys()
    }

    /// True iff nodes can be resolved to a stable dense contiguous `u32` id
    /// (disk-backed store only). See [`Self::node_dense_id`].
    pub(crate) fn supports_dense_ids(&self) -> bool {
        self.store.supports_dense_ids()
    }

    /// Resolve a node to its stable dense contiguous id, or `None` if the node
    /// is absent or the store has no dense-id ordering.
    ///
    /// Pure in-RAM lookup (no disk read) — safe for the Tarjan SCC hot path.
    /// Invariant: `node_keys()[node_dense_id(n)] == n` for every node `n`.
    pub(crate) fn node_dense_id(&self, node: &BehaviorGraphNode) -> Option<u32> {
        self.store.dense_id(*node)
    }

    /// Resolve a fingerprint-only path to concrete states via the state cache.
    ///
    /// Part of #3746: Tolerates missing states from the non-atomic seen_fps/seen
    /// insert race in shared-cache parallel mode. Missing states are skipped
    /// instead of producing a hard error, yielding a potentially shorter trace.
    /// Checker-owned compact exploration has no such race: every referenced
    /// payload must be present, so a missing state is an invariant error.
    pub(crate) fn resolve_fingerprint_trace(
        &self,
        trace: &[(Fingerprint, usize)],
    ) -> EvalResult<Vec<(State, usize)>> {
        let mut resolved: Vec<(State, usize)> = Vec::with_capacity(trace.len());
        for (state_fp, tableau_idx) in trace {
            let node = BehaviorGraphNode::new(*state_fp, *tableau_idx);
            if let Some(state) = self.get_state(&node) {
                resolved.push((state, *tableau_idx));
            } else if self.has_owned_state_cache() {
                return Err(invariant_error(format!(
                    "owned compact cache is missing trace payload {state_fp} at tableau node {tableau_idx}"
                )));
            }
        }
        if resolved.is_empty() && !trace.is_empty() {
            return Err(invariant_error(format!(
                "trace reconstruction: all {} state(s) missing from cache — \
                 cannot produce any counterexample trace",
                trace.len(),
            )));
        }
        Ok(resolved)
    }

    /// Reconstruct a trace from an initial state to the given node
    // Currently test-only convenience; production liveness uses fingerprint traces.
    // Retained for future counterexample trace reconstruction.
    #[cfg(test)]
    pub fn reconstruct_trace(&self, end: BehaviorGraphNode) -> EvalResult<Vec<(State, usize)>> {
        let trace = self.reconstruct_fingerprint_trace(end)?;
        self.resolve_fingerprint_trace(&trace)
    }

    /// Reconstruct a fingerprint-only trace from an initial node to `end`.
    pub(crate) fn reconstruct_fingerprint_trace(
        &self,
        end: BehaviorGraphNode,
    ) -> EvalResult<Vec<(Fingerprint, usize)>> {
        self.store.reconstruct_fingerprint_trace(end)
    }
}

impl Default for BehaviorGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "behavior_graph_tests.rs"]
mod tests;
