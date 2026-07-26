// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Behavior graph topology storage backends.
//!
//! Part of #2732: this module keeps liveness graph topology storage separate
//! from concrete-state caching. The default backend remains an in-memory
//! `FxHashMap`, but Slice E also needs a runtime-selectable disk backend that
//! persists node records and exposes explicit update operations instead of
//! long-lived `&mut NodeInfo` borrows.

use crate::error::{EvalError, EvalResult};
use crate::liveness::behavior_graph::{BehaviorGraphNode, NodeInfo};
use crate::liveness::checker::{ActionCheckMatrix, CheckMask};
use crate::liveness::debug::liveness_inmemory_node_limit;
use crate::liveness::storage::disk_graph::DiskGraphStore;
use crate::state::Fingerprint;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::fmt::Display;
use std::ops::Deref;
use tempfile::TempDir;

/// Invariant violation error for graph store operations.
pub(crate) fn invariant_error(message: String) -> EvalError {
    EvalError::Internal {
        message: format!("behavior graph invariant violated: {message}"),
        span: None,
    }
}

/// 2026-07 OOM audit: coarse cadence (growth operations between polls) for
/// the live memory guard on behavior-graph growth. One growth operation is a
/// node or edge insertion; at ~200 bytes per `NodeInfo` the between-poll
/// overshoot is bounded to ~13 MB while the poll (one OS probe) stays
/// amortized to nothing.
const GRAPH_GROWTH_MEMORY_POLL_INTERVAL: usize = 65_536;

/// Live memory pressure probe for the post-BFS liveness phase.
///
/// The in-memory store's node-count cap is an ITEM cap: it ignores per-item
/// payload bytes (successor `Vec`s,
/// check masks, retained witness states), so byte-honest protection needs an
/// RSS poll. The checker's configured `MemoryPolicy` object is not reachable
/// from the store, so this reconstructs an equivalent one:
///
/// - When the user set an explicit `--memory-limit`, honor THAT grant (it is
///   published process-globally by `set_memory_limit`). The 2026-07-02 audit
///   follow-up caught the earlier version freezing the auto-detected
///   per-instance share in a `OnceLock`: on a host running a concurrent
///   `cargo`/`rustc` build the one-shot instance count made that share several
///   times smaller than an explicit limit, so a liveness run could decline
///   with tens of GB of granted headroom still unused.
/// - Otherwise fall back to the auto-detected
///   [`crate::memory::MemoryPolicy::from_system_default`] limit
///   (confinement/cgroup-capped share of RAM), cached in a `OnceLock` since
///   its process-counting probe is the only expensive part.
///
/// Either way the same Critical threshold and collective free-memory floor
/// apply. Fail-soft: if policy construction or the RSS probe fails the guard
/// never fires (`false`) — a probe failure must never abort a run.
fn liveness_growth_memory_pressure_critical() -> bool {
    // Explicit user limit takes precedence, re-read each poll (it is set once
    // at checker setup, before this post-BFS phase; `MemoryPolicy::new` is a
    // cheap value construction — the RSS/floor probes live in `check`).
    if let Some(limit) = crate::memory::configured_memory_limit_bytes() {
        return crate::memory::MemoryPolicy::new(limit).check()
            == crate::memory::MemoryPressure::Critical;
    }
    static POLICY: std::sync::OnceLock<Option<crate::memory::MemoryPolicy>> =
        std::sync::OnceLock::new();
    POLICY
        .get_or_init(crate::memory::MemoryPolicy::from_system_default)
        .as_ref()
        .is_some_and(|policy| policy.check() == crate::memory::MemoryPressure::Critical)
}

/// Read-only node info access that can be borrowed (in-memory) or owned (disk).
#[derive(Clone)]
pub(crate) enum NodeInfoView<'a> {
    Borrowed {
        info: &'a NodeInfo,
        successors: SuccessorRow<'a>,
    },
    Owned(NodeInfo),
}

impl NodeInfoView<'_> {
    /// Return this node's logical successor row.
    ///
    /// The row preserves insertion order, duplicate edges, and the exact edge
    /// index used by [`ActionCheckMatrix`]. Callers must use this accessor
    /// rather than reaching through to `NodeInfo::successors`: the in-memory
    /// backend may store completed rows outside `NodeInfo` in a compact layout.
    #[inline]
    pub(crate) fn successors(&self) -> SuccessorRow<'_> {
        match self {
            Self::Borrowed { successors, .. } => *successors,
            Self::Owned(info) => SuccessorRow::Raw(&info.successors),
        }
    }

    /// Convert to an owned `NodeInfo`, cloning if borrowed.
    #[cfg(test)]
    pub(crate) fn into_owned(self) -> NodeInfo {
        let successors = self.successors().iter().copied().collect();
        let mut info = match self {
            Self::Borrowed { info, .. } => info.clone(),
            Self::Owned(info) => info,
        };
        // Keep this test-only compatibility conversion topology-faithful when
        // an in-memory view is backed by a compact row outside `NodeInfo`.
        info.successors = successors;
        info
    }
}

/// Cheap read-only view of one logical successor row.
///
/// The physical adjacency container stays hidden. Edge order, duplicates, and
/// indices are part of the logical graph contract.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SuccessorRow<'a> {
    /// The graph's current per-node `Vec` representation.
    Raw(&'a [BehaviorGraphNode]),
    /// A completed in-memory row stored as dense ids in shared CSR arrays.
    Dense {
        ids: &'a [u32],
        node_keys: &'a [BehaviorGraphNode],
    },
}

impl<'a> SuccessorRow<'a> {
    #[inline]
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Raw(row) => row.len(),
            Self::Dense { ids, .. } => ids.len(),
        }
    }

    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        self.len() == 0
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn get(self, index: usize) -> Option<&'a BehaviorGraphNode> {
        match self {
            Self::Raw(row) => row.get(index),
            Self::Dense { ids, node_keys } => ids
                .get(index)
                .and_then(|dense_id| node_keys.get(*dense_id as usize)),
        }
    }

    #[inline]
    pub(crate) fn iter(self) -> SuccessorIter<'a> {
        match self {
            Self::Raw(row) => SuccessorIter::Raw(row.iter()),
            Self::Dense { ids, node_keys } => SuccessorIter::Dense {
                ids: ids.iter(),
                node_keys,
            },
        }
    }

    #[inline]
    pub(crate) fn contains(self, node: &BehaviorGraphNode) -> bool {
        self.iter().any(|successor| successor == node)
    }

    #[inline]
    pub(crate) fn position(self, node: &BehaviorGraphNode) -> Option<usize> {
        self.iter().position(|successor| successor == node)
    }
}

/// Iterator over a logical successor row.
///
/// Items remain references so existing edge filters can borrow a successor
/// without copying it. A compact dense-id row can map ids through the stable
/// node-key array in another iterator variant without changing callers.
pub(crate) enum SuccessorIter<'a> {
    Raw(std::slice::Iter<'a, BehaviorGraphNode>),
    Dense {
        ids: std::slice::Iter<'a, u32>,
        node_keys: &'a [BehaviorGraphNode],
    },
}

impl<'a> Iterator for SuccessorIter<'a> {
    type Item = &'a BehaviorGraphNode;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Raw(iter) => iter.next(),
            Self::Dense { ids, node_keys } => Some(&node_keys[*ids.next()? as usize]),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Raw(iter) => iter.size_hint(),
            Self::Dense { ids, .. } => ids.size_hint(),
        }
    }
}

impl DoubleEndedIterator for SuccessorIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Raw(iter) => iter.next_back(),
            Self::Dense { ids, node_keys } => Some(&node_keys[*ids.next_back()? as usize]),
        }
    }
}

impl ExactSizeIterator for SuccessorIter<'_> {}
impl std::iter::FusedIterator for SuccessorIter<'_> {}

impl<'a> IntoIterator for SuccessorRow<'a> {
    type Item = &'a BehaviorGraphNode;
    type IntoIter = SuccessorIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Deref for NodeInfoView<'_> {
    type Target = NodeInfo;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed { info, .. } => info,
            Self::Owned(info) => info,
        }
    }
}

/// Dense CSR adjacency installed after in-memory graph construction completes.
#[derive(Debug, Clone)]
struct PackedSuccessors {
    /// Row boundaries in `targets`; always exactly `node_count + 1` entries.
    offsets: Vec<u32>,
    /// Stable dense destination ids in logical edge insertion order.
    targets: Vec<u32>,
}

impl PackedSuccessors {
    #[inline]
    fn row(&self, dense_id: u32) -> Option<&[u32]> {
        let dense_id = dense_id as usize;
        let start = *self.offsets.get(dense_id)? as usize;
        let end = *self.offsets.get(dense_id.checked_add(1)?)? as usize;
        self.targets.get(start..end)
    }
}

/// Borrowed completed in-memory topology for arena algorithms.
///
/// Dense target ids index both `node_keys` and `node_infos`; offsets delimit
/// one logical row per dense source id. Raw in-memory and disk stores do not
/// expose this view.
#[derive(Clone, Copy)]
pub(crate) struct PackedTarjanView<'a> {
    pub(crate) node_keys: &'a [BehaviorGraphNode],
    pub(crate) node_infos: &'a [NodeInfo],
    pub(crate) offsets: &'a [u32],
    pub(crate) targets: &'a [u32],
}

/// In-memory behavior graph topology store with a compact reverse index.
///
/// `node_ids` contains only `(BehaviorGraphNode, u32)` entries; the much larger
/// [`NodeInfo`] values live densely in `node_infos`. Besides avoiding
/// `NodeInfo` storage in every empty hash bucket, insertion order becomes a
/// stable dense id. Tarjan can therefore reuse this reverse index instead of
/// allocating a second node-to-id hash table at peak RSS. Keys and payloads
/// use separate vectors so node-key scans touch only the compact contiguous
/// key array rather than striding through every large `NodeInfo`.
#[derive(Debug, Clone)]
pub(crate) struct InMemoryGraphStore {
    node_ids: FxHashMap<BehaviorGraphNode, u32>,
    node_keys: Vec<BehaviorGraphNode>,
    node_infos: Vec<NodeInfo>,
    /// Completed adjacency. While present, every `NodeInfo::successors` is
    /// empty and topology is immutable; check-mask fields remain mutable.
    packed_successors: Option<PackedSuccessors>,
    pub(crate) init_nodes_list: Vec<BehaviorGraphNode>,
    node_limit: Option<usize>,
}

impl InMemoryGraphStore {
    fn validate_action_mask_alignment(
        node: BehaviorGraphNode,
        info: &NodeInfo,
        row_len: usize,
    ) -> EvalResult<()> {
        let masks_unpopulated = info.action_check_masks.len() == 0
            && info.action_check_masks.check_count() == 0
            && info.action_check_masks.as_words().is_empty();
        if !masks_unpopulated && info.action_check_masks.len() != row_len {
            return Err(invariant_error(format!(
                "cannot pack in-memory successors: action-check rows ({}) do not align with successor row ({row_len}) for node {node}",
                info.action_check_masks.len(),
            )));
        }
        Ok(())
    }

    pub(crate) fn new() -> Self {
        Self {
            node_ids: FxHashMap::default(),
            node_keys: Vec::new(),
            node_infos: Vec::new(),
            packed_successors: None,
            init_nodes_list: Vec::new(),
            node_limit: liveness_inmemory_node_limit(),
        }
    }

    #[cfg(test)]
    #[inline]
    fn contains(&self, node: BehaviorGraphNode) -> bool {
        self.node_ids.contains_key(&node)
    }

    #[inline]
    fn dense_id(&self, node: BehaviorGraphNode) -> Option<u32> {
        self.node_ids.get(&node).copied()
    }

    #[inline]
    fn successor_row_by_dense_id(&self, dense_id: u32) -> Option<SuccessorRow<'_>> {
        let info = self.node_infos.get(dense_id as usize)?;
        match &self.packed_successors {
            Some(packed) => Some(SuccessorRow::Dense {
                ids: packed.row(dense_id)?,
                node_keys: &self.node_keys,
            }),
            None => Some(SuccessorRow::Raw(&info.successors)),
        }
    }

    #[inline]
    fn node_info_view(&self, node: &BehaviorGraphNode) -> Option<NodeInfoView<'_>> {
        let dense_id = self.dense_id(*node)?;
        debug_assert_eq!(self.node_keys.get(dense_id as usize), Some(node));
        let info = self.node_infos.get(dense_id as usize)?;
        let successors = self.successor_row_by_dense_id(dense_id)?;
        Some(NodeInfoView::Borrowed { info, successors })
    }

    #[inline]
    fn node_info_mut(&mut self, node: &BehaviorGraphNode) -> Option<&mut NodeInfo> {
        if self.packed_successors.is_some() {
            return None;
        }
        let dense_id = self.dense_id(*node)? as usize;
        debug_assert_eq!(self.node_keys.get(dense_id), Some(node));
        self.node_infos.get_mut(dense_id)
    }

    fn insert_if_absent_with(
        &mut self,
        node: BehaviorGraphNode,
        make_info: impl FnOnce() -> NodeInfo,
    ) -> EvalResult<bool> {
        if self.packed_successors.is_some() {
            return Err(invariant_error(format!(
                "cannot add node {node}: in-memory liveness graph topology is packed"
            )));
        }
        let node_count = self.node_infos.len();
        let node_limit = self.node_limit;
        match self.node_ids.entry(node) {
            Entry::Occupied(_) => Ok(false),
            Entry::Vacant(entry) => {
                if let Some(limit) = node_limit {
                    if node_count >= limit {
                        return Err(EvalError::Internal {
                            message: format!(
                                "in-memory liveness graph node limit exceeded: limit={limit}, attempted node {node}; enable disk-backed liveness graph or raise TY_LIVENESS_INMEMORY_NODE_LIMIT"
                            ),
                            span: None,
                        });
                    }
                }
                // Tarjan stores the node count in `u32`, so leave the final
                // value available as a count rather than assigning it as an id.
                if node_count >= u32::MAX as usize {
                    return Err(invariant_error(format!(
                        "in-memory liveness graph cannot exceed u32::MAX nodes: {node}"
                    )));
                }
                let dense_id = node_count as u32;
                entry.insert(dense_id);
                self.node_keys.push(node);
                self.node_infos.push(make_info());
                debug_assert_eq!(self.node_keys.len(), self.node_infos.len());
                Ok(true)
            }
        }
    }

    /// Convert completed raw successor rows to stable-dense-id CSR.
    ///
    /// Validation and all fallible allocation finish before the first raw row
    /// is drained. After draining begins, exact capacity guarantees make every
    /// push infallible, so an error can never leave a half-converted graph.
    fn pack_successors(&mut self) -> EvalResult<()> {
        if let Some(packed) = &self.packed_successors {
            for (dense_id, (&node, info)) in self.node_keys.iter().zip(&self.node_infos).enumerate()
            {
                let row_len = packed
                    .row(dense_id as u32)
                    .ok_or_else(|| {
                        invariant_error(format!(
                            "cannot validate packed in-memory successors: missing row for node {node}"
                        ))
                    })?
                    .len();
                Self::validate_action_mask_alignment(node, info, row_len)?;
            }
            return Ok(());
        }

        let node_count = self.node_infos.len();
        if self.node_keys.len() != node_count || self.node_ids.len() != node_count {
            return Err(invariant_error(format!(
                "cannot pack in-memory successors: dense topology cardinalities disagree (ids={}, keys={}, infos={node_count})",
                self.node_ids.len(),
                self.node_keys.len(),
            )));
        }
        if u32::try_from(node_count).is_err() {
            return Err(invariant_error(format!(
                "cannot pack in-memory successors: node count {node_count} exceeds u32::MAX"
            )));
        }
        let offset_count = node_count.checked_add(1).ok_or_else(|| {
            invariant_error(format!(
                "cannot pack in-memory successors: offset count overflows for {node_count} nodes"
            ))
        })?;

        let mut edge_count = 0usize;
        for (dense_id, (node, info)) in self.node_keys.iter().zip(&self.node_infos).enumerate() {
            let expected_id = u32::try_from(dense_id).map_err(|_| {
                invariant_error(format!(
                    "cannot pack in-memory successors: dense id {dense_id} exceeds u32::MAX"
                ))
            })?;
            if self.node_ids.get(node).copied() != Some(expected_id) {
                return Err(invariant_error(format!(
                    "cannot pack in-memory successors: reverse index disagrees at dense id {dense_id} for node {node}"
                )));
            }

            let row_len = info.successors.len();
            edge_count = edge_count.checked_add(row_len).ok_or_else(|| {
                invariant_error(format!(
                    "cannot pack in-memory successors: edge count overflows at node {node}"
                ))
            })?;
            Self::validate_action_mask_alignment(*node, info, row_len)?;
            for target in &info.successors {
                if !self.node_ids.contains_key(target) {
                    return Err(invariant_error(format!(
                        "cannot pack in-memory successors: edge {node} -> {target} references a missing target"
                    )));
                }
            }
        }
        if u32::try_from(edge_count).is_err() {
            return Err(invariant_error(format!(
                "cannot pack in-memory successors: edge count {edge_count} exceeds u32::MAX"
            )));
        }

        let mut offsets = Vec::new();
        offsets.try_reserve_exact(offset_count).map_err(|error| {
            invariant_error(format!(
                "cannot allocate {offset_count} packed successor offsets: {error}"
            ))
        })?;
        let mut targets = Vec::new();
        targets.try_reserve_exact(edge_count).map_err(|error| {
            invariant_error(format!(
                "cannot allocate {edge_count} packed successor targets: {error}"
            ))
        })?;

        // No fallible operation is permitted below this point until the
        // completed representation is installed.
        offsets.push(0);
        let node_ids = &self.node_ids;
        for info in &mut self.node_infos {
            for target in std::mem::take(&mut info.successors) {
                let target_id = *node_ids
                    .get(&target)
                    .expect("packed successor target validated before topology mutation");
                targets.push(target_id);
            }
            offsets.push(
                u32::try_from(targets.len())
                    .expect("packed successor edge count validated before topology mutation"),
            );
        }
        debug_assert_eq!(offsets.len(), offset_count);
        debug_assert_eq!(targets.len(), edge_count);
        self.packed_successors = Some(PackedSuccessors { offsets, targets });
        Ok(())
    }

    #[inline]
    fn packed_tarjan_view(&self) -> Option<PackedTarjanView<'_>> {
        let packed = self.packed_successors.as_ref()?;
        Some(PackedTarjanView {
            node_keys: &self.node_keys,
            node_infos: &self.node_infos,
            offsets: &packed.offsets,
            targets: &packed.targets,
        })
    }
}

impl Default for InMemoryGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum GraphStoreBackend {
    InMemory(InMemoryGraphStore),
    Disk {
        store: RefCell<DiskGraphStore>,
        _backing_dir: TempDir,
    },
}

/// Runtime-selectable topology store used by [`super::behavior_graph::BehaviorGraph`].
#[derive(Debug)]
pub(crate) struct RuntimeGraphStore {
    backend: GraphStoreBackend,
    /// 2026-07 OOM audit: growth operations (node/edge insertions) since the
    /// last live memory poll; see [`Self::check_growth_memory_budget`].
    growth_ops_since_memory_poll: usize,
}

impl RuntimeGraphStore {
    pub(crate) fn new_in_memory() -> Self {
        Self {
            backend: GraphStoreBackend::InMemory(InMemoryGraphStore::new()),
            growth_ops_since_memory_poll: 0,
        }
    }

    pub(crate) fn new_disk_backed(ptr_table_capacity: usize) -> EvalResult<Self> {
        let backing_dir = TempDir::new().map_err(|error| {
            invariant_error(format!(
                "create disk-backed liveness graph tempdir: {error}"
            ))
        })?;
        let store = DiskGraphStore::with_capacity(backing_dir.path(), ptr_table_capacity).map_err(
            |error| {
                invariant_error(format!(
                    "create disk-backed liveness graph store in {}: {error}",
                    backing_dir.path().display()
                ))
            },
        )?;
        Ok(Self {
            backend: GraphStoreBackend::Disk {
                store: RefCell::new(store),
                _backing_dir: backing_dir,
            },
            growth_ops_since_memory_poll: 0,
        })
    }

    fn map_disk_error(context: &str, error: impl Display) -> EvalError {
        invariant_error(format!("{context}: {error}"))
    }

    /// 2026-07 OOM audit: live memory poll on the graph growth loops.
    ///
    /// The post-BFS liveness phase (behavior-graph construction) previously
    /// had ZERO RSS polls; its only guard was the node-count cap, which is
    /// blind to per-item edge/witness payload bytes. Polls every
    /// [`GRAPH_GROWTH_MEMORY_POLL_INTERVAL`] insertions; on a hit, returns
    /// the same fail-closed `EvalError::Internal` shape as the node-count
    /// cap, which the production explore loops propagate (`?`) into the
    /// liveness checker's RuntimeFailure / "liveness could not complete"
    /// path — the run declines as inconclusive, never a wrong verdict.
    fn check_growth_memory_budget(&mut self) -> EvalResult<()> {
        self.growth_ops_since_memory_poll += 1;
        if self.growth_ops_since_memory_poll < GRAPH_GROWTH_MEMORY_POLL_INTERVAL {
            return Ok(());
        }
        self.growth_ops_since_memory_poll = 0;
        if liveness_growth_memory_pressure_critical() {
            return Err(EvalError::Internal {
                message: "liveness graph growth stopped: process memory footprint reached \
                          the critical threshold of the auto-detected memory limit; \
                          liveness checking could not complete (enable the disk-backed \
                          liveness graph or raise the memory limit)"
                    .to_string(),
                span: None,
            });
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, node: BehaviorGraphNode) -> bool {
        match &self.backend {
            GraphStoreBackend::InMemory(store) => store.contains(node),
            GraphStoreBackend::Disk { store, .. } => store.borrow().contains(node),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_disk_backed(&self) -> bool {
        matches!(self.backend, GraphStoreBackend::Disk { .. })
    }

    /// True iff this backend can resolve a node to a stable dense contiguous
    /// `u32` id. The in-memory store assigns insertion-order ids in its compact
    /// reverse index; the disk store keeps the same invariant in its pointer
    /// table.
    pub(crate) fn supports_dense_ids(&self) -> bool {
        true
    }

    /// Return the node's stable dense contiguous id, or `None` if the node is
    /// absent or the backend does not support dense ids.
    ///
    /// Pure in-RAM lookup for both backends, so it is safe on the Tarjan SCC
    /// hot path.
    pub(crate) fn dense_id(&self, node: BehaviorGraphNode) -> Option<u32> {
        match &self.backend {
            GraphStoreBackend::InMemory(store) => store.dense_id(node),
            GraphStoreBackend::Disk { store, .. } => store.borrow().dense_id_of(node),
        }
    }

    /// Reject topology growth after the completed in-memory graph has moved
    /// its successor rows into immutable CSR. Disk topology remains mutable.
    pub(crate) fn ensure_topology_mutable(&self) -> EvalResult<()> {
        if matches!(
            &self.backend,
            GraphStoreBackend::InMemory(store) if store.packed_successors.is_some()
        ) {
            return Err(invariant_error(
                "cannot mutate in-memory liveness graph after successor packing".to_string(),
            ));
        }
        Ok(())
    }

    /// Pack completed in-memory successor rows. Returns `true` for an in-memory
    /// graph (including an already-packed graph) and `false` for disk storage.
    pub(crate) fn pack_inmemory_successors(&mut self) -> EvalResult<bool> {
        match &mut self.backend {
            GraphStoreBackend::InMemory(store) => {
                store.pack_successors()?;
                Ok(true)
            }
            GraphStoreBackend::Disk { .. } => Ok(false),
        }
    }

    /// Borrow packed in-memory topology for Tarjan without cloning it.
    #[inline]
    pub(crate) fn packed_tarjan_view(&self) -> Option<PackedTarjanView<'_>> {
        match &self.backend {
            GraphStoreBackend::InMemory(store) => store.packed_tarjan_view(),
            GraphStoreBackend::Disk { .. } => None,
        }
    }

    pub(crate) fn add_init_node(&mut self, node: BehaviorGraphNode) -> EvalResult<bool> {
        self.ensure_topology_mutable()?;
        self.check_growth_memory_budget()?;
        match &mut self.backend {
            GraphStoreBackend::InMemory(store) => {
                let inserted = store.insert_if_absent_with(node, || NodeInfo {
                    successors: Vec::new(),
                    trace_parent: None,
                    state_check_mask: CheckMask::new(),
                    action_check_masks: ActionCheckMatrix::new(),
                })?;
                if inserted {
                    store.init_nodes_list.push(node);
                }
                Ok(inserted)
            }
            GraphStoreBackend::Disk { store, .. } => {
                let mut store = store.borrow_mut();
                if store.contains(node) {
                    return Ok(false);
                }
                let info = NodeInfo {
                    successors: Vec::new(),
                    trace_parent: None,
                    state_check_mask: CheckMask::new(),
                    action_check_masks: ActionCheckMatrix::new(),
                };
                store
                    .append_node(node, &info)
                    .map_err(|error| Self::map_disk_error("append initial liveness node", error))?;
                store.mark_init_node(node);
                Ok(true)
            }
        }
    }

    pub(crate) fn add_successor(
        &mut self,
        from: BehaviorGraphNode,
        to: BehaviorGraphNode,
    ) -> EvalResult<bool> {
        self.ensure_topology_mutable()?;
        self.check_growth_memory_budget()?;
        match &mut self.backend {
            GraphStoreBackend::InMemory(store) => {
                {
                    let from_info = store.node_info_mut(&from).ok_or_else(|| {
                        invariant_error(format!(
                            "cannot add successor edge from {from} to {to}: source node is missing"
                        ))
                    })?;
                    from_info.successors.push(to);
                    // Any topology mutation invalidates the edge-aligned masks.
                    // They are repopulated after graph construction.
                    from_info.action_check_masks = ActionCheckMatrix::new();
                }
                store.insert_if_absent_with(to, || NodeInfo {
                    successors: Vec::new(),
                    trace_parent: None,
                    state_check_mask: CheckMask::new(),
                    action_check_masks: ActionCheckMatrix::new(),
                })
            }
            GraphStoreBackend::Disk { store, .. } => {
                let mut store = store.borrow_mut();
                let mut from_info = store
                    .read_node(from)
                    .map_err(|error| Self::map_disk_error("read source liveness node", error))?
                    .ok_or_else(|| {
                        invariant_error(format!(
                            "cannot add successor edge from {from} to {to}: source node is missing"
                        ))
                    })?;
                from_info.successors.push(to);
                // A disk read may have normalized an unpopulated zero-width
                // matrix to the old successor count. Reset it before writing
                // the changed topology so alignment is unambiguous.
                from_info.action_check_masks = ActionCheckMatrix::new();
                if !store.contains(to) {
                    let to_info = NodeInfo {
                        successors: Vec::new(),
                        trace_parent: Some(Box::new(from)),
                        state_check_mask: CheckMask::new(),
                        action_check_masks: ActionCheckMatrix::new(),
                    };
                    store.append_node(to, &to_info).map_err(|error| {
                        Self::map_disk_error("append successor liveness node", error)
                    })?;
                    store.update_node(from, &from_info).map_err(|error| {
                        Self::map_disk_error("persist updated source liveness node", error)
                    })?;
                    return Ok(true);
                }
                store.update_node(from, &from_info).map_err(|error| {
                    Self::map_disk_error("persist updated source liveness node", error)
                })?;
                Ok(false)
            }
        }
    }

    pub(crate) fn get_node_info<'a>(
        &'a self,
        node: &BehaviorGraphNode,
    ) -> EvalResult<Option<NodeInfoView<'a>>> {
        match &self.backend {
            GraphStoreBackend::InMemory(store) => Ok(store.node_info_view(node)),
            GraphStoreBackend::Disk { store, .. } => {
                let info = store
                    .borrow_mut()
                    .read_node(*node)
                    .map_err(|error| Self::map_disk_error("read liveness node", error))?;
                Ok(info.map(NodeInfoView::Owned))
            }
        }
    }

    /// Inspect a node's logical successor row without exposing its physical
    /// adjacency representation.
    ///
    /// The callback form lets an in-memory backend lend a compact row while a
    /// disk backend lends from the decoded `NodeInfo` it owns for this call.
    /// The callback result cannot borrow the row, so no disk-local reference
    /// can escape.
    pub(crate) fn with_successors<R>(
        &self,
        node: &BehaviorGraphNode,
        inspect: impl for<'row> FnOnce(SuccessorRow<'row>) -> R,
    ) -> EvalResult<Option<R>> {
        match &self.backend {
            GraphStoreBackend::InMemory(store) => {
                let Some(dense_id) = store.dense_id(*node) else {
                    return Ok(None);
                };
                Ok(store.successor_row_by_dense_id(dense_id).map(inspect))
            }
            GraphStoreBackend::Disk { store, .. } => {
                let info = store
                    .borrow_mut()
                    .read_node(*node)
                    .map_err(|error| Self::map_disk_error("read liveness successor row", error))?;
                Ok(info
                    .as_ref()
                    .map(|info| inspect(SuccessorRow::Raw(&info.successors))))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn update_node_info<R>(
        &mut self,
        node: &BehaviorGraphNode,
        update: impl FnOnce(&mut NodeInfo) -> R,
    ) -> EvalResult<Option<R>> {
        self.ensure_topology_mutable()?;
        match &mut self.backend {
            GraphStoreBackend::InMemory(store) => Ok(store.node_info_mut(node).map(update)),
            GraphStoreBackend::Disk { store, .. } => {
                let mut store = store.borrow_mut();
                let mut info = match store
                    .read_node(*node)
                    .map_err(|error| Self::map_disk_error("read liveness node for update", error))?
                {
                    Some(info) => info,
                    None => return Ok(None),
                };
                let result = update(&mut info);
                store.update_node(*node, &info).map_err(|error| {
                    Self::map_disk_error("persist updated liveness node", error)
                })?;
                Ok(Some(result))
            }
        }
    }

    /// Update the check masks aligned with a node's logical successor row.
    ///
    /// The narrow split-borrow API is intentional: completed in-memory graphs
    /// may move adjacency out of `NodeInfo`, while mask repopulation remains
    /// legal on retry paths. Disk records continue to own raw successor rows.
    pub(crate) fn update_node_masks<R>(
        &mut self,
        node: &BehaviorGraphNode,
        update: impl for<'row> FnOnce(SuccessorRow<'row>, &mut CheckMask, &mut ActionCheckMatrix) -> R,
    ) -> EvalResult<Option<R>> {
        match &mut self.backend {
            GraphStoreBackend::InMemory(store) => {
                let Some(dense_id) = store.dense_id(*node) else {
                    return Ok(None);
                };
                debug_assert_eq!(store.node_keys.get(dense_id as usize), Some(node));
                let dense_idx = dense_id as usize;
                match &store.packed_successors {
                    Some(packed) => {
                        let Some(ids) = packed.row(dense_id) else {
                            return Ok(None);
                        };
                        let node_keys = &store.node_keys;
                        let Some(info) = store.node_infos.get_mut(dense_idx) else {
                            return Ok(None);
                        };
                        Ok(Some(update(
                            SuccessorRow::Dense { ids, node_keys },
                            &mut info.state_check_mask,
                            &mut info.action_check_masks,
                        )))
                    }
                    None => {
                        let Some(info) = store.node_infos.get_mut(dense_idx) else {
                            return Ok(None);
                        };
                        let NodeInfo {
                            successors,
                            state_check_mask,
                            action_check_masks,
                            ..
                        } = info;
                        Ok(Some(update(
                            SuccessorRow::Raw(successors),
                            state_check_mask,
                            action_check_masks,
                        )))
                    }
                }
            }
            GraphStoreBackend::Disk { store, .. } => {
                let mut store = store.borrow_mut();
                let mut info = match store.read_node(*node).map_err(|error| {
                    Self::map_disk_error("read liveness node for mask update", error)
                })? {
                    Some(info) => info,
                    None => return Ok(None),
                };
                let result = {
                    let NodeInfo {
                        successors,
                        state_check_mask,
                        action_check_masks,
                        ..
                    } = &mut info;
                    update(
                        SuccessorRow::Raw(successors),
                        state_check_mask,
                        action_check_masks,
                    )
                };
                store.update_node(*node, &info).map_err(|error| {
                    Self::map_disk_error("persist updated liveness node masks", error)
                })?;
                Ok(Some(result))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn get_node_info_mut(&mut self, node: &BehaviorGraphNode) -> Option<&mut NodeInfo> {
        match &mut self.backend {
            GraphStoreBackend::InMemory(store) => store.node_info_mut(node),
            GraphStoreBackend::Disk { .. } => None,
        }
    }

    pub(crate) fn reconstruct_fingerprint_trace(
        &self,
        end: BehaviorGraphNode,
    ) -> EvalResult<Vec<(Fingerprint, usize)>> {
        const UNSEEN: u32 = u32::MAX;

        match &self.backend {
            GraphStoreBackend::InMemory(store) => {
                let end_id = store.dense_id(end).ok_or_else(|| {
                    invariant_error(format!(
                        "fingerprint trace reconstruction reached missing node {end}"
                    ))
                })?;
                let node_count = store.node_infos.len();
                let mut parents = Vec::new();
                parents.try_reserve_exact(node_count).map_err(|error| {
                    invariant_error(format!(
                        "cannot allocate in-memory trace parent scratch for {node_count} nodes: {error}"
                    ))
                })?;
                parents.resize(node_count, UNSEEN);
                let mut queue = Vec::new();
                queue.try_reserve_exact(node_count).map_err(|error| {
                    invariant_error(format!(
                        "cannot allocate in-memory trace BFS scratch for {node_count} nodes: {error}"
                    ))
                })?;
                let mut queue_head = 0usize;
                for &init in &store.init_nodes_list {
                    let init_id = store.dense_id(init).ok_or_else(|| {
                        invariant_error(format!(
                            "fingerprint trace reconstruction reached missing initial node {init}"
                        ))
                    })?;
                    if parents[init_id as usize] == UNSEEN {
                        parents[init_id as usize] = init_id;
                        queue.push(init_id);
                    }
                }

                while parents[end_id as usize] == UNSEEN {
                    let source_id = *queue.get(queue_head).ok_or_else(|| {
                        invariant_error(format!(
                            "fingerprint trace reconstruction could not reach node {end} from any initial node"
                        ))
                    })?;
                    queue_head += 1;
                    if let Some(packed) = &store.packed_successors {
                        let successor_ids = packed.row(source_id).ok_or_else(|| {
                            invariant_error(format!(
                                "fingerprint trace reconstruction reached missing packed dense node {source_id}"
                            ))
                        })?;
                        for &successor_id in successor_ids {
                            if parents[successor_id as usize] == UNSEEN {
                                parents[successor_id as usize] = source_id;
                                queue.push(successor_id);
                            }
                        }
                    } else {
                        let successors = store.successor_row_by_dense_id(source_id).ok_or_else(|| {
                            invariant_error(format!(
                                "fingerprint trace reconstruction reached missing dense node {source_id}"
                            ))
                        })?;
                        for successor in successors {
                            let successor_id = store.dense_id(*successor).ok_or_else(|| {
                                invariant_error(format!(
                                    "fingerprint trace reconstruction reached missing successor node {successor}"
                                ))
                            })?;
                            if parents[successor_id as usize] == UNSEEN {
                                parents[successor_id as usize] = source_id;
                                queue.push(successor_id);
                            }
                        }
                    }
                }

                let mut trace = Vec::new();
                let mut current_id = end_id;
                loop {
                    let node = *store.node_keys.get(current_id as usize).ok_or_else(|| {
                        invariant_error(format!(
                            "fingerprint trace reconstruction reached missing dense node {current_id}"
                        ))
                    })?;
                    trace.push((node.state_fp, node.tableau_idx));
                    let parent_id = parents[current_id as usize];
                    if parent_id == current_id {
                        break;
                    }
                    if parent_id == UNSEEN {
                        return Err(invariant_error(format!(
                            "fingerprint trace reconstruction found no parent for node {node}"
                        )));
                    }
                    current_id = parent_id;
                }
                trace.reverse();
                Ok(trace)
            }
            GraphStoreBackend::Disk { store, .. } => {
                let mut trace = Vec::new();
                let mut current = Some(end);
                while let Some(node) = current {
                    let info = store
                        .borrow_mut()
                        .read_node(node)
                        .map_err(|error| {
                            Self::map_disk_error(
                                "read liveness node during fingerprint trace reconstruction",
                                error,
                            )
                        })?
                        .ok_or_else(|| {
                            invariant_error(format!(
                                "fingerprint trace reconstruction reached missing node {node}"
                            ))
                        })?;
                    trace.push((node.state_fp, node.tableau_idx));
                    current = info.trace_parent.as_deref().copied();
                }
                trace.reverse();
                Ok(trace)
            }
        }
    }

    pub(crate) fn node_keys(&self) -> Vec<BehaviorGraphNode> {
        match &self.backend {
            GraphStoreBackend::InMemory(store) => store.node_keys.clone(),
            GraphStoreBackend::Disk { store, .. } => store.borrow().all_nodes().to_vec(),
        }
    }

    #[allow(dead_code)] // Used in tests via BehaviorGraph::len and disk_graph_tests
    pub(crate) fn node_count(&self) -> usize {
        match &self.backend {
            GraphStoreBackend::InMemory(store) => store.node_infos.len(),
            GraphStoreBackend::Disk { store, .. } => store.borrow().node_count(),
        }
    }

    #[cfg(test)]
    pub(crate) fn init_nodes(&self) -> Vec<BehaviorGraphNode> {
        match &self.backend {
            GraphStoreBackend::InMemory(store) => store.init_nodes_list.clone(),
            GraphStoreBackend::Disk { store, .. } => store.borrow().init_nodes().to_vec(),
        }
    }
}
