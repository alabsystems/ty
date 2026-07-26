// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Successor graph abstraction with in-memory and disk-backed backends.
//!
//! Part of #3176: replaces the raw `FxHashMap<Fingerprint, Vec<Fingerprint>>`
//! in `LivenessCacheState.successors` with a dispatch enum that can use either
//! an in-memory HashMap (default) or a disk-backed store for large state spaces.

use super::disk_successor_graph::DiskSuccessorGraph;
use crate::check::CheckError;
use crate::liveness::debug::{
    liveness_effective_unit_limit, liveness_inmemory_successor_limit,
    LIVENESS_BYTES_PER_SUCCESSOR_ENTRY,
};
use crate::state::Fingerprint;
use crate::EvalError;
use rustc_hash::{FxHashMap, FxHashSet};

/// In-memory successor cache with an optional entry budget.
///
/// When the entry count reaches the configured limit, the
/// `SuccessorGraph` auto-migrates all entries to a disk-backed store
/// instead of returning a hard error. This prevents BFS failures while moving
/// successor payloads out of RAM; the disk backend still keeps an O(states)
/// parent-offset index in memory.
///
/// Tier-1 #4: the migration gate is byte-aware. The `entry_limit` is derived
/// from the smaller of the historical entry-count limit
/// (`TY_LIVENESS_INMEMORY_SUCCESSOR_LIMIT`) and the byte budget
/// (`TY_LIVENESS_DISK_BUDGET_MB`) converted into an equivalent entry count via
/// `LIVENESS_BYTES_PER_SUCCESSOR_ENTRY`. The count comparison in
/// [`Self::should_migrate`] stays O(1), so when the byte budget is disabled
/// (default) the migration decision is unchanged. The conversion uses a fixed
/// per-entry estimate; a precise running-bytes gate is a documented follow-up.
pub(crate) struct InMemorySuccessorGraph {
    map: FxHashMap<Fingerprint, Vec<Fingerprint>>,
    entry_limit: Option<usize>,
}

impl InMemorySuccessorGraph {
    fn new() -> Self {
        Self {
            map: FxHashMap::default(),
            // Fold the byte budget into the entry limit so the count gate is
            // also tightened to whichever bound is smaller. Preserves the
            // historical entry-count limit when the byte budget is disabled.
            entry_limit: liveness_effective_unit_limit(
                liveness_inmemory_successor_limit(),
                LIVENESS_BYTES_PER_SUCCESSOR_ENTRY,
            ),
        }
    }

    /// Check if inserting a new entry would exceed the in-memory budget.
    ///
    /// Returns `true` when the entry count reaches the (byte-budget-aware)
    /// configured limit, signaling that the caller should migrate to disk
    /// BEFORE inserting. Overwrites of existing entries never trigger migration
    /// (they do not grow the entry count).
    fn should_migrate(&self, parent_fp: Fingerprint) -> bool {
        if self.map.contains_key(&parent_fp) {
            return false; // Overwrite of existing entry — no growth.
        }
        if let Some(limit) = self.entry_limit {
            return self.map.len() >= limit;
        }
        false
    }

    /// Insert unconditionally (caller has already checked migration threshold).
    fn insert_unchecked(&mut self, parent_fp: Fingerprint, successors: Vec<Fingerprint>) {
        self.map.insert(parent_fp, successors);
    }
}

/// Successor graph for BFS liveness caching.
///
/// Stores `parent_fp -> [child_fps]` relationships discovered during BFS.
/// Read during post-BFS liveness checking for behavior graph construction
/// and SCC analysis.
pub(crate) enum SuccessorGraph {
    /// In-memory HashMap (default, fast, but O(states × avg_succs × 8) bytes).
    InMemory(InMemorySuccessorGraph),
    /// Append-only O(states + edges) file with a resident O(states) hash index
    /// and fixed-slot direct cache. Cached Vec bytes depend on fanout.
    Disk(DiskSuccessorGraph),
}

impl Default for SuccessorGraph {
    fn default() -> Self {
        Self::in_memory()
    }
}

impl SuccessorGraph {
    /// Create an in-memory successor graph (default).
    pub(crate) fn in_memory() -> Self {
        SuccessorGraph::InMemory(InMemorySuccessorGraph::new())
    }

    /// Create a disk-backed successor graph.
    ///
    /// Returns `Err` if the backing temp file cannot be created.
    pub(crate) fn disk() -> std::io::Result<Self> {
        Ok(SuccessorGraph::Disk(DiskSuccessorGraph::new()?))
    }

    /// Move an in-memory graph to disk immediately and set its direct-cache size.
    ///
    /// Unlike the insertion-time entry-limit gate, this is an explicit phase
    /// transition used when the liveness memory guard wants to retain only
    /// adjacency. Creating the backing file and writer clone happens before the
    /// in-memory map is drained, so those recoverable creation errors leave the
    /// original graph untouched. Record-write failures retain the disk backend's
    /// existing fatal-error contract.
    ///
    /// Returns `Ok(true)` when migration occurred and `Ok(false)` when the graph
    /// was already disk-backed. A new disk graph uses exactly `cache_slots`; an
    /// existing disk graph is shrunk to at most that many slots and never grown.
    /// The O(states) parent-offset index is retained.
    pub(crate) fn migrate_to_disk_with_cache_slots(
        &mut self,
        cache_slots: usize,
    ) -> std::io::Result<bool> {
        if let SuccessorGraph::Disk(disk) = self {
            disk.shrink_cache_to(cache_slots);
            return Ok(false);
        }

        let mut disk = DiskSuccessorGraph::with_cache_slots(cache_slots)?;
        let SuccessorGraph::InMemory(graph) = self else {
            unreachable!("disk successor graph handled above");
        };
        for (fp, successors) in graph.map.drain() {
            disk.insert(fp, successors);
        }
        *self = SuccessorGraph::Disk(disk);
        Ok(true)
    }

    /// Insert a parent fingerprint and its successor list.
    ///
    /// Part of #4080: When the in-memory backend reaches its entry limit
    /// (`TY_LIVENESS_INMEMORY_SUCCESSOR_LIMIT`, default 5M), all
    /// entries are automatically migrated to a disk-backed store. This
    /// converts a previously hard BFS failure into graceful degradation by
    /// removing edge-list heap growth. Resident metadata remains O(states).
    pub(crate) fn insert(
        &mut self,
        parent_fp: Fingerprint,
        successors: Vec<Fingerprint>,
    ) -> Result<(), CheckError> {
        match self {
            SuccessorGraph::InMemory(graph) => {
                if graph.should_migrate(parent_fp) {
                    // Migrate all existing entries to disk.
                    let mut disk = DiskSuccessorGraph::new().map_err(|e| EvalError::Internal {
                        message: format!(
                            "failed to create disk successor graph during \
                                 auto-migration: {e}"
                        ),
                        span: None,
                    })?;

                    let entry_count = graph.map.len();
                    eprintln!(
                        "Note: liveness successor cache auto-migrating to disk \
                         ({entry_count} entries at configured limit). \
                         Edge lists will be stored on disk; an O(states) offset \
                         index and fixed-slot read cache remain in memory."
                    );

                    // Drain in-memory map into disk.
                    for (fp, succs) in graph.map.drain() {
                        disk.insert(fp, succs);
                    }
                    // Insert the current entry that triggered migration.
                    disk.insert(parent_fp, successors);

                    *self = SuccessorGraph::Disk(disk);
                    Ok(())
                } else {
                    graph.insert_unchecked(parent_fp, successors);
                    Ok(())
                }
            }
            SuccessorGraph::Disk(disk) => {
                disk.insert(parent_fp, successors);
                Ok(())
            }
        }
    }

    /// Look up successor fingerprints for a parent.
    ///
    /// Returns `None` if the parent was never inserted.
    ///
    /// For the in-memory backend this clones the Vec; for disk it reads from
    /// the file (or cache). Both paths return owned data.
    ///
    /// Takes `&self` (not `&mut self`) because the disk backend uses interior
    /// mutability (`RefCell`) for its read cache.
    pub(crate) fn get(&self, fp: &Fingerprint) -> Option<Vec<Fingerprint>> {
        match self {
            SuccessorGraph::InMemory(graph) => graph.map.get(fp).cloned(),
            SuccessorGraph::Disk(disk) => disk.get(fp),
        }
    }

    /// Whether a parent entry exists, without cloning or reading its payload.
    pub(crate) fn contains_parent(&self, fp: &Fingerprint) -> bool {
        match self {
            SuccessorGraph::InMemory(graph) => graph.map.contains_key(fp),
            SuccessorGraph::Disk(disk) => disk.contains_parent(fp),
        }
    }

    /// Borrow successor fingerprints for a parent (in-memory backend only).
    ///
    /// Returns `None` if the parent was never inserted or if the backend is
    /// disk-backed (disk reads require owned data). Callers that only iterate
    /// over successors should prefer this over [`get`] to avoid cloning the
    /// entire `Vec<Fingerprint>` on every lookup.
    ///
    /// Part of #4080: eliminates unnecessary clone() on the hot path for
    /// in-memory successor lookups during liveness checking.
    pub(crate) fn get_ref(&self, fp: &Fingerprint) -> Option<&[Fingerprint]> {
        match self {
            SuccessorGraph::InMemory(graph) => graph.map.get(fp).map(|v| v.as_slice()),
            SuccessorGraph::Disk(_) => None,
        }
    }

    /// Access the inner HashMap (in-memory backend only).
    ///
    /// Returns `None` for disk backend. Used by the safety-temporal path
    /// which iterates all entries — a pattern only feasible in-memory.
    /// When using disk backend, the safety-temporal path falls through
    /// to the SCC checker instead.
    pub(crate) fn as_inner_map(&self) -> Option<&FxHashMap<Fingerprint, Vec<Fingerprint>>> {
        match self {
            SuccessorGraph::InMemory(graph) => Some(&graph.map),
            SuccessorGraph::Disk(_) => None,
        }
    }

    /// Number of distinct parent entries.
    pub(crate) fn len(&self) -> usize {
        match self {
            SuccessorGraph::InMemory(graph) => graph.map.len(),
            SuccessorGraph::Disk(disk) => disk.len(),
        }
    }

    /// Total number of successor fingerprints across all entries (for diagnostics).
    pub(crate) fn total_successors(&self) -> usize {
        match self {
            SuccessorGraph::InMemory(graph) => graph.map.values().map(Vec::len).sum(),
            SuccessorGraph::Disk(disk) => disk.total_successors(),
        }
    }

    /// Collect all fingerprints referenced by the successor graph.
    ///
    /// This is used by cold-path fp-only replay fallback to identify states
    /// that must be reconstructed even when the primary BFS replay seed is
    /// unavailable. Both backends return the same parent+successor set.
    pub(crate) fn collect_all_fingerprints(&self) -> FxHashSet<Fingerprint> {
        match self {
            SuccessorGraph::InMemory(graph) => {
                let total_successors: usize = graph.map.values().map(Vec::len).sum();
                let mut fingerprints = FxHashSet::with_capacity_and_hasher(
                    graph.map.len().saturating_add(total_successors),
                    Default::default(),
                );
                for (&parent_fp, successors) in &graph.map {
                    fingerprints.insert(parent_fp);
                    fingerprints.extend(successors.iter().copied());
                }
                fingerprints
            }
            SuccessorGraph::Disk(disk) => disk.collect_all_fingerprints(),
        }
    }

    /// Discard all entries.
    pub(crate) fn clear(&mut self) {
        match self {
            SuccessorGraph::InMemory(graph) => graph.map.clear(),
            SuccessorGraph::Disk(disk) => disk.clear(),
        }
    }

    /// Whether this is the disk backend.
    pub(crate) fn is_disk(&self) -> bool {
        matches!(self, SuccessorGraph::Disk(_))
    }

    /// Selected structural heap estimate for the successor graph.
    ///
    /// Part of #4080: OOM safety — liveness cache memory accounting.
    /// For the in-memory backend: HashMap overhead + per-entry Vec allocations.
    /// For the disk backend, only a 20-byte-per-parent offset-index heuristic is
    /// counted. Direct-cache Vecs, mapped/page-cache residency, and allocator
    /// overhead are intentionally excluded.
    pub(crate) fn estimate_memory_bytes(&self) -> usize {
        match self {
            SuccessorGraph::InMemory(graph) => {
                let capacity = graph.map.capacity();
                let entry_size = std::mem::size_of::<Fingerprint>()
                    .saturating_add(std::mem::size_of::<Vec<Fingerprint>>())
                    .saturating_add(1);
                let table_bytes = capacity.saturating_mul(entry_size);
                let vec_heap_bytes: usize = graph
                    .map
                    .values()
                    .map(|v| {
                        v.capacity()
                            .saturating_mul(std::mem::size_of::<Fingerprint>())
                    })
                    .sum();
                crate::memory::apply_fragmentation_overhead(
                    table_bytes.saturating_add(vec_heap_bytes),
                )
            }
            SuccessorGraph::Disk(disk) => {
                // Disk backend: only the in-memory index (offset map).
                disk.len().saturating_mul(20)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_ref_returns_slice_for_inmemory() {
        let mut graph = SuccessorGraph::in_memory();
        let parent = Fingerprint(1);
        let child_a = Fingerprint(10);
        let child_b = Fingerprint(20);
        graph.insert(parent, vec![child_a, child_b]).unwrap();

        let slice = graph
            .get_ref(&parent)
            .expect("in-memory get_ref should return Some");
        assert_eq!(slice, &[child_a, child_b]);
    }

    #[test]
    fn test_get_ref_returns_none_for_missing() {
        let graph = SuccessorGraph::in_memory();
        assert!(graph.get_ref(&Fingerprint(999)).is_none());
    }

    #[test]
    fn test_get_ref_returns_none_for_disk() {
        let graph = SuccessorGraph::disk().unwrap();
        assert!(graph.get_ref(&Fingerprint(1)).is_none());
    }

    #[test]
    fn test_get_and_get_ref_consistent() {
        let mut graph = SuccessorGraph::in_memory();
        let parent = Fingerprint(42);
        let children = vec![Fingerprint(100), Fingerprint(200), Fingerprint(300)];
        graph.insert(parent, children.clone()).unwrap();

        let owned = graph.get(&parent).unwrap();
        let borrowed = graph.get_ref(&parent).unwrap();
        assert_eq!(owned, borrowed);
    }

    fn assert_contains_parent_tracks_sources(mut graph: SuccessorGraph) {
        let parent_with_child = Fingerprint(1);
        let empty_parent = Fingerprint(2);
        let destination_only = Fingerprint(3);
        graph
            .insert(parent_with_child, vec![destination_only])
            .unwrap();
        graph.insert(empty_parent, Vec::new()).unwrap();

        assert!(graph.contains_parent(&parent_with_child));
        assert!(graph.contains_parent(&empty_parent));
        assert!(!graph.contains_parent(&destination_only));
    }

    #[test]
    fn test_contains_parent_for_inmemory_sources_including_empty() {
        assert_contains_parent_tracks_sources(SuccessorGraph::in_memory());
    }

    #[test]
    fn test_contains_parent_for_disk_sources_including_empty() {
        assert_contains_parent_tracks_sources(SuccessorGraph::disk().unwrap());
    }

    #[test]
    fn test_auto_migration_to_disk_on_limit() {
        // Create in-memory graph with a tiny limit for testing.
        let mut graph = SuccessorGraph::InMemory(InMemorySuccessorGraph {
            map: FxHashMap::default(),
            entry_limit: Some(10),
        });

        // Insert exactly 10 entries; the configured limit is still allowed.
        for i in 0..10 {
            graph
                .insert(Fingerprint(i), vec![Fingerprint(i + 100)])
                .expect("insert within limit should succeed");
        }
        assert!(
            !graph.is_disk(),
            "graph should stay in memory while at the configured limit"
        );

        // The 11th insert would exceed the configured limit, so migrate first.
        graph
            .insert(Fingerprint(10), vec![Fingerprint(110)])
            .expect("insert that triggers migration should succeed");

        assert!(
            graph.is_disk(),
            "graph should have auto-migrated to disk backend after reaching the limit"
        );

        // Verify all 11 entries survived migration.
        assert_eq!(graph.len(), 11);
        for i in 0..11 {
            let succs = graph
                .get(&Fingerprint(i))
                .expect("entry should survive migration");
            assert_eq!(succs, vec![Fingerprint(i + 100)]);
        }

        // Further inserts should work on the disk backend.
        graph
            .insert(Fingerprint(99), vec![Fingerprint(199)])
            .expect("post-migration insert should succeed");
        assert_eq!(graph.len(), 12);
        assert_eq!(graph.get(&Fingerprint(99)), Some(vec![Fingerprint(199)]));
    }

    #[test]
    fn test_explicit_migration_to_disk_preserves_order_duplicates_and_empty_entries() {
        let mut graph = SuccessorGraph::in_memory();
        graph
            .insert(
                Fingerprint(1),
                vec![Fingerprint(3), Fingerprint(2), Fingerprint(3)],
            )
            .unwrap();
        graph.insert(Fingerprint(2), Vec::new()).unwrap();

        assert_eq!(graph.migrate_to_disk_with_cache_slots(1).unwrap(), true);
        assert!(graph.is_disk());
        assert_eq!(graph.len(), 2);
        assert_eq!(graph.total_successors(), 3);
        assert_eq!(
            graph.get(&Fingerprint(1)),
            Some(vec![Fingerprint(3), Fingerprint(2), Fingerprint(3)])
        );
        assert_eq!(graph.get(&Fingerprint(2)), Some(Vec::new()));
        assert_eq!(graph.get(&Fingerprint(99)), None);

        assert_eq!(graph.migrate_to_disk_with_cache_slots(2).unwrap(), false);
        let SuccessorGraph::Disk(disk) = &graph else {
            panic!("graph must remain disk-backed");
        };
        assert_eq!(
            disk.cache_slots(),
            1,
            "an existing smaller cache must not grow"
        );
        assert_eq!(
            graph.get(&Fingerprint(1)),
            Some(vec![Fingerprint(3), Fingerprint(2), Fingerprint(3)])
        );
    }

    #[test]
    fn test_explicit_migration_shrinks_an_existing_disk_cache() {
        let mut graph = SuccessorGraph::disk().unwrap();
        graph.insert(Fingerprint(1), vec![Fingerprint(2)]).unwrap();
        let SuccessorGraph::Disk(disk) = &graph else {
            panic!("graph must start disk-backed");
        };
        assert!(disk.cache_slots() > 1);

        assert_eq!(graph.migrate_to_disk_with_cache_slots(1).unwrap(), false);
        let SuccessorGraph::Disk(disk) = &graph else {
            panic!("graph must remain disk-backed");
        };
        assert_eq!(disk.cache_slots(), 1);
        assert_eq!(graph.get(&Fingerprint(1)), Some(vec![Fingerprint(2)]));
    }

    #[test]
    fn test_no_migration_when_no_limit() {
        // No entry_limit means no migration.
        let mut graph = SuccessorGraph::InMemory(InMemorySuccessorGraph {
            map: FxHashMap::default(),
            entry_limit: None,
        });

        for i in 0..100 {
            graph
                .insert(Fingerprint(i), vec![Fingerprint(i + 1000)])
                .expect("insert without limit should succeed");
        }

        assert!(
            !graph.is_disk(),
            "graph without limit should stay in-memory"
        );
        assert_eq!(graph.len(), 100);
    }
}
