// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Liveness checker implementation
//!
//! This module implements the product graph exploration for liveness checking.
//! The product graph is the cross-product of:
//! - The state graph (TLA+ states connected by Next relation)
//! - The tableau automaton (derived from negation of liveness property)
//!
//! A liveness violation exists iff there's an accepting cycle in this product graph.
//!
//! # TLC Reference
//!
//! This follows TLC's implementation in:
//! - `tlc2/tool/liveness/LiveCheck.java` - Main liveness checker
//! - `tlc2/tool/liveness/LiveWorker.java` - SCC detection
//!
//! # Phases
//!
//! Phase B2 (this file): Behavior graph construction
//! Phase B3: SCC detection using Tarjan's algorithm
//!
//! # Module Structure
//!
//! - `types` — Data types, type aliases, and debug flags
//! - `enabled_cache` — Thread-local ENABLED evaluation cache
//! - `subscript_cache` — Subscript expression evaluation and value cache
//! - `explore` — BFS graph construction (initial states, successors, exploration loops)
//! - `plan` — Formula decomposition into DNF clauses and PEM plans
//! - `scc_checks` — SCC constraint satisfaction, witness cycle construction, path finding
//! - `checks` — Top-level liveness checking entrypoint and counterexample construction
//! - `eval` — Expression evaluation against concrete states and transitions

mod cache_stats;
mod check_mask;
mod checks;
mod ea_bitmask_query;
mod ea_precompute;
mod ea_precompute_cache;
mod ea_precompute_enabled;
mod ea_precompute_exact_raw;
mod ea_precompute_leaf_batch;
mod ea_precompute_profile;
mod ea_precompute_subscript;
pub(crate) mod enabled_cache;
mod eval;
mod explore;
pub(crate) mod leaf_result_cache;
mod plan;
mod scc_checks;
mod subscript_cache;
mod types;

pub(crate) use cache_stats::log_cache_stats;
pub(crate) use check_mask::{ActionCheckMatrix, CheckMask};

pub(crate) use enabled_cache::{
    census_enabled_cache_len, clear_enabled_cache, eval_enabled_cached, eval_enabled_cached_mut,
    get_enabled_cached, is_enabled_cached, release_enabled_cache_storage, set_enabled_cache,
};
pub(crate) use leaf_result_cache::{
    census_action_pred_cache_len, census_scan_pred_len, clear_leaf_result_cache,
    clear_scan_pred_results, enabled_action_pred_pair, enabled_true_streak, enum_exact_tag,
    eval_action_pred_cached, extend_enabled_action_pred_pairs, extend_enum_exact_tags,
    extend_full_population_tags, extend_whole_next_action_tags, extend_whole_next_enabled_tags,
    full_population_tag, get_scan_pred_result, insert_scan_pred_result, note_enabled_outcome,
    release_leaf_result_cache_storage, reset_enabled_streak, set_enabled_action_pred_pairs,
    subscript_watch_cached, whole_next_action_tag, whole_next_enabled_tag, WatchVarSet,
};
// Part of #3100 Phase A0: array-native inline subscript caching.
pub(crate) use subscript_cache::census_subscript_cache_len;
pub(crate) use subscript_cache::clear_subscript_value_cache;
pub(crate) use subscript_cache::eval_subscript_changed_array_cached;
pub(crate) use subscript_cache::release_subscript_cache_storage;
// Part of #liveness-leaf-memo: cached subscript probe for the inline ENABLED scan.
pub(crate) use subscript_cache::eval_subscript_changed_state_cached;
pub(crate) use subscript_cache::register_subscript_tag_classes;
pub(crate) use types::InlineCheckResults;

#[cfg(test)]
pub(crate) fn seed_regen_thread_local_storage_for_test(
    current_fp: crate::state::Fingerprint,
    next_fp: crate::state::Fingerprint,
) {
    enabled_cache::set_enabled_cache(current_fp, 7, true);
    subscript_cache::set_subscript_fp_cache(current_fp, 7, 11);
    leaf_result_cache::eval_action_pred_cached(true, current_fp, next_fp, 7, || Ok(true))
        .expect("test cache seed must evaluate");
    leaf_result_cache::insert_scan_pred_result(current_fp, next_fp, 7, true);
}
#[cfg(test)]
pub use types::LivenessConstraints;
#[cfg(test)]
pub(super) use types::SccEdgeList;
#[cfg(debug_assertions)]
pub(super) use types::{debug_action_pred, debug_bindings, debug_changed};
pub(super) use types::{debug_subscript, CounterexamplePath};
pub use types::{GroupedLivenessPlan, LivenessResult, LivenessStats, PemPlan};

use super::behavior_graph::{BehaviorGraph, BehaviorGraphNode};
use super::consistency::is_state_consistent;
pub(super) use super::live_expr;
use super::live_expr::LiveExpr;
use super::tableau::Tableau;
use super::SuccessorWitnessMap;
use crate::error::{EvalError, EvalResult};
use crate::eval::{Env, EvalCtx};
use crate::state::{compute_fingerprint_from_compact_array, ArrayState, Fingerprint, State};
use crate::var_index::VarRegistry;
use rustc_hash::FxHashMap;
use tla_eval::tir::TirProgram;

use std::sync::Arc;

/// Soft cap for instance-local consistency cache entries. Entries are
/// lightweight bools, so keep this aligned with the ENABLED cache.
const CONSISTENCY_CACHE_SOFT_CAP: usize = 200_000;
/// Soft cap for cached state environments. These entries are heavier than the
/// bool caches, so use the smaller subscript-cache-style bound.
const STATE_ENV_CACHE_SOFT_CAP: usize = 50_000;

/// Read-only compressed storage for a complete exact-raw adjacency relation.
///
/// Source fingerprints retain hash lookup semantics, while all ordered rows
/// share one edge allocation. Row contents are deliberately not sorted or
/// deduplicated: generation order, duplicate transitions, empty rows, and an
/// appended stuttering edge are observable by the existing ENABLED/action
/// evaluators and must survive the representation change exactly.
#[derive(Clone)]
struct ExactRawSuccessorCsr {
    row_by_source: FxHashMap<Fingerprint, u32>,
    offsets: Vec<u32>,
    edges: Vec<Fingerprint>,
}

impl ExactRawSuccessorCsr {
    fn row(&self, row: u32) -> &[Fingerprint] {
        let row = row as usize;
        let start = self.offsets[row] as usize;
        let end = self.offsets[row + 1] as usize;
        &self.edges[start..end]
    }
}

/// Borrowed adjacency row returned by [`StateSuccessorFingerprints::get`].
///
/// This preserves the small map-like API used throughout the checker without
/// exposing whether a row still has its own `Vec` allocation.
#[derive(Clone, Copy)]
struct SuccessorFingerprintRef<'a>(&'a [Fingerprint]);

impl<'a> SuccessorFingerprintRef<'a> {
    #[cfg(test)]
    fn as_slice(self) -> &'a [Fingerprint] {
        self.0
    }
}

impl std::ops::Deref for SuccessorFingerprintRef<'_> {
    type Target = [Fingerprint];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

/// O(1)-clone owned row used by mutable evaluation paths that must end their
/// borrow of `LivenessChecker` before evaluating the row.
enum OwnedSuccessorFingerprints {
    Sparse(Arc<Vec<Fingerprint>>),
    Frozen {
        csr: Arc<ExactRawSuccessorCsr>,
        start: u32,
        end: u32,
    },
}

impl std::ops::Deref for OwnedSuccessorFingerprints {
    type Target = [Fingerprint];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Sparse(row) => row.as_slice(),
            Self::Frozen { csr, start, end } => &csr.edges[*start as usize..*end as usize],
        }
    }
}

/// Exact-raw adjacency while it is being built, or after a structurally
/// complete relation has been packed into CSR storage.
///
/// Completeness is local to the liveness group that produced a CSR: a later
/// group can reach additional raw states because its tableau has different
/// consistency constraints. The first such row moves `Frozen` to `Extending`;
/// its sparse delta is appended at the next freeze instead of forcing an eager
/// full thaw. Keeping a distinct `Frozen` variant leaves the common packed
/// lookup path at one hash-table probe.
enum StateSuccessorFingerprints {
    Building(FxHashMap<Fingerprint, Arc<Vec<Fingerprint>>>),
    Frozen(Arc<ExactRawSuccessorCsr>),
    Extending {
        csr: Arc<ExactRawSuccessorCsr>,
        delta: FxHashMap<Fingerprint, Arc<Vec<Fingerprint>>>,
    },
}

impl Default for StateSuccessorFingerprints {
    fn default() -> Self {
        Self::Building(FxHashMap::default())
    }
}

impl StateSuccessorFingerprints {
    fn csr_counts_fit(row_count: usize, edge_count: usize) -> bool {
        row_count.checked_add(1).is_some()
            && u32::try_from(row_count).is_ok()
            && u32::try_from(edge_count).is_ok()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        match self {
            Self::Building(rows) => rows.len(),
            Self::Frozen(csr) => csr.row_by_source.len(),
            Self::Extending { csr, delta } => csr.row_by_source.len() + delta.len(),
        }
    }

    fn contains_key(&self, source: &Fingerprint) -> bool {
        match self {
            Self::Building(rows) => rows.contains_key(source),
            Self::Frozen(csr) => csr.row_by_source.contains_key(source),
            Self::Extending { csr, delta } => {
                csr.row_by_source.contains_key(source) || delta.contains_key(source)
            }
        }
    }

    fn get(&self, source: &Fingerprint) -> Option<SuccessorFingerprintRef<'_>> {
        let row = match self {
            Self::Building(rows) => rows.get(source)?.as_slice(),
            Self::Frozen(csr) => csr.row(*csr.row_by_source.get(source)?),
            Self::Extending { csr, delta } => {
                if let Some(&row) = csr.row_by_source.get(source) {
                    csr.row(row)
                } else {
                    delta.get(source)?.as_slice()
                }
            }
        };
        Some(SuccessorFingerprintRef(row))
    }

    fn get_owned(&self, source: &Fingerprint) -> Option<OwnedSuccessorFingerprints> {
        match self {
            Self::Building(rows) => rows
                .get(source)
                .map(|row| OwnedSuccessorFingerprints::Sparse(Arc::clone(row))),
            Self::Frozen(csr) => {
                let row = *csr.row_by_source.get(source)? as usize;
                Some(OwnedSuccessorFingerprints::Frozen {
                    csr: Arc::clone(csr),
                    start: csr.offsets[row],
                    end: csr.offsets[row + 1],
                })
            }
            Self::Extending { csr, delta } => {
                if let Some(&row) = csr.row_by_source.get(source) {
                    let row = row as usize;
                    return Some(OwnedSuccessorFingerprints::Frozen {
                        csr: Arc::clone(csr),
                        start: csr.offsets[row],
                        end: csr.offsets[row + 1],
                    });
                }
                delta
                    .get(source)
                    .map(|row| OwnedSuccessorFingerprints::Sparse(Arc::clone(row)))
            }
        }
    }

    fn insert(&mut self, source: Fingerprint, successors: Arc<Vec<Fingerprint>>) {
        match self {
            Self::Building(rows) => {
                rows.insert(source, successors);
            }
            Self::Frozen(csr) => {
                assert!(
                    !csr.row_by_source.contains_key(&source),
                    "packed exact-raw successor rows cannot be replaced"
                );
                let mut delta = FxHashMap::default();
                delta.insert(source, successors);
                *self = Self::Extending {
                    csr: Arc::clone(csr),
                    delta,
                };
            }
            Self::Extending { csr, delta } => {
                assert!(
                    !csr.row_by_source.contains_key(&source),
                    "packed exact-raw successor rows cannot be replaced"
                );
                delta.insert(source, successors);
            }
        }
    }

    #[cfg(test)]
    fn remove(&mut self, source: &Fingerprint) -> Option<Arc<Vec<Fingerprint>>> {
        match self {
            Self::Building(rows) => rows.remove(source),
            Self::Extending { csr, delta } if !csr.row_by_source.contains_key(source) => {
                delta.remove(source)
            }
            Self::Frozen(_) | Self::Extending { .. } => {
                panic!("complete exact-raw successor CSR cannot be mutated after freeze")
            }
        }
    }

    fn for_each_row(&self, mut visit: impl FnMut(Fingerprint, &[Fingerprint])) {
        match self {
            Self::Building(rows) => {
                for (&source, successors) in rows {
                    visit(source, successors.as_slice());
                }
            }
            Self::Frozen(csr) => {
                for (&source, &row) in &csr.row_by_source {
                    visit(source, csr.row(row));
                }
            }
            Self::Extending { csr, delta } => {
                for (&source, &row) in &csr.row_by_source {
                    visit(source, csr.row(row));
                }
                for (&source, successors) in delta {
                    visit(source, successors.as_slice());
                }
            }
        }
    }

    fn successor_count(&self) -> usize {
        let mut total = 0usize;
        self.for_each_row(|_, successors| {
            total = total.saturating_add(successors.len());
        });
        total
    }

    /// Convert a complete building map to CSR, or merge a packed relation's
    /// sparse cross-group delta into it. Oversized inputs stay in their current
    /// mutable representation so offsets can never truncate.
    fn freeze(&mut self) -> bool {
        match self {
            Self::Frozen(_) => true,
            Self::Building(rows) => {
                let row_count = rows.len();
                let Some(offset_count) = row_count.checked_add(1) else {
                    return false;
                };
                let Some(edge_count) = rows
                    .values()
                    .try_fold(0usize, |total, row| total.checked_add(row.len()))
                else {
                    return false;
                };
                if !Self::csr_counts_fit(row_count, edge_count) {
                    return false;
                }

                let Self::Building(rows) = std::mem::take(self) else {
                    unreachable!();
                };
                // Consume per-row allocations before allocating the replacement
                // source table. Keeping only compact source/edge vectors during
                // the copy avoids overlapping both full hash tables at the
                // freeze high-water mark.
                let mut sources = Vec::with_capacity(row_count);
                let mut offsets = Vec::with_capacity(offset_count);
                let mut edges = Vec::with_capacity(edge_count);
                offsets.push(0);
                for (source, successors) in rows {
                    sources.push(source);
                    edges.extend_from_slice(successors.as_slice());
                    offsets.push(
                        u32::try_from(edges.len())
                            .expect("exact-raw CSR edge count was checked before construction"),
                    );
                }

                let mut row_by_source = FxHashMap::default();
                row_by_source.reserve(row_count);
                for (row, source) in sources.into_iter().enumerate() {
                    row_by_source.insert(
                        source,
                        u32::try_from(row)
                            .expect("exact-raw CSR row count was checked before construction"),
                    );
                }
                debug_assert_eq!(row_by_source.len(), row_count);
                debug_assert_eq!(offsets.len(), offset_count);
                debug_assert_eq!(edges.len(), edge_count);
                *self = Self::Frozen(Arc::new(ExactRawSuccessorCsr {
                    row_by_source,
                    offsets,
                    edges,
                }));
                true
            }
            Self::Extending { csr, delta } => {
                let Some(row_count) = csr.row_by_source.len().checked_add(delta.len()) else {
                    return false;
                };
                let Some(delta_edge_count) = delta
                    .values()
                    .try_fold(0usize, |total, row| total.checked_add(row.len()))
                else {
                    return false;
                };
                let Some(edge_count) = csr.edges.len().checked_add(delta_edge_count) else {
                    return false;
                };
                if !Self::csr_counts_fit(row_count, edge_count) {
                    return false;
                }

                let Self::Extending { mut csr, delta } = std::mem::take(self) else {
                    unreachable!();
                };
                let packed = Arc::make_mut(&mut csr);
                packed.row_by_source.reserve(delta.len());
                packed.offsets.reserve(delta.len());
                packed.edges.reserve(delta_edge_count);
                for (source, successors) in delta {
                    debug_assert!(!packed.row_by_source.contains_key(&source));
                    let row = u32::try_from(packed.row_by_source.len())
                        .expect("exact-raw CSR row count was checked before extension");
                    packed.edges.extend_from_slice(successors.as_slice());
                    packed.offsets.push(
                        u32::try_from(packed.edges.len())
                            .expect("exact-raw CSR edge count was checked before extension"),
                    );
                    packed.row_by_source.insert(source, row);
                }
                debug_assert_eq!(packed.row_by_source.len(), row_count);
                debug_assert_eq!(packed.offsets.len(), row_count + 1);
                debug_assert_eq!(packed.edges.len(), edge_count);
                *self = Self::Frozen(csr);
                true
            }
        }
    }

    #[cfg(test)]
    fn is_frozen(&self) -> bool {
        matches!(self, Self::Frozen(_))
    }
}

/// Exact raw state-graph data reusable across on-the-fly liveness groups.
///
/// Both maps move together: every cached ordered successor list is resolved
/// through the paired compact payload map. This cache deliberately excludes
/// tableau/product-graph state, consistency results, and fairness masks, all of
/// which remain property- and group-specific.
pub(crate) struct ExactRawStateGraphCache {
    state_payloads: FxHashMap<Fingerprint, ArrayState>,
    successor_fps: StateSuccessorFingerprints,
}

impl ExactRawStateGraphCache {
    pub(crate) fn estimated_bytes_from_counts(
        state_payload_count: usize,
        successor_entry_count: usize,
        successor_value_count: usize,
        var_count: usize,
    ) -> usize {
        const STATE_MAP_ENTRY_OVERHEAD: usize = 24;
        const SUCCESSOR_MAP_ENTRY_BYTES: usize = 48;

        let array_state_bytes = var_count.saturating_mul(64).saturating_add(48);
        let state_bytes = state_payload_count
            .saturating_mul(array_state_bytes.saturating_add(STATE_MAP_ENTRY_OVERHEAD));
        let successor_entry_bytes = successor_entry_count.saturating_mul(SUCCESSOR_MAP_ENTRY_BYTES);

        state_bytes
            .saturating_add(successor_entry_bytes)
            .saturating_add(
                successor_value_count.saturating_mul(std::mem::size_of::<Fingerprint>()),
            )
    }

    /// Load-independent structural estimate used to bound cross-group reuse.
    /// Constants mirror the existing liveness regeneration estimate: compact
    /// values are charged at 64 bytes per variable, with hash/Arc/Vec overhead
    /// folded into fixed per-entry charges.
    pub(crate) fn estimated_bytes(&self, var_count: usize) -> usize {
        let successor_value_count = self.successor_fps.successor_count();
        Self::estimated_bytes_from_counts(
            self.state_payloads.len(),
            self.successor_fps.len(),
            successor_value_count,
            var_count,
        )
    }
}

fn retain_half_if_needed<K, V>(map: &mut FxHashMap<K, V>, soft_cap: usize) {
    if map.len() <= soft_cap {
        return;
    }

    let target = soft_cap / 2;
    let mut kept = 0usize;
    map.retain(|_, _| {
        if kept < target {
            kept += 1;
            true
        } else {
            false
        }
    });
}

/// Liveness checker that builds and analyzes the behavior graph
///
/// The behavior graph is the product of state graph × tableau automaton.
/// Liveness checking proceeds in phases:
/// 1. Build behavior graph during state exploration
/// 2. Find strongly connected components (SCCs)
/// 3. Check each SCC for accepting cycles
pub struct LivenessChecker {
    /// The tableau automaton for the liveness property
    tableau: Tableau,
    /// The behavior graph (product of state graph × tableau)
    graph: BehaviorGraph,
    /// Base evaluation context for checking consistency
    ctx: EvalCtx,
    /// Promises (<>r) extracted from the tableau temporal formula
    promises: Vec<LiveExpr>,
    /// AE/EA constraints that must be satisfied by a counterexample cycle (test-only reader)
    #[cfg(test)]
    constraints: LivenessConstraints,
    /// Cached state graph successors (needed for ENABLED), keyed by the
    /// behavior-graph state fingerprint.
    /// Arc-wrapped to allow O(1) clones during BFS instead of cloning the full Vec<State>.
    state_successors: FxHashMap<crate::state::Fingerprint, Arc<Vec<State>>>,
    /// Cached successor fingerprints for the direct fingerprint-only graph path,
    /// keyed by the behavior-graph state fingerprint.
    ///
    /// This preserves the zero-clone behavior of `explore_state_graph_direct_fp`.
    /// ENABLED evaluation resolves states through `BehaviorGraph::shared_state_cache`
    /// on demand instead of materializing a second `Vec<State>` per source state.
    state_successor_fps: StateSuccessorFingerprints,
    /// Cached consistency check results: (behavior-graph state_fp, tableau_idx) -> is_consistent
    consistency_cache: FxHashMap<(crate::state::Fingerprint, usize), bool>,
    /// Optional mapping from representative state fingerprint -> canonical fingerprint (symmetry).
    state_fp_to_canon_fp: Option<Arc<FxHashMap<Fingerprint, Fingerprint>>>,
    /// Concrete successor witnesses keyed by canonical source fingerprint.
    ///
    /// Each entry contains `(canonical_dest_fp, successor_state)` pairs for each
    /// concrete successor generated from the representative source state.
    ///
    /// When present, this is used to evaluate ENABLED and action predicates under symmetry.
    succ_witnesses: Option<Arc<SuccessorWitnessMap>>,
    /// Statistics
    stats: LivenessStats,
    /// Cached Env representations of states, keyed by fingerprint.
    /// Avoids repeated FxHashMap construction during SCC constraint checking.
    state_env_cache: FxHashMap<Fingerprint, Arc<Env>>,
    /// Whether evaluator provenance permits exact-raw fairness-mask
    /// reconstruction. Explicit TIR eval/parity/stats keep their evaluator
    /// observable and disable this optimization.
    exact_raw_fp_leaf_fast_path_allowed: bool,
}

impl LivenessChecker {
    fn behavior_graph_invariant_error(message: String) -> EvalError {
        EvalError::Internal {
            message: format!("behavior graph invariant violated: {message}"),
            span: None,
        }
    }

    // ---- Construction ----

    /// Create a new liveness checker
    ///
    /// # Arguments
    ///
    /// * `tableau` - The tableau automaton for the liveness property
    /// * `ctx` - Base evaluation context (with operators loaded)
    // Test-only constructor; production entry points use `new_from_env` so the
    // runtime graph-storage gate can switch between in-memory and disk-backed
    // behavior graphs.
    #[cfg(test)]
    pub fn new(tableau: Tableau, ctx: EvalCtx) -> Self {
        // Collect ALL promises from the formula. Promises are <> subformulas that
        // must be fulfilled somewhere in any accepting SCC. The tableau expansion
        // and is_fulfilling check handle the fulfillment semantics correctly for
        // promises with temporal bodies (e.g., <>(P /\ []Q)).
        //
        // Previously, promises with temporal-level bodies were filtered out, but this
        // caused false positives: for <>(terminated /\ []~terminationDetected), the
        // promise was not tracked, so any SCC was considered violating.
        let promises = tableau.formula().extract_promises();

        let _ = cache_stats::take_cache_stats();

        Self {
            tableau,
            graph: BehaviorGraph::new(),
            ctx,
            promises,
            #[cfg(test)]
            constraints: LivenessConstraints::default(),
            state_successors: FxHashMap::default(),
            state_successor_fps: StateSuccessorFingerprints::default(),
            consistency_cache: FxHashMap::default(),
            state_fp_to_canon_fp: None,
            succ_witnesses: None,
            stats: LivenessStats::default(),
            state_env_cache: FxHashMap::default(),
            exact_raw_fp_leaf_fast_path_allowed: true,
        }
    }

    /// Create a new liveness checker honoring runtime graph-storage gates.
    pub fn new_from_env(tableau: Tableau, ctx: EvalCtx) -> EvalResult<Self> {
        let promises = tableau.formula().extract_promises();
        let _ = cache_stats::take_cache_stats();

        Ok(Self {
            tableau,
            graph: BehaviorGraph::new_from_env()?,
            ctx,
            promises,
            #[cfg(test)]
            constraints: LivenessConstraints::default(),
            state_successors: FxHashMap::default(),
            state_successor_fps: StateSuccessorFingerprints::default(),
            consistency_cache: FxHashMap::default(),
            state_fp_to_canon_fp: None,
            succ_witnesses: None,
            stats: LivenessStats::default(),
            state_env_cache: FxHashMap::default(),
            exact_raw_fp_leaf_fast_path_allowed: true,
        })
    }

    /// Create a new liveness checker with auto-disk detection based on an
    /// estimated behavior-graph node count.
    ///
    /// When `estimated_nodes` exceeds the auto-disk threshold (default 2M),
    /// the behavior graph is automatically disk-backed to prevent OOM on
    /// multi-property liveness specs.
    pub fn new_from_env_with_hint(
        tableau: Tableau,
        ctx: EvalCtx,
        estimated_nodes: Option<usize>,
    ) -> EvalResult<Self> {
        let promises = tableau.formula().extract_promises();
        let _ = cache_stats::take_cache_stats();

        Ok(Self {
            tableau,
            graph: BehaviorGraph::new_from_env_with_hint(estimated_nodes)?,
            ctx,
            promises,
            #[cfg(test)]
            constraints: LivenessConstraints::default(),
            state_successors: FxHashMap::default(),
            state_successor_fps: StateSuccessorFingerprints::default(),
            consistency_cache: FxHashMap::default(),
            state_fp_to_canon_fp: None,
            succ_witnesses: None,
            stats: LivenessStats::default(),
            state_env_cache: FxHashMap::default(),
            exact_raw_fp_leaf_fast_path_allowed: true,
        })
    }

    /// Create a new liveness checker with additional AE/EA constraints.
    #[cfg(test)]
    pub fn new_with_constraints(
        tableau: Tableau,
        ctx: EvalCtx,
        constraints: LivenessConstraints,
    ) -> Self {
        let promises = tableau.formula().extract_promises();
        let _ = cache_stats::take_cache_stats();

        Self {
            tableau,
            graph: BehaviorGraph::new(),
            ctx,
            promises,
            constraints,
            state_successors: FxHashMap::default(),
            state_successor_fps: StateSuccessorFingerprints::default(),
            consistency_cache: FxHashMap::default(),
            state_fp_to_canon_fp: None,
            succ_witnesses: None,
            stats: LivenessStats::default(),
            state_env_cache: FxHashMap::default(),
            exact_raw_fp_leaf_fast_path_allowed: true,
        }
    }

    pub(crate) fn set_exact_raw_fp_leaf_fast_path_allowed(&mut self, allowed: bool) {
        self.exact_raw_fp_leaf_fast_path_allowed = allowed;
    }

    /// Provide precomputed successor information for symmetry-aware liveness evaluation.
    ///
    /// When symmetry reduction is enabled, liveness checking needs access to the concrete
    /// successor states (not just canonical fingerprints) to correctly evaluate `ENABLED`
    /// and action-level predicates. TLC evaluates action checks on the concrete successor
    /// states *before* applying symmetry.
    pub fn set_successor_maps(
        &mut self,
        state_fp_to_canon_fp: Arc<FxHashMap<Fingerprint, Fingerprint>>,
        succ_witnesses: Option<Arc<SuccessorWitnessMap>>,
    ) {
        self.state_fp_to_canon_fp = Some(state_fp_to_canon_fp);
        self.succ_witnesses = succ_witnesses;
    }

    /// Part of #3065: Set a shared state cache on the behavior graph.
    /// Allows fingerprint-based exploration without cloning State objects.
    pub fn set_behavior_graph_shared_cache(
        &mut self,
        cache: Arc<FxHashMap<Fingerprint, crate::state::ArrayState>>,
    ) {
        let registry = Arc::new(self.ctx.var_registry().clone());
        self.graph.set_shared_state_cache(cache, registry);
    }

    /// Enable a compact payload cache owned by this checker.
    ///
    /// On-the-fly exploration uses this mode when fingerprints are the states'
    /// exact raw fingerprints (no VIEW or symmetry). Concrete states are
    /// converted once as they are generated; subsequent graph and ENABLED
    /// lookups resolve them through compact `ArrayState` payloads.
    pub(crate) fn enable_owned_behavior_graph_state_cache(&mut self) {
        let registry = Arc::new(self.ctx.var_registry().clone());
        self.graph.enable_owned_state_cache(registry);
    }

    /// Adopt exact raw state-graph data produced by an earlier on-the-fly
    /// liveness group in this model-checking run.
    pub(crate) fn install_exact_raw_state_graph_cache(&mut self, cache: ExactRawStateGraphCache) {
        assert!(
            self.state_successors.is_empty() && self.state_successor_fps.is_empty(),
            "exact raw cache must be installed before successor exploration"
        );
        assert!(
            self.state_fp_to_canon_fp.is_none() && self.succ_witnesses.is_none(),
            "exact raw cache cannot be combined with canonical successor maps"
        );
        let registry = Arc::new(self.ctx.var_registry().clone());
        self.graph
            .install_owned_state_cache(cache.state_payloads, registry);
        self.state_successor_fps = cache.successor_fps;
    }

    /// Move exact raw state-graph data out after all checks and trace
    /// reconstruction have completed, ready for the next group/property.
    pub(crate) fn take_exact_raw_state_graph_cache(&mut self) -> Option<ExactRawStateGraphCache> {
        let state_payloads = self.graph.take_owned_state_cache()?;
        debug_assert!(self.state_successors.is_empty());
        Some(ExactRawStateGraphCache {
            state_payloads,
            successor_fps: std::mem::take(&mut self.state_successor_fps),
        })
    }

    /// Estimate an owned exact raw cache without moving it out of the checker.
    ///
    /// This permits the separate retained BFS graph to be released after a
    /// complete direct traversal or retained-adjacency translation, but before
    /// mask allocation and Tarjan. The checker-owned payloads remain in place
    /// for ENABLED evaluation and trace reconstruction.
    pub(crate) fn exact_raw_state_graph_cache_estimated_bytes(
        &self,
        var_count: usize,
    ) -> Option<usize> {
        let state_payload_count = self.graph.owned_state_cache_len()?;
        let mut successor_value_count = 0usize;
        let mut structurally_valid = true;
        self.state_successor_fps
            .for_each_row(|source_fp, successor_fps| {
                // A transferred exact cache is usable only while every adjacency
                // endpoint still has its paired compact payload. Validate that
                // invariant during the estimate pass used by early release.
                if self.graph.owned_state_cache_get(source_fp).is_none() {
                    structurally_valid = false;
                    return;
                }
                for &successor_fp in successor_fps.iter() {
                    if self.graph.owned_state_cache_get(successor_fp).is_none() {
                        structurally_valid = false;
                        return;
                    }
                }
                successor_value_count = successor_value_count.saturating_add(successor_fps.len());
            });
        if !structurally_valid {
            return None;
        }
        Some(ExactRawStateGraphCache::estimated_bytes_from_counts(
            state_payload_count,
            self.state_successor_fps.len(),
            successor_value_count,
            var_count,
        ))
    }

    /// Freeze a complete exact-raw adjacency relation into CSR storage.
    ///
    /// Completeness is proven structurally before the representation changes:
    /// every compact payload must be a source row, and every source and edge
    /// endpoint must have a payload. A partial cache stays mutable so later
    /// groups can fill missing rows by evaluating Next as before.
    pub(crate) fn freeze_complete_exact_raw_adjacency(&mut self) -> bool {
        let Some(state_payload_count) = self.graph.owned_state_cache_len() else {
            return false;
        };
        if self.state_successor_fps.len() != state_payload_count {
            return false;
        }

        let mut complete = true;
        self.state_successor_fps
            .for_each_row(|source_fp, successor_fps| {
                if self.graph.owned_state_cache_get(source_fp).is_none()
                    || successor_fps.iter().any(|&successor_fp| {
                        self.graph.owned_state_cache_get(successor_fp).is_none()
                    })
                {
                    complete = false;
                }
            });
        complete && self.state_successor_fps.freeze()
    }

    /// Whether an exact-raw source entry is present and its compact payload
    /// matches the supplied authoritative state.
    ///
    /// The single cache-estimate pass immediately before retained-graph
    /// release validates every successor payload globally.
    pub(crate) fn exact_raw_source_is_present_for(
        &self,
        raw_fp: Fingerprint,
        source: &ArrayState,
    ) -> bool {
        if !self
            .graph
            .owned_state_cache_get(raw_fp)
            .is_some_and(|cached| cached.values() == source.values())
        {
            return false;
        }
        self.state_successor_fps.contains_key(&raw_fp)
    }

    #[cfg(test)]
    pub(crate) fn remove_owned_exact_state_for_test(&mut self, fp: Fingerprint) {
        self.graph.remove_owned_state_for_test(fp);
    }

    /// Translate one retained BFS source into the checker-owned exact-raw
    /// cache. Ordered edges and duplicates are preserved, and the implicit
    /// stuttering edge is appended exactly as normal on-the-fly generation
    /// would append it.
    pub(crate) fn seed_exact_raw_source_from_arrays<'a, I>(
        &mut self,
        source: &ArrayState,
        successors: I,
        registry: &VarRegistry,
        add_stuttering: bool,
    ) -> bool
    where
        I: IntoIterator<Item = &'a ArrayState>,
    {
        if !self.graph.has_owned_state_cache() {
            return false;
        }

        let source_fp = compute_fingerprint_from_compact_array(source.values(), registry);
        self.graph.cache_owned_array_state(source_fp, source);
        if !self
            .graph
            .owned_state_cache_get(source_fp)
            .is_some_and(|cached| cached.values() == source.values())
        {
            return false;
        }

        let mut successor_fps = Vec::new();
        for successor in successors {
            let successor_fp = compute_fingerprint_from_compact_array(successor.values(), registry);
            self.graph.cache_owned_array_state(successor_fp, successor);
            if !self
                .graph
                .owned_state_cache_get(successor_fp)
                .is_some_and(|cached| cached.values() == successor.values())
            {
                return false;
            }
            successor_fps.push(successor_fp);
        }
        if add_stuttering {
            successor_fps.push(source_fp);
        }
        self.state_successor_fps
            .insert(source_fp, Arc::new(successor_fps));
        true
    }

    /// Part of #3065: Populate successor fingerprints from the behavior graph
    /// after fingerprint-based exploration.
    ///
    /// `explore_state_graph_direct_fp` works with fingerprints only and does
    /// not populate `state_successors`. But `populate_node_check_masks` needs
    /// successor information for ENABLED evaluation. Without this call, ENABLED
    /// always returns FALSE, causing spurious liveness violations for specs
    /// with WF/SF fairness constraints.
    pub fn populate_state_successor_fps_from_graph(&mut self) -> EvalResult<()> {
        for node in self.graph.node_keys() {
            let fp = node.state_fp;
            if self.state_successors.contains_key(&fp) || self.state_successor_fps.contains_key(&fp)
            {
                continue;
            }
            let succ_fps = self
                .graph
                .try_with_successors(&node, |successors| {
                    successors.iter().map(|succ| succ.state_fp).collect()
                })?
                .ok_or_else(|| {
                    Self::behavior_graph_invariant_error(format!(
                        "successor-adjacency source node {node} from node_keys is missing"
                    ))
                })?;
            let succ_fps: Vec<Fingerprint> = succ_fps;
            self.state_successor_fps.insert(fp, Arc::new(succ_fps));
        }
        Ok(())
    }

    // ---- Accessors ----

    /// Get the behavior graph
    #[cfg(test)]
    pub fn graph(&self) -> &BehaviorGraph {
        &self.graph
    }

    /// Get statistics
    pub fn stats(&self) -> &LivenessStats {
        &self.stats
    }

    /// Collect thread-local cache stats into `self.stats`.
    ///
    /// Call this once before reading stats, after liveness checking is complete.
    /// Thread-local counters are consumed (reset to 0) so this is not idempotent.
    /// Part of #4083: cache profiling.
    pub fn collect_cache_stats(&mut self) {
        let (subscript, enabled) = cache_stats::take_cache_stats();
        self.stats.subscript_cache_hits += subscript.hits;
        self.stats.subscript_cache_misses += subscript.misses;
        self.stats.subscript_cache_evictions += subscript.evictions;
        self.stats.enabled_cache_hits += enabled.hits;
        self.stats.enabled_cache_misses += enabled.misses;
        self.stats.enabled_cache_evictions += enabled.evictions;
    }

    /// Get AE/EA constraints.
    #[cfg(test)]
    pub fn constraints(&self) -> &LivenessConstraints {
        &self.constraints
    }

    // ---- State / consistency caching ----

    /// Get or create a cached Env for a state.
    ///
    /// This avoids repeated hash map construction during SCC constraint checking,
    /// which can be called thousands of times on the same states.
    fn get_cached_env(&mut self, state: &State) -> Arc<Env> {
        let fp = state.fingerprint();
        if let Some(env) = self.state_env_cache.get(&fp) {
            self.stats.state_env_cache_hits += 1;
            return Arc::clone(env);
        }
        self.stats.state_env_cache_misses += 1;

        // Build Env from state vars
        let mut env = Env::new();
        for (name, value) in state.vars() {
            env.insert(Arc::clone(name), value.clone());
        }
        let env = Arc::new(env);
        retain_half_if_needed(&mut self.state_env_cache, STATE_ENV_CACHE_SOFT_CAP);
        self.state_env_cache.insert(fp, Arc::clone(&env));
        env
    }

    fn get_cached_env_by_fp(&mut self, fp: Fingerprint) -> EvalResult<Arc<Env>> {
        if let Some(env) = self.state_env_cache.get(&fp) {
            self.stats.state_env_cache_hits += 1;
            return Ok(Arc::clone(env));
        }
        self.stats.state_env_cache_misses += 1;

        let env = {
            let state = self.graph.get_state_by_fp(fp).ok_or_else(|| {
                Self::behavior_graph_invariant_error(format!(
                    "state_env_cache: missing state for fingerprint {fp}"
                ))
            })?;
            let mut env = self.ctx.env().clone();
            for (name, value) in state.vars() {
                env.insert(Arc::clone(name), value.clone());
            }
            Arc::new(env)
        };

        retain_half_if_needed(&mut self.state_env_cache, STATE_ENV_CACHE_SOFT_CAP);
        self.state_env_cache.insert(fp, Arc::clone(&env));
        Ok(env)
    }

    fn successor_states_for_enabled(&self, fp: Fingerprint) -> EvalResult<Vec<State>> {
        // Owned compact exploration records the complete ordered state-graph
        // adjacency before tableau pruning. It is authoritative over any stale
        // witness/full-state maps left by another path.
        if self.graph.has_owned_state_cache() {
            let Some(succ_fps) = self.state_successor_fps.get(&fp) else {
                return Err(Self::behavior_graph_invariant_error(format!(
                    "owned compact cache is missing successor adjacency for source {fp}"
                )));
            };
            let mut successors = Vec::with_capacity(succ_fps.len());
            for succ_fp in succ_fps.iter().copied() {
                let successor = self.graph.get_state_by_fp(succ_fp).ok_or_else(|| {
                    Self::behavior_graph_invariant_error(format!(
                        "owned compact cache is missing successor payload {succ_fp} for source {fp}"
                    ))
                })?;
                successors.push(successor);
            }
            return Ok(successors);
        }
        if let Some(witnesses) = self.succ_witnesses.as_ref().and_then(|map| map.get(&fp)) {
            // Part of #2661: Convert ArrayState→State lazily on the SCC ENABLED path.
            let registry = self.ctx.var_registry();
            return Ok(witnesses
                .iter()
                .map(|(_, arr)| arr.to_state(registry))
                .collect());
        }
        if let Some(succs) = self.state_successors.get(&fp) {
            return Ok(succs.as_ref().clone());
        }
        let Some(succ_fps) = self.state_successor_fps.get(&fp) else {
            return Ok(Vec::new());
        };
        let mut successors = Vec::with_capacity(succ_fps.len());
        for succ_fp in succ_fps.iter().copied() {
            if let Some(successor) = self.graph.get_state_by_fp(succ_fp) {
                successors.push(successor);
            }
        }
        Ok(successors)
    }

    /// Check consistency with caching. Returns cached result if available.
    fn check_consistency_cached_with_fp<F>(
        &mut self,
        state: &State,
        state_fp: Fingerprint,
        tableau_idx: usize,
        get_successors: &mut F,
        tir: Option<&TirProgram<'_>>,
    ) -> EvalResult<bool>
    where
        F: FnMut(&State) -> EvalResult<Vec<State>>,
    {
        let cache_key = (state_fp, tableau_idx);

        // Check cache first
        if let Some(&cached) = self.consistency_cache.get(&cache_key) {
            self.stats.consistency_cache_hits += 1;
            return Ok(cached);
        }
        self.stats.consistency_cache_misses += 1;

        // Compute and cache the result
        self.stats.consistency_checks += 1;
        let tableau_node = match self.tableau.node(tableau_idx) {
            Some(n) => n,
            None => {
                return Err(EvalError::Internal {
                    message: format!(
                        "missing tableau node in check_consistency_cached: \
                         tableau_idx={tableau_idx} for state_fp={state_fp}"
                    ),
                    span: None,
                });
            }
        };

        let consistent = is_state_consistent(&self.ctx, state, tableau_node, get_successors, tir)?;
        retain_half_if_needed(&mut self.consistency_cache, CONSISTENCY_CACHE_SOFT_CAP);
        self.consistency_cache.insert(cache_key, consistent);
        Ok(consistent)
    }

    #[allow(dead_code)]
    fn check_consistency_cached<F>(
        &mut self,
        state: &State,
        tableau_idx: usize,
        get_successors: &mut F,
        tir: Option<&TirProgram<'_>>,
    ) -> EvalResult<bool>
    where
        F: FnMut(&State) -> EvalResult<Vec<State>>,
    {
        self.check_consistency_cached_with_fp(
            state,
            state.fingerprint(),
            tableau_idx,
            get_successors,
            tir,
        )
    }

    /// Find strongly connected components in the behavior graph
    ///
    /// Returns all SCCs using Tarjan's algorithm.
    pub fn find_sccs(&self) -> super::tarjan::TarjanResult {
        super::tarjan::find_sccs(&self.graph)
    }

    /// Find non-trivial cycles in the behavior graph.
    ///
    /// Returns a `TarjanResult` with SCCs filtered to actual cycles
    /// (not single nodes without self-loops). Callers should check
    /// `result.errors` for algorithm invariant violations. Part of #1817.
    #[cfg(test)]
    pub(crate) fn find_cycles(&self) -> super::tarjan::TarjanResult {
        super::tarjan::find_cycles(&self.graph)
    }
}

#[cfg(test)]
mod tests;
