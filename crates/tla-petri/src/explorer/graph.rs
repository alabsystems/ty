// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::VecDeque;

use crate::marking::{pack_marking_config, unpack_marking_config, MarkingConfig};
use crate::petri_net::PetriNet;
use crate::stubborn::PorStrategy;

use super::config::ExplorationConfig;
use super::setup::ExplorationSetup;
use super::state_registry::{GraphStateRegistry, StateAdmission};
use super::successors::{InterpretedSuccessorProvider, PetriSuccessorProvider, SuccessorVisit};

/// Reachability graph with compact integer state IDs.
///
/// Edges store `(successor_id, transition_index)` where the transition index
/// is a raw `u32` for compact storage (use `TransitionIdx(val)` to wrap).
pub(crate) struct ReachabilityGraph {
    /// Forward adjacency: `adj[state_id]` = list of `(successor_id, transition_fired)`.
    pub(crate) adj: Vec<Vec<(u32, u32)>>,
    /// Total number of reachable states discovered.
    pub(crate) num_states: u32,
    /// Whether BFS explored the full reachable state space.
    pub(crate) completed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphBuildMode {
    StructureOnly,
    WithMarkings,
}

impl GraphBuildMode {
    fn captures_markings(self) -> bool {
        matches!(self, Self::WithMarkings)
    }
}

struct GraphBuildArtifacts {
    graph: ReachabilityGraph,
    markings: Option<MarkingStore>,
}

/// Compact per-state marking storage for model checking.
///
/// Holds each state's marking in its PACKED byte encoding (at the net's token
/// width — typically 1 byte/place vs 8 for a `Vec<u64>`) plus the `MarkingConfig`
/// needed to decode it. The packed form is already computed during exploration (it
/// is the dedup key), so storing it and unpacking on read — rather than retaining
/// the fat `Vec<u64>` — is the dominant CTL memory win (~8x on token-conserving
/// nets). Decoding is LOSSLESS (the packed bytes are the canonical marking, not a
/// hash), so it changes no verdict.
pub(crate) struct MarkingStore {
    packed: Vec<Box<[u8]>>,
    config: MarkingConfig,
}

impl MarkingStore {
    fn new(config: MarkingConfig) -> Self {
        Self {
            packed: Vec::new(),
            config,
        }
    }

    fn push_packed(&mut self, packed: Box<[u8]>) {
        self.packed.push(packed);
    }

    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.packed.len()
    }

    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.packed.is_empty()
    }

    #[must_use]
    pub(crate) fn config(&self) -> &MarkingConfig {
        &self.config
    }

    /// The raw packed rows, indexed by state id — the `State` slice the CTL engine
    /// borrows (it decodes each row lazily via the atom evaluator).
    #[must_use]
    pub(crate) fn packed(&self) -> &[Box<[u8]>] {
        &self.packed
    }

    /// Decode state `i`'s marking into `out` (reused to avoid per-call allocation).
    pub(crate) fn unpack_into(&self, i: usize, out: &mut Vec<u64>) {
        unpack_marking_config(&self.packed[i], &self.config, out);
    }

    /// Decode state `i`'s marking into a fresh `Vec<u64>`.
    #[must_use]
    pub(crate) fn unpack(&self, i: usize) -> Vec<u64> {
        let mut out = Vec::new();
        self.unpack_into(i, &mut out);
        out
    }

    /// Decode ALL markings into a `Vec<Vec<u64>>`. Test/diagnostic convenience only —
    /// it re-materializes the fat form the store exists to avoid.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn unpack_all(&self) -> Vec<Vec<u64>> {
        (0..self.len()).map(|i| self.unpack(i)).collect()
    }

    /// Build a store from unpacked `Vec<u64>` markings (test-only). Packs at full u64
    /// width so any marking round-trips exactly.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_unpacked(markings: &[Vec<u64>]) -> Self {
        let num_places = markings.first().map_or(0, Vec::len);
        let config = MarkingConfig::standard(num_places, crate::marking::TokenWidth::U64);
        let mut buf = Vec::new();
        let packed = markings
            .iter()
            .map(|m| {
                buf.clear();
                pack_marking_config(m, &config, &mut buf);
                buf.as_slice().into()
            })
            .collect();
        Self { packed, config }
    }

    /// Test-only: replace every marking with `transform(marking)` (e.g. reduced ->
    /// full-net expansion), re-packing at full u64 width (the expanded markings may
    /// span more places than the stored config).
    #[cfg(test)]
    pub(crate) fn expand_in_place<E>(
        &mut self,
        mut transform: impl FnMut(&[u64]) -> Result<Vec<u64>, E>,
    ) -> Result<(), E> {
        let mut scratch = Vec::new();
        let mut expanded: Vec<Vec<u64>> = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            self.unpack_into(i, &mut scratch);
            expanded.push(transform(&scratch)?);
        }
        *self = Self::from_unpacked(&expanded);
        Ok(())
    }

    /// Rebuild the store by fallibly transforming every marking (e.g. reduced ->
    /// full-net expansion) and re-packing under `new_config`. Used by the CTL
    /// pipeline's slice/reduction expansion write path.
    pub(crate) fn try_rebuild_with<E>(
        &self,
        new_config: MarkingConfig,
        mut transform: impl FnMut(&[u64]) -> Result<Vec<u64>, E>,
    ) -> Result<Self, E> {
        let mut scratch = Vec::new();
        let mut buf = Vec::new();
        let mut out = Vec::with_capacity(self.packed.len());
        for i in 0..self.packed.len() {
            self.unpack_into(i, &mut scratch);
            let transformed = transform(&scratch)?;
            buf.clear();
            pack_marking_config(&transformed, &new_config, &mut buf);
            out.push(buf.as_slice().into());
        }
        Ok(Self {
            packed: out,
            config: new_config,
        })
    }
}

fn build_graph_core(
    net: &PetriNet,
    config: &ExplorationConfig,
    mode: GraphBuildMode,
) -> GraphBuildArtifacts {
    let ExplorationSetup {
        marking_config,
        pack_capacity,
        num_places,
        num_transitions,
        initial_packed,
    } = ExplorationSetup::analyze(net);

    let mut registry = GraphStateRegistry::with_initial(&initial_packed, config.max_states());
    let mut adj: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut markings = mode.captures_markings().then(|| {
        // Store the PACKED initial marking (the same compact form used for dedup),
        // not the fat `Vec<u64>` — see `MarkingStore`.
        let mut store = MarkingStore::new(marking_config.clone());
        store.push_packed(initial_packed.clone());
        store
    });
    let mut queue: VecDeque<(u32, Box<[u8]>)> = VecDeque::new();

    adj.push(Vec::new());
    queue.push_back((0, initial_packed));

    let mut completed = true;
    let mut current_tokens = Vec::with_capacity(num_places);
    // One adaptive probe for BOTH the wall-clock deadline and the memory
    // budget (audit 2026-07-02 → resource-guard refactor): it self-tunes its
    // poll cadence to the loop's speed and remaining headroom, so there are no
    // fixed 4096/512 counters. `completed = false` routes the caller
    // (`pipeline.rs`, `if !full.graph.completed`) into the per-property
    // fallback / `CannotCompute` — it never flips a verdict.
    let mut probe = crate::memory::explorer_probe(config.deadline());
    let por_strategy = PorStrategy::None;
    let mut successor_provider = InterpretedSuccessorProvider::new(
        net,
        &marking_config,
        pack_capacity,
        num_transitions,
        None,
        &por_strategy,
        None,
    );

    'explore: while let Some((sid, current_packed)) = queue.pop_front() {
        if probe.over_budget() {
            completed = false;
            break;
        }

        unpack_marking_config(&current_packed, &marking_config, &mut current_tokens);

        let mut edges = Vec::new();
        let mut reached_state_limit = false;
        successor_provider.for_each_successor(&mut current_tokens, &mut |successor| {
            // Per-successor tick bounds the byte-overshoot window WITHIN one
            // wide expansion (which can admit tens of thousands of ~100 KB
            // markings) — the per-pop tick alone cannot. Same adaptive probe.
            if probe.over_budget() {
                completed = false;
                reached_state_limit = true;
                return SuccessorVisit::Stop;
            }
            let next_id = match registry.admit_precomputed(successor.fingerprint, successor.packed)
            {
                StateAdmission::Existing(existing) => existing,
                StateAdmission::Inserted(new_state) => {
                    adj.push(Vec::new());
                    if let Some(markings) = markings.as_mut() {
                        // Store the packed bytes (already computed for dedup), not
                        // the fat `Vec<u64>` — the ~8x CTL memory win.
                        markings.push_packed(Box::from(successor.packed));
                    }
                    queue.push_back((new_state.state_id, new_state.packed));
                    new_state.state_id
                }
                StateAdmission::LimitReached => {
                    completed = false;
                    reached_state_limit = true;
                    return SuccessorVisit::Stop;
                }
                StateAdmission::FingerprintMismatch { .. } => {
                    completed = false;
                    reached_state_limit = true;
                    return SuccessorVisit::Stop;
                }
            };
            edges.push((next_id, successor.transition.0));
            SuccessorVisit::Continue
        });
        // Fail-closed (#22): a token-count overflow aborted enumeration of this
        // state's successors — the graph is incomplete, so stop and report it as
        // not-fully-explored (CANNOT_COMPUTE), never a complete-but-wrong graph.
        if successor_provider.token_overflow_declined() {
            completed = false;
            adj[sid as usize] = edges;
            break 'explore;
        }
        if reached_state_limit {
            adj[sid as usize] = edges;
            break 'explore;
        }
        adj[sid as usize] = edges;
    }

    GraphBuildArtifacts {
        graph: ReachabilityGraph {
            num_states: registry.len(),
            adj,
            completed,
        },
        markings,
    }
}

/// Build the full reachability graph via BFS.
///
/// Unlike [`super::explore`] which uses an observer pattern for early termination,
/// this function records all edges for structural analysis (SCC, liveness).
/// Uses compact marking storage and delta-based firing.
pub(crate) fn explore_and_build_graph(
    net: &PetriNet,
    config: &ExplorationConfig,
) -> ReachabilityGraph {
    build_graph_core(net, config, GraphBuildMode::StructureOnly).graph
}

/// Full reachability graph with stored markings for model checking.
///
/// Used by CTL and LTL examinations which need to evaluate state predicates
/// at each state during fixpoint/product computations.
pub(crate) struct FullReachabilityGraph {
    /// Adjacency structure and completion flag.
    pub(crate) graph: ReachabilityGraph,
    /// Packed marking for each state, indexed by state ID (decode via its methods).
    pub(crate) markings: MarkingStore,
}

/// Build the full reachability graph with markings via BFS.
///
/// Like [`explore_and_build_graph`] but also stores the marking for each state,
/// enabling CTL/LTL model checking which needs to evaluate predicates at every
/// state. Uses compact storage for dedup, delta-based firing, and preserves
/// full u64 markings for predicate evaluation.
pub(crate) fn explore_full(net: &PetriNet, config: &ExplorationConfig) -> FullReachabilityGraph {
    let GraphBuildArtifacts { graph, markings } =
        build_graph_core(net, config, GraphBuildMode::WithMarkings);
    FullReachabilityGraph {
        graph,
        markings: markings.expect("invariant: full graph mode captures markings"),
    }
}
