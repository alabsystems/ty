// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! On-the-fly product emptiness checking: system × GBA.
//!
//! Unlike [`super::product`] which requires a pre-built [`FullReachabilityGraph`],
//! this module computes system successors lazily by firing transitions on the
//! Petri net. Benefits:
//!
//! 1. **No full-graph memory cost.** The system reachability graph is never
//!    materialized — only the product graph is stored.
//! 2. **POR-compatible.** Stubborn set reduction can filter successors at
//!    each product state (Phase 3).
//! 3. **Early termination potential.** Future DFS-based emptiness can stop
//!    as soon as an accepting cycle is found.
//! 4. **Per-marking atom-satisfaction memo.** Atom truth depends only on the
//!    system marking, so each distinct marking's atom values are evaluated
//!    once into a bitmask and every GBA guard check becomes two bitwise
//!    tests — see [`MarkingTable`]. Pure caching; verdicts are bit-identical
//!    with the memo on or off.

use std::collections::VecDeque;
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::PnmlError;
use crate::explorer::ExplorationSetup;
use crate::marking::{pack_marking_config, unpack_marking_config, MarkingConfig};
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::reduction::ReducedNet;
use crate::resolved_predicate::{eval_predicate, ResolvedPredicate};
use crate::scc::tarjan_scc_slices;
use crate::stubborn::{compute_stubborn_set, DependencyGraph, PorStrategy};

use super::gba::{accept_bit, AcceptanceMasks, Gba, GbaStateId, GbaTransition};

const PRODUCT_STATE_LIMIT: usize = 50_000_000;

/// Pre-computed POR context for on-the-fly product exploration.
///
/// When present, the successor generation loop at each product state
/// calls [`compute_stubborn_set`] to fire only a subset of enabled
/// transitions, reducing the explored product graph while preserving
/// stutter-equivalent traces.
pub(crate) struct PorContext {
    pub(crate) dep: DependencyGraph,
    /// Static whole-formula visible set (sound for every product node) —
    /// the C2 fallback row whenever per-state visibility is off or anomalous.
    pub(crate) visible: Vec<TransitionIdx>,
    /// When true, the DFS builder additionally computes **per-Büchi-state**
    /// visibility rows (reachability-closed guard-atom sets, see
    /// `examinations::ltl_por::ltl_visible_per_gba_state`) and tests C2
    /// against the current product node's row. Every row is a subset of
    /// `visible`, so this only shrinks ample-set rejections — any anomaly
    /// falls back to the static row (always sound). `false` reproduces the
    /// static whole-formula behavior exactly.
    pub(crate) per_state_visibility: bool,
}

/// Check guard satisfaction by evaluating atoms directly against a marking.
///
/// This is the direct (non-memoized) eval path: it remains the fallback for
/// formulas with more than 64 atoms (or `TY_LTL_DISABLE_MEMO=1` runs), and
/// the oracle the memo cross-check asserts against per edge.
fn guard_satisfied_at_marking(
    trans: &GbaTransition,
    atoms: &[ResolvedPredicate],
    marking: &[u64],
    net: &PetriNet,
) -> bool {
    trans
        .pos_atoms
        .iter()
        .all(|&a| eval_predicate(&atoms[a], marking, net))
        && trans
            .neg_atoms
            .iter()
            .all(|&a| !eval_predicate(&atoms[a], marking, net))
}

// ─────────────────────────────────────────────────────────────────────────
// Per-system-marking atom-satisfaction memo (pure caching).
//
// Atom truth depends ONLY on the system marking: every `ResolvedPredicate`
// atom is a total, pure function of the ORIGINAL-net marking, which is itself
// a fixed function of the packed reduced-net marking (`unpack_marking_config`
// reconstructs implied places deterministically; `expand_marking_into`
// applies per-run-constant place maps / scales / constants / lateral-fusion
// offsets). The pre-memo code nevertheless re-ran
// unpack → expand → `eval_predicate` for every GBA edge of every product
// visit of the same system marking.
//
// The memo interns each distinct packed marking once and computes a ≤64-bit
// atom-satisfaction mask at FIRST insertion by the EXACT same eval path the
// per-edge code used (unpack the reduced marking → expand to original-net
// space → `eval_predicate` per atom — never on reduced tokens); guard checks
// then become two bitwise tests against per-GBA-edge (pos, neg) masks.
// Eager full-atom eval equals the short-circuited `all` because
// `eval_predicate` is total and pure. Pure caching: same packed bytes ⇒ same
// expanded marking ⇒ identical bits — verdicts are bit-identical with the
// memo on or off (debug builds, and release runs with `TY_LTL_MEMO_CHECK=1`,
// assert exactly that per edge). Injectivity of `pack` on reachable markings
// is already assumed by the pre-existing seen-marking dedupe.
// ─────────────────────────────────────────────────────────────────────────

/// `TY_LTL_DISABLE_MEMO=1` forces the direct per-edge eval path (A/B gate).
fn memo_disabled_by_env() -> bool {
    std::env::var_os("TY_LTL_DISABLE_MEMO").is_some_and(|v| !v.is_empty() && v != "0")
}

/// `TY_LTL_DUMP_LASSO=1` — dev-only diagnostic: when the product search finds
/// an accepting SCC (verdict FALSE), print a concrete lasso (stem + cycle)
/// with the fired transition names and the markings of every atom-relevant
/// place at each position. Output only; never changes exploration or verdicts.
fn lasso_dump_enabled() -> bool {
    std::env::var_os("TY_LTL_DUMP_LASSO").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Cross-check every memoized guard decision against the direct eval path:
/// always in debug builds, and in release when `TY_LTL_MEMO_CHECK=1`.
fn memo_check_enabled() -> bool {
    cfg!(debug_assertions)
        || std::env::var_os("TY_LTL_MEMO_CHECK").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Per-GBA-edge atom bitmasks: a guard holds at a marking with atom mask `m`
/// iff `(m & pos) == pos && (m & neg) == 0` — the mask-test equivalent of
/// [`guard_satisfied_at_marking`].
struct GuardMasks {
    /// Parallel to `gba.initial_transitions`.
    initial: Vec<(u64, u64)>,
    /// Parallel to `gba.transitions` (outer and inner).
    by_state: Vec<Vec<(u64, u64)>>,
}

/// Build the per-edge masks once per run. Returns `None` when the formula has
/// more than 64 atoms (or a malformed atom index), in which case callers keep
/// the direct per-edge eval path verbatim.
fn build_guard_masks(gba: &Gba, num_atoms: usize) -> Option<GuardMasks> {
    if num_atoms > 64 {
        return None;
    }
    let mask_of = |trans: &GbaTransition| -> Option<(u64, u64)> {
        let mut pos = 0u64;
        let mut neg = 0u64;
        for &a in &trans.pos_atoms {
            if a >= num_atoms {
                return None;
            }
            pos |= 1u64 << a;
        }
        for &a in &trans.neg_atoms {
            if a >= num_atoms {
                return None;
            }
            neg |= 1u64 << a;
        }
        Some((pos, neg))
    };
    let initial = gba
        .initial_transitions
        .iter()
        .map(mask_of)
        .collect::<Option<Vec<_>>>()?;
    let by_state = gba
        .transitions
        .iter()
        .map(|row| row.iter().map(mask_of).collect::<Option<Vec<_>>>())
        .collect::<Option<Vec<_>>>()?;
    Some(GuardMasks { initial, by_state })
}

/// Interned system markings: dedupe + distinct-marking budget (replacing the
/// former `seen_system_markings` set), dense `u32` ids (making product keys
/// `Copy` and storing each packed marking once instead of once per product
/// state), and the per-marking atom-satisfaction bitmask.
struct MarkingTable<'a> {
    atoms: &'a [ResolvedPredicate],
    reduced: &'a ReducedNet,
    original_net: &'a PetriNet,
    config: &'a MarkingConfig,
    max_system_states: usize,
    /// Atom masks are computed at intern time (≤64 atoms, memo not disabled).
    memo_engaged: bool,
    /// Assert memoized guard decisions against direct eval per edge.
    check: bool,
    ids: FxHashMap<Box<[u8]>, u32>,
    /// id → packed bytes (the single shared copy product states reference).
    bytes: Vec<Box<[u8]>>,
    /// id → atom-satisfaction bitmask: bit `i` = `eval_predicate(atoms[i])`
    /// at the marking, evaluated in ORIGINAL-net space. All-zero (unused)
    /// when the memo is not engaged.
    atom_masks: Vec<u64>,
    tokens_scratch: Vec<u64>,
    expanded_scratch: Vec<u64>,
}

impl<'a> MarkingTable<'a> {
    fn new(
        atoms: &'a [ResolvedPredicate],
        reduced: &'a ReducedNet,
        original_net: &'a PetriNet,
        config: &'a MarkingConfig,
        max_system_states: usize,
        memo_engaged: bool,
    ) -> Self {
        Self {
            atoms,
            reduced,
            original_net,
            config,
            max_system_states,
            memo_engaged,
            check: memo_engaged && memo_check_enabled(),
            ids: FxHashMap::default(),
            bytes: Vec::new(),
            atom_masks: Vec::new(),
            tokens_scratch: Vec::new(),
            expanded_scratch: Vec::new(),
        }
    }

    /// Intern a packed system marking, returning its dense id — or `None`
    /// when the distinct-marking budget is exhausted (the run is then
    /// inconclusive, exactly like the former `record_system_marking`).
    /// Lookups borrow `packed` directly; allocation happens only on a miss.
    ///
    /// On first insertion only, when the memo is engaged, the marking's atom
    /// mask is computed by the EXACT pre-memo eval path: unpack the reduced
    /// marking from the packed bytes, expand to original-net space, then
    /// `eval_predicate` per atom.
    fn intern_marking(&mut self, packed: &[u8]) -> Result<Option<u32>, PnmlError> {
        if let Some(&id) = self.ids.get(packed) {
            return Ok(Some(id));
        }
        if self.ids.len() >= self.max_system_states {
            return Ok(None);
        }
        let mask = if self.memo_engaged {
            unpack_marking_config(packed, self.config, &mut self.tokens_scratch);
            self.reduced
                .expand_marking_into(&self.tokens_scratch, &mut self.expanded_scratch)?;
            let mut mask = 0u64;
            for (i, atom) in self.atoms.iter().enumerate() {
                if eval_predicate(atom, &self.expanded_scratch, self.original_net) {
                    mask |= 1u64 << i;
                }
            }
            mask
        } else {
            0
        };
        let id = self.bytes.len() as u32;
        let owned: Box<[u8]> = packed.into();
        self.bytes.push(owned.clone());
        self.atom_masks.push(mask);
        self.ids.insert(owned, id);
        Ok(Some(id))
    }

    /// Unpack + expand marking `id` into `expanded_scratch` — the pre-memo
    /// per-successor eval path, used verbatim by the non-memo guard path and
    /// by the memo cross-check.
    fn expand_marking_of(&mut self, id: u32) -> Result<(), PnmlError> {
        unpack_marking_config(
            &self.bytes[id as usize],
            self.config,
            &mut self.tokens_scratch,
        );
        self.reduced
            .expand_marking_into(&self.tokens_scratch, &mut self.expanded_scratch)
    }
}

/// Invoke `f` for every transition in `transitions` whose guard holds at the
/// marking with id `mid` — bit-identical to calling
/// [`guard_satisfied_at_marking`] per edge. `f` also receives the
/// transition's index within `transitions` (the [`AcceptanceMasks::edge`]
/// key for the packed edge-acceptance words).
///
/// Memo path (`row` present, parallel to `transitions`): two bitwise tests
/// per edge against the marking's interned atom mask. Non-memo path: unpack +
/// expand once, then direct `eval_predicate` per edge — the pre-memo code
/// verbatim.
fn for_satisfied_edges<'g>(
    table: &mut MarkingTable<'_>,
    transitions: &'g [GbaTransition],
    row: Option<&[(u64, u64)]>,
    mid: u32,
    mut f: impl FnMut(usize, &'g GbaTransition),
) -> Result<(), PnmlError> {
    match row {
        Some(row) => {
            debug_assert_eq!(
                row.len(),
                transitions.len(),
                "guard-mask row must be parallel to the GBA transition list"
            );
            let mask = table.atom_masks[mid as usize];
            if table.check {
                table.expand_marking_of(mid)?;
                for (trans, &(pos, neg)) in transitions.iter().zip(row) {
                    let direct = guard_satisfied_at_marking(
                        trans,
                        table.atoms,
                        &table.expanded_scratch,
                        table.original_net,
                    );
                    let memoized = (mask & pos) == pos && (mask & neg) == 0;
                    assert_eq!(
                        memoized, direct,
                        "LTL atom-mask memo diverged from direct guard eval \
                         (marking id {mid}, pos={pos:#x}, neg={neg:#x}, mask={mask:#x})"
                    );
                }
            }
            for (e, (trans, &(pos, neg))) in transitions.iter().zip(row).enumerate() {
                if (mask & pos) == pos && (mask & neg) == 0 {
                    f(e, trans);
                }
            }
        }
        None => {
            table.expand_marking_of(mid)?;
            for (e, trans) in transitions.iter().enumerate() {
                if guard_satisfied_at_marking(
                    trans,
                    table.atoms,
                    &table.expanded_scratch,
                    table.original_net,
                ) {
                    f(e, trans);
                }
            }
        }
    }
    Ok(())
}

/// Env-gated (`TY_LTL_PRODUCT_STATS=1`) per-run product throughput stats,
/// printed on every exit path via `Drop` (deadline/budget returns included).
/// Measurement only — no effect on exploration or verdicts.
struct ProductStats {
    enabled: bool,
    label: &'static str,
    start: Instant,
    expansions: u64,
    product_states: usize,
    markings: usize,
}

impl ProductStats {
    fn new(label: &'static str) -> Self {
        Self {
            enabled: std::env::var_os("TY_LTL_PRODUCT_STATS")
                .is_some_and(|v| !v.is_empty() && v != "0"),
            label,
            start: Instant::now(),
            expansions: 0,
            product_states: 0,
            markings: 0,
        }
    }
}

impl Drop for ProductStats {
    fn drop(&mut self) {
        if self.enabled {
            let secs = self.start.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "LTL product stats [{}]: expansions={} product_states={} markings={} \
                 elapsed={:.3}s expansions/s={:.0} states/s={:.0}",
                self.label,
                self.expansions,
                self.product_states,
                self.markings,
                secs,
                self.expansions as f64 / secs,
                self.product_states as f64 / secs,
            );
        }
    }
}

/// Debug-only: pack → unpack must round-trip the initial marking. Both the
/// memo (which keys atom masks by packed bytes and re-derives token vectors
/// by unpacking) and the pre-memo explorer (which unpacks every product
/// node's marking, the initial one included) depend on this codec invariant;
/// this assert merely surfaces a violation early.
fn debug_assert_initial_roundtrip(setup: &ExplorationSetup, net: &PetriNet) {
    if cfg!(debug_assertions) {
        let mut tokens = Vec::new();
        unpack_marking_config(&setup.initial_packed, &setup.marking_config, &mut tokens);
        debug_assert_eq!(
            tokens, net.initial_marking,
            "pack/unpack failed to round-trip the initial marking"
        );
    }
}

/// On-the-fly product emptiness: system × GBA accepting cycle detection.
///
/// Builds the product graph lazily — system successors are computed by firing
/// transitions on the reduced `net`, then expanded to original-net space via
/// `reduced` for atom evaluation against `original_net`.
///
/// Returns `Some(true)` if an accepting cycle exists (formula is violated),
/// `Some(false)` if no accepting cycle (formula holds), or `None` if the
/// system-marking or product-state budget was exceeded.
pub(super) fn on_the_fly_product_emptiness(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    por: Option<&PorContext>,
    max_system_states: usize,
    deadline: Option<Instant>,
) -> Result<Option<bool>, PnmlError> {
    on_the_fly_impl(
        gba,
        net,
        reduced,
        original_net,
        atoms,
        por,
        max_system_states,
        PRODUCT_STATE_LIMIT,
        deadline,
        memo_disabled_by_env(),
    )
}

#[cfg(test)]
pub(super) fn on_the_fly_product_emptiness_with_limit(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    max_system_states: usize,
    product_state_limit: usize,
    deadline: Option<Instant>,
) -> Result<Option<bool>, PnmlError> {
    on_the_fly_impl(
        gba,
        net,
        reduced,
        original_net,
        atoms,
        None,
        max_system_states,
        product_state_limit,
        deadline,
        memo_disabled_by_env(),
    )
}

/// Test-only entry with an explicit memo toggle (BFS / no-POR path), for the
/// memo-vs-direct differential tests — deterministic regardless of env vars.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn on_the_fly_product_emptiness_with_limit_memo_toggle(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    max_system_states: usize,
    product_state_limit: usize,
    deadline: Option<Instant>,
    disable_memo: bool,
) -> Result<Option<bool>, PnmlError> {
    on_the_fly_impl(
        gba,
        net,
        reduced,
        original_net,
        atoms,
        None,
        max_system_states,
        product_state_limit,
        deadline,
        disable_memo,
    )
}

/// Test-only entry with an explicit memo toggle (DFS+POR path).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn on_the_fly_product_emptiness_por_memo_toggle(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    por: &PorContext,
    max_system_states: usize,
    deadline: Option<Instant>,
    disable_memo: bool,
) -> Result<Option<bool>, PnmlError> {
    on_the_fly_dfs_impl(
        gba,
        net,
        reduced,
        original_net,
        atoms,
        Some(por),
        max_system_states,
        PRODUCT_STATE_LIMIT,
        deadline,
        disable_memo,
        None,
    )
}

/// Collect all enabled transitions at a marking (fallback when POR is off or gives no reduction).
fn all_enabled(net: &PetriNet, marking: &[u64], num_transitions: usize) -> Vec<TransitionIdx> {
    (0..num_transitions)
        .map(|i| TransitionIdx(i as u32))
        .filter(|&t| net.is_enabled(marking, t))
        .collect()
}

/// Product adjacency in compressed-sparse-row form (audit S5): one flat
/// edge array sliced per state via `offsets`, replacing the per-state
/// `Vec<u32>` headers and allocations for the long-lived Tarjan/SCC phase.
pub(super) struct CsrProductAdj {
    /// `offsets.len() == state_count + 1`.
    offsets: Vec<u32>,
    edges: Vec<u32>,
}

impl CsrProductAdj {
    pub(super) fn state_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Successor list of state `s` — identical contents and order to the
    /// nested row it was flattened from.
    pub(super) fn neighbors(&self, s: usize) -> &[u32] {
        &self.edges[self.offsets[s] as usize..self.offsets[s + 1] as usize]
    }
}

/// Flatten a fully-built nested product adjacency into CSR form, freeing
/// each per-state `Vec` as it is copied so the peak holds at most one row
/// twice (audit S5). Row contents and order are preserved verbatim; only
/// the storage layout changes.
///
/// Returns `None` when the total edge count exceeds the `u32` offset space
/// (>4.29B product edges — beyond any budget the builders admit in
/// practice). Callers decline the run (fail closed, matching the sibling
/// state/deadline budget guards) rather than risk offset truncation.
pub(super) fn flatten_adjacency(adj: &mut Vec<Vec<u32>>) -> Option<CsrProductAdj> {
    let total: usize = adj.iter().map(Vec::len).sum();
    if u32::try_from(total).is_err() {
        return None;
    }
    let mut offsets = Vec::with_capacity(adj.len() + 1);
    offsets.push(0u32);
    let mut edges = Vec::with_capacity(total);
    for row in adj.iter_mut() {
        edges.extend_from_slice(row);
        *row = Vec::new(); // drop the row's allocation eagerly
        offsets.push(edges.len() as u32);
    }
    adj.clear();
    adj.shrink_to_fit();
    Some(CsrProductAdj { offsets, edges })
}

/// Mutable product-graph state shared by the BFS and DFS builders.
///
/// Product keys are `(system marking id, GBA state)` — `Copy` — and the
/// packed marking bytes live once in the [`MarkingTable`], not cloned per
/// product state / per probe.
struct ProductGraph {
    ids: FxHashMap<(u32, GbaStateId), u32>,
    adj: Vec<Vec<u32>>,
    /// Per-state acceptance, packed and strided by the codec's `num_words`
    /// (audit S3): `accept[pid*nw..(pid+1)*nw]` are the state's words —
    /// no per-state `Vec<bool>` allocation.
    accept: Vec<u64>,
    /// Per-state outgoing edges in generation order (NOT deduped — one entry
    /// per satisfied GBA transition, parallel to `edge_words`). The former
    /// `edge_accept` successor component.
    edge_succ: Vec<Vec<u32>>,
    /// Per-state packed edge-acceptance words, strided by `num_words`
    /// parallel to `edge_succ` — the former per-edge `Vec<bool>` clones.
    edge_words: Vec<Vec<u64>>,
    marking: Vec<u32>,
    gba_state: Vec<GbaStateId>,
    /// DFS-only (C3 cycle proviso); untouched by the BFS builder.
    on_stack: Vec<bool>,
    /// DFS-only; untouched by the BFS builder (the queue plays that role).
    expanded: Vec<bool>,
    /// TEST-ONLY: `fully_expanded[pid]` is true iff node `pid` fired its
    /// COMPLETE enabled set (no ample reduction, or C3 forced full expansion).
    /// Distinct from `expanded` (which means "visited / adjacency recorded").
    /// Consumed only by the C3-proviso structural verifier in the gate.
    #[cfg(test)]
    fully_expanded: Vec<bool>,
}

impl ProductGraph {
    fn new() -> Self {
        Self {
            ids: FxHashMap::default(),
            adj: Vec::new(),
            accept: Vec::new(),
            edge_succ: Vec::new(),
            edge_words: Vec::new(),
            marking: Vec::new(),
            gba_state: Vec::new(),
            on_stack: Vec::new(),
            expanded: Vec::new(),
            #[cfg(test)]
            fully_expanded: Vec::new(),
        }
    }

    /// Intern a product state `(marking id, gstate)`, returning its id. Newly
    /// discovered states start off-stack and un-expanded. State acceptance is
    /// the GBA state's pre-packed words, copied inline into `accept`.
    fn intern(&mut self, mid: u32, gstate: GbaStateId, acc: &AcceptanceMasks) -> u32 {
        let key = (mid, gstate);
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.ids.len() as u32;
        self.adj.push(Vec::new());
        self.edge_succ.push(Vec::new());
        self.edge_words.push(Vec::new());
        self.marking.push(mid);
        self.gba_state.push(gstate);
        self.accept.extend_from_slice(acc.state(gstate));
        self.on_stack.push(false);
        self.expanded.push(false);
        #[cfg(test)]
        self.fully_expanded.push(false);
        self.ids.insert(key, id);
        id
    }
}

#[allow(clippy::too_many_arguments)]
fn on_the_fly_impl(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    por: Option<&PorContext>,
    max_system_states: usize,
    product_state_limit: usize,
    deadline: Option<Instant>,
    disable_memo: bool,
) -> Result<Option<bool>, PnmlError> {
    if gba.num_states == 0 {
        return Ok(Some(false));
    }

    // When a POR context is supplied, route to the DFS builder that enforces
    // the ample-set cycle proviso (C3). The BFS builder below cannot detect the
    // closing edges C3 requires, so it is used only for the exact (no-POR) path.
    if por.is_some() {
        return on_the_fly_dfs_impl(
            gba,
            net,
            reduced,
            original_net,
            atoms,
            por,
            max_system_states,
            product_state_limit,
            deadline,
            disable_memo,
            None,
        );
    }

    let num_accept = gba.acceptance.len();
    let acc = AcceptanceMasks::from_gba(gba);
    let setup = ExplorationSetup::analyze(net);
    debug_assert_initial_roundtrip(&setup, net);

    let masks = if disable_memo {
        None
    } else {
        build_guard_masks(gba, atoms.len())
    };
    let mut table = MarkingTable::new(
        atoms,
        reduced,
        original_net,
        &setup.marking_config,
        max_system_states,
        masks.is_some(),
    );

    let mut g = ProductGraph::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    // Reusable buffers.
    let mut tokens_buf = Vec::with_capacity(setup.num_places);
    let mut pack_buf = Vec::with_capacity(setup.pack_capacity);

    let Some(init_mid) = table.intern_marking(&setup.initial_packed)? else {
        return Ok(None);
    };

    // Evaluate initial GBA transitions against the initial system state.
    let mut roots: Vec<u32> = Vec::new();
    {
        let row = masks.as_ref().map(|m| m.initial.as_slice());
        for_satisfied_edges(
            &mut table,
            &gba.initial_transitions,
            row,
            init_mid,
            |_, trans| {
                let before = g.ids.len();
                let pid = g.intern(init_mid, trans.successor, &acc);
                if g.ids.len() != before {
                    queue.push_back(pid);
                }
                roots.push(pid);
            },
        )?;
    }
    roots.sort_unstable();
    roots.dedup();

    // BFS to build product graph with lazy system successors.
    //
    // One adaptive probe for BOTH the deadline and the memory budget: the
    // product graph's growth is bounded otherwise only by ITEM counts
    // (PRODUCT_STATE_LIMIT, max_system_states = usize::MAX on the auto-sized
    // MCC path), which say nothing about bytes (adjacency, edge lists,
    // acceptance words, marking table all scale with the net). `Ok(None)` is
    // the standard inconclusive.
    let mut probe = crate::memory::explorer_probe(deadline);
    let mut stats = ProductStats::new("BFS");
    while let Some(pid) = queue.pop_front() {
        if probe.over_budget() {
            return Ok(None);
        }
        stats.expansions += 1;
        stats.product_states = g.ids.len();
        stats.markings = table.ids.len();

        let mid = g.marking[pid as usize];
        let gba_state = g.gba_state[pid as usize];

        // Unpack to get reduced-net tokens (kept live across fire/undo).
        unpack_marking_config(
            &table.bytes[mid as usize],
            &setup.marking_config,
            &mut tokens_buf,
        );

        // `por.is_some()` was routed to the DFS builder above, so the BFS
        // body always full-expands.
        let to_fire = all_enabled(net, &tokens_buf, setup.num_transitions);

        let edges = &gba.transitions[gba_state as usize];
        let row = masks
            .as_ref()
            .map(|m| m.by_state[gba_state as usize].as_slice());

        // `successors` is the generation-order edge list (parallel to
        // `edge_word_buf`, one entry per satisfied GBA transition); it is
        // copied into `edge_succ` before being sorted/deduped into `adj`.
        let mut successors: Vec<u32> = Vec::new();
        let mut edge_word_buf: Vec<u64> = Vec::new();

        // GBA guards are evaluated against the SUCCESSOR system state.
        // See product.rs comments for the GPVW construction semantics.
        let mut on_succ = |succ_mid: u32, e: usize, trans: &GbaTransition| {
            let before = g.ids.len();
            let succ_pid = g.intern(succ_mid, trans.successor, &acc);
            if g.ids.len() != before {
                queue.push_back(succ_pid);
            }
            successors.push(succ_pid);
            edge_word_buf.extend_from_slice(acc.edge(gba_state, e));
        };

        // Fire each transition, intern the successor marking, then test the
        // GBA guards against its (memoized) atom mask — no staging vector,
        // no per-successor Box allocation on memo hits.
        for &trans in &to_fire {
            // Fail-closed (#22): token-count overflow leaves `tokens_buf`
            // partially mutated, so do NOT undo — decline (Ok(None)) so the
            // product construction reports CANNOT_COMPUTE, never a wrong run.
            if net.apply_delta(&mut tokens_buf, trans).is_err() {
                return Ok(None);
            }
            pack_marking_config(&tokens_buf, &setup.marking_config, &mut pack_buf);
            net.undo_delta(&mut tokens_buf, trans);
            let Some(succ_mid) = table.intern_marking(&pack_buf)? else {
                return Ok(None);
            };
            for_satisfied_edges(&mut table, edges, row, succ_mid, |e, t| {
                on_succ(succ_mid, e, t)
            })?;
        }
        if to_fire.is_empty() {
            // Deadlock: self-loop with the current marking (`mid` is already
            // interned, so this is a guaranteed table hit).
            for_satisfied_edges(&mut table, edges, row, mid, |e, t| on_succ(mid, e, t))?;
        }
        // Explicitly end `on_succ` to release its mutable borrows of
        // `successors`/`edge_word_buf`/`g`/`queue` before they're used below.
        #[allow(clippy::drop_non_drop)]
        drop(on_succ);

        g.edge_succ[pid as usize] = successors.clone();
        g.edge_words[pid as usize] = edge_word_buf;
        successors.sort_unstable();
        successors.dedup();
        g.adj[pid as usize] = successors;

        if g.ids.len() > product_state_limit {
            return Ok(None);
        }
    }

    // Flatten the adjacency to CSR (and free the nested rows) before the
    // long-lived Tarjan/SCC phase (audit S5). `None` (u32 edge-offset
    // overflow) declines the run, like the state/deadline budgets above.
    let Some(csr) = flatten_adjacency(&mut g.adj) else {
        return Ok(None);
    };
    let accepting = find_accepting_scc(&csr, &g.accept, &g.edge_succ, &g.edge_words, num_accept);
    if let Some(scc) = accepting.as_deref() {
        if lasso_dump_enabled() {
            dump_accepting_lasso(&g, &csr, &mut table, net, &roots, scc);
        }
    }
    Ok(Some(accepting.is_some()))
}

/// SCC-based accepting-cycle detection over a fully-built product graph.
///
/// Returns `true` if some reachable non-trivial SCC satisfies all of the GBA's
/// (generalized) acceptance sets — i.e. the product language is non-empty and
/// the negated formula has an accepting run (the property is violated).
///
/// Shared verbatim by the BFS (full-expansion) and DFS+POR product builders so
/// that the trusted acceptance logic is identical regardless of which builder
/// produced the adjacency.
/// Returns the first accepting SCC itself (rather than a bare `bool`) so the
/// `TY_LTL_DUMP_LASSO` diagnostic can print a concrete witness.
/// `Some(_)` ⇔ the former `scc_has_accepting_cycle(..) == true`.
fn find_accepting_scc(
    product_adj: &CsrProductAdj,
    product_accept: &[u64],
    product_edge_succ: &[Vec<u32>],
    product_edge_words: &[Vec<u64>],
    num_accept: usize,
) -> Option<Vec<u32>> {
    if product_adj.state_count() == 0 {
        return None;
    }

    // Stride of the packed acceptance words (see [`AcceptanceMasks`]).
    let nw = num_accept.div_ceil(64);

    let sccs = tarjan_scc_slices(
        product_adj.state_count(),
        |v| product_adj.neighbors(v),
        |&w| w,
    );

    for scc in &sccs {
        // Non-trivial: has a cycle (size > 1, or size 1 with self-loop).
        let is_nontrivial = if scc.len() > 1 {
            true
        } else {
            let s = scc[0];
            product_adj.neighbors(s as usize).contains(&s)
        };
        if !is_nontrivial {
            continue;
        }

        if num_accept == 0 {
            return Some(scc.clone());
        }

        let scc_set: FxHashSet<u32> = scc.iter().copied().collect();

        let all_accepted = (0..num_accept).all(|i| {
            // State-based acceptance.
            let state_accepted = scc.iter().any(|&s| {
                let s = s as usize;
                accept_bit(&product_accept[s * nw..(s + 1) * nw], i)
            });
            if state_accepted {
                return true;
            }
            // Edge-based acceptance: both source and target in the SCC.
            scc.iter().any(|&s| {
                let s = s as usize;
                let words = &product_edge_words[s];
                product_edge_succ[s].iter().enumerate().any(|(e, succ)| {
                    scc_set.contains(succ) && accept_bit(&words[e * nw..(e + 1) * nw], i)
                })
            })
        });
        if all_accepted {
            return Some(scc.clone());
        }
    }

    None
}

/// Dev-only (`TY_LTL_DUMP_LASSO=1`): print a concrete accepting lasso.
///
/// Reconstructs a shortest stem (BFS over the product adjacency from the
/// initial product states) to the accepting SCC, then a cycle inside the SCC
/// back to the stem's entry state. For every position it prints the fired
/// system transition (reduced-net name; identity reduction = unfolded P/T
/// name), the GBA state, the per-atom truth values, and the tokens of every
/// atom-relevant place (places read by `IntLe` atoms plus the input places of
/// every transition referenced by an `IsFireable` atom). Pure diagnostics.
#[allow(clippy::too_many_lines)]
fn dump_accepting_lasso(
    g: &ProductGraph,
    adj: &CsrProductAdj,
    table: &mut MarkingTable<'_>,
    net: &PetriNet,
    roots: &[u32],
    scc: &[u32],
) {
    let scc_set: FxHashSet<u32> = scc.iter().copied().collect();

    // ── Stem: BFS from the initial product states to the SCC ──
    let mut parent: FxHashMap<u32, u32> = FxHashMap::default();
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for &r in roots {
        if seen.insert(r) {
            queue.push_back(r);
        }
    }
    let mut entry: Option<u32> = roots.iter().copied().find(|r| scc_set.contains(r));
    'bfs: while let Some(u) = queue.pop_front() {
        if entry.is_some() {
            break;
        }
        for &v in adj.neighbors(u as usize) {
            if seen.insert(v) {
                parent.insert(v, u);
                if scc_set.contains(&v) {
                    entry = Some(v);
                    break 'bfs;
                }
                queue.push_back(v);
            }
        }
    }
    let Some(entry) = entry else {
        eprintln!("TY_LTL_DUMP_LASSO: accepting SCC unreachable from the product roots (?)");
        return;
    };
    let mut stem: Vec<u32> = vec![entry];
    while let Some(&p) = parent.get(stem.last().expect("non-empty")) {
        stem.push(p);
    }
    stem.reverse();

    // ── Cycle: shortest path inside the SCC from `entry` back to `entry` ──
    // BFS from `entry` over SCC-internal edges; stop at the first expanded
    // node `u` with an edge back to `entry`. Cycle = entry → … → u → entry.
    let mut cyc_parent: FxHashMap<u32, u32> = FxHashMap::default();
    let mut cyc_seen: FxHashSet<u32> = FxHashSet::default();
    let mut cyc_queue: VecDeque<u32> = VecDeque::new();
    cyc_seen.insert(entry);
    cyc_queue.push_back(entry);
    let mut last_before_entry: Option<u32> = None;
    'cyc: while let Some(u) = cyc_queue.pop_front() {
        if adj.neighbors(u as usize).contains(&entry) {
            last_before_entry = Some(u);
            break 'cyc;
        }
        for &v in adj.neighbors(u as usize) {
            if scc_set.contains(&v) && cyc_seen.insert(v) {
                cyc_parent.insert(v, u);
                cyc_queue.push_back(v);
            }
        }
    }
    // `cycle` lists the full closed walk entry → … → entry (first == last).
    let mut cycle: Vec<u32> = Vec::new();
    if let Some(u) = last_before_entry {
        cycle.push(entry);
        let mut back = vec![u];
        let mut cur = u;
        while cur != entry {
            cur = cyc_parent[&cur];
            back.push(cur);
        }
        // `back` = u, …, entry — reverse the tail onto the walk.
        for &n in back.iter().rev().skip(1) {
            cycle.push(n);
        }
        cycle.push(entry);
    }

    // ── Atom-relevant places: IntLe places + input places of IsFireable
    //    transitions (original-net space, where atoms are evaluated). ──
    let original = table.original_net;
    let mut relevant: std::collections::HashSet<PlaceIdx> = std::collections::HashSet::new();
    let mut atom_transitions: Vec<Vec<TransitionIdx>> = Vec::new();
    for atom in table.atoms {
        atom.collect_places(&mut relevant);
        let mut ts = Vec::new();
        collect_fireable_transitions(atom, &mut ts);
        for &t in &ts {
            for arc in &original.transitions[t.0 as usize].inputs {
                relevant.insert(arc.place);
            }
        }
        atom_transitions.push(ts);
    }
    let mut relevant: Vec<PlaceIdx> = relevant.into_iter().collect();
    relevant.sort_unstable();

    eprintln!(
        "TY_LTL_DUMP_LASSO: accepting SCC of {} product state(s)",
        scc.len()
    );
    for (i, atom) in table.atoms.iter().enumerate() {
        let names: Vec<&str> = atom_transitions[i]
            .iter()
            .map(|t| original.transitions[t.0 as usize].id.as_str())
            .collect();
        eprintln!(
            "  atom[{i}]: {atom:?}{}",
            if names.is_empty() {
                String::new()
            } else {
                format!(" — is-fireable over {} instance(s): {names:?}", names.len())
            }
        );
    }

    for (i, &pid) in stem.iter().enumerate() {
        let fired = if i == 0 {
            "(init)".to_string()
        } else {
            lasso_fired_name(g, table, net, stem[i - 1], pid)
        };
        lasso_print_state(g, table, &relevant, "stem", i, pid, &fired);
    }
    if cycle.is_empty() {
        eprintln!("  cycle: (none reconstructed — SCC cycle search failed?)");
    } else {
        for i in 1..cycle.len() {
            let pid = cycle[i];
            let fired = lasso_fired_name(g, table, net, cycle[i - 1], pid);
            lasso_print_state(g, table, &relevant, "cycle", i, pid, &fired);
        }
        eprintln!(
            "  cycle closes back to stem entry pid={entry} (cycle length {})",
            cycle.len() - 1
        );
    }
}

/// `TY_LTL_DUMP_LASSO` helper: print one lasso position — product state,
/// GBA state, fired transition, per-atom truth values, and the non-zero
/// tokens of the atom-relevant places (original-net space).
fn lasso_print_state(
    g: &ProductGraph,
    table: &mut MarkingTable<'_>,
    relevant: &[PlaceIdx],
    label: &str,
    step: usize,
    pid: u32,
    fired: &str,
) {
    let mid = g.marking[pid as usize];
    if table.expand_marking_of(mid).is_err() {
        eprintln!("  {label}[{step}] pid={pid} <marking expansion failed>");
        return;
    }
    let expanded = table.expanded_scratch.clone();
    let original = table.original_net;
    let atom_vals: Vec<bool> = table
        .atoms
        .iter()
        .map(|a| eval_predicate(a, &expanded, original))
        .collect();
    let places: Vec<String> = relevant
        .iter()
        .filter_map(|p| {
            let tokens = expanded[p.0 as usize];
            (tokens > 0).then(|| format!("{}={}", original.places[p.0 as usize].id, tokens))
        })
        .collect();
    eprintln!(
        "  {label}[{step}] pid={pid} gba={} fired={fired} atoms={atom_vals:?} \
         marked-relevant-places=[{}]",
        g.gba_state[pid as usize],
        places.join(", ")
    );
}

/// `TY_LTL_DUMP_LASSO` helper: name the system transition that takes
/// marking(u) → marking(v) on the exploration (reduced) net, or label the
/// deadlock stutter self-loop.
fn lasso_fired_name(
    g: &ProductGraph,
    table: &mut MarkingTable<'_>,
    net: &PetriNet,
    u: u32,
    v: u32,
) -> String {
    let mid_u = g.marking[u as usize];
    let mid_v = g.marking[v as usize];
    let config = table.config;
    unpack_marking_config(
        &table.bytes[mid_u as usize],
        config,
        &mut table.tokens_scratch,
    );
    let mut tokens = table.tokens_scratch.clone();
    if mid_u == mid_v {
        // Either a deadlock stutter self-loop or an identity-delta fire.
        let any_enabled =
            (0..net.num_transitions()).any(|i| net.is_enabled(&tokens, TransitionIdx(i as u32)));
        if !any_enabled {
            return "(deadlock stutter)".to_string();
        }
    }
    let mut target = Vec::new();
    unpack_marking_config(&table.bytes[mid_v as usize], config, &mut target);
    for i in 0..net.num_transitions() {
        let t = TransitionIdx(i as u32);
        if !net.is_enabled(&tokens, t) {
            continue;
        }
        // #22: this is a best-effort counterexample-edge label only (no
        // verdict). A token-count overflow leaves `tokens` partially mutated,
        // so stop scanning and fall through to the generic label.
        if net.apply_delta(&mut tokens, t).is_err() {
            break;
        }
        let hit = tokens == target;
        net.undo_delta(&mut tokens, t);
        if hit {
            return net.transitions[i].id.clone();
        }
    }
    "(no single-transition step found)".to_string()
}

/// Collect the transition indices referenced by `IsFireable` atoms.
fn collect_fireable_transitions(pred: &ResolvedPredicate, out: &mut Vec<TransitionIdx>) {
    match pred {
        ResolvedPredicate::And(children) | ResolvedPredicate::Or(children) => {
            for child in children {
                collect_fireable_transitions(child, out);
            }
        }
        ResolvedPredicate::Not(inner) => collect_fireable_transitions(inner, out),
        ResolvedPredicate::IsFireable(ts) => out.extend(ts.iter().copied()),
        ResolvedPredicate::IntLe(..) | ResolvedPredicate::True | ResolvedPredicate::False => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────
// DFS + ample-set partial-order reduction (stutter-insensitive LTL).
//
// SOUNDNESS (verdict preservation). The reduced product is built by a DFS that
// enforces the four classical ample-set conditions for stutter-insensitive LTL
// (Clarke–Grumberg–Peled, "Model Checking", ch. 10; Peled 1996):
//
//   C0 (non-emptiness)  ample(s) = ∅ iff enabled(s) = ∅. Guaranteed by the
//        stubborn set (a non-deadlock state yields ≥1 enabled key transition);
//        genuine deadlocks self-loop, exactly as the full builder.
//   C1 (dependency)     no transition outside ample(s) that depends on a
//        transition in ample(s) can fire before one in ample(s). Guaranteed by
//        the Valmari/Schmidt stubborn-set closure (D1+D2) used here.
//   C2 (invisibility)   if ample(s) ⊊ enabled(s) then ample(s) contains no
//        VISIBLE transition. Visibility is per product node: the row for the
//        node's Büchi state covers the REACHABILITY-CLOSED guard-atom set of
//        that state (`ltl_visible_per_gba_state`) — every atom of every guard
//        the product can consult from this node onward, retarding (self-loop)
//        edges included. Closure monotonicity (successor rows ⊆ current row)
//        relativizes the classical exchange induction node-by-node: each
//        commuted ample transition is invisible to all guards in its own
//        future. When per-state rows are disabled or anomalous the static
//        whole-formula row `ltl_visible_reduced_transitions` is used — a sound
//        over-approximation at every node, and a superset of every per-state
//        row. (Over-approximating visibility only reduces POR savings — it
//        can never hide an atom change.)
//   C3 (cycle proviso)  if ample(s) ⊊ enabled(s) then ample(s) closes no cycle.
//        Enforced on the PRODUCT via the DFS stack: if any ample successor is a
//        product state currently on the search stack (a back/closing edge) we
//        fully expand s. In a DFS every directed cycle contains a back edge to
//        an on-stack ancestor, so every cycle of the built graph therefore
//        contains at least one fully-expanded state — no transition is "ignored"
//        forever along a cycle.
//
// These four conditions guarantee the reduced product is stutter-trace
// equivalent to the full product, hence preserves existence of accepting runs.
// POR is only ever engaged for X-free (stutter-insensitive) formulas — the
// caller passes `por = None` whenever the formula contains Next. With `por`
// present but `is_visible` covering every transition the DFS simply full-expands
// everywhere and is exactly the (unreduced) on-the-fly product.
// ─────────────────────────────────────────────────────────────────────────

/// Immutable per-run context for the DFS+POR builder.
struct DfsCtx<'a> {
    gba: &'a Gba,
    net: &'a PetriNet,
    por: &'a PorContext,
    setup: &'a ExplorationSetup,
    /// Per-GBA-edge atom masks; `None` ⇒ direct per-edge guard eval.
    masks: Option<GuardMasks>,
    /// Reduced-net transition index → visible to some Büchi atom of the
    /// WHOLE formula (the static set — sound at every product node).
    is_visible_static: Vec<bool>,
    /// Per-GBA-state visibility rows (reachability-closed guard-atom sets,
    /// built by `examinations::ltl_por::ltl_visible_per_gba_state`).
    /// `None` ⇒ per-state visibility disabled/anomalous: C2 always uses the
    /// static row, reproducing the whole-formula behavior bit-for-bit.
    is_visible_by_state: Option<Vec<Vec<bool>>>,
    /// Packed per-GBA-state / per-GBA-edge acceptance words (audit S3);
    /// `acc.num_accept` is the acceptance-set count.
    acc: AcceptanceMasks,
    product_state_limit: usize,
    /// TEST-ONLY mutation switch. When `true`, the DFS builder SUPPRESSES the
    /// C3 cycle proviso (it never force-expands a node that closes a cycle).
    /// This produces a deliberately UNSOUND reduced product used only by the
    /// C3 differential gate's teeth test to prove that dropping the proviso
    /// mis-decides a liveness property. Always `false` on every production and
    /// non-mutation path — `expand_product_node` reads it only to skip the
    /// `closes_cycle` force-expansion, changing nothing else.
    #[cfg(test)]
    c3_disabled: bool,
}

impl DfsCtx<'_> {
    /// Whether the C3 cycle proviso is active. Always `true` in production
    /// (the field does not exist outside `cfg(test)`); the test mutant flips
    /// it off to build the unsound reduced product the teeth test relies on.
    #[inline]
    fn c3_enabled(&self) -> bool {
        #[cfg(test)]
        {
            !self.c3_disabled
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    /// Visibility row for the C2 test at a product node whose Büchi component
    /// is `gstate`. Uses the per-state row when present and well-formed; on
    /// ANY anomaly (no table, state index out of range, row length mismatch)
    /// falls back to the static whole-formula row — always sound, because
    /// every per-state row is a subset of the static row and the static row
    /// is a sound over-approximation at every node.
    fn visible_row(&self, gstate: GbaStateId) -> &[bool] {
        if let Some(rows) = &self.is_visible_by_state {
            if let Some(row) = rows.get(gstate as usize) {
                if row.len() == self.is_visible_static.len() {
                    return row;
                }
            }
        }
        &self.is_visible_static
    }
}

/// Scratch buffers reused across node expansions.
struct DfsBuffers {
    tokens: Vec<u64>,
    pack: Vec<u8>,
}

/// A GBA-product successor descriptor: (system marking id, GBA successor,
/// packed per-acceptance-set edge words borrowed from the run's
/// [`AcceptanceMasks`]).
type SuccDescriptor<'g> = (u32, GbaStateId, &'g [u64]);

/// Fire `fired` from the marking with id `mid` and evaluate the GBA's outgoing
/// transitions at each system successor, producing product-successor
/// descriptors.
///
/// Returns `Ok(None)` if the distinct-system-marking budget overflowed (the
/// whole check is then inconclusive), mirroring the BFS builder.
fn generate_succ_descriptors<'c>(
    ctx: &'c DfsCtx<'_>,
    buf: &mut DfsBuffers,
    table: &mut MarkingTable<'_>,
    mid: u32,
    gba_state: GbaStateId,
    fired: &[TransitionIdx],
) -> Result<Option<Vec<SuccDescriptor<'c>>>, PnmlError> {
    unpack_marking_config(
        &table.bytes[mid as usize],
        &ctx.setup.marking_config,
        &mut buf.tokens,
    );

    let edges = &ctx.gba.transitions[gba_state as usize];
    let row = ctx
        .masks
        .as_ref()
        .map(|m| m.by_state[gba_state as usize].as_slice());

    let mut descriptors: Vec<SuccDescriptor<'c>> = Vec::new();
    for &trans in fired {
        // Fail-closed (#22): token-count overflow leaves `buf.tokens` partially
        // mutated, so do NOT undo — decline (Ok(None)) to make the check
        // inconclusive (CANNOT_COMPUTE), never a wrong run.
        if ctx.net.apply_delta(&mut buf.tokens, trans).is_err() {
            return Ok(None);
        }
        pack_marking_config(&buf.tokens, &ctx.setup.marking_config, &mut buf.pack);
        ctx.net.undo_delta(&mut buf.tokens, trans);
        let Some(succ_mid) = table.intern_marking(&buf.pack)? else {
            return Ok(None);
        };
        for_satisfied_edges(table, edges, row, succ_mid, |e, t| {
            descriptors.push((succ_mid, t.successor, ctx.acc.edge(gba_state, e)));
        })?;
    }
    if fired.is_empty() {
        // Deadlock: self-loop with the current marking (stutter extension) —
        // `mid` is already interned, so this is a guaranteed table hit.
        for_satisfied_edges(table, edges, row, mid, |e, t| {
            descriptors.push((mid, t.successor, ctx.acc.edge(gba_state, e)));
        })?;
    }
    Ok(Some(descriptors))
}

/// Expand product node `pid`: choose its ample (or full) transition set under
/// C0–C3, intern its successors and record adjacency.
///
/// Returns `Ok(false)` when a budget/limit overflow makes the run inconclusive
/// (caller returns `Ok(None)`), `Ok(true)` on success.
fn expand_product_node(
    ctx: &DfsCtx<'_>,
    g: &mut ProductGraph,
    buf: &mut DfsBuffers,
    table: &mut MarkingTable<'_>,
    pid: u32,
) -> Result<bool, PnmlError> {
    let mid = g.marking[pid as usize];
    let gba_state = g.gba_state[pid as usize];

    unpack_marking_config(
        &table.bytes[mid as usize],
        &ctx.setup.marking_config,
        &mut buf.tokens,
    );

    // Candidate ample set (stubborn enabled subset). `None` ⇒ no reduction.
    let reduced_set = compute_stubborn_set(
        ctx.net,
        &buf.tokens,
        &ctx.por.dep,
        &PorStrategy::DeadlockPreserving,
    );

    // C2: an ample proper subset must not contain a visible transition.
    //
    // Visibility is tested against the CURRENT node's Büchi-state row when
    // per-state visibility is active: the row covers the reachability-closed
    // guard-atom set of `gba_state` — i.e. every atom of every guard the
    // product can consult from this node onward (retarding edges included,
    // closed over GBA reachability), so an ample transition invisible to the
    // row commutes with the deferred transitions without changing ANY guard
    // evaluated later in the constructed run; monotonicity of the closure
    // (succ rows ⊆ this row) keeps that invariant at every deeper node.
    // Any anomaly falls back to the static whole-formula row (always sound).
    // An out-of-range transition index is conservatively treated as visible.
    let row = ctx.visible_row(gba_state);
    let ample: Option<Vec<TransitionIdx>> = match reduced_set {
        Some(rs)
            if !rs
                .iter()
                .any(|t| row.get(t.0 as usize).copied().unwrap_or(true)) =>
        {
            Some(rs)
        }
        _ => None,
    };

    // The full enabled set is only needed when no ample reduction applies or
    // when C3 forces a full expansion. `generate_succ_descriptors` leaves
    // `buf.tokens` restored to this node's marking (paired apply/undo), so
    // computing it lazily after the candidate pass reads the same marking.
    //
    // `did_full_expand`: true iff this node fired its COMPLETE enabled set
    // (no ample reduction, or C3 forced full expansion). Recorded for the
    // test-only C3 structural verifier (see `ProductCapture::fully_expanded`).
    #[cfg(test)]
    let mut did_full_expand = false;
    let descriptors: Vec<SuccDescriptor<'_>> = match ample {
        None => {
            #[cfg(test)]
            {
                did_full_expand = true;
            }
            let enabled = all_enabled(ctx.net, &buf.tokens, ctx.setup.num_transitions);
            match generate_succ_descriptors(ctx, buf, table, mid, gba_state, &enabled)? {
                Some(d) => d,
                None => return Ok(false),
            }
        }
        Some(ample_set) => {
            let candidate =
                match generate_succ_descriptors(ctx, buf, table, mid, gba_state, &ample_set)? {
                    Some(d) => d,
                    None => return Ok(false),
                };
            // C3 cycle proviso: if any ample successor is an already-discovered
            // product state currently on the DFS stack, this state closes a
            // cycle and must be fully expanded. Product keys are `Copy`, so the
            // probe is a plain map lookup — no per-descriptor clone.
            //
            // Self-loop case (length-1 cycle): the node currently being expanded
            // (`pid`) is NOT yet marked `on_stack` — the caller sets
            // `g.on_stack[pid] = true` only AFTER `expand_product_node` returns.
            // So an ample successor equal to `pid` itself (a product self-loop)
            // would be missed by the on-stack test alone, leaving the cycle {pid}
            // with no fully-expanded state — a C3 violation (the classical
            // "ignoring problem" for self-loops, which can yield a wrong "property
            // holds"). Detect `id == pid` explicitly so self-loops force a full
            // expansion. (Cycles of length >= 2 are still caught by `on_stack`:
            // their first-discovered node is a proper on-stack ancestor when the
            // back edge is generated.)
            let closes_cycle = candidate.iter().any(|&(smid, gs, _)| {
                g.ids
                    .get(&(smid, gs))
                    .is_some_and(|&id| id == pid || g.on_stack[id as usize])
            });
            // `c3_enabled()` is always true in production. The test mutant
            // flips it off so a cycle-closing ample set is NOT force-expanded —
            // the deliberately unsound reduction the C3 teeth test catches.
            if closes_cycle && ctx.c3_enabled() {
                #[cfg(test)]
                {
                    did_full_expand = true;
                }
                // Full-set regeneration: every successor marking is now a
                // guaranteed table hit — pack + hash + bit-tests only.
                let enabled = all_enabled(ctx.net, &buf.tokens, ctx.setup.num_transitions);
                match generate_succ_descriptors(ctx, buf, table, mid, gba_state, &enabled)? {
                    Some(d) => d,
                    None => return Ok(false),
                }
            } else {
                candidate
            }
        }
    };

    let mut successors: Vec<u32> = Vec::with_capacity(descriptors.len());
    let mut edge_word_buf: Vec<u64> = Vec::with_capacity(descriptors.len() * ctx.acc.num_words);
    for &(smid, gs, ea) in &descriptors {
        let succ_id = g.intern(smid, gs, &ctx.acc);
        successors.push(succ_id);
        edge_word_buf.extend_from_slice(ea);
    }
    g.edge_succ[pid as usize] = successors.clone();
    g.edge_words[pid as usize] = edge_word_buf;
    successors.sort_unstable();
    successors.dedup();
    g.adj[pid as usize] = successors;
    g.expanded[pid as usize] = true;
    #[cfg(test)]
    {
        // The `intern` calls above may have grown the per-node vectors; index
        // is in range because `pid` itself was interned before this call.
        g.fully_expanded[pid as usize] = did_full_expand;
    }

    if g.ids.len() > ctx.product_state_limit {
        return Ok(false);
    }
    Ok(true)
}

#[derive(Clone, Copy)]
struct DfsFrame {
    pid: u32,
    cursor: usize,
}

/// TEST-ONLY structural snapshot of the reduced product graph, captured at the
/// end of a DFS+POR build for the C3-proviso verifier. Lets the gate check the
/// per-cycle / per-SCC proviso directly on the built adjacency rather than only
/// through the verdict: every non-trivial SCC of `adj` must contain at least one
/// `expanded` (fully-expanded) node (C3). It also carries the verdict-relevant
/// acceptance arrays so the gate can recompute the accepting-SCC verdict on the
/// snapshot if needed. Pure observation — never read by any production path.
#[cfg(test)]
#[derive(Default)]
pub(super) struct ProductCapture {
    /// Product adjacency (`adj[pid]` = sorted, deduped successor pids).
    pub(super) adj: Vec<Vec<u32>>,
    /// `expanded[pid]` is true iff node `pid` fired its FULL enabled set (no
    /// ample reduction applied, or C3 forced a full expansion). A node whose
    /// ample set was a strict subset is NOT marked here.
    pub(super) fully_expanded: Vec<bool>,
    /// Packed state-acceptance words, strided by `num_accept.div_ceil(64)`
    /// per product state (see [`super::gba::AcceptanceMasks`]).
    pub(super) accept: Vec<u64>,
    /// Per-state outgoing edges in generation order (parallel to
    /// `edge_words`; NOT the deduped `adj`).
    pub(super) edge_succ: Vec<Vec<u32>>,
    /// Per-state packed edge-acceptance words, strided like `accept`.
    pub(super) edge_words: Vec<Vec<u64>>,
    pub(super) num_accept: usize,
}

/// TEST-ONLY knobs threaded into the DFS+POR builder by the C3 differential
/// gate: the mutation switch and an optional product-graph snapshot target.
#[cfg(test)]
struct C3Options<'c> {
    /// When `true`, the C3 cycle proviso is suppressed (the unsound mutant).
    c3_disabled: bool,
    /// When present, the built product graph is snapshotted into it.
    capture: Option<&'c mut ProductCapture>,
}

#[allow(clippy::too_many_arguments)]
fn on_the_fly_dfs_impl(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    por: Option<&PorContext>,
    max_system_states: usize,
    product_state_limit: usize,
    deadline: Option<Instant>,
    disable_memo: bool,
    product_size_out: Option<&mut usize>,
) -> Result<Option<bool>, PnmlError> {
    #[cfg(test)]
    let c3 = C3Options {
        c3_disabled: false,
        capture: None,
    };
    on_the_fly_dfs_impl_inner(
        gba,
        net,
        reduced,
        original_net,
        atoms,
        por,
        max_system_states,
        product_state_limit,
        deadline,
        disable_memo,
        product_size_out,
        #[cfg(test)]
        c3,
    )
}

#[allow(clippy::too_many_arguments)]
fn on_the_fly_dfs_impl_inner(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    por: Option<&PorContext>,
    max_system_states: usize,
    product_state_limit: usize,
    deadline: Option<Instant>,
    disable_memo: bool,
    product_size_out: Option<&mut usize>,
    #[cfg(test)] mut c3: C3Options<'_>,
) -> Result<Option<bool>, PnmlError> {
    let por = por.expect("on_the_fly_dfs_impl requires a POR context");
    let setup = ExplorationSetup::analyze(net);
    debug_assert_initial_roundtrip(&setup, net);

    let mut is_visible_static = vec![false; setup.num_transitions];
    for &t in &por.visible {
        if (t.0 as usize) < is_visible_static.len() {
            is_visible_static[t.0 as usize] = true;
        }
    }

    // Per-Büchi-state visibility rows (port-plan P2). Built from the GBA's
    // reachability-closed guard-atom sets; `None` (env-disabled, anomaly, or
    // a row-shape mismatch with the explore net) means C2 uses the static
    // whole-formula row everywhere — exactly the pre-P2 behavior.
    let is_visible_by_state: Option<Vec<Vec<bool>>> = if por.per_state_visibility {
        crate::examinations::ltl_por::ltl_visible_per_gba_state(gba, atoms, reduced).filter(
            |rows| {
                rows.len() == gba.num_states as usize
                    && rows.iter().all(|row| row.len() == setup.num_transitions)
            },
        )
    } else {
        None
    };

    // Invariant (debug builds): each per-state row is pointwise ⊆ the static
    // row — per-state visibility may only REMOVE visibility (the per-state
    // atom set is a subset of all formula atoms and visibility is monotone
    // in the atom set). A violation would mean the static fallback is not a
    // strict over-approximation of the per-state rows.
    #[cfg(debug_assertions)]
    if let Some(rows) = &is_visible_by_state {
        for (q, row) in rows.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                debug_assert!(
                    !v || is_visible_static[t],
                    "per-state visibility row {q} marks transition {t} visible \
                     but the static whole-formula row does not"
                );
            }
        }
    }

    let masks = if disable_memo {
        None
    } else {
        build_guard_masks(gba, atoms.len())
    };
    let memo_engaged = masks.is_some();

    let ctx = DfsCtx {
        gba,
        net,
        por,
        setup: &setup,
        masks,
        is_visible_static,
        is_visible_by_state,
        acc: AcceptanceMasks::from_gba(gba),
        product_state_limit,
        #[cfg(test)]
        c3_disabled: c3.c3_disabled,
    };

    let mut table = MarkingTable::new(
        atoms,
        reduced,
        original_net,
        &setup.marking_config,
        max_system_states,
        memo_engaged,
    );

    let mut g = ProductGraph::new();
    let mut buf = DfsBuffers {
        tokens: Vec::with_capacity(setup.num_places),
        pack: Vec::with_capacity(setup.pack_capacity),
    };

    let Some(init_mid) = table.intern_marking(&setup.initial_packed)? else {
        return Ok(None);
    };

    // Initial product states: GBA initial transitions whose guard holds at the
    // initial system marking.
    let mut roots: Vec<u32> = Vec::new();
    {
        let row = ctx.masks.as_ref().map(|m| m.initial.as_slice());
        for_satisfied_edges(
            &mut table,
            &gba.initial_transitions,
            row,
            init_mid,
            |_, trans| {
                roots.push(g.intern(init_mid, trans.successor, &ctx.acc));
            },
        )?;
    }
    roots.sort_unstable();
    roots.dedup();

    let mut stack: Vec<DfsFrame> = Vec::new();
    // One adaptive probe (deadline + memory) — same rationale as the BFS
    // builder: item caps alone do not bound bytes.
    let mut probe = crate::memory::explorer_probe(deadline);
    let mut stats = ProductStats::new("DFS+POR");

    for &root in &roots {
        if g.expanded[root as usize] {
            continue;
        }
        if !expand_product_node(&ctx, &mut g, &mut buf, &mut table, root)? {
            return Ok(None);
        }
        stats.expansions += 1;
        g.on_stack[root as usize] = true;
        stack.push(DfsFrame {
            pid: root,
            cursor: 0,
        });

        while let Some(&frame) = stack.last() {
            if probe.over_budget() {
                return Ok(None);
            }
            stats.product_states = g.ids.len();
            stats.markings = table.ids.len();

            let pid = frame.pid;
            if frame.cursor < g.adj[pid as usize].len() {
                let v = g.adj[pid as usize][frame.cursor];
                stack.last_mut().expect("stack non-empty").cursor += 1;
                if !g.expanded[v as usize] {
                    if !expand_product_node(&ctx, &mut g, &mut buf, &mut table, v)? {
                        return Ok(None);
                    }
                    stats.expansions += 1;
                    g.on_stack[v as usize] = true;
                    stack.push(DfsFrame { pid: v, cursor: 0 });
                }
            } else {
                g.on_stack[pid as usize] = false;
                stack.pop();
            }
        }
    }

    if let Some(out) = product_size_out {
        *out = g.ids.len();
    }

    // TEST-ONLY: snapshot the built reduced product for the C3 structural
    // verifier. Done before `find_accepting_scc` (read-only) so the gate sees
    // exactly the graph the verdict is computed from.
    #[cfg(test)]
    if let Some(cap) = c3.capture.as_deref_mut() {
        cap.adj = g.adj.clone();
        cap.fully_expanded = g.fully_expanded.clone();
        cap.accept = g.accept.clone();
        cap.edge_succ = g.edge_succ.clone();
        cap.edge_words = g.edge_words.clone();
        cap.num_accept = ctx.acc.num_accept;
    }

    // Flatten the adjacency to CSR (and free the nested rows) before the
    // long-lived Tarjan/SCC phase (audit S5). Done after the C3 capture so
    // the snapshot sees the graph exactly as built. `None` (u32 edge-offset
    // overflow) declines the run, like the state/deadline budgets above.
    let Some(csr) = flatten_adjacency(&mut g.adj) else {
        return Ok(None);
    };
    let accepting = find_accepting_scc(
        &csr,
        &g.accept,
        &g.edge_succ,
        &g.edge_words,
        ctx.acc.num_accept,
    );
    if let Some(scc) = accepting.as_deref() {
        if lasso_dump_enabled() {
            dump_accepting_lasso(&g, &csr, &mut table, net, &roots, scc);
        }
    }
    Ok(Some(accepting.is_some()))
}

/// Test-only POR product run that also reports the number of product states
/// built — used to assert that per-Büchi-state visibility prunes strictly
/// more than the static whole-formula set while preserving the verdict.
/// The size is meaningful only when the verdict is conclusive (`Some(_)`).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn on_the_fly_product_emptiness_por_with_size(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    por: &PorContext,
    max_system_states: usize,
    deadline: Option<Instant>,
) -> Result<(Option<bool>, usize), PnmlError> {
    let mut size = 0usize;
    let verdict = on_the_fly_dfs_impl(
        gba,
        net,
        reduced,
        original_net,
        atoms,
        Some(por),
        max_system_states,
        PRODUCT_STATE_LIMIT,
        deadline,
        memo_disabled_by_env(),
        Some(&mut size),
    )?;
    Ok((verdict, size))
}

/// TEST-ONLY entry for the C3 cycle-proviso differential gate.
///
/// Builds the DFS+POR reduced product with the C3 cycle proviso optionally
/// SUPPRESSED (`c3_disabled = true` → the unsound mutant) and snapshots the
/// resulting product graph into `capture` for the per-SCC structural verifier.
/// Returns the accepting-cycle verdict (`Some(true)` ⇔ ¬φ has an accepting run
/// ⇔ φ is violated), or `None` on budget/deadline overflow.
///
/// Production never calls this; `on_the_fly_dfs_impl` always passes
/// `c3_disabled = false` and `capture = None`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn on_the_fly_product_emptiness_c3_gate(
    gba: &Gba,
    net: &PetriNet,
    reduced: &ReducedNet,
    original_net: &PetriNet,
    atoms: &[ResolvedPredicate],
    por: &PorContext,
    max_system_states: usize,
    deadline: Option<Instant>,
    c3_disabled: bool,
    capture: Option<&mut ProductCapture>,
) -> Result<Option<bool>, PnmlError> {
    on_the_fly_dfs_impl_inner(
        gba,
        net,
        reduced,
        original_net,
        atoms,
        Some(por),
        max_system_states,
        PRODUCT_STATE_LIMIT,
        deadline,
        memo_disabled_by_env(),
        None,
        C3Options {
            c3_disabled,
            capture,
        },
    )
}

#[cfg(test)]
#[path = "on_the_fly_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ltl_c3_proviso_differential_proptest.rs"]
mod ltl_c3_proviso_differential_proptest;
