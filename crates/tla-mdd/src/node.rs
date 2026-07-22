// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! The MDD node store: a level-per-place multi-valued decision diagram with a
//! unique table and an apply cache.
//!
//! # Representation
//!
//! A node lives at a **level** `0..num_levels`. Levels map one-to-one to the
//! net's places in a fixed variable order: level `l` is place `l`, top
//! (level 0) to bottom (level `num_levels-1`). Below the last place level
//! sits the terminal level.
//!
//! A non-terminal node at level `l` has exactly `bounds[l] + 1` outgoing
//! edges, one per possible token count `v ∈ 0..=bounds[l]` for that place.
//! Edge `v` points to a child node at level `l + 1` (its *successor* level).
//! This is the defining MDD property: **k edges for k domain values, not
//! bit-blasted Boolean**. A place that can hold `0..=B` tokens contributes a
//! single level of out-degree `B+1`, regardless of how large `B` is — which
//! is exactly why an MDD is more compact than a BDD (which would spend
//! `ceil(log2(B+1))` Boolean levels on the same place).
//!
//! There are two terminals:
//! - [`MddRef::ONE`] — the accepting/true terminal: a path reaching it spells
//!   out a marking that is *in* the represented set.
//! - [`MddRef::ZERO`] — the rejecting/false terminal: paths to it are *not*
//!   in the set.
//!
//! # Reduction rules
//!
//! The store maintains a **fully-reduced ordered MDD** (a ROMDD):
//!
//! 1. **Unique table.** Two nodes at the same level with identical child-edge
//!    vectors are merged into one (`get_node` canonicalizes). So the diagram
//!    is *canonical*: equal sets share the same root `MddRef`.
//! 2. **Redundant-node suppression.** A node all of whose edges point to the
//!    *same* child is redundant (the place's value does not matter); it is
//!    skipped and replaced by that child. Combined with rule 1 this is the
//!    standard ROMDD reduction.
//!
//! Because redundant nodes are skipped, an edge can "long-jump" across
//! several levels — a child at level `c` reached from a parent at level `l`
//! may have `c > l + 1`. Every algorithm here keys on the explicit level
//! stored in each node and treats a level gap as "all skipped places are
//! unconstrained (any value 0..=bound)". The terminal carries the sentinel
//! level [`TERMINAL_LEVEL`].
//!
//! # Soundness
//!
//! This store is purely combinatorial; the only soundness obligation is that
//! the reduction rules preserve the represented marking *set* exactly. They
//! do (standard ROMDD theory), and the crate's differential proptest battery
//! pins the reachable-set count against the explicit BFS oracle on thousands
//! of random nets, 0 disagreements. The kernel is **gate-only** (a new,
//! not-yet-production engine) until that battery and the BDD cross-check have
//! run in CI for long enough to trust.

use std::collections::HashMap;

/// Fraction of effective machine/confinement memory the MDD node store may
/// occupy. ~0.25 reproduces the historic fixed 2 GiB cap on an 8 GiB machine
/// while scaling proportionally to the host / MCC confinement. Derived from
/// `effective_total_bytes` (a STABLE machine property) — not
/// `effective_available` — so the cap does not shrink as the very store it
/// bounds consumes free memory.
const STORE_BYTES_FRACTION: f64 = 0.25;
/// Fallback when memory detection fails (the historic fixed value).
const FALLBACK_MAX_STORE_BYTES: usize = 2 << 30; // 2 GiB
const MIN_MAX_STORE_BYTES: usize = 512 << 20; // 512 MiB
const MAX_MAX_STORE_BYTES: usize = 32_usize << 30; // 32 GiB

/// Effective memory charged per interior MDD node incl. its unique-table entry,
/// amortized apply-cache slot, and count metadata (audit 2026-07-02). Sized so
/// the derived node cap reproduces the historic 8M-node cap at the 2 GiB store
/// budget (2 GiB / 256 ≈ 8M).
const MDD_NODE_MEMORY_COST: usize = 256;
const MIN_MAX_INTERIOR_NODES: usize = 250_000;
/// Hard ceiling on the interior-node cap, kept well under the u32 `MddRef`
/// id-space (~4.29 B). The node arena indexes nodes by `u32`, so
/// [`MddStore::get_node`] would overflow `u32` at ~4.29 B nodes. Clamping the
/// DERIVED cap here GUARANTEES the fixpoint drivers (which decline gracefully
/// once `interior_node_count() > max_interior_nodes()`) always fire before the
/// arena could overflow — so any future raise of the store-byte budget can
/// never turn the arena backstop into a reachable crash. ~1.07 B ≈ 4× the
/// 32 GiB/256 store-derived max, a comfortable margin under `u32::MAX`.
const MAX_INTERIOR_NODES_CEILING: usize = 1 << 30;

/// Byte-honest store cap, DERIVED from effective machine/confinement memory
/// instead of a fixed magic constant (audit 2026-07-02): per-node bytes scale
/// with the level's DOMAIN SIZE (a huge-bound place means millions of edges per
/// node, stored twice), so an item-count cap alone admits multi-GB stores.
/// Every fixpoint guard that checks the node cap also checks
/// [`MddStore::approx_store_bytes`] against this. Fail-closed: exceeding it
/// declines the run, never a verdict.
#[must_use]
pub fn max_store_bytes() -> usize {
    tla_resource::platform::effective_total_bytes()
        .map(|total| (total as f64 * STORE_BYTES_FRACTION) as usize)
        .unwrap_or(FALLBACK_MAX_STORE_BYTES)
        .clamp(MIN_MAX_STORE_BYTES, MAX_MAX_STORE_BYTES)
}

/// Interior-node cap, DERIVED from [`max_store_bytes`] and the per-node memory
/// cost — the item-count companion to the byte cap, adaptive to the machine.
#[must_use]
pub fn max_interior_nodes() -> usize {
    (max_store_bytes() / MDD_NODE_MEMORY_COST)
        .clamp(MIN_MAX_INTERIOR_NODES, MAX_INTERIOR_NODES_CEILING)
}

/// Cooperative-abort marker (audit 2026-07-11). [`MddStore::get_node`] raises it
/// via `panic_any` when the armed [`MddStore::set_abort_probe`] probe trips
/// mid-image-round; [`catch_mdd_abort`] folds it back into a fail-closed
/// decline. Sound because the panic fires BEFORE any store mutation, so a
/// caught store is left canonical (and the fixpoint discards it regardless).
/// Mirrors tla-bdd's `BddAbort`.
pub(crate) struct MddAbort;

/// Run `f`, folding an [`MddAbort`] unwind into `None` (fail-closed) and
/// resuming any other panic. The engines map `None` to their
/// [`CountError::ResourceCap`](crate::reach::CountError) decline, so the
/// in-operation probe surfaces exactly like the boundary deadline check.
///
/// `AssertUnwindSafe` is justified: the only state the closure mutates is the
/// store it creates internally, which is thrown away on the abort path — no
/// caller-visible value is left in a broken state.
pub(crate) fn catch_mdd_abort<T>(f: impl FnOnce() -> T) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Some(value),
        Err(payload) if payload.is::<MddAbort>() => None,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Sentinel level for the two terminal nodes. Larger than any real place
/// level, so "deeper than every place" ordering comparisons work directly.
pub const TERMINAL_LEVEL: u32 = u32::MAX;

/// A handle to an MDD node inside a [`MddStore`].
///
/// Indices `0` and `1` are reserved for the [`MddRef::ZERO`] and
/// [`MddRef::ONE`] terminals; all other indices are interior nodes stored in
/// the [`MddStore`]'s internal node arena. A `MddRef` is only meaningful
/// relative to the store that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MddRef(pub(crate) u32);

impl MddRef {
    /// The false / rejecting terminal (empty contribution to the set).
    pub const ZERO: MddRef = MddRef(0);
    /// The true / accepting terminal (a completed in-set path ends here).
    pub const ONE: MddRef = MddRef(1);

    /// True iff this is one of the two terminals.
    #[inline]
    #[must_use]
    pub fn is_terminal(self) -> bool {
        self.0 <= 1
    }

    /// True iff this is the accepting (`ONE`) terminal.
    #[inline]
    #[must_use]
    pub fn is_one(self) -> bool {
        self.0 == 1
    }

    /// True iff this is the rejecting (`ZERO`) terminal.
    #[inline]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
}

/// An interior MDD node: a level plus one child edge per domain value.
///
/// `children.len() == bounds[level] + 1`. `children[v]` is the node reached
/// when this place holds exactly `v` tokens.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Node {
    pub(crate) level: u32,
    pub(crate) children: Vec<MddRef>,
}

/// The MDD node store: arena of interior nodes + unique table.
///
/// Owns every node referenced by the `MddRef`s it hands out. The unique table
/// (`unique`) maps a node's structural key to its canonical `MddRef`, giving
/// the merge / canonicity rule. Per-place domain sizes come from `bounds`.
#[derive(Debug)]
pub struct MddStore {
    /// Per-level (= per-place) token upper bound. `bounds.len()` is the number
    /// of place levels; level `l` has domain `0..=bounds[l]`.
    pub(crate) bounds: Vec<u64>,
    /// Node arena, indexed by `MddRef.0`. Slots `0` and `1` are placeholder
    /// terminal entries that are never read structurally (terminals are
    /// detected via [`MddRef::is_terminal`]).
    pub(crate) nodes: Vec<Node>,
    /// Unique table: structural key → canonical interior `MddRef`.
    unique: HashMap<Node, MddRef>,
    /// LIVE child edges across all interned interior nodes, counted ONCE per
    /// node (audit 2026-07-02): the byte-honest size driver, since per-node
    /// bytes scale with the level's domain size (a huge-bound place can have
    /// millions of edges per node) while the engines' `max_interior_nodes`
    /// caps count only items. Feeds [`Self::approx_store_bytes`]. Decremented by
    /// [`Self::gc`] when a node is swept, so the byte cap is LIVE not cumulative.
    child_edge_count: usize,
    /// LIVE interior node count (excludes the two terminals and any swept
    /// tombstone slot). Equals `nodes.len() - 2` for a store that never GCs.
    live_interior: usize,
    /// Freed interior slot indices, recycled by [`Self::get_node`] before the
    /// arena grows. The GC is NON-MOVING: a surviving node keeps its `MddRef`
    /// (stable ids), so caller-held roots and the engines' O(1) root-equality
    /// convergence stay valid across a collection.
    free_list: Vec<u32>,
    /// `live_interior` at the last [`Self::gc`], for the adaptive collection
    /// trigger ([`Self::should_collect`]).
    last_gc_live: usize,
    /// PERSISTENT apply caches, one slot per binary set op (indexed by
    /// [`ApplyOp`] as `usize`): kept warm across calls instead of a fresh map
    /// per call, so consecutive `union(next, img)` / `intersect` / `difference`
    /// calls reuse shared sub-results. Each op gets its OWN slot because the
    /// result differs by op for the same operand pair. Each is SCRUBBED (not a
    /// mark root) on every [`Self::gc`] — entries referencing a freed node are
    /// dropped before the id can be reused — and cleared when it exceeds
    /// [`Self::apply_cache_cap`], so none can grow unbounded or return a stale
    /// node. Stable ids keep every surviving entry valid across a collection.
    apply_caches: [HashMap<(MddRef, MddRef), MddRef>; ApplyOp::COUNT],
    /// Cooperative in-operation abort probe (audit 2026-07-11): an adaptive
    /// [`MemoryProbe`](tla_resource::MemoryProbe) carrying a live process-
    /// footprint ceiling, the collective free-memory floor, and an optional
    /// wall-clock deadline, ticked per FRESH interior-node interning in
    /// [`Self::get_node`]. Before this the store had ONLY the boundary-level
    /// `max_interior_nodes` cap + per-round deadline poll — a single monster
    /// `relprod`/`saturate` image round (millions of `get_node` calls between
    /// two safepoints) could overrun the deadline or the footprint ceiling
    /// without ever returning to a boundary. Mirrors the tla-bdd `mk` probe.
    /// `None` (the default) is byte-identical to the pre-probe engine.
    abort_probe: Option<tla_resource::MemoryProbe>,
}

/// A binary set operation with its own persistent apply cache. The discriminant
/// indexes [`MddStore::apply_caches`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ApplyOp {
    Union = 0,
    Intersect = 1,
    Difference = 2,
}

impl ApplyOp {
    /// Number of distinct ops (the apply-cache array length).
    pub(crate) const COUNT: usize = 3;
}

/// Bytes charged per persistent-apply-cache entry (2 key `MddRef`s + 1 value
/// `MddRef` + hashbrown control/overhead), for deriving the cache size cap.
const APPLY_CACHE_ENTRY_BYTES: usize = 48;

impl MddStore {
    /// Create an empty store for a net with the given per-place token bounds.
    ///
    /// `bounds[l]` is the maximum token count place `l` may hold (its domain
    /// is `0..=bounds[l]`, i.e. `bounds[l] + 1` edges at level `l`).
    #[must_use]
    pub fn new(bounds: Vec<u64>) -> Self {
        // Reserve the two terminal slots with dummy nodes (never inspected).
        let terminal_dummy = Node {
            level: TERMINAL_LEVEL,
            children: Vec::new(),
        };
        Self {
            bounds,
            nodes: vec![terminal_dummy.clone(), terminal_dummy],
            unique: HashMap::new(),
            child_edge_count: 0,
            live_interior: 0,
            free_list: Vec::new(),
            last_gc_live: 0,
            apply_caches: std::array::from_fn(|_| HashMap::new()),
            abort_probe: None,
        }
    }

    /// Arm the cooperative in-operation abort probe for the fixpoint that owns
    /// this store: a live footprint ceiling + collective free-memory floor
    /// (adaptive to the machine / MCC confinement) plus the optional wall-clock
    /// `deadline`, checked per fresh interior-node interning in
    /// [`Self::get_node`]. Installing a probe (vs the default `None`) is the
    /// ONLY behavioral change — an armed run backs off within one image round
    /// under real memory pressure or a passed deadline instead of only at the
    /// next round/GC safepoint. Fail-closed: the probe trip unwinds
    /// [`MddAbort`] out of the fixpoint, which [`crate::catch_mdd_abort`] folds
    /// into a `ResourceCap` decline. `None` disarms (byte-identical old path).
    pub(crate) fn set_abort_probe(&mut self, deadline: Option<std::time::Instant>) {
        self.abort_probe = Some(tla_resource::MemoryProbe::new(
            tla_resource::MemoryBudget::symbolic_explorer(),
            deadline,
        ));
    }

    /// Max entries a persistent apply cache may hold before it is cleared —
    /// derived so each cache's bytes stay ~a quarter of the store byte budget
    /// (adaptive to the machine, no fixed magic). The cache is rebuildable, so
    /// clearing it on overflow is a pure size bound, never a decline.
    #[inline]
    pub(crate) fn apply_cache_cap(&self) -> usize {
        (max_store_bytes() / 4 / APPLY_CACHE_ENTRY_BYTES).max(1)
    }

    /// Take the persistent apply cache for `op` out for the duration of one
    /// call (so the recursion can borrow the store mutably for `get_node`),
    /// clearing it first if it has exceeded its size cap.
    #[inline]
    pub(crate) fn take_apply_cache(&mut self, op: ApplyOp) -> HashMap<(MddRef, MddRef), MddRef> {
        let cap = self.apply_cache_cap();
        let slot = &mut self.apply_caches[op as usize];
        if slot.len() > cap {
            slot.clear();
        }
        std::mem::take(slot)
    }

    /// Restore the persistent apply cache for `op` after a call.
    #[inline]
    pub(crate) fn put_apply_cache(
        &mut self,
        op: ApplyOp,
        cache: HashMap<(MddRef, MddRef), MddRef>,
    ) {
        self.apply_caches[op as usize] = cache;
    }

    /// Approximate resident bytes of the store: each interned node holds its
    /// children Vec TWICE (once in the arena, once cloned as the unique-table
    /// key) at 4 bytes per edge, plus per-node struct/entry overhead. Used by
    /// the engines' fixpoint guards as the byte-honest companion to their
    /// interior-node-count caps.
    #[must_use]
    pub fn approx_store_bytes(&self) -> usize {
        // 2 copies × 4 B/edge, + Node struct (level + Vec header ≈ 32 B) × 2
        // + unique-table entry overhead ≈ 48 B per node.
        self.child_edge_count
            .saturating_mul(2 * std::mem::size_of::<MddRef>())
            .saturating_add(self.nodes.len().saturating_mul(112))
    }

    /// Number of place levels (= number of places in the net).
    #[inline]
    #[must_use]
    pub fn num_levels(&self) -> usize {
        self.bounds.len()
    }

    /// Domain size (number of edges) for a level.
    #[inline]
    #[must_use]
    pub fn domain_size(&self, level: u32) -> usize {
        (self.bounds[level as usize] + 1) as usize
    }

    /// The level a node sits at (`TERMINAL_LEVEL` for the terminals).
    #[inline]
    #[must_use]
    pub(crate) fn level_of(&self, node: MddRef) -> u32 {
        if node.is_terminal() {
            TERMINAL_LEVEL
        } else {
            self.nodes[node.0 as usize].level
        }
    }

    /// Child of an interior node along edge `value`.
    #[inline]
    pub(crate) fn child(&self, node: MddRef, value: u64) -> MddRef {
        debug_assert!(!node.is_terminal());
        self.nodes[node.0 as usize].children[value as usize]
    }

    /// Canonicalize and intern an interior node at `level` with the given
    /// child edges, applying the two ROMDD reduction rules.
    ///
    /// # Panics
    /// Debug-asserts that `children.len()` matches the level's domain size and
    /// that `level` is a real place level.
    pub(crate) fn get_node(&mut self, level: u32, children: Vec<MddRef>) -> MddRef {
        debug_assert!((level as usize) < self.bounds.len(), "level out of range");
        debug_assert_eq!(
            children.len(),
            self.domain_size(level),
            "child-edge count must equal the level's domain size"
        );

        // Reduction rule 2: redundant-node suppression. If every edge points
        // to the same child, the node carries no information — return the
        // child directly. (`split_first` gives the first edge; `all` checks
        // the rest equal it.)
        if let Some((first, rest)) = children.split_first() {
            if rest.iter().all(|c| c == first) {
                return *first;
            }
        }

        let key = Node { level, children };
        // Reduction rule 1: unique-table merge.
        if let Some(&existing) = self.unique.get(&key) {
            return existing;
        }
        // Cooperative in-operation abort (audit 2026-07-11): ONLY fresh
        // interior insertions are charged (a unique-table hit above is free),
        // and the panic fires BEFORE the mutations below so the store stays
        // canonical whether or not the unwind is caught by
        // [`crate::catch_mdd_abort`]. The probe self-amortizes its clock/syscall
        // cadence, so the per-node check is O(1) amortized. Mirrors tla-bdd `mk`.
        if let Some(probe) = &mut self.abort_probe {
            if probe.over_budget() {
                std::panic::panic_any(MddAbort);
            }
        }
        // Test hook: force the cooperative abort on the first fresh interior
        // node of an armed store, so the forced-abort differential test
        // exercises the panic → catch_mdd_abort → ResourceCap decline path
        // deterministically (independent of real footprint / wall-clock).
        #[cfg(test)]
        if self.abort_probe.is_some() && ABORT_STRESS.with(std::cell::Cell::get) {
            std::panic::panic_any(MddAbort);
        }
        self.child_edge_count = self.child_edge_count.saturating_add(key.children.len());
        self.live_interior += 1;
        // Non-moving GC: reuse a freed slot (stable ids) before growing the arena.
        let node_ref = if let Some(idx) = self.free_list.pop() {
            self.nodes[idx as usize] = key.clone();
            MddRef(idx)
        } else {
            // Defense-in-depth backstop. Unreachable in practice: `max_interior_nodes`
            // is clamped to `MAX_INTERIOR_NODES_CEILING` (~1.07 B, ¼ of the u32 arena
            // id-space), and every fixpoint driver declines gracefully once
            // `interior_node_count()` crosses that cap — so the arena high-water is
            // bounded far below `u32::MAX` before it could reach here.
            let idx = u32::try_from(self.nodes.len())
                .expect("MDD node arena overflowed u32 (node cap should have declined first)");
            self.nodes.push(key.clone());
            MddRef(idx)
        };
        self.unique.insert(key, node_ref);
        node_ref
    }

    /// Number of live interior nodes (excludes the two terminals).
    ///
    /// Diagnostic: this is the MDD's node count, the figure that is small on
    /// counter / conserved nets where the BDD lane's node count explodes.
    #[must_use]
    pub fn interior_node_count(&self) -> usize {
        self.live_interior
    }

    /// Number of entries in the unique table (== live interior nodes when the
    /// store is consistent; used by GC invariant tests).
    #[cfg(test)]
    pub(crate) fn unique_len(&self) -> usize {
        self.unique.len()
    }

    /// Non-moving mark-sweep garbage collection: free every interior node NOT
    /// reachable from `roots` (or the terminals), recycling its slot.
    ///
    /// Ids are STABLE: a surviving node keeps its `MddRef`, so caller-held roots
    /// and the engines' O(1) root-equality convergence (`next == reach`) remain
    /// valid across a collection, and no edge vector / unique key is rewritten.
    ///
    /// # Soundness contract
    ///
    /// `roots` MUST list every `MddRef` the caller — and any live cache — will
    /// use after this call. Call ONLY at a quiescent safepoint (e.g. the top of
    /// a fixpoint round), NEVER from inside [`Self::get_node`] or an apply
    /// recursion whose working refs are not in `roots`. Under-supplying roots
    /// frees live nodes ⇒ a dangling/aliased `MddRef`. Terminals are always
    /// live. Freeing only unreachable nodes preserves the represented set of
    /// every root exactly, so every count/verdict is unchanged.
    pub fn gc(&mut self, roots: &[MddRef]) -> GcStats {
        // ---- Mark: DFS over `children` from each non-terminal root. ----
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<u32> = Vec::new();
        for &r in roots {
            if !r.is_terminal() && !seen[r.0 as usize] {
                seen[r.0 as usize] = true;
                stack.push(r.0);
            }
        }
        while let Some(idx) = stack.pop() {
            let n = self.nodes[idx as usize].children.len();
            for i in 0..n {
                let c = self.nodes[idx as usize].children[i];
                if !c.is_terminal() && !seen[c.0 as usize] {
                    seen[c.0 as usize] = true;
                    stack.push(c.0);
                }
            }
        }

        // ---- Sweep: free unmarked, non-tombstone interior slots. ----
        // Interior nodes always have >= 2 children (redundant-node suppression
        // in `get_node` never creates a node whose edges are all equal, so a
        // 1-edge node is impossible), while a tombstone (already-freed slot) has
        // an EMPTY children vec — so `children.is_empty()` distinguishes them.
        let mut freed = 0usize;
        for idx in 2..self.nodes.len() {
            if seen[idx] || self.nodes[idx].children.is_empty() {
                continue;
            }
            let dead = std::mem::replace(
                &mut self.nodes[idx],
                Node {
                    level: TERMINAL_LEVEL,
                    children: Vec::new(),
                },
            );
            self.child_edge_count = self.child_edge_count.saturating_sub(dead.children.len());
            self.live_interior -= 1;
            self.unique.remove(&dead);
            self.free_list.push(idx as u32);
            freed += 1;
        }

        // Scrub every persistent apply cache with the SAME mark bitset: drop any
        // entry whose operands or result reference a freed (unmarked, non-
        // terminal) node, BEFORE those ids can be recycled by get_node — so a
        // survivor stays valid (stable ids) but no stale entry can resurrect a
        // freed/reused node. The caches are NOT mark roots, so cached-but-
        // unreachable intermediates are correctly collected.
        let live = |r: MddRef| r.is_terminal() || seen[r.0 as usize];
        for cache in &mut self.apply_caches {
            cache.retain(|&(a, b), &mut v| live(a) && live(b) && live(v));
        }

        self.last_gc_live = self.live_interior;
        GcStats {
            freed,
            live: self.live_interior,
        }
    }

    /// Adaptive collection trigger: whether a safepoint should [`Self::gc`] now.
    ///
    /// Derived, not a fixed cadence: fires once the live node count reaches half
    /// the machine-derived [`max_interior_nodes`] cap AND has grown by another
    /// quarter-cap since the last collection — so a collection that frees little
    /// (a genuinely large live set) does not re-fire every round, yet the store
    /// always collects before the cap would trip and decline the run.
    #[must_use]
    pub fn should_collect(&self) -> bool {
        // Test hook: force a collection on every non-empty store so the
        // gc-forcing differential tests exercise driver root-supply end to end.
        #[cfg(test)]
        if GC_STRESS.with(std::cell::Cell::get) {
            return self.live_interior > 0;
        }
        let cap = max_interior_nodes();
        self.live_interior >= cap / 2
            && self.live_interior >= self.last_gc_live.saturating_add(cap / 4)
    }
}

#[cfg(test)]
thread_local! {
    /// Per-thread flag: when set, [`MddStore::should_collect`] fires on every
    /// non-empty store. Lets a differential test force a collection every
    /// fixpoint round and confirm a driver supplies the COMPLETE root set.
    static GC_STRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only: toggle the [`should_collect`](MddStore::should_collect) stress flag.
#[cfg(test)]
pub(crate) fn set_gc_stress(on: bool) {
    GC_STRESS.with(|c| c.set(on));
}

#[cfg(test)]
thread_local! {
    /// Per-thread flag: when set, [`MddStore::get_node`] raises [`MddAbort`] on
    /// the first fresh interior interning of an ARMED store (one with a probe
    /// installed via [`MddStore::set_abort_probe`]). Lets a differential test
    /// force the cooperative-abort path every run and confirm it folds into a
    /// `ResourceCap` decline with no corruption of the unarmed path.
    static ABORT_STRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only: toggle the [`get_node`](MddStore::get_node) forced-abort stress flag.
#[cfg(test)]
pub(crate) fn set_abort_stress(on: bool) {
    ABORT_STRESS.with(|c| c.set(on));
}

/// Outcome of a [`MddStore::gc`] pass.
#[derive(Debug, Clone, Copy)]
pub struct GcStats {
    /// Interior nodes freed this collection.
    pub freed: usize,
    /// Live interior nodes remaining after the collection.
    pub live: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_cap_stays_below_the_u32_arena_id_space() {
        // Posture invariant: the DERIVED interior-node cap must stay strictly
        // below the u32 `MddRef` id-space so a fixpoint driver always declines
        // (graceful ResourceCap) before `get_node` could overflow the arena and
        // hit the panic backstop. This must hold for ANY machine memory, so
        // check the hard ceiling that clamps the live derivation.
        assert!(
            MAX_INTERIOR_NODES_CEILING < u32::MAX as usize,
            "node cap ceiling must be under the u32 arena id-space"
        );
        // A wide margin — the ceiling (2^30) is a quarter of the 2^32 id-space.
        assert!(
            MAX_INTERIOR_NODES_CEILING <= 1usize << 31,
            "keep a >=2x arena margin"
        );
        // The live cap (whatever this machine derives) never exceeds the ceiling.
        assert!(max_interior_nodes() <= MAX_INTERIOR_NODES_CEILING);
        assert!(max_interior_nodes() >= MIN_MAX_INTERIOR_NODES);
    }

    #[test]
    fn terminals_are_terminal() {
        assert!(MddRef::ZERO.is_terminal());
        assert!(MddRef::ONE.is_terminal());
        assert!(MddRef::ONE.is_one());
        assert!(MddRef::ZERO.is_zero());
        assert!(!MddRef::ONE.is_zero());
    }

    // ── Garbage collection (non-moving mark-sweep) invariants ──────────────

    /// Build a store with two levels (domain {0,1} each) holding two DISJOINT
    /// structures rooted at `t1` and `t2`, returning `(store, t1, t2, b1, b2)`.
    fn two_disjoint_trees() -> (MddStore, MddRef, MddRef, MddRef, MddRef) {
        let mut s = MddStore::new(vec![1, 1]);
        let b1 = s.get_node(1, vec![MddRef::ZERO, MddRef::ONE]); // v=1 accepts
        let b2 = s.get_node(1, vec![MddRef::ONE, MddRef::ZERO]); // v=0 accepts
        let t1 = s.get_node(0, vec![b1, MddRef::ZERO]);
        let t2 = s.get_node(0, vec![b2, MddRef::ZERO]);
        (s, t1, t2, b1, b2)
    }

    #[test]
    fn gc_i1_frees_only_unreachable_and_preserves_root_set() {
        // I1 + count-preservation: gc(&[t1]) frees t2's subtree, keeps t1's, and
        // t1's represented marking set is byte-for-byte unchanged.
        let (mut s, t1, _t2, b1, _b2) = two_disjoint_trees();
        assert_eq!(s.interior_node_count(), 4);
        let before = s.count_markings(t1);

        let stats = s.gc(&[t1]);
        assert_eq!(stats.freed, 2, "t2 and b2 are unreachable from t1");
        assert_eq!(s.interior_node_count(), 2, "only b1 + t1 survive");
        assert_eq!(s.count_markings(t1), before, "root set unchanged by gc");
        // b1 (reachable via t1) is still a valid, resolvable node.
        assert_eq!(s.get_node(1, vec![MddRef::ZERO, MddRef::ONE]), b1);
    }

    #[test]
    fn gc_i2_ids_are_stable_across_collection() {
        // I2: a survivor keeps its MddRef; re-interning the identical structure
        // returns the SAME ref (canonicity + stable ids).
        let (mut s, t1, _t2, _b1, _b2) = two_disjoint_trees();
        s.gc(&[t1]);
        let b1 = s.get_node(1, vec![MddRef::ZERO, MddRef::ONE]);
        let t1_again = s.get_node(0, vec![b1, MddRef::ZERO]);
        assert_eq!(t1_again, t1, "surviving structure keeps its id after gc");
    }

    #[test]
    fn gc_i3_unique_table_has_no_dead_entries() {
        // I3: after gc, the unique table holds exactly the live interior nodes.
        let (mut s, t1, _t2, _b1, _b2) = two_disjoint_trees();
        s.gc(&[t1]);
        assert_eq!(s.unique_len(), s.interior_node_count());
    }

    #[test]
    fn gc_reclaims_slots_so_the_arena_does_not_grow() {
        // The flagship VALUE: GC makes the cap LIVE. After freeing nodes, the
        // recycled slots are reused so the arena high-water (`nodes.len()`) does
        // NOT grow when rebuilding an equal amount of structure — the run's
        // arena is bounded by the peak SIMULTANEOUSLY-live set, not cumulative.
        let (mut s, t1, _t2, _b1, _b2) = two_disjoint_trees();
        let high_water_before = s.nodes.len();
        s.gc(&[t1]); // frees b2 + t2 → 2 recycled slots
        assert_eq!(s.free_list.len(), 2);
        // Rebuild two fresh nodes: they reuse the recycled slots, not new arena.
        let x = s.get_node(1, vec![MddRef::ONE, MddRef::ZERO]);
        let _y = s.get_node(0, vec![x, MddRef::ONE]);
        assert_eq!(
            s.nodes.len(),
            high_water_before,
            "recycled slots must be reused before the arena grows"
        );
        assert!(s.free_list.is_empty(), "both recycled slots were reused");
    }

    #[test]
    fn persistent_union_cache_is_sound_across_gc() {
        let mut s = MddStore::new(vec![1, 1]);
        let a = s.get_node(1, vec![MddRef::ZERO, MddRef::ONE]);
        let sa = s.get_node(0, vec![a, MddRef::ZERO]);
        let b = s.get_node(1, vec![MddRef::ONE, MddRef::ZERO]);
        let sb = s.get_node(0, vec![b, MddRef::ZERO]);
        let u = s.union(sa, sb); // populates the persistent union cache
        let u_count = s.count_markings(u);

        // GC keeping operands + result live: the cache entry survives (all refs
        // live, stable ids), so a re-union is served soundly and identically.
        s.gc(&[sa, sb, u]);
        assert_eq!(
            s.union(sa, sb),
            u,
            "canonical union stable across gc + cache"
        );
        assert_eq!(s.count_markings(u), u_count);

        // GC freeing sb + u (keep only sa): the cache entry (sa,sb)->u references
        // now-freed nodes and MUST be scrubbed, so it can never resurrect a
        // freed/reused id. Rebuild sb's structure and re-union — correct fresh.
        s.gc(&[sa]);
        let b2 = s.get_node(1, vec![MddRef::ONE, MddRef::ZERO]);
        let sb2 = s.get_node(0, vec![b2, MddRef::ZERO]);
        let u3 = s.union(sa, sb2);
        assert_eq!(
            s.count_markings(u3),
            u_count,
            "re-union after the stale entry is scrubbed recomputes correctly"
        );
    }

    #[test]
    fn persistent_intersect_difference_caches_sound_across_gc() {
        let mut s = MddStore::new(vec![1, 1]);
        let a = s.get_node(1, vec![MddRef::ONE, MddRef::ONE]);
        let sa = s.get_node(0, vec![a, MddRef::ZERO]); // {(0,0),(0,1)}
        let b = s.get_node(1, vec![MddRef::ONE, MddRef::ZERO]);
        let sb = s.get_node(0, vec![b, MddRef::ZERO]); // {(0,0)}
        let inter = s.intersect(sa, sb); // {(0,0)}
        let diff = s.difference(sa, sb); // {(0,1)}
        let inter_count = s.count_markings(inter);
        let diff_count = s.count_markings(diff);

        // GC keeping everything live: each op's persistent cache entry survives
        // (stable ids), so a recompute is served canonically and identically.
        s.gc(&[sa, sb, inter, diff]);
        assert_eq!(
            s.intersect(sa, sb),
            inter,
            "intersect stable across gc + cache"
        );
        assert_eq!(
            s.difference(sa, sb),
            diff,
            "difference stable across gc + cache"
        );

        // GC freeing sb + results (keep sa): stale entries referencing freed
        // nodes MUST be scrubbed from BOTH caches. Rebuild sb and recompute.
        s.gc(&[sa]);
        let b2 = s.get_node(1, vec![MddRef::ONE, MddRef::ZERO]);
        let sb2 = s.get_node(0, vec![b2, MddRef::ZERO]);
        let re_i = s.intersect(sa, sb2);
        assert_eq!(
            s.count_markings(re_i),
            inter_count,
            "re-intersect after scrub recomputes correctly"
        );
        let re_d = s.difference(sa, sb2);
        assert_eq!(
            s.count_markings(re_d),
            diff_count,
            "re-difference after scrub recomputes correctly"
        );
    }

    #[test]
    fn gc_i5_terminals_are_pinned() {
        // I5: gc(&[]) frees ALL interior nodes; terminals stay valid.
        let (mut s, _t1, _t2, _b1, _b2) = two_disjoint_trees();
        let stats = s.gc(&[]);
        assert_eq!(stats.freed, 4);
        assert_eq!(s.interior_node_count(), 0);
        assert!(MddRef::ZERO.is_terminal() && MddRef::ONE.is_terminal());
        // The store is reusable after a full sweep: a fresh node still builds,
        // reusing a recycled slot, and the live accounting reflects exactly it.
        let rebuilt = s.get_node(1, vec![MddRef::ZERO, MddRef::ONE]);
        assert!(!rebuilt.is_terminal());
        assert_eq!(s.interior_node_count(), 1);
        assert_eq!(s.unique_len(), 1);
    }

    #[test]
    fn gc_i6_no_stale_key_resurrection_on_slot_reuse() {
        // I6: after a node is freed its unique key is purged, so re-interning a
        // structure into a recycled slot stays canonical and never resurrects a
        // stale id — even when TWO distinct structures both reuse freed slots.
        let (mut s, t1, _t2, _b1, _b2) = two_disjoint_trees();
        s.gc(&[t1]); // frees b2 + t2 → two slots on the free list

        // A DIFFERENT structure (level 0) reuses one freed slot.
        let d = s.get_node(0, vec![MddRef::ONE, MddRef::ZERO]);
        // The freed structure (level 1) re-interns FRESH into the other slot.
        let b2_new = s.get_node(1, vec![MddRef::ONE, MddRef::ZERO]);
        assert_ne!(
            b2_new, d,
            "distinct structures never alias, even after slot reuse"
        );
        // Canonicity holds: a second intern of the same structure merges.
        let b2_new2 = s.get_node(1, vec![MddRef::ONE, MddRef::ZERO]);
        assert_eq!(b2_new, b2_new2, "canonicity holds after slot reuse");
        assert_eq!(s.unique_len(), s.interior_node_count());
    }

    #[test]
    fn unique_table_merges_structurally_equal_nodes() {
        // Two place levels, both with bound 1 (domain {0,1}).
        let mut s = MddStore::new(vec![1, 1]);
        // A bottom-level node distinguishing v=0 (->ZERO) from v=1 (->ONE).
        let a = s.get_node(1, vec![MddRef::ZERO, MddRef::ONE]);
        let b = s.get_node(1, vec![MddRef::ZERO, MddRef::ONE]);
        assert_eq!(a, b, "structurally equal nodes must merge");
        assert_eq!(s.interior_node_count(), 1);
    }

    #[test]
    fn redundant_node_is_suppressed() {
        let mut s = MddStore::new(vec![2]);
        // All three edges to ONE: redundant, must collapse to ONE itself.
        let n = s.get_node(0, vec![MddRef::ONE, MddRef::ONE, MddRef::ONE]);
        assert_eq!(n, MddRef::ONE);
        assert_eq!(s.interior_node_count(), 0, "no interior node created");
    }

    #[test]
    fn non_redundant_node_is_kept() {
        let mut s = MddStore::new(vec![2]);
        let n = s.get_node(0, vec![MddRef::ZERO, MddRef::ONE, MddRef::ONE]);
        assert!(!n.is_terminal());
        assert_eq!(s.level_of(n), 0);
        assert_eq!(s.child(n, 0), MddRef::ZERO);
        assert_eq!(s.child(n, 1), MddRef::ONE);
        assert_eq!(s.child(n, 2), MddRef::ONE);
    }
}
