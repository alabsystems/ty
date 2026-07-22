// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! The SYMBOLIC reachability engine: a relational-product BFS fixpoint and a
//! node-level saturation fixpoint, both built on the [`crate::image`] symbolic
//! transition image.
//!
//! # Two symbolic strategies, one set representation
//!
//! Both strategies hold the reachable set `R` as a single canonical MDD and
//! grow it to a fixpoint. They differ only in *iteration order*:
//!
//! - [`MddNet::reachable_count_relprod`] — **breadth-first relational
//!   product.** Each round computes the image of the *whole* current set under
//!   every transition (via [`crate::image::transition_image`], which never
//!   enumerates markings) and unions the results in. The per-round cost is a
//!   function of the number of distinct MDD nodes, not `|frontier|`. This is
//!   the direct upgrade of the explicit [`crate::reach`] kernel.
//!
//! - [`MddNet::reachable_count_saturation`] — **node-level saturation.** Events
//!   are banded by their shallowest touched level
//!   ([`crate::image::shallowest_level`]) — the saturation `Top` in this
//!   top=level-0 orientation. A node at level `l` is saturated by firing every
//!   event banded at `l` to a *local fixpoint*, recursively saturating the
//!   deeper nodes first; the per-level remap then recurses downward over each
//!   event's whole support. Saturation is the classic Ciardo et al. iteration
//!   strategy; on conserved / counter / highly-concurrent nets it keeps the
//!   peak node count far below the breadth-first peak because work stays
//!   confined to a level band. A relational-product verification sweep after
//!   each pass makes the result an unconditionally sound fixpoint even under
//!   fully-reduced (skipped-level) MDDs.
//!
//! Both must produce the **exact same** reachable count as the explicit kernel
//! and the `tla-dd` BFS oracle — that is the soundness gate.
//!
//! # Soundness posture (ABSOLUTE, unchanged)
//!
//! Still **gate-only**: nothing here feeds a production verdict. Every path is
//! fail-closed — overflow past `u64::MAX` and the node budget / deadline return
//! [`CountError`], never a wrapped or partial count. The differential battery
//! (`tests/crosscheck_bfs.rs`) requires the relprod and saturation counts to
//! equal BOTH the explicit MDD kernel AND the BFS oracle, 0 disagreements.
//!
//! # LGPL warning
//!
//! Saturation here was written from the published algorithmic description
//! (Ciardo, Lüttgen, Siminiceanu, "Saturation: An Efficient Iteration Strategy
//! for Symbolic State-Space Generation", TACAS 2001). The LGPL Meddly / libDDD
//! saturation sources are license-incompatible with Apache-2.0 and were **not**
//! consulted.

use crate::image::{shallowest_level, transition_image, Imager};
use crate::node::{MddRef, MddStore};
use crate::reach::{CountError, MddNet, ReachResult};
use std::collections::HashMap;
use std::time::Instant;

/// Hard ceiling on live interior MDD nodes for the symbolic engine. If a net
/// needs more than this, the engine DECLINES (fail-closed) rather than risk an
/// OOM. Matches the explicit kernel's posture.
/// Interior-node cap, DERIVED from effective memory (was a fixed 8_000_000).
/// Adaptive to the machine/confinement via the shared node-store budget.
#[inline]
fn max_interior_nodes() -> usize {
    crate::node::max_interior_nodes()
}

/// Iteration backstop for the relational-product fixpoint. The fixpoint is
/// monotone (R only grows, bounded by `Π(bound+1)`), so it always converges;
/// this guards against a logic bug, not a semantic limit.
const MAX_ROUNDS: u32 = 100_000_000;

impl MddNet {
    /// Optional wall-clock deadline check helper. Returns `Err(ResourceCap)`
    /// when `deadline` has passed. `None` deadline never trips.
    #[inline]
    fn check_deadline(deadline: Option<Instant>) -> Result<(), CountError> {
        if let Some(d) = deadline {
            if Instant::now() >= d {
                return Err(CountError::ResourceCap("deadline exceeded".to_string()));
            }
        }
        Ok(())
    }

    /// Exact reachable-state count via the SYMBOLIC relational-product BFS
    /// fixpoint. `deadline` is an optional wall-clock cap (the engine declines
    /// rather than overrun it). Fail-closed on every error path.
    pub fn reachable_count_relprod(
        &self,
        deadline: Option<Instant>,
    ) -> Result<ReachResult, CountError> {
        let (store, reach, rounds, peak_nodes) = self.relprod_fixpoint(deadline)?;
        let state_count = self.count_or_err(&store, reach)?;
        Ok(ReachResult {
            state_count,
            iterations: rounds,
            interior_nodes: store.interior_node_count(),
            peak_interior_nodes: peak_nodes,
        })
    }

    /// Build the reachable set via the symbolic relational-product fixpoint,
    /// returning the store, the canonical root, and the round count.
    ///
    /// This is the SET-returning entry the metric extractor consumes: the
    /// relprod set is pinned EQUAL to the saturation and BFS reachable set by
    /// the differential battery, so reading metrics off it is reading them off
    /// the cross-checked reachable set. Fail-closed exactly like
    /// [`Self::reachable_count_relprod`] (which is now a thin wrapper around the
    /// shared [`Self::relprod_fixpoint`] core).
    pub(crate) fn build_reachable_relprod(
        &self,
        deadline: Option<Instant>,
    ) -> Result<(MddStore, MddRef, u32), CountError> {
        let (store, reach, rounds, _peak) = self.relprod_fixpoint(deadline)?;
        Ok((store, reach, rounds))
    }

    /// Shared relational-product fixpoint core. Returns `(store, root, rounds,
    /// peak_interior_nodes)`. Single source of truth so the count path and the
    /// metric path provably build the IDENTICAL reachable set.
    ///
    /// Wraps the [`Self::relprod_fixpoint_inner`] core in [`crate::catch_mdd_abort`]
    /// so the store's per-node cooperative abort probe (armed below) folds a
    /// mid-round footprint/deadline trip into the same `ResourceCap` decline the
    /// boundary deadline check produces — the run backs off within one image
    /// round, not only at the next safepoint.
    fn relprod_fixpoint(
        &self,
        deadline: Option<Instant>,
    ) -> Result<(MddStore, MddRef, u32, usize), CountError> {
        match crate::catch_mdd_abort(|| self.relprod_fixpoint_inner(deadline)) {
            Some(r) => r,
            None => Err(CountError::ResourceCap(
                "mdd relprod cooperative abort (footprint/deadline) hit mid-round".to_string(),
            )),
        }
    }

    fn relprod_fixpoint_inner(
        &self,
        deadline: Option<Instant>,
    ) -> Result<(MddStore, MddRef, u32, usize), CountError> {
        self.validate()?;
        let mut store = MddStore::new(self.bounds.clone());
        store.set_abort_probe(deadline);
        // Current variable order (`order[level] = place`); identity until the
        // first sift, so the non-sifting path is unchanged. Transitions are
        // permuted into level-space through it before the level-based image op.
        let mut order: Vec<usize> = (0..self.bounds.len()).collect();
        let init =
            crate::sift_runtime::singleton_ordered(&mut store, &self.initial_marking, &order);
        let mut reach = init;
        let mut peak_nodes = store.interior_node_count();
        // Sift at most once per run, when the diagram first approaches the cap.
        let mut sifted = false;
        let sift_watermark = max_interior_nodes() / 4 * 3;

        // Convergence is tested by CANONICAL ROOT EQUALITY (`next == reach`), an
        // O(1) `MddRef` (u32) comparison — not a per-round `BigUint` model-count.
        // The store is a fully-reduced ordered MDD, so equal sets share the same
        // root; the fixpoint is monotone (`next ⊇ reach`), so `next == reach` iff
        // the round added no new states. This is exact at ANY magnitude (it is
        // structural, never numeric) and avoids re-counting the whole diagram
        // every round. The final count is taken ONCE by the public wrappers.
        let mut rounds: u32 = 0;

        loop {
            rounds += 1;
            if rounds > MAX_ROUNDS {
                return Err(CountError::ResourceCap(
                    "round backstop exceeded".to_string(),
                ));
            }
            Self::check_deadline(deadline)?;

            // GC safepoint (non-moving): at the top of the round `reach` is the
            // ONLY live root and no MddRef-valued cache spans it (each apply
            // builds a fresh cache), so freeing unreachable nodes makes the caps
            // LIVE without disturbing the canonical root or `next == reach`.
            if store.should_collect() {
                store.gc(&[reach]);
            }

            // Sifting safepoint: shrink the diagram by dynamic variable
            // reordering when it grows large (this engine is node-count-bound —
            // no marking enumeration — so it benefits most). `reach` is the ONLY
            // live root ⇒ clean single-root remap; `order` composes the reorder.
            if crate::sift_runtime::want_sift(&store, self.bounds.len(), sifted, sift_watermark) {
                let (new_store, new_roots, chosen) = store.sift(&[reach]);
                reach = new_roots[0];
                store = new_store;
                order = crate::sift_runtime::compose_order(&order, &chosen);
                sifted = true;
            }

            // Symbolic image of the WHOLE current set under every transition,
            // unioned back in. No marking enumeration. Each transition is
            // permuted into the current level order so the level-based image op
            // applies each place's delta at the level that place now occupies.
            let mut next = reach;
            for t in &self.transitions {
                let t_level = crate::sift_runtime::transition_ordered(t, &order);
                let img = transition_image(&mut store, reach, &t_level);
                next = store.union(next, img);
                peak_nodes = peak_nodes.max(store.interior_node_count());
                if peak_nodes > max_interior_nodes()
                    || store.approx_store_bytes() > crate::node::max_store_bytes()
                {
                    return Err(CountError::ResourceCap(format!(
                        "interior node cap {} or store byte cap exceeded (relprod)",
                        max_interior_nodes()
                    )));
                }
            }

            // Canonical fixpoint witness: no new states ⇒ same canonical root.
            if next == reach {
                return Ok((store, reach, rounds, peak_nodes));
            }
            reach = next;
        }
    }

    /// Exact reachable-state count via NODE-LEVEL SATURATION.
    ///
    /// Events are banded by their SHALLOWEST touched level (the smallest place
    /// index the transition reads or writes). The engine saturates the set MDD:
    /// a node at
    /// level `l` is *saturated* when firing every event banded at `l`
    /// (recursively saturating any newly built deeper nodes first) reaches a
    /// local fixpoint. Because most P/T events touch only a few places, work
    /// stays confined to a level band and the peak node count is typically far
    /// below the breadth-first relprod peak — the saturation scalability win.
    ///
    /// # Soundness — verified global fixpoint
    ///
    /// Node-level saturation over a *fully-reduced* MDD has a known subtlety:
    /// when a redundant (skipped) level is reduced away, no node exists at that
    /// level for its banded events to fire at, which could leave the set short
    /// of the true fixpoint. Rather than reason about every reduction case, we
    /// make the result UNCONDITIONALLY sound by following each saturation pass
    /// with a breadth-first relational-product verification sweep (the symbolic
    /// transition image of the whole set under every transition): if that
    /// sweep adds any marking, the set was not a fixpoint, so we union it in and
    /// re-saturate. The loop terminates because the set is monotone and bounded
    /// by `Π(bound+1)`. In the common case the verification sweep adds nothing
    /// (saturation already reached the fixpoint) and costs one extra pass; in
    /// the worst case it degrades gracefully toward relprod — never to a wrong
    /// count. This is what lets saturation keep its peak-node win while
    /// guaranteeing the SAME count as the explicit kernel and the BFS oracle.
    ///
    /// Fail-closed on overflow / node budget / deadline, exactly like relprod.
    pub fn reachable_count_saturation(
        &self,
        deadline: Option<Instant>,
    ) -> Result<ReachResult, CountError> {
        let (store, reach, rounds, peak_nodes) = self.saturate_fixpoint(deadline)?;
        let state_count = self.count_or_err(&store, reach)?;
        Ok(ReachResult {
            state_count,
            iterations: rounds,
            interior_nodes: store.interior_node_count(),
            peak_interior_nodes: peak_nodes,
        })
    }

    /// Build the reachable set via NODE-LEVEL SATURATION, returning the store,
    /// the canonical root, and the round count.
    ///
    /// This is the SET-returning entry the SCALABLE metric path consumes: the
    /// saturated set is pinned EQUAL to the relprod and BFS reachable set by the
    /// differential battery (`tests/crosscheck_bfs.rs`), so reading the four
    /// StateSpace metrics off it is reading them off the cross-checked reachable
    /// set — but built by the saturation engine, which CONVERGES on the
    /// high-diameter conserved / counter nets where the breadth-first
    /// [`Self::build_reachable_relprod`] times out. Fail-closed exactly like
    /// [`Self::reachable_count_saturation`].
    pub(crate) fn build_reachable_saturation(
        &self,
        deadline: Option<Instant>,
    ) -> Result<(MddStore, MddRef, u32), CountError> {
        let (store, reach, rounds, _peak) = self.saturate_fixpoint(deadline)?;
        Ok((store, reach, rounds))
    }

    /// EXACT `UpperBounds`: for each query coefficient vector `c`, the reachable
    /// maximum `max_{M ∈ R} Σ_p c[p]·m[p]`. Builds the reachable set `R` once via
    /// saturation, then evaluates every query against it (the native MDD twin of
    /// `tla_dd::dispatch_upper_bounds_for_queries`). Each entry is `Some(bound)`
    /// or `None` if that query's max overflowed `i128` / its length mismatched
    /// (fail-closed — the caller leaves that tracker unresolved). The whole call
    /// returns `Err` only if `R` itself cannot be built (overflow / cap /
    /// deadline). Every `c[p]` must be the per-place coefficient (length =
    /// number of places).
    pub fn upper_bounds(
        &self,
        queries: &[Vec<i128>],
        deadline: Option<Instant>,
    ) -> Result<Vec<Option<i128>>, CountError> {
        let (store, reach, _rounds) = self.build_reachable_saturation(deadline)?;
        Ok(queries
            .iter()
            .map(|c| crate::metrics::max_weighted_sum_of(&store, reach, c))
            .collect())
    }

    /// FORMAL SOUNDNESS WITNESS for the saturated reachable set `R`.
    ///
    /// Discharges the two inductive obligations whose conjunction proves
    /// `R ⊇ reachable` — i.e. the saturation MISSES NO reachable marking, the
    /// soundness-critical direction of `R = reachable`:
    ///
    /// - **(I1) `init ∈ R`** — checked as `union(singleton(init), R) == R`
    ///   (`union(A, R) == R` iff `A ⊆ R`, exact since the store is a canonical
    ///   ROMDD so structural equality is `MddRef` equality);
    /// - **(I2) `∀t. image_t(R) ⊆ R`** — `R` is closed under every transition,
    ///   checked as `union(transition_image(R, t), R) == R`.
    ///
    /// (I1) ∧ (I2) make `R` an inductive invariant containing `init`, so by
    /// induction on firing sequences every reachable marking lies in `R`. This
    /// is exactly the proof obligation a Trust `SafetyCertificate` carries for a
    /// reachability fixpoint — a formal proof, independent of the (arena/
    /// unique-table) internals of how `R` was built. Returns `Ok(true)` iff both
    /// obligations hold; `Ok(false)` signals a soundness violation (a logic bug
    /// — must never occur); `Err` only on a build resource cap / overflow.
    ///
    /// Public so the cross-check battery (an external integration test) and the
    /// future `ty.cert/v1` SafetyCertificate emitter can discharge the proof.
    pub fn verify_saturation_inductive_fixpoint(
        &self,
        deadline: Option<Instant>,
    ) -> Result<bool, CountError> {
        let (mut store, reach, _rounds) = self.build_reachable_saturation(deadline)?;
        // (I1) init ⊆ R.
        let init = store.singleton(&self.initial_marking);
        if store.union(init, reach) != reach {
            return Ok(false);
        }
        // (I2) R closed under every transition: image_t(R) ⊆ R.
        for t in &self.transitions {
            let img = crate::image::transition_image(&mut store, reach, t);
            if store.union(img, reach) != reach {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Shared node-level saturation fixpoint core. Returns `(store, root,
    /// rounds, peak_interior_nodes)`. Single source of truth so the count path
    /// and the metric path provably build the IDENTICAL saturated reachable set
    /// — and so the metric path inherits saturation's convergence on
    /// high-diameter nets verbatim.
    fn saturate_fixpoint(
        &self,
        deadline: Option<Instant>,
    ) -> Result<(MddStore, MddRef, u32, usize), CountError> {
        match crate::catch_mdd_abort(|| self.saturate_fixpoint_inner(deadline)) {
            Some(r) => r,
            None => Err(CountError::ResourceCap(
                "mdd saturation cooperative abort (footprint/deadline) hit mid-round".to_string(),
            )),
        }
    }

    fn saturate_fixpoint_inner(
        &self,
        deadline: Option<Instant>,
    ) -> Result<(MddStore, MddRef, u32, usize), CountError> {
        self.validate()?;
        let mut store = MddStore::new(self.bounds.clone());
        store.set_abort_probe(deadline);
        // Current variable order + the net permuted into that level-space.
        // Identity until the first sift, so the non-sift path is byte-for-byte
        // unchanged. Sifting the saturation engine works by permuting the WHOLE
        // net (bounds/init/transitions) into level-space and running the
        // UNCHANGED saturator on it — its event banding (by level) and firing
        // (by place index) then operate correctly in the new order, with no
        // permutation threaded through the Saturator internals.
        let mut order: Vec<usize> = (0..self.bounds.len()).collect();
        let mut current_net = self.clone();
        let init = store.singleton(&current_net.initial_marking);

        // Band events by their SHALLOWEST touched level (recomputed on every
        // reorder). `events_at[l]` holds the indices of transitions fired when
        // saturating a node at level `l`. Empty-effect transitions are dropped.
        let mut events_at = band_events(&current_net);

        let mut peak_nodes = store.interior_node_count();
        let mut reach = init;
        let mut rounds: u32 = 0;
        let mut sifted = false;
        let sift_watermark = max_interior_nodes() / 4 * 3;

        loop {
            rounds += 1;
            if rounds > MAX_ROUNDS {
                return Err(CountError::ResourceCap(
                    "round backstop exceeded".to_string(),
                ));
            }
            Self::check_deadline(deadline)?;
            // GC safepoint (non-moving): `reach` is the only live root at the top
            // of the round; the sat_cache is rebuilt fresh per pass below, so
            // nothing MddRef-valued spans this point. Stable ids keep the
            // `swept == round_start` witness valid. Fires only on large runs.
            if store.should_collect() {
                store.gc(&[reach]);
            }

            // Sifting safepoint: shrink the diagram by dynamic variable
            // reordering when it grows large. `reach` is the ONLY live root ⇒
            // clean single-root remap; the whole net is re-permuted into the new
            // level order (permuted_net) and the event banding recomputed, so the
            // saturator below runs unchanged on the reordered store.
            if crate::sift_runtime::want_sift(&store, self.bounds.len(), sifted, sift_watermark) {
                let (new_store, new_roots, chosen) = store.sift(&[reach]);
                reach = new_roots[0];
                store = new_store;
                order = crate::sift_runtime::compose_order(&order, &chosen);
                current_net = crate::sift_runtime::permuted_net(self, &order);
                events_at = band_events(&current_net);
                sifted = true;
            }

            // Canonical root at round start: convergence is the O(1) structural
            // test `swept == round_start` (see `relprod_fixpoint`) — the whole
            // round (saturation + verification sweep) added nothing — instead of
            // a per-round `BigUint` model-count. Exact at any magnitude.
            let round_start = reach;

            // --- Saturation pass. Fresh sat_cache per pass: the store has
            // grown, and saturated forms are only valid against the structure
            // present when they were computed. ---
            let mut sat = Saturator {
                net: &current_net,
                events_at: events_at.clone(),
                sat_cache: HashMap::new(),
                peak_nodes,
                deadline,
            };
            reach = sat.saturate(&mut store, reach)?;
            peak_nodes = sat.peak_nodes.max(store.interior_node_count());

            // --- Verification sweep: one breadth-first relprod round. If it
            // adds nothing, `reach` is the true fixpoint and we are done. ---
            let mut swept = reach;
            for t in &current_net.transitions {
                // Check the deadline per transition so a long sweep over a
                // many-transition net stays responsive to the wall-clock cap
                // (fail-closed → decline, never overrun-and-return-stale).
                Self::check_deadline(deadline)?;
                let img = transition_image(&mut store, reach, t);
                swept = store.union(swept, img);
                peak_nodes = peak_nodes.max(store.interior_node_count());
                if peak_nodes > max_interior_nodes()
                    || store.approx_store_bytes() > crate::node::max_store_bytes()
                {
                    return Err(CountError::ResourceCap(format!(
                        "interior node cap {} or store byte cap exceeded (saturation verify)",
                        max_interior_nodes()
                    )));
                }
            }

            if swept == round_start {
                // The whole round added nothing ⇒ fixpoint. Return the set.
                return Ok((store, reach, rounds, peak_nodes));
            }
            // Verification found new states ⇒ saturation under-shot (a reduced
            // level hid a band). Absorb and re-saturate; the loop is monotone.
            reach = swept;
        }
    }
}

/// Band each transition by its SHALLOWEST touched level for saturation:
/// `events_at[l]` = indices of transitions whose shallowest effect is at level
/// `l`; empty-effect transitions are dropped. Recomputed whenever the variable
/// order changes (a sift), since a transition's shallowest level moves with the
/// reordering — so it is always called on the current level-space net.
fn band_events(net: &MddNet) -> Vec<Vec<usize>> {
    let n = net.bounds.len();
    let mut events_at: Vec<Vec<usize>> = vec![Vec::new(); n.max(1)];
    for (ti, t) in net.transitions.iter().enumerate() {
        if let Some(b) = shallowest_level(t) {
            events_at[b].push(ti);
        }
    }
    // Intra-band event ORDER (frontier report, tractable item 4): sort each
    // band by SUPPORT SIZE (places with a nonzero pre or post), ascending,
    // tie-broken by declaration index — deterministic. Small-support events
    // fire first during per-node saturation, keeping transient MDDs small
    // before wide events rewrite deeper suffixes. PURE firing-order tuning:
    // saturation is confluent (the reachable fixpoint is order-independent),
    // so the RESULT is byte-identical — only peak transient node counts drop.
    for band in &mut events_at {
        band.sort_by_key(|&ti| {
            let t = &net.transitions[ti];
            let support = t
                .pre
                .iter()
                .zip(&t.post)
                .filter(|(pre, post)| **pre != 0 || **post != 0)
                .count();
            (support, ti)
        });
    }
    events_at
}

/// Internal saturation driver. Holds the event banding, the saturate cache, and
/// resource accounting for one `reachable_count_saturation` run.
struct Saturator<'n> {
    net: &'n MddNet,
    /// `events_at[l]` = transition indices whose SHALLOWEST touched level is `l`
    /// (their saturation band; see [`shallowest_level`]).
    events_at: Vec<Vec<usize>>,
    /// Saturate cache: pre-saturation node → its saturated form.
    sat_cache: HashMap<MddRef, MddRef>,
    peak_nodes: usize,
    deadline: Option<Instant>,
}

impl<'n> Saturator<'n> {
    #[inline]
    fn budget_ok(&mut self, store: &MddStore) -> Result<(), CountError> {
        let c = store.interior_node_count();
        self.peak_nodes = self.peak_nodes.max(c);
        if c > max_interior_nodes() || store.approx_store_bytes() > crate::node::max_store_bytes() {
            return Err(CountError::ResourceCap(format!(
                "interior node cap {} or store byte cap exceeded (saturation)",
                max_interior_nodes()
            )));
        }
        MddNet::check_deadline(self.deadline)
    }

    /// Saturate the node `root`, viewed at its own level downward: first
    /// recursively saturate all children (so the diagram below `root` is
    /// already at fixpoint), then fire every event banded at `root`'s level to
    /// a local fixpoint, re-saturating as new deeper structure appears.
    ///
    /// Returns the fully-saturated node. Idempotent and cached.
    fn saturate(&mut self, store: &mut MddStore, root: MddRef) -> Result<MddRef, CountError> {
        if root.is_terminal() {
            return Ok(root);
        }
        if let Some(&hit) = self.sat_cache.get(&root) {
            return Ok(hit);
        }
        self.budget_ok(store)?;

        let level = store.level_of(root) as usize;
        let dom = store.domain_size(level as u32);

        // Step 1: recursively saturate every child so everything below this
        // node is already at fixpoint before we fire this level's band.
        let mut children: Vec<MddRef> = Vec::with_capacity(dom);
        for v in 0..dom as u64 {
            let c = store.child(root, v);
            let sc = self.saturate(store, c)?;
            children.push(sc);
        }
        let mut node = store.get_node(level as u32, children);

        // Step 2: fire every event banded at THIS level to a local fixpoint.
        //
        // The image of a level-`l` event is again a node at level `l`. We must
        // bring its *deeper* structure (levels > l) to fixpoint before merging
        // — but we must NOT re-run this level's band on the image here, because
        // that is exactly what this outer loop does. Re-saturating the image at
        // its own level would recurse into the same band forever (the image is
        // a fresh node each round). So we saturate only the image's children
        // (`saturate_children`, strictly deeper) and let the loop converge
        // level `l`. This is the standard saturation `SatFire` decomposition.
        let band = self.events_at[level].clone();
        if !band.is_empty() {
            loop {
                let before = node;
                for &ti in &band {
                    let t = &self.net.transitions[ti];
                    let mut imager = Imager::new(t);
                    let img_raw = imager.image_from_level(store, node, level);
                    if img_raw.is_zero() {
                        continue;
                    }
                    // Saturate only the deeper structure of the image.
                    let img = self.saturate_children(store, img_raw, level)?;
                    node = store.union(node, img);
                    self.budget_ok(store)?;
                }
                if node == before {
                    break; // local fixpoint at this level reached
                }
            }
        }

        self.sat_cache.insert(root, node);
        Ok(node)
    }

    /// Saturate only the structure strictly BELOW `node`'s firing level
    /// `fire_level`: every child (which sits at a level `> fire_level`) is
    /// fully `saturate`d, but the node's own level is left to the caller's band
    /// loop. This is the image-merge helper for `SatFire` — it guarantees the
    /// merged image's deeper levels are at fixpoint without re-entering the
    /// firing level's band (which would not terminate).
    ///
    /// If `node` itself sits deeper than `fire_level + 1` (the image collapsed
    /// the band level out because it was redundant), then `node` is a deeper
    /// node and a full `saturate` of it is correct and terminating (its own
    /// band is a strictly-deeper level).
    fn saturate_children(
        &mut self,
        store: &mut MddStore,
        node: MddRef,
        fire_level: usize,
    ) -> Result<MddRef, CountError> {
        if node.is_terminal() {
            return Ok(node);
        }
        let level = store.level_of(node) as usize;
        if level > fire_level {
            // The image node sits strictly deeper than the firing level (the
            // firing level was redundant and reduced away). Fully saturating it
            // is correct and terminates — its band is deeper than fire_level.
            return self.saturate(store, node);
        }
        // node is exactly at fire_level: saturate each child (strictly deeper),
        // but do not fire fire_level's band here.
        debug_assert_eq!(level, fire_level);
        let dom = store.domain_size(level as u32);
        let mut children = Vec::with_capacity(dom);
        for v in 0..dom as u64 {
            let c = store.child(node, v);
            let sc = self.saturate(store, c)?;
            children.push(sc);
        }
        self.budget_ok(store)?;
        Ok(store.get_node(level as u32, children))
    }
}

#[cfg(test)]
mod tests {
    use crate::reach::{MddNet, MddTransition};

    fn t(pre: Vec<u64>, post: Vec<u64>) -> MddTransition {
        MddTransition { pre, post }
    }

    /// All three engines (explicit kernel, relprod, saturation) must agree.
    fn agree(net: &MddNet) -> u64 {
        let explicit = net.reachable_count().expect("explicit ok").state_count;
        let relprod = net
            .reachable_count_relprod(None)
            .expect("relprod ok")
            .state_count;
        let sat = net
            .reachable_count_saturation(None)
            .expect("saturation ok")
            .state_count;
        assert_eq!(explicit, relprod, "relprod disagrees with explicit kernel");
        assert_eq!(explicit, sat, "saturation disagrees with explicit kernel");
        explicit
    }

    #[test]
    fn shuttle_two_states() {
        let net = MddNet {
            bounds: vec![1, 1],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])],
        };
        assert_eq!(agree(&net), 2);
    }

    #[test]
    fn relprod_and_saturation_stable_under_forced_gc() {
        // End-to-end GC root-supply validation for the relprod + saturation
        // safepoints (GC step 3/4): forcing gc(&[reach]) at every round must not
        // change either engine's count. A multi-round two-counter net so GC
        // fires on a growing set each round.
        let net = MddNet {
            bounds: vec![6, 6],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        };
        let baseline = agree(&net); // 7 * 7 = 49 reachable markings
        assert_eq!(baseline, 49);

        crate::node::set_gc_stress(true);
        let relprod = net
            .reachable_count_relprod(None)
            .expect("relprod ok")
            .state_count;
        let sat = net
            .reachable_count_saturation(None)
            .expect("saturation ok")
            .state_count;
        crate::node::set_gc_stress(false);

        assert_eq!(relprod, baseline, "relprod count stable under forced gc");
        assert_eq!(sat, baseline, "saturation count stable under forced gc");
    }

    #[test]
    fn relprod_and_saturation_decline_gracefully_under_forced_abort() {
        // Cooperative in-operation abort (audit 2026-07-11): with the abort
        // probe ARMED (both fixpoints install one via set_abort_probe) and the
        // forced-abort stress hook on, get_node raises MddAbort on the first
        // fresh interior node; catch_mdd_abort must fold that unwind into a
        // ResourceCap decline (NOT a process abort, NOT a wrong count), and the
        // count must be exact again once the hook is cleared — proving the panic
        // path leaves no corruption and the unarmed path is unaffected.
        let net = MddNet {
            bounds: vec![6, 6],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        };
        let baseline = agree(&net); // 7 * 7 = 49 reachable markings
        assert_eq!(baseline, 49);

        crate::node::set_abort_stress(true);
        let relprod = net.reachable_count_relprod(None);
        let sat = net.reachable_count_saturation(None);
        crate::node::set_abort_stress(false);

        assert!(
            matches!(relprod, Err(crate::CountError::ResourceCap(_))),
            "forced abort must fold into a ResourceCap decline, got {relprod:?}"
        );
        assert!(
            matches!(sat, Err(crate::CountError::ResourceCap(_))),
            "forced abort must fold into a ResourceCap decline, got {sat:?}"
        );

        // No lingering corruption: the exact count returns once the hook clears.
        assert_eq!(
            net.reachable_count_relprod(None)
                .expect("relprod ok")
                .state_count,
            baseline,
            "count is exact again after a forced abort (no corruption)"
        );
        assert_eq!(
            net.reachable_count_saturation(None)
                .expect("saturation ok")
                .state_count,
            baseline,
            "saturation exact again after a forced abort (no corruption)"
        );
    }

    #[test]
    fn relprod_and_saturation_stable_under_forced_sift() {
        // End-to-end sift soundness for BOTH symbolic safepoints (relprod +
        // saturation): forcing a reorder at EVERY round must not change either
        // count. relprod permutes each transition into level-space; saturation
        // re-permutes the WHOLE net (permuted_net) and re-bands events. A
        // mis-permutation would apply token deltas / band events at the wrong
        // levels and corrupt the count. Asymmetric bounds ⇒ non-trivial permute.
        let nets = [
            MddNet {
                bounds: vec![6, 6],
                initial_marking: vec![0, 0],
                transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
            },
            MddNet {
                bounds: vec![5, 2, 3],
                initial_marking: vec![0, 0, 0],
                transitions: vec![
                    t(vec![0, 0, 0], vec![1, 0, 0]),
                    t(vec![1, 0, 0], vec![0, 1, 0]),
                    t(vec![0, 1, 0], vec![0, 0, 1]),
                ],
            },
        ];
        for net in &nets {
            let normal_r = net.reachable_count_relprod(None).expect("ok").state_count;
            let normal_s = net
                .reachable_count_saturation(None)
                .expect("ok")
                .state_count;
            crate::sift_runtime::set_sift_stress(true);
            let forced_r = net.reachable_count_relprod(None).expect("ok").state_count;
            let forced_s = net
                .reachable_count_saturation(None)
                .expect("ok")
                .state_count;
            crate::sift_runtime::set_sift_stress(false);
            assert_eq!(
                forced_r, normal_r,
                "forced sift changed the relprod count for bounds {:?}",
                net.bounds
            );
            assert_eq!(
                forced_s, normal_s,
                "forced sift changed the saturation count for bounds {:?}",
                net.bounds
            );
        }
    }

    /// `MddNet::upper_bounds` — the EXACT reachable max of each linear query over
    /// R, the native twin of the BDD UpperBounds lane. Two independent counters
    /// (R = {(a,b): 0<=a<=3, 0<=b<=2}); the per-query bounds must match the
    /// hand-computed maxima of Σ c_p·m[p].
    #[test]
    fn upper_bounds_exact_per_query() {
        let net = MddNet {
            bounds: vec![3, 2],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        };
        let queries = vec![
            vec![1i128, 1], // max a+b   = 3+2 = 5
            vec![2, 5],     // max 2a+5b = 6+10 = 16
            vec![1, 0],     // max a     = 3
            vec![-1, 3],    // max -a+3b = 0+6 = 6
        ];
        let bounds = net.upper_bounds(&queries, None).expect("R built");
        assert_eq!(
            bounds,
            vec![Some(5), Some(16), Some(3), Some(6)],
            "exact reachable maxima per linear query",
        );
    }

    /// FORMAL SOUNDNESS PROOF: the saturated reachable set `R` is a sound
    /// inductive reachability fixpoint — `init ∈ R ∧ ∀t. image_t(R) ⊆ R` — so
    /// `R ⊇ reachable` (no reachable marking is missed). Discharged STRUCTURALLY
    /// (not by count-matching) across a battery spanning the saturation regimes:
    /// shuttle, counter, product, conserved ring, weighted arcs. This is the
    /// proof obligation that justifies the MDD lane being authoritative — the
    /// soundness direction proven, independent of the build internals.
    #[test]
    fn saturation_result_is_a_sound_inductive_fixpoint() {
        let battery = vec![
            // shuttle (2 states)
            MddNet {
                bounds: vec![1, 1],
                initial_marking: vec![1, 0],
                transitions: vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])],
            },
            // counter full range
            MddNet {
                bounds: vec![5],
                initial_marking: vec![0],
                transitions: vec![t(vec![0], vec![1])],
            },
            // two independent counters (product)
            MddNet {
                bounds: vec![3, 3],
                initial_marking: vec![0, 0],
                transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
            },
            // conserved token ring
            MddNet {
                bounds: vec![1, 1, 1],
                initial_marking: vec![1, 0, 0],
                transitions: vec![
                    t(vec![1, 0, 0], vec![0, 1, 0]),
                    t(vec![0, 1, 0], vec![0, 0, 1]),
                    t(vec![0, 0, 1], vec![1, 0, 0]),
                ],
            },
            // weighted arcs (>1 tokens)
            MddNet {
                bounds: vec![4, 4],
                initial_marking: vec![4, 0],
                transitions: vec![t(vec![2, 0], vec![0, 1])],
            },
        ];
        for (i, net) in battery.iter().enumerate() {
            let sound = net
                .verify_saturation_inductive_fixpoint(None)
                .unwrap_or_else(|e| panic!("battery[{i}] build failed: {e:?}"));
            assert!(
                sound,
                "battery[{i}]: saturated R is NOT a sound inductive fixpoint \
                 (init ⊆ R ∧ closed-under-Next FAILED)",
            );
        }
    }

    #[test]
    fn counter_full_range() {
        let net = MddNet {
            bounds: vec![5],
            initial_marking: vec![0],
            transitions: vec![t(vec![0], vec![1])],
        };
        assert_eq!(agree(&net), 6);
    }

    #[test]
    fn two_independent_counters_product() {
        let net = MddNet {
            bounds: vec![3, 3],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        };
        assert_eq!(agree(&net), 16);
    }

    #[test]
    fn conserved_token_ring() {
        let net = MddNet {
            bounds: vec![1, 1, 1],
            initial_marking: vec![1, 0, 0],
            transitions: vec![
                t(vec![1, 0, 0], vec![0, 1, 0]),
                t(vec![0, 1, 0], vec![0, 0, 1]),
                t(vec![0, 0, 1], vec![1, 0, 0]),
            ],
        };
        assert_eq!(agree(&net), 3);
    }

    #[test]
    fn weighted_arcs_with_bound_truncation() {
        let net = MddNet {
            bounds: vec![1, 2],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 2]), t(vec![0, 1], vec![1, 0])],
        };
        // Let the explicit kernel decide the count; assert all three agree.
        let c = agree(&net);
        assert!(c >= 2);
    }

    #[test]
    fn conserved_n_token_line_saturation_shrinks_peak() {
        // A conserved net: N tokens on a line p0..p4, each transition moves one
        // token right or left. Token count conserved ⇒ saturation should keep
        // the peak node count modest. We only assert correctness here; peak
        // measurements live in the crosscheck battery / report.
        let n_places = 5;
        let cap = 3u64;
        let bounds = vec![cap; n_places];
        let mut transitions = Vec::new();
        for p in 0..n_places - 1 {
            // move right: consume 1 from p, produce 1 on p+1
            let mut pre = vec![0u64; n_places];
            let mut post = vec![0u64; n_places];
            pre[p] = 1;
            post[p + 1] = 1;
            transitions.push(t(pre, post));
            // move left
            let mut pre = vec![0u64; n_places];
            let mut post = vec![0u64; n_places];
            pre[p + 1] = 1;
            post[p] = 1;
            transitions.push(t(pre, post));
        }
        let mut initial_marking = vec![0u64; n_places];
        initial_marking[0] = 2; // 2 tokens
        let net = MddNet {
            bounds,
            initial_marking,
            transitions,
        };
        let _ = agree(&net);
    }

    #[test]
    fn deadline_declines_not_crashes() {
        // A net with a state space large enough to take measurable time, with a
        // deadline in the past ⇒ must DECLINE (fail-closed), never panic.
        let net = MddNet {
            bounds: vec![50, 50, 50],
            initial_marking: vec![0, 0, 0],
            transitions: vec![
                t(vec![0, 0, 0], vec![1, 0, 0]),
                t(vec![0, 0, 0], vec![0, 1, 0]),
                t(vec![0, 0, 0], vec![0, 0, 1]),
            ],
        };
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap();
        assert!(net.reachable_count_relprod(Some(past)).is_err());
        assert!(net.reachable_count_saturation(Some(past)).is_err());
    }

    #[test]
    fn malformed_declined_all_engines() {
        let net = MddNet {
            bounds: vec![1, 1],
            initial_marking: vec![0],
            transitions: vec![],
        };
        assert!(net.reachable_count_relprod(None).is_err());
        assert!(net.reachable_count_saturation(None).is_err());
    }

    #[test]
    fn zero_places_one_marking() {
        let net = MddNet {
            bounds: vec![],
            initial_marking: vec![],
            transitions: vec![],
        };
        assert_eq!(net.reachable_count_relprod(None).unwrap().state_count, 1);
        assert_eq!(net.reachable_count_saturation(None).unwrap().state_count, 1);
    }
}
