// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! The four MCC `StateSpace` metrics, computed SYMBOLICALLY over the reachable
//! set MDD.
//!
//! The production `StateSpace` examination reports four numbers per net:
//!
//! - `state_count` — `|R|`, the number of reachable markings.
//! - `edge_count`  — Σ over reachable markings `M` of the number of transitions
//!   ENABLED at `M` (where "enabled" means the guard holds AND the successor is
//!   in-bounds — exactly the BFS observer's `on_transition_fire` count and the
//!   `tla-dd` `edge_count` / `bfs_full_metrics` convention).
//! - `max_token_in_place` — `max_{M ∈ R, p} m[p]`.
//! - `max_token_sum`       — `max_{M ∈ R} Σ_p m[p]`.
//!
//! All four are read off the compact reachable-set MDD without enumerating
//! markings, so the cost stays a function of the MDD node count, not `|R|`.
//!
//! # Soundness posture (ABSOLUTE)
//!
//! Every metric is EXACT or the whole bundle DECLINES (fail-closed):
//! `count_markings` already declines past `u64::MAX`; `edge_count` uses checked
//! `u64` arithmetic and declines on overflow; `max_token_*` are bounded by the
//! per-place bounds (saturating) so they cannot wrap. A decline returns
//! [`CountError`], never a wrong value.
//!
//! The four metrics are cross-checked against an independent explicit-BFS
//! metric oracle (the same `Σ enabled-firings` / per-place / total-sum
//! semantics the production BDD lane and the BFS observer use) on the
//! differential battery in `tests/crosscheck_bfs.rs`, 0 disagreements required.
//! Until that battery is green no production path may consume these metrics.

use crate::node::{MddRef, MddStore, TERMINAL_LEVEL};
use crate::reach::{CountError, MddNet, MddTransition};
use crate::set_ops::big_to_u128;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tla_bignum::{BigUint, ToPrimitive};

/// The four MCC `StateSpace` metrics. Mirrors `tla_dd::DdStateSpaceMetrics`.
///
/// The authoritative `|R|` and edge counts are now the arbitrary-precision
/// [`Self::state_count_big`] / [`Self::edge_count_big`] fields, so a
/// structurally-computable reachable set BEYOND `u128` (e.g. FMS ≈1e47,
/// Kanban/Philosophers families up to ≈1e238) is REPORTED rather than declining
/// on the `u128` carrier. The narrowed `u64`/`u128` fields are retained for the
/// BDD cross-check and back-compat; they are exact whenever the count fits and
/// otherwise saturated markers (the `_big` field is then the source of truth).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MddStateSpaceMetrics {
    /// `|R|` — number of reachable markings, narrowed to `u64`. Present for
    /// back-compat / the BDD cross-check (whose count is `u64`). `Some` iff the
    /// count fits `u64`; `None` when `u64::MAX < |R|` (in which case
    /// [`Self::state_count_big`] carries the exact value).
    pub state_count: Option<u64>,
    /// `|R|` — number of reachable markings narrowed to `u128`, exact when
    /// `|R| <= u128::MAX` (covers up to ≈ 3.4e38). When `|R| > u128::MAX` this
    /// is the saturated marker `u128::MAX` and [`Self::state_count_big`] is the
    /// source of truth. Retained for back-compat / the in-range BDD cross-check.
    pub state_count_u128: u128,
    /// `|R|` — EXACT number of reachable markings as an arbitrary-precision
    /// [`BigUint`]. The authoritative count: never declines on magnitude, so
    /// astronomically-large-but-finite reachable sets are reported. The MCC
    /// `STATE_SPACE STATES` row is emitted from this value at full precision.
    pub state_count_big: BigUint,
    /// Σ over reachable markings of the number of enabled (in-bounds) firings,
    /// narrowed to `u128` (exact when it fits, else the saturated marker
    /// `u128::MAX`; [`Self::edge_count_big`] is then the source of truth).
    pub edge_count: u128,
    /// Σ over reachable markings of the number of enabled (in-bounds) firings,
    /// EXACT as an arbitrary-precision [`BigUint`]. The authoritative edge
    /// count; the MCC `STATE_SPACE TRANSITIONS` row is emitted from it at full
    /// precision.
    pub edge_count_big: BigUint,
    /// `max_{M ∈ R, p} m[p]`.
    pub max_token_in_place: u64,
    /// `max_{M ∈ R} Σ_p m[p]`.
    pub max_token_sum: u64,
    /// Number of fixpoint rounds the underlying reachability engine used
    /// (diagnostic only).
    pub iterations: u32,
}

impl MddNet {
    /// Compute ALL FOUR `StateSpace` metrics symbolically off the reachable
    /// set, using the SYMBOLIC relational-product fixpoint to build `R` (the
    /// compact pillar for counter / conserved nets; its set is pinned EQUAL to
    /// the saturation and BFS reachable set by the differential battery).
    /// `deadline` is an optional wall-clock cap. Fail-closed: any overflow /
    /// node-budget / deadline returns [`CountError`], never a partial or wrong
    /// metric bundle.
    pub fn state_space_metrics(
        &self,
        deadline: Option<Instant>,
    ) -> Result<MddStateSpaceMetrics, CountError> {
        self.validate()?;
        // Build the reachable set via NODE-LEVEL SATURATION (the scalable
        // engine). Saturation converges on the high-diameter conserved /
        // counter nets where the breadth-first relprod fixpoint times out, so
        // wiring the metric path to it is what lets the lane actually decide
        // those nets. The saturated set is pinned EQUAL to the relprod and BFS
        // reachable set by the differential battery, so the four metrics read
        // off it are exactly the cross-checked metrics.
        let (mut store, reach, iterations) = self.build_reachable_saturation(deadline)?;
        self.metrics_from_reachable(&mut store, reach, iterations)
    }

    /// Build the reachable set via the breadth-first relational-product
    /// fixpoint and extract the four StateSpace metrics. Retained as a
    /// CROSS-CHECK fallback: on the structured nets the relprod converges on,
    /// its metric bundle must equal the saturation bundle exactly (the
    /// differential battery pins this). Production uses
    /// [`Self::state_space_metrics`] (saturation); this exists so the battery
    /// can verify the two set-build engines agree on every metric.
    pub fn state_space_metrics_relprod(
        &self,
        deadline: Option<Instant>,
    ) -> Result<MddStateSpaceMetrics, CountError> {
        self.validate()?;
        let (mut store, reach, iterations) = self.build_reachable_relprod(deadline)?;
        self.metrics_from_reachable(&mut store, reach, iterations)
    }

    /// DIAGNOSTIC: build the reachable set via saturation and return the final
    /// interior-node count of its MDD representation — the reachable-set size,
    /// i.e. the quantity the `max_interior_nodes` cap bounds. Used to measure the
    /// effect of variable ordering on MDD size directly (not via the transition-
    /// span proxy): a good place→level order keeps this count small where an
    /// arbitrary order inflates it past the cap.
    ///
    /// This is NOT on the compared-metrics path (the count is order-dependent, so
    /// it is deliberately kept off [`MddStateSpaceMetrics`], whose `PartialEq` the
    /// differential battery relies on). Fail-closed like the metric path: any
    /// overflow / node-budget / deadline returns [`CountError`].
    pub fn reachable_set_node_count(&self, deadline: Option<Instant>) -> Result<usize, CountError> {
        self.validate()?;
        let (store, _reach, _iterations) = self.build_reachable_saturation(deadline)?;
        Ok(store.interior_node_count())
    }

    /// Extract the four metrics from a pre-built reachable-set root. Split out
    /// so the cross-check battery can feed a set built by ANY engine and
    /// confirm metric extraction is engine-agnostic.
    ///
    /// `store` is taken `&mut` because `edge_count` builds per-transition
    /// `Fireable` sets and their intersections (purely additive — `reach` and
    /// its nodes are never mutated, so it stays canonical). The two
    /// `max_token_*` reads are then performed on the immutable view.
    pub(crate) fn metrics_from_reachable(
        &self,
        store: &mut MddStore,
        reach: MddRef,
        iterations: u32,
    ) -> Result<MddStateSpaceMetrics, CountError> {
        // EXACT arbitrary-precision count — the authoritative |R|, never
        // declines on magnitude. Narrow to `u128` / `u64` for back-compat: exact
        // when it fits, the saturated marker `u128::MAX` (resp. `None`) when it
        // does not. The `_big` field is the source of truth either way.
        let state_count_big = store.count_markings_big(reach);
        let state_count_u128 = big_to_u128(&state_count_big).unwrap_or(u128::MAX);
        let state_count = state_count_big.to_u64();
        // EXACT edge count (Σ |R ∩ Fireable(t)|), bignum; narrow likewise.
        let edge_count_big = self.edge_count_big(store, reach);
        let edge_count = big_to_u128(&edge_count_big).unwrap_or(u128::MAX);
        let max_token_in_place = max_token_in_place(store, reach);
        let max_token_sum = max_token_sum(store, reach);
        Ok(MddStateSpaceMetrics {
            state_count,
            state_count_u128,
            state_count_big,
            edge_count,
            edge_count_big,
            max_token_in_place,
            max_token_sum,
            iterations,
        })
    }

    /// Σ over reachable markings of the number of enabled (in-bounds) firings.
    ///
    /// For each transition `t`, the markings that contribute a firing are
    /// exactly `R ∩ Fireable(t)`, where `Fireable(t)` is the set of markings at
    /// which `t`'s guard holds AND the successor is in-bounds. Because each
    /// place's fireability condition (`pre[l] <= v` and `v - pre[l] + post[l] <=
    /// bound[l]`) is INDEPENDENT per place, `Fireable(t)` is a product set — a
    /// per-level chain — and the count is `|R ∩ Fireable(t)|`. Summing over
    /// transitions gives the edge count, matching the BFS observer's
    /// `on_transition_fire` semantics and `tla_dd::edge_count`.
    ///
    /// Each per-transition term is the EXACT `BigUint` count of `R ∩ Fireable(t)`
    /// ([`MddStore::count_markings_big`]) and the running total is an exact
    /// `BigUint` sum, so the edge count NEVER declines on magnitude (the caller
    /// narrows it fail-closed to `u128`/`u64` for the back-compat fields, leaving
    /// `edge_count_big` as the source of truth).
    fn edge_count_big(&self, store: &mut MddStore, reach: MddRef) -> BigUint {
        let mut total = BigUint::from(0u32);
        for t in &self.transitions {
            let fireable = build_fireable_set(store, &self.bounds, t);
            let inter = store.intersect(reach, fireable);
            // EXACT per-transition fireable subset count; sums never decline on
            // magnitude (bignum), so the edge total is exact for any net.
            total += store.count_markings_big(inter);
        }
        total
    }
}

/// Build `Fireable(t)` — the set of markings at which `t` is enabled and its
/// successor is in-bounds — as an MDD over the net's `bounds`.
///
/// Per place `l` the condition is `pre[l] <= v` and `v - pre[l] + post[l] <=
/// bound[l]`. These are independent across places, so the set is the product
/// of per-level allowed-value sets: a chain where each level's node sends every
/// allowed value to the all-allowed-below subtree and every disallowed value to
/// `ZERO`. Built bottom-up so each child exists before its parent is interned.
/// An empty net (zero places) yields `ONE` (the single empty marking always
/// fires the empty transition) — handled by the loop running zero times.
pub(crate) fn build_fireable_set(
    store: &mut MddStore,
    bounds: &[u64],
    t: &MddTransition,
) -> MddRef {
    let n = bounds.len();
    let mut acc = MddRef::ONE;
    for level in (0..n).rev() {
        let dom = store.domain_size(level as u32);
        let pre_l = t.pre[level];
        let post_l = t.post[level];
        let mut children = vec![MddRef::ZERO; dom];
        for v in 0..dom as u64 {
            // Guard: enough tokens to consume.
            if v < pre_l {
                continue;
            }
            // Successor in-bounds (no underflow: v >= pre_l).
            let next = v - pre_l + post_l;
            if next > bounds[level] {
                continue;
            }
            children[v as usize] = acc;
        }
        acc = store.get_node(level as u32, children);
    }
    acc
}

/// `max_token_in_place` over the set rooted at `reach`.
///
/// A value `v` at level `l` is attained by some in-set marking iff either (a)
/// some reachable node sits at level `l` with a non-`ZERO` child along edge
/// `v`, or (b) level `l` is FREE on some surviving path (skipped by a long-jump
/// edge to a non-`ZERO` child, skipped above the root, or skipped below an edge
/// into `ONE`) — in which case every value `0..=bound[l]` is attained.
///
/// So: collect, per level, whether the level is ever free on a surviving path
/// (⇒ contributes `bound[l]`), and the max explicit edge value at any node on
/// that level. The overall maximum is the answer. Bounded by `max bound[l]`, so
/// it cannot overflow.
fn max_token_in_place(store: &MddStore, reach: MddRef) -> u64 {
    if reach.is_zero() {
        return 0; // empty set (degenerate; R is never empty in practice)
    }
    let n = store.num_levels();
    if n == 0 {
        return 0;
    }
    // `best[l]` = max value attained at level `l` over in-set markings.
    let mut best = vec![0u64; n];

    // Free levels above the root: root is non-ZERO, so a surviving path exists;
    // levels above it are unconstrained ⇒ full bound.
    let root_level = level_index(store, reach);
    for (l, slot) in best.iter_mut().enumerate().take(root_level.min(n)) {
        *slot = (*slot).max(store.bounds[l]);
    }

    // Walk every reachable node once.
    let mut seen: HashSet<MddRef> = HashSet::new();
    let mut stack = vec![reach];
    while let Some(node) = stack.pop() {
        if node.is_terminal() {
            continue;
        }
        if !seen.insert(node) {
            continue;
        }
        let level = store.level_of(node) as usize;
        let dom = store.domain_size(level as u32);
        for v in 0..dom as u64 {
            let child = store.child(node, v);
            if child.is_zero() {
                continue;
            }
            // Explicit value `v` at this level is attained.
            best[level] = best[level].max(v);
            // Levels strictly between this node and `child` are free on this
            // surviving path ⇒ full bound.
            let child_upper = level_index(store, child).min(n);
            for (l, slot) in best
                .iter_mut()
                .enumerate()
                .take(child_upper)
                .skip(level + 1)
            {
                *slot = (*slot).max(store.bounds[l]);
            }
            stack.push(child);
        }
    }

    best.into_iter().max().unwrap_or(0)
}

/// `max_token_sum` over the set rooted at `reach`: the maximum, over in-set
/// markings, of the total token count.
///
/// A memoized DFS computes `max_sum_below(node)` = the max, over in-set
/// completions from `node` downward, of the sum of values at `node`'s level and
/// below. For an interior node at level `l` it is
/// `max_{v : child(n,v) != ZERO} ( v + gap_bonus(l, child) + max_sum_below(child) )`,
/// where `gap_bonus` adds the bounds of any levels skipped between `l` and
/// `child` (those places are free, so to MAXIMISE we take their full bound).
/// `ONE` contributes `0` (the below-span bonus is added by the caller's edge).
/// The result adds the top-gap bonus (free levels above the root).
///
/// Bounded by `Σ bound[l]` (saturating), so no wrap within the engine's
/// bounded-net scope.
fn max_token_sum(store: &MddStore, reach: MddRef) -> u64 {
    if reach.is_zero() {
        return 0;
    }
    let n = store.num_levels();
    // Free levels above the root contribute their full bound (path survives).
    let root_level = level_index(store, reach);
    let mut top_bonus: u64 = 0;
    for l in 0..root_level.min(n) {
        top_bonus = top_bonus.saturating_add(store.bounds[l]);
    }
    let mut memo: HashMap<MddRef, u64> = HashMap::new();
    top_bonus.saturating_add(max_sum_below(store, reach, &mut memo))
}

/// `max_sum_below` recursion (see [`max_token_sum`]). Returns the maximum sum
/// of values at `node`'s level and strictly below, over in-set completions.
/// `ZERO` is unreachable (callers skip it); `ONE` (bottom) contributes 0.
fn max_sum_below(store: &MddStore, node: MddRef, memo: &mut HashMap<MddRef, u64>) -> u64 {
    if node.is_one() {
        return 0;
    }
    debug_assert!(!node.is_zero(), "ZERO must be filtered by the caller");
    if let Some(&c) = memo.get(&node) {
        return c;
    }
    let level = store.level_of(node) as usize;
    let dom = store.domain_size(level as u32);
    let n = store.num_levels();
    let mut best: u64 = 0;
    for v in 0..dom as u64 {
        let child = store.child(node, v);
        if child.is_zero() {
            continue;
        }
        // Free levels skipped between this node and `child`: take full bounds.
        let child_upper = level_index(store, child).min(n);
        let mut gap_bonus: u64 = 0;
        for l in (level + 1)..child_upper {
            gap_bonus = gap_bonus.saturating_add(store.bounds[l]);
        }
        let sub = max_sum_below(store, child, memo);
        let candidate = v.saturating_add(gap_bonus).saturating_add(sub);
        best = best.max(candidate);
    }
    memo.insert(node, best);
    best
}

/// `max_{M ∈ R} Σ_p coeffs[p]·m[p]` — the EXACT arbitrary-coefficient
/// `UpperBounds` value over the reachable set rooted at `reach`. Generalizes
/// [`max_token_sum`] (all coeffs `1`): for the node's own level it maximises
/// over the in-set values at that level (so NEGATIVE coefficients are handled —
/// the value minimising that term, among reachable ones, is chosen); a FREE
/// level `l` (skipped in the reduced MDD ⇒ every value `0..=bounds[l]` is in the
/// set) contributes `max(0, coeffs[l])·bounds[l]` (a positive-coeff place takes
/// its full bound, a non-positive one takes 0). Returns `None` on `i128`
/// overflow (fail-closed) or a `coeffs`/levels length mismatch. `coeffs.len()`
/// must equal the place count.
#[must_use]
pub fn max_weighted_sum_of(store: &MddStore, reach: MddRef, coeffs: &[i128]) -> Option<i128> {
    let n = store.num_levels();
    if coeffs.len() != n {
        return None;
    }
    if reach.is_zero() {
        return Some(0); // empty set (does not occur for a live reachable set)
    }
    let free_contrib =
        |l: usize| -> Option<i128> { coeffs[l].max(0).checked_mul(store.bounds[l] as i128) };
    // Free levels above the root contribute their best in-set value.
    let root_level = level_index(store, reach);
    let mut top_bonus: i128 = 0;
    for l in 0..root_level.min(n) {
        top_bonus = top_bonus.checked_add(free_contrib(l)?)?;
    }
    let mut memo: HashMap<MddRef, Option<i128>> = HashMap::new();
    let below = max_weighted_below(store, reach, coeffs, &mut memo)?;
    top_bonus.checked_add(below)
}

/// `max_weighted_sum_of` recursion (the weighted analogue of [`max_sum_below`]).
/// Returns `None` on `i128` overflow.
fn max_weighted_below(
    store: &MddStore,
    node: MddRef,
    coeffs: &[i128],
    memo: &mut HashMap<MddRef, Option<i128>>,
) -> Option<i128> {
    if node.is_one() {
        return Some(0);
    }
    debug_assert!(!node.is_zero(), "ZERO must be filtered by the caller");
    if let Some(&c) = memo.get(&node) {
        return c;
    }
    let level = store.level_of(node) as usize;
    let dom = store.domain_size(level as u32);
    let n = store.num_levels();
    let mut best: Option<i128> = None;
    for v in 0..dom as u64 {
        let child = store.child(node, v);
        if child.is_zero() {
            continue;
        }
        // Free levels skipped between this node and `child`: best in-set value.
        let child_upper = level_index(store, child).min(n);
        let mut gap_bonus: i128 = 0;
        for l in (level + 1)..child_upper {
            gap_bonus =
                gap_bonus.checked_add(coeffs[l].max(0).checked_mul(store.bounds[l] as i128)?)?;
        }
        let sub = max_weighted_below(store, child, coeffs, memo)?;
        let candidate = coeffs[level]
            .checked_mul(v as i128)?
            .checked_add(gap_bonus)?
            .checked_add(sub)?;
        best = Some(best.map_or(candidate, |b: i128| b.max(candidate)));
    }
    // A non-ZERO inner node always has >=1 non-zero child (reducedness).
    let result = Some(best.unwrap_or(0));
    memo.insert(node, result);
    result
}

// ---------------------------------------------------------------------------
// Public metric-read surface for the BINDING-QUANTIFIED colored StateSpace path.
//
// The quantified driver (`colored_image::colored_transition_image_quantified`)
// builds the reachable SET directly in a caller-owned `MddStore` WITHOUT a
// materialized `MddNet`/transition list (the binding count is too large to
// materialize). The three set-read metrics below therefore need a store+root
// entry point independent of `MddNet`; `edge_count` is computed binding-
// quantified by the caller (it sums `|R ∩ Fireable(b)|` over bindings), reusing
// the public [`fireable_set`]. These are thin re-exports of the SAME functions
// the `MddNet` metric path uses, so the two paths read IDENTICAL metrics off an
// identical reachable set — the differential gate.
// ---------------------------------------------------------------------------

/// Public: `max_{M ∈ R, p} m[p]` over the set rooted at `reach`. Same function
/// the `MddNet` metric path uses (the internal `max_token_in_place`): the
/// largest token count any single place attains over the reachable markings.
#[must_use]
pub fn max_token_in_place_of(store: &MddStore, reach: MddRef) -> u64 {
    max_token_in_place(store, reach)
}

/// Public: `max_{M ∈ R} Σ_p m[p]` over the set rooted at `reach`. Same function
/// the `MddNet` metric path uses (the internal `max_token_sum`): the largest
/// total token count, summed across all places, over the reachable markings.
#[must_use]
pub fn max_token_sum_of(store: &MddStore, reach: MddRef) -> u64 {
    max_token_sum(store, reach)
}

/// Public: `Fireable(t)` over the given `bounds` — the EXACT set the `edge_count`
/// metric intersects with the reachable set. The binding-quantified `edge_count`
/// sums `|R ∩ fireable_set(b)|` over a transition's bindings (counts ADD, no
/// double-count: a binding is one distinct event, and `R ∩ Fireable(b)` is the
/// markings that fire THAT binding). Thin re-export of the internal
/// `build_fireable_set` the `MddNet` edge-count path uses.
#[must_use]
pub fn fireable_set(store: &mut MddStore, bounds: &[u64], t: &MddTransition) -> MddRef {
    build_fireable_set(store, bounds, t)
}

/// The level a node occupies as a `usize` index into `bounds`, with the
/// terminal mapped to `num_levels` (one past the last place level) so a
/// long-jump edge into a terminal correctly treats the remaining place levels
/// as a skipped (free) span.
#[inline]
fn level_index(store: &MddStore, node: MddRef) -> usize {
    let l = store.level_of(node);
    if l == TERMINAL_LEVEL {
        store.num_levels()
    } else {
        l as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reach::MddTransition;

    fn t(pre: Vec<u64>, post: Vec<u64>) -> MddTransition {
        MddTransition { pre, post }
    }

    /// `max_weighted_sum_of` (the exact arbitrary-coefficient UpperBounds value)
    /// must (a) generalize `max_token_sum_of` on all-1 coefficients and (b) equal
    /// the brute-force max of `Σ cₚ·m[p]` over the reachable set, including
    /// NEGATIVE coefficients. The MDD capability the BDD UpperBounds lane has.
    #[test]
    fn max_weighted_sum_is_exact_for_arbitrary_coefficients() {
        // Two independent counters: R = {(a,b): 0<=a<=3, 0<=b<=2} (full product).
        let net = MddNet {
            bounds: vec![3, 2],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        };
        let (store, reach, _) = net.build_reachable_saturation(None).expect("R built");
        let bf = |c: &[i128]| -> i128 {
            let mut best = i128::MIN;
            for a in 0..=net.bounds[0] as i128 {
                for b in 0..=net.bounds[1] as i128 {
                    best = best.max(c[0] * a + c[1] * b);
                }
            }
            best
        };
        // (a) generalizes max_token_sum on all-1 coefficients.
        assert_eq!(
            max_weighted_sum_of(&store, reach, &[1, 1]),
            Some(max_token_sum_of(&store, reach) as i128),
        );
        // (b) exact vs brute force, including negative + zero coefficients.
        for c in [
            vec![1i128, 1],
            vec![2, 5],
            vec![-1, 3],
            vec![5, -2],
            vec![0, 1],
        ] {
            assert_eq!(
                max_weighted_sum_of(&store, reach, &c),
                Some(bf(&c)),
                "coeffs {c:?}"
            );
        }
    }

    /// Brute-force reference: enumerate the reachable set explicitly (same
    /// firing rule as the engines) and compute the four metrics directly,
    /// matching `tla_dd::bfs_full_metrics`.
    fn bfs_metrics(net: &MddNet) -> (u64, u64, u64, u64) {
        use std::collections::HashSet;
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        seen.insert(net.initial_marking.clone());
        let mut frontier = vec![net.initial_marking.clone()];
        let mut edges: u64 = 0;
        let mut max_in_place: u64 = net.initial_marking.iter().copied().max().unwrap_or(0);
        let mut max_sum: u64 = net.initial_marking.iter().sum();
        while let Some(m) = frontier.pop() {
            for tr in &net.transitions {
                if !m.iter().zip(&tr.pre).all(|(mv, pv)| mv >= pv) {
                    continue;
                }
                let mut next = m.clone();
                let mut ok = true;
                for p in 0..next.len() {
                    let v = next[p] - tr.pre[p] + tr.post[p];
                    if v > net.bounds[p] {
                        ok = false;
                        break;
                    }
                    next[p] = v;
                }
                if !ok {
                    continue;
                }
                edges += 1;
                if seen.insert(next.clone()) {
                    let s: u64 = next.iter().sum();
                    let mxp = next.iter().copied().max().unwrap_or(0);
                    max_sum = max_sum.max(s);
                    max_in_place = max_in_place.max(mxp);
                    frontier.push(next);
                }
            }
        }
        (seen.len() as u64, edges, max_in_place, max_sum)
    }

    fn check(net: &MddNet) {
        let (rc, ec, mip, msum) = bfs_metrics(net);
        let m = net.state_space_metrics(None).expect("metrics ok");
        assert_eq!(m.state_count, Some(rc), "state_count");
        assert_eq!(m.state_count_u128, rc as u128, "state_count_u128");
        // The bignum field is the exact authoritative count and equals the
        // narrowed u128 on every in-range case (IDENTICAL-on-u128 proof).
        assert_eq!(m.state_count_big, BigUint::from(rc), "state_count_big");
        assert_eq!(
            big_to_u128(&m.state_count_big),
            Some(m.state_count_u128),
            "big narrows to the u128 field",
        );
        assert_eq!(m.edge_count, ec as u128, "edge_count");
        assert_eq!(m.edge_count_big, BigUint::from(ec), "edge_count_big");
        assert_eq!(m.max_token_in_place, mip, "max_token_in_place");
        assert_eq!(m.max_token_sum, msum, "max_token_sum");
        // The relprod set-build fallback must produce the IDENTICAL metric
        // bundle on the small nets where it also converges.
        let mr = net
            .state_space_metrics_relprod(None)
            .expect("relprod metrics ok");
        assert_eq!(mr.state_count_big, m.state_count_big, "relprod state_count");
        assert_eq!(mr.edge_count_big, m.edge_count_big, "relprod edge_count");
        assert_eq!(
            mr.max_token_in_place, m.max_token_in_place,
            "relprod max_token_in_place"
        );
        assert_eq!(mr.max_token_sum, m.max_token_sum, "relprod max_token_sum");
    }

    #[test]
    fn swap_metrics() {
        check(&MddNet {
            bounds: vec![1, 1],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])],
        });
    }

    #[test]
    fn chain_metrics() {
        check(&MddNet {
            bounds: vec![2, 2, 2],
            initial_marking: vec![2, 0, 0],
            transitions: vec![
                t(vec![1, 0, 0], vec![0, 1, 0]),
                t(vec![0, 1, 0], vec![0, 0, 1]),
            ],
        });
    }

    #[test]
    fn independent_counters_metrics() {
        check(&MddNet {
            bounds: vec![3, 3],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        });
    }

    #[test]
    fn weighted_truncating_metrics() {
        check(&MddNet {
            bounds: vec![1, 2],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 2]), t(vec![0, 1], vec![1, 0])],
        });
    }

    #[test]
    fn conserved_ring_metrics() {
        check(&MddNet {
            bounds: vec![1, 1, 1],
            initial_marking: vec![1, 0, 0],
            transitions: vec![
                t(vec![1, 0, 0], vec![0, 1, 0]),
                t(vec![0, 1, 0], vec![0, 0, 1]),
                t(vec![0, 0, 1], vec![1, 0, 0]),
            ],
        });
    }

    #[test]
    fn no_transitions_single_marking() {
        check(&MddNet {
            bounds: vec![2, 3],
            initial_marking: vec![1, 2],
            transitions: vec![],
        });
    }

    /// >u128 metric CARRIER on the reduced reachable set, built directly.
    ///
    /// 130 INDEPENDENT 1-safe places, each fed by its own always-enabled source
    /// transition (`pre = 0`, `post[i] = 1`). Each place is independently
    /// reachable at 0 or 1, so the reachable set is the full cube `{0,1}^130`
    /// and `|R| = 2^130 ≈ 1.36e39`, FAR beyond `u128::MAX ≈ 3.4e38`. The cube's
    /// REDUCED MDD is the terminal `ONE` (every level free ⇒ redundant-node
    /// suppressed), so we feed that reduced root straight to
    /// `metrics_from_reachable`. This exercises the bignum count/edge carrier at
    /// >u128 magnitude — the EXACT `2^130` is reported and the narrowed
    /// > `u128`/`u64` fields saturate to their markers — in `O(places²)`, WITHOUT
    /// > the exponential wide-free SET BUILD (which is the saturation/relprod
    /// > engine's pre-existing cost on always-enabled wide nets, and is separately
    /// > deadline-bounded; see `wide_free_net_build_declines_under_deadline`). The
    /// > build engines are pinned to the BFS oracle by `tests/crosscheck_bfs.rs`;
    /// > here we pin the metric EXTRACTION carrier. Hand-checked closed form:
    /// > |R| = 2^130, edges = 130 · 2^129 (Σ over markings of #places at 0),
    /// > max-in-place = 1, max-sum = 130.
    #[test]
    fn above_u128_metric_carrier_reports_exact_bignum() {
        let n = 130usize;
        let bounds = vec![1u64; n];
        let mut transitions = Vec::with_capacity(n);
        for i in 0..n {
            let mut post = vec![0u64; n];
            post[i] = 1;
            transitions.push(t(vec![0u64; n], post));
        }
        let net = MddNet {
            bounds: bounds.clone(),
            initial_marking: vec![0u64; n],
            transitions,
        };
        // The reduced full-cube reachable set over `bounds` is the terminal ONE.
        let mut store = MddStore::new(bounds);
        let reach = MddRef::ONE;
        let m = net
            .metrics_from_reachable(&mut store, reach, 1)
            .expect("metric carrier on the reduced reachable set");
        let two = BigUint::from(2u32);
        assert_eq!(m.state_count_big, two.pow(130), "|R| = 2^130 exactly");
        assert!(
            m.state_count_big > BigUint::from(u128::MAX),
            "the count is genuinely > u128::MAX",
        );
        // Narrowed fields saturate (markers); state_count (u64) is None.
        assert_eq!(m.state_count, None, "does not fit u64");
        assert_eq!(
            m.state_count_u128,
            u128::MAX,
            "saturated marker (does not fit u128)"
        );
        // Edges: Σ over markings of #(places currently at 0) = 130 · 2^129.
        let expected_edges = BigUint::from(130u32) * two.pow(129);
        assert_eq!(m.edge_count_big, expected_edges, "edges = 130 · 2^129");
        // max_token_in_place = 1, max_token_sum = 130 (all places at 1).
        assert_eq!(m.max_token_in_place, 1);
        assert_eq!(m.max_token_sum, 130);
    }

    /// The wide-free SET BUILD is exponential — each fixpoint round materializes
    /// the wide product set before it reduces — so under a wall-clock deadline
    /// the metric path must DECLINE fail-closed, never hang or overrun. (This
    /// build cost is pre-existing engine behavior on always-enabled wide nets,
    /// exposed once the bignum count carrier removed the implicit `u128`-overflow
    /// early-exit; production ALWAYS passes a deadline, so it declines cleanly.)
    #[test]
    fn wide_free_net_build_declines_under_deadline() {
        let n = 130usize;
        let bounds = vec![1u64; n];
        let mut transitions = Vec::with_capacity(n);
        for i in 0..n {
            let mut post = vec![0u64; n];
            post[i] = 1;
            transitions.push(t(vec![0u64; n], post));
        }
        let net = MddNet {
            bounds,
            initial_marking: vec![0u64; n],
            transitions,
        };
        let start = std::time::Instant::now();
        let r = net.state_space_metrics(Some(start + std::time::Duration::from_secs(3)));
        let elapsed = start.elapsed();
        assert!(
            r.is_err(),
            "wide-free build must decline (fail-closed) under a deadline, got {r:?}",
        );
        assert!(
            elapsed < std::time::Duration::from_secs(12),
            "deadline must be honored — no overrun (elapsed {elapsed:?})",
        );
    }

    #[test]
    fn high_bound_conserved_shuttle_metrics() {
        // p0 + p1 = 17: 18 markings; edges, max-in-place, max-sum all exact.
        check(&MddNet {
            bounds: vec![17, 17],
            initial_marking: vec![17, 0],
            transitions: vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])],
        });
    }
}
