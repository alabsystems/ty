// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Symbolic relational-product image — fire a transition over a whole set MDD
//! at once, without enumerating markings.
//!
//! # The relation is functional, so the image is a remap
//!
//! A bounded P/T transition `t` defines, *per place*, a partial deterministic
//! function on token counts:
//!
//! ```text
//!     next[p] = m[p] - pre[p] + post[p]      (guard: m[p] >= pre[p])
//! ```
//!
//! and the firing is rejected wholesale if any `next[p] > bound[p]` — exactly
//! the [`crate::reach`] / `tla-dd` BFS firing rule. Because each place's
//! next-value is a *function* of its current value (not a general relation),
//! the transition relation factorises into one independent per-level remap.
//! That means the image (relational product) over the set MDD is a structural
//! recursion that, at every level `l`, **moves** the child reached by current
//! value `v` to the slot for `v' = v - pre[l] + post[l]`:
//!
//! - source value `v < pre[l]` → guard fails → that branch contributes nothing;
//! - target value `v' > bound[l]` → bound-truncated out → contributes nothing;
//! - otherwise the recursed image of the `v` child is placed at edge `v'`.
//!
//! Several distinct source values can map to the same target value (when
//! `pre[l] != post[l]` shifts the domain), so the per-level remap **unions**
//! the images that collide on a target slot. This is the only place general
//! `apply(∪)` is needed inside the image.
//!
//! # Why this is a true symbolic image (not enumerate-and-fire)
//!
//! The recursion keys on `(node)` and is fully cached per transition, so the
//! work is proportional to the number of **distinct MDD nodes** touched, not
//! the number of markings the set encodes. A set of `10^9` markings sharing a
//! handful of nodes is fired in a handful of node visits. This is the property
//! the explicit [`crate::reach`] kernel lacks (it enumerates `|frontier|`),
//! and it is what lets the relational-product BFS fixpoint scale.
//!
//! # Soundness
//!
//! The image is the exact set `{ fire(t, m) : m ∈ S, t enabled at m,
//! fire in-bounds }`. The differential battery pins the relational-product
//! BFS reachable count to both the explicit MDD kernel and the `tla-dd` BFS
//! oracle with 0 disagreements. The image carries the same fail-closed
//! node-budget the rest of the crate uses.

use crate::node::{MddRef, MddStore};
use crate::reach::MddTransition;
use std::collections::HashMap;

/// A per-transition image operator with its own recursion cache.
///
/// One [`Imager`] is built per transition (its `pre`/`post` are baked in) and
/// reused across a whole image computation, so the structural-recursion cache
/// is warm for every node shared in the set MDD.
pub(crate) struct Imager<'t> {
    pre: &'t [u64],
    post: &'t [u64],
    /// Recursion cache: set-node → its image node, valid for *this*
    /// transition only. Keyed by the source node ref.
    cache: HashMap<MddRef, MddRef>,
}

impl<'t> Imager<'t> {
    pub(crate) fn new(t: &'t MddTransition) -> Self {
        Self {
            pre: &t.pre,
            post: &t.post,
            cache: HashMap::new(),
        }
    }

    /// Image of the set rooted at `node`: the set of successors obtained by
    /// firing this transition at every marking in the set (guard + bound
    /// enforced per level). Returns [`MddRef::ZERO`] when no marking fires.
    pub(crate) fn image(&mut self, store: &mut MddStore, node: MddRef) -> MddRef {
        self.image_at(store, node, 0)
    }

    /// Image of `node` interpreted as a set over levels `from..num_levels`,
    /// i.e. the levels above `from` are not part of this sub-MDD at all (the
    /// caller — saturation — fires a banded event whose entire support lies at
    /// or below `from`). Levels above `from` are NOT remapped. Used by the
    /// saturation driver to fire an event at the node sitting at its band
    /// level.
    pub(crate) fn image_from_level(
        &mut self,
        store: &mut MddStore,
        node: MddRef,
        from: usize,
    ) -> MddRef {
        debug_assert!(self.pre.iter().take(from).all(|&p| p == 0));
        debug_assert!(self.post.iter().take(from).all(|&p| p == 0));
        self.image_at(store, node, from)
    }

    /// Image of `node` viewed *at* `level` (the level the caller is currently
    /// remapping). When the node sits deeper than `level`, `level` is a free
    /// place on the set side: the recursion still has to apply this level's
    /// remap, treating the current value as unconstrained.
    fn image_at(&mut self, store: &mut MddStore, node: MddRef, level: usize) -> MddRef {
        let n = store.num_levels();
        if node.is_zero() {
            return MddRef::ZERO;
        }
        if level == n {
            // Past the last place level: a surviving path is ONE iff we got
            // here from ONE (ZERO is pruned above). All places have been
            // remapped, so the image of this leaf is itself.
            debug_assert!(node.is_one());
            return MddRef::ONE;
        }

        // Cache only the canonical (node-at-its-own-level) entry. A node seen
        // at a *deeper* level than its own is being treated as free above it;
        // those frames recurse straight back to the same node at `level + 1`,
        // so a separate cache slot would be redundant.
        let node_level = store.level_of(node);
        let at_own_level = !node.is_one() && node_level as usize == level;
        if at_own_level {
            if let Some(&hit) = self.cache.get(&node) {
                return hit;
            }
        }

        let dom = store.domain_size(level as u32);
        // Build the children of the image node at this level. Every target
        // slot starts empty (ZERO) and accumulates the unioned image of each
        // source value that maps to it.
        let mut out_children = vec![MddRef::ZERO; dom];
        let pre_l = self.pre[level];
        let post_l = self.post[level];

        for v in 0..dom as u64 {
            // Guard: the source value must cover the consumed tokens.
            if v < pre_l {
                continue;
            }
            let target = v - pre_l + post_l;
            // Bound-truncation: firing rejected wholesale if out of range.
            if target >= dom as u64 {
                continue;
            }
            // The set-side child reached by current value `v` at this level.
            let child = if node.is_one() || node_level as usize > level {
                // Free/skipped level on the set side ⇒ value `v` stays at the
                // same node, which is unconstrained below this level.
                node
            } else {
                store.child(node, v)
            };
            if child.is_zero() {
                continue;
            }
            let img_child = self.image_at(store, child, level + 1);
            if img_child.is_zero() {
                continue;
            }
            // Collisions: several source `v` may land on the same `target`.
            out_children[target as usize] = store.union(out_children[target as usize], img_child);
        }

        let result = store.get_node(level as u32, out_children);
        if at_own_level {
            self.cache.insert(node, result);
        }
        result
    }
}

/// Image of `set` under a single transition `t` — the public entry the
/// relational-product fixpoint calls. Pure successor set (does not include the
/// source markings); the caller unions it back in.
#[must_use]
pub(crate) fn transition_image(store: &mut MddStore, set: MddRef, t: &MddTransition) -> MddRef {
    if set.is_zero() {
        return MddRef::ZERO;
    }
    let mut imager = Imager::new(t);
    imager.image(store, set)
}

/// A per-transition **backward** image (pre-image) operator with its own cache.
///
/// # The pre-image is the inverse remap — and it is a function, not a relation
///
/// The forward image (above) SCATTERS a source value `v` to target `v' = v -
/// pre[l] + post[l]`, with several source values colliding on a target slot
/// (hence the per-level union). The pre-image GATHERS: a current (source) value
/// `v` survives at level `l` iff it could fire `t` into some marking that the
/// target `set` accepts. Per place that is exactly:
///
/// ```text
///     v >= pre[l]                              (guard: enough tokens to consume)
///     v' = v - pre[l] + post[l] <= bound[l]    (successor in-bounds — bound-truncated out otherwise)
///     set's child along edge v' is non-ZERO    (the successor is in the target set)
/// ```
///
/// Because `v'` is a *function* of `v` (each source value maps to exactly one
/// target slot to read), the per-level remap is a plain gather — there is **no
/// collision and no union**, which makes the pre-image structurally simpler
/// than the forward image. The guard/bound conditions are byte-identical to the
/// forward fire (`reach::fire`) and `build_fireable_set`, and skipped-level
/// handling on the `set` side matches `Imager::image_at` + `MddStore::edge_at`
/// (a `set` node deeper than the current level is unconstrained — every value
/// stays at the same node).
///
/// # Soundness
///
/// `preimage(set, t)` is the exact set `{ m : t enabled at m, fire(t,m) is
/// in-bounds, fire(t,m) ∈ set }`. The differential battery pins the union over
/// transitions to the explicit predecessor set on the reachable graph, and
/// triangulates `m ∈ preimage(set,t)  ⇔  image({m},t) ∩ set ≠ ∅` against the
/// battle-tested forward image.
pub(crate) struct Preimager<'t> {
    pre: &'t [u64],
    post: &'t [u64],
    /// The target set whose predecessors we gather (root, used to read the
    /// successor child along edge `v'` at each level). Held by ref so the cache
    /// is valid only for *this* (transition, set) pair — a fresh `Preimager` is
    /// built per call, exactly like `Imager`.
    set: MddRef,
    /// Recursion cache: (set-node-at-this-level) → its pre-image node, valid for
    /// *this* transition + target only. Keyed by the SOURCE set node ref read at
    /// its own level.
    cache: HashMap<MddRef, MddRef>,
}

impl<'t> Preimager<'t> {
    pub(crate) fn new(t: &'t MddTransition, set: MddRef) -> Self {
        Self {
            pre: &t.pre,
            post: &t.post,
            set,
            cache: HashMap::new(),
        }
    }

    /// Pre-image of the whole target set: the set of markings that fire this
    /// transition (in bounds) into the target. Returns [`MddRef::ZERO`] when no
    /// marking is a predecessor.
    pub(crate) fn preimage(&mut self, store: &mut MddStore) -> MddRef {
        let set = self.set;
        self.preimage_at(store, set, 0)
    }

    /// Pre-image where `set_node` is the target set viewed *at* `level` (the
    /// level whose remap we are currently inverting). When `set_node` sits
    /// deeper than `level`, `level` is a free place on the target side: every
    /// successor value `v'` leads back to the same `set_node` (stay-put),
    /// matching `MddStore::edge_at`.
    fn preimage_at(&mut self, store: &mut MddStore, set_node: MddRef, level: usize) -> MddRef {
        let n = store.num_levels();
        if set_node.is_zero() {
            // The target sub-set is empty below here ⇒ no predecessors.
            return MddRef::ZERO;
        }
        if level == n {
            // Past the last place level. A surviving target path is ONE (ZERO
            // is pruned above), and all levels have been inverted, so the
            // pre-image of this leaf is itself.
            debug_assert!(set_node.is_one());
            return MddRef::ONE;
        }

        // Cache only the canonical (set-node-at-its-own-level) entry — same
        // policy as `Imager::image_at`: a node seen deeper than its own level is
        // treated as free above it and recurses straight back to the same node.
        let set_level = store.level_of(set_node);
        let at_own_level = !set_node.is_one() && set_level as usize == level;
        if at_own_level {
            if let Some(&hit) = self.cache.get(&set_node) {
                return hit;
            }
        }

        let dom = store.domain_size(level as u32);
        let pre_l = self.pre[level];
        let post_l = self.post[level];

        // Build the pre-image node's children at this level. Edge `v` (a SOURCE
        // value) is non-ZERO iff `v` can fire this level (guard + bound) into a
        // target value `v'` whose surviving target sub-MDD has a non-empty
        // pre-image below. No collision: each `v` reads exactly one `v'`.
        let mut out_children = vec![MddRef::ZERO; dom];
        for v in 0..dom as u64 {
            // Guard: enough tokens to consume (identical to forward fire).
            if v < pre_l {
                continue;
            }
            let target = v - pre_l + post_l;
            // Bound-truncation: a firing whose successor is out of range is not
            // in the (bounded) state space — exactly the forward rule.
            if target >= dom as u64 {
                continue;
            }
            // The target-side child reached by the SUCCESSOR value `v'` at this
            // level. A target node deeper than `level` (or ONE) is free here:
            // value `v'` stays at the same node (matches `edge_at` stay-put).
            let succ_child = if set_node.is_one() || set_level as usize > level {
                set_node
            } else {
                store.child(set_node, target)
            };
            if succ_child.is_zero() {
                continue;
            }
            let pre_child = self.preimage_at(store, succ_child, level + 1);
            if pre_child.is_zero() {
                continue;
            }
            // Functional gather: each source value `v` lands on its own slot.
            out_children[v as usize] = pre_child;
        }

        let result = store.get_node(level as u32, out_children);
        if at_own_level {
            self.cache.insert(set_node, result);
        }
        result
    }
}

/// Backward image (pre-image) of `set` under a single transition `t`: the set
/// of markings that fire `t` (in bounds) into `set`. Pure predecessor set; the
/// `pre_e` caller intersects it with the reachable set. Mirrors
/// [`transition_image`].
#[must_use]
pub(crate) fn transition_preimage(store: &mut MddStore, set: MddRef, t: &MddTransition) -> MddRef {
    if set.is_zero() {
        return MddRef::ZERO;
    }
    let mut pre = Preimager::new(t, set);
    pre.preimage(store)
}

/// The SHALLOWEST place level this transition touches — the smallest place
/// index with a non-zero `pre` (a guard, so the event reads that level) or
/// `post` (a write). In this MDD, level 0 is the top (root) and level `n-1`
/// the bottom (terminal-ward), so the shallowest touched level is the event's
/// `Top` in the saturation sense: firing the event at the node sitting at this
/// level lets the per-level remap recurse *downward* over the event's entire
/// support. `None` for an empty transition (no guard, no effect) — a no-op
/// that saturation skips.
///
/// Soundness: banding by the shallowest touched level is the load-bearing
/// quantity for saturation. If it were too deep, the event would be fired at a
/// node below part of its own support, silently dropping firings and
/// under-counting `|R|`. The differential battery pins the saturated count to
/// the explicit oracle precisely to catch any such banding error.
#[must_use]
pub(crate) fn shallowest_level(t: &MddTransition) -> Option<usize> {
    (0..t.pre.len()).find(|&p| t.pre[p] != 0 || t.post[p] != 0)
}

#[cfg(test)]
mod preimage_tests {
    use super::*;
    use std::collections::HashSet;

    fn t(pre: Vec<u64>, post: Vec<u64>) -> MddTransition {
        MddTransition { pre, post }
    }

    /// Enumerate every full marking over `bounds`.
    fn all_markings(bounds: &[u64]) -> Vec<Vec<u64>> {
        let mut out = vec![vec![]];
        for &b in bounds {
            let mut next = Vec::new();
            for partial in &out {
                for v in 0..=b {
                    let mut m = partial.clone();
                    m.push(v);
                    next.push(m);
                }
            }
            out = next;
        }
        out
    }

    /// Explicit forward fire (guard + bound-truncation), matching `reach::fire`.
    fn fire(bounds: &[u64], m: &[u64], t: &MddTransition) -> Option<Vec<u64>> {
        if !m.iter().zip(&t.pre).all(|(mv, pv)| mv >= pv) {
            return None;
        }
        let mut next = m.to_vec();
        for p in 0..next.len() {
            let v = next[p] - t.pre[p] + t.post[p];
            if v > bounds[p] {
                return None;
            }
            next[p] = v;
        }
        Some(next)
    }

    /// Build a set MDD from an explicit list of markings.
    fn set_of(store: &mut MddStore, markings: &[Vec<u64>]) -> MddRef {
        let mut acc = MddRef::ZERO;
        for m in markings {
            let s = store.singleton(m);
            acc = store.union(acc, s);
        }
        acc
    }

    /// Materialize an MDD set back to the explicit set of markings.
    fn members(store: &MddStore, root: MddRef, bounds: &[u64]) -> HashSet<Vec<u64>> {
        all_markings(bounds)
            .into_iter()
            .filter(|m| {
                // Membership: walk the singleton ∩ root and check non-zero is
                // overkill; instead intersect a freshly built singleton. Use a
                // direct walk via the public count by intersecting.
                contains(store, root, m)
            })
            .collect()
    }

    /// Is marking `m` in the set rooted at `root`? Follows the MDD respecting
    /// skipped (redundant) levels.
    fn contains(store: &MddStore, root: MddRef, m: &[u64]) -> bool {
        let mut node = root;
        for (level, &v) in m.iter().enumerate() {
            if node.is_zero() {
                return false;
            }
            if node.is_one() {
                // All remaining levels free ⇒ in the set.
                return true;
            }
            let nl = store.level_of(node) as usize;
            if nl > level {
                // Level skipped (free) ⇒ value v allowed, stay at node.
                continue;
            }
            debug_assert_eq!(nl, level);
            node = store.child(node, v);
        }
        node.is_one()
    }

    fn nets() -> Vec<MddTransition> {
        vec![
            t(vec![1, 0], vec![0, 1]),
            t(vec![0, 1], vec![1, 0]),
            t(vec![1, 0], vec![0, 2]), // weighted (with bound truncation)
            t(vec![0], vec![1]),       // single-place counter
        ]
    }

    /// `transition_preimage(S, t)` equals the explicit predecessor set
    /// `{ m : fire(t,m) ∈ S }` over the full bounded universe, on a variety of
    /// transitions and target sets.
    #[test]
    fn preimage_equals_explicit_predecessors() {
        let bounds_cases: Vec<Vec<u64>> = vec![vec![1, 1], vec![2, 2], vec![3], vec![2, 1, 2]];
        for bounds in bounds_cases {
            let universe = all_markings(&bounds);
            // A handful of target sets: each prefix, each singleton, all, none.
            let mut targets: Vec<Vec<Vec<u64>>> = vec![vec![], universe.clone()];
            for m in &universe {
                targets.push(vec![m.clone()]);
            }
            // Half-sets too.
            targets.push(universe.iter().step_by(2).cloned().collect());

            for tr in nets() {
                if tr.pre.len() != bounds.len() {
                    continue;
                }
                for target in &targets {
                    let mut store = MddStore::new(bounds.clone());
                    let set = set_of(&mut store, target);
                    let pre = transition_preimage(&mut store, set, &tr);
                    let got = members(&store, pre, &bounds);

                    let target_set: HashSet<Vec<u64>> = target.iter().cloned().collect();
                    let want: HashSet<Vec<u64>> = universe
                        .iter()
                        .filter(|m| {
                            fire(&bounds, m, &tr).is_some_and(|succ| target_set.contains(&succ))
                        })
                        .cloned()
                        .collect();
                    assert_eq!(
                        got, want,
                        "preimage mismatch bounds={bounds:?} tr={tr:?} target={target:?}"
                    );
                }
            }
        }
    }

    /// Triangulation against the battle-tested forward image:
    /// `m ∈ preimage(S, t)  ⇔  image({m}, t) ∩ S ≠ ∅`.
    #[test]
    fn preimage_triangulates_forward_image() {
        let bounds_cases: Vec<Vec<u64>> = vec![vec![1, 1], vec![2, 2], vec![3], vec![2, 1, 2]];
        for bounds in bounds_cases {
            let universe = all_markings(&bounds);
            let targets: Vec<Vec<Vec<u64>>> = vec![
                universe.clone(),
                universe.iter().step_by(2).cloned().collect(),
                universe.iter().rev().take(2).cloned().collect(),
            ];
            for tr in nets() {
                if tr.pre.len() != bounds.len() {
                    continue;
                }
                for target in &targets {
                    let mut store = MddStore::new(bounds.clone());
                    let set = set_of(&mut store, target);
                    let pre = transition_preimage(&mut store, set, &tr);

                    for m in &universe {
                        let in_pre = contains(&store, pre, m);
                        // image({m}, t) ∩ S ≠ ∅: build the singleton image and
                        // check overlap with the target.
                        let single = store.singleton(m);
                        let img = transition_image(&mut store, single, &tr);
                        let inter = store.intersect(img, set);
                        let overlap = !inter.is_zero();
                        assert_eq!(
                            in_pre, overlap,
                            "triangulation mismatch m={m:?} bounds={bounds:?} tr={tr:?}"
                        );
                    }
                }
            }
        }
    }

    /// Union of per-transition pre-images is the predecessor set under the WHOLE
    /// relation: `⋃_t preimage(S,t) = { m : ∃t. fire(t,m) ∈ S }`.
    #[test]
    fn preimage_union_is_whole_relation_predecessors() {
        let bounds = vec![2, 2];
        let trs = vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])];
        let universe = all_markings(&bounds);
        let target: Vec<Vec<u64>> = vec![vec![0, 1], vec![1, 1]];
        let target_set: HashSet<Vec<u64>> = target.iter().cloned().collect();

        let mut store = MddStore::new(bounds.clone());
        let set = set_of(&mut store, &target);
        let mut acc = MddRef::ZERO;
        for tr in &trs {
            let pre = transition_preimage(&mut store, set, tr);
            acc = store.union(acc, pre);
        }
        let got = members(&store, acc, &bounds);
        let want: HashSet<Vec<u64>> = universe
            .iter()
            .filter(|m| {
                trs.iter()
                    .any(|tr| fire(&bounds, m, tr).is_some_and(|s| target_set.contains(&s)))
            })
            .cloned()
            .collect();
        assert_eq!(got, want);
    }
}
