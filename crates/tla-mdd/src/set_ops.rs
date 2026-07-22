// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Set operations over the MDD: build a singleton, union, and the exact
//! reachable-state **count**.
//!
//! Everything here treats an `MddRef` as the characteristic function of a set
//! of full markings (one token value per place level). The two soundness-
//! critical operations are [`MddStore::union`] (used by the fixpoint to
//! accumulate the reachable set) and [`MddStore::count_markings`] (the exact
//! count the engine reports).
//!
//! ## Skipped levels
//!
//! Because redundant nodes are suppressed (see [`crate::node`]), an edge can
//! jump over levels. A skipped level means "this place is unconstrained — any
//! value in `0..=bound` is allowed". `union` materializes the skipped node on
//! demand (`expand_to`) so both operands are compared level-by-level, and
//! `count_markings` multiplies in the skipped levels' domain sizes.

use crate::node::{ApplyOp, MddRef, MddStore, TERMINAL_LEVEL};
use std::collections::{HashMap, HashSet};
use tla_bignum::{BigUint, One, ToPrimitive, Zero};

impl MddStore {
    /// Characteristic MDD of the linear inequality
    /// `Σ_l coeffs[l] · m[l] <= k` over the bounded marking universe, EXACT.
    ///
    /// `coeffs[l]` is the integer coefficient (possibly negative) of place
    /// `l`'s token count; `k` is the right-hand constant. The set
    /// `{ m : Σ_l coeffs[l]·m[l] <= k }` is built by a memoized top-down DP that
    /// carries the partial sum chosen above the current level and prunes whole
    /// subtrees the moment the inequality is decided regardless of the remaining
    /// levels (so the MDD stays compact — one node per distinguishable partial
    /// sum per level). This is the construction the petri-side CTL atom lowering
    /// uses for `IntLe` predicates; it is the multiset-sum semantics
    /// (`tla_dd`'s `eval_dd_int_expr` / `compile_int_le`), exact on every
    /// bounded marking.
    ///
    /// # Panics
    /// Debug-asserts `coeffs.len() == num_levels()`.
    #[must_use]
    pub fn linear_le_set(&mut self, coeffs: &[i128], k: i128) -> MddRef {
        debug_assert_eq!(coeffs.len(), self.num_levels());
        let n = self.num_levels();
        // Per-level [min,max] contribution from level..n (for early decision).
        // suffix_min[l] = Σ_{j>=l} min(0, coeffs[j])·0 or coeffs[j]·bound — i.e.
        // the smallest / largest the tail can add.
        let mut suffix_min = vec![0i128; n + 1];
        let mut suffix_max = vec![0i128; n + 1];
        for l in (0..n).rev() {
            let c = coeffs[l];
            let b = self.bounds[l] as i128;
            // value v ∈ [0, b]; contribution c·v ranges over [min(0, c·b), max(0, c·b)].
            let (lo, hi) = if c >= 0 { (0, c * b) } else { (c * b, 0) };
            suffix_min[l] = suffix_min[l + 1] + lo;
            suffix_max[l] = suffix_max[l + 1] + hi;
        }
        let mut memo: HashMap<(usize, i128), MddRef> = HashMap::new();
        self.linear_le_rec(coeffs, k, 0, 0, &suffix_min, &suffix_max, &mut memo)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_le_rec(
        &mut self,
        coeffs: &[i128],
        k: i128,
        level: usize,
        acc: i128,
        suffix_min: &[i128],
        suffix_max: &[i128],
        memo: &mut HashMap<(usize, i128), MddRef>,
    ) -> MddRef {
        // Early decision independent of the remaining levels:
        //   acc + suffix_max[level] <= k  ⇒ always satisfied ⇒ ONE.
        //   acc + suffix_min[level] >  k  ⇒ never satisfied  ⇒ ZERO.
        if acc + suffix_max[level] <= k {
            return MddRef::ONE;
        }
        if acc + suffix_min[level] > k {
            return MddRef::ZERO;
        }
        // Undecided ⇒ there is at least one more level (a fully-consumed prefix
        // would have been decided by the bounds above, since suffix_*[n] == 0).
        debug_assert!(level < self.num_levels());
        if let Some(&hit) = memo.get(&(level, acc)) {
            return hit;
        }
        let dom = self.domain_size(level as u32);
        let c = coeffs[level];
        let mut children = Vec::with_capacity(dom);
        for v in 0..dom as u64 {
            let next_acc = acc + c * v as i128;
            let child =
                self.linear_le_rec(coeffs, k, level + 1, next_acc, suffix_min, suffix_max, memo);
            children.push(child);
        }
        let result = self.get_node(level as u32, children);
        memo.insert((level, acc), result);
        result
    }

    /// Characteristic MDD of "this transition's GUARD holds" — the set
    /// `{ m : m[l] >= pre[l] for all l }`. Guard-ONLY (no bound truncation on a
    /// successor), matching the `IsFireable` atom convention (`tla_dd`'s
    /// `compile_is_fireable`: enabledness is the guard alone). A per-level chain
    /// where each level allows `v >= pre[l]`. Built bottom-up.
    ///
    /// # Panics
    /// Debug-asserts `pre.len() == num_levels()`.
    #[must_use]
    pub fn guard_set(&mut self, pre: &[u64]) -> MddRef {
        debug_assert_eq!(pre.len(), self.num_levels());
        let n = self.num_levels();
        let mut acc = MddRef::ONE;
        for level in (0..n).rev() {
            let dom = self.domain_size(level as u32);
            let mut children = vec![MddRef::ZERO; dom];
            for v in 0..dom as u64 {
                if v >= pre[level] {
                    children[v as usize] = acc;
                }
            }
            acc = self.get_node(level as u32, children);
        }
        acc
    }

    /// Build the MDD for a single marking: the singleton set `{marking}`.
    ///
    /// `marking[l]` is the token count for place/level `l`; it must satisfy
    /// `marking[l] <= bounds[l]`. Returns [`MddRef::ZERO`] only if the marking
    /// is out of range (caller treats that as "not representable", never as a
    /// silent membership). On a valid marking the result is a chain of one
    /// node per level, edge `marking[l]` to the next level and every other
    /// edge to `ZERO`.
    ///
    /// # Panics
    /// Debug-asserts `marking.len() == num_levels()`.
    #[must_use]
    pub fn singleton(&mut self, marking: &[u64]) -> MddRef {
        debug_assert_eq!(marking.len(), self.num_levels());
        // Range check up front: an out-of-range marking is the empty set, not
        // a wrong membership (fail-closed).
        for (l, &v) in marking.iter().enumerate() {
            if v > self.bounds[l] {
                return MddRef::ZERO;
            }
        }
        // Build bottom-up so each child already exists when its parent is
        // interned.
        let mut acc = MddRef::ONE;
        for level in (0..self.num_levels()).rev() {
            let dom = self.domain_size(level as u32);
            let chosen = marking[level] as usize;
            let mut children = vec![MddRef::ZERO; dom];
            children[chosen] = acc;
            acc = self.get_node(level as u32, children);
        }
        acc
    }

    /// Set union of two MDDs. Canonical: equal sets yield the same `MddRef`.
    ///
    /// Implements the standard recursive `apply(∪)` with an apply cache, plus
    /// on-demand level expansion so a skipped (redundant) level on one operand
    /// is compared against the explicit level of the other.
    #[must_use]
    pub fn union(&mut self, a: MddRef, b: MddRef) -> MddRef {
        // PERSISTENT apply cache: kept warm across calls (taken out for the
        // duration so the recursion can borrow the store for get_node, then put
        // back), instead of a fresh map per call. Scrubbed on gc + size-capped,
        // so it is always sound and bounded (see MddStore::apply_caches).
        let mut cache = self.take_apply_cache(ApplyOp::Union);
        let result = self.union_rec(a, b, &mut cache);
        self.put_apply_cache(ApplyOp::Union, cache);
        result
    }

    fn union_rec(
        &mut self,
        a: MddRef,
        b: MddRef,
        cache: &mut HashMap<(MddRef, MddRef), MddRef>,
    ) -> MddRef {
        // Terminal shortcuts. ONE absorbs (whole subtree in the set); ZERO is
        // the identity.
        if a == b {
            return a;
        }
        if a.is_one() || b.is_one() {
            return MddRef::ONE;
        }
        if a.is_zero() {
            return b;
        }
        if b.is_zero() {
            return a;
        }

        // Order the pair so the cache key is symmetric (union is commutative).
        let key = if a.0 <= b.0 { (a, b) } else { (b, a) };
        if let Some(&hit) = cache.get(&key) {
            return hit;
        }

        // Both interior. Recurse at the shallower (smaller) level; the deeper
        // operand is "unconstrained" at that level, expand it.
        let la = self.level_of(a);
        let lb = self.level_of(b);
        let level = la.min(lb);
        let dom = self.domain_size(level);

        let mut children = Vec::with_capacity(dom);
        for v in 0..dom as u64 {
            let ca = self.edge_at(a, level, v);
            let cb = self.edge_at(b, level, v);
            let child = self.union_rec(ca, cb, cache);
            children.push(child);
        }
        let result = self.get_node(level, children);
        cache.insert(key, result);
        result
    }

    /// Set intersection of two MDDs. Canonical: equal sets yield the same
    /// `MddRef`. Purely additive (interns new nodes; never mutates `a`/`b`),
    /// so both operands stay valid afterward.
    ///
    /// Implements the standard recursive `apply(∩)` with an apply cache, plus
    /// the same on-demand level expansion as [`MddStore::union`]: a skipped
    /// (redundant) level on one operand is compared against the explicit level
    /// of the other by treating the skipped level as unconstrained (the node
    /// stays put).
    #[must_use]
    pub fn intersect(&mut self, a: MddRef, b: MddRef) -> MddRef {
        // Persistent apply cache (own slot; see MddStore::apply_caches).
        let mut cache = self.take_apply_cache(ApplyOp::Intersect);
        let result = self.intersect_rec(a, b, &mut cache);
        self.put_apply_cache(ApplyOp::Intersect, cache);
        result
    }

    fn intersect_rec(
        &mut self,
        a: MddRef,
        b: MddRef,
        cache: &mut HashMap<(MddRef, MddRef), MddRef>,
    ) -> MddRef {
        // Terminal shortcuts. ZERO annihilates (empty ∩ anything = empty); ONE
        // is the identity (whole-subtree ∩ S = S).
        if a == b {
            return a;
        }
        if a.is_zero() || b.is_zero() {
            return MddRef::ZERO;
        }
        if a.is_one() {
            return b;
        }
        if b.is_one() {
            return a;
        }

        // Order the pair so the cache key is symmetric (intersection commutes).
        let key = if a.0 <= b.0 { (a, b) } else { (b, a) };
        if let Some(&hit) = cache.get(&key) {
            return hit;
        }

        // Both interior. Recurse at the shallower (smaller) level; the deeper
        // operand is "unconstrained" at that level, expand it.
        let la = self.level_of(a);
        let lb = self.level_of(b);
        let level = la.min(lb);
        let dom = self.domain_size(level);

        let mut children = Vec::with_capacity(dom);
        for v in 0..dom as u64 {
            let ca = self.edge_at(a, level, v);
            let cb = self.edge_at(b, level, v);
            let child = self.intersect_rec(ca, cb, cache);
            children.push(child);
        }
        let result = self.get_node(level, children);
        cache.insert(key, result);
        result
    }

    /// Set difference `a \ b` (markings in `a` but not in `b`). Canonical:
    /// equal sets yield the same `MddRef`. Purely additive (interns new nodes;
    /// never mutates `a`/`b`), so both operands stay valid afterward.
    ///
    /// The third apply-style op (like [`MddStore::union`] / [`MddStore::intersect`]),
    /// leaf rule `a ∧ ¬b`. The result is a subset of `a` by construction, so a
    /// reachable-confined complement is `difference(reachable, S)` and the
    /// reachable deadlock set is `difference(reachable, ⋃_t Fireable(t))` — the
    /// SAME bound-truncated fireability the transition relation uses. Note
    /// difference is NOT commutative, so (unlike union/intersect) the cache key
    /// is the ordered pair `(a, b)`.
    #[must_use]
    pub fn difference(&mut self, a: MddRef, b: MddRef) -> MddRef {
        // Persistent apply cache (own slot; difference is NOT commutative, so its
        // rec keeps the ordered (a,b) key — see MddStore::apply_caches).
        let mut cache = self.take_apply_cache(ApplyOp::Difference);
        let result = self.difference_rec(a, b, &mut cache);
        self.put_apply_cache(ApplyOp::Difference, cache);
        result
    }

    fn difference_rec(
        &mut self,
        a: MddRef,
        b: MddRef,
        cache: &mut HashMap<(MddRef, MddRef), MddRef>,
    ) -> MddRef {
        // Terminal shortcuts.
        // a \ a = ∅.
        if a == b {
            return MddRef::ZERO;
        }
        // ∅ \ b = ∅.
        if a.is_zero() {
            return MddRef::ZERO;
        }
        // a \ ∅ = a.
        if b.is_zero() {
            return a;
        }
        // a \ ONE = ∅: ONE is the universe (every value at every remaining
        // level is in `b`), so nothing in `a` survives.
        if b.is_one() {
            return MddRef::ZERO;
        }
        // a == ONE \ b (b interior, non-ONE): ONE means "all completions
        // allowed", so we must subtract `b` value-by-value — handled by the
        // recursion below with `edge_at` treating ONE as unconstrained. (No
        // shortcut here: `ONE \ b` is generally non-empty and non-ONE.)

        // Difference is NOT commutative ⇒ ordered cache key.
        let key = (a, b);
        if let Some(&hit) = cache.get(&key) {
            return hit;
        }

        // Recurse at the shallower (smaller) level; the deeper operand is
        // unconstrained at that level (`edge_at` stays put), exactly as in
        // union/intersect. ONE is treated as deeper-than-every-level
        // (TERMINAL_LEVEL) and therefore unconstrained — `edge_at(ONE, ..)` =
        // ONE — so `ONE \ b` subtracts `b` at every level correctly.
        let la = self.level_of(a);
        let lb = self.level_of(b);
        let level = la.min(lb);
        let dom = self.domain_size(level);

        let mut children = Vec::with_capacity(dom);
        for v in 0..dom as u64 {
            let ca = self.edge_at(a, level, v);
            let cb = self.edge_at(b, level, v);
            let child = self.difference_rec(ca, cb, cache);
            children.push(child);
        }
        let result = self.get_node(level, children);
        cache.insert(key, result);
        result
    }

    /// Cofactor / restrict: fix variable `var` to `val` and PROJECT `var` OUT —
    /// the set `{ m with the var component dropped : m ∈ set(node) ∧ m[var] = val }`.
    /// The result never tests `var`. Additive (interns new nodes; `node` stays
    /// valid). This is the core primitive for variable reordering (sifting) and
    /// existential abstraction ([`Self::exists`]).
    ///
    /// Sound on the skip-reduced MDD: a node *below* `var`'s level leaves `var`
    /// free (absent from its subtree), so it is returned unchanged; a node *at*
    /// `var`'s level is replaced by its `val` child (dropping `var`); a node
    /// *above* `var` is rebuilt with each child cofactored.
    #[must_use]
    pub fn cofactor(&mut self, node: MddRef, var: u32, val: u64) -> MddRef {
        debug_assert!(
            (var as usize) >= self.bounds.len() || val < self.domain_size(var) as u64,
            "cofactor value {val} out of variable {var}'s domain"
        );
        let mut memo: HashMap<MddRef, MddRef> = HashMap::new();
        self.cofactor_rec(node, var, val, &mut memo)
    }

    fn cofactor_rec(
        &mut self,
        node: MddRef,
        var: u32,
        val: u64,
        memo: &mut HashMap<MddRef, MddRef>,
    ) -> MddRef {
        let nl = self.level_of(node);
        // Terminal, or the node sits below `var` — `var` is free and absent from
        // this subtree, so restricting it changes nothing.
        if node.is_terminal() || nl > var {
            return node;
        }
        if nl == var {
            // This node tests `var`: select the `val` edge, dropping `var`.
            return self.child(node, val);
        }
        // nl < var: rebuild the node with each child cofactored.
        if let Some(&hit) = memo.get(&node) {
            return hit;
        }
        let dom = self.domain_size(nl);
        let mut children = Vec::with_capacity(dom);
        for k in 0..dom as u64 {
            let c = self.child(node, k);
            children.push(self.cofactor_rec(c, var, val, memo));
        }
        let result = self.get_node(nl, children);
        memo.insert(node, result);
        result
    }

    /// Existential abstraction `∃ var. set(node)`: the markings that agree with
    /// some member of `node` on every variable EXCEPT `var` (which is abstracted
    /// away). Equivalently `⋃_val cofactor(node, var, val)`. The result never
    /// tests `var`. A no-op if `var` is out of range.
    #[must_use]
    pub fn exists(&mut self, node: MddRef, var: u32) -> MddRef {
        if (var as usize) >= self.bounds.len() {
            return node;
        }
        let dom = self.domain_size(var);
        let mut acc = MddRef::ZERO;
        for val in 0..dom as u64 {
            let cf = self.cofactor(node, var, val);
            acc = self.union(acc, cf);
        }
        acc
    }

    /// Rebuild every root under a new variable order, returning a FRESH store
    /// (with the permuted per-variable bounds) and the corresponding new roots.
    /// `new_order` is a permutation of `0..num_vars`: level `pos` of the result
    /// tests variable `new_order[pos]`. Each root's represented SET is preserved
    /// exactly — variables are re-encoded, not changed. This is the sound,
    /// store-wide core of dynamic variable reordering (sifting).
    ///
    /// Reorder-by-REBUILD, not in-place adjacent swap (intricate on a skip-
    /// reduced MDD): pos-by-pos, [`Self::cofactor`] the current node on the
    /// variable placed at `pos` for each of its values and recurse, memoized on
    /// `(old_node, pos)` for sharing. Reads the old store, writes the new one, so
    /// the caller must swap in the returned store + roots atomically (old
    /// `MddRef`s belong to the old store).
    #[must_use]
    pub fn reordered(&mut self, roots: &[MddRef], new_order: &[usize]) -> (MddStore, Vec<MddRef>) {
        debug_assert_eq!(
            new_order.len(),
            self.bounds.len(),
            "reorder must cover every variable exactly once"
        );
        let new_bounds: Vec<u64> = new_order.iter().map(|&v| self.bounds[v]).collect();
        let mut new_store = MddStore::new(new_bounds);
        let mut memo: HashMap<(MddRef, usize), MddRef> = HashMap::new();
        let new_roots = roots
            .iter()
            .map(|&r| self.rebuild_reorder(r, 0, new_order, &mut new_store, &mut memo))
            .collect();
        (new_store, new_roots)
    }

    fn rebuild_reorder(
        &mut self,
        node: MddRef,
        pos: usize,
        new_order: &[usize],
        new_store: &mut MddStore,
        memo: &mut HashMap<(MddRef, usize), MddRef>,
    ) -> MddRef {
        // All variables placed ⇒ the fully-cofactored node is a terminal, which
        // is shared (ONE/ZERO are store-independent) and valid in the new store.
        if pos == new_order.len() {
            debug_assert!(
                node.is_terminal(),
                "reorder reached the leaf with a non-terminal"
            );
            return node;
        }
        if let Some(&hit) = memo.get(&(node, pos)) {
            return hit;
        }
        let v = new_order[pos] as u32;
        let dom = self.domain_size(v);
        let mut children = Vec::with_capacity(dom);
        for val in 0..dom as u64 {
            let cf = self.cofactor(node, v, val);
            let child = self.rebuild_reorder(cf, pos + 1, new_order, new_store, memo);
            children.push(child);
        }
        let result = new_store.get_node(pos as u32, children);
        memo.insert((node, pos), result);
        result
    }

    /// Number of distinct interior (non-terminal) nodes reachable from `roots` —
    /// the MDD size that variable reordering (sifting) minimizes.
    #[must_use]
    pub fn nodes_reachable(&self, roots: &[MddRef]) -> usize {
        let mut seen: HashSet<MddRef> = HashSet::new();
        let mut stack: Vec<MddRef> = roots.iter().copied().filter(|r| !r.is_terminal()).collect();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            let dom = self.domain_size(self.level_of(n));
            for v in 0..dom as u64 {
                let c = self.child(n, v);
                if !c.is_terminal() {
                    stack.push(c);
                }
            }
        }
        seen.len()
    }

    /// Dynamic variable reordering (sifting): find an order that shrinks the MDD
    /// for `roots`, returning the reordered store, the remapped roots, AND the
    /// chosen order (`order[pos]` = the current-store variable placed at level
    /// `pos`, for the caller to compose with its own place↔level map). Greedy
    /// adjacent-transposition hill-climb — repeatedly try swapping each adjacent
    /// variable pair (via the sound [`Self::reordered`]) and keep any that
    /// reduces [`Self::nodes_reachable`], until no swap helps. Set-preserving by
    /// construction and never larger than the starting order. `self` is used as
    /// the fixed reference; the caller swaps in the returned store + roots.
    #[must_use]
    pub fn sift(&mut self, roots: &[MddRef]) -> (MddStore, Vec<MddRef>, Vec<usize>) {
        let n = self.bounds.len();
        let mut best_order: Vec<usize> = (0..n).collect();
        let (mut best_store, mut best_roots) = self.reordered(roots, &best_order);
        let mut best_size = best_store.nodes_reachable(&best_roots);
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n.saturating_sub(1) {
                let mut trial = best_order.clone();
                trial.swap(i, i + 1);
                // Drop the cofactor garbage from prior trials so `self` stays
                // bounded (it is only a scratch reference here).
                self.gc(roots);
                let (ts, tr) = self.reordered(roots, &trial);
                let size = ts.nodes_reachable(&tr);
                if size < best_size {
                    best_size = size;
                    best_order = trial;
                    best_store = ts;
                    best_roots = tr;
                    improved = true;
                }
            }
        }
        (best_store, best_roots, best_order)
    }

    /// The child of `node` along edge `value`, *as seen at `level`*.
    ///
    /// If `node` actually sits at `level`, this is its real child. If `node`
    /// sits *below* `level` (its level was skipped because it was redundant
    /// there), then at `level` the node is unconstrained, so every value leads
    /// back to `node` itself. Terminals are likewise unconstrained at every
    /// level above the terminal level. This is what makes union correct in the
    /// presence of suppressed levels.
    #[inline]
    fn edge_at(&self, node: MddRef, level: u32, value: u64) -> MddRef {
        let nl = self.level_of(node);
        debug_assert!(nl >= level, "operand is shallower than the union level");
        if nl == level {
            self.child(node, value)
        } else {
            // Skipped/terminal at this level ⇒ unconstrained ⇒ stay put.
            node
        }
    }

    /// Raw exact reachable-state count for `root` as a `u128`, fail-closed on
    /// overflow.
    ///
    /// Returns `Some(count)` when the count fits in `u128` (up to `u128::MAX`
    /// ≈ 3.4e38), or `None` when it exceeds `u128::MAX` — the caller must
    /// DECLINE rather than report a wrapped count (soundness). The count is
    /// computed exactly as a [`BigUint`] ([`Self::count_markings_big`]) and
    /// narrowed to `u128` here, so the only way to get `None` is a genuine
    /// `> u128::MAX` state space, never an arithmetic artifact. This is the
    /// widened entry the StateSpace metric path consumes so
    /// astronomically-large-but-still-finite reachable sets (e.g. high-bound
    /// counter / Philosophers nets ≈ 1e23) are reportable, while genuinely
    /// `> u128` spaces (e.g. FMS ≈ 1e47) fail closed at this carrier (the
    /// `_big` field remaining the source of truth).
    #[must_use]
    pub fn count_markings_u128(&self, root: MddRef) -> Option<u128> {
        // The count is now carried by the EXACT bignum entry
        // (`count_markings_big`) and narrowed fail-closed to `u128` here, so the
        // two paths cannot disagree on an in-range count. A count `> u128::MAX`
        // declines (`None`), exactly as before; the previous version's
        // saturating-`u128` recursion declined at the same point (its saturated
        // sentinel marked unrepresentability). The exact path additionally
        // reports a genuine `u128::MAX` count instead of declining — a strict
        // improvement that never produces a wrong number.
        big_to_u128(&self.count_markings_big(root))
    }

    /// Exact reachable-state count for `root`, fail-closed on overflow.
    ///
    /// Returns `Some(count)` when the count fits in `u64`, or `None` when it
    /// exceeds `u64::MAX` (the caller must DECLINE rather than report a
    /// wrapped/garbage count — soundness). The count itself is computed exactly
    /// (via [`Self::count_markings_big`]) and only narrowed to `u64` here, so the
    /// only way to get `None` is a genuine `> u64::MAX` state space — never an
    /// arithmetic artifact.
    #[must_use]
    pub fn count_markings(&self, root: MddRef) -> Option<u64> {
        // Reuse the widened u128 entry, then narrow fail-closed to u64.
        u64::try_from(self.count_markings_u128(root)?).ok()
    }

    /// `count_from` widened to arbitrary precision (`BigUint`). IDENTICAL
    /// recursion to [`Self::count_from`] — sum over a node's edges of
    /// `count(child) · gap_factor(level, child)` — but with EXACT (never
    /// saturating) bignum arithmetic, so the result is the true model count
    /// regardless of magnitude. The `u128` variant is recovered by narrowing
    /// this value, guaranteeing the two agree on every in-`u128` count.
    fn count_from_big(&self, root: MddRef, memo: &mut HashMap<MddRef, BigUint>) -> BigUint {
        if root.is_zero() {
            return BigUint::zero();
        }
        if root.is_one() {
            return BigUint::one();
        }
        if let Some(c) = memo.get(&root) {
            return c.clone();
        }
        let level = self.level_of(root);
        let mut total = BigUint::zero();
        for v in 0..self.domain_size(level) as u64 {
            let child = self.child(root, v);
            let sub = self.count_from_big(child, memo);
            if sub.is_zero() {
                continue;
            }
            total += sub * self.gap_factor_big(level, child);
        }
        memo.insert(root, total.clone());
        total
    }

    /// [`Self::gap_factor`] widened to `BigUint` (exact, never saturating).
    fn gap_factor_big(&self, parent_level: u32, child: MddRef) -> BigUint {
        let child_level = self.level_of(child);
        let upper = if child_level == TERMINAL_LEVEL {
            self.num_levels() as u32
        } else {
            child_level
        };
        let mut factor = BigUint::one();
        for l in (parent_level + 1)..upper {
            factor *= BigUint::from(self.domain_size(l) as u64);
        }
        factor
    }

    /// Exact reachable-state count for `root` as an arbitrary-precision
    /// [`BigUint`] — the single source of truth for the count metric.
    ///
    /// Unlike [`Self::count_markings_u128`] (which declines past `u128::MAX`)
    /// and [`Self::count_markings`] (which declines past `u64::MAX`), this
    /// NEVER declines on magnitude: the MDD already represents the set
    /// exactly, so the model count is exact for ANY finite reachable set,
    /// however astronomically large (e.g. FMS ≈1e47, Kanban/Philosophers
    /// families up to ≈1e238). This is the widened entry the StateSpace metric
    /// path consumes so a structurally-computable count beyond `u128` is
    /// REPORTED rather than declining on the representational cap.
    ///
    /// SOUNDNESS: the recursion is identical to the `u128` path; the only
    /// difference is the carrier type (exact bignum vs. saturating `u128`), so
    /// for any count `<= u128::MAX` the two are bit-for-bit equal (pinned by
    /// `count_big_matches_u128_in_range`).
    #[must_use]
    pub fn count_markings_big(&self, root: MddRef) -> BigUint {
        let mut memo = HashMap::new();
        let base = self.count_from_big(root, &mut memo);
        let root_level = self.level_of(root);
        let top_span_end = if root_level == TERMINAL_LEVEL {
            self.num_levels() as u32
        } else {
            root_level
        };
        let mut factor = BigUint::one();
        for l in 0..top_span_end {
            factor *= BigUint::from(self.domain_size(l) as u64);
        }
        base * factor
    }
}

/// Narrow a [`BigUint`] count to `u128`, fail-closed: `None` when it does not
/// fit (`> u128::MAX`). Shared narrowing so the `u128` and bignum count paths
/// can never disagree on the in-range cases.
#[must_use]
pub fn big_to_u128(count: &BigUint) -> Option<u128> {
    count.to_u128()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_then_count_is_one() {
        let mut s = MddStore::new(vec![3, 2, 5]);
        let m = s.singleton(&[2, 1, 4]);
        assert_eq!(s.count_markings(m), Some(1));
    }

    #[test]
    fn singleton_out_of_range_is_empty() {
        let mut s = MddStore::new(vec![1, 1]);
        let m = s.singleton(&[2, 0]); // 2 > bound 1
        assert_eq!(m, MddRef::ZERO);
        assert_eq!(s.count_markings(m), Some(0));
    }

    #[test]
    fn union_of_two_distinct_singletons_counts_two() {
        let mut s = MddStore::new(vec![2, 2]);
        let a = s.singleton(&[0, 0]);
        let b = s.singleton(&[1, 2]);
        let u = s.union(a, b);
        assert_eq!(s.count_markings(u), Some(2));
    }

    #[test]
    fn union_idempotent() {
        let mut s = MddStore::new(vec![3, 3, 3]);
        let a = s.singleton(&[1, 2, 3]);
        let u = s.union(a, a);
        assert_eq!(u, a);
        assert_eq!(s.count_markings(u), Some(1));
    }

    #[test]
    fn one_terminal_counts_full_product() {
        // ONE = "everything": every place free ⇒ Π (bound+1).
        let s = MddStore::new(vec![1, 2, 4]); // domains 2,3,5
        assert_eq!(s.count_markings(MddRef::ONE), Some(2 * 3 * 5));
    }

    #[test]
    fn zero_terminal_counts_zero() {
        let s = MddStore::new(vec![1, 2, 4]);
        assert_eq!(s.count_markings(MddRef::ZERO), Some(0));
    }

    #[test]
    fn redundant_top_level_scaled_in() {
        // Build {(*, 0), (*, 1)} over two places where place 0 is free.
        // Represented by a node at level 1 only (level 0 redundant) — the
        // top-span scaling must multiply by domain(0).
        let mut s = MddStore::new(vec![4, 1]); // domains 5, 2
                                               // singleton (3,0) ∪ (3,1) keeps place 0 fixed; instead union over all
                                               // of place 0 to force level-0 suppression:
        let mut acc = MddRef::ZERO;
        for v0 in 0..=4u64 {
            let s0 = s.singleton(&[v0, 0]);
            let s1 = s.singleton(&[v0, 1]);
            acc = s.union(acc, s0);
            acc = s.union(acc, s1);
        }
        // That's the full space: 5 * 2 = 10 markings.
        assert_eq!(s.count_markings(acc), Some(10));
        // And place 0 should be suppressed (acc collapses toward ONE).
        assert_eq!(acc, MddRef::ONE);
    }

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

    fn contains(store: &MddStore, root: MddRef, m: &[u64]) -> bool {
        let mut node = root;
        for (level, &v) in m.iter().enumerate() {
            if node.is_zero() {
                return false;
            }
            if node.is_one() {
                return true;
            }
            let nl = store.level_of(node) as usize;
            if nl > level {
                continue;
            }
            node = store.child(node, v);
        }
        node.is_one()
    }

    fn grid3() -> Vec<Vec<u64>> {
        (0..=2)
            .flat_map(|x| (0..=2u64).flat_map(move |y| (0..=2u64).map(move |z| vec![x, y, z])))
            .collect()
    }

    fn build_set(s: &mut MddStore, set: &[Vec<u64>]) -> MddRef {
        let mut root = MddRef::ZERO;
        for m in set {
            let x = s.singleton(m);
            root = s.union(root, x);
        }
        root
    }

    #[test]
    fn cofactor_matches_bruteforce() {
        // Exhaustive: for every (var, val) and every marking, cofactor membership
        // must equal set membership with that variable pinned to val.
        let mut s = MddStore::new(vec![2u64, 2, 2]);
        let all = grid3();
        let set: Vec<Vec<u64>> = all
            .iter()
            .filter(|m| m[0] + m[2] <= m[1] + 1)
            .cloned()
            .collect();
        let root = build_set(&mut s, &set);

        for var in 0u32..3 {
            for val in 0u64..=2 {
                let cf = s.cofactor(root, var, val);
                for m in &all {
                    let mut pinned = m.clone();
                    pinned[var as usize] = val;
                    assert_eq!(
                        contains(&s, cf, m),
                        set.contains(&pinned),
                        "cofactor(var={var}, val={val}) mismatch at {m:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn exists_matches_bruteforce() {
        // Exhaustive: ∃var membership must equal "some val makes the pinned
        // marking a member".
        let mut s = MddStore::new(vec![2u64, 2, 2]);
        let all = grid3();
        let set: Vec<Vec<u64>> = all.iter().filter(|m| m[0] == m[1]).cloned().collect();
        let root = build_set(&mut s, &set);

        for var in 0u32..3 {
            let ex = s.exists(root, var);
            for m in &all {
                let expect = (0u64..=2).any(|val| {
                    let mut pinned = m.clone();
                    pinned[var as usize] = val;
                    set.contains(&pinned)
                });
                assert_eq!(
                    contains(&s, ex, m),
                    expect,
                    "exists(var={var}) mismatch at {m:?}"
                );
            }
        }
    }

    #[test]
    fn reordered_preserves_set_exhaustive() {
        // For several variable permutations, the reordered MDD must represent the
        // SAME set: same marking count, and membership invariant under the
        // corresponding level permutation. Uses non-uniform domains to exercise
        // the bounds permutation.
        let bounds = vec![2u64, 1, 2];
        let mut s = MddStore::new(bounds.clone());
        let all: Vec<Vec<u64>> = (0..=2)
            .flat_map(|x| (0..=1u64).flat_map(move |y| (0..=2u64).map(move |z| vec![x, y, z])))
            .collect();
        let set: Vec<Vec<u64>> = all
            .iter()
            .filter(|m| m[0] + m[1] >= m[2])
            .cloned()
            .collect();
        let root = build_set(&mut s, &set);
        let orig_count = s.count_markings(root);

        for order in [
            vec![0usize, 1, 2], // identity
            vec![2, 0, 1],
            vec![1, 2, 0],
            vec![2, 1, 0],
            vec![0, 2, 1],
        ] {
            let (ns, nroots) = s.reordered(&[root], &order);
            let nroot = nroots[0];
            assert_eq!(
                ns.count_markings(nroot),
                orig_count,
                "count under {order:?}"
            );
            for m in &all {
                // Level pos of the new MDD tests variable order[pos].
                let m_new: Vec<u64> = order.iter().map(|&v| m[v]).collect();
                assert_eq!(
                    contains(&ns, nroot, &m_new),
                    contains(&s, root, m),
                    "membership under {order:?} at {m:?}"
                );
            }
        }
    }

    #[test]
    fn sift_preserves_set_and_shrinks_order_sensitive_mdd() {
        // {m0==m2 ∧ m1==m3}: order-sensitive. Identity [0,1,2,3] interleaves the
        // matched pairs (must remember m0,m1 before matching m2,m3); the single
        // adjacent swap 1↔2 → [0,2,1,3] groups each pair and is smaller — so the
        // hill-climb must strictly shrink the MDD while preserving the set.
        let bounds = vec![1u64, 1, 1, 1];
        let mut s = MddStore::new(bounds);
        let all: Vec<Vec<u64>> = (0..16u64)
            .map(|i| vec![i & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1])
            .collect();
        let set: Vec<Vec<u64>> = all
            .iter()
            .filter(|m| m[0] == m[2] && m[1] == m[3])
            .cloned()
            .collect();
        let root = build_set(&mut s, &set);
        let orig_count = s.count_markings(root);
        let orig_size = s.nodes_reachable(&[root]);

        let (ns, nroots, _order) = s.sift(&[root]);
        assert_eq!(
            ns.count_markings(nroots[0]),
            orig_count,
            "sift preserves the set"
        );
        let new_size = ns.nodes_reachable(&nroots);
        assert!(
            new_size <= orig_size,
            "sift never grows the MDD ({new_size} <= {orig_size})"
        );
        assert!(
            new_size < orig_size,
            "sift shrinks this order-sensitive MDD ({new_size} < {orig_size})"
        );
    }

    #[test]
    fn linear_le_set_matches_bruteforce() {
        // bounds [2,3,2]; test several coefficient/k combinations incl. negatives.
        let bounds = vec![2u64, 3, 2];
        let cases: Vec<(Vec<i128>, i128)> = vec![
            (vec![1, 1, 1], 3),     // sum <= 3
            (vec![1, 1, 1], 0),     // all zero
            (vec![1, 1, 1], 100),   // always
            (vec![1, 1, 1], -1),    // never
            (vec![2, 1, 0], 4),     // weighted
            (vec![1, -1, 0], 0),    // m0 <= m1
            (vec![-1, -1, -1], -2), // sum >= 2  (i.e. -sum <= -2)
            (vec![3, -2, 1], 2),
        ];
        for (coeffs, k) in cases {
            let mut s = MddStore::new(bounds.clone());
            let set = s.linear_le_set(&coeffs, k);
            for m in all_markings(&bounds) {
                let lhs: i128 = coeffs.iter().zip(&m).map(|(c, v)| c * *v as i128).sum();
                let want = lhs <= k;
                assert_eq!(
                    contains(&s, set, &m),
                    want,
                    "linear_le coeffs={coeffs:?} k={k} m={m:?}"
                );
            }
        }
    }

    #[test]
    fn guard_set_matches_bruteforce() {
        let bounds = vec![2u64, 1, 3];
        for pre in [vec![1u64, 0, 2], vec![0, 0, 0], vec![2, 1, 3]] {
            let mut s = MddStore::new(bounds.clone());
            let g = s.guard_set(&pre);
            for m in all_markings(&bounds) {
                let want = m.iter().zip(&pre).all(|(v, p)| v >= p);
                assert_eq!(contains(&s, g, &m), want, "guard pre={pre:?} m={m:?}");
            }
        }
    }

    #[test]
    fn difference_basic() {
        let mut s = MddStore::new(vec![2, 2]);
        let a = s.singleton(&[0, 0]);
        let b = s.singleton(&[1, 1]);
        let ab = s.union(a, b);
        // (a∪b) \ b = a.
        let d = s.difference(ab, b);
        assert_eq!(d, a);
        // a \ a = ∅.
        assert_eq!(s.difference(a, a), MddRef::ZERO);
        // a \ ∅ = a.
        assert_eq!(s.difference(a, MddRef::ZERO), a);
        // ∅ \ a = ∅.
        assert_eq!(s.difference(MddRef::ZERO, a), MddRef::ZERO);
        // a \ ONE (universe) = ∅.
        assert_eq!(s.difference(a, MddRef::ONE), MddRef::ZERO);
    }

    #[test]
    fn difference_complement_within_universe() {
        // ONE \ {one marking} should be the full space minus that marking.
        let mut s = MddStore::new(vec![1, 1]); // 4 markings
        let m = s.singleton(&[0, 1]);
        let comp = s.difference(MddRef::ONE, m);
        assert_eq!(s.count_markings(comp), Some(3));
        // And re-subtracting gives ONE minus 4 = the other 3 still; subtracting
        // the complement from ONE returns the singleton.
        let back = s.difference(MddRef::ONE, comp);
        assert_eq!(back, m);
    }

    #[test]
    fn difference_matches_bruteforce() {
        // Cross-check difference against explicit set subtraction on a grid.
        let bounds = vec![2u64, 2];
        let mut s = MddStore::new(bounds.clone());
        let all: Vec<Vec<u64>> = (0..=2)
            .flat_map(|x| (0..=2).map(move |y| vec![x, y]))
            .collect();
        let a_set: Vec<Vec<u64>> = all.iter().filter(|m| m[0] + m[1] <= 3).cloned().collect();
        let b_set: Vec<Vec<u64>> = all.iter().filter(|m| m[0] == 1).cloned().collect();
        let mut a = MddRef::ZERO;
        for m in &a_set {
            let x = s.singleton(m);
            a = s.union(a, x);
        }
        let mut b = MddRef::ZERO;
        for m in &b_set {
            let x = s.singleton(m);
            b = s.union(b, x);
        }
        let d = s.difference(a, b);
        let want_count = a_set.iter().filter(|m| !b_set.contains(m)).count() as u64;
        assert_eq!(s.count_markings(d), Some(want_count));
    }

    #[test]
    fn count_big_matches_u128_in_range() {
        // The bignum count must equal the u128 count exactly on every in-range
        // case (the IDENTICAL-on-u128 proof at the count level).
        let mut s = MddStore::new(vec![3, 3, 3, 3]);
        let markings = [[0, 0, 0, 0], [1, 2, 3, 0], [3, 3, 3, 3], [0, 1, 0, 1]];
        let mut acc = MddRef::ZERO;
        for m in &markings {
            let node = s.singleton(m);
            acc = s.union(acc, node);
        }
        let big = s.count_markings_big(acc);
        let u128c = s.count_markings_u128(acc).expect("in range");
        assert_eq!(big_to_u128(&big), Some(u128c));
        assert_eq!(big, BigUint::from(4u32));
        // ONE = full product 4^4 = 256.
        let big_one = s.count_markings_big(MddRef::ONE);
        assert_eq!(big_one, BigUint::from(256u32));
        assert_eq!(big_to_u128(&big_one), s.count_markings_u128(MddRef::ONE));
        // ZERO = 0.
        assert_eq!(s.count_markings_big(MddRef::ZERO), BigUint::zero());
    }

    #[test]
    fn count_big_reports_above_u128_exactly() {
        // 200 free 1-safe places (domain 2 each): |R| of the full space (ONE)
        // is 2^200 ≈ 1.6e60, FAR beyond u128::MAX (≈3.4e38). The u128 entry
        // DECLINES (None); the bignum entry reports the EXACT 2^200. This is the
        // representational unblock: an MDD-compact net whose count exceeds u128
        // is now reported instead of declining.
        let bounds = vec![1u64; 200];
        let s = MddStore::new(bounds);
        let big = s.count_markings_big(MddRef::ONE);
        let expected = BigUint::from(2u32).pow(200);
        assert_eq!(big, expected, "ONE over 200 1-safe places = 2^200");
        assert!(
            s.count_markings_u128(MddRef::ONE).is_none(),
            "2^200 must DECLINE the u128 entry (fail-closed above u128::MAX)",
        );
        // And it is genuinely > u128::MAX.
        assert!(big > BigUint::from(u128::MAX));
    }

    #[test]
    fn many_singletons_union_counts_distinct() {
        let mut s = MddStore::new(vec![3, 3, 3, 3]);
        let markings = [
            [0, 0, 0, 0],
            [1, 2, 3, 0],
            [3, 3, 3, 3],
            [1, 2, 3, 0], // duplicate
            [0, 1, 0, 1],
        ];
        let mut acc = MddRef::ZERO;
        for m in &markings {
            let node = s.singleton(m);
            acc = s.union(acc, node);
        }
        assert_eq!(s.count_markings(acc), Some(4), "4 distinct markings");
    }
}
