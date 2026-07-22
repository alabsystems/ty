// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Complemented-edge ROBDD core — the next node-store perf step toward oxidd
//! parity (oxidd uses complemented edges; the current [`crate::Bdd`] does not).
//!
//! A complemented edge carries a "negate" bit alongside the node index, so `f`
//! and `¬f` share ONE node (≈2× fewer nodes) and `not` is O(1) (flip the bit).
//! Canonicity rule (CUDD-style): the `hi` (then) edge of a stored node is NEVER
//! complemented — any complement is pushed onto the `lo` edge and the incoming
//! edge — and there is a single terminal (`ONE`); `ZERO = ¬ONE`.
//!
//! This lands ADDITIVELY (separate from the production [`crate::Bdd`], which the
//! whole toolchain still uses) so it can be built and validated in isolation
//! before the live core is migrated onto it. The subtle part — `sat_count` under
//! complement — is implemented + tested here precisely so the migration is then
//! mechanical.

use crate::HashMap;
use std::collections::HashMap as StdHashMap;

/// A complemented edge: bit 31 is the complement flag; bits 0..31 index a node
/// (`0` = the terminal node). `ONE`/`ZERO` are the constants.
pub type Edge = u32;

const COMP: u32 = 1 << 31;
const IDX_MASK: u32 = !COMP;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CNode {
    var: u32,
    lo: Edge,
    hi: Edge, // invariant: never complemented
}

/// A complemented-edge ROBDD manager.
#[derive(Default)]
pub struct CBdd {
    nodes: Vec<CNode>, // node index 0 is the terminal sentinel (unused slot 0)
    unique: HashMap<CNode, u32>,
    ite_cache: HashMap<(Edge, Edge, Edge), Edge>,
}

impl CBdd {
    /// The constant-`true` function.
    pub const ONE: Edge = 0;
    /// The constant-`false` function (`¬ONE`).
    pub const ZERO: Edge = COMP;
    const TERMINAL_VAR: u32 = u32::MAX;

    /// A fresh manager.
    #[must_use]
    pub fn new() -> Self {
        // Reserve node index 0 as the terminal sentinel so real nodes are >= 1.
        Self {
            nodes: vec![CNode {
                var: u32::MAX,
                lo: 0,
                hi: 0,
            }],
            unique: HashMap::default(),
            ite_cache: HashMap::default(),
        }
    }

    #[inline]
    fn index(e: Edge) -> u32 {
        e & IDX_MASK
    }
    #[inline]
    fn is_comp(e: Edge) -> bool {
        e & COMP != 0
    }
    #[inline]
    fn is_terminal(e: Edge) -> bool {
        Self::index(e) == 0
    }

    /// O(1) negation — the whole point of complemented edges.
    #[inline]
    #[must_use]
    pub fn not(e: Edge) -> Edge {
        e ^ COMP
    }

    fn var_of(&self, e: Edge) -> u32 {
        if Self::is_terminal(e) {
            Self::TERMINAL_VAR
        } else {
            self.nodes[Self::index(e) as usize].var
        }
    }

    /// The reduced + canonical node constructor for `if var { hi } else { lo }`,
    /// enforcing the "hi is never complemented" rule by pushing any complement
    /// onto the result edge.
    pub fn mk(&mut self, var: u32, lo: Edge, hi: Edge) -> Edge {
        if lo == hi {
            return lo; // redundant test
        }
        // Canonicalize: if hi is complemented, complement both children and the
        // returned edge (¬(ITE(v,hi,lo)) = ITE(v,¬hi,¬lo)).
        if Self::is_comp(hi) {
            let inner = self.mk_raw(var, Self::not(lo), Self::not(hi));
            return Self::not(inner);
        }
        self.mk_raw(var, lo, hi)
    }

    fn mk_raw(&mut self, var: u32, lo: Edge, hi: Edge) -> Edge {
        debug_assert!(!Self::is_comp(hi), "hi edge must be regular");
        let node = CNode { var, lo, hi };
        if let Some(&id) = self.unique.get(&node) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        self.unique.insert(node, id);
        id // a regular (uncomplemented) edge to the new node
    }

    /// The function for variable `var`.
    pub fn var(&mut self, var: u32) -> Edge {
        self.mk(var, Self::ZERO, Self::ONE)
    }

    fn cofactor(&self, e: Edge, var: u32, value: bool) -> Edge {
        if self.var_of(e) != var {
            return e;
        }
        let n = &self.nodes[Self::index(e) as usize];
        let child = if value { n.hi } else { n.lo };
        // distribute the edge's complement onto the chosen child
        if Self::is_comp(e) {
            Self::not(child)
        } else {
            child
        }
    }

    /// If-then-else.
    pub fn ite(&mut self, f: Edge, g: Edge, h: Edge) -> Edge {
        if f == Self::ONE {
            return g;
        }
        if f == Self::ZERO {
            return h;
        }
        if g == h {
            return g;
        }
        if g == Self::ONE && h == Self::ZERO {
            return f;
        }
        if g == Self::ZERO && h == Self::ONE {
            return Self::not(f);
        }
        if let Some(&id) = self.ite_cache.get(&(f, g, h)) {
            return id;
        }
        let v = self.var_of(f).min(self.var_of(g)).min(self.var_of(h));
        let f0 = self.cofactor(f, v, false);
        let f1 = self.cofactor(f, v, true);
        let g0 = self.cofactor(g, v, false);
        let g1 = self.cofactor(g, v, true);
        let h0 = self.cofactor(h, v, false);
        let h1 = self.cofactor(h, v, true);
        let lo = self.ite(f0, g0, h0);
        let hi = self.ite(f1, g1, h1);
        let res = self.mk(v, lo, hi);
        self.ite_cache.insert((f, g, h), res);
        res
    }

    /// Logical AND.
    pub fn and(&mut self, f: Edge, g: Edge) -> Edge {
        self.ite(f, g, Self::ZERO)
    }
    /// Logical OR.
    pub fn or(&mut self, f: Edge, g: Edge) -> Edge {
        self.ite(f, Self::ONE, g)
    }
    /// Logical XOR.
    pub fn xor(&mut self, f: Edge, g: Edge) -> Edge {
        let ng = Self::not(g);
        self.ite(f, ng, g)
    }

    /// Existentially quantify `vars` out of `e`: `∃ vars. e`.
    pub fn exists(&mut self, e: Edge, vars: &[u32]) -> Edge {
        let qset: std::collections::HashSet<u32> = vars.iter().copied().collect();
        let mut cache: HashMap<Edge, Edge> = HashMap::default();
        self.exists_rec(e, &qset, &mut cache)
    }

    fn exists_rec(
        &mut self,
        e: Edge,
        qset: &std::collections::HashSet<u32>,
        cache: &mut HashMap<Edge, Edge>,
    ) -> Edge {
        if Self::is_terminal(e) {
            return e;
        }
        let v = self.var_of(e);
        if qset.iter().all(|&q| q < v) {
            return e; // no quantified var at or below this level
        }
        if let Some(&id) = cache.get(&e) {
            return id;
        }
        let lo = self.cofactor(e, v, false);
        let hi = self.cofactor(e, v, true);
        let qlo = self.exists_rec(lo, qset, cache);
        let qhi = self.exists_rec(hi, qset, cache);
        let res = if qset.contains(&v) {
            self.or(qlo, qhi)
        } else {
            self.mk(v, qlo, qhi)
        };
        cache.insert(e, res);
        res
    }

    /// Fused relational product `∃ vars. (f ∧ g)` — the symbolic image step,
    /// quantify-on-the-fly (never materializes `f ∧ g`).
    pub fn and_exists(&mut self, f: Edge, g: Edge, vars: &[u32]) -> Edge {
        let qset: std::collections::HashSet<u32> = vars.iter().copied().collect();
        let mut cache: HashMap<(Edge, Edge), Edge> = HashMap::default();
        self.and_exists_rec(f, g, &qset, vars, &mut cache)
    }

    fn and_exists_rec(
        &mut self,
        f: Edge,
        g: Edge,
        qset: &std::collections::HashSet<u32>,
        vars: &[u32],
        cache: &mut HashMap<(Edge, Edge), Edge>,
    ) -> Edge {
        if f == Self::ZERO || g == Self::ZERO {
            return Self::ZERO;
        }
        if f == Self::ONE && g == Self::ONE {
            return Self::ONE;
        }
        if f == Self::ONE {
            return self.exists(g, vars);
        }
        if g == Self::ONE || f == g {
            return self.exists(f, vars);
        }
        let key = if f <= g { (f, g) } else { (g, f) };
        if let Some(&id) = cache.get(&key) {
            return id;
        }
        let v = self.var_of(f).min(self.var_of(g));
        let f0 = self.cofactor(f, v, false);
        let f1 = self.cofactor(f, v, true);
        let g0 = self.cofactor(g, v, false);
        let g1 = self.cofactor(g, v, true);
        let lo = self.and_exists_rec(f0, g0, qset, vars, cache);
        let hi = self.and_exists_rec(f1, g1, qset, vars, cache);
        let res = if qset.contains(&v) {
            self.or(lo, hi)
        } else {
            self.mk(v, lo, hi)
        };
        cache.insert(key, res);
        res
    }

    /// Relabel variables by an ORDER-PRESERVING map (`map[v]`, else `v`) — the
    /// next→current rename after the image. Complement bits carry through
    /// unchanged (renaming a variable does not change polarity).
    pub fn rename(&mut self, e: Edge, map: &StdHashMap<u32, u32>) -> Edge {
        let mut cache: HashMap<Edge, Edge> = HashMap::default();
        self.rename_rec(e, map, &mut cache)
    }

    fn rename_rec(
        &mut self,
        e: Edge,
        map: &StdHashMap<u32, u32>,
        cache: &mut HashMap<Edge, Edge>,
    ) -> Edge {
        if Self::is_terminal(e) {
            return e;
        }
        if let Some(&id) = cache.get(&e) {
            return id;
        }
        let idx = Self::index(e) as usize;
        let (var, lo, hi) = {
            let n = &self.nodes[idx];
            (n.var, n.lo, n.hi)
        };
        let rlo = self.rename_rec(lo, map, cache);
        let rhi = self.rename_rec(hi, map, cache);
        let nv = map.get(&var).copied().unwrap_or(var);
        let node = self.mk(nv, rlo, rhi);
        let res = if Self::is_comp(e) {
            Self::not(node)
        } else {
            node
        };
        cache.insert(e, res);
        res
    }

    /// Forward symbolic reachability fixpoint on the complement core (the
    /// native-BDD reachable-set construction, complemented-edge variant).
    /// `R ← R ∨ rename(∃current. R ∧ trans)` to a least fixpoint.
    pub fn reachable(&mut self, init: Edge, trans: Edge, current: &[u32], next: &[u32]) -> Edge {
        debug_assert_eq!(current.len(), next.len());
        let n2c: StdHashMap<u32, u32> = next.iter().copied().zip(current.iter().copied()).collect();
        // Frontier (BFS) reachability — image only newly-discovered states.
        let mut r = init;
        let mut frontier = init;
        loop {
            let img_next = self.and_exists(frontier, trans, current);
            let img = self.rename(img_next, &n2c);
            let not_r = Self::not(r);
            let new = self.and(img, not_r);
            if new == Self::ZERO {
                return r;
            }
            r = self.or(r, new);
            frontier = new;
        }
    }

    /// Number of distinct nodes reachable from `e` (ignoring complement bits).
    #[must_use]
    pub fn node_count(&self, e: Edge) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![Self::index(e)];
        while let Some(n) = stack.pop() {
            if n == 0 || !seen.insert(n) {
                continue;
            }
            let node = &self.nodes[n as usize];
            stack.push(Self::index(node.lo));
            stack.push(Self::index(node.hi));
        }
        seen.len()
    }

    /// EXACT model count over variables `0..num_vars`, handling complement. At a
    /// complemented edge the count is the complement over that level's range:
    /// `count(¬f over [v,N)) = 2^(N−v) − count(f over [v,N))`.
    ///
    /// Fail-closed: returns `None` whenever the exact count cannot be
    /// guaranteed to fit in `u128` — a free-variable span wider than 127 or an
    /// intermediate multiply/add overflow. Never returns a silently clamped or
    /// saturated value.
    #[must_use]
    pub fn sat_count(&self, root: Edge, num_vars: u32) -> Option<u128> {
        let mut memo: HashMap<u32, u128> = HashMap::default();
        let top = self.var_of(root).min(num_vars);
        if top > 127 {
            return None;
        }
        // mc(root) counts over [var_of(root), num_vars); free vars above double.
        self.mc(root, num_vars, &mut memo)?
            .checked_mul(1u128 << top)
    }

    /// Minterms of edge `e`'s function over `[var_of(e), num_vars)`. `None` on
    /// any inexactness (span > 127 or `u128` overflow); only exact (`Some`)
    /// sub-counts are memoised.
    fn mc(&self, e: Edge, num_vars: u32, memo: &mut HashMap<u32, u128>) -> Option<u128> {
        if Self::is_terminal(e) {
            return Some(if Self::is_comp(e) { 0 } else { 1 });
        }
        // Count the REGULAR node's function, then apply the edge complement.
        let idx = Self::index(e);
        let regular = if let Some(&c) = memo.get(&idx) {
            c
        } else {
            let n = &self.nodes[idx as usize];
            let v = n.var;
            let span =
                |child: Edge, this: &Self| -> u32 { this.var_of(child).min(num_vars) - v - 1 };
            let lo_span = span(n.lo, self);
            if lo_span > 127 {
                return None;
            }
            let clo = self
                .mc(n.lo, num_vars, memo)?
                .checked_mul(1u128 << lo_span)?;
            let hi_span = span(n.hi, self);
            if hi_span > 127 {
                return None;
            }
            let chi = self
                .mc(n.hi, num_vars, memo)?
                .checked_mul(1u128 << hi_span)?;
            let c = clo.checked_add(chi)?;
            memo.insert(idx, c);
            c
        };
        if Self::is_comp(e) {
            let v = self.nodes[idx as usize].var;
            let width = num_vars - v;
            if width > 127 {
                return None;
            }
            // regular <= 2^width always, so the subtraction cannot underflow.
            Some((1u128 << width) - regular)
        } else {
            Some(regular)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_is_o1_and_canonical() {
        let mut b = CBdd::new();
        let a = b.var(0);
        // ¬¬a == a (pure bit flip).
        assert_eq!(CBdd::not(CBdd::not(a)), a);
        // a and ¬a share the SAME underlying node (complement bit only).
        assert_eq!(CBdd::index(a), CBdd::index(CBdd::not(a)));
        // constants
        assert_eq!(CBdd::not(CBdd::ONE), CBdd::ZERO);
        assert_eq!(CBdd::not(CBdd::ZERO), CBdd::ONE);
    }

    #[test]
    fn boolean_ops_and_sat_count() {
        let mut b = CBdd::new();
        let (a, bb, c) = (b.var(0), b.var(1), b.var(2));
        // f = (a∧b)∨c over 3 vars: 5 models.
        let ab = b.and(a, bb);
        let f = b.or(ab, c);
        assert_eq!(b.sat_count(f, 3), Some(5));
        // constants
        assert_eq!(b.sat_count(CBdd::ONE, 3), Some(8));
        assert_eq!(b.sat_count(CBdd::ZERO, 3), Some(0));
        // single var over 4 vars: 8.
        assert_eq!(b.sat_count(a, 4), Some(8));
        // ¬a over 4 vars: 8 (complement of 8 out of 16).
        assert_eq!(b.sat_count(CBdd::not(a), 4), Some(8));
        // a∧b over 2 vars: 1 model (11).
        assert_eq!(b.sat_count(ab, 2), Some(1));
        // ¬(a∧b) over 2 vars: 3 models.
        assert_eq!(b.sat_count(CBdd::not(ab), 2), Some(3));
    }

    /// Regression (finding: sat_count silently clamped >127-var shifts and
    /// saturated u128 while documented EXACT): counts that are not exactly
    /// representable must come back `None`, never a clamped value — the
    /// complemented-edge twin of the plain-`Bdd` fail-closed test.
    #[test]
    fn sat_count_fails_closed_beyond_127_vars() {
        let mut b = CBdd::new();
        let a = b.var(0);

        // Exact boundary: 2^127 fits in u128.
        assert_eq!(b.sat_count(CBdd::ONE, 127), Some(1u128 << 127));
        assert_eq!(b.sat_count(a, 128), Some(1u128 << 127));
        // A complemented edge still counts exactly within the boundary.
        assert_eq!(b.sat_count(CBdd::not(a), 127), Some(1u128 << 126));
        // 2^128 does not fit: fail closed.
        assert_eq!(b.sat_count(CBdd::ONE, 128), None);
        // x0 over 200 free vars has 2^199 models: fail closed, not clamped.
        assert_eq!(b.sat_count(a, 200), None);
        assert_eq!(b.sat_count(CBdd::not(a), 200), None);
        // Complement over a >127-var span (¬x0 over 129 vars): fail closed.
        assert_eq!(b.sat_count(CBdd::not(a), 129), None);
    }

    #[test]
    fn canonical_equality_and_de_morgan() {
        let mut b = CBdd::new();
        let (a, bb, c) = (b.var(0), b.var(1), b.var(2));
        let ab = b.and(a, bb);
        // canonicity: commutative builds share the edge.
        let f1 = b.or(ab, c);
        let f2 = b.or(c, ab);
        assert_eq!(f1, f2);
        // De Morgan: ¬(a∧b) == ¬a ∨ ¬b — and with complement edges these are the
        // SAME edge (¬(a∧b) is literally not(ab)).
        let na = CBdd::not(a);
        let nb = CBdd::not(bb);
        let dm = b.or(na, nb);
        assert_eq!(CBdd::not(ab), dm);
    }

    #[test]
    fn complement_sharing_reduces_nodes() {
        // The whole point: f and ¬f share ONE node. Build a function that uses
        // both polarities of a sub-BDD; the complement core stores it once.
        let mut b = CBdd::new();
        let (a, bb, c) = (b.var(0), b.var(1), b.var(2));
        let ab = b.and(a, bb); // a sub-BDD
        let nab = CBdd::not(ab); // its negation — SAME node, complement bit
                                 // (a∧b ∧ c) ∨ (¬(a∧b) ∧ ¬c): uses ab and ¬ab.
        let nc = CBdd::not(c);
        let t1 = b.and(ab, c);
        let t2 = b.and(nab, nc);
        let f = b.or(t1, t2);
        // ab and ¬ab share their node ⇒ the structure is compact. Sanity: the
        // node for ab and the node for ¬ab are identical (index-equal).
        assert_eq!(CBdd::index(ab), CBdd::index(nab));
        assert!(b.node_count(f) >= 1);
    }

    #[test]
    fn rename_relabels_order_preservingly() {
        // f = x0 ∧ x2 over vars {0,2}; rename {0->1, 2->3} ⇒ x1 ∧ x3.
        let mut b = CBdd::new();
        let (x0, x2) = (b.var(0), b.var(2));
        let f = b.and(x0, x2);
        let map: std::collections::HashMap<u32, u32> =
            [(0u32, 1u32), (2u32, 3u32)].into_iter().collect();
        let g = b.rename(f, &map);
        // build x1 ∧ x3 directly and compare (canonical ⇒ same edge).
        let (x1, x3) = (b.var(1), b.var(3));
        let expect = b.and(x1, x3);
        assert_eq!(g, expect);
        // count over 4 vars: x1∧x3 ⇒ 4 models (x0,x2 free).
        assert_eq!(b.sat_count(g, 4), Some(4));
        // rename preserves complement: ¬f renamed = ¬(f renamed).
        let ng = b.rename(CBdd::not(f), &map);
        assert_eq!(ng, CBdd::not(expect));
    }

    #[test]
    fn reachability_on_complement_core() {
        // 2-bit mod-4 counter: current vars {0,1}, next {2,3}; reaches all 4.
        let mut b = CBdd::new();
        let (x0, x1) = (b.var(0), b.var(1));
        let (n0, n1) = (b.var(2), b.var(3));
        let nx0 = CBdd::not(x0);
        let nx1 = CBdd::not(x1);
        let nn0 = CBdd::not(n0);
        let nn1 = CBdd::not(n1);
        let clause = |b: &mut CBdd, a, c, d, e| {
            let ab = b.and(a, c);
            let de = b.and(d, e);
            b.and(ab, de)
        };
        let t00 = clause(&mut b, nx1, nx0, nn1, n0);
        let t01 = clause(&mut b, nx1, x0, n1, nn0);
        let t10 = clause(&mut b, x1, nx0, n1, n0);
        let t11 = clause(&mut b, x1, x0, nn1, nn0);
        let trans = {
            let a = b.or(t00, t01);
            let c = b.or(t10, t11);
            b.or(a, c)
        };
        let init = b.and(nx1, nx0);
        let r = b.reachable(init, trans, &[0, 1], &[2, 3]);
        assert_eq!(
            b.sat_count(r, 2),
            Some(4),
            "complement-core counter reaches all 4 states"
        );
    }

    #[test]
    fn xor_count() {
        let mut b = CBdd::new();
        let (a, bb) = (b.var(0), b.var(1));
        let x = b.xor(a, bb);
        assert_eq!(b.sat_count(x, 2), Some(2)); // 01, 10
        let eq = CBdd::not(x);
        assert_eq!(b.sat_count(eq, 2), Some(2)); // 00, 11
    }
}
