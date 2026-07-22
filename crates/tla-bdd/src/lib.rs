// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `tla-bdd` — TY's own native Reduced Ordered Binary Decision Diagram (ROBDD)
//! engine: the from-scratch replacement for the external `oxidd` stack.
//!
//! # Why
//!
//! The `tla-dd` lane currently builds its BDDs on `oxidd` (`oxidd-core`,
//! `oxidd-rules-bdd`, `oxidd-manager-index`, `oxidd-cache`, plus the
//! `oxidd-reorder` we vendored into `tla-dd-reorder`). To **fully remove and
//! replace** that external stack with a TY-owned native copy — preserving every
//! BDD lane's exact coverage (no MDD-subsumption needed) — we need a native
//! ROBDD. This crate is its canonical core.
//!
//! # This increment
//!
//! The reduced/ordered/canonical core: a [`Bdd`] manager with a node store, a
//! unique table (so structurally-equal nodes share one id — canonicity), the
//! reduced [`Bdd::mk`] constructor, recursive if-then-else [`Bdd::ite`] with a
//! computed cache, the derived Boolean ops, exact [`Bdd::sat_count`], and
//! [`Bdd::node_count`]. Variables are `u32`, ordered by index (smaller index =
//! closer to the root). Two terminals: [`Bdd::FALSE`] and [`Bdd::TRUE`].
//!
//! Canonicity invariant: a function has exactly ONE node id in a given `Bdd`
//! (reduced — no node with `lo == hi`; ordered — children have strictly larger
//! var; shared — the unique table dedups). So `==` on [`NodeId`] is semantic
//! equality, exactly as the rest of the toolchain relies on for BDDs.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::HashMap as StdHashMap;
use std::hash::{BuildHasherDefault, Hasher};

pub mod cedge;
pub mod ltl_product;
pub mod petri;

/// A fast, zero-dep FxHash-style hasher for the engine's internal tables.
/// `std`'s default `HashMap` uses SipHash (DoS-resistant but slow); the unique
/// table and computed caches are keyed by small integer tuples with no
/// adversarial input, so a multiply-rotate hash is sound here and 2–5× faster on
/// the hash-bound operations that dominate BDD work. (Same scheme rustc itself
/// uses internally.)
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_K: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(FX_K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }
}

/// `HashMap` specialized to the fast internal hasher.
type HashMap<K, V> = StdHashMap<K, V, BuildHasherDefault<FxHasher>>;

/// A node handle. `0` and `1` are the constant terminals; `>= 2` indexes an
/// inner node. Canonical: equal `NodeId`s ⟺ equal functions (within one [`Bdd`]).
pub type NodeId = u32;

/// An inner decision node: `if var { hi } else { lo }`. Reduced ⇒ `lo != hi`;
/// ordered ⇒ `var < var(lo)` and `var < var(hi)` (terminals have var `∞`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Node {
    var: u32,
    lo: NodeId,
    hi: NodeId,
}

/// Panic payload for the cooperative abort in [`Bdd::mk`] (audit 2026-07-02).
/// Raised BEFORE any store mutation, so the manager stays canonical across an
/// unwind; [`Bdd::reachable_within`] catches it and declines (`None`), and the
/// examination lanes' worker threads fold an escaped abort panic into their
/// existing panic-equals-decline handling.
#[derive(Debug)]
pub struct BddAbort;

/// Bytes charged per BDD node for budget derivation: ~12 B in-struct + ~36 B
/// amortized unique-table entry (audit 2026-07-02).
const BYTES_PER_NODE: usize = 48;
/// Fraction of effective-available memory the node store may occupy as a
/// STRUCTURAL backstop. ~0.20 reproduces the historic fixed 64M-node (~3 GB)
/// budget on a 16 GB machine while scaling proportionally to the host / MCC
/// confinement — the live footprint ceiling + collective free-memory floor in
/// the [`MemoryProbe`](tla_resource::MemoryProbe) are the PRIMARY guard.
const NODE_BUDGET_FRACTION: f64 = 0.20;
thread_local! {
    /// Test hook: when set, forces a sift at EVERY reachability BFS round so the
    /// mid-fixpoint manager swap + root remap is exercised on every net.
    static BDD_SIFT_STRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Test hook: when set, forces a copying-GC compaction at EVERY BFS round.
    static BDD_GC_STRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Force sifting every BFS round (tests only); returns the previous setting.
#[cfg(test)]
pub(crate) fn set_bdd_sift_stress(on: bool) -> bool {
    BDD_SIFT_STRESS.with(|c| c.replace(on))
}

/// Force GC compaction every BFS round (tests only); returns the previous value.
#[cfg(test)]
pub(crate) fn set_bdd_gc_stress(on: bool) -> bool {
    BDD_GC_STRESS.with(|c| c.replace(on))
}

#[inline]
fn bdd_sift_stress() -> bool {
    BDD_SIFT_STRESS.with(|c| c.get())
}

#[inline]
fn bdd_gc_stress() -> bool {
    BDD_GC_STRESS.with(|c| c.get())
}
/// Fallback node budget when memory detection fails (the historic fixed value).
const FALLBACK_ABORT_NODE_BUDGET: usize = 64 * 1024 * 1024;
/// Floor/ceiling for the derived node budget. The ceiling stays well under the
/// u32 `NodeId` id-space (~4.29 B) so a budget can never alias the id range.
const MIN_ABORT_NODE_BUDGET: usize = 1 << 20; // 1M nodes
const MAX_ABORT_NODE_BUDGET: usize = 1 << 30; // ~1.07B nodes

/// Node-store budget for the deadline-aware Petri entry points, DERIVED from
/// effective-available memory (adaptive to the machine / MCC confinement)
/// instead of a fixed magic constant. Clamped to `[MIN, MAX]`; falls back to
/// the historic 64M on detection failure. The [`Bdd`] additionally enforces a
/// live process-footprint ceiling and the collective free-memory floor via the
/// probe installed by [`Bdd::set_abort_limits`].
#[must_use]
pub fn default_abort_node_budget() -> usize {
    tla_resource::platform::effective_available_bytes()
        .map(|bytes| ((bytes as f64 * NODE_BUDGET_FRACTION) as usize) / BYTES_PER_NODE)
        .unwrap_or(FALLBACK_ABORT_NODE_BUDGET)
        .clamp(MIN_ABORT_NODE_BUDGET, MAX_ABORT_NODE_BUDGET)
}

/// Run `f`, folding a [`BddAbort`] unwind into a `None` decline (fail-closed)
/// and resuming any other panic. Sound because the abort fires before any
/// store mutation, so a caught manager remains canonical.
pub fn catch_abort<T>(f: impl FnOnce() -> Option<T>) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) if payload.is::<BddAbort>() => None,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Outcome of a target-aware reachability fixpoint
/// ([`Bdd::reachable_target_within`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReachOutcome {
    /// The least fixpoint converged and no target state was reached — an EXACT
    /// proof that the target set is unreachable from `init`.
    Fixpoint {
        /// The full reachable set (the least fixpoint).
        reached: NodeId,
        /// Number of image rounds to convergence.
        rounds: usize,
    },
    /// A target state is reachable.
    TargetHit {
        /// MINIMAL number of transition steps from an initial state to a
        /// target state (0 = a target state is initial).
        depth: usize,
    },
}

/// A Reduced Ordered BDD manager: owns the node store, the unique table
/// (canonicity), and the ITE computed cache.
#[derive(Default)]
pub struct Bdd {
    nodes: Vec<Node>,
    unique: HashMap<Node, NodeId>,
    ite_cache: HashMap<(NodeId, NodeId, NodeId), NodeId>,
    /// Variable→level map for ordering. EMPTY ⇒ identity (`level(v) == v`), the
    /// production default — so every ordering decision is byte-identical to the
    /// pre-decoupling engine until a reorder sets a non-identity permutation.
    /// Decoupling var from level is what lets reordering preserve var identity.
    level_of: Vec<u32>,
    /// Cooperative abort limits (audit 2026-07-02): before this, the manager
    /// had NO node or byte cap anywhere and deadlines were polled only
    /// between fixpoint rounds — a single monster `ite`/`and_exists` round
    /// (or a detached worker past its caller's timeout) grew unbounded.
    /// `None`/`None` (the default) reproduces the old unlimited behavior.
    abort_node_budget: Option<usize>,
    /// Adaptive deadline + memory probe (a live process-footprint ceiling and
    /// the collective free-memory floor), ticked per FRESH node insertion in
    /// [`Self::mk`]. Replaces the hand-rolled deadline stride: the probe
    /// self-tunes its clock/syscall cadence to the insertion rate and tightens
    /// as the footprint nears the ceiling. `None` disables the live guard.
    abort_probe: Option<tla_resource::MemoryProbe>,
    /// Cooperative external cancellation (audit 2026-07-10): a shared flag a
    /// portfolio/orchestrator sets when another engine has already won. Checked
    /// per FRESH node insertion in [`Self::mk`] (in-operation, like the probe)
    /// and per fixpoint round, folding into the same [`BddAbort`] decline path.
    /// `None` (the default) is byte-identical to the pre-flag engine.
    abort_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Bdd {
    /// The constant-`false` terminal.
    pub const FALSE: NodeId = 0;
    /// The constant-`true` terminal.
    pub const TRUE: NodeId = 1;
    /// Sentinel "variable index" of a terminal: greater than any real var, so
    /// ordering comparisons treat terminals as below every decision level.
    const TERMINAL_VAR: u32 = u32::MAX;

    /// A fresh, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Is `node` a terminal (constant)?
    #[must_use]
    pub fn is_terminal(node: NodeId) -> bool {
        node <= Self::TRUE
    }

    /// The decision variable of `node` (or [`Self::TERMINAL_VAR`] for terminals).
    fn var_of(&self, node: NodeId) -> u32 {
        if Self::is_terminal(node) {
            Self::TERMINAL_VAR
        } else {
            self.nodes[(node - 2) as usize].var
        }
    }

    /// The ORDER LEVEL of variable `var` (identity unless a reorder set a
    /// permutation). All ordering comparisons go through this, not the raw var.
    #[inline]
    fn level(&self, var: u32) -> u32 {
        if var == Self::TERMINAL_VAR {
            return Self::TERMINAL_VAR;
        }
        self.level_of.get(var as usize).copied().unwrap_or(var)
    }

    /// The order level of `node`'s variable (terminals sit below every level).
    #[inline]
    fn node_level(&self, node: NodeId) -> u32 {
        self.level(self.var_of(node))
    }

    /// Evaluate the function `f` on a full variable assignment (`assignment[v]`
    /// is the value of variable `v`): follow `hi`/`lo` per the tested variable
    /// down to a terminal. This is the function-identity probe that distinguishes
    /// a var-PRESERVING reorder (same `eval` on every assignment) from a
    /// relabeling one — the validation primitive the structural reorder needs,
    /// and generally useful for testing.
    #[must_use]
    pub fn eval(&self, f: NodeId, assignment: &[bool]) -> bool {
        let mut node = f;
        while !Self::is_terminal(node) {
            let n = &self.nodes[(node - 2) as usize];
            node = if assignment[n.var as usize] {
                n.hi
            } else {
                n.lo
            };
        }
        node == Self::TRUE
    }

    /// The `(lo, hi)` children of an inner node. Panics on a terminal.
    fn children(&self, node: NodeId) -> (NodeId, NodeId) {
        debug_assert!(!Self::is_terminal(node));
        let n = self.nodes[(node - 2) as usize];
        (n.lo, n.hi)
    }

    /// Set cooperative abort limits: a node-store budget (in NODES) and/or a
    /// wall-clock deadline, checked in [`Self::mk`] before each FRESH node
    /// insertion. Once exceeded, `mk` panics with [`BddAbort`] — BEFORE any
    /// mutation, so the store remains canonical across the unwind.
    /// [`Self::reachable_within`] catches the abort and declines (`None`);
    /// callers driving raw ops directly should either wrap them in
    /// `catch_unwind` or run the manager on a worker thread whose panic
    /// already maps to a decline (the examination-lane pattern). `None`/`None`
    /// disables the live guard (the default).
    ///
    /// Beyond the node cap and deadline, this installs an adaptive
    /// [`MemoryProbe`](tla_resource::MemoryProbe) carrying a live
    /// process-footprint ceiling (a fraction of effective-available memory) and
    /// the collective free-memory floor, so a symbolic run backs off under real
    /// memory pressure — and cooperatively with concurrent MCC solvers — rather
    /// than only at the fixed node count.
    pub fn set_abort_limits(
        &mut self,
        node_budget: Option<usize>,
        deadline: Option<std::time::Instant>,
    ) {
        self.abort_node_budget = node_budget;
        self.abort_probe = if node_budget.is_some() || deadline.is_some() {
            Some(tla_resource::MemoryProbe::new(
                tla_resource::MemoryBudget::symbolic_explorer(),
                deadline,
            ))
        } else {
            None
        };
    }

    /// Arm cooperative external cancellation: when `flag` reads `true`, the
    /// next fresh node insertion (and the next fixpoint round) aborts with the
    /// same fail-closed [`BddAbort`] decline the node budget uses. A portfolio
    /// sets the shared flag when another engine has already produced the
    /// verdict, so a losing symbolic run stops within one operation instead of
    /// running to its deadline. `None` disarms.
    pub fn set_abort_flag(&mut self, flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) {
        self.abort_flag = flag;
    }

    /// True when the armed cancellation flag (if any) has been raised.
    fn abort_flag_raised(&self) -> bool {
        self.abort_flag
            .as_ref()
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The REDUCED node constructor: returns the canonical id for
    /// `if var { hi } else { lo }`. Eliminates redundant tests (`lo == hi`) and
    /// shares structurally-equal nodes via the unique table.
    ///
    /// # Panics
    ///
    /// Panics with [`BddAbort`] — before any store mutation — when limits set
    /// via [`Self::set_abort_limits`] are exceeded at a fresh insertion.
    pub fn mk(&mut self, var: u32, lo: NodeId, hi: NodeId) -> NodeId {
        if lo == hi {
            return lo; // reduction: the test on `var` is redundant
        }
        debug_assert!(
            self.level(var) < self.node_level(lo),
            "ordering: level(var) < level(lo)"
        );
        debug_assert!(
            self.level(var) < self.node_level(hi),
            "ordering: level(var) < level(hi)"
        );
        let node = Node { var, lo, hi };
        if let Some(&id) = self.unique.get(&node) {
            return id; // sharing: this function already exists
        }
        // Cooperative abort (audit 2026-07-02) — only fresh insertions are
        // charged, and the panic fires BEFORE the push/insert pair so the
        // store stays canonical whether or not the unwind is caught. The
        // node-count cap is a cheap O(1) structural backstop; the adaptive
        // probe enforces the live footprint ceiling, collective floor, and
        // deadline (self-amortizing its clock/syscall cadence).
        if let Some(budget) = self.abort_node_budget {
            if self.nodes.len() >= budget {
                std::panic::panic_any(BddAbort);
            }
        }
        if self.abort_flag_raised() {
            std::panic::panic_any(BddAbort);
        }
        if let Some(probe) = &mut self.abort_probe {
            if probe.over_budget() {
                std::panic::panic_any(BddAbort);
            }
        }
        let id = self.nodes.len() as u32 + 2;
        self.nodes.push(node);
        self.unique.insert(node, id);
        id
    }

    /// The ROBDD for a single variable `var` (`if var { true } else { false }`).
    pub fn var(&mut self, var: u32) -> NodeId {
        self.mk(var, Self::FALSE, Self::TRUE)
    }

    /// Cofactor `node` by `var = value`: the function with `var` fixed. For a
    /// node testing exactly `var`, takes the matching child; otherwise (the node
    /// is below `var` in the order, so independent of it) returns `node`.
    fn cofactor(&self, node: NodeId, var: u32, value: bool) -> NodeId {
        if self.var_of(node) == var {
            let (lo, hi) = self.children(node);
            if value {
                hi
            } else {
                lo
            }
        } else {
            node
        }
    }

    /// If-then-else: the canonical BDD for `if f { g } else { h }`. The universal
    /// Boolean primitive — `and`/`or`/`not`/`xor`/`implies` are derived from it.
    pub fn ite(&mut self, f: NodeId, g: NodeId, h: NodeId) -> NodeId {
        // Terminal / trivial reductions.
        if f == Self::TRUE {
            return g;
        }
        if f == Self::FALSE {
            return h;
        }
        if g == h {
            return g;
        }
        if g == Self::TRUE && h == Self::FALSE {
            return f; // ite(f, 1, 0) == f
        }
        if let Some(&id) = self.ite_cache.get(&(f, g, h)) {
            return id;
        }
        // Split on the top variable among f, g, h — the one with the minimum
        // ORDER LEVEL (== minimum var under the identity default).
        let v = {
            let mut best_var = Self::TERMINAL_VAR;
            let mut best_level = Self::TERMINAL_VAR;
            for &node in &[f, g, h] {
                let lv = self.node_level(node);
                if lv < best_level {
                    best_level = lv;
                    best_var = self.var_of(node);
                }
            }
            best_var
        };
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
    pub fn and(&mut self, f: NodeId, g: NodeId) -> NodeId {
        self.ite(f, g, Self::FALSE)
    }

    /// Logical OR.
    pub fn or(&mut self, f: NodeId, g: NodeId) -> NodeId {
        self.ite(f, Self::TRUE, g)
    }

    /// Logical NOT.
    pub fn not(&mut self, f: NodeId) -> NodeId {
        self.ite(f, Self::FALSE, Self::TRUE)
    }

    /// Logical XOR.
    pub fn xor(&mut self, f: NodeId, g: NodeId) -> NodeId {
        let not_g = self.not(g);
        self.ite(f, not_g, g)
    }

    /// Existentially quantify the variables in `vars` (a sorted, deduplicated
    /// var list) out of `f`: `∃ vars. f`. A node testing a quantified variable
    /// is replaced by `lo ∨ hi` (the variable disappears); other nodes recurse
    /// structurally. Cached per call. This is the projection half of the
    /// symbolic reachability image.
    pub fn exists(&mut self, f: NodeId, vars: &[u32]) -> NodeId {
        let mut cache: HashMap<NodeId, NodeId> = HashMap::default();
        self.exists_rec(f, vars, &mut cache)
    }

    fn exists_rec(
        &mut self,
        f: NodeId,
        vars: &[u32],
        cache: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if Self::is_terminal(f) {
            return f;
        }
        let v = self.var_of(f);
        // No quantified variable at or below this LEVEL ⇒ `f` is unchanged.
        if vars.iter().all(|&q| self.level(q) < self.node_level(f)) {
            return f;
        }
        if let Some(&id) = cache.get(&f) {
            return id;
        }
        let (lo, hi) = self.children(f);
        let qlo = self.exists_rec(lo, vars, cache);
        let qhi = self.exists_rec(hi, vars, cache);
        let res = if vars.contains(&v) {
            self.or(qlo, qhi) // quantify this variable away
        } else {
            self.mk(v, qlo, qhi)
        };
        cache.insert(f, res);
        res
    }

    /// Fused relational-product primitive `∃ vars. (f ∧ g)` — the symbolic
    /// reachability image (`∃ current. R(current) ∧ T(current,next)`), the role
    /// oxidd's fused `apply_exists(And, ..)` plays for the `tla-dd` lane.
    ///
    /// FUSED single recursion: the conjunction and the quantification are
    /// computed together, so the full `f ∧ g` is never materialised — at a
    /// quantified level the two cofactor-products are OR-ed immediately
    /// (`∃v. ITE(v, f₁∧g₁, f₀∧g₀) = (f₀∧g₀) ∨ (f₁∧g₁)` since the cofactors drop
    /// `v`). This is the key image-step perf win toward oxidd parity; it is
    /// behaviour-identical to the old `exists(and(f,g),vars)` (the reachability,
    /// fair-cycle, and 6000-net batteries pin that).
    pub fn and_exists(&mut self, f: NodeId, g: NodeId, vars: &[u32]) -> NodeId {
        let qset: std::collections::HashSet<u32> = vars.iter().copied().collect();
        let mut cache: HashMap<(NodeId, NodeId), NodeId> = HashMap::default();
        self.and_exists_rec(f, g, &qset, vars, &mut cache)
    }

    fn and_exists_rec(
        &mut self,
        f: NodeId,
        g: NodeId,
        qset: &std::collections::HashSet<u32>,
        vars: &[u32],
        cache: &mut HashMap<(NodeId, NodeId), NodeId>,
    ) -> NodeId {
        // Terminal / trivial cases.
        if f == Self::FALSE || g == Self::FALSE {
            return Self::FALSE;
        }
        if f == Self::TRUE && g == Self::TRUE {
            return Self::TRUE;
        }
        if f == Self::TRUE {
            return self.exists(g, vars);
        }
        if g == Self::TRUE || f == g {
            return self.exists(f, vars);
        }
        let key = if f <= g { (f, g) } else { (g, f) };
        if let Some(&id) = cache.get(&key) {
            return id;
        }
        // Split on the variable with the minimum LEVEL among f, g.
        let v = {
            let (vf, vg) = (self.var_of(f), self.var_of(g));
            if self.level(vf) <= self.level(vg) {
                vf
            } else {
                vg
            }
        };
        let f0 = self.cofactor(f, v, false);
        let f1 = self.cofactor(f, v, true);
        let g0 = self.cofactor(g, v, false);
        let g1 = self.cofactor(g, v, true);
        let lo = self.and_exists_rec(f0, g0, qset, vars, cache);
        let hi = self.and_exists_rec(f1, g1, qset, vars, cache);
        let res = if qset.contains(&v) {
            self.or(lo, hi) // quantify v away (cofactors already dropped it)
        } else {
            self.mk(v, lo, hi)
        };
        cache.insert(key, res);
        res
    }

    /// The BDD of the linear inequality `Σ (w·x_var) ≤ k` over Boolean variables.
    /// `terms` is `(var, weight)` pairs sorted by ascending `var` (distinct
    /// vars); `weight`/`k` are signed `i128`. This is the characteristic set of a
    /// token-cardinality atom (`Σ cₚ·m[p] ≤ k` over the bit-blasted places).
    /// Built as a threshold BDD with suffix-sum pruning, so it stays polynomial.
    pub fn linear_le(&mut self, terms: &[(u32, i128)], k: i128) -> NodeId {
        // Suffix sums of the positive / negative weight mass, for pruning.
        let n = terms.len();
        let mut suf_max = vec![0i128; n + 1];
        let mut suf_min = vec![0i128; n + 1];
        for i in (0..n).rev() {
            let w = terms[i].1;
            suf_max[i] = suf_max[i + 1] + w.max(0);
            suf_min[i] = suf_min[i + 1] + w.min(0);
        }
        let mut memo: HashMap<(usize, i128), NodeId> = HashMap::default();
        self.linear_le_rec(terms, 0, k, &suf_max, &suf_min, &mut memo)
    }

    fn linear_le_rec(
        &mut self,
        terms: &[(u32, i128)],
        idx: usize,
        budget: i128,
        suf_max: &[i128],
        suf_min: &[i128],
        memo: &mut HashMap<(usize, i128), NodeId>,
    ) -> NodeId {
        // Prune: if even the largest possible remaining sum fits, always TRUE;
        // if even the smallest can't fit, always FALSE.
        if suf_max[idx] <= budget {
            return Self::TRUE;
        }
        if suf_min[idx] > budget {
            return Self::FALSE;
        }
        if idx == terms.len() {
            return if budget >= 0 { Self::TRUE } else { Self::FALSE };
        }
        if let Some(&id) = memo.get(&(idx, budget)) {
            return id;
        }
        let (var, w) = terms[idx];
        let lo = self.linear_le_rec(terms, idx + 1, budget, suf_max, suf_min, memo); // x=0
        let hi = self.linear_le_rec(terms, idx + 1, budget - w, suf_max, suf_min, memo); // x=1
        let res = self.mk(var, lo, hi);
        memo.insert((idx, budget), res);
        res
    }

    /// `max` over the satisfying assignments of `f` of `Σ_v weights[v]·assign[v]`
    /// — the exact weighted maximum over the represented set (the bit-level
    /// primitive behind the `max_token_sum` StateSpace metric). A free
    /// (skipped) variable `v` contributes `max(0, weights[v])` (set it to 1 iff
    /// that helps). Returns `0` for the empty set (`f == FALSE`). `weights.len()`
    /// must cover every variable of `f`.
    #[must_use]
    pub fn max_weighted(&self, root: NodeId, weights: &[i128], num_vars: u32) -> i128 {
        if root == Self::FALSE {
            return 0;
        }
        let free = |from: u32, to: u32| -> i128 {
            // best contribution of the free variables in [from, to)
            (from..to).map(|v| weights[v as usize].max(0)).sum()
        };
        let mut memo: HashMap<NodeId, i128> = HashMap::default();
        let below = self.max_weighted_rec(root, weights, num_vars, &mut memo);
        free(0, self.var_of(root).min(num_vars)) + below
    }

    fn max_weighted_rec(
        &self,
        node: NodeId,
        weights: &[i128],
        num_vars: u32,
        memo: &mut HashMap<NodeId, i128>,
    ) -> i128 {
        if node == Self::TRUE {
            return 0;
        }
        debug_assert_ne!(node, Self::FALSE);
        if let Some(&c) = memo.get(&node) {
            return c;
        }
        let v = self.var_of(node);
        let (lo, hi) = self.children(node);
        let free = |from: u32, to: u32| -> i128 {
            ((from + 1)..to).map(|x| weights[x as usize].max(0)).sum()
        };
        // lo branch: var v = 0 (no weight); hi branch: var v = 1 (+weights[v]).
        let opt = |child: NodeId,
                   take: bool,
                   this: &Self,
                   memo: &mut HashMap<NodeId, i128>|
         -> Option<i128> {
            if child == Self::FALSE {
                return None;
            }
            let gap = free(v, this.var_of(child).min(num_vars));
            let sub = this.max_weighted_rec(child, weights, num_vars, memo);
            let here = if take { weights[v as usize] } else { 0 };
            Some(here + gap + sub)
        };
        let lo_opt = opt(lo, false, self, memo);
        let hi_opt = opt(hi, true, self, memo);
        let res = match (lo_opt, hi_opt) {
            (Some(a), Some(b)) => a.max(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => i128::MIN, // unreachable for a non-FALSE node
        };
        memo.insert(node, res);
        res
    }

    /// Relabel variables of `f` by an ORDER-PRESERVING map (`map[v]`, or `v` if
    /// absent). Used to rename next-state variables back to current-state ones
    /// after the image. The map MUST be monotonic and collision-free on `f`'s
    /// variables (e.g. interleaved `current=2i`, `next=2i+1` ⇒ `2i+1 ↦ 2i`,
    /// applied to a next-only BDD), so the rebuilt children keep strictly-larger
    /// vars and canonicity holds.
    pub fn rename(&mut self, f: NodeId, map: &StdHashMap<u32, u32>) -> NodeId {
        let mut cache: HashMap<NodeId, NodeId> = HashMap::default();
        self.rename_rec(f, map, &mut cache)
    }

    fn rename_rec(
        &mut self,
        f: NodeId,
        map: &StdHashMap<u32, u32>,
        cache: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if Self::is_terminal(f) {
            return f;
        }
        if let Some(&id) = cache.get(&f) {
            return id;
        }
        let v = self.var_of(f);
        let (lo, hi) = self.children(f);
        let rlo = self.rename_rec(lo, map, cache);
        let rhi = self.rename_rec(hi, map, cache);
        let nv = map.get(&v).copied().unwrap_or(v);
        let res = self.mk(nv, rlo, rhi);
        cache.insert(f, res);
        res
    }

    /// General cofactor: `f` with variable `var` fixed to `value` (var may sit
    /// anywhere in the order, unlike [`Self::cofactor`] which only peels the top
    /// var). Recursive + per-call memoized.
    pub fn restrict(&mut self, f: NodeId, var: u32, value: bool) -> NodeId {
        let mut cache: HashMap<NodeId, NodeId> = HashMap::default();
        self.restrict_rec(f, var, value, &mut cache)
    }

    fn restrict_rec(
        &mut self,
        f: NodeId,
        var: u32,
        value: bool,
        cache: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if Self::is_terminal(f) {
            return f;
        }
        let v = self.var_of(f);
        if self.level(v) > self.level(var) {
            return f; // `var` is above f's top LEVEL ⇒ f independent of it
        }
        if let Some(&id) = cache.get(&f) {
            return id;
        }
        let (lo, hi) = self.children(f);
        let res = if v == var {
            if value {
                hi
            } else {
                lo
            }
        } else {
            let rlo = self.restrict_rec(lo, var, value, cache);
            let rhi = self.restrict_rec(hi, var, value, cache);
            self.mk(v, rlo, rhi)
        };
        cache.insert(f, res);
        res
    }

    /// Rebuild `f` under a new variable order — TY's native variable reordering
    /// (the scalability lever `tla-dd` currently gets from `oxidd` +
    /// `tla-dd-reorder`, which `tla-bdd` previously lacked entirely).
    ///
    /// `order[new_level] = the variable to test at that level`. The result is a
    /// canonical BDD over variables `0..order.len()` where level `L` represents
    /// the original variable `order[L]` — so it is ANSWER-PRESERVING in the
    /// permutation-invariant sense: `reorder(f, π).sat_count(n) == f.sat_count(n)`
    /// when `π` is a permutation of `f`'s `n` variables (the model COUNT is
    /// order-independent), while the node count changes — the whole point. The
    /// caller tracks the level→var map for any per-variable interpretation.
    ///
    /// This first form rebuilds via cofactor recursion (memoised); the efficient
    /// in-place adjacent-level sift is a later structural improvement.
    pub fn reorder(&mut self, f: NodeId, order: &[u32]) -> NodeId {
        let mut cache: HashMap<(NodeId, usize), NodeId> = HashMap::default();
        self.reorder_rec(f, order, 0, &mut cache)
    }

    fn reorder_rec(
        &mut self,
        f: NodeId,
        order: &[u32],
        level: usize,
        cache: &mut HashMap<(NodeId, usize), NodeId>,
    ) -> NodeId {
        if Self::is_terminal(f) || level == order.len() {
            return f;
        }
        if let Some(&id) = cache.get(&(f, level)) {
            return id;
        }
        let var = order[level];
        let f0 = self.restrict(f, var, false);
        let f1 = self.restrict(f, var, true);
        let lo = self.reorder_rec(f0, order, level + 1, cache);
        let hi = self.reorder_rec(f1, order, level + 1, cache);
        let res = self.mk(level as u32, lo, hi);
        cache.insert((f, level), res);
        res
    }

    /// VAR-IDENTITY-PRESERVING reorder: rebuild `f` under `new_order` into a
    /// FRESH manager whose `level_of` is the new order from the start, returning
    /// `(new_manager, root)`. Variable identities are preserved (var `x` stays
    /// `x`, just at a new level), so the result is the SAME function — exactly
    /// what `tla-dd`'s `set_variable_order` needs (unlike [`Self::reorder`], which
    /// relabels vars to levels).
    ///
    /// The two-manager split is the key: cofactors are taken in `self` (the OLD
    /// manager, still in its old/identity order — valid), while nodes are built
    /// in `nb` (the NEW manager, new order — valid). Neither manager ever holds
    /// mixed-order nodes, so canonicity holds in both. `new_order` must list every
    /// variable of `f`. Memoised ⇒ polynomial in the BDD sizes.
    ///
    /// Answer-preserving in the STRONG sense: `nb.eval(root, a) == self.eval(f, a)`
    /// for every assignment `a`, and `nb.sat_count(root, n) == self.sat_count(f, n)`
    /// — while the node count changes (the scalability win). This is the
    /// correctness-complete first form; the efficient in-place adjacent swap (which
    /// needs the mutable/indexed node-store rearchitecture) is the later perf form.
    #[must_use]
    pub fn reorder_into(&mut self, f: NodeId, new_order: &[u32]) -> (Bdd, NodeId) {
        let mut nb = Bdd::new();
        let max_var = new_order.iter().copied().max().unwrap_or(0) as usize;
        nb.level_of = vec![0u32; max_var + 1];
        for (lvl, &var) in new_order.iter().enumerate() {
            nb.level_of[var as usize] = lvl as u32;
        }
        let mut cache: HashMap<(NodeId, usize), NodeId> = HashMap::default();
        let root = self.reorder_into_rec(&mut nb, f, new_order, 0, &mut cache);
        (nb, root)
    }

    /// Reorder SEVERAL functions into ONE new manager under `new_order`, sharing
    /// the rebuild cache so a subgraph shared between roots is reordered once.
    /// Returns the new `Bdd` and the reordered roots (index-aligned with
    /// `roots`). This is the multi-root prerequisite for reordering a whole
    /// reachability state (e.g. the reached set AND the transition relation)
    /// CONSISTENTLY in one manager — variable IDENTITY is preserved (only the
    /// level changes), so var-index lists (`current`/`next`) stay valid across
    /// the reorder. The shared cache is sound because a `(old_node, level)` key
    /// reorders to the same node regardless of which root reached it.
    pub fn reorder_all_into(&mut self, roots: &[NodeId], new_order: &[u32]) -> (Bdd, Vec<NodeId>) {
        let mut nb = Bdd::new();
        let max_var = new_order.iter().copied().max().unwrap_or(0) as usize;
        nb.level_of = vec![0u32; max_var + 1];
        for (lvl, &var) in new_order.iter().enumerate() {
            nb.level_of[var as usize] = lvl as u32;
        }
        let mut cache: HashMap<(NodeId, usize), NodeId> = HashMap::default();
        let new_roots = roots
            .iter()
            .map(|&r| self.reorder_into_rec(&mut nb, r, new_order, 0, &mut cache))
            .collect();
        (nb, new_roots)
    }

    /// Multi-root sifting: find a variable order that shrinks the COMBINED BDD
    /// of `roots` (over `n` vars), returning the reordered manager, the remapped
    /// roots (index-aligned), and the chosen order. Greedy adjacent-transposition
    /// hill-climb built on the sound [`Self::reorder_all_into`]; answer-preserving
    /// for every root (only the total node count shrinks). This is the driver
    /// that lets a whole reachability state (reached set + relation + frontier)
    /// be reordered together at a fixpoint safepoint.
    pub fn sift_all(&mut self, roots: &[NodeId], n: u32) -> (Bdd, Vec<NodeId>, Vec<u32>) {
        let mut best_order: Vec<u32> = (0..n).collect();
        let (mut best_bdd, mut best_roots) = self.reorder_all_into(roots, &best_order);
        let mut best_size: usize = best_roots.iter().map(|&r| best_bdd.node_count(r)).sum();
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..(n as usize).saturating_sub(1) {
                let mut cand = best_order.clone();
                cand.swap(i, i + 1);
                let (bdd, rs) = self.reorder_all_into(roots, &cand);
                let size: usize = rs.iter().map(|&r| bdd.node_count(r)).sum();
                if size < best_size {
                    best_size = size;
                    best_order = cand;
                    best_bdd = bdd;
                    best_roots = rs;
                    improved = true;
                }
            }
        }
        (best_bdd, best_roots, best_order)
    }

    /// Current variable order as `order[level] = var` (identity when no reorder
    /// has run — the production default). After any reorder, `level_of` is a full
    /// permutation of `0..num_vars`, so this inverts it.
    fn current_order(&self, num_vars: u32) -> Vec<u32> {
        if self.level_of.is_empty() {
            return (0..num_vars).collect();
        }
        let n = num_vars as usize;
        let mut order = vec![0u32; n];
        for var in 0..n {
            order[self.level_of[var] as usize] = var as u32;
        }
        order
    }

    /// Copying garbage collection: rebuild `roots` into a fresh manager under the
    /// CURRENT order, dropping every node not reachable from a root — the order
    /// (and every represented function) is UNCHANGED; only dead nodes are
    /// reclaimed. Sound because BDD reachability converges on `new == FALSE`, not
    /// on node-id stability, so a renumbering copy is safe (unlike the MDD, whose
    /// O(1) root-equality convergence needs a non-moving sweep). Reuses the
    /// validated multi-root reorder.
    pub fn compact(&mut self, roots: &[NodeId], num_vars: u32) -> (Bdd, Vec<NodeId>) {
        let order = self.current_order(num_vars);
        self.reorder_all_into(roots, &order)
    }

    /// Install a rebuilt manager `nb` (from [`Self::sift_all`] / [`Self::compact`])
    /// as the live store, carrying the abort machinery across the swap so
    /// deadlines/caps keep firing, and return the remapped reachability roots
    /// (index-aligned with the roots passed to the rebuild).
    fn install_rebuilt(&mut self, mut nb: Bdd, roots: Vec<NodeId>) -> Vec<NodeId> {
        nb.abort_probe = self.abort_probe.take();
        nb.abort_node_budget = self.abort_node_budget;
        nb.abort_flag = self.abort_flag.take();
        *self = nb;
        roots
    }

    fn reorder_into_rec(
        &mut self,
        nb: &mut Bdd,
        f: NodeId,
        new_order: &[u32],
        level: usize,
        cache: &mut HashMap<(NodeId, usize), NodeId>,
    ) -> NodeId {
        if f == Self::FALSE {
            return Self::FALSE;
        }
        if f == Self::TRUE {
            return Self::TRUE;
        }
        // All listed vars cofactored ⇒ f must be terminal (handled above); guard.
        if level == new_order.len() {
            return f;
        }
        if let Some(&id) = cache.get(&(f, level)) {
            return id;
        }
        let var = new_order[level];
        let f0 = self.restrict(f, var, false); // OLD manager, old order
        let f1 = self.restrict(f, var, true);
        let lo = self.reorder_into_rec(nb, f0, new_order, level + 1, cache);
        let hi = self.reorder_into_rec(nb, f1, new_order, level + 1, cache);
        let res = nb.mk(var, lo, hi); // NEW manager, new order — var identity kept
        cache.insert((f, level), res);
        res
    }

    /// Greedy adjacent-swap sift: search variable orders for a SMALLER BDD of
    /// `f` (over `n` variables), returning `(best_node, best_order)`. This is the
    /// scalability lever — answer-preserving (the model count is invariant; only
    /// the node count shrinks) — built on [`Self::reorder`]. `best_order[level]`
    /// is the variable at that level in the winning order.
    ///
    /// Greedy + monotone (only accepts size reductions) ⇒ terminates; each round
    /// is `O(n)` candidate reorders. This is the rebuild-based driver; the
    /// in-place sift is the later structural optimization.
    pub fn sift(&mut self, f: NodeId, n: u32) -> (NodeId, Vec<u32>) {
        let mut order: Vec<u32> = (0..n).collect();
        let mut best = self.reorder(f, &order);
        let mut best_nc = self.node_count(best);
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n.saturating_sub(1) as usize {
                let mut cand = order.clone();
                cand.swap(i, i + 1);
                let r = self.reorder(f, &cand);
                let nc = self.node_count(r);
                if nc < best_nc {
                    best = r;
                    best_nc = nc;
                    order = cand;
                    improved = true;
                }
            }
        }
        (best, order)
    }

    /// Is `a ⊆ b`? (`a ∧ ¬b == FALSE`.) Canonical, so this is exact.
    #[must_use]
    pub fn subset(&mut self, a: NodeId, b: NodeId) -> bool {
        let nb = self.not(b);
        self.and(a, nb) == Self::FALSE
    }

    /// One forward image step: the markings reachable in exactly one transition
    /// from `set`, i.e. `rename(∃current. set ∧ trans, next→current)`. `current`
    /// is the current-var list; `n2c` maps each next var back to its current var.
    pub fn post_image(
        &mut self,
        trans: NodeId,
        set: NodeId,
        current: &[u32],
        n2c: &StdHashMap<u32, u32>,
    ) -> NodeId {
        let img_next = self.and_exists(trans, set, current);
        self.rename(img_next, n2c)
    }

    /// Forward symbolic reachability fixpoint: the set of markings reachable
    /// from `init` under transition relation `trans` (a BDD over current ∪ next
    /// variables). `current` / `next` are the paired variable lists
    /// (`next[i]` renames to `current[i]`). Iterates
    /// `R ← R ∨ rename(∃current. R ∧ trans)` to a least fixpoint (monotone, so
    /// it always converges). This is the native-BDD equivalent of the symbolic
    /// reachable-set construction `tla-dd` currently gets from oxidd.
    pub fn reachable(
        &mut self,
        init: NodeId,
        trans: NodeId,
        current: &[u32],
        next: &[u32],
    ) -> NodeId {
        // No deadline ⇒ the fixpoint always converges to `Some`.
        self.reachable_within(init, trans, current, next, None)
            .expect("reachable_within(None) never times out")
    }

    /// Deadline-aware forward reachability: identical least fixpoint as
    /// [`Self::reachable`], but checks `deadline` at the top of each BFS round and
    /// returns `None` (a fail-closed DECLINE) if it has passed. This is the
    /// wall-clock budget mechanism that makes `tla-bdd` SAFE to wire into a
    /// production examination lane (the oxidd lanes run under
    /// `tla_dd::set_thread_deadline`; without this a detached tla-bdd worker would
    /// run its fixpoint unbounded — the leaked-unbounded-worker hazard). `None`
    /// deadline ⇒ run to convergence (the `reachable` behavior).
    #[must_use]
    pub fn reachable_within(
        &mut self,
        init: NodeId,
        trans: NodeId,
        current: &[u32],
        next: &[u32],
        deadline: Option<std::time::Instant>,
    ) -> Option<NodeId> {
        match self.reachable_target_within(
            init,
            trans,
            current,
            current,
            next,
            Self::FALSE,
            deadline,
        )? {
            ReachOutcome::Fixpoint { reached, .. } => Some(reached),
            // A FALSE target intersects nothing, so the generalized fixpoint
            // can never early-exit on it.
            ReachOutcome::TargetHit { .. } => unreachable!("FALSE target never hits"),
        }
    }

    /// Target-aware forward reachability: the same deadline/GC/sift-safepointed
    /// least fixpoint as [`Self::reachable_within`], generalized two ways so any
    /// frontend (Petri, TLA+, hardware) can drive it:
    ///
    /// - **`quantify` is decoupled from the renamed state pair.** Each image
    ///   step computes `rename(∃quantify. frontier ∧ trans, next→current)`.
    ///   For a pure state relation `quantify == current` (the
    ///   [`Self::reachable_within`] case); a hardware-style relation
    ///   `T(x, i, x')` with primary-input variables passes
    ///   `quantify = current ∪ inputs` so the inputs are existentially
    ///   abstracted every round.
    /// - **`target` early-exit with the transition depth.** When the target set
    ///   first intersects a newly discovered frontier ring, the fixpoint stops
    ///   and reports the ring index — the minimal number of transition steps
    ///   from `init` to a target state (0 = a target state is initial). Because
    ///   rings are checked exactly when first discovered, the reported depth is
    ///   minimal. Pass [`Bdd::FALSE`] for a plain fixpoint.
    ///
    /// Returns `None` on deadline/node-budget exhaustion (a fail-closed
    /// DECLINE), `Some(ReachOutcome::TargetHit { depth })` when a target state
    /// is reachable, and `Some(ReachOutcome::Fixpoint { reached, rounds })`
    /// when the least fixpoint converges with no target state reached — an
    /// EXACT proof that no target state is reachable.
    #[must_use]
    pub fn reachable_target_within(
        &mut self,
        init: NodeId,
        trans: NodeId,
        quantify: &[u32],
        current: &[u32],
        next: &[u32],
        target: NodeId,
        deadline: Option<std::time::Instant>,
    ) -> Option<ReachOutcome> {
        debug_assert_eq!(current.len(), next.len());
        // Arm the IN-OPERATION abort with this deadline too (audit
        // 2026-07-02): the per-round check below cannot stop a single
        // monster image round; `mk` now can, panicking with `BddAbort`
        // before any store mutation. Keep the earlier of an already-armed
        // deadline and this one, and restore the previous limits on exit.
        let saved_probe = self.abort_probe.take();
        let armed_deadline = match (saved_probe.as_ref().and_then(|p| p.deadline()), deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.abort_probe = Some(tla_resource::MemoryProbe::new(
            tla_resource::MemoryBudget::symbolic_explorer(),
            armed_deadline,
        ));
        let next_to_current: StdHashMap<u32, u32> =
            next.iter().copied().zip(current.iter().copied()).collect();
        // The store stays canonical across a `BddAbort` unwind (the panic
        // fires before mutation), so resuming use of `self` afterwards is
        // sound — that is what makes `AssertUnwindSafe` correct here.
        // Highest variable index over quantify+current+next: the sift order
        // must cover every variable the state uses (else a reorder would drop
        // levels).
        let num_vars = quantify
            .iter()
            .chain(current.iter())
            .chain(next.iter())
            .copied()
            .max()
            .map_or(0, |m| m + 1);
        // Reorder trigger: sift ONCE when the reached set's BDD first crosses
        // 3/4 of the node budget (best-effort before an abort), or every round
        // under the test stress hook. Reordering is answer-preserving.
        let sift_watermark = self.abort_node_budget.map(|b| b / 4 * 3);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // FRONTIER (BFS) reachability: image only the NEWLY-discovered
            // states each round, not all of R. Same least fixpoint, but the
            // per-round image operand is the (usually smaller) frontier
            // instead of the whole reached set — the standard symbolic-BFS
            // speedup.
            let mut r = init;
            let mut frontier = init;
            let mut trans = trans;
            let mut target = target;
            let mut sifted = false;
            // Ring 0 = init: a target state that is initial is hit at depth 0.
            if self.and(init, target) != Self::FALSE {
                return Some(ReachOutcome::TargetHit { depth: 0 });
            }
            let mut rounds = 0usize;
            loop {
                if let Some(d) = deadline {
                    if std::time::Instant::now() >= d {
                        return None; // budget exhausted ⇒ decline (fail-closed)
                    }
                }
                if self.abort_flag_raised() {
                    return None; // externally cancelled ⇒ decline (fail-closed)
                }
                // Sifting safepoint: reorder the WHOLE state (reached set +
                // relation + frontier + target) together into a smaller manager,
                // then swap it in and remap the roots. Variable identity is
                // preserved, so `quantify`/`current`/`next`/`next_to_current`
                // stay valid. The abort machinery is carried across the swap so
                // deadlines/caps keep firing. Answer-preserving, so the least
                // fixpoint is unchanged.
                let reachable = self.node_count(r);
                let want_sift = num_vars >= 2
                    && (bdd_sift_stress()
                        || (!sifted && sift_watermark.is_some_and(|w| reachable > w)));
                // GC: reclaim dead nodes by a copying rebuild to the CURRENT order
                // when the arena is ≥half garbage relative to the reached set (a
                // standard doubling-style trigger) AND has grown to a meaningful
                // fraction (1/16) of the node budget — both thresholds DERIVED
                // from the live budget, no fixed magic floor. Cheaper than sift
                // (no order search); keeps the monotone arena bounded below the cap.
                let want_compact = num_vars >= 2
                    && !want_sift
                    && (bdd_gc_stress()
                        || self.abort_node_budget.is_some_and(|budget| {
                            self.nodes.len() > reachable.saturating_mul(2)
                                && self.nodes.len() > budget / 16
                        }));
                if want_sift {
                    let (nb, roots, _order) =
                        self.sift_all(&[r, frontier, trans, target], num_vars);
                    let roots = self.install_rebuilt(nb, roots);
                    (r, frontier, trans, target) = (roots[0], roots[1], roots[2], roots[3]);
                    sifted = true;
                } else if want_compact {
                    let (nb, roots) = self.compact(&[r, frontier, trans, target], num_vars);
                    let roots = self.install_rebuilt(nb, roots);
                    (r, frontier, trans, target) = (roots[0], roots[1], roots[2], roots[3]);
                }
                // image of the frontier: ∃quantify. (frontier ∧ trans), renamed.
                let img_next = self.and_exists(frontier, trans, quantify);
                let img = self.rename(img_next, &next_to_current);
                // genuinely new states = img \ R
                let not_r = self.not(r);
                let new = self.and(img, not_r);
                if new == Self::FALSE {
                    return Some(ReachOutcome::Fixpoint { reached: r, rounds }); // least fixpoint
                }
                rounds += 1;
                // A target state first appears in the ring where it is first
                // discovered, so checking only `new` both suffices and yields
                // the MINIMAL depth (earlier rings were checked when new).
                if self.and(new, target) != Self::FALSE {
                    return Some(ReachOutcome::TargetHit { depth: rounds });
                }
                r = self.or(r, new);
                frontier = new;
            }
        }));
        self.abort_probe = saved_probe;
        match result {
            Ok(outcome) => outcome,
            Err(payload) if payload.is::<BddAbort>() => None, // decline (fail-closed)
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Number of distinct inner nodes reachable from `root` (the BDD "size").
    #[must_use]
    pub fn node_count(&self, root: NodeId) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            if Self::is_terminal(n) || !seen.insert(n) {
                continue;
            }
            let (lo, hi) = self.children(n);
            stack.push(lo);
            stack.push(hi);
        }
        seen.len()
    }

    /// EXACT model count: the number of assignments over variables `0..num_vars`
    /// that satisfy `root`. `u128`-internal; the ROBDD's structure makes this a
    /// linear-in-node-count memoised traversal (no enumeration). Variables of
    /// `root` must all be `< num_vars`.
    ///
    /// Fail-closed: returns `None` whenever the exact count cannot be
    /// guaranteed to fit in `u128` — a free-variable span wider than 127 or an
    /// intermediate multiply/add overflow. Never returns a silently clamped or
    /// saturated value.
    #[must_use]
    pub fn sat_count(&self, root: NodeId, num_vars: u32) -> Option<u128> {
        // `rec(node)` = #satisfying assignments of the sub-function over the
        // variables in `[var_of(node), num_vars)` — i.e. with the node's own
        // level as the first free variable. Free (skipped) variables between a
        // node and its child each double the count.
        let mut memo: HashMap<NodeId, u128> = HashMap::default();
        let res = self.sat_rec(root, num_vars, &mut memo)?;
        // Account for variables above the root (free) — measured in LEVELS.
        let top = self.node_level(root).min(num_vars);
        if top > 127 {
            return None;
        }
        res.checked_mul(1u128 << top)
    }

    /// `None` on any inexactness (span > 127 or `u128` overflow); only exact
    /// (`Some`) sub-counts are memoised.
    fn sat_rec(
        &self,
        node: NodeId,
        num_vars: u32,
        memo: &mut HashMap<NodeId, u128>,
    ) -> Option<u128> {
        if node == Self::FALSE {
            return Some(0);
        }
        if node == Self::TRUE {
            return Some(1);
        }
        if let Some(&c) = memo.get(&node) {
            return Some(c);
        }
        let v = self.node_level(node); // the node's ORDER LEVEL
        let (lo, hi) = self.children(node);
        let span = |child: NodeId, this: &Self| -> u32 {
            // free variables (levels) strictly between `v` and the child's level
            this.node_level(child).min(num_vars) - v - 1
        };
        let lo_span = span(lo, self);
        if lo_span > 127 {
            return None;
        }
        let cl = self
            .sat_rec(lo, num_vars, memo)?
            .checked_mul(1u128 << lo_span)?;
        let hi_span = span(hi, self);
        if hi_span > 127 {
            return None;
        }
        let ch = self
            .sat_rec(hi, num_vars, memo)?
            .checked_mul(1u128 << hi_span)?;
        let res = cl.checked_add(ch)?;
        memo.insert(node, res);
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cooperative abort (audit 2026-07-02): a node budget must stop growth,
    /// the panic must carry [`BddAbort`], fire BEFORE mutation (store stays
    /// canonical), and `reachable_within` must fold it into a `None` decline.
    #[test]
    fn abort_limits_stop_growth_and_keep_the_store_canonical() {
        let mut b = Bdd::new();
        let x = b.var(0);
        let y = b.var(1);
        let nodes_before = b.nodes.len();
        b.set_abort_limits(Some(b.nodes.len()), None);
        // A raw op that needs a FRESH node must panic with BddAbort...
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.and(x, y)));
        let payload = outcome.expect_err("budget must abort the fresh insertion");
        assert!(payload.is::<BddAbort>(), "abort payload must be BddAbort");
        // ...without mutating the store (canonicity preserved).
        assert_eq!(b.nodes.len(), nodes_before, "abort must precede mutation");
        // Cache hits and reductions still work under an exhausted budget.
        assert_eq!(b.var(0), x, "unique-table hits need no fresh node");
        // Lifting the budget resumes normal operation on the SAME manager.
        b.set_abort_limits(None, None);
        let and = b.and(x, y);
        assert!(and > Bdd::TRUE, "manager stays usable after an abort");
        // reachable_within folds an in-operation abort into a None decline:
        // arm an already-expired deadline so the very first image aborts.
        let expired = std::time::Instant::now() - std::time::Duration::from_secs(1);
        b.set_abort_limits(Some(0), Some(expired));
        assert_eq!(
            b.reachable_within(and, and, &[0], &[1], None),
            None,
            "reachable_within must decline on an in-operation abort"
        );
        b.set_abort_limits(None, None);
    }

    #[test]
    fn terminals_and_canonicity() {
        let mut b = Bdd::new();
        let a = b.var(0);
        let a2 = b.var(0);
        assert_eq!(a, a2, "the same variable must share one canonical node");
        // reduction: ite(a, x, x) == x
        let x = b.var(1);
        let red = b.ite(a, x, x);
        assert_eq!(red, x);
        // ite(a, TRUE, FALSE) == a
        assert_eq!(b.ite(a, Bdd::TRUE, Bdd::FALSE), a);
    }

    #[test]
    fn boolean_ops_and_canonical_equality() {
        let mut b = Bdd::new();
        let (a, c) = (b.var(0), b.var(2));
        let bb = b.var(1);
        // (a ∧ b) ∨ c   built two ways must be the SAME node (canonicity).
        let ab = b.and(a, bb);
        let f1 = b.or(ab, c);
        let ca = b.or(c, ab); // or is commutative
        assert_eq!(f1, ca, "canonical: commutative builds share a node");
        // De Morgan: ¬(a ∧ b) == ¬a ∨ ¬b
        let not_ab = b.not(ab);
        let na = b.not(a);
        let nb = b.not(bb);
        let dm = b.or(na, nb);
        assert_eq!(not_ab, dm, "De Morgan must hold structurally");
    }

    #[test]
    fn sat_count_exact() {
        let mut b = Bdd::new();
        let (a, bb, c) = (b.var(0), b.var(1), b.var(2));
        // f = (a ∧ b) ∨ c over 3 vars: c=1 (4) + c=0,a=1,b=1 (1) = 5.
        let ab = b.and(a, bb);
        let f = b.or(ab, c);
        assert_eq!(b.sat_count(f, 3), Some(5));
        // constants
        assert_eq!(b.sat_count(Bdd::FALSE, 3), Some(0));
        assert_eq!(b.sat_count(Bdd::TRUE, 3), Some(8));
        // single var over 4 vars: a=1 → 2^3 = 8.
        assert_eq!(b.sat_count(a, 4), Some(8));
    }

    /// Regression (finding: sat_count silently clamped >127-var shifts and
    /// saturated u128 while documented EXACT): counts that are not exactly
    /// representable must come back `None`, never a clamped value.
    #[test]
    fn sat_count_fails_closed_beyond_127_vars() {
        let mut b = Bdd::new();
        let a = b.var(0);

        // Exact boundary: 2^127 fits in u128.
        assert_eq!(b.sat_count(Bdd::TRUE, 127), Some(1u128 << 127));
        assert_eq!(b.sat_count(a, 128), Some(1u128 << 127));
        // 2^128 does not: fail closed.
        assert_eq!(b.sat_count(Bdd::TRUE, 128), None);
        // x0 over 200 free vars has 2^199 models: fail closed, not clamped.
        assert_eq!(b.sat_count(a, 200), None);
        // >127 free vars above the root (a var-129 literal over 130 vars).
        let hi = b.var(129);
        assert_eq!(b.sat_count(hi, 130), None);
    }

    #[test]
    fn xor_and_node_count() {
        let mut b = Bdd::new();
        let (a, bb) = (b.var(0), b.var(1));
        let x = b.xor(a, bb);
        // a⊕b over 2 vars has exactly 2 models (01, 10).
        assert_eq!(b.sat_count(x, 2), Some(2));
        // ⋀(xi↔yi)-style: interleaved a↔b is a small BDD; just assert it's finite
        // and the equivalence has 2 models (00, 11) over 2 vars.
        let eq = b.not(x);
        assert_eq!(b.sat_count(eq, 2), Some(2));
        assert!(b.node_count(x) >= 1);
    }

    #[test]
    fn exists_and_relational_product() {
        let mut b = Bdd::new();
        let (a, bb) = (b.var(0), b.var(1));
        // ∃b. (a ∧ b) == a   (b can be set true to satisfy when a holds)
        let ab = b.and(a, bb);
        assert_eq!(b.exists(ab, &[1]), a);
        // ∃a. (a ∨ b) == TRUE (pick a = true)
        let aorb = b.or(a, bb);
        assert_eq!(b.exists(aorb, &[0]), Bdd::TRUE);
        // ∃a. (a ∧ ¬a-ish): ∃a. a == TRUE; ∃a,b. (a∧b) == TRUE
        assert_eq!(b.exists(a, &[0]), Bdd::TRUE);
        assert_eq!(b.exists(ab, &[0, 1]), Bdd::TRUE);
        // quantifying a variable the function ignores is a no-op.
        assert_eq!(b.exists(a, &[2]), a);
        // fused relational product: ∃b. (a ∧ b) == a (the image primitive).
        assert_eq!(b.and_exists(a, bb, &[1]), a);
        // ∃ over (a∧b)∨c of {a,b} = c ∨ (∃a,b. a∧b) = c ∨ TRUE = TRUE.
        let (c, _) = (b.var(2), 0);
        let abc = {
            let ab2 = b.and(a, bb);
            b.or(ab2, c)
        };
        assert_eq!(b.exists(abc, &[0, 1]), Bdd::TRUE);
        // ∃a. ((a∧b)∨c) = b ∨ c (a=1 ⇒ b∨c, a=0 ⇒ c; OR = b∨c).
        let bc = b.or(bb, c);
        assert_eq!(b.exists(abc, &[0]), bc);
    }

    #[test]
    fn eval_matches_truth_table() {
        let mut b = Bdd::new();
        let (a, bb, c) = (b.var(0), b.var(1), b.var(2));
        let ab = b.and(a, bb);
        let f = b.or(ab, c); // (a∧b)∨c
                             // exhaustive truth table over 3 vars.
        for bits in 0u8..8 {
            let asn = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
            let expect = (asn[0] && asn[1]) || asn[2];
            assert_eq!(b.eval(f, &asn), expect, "eval mismatch at {asn:?}");
        }
        // constants
        assert!(b.eval(Bdd::TRUE, &[false, false, false]));
        assert!(!b.eval(Bdd::FALSE, &[true, true, true]));
        // eval ∘ reorder check is only meaningful for the var-PRESERVING reorder;
        // the current rebuild-relabel reorder relabels vars, so eval-equality is
        // NOT expected there — eval exists to validate the structural reorder
        // (var↔level decoupling) when it lands.
    }

    #[test]
    fn restrict_and_reorder_preserve_meaning() {
        let mut b = Bdd::new();
        let (a, bb, c) = (b.var(0), b.var(1), b.var(2));
        let ab = b.and(a, bb);
        let f = b.or(ab, c); // (a∧b)∨c
                             // general restrict: f[a:=1] = b∨c ; f[a:=0] = c.
        let bc = b.or(bb, c);
        assert_eq!(b.restrict(f, 0, true), bc);
        assert_eq!(b.restrict(f, 0, false), c);
        // restrict by an absent/independent var is a no-op (var 5 not in f).
        assert_eq!(b.restrict(f, 5, true), f);

        // reorder is answer-preserving in the permutation-invariant sense: the
        // MODEL COUNT is unchanged while the variable order (and node count) may
        // change. g = (x0∧x1)∨(x2∧x3) over 4 vars has 7 models.
        let (x0, x1, x2, x3) = (b.var(0), b.var(1), b.var(2), b.var(3));
        let g = {
            let a0 = b.and(x0, x1);
            let a1 = b.and(x2, x3);
            b.or(a0, a1)
        };
        assert_eq!(b.sat_count(g, 4), Some(7));
        // a non-trivial permutation (swap middle levels) preserves the count.
        let r = b.reorder(g, &[0, 2, 1, 3]);
        assert_eq!(
            b.sat_count(r, 4),
            Some(7),
            "reorder must preserve the model count"
        );
        // identity reorder also preserves it.
        let r_id = b.reorder(g, &[0, 1, 2, 3]);
        assert_eq!(b.sat_count(r_id, 4), Some(7));

        // sift: answer-preserving, and never worse than the identity order.
        let r_idnc = b.reorder(g, &[0, 1, 2, 3]);
        let id_nc = b.node_count(r_idnc);
        let (best, order) = b.sift(g, 4);
        assert_eq!(
            b.sat_count(best, 4),
            Some(7),
            "sift preserves the model count"
        );
        assert!(
            b.node_count(best) <= id_nc,
            "sift never increases node count"
        );
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn reorder_into_preserves_function_var_identity() {
        let mut b = Bdd::new();
        let (x0, x1, x2, x3) = (b.var(0), b.var(1), b.var(2), b.var(3));
        // g = (x0∧x1)∨(x2∧x3) over 4 vars.
        let g = {
            let a0 = b.and(x0, x1);
            let a1 = b.and(x2, x3);
            b.or(a0, a1)
        };
        // var-preserving reorder under a non-trivial permutation.
        let order = [0u32, 2, 1, 3];
        let (nb, root) = b.reorder_into(g, &order);
        // STRONG answer-preservation: identical on EVERY assignment (var identity
        // preserved — distinguishes this from the relabeling `reorder`).
        for bits in 0u8..16 {
            let a = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
            assert_eq!(
                nb.eval(root, &a),
                b.eval(g, &a),
                "reorder changed the function at {a:?}"
            );
        }
        // count preserved.
        assert_eq!(nb.sat_count(root, 4), b.sat_count(g, 4));
        // identity reorder is also function-preserving.
        let (nb2, root2) = b.reorder_into(g, &[0, 1, 2, 3]);
        assert_eq!(nb2.sat_count(root2, 4), b.sat_count(g, 4));
    }

    #[test]
    fn reorder_all_into_preserves_every_function() {
        // Multi-root reorder: reorder TWO functions into ONE manager and confirm
        // BOTH are preserved on EVERY assignment (var identity kept, shared cache
        // sound). This is the prerequisite for reordering a whole reachability
        // state (reach + trans) consistently.
        let mut b = Bdd::new();
        let (x0, x1, x2, x3) = (b.var(0), b.var(1), b.var(2), b.var(3));
        let f = {
            let a = b.and(x0, x1);
            let c = b.and(x2, x3);
            b.or(a, c) // (x0∧x1)∨(x2∧x3) — shares subgraphs with g below
        };
        let g = {
            let a = b.or(x0, x2);
            b.and(a, x3) // (x0∨x2)∧x3
        };
        let order = [0u32, 2, 1, 3];
        let (nb, roots) = b.reorder_all_into(&[f, g], &order);
        assert_eq!(roots.len(), 2);
        for bits in 0u8..16 {
            let a = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
            assert_eq!(nb.eval(roots[0], &a), b.eval(f, &a), "f changed at {a:?}");
            assert_eq!(nb.eval(roots[1], &a), b.eval(g, &a), "g changed at {a:?}");
        }
        assert_eq!(nb.sat_count(roots[0], 4), b.sat_count(f, 4));
        assert_eq!(nb.sat_count(roots[1], 4), b.sat_count(g, 4));
    }

    #[test]
    fn reachable_stable_under_forced_bdd_sift() {
        // 2-bit counter: current [0,1]=(c0,c1), next [2,3]=(n0,n1); +1 mod 4.
        // Reachable from state 0 = all 4 states. Forcing a sift (whole-state
        // reorder + manager swap) at EVERY BFS round must not change the count —
        // validates the mid-fixpoint remap of reach/frontier/trans and the
        // carried-across-swap abort machinery.
        fn build() -> (Bdd, NodeId, NodeId) {
            let mut b = Bdd::new();
            let (c0, c1, n0, n1) = (b.var(0), b.var(1), b.var(2), b.var(3));
            let (nc0, nc1, nn0, nn1) = (b.not(c0), b.not(c1), b.not(n0), b.not(n1));
            let t0 = {
                let a = b.and(nc0, nc1);
                let x = b.and(n0, nn1);
                b.and(a, x)
            }; // 0→1
            let t1 = {
                let a = b.and(c0, nc1);
                let x = b.and(nn0, n1);
                b.and(a, x)
            }; // 1→2
            let t2 = {
                let a = b.and(nc0, c1);
                let x = b.and(n0, n1);
                b.and(a, x)
            }; // 2→3
            let t3 = {
                let a = b.and(c0, c1);
                let x = b.and(nn0, nn1);
                b.and(a, x)
            }; // 3→0
            let trans = {
                let x = b.or(t0, t1);
                let y = b.or(t2, t3);
                b.or(x, y)
            };
            let init = b.and(nc0, nc1); // state 0
            (b, init, trans)
        }
        let (current, next) = ([0u32, 1], [2u32, 3]);

        let (mut b, init, trans) = build();
        let r = b.reachable(init, trans, &current, &next);
        let normal = b.sat_count(r, 2);
        assert_eq!(normal, Some(4), "counter reaches all 4 states");

        let (mut b2, init2, trans2) = build();
        set_bdd_sift_stress(true);
        let r2 = b2.reachable(init2, trans2, &current, &next);
        set_bdd_sift_stress(false);
        assert_eq!(
            b2.sat_count(r2, 2),
            normal,
            "forcing a whole-state sift every BFS round must not change the count"
        );

        // Same for the copying-GC compaction path: forcing a rebuild-to-current-
        // order (drops dead nodes, keeps the order) at EVERY round must also
        // preserve the count and remap the roots correctly.
        let (mut b3, init3, trans3) = build();
        set_bdd_gc_stress(true);
        let r3 = b3.reachable(init3, trans3, &current, &next);
        set_bdd_gc_stress(false);
        assert_eq!(
            b3.sat_count(r3, 2),
            normal,
            "forcing a copying-GC compaction every BFS round must not change the count"
        );
    }

    #[test]
    fn reachable_target_reports_minimal_depth_under_rebuild_stress() {
        // 2-bit counter 0→1→2→3→0 (current vars 0,1; next vars 2,3), init=0.
        fn build() -> (Bdd, NodeId, NodeId, NodeId) {
            let mut b = Bdd::new();
            let c0 = b.var(0);
            let c1 = b.var(1);
            let n0 = b.var(2);
            let n1 = b.var(3);
            let (nc0, nc1, nn0, nn1) = {
                let a = b.not(c0);
                let x = b.not(c1);
                let y = b.not(n0);
                let z = b.not(n1);
                (a, x, y, z)
            };
            let t0 = {
                let a = b.and(nc0, nc1);
                let x = b.and(n0, nn1);
                b.and(a, x)
            }; // 0→1
            let t1 = {
                let a = b.and(c0, nc1);
                let x = b.and(nn0, n1);
                b.and(a, x)
            }; // 1→2
            let t2 = {
                let a = b.and(nc0, c1);
                let x = b.and(n0, n1);
                b.and(a, x)
            }; // 2→3
            let t3 = {
                let a = b.and(c0, c1);
                let x = b.and(nn0, nn1);
                b.and(a, x)
            }; // 3→0
            let trans = {
                let x = b.or(t0, t1);
                let y = b.or(t2, t3);
                b.or(x, y)
            };
            let init = b.and(nc0, nc1); // state 0
            let target = b.and(c0, c1); // state 3
            (b, init, trans, target)
        }
        let (current, next) = ([0u32, 1], [2u32, 3]);

        // Plain: state 3 is exactly 3 steps from state 0.
        let (mut b, init, trans, target) = build();
        let hit = b
            .reachable_target_within(init, trans, &current, &current, &next, target, None)
            .unwrap();
        assert_eq!(hit, ReachOutcome::TargetHit { depth: 3 });

        // Depth 0: the target contains the initial state.
        let (mut b0, init0, trans0, _) = build();
        let hit0 = b0
            .reachable_target_within(init0, trans0, &current, &current, &next, init0, None)
            .unwrap();
        assert_eq!(hit0, ReachOutcome::TargetHit { depth: 0 });

        // Under forced sift + GC rebuilds every round, the TARGET root must be
        // remapped with the other three roots — same minimal depth.
        for (set_stress, label) in [
            (set_bdd_sift_stress as fn(bool) -> bool, "sift"),
            (set_bdd_gc_stress as fn(bool) -> bool, "gc"),
        ] {
            let (mut bs, inits, transs, targets) = build();
            set_stress(true);
            let hits = bs
                .reachable_target_within(inits, transs, &current, &current, &next, targets, None)
                .unwrap();
            set_stress(false);
            assert_eq!(
                hits,
                ReachOutcome::TargetHit { depth: 3 },
                "{label}-stress rebuild must remap the target root"
            );
        }

        // FALSE target = plain fixpoint: full 4-state set, rounds reported.
        let (mut bf, initf, transf, _) = build();
        let fix = bf
            .reachable_target_within(initf, transf, &current, &current, &next, Bdd::FALSE, None)
            .unwrap();
        match fix {
            ReachOutcome::Fixpoint { reached, rounds } => {
                assert_eq!(bf.sat_count(reached, 2), Some(4));
                assert_eq!(rounds, 3, "3 image rounds discover states 1,2,3");
            }
            other => panic!("expected fixpoint, got {other:?}"),
        }
    }

    #[test]
    fn inductive_check_has_teeth() {
        // 1-bit toggle: cur var 0, next var 1; 0↔1. trans = (¬c∧n)∨(c∧¬n).
        let mut b = Bdd::new();
        let c = b.var(0);
        let n = b.var(1);
        let nc = b.not(c);
        let nn = b.not(n);
        let t01 = b.and(nc, n);
        let t10 = b.and(c, nn);
        let trans = b.or(t01, t10);
        let current = [0u32];
        let n2c: StdHashMap<u32, u32> = [(1u32, 0u32)].into_iter().collect();
        let init = nc; // state 0
                       // A too-small candidate R = {0}: post_image = {1} ⊄ {0} ⇒ NOT inductive.
        let img_bad = b.post_image(trans, init, &current, &n2c);
        assert!(
            !b.subset(img_bad, init),
            "the check must REJECT a non-closed set"
        );
        // The true reachable R = {0,1} = TRUE over var 0: closed.
        let full = Bdd::TRUE;
        let img_full = b.post_image(trans, full, &current, &n2c);
        assert!(b.subset(init, full));
        assert!(
            b.subset(img_full, full),
            "the check must ACCEPT the closed reachable set"
        );
    }

    #[test]
    fn linear_le_threshold_bdd() {
        let mut b = Bdd::new();
        // Σ x_i <= 1 over 3 unit vars: #models = C(3,0)+C(3,1) = 4.
        let le1 = b.linear_le(&[(0, 1), (1, 1), (2, 1)], 1);
        assert_eq!(b.sat_count(le1, 3), Some(4));
        // <= 3 over 3 vars: all 8.
        let le3 = b.linear_le(&[(0, 1), (1, 1), (2, 1)], 3);
        assert_eq!(b.sat_count(le3, 3), Some(8));
        assert_eq!(le3, Bdd::TRUE);
        // <= -1: none.
        let lem1 = b.linear_le(&[(0, 1), (1, 1)], -1);
        assert_eq!(lem1, Bdd::FALSE);
        // weighted: 2·x0 + x1 <= 1 over 2 vars: (0,0),(0,1) ⇒ 2 models.
        let w = b.linear_le(&[(0, 2), (1, 1)], 1);
        assert_eq!(b.sat_count(w, 2), Some(2));
    }

    #[test]
    fn native_symbolic_reachability_count() {
        // The native-BDD equivalent of oxidd-backed reachability, cross-checked
        // against the explicit reachable-state count.
        //
        // System: a 2-bit counter mod 4. State = (x0 low, x1 high). Current vars
        // {0,1} (above), next vars {2,3} (below) — so the reachable set R is over
        // {0,1} and `sat_count(R, 2)` counts exactly the current bits. From 0 it
        // reaches all 4 states (0→1→2→3→0).
        let mut b = Bdd::new();
        let (x0, x1) = (b.var(0), b.var(1)); // current low/high bit
        let (n0, n1) = (b.var(2), b.var(3)); // next low/high bit
        let nx0 = b.not(x0);
        let nx1 = b.not(x1);
        let nn0 = b.not(n0);
        let nn1 = b.not(n1);
        // increment relation over (x1 x0) -> (n1 n0):
        //  00->01, 01->10, 10->11, 11->00
        let clause = |b: &mut Bdd, a, c, d, e| {
            let ab = b.and(a, c);
            let de = b.and(d, e);
            b.and(ab, de)
        };
        let t00 = clause(&mut b, nx1, nx0, nn1, n0); // 00 -> 01
        let t01 = clause(&mut b, nx1, x0, n1, nn0); // 01 -> 10
        let t10 = clause(&mut b, x1, nx0, n1, n0); // 10 -> 11
        let t11 = clause(&mut b, x1, x0, nn1, nn0); // 11 -> 00
        let trans = {
            let a = b.or(t00, t01);
            let c = b.or(t10, t11);
            b.or(a, c)
        };
        let init = b.and(nx1, nx0); // start at 00
        let r = b.reachable(init, trans, &[0, 1], &[2, 3]);
        // All 4 states reachable ⇒ over the 2 current vars {0,1}, |R| = 4.
        assert_eq!(
            b.sat_count(r, 2),
            Some(4),
            "2-bit counter reaches all 4 states"
        );

        // A non-strongly-connected system: only the increment 00->01 (no other
        // transitions) ⇒ reachable {00, 01} = 2 states.
        let init2 = b.and(nx1, nx0);
        let r2 = b.reachable(init2, t00, &[0, 1], &[2, 3]);
        assert_eq!(
            b.sat_count(r2, 2),
            Some(2),
            "single-step system reaches 2 states"
        );

        // Deadline support (the wiring budget mechanism): a None deadline gives
        // the same fixpoint; an already-passed deadline DECLINES (fail-closed).
        let r_nd = b
            .reachable_within(init, trans, &[0, 1], &[2, 3], None)
            .expect("None deadline never declines");
        assert_eq!(
            b.sat_count(r_nd, 2),
            Some(4),
            "reachable_within(None) == reachable"
        );
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        assert_eq!(
            b.reachable_within(init, trans, &[0, 1], &[2, 3], Some(past)),
            None,
            "a passed deadline must decline (fail-closed), not return a partial set"
        );
    }
}
