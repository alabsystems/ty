// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! The reachability / exact-count kernel for bounded P/T nets.
//!
//! # Net model
//!
//! [`MddNet`] mirrors the `tla-dd::DdNetSpec` shape one-for-one so the two
//! lanes consume identical inputs and can be cross-checked directly:
//! per-place `bounds`, an `initial_marking`, and a list of transitions each
//! with `pre` / `post` per-place vectors. The firing rule is exactly the BDD
//! lane's BFS oracle:
//!
//! - **Enabled** at marking `m` iff `m[p] >= pre[p]` for every place `p`.
//! - **Fire**: `next[p] = m[p] - pre[p] + post[p]`. If any `next[p] >
//!   bounds[p]`, the firing is **rejected wholesale** (the successor is not in
//!   the bounded state space), matching the oracle's bound-truncation.
//!
//! # Kernel shape (FIRST INCREMENT — symbolic SET, explicit IMAGE)
//!
//! The reachable set `R` is held as a single canonical MDD — compact,
//! deduplicated, one level per place. That symbolic representation is the
//! whole point: on counter / conserved nets it stays small where an explicit
//! `HashSet<Vec<u64>>` or a bit-blasted BDD would blow up.
//!
//! The image step in this first increment is *explicit per-marking firing*: a
//! chaining fixpoint enumerates the current frontier's markings out of the
//! MDD, fires every transition, and unions the resulting singletons back into
//! `R`. So:
//!
//! - The **state SET** is symbolic (compact MDD union/count).
//! - The **image** is explicit (enumerate-and-fire), not yet a relational
//!   product over a transition-relation MDD.
//!
//! This is deliberately the simplest correct kernel; the next increment
//! (documented in the crate README / report) replaces the explicit image with
//! a true symbolic relational-product image + saturation so the per-iteration
//! cost stops depending on `|frontier|`. Correctness is identical either way —
//! and that is what the differential proptest battery pins down.
//!
//! # Soundness
//!
//! Every reported count is cross-checked against `tla-dd`'s explicit BFS
//! oracle (`bfs_reachable_set_count`) — the same ground truth the production
//! BDD lane validates against — on a random-net proptest battery, 0
//! disagreements. The kernel is **gate-only**: it is a new engine and must not
//! feed production verdicts until that battery has soaked. Overflow and
//! resource limits are fail-closed: [`CountError`] is returned (never a
//! wrapped/garbage count, never an unbounded loop).

use crate::node::{MddRef, MddStore};

/// A transition: per-place tokens consumed (`pre`) and produced (`post`).
///
/// Both vectors are indexed by place and must have length equal to the net's
/// place count. Mirrors `tla-dd::DdTransition`.
#[derive(Debug, Clone)]
pub struct MddTransition {
    /// Tokens consumed per place when this transition fires.
    pub pre: Vec<u64>,
    /// Tokens produced per place when this transition fires.
    pub post: Vec<u64>,
}

/// A bounded P/T net for the MDD kernel. Mirrors `tla-dd::DdNetSpec`.
#[derive(Debug, Clone)]
pub struct MddNet {
    /// Per-place token upper bound (place `p` domain is `0..=bounds[p]`).
    pub bounds: Vec<u64>,
    /// Initial marking. Each `initial_marking[p]` must satisfy `<= bounds[p]`.
    pub initial_marking: Vec<u64>,
    /// Transitions; each `pre`/`post` has length `bounds.len()`.
    pub transitions: Vec<MddTransition>,
}

/// Why the kernel declined to produce a count (fail-closed; never a wrong
/// answer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CountError {
    /// The net is malformed: a length mismatch among `bounds` /
    /// `initial_marking` / a transition's `pre`/`post`, or an initial marking
    /// out of range.
    Malformed(String),
    /// The exact reachable-state count exceeded `u64::MAX`, so it cannot be
    /// reported soundly. (Internal arithmetic is `u128`/saturating, so this is
    /// a genuine over-`u64` state space, not an arithmetic artifact.)
    CountOverflow,
    /// A hard resource cap was hit before the fixpoint converged (node count
    /// or iteration backstop). The caller must DECLINE — a partial set would
    /// under-count.
    ResourceCap(String),
}

/// Hard ceiling on live interior MDD nodes. A small-net kernel: if a net needs
/// more than this, we DECLINE (the production engine falls through to another
/// lane) rather than risk an OOM. Generous enough for the proptest battery and
/// small MCC fixtures; this is the "small bounded nets" scope.
/// Interior-node cap, DERIVED from effective memory (was a fixed 4_000_000).
/// Adaptive to the machine/confinement via the shared node-store budget.
#[inline]
fn max_interior_nodes() -> usize {
    crate::node::max_interior_nodes() / 2
}

use crate::sift_runtime::{singleton_ordered, to_place_marking, want_sift};

/// Iteration backstop for the chaining fixpoint. The fixpoint is monotone
/// (R only grows, bounded by `Π(bound+1)`), so it always converges; this is a
/// pure safety net against a logic bug, not a semantic limit.
const MAX_ITERATIONS: u32 = 100_000_000;

/// Result of a reachability run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReachResult {
    /// `|R|` — exact number of reachable markings.
    pub state_count: u64,
    /// Number of fixpoint iterations / rounds performed (diagnostic). For
    /// saturation this is `1` (saturation is not a round-based fixpoint).
    pub iterations: u32,
    /// Live interior MDD node count at convergence (diagnostic — the figure
    /// that stays small on counter/conserved nets).
    pub interior_nodes: usize,
    /// Peak live interior MDD node count observed at any point during the run.
    /// This is the memory-relevant figure: saturation's whole purpose is to
    /// keep this peak far below the breadth-first relprod peak on conserved /
    /// counter nets. For the explicit kernel we report the convergence count
    /// (it does not track a separate transient peak).
    pub peak_interior_nodes: usize,
}

impl MddNet {
    /// Validate net shape (lengths + initial-marking range). Fail-closed.
    pub(crate) fn validate(&self) -> Result<(), CountError> {
        let n = self.bounds.len();
        if self.initial_marking.len() != n {
            return Err(CountError::Malformed(format!(
                "initial marking has {} entries, net has {} places",
                self.initial_marking.len(),
                n
            )));
        }
        for (p, (&m0, &b)) in self.initial_marking.iter().zip(&self.bounds).enumerate() {
            if m0 > b {
                return Err(CountError::Malformed(format!(
                    "initial marking[{p}] = {m0} exceeds bound {b}"
                )));
            }
        }
        for (ti, t) in self.transitions.iter().enumerate() {
            if t.pre.len() != n || t.post.len() != n {
                return Err(CountError::Malformed(format!(
                    "transition {ti}: pre/post length mismatch (pre={}, post={}, places={n})",
                    t.pre.len(),
                    t.post.len()
                )));
            }
        }
        Ok(())
    }

    /// Compute the exact reachable-state count via the symbolic-set chaining
    /// fixpoint. Fail-closed on every error path.
    ///
    /// The [`crate::catch_mdd_abort`] wrapper folds the store's per-node
    /// cooperative abort probe (a live footprint ceiling + collective
    /// free-memory floor, armed below) into the same `ResourceCap` decline as
    /// the explicit per-successor caps — so this deadline-less oracle also backs
    /// off under real memory pressure mid-round, not only at each union check.
    pub fn reachable_count(&self) -> Result<ReachResult, CountError> {
        match crate::catch_mdd_abort(|| self.reachable_count_inner()) {
            Some(r) => r,
            None => Err(CountError::ResourceCap(
                "mdd chaining cooperative abort (footprint floor) hit mid-round".to_string(),
            )),
        }
    }

    fn reachable_count_inner(&self) -> Result<ReachResult, CountError> {
        self.validate()?;

        let mut store = MddStore::new(self.bounds.clone());
        // Footprint/collective-floor probe (no wall-clock deadline on this
        // oracle path); armed so a monster round backs off under real memory
        // pressure, cooperatively with concurrent MCC solvers.
        store.set_abort_probe(None);
        // Current variable order: `order[level] = place` tested at that level.
        // Sifting (below) permutes it; every marking↔MDD conversion goes through
        // it. Identity until the first reorder, so the non-sifting path is
        // byte-for-byte the previous behaviour (level == place).
        let mut order: Vec<usize> = (0..self.bounds.len()).collect();
        // R := { initial_marking }.
        let init = singleton_ordered(&mut store, &self.initial_marking, &order);
        // initial_marking was range-checked in validate(), so this is non-ZERO
        // unless the net has zero places (degenerate but valid: 1 reachable
        // marking, the empty marking).
        let mut reach = init;

        // The chaining fixpoint. We hold the *whole* reachable set as `reach`
        // and, each iteration, enumerate its markings, fire every transition,
        // and union the successors in. Convergence is detected by CANONICAL ROOT
        // EQUALITY (`next == reach`) — an O(1) `MddRef` comparison — not a
        // per-round model-count: the store is a fully-reduced MDD (equal sets
        // share a root) and R is monotone, so `next == reach` iff R stopped
        // growing. The count is taken ONCE, at the fixpoint.
        let mut iterations: u32 = 0;
        // Sift at most once per run — when the MDD first approaches the node cap.
        let mut sifted = false;
        let sift_watermark = max_interior_nodes() / 4 * 3;

        loop {
            iterations += 1;
            if iterations > MAX_ITERATIONS {
                return Err(CountError::ResourceCap(
                    "iteration backstop exceeded".to_string(),
                ));
            }

            // GC safepoint (non-moving mark-sweep). `reach` is the ONLY live
            // root here and no MddRef-valued cache spans this point (union builds
            // a fresh cache per call), so freeing unreachable interior nodes
            // makes the node/byte cap LIVE — reclaimed each round instead of
            // accumulating across the whole run — without disturbing the
            // canonical root or the `next == reach` convergence witness.
            if store.should_collect() {
                store.gc(&[reach]);
            }

            // Sifting safepoint (dynamic variable reordering). When the reachable
            // set's MDD grows large, shrink it by reordering so the node cap is
            // pushed out. `reach` is the ONLY live root, so the store swap is a
            // clean single-root remap; `order` is composed with the chosen
            // reorder. Bounded: at most once per run (the test hook forces every
            // round to exercise the permutation), only for nets small enough that
            // the O(L²) sift is worth it. Semantics-preserving, so it never
            // disturbs the set or the `next == reach` fixpoint.
            if want_sift(&store, self.bounds.len(), sifted, sift_watermark) {
                let (new_store, new_roots, chosen) = store.sift(&[reach]);
                reach = new_roots[0];
                store = new_store;
                order = crate::sift_runtime::compose_order(&order, &chosen);
                sifted = true;
            }

            // Enumerate current reachable markings and fire all transitions.
            let markings = enumerate(&store, reach);
            let mut next = reach;
            for lm in &markings {
                let m = to_place_marking(lm, &order);
                for t in &self.transitions {
                    if let Some(succ) = fire(&self.bounds, &m, t) {
                        let s = singleton_ordered(&mut store, &succ, &order);
                        next = store.union(next, s);
                        if store.interior_node_count() > max_interior_nodes()
                            || store.approx_store_bytes() > crate::node::max_store_bytes()
                        {
                            return Err(CountError::ResourceCap(format!(
                                "interior node cap {} or store byte cap exceeded",
                                max_interior_nodes()
                            )));
                        }
                    }
                }
            }

            if next == reach {
                // Fixpoint: R unchanged this round ⇒ converged. Count once.
                return Ok(ReachResult {
                    state_count: self.count_or_err(&store, reach)?,
                    iterations,
                    interior_nodes: store.interior_node_count(),
                    peak_interior_nodes: store.interior_node_count(),
                });
            }
            reach = next;
        }
    }

    #[inline]
    pub(crate) fn count_or_err(&self, store: &MddStore, root: MddRef) -> Result<u64, CountError> {
        store.count_markings(root).ok_or(CountError::CountOverflow)
    }

    /// Arbitrary-precision (`BigUint`) exact count. NEVER declines on magnitude:
    /// the MDD already represents the set exactly, so `|R|` is exact for ANY
    /// finite reachable set — including reachable sets `> u128::MAX` (e.g. FMS
    /// ≈1e47, Kanban/Philosophers ≈1e238) that overflow the `u64`/`u128`
    /// carriers.
    ///
    /// The fixpoint drivers no longer use this as their convergence measure —
    /// they now converge on O(1) canonical ROOT EQUALITY (`next == reach`), which
    /// is exact at any magnitude AND avoids re-counting the diagram every round.
    /// Retained as the exact-count primitive (validated by the `set_ops`
    /// above-`u128` tests) for callers that need a magnitude-independent count.
    #[inline]
    #[allow(dead_code)] // exact-count primitive; exercised by the set_ops battery
    pub(crate) fn count_big(&self, store: &MddStore, root: MddRef) -> tla_bignum::BigUint {
        store.count_markings_big(root)
    }
}

/// Fire `t` at marking `m` under per-place `bounds`. Returns the successor, or
/// `None` if disabled or if any place would exceed its bound (bound-truncated
/// out of the state space — exactly the BDD-lane BFS oracle's rule).
fn fire(bounds: &[u64], m: &[u64], t: &MddTransition) -> Option<Vec<u64>> {
    // Enabled check.
    if !m.iter().zip(&t.pre).all(|(&mv, &pv)| mv >= pv) {
        return None;
    }
    let mut next = vec![0u64; m.len()];
    for p in 0..m.len() {
        // m[p] >= pre[p] (checked above), so the subtraction cannot underflow.
        let v = m[p] - t.pre[p] + t.post[p];
        if v > bounds[p] {
            return None;
        }
        next[p] = v;
    }
    Some(next)
}

/// Enumerate every full marking in the set represented by `root`.
///
/// Walks the MDD top-down, materializing each path. Skipped (redundant) levels
/// are expanded to their full domain. Used only to drive the explicit image in
/// this first increment; the next increment's relational-product image removes
/// the need to enumerate.
fn enumerate(store: &MddStore, root: MddRef) -> Vec<Vec<u64>> {
    let n = store.num_levels();
    let mut out = Vec::new();
    if root.is_zero() {
        return out;
    }
    let mut current = vec![0u64; n];
    enumerate_rec(store, root, 0, &mut current, &mut out);
    out
}

fn enumerate_rec(
    store: &MddStore,
    node: MddRef,
    level: usize,
    current: &mut Vec<u64>,
    out: &mut Vec<Vec<u64>>,
) {
    let n = store.num_levels();
    if level == n {
        // Reached past the last place level. A path that survives here is a
        // completed in-set marking iff it terminates at ONE (ZERO subtrees are
        // pruned before recursion, so reaching here means ONE).
        debug_assert!(node.is_one());
        out.push(current.clone());
        return;
    }

    if node.is_zero() {
        return;
    }

    let node_level = store.level_of(node);
    if node.is_one() || node_level as usize > level {
        // This place level is skipped/free ⇒ every value 0..=bound is allowed;
        // recurse into the same node at the next level.
        for v in 0..=store.bounds[level] {
            current[level] = v;
            enumerate_rec(store, node, level + 1, current, out);
        }
    } else {
        // node sits exactly at this level: follow each non-ZERO edge.
        debug_assert_eq!(node_level as usize, level);
        for v in 0..store.domain_size(node_level) as u64 {
            let child = store.child(node, v);
            if child.is_zero() {
                continue;
            }
            current[level] = v;
            enumerate_rec(store, child, level + 1, current, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(pre: Vec<u64>, post: Vec<u64>) -> MddTransition {
        MddTransition { pre, post }
    }

    #[test]
    fn no_transitions_one_state() {
        let net = MddNet {
            bounds: vec![1, 1],
            initial_marking: vec![0, 1],
            transitions: vec![],
        };
        let r = net.reachable_count().expect("ok");
        assert_eq!(r.state_count, 1);
    }

    #[test]
    fn single_token_shuttle_two_states() {
        // p0 -> p1 -> p0; one token; reachable: (1,0) and (0,1).
        let net = MddNet {
            bounds: vec![1, 1],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])],
        };
        let r = net.reachable_count().expect("ok");
        assert_eq!(r.state_count, 2);
    }

    #[test]
    fn counter_net_full_range() {
        // One place, t increments it; bound 5; reachable: 0..=5 ⇒ 6 states.
        let net = MddNet {
            bounds: vec![5],
            initial_marking: vec![0],
            transitions: vec![t(vec![0], vec![1])],
        };
        let r = net.reachable_count().expect("ok");
        assert_eq!(r.state_count, 6);
    }

    #[test]
    fn two_independent_counters_product() {
        // Two independent places each 0..=3 ⇒ 4*4 = 16 reachable markings.
        let net = MddNet {
            bounds: vec![3, 3],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        };
        let r = net.reachable_count().expect("ok");
        assert_eq!(r.state_count, 16);
    }

    #[test]
    fn reachable_count_is_stable_under_forced_gc() {
        // End-to-end GC soundness: forcing gc(&[reach]) at EVERY round (via the
        // stress hook) must not change the reachable count — this validates that
        // the reach.rs safepoint supplies the COMPLETE live-root set (a root
        // under-supply would free live nodes and corrupt the count or panic).
        // A multi-round two-counter net so GC fires on a growing set each round.
        let net = MddNet {
            bounds: vec![8, 8],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        };
        let normal = net.reachable_count().expect("ok").state_count;
        assert_eq!(normal, 81, "9*9 reachable markings");

        crate::node::set_gc_stress(true);
        let forced = net.reachable_count().expect("ok").state_count;
        crate::node::set_gc_stress(false);

        assert_eq!(
            forced, normal,
            "forcing gc every round must not change the reachable count"
        );
    }

    #[test]
    fn reachable_count_is_stable_under_forced_sift() {
        // End-to-end sift soundness: forcing a reorder at EVERY round (stress
        // hook) must not change the reachable count — validates the place↔level
        // permutation threading through singleton/enumerate/fire. If the
        // permutation were mis-applied, the fired successors would land on the
        // wrong variables and the count (or a range check) would break.
        // Includes ASYMMETRIC bounds so the reorder permutes non-trivially.
        let nets = [
            MddNet {
                bounds: vec![8, 8],
                initial_marking: vec![0, 0],
                transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
            },
            MddNet {
                bounds: vec![1, 1, 1],
                initial_marking: vec![1, 0, 0],
                transitions: vec![
                    t(vec![1, 0, 0], vec![0, 1, 0]),
                    t(vec![0, 1, 0], vec![0, 0, 1]),
                    t(vec![0, 0, 1], vec![1, 0, 0]),
                ],
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
            let normal = net.reachable_count().expect("ok").state_count;
            crate::sift_runtime::set_sift_stress(true);
            let forced = net.reachable_count().expect("ok").state_count;
            crate::sift_runtime::set_sift_stress(false);
            assert_eq!(
                forced, normal,
                "forcing sift every round changed the reachable count for bounds {:?}",
                net.bounds
            );
        }
    }

    #[test]
    fn conserved_token_ring_three_places() {
        // One token rotates p0->p1->p2->p0. Reachable: 3 states.
        let net = MddNet {
            bounds: vec![1, 1, 1],
            initial_marking: vec![1, 0, 0],
            transitions: vec![
                t(vec![1, 0, 0], vec![0, 1, 0]),
                t(vec![0, 1, 0], vec![0, 0, 1]),
                t(vec![0, 0, 1], vec![1, 0, 0]),
            ],
        };
        let r = net.reachable_count().expect("ok");
        assert_eq!(r.state_count, 3);
    }

    #[test]
    fn malformed_length_declined() {
        let net = MddNet {
            bounds: vec![1, 1],
            initial_marking: vec![0], // wrong length
            transitions: vec![],
        };
        assert!(matches!(
            net.reachable_count(),
            Err(CountError::Malformed(_))
        ));
    }

    #[test]
    fn initial_out_of_range_declined() {
        let net = MddNet {
            bounds: vec![1],
            initial_marking: vec![5],
            transitions: vec![],
        };
        assert!(matches!(
            net.reachable_count(),
            Err(CountError::Malformed(_))
        ));
    }

    #[test]
    fn zero_places_one_empty_marking() {
        // Degenerate but valid: no places ⇒ exactly one marking (the empty
        // one), no transitions can fire.
        let net = MddNet {
            bounds: vec![],
            initial_marking: vec![],
            transitions: vec![],
        };
        let r = net.reachable_count().expect("ok");
        assert_eq!(r.state_count, 1);
    }
}
