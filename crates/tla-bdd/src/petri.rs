// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Native symbolic StateSpace for **1-safe** P/T nets on [`crate::Bdd`] — the
//! native-BDD equivalent of the reachable-set construction `tla-dd` gets from
//! oxidd, end-to-end on real net specs (not hand-built relations).
//!
//! # Encoding
//!
//! One Boolean variable per place per time-frame: current place `p` is BDD var
//! `p`, its next-state copy is var `num_places + p` (a current block above a next
//! block, so the image quantifies the top block and renames the bottom one back
//! up). For a 1-safe net (`bound = 1`), the per-place fire relation
//! `next[p] = current[p] − pre[p] + post[p]` confined to `{0,1}` is:
//!
//! | `pre,post` | meaning | relation on `(cₚ, nₚ)` |
//! |---|---|---|
//! | `0,0` | frame (untouched) | `nₚ ↔ cₚ` |
//! | `1,0` | consume | `cₚ ∧ ¬nₚ` |
//! | `0,1` | produce | `¬cₚ ∧ nₚ` (disabled if `p` already full — 1-safe) |
//! | `1,1` | read / self-loop | `cₚ ∧ nₚ` |
//!
//! `T = ⋁ₜ ⋀ₚ rel(t,p)`; reachability is [`crate::Bdd::reachable`]; `|R|` is
//! `sat_count` over the current block. This is the FIRST increment (1-safe);
//! multi-bit place encodings (general bounds) layer on the same scaffold.

use crate::{Bdd, NodeId};

/// A 1-safe place/transition net (every place bound 1).
pub struct OneSafeNet {
    /// Number of places.
    pub num_places: usize,
    /// Initial marking (one bool per place).
    pub init: Vec<bool>,
    /// Transitions.
    pub transitions: Vec<OneSafeTransition>,
}

/// A transition's `pre`/`post` incidence (one bool per place; 1-safe).
pub struct OneSafeTransition {
    /// Input places (consumed).
    pub pre: Vec<bool>,
    /// Output places (produced).
    pub post: Vec<bool>,
}

/// Exact reachable-state count of a 1-safe net, computed symbolically on the
/// native ROBDD engine. `None` (fail-closed) if the exact count is not
/// representable — more than 127 place variables or `u128` overflow.
#[must_use]
pub fn reachable_count(net: &OneSafeNet) -> Option<u128> {
    let np = net.num_places;
    let cur = |p: usize| p as u32;
    let nxt = |p: usize| (np + p) as u32;
    let mut b = Bdd::new();

    // init: the marking's bit pattern over the current block.
    let mut init = Bdd::TRUE;
    for p in 0..np {
        let vp = b.var(cur(p));
        let lit = if net.init[p] { vp } else { b.not(vp) };
        init = b.and(init, lit);
    }

    // transition relation T = ⋁ₜ ⋀ₚ rel(t,p)
    let mut trans = Bdd::FALSE;
    for t in &net.transitions {
        let mut rel = Bdd::TRUE;
        for p in 0..np {
            let cp = b.var(cur(p));
            let n_p = b.var(nxt(p));
            let rel_p = match (t.pre[p], t.post[p]) {
                (false, false) => {
                    // frame: nₚ ↔ cₚ
                    let x = b.xor(cp, n_p);
                    b.not(x)
                }
                (true, false) => {
                    // consume: cₚ ∧ ¬nₚ
                    let nn = b.not(n_p);
                    b.and(cp, nn)
                }
                (false, true) => {
                    // produce (disabled if already full): ¬cₚ ∧ nₚ
                    let nc = b.not(cp);
                    b.and(nc, n_p)
                }
                (true, true) => {
                    // read / self-loop: cₚ ∧ nₚ
                    b.and(cp, n_p)
                }
            };
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }

    let current: Vec<u32> = (0..np).map(cur).collect();
    let next: Vec<u32> = (0..np).map(nxt).collect();
    let r = b.reachable(init, trans, &current, &next);
    b.sat_count(r, np as u32)
}

/// A general bounded place/transition net (place `p` ranges `0..=bounds[p]`).
#[derive(Debug, Clone)]
pub struct BoundedNet {
    /// Per-place token upper bound.
    pub bounds: Vec<u64>,
    /// Initial marking (`<= bounds[p]`).
    pub init: Vec<u64>,
    /// Transitions (`pre`/`post` arc weights, one per place).
    pub transitions: Vec<BoundedTransition>,
}

/// A transition's `pre`/`post` arc weights (one per place).
#[derive(Debug, Clone)]
pub struct BoundedTransition {
    /// Tokens consumed per place.
    pub pre: Vec<u64>,
    /// Tokens produced per place.
    pub post: Vec<u64>,
}

/// Bits needed to represent `0..=bound`.
fn bits_for(bound: u64) -> usize {
    (64 - bound.leading_zeros()).max(1) as usize
}

/// Reachable-state count of a general bounded net on the plain [`crate::Bdd`],
/// COUNT ONLY (no edges/max-token metrics) — the apples-to-apples partner of
/// [`reachable_count_bounded_cedge`] for fair engine benchmarking. `None`
/// (fail-closed) if the exact count is not representable in `u128`.
#[must_use]
pub fn reachable_count_only_bdd(net: &BoundedNet) -> Option<u128> {
    let np = net.bounds.len();
    let widths: Vec<usize> = net.bounds.iter().map(|&b| bits_for(b)).collect();
    let mut offset = vec![0usize; np];
    let mut acc = 0usize;
    for p in 0..np {
        offset[p] = acc;
        acc += widths[p];
    }
    let total = acc;
    let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
    let nxt_bit = |p: usize, i: usize| (total + offset[p] + i) as u32;
    let mut b = Bdd::new();
    let value = |b: &mut Bdd, p: usize, v: u64, in_next: bool| -> NodeId {
        let mut a = Bdd::TRUE;
        for i in 0..widths[p] {
            let var = if in_next {
                nxt_bit(p, i)
            } else {
                cur_bit(p, i)
            };
            let vb = b.var(var);
            let lit = if (v >> i) & 1 == 1 { vb } else { b.not(vb) };
            a = b.and(a, lit);
        }
        a
    };
    let mut init = Bdd::TRUE;
    for p in 0..np {
        let m = value(&mut b, p, net.init[p], false);
        init = b.and(init, m);
    }
    let mut trans = Bdd::FALSE;
    for t in &net.transitions {
        let mut rel = Bdd::TRUE;
        for p in 0..np {
            let mut rel_p = Bdd::FALSE;
            for v in 0..=net.bounds[p] {
                if v < t.pre[p] || v - t.pre[p] + t.post[p] > net.bounds[p] {
                    continue;
                }
                let cv = value(&mut b, p, v, false);
                let nvb = value(&mut b, p, v - t.pre[p] + t.post[p], true);
                let pair = b.and(cv, nvb);
                rel_p = b.or(rel_p, pair);
            }
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }
    let current: Vec<u32> = (0..total as u32).collect();
    let next: Vec<u32> = (total as u32..2 * total as u32).collect();
    let r = b.reachable(init, trans, &current, &next);
    b.sat_count(r, total as u32)
}

/// UpperBounds on the native ROBDD engine: for each query (per-place integer
/// coefficients), the maximum of `Σ_p coeffs[p]·m[p]` over the reachable set —
/// the tla-bdd twin of `tla_mdd::MddNet::upper_bounds`. Builds `R` once, then
/// `max_weighted` per query with bit-level weights `coeffs[p]·2^i`. Returns one
/// bound per query (in order). The reachability/StateSpace lane's UpperBounds
/// sibling, for routing the UB lane off oxidd.
#[must_use]
pub fn upper_bounds_bounded(net: &BoundedNet, queries: &[Vec<i128>]) -> Vec<i128> {
    upper_bounds_bounded_within(net, queries, None).expect("reachable_within(None) never declines")
}

/// Deadline-aware [`upper_bounds_bounded`]: `None` (fail-closed decline) if the
/// reachable-set fixpoint exceeds `deadline` — the budget contract for safe
/// production wiring of the UpperBounds lane (mirrors `evaluate_reachability_within`).
///
/// [`crate::catch_abort`] wraps the WHOLE body (audit 2026-07-02 follow-up):
/// the armed node/deadline abort can panic with `BddAbort` in the query phase
/// too, not just inside `reachable_within`'s own catch — so folding it here
/// keeps the decline self-contained for every caller, isolated or not.
#[must_use]
pub fn upper_bounds_bounded_within(
    net: &BoundedNet,
    queries: &[Vec<i128>],
    deadline: Option<std::time::Instant>,
) -> Option<Vec<i128>> {
    crate::catch_abort(|| upper_bounds_bounded_within_inner(net, queries, deadline))
}

fn upper_bounds_bounded_within_inner(
    net: &BoundedNet,
    queries: &[Vec<i128>],
    deadline: Option<std::time::Instant>,
) -> Option<Vec<i128>> {
    let np = net.bounds.len();
    let widths: Vec<usize> = net.bounds.iter().map(|&b| bits_for(b)).collect();
    let mut offset = vec![0usize; np];
    let mut acc = 0usize;
    for p in 0..np {
        offset[p] = acc;
        acc += widths[p];
    }
    let total = acc;
    let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
    let nxt_bit = |p: usize, i: usize| (total + offset[p] + i) as u32;
    let mut b = Bdd::new();
    // Cooperative abort (audit 2026-07-02): bounds construction, the
    // reachable fixpoint, AND the post-reachable query/sat phase. Any
    // BddAbort (store left canonical — the panic precedes mutation) unwinds
    // to the `_inner` wrapper's `catch_abort`, which folds it into `None`.
    b.set_abort_limits(Some(crate::default_abort_node_budget()), deadline);
    let value = |b: &mut Bdd, p: usize, v: u64, in_next: bool| -> NodeId {
        let mut a = Bdd::TRUE;
        for i in 0..widths[p] {
            let var = if in_next {
                nxt_bit(p, i)
            } else {
                cur_bit(p, i)
            };
            let vb = b.var(var);
            let lit = if (v >> i) & 1 == 1 { vb } else { b.not(vb) };
            a = b.and(a, lit);
        }
        a
    };
    let mut init = Bdd::TRUE;
    for p in 0..np {
        let m = value(&mut b, p, net.init[p], false);
        init = b.and(init, m);
    }
    let mut trans = Bdd::FALSE;
    for t in &net.transitions {
        let mut rel = Bdd::TRUE;
        for p in 0..np {
            let mut rel_p = Bdd::FALSE;
            for v in 0..=net.bounds[p] {
                if v < t.pre[p] || v - t.pre[p] + t.post[p] > net.bounds[p] {
                    continue;
                }
                let cv = value(&mut b, p, v, false);
                let nvb = value(&mut b, p, v - t.pre[p] + t.post[p], true);
                let pair = b.and(cv, nvb);
                rel_p = b.or(rel_p, pair);
            }
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }
    let current: Vec<u32> = (0..total as u32).collect();
    let next: Vec<u32> = (total as u32..2 * total as u32).collect();
    let r = b.reachable_within(init, trans, &current, &next, deadline)?;
    Some(
        queries
            .iter()
            .map(|coeffs| {
                // bit-level weights: place p, bit i carries coeffs[p]·2^i.
                let mut weights = vec![0i128; total];
                for (p, &c) in coeffs.iter().enumerate().take(np) {
                    for i in 0..widths[p] {
                        weights[offset[p] + i] = c * (1i128 << i);
                    }
                }
                b.max_weighted(r, &weights, total as u32)
            })
            .collect(),
    )
}

/// Reachable-state count of a general bounded net computed on the
/// COMPLEMENTED-EDGE core ([`crate::cedge::CBdd`]) — the same bit-blasted
/// encoding as [`reachable_count_bounded`] but on the ~2×-denser engine. Used to
/// validate the complement core end-to-end on real specs (it must match the
/// plain-`Bdd` count) on the way to migrating production reachability onto it.
/// `None` (fail-closed) if the exact count is not representable in `u128`.
#[must_use]
pub fn reachable_count_bounded_cedge(net: &BoundedNet) -> Option<u128> {
    use crate::cedge::CBdd;
    let np = net.bounds.len();
    let widths: Vec<usize> = net.bounds.iter().map(|&b| bits_for(b)).collect();
    let mut offset = vec![0usize; np];
    let mut acc = 0usize;
    for p in 0..np {
        offset[p] = acc;
        acc += widths[p];
    }
    let total = acc;
    let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
    let nxt_bit = |p: usize, i: usize| (total + offset[p] + i) as u32;
    let mut b = CBdd::new();
    let value = |b: &mut CBdd, p: usize, v: u64, in_next: bool| -> u32 {
        let mut a = CBdd::ONE;
        for i in 0..widths[p] {
            let var = if in_next {
                nxt_bit(p, i)
            } else {
                cur_bit(p, i)
            };
            let vb = b.var(var);
            let lit = if (v >> i) & 1 == 1 { vb } else { CBdd::not(vb) };
            a = b.and(a, lit);
        }
        a
    };
    let mut init = CBdd::ONE;
    for p in 0..np {
        let m = value(&mut b, p, net.init[p], false);
        init = b.and(init, m);
    }
    let mut trans = CBdd::ZERO;
    for t in &net.transitions {
        let mut rel = CBdd::ONE;
        for p in 0..np {
            let mut rel_p = CBdd::ZERO;
            for v in 0..=net.bounds[p] {
                if v < t.pre[p] || v - t.pre[p] + t.post[p] > net.bounds[p] {
                    continue;
                }
                let cv = value(&mut b, p, v, false);
                let nvb = value(&mut b, p, v - t.pre[p] + t.post[p], true);
                let pair = b.and(cv, nvb);
                rel_p = b.or(rel_p, pair);
            }
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }
    let current: Vec<u32> = (0..total as u32).collect();
    let next: Vec<u32> = (total as u32..2 * total as u32).collect();
    let r = b.reachable(init, trans, &current, &next);
    b.sat_count(r, total as u32)
}

/// The four MCC `StateSpace` metrics computed natively on the ROBDD engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateSpaceMetrics {
    /// Number of reachable markings `|R|`.
    pub states: u128,
    /// Number of reachability-graph arcs (enabled firings, incl. to seen states).
    pub edges: u128,
    /// `max_{m ∈ R, p} m[p]`.
    pub max_token_in_place: u64,
    /// `max_{m ∈ R} Σ_p m[p]`.
    pub max_token_sum: u64,
}

/// Reachable-state count of a general bounded net (the `states` metric).
/// `None` (fail-closed) if the exact count is not representable in `u128`.
#[must_use]
pub fn reachable_count_bounded_checked(net: &BoundedNet) -> Option<u128> {
    state_space_metrics_bounded_core(net).0
}

/// LEGACY saturating twin of [`reachable_count_bounded_checked`], kept only so
/// existing consumers (tla-petri's `mdd_common::reachable_count_via_bdd`)
/// compile unchanged: an inexact count comes back as `u128::MAX` instead of
/// `None`. New code should call the checked variant.
#[must_use]
pub fn reachable_count_bounded(net: &BoundedNet) -> u128 {
    reachable_count_bounded_checked(net).unwrap_or(u128::MAX)
}

/// All four `StateSpace` metrics of a general bounded net, computed symbolically
/// on the native ROBDD engine via a multi-bit (binary) place encoding — the full
/// StateSpace output `tla-dd` produces via oxidd. `None` (fail-closed) if either
/// count (`states` or `edges`) is inexact — >127 state bits or `u128` overflow.
#[must_use]
pub fn state_space_metrics_bounded_checked(net: &BoundedNet) -> Option<StateSpaceMetrics> {
    let (states, edges, max_token_in_place, max_token_sum) = state_space_metrics_bounded_core(net);
    Some(StateSpaceMetrics {
        states: states?,
        edges: edges?,
        max_token_in_place,
        max_token_sum,
    })
}

/// LEGACY saturating twin of [`state_space_metrics_bounded_checked`], kept only
/// so existing consumers (tla-petri's `mdd_common::state_space_metrics_via_bdd`)
/// compile unchanged: an inexact `states`/`edges` count comes back as
/// `u128::MAX` instead of `None` (the token maxima are always exact). New code
/// should call the checked variant.
#[must_use]
pub fn state_space_metrics_bounded(net: &BoundedNet) -> StateSpaceMetrics {
    let (states, edges, max_token_in_place, max_token_sum) = state_space_metrics_bounded_core(net);
    StateSpaceMetrics {
        states: states.unwrap_or(u128::MAX),
        edges: edges.unwrap_or(u128::MAX),
        max_token_in_place,
        max_token_sum,
    }
}

/// Shared implementation: `(states, edges, max_token_in_place, max_token_sum)`
/// with `None` for a count that is not exactly representable in `u128`.
fn state_space_metrics_bounded_core(net: &BoundedNet) -> (Option<u128>, Option<u128>, u64, u64) {
    let np = net.bounds.len();
    let widths: Vec<usize> = net.bounds.iter().map(|&b| bits_for(b)).collect();
    // Current-block bit offsets; the next block follows the whole current block.
    let mut offset = vec![0usize; np];
    let mut acc = 0usize;
    for p in 0..np {
        offset[p] = acc;
        acc += widths[p];
    }
    let total_cur_bits = acc;
    let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
    let nxt_bit = |p: usize, i: usize| (total_cur_bits + offset[p] + i) as u32;

    let mut b = Bdd::new();
    // value(p, v, in_next): BDD asserting place p's bits equal v in the chosen frame.
    let value = |b: &mut Bdd, p: usize, v: u64, in_next: bool| -> NodeId {
        let mut acc = Bdd::TRUE;
        for i in 0..widths[p] {
            let var = if in_next {
                nxt_bit(p, i)
            } else {
                cur_bit(p, i)
            };
            let vb = b.var(var);
            let lit = if (v >> i) & 1 == 1 { vb } else { b.not(vb) };
            acc = b.and(acc, lit);
        }
        acc
    };

    let mut init = Bdd::TRUE;
    for p in 0..np {
        let m = value(&mut b, p, net.init[p], false);
        init = b.and(init, m);
    }

    let mut trans = Bdd::FALSE;
    for t in &net.transitions {
        let mut rel = Bdd::TRUE;
        for p in 0..np {
            let pre = t.pre[p];
            let post = t.post[p];
            // rel_p = ⋁_{valid v} (cur[p]==v ∧ nxt[p]==v−pre+post)
            // valid v: v >= pre  and  0 <= v−pre+post <= bound (bound checked).
            let mut rel_p = Bdd::FALSE;
            for v in 0..=net.bounds[p] {
                if v < pre {
                    continue;
                }
                let nv = v - pre + post;
                if nv > net.bounds[p] {
                    continue; // firing would exceed the bound ⇒ disabled here
                }
                let cv = value(&mut b, p, v, false);
                let nvb = value(&mut b, p, nv, true);
                let pair = b.and(cv, nvb);
                rel_p = b.or(rel_p, pair);
            }
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }

    let current: Vec<u32> = (0..np)
        .flat_map(|p| (0..widths[p]).map(move |i| (p, i)))
        .map(|(p, i)| cur_bit(p, i))
        .collect();
    let next: Vec<u32> = (0..np)
        .flat_map(|p| (0..widths[p]).map(move |i| (p, i)))
        .map(|(p, i)| nxt_bit(p, i))
        .collect();
    let r = b.reachable(init, trans, &current, &next);

    // (1) states = |R| over the current block.
    let states = b.sat_count(r, total_cur_bits as u32);

    // (2) max_token_in_place: largest v any place attains over R (enumerate v
    // high→low per place; the first non-empty `R ∧ (p==v)` is that place's max).
    let mut max_token_in_place = 0u64;
    for p in 0..np {
        for v in (0..=net.bounds[p]).rev() {
            let pv = value(&mut b, p, v, false);
            let inter = b.and(r, pv);
            if inter != Bdd::FALSE {
                max_token_in_place = max_token_in_place.max(v);
                break;
            }
        }
    }

    // (3) max_token_sum = max_{m∈R} Σ_p m[p]. Bit b_i of place p contributes 2^i
    // to its value, so the weighted bit-max over R is the token-sum max.
    let mut weights = vec![0i128; total_cur_bits];
    for p in 0..np {
        for i in 0..widths[p] {
            weights[offset[p] + i] = 1i128 << i;
        }
    }
    let max_token_sum = b.max_weighted(r, &weights, total_cur_bits as u32) as u64;

    // (4) edges = Σ_t |{m ∈ R : t enabled (fires to a valid successor)}|. The
    // enabling set of t is the source projection of its relation, rebuilt
    // directly: ⋀_p ⋁_{valid v} (p == v). `None` on any inexact per-transition
    // count or on `u128` overflow of the sum (fail-closed, no saturation).
    let mut edges: Option<u128> = Some(0);
    for t in &net.transitions {
        let mut src = Bdd::TRUE;
        for p in 0..np {
            let mut src_p = Bdd::FALSE;
            for v in 0..=net.bounds[p] {
                if v < t.pre[p] {
                    continue;
                }
                if v - t.pre[p] + t.post[p] > net.bounds[p] {
                    continue;
                }
                let cv = value(&mut b, p, v, false);
                src_p = b.or(src_p, cv);
            }
            src = b.and(src, src_p);
        }
        let enabled = b.and(r, src);
        edges = match (edges, b.sat_count(enabled, total_cur_bits as u32)) {
            (Some(acc), Some(c)) => acc.checked_add(c),
            _ => None,
        };
    }

    (states, edges, max_token_in_place, max_token_sum)
}

/// A reachability atom/formula over a bounded net (the fragment the
/// ReachabilityFireability examination needs — fireability + Boolean structure).
#[derive(Clone)]
pub enum Pred {
    /// Transition `t` is enabled (every input place has `>= pre` tokens).
    Fireable(usize),
    /// Token cardinality `Σ_p coeffs[p]·m[p] ≤ k` (the cardinality atom).
    TokenLe {
        /// Per-place coefficient.
        coeffs: Vec<i128>,
        /// Right-hand-side bound.
        k: i128,
    },
    /// Conjunction.
    And(Vec<Pred>),
    /// Disjunction.
    Or(Vec<Pred>),
    /// Negation.
    Not(Box<Pred>),
}

/// `EF φ` (some reachable marking satisfies φ) or `AG φ` (all do).
pub enum Query {
    /// Exists-finally.
    Ef(Pred),
    /// Always-globally.
    Ag(Pred),
}

/// The native-BDD reachability **soundness certificate**, mirroring the MDD
/// lane's `verify_saturation_inductive_fixpoint`. Builds the reachable set `R`
/// and *independently verifies* it is an inductive invariant containing the
/// initial marking:
///   `init ⊆ R`  ∧  `post_image(R) ⊆ R`.
/// By induction those two facts entail `R ⊇ reachable` (no reachable marking is
/// missed) — a machine-checkable proof of soundness, not just a differential
/// test. Returns `true` iff both hold (it always should for the computed `R`;
/// the check is proof-carrying defense-in-depth that catches any engine bug).
#[must_use]
pub fn reachable_is_sound_inductive_invariant(net: &BoundedNet) -> bool {
    let np = net.bounds.len();
    let widths: Vec<usize> = net.bounds.iter().map(|&b| bits_for(b)).collect();
    let mut offset = vec![0usize; np];
    let mut acc = 0usize;
    for p in 0..np {
        offset[p] = acc;
        acc += widths[p];
    }
    let total = acc;
    let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
    let nxt_bit = |p: usize, i: usize| (total + offset[p] + i) as u32;
    let mut b = Bdd::new();
    let value = |b: &mut Bdd, p: usize, v: u64, in_next: bool| -> NodeId {
        let mut a = Bdd::TRUE;
        for i in 0..widths[p] {
            let var = if in_next {
                nxt_bit(p, i)
            } else {
                cur_bit(p, i)
            };
            let vb = b.var(var);
            let lit = if (v >> i) & 1 == 1 { vb } else { b.not(vb) };
            a = b.and(a, lit);
        }
        a
    };
    let mut init = Bdd::TRUE;
    for p in 0..np {
        let m = value(&mut b, p, net.init[p], false);
        init = b.and(init, m);
    }
    let mut trans = Bdd::FALSE;
    for t in &net.transitions {
        let mut rel = Bdd::TRUE;
        for p in 0..np {
            let mut rel_p = Bdd::FALSE;
            for v in 0..=net.bounds[p] {
                if v < t.pre[p] || v - t.pre[p] + t.post[p] > net.bounds[p] {
                    continue;
                }
                let cv = value(&mut b, p, v, false);
                let nvb = value(&mut b, p, v - t.pre[p] + t.post[p], true);
                let pair = b.and(cv, nvb);
                rel_p = b.or(rel_p, pair);
            }
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }
    let current: Vec<u32> = (0..total as u32).collect();
    let next: Vec<u32> = (total as u32..2 * total as u32).collect();
    let n2c: std::collections::HashMap<u32, u32> =
        next.iter().copied().zip(current.iter().copied()).collect();
    let r = b.reachable(init, trans, &current, &next);
    // (1) init ⊆ R
    if !b.subset(init, r) {
        return false;
    }
    // (2) post_image(R) ⊆ R  (inductive closure)
    let img = b.post_image(trans, r, &current, &n2c);
    b.subset(img, r)
}

/// The characteristic-set BDD of a state predicate over the current-frame bits.
/// `⟦Fireable t⟧ = ⋀_p ⋁_{v≥pre[p]} (p==v)`; `⟦TokenLe⟧` is the threshold BDD
/// `linear_le`; Boolean structure composes. `value(b,p,v,in_next)` asserts place
/// `p == v`; `offset`/`widths` are the bit layout.
#[allow(clippy::too_many_arguments)]
fn charset(
    b: &mut Bdd,
    net: &BoundedNet,
    p: &Pred,
    value: &dyn Fn(&mut Bdd, usize, u64, bool) -> NodeId,
    offset: &[usize],
    widths: &[usize],
) -> NodeId {
    match p {
        Pred::Fireable(t) => {
            let tr = &net.transitions[*t];
            let np = net.bounds.len();
            let mut g = Bdd::TRUE;
            for pl in 0..np {
                let mut gp = Bdd::FALSE;
                for v in tr.pre[pl]..=net.bounds[pl] {
                    let cv = value(b, pl, v, false);
                    gp = b.or(gp, cv);
                }
                g = b.and(g, gp);
            }
            g
        }
        Pred::TokenLe { coeffs, k } => {
            let mut terms: Vec<(u32, i128)> = Vec::new();
            for (pl, &c) in coeffs.iter().enumerate() {
                for i in 0..widths[pl] {
                    terms.push(((offset[pl] + i) as u32, c * (1i128 << i)));
                }
            }
            terms.sort_by_key(|&(v, _)| v);
            b.linear_le(&terms, *k)
        }
        Pred::And(cs) => {
            let mut a = Bdd::TRUE;
            for c in cs {
                let cc = charset(b, net, c, value, offset, widths);
                a = b.and(a, cc);
            }
            a
        }
        Pred::Or(cs) => {
            let mut o = Bdd::FALSE;
            for c in cs {
                let cc = charset(b, net, c, value, offset, widths);
                o = b.or(o, cc);
            }
            o
        }
        Pred::Not(c) => {
            let cc = charset(b, net, c, value, offset, widths);
            b.not(cc)
        }
    }
}

/// Evaluate fireability reachability queries over a bounded net's reachable set,
/// natively on the ROBDD engine — the native-BDD equivalent of `tla-dd`'s
/// ReachabilityFireability lane. `EF φ ⟺ R ∧ ⟦φ⟧ ≠ ∅`; `AG φ ⟺ R ∧ ¬⟦φ⟧ = ∅`.
#[must_use]
pub fn evaluate_reachability(net: &BoundedNet, queries: &[Query]) -> Vec<bool> {
    evaluate_reachability_within(net, queries, None).expect("reachable_within(None) never declines")
}

/// Deadline-aware [`evaluate_reachability`]: returns `None` (fail-closed decline)
/// if the reachable-set fixpoint exceeds `deadline` — the budget contract a
/// production examination lane needs to wire this safely (mirrors the oxidd lanes'
/// `set_thread_deadline`). `None` deadline ⇒ run to convergence.
///
/// [`crate::catch_abort`] wraps the WHOLE body (audit 2026-07-02 follow-up):
/// a `BddAbort` can panic in the query phase after the internally-caught
/// `reachable_within`, so folding it here keeps the decline self-contained.
#[must_use]
pub fn evaluate_reachability_within(
    net: &BoundedNet,
    queries: &[Query],
    deadline: Option<std::time::Instant>,
) -> Option<Vec<bool>> {
    crate::catch_abort(|| evaluate_reachability_within_inner(net, queries, deadline))
}

fn evaluate_reachability_within_inner(
    net: &BoundedNet,
    queries: &[Query],
    deadline: Option<std::time::Instant>,
) -> Option<Vec<bool>> {
    let np = net.bounds.len();
    let widths: Vec<usize> = net.bounds.iter().map(|&b| bits_for(b)).collect();
    let mut offset = vec![0usize; np];
    let mut acc = 0usize;
    for p in 0..np {
        offset[p] = acc;
        acc += widths[p];
    }
    let total_cur_bits = acc;
    let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
    let nxt_bit = |p: usize, i: usize| (total_cur_bits + offset[p] + i) as u32;
    let mut b = Bdd::new();
    // Cooperative abort (audit 2026-07-02): bounds construction, the
    // reachable fixpoint, AND the post-reachable query/sat phase. Any
    // BddAbort (store left canonical — the panic precedes mutation) unwinds
    // to the `_inner` wrapper's `catch_abort`, which folds it into `None`.
    b.set_abort_limits(Some(crate::default_abort_node_budget()), deadline);
    let value = |b: &mut Bdd, p: usize, v: u64, in_next: bool| -> NodeId {
        let mut a = Bdd::TRUE;
        for i in 0..widths[p] {
            let var = if in_next {
                nxt_bit(p, i)
            } else {
                cur_bit(p, i)
            };
            let vb = b.var(var);
            let lit = if (v >> i) & 1 == 1 { vb } else { b.not(vb) };
            a = b.and(a, lit);
        }
        a
    };
    // init + transition relation (same encoding as the metrics path).
    let mut init = Bdd::TRUE;
    for p in 0..np {
        let m = value(&mut b, p, net.init[p], false);
        init = b.and(init, m);
    }
    let mut trans = Bdd::FALSE;
    for t in &net.transitions {
        let mut rel = Bdd::TRUE;
        for p in 0..np {
            let mut rel_p = Bdd::FALSE;
            for v in 0..=net.bounds[p] {
                if v < t.pre[p] || v - t.pre[p] + t.post[p] > net.bounds[p] {
                    continue;
                }
                let cv = value(&mut b, p, v, false);
                let nvb = value(&mut b, p, v - t.pre[p] + t.post[p], true);
                let pair = b.and(cv, nvb);
                rel_p = b.or(rel_p, pair);
            }
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }
    let current: Vec<u32> = (0..np)
        .flat_map(|p| (0..widths[p]).map(move |i| cur_bit(p, i)))
        .collect();
    let next: Vec<u32> = (0..np)
        .flat_map(|p| (0..widths[p]).map(move |i| nxt_bit(p, i)))
        .collect();
    let r = b.reachable_within(init, trans, &current, &next, deadline)?;

    Some(
        queries
            .iter()
            .map(|q| match q {
                Query::Ef(p) => {
                    let cs = charset(&mut b, net, p, &value, &offset, &widths);
                    let inter = b.and(r, cs);
                    inter != Bdd::FALSE // ∃ reachable marking ⊨ φ
                }
                Query::Ag(p) => {
                    let cs = charset(&mut b, net, p, &value, &offset, &widths);
                    let nc = b.not(cs);
                    let viol = b.and(r, nc);
                    viol == Bdd::FALSE // no reachable marking ⊭ φ
                }
            })
            .collect(),
    )
}

/// Does the reachable graph of `net` contain a cycle through a marking
/// satisfying `accepting`? This is the **Büchi-emptiness / fair-cycle** core of
/// LTL model checking (a reachable accepting cycle ⟺ the Büchi product language
/// is non-empty ⟺ the property is violated), computed natively on the ROBDD
/// engine by the one-acceptance-set Emerson-Lei fixpoint
/// `νZ. Z ∩ pre⁺_Z(Z ∩ F)` over the reachable set — `pre⁺` is the transitive
/// backward image restricted to `Z`. Returns `true` iff such a cycle exists.
///
/// This is the symbolic engine LTL needs; the remaining LTL piece is the
/// formula→Büchi translation feeding `accepting` (and the product), layered on
/// top.
#[must_use]
pub fn fair_cycle_exists(net: &BoundedNet, accepting: &Pred, within: Option<&Pred>) -> bool {
    fair_cycle_exists_generalized(net, std::slice::from_ref(accepting), within)
}

/// Generalized (multi-acceptance) symbolic fair-cycle: is there a reachable cycle
/// (within `within`, else within R) that visits EVERY one of the `accepting`
/// sets infinitely often? This is the GENERALIZED Büchi emptiness test — the
/// symbolic emptiness core for an LTL `¬φ` GBA product (k acceptance sets ⇒ a
/// fair lasso must touch all k). Reduces to [`fair_cycle_exists`] for a single
/// set; empty `accepting` ⇒ any reachable cycle qualifies. Generalized
/// Emerson-Lei: `Z := Z ∩ ⋂_i pre⁺_Z(Z ∩ F_i)` iterated to the gfp.
pub fn fair_cycle_exists_generalized(
    net: &BoundedNet,
    accepting: &[Pred],
    within: Option<&Pred>,
) -> bool {
    let np = net.bounds.len();
    let widths: Vec<usize> = net.bounds.iter().map(|&b| bits_for(b)).collect();
    let mut offset = vec![0usize; np];
    let mut acc = 0usize;
    for p in 0..np {
        offset[p] = acc;
        acc += widths[p];
    }
    let total = acc;
    let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
    let nxt_bit = |p: usize, i: usize| (total + offset[p] + i) as u32;
    let mut b = Bdd::new();
    let value = |b: &mut Bdd, p: usize, v: u64, in_next: bool| -> NodeId {
        let mut a = Bdd::TRUE;
        for i in 0..widths[p] {
            let var = if in_next {
                nxt_bit(p, i)
            } else {
                cur_bit(p, i)
            };
            let vb = b.var(var);
            let lit = if (v >> i) & 1 == 1 { vb } else { b.not(vb) };
            a = b.and(a, lit);
        }
        a
    };
    let mut init = Bdd::TRUE;
    for p in 0..np {
        let m = value(&mut b, p, net.init[p], false);
        init = b.and(init, m);
    }
    let mut trans = Bdd::FALSE;
    for t in &net.transitions {
        let mut rel = Bdd::TRUE;
        for p in 0..np {
            let mut rel_p = Bdd::FALSE;
            for v in 0..=net.bounds[p] {
                if v < t.pre[p] || v - t.pre[p] + t.post[p] > net.bounds[p] {
                    continue;
                }
                let cv = value(&mut b, p, v, false);
                let nvb = value(&mut b, p, v - t.pre[p] + t.post[p], true);
                let pair = b.and(cv, nvb);
                rel_p = b.or(rel_p, pair);
            }
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }
    let current: Vec<u32> = (0..total as u32).collect();
    let next: Vec<u32> = (total as u32..2 * total as u32).collect();
    let r = b.reachable(init, trans, &current, &next);
    let c2n: std::collections::HashMap<u32, u32> =
        (0..total as u32).map(|v| (v, total as u32 + v)).collect();
    // Domain bounds the search: default = R (cycle THROUGH accepting, the FG
    // pattern); `within` restricts to a subgraph (cycle ENTIRELY WITHIN it, the
    // GF pattern with `within = accepting = ¬φ`).
    let dom = match within {
        None => r,
        Some(w) => {
            let cs = charset(&mut b, net, w, &value, &offset, &widths);
            b.and(r, cs)
        }
    };
    // Each acceptance set, confined to R ∩ dom. Empty ⇒ {dom} (any cycle).
    let acc_sets: Vec<NodeId> = if accepting.is_empty() {
        vec![dom]
    } else {
        accepting
            .iter()
            .map(|p| {
                let cs = charset(&mut b, net, p, &value, &offset, &widths);
                let ar = b.and(r, cs);
                b.and(ar, dom)
            })
            .collect()
    };
    generalized_el_core(&mut b, trans, dom, &acc_sets, &next, &c2n) != Bdd::FALSE
}

/// The generalized Emerson-Lei fair-cycle gfp on an ARBITRARY transition relation:
/// the largest `Z ⊆ dom` such that every `Z`-state can reach (within `Z`, ≥1 step)
/// some state of EACH acceptance set — i.e. lies on a cycle visiting all `acc_sets`
/// i.o. `Z := Z ∩ ⋂_i pre⁺_Z(Z ∩ F_i)` to the gfp. LAYOUT-AGNOSTIC: it takes the
/// transition relation, the next-variable list, and the cur→next rename map
/// directly, so it serves BOTH the bounded-net fair-cycle AND (reused unchanged)
/// the forthcoming LTL GBA×net symbolic product, whose bit layout differs.
/// Returns the (possibly empty) fair set `Z`; emptiness ⇔ no accepting run.
fn generalized_el_core(
    b: &mut Bdd,
    trans: NodeId,
    dom: NodeId,
    acc_sets: &[NodeId],
    next: &[u32],
    c2n: &std::collections::HashMap<u32, u32>,
) -> NodeId {
    // pre⁺ within `z` reaching `seed`: μW. (EX(seed ∪ W)) ∩ z.
    let pre_plus_in = |b: &mut Bdd, z: NodeId, seed: NodeId| -> NodeId {
        let mut w = Bdd::FALSE;
        loop {
            let target = b.or(seed, w);
            let tn = b.rename(target, c2n);
            let ex = b.and_exists(trans, tn, next); // EX target
            let nw = b.and(ex, z);
            if nw == w {
                return w;
            }
            w = nw;
        }
    };
    let mut z = dom;
    loop {
        let mut z2 = z;
        for &f in acc_sets {
            let f_in_z = b.and(z, f);
            let reach_back = pre_plus_in(b, z, f_in_z);
            z2 = b.and(z2, reach_back);
        }
        if z2 == z {
            break;
        }
        z = z2;
    }
    z
}

/// A CTL formula over a bounded net (the E-fragment that underpins both the CTL
/// examination and — via `EG` over a Büchi product — the LTL emptiness check).
pub enum Ctl {
    /// A state predicate (fireability / cardinality / Boolean over atoms).
    Atom(Pred),
    /// `EX φ` — some successor satisfies φ.
    Ex(Box<Ctl>),
    /// `EF φ` — some path eventually satisfies φ.
    Ef(Box<Ctl>),
    /// `EG φ` — some path globally satisfies φ.
    Eg(Box<Ctl>),
    /// `E[φ U ψ]` — some path where φ holds until ψ.
    Eu(Box<Ctl>, Box<Ctl>),
    /// Conjunction.
    And(Box<Ctl>, Box<Ctl>),
    /// Disjunction.
    Or(Box<Ctl>, Box<Ctl>),
    /// Negation.
    Not(Box<Ctl>),
}

/// Does the INITIAL marking of `net` satisfy the CTL formula `f`? Evaluated
/// natively on the ROBDD engine via pre-image fixpoints (`EX = R ∩ pre(·)`,
/// `EF = μY. φ ∨ EX Y`, `EG = νY. φ ∧ EX Y`), confined to the reachable set —
/// the native-BDD equivalent of `tla-dd`'s symbolic CTL lane.
#[must_use]
pub fn evaluate_ctl(net: &BoundedNet, f: &Ctl) -> bool {
    evaluate_ctl_within(net, f, None).expect("reachable_within(None) never declines")
}

/// Deadline-aware [`evaluate_ctl`]: `None` (fail-closed decline) if the
/// reachable-set fixpoint exceeds `deadline` — the budget contract for safe
/// production wiring of the CTL lane. The sat fixpoints run over the (finite)
/// reachable set, so bounding the R build bounds the whole evaluation.
///
/// [`crate::catch_abort`] wraps the WHOLE body (audit 2026-07-02 follow-up):
/// the CTL sat/pre-image phase runs AFTER the internally-caught
/// `reachable_within` with the abort still armed, so a budget/deadline hit
/// there would otherwise panic with `BddAbort` uncaught into the (possibly
/// un-isolated, default-on) production CTL lane. Folding it here makes the
/// decline self-contained for every caller.
#[must_use]
pub fn evaluate_ctl_within(
    net: &BoundedNet,
    f: &Ctl,
    deadline: Option<std::time::Instant>,
) -> Option<bool> {
    crate::catch_abort(|| evaluate_ctl_within_inner(net, f, deadline))
}

fn evaluate_ctl_within_inner(
    net: &BoundedNet,
    f: &Ctl,
    deadline: Option<std::time::Instant>,
) -> Option<bool> {
    let np = net.bounds.len();
    let widths: Vec<usize> = net.bounds.iter().map(|&b| bits_for(b)).collect();
    let mut offset = vec![0usize; np];
    let mut acc = 0usize;
    for p in 0..np {
        offset[p] = acc;
        acc += widths[p];
    }
    let total = acc;
    let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
    let nxt_bit = |p: usize, i: usize| (total + offset[p] + i) as u32;
    let mut b = Bdd::new();
    // Cooperative abort (audit 2026-07-02): bounds construction, the
    // reachable fixpoint, AND the post-reachable query/sat phase. Any
    // BddAbort (store left canonical — the panic precedes mutation) unwinds
    // to the `_inner` wrapper's `catch_abort`, which folds it into `None`.
    b.set_abort_limits(Some(crate::default_abort_node_budget()), deadline);
    let value = |b: &mut Bdd, p: usize, v: u64, in_next: bool| -> NodeId {
        let mut a = Bdd::TRUE;
        for i in 0..widths[p] {
            let var = if in_next {
                nxt_bit(p, i)
            } else {
                cur_bit(p, i)
            };
            let vb = b.var(var);
            let lit = if (v >> i) & 1 == 1 { vb } else { b.not(vb) };
            a = b.and(a, lit);
        }
        a
    };
    let mut init = Bdd::TRUE;
    for p in 0..np {
        let m = value(&mut b, p, net.init[p], false);
        init = b.and(init, m);
    }
    let mut trans = Bdd::FALSE;
    for t in &net.transitions {
        let mut rel = Bdd::TRUE;
        for p in 0..np {
            let mut rel_p = Bdd::FALSE;
            for v in 0..=net.bounds[p] {
                if v < t.pre[p] || v - t.pre[p] + t.post[p] > net.bounds[p] {
                    continue;
                }
                let cv = value(&mut b, p, v, false);
                let nvb = value(&mut b, p, v - t.pre[p] + t.post[p], true);
                let pair = b.and(cv, nvb);
                rel_p = b.or(rel_p, pair);
            }
            rel = b.and(rel, rel_p);
        }
        trans = b.or(trans, rel);
    }
    let current: Vec<u32> = (0..total as u32).collect();
    let next: Vec<u32> = (total as u32..2 * total as u32).collect();
    let r = b.reachable_within(init, trans, &current, &next, deadline)?;
    // cur→next rename map for the pre-image (var v ↦ total+v, monotone).
    let c2n: std::collections::HashMap<u32, u32> =
        (0..total as u32).map(|v| (v, total as u32 + v)).collect();

    // pre(y) = ∃next. (T ∧ y[cur→next]) — markings with a successor in `y`.
    fn pre(
        b: &mut Bdd,
        trans: NodeId,
        y: NodeId,
        c2n: &std::collections::HashMap<u32, u32>,
        next: &[u32],
    ) -> NodeId {
        let y_next = b.rename(y, c2n);
        b.and_exists(trans, y_next, next)
    }

    // sat(f) ⊆ R: the reachable markings satisfying f.
    fn sat(
        b: &mut Bdd,
        net: &BoundedNet,
        f: &Ctl,
        r: NodeId,
        trans: NodeId,
        c2n: &std::collections::HashMap<u32, u32>,
        next: &[u32],
        value: &dyn Fn(&mut Bdd, usize, u64, bool) -> NodeId,
        offset: &[usize],
        widths: &[usize],
    ) -> NodeId {
        match f {
            Ctl::Atom(p) => {
                let cs = charset(b, net, p, value, offset, widths);
                b.and(r, cs)
            }
            Ctl::Not(g) => {
                let s = sat(b, net, g, r, trans, c2n, next, value, offset, widths);
                let ns = b.not(s);
                b.and(r, ns) // complement within R
            }
            Ctl::And(g, h) => {
                let sg = sat(b, net, g, r, trans, c2n, next, value, offset, widths);
                let sh = sat(b, net, h, r, trans, c2n, next, value, offset, widths);
                b.and(sg, sh)
            }
            Ctl::Or(g, h) => {
                let sg = sat(b, net, g, r, trans, c2n, next, value, offset, widths);
                let sh = sat(b, net, h, r, trans, c2n, next, value, offset, widths);
                b.or(sg, sh)
            }
            Ctl::Ex(g) => {
                let s = sat(b, net, g, r, trans, c2n, next, value, offset, widths);
                let p = pre(b, trans, s, c2n, next);
                b.and(r, p)
            }
            Ctl::Ef(g) => {
                // μY. φ ∨ (R ∩ pre Y)
                let phi = sat(b, net, g, r, trans, c2n, next, value, offset, widths);
                let mut y = phi;
                loop {
                    let p = pre(b, trans, y, c2n, next);
                    let rp = b.and(r, p);
                    let ny = b.or(phi, rp);
                    if ny == y {
                        return y;
                    }
                    y = ny;
                }
            }
            Ctl::Eg(g) => {
                // EG φ = νY. φ ∧ (deadlock ∨ (R ∩ pre Y)) — MAXIMAL-PATH semantics:
                // a deadlock (reachable marking with NO successor) where φ holds
                // STAYS in EG φ, because the single-state finite path is a maximal
                // path on which φ holds globally. This is the non-totalized MCC
                // convention (matches tla-mdd's CtlEvaluator `EG φ = νZ. φ ∧
                // (deadlock ∨ EX Z)`, cross-checked vs tla-mc-core::CtlEngine).
                // Without the `deadlock ∨` term, AF = ¬EG¬ is WRONG at deadlocks
                // (the bug the exhaustive CtlChecker caught — 77 disagreements).
                let phi = sat(b, net, g, r, trans, c2n, next, value, offset, widths);
                // deadlock = R ∧ ¬(∃next. trans): reachable markings with no successor.
                let deadlock = {
                    let has_succ = b.exists(trans, next);
                    let no_succ = b.not(has_succ);
                    b.and(r, no_succ)
                };
                let mut y = phi;
                loop {
                    let p = pre(b, trans, y, c2n, next);
                    let rp = b.and(r, p);
                    let dead_or_step = b.or(deadlock, rp);
                    let ny = b.and(phi, dead_or_step);
                    if ny == y {
                        return y;
                    }
                    y = ny;
                }
            }
            Ctl::Eu(g, h) => {
                // E[φ U ψ] = μY. ψ ∨ (φ ∧ R ∩ pre Y)
                let phi = sat(b, net, g, r, trans, c2n, next, value, offset, widths);
                let psi = sat(b, net, h, r, trans, c2n, next, value, offset, widths);
                let mut y = psi;
                loop {
                    let p = pre(b, trans, y, c2n, next);
                    let rp = b.and(r, p);
                    let step = b.and(phi, rp);
                    let ny = b.or(psi, step);
                    if ny == y {
                        return y;
                    }
                    y = ny;
                }
            }
        }
    }

    let s = sat(
        &mut b, net, f, r, trans, &c2n, &next, &value, &offset, &widths,
    );
    // init ⊨ f  ⟺  init ⊆ sat(f)  ⟺  init ∧ ¬sat = ∅.
    let ns = b.not(s);
    let viol = b.and(init, ns);
    Some(viol == Bdd::FALSE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Explicit BFS reachable-state count for a 1-safe net — the ground-truth
    /// oracle (same firing rule, disabled on over-bound).
    fn bfs_count(net: &OneSafeNet) -> u128 {
        let np = net.num_places;
        let mut seen: HashSet<Vec<bool>> = HashSet::new();
        seen.insert(net.init.clone());
        let mut frontier = vec![net.init.clone()];
        while let Some(m) = frontier.pop() {
            for t in &net.transitions {
                // enabled: every pre place marked
                if !(0..np).all(|p| !t.pre[p] || m[p]) {
                    continue;
                }
                let mut next = m.clone();
                let mut ok = true;
                for p in 0..np {
                    let v = m[p] as i32 - t.pre[p] as i32 + t.post[p] as i32;
                    if v < 0 || v > 1 {
                        ok = false;
                        break;
                    }
                    next[p] = v == 1;
                }
                if ok && seen.insert(next.clone()) {
                    frontier.push(next);
                }
            }
        }
        seen.len() as u128
    }

    fn t(pre: &[u8], post: &[u8]) -> OneSafeTransition {
        OneSafeTransition {
            pre: pre.iter().map(|&x| x == 1).collect(),
            post: post.iter().map(|&x| x == 1).collect(),
        }
    }

    fn check(net: &OneSafeNet) {
        assert_eq!(
            reachable_count(net),
            Some(bfs_count(net)),
            "native BDD reachable-count must match explicit BFS"
        );
    }

    #[test]
    fn shuttle_two_states() {
        // p0 <-> p1 token shuttle: 2 reachable states.
        let net = OneSafeNet {
            num_places: 2,
            init: vec![true, false],
            transitions: vec![t(&[1, 0], &[0, 1]), t(&[0, 1], &[1, 0])],
        };
        assert_eq!(reachable_count(&net), Some(2));
        check(&net);
    }

    #[test]
    fn three_place_ring() {
        // token rotates p0->p1->p2->p0: 3 states.
        let net = OneSafeNet {
            num_places: 3,
            init: vec![true, false, false],
            transitions: vec![
                t(&[1, 0, 0], &[0, 1, 0]),
                t(&[0, 1, 0], &[0, 0, 1]),
                t(&[0, 0, 1], &[1, 0, 0]),
            ],
        };
        assert_eq!(reachable_count(&net), Some(3));
        check(&net);
    }

    #[test]
    fn independent_flags_product() {
        // 3 independent set-able flags (each 0->1, latched): reachable = 2^3 = 8.
        let net = OneSafeNet {
            num_places: 3,
            init: vec![false, false, false],
            transitions: vec![
                t(&[0, 0, 0], &[1, 0, 0]),
                t(&[0, 0, 0], &[0, 1, 0]),
                t(&[0, 0, 0], &[0, 0, 1]),
            ],
        };
        assert_eq!(reachable_count(&net), Some(8));
        check(&net);
    }

    #[test]
    fn fork_join_mutex() {
        // A small mutex: idle p0, crit p1, lock p2. acquire: p0,p2 -> p1.
        // release: p1 -> p0,p2. init idle + lock free.
        let net = OneSafeNet {
            num_places: 3,
            init: vec![true, false, true],
            transitions: vec![
                t(&[1, 0, 1], &[0, 1, 0]), // acquire
                t(&[0, 1, 0], &[1, 0, 1]), // release
            ],
        };
        check(&net);
        assert_eq!(reachable_count(&net), Some(2));
    }

    // ---- general bounded (multi-bit) ----

    fn bt(pre: &[u64], post: &[u64]) -> BoundedTransition {
        BoundedTransition {
            pre: pre.to_vec(),
            post: post.to_vec(),
        }
    }

    /// Explicit BFS over a general bounded net computing all four metrics (the
    /// oracle). Edges counts every enabled firing (incl. to already-seen states).
    fn bfs_bounded(net: &BoundedNet) -> StateSpaceMetrics {
        let np = net.bounds.len();
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        seen.insert(net.init.clone());
        let mut frontier = vec![net.init.clone()];
        let mut edges: u128 = 0;
        let mut max_in_place = *net.init.iter().max().unwrap_or(&0);
        let mut max_sum: u64 = net.init.iter().sum();
        while let Some(m) = frontier.pop() {
            for t in &net.transitions {
                if !(0..np).all(|p| m[p] >= t.pre[p]) {
                    continue;
                }
                let mut next = m.clone();
                let mut ok = true;
                for p in 0..np {
                    let v = m[p] - t.pre[p] + t.post[p];
                    if v > net.bounds[p] {
                        ok = false;
                        break;
                    }
                    next[p] = v;
                }
                if !ok {
                    continue;
                }
                edges += 1; // an enabled firing = one reachability-graph arc
                if seen.insert(next.clone()) {
                    max_in_place = max_in_place.max(*next.iter().max().unwrap_or(&0));
                    max_sum = max_sum.max(next.iter().sum());
                    frontier.push(next);
                }
            }
        }
        StateSpaceMetrics {
            states: seen.len() as u128,
            edges,
            max_token_in_place: max_in_place,
            max_token_sum: max_sum,
        }
    }

    fn check_bounded(net: &BoundedNet) {
        assert_eq!(
            state_space_metrics_bounded_checked(net),
            Some(bfs_bounded(net)),
            "native multi-bit BDD StateSpace metrics must match explicit BFS"
        );
        // The legacy saturating shim must agree wherever the count is exact.
        assert_eq!(
            state_space_metrics_bounded(net),
            bfs_bounded(net),
            "legacy shim must match the checked metrics on exact counts"
        );
    }

    #[test]
    fn counter_full_range() {
        // a single 0..=5 counter incremented: 6 states (bits=3, so 8 patterns
        // but only 0..5 reachable — exercises the unused-pattern handling).
        let net = BoundedNet {
            bounds: vec![5],
            init: vec![0],
            transitions: vec![bt(&[0], &[1])],
        };
        assert_eq!(reachable_count_bounded(&net), 6);
        check_bounded(&net);
    }

    #[test]
    fn two_independent_counters_product() {
        // bounds 3 and 2, each incrementable: (3+1)*(2+1) = 12 states.
        let net = BoundedNet {
            bounds: vec![3, 2],
            init: vec![0, 0],
            transitions: vec![bt(&[0, 0], &[1, 0]), bt(&[0, 0], &[0, 1])],
        };
        assert_eq!(reachable_count_bounded(&net), 12);
        check_bounded(&net);
    }

    /// Audit 2026-07-02 follow-up: the deadline-aware CTL/reachability/UB entry
    /// points arm a cooperative BddAbort and run a query/sat phase AFTER the
    /// internally-caught `reachable_within`. A whole-body `catch_abort` must
    /// fold an abort from ANY phase into `None` — never let a `BddAbort` panic
    /// escape into a production caller. An already-expired deadline forces the
    /// decline path deterministically; the normal (`None` deadline) path must
    /// still return `Some`.
    #[test]
    fn deadline_aware_entry_points_decline_without_panicking() {
        let net = BoundedNet {
            bounds: vec![3, 2],
            init: vec![0, 0],
            transitions: vec![bt(&[0, 0], &[1, 0]), bt(&[0, 0], &[0, 1])],
        };
        let expired = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let ctl = Ctl::Ef(Box::new(Ctl::Atom(Pred::Fireable(0))));

        // Expired deadline ⇒ fail-closed decline (None), no panic.
        assert_eq!(evaluate_ctl_within(&net, &ctl, Some(expired)), None);
        assert_eq!(
            evaluate_reachability_within(&net, &[Query::Ef(Pred::Fireable(0))], Some(expired)),
            None
        );
        assert_eq!(
            upper_bounds_bounded_within(&net, &[vec![1, 1]], Some(expired)),
            None
        );

        // No deadline ⇒ normal evaluation still returns a verdict.
        assert!(evaluate_ctl_within(&net, &ctl, None).is_some());
        assert!(
            evaluate_reachability_within(&net, &[Query::Ef(Pred::Fireable(0))], None).is_some()
        );
        assert!(upper_bounds_bounded_within(&net, &[vec![1, 1]], None).is_some());
    }

    #[test]
    fn weighted_conserved() {
        // bound 4, init [4,0]; transition moves 2 from p0 to 1 in p1.
        // 4->{(4,0),(2,1),(0,2)} = 3 states.
        let net = BoundedNet {
            bounds: vec![4, 4],
            init: vec![4, 0],
            transitions: vec![bt(&[2, 0], &[0, 1])],
        };
        assert_eq!(reachable_count_bounded(&net), 3);
        check_bounded(&net);
    }

    #[test]
    fn bounded_matches_one_safe_on_safe_net() {
        // The bounded encoder must agree with the 1-safe encoder on a 1-safe net.
        let bounded = BoundedNet {
            bounds: vec![1, 1, 1],
            init: vec![1, 0, 0],
            transitions: vec![
                bt(&[1, 0, 0], &[0, 1, 0]),
                bt(&[0, 1, 0], &[0, 0, 1]),
                bt(&[0, 0, 1], &[1, 0, 0]),
            ],
        };
        check_bounded(&bounded);
        assert_eq!(reachable_count_bounded(&bounded), 3);
    }

    /// Explicit reachable-marking set (oracle for the fireability queries).
    fn bfs_reach_set(net: &BoundedNet) -> Vec<Vec<u64>> {
        let np = net.bounds.len();
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        seen.insert(net.init.clone());
        let mut frontier = vec![net.init.clone()];
        while let Some(m) = frontier.pop() {
            for t in &net.transitions {
                if !(0..np).all(|p| m[p] >= t.pre[p]) {
                    continue;
                }
                let mut next = m.clone();
                let mut ok = true;
                for p in 0..np {
                    let v = m[p] - t.pre[p] + t.post[p];
                    if v > net.bounds[p] {
                        ok = false;
                        break;
                    }
                    next[p] = v;
                }
                if ok && seen.insert(next.clone()) {
                    frontier.push(next);
                }
            }
        }
        seen.into_iter().collect()
    }

    fn eval_pred(net: &BoundedNet, m: &[u64], p: &Pred) -> bool {
        match p {
            Pred::Fireable(t) => {
                let tr = &net.transitions[*t];
                (0..net.bounds.len()).all(|pl| m[pl] >= tr.pre[pl])
            }
            Pred::TokenLe { coeffs, k } => {
                let s: i128 = coeffs.iter().zip(m).map(|(&c, &mv)| c * mv as i128).sum();
                s <= *k
            }
            Pred::And(cs) => cs.iter().all(|c| eval_pred(net, m, c)),
            Pred::Or(cs) => cs.iter().any(|c| eval_pred(net, m, c)),
            Pred::Not(c) => !eval_pred(net, m, c),
        }
    }

    #[test]
    fn cardinality_reachability_matches_bfs() {
        // Two independent counters (bounds 3, 2). Cardinality atoms over R.
        let net = BoundedNet {
            bounds: vec![3, 2],
            init: vec![0, 0],
            transitions: vec![bt(&[0, 0], &[1, 0]), bt(&[0, 0], &[0, 1])],
        };
        let r = bfs_reach_set(&net);
        let mk = |coeffs: Vec<i128>, k: i128| Pred::TokenLe { coeffs, k };
        // EF(p0+p1 <= 1); AG(p0 <= 3); EF(p0 >= 3 i.e. -p0 <= -3); AG(p0+p1 <= 5).
        let preds = [
            mk(vec![1, 1], 1),
            mk(vec![1, 0], 3),
            mk(vec![-1, 0], -3),
            mk(vec![1, 1], 5),
        ];
        let queries: Vec<Query> = preds
            .iter()
            .flat_map(|p| {
                let c = |p: &Pred| match p {
                    Pred::TokenLe { coeffs, k } => Pred::TokenLe {
                        coeffs: coeffs.clone(),
                        k: *k,
                    },
                    _ => unreachable!(),
                };
                [Query::Ef(c(p)), Query::Ag(c(p))]
            })
            .collect();
        let got = evaluate_reachability(&net, &queries);
        let mut want = Vec::new();
        for p in &preds {
            want.push(r.iter().any(|m| eval_pred(&net, m, p)));
            want.push(r.iter().all(|m| eval_pred(&net, m, p)));
        }
        assert_eq!(
            got, want,
            "native BDD cardinality EF/AG must match BFS over R"
        );
    }

    // ---- CTL (pre-image fixpoints) ----

    use std::collections::BTreeMap;

    /// Explicit CTL sat-set over the reachable graph (the oracle).
    fn explicit_ctl_sat(net: &BoundedNet, f: &Ctl) -> (HashSet<Vec<u64>>, Vec<u64>) {
        let np = net.bounds.len();
        // reachable graph
        let reach: Vec<Vec<u64>> = bfs_reach_set(net);
        let reach_set: HashSet<Vec<u64>> = reach.iter().cloned().collect();
        let mut succ: BTreeMap<Vec<u64>, Vec<Vec<u64>>> = BTreeMap::new();
        for m in &reach {
            let mut ss = Vec::new();
            for t in &net.transitions {
                if !(0..np).all(|p| m[p] >= t.pre[p]) {
                    continue;
                }
                let mut n = m.clone();
                let mut ok = true;
                for p in 0..np {
                    let v = m[p] - t.pre[p] + t.post[p];
                    if v > net.bounds[p] {
                        ok = false;
                        break;
                    }
                    n[p] = v;
                }
                if ok {
                    ss.push(n);
                }
            }
            succ.insert(m.clone(), ss);
        }
        fn rec(
            net: &BoundedNet,
            f: &Ctl,
            reach: &[Vec<u64>],
            reach_set: &HashSet<Vec<u64>>,
            succ: &BTreeMap<Vec<u64>, Vec<Vec<u64>>>,
        ) -> HashSet<Vec<u64>> {
            match f {
                Ctl::Atom(p) => reach
                    .iter()
                    .filter(|m| eval_pred(net, m, p))
                    .cloned()
                    .collect(),
                Ctl::Not(g) => {
                    let s = rec(net, g, reach, reach_set, succ);
                    reach_set.difference(&s).cloned().collect()
                }
                Ctl::And(g, h) => {
                    let a = rec(net, g, reach, reach_set, succ);
                    let b = rec(net, h, reach, reach_set, succ);
                    a.intersection(&b).cloned().collect()
                }
                Ctl::Or(g, h) => {
                    let a = rec(net, g, reach, reach_set, succ);
                    let b = rec(net, h, reach, reach_set, succ);
                    a.union(&b).cloned().collect()
                }
                Ctl::Ex(g) => {
                    let s = rec(net, g, reach, reach_set, succ);
                    reach
                        .iter()
                        .filter(|m| succ[*m].iter().any(|n| s.contains(n)))
                        .cloned()
                        .collect()
                }
                Ctl::Ef(g) => {
                    let mut s = rec(net, g, reach, reach_set, succ);
                    loop {
                        let mut grew = false;
                        for m in reach {
                            if !s.contains(m) && succ[m].iter().any(|n| s.contains(n)) {
                                s.insert(m.clone());
                                grew = true;
                            }
                        }
                        if !grew {
                            return s;
                        }
                    }
                }
                Ctl::Eg(g) => {
                    // Maximal-path EG: a DEADLOCK (no successors) where φ holds
                    // stays in EG φ; only drop a state that has successors but none
                    // in the set (matches the deadlock term in `evaluate_ctl`).
                    let phi = rec(net, g, reach, reach_set, succ);
                    let mut s = phi.clone();
                    loop {
                        let mut shrunk = false;
                        let cur: Vec<Vec<u64>> = s.iter().cloned().collect();
                        for m in cur {
                            if !succ[&m].is_empty() && !succ[&m].iter().any(|n| s.contains(n)) {
                                s.remove(&m);
                                shrunk = true;
                            }
                        }
                        if !shrunk {
                            return s;
                        }
                    }
                }
                Ctl::Eu(g, h) => {
                    // E[φ U ψ]: least set with ψ ∪ {m∈φ : some succ in set}.
                    let phi = rec(net, g, reach, reach_set, succ);
                    let psi = rec(net, h, reach, reach_set, succ);
                    let mut s = psi.clone();
                    loop {
                        let mut grew = false;
                        for m in reach {
                            if !s.contains(m)
                                && phi.contains(m)
                                && succ[m].iter().any(|n| s.contains(n))
                            {
                                s.insert(m.clone());
                                grew = true;
                            }
                        }
                        if !grew {
                            return s;
                        }
                    }
                }
            }
        }
        let s = rec(net, f, &reach, &reach_set, &succ);
        (s, net.init.clone())
    }

    fn check_ctl(net: &BoundedNet, f: Ctl, f2: Ctl) {
        let got = evaluate_ctl(net, &f);
        let (sat, init) = explicit_ctl_sat(net, &f2);
        assert_eq!(
            got,
            sat.contains(&init),
            "native CTL must match explicit CTL"
        );
    }

    #[test]
    fn ctl_pre_image_fixpoints_match_explicit() {
        // Mutex: idle p0, crit p1, lock p2. acquire t0, release t1.
        let net = BoundedNet {
            bounds: vec![1, 1, 1],
            init: vec![1, 0, 1],
            transitions: vec![bt(&[1, 0, 1], &[0, 1, 0]), bt(&[0, 1, 0], &[1, 0, 1])],
        };
        let crit = || Pred::TokenLe {
            coeffs: vec![0, -1, 0],
            k: -1,
        }; // p1 >= 1
           // EF crit (can reach critical section) — true.
        check_ctl(
            &net,
            Ctl::Ef(Box::new(Ctl::Atom(crit()))),
            Ctl::Ef(Box::new(Ctl::Atom(crit()))),
        );
        // EG ¬crit (a path never entering crit) — true (stay idle? no transition keeps idle; mutex toggles).
        check_ctl(
            &net,
            Ctl::Eg(Box::new(Ctl::Not(Box::new(Ctl::Atom(crit()))))),
            Ctl::Eg(Box::new(Ctl::Not(Box::new(Ctl::Atom(crit()))))),
        );
        // EX crit (acquire fireable from init) — true.
        check_ctl(
            &net,
            Ctl::Ex(Box::new(Ctl::Atom(crit()))),
            Ctl::Ex(Box::new(Ctl::Atom(crit()))),
        );
        // EG crit (an infinite path always-critical) — false (must release).
        check_ctl(
            &net,
            Ctl::Eg(Box::new(Ctl::Atom(crit()))),
            Ctl::Eg(Box::new(Ctl::Atom(crit()))),
        );

        // A counter 0..3, increment-only: EG(p<=3) true (always), EF(p>=3) true.
        let counter = BoundedNet {
            bounds: vec![3],
            init: vec![0],
            transitions: vec![bt(&[0], &[1])],
        };
        let ge3 = || Pred::TokenLe {
            coeffs: vec![-1],
            k: -3,
        }; // p >= 3
        check_ctl(
            &counter,
            Ctl::Ef(Box::new(Ctl::Atom(ge3()))),
            Ctl::Ef(Box::new(Ctl::Atom(ge3()))),
        );
        // EG(p>=3): only the top state self-loops? increment-only has NO self-loop at 3
        // (firing would exceed bound ⇒ disabled), so state 3 is a deadlock ⇒ EG(p>=3) false.
        check_ctl(
            &counter,
            Ctl::Eg(Box::new(Ctl::Atom(ge3()))),
            Ctl::Eg(Box::new(Ctl::Atom(ge3()))),
        );
    }

    /// Explicit: does the reachable graph have a cycle through an accepting state?
    fn explicit_fair_cycle(net: &BoundedNet, accepting: &Pred) -> bool {
        let np = net.bounds.len();
        let reach = bfs_reach_set(net);
        let succ = |m: &[u64]| -> Vec<Vec<u64>> {
            let mut ss = Vec::new();
            for t in &net.transitions {
                if !(0..np).all(|p| m[p] >= t.pre[p]) {
                    continue;
                }
                let mut n = m.to_vec();
                let mut ok = true;
                for p in 0..np {
                    let v = m[p] - t.pre[p] + t.post[p];
                    if v > net.bounds[p] {
                        ok = false;
                        break;
                    }
                    n[p] = v;
                }
                if ok {
                    ss.push(n);
                }
            }
            ss
        };
        // For each reachable accepting state s, is s reachable from itself in ≥1 step?
        for s in &reach {
            if !eval_pred(net, s, accepting) {
                continue;
            }
            let mut seen: HashSet<Vec<u64>> = HashSet::new();
            let mut frontier: Vec<Vec<u64>> = succ(s); // ≥1 step
            while let Some(m) = frontier.pop() {
                if &m == s {
                    return true; // returned to s ⇒ cycle through accepting s
                }
                if seen.insert(m.clone()) {
                    frontier.extend(succ(&m));
                }
            }
        }
        false
    }

    #[test]
    fn fair_cycle_matches_explicit_buchi_emptiness() {
        // Mutex toggles idle<->crit forever ⇒ a cycle through crit exists.
        let mutex = BoundedNet {
            bounds: vec![1, 1, 1],
            init: vec![1, 0, 1],
            transitions: vec![bt(&[1, 0, 1], &[0, 1, 0]), bt(&[0, 1, 0], &[1, 0, 1])],
        };
        let crit = Pred::TokenLe {
            coeffs: vec![0, -1, 0],
            k: -1,
        }; // p1 >= 1
        assert!(fair_cycle_exists(&mutex, &crit, None));
        assert_eq!(
            fair_cycle_exists(&mutex, &crit, None),
            explicit_fair_cycle(&mutex, &crit)
        );

        // Increment-only counter is ACYCLIC ⇒ no fair cycle through any state.
        let counter = BoundedNet {
            bounds: vec![3],
            init: vec![0],
            transitions: vec![bt(&[0], &[1])],
        };
        let any = Pred::TokenLe {
            coeffs: vec![0],
            k: 0,
        }; // every marking (0 <= 0)
        assert!(!fair_cycle_exists(&counter, &any, None));
        assert_eq!(
            fair_cycle_exists(&counter, &any, None),
            explicit_fair_cycle(&counter, &any)
        );

        // Counter with a reset (3 -> 0) ⇒ a cycle exists; accepting = {p==0} on it.
        let cyclic = BoundedNet {
            bounds: vec![3],
            init: vec![0],
            transitions: vec![bt(&[0], &[1]), bt(&[3], &[0])],
        };
        let at0 = Pred::TokenLe {
            coeffs: vec![1],
            k: 0,
        }; // p == 0
        assert!(fair_cycle_exists(&cyclic, &at0, None));
        assert_eq!(
            fair_cycle_exists(&cyclic, &at0, None),
            explicit_fair_cycle(&cyclic, &at0)
        );
        // accepting = {p==2}: also on the cycle ⇒ true.
        let at2 = Pred::And(vec![
            Pred::TokenLe {
                coeffs: vec![1],
                k: 2,
            },
            Pred::TokenLe {
                coeffs: vec![-1],
                k: -2,
            },
        ]);
        assert_eq!(
            fair_cycle_exists(&cyclic, &at2, None),
            explicit_fair_cycle(&cyclic, &at2)
        );
    }

    #[test]
    fn generalized_fair_cycle_multi_acceptance() {
        // 1 token rotating between p0 and p1: states [1,0] <-> [0,1].
        let rot = BoundedNet {
            bounds: vec![1, 1],
            init: vec![1, 0],
            transitions: vec![bt(&[1, 0], &[0, 1]), bt(&[0, 1], &[1, 0])],
        };
        let in_p0 = Pred::TokenLe {
            coeffs: vec![-1, 0],
            k: -1,
        }; // p0 >= 1 ([1,0])
        let in_p1 = Pred::TokenLe {
            coeffs: vec![0, -1],
            k: -1,
        }; // p1 >= 1 ([0,1])
        let unsat = Pred::TokenLe {
            coeffs: vec![-1, 0],
            k: -2,
        }; // p0 >= 2 (impossible)
           // The rotation cycle visits BOTH p0-marked and p1-marked states i.o.
        assert!(
            fair_cycle_exists_generalized(&rot, &[in_p0.clone(), in_p1.clone()], None),
            "the rotation cycle must hit both acceptance sets"
        );
        // One set reduces EXACTLY to the plain single-set fair-cycle.
        assert_eq!(
            fair_cycle_exists_generalized(&rot, std::slice::from_ref(&in_p0), None),
            fair_cycle_exists(&rot, &in_p0, None)
        );
        // No reachable state satisfies `unsat`, so no fair cycle can visit it.
        assert!(
            !fair_cycle_exists_generalized(&rot, &[in_p0.clone(), unsat], None),
            "cannot visit an unsatisfiable acceptance set"
        );
        // Empty acceptance ⇒ any reachable cycle qualifies (the rotation is one).
        assert!(fair_cycle_exists_generalized(&rot, &[], None));
    }

    #[test]
    fn fireability_reachability_matches_bfs() {
        // Mutex: idle p0, crit p1, lock p2. acquire t0: p0,p2->p1. release t1: p1->p0,p2.
        let net = BoundedNet {
            bounds: vec![1, 1, 1],
            init: vec![1, 0, 1],
            transitions: vec![bt(&[1, 0, 1], &[0, 1, 0]), bt(&[0, 1, 0], &[1, 0, 1])],
        };
        let r = bfs_reach_set(&net);
        let preds = [
            Pred::Fireable(0), // can acquire
            Pred::Fireable(1), // can release
            Pred::Not(Box::new(Pred::And(vec![
                Pred::Fireable(0),
                Pred::Fireable(1),
            ]))), // never both
        ];
        // Build the query list: EF and AG of each predicate.
        let mut queries = Vec::new();
        for p in &preds {
            let clone = |p: &Pred| -> Pred {
                match p {
                    Pred::Fireable(t) => Pred::Fireable(*t),
                    Pred::And(_) => Pred::And(vec![Pred::Fireable(0), Pred::Fireable(1)]),
                    _ => Pred::Not(Box::new(Pred::And(vec![
                        Pred::Fireable(0),
                        Pred::Fireable(1),
                    ]))),
                }
            };
            queries.push(Query::Ef(clone(p)));
            queries.push(Query::Ag(clone(p)));
        }
        let got = evaluate_reachability(&net, &queries);
        // Oracle.
        let mut want = Vec::new();
        for p in &preds {
            want.push(r.iter().any(|m| eval_pred(&net, m, p))); // EF
            want.push(r.iter().all(|m| eval_pred(&net, m, p))); // AG
        }
        assert_eq!(
            got, want,
            "native BDD fireability EF/AG must match BFS over R"
        );
    }
}
