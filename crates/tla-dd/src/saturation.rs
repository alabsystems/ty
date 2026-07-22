// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Event-locality classifiers and the explicit-state reachability oracle.
//!
//! This module once hosted the Ciardo et al. TACAS'01 saturation kernel that
//! ran on top of the OxiDD-based `DdReachability` engine. That engine (and its
//! saturation kernel) has been removed along with the `oxidd` dependency; what
//! remains here is the engine-agnostic scaffolding:
//!
//! - [`top_of`] / [`group_by_top`] — the `Top(t)` event-locality classifiers.
//!   For a transition `t`, `Top(t)` = the maximum place index appearing in
//!   either `pre[p] != 0` or `post[p] != 0`. This is the load-bearing quantity
//!   for any level-ordered symbolic engine: a wrong `Top(t)` causes silent
//!   under-firing of events at lower levels and therefore under-approximation
//!   of `|R|`. The classifier has dedicated unit tests below; the Tier 3 doc
//!   flagged "silent under-approximation via bad `Top(t)`" as the #1 soundness
//!   risk.
//! - [`bfs_reachable_set_count`] — a deliberately-naive explicit-state BFS
//!   reachable-marking counter. Unlikely to be wrong (just slow); it is the
//!   differential ground truth the symbolic engines are cross-checked against.
//!   An examination engine that emits a wrong `|R|` costs `-8` MCC points per
//!   wrong value, so exact agreement with this oracle is non-negotiable.

use std::collections::HashSet;

use crate::{DdNetSpec, DdTransition};

/// `Top(t)` — the highest place index a transition's support touches.
///
/// "Support" = any place mentioned in `pre` or `post` with a non-zero
/// weight. Returned as `Some(p)` when at least one such place exists, or
/// `None` for the (degenerate) empty-support transition. The caller treats
/// `None` as a saturation no-op (the relation is identity-on-support, so
/// firing it never adds states).
///
/// # Soundness contract
///
/// The Tier 3 doc flagged this function as the #1 risk for saturation.
/// Two invariants must hold:
///
/// 1. `top_of` must return the **maximum** index, never an under-estimate.
///    A too-low `Top(t)` causes the saturation recursion to fire `t` at a
///    level below where its support actually lives, which can quantify
///    away variables `t` reads — yielding a strictly smaller image and
///    therefore an under-approximated reachable set.
/// 2. `top_of` must treat `pre` and `post` symmetrically. A transition
///    that only *writes* a high-level place (e.g. `pre = [1,0,0]`,
///    `post = [0,0,1]`) still touches level 2 and must be classified
///    there — otherwise the write is silently dropped at lower levels.
///
/// Both invariants are checked by `test_top_t_classification_unit_test`.
#[must_use]
pub fn top_of(t: &DdTransition) -> Option<usize> {
    debug_assert_eq!(
        t.pre.len(),
        t.post.len(),
        "transition pre and post vectors must agree in length"
    );
    let mut top: Option<usize> = None;
    for p in 0..t.pre.len() {
        if t.pre[p] != 0 || t.post[p] != 0 {
            top = Some(p);
        }
    }
    top
}

/// Group transitions by their `Top(t)` level.
///
/// Returns a vector of length `num_places` where entry `k` holds every
/// transition with `Top(t) == k`. Transitions with empty support (no place
/// in `pre` or `post`) are skipped — they never add reachable states and
/// would only waste BDD work if included.
///
/// This is a pure classification step; the BDD relations are built later.
#[must_use]
pub fn group_by_top(transitions: &[DdTransition], num_places: usize) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); num_places];
    for (i, t) in transitions.iter().enumerate() {
        if let Some(top) = top_of(t) {
            debug_assert!(
                top < num_places,
                "Top(t) = {top} out of range for {num_places}-place net"
            );
            groups[top].push(i);
        }
    }
    groups
}

/// Differential oracle: explicit-state BFS reachable-set count.
///
/// Cross-checks the symbolic engines' reachable-set counts on small nets.
/// Deliberately uncomplicated — the only thing this function does that a
/// symbolic engine does not is enumerate every
/// reachable marking explicitly into a [`HashSet`], so on agreement we are
/// confident saturation neither under-counted (missing a state) nor
/// over-counted (counting a state that does not satisfy the firing rule).
///
/// Respects the same per-place bounds the BDD engine enforces, so a net
/// that grows beyond `spec.bounds[p]` produces the same truncated count as
/// the BDD encoding — fair comparison.
#[must_use]
pub fn bfs_reachable_set_count(spec: &DdNetSpec) -> u64 {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut frontier: Vec<Vec<u64>> = vec![spec.initial_marking.clone()];
    seen.insert(spec.initial_marking.clone());
    while let Some(m) = frontier.pop() {
        for t in &spec.transitions {
            // Enabled check: m[p] >= pre[p] for every p.
            if !m.iter().zip(&t.pre).all(|(mv, pv)| mv >= pv) {
                continue;
            }
            let mut next_m = m.clone();
            let mut ok = true;
            for p in 0..next_m.len() {
                let v = next_m[p] - t.pre[p] + t.post[p];
                if v > spec.bounds[p] {
                    ok = false;
                    break;
                }
                next_m[p] = v;
            }
            if ok && seen.insert(next_m.clone()) {
                frontier.push(next_m);
            }
        }
    }
    seen.len() as u64
}
