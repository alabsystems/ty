// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Place invariant computation for structural Petri net analysis.
//!
//! Computes semi-positive P-invariants using the Farkas algorithm
//! (Fourier-Motzkin elimination on the incidence matrix). A P-invariant
//! is a non-negative integer vector y ≥ 0 satisfying y^T · C = 0, where
//! C is the incidence matrix. The key property: for any reachable marking
//! m, y^T · m = y^T · m₀ (invariant quantity).
//!
//! Structural bounds derived from P-invariants enable answering MCC
//! examinations without state space exploration:
//! - **OneSafe**: prove all places ≤ 1 token structurally
//! - **UpperBounds**: tight bounds on place-set token sums

use crate::petri_net::PetriNet;

/// A semi-positive P-invariant: y ≥ 0 with y^T · C = 0.
///
/// The support is stored **sparsely** — only the nonzero `(place, weight)`
/// entries, sorted ascending by place index, all weights > 0. A dense
/// per-place vector is O(num_places) *per invariant*, which on wide nets is
/// catastrophic: AirplaneLD-PT-4000 yields ~12 002 (mostly constant-place,
/// support-1) invariants over 28 019 places, i.e. ~2.7 GB per dense copy and
/// an OOM once a few lanes hold copies concurrently. Sparse storage is
/// O(support), so each constant-place invariant costs one entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PInvariant {
    /// Nonzero place weights as `(place_index, weight)`, sorted ascending by
    /// place index. All weights > 0. Empty only for the degenerate all-zero
    /// vector, which is never produced (such rows are dropped).
    pub weights: Vec<(u32, u64)>,
    /// Conserved quantity: y^T · m₀. Equal to y^T · m for all reachable m.
    pub token_count: u64,
}

impl PInvariant {
    /// Weight of `place` in this invariant (`0` if outside the support).
    /// O(log support) via binary search on the sorted entries.
    pub(crate) fn weight(&self, place: usize) -> u64 {
        let p = place as u32;
        self.weights
            .binary_search_by_key(&p, |&(q, _)| q)
            .map(|i| self.weights[i].1)
            .unwrap_or(0)
    }

    /// True iff `place` has a positive weight (is in the support).
    pub(crate) fn covers(&self, place: usize) -> bool {
        self.weight(place) > 0
    }

    /// Number of places in the support (all have weight > 0).
    pub(crate) fn support_len(&self) -> usize {
        self.weights.len()
    }

    /// Iterate the support as `(place, weight)` with `weight > 0`, ascending by
    /// place index.
    pub(crate) fn support(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.weights.iter().map(|&(p, w)| (p as usize, w))
    }
}

/// Maximum augmented row count before aborting the Farkas algorithm.
/// Prevents combinatorial explosion on pathological nets.
const MAX_ROWS: usize = 10_000;

/// Compute semi-positive P-invariants via the Farkas algorithm.
///
/// Returns all minimal semi-positive P-invariants discovered. For nets
/// with no transitions, returns a unit invariant per place (every place
/// is trivially invariant). Aborts gracefully if intermediate row count
/// exceeds [`MAX_ROWS`], returning whatever invariants were found.
pub(crate) fn compute_p_invariants(net: &PetriNet) -> Vec<PInvariant> {
    let np = net.num_places();
    let nt = net.num_transitions();

    if np == 0 {
        return vec![];
    }

    // No transitions: every place has constant token count.
    if nt == 0 {
        return (0..np)
            .map(|p| PInvariant {
                weights: vec![(p as u32, 1)],
                token_count: net.initial_marking[p],
            })
            .collect();
    }

    // SPARSE Farkas: one row per place, eliminating transition columns.
    let rows = sparse_incidence_rows(net, np);
    farkas_elimination_sparse(rows, nt, &net.initial_marking).0
}

/// A single sparse row of the augmented Farkas matrix `[ C-part | I-part ]`.
///
/// Both halves store only their nonzero entries, kept **sorted ascending by
/// index** so that two rows can be combined by a linear merge. `left` is the
/// incidence (elimination) part — `(column, coefficient)`; `right` is the
/// accumulator part — `(index, weight)` — which becomes the invariant's
/// weight vector once `left` is fully eliminated.
struct SparseRow {
    /// Incidence columns being eliminated: `(col, coeff)`, sorted by `col`,
    /// no zero coefficients.
    left: Vec<(u32, i64)>,
    /// Accumulated weights: `(index, weight)`, sorted by `index`, all > 0.
    right: Vec<(u32, u64)>,
}

/// Build the place-indexed sparse incidence rows: row `p` holds the nonzero
/// `(transition, C[p][t])` entries, where `C[p][t] = out(p,t) - in(p,t)`.
///
/// This is the sparse analogue of the dense `incidence_matrix` — it never
/// allocates the O(places × transitions) dense array (which alone is tens of
/// gigabytes on the largest MCC nets, e.g. AirplaneLD-PT-4000 at 28 019 ×
/// 32 008), only O(arcs) storage. The right (identity) half is created inside
/// [`farkas_elimination_sparse`].
fn sparse_incidence_rows(net: &PetriNet, np: usize) -> Vec<Vec<(u32, i64)>> {
    // Accumulate per place: a transition may appear in both inputs and
    // outputs of the same place (net effect), and weights must combine.
    let mut acc: Vec<std::collections::BTreeMap<u32, i64>> =
        (0..np).map(|_| std::collections::BTreeMap::new()).collect();
    for (tidx, trans) in net.transitions.iter().enumerate() {
        let t = tidx as u32;
        for arc in &trans.inputs {
            *acc[arc.place.0 as usize].entry(t).or_insert(0) -= arc.weight as i64;
        }
        for arc in &trans.outputs {
            *acc[arc.place.0 as usize].entry(t).or_insert(0) += arc.weight as i64;
        }
    }
    acc.into_iter()
        .map(|m| m.into_iter().filter(|&(_, v)| v != 0).collect())
        .collect()
}

/// Build the dense incidence matrix `C[p][t] = out(p,t) - in(p,t)`.
///
/// Retained only for differential test assertions (verifying `y^T·C = 0`).
/// The production P-invariant / T-semiflow paths use the sparse rows above —
/// the dense matrix is O(places × transitions) and must never be built for
/// real MCC nets.
#[cfg(test)]
fn incidence_matrix(net: &PetriNet) -> Vec<Vec<i64>> {
    let np = net.num_places();
    let nt = net.num_transitions();
    let mut c = vec![vec![0i64; nt]; np];

    for (tidx, trans) in net.transitions.iter().enumerate() {
        for arc in &trans.inputs {
            c[arc.place.0 as usize][tidx] -= arc.weight as i64;
        }
        for arc in &trans.outputs {
            c[arc.place.0 as usize][tidx] += arc.weight as i64;
        }
    }
    c
}

/// Sparse Farkas (Fourier-Motzkin) elimination for semi-positive invariants.
///
/// The augmented matrix is `[ C-part | I-part ]` with one row per accumulator
/// index (`num_right` rows — places for P-invariants, transitions for
/// T-semiflows). Transition (resp. place) columns `0..num_cols` are eliminated
/// left-to-right; afterwards every row whose `left` half is empty is an
/// invariant, read off from its `right` half.
///
/// This is the sparse, memory-bounded analogue of the textbook dense
/// elimination. It is mathematically identical — same column order, same
/// positive×negative combination rule, same GCD reduction, same `MAX_ROWS`
/// and coefficient-overflow guards — so it yields the *same* set of minimal
/// invariants, but never allocates a dense matrix.
///
/// ## Min-column bucketing (why this is fast *and* small)
///
/// After column `j` is eliminated, every surviving row has a zero coefficient
/// in columns `0..=j`; equivalently, a row's *smallest* present column is
/// strictly increasing as elimination proceeds. We therefore bucket each row
/// by its minimum present column. Processing column `j` touches only
/// `bucket[j]` — the rows whose minimum column is exactly `j` (a nonzero
/// coefficient there) — instead of rescanning the whole matrix. Each
/// pos×neg combination cancels column `j`, so the resulting row's minimum
/// column is `> j` and it lands in a later bucket. Rows that have a nonzero in
/// column `j` but no opposite-sign partner cannot be zeroed there, so they are
/// dropped (they can never extend to an invariant) — exactly as in the dense
/// formulation.
///
/// `initial_rows[i]` is the `left` (incidence) half of row `i`; the `right`
/// half is the identity row `{ i: 1 }`, created here.
fn farkas_elimination_sparse(
    initial_rows: Vec<Vec<(u32, i64)>>,
    num_cols: usize,
    initial_marking: &[u64],
) -> (Vec<PInvariant>, bool) {
    // Bucket non-eliminated rows by their minimum present column. Rows whose
    // `left` is already empty are completed invariant candidates (`done`).
    let mut buckets: Vec<Vec<SparseRow>> = (0..num_cols.max(1)).map(|_| Vec::new()).collect();
    let mut done: Vec<SparseRow> = Vec::new();
    let mut total_rows: usize = 0;

    for (idx, left) in initial_rows.into_iter().enumerate() {
        let right = vec![(idx as u32, 1u64)];
        let row = SparseRow { left, right };
        match row.left.first() {
            Some(&(minc, _)) => buckets[minc as usize].push(row),
            None => done.push(row),
        }
        total_rows += 1;
    }

    let mut truncated = false;

    'columns: for j in 0..num_cols {
        // Rows whose minimum present column is exactly `j` — i.e. the only
        // rows with a nonzero coefficient in column `j`. Consume the bucket.
        let bucket = std::mem::take(&mut buckets[j]);
        if bucket.is_empty() {
            continue;
        }
        total_rows -= bucket.len();

        // Partition by sign of the column-`j` coefficient, which for these
        // rows is `left[0]` (the minimum column == j, guaranteed nonzero).
        let mut pos: Vec<SparseRow> = Vec::new();
        let mut neg: Vec<SparseRow> = Vec::new();
        for row in bucket {
            if row.left[0].1 > 0 {
                pos.push(row);
            } else {
                neg.push(row);
            }
        }

        // Combine each positive×negative pair to cancel column `j`.
        //
        // Coefficient-explosion guard (#GPPP-PT-C1000N panic fix): Farkas
        // elimination can blow combined coefficients up exponentially even
        // from a {-1,0,1} incidence matrix. We compute each entry in i128/
        // u128 and skip any pair whose result does not fit native i64/u64.
        // Skipping is sound — produced invariants still satisfy y^T·C = 0;
        // we merely return a subset (flagged incomplete), matching the
        // MAX_ROWS truncation policy so callers fall back to BMC/BFS.
        for p in &pos {
            for n in &neg {
                if total_rows >= MAX_ROWS {
                    truncated = true;
                    break 'columns;
                }
                let a = p.left[0].1 as i128; // positive
                let b = -(n.left[0].1 as i128); // positive (negated negative)

                // new = b·p + a·n, which zeroes column j in both halves.
                let Some(new_left) = combine_sparse_left(&p.left, &n.left, b, a) else {
                    truncated = true;
                    continue;
                };
                let Some(new_right) = combine_sparse_right(&p.right, &n.right, b, a) else {
                    truncated = true;
                    continue;
                };

                let (mut new_left, mut new_right) = (new_left, new_right);
                reduce_sparse(&mut new_left, &mut new_right);

                if new_right.iter().any(|&(_, v)| v > 0) {
                    let row = SparseRow {
                        left: new_left,
                        right: new_right,
                    };
                    match row.left.first() {
                        // Cancelling column j leaves only columns > j, so the
                        // new minimum column is strictly past j — a later
                        // bucket — preserving the monotonic invariant.
                        Some(&(minc, _)) => buckets[minc as usize].push(row),
                        None => done.push(row),
                    }
                    total_rows += 1;
                }
            }
        }
    }

    let complete = !truncated;

    // Extract invariants from fully-eliminated rows (empty `left`).
    //
    // token_count = y^T · m₀, accumulated in u128 with overflow guards so a
    // single oversized invariant cannot panic the whole computation; an
    // overflowing invariant is simply dropped (subset semantics).
    let mut invariants = Vec::new();
    for row in done {
        if !row.right.iter().any(|&(_, v)| v > 0) {
            continue;
        }
        let mut acc: u128 = 0;
        let mut overflowed = false;
        for &(idx, w) in &row.right {
            let term = (w as u128) * (initial_marking[idx as usize] as u128);
            match acc.checked_add(term) {
                Some(v) => acc = v,
                None => {
                    overflowed = true;
                    break;
                }
            }
        }
        if overflowed {
            continue;
        }
        let Ok(token_count) = u64::try_from(acc) else {
            continue;
        };
        // `row.right` is already the sparse, sorted, all-positive weight vector
        // the invariant needs — move it in directly (no dense reconstruction).
        invariants.push(PInvariant {
            weights: row.right,
            token_count,
        });
    }

    deduplicate_invariants(&mut invariants);
    (invariants, complete)
}

/// Merge two sorted sparse i64 rows as `coef_p·p + coef_n·n`, dropping zeros.
/// Computes in i128 and downcasts; returns `None` if any entry overflows i64.
fn combine_sparse_left(
    p: &[(u32, i64)],
    n: &[(u32, i64)],
    coef_p: i128,
    coef_n: i128,
) -> Option<Vec<(u32, i64)>> {
    let mut out: Vec<(u32, i64)> = Vec::with_capacity(p.len() + n.len());
    let (mut i, mut k) = (0usize, 0usize);
    while i < p.len() && k < n.len() {
        let (pc, pv) = p[i];
        let (nc, nv) = n[k];
        match pc.cmp(&nc) {
            std::cmp::Ordering::Less => {
                out.push((pc, i64::try_from(coef_p * pv as i128).ok()?));
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push((nc, i64::try_from(coef_n * nv as i128).ok()?));
                k += 1;
            }
            std::cmp::Ordering::Equal => {
                let v = coef_p * pv as i128 + coef_n * nv as i128;
                if v != 0 {
                    out.push((pc, i64::try_from(v).ok()?));
                }
                i += 1;
                k += 1;
            }
        }
    }
    while i < p.len() {
        let (pc, pv) = p[i];
        out.push((pc, i64::try_from(coef_p * pv as i128).ok()?));
        i += 1;
    }
    while k < n.len() {
        let (nc, nv) = n[k];
        out.push((nc, i64::try_from(coef_n * nv as i128).ok()?));
        k += 1;
    }
    Some(out)
}

/// Merge two sorted sparse u64 rows as `coef_p·p + coef_n·n`. All inputs are
/// non-negative (`coef_p,coef_n > 0`), so the result is non-negative; returns
/// `None` if any entry overflows u64.
fn combine_sparse_right(
    p: &[(u32, u64)],
    n: &[(u32, u64)],
    coef_p: i128,
    coef_n: i128,
) -> Option<Vec<(u32, u64)>> {
    let mut out: Vec<(u32, u64)> = Vec::with_capacity(p.len() + n.len());
    let (mut i, mut k) = (0usize, 0usize);
    while i < p.len() && k < n.len() {
        let (pc, pv) = p[i];
        let (nc, nv) = n[k];
        match pc.cmp(&nc) {
            std::cmp::Ordering::Less => {
                out.push((pc, u64::try_from(coef_p * pv as i128).ok()?));
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push((nc, u64::try_from(coef_n * nv as i128).ok()?));
                k += 1;
            }
            std::cmp::Ordering::Equal => {
                let v = coef_p * pv as i128 + coef_n * nv as i128;
                if v != 0 {
                    out.push((pc, u64::try_from(v).ok()?));
                }
                i += 1;
                k += 1;
            }
        }
    }
    while i < p.len() {
        let (pc, pv) = p[i];
        out.push((pc, u64::try_from(coef_p * pv as i128).ok()?));
        i += 1;
    }
    while k < n.len() {
        let (nc, nv) = n[k];
        out.push((nc, u64::try_from(coef_n * nv as i128).ok()?));
        k += 1;
    }
    Some(out)
}

/// GCD of two non-negative integers (Euclidean algorithm).
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Divide a sparse row's elements by their common GCD to keep numbers small.
/// The GCD is taken over both halves (left coefficients and right weights);
/// since neither half ever has zero stored entries, this preserves all
/// indices and only scales the values.
fn reduce_sparse(left: &mut [(u32, i64)], right: &mut [(u32, u64)]) {
    let mut g = 0u64;
    for &(_, v) in left.iter() {
        g = gcd(g, v.unsigned_abs());
    }
    for &(_, v) in right.iter() {
        g = gcd(g, v);
    }
    if g > 1 {
        for (_, v) in left.iter_mut() {
            *v /= g as i64;
        }
        for (_, v) in right.iter_mut() {
            *v /= g;
        }
    }
}

/// Remove duplicate invariants (identical weight vectors).
fn deduplicate_invariants(invariants: &mut Vec<PInvariant>) {
    invariants.sort_by(|a, b| a.weights.cmp(&b.weights));
    invariants.dedup_by(|a, b| a.weights == b.weights);
}

/// Structural upper bound for a single place from P-invariants.
///
/// Returns `min_{y: y_p > 0} floor(y.token_count / y_p)`.
/// Returns `None` if no invariant covers the place.
pub(crate) fn structural_place_bound(invariants: &[PInvariant], place: usize) -> Option<u64> {
    invariants
        .iter()
        .filter_map(|inv| {
            let w = inv.weight(place);
            (w > 0).then(|| inv.token_count / w)
        })
        .min()
}

/// Structural upper bound for the sum of tokens over a place set.
///
/// For a P-invariant y covering all places in S (y_p > 0 for all p ∈ S):
///   min(y_p) · Σ m(p) ≤ Σ y_p · m(p) ≤ y.token_count
///   ⟹ Σ m(p) ≤ token_count / min(y_p for p ∈ S)
///
/// Returns `None` if no invariant covers all places in `places`.
pub(crate) fn structural_set_bound(invariants: &[PInvariant], places: &[usize]) -> Option<u64> {
    if places.is_empty() {
        return Some(0);
    }

    invariants
        .iter()
        .filter(|inv| places.iter().all(|&p| inv.covers(p)))
        .map(|inv| {
            let min_weight = places.iter().map(|&p| inv.weight(p)).min().unwrap();
            inv.token_count / min_weight
        })
        .min()
}

/// Compute structural upper bounds for all places in a net.
///
/// Returns a vector indexed by place, with `Some(bound)` for places covered
/// by at least one P-invariant, `None` for uncovered places.
pub(crate) fn structural_place_bounds(net: &PetriNet) -> Vec<Option<u64>> {
    let invariants = compute_p_invariants(net);
    (0..net.num_places())
        .map(|p| structural_place_bound(&invariants, p))
        .collect()
}

/// Prove — purely structurally, with no LP solve — that some place is
/// *stably marked*: its token count equals its initial marking `m0[p]` in
/// **every** reachable marking. Returns `Some(p)` for the first such place,
/// or `None` if no place is provably constant.
///
/// This is the structural complement of the zero-incidence-row constant-place
/// test (`reduction::analysis::find_constant_places`) and of the LP/state-
/// equation pinning sweep (`lp_state_equation::lp_pinned_place`). It catches
/// constant places those two miss — coupled places pinned by a multi-place
/// P-invariant, and places kept constant because every transition that would
/// disturb them is structurally dead — with no LP solve, no size gate, and no
/// per-place time slice. It can only ever certify constancy (a sound
/// `StableMarking = TRUE` witness); an inconclusive net yields `None` and the
/// caller falls through to the LP/BMC/PDR/BFS engines unchanged.
///
/// Two independent sound mechanisms are tried:
///
/// 1. **P-invariant pinning** ([`p_invariant_pinned_place`]): a place whose
///    structural upper *and* lower bounds (both derived from semi-positive
///    P-invariants) coincide at `m0[p]`.
/// 2. **Dead-transition constancy** ([`dead_transition_constant_place`]): a
///    place whose net effect is zero across every transition that is not
///    provably dead (a transition can never fire if one of its input places is
///    bounded — by a P-invariant or by being structurally unproducible — below
///    the arc weight). Dead transitions never fire, so they cannot disturb the
///    place.
pub(crate) fn structural_stable_place(net: &PetriNet) -> Option<usize> {
    if net.num_places() == 0 {
        return None;
    }
    let invariants = compute_p_invariants(net);
    p_invariant_pinned_place(net, &invariants)
        .or_else(|| dead_transition_constant_place(net, &invariants))
}

/// P-invariant pinning: a place whose P-invariant-derived upper and lower
/// bounds both equal `m0[p]`, forcing `m[p] = m0[p]` in every reachable
/// marking. See [`structural_stable_place`].
///
/// ## Soundness
///
/// Every [`PInvariant`] `y` satisfies `y·m = y·m0 = c` for all reachable `m`
/// (truncation only drops invariants, never produces a spurious one). For a
/// place `p`:
///
/// * **Upper bound.** `structural_place_bound(p) = UB[p]` is a valid bound
///   (`m[p] ≤ UB[p]` for all reachable `m`).
/// * **Lower bound.** For an invariant `y` with `y[p] > 0` whose *other*
///   support places all have a finite upper bound,
///   `y[p]·m[p] = c − Σ_{q≠p} y[q]·m[q] ≥ c − Σ_{q≠p} y[q]·UB[q]`,
///   so `m[p] ≥ ⌈(c − Σ_{q≠p} y[q]·UB[q]) / y[p]⌉ =: LB`.
///
/// Because `m0` is reachable, `m0[p] ≤ UB[p]` and `LB ≤ m0[p]` always hold;
/// when `UB[p] == m0[p]` and some invariant drives `LB == m0[p]` the bounds
/// coincide at `m0[p]`, forcing `m[p] = m0[p]`. All integer arithmetic is in
/// `i128`/`u128`; on any overflow the contributing invariant is conservatively
/// skipped, which can only weaken the proof, never make it unsound.
fn p_invariant_pinned_place(net: &PetriNet, invariants: &[PInvariant]) -> Option<usize> {
    let np = net.num_places();
    if invariants.is_empty() {
        return None;
    }

    // Per-place upper bounds (None = uncovered = unbounded for our purposes).
    let upper: Vec<Option<u64>> = (0..np)
        .map(|p| structural_place_bound(invariants, p))
        .collect();

    for p in 0..np {
        // Need a tight upper bound m[p] ≤ UB[p] = m0[p].
        let m0 = net.initial_marking[p];
        if upper[p] != Some(m0) {
            continue;
        }

        // Need an invariant whose lower bound on m[p] also reaches m0[p].
        let lower_reaches_m0 = invariants.iter().any(|inv| {
            let yp = inv.weight(p);
            if yp == 0 {
                return false;
            }
            // num = c − Σ_{q≠p} y[q]·UB[q]; skip if any other support place is
            // unbounded (no finite lower bound derivable from this invariant).
            let mut sum_other: u128 = 0;
            let mut unbounded = false;
            for (q, w) in inv.support() {
                if q == p {
                    continue;
                }
                let Some(ub_q) = upper[q] else {
                    unbounded = true;
                    break;
                };
                let term = (w as u128).checked_mul(ub_q as u128);
                match term.and_then(|t| sum_other.checked_add(t)) {
                    Some(v) => sum_other = v,
                    None => {
                        unbounded = true;
                        break;
                    }
                }
            }
            if unbounded {
                return false;
            }
            // num in i128; c = inv.token_count.
            let num = (inv.token_count as i128) - (sum_other as i128);
            if num <= 0 {
                // Lower bound ≤ 0 ⇒ cannot reach a positive m0; for m0 == 0 the
                // upper bound m[p] ≤ 0 already forces constancy.
                return m0 == 0;
            }
            // lb = ⌈num / y[p]⌉ with positive integers.
            let yp = yp as i128;
            let lb = (num + yp - 1) / yp;
            lb >= m0 as i128
        });

        if lower_reaches_m0 {
            return Some(p);
        }
    }

    None
}

/// Dead-transition constancy: a place whose net effect is zero across every
/// transition not provably dead. See [`structural_stable_place`].
///
/// ## Soundness
///
/// A per-place upper bound `ub[p]` is maintained, seeded from P-invariants
/// (`structural_place_bound`, sound) and `u64::MAX` (≡ ∞) for uncovered
/// places. A monotone least-fixpoint then tightens it:
///
/// * `dead[t]` is set when some input arc `(p, w)` has `ub[p] < w`: place `p`
///   can never hold `w` tokens, so `t` can never be enabled — it never fires.
/// * `ub[p]` is set to `0` when `m0[p] == 0` and every transition with an
///   output arc to `p` is already `dead`: with no live producer and an empty
///   start, `p` can only ever lose tokens, so it stays `0`.
///
/// Both rules only ever *add* sound facts (a place truly cannot exceed a sound
/// bound; a transition with an unsatisfiable input truly never fires; an empty
/// place with only dead producers truly stays empty), so the fixpoint is
/// sound. At convergence, a place `q` whose token count is unchanged by every
/// non-dead transition (`in_weight(q,t) == out_weight(q,t)` for all live `t`)
/// is constant: only live transitions can fire, and none of them disturbs `q`.
fn dead_transition_constant_place(net: &PetriNet, invariants: &[PInvariant]) -> Option<usize> {
    let np = net.num_places();
    let nt = net.num_transitions();
    if nt == 0 {
        // No transitions ⇒ every place is trivially constant (handled by the
        // caller's zero-row test, but return the first place for completeness).
        return (np > 0).then_some(0);
    }

    // ub[p]: sound upper bound on m[p]. u64::MAX is our ∞ sentinel.
    let mut ub: Vec<u64> = (0..np)
        .map(|p| structural_place_bound(invariants, p).unwrap_or(u64::MAX))
        .collect();
    // Tighten with the initial marking for places no transition produces at all
    // (m0[p] == 0 and no output arc to p ⇒ p ≡ 0): seeds the empty-propagation.
    let mut has_producer = vec![false; np];
    for t in &net.transitions {
        for arc in &t.outputs {
            has_producer[arc.place.0 as usize] = true;
        }
    }
    for p in 0..np {
        if net.initial_marking[p] == 0 && !has_producer[p] {
            ub[p] = 0;
        }
    }

    let mut dead = vec![false; nt];
    // Monotone fixpoint: re-derive dead transitions and empty places until
    // nothing changes. Bounded by nt + np iterations (each round flips at least
    // one flag, or we stop).
    loop {
        let mut changed = false;

        // (1) A transition with an input it can never satisfy is dead.
        for (ti, t) in net.transitions.iter().enumerate() {
            if dead[ti] {
                continue;
            }
            let unsatisfiable = t
                .inputs
                .iter()
                .any(|arc| ub[arc.place.0 as usize] < arc.weight);
            if unsatisfiable {
                dead[ti] = true;
                changed = true;
            }
        }

        // (2) An empty-start place whose every producer is now dead stays empty.
        for p in 0..np {
            if ub[p] == 0 || net.initial_marking[p] != 0 {
                continue;
            }
            let all_producers_dead = net.transitions.iter().enumerate().all(|(ti, t)| {
                // A transition produces p only if its net effect on p is > 0.
                let out: u64 = t
                    .outputs
                    .iter()
                    .filter(|a| a.place.0 as usize == p)
                    .map(|a| a.weight)
                    .sum();
                let inn: u64 = t
                    .inputs
                    .iter()
                    .filter(|a| a.place.0 as usize == p)
                    .map(|a| a.weight)
                    .sum();
                out <= inn || dead[ti]
            });
            if all_producers_dead {
                ub[p] = 0;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // A place unchanged by every live transition is constant.
    for p in 0..np {
        let disturbed = net.transitions.iter().enumerate().any(|(ti, t)| {
            if dead[ti] {
                return false;
            }
            let out: i64 = t
                .outputs
                .iter()
                .filter(|a| a.place.0 as usize == p)
                .map(|a| a.weight as i64)
                .sum();
            let inn: i64 = t
                .inputs
                .iter()
                .filter(|a| a.place.0 as usize == p)
                .map(|a| a.weight as i64)
                .sum();
            out != inn
        });
        if !disturbed {
            // Every live transition has zero net effect on p ⇒ p ≡ m0[p].
            // (When nt > 0 and ALL transitions are dead, every place qualifies;
            // that is still a sound StableMarking witness — the net's reachable
            // set is the singleton {m0}.)
            return Some(p);
        }
    }

    None
}

// ── Implied place detection ──────────────────────────────────────────

/// An implied place: its token count is fully determined by a P-invariant
/// and other non-implied (kept) places. Excluding implied places from
/// the packed hash key reduces per-state memory during BFS exploration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImpliedPlace {
    /// Index of the implied place in the net's place vector.
    pub place: usize,
    /// How to reconstruct this place's token count from kept places.
    pub reconstruction: ImpliedPlaceReconstruction,
}

/// Reconstruction equation for an implied place.
///
/// `m(place) = (constant - sum(weight_i * m(kept_i))) / divisor`
///
/// Division is exact for all reachable markings (guaranteed by the
/// P-invariant property y^T · m = constant).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImpliedPlaceReconstruction {
    /// Conserved quantity: y^T · m₀.
    pub constant: u64,
    /// Invariant weight of the implied place (always > 0).
    pub divisor: u64,
    /// (place_index, invariant_weight) pairs for kept places in the support.
    pub terms: Vec<(usize, u64)>,
}

/// Find implied places using a greedy selection over P-invariants.
///
/// For each invariant (processed by ascending support size), selects at most
/// one place to exclude. Guarantees no chained reconstruction: every term
/// in a reconstruction references only kept (non-excluded) places.
///
/// Prefers weight-1 places (exact division without remainder).
pub(crate) fn find_implied_places(invariants: &[PInvariant]) -> Vec<ImpliedPlace> {
    if invariants.is_empty() {
        return vec![];
    }

    // `removed`/`must_keep` are only ever indexed at support places, so size
    // them to one past the largest place index appearing in any support
    // (weights are now sparse — `weights.len()` is the support size, not the
    // place count).
    let num_places = invariants
        .iter()
        .flat_map(|inv| inv.weights.iter().map(|&(p, _)| p as usize))
        .max()
        .map_or(0, |m| m + 1);

    // Sort invariants by ascending support size (tighter invariants first)
    let mut sorted_indices: Vec<usize> = (0..invariants.len()).collect();
    sorted_indices.sort_by_key(|&i| invariants[i].support_len());

    let mut removed = vec![false; num_places];
    let mut must_keep = vec![false; num_places];
    let mut result = Vec::new();

    for &inv_idx in &sorted_indices {
        let inv = &invariants[inv_idx];
        let support: Vec<usize> = inv.support().map(|(p, _)| p).collect();

        if support.len() < 2 {
            continue; // Need at least 2 support places for reconstruction
        }

        // Find best candidate to remove:
        // - Not already removed or marked must_keep
        // - All other support places are non-removed (independence guarantee)
        // - Prefer weight 1 (no division)
        let candidate = support
            .iter()
            .copied()
            .filter(|&p| !removed[p] && !must_keep[p])
            .filter(|&p| support.iter().all(|&q| q == p || !removed[q]))
            .min_by_key(|&p| {
                let w = inv.weight(p);
                if w == 1 {
                    0u64
                } else {
                    w
                }
            });

        let candidate_place = match candidate {
            Some(p) => p,
            None => continue,
        };

        let terms: Vec<(usize, u64)> = support
            .iter()
            .filter(|&&p| p != candidate_place)
            .map(|&p| (p, inv.weight(p)))
            .collect();

        removed[candidate_place] = true;
        for &p in &support {
            if p != candidate_place {
                must_keep[p] = true;
            }
        }

        result.push(ImpliedPlace {
            place: candidate_place,
            reconstruction: ImpliedPlaceReconstruction {
                constant: inv.token_count,
                divisor: inv.weight(candidate_place),
                terms,
            },
        });
    }

    result.sort_by_key(|ip| ip.place);
    result
}

// ── T-semiflow (T-invariant) computation ────────────────────────────

/// A semi-positive T-semiflow: x ≥ 0 with C · x = 0.
///
/// Transitions covered by at least one T-semiflow participate in
/// repeatable firing sequences. Uncovered transitions can fire at most
/// finitely many times in a bounded net — disproving L4-liveness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TSemiflow {
    /// Nonzero transition weights as `(transition_index, weight)`, sorted
    /// ascending, all > 0. Sparse for the same wide-net reason as
    /// [`PInvariant::weights`].
    pub weights: Vec<(u32, u64)>,
}

/// Result of T-semiflow computation: semiflows plus a completeness flag.
pub(crate) struct TSemiflowResult {
    /// Discovered T-semiflows (may be a subset if Farkas was truncated).
    pub semiflows: Vec<TSemiflow>,
    /// Whether the Farkas algorithm completed without hitting MAX_ROWS.
    /// If `false`, the semiflow list may be incomplete — callers must NOT
    /// conclude non-coverage from a truncated result (soundness issue).
    pub complete: bool,
}

/// Compute semi-positive T-semiflows via Farkas on C^T.
///
/// T-semiflows satisfy C · x = 0 ⟺ x^T · C^T = 0. This is structurally
/// identical to P-invariant computation on the transposed matrix. We reuse
/// `farkas_elimination` with a dummy marking (the token_count field is
/// meaningless for T-semiflows and is discarded).
///
/// Returns `TSemiflowResult` with a `complete` flag. Callers using the
/// result to prove non-coverage (e.g., structural non-liveness) MUST check
/// `complete == true` before concluding — an incomplete result may miss
/// semiflows that would cover the transition.
pub(crate) fn compute_t_semiflows(net: &PetriNet) -> TSemiflowResult {
    let np = net.num_places();
    let nt = net.num_transitions();

    if nt == 0 {
        return TSemiflowResult {
            semiflows: vec![],
            complete: true,
        };
    }

    // No places: C is empty, every transition vector is a semiflow.
    if np == 0 {
        return TSemiflowResult {
            semiflows: (0..nt)
                .map(|t| TSemiflow {
                    weights: vec![(t as u32, 1)],
                })
                .collect(),
            complete: true,
        };
    }

    // Sparse C^T: one row per transition (its place-column incidence), then
    // eliminate place columns. Reuses the same sparse Farkas core; the dummy
    // marking is discarded (token_count is meaningless for T-semiflows).
    let rows = sparse_transposed_incidence_rows(net, nt);
    let dummy_marking = vec![0u64; nt];
    let (p_invs, complete) = farkas_elimination_sparse(rows, np, &dummy_marking);
    TSemiflowResult {
        semiflows: p_invs
            .into_iter()
            .map(|inv| TSemiflow {
                weights: inv.weights,
            })
            .collect(),
        complete,
    }
}

/// Build the transition-indexed sparse rows of the transposed incidence
/// matrix: row `t` holds the nonzero `(place, C[p][t])` entries. Sparse
/// analogue of the dense `C^T` — O(arcs) storage, never O(transitions ×
/// places).
fn sparse_transposed_incidence_rows(net: &PetriNet, nt: usize) -> Vec<Vec<(u32, i64)>> {
    let mut acc: Vec<std::collections::BTreeMap<u32, i64>> =
        (0..nt).map(|_| std::collections::BTreeMap::new()).collect();
    for (tidx, trans) in net.transitions.iter().enumerate() {
        for arc in &trans.inputs {
            *acc[tidx].entry(arc.place.0).or_insert(0) -= arc.weight as i64;
        }
        for arc in &trans.outputs {
            *acc[tidx].entry(arc.place.0).or_insert(0) += arc.weight as i64;
        }
    }
    acc.into_iter()
        .map(|m| m.into_iter().filter(|&(_, v)| v != 0).collect())
        .collect()
}

/// Check if every transition is covered by at least one T-semiflow.
pub(crate) fn all_transitions_covered(semiflows: &[TSemiflow], num_transitions: usize) -> bool {
    // Collect the union of all semiflow supports once (each `weights` is the
    // sparse, sorted, all-positive support), then verify full coverage.
    let mut covered = vec![false; num_transitions];
    for sf in semiflows {
        for &(t, _) in &sf.weights {
            if let Some(slot) = covered.get_mut(t as usize) {
                *slot = true;
            }
        }
    }
    covered.into_iter().all(|c| c)
}

#[cfg(test)]
#[path = "invariant_tests.rs"]
mod tests;
