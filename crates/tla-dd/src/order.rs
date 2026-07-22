// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Static place-ordering (BDD variable ordering) for the DD reachability
//! engine.
//!
//! A BDD's size — and therefore the feasibility of the symbolic
//! reachability fixpoint within a wall-clock budget — is dominated by the
//! variable order. This crate identifies one Petri *place* with one BDD
//! *level* (the saturation level model, see [`crate::saturation`]), so the
//! place order **is** the variable order. The MVP allocated variables in
//! PNML index order, which is arbitrary; on the mid-size MCC nets that is
//! the difference between converging in milliseconds and blowing the
//! budget (the StateSpace/UpperBounds pool TY targets).
//!
//! We compute a static order with the **FORCE** heuristic (Aloul, Markov,
//! Sakallah, "FORCE: A Fast and Easy-to-Implement Variable-Ordering
//! Heuristic", GLSVLSI 2003): model each transition as a hyperedge over the
//! places in its support, then iteratively pull each place toward the
//! centre of gravity of the transitions touching it. This clusters places
//! that fire together into a contiguous, narrow band of levels, which is
//! exactly what shrinks the per-level saturation relations and the
//! reachable-set BDD.
//!
//! # Soundness
//!
//! Ordering is applied by **permuting places** before the BDD is built and
//! is therefore answer-preserving by construction:
//!
//! - `state_count`, `edge_count`, `max_token_in_place`, and
//!   `max_token_sum` are all defined as functions of the *set* of reachable
//!   markings and the reachability graph; relabeling places is a bijection
//!   on markings that preserves the firing relation, so every one of these
//!   metrics is invariant.
//! - An UpperBounds query `max_{M∈R} Σ_p coeff[p]·m[p]` is invariant when
//!   the coefficient vector is permuted with the *same* permutation
//!   ([`permute_query`]): the per-term products are merely reindexed.
//!
//! So a permuted run and an unpermuted run compute the *same* numbers — the
//! order only changes BDD size (hence speed/feasibility). This is asserted
//! exhaustively against the exhaustive-BFS oracle in the crate's
//! differential tests (`order::tests` plus the dispatch differentials),
//! with zero tolerated disagreements.
//!
//! The heuristic is also **self-guarding**: it keeps the FORCE order only
//! when it strictly improves the total transition *span* over the identity
//! order (a sound proxy for BDD locality), so a net whose PNML order is
//! already good is never made worse. [`force_place_order_seeded`] extends
//! the candidate set with a structural seed (e.g. the NUPN unit-hierarchy
//! block order) under the same span guard — the seed is performance-only
//! and can never displace a better order.

use crate::{DdNetSpec, DdTransition};

/// Number of FORCE refinement sweeps. Each sweep is `O(Σ_t |support(t)|)`;
/// the order converges quickly, so a small fixed count is enough and keeps
/// the precompute negligible relative to the BDD fixpoint it accelerates.
const FORCE_SWEEPS: usize = 12;

/// Compute a static place order for `spec` via the FORCE heuristic.
///
/// Returns a permutation `order` of `0..num_places` where `order[level]` is
/// the original place index that should occupy BDD level `level` (level `0`
/// = innermost / bottom of the diagram). The returned order is the identity
/// when ordering cannot help (no transitions, `<= 2` places) or when FORCE
/// fails to improve the transition-span proxy over the identity order.
#[must_use]
pub fn force_place_order(spec: &DdNetSpec) -> Vec<usize> {
    force_place_order_seeded(spec, None)
}

/// [`force_place_order`] with an optional structural *seed* order (e.g. the
/// NUPN unit-hierarchy block order of an MCC net — nested units of mutually
/// exclusive places are exactly the locality structure a good DD order
/// wants).
///
/// The seed extends the candidate set; selection stays span-guarded:
///
/// 1. identity (the PNML order),
/// 2. FORCE refined from the identity (today's behavior),
/// 3. the seed itself (a structural block order can beat its own FORCE
///    refinement — observed on TokenRing),
/// 4. FORCE refined from the seed.
///
/// The returned order is the candidate with the minimal total transition
/// span, ties broken in the listed preference order — so with `seed = None`
/// (or an invalid seed) this is **bit-identical** to [`force_place_order`],
/// and a seed can never make the chosen order worse than today's choice.
///
/// # Soundness
///
/// Performance-only by construction: *any* permutation of `0..n` is
/// answer-preserving (see the module soundness note — callers apply it via
/// [`permute_spec`]/[`permute_query`]). A seed that is not a permutation of
/// `0..n` is ignored rather than trusted.
#[must_use]
pub fn force_place_order_seeded(spec: &DdNetSpec, seed: Option<&[usize]>) -> Vec<usize> {
    let n = spec.bounds.len();
    let identity: Vec<usize> = (0..n).collect();
    if n <= 2 || spec.transitions.is_empty() {
        return identity;
    }

    // Per-transition support (place indices touched in pre or post).
    let supports: Vec<Vec<usize>> = spec
        .transitions
        .iter()
        .map(support_of)
        .filter(|s| s.len() >= 2) // single-place / empty events impose no order constraint
        .collect();
    if supports.is_empty() {
        return identity;
    }

    // Candidates in tie-break preference order (earlier wins on equal span).
    let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(4);
    candidates.push(force_refine(n, &supports, &identity));
    if let Some(seed) = seed {
        if is_permutation(seed, n) {
            candidates.push(seed.to_vec());
            candidates.push(force_refine(n, &supports, seed));
        }
    }

    // Span-guarded argmin. The identity is the incumbent: a candidate must
    // *strictly* reduce the total transition span to displace it (never make
    // a net with an already-good PNML order worse), and later candidates
    // must strictly beat earlier ones.
    let mut best = identity;
    let mut best_span = total_span(spec, &best);
    for candidate in candidates {
        let span = total_span(spec, &candidate);
        if span < best_span {
            best = candidate;
            best_span = span;
        }
    }
    best
}

/// `true` iff `seed` is a permutation of `0..n`.
fn is_permutation(seed: &[usize], n: usize) -> bool {
    if seed.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &p in seed {
        if p >= n || seen[p] {
            return false;
        }
        seen[p] = true;
    }
    true
}

/// Run the FORCE centre-of-gravity sweeps starting from `init_order` (a
/// permutation of `0..n`; `init_order[rank] = place`). Returns the settled
/// order — *not* span-guarded; the caller compares candidates.
fn force_refine(n: usize, supports: &[Vec<usize>], init_order: &[usize]) -> Vec<usize> {
    // FORCE: positions start at the seed ranks; each sweep recomputes
    // every place's position as the mean centre-of-gravity of the
    // transitions touching it, then re-ranks.
    let mut pos: Vec<f64> = vec![0.0; n];
    for (rank, &p) in init_order.iter().enumerate() {
        pos[p] = rank as f64;
    }
    for _ in 0..FORCE_SWEEPS {
        let mut cog_sum = vec![0.0f64; n];
        let mut cog_cnt = vec![0u32; n];
        for support in supports {
            let mut s = 0.0f64;
            for &p in support {
                s += pos[p];
            }
            let cog = s / support.len() as f64;
            for &p in support {
                cog_sum[p] += cog;
                cog_cnt[p] += 1;
            }
        }
        let mut new_pos = pos.clone();
        for p in 0..n {
            if cog_cnt[p] > 0 {
                new_pos[p] = cog_sum[p] / cog_cnt[p] as f64;
            }
            // Untouched places keep their previous position (stable).
        }
        // Re-rank: order places by new position, tie-break by original
        // index for determinism, then snap positions back to integer ranks
        // so the next sweep operates on a stable scale.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            new_pos[a]
                .partial_cmp(&new_pos[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for (rank, &p) in idx.iter().enumerate() {
            pos[p] = rank as f64;
        }
    }

    // Final order = places sorted by their settled rank.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        pos[a]
            .partial_cmp(&pos[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// Places touched by a transition (non-zero pre or post weight), ascending.
fn support_of(t: &DdTransition) -> Vec<usize> {
    let mut s = Vec::new();
    for p in 0..t.pre.len() {
        if t.pre[p] != 0 || t.post[p] != 0 {
            s.push(p);
        }
    }
    s
}

/// Sum over transitions of `Top(t) - Bot(t)` in the level coordinates
/// induced by `order` (`inv[orig] = level`). Lower is better for saturation
/// locality. Empty/singleton-support transitions contribute zero.
fn total_span(spec: &DdNetSpec, order: &[usize]) -> u64 {
    let inv = invert_order(order);
    let mut total: u64 = 0;
    for t in &spec.transitions {
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        let mut touched = false;
        for p in 0..t.pre.len() {
            if t.pre[p] != 0 || t.post[p] != 0 {
                let lvl = inv[p];
                lo = lo.min(lvl);
                hi = hi.max(lvl);
                touched = true;
            }
        }
        if touched {
            total += (hi - lo) as u64;
        }
    }
    total
}

/// Invert a level→place order into a place→level map (`inv[orig] = level`).
#[must_use]
pub fn invert_order(order: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; order.len()];
    for (level, &orig) in order.iter().enumerate() {
        inv[orig] = level;
    }
    inv
}

// ---------------------------------------------------------------------------
// Bit-level (slot) ordering for binary-encoded nets.
// ---------------------------------------------------------------------------
//
// `force_place_order` orders **places** — one place = one variable band. The
// binary encoding lays each high-bound place out as a contiguous band of its
// `k` bits; bits of *different* places are never interleaved. For coupled
// high-bound places (e.g. conserved cells `a + b = const`, a shuttle, a
// chain) grouping each place's whole band contiguously can blow up the BDD,
// while interleaving correlated places' bits bit-for-bit shrinks it
// dramatically. This is the standard lever DD tools pull on such nets.
//
// A "slot" is the unit the layout places contiguously:
//   * a **unary** place contributes ONE atomic slot (its whole one-hot band —
//     a one-hot group is mutually-exclusive indicators, not coordinates, so
//     splitting it across the order is meaningless and would only disturb the
//     well-tuned low-bound nets); and
//   * a **binary** place contributes ONE slot **per bit** (LSB first), so its
//     bits can be interleaved with the bits of correlated places.
//
// A bit-slot order is a permutation of all slots → BDD-level rank. The
// identity order (places in spec order, slots within a place in index order)
// reproduces today's layout exactly. Selection is span-guarded at the bit
// level: a candidate displaces the identity only if it *strictly* reduces the
// total bit-level transition span, so a net whose place-banded layout is
// already good is never made worse.
//
// # Soundness
//
// The chosen order only changes which BDD *level* each `(place, bit, side)`
// variable occupies — a relabeling of variables, which a BDD computes the
// same function under (answer-preserving by construction; see the module
// note). `var_current[p][i]` / `var_next[p][i]` and the rename tables still
// point at the correct cur/next-paired variable functions, so every metric
// and the saturation kernel (which indexes per-place variables by content,
// not by raw level) are invariant. The differential battery re-confirms it.

/// One unit of the bit-level layout: a place plus, for binary places, which
/// bit of that place this slot is (`None` = the place's whole one-hot band,
/// used for unary places).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitSlot {
    /// Original place index this slot belongs to.
    pub place: usize,
    /// `Some(i)` = bit `i` (LSB first) of a binary place; `None` = the whole
    /// one-hot band of a unary place.
    pub bit: Option<u32>,
}

/// Build the identity (today's) slot list for the given per-place slot
/// counts: places in spec order, and within a binary place its bits in
/// LSB-first index order; one atomic slot for each unary place.
///
/// `bit_counts[p] == Some(k)` ⇒ place `p` is binary with `k` bits;
/// `bit_counts[p] == None` ⇒ place `p` is unary (one atomic slot).
#[must_use]
fn identity_slots(bit_counts: &[Option<u32>]) -> Vec<BitSlot> {
    let mut slots = Vec::new();
    for (p, &bc) in bit_counts.iter().enumerate() {
        match bc {
            None => slots.push(BitSlot {
                place: p,
                bit: None,
            }),
            Some(k) => {
                for i in 0..k {
                    slots.push(BitSlot {
                        place: p,
                        bit: Some(i),
                    });
                }
            }
        }
    }
    slots
}

/// Compute a static **bit-level** slot order for `spec`.
///
/// Returns a permutation of the net's slots (one per unary place, one per bit
/// of each binary place) giving the order in which the binary var layout
/// should lay slots down across BDD levels. The result is the identity slot
/// order (= today's place-banded layout) unless a bit-level candidate
/// strictly reduces the total bit-level transition span.
///
/// `bit_counts[p]` selects the per-place encoding granularity: `Some(k)` for
/// a binary place with `k` bits, `None` for a unary place (one atomic slot).
/// This MUST agree with the encoding the BDD layout actually uses, or the
/// returned order will not be a valid layout permutation.
///
/// # Soundness
///
/// Performance-only: any permutation of the slots is answer-preserving (it
/// only renumbers BDD variables). The span guard guarantees we never adopt a
/// worse-than-identity order.
#[must_use]
pub fn force_bit_slot_order(spec: &DdNetSpec, bit_counts: &[Option<u32>]) -> Vec<BitSlot> {
    let n = spec.bounds.len();
    debug_assert_eq!(bit_counts.len(), n);
    let identity = identity_slots(bit_counts);
    let num_slots = identity.len();
    // Nothing to gain if every place is a single atomic slot (all unary) or
    // there are too few slots / no transitions to constrain an order.
    let any_binary = bit_counts.iter().any(|bc| bc.is_some());
    if !any_binary || num_slots <= 2 || spec.transitions.is_empty() {
        return identity;
    }

    // Map (place, slot-within-place) -> global slot id under the identity
    // layout, so a transition's per-place supports translate to slot sets.
    // `slot_base[p]` = id of place p's first slot; `slot_len[p]` = #slots.
    let mut slot_base = vec![0usize; n];
    let mut slot_len = vec![0usize; n];
    {
        let mut id = 0usize;
        for (p, &bc) in bit_counts.iter().enumerate() {
            slot_base[p] = id;
            let len = match bc {
                None => 1,
                Some(k) => k as usize,
            };
            slot_len[p] = len;
            id += len;
        }
    }

    // Per-transition slot support: every slot of every place the transition
    // touches. The bit-blasted relation of a binary place couples *all* its
    // bits, so the whole place's slot range participates as one hyperedge.
    let supports: Vec<Vec<usize>> = spec
        .transitions
        .iter()
        .map(|t| {
            let mut s = Vec::new();
            for p in 0..t.pre.len() {
                if t.pre[p] != 0 || t.post[p] != 0 {
                    for off in 0..slot_len[p] {
                        s.push(slot_base[p] + off);
                    }
                }
            }
            s
        })
        .filter(|s| s.len() >= 2)
        .collect();
    if supports.is_empty() {
        return identity;
    }

    // Candidate orders in tie-break preference order. Each is a permutation
    // of `0..num_slots` (global slot ids); we materialise the chosen one back
    // into `Vec<BitSlot>` at the end.
    let id_perm: Vec<usize> = (0..num_slots).collect();
    let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(3);
    // (a) FORCE over the per-bit slots, refined from the identity layout.
    candidates.push(force_refine(num_slots, &supports, &id_perm));
    // (b) Correlated-place bit-for-bit interleave: a structural seed that
    //     interleaves the bits of binary places that co-occur in transitions,
    //     then FORCE-refined. Generated under the same span guard, so it can
    //     never make a net worse.
    if let Some(seed) = interleave_seed(n, bit_counts, &slot_base, &slot_len, &spec.transitions) {
        candidates.push(seed.clone());
        candidates.push(force_refine(num_slots, &supports, &seed));
    }
    // (c) P-invariant-correlated interleave: clusters only the value-correlated
    //     (token-conserving-exchange) places, a sharper grouping than (b) on
    //     nets whose transition graph collapses into one component. Same span
    //     guard, so it too can never make a net worse.
    if let Some(seed) =
        pinvariant_interleave_seed(n, bit_counts, &slot_base, &slot_len, &spec.transitions)
    {
        candidates.push(seed.clone());
        candidates.push(force_refine(num_slots, &supports, &seed));
    }

    // Span-guarded argmin over global-slot permutations. Identity is the
    // incumbent; a candidate must STRICTLY reduce the bit-level span to win.
    let mut best = id_perm;
    let mut best_span = slot_total_span(&supports, &best, num_slots);
    for candidate in candidates {
        let span = slot_total_span(&supports, &candidate, num_slots);
        if span < best_span {
            best = candidate;
            best_span = span;
        }
    }

    // Materialise the winning global-slot permutation back into BitSlots.
    best.into_iter().map(|gid| identity[gid]).collect()
}

/// Candidate **bit-level** slot orders for a binary-encoded net, identity
/// first, in tie-break preference order, de-duplicated.
///
/// Unlike [`force_bit_slot_order`] (which picks one order under the cheap
/// transition-span proxy), this returns *all* the structurally-motivated
/// candidates so a caller can pick among them under a stronger proxy — e.g.
/// **measured reachable-BDD node counts**, which is the proxy that actually
/// captures the bit-interleave win on value-correlated high-bound places
/// (the transition-span proxy is blind to it: a transition touching every
/// bit of two coupled places has the same span under any order, yet
/// interleaving its bits can shrink the reachable BDD by an order of
/// magnitude — see the `diag_bit_order_shrink_probe` measurements).
///
/// The candidates are:
/// 1. identity (today's place-banded layout) — always first, the incumbent;
/// 2. FORCE refined over the per-bit slots;
/// 3. the correlated-place bit-for-bit interleave seed;
/// 4. FORCE refined from that seed.
///
/// Every entry is a valid permutation of the net's slots (so all are
/// answer-preserving). Returns just `[identity]` when there is no binary
/// place / nothing to interleave.
#[must_use]
pub fn bit_order_candidates(spec: &DdNetSpec, bit_counts: &[Option<u32>]) -> Vec<Vec<BitSlot>> {
    let n = spec.bounds.len();
    debug_assert_eq!(bit_counts.len(), n);
    let identity = identity_slots(bit_counts);
    let num_slots = identity.len();
    let any_binary = bit_counts.iter().any(|bc| bc.is_some());
    if !any_binary || num_slots <= 2 || spec.transitions.is_empty() {
        return vec![identity];
    }

    let mut slot_base = vec![0usize; n];
    let mut slot_len = vec![0usize; n];
    {
        let mut id = 0usize;
        for (p, &bc) in bit_counts.iter().enumerate() {
            slot_base[p] = id;
            let len = bc.map_or(1, |k| k as usize);
            slot_len[p] = len;
            id += len;
        }
    }
    let supports: Vec<Vec<usize>> = spec
        .transitions
        .iter()
        .map(|t| {
            let mut s = Vec::new();
            for p in 0..t.pre.len() {
                if t.pre[p] != 0 || t.post[p] != 0 {
                    for off in 0..slot_len[p] {
                        s.push(slot_base[p] + off);
                    }
                }
            }
            s
        })
        .filter(|s| s.len() >= 2)
        .collect();

    let id_perm: Vec<usize> = (0..num_slots).collect();
    // Materialise a global-slot permutation into BitSlots.
    let to_slots = |perm: &[usize]| -> Vec<BitSlot> { perm.iter().map(|&g| identity[g]).collect() };

    let mut out: Vec<Vec<BitSlot>> = vec![identity.clone()];
    let push_unique = |cand: Vec<BitSlot>, out: &mut Vec<Vec<BitSlot>>| {
        if !out.contains(&cand) {
            out.push(cand);
        }
    };
    if !supports.is_empty() {
        push_unique(
            to_slots(&force_refine(num_slots, &supports, &id_perm)),
            &mut out,
        );
    }
    if let Some(seed) = interleave_seed(n, bit_counts, &slot_base, &slot_len, &spec.transitions) {
        push_unique(to_slots(&seed), &mut out);
        if !supports.is_empty() {
            push_unique(
                to_slots(&force_refine(num_slots, &supports, &seed)),
                &mut out,
            );
        }
    }
    // P-invariant-correlated interleave (value-correlated clustering — see
    // `pinvariant_interleave_seed`). De-duplicated against the co-occurrence
    // seed: on a net where the two clusterings coincide (every shared
    // transition IS a conservative exchange) this adds nothing, no extra build.
    if let Some(seed) =
        pinvariant_interleave_seed(n, bit_counts, &slot_base, &slot_len, &spec.transitions)
    {
        push_unique(to_slots(&seed), &mut out);
        if !supports.is_empty() {
            push_unique(
                to_slots(&force_refine(num_slots, &supports, &seed)),
                &mut out,
            );
        }
    }
    out
}

/// Total bit-level transition span under a global-slot permutation
/// (`order[rank] = slot_id`). Sum over transitions of
/// `max_rank(support) - min_rank(support)`. Lower is better for locality.
fn slot_total_span(supports: &[Vec<usize>], order: &[usize], num_slots: usize) -> u64 {
    // rank[slot_id] = level.
    let mut rank = vec![0usize; num_slots];
    for (level, &sid) in order.iter().enumerate() {
        rank[sid] = level;
    }
    let mut total: u64 = 0;
    for support in supports {
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for &sid in support {
            let r = rank[sid];
            lo = lo.min(r);
            hi = hi.max(r);
        }
        total += (hi - lo) as u64;
    }
    total
}

/// Disjoint-set find with path-halving (shared by the interleave seeds).
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Union the disjoint sets of `a` and `b`.
fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// Emit a bit-level slot order from a place-clustering union-find.
///
/// Walks places in original order; when it reaches the first (lowest-index)
/// binary member of a cluster, it emits that whole cluster's bits
/// **bit-for-bit interleaved** (all LSBs of the cluster, then all bit-1s, …),
/// so a coupled invariant like `a + b = const` keeps each binary weight column
/// local — the carry chain that couples the places lives in one narrow band
/// instead of being smeared across two far-apart place bands. Unary places and
/// singleton binary clusters are emitted as their identity slot block. Returns
/// `None` when no cluster interleaves `>= 2` binary places (no interleave to
/// try) — so a `None` here means "this clustering offers nothing the identity
/// doesn't already have".
///
/// This is the shared materialiser for [`interleave_seed`] (clusters by raw
/// transition co-occurrence) and [`pinvariant_interleave_seed`] (clusters by
/// the sharper token-conservation / P-semiflow footprint). The two differ only
/// in *how* `parent` is built; the layout shape is identical, so both produce
/// answer-preserving permutations and both are span/node-guarded by the caller.
fn emit_interleaved_seed(
    n: usize,
    bit_counts: &[Option<u32>],
    slot_base: &[usize],
    slot_len: &[usize],
    parent: &mut [usize],
) -> Option<Vec<usize>> {
    // Cluster representative -> member binary places, in ascending place
    // order (the `for p in 0..n` walk inserts in order, and the emit pass
    // below re-walks places in order, so the layout is fully deterministic).
    let mut cluster_members: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for p in 0..n {
        if bit_counts[p].is_none() {
            continue; // unary: kept atomic in the emit pass below
        }
        let r = uf_find(parent, p);
        cluster_members.entry(r).or_default().push(p);
    }
    // Only worthwhile if some cluster interleaves >= 2 binary places.
    if !cluster_members.values().any(|m| m.len() >= 2) {
        return None;
    }

    // Emit pass: walk places in original order; when we reach the first
    // (lowest-index) binary member of a cluster, emit that whole cluster's
    // interleaved bits; emit unary places (and singleton binary places) as
    // their identity slot block in place order. Each place is emitted once.
    let mut emitted = vec![false; n];
    let mut out: Vec<usize> = Vec::new();
    for p in 0..n {
        if emitted[p] {
            continue;
        }
        match bit_counts[p] {
            None => {
                out.push(slot_base[p]); // unary: single atomic slot
                emitted[p] = true;
            }
            Some(_) => {
                let r = uf_find(parent, p);
                let members = &cluster_members[&r];
                if members.len() < 2 {
                    // Singleton binary place: emit its bits contiguously.
                    for off in 0..slot_len[p] {
                        out.push(slot_base[p] + off);
                    }
                    emitted[p] = true;
                } else {
                    // Interleave the cluster's binary places bit-for-bit.
                    let max_bits = members.iter().map(|&m| slot_len[m]).max().unwrap_or(0);
                    for bit in 0..max_bits {
                        for &m in members {
                            if bit < slot_len[m] {
                                out.push(slot_base[m] + bit);
                            }
                        }
                    }
                    for &m in members {
                        emitted[m] = true;
                    }
                }
            }
        }
    }
    debug_assert_eq!(out.len(), slot_len.iter().sum::<usize>());
    Some(out)
}

/// Build a structural "interleave correlated places' bits" seed.
///
/// Groups binary places into clusters of places that share a transition
/// (union-find over the transition supports), then for each cluster emits its
/// places' bits **bit-for-bit interleaved** (all LSBs, then all bit-1s, …) so
/// a coupled invariant like `a + b = const` keeps each weight column local.
/// Unary places and binary places that share no transition with another
/// binary place are emitted as their identity slot block. Returns `None` when
/// no two binary places are correlated (no interleave to try).
fn interleave_seed(
    n: usize,
    bit_counts: &[Option<u32>],
    slot_base: &[usize],
    slot_len: &[usize],
    transitions: &[DdTransition],
) -> Option<Vec<usize>> {
    // Union-find over places, unioning places that co-occur in a transition's
    // support. We only care about clusters that contain >= 2 binary places.
    let mut parent: Vec<usize> = (0..n).collect();
    for t in transitions {
        let support: Vec<usize> = (0..t.pre.len())
            .filter(|&p| t.pre[p] != 0 || t.post[p] != 0)
            .collect();
        for w in support.windows(2) {
            uf_union(&mut parent, w[0], w[1]);
        }
    }
    emit_interleaved_seed(n, bit_counts, slot_base, slot_len, &mut parent)
}

/// Build a **P-invariant-correlated** interleave seed.
///
/// This is the value-correlated ordering the StateSpace lane wants. Two places
/// are clustered iff some transition is a **token-conserving exchange** between
/// them — a transition `t` whose net effect `Δ[p] = post[p] − pre[p]` moves
/// tokens *out of* one place and *into* the other (`Δ[p]·Δ[q] < 0`). That is
/// exactly the local structural footprint of a shared P-semiflow (a place
/// invariant `Σ_p y[p]·m[p] = const`): along such a transition the tokens lost
/// by `p` are gained by `q`, so their counts are *value-correlated* (move in
/// lockstep / anti-lockstep). Binary-encoding those two correlated counters and
/// interleaving their bits bit-for-bit collapses the coupling carry chain into
/// one narrow BDD band — the canonical `a + b = const` shrink (see
/// `diag_bit_order_shrink_probe`).
///
/// # Why this is sharper than [`interleave_seed`]
///
/// [`interleave_seed`] unions *every* pair of places that co-occur in *any*
/// transition's support — including places merely **read** by a transition
/// (`pre == post`, a side condition) and places joined only through a pure
/// source/sink event. On a real net that frequently collapses into one giant
/// transition-connected component, and interleaving an arbitrary giant cluster
/// (mixing uncorrelated counters) is no better — sometimes worse — than the
/// place-banded identity, so the span/node guard simply discards it. Requiring
/// a *conservative exchange* keeps only the pairs whose token counts genuinely
/// co-vary, yielding tighter, value-correlated clusters that interleave well.
/// Both seeds are offered as candidates; the measured-node guard keeps whichever
/// (if either) actually shrinks the reachable BDD, with identity as incumbent.
///
/// Returns `None` when no two binary places are conservation-correlated.
///
/// # Soundness
///
/// Performance-only: the result is a permutation of the net's slots (the emit
/// pass places every slot exactly once), so it only renumbers BDD variables.
/// The conservation test is a *heuristic* for which places to interleave; it
/// need not be a true P-invariant — a mis-grouping can at worst yield a larger
/// BDD, which the node-count guard then declines in favour of the incumbent.
fn pinvariant_interleave_seed(
    n: usize,
    bit_counts: &[Option<u32>],
    slot_base: &[usize],
    slot_len: &[usize],
    transitions: &[DdTransition],
) -> Option<Vec<usize>> {
    // Union-find over places, unioning a pair (p, q) only when some transition
    // moves tokens between them in opposite directions — the P-semiflow
    // footprint. We require the net deltas to have opposite signs so that a
    // read-arc (Δ = 0) or two co-produced/co-consumed places that are NOT
    // exchanging (same-sign Δ, e.g. a fork that fills both) do not merge: only
    // genuine conservation pairs (token leaves p, enters q) cluster.
    let mut parent: Vec<usize> = (0..n).collect();
    for t in transitions {
        // Net deltas, and the indices that gain vs lose tokens. Clamp to `n`
        // (the union-find / parent size) as well as the vectors' own lengths,
        // so a malformed over-long transition can never index out of bounds.
        let mut gain: Vec<usize> = Vec::new();
        let mut lose: Vec<usize> = Vec::new();
        let len = t.pre.len().min(t.post.len()).min(n);
        for p in 0..len {
            let pre = t.pre[p] as i128;
            let post = t.post[p] as i128;
            let delta = post - pre;
            if delta > 0 {
                gain.push(p);
            } else if delta < 0 {
                lose.push(p);
            }
        }
        // A conservative exchange has at least one loser and one gainer; union
        // every loser with every gainer (the carry chain couples each such
        // pair). Bounded work: |lose|·|gain| <= |support(t)|^2.
        for &l in &lose {
            for &g in &gain {
                uf_union(&mut parent, l, g);
            }
        }
    }
    emit_interleaved_seed(n, bit_counts, slot_base, slot_len, &mut parent)
}

/// Relabel `spec` so that place index `level` is the original place
/// `order[level]`. The returned spec is *isomorphic* to `spec` (same net,
/// renamed places), so every reachability metric computed from it equals
/// the metric on `spec`. See the module soundness note.
#[must_use]
pub fn permute_spec(spec: &DdNetSpec, order: &[usize]) -> DdNetSpec {
    let n = order.len();
    debug_assert_eq!(n, spec.bounds.len());
    let inv = invert_order(order);
    let bounds: Vec<u64> = order.iter().map(|&o| spec.bounds[o]).collect();
    let initial_marking: Vec<u64> = order.iter().map(|&o| spec.initial_marking[o]).collect();
    let transitions: Vec<DdTransition> = spec
        .transitions
        .iter()
        .map(|t| {
            let mut pre = vec![0u64; n];
            let mut post = vec![0u64; n];
            for orig in 0..n {
                pre[inv[orig]] = t.pre[orig];
                post[inv[orig]] = t.post[orig];
            }
            DdTransition { pre, post }
        })
        .collect();
    DdNetSpec {
        bounds,
        initial_marking,
        transitions,
    }
}

/// Permute an UpperBounds query coefficient vector into the level
/// coordinates of a permuted spec (`inv[orig] = level`). The query's
/// maximum value is unchanged (the products are merely reindexed).
#[must_use]
pub fn permute_query(coeffs: &[u64], inv: &[usize]) -> Vec<u64> {
    let mut out = vec![0u64; coeffs.len()];
    for (orig, &c) in coeffs.iter().enumerate() {
        out[inv[orig]] = c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DdTransition;

    fn t(pre: Vec<u64>, post: Vec<u64>) -> DdTransition {
        DdTransition { pre, post }
    }

    /// A linear pipeline whose PNML order is deliberately *shuffled* so a
    /// good order is non-trivial: places 0..5 form a chain but are listed
    /// as 0,3,1,4,2,5 (interleaved), so neighbours in the firing relation
    /// are far apart in index space.
    fn shuffled_chain() -> DdNetSpec {
        // Logical chain a->b->c->d->e->f with one token, mapped to indices
        // a=0,b=2,c=4,d=1,e=3,f=5 (an interleave). Each transition moves a
        // token one step along the logical chain.
        let logical_to_idx = [0usize, 2, 4, 1, 3, 5];
        let n = 6;
        let mut transitions = Vec::new();
        for step in 0..5 {
            let from = logical_to_idx[step];
            let to = logical_to_idx[step + 1];
            let mut pre = vec![0u64; n];
            let mut post = vec![0u64; n];
            pre[from] = 1;
            post[to] = 1;
            transitions.push(t(pre, post));
        }
        let mut initial_marking = vec![0u64; n];
        initial_marking[logical_to_idx[0]] = 1;
        DdNetSpec {
            bounds: vec![1; n],
            initial_marking,
            transitions,
        }
    }

    #[test]
    fn force_order_improves_span_on_shuffled_chain() {
        let spec = shuffled_chain();
        let order = force_place_order(&spec);
        // FORCE must beat the identity span on this deliberately-bad layout.
        let identity: Vec<usize> = (0..spec.bounds.len()).collect();
        assert!(
            total_span(&spec, &order) < total_span(&spec, &identity),
            "FORCE order span {} should beat identity span {}",
            total_span(&spec, &order),
            total_span(&spec, &identity),
        );
        // And it must be a valid permutation.
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, identity, "order must be a permutation of 0..n");
    }

    // ---------------------------------------------------------------------
    // Bit-level (slot) ordering unit tests.
    // ---------------------------------------------------------------------

    /// `force_bit_slot_order` must always return a valid permutation of the
    /// net's slots: one atomic slot per unary place, one bit slot per bit of
    /// each binary place, every (place, bit) exactly once.
    #[test]
    fn bit_slot_order_is_a_valid_slot_permutation() {
        // Coupled conserved chain: 4 binary places (6 bits each) + 1 unary.
        let spec = DdNetSpec {
            bounds: vec![40, 40, 40, 40, 5],
            initial_marking: vec![40, 0, 0, 0, 0],
            transitions: vec![
                t(vec![1, 0, 0, 0, 0], vec![0, 1, 0, 0, 0]),
                t(vec![0, 1, 0, 0, 0], vec![0, 0, 1, 0, 0]),
                t(vec![0, 0, 1, 0, 0], vec![0, 0, 0, 1, 0]),
                t(vec![0, 0, 0, 1, 0], vec![0, 0, 0, 0, 1]),
            ],
        };
        // bound 40 ⇒ 6 bits; bound 5 ⇒ unary atom.
        let bit_counts = vec![Some(6), Some(6), Some(6), Some(6), None];
        let order = force_bit_slot_order(&spec, &bit_counts);
        let identity = identity_slots(&bit_counts);
        assert_eq!(order.len(), identity.len(), "same number of slots");
        // Multiset of (place, bit) keys must equal the identity's.
        let mut got: Vec<(usize, Option<u32>)> = order.iter().map(|s| (s.place, s.bit)).collect();
        let mut want: Vec<(usize, Option<u32>)> =
            identity.iter().map(|s| (s.place, s.bit)).collect();
        got.sort();
        want.sort();
        assert_eq!(
            got, want,
            "bit-slot order must be a permutation of the net's slots"
        );
    }

    /// On a deliberately bad place layout (a coupled chain whose places are
    /// listed interleaved so neighbours are far apart), the bit-level FORCE /
    /// interleave order must STRICTLY beat the identity bit-level span — the
    /// span guard then adopts it. (When it cannot beat identity the guard
    /// keeps identity; that no-regression direction is covered by the
    /// `None == identity` differential in lib.rs.)
    #[test]
    fn bit_slot_order_can_beat_identity_span_on_coupled_chain() {
        // Conserved chain a->b->c->d, all binary (6 bits), but listed in a
        // shuffled place order a=0, c=1, b=2, d=3 so the place band layout is
        // poor; bit-level FORCE/interleave should improve the bit-level span.
        let spec = DdNetSpec {
            bounds: vec![40, 40, 40, 40],
            initial_marking: vec![40, 0, 0, 0],
            transitions: vec![
                // a(0)->b(2)
                t(vec![1, 0, 0, 0], vec![0, 0, 1, 0]),
                // b(2)->c(1)
                t(vec![0, 0, 1, 0], vec![0, 1, 0, 0]),
                // c(1)->d(3)
                t(vec![0, 1, 0, 0], vec![0, 0, 0, 1]),
            ],
        };
        let bit_counts = vec![Some(6), Some(6), Some(6), Some(6)];
        // Identity bit span.
        let identity = identity_slots(&bit_counts);
        let num_slots = identity.len();
        let id_perm: Vec<usize> = (0..num_slots).collect();
        // Rebuild the supports the same way `force_bit_slot_order` does.
        let mut slot_base = [0usize; 4];
        let mut slot_len = [0usize; 4];
        {
            let mut id = 0usize;
            for (p, &bc) in bit_counts.iter().enumerate() {
                slot_base[p] = id;
                let len = bc.map_or(1, |k| k as usize);
                slot_len[p] = len;
                id += len;
            }
        }
        let supports: Vec<Vec<usize>> = spec
            .transitions
            .iter()
            .map(|t| {
                let mut s = Vec::new();
                for p in 0..t.pre.len() {
                    if t.pre[p] != 0 || t.post[p] != 0 {
                        for off in 0..slot_len[p] {
                            s.push(slot_base[p] + off);
                        }
                    }
                }
                s
            })
            .collect();
        let id_span = slot_total_span(&supports, &id_perm, num_slots);

        let order = force_bit_slot_order(&spec, &bit_counts);
        // Translate the chosen BitSlot order back to a global-slot permutation
        // to score its span with the same metric.
        let key_to_gid: std::collections::HashMap<(usize, Option<u32>), usize> = identity
            .iter()
            .enumerate()
            .map(|(gid, s)| ((s.place, s.bit), gid))
            .collect();
        let chosen_perm: Vec<usize> = order
            .iter()
            .map(|s| key_to_gid[&(s.place, s.bit)])
            .collect();
        let chosen_span = slot_total_span(&supports, &chosen_perm, num_slots);
        assert!(
            chosen_span <= id_span,
            "bit-slot order span {chosen_span} must never exceed identity span {id_span}",
        );
        // On this deliberately-bad layout the bit order should strictly help.
        assert!(
            chosen_span < id_span,
            "expected bit-slot order to strictly beat identity span on the shuffled chain \
             (chosen={chosen_span}, identity={id_span})",
        );
    }

    /// A net with no binary place (all unary) yields the identity slot order
    /// unchanged — the bit-level lever is binary-only by construction.
    #[test]
    fn bit_slot_order_identity_when_all_unary() {
        let spec = shuffled_chain();
        let bit_counts = vec![None; spec.bounds.len()];
        let order = force_bit_slot_order(&spec, &bit_counts);
        assert_eq!(order, identity_slots(&bit_counts));
    }

    // ---------------------------------------------------------------------
    // P-invariant-correlated interleave seed.
    // ---------------------------------------------------------------------

    /// Helper: build the (slot_base, slot_len) tables a seed function needs.
    fn slot_tables(bit_counts: &[Option<u32>]) -> (Vec<usize>, Vec<usize>) {
        let n = bit_counts.len();
        let mut slot_base = vec![0usize; n];
        let mut slot_len = vec![0usize; n];
        let mut id = 0usize;
        for (p, &bc) in bit_counts.iter().enumerate() {
            slot_base[p] = id;
            let len = bc.map_or(1, |k| k as usize);
            slot_len[p] = len;
            id += len;
        }
        (slot_base, slot_len)
    }

    /// Assert a candidate global-slot permutation is a valid permutation of all
    /// `num_slots` slots.
    fn assert_is_slot_permutation(perm: &[usize], num_slots: usize) {
        assert_eq!(perm.len(), num_slots, "seed must cover every slot once");
        let mut sorted = perm.to_vec();
        sorted.sort_unstable();
        let expect: Vec<usize> = (0..num_slots).collect();
        assert_eq!(sorted, expect, "seed must be a permutation of 0..num_slots");
    }

    /// The P-invariant seed clusters the two places joined by a conserving
    /// exchange (`a -> b`, token leaves a / enters b) and emits their bits
    /// bit-for-bit interleaved, while leaving an uncorrelated binary place
    /// (only ever produced, never exchanged) as its own contiguous block.
    #[test]
    fn pinvariant_seed_clusters_conserving_exchange_only() {
        // Places: a(0), b(1) conserved (a+b const); c(2) is a free counter
        // (its own +1 transition, never exchanges with a or b).
        let spec = DdNetSpec {
            bounds: vec![63, 63, 63],
            initial_marking: vec![63, 0, 0],
            transitions: vec![
                t(vec![1, 0, 0], vec![0, 1, 0]), // a -> b (conservative exchange)
                t(vec![0, 0, 0], vec![0, 0, 1]), // produce c (no exchange)
            ],
        };
        let bit_counts = vec![Some(6), Some(6), Some(6)];
        let (slot_base, slot_len) = slot_tables(&bit_counts);
        let seed =
            pinvariant_interleave_seed(3, &bit_counts, &slot_base, &slot_len, &spec.transitions)
                .expect("a/b form a conservation pair → a seed exists");
        let num_slots: usize = slot_len.iter().sum();
        assert_is_slot_permutation(&seed, num_slots);

        // a (slots 0..6) and b (slots 6..12) must be interleaved bit-for-bit at
        // the front: a0,b0,a1,b1,...; c (slots 12..18) follows as a block.
        let identity = identity_slots(&bit_counts);
        let as_keys: Vec<(usize, Option<u32>)> = seed
            .iter()
            .map(|&g| (identity[g].place, identity[g].bit))
            .collect();
        let mut expect: Vec<(usize, Option<u32>)> = Vec::new();
        for bit in 0..6u32 {
            expect.push((0, Some(bit)));
            expect.push((1, Some(bit)));
        }
        for bit in 0..6u32 {
            expect.push((2, Some(bit)));
        }
        assert_eq!(
            as_keys, expect,
            "P-invariant seed must interleave only the conserved pair a,b and keep c a block",
        );
    }

    /// The discriminating case: raw co-occurrence (`interleave_seed`) merges a
    /// pure source/sink-joined or read-joined place into a giant cluster, while
    /// the P-invariant seed clusters ONLY the genuinely conserved pair.
    #[test]
    fn pinvariant_seed_is_sharper_than_cooccurrence_on_read_arc() {
        // Transition t1 conserves a<->b. Transition t2 reads c (pre==post on c)
        // while also moving a token a->b again, so c co-occurs with a,b in t2's
        // support and the co-occurrence union-find merges {a,b,c}. But c only
        // has a read arc (no net delta), so it is NOT value-correlated and the
        // P-invariant seed must leave it OUT of the {a,b} cluster.
        let spec = DdNetSpec {
            bounds: vec![63, 63, 63],
            initial_marking: vec![63, 0, 1],
            transitions: vec![
                t(vec![1, 0, 0], vec![0, 1, 0]), // a -> b
                t(vec![1, 0, 1], vec![0, 1, 1]), // a -> b, read c (c: pre==post)
            ],
        };
        let bit_counts = vec![Some(6), Some(6), Some(6)];
        let (slot_base, slot_len) = slot_tables(&bit_counts);

        // Co-occurrence seed merges {a,b,c}: c's bits get interleaved in.
        let coocc =
            interleave_seed(3, &bit_counts, &slot_base, &slot_len, &spec.transitions).unwrap();
        let identity = identity_slots(&bit_counts);
        let coocc_keys: Vec<usize> = coocc.iter().map(|&g| identity[g].place).collect();
        // The first six emitted slots of the co-occurrence cluster touch all of
        // a,b,c (giant cluster), so place 2 appears within the first 6 slots.
        assert!(
            coocc_keys[..6].contains(&2),
            "co-occurrence seed should (spuriously) pull c into the front cluster: {coocc_keys:?}",
        );

        // P-invariant seed keeps {a,b} only; c (read-only, Δ=0) stays a block at
        // the back, so place 2 never appears within the first interleaved band.
        let pinv =
            pinvariant_interleave_seed(3, &bit_counts, &slot_base, &slot_len, &spec.transitions)
                .unwrap();
        let pinv_keys: Vec<usize> = pinv.iter().map(|&g| identity[g].place).collect();
        assert!(
            !pinv_keys[..12].contains(&2),
            "P-invariant seed must keep read-only c OUT of the a,b conservation cluster: {pinv_keys:?}",
        );
        let num_slots: usize = slot_len.iter().sum();
        assert_is_slot_permutation(&pinv, num_slots);
    }

    /// A fork that *fills* two places from one source (both gain tokens, same
    /// sign Δ, no exchange) must NOT be treated as a conservation pair: the
    /// P-invariant seed returns `None` (nothing value-correlated to interleave),
    /// whereas co-occurrence would merge them.
    #[test]
    fn pinvariant_seed_ignores_same_sign_fork() {
        let spec = DdNetSpec {
            bounds: vec![63, 63],
            initial_marking: vec![0, 0],
            // Single transition produces into BOTH a and b (both Δ = +1): a
            // co-production, not an exchange. No token leaves either place.
            transitions: vec![t(vec![0, 0], vec![1, 1])],
        };
        let bit_counts = vec![Some(6), Some(6)];
        let (slot_base, slot_len) = slot_tables(&bit_counts);
        // Co-occurrence DOES cluster them (they share the transition support).
        assert!(
            interleave_seed(2, &bit_counts, &slot_base, &slot_len, &spec.transitions).is_some(),
            "co-occurrence merges the co-produced pair",
        );
        // P-invariant seed sees no loser/gainer exchange ⇒ no conservation pair.
        assert!(
            pinvariant_interleave_seed(2, &bit_counts, &slot_base, &slot_len, &spec.transitions)
                .is_none(),
            "same-sign co-production is not a conservation pair → no P-invariant seed",
        );
    }

    /// `bit_order_candidates` includes the P-invariant seed and every returned
    /// candidate is a valid slot permutation (answer-preserving by
    /// construction). Identity is always candidate 0.
    #[test]
    fn bit_order_candidates_include_valid_pinvariant_seed() {
        // Two conserved pairs joined only by a read arc — the discriminating
        // shape where the P-invariant seed differs from co-occurrence.
        let spec = DdNetSpec {
            bounds: vec![63, 63, 63, 63],
            initial_marking: vec![63, 0, 63, 0],
            transitions: vec![
                t(vec![1, 0, 0, 0], vec![0, 1, 0, 0]), // a -> b
                t(vec![0, 0, 1, 0], vec![0, 0, 0, 1]), // c -> d
            ],
        };
        let bit_counts = vec![Some(6), Some(6), Some(6), Some(6)];
        let cands = bit_order_candidates(&spec, &bit_counts);
        let identity = identity_slots(&bit_counts);
        assert_eq!(
            cands[0], identity,
            "identity must be the incumbent candidate"
        );
        // Every candidate is a permutation of the same slot multiset.
        let mut want: Vec<(usize, Option<u32>)> =
            identity.iter().map(|s| (s.place, s.bit)).collect();
        want.sort();
        for (i, c) in cands.iter().enumerate() {
            let mut got: Vec<(usize, Option<u32>)> = c.iter().map(|s| (s.place, s.bit)).collect();
            got.sort();
            assert_eq!(got, want, "candidate #{i} must be a valid slot permutation");
        }
        // The P-invariant seed (interleaves a,b and c,d separately) must be
        // present among the candidates.
        let mut pinv_expect: Vec<(usize, Option<u32>)> = Vec::new();
        for bit in 0..6u32 {
            pinv_expect.push((0, Some(bit)));
            pinv_expect.push((1, Some(bit)));
        }
        for bit in 0..6u32 {
            pinv_expect.push((2, Some(bit)));
            pinv_expect.push((3, Some(bit)));
        }
        let present = cands
            .iter()
            .any(|c| c.iter().map(|s| (s.place, s.bit)).collect::<Vec<_>>() == pinv_expect);
        assert!(
            present,
            "the P-invariant pairwise-interleave order must be a candidate"
        );
    }
}
