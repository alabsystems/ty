// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Base-and-strong-generating-set (BSGS) over the place-permutation group.
//!
//! This module implements *exact* general group-orbit counting for the
//! COUPLED / diagonal symmetry groups that the per-orbit multinomial
//! ([`super::orbit_size::multinomial_orbit_size`]) cannot count — e.g.
//! Anderson-PT-04's single order-24 group permuting 23 size-4 orbits
//! *together*. The multinomial assumes each place orbit is a *full* symmetric
//! group acting independently (a direct product `∏_j Sym(G_j)`); a coupled
//! group is a strict subgroup of that product, so the multinomial OVER-counts
//! and the per-orbit-sort canonical form OVER-merges. Both are unsound for
//! coupled groups, which is why the explorer currently *refuses* them and
//! falls back to exact, un-reduced exploration.
//!
//! # What this provides
//!
//! Given the place-domain generators (each a permutation `π` of `0..num_p`,
//! action `applied[i] = marking[π[i]]`, exactly as in
//! [`super::orbit_size::orbit_size_of`] and
//! [`super::symmetry::PetriCanonicalizer::canonicalize`]), deterministic
//! Schreier–Sims builds a base `B = [b_0, …, b_{k-1}]` with per-level
//! fundamental orbits + coset transversals (Schreier vectors), a strong
//! generating set, and `|G| = ∏_i |Δ_i|` as `u128`.
//!
//! Then, **exactly** for any finite group action:
//!
//! - [`Bsgs::orbit_size`]`(m)` = `|orbit(m)|`, by a HYBRID dispatch that is
//!   exact on both branches: for small `|G|` the enumerative orbit BFS
//!   ([`Bsgs::orbit_size_enumerative`], ≤`|G|` cheap image applications); for
//!   larger `|G|` the non-enumerative REGIME-B quotient `|orbit(m)| = |G| /
//!   |Stab_G(m)|`, with `|Stab_G(m)|` from [`Bsgs::stabilizer_order`] (a
//!   same-base stabilizer chain that never enumerates `|G|`). The quotient is
//!   exact by Lagrange; returns `None` (→ CANNOT_COMPUTE) on `u128`/`u64`
//!   overflow or non-divisibility — fail-closed, never a wrong number. The
//!   crossover threshold affects only SPEED, never the count (the branches are
//!   differentially cross-checked over all markings of many groups).
//! - [`Bsgs::canonical_image`]`(m)` = the lexicographically least marking in
//!   `m`'s `G`-orbit (smallest-image), giving EXACTLY one representative per
//!   `G`-orbit so `Σ_reps |orbit(rep)| = |R|`.
//!
//! # Exactness
//!
//! Rests on three classical results applied to the concrete place action:
//! (1) orbit–stabilizer `|orbit(m)| = |G|/|Stab_G(m)|` holds for ANY finite
//! group action (no full-symmetric hypothesis); (2) `|G| = ∏|Δ_i|` is exact
//! from a sift-verified deterministic BSGS; (3) `g ∈ Stab_G(m) ⟺
//! applied(g,m) == m` is an exact `O(num_p)` membership oracle, so a fresh
//! Schreier–Sims on a generating set of `Stab_G(m)` yields `|Stab_G(m)|`
//! exactly.
//!
//! Because the generators ARE the full discovered place-symmetry group
//! regardless of the truncatable permutation-closure cache, this path is
//! immune to `PETRI_CANONICALIZER_CLOSURE_BUDGET` truncation by construction.

/// A permutation of `0..n` stored as its image vector with a cached inverse so
/// that sifting (multiply by transversal inverses) is allocation-light.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Perm {
    /// `img[i]` = image of point `i`.
    img: Vec<u32>,
    /// `inv[img[i]] = i`.
    inv: Vec<u32>,
}

impl Perm {
    fn identity(n: usize) -> Self {
        let img: Vec<u32> = (0..n as u32).collect();
        let inv = img.clone();
        Self { img, inv }
    }

    fn from_img(img: Vec<u32>) -> Self {
        let mut inv = vec![0u32; img.len()];
        for (i, &j) in img.iter().enumerate() {
            inv[j as usize] = i as u32;
        }
        Self { img, inv }
    }

    #[inline]
    fn apply(&self, p: u32) -> u32 {
        self.img[p as usize]
    }

    #[inline]
    fn apply_inv(&self, p: u32) -> u32 {
        self.inv[p as usize]
    }

    /// `self ∘ other` (apply `other` first, then `self`): maps
    /// `i -> self[other[i]]`.
    fn compose(&self, other: &Perm) -> Perm {
        let n = self.img.len();
        let mut img = vec![0u32; n];
        for i in 0..n {
            img[i] = self.img[other.img[i] as usize];
        }
        Perm::from_img(img)
    }

    fn inverse(&self) -> Perm {
        Perm {
            img: self.inv.clone(),
            inv: self.img.clone(),
        }
    }

    fn is_identity(&self) -> bool {
        self.img.iter().enumerate().all(|(i, &j)| i as u32 == j)
    }
}

/// One level of the stabilizer chain: a base point, its fundamental orbit
/// under the level's stabilizer subgroup `G^(i)`, and a transversal of coset
/// reps. The level's generating set is NOT stored here — it is derived on
/// demand as `{ g ∈ sgs : g fixes b_0..b_{i-1} }` (a generator at any level
/// fixes all earlier base points, so it generates every earlier stabilizer's
/// action that fixes those points).
#[derive(Clone, Debug)]
struct Level {
    /// The base point `b_i` for this level.
    base: u32,
    /// `transversal[p]` is a coset rep `u ∈ G^(i)` with `u(b_i) = p`, for each
    /// `p` in the fundamental orbit `Δ_i`; `None` otherwise.
    transversal: Vec<Option<Perm>>,
    /// The fundamental orbit `Δ_i`, in BFS-discovery order.
    orbit: Vec<u32>,
}

impl Level {
    fn new(base: u32, n: usize) -> Self {
        Self {
            base,
            transversal: vec![None; n],
            orbit: Vec::new(),
        }
    }
}

/// Upper bound on `|G|` for which [`Bsgs::build`] materializes the full element
/// list used by the FAST [`Bsgs::canonical_image`] lex-min. Above this the
/// element list is left empty and `canonical_image` uses the pruned backtrack
/// (correct, just slower). Chosen well above the default coupled-group budget
/// (`PETRI_CANONICALIZER_BSGS_GROUP_BUDGET`, 1024) so every coupled group
/// admitted on the default path gets the fast image; the secondary entry cap
/// bounds memory when the budget is raised via `TY_MCC_BSGS_GROUP_BUDGET`.
const CANONICAL_ELEMENT_GROUP_CAP: u128 = 200_000;

/// Upper bound on `|G| · n` (cache entries, `u32` each) for the
/// [`Bsgs::canonical_image`] element list — caps the cache at ~256 MB so a
/// large degree `n` cannot blow memory even when `|G|` is under
/// [`CANONICAL_ELEMENT_GROUP_CAP`].
const CANONICAL_ELEMENT_ENTRY_CAP: u128 = 64_000_000;

/// The base-and-strong-generating-set of a place-permutation group.
#[derive(Clone, Debug)]
pub(crate) struct Bsgs {
    n: usize,
    levels: Vec<Level>,
    /// Flat strong generating set. Level `i`'s generators are exactly those
    /// `g ∈ sgs` fixing `b_0..b_{i-1}` pointwise.
    sgs: Vec<Perm>,
    /// `|G| = ∏_i |Δ_i|`. `None` iff the product overflowed `u128` at build
    /// time (a genuinely astronomically large group).
    order: Option<u128>,
    /// Materialized image vectors of EVERY element of `G` (`elements[k][i]` =
    /// `g_k(i)`), populated by [`Bsgs::build`] when `|G|` is within
    /// [`CANONICAL_ELEMENT_GROUP_CAP`]/[`CANONICAL_ELEMENT_ENTRY_CAP`]. Empty
    /// otherwise (and for the internal stabilizer-chain BSGS instances built by
    /// [`build_chain`], which never canonicalize). When present,
    /// [`Bsgs::canonical_image`] takes the FAST `O(|G|·n)` lex-min over this
    /// list instead of the pruned per-call backtrack. The list is exact (one
    /// entry per group element via the unique transversal factorization), so
    /// the lex-min is the true `G`-orbit minimal image — identical to the
    /// backtrack, asserted in debug builds and the unit tests.
    elements: Vec<Vec<u32>>,
}

impl Bsgs {
    pub(crate) fn order(&self) -> Option<u128> {
        self.order
    }

    /// Number of points the group acts on (`num_p`).
    pub(crate) fn degree(&self) -> usize {
        self.n
    }

    /// Build a BSGS from place-domain generators via deterministic
    /// Schreier–Sims. `generators` are permutations of `0..n` (each `Vec<usize>`
    /// of length `n`). Returns `None` if `n == 0` or there is no nontrivial
    /// generator.
    pub(crate) fn build(generators: &[Vec<usize>], n: usize) -> Option<Bsgs> {
        if n == 0 {
            return None;
        }
        let gens: Vec<Perm> = generators
            .iter()
            .filter(|g| g.len() == n)
            .map(|g| Perm::from_img(g.iter().map(|&x| x as u32).collect()))
            .filter(|p| !p.is_identity())
            .collect();
        if gens.is_empty() {
            return None;
        }
        let mut bsgs = build_chain(gens, n);
        bsgs.precompute_canonical_elements();
        Some(bsgs)
    }

    /// Materialize the full element-image list for the FAST
    /// [`Self::canonical_image`] lex-min, when `|G|` is within the cache caps.
    /// Idempotent; a no-op (leaves `elements` empty) when `|G|` overflowed
    /// `u128` or exceeds [`CANONICAL_ELEMENT_GROUP_CAP`] /
    /// [`CANONICAL_ELEMENT_ENTRY_CAP`], in which case `canonical_image` uses the
    /// pruned backtrack.
    fn precompute_canonical_elements(&mut self) {
        if !self.elements.is_empty() {
            return;
        }
        let order = match self.order {
            Some(o) => o,
            None => return,
        };
        if order > CANONICAL_ELEMENT_GROUP_CAP
            || order.saturating_mul(self.n as u128) > CANONICAL_ELEMENT_ENTRY_CAP
        {
            return;
        }
        self.elements = self
            .enumerate_elements()
            .into_iter()
            .map(|p| p.img)
            .collect();
    }

    /// `|G|` threshold below which the ENUMERATIVE orbit BFS is cheaper than the
    /// regime-B `|G|/|Stab|` backtrack. Measured crossover (release, coupled
    /// groups): at `|G|=24` enumerative is ~3x faster (≤24 images is trivial),
    /// while at `|G|=5040` regime-B is ~3.6x faster (enumerative's `O(|G|)` blows
    /// up). We dispatch enumerative for `|G| ≤ this` and regime-B above, so
    /// `orbit_size` is NEVER slower than the prior enumerative path AND scales to
    /// large coupled groups. Both paths are EXACT (differentially cross-checked
    /// over all markings of many groups), so the threshold affects only SPEED,
    /// never the count — it can be retuned freely.
    const ENUMERATIVE_ORBIT_GROUP_BOUND: u128 = 256;

    /// `|orbit(m)|` — the number of distinct markings in `m`'s `G`-orbit.
    ///
    /// HYBRID, both branches EXACT:
    ///   * `|G| ≤ ENUMERATIVE_ORBIT_GROUP_BOUND`: the enumerative orbit BFS
    ///     (cheap when `|G|` is tiny — ≤ `|G|` image applications).
    ///   * larger `|G|`: REGIME B — `|orbit(m)| = |G| / |Stab_G(m)|` by
    ///     orbit–stabilizer, with `|G|` exact from the BSGS and `|Stab_G(m)|`
    ///     from [`Self::stabilizer_order`] (a same-base stabilizer chain that
    ///     NEVER enumerates `|G|` images). Exact by Lagrange: `|Stab|` divides
    ///     `|G|`, so the integer division is remainder-free.
    ///
    /// Returns `None` (fail-closed → CANNOT_COMPUTE) when the marking length
    /// mismatches, `|G|` overflowed `u128` at build time, the quotient does not
    /// fit `u64`, or — defensively — `|Stab|` does not divide `|G|` (which would
    /// signal a stabilizer-chain bug; we refuse rather than emit a wrong count).
    pub(crate) fn orbit_size(&self, marking: &[u64]) -> Option<u64> {
        if marking.len() != self.n {
            return None;
        }
        // Short-circuit the common symmetric marking: if every strong generator
        // fixes m, the whole group fixes it and the orbit is a singleton.
        // (Exact regardless of |G|, so checked before the |G| bound.)
        if self.sgs.iter().all(|g| preserves_marking(g, marking)) {
            return Some(1);
        }
        // |orbit(m)| divides |G| (orbit–stabilizer). If |G| itself overflowed
        // u128 at build time the quotient is unsafe to bound -> fail closed.
        let order = self.order?;
        // Small group: the enumerative BFS is faster than the stabilizer
        // backtrack (≤ |G| cheap image applications). Exact by definition.
        if order <= Self::ENUMERATIVE_ORBIT_GROUP_BOUND {
            return self.orbit_size_enumerative(marking);
        }
        // Large group: regime-B |G|/|Stab| (no |G| enumeration).
        let stab = self.stabilizer_order(marking)?;
        if stab == 0 || order % stab != 0 {
            // Lagrange guarantees divisibility for the TRUE stabilizer order; a
            // non-zero remainder means the chain mis-counted -> refuse.
            return None;
        }
        let size = order / stab;
        debug_assert!(size <= order, "orbit cannot exceed |G|");
        debug_assert_eq!(
            size,
            self.orbit_size_enumerative(marking)
                .map(u128::from)
                .unwrap_or(size),
            "regime-B |G|/|Stab| must match the enumerative orbit size",
        );
        u64::try_from(size).ok()
    }

    /// ENUMERATIVE reference orbit size (the previous regime-A implementation),
    /// kept as the cross-check oracle for [`Self::orbit_size`]: starting from
    /// `m`, repeatedly apply the strong generators and collect distinct image
    /// markings until closure. This is `|orbit(m)|` by definition. `O(|orbit|)`
    /// in images, so only safe for groups with `|G| ≤ u64::MAX`. Returns `None`
    /// on length mismatch or `|G|` overflow.
    pub(crate) fn orbit_size_enumerative(&self, marking: &[u64]) -> Option<u64> {
        if marking.len() != self.n {
            return None;
        }
        if self.sgs.iter().all(|g| preserves_marking(g, marking)) {
            return Some(1);
        }
        let order = self.order?;
        if order > u64::MAX as u128 {
            return None;
        }
        use std::collections::HashSet;
        let n = self.n;
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        seen.insert(marking.to_vec());
        let mut frontier: Vec<Vec<u64>> = vec![marking.to_vec()];
        let mut scratch = vec![0u64; n];
        while let Some(cur) = frontier.pop() {
            for g in &self.sgs {
                for i in 0..n {
                    scratch[i] = cur[g.apply(i as u32) as usize];
                }
                if !seen.contains(&scratch) {
                    let img = scratch.clone();
                    seen.insert(img.clone());
                    frontier.push(img);
                }
            }
        }
        let size = seen.len() as u128;
        debug_assert!(size <= order, "orbit cannot exceed |G|");
        debug_assert_eq!(order % size, 0, "orbit-stabilizer: |orbit| divides |G|");
        u64::try_from(size).ok()
    }

    /// `|Stab_G(m)|` — the order of the marking stabilizer `H = { g ∈ G :
    /// applied(g,m) == m }` — WITHOUT enumerating `G`.
    ///
    /// # Algorithm (exact, complete, non-enumerative)
    ///
    /// `H` is the pointwise stabilizer of the COLORING `m`: `g ∈ H ⟺ m[g(p)] ==
    /// m[p] ∀p` (equivalently, the color classes `P_v = {p : m[p]=v}` are each
    /// preserved SETWISE). We compute a complete stabilizer chain for `H` along
    /// `G`'s OWN base `b_0,…,b_{k-1}` by a **pruned base-image backtrack** (a
    /// minimal Leon-style partition backtrack): we descend `G`'s stabilizer
    /// chain level by level, and at each level enumerate the admissible images
    /// of the base point `b_i` — those `q` in `G`'s fundamental orbit `Δ_i` with
    /// `m[q] == m[b_i]` (color-preservation is NECESSARY for any `h ∈ H` with
    /// `h(b_i)=q`, since `m[q]=m[h(b_i)]=m[b_i]`). For each admissible `q` we
    /// extend the partial base image and recurse into `G^(i+1)`'s coset via the
    /// transversal element `u_q`. A leaf (all base points imaged) yields a
    /// concrete `g ∈ G`; if it preserves `m` (exact oracle) it is an element of
    /// `H`, which we SIFT into an incrementally-built BSGS for `H`.
    ///
    /// Two prunes keep this `O(k·|Δ_i|·num_p)` rather than `O(|G|)`:
    ///   * COLOR prune: skip `q` with `m[q] != m[b_i]` (no `H`-element there).
    ///   * COSET prune: once the partial `H`-chain already reaches base image
    ///     `q` at level `i` (its level-`i` transversal covers `q`), every
    ///     `H`-element with that prefix lies in a coset already represented, so
    ///     exploring it cannot grow `H` — skip it. This is the standard
    ///     "one element per coset of the known subgroup" backtrack prune, and it
    ///     bounds the leaves reached to `O(|H|/|H^{(i+1)}|)` discoveries per
    ///     level, i.e. just enough to GENERATE `H`, never all `|G|` elements.
    ///
    /// Completeness: the backtrack visits at least one `g ∈ G` for every
    /// color-consistent base image, and sifts every `H`-element it meets; the
    /// coset prune only skips elements provably in the span of already-found
    /// generators. Hence the resulting chain is a complete BSGS for `H` and
    /// `|H| = ∏_i |Δ^H_i|` is EXACT (the same product formula trusted for `|G|`).
    /// Cross-checked: [`Self::orbit_size`] additionally REFUSES (returns `None`)
    /// unless `|H|` divides `|G|`, and the unit tests assert `|G|/|H|` equals the
    /// enumerative orbit size over ALL markings of many group shapes.
    ///
    /// Returns `None` on length mismatch or `u128` overflow of `|H|`.
    pub(crate) fn stabilizer_order(&self, marking: &[u64]) -> Option<u128> {
        if marking.len() != self.n {
            return None;
        }
        let n = self.n;
        // Collect a generating set of H by the pruned base-image backtrack, then
        // hand it to the TRUSTED `build_chain` to read off `|H| = ∏|Δ^H_i|`. The
        // backtrack maintains a running BSGS of the elements found so far (for
        // the soundness-preserving coset prune); `found` is that BSGS.
        let mut collector = StabCollector { found: None, n };
        self.stab_backtrack(0, &Perm::identity(n), marking, &mut collector);
        match collector.found {
            None => Some(1), // only the identity stabilizes ⇒ |H| = 1
            Some(chain) => chain.order,
        }
    }

    /// One DFS frame of the [`Self::stabilizer_order`] base-image backtrack.
    /// `accum` ∈ G maps `b_0..b_{level-1}` to the committed images. We extend by
    /// choosing `accum(b_level)`'s target among the color-matching points of
    /// `Δ_level` (realized via `G`'s level-`level` transversal), prune cosets
    /// already spanned by the found `H`-generators, and recurse.
    fn stab_backtrack(
        &self,
        level: usize,
        accum: &Perm,
        marking: &[u64],
        collector: &mut StabCollector,
    ) {
        if level == self.levels.len() {
            // Leaf: `accum` is a concrete g ∈ G. It is in H iff it preserves m;
            // record it (the collector rebuilds H's BSGS from accumulated gens).
            if preserves_marking(accum, marking) {
                collector.add(accum.clone());
            }
            return;
        }
        let lvl = &self.levels[level];
        let base_pt = lvl.base;
        for q in &lvl.orbit {
            let rep = match &lvl.transversal[*q as usize] {
                Some(r) => r,
                None => continue,
            };
            // Image of base_pt if we pick coset `q`: accum(q). COLOR prune — any
            // H-extension needs m[accum(q)] == m[base_pt] (necessary since an
            // h ∈ H with h(base_pt)=accum(q) forces m[accum(q)]=m[base_pt]).
            let img = accum.apply(*q) as usize;
            if marking[img] != marking[base_pt as usize] {
                continue;
            }
            // COSET prune (SOUND): skip the subtree only when the found
            // H-generators already realize the FULL base-image prefix
            // (accum(b_0),…,accum(b_{level-1}), accum(q)). If a known h ∈ ⟨found⟩
            // realizes it, any further g ∈ H with this prefix has h^{-1}g fixing
            // b_0..b_level, hence g ∈ h·H^(level+1) where H^(level+1) is already
            // generated under h's subtree — so this subtree yields no new
            // generator. Tested by sifting the prefix through ⟨found⟩'s chain.
            if collector.prefix_spanned(accum, *q, level, &self.levels) {
                continue;
            }
            let g_partial = accum.compose(rep);
            self.stab_backtrack(level + 1, &g_partial, marking, collector);
        }
    }

    /// Exact membership test `g ∈ G` by sifting `g` through the stabilizer
    /// chain: strip each base point's image via its transversal rep; `g ∈ G`
    /// iff the residue reaches the identity after all levels. `O(k·n)`, no
    /// enumeration. Used by the stabilizer backtrack's coset prune.
    fn contains(&self, g: &Perm) -> bool {
        let mut residue = g.clone();
        for level in &self.levels {
            let img = residue.apply(level.base);
            match level.transversal[img as usize].as_ref() {
                Some(u) => residue = u.inverse().compose(&residue),
                None => return false, // base image outside Δ_i ⇒ g ∉ G
            }
        }
        residue.is_identity()
    }

    /// Generators of the level-`i` stabilizer subgroup `G^(i)`: strong
    /// generators fixing `b_0..b_{i-1}` pointwise.
    fn level_gens(&self, i: usize) -> Vec<Perm> {
        let earlier: Vec<u32> = self.levels[..i].iter().map(|l| l.base).collect();
        self.sgs
            .iter()
            .filter(|g| earlier.iter().all(|&b| g.apply(b) == b))
            .cloned()
            .collect()
    }

    /// The lexicographically least marking in `m`'s `G`-orbit — the canonical
    /// representative. Yields EXACTLY one representative per `G`-orbit.
    ///
    /// The orbit of `m` is `{ w : w[i] = m[g(i)], g ∈ G }` (the same set the
    /// orbit BFS in [`Self::orbit_size`] enumerates). The canonical image is
    /// the lexicographically least `w`. Minimizing `w` lexicographically means
    /// greedily minimizing `m[g(0)]`, then `m[g(1)]`, … over `g ∈ G`.
    ///
    /// FAST PATH — when [`Self::build`] materialized the element list (`|G|`
    /// within the cache caps, which holds for every coupled group admitted on
    /// the default path), the lex-min is a direct `O(|G|·n)` scan over the
    /// precomputed group elements with an early-exit comparison: for each
    /// `g ∈ G` form `apply(g, m)[i] = m[g(i)]` and keep the lexicographically
    /// least. This avoids the per-call Schreier-vector BFS, `Perm`
    /// compositions, and `HashSet` allocations of the backtrack below — the
    /// measured wall-time wall that made the coupled quotient a net loss
    /// (Anderson-PT-05 |G|=120 dropped from ~18 s to well under the un-reduced
    /// ~5 s). The result is byte-identical to the backtrack (asserted in debug
    /// builds and exhaustively in the unit tests), so this is a SPEED change
    /// only — never a different canonical form, never a different verdict.
    ///
    /// Falls back to the pruned backtrack when the element list is absent
    /// (`|G|` over the cache caps — only reachable via a raised
    /// `TY_MCC_BSGS_GROUP_BUDGET`).
    pub(crate) fn canonical_image(&self, marking: &[u64]) -> Vec<u64> {
        if !self.elements.is_empty() && marking.len() == self.n {
            let result = self.canonical_image_cached(marking);
            debug_assert_eq!(
                result,
                self.canonical_image_backtrack(marking),
                "cached lex-min canonical image must equal the pruned backtrack",
            );
            return result;
        }
        self.canonical_image_backtrack(marking)
    }

    /// Lexicographically least `apply(g, m)` over the precomputed element list.
    /// Requires `marking.len() == self.n` and a non-empty `elements` (the
    /// identity is always present). Exact by construction: `elements` is the
    /// complete group, so the minimum is the true `G`-orbit minimal image.
    fn canonical_image_cached(&self, marking: &[u64]) -> Vec<u64> {
        let n = self.n;
        let mut best: Vec<u64> = Vec::with_capacity(n);
        let first = &self.elements[0];
        for i in 0..n {
            best.push(marking[first[i] as usize]);
        }
        for g in &self.elements[1..] {
            // Lexicographic compare apply(g, m) vs best, early-exit at the first
            // differing coordinate.
            let mut better = false;
            for i in 0..n {
                let v = marking[g[i] as usize];
                match v.cmp(&best[i]) {
                    std::cmp::Ordering::Less => {
                        better = true;
                        break;
                    }
                    std::cmp::Ordering::Greater => break,
                    std::cmp::Ordering::Equal => {}
                }
            }
            if better {
                for i in 0..n {
                    best[i] = marking[g[i] as usize];
                }
            }
        }
        best
    }

    /// PRUNED BACKTRACK over a base ordered `0,1,…,n-1`: we build `g` by fixing
    /// its images of points `0,1,2,…` in order. At step `i`, having fixed
    /// `g(0..i)`, the admissible values for `g(i)` are the orbit of `i` under
    /// the pointwise stabilizer of `{0,…,i-1}` (the elements of `G` consistent
    /// with the prefix). We pick the value(s) minimizing `m[g(i)]`, keeping all
    /// ties as live partial maps, and prune any partial map whose committed
    /// prefix already exceeds the best. This explores only the lex-minimal
    /// frontier, not the whole group, so it is fast even for `|G|` in the
    /// thousands. Exactness is differentially verified against the enumerative
    /// `enumerate_elements` minimum in the unit tests.
    fn canonical_image_backtrack(&self, marking: &[u64]) -> Vec<u64> {
        let n = self.n;
        if self.sgs.is_empty() || n == 0 {
            return marking.to_vec();
        }
        // A live partial map is a concrete group element `g` consistent with the
        // committed prefix `g(0),…,g(i-1)`. We seed with a generating set of G
        // (the SGS plus identity) — every group element is reachable as a
        // product, but for the GREEDY MINIMAL we only need, at each step, the
        // orbit of point `i` under the stabilizer of the committed prefix. We
        // realize that by maintaining the set of partial group elements; at
        // each level we extend by the stabilizer's coset reps.
        //
        // Implementation via repeated stabilizer-orbit: maintain `cands`, the
        // set of group elements (Perm) achieving the current best prefix. Start
        // from all of G represented compactly by the chain; but to avoid
        // enumerating G we keep `cands` as actual Perms and grow them lazily.
        //
        // To bound memory we represent `cands` as the set of distinct PREFIX
        // assignments g(0..i) -> the minimal one, plus one witness perm each.
        // Concretely: BFS over "partial images" keyed by the assignment tuple.
        // For the small groups in scope this is cheap and provably exact.
        //
        // We use the following exact procedure:
        //   live: Vec<Perm> = a transversal of the cosets consistent with the
        //         committed prefix; initially the full SGS-generated coset set
        //         for the empty prefix = the right cosets of the trivial group,
        //         i.e. all of G. We materialize G lazily LEVEL BY LEVEL using
        //         the identity-ordered pointwise-stabilizer orbits.
        //
        // Build an identity-ordered stabilizer chain ON DEMAND: the orbit of
        // point i under the group fixing 0..i-1 pointwise, with coset reps.
        let mut prefix_fixed: Vec<u32> = Vec::new(); // points 0..i-1 already pinned
                                                     // `coset_reps`: representatives of the cosets of Stab(prefix) in
                                                     // Stab(prefix without last) — i.e. how point i can map. We carry the
                                                     // accumulated partial element(s).
        let mut live: Vec<Perm> = vec![Perm::identity(n)];
        // best_value[i] = the chosen minimal m[g(i)] at each decided position.
        let mut result = marking.to_vec();
        for i in 0..n as u32 {
            // For each live partial element g, the admissible images of point i
            // are { (g ∘ s)(i) : s ∈ Stab_G(prefix) } — but we track live as
            // elements already mapping the prefix correctly. Compute, per live
            // g, the orbit of i under the stabilizer of prefix_fixed restricted
            // to extensions of g. To keep it exact and simple, compute the
            // orbit of point i under the subgroup of G fixing prefix_fixed
            // pointwise (same for all live g sharing that prefix), then for each
            // reachable target value pick the minimum.
            //
            // Stab generators for the current prefix:
            let stab_gens: Vec<&Perm> = self
                .sgs
                .iter()
                .filter(|s| prefix_fixed.iter().all(|&p| s.apply(p) == p))
                .collect();
            // For each live g, image candidates for position i are
            // { g(s(i)) : s ∈ ⟨stab_gens⟩ }. Compute the orbit of i under
            // ⟨stab_gens⟩ once (independent of g), then map through each g.
            let orbit_i = orbit_points(i, &stab_gens, n);
            // Determine the minimal achievable m[g(i)] over live g and orbit.
            let mut best_val: Option<u64> = None;
            for g in &live {
                for &q in &orbit_i {
                    let v = marking[g.apply(q) as usize];
                    best_val = Some(best_val.map_or(v, |b| b.min(v)));
                }
            }
            let best_val = best_val.expect("orbit non-empty");
            result[i as usize] = best_val;
            // Keep only the live elements (extended) that realize best_val at
            // position i AND map prefix correctly. For each live g and each
            // s-coset rep mapping i to a point q with m[g(q)]==best_val, the
            // extended element g' = g ∘ (rep mapping i->q) pins g'(i)=g(q).
            let reps = stabilizer_coset_reps(i, &stab_gens, n);
            let mut next_live: Vec<Perm> = Vec::new();
            let mut seen_prefix: std::collections::HashSet<Vec<u32>> =
                std::collections::HashSet::new();
            for g in &live {
                for rep in &reps {
                    let gq = g.compose(rep); // gq fixes prefix, gq(i) = g(rep(i))
                    if marking[gq.apply(i) as usize] == best_val {
                        // Dedup by the committed prefix images 0..=i to bound
                        // the frontier (only one witness per distinct prefix).
                        let key: Vec<u32> = (0..=i).map(|p| gq.apply(p)).collect();
                        if seen_prefix.insert(key) {
                            next_live.push(gq);
                        }
                    }
                }
            }
            live = next_live;
            prefix_fixed.push(i);
            if live.is_empty() {
                // Should not happen (best_val was achievable); fail safe.
                break;
            }
        }
        result
    }

    /// Enumerate all elements of `G` as products of one transversal rep per
    /// level (each group element factors uniquely as `u_0·u_1·…·u_{k-1}`).
    fn enumerate_elements(&self) -> Vec<Perm> {
        let mut frontier: Vec<Perm> = vec![Perm::identity(self.n)];
        for level in &self.levels {
            let reps: Vec<&Perm> = level.transversal.iter().flatten().collect();
            let mut next: Vec<Perm> = Vec::with_capacity(frontier.len() * reps.len());
            for g in &frontier {
                for rep in &reps {
                    next.push(g.compose(rep));
                }
            }
            frontier = next;
        }
        frontier
    }

    /// Materialize the full element list (image vectors) of `G`, for build-time
    /// differential self-checks. `None` if `|G|` exceeds `cap`.
    #[cfg(test)]
    pub(crate) fn all_elements(&self, cap: usize) -> Option<Vec<Vec<usize>>> {
        if self.order.is_some_and(|o| o > cap as u128) {
            return None;
        }
        Some(
            self.enumerate_elements()
                .into_iter()
                .map(|p| p.img.iter().map(|&x| x as usize).collect())
                .collect(),
        )
    }
}

/// Accumulates the `H = Stab_G(m)` elements discovered by the base-image
/// backtrack into an exact BSGS via the TRUSTED [`build_chain`]. Maintains a
/// running BSGS of the found generators so the backtrack's coset prune can ask
/// "is this base-image prefix already spanned?" without enumerating `G`.
struct StabCollector {
    /// Running BSGS of `⟨found H-generators⟩`. `None` until the first
    /// non-identity stabilizer element is recorded.
    found: Option<Bsgs>,
    n: usize,
}

impl StabCollector {
    /// Record a discovered `h ∈ H` (a concrete permutation). We re-run
    /// `build_chain` over the accumulated strong generators plus `h`; this is
    /// cheap because the number of generators of these small stabilizers is
    /// tiny, and it keeps `|H|` computation on the already-differentially-tested
    /// `build_chain` path (no second order-counting implementation to trust).
    fn add(&mut self, h: Perm) {
        if h.is_identity() {
            return;
        }
        match &mut self.found {
            None => {
                self.found = Some(build_chain(vec![h], self.n));
            }
            Some(chain) => {
                // Fast skip: if `h` already sifts to identity it adds nothing.
                if chain.contains(&h) {
                    return;
                }
                let mut gens = chain.sgs.clone();
                gens.push(h);
                self.found = Some(build_chain(gens, self.n));
            }
        }
    }

    /// May the backtrack prune the subtree rooted at the choice "map `b_level`
    /// to `accum(q)`"? Sound to prune iff the concrete `G`-element
    /// `elem = accum ∘ rep_q` (which realizes this exact base-image prefix on
    /// `b_0..b_level`) already lies in `⟨found⟩`.
    ///
    /// Soundness: every `g ∈ H` explored under this node shares `elem`'s images
    /// on `b_0..b_level`, so `elem⁻¹g` fixes `b_0..b_level` pointwise, i.e.
    /// `elem⁻¹g ∈ H^(level+1)`. The generators of `H^(level+1)` are discovered in
    /// the IDENTITY-prefix branch (mapping each base point to itself), which is
    /// never color- or coset-pruned away; together with `elem ∈ ⟨found⟩` this
    /// gives `g = elem·(elem⁻¹g) ∈ ⟨found⟩`. Hence pruning loses no generator and
    /// the final `⟨found⟩` equals `H` exactly. If `elem ∉ ⟨found⟩` we do NOT
    /// prune (explore) — conservative, never unsound. Returns `false` when
    /// `⟨found⟩` is still empty.
    fn prefix_spanned(&self, accum: &Perm, q: u32, level: usize, parent_levels: &[Level]) -> bool {
        let chain = match &self.found {
            Some(c) => c,
            None => return false,
        };
        // Realize the prefix as a concrete G-element: accum ∘ rep_q maps
        // b_level -> accum(q) (rep_q ∈ G^(level) moves b_level -> q) and agrees
        // with accum on the earlier base points.
        let elem = match parent_levels[level].transversal[q as usize].as_ref() {
            Some(rep) => accum.compose(rep),
            None => return false,
        };
        // Prune iff elem ∈ ⟨found⟩ (membership via the found chain's sift).
        chain.contains(&elem)
    }
}

/// Build a stabilizer chain (BSGS) from `gens` via deterministic
/// Schreier–Sims, computing `|G| = ∏|Δ_i|`.
fn build_chain(gens: Vec<Perm>, n: usize) -> Bsgs {
    let mut bsgs = Bsgs {
        n,
        levels: Vec::new(),
        sgs: Vec::new(),
        order: Some(1),
        elements: Vec::new(),
    };
    for g in gens {
        bsgs.sift(g);
    }
    // Strongness completion: close under Schreier generators until the chain is
    // stable (every Schreier generator strips to identity). Deterministic, no
    // randomization. RESTART the whole scan whenever the chain changes, so
    // every Schreier generator is formed against the CURRENT (fully grown)
    // level state.
    loop {
        let mut changed = false;
        'scan: for i in 0..bsgs.levels.len() {
            let orbit_i = bsgs.levels[i].orbit.clone();
            let gens_i = bsgs.level_gens(i);
            for &p in &orbit_i {
                let u_p = bsgs.levels[i].transversal[p as usize]
                    .clone()
                    .expect("orbit point has a transversal rep");
                for s in &gens_i {
                    let sp = s.apply(p);
                    let u_sp = bsgs.levels[i].transversal[sp as usize]
                        .clone()
                        .expect("image stays in orbit (orbit closed under level gens)");
                    // Schreier generator u_{s(p)}^{-1} ∘ s ∘ u_p ∈ G^(i+1).
                    let schreier = u_sp.inverse().compose(&s.compose(&u_p));
                    if !schreier.is_identity() && bsgs.sift(schreier) {
                        changed = true;
                        break 'scan; // chain grew: restart scan from the top
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    // |G| = ∏|Δ_i| with checked u128 multiplication; None on overflow.
    let mut order: Option<u128> = Some(1);
    for level in &bsgs.levels {
        order = order.and_then(|o| o.checked_mul(level.orbit.len() as u128));
    }
    bsgs.order = order;
    bsgs
}

impl Bsgs {
    /// Sift `g` through the chain from level 0. If a non-identity residue
    /// appears at some level (it maps that level's base point out of its
    /// current fundamental orbit) or below all levels, add the residue to the
    /// flat strong generating set (extending the base with a new point if
    /// needed) and rebuild every affected orbit. Returns `true` iff the chain
    /// changed.
    ///
    /// A residue stripped down to level `i` fixes `b_0..b_{i-1}`, so once
    /// added to the SGS it participates as a generator of every stabilizer
    /// `G^(0)..G^(i)`; hence we rebuild all levels' orbits (cheap for the small
    /// coupled groups in scope).
    fn sift(&mut self, g: Perm) -> bool {
        let mut residue = g;
        let mut i = 0;
        loop {
            if i >= self.levels.len() {
                if residue.is_identity() {
                    return false;
                }
                let moved = (0..self.n as u32)
                    .find(|&p| residue.apply(p) != p)
                    .expect("non-identity residue moves some point");
                self.levels.push(Level::new(moved, self.n));
                self.sgs.push(residue);
                self.rebuild_all_orbits();
                return true;
            }
            let base_i = self.levels[i].base;
            let img = residue.apply(base_i);
            match self.levels[i].transversal[img as usize].clone() {
                Some(u) => {
                    // residue := u^{-1} ∘ residue now fixes base_i; descend.
                    residue = u.inverse().compose(&residue);
                    i += 1;
                }
                None => {
                    // residue ∈ G^(i) and moves b_i out of Δ_i: a new strong
                    // generator. Add to the flat SGS and regrow orbits.
                    self.sgs.push(residue);
                    self.rebuild_all_orbits();
                    return true;
                }
            }
        }
    }

    /// Recompute every level's fundamental orbit and transversal from the flat
    /// SGS (each level uses the SGS elements fixing its earlier base points).
    fn rebuild_all_orbits(&mut self) {
        for level in 0..self.levels.len() {
            self.rebuild_orbit(level);
        }
    }

    /// Recompute the fundamental orbit and transversal at `level` from the
    /// level-`level` stabilizer generators by deterministic BFS from the base.
    fn rebuild_orbit(&mut self, level: usize) {
        let base = self.levels[level].base;
        let n = self.n;
        let mut transversal: Vec<Option<Perm>> = vec![None; n];
        let mut orbit: Vec<u32> = Vec::new();
        transversal[base as usize] = Some(Perm::identity(n));
        orbit.push(base);
        let gens = self.level_gens(level);
        let mut head = 0;
        while head < orbit.len() {
            let p = orbit[head];
            head += 1;
            let u_p = transversal[p as usize].clone().expect("queued has rep");
            for s in &gens {
                let q = s.apply(p);
                if transversal[q as usize].is_none() {
                    transversal[q as usize] = Some(s.compose(&u_p));
                    orbit.push(q);
                }
            }
        }
        self.levels[level].transversal = transversal;
        self.levels[level].orbit = orbit;
    }
}

/// `true` iff applying `g` to `marking` leaves it unchanged: `g ∈ Stab(m)`.
/// Action: `applied[i] = marking[g(i)]`; equals `marking` iff
/// `marking[g(i)] == marking[i]` for all `i`.
#[inline]
fn preserves_marking(g: &Perm, marking: &[u64]) -> bool {
    (0..marking.len()).all(|i| marking[g.apply(i as u32) as usize] == marking[i])
}

/// The orbit of `point` under `⟨gens⟩` (BFS over `0..n`), in discovery order.
fn orbit_points(point: u32, gens: &[&Perm], n: usize) -> Vec<u32> {
    let mut seen = vec![false; n];
    let mut orbit = vec![point];
    seen[point as usize] = true;
    let mut head = 0;
    while head < orbit.len() {
        let p = orbit[head];
        head += 1;
        for g in gens {
            let q = g.apply(p);
            if !seen[q as usize] {
                seen[q as usize] = true;
                orbit.push(q);
            }
        }
    }
    orbit
}

/// Coset representatives (a transversal) for the orbit of `point` under
/// `⟨gens⟩`: for every point `q` in the orbit, a permutation `rep ∈ ⟨gens⟩`
/// with `rep(point) = q`. Computed by a Schreier-vector BFS. Returns one rep
/// per orbit point (including `point` itself, mapped by the identity).
fn stabilizer_coset_reps(point: u32, gens: &[&Perm], n: usize) -> Vec<Perm> {
    let mut transversal: Vec<Option<Perm>> = vec![None; n];
    let mut orbit = vec![point];
    transversal[point as usize] = Some(Perm::identity(n));
    let mut head = 0;
    while head < orbit.len() {
        let p = orbit[head];
        head += 1;
        let u_p = transversal[p as usize].clone().expect("queued has rep");
        for g in gens {
            let q = g.apply(p);
            if transversal[q as usize].is_none() {
                // rep mapping point -> q is g ∘ u_p (u_p maps point->p, g maps p->q)
                transversal[q as usize] = Some(g.compose(&u_p));
                orbit.push(q);
            }
        }
    }
    orbit
        .into_iter()
        .map(|q| transversal[q as usize].clone().expect("orbit has rep"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Brute-force the full group from generators by BFS closure (small
    /// groups), returning the set of image vectors.
    fn brute_group(generators: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
        let identity: Vec<usize> = (0..n).collect();
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        seen.insert(identity.clone());
        let mut frontier = vec![identity];
        while let Some(g) = frontier.pop() {
            for gen in generators {
                // compose: (gen ∘ g)[i] = gen[g[i]]
                let comp: Vec<usize> = (0..n).map(|i| gen[g[i]]).collect();
                if seen.insert(comp.clone()) {
                    frontier.push(comp);
                }
            }
        }
        seen.into_iter().collect()
    }

    /// Brute-force orbit of a marking under the group (action applied[i]=m[g[i]]).
    fn brute_orbit(group: &[Vec<usize>], marking: &[u64]) -> HashSet<Vec<u64>> {
        let n = marking.len();
        let mut orbit = HashSet::new();
        for g in group {
            let img: Vec<u64> = (0..n).map(|i| marking[g[i]]).collect();
            orbit.insert(img);
        }
        orbit
    }

    /// Brute-force `|Stab_G(m)|` = number of group elements fixing the marking
    /// under the action `applied(g,m)[i] = m[g(i)]`, i.e. `m[g(i)] == m[i] ∀i`.
    fn brute_stabilizer_order(group: &[Vec<usize>], marking: &[u64]) -> u128 {
        let n = marking.len();
        group
            .iter()
            .filter(|g| (0..n).all(|i| marking[g[i]] == marking[i]))
            .count() as u128
    }

    /// Enumerate every marking in `{0..vals}^n` and assert, for each, that the
    /// regime-B pieces agree EXACTLY with the enumerative / brute-force oracles:
    ///   * `stabilizer_order(m)` == |{g ∈ G : g·m = m}| (brute force),
    ///   * `orbit_size(m)` (regime-B |G|/|Stab|) == `orbit_size_enumerative(m)`
    ///     == |brute-force orbit|,
    ///   * `|G| == |Stab(m)| · |orbit(m)|` (orbit–stabilizer holds exactly).
    ///
    /// This is the key soundness oracle for regime B over ALL markings.
    fn assert_regime_b_matches_oracles(generators: &[Vec<usize>], n: usize, vals: u64) {
        let bsgs = Bsgs::build(generators, n).expect("non-empty group");
        let group = brute_group(generators, n);
        let order = bsgs.order().expect("order fits");
        let total: u64 = vals.pow(n as u32);
        for code in 0..total {
            let mut c = code;
            let marking: Vec<u64> = (0..n)
                .map(|_| {
                    let d = c % vals;
                    c /= vals;
                    d
                })
                .collect();

            let brute_stab = brute_stabilizer_order(&group, &marking);
            let got_stab = bsgs
                .stabilizer_order(&marking)
                .expect("|Stab| fits u128 for small groups");
            assert_eq!(
                got_stab, brute_stab,
                "stabilizer_order({marking:?}) must equal brute-force |Stab| for gens {generators:?}",
            );

            let brute_orbit_sz = brute_orbit(&group, &marking).len() as u64;
            let enum_orbit = bsgs
                .orbit_size_enumerative(&marking)
                .expect("enumerative orbit fits");
            // The PUBLIC orbit_size (hybrid dispatch) and the EXPLICIT regime-B
            // quotient |G|/|Stab| (threshold-independent) must BOTH equal the
            // enumerative/brute-force orbit. Computing |G|/|Stab| directly here
            // exercises the regime-B division even on small groups (where the
            // hybrid would otherwise route orbit_size to the enumerative path).
            let dispatched = bsgs.orbit_size(&marking).expect("orbit_size fits");
            assert_eq!(order % got_stab, 0, "Lagrange: |Stab| divides |G|");
            let regime_b_quotient = (order / got_stab) as u64;
            assert_eq!(
                enum_orbit, brute_orbit_sz,
                "enumerative orbit must match brute force for {marking:?}",
            );
            assert_eq!(
                regime_b_quotient, enum_orbit,
                "regime-B |G|/|Stab| must EQUAL enumerative orbit size for {marking:?} (gens {generators:?})",
            );
            assert_eq!(
                dispatched, enum_orbit,
                "hybrid orbit_size must EQUAL enumerative orbit size for {marking:?}",
            );
            // Orbit–stabilizer identity holds exactly.
            assert_eq!(
                order,
                got_stab * regime_b_quotient as u128,
                "|G| = |Stab|·|orbit| must hold exactly for {marking:?}",
            );
        }
    }

    #[test]
    fn regime_b_vs_enumerative_all_markings_symmetric_groups() {
        // Sym(3), Sym(4), Sym(5) via adjacent transpositions.
        for n in 3..=5usize {
            let mut gens = Vec::new();
            for i in 0..n - 1 {
                let mut g: Vec<usize> = (0..n).collect();
                g.swap(i, i + 1);
                gens.push(g);
            }
            // Use enough distinct values to exercise non-trivial stabilizers and
            // the full-symmetric all-distinct orbit (n! ≤ 120 fits u64).
            let vals = if n <= 4 { 4 } else { 3 };
            assert_regime_b_matches_oracles(&gens, n, vals as u64);
        }
    }

    #[test]
    fn regime_b_vs_enumerative_all_markings_cyclic_and_dihedral() {
        // Cyclic Z_n and dihedral D_n for several n.
        for n in 3..=6usize {
            let rot: Vec<usize> = (1..=n).map(|i| i % n).collect();
            assert_regime_b_matches_oracles(std::slice::from_ref(&rot), n, 3);
            // Dihedral: rotation + reflection (reverse all but point 0).
            let mut refl: Vec<usize> = (0..n).collect();
            refl[1..].reverse();
            assert_regime_b_matches_oracles(&[rot, refl], n, 3);
        }
    }

    #[test]
    fn regime_b_vs_enumerative_all_markings_coupled_s4_diagonal() {
        // S4 acting DIAGONALLY on two size-4 orbits permuted TOGETHER (|G|=24).
        let swap01 = vec![1, 0, 2, 3, 5, 4, 6, 7];
        let swap12 = vec![0, 2, 1, 3, 4, 6, 5, 7];
        let swap23 = vec![0, 1, 3, 2, 4, 5, 7, 6];
        // Binary markings over 8 places (256 total) exhaustively, plus a denser
        // value range on the first orbit via a small ternary sweep is covered by
        // the binary pass already discriminating all stabilizer shapes here.
        assert_regime_b_matches_oracles(&[swap01, swap12, swap23], 8, 2);
    }

    #[test]
    fn regime_b_vs_enumerative_all_markings_wreath_like() {
        // Wreath-like Z2 ≀ Z2 of order 8 on 4 points.
        let a = vec![1, 0, 2, 3];
        let b = vec![0, 1, 3, 2];
        let c = vec![2, 3, 0, 1];
        assert_regime_b_matches_oracles(&[a, b, c], 4, 4);
    }

    /// Sym(7) (|G|=5040 > ENUMERATIVE_ORBIT_GROUP_BOUND) so the PUBLIC
    /// `orbit_size` takes the REGIME-B `|G|/|Stab|` branch. Exhaustively verify
    /// over a binary cube that the dispatched orbit_size and `stabilizer_order`
    /// match brute force, the orbit–stabilizer identity holds, AND the regime-B
    /// weights partition the cube (`Σ_reps |orbit(rep)| == |R|`). This is the
    /// large-group case the regime-B branch exists for.
    #[test]
    fn regime_b_branch_exact_on_large_group_sym7() {
        let mut gens = Vec::new();
        for i in 0..6 {
            let mut g: Vec<usize> = (0..7).collect();
            g.swap(i, i + 1);
            gens.push(g);
        }
        let bsgs = Bsgs::build(&gens, 7).unwrap();
        let group = brute_group(&gens, 7);
        let order = bsgs.order().unwrap();
        assert_eq!(order, 5040);
        assert!(
            order > Bsgs::ENUMERATIVE_ORBIT_GROUP_BOUND,
            "must hit regime-B"
        );
        let mut reps: HashSet<Vec<u64>> = HashSet::new();
        let mut total = 0u64;
        let mut weight_sum = 0u64;
        for bits in 0u32..128 {
            let marking: Vec<u64> = (0..7).map(|i| ((bits >> i) & 1) as u64).collect();
            total += 1;
            let brute_stab = brute_stabilizer_order(&group, &marking);
            let got_stab = bsgs.stabilizer_order(&marking).unwrap();
            assert_eq!(got_stab, brute_stab, "Sym7 |Stab|({marking:?})");
            let brute_orbit_sz = brute_orbit(&group, &marking).len() as u64;
            let dispatched = bsgs.orbit_size(&marking).unwrap();
            assert_eq!(
                dispatched, brute_orbit_sz,
                "Sym7 regime-B orbit({marking:?})"
            );
            assert_eq!(
                order,
                got_stab * dispatched as u128,
                "Sym7 |G|=|Stab|·|orbit|"
            );
            let canon = bsgs.canonical_image(&marking);
            assert_eq!(canon, brute_min_image(&group, &marking));
            if reps.insert(canon.clone()) {
                weight_sum += bsgs.orbit_size(&canon).unwrap();
            }
        }
        assert_eq!(weight_sum, total, "Sym7 regime-B Σ|orbit|=|R|");
    }

    #[test]
    fn regime_b_partition_sigma_orbit_equals_total_over_cube() {
        // Σ_reps |orbit(rep)| (regime-B weights) == |R| (total markings) for
        // several groups over a small value cube — the partition identity that
        // makes |R| exact end to end.
        let cases: Vec<(Vec<Vec<usize>>, usize, u64)> = vec![
            // Sym(4)
            (
                vec![vec![1, 0, 2, 3], vec![0, 2, 1, 3], vec![0, 1, 3, 2]],
                4,
                3,
            ),
            // Cyclic Z6
            (vec![vec![1, 2, 3, 4, 5, 0]], 6, 2),
            // Coupled diagonal S4 on two size-4 orbits
            (
                vec![
                    vec![1, 0, 2, 3, 5, 4, 6, 7],
                    vec![0, 2, 1, 3, 4, 6, 5, 7],
                    vec![0, 1, 3, 2, 4, 5, 7, 6],
                ],
                8,
                2,
            ),
        ];
        for (gens, n, vals) in cases {
            let bsgs = Bsgs::build(&gens, n).unwrap();
            let group = brute_group(&gens, n);
            let mut reps: HashSet<Vec<u64>> = HashSet::new();
            let mut total = 0u64;
            let mut weight_sum = 0u64;
            let count: u64 = (vals).pow(n as u32);
            for code in 0..count {
                let mut c = code;
                let marking: Vec<u64> = (0..n)
                    .map(|_| {
                        let d = c % vals;
                        c /= vals;
                        d
                    })
                    .collect();
                total += 1;
                let canon = bsgs.canonical_image(&marking);
                assert_eq!(canon, brute_min_image(&group, &marking));
                if reps.insert(canon.clone()) {
                    weight_sum += bsgs.orbit_size(&canon).unwrap();
                }
            }
            assert_eq!(
                weight_sum, total,
                "Σ_reps |orbit(rep)| (regime-B) must equal |R| for gens {gens:?}",
            );
        }
    }

    /// Brute-force lex-min image, matching `canonical_image`'s action
    /// `image[j] = marking[g^{-1}(j)]`.
    fn brute_min_image(group: &[Vec<usize>], marking: &[u64]) -> Vec<u64> {
        let n = marking.len();
        let mut best: Option<Vec<u64>> = None;
        for g in group {
            let mut inv = vec![0usize; n];
            for (i, &j) in g.iter().enumerate() {
                inv[j] = i;
            }
            let img: Vec<u64> = (0..n).map(|j| marking[inv[j]]).collect();
            match &best {
                Some(b) if &img >= b => {}
                _ => best = Some(img),
            }
        }
        best.unwrap()
    }

    fn assert_order(generators: &[Vec<usize>], n: usize) -> Bsgs {
        let bsgs = Bsgs::build(generators, n).expect("non-empty");
        let group = brute_group(generators, n);
        assert_eq!(
            bsgs.order().expect("order fits"),
            group.len() as u128,
            "BSGS |G| must equal brute-force group order for gens {generators:?}",
        );
        bsgs
    }

    #[test]
    fn order_single_transposition() {
        let gens = vec![vec![1, 0, 2]];
        assert_order(&gens, 3);
    }

    #[test]
    fn order_sym3() {
        let gens = vec![vec![1, 0, 2], vec![0, 2, 1]];
        assert_order(&gens, 3);
    }

    #[test]
    fn order_sym4() {
        let gens = vec![vec![1, 0, 2, 3], vec![0, 2, 1, 3], vec![0, 1, 3, 2]];
        assert_order(&gens, 4);
    }

    #[test]
    fn order_sym5_full() {
        // Sym(5) via adjacent transpositions: |G| = 120.
        let mut gens = Vec::new();
        for i in 0..4 {
            let mut g: Vec<usize> = (0..5).collect();
            g.swap(i, i + 1);
            gens.push(g);
        }
        assert_order(&gens, 5);
    }

    #[test]
    fn order_cyclic_z5() {
        let gens = vec![vec![1, 2, 3, 4, 0]];
        assert_order(&gens, 5);
    }

    #[test]
    fn order_dihedral_d5() {
        let rot = vec![1, 2, 3, 4, 0];
        let refl = vec![0, 4, 3, 2, 1];
        assert_order(&[rot, refl], 5);
    }

    #[test]
    fn order_coupled_s4_diagonal_two_orbits() {
        // S4 acting DIAGONALLY on two size-4 orbits permuted TOGETHER.
        // |G| = 24 (not 24*24).
        let swap01 = vec![1, 0, 2, 3, 5, 4, 6, 7];
        let swap12 = vec![0, 2, 1, 3, 4, 6, 5, 7];
        let swap23 = vec![0, 1, 3, 2, 4, 5, 7, 6];
        assert_order(&[swap01, swap12, swap23], 8);
    }

    #[test]
    fn order_wreath_like_coupling() {
        // Two independent size-2 swaps PLUS a generator swapping the two blocks
        // together: a wreath-like Z2 ≀ Z2 of order 8.
        let a = vec![1, 0, 2, 3]; // swap block-0 internally
        let b = vec![0, 1, 3, 2]; // swap block-1 internally
        let c = vec![2, 3, 0, 1]; // swap the two blocks
        assert_order(&[a, b, c], 4);
    }

    #[test]
    fn orbit_size_matches_brute_force_sym4() {
        let gens = vec![vec![1, 0, 2, 3], vec![0, 2, 1, 3], vec![0, 1, 3, 2]];
        let bsgs = assert_order(&gens, 4);
        let group = brute_group(&gens, 4);
        for marking in [
            [1u64, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [3, 2, 1, 0],
            [2, 2, 1, 1],
        ] {
            let expected = brute_orbit(&group, &marking).len() as u64;
            let got = bsgs.orbit_size(&marking).expect("fits");
            assert_eq!(got, expected, "orbit_size({marking:?})");
        }
    }

    #[test]
    fn orbit_size_matches_brute_force_coupled_s4() {
        let swap01 = vec![1, 0, 2, 3, 5, 4, 6, 7];
        let swap12 = vec![0, 2, 1, 3, 4, 6, 5, 7];
        let swap23 = vec![0, 1, 3, 2, 4, 5, 7, 6];
        let gens = [swap01, swap12, swap23];
        let bsgs = assert_order(&gens, 8);
        let group = brute_group(&gens, 8);
        for marking in [
            [1u64, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 9, 0, 0, 0],
            [1, 0, 0, 0, 0, 9, 0, 0],
            [1, 1, 0, 0, 2, 2, 0, 0],
            [3, 2, 1, 0, 0, 0, 0, 0],
            [5, 6, 7, 8, 1, 2, 3, 4],
        ] {
            let expected = brute_orbit(&group, &marking).len() as u64;
            let got = bsgs.orbit_size(&marking).expect("fits");
            assert_eq!(got, expected, "coupled orbit_size({marking:?})");
        }
    }

    #[test]
    fn orbit_size_matches_brute_force_cyclic_z6() {
        let gens = vec![vec![1, 2, 3, 4, 5, 0]];
        let bsgs = assert_order(&gens, 6);
        let group = brute_group(&gens, 6);
        for marking in [
            [1u64, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [2, 1, 2, 1, 2, 1],
        ] {
            let expected = brute_orbit(&group, &marking).len() as u64;
            let got = bsgs.orbit_size(&marking).expect("fits");
            assert_eq!(got, expected, "cyclic orbit_size({marking:?})");
        }
    }

    #[test]
    fn orbit_size_matches_brute_force_dihedral_d4() {
        // D4 on a square (4 points): rotation + reflection, |G| = 8.
        let rot = vec![1, 2, 3, 0];
        let refl = vec![0, 3, 2, 1];
        let bsgs = assert_order(&[rot, refl], 4);
        let group = brute_group(&[vec![1, 2, 3, 0], vec![0, 3, 2, 1]], 4);
        for marking in [
            [1u64, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 2, 1, 2],
            [3, 2, 1, 0],
        ] {
            let expected = brute_orbit(&group, &marking).len() as u64;
            let got = bsgs.orbit_size(&marking).expect("fits");
            assert_eq!(got, expected, "dihedral orbit_size({marking:?})");
        }
    }

    #[test]
    fn canonical_image_is_one_rep_per_orbit_sym3() {
        let gens = vec![vec![1, 0, 2], vec![0, 2, 1]];
        let bsgs = Bsgs::build(&gens, 3).unwrap();
        let group = brute_group(&gens, 3);
        for marking in [[1u64, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 0], [1, 2, 3]] {
            let got = bsgs.canonical_image(&marking);
            let expected = brute_min_image(&group, &marking);
            assert_eq!(got, expected, "canonical_image({marking:?})");
        }
        let base = [2u64, 1, 0];
        let canon = bsgs.canonical_image(&base);
        for g in &group {
            let img: Vec<u64> = (0..3).map(|i| base[g[i]]).collect();
            assert_eq!(bsgs.canonical_image(&img), canon);
        }
    }

    #[test]
    fn canonical_image_coupled_one_rep_per_orbit() {
        let swap01 = vec![1, 0, 2, 3, 5, 4, 6, 7];
        let swap12 = vec![0, 2, 1, 3, 4, 6, 5, 7];
        let swap23 = vec![0, 1, 3, 2, 4, 5, 7, 6];
        let gens = [swap01, swap12, swap23];
        let bsgs = Bsgs::build(&gens, 8).unwrap();
        let group = brute_group(&gens, 8);
        let mut reps: HashSet<Vec<u64>> = HashSet::new();
        let mut total_markings = 0u64;
        let mut weight_sum = 0u64;
        for bits in 0u32..256 {
            let marking: Vec<u64> = (0..8).map(|i| ((bits >> i) & 1) as u64).collect();
            total_markings += 1;
            let canon = bsgs.canonical_image(&marking);
            assert_eq!(canon, brute_min_image(&group, &marking));
            if reps.insert(canon.clone()) {
                weight_sum += bsgs.orbit_size(&canon).unwrap();
            }
        }
        assert_eq!(
            weight_sum, total_markings,
            "Σ_reps |orbit(rep)| must equal |R| (partition property)",
        );
    }

    #[test]
    fn partition_property_cyclic_z6_over_binary_cube() {
        let gens = vec![vec![1, 2, 3, 4, 5, 0]];
        let bsgs = Bsgs::build(&gens, 6).unwrap();
        let group = brute_group(&gens, 6);
        let mut reps: HashSet<Vec<u64>> = HashSet::new();
        let mut total = 0u64;
        let mut weight_sum = 0u64;
        for bits in 0u32..64 {
            let marking: Vec<u64> = (0..6).map(|i| ((bits >> i) & 1) as u64).collect();
            total += 1;
            let canon = bsgs.canonical_image(&marking);
            assert_eq!(canon, brute_min_image(&group, &marking));
            if reps.insert(canon.clone()) {
                weight_sum += bsgs.orbit_size(&canon).unwrap();
            }
        }
        assert_eq!(weight_sum, total, "cyclic partition property");
    }

    #[test]
    fn stabilizer_full_symmetric_reduces_to_multinomial() {
        let gens = vec![vec![1, 0, 2, 3], vec![0, 2, 1, 3], vec![0, 1, 3, 2]];
        let bsgs = Bsgs::build(&gens, 4).unwrap();
        // [2,1,1,0]: 4!/(1!·2!·1!) = 12.
        assert_eq!(bsgs.orbit_size(&[2, 1, 1, 0]).unwrap(), 12);
        // [1,1,1,1]: 4!/4! = 1.
        assert_eq!(bsgs.orbit_size(&[1, 1, 1, 1]).unwrap(), 1);
        // [3,2,1,0]: 4! = 24.
        assert_eq!(bsgs.orbit_size(&[3, 2, 1, 0]).unwrap(), 24);
    }

    #[test]
    fn orbit_size_overflow_fails_closed() {
        // Sym(25): all-distinct marking -> orbit 25! ~ 1.5e25 > u64::MAX -> None.
        let mut gens = Vec::new();
        for i in 0..24 {
            let mut g: Vec<usize> = (0..25).collect();
            g.swap(i, i + 1);
            gens.push(g);
        }
        let bsgs = Bsgs::build(&gens, 25).unwrap();
        let marking: Vec<u64> = (0..25).map(|i| i as u64).collect();
        assert_eq!(bsgs.orbit_size(&marking), None);
        // Symmetric marking -> orbit 1.
        let flat: Vec<u64> = vec![1; 25];
        assert_eq!(bsgs.orbit_size(&flat), Some(1));
    }

    #[test]
    fn degree_reported() {
        let gens = vec![vec![1, 0, 2, 3]];
        let bsgs = Bsgs::build(&gens, 4).unwrap();
        assert_eq!(bsgs.degree(), 4);
    }

    /// The PRUNED `canonical_image` must equal the ENUMERATIVE minimum over the
    /// full group, on every marking — across several group shapes including a
    /// larger Sym(5) where pruning matters. This is the key soundness oracle:
    /// the canonical form must be the true lex-min (exactly one rep per orbit).
    fn assert_canonical_matches_enumeration(
        generators: &[Vec<usize>],
        n: usize,
        markings: &[Vec<u64>],
    ) {
        let bsgs = Bsgs::build(generators, n).unwrap();
        let group = brute_group(generators, n);
        for m in markings {
            let pruned = bsgs.canonical_image(m);
            let enumerated = brute_min_image(&group, m);
            assert_eq!(
                pruned, enumerated,
                "pruned canonical_image must equal enumerative lex-min for {m:?} (gens {generators:?})",
            );
        }
    }

    #[test]
    fn canonical_image_pruned_matches_enumeration_sym5() {
        let mut gens = Vec::new();
        for i in 0..4 {
            let mut g: Vec<usize> = (0..5).collect();
            g.swap(i, i + 1);
            gens.push(g);
        }
        let markings: Vec<Vec<u64>> = vec![
            vec![1, 0, 0, 0, 0],
            vec![2, 1, 0, 0, 0],
            vec![3, 2, 1, 0, 0],
            vec![1, 2, 3, 4, 5],
            vec![2, 2, 1, 1, 0],
            vec![5, 4, 3, 2, 1],
            vec![0, 0, 0, 0, 7],
        ];
        assert_canonical_matches_enumeration(&gens, 5, &markings);
    }

    #[test]
    fn canonical_image_pruned_matches_enumeration_coupled_and_cyclic() {
        // Coupled diagonal S4 on two size-4 orbits.
        let coupled = [
            vec![1, 0, 2, 3, 5, 4, 6, 7],
            vec![0, 2, 1, 3, 4, 6, 5, 7],
            vec![0, 1, 3, 2, 4, 5, 7, 6],
        ];
        let coupled_markings: Vec<Vec<u64>> = vec![
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            vec![3, 2, 1, 0, 9, 8, 7, 6],
            vec![0, 0, 0, 0, 1, 2, 3, 4],
            vec![2, 1, 0, 0, 0, 0, 1, 2],
        ];
        assert_canonical_matches_enumeration(&coupled, 8, &coupled_markings);

        // Cyclic Z6 and dihedral D5.
        let cyc = [vec![1, 2, 3, 4, 5, 0]];
        let cyc_markings: Vec<Vec<u64>> = vec![
            vec![1, 0, 0, 0, 0, 0],
            vec![1, 0, 1, 0, 1, 0],
            vec![2, 1, 0, 0, 1, 2],
            vec![5, 4, 3, 2, 1, 0],
        ];
        assert_canonical_matches_enumeration(&cyc, 6, &cyc_markings);

        let d5 = [vec![1, 2, 3, 4, 0], vec![0, 4, 3, 2, 1]];
        let d5_markings: Vec<Vec<u64>> = vec![
            vec![1, 0, 0, 0, 0],
            vec![2, 1, 0, 1, 2],
            vec![3, 1, 4, 1, 5],
        ];
        assert_canonical_matches_enumeration(&d5, 5, &d5_markings);
    }

    /// Full partition + canonical-consistency check on Sym(5) over a small
    /// integer cube: every marking canonicalizes to the enumerative lex-min,
    /// markings in the same orbit collapse to the SAME rep, and
    /// `Σ_reps |orbit(rep)| == |R|`.
    #[test]
    fn partition_and_canonical_consistency_sym5_cube() {
        let mut gens = Vec::new();
        for i in 0..4 {
            let mut g: Vec<usize> = (0..5).collect();
            g.swap(i, i + 1);
            gens.push(g);
        }
        let bsgs = Bsgs::build(&gens, 5).unwrap();
        let group = brute_group(&gens, 5);
        let mut reps: HashSet<Vec<u64>> = HashSet::new();
        let mut total = 0u64;
        let mut weight_sum = 0u64;
        // values in {0,1,2} on 5 places -> 243 markings.
        for code in 0u32..243 {
            let mut c = code;
            let marking: Vec<u64> = (0..5)
                .map(|_| {
                    let d = (c % 3) as u64;
                    c /= 3;
                    d
                })
                .collect();
            total += 1;
            let canon = bsgs.canonical_image(&marking);
            assert_eq!(canon, brute_min_image(&group, &marking));
            // Orbit-consistency: any group image canonicalizes identically.
            for g in &group {
                let img: Vec<u64> = (0..5).map(|i| marking[g[i]]).collect();
                assert_eq!(bsgs.canonical_image(&img), canon);
            }
            if reps.insert(canon.clone()) {
                weight_sum += bsgs.orbit_size(&canon).unwrap();
            }
        }
        assert_eq!(weight_sum, total, "Σ_reps |orbit(rep)| must equal |R|");
    }

    /// The FAST cached lex-min, the pruned BACKTRACK, and the brute-force
    /// enumeration must agree on EVERY marking of a value cube, for several
    /// group shapes — and the element cache must actually be populated (so the
    /// default coupled path really takes the fast branch). This is the soundness
    /// oracle for the new `canonical_image` fast path: a wrong canonical form is
    /// a wrong orbit partition is a wrong StateSpace count.
    #[test]
    fn cached_canonical_image_equals_backtrack_and_brute() {
        let cases: Vec<(Vec<Vec<usize>>, usize, u64)> = vec![
            // Sym(5) via adjacent transpositions, |G|=120.
            (
                (0..4)
                    .map(|i| {
                        let mut g: Vec<usize> = (0..5).collect();
                        g.swap(i, i + 1);
                        g
                    })
                    .collect(),
                5,
                3,
            ),
            // Coupled diagonal S4 on two size-4 orbits, |G|=24.
            (
                vec![
                    vec![1, 0, 2, 3, 5, 4, 6, 7],
                    vec![0, 2, 1, 3, 4, 6, 5, 7],
                    vec![0, 1, 3, 2, 4, 5, 7, 6],
                ],
                8,
                2,
            ),
            // Cyclic Z6.
            (vec![vec![1, 2, 3, 4, 5, 0]], 6, 3),
            // Dihedral D5.
            (vec![vec![1, 2, 3, 4, 0], vec![0, 4, 3, 2, 1]], 5, 3),
        ];
        for (gens, n, vals) in cases {
            let bsgs = Bsgs::build(&gens, n).unwrap();
            assert!(
                !bsgs.elements.is_empty(),
                "element cache must be populated for |G|={:?} (gens {gens:?})",
                bsgs.order(),
            );
            let group = brute_group(&gens, n);
            let total: u64 = vals.pow(n as u32);
            for code in 0..total {
                let mut c = code;
                let marking: Vec<u64> = (0..n)
                    .map(|_| {
                        let d = c % vals;
                        c /= vals;
                        d
                    })
                    .collect();
                let cached = bsgs.canonical_image_cached(&marking);
                let backtrack = bsgs.canonical_image_backtrack(&marking);
                let brute = brute_min_image(&group, &marking);
                assert_eq!(cached, brute, "cached lex-min vs brute for {marking:?}");
                assert_eq!(
                    cached, backtrack,
                    "cached lex-min vs backtrack for {marking:?}",
                );
            }
        }
    }

    /// MICRO-BENCHMARK (run with `--ignored --nocapture`): per-call wall time of
    /// regime-B `orbit_size` (|G|/|Stab|) vs the enumerative `orbit_size_enumerative`
    /// on a coupled Anderson-like group, to quantify the regime-B speedup. Not a
    /// correctness assertion — values are differentially checked elsewhere.
    #[test]
    #[ignore = "timing micro-benchmark; run explicitly with --ignored --nocapture"]
    fn bench_regime_b_vs_enumerative_orbit_size() {
        use std::time::Instant;
        // Coupled diagonal S4 on TWO size-4 orbits (|G|=24), several markings of
        // varied stabilizer size; an Anderson-shaped coupling.
        let gens = [
            vec![1, 0, 2, 3, 5, 4, 6, 7],
            vec![0, 2, 1, 3, 4, 6, 5, 7],
            vec![0, 1, 3, 2, 4, 5, 7, 6],
        ];
        let bsgs = Bsgs::build(&gens, 8).unwrap();
        let markings: Vec<Vec<u64>> = vec![
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 1, 0, 0, 1, 1, 0, 0],
            vec![3, 2, 1, 0, 9, 8, 7, 6],
            vec![1, 1, 1, 0, 0, 0, 0, 0],
        ];

        // Also bench a LARGE group (Sym(7), |G|=5040, AirplaneLD-shaped) where
        // the enumerative O(|G|) BFS should blow up but regime-B |G|/|Stab| stays
        // bounded — the regime where regime-B is supposed to win.
        let mut big_gens = Vec::new();
        for i in 0..6 {
            let mut g: Vec<usize> = (0..7).collect();
            g.swap(i, i + 1);
            big_gens.push(g);
        }
        let big = Bsgs::build(&big_gens, 7).unwrap();
        let big_markings: Vec<Vec<u64>> = vec![
            vec![1, 0, 0, 0, 0, 0, 0], // large orbit (7), tiny stab
            vec![6, 5, 4, 3, 2, 1, 0], // all distinct: orbit 5040, stab 1
            vec![1, 1, 1, 0, 0, 0, 0], // mixed
        ];
        let big_iters = 5_000;
        let bt0 = Instant::now();
        let mut ba = 0u64;
        for _ in 0..big_iters {
            for m in &big_markings {
                ba += big.orbit_size(m).unwrap();
            }
        }
        let big_b = bt0.elapsed();
        let bt1 = Instant::now();
        let mut be = 0u64;
        for _ in 0..big_iters {
            for m in &big_markings {
                be += big.orbit_size_enumerative(m).unwrap();
            }
        }
        let big_e = bt1.elapsed();
        assert_eq!(ba, be);
        let bcalls = (big_iters * big_markings.len()) as f64;
        eprintln!(
            "Sym(7) |G|=5040 regime-B: {:.1} ns/call | enumerative: {:.1} ns/call | speedup {:.2}x",
            big_b.as_nanos() as f64 / bcalls,
            big_e.as_nanos() as f64 / bcalls,
            big_e.as_nanos() as f64 / big_b.as_nanos() as f64,
        );

        let iters = 200_000;
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            for m in &markings {
                acc += bsgs.orbit_size(m).unwrap();
            }
        }
        let regime_b = t0.elapsed();
        let t1 = Instant::now();
        let mut acc2 = 0u64;
        for _ in 0..iters {
            for m in &markings {
                acc2 += bsgs.orbit_size_enumerative(m).unwrap();
            }
        }
        let enumv = t1.elapsed();
        assert_eq!(acc, acc2, "both must agree");
        let calls = (iters * markings.len()) as f64;
        eprintln!(
            "regime-B orbit_size: {:.3} ns/call | enumerative: {:.3} ns/call | speedup {:.2}x",
            regime_b.as_nanos() as f64 / calls,
            enumv.as_nanos() as f64 / calls,
            enumv.as_nanos() as f64 / regime_b.as_nanos() as f64,
        );
    }
}
