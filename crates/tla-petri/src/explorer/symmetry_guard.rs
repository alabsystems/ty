// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness guard for place-swap symmetry reduction.
//!
//! The canonicalizer in [`super::symmetry`] sorts marking values within each
//! discovered place orbit. By the proof in
//! `docs/theorems/2026-05-26-place-swap-symmetry-soundness.md` §2.5, this is
//! sound only for property predicates that are σ-invariant under every
//! permutation in the symmetric group on each orbit.
//!
//! Examinations that ask about specific places (UpperBounds, ReachabilityCardinality,
//! ReachabilityFireability, LTL*, CTL*) generally have predicates that distinguish
//! places within an orbit, breaking σ-invariance. Applying canonicalization to
//! their search is unsound.
//!
//! **QuasiLiveness and Liveness are also excluded.** They aggregate a
//! *per-transition* predicate (`enabled(t)` reachability for L1/QuasiLiveness;
//! `AG EF enabled(t)` for L4/Liveness). `enabled(t)` for a *specific*
//! transition is NOT σ-invariant: a place-orbit permutation σ maps `t` to a
//! *different* transition σ(t), so a marking enabling `t` is orbit-equivalent
//! to one enabling σ(t), not `t`. A canonicalizing BFS only ever visits
//! canonical orbit representatives and (via `successors.rs`) records the
//! transition index enabled *there*, so a transition enabled solely at a
//! non-canonical orbit member is never observed firing — a false-negative
//! ("not quasi-live") verdict worth −16 MCC points. Counterexample (symmetric
//! orbit `{p0,p1}` fed by source `s`; m0=[0,0,1]; t_load0:s→p0, t_load1:s→p1,
//! t_use0:p0→, t_use1:p1→): the net is quasi-live, but ascending orbit
//! canonicalization collapses the `p0`-high and `p1`-high markings, so only
//! one of `t_use0`/`t_use1` is ever observed and the verdict flips to False.
//! See the regression test
//! `examination_non_property::liveness::test_quasi_liveness_canonicalization_records_all_orbit_transitions`.
//! QuasiLiveness was an *active* bug (its verdict path uses the canonicalizing
//! observer). Liveness is excluded as defense-in-depth: its current L4 verdict
//! path uses only structural analysis, BMC, mu-calculus, and the *full*-graph
//! SCC check (none canonicalize), but the identical σ-invariance failure would
//! make a future observer-based L4 fast-path unsound.
//!
//! **StateSpace is ENABLED via an exact orbit-quotient count.** Every value
//! StateSpace publishes (state count, edge count, max-token-in-place,
//! max-token-sum) is σ-invariant. The two cardinalities (|R|, |E|) would be
//! collapsed by orbit dedup — the explorer issues one callback per canonical
//! representative — but they are recovered EXACTLY by weighting each
//! representative by the size of its place-symmetry orbit:
//!   |R| = Σ_reps |orbit(rep)|              (orbits partition R), and
//!   |E| = Σ_reps |orbit(rep)|·deg(rep)     (using the SOURCE orbit size).
//!
//! The orbit size and the canonical form are BOTH dispatched on a single
//! `count_mode` field of `PetriCanonicalizer`, so reps and weights provably
//! index the SAME `G`-orbit partition of R (no second source of truth that
//! could drift). Two exact modes:
//!   * `Multinomial` — when the discovered place orbits are a direct product of
//!     FULL symmetric groups (`orbits_are_full_symmetric`). Canonical form =
//!     per-orbit ascending sort; weight = multinomial `|G_j|!/∏_v c_{j,v}!`
//!     (`orbit_size::multinomial_orbit_size`). Fast O(places·log places)/rep.
//!   * `GroupOrbit` — when the orbits form a COUPLED / diagonal group that is a
//!     STRICT subgroup of that product (e.g. a cyclic ring, or Anderson's one
//!     order-24 group permuting 23 size-4 orbits together). The multinomial
//!     would OVER-count and the per-orbit sort would OVER-merge, so a
//!     deterministic Schreier–Sims base-and-strong-generating-set (BSGS) over
//!     the place-domain generators (`explorer::bsgs`) drives BOTH the true
//!     `G`-orbit minimal-image canonical form (one rep per orbit) and the exact
//!     orbit size `|orbit(m)|` (orbit–stabilizer, holds for ANY finite group —
//!     no full-symmetric hypothesis). Admitted only when the exact `|G|` (from
//!     the BSGS) is within budget so per-rep cost stays bounded; larger coupled
//!     groups fall back to exact, un-reduced exploration (never a wrong count).
//!
//! The weight is threaded through BOTH the sequential explorer and the parallel
//! summary, fixing the prior parallel undercount (51 vs 243 on Philosophers-5);
//! a u64 overflow fails closed to CANNOT_COMPUTE. Both modes use only the
//! place-domain generators (the full group regardless of the truncatable
//! permutation-closure cache), so they are immune to
//! `PETRI_CANONICALIZER_CLOSURE_BUDGET` truncation. With the coupled path,
//! Anderson-PT-04 and Philosophers report the exact `29641 97516 1 6` and
//! `243 945 1 10` via the orbit quotient; AirplaneLD-PT-0010 (|G|=5040, over
//! the per-rep budget) reports the exact `43463 183664 1 38` via the fallback.
//!
//! **StableMarking is also excluded.** Its observer in
//! `crates/tla-petri/src/examinations/stable_marking.rs` tracks per-place
//! stability *by place index* (`self.stable[i]` compared against
//! `initial_tokens[i]`). Place-swap canonicalization sorts token counts
//! *within* each orbit, permuting which physical place occupies index `i`, so
//! the index-keyed stability flags stop tracking a fixed place. This is a
//! *wrong-verdict* bug, not a mere undercount. Counterexample (symmetric orbit
//! `{p0,p1}` fed by source `s`; m0=[1,0,0]; t0:s→p0, t1:s→p1): every place is
//! actually unstable (s drops 1→0, p0 rises 0→1 on the t0 branch, p1 rises
//! 0→1 on the t1 branch), so the truth is FALSE. But ascending orbit
//! canonicalization maps both reachable markings [0,1,0] and [0,0,1] to the
//! single canonical form [0,0,1]; index 1 then only ever shows token count 0,
//! matching its initial value, so the observer reports place index 1 as
//! "stable" and flips the verdict to a false TRUE (−16 MCC points). Stability
//! is inherently a per-place property that no orbit-quotient observer can
//! recover, so we refuse canonicalization for StableMarking regardless of
//! predicate σ-invariance. See the regression test in
//! `examinations::stable_marking` and the per-observer fail-closed override
//! `StableMarkingObserver::canonicalization_safe`.

use crate::examination::Examination;

/// Returns `true` iff it is sound to enable place-swap canonicalization for
/// the given MCC examination *without* an additional property-specific
/// σ-invariance proof.
///
/// See `docs/theorems/2026-05-26-place-swap-symmetry-soundness.md` §2.5 for
/// the classification underlying this function.
#[must_use]
pub(crate) const fn canonicalization_is_sound(examination: Examination) -> bool {
    match examination {
        // σ-invariant on any place orbit *and* the examination's observer
        // reports a value that does not depend on the cardinality of
        // orbit-distinct markings.
        Examination::ReachabilityDeadlock | Examination::OneSafe => true,

        // StableMarking's observer tracks per-place stability *by place index*
        // (`self.stable[i]` vs `initial_tokens[i]`). Place-swap canonicalization
        // sorts token counts within each orbit, permuting which physical place
        // occupies index `i`, so the index-keyed flags stop tracking a fixed
        // place. This is a wrong-verdict bug: with the symmetric net s,p0,p1
        // (t0:s→p0, t1:s→p1, m0=[1,0,0], orbit {p0,p1}) every place is actually
        // unstable (truth = FALSE), but canonicalization collapses [0,1,0] and
        // [0,0,1] to the single canonical form [0,0,1]; index 1 then only ever
        // shows token 0, so the observer reports a stable place and flips the
        // verdict to a false TRUE (−16 MCC points). Stability is inherently a
        // per-place property; no orbit-quotient observer can recover it. See the
        // module doc comment for the worked counterexample.
        Examination::StableMarking => false,

        // StateSpace publishes four σ-invariant *values*: |R|, |E|,
        // max_token_in_place, max_token_sum. The latter two are σ-invariant
        // maxima (untouched). The first two are cardinalities that orbit dedup
        // would otherwise collapse, but they are now recovered EXACTLY by
        // weighting each canonical orbit representative by the closed-form size
        // of its place-symmetry orbit (`orbit_size::multinomial_orbit_size`):
        //   |R| = Σ_reps |orbit(rep)|              (orbits partition R)
        //   |E| = Σ_reps |orbit(rep)| · deg(rep)   (free σ-action on out-edges,
        //                                           using the SOURCE orbit size)
        // The multinomial `|G_j|! / ∏_v c_{j,v}!` is exact because each
        // discovered orbit G_j is a *full* symmetric group: every in-orbit
        // transposition is a verified H1+H2 automorphism, gated at canonicalizer
        // construction by `PetriCanonicalizer::orbits_are_full_symmetric`
        // (nets that fail it fall back to exact, un-reduced exploration, never a
        // wrong count). Both the sequential explorer and the parallel summary
        // thread the weight (`on_new_state_with_orbit` /
        // `on_transition_fire_with_orbit`), fixing the prior 51-vs-243
        // Philosophers-5 parallel undercount. A u64 overflow of the multinomial
        // fails closed to CANNOT_COMPUTE (`on_orbit_overflow`), never a
        // truncated number. The closed-form size needs only the disjoint orbit
        // GROUPS, not the truncatable permutation-closure cache, so it is immune
        // to PETRI_CANONICALIZER_CLOSURE_BUDGET truncation and is
        // O(places·log places) per rep instead of the legacy
        // O(states·|G|·places) enumerator that HUNG on Anderson-PT-04 /
        // AirplaneLD-PT-0010. See
        // docs/theorems/2026-05-26-place-swap-symmetry-soundness.md §2.5.1 and
        // the unit/integration tests in `explorer::orbit_size` and
        // `examinations::state_space`.
        Examination::StateSpace => true,

        // QuasiLiveness and Liveness aggregate a *per-transition* predicate
        // (`enabled(t)` reachability for L1/QuasiLiveness; `AG EF enabled(t)`
        // for L4/Liveness). `enabled(t)` for a *specific* transition `t` is
        // NOT σ-invariant: a place-orbit permutation σ maps `t` to a different
        // transition σ(t), so a marking enabling `t` is orbit-equivalent to
        // one enabling σ(t), not `t`. A canonicalizing BFS that records which
        // specific transition fired (`QuasiLivenessObserver`) misses
        // transitions enabled only at non-canonical orbit members, producing a
        // false "not quasi-live" verdict (−16 MCC points). See the module doc
        // comment for the counterexample and the regression test in
        // `examination_non_property::liveness`. QuasiLiveness was an *active*
        // bug (its verdict path uses the canonicalizing observer); Liveness is
        // excluded as defense-in-depth — its current verdict path uses only
        // structural analysis, BMC, mu-calculus, and the *full*-graph SCC check
        // (none canonicalize), but the same σ-invariance failure would make a
        // future observer-based L4 fast-path unsound.
        Examination::QuasiLiveness | Examination::Liveness => false,

        // Predicate may distinguish individual places within an orbit.
        // Canonicalization is unsound without a per-property σ-invariance
        // proof; refuse to enable it.
        Examination::UpperBounds
        | Examination::ReachabilityCardinality
        | Examination::ReachabilityFireability
        | Examination::LTLCardinality
        | Examination::LTLFireability
        | Examination::CTLCardinality
        | Examination::CTLFireability => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_matches_soundness_proof() {
        // Per docs/theorems/2026-05-26-place-swap-symmetry-soundness.md §2.5:
        // examinations whose predicate is σ-invariant on every orbit *and*
        // whose observer output does not collapse with orbit deduplication.
        for examination in [
            Examination::ReachabilityDeadlock,
            Examination::OneSafe,
            // StateSpace recovers |R|/|E| exactly from the orbit quotient via the
            // closed-form multinomial orbit size (gated on full-symmetric orbits;
            // overflow fails closed). max-token values are σ-invariant maxima.
            Examination::StateSpace,
        ] {
            assert!(
                canonicalization_is_sound(examination),
                "{} must be classified as σ-invariant",
                examination.as_str(),
            );
        }

        // StableMarking tracks per-place stability by place index; place-swap
        // canonicalization permutes indices within orbits, so an actually
        // unstable place can masquerade as stable (matching its initial token
        // count by coincidence of the sort), flipping the verdict to a false
        // TRUE (−16 MCC). Counterexample s,p0,p1 in the module doc. Must refuse.
        assert!(
            !canonicalization_is_sound(Examination::StableMarking),
            "StableMarking tracks per-place stability by index; canonicalization \
             permutes indices and must be refused",
        );

        // QuasiLiveness and Liveness aggregate per-transition `enabled(t)`,
        // which is NOT σ-invariant under place-swap (σ maps t to σ(t)).
        // QuasiLiveness's observer-based verdict path recorded transitions
        // fired only from canonical representatives, missing transitions
        // enabled solely at non-canonical orbit members → false "not
        // quasi-live" verdict (−16 MCC). Liveness is excluded by the same
        // argument (defense-in-depth; its current path does not canonicalize).
        for examination in [Examination::QuasiLiveness, Examination::Liveness] {
            assert!(
                !canonicalization_is_sound(examination),
                "{} aggregates per-transition enabled(t), which is not \
                 σ-invariant; canonicalization must be refused",
                examination.as_str(),
            );
        }

        // StateSpace now recovers |R| and |E| EXACTLY from the orbit quotient:
        // each canonical orbit representative is weighted by the closed-form
        // multinomial orbit size (`orbit_size::multinomial_orbit_size`), threaded
        // through both the sequential explorer and the parallel summary. The
        // weight is exact because every discovered orbit is a full symmetric
        // group (gated by `orbits_are_full_symmetric`), and overflow fails closed
        // to CANNOT_COMPUTE. This fixes the prior parallel undercount (51 vs 243
        // on Philosophers-5) and the sequential O(states·|G|·places) hang on
        // Anderson-PT-04/AirplaneLD-PT-0010. So StateSpace is now sound to
        // canonicalize. See orbit_size unit tests and the integration tests.

        for examination in [
            Examination::UpperBounds,
            Examination::ReachabilityCardinality,
            Examination::ReachabilityFireability,
            Examination::LTLCardinality,
            Examination::LTLFireability,
            Examination::CTLCardinality,
            Examination::CTLFireability,
        ] {
            assert!(
                !canonicalization_is_sound(examination),
                "{} must not enable canonicalization without a property proof",
                examination.as_str(),
            );
        }
    }

    #[test]
    fn all_examination_kinds_are_classified() {
        // Compile-time exhaustiveness check: if a new examination variant is
        // added, the match in canonicalization_is_sound() must be updated and
        // this test should be expanded.
        for examination in Examination::ALL {
            let _ = canonicalization_is_sound(examination);
        }
    }
}
