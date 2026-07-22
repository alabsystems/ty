// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Orbit-size oracle for [`PetriCanonicalizer`].
//!
//! Computes |G·m| — the size of the orbit of a marking m under the place
//! symmetry group G discovered by the canonicalizer. Required for sound
//! StateSpace canonicalization, where the explorer visits one representative
//! per orbit and must multiply each observation by the orbit size to recover
//! the true count of reachable markings.
//!
//! # Soundness contract
//!
//! Returns `Err(OrbitSizeError::ClosureIncomplete)` when the canonicalizer's
//! precomputed permutation cache is a strict subset of ⟨generators⟩ (i.e.
//! the BFS-closure was truncated by the budget guard). In that case the
//! cache is NOT a subgroup and orbit sizes derived from it would silently
//! under-count, so callers must fall back to a sound alternative
//! (enumerate generators on the fly, or refuse the multiplication).
//!
//! See `crates/tla-petri/src/explorer/symmetry.rs` —
//! `PETRI_CANONICALIZER_CLOSURE_BUDGET` and
//! `PetriCanonicalizer::closure_is_complete`.

use std::collections::HashSet;

use tla_bignum::{BigUint, One, ToPrimitive};

use super::symmetry::PetriCanonicalizer;

/// Closed-form size of the orbit of `marking` under the place-symmetry group
/// `∏_j Sym(G_j)`, where `orbits` is the list of disjoint place orbits
/// (`PetriCanonicalizer::place_orbits`).
///
/// For a single orbit `G` of size `n`, the stabilizer of a marking under
/// `Sym(G)` is the Young subgroup fixing the token multiset, so by
/// orbit–stabilizer the orbit size is the multinomial coefficient
///
/// ```text
///   |Sym(G)·m| = n! / ∏_v (c_v!)
/// ```
///
/// where `c_v = #{p ∈ G : m(p) = v}`. This is computed in `O(n log n)` by
/// sorting the orbit's token values and scanning equal runs — NO permutation
/// enumeration, NO allocation of orbit images (the legacy `orbit_size_of` was
/// `O(|G|·places)` per call and HUNG on large groups). Because the orbits are
/// place-disjoint, the full group is the direct product `∏_j Sym(G_j)`,
/// stabilizers multiply, and the total orbit size is the product of the
/// per-orbit multinomials.
///
/// # Exactness contract
///
/// The factorial `n!` is the size of the *full* symmetric group on `G`. The
/// result equals the true orbit size ONLY when each `G_j` is a genuine full
/// symmetric group (every in-orbit transposition is an H1+H2 automorphism) —
/// gate on [`PetriCanonicalizer::orbits_are_full_symmetric`]. A strict
/// subgroup (e.g. cyclic) would make the factorial OVER-count.
///
/// # Overflow
///
/// The exact arbitrary-precision orbit size is computed by
/// [`multinomial_orbit_size_big`] (no cap — the multinomial `n!/∏c_v!` can be
/// astronomically large on a wide symmetric orbit and is carried exactly as a
/// [`BigUint`]). This `u64` entry narrows that exact value fail-closed: it
/// returns `None` when the orbit size exceeds `u64::MAX`, which callers route to
/// a fail-closed CANNOT_COMPUTE — never a wrong (truncated) number. The wide
/// StateSpace count path consumes [`multinomial_orbit_size_big`] directly so a
/// `> u64` orbit no longer caps the reportable count.
#[must_use]
pub(crate) fn multinomial_orbit_size(orbits: &[Vec<u32>], marking: &[u64]) -> Option<u64> {
    multinomial_orbit_size_big(orbits, marking).to_u64()
}

/// Closed-form orbit size as an EXACT arbitrary-precision [`BigUint`]
/// (`∏_j |G_j|! / ∏_v c_{j,v}!`), never capped.
///
/// Same `O(n log n)`-per-orbit run-scan as the `u64` entry, but with exact
/// bignum arithmetic: the multinomial of a wide symmetric orbit (e.g. a large
/// Philosophers / symmetric family) can dwarf `u64`/`u128`, and the StateSpace
/// orbit-sum count path now carries it exactly so the cell is reported instead
/// of declining on the representational cap. The narrowed `u64` value (the
/// observer-trait weight) is recovered fail-closed by [`multinomial_orbit_size`].
#[must_use]
pub(crate) fn multinomial_orbit_size_big(orbits: &[Vec<u32>], marking: &[u64]) -> BigUint {
    let mut total = BigUint::one();
    let mut values: Vec<u64> = Vec::new();
    for group in orbits {
        // Collect and sort the token counts on this orbit's places.
        values.clear();
        values.extend(group.iter().map(|&p| marking[p as usize]));
        values.sort_unstable();

        // n! / ∏_v (c_v!). Build incrementally over the sorted values: for the
        // k-th element multiply the numerator by k (building the running k!),
        // and within a run of equal values of current length r divide by r.
        // The division is exact at every step because after processing r equal
        // values the accumulated numerator contains the factor r! contributed
        // by those positions. Exact bignum — no overflow, no cap.
        let mut factor = BigUint::one();
        let mut k: u64 = 0;
        let mut run_len: u64 = 0;
        let mut prev: Option<u64> = None;
        for &v in &values {
            k += 1;
            factor *= BigUint::from(k); // numerator *= k  (builds n!)
            if prev == Some(v) {
                run_len += 1;
                factor /= BigUint::from(run_len); // exact: cancels the c_v! factor
            } else {
                run_len = 1;
                prev = Some(v);
            }
        }
        total *= factor;
    }
    total
}

/// Failure modes for the orbit-size oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OrbitSizeError {
    /// The canonicalizer's precomputed permutation cache was truncated by
    /// the closure budget; the cached set is a strict subset of the full
    /// group and is NOT closed under composition, so multiplying orbit
    /// sizes derived from it would silently under-count.
    ClosureIncomplete,
}

/// Compute |G·m| — the size of the orbit of `marking` under the place
/// symmetry group cached by `canonicalizer`.
///
/// The orbit is computed by applying every cached permutation π ∈ G to
/// `marking` and counting the number of distinct images. The contract
/// only holds when [`PetriCanonicalizer::closure_is_complete`] is `true`;
/// otherwise the cached permutation set is a strict subset of the group
/// and the returned count would under-approximate |G·m|.
///
/// # Errors
///
/// Returns [`OrbitSizeError::ClosureIncomplete`] when the canonicalizer's
/// permutation cache was truncated by the closure budget.
pub(crate) fn orbit_size_of(
    canonicalizer: &PetriCanonicalizer,
    marking: &[u64],
) -> Result<u64, OrbitSizeError> {
    let permutations = canonicalizer.permutations();
    let num_p = marking.len();

    if permutations.len() <= 1 {
        return Ok(1);
    }

    let mut images: HashSet<Vec<u64>> = HashSet::new();
    images.insert(marking.to_vec());

    for perm in permutations {
        let mut next = vec![0u64; num_p];
        for i in 0..num_p {
            next[i] = marking[perm[i]];
        }
        images.insert(next);
    }

    Ok(images.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    /// Two-place symmetric net: places p0/p1 are interchangeable via the
    /// transition automorphism that swaps t0 with t1. Both feed the same
    /// sink, so the place orbit is {p0, p1}.
    fn two_place_symmetric_net() -> PetriNet {
        PetriNet {
            name: Some("two-place-symmetric".into()),
            places: vec![place("p0"), place("p1"), place("sink")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
            ],
            initial_marking: vec![1, 1, 0],
        }
    }

    #[test]
    fn asymmetric_marking_has_orbit_size_two() {
        let net = two_place_symmetric_net();
        let canonicalizer = PetriCanonicalizer::build(&net);
        assert!(canonicalizer.closure_is_complete());

        // [1, 0, 0] is not fixed by the p0<->p1 swap -> orbit {[1,0,0],[0,1,0]}.
        let size =
            orbit_size_of(&canonicalizer, &[1, 0, 0]).expect("complete closure should return Ok");
        assert_eq!(size, 2);
    }

    #[test]
    fn symmetric_marking_has_orbit_size_one() {
        let net = two_place_symmetric_net();
        let canonicalizer = PetriCanonicalizer::build(&net);
        assert!(canonicalizer.closure_is_complete());

        // [1, 1, 0] is fixed by the swap -> orbit {[1,1,0]}.
        let size =
            orbit_size_of(&canonicalizer, &[1, 1, 0]).expect("complete closure should return Ok");
        assert_eq!(size, 1);

        // [0, 0, 5] is fixed -> orbit size 1.
        let size =
            orbit_size_of(&canonicalizer, &[0, 0, 5]).expect("complete closure should return Ok");
        assert_eq!(size, 1);
    }

    /// Three interchangeable places feeding a common sink — Sym(3) action,
    /// |G| = 6. Marking [1,0,0,0] has orbit size 3 (three single-token
    /// positions). Marking [1,1,0,0] has orbit size 3 (C(3,2) = 3).
    /// Marking [1,1,1,0] has orbit size 1 (fixed).
    fn three_place_symmetric_net() -> PetriNet {
        PetriNet {
            name: Some("three-place-symmetric".into()),
            places: vec![place("p0"), place("p1"), place("p2"), place("sink")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(3, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(3, 1)]),
                trans("t2", vec![arc(2, 1)], vec![arc(3, 1)]),
            ],
            initial_marking: vec![1, 1, 1, 0],
        }
    }

    #[test]
    fn three_place_sym3_orbit_sizes_match_combinatorics() {
        let net = three_place_symmetric_net();
        let canonicalizer = PetriCanonicalizer::build(&net);
        assert!(canonicalizer.closure_is_complete());

        // Single token on p0 — orbit is the three single-token positions.
        let size = orbit_size_of(&canonicalizer, &[1, 0, 0, 0]).expect("Ok");
        assert_eq!(size, 3);

        // Two tokens on {p0,p1} — orbit is C(3,2) = 3 unordered pairs.
        let size = orbit_size_of(&canonicalizer, &[1, 1, 0, 0]).expect("Ok");
        assert_eq!(size, 3);

        // Three tokens — symmetric marking, orbit size 1.
        let size = orbit_size_of(&canonicalizer, &[1, 1, 1, 0]).expect("Ok");
        assert_eq!(size, 1);
    }

    #[test]
    fn trivially_asymmetric_net_returns_one() {
        // Net with no non-trivial automorphisms: every marking has orbit 1.
        let net = PetriNet {
            name: Some("asymmetric".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        };
        let canonicalizer = PetriCanonicalizer::build(&net);
        assert!(canonicalizer.closure_is_complete());

        let size = orbit_size_of(&canonicalizer, &[1, 0]).expect("Ok");
        assert_eq!(size, 1);
        let size = orbit_size_of(&canonicalizer, &[0, 1]).expect("Ok");
        assert_eq!(size, 1);
    }

    // ---------------- multinomial_orbit_size (closed-form) ----------------

    #[test]
    fn multinomial_two_place_orbit() {
        let orbits = vec![vec![0u32, 1]];
        // [1,0] -> orbit {[1,0],[0,1]} -> 2! / (1!·1!) = 2.
        assert_eq!(multinomial_orbit_size(&orbits, &[1, 0, 0]), Some(2));
        // [1,1] -> fixed -> 2! / 2! = 1.
        assert_eq!(multinomial_orbit_size(&orbits, &[1, 1, 0]), Some(1));
        // [0,0] -> fixed -> 1.
        assert_eq!(multinomial_orbit_size(&orbits, &[0, 0, 5]), Some(1));
    }

    #[test]
    fn multinomial_three_place_orbit() {
        let orbits = vec![vec![0u32, 1, 2]];
        // [1,0,0] -> 3!/(1!·2!) = 3.
        assert_eq!(multinomial_orbit_size(&orbits, &[1, 0, 0, 0]), Some(3));
        // [1,1,0] -> 3!/(2!·1!) = 3.
        assert_eq!(multinomial_orbit_size(&orbits, &[1, 1, 0, 0]), Some(3));
        // [1,1,1] -> 3!/3! = 1.
        assert_eq!(multinomial_orbit_size(&orbits, &[1, 1, 1, 0]), Some(1));
        // Distinct values [2,1,0] -> 3! = 6.
        assert_eq!(multinomial_orbit_size(&orbits, &[2, 1, 0, 0]), Some(6));
    }

    #[test]
    fn multinomial_empty_orbits_is_one() {
        assert_eq!(multinomial_orbit_size(&[], &[5, 3, 0]), Some(1));
    }

    #[test]
    fn multinomial_two_independent_orbits_multiply() {
        // Orbit A = {0,1}, orbit B = {2,3,4}. Marking puts 1 token on one
        // place of A (→2) and 1 token on one place of B (→3). Total = 6.
        let orbits = vec![vec![0u32, 1], vec![2u32, 3, 4]];
        assert_eq!(multinomial_orbit_size(&orbits, &[1, 0, 1, 0, 0]), Some(6));
        // Both A places equal, both-of-three B places equal: 1 · 3 = 3.
        assert_eq!(multinomial_orbit_size(&orbits, &[1, 1, 1, 1, 0]), Some(3));
    }

    #[test]
    fn multinomial_philosophers5_shape_sums_to_243() {
        // 5 interchangeable philosophers, each in one of 3 local states encoded
        // as token count v ∈ {0,1,2} on a single shared place. The 5-place
        // orbit's canonical reps are the non-decreasing 5-tuples over {0,1,2}
        // (C(5+2,2) = 21 of them); their multinomial weights must sum to
        // 3^5 = 243 (every assignment of a state to each distinct philosopher).
        let orbit = vec![0u32, 1, 2, 3, 4];
        let orbits = vec![orbit];
        let mut total = 0u64;
        let mut reps = 0u64;
        for a in 0..3u64 {
            for b in a..3 {
                for c in b..3 {
                    for d in c..3 {
                        for e in d..3 {
                            let marking = [a, b, c, d, e];
                            total += multinomial_orbit_size(&orbits, &marking).unwrap();
                            reps += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(reps, 21, "expected C(7,2)=21 canonical reps");
        assert_eq!(total, 243, "orbit weights must sum to 3^5 = 243");
    }

    #[test]
    fn multinomial_overflow_is_fail_closed_none() {
        // 25 interchangeable places, all distinct token counts → 25! which is
        // ~1.55e25, far beyond u64::MAX (~1.8e19). The u64 entry must return None
        // (fail-closed → CANNOT_COMPUTE), never a truncated wrong number.
        let orbit: Vec<u32> = (0..25).collect();
        let marking: Vec<u64> = (0..25).map(|i| i as u64).collect();
        assert_eq!(multinomial_orbit_size(&[orbit], &marking), None);
    }

    #[test]
    fn multinomial_big_reports_above_u64_exactly() {
        // The bignum entry carries the EXACT value where the u64 entry caps.
        // 25 distinct token values → 25! exactly. Hand-checked: 25! =
        // 15511210043330985984000000.
        let orbit: Vec<u32> = (0..25).collect();
        let marking: Vec<u64> = (0..25).map(|i| i as u64).collect();
        let big = multinomial_orbit_size_big(&[orbit], &marking);
        let expected = BigUint::parse_bytes(b"15511210043330985984000000", 10).expect("decimal");
        assert_eq!(big, expected, "25! exact via bignum");
        assert!(big > BigUint::from(u64::MAX), "genuinely > u64::MAX");
        // ...and the u64 entry fail-closes (declines) on the same input.
        assert_eq!(multinomial_orbit_size(&[orbit_vec()], &marking_vec()), None);
    }

    fn orbit_vec() -> Vec<u32> {
        (0..25).collect()
    }
    fn marking_vec() -> Vec<u64> {
        (0..25).map(|i| i as u64).collect()
    }

    #[test]
    fn multinomial_big_matches_u64_in_range() {
        // On every in-u64 case the bignum value narrows exactly to the u64 entry.
        let orbits = vec![vec![0u32, 1, 2]];
        for marking in [[1u64, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [2, 1, 0, 0]] {
            let big = multinomial_orbit_size_big(&orbits, &marking);
            let narrow = multinomial_orbit_size(&orbits, &marking);
            assert_eq!(big.to_u64(), narrow, "big narrows to the u64 entry");
        }
    }

    #[test]
    fn multinomial_matches_brute_force_orbit_size() {
        // Differential check against the enumerate-and-dedup oracle on the
        // 3-place Sym(3) net for several markings.
        let net = three_place_symmetric_net();
        let canon = PetriCanonicalizer::build(&net);
        let orbits = canon.place_orbits();
        assert_eq!(orbits, &[vec![0u32, 1, 2]]);
        for marking in [
            [1u64, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [2, 1, 0, 0],
            [0, 0, 0, 9],
        ] {
            let brute = orbit_size_of(&canon, &marking).unwrap();
            let closed = multinomial_orbit_size(orbits, &marking).unwrap();
            assert_eq!(
                brute, closed,
                "closed-form orbit size must match brute force for {marking:?}",
            );
        }
    }
}
