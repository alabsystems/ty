// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Differential cross-check: the Tier-1 structural reduction-equation
//! StateSpace recognizer MUST agree with an explicit-state BFS oracle on every
//! net it accepts, with **0 disagreements**.
//!
//! The Tier-1 lane (`tier1_structural_state_space_stats`, exercised here via
//! the `tla_petri::examination::tier1_crosscheck_hook` test hook) is a PARTIAL
//! method: on a recognized strongly-connected single-simplex component it emits
//! the certified closed form (`states = multichoose(d, n)`, `edges = Σ_t
//! multichoose(d, n−Σpre_t)`); otherwise it falls back to per-component BFS, and
//! on any doubt it DECLINES (returns `None`).
//!
//! The soundness contract this battery gates: **whenever Tier-1 returns `Some`,
//! ALL FOUR metrics (states, edges, max_token_in_place, max_token_sum) must
//! equal the explicit-BFS result EXACTLY.** A Tier-1 `Some` that disagrees with
//! BFS is a hard FAILURE (panic). A Tier-1 `None` is allowed (partial method —
//! the existing lanes handle it, fail-closed).
//!
//! Mirrors the discipline of `tla-mdd/tests/crosscheck_bfs.rs`: a LIGHTWEIGHT
//! inline BFS oracle (identical firing rule + metric semantics to the petri
//! `StateSpaceObserver`: edges = Σ enabled firings; max_token_in_place = max
//! per-place value over reachable markings; max_token_sum = max total), a
//! deterministic gate of hand-picked nets (with non-vacuity assertions), and a
//! wide randomized proptest battery. The token/place bounds are chosen so every
//! generated net's `|R|` is small, so the inline BFS oracle always terminates.

use proptest::prelude::*;
use std::collections::HashSet;
use tla_bignum::BigUint;
use tla_petri::examination::tier1_crosscheck_hook;

/// One generated net: place count, initial marking, and `(inputs, outputs)`
/// per transition where each arc is `(place_index, weight)`.
#[derive(Debug, Clone)]
struct GenNet {
    num_places: usize,
    initial_marking: Vec<u64>,
    transitions: Vec<(Vec<(u32, u64)>, Vec<(u32, u64)>)>,
}

impl GenNet {
    /// Dense per-place `(pre, post)` weight vectors for each transition (arcs
    /// summed per place — parallel arcs to the same place add).
    fn dense_transitions(&self) -> Vec<(Vec<u64>, Vec<u64>)> {
        self.transitions
            .iter()
            .map(|(inputs, outputs)| {
                let mut pre = vec![0u64; self.num_places];
                let mut post = vec![0u64; self.num_places];
                for &(p, w) in inputs {
                    pre[p as usize] += w;
                }
                for &(p, w) in outputs {
                    post[p as usize] += w;
                }
                (pre, post)
            })
            .collect()
    }
}

/// Lightweight explicit-BFS four-metric oracle (identical firing rule + metric
/// semantics to the petri `StateSpaceObserver`). Returns `None` if the
/// reachable set exceeds `cap` (treated as "did not complete"), so the caller
/// only cross-checks when the oracle genuinely enumerated the full set.
fn bfs_metrics(net: &GenNet, cap: usize) -> Option<(BigUint, BigUint, u64, u64)> {
    let trans = net.dense_transitions();
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    seen.insert(net.initial_marking.clone());
    let mut frontier: Vec<Vec<u64>> = vec![net.initial_marking.clone()];
    let mut edges: u128 = 0;
    let mut max_in_place: u64 = net.initial_marking.iter().copied().max().unwrap_or(0);
    let mut max_sum: u64 = net.initial_marking.iter().sum();
    while let Some(m) = frontier.pop() {
        for (pre, post) in &trans {
            // enabled iff m[p] >= pre[p] for all p
            if !m.iter().zip(pre).all(|(mv, pv)| mv >= pv) {
                continue;
            }
            // next = m - pre + post (checked: no place overflow, no underflow
            // since enabled guarantees m >= pre)
            let mut next = m.clone();
            let mut ok = true;
            for p in 0..next.len() {
                match next[p]
                    .checked_sub(pre[p])
                    .and_then(|v| v.checked_add(post[p]))
                {
                    Some(v) => next[p] = v,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            edges += 1;
            if seen.insert(next.clone()) {
                if seen.len() > cap {
                    return None; // exceeded cap — treat as "did not complete"
                }
                let s: u64 = next.iter().sum();
                let mxp = next.iter().copied().max().unwrap_or(0);
                max_sum = max_sum.max(s);
                max_in_place = max_in_place.max(mxp);
                frontier.push(next);
            }
        }
    }
    Some((
        BigUint::from(seen.len() as u64),
        BigUint::from(edges),
        max_in_place,
        max_sum,
    ))
}

/// Run the Tier-1 lane (via the hook) and assert the Tier-1 ⊆ BFS soundness
/// contract against the lightweight inline oracle. Returns whether Tier-1
/// produced a count (for non-vacuity bookkeeping).
fn check(net: &GenNet) -> Tier1Outcome {
    let tier1 = tier1_crosscheck_hook(
        net.num_places,
        net.initial_marking.clone(),
        net.transitions.clone(),
        // Cap the per-component BFS FALLBACK inside the lane at the SAME bound as
        // the inline oracle below, so any net Tier-1 answers (recognized OR via
        // fallback) is one the oracle can also enumerate for the cross-check.
        200_000,
    );

    let Some(t) = tier1 else {
        return Tier1Outcome::Declined;
    };

    // Tier-1 produced a count ⇒ the inline oracle MUST agree exactly. A modest
    // cap keeps a random UNBOUNDED net (e.g. a pure-output, always-enabled
    // transition) from making the oracle run forever — but Tier-1 NEVER returns
    // Some for such a net (its gate requires a conservation P-invariant), so a
    // Tier-1-Some net always has a small bounded reachable set the oracle
    // enumerates well under this cap. If the oracle genuinely caps out on a
    // Tier-1-Some net, that is a real soundness concern (Tier-1 answered a net
    // too big to cross-check) — flag it.
    let bfs = bfs_metrics(net, 200_000).unwrap_or_else(|| {
        panic!(
            "Tier-1 returned Some {t:?} but the inline BFS oracle exceeded its cap \
             (cannot cross-check) for net {net:?}",
        )
    });

    assert_eq!(
        t.0, bfs.0,
        "Tier-1 states {} != BFS {} for {net:?}",
        t.0, bfs.0
    );
    assert_eq!(
        t.1, bfs.1,
        "Tier-1 edges {} != BFS {} for {net:?}",
        t.1, bfs.1
    );
    assert_eq!(
        t.2, bfs.2,
        "Tier-1 max_token_in_place {} != BFS {} for {net:?}",
        t.2, bfs.2
    );
    assert_eq!(
        t.3, bfs.3,
        "Tier-1 max_token_sum {} != BFS {} for {net:?}",
        t.3, bfs.3
    );
    Tier1Outcome::Agreed
}

enum Tier1Outcome {
    Agreed,
    Declined,
}

// ---------------------------------------------------------------------------
// Deterministic gate — fixed nets, no randomness. Exercises BOTH the recognizer
// path (strongly-connected single-simplex nets) and the BFS-fallback path, and
// pins the closed form against the oracle.
// ---------------------------------------------------------------------------

/// A WIDE strongly-connected unit state machine: `d` places, a directed ring
/// `p_i -> p_{(i+1) mod d}` (one unit move per place, guaranteeing strong
/// connectivity), PLUS a sprinkling of EXTRA unit moves (back-edges, chords, and
/// self-loops) so the recognized net is not merely a ring. With `n` tokens
/// initially on place 0 this is a conserving ordinary state machine the Tier-1
/// FAST-PATH must recognize (it is the exact shape the synthesized-conservation
/// reorder targets — on the production nets it is wide enough that the old
/// Farkas `compute_p_invariants` would truncate at `MAX_ROWS` and decline). The
/// reachable set is the full simplex `{x : Σx = n}`; with small `n` it stays
/// enumerable by the inline BFS oracle so the fast-path is differentially
/// cross-checked. `extra` is a list of `(src, dst)` unit moves to add (each a
/// `p_src -> p_dst` weight-1 transition; `src == dst` is a valid self-loop).
fn wide_unit_state_machine(d: usize, n: u64, extra: &[(usize, usize)]) -> GenNet {
    assert!(d >= 1);
    let mut initial_marking = vec![0u64; d];
    initial_marking[0] = n;
    let mut transitions = Vec::with_capacity(d + extra.len());
    // The spanning directed ring (strong connectivity).
    for i in 0..d {
        let next = ((i + 1) % d) as u32;
        transitions.push((vec![(i as u32, 1)], vec![(next, 1)]));
    }
    // Extra unit moves (back-edges / chords / self-loops). All within range.
    for &(s, t) in extra {
        let s = (s % d) as u32;
        let t = (t % d) as u32;
        transitions.push((vec![(s, 1)], vec![(t, 1)]));
    }
    GenNet {
        num_places: d,
        initial_marking,
        transitions,
    }
}

/// Inject CONSTANT READ self-loop places onto an existing net (the BART shape:
/// per-unit state machines coupled only through always-marked constant-guard
/// resource places). For each requested `(weight, init_extra)` a fresh place
/// `r_k` is appended with `init(r_k) = weight + init_extra` (so `init >= weight`,
/// the gate-(b) condition), and a balanced `(r_k, weight)` self-loop arc is added
/// to EVERY existing transition (so `pre(t,r_k) == post(t,r_k) == weight` for all
/// `t`, the gate-(a) condition). Under the constant-read R-reduction these places
/// are provably redundant: `|R|` and the edge set are UNCHANGED, while
/// `max_token_in_place`/`max_token_sum` rise by the constant amounts (which the
/// lane folds back). The BFS oracle runs on this ORIGINAL (unstripped) net, so
/// any disagreement is a real soundness failure.
///
/// `weight == 0` injects no arc (no real coupling) — still a valid constant read
/// place (init >= 0 trivially); the strip removes the now-isolated place.
/// `weight >= 1` makes a genuine read guard the strip must prove always-satisfied.
fn inject_constant_read_places(base: &GenNet, resources: &[(u64, u64)]) -> GenNet {
    let mut net = base.clone();
    for &(weight, init_extra) in resources {
        let r = net.num_places as u32;
        net.num_places += 1;
        net.initial_marking.push(weight.saturating_add(init_extra));
        if weight > 0 {
            for (inputs, outputs) in net.transitions.iter_mut() {
                inputs.push((r, weight));
                outputs.push((r, weight));
            }
        }
    }
    net
}

/// A directed token ring of `d` places with `n` tokens on place 0 and one unit
/// move `p_i -> p_{(i+1) mod d}` per place — the canonical recognized net.
fn directed_ring(d: usize, n: u64) -> GenNet {
    let mut initial_marking = vec![0u64; d];
    initial_marking[0] = n;
    let mut transitions = Vec::with_capacity(d);
    for i in 0..d {
        let next = ((i + 1) % d) as u32;
        transitions.push((vec![(i as u32, 1)], vec![(next, 1)]));
    }
    GenNet {
        num_places: d,
        initial_marking,
        transitions,
    }
}

#[test]
fn deterministic_gate_known_nets() {
    let mut recognized_nontrivial = 0u32;

    // Recognized directed rings of various sizes (BFS-enumerable).
    for (d, n) in [(2usize, 1u64), (3, 2), (4, 3), (5, 2), (3, 4), (6, 3)] {
        let net = directed_ring(d, n);
        match check(&net) {
            Tier1Outcome::Agreed => recognized_nontrivial += 1,
            Tier1Outcome::Declined => panic!("directed ring d={d} n={n} must be recognized"),
        }
    }

    // A two-component net: an independent pair of directed rings. Tier-1 should
    // recognize BOTH and compose the product.
    let mut net = GenNet {
        num_places: 6,
        initial_marking: vec![2, 0, 0, 3, 0, 0],
        transitions: Vec::new(),
    };
    for i in 0..3usize {
        let next = ((i + 1) % 3) as u32;
        net.transitions.push((vec![(i as u32, 1)], vec![(next, 1)]));
    }
    for i in 0..3usize {
        let p = (3 + i) as u32;
        let next = (3 + (i + 1) % 3) as u32;
        net.transitions.push((vec![(p, 1)], vec![(next, 1)]));
    }
    match check(&net) {
        Tier1Outcome::Agreed => recognized_nontrivial += 1,
        Tier1Outcome::Declined => panic!("two-ring product net must be recognized"),
    }

    // A net Tier-1 must DECLINE (non-conserving drain). Any outcome is fine —
    // only soundness matters (asserted inside `check`).
    let drain = GenNet {
        num_places: 1,
        initial_marking: vec![3],
        transitions: vec![(vec![(0, 1)], vec![])],
    };
    let _ = check(&drain);

    assert!(
        recognized_nontrivial >= 5,
        "deterministic gate vacuous: only {recognized_nontrivial} recognized multi-state nets",
    );
}

/// The WIDE FAST-PATH gate: strongly-connected unit state machines with MANY
/// places (and extra back-edges / chords / self-loops) and a small token count
/// must be recognized by the Tier-1 fast-path and agree with the inline BFS
/// oracle EXACTLY. These are the nets the synthesized-conservation reorder
/// targets: on the production instances (NeighborGrid / Diffusion2D) they are
/// wide enough that the OLD Farkas `compute_p_invariants` truncates at
/// `MAX_ROWS` and the recognizer spuriously declined. Token counts are kept
/// small so `multichoose(d, n)` stays under the oracle's enumeration cap.
#[test]
fn deterministic_wide_unit_state_machine_fast_path_matches_oracle() {
    let mut fired = 0u32;

    // Pure wide rings of growing width with n = 1 (|R| = d) and n = 2
    // (|R| = d(d+1)/2). d = 120, n = 2 ⇒ |R| = 7260 (well under the cap).
    for (d, n) in [
        (40usize, 1u64),
        (80, 1),
        (120, 1),
        (40, 2),
        (80, 2),
        (120, 2),
    ] {
        let net = wide_unit_state_machine(d, n, &[]);
        match check(&net) {
            Tier1Outcome::Agreed => fired += 1,
            Tier1Outcome::Declined => {
                panic!("wide unit state machine d={d} n={n} must be recognized by the fast-path")
            }
        }
    }

    // Wide nets with EXTRA unit moves: back-edges, chords, AND self-loops. Still
    // strongly connected (the spanning ring), still conserving, still the full
    // simplex — so the fast-path must fire and agree. The self-loops (e.g.
    // (3,3)) specifically exercise the "self-loop is a valid unit move" edge
    // case. d=60, n=3 ⇒ |R| = C(62,3) = 37820 (under the cap).
    let extra = [
        (5usize, 0usize),
        (10, 3),
        (3, 3),
        (40, 7),
        (59, 30),
        (20, 20),
    ];
    for (d, n) in [(40usize, 2u64), (60, 3), (90, 2)] {
        let net = wide_unit_state_machine(d, n, &extra);
        match check(&net) {
            Tier1Outcome::Agreed => fired += 1,
            Tier1Outcome::Declined => panic!(
                "wide unit state machine with back-edges/chords/self-loops d={d} n={n} \
                 must be recognized by the fast-path"
            ),
        }
    }

    assert!(
        fired >= 9,
        "wide fast-path gate vacuous: only {fired} recognized"
    );
}

/// CONSTANT-READ-ARC R-REDUCTION deterministic gate (the BART shape): inject
/// always-marked constant-guard resource places onto KNOWN strongly-connected
/// unit-SM rings (single ring and independent ring pairs) and assert Tier-1
/// recognizes the stripped net and agrees with the BFS oracle on the ORIGINAL
/// (unstripped) net EXACTLY — 0 disagreements across all four metrics. WITHOUT
/// the reduction these nets are connected (the resource couples every transition)
/// with 2-in/2-out transitions and the recognizer would decline; WITH it they
/// decouple into the recognized rings.
#[test]
fn deterministic_constant_read_reduction_matches_oracle() {
    let mut fired = 0u32;

    // Single ring + various constant-read resources (genuine guards and a no-op).
    for (d, n) in [(3usize, 2u64), (5, 2), (6, 3), (4, 3)] {
        for resources in [
            &[(1u64, 0u64)][..],           // one resource, init == weight (boundary)
            &[(1, 3)][..],                 // init well above weight
            &[(2, 0), (1, 5)][..],         // two resources, mixed weights
            &[(0, 4)][..],                 // no-op (isolated) constant place
            &[(3, 0), (0, 2), (1, 1)][..], // mix of guards + isolated
        ] {
            let base = directed_ring(d, n);
            let net = inject_constant_read_places(&base, resources);
            match check(&net) {
                Tier1Outcome::Agreed => fired += 1,
                Tier1Outcome::Declined => panic!(
                    "constant-read-coupled ring d={d} n={n} resources={resources:?} \
                     must decouple + be recognized"
                ),
            }
        }
    }

    // Two independent rings coupled ONLY by a shared constant resource — the
    // literal BART topology (per-train SMs + a shared DistStation resource).
    let mut two_rings = GenNet {
        num_places: 6,
        initial_marking: vec![1, 0, 0, 1, 0, 0],
        transitions: Vec::new(),
    };
    for base in [0u32, 3] {
        for i in 0..3u32 {
            let p = base + i;
            let next = base + (i + 1) % 3;
            two_rings.transitions.push((vec![(p, 1)], vec![(next, 1)]));
        }
    }
    for resources in [&[(1u64, 0u64)][..], &[(1, 2), (2, 0)][..], &[(0, 3)][..]] {
        let net = inject_constant_read_places(&two_rings, resources);
        match check(&net) {
            Tier1Outcome::Agreed => fired += 1,
            Tier1Outcome::Declined => {
                panic!("two rings + shared constant resource {resources:?} must decouple")
            }
        }
    }

    assert!(
        fired >= 20,
        "constant-read gate vacuous: only {fired} recognized constant-read nets",
    );
}

#[test]
fn directed_ring_closed_form_matches_oracle() {
    // Pin the closed form on a recognized ring against the hand-computed value.
    // d=4, n=3: states = C(6,3) = 20; edges = 4 * C(5,2) = 4*10 = 40.
    let net = directed_ring(4, 3);
    let t = tier1_crosscheck_hook(
        net.num_places,
        net.initial_marking.clone(),
        net.transitions.clone(),
        2_000_000,
    )
    .expect("d=4 n=3 ring must be recognized");
    let b = bfs_metrics(&net, 5_000_000).expect("inline BFS completes on the tiny ring");
    assert_eq!(t.0, BigUint::from(20u32), "states = C(6,3) = 20");
    assert_eq!(t.1, BigUint::from(40u32), "edges = 4 * C(5,2) = 40");
    assert_eq!(t.2, 3, "max_token_in_place = n = 3");
    assert_eq!(t.3, 3, "max_token_sum = n = 3");
    assert_eq!(
        (t.0, t.1, t.2, t.3),
        (b.0, b.1, b.2, b.3),
        "Tier-1 == inline BFS"
    );
}

// ---------------------------------------------------------------------------
// Randomized differential battery.
// ---------------------------------------------------------------------------

/// Strategy for a random small bounded net. Two flavors are mixed:
///   (a) RANDOM nets (up to 5 places, up to 6 transitions, small arc weights,
///       small initial markings) — mostly NOT recognized, exercising the
///       recognizer's DECLINE paths and (for bounded ones) the BFS fallback;
///   (b) DIRECTED RINGS (the recognized shape) — so the recognizer is actually
///       exercised on the path that emits the closed form, not just declining.
/// Bounds keep the reachable set tiny so the inline BFS oracle always finishes.
fn arb_net() -> impl Strategy<Value = GenNet> {
    let random = (1usize..=5).prop_flat_map(|num_places| {
        let init = prop::collection::vec(0u64..=2, num_places);
        let trans = prop::collection::vec(
            (
                prop::collection::vec((0u32..num_places as u32, 1u64..=2), 0..=2),
                prop::collection::vec((0u32..num_places as u32, 1u64..=2), 0..=2),
            ),
            0..=6,
        );
        (init, trans).prop_map(move |(initial_marking, transitions)| GenNet {
            num_places,
            initial_marking,
            transitions,
        })
    });

    // Directed rings: 1..=6 places, 0..=6 tokens — the recognized shape.
    let rings = (1usize..=6, 0u64..=6).prop_map(|(d, n)| directed_ring(d, n));

    // WIDE unit state machines: 8..=60 places (far past the small `random` /
    // `rings` widths) with a small token count (so multichoose(d, n) stays
    // enumerable by the inline oracle) and a random set of EXTRA unit moves
    // (back-edges / chords / self-loops). This differentially exercises the
    // synthesized-conservation FAST-PATH — the exact shape the reorder targets —
    // on every case, not just the deterministic gate. Token count 0..=2 keeps
    // |R| = multichoose(d, n) ≤ d(d+1)/2 ≤ 1830 for d ≤ 60, well under the cap.
    let wide = (8usize..=60usize).prop_flat_map(|d| {
        let n = 0u64..=2;
        // 0..=8 extra unit moves, each (src, dst) drawn over the net's places
        // (src == dst is a valid self-loop). The spanning ring already makes the
        // net strongly connected, so any extra edges keep it recognized.
        let extra = prop::collection::vec((0usize..d, 0usize..d), 0..=8);
        (Just(d), n, extra).prop_map(|(d, n, extra)| wide_unit_state_machine(d, n, &extra))
    });

    // CONSTANT-READ lane (the BART shape): a strongly-connected unit-SM base
    // (ring or wide-with-chords) COUPLED through random always-marked constant
    // read resource places. The base alone is recognized; the injected resources
    // must be STRIPPED by the R-reduction (otherwise the net is connected with
    // 2-in/2-out transitions and would decline). Token counts kept tiny so the
    // BFS oracle enumerates the (unstripped) original. Resources: 1..=4, each a
    // `(weight 0..=2, init_extra 0..=3)` — weight 0 is an isolated constant
    // place, weight>=1 a genuine read guard (init = weight + init_extra >= weight,
    // satisfying gate (b)).
    let const_read = (3usize..=30usize).prop_flat_map(|d| {
        let n = 0u64..=2u64;
        let extra = prop::collection::vec((0usize..d, 0usize..d), 0..=5);
        let resources = prop::collection::vec((0u64..=2u64, 0u64..=3u64), 1..=4);
        (Just(d), n, extra, resources).prop_map(|(d, n, extra, resources)| {
            let base = wide_unit_state_machine(d, n, &extra);
            inject_constant_read_places(&base, &resources)
        })
    });

    // Mix: random (declines + small BFS fallback), small rings (recognized),
    // WIDE unit state machines (the fast-path), and CONSTANT-READ-coupled nets
    // (the R-reduction). Weight the structured lanes heavily for coverage.
    prop_oneof![3 => random, 1 => rings, 2 => wide, 2 => const_read]
}

proptest! {
    // Wide soaked battery (the inline oracle is cheap, so we run many cases).
    #![proptest_config(ProptestConfig {
        cases: 8192,
        max_shrink_iters: 8192,
        ..ProptestConfig::default()
    })]

    /// The breadth gate: every generated net must satisfy the Tier-1 ⊆ BFS
    /// contract. `check` panics on any disagreement with the offending net.
    #[test]
    fn tier1_agrees_with_bfs_oracle(net in arb_net()) {
        let _ = check(&net);
    }
}

/// Non-vacuity proof that the recognizer's EMIT path (not just its DECLINE
/// path) is exercised and agrees with the oracle: sweep a spread of directed
/// rings (the recognized full-simplex shape) of varying dimension and token
/// count, and assert Tier-1 FIRES on every one AND matches the inline BFS
/// oracle exactly. Uses only the recognized shape (no heavyweight production
/// BFS fallback), so it is fast and 100% emit-path coverage. The broad
/// random-vs-recognized differential coverage is the 8192-case `proptest` above.
#[test]
fn recognizer_emit_path_fires_and_matches_oracle_on_rings() {
    let mut fired = 0u32;
    let mut total = 0u32;
    // Dimensions and token counts whose simplex is still small enough for the
    // inline oracle to enumerate (multichoose(d, n) stays modest here).
    for d in 1usize..=7 {
        for n in 0u64..=6 {
            let net = directed_ring(d, n);
            total += 1;
            match check(&net) {
                Tier1Outcome::Agreed => fired += 1,
                Tier1Outcome::Declined => {
                    panic!("recognizer must FIRE on directed ring d={d} n={n} (recognized shape)")
                }
            }
        }
    }
    // Every recognized net fired AND matched the oracle (the `check` assertions).
    assert_eq!(
        fired, total,
        "recognizer should fire on every directed ring ({fired}/{total})",
    );
    assert!(
        total >= 40,
        "non-vacuity: too few recognized nets ({total})"
    );
}
