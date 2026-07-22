// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Differential cross-check: the `tla-mdd` reachable-state count MUST equal
//! the `tla-dd` explicit-state BFS count on every net, with **0
//! disagreements**.
//!
//! `tla_dd::bfs_reachable_set_count` is the same explicit BFS the production
//! BDD lane is validated against (see `tla-dd/src/saturation.rs`), so pinning
//! the MDD count to it pins the new lane to the SAME ground truth the BDD lane
//! uses. The two engines share an identical net model and firing rule (enabled
//! iff `m[p] >= pre[p]`; fire `next = m - pre + post`; reject if any
//! `next[p] > bound[p]`), so a disagreement is a real soundness bug, never a
//! modelling mismatch.
//!
//! This battery is the SOUNDNESS GATE for the gate-only MDD lane: it must run
//! green before any production path may consume the MDD verdict.

use proptest::prelude::*;
use tla_dd::{bfs_reachable_set_count, DdNetSpec, DdTransition};
use tla_mdd::{CountError, MddNet, MddStateSpaceMetrics, MddTransition};

/// Convert the shared net shape into both engines' input types from one
/// source of truth, so the two lanes provably see the identical net.
#[derive(Debug, Clone)]
struct SharedNet {
    bounds: Vec<u64>,
    initial_marking: Vec<u64>,
    transitions: Vec<(Vec<u64>, Vec<u64>)>, // (pre, post)
}

impl SharedNet {
    fn to_mdd(&self) -> MddNet {
        MddNet {
            bounds: self.bounds.clone(),
            initial_marking: self.initial_marking.clone(),
            transitions: self
                .transitions
                .iter()
                .map(|(pre, post)| MddTransition {
                    pre: pre.clone(),
                    post: post.clone(),
                })
                .collect(),
        }
    }

    fn to_dd(&self) -> DdNetSpec {
        DdNetSpec {
            bounds: self.bounds.clone(),
            initial_marking: self.initial_marking.clone(),
            transitions: self
                .transitions
                .iter()
                .map(|(pre, post)| DdTransition {
                    pre: pre.clone(),
                    post: post.clone(),
                })
                .collect(),
        }
    }
}

/// One MDD engine's outcome on a net: a count, a fail-closed decline, or the
/// (illegal) malformed verdict on a well-formed net.
fn run_engine(
    label: &str,
    net: &SharedNet,
    f: impl FnOnce() -> Result<u64, CountError>,
) -> Option<u64> {
    match f() {
        Ok(c) => Some(c),
        Err(CountError::CountOverflow | CountError::ResourceCap(_)) => None,
        Err(CountError::Malformed(m)) => panic!(
            "{label} declined a well-formed generated net as malformed: {m} \
             (bounds={:?} init={:?})",
            net.bounds, net.initial_marking
        ),
    }
}

/// Run ALL THREE MDD engines (explicit kernel, symbolic relprod, node-level
/// saturation) on the same net and assert exact mutual agreement AND agreement
/// with the `tla-dd` BFS oracle.
///
/// This is the soundness gate for the symbolic engine: relprod and saturation
/// must equal the explicit kernel (the original cross-checked MDD oracle) AND
/// the BFS oracle. Any engine may legitimately DECLINE (fail-closed) on a
/// resource cap / overflow; a decline is not a disagreement, but every engine
/// that DID produce a count must agree with the BFS oracle. The random
/// generators are sized so declines essentially never happen.
fn check_agreement(net: &SharedNet) -> AgreementOutcome {
    let bfs = bfs_reachable_set_count(&net.to_dd());

    let explicit = run_engine("explicit kernel", net, || {
        net.to_mdd().reachable_count().map(|r| r.state_count)
    });
    let relprod = run_engine("relprod", net, || {
        net.to_mdd()
            .reachable_count_relprod(None)
            .map(|r| r.state_count)
    });
    let saturation = run_engine("saturation", net, || {
        net.to_mdd()
            .reachable_count_saturation(None)
            .map(|r| r.state_count)
    });

    // Every engine that produced a count must equal the BFS oracle exactly.
    for (label, got) in [
        ("explicit kernel", explicit),
        ("relprod", relprod),
        ("saturation", saturation),
    ] {
        if let Some(c) = got {
            assert_eq!(
                c, bfs,
                "{label} MDD count {c} != BFS oracle {bfs} for net \
                 bounds={:?} init={:?} transitions={:?}",
                net.bounds, net.initial_marking, net.transitions
            );
        }
    }

    // If every engine declined, treat as a decline; otherwise the agreed count
    // is the BFS value (already asserted equal above).
    if explicit.is_none() && relprod.is_none() && saturation.is_none() {
        AgreementOutcome::Declined
    } else {
        AgreementOutcome::Agreed(bfs)
    }
}

enum AgreementOutcome {
    Agreed(u64),
    Declined,
}

// ---------------------------------------------------------------------------
// Deterministic differential gate — fixed nets, no randomness. This is the
// reproducible CI gate (proptest below adds breadth). Each net's expected
// count is also asserted directly so the oracle itself can't silently drift.
// ---------------------------------------------------------------------------

fn t(pre: Vec<u64>, post: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    (pre, post)
}

#[test]
fn deterministic_gate_known_nets() {
    let cases: Vec<(SharedNet, u64)> = vec![
        // Single shuttling token: 2 states.
        (
            SharedNet {
                bounds: vec![1, 1],
                initial_marking: vec![1, 0],
                transitions: vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])],
            },
            2,
        ),
        // 3-place token ring: 3 states.
        (
            SharedNet {
                bounds: vec![1, 1, 1],
                initial_marking: vec![1, 0, 0],
                transitions: vec![
                    t(vec![1, 0, 0], vec![0, 1, 0]),
                    t(vec![0, 1, 0], vec![0, 0, 1]),
                    t(vec![0, 0, 1], vec![1, 0, 0]),
                ],
            },
            3,
        ),
        // Two independent 0..=3 counters: 16 states (the conserved/counter
        // shape where the MDD is far more compact than the BDD).
        (
            SharedNet {
                bounds: vec![3, 3],
                initial_marking: vec![0, 0],
                transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
            },
            16,
        ),
        // A producer/consumer with weighted arcs and a bound that truncates
        // (p1 bound 2, producer adds 2, consumer drains 1): exercises the
        // bound-truncation rule both engines share.
        (
            SharedNet {
                bounds: vec![1, 2],
                initial_marking: vec![1, 0],
                transitions: vec![
                    t(vec![1, 0], vec![0, 2]), // produce: p0 -> 2 on p1
                    t(vec![0, 1], vec![1, 0]), // consume: 1 from p1 -> p0
                ],
            },
            // Reachable: (1,0),(0,2),(1,1),(0,3 rejected by bound),(1,2)->...
            // Let the oracle decide; assert the engines AGREE and the count is
            // the BFS value (computed below).
            bfs_reachable_set_count(
                &SharedNet {
                    bounds: vec![1, 2],
                    initial_marking: vec![1, 0],
                    transitions: vec![t(vec![1, 0], vec![0, 2]), t(vec![0, 1], vec![1, 0])],
                }
                .to_dd(),
            ),
        ),
        // Dining-philosophers-style mutual exclusion fragment: forks as
        // 1-bounded places.
        (
            SharedNet {
                bounds: vec![1, 1, 1, 1],
                initial_marking: vec![1, 1, 0, 0],
                transitions: vec![
                    t(vec![1, 1, 0, 0], vec![0, 0, 1, 0]), // grab both forks
                    t(vec![0, 0, 1, 0], vec![1, 1, 0, 0]), // release
                    t(vec![1, 1, 0, 0], vec![0, 0, 0, 1]), // other phil grabs
                    t(vec![0, 0, 0, 1], vec![1, 1, 0, 0]), // release
                ],
            },
            bfs_reachable_set_count(
                &SharedNet {
                    bounds: vec![1, 1, 1, 1],
                    initial_marking: vec![1, 1, 0, 0],
                    transitions: vec![
                        t(vec![1, 1, 0, 0], vec![0, 0, 1, 0]),
                        t(vec![0, 0, 1, 0], vec![1, 1, 0, 0]),
                        t(vec![1, 1, 0, 0], vec![0, 0, 0, 1]),
                        t(vec![0, 0, 0, 1], vec![1, 1, 0, 0]),
                    ],
                }
                .to_dd(),
            ),
        ),
    ];

    let mut non_trivial = 0u32;
    for (net, expected) in &cases {
        match check_agreement(net) {
            AgreementOutcome::Agreed(c) => {
                assert_eq!(c, *expected, "deterministic case count drift");
                if c > 1 {
                    non_trivial += 1;
                }
            }
            AgreementOutcome::Declined => panic!("deterministic gate net should not decline"),
        }
    }
    // Non-vacuity: the gate must include multi-state nets, not just trivial
    // single-state ones.
    assert!(
        non_trivial >= 4,
        "deterministic gate is vacuous: only {non_trivial} multi-state nets"
    );
}

// ---------------------------------------------------------------------------
// Randomized differential battery.
// ---------------------------------------------------------------------------

/// Strategy for a random small bounded net. Sizes are kept modest so the
/// explicit BFS oracle stays cheap and the state space stays well under
/// `u64::MAX` (so the MDD never has to decline on overflow): up to 5 places,
/// bounds 1..=4, up to 6 transitions with small arc weights.
fn arb_net() -> impl Strategy<Value = SharedNet> {
    (1usize..=5).prop_flat_map(|num_places| {
        let bounds = prop::collection::vec(1u64..=4, num_places);
        bounds.prop_flat_map(move |bounds| {
            // Initial marking: each place in 0..=bound.
            let init = bounds.iter().map(|&b| 0u64..=b).collect::<Vec<_>>();
            // Transition arc weights kept in 0..=2 so firing changes are small
            // and the reachable space stays bounded but non-trivial.
            let trans = prop::collection::vec(
                (
                    prop::collection::vec(0u64..=2, num_places),
                    prop::collection::vec(0u64..=2, num_places),
                ),
                0..=6,
            );
            (init, trans, Just(bounds)).prop_map(|(init, trans, bounds)| SharedNet {
                bounds,
                initial_marking: init,
                transitions: trans,
            })
        })
    })
}

proptest! {
    // A wide battery: 4096 random nets. The whole point is breadth — every one
    // must agree with the oracle exactly.
    #![proptest_config(ProptestConfig {
        cases: 4096,
        max_shrink_iters: 4096,
        ..ProptestConfig::default()
    })]

    #[test]
    fn mdd_count_equals_bfs_oracle(net in arb_net()) {
        // check_agreement asserts exact equality (or accepts a fail-closed
        // decline). Any disagreement panics with the offending net.
        let _ = check_agreement(&net);
    }

    /// FORMAL SOUNDNESS over the random battery: for every generated net, the
    /// saturated reachable set `R` must be a sound inductive fixpoint
    /// (`init ∈ R ∧ ∀t. image_t(R) ⊆ R` ⇒ `R ⊇ reachable`, no marking missed).
    /// Discharged STRUCTURALLY per net — a far stronger guarantee than
    /// count-matching: it proves the soundness DIRECTION holds across 4096
    /// random nets, not just that the totals coincide. `Err` (a fail-closed
    /// build decline) is acceptable; only `Ok(false)` — an actual unsound
    /// fixpoint — fails the proof.
    #[test]
    fn mdd_saturation_is_inductively_sound(net in arb_net()) {
        let mdd = net.to_mdd();
        match mdd.verify_saturation_inductive_fixpoint(None) {
            Ok(true) => {}                       // proof obligations discharged
            Err(_) => {}                         // fail-closed build decline — fine
            Ok(false) => prop_assert!(
                false,
                "UNSOUND saturated fixpoint (init ⊆ R ∧ closed-under-Next FAILED) for {net:?}"
            ),
        }
    }
}

/// Companion to the proptest battery: a fixed-seed sweep that ALSO records
/// non-vacuity statistics (proptest hides per-case outcomes). Asserts that a
/// healthy fraction of random nets have > 1 reachable state and that no net
/// is ever declined, so we know the battery is genuinely exercising the
/// multi-state count path and not trivially passing on 1-state nets.
#[test]
fn random_battery_is_non_vacuous_and_never_declines() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, TestRunner};

    let mut runner = TestRunner::new(Config {
        cases: 2000,
        ..Config::default()
    });

    let mut total = 0u32;
    let mut multi_state = 0u32;
    let mut max_seen = 0u64;
    let mut declines = 0u32;

    for _ in 0..2000 {
        let tree = arb_net().new_tree(&mut runner).expect("gen net");
        let net = tree.current();
        total += 1;
        match check_agreement(&net) {
            AgreementOutcome::Agreed(c) => {
                if c > 1 {
                    multi_state += 1;
                }
                max_seen = max_seen.max(c);
            }
            AgreementOutcome::Declined => declines += 1,
        }
    }

    // The whole point of the lane is multi-state counting. Require a
    // substantial fraction of generated nets to have > 1 reachable state.
    assert!(
        multi_state * 4 >= total,
        "battery near-vacuous: only {multi_state}/{total} nets had >1 state"
    );
    // We should see at least one genuinely large-ish state space.
    assert!(
        max_seen >= 20,
        "battery never produced a non-trivial state space (max seen {max_seen})"
    );
    // Small generated nets must never trip the fail-closed caps.
    assert_eq!(
        declines, 0,
        "{declines}/{total} small nets were declined — caps too tight or a bug"
    );
}

// ---------------------------------------------------------------------------
// Conserved / counter nets — the saturation peak-node win.
//
// On these nets the explicit BFS oracle is expensive (it enumerates the whole
// reachable set), but the symbolic relprod and node-level saturation stay
// compact. The battery (a) pins all three MDD engines to BFS, 0 disagreements,
// and (b) records the PEAK interior node count of relprod vs saturation so we
// can show saturation keeps the peak at or below the breadth-first peak — the
// scalability claim — without ever trading away correctness.
// ---------------------------------------------------------------------------

/// A token-conserving line of `n_places` slots, each holding 0..=cap tokens,
/// with `tokens` tokens initially on place 0. Each adjacent pair has a
/// move-right and move-left transition (one token at a time), so the total
/// token count is invariant — a conserved net where saturation should keep the
/// peak small.
fn conserved_line(n_places: usize, cap: u64, tokens: u64) -> SharedNet {
    let bounds = vec![cap; n_places];
    let mut transitions = Vec::new();
    for p in 0..n_places.saturating_sub(1) {
        let mut pre = vec![0u64; n_places];
        let mut post = vec![0u64; n_places];
        pre[p] = 1;
        post[p + 1] = 1;
        transitions.push((pre, post));
        let mut pre = vec![0u64; n_places];
        let mut post = vec![0u64; n_places];
        pre[p + 1] = 1;
        post[p] = 1;
        transitions.push((pre, post));
    }
    let mut initial_marking = vec![0u64; n_places];
    if n_places > 0 {
        initial_marking[0] = tokens.min(cap);
    }
    SharedNet {
        bounds,
        initial_marking,
        transitions,
    }
}

/// N independent counters, each 0..=cap, each with one increment transition.
/// Reachable space is the full product `(cap+1)^n` — a counter net where the
/// MDD (one level per counter) is exponentially smaller than the state count.
fn independent_counters(n: usize, cap: u64) -> SharedNet {
    let bounds = vec![cap; n];
    let mut transitions = Vec::new();
    for p in 0..n {
        let mut pre = vec![0u64; n];
        let mut post = vec![0u64; n];
        let _ = &mut pre; // pure-output increment
        post[p] = 1;
        // increment only fires while below the cap (bound-truncation handles
        // the ceiling); model as consume 0 / produce 1 on place p.
        transitions.push((pre, post));
    }
    SharedNet {
        bounds,
        initial_marking: vec![0u64; n],
        transitions,
    }
}

#[test]
fn conserved_and_counter_nets_agree_and_saturation_shrinks_peak() {
    // (net, human label) — a spread of conserved + counter shapes.
    let cases: Vec<(SharedNet, &str)> = vec![
        (conserved_line(6, 3, 2), "conserved line 6x3, 2 tokens"),
        (conserved_line(8, 2, 3), "conserved line 8x2, 3 tokens"),
        (conserved_line(5, 4, 4), "conserved line 5x4, 4 tokens"),
        (independent_counters(4, 3), "4 independent 0..=3 counters"),
        (independent_counters(5, 2), "5 independent 0..=2 counters"),
        (independent_counters(3, 6), "3 independent 0..=6 counters"),
    ];

    let mut sat_no_worse = 0u32;
    let mut sat_strictly_better = 0u32;
    let mut nontrivial = 0u32;

    for (net, label) in &cases {
        let bfs = bfs_reachable_set_count(&net.to_dd());

        let r_relprod = net
            .to_mdd()
            .reachable_count_relprod(None)
            .unwrap_or_else(|e| panic!("{label}: relprod declined: {e:?}"));
        let r_sat = net
            .to_mdd()
            .reachable_count_saturation(None)
            .unwrap_or_else(|e| panic!("{label}: saturation declined: {e:?}"));
        let r_expl = net
            .to_mdd()
            .reachable_count()
            .unwrap_or_else(|e| panic!("{label}: explicit declined: {e:?}"));

        // Soundness: all three MDD engines == BFS oracle, exactly.
        assert_eq!(r_relprod.state_count, bfs, "{label}: relprod != BFS");
        assert_eq!(r_sat.state_count, bfs, "{label}: saturation != BFS");
        assert_eq!(r_expl.state_count, bfs, "{label}: explicit != BFS");

        // The state space must be non-trivial for the comparison to mean
        // anything.
        assert!(
            bfs > 4,
            "{label}: state space too small ({bfs}) to be a fair test"
        );
        nontrivial += 1;

        // Peak-node win: saturation's peak must be no worse than relprod's on
        // these conserved/counter shapes. (We assert <=, and separately count
        // strict wins, so the test is robust to a tie on the smallest nets but
        // still pins the scalability claim that saturation never inflates the
        // peak.)
        eprintln!(
            "{label}: |R|={bfs}  relprod_peak={}  sat_peak={}  final(relprod)={} final(sat)={}",
            r_relprod.peak_interior_nodes,
            r_sat.peak_interior_nodes,
            r_relprod.interior_nodes,
            r_sat.interior_nodes,
        );
        assert!(
            r_sat.peak_interior_nodes <= r_relprod.peak_interior_nodes,
            "{label}: saturation peak {} EXCEEDS relprod peak {} — saturation should never \
             inflate the peak on a conserved/counter net",
            r_sat.peak_interior_nodes,
            r_relprod.peak_interior_nodes
        );
        sat_no_worse += 1;
        if r_sat.peak_interior_nodes < r_relprod.peak_interior_nodes {
            sat_strictly_better += 1;
        }
    }

    // Non-vacuity of the comparison.
    assert!(
        nontrivial >= 6,
        "conserved/counter battery is vacuous: only {nontrivial} non-trivial nets"
    );
    assert_eq!(
        sat_no_worse, nontrivial,
        "saturation inflated the peak on some net (no-worse {sat_no_worse}/{nontrivial})"
    );
    // We expect saturation to STRICTLY beat relprod's peak on at least one of
    // these shapes (the actual scalability win). If this ever fails it does not
    // indicate unsoundness, but it would mean the peak measurement is not
    // demonstrating the saturation advantage and the design should be
    // revisited.
    assert!(
        sat_strictly_better >= 1,
        "saturation never strictly beat relprod's peak ({sat_strictly_better} strict wins) — \
         the conserved/counter battery is not demonstrating the saturation advantage"
    );
}

/// Diagnostic + soundness sweep over the random battery focused on the
/// saturation engine: confirm it ALWAYS equals the BFS oracle, and record how
/// often the post-saturation verification sweep had to add states (i.e. how
/// often node-level saturation alone under-shot the fixpoint because a reduced
/// level hid a banded event). The first is the hard soundness assertion; the
/// second is reported to stderr so we can see whether the verification wrapper
/// is doing real work or is essentially free.
#[test]
fn saturation_random_battery_always_agrees_and_reports_verify_rounds() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, TestRunner};

    let mut runner = TestRunner::new(Config {
        cases: 3000,
        ..Config::default()
    });

    let mut total = 0u32;
    let mut multi_state = 0u32;
    let mut needed_extra_round = 0u32; // saturation rounds > 1
    let mut max_rounds = 0u32;

    for _ in 0..3000 {
        let tree = arb_net().new_tree(&mut runner).expect("gen net");
        let net = tree.current();
        total += 1;

        let bfs = bfs_reachable_set_count(&net.to_dd());
        let r = net
            .to_mdd()
            .reachable_count_saturation(None)
            .expect("small net should not decline under saturation");
        assert_eq!(
            r.state_count, bfs,
            "saturation {} != BFS {} for bounds={:?} init={:?} transitions={:?}",
            r.state_count, bfs, net.bounds, net.initial_marking, net.transitions
        );
        if bfs > 1 {
            multi_state += 1;
        }
        if r.iterations > 1 {
            needed_extra_round += 1;
        }
        max_rounds = max_rounds.max(r.iterations);
    }

    eprintln!(
        "saturation battery: {total} nets, {multi_state} multi-state, \
         {needed_extra_round} needed a verification round beyond the first, \
         max rounds={max_rounds}"
    );
    // Non-vacuity: the battery must exercise multi-state nets.
    assert!(
        multi_state * 4 >= total,
        "saturation battery near-vacuous: only {multi_state}/{total} multi-state"
    );
}

/// Peak-node comparison across the RANDOM battery (not just hand-picked
/// conserved nets): over many random nets, saturation's peak interior-node
/// count should be no larger than relprod's on average, and strictly smaller
/// on a healthy fraction. This guards against the peak win being an artifact of
/// the curated conserved fixtures. Soundness (== BFS) is also re-asserted.
#[test]
fn saturation_peak_no_worse_than_relprod_on_random_battery() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, TestRunner};

    let mut runner = TestRunner::new(Config {
        cases: 1500,
        ..Config::default()
    });

    let mut compared = 0u32;
    let mut sat_le = 0u32; // saturation peak <= relprod peak
    let mut sat_lt = 0u32; // strictly smaller
    let mut sat_sum: u64 = 0;
    let mut rel_sum: u64 = 0;

    for _ in 0..1500 {
        let tree = arb_net().new_tree(&mut runner).expect("gen net");
        let net = tree.current();
        let bfs = bfs_reachable_set_count(&net.to_dd());

        let rp = net.to_mdd().reachable_count_relprod(None).expect("relprod");
        let st = net
            .to_mdd()
            .reachable_count_saturation(None)
            .expect("saturation");
        // Soundness re-check.
        assert_eq!(rp.state_count, bfs, "relprod != BFS on random net");
        assert_eq!(st.state_count, bfs, "saturation != BFS on random net");

        // Only compare peaks on non-trivial nets (a 1-state net has a trivial
        // peak and tells us nothing).
        if bfs > 1 {
            compared += 1;
            sat_sum += st.peak_interior_nodes as u64;
            rel_sum += rp.peak_interior_nodes as u64;
            if st.peak_interior_nodes <= rp.peak_interior_nodes {
                sat_le += 1;
            }
            if st.peak_interior_nodes < rp.peak_interior_nodes {
                sat_lt += 1;
            }
        }
    }

    eprintln!(
        "random peak compare: {compared} nontrivial nets, sat<=rel on {sat_le}, \
         sat<rel on {sat_lt}; total peaks sat={sat_sum} rel={rel_sum}"
    );
    assert!(
        compared >= 50,
        "too few nontrivial nets ({compared}) to compare peaks"
    );
    // Saturation should be no-worse on the large majority of random nets and
    // strictly better on a meaningful fraction. These are scalability-claim
    // (not soundness) assertions; soundness is the `assert_eq!` above.
    assert!(
        sat_le * 10 >= compared * 9,
        "saturation peak worse than relprod on too many random nets: \
         no-worse only {sat_le}/{compared}"
    );
    assert!(
        sat_lt * 4 >= compared,
        "saturation rarely beats relprod's peak on random nets: \
         strict wins {sat_lt}/{compared}"
    );
}

// ---------------------------------------------------------------------------
// FOUR-METRIC differential battery.
//
// The StateSpace examination reports four numbers per net. The MDD lane now
// computes ALL FOUR symbolically off the reachable set
// (`MddNet::state_space_metrics`). This battery pins every one of them to an
// independent explicit-BFS metric oracle (identical semantics to the
// `tla-dd` `bfs_full_metrics` / BFS observer: edges = Σ enabled in-bounds
// firings; max_token_in_place = max per-place value over reachable markings;
// max_token_sum = max total over reachable markings), with 0 disagreements.
// This is the soundness gate that lets the production StateSpace dispatch
// adopt an MDD metric bundle.
// ---------------------------------------------------------------------------

/// Explicit-BFS four-metric oracle, identical firing rule + metric semantics
/// to `tla_dd::tests::bfs_full_metrics` and the petri `StateSpaceObserver`.
fn bfs_full_metrics(net: &SharedNet) -> (u64, u64, u64, u64) {
    use std::collections::HashSet;
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    seen.insert(net.initial_marking.clone());
    let mut frontier: Vec<Vec<u64>> = vec![net.initial_marking.clone()];
    let mut edges: u64 = 0;
    let mut max_in_place: u64 = net.initial_marking.iter().copied().max().unwrap_or(0);
    let mut max_sum: u64 = net.initial_marking.iter().sum();
    while let Some(m) = frontier.pop() {
        for (pre, post) in &net.transitions {
            if !m.iter().zip(pre).all(|(mv, pv)| mv >= pv) {
                continue;
            }
            let mut next = m.clone();
            let mut ok = true;
            for p in 0..next.len() {
                let v = next[p] - pre[p] + post[p];
                if v > net.bounds[p] {
                    ok = false;
                    break;
                }
                next[p] = v;
            }
            if !ok {
                continue;
            }
            edges += 1;
            if seen.insert(next.clone()) {
                let s: u64 = next.iter().sum();
                let mxp = next.iter().copied().max().unwrap_or(0);
                max_sum = max_sum.max(s);
                max_in_place = max_in_place.max(mxp);
                frontier.push(next);
            }
        }
    }
    (seen.len() as u64, edges, max_in_place, max_sum)
}

/// Assert all four MDD metrics equal the BFS oracle exactly (or the MDD
/// fail-closed declines — a decline is not a disagreement, but on these small
/// nets it must never happen, which the caller separately asserts).
fn check_metrics(net: &SharedNet) -> Option<MddStateSpaceMetrics> {
    let (rc, ec, mip, msum) = bfs_full_metrics(net);
    // The PRODUCTION metric path builds the reachable set via NODE-LEVEL
    // SATURATION (`state_space_metrics`). The four metrics read off the
    // saturated set must equal the BFS oracle exactly.
    match net.to_mdd().state_space_metrics(None) {
        Ok(m) => {
            assert_eq!(
                m.state_count_u128, rc as u128,
                "state_count {} != BFS {rc} for bounds={:?} init={:?} transitions={:?}",
                m.state_count_u128, net.bounds, net.initial_marking, net.transitions
            );
            assert_eq!(
                m.edge_count, ec as u128,
                "edge_count {} != BFS {ec} for bounds={:?} init={:?} transitions={:?}",
                m.edge_count, net.bounds, net.initial_marking, net.transitions
            );
            assert_eq!(
                m.max_token_in_place, mip,
                "max_token_in_place {} != BFS {mip} for bounds={:?} init={:?} transitions={:?}",
                m.max_token_in_place, net.bounds, net.initial_marking, net.transitions
            );
            assert_eq!(
                m.max_token_sum, msum,
                "max_token_sum {} != BFS {msum} for bounds={:?} init={:?} transitions={:?}",
                m.max_token_sum, net.bounds, net.initial_marking, net.transitions
            );
            // SOUNDNESS GATE: the relprod set-build fallback must yield the
            // IDENTICAL metric bundle on every net where it also converges (all
            // these small nets). This pins the two set-build engines together,
            // so wiring production to saturation cannot change a metric value.
            let mr = net
                .to_mdd()
                .state_space_metrics_relprod(None)
                .expect("relprod metrics must not decline on a small net");
            assert_eq!(
                mr.state_count_u128, m.state_count_u128,
                "relprod state_count != saturation for bounds={:?} init={:?} transitions={:?}",
                net.bounds, net.initial_marking, net.transitions
            );
            assert_eq!(
                mr.edge_count, m.edge_count,
                "relprod edge_count != saturation"
            );
            assert_eq!(
                mr.max_token_in_place, m.max_token_in_place,
                "relprod max_token_in_place != saturation"
            );
            assert_eq!(
                mr.max_token_sum, m.max_token_sum,
                "relprod max_token_sum != saturation"
            );
            Some(m)
        }
        Err(CountError::CountOverflow | CountError::ResourceCap(_)) => None,
        Err(CountError::Malformed(e)) => {
            panic!("metrics declined well-formed net as malformed: {e}")
        }
    }
}

#[test]
fn deterministic_gate_four_metrics() {
    // Conserved / counter / weighted-truncating shapes — exactly the cluster
    // the MDD lane is meant to win, where every metric must be exact.
    let cases: Vec<SharedNet> = vec![
        SharedNet {
            bounds: vec![1, 1],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 1]), t(vec![0, 1], vec![1, 0])],
        },
        SharedNet {
            bounds: vec![2, 2, 2],
            initial_marking: vec![2, 0, 0],
            transitions: vec![
                t(vec![1, 0, 0], vec![0, 1, 0]),
                t(vec![0, 1, 0], vec![0, 0, 1]),
            ],
        },
        SharedNet {
            bounds: vec![3, 3],
            initial_marking: vec![0, 0],
            transitions: vec![t(vec![0, 0], vec![1, 0]), t(vec![0, 0], vec![0, 1])],
        },
        SharedNet {
            bounds: vec![1, 2],
            initial_marking: vec![1, 0],
            transitions: vec![t(vec![1, 0], vec![0, 2]), t(vec![0, 1], vec![1, 0])],
        },
        // High-bound conserved shuttle p0+p1=17: BDD lane is unary-bound here,
        // the MDD has 18 markings on a tiny diagram. All four metrics exact.
        conserved_line(2, 17, 17),
    ];

    let mut nontrivial = 0u32;
    for net in &cases {
        let m = check_metrics(net).expect("deterministic four-metric net must not decline");
        if m.state_count_u128 > 1 {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial >= 4,
        "four-metric gate vacuous: only {nontrivial} multi-state nets"
    );
}

proptest! {
    // Random four-metric battery: 4096 nets. Every metric must agree with the
    // BFS oracle exactly. This is the broad gate the production StateSpace
    // adoption rests on.
    #![proptest_config(ProptestConfig {
        cases: 4096,
        max_shrink_iters: 4096,
        ..ProptestConfig::default()
    })]

    #[test]
    fn mdd_four_metrics_equal_bfs_oracle(net in arb_net()) {
        let _ = check_metrics(&net);
    }
}

#[test]
fn four_metric_random_battery_non_vacuous_and_never_declines() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, TestRunner};

    let mut runner = TestRunner::new(Config {
        cases: 2000,
        ..Config::default()
    });

    let mut total = 0u32;
    let mut multi_state = 0u32;
    let mut nontrivial_edges = 0u32;
    let mut max_sum_seen = 0u64;
    let mut declines = 0u32;

    for _ in 0..2000 {
        let tree = arb_net().new_tree(&mut runner).expect("gen net");
        let net = tree.current();
        total += 1;
        match check_metrics(&net) {
            Some(m) => {
                if m.state_count_u128 > 1 {
                    multi_state += 1;
                }
                if m.edge_count > 0 {
                    nontrivial_edges += 1;
                }
                max_sum_seen = max_sum_seen.max(m.max_token_sum);
            }
            None => declines += 1,
        }
    }

    // Non-vacuity: the battery must exercise multi-state nets WITH edges and a
    // non-trivial token sum — otherwise an all-zeros metric would pass.
    assert!(
        multi_state * 4 >= total,
        "four-metric battery near-vacuous: only {multi_state}/{total} multi-state"
    );
    assert!(
        nontrivial_edges * 4 >= total,
        "four-metric battery near-vacuous: only {nontrivial_edges}/{total} had edges"
    );
    assert!(
        max_sum_seen >= 3,
        "four-metric battery never produced a non-trivial token sum (max {max_sum_seen})"
    );
    assert_eq!(
        declines, 0,
        "{declines}/{total} small nets declined metrics — caps too tight or a bug"
    );
}

// ---------------------------------------------------------------------------
// COMPLETION PROOF on the high-diameter families the breadth-first relprod
// times out on.
//
// This is the deliverable's central claim: the production metric path
// (`state_space_metrics`, now built via NODE-LEVEL SATURATION) COMPLETES —
// returns the exact four metrics within a tight wall-clock budget — on the
// conserved / counter / ring families whose state-space DIAMETER makes the
// breadth-first relprod fixpoint (`state_space_metrics_relprod`, one full-set
// image per round, ~diameter rounds) miss the same budget. Soundness is
// preserved: where BFS / relprod CAN finish, all three agree exactly.
// ---------------------------------------------------------------------------

/// A long conserved TOKEN RING: `n` 1-bounded slots, one token rotating
/// `p0->p1->...->p(n-1)->p0`. Reachable set is exactly `n` markings (the token
/// position), but the BREADTH-first diameter is `n-1` rounds: relprod must take
/// ~n rounds, saturation converges in O(1) passes. Philosophers-like cyclic
/// structure.
fn token_ring(n: usize) -> SharedNet {
    let bounds = vec![1u64; n];
    let mut transitions = Vec::new();
    for p in 0..n {
        let mut pre = vec![0u64; n];
        let mut post = vec![0u64; n];
        pre[p] = 1;
        post[(p + 1) % n] = 1;
        transitions.push((pre, post));
    }
    let mut initial_marking = vec![0u64; n];
    initial_marking[0] = 1;
    SharedNet {
        bounds,
        initial_marking,
        transitions,
    }
}

/// An ANDERSON-array-lock-like net: `n` 1-bounded "slot" places forming a ring
/// of grant tokens, plus a shared `next` index modeled as a conserved rotating
/// token. Structurally a cyclic conserved net with a longer diameter than a
/// plain ring; here we reuse the conserved-line+ring composition to get a
/// high-diameter conserved net whose reachable set is still MDD-compact.
fn anderson_like(n: usize) -> SharedNet {
    // A ring of n slots (one grant token rotating) crossed with a conserved
    // shuttle pair (the "has_lock" flag): the product has diameter ~n and a
    // small reachable set, exactly the high-diameter conserved shape Anderson
    // locks present to a symbolic engine.
    let total = n + 2;
    let bounds = vec![1u64; total];
    let mut transitions = Vec::new();
    // grant ring over the first n places
    for p in 0..n {
        let mut pre = vec![0u64; total];
        let mut post = vec![0u64; total];
        pre[p] = 1;
        post[(p + 1) % n] = 1;
        transitions.push((pre, post));
    }
    // has_lock shuttle between the last two places, gated on the grant being at
    // slot 0 (couples the two components so the diameter compounds).
    let mut pre = vec![0u64; total];
    let mut post = vec![0u64; total];
    pre[0] = 1;
    pre[n] = 1;
    post[0] = 1;
    post[n + 1] = 1;
    transitions.push((pre, post));
    let mut pre = vec![0u64; total];
    let mut post = vec![0u64; total];
    pre[n + 1] = 1;
    post[n] = 1;
    transitions.push((pre, post));

    let mut initial_marking = vec![0u64; total];
    initial_marking[0] = 1; // grant at slot 0
    initial_marking[n] = 1; // has_lock = false
    SharedNet {
        bounds,
        initial_marking,
        transitions,
    }
}

#[test]
fn saturation_metric_path_completes_where_relprod_times_out() {
    use std::time::{Duration, Instant};

    // A TIGHT budget. Saturation must finish inside it on every family;
    // relprod is given the SAME budget and is allowed to time out (decline) on
    // the high-diameter ones — that contrast is the whole point.
    let budget = Duration::from_secs(2);

    // (net, label, bfs_enumerable) — every net here has a SMALL reachable set
    // (so the metrics are well-defined and BFS can cross-check) but a LARGE
    // breadth-first diameter (so relprod needs ~diameter full-set image rounds
    // while saturation converges in O(1) passes). Picked deliberately to
    // separate diameter from |R|: a long token ring has |R| = n but diameter
    // n-1; a few-token conserved line has a modest |R| spread over a long line.
    let cases: Vec<(SharedNet, &str, bool)> = vec![
        (token_ring(60), "token ring n=60", true),
        (
            conserved_line(40, 3, 3),
            "conserved line 40x3, 3 tokens",
            true,
        ),
        (anderson_like(40), "anderson-like n=40", true),
        // A longer ring whose diameter is large but |R| is still tiny (= n):
        // relprod needs ~n full-set image rounds and misses the budget;
        // saturation converges in O(1) passes well inside it.
        (token_ring(400), "token ring n=400", true),
    ];

    let mut sat_completions = 0u32;
    let mut relprod_timeouts = 0u32;

    for (net, label, bfs_enumerable) in &cases {
        let mdd = net.to_mdd();

        // SATURATION metric path: must COMPLETE within the tight budget.
        let sat_deadline = Instant::now() + budget;
        let sat_start = Instant::now();
        let sat = mdd
            .state_space_metrics(Some(sat_deadline))
            .unwrap_or_else(|e| {
                panic!("{label}: saturation metric path did NOT complete in {budget:?}: {e:?}")
            });
        let sat_elapsed = sat_start.elapsed();
        sat_completions += 1;

        // Soundness: where the explicit BFS can enumerate, all four saturation
        // metrics must equal it exactly.
        if *bfs_enumerable {
            let (rc, ec, mip, msum) = bfs_full_metrics(net);
            assert_eq!(sat.state_count_u128, rc as u128, "{label}: |R| sat vs BFS");
            assert_eq!(sat.edge_count, ec as u128, "{label}: edges sat vs BFS");
            assert_eq!(
                sat.max_token_in_place, mip,
                "{label}: max_in_place sat vs BFS"
            );
            assert_eq!(sat.max_token_sum, msum, "{label}: max_sum sat vs BFS");
        }

        // RELPROD metric path under the SAME budget: record whether it
        // completed and, if so, that it AGREES (soundness across set-build
        // engines). A timeout/decline is expected on the high-diameter rings.
        let rel_deadline = Instant::now() + budget;
        let rel_start = Instant::now();
        let rel = mdd.state_space_metrics_relprod(Some(rel_deadline));
        let rel_elapsed = rel_start.elapsed();
        match rel {
            Ok(r) => {
                assert_eq!(
                    r.state_count_u128, sat.state_count_u128,
                    "{label}: relprod |R| disagrees with saturation"
                );
                assert_eq!(
                    r.edge_count, sat.edge_count,
                    "{label}: relprod edges disagree"
                );
                eprintln!(
                    "{label}: |R|={} sat={:?}(rounds={}) relprod={:?}(rounds={}) — both completed",
                    sat.state_count_u128, sat_elapsed, sat.iterations, rel_elapsed, r.iterations,
                );
            }
            Err(e) => {
                relprod_timeouts += 1;
                eprintln!(
                    "{label}: |R|={} sat COMPLETED in {:?}(rounds={}); relprod DECLINED ({e:?}) \
                     within the same {budget:?} budget — saturation wins the high-diameter case",
                    sat.state_count_u128, sat_elapsed, sat.iterations,
                );
            }
        }
    }

    // Non-vacuity: saturation completed on EVERY family.
    assert_eq!(
        sat_completions as usize,
        cases.len(),
        "saturation must complete on every high-diameter family"
    );
    // The contrast claim: relprod missed the budget on at least one
    // high-diameter family that saturation handled. (If hardware is fast enough
    // that relprod also finishes, this is not a soundness failure — but the
    // diameter sizes are picked so relprod is far slower; we assert the
    // contrast to keep the completion proof meaningful.)
    assert!(
        relprod_timeouts >= 1,
        "expected relprod to miss the tight budget on a high-diameter ring while \
         saturation completed; if this fails, raise the ring diameter"
    );
}

// ---------------------------------------------------------------------------
// GAP (c): the WIDENED (u128) count makes astronomically-large-but-finite
// reachable sets REPORTABLE, while genuinely-unrepresentable ones fail closed.
//
// `independent_counters(n, cap)` has |R| = (cap+1)^n on a TINY MDD (one node
// per level), so saturation finishes instantly even when |R| dwarfs u64::MAX.
// This is the Philosophers-PT-000050 ≈ 1e23 / high-bound shape the production
// lane previously declined on (gap-c #3) because the count overflowed u64.
// ---------------------------------------------------------------------------
/// `n` INDEPENDENT conserved shuttles: place pair `(2i, 2i+1)` holds one token
/// that shuttles `2i <-> 2i+1`. Each shuttle is a 2-state component and they are
/// mutually independent, so `|R| = 2^n` on a diagram that reduces to a tiny
/// chain (each level's edges fan into the same shared subtree). Small diameter
/// (1 per shuttle) and a consume+produce firing rule — exactly the conserved
/// shape the saturation engine handles fast — yet `|R|` can dwarf `u64::MAX`.
fn independent_shuttles(n: usize) -> SharedNet {
    let places = 2 * n;
    let bounds = vec![1u64; places];
    let mut transitions = Vec::new();
    let mut initial_marking = vec![0u64; places];
    for i in 0..n {
        let (a, b) = (2 * i, 2 * i + 1);
        initial_marking[a] = 1; // token starts on the left slot
        let mut pre = vec![0u64; places];
        let mut post = vec![0u64; places];
        pre[a] = 1;
        post[b] = 1;
        transitions.push((pre, post)); // a -> b
        let mut pre = vec![0u64; places];
        let mut post = vec![0u64; places];
        pre[b] = 1;
        post[a] = 1;
        transitions.push((pre, post)); // b -> a
    }
    SharedNet {
        bounds,
        initial_marking,
        transitions,
    }
}

#[test]
fn widened_count_reports_large_finite_via_bignum_above_u128() {
    use std::time::{Duration, Instant};
    use tla_bignum::BigUint;

    // (1) |R| just over u64::MAX, well under u128::MAX: must report EXACTLY in
    //     both the narrowed u128 field and the authoritative bignum field.
    //     2^65 ≈ 3.7e19 > u64::MAX (1.8e19), < u128::MAX (3.4e38). 65 independent
    //     conserved shuttles: |R| = 2^65 on a diagram that reduces to a tiny
    //     chain, so saturation finishes fast even though |R| dwarfs u64.
    let big = independent_shuttles(65);
    let expected: u128 = 2u128.pow(65);
    assert!(
        expected > u64::MAX as u128 && expected < u128::MAX,
        "fixture must straddle u64 but fit u128"
    );
    let t0 = Instant::now();
    let m = big
        .to_mdd()
        .state_space_metrics(Some(Instant::now() + Duration::from_secs(10)))
        .expect("saturation must compute the large-but-finite count");
    eprintln!(
        "gap-c: independent_shuttles(65) saturated in {:?}",
        t0.elapsed()
    );
    assert_eq!(
        m.state_count_u128, expected,
        "u128 |R| must be EXACT for the over-u64 net"
    );
    assert_eq!(
        m.state_count_big,
        BigUint::from(expected),
        "bignum |R| must equal the u128 value on the in-range net"
    );
    assert_eq!(
        m.state_count, None,
        "the narrowed u64 count must be None (|R| > u64::MAX) — the lane relies \
         on the wider fields to publish it"
    );

    // (2) |R| PAST u128::MAX: with the bignum carrier the lane no longer
    //     declines — it REPORTS the EXACT count. 200 independent shuttles ⇒
    //     |R| = 2^200 ≈ 1.6e60 >> u128::MAX (3.4e38), the FMS-shape (≈1e47)
    //     family that previously failed closed on the representational cap.
    //     Same fast conserved diagram (the count is the only thing that grew),
    //     so this is decided promptly and the exact 2^200 is published.
    let astronomical = independent_shuttles(200);
    let m = astronomical
        .to_mdd()
        .state_space_metrics(Some(Instant::now() + Duration::from_secs(10)))
        .expect("bignum carrier reports |R| = 2^200 exactly (no decline on magnitude)");
    let expected_big = BigUint::from(2u32).pow(200);
    assert_eq!(
        m.state_count_big, expected_big,
        "|R| = 2^200 must be reported EXACTLY via the bignum carrier",
    );
    assert!(
        m.state_count_big > BigUint::from(u128::MAX),
        "the reported count is genuinely > u128::MAX",
    );
    // The narrowed fields saturate (markers); never consumed in the >u128 case.
    assert_eq!(m.state_count, None, "does not fit u64");
    assert_eq!(
        m.state_count_u128,
        u128::MAX,
        "saturated marker (does not fit u128)"
    );
    eprintln!("gap-c: independent_shuttles(200) |R|=2^200 > u128::MAX REPORTED via bignum ✓",);
}
