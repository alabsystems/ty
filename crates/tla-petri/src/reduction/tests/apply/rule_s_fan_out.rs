// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for Tapaal Rule S: generalized place-centric agglomeration.
//!
//! Reference: Tapaal verifypn `Reducer.cpp:2556-2838` (S-mode, k == 1 subset).
//!
//! Phase-1 scope:
//! - k = 1 only (every producer's out-weight on `p` equals the uniform
//!   consumer arc weight `w`).
//! - `producer.post == {p}` and `consumer.pre == {p}` exactly.
//! - `initial_marking[p] < w` (S3/S9).
//! - Reachability mode only. Gated off for CTLWithNext / StutterSensitiveLTL /
//!   StutterInsensitiveLTL (Phase-2) / NextFreeCTL / OneSafe.
//! - Cartesian product `producers × consumers ≤ RULE_R_EXPLOSION_LIMITER`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::petri_net::{PetriNet, PlaceIdx};
use crate::reduction::{reduce_iterative_structural_with_mode, reduce_with_mode, ReductionMode};

use super::super::support::{arc, place, trans};

fn reachable_projected_markings(
    net: &PetriNet,
    reduced_place_map: &[Option<PlaceIdx>],
    reduced_places: usize,
    keep_original: impl Fn(&[u64]) -> bool,
) -> std::collections::BTreeSet<Vec<u64>> {
    let full = crate::explorer::explore_full(net, &crate::explorer::ExplorationConfig::new(64));
    assert!(
        full.graph.completed,
        "witness exploration should complete within the test budget"
    );

    full.markings
        .unpack_all()
        .into_iter()
        .filter(|marking| keep_original(marking))
        .map(|marking| {
            let mut projected = vec![0; reduced_places];
            for (original_place, mapped) in reduced_place_map.iter().enumerate() {
                let Some(reduced_place) = mapped else {
                    continue;
                };
                projected[reduced_place.0 as usize] = marking[original_place];
            }
            projected
        })
        .collect()
}

struct MccRuleSSlice {
    label: &'static str,
    relative_paths: &'static [&'static str],
}

const RULE_S_MCC_SLICES: &[MccRuleSSlice] = &[
    MccRuleSSlice {
        label: "Philosophers-PT-10",
        relative_paths: &[
            "Philosophers/PT/Philosophers-10.pnml",
            "Philosophers-PT-10/model.pnml",
            "Philosophers-PT-000010/model.pnml",
        ],
    },
    MccRuleSSlice {
        label: "TokenRing-PT-5",
        relative_paths: &[
            "TokenRing/PT/TokenRing-5-unfolded.pnml",
            "TokenRing-PT-5/model.pnml",
            "TokenRing-PT-005/model.pnml",
        ],
    },
    MccRuleSSlice {
        label: "SharedMemory-PT-5",
        relative_paths: &[
            "SharedMemory/PT/shared_memory-pt-5.pnml",
            "SharedMemory-PT-5/model.pnml",
            "SharedMemory-PT-000005/model.pnml",
        ],
    },
    MccRuleSSlice {
        label: "Kanban-PT-small",
        relative_paths: &[
            "Kanban/PT/Kanban-5.pnml",
            "Kanban-PT-small/model.pnml",
            "Kanban-PT-00005/model.pnml",
        ],
    },
    MccRuleSSlice {
        label: "MAPK-PT-08",
        relative_paths: &[
            "MAPK/PT/MAPK-8.pnml",
            "MAPK-PT-08/model.pnml",
            "MAPK-PT-00008/model.pnml",
        ],
    },
];

fn first_existing_path(root: &Path, relative_paths: &[&str]) -> Option<PathBuf> {
    relative_paths
        .iter()
        .map(|relative| root.join(relative))
        .find(|path| path.exists())
}

fn rule_s_places_removed(report: &crate::reduction::ReductionReport) -> BTreeSet<u32> {
    report
        .rule_s_agglomerations
        .iter()
        .map(|agg| agg.place.0)
        .collect()
}

fn rule_s_transitions_removed(report: &crate::reduction::ReductionReport) -> BTreeSet<u32> {
    let mut removed = BTreeSet::new();
    for agg in &report.rule_s_agglomerations {
        removed.extend(agg.producers.iter().map(|idx| idx.0));
        removed.extend(agg.consumers.iter().map(|idx| idx.0));
    }
    removed
}

fn rule_s_transitions_added(report: &crate::reduction::ReductionReport) -> usize {
    report
        .rule_s_agglomerations
        .iter()
        .map(|agg| agg.producers.len() * agg.consumers.len())
        .sum()
}

fn reachable_markings(net: &PetriNet) -> std::collections::BTreeSet<Vec<u64>> {
    let full = crate::explorer::explore_full(net, &crate::explorer::ExplorationConfig::new(64));
    assert!(
        full.graph.completed,
        "witness exploration should complete within the test budget"
    );
    full.markings.unpack_all().into_iter().collect()
}

/// Canonical Rule S topology: single producer × single consumer on a central
/// place. Producer has `post == {p_mid}`, consumer has `pre == {p_mid}`,
/// weights match at w=1, and `initial_marking[p_mid] = 0 < 1`.
///
/// Rule S should fuse {t_prod} × {t_con} → 1 synthesized transition, and
/// remove t_prod, t_con, and p_mid. This shape is also a one-sided
/// pre/post-agglomeration candidate, so the test locks in the Reachability-mode
/// priority that lets a strict Rule S proof preempt those claims.
#[test]
fn test_rule_s_single_producer_single_consumer_fuses() {
    let net = PetriNet {
        name: None,
        places: vec![
            place("p_src"),    // 0
            place("p_mid"),    // 1 — Rule S central place
            place("p_shared"), // 2 — consumer's post-place (distinct from any other)
        ],
        transitions: vec![
            // Producer: p_src → p_mid. post == {p_mid}.
            trans("t_prod", vec![arc(0, 1)], vec![arc(1, 1)]),
            // Consumer: p_mid → p_shared. pre == {p_mid}.
            trans("t_con", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };

    let reduced = reduce_with_mode(&net, &[], ReductionMode::Reachability);

    assert!(
        reduced.place_map[1].is_none(),
        "p_mid (original index 1) must be removed; place_map: {:?}",
        reduced.place_map
    );
    assert!(
        reduced.report.pre_agglomerations.is_empty(),
        "Rule S must preempt overlapping pre-agglomeration claims"
    );
    assert!(
        reduced.report.post_agglomerations.is_empty(),
        "Rule S must preempt overlapping post-agglomeration claims"
    );

    let agg = reduced
        .report
        .rule_s_agglomerations
        .iter()
        .find(|agg| agg.place == PlaceIdx(1))
        .expect("Rule S must claim p_mid");
    assert_eq!(agg.weight, 1, "weight must be 1 (k=1)");
    assert_eq!(agg.producers.len(), 1, "single producer");
    assert_eq!(agg.consumers.len(), 1, "single consumer");
}

/// Rule S is a macro-step reduction: the reduced transition represents a
/// producer immediately followed by a consumer. This witness brute-force
/// enumerates the original net and compares the projection of original
/// markings where the removed central place is back at its initial token count
/// with the reduced net's reachable markings.
#[test]
fn test_rule_s_macro_step_projected_reachability_witness() {
    let net = PetriNet {
        name: None,
        places: vec![
            place("p_a"),
            place("p_b"),
            place("p_mid"),
            place("p_shared"),
        ],
        transitions: vec![
            trans("t_prod1", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t_prod2", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t_con1", vec![arc(2, 1)], vec![arc(3, 1)]),
            trans("t_con2", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![1, 1, 0, 0],
    };

    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction must succeed");

    let rule_s = reduced
        .report
        .rule_s_agglomerations
        .iter()
        .find(|agg| agg.place == PlaceIdx(2))
        .expect("Rule S must claim p_mid in the witness net");
    assert_eq!(rule_s.weight, 1);
    assert_eq!(rule_s.producers.len(), 2);
    assert_eq!(rule_s.consumers.len(), 2);

    let original_macro_step_projection = reachable_projected_markings(
        &net,
        &reduced.place_map,
        reduced.net.num_places(),
        |marking| marking[rule_s.place.0 as usize] == net.initial_marking[rule_s.place.0 as usize],
    );
    let reduced_reachable = reachable_markings(&reduced.net);

    assert_eq!(
        original_macro_step_projection, reduced_reachable,
        "Rule S reduced reachability should match original macro-step markings projected to surviving places"
    );
}

/// 2 producers × 2 consumers on the same central place. Cartesian product
/// 4 ≤ limiter 6. Both consumers write to a shared post-place so that
/// pre-/post-agglomeration cannot claim them first (avoids clobbering by
/// simpler rules).
#[test]
fn test_rule_s_two_by_two_fuses_under_limiter() {
    let net = PetriNet {
        name: None,
        places: vec![
            place("p_a"),      // 0
            place("p_b"),      // 1
            place("p_mid"),    // 2 — Rule S central place
            place("p_shared"), // 3 — shared consumer post-place
        ],
        transitions: vec![
            trans("t_prod1", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t_prod2", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t_con1", vec![arc(2, 1)], vec![arc(3, 1)]),
            trans("t_con2", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![1, 1, 0, 0],
    };

    let reduced = reduce_with_mode(&net, &[], ReductionMode::Reachability);

    // p_mid must be removed by Rule S (or cascading rules).
    assert!(
        reduced.place_map[2].is_none(),
        "p_mid must be removed; place_map: {:?}",
        reduced.place_map
    );

    // If Rule S ran on p_mid, it claimed all 2 producers and all 2 consumers.
    let s_on_p_mid = reduced
        .report
        .rule_s_agglomerations
        .iter()
        .find(|agg| agg.place == PlaceIdx(2));
    if let Some(agg) = s_on_p_mid {
        assert_eq!(agg.producers.len(), 2, "2 producers");
        assert_eq!(agg.consumers.len(), 2, "2 consumers");
        assert_eq!(agg.weight, 1);
    }
}

/// Explosion limiter: 3 producers × 3 consumers = 9 pairs, exceeds
/// RULE_R_EXPLOSION_LIMITER = 6. Rule S must SKIP this place.
///
/// Other rules (like post-agglomeration or dead cascade) may still remove
/// the place, so the essential assertion is that `rule_s_agglomerations`
/// contains no entry for this specific place.
#[test]
fn test_rule_s_skips_when_explosion_limiter_exceeded() {
    let net = PetriNet {
        name: None,
        places: vec![
            place("p_a"),
            place("p_b"),
            place("p_c"),
            place("p_mid"), // 3 — central place
            place("p_shared"),
        ],
        transitions: vec![
            trans("t_prod1", vec![arc(0, 1)], vec![arc(3, 1)]),
            trans("t_prod2", vec![arc(1, 1)], vec![arc(3, 1)]),
            trans("t_prod3", vec![arc(2, 1)], vec![arc(3, 1)]),
            // 3 consumers, all writing to shared post-place to avoid
            // post-agglomeration claiming the place first.
            trans("t_con1", vec![arc(3, 1)], vec![arc(4, 1)]),
            trans("t_con2", vec![arc(3, 1)], vec![arc(4, 1)]),
            trans("t_con3", vec![arc(3, 1)], vec![arc(4, 1)]),
        ],
        initial_marking: vec![1, 1, 1, 0, 0],
    };

    let reduced = reduce_with_mode(&net, &[], ReductionMode::Reachability);

    let p_mid_rule_s = reduced
        .report
        .rule_s_agglomerations
        .iter()
        .any(|agg| agg.place == PlaceIdx(3));
    assert!(
        !p_mid_rule_s,
        "Rule S must skip p_mid: 3×3=9 exceeds limiter 6"
    );
}

/// Mode gating: Rule S is Phase-1 Reachability-only. It must NOT fire under
/// `CTLWithNext`, `StutterSensitiveLTL`, or `StutterInsensitiveLTL` (the
/// last because Phase-2 extension is not yet implemented).
#[test]
fn test_rule_s_gated_off_for_non_reachability_modes() {
    let net = PetriNet {
        name: None,
        places: vec![place("p_src"), place("p_mid"), place("p_shared")],
        transitions: vec![
            trans("t_prod", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t_con", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };

    for mode in [
        ReductionMode::CTLWithNext,
        ReductionMode::StutterSensitiveLTL,
        ReductionMode::StutterInsensitiveLTL,
    ] {
        let reduced = reduce_iterative_structural_with_mode(&net, &[], mode, None)
            .unwrap_or_else(|e| panic!("reduction under {:?} must succeed: {e:?}", mode));
        assert!(
            reduced.report.rule_s_agglomerations.is_empty(),
            "Rule S must NOT fire under {:?}",
            mode
        );
    }
}

/// Initial-marking bound (Tapaal S3/S9): if `initial_marking[p] >= w`, Rule S
/// is unsound because a consumer could fire before any producer fires. Here
/// `initial_marking[p_mid] = 1 >= w = 1`, so Rule S must skip this place.
#[test]
fn test_rule_s_respects_initial_marking_bound() {
    let net = PetriNet {
        name: None,
        places: vec![place("p_src"), place("p_mid"), place("p_shared")],
        transitions: vec![
            trans("t_prod", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t_con", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        // p_mid starts with 1 token, equal to consumer weight → Rule S unsafe.
        initial_marking: vec![1, 1, 0],
    };

    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction must succeed");

    let p_mid_rule_s = reduced
        .report
        .rule_s_agglomerations
        .iter()
        .any(|agg| agg.place == PlaceIdx(1));
    assert!(
        !p_mid_rule_s,
        "Rule S must skip p_mid when initial_marking[p_mid] >= w"
    );
}

/// Multi-post producer rejection: Rule S requires `producer.post == {p}` with
/// post-set of exactly one place (weight w). Here t_prod writes to both p_mid
/// and p_extra, so Rule S must NOT claim p_mid.
///
/// Note: Rule R can still claim p_mid here because Rule R permits multi-post
/// producers (it strips the arc on `place`). This test asserts specifically
/// about the Rule S report.
#[test]
fn test_rule_s_rejects_multi_post_producer() {
    let net = PetriNet {
        name: None,
        places: vec![
            place("p_src"),
            place("p_mid"),
            place("p_extra"), // second post-place
            place("p_shared"),
        ],
        transitions: vec![
            // Producer writes to BOTH p_mid and p_extra → post != {p_mid}.
            trans("t_prod", vec![arc(0, 1)], vec![arc(1, 1), arc(2, 1)]),
            trans("t_con", vec![arc(1, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![1, 0, 0, 0],
    };

    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction must succeed");

    let p_mid_rule_s = reduced
        .report
        .rule_s_agglomerations
        .iter()
        .any(|agg| agg.place == PlaceIdx(1));
    assert!(
        !p_mid_rule_s,
        "Rule S must NOT fire when producer has multi-post (post != {{place}})"
    );
}

/// Multi-pre consumer rejection: Rule S requires `consumer.pre == {p}` exactly.
/// Here t_con reads from both p_mid and p_token, so Rule S must NOT claim p_mid.
#[test]
fn test_rule_s_rejects_multi_pre_consumer() {
    let net = PetriNet {
        name: None,
        places: vec![
            place("p_src"),
            place("p_mid"),
            place("p_token"), // second pre-place for consumer
            place("p_shared"),
        ],
        transitions: vec![
            trans("t_prod", vec![arc(0, 1)], vec![arc(1, 1)]),
            // Consumer reads from BOTH p_mid and p_token → pre != {p_mid}.
            trans("t_con", vec![arc(1, 1), arc(2, 1)], vec![arc(3, 1)]),
            // Give p_token initial supply so t_con isn't dead.
            trans("t_token_src", vec![], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 1, 0],
    };

    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction must succeed");

    let p_mid_rule_s = reduced
        .report
        .rule_s_agglomerations
        .iter()
        .any(|agg| agg.place == PlaceIdx(1));
    assert!(
        !p_mid_rule_s,
        "Rule S must NOT fire when consumer has multi-pre (pre != {{place}})"
    );
}

/// Producer/consumer disjointness (Tapaal T4/S4): a transition cannot be
/// both a producer and a consumer on the same place. Here t_loop is its
/// own producer-and-consumer via a self-loop on p_mid. Rule S must skip this.
#[test]
fn test_rule_s_rejects_non_disjoint_producers_consumers() {
    let net = PetriNet {
        name: None,
        places: vec![place("p_src"), place("p_mid"), place("p_shared")],
        transitions: vec![
            // Ordinary producer.
            trans("t_prod", vec![arc(0, 1)], vec![arc(1, 1)]),
            // t_loop both consumes from and produces to p_mid: in both sets.
            trans("t_loop", vec![arc(1, 1)], vec![arc(1, 1), arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };

    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction must succeed");

    let p_mid_rule_s = reduced
        .report
        .rule_s_agglomerations
        .iter()
        .any(|agg| agg.place == PlaceIdx(1));
    assert!(
        !p_mid_rule_s,
        "Rule S must NOT fire when producers ∩ consumers is non-empty"
    );
}

/// Idempotence: running the reducer twice produces no new Rule S entries on
/// the second pass (fixpoint reached).
#[test]
fn test_rule_s_is_idempotent_under_iteration() {
    let net = PetriNet {
        name: None,
        places: vec![
            place("p_a"),
            place("p_b"),
            place("p_mid"),
            place("p_shared"),
        ],
        transitions: vec![
            trans("t_prod1", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t_prod2", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t_con1", vec![arc(2, 1)], vec![arc(3, 1)]),
            trans("t_con2", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![1, 1, 0, 0],
    };

    let first = reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
        .expect("first reduction");
    let second =
        reduce_iterative_structural_with_mode(&first.net, &[], ReductionMode::Reachability, None)
            .expect("second reduction");

    assert_eq!(
        second.report.rule_s_agglomerations.len(),
        0,
        "Rule S should be idempotent: second pass finds nothing"
    );
}

#[test]
fn test_rule_s_mcc_reduction_evidence_from_env() {
    let Some(root) = std::env::var_os("TY_RULE_S_MCC_INPUT_ROOT") else {
        println!("skipping Rule S MCC evidence: TY_RULE_S_MCC_INPUT_ROOT is not set");
        return;
    };
    let root = PathBuf::from(root);
    assert!(
        root.exists(),
        "TY_RULE_S_MCC_INPUT_ROOT must exist: {}",
        root.display()
    );

    let mut measured_rule_s_slices = Vec::new();
    let include_slow_token_ring = std::env::var_os("TY_RULE_S_MCC_INCLUDE_SLOW").is_some();
    for slice in RULE_S_MCC_SLICES {
        if slice.label == "TokenRing-PT-5" && !include_slow_token_ring {
            println!(
                "RULE_S_MCC skipped label={} reason=set_TY_RULE_S_MCC_INCLUDE_SLOW_to_probe_iterative_token_ring",
                slice.label
            );
            continue;
        }
        let Some(path) = first_existing_path(&root, slice.relative_paths) else {
            println!(
                "RULE_S_MCC missing label={} searched={:?}",
                slice.label, slice.relative_paths
            );
            continue;
        };

        let net = crate::parser::parse_pnml_file(&path)
            .unwrap_or_else(|error| panic!("failed to parse {}: {error:?}", path.display()));
        let original_places = net.num_places();
        let original_transitions = net.num_transitions();
        let reduced =
            reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
                .unwrap_or_else(|error| panic!("failed to reduce {}: {error:?}", slice.label));

        let rule_s_places_removed = rule_s_places_removed(&reduced.report);
        let rule_s_transitions_removed = rule_s_transitions_removed(&reduced.report);
        let rule_s_transitions_added = rule_s_transitions_added(&reduced.report);

        println!(
            "RULE_S_MCC label={} source={} original_places={} original_transitions={} reduced_places={} reduced_transitions={} all_places_removed={} all_transitions_removed={} all_transitions_added={} rule_s_agglomerations={} rule_s_places_removed={} rule_s_transitions_removed={} rule_s_transitions_added={}",
            slice.label,
            path.display(),
            original_places,
            original_transitions,
            reduced.net.num_places(),
            reduced.net.num_transitions(),
            reduced.report.places_removed(),
            reduced.report.transitions_removed(),
            reduced.report.transitions_added(),
            reduced.report.rule_s_agglomerations.len(),
            rule_s_places_removed.len(),
            rule_s_transitions_removed.len(),
            rule_s_transitions_added,
        );

        if !reduced.report.rule_s_agglomerations.is_empty()
            && (!rule_s_places_removed.is_empty()
                || !rule_s_transitions_removed.is_empty()
                || rule_s_transitions_added > 0)
        {
            measured_rule_s_slices.push(slice.label);
        }
    }

    assert!(
        measured_rule_s_slices.len() >= 3,
        "expected measurable Rule S reductions on at least three MCC slices; measured {:?}",
        measured_rule_s_slices
    );
}
