// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness tests for the incremental enabled-set update.
//!
//! The load-bearing invariant: for EVERY state and EVERY fired transition, the
//! incrementally-derived child enabled bitmap is BIT-FOR-BIT identical to a
//! full scan of the child marking (`is_enabled` for every transition). These
//! tests exercise the corner cases the incremental update must mirror exactly:
//! weighted arcs, source transitions (no inputs), and self-loops / read arcs
//! (a place in both `t.inputs` and `t.outputs`).

use super::{full_scan_enabled_bitmap, incremental_enabled_update, PlaceConsumerIndex};
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};

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

/// Re-firing every enabled transition from `marking`, assert the incremental
/// child bitmap equals the full-scan child bitmap for each.
fn assert_incremental_matches_full_scan(net: &PetriNet, marking: &[u64]) {
    let nt = net.num_transitions();
    let index = PlaceConsumerIndex::build(net);

    let mut parent_enabled = Vec::new();
    full_scan_enabled_bitmap(net, marking, nt, &mut parent_enabled);

    let mut seen = vec![false; net.num_places()];
    let mut incremental_child = Vec::new();
    let mut full_child = Vec::new();

    for tidx in 0..nt {
        let t = TransitionIdx(tidx as u32);
        if !net.is_enabled(marking, t) {
            continue;
        }
        let child = net.fire(marking, t).expect("enabled transition fires");

        incremental_enabled_update(
            net,
            &index,
            &parent_enabled,
            &child,
            t,
            &mut seen,
            &mut incremental_child,
        );
        full_scan_enabled_bitmap(net, &child, nt, &mut full_child);

        assert_eq!(
            incremental_child, full_child,
            "incremental child enabled set diverged after firing {t:?} at {marking:?}"
        );
        // The scratch must be clean after each call (all-false), so a stale
        // entry can never leak into the next firing's update.
        assert!(seen.iter().all(|&b| !b), "seen scratch not reset");
    }
}

#[test]
fn incremental_matches_full_scan_weighted_arcs() {
    // Weighted input/output arcs: the incremental re-eval must compare against
    // `arc.weight`, not assume weight 1.
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 3)], vec![arc(1, 2)]),
            trans("t1", vec![arc(1, 2)], vec![arc(2, 1)]),
            trans("t2", vec![arc(0, 5)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![6, 0, 0],
    };
    for m in [
        vec![6, 0, 0],
        vec![3, 2, 0],
        vec![0, 4, 0],
        vec![5, 0, 0],
        vec![2, 2, 0],
    ] {
        assert_incremental_matches_full_scan(&net, &m);
    }
}

#[test]
fn incremental_matches_full_scan_source_transition() {
    // t0 is a SOURCE transition (no inputs) — always enabled, never disabled by
    // any firing. The incremental update must copy its `true` untouched (it is
    // in no consumer list).
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![], vec![arc(0, 1)]),
            trans("t1", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t2", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![0, 0],
    };
    for m in [vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 1]] {
        assert_incremental_matches_full_scan(&net, &m);
        assert!(net.is_enabled(&m, TransitionIdx(0)));
    }
}

#[test]
fn incremental_matches_full_scan_self_loop() {
    // t0 has p0 as BOTH an input (weight 1) and an output (weight 1): a
    // self-loop / read arc. Firing it leaves p0 unchanged but the place must be
    // visited exactly once by the seen-scratch dedup.
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(0, 1), arc(1, 1)]),
            trans("t1", vec![arc(0, 2)], vec![arc(1, 1)]),
            trans("t2", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![2, 0],
    };
    for m in [vec![2, 0], vec![1, 1], vec![2, 3], vec![1, 0]] {
        assert_incremental_matches_full_scan(&net, &m);
    }
}

#[test]
fn incremental_disables_via_consumed_input() {
    // Firing t0 drains p0 below t1's threshold, DISABLING t1 — the incremental
    // update must flip t1 false (a transition leaving the enabled set).
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 2)], vec![arc(1, 1)]),
            trans("t1", vec![arc(0, 1)], vec![arc(1, 1)]),
        ],
        initial_marking: vec![2, 0],
    };
    assert_incremental_matches_full_scan(&net, &[2, 0]);
    let index = PlaceConsumerIndex::build(&net);
    let mut parent = Vec::new();
    full_scan_enabled_bitmap(&net, &[2, 0], 2, &mut parent);
    assert_eq!(parent, vec![true, true]);
    let child = net.fire(&[2, 0], TransitionIdx(0)).unwrap();
    let mut seen = vec![false; 2];
    let mut out = Vec::new();
    incremental_enabled_update(
        &net,
        &index,
        &parent,
        &child,
        TransitionIdx(0),
        &mut seen,
        &mut out,
    );
    assert_eq!(
        out,
        vec![false, false],
        "t1 must be disabled after p0 drained"
    );
}

#[test]
fn incremental_matches_full_scan_pure_sink_output() {
    // t0 deposits into a pure SINK place (no transition consumes from it):
    // firing t0 triggers zero re-evaluations from the sink's empty consumer list.
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("sink")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(0, 1)], vec![arc(1, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };
    for m in [vec![1, 0, 0], vec![2, 0, 0], vec![0, 0, 5]] {
        assert_incremental_matches_full_scan(&net, &m);
    }
}

#[test]
fn place_consumer_index_lists_only_input_arcs() {
    // The index keys off INPUT arcs only (enabledness depends on inputs).
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(0, 1), arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 1, 0],
    };
    let index = PlaceConsumerIndex::build(&net);
    assert_eq!(index.consumers_of(0), &[TransitionIdx(0), TransitionIdx(1)]);
    assert_eq!(index.consumers_of(1), &[TransitionIdx(1)]);
    assert!(index.consumers_of(2).is_empty());
}
