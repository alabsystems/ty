// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-check the SOUND structural deadlock-freedom shortcut
//! (`structural::lp_siphon_deadlock_free`) against the EXHAUSTIVE explicit-state
//! engine on real MCC benchmark models.
//!
//! Contract under test (the only thing that matters for soundness):
//! - When the shortcut returns `Some(true)` ("deadlock-free"), the exhaustive
//!   engine must NEVER find a reachable deadlock. A wrong `Some(true)` is the
//!   worst MCC outcome, so this is the load-bearing assertion.
//! - On models with a genuine deadlock the shortcut must DECLINE (`None`); it is
//!   the exact engine that produces the witnessed `TRUE`.
//!
//! Models are read from `tmp_benchmark_models/<model>/model.pnml` (the local
//! corpus). Absent models are SKIPPED so the suite stays green on a clean
//! checkout.

use std::path::PathBuf;

use crate::examinations::deadlock::DeadlockObserver;
use crate::explorer::{explore_observer, ExplorationConfig};
use crate::parser::parse_pnml_dir;
use crate::petri_net::PetriNet;
use crate::structural::lp_siphon_deadlock_free;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn model_dir(model: &str) -> PathBuf {
    workspace_root().join("tmp_benchmark_models").join(model)
}

fn try_load(model: &str) -> Option<PetriNet> {
    let dir = model_dir(model);
    if !dir.join("model.pnml").exists() {
        eprintln!("SKIP: {model} not present under tmp_benchmark_models");
        return None;
    }
    match parse_pnml_dir(&dir) {
        Ok(net) => Some(net),
        Err(error) => {
            eprintln!("SKIP: {model} failed to parse: {error}");
            None
        }
    }
}

/// Exhaustive deadlock oracle: full (no-POR) BFS over the reachable set.
///
/// Returns:
/// - `Some(true)`  — a reachable deadlock exists (ground truth TRUE),
/// - `Some(false)` — exploration completed with no deadlock (ground truth FALSE),
/// - `None`        — exploration did not complete within the state cap (unknown).
fn exhaustive_deadlock(net: &PetriNet, max_states: usize) -> Option<bool> {
    let config = ExplorationConfig::new(max_states);
    let mut observer = DeadlockObserver::new();
    let result = explore_observer(net, &config, &mut observer);
    if observer.found_deadlock() {
        Some(true)
    } else if result.completed {
        Some(false)
    } else {
        None
    }
}

/// Core soundness invariant: the structural shortcut never disagrees with the
/// exhaustive engine. If the shortcut says "deadlock-free", the exhaustive
/// engine must not be able to reach a deadlock.
fn assert_shortcut_consistent_with_exhaustive(model: &str, max_states: usize) {
    let Some(net) = try_load(model) else {
        return;
    };
    let shortcut = lp_siphon_deadlock_free(&net, None);
    let oracle = exhaustive_deadlock(&net, max_states);
    eprintln!("{model}: shortcut={shortcut:?} exhaustive={oracle:?}");

    if shortcut == Some(true) {
        assert_ne!(
            oracle,
            Some(true),
            "{model}: structural shortcut certified DEADLOCK-FREE but the \
             exhaustive engine found a reachable deadlock — UNSOUND",
        );
    }
    // The shortcut must never emit anything but `Some(true)` or `None`.
    assert!(
        matches!(shortcut, Some(true) | None),
        "{model}: shortcut returned {shortcut:?}; it may only prove freedom or decline",
    );
}

#[test]
fn tokenring_pt_005_shortcut_matches_exhaustive() {
    let model = "TokenRing-PT-005";
    let Some(net) = try_load(model) else {
        return;
    };
    // Deadlock-free family: the shortcut should fire and the exhaustive engine
    // must agree.
    let shortcut = lp_siphon_deadlock_free(&net, None);
    let oracle = exhaustive_deadlock(&net, 50_000_000);
    eprintln!("{model}: shortcut={shortcut:?} exhaustive={oracle:?}");
    assert_eq!(
        shortcut,
        Some(true),
        "{model} is deadlock-free; the siphon-LP shortcut should prove it",
    );
    if let Some(oracle) = oracle {
        assert!(
            !oracle,
            "{model}: exhaustive engine must agree it is deadlock-free"
        );
    }
}

#[test]
fn anderson_pt_04_shortcut_matches_exhaustive() {
    assert_shortcut_consistent_with_exhaustive("Anderson-PT-04", 50_000_000);
}

#[test]
fn shared_memory_pt_000010_shortcut_matches_exhaustive() {
    assert_shortcut_consistent_with_exhaustive("SharedMemory-PT-000010", 50_000_000);
}

#[test]
fn csrepetitions_pt_02_genuine_deadlock_is_declined() {
    let model = "CSRepetitions-PT-02";
    let Some(net) = try_load(model) else {
        return;
    };
    // CSRepetitions-PT-02 has a genuine reachable deadlock. The diagnosis probe
    // (with the INCOMPLETE single-seed enumerator) wrongly reported all siphons
    // non-emptiable; the COMPLETE enumerator must find the emptiable siphon and
    // DECLINE so the exact engine returns the witnessed TRUE.
    let shortcut = lp_siphon_deadlock_free(&net, None);
    let oracle = exhaustive_deadlock(&net, 50_000_000);
    eprintln!("{model}: shortcut={shortcut:?} exhaustive={oracle:?}");
    assert_eq!(
        shortcut, None,
        "{model} genuinely deadlocks; the shortcut MUST decline, never certify freedom",
    );
    if let Some(oracle) = oracle {
        assert!(
            oracle,
            "{model}: exhaustive engine should confirm the deadlock"
        );
    }
}

#[test]
fn philosophers_pt_000005_genuine_deadlock_is_declined() {
    let model = "Philosophers-PT-000005";
    let Some(net) = try_load(model) else {
        return;
    };
    let shortcut = lp_siphon_deadlock_free(&net, None);
    let oracle = exhaustive_deadlock(&net, 50_000_000);
    eprintln!("{model}: shortcut={shortcut:?} exhaustive={oracle:?}");
    assert_eq!(
        shortcut, None,
        "{model} genuinely deadlocks; the shortcut MUST decline",
    );
    if let Some(oracle) = oracle {
        assert!(
            oracle,
            "{model}: exhaustive engine should confirm the deadlock"
        );
    }
}

#[test]
fn flexible_barrier_pt_04b_consistency() {
    // FlexibleBarrier-PT-04b has emptiable siphons (diagnosis: 11/21), so the
    // shortcut declines; only assert no unsound `Some(true)`.
    assert_shortcut_consistent_with_exhaustive("FlexibleBarrier-PT-04b", 50_000_000);
}

#[test]
fn hexagonal_grid_pt_110_shortcut_matches_exhaustive() {
    // HexagonalGrid-PT-110 is another family on which the siphon-LP shortcut
    // fires (`Some(true)`). Cross-check it against the exhaustive engine: the
    // shortcut must not certify freedom on a net that can reach a deadlock.
    assert_shortcut_consistent_with_exhaustive("HexagonalGrid-PT-110", 80_000_000);
}
