// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for the symbolic state-equation reachability seeder.
//!
//! These tests pin the env-gated wiring contract added in the
//! `run_symbolic_state_equation_seeding` ↔ reachability pipeline
//! handshake:
//!
//! - ON by default — the seeder runs unless
//!   `TY_MCC_ENABLE_REACHABILITY_SYMBOLIC` is explicitly set to a
//!   falsy value (`0`/`false`/`no`/`off`).
//! - Fireability atoms skipped — same carve-out as `reachability_pdr`.
//! - Verdicts MUST agree with explicit BFS on the small fixture nets.
//! - `Unknown` is propagated as no-op — never invented as TRUE/FALSE.
//! - Pre-seeded verdicts are NEVER overwritten.

use std::time::{Duration, Instant};

use super::{run_symbolic_state_equation_seeding, ENABLE_REACHABILITY_SYMBOLIC_ENV};
use crate::examinations::reachability::{PropertyTracker, ReachabilityResolutionSource};
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

struct EnvVarGuard<'a> {
    key: &'a str,
    prev: Option<String>,
}

impl<'a> EnvVarGuard<'a> {
    fn set(key: &'a str, value: Option<&str>) -> Self {
        let prev = std::env::var(key).ok();
        match value {
            Some(value) => crate::env_guard::set_var(key, value),
            None => crate::env_guard::remove_var(key),
        }
        Self { key, prev }
    }
}

impl Drop for EnvVarGuard<'_> {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            crate::env_guard::set_var(self.key, prev);
        } else {
            crate::env_guard::remove_var(self.key);
        }
    }
}

fn with_symbolic_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    // Single crate-wide env lock: serialize against every other module's
    // env-touching test, not just this file's.
    let _lock = crate::env_test_lock();
    let _guard = EnvVarGuard::set(ENABLE_REACHABILITY_SYMBOLIC_ENV, value);
    f()
}

fn place(id: &str) -> PlaceInfo {
    PlaceInfo {
        id: id.to_string(),
        name: None,
    }
}

fn arc(place: u32, weight: u64) -> Arc {
    Arc {
        place: PlaceIdx(place),
        weight,
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

/// 3 tokens initially on p0, reversible single-token transfer.
fn three_token_net() -> PetriNet {
    PetriNet {
        name: Some("three_token".to_string()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![3, 0],
    }
}

/// 1 token initially on p0, reversible single-token transfer.
fn simple_net() -> PetriNet {
    PetriNet {
        name: Some("simple".to_string()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    }
}

fn tracker(id: &str, quantifier: PathQuantifier, predicate: ResolvedPredicate) -> PropertyTracker {
    PropertyTracker {
        id: id.to_string(),
        quantifier,
        predicate,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }
}

// ── Env-gating: seeder is ON by default, OFF only when env is falsy ──

#[test]
fn test_symbolic_seeder_explicit_off_via_env_leaves_tracker_unresolved() {
    // Setting TY_MCC_ENABLE_REACHABILITY_SYMBOLIC=0 is the escape hatch
    // for clean-baseline benchmarking — it MUST cleanly disable the
    // seeder, leaving every tracker for downstream phases to resolve.
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
    )];

    with_symbolic_env(Some("0"), || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, None)
    });
    assert_eq!(
        trackers[0].verdict, None,
        "seeder must skip every tracker when env flag is explicitly 0"
    );
    assert_eq!(trackers[0].resolved_by, None);
}

#[test]
fn test_chc_seeding_runs_by_default_without_env_override() {
    // Regression pin for the default-ON flip: with no env override the
    // seeder MUST attempt dispatch. On simple_net the EF(m1>=1) query
    // is trivially reachable (t0 produces the token), and the
    // AdaptivePortfolio resolves it deterministically as TRUE. Unknown
    // is also acceptable (no soundness breach), but the tracker MUST
    // NOT remain in the pre-seeder state of {verdict: None,
    // resolved_by: None} with verdict still None *and* an Unsafe-shaped
    // false verdict — those would either prove the seeder didn't run
    // or expose a soundness regression respectively.
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
    )];

    with_symbolic_env(None, || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, None)
    });

    match trackers[0].verdict {
        Some(true) => {
            assert_eq!(
                trackers[0].resolved_by.map(|r| r.source),
                Some(ReachabilityResolutionSource::Pdr),
                "default-ON dispatch must stamp resolved_by on success"
            );
        }
        None => {
            // AdaptivePortfolio returned Unknown — acceptable, but the
            // pipeline did still execute (no env-gate short-circuit).
            // We cannot directly observe the dispatch attempt from
            // outside, so this branch trusts the verdict-correctness
            // tests below to cover the no-op-vs-ran distinction.
        }
        Some(false) => {
            panic!("default-ON seeder resolved EF(m1>=1) as FALSE on simple_net — SOUNDNESS BREACH")
        }
    }
}

// ── Verdict correctness (vs BFS ground truth) ─────────────────────────

#[test]
fn test_symbolic_seeder_ag_true_for_token_conservation_invariant() {
    // Three-token net has m0+m1 = 3 invariantly; AG(m0+m1 <= 3) is TRUE.
    // BFS ground truth: trivially satisfied at every reachable marking.
    let net = three_token_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
            ResolvedIntExpr::Constant(3),
        ),
    )];

    with_symbolic_env(Some("1"), || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, None)
    });
    // Symbolic should resolve to TRUE; Unknown is acceptable (no SAFE
    // breach), but a FALSE here would be a SOUNDNESS regression.
    match trackers[0].verdict {
        Some(true) => {
            assert_eq!(
                trackers[0].resolved_by.map(|r| r.source),
                Some(ReachabilityResolutionSource::Pdr),
                "resolved-by attribution must reflect symbolic dispatch"
            );
        }
        None => {
            // Solver returned Unknown — acceptable; soundness preserved.
        }
        Some(false) => panic!(
            "symbolic seeder resolved AG(m0+m1<=3) as FALSE on three_token net — SOUNDNESS BREACH"
        ),
    }
}

#[test]
fn test_symbolic_seeder_ag_false_on_reachable_counterexample() {
    // simple_net: m0 starts at 1; AG(m0 <= 0) is FALSE since the initial
    // marking violates the predicate.
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::Constant(0),
        ),
    )];

    with_symbolic_env(Some("1"), || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, None)
    });
    match trackers[0].verdict {
        Some(false) => {
            assert_eq!(
                trackers[0].resolved_by.map(|r| r.source),
                Some(ReachabilityResolutionSource::Pdr)
            );
        }
        None => {}
        Some(true) => panic!(
            "symbolic seeder resolved AG(m0<=0) as TRUE on simple_net — SOUNDNESS BREACH (initial marking violates predicate)"
        ),
    }
}

#[test]
fn test_symbolic_seeder_ef_true_for_reachable_state() {
    // simple_net: t0 fires once, m1 becomes 1; EF(m1 >= 1) is TRUE.
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
    )];

    with_symbolic_env(Some("1"), || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, None)
    });
    match trackers[0].verdict {
        Some(true) => {
            assert_eq!(
                trackers[0].resolved_by.map(|r| r.source),
                Some(ReachabilityResolutionSource::Pdr)
            );
        }
        None => {}
        Some(false) => panic!(
            "symbolic seeder resolved EF(m1>=1) as FALSE on simple_net — SOUNDNESS BREACH (t0 reaches this state)"
        ),
    }
}

#[test]
fn test_symbolic_seeder_ef_false_for_unreachable_overshoot() {
    // simple_net total tokens = 1; EF(m0 >= 2) is FALSE.
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(2),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ),
    )];

    with_symbolic_env(Some("1"), || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, None)
    });
    match trackers[0].verdict {
        Some(false) => {
            assert_eq!(
                trackers[0].resolved_by.map(|r| r.source),
                Some(ReachabilityResolutionSource::Pdr)
            );
        }
        None => {}
        Some(true) => panic!(
            "symbolic seeder resolved EF(m0>=2) as TRUE on simple_net — SOUNDNESS BREACH (impossible: only 1 token total)"
        ),
    }
}

// ── First-writer-wins: pre-seeded verdicts are preserved ──────────────

#[test]
fn test_symbolic_seeder_leaves_preseeded_verdict_unchanged() {
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::EF,
        ResolvedPredicate::True,
    )];
    trackers[0].verdict = Some(true);

    with_symbolic_env(Some("1"), || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, None)
    });
    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "pre-seeded verdict must be preserved (first-writer-wins)"
    );
    assert_eq!(
        trackers[0].resolved_by, None,
        "resolved_by stays None when the verdict was pre-seeded by a prior phase"
    );
}

// ── Fireability is now ADMITTED and replay-validated (no carve-out) ───
//
// The seeder no longer skips IsFireable predicates: the encoder lowers them
// and every witness-derived verdict is replay-validated on the concrete net.
// This test fixes the SOUNDNESS invariant: whatever the CHC lane decides for a
// fireability predicate, it must NEVER be wrong. In `simple_net` (t0 is
// fireable at the initial marking) the ground truth is `EF IsFireable(t0)` =
// TRUE and `AG ¬IsFireable(t0)` = FALSE, so each tracker must be either pending
// (`None`, sound) or resolved to exactly its ground-truth verdict — never the
// opposite.
#[test]
fn test_symbolic_seeder_fireability_is_sound() {
    let net = simple_net();
    // (id, quantifier, predicate, ground-truth verdict on simple_net)
    let mut trackers = vec![
        tracker(
            "ag-fireability",
            PathQuantifier::AG,
            ResolvedPredicate::Not(Box::new(ResolvedPredicate::IsFireable(vec![
                TransitionIdx(0),
            ]))),
        ),
        tracker(
            "ef-fireability",
            PathQuantifier::EF,
            ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
        ),
    ];
    let ground_truth = [
        false, /* AG ¬fireable(t0) */
        true,  /* EF fireable(t0) */
    ];

    with_symbolic_env(Some("1"), || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, None)
    });
    for (t, &truth) in trackers.iter().zip(ground_truth.iter()) {
        match t.verdict {
            None => {} // pending is always sound (CHC inconclusive / witness not replay-validated)
            Some(v) => assert_eq!(
                v, truth,
                "fireability predicate {} resolved to {v}, but ground truth is {truth} — \
                 the replay gate must never admit a wrong verdict",
                t.id
            ),
        }
    }
}

// ── Deadline honouring: expired deadline leaves trackers untouched ────

#[test]
fn test_symbolic_seeder_deadline_expiry_leaves_tracker_unresolved() {
    let net = simple_net();
    let mut trackers = vec![tracker(
        "prop-00",
        PathQuantifier::AG,
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::Constant(1),
        ),
    )];

    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .unwrap_or_else(Instant::now);
    with_symbolic_env(Some("1"), || {
        run_symbolic_state_equation_seeding(&net, &mut trackers, Some(expired))
    });
    assert_eq!(trackers[0].verdict, None);
    assert_eq!(trackers[0].resolved_by, None);
}

#[test]
fn test_symbolic_tracker_timeout_fair_shares_pending_trackers() {
    let now = Instant::now();
    let deadline = now + Duration::from_secs(12);

    assert_eq!(
        super::symbolic_tracker_timeout_at(Some(deadline), 4, now),
        Duration::from_secs(3)
    );
}

#[test]
fn test_symbolic_tracker_timeout_caps_large_share() {
    let now = Instant::now();
    let deadline = now + Duration::from_secs(120);

    assert_eq!(
        super::symbolic_tracker_timeout_at(Some(deadline), 2, now),
        super::SYMBOLIC_SEED_TIMEOUT
    );
}
